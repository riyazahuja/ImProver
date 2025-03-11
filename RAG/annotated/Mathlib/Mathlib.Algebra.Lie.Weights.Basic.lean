variable (M) in
/-- If `M` is a representation of a Lie algebra `L` and `χ : L → R` is a family of scalars,
then `weightSpace M χ` is the intersection of the `χ x`-eigenspaces
of the action of `x` on `M` as `x` ranges over `L`. -/
def weightSpace (χ : L → R) : LieSubmodule R L M where
  __ := ⨅ x : L, (toEnd R L M x).eigenspace (χ x)
                         /-
                           K : Type u_1
                           R : Type u_2
                           L : Type u_3
                           M : Type u_4
                           inst✝⁶ : CommRing R
                           inst✝⁵ : LieRing L
                           inst✝⁴ : LieAlgebra R L
                           inst✝³ : AddCommGroup M
                           inst✝² : Module R M
                           inst✝¹ : LieRingModule L M
                           inst✝ : LieModule R L M
                           χ : L → R
                           x : L
                           m : M
                           hm : Membership.mem __spread✝⁻⁰.carrier m
                           ⊢ Membership.mem __spread✝⁻⁰.carrier (Bracket.bracket x m)
                         -/
  lie_mem {x m} hm := by simp_all [smul_comm (χ x)]
                         /-
                           🎉 no goals
                         -/


lemma mem_weightSpace (χ : L → R) (m : M) : m ∈ weightSpace M χ ↔ ∀ x, ⁅x, m⁆ = χ x • m := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    χ : L → R
    m : M
    ⊢ Iff (Membership.mem (LieModule.weightSpace M χ) m) (∀ (x : L), Eq (Bracket.b …
  -/
  simp [weightSpace]
  /-
    🎉 no goals
  -/


/-- Until we define `LieModule.genWeightSpaceOf`, it is useful to have some notation as follows: -/
local notation3 "𝕎("M", " χ", " x")" => (toEnd R L M x).maxGenEigenspace χ


/-- See also `bourbaki1975b` Chapter VII §1.1, Proposition 2 (ii). -/
protected theorem weight_vector_multiplication (M₁ M₂ M₃ : Type*)
    [AddCommGroup M₁] [Module R M₁] [LieRingModule L M₁] [LieModule R L M₁] [AddCommGroup M₂]
    [Module R M₂] [LieRingModule L M₂] [LieModule R L M₂] [AddCommGroup M₃] [Module R M₃]
    [LieRingModule L M₃] [LieModule R L M₃] (g : M₁ ⊗[R] M₂ →ₗ⁅R,L⁆ M₃) (χ₁ χ₂ : R) (x : L) :
    LinearMap.range ((g : M₁ ⊗[R] M₂ →ₗ[R] M₃).comp (mapIncl 𝕎(M₁, χ₁, x) 𝕎(M₂, χ₂, x))) ≤
      𝕎(M₃, χ₁ + χ₂, x) := by
  -- Unpack the statement of the goal.
  /-
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    ⊢ LE.le (LinearMap.range ((↑g).comp (TensorProduct.mapIncl (((LieModule.toEnd  …
  -/
  intro m₃
  simp only [TensorProduct.mapIncl, LinearMap.mem_range, LinearMap.coe_comp,
    LieModuleHom.coe_toLinearMap, Function.comp_apply, Pi.add_apply, exists_imp,
    Module.End.mem_maxGenEigenspace]
  /-
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    m₃ : M₃
    ⊢ ∀ (x_1 : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toE …
  -/
  rintro t rfl
  -- Set up some notation.
  /-
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    ⊢ Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMu …
  -/
  let F : Module.End R M₃ := toEnd R L M₃ x - (χ₁ + χ₂) • ↑1
  -- The goal is linear in `t` so use induction to reduce to the case that `t` is a pure tensor.
  /-
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    ⊢ Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMu …
  -/
  refine t.induction_on ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_2
      L : Type u_3
      inst✝¹⁴ : CommRing R
      inst✝¹³ : LieRing L
      inst✝¹² : LieAlgebra R L
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      inst✝¹¹ : AddCommGroup M₁
      inst✝¹⁰ : Module R M₁
      inst✝⁹ : LieRingModule L M₁
      inst✝⁸ : LieModule R L M₁
      inst✝⁷ : AddCommGroup M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : LieRingModule L M₂
      inst✝⁴ : LieModule R L M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module R M₃
      inst✝¹ : LieRingModule L M₃
      inst✝ : LieModule R L M₃
      g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
      χ₁ χ₂ : R
      x : L
      t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
      F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
      ⊢ Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMu …
    -/
  · use 0; simp only [LinearMap.map_zero, LieModuleHom.map_zero]
           /-
             🎉 no goals
           -/
  /-
    case refine_2
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    ⊢ ∀ (x_1 : Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L M₁) x).max …
  -/
  swap
    /-
      case refine_3
      R : Type u_2
      L : Type u_3
      inst✝¹⁴ : CommRing R
      inst✝¹³ : LieRing L
      inst✝¹² : LieAlgebra R L
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      inst✝¹¹ : AddCommGroup M₁
      inst✝¹⁰ : Module R M₁
      inst✝⁹ : LieRingModule L M₁
      inst✝⁸ : LieModule R L M₁
      inst✝⁷ : AddCommGroup M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : LieRingModule L M₂
      inst✝⁴ : LieModule R L M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module R M₃
      inst✝¹ : LieRingModule L M₃
      inst✝ : LieModule R L M₃
      g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
      χ₁ χ₂ : R
      x : L
      t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
      F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
      ⊢ ∀ (x_1 y : TensorProduct R (Subtype fun x_2 => Membership.mem (((LieModule.t …
    -/
  · rintro t₁ t₂ ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩; use max k₁ k₂
    simp only [LieModuleHom.map_add, LinearMap.map_add,
      LinearMap.pow_map_zero_of_le (le_max_left k₁ k₂) hk₁,
      LinearMap.pow_map_zero_of_le (le_max_right k₁ k₂) hk₂, add_zero]
  -- Now the main argument: pure tensors.
  /-
    case refine_2
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    ⊢ ∀ (x_1 : Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L M₁) x).max …
  -/
  rintro ⟨m₁, hm₁⟩ ⟨m₂, hm₂⟩
  /-
    case refine_2.mk.mk
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    hm₁ : Membership.mem (((LieModule.toEnd R L M₁) x).maxGenEigenspace χ₁) m₁
    m₂ : M₂
    hm₂ : Membership.mem (((LieModule.toEnd R L M₂) x).maxGenEigenspace χ₂) m₂
    ⊢ Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMu …
  -/
  change ∃ k, (F ^ k) ((g : M₁ ⊗[R] M₂ →ₗ[R] M₃) (m₁ ⊗ₜ m₂)) = (0 : M₃)
  -- Eliminate `g` from the picture.
  /-
    case refine_2.mk.mk
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    hm₁ : Membership.mem (((LieModule.toEnd R L M₁) x).maxGenEigenspace χ₁) m₁
    m₂ : M₂
    hm₂ : Membership.mem (((LieModule.toEnd R L M₂) x).maxGenEigenspace χ₂) m₂
    ⊢ Exists fun k => Eq ((HPow.hPow F k) (↑g (TensorProduct.tmul R m₁ m₂))) 0
  -/
  let f₁ : Module.End R (M₁ ⊗[R] M₂) := (toEnd R L M₁ x - χ₁ • ↑1).rTensor M₂
  /-
    case refine_2.mk.mk
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    hm₁ : Membership.mem (((LieModule.toEnd R L M₁) x).maxGenEigenspace χ₁) m₁
    m₂ : M₂
    hm₂ : Membership.mem (((LieModule.toEnd R L M₂) x).maxGenEigenspace χ₂) m₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    ⊢ Exists fun k => Eq ((HPow.hPow F k) (↑g (TensorProduct.tmul R m₁ m₂))) 0
  -/
  let f₂ : Module.End R (M₁ ⊗[R] M₂) := (toEnd R L M₂ x - χ₂ • ↑1).lTensor M₁
  have h_comm_square : F ∘ₗ ↑g = (g : M₁ ⊗[R] M₂ →ₗ[R] M₃).comp (f₁ + f₂) := by
    ext m₁ m₂
    simp only [f₁, f₂, F, ← g.map_lie x (m₁ ⊗ₜ m₂), add_smul, sub_tmul, tmul_sub, smul_tmul,
      lie_tmul_right, tmul_smul, toEnd_apply_apply, LieModuleHom.map_smul,
      LinearMap.one_apply, LieModuleHom.coe_toLinearMap, LinearMap.smul_apply, Function.comp_apply,
      LinearMap.coe_comp, LinearMap.rTensor_tmul, LieModuleHom.map_add, LinearMap.add_apply,
      LieModuleHom.map_sub, LinearMap.sub_apply, LinearMap.lTensor_tmul,
      AlgebraTensorModule.curry_apply, TensorProduct.curry_apply, LinearMap.toFun_eq_coe,
      LinearMap.coe_restrictScalars]
    abel
  /-
    case refine_2.mk.mk
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    hm₁ : Membership.mem (((LieModule.toEnd R L M₁) x).maxGenEigenspace χ₁) m₁
    m₂ : M₂
    hm₂ : Membership.mem (((LieModule.toEnd R L M₂) x).maxGenEigenspace χ₂) m₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    ⊢ Exists fun k => Eq ((HPow.hPow F k) (↑g (TensorProduct.tmul R m₁ m₂))) 0
  -/
  rsuffices ⟨k, hk⟩ : ∃ k : ℕ, ((f₁ + f₂) ^ k) (m₁ ⊗ₜ m₂) = 0
    /-
      case refine_2.mk.mk.intro
      R : Type u_2
      L : Type u_3
      inst✝¹⁴ : CommRing R
      inst✝¹³ : LieRing L
      inst✝¹² : LieAlgebra R L
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      inst✝¹¹ : AddCommGroup M₁
      inst✝¹⁰ : Module R M₁
      inst✝⁹ : LieRingModule L M₁
      inst✝⁸ : LieModule R L M₁
      inst✝⁷ : AddCommGroup M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : LieRingModule L M₂
      inst✝⁴ : LieModule R L M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module R M₃
      inst✝¹ : LieRingModule L M₃
      inst✝ : LieModule R L M₃
      g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
      χ₁ χ₂ : R
      x : L
      t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
      F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
      m₁ : M₁
      hm₁ : Membership.mem (((LieModule.toEnd R L M₁) x).maxGenEigenspace χ₁) m₁
      m₂ : M₂
      hm₂ : Membership.mem (((LieModule.toEnd R L M₂) x).maxGenEigenspace χ₂) m₂
      f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
      f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
      h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
      k : Nat
      hk : Eq ((HPow.hPow (HAdd.hAdd f₁ f₂) k) (TensorProduct.tmul R m₁ m₂)) 0
      ⊢ Exists fun k => Eq ((HPow.hPow F k) (↑g (TensorProduct.tmul R m₁ m₂))) 0
    -/
  · use k
    /-
      case h
      R : Type u_2
      L : Type u_3
      inst✝¹⁴ : CommRing R
      inst✝¹³ : LieRing L
      inst✝¹² : LieAlgebra R L
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      inst✝¹¹ : AddCommGroup M₁
      inst✝¹⁰ : Module R M₁
      inst✝⁹ : LieRingModule L M₁
      inst✝⁸ : LieModule R L M₁
      inst✝⁷ : AddCommGroup M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : LieRingModule L M₂
      inst✝⁴ : LieModule R L M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module R M₃
      inst✝¹ : LieRingModule L M₃
      inst✝ : LieModule R L M₃
      g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
      χ₁ χ₂ : R
      x : L
      t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
      F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
      m₁ : M₁
      hm₁ : Membership.mem (((LieModule.toEnd R L M₁) x).maxGenEigenspace χ₁) m₁
      m₂ : M₂
      hm₂ : Membership.mem (((LieModule.toEnd R L M₂) x).maxGenEigenspace χ₂) m₂
      f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
      f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
      h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
      k : Nat
      hk : Eq ((HPow.hPow (HAdd.hAdd f₁ f₂) k) (TensorProduct.tmul R m₁ m₂)) 0
      ⊢ Eq ((HPow.hPow F k) (↑g (TensorProduct.tmul R m₁ m₂))) 0
    -/
    change (F ^ k) (g.toLinearMap (m₁ ⊗ₜ[R] m₂)) = 0
    rw [← LinearMap.comp_apply, LinearMap.commute_pow_left_of_commute h_comm_square,
      LinearMap.comp_apply, hk, LinearMap.map_zero]
  -- Unpack the information we have about `m₁`, `m₂`.
  /-
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    hm₁ : Membership.mem (((LieModule.toEnd R L M₁) x).maxGenEigenspace χ₁) m₁
    m₂ : M₂
    hm₂ : Membership.mem (((LieModule.toEnd R L M₂) x).maxGenEigenspace χ₂) m₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    ⊢ Exists fun k => Eq ((HPow.hPow (HAdd.hAdd f₁ f₂) k) (TensorProduct.tmul R m₁ …
  -/
  simp only [Module.End.mem_maxGenEigenspace] at hm₁ hm₂
  /-
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    m₂ : M₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    hm₁ : Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) ( …
    hm₂ : Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) ( …
    ⊢ Exists fun k => Eq ((HPow.hPow (HAdd.hAdd f₁ f₂) k) (TensorProduct.tmul R m₁ …
  -/
  obtain ⟨k₁, hk₁⟩ := hm₁
  /-
    case intro
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    m₂ : M₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    hm₂ : Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) ( …
    k₁ : Nat
    hk₁ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) (HSMul.hSMul χ₁ 1 …
    ⊢ Exists fun k => Eq ((HPow.hPow (HAdd.hAdd f₁ f₂) k) (TensorProduct.tmul R m₁ …
  -/
  obtain ⟨k₂, hk₂⟩ := hm₂
  have hf₁ : (f₁ ^ k₁) (m₁ ⊗ₜ m₂) = 0 := by
    simp only [f₁, hk₁, zero_tmul, LinearMap.rTensor_tmul, LinearMap.rTensor_pow]
  have hf₂ : (f₂ ^ k₂) (m₁ ⊗ₜ m₂) = 0 := by
    simp only [f₂, hk₂, tmul_zero, LinearMap.lTensor_tmul, LinearMap.lTensor_pow]
  -- It's now just an application of the binomial theorem.
  /-
    case intro.intro
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    m₂ : M₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    k₁ : Nat
    hk₁ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) (HSMul.hSMul χ₁ 1 …
    k₂ : Nat
    hk₂ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul χ₂ 1 …
    hf₁ : Eq ((HPow.hPow f₁ k₁) (TensorProduct.tmul R m₁ m₂)) 0
    hf₂ : Eq ((HPow.hPow f₂ k₂) (TensorProduct.tmul R m₁ m₂)) 0
    ⊢ Exists fun k => Eq ((HPow.hPow (HAdd.hAdd f₁ f₂) k) (TensorProduct.tmul R m₁ …
  -/
  use k₁ + k₂ - 1
  have hf_comm : Commute f₁ f₂ := by
    ext m₁ m₂
    simp only [f₁, f₂, LinearMap.mul_apply, LinearMap.rTensor_tmul, LinearMap.lTensor_tmul,
      AlgebraTensorModule.curry_apply, LinearMap.toFun_eq_coe, LinearMap.lTensor_tmul,
      TensorProduct.curry_apply, LinearMap.coe_restrictScalars]
  /-
    case h
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    m₂ : M₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    k₁ : Nat
    hk₁ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) (HSMul.hSMul χ₁ 1 …
    k₂ : Nat
    hk₂ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul χ₂ 1 …
    hf₁ : Eq ((HPow.hPow f₁ k₁) (TensorProduct.tmul R m₁ m₂)) 0
    hf₂ : Eq ((HPow.hPow f₂ k₂) (TensorProduct.tmul R m₁ m₂)) 0
    hf_comm : Commute f₁ f₂
    ⊢ Eq ((HPow.hPow (HAdd.hAdd f₁ f₂) (HSub.hSub (HAdd.hAdd k₁ k₂) 1)) (TensorPro …
  -/
  rw [hf_comm.add_pow']
  simp only [TensorProduct.mapIncl, Submodule.subtype_apply, Finset.sum_apply, Submodule.coe_mk,
    LinearMap.coeFn_sum, TensorProduct.map_tmul, LinearMap.smul_apply]
  -- The required sum is zero because each individual term is zero.
  /-
    case h
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    m₂ : M₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    k₁ : Nat
    hk₁ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) (HSMul.hSMul χ₁ 1 …
    k₂ : Nat
    hk₂ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul χ₂ 1 …
    hf₁ : Eq ((HPow.hPow f₁ k₁) (TensorProduct.tmul R m₁ m₂)) 0
    hf₂ : Eq ((HPow.hPow f₂ k₂) (TensorProduct.tmul R m₁ m₂)) 0
    hf_comm : Commute f₁ f₂
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HSub.hSub (HAdd.hAdd k₁ k₂) 1)).su …
  -/
  apply Finset.sum_eq_zero
  /-
    case h.h
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    m₂ : M₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    k₁ : Nat
    hk₁ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) (HSMul.hSMul χ₁ 1 …
    k₂ : Nat
    hk₂ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul χ₂ 1 …
    hf₁ : Eq ((HPow.hPow f₁ k₁) (TensorProduct.tmul R m₁ m₂)) 0
    hf₂ : Eq ((HPow.hPow f₂ k₂) (TensorProduct.tmul R m₁ m₂)) 0
    hf_comm : Commute f₁ f₂
    ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
  -/
  rintro ⟨i, j⟩ hij
  -- Eliminate the binomial coefficients from the picture.
  /-
    case h.h.mk
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    m₂ : M₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    k₁ : Nat
    hk₁ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) (HSMul.hSMul χ₁ 1 …
    k₂ : Nat
    hk₂ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul χ₂ 1 …
    hf₁ : Eq ((HPow.hPow f₁ k₁) (TensorProduct.tmul R m₁ m₂)) 0
    hf₂ : Eq ((HPow.hPow f₂ k₂) (TensorProduct.tmul R m₁ m₂)) 0
    hf_comm : Commute f₁ f₂
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HSub.hSub (HAdd.hAd …
    ⊢ Eq (HSMul.hSMul ((HSub.hSub (HAdd.hAdd k₁ k₂) 1).choose { fst := i, snd := j …
  -/
  suffices (f₁ ^ i * f₂ ^ j) (m₁ ⊗ₜ m₂) = 0 by rw [this]; apply smul_zero
  -- Finish off with appropriate case analysis.
  /-
    case h.h.mk
    R : Type u_2
    L : Type u_3
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : LieRingModule L M₁
    inst✝⁸ : LieModule R L M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : LieRingModule L M₂
    inst✝⁴ : LieModule R L M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : LieRingModule L M₃
    inst✝ : LieModule R L M₃
    g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
    χ₁ χ₂ : R
    x : L
    t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
    F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
    m₁ : M₁
    m₂ : M₂
    f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
    f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
    h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
    k₁ : Nat
    hk₁ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) (HSMul.hSMul χ₁ 1 …
    k₂ : Nat
    hk₂ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul χ₂ 1 …
    hf₁ : Eq ((HPow.hPow f₁ k₁) (TensorProduct.tmul R m₁ m₂)) 0
    hf₂ : Eq ((HPow.hPow f₂ k₂) (TensorProduct.tmul R m₁ m₂)) 0
    hf_comm : Commute f₁ f₂
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HSub.hSub (HAdd.hAd …
    ⊢ Eq ((HMul.hMul (HPow.hPow f₁ i) (HPow.hPow f₂ j)) (TensorProduct.tmul R m₁ m …
  -/
  cases' Nat.le_or_le_of_add_eq_add_pred (Finset.mem_antidiagonal.mp hij) with hi hj
  · rw [(hf_comm.pow_pow i j).eq, LinearMap.mul_apply, LinearMap.pow_map_zero_of_le hi hf₁,
      LinearMap.map_zero]
    /-
      case h.h.mk.inr
      R : Type u_2
      L : Type u_3
      inst✝¹⁴ : CommRing R
      inst✝¹³ : LieRing L
      inst✝¹² : LieAlgebra R L
      M₁ : Type u_5
      M₂ : Type u_6
      M₃ : Type u_7
      inst✝¹¹ : AddCommGroup M₁
      inst✝¹⁰ : Module R M₁
      inst✝⁹ : LieRingModule L M₁
      inst✝⁸ : LieModule R L M₁
      inst✝⁷ : AddCommGroup M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : LieRingModule L M₂
      inst✝⁴ : LieModule R L M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module R M₃
      inst✝¹ : LieRingModule L M₃
      inst✝ : LieModule R L M₃
      g : LieModuleHom R L (TensorProduct R M₁ M₂) M₃
      χ₁ χ₂ : R
      x : L
      t : TensorProduct R (Subtype fun x_1 => Membership.mem (((LieModule.toEnd R L  …
      F : Module.End R M₃ := HSub.hSub ((LieModule.toEnd R L M₃) x) (HSMul.hSMul (HA …
      m₁ : M₁
      m₂ : M₂
      f₁ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.rTensor M₂ (HSub.hSub ( …
      f₂ : Module.End R (TensorProduct R M₁ M₂) := LinearMap.lTensor M₁ (HSub.hSub ( …
      h_comm_square : Eq (LinearMap.comp F ↑g) ((↑g).comp (HAdd.hAdd f₁ f₂))
      k₁ : Nat
      hk₁ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₁) x) (HSMul.hSMul χ₁ 1 …
      k₂ : Nat
      hk₂ : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul χ₂ 1 …
      hf₁ : Eq ((HPow.hPow f₁ k₁) (TensorProduct.tmul R m₁ m₂)) 0
      hf₂ : Eq ((HPow.hPow f₂ k₂) (TensorProduct.tmul R m₁ m₂)) 0
      hf_comm : Commute f₁ f₂
      i j : Nat
      hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HSub.hSub (HAdd.hAd …
      hj : LE.le k₂ { fst := i, snd := j }.2
      ⊢ Eq ((HMul.hMul (HPow.hPow f₁ i) (HPow.hPow f₂ j)) (TensorProduct.tmul R m₁ m …
    -/
  · rw [LinearMap.mul_apply, LinearMap.pow_map_zero_of_le hj hf₂, LinearMap.map_zero]
    /-
      🎉 no goals
    -/


lemma lie_mem_maxGenEigenspace_toEnd
    {χ₁ χ₂ : R} {x y : L} {m : M} (hy : y ∈ 𝕎(L, χ₁, x)) (hm : m ∈ 𝕎(M, χ₂, x)) :
    ⁅y, m⁆ ∈ 𝕎(M, χ₁ + χ₂, x) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    χ₁ χ₂ : R
    x y : L
    m : M
    hy : Membership.mem (((LieModule.toEnd R L L) x).maxGenEigenspace χ₁) y
    hm : Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace χ₂) m
    ⊢ Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace (HAdd.hAdd χ₁ χ …
  -/
  apply LieModule.weight_vector_multiplication L M M (toModuleHom R L M) χ₁ χ₂
  simp only [LieModuleHom.coe_toLinearMap, Function.comp_apply, LinearMap.coe_comp,
    TensorProduct.mapIncl, LinearMap.mem_range]
  /-
    case a
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    χ₁ χ₂ : R
    x y : L
    m : M
    hy : Membership.mem (((LieModule.toEnd R L L) x).maxGenEigenspace χ₁) y
    hm : Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace χ₂) m
    ⊢ Exists fun y_1 => Eq ((LieModule.toModuleHom R L M) ((TensorProduct.map (((L …
  -/
  use ⟨y, hy⟩ ⊗ₜ ⟨m, hm⟩
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    χ₁ χ₂ : R
    x y : L
    m : M
    hy : Membership.mem (((LieModule.toEnd R L L) x).maxGenEigenspace χ₁) y
    hm : Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace χ₂) m
    ⊢ Eq ((LieModule.toModuleHom R L M) ((TensorProduct.map (((LieModule.toEnd R L …
  -/
  simp only [Submodule.subtype_apply, toModuleHom_apply, TensorProduct.map_tmul]
  /-
    🎉 no goals
  -/


/-- If `M` is a representation of a nilpotent Lie algebra `L`, `χ` is a scalar, and `x : L`, then
`genWeightSpaceOf M χ x` is the maximal generalized `χ`-eigenspace of the action of `x` on `M`.

It is a Lie submodule because `L` is nilpotent. -/
def genWeightSpaceOf [LieAlgebra.IsNilpotent R L] (χ : R) (x : L) : LieSubmodule R L M :=
  { 𝕎(M, χ, x) with
    lie_mem := by
      /-
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        χ : R
        x : L
        ⊢ ∀ {x : L} {m : M}, Membership.mem __src✝.carrier m → Membership.mem __src✝.c …
      -/
      intro y m hm
      simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
        Submodule.mem_toAddSubmonoid] at hm ⊢
      /-
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        χ : R
        x y : L
        m : M
        hm : Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace χ) m
        ⊢ Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace χ) (Bracket.bra …
      -/
      rw [← zero_add χ]
      /-
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        χ : R
        x y : L
        m : M
        hm : Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace χ) m
        ⊢ Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace (HAdd.hAdd 0 χ) …
      -/
      exact lie_mem_maxGenEigenspace_toEnd (by simp) hm }
      /-
        🎉 no goals
      -/


theorem mem_genWeightSpaceOf (χ : R) (x : L) (m : M) :
    m ∈ genWeightSpaceOf M χ x ↔ ∃ k : ℕ, ((toEnd R L M x - χ • ↑1) ^ k) m = 0 := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : R
    x : L
    m : M
    ⊢ Iff (Membership.mem (LieModule.genWeightSpaceOf M χ x) m) (Exists fun k => E …
  -/
  simp [genWeightSpaceOf]
  /-
    🎉 no goals
  -/


theorem coe_genWeightSpaceOf_zero (x : L) :
    ↑(genWeightSpaceOf M (0 : R) x) = ⨆ k, LinearMap.ker (toEnd R L M x ^ k) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    x : L
    ⊢ Eq (↑(LieModule.genWeightSpaceOf M 0 x)) (iSup fun k => LinearMap.ker (HPow. …
  -/
  simp [genWeightSpaceOf, ← Module.End.iSup_genEigenspace_eq]
  /-
    🎉 no goals
  -/


/-- If `M` is a representation of a nilpotent Lie algebra `L`
and `χ : L → R` is a family of scalars,
then `genWeightSpace M χ` is the intersection of the maximal generalized `χ x`-eigenspaces
of the action of `x` on `M` as `x` ranges over `L`.

It is a Lie submodule because `L` is nilpotent. -/
def genWeightSpace (χ : L → R) : LieSubmodule R L M :=
  ⨅ x, genWeightSpaceOf M (χ x) x


theorem mem_genWeightSpace (χ : L → R) (m : M) :
    m ∈ genWeightSpace M χ ↔ ∀ x, ∃ k : ℕ, ((toEnd R L M x - χ x • ↑1) ^ k) m = 0 := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    m : M
    ⊢ Iff (Membership.mem (LieModule.genWeightSpace M χ) m) (∀ (x : L), Exists fun …
  -/
  simp [genWeightSpace, mem_genWeightSpaceOf]
  /-
    🎉 no goals
  -/


lemma genWeightSpace_le_genWeightSpaceOf (x : L) (χ : L → R) :
    genWeightSpace M χ ≤ genWeightSpaceOf M (χ x) x :=
  iInf_le _ x


lemma weightSpace_le_genWeightSpace (χ : L → R) :
    weightSpace M χ ≤ genWeightSpace M χ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    ⊢ LE.le (LieModule.weightSpace M χ) (LieModule.genWeightSpace M χ)
  -/
  apply le_iInf
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    ⊢ ∀ (i : L), LE.le (LieModule.weightSpace M χ) (LieModule.genWeightSpaceOf M ( …
  -/
  intro x
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    x : L
    ⊢ LE.le (LieModule.weightSpace M χ) (LieModule.genWeightSpaceOf M (χ x) x)
  -/
  rw [← (LieSubmodule.toSubmodule_orderEmbedding R L M).le_iff_le]
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    x : L
    ⊢ LE.le ((LieSubmodule.toSubmodule_orderEmbedding R L M) (LieModule.weightSpac …
  -/
  apply (iInf_le _ x).trans
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    x : L
    ⊢ LE.le (((LieModule.toEnd R L M) x).eigenspace (χ x)) ((LieSubmodule.toSubmod …
  -/
  exact ((toEnd R L M x).genEigenspace (χ x)).monotone le_top
  /-
    🎉 no goals
  -/


variable (R L) in
/-- A weight of a Lie module is a map `L → R` such that the corresponding weight space is
non-trivial. -/
structure Weight where
  /-- The family of eigenvalues corresponding to a weight. -/
  toFun : L → R
  genWeightSpace_ne_bot' : genWeightSpace M toFun ≠ ⊥


instance instFunLike : FunLike (Weight R L M) L R where
  coe χ := χ.1
                               /-
                                 K : Type u_1
                                 R : Type u_2
                                 L : Type u_3
                                 M : Type u_4
                                 inst✝⁷ : CommRing R
                                 inst✝⁶ : LieRing L
                                 inst✝⁵ : LieAlgebra R L
                                 inst✝⁴ : AddCommGroup M
                                 inst✝³ : Module R M
                                 inst✝² : LieRingModule L M
                                 inst✝¹ : LieModule R L M
                                 inst✝ : LieAlgebra.IsNilpotent R L
                                 χ₁ χ₂ : LieModule.Weight R L M
                                 h : Eq ((fun χ => χ.toFun) χ₁) ((fun χ => χ.toFun) χ₂)
                                 ⊢ Eq χ₁ χ₂
                               -/
  coe_injective' χ₁ χ₂ h := by cases χ₁; cases χ₂; simp_all
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp] lemma coe_weight_mk (χ : L → R) (h) :
    (↑(⟨χ, h⟩ : Weight R L M) : L → R) = χ :=
  rfl


lemma genWeightSpace_ne_bot (χ : Weight R L M) : genWeightSpace M χ ≠ ⊥ := χ.genWeightSpace_ne_bot'


@[ext] lemma ext {χ₁ χ₂ : Weight R L M} (h : ∀ x, χ₁ x = χ₂ x) : χ₁ = χ₂ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : LieModule.Weight R L M
    h : ∀ (x : L), Eq (χ₁ x) (χ₂ x)
    ⊢ Eq χ₁ χ₂
  -/
  cases' χ₁ with f₁ _; cases' χ₂ with f₂ _; aesop
                                            /-
                                              🎉 no goals
                                            -/


                                                                          /-
                                                                            R : Type u_2
                                                                            L : Type u_3
                                                                            M : Type u_4
                                                                            inst✝⁷ : CommRing R
                                                                            inst✝⁶ : LieRing L
                                                                            inst✝⁵ : LieAlgebra R L
                                                                            inst✝⁴ : AddCommGroup M
                                                                            inst✝³ : Module R M
                                                                            inst✝² : LieRingModule L M
                                                                            inst✝¹ : LieModule R L M
                                                                            inst✝ : LieAlgebra.IsNilpotent R L
                                                                            χ₁ χ₂ : LieModule.Weight R L M
                                                                            ⊢ Iff (Eq ⇑χ₁ ⇑χ₂) (Eq χ₁ χ₂)
                                                                          -/
lemma ext_iff' {χ₁ χ₂ : Weight R L M} : (χ₁ : L → R) = χ₂ ↔ χ₁ = χ₂ := by aesop
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma exists_ne_zero (χ : Weight R L M) :
    ∃ x ∈ genWeightSpace M χ, x ≠ 0 := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : LieModule.Weight R L M
    ⊢ Exists fun x => And (Membership.mem (LieModule.genWeightSpace M ⇑χ) x) (Ne x …
  -/
  simpa [LieSubmodule.eq_bot_iff] using χ.genWeightSpace_ne_bot
  /-
    🎉 no goals
  -/


instance [Subsingleton M] : IsEmpty (Weight R L M) :=
  ⟨fun h ↦ h.2 (Subsingleton.elim _ _)⟩


instance [Nontrivial (genWeightSpace M (0 : L → R))] : Zero (Weight R L M) :=
  ⟨0, fun e ↦ not_nontrivial (⊥ : LieSubmodule R L M) (e ▸ ‹_›)⟩


@[simp]
lemma coe_zero [Nontrivial (genWeightSpace M (0 : L → R))] : ((0 : Weight R L M) : L → R) = 0 := rfl


lemma zero_apply [Nontrivial (genWeightSpace M (0 : L → R))] (x) : (0 : Weight R L M) x = 0 := rfl


/-- The proposition that a weight of a Lie module is zero.

We make this definition because we cannot define a `Zero (Weight R L M)` instance since the weight
space of the zero function can be trivial. -/
def IsZero (χ : Weight R L M) := (χ : L → R) = 0


@[simp] lemma IsZero.eq {χ : Weight R L M} (hχ : χ.IsZero) : (χ : L → R) = 0 := hχ


@[simp] lemma coe_eq_zero_iff (χ : Weight R L M) : (χ : L → R) = 0 ↔ χ.IsZero := Iff.rfl


lemma isZero_iff_eq_zero [Nontrivial (genWeightSpace M (0 : L → R))] {χ : Weight R L M} :
    χ.IsZero ↔ χ = 0 := Weight.ext_iff' (χ₂ := 0)


lemma isZero_zero [Nontrivial (genWeightSpace M (0 : L → R))] : IsZero (0 : Weight R L M) := rfl


/-- The proposition that a weight of a Lie module is non-zero. -/
abbrev IsNonZero (χ : Weight R L M) := ¬ IsZero (χ : Weight R L M)


lemma isNonZero_iff_ne_zero [Nontrivial (genWeightSpace M (0 : L → R))] {χ : Weight R L M} :
    χ.IsNonZero ↔ χ ≠ 0 := isZero_iff_eq_zero.not


noncomputable instance : DecidablePred (IsNonZero (R := R) (L := L) (M := M)) := Classical.decPred _


variable (R L M) in
/-- The set of weights is equivalent to a subtype. -/
def equivSetOf : Weight R L M ≃ {χ : L → R | genWeightSpace M χ ≠ ⊥} where
  toFun w := ⟨w.1, w.2⟩
  invFun w := ⟨w.1, w.2⟩
                   /-
                     K : Type u_1
                     R : Type u_2
                     L : Type u_3
                     M : Type u_4
                     inst✝⁷ : CommRing R
                     inst✝⁶ : LieRing L
                     inst✝⁵ : LieAlgebra R L
                     inst✝⁴ : AddCommGroup M
                     inst✝³ : Module R M
                     inst✝² : LieRingModule L M
                     inst✝¹ : LieModule R L M
                     inst✝ : LieAlgebra.IsNilpotent R L
                     w : LieModule.Weight R L M
                     ⊢ Eq ((fun w => { toFun := ↑w, genWeightSpace_ne_bot' := ⋯ }) ((fun w => ⟨w.to …
                   -/
  left_inv w := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      K : Type u_1
                      R : Type u_2
                      L : Type u_3
                      M : Type u_4
                      inst✝⁷ : CommRing R
                      inst✝⁶ : LieRing L
                      inst✝⁵ : LieAlgebra R L
                      inst✝⁴ : AddCommGroup M
                      inst✝³ : Module R M
                      inst✝² : LieRingModule L M
                      inst✝¹ : LieModule R L M
                      inst✝ : LieAlgebra.IsNilpotent R L
                      w : ↑(setOf fun χ => Ne (LieModule.genWeightSpace M χ) Bot.bot)
                      ⊢ Eq ((fun w => ⟨w.toFun, ⋯⟩) ((fun w => { toFun := ↑w, genWeightSpace_ne_bot' …
                    -/
  right_inv w := by simp
                    /-
                      🎉 no goals
                    -/


lemma genWeightSpaceOf_ne_bot (χ : Weight R L M) (x : L) :
    genWeightSpaceOf M (χ x) x ≠ ⊥ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : LieModule.Weight R L M
    x : L
    ⊢ Ne (LieModule.genWeightSpaceOf M (χ x) x) Bot.bot
  -/
  have : ⨅ x, genWeightSpaceOf M (χ x) x ≠ ⊥ := χ.genWeightSpace_ne_bot
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : LieModule.Weight R L M
    x : L
    this : Ne (iInf fun x => LieModule.genWeightSpaceOf M (χ x) x) Bot.bot
    ⊢ Ne (LieModule.genWeightSpaceOf M (χ x) x) Bot.bot
  -/
  contrapose! this
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : LieModule.Weight R L M
    x : L
    this : Eq (LieModule.genWeightSpaceOf M (χ x) x) Bot.bot
    ⊢ Eq (iInf fun x => LieModule.genWeightSpaceOf M (χ x) x) Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : LieModule.Weight R L M
    x : L
    this : Eq (LieModule.genWeightSpaceOf M (χ x) x) Bot.bot
    ⊢ LE.le (iInf fun x => LieModule.genWeightSpaceOf M (χ x) x) Bot.bot
  -/
  exact le_of_le_of_eq (iInf_le _ _) this
  /-
    🎉 no goals
  -/


lemma hasEigenvalueAt (χ : Weight R L M) (x : L) :
    (toEnd R L M x).HasEigenvalue (χ x) := by
  obtain ⟨k : ℕ, hk : (toEnd R L M x).genEigenspace (χ x) k ≠ ⊥⟩ := by
    simpa [genWeightSpaceOf, ← Module.End.iSup_genEigenspace_eq] using χ.genWeightSpaceOf_ne_bot x
  /-
    case intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : LieModule.Weight R L M
    x : L
    k : Nat
    hk : Ne ((((LieModule.toEnd R L M) x).genEigenspace (χ x)) ↑k) Bot.bot
    ⊢ ((LieModule.toEnd R L M) x).HasEigenvalue (χ x)
  -/
  exact Module.End.hasEigenvalue_of_hasGenEigenvalue hk
  /-
    🎉 no goals
  -/


lemma apply_eq_zero_of_isNilpotent [NoZeroSMulDivisors R M] [IsReduced R]
    (x : L) (h : _root_.IsNilpotent (toEnd R L M x)) (χ : Weight R L M) :
    χ x = 0 :=
  ((χ.hasEigenvalueAt x).isNilpotent_of_isNilpotent h).eq_zero


/-- See also the more useful form `LieModule.zero_genWeightSpace_eq_top_of_nilpotent`. -/
@[simp]
theorem zero_genWeightSpace_eq_top_of_nilpotent' [IsNilpotent R L M] :
    genWeightSpace M (0 : L → R) = ⊤ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Eq (LieModule.genWeightSpace M 0) Top.top
  -/
  ext
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsNilpotent R L M
    m✝ : M
    ⊢ Iff (Membership.mem (LieModule.genWeightSpace M 0) m✝) (Membership.mem Top.t …
  -/
  simp [genWeightSpace, genWeightSpaceOf]
  /-
    🎉 no goals
  -/


theorem coe_genWeightSpace_of_top (χ : L → R) :
    (genWeightSpace M (χ ∘ (⊤ : LieSubalgebra R L).incl) : Submodule R M) = genWeightSpace M χ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    ⊢ Eq ↑(LieModule.genWeightSpace M (Function.comp χ ⇑Top.top.incl)) ↑(LieModule …
  -/
  ext m
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    m : M
    ⊢ Iff (Membership.mem (↑(LieModule.genWeightSpace M (Function.comp χ ⇑Top.top. …
  -/
  simp only [mem_genWeightSpace, LieSubmodule.mem_toSubmodule, Subtype.forall]
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    m : M
    ⊢ Iff (∀ (a : L) (b : Membership.mem Top.top a), Exists fun k => Eq ((HPow.hPo …
  -/
  apply forall_congr'
  /-
    case h.h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ : L → R
    m : M
    ⊢ ∀ (a : L), Iff (∀ (b : Membership.mem Top.top a), Exists fun k => Eq ((HPow. …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_genWeightSpace_eq_top_of_nilpotent [IsNilpotent R L M] :
    genWeightSpace M (0 : (⊤ : LieSubalgebra R L) → R) = ⊤ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Eq (LieModule.genWeightSpace M 0) Top.top
  -/
  ext m
  simp only [mem_genWeightSpace, Pi.zero_apply, zero_smul, sub_zero, Subtype.forall,
    forall_true_left, LieSubalgebra.toEnd_mk, LieSubalgebra.mem_top, LieSubmodule.mem_top, iff_true]
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsNilpotent R L M
    m : M
    ⊢ ∀ (a : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) a) k) m) 0
  -/
  intro x
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsNilpotent R L M
    m : M
    x : L
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) m) 0
  -/
  obtain ⟨k, hk⟩ := exists_forall_pow_toEnd_eq_zero R L M
  /-
    case h.intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsNilpotent R L M
    m : M
    x : L
    k : Nat
    hk : ∀ (x : L), Eq (HPow.hPow ((LieModule.toEnd R L M) x) k) 0
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) m) 0
  -/
  exact ⟨k, by simp [hk x]⟩
  /-
    🎉 no goals
  -/


theorem exists_genWeightSpace_le_ker_of_isNoetherian [IsNoetherian R M] (χ : L → R) (x : L) :
    ∃ k : ℕ,
      genWeightSpace M χ ≤ LinearMap.ker ((toEnd R L M x - algebraMap R _ (χ x)) ^ k) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    χ : L → R
    x : L
    ⊢ Exists fun k => LE.le (↑(LieModule.genWeightSpace M χ)) (LinearMap.ker (HPow …
  -/
  use (toEnd R L M x).maxGenEigenspaceIndex (χ x)
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    χ : L → R
    x : L
    ⊢ LE.le (↑(LieModule.genWeightSpace M χ)) (LinearMap.ker (HPow.hPow (HSub.hSub …
  -/
  intro m hm
  replace hm : m ∈ (toEnd R L M x).maxGenEigenspace (χ x) :=
    genWeightSpace_le_genWeightSpaceOf M x χ hm
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    χ : L → R
    x : L
    m : M
    hm : Membership.mem (((LieModule.toEnd R L M) x).maxGenEigenspace (χ x)) m
    ⊢ Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub ((LieModule.toEnd R L M) …
  -/
  rwa [Module.End.maxGenEigenspace_eq, Module.End.genEigenspace_nat] at hm
  /-
    🎉 no goals
  -/


variable (R) in
theorem exists_genWeightSpace_zero_le_ker_of_isNoetherian
    [IsNoetherian R M] (x : L) :
    ∃ k : ℕ, genWeightSpace M (0 : L → R) ≤ LinearMap.ker (toEnd R L M x ^ k) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    x : L
    ⊢ Exists fun k => LE.le (↑(LieModule.genWeightSpace M 0)) (LinearMap.ker (HPow …
  -/
  simpa using exists_genWeightSpace_le_ker_of_isNoetherian M (0 : L → R) x
  /-
    🎉 no goals
  -/


lemma isNilpotent_toEnd_sub_algebraMap [IsNoetherian R M] (χ : L → R) (x : L) :
    _root_.IsNilpotent <| toEnd R L (genWeightSpace M χ) x - algebraMap R _ (χ x) := by
  have : toEnd R L (genWeightSpace M χ) x - algebraMap R _ (χ x) =
      (toEnd R L M x - algebraMap R _ (χ x)).restrict
        (fun m hm ↦ sub_mem (LieSubmodule.lie_mem _ hm) (Submodule.smul_mem _ _ hm)) := by
    rfl
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    χ : L → R
    x : L
    this : Eq (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Membership.mem (L …
    ⊢ _root_.IsNilpotent (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Member …
  -/
  obtain ⟨k, hk⟩ := exists_genWeightSpace_le_ker_of_isNoetherian M χ x
  /-
    case intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    χ : L → R
    x : L
    this : Eq (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Membership.mem (L …
    k : Nat
    hk : LE.le (↑(LieModule.genWeightSpace M χ)) (LinearMap.ker (HPow.hPow (HSub.h …
    ⊢ _root_.IsNilpotent (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Member …
  -/
  use k
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    χ : L → R
    x : L
    this : Eq (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Membership.mem (L …
    k : Nat
    hk : LE.le (↑(LieModule.genWeightSpace M χ)) (LinearMap.ker (HPow.hPow (HSub.h …
    ⊢ Eq (HPow.hPow (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Membership. …
  -/
  ext ⟨m, hm⟩
  simp only [this, LinearMap.pow_restrict _, LinearMap.zero_apply, ZeroMemClass.coe_zero,
    ZeroMemClass.coe_eq_zero]
  /-
    case h.h.mk.a
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    χ : L → R
    x : L
    this : Eq (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Membership.mem (L …
    k : Nat
    hk : LE.le (↑(LieModule.genWeightSpace M χ)) (LinearMap.ker (HPow.hPow (HSub.h …
    m : M
    hm : Membership.mem (LieModule.genWeightSpace M χ) m
    ⊢ Eq (((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M) x) ((algebraMap R (Modul …
  -/
  exact ZeroMemClass.coe_eq_zero.mp (hk hm)
  /-
    🎉 no goals
  -/


/-- A (nilpotent) Lie algebra acts nilpotently on the zero weight space of a Noetherian Lie
module. -/
theorem isNilpotent_toEnd_genWeightSpace_zero [IsNoetherian R M] (x : L) :
    _root_.IsNilpotent <| toEnd R L (genWeightSpace M (0 : L → R)) x := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    x : L
    ⊢ _root_.IsNilpotent ((LieModule.toEnd R L (Subtype fun x => Membership.mem (L …
  -/
  simpa using isNilpotent_toEnd_sub_algebraMap M (0 : L → R) x
  /-
    🎉 no goals
  -/


/-- By Engel's theorem, the zero weight space of a Noetherian Lie module is nilpotent. -/
instance [IsNoetherian R M] :
    IsNilpotent R L (genWeightSpace M (0 : L → R)) :=
  isNilpotent_iff_forall'.mpr <| isNilpotent_toEnd_genWeightSpace_zero M


@[simp]
lemma genWeightSpace_zero_normalizer_eq_self :
    (genWeightSpace M (0 : L → R)).normalizer = genWeightSpace M 0 := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    ⊢ Eq (LieModule.genWeightSpace M 0).normalizer (LieModule.genWeightSpace M 0)
  -/
  refine le_antisymm ?_ (LieSubmodule.le_normalizer _)
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    ⊢ LE.le (LieModule.genWeightSpace M 0).normalizer (LieModule.genWeightSpace M 0)
  -/
  intro m hm
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    m : M
    hm : Membership.mem (LieModule.genWeightSpace M 0).normalizer m
    ⊢ Membership.mem (LieModule.genWeightSpace M 0) m
  -/
  rw [LieSubmodule.mem_normalizer] at hm
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    m : M
    hm : ∀ (x : L), Membership.mem (LieModule.genWeightSpace M 0) (Bracket.bracket …
    ⊢ Membership.mem (LieModule.genWeightSpace M 0) m
  -/
  simp only [mem_genWeightSpace, Pi.zero_apply, zero_smul, sub_zero] at hm ⊢
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    m : M
    hm : ∀ (x x_1 : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x_ …
    ⊢ ∀ (x : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) m) 0
  -/
  intro y
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    m : M
    hm : ∀ (x x_1 : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x_ …
    y : L
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k) m) 0
  -/
  obtain ⟨k, hk⟩ := hm y y
  /-
    case intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    m : M
    hm : ∀ (x x_1 : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x_ …
    y : L
    k : Nat
    hk : Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k) (Bracket.bracket y m)) 0
    ⊢ Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k) m) 0
  -/
  use k + 1
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    m : M
    hm : ∀ (x x_1 : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x_ …
    y : L
    k : Nat
    hk : Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k) (Bracket.bracket y m)) 0
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R L M) y) (HAdd.hAdd k 1)) m) 0
  -/
  simpa [pow_succ, LinearMap.mul_eq_comp]
  /-
    🎉 no goals
  -/


lemma iSup_ucs_le_genWeightSpace_zero :
    ⨆ k, (⊥ : LieSubmodule R L M).ucs k ≤ genWeightSpace M (0 : L → R) := by
  simpa using
    LieSubmodule.ucs_le_of_normalizer_eq_self (genWeightSpace_zero_normalizer_eq_self R L M)


/-- See also `LieModule.iInf_lowerCentralSeries_eq_posFittingComp`. -/
lemma iSup_ucs_eq_genWeightSpace_zero [IsNoetherian R M] :
    ⨆ k, (⊥ : LieSubmodule R L M).ucs k = genWeightSpace M (0 : L → R) := by
  obtain ⟨k, hk⟩ := (LieSubmodule.isNilpotent_iff_exists_self_le_ucs
    <| genWeightSpace M (0 : L → R)).mp inferInstance
  /-
    case intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    k : Nat
    hk : LE.le (LieModule.genWeightSpace M 0) (LieSubmodule.ucs k Bot.bot)
    ⊢ Eq (iSup fun k => LieSubmodule.ucs k Bot.bot) (LieModule.genWeightSpace M 0)
  -/
  refine le_antisymm (iSup_ucs_le_genWeightSpace_zero R L M) (le_trans hk ?_)
  /-
    case intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : IsNoetherian R M
    k : Nat
    hk : LE.le (LieModule.genWeightSpace M 0) (LieSubmodule.ucs k Bot.bot)
    ⊢ LE.le (LieSubmodule.ucs k Bot.bot) (iSup fun k => LieSubmodule.ucs k Bot.bot)
  -/
  exact le_iSup (fun k ↦ (⊥ : LieSubmodule R L M).ucs k) k
  /-
    🎉 no goals
  -/


/-- If `M` is a representation of a nilpotent Lie algebra `L`, and `x : L`, then
`posFittingCompOf R M x` is the infimum of the decreasing system
`range φₓ ⊇ range φₓ² ⊇ range φₓ³ ⊇ ⋯` where `φₓ : End R M := toEnd R L M x`. We call this
the "positive Fitting component" because with appropriate assumptions (e.g., `R` is a field and
`M` is finite-dimensional) `φₓ` induces the so-called Fitting decomposition: `M = M₀ ⊕ M₁` where
`M₀ = genWeightSpaceOf M 0 x` and `M₁ = posFittingCompOf R M x`.

It is a Lie submodule because `L` is nilpotent. -/
def posFittingCompOf (x : L) : LieSubmodule R L M :=
  { toSubmodule := ⨅ k, LinearMap.range (toEnd R L M x ^ k)
    lie_mem := by
      /-
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        ⊢ ∀ {x_1 : L} {m : M}, Membership.mem (iInf fun k => LinearMap.range (HPow.hPo …
      -/
      set φ := toEnd R L M x
      /-
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        ⊢ ∀ {x : L} {m : M}, Membership.mem (iInf fun k => LinearMap.range (HPow.hPow  …
      -/
      intros y m hm
      simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
        Submodule.mem_toAddSubmonoid, Submodule.mem_iInf, LinearMap.mem_range] at hm ⊢
      /-
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) m
        ⊢ ∀ (i : Nat), Exists fun y_1 => Eq ((HPow.hPow φ i) y_1) (Bracket.bracket y m)
      -/
      intro k
      /-
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) m
        k : Nat
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y m)
      -/
      obtain ⟨N, hN⟩ := LieAlgebra.nilpotent_ad_of_nilpotent_algebra R L
      /-
        case intro
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) m
        k N : Nat
        hN : ∀ (x : L), Eq (HPow.hPow ((LieAlgebra.ad R L) x) N) 0
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y m)
      -/
      obtain ⟨m, rfl⟩ := hm (N + k)
      /-
        case intro.intro
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        k N : Nat
        hN : ∀ (x : L), Eq (HPow.hPow ((LieAlgebra.ad R L) x) N) 0
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) ((HPow.hPow φ (HAdd.h …
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y ((HPow.hPow φ  …
      -/
      let f₁ : Module.End R (L ⊗[R] M) := (LieAlgebra.ad R L x).rTensor M
      /-
        case intro.intro
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        k N : Nat
        hN : ∀ (x : L), Eq (HPow.hPow ((LieAlgebra.ad R L) x) N) 0
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) ((HPow.hPow φ (HAdd.h …
        f₁ : Module.End R (TensorProduct R L M) := LinearMap.rTensor M ((LieAlgebra.ad …
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y ((HPow.hPow φ  …
      -/
      let f₂ : Module.End R (L ⊗[R] M) := φ.lTensor L
      /-
        case intro.intro
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        k N : Nat
        hN : ∀ (x : L), Eq (HPow.hPow ((LieAlgebra.ad R L) x) N) 0
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) ((HPow.hPow φ (HAdd.h …
        f₁ : Module.End R (TensorProduct R L M) := LinearMap.rTensor M ((LieAlgebra.ad …
        f₂ : Module.End R (TensorProduct R L M) := LinearMap.lTensor L φ
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y ((HPow.hPow φ  …
      -/
      replace hN : f₁ ^ N = 0 := by ext; simp [f₁, hN]
      /-
        case intro.intro
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        k N : Nat
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) ((HPow.hPow φ (HAdd.h …
        f₁ : Module.End R (TensorProduct R L M) := LinearMap.rTensor M ((LieAlgebra.ad …
        f₂ : Module.End R (TensorProduct R L M) := LinearMap.lTensor L φ
        hN : Eq (HPow.hPow f₁ N) 0
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y ((HPow.hPow φ  …
      -/
      have h₁ : Commute f₁ f₂ := by ext; simp [f₁, f₂]
      /-
        case intro.intro
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        k N : Nat
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) ((HPow.hPow φ (HAdd.h …
        f₁ : Module.End R (TensorProduct R L M) := LinearMap.rTensor M ((LieAlgebra.ad …
        f₂ : Module.End R (TensorProduct R L M) := LinearMap.lTensor L φ
        hN : Eq (HPow.hPow f₁ N) 0
        h₁ : Commute f₁ f₂
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y ((HPow.hPow φ  …
      -/
      have h₂ : φ ∘ₗ toModuleHom R L M = toModuleHom R L M ∘ₗ (f₁ + f₂) := by ext; simp [φ, f₁, f₂]
      /-
        case intro.intro
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        k N : Nat
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) ((HPow.hPow φ (HAdd.h …
        f₁ : Module.End R (TensorProduct R L M) := LinearMap.rTensor M ((LieAlgebra.ad …
        f₂ : Module.End R (TensorProduct R L M) := LinearMap.lTensor L φ
        hN : Eq (HPow.hPow f₁ N) 0
        h₁ : Commute f₁ f₂
        h₂ : Eq (LinearMap.comp φ ↑(LieModule.toModuleHom R L M)) ((↑(LieModule.toModu …
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y ((HPow.hPow φ  …
      -/
      obtain ⟨q, hq⟩ := h₁.add_pow_dvd_pow_of_pow_eq_zero_right (N + k).le_succ hN
      /-
        case intro.intro.intro
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        k N : Nat
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) ((HPow.hPow φ (HAdd.h …
        f₁ : Module.End R (TensorProduct R L M) := LinearMap.rTensor M ((LieAlgebra.ad …
        f₂ : Module.End R (TensorProduct R L M) := LinearMap.lTensor L φ
        hN : Eq (HPow.hPow f₁ N) 0
        h₁ : Commute f₁ f₂
        h₂ : Eq (LinearMap.comp φ ↑(LieModule.toModuleHom R L M)) ((↑(LieModule.toModu …
        q : Module.End R (TensorProduct R L M)
        hq : Eq (HPow.hPow f₂ (HAdd.hAdd N k)) (HMul.hMul (HPow.hPow (HAdd.hAdd f₁ f₂) …
        ⊢ Exists fun y_1 => Eq ((HPow.hPow φ k) y_1) (Bracket.bracket y ((HPow.hPow φ  …
      -/
      use toModuleHom R L M (q (y ⊗ₜ m))
      /-
        case h
        K : Type u_1
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁷ : CommRing R
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        inst✝¹ : LieModule R L M
        inst✝ : LieAlgebra.IsNilpotent R L
        x : L
        φ : Module.End R M := (LieModule.toEnd R L M) x
        y : L
        k N : Nat
        m : M
        hm : ∀ (i : Nat), Exists fun y => Eq ((HPow.hPow φ i) y) ((HPow.hPow φ (HAdd.h …
        f₁ : Module.End R (TensorProduct R L M) := LinearMap.rTensor M ((LieAlgebra.ad …
        f₂ : Module.End R (TensorProduct R L M) := LinearMap.lTensor L φ
        hN : Eq (HPow.hPow f₁ N) 0
        h₁ : Commute f₁ f₂
        h₂ : Eq (LinearMap.comp φ ↑(LieModule.toModuleHom R L M)) ((↑(LieModule.toModu …
        q : Module.End R (TensorProduct R L M)
        hq : Eq (HPow.hPow f₂ (HAdd.hAdd N k)) (HMul.hMul (HPow.hPow (HAdd.hAdd f₁ f₂) …
        ⊢ Eq ((HPow.hPow φ k) ((LieModule.toModuleHom R L M) (q (TensorProduct.tmul R  …
      -/
      change (φ ^ k).comp ((toModuleHom R L M : L ⊗[R] M →ₗ[R] M)) _ = _
      simp [φ, f₁, f₂, LinearMap.commute_pow_left_of_commute h₂,
        LinearMap.comp_apply (g := (f₁ + f₂) ^ k), ← LinearMap.comp_apply (g := q),
        ← LinearMap.mul_eq_comp, ← hq] }


variable {M} in
lemma mem_posFittingCompOf (x : L) (m : M) :
    m ∈ posFittingCompOf R M x ↔ ∀ (k : ℕ), ∃ n, (toEnd R L M x ^ k) n = m := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    x : L
    m : M
    ⊢ Iff (Membership.mem (LieModule.posFittingCompOf R M x) m) (∀ (k : Nat), Exis …
  -/
  simp [posFittingCompOf]
  /-
    🎉 no goals
  -/


@[simp] lemma posFittingCompOf_le_lowerCentralSeries (x : L) (k : ℕ) :
    posFittingCompOf R M x ≤ lowerCentralSeries R L M k := by
  suffices ∀ m l, (toEnd R L M x ^ l) m ∈ lowerCentralSeries R L M l by
    intro m hm
    obtain ⟨n, rfl⟩ := (mem_posFittingCompOf R x m).mp hm k
    exact this n k
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    x : L
    k : Nat
    ⊢ ∀ (m : M) (l : Nat), Membership.mem (LieModule.lowerCentralSeries R L M l) ( …
  -/
  intro m l
  induction l with
  | zero => simp
  | succ l ih =>
    simp only [lowerCentralSeries_succ, pow_succ', LinearMap.mul_apply]
    exact LieSubmodule.lie_mem_lie (LieSubmodule.mem_top x) ih


@[simp] lemma posFittingCompOf_eq_bot_of_isNilpotent
    [IsNilpotent R L M] (x : L) :
    posFittingCompOf R M x = ⊥ := by
  simp_rw [eq_bot_iff, ← iInf_lowerCentralSeries_eq_bot_of_isNilpotent, le_iInf_iff,
    posFittingCompOf_le_lowerCentralSeries, forall_const]


/-- If `M` is a representation of a nilpotent Lie algebra `L` with coefficients in `R`, then
`posFittingComp R L M` is the span of the positive Fitting components of the action of `x` on `M`,
as `x` ranges over `L`.

It is a Lie submodule because `L` is nilpotent. -/
def posFittingComp : LieSubmodule R L M :=
  ⨆ x, posFittingCompOf R M x


lemma mem_posFittingComp (m : M) :
    m ∈ posFittingComp R L M ↔ m ∈ ⨆ (x : L), posFittingCompOf R M x := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    m : M
    ⊢ Iff (Membership.mem (LieModule.posFittingComp R L M) m) (Membership.mem (iSu …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma posFittingCompOf_le_posFittingComp (x : L) :
    posFittingCompOf R M x ≤ posFittingComp R L M := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    x : L
    ⊢ LE.le (LieModule.posFittingCompOf R M x) (LieModule.posFittingComp R L M)
  -/
  rw [posFittingComp]; exact le_iSup (posFittingCompOf R M) x
                       /-
                         🎉 no goals
                       -/


lemma posFittingComp_le_iInf_lowerCentralSeries :
    posFittingComp R L M ≤ ⨅ k, lowerCentralSeries R L M k := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    ⊢ LE.le (LieModule.posFittingComp R L M) (iInf fun k => LieModule.lowerCentral …
  -/
  simp [posFittingComp]
  /-
    🎉 no goals
  -/


/-- See also `LieModule.iSup_ucs_eq_genWeightSpace_zero`. -/
@[simp] lemma iInf_lowerCentralSeries_eq_posFittingComp
    [IsNoetherian R M] [IsArtinian R M] :
    ⨅ k, lowerCentralSeries R L M k = posFittingComp R L M := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    ⊢ Eq (iInf fun k => LieModule.lowerCentralSeries R L M k) (LieModule.posFittin …
  -/
  refine le_antisymm ?_ (posFittingComp_le_iInf_lowerCentralSeries R L M)
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    ⊢ LE.le (iInf fun k => LieModule.lowerCentralSeries R L M k) (LieModule.posFit …
  -/
  apply iInf_lcs_le_of_isNilpotent_quot
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    ⊢ LieModule.IsNilpotent R L (HasQuotient.Quotient M (LieModule.posFittingComp  …
  -/
  rw [LieModule.isNilpotent_iff_forall']
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    ⊢ ∀ (x : L), _root_.IsNilpotent ((LieModule.toEnd R L (HasQuotient.Quotient M  …
  -/
  intro x
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    x : L
    ⊢ _root_.IsNilpotent ((LieModule.toEnd R L (HasQuotient.Quotient M (LieModule. …
  -/
  obtain ⟨k, hk⟩ := Filter.eventually_atTop.mp (toEnd R L M x).eventually_iInf_range_pow_eq
  /-
    case h.intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    x : L
    k : Nat
    hk : ∀ (b : Nat), GE.ge b k → Eq (iInf fun m => LinearMap.range (HPow.hPow ((L …
    ⊢ _root_.IsNilpotent ((LieModule.toEnd R L (HasQuotient.Quotient M (LieModule. …
  -/
  use k
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    x : L
    k : Nat
    hk : ∀ (b : Nat), GE.ge b k → Eq (iInf fun m => LinearMap.range (HPow.hPow ((L …
    ⊢ Eq (HPow.hPow ((LieModule.toEnd R L (HasQuotient.Quotient M (LieModule.posFi …
  -/
  ext ⟨m⟩
  /-
    case h.h.mk
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    x : L
    k : Nat
    hk : ∀ (b : Nat), GE.ge b k → Eq (iInf fun m => LinearMap.range (HPow.hPow ((L …
    x✝ : HasQuotient.Quotient M (LieModule.posFittingComp R L M)
    m : M
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R L (HasQuotient.Quotient M (LieModule.posF …
  -/
  set F := posFittingComp R L M
  replace hk : (toEnd R L M x ^ k) m ∈ F := by
    apply posFittingCompOf_le_posFittingComp R L M x
    simp_rw [← LieSubmodule.mem_toSubmodule, posFittingCompOf, hk k (le_refl k)]
    apply LinearMap.mem_range_self
  suffices (toEnd R L (M ⧸ F) x ^ k) (LieSubmodule.Quotient.mk (N := F) m) =
    LieSubmodule.Quotient.mk (N := F) ((toEnd R L M x ^ k) m)
      by simpa [Submodule.Quotient.quot_mk_eq_mk, this]
  have := LinearMap.congr_fun (LinearMap.commute_pow_left_of_commute
    (LieSubmodule.Quotient.toEnd_comp_mk' F x) k) m
  /-
    case h.h.mk
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    x : L
    k : Nat
    m : M
    F : LieSubmodule R L M := LieModule.posFittingComp R L M
    x✝ : HasQuotient.Quotient M F
    hk : Membership.mem F ((HPow.hPow ((LieModule.toEnd R L M) x) k) m)
    this : Eq ((LinearMap.comp (HPow.hPow ((LieModule.toEnd R L (HasQuotient.Quoti …
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R L (HasQuotient.Quotient M F)) x) k) (LieS …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


@[simp] lemma posFittingComp_eq_bot_of_isNilpotent
    [IsNilpotent R L M] :
    posFittingComp R L M = ⊥ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Eq (LieModule.posFittingComp R L M) Bot.bot
  -/
  simp [posFittingComp]
  /-
    🎉 no goals
  -/


lemma map_posFittingComp_le :
    (posFittingComp R L M).map f ≤ posFittingComp R L M₂ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    ⊢ LE.le (LieSubmodule.map f (LieModule.posFittingComp R L M)) (LieModule.posFi …
  -/
  rw [posFittingComp, posFittingComp, LieSubmodule.map_iSup]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    ⊢ LE.le (iSup fun i => LieSubmodule.map f (LieModule.posFittingCompOf R M i))  …
  -/
  refine iSup_mono fun y ↦ LieSubmodule.map_le_iff_le_comap.mpr fun m hm ↦ ?_
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    y : L
    m : M
    hm : Membership.mem (LieModule.posFittingCompOf R M y) m
    ⊢ Membership.mem (LieSubmodule.comap f (LieModule.posFittingCompOf R M₂ y)) m
  -/
  simp only [mem_posFittingCompOf] at hm
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    y : L
    m : M
    hm : ∀ (k : Nat), Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k …
    ⊢ Membership.mem (LieSubmodule.comap f (LieModule.posFittingCompOf R M₂ y)) m
  -/
  simp only [LieSubmodule.mem_comap, mem_posFittingCompOf]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    y : L
    m : M
    hm : ∀ (k : Nat), Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k …
    ⊢ ∀ (k : Nat), Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M₂) y) k)  …
  -/
  intro k
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    y : L
    m : M
    hm : ∀ (k : Nat), Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k …
    k : Nat
    ⊢ Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M₂) y) k) n) (f m)
  -/
  obtain ⟨n, hn⟩ := hm k
  /-
    case intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    y : L
    m : M
    hm : ∀ (k : Nat), Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k …
    k : Nat
    n : M
    hn : Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k) n) m
    ⊢ Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M₂) y) k) n) (f m)
  -/
  use f n
  /-
    case h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    y : L
    m : M
    hm : ∀ (k : Nat), Exists fun n => Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k …
    k : Nat
    n : M
    hn : Eq ((HPow.hPow ((LieModule.toEnd R L M) y) k) n) m
    ⊢ Eq ((HPow.hPow ((LieModule.toEnd R L M₂) y) k) (f n)) (f m)
  -/
  rw [LieModule.toEnd_pow_apply_map, hn]
  /-
    🎉 no goals
  -/


lemma map_genWeightSpace_le :
    (genWeightSpace M χ).map f ≤ genWeightSpace M₂ χ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    ⊢ LE.le (LieSubmodule.map f (LieModule.genWeightSpace M χ)) (LieModule.genWeig …
  -/
  rw [LieSubmodule.map_le_iff_le_comap]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    ⊢ LE.le (LieModule.genWeightSpace M χ) (LieSubmodule.comap f (LieModule.genWei …
  -/
  intro m hm
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    m : M
    hm : Membership.mem (LieModule.genWeightSpace M χ) m
    ⊢ Membership.mem (LieSubmodule.comap f (LieModule.genWeightSpace M₂ χ)) m
  -/
  simp only [LieSubmodule.mem_comap, mem_genWeightSpace]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    m : M
    hm : Membership.mem (LieModule.genWeightSpace M χ) m
    ⊢ ∀ (x : L), Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M …
  -/
  intro x
  have : (toEnd R L M₂ x - χ x • ↑1) ∘ₗ f = f ∘ₗ (toEnd R L M x - χ x • ↑1) := by
    ext; simp
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    m : M
    hm : Membership.mem (LieModule.genWeightSpace M χ) m
    x : L
    this : Eq ((HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul (χ x) 1)).comp …
    ⊢ Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMu …
  -/
  obtain ⟨k, h⟩ := (mem_genWeightSpace _ _ _).mp hm x
  /-
    case intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    m : M
    hm : Membership.mem (LieModule.genWeightSpace M χ) m
    x : L
    this : Eq ((HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul (χ x) 1)).comp …
    k : Nat
    h : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M) x) (HSMul.hSMul (χ x) 1 …
    ⊢ Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMu …
  -/
  exact ⟨k, by simpa [h] using LinearMap.congr_fun (LinearMap.commute_pow_left_of_commute this k) m⟩
  /-
    🎉 no goals
  -/


lemma comap_genWeightSpace_eq_of_injective (hf : Injective f) :
    (genWeightSpace M₂ χ).comap f = genWeightSpace M χ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    hf : Function.Injective ⇑f
    ⊢ Eq (LieSubmodule.comap f (LieModule.genWeightSpace M₂ χ)) (LieModule.genWeig …
  -/
  refine le_antisymm (fun m hm ↦ ?_) ?_
    /-
      case refine_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      M₂ : Type u_5
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L M₂
      inst✝ : LieModule R L M₂
      χ : L → R
      f : LieModuleHom R L M M₂
      hf : Function.Injective ⇑f
      m : M
      hm : Membership.mem (LieSubmodule.comap f (LieModule.genWeightSpace M₂ χ)) m
      ⊢ Membership.mem (LieModule.genWeightSpace M χ) m
    -/
  · simp only [LieSubmodule.mem_comap, mem_genWeightSpace] at hm
    /-
      case refine_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      M₂ : Type u_5
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L M₂
      inst✝ : LieModule R L M₂
      χ : L → R
      f : LieModuleHom R L M M₂
      hf : Function.Injective ⇑f
      m : M
      hm : ∀ (x : L), Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R  …
      ⊢ Membership.mem (LieModule.genWeightSpace M χ) m
    -/
    simp only [mem_genWeightSpace]
    /-
      case refine_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      M₂ : Type u_5
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L M₂
      inst✝ : LieModule R L M₂
      χ : L → R
      f : LieModuleHom R L M M₂
      hf : Function.Injective ⇑f
      m : M
      hm : ∀ (x : L), Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R  …
      ⊢ ∀ (x : L), Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M …
    -/
    intro x
    have h : (toEnd R L M₂ x - χ x • ↑1) ∘ₗ f =
             f ∘ₗ (toEnd R L M x - χ x • ↑1) := by ext; simp
    /-
      case refine_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      M₂ : Type u_5
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L M₂
      inst✝ : LieModule R L M₂
      χ : L → R
      f : LieModuleHom R L M M₂
      hf : Function.Injective ⇑f
      m : M
      hm : ∀ (x : L), Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R  …
      x : L
      h : Eq ((HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul (χ x) 1)).comp ↑f …
      ⊢ Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M) x) (HSMul …
    -/
    obtain ⟨k, hk⟩ := hm x
    /-
      case refine_1.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      M₂ : Type u_5
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L M₂
      inst✝ : LieModule R L M₂
      χ : L → R
      f : LieModuleHom R L M M₂
      hf : Function.Injective ⇑f
      m : M
      hm : ∀ (x : L), Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R  …
      x : L
      h : Eq ((HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul (χ x) 1)).comp ↑f …
      k : Nat
      hk : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul (χ x) …
      ⊢ Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M) x) (HSMul …
    -/
    use k
    suffices f (((toEnd R L M x - χ x • ↑1) ^ k) m) = 0 by
      rw [← f.map_zero] at this; exact hf this
    /-
      case h
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      M₂ : Type u_5
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L M₂
      inst✝ : LieModule R L M₂
      χ : L → R
      f : LieModuleHom R L M M₂
      hf : Function.Injective ⇑f
      m : M
      hm : ∀ (x : L), Exists fun k => Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R  …
      x : L
      h : Eq ((HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul (χ x) 1)).comp ↑f …
      k : Nat
      hk : Eq ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M₂) x) (HSMul.hSMul (χ x) …
      ⊢ Eq (f ((HPow.hPow (HSub.hSub ((LieModule.toEnd R L M) x) (HSMul.hSMul (χ x)  …
    -/
    simpa [hk] using (LinearMap.congr_fun (LinearMap.commute_pow_left_of_commute h k) m).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      M₂ : Type u_5
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L M₂
      inst✝ : LieModule R L M₂
      χ : L → R
      f : LieModuleHom R L M M₂
      hf : Function.Injective ⇑f
      ⊢ LE.le (LieModule.genWeightSpace M χ) (LieSubmodule.comap f (LieModule.genWei …
    -/
  · rw [← LieSubmodule.map_le_iff_le_comap]
    /-
      case refine_2
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      M₂ : Type u_5
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : LieRingModule L M₂
      inst✝ : LieModule R L M₂
      χ : L → R
      f : LieModuleHom R L M M₂
      hf : Function.Injective ⇑f
      ⊢ LE.le (LieSubmodule.map f (LieModule.genWeightSpace M χ)) (LieModule.genWeig …
    -/
    exact map_genWeightSpace_le f
    /-
      🎉 no goals
    -/


lemma map_genWeightSpace_eq_of_injective (hf : Injective f) :
    (genWeightSpace M χ).map f = genWeightSpace M₂ χ ⊓ f.range := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    hf : Function.Injective ⇑f
    ⊢ Eq (LieSubmodule.map f (LieModule.genWeightSpace M χ)) (Min.min (LieModule.g …
  -/
  refine le_antisymm (le_inf_iff.mpr ⟨map_genWeightSpace_le f, LieSubmodule.map_le_range f⟩) ?_
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    hf : Function.Injective ⇑f
    ⊢ LE.le (Min.min (LieModule.genWeightSpace M₂ χ) f.range) (LieSubmodule.map f  …
  -/
  rintro - ⟨hm, ⟨m, rfl⟩⟩
  simp only [← comap_genWeightSpace_eq_of_injective hf, LieSubmodule.mem_map,
    LieSubmodule.mem_comap]
  /-
    case intro.intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    f : LieModuleHom R L M M₂
    hf : Function.Injective ⇑f
    m : M
    hm : Membership.mem (↑↑(LieModule.genWeightSpace M₂ χ)) (f m)
    ⊢ Exists fun m_1 => And (Membership.mem (LieModule.genWeightSpace M₂ χ) (f m_1 …
  -/
  exact ⟨m, hm, rfl⟩
  /-
    🎉 no goals
  -/


lemma map_genWeightSpace_eq (e : M ≃ₗ⁅R,L⁆ M₂) :
    (genWeightSpace M χ).map e = genWeightSpace M₂ χ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    χ : L → R
    e : LieModuleEquiv R L M M₂
    ⊢ Eq (LieSubmodule.map e.toLieModuleHom (LieModule.genWeightSpace M χ)) (LieMo …
  -/
  simp [map_genWeightSpace_eq_of_injective e.injective]
  /-
    🎉 no goals
  -/


lemma map_posFittingComp_eq (e : M ≃ₗ⁅R,L⁆ M₂) :
    (posFittingComp R L M).map e = posFittingComp R L M₂ := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    e : LieModuleEquiv R L M M₂
    ⊢ Eq (LieSubmodule.map e.toLieModuleHom (LieModule.posFittingComp R L M)) (Lie …
  -/
  refine le_antisymm (map_posFittingComp_le _) ?_
  suffices posFittingComp R L M₂ = ((posFittingComp R L M₂).map (e.symm : M₂ →ₗ⁅R,L⁆ M)).map e by
    rw [this]
    exact LieSubmodule.map_mono (map_posFittingComp_le _)
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    e : LieModuleEquiv R L M M₂
    ⊢ Eq (LieModule.posFittingComp R L M₂) (LieSubmodule.map e.toLieModuleHom (Lie …
  -/
  rw [← LieSubmodule.map_comp]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    e : LieModuleEquiv R L M M₂
    ⊢ Eq (LieModule.posFittingComp R L M₂) (LieSubmodule.map (e.comp e.symm.toLieM …
  -/
  convert LieSubmodule.map_id
  /-
    case h.e'_2
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    e : LieModuleEquiv R L M M₂
    ⊢ Eq (LieModule.posFittingComp R L M₂) (LieSubmodule.map LieModuleHom.id (LieS …
  -/
  ext
  /-
    case h.e'_2.h
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    M₂ : Type u_5
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    e : LieModuleEquiv R L M M₂
    m✝ : M₂
    ⊢ Iff (Membership.mem (LieModule.posFittingComp R L M₂) m✝) (Membership.mem (L …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma posFittingComp_map_incl_sup_of_codisjoint [IsNoetherian R M] [IsArtinian R M]
    {N₁ N₂ : LieSubmodule R L M} (h : Codisjoint N₁ N₂) :
    (posFittingComp R L N₁).map N₁.incl ⊔ (posFittingComp R L N₂).map N₂.incl =
    posFittingComp R L M := by
  obtain ⟨l, hl⟩ := Filter.eventually_atTop.mp <|
    (eventually_iInf_lowerCentralSeries_eq R L N₁).and <|
    (eventually_iInf_lowerCentralSeries_eq R L N₂).and
    (eventually_iInf_lowerCentralSeries_eq R L M)
  /-
    case intro
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    N₁ N₂ : LieSubmodule R L M
    h : Codisjoint N₁ N₂
    l : Nat
    hl : ∀ (b : Nat), GE.ge b l → And (Eq (iInf fun k => LieModule.lowerCentralSer …
    ⊢ Eq (Max.max (LieSubmodule.map N₁.incl (LieModule.posFittingComp R L (Subtype …
  -/
  obtain ⟨hl₁, hl₂, hl₃⟩ := hl l (le_refl _)
  simp_rw [← iInf_lowerCentralSeries_eq_posFittingComp, hl₁, hl₂, hl₃,
    LieSubmodule.lowerCentralSeries_map_eq_lcs, ← LieSubmodule.lcs_sup, lowerCentralSeries,
    h.eq_top]


lemma genWeightSpace_genWeightSpaceOf_map_incl (x : L) (χ : L → R) :
    (genWeightSpace (genWeightSpaceOf M (χ x) x) χ).map (genWeightSpaceOf M (χ x) x).incl =
    genWeightSpace M χ := by
  simpa [map_genWeightSpace_eq_of_injective (genWeightSpaceOf M (χ x) x).injective_incl]
    using genWeightSpace_le_genWeightSpaceOf M x χ


lemma isCompl_genWeightSpaceOf_zero_posFittingCompOf (x : L) :
    IsCompl (genWeightSpaceOf M 0 x) (posFittingCompOf R M x) := by
  simpa only [isCompl_iff, codisjoint_iff, disjoint_iff, ← LieSubmodule.toSubmodule_inj,
    LieSubmodule.sup_toSubmodule, LieSubmodule.inf_toSubmodule,
    LieSubmodule.top_toSubmodule, LieSubmodule.bot_toSubmodule, coe_genWeightSpaceOf_zero] using
    (toEnd R L M x).isCompl_iSup_ker_pow_iInf_range_pow


/-- This lemma exists only to simplify the proof of
`LieModule.isCompl_genWeightSpace_zero_posFittingComp`. -/
private lemma isCompl_genWeightSpace_zero_posFittingComp_aux
    (h : ∀ N < (⊤ : LieSubmodule R L M), IsCompl (genWeightSpace N 0) (posFittingComp R L N)) :
    IsCompl (genWeightSpace M 0) (posFittingComp R L M) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
    ⊢ IsCompl (LieModule.genWeightSpace M 0) (LieModule.posFittingComp R L M)
  -/
  set M₀ := genWeightSpace M (0 : L → R)
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
    M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
    ⊢ IsCompl M₀ (LieModule.posFittingComp R L M)
  -/
  set M₁ := posFittingComp R L M
  rcases forall_or_exists_not (fun (x : L) ↦ genWeightSpaceOf M (0 : R) x = ⊤)
    with h | ⟨x, hx : genWeightSpaceOf M (0 : R) x ≠ ⊤⟩
    /-
      case inl
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h✝ : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeigh …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      h : ∀ (a : L), Eq (LieModule.genWeightSpaceOf M 0 a) Top.top
      ⊢ IsCompl M₀ M₁
    -/
  · suffices IsNilpotent R L M by simp [M₀, M₁, isCompl_top_bot]
    /-
      case inl
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h✝ : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeigh …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      h : ∀ (a : L), Eq (LieModule.genWeightSpaceOf M 0 a) Top.top
      ⊢ LieModule.IsNilpotent R L M
    -/
    replace h : M₀ = ⊤ := by simpa [M₀, genWeightSpace]
    /-
      case inl
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h✝ : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeigh …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      h : Eq M₀ Top.top
      ⊢ LieModule.IsNilpotent R L M
    -/
    rw [← LieModule.isNilpotent_of_top_iff', ← h]
    /-
      case inl
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h✝ : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeigh …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      h : Eq M₀ Top.top
      ⊢ LieModule.IsNilpotent R L (Subtype fun x => Membership.mem M₀ x)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      x : L
      hx : Ne (LieModule.genWeightSpaceOf M 0 x) Top.top
      ⊢ IsCompl M₀ M₁
    -/
  · set M₀ₓ := genWeightSpaceOf M (0 : R) x
    /-
      case inr.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      x : L
      M₀ₓ : LieSubmodule R L M := LieModule.genWeightSpaceOf M 0 x
      hx : Ne M₀ₓ Top.top
      ⊢ IsCompl M₀ M₁
    -/
    set M₁ₓ := posFittingCompOf R M x
    /-
      case inr.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      x : L
      M₀ₓ : LieSubmodule R L M := LieModule.genWeightSpaceOf M 0 x
      hx : Ne M₀ₓ Top.top
      M₁ₓ : LieSubmodule R L M := LieModule.posFittingCompOf R M x
      ⊢ IsCompl M₀ M₁
    -/
    set M₀ₓ₀ := genWeightSpace M₀ₓ (0 : L → R)
    /-
      case inr.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      x : L
      M₀ₓ : LieSubmodule R L M := LieModule.genWeightSpaceOf M 0 x
      hx : Ne M₀ₓ Top.top
      M₁ₓ : LieSubmodule R L M := LieModule.posFittingCompOf R M x
      M₀ₓ₀ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.g …
      ⊢ IsCompl M₀ M₁
    -/
    set M₀ₓ₁ := posFittingComp R L M₀ₓ
    /-
      case inr.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      x : L
      M₀ₓ : LieSubmodule R L M := LieModule.genWeightSpaceOf M 0 x
      hx : Ne M₀ₓ Top.top
      M₁ₓ : LieSubmodule R L M := LieModule.posFittingCompOf R M x
      M₀ₓ₀ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.g …
      M₀ₓ₁ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.p …
      ⊢ IsCompl M₀ M₁
    -/
    have h₁ : IsCompl M₀ₓ M₁ₓ := isCompl_genWeightSpaceOf_zero_posFittingCompOf R L M x
    /-
      case inr.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      x : L
      M₀ₓ : LieSubmodule R L M := LieModule.genWeightSpaceOf M 0 x
      hx : Ne M₀ₓ Top.top
      M₁ₓ : LieSubmodule R L M := LieModule.posFittingCompOf R M x
      M₀ₓ₀ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.g …
      M₀ₓ₁ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.p …
      h₁ : IsCompl M₀ₓ M₁ₓ
      ⊢ IsCompl M₀ M₁
    -/
    have h₂ : IsCompl M₀ₓ₀ M₀ₓ₁ := h M₀ₓ hx.lt_top
    have h₃ : M₀ₓ₀.map M₀ₓ.incl = M₀ := by
      rw [map_genWeightSpace_eq_of_injective M₀ₓ.injective_incl, inf_eq_left,
        LieSubmodule.range_incl]
      exact iInf_le _ x
    have h₄ : M₀ₓ₁.map M₀ₓ.incl ⊔ M₁ₓ = M₁ := by
      apply le_antisymm <| sup_le_iff.mpr
        ⟨map_posFittingComp_le _, posFittingCompOf_le_posFittingComp R L M x⟩
      rw [← posFittingComp_map_incl_sup_of_codisjoint h₁.codisjoint]
      exact sup_le_sup_left LieSubmodule.map_incl_le _
    /-
      case inr.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      x : L
      M₀ₓ : LieSubmodule R L M := LieModule.genWeightSpaceOf M 0 x
      hx : Ne M₀ₓ Top.top
      M₁ₓ : LieSubmodule R L M := LieModule.posFittingCompOf R M x
      M₀ₓ₀ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.g …
      M₀ₓ₁ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.p …
      h₁ : IsCompl M₀ₓ M₁ₓ
      h₂ : IsCompl M₀ₓ₀ M₀ₓ₁
      h₃ : Eq (LieSubmodule.map M₀ₓ.incl M₀ₓ₀) M₀
      h₄ : Eq (Max.max (LieSubmodule.map M₀ₓ.incl M₀ₓ₁) M₁ₓ) M₁
      ⊢ IsCompl M₀ M₁
    -/
    rw [← h₃, ← h₄]
    /-
      case inr.intro
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      inst✝⁴ : LieRingModule L M
      inst✝³ : LieModule R L M
      inst✝² : LieAlgebra.IsNilpotent R L
      inst✝¹ : IsNoetherian R M
      inst✝ : IsArtinian R M
      h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
      M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
      M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
      x : L
      M₀ₓ : LieSubmodule R L M := LieModule.genWeightSpaceOf M 0 x
      hx : Ne M₀ₓ Top.top
      M₁ₓ : LieSubmodule R L M := LieModule.posFittingCompOf R M x
      M₀ₓ₀ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.g …
      M₀ₓ₁ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.p …
      h₁ : IsCompl M₀ₓ M₁ₓ
      h₂ : IsCompl M₀ₓ₀ M₀ₓ₁
      h₃ : Eq (LieSubmodule.map M₀ₓ.incl M₀ₓ₀) M₀
      h₄ : Eq (Max.max (LieSubmodule.map M₀ₓ.incl M₀ₓ₁) M₁ₓ) M₁
      ⊢ IsCompl (LieSubmodule.map M₀ₓ.incl M₀ₓ₀) (Max.max (LieSubmodule.map M₀ₓ.incl …
    -/
    apply Disjoint.isCompl_sup_right_of_isCompl_sup_left
    · rw [disjoint_iff, ← LieSubmodule.map_inf M₀ₓ.injective_incl, h₂.inf_eq_bot,
        LieSubmodule.map_bot]
      /-
        case inr.intro.hcomp
        R : Type u_2
        L : Type u_3
        M : Type u_4
        inst✝⁹ : CommRing R
        inst✝⁸ : LieRing L
        inst✝⁷ : LieAlgebra R L
        inst✝⁶ : AddCommGroup M
        inst✝⁵ : Module R M
        inst✝⁴ : LieRingModule L M
        inst✝³ : LieModule R L M
        inst✝² : LieAlgebra.IsNilpotent R L
        inst✝¹ : IsNoetherian R M
        inst✝ : IsArtinian R M
        h : ∀ (N : LieSubmodule R L M), LT.lt N Top.top → IsCompl (LieModule.genWeight …
        M₀ : LieSubmodule R L M := LieModule.genWeightSpace M 0
        M₁ : LieSubmodule R L M := LieModule.posFittingComp R L M
        x : L
        M₀ₓ : LieSubmodule R L M := LieModule.genWeightSpaceOf M 0 x
        hx : Ne M₀ₓ Top.top
        M₁ₓ : LieSubmodule R L M := LieModule.posFittingCompOf R M x
        M₀ₓ₀ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.g …
        M₀ₓ₁ : LieSubmodule R L (Subtype fun x => Membership.mem M₀ₓ x) := LieModule.p …
        h₁ : IsCompl M₀ₓ M₁ₓ
        h₂ : IsCompl M₀ₓ₀ M₀ₓ₁
        h₃ : Eq (LieSubmodule.map M₀ₓ.incl M₀ₓ₀) M₀
        h₄ : Eq (Max.max (LieSubmodule.map M₀ₓ.incl M₀ₓ₁) M₁ₓ) M₁
        ⊢ IsCompl (Max.max (LieSubmodule.map M₀ₓ.incl M₀ₓ₀) (LieSubmodule.map M₀ₓ.incl …
      -/
    · rwa [← LieSubmodule.map_sup, h₂.sup_eq_top, LieModuleHom.map_top, LieSubmodule.range_incl]
      /-
        🎉 no goals
      -/


/-- This is the Fitting decomposition of the Lie module `M`. -/
lemma isCompl_genWeightSpace_zero_posFittingComp :
    IsCompl (genWeightSpace M 0) (posFittingComp R L M) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    ⊢ IsCompl (LieModule.genWeightSpace M 0) (LieModule.posFittingComp R L M)
  -/
  let P : LieSubmodule R L M → Prop := fun N ↦ IsCompl (genWeightSpace N 0) (posFittingComp R L N)
  suffices P ⊤ by
    let e := LieModuleEquiv.ofTop R L M
    rw [← map_genWeightSpace_eq e, ← map_posFittingComp_eq e]
    exact (LieSubmodule.orderIsoMapComap e).isCompl_iff.mp this
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    P : LieSubmodule R L M → Prop := fun N => IsCompl (LieModule.genWeightSpace (S …
    ⊢ P Top.top
  -/
  refine (LieSubmodule.wellFoundedLT_of_isArtinian R L M).induction (C := P) _ fun N hN ↦ ?_
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    P : LieSubmodule R L M → Prop := fun N => IsCompl (LieModule.genWeightSpace (S …
    N : LieSubmodule R L M
    hN : ∀ (y : LieSubmodule R L M), LT.lt y N → P y
    ⊢ P N
  -/
  refine isCompl_genWeightSpace_zero_posFittingComp_aux R L N fun N' hN' ↦ ?_
  suffices IsCompl (genWeightSpace (N'.map N.incl) 0) (posFittingComp R L (N'.map N.incl)) by
    let e := LieSubmodule.equivMapOfInjective N' N.injective_incl
    rw [← map_genWeightSpace_eq e, ← map_posFittingComp_eq e] at this
    exact (LieSubmodule.orderIsoMapComap e).isCompl_iff.mpr this
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    P : LieSubmodule R L M → Prop := fun N => IsCompl (LieModule.genWeightSpace (S …
    N : LieSubmodule R L M
    hN : ∀ (y : LieSubmodule R L M), LT.lt y N → P y
    N' : LieSubmodule R L (Subtype fun x => Membership.mem N x)
    hN' : LT.lt N' Top.top
    ⊢ IsCompl (LieModule.genWeightSpace (Subtype fun x => Membership.mem (LieSubmo …
  -/
  exact hN _ (LieSubmodule.map_incl_lt_iff_lt_top.mpr hN')
  /-
    🎉 no goals
  -/


lemma disjoint_genWeightSpaceOf [NoZeroSMulDivisors R M] {x : L} {φ₁ φ₂ : R} (h : φ₁ ≠ φ₂) :
    Disjoint (genWeightSpaceOf M φ₁ x) (genWeightSpaceOf M φ₂ x) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    x : L
    φ₁ φ₂ : R
    h : Ne φ₁ φ₂
    ⊢ Disjoint (LieModule.genWeightSpaceOf M φ₁ x) (LieModule.genWeightSpaceOf M φ …
  -/
  rw [LieSubmodule.disjoint_iff_toSubmodule]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    x : L
    φ₁ φ₂ : R
    h : Ne φ₁ φ₂
    ⊢ Disjoint ↑(LieModule.genWeightSpaceOf M φ₁ x) ↑(LieModule.genWeightSpaceOf M …
  -/
  dsimp [genWeightSpaceOf]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    x : L
    φ₁ φ₂ : R
    h : Ne φ₁ φ₂
    ⊢ Disjoint (((LieModule.toEnd R L M) x).maxGenEigenspace φ₁) (((LieModule.toEn …
  -/
  exact Module.End.disjoint_genEigenspace _ h _ _
  /-
    🎉 no goals
  -/


lemma disjoint_genWeightSpace [NoZeroSMulDivisors R M] {χ₁ χ₂ : L → R} (h : χ₁ ≠ χ₂) :
    Disjoint (genWeightSpace M χ₁) (genWeightSpace M χ₂) := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    χ₁ χ₂ : L → R
    h : Ne χ₁ χ₂
    ⊢ Disjoint (LieModule.genWeightSpace M χ₁) (LieModule.genWeightSpace M χ₂)
  -/
  obtain ⟨x, hx⟩ : ∃ x, χ₁ x ≠ χ₂ x := Function.ne_iff.mp h
  exact (disjoint_genWeightSpaceOf R L M hx).mono
    (genWeightSpace_le_genWeightSpaceOf M x χ₁) (genWeightSpace_le_genWeightSpaceOf M x χ₂)


lemma injOn_genWeightSpace [NoZeroSMulDivisors R M] :
    InjOn (fun (χ : L → R) ↦ genWeightSpace M χ) {χ | genWeightSpace M χ ≠ ⊥} := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    ⊢ Set.InjOn (fun χ => LieModule.genWeightSpace M χ) (setOf fun χ => Ne (LieMod …
  -/
  rintro χ₁ _ χ₂ hχ₂ (hχ₁₂ : genWeightSpace M χ₁ = genWeightSpace M χ₂)
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    χ₁ : L → R
    a✝ : Membership.mem (setOf fun χ => Ne (LieModule.genWeightSpace M χ) Bot.bot) …
    χ₂ : L → R
    hχ₂ : Membership.mem (setOf fun χ => Ne (LieModule.genWeightSpace M χ) Bot.bot …
    hχ₁₂ : Eq (LieModule.genWeightSpace M χ₁) (LieModule.genWeightSpace M χ₂)
    ⊢ Eq χ₁ χ₂
  -/
  contrapose! hχ₂
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    χ₁ : L → R
    a✝ : Membership.mem (setOf fun χ => Ne (LieModule.genWeightSpace M χ) Bot.bot) …
    χ₂ : L → R
    hχ₁₂ : Eq (LieModule.genWeightSpace M χ₁) (LieModule.genWeightSpace M χ₂)
    hχ₂ : Ne χ₁ χ₂
    ⊢ Not (Membership.mem (setOf fun χ => Ne (LieModule.genWeightSpace M χ) Bot.bo …
  -/
  simpa [hχ₁₂] using disjoint_genWeightSpace R L M hχ₂
  /-
    🎉 no goals
  -/


/-- Lie module weight spaces are independent.

See also `LieModule.iSupIndep_genWeightSpace'`. -/
lemma iSupIndep_genWeightSpace [NoZeroSMulDivisors R M] :
    iSupIndep fun χ : L → R ↦ genWeightSpace M χ := by
  simp only [LieSubmodule.iSupIndep_iff_toSubmodule, genWeightSpace,
    LieSubmodule.iInf_toSubmodule]
  exact Module.End.independent_iInf_maxGenEigenspace_of_forall_mapsTo (toEnd R L M)
    (fun x y φ z ↦ (genWeightSpaceOf M φ y).lie_mem)


@[deprecated (since := "2024-11-24")] alias independent_genWeightSpace := iSupIndep_genWeightSpace


lemma iSupIndep_genWeightSpace' [NoZeroSMulDivisors R M] :
    iSupIndep fun χ : Weight R L M ↦ genWeightSpace M χ :=
  (iSupIndep_genWeightSpace R L M).comp <|
    Subtype.val_injective.comp (Weight.equivSetOf R L M).injective


@[deprecated (since := "2024-11-24")] alias independent_genWeightSpace' := iSupIndep_genWeightSpace'


lemma iSupIndep_genWeightSpaceOf [NoZeroSMulDivisors R M] (x : L) :
    iSupIndep fun (χ : R) ↦ genWeightSpaceOf M χ x := by
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    x : L
    ⊢ iSupIndep fun χ => LieModule.genWeightSpaceOf M χ x
  -/
  rw [LieSubmodule.iSupIndep_iff_toSubmodule]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    x : L
    ⊢ iSupIndep fun i => ↑(LieModule.genWeightSpaceOf M i x)
  -/
  dsimp [genWeightSpaceOf]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : NoZeroSMulDivisors R M
    x : L
    ⊢ iSupIndep fun i => ((LieModule.toEnd R L M) x).maxGenEigenspace i
  -/
  exact (toEnd R L M x).independent_genEigenspace _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias independent_genWeightSpaceOf := iSupIndep_genWeightSpaceOf


lemma finite_genWeightSpaceOf_ne_bot [NoZeroSMulDivisors R M] [IsNoetherian R M] (x : L) :
    {χ : R | genWeightSpaceOf M χ x ≠ ⊥}.Finite :=
  WellFoundedGT.finite_ne_bot_of_iSupIndep (iSupIndep_genWeightSpaceOf R L M x)


lemma finite_genWeightSpace_ne_bot [NoZeroSMulDivisors R M] [IsNoetherian R M] :
    {χ : L → R | genWeightSpace M χ ≠ ⊥}.Finite :=
  WellFoundedGT.finite_ne_bot_of_iSupIndep (iSupIndep_genWeightSpace R L M)


instance Weight.instFinite [NoZeroSMulDivisors R M] [IsNoetherian R M] :
    Finite (Weight R L M) := by
  /-
    K : Type u_1
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    ⊢ Finite (LieModule.Weight R L M)
  -/
  have : Finite {χ : L → R | genWeightSpace M χ ≠ ⊥} := finite_genWeightSpace_ne_bot R L M
  /-
    K : Type u_1
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : LieRing L
    inst✝⁷ : LieAlgebra R L
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : LieRingModule L M
    inst✝³ : LieModule R L M
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    this : Finite ↑(setOf fun χ => Ne (LieModule.genWeightSpace M χ) Bot.bot)
    ⊢ Finite (LieModule.Weight R L M)
  -/
  exact Finite.of_injective (equivSetOf R L M) (equivSetOf R L M).injective
  /-
    🎉 no goals
  -/


noncomputable instance Weight.instFintype [NoZeroSMulDivisors R M] [IsNoetherian R M] :
    Fintype (Weight R L M) :=
  Fintype.ofFinite _


/-- A Lie module `M` of a Lie algebra `L` is triangularizable if the endomorphism of `M` defined by
any `x : L` is triangularizable. -/
class IsTriangularizable : Prop where
  maxGenEigenspace_eq_top : ∀ x, ⨆ φ, (toEnd R L M x).maxGenEigenspace φ = ⊤


instance (L' : LieSubalgebra R L) [IsTriangularizable R L M] : IsTriangularizable R L' M where
  maxGenEigenspace_eq_top x := IsTriangularizable.maxGenEigenspace_eq_top (x : L)


instance (I : LieIdeal R L) [IsTriangularizable R L M] : IsTriangularizable R I M where
  maxGenEigenspace_eq_top x := IsTriangularizable.maxGenEigenspace_eq_top (x : L)


instance [IsTriangularizable R L M] : IsTriangularizable R (LieModule.toEnd R L M).range M where
  maxGenEigenspace_eq_top := by
    /-
      K : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      inst✝ : LieModule.IsTriangularizable R L M
      ⊢ ∀ (x : Subtype fun x => Membership.mem (LieModule.toEnd R L M).range x), Eq  …
    -/
    rintro ⟨-, x, rfl⟩
    /-
      case mk.intro
      K : Type u_1
      R : Type u_2
      L : Type u_3
      M : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      inst✝¹ : LieAlgebra.IsNilpotent R L
      inst✝ : LieModule.IsTriangularizable R L M
      x : L
      ⊢ Eq (iSup fun φ => ((LieModule.toEnd R (Subtype fun x => Membership.mem (LieM …
    -/
    exact IsTriangularizable.maxGenEigenspace_eq_top x
    /-
      🎉 no goals
    -/


@[simp]
lemma iSup_genWeightSpaceOf_eq_top [IsTriangularizable R L M] (x : L) :
    ⨆ (φ : R), genWeightSpaceOf M φ x = ⊤ := by
  rw [← LieSubmodule.toSubmodule_inj, LieSubmodule.iSup_toSubmodule,
    LieSubmodule.top_toSubmodule]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsTriangularizable R L M
    x : L
    ⊢ Eq (iSup fun i => ↑(LieModule.genWeightSpaceOf M i x)) Top.top
  -/
  dsimp [genWeightSpaceOf]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : LieAlgebra.IsNilpotent R L
    inst✝ : LieModule.IsTriangularizable R L M
    x : L
    ⊢ Eq (iSup fun i => ((LieModule.toEnd R L M) x).maxGenEigenspace i) Top.top
  -/
  exact IsTriangularizable.maxGenEigenspace_eq_top x
  /-
    🎉 no goals
  -/


open LinearMap Module in
@[simp]
lemma trace_toEnd_genWeightSpace [IsDomain R] [IsPrincipalIdealRing R]
    [Module.Free R M] [Module.Finite R M] (χ : L → R) (x : L) :
    trace R _ (toEnd R L (genWeightSpace M χ) x) = finrank R (genWeightSpace M χ) • χ x := by
  suffices _root_.IsNilpotent ((toEnd R L (genWeightSpace M χ) x) - χ x • LinearMap.id) by
    replace this := (isNilpotent_trace_of_isNilpotent this).eq_zero
    rwa [map_sub, map_smul, trace_id, sub_eq_zero, smul_eq_mul, mul_comm,
      ← nsmul_eq_mul] at this
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    χ : L → R
    x : L
    ⊢ _root_.IsNilpotent (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Member …
  -/
  rw [← Module.algebraMap_end_eq_smul_id]
  /-
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    χ : L → R
    x : L
    ⊢ _root_.IsNilpotent (HSub.hSub ((LieModule.toEnd R L (Subtype fun x => Member …
  -/
  exact isNilpotent_toEnd_sub_algebraMap M χ x
  /-
    🎉 no goals
  -/


instance instIsTriangularizableOfIsAlgClosed [IsAlgClosed K] : IsTriangularizable K L M :=
  ⟨fun _ ↦ Module.End.iSup_maxGenEigenspace_eq_top _⟩


instance (N : LieSubmodule K L M) [IsTriangularizable K L M] : IsTriangularizable K L N := by
  /-
    K : Type u_1
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    inst✝⁹ : LieRingModule L M
    inst✝⁸ : LieModule R L M
    inst✝⁷ : LieAlgebra.IsNilpotent R L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : Module K M
    inst✝³ : LieModule K L M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : FiniteDimensional K M
    N : LieSubmodule K L M
    inst✝ : LieModule.IsTriangularizable K L M
    ⊢ LieModule.IsTriangularizable K L (Subtype fun x => Membership.mem N x)
  -/
  refine ⟨fun y ↦ ?_⟩
  /-
    K : Type u_1
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    inst✝⁹ : LieRingModule L M
    inst✝⁸ : LieModule R L M
    inst✝⁷ : LieAlgebra.IsNilpotent R L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : Module K M
    inst✝³ : LieModule K L M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : FiniteDimensional K M
    N : LieSubmodule K L M
    inst✝ : LieModule.IsTriangularizable K L M
    y : L
    ⊢ Eq (iSup fun φ => ((LieModule.toEnd K L (Subtype fun x => Membership.mem N x …
  -/
  rw [← N.toEnd_restrict_eq_toEnd y]
  /-
    K : Type u_1
    R : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁴ : CommRing R
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    inst✝⁹ : LieRingModule L M
    inst✝⁸ : LieModule R L M
    inst✝⁷ : LieAlgebra.IsNilpotent R L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : Module K M
    inst✝³ : LieModule K L M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : FiniteDimensional K M
    N : LieSubmodule K L M
    inst✝ : LieModule.IsTriangularizable K L M
    y : L
    ⊢ Eq (iSup fun φ => Module.End.maxGenEigenspace (LinearMap.restrict ((LieModul …
  -/
  exact Module.End.genEigenspace_restrict_eq_top _ (IsTriangularizable.maxGenEigenspace_eq_top y)
  /-
    🎉 no goals
  -/


/-- For a triangularizable Lie module in finite dimensions, the weight spaces span the entire space.

See also `LieModule.iSup_genWeightSpace_eq_top'`. -/
lemma iSup_genWeightSpace_eq_top [IsTriangularizable K L M] :
    ⨆ χ : L → K, genWeightSpace M χ = ⊤ := by
  simp only [← LieSubmodule.toSubmodule_inj, LieSubmodule.iSup_toSubmodule,
    LieSubmodule.iInf_toSubmodule, LieSubmodule.top_toSubmodule, genWeightSpace]
  refine Module.End.iSup_iInf_maxGenEigenspace_eq_top_of_forall_mapsTo (toEnd K L M)
    (fun x y φ z ↦ (genWeightSpaceOf M φ y).lie_mem) ?_
  /-
    K : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : Module K M
    inst✝³ : LieModule K L M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : FiniteDimensional K M
    inst✝ : LieModule.IsTriangularizable K L M
    ⊢ ∀ (i : L), Eq (iSup fun μ => ((LieModule.toEnd K L M) i).maxGenEigenspace μ) …
  -/
  apply IsTriangularizable.maxGenEigenspace_eq_top
  /-
    🎉 no goals
  -/


lemma iSup_genWeightSpace_eq_top' [IsTriangularizable K L M] :
    ⨆ χ : Weight K L M, genWeightSpace M χ = ⊤ := by
  /-
    K : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : Module K M
    inst✝³ : LieModule K L M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : FiniteDimensional K M
    inst✝ : LieModule.IsTriangularizable K L M
    ⊢ Eq (iSup fun χ => LieModule.genWeightSpace M ⇑χ) Top.top
  -/
  have := iSup_genWeightSpace_eq_top K L M
  /-
    K : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : Module K M
    inst✝³ : LieModule K L M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : FiniteDimensional K M
    inst✝ : LieModule.IsTriangularizable K L M
    this : Eq (iSup fun χ => LieModule.genWeightSpace M χ) Top.top
    ⊢ Eq (iSup fun χ => LieModule.genWeightSpace M ⇑χ) Top.top
  -/
  erw [← iSup_ne_bot_subtype, ← (Weight.equivSetOf K L M).iSup_comp] at this
  /-
    K : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁹ : LieRing L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : Module K M
    inst✝³ : LieModule K L M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : FiniteDimensional K M
    inst✝ : LieModule.IsTriangularizable K L M
    this : Eq (iSup fun x => LieModule.genWeightSpace M ↑((LieModule.Weight.equivS …
    ⊢ Eq (iSup fun χ => LieModule.genWeightSpace M ⇑χ) Top.top
  -/
  exact this
  /-
    🎉 no goals
  -/


