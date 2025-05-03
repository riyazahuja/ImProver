variable (R A) in
/-- The tensor product of two bilinear maps injects into bilinear maps on tensor products.

Note this is heterobasic; the bilinear map on the left can take values in a module over a
(commutative) algebra over the ring of the module in which the right bilinear map is valued. -/
def tensorDistrib :
    (BilinMap A M₁ N₁ ⊗[R] BilinMap R M₂ N₂) →ₗ[A] BilinMap A (M₁ ⊗[R] M₂) (N₁ ⊗[R] N₂) :=
  (TensorProduct.lift.equiv A (M₁ ⊗[R] M₂) (M₁ ⊗[R] M₂) (N₁ ⊗[R] N₂)).symm.toLinearMap ∘ₗ
 ((LinearMap.llcomp A _ _ _).flip
   (TensorProduct.AlgebraTensorModule.tensorTensorTensorComm R A M₁ M₂ M₁ M₂).toLinearMap)
  ∘ₗ TensorProduct.AlgebraTensorModule.homTensorHomMap R _ _ _ _ _ _
  ∘ₗ (TensorProduct.AlgebraTensorModule.congr
    (TensorProduct.lift.equiv A M₁ M₁ N₁)
    (TensorProduct.lift.equiv R _ _ _)).toLinearMap


@[simp]
theorem tensorDistrib_tmul (B₁ : BilinMap A M₁ N₁) (B₂ : BilinMap R M₂ N₂) (m₁ : M₁) (m₂ : M₂)
    (m₁' : M₁) (m₂' : M₂) :
    tensorDistrib R A (B₁ ⊗ₜ B₂) (m₁ ⊗ₜ m₂) (m₁' ⊗ₜ m₂')
      = B₁ m₁ m₁' ⊗ₜ B₂ m₂ m₂' :=
  rfl


/-- The tensor product of two bilinear forms, a shorthand for dot notation. -/
protected abbrev tmul (B₁ : BilinMap A M₁ N₁) (B₂ : BilinMap R M₂ N₂) :
    BilinMap A (M₁ ⊗[R] M₂) (N₁ ⊗[R] N₂) :=
  tensorDistrib R A (B₁ ⊗ₜ[R] B₂)


variable (A) in
/-- The base change of a bilinear map (also known as "extension of scalars"). -/
protected def baseChange (B : BilinMap R M₂ N₂) : BilinMap A (A ⊗[R] M₂) (A ⊗[R] N₂) :=
  BilinMap.tmul (R := R) (A := A) (M₁ := A) (M₂ := M₂) (LinearMap.mul A A) B


@[simp]
theorem baseChange_tmul (B₂ : BilinMap R M₂ N₂) (a : A) (m₂ : M₂)
    (a' : A) (m₂' : M₂) :
    B₂.baseChange A (a ⊗ₜ m₂) (a' ⊗ₜ m₂') = (a * a') ⊗ₜ (B₂ m₂ m₂')  :=
  rfl


variable (R A) in
/-- The tensor product of two bilinear forms injects into bilinear forms on tensor products.

Note this is heterobasic; the bilinear form on the left can take values in an (commutative) algebra
over the ring in which the right bilinear form is valued. -/
def tensorDistrib : BilinForm A M₁ ⊗[R] BilinForm R M₂ →ₗ[A] BilinForm A (M₁ ⊗[R] M₂) :=
  (AlgebraTensorModule.rid R A A).congrRight₂.toLinearMap ∘ₗ (BilinMap.tensorDistrib R A)


variable (R A) in

-- TODO: make the RHS `MulOpposite.op (B₂ m₂ m₂') • B₁ m₁ m₁'` so that this has a nicer defeq for
-- `R = A` of `B₁ m₁ m₁' * B₂ m₂ m₂'`, as it did before the generalization in https://github.com/leanprover-community/mathlib4/pull/6306.
@[simp]
theorem tensorDistrib_tmul (B₁ : BilinForm A M₁) (B₂ : BilinForm R M₂) (m₁ : M₁) (m₂ : M₂)
    (m₁' : M₁) (m₂' : M₂) :
    tensorDistrib R A (B₁ ⊗ₜ B₂) (m₁ ⊗ₜ m₂) (m₁' ⊗ₜ m₂')
      = B₂ m₂ m₂' • B₁ m₁ m₁' :=
  rfl


/-- The tensor product of two bilinear forms, a shorthand for dot notation. -/
protected abbrev tmul (B₁ : BilinForm A M₁) (B₂ : BilinMap  R M₂ R) : BilinMap A (M₁ ⊗[R] M₂) A :=
  tensorDistrib R A (B₁ ⊗ₜ[R] B₂)


attribute [local ext] TensorProduct.ext in
/-- A tensor product of symmetric bilinear forms is symmetric. -/
lemma _root_.LinearMap.IsSymm.tmul {B₁ : BilinForm A M₁} {B₂ : BilinForm R M₂}
    (hB₁ : B₁.IsSymm) (hB₂ : B₂.IsSymm) : (B₁.tmul B₂).IsSymm := by
  /-
    R : Type uR
    A : Type uA
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁹ : CommSemiring R
    inst✝⁸ : CommSemiring A
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Algebra R A
    inst✝⁴ : Module R M₁
    inst✝³ : Module A M₁
    inst✝² : SMulCommClass R A M₁
    inst✝¹ : IsScalarTower R A M₁
    inst✝ : Module R M₂
    B₁ : LinearMap.BilinForm A M₁
    B₂ : LinearMap.BilinForm R M₂
    hB₁ : LinearMap.IsSymm B₁
    hB₂ : LinearMap.IsSymm B₂
    ⊢ LinearMap.IsSymm (B₁.tmul B₂)
  -/
  rw [LinearMap.isSymm_iff_eq_flip]
  /-
    R : Type uR
    A : Type uA
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁹ : CommSemiring R
    inst✝⁸ : CommSemiring A
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Algebra R A
    inst✝⁴ : Module R M₁
    inst✝³ : Module A M₁
    inst✝² : SMulCommClass R A M₁
    inst✝¹ : IsScalarTower R A M₁
    inst✝ : Module R M₂
    B₁ : LinearMap.BilinForm A M₁
    B₂ : LinearMap.BilinForm R M₂
    hB₁ : LinearMap.IsSymm B₁
    hB₂ : LinearMap.IsSymm B₂
    ⊢ Eq (B₁.tmul B₂) (LinearMap.flip (B₁.tmul B₂))
  -/
  ext x₁ x₂ y₁ y₂
  /-
    case a.h.h.a.h.h
    R : Type uR
    A : Type uA
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁹ : CommSemiring R
    inst✝⁸ : CommSemiring A
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Algebra R A
    inst✝⁴ : Module R M₁
    inst✝³ : Module A M₁
    inst✝² : SMulCommClass R A M₁
    inst✝¹ : IsScalarTower R A M₁
    inst✝ : Module R M₂
    B₁ : LinearMap.BilinForm A M₁
    B₂ : LinearMap.BilinForm R M₂
    hB₁ : LinearMap.IsSymm B₁
    hB₂ : LinearMap.IsSymm B₂
    x₁ : M₁
    x₂ : M₂
    y₁ : M₁
    y₂ : M₂
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
  -/
  exact congr_arg₂ (HSMul.hSMul) (hB₂ x₂ y₂) (hB₁ x₁ y₁)
  /-
    🎉 no goals
  -/


variable (A) in
/-- The base change of a bilinear form. -/
protected def baseChange (B : BilinForm R M₂) : BilinForm A (A ⊗[R] M₂) :=
  BilinForm.tmul (R := R) (A := A) (M₁ := A) (M₂ := M₂) (LinearMap.mul A A) B


@[simp]
theorem baseChange_tmul (B₂ : BilinForm R M₂) (a : A) (m₂ : M₂)
    (a' : A) (m₂' : M₂) :
    B₂.baseChange A (a ⊗ₜ m₂) (a' ⊗ₜ m₂') = (B₂ m₂ m₂') • (a * a') :=
  rfl


variable (A) in
/-- The base change of a symmetric bilinear form is symmetric. -/
lemma IsSymm.baseChange {B₂ : BilinForm R M₂} (hB₂ : B₂.IsSymm) : (B₂.baseChange A).IsSymm :=
  IsSymm.tmul mul_comm hB₂


variable (R) in
/-- `tensorDistrib` as an equivalence. -/
noncomputable def tensorDistribEquiv :
    BilinForm R M₁ ⊗[R] BilinForm R M₂ ≃ₗ[R] BilinForm R (M₁ ⊗[R] M₂) :=
  -- the same `LinearEquiv`s as from `tensorDistrib`,
  -- but with the inner linear map also as an equiv
  TensorProduct.congr (TensorProduct.lift.equiv R _ _ _) (TensorProduct.lift.equiv R _ _ _) ≪≫ₗ
  TensorProduct.dualDistribEquiv R (M₁ ⊗ M₁) (M₂ ⊗ M₂) ≪≫ₗ
  (TensorProduct.tensorTensorTensorComm R _ _ _ _).dualMap ≪≫ₗ
  (TensorProduct.lift.equiv R _ _ _).symm


@[simp]
theorem tensorDistribEquiv_tmul (B₁ : BilinForm R M₁) (B₂ : BilinForm R M₂) (m₁ : M₁) (m₂ : M₂)
    (m₁' : M₁) (m₂' : M₂) :
    tensorDistribEquiv R (M₁ := M₁) (M₂ := M₂) (B₁ ⊗ₜ[R] B₂) (m₁ ⊗ₜ m₂) (m₁' ⊗ₜ m₂')
      = B₁ m₁ m₁' * B₂ m₂ m₂' :=
  rfl


variable (R M₁ M₂) in
-- TODO: make this `rfl`
@[simp]
theorem tensorDistribEquiv_toLinearMap :
    (tensorDistribEquiv R (M₁ := M₁) (M₂ := M₂)).toLinearMap = tensorDistrib R R := by
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module R M₁
    inst✝⁴ : Module R M₂
    inst✝³ : Module.Free R M₁
    inst✝² : Module.Finite R M₁
    inst✝¹ : Module.Free R M₂
    inst✝ : Module.Finite R M₂
    ⊢ Eq (↑(LinearMap.BilinForm.tensorDistribEquiv R)) (LinearMap.BilinForm.tensor …
  -/
  ext B₁ B₂ : 3
  /-
    case a.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module R M₁
    inst✝⁴ : Module R M₂
    inst✝³ : Module.Free R M₁
    inst✝² : Module.Finite R M₁
    inst✝¹ : Module.Free R M₂
    inst✝ : Module.Finite R M₂
    B₁ : LinearMap.BilinForm R M₁
    B₂ : LinearMap.BilinForm R M₂
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry ↑(LinearMap.BilinForm.tensorDi …
  -/
  ext
  /-
    case a.h.h.a.h.h.a.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : AddCommGroup M₂
    inst✝⁵ : Module R M₁
    inst✝⁴ : Module R M₂
    inst✝³ : Module.Free R M₁
    inst✝² : Module.Finite R M₁
    inst✝¹ : Module.Free R M₂
    inst✝ : Module.Finite R M₂
    B₁ : LinearMap.BilinForm R M₁
    B₂ : LinearMap.BilinForm R M₂
    x✝³ : M₁
    x✝² : M₂
    x✝¹ : M₁
    x✝ : M₂
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
  -/
  exact mul_comm _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem tensorDistribEquiv_apply (B : BilinForm R M₁ ⊗ BilinForm R M₂) :
    tensorDistribEquiv R (M₁ := M₁) (M₂ := M₂) B = tensorDistrib R R B :=
  DFunLike.congr_fun (tensorDistribEquiv_toLinearMap R M₁ M₂) B


