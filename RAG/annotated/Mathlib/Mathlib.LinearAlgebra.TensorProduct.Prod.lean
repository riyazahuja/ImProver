/-- Tensor products distribute over a product on the right. -/
def prodRight : M₁ ⊗[R] (M₂ × M₃) ≃ₗ[R] ((M₁ ⊗[R] M₂) × (M₁ ⊗[R] M₃)) :=
  LinearEquiv.ofLinear
    (lift <|
      LinearMap.prodMapLinear R M₂ M₃ (M₁ ⊗[R] M₂) (M₁ ⊗[R] M₃) R
        ∘ₗ LinearMap.prod (mk _ _ _) (mk _ _ _))
    (LinearMap.coprod
      (LinearMap.lTensor _ <| LinearMap.inl _ _ _)
      (LinearMap.lTensor _ <| LinearMap.inr _ _ _))
        /-
          R : Type uR
          M₁ : Type uM₁
          M₂ : Type uM₂
          M₃ : Type uM₃
          inst✝⁶ : CommSemiring R
          inst✝⁵ : AddCommMonoid M₁
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : AddCommMonoid M₃
          inst✝² : Module R M₁
          inst✝¹ : Module R M₂
          inst✝ : Module R M₃
          ⊢ Eq ((TensorProduct.lift ((LinearMap.prodMapLinear R M₂ M₃ (TensorProduct R M …
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
    (by ext <;> simp)
                /-
                  🎉 no goals
                -/
        /-
          R : Type uR
          M₁ : Type uM₁
          M₂ : Type uM₂
          M₃ : Type uM₃
          inst✝⁶ : CommSemiring R
          inst✝⁵ : AddCommMonoid M₁
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : AddCommMonoid M₃
          inst✝² : Module R M₁
          inst✝¹ : Module R M₂
          inst✝ : Module R M₃
          ⊢ Eq (((LinearMap.lTensor M₁ (LinearMap.inl R M₂ M₃)).coprod (LinearMap.lTenso …
        -/
                /-
                  🎉 no goals
                -/
    (by ext <;> simp)
                /-
                  🎉 no goals
                -/


@[simp] theorem prodRight_tmul (m₁ : M₁) (m₂ : M₂) (m₃ : M₃) :
    prodRight R M₁ M₂ M₃ (m₁ ⊗ₜ (m₂, m₃)) = (m₁ ⊗ₜ m₂, m₁ ⊗ₜ m₃) :=
  rfl


@[simp] theorem prodRight_symm_tmul (m₁ : M₁) (m₂ : M₂) (m₃ : M₃) :
    (prodRight R M₁ M₂ M₃).symm (m₁ ⊗ₜ m₂, m₁ ⊗ₜ m₃) = (m₁ ⊗ₜ (m₂, m₃)) :=
  (LinearEquiv.symm_apply_eq _).mpr rfl


/-- Tensor products distribute over a product on the left . -/
def prodLeft : (M₁ × M₂) ⊗[R] M₃ ≃ₗ[R] ((M₁ ⊗[R] M₃) × (M₂ ⊗[R] M₃)) :=
  TensorProduct.comm _ _ _
    ≪≫ₗ TensorProduct.prodRight R _ _ _
    ≪≫ₗ (TensorProduct.comm R _ _).prod (TensorProduct.comm R _ _)


@[simp] theorem prodLeft_tmul (m₁ : M₁) (m₂ : M₂) (m₃ : M₃) :
    prodLeft R M₁ M₂ M₃ ((m₁, m₂) ⊗ₜ m₃) = (m₁ ⊗ₜ m₃, m₂ ⊗ₜ m₃) :=
  rfl


@[simp] theorem prodLeft_symm_tmul (m₁ : M₁) (m₂ : M₂) (m₃ : M₃) :
    (prodLeft R M₁ M₂ M₃).symm (m₁ ⊗ₜ m₃, m₂ ⊗ₜ m₃) = ((m₁, m₂) ⊗ₜ m₃) :=
  (LinearEquiv.symm_apply_eq _).mpr rfl


