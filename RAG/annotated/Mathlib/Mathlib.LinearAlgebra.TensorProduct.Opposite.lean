/-- `MulOpposite` distributes over `TensorProduct`. Note this is an `S`-algebra morphism, where
`A/S/R` is a tower of algebras. -/
def opAlgEquiv : Aᵐᵒᵖ ⊗[R] Bᵐᵒᵖ ≃ₐ[S] (A ⊗[R] B)ᵐᵒᵖ :=
  letI e₁ : Aᵐᵒᵖ ⊗[R] Bᵐᵒᵖ ≃ₗ[S] (A ⊗[R] B)ᵐᵒᵖ :=
    TensorProduct.AlgebraTensorModule.congr
      (opLinearEquiv S).symm (opLinearEquiv R).symm ≪≫ₗ opLinearEquiv S
  letI e₂ : A ⊗[R] B ≃ₗ[S] (Aᵐᵒᵖ ⊗[R] Bᵐᵒᵖ)ᵐᵒᵖ :=
    TensorProduct.AlgebraTensorModule.congr (opLinearEquiv S) (opLinearEquiv R) ≪≫ₗ opLinearEquiv S
  AlgEquiv.ofAlgHom
    (algHomOfLinearMapTensorProduct e₁.toLinearMap
                                             /-
                                               R : Type u_1
                                               S : Type u_2
                                               A : Type u_3
                                               B : Type u_4
                                               inst✝⁸ : CommSemiring R
                                               inst✝⁷ : CommSemiring S
                                               inst✝⁶ : Semiring A
                                               inst✝⁵ : Semiring B
                                               inst✝⁴ : Algebra R S
                                               inst✝³ : Algebra R A
                                               inst✝² : Algebra R B
                                               inst✝¹ : Algebra S A
                                               inst✝ : IsScalarTower R S A
                                               e₁ : LinearEquiv (RingHom.id S) (TensorProduct R (MulOpposite A) (MulOpposite  …
                                               e₂ : LinearEquiv (RingHom.id S) (TensorProduct R A B) (MulOpposite (TensorProd …
                                               a₁ a₂ : MulOpposite A
                                               b₁ b₂ : MulOpposite B
                                               ⊢ Eq (MulOpposite.unop (↑e₁ (TensorProduct.tmul R (HMul.hMul a₁ a₂) (HMul.hMul …
                                             -/
      (fun a₁ a₂ b₁ b₂ => unop_injective (by with_unfolding_all rfl)) (unop_injective rfl))
                                             /-
                                               🎉 no goals
                                             -/
    (AlgHom.opComm <| algHomOfLinearMapTensorProduct e₂.toLinearMap
                                             /-
                                               R : Type u_1
                                               S : Type u_2
                                               A : Type u_3
                                               B : Type u_4
                                               inst✝⁸ : CommSemiring R
                                               inst✝⁷ : CommSemiring S
                                               inst✝⁶ : Semiring A
                                               inst✝⁵ : Semiring B
                                               inst✝⁴ : Algebra R S
                                               inst✝³ : Algebra R A
                                               inst✝² : Algebra R B
                                               inst✝¹ : Algebra S A
                                               inst✝ : IsScalarTower R S A
                                               e₁ : LinearEquiv (RingHom.id S) (TensorProduct R (MulOpposite A) (MulOpposite  …
                                               e₂ : LinearEquiv (RingHom.id S) (TensorProduct R A B) (MulOpposite (TensorProd …
                                               a₁ a₂ : A
                                               b₁ b₂ : B
                                               ⊢ Eq (MulOpposite.unop (↑e₂ (TensorProduct.tmul R (HMul.hMul a₁ a₂) (HMul.hMul …
                                             -/
      (fun a₁ a₂ b₁ b₂ => unop_injective (by with_unfolding_all rfl)) (unop_injective rfl))
                                             /-
                                               🎉 no goals
                                             -/
                                    /-
                                      R : Type u_1
                                      S : Type u_2
                                      A : Type u_3
                                      B : Type u_4
                                      inst✝⁸ : CommSemiring R
                                      inst✝⁷ : CommSemiring S
                                      inst✝⁶ : Semiring A
                                      inst✝⁵ : Semiring B
                                      inst✝⁴ : Algebra R S
                                      inst✝³ : Algebra R A
                                      inst✝² : Algebra R B
                                      inst✝¹ : Algebra S A
                                      inst✝ : IsScalarTower R S A
                                      e₁ : LinearEquiv (RingHom.id S) (TensorProduct R (MulOpposite A) (MulOpposite  …
                                      e₂ : LinearEquiv (RingHom.id S) (TensorProduct R A B) (MulOpposite (TensorProd …
                                      ⊢ Eq (AlgHom.op.symm ((Algebra.TensorProduct.algHomOfLinearMapTensorProduct ↑e …
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
    (AlgHom.op.symm.injective <| by ext <;> rfl) (by ext <;> rfl)
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem opAlgEquiv_apply (x : Aᵐᵒᵖ ⊗[R] Bᵐᵒᵖ) :
    opAlgEquiv R S A B x =
      op (_root_.TensorProduct.map
        (opLinearEquiv R).symm.toLinearMap (opLinearEquiv R).symm.toLinearMap x) :=
  rfl


theorem opAlgEquiv_symm_apply (x : (A ⊗[R] B)ᵐᵒᵖ) :
    (opAlgEquiv R S A B).symm x =
      _root_.TensorProduct.map (opLinearEquiv R).toLinearMap (opLinearEquiv R).toLinearMap x.unop :=
  rfl


@[simp]
theorem opAlgEquiv_tmul (a : Aᵐᵒᵖ) (b : Bᵐᵒᵖ) :
    opAlgEquiv R S A B (a ⊗ₜ[R] b) = op (a.unop ⊗ₜ b.unop) :=
  rfl


@[simp]
theorem opAlgEquiv_symm_tmul (a : A) (b : B) :
    (opAlgEquiv R S A B).symm (op <| a ⊗ₜ[R] b) = op a ⊗ₜ op b :=
  rfl


