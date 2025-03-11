@[simp]
lemma piRightHom_one : piRightHom R S A B 1 = 1 := rfl


variable {R S A B} in
@[simp]
lemma piRightHom_mul (x y : A ⊗[R] ∀ i, B i) :
    piRightHom R S A B (x * y) = piRightHom R S A B x * piRightHom R S A B y := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommSemiring R
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    ι : Type u_4
    B : ι → Type u_5
    inst✝¹ : (i : ι) → CommSemiring (B i)
    inst✝ : (i : ι) → Algebra R (B i)
    x y : TensorProduct R A ((i : ι) → B i)
    ⊢ Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul x y)) (HMul.hMul ((TensorP …
  -/
  induction x
    /-
      case zero
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : CommSemiring A
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra S A
      inst✝² : IsScalarTower R S A
      ι : Type u_4
      B : ι → Type u_5
      inst✝¹ : (i : ι) → CommSemiring (B i)
      inst✝ : (i : ι) → Algebra R (B i)
      y : TensorProduct R A ((i : ι) → B i)
      ⊢ Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul 0 y)) (HMul.hMul ((TensorP …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case tmul
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : CommSemiring A
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra S A
      inst✝² : IsScalarTower R S A
      ι : Type u_4
      B : ι → Type u_5
      inst✝¹ : (i : ι) → CommSemiring (B i)
      inst✝ : (i : ι) → Algebra R (B i)
      y : TensorProduct R A ((i : ι) → B i)
      x✝ : A
      y✝ : (i : ι) → B i
      ⊢ Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul (TensorProduct.tmul R x✝ y …
    -/
  · induction y
      /-
        case tmul.zero
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        inst✝⁵ : CommSemiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra S A
        inst✝² : IsScalarTower R S A
        ι : Type u_4
        B : ι → Type u_5
        inst✝¹ : (i : ι) → CommSemiring (B i)
        inst✝ : (i : ι) → Algebra R (B i)
        x✝ : A
        y✝ : (i : ι) → B i
        ⊢ Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul (TensorProduct.tmul R x✝ y …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case tmul.tmul
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        inst✝⁵ : CommSemiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra S A
        inst✝² : IsScalarTower R S A
        ι : Type u_4
        B : ι → Type u_5
        inst✝¹ : (i : ι) → CommSemiring (B i)
        inst✝ : (i : ι) → Algebra R (B i)
        x✝¹ : A
        y✝¹ : (i : ι) → B i
        x✝ : A
        y✝ : (i : ι) → B i
        ⊢ Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul (TensorProduct.tmul R x✝¹  …
      -/
    · ext j
      /-
        case tmul.tmul.h
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        inst✝⁵ : CommSemiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra S A
        inst✝² : IsScalarTower R S A
        ι : Type u_4
        B : ι → Type u_5
        inst✝¹ : (i : ι) → CommSemiring (B i)
        inst✝ : (i : ι) → Algebra R (B i)
        x✝¹ : A
        y✝¹ : (i : ι) → B i
        x✝ : A
        y✝ : (i : ι) → B i
        j : ι
        ⊢ Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul (TensorProduct.tmul R x✝¹  …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case tmul.add
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        inst✝⁵ : CommSemiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra S A
        inst✝² : IsScalarTower R S A
        ι : Type u_4
        B : ι → Type u_5
        inst✝¹ : (i : ι) → CommSemiring (B i)
        inst✝ : (i : ι) → Algebra R (B i)
        x✝¹ : A
        y✝¹ : (i : ι) → B i
        x✝ y✝ : TensorProduct R A ((i : ι) → B i)
        a✝¹ : Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul (TensorProduct.tmul R  …
        a✝ : Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul (TensorProduct.tmul R x …
        ⊢ Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul (TensorProduct.tmul R x✝¹  …
      -/
    · simp_all [mul_add]
      /-
        🎉 no goals
      -/
    /-
      case add
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : CommSemiring A
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra S A
      inst✝² : IsScalarTower R S A
      ι : Type u_4
      B : ι → Type u_5
      inst✝¹ : (i : ι) → CommSemiring (B i)
      inst✝ : (i : ι) → Algebra R (B i)
      y x✝ y✝ : TensorProduct R A ((i : ι) → B i)
      a✝¹ : Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul x✝ y)) (HMul.hMul ((Te …
      a✝ : Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul y✝ y)) (HMul.hMul ((Ten …
      ⊢ Eq ((TensorProduct.piRightHom R S A B) (HMul.hMul (HAdd.hAdd x✝ y✝) y)) (HMu …
    -/
  · simp_all [add_mul]
    /-
      🎉 no goals
    -/


/-- The canonical map `A ⊗[R] (∀ i, B i) →ₐ[S] ∀ i, A ⊗[R] B i`. This is an isomorphism
if `ι` is finite (see `Algebra.TensorProduct.piRight`). -/
noncomputable def piRightHom : A ⊗[R] (∀ i, B i) →ₐ[S] ∀ i, A ⊗[R] B i :=
                                                                   /-
                                                                     R : Type u_1
                                                                     S : Type u_2
                                                                     A : Type u_3
                                                                     inst✝⁸ : CommSemiring R
                                                                     inst✝⁷ : CommSemiring S
                                                                     inst✝⁶ : Algebra R S
                                                                     inst✝⁵ : CommSemiring A
                                                                     inst✝⁴ : Algebra R A
                                                                     inst✝³ : Algebra S A
                                                                     inst✝² : IsScalarTower R S A
                                                                     ι : Type u_4
                                                                     B : ι → Type u_5
                                                                     inst✝¹ : (i : ι) → CommSemiring (B i)
                                                                     inst✝ : (i : ι) → Algebra R (B i)
                                                                     ⊢ Eq ((TensorProduct.piRightHom R S A B) 1) 1
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  AlgHom.ofLinearMap (_root_.TensorProduct.piRightHom R S A B) (by simp) (by simp)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- Tensor product of rings commutes with finite products on the right. -/
noncomputable def Algebra.TensorProduct.piRight :
    A ⊗[R] (∀ i, B i) ≃ₐ[S] ∀ i, A ⊗[R] B i :=
                                                                    /-
                                                                      R : Type u_1
                                                                      S : Type u_2
                                                                      A : Type u_3
                                                                      inst✝¹⁰ : CommSemiring R
                                                                      inst✝⁹ : CommSemiring S
                                                                      inst✝⁸ : Algebra R S
                                                                      inst✝⁷ : CommSemiring A
                                                                      inst✝⁶ : Algebra R A
                                                                      inst✝⁵ : Algebra S A
                                                                      inst✝⁴ : IsScalarTower R S A
                                                                      ι : Type u_4
                                                                      B : ι → Type u_5
                                                                      inst✝³ : (i : ι) → CommSemiring (B i)
                                                                      inst✝² : (i : ι) → Algebra R (B i)
                                                                      inst✝¹ : Fintype ι
                                                                      inst✝ : DecidableEq ι
                                                                      ⊢ Eq ((_root_.TensorProduct.piRight R S A B) 1) 1
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  AlgEquiv.ofLinearEquiv (_root_.TensorProduct.piRight R S A B) (by simp) (by simp)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
lemma Algebra.TensorProduct.piRight_tmul (x : A) (f : ∀ i, B i) :
    piRight R S A B (x ⊗ₜ f) = (fun j ↦ x ⊗ₜ f j) := rfl


