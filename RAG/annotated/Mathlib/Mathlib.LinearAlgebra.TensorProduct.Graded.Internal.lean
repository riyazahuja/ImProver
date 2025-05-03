variable (R) in
/-- A Type synonym for `A ⊗[R] B`, but with multiplication as `TensorProduct.gradedMul`.

This has notation `𝒜 ᵍ⊗[R] ℬ`. -/
@[nolint unusedArguments]
def GradedTensorProduct
    (𝒜 : ι → Submodule R A) (ℬ : ι → Submodule R B)
    [GradedAlgebra 𝒜] [GradedAlgebra ℬ] :
    Type _ :=
  A ⊗[R] B


@[inherit_doc GradedTensorProduct]
scoped[TensorProduct] notation:100 𝒜 " ᵍ⊗[" R "] " ℬ:100 => GradedTensorProduct R 𝒜 ℬ


instance instAddCommGroupWithOne : AddCommGroupWithOne (𝒜 ᵍ⊗[R] ℬ) :=
  Algebra.TensorProduct.instAddCommGroupWithOne

instance : Module R (𝒜 ᵍ⊗[R] ℬ) := TensorProduct.leftModule


variable (R) in
/-- The casting equivalence to move between regular and graded tensor products. -/
def of : A ⊗[R] B ≃ₗ[R] 𝒜 ᵍ⊗[R] ℬ := LinearEquiv.refl _ _


@[simp]
theorem of_one : of R 𝒜 ℬ 1 = 1 := rfl


@[simp]
theorem of_symm_one : (of R 𝒜 ℬ).symm 1 = 1 := rfl


@[simp]
theorem of_symm_of (x : A ⊗[R] B) : (of R 𝒜 ℬ).symm (of R 𝒜 ℬ x) = x := rfl


@[simp]
theorem symm_of_of (x : 𝒜 ᵍ⊗[R] ℬ) : of R 𝒜 ℬ ((of R 𝒜 ℬ).symm x) = x := rfl


/-- Two linear maps from the graded tensor product agree if they agree on the underlying tensor
product. -/
@[ext]
theorem hom_ext {M} [AddCommMonoid M] [Module R M] ⦃f g : 𝒜 ᵍ⊗[R] ℬ →ₗ[R] M⦄
    (h : f ∘ₗ of R 𝒜 ℬ = (g ∘ₗ of R 𝒜 ℬ : A ⊗[R] B →ₗ[R] M)) :
    f = g :=
  h


variable (R) {𝒜 ℬ} in
/-- The graded tensor product of two elements of graded rings. -/
abbrev tmul (a : A) (b : B) : 𝒜 ᵍ⊗[R] ℬ := of R 𝒜 ℬ (a ⊗ₜ b)


@[inherit_doc]
notation:100 x " ᵍ⊗ₜ" y:100 => tmul _ x y


@[inherit_doc]
notation:100 x " ᵍ⊗ₜ[" R "] " y:100 => tmul R x y


variable (R) in
/-- An auxiliary construction to move between the graded tensor product of internally-graded objects
and the tensor product of direct sums. -/
noncomputable def auxEquiv : (𝒜 ᵍ⊗[R] ℬ) ≃ₗ[R] (⨁ i, 𝒜 i) ⊗[R] (⨁ i, ℬ i) :=
  let fA := (decomposeAlgEquiv 𝒜).toLinearEquiv
  let fB := (decomposeAlgEquiv ℬ).toLinearEquiv
  (of R 𝒜 ℬ).symm.trans (TensorProduct.congr fA fB)


theorem auxEquiv_tmul (a : A) (b : B) :
    auxEquiv R 𝒜 ℬ (a ᵍ⊗ₜ b) = decompose 𝒜 a ⊗ₜ decompose ℬ b := rfl


theorem auxEquiv_one : auxEquiv R 𝒜 ℬ 1 = 1 := by
  rw [← of_one, Algebra.TensorProduct.one_def, auxEquiv_tmul 𝒜 ℬ, DirectSum.decompose_one,
    DirectSum.decompose_one, Algebra.TensorProduct.one_def]


theorem auxEquiv_symm_one : (auxEquiv R 𝒜 ℬ).symm 1 = 1 :=
  (LinearEquiv.symm_apply_eq _).mpr (auxEquiv_one _ _).symm


/-- Auxiliary construction used to build the `Mul` instance and get distributivity of `+` and
`\smul`. -/
noncomputable def mulHom : (𝒜 ᵍ⊗[R] ℬ) →ₗ[R] (𝒜 ᵍ⊗[R] ℬ) →ₗ[R] (𝒜 ᵍ⊗[R] ℬ) := by
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    ⊢ LinearMap (RingHom.id R) (GradedTensorProduct R 𝒜 ℬ) (LinearMap (RingHom.id  …
  -/
  letI fAB1 := auxEquiv R 𝒜 ℬ
  have := ((gradedMul R (𝒜 ·) (ℬ ·)).compl₁₂ fAB1.toLinearMap fAB1.toLinearMap).compr₂
    fAB1.symm.toLinearMap
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    fAB1 : LinearEquiv (RingHom.id R) (GradedTensorProduct R 𝒜 ℬ) (TensorProduct R …
    this : LinearMap (RingHom.id R) (GradedTensorProduct R 𝒜 ℬ) (LinearMap (RingHo …
    ⊢ LinearMap (RingHom.id R) (GradedTensorProduct R 𝒜 ℬ) (LinearMap (RingHom.id  …
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem mulHom_apply (x y : 𝒜 ᵍ⊗[R] ℬ) :
    mulHom 𝒜 ℬ x y
      = (auxEquiv R 𝒜 ℬ).symm (gradedMul R (𝒜 ·) (ℬ ·) (auxEquiv R 𝒜 ℬ x) (auxEquiv R 𝒜 ℬ y)) :=
  rfl


/-- The multiplication on the graded tensor product.

See `GradedTensorProduct.coe_mul_coe` for a characterization on pure tensors. -/
instance : Mul (𝒜 ᵍ⊗[R] ℬ) where mul x y := mulHom 𝒜 ℬ x y


theorem mul_def (x y : 𝒜 ᵍ⊗[R] ℬ) : x * y = mulHom 𝒜 ℬ x y := rfl

-- Before https://github.com/leanprover-community/mathlib4/pull/8386 this was `@[simp]` but it times out when we try to apply it.

theorem auxEquiv_mul (x y : 𝒜 ᵍ⊗[R] ℬ) :
    auxEquiv R 𝒜 ℬ (x * y) = gradedMul R (𝒜 ·) (ℬ ·) (auxEquiv R 𝒜 ℬ x) (auxEquiv R 𝒜 ℬ y) :=
  LinearEquiv.eq_symm_apply _ |>.mp rfl


instance instMonoid : Monoid (𝒜 ᵍ⊗[R] ℬ) where
  mul_one x := by
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      x : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq (HMul.hMul x 1) x
    -/
    rw [mul_def, mulHom_apply, auxEquiv_one, gradedMul_one, LinearEquiv.symm_apply_apply]
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      x : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq (HMul.hMul 1 x) x
    -/
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      x y z : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq (HMul.hMul (HMul.hMul x y) z) (HMul.hMul x (HMul.hMul y z))
    -/
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      x y z : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq ((GradedTensorProduct.auxEquiv R 𝒜 ℬ).symm (((TensorProduct.gradedMul R ( …
    -/
  one_mul x := by
    /-
      🎉 no goals
    -/
    rw [mul_def, mulHom_apply, auxEquiv_one, one_gradedMul, LinearEquiv.symm_apply_apply]
  mul_assoc x y z := by
    simp_rw [mul_def, mulHom_apply, LinearEquiv.apply_symm_apply]
    rw [gradedMul_assoc]


instance instRing : Ring (𝒜 ᵍ⊗[R] ℬ) where
  __ := instAddCommGroupWithOne 𝒜 ℬ
  __ := instMonoid 𝒜 ℬ
                            /-
                              R : Type u_1
                              ι : Type u_2
                              A : Type u_3
                              B : Type u_4
                              inst✝⁹ : CommSemiring ι
                              inst✝⁸ : DecidableEq ι
                              inst✝⁷ : CommRing R
                              inst✝⁶ : Ring A
                              inst✝⁵ : Ring B
                              inst✝⁴ : Algebra R A
                              inst✝³ : Algebra R B
                              𝒜 : ι → Submodule R A
                              ℬ : ι → Submodule R B
                              inst✝² : GradedAlgebra 𝒜
                              inst✝¹ : GradedAlgebra ℬ
                              inst✝ : Module ι (Additive (Units Int))
                              x y z : GradedTensorProduct R 𝒜 ℬ
                              ⊢ Eq (HMul.hMul (HAdd.hAdd x y) z) (HAdd.hAdd (HMul.hMul x z) (HMul.hMul y z))
                            -/
                           /-
                             R : Type u_1
                             ι : Type u_2
                             A : Type u_3
                             B : Type u_4
                             inst✝⁹ : CommSemiring ι
                             inst✝⁸ : DecidableEq ι
                             inst✝⁷ : CommRing R
                             inst✝⁶ : Ring A
                             inst✝⁵ : Ring B
                             inst✝⁴ : Algebra R A
                             inst✝³ : Algebra R B
                             𝒜 : ι → Submodule R A
                             ℬ : ι → Submodule R B
                             inst✝² : GradedAlgebra 𝒜
                             inst✝¹ : GradedAlgebra ℬ
                             inst✝ : Module ι (Additive (Units Int))
                             x y z : GradedTensorProduct R 𝒜 ℬ
                             ⊢ Eq (HMul.hMul x (HAdd.hAdd y z)) (HAdd.hAdd (HMul.hMul x y) (HMul.hMul x z))
                           -/
  right_distrib x y z := by simp_rw [mul_def, LinearMap.map_add₂]
                           /-
                             🎉 no goals
                           -/
                            /-
                              🎉 no goals
                            -/
  left_distrib x y z := by simp_rw [mul_def, map_add]
                   /-
                     R : Type u_1
                     ι : Type u_2
                     A : Type u_3
                     B : Type u_4
                     inst✝⁹ : CommSemiring ι
                     inst✝⁸ : DecidableEq ι
                     inst✝⁷ : CommRing R
                     inst✝⁶ : Ring A
                     inst✝⁵ : Ring B
                     inst✝⁴ : Algebra R A
                     inst✝³ : Algebra R B
                     𝒜 : ι → Submodule R A
                     ℬ : ι → Submodule R B
                     inst✝² : GradedAlgebra 𝒜
                     inst✝¹ : GradedAlgebra ℬ
                     inst✝ : Module ι (Additive (Units Int))
                     x : GradedTensorProduct R 𝒜 ℬ
                     ⊢ Eq (HMul.hMul x 0) 0
                   -/
                   /-
                     R : Type u_1
                     ι : Type u_2
                     A : Type u_3
                     B : Type u_4
                     inst✝⁹ : CommSemiring ι
                     inst✝⁸ : DecidableEq ι
                     inst✝⁷ : CommRing R
                     inst✝⁶ : Ring A
                     inst✝⁵ : Ring B
                     inst✝⁴ : Algebra R A
                     inst✝³ : Algebra R B
                     𝒜 : ι → Submodule R A
                     ℬ : ι → Submodule R B
                     inst✝² : GradedAlgebra 𝒜
                     inst✝¹ : GradedAlgebra ℬ
                     inst✝ : Module ι (Additive (Units Int))
                     x : GradedTensorProduct R 𝒜 ℬ
                     ⊢ Eq (HMul.hMul 0 x) 0
                   -/
  mul_zero x := by simp_rw [mul_def, map_zero]
                   /-
                     🎉 no goals
                   -/
                   /-
                     🎉 no goals
                   -/
  zero_mul x := by simp_rw [mul_def, LinearMap.map_zero₂]


/-- The characterization of this multiplication on partially homogeneous elements. -/
theorem tmul_coe_mul_coe_tmul {j₁ i₂ : ι} (a₁ : A) (b₁ : ℬ j₁) (a₂ : 𝒜 i₂) (b₂ : B) :
    (a₁ ᵍ⊗ₜ[R] (b₁ : B) * (a₂ : A) ᵍ⊗ₜ[R] b₂ : 𝒜 ᵍ⊗[R] ℬ) =
      (-1 : ℤˣ)^(j₁ * i₂) • ((a₁ * a₂ : A) ᵍ⊗ₜ (b₁ * b₂ : B)) := by
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq (HMul.hMul (GradedTensorProduct.tmul R a₁ ↑b₁) (GradedTensorProduct.tmul  …
  -/
  dsimp only [mul_def, mulHom_apply, of_symm_of]
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq ((GradedTensorProduct.auxEquiv R 𝒜 ℬ).symm (((TensorProduct.gradedMul R ( …
  -/
  dsimp [auxEquiv, tmul]
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq ((GradedTensorProduct.of R 𝒜 ℬ) ((TensorProduct.congr (DirectSum.decompos …
  -/
  rw [decompose_coe, decompose_coe]
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq ((GradedTensorProduct.of R 𝒜 ℬ) ((TensorProduct.congr (DirectSum.decompos …
  -/
  simp_rw [← lof_eq_of R]
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq ((GradedTensorProduct.of R 𝒜 ℬ) ((TensorProduct.congr (DirectSum.decompos …
  -/
  rw [tmul_of_gradedMul_of_tmul]
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq ((GradedTensorProduct.of R 𝒜 ℬ) ((TensorProduct.congr (DirectSum.decompos …
  -/
  simp_rw [lof_eq_of R]
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specialize `map_smul` to `LinearEquiv.map_smul`
  rw [@Units.smul_def _ _ (_) (_), ← Int.cast_smul_eq_zsmul R, LinearEquiv.map_smul, map_smul,
    Int.cast_smul_eq_zsmul R, ← @Units.smul_def _ _ (_) (_)]
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HMul.hMul j₁ i₂)) ((GradedTensorProduct.of  …
  -/
  rw [congr_symm_tmul]
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HMul.hMul j₁ i₂)) ((GradedTensorProduct.of  …
  -/
  dsimp
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HMul.hMul j₁ i₂)) ((GradedTensorProduct.of  …
  -/
  simp_rw [decompose_symm_mul, decompose_symm_of, Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- A special case for when `b₁` has grade 0. -/
theorem tmul_zero_coe_mul_coe_tmul {i₂ : ι} (a₁ : A) (b₁ : ℬ 0) (a₂ : 𝒜 i₂) (b₂ : B) :
    (a₁ ᵍ⊗ₜ[R] (b₁ : B) * (a₂ : A) ᵍ⊗ₜ[R] b₂ : 𝒜 ᵍ⊗[R] ℬ) =
      ((a₁ * a₂ : A) ᵍ⊗ₜ (b₁ * b₂ : B)) := by
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    i₂ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ 0) x
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq (HMul.hMul (GradedTensorProduct.tmul R a₁ ↑b₁) (GradedTensorProduct.tmul  …
  -/
  rw [tmul_coe_mul_coe_tmul, zero_mul, uzpow_zero, one_smul]
  /-
    🎉 no goals
  -/


/-- A special case for when `a₂` has grade 0. -/
theorem tmul_coe_mul_zero_coe_tmul {j₁ : ι} (a₁ : A) (b₁ : ℬ j₁) (a₂ : 𝒜 0) (b₂ : B) :
    (a₁ ᵍ⊗ₜ[R] (b₁ : B) * (a₂ : A) ᵍ⊗ₜ[R] b₂ : 𝒜 ᵍ⊗[R] ℬ) =
      ((a₁ * a₂ : A) ᵍ⊗ₜ (b₁ * b₂ : B)) := by
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    a₂ : Subtype fun x => Membership.mem (𝒜 0) x
    b₂ : B
    ⊢ Eq (HMul.hMul (GradedTensorProduct.tmul R a₁ ↑b₁) (GradedTensorProduct.tmul  …
  -/
  rw [tmul_coe_mul_coe_tmul, mul_zero, uzpow_zero, one_smul]
  /-
    🎉 no goals
  -/


theorem tmul_one_mul_coe_tmul {i₂ : ι} (a₁ : A) (a₂ : 𝒜 i₂) (b₂ : B) :
    (a₁ ᵍ⊗ₜ[R] (1 : B) * (a₂ : A) ᵍ⊗ₜ[R] b₂ : 𝒜 ᵍ⊗[R] ℬ) = (a₁ * a₂ : A) ᵍ⊗ₜ (b₂ : B) := by
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    i₂ : ι
    a₁ : A
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq (HMul.hMul (GradedTensorProduct.tmul R a₁ 1) (GradedTensorProduct.tmul R  …
  -/
  convert tmul_zero_coe_mul_coe_tmul 𝒜 ℬ a₁ (@GradedMonoid.GOne.one _ (ℬ ·) _ _) a₂ b₂
  /-
    case h.e'_3.h.e'_17
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    i₂ : ι
    a₁ : A
    a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
    b₂ : B
    ⊢ Eq b₂ (HMul.hMul (↑GradedMonoid.GOne.one) b₂)
  -/
  rw [SetLike.coe_gOne, one_mul]
  /-
    🎉 no goals
  -/


theorem tmul_coe_mul_one_tmul {j₁ : ι} (a₁ : A) (b₁ : ℬ j₁) (b₂ : B) :
    (a₁ ᵍ⊗ₜ[R] (b₁ : B) * (1 : A) ᵍ⊗ₜ[R] b₂ : 𝒜 ᵍ⊗[R] ℬ) = (a₁ : A) ᵍ⊗ₜ (b₁ * b₂ : B) := by
  /-
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    b₂ : B
    ⊢ Eq (HMul.hMul (GradedTensorProduct.tmul R a₁ ↑b₁) (GradedTensorProduct.tmul  …
  -/
  convert tmul_coe_mul_zero_coe_tmul 𝒜 ℬ a₁ b₁ (@GradedMonoid.GOne.one _ (𝒜 ·) _ _) b₂
  /-
    case h.e'_3.h.e'_16
    R : Type u_1
    ι : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁹ : CommSemiring ι
    inst✝⁸ : DecidableEq ι
    inst✝⁷ : CommRing R
    inst✝⁶ : Ring A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra R B
    𝒜 : ι → Submodule R A
    ℬ : ι → Submodule R B
    inst✝² : GradedAlgebra 𝒜
    inst✝¹ : GradedAlgebra ℬ
    inst✝ : Module ι (Additive (Units Int))
    j₁ : ι
    a₁ : A
    b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
    b₂ : B
    ⊢ Eq a₁ (HMul.hMul a₁ ↑GradedMonoid.GOne.one)
  -/
  rw [SetLike.coe_gOne, mul_one]
  /-
    🎉 no goals
  -/


theorem tmul_one_mul_one_tmul (a₁ : A) (b₂ : B) :
    (a₁ ᵍ⊗ₜ[R] (1 : B) * (1 : A) ᵍ⊗ₜ[R] b₂ : 𝒜 ᵍ⊗[R] ℬ) = (a₁ : A) ᵍ⊗ₜ (b₂ : B) := by
  convert tmul_coe_mul_zero_coe_tmul 𝒜 ℬ
    a₁ (GradedMonoid.GOne.one (A := (ℬ ·))) (GradedMonoid.GOne.one (A := (𝒜 ·))) b₂
    /-
      case h.e'_3.h.e'_16
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      a₁ : A
      b₂ : B
      ⊢ Eq a₁ (HMul.hMul a₁ ↑GradedMonoid.GOne.one)
    -/
  · rw [SetLike.coe_gOne, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_17
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      a₁ : A
      b₂ : B
      ⊢ Eq b₂ (HMul.hMul (↑GradedMonoid.GOne.one) b₂)
    -/
  · rw [SetLike.coe_gOne, one_mul]
    /-
      🎉 no goals
    -/


/-- The ring morphism `A →+* A ⊗[R] B` sending `a` to `a ⊗ₜ 1`. -/
@[simps]
def includeLeftRingHom : A →+* 𝒜 ᵍ⊗[R] ℬ where
  toFun a := a ᵍ⊗ₜ 1
                  /-
                    R : Type u_1
                    ι : Type u_2
                    A : Type u_3
                    B : Type u_4
                    inst✝⁹ : CommSemiring ι
                    inst✝⁸ : DecidableEq ι
                    inst✝⁷ : CommRing R
                    inst✝⁶ : Ring A
                    inst✝⁵ : Ring B
                    inst✝⁴ : Algebra R A
                    inst✝³ : Algebra R B
                    𝒜 : ι → Submodule R A
                    ℬ : ι → Submodule R B
                    inst✝² : GradedAlgebra 𝒜
                    inst✝¹ : GradedAlgebra ℬ
                    inst✝ : Module ι (Additive (Units Int))
                    ⊢ Eq ((↑{ toFun := fun a => GradedTensorProduct.tmul R a 1, map_one' := ⋯, map …
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
                 /-
                   R : Type u_1
                   ι : Type u_2
                   A : Type u_3
                   B : Type u_4
                   inst✝⁹ : CommSemiring ι
                   inst✝⁸ : DecidableEq ι
                   inst✝⁷ : CommRing R
                   inst✝⁶ : Ring A
                   inst✝⁵ : Ring B
                   inst✝⁴ : Algebra R A
                   inst✝³ : Algebra R B
                   𝒜 : ι → Submodule R A
                   ℬ : ι → Submodule R B
                   inst✝² : GradedAlgebra 𝒜
                   inst✝¹ : GradedAlgebra ℬ
                   inst✝ : Module ι (Additive (Units Int))
                   ⊢ ∀ (x y : A), Eq ((↑{ toFun := fun a => GradedTensorProduct.tmul R a 1, map_o …
                 -/
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      a₁ a₂ : A
      ⊢ Eq ({ toFun := fun a => GradedTensorProduct.tmul R a 1, map_one' := ⋯ }.toFu …
    -/
  map_add' := by simp [tmul, TensorProduct.add_tmul]
                 /-
                   🎉 no goals
                 -/
  map_one' := rfl
  map_mul' a₁ a₂ := by
    dsimp
    classical
    rw [← DirectSum.sum_support_decompose 𝒜 a₂, Finset.mul_sum]
    simp_rw [tmul, sum_tmul, map_sum, Finset.mul_sum]
    congr
    ext i
    rw [← SetLike.coe_gOne ℬ, tmul_coe_mul_coe_tmul, zero_mul, uzpow_zero, one_smul,
      SetLike.coe_gOne, one_mul]


instance instAlgebra : Algebra R (𝒜 ᵍ⊗[R] ℬ) where
  toRingHom := (includeLeftRingHom 𝒜 ℬ).comp (algebraMap R A)
  commutes' r x := by
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      r : R
      x : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq (HMul.hMul (((GradedTensorProduct.includeLeftRingHom 𝒜 ℬ).comp (algebraMa …
    -/
    dsimp [mul_def, mulHom_apply, auxEquiv_tmul]
    simp_rw [DirectSum.decompose_algebraMap, DirectSum.decompose_one, algebraMap_gradedMul,
      gradedMul_algebraMap]
  smul_def' r x := by
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      r : R
      x : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (((GradedTensorProduct.includeLeftRingHom 𝒜  …
    -/
    dsimp [mul_def, mulHom_apply, auxEquiv_tmul]
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      r : R
      x : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq (HSMul.hSMul r x) ((GradedTensorProduct.auxEquiv R 𝒜 ℬ).symm (((TensorPro …
    -/
    simp_rw [DirectSum.decompose_algebraMap, DirectSum.decompose_one, algebraMap_gradedMul]
    -- Qualified `map_smul` to avoid a TC timeout https://github.com/leanprover-community/mathlib4/pull/8386
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      r : R
      x : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq (HSMul.hSMul r x) ((GradedTensorProduct.auxEquiv R 𝒜 ℬ).symm (HSMul.hSMul …
    -/
    erw [LinearMap.map_smul]
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      r : R
      x : GradedTensorProduct R 𝒜 ℬ
      ⊢ Eq (HSMul.hSMul r x) (HSMul.hSMul r (↑(GradedTensorProduct.auxEquiv R 𝒜 ℬ).s …
    -/
    erw [LinearEquiv.symm_apply_apply]
    /-
      🎉 no goals
    -/


lemma algebraMap_def (r : R) : algebraMap R (𝒜 ᵍ⊗[R] ℬ) r = algebraMap R A r ᵍ⊗ₜ[R] 1 := rfl


theorem tmul_algebraMap_mul_coe_tmul {i₂ : ι} (a₁ : A) (r : R) (a₂ : 𝒜 i₂) (b₂ : B) :
    (a₁ ᵍ⊗ₜ[R] algebraMap R B r * (a₂ : A) ᵍ⊗ₜ[R] b₂ : 𝒜 ᵍ⊗[R] ℬ)
      = (a₁ * a₂ : A) ᵍ⊗ₜ (algebraMap R B r * b₂ : B) :=
  tmul_zero_coe_mul_coe_tmul 𝒜 ℬ a₁ (GAlgebra.toFun (A := (ℬ ·)) r) a₂ b₂


theorem tmul_coe_mul_algebraMap_tmul {j₁ : ι} (a₁ : A) (b₁ : ℬ j₁) (r : R) (b₂ : B) :
    (a₁ ᵍ⊗ₜ[R] (b₁ : B) * algebraMap R A r ᵍ⊗ₜ[R] b₂ : 𝒜 ᵍ⊗[R] ℬ)
      = (a₁ * algebraMap R A r : A) ᵍ⊗ₜ (b₁ * b₂ : B) :=
  tmul_coe_mul_zero_coe_tmul 𝒜 ℬ a₁ b₁ (GAlgebra.toFun (A := (𝒜 ·)) r) b₂


/-- The algebra morphism `A →ₐ[R] A ⊗[R] B` sending `a` to `a ⊗ₜ 1`. -/
@[simps!]
def includeLeft : A →ₐ[R] 𝒜 ᵍ⊗[R] ℬ where
  toRingHom := includeLeftRingHom 𝒜 ℬ
  commutes' _ := rfl


/-- The algebra morphism `B →ₐ[R] A ⊗[R] B` sending `b` to `1 ⊗ₜ b`. -/
@[simps!]
def includeRight : B →ₐ[R] (𝒜 ᵍ⊗[R] ℬ) :=
  AlgHom.ofLinearMap (R := R) (A := B) (B := 𝒜 ᵍ⊗[R] ℬ)
    (f := {
       toFun := fun b => 1 ᵍ⊗ₜ b
                      /-
                        R : Type u_1
                        ι : Type u_2
                        A : Type u_3
                        B : Type u_4
                        inst✝⁹ : CommSemiring ι
                        inst✝⁸ : DecidableEq ι
                        inst✝⁷ : CommRing R
                        inst✝⁶ : Ring A
                        inst✝⁵ : Ring B
                        inst✝⁴ : Algebra R A
                        inst✝³ : Algebra R B
                        𝒜 : ι → Submodule R A
                        ℬ : ι → Submodule R B
                        inst✝² : GradedAlgebra 𝒜
                        inst✝¹ : GradedAlgebra ℬ
                        inst✝ : Module ι (Additive (Units Int))
                        ⊢ ∀ (x y : B), Eq ((fun b => GradedTensorProduct.tmul R 1 b) (HAdd.hAdd x y))  …
                      -/
       map_add' := by simp [tmul, TensorProduct.tmul_add]
                      /-
                        🎉 no goals
                      -/
                       /-
                         R : Type u_1
                         ι : Type u_2
                         A : Type u_3
                         B : Type u_4
                         inst✝⁹ : CommSemiring ι
                         inst✝⁸ : DecidableEq ι
                         inst✝⁷ : CommRing R
                         inst✝⁶ : Ring A
                         inst✝⁵ : Ring B
                         inst✝⁴ : Algebra R A
                         inst✝³ : Algebra R B
                         𝒜 : ι → Submodule R A
                         ℬ : ι → Submodule R B
                         inst✝² : GradedAlgebra 𝒜
                         inst✝¹ : GradedAlgebra ℬ
                         inst✝ : Module ι (Additive (Units Int))
                         ⊢ ∀ (m : R) (x : B), Eq ({ toFun := fun b => GradedTensorProduct.tmul R 1 b, m …
                       -/
       map_smul' := by simp [tmul, TensorProduct.tmul_smul] })
                       /-
                         🎉 no goals
                       -/
    (map_one := rfl)
    (map_mul := by
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : CommRing R
        inst✝⁶ : Ring A
        inst✝⁵ : Ring B
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝² : GradedAlgebra 𝒜
        inst✝¹ : GradedAlgebra ℬ
        inst✝ : Module ι (Additive (Units Int))
        ⊢ ∀ (x y : B), Eq ({ toFun := fun b => GradedTensorProduct.tmul R 1 b, map_add …
      -/
      rw [LinearMap.map_mul_iff]
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : CommRing R
        inst✝⁶ : Ring A
        inst✝⁵ : Ring B
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝² : GradedAlgebra 𝒜
        inst✝¹ : GradedAlgebra ℬ
        inst✝ : Module ι (Additive (Units Int))
        ⊢ Eq ((LinearMap.mul R B).compr₂ { toFun := fun b => GradedTensorProduct.tmul  …
      -/
      refine DirectSum.decompose_lhom_ext ℬ fun i₁ => ?_
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : CommRing R
        inst✝⁶ : Ring A
        inst✝⁵ : Ring B
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝² : GradedAlgebra 𝒜
        inst✝¹ : GradedAlgebra ℬ
        inst✝ : Module ι (Additive (Units Int))
        i₁ : ι
        ⊢ Eq (((LinearMap.mul R B).compr₂ { toFun := fun b => GradedTensorProduct.tmul …
      -/
      ext b₁ b₂ : 2
      /-
        case h.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : CommRing R
        inst✝⁶ : Ring A
        inst✝⁵ : Ring B
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝² : GradedAlgebra 𝒜
        inst✝¹ : GradedAlgebra ℬ
        inst✝ : Module ι (Additive (Units Int))
        i₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ i₁) x
        b₂ : B
        ⊢ Eq (((((LinearMap.mul R B).compr₂ { toFun := fun b => GradedTensorProduct.tm …
      -/
      dsimp
      /-
        case h.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁹ : CommSemiring ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : CommRing R
        inst✝⁶ : Ring A
        inst✝⁵ : Ring B
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝² : GradedAlgebra 𝒜
        inst✝¹ : GradedAlgebra ℬ
        inst✝ : Module ι (Additive (Units Int))
        i₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ i₁) x
        b₂ : B
        ⊢ Eq (GradedTensorProduct.tmul R 1 (HMul.hMul (↑b₁) b₂)) (HMul.hMul (GradedTen …
      -/
      rw [tmul_coe_mul_one_tmul])
      /-
        🎉 no goals
      -/


lemma algebraMap_def' (r : R) : algebraMap R (𝒜 ᵍ⊗[R] ℬ) r = 1 ᵍ⊗ₜ[R] algebraMap R B r :=
  (includeRight 𝒜 ℬ).commutes r |>.symm


/-- The forwards direction of the universal property; an algebra morphism out of the graded tensor
product can be assembled from maps on each component that (anti)commute on pure elements of the
corresponding graded algebras. -/
def lift (f : A →ₐ[R] C) (g : B →ₐ[R] C)
    (h_anti_commutes : ∀ ⦃i j⦄ (a : 𝒜 i) (b : ℬ j), f a * g b = (-1 : ℤˣ)^(j * i) • (g b * f a)) :
    (𝒜 ᵍ⊗[R] ℬ) →ₐ[R] C :=
  AlgHom.ofLinearMap
    (LinearMap.mul' R C
      ∘ₗ (TensorProduct.map f.toLinearMap g.toLinearMap)
      ∘ₗ ((of R 𝒜 ℬ).symm : 𝒜 ᵍ⊗[R] ℬ →ₗ[R] A ⊗[R] B))
    (by
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        ⊢ Eq (((LinearMap.mul' R C).comp ((TensorProduct.map f.toLinearMap g.toLinearM …
      -/
      dsimp [Algebra.TensorProduct.one_def]
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        ⊢ Eq (HMul.hMul (f 1) (g 1)) 1
      -/
      simp only [map_one, mul_one])
      /-
        🎉 no goals
      -/
    (by
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        ⊢ ∀ (x y : GradedTensorProduct R 𝒜 ℬ), Eq (((LinearMap.mul' R C).comp ((Tensor …
      -/
      rw [LinearMap.map_mul_iff]
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        ⊢ Eq ((LinearMap.mul R (GradedTensorProduct R 𝒜 ℬ)).compr₂ ((LinearMap.mul' R  …
      -/
      ext a₁ : 3
      /-
        case h.a.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        ⊢ Eq ((TensorProduct.AlgebraTensorModule.curry (((LinearMap.mul R (GradedTenso …
      -/
      refine DirectSum.decompose_lhom_ext ℬ fun j₁ => ?_
      /-
        case h.a.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((LinearMap.mul R (GradedTens …
      -/
      ext b₁ : 3
      /-
        case h.a.h.h.h.a
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
        ⊢ Eq (TensorProduct.AlgebraTensorModule.curry (((((TensorProduct.AlgebraTensor …
      -/
      refine DirectSum.decompose_lhom_ext 𝒜 fun i₂ => ?_
      /-
        case h.a.h.h.h.a
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
        i₂ : ι
        ⊢ Eq ((TensorProduct.AlgebraTensorModule.curry (((((TensorProduct.AlgebraTenso …
      -/
      ext a₂ b₂ : 2
      /-
        case h.a.h.h.h.a.h.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
        i₂ : ι
        a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
        b₂ : B
        ⊢ Eq ((((TensorProduct.AlgebraTensorModule.curry (((((TensorProduct.AlgebraTen …
      -/
      dsimp
      /-
        case h.a.h.h.h.a.h.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
        i₂ : ι
        a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
        b₂ : B
        ⊢ Eq ((LinearMap.mul' R C) ((TensorProduct.map f.toLinearMap g.toLinearMap) (( …
      -/
      rw [tmul_coe_mul_coe_tmul]
      /-
        case h.a.h.h.h.a.h.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
        i₂ : ι
        a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
        b₂ : B
        ⊢ Eq ((LinearMap.mul' R C) ((TensorProduct.map f.toLinearMap g.toLinearMap) (( …
      -/
      rw [@Units.smul_def _ _ (_) (_), ← Int.cast_smul_eq_zsmul R, map_smul, map_smul, map_smul]
      /-
        case h.a.h.h.h.a.h.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
        i₂ : ι
        a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
        b₂ : B
        ⊢ Eq (HSMul.hSMul (↑↑(HPow.hPow (-1) (HMul.hMul j₁ i₂))) ((LinearMap.mul' R C) …
      -/
      rw [Int.cast_smul_eq_zsmul R, ← @Units.smul_def _ _ (_) (_)]
      /-
        case h.a.h.h.h.a.h.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
        i₂ : ι
        a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
        b₂ : B
        ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HMul.hMul j₁ i₂)) ((LinearMap.mul' R C) ((T …
      -/
      rw [of_symm_of, map_tmul, LinearMap.mul'_apply]
      /-
        case h.a.h.h.h.a.h.h
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.499386
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        f : AlgHom R A C
        g : AlgHom R B C
        h_anti_commutes : ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i) x) (b …
        a₁ : A
        j₁ : ι
        b₁ : Subtype fun x => Membership.mem (ℬ j₁) x
        i₂ : ι
        a₂ : Subtype fun x => Membership.mem (𝒜 i₂) x
        b₂ : B
        ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HMul.hMul j₁ i₂)) (HMul.hMul (f.toLinearMap …
      -/
      simp_rw [AlgHom.toLinearMap_apply, map_mul]
      simp_rw [mul_assoc (f a₁), ← mul_assoc _ _ (g b₂), h_anti_commutes, mul_smul_comm,
        smul_mul_assoc, smul_smul, Int.units_mul_self, one_smul])


@[simp]
theorem lift_tmul (f : A →ₐ[R] C) (g : B →ₐ[R] C)
    (h_anti_commutes : ∀ ⦃i j⦄ (a : 𝒜 i) (b : ℬ j), f a * g b = (-1 : ℤˣ)^(j * i) • (g b * f a))
    (a : A) (b : B) :
    lift 𝒜 ℬ f g h_anti_commutes (a ᵍ⊗ₜ b) = f a * g b :=
  rfl


/-- The universal property of the graded tensor product; every algebra morphism uniquely factors
as a pair of algebra morphisms that anticommute with respect to the grading. -/
def liftEquiv :
    { fg : (A →ₐ[R] C) × (B →ₐ[R] C) //
        ∀ ⦃i j⦄ (a : 𝒜 i) (b : ℬ j), fg.1 a * fg.2 b = (-1 : ℤˣ)^(j * i) • (fg.2 b * fg.1 a)} ≃
      ((𝒜 ᵍ⊗[R] ℬ) →ₐ[R] C) where
  toFun fg := lift 𝒜 ℬ _ _ fg.prop
  invFun F := ⟨(F.comp (includeLeft 𝒜 ℬ), F.comp (includeRight 𝒜 ℬ)), fun i j a b => by
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring ι
      inst✝¹⁰ : DecidableEq ι
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : Ring B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝⁴ : GradedAlgebra 𝒜
      inst✝³ : GradedAlgebra ℬ
      inst✝² : Module ι (Additive (Units Int))
      C : Type ?u.559353
      inst✝¹ : Ring C
      inst✝ : Algebra R C
      F : AlgHom R (GradedTensorProduct R 𝒜 ℬ) C
      i j : ι
      a : Subtype fun x => Membership.mem (𝒜 i) x
      b : Subtype fun x => Membership.mem (ℬ j) x
      ⊢ Eq (HMul.hMul ({ fst := F.comp (GradedTensorProduct.includeLeft 𝒜 ℬ), snd := …
    -/
    dsimp
    rw [← map_mul, ← map_mul F, tmul_coe_mul_coe_tmul, one_mul, mul_one, AlgHom.map_smul_of_tower,
      tmul_one_mul_one_tmul, smul_smul, Int.units_mul_self, one_smul]⟩
                    /-
                      R : Type u_1
                      ι : Type u_2
                      A : Type u_3
                      B : Type u_4
                      inst✝¹¹ : CommSemiring ι
                      inst✝¹⁰ : DecidableEq ι
                      inst✝⁹ : CommRing R
                      inst✝⁸ : Ring A
                      inst✝⁷ : Ring B
                      inst✝⁶ : Algebra R A
                      inst✝⁵ : Algebra R B
                      𝒜 : ι → Submodule R A
                      ℬ : ι → Submodule R B
                      inst✝⁴ : GradedAlgebra 𝒜
                      inst✝³ : GradedAlgebra ℬ
                      inst✝² : Module ι (Additive (Units Int))
                      C : Type ?u.559353
                      inst✝¹ : Ring C
                      inst✝ : Algebra R C
                      fg : Subtype fun fg => ∀ ⦃i j : ι⦄ (a : Subtype fun x => Membership.mem (𝒜 i)  …
                      ⊢ Eq ((fun F => ⟨{ fst := F.comp (GradedTensorProduct.includeLeft 𝒜 ℬ), snd := …
                    -/
                                    /-
                                      🎉 no goals
                                    -/
  left_inv fg := by ext <;> (dsimp; simp only [map_one, mul_one, one_mul])
                                    /-
                                      🎉 no goals
                                    -/
  right_inv F := by
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring ι
      inst✝¹⁰ : DecidableEq ι
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : Ring B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝⁴ : GradedAlgebra 𝒜
      inst✝³ : GradedAlgebra ℬ
      inst✝² : Module ι (Additive (Units Int))
      C : Type ?u.559353
      inst✝¹ : Ring C
      inst✝ : Algebra R C
      F : AlgHom R (GradedTensorProduct R 𝒜 ℬ) C
      ⊢ Eq ((fun fg => GradedTensorProduct.lift 𝒜 ℬ (↑fg).1 (↑fg).2 ⋯) ((fun F => ⟨{ …
    -/
    apply AlgHom.toLinearMap_injective
    /-
      case a
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring ι
      inst✝¹⁰ : DecidableEq ι
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : Ring B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝⁴ : GradedAlgebra 𝒜
      inst✝³ : GradedAlgebra ℬ
      inst✝² : Module ι (Additive (Units Int))
      C : Type ?u.559353
      inst✝¹ : Ring C
      inst✝ : Algebra R C
      F : AlgHom R (GradedTensorProduct R 𝒜 ℬ) C
      ⊢ Eq ((fun fg => GradedTensorProduct.lift 𝒜 ℬ (↑fg).1 (↑fg).2 ⋯) ((fun F => ⟨{ …
    -/
    ext
    /-
      case a.h.a.h.h
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring ι
      inst✝¹⁰ : DecidableEq ι
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : Ring B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝⁴ : GradedAlgebra 𝒜
      inst✝³ : GradedAlgebra ℬ
      inst✝² : Module ι (Additive (Units Int))
      C : Type ?u.559353
      inst✝¹ : Ring C
      inst✝ : Algebra R C
      F : AlgHom R (GradedTensorProduct R 𝒜 ℬ) C
      x✝¹ : A
      x✝ : B
      ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((fun fg => GradedTensorProdu …
    -/
    dsimp
    /-
      case a.h.a.h.h
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommSemiring ι
      inst✝¹⁰ : DecidableEq ι
      inst✝⁹ : CommRing R
      inst✝⁸ : Ring A
      inst✝⁷ : Ring B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝⁴ : GradedAlgebra 𝒜
      inst✝³ : GradedAlgebra ℬ
      inst✝² : Module ι (Additive (Units Int))
      C : Type ?u.559353
      inst✝¹ : Ring C
      inst✝ : Algebra R C
      F : AlgHom R (GradedTensorProduct R 𝒜 ℬ) C
      x✝¹ : A
      x✝ : B
      ⊢ Eq (HMul.hMul (F (GradedTensorProduct.tmul R x✝¹ 1)) (F (GradedTensorProduct …
    -/
    rw [← map_mul, tmul_one_mul_one_tmul]
    /-
      🎉 no goals
    -/


/-- Two algebra morphism from the graded tensor product agree if their compositions with the left
and right inclusions agree. -/
@[ext]
lemma algHom_ext ⦃f g : (𝒜 ᵍ⊗[R] ℬ) →ₐ[R] C⦄
    (ha : f.comp (includeLeft 𝒜 ℬ) = g.comp (includeLeft 𝒜 ℬ))
    (hb : f.comp (includeRight 𝒜 ℬ) = g.comp (includeRight 𝒜 ℬ)) : f = g :=
  (liftEquiv 𝒜 ℬ).symm.injective <| Subtype.ext <| Prod.ext ha hb


/-- The non-trivial symmetric braiding, sending $a \otimes b$ to
$(-1)^{\deg a' \deg b} (b \otimes a)$. -/
def comm : (𝒜 ᵍ⊗[R] ℬ) ≃ₐ[R] (ℬ ᵍ⊗[R] 𝒜) :=
  AlgEquiv.ofLinearEquiv
    (auxEquiv R 𝒜 ℬ ≪≫ₗ gradedComm R _ _ ≪≫ₗ (auxEquiv R ℬ 𝒜).symm)
    (by
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.595990
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        ⊢ Eq ((((GradedTensorProduct.auxEquiv R 𝒜 ℬ).trans (TensorProduct.gradedComm R …
      -/
      dsimp
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.595990
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        ⊢ Eq ((GradedTensorProduct.auxEquiv R ℬ 𝒜).symm ((TensorProduct.gradedComm R ( …
      -/
      simp_rw [auxEquiv_one, gradedComm_one, auxEquiv_symm_one])
      /-
        🎉 no goals
      -/
    (fun x y => by
      /-
        R : Type u_1
        ι : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹¹ : CommSemiring ι
        inst✝¹⁰ : DecidableEq ι
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring A
        inst✝⁷ : Ring B
        inst✝⁶ : Algebra R A
        inst✝⁵ : Algebra R B
        𝒜 : ι → Submodule R A
        ℬ : ι → Submodule R B
        inst✝⁴ : GradedAlgebra 𝒜
        inst✝³ : GradedAlgebra ℬ
        inst✝² : Module ι (Additive (Units Int))
        C : Type ?u.595990
        inst✝¹ : Ring C
        inst✝ : Algebra R C
        x y : GradedTensorProduct R 𝒜 ℬ
        ⊢ Eq ((((GradedTensorProduct.auxEquiv R 𝒜 ℬ).trans (TensorProduct.gradedComm R …
      -/
      dsimp
      simp_rw [auxEquiv_mul, gradedComm_gradedMul, LinearEquiv.symm_apply_eq,
        ← gradedComm_gradedMul, auxEquiv_mul, LinearEquiv.apply_symm_apply, gradedComm_gradedMul])


lemma auxEquiv_comm (x : 𝒜 ᵍ⊗[R] ℬ) :
    auxEquiv R ℬ 𝒜 (comm 𝒜 ℬ x) = gradedComm R (𝒜 ·) (ℬ ·) (auxEquiv R 𝒜 ℬ x) :=
  LinearEquiv.eq_symm_apply _ |>.mp rfl


@[simp] lemma comm_coe_tmul_coe {i j : ι} (a : 𝒜 i) (b : ℬ j) :
    comm 𝒜 ℬ (a ᵍ⊗ₜ b) = (-1 : ℤˣ)^(j * i) • (b ᵍ⊗ₜ a : ℬ ᵍ⊗[R] 𝒜) :=
  (auxEquiv R ℬ 𝒜).injective <| by
    simp_rw [auxEquiv_comm, auxEquiv_tmul, decompose_coe, ← lof_eq_of R, gradedComm_of_tmul_of,
      @Units.smul_def _ _ (_) (_), ← Int.cast_smul_eq_zsmul R]
    -- Qualified `map_smul` to avoid a TC timeout https://github.com/leanprover-community/mathlib4/pull/8386
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      i j : ι
      a : Subtype fun x => Membership.mem (𝒜 i) x
      b : Subtype fun x => Membership.mem (ℬ j) x
      ⊢ Eq (HSMul.hSMul (↑↑(HPow.hPow (-1) (HMul.hMul j i))) (TensorProduct.tmul R ( …
    -/
    erw [LinearMap.map_smul, auxEquiv_tmul]
    /-
      R : Type u_1
      ι : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁹ : CommSemiring ι
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : CommRing R
      inst✝⁶ : Ring A
      inst✝⁵ : Ring B
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      𝒜 : ι → Submodule R A
      ℬ : ι → Submodule R B
      inst✝² : GradedAlgebra 𝒜
      inst✝¹ : GradedAlgebra ℬ
      inst✝ : Module ι (Additive (Units Int))
      i j : ι
      a : Subtype fun x => Membership.mem (𝒜 i) x
      b : Subtype fun x => Membership.mem (ℬ j) x
      ⊢ Eq (HSMul.hSMul (↑↑(HPow.hPow (-1) (HMul.hMul j i))) (TensorProduct.tmul R ( …
    -/
    simp_rw [decompose_coe, lof_eq_of]
    /-
      🎉 no goals
    -/


