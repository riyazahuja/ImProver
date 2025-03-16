@[to_additive]
noncomputable instance [One α] [Small α] : One (Shrink α) := (equivShrink _).symm.one


@[to_additive (attr := simp)]
lemma equivShrink_symm_one [One α] [Small α] : (equivShrink α).symm 1 = 1 :=
  (equivShrink α).symm_apply_apply 1

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [Mul α] [Small α] : Mul (Shrink α) := (equivShrink _).symm.mul


@[to_additive (attr := simp)]
lemma equivShrink_symm_mul [Mul α] [Small α] (x y : Shrink α) :
    (equivShrink α).symm (x * y) = (equivShrink α).symm x * (equivShrink α).symm y := by
  /-
    α : Type u_1
    inst✝¹ : Mul α
    inst✝ : Small.{u_2, u_1} α
    x y : Shrink.{u_2, u_1} α
    ⊢ Eq ((equivShrink α).symm (HMul.hMul x y)) (HMul.hMul ((equivShrink α).symm x …
  -/
  rw [Equiv.mul_def]
  /-
    α : Type u_1
    inst✝¹ : Mul α
    inst✝ : Small.{u_2, u_1} α
    x y : Shrink.{u_2, u_1} α
    ⊢ Eq ((equivShrink α).symm ((equivShrink α).symm.symm (HMul.hMul ((equivShrink …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma equivShrink_mul [Mul α] [Small α] (x y : α) :
    equivShrink α (x * y) = equivShrink α x * equivShrink α y := by
  /-
    α : Type u_1
    inst✝¹ : Mul α
    inst✝ : Small.{u_2, u_1} α
    x y : α
    ⊢ Eq ((equivShrink α) (HMul.hMul x y)) (HMul.hMul ((equivShrink α) x) ((equivS …
  -/
  rw [Equiv.mul_def]
  /-
    α : Type u_1
    inst✝¹ : Mul α
    inst✝ : Small.{u_2, u_1} α
    x y : α
    ⊢ Eq ((equivShrink α) (HMul.hMul x y)) ((equivShrink α).symm.symm (HMul.hMul ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma equivShrink_symm_smul {R : Type*} [SMul R α] [Small α] (r : R) (x : Shrink α) :
    (equivShrink α).symm (r • x) = r • (equivShrink α).symm x := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : SMul R α
    inst✝ : Small.{u_3, u_1} α
    r : R
    x : Shrink.{u_3, u_1} α
    ⊢ Eq ((equivShrink α).symm (HSMul.hSMul r x)) (HSMul.hSMul r ((equivShrink α). …
  -/
  rw [Equiv.smul_def]
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : SMul R α
    inst✝ : Small.{u_3, u_1} α
    r : R
    x : Shrink.{u_3, u_1} α
    ⊢ Eq ((equivShrink α).symm ((equivShrink α).symm.symm (HSMul.hSMul r ((equivSh …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma equivShrink_smul {R : Type*} [SMul R α] [Small α] (r : R) (x : α) :
    equivShrink α (r • x) = r • equivShrink α x := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : SMul R α
    inst✝ : Small.{u_3, u_1} α
    r : R
    x : α
    ⊢ Eq ((equivShrink α) (HSMul.hSMul r x)) (HSMul.hSMul r ((equivShrink α) x))
  -/
  rw [Equiv.smul_def]
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : SMul R α
    inst✝ : Small.{u_3, u_1} α
    r : R
    x : α
    ⊢ Eq ((equivShrink α) (HSMul.hSMul r x)) ((equivShrink α).symm.symm (HSMul.hSM …
  -/
  simp
  /-
    🎉 no goals
  -/

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [Div α] [Small α] : Div (Shrink α) := (equivShrink _).symm.div


@[to_additive (attr := simp)]
lemma equivShrink_symm_div [Div α] [Small α] (x y : Shrink α) :
    (equivShrink α).symm (x / y) = (equivShrink α).symm x / (equivShrink α).symm y := by
  /-
    α : Type u_1
    inst✝¹ : Div α
    inst✝ : Small.{u_2, u_1} α
    x y : Shrink.{u_2, u_1} α
    ⊢ Eq ((equivShrink α).symm (HDiv.hDiv x y)) (HDiv.hDiv ((equivShrink α).symm x …
  -/
  rw [Equiv.div_def]
  /-
    α : Type u_1
    inst✝¹ : Div α
    inst✝ : Small.{u_2, u_1} α
    x y : Shrink.{u_2, u_1} α
    ⊢ Eq ((equivShrink α).symm ((equivShrink α).symm.symm (HDiv.hDiv ((equivShrink …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma equivShrink_div [Div α] [Small α] (x y : α) :
    equivShrink α (x / y) = equivShrink α x / equivShrink α y := by
  /-
    α : Type u_1
    inst✝¹ : Div α
    inst✝ : Small.{u_2, u_1} α
    x y : α
    ⊢ Eq ((equivShrink α) (HDiv.hDiv x y)) (HDiv.hDiv ((equivShrink α) x) ((equivS …
  -/
  rw [Equiv.div_def]
  /-
    α : Type u_1
    inst✝¹ : Div α
    inst✝ : Small.{u_2, u_1} α
    x y : α
    ⊢ Eq ((equivShrink α) (HDiv.hDiv x y)) ((equivShrink α).symm.symm (HDiv.hDiv ( …
  -/
  simp
  /-
    🎉 no goals
  -/

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [Inv α] [Small α] : Inv (Shrink α) := (equivShrink _).symm.Inv


@[to_additive (attr := simp)]
lemma equivShrink_symm_inv [Inv α] [Small α] (x : Shrink α) :
    (equivShrink α).symm x⁻¹ = ((equivShrink α).symm x)⁻¹ := by
  /-
    α : Type u_1
    inst✝¹ : Inv α
    inst✝ : Small.{u_2, u_1} α
    x : Shrink.{u_2, u_1} α
    ⊢ Eq ((equivShrink α).symm (Inv.inv x)) (Inv.inv ((equivShrink α).symm x))
  -/
  rw [Equiv.inv_def]
  /-
    α : Type u_1
    inst✝¹ : Inv α
    inst✝ : Small.{u_2, u_1} α
    x : Shrink.{u_2, u_1} α
    ⊢ Eq ((equivShrink α).symm ((equivShrink α).symm.symm (Inv.inv ((equivShrink α …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma equivShrink_inv [Inv α] [Small α] (x : α) :
    equivShrink α x⁻¹ = (equivShrink α x)⁻¹ := by
  /-
    α : Type u_1
    inst✝¹ : Inv α
    inst✝ : Small.{u_2, u_1} α
    x : α
    ⊢ Eq ((equivShrink α) (Inv.inv x)) (Inv.inv ((equivShrink α) x))
  -/
  rw [Equiv.inv_def]
  /-
    α : Type u_1
    inst✝¹ : Inv α
    inst✝ : Small.{u_2, u_1} α
    x : α
    ⊢ Eq ((equivShrink α) (Inv.inv x)) ((equivShrink α).symm.symm (Inv.inv ((equiv …
  -/
  simp
  /-
    🎉 no goals
  -/

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [Semigroup α] [Small α] : Semigroup (Shrink α) :=
  (equivShrink _).symm.semigroup


instance [SemigroupWithZero α] [Small α] : SemigroupWithZero (Shrink α) :=
  (equivShrink _).symm.semigroupWithZero

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [CommSemigroup α] [Small α] : CommSemigroup (Shrink α) :=
  (equivShrink _).symm.commSemigroup


instance [MulZeroClass α] [Small α] : MulZeroClass (Shrink α) :=
  (equivShrink _).symm.mulZeroClass

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [MulOneClass α] [Small α] : MulOneClass (Shrink α) :=
  (equivShrink _).symm.mulOneClass


instance [MulZeroOneClass α] [Small α] : MulZeroOneClass (Shrink α) :=
  (equivShrink _).symm.mulZeroOneClass

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [Monoid α] [Small α] : Monoid (Shrink α) :=
  (equivShrink _).symm.monoid

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [CommMonoid α] [Small α] : CommMonoid (Shrink α) :=
  (equivShrink _).symm.commMonoid

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [Group α] [Small α] : Group (Shrink α) :=
  (equivShrink _).symm.group

-- TODO: noncomputable has to be specified explicitly. https://github.com/leanprover-community/mathlib4/issues/1074 (item 8)

@[to_additive]
noncomputable instance [CommGroup α] [Small α] : CommGroup (Shrink α) :=
  (equivShrink _).symm.commGroup

