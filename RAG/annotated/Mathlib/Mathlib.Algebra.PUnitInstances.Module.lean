@[to_additive]
instance smul : SMul R PUnit :=
  ⟨fun _ _ => unit⟩


@[to_additive (attr := simp)]
theorem smul_eq {R : Type*} (y : PUnit) (r : R) : r • y = unit :=
  rfl


@[to_additive]
instance : IsCentralScalar R PUnit :=
  ⟨fun _ _ => rfl⟩


@[to_additive]
instance : SMulCommClass R S PUnit :=
  ⟨fun _ _ _ => rfl⟩


@[to_additive]
instance instIsScalarTowerOfSMul [SMul R S] : IsScalarTower R S PUnit :=
  ⟨fun _ _ _ => rfl⟩


instance smulWithZero [Zero R] : SMulWithZero R PUnit where
  __ := PUnit.smul
                  /-
                    R : Type u_1
                    S : Type u_2
                    inst✝ : Zero R
                    ⊢ ∀ (a : R), Eq (HSMul.hSMul a 0) 0
                  -/
  smul_zero := by subsingleton
                  /-
                    🎉 no goals
                  -/
                  /-
                    R : Type u_1
                    S : Type u_2
                    inst✝ : Zero R
                    ⊢ ∀ (m : PUnit.{1}), Eq (HSMul.hSMul 0 m) 0
                  -/
  zero_smul := by subsingleton
                  /-
                    🎉 no goals
                  -/


instance mulAction [Monoid R] : MulAction R PUnit where
  __ := PUnit.smul
                 /-
                   R : Type u_1
                   S : Type u_2
                   inst✝ : Monoid R
                   ⊢ ∀ (b : PUnit.{?u.998 + 1}), Eq (HSMul.hSMul 1 b) b
                 -/
  one_smul := by subsingleton
                 /-
                   🎉 no goals
                 -/
                 /-
                   R : Type u_1
                   S : Type u_2
                   inst✝ : Monoid R
                   ⊢ ∀ (x y : R) (b : PUnit.{?u.998 + 1}), Eq (HSMul.hSMul (HMul.hMul x y) b) (HS …
                 -/
  mul_smul := by subsingleton
                 /-
                   🎉 no goals
                 -/


instance distribMulAction [Monoid R] : DistribMulAction R PUnit where
  __ := PUnit.mulAction
                  /-
                    R : Type u_1
                    S : Type u_2
                    inst✝ : Monoid R
                    ⊢ ∀ (a : R), Eq (HSMul.hSMul a 0) 0
                  -/
  smul_zero := by subsingleton
                  /-
                    🎉 no goals
                  -/
                 /-
                   R : Type u_1
                   S : Type u_2
                   inst✝ : Monoid R
                   ⊢ ∀ (a : R) (x y : PUnit.{?u.1148 + 1}), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (H …
                 -/
  smul_add := by subsingleton
                 /-
                   🎉 no goals
                 -/


instance mulDistribMulAction [Monoid R] : MulDistribMulAction R PUnit where
  __ := PUnit.mulAction
                 /-
                   R : Type u_1
                   S : Type u_2
                   inst✝ : Monoid R
                   ⊢ ∀ (r : R) (x y : PUnit.{?u.1352 + 1}), Eq (HSMul.hSMul r (HMul.hMul x y)) (H …
                 -/
  smul_mul := by subsingleton
                 /-
                   🎉 no goals
                 -/
                 /-
                   R : Type u_1
                   S : Type u_2
                   inst✝ : Monoid R
                   ⊢ ∀ (r : R), Eq (HSMul.hSMul r 1) 1
                 -/
  smul_one := by subsingleton
                 /-
                   🎉 no goals
                 -/


instance mulSemiringAction [Semiring R] : MulSemiringAction R PUnit :=
  { PUnit.distribMulAction, PUnit.mulDistribMulAction with }


instance mulActionWithZero [MonoidWithZero R] : MulActionWithZero R PUnit :=
  { PUnit.mulAction, PUnit.smulWithZero with }


instance module [Semiring R] : Module R PUnit where
  __ := PUnit.distribMulAction
                 /-
                   R : Type u_1
                   S : Type u_2
                   inst✝ : Semiring R
                   ⊢ ∀ (r s : R) (x : PUnit.{?u.2310 + 1}), Eq (HSMul.hSMul (HAdd.hAdd r s) x) (H …
                 -/
  add_smul := by subsingleton
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type u_1
                    S : Type u_2
                    inst✝ : Semiring R
                    ⊢ ∀ (x : PUnit.{?u.2310 + 1}), Eq (HSMul.hSMul 0 x) 0
                  -/
  zero_smul := by subsingleton
                  /-
                    🎉 no goals
                  -/


@[to_additive]
instance : SMul PUnit R where smul _ x := x


/-- The one-element type acts trivially on every element. -/
@[to_additive (attr := simp)]
lemma smul_eq' (r : PUnit) (a : R) : r • a = a := rfl


                                                                    /-
                                                                      R : Type u_1
                                                                      S : Type u_2
                                                                      inst✝ : SMul R S
                                                                      ⊢ ∀ (m : PUnit.{u_3 + 1}) (n : R) (a : S), Eq (HSMul.hSMul m (HSMul.hSMul n a) …
                                                                    -/
@[to_additive] instance [SMul R S] : SMulCommClass PUnit R S := ⟨by simp⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

                                                     /-
                                                       R : Type u_1
                                                       S : Type u_2
                                                       inst✝ : SMul R S
                                                       ⊢ ∀ (x : PUnit.{u_3 + 1}) (y : R) (z : S), Eq (HSMul.hSMul (HSMul.hSMul x y) z …
                                                     -/
instance [SMul R S] : IsScalarTower PUnit R S := ⟨by simp⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


instance : MulAction PUnit R where
  __ := inferInstanceAs (SMul PUnit R)
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


instance [Zero R] : SMulZeroClass PUnit R where
  __ := inferInstanceAs (SMul PUnit R)
  smul_zero _ := rfl


instance [AddMonoid R] : DistribMulAction PUnit R where
  __ := inferInstanceAs (MulAction PUnit R)
  __ := inferInstanceAs (SMulZeroClass PUnit R)
  smul_add _ _ _ := rfl


