instance instDistribSMul [AddZeroClass A] [AddCommMonoid B] [DistribSMul M B] :
    DistribSMul M (A →+ B) where
  smul_add _ _ _ := ext fun _ => smul_add _ _ _


instance instDistribMulAction : DistribMulAction R (A →+ B) where
  smul_zero := smul_zero
  smul_add := smul_add
  one_smul _ := ext fun _ => one_smul _ _
  mul_smul _ _ _ := ext fun _ => mul_smul _ _ _


@[simp] theorem coe_smul (r : R) (f : A →+ B) : ⇑(r • f) = r • ⇑f := rfl


theorem smul_apply (r : R) (f : A →+ B) (x : A) : (r • f) x = r • f x :=
  rfl


instance smulCommClass [SMulCommClass R S B] : SMulCommClass R S (A →+ B) :=
  ⟨fun _ _ _ => ext fun _ => smul_comm _ _ _⟩


instance isScalarTower [SMul R S] [IsScalarTower R S B] : IsScalarTower R S (A →+ B) :=
  ⟨fun _ _ _ => ext fun _ => smul_assoc _ _ _⟩


instance isCentralScalar [DistribMulAction Rᵐᵒᵖ B] [IsCentralScalar R B] :
    IsCentralScalar R (A →+ B) :=
  ⟨fun _ _ => ext fun _ => op_smul_eq_smul _ _⟩


instance instModule [Semiring R] [AddMonoid A] [AddCommMonoid B] [Module R B] : Module R (A →+ B) :=
  { add_smul := fun _ _ _=> ext fun _ => add_smul _ _ _
    zero_smul := fun _ => ext fun _ => zero_smul _ _ }


instance instDomMulActModule
    {S M M₂ : Type*} [Semiring S] [AddCommMonoid M] [AddCommMonoid M₂] [Module S M] :
    Module Sᵈᵐᵃ (M →+ M₂) where
  add_smul s s' f := AddMonoidHom.ext fun m ↦ by
    /-
      R : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      A : Type u_4
      B : Type u_5
      S : Type u_6
      M : Type u_7
      M₂ : Type u_8
      inst✝³ : Semiring S
      inst✝² : AddCommMonoid M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module S M
      s s' : DomMulAct S
      f : AddMonoidHom M M₂
      m : M
      ⊢ Eq ((HSMul.hSMul (HAdd.hAdd s s') f) m) ((HAdd.hAdd (HSMul.hSMul s f) (HSMul …
    -/
    simp_rw [AddMonoidHom.add_apply, DomMulAct.smul_addMonoidHom_apply, ← map_add, ← add_smul]; rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
  zero_smul _ := AddMonoidHom.ext fun _ ↦ by
    /-
      R : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      A : Type u_4
      B : Type u_5
      S : Type u_6
      M : Type u_7
      M₂ : Type u_8
      inst✝³ : Semiring S
      inst✝² : AddCommMonoid M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module S M
      x✝¹ : AddMonoidHom M M₂
      x✝ : M
      ⊢ Eq ((HSMul.hSMul 0 x✝¹) x✝) (0 x✝)
    -/
    erw [DomMulAct.smul_addMonoidHom_apply, zero_smul, map_zero]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance instDistribSMul [DistribSMul M A] : DistribSMul M (AddMonoid.End A) :=
  AddMonoidHom.instDistribSMul


instance instDistribMulAction : DistribMulAction R (AddMonoid.End A) :=
  AddMonoidHom.instDistribMulAction


@[simp] theorem coe_smul (r : R) (f : AddMonoid.End A) : ⇑(r • f) = r • ⇑f := rfl


theorem smul_apply (r : R) (f : AddMonoid.End A) (x : A) : (r • f) x = r • f x :=
  rfl


instance smulCommClass [SMulCommClass R S A] : SMulCommClass R S (AddMonoid.End A) :=
  AddMonoidHom.smulCommClass


instance isScalarTower [SMul R S] [IsScalarTower R S A] : IsScalarTower R S (AddMonoid.End A) :=
  AddMonoidHom.isScalarTower


instance isCentralScalar [DistribMulAction Rᵐᵒᵖ A] [IsCentralScalar R A] :
    IsCentralScalar R (AddMonoid.End A) :=
  AddMonoidHom.isCentralScalar


instance instModule [Semiring R] [AddCommMonoid A] [Module R A] : Module R (AddMonoid.End A) :=
  AddMonoidHom.instModule


/-- The tautological action by `AddMonoid.End α` on `α`.

This generalizes `AddMonoid.End.applyDistribMulAction`. -/
instance applyModule [AddCommMonoid A] : Module (AddMonoid.End A) A where
  add_smul _ _ _ := rfl
  zero_smul _ := rfl


/-- Scalar multiplication on the left as an additive monoid homomorphism. -/
@[simps! (config := .asFn)]
protected def smulLeft [Monoid M] [AddMonoid A] [DistribMulAction M A] (c : M) : A →+ A :=
  DistribMulAction.toAddMonoidHom _ c


/-- Scalar multiplication as a biadditive monoid homomorphism. We need `M` to be commutative
to have addition on `M →+ M`. -/
protected def smul [Semiring R] [AddCommMonoid M] [Module R M] : R →+ M →+ M :=
  (Module.toAddMonoidEnd R M).toAddMonoidHom


@[simp] theorem coe_smul' [Semiring R] [AddCommMonoid M] [Module R M] :
    ⇑(.smul : R →+ M →+ M) = AddMonoidHom.smulLeft := rfl


