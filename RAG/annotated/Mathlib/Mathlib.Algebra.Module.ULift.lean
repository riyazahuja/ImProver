@[to_additive]
instance smulLeft [SMul R M] : SMul (ULift R) M :=
  ⟨fun s x => s.down • x⟩


@[to_additive (attr := simp)]
theorem smul_def [SMul R M] (s : ULift R) (x : M) : s • x = s.down • x :=
  rfl


instance isScalarTower [SMul R M] [SMul M N] [SMul R N] [IsScalarTower R M N] :
    IsScalarTower (ULift R) M N :=
  ⟨fun x y z => show (x.down • y) • z = x.down • y • z from smul_assoc _ _ _⟩


instance isScalarTower' [SMul R M] [SMul M N] [SMul R N] [IsScalarTower R M N] :
    IsScalarTower R (ULift M) N :=
  ⟨fun x y z => show (x • y.down) • z = x • y.down • z from smul_assoc _ _ _⟩


instance isScalarTower'' [SMul R M] [SMul M N] [SMul R N] [IsScalarTower R M N] :
    IsScalarTower R M (ULift N) :=
                                                                 /-
                                                                   R : Type u
                                                                   M : Type v
                                                                   N : Type w
                                                                   inst✝³ : SMul R M
                                                                   inst✝² : SMul M N
                                                                   inst✝¹ : SMul R N
                                                                   inst✝ : IsScalarTower R M N
                                                                   x : R
                                                                   y : M
                                                                   z : ULift.{u_1, w} N
                                                                   ⊢ Eq { down := HSMul.hSMul (HSMul.hSMul x y) z.down } { down := HSMul.hSMul x  …
                                                                 -/
  ⟨fun x y z => show up ((x • y) • z.down) = ⟨x • y • z.down⟩ by rw [smul_assoc]⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


instance [SMul R M] [SMul Rᵐᵒᵖ M] [IsCentralScalar R M] : IsCentralScalar R (ULift M) :=
  ⟨fun r m => congr_arg up <| op_smul_eq_smul r m.down⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO this takes way longer to elaborate than it should

@[to_additive]
instance mulAction [Monoid R] [MulAction R M] : MulAction (ULift R) M where
  smul := (· • ·)
  mul_smul _ _ := mul_smul _ _
  one_smul := one_smul _


@[to_additive]
instance mulAction' [Monoid R] [MulAction R M] : MulAction R (ULift M) where
  smul := (· • ·)
  mul_smul := fun _ _ _ => congr_arg ULift.up <| mul_smul _ _ _
  one_smul := fun _ => congr_arg ULift.up <| one_smul _ _


instance smulZeroClass [Zero M] [SMulZeroClass R M] : SMulZeroClass (ULift R) M :=
  { ULift.smulLeft with smul_zero := fun _ => smul_zero _ }


instance smulZeroClass' [Zero M] [SMulZeroClass R M] : SMulZeroClass R (ULift M) where
                    /-
                      R : Type u
                      M : Type v
                      N : Type w
                      inst✝¹ : Zero M
                      inst✝ : SMulZeroClass R M
                      c : R
                      ⊢ Eq (HSMul.hSMul c 0) 0
                    -/
  smul_zero c := by { ext; simp [smul_zero] }
                    /-
                      🎉 no goals
                    -/


instance distribSMul [AddZeroClass M] [DistribSMul R M] : DistribSMul (ULift R) M where
  smul_add _ := smul_add _


instance distribSMul' [AddZeroClass M] [DistribSMul R M] : DistribSMul R (ULift M) where
  smul_add c f g := by
    /-
      R : Type u
      M : Type v
      N : Type w
      inst✝¹ : AddZeroClass M
      inst✝ : DistribSMul R M
      c : R
      f g : ULift.{?u.8327, v} M
      ⊢ Eq (HSMul.hSMul c (HAdd.hAdd f g)) (HAdd.hAdd (HSMul.hSMul c f) (HSMul.hSMul …
    -/
    ext
    /-
      case h
      R : Type u
      M : Type v
      N : Type w
      inst✝¹ : AddZeroClass M
      inst✝ : DistribSMul R M
      c : R
      f g : ULift.{?u.8327, v} M
      ⊢ Eq (HSMul.hSMul c (HAdd.hAdd f g)).down (HAdd.hAdd (HSMul.hSMul c f) (HSMul. …
    -/
    simp [smul_add]
    /-
      🎉 no goals
    -/


instance distribMulAction [Monoid R] [AddMonoid M] [DistribMulAction R M] :
    DistribMulAction (ULift R) M :=
  { ULift.mulAction, ULift.distribSMul with }


instance distribMulAction' [Monoid R] [AddMonoid M] [DistribMulAction R M] :
    DistribMulAction R (ULift M) :=
  { ULift.mulAction', ULift.distribSMul' with }


instance mulDistribMulAction [Monoid R] [Monoid M] [MulDistribMulAction R M] :
    MulDistribMulAction (ULift R) M where
  smul_one _ := smul_one _
  smul_mul _ := smul_mul' _


instance mulDistribMulAction' [Monoid R] [Monoid M] [MulDistribMulAction R M] :
    MulDistribMulAction R (ULift M) :=
  { ULift.mulAction' with
    smul_one := fun _ => by
      /-
        R : Type u
        M : Type v
        N : Type w
        inst✝² : Monoid R
        inst✝¹ : Monoid M
        inst✝ : MulDistribMulAction R M
        x✝ : R
        ⊢ Eq (HSMul.hSMul x✝ 1) 1
      -/
      ext
      /-
        case h
        R : Type u
        M : Type v
        N : Type w
        inst✝² : Monoid R
        inst✝¹ : Monoid M
        inst✝ : MulDistribMulAction R M
        x✝ : R
        ⊢ Eq (HSMul.hSMul x✝ 1).down (ULift.down 1)
      -/
      /-
        R : Type u
        M : Type v
        N : Type w
        inst✝² : Monoid R
        inst✝¹ : Monoid M
        inst✝ : MulDistribMulAction R M
        x✝² : R
        x✝¹ x✝ : ULift.{?u.12184, v} M
        ⊢ Eq (HSMul.hSMul x✝² (HMul.hMul x✝¹ x✝)) (HMul.hMul (HSMul.hSMul x✝² x✝¹) (HS …
      -/
      simp [smul_one]
      /-
        case h
        R : Type u
        M : Type v
        N : Type w
        inst✝² : Monoid R
        inst✝¹ : Monoid M
        inst✝ : MulDistribMulAction R M
        x✝² : R
        x✝¹ x✝ : ULift.{?u.12184, v} M
        ⊢ Eq (HSMul.hSMul x✝² (HMul.hMul x✝¹ x✝)).down (HMul.hMul (HSMul.hSMul x✝² x✝¹ …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    smul_mul := fun _ _ _ => by
      ext
      simp [smul_mul'] }


instance smulWithZero [Zero R] [Zero M] [SMulWithZero R M] : SMulWithZero (ULift R) M :=
  { ULift.smulLeft with
    smul_zero := fun _ => smul_zero _
    zero_smul := zero_smul _ }


instance smulWithZero' [Zero R] [Zero M] [SMulWithZero R M] : SMulWithZero R (ULift M) where
  smul_zero _ := ULift.ext _ _ <| smul_zero _
  zero_smul _ := ULift.ext _ _ <| zero_smul _ _


instance mulActionWithZero [MonoidWithZero R] [Zero M] [MulActionWithZero R M] :
    MulActionWithZero (ULift R) M :=
  { ULift.smulWithZero with
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO there seems to be a mismatch in whether
    -- the carrier is explicit here
    one_smul := one_smul _
    mul_smul := mul_smul }


instance mulActionWithZero' [MonoidWithZero R] [Zero M] [MulActionWithZero R M] :
    MulActionWithZero R (ULift M) :=
  { ULift.smulWithZero' with
    one_smul := one_smul _
    mul_smul := mul_smul }


instance module [Semiring R] [AddCommMonoid M] [Module R M] : Module (ULift R) M :=
  { ULift.smulWithZero with
    add_smul := fun _ _ => add_smul _ _
    smul_add := smul_add
    one_smul := one_smul _
    mul_smul := mul_smul }


instance module' [Semiring R] [AddCommMonoid M] [Module R M] : Module R (ULift M) :=
  { ULift.smulWithZero' with
    add_smul := fun _ _ _ => ULift.ext _ _ <| add_smul _ _ _
    one_smul := one_smul _
    mul_smul := mul_smul
    smul_add := smul_add }


/-- The `R`-linear equivalence between `ULift M` and `M`.

This is a linear version of `AddEquiv.ulift`. -/
@[simps apply symm_apply]
def moduleEquiv [Semiring R] [AddCommMonoid M] [Module R M] : ULift.{w} M ≃ₗ[R] M where
  toFun := ULift.down
  invFun := ULift.up
  map_smul' _ _ := rfl
  __ := AddEquiv.ulift


