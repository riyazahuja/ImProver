instance smulWithZero [Zero R] [Zero M] [Zero N] [SMulWithZero R M] [SMulWithZero R N] :
    SMulWithZero R (M × N) :=
  { Prod.smul with
    smul_zero := fun _ => Prod.ext (smul_zero _) (smul_zero _)
    zero_smul := fun _ => Prod.ext (zero_smul _ _) (zero_smul _ _) }


instance mulActionWithZero [MonoidWithZero R] [Zero M] [Zero N] [MulActionWithZero R M]
    [MulActionWithZero R N] : MulActionWithZero R (M × N) :=
  { Prod.mulAction with
    smul_zero := fun _ => Prod.ext (smul_zero _) (smul_zero _)
    zero_smul := fun _ => Prod.ext (zero_smul _ _) (zero_smul _ _) }


instance instModule [Semiring R] [AddCommMonoid M] [AddCommMonoid N] [Module R M] [Module R N] :
    Module R (M × N) :=
  { Prod.distribMulAction with
    add_smul := fun _ _ _ => mk.inj_iff.mpr ⟨add_smul _ _ _, add_smul _ _ _⟩
    zero_smul := fun _ => mk.inj_iff.mpr ⟨zero_smul _ _, zero_smul _ _⟩ }


