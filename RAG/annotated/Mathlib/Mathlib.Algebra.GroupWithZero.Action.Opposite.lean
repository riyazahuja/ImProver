instance instSMulZeroClass [AddMonoid α] [SMulZeroClass M α] : SMulZeroClass M αᵐᵒᵖ where
  smul_zero _ := unop_injective <| smul_zero _


instance instSMulWithZero [MonoidWithZero M] [AddMonoid α] [SMulWithZero M α] :
    SMulWithZero M αᵐᵒᵖ where
  zero_smul _ := unop_injective <| zero_smul _ _


instance instMulActionWithZero [MonoidWithZero M] [AddMonoid α] [MulActionWithZero M α] :
    MulActionWithZero M αᵐᵒᵖ where
  smul_zero _ := unop_injective <| smul_zero _
  zero_smul _ := unop_injective <| zero_smul _ _


instance instDistribMulAction [Monoid M] [AddMonoid α] [DistribMulAction M α] :
    DistribMulAction M αᵐᵒᵖ where
  smul_add _ _ _ := unop_injective <| smul_add _ _ _
  smul_zero _ := unop_injective <| smul_zero _


instance instMulDistribMulAction [Monoid M] [Monoid α] [MulDistribMulAction M α] :
    MulDistribMulAction M αᵐᵒᵖ where
  smul_mul _ _ _ := unop_injective <| smul_mul' _ _ _
  smul_one _ := unop_injective <| smul_one _


/-- `Monoid.toOppositeMulAction` is faithful on nontrivial cancellative monoids with zero. -/
instance CancelMonoidWithZero.toFaithfulSMul_opposite [CancelMonoidWithZero α]
    [Nontrivial α] : FaithfulSMul αᵐᵒᵖ α :=
  ⟨fun h => unop_injective <| mul_left_cancel₀ one_ne_zero (h 1)⟩

