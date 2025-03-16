/-- Left multiplication by a unit of a semiring as an additive automorphism. -/
@[simps! (config := { simpRhs := true })]
def mulLeft : Rˣ →* AddAut R :=
  DistribMulAction.toAddAut _ _


/-- Right multiplication by a unit of a semiring as an additive automorphism. -/
def mulRight (u : Rˣ) : AddAut R :=
  DistribMulAction.toAddAut Rᵐᵒᵖˣ R (Units.opEquiv.symm <| MulOpposite.op u)


@[simp]
theorem mulRight_apply (u : Rˣ) (x : R) : mulRight u x = x * u :=
  rfl


@[simp]
theorem mulRight_symm_apply (u : Rˣ) (x : R) : (mulRight u).symm x = x * u⁻¹ :=
  rfl


