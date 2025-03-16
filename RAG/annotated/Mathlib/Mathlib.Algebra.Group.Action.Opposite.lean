@[to_additive]
instance instMulAction [Monoid M] [MulAction M α] : MulAction M αᵐᵒᵖ where
  one_smul _ := unop_injective <| one_smul _ _
  mul_smul _ _ _ := unop_injective <| mul_smul _ _ _


@[to_additive]
instance instIsScalarTower [SMul M N] [SMul M α] [SMul N α] [IsScalarTower M N α] :
    IsScalarTower M N αᵐᵒᵖ where
  smul_assoc _ _ _ := unop_injective <| smul_assoc _ _ _


@[to_additive]
instance instSMulCommClass [SMul M α] [SMul N α] [SMulCommClass M N α] :
    SMulCommClass M N αᵐᵒᵖ where
  smul_comm _ _ _ := unop_injective <| smul_comm _ _ _


@[to_additive]
instance instIsCentralScalar [SMul M α] [SMul Mᵐᵒᵖ α] [IsCentralScalar M α] :
    IsCentralScalar M αᵐᵒᵖ where
  op_smul_eq_smul _ _ := unop_injective <| op_smul_eq_smul _ _


@[to_additive]
lemma op_smul_eq_op_smul_op [SMul M α] [SMul Mᵐᵒᵖ α] [IsCentralScalar M α] (r : M) (a : α) :
    op (r • a) = op r • op a := (op_smul_eq_smul r (op a)).symm


@[to_additive]
lemma unop_smul_eq_unop_smul_unop [SMul M α] [SMul Mᵐᵒᵖ α] [IsCentralScalar M α] (r : Mᵐᵒᵖ)
    (a : αᵐᵒᵖ) : unop (r • a) = unop r • unop a := (unop_smul_eq_smul r (unop a)).symm


/-- With `open scoped RightActions`, an alternative symbol for left actions, `r • m`.

In lemma names this is still called `smul`. -/
scoped notation3:74 r:75 " •> " m:74 => r • m


/-- With `open scoped RightActions`, a shorthand for right actions, `op r • m`.

In lemma names this is still called `op_smul`. -/
scoped notation3:73 m:73 " <• " r:74 => MulOpposite.op r • m


/-- With `open scoped RightActions`, an alternative symbol for left actions, `r • m`.

In lemma names this is still called `vadd`. -/
scoped notation3:74 r:75 " +ᵥ> " m:74 => r +ᵥ m


/-- With `open scoped RightActions`, a shorthand for right actions, `op r +ᵥ m`.

In lemma names this is still called `op_vadd`. -/
scoped notation3:73 m:73 " <+ᵥ " r:74 => AddOpposite.op r +ᵥ m


@[to_additive]
lemma op_smul_op_smul (b : β) (a₁ a₂ : α) : b <• a₁ <• a₂ = b <• (a₁ * a₂) := smul_smul _ _ _


@[to_additive]
lemma op_smul_mul (b : β) (a₁ a₂ : α) : b <• (a₁ * a₂) = b <• a₁ <• a₂ := mul_smul _ _ _


/-- Like `Mul.toSMul`, but multiplies on the right.

See also `Monoid.toOppositeMulAction` and `MonoidWithZero.toOppositeMulActionWithZero`. -/
@[to_additive "Like `Add.toVAdd`, but adds on the right.

  See also `AddMonoid.toOppositeAddAction`."]
instance Mul.toHasOppositeSMul [Mul α] : SMul αᵐᵒᵖ α where smul c x := x * c.unop


@[to_additive] lemma op_smul_eq_mul [Mul α] {a a' : α} : op a • a' = a' * a := rfl


@[to_additive (attr := simp)]
lemma MulOpposite.smul_eq_mul_unop [Mul α] {a : αᵐᵒᵖ} {a' : α} : a • a' = a' * a.unop := rfl


/-- The right regular action of a group on itself is transitive. -/
@[to_additive "The right regular action of an additive group on itself is transitive."]
instance MulAction.OppositeRegular.isPretransitive {G : Type*} [Group G] : IsPretransitive Gᵐᵒᵖ G :=
  ⟨fun x y => ⟨op (x⁻¹ * y), mul_inv_cancel_left _ _⟩⟩


@[to_additive]
instance Semigroup.opposite_smulCommClass [Semigroup α] : SMulCommClass αᵐᵒᵖ α α where
  smul_comm _ _ _ := mul_assoc _ _ _


@[to_additive]
instance Semigroup.opposite_smulCommClass' [Semigroup α] : SMulCommClass α αᵐᵒᵖ α :=
  SMulCommClass.symm _ _ _


@[to_additive]
instance CommSemigroup.isCentralScalar [CommSemigroup α] : IsCentralScalar α α where
  op_smul_eq_smul _ _ := mul_comm _ _


/-- Like `Monoid.toMulAction`, but multiplies on the right. -/
@[to_additive "Like `AddMonoid.toAddAction`, but adds on the right."]
instance Monoid.toOppositeMulAction [Monoid α] : MulAction αᵐᵒᵖ α where
  smul := (· • ·)
  one_smul := mul_one
  mul_smul _ _ _ := (mul_assoc _ _ _).symm


@[to_additive]
instance IsScalarTower.opposite_mid {M N} [Mul N] [SMul M N] [SMulCommClass M N N] :
    IsScalarTower M Nᵐᵒᵖ N where
  smul_assoc _ _ _ := mul_smul_comm _ _ _


@[to_additive]
instance SMulCommClass.opposite_mid {M N} [Mul N] [SMul M N] [IsScalarTower M N N] :
    SMulCommClass M Nᵐᵒᵖ N where
  smul_comm x y z := by
    /-
      M✝ : Type u_1
      N✝ : Type u_2
      α : Type u_3
      β : Type u_4
      M : Type u_5
      N : Type u_6
      inst✝² : Mul N
      inst✝¹ : SMul M N
      inst✝ : IsScalarTower M N N
      x : M
      y : MulOpposite N
      z : N
      ⊢ Eq (HSMul.hSMul x (HSMul.hSMul y z)) (HSMul.hSMul y (HSMul.hSMul x z))
    -/
    induction y using MulOpposite.rec'
    /-
      case h
      M✝ : Type u_1
      N✝ : Type u_2
      α : Type u_3
      β : Type u_4
      M : Type u_5
      N : Type u_6
      inst✝² : Mul N
      inst✝¹ : SMul M N
      inst✝ : IsScalarTower M N N
      x : M
      z X✝ : N
      ⊢ Eq (HSMul.hSMul x (HSMul.hSMul (MulOpposite.op X✝) z)) (HSMul.hSMul (MulOppo …
    -/
    simp only [smul_mul_assoc, MulOpposite.smul_eq_mul_unop]
    /-
      🎉 no goals
    -/

-- The above instance does not create an unwanted diamond, the two paths to
-- `MulAction αᵐᵒᵖ αᵐᵒᵖ` are defeq.

/-- `Monoid.toOppositeMulAction` is faithful on cancellative monoids. -/
@[to_additive "`AddMonoid.toOppositeAddAction` is faithful on cancellative monoids."]
instance LeftCancelMonoid.toFaithfulSMul_opposite [LeftCancelMonoid α] :
    FaithfulSMul αᵐᵒᵖ α where
  eq_of_smul_eq_smul h := unop_injective <| mul_left_cancel (h 1)

