instance : BraidedCategory Cᵒᵖ where
  braiding X Y := (β_ (unop Y) (unop X)).op


@[simp] lemma unop_tensorμ {C : Type*} [Category C] [MonoidalCategory C]
    [BraidedCategory C] (X Y W Z : Cᵒᵖ) :
    (tensorμ X W Y Z).unop = tensorμ X.unop Y.unop W.unop Z.unop := by
  simp only [unop_tensorObj, tensorμ, unop_comp, unop_inv_associator, unop_whiskerLeft,
    unop_hom_associator, unop_whiskerRight, unop_hom_braiding, Category.assoc]


@[simp] lemma op_tensorμ {C : Type*} [Category C] [MonoidalCategory C]
    [BraidedCategory C] (X Y W Z : C) :
    (tensorμ X W Y Z).op = tensorμ (op X) (op Y) (op W) (op Z) := by
  simp only [op_tensorObj, tensorμ, op_comp, op_inv_associator, op_whiskerLeft, op_hom_associator,
    op_whiskerRight, op_hom_braiding, Category.assoc]


