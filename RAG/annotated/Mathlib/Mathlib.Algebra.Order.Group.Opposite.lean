@[to_additive] instance : Preorder αᵐᵒᵖ := Preorder.lift unop


@[to_additive (attr := simp)] lemma unop_le_unop {a b : αᵐᵒᵖ} : a.unop ≤ b.unop ↔ a ≤ b := .rfl

@[to_additive (attr := simp)] lemma op_le_op {a b : α} : op a ≤ op b ↔ a ≤ b := .rfl


@[to_additive] instance [PartialOrder α] : PartialOrder αᵐᵒᵖ := PartialOrder.lift _ unop_injective


@[to_additive] instance : OrderedCommMonoid αᵐᵒᵖ where
                                                     /-
                                                       α : Type u_1
                                                       inst✝ : OrderedCommMonoid α
                                                       a b : MulOpposite α
                                                       hab : LE.le a b
                                                       c : MulOpposite α
                                                       ⊢ LE.le (MulOpposite.unop a) (MulOpposite.unop b)
                                                     -/
  mul_le_mul_left a b hab c := mul_le_mul_right' (by simpa) c.unop
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive (attr := simp)] lemma unop_le_one {a : αᵐᵒᵖ} : unop a ≤ 1 ↔ a ≤ 1 := .rfl

@[to_additive (attr := simp)] lemma one_le_unop {a : αᵐᵒᵖ} : 1 ≤ unop a ↔ 1 ≤ a := .rfl

@[to_additive (attr := simp)] lemma op_le_one {a : α} : op a ≤ 1 ↔ a ≤ 1 := .rfl

@[to_additive (attr := simp)] lemma one_le_op {a : α} : 1 ≤ op a ↔ 1 ≤ a := .rfl


@[to_additive] instance [OrderedCommGroup α] : OrderedCommGroup αᵐᵒᵖ where
  __ := instCommGroup
  __ := instOrderedCommMonoid


instance : OrderedAddCommMonoid αᵐᵒᵖ where
                                                   /-
                                                     α : Type u_1
                                                     inst✝ : OrderedAddCommMonoid α
                                                     a b : MulOpposite α
                                                     hab : LE.le a b
                                                     c : MulOpposite α
                                                     ⊢ LE.le (MulOpposite.unop a) (MulOpposite.unop b)
                                                   -/
  add_le_add_left a b hab c := add_le_add_left (by simpa) c.unop
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp] lemma unop_nonpos {a : αᵐᵒᵖ} : unop a ≤ 0 ↔ a ≤ 0 := .rfl

@[simp] lemma unop_nonneg {a : αᵐᵒᵖ} : 0 ≤ unop a ↔ 0 ≤ a := .rfl

@[simp] lemma op_nonpos {a : α} : op a ≤ 0 ↔ a ≤ 0 := .rfl

@[simp] lemma op_nonneg {a : α} : 0 ≤ op a ↔ 0 ≤ a := .rfl


instance [OrderedAddCommGroup α] : OrderedAddCommGroup αᵐᵒᵖ where
  __ := instAddCommGroup
  __ := instOrderedAddCommMonoid


instance : OrderedCommMonoid αᵃᵒᵖ where
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : OrderedCommMonoid α
                                                      a b : AddOpposite α
                                                      hab : LE.le a b
                                                      c : AddOpposite α
                                                      ⊢ LE.le (AddOpposite.unop a) (AddOpposite.unop b)
                                                    -/
  mul_le_mul_left a b hab c := mul_le_mul_left' (by simpa) c.unop
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp] lemma unop_le_one {a : αᵃᵒᵖ} : unop a ≤ 1 ↔ a ≤ 1 := .rfl

@[simp] lemma one_le_unop {a : αᵃᵒᵖ} : 1 ≤ unop a ↔ 1 ≤ a := .rfl

@[simp] lemma op_le_one {a : α} : op a ≤ 1 ↔ a ≤ 1 := .rfl

@[simp] lemma one_le_op {a : α} : 1 ≤ op a ↔ 1 ≤ a := .rfl


instance [OrderedCommGroup α] : OrderedCommGroup αᵃᵒᵖ where
  __ := instCommGroup
  __ := instOrderedCommMonoid


