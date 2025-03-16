@[to_additive]
instance orderedCommMonoid [OrderedCommMonoid α] : OrderedCommMonoid αᵒᵈ :=
  { mul_le_mul_left := fun _ _ h c => mul_le_mul_left' h c }


@[to_additive OrderDual.OrderedCancelAddCommMonoid.to_mulLeftReflectLE]
instance OrderedCancelCommMonoid.to_mulLeftReflectLE [OrderedCancelCommMonoid α] :
    MulLeftReflectLE αᵒᵈ where
    elim a b c := OrderedCancelCommMonoid.le_of_mul_le_mul_left (α := α) a c b
-- Porting note: Lean 3 to_additive name omits first namespace part


@[to_additive]
instance orderedCancelCommMonoid [OrderedCancelCommMonoid α] : OrderedCancelCommMonoid αᵒᵈ :=
  { le_of_mul_le_mul_left := fun _ _ _ : α => le_of_mul_le_mul_left' }


@[to_additive]
instance linearOrderedCancelCommMonoid [LinearOrderedCancelCommMonoid α] :
    LinearOrderedCancelCommMonoid αᵒᵈ :=
  { OrderDual.instLinearOrder α, OrderDual.orderedCancelCommMonoid with }


@[to_additive]
instance linearOrderedCommMonoid [LinearOrderedCommMonoid α] : LinearOrderedCommMonoid αᵒᵈ :=
  { OrderDual.instLinearOrder α, OrderDual.orderedCommMonoid with }


