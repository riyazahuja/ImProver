@[to_additive]
instance OrderDual.orderedCommGroup [OrderedCommGroup α] : OrderedCommGroup αᵒᵈ :=
  { OrderDual.orderedCommMonoid, OrderDual.instGroup with }


@[to_additive]
instance OrderDual.linearOrderedCommGroup [LinearOrderedCommGroup α] : LinearOrderedCommGroup αᵒᵈ :=
  { OrderDual.orderedCommGroup, OrderDual.instLinearOrder α with }

