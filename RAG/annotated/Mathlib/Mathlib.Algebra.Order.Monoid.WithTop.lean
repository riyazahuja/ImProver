instance orderedAddCommMonoid [OrderedAddCommMonoid α] : OrderedAddCommMonoid (WithTop α) where
  add_le_add_left _ _ := add_le_add_left


instance canonicallyOrderedAddCommMonoid [CanonicallyOrderedAddCommMonoid α] :
    CanonicallyOrderedAddCommMonoid (WithTop α) :=
  { WithTop.orderBot, WithTop.orderedAddCommMonoid, WithTop.existsAddOfLE with
    le_self_add := fun a b =>
      match a, b with
      | ⊤, ⊤ => le_rfl
      | (a : α), ⊤ => le_top
      | (a : α), (b : α) => WithTop.coe_le_coe.2 le_self_add
      | ⊤, (b : α) => le_rfl }


instance [CanonicallyLinearOrderedAddCommMonoid α] :
    CanonicallyLinearOrderedAddCommMonoid (WithTop α) :=
  { WithTop.canonicallyOrderedAddCommMonoid, WithTop.linearOrder with }


instance orderedAddCommMonoid [OrderedAddCommMonoid α] : OrderedAddCommMonoid (WithBot α) :=
  { WithBot.partialOrder, WithBot.addCommMonoid with
    add_le_add_left := fun _ _ h c => add_le_add_left h c }


instance linearOrderedAddCommMonoid [LinearOrderedAddCommMonoid α] :
    LinearOrderedAddCommMonoid (WithBot α) :=
  { WithBot.linearOrder, WithBot.orderedAddCommMonoid with }


