/-- The units of an ordered commutative monoid form an ordered commutative group. -/
@[to_additive
      "The units of an ordered commutative additive monoid form an ordered commutative
      additive group."]
instance Units.orderedCommGroup [OrderedCommMonoid α] : OrderedCommGroup αˣ :=
  { Units.instPartialOrderUnits, Units.instCommGroupUnits with
    mul_le_mul_left := fun _ _ h _ => (mul_le_mul_left' (α := α) h _) }


/-- The units of a linearly ordered commutative monoid form a linearly ordered commutative group. -/
@[to_additive "The units of a linearly ordered commutative additive monoid form a
linearly ordered commutative additive group."]
instance Units.instLinearOrderedCommGroup [LinearOrderedCommMonoid α] :
    LinearOrderedCommGroup αˣ where
  __ := Units.instLinearOrder
  __ := Units.orderedCommGroup

