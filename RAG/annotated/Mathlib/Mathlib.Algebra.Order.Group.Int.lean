instance instLinearOrderedAddCommGroup : LinearOrderedAddCommGroup ℤ where
  __ := instLinearOrder
  __ := instAddCommGroup
  add_le_add_left _ _ := Int.add_le_add_left


