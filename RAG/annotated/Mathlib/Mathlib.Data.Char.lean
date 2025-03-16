/--
Provides a `LinearOrder` instance on `Char`. `Char` is the type of Unicode scalar values.
-/
instance : LinearOrder Char where
  le_refl := fun _ => @le_refl ℕ _ _
  le_trans := fun _ _ _ => @le_trans ℕ _ _ _ _
  le_antisymm := fun _ _ h₁ h₂ => Char.eq_of_val_eq <| UInt32.eq_of_toBitVec_eq <|
    BitVec.le_antisymm h₁ h₂
  lt_iff_le_not_le := fun _ _ => @lt_iff_le_not_le ℕ _ _ _
  le_total := fun _ _ => @le_total ℕ _ _ _
  min := fun a b => if a ≤ b then a else b
  max := fun a b => if a ≤ b then b else a
  decidableLE := inferInstance

