instance canonicallyOrderedAddCommMonoid : CanonicallyOrderedAddCommMonoid PUnit where
                                        /-
                                          x✝² x✝¹ : PUnit.{?u.2 + 1}
                                          x✝ : LE.le x✝² x✝¹
                                          ⊢ Eq x✝¹ (HAdd.hAdd x✝² PUnit.unit)
                                        -/
  exists_add_of_le {_ _} _ := ⟨unit, by subsingleton⟩
                                        /-
                                          🎉 no goals
                                        -/
  add_le_add_left _ _ _ _ := trivial
  le_self_add _ _ := trivial


instance linearOrderedCancelAddCommMonoid : LinearOrderedCancelAddCommMonoid PUnit where
  __ := PUnit.instLinearOrder
  le_of_add_le_add_left _ _ _ _ := trivial
                        /-
                          ⊢ ∀ (a b : PUnit.{?u.335 + 1}), LE.le a b → ∀ (c : PUnit.{?u.335 + 1}), LE.le  …
                        -/
  add_le_add_left := by intros; rfl
                                /-
                                  🎉 no goals
                                -/


instance : LinearOrderedAddCommMonoidWithTop PUnit where
  top_add' _ := rfl


