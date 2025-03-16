instance instLinearOrderedCommRing : LinearOrderedCommRing ℚ where
  __ := Rat.linearOrder
  __ := Rat.commRing
                    /-
                      ⊢ LE.le 0 1
                    -/
  zero_le_one := by decide
                    /-
                      🎉 no goals
                    -/
  add_le_add_left := fun _ _ ab _ => Rat.add_le_add_left.2 ab
  mul_pos _ _ ha hb := (Rat.mul_nonneg ha.le hb.le).lt_of_ne' (mul_ne_zero ha.ne' hb.ne')

-- Extra instances to short-circuit type class resolution

                                     /-
                                       ⊢ LinearOrderedRing Rat
                                     -/
instance : LinearOrderedRing ℚ := by infer_instance
                                     /-
                                       🎉 no goals
                                     -/


                               /-
                                 ⊢ OrderedRing Rat
                               -/
instance : OrderedRing ℚ := by infer_instance
                               /-
                                 🎉 no goals
                               -/


                                         /-
                                           ⊢ LinearOrderedSemiring Rat
                                         -/
instance : LinearOrderedSemiring ℚ := by infer_instance
                                         /-
                                           🎉 no goals
                                         -/


                                   /-
                                     ⊢ OrderedSemiring Rat
                                   -/
instance : OrderedSemiring ℚ := by infer_instance
                                   /-
                                     🎉 no goals
                                   -/


                                             /-
                                               ⊢ LinearOrderedAddCommGroup Rat
                                             -/
instance : LinearOrderedAddCommGroup ℚ := by infer_instance
                                             /-
                                               🎉 no goals
                                             -/


                                       /-
                                         ⊢ OrderedAddCommGroup Rat
                                       -/
instance : OrderedAddCommGroup ℚ := by infer_instance
                                       /-
                                         🎉 no goals
                                       -/


                                              /-
                                                ⊢ OrderedCancelAddCommMonoid Rat
                                              -/
instance : OrderedCancelAddCommMonoid ℚ := by infer_instance
                                              /-
                                                🎉 no goals
                                              -/


                                        /-
                                          ⊢ OrderedAddCommMonoid Rat
                                        -/
instance : OrderedAddCommMonoid ℚ := by infer_instance
                                        /-
                                          🎉 no goals
                                        -/


