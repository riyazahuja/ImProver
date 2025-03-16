                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝² : CanonicallyLinearOrderedSemifield α
                                                                   inst✝¹ : Sub α
                                                                   inst✝ : OrderedSub α
                                                                   a b c : α
                                                                   ⊢ Eq (HDiv.hDiv (HSub.hSub a b) c) (HSub.hSub (HDiv.hDiv a c) (HDiv.hDiv b c))
                                                                 -/
theorem tsub_div (a b c : α) : (a - b) / c = a / c - b / c := by simp_rw [div_eq_mul_inv, tsub_mul]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


