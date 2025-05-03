@[simp]
theorem invOf_pos [Invertible a] : 0 < ⅟ a ↔ 0 < a :=
                            /-
                              α : Type u_1
                              inst✝¹ : LinearOrderedSemiring α
                              a : α
                              inst✝ : Invertible a
                              ⊢ LT.lt 0 (HMul.hMul a (Invertible.invOf a))
                            -/
  haveI : 0 < a * ⅟ a := by simp only [mul_invOf_self, zero_lt_one]
                            /-
                              🎉 no goals
                            -/
  ⟨fun h => pos_of_mul_pos_left this h.le, fun h => pos_of_mul_pos_right this h.le⟩


@[simp]
                                                            /-
                                                              α : Type u_1
                                                              inst✝¹ : LinearOrderedSemiring α
                                                              a : α
                                                              inst✝ : Invertible a
                                                              ⊢ Iff (LE.le (Invertible.invOf a) 0) (LE.le a 0)
                                                            -/
theorem invOf_nonpos [Invertible a] : ⅟ a ≤ 0 ↔ a ≤ 0 := by simp only [← not_lt, invOf_pos]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem invOf_nonneg [Invertible a] : 0 ≤ ⅟ a ↔ 0 ≤ a :=
                            /-
                              α : Type u_1
                              inst✝¹ : LinearOrderedSemiring α
                              a : α
                              inst✝ : Invertible a
                              ⊢ LT.lt 0 (HMul.hMul a (Invertible.invOf a))
                            -/
  haveI : 0 < a * ⅟ a := by simp only [mul_invOf_self, zero_lt_one]
                            /-
                              🎉 no goals
                            -/
  ⟨fun h => (pos_of_mul_pos_left this h).le, fun h => (pos_of_mul_pos_right this h).le⟩


@[simp]
                                                             /-
                                                               α : Type u_1
                                                               inst✝¹ : LinearOrderedSemiring α
                                                               a : α
                                                               inst✝ : Invertible a
                                                               ⊢ Iff (LT.lt (Invertible.invOf a) 0) (LT.lt a 0)
                                                             -/
theorem invOf_lt_zero [Invertible a] : ⅟ a < 0 ↔ a < 0 := by simp only [← not_le, invOf_nonneg]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem invOf_le_one [Invertible a] (h : 1 ≤ a) : ⅟ a ≤ 1 :=
  mul_invOf_self a ▸ le_mul_of_one_le_left (invOf_nonneg.2 <| zero_le_one.trans h) h


theorem pos_invOf_of_invertible_cast [Nontrivial α] (n : ℕ)
    [Invertible (n : α)] : 0 < ⅟(n : α) :=
  invOf_pos.2 <| Nat.cast_pos.2 <| pos_of_invertible_cast (α := α) n

