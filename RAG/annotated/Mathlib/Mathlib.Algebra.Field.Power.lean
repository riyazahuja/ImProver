theorem Odd.neg_zpow (h : Odd n) (a : α) : (-a) ^ n = -a ^ n := by
  /-
    α : Type u_1
    inst✝ : DivisionRing α
    n : Int
    h : Odd n
    a : α
    ⊢ Eq (HPow.hPow (Neg.neg a) n) (Neg.neg (HPow.hPow a n))
  -/
  have hn : n ≠ 0 := by rintro rfl; exact Int.not_even_iff_odd.2 h even_zero
  /-
    α : Type u_1
    inst✝ : DivisionRing α
    n : Int
    h : Odd n
    a : α
    hn : Ne n 0
    ⊢ Eq (HPow.hPow (Neg.neg a) n) (Neg.neg (HPow.hPow a n))
  -/
  obtain ⟨k, rfl⟩ := h
  simp_rw [zpow_add' (.inr (.inl hn)), zpow_one, zpow_mul, zpow_two, neg_mul_neg,
    neg_mul_eq_mul_neg]


                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : DivisionRing α
                                                                 n : Int
                                                                 h : Odd n
                                                                 ⊢ Eq (HPow.hPow (-1) n) (-1)
                                                               -/
theorem Odd.neg_one_zpow (h : Odd n) : (-1 : α) ^ n = -1 := by rw [h.neg_zpow, one_zpow]
                                                               /-
                                                                 🎉 no goals
                                                               -/


