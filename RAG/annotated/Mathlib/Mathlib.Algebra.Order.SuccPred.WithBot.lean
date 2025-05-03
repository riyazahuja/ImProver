lemma succ_natCast (n : ℕ) : succ (n : WithBot α) = n + 1 := by
  /-
    α : Type u_1
    inst✝³ : Preorder α
    inst✝² : OrderBot α
    inst✝¹ : AddMonoidWithOne α
    inst✝ : SuccAddOrder α
    n : Nat
    ⊢ Eq (↑n).succ (HAdd.hAdd (↑n) 1)
  -/
  rw [← WithBot.coe_natCast, succ_coe, Order.succ_eq_add_one]
  /-
    🎉 no goals
  -/


                                                         /-
                                                           α : Type u_1
                                                           inst✝³ : Preorder α
                                                           inst✝² : OrderBot α
                                                           inst✝¹ : AddMonoidWithOne α
                                                           inst✝ : SuccAddOrder α
                                                           ⊢ Eq (WithBot.succ 0) 1
                                                         -/
@[simp] lemma succ_zero : succ (0 : WithBot α) = 1 := by simpa using succ_natCast 0
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
                                                /-
                                                  α : Type u_1
                                                  inst✝³ : Preorder α
                                                  inst✝² : OrderBot α
                                                  inst✝¹ : AddMonoidWithOne α
                                                  inst✝ : SuccAddOrder α
                                                  ⊢ Eq (WithBot.succ 1) 2
                                                -/
lemma succ_one : succ (1 : WithBot α) = 2 := by simpa [one_add_one_eq_two] using succ_natCast 1
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
lemma succ_ofNat (n : ℕ) [n.AtLeastTwo] :
    succ (no_index (OfNat.ofNat n) : WithBot α) = OfNat.ofNat n + 1 := succ_natCast n


