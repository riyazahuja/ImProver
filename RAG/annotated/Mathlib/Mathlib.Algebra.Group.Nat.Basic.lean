instance instMulOneClass : MulOneClass ℕ where
  one_mul := Nat.one_mul
  mul_one := Nat.mul_one


instance instAddCancelCommMonoid : AddCancelCommMonoid ℕ where
  add := Nat.add
  add_assoc := Nat.add_assoc
  zero := Nat.zero
  zero_add := Nat.zero_add
  add_zero := Nat.add_zero
  add_comm := Nat.add_comm
  nsmul m n := m * n
  nsmul_zero := Nat.zero_mul
  nsmul_succ := succ_mul
  add_left_cancel _ _ _ := Nat.add_left_cancel


instance instCommMonoid : CommMonoid ℕ where
  mul := Nat.mul
  mul_assoc := Nat.mul_assoc
  one := Nat.succ Nat.zero
  one_mul := Nat.one_mul
  mul_one := Nat.mul_one
  mul_comm := Nat.mul_comm
  npow m n := n ^ m
  npow_zero := Nat.pow_zero
  npow_succ _ _ := rfl


                                                         /-
                                                           ⊢ AddCommMonoid Nat
                                                         -/
instance instAddCommMonoid    : AddCommMonoid ℕ    := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ AddMonoid Nat
                                                         -/
instance instAddMonoid        : AddMonoid ℕ        := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ Monoid Nat
                                                         -/
instance instMonoid           : Monoid ℕ           := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ CommSemigroup Nat
                                                         -/
instance instCommSemigroup    : CommSemigroup ℕ    := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ Semigroup Nat
                                                         -/
instance instSemigroup        : Semigroup ℕ        := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ AddCommSemigroup Nat
                                                         -/
instance instAddCommSemigroup : AddCommSemigroup ℕ := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           ⊢ AddSemigroup Nat
                                                         -/
instance instAddSemigroup     : AddSemigroup ℕ     := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp 900] protected lemma nsmul_eq_mul (m n : ℕ) : m • n = m * n := rfl


