theorem Invertible.ne_zero [MulZeroOneClass α] (a : α) [Nontrivial α] [Invertible a] : a ≠ 0 :=
  fun ha =>
  zero_ne_one <|
    calc
                        /-
                          α : Type u
                          inst✝² : MulZeroOneClass α
                          a : α
                          inst✝¹ : Nontrivial α
                          inst✝ : Invertible a
                          ha : Eq a 0
                          ⊢ Eq 0 (HMul.hMul (Invertible.invOf a) a)
                        -/
      0 = ⅟ a * a := by simp [ha]
                        /-
                          🎉 no goals
                        -/
      _ = 1 := invOf_mul_self


@[deprecated (since := "2024-08-15")] alias nonzero_of_invertible := Invertible.ne_zero


instance (priority := 100) Invertible.toNeZero [MulZeroOneClass α] [Nontrivial α] (a : α)
    [Invertible a] : NeZero a :=
  ⟨Invertible.ne_zero a⟩


/-- A variant of `Ring.inverse_unit`. -/
@[simp]
theorem Ring.inverse_invertible (x : α) [Invertible x] : Ring.inverse x = ⅟ x :=
  Ring.inverse_unit (unitOfInvertible _)


/-- `a⁻¹` is an inverse of `a` if `a ≠ 0` -/
def invertibleOfNonzero {a : α} (h : a ≠ 0) : Invertible a :=
  ⟨a⁻¹, inv_mul_cancel₀ h, mul_inv_cancel₀ h⟩


@[simp]
theorem invOf_eq_inv (a : α) [Invertible a] : ⅟ a = a⁻¹ :=
  invOf_eq_right_inv (mul_inv_cancel₀ (Invertible.ne_zero a))


@[simp]
theorem inv_mul_cancel_of_invertible (a : α) [Invertible a] : a⁻¹ * a = 1 :=
  inv_mul_cancel₀ (Invertible.ne_zero a)


@[simp]
theorem mul_inv_cancel_of_invertible (a : α) [Invertible a] : a * a⁻¹ = 1 :=
  mul_inv_cancel₀ (Invertible.ne_zero a)


/-- `a` is the inverse of `a⁻¹` -/
def invertibleInv {a : α} [Invertible a] : Invertible a⁻¹ :=
         /-
           α : Type u
           inst✝¹ : GroupWithZero α
           a : α
           inst✝ : Invertible a
           ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
         -/
         /-
           🎉 no goals
         -/
  ⟨a, by simp, by simp⟩
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem div_mul_cancel_of_invertible (a b : α) [Invertible b] : a / b * b = a :=
  div_mul_cancel₀ a (Invertible.ne_zero b)


@[simp]
theorem mul_div_cancel_of_invertible (a b : α) [Invertible b] : a * b / b = a :=
  mul_div_cancel_right₀ a (Invertible.ne_zero b)


@[simp]
theorem div_self_of_invertible (a : α) [Invertible a] : a / a = 1 :=
  div_self (Invertible.ne_zero a)


/-- `b / a` is the inverse of `a / b` -/
def invertibleDiv (a b : α) [Invertible a] [Invertible b] : Invertible (a / b) :=
             /-
               α : Type u
               inst✝² : GroupWithZero α
               a b : α
               inst✝¹ : Invertible a
               inst✝ : Invertible b
               ⊢ Eq (HMul.hMul (HDiv.hDiv b a) (HDiv.hDiv a b)) 1
             -/
             /-
               🎉 no goals
             -/
  ⟨b / a, by simp [← mul_div_assoc], by simp [← mul_div_assoc]⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem invOf_div (a b : α) [Invertible a] [Invertible b] [Invertible (a / b)] :
    ⅟ (a / b) = b / a :=
                         /-
                           α : Type u
                           inst✝³ : GroupWithZero α
                           a b : α
                           inst✝² : Invertible a
                           inst✝¹ : Invertible b
                           inst✝ : Invertible (HDiv.hDiv a b)
                           ⊢ Eq (HMul.hMul (HDiv.hDiv a b) (HDiv.hDiv b a)) 1
                         -/
  invOf_eq_right_inv (by simp [← mul_div_assoc])
                         /-
                           🎉 no goals
                         -/


