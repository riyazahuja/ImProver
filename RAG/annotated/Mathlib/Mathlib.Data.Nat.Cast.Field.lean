@[simp]
theorem cast_div [DivisionSemiring α] {m n : ℕ} (n_dvd : n ∣ m) (hn : (n : α) ≠ 0) :
    ((m / n : ℕ) : α) = m / n := by
  /-
    α : Type u_1
    inst✝ : DivisionSemiring α
    m n : Nat
    n_dvd : Dvd.dvd n m
    hn : Ne (↑n) 0
    ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
  -/
  rcases n_dvd with ⟨k, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝ : DivisionSemiring α
    n : Nat
    hn : Ne (↑n) 0
    k : Nat
    ⊢ Eq (↑(HDiv.hDiv (HMul.hMul n k) n)) (HDiv.hDiv ↑(HMul.hMul n k) ↑n)
  -/
  have : n ≠ 0 := by rintro rfl; simp at hn
  rw [Nat.mul_div_cancel_left _ <| zero_lt_of_ne_zero this, mul_comm n,
    cast_mul, mul_div_cancel_right₀ _ hn]


theorem cast_div_div_div_cancel_right [DivisionSemiring α] [CharZero α] {m n d : ℕ}
    (hn : d ∣ n) (hm : d ∣ m) :
    (↑(m / d) : α) / (↑(n / d) : α) = (m : α) / n := by
  /-
    α : Type u_1
    inst✝¹ : DivisionSemiring α
    inst✝ : CharZero α
    m n d : Nat
    hn : Dvd.dvd d n
    hm : Dvd.dvd d m
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv m d) ↑(HDiv.hDiv n d)) (HDiv.hDiv ↑m ↑n)
  -/
  rcases eq_or_ne d 0 with (rfl | hd); · simp [Nat.zero_dvd.1 hm]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    α : Type u_1
    inst✝¹ : DivisionSemiring α
    inst✝ : CharZero α
    m n d : Nat
    hn : Dvd.dvd d n
    hm : Dvd.dvd d m
    hd : Ne d 0
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv m d) ↑(HDiv.hDiv n d)) (HDiv.hDiv ↑m ↑n)
  -/
  replace hd : (d : α) ≠ 0 := by norm_cast
  /-
    case inr
    α : Type u_1
    inst✝¹ : DivisionSemiring α
    inst✝ : CharZero α
    m n d : Nat
    hn : Dvd.dvd d n
    hm : Dvd.dvd d m
    hd : Ne (↑d) 0
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv m d) ↑(HDiv.hDiv n d)) (HDiv.hDiv ↑m ↑n)
  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  rw [cast_div hm, cast_div hn, div_div_div_cancel_right₀ hd] <;> exact hd
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


