/-- Auxiliary lemma for norm_cast to move the cast `-↑n` upwards to `↑-↑n`.

(The restriction to `DivisionRing` is necessary, otherwise this would also apply in the case where
`R = ℤ` and cause nontermination.)
-/
@[norm_cast]
                                                                                  /-
                                                                                    R : Type u_2
                                                                                    inst✝ : DivisionRing R
                                                                                    n : Nat
                                                                                    ⊢ Eq (↑(Neg.neg ↑n)) (Neg.neg ↑n)
                                                                                  -/
theorem cast_neg_natCast {R} [DivisionRing R] (n : ℕ) : ((-n : ℤ) : R) = -n := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem cast_div [DivisionRing α] {m n : ℤ} (n_dvd : n ∣ m) (hn : (n : α) ≠ 0) :
    ((m / n : ℤ) : α) = m / n := by
  /-
    α : Type u_1
    inst✝ : DivisionRing α
    m n : Int
    n_dvd : Dvd.dvd n m
    hn : Ne (↑n) 0
    ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
  -/
  rcases n_dvd with ⟨k, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝ : DivisionRing α
    n : Int
    hn : Ne (↑n) 0
    k : Int
    ⊢ Eq (↑(HDiv.hDiv (HMul.hMul n k) n)) (HDiv.hDiv ↑(HMul.hMul n k) ↑n)
  -/
  have : n ≠ 0 := by rintro rfl; simp at hn
  /-
    case intro
    α : Type u_1
    inst✝ : DivisionRing α
    n : Int
    hn : Ne (↑n) 0
    k : Int
    this : Ne n 0
    ⊢ Eq (↑(HDiv.hDiv (HMul.hMul n k) n)) (HDiv.hDiv ↑(HMul.hMul n k) ↑n)
  -/
  rw [Int.mul_ediv_cancel_left _ this, mul_comm n, Int.cast_mul, mul_div_cancel_right₀ _ hn]
  /-
    🎉 no goals
  -/


