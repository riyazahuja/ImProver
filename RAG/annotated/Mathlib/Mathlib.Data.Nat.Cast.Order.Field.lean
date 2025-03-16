lemma cast_inv_le_one : ∀ n : ℕ, (n⁻¹ : α) ≤ 1
            /-
              α : Type u_1
              inst✝ : LinearOrderedSemifield α
              ⊢ LE.le (Inv.inv ↑0) 1
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                                         /-
                                           α : Type u_1
                                           inst✝ : LinearOrderedSemifield α
                                           n : Nat
                                           ⊢ LE.le 1 ↑(HAdd.hAdd n 1)
                                         -/
  | n + 1 => inv_le_one_of_one_le₀ <| by simp [Nat.cast_nonneg]
                                         /-
                                           🎉 no goals
                                         -/


/-- Natural division is always less than division in the field. -/
theorem cast_div_le {m n : ℕ} : ((m / n : ℕ) : α) ≤ m / n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    m n : Nat
    ⊢ LE.le (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
  -/
  cases n
    /-
      case zero
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      m : Nat
      ⊢ LE.le (↑(HDiv.hDiv m 0)) (HDiv.hDiv ↑m ↑0)
    -/
  · rw [cast_zero, div_zero, Nat.div_zero, cast_zero]
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    m n✝ : Nat
    ⊢ LE.le (↑(HDiv.hDiv m (HAdd.hAdd n✝ 1))) (HDiv.hDiv ↑m ↑(HAdd.hAdd n✝ 1))
  -/
  rw [le_div_iff₀, ← Nat.cast_mul, @Nat.cast_le]
    /-
      case succ
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      m n✝ : Nat
      ⊢ LE.le (HMul.hMul (HDiv.hDiv m (HAdd.hAdd n✝ 1)) (HAdd.hAdd n✝ 1)) m
    -/
  · exact Nat.div_mul_le_self m _
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      m n✝ : Nat
      ⊢ LT.lt 0 ↑(HAdd.hAdd n✝ 1)
    -/
  · exact Nat.cast_pos.2 (Nat.succ_pos _)
    /-
      🎉 no goals
    -/


theorem inv_pos_of_nat {n : ℕ} : 0 < ((n : α) + 1)⁻¹ :=
  inv_pos.2 <| add_pos_of_nonneg_of_pos n.cast_nonneg zero_lt_one


theorem one_div_pos_of_nat {n : ℕ} : 0 < 1 / ((n : α) + 1) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    n : Nat
    ⊢ LT.lt 0 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))
  -/
  rw [one_div]
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    n : Nat
    ⊢ LT.lt 0 (Inv.inv (HAdd.hAdd (↑n) 1))
  -/
  exact inv_pos_of_nat
  /-
    🎉 no goals
  -/


theorem one_div_le_one_div {n m : ℕ} (h : n ≤ m) : 1 / ((m : α) + 1) ≤ 1 / ((n : α) + 1) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    n m : Nat
    h : LE.le n m
    ⊢ LE.le (HDiv.hDiv 1 (HAdd.hAdd (↑m) 1)) (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))
  -/
  refine one_div_le_one_div_of_le ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      n m : Nat
      h : LE.le n m
      ⊢ LT.lt 0 (HAdd.hAdd (↑n) 1)
    -/
  · exact Nat.cast_add_one_pos _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      n m : Nat
      h : LE.le n m
      ⊢ LE.le (HAdd.hAdd (↑n) 1) (HAdd.hAdd (↑m) 1)
    -/
  · simpa
    /-
      🎉 no goals
    -/


theorem one_div_lt_one_div {n m : ℕ} (h : n < m) : 1 / ((m : α) + 1) < 1 / ((n : α) + 1) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    n m : Nat
    h : LT.lt n m
    ⊢ LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑m) 1)) (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))
  -/
  refine one_div_lt_one_div_of_lt ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      n m : Nat
      h : LT.lt n m
      ⊢ LT.lt 0 (HAdd.hAdd (↑n) 1)
    -/
  · exact Nat.cast_add_one_pos _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      n m : Nat
      h : LT.lt n m
      ⊢ LT.lt (HAdd.hAdd (↑n) 1) (HAdd.hAdd (↑m) 1)
    -/
  · simpa
    /-
      🎉 no goals
    -/


theorem one_div_cast_pos {n : ℕ} (hn : n ≠ 0) : 0 < 1 / (n : α) :=
  one_div_pos.mpr (cast_pos.mpr (Nat.pos_of_ne_zero hn))


theorem one_div_cast_nonneg (n : ℕ) : 0 ≤ 1 / (n : α) := one_div_nonneg.mpr (cast_nonneg' n)


theorem one_div_cast_ne_zero {n : ℕ} (hn : n ≠ 0) : 1 / (n : α) ≠ 0 :=
  _root_.ne_of_gt (one_div_cast_pos hn)


