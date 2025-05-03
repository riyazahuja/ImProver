lemma lt_add_one [One α] [AddZeroClass α] [PartialOrder α] [ZeroLEOneClass α]
    [NeZero (1 : α)] [AddLeftStrictMono α] (a : α) : a < a + 1 :=
  lt_add_of_pos_right _ zero_lt_one


lemma lt_one_add [One α] [AddZeroClass α] [PartialOrder α] [ZeroLEOneClass α]
    [NeZero (1 : α)] [AddRightStrictMono α] (a : α) : a < 1 + a :=
  lt_add_of_pos_left _ zero_lt_one


lemma zero_le_two [Preorder α] [ZeroLEOneClass α] [AddLeftMono α] :
    (0 : α) ≤ 2 := by
  /-
    α : Type u_1
    inst✝³ : AddMonoidWithOne α
    inst✝² : Preorder α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : AddLeftMono α
    ⊢ LE.le 0 2
  -/
  rw [← one_add_one_eq_two]
  /-
    α : Type u_1
    inst✝³ : AddMonoidWithOne α
    inst✝² : Preorder α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : AddLeftMono α
    ⊢ LE.le 0 (HAdd.hAdd 1 1)
  -/
  exact add_nonneg zero_le_one zero_le_one
  /-
    🎉 no goals
  -/


lemma zero_le_three [Preorder α] [ZeroLEOneClass α] [AddLeftMono α] :
    (0 : α) ≤ 3 := by
  /-
    α : Type u_1
    inst✝³ : AddMonoidWithOne α
    inst✝² : Preorder α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : AddLeftMono α
    ⊢ LE.le 0 3
  -/
  rw [← two_add_one_eq_three]
  /-
    α : Type u_1
    inst✝³ : AddMonoidWithOne α
    inst✝² : Preorder α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : AddLeftMono α
    ⊢ LE.le 0 (HAdd.hAdd 2 1)
  -/
  exact add_nonneg zero_le_two zero_le_one
  /-
    🎉 no goals
  -/


lemma zero_le_four [Preorder α] [ZeroLEOneClass α] [AddLeftMono α] :
    (0 : α) ≤ 4 := by
  /-
    α : Type u_1
    inst✝³ : AddMonoidWithOne α
    inst✝² : Preorder α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : AddLeftMono α
    ⊢ LE.le 0 4
  -/
  rw [← three_add_one_eq_four]
  /-
    α : Type u_1
    inst✝³ : AddMonoidWithOne α
    inst✝² : Preorder α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : AddLeftMono α
    ⊢ LE.le 0 (HAdd.hAdd 3 1)
  -/
  exact add_nonneg zero_le_three zero_le_one
  /-
    🎉 no goals
  -/


lemma one_le_two [LE α] [ZeroLEOneClass α] [AddLeftMono α] :
    (1 : α) ≤ 2 :=
  calc (1 : α) = 1 + 0 := (add_zero 1).symm
     _ ≤ 1 + 1 := add_le_add_left zero_le_one _
     _ = 2 := one_add_one_eq_two


lemma one_le_two' [LE α] [ZeroLEOneClass α] [AddRightMono α] :
    (1 : α) ≤ 2 :=
  calc (1 : α) = 0 + 1 := (zero_add 1).symm
     _ ≤ 1 + 1 := add_le_add_right zero_le_one _
     _ = 2 := one_add_one_eq_two


/-- See `zero_lt_two'` for a version with the type explicit. -/
@[simp] lemma zero_lt_two : (0 : α) < 2 := zero_lt_one.trans_le one_le_two


/-- See `zero_lt_three'` for a version with the type explicit. -/
@[simp] lemma zero_lt_three : (0 : α) < 3 := by
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : ZeroLEOneClass α
    inst✝¹ : NeZero 1
    inst✝ : AddLeftMono α
    ⊢ LT.lt 0 3
  -/
  rw [← two_add_one_eq_three]
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : ZeroLEOneClass α
    inst✝¹ : NeZero 1
    inst✝ : AddLeftMono α
    ⊢ LT.lt 0 (HAdd.hAdd 2 1)
  -/
  exact lt_add_of_lt_of_nonneg zero_lt_two zero_le_one
  /-
    🎉 no goals
  -/


/-- See `zero_lt_four'` for a version with the type explicit. -/
@[simp] lemma zero_lt_four : (0 : α) < 4 := by
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : ZeroLEOneClass α
    inst✝¹ : NeZero 1
    inst✝ : AddLeftMono α
    ⊢ LT.lt 0 4
  -/
  rw [← three_add_one_eq_four]
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : ZeroLEOneClass α
    inst✝¹ : NeZero 1
    inst✝ : AddLeftMono α
    ⊢ LT.lt 0 (HAdd.hAdd 3 1)
  -/
  exact lt_add_of_lt_of_nonneg zero_lt_three zero_le_one
  /-
    🎉 no goals
  -/


/-- See `zero_lt_two` for a version with the type implicit. -/
lemma zero_lt_two' : (0 : α) < 2 := zero_lt_two


/-- See `zero_lt_three` for a version with the type implicit. -/
lemma zero_lt_three' : (0 : α) < 3 := zero_lt_three


/-- See `zero_lt_four` for a version with the type implicit. -/
lemma zero_lt_four' : (0 : α) < 4 := zero_lt_four


instance ZeroLEOneClass.neZero.two : NeZero (2 : α) := ⟨zero_lt_two.ne'⟩

instance ZeroLEOneClass.neZero.three : NeZero (3 : α) := ⟨zero_lt_three.ne'⟩

instance ZeroLEOneClass.neZero.four : NeZero (4 : α) := ⟨zero_lt_four.ne'⟩


lemma one_lt_two [AddLeftStrictMono α] : (1 : α) < 2 := by
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : ZeroLEOneClass α
    inst✝¹ : NeZero 1
    inst✝ : AddLeftStrictMono α
    ⊢ LT.lt 1 2
  -/
  rw [← one_add_one_eq_two]
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : ZeroLEOneClass α
    inst✝¹ : NeZero 1
    inst✝ : AddLeftStrictMono α
    ⊢ LT.lt 1 (HAdd.hAdd 1 1)
  -/
  exact lt_add_one _
  /-
    🎉 no goals
  -/


alias two_pos := zero_lt_two


alias three_pos := zero_lt_three


alias four_pos := zero_lt_four

