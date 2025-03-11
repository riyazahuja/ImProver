@[to_additive]
lemma fn_min_mul_fn_max (f : α → β) (a b : α) : f (min a b) * f (max a b) = f a * f b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : CommSemigroup β
    f : α → β
    a b : α
    ⊢ Eq (HMul.hMul (f (Min.min a b)) (f (Max.max a b))) (HMul.hMul (f a) (f b))
  -/
                                   /-
                                     🎉 no goals
                                   -/
  obtain h | h := le_total a b <;> simp [h, mul_comm]
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive]
lemma fn_max_mul_fn_min (f : α → β) (a b : α) : f (max a b) * f (min a b) = f a * f b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : CommSemigroup β
    f : α → β
    a b : α
    ⊢ Eq (HMul.hMul (f (Max.max a b)) (f (Min.min a b))) (HMul.hMul (f a) (f b))
  -/
                                   /-
                                     🎉 no goals
                                   -/
  obtain h | h := le_total a b <;> simp [h, mul_comm]
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive (attr := simp)]
lemma min_mul_max (a b : α) : min a b * max a b = a * b := fn_min_mul_fn_max id _ _


@[to_additive (attr := simp)]
lemma max_mul_min (a b : α) : max a b * min a b = a * b := fn_max_mul_fn_min id _ _


@[to_additive]
theorem min_mul_mul_left (a b c : α) : min (a * b) (a * c) = a * min b c :=
  (monotone_id.const_mul' a).map_min.symm


@[to_additive]
theorem max_mul_mul_left (a b c : α) : max (a * b) (a * c) = a * max b c :=
  (monotone_id.const_mul' a).map_max.symm


@[to_additive]
theorem min_mul_mul_right (a b c : α) : min (a * c) (b * c) = min a b * c :=
  (monotone_id.mul_const' c).map_min.symm


@[to_additive]
theorem max_mul_mul_right (a b c : α) : max (a * c) (b * c) = max a b * c :=
  (monotone_id.mul_const' c).map_max.symm


@[to_additive]
theorem lt_or_lt_of_mul_lt_mul [MulLeftMono α] [MulRightMono α] {a₁ a₂ b₁ b₂ : α} :
    a₁ * b₁ < a₂ * b₂ → a₁ < a₂ ∨ b₁ < b₂ := by
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : Mul α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a₁ a₂ b₁ b₂ : α
    ⊢ LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂) → Or (LT.lt a₁ a₂) (LT.lt b₁ b₂)
  -/
  contrapose!
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : Mul α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightMono α
    a₁ a₂ b₁ b₂ : α
    ⊢ And (LE.le a₂ a₁) (LE.le b₂ b₁) → LE.le (HMul.hMul a₂ b₂) (HMul.hMul a₁ b₁)
  -/
  exact fun h => mul_le_mul' h.1 h.2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem le_or_lt_of_mul_le_mul [MulLeftMono α] [MulRightStrictMono α] {a₁ a₂ b₁ b₂ : α} :
    a₁ * b₁ ≤ a₂ * b₂ → a₁ ≤ a₂ ∨ b₁ < b₂ := by
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : Mul α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightStrictMono α
    a₁ a₂ b₁ b₂ : α
    ⊢ LE.le (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂) → Or (LE.le a₁ a₂) (LT.lt b₁ b₂)
  -/
  contrapose!
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : Mul α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightStrictMono α
    a₁ a₂ b₁ b₂ : α
    ⊢ And (LT.lt a₂ a₁) (LE.le b₂ b₁) → LT.lt (HMul.hMul a₂ b₂) (HMul.hMul a₁ b₁)
  -/
  exact fun h => mul_lt_mul_of_lt_of_le h.1 h.2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem lt_or_le_of_mul_le_mul [MulLeftStrictMono α] [MulRightMono α] {a₁ a₂ b₁ b₂ : α} :
    a₁ * b₁ ≤ a₂ * b₂ → a₁ < a₂ ∨ b₁ ≤ b₂ := by
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : Mul α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightMono α
    a₁ a₂ b₁ b₂ : α
    ⊢ LE.le (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂) → Or (LT.lt a₁ a₂) (LE.le b₁ b₂)
  -/
  contrapose!
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : Mul α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightMono α
    a₁ a₂ b₁ b₂ : α
    ⊢ And (LE.le a₂ a₁) (LT.lt b₂ b₁) → LT.lt (HMul.hMul a₂ b₂) (HMul.hMul a₁ b₁)
  -/
  exact fun h => mul_lt_mul_of_le_of_lt h.1 h.2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem le_or_le_of_mul_le_mul [MulLeftStrictMono α] [MulRightStrictMono α] {a₁ a₂ b₁ b₂ : α} :
    a₁ * b₁ ≤ a₂ * b₂ → a₁ ≤ a₂ ∨ b₁ ≤ b₂ := by
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : Mul α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a₁ a₂ b₁ b₂ : α
    ⊢ LE.le (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂) → Or (LE.le a₁ a₂) (LE.le b₁ b₂)
  -/
  contrapose!
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : Mul α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a₁ a₂ b₁ b₂ : α
    ⊢ And (LT.lt a₂ a₁) (LT.lt b₂ b₁) → LT.lt (HMul.hMul a₂ b₂) (HMul.hMul a₁ b₁)
  -/
  exact fun h => mul_lt_mul_of_lt_of_lt h.1 h.2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_lt_mul_iff_of_le_of_le [MulLeftMono α]
    [MulRightMono α] [MulLeftStrictMono α]
    [MulRightStrictMono α] {a₁ a₂ b₁ b₂ : α} (ha : a₁ ≤ a₂)
    (hb : b₁ ≤ b₂) : a₁ * b₁ < a₂ * b₂ ↔ a₁ < a₂ ∨ b₁ < b₂ := by
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : Mul α
    inst✝³ : MulLeftMono α
    inst✝² : MulRightMono α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a₁ a₂ b₁ b₂ : α
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    ⊢ Iff (LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂)) (Or (LT.lt a₁ a₂) (LT.lt b₁  …
  -/
  refine ⟨lt_or_lt_of_mul_lt_mul, fun h => ?_⟩
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : Mul α
    inst✝³ : MulLeftMono α
    inst✝² : MulRightMono α
    inst✝¹ : MulLeftStrictMono α
    inst✝ : MulRightStrictMono α
    a₁ a₂ b₁ b₂ : α
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    h : Or (LT.lt a₁ a₂) (LT.lt b₁ b₂)
    ⊢ LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂)
  -/
  rcases h with ha' | hb'
    /-
      case inl
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : Mul α
      inst✝³ : MulLeftMono α
      inst✝² : MulRightMono α
      inst✝¹ : MulLeftStrictMono α
      inst✝ : MulRightStrictMono α
      a₁ a₂ b₁ b₂ : α
      ha : LE.le a₁ a₂
      hb : LE.le b₁ b₂
      ha' : LT.lt a₁ a₂
      ⊢ LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂)
    -/
  · exact mul_lt_mul_of_lt_of_le ha' hb
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : Mul α
      inst✝³ : MulLeftMono α
      inst✝² : MulRightMono α
      inst✝¹ : MulLeftStrictMono α
      inst✝ : MulRightStrictMono α
      a₁ a₂ b₁ b₂ : α
      ha : LE.le a₁ a₂
      hb : LE.le b₁ b₂
      hb' : LT.lt b₁ b₂
      ⊢ LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂)
    -/
  · exact mul_lt_mul_of_le_of_lt ha hb'
    /-
      🎉 no goals
    -/


@[to_additive]
theorem min_le_mul_of_one_le_right [MulLeftMono α] {a b : α} (hb : 1 ≤ b) :
    min a b ≤ a * b :=
  min_le_iff.2 <| Or.inl <| le_mul_of_one_le_right' hb


@[to_additive]
theorem min_le_mul_of_one_le_left [MulRightMono α] {a b : α} (ha : 1 ≤ a) :
    min a b ≤ a * b :=
  min_le_iff.2 <| Or.inr <| le_mul_of_one_le_left' ha


@[to_additive]
theorem max_le_mul_of_one_le [MulLeftMono α] [MulRightMono α] {a b : α} (ha : 1 ≤ a) (hb : 1 ≤ b) :
    max a b ≤ a * b :=
  max_le_iff.2 ⟨le_mul_of_one_le_right' hb, le_mul_of_one_le_left' ha⟩


