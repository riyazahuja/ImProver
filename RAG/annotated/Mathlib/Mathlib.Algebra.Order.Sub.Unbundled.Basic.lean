@[simp]
theorem add_tsub_cancel_of_le (h : a ≤ b) : a + (b - a) = b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b : α
    h : LE.le a b
    ⊢ Eq (HAdd.hAdd a (HSub.hSub b a)) b
  -/
  refine le_antisymm ?_ le_add_tsub
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b : α
    h : LE.le a b
    ⊢ LE.le (HAdd.hAdd a (HSub.hSub b a)) b
  -/
  obtain ⟨c, rfl⟩ := exists_add_of_le h
  /-
    case intro
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a c : α
    h : LE.le a (HAdd.hAdd a c)
    ⊢ LE.le (HAdd.hAdd a (HSub.hSub (HAdd.hAdd a c) a)) (HAdd.hAdd a c)
  -/
  exact add_le_add_left add_tsub_le_left a
  /-
    🎉 no goals
  -/


theorem tsub_add_cancel_of_le (h : a ≤ b) : b - a + a = b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b : α
    h : LE.le a b
    ⊢ Eq (HAdd.hAdd (HSub.hSub b a) a) b
  -/
  rw [add_comm]
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b : α
    h : LE.le a b
    ⊢ Eq (HAdd.hAdd a (HSub.hSub b a)) b
  -/
  exact add_tsub_cancel_of_le h
  /-
    🎉 no goals
  -/


theorem add_le_of_le_tsub_right_of_le (h : b ≤ c) (h2 : a ≤ c - b) : a + b ≤ c :=
  (add_le_add_right h2 b).trans_eq <| tsub_add_cancel_of_le h


theorem add_le_of_le_tsub_left_of_le (h : a ≤ c) (h2 : b ≤ c - a) : a + b ≤ c :=
  (add_le_add_left h2 a).trans_eq <| add_tsub_cancel_of_le h


theorem tsub_le_tsub_iff_right (h : c ≤ b) : a - c ≤ b - c ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    h : LE.le c b
    ⊢ Iff (LE.le (HSub.hSub a c) (HSub.hSub b c)) (LE.le a b)
  -/
  rw [tsub_le_iff_right, tsub_add_cancel_of_le h]
  /-
    🎉 no goals
  -/


theorem tsub_left_inj (h1 : c ≤ a) (h2 : c ≤ b) : a - c = b - c ↔ a = b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    h1 : LE.le c a
    h2 : LE.le c b
    ⊢ Iff (Eq (HSub.hSub a c) (HSub.hSub b c)) (Eq a b)
  -/
  simp_rw [le_antisymm_iff, tsub_le_tsub_iff_right h1, tsub_le_tsub_iff_right h2]
  /-
    🎉 no goals
  -/


theorem tsub_inj_left (h₁ : a ≤ b) (h₂ : a ≤ c) : b - a = c - a → b = c :=
  (tsub_left_inj h₁ h₂).1


/-- See `lt_of_tsub_lt_tsub_right` for a stronger statement in a linear order. -/
theorem lt_of_tsub_lt_tsub_right_of_le (h : c ≤ b) (h2 : a - c < b - c) : a < b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    h : LE.le c b
    h2 : LT.lt (HSub.hSub a c) (HSub.hSub b c)
    ⊢ LT.lt a b
  -/
  refine ((tsub_le_tsub_iff_right h).mp h2.le).lt_of_ne ?_
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    h : LE.le c b
    h2 : LT.lt (HSub.hSub a c) (HSub.hSub b c)
    ⊢ Ne a b
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a c : α
    h : LE.le c a
    h2 : LT.lt (HSub.hSub a c) (HSub.hSub a c)
    ⊢ False
  -/
  exact h2.false
  /-
    🎉 no goals
  -/


theorem tsub_add_tsub_cancel (hab : b ≤ a) (hcb : c ≤ b) : a - b + (b - c) = a - c := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hab : LE.le b a
    hcb : LE.le c b
    ⊢ Eq (HAdd.hAdd (HSub.hSub a b) (HSub.hSub b c)) (HSub.hSub a c)
  -/
  convert tsub_add_cancel_of_le (tsub_le_tsub_right hab c) using 2
  /-
    case h.e'_2.h.e'_5
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hab : LE.le b a
    hcb : LE.le c b
    ⊢ Eq (HSub.hSub a b) (HSub.hSub (HSub.hSub a c) (HSub.hSub b c))
  -/
  rw [tsub_tsub, add_tsub_cancel_of_le hcb]
  /-
    🎉 no goals
  -/


theorem tsub_tsub_tsub_cancel_right (h : c ≤ b) : a - c - (b - c) = a - b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    h : LE.le c b
    ⊢ Eq (HSub.hSub (HSub.hSub a c) (HSub.hSub b c)) (HSub.hSub a b)
  -/
  rw [tsub_tsub, add_tsub_cancel_of_le h]
  /-
    🎉 no goals
  -/


protected theorem eq_tsub_iff_add_eq_of_le (hc : AddLECancellable c) (h : c ≤ b) :
    a = b - c ↔ a + c = b :=
  ⟨by
    /-
      α : Type u_1
      inst✝⁵ : AddCommSemigroup α
      inst✝⁴ : PartialOrder α
      inst✝³ : ExistsAddOfLE α
      inst✝² : AddLeftMono α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      a b c : α
      hc : AddLECancellable c
      h : LE.le c b
      ⊢ Eq a (HSub.hSub b c) → Eq (HAdd.hAdd a c) b
    -/
    rintro rfl
    /-
      α : Type u_1
      inst✝⁵ : AddCommSemigroup α
      inst✝⁴ : PartialOrder α
      inst✝³ : ExistsAddOfLE α
      inst✝² : AddLeftMono α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      b c : α
      hc : AddLECancellable c
      h : LE.le c b
      ⊢ Eq (HAdd.hAdd (HSub.hSub b c) c) b
    -/
    exact tsub_add_cancel_of_le h, hc.eq_tsub_of_add_eq⟩
    /-
      🎉 no goals
    -/


protected theorem tsub_eq_iff_eq_add_of_le (hb : AddLECancellable b) (h : b ≤ a) :
                                /-
                                  α : Type u_1
                                  inst✝⁵ : AddCommSemigroup α
                                  inst✝⁴ : PartialOrder α
                                  inst✝³ : ExistsAddOfLE α
                                  inst✝² : AddLeftMono α
                                  inst✝¹ : Sub α
                                  inst✝ : OrderedSub α
                                  a b c : α
                                  hb : AddLECancellable b
                                  h : LE.le b a
                                  ⊢ Iff (Eq (HSub.hSub a b) c) (Eq a (HAdd.hAdd c b))
                                -/
    a - b = c ↔ a = c + b := by rw [eq_comm, hb.eq_tsub_iff_add_eq_of_le h, eq_comm]
                                /-
                                  🎉 no goals
                                -/


protected theorem add_tsub_assoc_of_le (hc : AddLECancellable c) (h : c ≤ b) (a : α) :
    a + b - c = a + (b - c) := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    b c : α
    hc : AddLECancellable c
    h : LE.le c b
    a : α
    ⊢ Eq (HSub.hSub (HAdd.hAdd a b) c) (HAdd.hAdd a (HSub.hSub b c))
  -/
  conv_lhs => rw [← add_tsub_cancel_of_le h, add_comm c, ← add_assoc, hc.add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


protected theorem tsub_add_eq_add_tsub (hb : AddLECancellable b) (h : b ≤ a) :
                                /-
                                  α : Type u_1
                                  inst✝⁵ : AddCommSemigroup α
                                  inst✝⁴ : PartialOrder α
                                  inst✝³ : ExistsAddOfLE α
                                  inst✝² : AddLeftMono α
                                  inst✝¹ : Sub α
                                  inst✝ : OrderedSub α
                                  a b c : α
                                  hb : AddLECancellable b
                                  h : LE.le b a
                                  ⊢ Eq (HAdd.hAdd (HSub.hSub a b) c) (HSub.hSub (HAdd.hAdd a c) b)
                                -/
    a - b + c = a + c - b := by rw [add_comm a, hb.add_tsub_assoc_of_le h, add_comm]
                                /-
                                  🎉 no goals
                                -/


protected theorem tsub_tsub_assoc (hbc : AddLECancellable (b - c)) (h₁ : b ≤ a) (h₂ : c ≤ b) :
    a - (b - c) = a - b + c :=
                              /-
                                α : Type u_1
                                inst✝⁵ : AddCommSemigroup α
                                inst✝⁴ : PartialOrder α
                                inst✝³ : ExistsAddOfLE α
                                inst✝² : AddLeftMono α
                                inst✝¹ : Sub α
                                inst✝ : OrderedSub α
                                a b c : α
                                hbc : AddLECancellable (HSub.hSub b c)
                                h₁ : LE.le b a
                                h₂ : LE.le c b
                                ⊢ Eq a (HAdd.hAdd (HAdd.hAdd (HSub.hSub a b) c) (HSub.hSub b c))
                              -/
  hbc.tsub_eq_of_eq_add <| by rw [add_assoc, add_tsub_cancel_of_le h₂, tsub_add_cancel_of_le h₁]
                              /-
                                🎉 no goals
                              -/


protected theorem tsub_add_tsub_comm (hb : AddLECancellable b) (hd : AddLECancellable d)
    (hba : b ≤ a) (hdc : d ≤ c) : a - b + (c - d) = a + c - (b + d) := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c d : α
    hb : AddLECancellable b
    hd : AddLECancellable d
    hba : LE.le b a
    hdc : LE.le d c
    ⊢ Eq (HAdd.hAdd (HSub.hSub a b) (HSub.hSub c d)) (HSub.hSub (HAdd.hAdd a c) (H …
  -/
  rw [hb.tsub_add_eq_add_tsub hba, ← hd.add_tsub_assoc_of_le hdc, tsub_tsub, add_comm d]
  /-
    🎉 no goals
  -/


protected theorem le_tsub_iff_left (ha : AddLECancellable a) (h : a ≤ c) : b ≤ c - a ↔ a + b ≤ c :=
  ⟨add_le_of_le_tsub_left_of_le h, ha.le_tsub_of_add_le_left⟩


protected theorem le_tsub_iff_right (ha : AddLECancellable a) (h : a ≤ c) :
    b ≤ c - a ↔ b + a ≤ c := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    ha : AddLECancellable a
    h : LE.le a c
    ⊢ Iff (LE.le b (HSub.hSub c a)) (LE.le (HAdd.hAdd b a) c)
  -/
  rw [add_comm]
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    ha : AddLECancellable a
    h : LE.le a c
    ⊢ Iff (LE.le b (HSub.hSub c a)) (LE.le (HAdd.hAdd a b) c)
  -/
  exact ha.le_tsub_iff_left h
  /-
    🎉 no goals
  -/


protected theorem tsub_lt_iff_left (hb : AddLECancellable b) (hba : b ≤ a) :
    a - b < c ↔ a < b + c := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    hba : LE.le b a
    ⊢ Iff (LT.lt (HSub.hSub a b) c) (LT.lt a (HAdd.hAdd b c))
  -/
  refine ⟨hb.lt_add_of_tsub_lt_left, ?_⟩
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    hba : LE.le b a
    ⊢ LT.lt a (HAdd.hAdd b c) → LT.lt (HSub.hSub a b) c
  -/
  intro h; refine (tsub_le_iff_left.mpr h.le).lt_of_ne ?_
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    hba : LE.le b a
    h : LT.lt a (HAdd.hAdd b c)
    ⊢ Ne (HSub.hSub a b) c
  -/
  rintro rfl; exact h.ne' (add_tsub_cancel_of_le hba)
              /-
                🎉 no goals
              -/


protected theorem tsub_lt_iff_right (hb : AddLECancellable b) (hba : b ≤ a) :
    a - b < c ↔ a < c + b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    hba : LE.le b a
    ⊢ Iff (LT.lt (HSub.hSub a b) c) (LT.lt a (HAdd.hAdd c b))
  -/
  rw [add_comm]
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    hba : LE.le b a
    ⊢ Iff (LT.lt (HSub.hSub a b) c) (LT.lt a (HAdd.hAdd b c))
  -/
  exact hb.tsub_lt_iff_left hba
  /-
    🎉 no goals
  -/


protected theorem tsub_lt_iff_tsub_lt (hb : AddLECancellable b) (hc : AddLECancellable c)
    (h₁ : b ≤ a) (h₂ : c ≤ a) : a - b < c ↔ a - c < b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    hc : AddLECancellable c
    h₁ : LE.le b a
    h₂ : LE.le c a
    ⊢ Iff (LT.lt (HSub.hSub a b) c) (LT.lt (HSub.hSub a c) b)
  -/
  rw [hb.tsub_lt_iff_left h₁, hc.tsub_lt_iff_right h₂]
  /-
    🎉 no goals
  -/


protected theorem le_tsub_iff_le_tsub (ha : AddLECancellable a) (hc : AddLECancellable c)
    (h₁ : a ≤ b) (h₂ : c ≤ b) : a ≤ b - c ↔ c ≤ b - a := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    ha : AddLECancellable a
    hc : AddLECancellable c
    h₁ : LE.le a b
    h₂ : LE.le c b
    ⊢ Iff (LE.le a (HSub.hSub b c)) (LE.le c (HSub.hSub b a))
  -/
  rw [ha.le_tsub_iff_left h₁, hc.le_tsub_iff_right h₂]
  /-
    🎉 no goals
  -/


protected theorem lt_tsub_iff_right_of_le (hc : AddLECancellable c) (h : c ≤ b) :
    a < b - c ↔ a + c < b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LE.le c b
    ⊢ Iff (LT.lt a (HSub.hSub b c)) (LT.lt (HAdd.hAdd a c) b)
  -/
  refine ⟨fun h' => (add_le_of_le_tsub_right_of_le h h'.le).lt_of_ne ?_, hc.lt_tsub_of_add_lt_right⟩
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LE.le c b
    h' : LT.lt a (HSub.hSub b c)
    ⊢ Ne (HAdd.hAdd a c) b
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a c : α
    hc : AddLECancellable c
    h : LE.le c (HAdd.hAdd a c)
    h' : LT.lt a (HSub.hSub (HAdd.hAdd a c) c)
    ⊢ False
  -/
  exact h'.ne' hc.add_tsub_cancel_right
  /-
    🎉 no goals
  -/


protected theorem lt_tsub_iff_left_of_le (hc : AddLECancellable c) (h : c ≤ b) :
    a < b - c ↔ c + a < b := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LE.le c b
    ⊢ Iff (LT.lt a (HSub.hSub b c)) (LT.lt (HAdd.hAdd c a) b)
  -/
  rw [add_comm]
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LE.le c b
    ⊢ Iff (LT.lt a (HSub.hSub b c)) (LT.lt (HAdd.hAdd a c) b)
  -/
  exact hc.lt_tsub_iff_right_of_le h
  /-
    🎉 no goals
  -/


protected theorem tsub_inj_right (hab : AddLECancellable (a - b)) (h₁ : b ≤ a) (h₂ : c ≤ a)
    (h₃ : a - b = a - c) : b = c := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hab : AddLECancellable (HSub.hSub a b)
    h₁ : LE.le b a
    h₂ : LE.le c a
    h₃ : Eq (HSub.hSub a b) (HSub.hSub a c)
    ⊢ Eq b c
  -/
  rw [← hab.inj]
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hab : AddLECancellable (HSub.hSub a b)
    h₁ : LE.le b a
    h₂ : LE.le c a
    h₃ : Eq (HSub.hSub a b) (HSub.hSub a c)
    ⊢ Eq (HAdd.hAdd (HSub.hSub a b) b) (HAdd.hAdd (HSub.hSub a b) c)
  -/
  rw [tsub_add_cancel_of_le h₁, h₃, tsub_add_cancel_of_le h₂]
  /-
    🎉 no goals
  -/


protected theorem lt_of_tsub_lt_tsub_left_of_le [AddLeftReflectLT α]
    (hb : AddLECancellable b) (hca : c ≤ a) (h : a - b < a - c) : c < b := by
  /-
    α : Type u_1
    inst✝⁶ : AddCommSemigroup α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : ExistsAddOfLE α
    inst✝³ : AddLeftMono α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftReflectLT α
    hb : AddLECancellable b
    hca : LE.le c a
    h : LT.lt (HSub.hSub a b) (HSub.hSub a c)
    ⊢ LT.lt c b
  -/
  conv_lhs at h => rw [← tsub_add_cancel_of_le hca]
  /-
    α : Type u_1
    inst✝⁶ : AddCommSemigroup α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : ExistsAddOfLE α
    inst✝³ : AddLeftMono α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftReflectLT α
    hb : AddLECancellable b
    hca : LE.le c a
    h : LT.lt (HSub.hSub (HAdd.hAdd (HSub.hSub a c) c) b) (HSub.hSub a c)
    ⊢ LT.lt c b
  -/
  exact lt_of_add_lt_add_left (hb.lt_add_of_tsub_lt_right h)
  /-
    🎉 no goals
  -/


protected theorem tsub_lt_tsub_left_of_le (hab : AddLECancellable (a - b)) (h₁ : b ≤ a)
    (h : c < b) : a - b < a - c :=
  (tsub_le_tsub_left h.le _).lt_of_ne fun h' => h.ne' <| hab.tsub_inj_right h₁ (h.le.trans h₁) h'


protected theorem tsub_lt_tsub_right_of_le (hc : AddLECancellable c) (h : c ≤ a) (h2 : a < b) :
    a - c < b - c := by
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LE.le c a
    h2 : LT.lt a b
    ⊢ LT.lt (HSub.hSub a c) (HSub.hSub b c)
  -/
  apply hc.lt_tsub_of_add_lt_left
  /-
    α : Type u_1
    inst✝⁵ : AddCommSemigroup α
    inst✝⁴ : PartialOrder α
    inst✝³ : ExistsAddOfLE α
    inst✝² : AddLeftMono α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LE.le c a
    h2 : LT.lt a b
    ⊢ LT.lt (HAdd.hAdd c (HSub.hSub a c)) b
  -/
  rwa [add_tsub_cancel_of_le h]
  /-
    🎉 no goals
  -/


protected theorem tsub_lt_tsub_iff_left_of_le_of_le [AddLeftReflectLT α]
    (hb : AddLECancellable b) (hab : AddLECancellable (a - b)) (h₁ : b ≤ a) (h₂ : c ≤ a) :
    a - b < a - c ↔ c < b :=
  ⟨hb.lt_of_tsub_lt_tsub_left_of_le h₂, hab.tsub_lt_tsub_left_of_le h₁⟩


@[simp]
protected theorem add_tsub_tsub_cancel (hac : AddLECancellable (a - c)) (h : c ≤ a) :
    a + b - (a - c) = b + c :=
                              /-
                                α : Type u_1
                                inst✝⁵ : AddCommSemigroup α
                                inst✝⁴ : PartialOrder α
                                inst✝³ : ExistsAddOfLE α
                                inst✝² : AddLeftMono α
                                inst✝¹ : Sub α
                                inst✝ : OrderedSub α
                                a b c : α
                                hac : AddLECancellable (HSub.hSub a c)
                                h : LE.le c a
                                ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd (HAdd.hAdd b c) (HSub.hSub a c))
                              -/
  hac.tsub_eq_of_eq_add <| by rw [add_assoc, add_tsub_cancel_of_le h, add_comm]
                              /-
                                🎉 no goals
                              -/


protected theorem tsub_tsub_cancel_of_le (hba : AddLECancellable (b - a)) (h : a ≤ b) :
    b - (b - a) = a :=
  hba.tsub_eq_of_eq_add (add_tsub_cancel_of_le h).symm


protected theorem tsub_tsub_tsub_cancel_left (hab : AddLECancellable (a - b)) (h : b ≤ a) :
                                  /-
                                    α : Type u_1
                                    inst✝⁵ : AddCommSemigroup α
                                    inst✝⁴ : PartialOrder α
                                    inst✝³ : ExistsAddOfLE α
                                    inst✝² : AddLeftMono α
                                    inst✝¹ : Sub α
                                    inst✝ : OrderedSub α
                                    a b c : α
                                    hab : AddLECancellable (HSub.hSub a b)
                                    h : LE.le b a
                                    ⊢ Eq (HSub.hSub (HSub.hSub a c) (HSub.hSub a b)) (HSub.hSub b c)
                                  -/
    a - c - (a - b) = b - c := by rw [tsub_right_comm, hab.tsub_tsub_cancel_of_le h]
                                  /-
                                    🎉 no goals
                                  -/


theorem eq_tsub_iff_add_eq_of_le (h : c ≤ b) : a = b - c ↔ a + c = b :=
  Contravariant.AddLECancellable.eq_tsub_iff_add_eq_of_le h


theorem tsub_eq_iff_eq_add_of_le (h : b ≤ a) : a - b = c ↔ a = c + b :=
  Contravariant.AddLECancellable.tsub_eq_iff_eq_add_of_le h


/-- See `add_tsub_le_assoc` for an inequality. -/
theorem add_tsub_assoc_of_le (h : c ≤ b) (a : α) : a + b - c = a + (b - c) :=
  Contravariant.AddLECancellable.add_tsub_assoc_of_le h a


theorem tsub_add_eq_add_tsub (h : b ≤ a) : a - b + c = a + c - b :=
  Contravariant.AddLECancellable.tsub_add_eq_add_tsub h


theorem tsub_tsub_assoc (h₁ : b ≤ a) (h₂ : c ≤ b) : a - (b - c) = a - b + c :=
  Contravariant.AddLECancellable.tsub_tsub_assoc h₁ h₂


theorem tsub_add_tsub_comm (hba : b ≤ a) (hdc : d ≤ c) : a - b + (c - d) = a + c - (b + d) :=
  Contravariant.AddLECancellable.tsub_add_tsub_comm Contravariant.AddLECancellable hba hdc


theorem le_tsub_iff_left (h : a ≤ c) : b ≤ c - a ↔ a + b ≤ c :=
  Contravariant.AddLECancellable.le_tsub_iff_left h


theorem le_tsub_iff_right (h : a ≤ c) : b ≤ c - a ↔ b + a ≤ c :=
  Contravariant.AddLECancellable.le_tsub_iff_right h


theorem tsub_lt_iff_left (hbc : b ≤ a) : a - b < c ↔ a < b + c :=
  Contravariant.AddLECancellable.tsub_lt_iff_left hbc


theorem tsub_lt_iff_right (hbc : b ≤ a) : a - b < c ↔ a < c + b :=
  Contravariant.AddLECancellable.tsub_lt_iff_right hbc


theorem tsub_lt_iff_tsub_lt (h₁ : b ≤ a) (h₂ : c ≤ a) : a - b < c ↔ a - c < b :=
  Contravariant.AddLECancellable.tsub_lt_iff_tsub_lt Contravariant.AddLECancellable h₁ h₂


theorem le_tsub_iff_le_tsub (h₁ : a ≤ b) (h₂ : c ≤ b) : a ≤ b - c ↔ c ≤ b - a :=
  Contravariant.AddLECancellable.le_tsub_iff_le_tsub Contravariant.AddLECancellable h₁ h₂


/-- See `lt_tsub_iff_right` for a stronger statement in a linear order. -/
theorem lt_tsub_iff_right_of_le (h : c ≤ b) : a < b - c ↔ a + c < b :=
  Contravariant.AddLECancellable.lt_tsub_iff_right_of_le h


/-- See `lt_tsub_iff_left` for a stronger statement in a linear order. -/
theorem lt_tsub_iff_left_of_le (h : c ≤ b) : a < b - c ↔ c + a < b :=
  Contravariant.AddLECancellable.lt_tsub_iff_left_of_le h


/-- See `lt_of_tsub_lt_tsub_left` for a stronger statement in a linear order. -/
theorem lt_of_tsub_lt_tsub_left_of_le [AddLeftReflectLT α] (hca : c ≤ a)
    (h : a - b < a - c) : c < b :=
  Contravariant.AddLECancellable.lt_of_tsub_lt_tsub_left_of_le hca h


theorem tsub_lt_tsub_left_of_le : b ≤ a → c < b → a - b < a - c :=
  Contravariant.AddLECancellable.tsub_lt_tsub_left_of_le


theorem tsub_lt_tsub_right_of_le (h : c ≤ a) (h2 : a < b) : a - c < b - c :=
  Contravariant.AddLECancellable.tsub_lt_tsub_right_of_le h h2


theorem tsub_inj_right (h₁ : b ≤ a) (h₂ : c ≤ a) (h₃ : a - b = a - c) : b = c :=
  Contravariant.AddLECancellable.tsub_inj_right h₁ h₂ h₃


/-- See `tsub_lt_tsub_iff_left_of_le` for a stronger statement in a linear order. -/
theorem tsub_lt_tsub_iff_left_of_le_of_le [AddLeftReflectLT α] (h₁ : b ≤ a)
    (h₂ : c ≤ a) : a - b < a - c ↔ c < b :=
  Contravariant.AddLECancellable.tsub_lt_tsub_iff_left_of_le_of_le Contravariant.AddLECancellable h₁
    h₂


@[simp]
theorem add_tsub_tsub_cancel (h : c ≤ a) : a + b - (a - c) = b + c :=
  Contravariant.AddLECancellable.add_tsub_tsub_cancel h


/-- See `tsub_tsub_le` for an inequality. -/
theorem tsub_tsub_cancel_of_le (h : a ≤ b) : b - (b - a) = a :=
  Contravariant.AddLECancellable.tsub_tsub_cancel_of_le h


theorem tsub_tsub_tsub_cancel_left (h : b ≤ a) : a - c - (a - b) = b - c :=
  Contravariant.AddLECancellable.tsub_tsub_tsub_cancel_left h

-- note: not generalized to `AddLECancellable` because `add_tsub_add_eq_tsub_left` isn't

/-- The `tsub` version of `sub_sub_eq_add_sub`. -/
theorem tsub_tsub_eq_add_tsub_of_le
    (h : c ≤ b) : a - (b - c) = a + c - b := by
  /-
    α : Type u_1
    inst✝⁶ : AddCommSemigroup α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : ExistsAddOfLE α
    inst✝³ : AddLeftMono α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftReflectLE α
    h : LE.le c b
    ⊢ Eq (HSub.hSub a (HSub.hSub b c)) (HSub.hSub (HAdd.hAdd a c) b)
  -/
  obtain ⟨d, rfl⟩ := exists_add_of_le h
  /-
    case intro
    α : Type u_1
    inst✝⁶ : AddCommSemigroup α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : ExistsAddOfLE α
    inst✝³ : AddLeftMono α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a c : α
    inst✝ : AddLeftReflectLE α
    d : α
    h : LE.le c (HAdd.hAdd c d)
    ⊢ Eq (HSub.hSub a (HSub.hSub (HAdd.hAdd c d) c)) (HSub.hSub (HAdd.hAdd a c) (H …
  -/
  rw [add_tsub_cancel_left c, add_comm a c, add_tsub_add_eq_tsub_left]
  /-
    🎉 no goals
  -/


