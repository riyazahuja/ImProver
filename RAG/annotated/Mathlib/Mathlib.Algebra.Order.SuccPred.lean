/-- A typeclass for `succ x = x + 1`. -/
class SuccAddOrder (α : Type*) [Preorder α] [Add α] [One α] extends SuccOrder α where
  succ_eq_add_one (x : α) : succ x = x + 1


/-- A typeclass for `pred x = x - 1`. -/
class PredSubOrder (α : Type*) [Preorder α] [Sub α] [One α] extends PredOrder α where
  pred_eq_sub_one (x : α) : pred x = x - 1


theorem succ_eq_add_one (x : α) : succ x = x + 1 :=
  SuccAddOrder.succ_eq_add_one x


theorem add_one_le_of_lt (h : x < y) : x + 1 ≤ y := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : Preorder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    h : LT.lt x y
    ⊢ LE.le (HAdd.hAdd x 1) y
  -/
  rw [← succ_eq_add_one]
  /-
    α : Type u_1
    x y : α
    inst✝³ : Preorder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    h : LT.lt x y
    ⊢ LE.le (Order.succ x) y
  -/
  exact succ_le_of_lt h
  /-
    🎉 no goals
  -/


theorem add_one_le_iff_of_not_isMax (hx : ¬ IsMax x) : x + 1 ≤ y ↔ x < y := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : Preorder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    hx : Not (IsMax x)
    ⊢ Iff (LE.le (HAdd.hAdd x 1) y) (LT.lt x y)
  -/
  rw [← succ_eq_add_one, succ_le_iff_of_not_isMax hx]
  /-
    🎉 no goals
  -/


theorem add_one_le_iff [NoMaxOrder α] : x + 1 ≤ y ↔ x < y :=
  add_one_le_iff_of_not_isMax (not_isMax x)


@[simp]
theorem wcovBy_add_one (x : α) : x ⩿ x + 1 := by
  /-
    α : Type u_1
    inst✝³ : Preorder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    x : α
    ⊢ WCovBy x (HAdd.hAdd x 1)
  -/
  rw [← succ_eq_add_one]
  /-
    α : Type u_1
    inst✝³ : Preorder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    x : α
    ⊢ WCovBy x (Order.succ x)
  -/
  exact wcovBy_succ x
  /-
    🎉 no goals
  -/


@[simp]
theorem covBy_add_one [NoMaxOrder α] (x : α) : x ⋖ x + 1 := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : NoMaxOrder α
    x : α
    ⊢ CovBy x (HAdd.hAdd x 1)
  -/
  rw [← succ_eq_add_one]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : NoMaxOrder α
    x : α
    ⊢ CovBy x (Order.succ x)
  -/
  exact covBy_succ x
  /-
    🎉 no goals
  -/


theorem pred_eq_sub_one (x : α) : pred x = x - 1 :=
  PredSubOrder.pred_eq_sub_one x


theorem le_sub_one_of_lt (h : x < y) : x ≤ y - 1 := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : Preorder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    h : LT.lt x y
    ⊢ LE.le x (HSub.hSub y 1)
  -/
  rw [← pred_eq_sub_one]
  /-
    α : Type u_1
    x y : α
    inst✝³ : Preorder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    h : LT.lt x y
    ⊢ LE.le x (Order.pred y)
  -/
  exact le_pred_of_lt h
  /-
    🎉 no goals
  -/


theorem le_sub_one_iff_of_not_isMin (hy : ¬ IsMin y) : x ≤ y - 1 ↔ x < y := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : Preorder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    hy : Not (IsMin y)
    ⊢ Iff (LE.le x (HSub.hSub y 1)) (LT.lt x y)
  -/
  rw [← pred_eq_sub_one, le_pred_iff_of_not_isMin hy]
  /-
    🎉 no goals
  -/


theorem le_sub_one_iff [NoMinOrder α] : x ≤ y - 1 ↔ x < y :=
  le_sub_one_iff_of_not_isMin (not_isMin y)


@[simp]
theorem sub_one_wcovBy (x : α) : x - 1 ⩿ x := by
  /-
    α : Type u_1
    inst✝³ : Preorder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    x : α
    ⊢ WCovBy (HSub.hSub x 1) x
  -/
  rw [← pred_eq_sub_one]
  /-
    α : Type u_1
    inst✝³ : Preorder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    x : α
    ⊢ WCovBy (Order.pred x) x
  -/
  exact pred_wcovBy x
  /-
    🎉 no goals
  -/


@[simp]
theorem sub_one_covBy [NoMinOrder α] (x : α) : x - 1 ⋖ x := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : NoMinOrder α
    x : α
    ⊢ CovBy (HSub.hSub x 1) x
  -/
  rw [← pred_eq_sub_one]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : NoMinOrder α
    x : α
    ⊢ CovBy (Order.pred x) x
  -/
  exact pred_covBy x
  /-
    🎉 no goals
  -/


@[simp]
theorem succ_iterate [AddMonoidWithOne α] [SuccAddOrder α] (x : α) (n : ℕ) :
    succ^[n] x = x + n := by
  induction n with
  | zero =>
    rw [Function.iterate_zero_apply, Nat.cast_zero, add_zero]
  | succ n IH =>
    rw [Function.iterate_succ_apply', IH, Nat.cast_add, succ_eq_add_one, Nat.cast_one, add_assoc]


@[simp]
theorem pred_iterate [AddCommGroupWithOne α] [PredSubOrder α] (x : α) (n : ℕ) :
    pred^[n] x = x - n := by
  induction n with
  | zero =>
    rw [Function.iterate_zero_apply, Nat.cast_zero, sub_zero]
  | succ n IH =>
    rw [Function.iterate_succ_apply', IH, Nat.cast_add, pred_eq_sub_one, Nat.cast_one, sub_sub]


theorem not_isMax_zero [Zero α] [One α] [ZeroLEOneClass α] [NeZero (1 : α)] : ¬ IsMax (0 : α) := by
  /-
    α : Type u_1
    inst✝⁴ : PartialOrder α
    inst✝³ : Zero α
    inst✝² : One α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : NeZero 1
    ⊢ Not (IsMax 0)
  -/
  rw [not_isMax_iff]
  /-
    α : Type u_1
    inst✝⁴ : PartialOrder α
    inst✝³ : Zero α
    inst✝² : One α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : NeZero 1
    ⊢ Exists fun b => LT.lt 0 b
  -/
  exact ⟨1, one_pos⟩
  /-
    🎉 no goals
  -/


theorem one_le_iff_pos [AddMonoidWithOne α] [ZeroLEOneClass α] [NeZero (1 : α)]
    [SuccAddOrder α] : 1 ≤ x ↔ 0 < x := by
  /-
    α : Type u_1
    x : α
    inst✝⁴ : PartialOrder α
    inst✝³ : AddMonoidWithOne α
    inst✝² : ZeroLEOneClass α
    inst✝¹ : NeZero 1
    inst✝ : SuccAddOrder α
    ⊢ Iff (LE.le 1 x) (LT.lt 0 x)
  -/
  rw [← succ_le_iff_of_not_isMax not_isMax_zero, succ_eq_add_one, zero_add]
  /-
    🎉 no goals
  -/


theorem covBy_iff_add_one_eq [Add α] [One α] [SuccAddOrder α] [NoMaxOrder α] :
    x ⋖ y ↔ x + 1 = y := by
  /-
    α : Type u_1
    x y : α
    inst✝⁴ : PartialOrder α
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : NoMaxOrder α
    ⊢ Iff (CovBy x y) (Eq (HAdd.hAdd x 1) y)
  -/
  rw [← succ_eq_add_one]
  /-
    α : Type u_1
    x y : α
    inst✝⁴ : PartialOrder α
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : NoMaxOrder α
    ⊢ Iff (CovBy x y) (Eq (Order.succ x) y)
  -/
  exact succ_eq_iff_covBy.symm
  /-
    🎉 no goals
  -/


theorem covBy_iff_sub_one_eq [Sub α] [One α] [PredSubOrder α] [NoMinOrder α] :
    x ⋖ y ↔ y - 1 = x := by
  /-
    α : Type u_1
    x y : α
    inst✝⁴ : PartialOrder α
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : NoMinOrder α
    ⊢ Iff (CovBy x y) (Eq (HSub.hSub y 1) x)
  -/
  rw [← pred_eq_sub_one]
  /-
    α : Type u_1
    x y : α
    inst✝⁴ : PartialOrder α
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : NoMinOrder α
    ⊢ Iff (CovBy x y) (Eq (Order.pred y) x)
  -/
  exact pred_eq_iff_covBy.symm
  /-
    🎉 no goals
  -/


theorem IsSuccPrelimit.add_one_lt [Add α] [One α] [SuccAddOrder α]
    (hx : IsSuccPrelimit x) (hy : y < x) : y + 1 < x := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : PartialOrder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    hx : Order.IsSuccPrelimit x
    hy : LT.lt y x
    ⊢ LT.lt (HAdd.hAdd y 1) x
  -/
  rw [← succ_eq_add_one]
  /-
    α : Type u_1
    x y : α
    inst✝³ : PartialOrder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    hx : Order.IsSuccPrelimit x
    hy : LT.lt y x
    ⊢ LT.lt (Order.succ y) x
  -/
  exact hx.succ_lt hy
  /-
    🎉 no goals
  -/


theorem IsPredPrelimit.lt_sub_one [Sub α] [One α] [PredSubOrder α]
    (hx : IsPredPrelimit x) (hy : x < y) : x < y - 1 := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : PartialOrder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    hx : Order.IsPredPrelimit x
    hy : LT.lt x y
    ⊢ LT.lt x (HSub.hSub y 1)
  -/
  rw [← pred_eq_sub_one]
  /-
    α : Type u_1
    x y : α
    inst✝³ : PartialOrder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    hx : Order.IsPredPrelimit x
    hy : LT.lt x y
    ⊢ LT.lt x (Order.pred y)
  -/
  exact hx.lt_pred hy
  /-
    🎉 no goals
  -/


theorem IsSuccLimit.add_one_lt [Add α] [One α] [SuccAddOrder α]
    (hx : IsSuccLimit x) (hy : y < x) : y + 1 < x :=
  hx.isSuccPrelimit.add_one_lt hy


theorem IsPredLimit.lt_sub_one [Sub α] [One α] [PredSubOrder α]
    (hx : IsPredLimit x) (hy : x < y) : x < y - 1 :=
  hx.isPredPrelimit.lt_sub_one hy


theorem IsSuccPrelimit.add_natCast_lt [AddMonoidWithOne α] [SuccAddOrder α]
    (hx : IsSuccPrelimit x) (hy : y < x) : ∀ n : ℕ, y + n < x
            /-
              α : Type u_1
              x y : α
              inst✝² : PartialOrder α
              inst✝¹ : AddMonoidWithOne α
              inst✝ : SuccAddOrder α
              hx : Order.IsSuccPrelimit x
              hy : LT.lt y x
              ⊢ LT.lt (HAdd.hAdd y ↑0) x
            -/
  | 0 => by simpa
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      α : Type u_1
      x y : α
      inst✝² : PartialOrder α
      inst✝¹ : AddMonoidWithOne α
      inst✝ : SuccAddOrder α
      hx : Order.IsSuccPrelimit x
      hy : LT.lt y x
      n : Nat
      ⊢ LT.lt (HAdd.hAdd y ↑(HAdd.hAdd n 1)) x
    -/
    rw [Nat.cast_add_one, ← add_assoc]
    /-
      α : Type u_1
      x y : α
      inst✝² : PartialOrder α
      inst✝¹ : AddMonoidWithOne α
      inst✝ : SuccAddOrder α
      hx : Order.IsSuccPrelimit x
      hy : LT.lt y x
      n : Nat
      ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd y ↑n) 1) x
    -/
    exact hx.add_one_lt (hx.add_natCast_lt hy n)
    /-
      🎉 no goals
    -/


theorem IsPredPrelimit.lt_sub_natCast [AddCommGroupWithOne α] [PredSubOrder α]
    (hx : IsPredPrelimit x) (hy : x < y) : ∀ n : ℕ, x < y - n
            /-
              α : Type u_1
              x y : α
              inst✝² : PartialOrder α
              inst✝¹ : AddCommGroupWithOne α
              inst✝ : PredSubOrder α
              hx : Order.IsPredPrelimit x
              hy : LT.lt x y
              ⊢ LT.lt x (HSub.hSub y ↑0)
            -/
  | 0 => by simpa
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      α : Type u_1
      x y : α
      inst✝² : PartialOrder α
      inst✝¹ : AddCommGroupWithOne α
      inst✝ : PredSubOrder α
      hx : Order.IsPredPrelimit x
      hy : LT.lt x y
      n : Nat
      ⊢ LT.lt x (HSub.hSub y ↑(HAdd.hAdd n 1))
    -/
    rw [Nat.cast_add_one, ← sub_sub]
    /-
      α : Type u_1
      x y : α
      inst✝² : PartialOrder α
      inst✝¹ : AddCommGroupWithOne α
      inst✝ : PredSubOrder α
      hx : Order.IsPredPrelimit x
      hy : LT.lt x y
      n : Nat
      ⊢ LT.lt x (HSub.hSub (HSub.hSub y ↑n) 1)
    -/
    exact hx.lt_sub_one (hx.lt_sub_natCast hy n)
    /-
      🎉 no goals
    -/


theorem IsSuccLimit.add_natCast_lt [AddMonoidWithOne α] [SuccAddOrder α]
    (hx : IsSuccLimit x) (hy : y < x) : ∀ n : ℕ, y + n < x :=
  hx.isSuccPrelimit.add_natCast_lt hy


theorem IsPredLimit.lt_sub_natCast [AddCommGroupWithOne α] [PredSubOrder α]
    (hx : IsPredLimit x) (hy : x < y) : ∀ n : ℕ, x < y - n :=
  hx.isPredPrelimit.lt_sub_natCast hy


theorem le_of_lt_add_one (h : x < y + 1) : x ≤ y := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : LinearOrder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    h : LT.lt x (HAdd.hAdd y 1)
    ⊢ LE.le x y
  -/
  rw [← succ_eq_add_one] at h
  /-
    α : Type u_1
    x y : α
    inst✝³ : LinearOrder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    h : LT.lt x (Order.succ y)
    ⊢ LE.le x y
  -/
  exact le_of_lt_succ h
  /-
    🎉 no goals
  -/


theorem lt_add_one_iff_of_not_isMax (hy : ¬ IsMax y) : x < y + 1 ↔ x ≤ y := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : LinearOrder α
    inst✝² : Add α
    inst✝¹ : One α
    inst✝ : SuccAddOrder α
    hy : Not (IsMax y)
    ⊢ Iff (LT.lt x (HAdd.hAdd y 1)) (LE.le x y)
  -/
  rw [← succ_eq_add_one, lt_succ_iff_of_not_isMax hy]
  /-
    🎉 no goals
  -/


theorem lt_add_one_iff [NoMaxOrder α] : x < y + 1 ↔ x ≤ y :=
  lt_add_one_iff_of_not_isMax (not_isMax y)


theorem le_of_sub_one_lt (h : x - 1 < y) : x ≤ y := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : LinearOrder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    h : LT.lt (HSub.hSub x 1) y
    ⊢ LE.le x y
  -/
  rw [← pred_eq_sub_one] at h
  /-
    α : Type u_1
    x y : α
    inst✝³ : LinearOrder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    h : LT.lt (Order.pred x) y
    ⊢ LE.le x y
  -/
  exact le_of_pred_lt h
  /-
    🎉 no goals
  -/


theorem sub_one_lt_iff_of_not_isMin (hx : ¬ IsMin x) : x - 1 < y ↔ x ≤ y := by
  /-
    α : Type u_1
    x y : α
    inst✝³ : LinearOrder α
    inst✝² : Sub α
    inst✝¹ : One α
    inst✝ : PredSubOrder α
    hx : Not (IsMin x)
    ⊢ Iff (LT.lt (HSub.hSub x 1) y) (LE.le x y)
  -/
  rw [← pred_eq_sub_one, pred_lt_iff_of_not_isMin hx]
  /-
    🎉 no goals
  -/


theorem sub_one_lt_iff [NoMinOrder α] : x - 1 < y ↔ x ≤ y :=
  sub_one_lt_iff_of_not_isMin (not_isMin x)


theorem lt_one_iff_nonpos [AddMonoidWithOne α] [ZeroLEOneClass α] [NeZero (1 : α)]
    [SuccAddOrder α] : x < 1 ↔ x ≤ 0 := by
  /-
    α : Type u_1
    x : α
    inst✝⁴ : LinearOrder α
    inst✝³ : AddMonoidWithOne α
    inst✝² : ZeroLEOneClass α
    inst✝¹ : NeZero 1
    inst✝ : SuccAddOrder α
    ⊢ Iff (LT.lt x 1) (LE.le x 0)
  -/
  rw [← lt_succ_iff_of_not_isMax not_isMax_zero, succ_eq_add_one, zero_add]
  /-
    🎉 no goals
  -/


lemma monotoneOn_of_le_add_one (hs : s.OrdConnected) :
    (∀ a, ¬ IsMax a → a ∈ s → a + 1 ∈ s → f a ≤ f (a + 1)) → MonotoneOn f s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    ⊢ (∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (HAdd.hAdd …
  -/
  simpa [Order.succ_eq_add_one] using monotoneOn_of_le_succ hs (f := f)
  /-
    🎉 no goals
  -/


lemma antitoneOn_of_add_one_le (hs : s.OrdConnected) :
    (∀ a, ¬ IsMax a → a ∈ s → a + 1 ∈ s → f (a + 1) ≤ f a) → AntitoneOn f s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    ⊢ (∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (HAdd.hAdd …
  -/
  simpa [Order.succ_eq_add_one] using antitoneOn_of_succ_le hs (f := f)
  /-
    🎉 no goals
  -/


lemma strictMonoOn_of_lt_add_one (hs : s.OrdConnected) :
    (∀ a, ¬ IsMax a → a ∈ s → a + 1 ∈ s → f a < f (a + 1)) → StrictMonoOn f s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    ⊢ (∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (HAdd.hAdd …
  -/
  simpa [Order.succ_eq_add_one] using strictMonoOn_of_lt_succ hs (f := f)
  /-
    🎉 no goals
  -/


lemma strictAntiOn_of_add_one_lt (hs : s.OrdConnected) :
    (∀ a, ¬ IsMax a → a ∈ s → a + 1 ∈ s → f (a + 1) < f a) → StrictAntiOn f s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    ⊢ (∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (HAdd.hAdd …
  -/
  simpa [Order.succ_eq_add_one] using strictAntiOn_of_succ_lt hs (f := f)
  /-
    🎉 no goals
  -/


lemma monotone_of_le_add_one : (∀ a, ¬ IsMax a → f a ≤ f (a + 1)) → Monotone f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : IsSuccArchimedean α
    f : α → β
    ⊢ (∀ (a : α), Not (IsMax a) → LE.le (f a) (f (HAdd.hAdd a 1))) → Monotone f
  -/
  simpa [Order.succ_eq_add_one] using monotone_of_le_succ (f := f)
  /-
    🎉 no goals
  -/


lemma antitone_of_add_one_le : (∀ a, ¬ IsMax a → f (a + 1) ≤ f a) → Antitone f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : IsSuccArchimedean α
    f : α → β
    ⊢ (∀ (a : α), Not (IsMax a) → LE.le (f (HAdd.hAdd a 1)) (f a)) → Antitone f
  -/
  simpa [Order.succ_eq_add_one] using antitone_of_succ_le (f := f)
  /-
    🎉 no goals
  -/


lemma strictMono_of_lt_add_one : (∀ a, ¬ IsMax a → f a < f (a + 1)) → StrictMono f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : IsSuccArchimedean α
    f : α → β
    ⊢ (∀ (a : α), Not (IsMax a) → LT.lt (f a) (f (HAdd.hAdd a 1))) → StrictMono f
  -/
  simpa [Order.succ_eq_add_one] using strictMono_of_lt_succ (f := f)
  /-
    🎉 no goals
  -/


lemma strictAnti_of_add_one_lt : (∀ a, ¬ IsMax a → f (a + 1) < f a) → StrictAnti f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Add α
    inst✝² : One α
    inst✝¹ : SuccAddOrder α
    inst✝ : IsSuccArchimedean α
    f : α → β
    ⊢ (∀ (a : α), Not (IsMax a) → LT.lt (f (HAdd.hAdd a 1)) (f a)) → StrictAnti f
  -/
  simpa [Order.succ_eq_add_one] using strictAnti_of_succ_lt (f := f)
  /-
    🎉 no goals
  -/


lemma monotoneOn_of_sub_one_le (hs : s.OrdConnected) :
    (∀ a, ¬ IsMin a → a ∈ s → a - 1 ∈ s → f (a - 1) ≤ f a) → MonotoneOn f s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    ⊢ (∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (HSub.hSub …
  -/
  simpa [Order.pred_eq_sub_one] using monotoneOn_of_pred_le hs (f := f)
  /-
    🎉 no goals
  -/


lemma antitoneOn_of_le_sub_one (hs : s.OrdConnected) :
    (∀ a, ¬ IsMin a → a ∈ s → a - 1 ∈ s → f a ≤ f (a - 1)) → AntitoneOn f s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    ⊢ (∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (HSub.hSub …
  -/
  simpa [Order.pred_eq_sub_one] using antitoneOn_of_le_pred hs (f := f)
  /-
    🎉 no goals
  -/


lemma strictMonoOn_of_sub_one_lt (hs : s.OrdConnected) :
    (∀ a, ¬ IsMin a → a ∈ s → a - 1 ∈ s → f (a - 1) < f a) → StrictMonoOn f s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    ⊢ (∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (HSub.hSub …
  -/
  simpa [Order.pred_eq_sub_one] using strictMonoOn_of_pred_lt hs (f := f)
  /-
    🎉 no goals
  -/


lemma strictAntiOn_of_lt_sub_one (hs : s.OrdConnected) :
    (∀ a, ¬ IsMin a → a ∈ s → a - 1 ∈ s → f a < f (a - 1)) → StrictAntiOn f s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    ⊢ (∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (HSub.hSub …
  -/
  simpa [Order.pred_eq_sub_one] using strictAntiOn_of_lt_pred hs (f := f)
  /-
    🎉 no goals
  -/


lemma monotone_of_sub_one_le : (∀ a, ¬ IsMin a → f (a - 1) ≤ f a) → Monotone f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : IsPredArchimedean α
    f : α → β
    ⊢ (∀ (a : α), Not (IsMin a) → LE.le (f (HSub.hSub a 1)) (f a)) → Monotone f
  -/
  simpa [Order.pred_eq_sub_one] using monotone_of_pred_le (f := f)
  /-
    🎉 no goals
  -/


lemma antitone_of_le_sub_one : (∀ a, ¬ IsMin a → f a ≤ f (a - 1)) → Antitone f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : IsPredArchimedean α
    f : α → β
    ⊢ (∀ (a : α), Not (IsMin a) → LE.le (f a) (f (HSub.hSub a 1))) → Antitone f
  -/
  simpa [Order.pred_eq_sub_one] using antitone_of_le_pred (f := f)
  /-
    🎉 no goals
  -/


lemma strictMono_of_sub_one_lt : (∀ a, ¬ IsMin a → f (a - 1) < f a) → StrictMono f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : IsPredArchimedean α
    f : α → β
    ⊢ (∀ (a : α), Not (IsMin a) → LT.lt (f (HSub.hSub a 1)) (f a)) → StrictMono f
  -/
  simpa [Order.pred_eq_sub_one] using strictMono_of_pred_lt (f := f)
  /-
    🎉 no goals
  -/


lemma strictAnti_of_lt_sub_one : (∀ a, ¬ IsMin a → f a < f (a - 1)) → StrictAnti f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁵ : PartialOrder α
    inst✝⁴ : Preorder β
    inst✝³ : Sub α
    inst✝² : One α
    inst✝¹ : PredSubOrder α
    inst✝ : IsPredArchimedean α
    f : α → β
    ⊢ (∀ (a : α), Not (IsMin a) → LT.lt (f a) (f (HSub.hSub a 1))) → StrictAnti f
  -/
  simpa [Order.pred_eq_sub_one] using strictAnti_of_lt_pred (f := f)
  /-
    🎉 no goals
  -/


