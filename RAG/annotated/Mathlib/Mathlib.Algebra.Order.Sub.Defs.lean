/-- `OrderedSub α` means that `α` has a subtraction characterized by `a - b ≤ c ↔ a ≤ c + b`.
In other words, `a - b` is the least `c` such that `a ≤ b + c`.

This is satisfied both by the subtraction in additive ordered groups and by truncated subtraction
in canonically ordered monoids on many specific types.
-/
class OrderedSub (α : Type*) [LE α] [Add α] [Sub α] : Prop where
  /-- `a - b` provides a lower bound on `c` such that `a ≤ c + b`. -/
  tsub_le_iff_right : ∀ a b c : α, a - b ≤ c ↔ a ≤ c + b


@[simp]
theorem tsub_le_iff_right [LE α] [Add α] [Sub α] [OrderedSub α] {a b c : α} :
    a - b ≤ c ↔ a ≤ c + b :=
  OrderedSub.tsub_le_iff_right a b c


/-- See `add_tsub_cancel_right` for the equality if `AddLeftReflectLE α`. -/
theorem add_tsub_le_right : a + b - b ≤ a :=
  tsub_le_iff_right.mpr le_rfl


theorem le_tsub_add : b ≤ b - a + a :=
  tsub_le_iff_right.mp le_rfl


                                                       /-
                                                         α : Type u_1
                                                         inst✝³ : Preorder α
                                                         inst✝² : AddCommSemigroup α
                                                         inst✝¹ : Sub α
                                                         inst✝ : OrderedSub α
                                                         a b c : α
                                                         ⊢ Iff (LE.le (HSub.hSub a b) c) (LE.le a (HAdd.hAdd b c))
                                                       -/
theorem tsub_le_iff_left : a - b ≤ c ↔ a ≤ b + c := by rw [tsub_le_iff_right, add_comm]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem le_add_tsub : a ≤ b + (a - b) :=
  tsub_le_iff_left.mp le_rfl


/-- See `add_tsub_cancel_left` for the equality if `AddLeftReflectLE α`. -/
theorem add_tsub_le_left : a + b - a ≤ b :=
  tsub_le_iff_left.mpr le_rfl


@[gcongr] theorem tsub_le_tsub_right (h : a ≤ b) (c : α) : a - c ≤ b - c :=
  tsub_le_iff_left.mpr <| h.trans le_add_tsub


                                                          /-
                                                            α : Type u_1
                                                            inst✝³ : Preorder α
                                                            inst✝² : AddCommSemigroup α
                                                            inst✝¹ : Sub α
                                                            inst✝ : OrderedSub α
                                                            a b c : α
                                                            ⊢ Iff (LE.le (HSub.hSub a b) c) (LE.le (HSub.hSub a c) b)
                                                          -/
theorem tsub_le_iff_tsub_le : a - b ≤ c ↔ a - c ≤ b := by rw [tsub_le_iff_left, tsub_le_iff_right]
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- See `tsub_tsub_cancel_of_le` for the equality. -/
theorem tsub_tsub_le : b - (b - a) ≤ a :=
  tsub_le_iff_right.mpr le_add_tsub


@[gcongr] theorem tsub_le_tsub_left (h : a ≤ b) (c : α) : c - b ≤ c - a :=
  tsub_le_iff_left.mpr <| le_add_tsub.trans <| add_le_add_right h _


@[gcongr] theorem tsub_le_tsub (hab : a ≤ b) (hcd : c ≤ d) : a - d ≤ b - c :=
  (tsub_le_tsub_right hab _).trans <| tsub_le_tsub_left hcd _


theorem antitone_const_tsub : Antitone fun x => c - x := fun _ _ hxy => tsub_le_tsub rfl.le hxy


/-- See `add_tsub_assoc_of_le` for the equality. -/
theorem add_tsub_le_assoc : a + b - c ≤ a + (b - c) := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HAdd.hAdd a b) c) (HAdd.hAdd a (HSub.hSub b c))
  -/
  rw [tsub_le_iff_left, add_left_comm]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd a (HAdd.hAdd c (HSub.hSub b c)))
  -/
  exact add_le_add_left le_add_tsub a
  /-
    🎉 no goals
  -/


/-- See `tsub_add_eq_add_tsub` for the equality. -/
theorem add_tsub_le_tsub_add : a + b - c ≤ a - c + b := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HAdd.hAdd a b) c) (HAdd.hAdd (HSub.hSub a c) b)
  -/
  rw [add_comm, add_comm _ b]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HAdd.hAdd b a) c) (HAdd.hAdd b (HSub.hSub a c))
  -/
  exact add_tsub_le_assoc
  /-
    🎉 no goals
  -/


theorem add_le_add_add_tsub : a + b ≤ a + c + (b - c) := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd (HAdd.hAdd a c) (HSub.hSub b c))
  -/
  rw [add_assoc]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd a (HAdd.hAdd c (HSub.hSub b c)))
  -/
  exact add_le_add_left le_add_tsub a
  /-
    🎉 no goals
  -/


theorem le_tsub_add_add : a + b ≤ a - c + (b + c) := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd (HSub.hSub a c) (HAdd.hAdd b c))
  -/
  rw [add_comm a, add_comm (a - c)]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HAdd.hAdd b a) (HAdd.hAdd (HAdd.hAdd b c) (HSub.hSub a c))
  -/
  exact add_le_add_add_tsub
  /-
    🎉 no goals
  -/


theorem tsub_le_tsub_add_tsub : a - c ≤ a - b + (b - c) := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub a c) (HAdd.hAdd (HSub.hSub a b) (HSub.hSub b c))
  -/
  rw [tsub_le_iff_left, ← add_assoc, add_right_comm]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le a (HAdd.hAdd (HAdd.hAdd c (HSub.hSub b c)) (HSub.hSub a b))
  -/
  exact le_add_tsub.trans (add_le_add_right le_add_tsub _)
  /-
    🎉 no goals
  -/


theorem tsub_tsub_tsub_le_tsub : c - a - (c - b) ≤ b - a := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HSub.hSub c a) (HSub.hSub c b)) (HSub.hSub b a)
  -/
  rw [tsub_le_iff_left, tsub_le_iff_left, add_left_comm]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le c (HAdd.hAdd (HSub.hSub c b) (HAdd.hAdd a (HSub.hSub b a)))
  -/
  exact le_tsub_add.trans (add_le_add_left le_add_tsub _)
  /-
    🎉 no goals
  -/


theorem tsub_tsub_le_tsub_add {a b c : α} : a - (b - c) ≤ a - b + c :=
  tsub_le_iff_right.2 <|
    calc
      a ≤ a - b + b := le_tsub_add
      _ ≤ a - b + (c + (b - c)) := add_le_add_left le_add_tsub _
      _ = a - b + c + (b - c) := (add_assoc _ _ _).symm


/-- See `tsub_add_tsub_comm` for the equality. -/
theorem add_tsub_add_le_tsub_add_tsub : a + b - (c + d) ≤ a - c + (b - d) := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c d : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HAdd.hAdd a b) (HAdd.hAdd c d)) (HAdd.hAdd (HSub.hSub a c) …
  -/
  rw [add_comm c, tsub_le_iff_left, add_assoc, ← tsub_le_iff_left, ← tsub_le_iff_left]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c d : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HSub.hSub (HAdd.hAdd a b) d) c) (HAdd.hAdd (HSub.hSub a c) …
  -/
  refine (tsub_le_tsub_right add_tsub_le_assoc c).trans ?_
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c d : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HAdd.hAdd a (HSub.hSub b d)) c) (HAdd.hAdd (HSub.hSub a c) …
  -/
  rw [add_comm a, add_comm (a - c)]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c d : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (HSub.hSub b d) a) c) (HAdd.hAdd (HSub.hSub b d) …
  -/
  exact add_tsub_le_assoc
  /-
    🎉 no goals
  -/


/-- See `add_tsub_add_eq_tsub_left` for the equality. -/
theorem add_tsub_add_le_tsub_left : a + b - (a + c) ≤ b - c := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HAdd.hAdd a b) (HAdd.hAdd a c)) (HSub.hSub b c)
  -/
  rw [tsub_le_iff_left, add_assoc]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd a (HAdd.hAdd c (HSub.hSub b c)))
  -/
  exact add_le_add_left le_add_tsub _
  /-
    🎉 no goals
  -/


/-- See `add_tsub_add_eq_tsub_right` for the equality. -/
theorem add_tsub_add_le_tsub_right : a + c - (b + c) ≤ a - b := by
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HSub.hSub (HAdd.hAdd a c) (HAdd.hAdd b c)) (HSub.hSub a b)
  -/
  rw [tsub_le_iff_left, add_right_comm]
  /-
    α : Type u_1
    inst✝⁴ : Preorder α
    inst✝³ : AddCommSemigroup α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    a b c : α
    inst✝ : AddLeftMono α
    ⊢ LE.le (HAdd.hAdd a c) (HAdd.hAdd (HAdd.hAdd b (HSub.hSub a b)) c)
  -/
  exact add_le_add_right le_add_tsub c
  /-
    🎉 no goals
  -/


protected theorem le_add_tsub_swap (hb : AddLECancellable b) : a ≤ b + a - b :=
  hb le_add_tsub


protected theorem le_add_tsub (hb : AddLECancellable b) : a ≤ a + b - b := by
  /-
    α : Type u_1
    inst✝³ : Preorder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b : α
    hb : AddLECancellable b
    ⊢ LE.le a (HSub.hSub (HAdd.hAdd a b) b)
  -/
  rw [add_comm]
  /-
    α : Type u_1
    inst✝³ : Preorder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b : α
    hb : AddLECancellable b
    ⊢ LE.le a (HSub.hSub (HAdd.hAdd b a) b)
  -/
  exact hb.le_add_tsub_swap
  /-
    🎉 no goals
  -/


protected theorem le_tsub_of_add_le_left (ha : AddLECancellable a) (h : a + b ≤ c) : b ≤ c - a :=
  ha <| h.trans le_add_tsub


protected theorem le_tsub_of_add_le_right (hb : AddLECancellable b) (h : a + b ≤ c) : a ≤ c - b :=
                                  /-
                                    α : Type u_1
                                    inst✝³ : Preorder α
                                    inst✝² : AddCommSemigroup α
                                    inst✝¹ : Sub α
                                    inst✝ : OrderedSub α
                                    a b c : α
                                    hb : AddLECancellable b
                                    h : LE.le (HAdd.hAdd a b) c
                                    ⊢ LE.le (HAdd.hAdd b a) c
                                  -/
  hb.le_tsub_of_add_le_left <| by rwa [add_comm]
                                  /-
                                    🎉 no goals
                                  -/


theorem le_add_tsub_swap : a ≤ b + a - b :=
  Contravariant.AddLECancellable.le_add_tsub_swap


theorem le_add_tsub' : a ≤ a + b - b :=
  Contravariant.AddLECancellable.le_add_tsub


theorem le_tsub_of_add_le_left (h : a + b ≤ c) : b ≤ c - a :=
  Contravariant.AddLECancellable.le_tsub_of_add_le_left h


theorem le_tsub_of_add_le_right (h : a + b ≤ c) : a ≤ c - b :=
  Contravariant.AddLECancellable.le_tsub_of_add_le_right h


                                              /-
                                                α : Type u_1
                                                inst✝³ : Preorder α
                                                inst✝² : AddCommMonoid α
                                                inst✝¹ : Sub α
                                                inst✝ : OrderedSub α
                                                a b : α
                                                ⊢ Iff (LE.le (HSub.hSub a b) 0) (LE.le a b)
                                              -/
theorem tsub_nonpos : a - b ≤ 0 ↔ a ≤ b := by rw [tsub_le_iff_left, add_zero]
                                              /-
                                                🎉 no goals
                                              -/


alias ⟨_, tsub_nonpos_of_le⟩ := tsub_nonpos


theorem tsub_tsub (b a c : α) : b - a - c = b - (a + c) := by
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    b a c : α
    ⊢ Eq (HSub.hSub (HSub.hSub b a) c) (HSub.hSub b (HAdd.hAdd a c))
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝³ : PartialOrder α
      inst✝² : AddCommSemigroup α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      b a c : α
      ⊢ LE.le (HSub.hSub (HSub.hSub b a) c) (HSub.hSub b (HAdd.hAdd a c))
    -/
  · rw [tsub_le_iff_left, tsub_le_iff_left, ← add_assoc, ← tsub_le_iff_left]
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝³ : PartialOrder α
      inst✝² : AddCommSemigroup α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      b a c : α
      ⊢ LE.le (HSub.hSub b (HAdd.hAdd a c)) (HSub.hSub (HSub.hSub b a) c)
    -/
  · rw [tsub_le_iff_left, add_assoc, ← tsub_le_iff_left, ← tsub_le_iff_left]
    /-
      🎉 no goals
    -/


theorem tsub_add_eq_tsub_tsub (a b c : α) : a - (b + c) = a - b - c :=
  (tsub_tsub _ _ _).symm


theorem tsub_add_eq_tsub_tsub_swap (a b c : α) : a - (b + c) = a - c - b := by
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    ⊢ Eq (HSub.hSub a (HAdd.hAdd b c)) (HSub.hSub (HSub.hSub a c) b)
  -/
  rw [add_comm]
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    ⊢ Eq (HSub.hSub a (HAdd.hAdd c b)) (HSub.hSub (HSub.hSub a c) b)
  -/
  apply tsub_add_eq_tsub_tsub
  /-
    🎉 no goals
  -/


theorem tsub_right_comm : a - b - c = a - c - b := by
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    ⊢ Eq (HSub.hSub (HSub.hSub a b) c) (HSub.hSub (HSub.hSub a c) b)
  -/
  rw [← tsub_add_eq_tsub_tsub, tsub_add_eq_tsub_tsub_swap]
  /-
    🎉 no goals
  -/


/-- See `AddLECancellable.tsub_eq_of_eq_add'` for a version assuming that `a = c + b` itself is
cancellable rather than `b`. -/
protected theorem tsub_eq_of_eq_add (hb : AddLECancellable b) (h : a = c + b) : a - b = c :=
  le_antisymm (tsub_le_iff_right.mpr h.le) <| by
    /-
      α : Type u_1
      inst✝³ : PartialOrder α
      inst✝² : AddCommSemigroup α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      a b c : α
      hb : AddLECancellable b
      h : Eq a (HAdd.hAdd c b)
      ⊢ LE.le c (HSub.hSub a b)
    -/
    rw [h]
    /-
      α : Type u_1
      inst✝³ : PartialOrder α
      inst✝² : AddCommSemigroup α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      a b c : α
      hb : AddLECancellable b
      h : Eq a (HAdd.hAdd c b)
      ⊢ LE.le c (HSub.hSub (HAdd.hAdd c b) b)
    -/
    exact hb.le_add_tsub
    /-
      🎉 no goals
    -/


/-- Weaker version of `AddLECancellable.tsub_eq_of_eq_add` assuming that `a = c + b` itself is
cancellable rather than `b`. -/
protected lemma tsub_eq_of_eq_add' [AddLeftMono α] (ha : AddLECancellable a)
    (h : a = c + b) : a - b = c := (h ▸ ha).of_add_right.tsub_eq_of_eq_add h


/-- See `AddLECancellable.eq_tsub_of_add_eq'` for a version assuming that `b = a + c` itself is
cancellable rather than `c`. -/
protected theorem eq_tsub_of_add_eq (hc : AddLECancellable c) (h : a + c = b) : a = b - c :=
  (hc.tsub_eq_of_eq_add h.symm).symm


/-- Weaker version of `AddLECancellable.eq_tsub_of_add_eq` assuming that `b = a + c` itself is
cancellable rather than `c`. -/
protected lemma eq_tsub_of_add_eq' [AddLeftMono α] (hb : AddLECancellable b)
    (h : a + c = b) : a = b - c := (hb.tsub_eq_of_eq_add' h.symm).symm


/-- See `AddLECancellable.tsub_eq_of_eq_add_rev'` for a version assuming that `a = b + c` itself is
cancellable rather than `b`. -/
protected theorem tsub_eq_of_eq_add_rev (hb : AddLECancellable b) (h : a = b + c) : a - b = c :=
                             /-
                               α : Type u_1
                               inst✝³ : PartialOrder α
                               inst✝² : AddCommSemigroup α
                               inst✝¹ : Sub α
                               inst✝ : OrderedSub α
                               a b c : α
                               hb : AddLECancellable b
                               h : Eq a (HAdd.hAdd b c)
                               ⊢ Eq a (HAdd.hAdd c b)
                             -/
  hb.tsub_eq_of_eq_add <| by rw [add_comm, h]
                             /-
                               🎉 no goals
                             -/


/-- Weaker version of `AddLECancellable.tsub_eq_of_eq_add_rev` assuming that `a = b + c` itself is
cancellable rather than `b`. -/
protected lemma tsub_eq_of_eq_add_rev' [AddLeftMono α]
    (ha : AddLECancellable a) (h : a = b + c) : a - b = c :=
                              /-
                                α : Type u_1
                                inst✝⁴ : PartialOrder α
                                inst✝³ : AddCommSemigroup α
                                inst✝² : Sub α
                                inst✝¹ : OrderedSub α
                                a b c : α
                                inst✝ : AddLeftMono α
                                ha : AddLECancellable a
                                h : Eq a (HAdd.hAdd b c)
                                ⊢ Eq a (HAdd.hAdd c b)
                              -/
  ha.tsub_eq_of_eq_add' <| by rw [add_comm, h]
                              /-
                                🎉 no goals
                              -/


@[simp]
protected theorem add_tsub_cancel_right (hb : AddLECancellable b) : a + b - b = a :=
                             /-
                               α : Type u_1
                               inst✝³ : PartialOrder α
                               inst✝² : AddCommSemigroup α
                               inst✝¹ : Sub α
                               inst✝ : OrderedSub α
                               a b : α
                               hb : AddLECancellable b
                               ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd a b)
                             -/
  hb.tsub_eq_of_eq_add <| by rw [add_comm]
                             /-
                               🎉 no goals
                             -/


@[simp]
protected theorem add_tsub_cancel_left (ha : AddLECancellable a) : a + b - a = b :=
  ha.tsub_eq_of_eq_add <| add_comm a b


protected theorem lt_add_of_tsub_lt_left (hb : AddLECancellable b) (h : a - b < c) : a < b + c := by
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    h : LT.lt (HSub.hSub a b) c
    ⊢ LT.lt a (HAdd.hAdd b c)
  -/
  rw [lt_iff_le_and_ne, ← tsub_le_iff_left]
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    h : LT.lt (HSub.hSub a b) c
    ⊢ And (LE.le (HSub.hSub a b) c) (Ne a (HAdd.hAdd b c))
  -/
  refine ⟨h.le, ?_⟩
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hb : AddLECancellable b
    h : LT.lt (HSub.hSub a b) c
    ⊢ Ne a (HAdd.hAdd b c)
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    b c : α
    hb : AddLECancellable b
    h : LT.lt (HSub.hSub (HAdd.hAdd b c) b) c
    ⊢ False
  -/
  simp [hb] at h
  /-
    🎉 no goals
  -/


protected theorem lt_add_of_tsub_lt_right (hc : AddLECancellable c) (h : a - c < b) :
    a < b + c := by
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LT.lt (HSub.hSub a c) b
    ⊢ LT.lt a (HAdd.hAdd b c)
  -/
  rw [lt_iff_le_and_ne, ← tsub_le_iff_right]
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LT.lt (HSub.hSub a c) b
    ⊢ And (LE.le (HSub.hSub a c) b) (Ne a (HAdd.hAdd b c))
  -/
  refine ⟨h.le, ?_⟩
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    hc : AddLECancellable c
    h : LT.lt (HSub.hSub a c) b
    ⊢ Ne a (HAdd.hAdd b c)
  -/
  rintro rfl
  /-
    α : Type u_1
    inst✝³ : PartialOrder α
    inst✝² : AddCommSemigroup α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    b c : α
    hc : AddLECancellable c
    h : LT.lt (HSub.hSub (HAdd.hAdd b c) c) b
    ⊢ False
  -/
  simp [hc] at h
  /-
    🎉 no goals
  -/


protected theorem lt_tsub_of_add_lt_right (hc : AddLECancellable c) (h : a + c < b) : a < b - c :=
  (hc.le_tsub_of_add_le_right h.le).lt_of_ne <| by
    /-
      α : Type u_1
      inst✝³ : PartialOrder α
      inst✝² : AddCommSemigroup α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      a b c : α
      hc : AddLECancellable c
      h : LT.lt (HAdd.hAdd a c) b
      ⊢ Ne a (HSub.hSub b c)
    -/
    rintro rfl
    /-
      α : Type u_1
      inst✝³ : PartialOrder α
      inst✝² : AddCommSemigroup α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      b c : α
      hc : AddLECancellable c
      h : LT.lt (HAdd.hAdd (HSub.hSub b c) c) b
      ⊢ False
    -/
    exact h.not_le le_tsub_add
    /-
      🎉 no goals
    -/


protected theorem lt_tsub_of_add_lt_left (ha : AddLECancellable a) (h : a + c < b) : c < b - a :=
                                   /-
                                     α : Type u_1
                                     inst✝³ : PartialOrder α
                                     inst✝² : AddCommSemigroup α
                                     inst✝¹ : Sub α
                                     inst✝ : OrderedSub α
                                     a b c : α
                                     ha : AddLECancellable a
                                     h : LT.lt (HAdd.hAdd a c) b
                                     ⊢ LT.lt (HAdd.hAdd c a) b
                                   -/
  ha.lt_tsub_of_add_lt_right <| by rwa [add_comm]
                                   /-
                                     🎉 no goals
                                   -/


theorem tsub_eq_of_eq_add (h : a = c + b) : a - b = c :=
  Contravariant.AddLECancellable.tsub_eq_of_eq_add h


theorem eq_tsub_of_add_eq (h : a + c = b) : a = b - c :=
  Contravariant.AddLECancellable.eq_tsub_of_add_eq h


theorem tsub_eq_of_eq_add_rev (h : a = b + c) : a - b = c :=
  Contravariant.AddLECancellable.tsub_eq_of_eq_add_rev h


@[simp]
theorem add_tsub_cancel_right (a b : α) : a + b - b = a :=
  Contravariant.AddLECancellable.add_tsub_cancel_right


@[simp]
theorem add_tsub_cancel_left (a b : α) : a + b - a = b :=
  Contravariant.AddLECancellable.add_tsub_cancel_left


/-- A more general version of the reverse direction of `sub_eq_sub_iff_add_eq_add` -/
theorem tsub_eq_tsub_of_add_eq_add (h : a + d = c + b) : a - b = c - d := by
  calc a - b = a + d - d - b := by rw [add_tsub_cancel_right]
           _ = c + b - b - d := by rw [h, tsub_right_comm]
           _ = c - d := by rw [add_tsub_cancel_right]


theorem lt_add_of_tsub_lt_left (h : a - b < c) : a < b + c :=
  Contravariant.AddLECancellable.lt_add_of_tsub_lt_left h


theorem lt_add_of_tsub_lt_right (h : a - c < b) : a < b + c :=
  Contravariant.AddLECancellable.lt_add_of_tsub_lt_right h


/-- This lemma (and some of its corollaries) also holds for `ENNReal`, but this proof doesn't work
for it. Maybe we should add this lemma as field to `OrderedSub`? -/
theorem lt_tsub_of_add_lt_left : a + c < b → c < b - a :=
  Contravariant.AddLECancellable.lt_tsub_of_add_lt_left


theorem lt_tsub_of_add_lt_right : a + c < b → a < b - c :=
  Contravariant.AddLECancellable.lt_tsub_of_add_lt_right


theorem add_tsub_add_eq_tsub_right (a c b : α) : a + c - (b + c) = a - b := by
  /-
    α : Type u_1
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddCommSemigroup α
    inst✝³ : Sub α
    inst✝² : OrderedSub α
    inst✝¹ : AddLeftMono α
    inst✝ : AddLeftReflectLE α
    a c b : α
    ⊢ Eq (HSub.hSub (HAdd.hAdd a c) (HAdd.hAdd b c)) (HSub.hSub a b)
  -/
  refine add_tsub_add_le_tsub_right.antisymm (tsub_le_iff_right.2 <| ?_)
  /-
    α : Type u_1
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddCommSemigroup α
    inst✝³ : Sub α
    inst✝² : OrderedSub α
    inst✝¹ : AddLeftMono α
    inst✝ : AddLeftReflectLE α
    a c b : α
    ⊢ LE.le a (HAdd.hAdd (HSub.hSub (HAdd.hAdd a c) (HAdd.hAdd b c)) b)
  -/
  apply le_of_add_le_add_right
  /-
    case bc
    α : Type u_1
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddCommSemigroup α
    inst✝³ : Sub α
    inst✝² : OrderedSub α
    inst✝¹ : AddLeftMono α
    inst✝ : AddLeftReflectLE α
    a c b : α
    ⊢ LE.le (HAdd.hAdd a ?a) (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HAdd.hAdd a c) (HAd …
  -/
  rw [add_assoc]
  /-
    case bc
    α : Type u_1
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddCommSemigroup α
    inst✝³ : Sub α
    inst✝² : OrderedSub α
    inst✝¹ : AddLeftMono α
    inst✝ : AddLeftReflectLE α
    a c b : α
    ⊢ LE.le (HAdd.hAdd a ?bc.c) (HAdd.hAdd (HSub.hSub (HAdd.hAdd a c) (HAdd.hAdd b …
  -/
  exact le_tsub_add
  /-
    🎉 no goals
  -/


theorem add_tsub_add_eq_tsub_left (a b c : α) : a + b - (a + c) = b - c := by
  /-
    α : Type u_1
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddCommSemigroup α
    inst✝³ : Sub α
    inst✝² : OrderedSub α
    inst✝¹ : AddLeftMono α
    inst✝ : AddLeftReflectLE α
    a b c : α
    ⊢ Eq (HSub.hSub (HAdd.hAdd a b) (HAdd.hAdd a c)) (HSub.hSub b c)
  -/
  rw [add_comm a b, add_comm a c, add_tsub_add_eq_tsub_right]
  /-
    🎉 no goals
  -/


/-- See `lt_of_tsub_lt_tsub_right_of_le` for a weaker statement in a partial order. -/
theorem lt_of_tsub_lt_tsub_right (h : a - c < b - c) : a < b :=
  lt_imp_lt_of_le_imp_le (fun h => tsub_le_tsub_right h c) h


/-- See `lt_tsub_iff_right_of_le` for a weaker statement in a partial order. -/
theorem lt_tsub_iff_right : a < b - c ↔ a + c < b :=
  lt_iff_lt_of_le_iff_le tsub_le_iff_right


/-- See `lt_tsub_iff_left_of_le` for a weaker statement in a partial order. -/
theorem lt_tsub_iff_left : a < b - c ↔ c + a < b :=
  lt_iff_lt_of_le_iff_le tsub_le_iff_left


theorem lt_tsub_comm : a < b - c ↔ c < b - a :=
  lt_tsub_iff_left.trans lt_tsub_iff_right.symm


/-- See `lt_of_tsub_lt_tsub_left_of_le` for a weaker statement in a partial order. -/
theorem lt_of_tsub_lt_tsub_left (h : a - b < a - c) : c < b :=
  lt_imp_lt_of_le_imp_le (fun h => tsub_le_tsub_left h a) h


@[simp]
theorem tsub_zero (a : α) : a - 0 = a :=
  AddLECancellable.tsub_eq_of_eq_add addLECancellable_zero (add_zero _).symm


