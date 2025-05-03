/-- Order equipped with a sensible successor function. -/
@[ext]
class SuccOrder (α : Type*) [Preorder α] where
  /-- Successor function -/
  succ : α → α
  /-- Proof of basic ordering with respect to `succ`-/
  le_succ : ∀ a, a ≤ succ a
  /-- Proof of interaction between `succ` and maximal element -/
  max_of_succ_le {a} : succ a ≤ a → IsMax a
  /-- Proof that `succ a` is the least element greater than `a`-/
  succ_le_of_lt {a b} : a < b → succ a ≤ b


/-- Order equipped with a sensible predecessor function. -/
@[ext]
class PredOrder (α : Type*) [Preorder α] where
  /-- Predecessor function -/
  pred : α → α
  /-- Proof of basic ordering with respect to `pred`-/
  pred_le : ∀ a, pred a ≤ a
  /-- Proof of interaction between `pred` and minimal element -/
  min_of_le_pred {a} : a ≤ pred a → IsMin a
  /-- Proof that `pred b` is the greatest element less than `b`-/
  le_pred_of_lt {a b} : a < b → a ≤ pred b


instance [Preorder α] [SuccOrder α] :
    PredOrder αᵒᵈ where
  pred := toDual ∘ SuccOrder.succ ∘ ofDual
  pred_le := by
    simp only [comp, OrderDual.forall, ofDual_toDual, toDual_le_toDual,
     SuccOrder.le_succ, implies_true]
                         /-
                           α : Type u_1
                           β : Type u_2
                           inst✝¹ : Preorder α
                           inst✝ : SuccOrder α
                           a✝ : OrderDual α
                           h : LE.le a✝ (Function.comp (⇑OrderDual.toDual) (Function.comp SuccOrder.succ  …
                           ⊢ IsMin a✝
                         -/
  min_of_le_pred h := by apply SuccOrder.max_of_succ_le h
                         /-
                           🎉 no goals
                         -/
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝¹ : Preorder α
                        inst✝ : SuccOrder α
                        ⊢ ∀ {a b : OrderDual α}, LT.lt a b → LE.le a (Function.comp (⇑OrderDual.toDual …
                      -/
  le_pred_of_lt := by intro a b h; exact SuccOrder.succ_le_of_lt h
                                   /-
                                     🎉 no goals
                                   -/


instance [Preorder α] [PredOrder α] :
    SuccOrder αᵒᵈ where
  succ := toDual ∘ PredOrder.pred ∘ ofDual
  le_succ := by
    simp only [comp, OrderDual.forall, ofDual_toDual, toDual_le_toDual,
     PredOrder.pred_le, implies_true]
                         /-
                           α : Type u_1
                           β : Type u_2
                           inst✝¹ : Preorder α
                           inst✝ : PredOrder α
                           a✝ : OrderDual α
                           h : LE.le (Function.comp (⇑OrderDual.toDual) (Function.comp PredOrder.pred ⇑Or …
                           ⊢ IsMax a✝
                         -/
  max_of_succ_le h := by apply PredOrder.min_of_le_pred h
                         /-
                           🎉 no goals
                         -/
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝¹ : Preorder α
                        inst✝ : PredOrder α
                        ⊢ ∀ {a b : OrderDual α}, LT.lt a b → LE.le (Function.comp (⇑OrderDual.toDual)  …
                      -/
  succ_le_of_lt := by intro a b h; exact PredOrder.le_pred_of_lt h
                                   /-
                                     🎉 no goals
                                   -/


/-- A constructor for `SuccOrder α` usable when `α` has no maximal element. -/
def SuccOrder.ofSuccLeIff (succ : α → α) (hsucc_le_iff : ∀ {a b}, succ a ≤ b ↔ a < b) :
    SuccOrder α :=
  { succ
    le_succ := fun _ => (hsucc_le_iff.1 le_rfl).le
    max_of_succ_le := fun ha => (lt_irrefl _ <| hsucc_le_iff.1 ha).elim
    succ_le_of_lt := fun h => hsucc_le_iff.2 h }


/-- A constructor for `PredOrder α` usable when `α` has no minimal element. -/
def PredOrder.ofLePredIff (pred : α → α) (hle_pred_iff : ∀ {a b}, a ≤ pred b ↔ a < b) :
    PredOrder α :=
  { pred
    pred_le := fun _ => (hle_pred_iff.1 le_rfl).le
    min_of_le_pred := fun ha => (lt_irrefl _ <| hle_pred_iff.1 ha).elim
    le_pred_of_lt := fun h => hle_pred_iff.2 h }


/-- A constructor for `SuccOrder α` for `α` a linear order. -/
@[simps]
def SuccOrder.ofCore (succ : α → α) (hn : ∀ {a}, ¬IsMax a → ∀ b, a < b ↔ succ a ≤ b)
    (hm : ∀ a, IsMax a → succ a = a) : SuccOrder α :=
  { succ
    succ_le_of_lt := fun {a b} =>
      by_cases (fun h hab => (hm a h).symm ▸ hab.le) fun h => (hn h b).mp
    le_succ := fun a =>
                                                                   /-
                                                                     α : Type u_1
                                                                     β : Type u_2
                                                                     inst✝ : LinearOrder α
                                                                     succ : α → α
                                                                     hn : ∀ {a : α}, Not (IsMax a) → ∀ (b : α), Iff (LT.lt a b) (LE.le (succ a) b)
                                                                     hm : ∀ (a : α), IsMax a → Eq (succ a) a
                                                                     a : α
                                                                     h : Not (IsMax a)
                                                                     ⊢ LT.lt a (succ a)
                                                                   -/
      by_cases (fun h => (hm a h).symm.le) fun h => le_of_lt <| by simpa using (hn h a).not
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                            /-
                                                              α : Type u_1
                                                              β : Type u_2
                                                              inst✝ : LinearOrder α
                                                              succ : α → α
                                                              hn : ∀ {a : α}, Not (IsMax a) → ∀ (b : α), Iff (LT.lt a b) (LE.le (succ a) b)
                                                              hm : ∀ (a : α), IsMax a → Eq (succ a) a
                                                              a : α
                                                              h : Not (IsMax a)
                                                              ⊢ Not (LE.le (succ a) a)
                                                            -/
    max_of_succ_le := fun {a} => not_imp_not.mp fun h => by simpa using (hn h a).not }
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- A constructor for `PredOrder α` for `α` a linear order. -/
@[simps]
def PredOrder.ofCore (pred : α → α)
    (hn : ∀ {a}, ¬IsMin a → ∀ b, b ≤ pred a ↔ b < a) (hm : ∀ a, IsMin a → pred a = a) :
    PredOrder α :=
  { pred
    le_pred_of_lt := fun {a b} =>
      by_cases (fun h hab => (hm b h).symm ▸ hab.le) fun h => (hn h a).mpr
    pred_le := fun a =>
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                inst✝ : LinearOrder α
                                                                pred : α → α
                                                                hn : ∀ {a : α}, Not (IsMin a) → ∀ (b : α), Iff (LE.le b (pred a)) (LT.lt b a)
                                                                hm : ∀ (a : α), IsMin a → Eq (pred a) a
                                                                a : α
                                                                h : Not (IsMin a)
                                                                ⊢ LT.lt (pred a) a
                                                              -/
      by_cases (fun h => (hm a h).le) fun h => le_of_lt <| by simpa using (hn h a).not
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                            /-
                                                              α : Type u_1
                                                              β : Type u_2
                                                              inst✝ : LinearOrder α
                                                              pred : α → α
                                                              hn : ∀ {a : α}, Not (IsMin a) → ∀ (b : α), Iff (LE.le b (pred a)) (LT.lt b a)
                                                              hm : ∀ (a : α), IsMin a → Eq (pred a) a
                                                              a : α
                                                              h : Not (IsMin a)
                                                              ⊢ Not (LE.le a (pred a))
                                                            -/
    min_of_le_pred := fun {a} => not_imp_not.mp fun h => by simpa using (hn h a).not }
                                                            /-
                                                              🎉 no goals
                                                            -/


open Classical in
/-- A well-order is a `SuccOrder`. -/
noncomputable def SuccOrder.ofLinearWellFoundedLT [WellFoundedLT α] : SuccOrder α :=
  ofCore (fun a ↦ if h : (Ioi a).Nonempty then wellFounded_lt.min _ h else a)
    (fun ha _ ↦ by
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder α
        inst✝ : WellFoundedLT α
        a✝ : α
        ha : Not (IsMax a✝)
        x✝ : α
        ⊢ Iff (LT.lt a✝ x✝) (LE.le ((fun a => dite (Set.Ioi a).Nonempty (fun h => ⋯.mi …
      -/
      rw [not_isMax_iff] at ha
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder α
        inst✝ : WellFoundedLT α
        a✝ : α
        ha : Exists fun b => LT.lt a✝ b
        x✝ : α
        ⊢ Iff (LT.lt a✝ x✝) (LE.le ((fun a => dite (Set.Ioi a).Nonempty (fun h => ⋯.mi …
      -/
      simp_rw [Set.Nonempty, mem_Ioi, dif_pos ha]
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder α
        inst✝ : WellFoundedLT α
        a✝ : α
        ha : Exists fun b => LT.lt a✝ b
        x✝ : α
        ⊢ Iff (LT.lt a✝ x✝) (LE.le (⋯.min (Set.Ioi a✝) ⋯) x✝)
      -/
      exact ⟨(wellFounded_lt.min_le · ha), lt_of_lt_of_le (wellFounded_lt.min_mem _ ha)⟩)
      /-
        🎉 no goals
      -/
    fun _ ha ↦ dif_neg (not_not_intro ha <| not_isMax_iff.mpr ·)


/-- A linear order with well-founded greater-than relation is a `PredOrder`. -/
noncomputable def PredOrder.ofLinearWellFoundedGT (α) [LinearOrder α] [WellFoundedGT α] :
    PredOrder α := letI := SuccOrder.ofLinearWellFoundedLT αᵒᵈ; inferInstanceAs (PredOrder αᵒᵈᵒᵈ)


/-- The successor of an element. If `a` is not maximal, then `succ a` is the least element greater
than `a`. If `a` is maximal, then `succ a = a`. -/
def succ : α → α :=
  SuccOrder.succ


theorem le_succ : ∀ a : α, a ≤ succ a :=
  SuccOrder.le_succ


theorem max_of_succ_le {a : α} : succ a ≤ a → IsMax a :=
  SuccOrder.max_of_succ_le


theorem succ_le_of_lt {a b : α} : a < b → succ a ≤ b :=
  SuccOrder.succ_le_of_lt


alias _root_.LT.lt.succ_le := succ_le_of_lt


@[simp]
theorem succ_le_iff_isMax : succ a ≤ a ↔ IsMax a :=
  ⟨max_of_succ_le, fun h => h <| le_succ _⟩


alias ⟨_root_.IsMax.of_succ_le, _root_.IsMax.succ_le⟩ := succ_le_iff_isMax


@[simp]
theorem lt_succ_iff_not_isMax : a < succ a ↔ ¬IsMax a :=
  ⟨not_isMax_of_lt, fun ha => (le_succ a).lt_of_not_le fun h => ha <| max_of_succ_le h⟩


alias ⟨_, lt_succ_of_not_isMax⟩ := lt_succ_iff_not_isMax


theorem wcovBy_succ (a : α) : a ⩿ succ a :=
  ⟨le_succ a, fun _ hb => (succ_le_of_lt hb).not_lt⟩


theorem covBy_succ_of_not_isMax (h : ¬IsMax a) : a ⋖ succ a :=
  (wcovBy_succ a).covBy_of_lt <| lt_succ_of_not_isMax h


theorem lt_succ_of_le_of_not_isMax (hab : b ≤ a) (ha : ¬IsMax a) : b < succ a :=
  hab.trans_lt <| lt_succ_of_not_isMax ha


theorem succ_le_iff_of_not_isMax (ha : ¬IsMax a) : succ a ≤ b ↔ a < b :=
  ⟨(lt_succ_of_not_isMax ha).trans_le, succ_le_of_lt⟩


lemma succ_lt_succ_of_not_isMax (h : a < b) (hb : ¬ IsMax b) : succ a < succ b :=
  lt_succ_of_le_of_not_isMax (succ_le_of_lt h) hb


@[simp, mono]
theorem succ_le_succ (h : a ≤ b) : succ a ≤ succ b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    h : LE.le a b
    ⊢ LE.le (Order.succ a) (Order.succ b)
  -/
  by_cases hb : IsMax b
    /-
      case pos
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a b : α
      h : LE.le a b
      hb : IsMax b
      ⊢ LE.le (Order.succ a) (Order.succ b)
    -/
  · by_cases hba : b ≤ a
      /-
        case pos
        α : Type u_1
        inst✝¹ : Preorder α
        inst✝ : SuccOrder α
        a b : α
        h : LE.le a b
        hb : IsMax b
        hba : LE.le b a
        ⊢ LE.le (Order.succ a) (Order.succ b)
      -/
    · exact (hb <| hba.trans <| le_succ _).trans (le_succ _)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : Preorder α
        inst✝ : SuccOrder α
        a b : α
        h : LE.le a b
        hb : IsMax b
        hba : Not (LE.le b a)
        ⊢ LE.le (Order.succ a) (Order.succ b)
      -/
    · exact succ_le_of_lt ((h.lt_of_not_le hba).trans_le <| le_succ b)
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a b : α
      h : LE.le a b
      hb : Not (IsMax b)
      ⊢ LE.le (Order.succ a) (Order.succ b)
    -/
  · rw [succ_le_iff_of_not_isMax fun ha => hb <| ha.mono h]
    /-
      case neg
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a b : α
      h : LE.le a b
      hb : Not (IsMax b)
      ⊢ LT.lt a (Order.succ b)
    -/
    apply lt_succ_of_le_of_not_isMax h hb
    /-
      🎉 no goals
    -/


theorem succ_mono : Monotone (succ : α → α) := fun _ _ => succ_le_succ


/-- See also `Order.succ_eq_of_covBy`. -/
lemma le_succ_of_wcovBy (h : a ⩿ b) : b ≤ succ a := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    h : WCovBy a b
    ⊢ LE.le b (Order.succ a)
  -/
  obtain hab | ⟨-, hba⟩ := h.covBy_or_le_and_le
    /-
      case inl
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a b : α
      h : WCovBy a b
      hab : CovBy a b
      ⊢ LE.le b (Order.succ a)
    -/
  · by_contra hba
    /-
      case inl
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a b : α
      h : WCovBy a b
      hab : CovBy a b
      hba : Not (LE.le b (Order.succ a))
      ⊢ False
    -/
    exact h.2 (lt_succ_of_not_isMax hab.lt.not_isMax) <| hab.lt.succ_le.lt_of_not_le hba
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a b : α
      h : WCovBy a b
      hba : LE.le b a
      ⊢ LE.le b (Order.succ a)
    -/
  · exact hba.trans (le_succ _)
    /-
      🎉 no goals
    -/


alias _root_.WCovBy.le_succ := le_succ_of_wcovBy


theorem le_succ_iterate (k : ℕ) (x : α) : x ≤ succ^[k] x :=
  id_le_iterate_of_id_le le_succ _ _


theorem isMax_iterate_succ_of_eq_of_lt {n m : ℕ} (h_eq : succ^[n] a = succ^[m] a)
    (h_lt : n < m) : IsMax (succ^[n] a) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a : α
    n m : Nat
    h_eq : Eq (Nat.iterate Order.succ n a) (Nat.iterate Order.succ m a)
    h_lt : LT.lt n m
    ⊢ IsMax (Nat.iterate Order.succ n a)
  -/
  refine max_of_succ_le (le_trans ?_ h_eq.symm.le)
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a : α
    n m : Nat
    h_eq : Eq (Nat.iterate Order.succ n a) (Nat.iterate Order.succ m a)
    h_lt : LT.lt n m
    ⊢ LE.le (Order.succ (Nat.iterate Order.succ n a)) (Nat.iterate Order.succ m a)
  -/
  rw [← iterate_succ_apply' succ]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a : α
    n m : Nat
    h_eq : Eq (Nat.iterate Order.succ n a) (Nat.iterate Order.succ m a)
    h_lt : LT.lt n m
    ⊢ LE.le (Nat.iterate Order.succ n.succ a) (Nat.iterate Order.succ m a)
  -/
  have h_le : n + 1 ≤ m := Nat.succ_le_of_lt h_lt
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a : α
    n m : Nat
    h_eq : Eq (Nat.iterate Order.succ n a) (Nat.iterate Order.succ m a)
    h_lt : LT.lt n m
    h_le : LE.le (HAdd.hAdd n 1) m
    ⊢ LE.le (Nat.iterate Order.succ n.succ a) (Nat.iterate Order.succ m a)
  -/
  exact Monotone.monotone_iterate_of_le_map succ_mono (le_succ a) h_le
  /-
    🎉 no goals
  -/


theorem isMax_iterate_succ_of_eq_of_ne {n m : ℕ} (h_eq : succ^[n] a = succ^[m] a)
    (h_ne : n ≠ m) : IsMax (succ^[n] a) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a : α
    n m : Nat
    h_eq : Eq (Nat.iterate Order.succ n a) (Nat.iterate Order.succ m a)
    h_ne : Ne n m
    ⊢ IsMax (Nat.iterate Order.succ n a)
  -/
  rcases le_total n m with h | h
    /-
      case inl
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a : α
      n m : Nat
      h_eq : Eq (Nat.iterate Order.succ n a) (Nat.iterate Order.succ m a)
      h_ne : Ne n m
      h : LE.le n m
      ⊢ IsMax (Nat.iterate Order.succ n a)
    -/
  · exact isMax_iterate_succ_of_eq_of_lt h_eq (lt_of_le_of_ne h h_ne)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a : α
      n m : Nat
      h_eq : Eq (Nat.iterate Order.succ n a) (Nat.iterate Order.succ m a)
      h_ne : Ne n m
      h : LE.le m n
      ⊢ IsMax (Nat.iterate Order.succ n a)
    -/
  · rw [h_eq]
    /-
      case inr
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : SuccOrder α
      a : α
      n m : Nat
      h_eq : Eq (Nat.iterate Order.succ n a) (Nat.iterate Order.succ m a)
      h_ne : Ne n m
      h : LE.le m n
      ⊢ IsMax (Nat.iterate Order.succ m a)
    -/
    exact isMax_iterate_succ_of_eq_of_lt h_eq.symm (lt_of_le_of_ne h h_ne.symm)
    /-
      🎉 no goals
    -/


theorem Iic_subset_Iio_succ_of_not_isMax (ha : ¬IsMax a) : Iic a ⊆ Iio (succ a) :=
  fun _ => (lt_succ_of_le_of_not_isMax · ha)


theorem Ici_succ_of_not_isMax (ha : ¬IsMax a) : Ici (succ a) = Ioi a :=
  Set.ext fun _ => succ_le_iff_of_not_isMax ha


theorem Icc_subset_Ico_succ_right_of_not_isMax (hb : ¬IsMax b) : Icc a b ⊆ Ico a (succ b) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    ⊢ HasSubset.Subset (Set.Icc a b) (Set.Ico a (Order.succ b))
  -/
  rw [← Ici_inter_Iio, ← Ici_inter_Iic]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    ⊢ HasSubset.Subset (Inter.inter (Set.Ici a) (Set.Iic b)) (Inter.inter (Set.Ici …
  -/
  gcongr
  /-
    case H
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    ⊢ HasSubset.Subset (Set.Iic b) (Set.Iio (Order.succ b))
  -/
  intro _ h
  /-
    case H
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    a✝ : α
    h : Membership.mem (Set.Iic b) a✝
    ⊢ Membership.mem (Set.Iio (Order.succ b)) a✝
  -/
  apply lt_succ_of_le_of_not_isMax h hb
  /-
    🎉 no goals
  -/


theorem Ioc_subset_Ioo_succ_right_of_not_isMax (hb : ¬IsMax b) : Ioc a b ⊆ Ioo a (succ b) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    ⊢ HasSubset.Subset (Set.Ioc a b) (Set.Ioo a (Order.succ b))
  -/
  rw [← Ioi_inter_Iio, ← Ioi_inter_Iic]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    ⊢ HasSubset.Subset (Inter.inter (Set.Ioi a) (Set.Iic b)) (Inter.inter (Set.Ioi …
  -/
  gcongr
  /-
    case H
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    ⊢ HasSubset.Subset (Set.Iic b) (Set.Iio (Order.succ b))
  -/
  intro _ h
  /-
    case H
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    a✝ : α
    h : Membership.mem (Set.Iic b) a✝
    ⊢ Membership.mem (Set.Iio (Order.succ b)) a✝
  -/
  apply Iic_subset_Iio_succ_of_not_isMax hb h
  /-
    🎉 no goals
  -/


theorem Icc_succ_left_of_not_isMax (ha : ¬IsMax a) : Icc (succ a) b = Ioc a b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    ha : Not (IsMax a)
    ⊢ Eq (Set.Icc (Order.succ a) b) (Set.Ioc a b)
  -/
  rw [← Ici_inter_Iic, Ici_succ_of_not_isMax ha, Ioi_inter_Iic]
  /-
    🎉 no goals
  -/


theorem Ico_succ_left_of_not_isMax (ha : ¬IsMax a) : Ico (succ a) b = Ioo a b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : SuccOrder α
    a b : α
    ha : Not (IsMax a)
    ⊢ Eq (Set.Ico (Order.succ a) b) (Set.Ioo a b)
  -/
  rw [← Ici_inter_Iio, Ici_succ_of_not_isMax ha, Ioi_inter_Iio]
  /-
    🎉 no goals
  -/


theorem lt_succ (a : α) : a < succ a :=
  lt_succ_of_not_isMax <| not_isMax a


@[simp]
theorem lt_succ_of_le : a ≤ b → a < succ b :=
  (lt_succ_of_le_of_not_isMax · <| not_isMax b)


@[simp]
theorem succ_le_iff : succ a ≤ b ↔ a < b :=
  succ_le_iff_of_not_isMax <| not_isMax a


                                                                     /-
                                                                       α : Type u_1
                                                                       inst✝² : Preorder α
                                                                       inst✝¹ : SuccOrder α
                                                                       a b : α
                                                                       inst✝ : NoMaxOrder α
                                                                       hab : LT.lt a b
                                                                       ⊢ LT.lt (Order.succ a) (Order.succ b)
                                                                     -/
@[gcongr] theorem succ_lt_succ (hab : a < b) : succ a < succ b := by simp [hab]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem succ_strictMono : StrictMono (succ : α → α) := fun _ _ => succ_lt_succ


theorem covBy_succ (a : α) : a ⋖ succ a :=
  covBy_succ_of_not_isMax <| not_isMax a


@[simp]
theorem Iic_subset_Iio_succ (a : α) : Iic a ⊆ Iio (succ a) :=
  Iic_subset_Iio_succ_of_not_isMax <| not_isMax _


@[simp]
theorem Ici_succ (a : α) : Ici (succ a) = Ioi a :=
  Ici_succ_of_not_isMax <| not_isMax _


@[simp]
theorem Icc_subset_Ico_succ_right (a b : α) : Icc a b ⊆ Ico a (succ b) :=
  Icc_subset_Ico_succ_right_of_not_isMax <| not_isMax _


@[simp]
theorem Ioc_subset_Ioo_succ_right (a b : α) : Ioc a b ⊆ Ioo a (succ b) :=
  Ioc_subset_Ioo_succ_right_of_not_isMax <| not_isMax _


@[simp]
theorem Icc_succ_left (a b : α) : Icc (succ a) b = Ioc a b :=
  Icc_succ_left_of_not_isMax <| not_isMax _


@[simp]
theorem Ico_succ_left (a b : α) : Ico (succ a) b = Ioo a b :=
  Ico_succ_left_of_not_isMax <| not_isMax _


@[simp]
theorem succ_eq_iff_isMax : succ a = a ↔ IsMax a :=
  ⟨fun h => max_of_succ_le h.le, fun h => h.eq_of_ge <| le_succ _⟩


alias ⟨_, _root_.IsMax.succ_eq⟩ := succ_eq_iff_isMax


theorem le_le_succ_iff : a ≤ b ∧ b ≤ succ a ↔ b = a ∨ b = succ a := by
  refine
    ⟨fun h =>
      or_iff_not_imp_left.2 fun hba : b ≠ a =>
        h.2.antisymm (succ_le_of_lt <| h.1.lt_of_ne <| hba.symm),
      ?_⟩
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : SuccOrder α
    a b : α
    ⊢ Or (Eq b a) (Eq b (Order.succ a)) → And (LE.le a b) (LE.le b (Order.succ a))
  -/
  rintro (rfl | rfl)
    /-
      case inl
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : SuccOrder α
      b : α
      ⊢ And (LE.le b b) (LE.le b (Order.succ b))
    -/
  · exact ⟨le_rfl, le_succ b⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : SuccOrder α
      a : α
      ⊢ And (LE.le a (Order.succ a)) (LE.le (Order.succ a) (Order.succ a))
    -/
  · exact ⟨le_succ a, le_rfl⟩
    /-
      🎉 no goals
    -/


/-- See also `Order.le_succ_of_wcovBy`. -/
lemma succ_eq_of_covBy (h : a ⋖ b) : succ a = b := (succ_le_of_lt h.lt).antisymm h.wcovBy.le_succ


alias _root_.CovBy.succ_eq := succ_eq_of_covBy


theorem _root_.OrderIso.map_succ [PartialOrder β] [SuccOrder β] (f : α ≃o β) (a : α) :
    f (succ a) = succ (f a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : PartialOrder α
    inst✝² : SuccOrder α
    inst✝¹ : PartialOrder β
    inst✝ : SuccOrder β
    f : OrderIso α β
    a : α
    ⊢ Eq (f (Order.succ a)) (Order.succ (f a))
  -/
  by_cases h : IsMax a
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝³ : PartialOrder α
      inst✝² : SuccOrder α
      inst✝¹ : PartialOrder β
      inst✝ : SuccOrder β
      f : OrderIso α β
      a : α
      h : IsMax a
      ⊢ Eq (f (Order.succ a)) (Order.succ (f a))
    -/
  · rw [h.succ_eq, (f.isMax_apply.2 h).succ_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝³ : PartialOrder α
      inst✝² : SuccOrder α
      inst✝¹ : PartialOrder β
      inst✝ : SuccOrder β
      f : OrderIso α β
      a : α
      h : Not (IsMax a)
      ⊢ Eq (f (Order.succ a)) (Order.succ (f a))
    -/
  · exact (f.map_covBy.2 <| covBy_succ_of_not_isMax h).succ_eq.symm
    /-
      🎉 no goals
    -/


theorem succ_eq_iff_covBy : succ a = b ↔ a ⋖ b :=
      /-
        α : Type u_1
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        a b : α
        inst✝ : NoMaxOrder α
        ⊢ Eq (Order.succ a) b → CovBy a b
      -/
  ⟨by rintro rfl; exact covBy_succ _, CovBy.succ_eq⟩
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem succ_top : succ (⊤ : α) = ⊤ := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : OrderTop α
    ⊢ Eq (Order.succ Top.top) Top.top
  -/
  rw [succ_eq_iff_isMax, isMax_iff_eq_top]
  /-
    🎉 no goals
  -/


theorem succ_le_iff_eq_top : succ a ≤ a ↔ a = ⊤ :=
  succ_le_iff_isMax.trans isMax_iff_eq_top


theorem lt_succ_iff_ne_top : a < succ a ↔ a ≠ ⊤ :=
  lt_succ_iff_not_isMax.trans not_isMax_iff_ne_top


theorem bot_lt_succ (a : α) : ⊥ < succ a :=
  (lt_succ_of_not_isMax not_isMax_bot).trans_le <| succ_mono bot_le


theorem succ_ne_bot (a : α) : succ a ≠ ⊥ :=
  (bot_lt_succ a).ne'


theorem le_of_lt_succ {a b : α} : a < succ b → a ≤ b := fun h ↦ by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    h : LT.lt a (Order.succ b)
    ⊢ LE.le a b
  -/
  by_contra! nh
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    h : LT.lt a (Order.succ b)
    nh : LT.lt b a
    ⊢ False
  -/
  exact (h.trans_le (succ_le_of_lt nh)).false
  /-
    🎉 no goals
  -/


theorem lt_succ_iff_of_not_isMax (ha : ¬IsMax a) : b < succ a ↔ b ≤ a :=
  ⟨le_of_lt_succ, fun h => h.trans_lt <| lt_succ_of_not_isMax ha⟩


theorem succ_lt_succ_iff_of_not_isMax (ha : ¬IsMax a) (hb : ¬IsMax b) :
    succ a < succ b ↔ a < b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    ha : Not (IsMax a)
    hb : Not (IsMax b)
    ⊢ Iff (LT.lt (Order.succ a) (Order.succ b)) (LT.lt a b)
  -/
  rw [lt_succ_iff_of_not_isMax hb, succ_le_iff_of_not_isMax ha]
  /-
    🎉 no goals
  -/


theorem succ_le_succ_iff_of_not_isMax (ha : ¬IsMax a) (hb : ¬IsMax b) :
    succ a ≤ succ b ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    ha : Not (IsMax a)
    hb : Not (IsMax b)
    ⊢ Iff (LE.le (Order.succ a) (Order.succ b)) (LE.le a b)
  -/
  rw [succ_le_iff_of_not_isMax ha, lt_succ_iff_of_not_isMax hb]
  /-
    🎉 no goals
  -/


theorem Iio_succ_of_not_isMax (ha : ¬IsMax a) : Iio (succ a) = Iic a :=
  Set.ext fun _ => lt_succ_iff_of_not_isMax ha


theorem Ico_succ_right_of_not_isMax (hb : ¬IsMax b) : Ico a (succ b) = Icc a b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    ⊢ Eq (Set.Ico a (Order.succ b)) (Set.Icc a b)
  -/
  rw [← Ici_inter_Iio, Iio_succ_of_not_isMax hb, Ici_inter_Iic]
  /-
    🎉 no goals
  -/


theorem Ioo_succ_right_of_not_isMax (hb : ¬IsMax b) : Ioo a (succ b) = Ioc a b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    hb : Not (IsMax b)
    ⊢ Eq (Set.Ioo a (Order.succ b)) (Set.Ioc a b)
  -/
  rw [← Ioi_inter_Iio, Iio_succ_of_not_isMax hb, Ioi_inter_Iic]
  /-
    🎉 no goals
  -/


theorem succ_eq_succ_iff_of_not_isMax (ha : ¬IsMax a) (hb : ¬IsMax b) :
    succ a = succ b ↔ a = b := by
  rw [eq_iff_le_not_lt, eq_iff_le_not_lt, succ_le_succ_iff_of_not_isMax ha hb,
    succ_lt_succ_iff_of_not_isMax ha hb]


theorem le_succ_iff_eq_or_le : a ≤ succ b ↔ a = succ b ∨ a ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    ⊢ Iff (LE.le a (Order.succ b)) (Or (Eq a (Order.succ b)) (LE.le a b))
  -/
  by_cases hb : IsMax b
    /-
      case pos
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : SuccOrder α
      a b : α
      hb : IsMax b
      ⊢ Iff (LE.le a (Order.succ b)) (Or (Eq a (Order.succ b)) (LE.le a b))
    -/
  · rw [hb.succ_eq, or_iff_right_of_imp le_of_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : SuccOrder α
      a b : α
      hb : Not (IsMax b)
      ⊢ Iff (LE.le a (Order.succ b)) (Or (Eq a (Order.succ b)) (LE.le a b))
    -/
  · rw [← lt_succ_iff_of_not_isMax hb, le_iff_eq_or_lt]
    /-
      🎉 no goals
    -/


theorem lt_succ_iff_eq_or_lt_of_not_isMax (hb : ¬IsMax b) : a < succ b ↔ a = b ∨ a < b :=
  (lt_succ_iff_of_not_isMax hb).trans le_iff_eq_or_lt


theorem not_isMin_succ [Nontrivial α] (a : α) : ¬ IsMin (succ a) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : Nontrivial α
    a : α
    ⊢ Not (IsMin (Order.succ a))
  -/
  obtain ha | ha := (le_succ a).eq_or_lt
    /-
      case inl
      α : Type u_1
      inst✝² : LinearOrder α
      inst✝¹ : SuccOrder α
      inst✝ : Nontrivial α
      a : α
      ha : Eq a (Order.succ a)
      ⊢ Not (IsMin (Order.succ a))
    -/
  · exact (ha ▸ succ_eq_iff_isMax.1 ha.symm).not_isMin
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : LinearOrder α
      inst✝¹ : SuccOrder α
      inst✝ : Nontrivial α
      a : α
      ha : LT.lt a (Order.succ a)
      ⊢ Not (IsMin (Order.succ a))
    -/
  · exact not_isMin_of_lt ha
    /-
      🎉 no goals
    -/


theorem Iic_succ (a : α) : Iic (succ a) = insert (succ a) (Iic a) :=
  ext fun _ => le_succ_iff_eq_or_le


theorem Icc_succ_right (h : a ≤ succ b) : Icc a (succ b) = insert (succ b) (Icc a b) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    h : LE.le a (Order.succ b)
    ⊢ Eq (Set.Icc a (Order.succ b)) (Insert.insert (Order.succ b) (Set.Icc a b))
  -/
  simp_rw [← Ici_inter_Iic, Iic_succ, inter_insert_of_mem (mem_Ici.2 h)]
  /-
    🎉 no goals
  -/


theorem Ioc_succ_right (h : a < succ b) : Ioc a (succ b) = insert (succ b) (Ioc a b) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    h : LT.lt a (Order.succ b)
    ⊢ Eq (Set.Ioc a (Order.succ b)) (Insert.insert (Order.succ b) (Set.Ioc a b))
  -/
  simp_rw [← Ioi_inter_Iic, Iic_succ, inter_insert_of_mem (mem_Ioi.2 h)]
  /-
    🎉 no goals
  -/


theorem Iio_succ_eq_insert_of_not_isMax (h : ¬IsMax a) : Iio (succ a) = insert a (Iio a) :=
  ext fun _ => lt_succ_iff_eq_or_lt_of_not_isMax h


theorem Ico_succ_right_eq_insert_of_not_isMax (h₁ : a ≤ b) (h₂ : ¬IsMax b) :
    Ico a (succ b) = insert b (Ico a b) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    h₁ : LE.le a b
    h₂ : Not (IsMax b)
    ⊢ Eq (Set.Ico a (Order.succ b)) (Insert.insert b (Set.Ico a b))
  -/
  simp_rw [← Iio_inter_Ici, Iio_succ_eq_insert_of_not_isMax h₂, insert_inter_of_mem (mem_Ici.2 h₁)]
  /-
    🎉 no goals
  -/


theorem Ioo_succ_right_eq_insert_of_not_isMax (h₁ : a < b) (h₂ : ¬IsMax b) :
    Ioo a (succ b) = insert b (Ioo a b) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : SuccOrder α
    a b : α
    h₁ : LT.lt a b
    h₂ : Not (IsMax b)
    ⊢ Eq (Set.Ioo a (Order.succ b)) (Insert.insert b (Set.Ioo a b))
  -/
  simp_rw [← Iio_inter_Ioi, Iio_succ_eq_insert_of_not_isMax h₂, insert_inter_of_mem (mem_Ioi.2 h₁)]
  /-
    🎉 no goals
  -/


@[simp]
theorem lt_succ_iff : a < succ b ↔ a ≤ b :=
  lt_succ_iff_of_not_isMax <| not_isMax b


                                                         /-
                                                           α : Type u_1
                                                           inst✝² : LinearOrder α
                                                           inst✝¹ : SuccOrder α
                                                           a b : α
                                                           inst✝ : NoMaxOrder α
                                                           ⊢ Iff (LE.le (Order.succ a) (Order.succ b)) (LE.le a b)
                                                         -/
theorem succ_le_succ_iff : succ a ≤ succ b ↔ a ≤ b := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/

                                                         /-
                                                           α : Type u_1
                                                           inst✝² : LinearOrder α
                                                           inst✝¹ : SuccOrder α
                                                           a b : α
                                                           inst✝ : NoMaxOrder α
                                                           ⊢ Iff (LT.lt (Order.succ a) (Order.succ b)) (LT.lt a b)
                                                         -/
theorem succ_lt_succ_iff : succ a < succ b ↔ a < b := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


alias ⟨le_of_succ_le_succ, _⟩ := succ_le_succ_iff

alias ⟨lt_of_succ_lt_succ, _⟩ := succ_lt_succ_iff

-- TODO: prove for a succ-archimedean non-linear order with bottom

@[simp]
theorem Iio_succ (a : α) : Iio (succ a) = Iic a :=
  Iio_succ_of_not_isMax <| not_isMax _


@[simp]
theorem Ico_succ_right (a b : α) : Ico a (succ b) = Icc a b :=
  Ico_succ_right_of_not_isMax <| not_isMax _

-- TODO: prove for a succ-archimedean non-linear order

@[simp]
theorem Ioo_succ_right (a b : α) : Ioo a (succ b) = Ioc a b :=
  Ioo_succ_right_of_not_isMax <| not_isMax _


@[simp]
theorem succ_eq_succ_iff : succ a = succ b ↔ a = b :=
  succ_eq_succ_iff_of_not_isMax (not_isMax a) (not_isMax b)


theorem succ_injective : Injective (succ : α → α) := fun _ _ => succ_eq_succ_iff.1


theorem succ_ne_succ_iff : succ a ≠ succ b ↔ a ≠ b :=
  succ_injective.ne_iff


alias ⟨_, succ_ne_succ⟩ := succ_ne_succ_iff


theorem lt_succ_iff_eq_or_lt : a < succ b ↔ a = b ∨ a < b :=
  lt_succ_iff.trans le_iff_eq_or_lt


theorem Iio_succ_eq_insert (a : α) : Iio (succ a) = insert a (Iio a) :=
  Iio_succ_eq_insert_of_not_isMax <| not_isMax a


theorem Ico_succ_right_eq_insert (h : a ≤ b) : Ico a (succ b) = insert b (Ico a b) :=
  Ico_succ_right_eq_insert_of_not_isMax h <| not_isMax b


theorem Ioo_succ_right_eq_insert (h : a < b) : Ioo a (succ b) = insert b (Ioo a b) :=
  Ioo_succ_right_eq_insert_of_not_isMax h <| not_isMax b


                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝³ : LinearOrder α
                                                                    inst✝² : SuccOrder α
                                                                    a : α
                                                                    inst✝¹ : OrderBot α
                                                                    inst✝ : NoMaxOrder α
                                                                    ⊢ Iff (LT.lt a (Order.succ Bot.bot)) (Eq a Bot.bot)
                                                                  -/
theorem lt_succ_bot_iff [NoMaxOrder α] : a < succ ⊥ ↔ a = ⊥ := by rw [lt_succ_iff, le_bot_iff]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem le_succ_bot_iff : a ≤ succ ⊥ ↔ a = ⊥ ∨ a = succ ⊥ := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    a : α
    inst✝ : OrderBot α
    ⊢ Iff (LE.le a (Order.succ Bot.bot)) (Or (Eq a Bot.bot) (Eq a (Order.succ Bot. …
  -/
  rw [le_succ_iff_eq_or_le, le_bot_iff, or_comm]
  /-
    🎉 no goals
  -/


/-- There is at most one way to define the successors in a `PartialOrder`. -/
instance [PartialOrder α] : Subsingleton (SuccOrder α) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : PartialOrder α
      ⊢ ∀ (a b : SuccOrder α), Eq a b
    -/
    intro h₀ h₁
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : PartialOrder α
      h₀ h₁ : SuccOrder α
      ⊢ Eq h₀ h₁
    -/
    ext a
    /-
      case succ.h
      α : Type u_1
      β : Type u_2
      inst✝ : PartialOrder α
      h₀ h₁ : SuccOrder α
      a : α
      ⊢ Eq (SuccOrder.succ a) (SuccOrder.succ a)
    -/
    by_cases ha : IsMax a
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝ : PartialOrder α
        h₀ h₁ : SuccOrder α
        a : α
        ha : IsMax a
        ⊢ Eq (SuccOrder.succ a) (SuccOrder.succ a)
      -/
    · exact (@IsMax.succ_eq _ _ h₀ _ ha).trans ha.succ_eq.symm
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝ : PartialOrder α
        h₀ h₁ : SuccOrder α
        a : α
        ha : Not (IsMax a)
        ⊢ Eq (SuccOrder.succ a) (SuccOrder.succ a)
      -/
    · exact @CovBy.succ_eq _ _ h₀ _ _ (covBy_succ_of_not_isMax ha)⟩
      /-
        🎉 no goals
      -/


theorem succ_eq_sInf [CompleteLattice α] [SuccOrder α] (a : α) :
    succ a = sInf (Set.Ioi a) := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    inst✝ : SuccOrder α
    a : α
    ⊢ Eq (Order.succ a) (InfSet.sInf (Set.Ioi a))
  -/
  apply (le_sInf fun b => succ_le_of_lt).antisymm
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    inst✝ : SuccOrder α
    a : α
    ⊢ LE.le (InfSet.sInf (Set.Ioi a)) (Order.succ a)
  -/
  obtain rfl | ha := eq_or_ne a ⊤
    /-
      case inl
      α : Type u_1
      inst✝¹ : CompleteLattice α
      inst✝ : SuccOrder α
      ⊢ LE.le (InfSet.sInf (Set.Ioi Top.top)) (Order.succ Top.top)
    -/
  · rw [succ_top]
    /-
      case inl
      α : Type u_1
      inst✝¹ : CompleteLattice α
      inst✝ : SuccOrder α
      ⊢ LE.le (InfSet.sInf (Set.Ioi Top.top)) Top.top
    -/
    exact le_top
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : CompleteLattice α
      inst✝ : SuccOrder α
      a : α
      ha : Ne a Top.top
      ⊢ LE.le (InfSet.sInf (Set.Ioi a)) (Order.succ a)
    -/
  · exact sInf_le (lt_succ_iff_ne_top.2 ha)
    /-
      🎉 no goals
    -/


theorem succ_eq_iInf [CompleteLattice α] [SuccOrder α] (a : α) : succ a = ⨅ b > a, b := by
  /-
    α : Type u_1
    inst✝¹ : CompleteLattice α
    inst✝ : SuccOrder α
    a : α
    ⊢ Eq (Order.succ a) (iInf fun b => iInf fun h => b)
  -/
  rw [succ_eq_sInf, iInf_subtype', iInf, Subtype.range_coe_subtype, Ioi]
  /-
    🎉 no goals
  -/


theorem succ_eq_csInf [ConditionallyCompleteLattice α] [SuccOrder α] [NoMaxOrder α] (a : α) :
    succ a = sInf (Set.Ioi a) := by
  /-
    α : Type u_1
    inst✝² : ConditionallyCompleteLattice α
    inst✝¹ : SuccOrder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Eq (Order.succ a) (InfSet.sInf (Set.Ioi a))
  -/
  apply (le_csInf nonempty_Ioi fun b => succ_le_of_lt).antisymm
  /-
    α : Type u_1
    inst✝² : ConditionallyCompleteLattice α
    inst✝¹ : SuccOrder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ LE.le (InfSet.sInf (Set.Ioi a)) (Order.succ a)
  -/
  exact csInf_le ⟨a, fun b => le_of_lt⟩ <| lt_succ a
  /-
    🎉 no goals
  -/


/-- The predecessor of an element. If `a` is not minimal, then `pred a` is the greatest element less
than `a`. If `a` is minimal, then `pred a = a`. -/
def pred : α → α :=
  PredOrder.pred


theorem pred_le : ∀ a : α, pred a ≤ a :=
  PredOrder.pred_le


theorem min_of_le_pred {a : α} : a ≤ pred a → IsMin a :=
  PredOrder.min_of_le_pred


theorem le_pred_of_lt {a b : α} : a < b → a ≤ pred b :=
  PredOrder.le_pred_of_lt


alias _root_.LT.lt.le_pred := le_pred_of_lt


@[simp]
theorem le_pred_iff_isMin : a ≤ pred a ↔ IsMin a :=
  ⟨min_of_le_pred, fun h => h <| pred_le _⟩


alias ⟨_root_.IsMin.of_le_pred, _root_.IsMin.le_pred⟩ := le_pred_iff_isMin


@[simp]
theorem pred_lt_iff_not_isMin : pred a < a ↔ ¬IsMin a :=
  ⟨not_isMin_of_lt, fun ha => (pred_le a).lt_of_not_le fun h => ha <| min_of_le_pred h⟩


alias ⟨_, pred_lt_of_not_isMin⟩ := pred_lt_iff_not_isMin


theorem pred_wcovBy (a : α) : pred a ⩿ a :=
  ⟨pred_le a, fun _ hb nh => (le_pred_of_lt nh).not_lt hb⟩


theorem pred_covBy_of_not_isMin (h : ¬IsMin a) : pred a ⋖ a :=
  (pred_wcovBy a).covBy_of_lt <| pred_lt_of_not_isMin h


theorem pred_lt_of_not_isMin_of_le (ha : ¬IsMin a) : a ≤ b → pred a < b :=
  (pred_lt_of_not_isMin ha).trans_le


theorem le_pred_iff_of_not_isMin (ha : ¬IsMin a) : b ≤ pred a ↔ b < a :=
  ⟨fun h => h.trans_lt <| pred_lt_of_not_isMin ha, le_pred_of_lt⟩


lemma pred_lt_pred_of_not_isMin (h : a < b) (ha : ¬ IsMin a) : pred a < pred b :=
  pred_lt_of_not_isMin_of_le ha <| le_pred_of_lt h


theorem pred_le_pred_of_not_isMin_of_le (ha : ¬IsMin a) (hb : ¬IsMin b) :
    a ≤ b → pred a ≤ pred b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    hb : Not (IsMin b)
    ⊢ LE.le a b → LE.le (Order.pred a) (Order.pred b)
  -/
  rw [le_pred_iff_of_not_isMin hb]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    hb : Not (IsMin b)
    ⊢ LE.le a b → LT.lt (Order.pred a) b
  -/
  apply pred_lt_of_not_isMin_of_le ha
  /-
    🎉 no goals
  -/


@[simp, mono]
theorem pred_le_pred {a b : α} (h : a ≤ b) : pred a ≤ pred b :=
  succ_le_succ h.dual


theorem pred_mono : Monotone (pred : α → α) := fun _ _ => pred_le_pred


/-- See also `Order.pred_eq_of_covBy`. -/
lemma pred_le_of_wcovBy (h : a ⩿ b) : pred b ≤ a := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    a b : α
    h : WCovBy a b
    ⊢ LE.le (Order.pred b) a
  -/
  obtain hab | ⟨-, hba⟩ := h.covBy_or_le_and_le
    /-
      case inl
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : PredOrder α
      a b : α
      h : WCovBy a b
      hab : CovBy a b
      ⊢ LE.le (Order.pred b) a
    -/
  · by_contra hba
    /-
      case inl
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : PredOrder α
      a b : α
      h : WCovBy a b
      hab : CovBy a b
      hba : Not (LE.le (Order.pred b) a)
      ⊢ False
    -/
    exact h.2 (hab.lt.le_pred.lt_of_not_le hba) (pred_lt_of_not_isMin hab.lt.not_isMin)
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : PredOrder α
      a b : α
      h : WCovBy a b
      hba : LE.le b a
      ⊢ LE.le (Order.pred b) a
    -/
  · exact (pred_le _).trans hba
    /-
      🎉 no goals
    -/


alias _root_.WCovBy.pred_le := pred_le_of_wcovBy


theorem pred_iterate_le (k : ℕ) (x : α) : pred^[k] x ≤ x := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    k : Nat
    x : α
    ⊢ LE.le (Nat.iterate Order.pred k x) x
  -/
  conv_rhs => rw [(by simp only [Function.iterate_id, id] : x = id^[k] x)]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    k : Nat
    x : α
    ⊢ LE.le (Nat.iterate Order.pred k x) (Nat.iterate id k x)
  -/
  exact Monotone.iterate_le_of_le pred_mono pred_le k x
  /-
    🎉 no goals
  -/


theorem isMin_iterate_pred_of_eq_of_lt {n m : ℕ} (h_eq : pred^[n] a = pred^[m] a)
    (h_lt : n < m) : IsMin (pred^[n] a) :=
  @isMax_iterate_succ_of_eq_of_lt αᵒᵈ _ _ _ _ _ h_eq h_lt


theorem isMin_iterate_pred_of_eq_of_ne {n m : ℕ} (h_eq : pred^[n] a = pred^[m] a)
    (h_ne : n ≠ m) : IsMin (pred^[n] a) :=
  @isMax_iterate_succ_of_eq_of_ne αᵒᵈ _ _ _ _ _ h_eq h_ne


theorem Ici_subset_Ioi_pred_of_not_isMin (ha : ¬IsMin a) : Ici a ⊆ Ioi (pred a) :=
  fun _ ↦ pred_lt_of_not_isMin_of_le ha


theorem Iic_pred_of_not_isMin (ha : ¬IsMin a) : Iic (pred a) = Iio a :=
  Set.ext fun _ => le_pred_iff_of_not_isMin ha


theorem Icc_subset_Ioc_pred_left_of_not_isMin (ha : ¬IsMin a) : Icc a b ⊆ Ioc (pred a) b := by
 /-
   α : Type u_1
   inst✝¹ : Preorder α
   inst✝ : PredOrder α
   a b : α
   ha : Not (IsMin a)
   ⊢ HasSubset.Subset (Set.Icc a b) (Set.Ioc (Order.pred a) b)
 -/
 rw [← Ioi_inter_Iic, ← Ici_inter_Iic]
 /-
   α : Type u_1
   inst✝¹ : Preorder α
   inst✝ : PredOrder α
   a b : α
   ha : Not (IsMin a)
   ⊢ HasSubset.Subset (Inter.inter (Set.Ici a) (Set.Iic b)) (Inter.inter (Set.Ioi …
 -/
 gcongr
 /-
   case H
   α : Type u_1
   inst✝¹ : Preorder α
   inst✝ : PredOrder α
   a b : α
   ha : Not (IsMin a)
   ⊢ HasSubset.Subset (Set.Ici a) (Set.Ioi (Order.pred a))
 -/
 apply Ici_subset_Ioi_pred_of_not_isMin ha
 /-
   🎉 no goals
 -/


theorem Ico_subset_Ioo_pred_left_of_not_isMin (ha : ¬IsMin a) : Ico a b ⊆ Ioo (pred a) b  := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    ⊢ HasSubset.Subset (Set.Ico a b) (Set.Ioo (Order.pred a) b)
  -/
  rw [← Ioi_inter_Iio, ← Ici_inter_Iio]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    ⊢ HasSubset.Subset (Inter.inter (Set.Ici a) (Set.Iio b)) (Inter.inter (Set.Ioi …
  -/
  gcongr
  /-
    case H
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    ⊢ HasSubset.Subset (Set.Ici a) (Set.Ioi (Order.pred a))
  -/
  apply Ici_subset_Ioi_pred_of_not_isMin ha
  /-
    🎉 no goals
  -/


theorem Icc_pred_right_of_not_isMin (ha : ¬IsMin b) : Icc a (pred b) = Ico a b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin b)
    ⊢ Eq (Set.Icc a (Order.pred b)) (Set.Ico a b)
  -/
  rw [← Ici_inter_Iic, Iic_pred_of_not_isMin ha, Ici_inter_Iio]
  /-
    🎉 no goals
  -/


theorem Ioc_pred_right_of_not_isMin (ha : ¬IsMin b) : Ioc a (pred b) = Ioo a b := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin b)
    ⊢ Eq (Set.Ioc a (Order.pred b)) (Set.Ioo a b)
  -/
  rw [← Ioi_inter_Iic, Iic_pred_of_not_isMin ha, Ioi_inter_Iio]
  /-
    🎉 no goals
  -/


theorem pred_lt (a : α) : pred a < a :=
  pred_lt_of_not_isMin <| not_isMin a


@[simp]
theorem pred_lt_of_le : a ≤ b → pred a < b :=
  pred_lt_of_not_isMin_of_le <| not_isMin a


@[simp]
theorem le_pred_iff : a ≤ pred b ↔ a < b :=
  le_pred_iff_of_not_isMin <| not_isMin b


                                                           /-
                                                             α : Type u_1
                                                             inst✝² : Preorder α
                                                             inst✝¹ : PredOrder α
                                                             a b : α
                                                             inst✝ : NoMinOrder α
                                                             ⊢ LE.le a b → LE.le (Order.pred a) (Order.pred b)
                                                           -/
theorem pred_le_pred_of_le : a ≤ b → pred a ≤ pred b := by intro; simp_all
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                     /-
                                                       α : Type u_1
                                                       inst✝² : Preorder α
                                                       inst✝¹ : PredOrder α
                                                       a b : α
                                                       inst✝ : NoMinOrder α
                                                       ⊢ LT.lt a b → LT.lt (Order.pred a) (Order.pred b)
                                                     -/
theorem pred_lt_pred : a < b → pred a < pred b := by intro; simp_all
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem pred_strictMono : StrictMono (pred : α → α) := fun _ _ => pred_lt_pred


theorem pred_covBy (a : α) : pred a ⋖ a :=
  pred_covBy_of_not_isMin <| not_isMin a


@[simp]
theorem Ici_subset_Ioi_pred (a : α) : Ici a ⊆ Ioi (pred a) :=
  Ici_subset_Ioi_pred_of_not_isMin <| not_isMin a


@[simp]
theorem Iic_pred (a : α) : Iic (pred a) = Iio a :=
  Iic_pred_of_not_isMin <| not_isMin a


@[simp]
theorem Icc_subset_Ioc_pred_left (a b : α) : Icc a b ⊆ Ioc (pred a) b :=
  Icc_subset_Ioc_pred_left_of_not_isMin <| not_isMin _


@[simp]
theorem Ico_subset_Ioo_pred_left (a b : α) : Ico a b ⊆ Ioo (pred a) b :=
  Ico_subset_Ioo_pred_left_of_not_isMin <| not_isMin _


@[simp]
theorem Icc_pred_right (a b : α) : Icc a (pred b) = Ico a b :=
  Icc_pred_right_of_not_isMin <| not_isMin _


@[simp]
theorem Ioc_pred_right (a b : α) : Ioc a (pred b) = Ioo a b :=
  Ioc_pred_right_of_not_isMin <| not_isMin _


@[simp]
theorem pred_eq_iff_isMin : pred a = a ↔ IsMin a :=
  ⟨fun h => min_of_le_pred h.ge, fun h => h.eq_of_le <| pred_le _⟩


alias ⟨_, _root_.IsMin.pred_eq⟩ := pred_eq_iff_isMin


theorem pred_le_le_iff {a b : α} : pred a ≤ b ∧ b ≤ a ↔ b = a ∨ b = pred a := by
  refine
    ⟨fun h =>
      or_iff_not_imp_left.2 fun hba : b ≠ a => (le_pred_of_lt <| h.2.lt_of_ne hba).antisymm h.1, ?_⟩
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : PredOrder α
    a b : α
    ⊢ Or (Eq b a) (Eq b (Order.pred a)) → And (LE.le (Order.pred a) b) (LE.le b a)
  -/
  rintro (rfl | rfl)
    /-
      case inl
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : PredOrder α
      b : α
      ⊢ And (LE.le (Order.pred b) b) (LE.le b b)
    -/
  · exact ⟨pred_le b, le_rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : PredOrder α
      a : α
      ⊢ And (LE.le (Order.pred a) (Order.pred a)) (LE.le (Order.pred a) a)
    -/
  · exact ⟨le_rfl, pred_le a⟩
    /-
      🎉 no goals
    -/


/-- See also `Order.pred_le_of_wcovBy`. -/
lemma pred_eq_of_covBy (h : a ⋖ b) : pred b = a := h.wcovBy.pred_le.antisymm (le_pred_of_lt h.lt)


alias _root_.CovBy.pred_eq := pred_eq_of_covBy


theorem _root_.OrderIso.map_pred {β : Type*} [PartialOrder β] [PredOrder β] (f : α ≃o β) (a : α) :
    f (pred a) = pred (f a) :=
  f.dual.map_succ a


theorem pred_eq_iff_covBy : pred b = a ↔ a ⋖ b :=
  ⟨by
    /-
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      a b : α
      inst✝ : NoMinOrder α
      ⊢ Eq (Order.pred b) a → CovBy a b
    -/
    rintro rfl
    /-
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      b : α
      inst✝ : NoMinOrder α
      ⊢ CovBy (Order.pred b) b
    -/
    exact pred_covBy _, CovBy.pred_eq⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem pred_bot : pred (⊥ : α) = ⊥ :=
  isMin_bot.pred_eq


theorem le_pred_iff_eq_bot : a ≤ pred a ↔ a = ⊥ :=
  @succ_le_iff_eq_top αᵒᵈ _ _ _ _


theorem pred_lt_iff_ne_bot : pred a < a ↔ a ≠ ⊥ :=
  @lt_succ_iff_ne_top αᵒᵈ _ _ _ _


theorem pred_lt_top (a : α) : pred a < ⊤ :=
  (pred_mono le_top).trans_lt <| pred_lt_of_not_isMin not_isMin_top


theorem pred_ne_top (a : α) : pred a ≠ ⊤ :=
  (pred_lt_top a).ne


theorem le_of_pred_lt {a b : α} : pred a < b → a ≤ b := fun h ↦ by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    h : LT.lt (Order.pred a) b
    ⊢ LE.le a b
  -/
  by_contra! nh
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    h : LT.lt (Order.pred a) b
    nh : LT.lt b a
    ⊢ False
  -/
  exact le_pred_of_lt nh |>.trans_lt h |>.false
  /-
    🎉 no goals
  -/


theorem pred_lt_iff_of_not_isMin (ha : ¬IsMin a) : pred a < b ↔ a ≤ b :=
  ⟨le_of_pred_lt, (pred_lt_of_not_isMin ha).trans_le⟩


theorem pred_lt_pred_iff_of_not_isMin (ha : ¬IsMin a) (hb : ¬IsMin b) :
    pred a < pred b ↔ a < b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    hb : Not (IsMin b)
    ⊢ Iff (LT.lt (Order.pred a) (Order.pred b)) (LT.lt a b)
  -/
  rw [pred_lt_iff_of_not_isMin ha, le_pred_iff_of_not_isMin hb]
  /-
    🎉 no goals
  -/


theorem pred_le_pred_iff_of_not_isMin (ha : ¬IsMin a) (hb : ¬IsMin b) :
    pred a ≤ pred b ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    hb : Not (IsMin b)
    ⊢ Iff (LE.le (Order.pred a) (Order.pred b)) (LE.le a b)
  -/
  rw [le_pred_iff_of_not_isMin hb, pred_lt_iff_of_not_isMin ha]
  /-
    🎉 no goals
  -/


theorem Ioi_pred_of_not_isMin (ha : ¬IsMin a) : Ioi (pred a) = Ici a :=
  Set.ext fun _ => pred_lt_iff_of_not_isMin ha


theorem Ioc_pred_left_of_not_isMin (ha : ¬IsMin a) : Ioc (pred a) b = Icc a b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    ⊢ Eq (Set.Ioc (Order.pred a) b) (Set.Icc a b)
  -/
  rw [← Ioi_inter_Iic, Ioi_pred_of_not_isMin ha, Ici_inter_Iic]
  /-
    🎉 no goals
  -/


theorem Ioo_pred_left_of_not_isMin (ha : ¬IsMin a) : Ioo (pred a) b = Ico a b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    ha : Not (IsMin a)
    ⊢ Eq (Set.Ioo (Order.pred a) b) (Set.Ico a b)
  -/
  rw [← Ioi_inter_Iio, Ioi_pred_of_not_isMin ha, Ici_inter_Iio]
  /-
    🎉 no goals
  -/


theorem pred_eq_pred_iff_of_not_isMin (ha : ¬IsMin a) (hb : ¬IsMin b) :
    pred a = pred b ↔ a = b := by
  rw [eq_iff_le_not_lt, eq_iff_le_not_lt, pred_le_pred_iff_of_not_isMin ha hb,
    pred_lt_pred_iff_of_not_isMin ha hb]


theorem pred_le_iff_eq_or_le : pred a ≤ b ↔ b = pred a ∨ a ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    ⊢ Iff (LE.le (Order.pred a) b) (Or (Eq b (Order.pred a)) (LE.le a b))
  -/
  by_cases ha : IsMin a
    /-
      case pos
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : PredOrder α
      a b : α
      ha : IsMin a
      ⊢ Iff (LE.le (Order.pred a) b) (Or (Eq b (Order.pred a)) (LE.le a b))
    -/
  · rw [ha.pred_eq, or_iff_right_of_imp ge_of_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : PredOrder α
      a b : α
      ha : Not (IsMin a)
      ⊢ Iff (LE.le (Order.pred a) b) (Or (Eq b (Order.pred a)) (LE.le a b))
    -/
  · rw [← pred_lt_iff_of_not_isMin ha, le_iff_eq_or_lt, eq_comm]
    /-
      🎉 no goals
    -/


theorem pred_lt_iff_eq_or_lt_of_not_isMin (ha : ¬IsMin a) : pred a < b ↔ a = b ∨ a < b :=
  (pred_lt_iff_of_not_isMin ha).trans le_iff_eq_or_lt


theorem not_isMax_pred [Nontrivial α] (a : α) : ¬ IsMax (pred a) :=
  not_isMin_succ (α := αᵒᵈ) a


theorem Ici_pred (a : α) : Ici (pred a) = insert (pred a) (Ici a) :=
  ext fun _ => pred_le_iff_eq_or_le


theorem Ioi_pred_eq_insert_of_not_isMin (ha : ¬IsMin a) : Ioi (pred a) = insert a (Ioi a) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a : α
    ha : Not (IsMin a)
    ⊢ Eq (Set.Ioi (Order.pred a)) (Insert.insert a (Set.Ioi a))
  -/
  ext x; simp only [insert, mem_setOf, @eq_comm _ x a, mem_Ioi, Set.insert]
  /-
    case h
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a : α
    ha : Not (IsMin a)
    x : α
    ⊢ Iff (LT.lt (Order.pred a) x) (Or (Eq a x) (LT.lt a x))
  -/
  exact pred_lt_iff_eq_or_lt_of_not_isMin ha
  /-
    🎉 no goals
  -/


theorem Icc_pred_left (h : pred a ≤ b) : Icc (pred a) b = insert (pred a) (Icc a b) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    h : LE.le (Order.pred a) b
    ⊢ Eq (Set.Icc (Order.pred a) b) (Insert.insert (Order.pred a) (Set.Icc a b))
  -/
  simp_rw [← Ici_inter_Iic, Ici_pred, insert_inter_of_mem (mem_Iic.2 h)]
  /-
    🎉 no goals
  -/


theorem Ico_pred_left (h : pred a < b) : Ico (pred a) b = insert (pred a) (Ico a b) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : PredOrder α
    a b : α
    h : LT.lt (Order.pred a) b
    ⊢ Eq (Set.Ico (Order.pred a) b) (Insert.insert (Order.pred a) (Set.Ico a b))
  -/
  simp_rw [← Ici_inter_Iio, Ici_pred, insert_inter_of_mem (mem_Iio.2 h)]
  /-
    🎉 no goals
  -/


@[simp]
theorem pred_lt_iff : pred a < b ↔ a ≤ b :=
  pred_lt_iff_of_not_isMin <| not_isMin a


                                                         /-
                                                           α : Type u_1
                                                           inst✝² : LinearOrder α
                                                           inst✝¹ : PredOrder α
                                                           a b : α
                                                           inst✝ : NoMinOrder α
                                                           ⊢ Iff (LE.le (Order.pred a) (Order.pred b)) (LE.le a b)
                                                         -/
theorem pred_le_pred_iff : pred a ≤ pred b ↔ a ≤ b := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                         /-
                                                           α : Type u_1
                                                           inst✝² : LinearOrder α
                                                           inst✝¹ : PredOrder α
                                                           a b : α
                                                           inst✝ : NoMinOrder α
                                                           ⊢ Iff (LT.lt (Order.pred a) (Order.pred b)) (LT.lt a b)
                                                         -/
theorem pred_lt_pred_iff : pred a < pred b ↔ a < b := by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


alias ⟨le_of_pred_le_pred, _⟩ := pred_le_pred_iff


alias ⟨lt_of_pred_lt_pred, _⟩ := pred_lt_pred_iff

-- TODO: prove for a pred-archimedean non-linear order with top

@[simp]
theorem Ioi_pred (a : α) : Ioi (pred a) = Ici a :=
  Ioi_pred_of_not_isMin <| not_isMin a


@[simp]
theorem Ioc_pred_left (a b : α) : Ioc (pred a) b = Icc a b :=
  Ioc_pred_left_of_not_isMin <| not_isMin _

-- TODO: prove for a pred-archimedean non-linear order

@[simp]
theorem Ioo_pred_left (a b : α) : Ioo (pred a) b = Ico a b :=
  Ioo_pred_left_of_not_isMin <| not_isMin _


@[simp]
theorem pred_eq_pred_iff : pred a = pred b ↔ a = b := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    a b : α
    inst✝ : NoMinOrder α
    ⊢ Iff (Eq (Order.pred a) (Order.pred b)) (Eq a b)
  -/
  simp_rw [eq_iff_le_not_lt, pred_le_pred_iff, pred_lt_pred_iff]
  /-
    🎉 no goals
  -/


theorem pred_injective : Injective (pred : α → α) := fun _ _ => pred_eq_pred_iff.1


theorem pred_ne_pred_iff : pred a ≠ pred b ↔ a ≠ b :=
  pred_injective.ne_iff


alias ⟨_, pred_ne_pred⟩ := pred_ne_pred_iff


theorem pred_lt_iff_eq_or_lt : pred a < b ↔ a = b ∨ a < b :=
  pred_lt_iff.trans le_iff_eq_or_lt


theorem Ioi_pred_eq_insert (a : α) : Ioi (pred a) = insert a (Ioi a) :=
  ext fun _ => pred_lt_iff_eq_or_lt.trans <| or_congr_left eq_comm


theorem Ico_pred_right_eq_insert (h : a ≤ b) : Ioc (pred a) b = insert a (Ioc a b) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    a b : α
    inst✝ : NoMinOrder α
    h : LE.le a b
    ⊢ Eq (Set.Ioc (Order.pred a) b) (Insert.insert a (Set.Ioc a b))
  -/
  simp_rw [← Ioi_inter_Iic, Ioi_pred_eq_insert, insert_inter_of_mem (mem_Iic.2 h)]
  /-
    🎉 no goals
  -/


theorem Ioo_pred_right_eq_insert (h : a < b) : Ioo (pred a) b = insert a (Ioo a b) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    a b : α
    inst✝ : NoMinOrder α
    h : LT.lt a b
    ⊢ Eq (Set.Ioo (Order.pred a) b) (Insert.insert a (Set.Ioo a b))
  -/
  simp_rw [← Ioi_inter_Iio, Ioi_pred_eq_insert, insert_inter_of_mem (mem_Iio.2 h)]
  /-
    🎉 no goals
  -/


theorem pred_top_lt_iff [NoMinOrder α] : pred ⊤ < a ↔ a = ⊤ :=
  @lt_succ_bot_iff αᵒᵈ _ _ _ _ _


theorem pred_top_le_iff : pred ⊤ ≤ a ↔ a = ⊤ ∨ a = pred ⊤ :=
  @le_succ_bot_iff αᵒᵈ _ _ _ _


/-- There is at most one way to define the predecessors in a `PartialOrder`. -/
instance [PartialOrder α] : Subsingleton (PredOrder α) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : PartialOrder α
      ⊢ ∀ (a b : PredOrder α), Eq a b
    -/
    intro h₀ h₁
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : PartialOrder α
      h₀ h₁ : PredOrder α
      ⊢ Eq h₀ h₁
    -/
    ext a
    /-
      case pred.h
      α : Type u_1
      β : Type u_2
      inst✝ : PartialOrder α
      h₀ h₁ : PredOrder α
      a : α
      ⊢ Eq (PredOrder.pred a) (PredOrder.pred a)
    -/
    by_cases ha : IsMin a
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝ : PartialOrder α
        h₀ h₁ : PredOrder α
        a : α
        ha : IsMin a
        ⊢ Eq (PredOrder.pred a) (PredOrder.pred a)
      -/
    · exact (@IsMin.pred_eq _ _ h₀ _ ha).trans ha.pred_eq.symm
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝ : PartialOrder α
        h₀ h₁ : PredOrder α
        a : α
        ha : Not (IsMin a)
        ⊢ Eq (PredOrder.pred a) (PredOrder.pred a)
      -/
    · exact @CovBy.pred_eq _ _ h₀ _ _ (pred_covBy_of_not_isMin ha)⟩
      /-
        🎉 no goals
      -/


theorem pred_eq_sSup [CompleteLattice α] [PredOrder α] :
    ∀ a : α, pred a = sSup (Set.Iio a) :=
  succ_eq_sInf (α := αᵒᵈ)


theorem pred_eq_iSup [CompleteLattice α] [PredOrder α] (a : α) : pred a = ⨆ b < a, b :=
  succ_eq_iInf (α := αᵒᵈ) a


theorem pred_eq_csSup [ConditionallyCompleteLattice α] [PredOrder α] [NoMinOrder α] (a : α) :
    pred a = sSup (Set.Iio a) :=
  succ_eq_csInf (α := αᵒᵈ) a


lemma le_succ_pred (a : α) : a ≤ succ (pred a) := (pred_wcovBy _).le_succ

lemma pred_succ_le (a : α) : pred (succ a) ≤ a := (wcovBy_succ _).pred_le


lemma pred_le_iff_le_succ : pred a ≤ b ↔ a ≤ succ b where
  mp hab := (le_succ_pred _).trans (succ_mono hab)
  mpr hab := (pred_mono hab).trans (pred_succ_le _)


lemma gc_pred_succ : GaloisConnection (pred : α → α) succ := fun _ _ ↦ pred_le_iff_le_succ


@[simp]
theorem succ_pred_of_not_isMin (h : ¬IsMin a) : succ (pred a) = a :=
  CovBy.succ_eq (pred_covBy_of_not_isMin h)


@[simp]
theorem pred_succ_of_not_isMax (h : ¬IsMax a) : pred (succ a) = a :=
  CovBy.pred_eq (covBy_succ_of_not_isMax h)


theorem succ_pred [NoMinOrder α] (a : α) : succ (pred a) = a :=
  CovBy.succ_eq (pred_covBy _)


theorem pred_succ [NoMaxOrder α] (a : α) : pred (succ a) = a :=
  CovBy.pred_eq (covBy_succ _)


theorem pred_succ_iterate_of_not_isMax (i : α) (n : ℕ) (hin : ¬IsMax (succ^[n - 1] i)) :
    pred^[n] (succ^[n] i) = i := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : PredOrder α
    i : α
    n : Nat
    hin : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i))
    ⊢ Eq (Nat.iterate Order.pred n (Nat.iterate Order.succ n i)) i
  -/
  induction' n with n hn
    /-
      case zero
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : PredOrder α
      i : α
      hin : Not (IsMax (Nat.iterate Order.succ (HSub.hSub 0 1) i))
      ⊢ Eq (Nat.iterate Order.pred 0 (Nat.iterate Order.succ 0 i)) i
    -/
  · simp only [Nat.zero_eq, Function.iterate_zero, id]
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : PredOrder α
    i : α
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i)) → Eq (Nat.iterate  …
    hin : Not (IsMax (Nat.iterate Order.succ (HSub.hSub (HAdd.hAdd n 1) 1) i))
    ⊢ Eq (Nat.iterate Order.pred (HAdd.hAdd n 1) (Nat.iterate Order.succ (HAdd.hAd …
  -/
  rw [Nat.succ_sub_succ_eq_sub, Nat.sub_zero] at hin
  have h_not_max : ¬IsMax (succ^[n - 1] i) := by
    cases' n with n
    · simpa using hin
    rw [Nat.succ_sub_succ_eq_sub, Nat.sub_zero] at hn ⊢
    have h_sub_le : succ^[n] i ≤ succ^[n.succ] i := by
      rw [Function.iterate_succ']
      exact le_succ _
    refine fun h_max => hin fun j hj => ?_
    have hj_le : j ≤ succ^[n] i := h_max (h_sub_le.trans hj)
    exact hj_le.trans h_sub_le
  /-
    case succ
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : PredOrder α
    i : α
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i)) → Eq (Nat.iterate  …
    hin : Not (IsMax (Nat.iterate Order.succ n i))
    h_not_max : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i))
    ⊢ Eq (Nat.iterate Order.pred (HAdd.hAdd n 1) (Nat.iterate Order.succ (HAdd.hAd …
  -/
  rw [Function.iterate_succ, Function.iterate_succ']
  /-
    case succ
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : PredOrder α
    i : α
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i)) → Eq (Nat.iterate  …
    hin : Not (IsMax (Nat.iterate Order.succ n i))
    h_not_max : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i))
    ⊢ Eq (Function.comp (Nat.iterate Order.pred n) Order.pred (Function.comp Order …
  -/
  simp only [Function.comp_apply]
  /-
    case succ
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : PredOrder α
    i : α
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i)) → Eq (Nat.iterate  …
    hin : Not (IsMax (Nat.iterate Order.succ n i))
    h_not_max : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i))
    ⊢ Eq (Nat.iterate Order.pred n (Order.pred (Order.succ (Nat.iterate Order.succ …
  -/
  rw [pred_succ_of_not_isMax hin]
  /-
    case succ
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : PredOrder α
    i : α
    n : Nat
    hn : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i)) → Eq (Nat.iterate  …
    hin : Not (IsMax (Nat.iterate Order.succ n i))
    h_not_max : Not (IsMax (Nat.iterate Order.succ (HSub.hSub n 1) i))
    ⊢ Eq (Nat.iterate Order.pred n (Nat.iterate Order.succ n i)) i
  -/
  exact hn h_not_max
  /-
    🎉 no goals
  -/


theorem succ_pred_iterate_of_not_isMin (i : α) (n : ℕ) (hin : ¬IsMin (pred^[n - 1] i)) :
    succ^[n] (pred^[n] i) = i :=
  @pred_succ_iterate_of_not_isMax αᵒᵈ _ _ _ i n hin


instance : SuccOrder (WithTop α) where
  succ a :=
    match a with
    | ⊤ => ⊤
    | Option.some a => ite (succ a = a) ⊤ (some (succ a))
  le_succ a := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a : WithTop α
      ⊢ LE.le a ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fun …
    -/
    cases' a with a a
      /-
        case top
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        ⊢ LE.le Top.top ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α)  …
      -/
    · exact le_top
      /-
        🎉 no goals
      -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a : α
      ⊢ LE.le (↑a) ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a ( …
    -/
    change _ ≤ ite _ _ _
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a : α
      ⊢ LE.le (↑a) (ite (Eq (Order.succ a) a) Top.top ↑(Order.succ a))
    -/
    split_ifs
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a : α
        h✝ : Eq (Order.succ a) a
        ⊢ LE.le (↑a) Top.top
      -/
    · exact le_top
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a : α
        h✝ : Not (Eq (Order.succ a) a)
        ⊢ LE.le ↑a ↑(Order.succ a)
      -/
    · exact coe_le_coe.2 (le_succ a)
      /-
        🎉 no goals
      -/
  max_of_succ_le {a} ha := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a : WithTop α
      ha : LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fu …
      ⊢ IsMax a
    -/
    cases a
      /-
        case top
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        ha : LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fu …
        ⊢ IsMax Top.top
      -/
    · exact isMax_top
      /-
        🎉 no goals
      -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a✝ : α
      ha : LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fu …
      ⊢ IsMax ↑a✝
    -/
    dsimp only at ha
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a✝ : α
      ha : LE.le (ite (Eq (Order.succ a✝) a✝) Top.top ↑(Order.succ a✝)) ↑a✝
      ⊢ IsMax ↑a✝
    -/
    split_ifs at ha with ha'
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a✝ : α
        ha' : Eq (Order.succ a✝) a✝
        ha : LE.le Top.top ↑a✝
        ⊢ IsMax ↑a✝
      -/
    · exact (not_top_le_coe _ ha).elim
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a✝ : α
        ha' : Not (Eq (Order.succ a✝) a✝)
        ha : LE.le ↑(Order.succ a✝) ↑a✝
        ⊢ IsMax ↑a✝
      -/
    · rw [coe_le_coe, succ_le_iff_isMax, ← succ_eq_iff_isMax] at ha
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a✝ : α
        ha' : Not (Eq (Order.succ a✝) a✝)
        ha : Eq (Order.succ a✝) a✝
        ⊢ IsMax ↑a✝
      -/
      exact (ha' ha).elim
      /-
        🎉 no goals
      -/
  succ_le_of_lt {a b} h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a b : WithTop α
      h : LT.lt a b
      ⊢ LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fun _ …
    -/
    cases b
      /-
        case top
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a : WithTop α
        h : LT.lt a Top.top
        ⊢ LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fun _ …
      -/
    · exact le_top
      /-
        🎉 no goals
      -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a : WithTop α
      a✝ : α
      h : LT.lt a ↑a✝
      ⊢ LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fun _ …
    -/
    cases a
      /-
        case coe.top
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a✝ : α
        h : LT.lt Top.top ↑a✝
        ⊢ LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fun _ …
      -/
    · exact (not_top_lt h).elim
      /-
        🎉 no goals
      -/
    /-
      case coe.coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a✝¹ a✝ : α
      h : LT.lt ↑a✝ ↑a✝¹
      ⊢ LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fun _ …
    -/
    rw [coe_lt_coe] at h
    /-
      case coe.coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a✝¹ a✝ : α
      h : LT.lt a✝ a✝¹
      ⊢ LE.le ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fun _ …
    -/
    change ite _ _ _ ≤ _
    /-
      case coe.coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
      a✝¹ a✝ : α
      h : LT.lt a✝ a✝¹
      ⊢ LE.le (ite (Eq (Order.succ a✝) a✝) Top.top ↑(Order.succ a✝)) ↑a✝¹
    -/
    split_ifs with ha
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a✝¹ a✝ : α
        h : LT.lt a✝ a✝¹
        ha : Eq (Order.succ a✝) a✝
        ⊢ LE.le Top.top ↑a✝¹
      -/
    · rw [succ_eq_iff_isMax] at ha
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a✝¹ a✝ : α
        h : LT.lt a✝ a✝¹
        ha : IsMax a✝
        ⊢ LE.le Top.top ↑a✝¹
      -/
      exact (ha.not_lt h).elim
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.succ a) a)
        a✝¹ a✝ : α
        h : LT.lt a✝ a✝¹
        ha : Not (Eq (Order.succ a✝) a✝)
        ⊢ LE.le ↑(Order.succ a✝) ↑a✝¹
      -/
    · exact coe_le_coe.2 (succ_le_of_lt h)
      /-
        🎉 no goals
      -/


@[simp]
theorem succ_coe_of_isMax {a : α} (h : IsMax a) : succ ↑a = (⊤ : WithTop α) :=
  dif_pos (succ_eq_iff_isMax.2 h)


theorem succ_coe_of_not_isMax {a : α} (h : ¬ IsMax a) : succ (↑a : WithTop α) = ↑(succ a) :=
  dif_neg (succ_eq_iff_isMax.not.2 h)


@[simp]
theorem succ_coe [NoMaxOrder α] {a : α} : succ (↑a : WithTop α) = ↑(succ a) :=
  succ_coe_of_not_isMax <| not_isMax a


instance : PredOrder (WithTop α) where
  pred a :=
    match a with
    | ⊤ => some ⊤
    | Option.some a => some (pred a)
  pred_le a :=
    match a with
    | ⊤ => le_top
    | Option.some a => coe_le_coe.2 (pred_le a)
  min_of_le_pred {a} ha := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : OrderTop α
      inst✝ : PredOrder α
      a : WithTop α
      ha : LE.le a ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a ( …
      ⊢ IsMin a
    -/
    cases a
      /-
        case top
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderTop α
        inst✝ : PredOrder α
        ha : LE.le Top.top ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop  …
        ⊢ IsMin Top.top
      -/
    · exact ((coe_lt_top (⊤ : α)).not_le ha).elim
      /-
        🎉 no goals
      -/
      /-
        case coe
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderTop α
        inst✝ : PredOrder α
        a✝ : α
        ha : LE.le (↑a✝) ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) …
        ⊢ IsMin ↑a✝
      -/
    · exact (min_of_le_pred <| coe_le_coe.1 ha).withTop
      /-
        🎉 no goals
      -/
  le_pred_of_lt {a b} h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : OrderTop α
      inst✝ : PredOrder α
      a b : WithTop α
      h : LT.lt a b
      ⊢ LE.le a ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a (fun …
    -/
    cases a
      /-
        case top
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderTop α
        inst✝ : PredOrder α
        b : WithTop α
        h : LT.lt Top.top b
        ⊢ LE.le Top.top ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α)  …
      -/
    · exact (le_top.not_lt h).elim
      /-
        🎉 no goals
      -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : OrderTop α
      inst✝ : PredOrder α
      b : WithTop α
      a✝ : α
      h : LT.lt (↑a✝) b
      ⊢ LE.le (↑a✝) ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a  …
    -/
    cases b
      /-
        case coe.top
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderTop α
        inst✝ : PredOrder α
        a✝ : α
        h : LT.lt (↑a✝) Top.top
        ⊢ LE.le (↑a✝) ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a  …
      -/
    · exact coe_le_coe.2 le_top
      /-
        🎉 no goals
      -/
    /-
      case coe.coe
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : OrderTop α
      inst✝ : PredOrder α
      a✝¹ a✝ : α
      h : LT.lt ↑a✝¹ ↑a✝
      ⊢ LE.le (↑a✝¹) ((fun a => WithTop.instSuccOrder.match_1 (fun a => WithTop α) a …
    -/
    exact coe_le_coe.2 (le_pred_of_lt <| coe_lt_coe.1 h)
    /-
      🎉 no goals
    -/


/-- Not to be confused with `WithTop.pred_bot`, which is about `WithTop.pred`. -/
@[simp] lemma orderPred_top : pred (⊤ : WithTop α) = ↑(⊤ : α) := rfl


/-- Not to be confused with `WithTop.pred_coe`, which is about `WithTop.pred`. -/
@[simp] lemma orderPred_coe (a : α) : pred (↑a : WithTop α) = ↑(pred a) := rfl


@[simp]
theorem pred_untop :
    ∀ (a : WithTop α) (ha : a ≠ ⊤),
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               inst✝² : Preorder α
                                               inst✝¹ : OrderTop α
                                               inst✝ : PredOrder α
                                               a : WithTop α
                                               ha : Ne a Top.top
                                               ⊢ Ne (Order.pred a) Top.top
                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
      pred (a.untop ha) = (pred a).untop (by induction a <;> simp)
                                                             /-
                                                               🎉 no goals
                                                             -/
  | ⊤, ha => (ha rfl).elim
  | (a : α), _ => rfl


instance [hα : Nonempty α] : IsEmpty (PredOrder (WithTop α)) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : NoMaxOrder α
      hα : Nonempty α
      ⊢ PredOrder (WithTop α) → False
    -/
    intro
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : NoMaxOrder α
      hα : Nonempty α
      a✝ : PredOrder (WithTop α)
      ⊢ False
    -/
    cases' h : pred (⊤ : WithTop α) with a ha
      /-
        case top
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : NoMaxOrder α
        hα : Nonempty α
        a✝ : PredOrder (WithTop α)
        h : Eq (Order.pred Top.top) Top.top
        ⊢ False
      -/
    · exact hα.elim fun a => (min_of_le_pred h.ge).not_lt <| coe_lt_top a
      /-
        🎉 no goals
      -/
      /-
        case coe
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : NoMaxOrder α
        hα : Nonempty α
        a✝ : PredOrder (WithTop α)
        a : α
        h : Eq (Order.pred Top.top) ↑a
        ⊢ False
      -/
    · obtain ⟨c, hc⟩ := exists_gt a
      /-
        case coe.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : NoMaxOrder α
        hα : Nonempty α
        a✝ : PredOrder (WithTop α)
        a : α
        h : Eq (Order.pred Top.top) ↑a
        c : α
        hc : LT.lt a c
        ⊢ False
      -/
      rw [← coe_lt_coe, ← h] at hc
      /-
        case coe.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : NoMaxOrder α
        hα : Nonempty α
        a✝ : PredOrder (WithTop α)
        a : α
        h : Eq (Order.pred Top.top) ↑a
        c : α
        hc : LT.lt (Order.pred Top.top) ↑c
        ⊢ False
      -/
      exact (le_pred_of_lt (coe_lt_top c)).not_lt hc⟩
      /-
        🎉 no goals
      -/


instance : SuccOrder (WithBot α) where
  succ a :=
    match a with
    | ⊥ => some ⊥
    | Option.some a => some (succ a)
  le_succ a :=
    match a with
    | ⊥ => bot_le
    | Option.some a => coe_le_coe.2 (le_succ a)
  max_of_succ_le {a} ha := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : OrderBot α
      inst✝ : SuccOrder α
      a : WithBot α
      ha : LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fu …
      ⊢ IsMax a
    -/
    cases a
      /-
        case bot
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderBot α
        inst✝ : SuccOrder α
        ha : LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fu …
        ⊢ IsMax Bot.bot
      -/
    · exact ((bot_lt_coe (⊥ : α)).not_le ha).elim
      /-
        🎉 no goals
      -/
      /-
        case coe
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderBot α
        inst✝ : SuccOrder α
        a✝ : α
        ha : LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fu …
        ⊢ IsMax ↑a✝
      -/
    · exact (max_of_succ_le <| coe_le_coe.1 ha).withBot
      /-
        🎉 no goals
      -/
  succ_le_of_lt {a b} h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : OrderBot α
      inst✝ : SuccOrder α
      a b : WithBot α
      h : LT.lt a b
      ⊢ LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun _ …
    -/
    cases b
      /-
        case bot
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderBot α
        inst✝ : SuccOrder α
        a : WithBot α
        h : LT.lt a Bot.bot
        ⊢ LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun _ …
      -/
    · exact (not_lt_bot h).elim
      /-
        🎉 no goals
      -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : Preorder α
      inst✝¹ : OrderBot α
      inst✝ : SuccOrder α
      a : WithBot α
      a✝ : α
      h : LT.lt a ↑a✝
      ⊢ LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun _ …
    -/
    cases a
      /-
        case coe.bot
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderBot α
        inst✝ : SuccOrder α
        a✝ : α
        h : LT.lt Bot.bot ↑a✝
        ⊢ LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun _ …
      -/
    · exact coe_le_coe.2 bot_le
      /-
        🎉 no goals
      -/
      /-
        case coe.coe
        α : Type u_1
        β : Type u_2
        inst✝² : Preorder α
        inst✝¹ : OrderBot α
        inst✝ : SuccOrder α
        a✝¹ a✝ : α
        h : LT.lt ↑a✝ ↑a✝¹
        ⊢ LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun _ …
      -/
    · exact coe_le_coe.2 (succ_le_of_lt <| coe_lt_coe.1 h)
      /-
        🎉 no goals
      -/


/-- Not to be confused with `WithBot.succ_bot`, which is about `WithBot.succ`. -/
@[simp] lemma orderSucc_bot : succ (⊥ : WithBot α) = ↑(⊥ : α) := rfl


/-- Not to be confused with `WithBot.succ_coe`, which is about `WithBot.succ`. -/
@[simp] lemma orderSucc_coe (a : α) : succ (↑a : WithBot α) = ↑(succ a) := rfl


@[simp]
theorem succ_unbot :
    ∀ (a : WithBot α) (ha : a ≠ ⊥),
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               inst✝² : Preorder α
                                               inst✝¹ : OrderBot α
                                               inst✝ : SuccOrder α
                                               a : WithBot α
                                               ha : Ne a Bot.bot
                                               ⊢ Ne (Order.succ a) Bot.bot
                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
      succ (a.unbot ha) = (succ a).unbot (by induction a <;> simp)
                                                             /-
                                                               🎉 no goals
                                                             -/
  | ⊥, ha => (ha rfl).elim
  | (a : α), _ => rfl


instance : PredOrder (WithBot α) where
  pred a :=
    match a with
    | ⊥ => ⊥
    | Option.some a => ite (pred a = a) ⊥ (some (pred a))
  pred_le a := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a : WithBot α
      ⊢ LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun _ …
    -/
    cases' a with a a
      /-
        case bot
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        ⊢ LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun _ …
      -/
    · exact bot_le
      /-
        🎉 no goals
      -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a : α
      ⊢ LE.le ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun _ …
    -/
    change ite _ _ _ ≤ _
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a : α
      ⊢ LE.le (ite (Eq (Order.pred a) a) Bot.bot ↑(Order.pred a)) ↑a
    -/
    split_ifs
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a : α
        h✝ : Eq (Order.pred a) a
        ⊢ LE.le Bot.bot ↑a
      -/
    · exact bot_le
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a : α
        h✝ : Not (Eq (Order.pred a) a)
        ⊢ LE.le ↑(Order.pred a) ↑a
      -/
    · exact coe_le_coe.2 (pred_le a)
      /-
        🎉 no goals
      -/
  min_of_le_pred {a} ha := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a : WithBot α
      ha : LE.le a ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a ( …
      ⊢ IsMin a
    -/
    cases' a with a a
      /-
        case bot
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        ha : LE.le Bot.bot ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot  …
        ⊢ IsMin Bot.bot
      -/
    · exact isMin_bot
      /-
        🎉 no goals
      -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a : α
      ha : LE.le (↑a) ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α)  …
      ⊢ IsMin ↑a
    -/
    dsimp only at ha
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a : α
      ha : LE.le (↑a) (ite (Eq (Order.pred a) a) Bot.bot ↑(Order.pred a))
      ⊢ IsMin ↑a
    -/
    split_ifs at ha with ha'
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a : α
        ha' : Eq (Order.pred a) a
        ha : LE.le (↑a) Bot.bot
        ⊢ IsMin ↑a
      -/
    · exact (not_coe_le_bot _ ha).elim
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a : α
        ha' : Not (Eq (Order.pred a) a)
        ha : LE.le ↑a ↑(Order.pred a)
        ⊢ IsMin ↑a
      -/
    · rw [coe_le_coe, le_pred_iff_isMin, ← pred_eq_iff_isMin] at ha
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a : α
        ha' : Not (Eq (Order.pred a) a)
        ha : Eq (Order.pred a) a
        ⊢ IsMin ↑a
      -/
      exact (ha' ha).elim
      /-
        🎉 no goals
      -/
  le_pred_of_lt {a b} h := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a b : WithBot α
      h : LT.lt a b
      ⊢ LE.le a ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a (fun …
    -/
    cases a
      /-
        case bot
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        b : WithBot α
        h : LT.lt Bot.bot b
        ⊢ LE.le Bot.bot ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α)  …
      -/
    · exact bot_le
      /-
        🎉 no goals
      -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      b : WithBot α
      a✝ : α
      h : LT.lt (↑a✝) b
      ⊢ LE.le (↑a✝) ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a  …
    -/
    cases b
      /-
        case coe.bot
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a✝ : α
        h : LT.lt (↑a✝) Bot.bot
        ⊢ LE.le (↑a✝) ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a  …
      -/
    · exact (not_lt_bot h).elim
      /-
        🎉 no goals
      -/
    /-
      case coe.coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a✝¹ a✝ : α
      h : LT.lt ↑a✝¹ ↑a✝
      ⊢ LE.le (↑a✝¹) ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a …
    -/
    rw [coe_lt_coe] at h
    /-
      case coe.coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a✝¹ a✝ : α
      h : LT.lt a✝¹ a✝
      ⊢ LE.le (↑a✝¹) ((fun a => WithBot.instSuccOrder.match_1 (fun a => WithBot α) a …
    -/
    change _ ≤ ite _ _ _
    /-
      case coe.coe
      α : Type u_1
      β : Type u_2
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
      a✝¹ a✝ : α
      h : LT.lt a✝¹ a✝
      ⊢ LE.le (↑a✝¹) (ite (Eq (Order.pred a✝) a✝) Bot.bot ↑(Order.pred a✝))
    -/
    split_ifs with hb
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a✝¹ a✝ : α
        h : LT.lt a✝¹ a✝
        hb : Eq (Order.pred a✝) a✝
        ⊢ LE.le (↑a✝¹) Bot.bot
      -/
    · rw [pred_eq_iff_isMin] at hb
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a✝¹ a✝ : α
        h : LT.lt a✝¹ a✝
        hb : IsMin a✝
        ⊢ LE.le (↑a✝¹) Bot.bot
      -/
      exact (hb.not_lt h).elim
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : PartialOrder α
        inst✝¹ : PredOrder α
        inst✝ : (a : α) → Decidable (Eq (Order.pred a) a)
        a✝¹ a✝ : α
        h : LT.lt a✝¹ a✝
        hb : Not (Eq (Order.pred a✝) a✝)
        ⊢ LE.le ↑a✝¹ ↑(Order.pred a✝)
      -/
    · exact coe_le_coe.2 (le_pred_of_lt h)
      /-
        🎉 no goals
      -/


@[simp]
theorem pred_coe_of_isMin {a : α} (h : IsMin a) : pred ↑a = (⊥ : WithBot α) :=
  dif_pos (pred_eq_iff_isMin.2 h)


theorem pred_coe_of_not_isMin {a : α} (h : ¬ IsMin a) : pred (↑a : WithBot α) = ↑(pred a) :=
  dif_neg (pred_eq_iff_isMin.not.2 h)


theorem pred_coe [NoMinOrder α] {a : α} : pred (↑a : WithBot α) = ↑(pred a) :=
  pred_coe_of_not_isMin <| not_isMin a


instance [hα : Nonempty α] : IsEmpty (SuccOrder (WithBot α)) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : NoMinOrder α
      hα : Nonempty α
      ⊢ SuccOrder (WithBot α) → False
    -/
    intro
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : NoMinOrder α
      hα : Nonempty α
      a✝ : SuccOrder (WithBot α)
      ⊢ False
    -/
    cases' h : succ (⊥ : WithBot α) with a ha
      /-
        case bot
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : NoMinOrder α
        hα : Nonempty α
        a✝ : SuccOrder (WithBot α)
        h : Eq (Order.succ Bot.bot) Bot.bot
        ⊢ False
      -/
    · exact hα.elim fun a => (max_of_succ_le h.le).not_lt <| bot_lt_coe a
      /-
        🎉 no goals
      -/
      /-
        case coe
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : NoMinOrder α
        hα : Nonempty α
        a✝ : SuccOrder (WithBot α)
        a : α
        h : Eq (Order.succ Bot.bot) ↑a
        ⊢ False
      -/
    · obtain ⟨c, hc⟩ := exists_lt a
      /-
        case coe.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : NoMinOrder α
        hα : Nonempty α
        a✝ : SuccOrder (WithBot α)
        a : α
        h : Eq (Order.succ Bot.bot) ↑a
        c : α
        hc : LT.lt c a
        ⊢ False
      -/
      rw [← coe_lt_coe, ← h] at hc
      /-
        case coe.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : Preorder α
        inst✝ : NoMinOrder α
        hα : Nonempty α
        a✝ : SuccOrder (WithBot α)
        a : α
        h : Eq (Order.succ Bot.bot) ↑a
        c : α
        hc : LT.lt (↑c) (Order.succ Bot.bot)
        ⊢ False
      -/
      exact (succ_le_of_lt (bot_lt_coe _)).not_lt hc⟩
      /-
        🎉 no goals
      -/


/-- `SuccOrder` transfers across equivalences between orders. -/
protected abbrev SuccOrder.ofOrderIso [SuccOrder X] (f : X ≃o Y) : SuccOrder Y where
  succ y := f (succ (f.symm y))
                  /-
                    α : Type u_1
                    β : Type u_2
                    X : Type u_3
                    Y : Type u_4
                    inst✝² : Preorder X
                    inst✝¹ : Preorder Y
                    inst✝ : SuccOrder X
                    f : OrderIso X Y
                    y : Y
                    ⊢ LE.le y ((fun y => f (SuccOrder.succ (f.symm y))) y)
                  -/
  le_succ y := by rw [← map_inv_le_iff f]; exact le_succ (f.symm y)
                                           /-
                                             🎉 no goals
                                           -/
  max_of_succ_le h := by
    /-
      α : Type u_1
      β : Type u_2
      X : Type u_3
      Y : Type u_4
      inst✝² : Preorder X
      inst✝¹ : Preorder Y
      inst✝ : SuccOrder X
      f : OrderIso X Y
      a✝ : Y
      h : LE.le ((fun y => f (SuccOrder.succ (f.symm y))) a✝) a✝
      ⊢ IsMax a✝
    -/
    rw [← f.symm.isMax_apply]
    /-
      α : Type u_1
      β : Type u_2
      X : Type u_3
      Y : Type u_4
      inst✝² : Preorder X
      inst✝¹ : Preorder Y
      inst✝ : SuccOrder X
      f : OrderIso X Y
      a✝ : Y
      h : LE.le ((fun y => f (SuccOrder.succ (f.symm y))) a✝) a✝
      ⊢ IsMax (f.symm a✝)
    -/
    refine max_of_succ_le ?_
    /-
      α : Type u_1
      β : Type u_2
      X : Type u_3
      Y : Type u_4
      inst✝² : Preorder X
      inst✝¹ : Preorder Y
      inst✝ : SuccOrder X
      f : OrderIso X Y
      a✝ : Y
      h : LE.le ((fun y => f (SuccOrder.succ (f.symm y))) a✝) a✝
      ⊢ LE.le (SuccOrder.succ (f.symm a✝)) (f.symm a✝)
    -/
    simp [f.le_symm_apply, h]
    /-
      🎉 no goals
    -/
                        /-
                          α : Type u_1
                          β : Type u_2
                          X : Type u_3
                          Y : Type u_4
                          inst✝² : Preorder X
                          inst✝¹ : Preorder Y
                          inst✝ : SuccOrder X
                          f : OrderIso X Y
                          a✝ b✝ : Y
                          h : LT.lt a✝ b✝
                          ⊢ LE.le ((fun y => f (SuccOrder.succ (f.symm y))) a✝) b✝
                        -/
  succ_le_of_lt h := by rw [← le_map_inv_iff]; exact succ_le_of_lt (by simp [h])
                                               /-
                                                 🎉 no goals
                                               -/

-- See note [reducible non instances]

/-- `PredOrder` transfers across equivalences between orders. -/
protected abbrev PredOrder.ofOrderIso [PredOrder X] (f : X ≃o Y) :
    PredOrder Y where
  pred y := f (pred (f.symm y))
                  /-
                    α : Type u_1
                    β : Type u_2
                    X : Type u_3
                    Y : Type u_4
                    inst✝² : Preorder X
                    inst✝¹ : Preorder Y
                    inst✝ : PredOrder X
                    f : OrderIso X Y
                    y : Y
                    ⊢ LE.le ((fun y => f (PredOrder.pred (f.symm y))) y) y
                  -/
  pred_le y := by rw [← le_map_inv_iff f]; exact pred_le (f.symm y)
                                           /-
                                             🎉 no goals
                                           -/
  min_of_le_pred h := by
    /-
      α : Type u_1
      β : Type u_2
      X : Type u_3
      Y : Type u_4
      inst✝² : Preorder X
      inst✝¹ : Preorder Y
      inst✝ : PredOrder X
      f : OrderIso X Y
      a✝ : Y
      h : LE.le a✝ ((fun y => f (PredOrder.pred (f.symm y))) a✝)
      ⊢ IsMin a✝
    -/
    rw [← f.symm.isMin_apply]
    /-
      α : Type u_1
      β : Type u_2
      X : Type u_3
      Y : Type u_4
      inst✝² : Preorder X
      inst✝¹ : Preorder Y
      inst✝ : PredOrder X
      f : OrderIso X Y
      a✝ : Y
      h : LE.le a✝ ((fun y => f (PredOrder.pred (f.symm y))) a✝)
      ⊢ IsMin (f.symm a✝)
    -/
    refine min_of_le_pred ?_
    /-
      α : Type u_1
      β : Type u_2
      X : Type u_3
      Y : Type u_4
      inst✝² : Preorder X
      inst✝¹ : Preorder Y
      inst✝ : PredOrder X
      f : OrderIso X Y
      a✝ : Y
      h : LE.le a✝ ((fun y => f (PredOrder.pred (f.symm y))) a✝)
      ⊢ LE.le (f.symm a✝) (PredOrder.pred (f.symm a✝))
    -/
    simp [f.symm_apply_le, h]
    /-
      🎉 no goals
    -/
                        /-
                          α : Type u_1
                          β : Type u_2
                          X : Type u_3
                          Y : Type u_4
                          inst✝² : Preorder X
                          inst✝¹ : Preorder Y
                          inst✝ : PredOrder X
                          f : OrderIso X Y
                          a✝ b✝ : Y
                          h : LT.lt a✝ b✝
                          ⊢ LE.le a✝ ((fun y => f (PredOrder.pred (f.symm y))) b✝)
                        -/
  le_pred_of_lt h := by rw [← map_inv_le_iff]; exact le_pred_of_lt (by simp [h])
                                               /-
                                                 🎉 no goals
                                               -/


open scoped Classical in
noncomputable instance Set.OrdConnected.predOrder [PredOrder α] :
    PredOrder s where
  pred x := if h : Order.pred x.1 ∈ s then ⟨Order.pred x.1, h⟩ else x
                              /-
                                α✝ : Type u_1
                                β : Type u_2
                                α : Type u_3
                                inst✝² : PartialOrder α
                                s : Set α
                                inst✝¹ : s.OrdConnected
                                inst✝ : PredOrder α
                                x✝ : ↑s
                                x : α
                                hx : Membership.mem s x
                                ⊢ LE.le ((fun x => dite (Membership.mem s (Order.pred ↑x)) (fun h => ⟨Order.pr …
                              -/
                                               /-
                                                 🎉 no goals
                                               -/
  pred_le := fun ⟨x, hx⟩ ↦ by dsimp; split <;> simp_all [Order.pred_le]
                                               /-
                                                 🎉 no goals
                                               -/
  min_of_le_pred := @fun ⟨x, hx⟩ h ↦ by
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      inst✝² : PartialOrder α
      s : Set α
      inst✝¹ : s.OrdConnected
      inst✝ : PredOrder α
      x✝ : ↑s
      x : α
      hx : Membership.mem s x
      h : LE.le ⟨x, hx⟩ ((fun x => dite (Membership.mem s (Order.pred ↑x)) (fun h => …
      ⊢ IsMin ⟨x, hx⟩
    -/
    dsimp at h
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      inst✝² : PartialOrder α
      s : Set α
      inst✝¹ : s.OrdConnected
      inst✝ : PredOrder α
      x✝ : ↑s
      x : α
      hx : Membership.mem s x
      h : LE.le ⟨x, hx⟩ (dite (Membership.mem s (Order.pred x)) (fun h => ⟨Order.pre …
      ⊢ IsMin ⟨x, hx⟩
    -/
    split_ifs at h with h'
      /-
        case pos
        α✝ : Type u_1
        β : Type u_2
        α : Type u_3
        inst✝² : PartialOrder α
        s : Set α
        inst✝¹ : s.OrdConnected
        inst✝ : PredOrder α
        x✝ : ↑s
        x : α
        hx : Membership.mem s x
        h' : Membership.mem s (Order.pred x)
        h : LE.le ⟨x, hx⟩ ⟨Order.pred x, h'⟩
        ⊢ IsMin ⟨x, hx⟩
      -/
    · simp only [Subtype.mk_le_mk, Order.le_pred_iff_isMin] at h
      /-
        case pos
        α✝ : Type u_1
        β : Type u_2
        α : Type u_3
        inst✝² : PartialOrder α
        s : Set α
        inst✝¹ : s.OrdConnected
        inst✝ : PredOrder α
        x✝ : ↑s
        x : α
        hx : Membership.mem s x
        h' : Membership.mem s (Order.pred x)
        h : IsMin x
        ⊢ IsMin ⟨x, hx⟩
      -/
      rintro ⟨y, _⟩ hy
      /-
        case pos.mk
        α✝ : Type u_1
        β : Type u_2
        α : Type u_3
        inst✝² : PartialOrder α
        s : Set α
        inst✝¹ : s.OrdConnected
        inst✝ : PredOrder α
        x✝ : ↑s
        x : α
        hx : Membership.mem s x
        h' : Membership.mem s (Order.pred x)
        h : IsMin x
        y : α
        property✝ : Membership.mem s y
        hy : LE.le ⟨y, property✝⟩ ⟨x, hx⟩
        ⊢ LE.le ⟨x, hx⟩ ⟨y, property✝⟩
      -/
      simp [h hy]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α✝ : Type u_1
        β : Type u_2
        α : Type u_3
        inst✝² : PartialOrder α
        s : Set α
        inst✝¹ : s.OrdConnected
        inst✝ : PredOrder α
        x✝ : ↑s
        x : α
        hx : Membership.mem s x
        h' : Not (Membership.mem s (Order.pred x))
        h : LE.le ⟨x, hx⟩ ⟨x, hx⟩
        ⊢ IsMin ⟨x, hx⟩
      -/
    · rintro ⟨y, hy⟩ h
      /-
        case neg.mk
        α✝ : Type u_1
        β : Type u_2
        α : Type u_3
        inst✝² : PartialOrder α
        s : Set α
        inst✝¹ : s.OrdConnected
        inst✝ : PredOrder α
        x✝ : ↑s
        x : α
        hx : Membership.mem s x
        h' : Not (Membership.mem s (Order.pred x))
        h✝ : LE.le ⟨x, hx⟩ ⟨x, hx⟩
        y : α
        hy : Membership.mem s y
        h : LE.le ⟨y, hy⟩ ⟨x, hx⟩
        ⊢ LE.le ⟨x, hx⟩ ⟨y, hy⟩
      -/
      rcases h.lt_or_eq with h | h
        /-
          case neg.mk.inl
          α✝ : Type u_1
          β : Type u_2
          α : Type u_3
          inst✝² : PartialOrder α
          s : Set α
          inst✝¹ : s.OrdConnected
          inst✝ : PredOrder α
          x✝ : ↑s
          x : α
          hx : Membership.mem s x
          h' : Not (Membership.mem s (Order.pred x))
          h✝¹ : LE.le ⟨x, hx⟩ ⟨x, hx⟩
          y : α
          hy : Membership.mem s y
          h✝ : LE.le ⟨y, hy⟩ ⟨x, hx⟩
          h : LT.lt ⟨y, hy⟩ ⟨x, hx⟩
          ⊢ LE.le ⟨x, hx⟩ ⟨y, hy⟩
        -/
      · simp only [Subtype.mk_lt_mk] at h
        /-
          case neg.mk.inl
          α✝ : Type u_1
          β : Type u_2
          α : Type u_3
          inst✝² : PartialOrder α
          s : Set α
          inst✝¹ : s.OrdConnected
          inst✝ : PredOrder α
          x✝ : ↑s
          x : α
          hx : Membership.mem s x
          h' : Not (Membership.mem s (Order.pred x))
          h✝¹ : LE.le ⟨x, hx⟩ ⟨x, hx⟩
          y : α
          hy : Membership.mem s y
          h✝ : LE.le ⟨y, hy⟩ ⟨x, hx⟩
          h : LT.lt y x
          ⊢ LE.le ⟨x, hx⟩ ⟨y, hy⟩
        -/
        have := h.le_pred
        /-
          case neg.mk.inl
          α✝ : Type u_1
          β : Type u_2
          α : Type u_3
          inst✝² : PartialOrder α
          s : Set α
          inst✝¹ : s.OrdConnected
          inst✝ : PredOrder α
          x✝ : ↑s
          x : α
          hx : Membership.mem s x
          h' : Not (Membership.mem s (Order.pred x))
          h✝¹ : LE.le ⟨x, hx⟩ ⟨x, hx⟩
          y : α
          hy : Membership.mem s y
          h✝ : LE.le ⟨y, hy⟩ ⟨x, hx⟩
          h : LT.lt y x
          this : LE.le y (Order.pred x)
          ⊢ LE.le ⟨x, hx⟩ ⟨y, hy⟩
        -/
        absurd h'
        /-
          case neg.mk.inl
          α✝ : Type u_1
          β : Type u_2
          α : Type u_3
          inst✝² : PartialOrder α
          s : Set α
          inst✝¹ : s.OrdConnected
          inst✝ : PredOrder α
          x✝ : ↑s
          x : α
          hx : Membership.mem s x
          h' : Not (Membership.mem s (Order.pred x))
          h✝¹ : LE.le ⟨x, hx⟩ ⟨x, hx⟩
          y : α
          hy : Membership.mem s y
          h✝ : LE.le ⟨y, hy⟩ ⟨x, hx⟩
          h : LT.lt y x
          this : LE.le y (Order.pred x)
          ⊢ Membership.mem s (Order.pred x)
        -/
        apply out' hy hx
        /-
          case neg.mk.inl.a
          α✝ : Type u_1
          β : Type u_2
          α : Type u_3
          inst✝² : PartialOrder α
          s : Set α
          inst✝¹ : s.OrdConnected
          inst✝ : PredOrder α
          x✝ : ↑s
          x : α
          hx : Membership.mem s x
          h' : Not (Membership.mem s (Order.pred x))
          h✝¹ : LE.le ⟨x, hx⟩ ⟨x, hx⟩
          y : α
          hy : Membership.mem s y
          h✝ : LE.le ⟨y, hy⟩ ⟨x, hx⟩
          h : LT.lt y x
          this : LE.le y (Order.pred x)
          ⊢ Membership.mem (Set.Icc y x) (Order.pred x)
        -/
        simp [this, Order.pred_le]
        /-
          🎉 no goals
        -/
        /-
          case neg.mk.inr
          α✝ : Type u_1
          β : Type u_2
          α : Type u_3
          inst✝² : PartialOrder α
          s : Set α
          inst✝¹ : s.OrdConnected
          inst✝ : PredOrder α
          x✝ : ↑s
          x : α
          hx : Membership.mem s x
          h' : Not (Membership.mem s (Order.pred x))
          h✝¹ : LE.le ⟨x, hx⟩ ⟨x, hx⟩
          y : α
          hy : Membership.mem s y
          h✝ : LE.le ⟨y, hy⟩ ⟨x, hx⟩
          h : Eq ⟨y, hy⟩ ⟨x, hx⟩
          ⊢ LE.le ⟨x, hx⟩ ⟨y, hy⟩
        -/
      · simp [h]
        /-
          🎉 no goals
        -/
  le_pred_of_lt := @fun ⟨b, hb⟩ ⟨c, hc⟩ h ↦ by
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      inst✝² : PartialOrder α
      s : Set α
      inst✝¹ : s.OrdConnected
      inst✝ : PredOrder α
      x✝¹ x✝ : ↑s
      b : α
      hb : Membership.mem s b
      c : α
      hc : Membership.mem s c
      h : LT.lt ⟨b, hb⟩ ⟨c, hc⟩
      ⊢ LE.le ⟨b, hb⟩ ((fun x => dite (Membership.mem s (Order.pred ↑x)) (fun h => ⟨ …
    -/
    rw [Subtype.mk_lt_mk] at h
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      inst✝² : PartialOrder α
      s : Set α
      inst✝¹ : s.OrdConnected
      inst✝ : PredOrder α
      x✝¹ x✝ : ↑s
      b : α
      hb : Membership.mem s b
      c : α
      hc : Membership.mem s c
      h : LT.lt b c
      ⊢ LE.le ⟨b, hb⟩ ((fun x => dite (Membership.mem s (Order.pred ↑x)) (fun h => ⟨ …
    -/
    dsimp only
    /-
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      inst✝² : PartialOrder α
      s : Set α
      inst✝¹ : s.OrdConnected
      inst✝ : PredOrder α
      x✝¹ x✝ : ↑s
      b : α
      hb : Membership.mem s b
      c : α
      hc : Membership.mem s c
      h : LT.lt b c
      ⊢ LE.le ⟨b, hb⟩ (dite (Membership.mem s (Order.pred c)) (fun h => ⟨Order.pred  …
    -/
    split
      /-
        case isTrue
        α✝ : Type u_1
        β : Type u_2
        α : Type u_3
        inst✝² : PartialOrder α
        s : Set α
        inst✝¹ : s.OrdConnected
        inst✝ : PredOrder α
        x✝¹ x✝ : ↑s
        b : α
        hb : Membership.mem s b
        c : α
        hc : Membership.mem s c
        h : LT.lt b c
        h✝ : Membership.mem s (Order.pred c)
        ⊢ LE.le ⟨b, hb⟩ ⟨Order.pred c, h✝⟩
      -/
    · exact h.le_pred
      /-
        🎉 no goals
      -/
      /-
        case isFalse
        α✝ : Type u_1
        β : Type u_2
        α : Type u_3
        inst✝² : PartialOrder α
        s : Set α
        inst✝¹ : s.OrdConnected
        inst✝ : PredOrder α
        x✝¹ x✝ : ↑s
        b : α
        hb : Membership.mem s b
        c : α
        hc : Membership.mem s c
        h : LT.lt b c
        h✝ : Not (Membership.mem s (Order.pred c))
        ⊢ LE.le ⟨b, hb⟩ ⟨c, hc⟩
      -/
    · exact h.le
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
lemma coe_pred_of_mem [PredOrder α] {a : s} (h : pred a.1 ∈ s) :
    (pred a).1 = pred ↑a := by classical
  change Subtype.val (dite ..) = _
  simp [h]


lemma isMin_of_not_pred_mem [PredOrder α] {a : s} (h : pred ↑a ∉ s) : IsMin a := by classical
  rw [← pred_eq_iff_isMin]
  change dite .. = _
  simp [h]


lemma not_pred_mem_iff_isMin [PredOrder α] [NoMinOrder α] {a : s} :
    pred ↑a ∉ s ↔ IsMin a where
  mp := isMin_of_not_pred_mem
  mpr h nh := by
    /-
      α : Type u_3
      inst✝³ : PartialOrder α
      s : Set α
      inst✝² : s.OrdConnected
      inst✝¹ : PredOrder α
      inst✝ : NoMinOrder α
      a : ↑s
      h : IsMin a
      nh : Membership.mem s (Order.pred ↑a)
      ⊢ False
    -/
    replace h := congr($h.pred_eq.1)
    /-
      α : Type u_3
      inst✝³ : PartialOrder α
      s : Set α
      inst✝² : s.OrdConnected
      inst✝¹ : PredOrder α
      inst✝ : NoMinOrder α
      a : ↑s
      nh : Membership.mem s (Order.pred ↑a)
      h : Eq ↑(Order.pred a) ↑a
      ⊢ False
    -/
    rw [coe_pred_of_mem nh] at h
    /-
      α : Type u_3
      inst✝³ : PartialOrder α
      s : Set α
      inst✝² : s.OrdConnected
      inst✝¹ : PredOrder α
      inst✝ : NoMinOrder α
      a : ↑s
      nh : Membership.mem s (Order.pred ↑a)
      h : Eq (Order.pred ↑a) ↑a
      ⊢ False
    -/
    simp at h
    /-
      🎉 no goals
    -/


noncomputable instance Set.OrdConnected.succOrder [SuccOrder α] :
    SuccOrder s :=
  letI : PredOrder sᵒᵈ := inferInstanceAs (PredOrder (OrderDual.ofDual ⁻¹' s))
  inferInstanceAs (SuccOrder sᵒᵈᵒᵈ)


@[simp, norm_cast]
lemma coe_succ_of_mem [SuccOrder α] {a : s} (h : succ ↑a ∈ s) :
    (succ a).1 = succ ↑a := by classical
  change Subtype.val (dite ..) = _
  split_ifs <;> trivial


lemma isMax_of_not_succ_mem [SuccOrder α] {a : s} (h : succ ↑a ∉ s) : IsMax a := by classical
  rw [← succ_eq_iff_isMax]
  change dite .. = _
  split_ifs <;> trivial


lemma not_succ_mem_iff_isMax [SuccOrder α] [NoMaxOrder α] {a : s} :
    succ ↑a ∉ s ↔ IsMax a where
  mp := isMax_of_not_succ_mem
  mpr h nh := by
    /-
      α : Type u_3
      inst✝³ : PartialOrder α
      s : Set α
      inst✝² : s.OrdConnected
      inst✝¹ : SuccOrder α
      inst✝ : NoMaxOrder α
      a : ↑s
      h : IsMax a
      nh : Membership.mem s (Order.succ ↑a)
      ⊢ False
    -/
    replace h := congr($h.succ_eq.1)
    /-
      α : Type u_3
      inst✝³ : PartialOrder α
      s : Set α
      inst✝² : s.OrdConnected
      inst✝¹ : SuccOrder α
      inst✝ : NoMaxOrder α
      a : ↑s
      nh : Membership.mem s (Order.succ ↑a)
      h : Eq ↑(Order.succ a) ↑a
      ⊢ False
    -/
    rw [coe_succ_of_mem nh] at h
    /-
      α : Type u_3
      inst✝³ : PartialOrder α
      s : Set α
      inst✝² : s.OrdConnected
      inst✝¹ : SuccOrder α
      inst✝ : NoMaxOrder α
      a : ↑s
      nh : Membership.mem s (Order.succ ↑a)
      h : Eq (Order.succ ↑a) ↑a
      ⊢ False
    -/
    simp at h
    /-
      🎉 no goals
    -/


