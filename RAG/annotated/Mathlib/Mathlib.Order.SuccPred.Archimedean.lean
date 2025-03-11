/-- A `SuccOrder` is succ-archimedean if one can go from any two comparable elements by iterating
`succ` -/
class IsSuccArchimedean (α : Type*) [Preorder α] [SuccOrder α] : Prop where
  /-- If `a ≤ b` then one can get to `a` from `b` by iterating `succ` -/
  exists_succ_iterate_of_le {a b : α} (h : a ≤ b) : ∃ n, succ^[n] a = b


/-- A `PredOrder` is pred-archimedean if one can go from any two comparable elements by iterating
`pred` -/
class IsPredArchimedean (α : Type*) [Preorder α] [PredOrder α] : Prop where
  /-- If `a ≤ b` then one can get to `b` from `a` by iterating `pred` -/
  exists_pred_iterate_of_le {a b : α} (h : a ≤ b) : ∃ n, pred^[n] b = a


instance : IsPredArchimedean αᵒᵈ :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝² : Preorder α
                       inst✝¹ : SuccOrder α
                       inst✝ : IsSuccArchimedean α
                       a✝ b✝ : α
                       a b : OrderDual α
                       h : LE.le a b
                       ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
                     -/
  ⟨fun {a b} h => by convert exists_succ_iterate_of_le h.ofDual⟩
                     /-
                       🎉 no goals
                     -/


theorem LE.le.exists_succ_iterate (h : a ≤ b) : ∃ n, succ^[n] a = b :=
  exists_succ_iterate_of_le h


theorem exists_succ_iterate_iff_le : (∃ n, succ^[n] a = b) ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    a b : α
    ⊢ Iff (Exists fun n => Eq (Nat.iterate Order.succ n a) b) (LE.le a b)
  -/
  refine ⟨?_, exists_succ_iterate_of_le⟩
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    a b : α
    ⊢ (Exists fun n => Eq (Nat.iterate Order.succ n a) b) → LE.le a b
  -/
  rintro ⟨n, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    a : α
    n : Nat
    ⊢ LE.le a (Nat.iterate Order.succ n a)
  -/
  exact id_le_iterate_of_id_le le_succ n a
  /-
    🎉 no goals
  -/


/-- Induction principle on a type with a `SuccOrder` for all elements above a given element `m`. -/
@[elab_as_elim]
theorem Succ.rec {P : α → Prop} {m : α} (h0 : P m) (h1 : ∀ n, m ≤ n → P n → P (succ n)) ⦃n : α⦄
    (hmn : m ≤ n) : P n := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    P : α → Prop
    m : α
    h0 : P m
    h1 : ∀ (n : α), LE.le m n → P n → P (Order.succ n)
    n : α
    hmn : LE.le m n
    ⊢ P n
  -/
  obtain ⟨n, rfl⟩ := hmn.exists_succ_iterate; clear hmn
  /-
    case intro
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    P : α → Prop
    m : α
    h0 : P m
    h1 : ∀ (n : α), LE.le m n → P n → P (Order.succ n)
    n : Nat
    ⊢ P (Nat.iterate Order.succ n m)
  -/
  induction' n with n ih
    /-
      case intro.zero
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      P : α → Prop
      m : α
      h0 : P m
      h1 : ∀ (n : α), LE.le m n → P n → P (Order.succ n)
      ⊢ P (Nat.iterate Order.succ 0 m)
    -/
  · exact h0
    /-
      🎉 no goals
    -/
    /-
      case intro.succ
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      P : α → Prop
      m : α
      h0 : P m
      h1 : ∀ (n : α), LE.le m n → P n → P (Order.succ n)
      n : Nat
      ih : P (Nat.iterate Order.succ n m)
      ⊢ P (Nat.iterate Order.succ (HAdd.hAdd n 1) m)
    -/
  · rw [Function.iterate_succ_apply']
    /-
      case intro.succ
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      P : α → Prop
      m : α
      h0 : P m
      h1 : ∀ (n : α), LE.le m n → P n → P (Order.succ n)
      n : Nat
      ih : P (Nat.iterate Order.succ n m)
      ⊢ P (Order.succ (Nat.iterate Order.succ n m))
    -/
    exact h1 _ (id_le_iterate_of_id_le le_succ n m) ih
    /-
      🎉 no goals
    -/


theorem Succ.rec_iff {p : α → Prop} (hsucc : ∀ a, p a ↔ p (succ a)) {a b : α} (h : a ≤ b) :
    p a ↔ p b := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    p : α → Prop
    hsucc : ∀ (a : α), Iff (p a) (p (Order.succ a))
    a b : α
    h : LE.le a b
    ⊢ Iff (p a) (p b)
  -/
  obtain ⟨n, rfl⟩ := h.exists_succ_iterate
  /-
    case intro
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    p : α → Prop
    hsucc : ∀ (a : α), Iff (p a) (p (Order.succ a))
    a : α
    n : Nat
    h : LE.le a (Nat.iterate Order.succ n a)
    ⊢ Iff (p a) (p (Nat.iterate Order.succ n a))
  -/
  exact Iterate.rec (fun b => p a ↔ p b) (fun c hc => hc.trans (hsucc _)) Iff.rfl n
  /-
    🎉 no goals
  -/


lemma le_total_of_codirected {r v₁ v₂ : α} (h₁ : r ≤ v₁) (h₂ : r ≤ v₂) : v₁ ≤ v₂ ∨ v₂ ≤ v₁ := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r v₁ v₂ : α
    h₁ : LE.le r v₁
    h₂ : LE.le r v₂
    ⊢ Or (LE.le v₁ v₂) (LE.le v₂ v₁)
  -/
  obtain ⟨n, rfl⟩ := h₁.exists_succ_iterate
  /-
    case intro
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r v₂ : α
    h₂ : LE.le r v₂
    n : Nat
    h₁ : LE.le r (Nat.iterate Order.succ n r)
    ⊢ Or (LE.le (Nat.iterate Order.succ n r) v₂) (LE.le v₂ (Nat.iterate Order.succ …
  -/
  obtain ⟨m, rfl⟩ := h₂.exists_succ_iterate
  /-
    case intro.intro
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α
    n : Nat
    h₁ : LE.le r (Nat.iterate Order.succ n r)
    m : Nat
    h₂ : LE.le r (Nat.iterate Order.succ m r)
    ⊢ Or (LE.le (Nat.iterate Order.succ n r) (Nat.iterate Order.succ m r)) (LE.le  …
  -/
  clear h₁ h₂
  /-
    case intro.intro
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α
    n m : Nat
    ⊢ Or (LE.le (Nat.iterate Order.succ n r) (Nat.iterate Order.succ m r)) (LE.le  …
  -/
  wlog h : n ≤ m
    /-
      case intro.intro.inr
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α
      n m : Nat
      this : ∀ {α : Type u_1} [inst : Preorder α] [inst_1 : SuccOrder α] [inst_2 : I …
      h : Not (LE.le n m)
      ⊢ Or (LE.le (Nat.iterate Order.succ n r) (Nat.iterate Order.succ m r)) (LE.le  …
    -/
  · rw [Or.comm]
    /-
      case intro.intro.inr
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α
      n m : Nat
      this : ∀ {α : Type u_1} [inst : Preorder α] [inst_1 : SuccOrder α] [inst_2 : I …
      h : Not (LE.le n m)
      ⊢ Or (LE.le (Nat.iterate Order.succ m r) (Nat.iterate Order.succ n r)) (LE.le  …
    -/
    apply this
    /-
      case intro.intro.inr.h
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α
      n m : Nat
      this : ∀ {α : Type u_1} [inst : Preorder α] [inst_1 : SuccOrder α] [inst_2 : I …
      h : Not (LE.le n m)
      ⊢ LE.le m n
    -/
    exact Nat.le_of_not_ge h
    /-
      🎉 no goals
    -/
  /-
    α✝ : Type u_1
    inst✝³ : Preorder α✝
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α
    n m : Nat
    h : LE.le n m
    ⊢ Or (LE.le (Nat.iterate Order.succ n r) (Nat.iterate Order.succ m r)) (LE.le  …
  -/
  left
  /-
    case h
    α✝ : Type u_1
    inst✝³ : Preorder α✝
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α
    n m : Nat
    h : LE.le n m
    ⊢ LE.le (Nat.iterate Order.succ n r) (Nat.iterate Order.succ m r)
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h
  /-
    case h.intro
    α✝ : Type u_1
    inst✝³ : Preorder α✝
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α
    n k : Nat
    h : LE.le n (HAdd.hAdd n k)
    ⊢ LE.le (Nat.iterate Order.succ n r) (Nat.iterate Order.succ (HAdd.hAdd n k) r)
  -/
  rw [Nat.add_comm, Function.iterate_add, Function.comp_apply]
  /-
    case h.intro
    α✝ : Type u_1
    inst✝³ : Preorder α✝
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α
    n k : Nat
    h : LE.le n (HAdd.hAdd n k)
    ⊢ LE.le (Nat.iterate Order.succ n r) (Nat.iterate Order.succ k (Nat.iterate Or …
  -/
  apply Order.le_succ_iterate
  /-
    🎉 no goals
  -/


instance : IsSuccArchimedean αᵒᵈ :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       inst✝² : Preorder α
                       inst✝¹ : PredOrder α
                       inst✝ : IsPredArchimedean α
                       a✝ b✝ : α
                       a b : OrderDual α
                       h : LE.le a b
                       ⊢ Exists fun n => Eq (Nat.iterate Order.succ n a) b
                     -/
  ⟨fun {a b} h => by convert exists_pred_iterate_of_le h.ofDual⟩
                     /-
                       🎉 no goals
                     -/


theorem LE.le.exists_pred_iterate (h : a ≤ b) : ∃ n, pred^[n] b = a :=
  exists_pred_iterate_of_le h


theorem exists_pred_iterate_iff_le : (∃ n, pred^[n] b = a) ↔ a ≤ b :=
  exists_succ_iterate_iff_le (α := αᵒᵈ)


/-- Induction principle on a type with a `PredOrder` for all elements below a given element `m`. -/
@[elab_as_elim]
theorem Pred.rec {P : α → Prop} {m : α} (h0 : P m) (h1 : ∀ n, n ≤ m → P n → P (pred n)) ⦃n : α⦄
    (hmn : n ≤ m) : P n :=
  Succ.rec (α := αᵒᵈ) (P := P) h0 h1 hmn


theorem Pred.rec_iff {p : α → Prop} (hsucc : ∀ a, p a ↔ p (pred a)) {a b : α} (h : a ≤ b) :
    p a ↔ p b :=
  (Succ.rec_iff (α := αᵒᵈ) hsucc h).symm


lemma le_total_of_directed {r v₁ v₂ : α} (h₁ : v₁ ≤ r) (h₂ : v₂ ≤ r) : v₁ ≤ v₂ ∨ v₂ ≤ v₁ :=
  Or.symm (le_total_of_codirected (α := αᵒᵈ) h₁ h₂)


lemma lt_or_le_of_codirected [SuccOrder α] [IsSuccArchimedean α] {r v₁ v₂ : α} (h₁ : r ≤ v₁)
    (h₂ : r ≤ v₂) : v₁ < v₂ ∨ v₂ ≤ v₁ := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r v₁ v₂ : α
    h₁ : LE.le r v₁
    h₂ : LE.le r v₂
    ⊢ Or (LT.lt v₁ v₂) (LE.le v₂ v₁)
  -/
  rw [Classical.or_iff_not_imp_right]
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r v₁ v₂ : α
    h₁ : LE.le r v₁
    h₂ : LE.le r v₂
    ⊢ Not (LE.le v₂ v₁) → LT.lt v₁ v₂
  -/
  intro nh
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r v₁ v₂ : α
    h₁ : LE.le r v₁
    h₂ : LE.le r v₂
    nh : Not (LE.le v₂ v₁)
    ⊢ LT.lt v₁ v₂
  -/
  rcases le_total_of_codirected h₁ h₂ with h | h
    /-
      case inl
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r v₁ v₂ : α
      h₁ : LE.le r v₁
      h₂ : LE.le r v₂
      nh : Not (LE.le v₂ v₁)
      h : LE.le v₁ v₂
      ⊢ LT.lt v₁ v₂
    -/
  · apply lt_of_le_of_ne h (ne_of_not_le nh).symm
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r v₁ v₂ : α
      h₁ : LE.le r v₁
      h₂ : LE.le r v₂
      nh : Not (LE.le v₂ v₁)
      h : LE.le v₂ v₁
      ⊢ LT.lt v₁ v₂
    -/
  · contradiction
    /-
      🎉 no goals
    -/


/--
This isn't an instance due to a loop with `LinearOrder`.
-/
-- See note [reducible non instances]
abbrev IsSuccArchimedean.linearOrder [SuccOrder α] [IsSuccArchimedean α]
     [DecidableEq α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
     [IsDirected α (· ≥ ·)] : LinearOrder α where
  le_total a b :=
    have ⟨c, ha, hb⟩ := directed_of (· ≥ ·) a b
    le_total_of_codirected ha hb
  decidableEq := inferInstance
  decidableLE := inferInstance
  decidableLT := inferInstance


lemma lt_or_le_of_directed [PredOrder α] [IsPredArchimedean α] {r v₁ v₂ : α} (h₁ : v₁ ≤ r)
    (h₂ : v₂ ≤ r) : v₁ < v₂ ∨ v₂ ≤ v₁ := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    r v₁ v₂ : α
    h₁ : LE.le v₁ r
    h₂ : LE.le v₂ r
    ⊢ Or (LT.lt v₁ v₂) (LE.le v₂ v₁)
  -/
  rw [Classical.or_iff_not_imp_right]
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    r v₁ v₂ : α
    h₁ : LE.le v₁ r
    h₂ : LE.le v₂ r
    ⊢ Not (LE.le v₂ v₁) → LT.lt v₁ v₂
  -/
  intro nh
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    r v₁ v₂ : α
    h₁ : LE.le v₁ r
    h₂ : LE.le v₂ r
    nh : Not (LE.le v₂ v₁)
    ⊢ LT.lt v₁ v₂
  -/
  rcases le_total_of_directed h₁ h₂ with h | h
    /-
      case inl
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      r v₁ v₂ : α
      h₁ : LE.le v₁ r
      h₂ : LE.le v₂ r
      nh : Not (LE.le v₂ v₁)
      h : LE.le v₁ v₂
      ⊢ LT.lt v₁ v₂
    -/
  · apply lt_of_le_of_ne h (ne_of_not_le nh).symm
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      r v₁ v₂ : α
      h₁ : LE.le v₁ r
      h₂ : LE.le v₂ r
      nh : Not (LE.le v₂ v₁)
      h : LE.le v₂ v₁
      ⊢ LT.lt v₁ v₂
    -/
  · contradiction
    /-
      🎉 no goals
    -/


/--
This isn't an instance due to a loop with `LinearOrder`.
-/
-- See note [reducible non instances]
abbrev IsPredArchimedean.linearOrder [PredOrder α] [IsPredArchimedean α]
     [DecidableEq α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
     [IsDirected α (· ≤ ·)] : LinearOrder α :=
  letI : LinearOrder αᵒᵈ := IsSuccArchimedean.linearOrder
  inferInstanceAs (LinearOrder αᵒᵈᵒᵈ)


lemma succ_max (a b : α) : succ (max a b) = max (succ a) (succ b) := succ_mono.map_max

lemma succ_min (a b : α) : succ (min a b) = min (succ a) (succ b) := succ_mono.map_min


theorem exists_succ_iterate_or : (∃ n, succ^[n] a = b) ∨ ∃ n, succ^[n] b = a :=
  (le_total a b).imp exists_succ_iterate_of_le exists_succ_iterate_of_le


theorem Succ.rec_linear {p : α → Prop} (hsucc : ∀ a, p a ↔ p (succ a)) (a b : α) : p a ↔ p b :=
  (le_total a b).elim (Succ.rec_iff hsucc) fun h => (Succ.rec_iff hsucc h).symm


lemma pred_max (a b : α) : pred (max a b) = max (pred a) (pred b) := pred_mono.map_max

lemma pred_min (a b : α) : pred (min a b) = min (pred a) (pred b) := pred_mono.map_min


theorem exists_pred_iterate_or : (∃ n, pred^[n] b = a) ∨ ∃ n, pred^[n] a = b :=
  (le_total a b).imp exists_pred_iterate_of_le exists_pred_iterate_of_le


theorem Pred.rec_linear {p : α → Prop} (hsucc : ∀ a, p a ↔ p (pred a)) (a b : α) : p a ↔ p b :=
  (le_total a b).elim (Pred.rec_iff hsucc) fun h => (Pred.rec_iff hsucc h).symm


lemma StrictMono.not_bddAbove_range_of_isSuccArchimedean [NoMaxOrder α] [SuccOrder β]
    [IsSuccArchimedean β] (hf : StrictMono f) : ¬ BddAbove (Set.range f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Nonempty α
    inst✝³ : Preorder β
    f : α → β
    inst✝² : NoMaxOrder α
    inst✝¹ : SuccOrder β
    inst✝ : IsSuccArchimedean β
    hf : StrictMono f
    ⊢ Not (BddAbove (Set.range f))
  -/
  rintro ⟨m, hm⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Nonempty α
    inst✝³ : Preorder β
    f : α → β
    inst✝² : NoMaxOrder α
    inst✝¹ : SuccOrder β
    inst✝ : IsSuccArchimedean β
    hf : StrictMono f
    m : β
    hm : Membership.mem (upperBounds (Set.range f)) m
    ⊢ False
  -/
  have hm' : ∀ a, f a ≤ m := fun a ↦ hm <| Set.mem_range_self _
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Nonempty α
    inst✝³ : Preorder β
    f : α → β
    inst✝² : NoMaxOrder α
    inst✝¹ : SuccOrder β
    inst✝ : IsSuccArchimedean β
    hf : StrictMono f
    m : β
    hm : Membership.mem (upperBounds (Set.range f)) m
    hm' : ∀ (a : α), LE.le (f a) m
    ⊢ False
  -/
  obtain ⟨a₀⟩ := ‹Nonempty α›
  suffices ∀ b, f a₀ ≤ b → ∃ a, b < f a by
    obtain ⟨a, ha⟩ : ∃ a, m < f a := this m (hm' a₀)
    exact ha.not_le (hm' a)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Nonempty α
    inst✝³ : Preorder β
    f : α → β
    inst✝² : NoMaxOrder α
    inst✝¹ : SuccOrder β
    inst✝ : IsSuccArchimedean β
    hf : StrictMono f
    m : β
    hm : Membership.mem (upperBounds (Set.range f)) m
    hm' : ∀ (a : α), LE.le (f a) m
    a₀ : α
    ⊢ ∀ (b : β), LE.le (f a₀) b → Exists fun a => LT.lt b (f a)
  -/
  have h : ∀ a, ∃ a', f a < f a' := fun a ↦ (exists_gt a).imp (fun a' h ↦ hf h)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Nonempty α
    inst✝³ : Preorder β
    f : α → β
    inst✝² : NoMaxOrder α
    inst✝¹ : SuccOrder β
    inst✝ : IsSuccArchimedean β
    hf : StrictMono f
    m : β
    hm : Membership.mem (upperBounds (Set.range f)) m
    hm' : ∀ (a : α), LE.le (f a) m
    a₀ : α
    h : ∀ (a : α), Exists fun a' => LT.lt (f a) (f a')
    ⊢ ∀ (b : β), LE.le (f a₀) b → Exists fun a => LT.lt b (f a)
  -/
  apply Succ.rec
    /-
      case intro.intro.h0
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Nonempty α
      inst✝³ : Preorder β
      f : α → β
      inst✝² : NoMaxOrder α
      inst✝¹ : SuccOrder β
      inst✝ : IsSuccArchimedean β
      hf : StrictMono f
      m : β
      hm : Membership.mem (upperBounds (Set.range f)) m
      hm' : ∀ (a : α), LE.le (f a) m
      a₀ : α
      h : ∀ (a : α), Exists fun a' => LT.lt (f a) (f a')
      ⊢ Exists fun a => LT.lt (f a₀) (f a)
    -/
  · exact h a₀
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.h1
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Nonempty α
    inst✝³ : Preorder β
    f : α → β
    inst✝² : NoMaxOrder α
    inst✝¹ : SuccOrder β
    inst✝ : IsSuccArchimedean β
    hf : StrictMono f
    m : β
    hm : Membership.mem (upperBounds (Set.range f)) m
    hm' : ∀ (a : α), LE.le (f a) m
    a₀ : α
    h : ∀ (a : α), Exists fun a' => LT.lt (f a) (f a')
    ⊢ ∀ (n : β), LE.le (f a₀) n → (Exists fun a => LT.lt n (f a)) → Exists fun a = …
  -/
  rintro b _ ⟨a, hba⟩
  /-
    case intro.intro.h1.intro
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Nonempty α
    inst✝³ : Preorder β
    f : α → β
    inst✝² : NoMaxOrder α
    inst✝¹ : SuccOrder β
    inst✝ : IsSuccArchimedean β
    hf : StrictMono f
    m : β
    hm : Membership.mem (upperBounds (Set.range f)) m
    hm' : ∀ (a : α), LE.le (f a) m
    a₀ : α
    h : ∀ (a : α), Exists fun a' => LT.lt (f a) (f a')
    b : β
    a✝ : LE.le (f a₀) b
    a : α
    hba : LT.lt b (f a)
    ⊢ Exists fun a => LT.lt (Order.succ b) (f a)
  -/
  exact (h a).imp (fun a' ↦ (succ_le_of_lt hba).trans_lt)
  /-
    🎉 no goals
  -/


@[deprecated StrictMono.not_bddAbove_range_of_isSuccArchimedean (since := "2024-09-21")]
alias StrictMono.not_bddAbove_range := StrictMono.not_bddAbove_range_of_isSuccArchimedean


lemma StrictMono.not_bddBelow_range_of_isPredArchimedean [NoMinOrder α] [PredOrder β]
    [IsPredArchimedean β] (hf : StrictMono f) : ¬ BddBelow (Set.range f) :=
  hf.dual.not_bddAbove_range_of_isSuccArchimedean


@[deprecated StrictMono.not_bddBelow_range_of_isPredArchimedean (since := "2024-09-21")]
alias StrictMono.not_bddBelow_range := StrictMono.not_bddBelow_range_of_isPredArchimedean


lemma StrictAnti.not_bddBelow_range_of_isSuccArchimedean [NoMinOrder α] [SuccOrder β]
    [IsSuccArchimedean β] (hf : StrictAnti f) : ¬ BddAbove (Set.range f) :=
  hf.dual_right.not_bddBelow_range_of_isPredArchimedean


@[deprecated StrictAnti.not_bddBelow_range_of_isSuccArchimedean (since := "2024-09-21")]
alias StrictAnti.not_bddAbove_range := StrictAnti.not_bddBelow_range_of_isSuccArchimedean


lemma StrictAnti.not_bddBelow_range_of_isPredArchimedean [NoMaxOrder α] [PredOrder β]
    [IsPredArchimedean β] (hf : StrictAnti f) : ¬ BddBelow (Set.range f) :=
  hf.dual_right.not_bddAbove_range_of_isSuccArchimedean


@[deprecated StrictAnti.not_bddBelow_range_of_isPredArchimedean (since := "2024-09-21")]
alias StrictAnti.not_bddBelow_range := StrictAnti.not_bddBelow_range_of_isPredArchimedean


instance (priority := 100) WellFoundedLT.toIsPredArchimedean [h : WellFoundedLT α]
    [PredOrder α] : IsPredArchimedean α :=
  ⟨fun {a b} => by
    refine WellFounded.fix (C := fun b => a ≤ b → ∃ n, Nat.iterate pred n b = a)
      h.wf ?_ b
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      h : WellFoundedLT α
      inst✝ : PredOrder α
      a b : α
      ⊢ ∀ (x : α), (∀ (y : α), LT.lt y x → (fun b => LE.le a b → Exists fun n => Eq  …
    -/
    intros b ih hab
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      h : WellFoundedLT α
      inst✝ : PredOrder α
      a b✝ b : α
      ih : ∀ (y : α), LT.lt y b → (fun b => LE.le a b → Exists fun n => Eq (Nat.iter …
      hab : LE.le a b
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
    -/
    replace hab := eq_or_lt_of_le hab
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      h : WellFoundedLT α
      inst✝ : PredOrder α
      a b✝ b : α
      ih : ∀ (y : α), LT.lt y b → (fun b => LE.le a b → Exists fun n => Eq (Nat.iter …
      hab : Or (Eq a b) (LT.lt a b)
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
    -/
    rcases hab with (rfl | hab)
      /-
        case inl
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        h : WellFoundedLT α
        inst✝ : PredOrder α
        a b : α
        ih : ∀ (y : α), LT.lt y a → (fun b => LE.le a b → Exists fun n => Eq (Nat.iter …
        ⊢ Exists fun n => Eq (Nat.iterate Order.pred n a) a
      -/
    · exact ⟨0, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      h : WellFoundedLT α
      inst✝ : PredOrder α
      a b✝ b : α
      ih : ∀ (y : α), LT.lt y b → (fun b => LE.le a b → Exists fun n => Eq (Nat.iter …
      hab : LT.lt a b
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
    -/
    rcases eq_or_lt_of_le (pred_le b) with hb | hb
      /-
        case inr.inl
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        h : WellFoundedLT α
        inst✝ : PredOrder α
        a b✝ b : α
        ih : ∀ (y : α), LT.lt y b → (fun b => LE.le a b → Exists fun n => Eq (Nat.iter …
        hab : LT.lt a b
        hb : Eq (Order.pred b) b
        ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
      -/
    · cases (min_of_le_pred hb.ge).not_lt hab
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      h : WellFoundedLT α
      inst✝ : PredOrder α
      a b✝ b : α
      ih : ∀ (y : α), LT.lt y b → (fun b => LE.le a b → Exists fun n => Eq (Nat.iter …
      hab : LT.lt a b
      hb : LT.lt (Order.pred b) b
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
    -/
    dsimp at ih
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      h : WellFoundedLT α
      inst✝ : PredOrder α
      a b✝ b : α
      ih : ∀ (y : α), LT.lt y b → LE.le a y → Exists fun n => Eq (Nat.iterate Order. …
      hab : LT.lt a b
      hb : LT.lt (Order.pred b) b
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
    -/
    obtain ⟨k, hk⟩ := ih (pred b) hb (le_pred_of_lt hab)
    /-
      case inr.inr.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      h : WellFoundedLT α
      inst✝ : PredOrder α
      a b✝ b : α
      ih : ∀ (y : α), LT.lt y b → LE.le a y → Exists fun n => Eq (Nat.iterate Order. …
      hab : LT.lt a b
      hb : LT.lt (Order.pred b) b
      k : Nat
      hk : Eq (Nat.iterate Order.pred k (Order.pred b)) a
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
    -/
    refine ⟨k + 1, ?_⟩
    /-
      case inr.inr.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      h : WellFoundedLT α
      inst✝ : PredOrder α
      a b✝ b : α
      ih : ∀ (y : α), LT.lt y b → LE.le a y → Exists fun n => Eq (Nat.iterate Order. …
      hab : LT.lt a b
      hb : LT.lt (Order.pred b) b
      k : Nat
      hk : Eq (Nat.iterate Order.pred k (Order.pred b)) a
      ⊢ Eq (Nat.iterate Order.pred (HAdd.hAdd k 1) b) a
    -/
    rw [iterate_add_apply, iterate_one, hk]⟩
    /-
      🎉 no goals
    -/


instance (priority := 100) WellFoundedGT.toIsSuccArchimedean [h : WellFoundedGT α]
    [SuccOrder α] : IsSuccArchimedean α :=
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝¹ : PartialOrder α
                                        h : WellFoundedGT α
                                        inst✝ : SuccOrder α
                                        ⊢ IsPredArchimedean (OrderDual α)
                                      -/
  let h : IsPredArchimedean αᵒᵈ := by infer_instance
                                      /-
                                        🎉 no goals
                                      -/
  ⟨h.1⟩


theorem Succ.rec_bot (p : α → Prop) (hbot : p ⊥) (hsucc : ∀ a, p a → p (succ a)) (a : α) : p a :=
  Succ.rec hbot (fun x _ h => hsucc x h) (bot_le : ⊥ ≤ a)


theorem Pred.rec_top (p : α → Prop) (htop : p ⊤) (hpred : ∀ a, p a → p (pred a)) (a : α) : p a :=
  Pred.rec htop (fun x _ h => hpred x h) (le_top : a ≤ ⊤)


lemma SuccOrder.forall_ne_bot_iff
    [Nontrivial α] [PartialOrder α] [OrderBot α] [SuccOrder α] [IsSuccArchimedean α]
    (P : α → Prop) :
    (∀ i, i ≠ ⊥ → P i) ↔ (∀ i, P (SuccOrder.succ i)) := by
  /-
    α : Type u_1
    inst✝⁴ : Nontrivial α
    inst✝³ : PartialOrder α
    inst✝² : OrderBot α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    P : α → Prop
    ⊢ Iff (∀ (i : α), Ne i Bot.bot → P i) (∀ (i : α), P (SuccOrder.succ i))
  -/
  refine ⟨fun h i ↦ h _ (Order.succ_ne_bot i), fun h i hi ↦ ?_⟩
  /-
    α : Type u_1
    inst✝⁴ : Nontrivial α
    inst✝³ : PartialOrder α
    inst✝² : OrderBot α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    P : α → Prop
    h : ∀ (i : α), P (SuccOrder.succ i)
    i : α
    hi : Ne i Bot.bot
    ⊢ P i
  -/
  obtain ⟨j, rfl⟩ := exists_succ_iterate_of_le (bot_le : ⊥ ≤ i)
  /-
    case intro
    α : Type u_1
    inst✝⁴ : Nontrivial α
    inst✝³ : PartialOrder α
    inst✝² : OrderBot α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    P : α → Prop
    h : ∀ (i : α), P (SuccOrder.succ i)
    j : Nat
    hi : Ne (Nat.iterate Order.succ j Bot.bot) Bot.bot
    ⊢ P (Nat.iterate Order.succ j Bot.bot)
  -/
  have hj : 0 < j := by apply Nat.pos_of_ne_zero; contrapose! hi; simp [hi]
  /-
    case intro
    α : Type u_1
    inst✝⁴ : Nontrivial α
    inst✝³ : PartialOrder α
    inst✝² : OrderBot α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    P : α → Prop
    h : ∀ (i : α), P (SuccOrder.succ i)
    j : Nat
    hi : Ne (Nat.iterate Order.succ j Bot.bot) Bot.bot
    hj : LT.lt 0 j
    ⊢ P (Nat.iterate Order.succ j Bot.bot)
  -/
  rw [← Nat.succ_pred_eq_of_pos hj]
  /-
    case intro
    α : Type u_1
    inst✝⁴ : Nontrivial α
    inst✝³ : PartialOrder α
    inst✝² : OrderBot α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    P : α → Prop
    h : ∀ (i : α), P (SuccOrder.succ i)
    j : Nat
    hi : Ne (Nat.iterate Order.succ j Bot.bot) Bot.bot
    hj : LT.lt 0 j
    ⊢ P (Nat.iterate Order.succ j.pred.succ Bot.bot)
  -/
  simp only [Function.iterate_succ', Function.comp_apply]
  /-
    case intro
    α : Type u_1
    inst✝⁴ : Nontrivial α
    inst✝³ : PartialOrder α
    inst✝² : OrderBot α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    P : α → Prop
    h : ∀ (i : α), P (SuccOrder.succ i)
    j : Nat
    hi : Ne (Nat.iterate Order.succ j Bot.bot) Bot.bot
    hj : LT.lt 0 j
    ⊢ P (Order.succ (Nat.iterate Order.succ j.pred Bot.bot))
  -/
  apply h
  /-
    🎉 no goals
  -/


lemma BddAbove.exists_isGreatest_of_nonempty {X : Type*} [LinearOrder X] [SuccOrder X]
    [IsSuccArchimedean X] {S : Set X} (hS : BddAbove S) (hS' : S.Nonempty) :
    ∃ x, IsGreatest S x := by
  /-
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    hS : BddAbove S
    hS' : S.Nonempty
    ⊢ Exists fun x => IsGreatest S x
  -/
  obtain ⟨m, hm⟩ := hS
  /-
    case intro
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    hS' : S.Nonempty
    m : X
    hm : Membership.mem (upperBounds S) m
    ⊢ Exists fun x => IsGreatest S x
  -/
  obtain ⟨n, hn⟩ := hS'
  /-
    case intro.intro
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    m : X
    hm : Membership.mem (upperBounds S) m
    n : X
    hn : Membership.mem S n
    ⊢ Exists fun x => IsGreatest S x
  -/
  by_cases hm' : m ∈ S
    /-
      case pos
      X : Type u_3
      inst✝² : LinearOrder X
      inst✝¹ : SuccOrder X
      inst✝ : IsSuccArchimedean X
      S : Set X
      m : X
      hm : Membership.mem (upperBounds S) m
      n : X
      hn : Membership.mem S n
      hm' : Membership.mem S m
      ⊢ Exists fun x => IsGreatest S x
    -/
  · exact ⟨_, hm', hm⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    m : X
    hm : Membership.mem (upperBounds S) m
    n : X
    hn : Membership.mem S n
    hm' : Not (Membership.mem S m)
    ⊢ Exists fun x => IsGreatest S x
  -/
  have hn' := hm hn
  /-
    case neg
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    m : X
    hm : Membership.mem (upperBounds S) m
    n : X
    hn : Membership.mem S n
    hm' : Not (Membership.mem S m)
    hn' : LE.le n m
    ⊢ Exists fun x => IsGreatest S x
  -/
  revert hn hm hm'
  /-
    case neg
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    m n : X
    hn' : LE.le n m
    ⊢ Membership.mem (upperBounds S) m → Membership.mem S n → Not (Membership.mem  …
  -/
  refine Succ.rec ?_ ?_ hn'
    /-
      case neg.refine_1
      X : Type u_3
      inst✝² : LinearOrder X
      inst✝¹ : SuccOrder X
      inst✝ : IsSuccArchimedean X
      S : Set X
      m n : X
      hn' : LE.le n m
      ⊢ Membership.mem (upperBounds S) n → Membership.mem S n → Not (Membership.mem  …
    -/
  · simp (config := {contextual := true})
    /-
      🎉 no goals
    -/
  /-
    case neg.refine_2
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    m n : X
    hn' : LE.le n m
    ⊢ ∀ (n_1 : X), LE.le n n_1 → (Membership.mem (upperBounds S) n_1 → Membership. …
  -/
  intro m _ IH hm hn hm'
  /-
    case neg.refine_2
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    m✝ n : X
    hn' : LE.le n m✝
    m : X
    a✝ : LE.le n m
    IH : Membership.mem (upperBounds S) m → Membership.mem S n → Not (Membership.m …
    hm : Membership.mem (upperBounds S) (Order.succ m)
    hn : Membership.mem S n
    hm' : Not (Membership.mem S (Order.succ m))
    ⊢ Exists fun x => IsGreatest S x
  -/
  rw [mem_upperBounds] at IH hm
  /-
    case neg.refine_2
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    m✝ n : X
    hn' : LE.le n m✝
    m : X
    a✝ : LE.le n m
    IH : (∀ (x : X), Membership.mem S x → LE.le x m) → Membership.mem S n → Not (M …
    hm : ∀ (x : X), Membership.mem S x → LE.le x (Order.succ m)
    hn : Membership.mem S n
    hm' : Not (Membership.mem S (Order.succ m))
    ⊢ Exists fun x => IsGreatest S x
  -/
  simp_rw [Order.le_succ_iff_eq_or_le] at hm
  replace hm : ∀ x ∈ S, x ≤ m := by
    intro x hx
    refine (hm x hx).resolve_left ?_
    rintro rfl
    exact hm' hx
  /-
    case neg.refine_2
    X : Type u_3
    inst✝² : LinearOrder X
    inst✝¹ : SuccOrder X
    inst✝ : IsSuccArchimedean X
    S : Set X
    m✝ n : X
    hn' : LE.le n m✝
    m : X
    a✝ : LE.le n m
    IH : (∀ (x : X), Membership.mem S x → LE.le x m) → Membership.mem S n → Not (M …
    hn : Membership.mem S n
    hm' : Not (Membership.mem S (Order.succ m))
    hm : ∀ (x : X), Membership.mem S x → LE.le x m
    ⊢ Exists fun x => IsGreatest S x
  -/
  by_cases hmS : m ∈ S
    /-
      case pos
      X : Type u_3
      inst✝² : LinearOrder X
      inst✝¹ : SuccOrder X
      inst✝ : IsSuccArchimedean X
      S : Set X
      m✝ n : X
      hn' : LE.le n m✝
      m : X
      a✝ : LE.le n m
      IH : (∀ (x : X), Membership.mem S x → LE.le x m) → Membership.mem S n → Not (M …
      hn : Membership.mem S n
      hm' : Not (Membership.mem S (Order.succ m))
      hm : ∀ (x : X), Membership.mem S x → LE.le x m
      hmS : Membership.mem S m
      ⊢ Exists fun x => IsGreatest S x
    -/
  · exact ⟨m, hmS, hm⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_3
      inst✝² : LinearOrder X
      inst✝¹ : SuccOrder X
      inst✝ : IsSuccArchimedean X
      S : Set X
      m✝ n : X
      hn' : LE.le n m✝
      m : X
      a✝ : LE.le n m
      IH : (∀ (x : X), Membership.mem S x → LE.le x m) → Membership.mem S n → Not (M …
      hn : Membership.mem S n
      hm' : Not (Membership.mem S (Order.succ m))
      hm : ∀ (x : X), Membership.mem S x → LE.le x m
      hmS : Not (Membership.mem S m)
      ⊢ Exists fun x => IsGreatest S x
    -/
  · exact IH hm hn hmS
    /-
      🎉 no goals
    -/


lemma BddBelow.exists_isLeast_of_nonempty {X : Type*} [LinearOrder X] [PredOrder X]
    [IsPredArchimedean X] {S : Set X} (hS : BddBelow S) (hS' : S.Nonempty) :
    ∃ x, IsLeast S x :=
  hS.dual.exists_isGreatest_of_nonempty hS'


/-- `IsSuccArchimedean` transfers across equivalences between `SuccOrder`s. -/
protected lemma IsSuccArchimedean.of_orderIso [SuccOrder X] [IsSuccArchimedean X] [SuccOrder Y]
    (f : X ≃o Y) : IsSuccArchimedean Y where
  exists_succ_iterate_of_le {a b} h := by
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : SuccOrder X
      inst✝¹ : IsSuccArchimedean X
      inst✝ : SuccOrder Y
      f : OrderIso X Y
      a b : Y
      h : LE.le a b
      ⊢ Exists fun n => Eq (Nat.iterate Order.succ n a) b
    -/
    refine (exists_succ_iterate_of_le ((map_inv_le_map_inv_iff f).mpr h)).imp ?_
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : SuccOrder X
      inst✝¹ : IsSuccArchimedean X
      inst✝ : SuccOrder Y
      f : OrderIso X Y
      a b : Y
      h : LE.le a b
      ⊢ ∀ (a_1 : Nat), Eq (Nat.iterate Order.succ a_1 (EquivLike.inv f a)) (EquivLik …
    -/
    intro n
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : SuccOrder X
      inst✝¹ : IsSuccArchimedean X
      inst✝ : SuccOrder Y
      f : OrderIso X Y
      a b : Y
      h : LE.le a b
      n : Nat
      ⊢ Eq (Nat.iterate Order.succ n (EquivLike.inv f a)) (EquivLike.inv f b) → Eq ( …
    -/
    rw [← f.apply_eq_iff_eq, EquivLike.apply_inv_apply]
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : SuccOrder X
      inst✝¹ : IsSuccArchimedean X
      inst✝ : SuccOrder Y
      f : OrderIso X Y
      a b : Y
      h : LE.le a b
      n : Nat
      ⊢ Eq (f (Nat.iterate Order.succ n (EquivLike.inv f a))) b → Eq (Nat.iterate Or …
    -/
    rintro rfl
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : SuccOrder X
      inst✝¹ : IsSuccArchimedean X
      inst✝ : SuccOrder Y
      f : OrderIso X Y
      a : Y
      n : Nat
      h : LE.le a (f (Nat.iterate Order.succ n (EquivLike.inv f a)))
      ⊢ Eq (Nat.iterate Order.succ n a) (f (Nat.iterate Order.succ n (EquivLike.inv  …
    -/
    clear h
    induction n generalizing a with
    | zero => simp
    | succ n IH => simp only [Function.iterate_succ', Function.comp_apply, IH, f.map_succ]


/-- `IsPredArchimedean` transfers across equivalences between `PredOrder`s. -/
protected lemma IsPredArchimedean.of_orderIso [PredOrder X] [IsPredArchimedean X] [PredOrder Y]
    (f : X ≃o Y) : IsPredArchimedean Y where
  exists_pred_iterate_of_le {a b} h := by
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : PredOrder X
      inst✝¹ : IsPredArchimedean X
      inst✝ : PredOrder Y
      f : OrderIso X Y
      a b : Y
      h : LE.le a b
      ⊢ Exists fun n => Eq (Nat.iterate Order.pred n b) a
    -/
    refine (exists_pred_iterate_of_le ((map_inv_le_map_inv_iff f).mpr h)).imp ?_
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : PredOrder X
      inst✝¹ : IsPredArchimedean X
      inst✝ : PredOrder Y
      f : OrderIso X Y
      a b : Y
      h : LE.le a b
      ⊢ ∀ (a_1 : Nat), Eq (Nat.iterate Order.pred a_1 (EquivLike.inv f b)) (EquivLik …
    -/
    intro n
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : PredOrder X
      inst✝¹ : IsPredArchimedean X
      inst✝ : PredOrder Y
      f : OrderIso X Y
      a b : Y
      h : LE.le a b
      n : Nat
      ⊢ Eq (Nat.iterate Order.pred n (EquivLike.inv f b)) (EquivLike.inv f a) → Eq ( …
    -/
    rw [← f.apply_eq_iff_eq, EquivLike.apply_inv_apply]
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : PredOrder X
      inst✝¹ : IsPredArchimedean X
      inst✝ : PredOrder Y
      f : OrderIso X Y
      a b : Y
      h : LE.le a b
      n : Nat
      ⊢ Eq (f (Nat.iterate Order.pred n (EquivLike.inv f b))) a → Eq (Nat.iterate Or …
    -/
    rintro rfl
    /-
      X : Type u_3
      Y : Type u_4
      inst✝⁴ : PartialOrder X
      inst✝³ : PartialOrder Y
      inst✝² : PredOrder X
      inst✝¹ : IsPredArchimedean X
      inst✝ : PredOrder Y
      f : OrderIso X Y
      b : Y
      n : Nat
      h : LE.le (f (Nat.iterate Order.pred n (EquivLike.inv f b))) b
      ⊢ Eq (Nat.iterate Order.pred n b) (f (Nat.iterate Order.pred n (EquivLike.inv  …
    -/
    clear h
    induction n generalizing b with
    | zero => simp
    | succ n IH => simp only [Function.iterate_succ', Function.comp_apply, IH, f.map_pred]


instance Set.OrdConnected.isPredArchimedean [PredOrder α] [IsPredArchimedean α]
    (s : Set α) [s.OrdConnected] : IsPredArchimedean s where
  exists_pred_iterate_of_le := @fun ⟨b, hb⟩ ⟨c, hc⟩ hbc ↦ by classical
    simp only [Subtype.mk_le_mk] at hbc
    obtain ⟨n, hn⟩ := hbc.exists_pred_iterate
    use n
    induction n generalizing c with
    | zero => simp_all
    | succ n hi =>
      simp_all only [Function.iterate_succ, Function.comp_apply]
      change Order.pred^[n] (dite ..) = _
      split_ifs with h
      · dsimp only at h ⊢
        apply hi _ _ _ hn
        · rw [← hn]
          apply Order.pred_iterate_le
      · have : Order.pred (⟨c, hc⟩ : s) = ⟨c, hc⟩ := by
          change dite .. = _
          simp [h]
        rw [Function.iterate_fixed]
        · simp only [Order.pred_eq_iff_isMin] at this
          apply (this.eq_of_le _).symm
          exact hbc
        · exact this


instance Set.OrdConnected.isSuccArchimedean [SuccOrder α] [IsSuccArchimedean α]
    (s : Set α) [s.OrdConnected] : IsSuccArchimedean s :=
  letI : IsPredArchimedean sᵒᵈ := inferInstanceAs (IsPredArchimedean (OrderDual.ofDual ⁻¹' s))
  inferInstanceAs (IsSuccArchimedean sᵒᵈᵒᵈ)


lemma monotoneOn_of_le_succ (hs : s.OrdConnected)
    (hf : ∀ a, ¬ IsMax a → a ∈ s → succ a ∈ s → f a ≤ f (succ a)) : MonotoneOn f s := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    ⊢ MonotoneOn f s
  -/
  rintro a ha b hb hab
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LE.le a b
    ⊢ LE.le (f a) (f b)
  -/
  obtain ⟨n, rfl⟩ := exists_succ_iterate_of_le hab
  /-
    case intro
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    n : Nat
    hb : Membership.mem s (Nat.iterate Order.succ n a)
    hab : LE.le a (Nat.iterate Order.succ n a)
    ⊢ LE.le (f a) (f (Nat.iterate Order.succ n a))
  -/
  clear hab
  /-
    case intro
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    n : Nat
    hb : Membership.mem s (Nat.iterate Order.succ n a)
    ⊢ LE.le (f a) (f (Nat.iterate Order.succ n a))
  -/
  induction' n with n hn
    /-
      case intro.zero
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      hb : Membership.mem s (Nat.iterate Order.succ 0 a)
      ⊢ LE.le (f a) (f (Nat.iterate Order.succ 0 a))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.succ n a) → LE.le (f a) (f (Nat.itera …
    hb : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
    ⊢ LE.le (f a) (f (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
  -/
  rw [Function.iterate_succ_apply'] at hb ⊢
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.succ n a) → LE.le (f a) (f (Nat.itera …
    hb : Membership.mem s (Order.succ (Nat.iterate Order.succ n a))
    ⊢ LE.le (f a) (f (Order.succ (Nat.iterate Order.succ n a)))
  -/
  have : succ^[n] a ∈ s := hs.1 ha hb ⟨le_succ_iterate .., le_succ _⟩
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.succ n a) → LE.le (f a) (f (Nat.itera …
    hb : Membership.mem s (Order.succ (Nat.iterate Order.succ n a))
    this : Membership.mem s (Nat.iterate Order.succ n a)
    ⊢ LE.le (f a) (f (Order.succ (Nat.iterate Order.succ n a)))
  -/
  by_cases hb' : IsMax (succ^[n] a)
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.succ n a) → LE.le (f a) (f (Nat.itera …
      hb : Membership.mem s (Order.succ (Nat.iterate Order.succ n a))
      this : Membership.mem s (Nat.iterate Order.succ n a)
      hb' : IsMax (Nat.iterate Order.succ n a)
      ⊢ LE.le (f a) (f (Order.succ (Nat.iterate Order.succ n a)))
    -/
  · rw [succ_eq_iff_isMax.2 hb']
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.succ n a) → LE.le (f a) (f (Nat.itera …
      hb : Membership.mem s (Order.succ (Nat.iterate Order.succ n a))
      this : Membership.mem s (Nat.iterate Order.succ n a)
      hb' : IsMax (Nat.iterate Order.succ n a)
      ⊢ LE.le (f a) (f (Nat.iterate Order.succ n a))
    -/
    exact hn this
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.succ n a) → LE.le (f a) (f (Nat.itera …
      hb : Membership.mem s (Order.succ (Nat.iterate Order.succ n a))
      this : Membership.mem s (Nat.iterate Order.succ n a)
      hb' : Not (IsMax (Nat.iterate Order.succ n a))
      ⊢ LE.le (f a) (f (Order.succ (Nat.iterate Order.succ n a)))
    -/
  · exact (hn this).trans (hf _ hb' this hb)
    /-
      🎉 no goals
    -/


lemma antitoneOn_of_succ_le (hs : s.OrdConnected)
    (hf : ∀ a, ¬ IsMax a → a ∈ s → succ a ∈ s → f (succ a) ≤ f a) : AntitoneOn f s :=
  monotoneOn_of_le_succ (β := βᵒᵈ) hs hf


lemma strictMonoOn_of_lt_succ (hs : s.OrdConnected)
    (hf : ∀ a, ¬ IsMax a → a ∈ s → succ a ∈ s → f a < f (succ a)) : StrictMonoOn f s := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    ⊢ StrictMonoOn f s
  -/
  rintro a ha b hb hab
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LT.lt a b
    ⊢ LT.lt (f a) (f b)
  -/
  obtain ⟨n, rfl⟩ := exists_succ_iterate_of_le hab.le
  /-
    case intro
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    n : Nat
    hb : Membership.mem s (Nat.iterate Order.succ n a)
    hab : LT.lt a (Nat.iterate Order.succ n a)
    ⊢ LT.lt (f a) (f (Nat.iterate Order.succ n a))
  -/
  obtain _ | n := n
    /-
      case intro.zero
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      hb : Membership.mem s (Nat.iterate Order.succ 0 a)
      hab : LT.lt a (Nat.iterate Order.succ 0 a)
      ⊢ LT.lt (f a) (f (Nat.iterate Order.succ 0 a))
    -/
  · simp at hab
    /-
      🎉 no goals
    -/
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    n : Nat
    hb : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
    hab : LT.lt a (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
    ⊢ LT.lt (f a) (f (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
  -/
  apply not_isMax_of_lt at hab
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    n : Nat
    hb : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
    hab : Not (IsMax a)
    ⊢ LT.lt (f a) (f (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
  -/
  induction' n with n hn
    /-
      case intro.succ.zero
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      hab : Not (IsMax a)
      hb : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd 0 1) a)
      ⊢ LT.lt (f a) (f (Nat.iterate Order.succ (HAdd.hAdd 0 1) a))
    -/
  · simpa using hf _ hab ha hb
    /-
      🎉 no goals
    -/
  /-
    case intro.succ.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    hab : Not (IsMax a)
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a) → LT.lt (f a) …
    hb : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd (HAdd.hAdd n 1) 1) a)
    ⊢ LT.lt (f a) (f (Nat.iterate Order.succ (HAdd.hAdd (HAdd.hAdd n 1) 1) a))
  -/
  rw [Function.iterate_succ_apply'] at hb ⊢
  /-
    case intro.succ.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    hab : Not (IsMax a)
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a) → LT.lt (f a) …
    hb : Membership.mem s (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
    ⊢ LT.lt (f a) (f (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a)))
  -/
  have : succ^[n + 1] a ∈ s := hs.1 ha hb ⟨le_succ_iterate .., le_succ _⟩
  /-
    case intro.succ.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
    a : α
    ha : Membership.mem s a
    hab : Not (IsMax a)
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a) → LT.lt (f a) …
    hb : Membership.mem s (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
    this : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
    ⊢ LT.lt (f a) (f (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a)))
  -/
  by_cases hb' : IsMax (succ^[n + 1] a)
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      hab : Not (IsMax a)
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a) → LT.lt (f a) …
      hb : Membership.mem s (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
      this : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
      hb' : IsMax (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
      ⊢ LT.lt (f a) (f (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a)))
    -/
  · rw [succ_eq_iff_isMax.2 hb']
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      hab : Not (IsMax a)
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a) → LT.lt (f a) …
      hb : Membership.mem s (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
      this : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
      hb' : IsMax (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
      ⊢ LT.lt (f a) (f (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
    -/
    exact hn this
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMax a) → Membership.mem s a → Membership.mem s (Order.s …
      a : α
      ha : Membership.mem s a
      hab : Not (IsMax a)
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a) → LT.lt (f a) …
      hb : Membership.mem s (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
      this : Membership.mem s (Nat.iterate Order.succ (HAdd.hAdd n 1) a)
      hb' : Not (IsMax (Nat.iterate Order.succ (HAdd.hAdd n 1) a))
      ⊢ LT.lt (f a) (f (Order.succ (Nat.iterate Order.succ (HAdd.hAdd n 1) a)))
    -/
  · exact (hn this).trans (hf _ hb' this hb)
    /-
      🎉 no goals
    -/


lemma strictAntiOn_of_succ_lt (hs : s.OrdConnected)
    (hf : ∀ a, ¬ IsMax a → a ∈ s → succ a ∈ s → f (succ a) < f a) : StrictAntiOn f s :=
  strictMonoOn_of_lt_succ (β := βᵒᵈ) hs hf


lemma monotone_of_le_succ (hf : ∀ a, ¬ IsMax a → f a ≤ f (succ a)) : Monotone f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    f : α → β
    hf : ∀ (a : α), Not (IsMax a) → LE.le (f a) (f (Order.succ a))
    ⊢ Monotone f
  -/
  simpa using monotoneOn_of_le_succ Set.ordConnected_univ (by simpa using hf)
  /-
    🎉 no goals
  -/


lemma antitone_of_succ_le (hf : ∀ a, ¬ IsMax a → f (succ a) ≤ f a) : Antitone f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    f : α → β
    hf : ∀ (a : α), Not (IsMax a) → LE.le (f (Order.succ a)) (f a)
    ⊢ Antitone f
  -/
  simpa using antitoneOn_of_succ_le Set.ordConnected_univ (by simpa using hf)
  /-
    🎉 no goals
  -/


lemma strictMono_of_lt_succ (hf : ∀ a, ¬ IsMax a → f a < f (succ a)) : StrictMono f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    f : α → β
    hf : ∀ (a : α), Not (IsMax a) → LT.lt (f a) (f (Order.succ a))
    ⊢ StrictMono f
  -/
  simpa using strictMonoOn_of_lt_succ Set.ordConnected_univ (by simpa using hf)
  /-
    🎉 no goals
  -/


lemma strictAnti_of_succ_lt (hf : ∀ a, ¬ IsMax a → f (succ a) < f a) : StrictAnti f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    f : α → β
    hf : ∀ (a : α), Not (IsMax a) → LT.lt (f (Order.succ a)) (f a)
    ⊢ StrictAnti f
  -/
  simpa using strictAntiOn_of_succ_lt Set.ordConnected_univ (by simpa using hf)
  /-
    🎉 no goals
  -/


lemma monotoneOn_of_pred_le (hs : s.OrdConnected)
    (hf : ∀ a, ¬ IsMin a → a ∈ s → pred a ∈ s → f (pred a) ≤ f a) : MonotoneOn f s := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    ⊢ MonotoneOn f s
  -/
  rintro a ha b hb hab
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LE.le a b
    ⊢ LE.le (f a) (f b)
  -/
  obtain ⟨n, rfl⟩ := exists_pred_iterate_of_le hab
  /-
    case intro
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    n : Nat
    ha : Membership.mem s (Nat.iterate Order.pred n b)
    hab : LE.le (Nat.iterate Order.pred n b) b
    ⊢ LE.le (f (Nat.iterate Order.pred n b)) (f b)
  -/
  clear hab
  /-
    case intro
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    n : Nat
    ha : Membership.mem s (Nat.iterate Order.pred n b)
    ⊢ LE.le (f (Nat.iterate Order.pred n b)) (f b)
  -/
  induction' n with n hn
    /-
      case intro.zero
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      ha : Membership.mem s (Nat.iterate Order.pred 0 b)
      ⊢ LE.le (f (Nat.iterate Order.pred 0 b)) (f b)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.pred n b) → LE.le (f (Nat.iterate Ord …
    ha : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
    ⊢ LE.le (f (Nat.iterate Order.pred (HAdd.hAdd n 1) b)) (f b)
  -/
  rw [Function.iterate_succ_apply'] at ha ⊢
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.pred n b) → LE.le (f (Nat.iterate Ord …
    ha : Membership.mem s (Order.pred (Nat.iterate Order.pred n b))
    ⊢ LE.le (f (Order.pred (Nat.iterate Order.pred n b))) (f b)
  -/
  have : pred^[n] b ∈ s := hs.1 ha hb ⟨pred_le _, pred_iterate_le ..⟩
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.pred n b) → LE.le (f (Nat.iterate Ord …
    ha : Membership.mem s (Order.pred (Nat.iterate Order.pred n b))
    this : Membership.mem s (Nat.iterate Order.pred n b)
    ⊢ LE.le (f (Order.pred (Nat.iterate Order.pred n b))) (f b)
  -/
  by_cases ha' : IsMin (pred^[n] b)
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.pred n b) → LE.le (f (Nat.iterate Ord …
      ha : Membership.mem s (Order.pred (Nat.iterate Order.pred n b))
      this : Membership.mem s (Nat.iterate Order.pred n b)
      ha' : IsMin (Nat.iterate Order.pred n b)
      ⊢ LE.le (f (Order.pred (Nat.iterate Order.pred n b))) (f b)
    -/
  · rw [pred_eq_iff_isMin.2 ha']
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.pred n b) → LE.le (f (Nat.iterate Ord …
      ha : Membership.mem s (Order.pred (Nat.iterate Order.pred n b))
      this : Membership.mem s (Nat.iterate Order.pred n b)
      ha' : IsMin (Nat.iterate Order.pred n b)
      ⊢ LE.le (f (Nat.iterate Order.pred n b)) (f b)
    -/
    exact hn this
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.pred n b) → LE.le (f (Nat.iterate Ord …
      ha : Membership.mem s (Order.pred (Nat.iterate Order.pred n b))
      this : Membership.mem s (Nat.iterate Order.pred n b)
      ha' : Not (IsMin (Nat.iterate Order.pred n b))
      ⊢ LE.le (f (Order.pred (Nat.iterate Order.pred n b))) (f b)
    -/
  · exact (hn this).trans' (hf _ ha' this ha)
    /-
      🎉 no goals
    -/


lemma antitoneOn_of_le_pred (hs : s.OrdConnected)
    (hf : ∀ a, ¬ IsMin a → a ∈ s → pred a ∈ s → f a ≤ f (pred a)) : AntitoneOn f s :=
  monotoneOn_of_pred_le (β := βᵒᵈ) hs hf


lemma strictMonoOn_of_pred_lt (hs : s.OrdConnected)
    (hf : ∀ a, ¬ IsMin a → a ∈ s → pred a ∈ s → f (pred a) < f a) : StrictMonoOn f s := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    ⊢ StrictMonoOn f s
  -/
  rintro a ha b hb hab
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LT.lt a b
    ⊢ LT.lt (f a) (f b)
  -/
  obtain ⟨n, rfl⟩ := exists_pred_iterate_of_le hab.le
  /-
    case intro
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    n : Nat
    ha : Membership.mem s (Nat.iterate Order.pred n b)
    hab : LT.lt (Nat.iterate Order.pred n b) b
    ⊢ LT.lt (f (Nat.iterate Order.pred n b)) (f b)
  -/
  obtain _ | n := n
    /-
      case intro.zero
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      ha : Membership.mem s (Nat.iterate Order.pred 0 b)
      hab : LT.lt (Nat.iterate Order.pred 0 b) b
      ⊢ LT.lt (f (Nat.iterate Order.pred 0 b)) (f b)
    -/
  · simp at hab
    /-
      🎉 no goals
    -/
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    n : Nat
    ha : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
    hab : LT.lt (Nat.iterate Order.pred (HAdd.hAdd n 1) b) b
    ⊢ LT.lt (f (Nat.iterate Order.pred (HAdd.hAdd n 1) b)) (f b)
  -/
  apply not_isMin_of_lt at hab
  /-
    case intro.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    n : Nat
    ha : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
    hab : Not (IsMin b)
    ⊢ LT.lt (f (Nat.iterate Order.pred (HAdd.hAdd n 1) b)) (f b)
  -/
  induction' n with n hn
    /-
      case intro.succ.zero
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      hab : Not (IsMin b)
      ha : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd 0 1) b)
      ⊢ LT.lt (f (Nat.iterate Order.pred (HAdd.hAdd 0 1) b)) (f b)
    -/
  · simpa using hf _ hab hb ha
    /-
      🎉 no goals
    -/
  /-
    case intro.succ.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    hab : Not (IsMin b)
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b) → LT.lt (f (N …
    ha : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd (HAdd.hAdd n 1) 1) b)
    ⊢ LT.lt (f (Nat.iterate Order.pred (HAdd.hAdd (HAdd.hAdd n 1) 1) b)) (f b)
  -/
  rw [Function.iterate_succ_apply'] at ha ⊢
  /-
    case intro.succ.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    hab : Not (IsMin b)
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b) → LT.lt (f (N …
    ha : Membership.mem s (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))
    ⊢ LT.lt (f (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))) (f b)
  -/
  have : pred^[n + 1] b ∈ s := hs.1 ha hb ⟨pred_le _, pred_iterate_le ..⟩
  /-
    case intro.succ.succ
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    s : Set α
    f : α → β
    hs : s.OrdConnected
    hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
    b : α
    hb : Membership.mem s b
    hab : Not (IsMin b)
    n : Nat
    hn : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b) → LT.lt (f (N …
    ha : Membership.mem s (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))
    this : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
    ⊢ LT.lt (f (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))) (f b)
  -/
  by_cases ha' : IsMin (pred^[n + 1] b)
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      hab : Not (IsMin b)
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b) → LT.lt (f (N …
      ha : Membership.mem s (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))
      this : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
      ha' : IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
      ⊢ LT.lt (f (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))) (f b)
    -/
  · rw [pred_eq_iff_isMin.2 ha']
    /-
      case pos
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      hab : Not (IsMin b)
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b) → LT.lt (f (N …
      ha : Membership.mem s (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))
      this : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
      ha' : IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
      ⊢ LT.lt (f (Nat.iterate Order.pred (HAdd.hAdd n 1) b)) (f b)
    -/
    exact hn this
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_3
      β : Type u_4
      inst✝³ : PartialOrder α
      inst✝² : Preorder β
      inst✝¹ : PredOrder α
      inst✝ : IsPredArchimedean α
      s : Set α
      f : α → β
      hs : s.OrdConnected
      hf : ∀ (a : α), Not (IsMin a) → Membership.mem s a → Membership.mem s (Order.p …
      b : α
      hb : Membership.mem s b
      hab : Not (IsMin b)
      n : Nat
      hn : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b) → LT.lt (f (N …
      ha : Membership.mem s (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))
      this : Membership.mem s (Nat.iterate Order.pred (HAdd.hAdd n 1) b)
      ha' : Not (IsMin (Nat.iterate Order.pred (HAdd.hAdd n 1) b))
      ⊢ LT.lt (f (Order.pred (Nat.iterate Order.pred (HAdd.hAdd n 1) b))) (f b)
    -/
  · exact (hn this).trans' (hf _ ha' this ha)
    /-
      🎉 no goals
    -/


lemma strictAntiOn_of_lt_pred (hs : s.OrdConnected)
    (hf : ∀ a, ¬ IsMin a → a ∈ s → pred a ∈ s → f a < f (pred a)) : StrictAntiOn f s :=
  strictMonoOn_of_pred_lt (β := βᵒᵈ) hs hf


lemma monotone_of_pred_le (hf : ∀ a, ¬ IsMin a → f (pred a) ≤ f a) : Monotone f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    f : α → β
    hf : ∀ (a : α), Not (IsMin a) → LE.le (f (Order.pred a)) (f a)
    ⊢ Monotone f
  -/
  simpa using monotoneOn_of_pred_le Set.ordConnected_univ (by simpa using hf)
  /-
    🎉 no goals
  -/


lemma antitone_of_le_pred (hf : ∀ a, ¬ IsMin a → f a ≤ f (pred a)) : Antitone f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    f : α → β
    hf : ∀ (a : α), Not (IsMin a) → LE.le (f a) (f (Order.pred a))
    ⊢ Antitone f
  -/
  simpa using antitoneOn_of_le_pred Set.ordConnected_univ (by simpa using hf)
  /-
    🎉 no goals
  -/


lemma strictMono_of_pred_lt (hf : ∀ a, ¬ IsMin a → f (pred a) < f a) : StrictMono f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    f : α → β
    hf : ∀ (a : α), Not (IsMin a) → LT.lt (f (Order.pred a)) (f a)
    ⊢ StrictMono f
  -/
  simpa using strictMonoOn_of_pred_lt Set.ordConnected_univ (by simpa using hf)
  /-
    🎉 no goals
  -/


lemma strictAnti_of_lt_pred (hf : ∀ a, ¬ IsMin a → f a < f (pred a)) : StrictAnti f := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : PartialOrder α
    inst✝² : Preorder β
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    f : α → β
    hf : ∀ (a : α), Not (IsMin a) → LT.lt (f a) (f (Order.pred a))
    ⊢ StrictAnti f
  -/
  simpa using strictAntiOn_of_lt_pred Set.ordConnected_univ (by simpa using hf)
  /-
    🎉 no goals
  -/


