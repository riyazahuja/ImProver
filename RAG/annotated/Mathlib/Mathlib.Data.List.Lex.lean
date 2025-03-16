/-- Given a strict order `<` on `α`, the lexicographic strict order on `List α`, for which
`[a0, ..., an] < [b0, ..., b_k]` if `a0 < b0` or `a0 = b0` and `[a1, ..., an] < [b1, ..., bk]`.
The definition is given for any relation `r`, not only strict orders. -/
inductive Lex (r : α → α → Prop) : List α → List α → Prop
  | nil {a l} : Lex r [] (a :: l)
  | cons {a l₁ l₂} (h : Lex r l₁ l₂) : Lex r (a :: l₁) (a :: l₂)
  | rel {a₁ l₁ a₂ l₂} (h : r a₁ a₂) : Lex r (a₁ :: l₁) (a₂ :: l₂)


theorem cons_iff {r : α → α → Prop} [IsIrrefl α r] {a l₁ l₂} :
    Lex r (a :: l₁) (a :: l₂) ↔ Lex r l₁ l₂ :=
               /-
                 α : Type u
                 r : α → α → Prop
                 inst✝ : IsIrrefl α r
                 a : α
                 l₁ l₂ : List α
                 h : List.Lex r (List.cons a l₁) (List.cons a l₂)
                 ⊢ List.Lex r l₁ l₂
               -/
  ⟨fun h => by cases' h with _ _ _ _ _ h _ _ _ _ h; exacts [h, (irrefl_of r a h).elim], Lex.cons⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem not_nil_right (r : α → α → Prop) (l : List α) : ¬Lex r l [] :=
  nofun


theorem nil_left_or_eq_nil {r : α → α → Prop} (l : List α) : List.Lex r [] l ∨ l = [] :=
  match l with
  | [] => Or.inr rfl
  | (_ :: _) => Or.inl nil


@[simp]
theorem singleton_iff {r : α → α → Prop} (a b : α) : List.Lex r [a] [b] ↔ r a b :=
  ⟨fun | rel h => h, List.Lex.rel⟩


instance isOrderConnected (r : α → α → Prop) [IsOrderConnected α r] [IsTrichotomous α r] :
    IsOrderConnected (List α) (Lex r) where
  conn := aux where
    aux
    | _, [], _ :: _, nil => Or.inr nil
    | _, [], _ :: _, rel _ => Or.inr nil
    | _, [], _ :: _, cons _ => Or.inr nil
    | _, _ :: _, _ :: _, nil => Or.inl nil
    | _ :: _, b :: _, _ :: _, rel h => (IsOrderConnected.conn _ b _ h).imp rel rel
    | a :: l₁, b :: l₂, _ :: l₃, cons h => by
      /-
        α : Type u
        r : α → α → Prop
        inst✝¹ : IsOrderConnected α r
        inst✝ : IsTrichotomous α r
        a : α
        l₁ : List α
        b : α
        l₂ l₃ : List α
        h : List.Lex r l₁ l₃
        ⊢ Or (List.Lex r (List.cons a l₁) (List.cons b l₂)) (List.Lex r (List.cons b l …
      -/
      rcases trichotomous_of r a b with (ab | rfl | ab)
        /-
          case inl
          α : Type u
          r : α → α → Prop
          inst✝¹ : IsOrderConnected α r
          inst✝ : IsTrichotomous α r
          a : α
          l₁ : List α
          b : α
          l₂ l₃ : List α
          h : List.Lex r l₁ l₃
          ab : r a b
          ⊢ Or (List.Lex r (List.cons a l₁) (List.cons b l₂)) (List.Lex r (List.cons b l …
        -/
      · exact Or.inl (rel ab)
        /-
          🎉 no goals
        -/
        /-
          case inr.inl
          α : Type u
          r : α → α → Prop
          inst✝¹ : IsOrderConnected α r
          inst✝ : IsTrichotomous α r
          a : α
          l₁ l₂ l₃ : List α
          h : List.Lex r l₁ l₃
          ⊢ Or (List.Lex r (List.cons a l₁) (List.cons a l₂)) (List.Lex r (List.cons a l …
        -/
      · exact (aux _ l₂ _ h).imp cons cons
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          α : Type u
          r : α → α → Prop
          inst✝¹ : IsOrderConnected α r
          inst✝ : IsTrichotomous α r
          a : α
          l₁ : List α
          b : α
          l₂ l₃ : List α
          h : List.Lex r l₁ l₃
          ab : r b a
          ⊢ Or (List.Lex r (List.cons a l₁) (List.cons b l₂)) (List.Lex r (List.cons b l …
        -/
      · exact Or.inr (rel ab)
        /-
          🎉 no goals
        -/


instance isTrichotomous (r : α → α → Prop) [IsTrichotomous α r] :
    IsTrichotomous (List α) (Lex r) where
  trichotomous := aux where
    aux
    | [], [] => Or.inr (Or.inl rfl)
    | [], _ :: _ => Or.inl nil
    | _ :: _, [] => Or.inr (Or.inr nil)
    | a :: l₁, b :: l₂ => by
      /-
        α : Type u
        r : α → α → Prop
        inst✝ : IsTrichotomous α r
        a : α
        l₁ : List α
        b : α
        l₂ : List α
        ⊢ Or (List.Lex r (List.cons a l₁) (List.cons b l₂)) (Or (Eq (List.cons a l₁) ( …
      -/
      rcases trichotomous_of r a b with (ab | rfl | ab)
        /-
          case inl
          α : Type u
          r : α → α → Prop
          inst✝ : IsTrichotomous α r
          a : α
          l₁ : List α
          b : α
          l₂ : List α
          ab : r a b
          ⊢ Or (List.Lex r (List.cons a l₁) (List.cons b l₂)) (Or (Eq (List.cons a l₁) ( …
        -/
      · exact Or.inl (rel ab)
        /-
          🎉 no goals
        -/
        /-
          case inr.inl
          α : Type u
          r : α → α → Prop
          inst✝ : IsTrichotomous α r
          a : α
          l₁ l₂ : List α
          ⊢ Or (List.Lex r (List.cons a l₁) (List.cons a l₂)) (Or (Eq (List.cons a l₁) ( …
        -/
      · exact (aux l₁ l₂).imp cons (Or.imp (congr_arg _) cons)
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          α : Type u
          r : α → α → Prop
          inst✝ : IsTrichotomous α r
          a : α
          l₁ : List α
          b : α
          l₂ : List α
          ab : r b a
          ⊢ Or (List.Lex r (List.cons a l₁) (List.cons b l₂)) (Or (Eq (List.cons a l₁) ( …
        -/
      · exact Or.inr (Or.inr (rel ab))
        /-
          🎉 no goals
        -/


instance isAsymm (r : α → α → Prop) [IsAsymm α r] : IsAsymm (List α) (Lex r) where
  asymm := aux where
    aux
    | _, _, Lex.rel h₁, Lex.rel h₂ => asymm h₁ h₂
    | _, _, Lex.rel h₁, Lex.cons _ => asymm h₁ h₁
    | _, _, Lex.cons _, Lex.rel h₂ => asymm h₂ h₂
    | _, _, Lex.cons h₁, Lex.cons h₂ => aux _ _ h₁ h₂


@[deprecated "No deprecation message was provided." (since := "2024-07-30")]
instance isStrictTotalOrder (r : α → α → Prop) [IsStrictTotalOrder α r] :
    IsStrictTotalOrder (List α) (Lex r) :=
  { isStrictWeakOrder_of_isOrderConnected with }


instance decidableRel [DecidableEq α] (r : α → α → Prop) [DecidableRel r] : DecidableRel (Lex r)
                                  /-
                                    α : Type u
                                    inst✝¹ : DecidableEq α
                                    r : α → α → Prop
                                    inst✝ : DecidableRel r
                                    l₁ : List α
                                    h : List.Lex r l₁ List.nil
                                    ⊢ False
                                  -/
  | l₁, [] => isFalse fun h => by cases h
                                  /-
                                    🎉 no goals
                                  -/
  | [], _ :: _ => isTrue Lex.nil
  | a :: l₁, b :: l₂ => by
    /-
      α : Type u
      inst✝¹ : DecidableEq α
      r : α → α → Prop
      inst✝ : DecidableRel r
      a : α
      l₁ : List α
      b : α
      l₂ : List α
      ⊢ Decidable (List.Lex r (List.cons a l₁) (List.cons b l₂))
    -/
    haveI := decidableRel r l₁ l₂
    /-
      α : Type u
      inst✝¹ : DecidableEq α
      r : α → α → Prop
      inst✝ : DecidableRel r
      a : α
      l₁ : List α
      b : α
      l₂ : List α
      this : Decidable (List.Lex r l₁ l₂)
      ⊢ Decidable (List.Lex r (List.cons a l₁) (List.cons b l₂))
    -/
    refine decidable_of_iff (r a b ∨ a = b ∧ Lex r l₁ l₂) ⟨fun h => ?_, fun h => ?_⟩
      /-
        case refine_1
        α : Type u
        inst✝¹ : DecidableEq α
        r : α → α → Prop
        inst✝ : DecidableRel r
        a : α
        l₁ : List α
        b : α
        l₂ : List α
        this : Decidable (List.Lex r l₁ l₂)
        h : Or (r a b) (And (Eq a b) (List.Lex r l₁ l₂))
        ⊢ List.Lex r (List.cons a l₁) (List.cons b l₂)
      -/
    · rcases h with (h | ⟨rfl, h⟩)
        /-
          case refine_1.inl
          α : Type u
          inst✝¹ : DecidableEq α
          r : α → α → Prop
          inst✝ : DecidableRel r
          a : α
          l₁ : List α
          b : α
          l₂ : List α
          this : Decidable (List.Lex r l₁ l₂)
          h : r a b
          ⊢ List.Lex r (List.cons a l₁) (List.cons b l₂)
        -/
      · exact Lex.rel h
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inr.intro
          α : Type u
          inst✝¹ : DecidableEq α
          r : α → α → Prop
          inst✝ : DecidableRel r
          a : α
          l₁ l₂ : List α
          this : Decidable (List.Lex r l₁ l₂)
          h : List.Lex r l₁ l₂
          ⊢ List.Lex r (List.cons a l₁) (List.cons a l₂)
        -/
      · exact Lex.cons h
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        α : Type u
        inst✝¹ : DecidableEq α
        r : α → α → Prop
        inst✝ : DecidableRel r
        a : α
        l₁ : List α
        b : α
        l₂ : List α
        this : Decidable (List.Lex r l₁ l₂)
        h : List.Lex r (List.cons a l₁) (List.cons b l₂)
        ⊢ Or (r a b) (And (Eq a b) (List.Lex r l₁ l₂))
      -/
    · rcases h with (_ | h | h)
        /-
          case refine_2.cons
          α : Type u
          inst✝¹ : DecidableEq α
          r : α → α → Prop
          inst✝ : DecidableRel r
          a : α
          l₁ l₂ : List α
          this : Decidable (List.Lex r l₁ l₂)
          h : List.Lex r l₁ l₂
          ⊢ Or (r a a) (And (Eq a a) (List.Lex r l₁ l₂))
        -/
      · exact Or.inr ⟨rfl, h⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_2.rel
          α : Type u
          inst✝¹ : DecidableEq α
          r : α → α → Prop
          inst✝ : DecidableRel r
          a : α
          l₁ : List α
          b : α
          l₂ : List α
          this : Decidable (List.Lex r l₁ l₂)
          h : r a b
          ⊢ Or (r a b) (And (Eq a b) (List.Lex r l₁ l₂))
        -/
      · exact Or.inl h
        /-
          🎉 no goals
        -/


theorem append_right (r : α → α → Prop) : ∀ {s₁ s₂} (t), Lex r s₁ s₂ → Lex r s₁ (s₂ ++ t)
  | _, _, _, nil => nil
  | _, _, _, cons h => cons (append_right r _ h)
  | _, _, _, rel r => rel r


theorem append_left (R : α → α → Prop) {t₁ t₂} (h : Lex R t₁ t₂) : ∀ s, Lex R (s ++ t₁) (s ++ t₂)
  | [] => h
  | _ :: l => cons (append_left R h l)


theorem imp {r s : α → α → Prop} (H : ∀ a b, r a b → s a b) : ∀ l₁ l₂, Lex r l₁ l₂ → Lex s l₁ l₂
  | _, _, nil => nil
  | _, _, cons h => cons (imp H _ _ h)
  | _, _, rel r => rel (H _ _ r)


theorem to_ne : ∀ {l₁ l₂ : List α}, Lex (· ≠ ·) l₁ l₂ → l₁ ≠ l₂
  | _, _, cons h, e => to_ne h (List.cons.inj e).2
  | _, _, rel r, e => r (List.cons.inj e).1


theorem _root_.Decidable.List.Lex.ne_iff [DecidableEq α] {l₁ l₂ : List α}
    (H : length l₁ ≤ length l₂) : Lex (· ≠ ·) l₁ l₂ ↔ l₁ ≠ l₂ :=
  ⟨to_ne, fun h => by
    /-
      α : Type u
      inst✝ : DecidableEq α
      l₁ l₂ : List α
      H : LE.le l₁.length l₂.length
      h : Ne l₁ l₂
      ⊢ List.Lex (fun x1 x2 => Ne x1 x2) l₁ l₂
    -/
    induction' l₁ with a l₁ IH generalizing l₂ <;> cases' l₂ with b l₂
      /-
        case nil.nil
        α : Type u
        inst✝ : DecidableEq α
        H : LE.le List.nil.length List.nil.length
        h : Ne List.nil List.nil
        ⊢ List.Lex (fun x1 x2 => Ne x1 x2) List.nil List.nil
      -/
    · contradiction
      /-
        🎉 no goals
      -/
      /-
        case nil.cons
        α : Type u
        inst✝ : DecidableEq α
        b : α
        l₂ : List α
        H : LE.le List.nil.length (List.cons b l₂).length
        h : Ne List.nil (List.cons b l₂)
        ⊢ List.Lex (fun x1 x2 => Ne x1 x2) List.nil (List.cons b l₂)
      -/
    · apply nil
      /-
        🎉 no goals
      -/
      /-
        case cons.nil
        α : Type u
        inst✝ : DecidableEq α
        a : α
        l₁ : List α
        IH : ∀ {l₂ : List α}, LE.le l₁.length l₂.length → Ne l₁ l₂ → List.Lex (fun x1  …
        H : LE.le (List.cons a l₁).length List.nil.length
        h : Ne (List.cons a l₁) List.nil
        ⊢ List.Lex (fun x1 x2 => Ne x1 x2) (List.cons a l₁) List.nil
      -/
    · exact (not_lt_of_ge H).elim (succ_pos _)
      /-
        🎉 no goals
      -/
      /-
        case cons.cons
        α : Type u
        inst✝ : DecidableEq α
        a : α
        l₁ : List α
        IH : ∀ {l₂ : List α}, LE.le l₁.length l₂.length → Ne l₁ l₂ → List.Lex (fun x1  …
        b : α
        l₂ : List α
        H : LE.le (List.cons a l₁).length (List.cons b l₂).length
        h : Ne (List.cons a l₁) (List.cons b l₂)
        ⊢ List.Lex (fun x1 x2 => Ne x1 x2) (List.cons a l₁) (List.cons b l₂)
      -/
    · by_cases ab : a = b
        /-
          case pos
          α : Type u
          inst✝ : DecidableEq α
          a : α
          l₁ : List α
          IH : ∀ {l₂ : List α}, LE.le l₁.length l₂.length → Ne l₁ l₂ → List.Lex (fun x1  …
          b : α
          l₂ : List α
          H : LE.le (List.cons a l₁).length (List.cons b l₂).length
          h : Ne (List.cons a l₁) (List.cons b l₂)
          ab : Eq a b
          ⊢ List.Lex (fun x1 x2 => Ne x1 x2) (List.cons a l₁) (List.cons b l₂)
        -/
      · subst b
        /-
          case pos
          α : Type u
          inst✝ : DecidableEq α
          a : α
          l₁ : List α
          IH : ∀ {l₂ : List α}, LE.le l₁.length l₂.length → Ne l₁ l₂ → List.Lex (fun x1  …
          l₂ : List α
          H : LE.le (List.cons a l₁).length (List.cons a l₂).length
          h : Ne (List.cons a l₁) (List.cons a l₂)
          ⊢ List.Lex (fun x1 x2 => Ne x1 x2) (List.cons a l₁) (List.cons a l₂)
        -/
        apply cons
        /-
          case pos.h
          α : Type u
          inst✝ : DecidableEq α
          a : α
          l₁ : List α
          IH : ∀ {l₂ : List α}, LE.le l₁.length l₂.length → Ne l₁ l₂ → List.Lex (fun x1  …
          l₂ : List α
          H : LE.le (List.cons a l₁).length (List.cons a l₂).length
          h : Ne (List.cons a l₁) (List.cons a l₂)
          ⊢ List.Lex (fun x1 x2 => Ne x1 x2) l₁ l₂
        -/
        exact IH (le_of_succ_le_succ H) (mt (congr_arg _) h)
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u
          inst✝ : DecidableEq α
          a : α
          l₁ : List α
          IH : ∀ {l₂ : List α}, LE.le l₁.length l₂.length → Ne l₁ l₂ → List.Lex (fun x1  …
          b : α
          l₂ : List α
          H : LE.le (List.cons a l₁).length (List.cons b l₂).length
          h : Ne (List.cons a l₁) (List.cons b l₂)
          ab : Not (Eq a b)
          ⊢ List.Lex (fun x1 x2 => Ne x1 x2) (List.cons a l₁) (List.cons b l₂)
        -/
      · exact rel ab ⟩
        /-
          🎉 no goals
        -/


theorem ne_iff {l₁ l₂ : List α} (H : length l₁ ≤ length l₂) : Lex (· ≠ ·) l₁ l₂ ↔ l₁ ≠ l₂ := by
  classical
  exact Decidable.List.Lex.ne_iff H


instance LT' [LT α] : LT (List α) :=
  ⟨Lex (· < ·)⟩


theorem nil_lt_cons [LT α] (a : α) (l : List α) : [] < a :: l :=
  Lex.nil


instance [LinearOrder α] : LinearOrder (List α) :=
  linearOrderOfSTO (Lex (· < ·))

--Note: this overrides an instance in core lean

instance LE' [LinearOrder α] : LE (List α) :=
  Preorder.toLE


theorem lt_iff_lex_lt [LinearOrder α] (l l' : List α) : lt l l' ↔ Lex (· < ·) l l' := by
  /-
    α : Type u
    inst✝ : LinearOrder α
    l l' : List α
    ⊢ Iff (l.lt l') (List.Lex (fun x1 x2 => LT.lt x1 x2) l l')
  -/
  constructor <;>
  /-
    case mp
    α : Type u
    inst✝ : LinearOrder α
    l l' : List α
    ⊢ l.lt l' → List.Lex (fun x1 x2 => LT.lt x1 x2) l l'
  -/
  intro h
  · induction h with
    | nil b bs => exact Lex.nil
    | @head a as b bs hab => apply Lex.rel; assumption
    | @tail a as b bs hab hba _ ih =>
      have heq : a = b := _root_.le_antisymm (le_of_not_lt hba) (le_of_not_lt hab)
      subst b; apply Lex.cons; assumption
  · induction h with
    | @nil a as => apply lt.nil
    | @cons a as bs _ ih => apply lt.tail <;> simp [ih]
    | @rel a as b bs h => apply lt.head; assumption


@[simp]
theorem nil_le {α} [LinearOrder α] {l : List α} : [] ≤ l :=
  match l with
  | [] => le_rfl
  | _ :: _ => le_of_lt <| nil_lt_cons _ _


theorem head_le_of_lt [Preorder α] {a a' : α} {l l' : List α} (h : (a' :: l') < (a :: l)) :
    a' ≤ a :=
  match h with
  | .cons _ => le_rfl
  | .rel h => h.le


theorem head!_le_of_lt [Preorder α] [Inhabited α] (l l' : List α) (h : l' < l) (hl' : l' ≠ []) :
    l'.head! ≤ l.head! := by
  /-
    α : Type u
    inst✝¹ : Preorder α
    inst✝ : Inhabited α
    l l' : List α
    h : LT.lt l' l
    hl' : Ne l' List.nil
    ⊢ LE.le l'.head! l.head!
  -/
  replace h : List.Lex (· < ·) l' l := h
  /-
    α : Type u
    inst✝¹ : Preorder α
    inst✝ : Inhabited α
    l l' : List α
    hl' : Ne l' List.nil
    h : List.Lex (fun x1 x2 => LT.lt x1 x2) l' l
    ⊢ LE.le l'.head! l.head!
  -/
  by_cases hl : l = []
    /-
      case pos
      α : Type u
      inst✝¹ : Preorder α
      inst✝ : Inhabited α
      l l' : List α
      hl' : Ne l' List.nil
      h : List.Lex (fun x1 x2 => LT.lt x1 x2) l' l
      hl : Eq l List.nil
      ⊢ LE.le l'.head! l.head!
    -/
  · simp [hl] at h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝¹ : Preorder α
      inst✝ : Inhabited α
      l l' : List α
      hl' : Ne l' List.nil
      h : List.Lex (fun x1 x2 => LT.lt x1 x2) l' l
      hl : Not (Eq l List.nil)
      ⊢ LE.le l'.head! l.head!
    -/
  · rw [← List.cons_head!_tail hl', ← List.cons_head!_tail hl] at h
    /-
      case neg
      α : Type u
      inst✝¹ : Preorder α
      inst✝ : Inhabited α
      l l' : List α
      hl' : Ne l' List.nil
      h : List.Lex (fun x1 x2 => LT.lt x1 x2) (List.cons l'.head! l'.tail) (List.con …
      hl : Not (Eq l List.nil)
      ⊢ LE.le l'.head! l.head!
    -/
    exact head_le_of_lt h
    /-
      🎉 no goals
    -/


theorem cons_le_cons [LinearOrder α] (a : α) {l l' : List α} (h : l' ≤ l) :
    a :: l' ≤ a :: l := by
  /-
    α : Type u
    inst✝ : LinearOrder α
    a : α
    l l' : List α
    h : LE.le l' l
    ⊢ LE.le (List.cons a l') (List.cons a l)
  -/
  rw [le_iff_lt_or_eq] at h ⊢
  /-
    α : Type u
    inst✝ : LinearOrder α
    a : α
    l l' : List α
    h : Or (LT.lt l' l) (Eq l' l)
    ⊢ Or (LT.lt (List.cons a l') (List.cons a l)) (Eq (List.cons a l') (List.cons  …
  -/
  exact h.imp .cons (congr_arg _)
  /-
    🎉 no goals
  -/


