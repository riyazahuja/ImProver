@[simp]
theorem destutter'_nil : destutter' R a [] = [a] :=
  rfl


theorem destutter'_cons :
    (b :: l).destutter' R a = if R a b then a :: destutter' R b l else destutter' R a l :=
  rfl


@[simp]
theorem destutter'_cons_pos (h : R b a) : (a :: l).destutter' R b = b :: l.destutter' R a := by
  /-
    α : Type u_1
    l : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    a b : α
    h : R b a
    ⊢ Eq (List.destutter' R b (List.cons a l)) (List.cons b (List.destutter' R a l))
  -/
  rw [destutter', if_pos h]
  /-
    🎉 no goals
  -/


@[simp]
theorem destutter'_cons_neg (h : ¬R b a) : (a :: l).destutter' R b = l.destutter' R b := by
  /-
    α : Type u_1
    l : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    a b : α
    h : Not (R b a)
    ⊢ Eq (List.destutter' R b (List.cons a l)) (List.destutter' R b l)
  -/
  rw [destutter', if_neg h]
  /-
    🎉 no goals
  -/


@[simp]
theorem destutter'_singleton : [b].destutter' R a = if R a b then [a, b] else [a] := by
  /-
    α : Type u_1
    R : α → α → Prop
    inst✝ : DecidableRel R
    a b : α
    ⊢ Eq (List.destutter' R a (List.cons b List.nil)) (ite (R a b) (List.cons a (L …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp! [h]
                       /-
                         🎉 no goals
                       -/


theorem destutter'_sublist (a) : l.destutter' R a <+ a :: l := by
  /-
    α : Type u_1
    l : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    a : α
    ⊢ (List.destutter' R a l).Sublist (List.cons a l)
  -/
  induction' l with b l hl generalizing a
    /-
      case nil
      α : Type u_1
      l : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      a : α
      ⊢ (List.destutter' R a List.nil).Sublist (List.cons a List.nil)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    l✝ : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    b : α
    l : List α
    hl : ∀ (a : α), (List.destutter' R a l).Sublist (List.cons a l)
    a : α
    ⊢ (List.destutter' R a (List.cons b l)).Sublist (List.cons a (List.cons b l))
  -/
  rw [destutter']
  /-
    case cons
    α : Type u_1
    l✝ : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    b : α
    l : List α
    hl : ∀ (a : α), (List.destutter' R a l).Sublist (List.cons a l)
    a : α
    ⊢ (ite (R a b) (List.cons a (List.destutter' R b l)) (List.destutter' R a l)). …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      l✝ : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      b : α
      l : List α
      hl : ∀ (a : α), (List.destutter' R a l).Sublist (List.cons a l)
      a : α
      h✝ : R a b
      ⊢ (List.cons a (List.destutter' R b l)).Sublist (List.cons a (List.cons b l))
    -/
  · exact Sublist.cons₂ a (hl b)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      l✝ : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      b : α
      l : List α
      hl : ∀ (a : α), (List.destutter' R a l).Sublist (List.cons a l)
      a : α
      h✝ : Not (R a b)
      ⊢ (List.destutter' R a l).Sublist (List.cons a (List.cons b l))
    -/
  · exact (hl a).trans ((l.sublist_cons_self b).cons_cons a)
    /-
      🎉 no goals
    -/


theorem mem_destutter' (a) : a ∈ l.destutter' R a := by
  /-
    α : Type u_1
    l : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    a : α
    ⊢ Membership.mem (List.destutter' R a l) a
  -/
  induction' l with b l hl
    /-
      case nil
      α : Type u_1
      l : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      a : α
      ⊢ Membership.mem (List.destutter' R a List.nil) a
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    l✝ : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    a b : α
    l : List α
    hl : Membership.mem (List.destutter' R a l) a
    ⊢ Membership.mem (List.destutter' R a (List.cons b l)) a
  -/
  rw [destutter']
  /-
    case cons
    α : Type u_1
    l✝ : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    a b : α
    l : List α
    hl : Membership.mem (List.destutter' R a l) a
    ⊢ Membership.mem (ite (R a b) (List.cons a (List.destutter' R b l)) (List.dest …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      l✝ : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      a b : α
      l : List α
      hl : Membership.mem (List.destutter' R a l) a
      h✝ : R a b
      ⊢ Membership.mem (List.cons a (List.destutter' R b l)) a
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      l✝ : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      a b : α
      l : List α
      hl : Membership.mem (List.destutter' R a l) a
      h✝ : Not (R a b)
      ⊢ Membership.mem (List.destutter' R a l) a
    -/
  · assumption
    /-
      🎉 no goals
    -/


theorem destutter'_is_chain : ∀ l : List α, ∀ {a b}, R a b → (l.destutter' R b).Chain R a
  | [], _, _, h => chain_singleton.mpr h
  | c :: l, a, b, h => by
    /-
      α : Type u_1
      R : α → α → Prop
      inst✝ : DecidableRel R
      c : α
      l : List α
      a b : α
      h : R a b
      ⊢ List.Chain R a (List.destutter' R b (List.cons c l))
    -/
    rw [destutter']
    /-
      α : Type u_1
      R : α → α → Prop
      inst✝ : DecidableRel R
      c : α
      l : List α
      a b : α
      h : R a b
      ⊢ List.Chain R a (ite (R b c) (List.cons b (List.destutter' R c l)) (List.dest …
    -/
    split_ifs with hbc
      /-
        case pos
        α : Type u_1
        R : α → α → Prop
        inst✝ : DecidableRel R
        c : α
        l : List α
        a b : α
        h : R a b
        hbc : R b c
        ⊢ List.Chain R a (List.cons b (List.destutter' R c l))
      -/
    · rw [chain_cons]
      /-
        case pos
        α : Type u_1
        R : α → α → Prop
        inst✝ : DecidableRel R
        c : α
        l : List α
        a b : α
        h : R a b
        hbc : R b c
        ⊢ And (R a b) (List.Chain R b (List.destutter' R c l))
      -/
      exact ⟨h, destutter'_is_chain l hbc⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        R : α → α → Prop
        inst✝ : DecidableRel R
        c : α
        l : List α
        a b : α
        h : R a b
        hbc : Not (R b c)
        ⊢ List.Chain R a (List.destutter' R b l)
      -/
    · exact destutter'_is_chain l h
      /-
        🎉 no goals
      -/


theorem destutter'_is_chain' (a) : (l.destutter' R a).Chain' R := by
  /-
    α : Type u_1
    l : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    a : α
    ⊢ List.Chain' R (List.destutter' R a l)
  -/
  induction' l with b l hl generalizing a
    /-
      case nil
      α : Type u_1
      l : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      a : α
      ⊢ List.Chain' R (List.destutter' R a List.nil)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    l✝ : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    b : α
    l : List α
    hl : ∀ (a : α), List.Chain' R (List.destutter' R a l)
    a : α
    ⊢ List.Chain' R (List.destutter' R a (List.cons b l))
  -/
  rw [destutter']
  /-
    case cons
    α : Type u_1
    l✝ : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    b : α
    l : List α
    hl : ∀ (a : α), List.Chain' R (List.destutter' R a l)
    a : α
    ⊢ List.Chain' R (ite (R a b) (List.cons a (List.destutter' R b l)) (List.destu …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      l✝ : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      b : α
      l : List α
      hl : ∀ (a : α), List.Chain' R (List.destutter' R a l)
      a : α
      h : R a b
      ⊢ List.Chain' R (List.cons a (List.destutter' R b l))
    -/
  · exact destutter'_is_chain R l h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      l✝ : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      b : α
      l : List α
      hl : ∀ (a : α), List.Chain' R (List.destutter' R a l)
      a : α
      h : Not (R a b)
      ⊢ List.Chain' R (List.destutter' R a l)
    -/
  · exact hl a
    /-
      🎉 no goals
    -/


theorem destutter'_of_chain (h : l.Chain R a) : l.destutter' R a = a :: l := by
  /-
    α : Type u_1
    l : List α
    R : α → α → Prop
    inst✝ : DecidableRel R
    a : α
    h : List.Chain R a l
    ⊢ Eq (List.destutter' R a l) (List.cons a l)
  -/
  induction' l with b l hb generalizing a
    /-
      case nil
      α : Type u_1
      R : α → α → Prop
      inst✝ : DecidableRel R
      a : α
      h : List.Chain R a List.nil
      ⊢ Eq (List.destutter' R a List.nil) (List.cons a List.nil)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    R : α → α → Prop
    inst✝ : DecidableRel R
    b : α
    l : List α
    hb : ∀ {a : α}, List.Chain R a l → Eq (List.destutter' R a l) (List.cons a l)
    a : α
    h : List.Chain R a (List.cons b l)
    ⊢ Eq (List.destutter' R a (List.cons b l)) (List.cons a (List.cons b l))
  -/
  obtain ⟨h, hc⟩ := chain_cons.mp h
  /-
    case cons.intro
    α : Type u_1
    R : α → α → Prop
    inst✝ : DecidableRel R
    b : α
    l : List α
    hb : ∀ {a : α}, List.Chain R a l → Eq (List.destutter' R a l) (List.cons a l)
    a : α
    h✝ : List.Chain R a (List.cons b l)
    h : R a b
    hc : List.Chain R b l
    ⊢ Eq (List.destutter' R a (List.cons b l)) (List.cons a (List.cons b l))
  -/
  rw [l.destutter'_cons_pos h, hb hc]
  /-
    🎉 no goals
  -/


@[simp]
theorem destutter'_eq_self_iff (a) : l.destutter' R a = a :: l ↔ l.Chain R a :=
  ⟨fun h => by
    suffices Chain' R (a::l) by
      assumption
    /-
      α : Type u_1
      l : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      a : α
      h : Eq (List.destutter' R a l) (List.cons a l)
      ⊢ List.Chain' R (List.cons a l)
    -/
    rw [← h]
    /-
      α : Type u_1
      l : List α
      R : α → α → Prop
      inst✝ : DecidableRel R
      a : α
      h : Eq (List.destutter' R a l) (List.cons a l)
      ⊢ List.Chain' R (List.destutter' R a l)
    -/
    exact l.destutter'_is_chain' R a, destutter'_of_chain _ _⟩
    /-
      🎉 no goals
    -/


theorem destutter'_ne_nil : l.destutter' R a ≠ [] :=
  ne_nil_of_mem <| l.mem_destutter' R a


@[simp]
theorem destutter_nil : ([] : List α).destutter R = [] :=
  rfl


theorem destutter_cons' : (a :: l).destutter R = destutter' R a l :=
  rfl


theorem destutter_cons_cons :
    (a :: b :: l).destutter R = if R a b then a :: destutter' R b l else destutter' R a l :=
  rfl


@[simp]
theorem destutter_singleton : destutter R [a] = [a] :=
  rfl


@[simp]
theorem destutter_pair : destutter R [a, b] = if R a b then [a, b] else [a] :=
  destutter_cons_cons _ R


theorem destutter_sublist : ∀ l : List α, l.destutter R <+ l
  | [] => Sublist.slnil
  | h :: l => l.destutter'_sublist R h


theorem destutter_is_chain' : ∀ l : List α, (l.destutter R).Chain' R
  | [] => List.chain'_nil
  | h :: l => l.destutter'_is_chain' R h


theorem destutter_of_chain' : ∀ l : List α, l.Chain' R → l.destutter R = l
  | [], _ => rfl
  | _ :: l, h => l.destutter'_of_chain _ h


@[simp]
theorem destutter_eq_self_iff : ∀ l : List α, l.destutter R = l ↔ l.Chain' R
             /-
               α : Type u_1
               R : α → α → Prop
               inst✝ : DecidableRel R
               ⊢ Iff (Eq (List.destutter R List.nil) List.nil) (List.Chain' R List.nil)
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | a :: l => l.destutter'_eq_self_iff R a


theorem destutter_idem : (l.destutter R).destutter R = l.destutter R :=
  destutter_of_chain' R _ <| l.destutter_is_chain' R


@[simp]
theorem destutter_eq_nil : ∀ {l : List α}, destutter R l = [] ↔ l = []
  | [] => Iff.rfl
  | _ :: l => ⟨fun h => absurd h <| l.destutter'_ne_nil R, fun h => nomatch h⟩


/-- For a relation-preserving map, `destutter` commutes with `map`. -/
theorem map_destutter {f : α → β} : ∀ {l : List α}, (∀ a ∈ l, ∀ b ∈ l, R a b ↔ R₂ (f a) (f b)) →
    (l.destutter R).map f = (l.map f).destutter R₂
                 /-
                   α : Type u_1
                   β : Type u_2
                   R : α → α → Prop
                   inst✝¹ : DecidableRel R
                   R₂ : β → β → Prop
                   inst✝ : DecidableRel R₂
                   f : α → β
                   hl : ∀ (a : α), Membership.mem List.nil a → ∀ (b : α), Membership.mem List.nil …
                   ⊢ Eq (List.map f (List.destutter R List.nil)) (List.destutter R₂ (List.map f L …
                 -/
  | [], hl => by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    α : Type u_1
                    β : Type u_2
                    R : α → α → Prop
                    inst✝¹ : DecidableRel R
                    R₂ : β → β → Prop
                    inst✝ : DecidableRel R₂
                    f : α → β
                    a : α
                    hl : ∀ (a_1 : α), Membership.mem (List.cons a List.nil) a_1 → ∀ (b : α), Membe …
                    ⊢ Eq (List.map f (List.destutter R (List.cons a List.nil))) (List.destutter R₂ …
                  -/
  | [a], hl => by simp
                  /-
                    🎉 no goals
                  -/
  | a :: b :: l, hl => by
    /-
      α : Type u_1
      β : Type u_2
      R : α → α → Prop
      inst✝¹ : DecidableRel R
      R₂ : β → β → Prop
      inst✝ : DecidableRel R₂
      f : α → β
      a b : α
      l : List α
      hl : ∀ (a_1 : α), Membership.mem (List.cons a (List.cons b l)) a_1 → ∀ (b_1 :  …
      ⊢ Eq (List.map f (List.destutter R (List.cons a (List.cons b l)))) (List.destu …
    -/
    have := hl a (by simp) b (by simp)
    /-
      α : Type u_1
      β : Type u_2
      R : α → α → Prop
      inst✝¹ : DecidableRel R
      R₂ : β → β → Prop
      inst✝ : DecidableRel R₂
      f : α → β
      a b : α
      l : List α
      hl : ∀ (a_1 : α), Membership.mem (List.cons a (List.cons b l)) a_1 → ∀ (b_1 :  …
      this : Iff (R a b) (R₂ (f a) (f b))
      ⊢ Eq (List.map f (List.destutter R (List.cons a (List.cons b l)))) (List.destu …
    -/
    simp_rw [map_cons, destutter_cons_cons, ← this]
    /-
      α : Type u_1
      β : Type u_2
      R : α → α → Prop
      inst✝¹ : DecidableRel R
      R₂ : β → β → Prop
      inst✝ : DecidableRel R₂
      f : α → β
      a b : α
      l : List α
      hl : ∀ (a_1 : α), Membership.mem (List.cons a (List.cons b l)) a_1 → ∀ (b_1 :  …
      this : Iff (R a b) (R₂ (f a) (f b))
      ⊢ Eq (List.map f (ite (R a b) (List.cons a (List.destutter' R b l)) (List.dest …
    -/
    by_cases hr : R a b <;>
      simp [hr, ← destutter_cons', map_destutter fun c hc d hd ↦ hl _ (cons_subset_cons _
        (subset_cons_self _ _) hc) _ (cons_subset_cons _ (subset_cons_self _ _) hd),
        map_destutter fun c hc d hd ↦ hl _ (subset_cons_self _ _ hc) _ (subset_cons_self _ _ hd)]


/-- For a injective function `f`, `destutter' (·≠·)` commutes with `map f`. -/
theorem map_destutter_ne {f : α → β} (h : Injective f) [DecidableEq α] [DecidableEq β] :
    (l.destutter (·≠·)).map f = (l.map f).destutter (·≠·) :=
  map_destutter fun _ _ _ _ ↦ h.ne_iff.symm


/-- `destutter'` on a relation like ≠ or <, whose negation is transitive, has length monotone
under a `¬R` changing of the first element. -/
theorem length_destutter'_cotrans_ge [i : IsTrans α Rᶜ] :
    ∀ {a} {l : List α}, ¬R b a → (l.destutter' R b).length ≤ (l.destutter' R a).length
                     /-
                       α : Type u_1
                       R : α → α → Prop
                       inst✝ : DecidableRel R
                       b : α
                       i : IsTrans α (HasCompl.compl R)
                       a : α
                       hba : Not (R b a)
                       ⊢ LE.le (List.destutter' R b List.nil).length (List.destutter' R a List.nil).l …
                     -/
  | a, [], hba => by simp
                     /-
                       🎉 no goals
                     -/
  | a, c :: l, hba => by
    /-
      α : Type u_1
      R : α → α → Prop
      inst✝ : DecidableRel R
      b : α
      i : IsTrans α (HasCompl.compl R)
      a c : α
      l : List α
      hba : Not (R b a)
      ⊢ LE.le (List.destutter' R b (List.cons c l)).length (List.destutter' R a (Lis …
    -/
    by_cases hbc : R b c
    case pos =>
      have hac : ¬Rᶜ a c := (mt (_root_.trans hba)) (not_not.2 hbc)
      simp_rw [destutter', if_pos (not_not.1 hac), if_pos hbc, length_cons, le_refl]
    case neg =>
      simp only [destutter', if_neg hbc]
      by_cases hac : R a c
      case pos =>
        simp only [if_pos hac, length_cons]
        exact Nat.le_succ_of_le (length_destutter'_cotrans_ge hbc)
      case neg =>
        simp only [if_neg hac]
        exact length_destutter'_cotrans_ge hba


/-- `List.destutter'` on a relation like `≠`, whose negation is an equivalence, gives the same
length if the first elements are not related. -/
theorem length_destutter'_congr [IsEquiv α Rᶜ] (hab : ¬R a b) :
    (l.destutter' R a).length = (l.destutter' R b).length :=
  (length_destutter'_cotrans_ge hab).antisymm <| length_destutter'_cotrans_ge (symm hab : Rᶜ b a)


/-- `List.destutter'` on a relation like ≠, whose negation is an equivalence, has length
    monotonic under List.cons -/
/-
TODO: Replace this lemma by the more general version:
theorem Sublist.length_destutter'_mono [IsEquiv α Rᶜ] (h : a :: l₁ <+ b :: l₂) :
    (List.destutter' R a l₁).length ≤ (List.destutter' R b l₂).length
-/
theorem le_length_destutter'_cons [IsEquiv α Rᶜ] :
    ∀ {l : List α}, (l.destutter' R b).length ≤ ((b :: l).destutter' R a).length
             /-
               α : Type u_1
               R : α → α → Prop
               inst✝¹ : DecidableRel R
               a b : α
               inst✝ : IsEquiv α (HasCompl.compl R)
               ⊢ LE.le (List.destutter' R b List.nil).length (List.destutter' R a (List.cons  …
             -/
                                        /-
                                          🎉 no goals
                                        -/
  | [] => by by_cases hab : (R a b) <;> simp_all [Nat.le_succ]
                                        /-
                                          🎉 no goals
                                        -/
  | c :: cs => by
    /-
      α : Type u_1
      R : α → α → Prop
      inst✝¹ : DecidableRel R
      a b : α
      inst✝ : IsEquiv α (HasCompl.compl R)
      c : α
      cs : List α
      ⊢ LE.le (List.destutter' R b (List.cons c cs)).length (List.destutter' R a (Li …
    -/
    by_cases hab : R a b
    /-
      case pos
      α : Type u_1
      R : α → α → Prop
      inst✝¹ : DecidableRel R
      a b : α
      inst✝ : IsEquiv α (HasCompl.compl R)
      c : α
      cs : List α
      hab : R a b
      ⊢ LE.le (List.destutter' R b (List.cons c cs)).length (List.destutter' R a (Li …
    -/
    case pos => simp [destutter', if_pos hab, Nat.le_succ]
    /-
      case neg
      α : Type u_1
      R : α → α → Prop
      inst✝¹ : DecidableRel R
      a b : α
      inst✝ : IsEquiv α (HasCompl.compl R)
      c : α
      cs : List α
      hab : Not (R a b)
      ⊢ LE.le (List.destutter' R b (List.cons c cs)).length (List.destutter' R a (Li …
    -/
    obtain hac | hac : R a c ∨ Rᶜ a c := em _
      /-
        case neg.inl
        α : Type u_1
        R : α → α → Prop
        inst✝¹ : DecidableRel R
        a b : α
        inst✝ : IsEquiv α (HasCompl.compl R)
        c : α
        cs : List α
        hab : Not (R a b)
        hac : R a c
        ⊢ LE.le (List.destutter' R b (List.cons c cs)).length (List.destutter' R a (Li …
      -/
    · have hbc : ¬Rᶜ b c := mt (_root_.trans hab) (not_not.2 hac)
      /-
        case neg.inl
        α : Type u_1
        R : α → α → Prop
        inst✝¹ : DecidableRel R
        a b : α
        inst✝ : IsEquiv α (HasCompl.compl R)
        c : α
        cs : List α
        hab : Not (R a b)
        hac : R a c
        hbc : Not (HasCompl.compl R b c)
        ⊢ LE.le (List.destutter' R b (List.cons c cs)).length (List.destutter' R a (Li …
      -/
      simp [destutter', if_pos hac, if_pos (not_not.1 hbc), if_neg hab]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        α : Type u_1
        R : α → α → Prop
        inst✝¹ : DecidableRel R
        a b : α
        inst✝ : IsEquiv α (HasCompl.compl R)
        c : α
        cs : List α
        hab : Not (R a b)
        hac : HasCompl.compl R a c
        ⊢ LE.le (List.destutter' R b (List.cons c cs)).length (List.destutter' R a (Li …
      -/
    · have hbc : ¬R b c := trans (symm hab) hac
      /-
        case neg.inr
        α : Type u_1
        R : α → α → Prop
        inst✝¹ : DecidableRel R
        a b : α
        inst✝ : IsEquiv α (HasCompl.compl R)
        c : α
        cs : List α
        hab : Not (R a b)
        hac : HasCompl.compl R a c
        hbc : Not (R b c)
        ⊢ LE.le (List.destutter' R b (List.cons c cs)).length (List.destutter' R a (Li …
      -/
      simp only [destutter', if_neg hbc, if_neg hac, if_neg hab]
      /-
        case neg.inr
        α : Type u_1
        R : α → α → Prop
        inst✝¹ : DecidableRel R
        a b : α
        inst✝ : IsEquiv α (HasCompl.compl R)
        c : α
        cs : List α
        hab : Not (R a b)
        hac : HasCompl.compl R a c
        hbc : Not (R b c)
        ⊢ LE.le (List.destutter' R b cs).length (List.destutter' R a cs).length
      -/
      exact (length_destutter'_congr cs hab).ge
      /-
        🎉 no goals
      -/


/-- `List.destutter` on a relation like ≠, whose negation is an equivalence, has length
monotone under List.cons -/
theorem length_destutter_le_length_destutter_cons [IsEquiv α Rᶜ] :
    ∀ {l : List α}, (l.destutter R).length ≤ ((a :: l).destutter R).length
             /-
               α : Type u_1
               R : α → α → Prop
               inst✝¹ : DecidableRel R
               a : α
               inst✝ : IsEquiv α (HasCompl.compl R)
               ⊢ LE.le (List.destutter R List.nil).length (List.destutter R (List.cons a List …
             -/
  | [] => by simp [destutter]
             /-
               🎉 no goals
             -/
  | b :: l => le_length_destutter'_cons


/-- `destutter ≠` has length monotone under `List.cons`. -/
theorem length_destutter_ne_le_length_destutter_cons [DecidableEq α] :
    (l.destutter (· ≠ ·)).length ≤ ((a :: l).destutter (· ≠ ·)).length :=
  length_destutter_le_length_destutter_cons


/-- `destutter` of relations like `≠`, whose negation is an equivalence relation,
gives a list of maximal length over any chain.

In other words, `l.destutter R` is an `R`-chain sublist of `l`, and is at least as long as any other
`R`-chain sublist. -/
lemma Chain'.length_le_length_destutter [IsEquiv α Rᶜ] :
    ∀ {l₁ l₂ : List α}, l₁ <+ l₂ → l₁.Chain' R → l₁.length ≤ (l₂.destutter R).length
  -- `l₁ := []`, `l₂ := []`
                       /-
                         α : Type u_1
                         R : α → α → Prop
                         inst✝¹ : DecidableRel R
                         inst✝ : IsEquiv α (HasCompl.compl R)
                         x✝¹ : List.nil.Sublist List.nil
                         x✝ : List.Chain' R List.nil
                         ⊢ LE.le List.nil.length (List.destutter R List.nil).length
                       -/
  | [], [], _, _ => by simp
                       /-
                         🎉 no goals
                       -/
  -- `l₁ := l₁`, `l₂ := a :: l₂`
  | l₁, _, .cons (l₂ := l₂) a hl, hl₁ =>
    (hl₁.length_le_length_destutter hl).trans length_destutter_le_length_destutter_cons
  -- `l₁ := [a]`, `l₂ := a :: l₂`
                                                       /-
                                                         α : Type u_1
                                                         R : α → α → Prop
                                                         inst✝¹ : DecidableRel R
                                                         inst✝ : IsEquiv α (HasCompl.compl R)
                                                         l₁ : List α
                                                         a : α
                                                         hl : List.nil.Sublist l₁
                                                         hl₁ : List.Chain' R (List.cons a List.nil)
                                                         ⊢ LE.le (List.cons a List.nil).length (List.destutter R (List.cons a l₁)).length
                                                       -/
  | _, _, .cons₂ (l₁ := []) (l₂ := l₁) a hl, hl₁ => by simp [Nat.one_le_iff_ne_zero]
                                                       /-
                                                         🎉 no goals
                                                       -/
  -- `l₁ := a :: l₁`, `l₂ := a :: b :: l₂`
  | _, _, .cons₂ a <| .cons (l₁ := l₁) (l₂ := l₂) b hl, hl₁ => by
    /-
      α : Type u_1
      R : α → α → Prop
      inst✝¹ : DecidableRel R
      inst✝ : IsEquiv α (HasCompl.compl R)
      l₁ : List α
      b : α
      l₂ : List α
      a : α
      hl : l₁.Sublist l₂
      hl₁ : List.Chain' R (List.cons a l₁)
      ⊢ LE.le (List.cons a l₁).length (List.destutter R (List.cons a (List.cons b l₂ …
    -/
    by_cases hab : R a b
      /-
        case pos
        α : Type u_1
        R : α → α → Prop
        inst✝¹ : DecidableRel R
        inst✝ : IsEquiv α (HasCompl.compl R)
        l₁ : List α
        b : α
        l₂ : List α
        a : α
        hl : l₁.Sublist l₂
        hl₁ : List.Chain' R (List.cons a l₁)
        hab : R a b
        ⊢ LE.le (List.cons a l₁).length (List.destutter R (List.cons a (List.cons b l₂ …
      -/
    · simpa [destutter_cons_cons, hab] using hl₁.tail.length_le_length_destutter (hl.cons _)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        R : α → α → Prop
        inst✝¹ : DecidableRel R
        inst✝ : IsEquiv α (HasCompl.compl R)
        l₁ : List α
        b : α
        l₂ : List α
        a : α
        hl : l₁.Sublist l₂
        hl₁ : List.Chain' R (List.cons a l₁)
        hab : Not (R a b)
        ⊢ LE.le (List.cons a l₁).length (List.destutter R (List.cons a (List.cons b l₂ …
      -/
    · simpa [destutter_cons_cons, hab] using hl₁.length_le_length_destutter (hl.cons₂ _)
      /-
        🎉 no goals
      -/
  -- `l₁ := a :: b :: l₁`, `l₂ := a :: b :: l₂`
  | _, _, .cons₂ a <| .cons₂ (l₁ := l₁) (l₂ := l₂) b hl, hl₁ => by
    simpa [destutter_cons_cons, rel_of_chain_cons hl₁]
      using hl₁.tail.length_le_length_destutter (hl.cons₂ _)


/-- `destutter` of `≠` gives a list of maximal length over any chain.

In other words, `l.destutter (· ≠ ·)` is a `≠`-chain sublist of `l`, and is at least as long as any
other `≠`-chain sublist. -/
lemma Chain'.length_le_length_destutter_ne [DecidableEq α] (hl : l₁ <+ l₂)
    (hl₁ : l₁.Chain' (· ≠ ·)) : l₁.length ≤ (l₂.destutter (· ≠ ·)).length :=
  hl₁.length_le_length_destutter hl


