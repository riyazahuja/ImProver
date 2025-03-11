/-- Property that an element `x : α` of `l : List α` can be found in the list more than once. -/
inductive Duplicate (x : α) : List α → Prop
  | cons_mem {l : List α} : x ∈ l → Duplicate x (x :: l)
  | cons_duplicate {y : α} {l : List α} : Duplicate x l → Duplicate x (y :: l)


local infixl:50 " ∈+ " => List.Duplicate


theorem Mem.duplicate_cons_self (h : x ∈ l) : x ∈+ x :: l :=
  Duplicate.cons_mem h


theorem Duplicate.duplicate_cons (h : x ∈+ l) (y : α) : x ∈+ y :: l :=
  Duplicate.cons_duplicate h


theorem Duplicate.mem (h : x ∈+ l) : x ∈ l := by
  induction h with
  | cons_mem => exact mem_cons_self _ _
  | cons_duplicate _ hm => exact mem_cons_of_mem _ hm


theorem Duplicate.mem_cons_self (h : x ∈+ x :: l) : x ∈ l := by
  /-
    α : Type u_1
    l : List α
    x : α
    h : List.Duplicate x (List.cons x l)
    ⊢ Membership.mem l x
  -/
  cases' h with _ h _ _ h
    /-
      case cons_mem
      α : Type u_1
      l : List α
      x : α
      h : Membership.mem l x
      ⊢ Membership.mem l x
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case cons_duplicate
      α : Type u_1
      l : List α
      x : α
      h : List.Duplicate x l
      ⊢ Membership.mem l x
    -/
  · exact h.mem
    /-
      🎉 no goals
    -/


@[simp]
theorem duplicate_cons_self_iff : x ∈+ x :: l ↔ x ∈ l :=
  ⟨Duplicate.mem_cons_self, Mem.duplicate_cons_self⟩


theorem Duplicate.ne_nil (h : x ∈+ l) : l ≠ [] := fun H => (mem_nil_iff x).mp (H ▸ h.mem)


@[simp]
theorem not_duplicate_nil (x : α) : ¬x ∈+ [] := fun H => H.ne_nil rfl


theorem Duplicate.ne_singleton (h : x ∈+ l) (y : α) : l ≠ [y] := by
  /-
    α : Type u_1
    l : List α
    x : α
    h : List.Duplicate x l
    y : α
    ⊢ Ne l (List.cons y List.nil)
  -/
  induction' h with l' h z l' h _
    /-
      case cons_mem
      α : Type u_1
      l : List α
      x y : α
      l' : List α
      h : Membership.mem l' x
      ⊢ Ne (List.cons x l') (List.cons y List.nil)
    -/
  · simp [ne_nil_of_mem h]
    /-
      🎉 no goals
    -/
    /-
      case cons_duplicate
      α : Type u_1
      l : List α
      x y z : α
      l' : List α
      h : List.Duplicate x l'
      a_ih✝ : Ne l' (List.cons y List.nil)
      ⊢ Ne (List.cons z l') (List.cons y List.nil)
    -/
  · simp [ne_nil_of_mem h.mem]
    /-
      🎉 no goals
    -/


@[simp]
theorem not_duplicate_singleton (x y : α) : ¬x ∈+ [y] := fun H => H.ne_singleton _ rfl


theorem Duplicate.elim_nil (h : x ∈+ []) : False :=
  not_duplicate_nil x h


theorem Duplicate.elim_singleton {y : α} (h : x ∈+ [y]) : False :=
  not_duplicate_singleton x y h


theorem duplicate_cons_iff {y : α} : x ∈+ y :: l ↔ y = x ∧ x ∈ l ∨ x ∈+ l := by
  /-
    α : Type u_1
    l : List α
    x y : α
    ⊢ Iff (List.Duplicate x (List.cons y l)) (Or (And (Eq y x) (Membership.mem l x …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      l : List α
      x y : α
      h : List.Duplicate x (List.cons y l)
      ⊢ Or (And (Eq y x) (Membership.mem l x)) (List.Duplicate x l)
    -/
  · cases' h with _ hm _ _ hm
      /-
        case refine_1.cons_mem
        α : Type u_1
        l : List α
        x : α
        hm : Membership.mem l x
        ⊢ Or (And (Eq x x) (Membership.mem l x)) (List.Duplicate x l)
      -/
    · exact Or.inl ⟨rfl, hm⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.cons_duplicate
        α : Type u_1
        l : List α
        x y : α
        hm : List.Duplicate x l
        ⊢ Or (And (Eq y x) (Membership.mem l x)) (List.Duplicate x l)
      -/
    · exact Or.inr hm
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      l : List α
      x y : α
      h : Or (And (Eq y x) (Membership.mem l x)) (List.Duplicate x l)
      ⊢ List.Duplicate x (List.cons y l)
    -/
  · rcases h with (⟨rfl | h⟩ | h)
      /-
        case refine_2.inl.intro.refl
        α : Type u_1
        l : List α
        x : α
        right✝ : Membership.mem l x
        ⊢ List.Duplicate x (List.cons x l)
      -/
    · simpa
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        α : Type u_1
        l : List α
        x y : α
        h : List.Duplicate x l
        ⊢ List.Duplicate x (List.cons y l)
      -/
    · exact h.cons_duplicate
      /-
        🎉 no goals
      -/


theorem Duplicate.of_duplicate_cons {y : α} (h : x ∈+ y :: l) (hx : x ≠ y) : x ∈+ l := by
  /-
    α : Type u_1
    l : List α
    x y : α
    h : List.Duplicate x (List.cons y l)
    hx : Ne x y
    ⊢ List.Duplicate x l
  -/
  simpa [duplicate_cons_iff, hx.symm] using h
  /-
    🎉 no goals
  -/


theorem duplicate_cons_iff_of_ne {y : α} (hne : x ≠ y) : x ∈+ y :: l ↔ x ∈+ l := by
  /-
    α : Type u_1
    l : List α
    x y : α
    hne : Ne x y
    ⊢ Iff (List.Duplicate x (List.cons y l)) (List.Duplicate x l)
  -/
  simp [duplicate_cons_iff, hne.symm]
  /-
    🎉 no goals
  -/


theorem Duplicate.mono_sublist {l' : List α} (hx : x ∈+ l) (h : l <+ l') : x ∈+ l' := by
  /-
    α : Type u_1
    l : List α
    x : α
    l' : List α
    hx : List.Duplicate x l
    h : l.Sublist l'
    ⊢ List.Duplicate x l'
  -/
  induction' h with l₁ l₂ y _ IH l₁ l₂ y h IH
    /-
      case slnil
      α : Type u_1
      l : List α
      x : α
      l' : List α
      hx : List.Duplicate x List.nil
      ⊢ List.Duplicate x List.nil
    -/
  · exact hx
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      l : List α
      x : α
      l' l₁ l₂ : List α
      y : α
      a✝ : l₁.Sublist l₂
      IH : List.Duplicate x l₁ → List.Duplicate x l₂
      hx : List.Duplicate x l₁
      ⊢ List.Duplicate x (List.cons y l₂)
    -/
  · exact (IH hx).duplicate_cons _
    /-
      🎉 no goals
    -/
    /-
      case cons₂
      α : Type u_1
      l : List α
      x : α
      l' l₁ l₂ : List α
      y : α
      h : l₁.Sublist l₂
      IH : List.Duplicate x l₁ → List.Duplicate x l₂
      hx : List.Duplicate x (List.cons y l₁)
      ⊢ List.Duplicate x (List.cons y l₂)
    -/
  · rw [duplicate_cons_iff] at hx ⊢
    /-
      case cons₂
      α : Type u_1
      l : List α
      x : α
      l' l₁ l₂ : List α
      y : α
      h : l₁.Sublist l₂
      IH : List.Duplicate x l₁ → List.Duplicate x l₂
      hx : Or (And (Eq y x) (Membership.mem l₁ x)) (List.Duplicate x l₁)
      ⊢ Or (And (Eq y x) (Membership.mem l₂ x)) (List.Duplicate x l₂)
    -/
    rcases hx with (⟨rfl, hx⟩ | hx)
      /-
        case cons₂.inl.intro
        α : Type u_1
        l l' l₁ l₂ : List α
        y : α
        h : l₁.Sublist l₂
        IH : List.Duplicate y l₁ → List.Duplicate y l₂
        hx : Membership.mem l₁ y
        ⊢ Or (And (Eq y y) (Membership.mem l₂ y)) (List.Duplicate y l₂)
      -/
    · simp [h.subset hx]
      /-
        🎉 no goals
      -/
      /-
        case cons₂.inr
        α : Type u_1
        l : List α
        x : α
        l' l₁ l₂ : List α
        y : α
        h : l₁.Sublist l₂
        IH : List.Duplicate x l₁ → List.Duplicate x l₂
        hx : List.Duplicate x l₁
        ⊢ Or (And (Eq y x) (Membership.mem l₂ x)) (List.Duplicate x l₂)
      -/
    · simp [IH hx]
      /-
        🎉 no goals
      -/


/-- The contrapositive of `List.nodup_iff_sublist`. -/
theorem duplicate_iff_sublist : x ∈+ l ↔ [x, x] <+ l := by
  /-
    α : Type u_1
    l : List α
    x : α
    ⊢ Iff (List.Duplicate x l) ((List.cons x (List.cons x List.nil)).Sublist l)
  -/
  induction' l with y l IH
    /-
      case nil
      α : Type u_1
      l : List α
      x : α
      ⊢ Iff (List.Duplicate x List.nil) ((List.cons x (List.cons x List.nil)).Sublis …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      l✝ : List α
      x y : α
      l : List α
      IH : Iff (List.Duplicate x l) ((List.cons x (List.cons x List.nil)).Sublist l)
      ⊢ Iff (List.Duplicate x (List.cons y l)) ((List.cons x (List.cons x List.nil)) …
    -/
  · by_cases hx : x = y
      /-
        case pos
        α : Type u_1
        l✝ : List α
        x y : α
        l : List α
        IH : Iff (List.Duplicate x l) ((List.cons x (List.cons x List.nil)).Sublist l)
        hx : Eq x y
        ⊢ Iff (List.Duplicate x (List.cons y l)) ((List.cons x (List.cons x List.nil)) …
      -/
    · simp [hx, cons_sublist_cons, singleton_sublist]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        l✝ : List α
        x y : α
        l : List α
        IH : Iff (List.Duplicate x l) ((List.cons x (List.cons x List.nil)).Sublist l)
        hx : Not (Eq x y)
        ⊢ Iff (List.Duplicate x (List.cons y l)) ((List.cons x (List.cons x List.nil)) …
      -/
    · rw [duplicate_cons_iff_of_ne hx, IH]
      /-
        case neg
        α : Type u_1
        l✝ : List α
        x y : α
        l : List α
        IH : Iff (List.Duplicate x l) ((List.cons x (List.cons x List.nil)).Sublist l)
        hx : Not (Eq x y)
        ⊢ Iff ((List.cons x (List.cons x List.nil)).Sublist l) ((List.cons x (List.con …
      -/
      refine ⟨sublist_cons_of_sublist y, fun h => ?_⟩
      /-
        case neg
        α : Type u_1
        l✝ : List α
        x y : α
        l : List α
        IH : Iff (List.Duplicate x l) ((List.cons x (List.cons x List.nil)).Sublist l)
        hx : Not (Eq x y)
        h : (List.cons x (List.cons x List.nil)).Sublist (List.cons y l)
        ⊢ (List.cons x (List.cons x List.nil)).Sublist l
      -/
      cases h
        /-
          case neg.cons
          α : Type u_1
          l✝ : List α
          x y : α
          l : List α
          IH : Iff (List.Duplicate x l) ((List.cons x (List.cons x List.nil)).Sublist l)
          hx : Not (Eq x y)
          a✝ : (List.cons x (List.cons x List.nil)).Sublist l
          ⊢ (List.cons x (List.cons x List.nil)).Sublist l
        -/
      · assumption
        /-
          🎉 no goals
        -/
        /-
          case neg.cons₂
          α : Type u_1
          l✝ : List α
          x : α
          l : List α
          IH : Iff (List.Duplicate x l) ((List.cons x (List.cons x List.nil)).Sublist l)
          hx : Not (Eq x x)
          a✝ : (List.cons x List.nil).Sublist l
          ⊢ (List.cons x (List.cons x List.nil)).Sublist l
        -/
      · contradiction
        /-
          🎉 no goals
        -/


theorem nodup_iff_forall_not_duplicate : Nodup l ↔ ∀ x : α, ¬x ∈+ l := by
  /-
    α : Type u_1
    l : List α
    ⊢ Iff l.Nodup (∀ (x : α), Not (List.Duplicate x l))
  -/
  simp_rw [nodup_iff_sublist, duplicate_iff_sublist]
  /-
    🎉 no goals
  -/


theorem exists_duplicate_iff_not_nodup : (∃ x : α, x ∈+ l) ↔ ¬Nodup l := by
  /-
    α : Type u_1
    l : List α
    ⊢ Iff (Exists fun x => List.Duplicate x l) (Not l.Nodup)
  -/
  simp [nodup_iff_forall_not_duplicate]
  /-
    🎉 no goals
  -/


theorem Duplicate.not_nodup (h : x ∈+ l) : ¬Nodup l := fun H =>
  nodup_iff_forall_not_duplicate.mp H _ h


theorem duplicate_iff_two_le_count [DecidableEq α] : x ∈+ l ↔ 2 ≤ count x l := by
  /-
    α : Type u_1
    l : List α
    x : α
    inst✝ : DecidableEq α
    ⊢ Iff (List.Duplicate x l) (LE.le 2 (List.count x l))
  -/
  simp [replicate_succ, duplicate_iff_sublist, le_count_iff_replicate_sublist]
  /-
    🎉 no goals
  -/


instance decidableDuplicate [DecidableEq α] (x : α) : ∀ l : List α, Decidable (x ∈+ l)
  | [] => isFalse (not_duplicate_nil x)
  | y :: l =>
    match decidableDuplicate x l with
    | isTrue h => isTrue (h.duplicate_cons y)
    | isFalse h =>
      if hx : y = x ∧ x ∈ l then isTrue (hx.left.symm ▸ List.Mem.duplicate_cons_self hx.right)
                       /-
                         α : Type u_1
                         l✝ : List α
                         x✝ : α
                         inst✝ : DecidableEq α
                         x y : α
                         l : List α
                         h : Not (List.Duplicate x l)
                         hx : Not (And (Eq y x) (Membership.mem l x))
                         ⊢ Not (List.Duplicate x (List.cons y l))
                       -/
      else isFalse (by simpa [duplicate_cons_iff, h] using hx)
                       /-
                         🎉 no goals
                       -/


