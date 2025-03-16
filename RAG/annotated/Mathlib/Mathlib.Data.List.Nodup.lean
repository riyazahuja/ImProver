protected theorem Pairwise.nodup {l : List α} {r : α → α → Prop} [IsIrrefl α r] (h : Pairwise r l) :
    Nodup l :=
  h.imp ne_of_irrefl


theorem rel_nodup {r : α → β → Prop} (hr : Relator.BiUnique r) : (Forall₂ r ⇒ (· ↔ ·)) Nodup Nodup
                            /-
                              α : Type u
                              β : Type v
                              r : α → β → Prop
                              hr : Relator.BiUnique r
                              ⊢ (fun x1 x2 => Iff x1 x2) List.nil.Nodup List.nil.Nodup
                            -/
  | _, _, Forall₂.nil => by simp only [nodup_nil]
                            /-
                              🎉 no goals
                            -/
  | _, _, Forall₂.cons hab h => by
    simpa only [nodup_cons] using
      Relator.rel_and (Relator.rel_not (rel_mem hr hab h)) (rel_nodup hr h)


protected theorem Nodup.cons (ha : a ∉ l) (hl : Nodup l) : Nodup (a :: l) :=
  nodup_cons.2 ⟨ha, hl⟩


theorem nodup_singleton (a : α) : Nodup [a] :=
  pairwise_singleton _ _


theorem Nodup.of_cons (h : Nodup (a :: l)) : Nodup l :=
  (nodup_cons.1 h).2


theorem Nodup.not_mem (h : (a :: l).Nodup) : a ∉ l :=
  (nodup_cons.1 h).1


theorem not_nodup_cons_of_mem : a ∈ l → ¬Nodup (a :: l) :=
  imp_not_comm.1 Nodup.not_mem



theorem not_nodup_pair (a : α) : ¬Nodup [a, a] :=
  not_nodup_cons_of_mem <| mem_singleton_self _


theorem nodup_iff_sublist {l : List α} : Nodup l ↔ ∀ a, ¬[a, a] <+ l :=
  ⟨fun d a h => not_nodup_pair a (d.sublist h),
    by
      /-
        α : Type u
        l : List α
        ⊢ (∀ (a : α), Not ((List.cons a (List.cons a List.nil)).Sublist l)) → l.Nodup
      -/
      induction' l with a l IH <;> intro h; · exact nodup_nil
                                              /-
                                                🎉 no goals
                                              -/
      exact (IH fun a s => h a <| sublist_cons_of_sublist _ s).cons fun al =>
        h a <| (singleton_sublist.2 al).cons_cons _⟩


theorem nodup_iff_injective_getElem {l : List α} :
    Nodup l ↔ Function.Injective (fun i : Fin l.length => l[i.1]) :=
  pairwise_iff_getElem.trans
    ⟨fun h i j hg => by
      /-
        α : Type u
        l : List α
        h : ∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j …
        i j : Fin l.length
        hg : Eq ((fun i => GetElem.getElem l ↑i ⋯) i) ((fun i => GetElem.getElem l ↑i  …
        ⊢ Eq i j
      -/
      cases' i with i hi; cases' j with j hj
      /-
        case mk.mk
        α : Type u
        l : List α
        h : ∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j …
        i : Nat
        hi : LT.lt i l.length
        j : Nat
        hj : LT.lt j l.length
        hg : Eq ((fun i => GetElem.getElem l ↑i ⋯) ⟨i, hi⟩) ((fun i => GetElem.getElem …
        ⊢ Eq ⟨i, hi⟩ ⟨j, hj⟩
      -/
      rcases lt_trichotomy i j with (hij | rfl | hji)
        /-
          case mk.mk.inl
          α : Type u
          l : List α
          h : ∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j …
          i : Nat
          hi : LT.lt i l.length
          j : Nat
          hj : LT.lt j l.length
          hg : Eq ((fun i => GetElem.getElem l ↑i ⋯) ⟨i, hi⟩) ((fun i => GetElem.getElem …
          hij : LT.lt i j
          ⊢ Eq ⟨i, hi⟩ ⟨j, hj⟩
        -/
      · exact (h i j hi hj hij hg).elim
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.inr.inl
          α : Type u
          l : List α
          h : ∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j …
          i : Nat
          hi hj : LT.lt i l.length
          hg : Eq ((fun i => GetElem.getElem l ↑i ⋯) ⟨i, hi⟩) ((fun i => GetElem.getElem …
          ⊢ Eq ⟨i, hi⟩ ⟨i, hj⟩
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.inr.inr
          α : Type u
          l : List α
          h : ∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j …
          i : Nat
          hi : LT.lt i l.length
          j : Nat
          hj : LT.lt j l.length
          hg : Eq ((fun i => GetElem.getElem l ↑i ⋯) ⟨i, hi⟩) ((fun i => GetElem.getElem …
          hji : LT.lt j i
          ⊢ Eq ⟨i, hi⟩ ⟨j, hj⟩
        -/
      · exact (h j i hj hi hji hg.symm).elim,
        /-
          🎉 no goals
        -/
      fun hinj i j hi hj hij h => Nat.ne_of_lt hij (Fin.val_eq_of_eq (@hinj ⟨i, hi⟩ ⟨j, hj⟩ h))⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10756): new theorem

theorem nodup_iff_injective_get {l : List α} :
    Nodup l ↔ Function.Injective l.get := by
  /-
    α : Type u
    l : List α
    ⊢ Iff l.Nodup (Function.Injective l.get)
  -/
  rw [nodup_iff_injective_getElem]
  /-
    α : Type u
    l : List α
    ⊢ Iff (Function.Injective fun i => GetElem.getElem l ↑i ⋯) (Function.Injective …
  -/
  change _ ↔ Injective (fun i => l.get i)
  /-
    α : Type u
    l : List α
    ⊢ Iff (Function.Injective fun i => GetElem.getElem l ↑i ⋯) (Function.Injective …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Nodup.get_inj_iff {l : List α} (h : Nodup l) {i j : Fin l.length} :
    l.get i = l.get j ↔ i = j :=
  (nodup_iff_injective_get.1 h).eq_iff


theorem Nodup.getElem_inj_iff {l : List α} (h : Nodup l)
    {i : Nat} {hi : i < l.length} {j : Nat} {hj : j < l.length} :
    l[i] = l[j] ↔ i = j := by
  /-
    α : Type u
    l : List α
    h : l.Nodup
    i : Nat
    hi : LT.lt i l.length
    j : Nat
    hj : LT.lt j l.length
    ⊢ Iff (Eq (GetElem.getElem l i hi) (GetElem.getElem l j hj)) (Eq i j)
  -/
  have := @Nodup.get_inj_iff _ _ h ⟨i, hi⟩ ⟨j, hj⟩
  /-
    α : Type u
    l : List α
    h : l.Nodup
    i : Nat
    hi : LT.lt i l.length
    j : Nat
    hj : LT.lt j l.length
    this : Iff (Eq (l.get ⟨i, hi⟩) (l.get ⟨j, hj⟩)) (Eq ⟨i, hi⟩ ⟨j, hj⟩)
    ⊢ Iff (Eq (GetElem.getElem l i hi) (GetElem.getElem l j hj)) (Eq i j)
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem nodup_iff_getElem?_ne_getElem? {l : List α} :
    l.Nodup ↔ ∀ i j : ℕ, i < j → j < l.length → l[i]? ≠ l[j]? := by
  /-
    α : Type u
    l : List α
    ⊢ Iff l.Nodup (∀ (i j : Nat), LT.lt i j → LT.lt j l.length → Ne (GetElem?.getE …
  -/
  rw [Nodup, pairwise_iff_getElem]
  /-
    α : Type u
    l : List α
    ⊢ Iff (∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt  …
  -/
  constructor
    /-
      case mp
      α : Type u
      l : List α
      ⊢ (∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j  …
    -/
  · intro h i j hij hj
    /-
      case mp
      α : Type u
      l : List α
      h : ∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j …
      i j : Nat
      hij : LT.lt i j
      hj : LT.lt j l.length
      ⊢ Ne (GetElem?.getElem? l i) (GetElem?.getElem? l j)
    -/
    rw [getElem?_eq_getElem (lt_trans hij hj), getElem?_eq_getElem hj, Ne, Option.some_inj]
    /-
      case mp
      α : Type u
      l : List α
      h : ∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j …
      i j : Nat
      hij : LT.lt i j
      hj : LT.lt j l.length
      ⊢ Not (Eq (GetElem.getElem l i ⋯) (GetElem.getElem l j hj))
    -/
    exact h _ _ (by omega) hj hij
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      l : List α
      ⊢ (∀ (i j : Nat), LT.lt i j → LT.lt j l.length → Ne (GetElem?.getElem? l i) (G …
    -/
  · intro h i j hi hj hij
    /-
      case mpr
      α : Type u
      l : List α
      h : ∀ (i j : Nat), LT.lt i j → LT.lt j l.length → Ne (GetElem?.getElem? l i) ( …
      i j : Nat
      hi : LT.lt i l.length
      hj : LT.lt j l.length
      hij : LT.lt i j
      ⊢ Ne (GetElem.getElem l i hi) (GetElem.getElem l j hj)
    -/
    rw [Ne, ← Option.some_inj, ← getElem?_eq_getElem, ← getElem?_eq_getElem]
    /-
      case mpr
      α : Type u
      l : List α
      h : ∀ (i j : Nat), LT.lt i j → LT.lt j l.length → Ne (GetElem?.getElem? l i) ( …
      i j : Nat
      hi : LT.lt i l.length
      hj : LT.lt j l.length
      hij : LT.lt i j
      ⊢ Not (Eq (GetElem?.getElem? l i) (GetElem?.getElem? l j))
    -/
    exact h i j hij hj
    /-
      🎉 no goals
    -/


theorem nodup_iff_get?_ne_get? {l : List α} :
    l.Nodup ↔ ∀ i j : ℕ, i < j → j < l.length → l.get? i ≠ l.get? j := by
  /-
    α : Type u
    l : List α
    ⊢ Iff l.Nodup (∀ (i j : Nat), LT.lt i j → LT.lt j l.length → Ne (l.get? i) (l. …
  -/
  simp [nodup_iff_getElem?_ne_getElem?]
  /-
    🎉 no goals
  -/


theorem Nodup.ne_singleton_iff {l : List α} (h : Nodup l) (x : α) :
    l ≠ [x] ↔ l = [] ∨ ∃ y ∈ l, y ≠ x := by
  /-
    α : Type u
    l : List α
    h : l.Nodup
    x : α
    ⊢ Iff (Ne l (List.cons x List.nil)) (Or (Eq l List.nil) (Exists fun y => And ( …
  -/
  induction' l with hd tl hl
    /-
      case nil
      α : Type u
      x : α
      h : List.nil.Nodup
      ⊢ Iff (Ne List.nil (List.cons x List.nil)) (Or (Eq List.nil List.nil) (Exists  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      x hd : α
      tl : List α
      hl : tl.Nodup → Iff (Ne tl (List.cons x List.nil)) (Or (Eq tl List.nil) (Exist …
      h : (List.cons hd tl).Nodup
      ⊢ Iff (Ne (List.cons hd tl) (List.cons x List.nil)) (Or (Eq (List.cons hd tl)  …
    -/
  · specialize hl h.of_cons
    /-
      case cons
      α : Type u
      x hd : α
      tl : List α
      h : (List.cons hd tl).Nodup
      hl : Iff (Ne tl (List.cons x List.nil)) (Or (Eq tl List.nil) (Exists fun y =>  …
      ⊢ Iff (Ne (List.cons hd tl) (List.cons x List.nil)) (Or (Eq (List.cons hd tl)  …
    -/
    by_cases hx : tl = [x]
      /-
        case pos
        α : Type u
        x hd : α
        tl : List α
        h : (List.cons hd tl).Nodup
        hl : Iff (Ne tl (List.cons x List.nil)) (Or (Eq tl List.nil) (Exists fun y =>  …
        hx : Eq tl (List.cons x List.nil)
        ⊢ Iff (Ne (List.cons hd tl) (List.cons x List.nil)) (Or (Eq (List.cons hd tl)  …
      -/
    · simpa [hx, and_comm, and_or_left] using h
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        x hd : α
        tl : List α
        h : (List.cons hd tl).Nodup
        hl : Iff (Ne tl (List.cons x List.nil)) (Or (Eq tl List.nil) (Exists fun y =>  …
        hx : Not (Eq tl (List.cons x List.nil))
        ⊢ Iff (Ne (List.cons hd tl) (List.cons x List.nil)) (Or (Eq (List.cons hd tl)  …
      -/
    · rw [← Ne, hl] at hx
      /-
        case neg
        α : Type u
        x hd : α
        tl : List α
        h : (List.cons hd tl).Nodup
        hl : Iff (Ne tl (List.cons x List.nil)) (Or (Eq tl List.nil) (Exists fun y =>  …
        hx : Or (Eq tl List.nil) (Exists fun y => And (Membership.mem tl y) (Ne y x))
        ⊢ Iff (Ne (List.cons hd tl) (List.cons x List.nil)) (Or (Eq (List.cons hd tl)  …
      -/
      rcases hx with (rfl | ⟨y, hy, hx⟩)
        /-
          case neg.inl
          α : Type u
          x hd : α
          h : (List.cons hd List.nil).Nodup
          hl : Iff (Ne List.nil (List.cons x List.nil)) (Or (Eq List.nil List.nil) (Exis …
          ⊢ Iff (Ne (List.cons hd List.nil) (List.cons x List.nil)) (Or (Eq (List.cons h …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case neg.inr.intro.intro
          α : Type u
          x hd : α
          tl : List α
          h : (List.cons hd tl).Nodup
          hl : Iff (Ne tl (List.cons x List.nil)) (Or (Eq tl List.nil) (Exists fun y =>  …
          y : α
          hy : Membership.mem tl y
          hx : Ne y x
          ⊢ Iff (Ne (List.cons hd tl) (List.cons x List.nil)) (Or (Eq (List.cons hd tl)  …
        -/
      · suffices ∃ y ∈ hd :: tl, y ≠ x by simpa [ne_nil_of_mem hy]
        /-
          case neg.inr.intro.intro
          α : Type u
          x hd : α
          tl : List α
          h : (List.cons hd tl).Nodup
          hl : Iff (Ne tl (List.cons x List.nil)) (Or (Eq tl List.nil) (Exists fun y =>  …
          y : α
          hy : Membership.mem tl y
          hx : Ne y x
          ⊢ Exists fun y => And (Membership.mem (List.cons hd tl) y) (Ne y x)
        -/
        exact ⟨y, mem_cons_of_mem _ hy, hx⟩
        /-
          🎉 no goals
        -/


theorem not_nodup_of_get_eq_of_ne (xs : List α) (n m : Fin xs.length)
    (h : xs.get n = xs.get m) (hne : n ≠ m) : ¬Nodup xs := by
  /-
    α : Type u
    xs : List α
    n m : Fin xs.length
    h : Eq (xs.get n) (xs.get m)
    hne : Ne n m
    ⊢ Not xs.Nodup
  -/
  rw [nodup_iff_injective_get]
  /-
    α : Type u
    xs : List α
    n m : Fin xs.length
    h : Eq (xs.get n) (xs.get m)
    hne : Ne n m
    ⊢ Not (Function.Injective xs.get)
  -/
  exact fun hinj => hne (hinj h)
  /-
    🎉 no goals
  -/


theorem indexOf_getElem [DecidableEq α] {l : List α} (H : Nodup l) (i : Nat) (h : i < l.length) :
    indexOf l[i] l = i :=
  suffices (⟨indexOf l[i] l, indexOf_lt_length.2 (getElem_mem _)⟩ : Fin l.length) = ⟨i, h⟩
    from Fin.val_eq_of_eq this
                                  /-
                                    α : Type u
                                    inst✝ : DecidableEq α
                                    l : List α
                                    H : l.Nodup
                                    i : Nat
                                    h : LT.lt i l.length
                                    ⊢ Eq (l.get ⟨List.indexOf (GetElem.getElem l i h) l, ⋯⟩) (l.get ⟨i, h⟩)
                                  -/
  nodup_iff_injective_get.1 H (by simp)
                                  /-
                                    🎉 no goals
                                  -/

-- This is incorrectly named and should be `indexOf_get`;
-- this already exists, so will require a deprecation dance.

theorem get_indexOf [DecidableEq α] {l : List α} (H : Nodup l) (i : Fin l.length) :
    indexOf (get l i) l = i := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    l : List α
    H : l.Nodup
    i : Fin l.length
    ⊢ Eq (List.indexOf (l.get i) l) ↑i
  -/
  simp [indexOf_getElem, H]
  /-
    🎉 no goals
  -/


theorem nodup_iff_count_le_one [DecidableEq α] {l : List α} : Nodup l ↔ ∀ a, count a l ≤ 1 :=
  nodup_iff_sublist.trans <|
    forall_congr' fun a =>
      have : replicate 2 a <+ l ↔ 1 < count a l := (le_count_iff_replicate_sublist ..).symm
      (not_congr this).trans not_lt


theorem nodup_iff_count_eq_one [DecidableEq α] : Nodup l ↔ ∀ a ∈ l, count a l = 1 :=
  nodup_iff_count_le_one.trans <| forall_congr' fun _ =>
    ⟨fun H h => H.antisymm (count_pos_iff.mpr h),
     fun H => if h : _ then (H h).le else (count_eq_zero.mpr h).trans_le (Nat.zero_le 1)⟩



@[simp]
theorem count_eq_one_of_mem [DecidableEq α] {a : α} {l : List α} (d : Nodup l) (h : a ∈ l) :
    count a l = 1 :=
  _root_.le_antisymm (nodup_iff_count_le_one.1 d a) (Nat.succ_le_of_lt (count_pos_iff.2 h))


theorem count_eq_of_nodup [DecidableEq α] {a : α} {l : List α} (d : Nodup l) :
    count a l = if a ∈ l then 1 else 0 := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a : α
    l : List α
    d : l.Nodup
    ⊢ Eq (List.count a l) (ite (Membership.mem l a) 1 0)
  -/
  split_ifs with h
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      a : α
      l : List α
      d : l.Nodup
      h : Membership.mem l a
      ⊢ Eq (List.count a l) 1
    -/
  · exact count_eq_one_of_mem d h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      a : α
      l : List α
      d : l.Nodup
      h : Not (Membership.mem l a)
      ⊢ Eq (List.count a l) 0
    -/
  · exact count_eq_zero_of_not_mem h
    /-
      🎉 no goals
    -/


theorem Nodup.of_append_left : Nodup (l₁ ++ l₂) → Nodup l₁ :=
  Nodup.sublist (sublist_append_left l₁ l₂)


theorem Nodup.of_append_right : Nodup (l₁ ++ l₂) → Nodup l₂ :=
  Nodup.sublist (sublist_append_right l₁ l₂)


theorem nodup_append {l₁ l₂ : List α} :
    Nodup (l₁ ++ l₂) ↔ Nodup l₁ ∧ Nodup l₂ ∧ Disjoint l₁ l₂ := by
  /-
    α : Type u
    l₁ l₂ : List α
    ⊢ Iff (HAppend.hAppend l₁ l₂).Nodup (And l₁.Nodup (And l₂.Nodup (l₁.Disjoint l …
  -/
  simp only [Nodup, pairwise_append, disjoint_iff_ne]
  /-
    🎉 no goals
  -/


theorem disjoint_of_nodup_append {l₁ l₂ : List α} (d : Nodup (l₁ ++ l₂)) : Disjoint l₁ l₂ :=
  (nodup_append.1 d).2.2


theorem Nodup.append (d₁ : Nodup l₁) (d₂ : Nodup l₂) (dj : Disjoint l₁ l₂) : Nodup (l₁ ++ l₂) :=
  nodup_append.2 ⟨d₁, d₂, dj⟩


theorem nodup_append_comm {l₁ l₂ : List α} : Nodup (l₁ ++ l₂) ↔ Nodup (l₂ ++ l₁) := by
  /-
    α : Type u
    l₁ l₂ : List α
    ⊢ Iff (HAppend.hAppend l₁ l₂).Nodup (HAppend.hAppend l₂ l₁).Nodup
  -/
  simp only [nodup_append, and_left_comm, disjoint_comm]
  /-
    🎉 no goals
  -/


theorem nodup_middle {a : α} {l₁ l₂ : List α} :
    Nodup (l₁ ++ a :: l₂) ↔ Nodup (a :: (l₁ ++ l₂)) := by
  simp only [nodup_append, not_or, and_left_comm, and_assoc, nodup_cons, mem_append,
    disjoint_cons_right]


theorem Nodup.of_map (f : α → β) {l : List α} : Nodup (map f l) → Nodup l :=
  (Pairwise.of_map f) fun _ _ => mt <| congr_arg f


theorem Nodup.map_on {f : α → β} (H : ∀ x ∈ l, ∀ y ∈ l, f x = f y → x = y) (d : Nodup l) :
    (map f l).Nodup :=
  Pairwise.map _ (fun a b ⟨ma, mb, n⟩ e => n (H a ma b mb e)) (Pairwise.and_mem.1 d)


theorem inj_on_of_nodup_map {f : α → β} {l : List α} (d : Nodup (map f l)) :
    ∀ ⦃x⦄, x ∈ l → ∀ ⦃y⦄, y ∈ l → f x = f y → x = y := by
  /-
    α : Type u
    β : Type v
    f : α → β
    l : List α
    d : (List.map f l).Nodup
    ⊢ ∀ ⦃x : α⦄, Membership.mem l x → ∀ ⦃y : α⦄, Membership.mem l y → Eq (f x) (f  …
  -/
  induction' l with hd tl ih
    /-
      case nil
      α : Type u
      β : Type v
      f : α → β
      d : (List.map f List.nil).Nodup
      ⊢ ∀ ⦃x : α⦄, Membership.mem List.nil x → ∀ ⦃y : α⦄, Membership.mem List.nil y  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      f : α → β
      hd : α
      tl : List α
      ih : (List.map f tl).Nodup → ∀ ⦃x : α⦄, Membership.mem tl x → ∀ ⦃y : α⦄, Membe …
      d : (List.map f (List.cons hd tl)).Nodup
      ⊢ ∀ ⦃x : α⦄, Membership.mem (List.cons hd tl) x → ∀ ⦃y : α⦄, Membership.mem (L …
    -/
  · simp only [map, nodup_cons, mem_map, not_exists, not_and, ← Ne.eq_def] at d
    /-
      case cons
      α : Type u
      β : Type v
      f : α → β
      hd : α
      tl : List α
      ih : (List.map f tl).Nodup → ∀ ⦃x : α⦄, Membership.mem tl x → ∀ ⦃y : α⦄, Membe …
      d : And (∀ (x : α), Membership.mem tl x → Ne (f x) (f hd)) (List.map f tl).Nodup
      ⊢ ∀ ⦃x : α⦄, Membership.mem (List.cons hd tl) x → ∀ ⦃y : α⦄, Membership.mem (L …
    -/
    simp only [mem_cons]
    /-
      case cons
      α : Type u
      β : Type v
      f : α → β
      hd : α
      tl : List α
      ih : (List.map f tl).Nodup → ∀ ⦃x : α⦄, Membership.mem tl x → ∀ ⦃y : α⦄, Membe …
      d : And (∀ (x : α), Membership.mem tl x → Ne (f x) (f hd)) (List.map f tl).Nodup
      ⊢ ∀ ⦃x : α⦄, Or (Eq x hd) (Membership.mem tl x) → ∀ ⦃y : α⦄, Or (Eq y hd) (Mem …
    -/
    rintro _ (rfl | h₁) _ (rfl | h₂) h₃
      /-
        case cons.inl.inl
        α : Type u
        β : Type v
        f : α → β
        tl : List α
        ih : (List.map f tl).Nodup → ∀ ⦃x : α⦄, Membership.mem tl x → ∀ ⦃y : α⦄, Membe …
        y✝ : α
        d : And (∀ (x : α), Membership.mem tl x → Ne (f x) (f y✝)) (List.map f tl).Nodup
        h₃ : Eq (f y✝) (f y✝)
        ⊢ Eq y✝ y✝
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case cons.inl.inr
        α : Type u
        β : Type v
        f : α → β
        tl : List α
        ih : (List.map f tl).Nodup → ∀ ⦃x : α⦄, Membership.mem tl x → ∀ ⦃y : α⦄, Membe …
        x✝ : α
        d : And (∀ (x : α), Membership.mem tl x → Ne (f x) (f x✝)) (List.map f tl).Nodup
        y✝ : α
        h₂ : Membership.mem tl y✝
        h₃ : Eq (f x✝) (f y✝)
        ⊢ Eq x✝ y✝
      -/
    · apply (d.1 _ h₂ h₃.symm).elim
      /-
        🎉 no goals
      -/
      /-
        case cons.inr.inl
        α : Type u
        β : Type v
        f : α → β
        tl : List α
        ih : (List.map f tl).Nodup → ∀ ⦃x : α⦄, Membership.mem tl x → ∀ ⦃y : α⦄, Membe …
        x✝ : α
        h₁ : Membership.mem tl x✝
        y✝ : α
        d : And (∀ (x : α), Membership.mem tl x → Ne (f x) (f y✝)) (List.map f tl).Nodup
        h₃ : Eq (f x✝) (f y✝)
        ⊢ Eq x✝ y✝
      -/
    · apply (d.1 _ h₁ h₃).elim
      /-
        🎉 no goals
      -/
      /-
        case cons.inr.inr
        α : Type u
        β : Type v
        f : α → β
        hd : α
        tl : List α
        ih : (List.map f tl).Nodup → ∀ ⦃x : α⦄, Membership.mem tl x → ∀ ⦃y : α⦄, Membe …
        d : And (∀ (x : α), Membership.mem tl x → Ne (f x) (f hd)) (List.map f tl).Nodup
        x✝ : α
        h₁ : Membership.mem tl x✝
        y✝ : α
        h₂ : Membership.mem tl y✝
        h₃ : Eq (f x✝) (f y✝)
        ⊢ Eq x✝ y✝
      -/
    · apply ih d.2 h₁ h₂ h₃
      /-
        🎉 no goals
      -/


theorem nodup_map_iff_inj_on {f : α → β} {l : List α} (d : Nodup l) :
    Nodup (map f l) ↔ ∀ x ∈ l, ∀ y ∈ l, f x = f y → x = y :=
  ⟨inj_on_of_nodup_map, fun h => d.map_on h⟩


protected theorem Nodup.map {f : α → β} (hf : Injective f) : Nodup l → Nodup (map f l) :=
  Nodup.map_on fun _ _ _ _ h => hf h


theorem nodup_map_iff {f : α → β} {l : List α} (hf : Injective f) : Nodup (map f l) ↔ Nodup l :=
  ⟨Nodup.of_map _, Nodup.map hf⟩


@[simp]
theorem nodup_attach {l : List α} : Nodup (attach l) ↔ Nodup l :=
  ⟨fun h => attach_map_subtype_val l ▸ h.map fun _ _ => Subtype.eq, fun h =>
    Nodup.of_map Subtype.val ((attach_map_subtype_val l).symm ▸ h)⟩


protected alias ⟨Nodup.of_attach, Nodup.attach⟩ := nodup_attach


theorem Nodup.pmap {p : α → Prop} {f : ∀ a, p a → β} {l : List α} {H}
    (hf : ∀ a ha b hb, f a ha = f b hb → a = b) (h : Nodup l) : Nodup (pmap f l H) := by
  /-
    α : Type u
    β : Type v
    p : α → Prop
    f : (a : α) → p a → β
    l : List α
    H : ∀ (a : α), Membership.mem l a → p a
    hf : ∀ (a : α) (ha : p a) (b : α) (hb : p b), Eq (f a ha) (f b hb) → Eq a b
    h : l.Nodup
    ⊢ (List.pmap f l H).Nodup
  -/
  rw [pmap_eq_map_attach]
  /-
    α : Type u
    β : Type v
    p : α → Prop
    f : (a : α) → p a → β
    l : List α
    H : ∀ (a : α), Membership.mem l a → p a
    hf : ∀ (a : α) (ha : p a) (b : α) (hb : p b), Eq (f a ha) (f b hb) → Eq a b
    h : l.Nodup
    ⊢ (List.map (fun x => f ↑x ⋯) l.attach).Nodup
  -/
  exact h.attach.map fun ⟨a, ha⟩ ⟨b, hb⟩ h => by congr; exact hf a (H _ ha) b (H _ hb) h
  /-
    🎉 no goals
  -/


theorem Nodup.filter (p : α → Bool) {l} : Nodup l → Nodup (filter p l) := by
  /-
    α : Type u
    p : α → Bool
    l : List α
    ⊢ l.Nodup → (List.filter p l).Nodup
  -/
  simpa using Pairwise.filter p
  /-
    🎉 no goals
  -/


@[simp]
theorem nodup_reverse {l : List α} : Nodup (reverse l) ↔ Nodup l :=
                               /-
                                 α : Type u
                                 l : List α
                                 ⊢ Iff (List.Pairwise (fun a b => Ne b a) l) l.Nodup
                               -/
  pairwise_reverse.trans <| by simp only [Nodup, Ne, eq_comm]
                               /-
                                 🎉 no goals
                               -/


lemma nodup_tail_reverse (l : List α) (h : l[0]? = l.getLast?) :
    Nodup l.reverse.tail ↔ Nodup l.tail := by
  induction l with
  | nil => simp
  | cons a l ih =>
    by_cases hl : l = []
    · aesop
    · simp_all only [List.get?_eq_getElem?, List.tail_reverse, List.nodup_reverse,
        List.dropLast_cons_of_ne_nil hl, List.tail_cons]
      simp only [length_cons, Nat.zero_lt_succ, getElem?_eq_getElem, getElem_cons_zero,
        Nat.add_one_sub_one, Nat.lt_add_one, Option.some.injEq, List.getElem_cons,
        show l.length ≠ 0 by aesop, ↓reduceDIte, getLast?_eq_getElem?] at h
      rw [h,
        show l.Nodup = (l.dropLast ++ [l.getLast hl]).Nodup by
          simp [List.dropLast_eq_take, ← List.drop_length_sub_one],
        List.nodup_append_comm]
      simp [List.getLast_eq_getElem]


theorem Nodup.erase_getElem [DecidableEq α] {l : List α} (hl : l.Nodup)
    (i : Nat) (h : i < l.length) : l.erase l[i] = l.eraseIdx ↑i := by
  induction l generalizing i with
  | nil => simp
  | cons a l IH =>
    cases i with
    | zero => simp
    | succ i =>
      rw [nodup_cons] at hl
      rw [erase_cons_tail]
      · simp [IH hl.2]
      · rw [beq_iff_eq]
        simp only [getElem_cons_succ]
        simp only [length_cons, Nat.succ_eq_add_one, Nat.add_lt_add_iff_right] at h
        exact mt (· ▸ getElem_mem h) hl.1


theorem Nodup.erase_get [DecidableEq α] {l : List α} (hl : l.Nodup) (i : Fin l.length) :
    l.erase (l.get i) = l.eraseIdx ↑i := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    i : Fin l.length
    ⊢ Eq (l.erase (l.get i)) (l.eraseIdx ↑i)
  -/
  simp [erase_getElem, hl]
  /-
    🎉 no goals
  -/


theorem Nodup.diff [DecidableEq α] : l₁.Nodup → (l₁.diff l₂).Nodup :=
  Nodup.sublist <| diff_sublist _ _


theorem nodup_flatten {L : List (List α)} :
    Nodup (flatten L) ↔ (∀ l ∈ L, Nodup l) ∧ Pairwise Disjoint L := by
  /-
    α : Type u
    L : List (List α)
    ⊢ Iff L.flatten.Nodup (And (∀ (l : List α), Membership.mem L l → l.Nodup) (Lis …
  -/
  simp only [Nodup, pairwise_flatten, disjoint_left.symm, forall_mem_ne]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2025-10-15")] alias nodup_join := nodup_flatten


theorem nodup_flatMap {l₁ : List α} {f : α → List β} :
    Nodup (l₁.flatMap f) ↔
      (∀ x ∈ l₁, Nodup (f x)) ∧ Pairwise (Disjoint on f) l₁ := by
  simp only [List.flatMap, nodup_flatten, pairwise_map, and_comm, and_left_comm, mem_map,
    exists_imp, and_imp]
  rw [show (∀ (l : List β) (x : α), f x = l → x ∈ l₁ → Nodup l) ↔ ∀ x : α, x ∈ l₁ → Nodup (f x)
      from forall_swap.trans <| forall_congr' fun _ => forall_eq']


@[deprecated (since := "2025-10-16")] alias nodup_bind := nodup_flatMap


protected theorem Nodup.product {l₂ : List β} (d₁ : l₁.Nodup) (d₂ : l₂.Nodup) :
    (l₁ ×ˢ l₂).Nodup :=
  nodup_flatMap.2
    ⟨fun a _ => d₂.map <| LeftInverse.injective fun b => (rfl : (a, b).2 = b),
      d₁.imp fun {a₁ a₂} n x h₁ h₂ => by
        /-
          α : Type u
          β : Type v
          l₁ : List α
          l₂ : List β
          d₁ : l₁.Nodup
          d₂ : l₂.Nodup
          a₁ a₂ : α
          n : Ne a₁ a₂
          x : Prod α β
          h₁ : Membership.mem ((fun a => List.map (Prod.mk a) l₂) a₁) x
          h₂ : Membership.mem ((fun a => List.map (Prod.mk a) l₂) a₂) x
          ⊢ False
        -/
        rcases mem_map.1 h₁ with ⟨b₁, _, rfl⟩
        /-
          case intro.intro
          α : Type u
          β : Type v
          l₁ : List α
          l₂ : List β
          d₁ : l₁.Nodup
          d₂ : l₂.Nodup
          a₁ a₂ : α
          n : Ne a₁ a₂
          b₁ : β
          left✝ : Membership.mem l₂ b₁
          h₁ : Membership.mem ((fun a => List.map (Prod.mk a) l₂) a₁) { fst := a₁, snd : …
          h₂ : Membership.mem ((fun a => List.map (Prod.mk a) l₂) a₂) { fst := a₁, snd : …
          ⊢ False
        -/
        rcases mem_map.1 h₂ with ⟨b₂, mb₂, ⟨⟩⟩
        /-
          case intro.intro.intro.intro.refl
          α : Type u
          β : Type v
          l₁ : List α
          l₂ : List β
          d₁ : l₁.Nodup
          d₂ : l₂.Nodup
          a₁ : α
          b₁ : β
          left✝ : Membership.mem l₂ b₁
          h₁ : Membership.mem ((fun a => List.map (Prod.mk a) l₂) a₁) { fst := a₁, snd : …
          n : Ne a₁ a₁
          h₂ : Membership.mem ((fun a => List.map (Prod.mk a) l₂) a₁) { fst := a₁, snd : …
          mb₂ : Membership.mem l₂ b₁
          ⊢ False
        -/
        exact n rfl⟩
        /-
          🎉 no goals
        -/


theorem Nodup.sigma {σ : α → Type*} {l₂ : ∀ a , List (σ a)} (d₁ : Nodup l₁)
    (d₂ : ∀ a , Nodup (l₂ a)) : (l₁.sigma l₂).Nodup :=
  nodup_flatMap.2
                                            /-
                                              α : Type u
                                              l₁ : List α
                                              σ : α → Type u_1
                                              l₂ : (a : α) → List (σ a)
                                              d₁ : l₁.Nodup
                                              d₂ : ∀ (a : α), (l₂ a).Nodup
                                              a : α
                                              x✝ : Membership.mem l₁ a
                                              b b' : σ a
                                              h : Eq ⟨a, b⟩ ⟨a, b'⟩
                                              ⊢ Eq b b'
                                            -/
    ⟨fun a _ => (d₂ a).map fun b b' h => by injection h with _ h,
                                            /-
                                              🎉 no goals
                                            -/
      d₁.imp fun {a₁ a₂} n x h₁ h₂ => by
        /-
          α : Type u
          l₁ : List α
          σ : α → Type u_1
          l₂ : (a : α) → List (σ a)
          d₁ : l₁.Nodup
          d₂ : ∀ (a : α), (l₂ a).Nodup
          a₁ a₂ : α
          n : Ne a₁ a₂
          x : Sigma fun a => σ a
          h₁ : Membership.mem ((fun a => List.map (Sigma.mk a) (l₂ a)) a₁) x
          h₂ : Membership.mem ((fun a => List.map (Sigma.mk a) (l₂ a)) a₂) x
          ⊢ False
        -/
        rcases mem_map.1 h₁ with ⟨b₁, _, rfl⟩
        /-
          case intro.intro
          α : Type u
          l₁ : List α
          σ : α → Type u_1
          l₂ : (a : α) → List (σ a)
          d₁ : l₁.Nodup
          d₂ : ∀ (a : α), (l₂ a).Nodup
          a₁ a₂ : α
          n : Ne a₁ a₂
          b₁ : σ a₁
          left✝ : Membership.mem (l₂ a₁) b₁
          h₁ : Membership.mem ((fun a => List.map (Sigma.mk a) (l₂ a)) a₁) ⟨a₁, b₁⟩
          h₂ : Membership.mem ((fun a => List.map (Sigma.mk a) (l₂ a)) a₂) ⟨a₁, b₁⟩
          ⊢ False
        -/
        rcases mem_map.1 h₂ with ⟨b₂, mb₂, ⟨⟩⟩
        /-
          case intro.intro.intro.intro.refl
          α : Type u
          l₁ : List α
          σ : α → Type u_1
          l₂ : (a : α) → List (σ a)
          d₁ : l₁.Nodup
          d₂ : ∀ (a : α), (l₂ a).Nodup
          a₁ : α
          b₁ : σ a₁
          left✝ : Membership.mem (l₂ a₁) b₁
          h₁ : Membership.mem ((fun a => List.map (Sigma.mk a) (l₂ a)) a₁) ⟨a₁, b₁⟩
          n : Ne a₁ a₁
          h₂ : Membership.mem ((fun a => List.map (Sigma.mk a) (l₂ a)) a₁) ⟨a₁, b₁⟩
          mb₂ : Membership.mem (l₂ a₁) b₁
          ⊢ False
        -/
        exact n rfl⟩
        /-
          🎉 no goals
        -/


protected theorem Nodup.filterMap {f : α → Option β} (h : ∀ a a' b, b ∈ f a → b ∈ f a' → a = a') :
    Nodup l → Nodup (filterMap f l) :=
                                                                         /-
                                                                           α : Type u
                                                                           β : Type v
                                                                           l : List α
                                                                           f : α → Option β
                                                                           h : ∀ (a a' : α) (b : β), Membership.mem (f a) b → Membership.mem (f a') b → E …
                                                                           a a' : α
                                                                           n : Ne a a'
                                                                           b : β
                                                                           bm : Membership.mem (f a) b
                                                                           b' : β
                                                                           bm' : Membership.mem (f a') b'
                                                                           e : Eq b b'
                                                                           ⊢ Membership.mem (f a) b'
                                                                         -/
  (Pairwise.filterMap f) @fun a a' n b bm b' bm' e => n <| h a a' b' (by rw [← e]; exact bm) bm'
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


protected theorem Nodup.concat (h : a ∉ l) (h' : l.Nodup) : (l.concat a).Nodup := by
  /-
    α : Type u
    l : List α
    a : α
    h : Not (Membership.mem l a)
    h' : l.Nodup
    ⊢ (l.concat a).Nodup
  -/
  rw [concat_eq_append]; exact h'.append (nodup_singleton _) (disjoint_singleton.2 h)
                         /-
                           🎉 no goals
                         -/


protected theorem Nodup.insert [DecidableEq α] (h : l.Nodup) : (l.insert a).Nodup :=
                        /-
                          α : Type u
                          l : List α
                          a : α
                          inst✝ : DecidableEq α
                          h : l.Nodup
                          h' : Membership.mem l a
                          ⊢ (List.insert a l).Nodup
                        -/
  if h' : a ∈ l then by rw [insert_of_mem h']; exact h
                                               /-
                                                 🎉 no goals
                                               -/
          /-
            α : Type u
            l : List α
            a : α
            inst✝ : DecidableEq α
            h : l.Nodup
            h' : Not (Membership.mem l a)
            ⊢ (List.insert a l).Nodup
          -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  else by rw [insert_of_not_mem h', nodup_cons]; constructor <;> assumption
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem Nodup.union [DecidableEq α] (l₁ : List α) (h : Nodup l₂) : (l₁ ∪ l₂).Nodup := by
  /-
    α : Type u
    l₂ : List α
    inst✝ : DecidableEq α
    l₁ : List α
    h : l₂.Nodup
    ⊢ (Union.union l₁ l₂).Nodup
  -/
  induction' l₁ with a l₁ ih generalizing l₂
    /-
      case nil
      α : Type u
      inst✝ : DecidableEq α
      l₂ : List α
      h : l₂.Nodup
      ⊢ (Union.union List.nil l₂).Nodup
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      inst✝ : DecidableEq α
      a : α
      l₁ : List α
      ih : ∀ {l₂ : List α}, l₂.Nodup → (Union.union l₁ l₂).Nodup
      l₂ : List α
      h : l₂.Nodup
      ⊢ (Union.union (List.cons a l₁) l₂).Nodup
    -/
  · exact (ih h).insert
    /-
      🎉 no goals
    -/


theorem Nodup.inter [DecidableEq α] (l₂ : List α) : Nodup l₁ → Nodup (l₁ ∩ l₂) :=
  Nodup.filter _


theorem Nodup.diff_eq_filter [BEq α] [LawfulBEq α] :
    ∀ {l₁ l₂ : List α} (_ : l₁.Nodup), l₁.diff l₂ = l₁.filter (· ∉ l₂)
                    /-
                      α : Type u
                      inst✝¹ : BEq α
                      inst✝ : LawfulBEq α
                      l₁ : List α
                      x✝ : l₁.Nodup
                      ⊢ Eq (l₁.diff List.nil) (List.filter (fun x => Decidable.decide (Not (Membersh …
                    -/
  | l₁, [], _ => by simp
                    /-
                      🎉 no goals
                    -/
  | l₁, a :: l₂, hl₁ => by
    /-
      α : Type u
      inst✝¹ : BEq α
      inst✝ : LawfulBEq α
      l₁ : List α
      a : α
      l₂ : List α
      hl₁ : l₁.Nodup
      ⊢ Eq (l₁.diff (List.cons a l₂)) (List.filter (fun x => Decidable.decide (Not ( …
    -/
    rw [diff_cons, (hl₁.erase _).diff_eq_filter, hl₁.erase_eq_filter, filter_filter]
    /-
      α : Type u
      inst✝¹ : BEq α
      inst✝ : LawfulBEq α
      l₁ : List α
      a : α
      l₂ : List α
      hl₁ : l₁.Nodup
      ⊢ Eq (List.filter (fun a_1 => (Decidable.decide (Not (Membership.mem l₂ a_1))) …
    -/
    simp only [decide_not, bne, Bool.and_comm, mem_cons, not_or, decide_mem_cons, Bool.not_or]
    /-
      🎉 no goals
    -/


theorem Nodup.mem_diff_iff [DecidableEq α] (hl₁ : l₁.Nodup) : a ∈ l₁.diff l₂ ↔ a ∈ l₁ ∧ a ∉ l₂ := by
  /-
    α : Type u
    l₁ l₂ : List α
    a : α
    inst✝ : DecidableEq α
    hl₁ : l₁.Nodup
    ⊢ Iff (Membership.mem (l₁.diff l₂) a) (And (Membership.mem l₁ a) (Not (Members …
  -/
  rw [hl₁.diff_eq_filter, mem_filter, decide_eq_true_iff]
  /-
    🎉 no goals
  -/


protected theorem Nodup.set :
    ∀ {l : List α} {n : ℕ} {a : α} (_ : l.Nodup) (_ : a ∉ l), (l.set n a).Nodup
  | [], _, _, _, _ => nodup_nil
  | _ :: _, 0, _, hl, ha => nodup_cons.2 ⟨mt (mem_cons_of_mem _) ha, (nodup_cons.1 hl).2⟩
  | _ :: _, _ + 1, _, hl, ha =>
    nodup_cons.2
      ⟨fun h =>
        (mem_or_eq_of_mem_set h).elim (nodup_cons.1 hl).1 fun hba => ha (hba ▸ mem_cons_self _ _),
        hl.of_cons.set (mt (mem_cons_of_mem _) ha)⟩


theorem Nodup.map_update [DecidableEq α] {l : List α} (hl : l.Nodup) (f : α → β) (x : α) (y : β) :
    l.map (Function.update f x y) =
      if x ∈ l then (l.map f).set (l.indexOf x) y else l.map f := by
  /-
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    l : List α
    hl : l.Nodup
    f : α → β
    x : α
    y : β
    ⊢ Eq (List.map (Function.update f x y) l) (ite (Membership.mem l x) ((List.map …
  -/
  induction' l with hd tl ihl; · simp
                                 /-
                                   🎉 no goals
                                 -/
  /-
    case cons
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    f : α → β
    x : α
    y : β
    hd : α
    tl : List α
    ihl : tl.Nodup → Eq (List.map (Function.update f x y) tl) (ite (Membership.mem …
    hl : (List.cons hd tl).Nodup
    ⊢ Eq (List.map (Function.update f x y) (List.cons hd tl)) (ite (Membership.mem …
  -/
  rw [nodup_cons] at hl
  /-
    case cons
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    f : α → β
    x : α
    y : β
    hd : α
    tl : List α
    ihl : tl.Nodup → Eq (List.map (Function.update f x y) tl) (ite (Membership.mem …
    hl : And (Not (Membership.mem tl hd)) tl.Nodup
    ⊢ Eq (List.map (Function.update f x y) (List.cons hd tl)) (ite (Membership.mem …
  -/
  simp only [mem_cons, map, ihl hl.2]
  /-
    case cons
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    f : α → β
    x : α
    y : β
    hd : α
    tl : List α
    ihl : tl.Nodup → Eq (List.map (Function.update f x y) tl) (ite (Membership.mem …
    hl : And (Not (Membership.mem tl hd)) tl.Nodup
    ⊢ Eq (List.cons (Function.update f x y hd) (ite (Membership.mem tl x) ((List.m …
  -/
  by_cases H : hd = x
    /-
      case pos
      α : Type u
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      x : α
      y : β
      hd : α
      tl : List α
      ihl : tl.Nodup → Eq (List.map (Function.update f x y) tl) (ite (Membership.mem …
      hl : And (Not (Membership.mem tl hd)) tl.Nodup
      H : Eq hd x
      ⊢ Eq (List.cons (Function.update f x y hd) (ite (Membership.mem tl x) ((List.m …
    -/
  · subst hd
    /-
      case pos
      α : Type u
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      x : α
      y : β
      tl : List α
      ihl : tl.Nodup → Eq (List.map (Function.update f x y) tl) (ite (Membership.mem …
      hl : And (Not (Membership.mem tl x)) tl.Nodup
      ⊢ Eq (List.cons (Function.update f x y x) (ite (Membership.mem tl x) ((List.ma …
    -/
    simp [set, hl.1]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      x : α
      y : β
      hd : α
      tl : List α
      ihl : tl.Nodup → Eq (List.map (Function.update f x y) tl) (ite (Membership.mem …
      hl : And (Not (Membership.mem tl hd)) tl.Nodup
      H : Not (Eq hd x)
      ⊢ Eq (List.cons (Function.update f x y hd) (ite (Membership.mem tl x) ((List.m …
    -/
  · simp [Ne.symm H, H, set, ← apply_ite (cons (f hd))]
    /-
      🎉 no goals
    -/


theorem Nodup.pairwise_of_forall_ne {l : List α} {r : α → α → Prop} (hl : l.Nodup)
    (h : ∀ a ∈ l, ∀ b ∈ l, a ≠ b → r a b) : l.Pairwise r := by
  /-
    α : Type u
    l : List α
    r : α → α → Prop
    hl : l.Nodup
    h : ∀ (a : α), Membership.mem l a → ∀ (b : α), Membership.mem l b → Ne a b → r …
    ⊢ List.Pairwise r l
  -/
  rw [pairwise_iff_forall_sublist]
  /-
    α : Type u
    l : List α
    r : α → α → Prop
    hl : l.Nodup
    h : ∀ (a : α), Membership.mem l a → ∀ (b : α), Membership.mem l b → Ne a b → r …
    ⊢ ∀ {a b : α}, (List.cons a (List.cons b List.nil)).Sublist l → r a b
  -/
  intro a b hab
  if heq : a = b then
    cases heq; have := nodup_iff_sublist.mp hl _ hab; contradiction
  else
    apply h <;> try (apply hab.subset; simp)
    exact heq


theorem Nodup.pairwise_of_set_pairwise {l : List α} {r : α → α → Prop} (hl : l.Nodup)
    (h : { x | x ∈ l }.Pairwise r) : l.Pairwise r :=
  hl.pairwise_of_forall_ne h


@[simp]
theorem Nodup.pairwise_coe [IsSymm α r] (hl : l.Nodup) :
    { a | a ∈ l }.Pairwise r ↔ l.Pairwise r := by
  /-
    α : Type u
    l : List α
    r : α → α → Prop
    inst✝ : IsSymm α r
    hl : l.Nodup
    ⊢ Iff ((setOf fun a => Membership.mem l a).Pairwise r) (List.Pairwise r l)
  -/
  induction' l with a l ih
    /-
      case nil
      α : Type u
      l : List α
      r : α → α → Prop
      inst✝ : IsSymm α r
      hl : List.nil.Nodup
      ⊢ Iff ((setOf fun a => Membership.mem List.nil a).Pairwise r) (List.Pairwise r …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u
    l✝ : List α
    r : α → α → Prop
    inst✝ : IsSymm α r
    a : α
    l : List α
    ih : l.Nodup → Iff ((setOf fun a => Membership.mem l a).Pairwise r) (List.Pair …
    hl : (List.cons a l).Nodup
    ⊢ Iff ((setOf fun a_1 => Membership.mem (List.cons a l) a_1).Pairwise r) (List …
  -/
  rw [List.nodup_cons] at hl
  have : ∀ b ∈ l, ¬a = b → r a b ↔ r a b := fun b hb =>
    imp_iff_right (ne_of_mem_of_not_mem hb hl.1).symm
  simp [Set.setOf_or, Set.pairwise_insert_of_symmetric fun _ _ ↦ symm_of r, ih hl.2, and_comm,
    forall₂_congr this]


theorem Nodup.take_eq_filter_mem [DecidableEq α] :
    ∀ {l : List α} {n : ℕ} (_ : l.Nodup), l.take n = l.filter (l.take n).elem
                   /-
                     α : Type u
                     inst✝ : DecidableEq α
                     n : Nat
                     x✝ : List.nil.Nodup
                     ⊢ Eq (List.take n List.nil) (List.filter (fun a => List.elem a (List.take n Li …
                   -/
  | [], n, _ => by simp
                   /-
                     🎉 no goals
                   -/
                     /-
                       α : Type u
                       inst✝ : DecidableEq α
                       b : α
                       l : List α
                       x✝ : (List.cons b l).Nodup
                       ⊢ Eq (List.take 0 (List.cons b l)) (List.filter (fun a => List.elem a (List.ta …
                     -/
  | b::l, 0, _ => by simp
                     /-
                       🎉 no goals
                     -/
  | b::l, n+1, hl => by
    /-
      α : Type u
      inst✝ : DecidableEq α
      b : α
      l : List α
      n : Nat
      hl : (List.cons b l).Nodup
      ⊢ Eq (List.take (HAdd.hAdd n 1) (List.cons b l)) (List.filter (fun a => List.e …
    -/
    rw [take_succ_cons, Nodup.take_eq_filter_mem (Nodup.of_cons hl), filter_cons_of_pos (by simp)]
    /-
      α : Type u
      inst✝ : DecidableEq α
      b : α
      l : List α
      n : Nat
      hl : (List.cons b l).Nodup
      ⊢ Eq (List.cons b (List.filter (fun a => List.elem a (List.take n l)) l)) (Lis …
    -/
    congr 1
    /-
      case e_tail
      α : Type u
      inst✝ : DecidableEq α
      b : α
      l : List α
      n : Nat
      hl : (List.cons b l).Nodup
      ⊢ Eq (List.filter (fun a => List.elem a (List.take n l)) l) (List.filter (fun  …
    -/
    refine List.filter_congr ?_
    /-
      case e_tail
      α : Type u
      inst✝ : DecidableEq α
      b : α
      l : List α
      n : Nat
      hl : (List.cons b l).Nodup
      ⊢ ∀ (x : α), Membership.mem l x → Eq (List.elem x (List.take n l)) (List.elem  …
    -/
    intro x hx
    /-
      case e_tail
      α : Type u
      inst✝ : DecidableEq α
      b : α
      l : List α
      n : Nat
      hl : (List.cons b l).Nodup
      x : α
      hx : Membership.mem l x
      ⊢ Eq (List.elem x (List.take n l)) (List.elem x (List.cons b (List.filter (fun …
    -/
    have : x ≠ b := fun h => (nodup_cons.1 hl).1 (h ▸ hx)
    /-
      case e_tail
      α : Type u
      inst✝ : DecidableEq α
      b : α
      l : List α
      n : Nat
      hl : (List.cons b l).Nodup
      x : α
      hx : Membership.mem l x
      this : Ne x b
      ⊢ Eq (List.elem x (List.take n l)) (List.elem x (List.cons b (List.filter (fun …
    -/
    simp (config := {contextual := true}) [List.mem_filter, this, hx]
    /-
      🎉 no goals
    -/

theorem Option.toList_nodup : ∀ o : Option α, o.toList.Nodup
  | none => List.nodup_nil
  | some x => List.nodup_singleton x

