theorem suffixLevenshtein_minimum_le_levenshtein_cons (xs : List α) (y ys) :
    (suffixLevenshtein C xs ys).1.minimum ≤ levenshtein C xs (y :: ys) := by
  induction xs with
  | nil =>
      simp only [suffixLevenshtein_nil', levenshtein_nil_cons,
        List.minimum_singleton, WithTop.coe_le_coe]
      exact le_add_of_nonneg_left (by simp)
  | cons x xs ih =>
    suffices
      (suffixLevenshtein C (x :: xs) ys).1.minimum ≤ (C.delete x + levenshtein C xs (y :: ys)) ∧
        (suffixLevenshtein C (x :: xs) ys).1.minimum ≤ (C.insert y + levenshtein C (x :: xs) ys) ∧
        (suffixLevenshtein C (x :: xs) ys).1.minimum ≤ (C.substitute x y + levenshtein C xs ys) by
      simpa [suffixLevenshtein_eq_tails_map]
    refine ⟨?_, ?_, ?_⟩
    · calc
        _ ≤ (suffixLevenshtein C xs ys).1.minimum := by
            simp [suffixLevenshtein_cons₁_fst, List.minimum_cons]
        _ ≤ ↑(levenshtein C xs (y :: ys)) := ih
        _ ≤ _ := by simp
    · calc
        (suffixLevenshtein C (x :: xs) ys).1.minimum ≤ (levenshtein C (x :: xs) ys) := by
            simp [suffixLevenshtein_cons₁_fst, List.minimum_cons]
        _ ≤ _ := by simp
    · calc
        (suffixLevenshtein C (x :: xs) ys).1.minimum ≤ (levenshtein C xs ys) := by
            simp only [suffixLevenshtein_cons₁_fst, List.minimum_cons]
            apply min_le_of_right_le
            cases xs
            · simp [suffixLevenshtein_nil']
            · simp [suffixLevenshtein_cons₁, List.minimum_cons]
        _ ≤ _ := by simp


theorem le_suffixLevenshtein_cons_minimum (xs : List α) (y ys) :
    (suffixLevenshtein C xs ys).1.minimum ≤ (suffixLevenshtein C xs (y :: ys)).1.minimum := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    ⊢ LE.le (↑(suffixLevenshtein C xs ys)).minimum (↑(suffixLevenshtein C xs (List …
  -/
  apply List.le_minimum_of_forall_le
  /-
    case h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    ⊢ ∀ (a : δ), Membership.mem (↑(suffixLevenshtein C xs (List.cons y ys))) a → L …
  -/
  simp only [suffixLevenshtein_eq_tails_map]
  /-
    case h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    ⊢ ∀ (a : δ), Membership.mem (List.map (fun xs' => levenshtein C xs' (List.cons …
  -/
  simp only [List.mem_map, List.mem_tails, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
  /-
    case h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    ⊢ ∀ (a : List α), a.IsSuffix xs → LE.le (List.map (fun xs' => levenshtein C xs …
  -/
  intro a suff
  /-
    case h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    ⊢ LE.le (List.map (fun xs' => levenshtein C xs' ys) xs.tails).minimum ↑(levens …
  -/
  refine (?_ : _ ≤ _).trans (suffixLevenshtein_minimum_le_levenshtein_cons _ _ _)
  /-
    case h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    ⊢ LE.le (List.map (fun xs' => levenshtein C xs' ys) xs.tails).minimum (↑(suffi …
  -/
  simp only [suffixLevenshtein_eq_tails_map]
  /-
    case h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    ⊢ LE.le (List.map (fun xs' => levenshtein C xs' ys) xs.tails).minimum (List.ma …
  -/
  apply List.le_minimum_of_forall_le
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    ⊢ ∀ (a_1 : δ), Membership.mem (List.map (fun xs' => levenshtein C xs' ys) a.ta …
  -/
  intro b m
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    b : δ
    m : Membership.mem (List.map (fun xs' => levenshtein C xs' ys) a.tails) b
    ⊢ LE.le (List.map (fun xs' => levenshtein C xs' ys) xs.tails).minimum ↑b
  -/
  replace m : ∃ a_1, a_1 <:+ a ∧ levenshtein C a_1 ys = b := by simpa using m
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    b : δ
    m : Exists fun a_1 => And (a_1.IsSuffix a) (Eq (levenshtein C a_1 ys) b)
    ⊢ LE.le (List.map (fun xs' => levenshtein C xs' ys) xs.tails).minimum ↑b
  -/
  obtain ⟨a', suff', rfl⟩ := m
  /-
    case h.h.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    a' : List α
    suff' : a'.IsSuffix a
    ⊢ LE.le (List.map (fun xs' => levenshtein C xs' ys) xs.tails).minimum ↑(levens …
  -/
  apply List.minimum_le_of_mem'
  /-
    case h.h.intro.intro.ha
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    a' : List α
    suff' : a'.IsSuffix a
    ⊢ Membership.mem (List.map (fun xs' => levenshtein C xs' ys) xs.tails) (levens …
  -/
  simp only [List.mem_map, List.mem_tails]
  /-
    case h.h.intro.intro.ha
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    a' : List α
    suff' : a'.IsSuffix a
    ⊢ Exists fun a => And (a.IsSuffix xs) (Eq (levenshtein C a ys) (levenshtein C  …
  -/
  suffices ∃ a, a <:+ xs ∧ levenshtein C a ys = levenshtein C a' ys by simpa
  /-
    case h.h.intro.intro.ha
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    C : Levenshtein.Cost α β δ
    inst✝ : CanonicallyLinearOrderedAddCommMonoid δ
    xs : List α
    y : β
    ys : List β
    a : List α
    suff : a.IsSuffix xs
    a' : List α
    suff' : a'.IsSuffix a
    ⊢ Exists fun a => And (a.IsSuffix xs) (Eq (levenshtein C a ys) (levenshtein C  …
  -/
  exact ⟨a', suff'.trans suff, rfl⟩
  /-
    🎉 no goals
  -/


theorem le_suffixLevenshtein_append_minimum (xs : List α) (ys₁ ys₂) :
    (suffixLevenshtein C xs ys₂).1.minimum ≤ (suffixLevenshtein C xs (ys₁ ++ ys₂)).1.minimum := by
  induction ys₁ with
  | nil => exact le_refl _
  | cons y ys₁ ih => exact ih.trans (le_suffixLevenshtein_cons_minimum _ _ _)


theorem suffixLevenshtein_minimum_le_levenshtein_append (xs ys₁ ys₂) :
    (suffixLevenshtein C xs ys₂).1.minimum ≤ levenshtein C xs (ys₁ ++ ys₂) := by
  cases ys₁ with
  | nil => exact List.minimum_le_of_mem' (List.getElem_mem _)
  | cons y ys₁ =>
      exact (le_suffixLevenshtein_append_minimum _ _ _).trans
        (suffixLevenshtein_minimum_le_levenshtein_cons _ _ _)


theorem le_levenshtein_cons (xs : List α) (y ys) :
    ∃ xs', xs' <:+ xs ∧ levenshtein C xs' ys ≤ levenshtein C xs (y :: ys) := by
  simpa [suffixLevenshtein_eq_tails_map, List.minimum_le_coe_iff] using
    suffixLevenshtein_minimum_le_levenshtein_cons (δ := δ) xs y ys


theorem le_levenshtein_append (xs : List α) (ys₁ ys₂) :
    ∃ xs', xs' <:+ xs ∧ levenshtein C xs' ys₂ ≤ levenshtein C xs (ys₁ ++ ys₂) := by
  simpa [suffixLevenshtein_eq_tails_map, List.minimum_le_coe_iff] using
    suffixLevenshtein_minimum_le_levenshtein_append (δ := δ) xs ys₁ ys₂

