/-- See also `List.subperm_ext_iff`. -/
lemma subperm_iff_count [DecidableEq α] : l₁ <+~ l₂ ↔ ∀ a, count a l₁ ≤ count a l₂ :=
  subperm_ext_iff.trans <| forall_congr' fun a ↦ by
    /-
      α : Type u_1
      l₁ l₂ : List α
      inst✝ : DecidableEq α
      a : α
      ⊢ Iff (Membership.mem l₁ a → LE.le (List.count a l₁) (List.count a l₂)) (LE.le …
    -/
                             /-
                               🎉 no goals
                             -/
    by_cases ha : a ∈ l₁ <;> simp [ha, count_eq_zero_of_not_mem]
                             /-
                               🎉 no goals
                             -/


lemma subperm_iff : l₁ <+~ l₂ ↔ ∃ l, l ~ l₂ ∧ l₁ <+ l := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    ⊢ Iff (l₁.Subperm l₂) (Exists fun l => And (l.Perm l₂) (l₁.Sublist l))
  -/
  refine ⟨?_, fun ⟨l, h₁, h₂⟩ ↦ h₂.subperm.trans h₁.subperm⟩
  /-
    α : Type u_1
    l₁ l₂ : List α
    ⊢ l₁.Subperm l₂ → Exists fun l => And (l.Perm l₂) (l₁.Sublist l)
  -/
  rintro ⟨l, h₁, h₂⟩
  /-
    case intro.intro
    α : Type u_1
    l₁ l₂ l : List α
    h₁ : l.Perm l₁
    h₂ : l.Sublist l₂
    ⊢ Exists fun l => And (l.Perm l₂) (l₁.Sublist l)
  -/
  obtain ⟨l', h₂⟩ := h₂.exists_perm_append
  /-
    case intro.intro.intro
    α : Type u_1
    l₁ l₂ l : List α
    h₁ : l.Perm l₁
    h₂✝ : l.Sublist l₂
    l' : List α
    h₂ : l₂.Perm (HAppend.hAppend l l')
    ⊢ Exists fun l => And (l.Perm l₂) (l₁.Sublist l)
  -/
  exact ⟨l₁ ++ l', (h₂.trans (h₁.append_right _)).symm, (prefix_append _ _).sublist⟩
  /-
    🎉 no goals
  -/


@[simp] lemma subperm_singleton_iff : l <+~ [a] ↔ l = [] ∨ l = [a] := by
  /-
    α : Type u_1
    l : List α
    a : α
    ⊢ Iff (l.Subperm (List.cons a List.nil)) (Or (Eq l List.nil) (Eq l (List.cons  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      l : List α
      a : α
      ⊢ l.Subperm (List.cons a List.nil) → Or (Eq l List.nil) (Eq l (List.cons a Lis …
    -/
  · rw [subperm_iff]
    /-
      case mp
      α : Type u_1
      l : List α
      a : α
      ⊢ (Exists fun l_1 => And (l_1.Perm (List.cons a List.nil)) (l.Sublist l_1)) →  …
    -/
    rintro ⟨s, hla, h⟩
    /-
      case mp.intro.intro
      α : Type u_1
      l : List α
      a : α
      s : List α
      hla : s.Perm (List.cons a List.nil)
      h : l.Sublist s
      ⊢ Or (Eq l List.nil) (Eq l (List.cons a List.nil))
    -/
    rwa [perm_singleton.mp hla, sublist_singleton] at h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      l : List α
      a : α
      ⊢ Or (Eq l List.nil) (Eq l (List.cons a List.nil)) → l.Subperm (List.cons a Li …
    -/
  · rintro (rfl | rfl)
    /-
      case mpr.inl
      α : Type u_1
      a : α
      ⊢ List.nil.Subperm (List.cons a List.nil)
    -/
    exacts [nil_subperm, Subperm.refl _]
    /-
      🎉 no goals
    -/


lemma subperm_cons_self : l <+~ a :: l := ⟨l, Perm.refl _, sublist_cons_self _ _⟩


protected alias ⟨subperm.of_cons, subperm.cons⟩ := subperm_cons


@[deprecated List.cons_subperm_of_not_mem_of_mem (since := "2024-12-11"), nolint unusedArguments]
theorem cons_subperm_of_mem {a : α} {l₁ l₂ : List α} (_ : Nodup l₁) (h₁ : a ∉ l₁) (h₂ : a ∈ l₂)
    (s : l₁ <+~ l₂) : a :: l₁ <+~ l₂ :=
  cons_subperm_of_not_mem_of_mem h₁ h₂ s


protected theorem Nodup.subperm (d : Nodup l₁) (H : l₁ ⊆ l₂) : l₁ <+~ l₂ :=
  subperm_of_subset d H


