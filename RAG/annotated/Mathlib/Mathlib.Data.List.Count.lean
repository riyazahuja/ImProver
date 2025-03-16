lemma countP_erase [DecidableEq α] (p : α → Bool) (l : List α) (a : α) :
    countP p (l.erase a) = countP p l - if a ∈ l ∧ p a then 1 else 0 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : α → Bool
    l : List α
    a : α
    ⊢ Eq (List.countP p (l.erase a)) (HSub.hSub (List.countP p l) (ite (And (Membe …
  -/
  rw [countP_eq_length_filter, countP_eq_length_filter, ← erase_filter, length_erase]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    p : α → Bool
    l : List α
    a : α
    ⊢ Eq (ite (Membership.mem (List.filter p l) a) (HSub.hSub (List.filter p l).le …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma count_diff [DecidableEq α] (a : α) (l₁ : List α) :
    ∀ l₂, count a (l₁.diff l₂) = count a l₁ - count a l₂
  | [] => rfl
  | b :: l₂ => by
    simp only [diff_cons, count_diff, count_erase, beq_iff_eq, Nat.sub_right_comm, count_cons,
      Nat.sub_add_eq]


lemma countP_diff [DecidableEq α] {l₁ l₂ : List α} (hl : l₂ <+~ l₁) (p : α → Bool) :
    countP p (l₁.diff l₂) = countP p l₁ - countP p l₂ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l₁ l₂ : List α
    hl : l₂.Subperm l₁
    p : α → Bool
    ⊢ Eq (List.countP p (l₁.diff l₂)) (HSub.hSub (List.countP p l₁) (List.countP p …
  -/
  refine (Nat.sub_eq_of_eq_add ?_).symm
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l₁ l₂ : List α
    hl : l₂.Subperm l₁
    p : α → Bool
    ⊢ Eq (List.countP p l₁) (HAdd.hAdd (List.countP p (l₁.diff l₂)) (List.countP p …
  -/
  rw [← countP_append]
  exact ((subperm_append_diff_self_of_count_le <| subperm_ext_iff.1 hl).symm.trans
    perm_append_comm).countP_eq _


@[simp]
theorem count_map_of_injective {β} [DecidableEq α] [DecidableEq β] (l : List α) (f : α → β)
    (hf : Function.Injective f) (x : α) : count (f x) (map f l) = count x l := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    l : List α
    f : α → β
    hf : Function.Injective f
    x : α
    ⊢ Eq (List.count (f x) (List.map f l)) (List.count x l)
  -/
  simp only [count, countP_map]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    l : List α
    f : α → β
    hf : Function.Injective f
    x : α
    ⊢ Eq (List.countP (Function.comp (fun x_1 => BEq.beq x_1 (f x)) f) l) (List.co …
  -/
  unfold Function.comp
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    l : List α
    f : α → β
    hf : Function.Injective f
    x : α
    ⊢ Eq (List.countP (fun x_1 => (fun x_2 => BEq.beq x_2 (f x)) (f x_1)) l) (List …
  -/
  simp only [hf.beq_eq]
  /-
    🎉 no goals
  -/


