/-- TFAE: The Following (propositions) Are Equivalent.

The `tfae_have` and `tfae_finish` tactics can be useful in proofs with `TFAE` goals.
-/
def TFAE (l : List Prop) : Prop :=
  ∀ x ∈ l, ∀ y ∈ l, x ↔ y


theorem tfae_nil : TFAE [] :=
  forall_mem_nil _


@[simp]
                                            /-
                                              p : Prop
                                              ⊢ (List.cons p List.nil).TFAE
                                            -/
theorem tfae_singleton (p) : TFAE [p] := by simp [TFAE, -eq_iff_iff]
                                            /-
                                              🎉 no goals
                                            -/


theorem tfae_cons_of_mem {a b} {l : List Prop} (h : b ∈ l) : TFAE (a :: l) ↔ (a ↔ b) ∧ TFAE l :=
                     /-
                       a b : Prop
                       l : List Prop
                       h : Membership.mem l b
                       H : (List.cons a l).TFAE
                       ⊢ Membership.mem (List.cons a l) a
                     -/
  ⟨fun H => ⟨H a (by simp) b (Mem.tail a h),
                     /-
                       🎉 no goals
                     -/
    fun _ hp _ hq => H _ (Mem.tail a hp) _ (Mem.tail a hq)⟩,
      by
        /-
          a b : Prop
          l : List Prop
          h : Membership.mem l b
          ⊢ And (Iff a b) l.TFAE → (List.cons a l).TFAE
        -/
        rintro ⟨ab, H⟩ p (_ | ⟨_, hp⟩) q (_ | ⟨_, hq⟩)
          /-
            case intro.head.head
            a b : Prop
            l : List Prop
            h : Membership.mem l b
            ab : Iff a b
            H : l.TFAE
            ⊢ Iff a a
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case intro.head.tail
            a b : Prop
            l : List Prop
            h : Membership.mem l b
            ab : Iff a b
            H : l.TFAE
            q : Prop
            hq : List.Mem q l
            ⊢ Iff a q
          -/
        · exact ab.trans (H _ h _ hq)
          /-
            🎉 no goals
          -/
          /-
            case intro.tail.head
            a b : Prop
            l : List Prop
            h : Membership.mem l b
            ab : Iff a b
            H : l.TFAE
            p : Prop
            hp : List.Mem p l
            ⊢ Iff p a
          -/
        · exact (ab.trans (H _ h _ hp)).symm
          /-
            🎉 no goals
          -/
          /-
            case intro.tail.tail
            a b : Prop
            l : List Prop
            h : Membership.mem l b
            ab : Iff a b
            H : l.TFAE
            p : Prop
            hp : List.Mem p l
            q : Prop
            hq : List.Mem q l
            ⊢ Iff p q
          -/
        · exact H _ hp _ hq⟩
          /-
            🎉 no goals
          -/


theorem tfae_cons_cons {a b} {l : List Prop} : TFAE (a :: b :: l) ↔ (a ↔ b) ∧ TFAE (b :: l) :=
  tfae_cons_of_mem (Mem.head _)


@[simp]
theorem tfae_cons_self {a} {l : List Prop} : TFAE (a :: a :: l) ↔ TFAE (a :: l) := by
  /-
    a : Prop
    l : List Prop
    ⊢ Iff (List.cons a (List.cons a l)).TFAE (List.cons a l).TFAE
  -/
  simp [tfae_cons_cons]
  /-
    🎉 no goals
  -/


theorem tfae_of_forall (b : Prop) (l : List Prop) (h : ∀ a ∈ l, a ↔ b) : TFAE l :=
  fun _a₁ h₁ _a₂ h₂ => (h _ h₁).trans (h _ h₂).symm


theorem tfae_of_cycle {a b} {l : List Prop} (h_chain : List.Chain (· → ·) a (b :: l))
    (h_last : getLastD l b → a) : TFAE (a :: b :: l) := by
  induction l generalizing a b with
  | nil => simp_all [tfae_cons_cons, iff_def]
  | cons c l IH =>
    simp only [tfae_cons_cons, getLastD_cons, tfae_singleton, and_true, chain_cons, Chain.nil] at *
    rcases h_chain with ⟨ab, ⟨bc, ch⟩⟩
    have := IH ⟨bc, ch⟩ (ab ∘ h_last)
    exact ⟨⟨ab, h_last ∘ (this.2 c (.head _) _ (getLastD_mem_cons _ _)).1 ∘ bc⟩, this⟩


theorem TFAE.out {l} (h : TFAE l) (n₁ n₂) {a b} (h₁ : List.get? l n₁ = some a := by rfl)
    (h₂ : List.get? l n₂ = some b := by rfl) : a ↔ b :=
  h _ (List.mem_of_get? h₁) _ (List.mem_of_get? h₂)


/-- If `P₁ x ↔ ... ↔ Pₙ x` for all `x`, then `(∀ x, P₁ x) ↔ ... ↔ (∀ x, Pₙ x)`.
Note: in concrete cases, Lean has trouble finding the list `[P₁, ..., Pₙ]` from the list
`[(∀ x, P₁ x), ..., (∀ x, Pₙ x)]`, but simply providing a list of underscores with the right
length makes it happier.

Example:
```lean
example (P₁ P₂ P₃ : ℕ → Prop) (H : ∀ n, [P₁ n, P₂ n, P₃ n].TFAE) :
    [∀ n, P₁ n, ∀ n, P₂ n, ∀ n, P₃ n].TFAE :=
  forall_tfae [_, _, _] H
```
-/
theorem forall_tfae {α : Type*} (l : List (α → Prop)) (H : ∀ a : α, (l.map (fun p ↦ p a)).TFAE) :
    (l.map (fun p ↦ ∀ a, p a)).TFAE := by
  /-
    α : Type u_1
    l : List (α → Prop)
    H : ∀ (a : α), (List.map (fun p => p a) l).TFAE
    ⊢ (List.map (fun p => ∀ (a : α), p a) l).TFAE
  -/
  simp only [TFAE, List.forall_mem_map]
  /-
    α : Type u_1
    l : List (α → Prop)
    H : ∀ (a : α), (List.map (fun p => p a) l).TFAE
    ⊢ ∀ (j : α → Prop), Membership.mem l j → ∀ (j_1 : α → Prop), Membership.mem l  …
  -/
  intros p₁ hp₁ p₂ hp₂
  exact forall_congr' fun a ↦ H a (p₁ a) (mem_map_of_mem (fun p ↦ p a) hp₁)
    (p₂ a) (mem_map_of_mem (fun p ↦ p a) hp₂)


/-- If `P₁ x ↔ ... ↔ Pₙ x` for all `x`, then `(∃ x, P₁ x) ↔ ... ↔ (∃ x, Pₙ x)`.
Note: in concrete cases, Lean has trouble finding the list `[P₁, ..., Pₙ]` from the list
`[(∃ x, P₁ x), ..., (∃ x, Pₙ x)]`, but simply providing a list of underscores with the right
length makes it happier.

Example:
```lean
example (P₁ P₂ P₃ : ℕ → Prop) (H : ∀ n, [P₁ n, P₂ n, P₃ n].TFAE) :
    [∃ n, P₁ n, ∃ n, P₂ n, ∃ n, P₃ n].TFAE :=
  exists_tfae [_, _, _] H
```
-/
theorem exists_tfae {α : Type*} (l : List (α → Prop)) (H : ∀ a : α, (l.map (fun p ↦ p a)).TFAE) :
    (l.map (fun p ↦ ∃ a, p a)).TFAE := by
  /-
    α : Type u_1
    l : List (α → Prop)
    H : ∀ (a : α), (List.map (fun p => p a) l).TFAE
    ⊢ (List.map (fun p => Exists fun a => p a) l).TFAE
  -/
  simp only [TFAE, List.forall_mem_map]
  /-
    α : Type u_1
    l : List (α → Prop)
    H : ∀ (a : α), (List.map (fun p => p a) l).TFAE
    ⊢ ∀ (j : α → Prop), Membership.mem l j → ∀ (j_1 : α → Prop), Membership.mem l  …
  -/
  intros p₁ hp₁ p₂ hp₂
  exact exists_congr fun a ↦ H a (p₁ a) (mem_map_of_mem (fun p ↦ p a) hp₁)
    (p₂ a) (mem_map_of_mem (fun p ↦ p a) hp₂)


theorem tfae_not_iff {l : List Prop} : TFAE (l.map Not) ↔ TFAE l := by
  classical
  simp only [TFAE, mem_map, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂,
    Decidable.not_iff_not]


alias ⟨_, TFAE.not⟩ := tfae_not_iff


