mk_iff_of_inductive_prop List.Pairwise List.pairwise_iff


theorem Pairwise.forall_of_forall (H : Symmetric R) (H₁ : ∀ x ∈ l, R x x) (H₂ : l.Pairwise R) :
    ∀ ⦃x⦄, x ∈ l → ∀ ⦃y⦄, y ∈ l → R x y :=
                                       /-
                                         α : Type u_1
                                         R : α → α → Prop
                                         l : List α
                                         H : Symmetric R
                                         H₁ : ∀ (x : α), Membership.mem l x → R x x
                                         H₂ : List.Pairwise R l
                                         ⊢ List.Pairwise (flip R) l
                                       -/
  H₂.forall_of_forall_of_flip H₁ <| by rwa [H.flip_eq]
                                       /-
                                         🎉 no goals
                                       -/


theorem Pairwise.forall (hR : Symmetric R) (hl : l.Pairwise R) :
    ∀ ⦃a⦄, a ∈ l → ∀ ⦃b⦄, b ∈ l → a ≠ b → R a b := by
  /-
    α : Type u_1
    R : α → α → Prop
    l : List α
    hR : Symmetric R
    hl : List.Pairwise R l
    ⊢ ∀ ⦃a : α⦄, Membership.mem l a → ∀ ⦃b : α⦄, Membership.mem l b → Ne a b → R a b
  -/
  apply Pairwise.forall_of_forall
    /-
      case H
      α : Type u_1
      R : α → α → Prop
      l : List α
      hR : Symmetric R
      hl : List.Pairwise R l
      ⊢ Symmetric fun x y => Ne x y → R x y
    -/
  · exact fun a b h hne => hR (h hne.symm)
    /-
      🎉 no goals
    -/
    /-
      case H₁
      α : Type u_1
      R : α → α → Prop
      l : List α
      hR : Symmetric R
      hl : List.Pairwise R l
      ⊢ ∀ (x : α), Membership.mem l x → Ne x x → R x x
    -/
  · exact fun _ _ hx => (hx rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case H₂
      α : Type u_1
      R : α → α → Prop
      l : List α
      hR : Symmetric R
      hl : List.Pairwise R l
      ⊢ List.Pairwise (fun x y => Ne x y → R x y) l
    -/
  · exact hl.imp (@fun a b h _ => by exact h)
    /-
      🎉 no goals
    -/


theorem Pairwise.set_pairwise (hl : Pairwise R l) (hr : Symmetric R) : { x | x ∈ l }.Pairwise R :=
  hl.forall hr

-- Porting note: Duplicate of `pairwise_map` but with `f` explicit.

@[deprecated "No deprecation message was provided." (since := "2024-02-25")]
theorem pairwise_map' (f : β → α) :
    ∀ {l : List β}, Pairwise R (map f l) ↔ Pairwise (R on f) l
             /-
               α : Type u_1
               β : Type u_2
               R : α → α → Prop
               f : β → α
               ⊢ Iff (List.Pairwise R (List.map f List.nil)) (List.Pairwise (Function.onFun R …
             -/
  | [] => by simp only [map, Pairwise.nil]
             /-
               🎉 no goals
             -/
  | b :: l => by
    simp only [map, pairwise_cons, mem_map, forall_exists_index, and_imp,
      forall_apply_eq_imp_iff₂, pairwise_map]


theorem pairwise_of_reflexive_of_forall_ne {l : List α} {r : α → α → Prop} (hr : Reflexive r)
    (h : ∀ a ∈ l, ∀ b ∈ l, a ≠ b → r a b) : l.Pairwise r := by
  /-
    α : Type u_1
    l : List α
    r : α → α → Prop
    hr : Reflexive r
    h : ∀ (a : α), Membership.mem l a → ∀ (b : α), Membership.mem l b → Ne a b → r …
    ⊢ List.Pairwise r l
  -/
  rw [pairwise_iff_forall_sublist]
  /-
    α : Type u_1
    l : List α
    r : α → α → Prop
    hr : Reflexive r
    h : ∀ (a : α), Membership.mem l a → ∀ (b : α), Membership.mem l b → Ne a b → r …
    ⊢ ∀ {a b : α}, (List.cons a (List.cons b List.nil)).Sublist l → r a b
  -/
  intro a b hab
  if heq : a = b then
    cases heq; apply hr
  else
    apply h <;> try (apply hab.subset; simp)
    exact heq


protected alias ⟨_, Pairwise.pwFilter⟩ := pwFilter_eq_self


