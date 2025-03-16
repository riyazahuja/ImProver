@[to_additive (attr := simp)]
theorem prod_insertNone (f : Option α → M) (s : Finset α) :
                                                                 /-
                                                                   α : Type u_1
                                                                   M : Type u_2
                                                                   inst✝ : CommMonoid M
                                                                   f : Option α → M
                                                                   s : Finset α
                                                                   ⊢ Eq ((Finset.insertNone s).prod fun x => f x) (HMul.hMul (f Option.none) (s.p …
                                                                 -/
    ∏ x ∈ insertNone s, f x = f none * ∏ x ∈ s, f (some x) := by simp [insertNone]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
theorem mul_prod_eq_prod_insertNone (f : α → M) (x : M) (s : Finset α) :
    x * ∏ i ∈ s, f i = ∏ i ∈ insertNone s, i.elim x f :=
  (prod_insertNone (fun i => i.elim x f) _).symm


@[to_additive]
theorem prod_eraseNone (f : α → M) (s : Finset (Option α)) :
    ∏ x ∈ eraseNone s, f x = ∏ x ∈ s, Option.elim' 1 f x := by
  classical calc
      ∏ x ∈ eraseNone s, f x = ∏ x ∈ (eraseNone s).map Embedding.some, Option.elim' 1 f x :=
        (prod_map (eraseNone s) Embedding.some <| Option.elim' 1 f).symm
      _ = ∏ x ∈ s.erase none, Option.elim' 1 f x := by rw [map_some_eraseNone]
      _ = ∏ x ∈ s, Option.elim' 1 f x := prod_erase _ rfl


