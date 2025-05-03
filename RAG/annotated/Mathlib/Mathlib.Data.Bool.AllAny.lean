@[deprecated (since := "2024-08-10")] alias all_iff_forall := all_eq_true


theorem all_iff_forall_prop : (all l fun a => p a) ↔ ∀ a ∈ l, p a := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    l : List α
    ⊢ Iff (Eq (l.all fun a => Decidable.decide (p a)) Bool.true) (∀ (a : α), Membe …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-10")] alias any_iff_exists := any_eq_true


                                                                        /-
                                                                          α : Type u_1
                                                                          p : α → Prop
                                                                          inst✝ : DecidablePred p
                                                                          l : List α
                                                                          ⊢ Iff (Eq (l.any fun a => Decidable.decide (p a)) Bool.true) (Exists fun a =>  …
                                                                        -/
theorem any_iff_exists_prop : (any l fun a => p a) ↔ ∃ a ∈ l, p a := by simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem any_of_mem {p : α → Bool} (h₁ : a ∈ l) (h₂ : p a) : any l p :=
  any_eq_true.2 ⟨_, h₁, h₂⟩


