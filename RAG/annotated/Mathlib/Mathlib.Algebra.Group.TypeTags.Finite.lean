instance [Finite α] : Finite (Additive α) :=
                        /-
                          α : Type u
                          inst✝ : Finite α
                          ⊢ Equiv α (Additive α)
                        -/
  Finite.of_equiv α (by rfl)
                        /-
                          🎉 no goals
                        -/


instance [Finite α] : Finite (Multiplicative α) :=
                        /-
                          α : Type u
                          inst✝ : Finite α
                          ⊢ Equiv α (Multiplicative α)
                        -/
  Finite.of_equiv α (by rfl)
                        /-
                          🎉 no goals
                        -/


instance [h : Infinite α] : Infinite (Additive α) := h


instance [h : Infinite α] : Infinite (Multiplicative α) := h

