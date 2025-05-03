theorem ULift.down_injective {α : Sort _} : Function.Injective (@ULift.down α)
                      /-
                        α : Type u_1
                        a b : α
                        x✝ : Eq { down := a }.down { down := b }.down
                        ⊢ Eq { down := a } { down := b }
                      -/
  | ⟨a⟩, ⟨b⟩, _ => by congr
                      /-
                        🎉 no goals
                      -/


@[simp] theorem ULift.down_inj {α : Sort _} {a b : ULift α} : a.down = b.down ↔ a = b :=
                                              /-
                                                α : Type u_1
                                                a b : ULift α
                                                h : Eq a b
                                                ⊢ Eq a.down b.down
                                              -/
  ⟨fun h ↦ ULift.down_injective h, fun h ↦ by rw [h]⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem PLift.down_injective : Function.Injective (@PLift.down α)
                      /-
                        α : Sort u_1
                        a b : α
                        x✝ : Eq { down := a }.down { down := b }.down
                        ⊢ Eq { down := a } { down := b }
                      -/
  | ⟨a⟩, ⟨b⟩, _ => by congr
                      /-
                        🎉 no goals
                      -/


@[simp] theorem PLift.down_inj {a b : PLift α} : a.down = b.down ↔ a = b :=
                                              /-
                                                α : Sort u_1
                                                a b : PLift α
                                                h : Eq a b
                                                ⊢ Eq a.down b.down
                                              -/
  ⟨fun h ↦ PLift.down_injective h, fun h ↦ by rw [h]⟩
                                              /-
                                                🎉 no goals
                                              -/

