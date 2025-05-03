instance [Finite α] [Finite β] : Finite (α ⊕ β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    ⊢ Finite (Sum α β)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this : Fintype α
    ⊢ Finite (Sum α β)
  -/
  haveI := Fintype.ofFinite β
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this✝ : Fintype α
    this : Fintype β
    ⊢ Finite (Sum α β)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem sum_left (β) [Finite (α ⊕ β)] : Finite α :=
  of_injective (Sum.inl : α → α ⊕ β) Sum.inl_injective


theorem sum_right (α) [Finite (α ⊕ β)] : Finite β :=
  of_injective (Sum.inr : β → α ⊕ β) Sum.inr_injective


