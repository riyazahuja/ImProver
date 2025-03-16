instance {β : α → Type*} [Finite α] [∀ a, Finite (β a)] : Finite (Σa, β a) := by
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝¹ : Finite α
    inst✝ : ∀ (a : α), Finite (β a)
    ⊢ Finite (Sigma fun a => β a)
  -/
  letI := Fintype.ofFinite α
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝¹ : Finite α
    inst✝ : ∀ (a : α), Finite (β a)
    this : Fintype α := Fintype.ofFinite α
    ⊢ Finite (Sigma fun a => β a)
  -/
  letI := fun a => Fintype.ofFinite (β a)
  /-
    α : Type u_1
    β : α → Type u_2
    inst✝¹ : Finite α
    inst✝ : ∀ (a : α), Finite (β a)
    this✝ : Fintype α := Fintype.ofFinite α
    this : (a : α) → Fintype (β a) := fun a => Fintype.ofFinite (β a)
    ⊢ Finite (Sigma fun a => β a)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {ι : Sort*} {π : ι → Sort*} [Finite ι] [∀ i, Finite (π i)] : Finite (Σ'i, π i) :=
  of_equiv _ (Equiv.psigmaEquivSigmaPLift π).symm


