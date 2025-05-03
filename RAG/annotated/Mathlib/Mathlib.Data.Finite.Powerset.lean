instance [Finite α] : Finite (Set α) := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Finite (Set α)
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u_1
    inst✝ : Finite α
    val✝ : Fintype α
    ⊢ Finite (Set α)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


