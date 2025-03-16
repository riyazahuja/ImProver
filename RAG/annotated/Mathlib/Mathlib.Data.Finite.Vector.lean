instance List.Vector.finite {α : Type*} [Finite α] {n : ℕ} : Finite (Vector α n) := by
  /-
    α : Type u_1
    inst✝ : Finite α
    n : Nat
    ⊢ Finite (List.Vector α n)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    inst✝ : Finite α
    n : Nat
    this : Fintype α
    ⊢ Finite (List.Vector α n)
  -/
  infer_instance
  /-
    🎉 no goals
  -/

