theorem MvPolynomial.continuous_eval : Continuous fun x ↦ eval x p := by
  /-
    X : Type u_1
    σ : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : CommSemiring X
    inst✝ : TopologicalSemiring X
    p : MvPolynomial σ X
    ⊢ Continuous fun x => (MvPolynomial.eval x) p
  -/
  continuity
  /-
    🎉 no goals
  -/

