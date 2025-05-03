@[simp]
theorem exp_eps : exp 𝕜 (eps : DualNumber R) = 1 + eps :=
  exp_inr _ _


@[simp]
theorem exp_smul_eps (r : R) : exp 𝕜 (r • eps : DualNumber R) = 1 + r • eps := by
  /-
    𝕜 : Type u_1
    R : Type u_2
    inst✝⁶ : Field 𝕜
    inst✝⁵ : CharZero 𝕜
    inst✝⁴ : CommRing R
    inst✝³ : Algebra 𝕜 R
    inst✝² : UniformSpace R
    inst✝¹ : TopologicalRing R
    inst✝ : T2Space R
    r : R
    ⊢ Eq (NormedSpace.exp 𝕜 (HSMul.hSMul r DualNumber.eps)) (HAdd.hAdd 1 (HSMul.hS …
  -/
  rw [eps, ← inr_smul, exp_inr]
  /-
    🎉 no goals
  -/


