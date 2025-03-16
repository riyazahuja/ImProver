@[simp]
theorem LinearMap.toMatrix_transpose (u : V₁ →ₗ[K] V₂) :
    LinearMap.toMatrix B₂.dualBasis B₁.dualBasis (Module.Dual.transpose (R := K) u) =
      (LinearMap.toMatrix B₁ B₂ u)ᵀ := by
  /-
    K : Type u_1
    V₁ : Type u_2
    V₂ : Type u_3
    ι₁ : Type u_4
    ι₂ : Type u_5
    inst✝⁸ : Field K
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : Module K V₁
    inst✝⁵ : AddCommGroup V₂
    inst✝⁴ : Module K V₂
    inst✝³ : Fintype ι₁
    inst✝² : Fintype ι₂
    inst✝¹ : DecidableEq ι₁
    inst✝ : DecidableEq ι₂
    B₁ : Basis ι₁ K V₁
    B₂ : Basis ι₂ K V₂
    u : LinearMap (RingHom.id K) V₁ V₂
    ⊢ Eq ((LinearMap.toMatrix B₂.dualBasis B₁.dualBasis) (Module.Dual.transpose u) …
  -/
  ext i j
  simp only [LinearMap.toMatrix_apply, Module.Dual.transpose_apply, B₁.dualBasis_repr,
    B₂.dualBasis_apply, Matrix.transpose_apply, LinearMap.comp_apply]


@[simp]
theorem Matrix.toLin_transpose (M : Matrix ι₁ ι₂ K) : Matrix.toLin B₁.dualBasis B₂.dualBasis Mᵀ =
    Module.Dual.transpose (R := K) (Matrix.toLin B₂ B₁ M) := by
  /-
    K : Type u_1
    V₁ : Type u_2
    V₂ : Type u_3
    ι₁ : Type u_4
    ι₂ : Type u_5
    inst✝⁸ : Field K
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : Module K V₁
    inst✝⁵ : AddCommGroup V₂
    inst✝⁴ : Module K V₂
    inst✝³ : Fintype ι₁
    inst✝² : Fintype ι₂
    inst✝¹ : DecidableEq ι₁
    inst✝ : DecidableEq ι₂
    B₁ : Basis ι₁ K V₁
    B₂ : Basis ι₂ K V₂
    M : Matrix ι₁ ι₂ K
    ⊢ Eq ((Matrix.toLin B₁.dualBasis B₂.dualBasis) M.transpose) (Module.Dual.trans …
  -/
  apply (LinearMap.toMatrix B₁.dualBasis B₂.dualBasis).injective
  /-
    case a
    K : Type u_1
    V₁ : Type u_2
    V₂ : Type u_3
    ι₁ : Type u_4
    ι₂ : Type u_5
    inst✝⁸ : Field K
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : Module K V₁
    inst✝⁵ : AddCommGroup V₂
    inst✝⁴ : Module K V₂
    inst✝³ : Fintype ι₁
    inst✝² : Fintype ι₂
    inst✝¹ : DecidableEq ι₁
    inst✝ : DecidableEq ι₂
    B₁ : Basis ι₁ K V₁
    B₂ : Basis ι₂ K V₂
    M : Matrix ι₁ ι₂ K
    ⊢ Eq ((LinearMap.toMatrix B₁.dualBasis B₂.dualBasis) ((Matrix.toLin B₁.dualBas …
  -/
  rw [LinearMap.toMatrix_toLin, LinearMap.toMatrix_transpose, LinearMap.toMatrix_toLin]
  /-
    🎉 no goals
  -/


