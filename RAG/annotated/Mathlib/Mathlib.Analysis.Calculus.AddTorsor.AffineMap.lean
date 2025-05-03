/-- A continuous affine map between normed vector spaces is smooth. -/
theorem contDiff {n : WithTop ℕ∞} (f : V →ᴬ[𝕜] W) : ContDiff 𝕜 n f := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace 𝕜 W
    n : WithTop ENat
    f : ContinuousAffineMap 𝕜 V W
    ⊢ ContDiff 𝕜 n ⇑f
  -/
  rw [f.decomp]
  /-
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace 𝕜 W
    n : WithTop ENat
    f : ContinuousAffineMap 𝕜 V W
    ⊢ ContDiff 𝕜 n (HAdd.hAdd (⇑f.contLinear) (Function.const V (f 0)))
  -/
  apply f.contLinear.contDiff.add
  /-
    𝕜 : Type u_1
    V : Type u_2
    W : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace 𝕜 W
    n : WithTop ENat
    f : ContinuousAffineMap 𝕜 V W
    ⊢ ContDiff 𝕜 n (Function.const V (f 0))
  -/
  exact contDiff_const
  /-
    🎉 no goals
  -/


