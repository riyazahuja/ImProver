/-- A map between two inner product spaces is a conformal map if and only if it preserves inner
products up to a scalar factor, i.e., there exists a positive `c : ℝ` such that
`⟪f u, f v⟫ = c * ⟪u, v⟫` for all `u`, `v`. -/
theorem isConformalMap_iff (f : E →L[ℝ] F) :
    IsConformalMap f ↔ ∃ c : ℝ, 0 < c ∧ ∀ u v : E, ⟪f u, f v⟫ = c * ⟪u, v⟫ := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real E
    inst✝ : InnerProductSpace Real F
    f : ContinuousLinearMap (RingHom.id Real) E F
    ⊢ Iff (IsConformalMap f) (Exists fun c => And (LT.lt 0 c) (∀ (u v : E), Eq (In …
  -/
  constructor
    /-
      case mp
      E : Type u_1
      F : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real E
      inst✝ : InnerProductSpace Real F
      f : ContinuousLinearMap (RingHom.id Real) E F
      ⊢ IsConformalMap f → Exists fun c => And (LT.lt 0 c) (∀ (u v : E), Eq (Inner.i …
    -/
  · rintro ⟨c₁, hc₁, li, rfl⟩
    /-
      case mp.intro.intro.intro
      E : Type u_1
      F : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real E
      inst✝ : InnerProductSpace Real F
      c₁ : Real
      hc₁ : Ne c₁ 0
      li : LinearIsometry (RingHom.id Real) E F
      ⊢ Exists fun c => And (LT.lt 0 c) (∀ (u v : E), Eq (Inner.inner ((HSMul.hSMul  …
    -/
    refine ⟨c₁ * c₁, mul_self_pos.2 hc₁, fun u v => ?_⟩
    simp only [real_inner_smul_left, real_inner_smul_right, mul_assoc, coe_smul',
      coe_toContinuousLinearMap, Pi.smul_apply, inner_map_map]
    /-
      case mpr
      E : Type u_1
      F : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real E
      inst✝ : InnerProductSpace Real F
      f : ContinuousLinearMap (RingHom.id Real) E F
      ⊢ (Exists fun c => And (LT.lt 0 c) (∀ (u v : E), Eq (Inner.inner (f u) (f v))  …
    -/
  · rintro ⟨c₁, hc₁, huv⟩
    obtain ⟨c, hc, rfl⟩ : ∃ c : ℝ, 0 < c ∧ c₁ = c * c :=
      ⟨√c₁, Real.sqrt_pos.2 hc₁, (Real.mul_self_sqrt hc₁.le).symm⟩
    /-
      case mpr.intro.intro.intro.intro
      E : Type u_1
      F : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real E
      inst✝ : InnerProductSpace Real F
      f : ContinuousLinearMap (RingHom.id Real) E F
      c : Real
      hc : LT.lt 0 c
      hc₁ : LT.lt 0 (HMul.hMul c c)
      huv : ∀ (u v : E), Eq (Inner.inner (f u) (f v)) (HMul.hMul (HMul.hMul c c) (In …
      ⊢ IsConformalMap f
    -/
    refine ⟨c, hc.ne', (c⁻¹ • f : E →ₗ[ℝ] F).isometryOfInner fun u v => ?_, ?_⟩
    · simp only [real_inner_smul_left, real_inner_smul_right, huv, mul_assoc, coe_smul,
        inv_mul_cancel_left₀ hc.ne', LinearMap.smul_apply, ContinuousLinearMap.coe_coe]
      /-
        case mpr.intro.intro.intro.intro.refine_2
        E : Type u_1
        F : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real E
        inst✝ : InnerProductSpace Real F
        f : ContinuousLinearMap (RingHom.id Real) E F
        c : Real
        hc : LT.lt 0 c
        hc₁ : LT.lt 0 (HMul.hMul c c)
        huv : ∀ (u v : E), Eq (Inner.inner (f u) (f v)) (HMul.hMul (HMul.hMul c c) (In …
        ⊢ Eq f (HSMul.hSMul c ((HSMul.hSMul (Inv.inv c) ↑f).isometryOfInner ⋯).toConti …
      -/
    · ext1 x
      /-
        case mpr.intro.intro.intro.intro.refine_2.h
        E : Type u_1
        F : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real E
        inst✝ : InnerProductSpace Real F
        f : ContinuousLinearMap (RingHom.id Real) E F
        c : Real
        hc : LT.lt 0 c
        hc₁ : LT.lt 0 (HMul.hMul c c)
        huv : ∀ (u v : E), Eq (Inner.inner (f u) (f v)) (HMul.hMul (HMul.hMul c c) (In …
        x : E
        ⊢ Eq (f x) ((HSMul.hSMul c ((HSMul.hSMul (Inv.inv c) ↑f).isometryOfInner ⋯).to …
      -/
      exact (smul_inv_smul₀ hc.ne' (f x)).symm
      /-
        🎉 no goals
      -/

