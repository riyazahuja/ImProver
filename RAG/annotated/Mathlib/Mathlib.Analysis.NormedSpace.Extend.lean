/-- Extend `fr : F →ₗ[ℝ] ℝ` to `F →ₗ[𝕜] 𝕜` in a way that will also be continuous and have its norm
bounded by `‖fr‖` if `fr` is continuous. -/
noncomputable def extendTo𝕜' (fr : F →ₗ[ℝ] ℝ) : F →ₗ[𝕜] 𝕜 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    F : Type u_2
    inst✝³ : AddCommGroup F
    inst✝² : Module Real F
    inst✝¹ : Module 𝕜 F
    inst✝ : IsScalarTower Real 𝕜 F
    fr : LinearMap (RingHom.id Real) F Real
    ⊢ LinearMap (RingHom.id 𝕜) F 𝕜
  -/
  let fc : F → 𝕜 := fun x => (fr x : 𝕜) - (I : 𝕜) * fr ((I : 𝕜) • x)
  have add : ∀ x y : F, fc (x + y) = fc x + fc y := by
    intro x y
    simp only [fc, smul_add, LinearMap.map_add, ofReal_add]
    rw [mul_add]
    abel
  have A : ∀ (c : ℝ) (x : F), (fr ((c : 𝕜) • x) : 𝕜) = (c : 𝕜) * (fr x : 𝕜) := by
    intro c x
    rw [← ofReal_mul]
    congr 1
    rw [RCLike.ofReal_alg, smul_assoc, fr.map_smul, Algebra.id.smul_eq_mul, one_smul]
  have smul_ℝ : ∀ (c : ℝ) (x : F), fc ((c : 𝕜) • x) = (c : 𝕜) * fc x := by
    intro c x
    dsimp only [fc]
    rw [A c x, smul_smul, mul_comm I (c : 𝕜), ← smul_smul, A, mul_sub]
    ring
  have smul_I : ∀ x : F, fc ((I : 𝕜) • x) = (I : 𝕜) * fc x := by
    intro x
    dsimp only [fc]
    cases' @I_mul_I_ax 𝕜 _ with h h
    · simp [h]
    rw [mul_sub, ← mul_assoc, smul_smul, h]
    simp only [neg_mul, LinearMap.map_neg, one_mul, one_smul, mul_neg, ofReal_neg, neg_smul,
      sub_neg_eq_add, add_comm]
  have smul_𝕜 : ∀ (c : 𝕜) (x : F), fc (c • x) = c • fc x := by
    intro c x
    rw [← re_add_im c, add_smul, add_smul, add, smul_ℝ, ← smul_smul, smul_ℝ, smul_I, ← mul_assoc]
    rfl
  exact
    { toFun := fc
      map_add' := add
      map_smul' := smul_𝕜 }


theorem extendTo𝕜'_apply (fr : F →ₗ[ℝ] ℝ) (x : F) :
    fr.extendTo𝕜' x = (fr x : 𝕜) - (I : 𝕜) * (fr ((I : 𝕜) • x) : 𝕜) := rfl


@[simp]
theorem extendTo𝕜'_apply_re (fr : F →ₗ[ℝ] ℝ) (x : F) : re (fr.extendTo𝕜' x : 𝕜) = fr x := by
  simp only [extendTo𝕜'_apply, map_sub, zero_mul, mul_zero, sub_zero,
    rclike_simps]


theorem norm_extendTo𝕜'_apply_sq (fr : F →ₗ[ℝ] ℝ) (x : F) :
    ‖(fr.extendTo𝕜' x : 𝕜)‖ ^ 2 = fr (conj (fr.extendTo𝕜' x : 𝕜) • x) :=
  calc
    ‖(fr.extendTo𝕜' x : 𝕜)‖ ^ 2 = re (conj (fr.extendTo𝕜' x) * fr.extendTo𝕜' x : 𝕜) := by
      /-
        𝕜 : Type u_1
        inst✝⁴ : RCLike 𝕜
        F : Type u_2
        inst✝³ : AddCommGroup F
        inst✝² : Module Real F
        inst✝¹ : Module 𝕜 F
        inst✝ : IsScalarTower Real 𝕜 F
        fr : LinearMap (RingHom.id Real) F Real
        x : F
        ⊢ Eq (HPow.hPow (Norm.norm (fr.extendTo𝕜' x)) 2) (RCLike.re (HMul.hMul ((starR …
      -/
      rw [RCLike.conj_mul, ← ofReal_pow, ofReal_re]
      /-
        🎉 no goals
      -/
    _ = fr (conj (fr.extendTo𝕜' x : 𝕜) • x) := by
      /-
        𝕜 : Type u_1
        inst✝⁴ : RCLike 𝕜
        F : Type u_2
        inst✝³ : AddCommGroup F
        inst✝² : Module Real F
        inst✝¹ : Module 𝕜 F
        inst✝ : IsScalarTower Real 𝕜 F
        fr : LinearMap (RingHom.id Real) F Real
        x : F
        ⊢ Eq (RCLike.re (HMul.hMul ((starRingEnd 𝕜) (fr.extendTo𝕜' x)) (fr.extendTo𝕜'  …
      -/
      rw [← smul_eq_mul, ← map_smul, extendTo𝕜'_apply_re]
      /-
        🎉 no goals
      -/


/-- The norm of the extension is bounded by `‖fr‖`. -/
theorem norm_extendTo𝕜'_bound (fr : F →L[ℝ] ℝ) (x : F) :
    ‖(fr.toLinearMap.extendTo𝕜' x : 𝕜)‖ ≤ ‖fr‖ * ‖x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    F : Type u_2
    inst✝³ : SeminormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace Real F
    inst✝ : IsScalarTower Real 𝕜 F
    fr : ContinuousLinearMap (RingHom.id Real) F Real
    x : F
    ⊢ LE.le (Norm.norm ((↑fr).extendTo𝕜' x)) (HMul.hMul (Norm.norm fr) (Norm.norm  …
  -/
  set lm : F →ₗ[𝕜] 𝕜 := fr.toLinearMap.extendTo𝕜'
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    F : Type u_2
    inst✝³ : SeminormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace Real F
    inst✝ : IsScalarTower Real 𝕜 F
    fr : ContinuousLinearMap (RingHom.id Real) F Real
    x : F
    lm : LinearMap (RingHom.id 𝕜) F 𝕜 := (↑fr).extendTo𝕜'
    ⊢ LE.le (Norm.norm (lm x)) (HMul.hMul (Norm.norm fr) (Norm.norm x))
  -/
  by_cases h : lm x = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : RCLike 𝕜
      F : Type u_2
      inst✝³ : SeminormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedSpace Real F
      inst✝ : IsScalarTower Real 𝕜 F
      fr : ContinuousLinearMap (RingHom.id Real) F Real
      x : F
      lm : LinearMap (RingHom.id 𝕜) F 𝕜 := (↑fr).extendTo𝕜'
      h : Eq (lm x) 0
      ⊢ LE.le (Norm.norm (lm x)) (HMul.hMul (Norm.norm fr) (Norm.norm x))
    -/
  · rw [h, norm_zero]
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : RCLike 𝕜
      F : Type u_2
      inst✝³ : SeminormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedSpace Real F
      inst✝ : IsScalarTower Real 𝕜 F
      fr : ContinuousLinearMap (RingHom.id Real) F Real
      x : F
      lm : LinearMap (RingHom.id 𝕜) F 𝕜 := (↑fr).extendTo𝕜'
      h : Eq (lm x) 0
      ⊢ LE.le 0 (HMul.hMul (Norm.norm fr) (Norm.norm x))
    -/
                         /-
                           🎉 no goals
                         -/
    apply mul_nonneg <;> exact norm_nonneg _
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    F : Type u_2
    inst✝³ : SeminormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace Real F
    inst✝ : IsScalarTower Real 𝕜 F
    fr : ContinuousLinearMap (RingHom.id Real) F Real
    x : F
    lm : LinearMap (RingHom.id 𝕜) F 𝕜 := (↑fr).extendTo𝕜'
    h : Not (Eq (lm x) 0)
    ⊢ LE.le (Norm.norm (lm x)) (HMul.hMul (Norm.norm fr) (Norm.norm x))
  -/
  rw [← mul_le_mul_left (norm_pos_iff.2 h), ← sq]
  calc
    ‖lm x‖ ^ 2 = fr (conj (lm x : 𝕜) • x) := fr.toLinearMap.norm_extendTo𝕜'_apply_sq x
    _ ≤ ‖fr (conj (lm x : 𝕜) • x)‖ := le_abs_self _
    _ ≤ ‖fr‖ * ‖conj (lm x : 𝕜) • x‖ := le_opNorm _ _
    _ = ‖(lm x : 𝕜)‖ * (‖fr‖ * ‖x‖) := by rw [norm_smul, norm_conj, mul_left_comm]


/-- Extend `fr : F →L[ℝ] ℝ` to `F →L[𝕜] 𝕜`. -/
noncomputable def extendTo𝕜' (fr : F →L[ℝ] ℝ) : F →L[𝕜] 𝕜 :=
  LinearMap.mkContinuous _ ‖fr‖ fr.norm_extendTo𝕜'_bound


theorem extendTo𝕜'_apply (fr : F →L[ℝ] ℝ) (x : F) :
    fr.extendTo𝕜' x = (fr x : 𝕜) - (I : 𝕜) * (fr ((I : 𝕜) • x) : 𝕜) := rfl


@[simp]
theorem norm_extendTo𝕜' (fr : F →L[ℝ] ℝ) : ‖(fr.extendTo𝕜' : F →L[𝕜] 𝕜)‖ = ‖fr‖ :=
  le_antisymm (LinearMap.mkContinuous_norm_le _ (norm_nonneg _) _) <|
    opNorm_le_bound _ (norm_nonneg _) fun x =>
      calc
        ‖fr x‖ = ‖re (fr.extendTo𝕜' x : 𝕜)‖ := congr_arg norm (fr.extendTo𝕜'_apply_re x).symm
        _ ≤ ‖(fr.extendTo𝕜' x : 𝕜)‖ := abs_re_le_norm _
        _ ≤ ‖(fr.extendTo𝕜' : F →L[𝕜] 𝕜)‖ * ‖x‖ := le_opNorm _ _


instance : NormedSpace 𝕜 (RestrictScalars ℝ 𝕜 F) := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    F : Type u_2
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    ⊢ NormedSpace 𝕜 (RestrictScalars Real 𝕜 F)
  -/
  unfold RestrictScalars
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    F : Type u_2
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    ⊢ NormedSpace 𝕜 F
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Extend `fr : RestrictScalars ℝ 𝕜 F →ₗ[ℝ] ℝ` to `F →ₗ[𝕜] 𝕜`. -/
noncomputable def LinearMap.extendTo𝕜 (fr : RestrictScalars ℝ 𝕜 F →ₗ[ℝ] ℝ) : F →ₗ[𝕜] 𝕜 :=
  fr.extendTo𝕜'


theorem LinearMap.extendTo𝕜_apply (fr : RestrictScalars ℝ 𝕜 F →ₗ[ℝ] ℝ) (x : F) :
    fr.extendTo𝕜 x = (fr x : 𝕜) - (I : 𝕜) * (fr ((I : 𝕜) • x) : 𝕜) := rfl


/-- Extend `fr : RestrictScalars ℝ 𝕜 F →L[ℝ] ℝ` to `F →L[𝕜] 𝕜`. -/
noncomputable def ContinuousLinearMap.extendTo𝕜 (fr : RestrictScalars ℝ 𝕜 F →L[ℝ] ℝ) : F →L[𝕜] 𝕜 :=
  fr.extendTo𝕜'


theorem ContinuousLinearMap.extendTo𝕜_apply (fr : RestrictScalars ℝ 𝕜 F →L[ℝ] ℝ) (x : F) :
    fr.extendTo𝕜 x = (fr x : 𝕜) - (I : 𝕜) * (fr ((I : 𝕜) • x) : 𝕜) := rfl


@[simp]
theorem ContinuousLinearMap.norm_extendTo𝕜 (fr : RestrictScalars ℝ 𝕜 F →L[ℝ] ℝ) :
    ‖fr.extendTo𝕜‖ = ‖fr‖ :=
  fr.norm_extendTo𝕜'

