local notation "𝕢" => qParam


lemma exists_one_half_le_im_and_norm_le (hk : k ≤ 0) (f : F) (τ : ℍ) :
    ∃ ξ : ℍ, 1 / 2 ≤ ξ.im ∧ ‖f τ‖ ≤ ‖f ξ‖ :=
  let ⟨γ, hγ, hdenom⟩ := exists_one_half_le_im_smul_and_norm_denom_le τ
  ⟨γ • τ, hγ, by simpa only [slash_action_eqn'' _ (mem_Gamma_one γ),
    norm_mul, norm_zpow] using le_mul_of_one_le_left (norm_nonneg _) <|
      one_le_zpow_of_nonpos₀ (norm_pos_iff.2 (denom_ne_zero _ _)) hdenom hk⟩


variable (k) in
/-- If a constant function is modular of weight `k`, then either `k = 0`, or the constant is `0`. -/
lemma wt_eq_zero_of_eq_const {f : F} {c : ℂ} (hf : ⇑f = Function.const _ c) :
    k = 0 ∨ c = 0 := by
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma 1) k
    f : F
    c : Complex
    hf : Eq (⇑f) (Function.const UpperHalfPlane c)
    ⊢ Or (Eq k 0) (Eq c 0)
  -/
  have hI := slash_action_eqn'' f (mem_Gamma_one S) I
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma 1) k
    f : F
    c : Complex
    hf : Eq (⇑f) (Function.const UpperHalfPlane c)
    hI : Eq (f (HSMul.hSMul ModularGroup.S UpperHalfPlane.I)) (HMul.hMul (HPow.hPo …
    ⊢ Or (Eq k 0) (Eq c 0)
  -/
  have h2I2 := slash_action_eqn'' f (mem_Gamma_one S) ⟨2 * Complex.I, by norm_num⟩
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma 1) k
    f : F
    c : Complex
    hf : Eq (⇑f) (Function.const UpperHalfPlane c)
    hI : Eq (f (HSMul.hSMul ModularGroup.S UpperHalfPlane.I)) (HMul.hMul (HPow.hPo …
    h2I2 : Eq (f (HSMul.hSMul ModularGroup.S ⟨HMul.hMul 2 Complex.I, ⋯⟩)) (HMul.hM …
    ⊢ Or (Eq k 0) (Eq c 0)
  -/
  simp only [sl_moeb, hf, Function.const, denom_S, coe_mk_subtype] at hI h2I2
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma 1) k
    f : F
    c : Complex
    hf : Eq (⇑f) (Function.const UpperHalfPlane c)
    hI : Eq c (HMul.hMul (HPow.hPow (↑UpperHalfPlane.I) k) c)
    h2I2 : Eq c (HMul.hMul (HPow.hPow (HMul.hMul 2 Complex.I) k) c)
    ⊢ Or (Eq k 0) (Eq c 0)
  -/
  nth_rw 1 [h2I2] at hI
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma 1) k
    f : F
    c : Complex
    hf : Eq (⇑f) (Function.const UpperHalfPlane c)
    hI : Eq (HMul.hMul (HPow.hPow (HMul.hMul 2 Complex.I) k) c) (HMul.hMul (HPow.h …
    h2I2 : Eq c (HMul.hMul (HPow.hPow (HMul.hMul 2 Complex.I) k) c)
    ⊢ Or (Eq k 0) (Eq c 0)
  -/
  simp only [mul_zpow, coe_I, mul_eq_mul_right_iff, mul_left_eq_self₀] at hI
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma 1) k
    f : F
    c : Complex
    hf : Eq (⇑f) (Function.const UpperHalfPlane c)
    h2I2 : Eq c (HMul.hMul (HPow.hPow (HMul.hMul 2 Complex.I) k) c)
    hI : Or (Or (Eq (HPow.hPow 2 k) 1) (Eq (HPow.hPow Complex.I k) 0)) (Eq c 0)
    ⊢ Or (Eq k 0) (Eq c 0)
  -/
  refine hI.imp_left (Or.casesOn · (fun H ↦ ?_) (False.elim ∘ zpow_ne_zero k I_ne_zero))
  rwa [← ofReal_ofNat, ← ofReal_zpow, ← ofReal_one, ofReal_inj,
    zpow_eq_one_iff_right₀ (by norm_num) (by norm_num)] at H


private theorem cuspFunction_eqOn_const_of_nonpos_wt (hk : k ≤ 0) (f : F) :
    Set.EqOn (cuspFunction 1 f) (const ℂ (cuspFunction 1 f 0)) (Metric.ball 0 1) := by
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
    hk : LE.le k 0
    f : F
    ⊢ Set.EqOn (SlashInvariantFormClass.cuspFunction 1 f) (Function.const Complex  …
  -/
  refine eq_const_of_exists_le (fun q hq ↦ ?_) (exp_nonneg (-π)) ?_ (fun q hq ↦ ?_)
    /-
      case refine_1
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      k : Int
      inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
      hk : LE.le k 0
      f : F
      q : Complex
      hq : Membership.mem (Metric.ball 0 1) q
      ⊢ DifferentiableWithinAt Complex (SlashInvariantFormClass.cuspFunction 1 f) (M …
    -/
  · exact (differentiableAt_cuspFunction 1 f (mem_ball_zero_iff.mp hq)).differentiableWithinAt
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      k : Int
      inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
      hk : LE.le k 0
      f : F
      ⊢ LT.lt (Real.exp (Neg.neg Real.pi)) 1
    -/
  · simp only [exp_lt_one_iff, Left.neg_neg_iff, pi_pos]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      k : Int
      inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
      hk : LE.le k 0
      f : F
      q : Complex
      hq : Membership.mem (Metric.ball 0 1) q
      ⊢ Exists fun w => And (Membership.mem (Metric.closedBall 0 (Real.exp (Neg.neg  …
    -/
  · simp only [Metric.mem_closedBall, dist_zero_right]
    /-
      case refine_3
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      k : Int
      inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
      hk : LE.le k 0
      f : F
      q : Complex
      hq : Membership.mem (Metric.ball 0 1) q
      ⊢ Exists fun w => And (LE.le (Norm.norm w) (Real.exp (Neg.neg Real.pi))) (LE.l …
    -/
    rcases eq_or_ne q 0 with rfl | hq'
      /-
        case refine_3.inl
        F : Type u_1
        inst✝¹ : FunLike F UpperHalfPlane Complex
        k : Int
        inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
        hk : LE.le k 0
        f : F
        hq : Membership.mem (Metric.ball 0 1) 0
        ⊢ Exists fun w => And (LE.le (Norm.norm w) (Real.exp (Neg.neg Real.pi))) (LE.l …
      -/
    · refine ⟨0, by simpa only [norm_zero] using exp_nonneg _, le_rfl⟩
      /-
        🎉 no goals
      -/
    · obtain ⟨ξ, hξ, hξ₂⟩ := exists_one_half_le_im_and_norm_le hk f
        ⟨_, im_invQParam_pos_of_abs_lt_one Real.zero_lt_one (mem_ball_zero_iff.mp hq) hq'⟩
      exact ⟨_, abs_qParam_le_of_one_half_le_im hξ,
        by simpa only [← eq_cuspFunction 1 f, Nat.cast_one, coe_mk_subtype,
          qParam_right_inv one_ne_zero hq'] using hξ₂⟩


private theorem levelOne_nonpos_wt_const (hk : k ≤ 0) (f : F) :
    ⇑f = Function.const _ (cuspFunction 1 f 0) := by
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
    hk : LE.le k 0
    f : F
    ⊢ Eq (⇑f) (Function.const UpperHalfPlane (SlashInvariantFormClass.cuspFunction …
  -/
  ext z
  have hQ : 𝕢 1 z ∈ (Metric.ball 0 1) := by
    simpa only [Metric.mem_ball, dist_zero_right, Complex.norm_eq_abs, neg_mul, mul_zero, div_one,
      Real.exp_zero] using (abs_qParam_lt_iff zero_lt_one 0 z.1).mpr z.2
  simpa only [← eq_cuspFunction 1 f z, Nat.cast_one, Function.const_apply] using
    (cuspFunction_eqOn_const_of_nonpos_wt hk f) hQ


lemma levelOne_neg_weight_eq_zero (hk : k < 0) (f : F) : ⇑f = 0 := by
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
    hk : LT.lt k 0
    f : F
    ⊢ Eq (⇑f) 0
  -/
  have hf := levelOne_nonpos_wt_const hk.le f
  /-
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    k : Int
    inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
    hk : LT.lt k 0
    f : F
    hf : Eq (⇑f) (Function.const UpperHalfPlane (SlashInvariantFormClass.cuspFunct …
    ⊢ Eq (⇑f) 0
  -/
  rcases wt_eq_zero_of_eq_const k hf with rfl | hf₀
    /-
      case inl
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      f : F
      hf : Eq (⇑f) (Function.const UpperHalfPlane (SlashInvariantFormClass.cuspFunct …
      inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) 0
      hk : LT.lt 0 0
      ⊢ Eq (⇑f) 0
    -/
  · exact (lt_irrefl _ hk).elim
    /-
      🎉 no goals
    -/
    /-
      case inr
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      k : Int
      inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma 1) k
      hk : LT.lt k 0
      f : F
      hf : Eq (⇑f) (Function.const UpperHalfPlane (SlashInvariantFormClass.cuspFunct …
      hf₀ : Eq (SlashInvariantFormClass.cuspFunction 1 f 0) 0
      ⊢ Eq (⇑f) 0
    -/
  · rw [hf, hf₀, const_zero]
    /-
      🎉 no goals
    -/


lemma levelOne_weight_zero_const [ModularFormClass F Γ(1) 0] (f : F) :
    ∃ c, ⇑f = Function.const _ c :=
  ⟨_, levelOne_nonpos_wt_const le_rfl f⟩


lemma ModularForm.levelOne_weight_zero_rank_one : Module.rank ℂ (ModularForm Γ(1) 0) = 1 := by
  /-
    ⊢ Eq (Module.rank Complex (ModularForm (CongruenceSubgroup.Gamma 1) 0)) 1
  -/
  refine rank_eq_one (const 1) (by simp [DFunLike.ne_iff]) fun g ↦ ?_
  /-
    g : ModularForm (CongruenceSubgroup.Gamma 1) 0
    ⊢ Exists fun c => Eq (HSMul.hSMul c (ModularForm.const 1)) g
  -/
  obtain ⟨c', hc'⟩ := levelOne_weight_zero_const g
  /-
    case intro
    g : ModularForm (CongruenceSubgroup.Gamma 1) 0
    c' : Complex
    hc' : Eq (⇑g) (Function.const UpperHalfPlane c')
    ⊢ Exists fun c => Eq (HSMul.hSMul c (ModularForm.const 1)) g
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma ModularForm.levelOne_neg_weight_rank_zero (hk : k < 0) :
    Module.rank ℂ (ModularForm Γ(1) k) = 0 := by
  /-
    k : Int
    hk : LT.lt k 0
    ⊢ Eq (Module.rank Complex (ModularForm (CongruenceSubgroup.Gamma 1) k)) 0
  -/
  refine rank_eq_zero_iff.mpr fun f ↦ ⟨_, one_ne_zero, ?_⟩
  /-
    k : Int
    hk : LT.lt k 0
    f : ModularForm (CongruenceSubgroup.Gamma 1) k
    ⊢ Eq (HSMul.hSMul 1 f) 0
  -/
  simpa only [one_smul, ← DFunLike.coe_injective.eq_iff] using levelOne_neg_weight_eq_zero hk f
  /-
    🎉 no goals
  -/

