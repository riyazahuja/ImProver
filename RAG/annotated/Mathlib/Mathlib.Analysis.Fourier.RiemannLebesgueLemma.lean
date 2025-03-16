local notation3 "i" => fun (w : V) => (1 / (2 * ‖w‖ ^ 2) : ℝ) • w


/-- Shifting `f` by `(1 / (2 * ‖w‖ ^ 2)) • w` negates the integral in the Riemann-Lebesgue lemma. -/
theorem fourierIntegral_half_period_translate {w : V} (hw : w ≠ 0) :
    (∫ v : V, 𝐞 (-⟪v, w⟫) • f (v + i w)) = -∫ v : V, 𝐞 (-⟪v, w⟫) • f v := by
  have hiw : ⟪i w, w⟫ = 1 / 2 := by
    rw [inner_smul_left, inner_self_eq_norm_sq_to_K, RCLike.ofReal_real_eq_id, id,
      RCLike.conj_to_real, ← div_div, div_mul_cancel₀]
    rwa [Ne, sq_eq_zero_iff, norm_eq_zero]
  have :
    (fun v : V => 𝐞 (-⟪v, w⟫) • f (v + i w)) =
      fun v : V => (fun x : V => -(𝐞 (-⟪x, w⟫) • f x)) (v + i w) := by
    ext1 v
    simp_rw [inner_add_left, hiw, Circle.smul_def, Real.fourierChar_apply, neg_add, mul_add,
      ofReal_add, add_mul, exp_add]
    have : 2 * π * -(1 / 2) = -π := by field_simp; ring
    rw [this, ofReal_neg, neg_mul, exp_neg, exp_pi_mul_I, inv_neg, inv_one, mul_neg_one, neg_smul,
      neg_neg]
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    w : V
    hw : Ne w 0
    hiw : Eq (Inner.inner ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow.h …
    this : Eq (fun v => HSMul.hSMul (Real.fourierChar (Neg.neg (Inner.inner v w))) …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => HSMul. …
  -/
  rw [this]
  -- Porting note:
  -- The next three lines had just been
  -- rw [integral_add_right_eq_self (fun (x : V) ↦ -(𝐞[-⟪x, w⟫]) • f x)
  --       ((fun w ↦ (1 / (2 * ‖w‖ ^ (2 : ℕ))) • w) w)]
  -- Unfortunately now we need to specify `volume`.
  have := integral_add_right_eq_self (μ := volume) (fun (x : V) ↦ -(𝐞 (-⟪x, w⟫) • f x))
    ((fun w ↦ (1 / (2 * ‖w‖ ^ (2 : ℕ))) • w) w)
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    w : V
    hw : Ne w 0
    hiw : Eq (Inner.inner ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow.h …
    this✝ : Eq (fun v => HSMul.hSMul (Real.fourierChar (Neg.neg (Inner.inner v w)) …
    this : Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => N …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => (fun x …
  -/
  rw [this]
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    w : V
    hw : Ne w 0
    hiw : Eq (Inner.inner ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow.h …
    this✝ : Eq (fun v => HSMul.hSMul (Real.fourierChar (Neg.neg (Inner.inner v w)) …
    this : Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => N …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => Neg.ne …
  -/
  simp only [neg_smul, integral_neg]
  /-
    🎉 no goals
  -/


/-- Rewrite the Fourier integral in a form that allows us to use uniform continuity. -/
theorem fourierIntegral_eq_half_sub_half_period_translate {w : V} (hw : w ≠ 0)
          /-
            E : Type u_1
            V : Type u_2
            inst✝⁶ : NormedAddCommGroup E
            inst✝⁵ : NormedSpace Complex E
            f : V → E
            inst✝⁴ : NormedAddCommGroup V
            inst✝³ : MeasurableSpace V
            inst✝² : BorelSpace V
            inst✝¹ : InnerProductSpace Real V
            inst✝ : FiniteDimensional Real V
            w : V
            hw : Ne w 0
            ⊢ MeasureTheory.Measure V
          -/
    (hf : Integrable f) :
          /-
            🎉 no goals
          -/
    ∫ v : V, 𝐞 (-⟪v, w⟫) • f v = (1 / (2 : ℂ)) • ∫ v : V, 𝐞 (-⟪v, w⟫) • (f v - f (v + i w)) := by
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    w : V
    hw : Ne w 0
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => HSMul. …
  -/
  simp_rw [smul_sub]
  rw [integral_sub, fourierIntegral_half_period_translate hw, sub_eq_add_neg, neg_neg, ←
    two_smul ℂ _, ← @smul_assoc _ _ _ _ _ _ (IsScalarTower.left ℂ), smul_eq_mul]
    /-
      E : Type u_1
      V : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      f : V → E
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : FiniteDimensional Real V
      w : V
      hw : Ne w 0
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => HSMul. …
    -/
  · norm_num
    /-
      🎉 no goals
    -/
  exacts [(Real.fourierIntegral_convergent_iff w).2 hf,
    (Real.fourierIntegral_convergent_iff w).2 (hf.comp_add_right _)]


/-- Riemann-Lebesgue Lemma for continuous and compactly-supported functions: the integral
`∫ v, exp (-2 * π * ⟪w, v⟫ * I) • f v` tends to 0 wrt `cocompact V`. Note that this is primarily
of interest as a preparatory step for the more general result
`tendsto_integral_exp_inner_smul_cocompact` in which `f` can be arbitrary. -/
theorem tendsto_integral_exp_inner_smul_cocompact_of_continuous_compact_support (hf1 : Continuous f)
    (hf2 : HasCompactSupport f) :
    Tendsto (fun w : V => ∫ v : V, 𝐞 (-⟪v, w⟫) • f v) (cocompact V) (𝓝 0) := by
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  refine NormedAddCommGroup.tendsto_nhds_zero.mpr fun ε hε => ?_
  suffices ∃ T : ℝ, ∀ w : V, T ≤ ‖w‖ → ‖∫ v : V, 𝐞 (-⟪v, w⟫) • f v‖ < ε by
    simp_rw [← comap_dist_left_atTop_eq_cocompact (0 : V), eventually_comap, eventually_atTop,
      dist_eq_norm', sub_zero]
    exact
      let ⟨T, hT⟩ := this
      ⟨T, fun b hb v hv => hT v (hv.symm ▸ hb)⟩
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun T => ∀ (w : V), LE.le T (Norm.norm w) → LT.lt (Norm.norm (Measure …
  -/
  obtain ⟨R, -, hR_bd⟩ : ∃ R : ℝ, 0 < R ∧ ∀ x : V, R ≤ ‖x‖ → f x = 0 := hf2.exists_pos_le_norm
  /-
    case intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    ⊢ Exists fun T => ∀ (w : V), LE.le T (Norm.norm w) → LT.lt (Norm.norm (Measure …
  -/
  let A := {v : V | ‖v‖ ≤ R + 1}
  have mA : MeasurableSet A := by
    suffices A = Metric.closedBall (0 : V) (R + 1) by
      rw [this]
      exact Metric.isClosed_ball.measurableSet
    simp_rw [A, Metric.closedBall, dist_eq_norm, sub_zero]
  obtain ⟨B, hB_pos, hB_vol⟩ : ∃ B : ℝ≥0, 0 < B ∧ volume A ≤ B := by
    have hc : IsCompact A := by
      simpa only [Metric.closedBall, dist_eq_norm, sub_zero] using isCompact_closedBall (0 : V) _
    let B₀ := volume A
    replace hc : B₀ < ⊤ := hc.measure_lt_top
    refine ⟨B₀.toNNReal + 1, add_pos_of_nonneg_of_pos B₀.toNNReal.coe_nonneg one_pos, ?_⟩
    rw [ENNReal.coe_add, ENNReal.coe_one, ENNReal.coe_toNNReal hc.ne]
    exact le_self_add
  --* Use uniform continuity to choose δ such that `‖x - y‖ < δ` implies `‖f x - f y‖ < ε / B`.
  obtain ⟨δ, hδ1, hδ2⟩ :=
    Metric.uniformContinuous_iff.mp (hf2.uniformContinuous_of_continuous hf1) (ε / B)
      (div_pos hε hB_pos)
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    ⊢ Exists fun T => ∀ (w : V), LE.le T (Norm.norm w) → LT.lt (Norm.norm (Measure …
  -/
  refine ⟨1 / 2 + 1 / (2 * δ), fun w hw_bd => ?_⟩
  have hw_ne : w ≠ 0 := by
    contrapose! hw_bd; rw [hw_bd, norm_zero]
    exact add_pos one_half_pos (one_div_pos.mpr <| mul_pos two_pos hδ1)
  have hw'_nm : ‖i w‖ = 1 / (2 * ‖w‖) := by
    rw [norm_smul, norm_div, Real.norm_of_nonneg (mul_nonneg two_pos.le <| sq_nonneg _), norm_one,
      sq, ← div_div, ← div_div, ← div_div, div_mul_cancel₀ _ (norm_eq_zero.not.mpr hw_ne)]
  --* Rewrite integral in terms of `f v - f (v + w')`.
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    ⊢ LT.lt (Norm.norm (MeasureTheory.integral MeasureTheory.MeasureSpace.volume f …
  -/
  have : ‖(1 / 2 : ℂ)‖ = 2⁻¹ := by norm_num
  rw [fourierIntegral_eq_half_sub_half_period_translate hw_ne
      (hf1.integrable_of_hasCompactSupport hf2),
    norm_smul, this, inv_mul_eq_div, div_lt_iff₀' two_pos]
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    ⊢ LT.lt (Norm.norm (MeasureTheory.integral MeasureTheory.MeasureSpace.volume f …
  -/
  refine lt_of_le_of_lt (norm_integral_le_integral_norm _) ?_
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    ⊢ LT.lt (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun a => Nor …
  -/
  simp_rw [Circle.norm_smul]
  --* Show integral can be taken over A only.
  have int_A : ∫ v : V, ‖f v - f (v + i w)‖ = ∫ v in A, ‖f v - f (v + i w)‖ := by
    refine (setIntegral_eq_integral_of_forall_compl_eq_zero fun v hv => ?_).symm
    dsimp only [A] at hv
    simp only [mem_setOf, not_le] at hv
    rw [hR_bd v _, hR_bd (v + i w) _, sub_zero, norm_zero]
    · rw [← sub_neg_eq_add]
      refine le_trans ?_ (norm_sub_norm_le _ _)
      rw [le_sub_iff_add_le, norm_neg]
      refine le_trans ?_ hv.le
      rw [add_le_add_iff_left, hw'_nm, ← div_div]
      refine (div_le_one <| norm_pos_iff.mpr hw_ne).mpr ?_
      refine le_trans (le_add_of_nonneg_right <| one_div_nonneg.mpr <| ?_) hw_bd
      exact (mul_pos (zero_lt_two' ℝ) hδ1).le
    · exact (le_add_of_nonneg_right zero_le_one).trans hv.le
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    int_A : Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v =>  …
    ⊢ LT.lt (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun a => Nor …
  -/
  rw [int_A]; clear int_A
  --* Bound integral using fact that `‖f v - f (v + w')‖` is small.
  have bdA : ∀ v : V, v ∈ A → ‖‖f v - f (v + i w)‖‖ ≤ ε / B := by
    simp_rw [norm_norm]
    simp_rw [dist_eq_norm] at hδ2
    refine fun x _ => (hδ2 ?_).le
    rw [sub_add_cancel_left, norm_neg, hw'_nm, ← div_div, div_lt_iff₀ (norm_pos_iff.mpr hw_ne), ←
      div_lt_iff₀' hδ1, div_div]
    exact (lt_add_of_pos_left _ one_half_pos).trans_le hw_bd
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
    ⊢ LT.lt (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict A) …
  -/
  have bdA2 := norm_setIntegral_le_of_norm_le_const (hB_vol.trans_lt ENNReal.coe_lt_top) bdA ?_
  /-
    case intro.intro.intro.intro.intro.intro.refine_2
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
    bdA2 : LE.le (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
    ⊢ LT.lt (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict A) …
  -/
  swap
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      E : Type u_1
      V : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      f : V → E
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : FiniteDimensional Real V
      hf1 : Continuous f
      hf2 : HasCompactSupport f
      ε : Real
      hε : GT.gt ε 0
      R : Real
      hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
      A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
      mA : MeasurableSet A
      B : NNReal
      hB_pos : LT.lt 0 B
      hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
      δ : Real
      hδ1 : GT.gt δ 0
      hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
      w : V
      hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
      hw_ne : Ne w 0
      hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
      this : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
      bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => Norm.norm (HSub.hSub (f x) (f ( …
    -/
  · apply Continuous.aestronglyMeasurable
    /-
      case intro.intro.intro.intro.intro.intro.refine_1.hf
      E : Type u_1
      V : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      f : V → E
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : FiniteDimensional Real V
      hf1 : Continuous f
      hf2 : HasCompactSupport f
      ε : Real
      hε : GT.gt ε 0
      R : Real
      hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
      A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
      mA : MeasurableSet A
      B : NNReal
      hB_pos : LT.lt 0 B
      hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
      δ : Real
      hδ1 : GT.gt δ 0
      hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
      w : V
      hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
      hw_ne : Ne w 0
      hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
      this : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
      bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
      ⊢ Continuous fun x => Norm.norm (HSub.hSub (f x) (f (HAdd.hAdd x ((fun w => HS …
    -/
    exact continuous_norm.comp <| hf1.sub <| hf1.comp <| continuous_id'.add continuous_const
    /-
      🎉 no goals
    -/
  have : ‖_‖ = ∫ v : V in A, ‖f v - f (v + i w)‖ :=
    Real.norm_of_nonneg (setIntegral_nonneg mA fun x _ => norm_nonneg _)
  /-
    case intro.intro.intro.intro.intro.intro.refine_2
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this✝ : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
    bdA2 : LE.le (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
    this : Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volum …
    ⊢ LT.lt (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict A) …
  -/
  rw [this] at bdA2
  /-
    case intro.intro.intro.intro.intro.intro.refine_2
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this✝ : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
    bdA2 : LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restri …
    this : Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volum …
    ⊢ LT.lt (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict A) …
  -/
  refine bdA2.trans_lt ?_
  rw [div_mul_eq_mul_div, div_lt_iff₀ (NNReal.coe_pos.mpr hB_pos), mul_comm (2 : ℝ), mul_assoc,
    mul_lt_mul_left hε]
  /-
    case intro.intro.intro.intro.intro.intro.refine_2
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this✝ : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
    bdA2 : LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restri …
    this : Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volum …
    ⊢ LT.lt (MeasureTheory.MeasureSpace.volume A).toReal (HMul.hMul 2 ↑B)
  -/
  refine (ENNReal.toReal_mono ENNReal.coe_ne_top hB_vol).trans_lt ?_
  /-
    case intro.intro.intro.intro.intro.intro.refine_2
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this✝ : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
    bdA2 : LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restri …
    this : Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volum …
    ⊢ LT.lt (↑B).toReal (HMul.hMul 2 ↑B)
  -/
  rw [ENNReal.coe_toReal, two_mul]
  /-
    case intro.intro.intro.intro.intro.intro.refine_2
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hf1 : Continuous f
    hf2 : HasCompactSupport f
    ε : Real
    hε : GT.gt ε 0
    R : Real
    hR_bd : ∀ (x : V), LE.le R (Norm.norm x) → Eq (f x) 0
    A : Set V := setOf fun v => LE.le (Norm.norm v) (HAdd.hAdd R 1)
    mA : MeasurableSet A
    B : NNReal
    hB_pos : LT.lt 0 B
    hB_vol : LE.le (MeasureTheory.MeasureSpace.volume A) ↑B
    δ : Real
    hδ1 : GT.gt δ 0
    hδ2 : ∀ ⦃a b : V⦄, LT.lt (Dist.dist a b) δ → LT.lt (Dist.dist (f a) (f b)) (HD …
    w : V
    hw_bd : LE.le (HAdd.hAdd (1 / 2) (HDiv.hDiv 1 (HMul.hMul 2 δ))) (Norm.norm w)
    hw_ne : Ne w 0
    hw'_nm : Eq (Norm.norm ((fun w => HSMul.hSMul (HDiv.hDiv 1 (HMul.hMul 2 (HPow. …
    this✝ : Eq (Norm.norm (1 / 2)) (Inv.inv 2)
    bdA : ∀ (v : V), Membership.mem A v → LE.le (Norm.norm (Norm.norm (HSub.hSub ( …
    bdA2 : LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restri …
    this : Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volum …
    ⊢ LT.lt (↑B) (HAdd.hAdd ↑B ↑B)
  -/
  exact lt_add_of_pos_left _ hB_pos
  /-
    🎉 no goals
  -/


/-- Riemann-Lebesgue lemma for functions on a real inner-product space: the integral
`∫ v, exp (-2 * π * ⟪w, v⟫ * I) • f v` tends to 0 as `w → ∞`. -/
theorem tendsto_integral_exp_inner_smul_cocompact :
    Tendsto (fun w : V => ∫ v, 𝐞 (-⟪v, w⟫) • f v) (cocompact V) (𝓝 0) := by
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  by_cases hfi : Integrable f; swap
    /-
      case neg
      E : Type u_1
      V : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      f : V → E
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : FiniteDimensional Real V
      hfi : Not (MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume)
      ⊢ Filter.Tendsto (fun w => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
    -/
  · convert tendsto_const_nhds (x := (0 : E)) with w
    /-
      case h.e.h.e'_3.h
      E : Type u_1
      V : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      f : V → E
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : FiniteDimensional Real V
      hfi : Not (MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume)
      w : V
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => HSMul. …
    -/
    apply integral_undef
    /-
      case h.e.h.e'_3.h.h
      E : Type u_1
      V : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      f : V → E
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : FiniteDimensional Real V
      hfi : Not (MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume)
      w : V
      ⊢ Not (MeasureTheory.Integrable (fun a => HSMul.hSMul (Real.fourierChar (Neg.n …
    -/
    rwa [Real.fourierIntegral_convergent_iff]
    /-
      🎉 no goals
    -/
  /-
    case pos
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hfi : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  refine Metric.tendsto_nhds.mpr fun ε hε => ?_
  obtain ⟨g, hg_supp, hfg, hg_cont, -⟩ :=
    hfi.exists_hasCompactSupport_integral_sub_le (div_pos hε two_pos)
  refine
    ((Metric.tendsto_nhds.mp
            (tendsto_integral_exp_inner_smul_cocompact_of_continuous_compact_support hg_cont
              hg_supp))
          _ (div_pos hε two_pos)).mp
      (Eventually.of_forall fun w hI => ?_)
  /-
    case pos.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hfi : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ε : Real
    hε : GT.gt ε 0
    g : V → E
    hg_supp : HasCompactSupport g
    hfg : LE.le (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => …
    hg_cont : Continuous g
    w : V
    hI : LT.lt (Dist.dist (MeasureTheory.integral MeasureTheory.MeasureSpace.volum …
    ⊢ LT.lt (Dist.dist (MeasureTheory.integral MeasureTheory.MeasureSpace.volume f …
  -/
  rw [dist_eq_norm] at hI ⊢
  have : ‖(∫ v, 𝐞 (-⟪v, w⟫) • f v) - ∫ v, 𝐞 (-⟪v, w⟫) • g v‖ ≤ ε / 2 := by
    refine le_trans ?_ hfg
    simp_rw [← integral_sub ((Real.fourierIntegral_convergent_iff w).2 hfi)
      ((Real.fourierIntegral_convergent_iff w).2 (hg_cont.integrable_of_hasCompactSupport hg_supp)),
      ← smul_sub, ← Pi.sub_apply]
    exact VectorFourier.norm_fourierIntegral_le_integral_norm 𝐞 _ bilinFormOfRealInner (f - g) w
  /-
    case pos.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hfi : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ε : Real
    hε : GT.gt ε 0
    g : V → E
    hg_supp : HasCompactSupport g
    hfg : LE.le (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => …
    hg_cont : Continuous g
    w : V
    hI : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.Measure …
    this : LE.le (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.Measu …
    ⊢ LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.MeasureSpa …
  -/
  replace := add_lt_add_of_le_of_lt this hI
  /-
    case pos.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hfi : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ε : Real
    hε : GT.gt ε 0
    g : V → E
    hg_supp : HasCompactSupport g
    hfg : LE.le (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => …
    hg_cont : Continuous g
    w : V
    hI : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.Measure …
    this : LT.lt (HAdd.hAdd (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureT …
    ⊢ LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.MeasureSpa …
  -/
  rw [add_halves] at this
  /-
    case pos.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hfi : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ε : Real
    hε : GT.gt ε 0
    g : V → E
    hg_supp : HasCompactSupport g
    hfg : LE.le (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => …
    hg_cont : Continuous g
    w : V
    hI : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.Measure …
    this : LT.lt (HAdd.hAdd (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureT …
    ⊢ LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.MeasureSpa …
  -/
  refine ((le_of_eq ?_).trans (norm_add_le _ _)).trans_lt this
  /-
    case pos.intro.intro.intro.intro
    E : Type u_1
    V : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    f : V → E
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : FiniteDimensional Real V
    hfi : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ε : Real
    hε : GT.gt ε 0
    g : V → E
    hg_supp : HasCompactSupport g
    hfg : LE.le (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => …
    hg_cont : Continuous g
    w : V
    hI : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.Measure …
    this : LT.lt (HAdd.hAdd (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureT …
    ⊢ Eq (Norm.norm (HSub.hSub (MeasureTheory.integral MeasureTheory.MeasureSpace. …
  -/
  simp only [sub_zero, sub_add_cancel]
  /-
    🎉 no goals
  -/


/-- The Riemann-Lebesgue lemma for functions on `ℝ`. -/
theorem Real.tendsto_integral_exp_smul_cocompact (f : ℝ → E) :
    Tendsto (fun w : ℝ => ∫ v : ℝ, 𝐞 (-(v * w)) • f v) (cocompact ℝ) (𝓝 0) :=
  tendsto_integral_exp_inner_smul_cocompact f


/-- The Riemann-Lebesgue lemma for functions on `ℝ`, formulated via `Real.fourierIntegral`. -/
theorem Real.zero_at_infty_fourierIntegral (f : ℝ → E) : Tendsto (𝓕 f) (cocompact ℝ) (𝓝 0) :=
  tendsto_integral_exp_inner_smul_cocompact f


/-- Riemann-Lebesgue lemma for functions on a finite-dimensional inner-product space, formulated
via dual space. **Do not use** -- it is only a stepping stone to
`tendsto_integral_exp_smul_cocompact` where the inner-product-space structure isn't required. -/
theorem tendsto_integral_exp_smul_cocompact_of_inner_product (μ : Measure V) [μ.IsAddHaarMeasure] :
    Tendsto (fun w : V →L[ℝ] ℝ => ∫ v, 𝐞 (-w v) • f v ∂μ) (cocompact (V →L[ℝ] ℝ)) (𝓝 0) := by
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Complex E
    f : V → E
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : MeasurableSpace V
    inst✝³ : BorelSpace V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral μ fun v => HSMul.hSMul (Real …
  -/
  rw [μ.isAddLeftInvariant_eq_smul volume]
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Complex E
    f : V → E
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : MeasurableSpace V
    inst✝³ : BorelSpace V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral (HSMul.hSMul (μ.addHaarScala …
  -/
  simp_rw [integral_smul_nnreal_measure]
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Complex E
    f : V → E
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : MeasurableSpace V
    inst✝³ : BorelSpace V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Filter.Tendsto (fun w => HSMul.hSMul (μ.addHaarScalarFactor MeasureTheory.Me …
  -/
  rw [← (smul_zero _ : Measure.addHaarScalarFactor μ volume • (0 : E) = 0)]
  /-
    E : Type u_1
    V : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Complex E
    f : V → E
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : MeasurableSpace V
    inst✝³ : BorelSpace V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Filter.Tendsto (fun w => HSMul.hSMul (μ.addHaarScalarFactor MeasureTheory.Me …
  -/
  apply Tendsto.const_smul
  /-
    case hf
    E : Type u_1
    V : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Complex E
    f : V → E
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : MeasurableSpace V
    inst✝³ : BorelSpace V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Filter.Tendsto (fun x => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  let A := (InnerProductSpace.toDual ℝ V).symm
  have : (fun w : V →L[ℝ] ℝ ↦ ∫ v, 𝐞 (-w v) • f v) = (fun w : V ↦ ∫ v, 𝐞 (-⟪v, w⟫) • f v) ∘ A := by
    ext1 w
    congr 1 with v : 1
    rw [← inner_conj_symm, RCLike.conj_to_real, InnerProductSpace.toDual_symm_apply]
  /-
    case hf
    E : Type u_1
    V : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Complex E
    f : V → E
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : MeasurableSpace V
    inst✝³ : BorelSpace V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    A : LinearIsometryEquiv (starRingEnd Real) (NormedSpace.Dual Real V) V := (Inn …
    this : Eq (fun w => MeasureTheory.integral MeasureTheory.MeasureSpace.volume f …
    ⊢ Filter.Tendsto (fun x => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  rw [this]
  exact (tendsto_integral_exp_inner_smul_cocompact f).comp
      A.toHomeomorph.toCocompactMap.cocompact_tendsto'


/-- Riemann-Lebesgue lemma for functions on a finite-dimensional real vector space, formulated via
dual space. -/
theorem tendsto_integral_exp_smul_cocompact (μ : Measure V) [μ.IsAddHaarMeasure] :
    Tendsto (fun w : V →L[ℝ] ℝ => ∫ v, 𝐞 (-w v) • f v ∂μ) (cocompact (V →L[ℝ] ℝ)) (𝓝 0) := by
  -- We have already proved the result for inner-product spaces, formulated in a way which doesn't
  -- refer to the inner product. So we choose an arbitrary inner-product space isomorphic to V
  -- and port the result over from there.
  /-
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral μ fun v => HSMul.hSMul (Real …
  -/
  let V' := EuclideanSpace ℝ (Fin (finrank ℝ V))
  /-
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral μ fun v => HSMul.hSMul (Real …
  -/
  have A : V ≃L[ℝ] V' := toEuclidean
  /-
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
    A : ContinuousLinearEquiv (RingHom.id Real) V V'
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral μ fun v => HSMul.hSMul (Real …
  -/
  borelize V'
  -- various equivs derived from A
  /-
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
    A : ContinuousLinearEquiv (RingHom.id Real) V V'
    this✝¹ : MeasurableSpace V' := borel V'
    this✝ : BorelSpace V'
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral μ fun v => HSMul.hSMul (Real …
  -/
  let Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
  -- isomorphism between duals derived from A
  /-
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
    A : ContinuousLinearEquiv (RingHom.id Real) V V'
    this✝¹ : MeasurableSpace V' := borel V'
    this✝ : BorelSpace V'
    Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral μ fun v => HSMul.hSMul (Real …
  -/
  let Adual : (V →L[ℝ] ℝ) ≃L[ℝ] V' →L[ℝ] ℝ := A.arrowCongrSL (.refl _ _)
  /-
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
    A : ContinuousLinearEquiv (RingHom.id Real) V V'
    this✝¹ : MeasurableSpace V' := borel V'
    this✝ : BorelSpace V'
    Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
    Adual : ContinuousLinearEquiv (RingHom.id Real) (ContinuousLinearMap (RingHom. …
    ⊢ Filter.Tendsto (fun w => MeasureTheory.integral μ fun v => HSMul.hSMul (Real …
  -/
  have : (μ.map Aₘ).IsAddHaarMeasure := A.isAddHaarMeasure_map _
  convert (tendsto_integral_exp_smul_cocompact_of_inner_product (f ∘ A.symm) (μ.map Aₘ)).comp
    Adual.toHomeomorph.toCocompactMap.cocompact_tendsto' with w
  /-
    case h.e'_3.h
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
    A : ContinuousLinearEquiv (RingHom.id Real) V V'
    this✝¹ : MeasurableSpace V' := borel V'
    this✝ : BorelSpace V'
    Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
    Adual : ContinuousLinearEquiv (RingHom.id Real) (ContinuousLinearMap (RingHom. …
    this : (MeasureTheory.Measure.map (⇑Aₘ) μ).IsAddHaarMeasure
    w : ContinuousLinearMap (RingHom.id Real) V Real
    ⊢ Eq (MeasureTheory.integral μ fun v => HSMul.hSMul (Real.fourierChar (Neg.neg …
  -/
  rw [Function.comp_apply, integral_map_equiv]
  /-
    case h.e'_3.h
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
    A : ContinuousLinearEquiv (RingHom.id Real) V V'
    this✝¹ : MeasurableSpace V' := borel V'
    this✝ : BorelSpace V'
    Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
    Adual : ContinuousLinearEquiv (RingHom.id Real) (ContinuousLinearMap (RingHom. …
    this : (MeasureTheory.Measure.map (⇑Aₘ) μ).IsAddHaarMeasure
    w : ContinuousLinearMap (RingHom.id Real) V Real
    ⊢ Eq (MeasureTheory.integral μ fun v => HSMul.hSMul (Real.fourierChar (Neg.neg …
  -/
  congr 1 with v : 1
  /-
    case h.e'_3.h.e_f.h
    E : Type u_1
    V : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    f : V → E
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : TopologicalSpace V
    inst✝⁷ : TopologicalAddGroup V
    inst✝⁶ : T2Space V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : Module Real V
    inst✝² : ContinuousSMul Real V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
    A : ContinuousLinearEquiv (RingHom.id Real) V V'
    this✝¹ : MeasurableSpace V' := borel V'
    this✝ : BorelSpace V'
    Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
    Adual : ContinuousLinearEquiv (RingHom.id Real) (ContinuousLinearMap (RingHom. …
    this : (MeasureTheory.Measure.map (⇑Aₘ) μ).IsAddHaarMeasure
    w : ContinuousLinearMap (RingHom.id Real) V Real
    v : V
    ⊢ Eq (HSMul.hSMul (Real.fourierChar (Neg.neg (w v))) (f v)) (HSMul.hSMul (Real …
  -/
  congr
  · -- Porting note: added `congr_arg`
    /-
      case h.e'_3.h.e_f.h.e_a.h.e_6.h.e_a
      E : Type u_1
      V : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Complex E
      f : V → E
      inst✝⁹ : AddCommGroup V
      inst✝⁸ : TopologicalSpace V
      inst✝⁷ : TopologicalAddGroup V
      inst✝⁶ : T2Space V
      inst✝⁵ : MeasurableSpace V
      inst✝⁴ : BorelSpace V
      inst✝³ : Module Real V
      inst✝² : ContinuousSMul Real V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
      A : ContinuousLinearEquiv (RingHom.id Real) V V'
      this✝¹ : MeasurableSpace V' := borel V'
      this✝ : BorelSpace V'
      Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
      Adual : ContinuousLinearEquiv (RingHom.id Real) (ContinuousLinearMap (RingHom. …
      this : (MeasureTheory.Measure.map (⇑Aₘ) μ).IsAddHaarMeasure
      w : ContinuousLinearMap (RingHom.id Real) V Real
      v : V
      ⊢ Eq (w v) ((Adual.toHomeomorph.toCocompactMap.toFun w) (Aₘ v))
    -/
    apply congr_arg w
    /-
      case h.e'_3.h.e_f.h.e_a.h.e_6.h.e_a
      E : Type u_1
      V : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Complex E
      f : V → E
      inst✝⁹ : AddCommGroup V
      inst✝⁸ : TopologicalSpace V
      inst✝⁷ : TopologicalAddGroup V
      inst✝⁶ : T2Space V
      inst✝⁵ : MeasurableSpace V
      inst✝⁴ : BorelSpace V
      inst✝³ : Module Real V
      inst✝² : ContinuousSMul Real V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
      A : ContinuousLinearEquiv (RingHom.id Real) V V'
      this✝¹ : MeasurableSpace V' := borel V'
      this✝ : BorelSpace V'
      Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
      Adual : ContinuousLinearEquiv (RingHom.id Real) (ContinuousLinearMap (RingHom. …
      this : (MeasureTheory.Measure.map (⇑Aₘ) μ).IsAddHaarMeasure
      w : ContinuousLinearMap (RingHom.id Real) V Real
      v : V
      ⊢ Eq v (↑↑A.symm (Aₘ v))
    -/
    exact (ContinuousLinearEquiv.symm_apply_apply A v).symm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e_f.h.e_a.e_a
      E : Type u_1
      V : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Complex E
      f : V → E
      inst✝⁹ : AddCommGroup V
      inst✝⁸ : TopologicalSpace V
      inst✝⁷ : TopologicalAddGroup V
      inst✝⁶ : T2Space V
      inst✝⁵ : MeasurableSpace V
      inst✝⁴ : BorelSpace V
      inst✝³ : Module Real V
      inst✝² : ContinuousSMul Real V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      V' : Type := EuclideanSpace Real (Fin (Module.finrank Real V))
      A : ContinuousLinearEquiv (RingHom.id Real) V V'
      this✝¹ : MeasurableSpace V' := borel V'
      this✝ : BorelSpace V'
      Aₘ : MeasurableEquiv V V' := A.toHomeomorph.toMeasurableEquiv
      Adual : ContinuousLinearEquiv (RingHom.id Real) (ContinuousLinearMap (RingHom. …
      this : (MeasureTheory.Measure.map (⇑Aₘ) μ).IsAddHaarMeasure
      w : ContinuousLinearMap (RingHom.id Real) V Real
      v : V
      ⊢ Eq v (A.symm (Aₘ v))
    -/
  · exact (ContinuousLinearEquiv.symm_apply_apply A v).symm
    /-
      🎉 no goals
    -/


/-- The Riemann-Lebesgue lemma, formulated in terms of `VectorFourier.fourierIntegral` (with the
pairing in the definition of `fourier_integral` taken to be the canonical pairing between `V` and
its dual space). -/
theorem Real.zero_at_infty_vector_fourierIntegral (μ : Measure V) [μ.IsAddHaarMeasure] :
    Tendsto (VectorFourier.fourierIntegral 𝐞 μ (topDualPairing ℝ V).flip f) (cocompact (V →L[ℝ] ℝ))
      (𝓝 0) :=
  _root_.tendsto_integral_exp_smul_cocompact f μ


