local infixr:25 " →ₛ " => SimpleFunc


theorem nnnorm_approxOn_le [OpensMeasurableSpace E] {f : β → E} (hf : Measurable f) {s : Set E}
    {y₀ : E} (h₀ : y₀ ∈ s) [SeparableSpace s] (x : β) (n : ℕ) :
    ‖approxOn f hf s y₀ h₀ n x - f x‖₊ ≤ ‖f x - y₀‖₊ := by
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f hf s y …
  -/
  have := edist_approxOn_le hf h₀ x n
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    this : LE.le (EDist.edist ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ n)  …
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f hf s y …
  -/
  rw [edist_comm y₀] at this
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    this : LE.le (EDist.edist ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ n)  …
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f hf s y …
  -/
  simp only [edist_nndist, nndist_eq_nnnorm] at this
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    this : LE.le ↑(NNNorm.nnnorm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f  …
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f hf s y …
  -/
  exact mod_cast this
  /-
    🎉 no goals
  -/


theorem norm_approxOn_y₀_le [OpensMeasurableSpace E] {f : β → E} (hf : Measurable f) {s : Set E}
    {y₀ : E} (h₀ : y₀ ∈ s) [SeparableSpace s] (x : β) (n : ℕ) :
    ‖approxOn f hf s y₀ h₀ n x - y₀‖ ≤ ‖f x - y₀‖ + ‖f x - y₀‖ := by
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    ⊢ LE.le (Norm.norm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ …
  -/
  have := edist_approxOn_y0_le hf h₀ x n
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    this : LE.le (EDist.edist y₀ ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀  …
    ⊢ LE.le (Norm.norm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ …
  -/
  repeat rw [edist_comm y₀, edist_eq_coe_nnnorm_sub] at this
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    this : LE.le (↑(NNNorm.nnnorm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f …
    ⊢ LE.le (Norm.norm (HSub.hSub ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ …
  -/
  exact mod_cast this
  /-
    🎉 no goals
  -/


theorem norm_approxOn_zero_le [OpensMeasurableSpace E] {f : β → E} (hf : Measurable f) {s : Set E}
    (h₀ : (0 : E) ∈ s) [SeparableSpace s] (x : β) (n : ℕ) :
    ‖approxOn f hf s 0 h₀ n x‖ ≤ ‖f x‖ + ‖f x‖ := by
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    h₀ : Membership.mem s 0
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    ⊢ LE.le (Norm.norm ((MeasureTheory.SimpleFunc.approxOn f hf s 0 h₀ n) x)) (HAd …
  -/
  have := edist_approxOn_y0_le hf h₀ x n
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    h₀ : Membership.mem s 0
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    this : LE.le (EDist.edist 0 ((MeasureTheory.SimpleFunc.approxOn f hf s 0 h₀ n) …
    ⊢ LE.le (Norm.norm ((MeasureTheory.SimpleFunc.approxOn f hf s 0 h₀ n) x)) (HAd …
  -/
  simp only [edist_comm (0 : E), edist_eq_coe_nnnorm] at this
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    h₀ : Membership.mem s 0
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    n : Nat
    this : LE.le (↑(NNNorm.nnnorm ((MeasureTheory.SimpleFunc.approxOn f hf s 0 h₀  …
    ⊢ LE.le (Norm.norm ((MeasureTheory.SimpleFunc.approxOn f hf s 0 h₀ n) x)) (HAd …
  -/
  exact mod_cast this
  /-
    🎉 no goals
  -/


theorem tendsto_approxOn_Lp_eLpNorm [OpensMeasurableSpace E] {f : β → E} (hf : Measurable f)
    {s : Set E} {y₀ : E} (h₀ : y₀ ∈ s) [SeparableSpace s] (hp_ne_top : p ≠ ∞) {μ : Measure β}
    (hμ : ∀ᵐ x ∂μ, f x ∈ closure s) (hi : eLpNorm (fun x => f x - y₀) p μ < ∞) :
    Tendsto (fun n => eLpNorm (⇑(approxOn f hf s y₀ h₀ n) - f) p μ) atTop (𝓝 0) := by
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    p : ENNReal
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hp_ne_top : Ne p Top.top
    μ : MeasureTheory.Measure β
    hμ : Filter.Eventually (fun x => Membership.mem (closure s) (f x)) (MeasureThe …
    hi : LT.lt (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) y₀) p μ) Top.top
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheory.S …
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      β : Type u_2
      E : Type u_4
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace E
      inst✝² : NormedAddCommGroup E
      p : ENNReal
      inst✝¹ : OpensMeasurableSpace E
      f : β → E
      hf : Measurable f
      s : Set E
      y₀ : E
      h₀ : Membership.mem s y₀
      inst✝ : TopologicalSpace.SeparableSpace ↑s
      hp_ne_top : Ne p Top.top
      μ : MeasureTheory.Measure β
      hμ : Filter.Eventually (fun x => Membership.mem (closure s) (f x)) (MeasureThe …
      hi : LT.lt (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) y₀) p μ) Top.top
      hp_zero : Eq p 0
      ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheory.S …
    -/
  · simpa only [hp_zero, eLpNorm_exponent_zero] using tendsto_const_nhds
    /-
      🎉 no goals
    -/
  /-
    case neg
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    p : ENNReal
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hp_ne_top : Ne p Top.top
    μ : MeasureTheory.Measure β
    hμ : Filter.Eventually (fun x => Membership.mem (closure s) (f x)) (MeasureThe …
    hi : LT.lt (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) y₀) p μ) Top.top
    hp_zero : Not (Eq p 0)
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheory.S …
  -/
  have hp : 0 < p.toReal := toReal_pos hp_zero hp_ne_top
  suffices
      Tendsto (fun n => ∫⁻ x, (‖approxOn f hf s y₀ h₀ n x - f x‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) atTop
        (𝓝 0) by
    simp only [eLpNorm_eq_lintegral_rpow_nnnorm hp_zero hp_ne_top]
    convert continuous_rpow_const.continuousAt.tendsto.comp this
    simp [zero_rpow_of_pos (_root_.inv_pos.mpr hp)]
  -- We simply check the conditions of the Dominated Convergence Theorem:
  -- (1) The function "`p`-th power of distance between `f` and the approximation" is measurable
  have hF_meas :
    ∀ n, Measurable fun x => (‖approxOn f hf s y₀ h₀ n x - f x‖₊ : ℝ≥0∞) ^ p.toReal := by
    simpa only [← edist_eq_coe_nnnorm_sub] using fun n =>
      (approxOn f hf s y₀ h₀ n).measurable_bind (fun y x => edist y (f x) ^ p.toReal) fun y =>
        (measurable_edist_right.comp hf).pow_const p.toReal
  -- (2) The functions "`p`-th power of distance between `f` and the approximation" are uniformly
  -- bounded, at any given point, by `fun x => ‖f x - y₀‖ ^ p.toReal`
  have h_bound :
    ∀ n, (fun x => (‖approxOn f hf s y₀ h₀ n x - f x‖₊ : ℝ≥0∞) ^ p.toReal) ≤ᵐ[μ] fun x =>
        (‖f x - y₀‖₊ : ℝ≥0∞) ^ p.toReal :=
    fun n =>
    Eventually.of_forall fun x =>
      rpow_le_rpow (coe_mono (nnnorm_approxOn_le hf h₀ x n)) toReal_nonneg
  -- (3) The bounding function `fun x => ‖f x - y₀‖ ^ p.toReal` has finite integral
  have h_fin : (∫⁻ a : β, (‖f a - y₀‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) ≠ ⊤ :=
    (lintegral_rpow_nnnorm_lt_top_of_eLpNorm_lt_top hp_zero hp_ne_top hi).ne
  -- (4) The functions "`p`-th power of distance between `f` and the approximation" tend pointwise
  -- to zero
  have h_lim :
    ∀ᵐ a : β ∂μ,
      Tendsto (fun n => (‖approxOn f hf s y₀ h₀ n a - f a‖₊ : ℝ≥0∞) ^ p.toReal) atTop (𝓝 0) := by
    filter_upwards [hμ] with a ha
    have : Tendsto (fun n => (approxOn f hf s y₀ h₀ n) a - f a) atTop (𝓝 (f a - f a)) :=
      (tendsto_approxOn hf h₀ ha).sub tendsto_const_nhds
    convert continuous_rpow_const.continuousAt.tendsto.comp (tendsto_coe.mpr this.nnnorm)
    simp [zero_rpow_of_pos hp]
  -- Then we apply the Dominated Convergence Theorem
  /-
    case neg
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    p : ENNReal
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    hf : Measurable f
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hp_ne_top : Ne p Top.top
    μ : MeasureTheory.Measure β
    hμ : Filter.Eventually (fun x => Membership.mem (closure s) (f x)) (MeasureThe …
    hi : LT.lt (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) y₀) p μ) Top.top
    hp_zero : Not (Eq p 0)
    hp : LT.lt 0 p.toReal
    hF_meas : ∀ (n : Nat), Measurable fun x => HPow.hPow (↑(NNNorm.nnnorm (HSub.hS …
    h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (fun x => HPow.hPow ( …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (HSu …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => HPow.hPow (↑(NNNo …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNN …
  -/
  simpa using tendsto_lintegral_of_dominated_convergence _ hF_meas h_bound h_fin h_lim
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias tendsto_approxOn_Lp_snorm := tendsto_approxOn_Lp_eLpNorm


theorem memℒp_approxOn [BorelSpace E] {f : β → E} {μ : Measure β} (fmeas : Measurable f)
    (hf : Memℒp f p μ) {s : Set E} {y₀ : E} (h₀ : y₀ ∈ s) [SeparableSpace s]
    (hi₀ : Memℒp (fun _ => y₀) p μ) (n : ℕ) : Memℒp (approxOn f fmeas s y₀ h₀ n) p μ := by
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    p : ENNReal
    inst✝¹ : BorelSpace E
    f : β → E
    μ : MeasureTheory.Measure β
    fmeas : Measurable f
    hf : MeasureTheory.Memℒp f p μ
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hi₀ : MeasureTheory.Memℒp (fun x => y₀) p μ
    n : Nat
    ⊢ MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.approxOn f fmeas s y₀ h₀ n)) …
  -/
  refine ⟨(approxOn f fmeas s y₀ h₀ n).aestronglyMeasurable, ?_⟩
  suffices eLpNorm (fun x => approxOn f fmeas s y₀ h₀ n x - y₀) p μ < ⊤ by
    have : Memℒp (fun x => approxOn f fmeas s y₀ h₀ n x - y₀) p μ :=
      ⟨(approxOn f fmeas s y₀ h₀ n - const β y₀).aestronglyMeasurable, this⟩
    convert eLpNorm_add_lt_top this hi₀
    ext x
    simp
  have hf' : Memℒp (fun x => ‖f x - y₀‖) p μ := by
    have h_meas : Measurable fun x => ‖f x - y₀‖ := by
      simp only [← dist_eq_norm]
      exact (continuous_id.dist continuous_const).measurable.comp fmeas
    refine ⟨h_meas.aemeasurable.aestronglyMeasurable, ?_⟩
    rw [eLpNorm_norm]
    convert eLpNorm_add_lt_top hf hi₀.neg with x
    simp [sub_eq_add_neg]
  have : ∀ᵐ x ∂μ, ‖approxOn f fmeas s y₀ h₀ n x - y₀‖ ≤ ‖‖f x - y₀‖ + ‖f x - y₀‖‖ := by
    filter_upwards with x
    convert norm_approxOn_y₀_le fmeas h₀ x n using 1
    rw [Real.norm_eq_abs, abs_of_nonneg]
    positivity
  calc
    eLpNorm (fun x => approxOn f fmeas s y₀ h₀ n x - y₀) p μ ≤
        eLpNorm (fun x => ‖f x - y₀‖ + ‖f x - y₀‖) p μ :=
      eLpNorm_mono_ae this
    _ < ⊤ := eLpNorm_add_lt_top hf' hf'


theorem tendsto_approxOn_range_Lp_eLpNorm [BorelSpace E] {f : β → E} (hp_ne_top : p ≠ ∞)
    {μ : Measure β} (fmeas : Measurable f) [SeparableSpace (range f ∪ {0} : Set E)]
    (hf : eLpNorm f p μ < ∞) :
                                                                        /-
                                                                          α : Type u_1
                                                                          β : Type u_2
                                                                          ι : Type u_3
                                                                          E : Type u_4
                                                                          F : Type u_5
                                                                          𝕜 : Type u_6
                                                                          inst✝⁵ : MeasurableSpace β
                                                                          inst✝⁴ : MeasurableSpace E
                                                                          inst✝³ : NormedAddCommGroup E
                                                                          inst✝² : NormedAddCommGroup F
                                                                          q : Real
                                                                          p : ENNReal
                                                                          inst✝¹ : BorelSpace E
                                                                          f : β → E
                                                                          hp_ne_top : Ne p Top.top
                                                                          μ : MeasureTheory.Measure β
                                                                          fmeas : Measurable f
                                                                          inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
                                                                          hf : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
                                                                          n : Nat
                                                                          ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) 0
                                                                        -/
    Tendsto (fun n => eLpNorm (⇑(approxOn f fmeas (range f ∪ {0}) 0 (by simp) n) - f) p μ)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
      atTop (𝓝 0) := by
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    p : ENNReal
    inst✝¹ : BorelSpace E
    f : β → E
    hp_ne_top : Ne p Top.top
    μ : MeasureTheory.Measure β
    fmeas : Measurable f
    inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
    hf : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheory.S …
  -/
  refine tendsto_approxOn_Lp_eLpNorm fmeas _ hp_ne_top ?_ ?_
    /-
      case refine_1
      β : Type u_2
      E : Type u_4
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace E
      inst✝² : NormedAddCommGroup E
      p : ENNReal
      inst✝¹ : BorelSpace E
      f : β → E
      hp_ne_top : Ne p Top.top
      μ : MeasureTheory.Measure β
      fmeas : Measurable f
      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
      hf : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
      ⊢ Filter.Eventually (fun x => Membership.mem (closure (Union.union (Set.range  …
    -/
  · filter_upwards with x using subset_closure (by simp)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      β : Type u_2
      E : Type u_4
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace E
      inst✝² : NormedAddCommGroup E
      p : ENNReal
      inst✝¹ : BorelSpace E
      f : β → E
      hp_ne_top : Ne p Top.top
      μ : MeasureTheory.Measure β
      fmeas : Measurable f
      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
      hf : LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HSub.hSub (f x) 0) p μ) Top.top
    -/
  · simpa using hf
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias tendsto_approxOn_range_Lp_snorm := tendsto_approxOn_range_Lp_eLpNorm


theorem memℒp_approxOn_range [BorelSpace E] {f : β → E} {μ : Measure β} (fmeas : Measurable f)
    [SeparableSpace (range f ∪ {0} : Set E)] (hf : Memℒp f p μ) (n : ℕ) :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    ι : Type u_3
                                                    E : Type u_4
                                                    F : Type u_5
                                                    𝕜 : Type u_6
                                                    inst✝⁵ : MeasurableSpace β
                                                    inst✝⁴ : MeasurableSpace E
                                                    inst✝³ : NormedAddCommGroup E
                                                    inst✝² : NormedAddCommGroup F
                                                    q : Real
                                                    p : ENNReal
                                                    inst✝¹ : BorelSpace E
                                                    f : β → E
                                                    μ : MeasureTheory.Measure β
                                                    fmeas : Measurable f
                                                    inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
                                                    hf : MeasureTheory.Memℒp f p μ
                                                    n : Nat
                                                    ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) 0
                                                  -/
    Memℒp (approxOn f fmeas (range f ∪ {0}) 0 (by simp) n) p μ :=
                                                  /-
                                                    🎉 no goals
                                                  -/
                                        /-
                                          β : Type u_2
                                          E : Type u_4
                                          inst✝⁴ : MeasurableSpace β
                                          inst✝³ : MeasurableSpace E
                                          inst✝² : NormedAddCommGroup E
                                          p : ENNReal
                                          inst✝¹ : BorelSpace E
                                          f : β → E
                                          μ : MeasureTheory.Measure β
                                          fmeas : Measurable f
                                          inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
                                          hf : MeasureTheory.Memℒp f p μ
                                          n : Nat
                                          ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) 0
                                        -/
  memℒp_approxOn fmeas hf (y₀ := 0) (by simp) zero_memℒp n
                                        /-
                                          🎉 no goals
                                        -/


theorem tendsto_approxOn_range_Lp [BorelSpace E] {f : β → E} [hp : Fact (1 ≤ p)] (hp_ne_top : p ≠ ∞)
    {μ : Measure β} (fmeas : Measurable f) [SeparableSpace (range f ∪ {0} : Set E)]
    (hf : Memℒp f p μ) :
    Tendsto
      (fun n =>
                                                                                       /-
                                                                                         α : Type u_1
                                                                                         β : Type u_2
                                                                                         ι : Type u_3
                                                                                         E : Type u_4
                                                                                         F : Type u_5
                                                                                         𝕜 : Type u_6
                                                                                         inst✝⁵ : MeasurableSpace β
                                                                                         inst✝⁴ : MeasurableSpace E
                                                                                         inst✝³ : NormedAddCommGroup E
                                                                                         inst✝² : NormedAddCommGroup F
                                                                                         q : Real
                                                                                         p : ENNReal
                                                                                         inst✝¹ : BorelSpace E
                                                                                         f : β → E
                                                                                         hp : Fact (LE.le 1 p)
                                                                                         hp_ne_top : Ne p Top.top
                                                                                         μ : MeasureTheory.Measure β
                                                                                         fmeas : Measurable f
                                                                                         inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
                                                                                         hf : MeasureTheory.Memℒp f p μ
                                                                                         n : Nat
                                                                                         ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) 0
                                                                                       -/
        (memℒp_approxOn_range fmeas hf n).toLp (approxOn f fmeas (range f ∪ {0}) 0 (by simp) n))
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
      atTop (𝓝 (hf.toLp f)) := by
  simpa only [Lp.tendsto_Lp_iff_tendsto_ℒp''] using
    tendsto_approxOn_range_Lp_eLpNorm hp_ne_top fmeas hf.2


/-- Any function in `ℒp` can be approximated by a simple function if `p < ∞`. -/
theorem _root_.MeasureTheory.Memℒp.exists_simpleFunc_eLpNorm_sub_lt {E : Type*}
    [NormedAddCommGroup E] {f : β → E} {μ : Measure β} (hf : Memℒp f p μ) (hp_ne_top : p ≠ ∞)
    {ε : ℝ≥0∞} (hε : ε ≠ 0) : ∃ g : β →ₛ E, eLpNorm (f - ⇑g) p μ < ε ∧ Memℒp g p μ := by
  /-
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    p : ENNReal
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f : β → E
    μ : MeasureTheory.Measure β
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_top : Ne p Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun g => And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f ⇑g) p μ) ε) ( …
  -/
  borelize E
  /-
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    p : ENNReal
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f : β → E
    μ : MeasureTheory.Measure β
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_top : Ne p Top.top
    ε : ENNReal
    hε : Ne ε 0
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ Exists fun g => And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f ⇑g) p μ) ε) ( …
  -/
  let f' := hf.1.mk f
  /-
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    p : ENNReal
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f : β → E
    μ : MeasureTheory.Measure β
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_top : Ne p Top.top
    ε : ENNReal
    hε : Ne ε 0
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
    ⊢ Exists fun g => And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f ⇑g) p μ) ε) ( …
  -/
  rsuffices ⟨g, hg, g_mem⟩ : ∃ g : β →ₛ E, eLpNorm (f' - ⇑g) p μ < ε ∧ Memℒp g p μ
    /-
      case intro.intro
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      p : ENNReal
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f : β → E
      μ : MeasureTheory.Measure β
      hf : MeasureTheory.Memℒp f p μ
      hp_ne_top : Ne p Top.top
      ε : ENNReal
      hε : Ne ε 0
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
      g : MeasureTheory.SimpleFunc β E
      hg : LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε
      g_mem : MeasureTheory.Memℒp (⇑g) p μ
      ⊢ Exists fun g => And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f ⇑g) p μ) ε) ( …
    -/
  · refine ⟨g, ?_, g_mem⟩
    /-
      case intro.intro
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      p : ENNReal
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f : β → E
      μ : MeasureTheory.Measure β
      hf : MeasureTheory.Memℒp f p μ
      hp_ne_top : Ne p Top.top
      ε : ENNReal
      hε : Ne ε 0
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
      g : MeasureTheory.SimpleFunc β E
      hg : LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε
      g_mem : MeasureTheory.Memℒp (⇑g) p μ
      ⊢ LT.lt (MeasureTheory.eLpNorm (HSub.hSub f ⇑g) p μ) ε
    -/
    suffices eLpNorm (f - ⇑g) p μ = eLpNorm (f' - ⇑g) p μ by rwa [this]
    /-
      case intro.intro
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      p : ENNReal
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f : β → E
      μ : MeasureTheory.Measure β
      hf : MeasureTheory.Memℒp f p μ
      hp_ne_top : Ne p Top.top
      ε : ENNReal
      hε : Ne ε 0
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
      g : MeasureTheory.SimpleFunc β E
      hg : LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε
      g_mem : MeasureTheory.Memℒp (⇑g) p μ
      ⊢ Eq (MeasureTheory.eLpNorm (HSub.hSub f ⇑g) p μ) (MeasureTheory.eLpNorm (HSub …
    -/
    apply eLpNorm_congr_ae
    /-
      case intro.intro.hfg
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      p : ENNReal
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f : β → E
      μ : MeasureTheory.Measure β
      hf : MeasureTheory.Memℒp f p μ
      hp_ne_top : Ne p Top.top
      ε : ENNReal
      hε : Ne ε 0
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
      g : MeasureTheory.SimpleFunc β E
      hg : LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε
      g_mem : MeasureTheory.Memℒp (⇑g) p μ
      ⊢ (MeasureTheory.ae μ).EventuallyEq (HSub.hSub f ⇑g) (HSub.hSub f' ⇑g)
    -/
    filter_upwards [hf.1.ae_eq_mk] with x hx
    /-
      case h
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      p : ENNReal
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f : β → E
      μ : MeasureTheory.Measure β
      hf : MeasureTheory.Memℒp f p μ
      hp_ne_top : Ne p Top.top
      ε : ENNReal
      hε : Ne ε 0
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
      g : MeasureTheory.SimpleFunc β E
      hg : LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε
      g_mem : MeasureTheory.Memℒp (⇑g) p μ
      x : β
      hx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f ⋯ x)
      ⊢ Eq (HSub.hSub f (⇑g) x) (HSub.hSub f' (⇑g) x)
    -/
    simpa only [Pi.sub_apply, sub_left_inj] using hx
    /-
      🎉 no goals
    -/
  /-
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    p : ENNReal
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f : β → E
    μ : MeasureTheory.Measure β
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_top : Ne p Top.top
    ε : ENNReal
    hε : Ne ε 0
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
    ⊢ Exists fun g => And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε)  …
  -/
  have hf' : Memℒp f' p μ := hf.ae_eq hf.1.ae_eq_mk
  /-
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    p : ENNReal
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f : β → E
    μ : MeasureTheory.Measure β
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_top : Ne p Top.top
    ε : ENNReal
    hε : Ne ε 0
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
    hf' : MeasureTheory.Memℒp f' p μ
    ⊢ Exists fun g => And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε)  …
  -/
  have f'meas : Measurable f' := hf.1.measurable_mk
  have : SeparableSpace (range f' ∪ {0} : Set E) :=
    StronglyMeasurable.separableSpace_range_union_singleton hf.1.stronglyMeasurable_mk
  rcases ((tendsto_approxOn_range_Lp_eLpNorm hp_ne_top f'meas hf'.2).eventually <|
    gt_mem_nhds hε.bot_lt).exists with ⟨n, hn⟩
  /-
    case intro
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    p : ENNReal
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f : β → E
    μ : MeasureTheory.Measure β
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_top : Ne p Top.top
    ε : ENNReal
    hε : Ne ε 0
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
    hf' : MeasureTheory.Memℒp f' p μ
    f'meas : Measurable f'
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f') (Singleton …
    n : Nat
    hn : LT.lt (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheory.SimpleFunc.appro …
    ⊢ Exists fun g => And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε)  …
  -/
  rw [← eLpNorm_neg, neg_sub] at hn
  /-
    case intro
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    p : ENNReal
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f : β → E
    μ : MeasureTheory.Measure β
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_top : Ne p Top.top
    ε : ENNReal
    hε : Ne ε 0
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f' : β → E := MeasureTheory.AEStronglyMeasurable.mk f ⋯
    hf' : MeasureTheory.Memℒp f' p μ
    f'meas : Measurable f'
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f') (Singleton …
    n : Nat
    hn : LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑(MeasureTheory.SimpleFunc.app …
    ⊢ Exists fun g => And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f' ⇑g) p μ) ε)  …
  -/
  exact ⟨_, hn, memℒp_approxOn_range f'meas hf' _⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias _root_.MeasureTheory.Memℒp.exists_simpleFunc_snorm_sub_lt :=
  _root_.MeasureTheory.Memℒp.exists_simpleFunc_eLpNorm_sub_lt


theorem tendsto_approxOn_L1_nnnorm [OpensMeasurableSpace E] {f : β → E} (hf : Measurable f)
    {s : Set E} {y₀ : E} (h₀ : y₀ ∈ s) [SeparableSpace s] {μ : Measure β}
    (hμ : ∀ᵐ x ∂μ, f x ∈ closure s) (hi : HasFiniteIntegral (fun x => f x - y₀) μ) :
    Tendsto (fun n => ∫⁻ x, ‖approxOn f hf s y₀ h₀ n x - f x‖₊ ∂μ) atTop (𝓝 0) := by
  simpa [eLpNorm_one_eq_lintegral_nnnorm] using
    tendsto_approxOn_Lp_eLpNorm hf h₀ one_ne_top hμ
      (by simpa [eLpNorm_one_eq_lintegral_nnnorm] using hi)


theorem integrable_approxOn [BorelSpace E] {f : β → E} {μ : Measure β} (fmeas : Measurable f)
    (hf : Integrable f μ) {s : Set E} {y₀ : E} (h₀ : y₀ ∈ s) [SeparableSpace s]
    (hi₀ : Integrable (fun _ => y₀) μ) (n : ℕ) : Integrable (approxOn f fmeas s y₀ h₀ n) μ := by
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    f : β → E
    μ : MeasureTheory.Measure β
    fmeas : Measurable f
    hf : MeasureTheory.Integrable f μ
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hi₀ : MeasureTheory.Integrable (fun x => y₀) μ
    n : Nat
    ⊢ MeasureTheory.Integrable (⇑(MeasureTheory.SimpleFunc.approxOn f fmeas s y₀ h …
  -/
  rw [← memℒp_one_iff_integrable] at hf hi₀ ⊢
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    f : β → E
    μ : MeasureTheory.Measure β
    fmeas : Measurable f
    hf : MeasureTheory.Memℒp f 1 μ
    s : Set E
    y₀ : E
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hi₀ : MeasureTheory.Memℒp (fun x => y₀) 1 μ
    n : Nat
    ⊢ MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.approxOn f fmeas s y₀ h₀ n)) …
  -/
  exact memℒp_approxOn fmeas hf h₀ hi₀ n
  /-
    🎉 no goals
  -/


theorem tendsto_approxOn_range_L1_nnnorm [OpensMeasurableSpace E] {f : β → E} {μ : Measure β}
    [SeparableSpace (range f ∪ {0} : Set E)] (fmeas : Measurable f) (hf : Integrable f μ) :
                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      ι : Type u_3
                                                                      E : Type u_4
                                                                      F : Type u_5
                                                                      𝕜 : Type u_6
                                                                      inst✝⁴ : MeasurableSpace β
                                                                      inst✝³ : MeasurableSpace E
                                                                      inst✝² : NormedAddCommGroup E
                                                                      inst✝¹ : OpensMeasurableSpace E
                                                                      f : β → E
                                                                      μ : MeasureTheory.Measure β
                                                                      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
                                                                      fmeas : Measurable f
                                                                      hf : MeasureTheory.Integrable f μ
                                                                      n : Nat
                                                                      x : β
                                                                      ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) 0
                                                                    -/
    Tendsto (fun n => ∫⁻ x, ‖approxOn f fmeas (range f ∪ {0}) 0 (by simp) n x - f x‖₊ ∂μ) atTop
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
      (𝓝 0) := by
  /-
    β : Type u_2
    E : Type u_4
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : OpensMeasurableSpace E
    f : β → E
    μ : MeasureTheory.Measure β
    inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
    fmeas : Measurable f
    hf : MeasureTheory.Integrable f μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm  …
  -/
  apply tendsto_approxOn_L1_nnnorm fmeas
    /-
      case hμ
      β : Type u_2
      E : Type u_4
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace E
      inst✝² : NormedAddCommGroup E
      inst✝¹ : OpensMeasurableSpace E
      f : β → E
      μ : MeasureTheory.Measure β
      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
      fmeas : Measurable f
      hf : MeasureTheory.Integrable f μ
      ⊢ Filter.Eventually (fun x => Membership.mem (closure (Union.union (Set.range  …
    -/
  · filter_upwards with x using subset_closure (by simp)
    /-
      🎉 no goals
    -/
    /-
      case hi
      β : Type u_2
      E : Type u_4
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace E
      inst✝² : NormedAddCommGroup E
      inst✝¹ : OpensMeasurableSpace E
      f : β → E
      μ : MeasureTheory.Measure β
      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
      fmeas : Measurable f
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.HasFiniteIntegral (fun x => HSub.hSub (f x) 0) μ
    -/
  · simpa using hf.2
    /-
      🎉 no goals
    -/


theorem integrable_approxOn_range [BorelSpace E] {f : β → E} {μ : Measure β} (fmeas : Measurable f)
    [SeparableSpace (range f ∪ {0} : Set E)] (hf : Integrable f μ) (n : ℕ) :
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         ι : Type u_3
                                                         E : Type u_4
                                                         F : Type u_5
                                                         𝕜 : Type u_6
                                                         inst✝⁴ : MeasurableSpace β
                                                         inst✝³ : MeasurableSpace E
                                                         inst✝² : NormedAddCommGroup E
                                                         inst✝¹ : BorelSpace E
                                                         f : β → E
                                                         μ : MeasureTheory.Measure β
                                                         fmeas : Measurable f
                                                         inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
                                                         hf : MeasureTheory.Integrable f μ
                                                         n : Nat
                                                         ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) 0
                                                       -/
    Integrable (approxOn f fmeas (range f ∪ {0}) 0 (by simp) n) μ :=
                                                       /-
                                                         🎉 no goals
                                                       -/
  integrable_approxOn fmeas hf _ (integrable_zero _ _ _) n


theorem exists_forall_norm_le (f : α →ₛ F) : ∃ C, ∀ x, ‖f x‖ ≤ C :=
  exists_forall_le (f.map fun x => ‖x‖)


theorem memℒp_zero (f : α →ₛ E) (μ : Measure α) : Memℒp f 0 μ :=
  memℒp_zero_iff_aestronglyMeasurable.mpr f.aestronglyMeasurable


theorem memℒp_top (f : α →ₛ E) (μ : Measure α) : Memℒp f ∞ μ :=
  let ⟨C, hfC⟩ := f.exists_forall_norm_le
  memℒp_top_of_bound f.aestronglyMeasurable C <| Eventually.of_forall hfC


protected theorem eLpNorm'_eq {p : ℝ} (f : α →ₛ F) (μ : Measure α) :
    eLpNorm' f p μ = (∑ y ∈ f.range, (‖y‖₊ : ℝ≥0∞) ^ p * μ (f ⁻¹' {y})) ^ (1 / p) := by
  have h_map : (fun a => (‖f a‖₊ : ℝ≥0∞) ^ p) = f.map fun a : F => (‖a‖₊ : ℝ≥0∞) ^ p := by
    simp; rfl
  /-
    α : Type u_1
    F : Type u_5
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup F
    p : Real
    f : MeasureTheory.SimpleFunc α F
    μ : MeasureTheory.Measure α
    h_map : Eq (fun a => HPow.hPow (↑(NNNorm.nnnorm (f a))) p) ⇑(MeasureTheory.Sim …
    ⊢ Eq (MeasureTheory.eLpNorm' (⇑f) p μ) (HPow.hPow (f.range.sum fun y => HMul.h …
  -/
  rw [eLpNorm'_eq_lintegral_nnnorm, h_map, lintegral_eq_lintegral, map_lintegral]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
protected alias snorm'_eq := SimpleFunc.eLpNorm'_eq


theorem measure_preimage_lt_top_of_memℒp (hp_pos : p ≠ 0) (hp_ne_top : p ≠ ∞) (f : α →ₛ E)
    (hf : Memℒp f p μ) (y : E) (hy_ne : y ≠ 0) : μ (f ⁻¹' {y}) < ∞ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Memℒp (⇑f) p μ
    y : E
    hy_ne : Ne y 0
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) Top.top
  -/
  have hp_pos_real : 0 < p.toReal := ENNReal.toReal_pos hp_pos hp_ne_top
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Memℒp (⇑f) p μ
    y : E
    hy_ne : Ne y 0
    hp_pos_real : LT.lt 0 p.toReal
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) Top.top
  -/
  have hf_eLpNorm := Memℒp.eLpNorm_lt_top hf
  rw [eLpNorm_eq_eLpNorm' hp_pos hp_ne_top, f.eLpNorm'_eq, one_div,
    ← @ENNReal.lt_rpow_inv_iff _ _ p.toReal⁻¹ (by simp [hp_pos_real]),
    @ENNReal.top_rpow_of_pos p.toReal⁻¹⁻¹ (by simp [hp_pos_real]),
    ENNReal.sum_lt_top] at hf_eLpNorm
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Memℒp (⇑f) p μ
    y : E
    hy_ne : Ne y 0
    hp_pos_real : LT.lt 0 p.toReal
    hf_eLpNorm : ∀ (a : E), Membership.mem f.range a → LT.lt (HMul.hMul (HPow.hPow …
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) Top.top
  -/
  by_cases hyf : y ∈ f.range
  /-
    case pos
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Memℒp (⇑f) p μ
    y : E
    hy_ne : Ne y 0
    hp_pos_real : LT.lt 0 p.toReal
    hf_eLpNorm : ∀ (a : E), Membership.mem f.range a → LT.lt (HMul.hMul (HPow.hPow …
    hyf : Membership.mem f.range y
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) Top.top
  -/
  swap
  · suffices h_empty : f ⁻¹' {y} = ∅ by
      rw [h_empty, measure_empty]; exact ENNReal.coe_lt_top
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Memℒp (⇑f) p μ
      y : E
      hy_ne : Ne y 0
      hp_pos_real : LT.lt 0 p.toReal
      hf_eLpNorm : ∀ (a : E), Membership.mem f.range a → LT.lt (HMul.hMul (HPow.hPow …
      hyf : Not (Membership.mem f.range y)
      ⊢ Eq (Set.preimage (⇑f) (Singleton.singleton y)) EmptyCollection.emptyCollection
    -/
    ext1 x
    /-
      case neg.h
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Memℒp (⇑f) p μ
      y : E
      hy_ne : Ne y 0
      hp_pos_real : LT.lt 0 p.toReal
      hf_eLpNorm : ∀ (a : E), Membership.mem f.range a → LT.lt (HMul.hMul (HPow.hPow …
      hyf : Not (Membership.mem f.range y)
      x : α
      ⊢ Iff (Membership.mem (Set.preimage (⇑f) (Singleton.singleton y)) x) (Membersh …
    -/
    rw [Set.mem_preimage, Set.mem_singleton_iff, mem_empty_iff_false, iff_false]
    /-
      case neg.h
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Memℒp (⇑f) p μ
      y : E
      hy_ne : Ne y 0
      hp_pos_real : LT.lt 0 p.toReal
      hf_eLpNorm : ∀ (a : E), Membership.mem f.range a → LT.lt (HMul.hMul (HPow.hPow …
      hyf : Not (Membership.mem f.range y)
      x : α
      ⊢ Not (Eq (f x) y)
    -/
    refine fun hxy => hyf ?_
    /-
      case neg.h
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Memℒp (⇑f) p μ
      y : E
      hy_ne : Ne y 0
      hp_pos_real : LT.lt 0 p.toReal
      hf_eLpNorm : ∀ (a : E), Membership.mem f.range a → LT.lt (HMul.hMul (HPow.hPow …
      hyf : Not (Membership.mem f.range y)
      x : α
      hxy : Eq (f x) y
      ⊢ Membership.mem f.range y
    -/
    rw [mem_range, Set.mem_range]
    /-
      case neg.h
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Memℒp (⇑f) p μ
      y : E
      hy_ne : Ne y 0
      hp_pos_real : LT.lt 0 p.toReal
      hf_eLpNorm : ∀ (a : E), Membership.mem f.range a → LT.lt (HMul.hMul (HPow.hPow …
      hyf : Not (Membership.mem f.range y)
      x : α
      hxy : Eq (f x) y
      ⊢ Exists fun y_1 => Eq (f y_1) y
    -/
    exact ⟨x, hxy⟩
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Memℒp (⇑f) p μ
    y : E
    hy_ne : Ne y 0
    hp_pos_real : LT.lt 0 p.toReal
    hf_eLpNorm : ∀ (a : E), Membership.mem f.range a → LT.lt (HMul.hMul (HPow.hPow …
    hyf : Membership.mem f.range y
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) Top.top
  -/
  specialize hf_eLpNorm y hyf
  /-
    case pos
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Memℒp (⇑f) p μ
    y : E
    hy_ne : Ne y 0
    hp_pos_real : LT.lt 0 p.toReal
    hyf : Membership.mem f.range y
    hf_eLpNorm : LT.lt (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm y)) p.toReal) (μ (Se …
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) Top.top
  -/
  rw [ENNReal.mul_lt_top_iff] at hf_eLpNorm
  cases hf_eLpNorm with
  | inl hf_eLpNorm => exact hf_eLpNorm.2
  | inr hf_eLpNorm =>
    cases hf_eLpNorm with
    | inl hf_eLpNorm =>
      refine absurd ?_ hy_ne
      simpa [hp_pos_real] using hf_eLpNorm
    | inr hf_eLpNorm => simp [hf_eLpNorm]


theorem memℒp_of_finite_measure_preimage (p : ℝ≥0∞) {f : α →ₛ E}
    (hf : ∀ y, y ≠ 0 → μ (f ⁻¹' {y}) < ∞) : Memℒp f p μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : MeasureTheory.SimpleFunc α E
    hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    ⊢ MeasureTheory.Memℒp (⇑f) p μ
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : MeasureTheory.SimpleFunc α E
      hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
      hp0 : Eq p 0
      ⊢ MeasureTheory.Memℒp (⇑f) p μ
    -/
  · rw [hp0, memℒp_zero_iff_aestronglyMeasurable]; exact f.aestronglyMeasurable
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    case neg
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : MeasureTheory.SimpleFunc α E
    hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    hp0 : Not (Eq p 0)
    ⊢ MeasureTheory.Memℒp (⇑f) p μ
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : MeasureTheory.SimpleFunc α E
      hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
      hp0 : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ MeasureTheory.Memℒp (⇑f) p μ
    -/
  · rw [hp_top]; exact memℒp_top f μ
                 /-
                   🎉 no goals
                 -/
  /-
    case neg
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : MeasureTheory.SimpleFunc α E
    hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ MeasureTheory.Memℒp (⇑f) p μ
  -/
  refine ⟨f.aestronglyMeasurable, ?_⟩
  /-
    case neg
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : MeasureTheory.SimpleFunc α E
    hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ LT.lt (MeasureTheory.eLpNorm (⇑f) p μ) Top.top
  -/
  rw [eLpNorm_eq_eLpNorm' hp0 hp_top, f.eLpNorm'_eq]
  /-
    case neg
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : MeasureTheory.SimpleFunc α E
    hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ LT.lt (HPow.hPow (f.range.sum fun y => HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm …
  -/
  refine ENNReal.rpow_lt_top_of_nonneg (by simp) (ENNReal.sum_lt_top.mpr fun y _ => ?_).ne
  /-
    case neg
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    f : MeasureTheory.SimpleFunc α E
    hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    hp0 : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    y : E
    x✝ : Membership.mem f.range y
    ⊢ LT.lt (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm y)) p.toReal) (μ (Set.preimage  …
  -/
  by_cases hy0 : y = 0
    /-
      case pos
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : MeasureTheory.SimpleFunc α E
      hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
      hp0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      y : E
      x✝ : Membership.mem f.range y
      hy0 : Eq y 0
      ⊢ LT.lt (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm y)) p.toReal) (μ (Set.preimage  …
    -/
  · simp [hy0, ENNReal.toReal_pos hp0 hp_top]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : MeasureTheory.SimpleFunc α E
      hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
      hp0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      y : E
      x✝ : Membership.mem f.range y
      hy0 : Not (Eq y 0)
      ⊢ LT.lt (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm y)) p.toReal) (μ (Set.preimage  …
    -/
  · refine ENNReal.mul_lt_top ?_ (hf y hy0)
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      p : ENNReal
      f : MeasureTheory.SimpleFunc α E
      hf : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
      hp0 : Not (Eq p 0)
      hp_top : Not (Eq p Top.top)
      y : E
      x✝ : Membership.mem f.range y
      hy0 : Not (Eq y 0)
      ⊢ LT.lt (HPow.hPow (↑(NNNorm.nnnorm y)) p.toReal) Top.top
    -/
    exact ENNReal.rpow_lt_top_of_nonneg ENNReal.toReal_nonneg ENNReal.coe_ne_top
    /-
      🎉 no goals
    -/


theorem memℒp_iff {f : α →ₛ E} (hp_pos : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    Memℒp f p μ ↔ ∀ y, y ≠ 0 → μ (f ⁻¹' {y}) < ∞ :=
  ⟨fun h => measure_preimage_lt_top_of_memℒp hp_pos hp_ne_top f h, fun h =>
    memℒp_of_finite_measure_preimage p h⟩


theorem integrable_iff {f : α →ₛ E} : Integrable f μ ↔ ∀ y, y ≠ 0 → μ (f ⁻¹' {y}) < ∞ :=
  memℒp_one_iff_integrable.symm.trans <| memℒp_iff one_ne_zero ENNReal.coe_ne_top


theorem memℒp_iff_integrable {f : α →ₛ E} (hp_pos : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    Memℒp f p μ ↔ Integrable f μ :=
  (memℒp_iff hp_pos hp_ne_top).trans integrable_iff.symm


theorem memℒp_iff_finMeasSupp {f : α →ₛ E} (hp_pos : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    Memℒp f p μ ↔ f.FinMeasSupp μ :=
  (memℒp_iff hp_pos hp_ne_top).trans finMeasSupp_iff.symm


theorem integrable_iff_finMeasSupp {f : α →ₛ E} : Integrable f μ ↔ f.FinMeasSupp μ :=
  integrable_iff.trans finMeasSupp_iff.symm


theorem FinMeasSupp.integrable {f : α →ₛ E} (h : f.FinMeasSupp μ) : Integrable f μ :=
  integrable_iff_finMeasSupp.2 h


theorem integrable_pair {f : α →ₛ E} {g : α →ₛ F} :
    Integrable f μ → Integrable g μ → Integrable (pair f g) μ := by
  /-
    α : Type u_1
    E : Type u_4
    F : Type u_5
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    g : MeasureTheory.SimpleFunc α F
    ⊢ MeasureTheory.Integrable (⇑f) μ → MeasureTheory.Integrable (⇑g) μ → MeasureT …
  -/
  simpa only [integrable_iff_finMeasSupp] using FinMeasSupp.pair
  /-
    🎉 no goals
  -/


theorem memℒp_of_isFiniteMeasure (f : α →ₛ E) (p : ℝ≥0∞) (μ : Measure α) [IsFiniteMeasure μ] :
    Memℒp f p μ :=
  let ⟨C, hfC⟩ := f.exists_forall_norm_le
  Memℒp.of_bound f.aestronglyMeasurable C <| Eventually.of_forall hfC


theorem integrable_of_isFiniteMeasure [IsFiniteMeasure μ] (f : α →ₛ E) : Integrable f μ :=
  memℒp_one_iff_integrable.mp (f.memℒp_of_isFiniteMeasure 1 μ)


theorem measure_preimage_lt_top_of_integrable (f : α →ₛ E) (hf : Integrable f μ) {x : E}
    (hx : x ≠ 0) : μ (f ⁻¹' {x}) < ∞ :=
  integrable_iff.mp hf x hx


theorem measure_support_lt_top_of_memℒp (f : α →ₛ E) (hf : Memℒp f p μ) (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) : μ (support f) < ∞ :=
  f.measure_support_lt_top ((memℒp_iff hp_ne_zero hp_ne_top).mp hf)


theorem measure_support_lt_top_of_integrable (f : α →ₛ E) (hf : Integrable f μ) :
    μ (support f) < ∞ :=
  f.measure_support_lt_top (integrable_iff.mp hf)


theorem measure_lt_top_of_memℒp_indicator (hp_pos : p ≠ 0) (hp_ne_top : p ≠ ∞) {c : E} (hc : c ≠ 0)
    {s : Set α} (hs : MeasurableSet s) (hcs : Memℒp ((const α c).piecewise s hs (const α 0)) p μ) :
    μ s < ⊤ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    p : ENNReal
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    c : E
    hc : Ne c 0
    s : Set α
    hs : MeasurableSet s
    hcs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureT …
    ⊢ LT.lt (μ s) Top.top
  -/
  have : Function.support (const α c) = Set.univ := Function.support_const hc
  simpa only [memℒp_iff_finMeasSupp hp_pos hp_ne_top, finMeasSupp_iff_support,
    support_indicator, Set.inter_univ, this] using hcs


/-- `Lp.simpleFunc` is a subspace of Lp consisting of equivalence classes of an integrable simple
    function. -/
def simpleFunc : AddSubgroup (Lp E p μ) where
  carrier := { f : Lp E p μ | ∃ s : α →ₛ E, (AEEqFun.mk s s.aestronglyMeasurable : α →ₘ[μ] E) = f }
  zero_mem' := ⟨0, rfl⟩
  add_mem' := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      E : Type u_4
      F : Type u_5
      𝕜 : Type u_6
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      μ : MeasureTheory.Measure α
      ⊢ ∀ {a b : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x}, Member …
    -/
    rintro f g ⟨s, hs⟩ ⟨t, ht⟩
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      E : Type u_4
      F : Type u_5
      𝕜 : Type u_6
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      μ : MeasureTheory.Measure α
      f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      s : MeasureTheory.SimpleFunc α E
      hs : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
      t : MeasureTheory.SimpleFunc α E
      ht : Eq (MeasureTheory.AEEqFun.mk ⇑t ⋯) ↑g
      ⊢ Membership.mem (setOf fun f => Exists fun s => Eq (MeasureTheory.AEEqFun.mk  …
    -/
    use s + t
    simp only [← hs, ← ht, AEEqFun.mk_add_mk, AddSubgroup.coe_add, AEEqFun.mk_eq_mk,
      SimpleFunc.coe_add]
  neg_mem' := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      E : Type u_4
      F : Type u_5
      𝕜 : Type u_6
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      μ : MeasureTheory.Measure α
      ⊢ ∀ {x : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x}, Membersh …
    -/
    rintro f ⟨s, hs⟩
    /-
      case intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      E : Type u_4
      F : Type u_5
      𝕜 : Type u_6
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      s : MeasureTheory.SimpleFunc α E
      hs : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
      ⊢ Membership.mem { carrier := setOf fun f => Exists fun s => Eq (MeasureTheory …
    -/
    use -s
    /-
      case h
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      E : Type u_4
      F : Type u_5
      𝕜 : Type u_6
      inst✝² : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      p : ENNReal
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      s : MeasureTheory.SimpleFunc α E
      hs : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
      ⊢ Eq (MeasureTheory.AEEqFun.mk ⇑(Neg.neg s) ⋯) ↑(Neg.neg f)
    -/
    simp only [← hs, AEEqFun.neg_mk, SimpleFunc.coe_neg, AEEqFun.mk_eq_mk, AddSubgroup.coe_neg]
    /-
      🎉 no goals
    -/


protected theorem eq' {f g : Lp.simpleFunc E p μ} : (f : α →ₘ[μ] E) = (g : α →ₘ[μ] E) → f = g :=
  Subtype.eq ∘ Subtype.eq


/-- If `E` is a normed space, `Lp.simpleFunc E p μ` is a `SMul`. Not declared as an
instance as it is (as of writing) used only in the construction of the Bochner integral. -/
protected def smul : SMul 𝕜 (Lp.simpleFunc E p μ) :=
  ⟨fun k f =>
    ⟨k • (f : Lp E p μ), by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        E : Type u_4
        F : Type u_5
        𝕜 : Type u_6
        inst✝⁵ : MeasurableSpace α
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedAddCommGroup F
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝² : NormedRing 𝕜
        inst✝¹ : Module 𝕜 E
        inst✝ : BoundedSMul 𝕜 E
        k : 𝕜
        f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
        ⊢ Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) (HSMul.hSMul k ↑f)
      -/
      rcases f with ⟨f, ⟨s, hs⟩⟩
      /-
        case mk.intro
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        E : Type u_4
        F : Type u_5
        𝕜 : Type u_6
        inst✝⁵ : MeasurableSpace α
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedAddCommGroup F
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝² : NormedRing 𝕜
        inst✝¹ : Module 𝕜 E
        inst✝ : BoundedSMul 𝕜 E
        k : 𝕜
        f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        s : MeasureTheory.SimpleFunc α E
        hs : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
        ⊢ Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) (HSMul.hSMul k ↑⟨f, ⋯⟩)
      -/
      use k • s
      /-
        case h
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        E : Type u_4
        F : Type u_5
        𝕜 : Type u_6
        inst✝⁵ : MeasurableSpace α
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedAddCommGroup F
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝² : NormedRing 𝕜
        inst✝¹ : Module 𝕜 E
        inst✝ : BoundedSMul 𝕜 E
        k : 𝕜
        f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        s : MeasureTheory.SimpleFunc α E
        hs : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
        ⊢ Eq (MeasureTheory.AEEqFun.mk ⇑(HSMul.hSMul k s) ⋯) ↑(HSMul.hSMul k ↑⟨f, ⋯⟩)
      -/
      apply Eq.trans (AEEqFun.smul_mk k s s.aestronglyMeasurable).symm _
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        E : Type u_4
        F : Type u_5
        𝕜 : Type u_6
        inst✝⁵ : MeasurableSpace α
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedAddCommGroup F
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝² : NormedRing 𝕜
        inst✝¹ : Module 𝕜 E
        inst✝ : BoundedSMul 𝕜 E
        k : 𝕜
        f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        s : MeasureTheory.SimpleFunc α E
        hs : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
        ⊢ Eq (HSMul.hSMul k (MeasureTheory.AEEqFun.mk ⇑s ⋯)) ↑(HSMul.hSMul k ↑⟨f, ⋯⟩)
      -/
      rw [hs]
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        E : Type u_4
        F : Type u_5
        𝕜 : Type u_6
        inst✝⁵ : MeasurableSpace α
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedAddCommGroup F
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝² : NormedRing 𝕜
        inst✝¹ : Module 𝕜 E
        inst✝ : BoundedSMul 𝕜 E
        k : 𝕜
        f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        s : MeasureTheory.SimpleFunc α E
        hs : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
        ⊢ Eq (HSMul.hSMul k ↑f) ↑(HSMul.hSMul k ↑⟨f, ⋯⟩)
      -/
      rfl⟩⟩
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
theorem coe_smul (c : 𝕜) (f : Lp.simpleFunc E p μ) :
    ((c • f : Lp.simpleFunc E p μ) : Lp E p μ) = c • (f : Lp E p μ) :=
  rfl


/-- If `E` is a normed space, `Lp.simpleFunc E p μ` is a module. Not declared as an
instance as it is (as of writing) used only in the construction of the Bochner integral. -/
protected def module : Module 𝕜 (Lp.simpleFunc E p μ) where
                   /-
                     α : Type u_1
                     β : Type u_2
                     ι : Type u_3
                     E : Type u_4
                     F : Type u_5
                     𝕜 : Type u_6
                     inst✝⁵ : MeasurableSpace α
                     inst✝⁴ : NormedAddCommGroup E
                     inst✝³ : NormedAddCommGroup F
                     p : ENNReal
                     μ : MeasureTheory.Measure α
                     inst✝² : NormedRing 𝕜
                     inst✝¹ : Module 𝕜 E
                     inst✝ : BoundedSMul 𝕜 E
                     f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
                     ⊢ Eq (HSMul.hSMul 1 f) f
                   -/
  one_smul f := by ext1; exact one_smul _ _
                         /-
                           🎉 no goals
                         -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         ι : Type u_3
                         E : Type u_4
                         F : Type u_5
                         𝕜 : Type u_6
                         inst✝⁵ : MeasurableSpace α
                         inst✝⁴ : NormedAddCommGroup E
                         inst✝³ : NormedAddCommGroup F
                         p : ENNReal
                         μ : MeasureTheory.Measure α
                         inst✝² : NormedRing 𝕜
                         inst✝¹ : Module 𝕜 E
                         inst✝ : BoundedSMul 𝕜 E
                         x y : 𝕜
                         f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x y) f) (HSMul.hSMul x (HSMul.hSMul y f))
                       -/
  mul_smul x y f := by ext1; exact mul_smul _ _ _
                             /-
                               🎉 no goals
                             -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         ι : Type u_3
                         E : Type u_4
                         F : Type u_5
                         𝕜 : Type u_6
                         inst✝⁵ : MeasurableSpace α
                         inst✝⁴ : NormedAddCommGroup E
                         inst✝³ : NormedAddCommGroup F
                         p : ENNReal
                         μ : MeasureTheory.Measure α
                         inst✝² : NormedRing 𝕜
                         inst✝¹ : Module 𝕜 E
                         inst✝ : BoundedSMul 𝕜 E
                         x : 𝕜
                         f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
                         ⊢ Eq (HSMul.hSMul x (HAdd.hAdd f g)) (HAdd.hAdd (HSMul.hSMul x f) (HSMul.hSMul …
                       -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      ι : Type u_3
                      E : Type u_4
                      F : Type u_5
                      𝕜 : Type u_6
                      inst✝⁵ : MeasurableSpace α
                      inst✝⁴ : NormedAddCommGroup E
                      inst✝³ : NormedAddCommGroup F
                      p : ENNReal
                      μ : MeasureTheory.Measure α
                      inst✝² : NormedRing 𝕜
                      inst✝¹ : Module 𝕜 E
                      inst✝ : BoundedSMul 𝕜 E
                      x : 𝕜
                      ⊢ Eq (HSMul.hSMul x 0) 0
                    -/
  smul_add x f g := by ext1; exact smul_add _ _ _
                          /-
                            🎉 no goals
                          -/
                             /-
                               🎉 no goals
                             -/
  smul_zero x := by ext1; exact smul_zero _
                       /-
                         α : Type u_1
                         β : Type u_2
                         ι : Type u_3
                         E : Type u_4
                         F : Type u_5
                         𝕜 : Type u_6
                         inst✝⁵ : MeasurableSpace α
                         inst✝⁴ : NormedAddCommGroup E
                         inst✝³ : NormedAddCommGroup F
                         p : ENNReal
                         μ : MeasureTheory.Measure α
                         inst✝² : NormedRing 𝕜
                         inst✝¹ : Module 𝕜 E
                         inst✝ : BoundedSMul 𝕜 E
                         x y : 𝕜
                         f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd x y) f) (HAdd.hAdd (HSMul.hSMul x f) (HSMul.hSMul …
                       -/
  add_smul x y f := by ext1; exact add_smul _ _ _
                             /-
                               🎉 no goals
                             -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      ι : Type u_3
                      E : Type u_4
                      F : Type u_5
                      𝕜 : Type u_6
                      inst✝⁵ : MeasurableSpace α
                      inst✝⁴ : NormedAddCommGroup E
                      inst✝³ : NormedAddCommGroup F
                      p : ENNReal
                      μ : MeasureTheory.Measure α
                      inst✝² : NormedRing 𝕜
                      inst✝¹ : Module 𝕜 E
                      inst✝ : BoundedSMul 𝕜 E
                      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
                      ⊢ Eq (HSMul.hSMul 0 f) 0
                    -/
  zero_smul f := by ext1; exact zero_smul _ _
                          /-
                            🎉 no goals
                          -/


/-- If `E` is a normed space, `Lp.simpleFunc E p μ` is a normed space. Not declared as an
instance as it is (as of writing) used only in the construction of the Bochner integral. -/
protected theorem boundedSMul [Fact (1 ≤ p)] : BoundedSMul 𝕜 (Lp.simpleFunc E p μ) :=
  BoundedSMul.of_norm_smul_le fun r f => (norm_smul_le r (f : Lp E p μ) : _)


/-- If `E` is a normed space, `Lp.simpleFunc E p μ` is a normed space. Not declared as an
instance as it is (as of writing) used only in the construction of the Bochner integral. -/
protected def normedSpace {𝕜} [NormedField 𝕜] [NormedSpace 𝕜 E] [Fact (1 ≤ p)] :
    NormedSpace 𝕜 (Lp.simpleFunc E p μ) :=
  ⟨norm_smul_le (α := 𝕜) (β := Lp.simpleFunc E p μ)⟩


/-- Construct the equivalence class `[f]` of a simple function `f` satisfying `Memℒp`. -/
abbrev toLp (f : α →ₛ E) (hf : Memℒp f p μ) : Lp.simpleFunc E p μ :=
  ⟨hf.toLp f, ⟨f, rfl⟩⟩


theorem toLp_eq_toLp (f : α →ₛ E) (hf : Memℒp f p μ) : (toLp f hf : Lp E p μ) = hf.toLp f :=
  rfl


theorem toLp_eq_mk (f : α →ₛ E) (hf : Memℒp f p μ) :
    (toLp f hf : α →ₘ[μ] E) = AEEqFun.mk f f.aestronglyMeasurable :=
  rfl


theorem toLp_zero : toLp (0 : α →ₛ E) zero_memℒp = (0 : Lp.simpleFunc E p μ) :=
  rfl


theorem toLp_add (f g : α →ₛ E) (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
    toLp (f + g) (hf.add hg) = toLp f hf + toLp g hg :=
  rfl


theorem toLp_neg (f : α →ₛ E) (hf : Memℒp f p μ) : toLp (-f) hf.neg = -toLp f hf :=
  rfl


theorem toLp_sub (f g : α →ₛ E) (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
    toLp (f - g) (hf.sub hg) = toLp f hf - toLp g hg := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Memℒp (⇑f) p μ
    hg : MeasureTheory.Memℒp (⇑g) p μ
    ⊢ Eq (MeasureTheory.Lp.simpleFunc.toLp (HSub.hSub f g) ⋯) (HSub.hSub (MeasureT …
  -/
  simp only [sub_eq_add_neg, ← toLp_neg, ← toLp_add]
  /-
    🎉 no goals
  -/


theorem toLp_smul (f : α →ₛ E) (hf : Memℒp f p μ) (c : 𝕜) :
    toLp (c • f) (hf.const_smul c) = c • toLp f hf :=
  rfl


nonrec theorem norm_toLp [Fact (1 ≤ p)] (f : α →ₛ E) (hf : Memℒp f p μ) :
    ‖toLp f hf‖ = ENNReal.toReal (eLpNorm f p μ) :=
  norm_toLp f hf


/-- Find a representative of a `Lp.simpleFunc`. -/
def toSimpleFunc (f : Lp.simpleFunc E p μ) : α →ₛ E :=
  Classical.choose f.2


/-- `(toSimpleFunc f)` is measurable. -/
@[measurability]
protected theorem measurable [MeasurableSpace E] (f : Lp.simpleFunc E p μ) :
    Measurable (toSimpleFunc f) :=
  (toSimpleFunc f).measurable


protected theorem stronglyMeasurable (f : Lp.simpleFunc E p μ) :
    StronglyMeasurable (toSimpleFunc f) :=
  (toSimpleFunc f).stronglyMeasurable


@[measurability]
protected theorem aemeasurable [MeasurableSpace E] (f : Lp.simpleFunc E p μ) :
    AEMeasurable (toSimpleFunc f) μ :=
  (simpleFunc.measurable f).aemeasurable


protected theorem aestronglyMeasurable (f : Lp.simpleFunc E p μ) :
    AEStronglyMeasurable (toSimpleFunc f) μ :=
  (simpleFunc.stronglyMeasurable f).aestronglyMeasurable


theorem toSimpleFunc_eq_toFun (f : Lp.simpleFunc E p μ) : toSimpleFunc f =ᵐ[μ] f :=
  show ⇑(toSimpleFunc f) =ᵐ[μ] ⇑(f : α →ₘ[μ] E) by
    convert (AEEqFun.coeFn_mk (toSimpleFunc f)
          (toSimpleFunc f).aestronglyMeasurable).symm using 2
    /-
      case h.e'_5.h.e'_6
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
      ⊢ Eq (↑↑f) (MeasureTheory.AEEqFun.mk ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFun …
    -/
    exact (Classical.choose_spec f.2).symm
    /-
      🎉 no goals
    -/


/-- `toSimpleFunc f` satisfies the predicate `Memℒp`. -/
protected theorem memℒp (f : Lp.simpleFunc E p μ) : Memℒp (toSimpleFunc f) p μ :=
  Memℒp.ae_eq (toSimpleFunc_eq_toFun f).symm <| mem_Lp_iff_memℒp.mp (f : Lp E p μ).2


theorem toLp_toSimpleFunc (f : Lp.simpleFunc E p μ) :
    toLp (toSimpleFunc f) (simpleFunc.memℒp f) = f :=
  simpleFunc.eq' (Classical.choose_spec f.2)


theorem toSimpleFunc_toLp (f : α →ₛ E) (hfi : Memℒp f p μ) : toSimpleFunc (toLp f hfi) =ᵐ[μ] f := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    hfi : MeasureTheory.Memℒp (⇑f) p μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  rw [← AEEqFun.mk_eq_mk]; exact Classical.choose_spec (toLp f hfi).2
                           /-
                             🎉 no goals
                           -/


theorem zero_toSimpleFunc : toSimpleFunc (0 : Lp.simpleFunc E p μ) =ᵐ[μ] 0 := by
  filter_upwards [toSimpleFunc_eq_toFun (0 : Lp.simpleFunc E p μ),
    Lp.coeFn_zero E 1 μ] with _ h₁ _
  /-
    case h
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    a✝¹ : α
    h₁ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc 0) a✝¹) (↑↑↑0 a✝¹)
    a✝ : Eq (↑↑0 a✝¹) (0 a✝¹)
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc 0) a✝¹) (0 a✝¹)
  -/
  rwa [h₁]
  /-
    🎉 no goals
  -/


theorem add_toSimpleFunc (f g : Lp.simpleFunc E p μ) :
    toSimpleFunc (f + g) =ᵐ[μ] toSimpleFunc f + toSimpleFunc g := by
  filter_upwards [toSimpleFunc_eq_toFun (f + g), toSimpleFunc_eq_toFun f,
    toSimpleFunc_eq_toFun g, Lp.coeFn_add (f : Lp E p μ) g] with _
  /-
    case h
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    a✝ : α
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (HAdd.hAdd f g)) a✝) (↑↑↑(HAdd …
  -/
  simp only [AddSubgroup.coe_add, Pi.add_apply]
  /-
    case h
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    a✝ : α
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (HAdd.hAdd f g)) a✝) (↑(HAdd.h …
  -/
  iterate 4 intro h; rw [h]
  /-
    🎉 no goals
  -/


theorem neg_toSimpleFunc (f : Lp.simpleFunc E p μ) : toSimpleFunc (-f) =ᵐ[μ] -toSimpleFunc f := by
  filter_upwards [toSimpleFunc_eq_toFun (-f), toSimpleFunc_eq_toFun f,
    Lp.coeFn_neg (f : Lp E p μ)] with _
  /-
    case h
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    a✝ : α
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (Neg.neg f)) a✝) (↑↑↑(Neg.neg  …
  -/
  simp only [Pi.neg_apply, AddSubgroup.coe_neg]
  /-
    case h
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    a✝ : α
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (Neg.neg f)) a✝) (↑(Neg.neg ↑↑ …
  -/
  repeat intro h; rw [h]
  /-
    🎉 no goals
  -/


theorem sub_toSimpleFunc (f g : Lp.simpleFunc E p μ) :
    toSimpleFunc (f - g) =ᵐ[μ] toSimpleFunc f - toSimpleFunc g := by
  filter_upwards [toSimpleFunc_eq_toFun (f - g), toSimpleFunc_eq_toFun f,
    toSimpleFunc_eq_toFun g, Lp.coeFn_sub (f : Lp E p μ) g] with _
  /-
    case h
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    a✝ : α
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (HSub.hSub f g)) a✝) (↑↑↑(HSub …
  -/
  simp only [AddSubgroup.coe_sub, Pi.sub_apply]
  /-
    case h
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    a✝ : α
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (HSub.hSub f g)) a✝) (↑(HSub.h …
  -/
  repeat' intro h; rw [h]
  /-
    🎉 no goals
  -/


theorem smul_toSimpleFunc (k : 𝕜) (f : Lp.simpleFunc E p μ) :
    toSimpleFunc (k • f) =ᵐ[μ] k • ⇑(toSimpleFunc f) := by
  filter_upwards [toSimpleFunc_eq_toFun (k • f), toSimpleFunc_eq_toFun f,
    Lp.coeFn_smul k (f : Lp E p μ)] with _
  /-
    case h
    α : Type u_1
    E : Type u_4
    𝕜 : Type u_6
    inst✝⁴ : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    k : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    a✝ : α
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (HSMul.hSMul k f)) a✝) (↑↑↑(HS …
  -/
  simp only [Pi.smul_apply, coe_smul]
  /-
    case h
    α : Type u_1
    E : Type u_4
    𝕜 : Type u_6
    inst✝⁴ : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    k : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    a✝ : α
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (HSMul.hSMul k f)) a✝) (↑↑(HSM …
  -/
  repeat intro h; rw [h]
  /-
    🎉 no goals
  -/


theorem norm_toSimpleFunc [Fact (1 ≤ p)] (f : Lp.simpleFunc E p μ) :
    ‖f‖ = ENNReal.toReal (eLpNorm (toSimpleFunc f) p μ) := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    ⊢ Eq (Norm.norm f) (MeasureTheory.eLpNorm (⇑(MeasureTheory.Lp.simpleFunc.toSim …
  -/
  simpa [toLp_toSimpleFunc] using norm_toLp (toSimpleFunc f) (simpleFunc.memℒp f)
  /-
    🎉 no goals
  -/


/-- The characteristic function of a finite-measure measurable set `s`, as an `Lp` simple function.
-/
def indicatorConst {s : Set α} (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (c : E) :
    Lp.simpleFunc E p μ :=
  toLp ((SimpleFunc.const _ c).piecewise s hs (SimpleFunc.const _ 0))
    (memℒp_indicator_const p hs c (Or.inr hμs))


@[simp]
theorem coe_indicatorConst {s : Set α} (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (c : E) :
    (↑(indicatorConst p hs hμs c) : Lp E p μ) = indicatorConstLp p hs hμs c :=
  rfl


theorem toSimpleFunc_indicatorConst {s : Set α} (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (c : E) :
    toSimpleFunc (indicatorConst p hs hμs c) =ᵐ[μ]
      (SimpleFunc.const _ c).piecewise s hs (SimpleFunc.const _ 0) :=
  Lp.simpleFunc.toSimpleFunc_toLp _ _


/-- To prove something for an arbitrary `Lp` simple function, with `0 < p < ∞`, it suffices to show
that the property holds for (multiples of) characteristic functions of finite-measure measurable
sets and is closed under addition (of functions with disjoint support). -/
@[elab_as_elim]
protected theorem induction (hp_pos : p ≠ 0) (hp_ne_top : p ≠ ∞) {P : Lp.simpleFunc E p μ → Prop}
    (h_ind :
      ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : μ s < ∞),
        P (Lp.simpleFunc.indicatorConst p hs hμs.ne c))
    (h_add :
      ∀ ⦃f g : α →ₛ E⦄,
        ∀ hf : Memℒp f p μ,
          ∀ hg : Memℒp g p μ,
            Disjoint (support f) (support g) →
              P (Lp.simpleFunc.toLp f hf) →
                P (Lp.simpleFunc.toLp g hg) → P (Lp.simpleFunc.toLp f hf + Lp.simpleFunc.toLp g hg))
    (f : Lp.simpleFunc E p μ) : P f := by
  suffices ∀ f : α →ₛ E, ∀ hf : Memℒp f p μ, P (toLp f hf) by
    rw [← toLp_toSimpleFunc f]
    apply this
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
    h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x
    ⊢ ∀ (f : MeasureTheory.SimpleFunc α E) (hf : MeasureTheory.Memℒp (⇑f) p μ), P  …
  -/
  clear f
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    hp_pos : Ne p 0
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
    h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
    ⊢ ∀ (f : MeasureTheory.SimpleFunc α E) (hf : MeasureTheory.Memℒp (⇑f) p μ), P  …
  -/
  apply SimpleFunc.induction
    /-
      case h_ind
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
      h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
      ⊢ ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hf : MeasureTheory.Memℒp (⇑(Me …
    -/
  · intro c s hs hf
    /-
      case h_ind
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
      h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
      c : E
      s : Set α
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
      ⊢ P (MeasureTheory.Lp.simpleFunc.toLp (MeasureTheory.SimpleFunc.piecewise s hs …
    -/
    by_cases hc : c = 0
      /-
        case pos
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_pos : Ne p 0
        hp_ne_top : Ne p Top.top
        P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
        h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
        h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
        c : E
        s : Set α
        hs : MeasurableSet s
        hf : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        hc : Eq c 0
        ⊢ P (MeasureTheory.Lp.simpleFunc.toLp (MeasureTheory.SimpleFunc.piecewise s hs …
      -/
    · convert h_ind 0 MeasurableSet.empty (by simp) using 1
      /-
        case h.e'_1
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_pos : Ne p 0
        hp_ne_top : Ne p Top.top
        P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
        h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
        h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
        c : E
        s : Set α
        hs : MeasurableSet s
        hf : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        hc : Eq c 0
        ⊢ Eq (MeasureTheory.Lp.simpleFunc.toLp (MeasureTheory.SimpleFunc.piecewise s h …
      -/
      ext1
      /-
        case h.e'_1.a
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_pos : Ne p 0
        hp_ne_top : Ne p Top.top
        P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
        h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
        h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
        c : E
        s : Set α
        hs : MeasurableSet s
        hf : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        hc : Eq c 0
        ⊢ Eq ↑(MeasureTheory.Lp.simpleFunc.toLp (MeasureTheory.SimpleFunc.piecewise s  …
      -/
      simp [hc]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
      h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
      c : E
      s : Set α
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
      hc : Not (Eq c 0)
      ⊢ P (MeasureTheory.Lp.simpleFunc.toLp (MeasureTheory.SimpleFunc.piecewise s hs …
    -/
    exact h_ind c hs (SimpleFunc.measure_lt_top_of_memℒp_indicator hp_pos hp_ne_top hc hs hf)
    /-
      🎉 no goals
    -/
    /-
      case h_add
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
      h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
      ⊢ ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄, Disjoint (Function.support ⇑f) (Func …
    -/
  · intro f g hfg hf hg hfg'
    obtain ⟨hf', hg'⟩ : Memℒp f p μ ∧ Memℒp g p μ :=
      (memℒp_add_of_disjoint hfg f.stronglyMeasurable g.stronglyMeasurable).mp hfg'
    /-
      case h_add.intro
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_pos : Ne p 0
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E p μ) x) →  …
      h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f)  …
      f g : MeasureTheory.SimpleFunc α E
      hfg : Disjoint (Function.support ⇑f) (Function.support ⇑g)
      hf : ∀ (hf : MeasureTheory.Memℒp (⇑f) p μ), P (MeasureTheory.Lp.simpleFunc.toL …
      hg : ∀ (hf : MeasureTheory.Memℒp (⇑g) p μ), P (MeasureTheory.Lp.simpleFunc.toL …
      hfg' : MeasureTheory.Memℒp (⇑(HAdd.hAdd f g)) p μ
      hf' : MeasureTheory.Memℒp (⇑f) p μ
      hg' : MeasureTheory.Memℒp (⇑g) p μ
      ⊢ P (MeasureTheory.Lp.simpleFunc.toLp (HAdd.hAdd f g) hfg')
    -/
    exact h_add hf' hg' hfg (hf hf') (hg hg')
    /-
      🎉 no goals
    -/


protected theorem uniformContinuous : UniformContinuous ((↑) : Lp.simpleFunc E p μ → Lp E p μ) :=
  uniformContinuous_comap


lemma isUniformEmbedding : IsUniformEmbedding ((↑) : Lp.simpleFunc E p μ → Lp E p μ) :=
  isUniformEmbedding_comap Subtype.val_injective


@[deprecated (since := "2024-10-01")] alias uniformEmbedding := isUniformEmbedding


theorem isUniformInducing : IsUniformInducing ((↑) : Lp.simpleFunc E p μ → Lp E p μ) :=
  simpleFunc.isUniformEmbedding.isUniformInducing


@[deprecated (since := "2024-10-05")]
alias uniformInducing := isUniformInducing


lemma isDenseEmbedding (hp_ne_top : p ≠ ∞) :
    IsDenseEmbedding ((↑) : Lp.simpleFunc E p μ → Lp E p μ) := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    ⊢ IsDenseEmbedding Subtype.val
  -/
  borelize E
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ IsDenseEmbedding Subtype.val
  -/
  apply simpleFunc.isUniformEmbedding.isDenseEmbedding
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ DenseRange Subtype.val
  -/
  intro f
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Membership.mem (closure (Set.range Subtype.val)) f
  -/
  rw [mem_closure_iff_seq_limit]
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Exists fun x => And (∀ (n : Nat), Membership.mem (Set.range Subtype.val) (x  …
  -/
  have hfi' : Memℒp f p μ := Lp.memℒp f
  haveI : SeparableSpace (range f ∪ {0} : Set E) :=
    (Lp.stronglyMeasurable f).separableSpace_range_union_singleton
  refine
    ⟨fun n =>
      toLp
        (SimpleFunc.approxOn f (Lp.stronglyMeasurable f).measurable (range f ∪ {0}) 0 _ n)
        (SimpleFunc.memℒp_approxOn_range (Lp.stronglyMeasurable f).measurable hfi' n),
      fun n => mem_range_self _, ?_⟩
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hfi' : MeasureTheory.Memℒp (↑↑f) p μ
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range ↑↑f) (Singleto …
    ⊢ Filter.Tendsto (fun n => ↑(MeasureTheory.Lp.simpleFunc.toLp (MeasureTheory.S …
  -/
  convert SimpleFunc.tendsto_approxOn_range_Lp hp_ne_top (Lp.stronglyMeasurable f).measurable hfi'
  /-
    case h.e'_5.h.e'_3
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hfi' : MeasureTheory.Memℒp (↑↑f) p μ
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range ↑↑f) (Singleto …
    ⊢ Eq f (MeasureTheory.Memℒp.toLp (↑↑f) hfi')
  -/
  rw [toLp_coeFn f (Lp.memℒp f)]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-30")]
alias denseEmbedding := isDenseEmbedding


protected theorem isDenseInducing (hp_ne_top : p ≠ ∞) :
    IsDenseInducing ((↑) : Lp.simpleFunc E p μ → Lp E p μ) :=
  (simpleFunc.isDenseEmbedding hp_ne_top).isDenseInducing


protected theorem denseRange (hp_ne_top : p ≠ ∞) :
    DenseRange ((↑) : Lp.simpleFunc E p μ → Lp E p μ) :=
  (simpleFunc.isDenseInducing hp_ne_top).dense


protected theorem dense (hp_ne_top : p ≠ ∞) : Dense (Lp.simpleFunc E p μ : Set (Lp E p μ)) := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝² : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    ⊢ Dense ↑(MeasureTheory.Lp.simpleFunc E p μ)
  -/
  simpa only [denseRange_subtype_val] using simpleFunc.denseRange (E := E) (μ := μ) hp_ne_top
  /-
    🎉 no goals
  -/


/-- The embedding of Lp simple functions into Lp functions, as a continuous linear map. -/
def coeToLp : Lp.simpleFunc E p μ →L[𝕜] Lp E p μ :=
  { AddSubgroup.subtype (Lp.simpleFunc E p μ) with
    map_smul' := fun _ _ => rfl
    cont := Lp.simpleFunc.uniformContinuous.continuous }


theorem coeFn_le (f g : Lp.simpleFunc G p μ) : (f : α → G) ≤ᵐ[μ] g ↔ f ≤ g := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyLE ↑↑↑f ↑↑↑g) (LE.le f g)
  -/
  rw [← Subtype.coe_le_coe, ← Lp.coeFn_le]
  /-
    🎉 no goals
  -/


instance instAddLeftMono : AddLeftMono (Lp.simpleFunc G p μ) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    E : Type u_4
    F : Type u_5
    𝕜 : Type u_6
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    ⊢ AddLeftMono (Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G  …
  -/
  refine ⟨fun f g₁ g₂ hg₁₂ => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    E : Type u_4
    F : Type u_5
    𝕜 : Type u_6
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    hg₁₂ : LE.le g₁ g₂
    ⊢ LE.le (HAdd.hAdd f g₁) (HAdd.hAdd f g₂)
  -/
  rw [← Lp.simpleFunc.coeFn_le] at hg₁₂ ⊢
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    E : Type u_4
    F : Type u_5
    𝕜 : Type u_6
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    hg₁₂ : (MeasureTheory.ae μ).EventuallyLE ↑↑↑g₁ ↑↑↑g₂
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑↑(HAdd.hAdd f g₁) ↑↑↑(HAdd.hAdd f g₂)
  -/
  have h_add_1 : ((f + g₁ : Lp.simpleFunc G p μ) : α → G) =ᵐ[μ] (f : α → G) + g₁ := Lp.coeFn_add _ _
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    E : Type u_4
    F : Type u_5
    𝕜 : Type u_6
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    hg₁₂ : (MeasureTheory.ae μ).EventuallyLE ↑↑↑g₁ ↑↑↑g₂
    h_add_1 : (MeasureTheory.ae μ).EventuallyEq (↑↑↑(HAdd.hAdd f g₁)) (HAdd.hAdd ↑ …
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑↑(HAdd.hAdd f g₁) ↑↑↑(HAdd.hAdd f g₂)
  -/
  have h_add_2 : ((f + g₂ : Lp.simpleFunc G p μ) : α → G) =ᵐ[μ] (f : α → G) + g₂ := Lp.coeFn_add _ _
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    E : Type u_4
    F : Type u_5
    𝕜 : Type u_6
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    hg₁₂ : (MeasureTheory.ae μ).EventuallyLE ↑↑↑g₁ ↑↑↑g₂
    h_add_1 : (MeasureTheory.ae μ).EventuallyEq (↑↑↑(HAdd.hAdd f g₁)) (HAdd.hAdd ↑ …
    h_add_2 : (MeasureTheory.ae μ).EventuallyEq (↑↑↑(HAdd.hAdd f g₂)) (HAdd.hAdd ↑ …
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑↑(HAdd.hAdd f g₁) ↑↑↑(HAdd.hAdd f g₂)
  -/
  filter_upwards [h_add_1, h_add_2, hg₁₂] with _ h1 h2 h3
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    E : Type u_4
    F : Type u_5
    𝕜 : Type u_6
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    hg₁₂ : (MeasureTheory.ae μ).EventuallyLE ↑↑↑g₁ ↑↑↑g₂
    h_add_1 : (MeasureTheory.ae μ).EventuallyEq (↑↑↑(HAdd.hAdd f g₁)) (HAdd.hAdd ↑ …
    h_add_2 : (MeasureTheory.ae μ).EventuallyEq (↑↑↑(HAdd.hAdd f g₂)) (HAdd.hAdd ↑ …
    a✝ : α
    h1 : Eq (↑↑↑(HAdd.hAdd f g₁) a✝) (HAdd.hAdd (↑↑↑f) (↑↑↑g₁) a✝)
    h2 : Eq (↑↑↑(HAdd.hAdd f g₂) a✝) (HAdd.hAdd (↑↑↑f) (↑↑↑g₂) a✝)
    h3 : LE.le (↑↑↑g₁ a✝) (↑↑↑g₂ a✝)
    ⊢ LE.le (↑↑↑(HAdd.hAdd f g₁) a✝) (↑↑↑(HAdd.hAdd f g₂) a✝)
  -/
  rw [h1, h2, Pi.add_apply, Pi.add_apply]
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    E : Type u_4
    F : Type u_5
    𝕜 : Type u_6
    inst✝³ : MeasurableSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    hg₁₂ : (MeasureTheory.ae μ).EventuallyLE ↑↑↑g₁ ↑↑↑g₂
    h_add_1 : (MeasureTheory.ae μ).EventuallyEq (↑↑↑(HAdd.hAdd f g₁)) (HAdd.hAdd ↑ …
    h_add_2 : (MeasureTheory.ae μ).EventuallyEq (↑↑↑(HAdd.hAdd f g₂)) (HAdd.hAdd ↑ …
    a✝ : α
    h1 : Eq (↑↑↑(HAdd.hAdd f g₁) a✝) (HAdd.hAdd (↑↑↑f) (↑↑↑g₁) a✝)
    h2 : Eq (↑↑↑(HAdd.hAdd f g₂) a✝) (HAdd.hAdd (↑↑↑f) (↑↑↑g₂) a✝)
    h3 : LE.le (↑↑↑g₁ a✝) (↑↑↑g₂ a✝)
    ⊢ LE.le (HAdd.hAdd (↑↑↑f a✝) (↑↑↑g₁ a✝)) (HAdd.hAdd (↑↑↑f a✝) (↑↑↑g₂ a✝))
  -/
  exact add_le_add le_rfl h3
  /-
    🎉 no goals
  -/


theorem coeFn_zero : (0 : Lp.simpleFunc G p μ) =ᵐ[μ] (0 : α → G) :=
  Lp.coeFn_zero _ _ _


theorem coeFn_nonneg (f : Lp.simpleFunc G p μ) : (0 : α → G) ≤ᵐ[μ] f ↔ 0 ≤ f := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyLE 0 ↑↑↑f) (LE.le 0 f)
  -/
  rw [← Subtype.coe_le_coe, Lp.coeFn_nonneg, AddSubmonoid.coe_zero]
  /-
    🎉 no goals
  -/


theorem exists_simpleFunc_nonneg_ae_eq {f : Lp.simpleFunc G p μ} (hf : 0 ≤ f) :
    ∃ f' : α →ₛ G, 0 ≤ f' ∧ f =ᵐ[μ] f' := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G p μ) x
    hf : LE.le 0 f
    ⊢ Exists fun f' => And (LE.le 0 f') ((MeasureTheory.ae μ).EventuallyEq ↑↑↑f ⇑f')
  -/
  rcases f with ⟨⟨f, hp⟩, g, (rfl : _ = f)⟩
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    g : MeasureTheory.SimpleFunc α G
    hp : Membership.mem (MeasureTheory.Lp G p μ) (MeasureTheory.AEEqFun.mk ⇑g ⋯)
    hf : LE.le 0 ⟨⟨MeasureTheory.AEEqFun.mk ⇑g ⋯, hp⟩, ⋯⟩
    ⊢ Exists fun f' => And (LE.le 0 f') ((MeasureTheory.ae μ).EventuallyEq ↑↑↑⟨⟨Me …
  -/
  change 0 ≤ᵐ[μ] g at hf
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    g : MeasureTheory.SimpleFunc α G
    hp : Membership.mem (MeasureTheory.Lp G p μ) (MeasureTheory.AEEqFun.mk ⇑g ⋯)
    hf : (MeasureTheory.ae μ).EventuallyLE 0 ⇑g
    ⊢ Exists fun f' => And (LE.le 0 f') ((MeasureTheory.ae μ).EventuallyEq ↑↑↑⟨⟨Me …
  -/
  refine ⟨g ⊔ 0, le_sup_right, (AEEqFun.coeFn_mk _ _).trans ?_⟩
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    g : MeasureTheory.SimpleFunc α G
    hp : Membership.mem (MeasureTheory.Lp G p μ) (MeasureTheory.AEEqFun.mk ⇑g ⋯)
    hf : (MeasureTheory.ae μ).EventuallyLE 0 ⇑g
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑g ⇑(Max.max g 0)
  -/
  exact hf.mono fun x hx ↦ (sup_of_le_left hx).symm
  /-
    🎉 no goals
  -/


/-- Coercion from nonnegative simple functions of Lp to nonnegative functions of Lp. -/
def coeSimpleFuncNonnegToLpNonneg :
    { g : Lp.simpleFunc G p μ // 0 ≤ g } → { g : Lp G p μ // 0 ≤ g } := fun g => ⟨g, g.2⟩


theorem denseRange_coeSimpleFuncNonnegToLpNonneg [hp : Fact (1 ≤ p)] (hp_ne_top : p ≠ ∞) :
    DenseRange (coeSimpleFuncNonnegToLpNonneg p μ G) := fun g ↦ by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    hp : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    g : Subtype fun g => LE.le 0 g
    ⊢ Membership.mem (closure (Set.range (MeasureTheory.Lp.simpleFunc.coeSimpleFun …
  -/
  borelize G
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    hp : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    g : Subtype fun g => LE.le 0 g
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    ⊢ Membership.mem (closure (Set.range (MeasureTheory.Lp.simpleFunc.coeSimpleFun …
  -/
  rw [mem_closure_iff_seq_limit]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    hp : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    g : Subtype fun g => LE.le 0 g
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    ⊢ Exists fun x => And (∀ (n : Nat), Membership.mem (Set.range (MeasureTheory.L …
  -/
  have hg_memℒp : Memℒp (g : α → G) p μ := Lp.memℒp (g : Lp G p μ)
  have zero_mem : (0 : G) ∈ (range (g : α → G) ∪ {0} : Set G) ∩ { y | 0 ≤ y } := by
    simp only [union_singleton, mem_inter_iff, mem_insert_iff, eq_self_iff_true, true_or,
      mem_setOf_eq, le_refl, and_self_iff]
  have : SeparableSpace ((range (g : α → G) ∪ {0}) ∩ { y | 0 ≤ y } : Set G) := by
    apply IsSeparable.separableSpace
    apply IsSeparable.mono _ Set.inter_subset_left
    exact
      (Lp.stronglyMeasurable (g : Lp G p μ)).isSeparable_range.union
        (finite_singleton _).isSeparable
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    hp : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    g : Subtype fun g => LE.le 0 g
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    hg_memℒp : MeasureTheory.Memℒp (↑↑↑g) p μ
    zero_mem : Membership.mem (Inter.inter (Union.union (Set.range ↑↑↑g) (Singleto …
    this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Union.union (Set.range ↑ …
    ⊢ Exists fun x => And (∀ (n : Nat), Membership.mem (Set.range (MeasureTheory.L …
  -/
  have g_meas : Measurable (g : α → G) := (Lp.stronglyMeasurable (g : Lp G p μ)).measurable
  let x n := SimpleFunc.approxOn (g : α → G) g_meas
    ((range (g : α → G) ∪ {0}) ∩ { y | 0 ≤ y }) 0 zero_mem n
  have hx_nonneg : ∀ n, 0 ≤ x n := by
    intro n a
    change x n a ∈ { y : G | 0 ≤ y }
    have A : (range (g : α → G) ∪ {0} : Set G) ∩ { y | 0 ≤ y } ⊆ { y | 0 ≤ y } :=
      inter_subset_right
    apply A
    exact SimpleFunc.approxOn_mem g_meas _ n a
  have hx_memℒp : ∀ n, Memℒp (x n) p μ :=
    SimpleFunc.memℒp_approxOn _ hg_memℒp _ ⟨aestronglyMeasurable_const, by simp⟩
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    hp : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    g : Subtype fun g => LE.le 0 g
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    hg_memℒp : MeasureTheory.Memℒp (↑↑↑g) p μ
    zero_mem : Membership.mem (Inter.inter (Union.union (Set.range ↑↑↑g) (Singleto …
    this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Union.union (Set.range ↑ …
    g_meas : Measurable ↑↑↑g
    x : Nat → MeasureTheory.SimpleFunc α G := fun n => MeasureTheory.SimpleFunc.ap …
    hx_nonneg : ∀ (n : Nat), LE.le 0 (x n)
    hx_memℒp : ∀ (n : Nat), MeasureTheory.Memℒp (⇑(x n)) p μ
    ⊢ Exists fun x => And (∀ (n : Nat), Membership.mem (Set.range (MeasureTheory.L …
  -/
  have h_toLp := fun n => Memℒp.coeFn_toLp (hx_memℒp n)
  have hx_nonneg_Lp : ∀ n, 0 ≤ toLp (x n) (hx_memℒp n) := by
    intro n
    rw [← Lp.simpleFunc.coeFn_le, Lp.simpleFunc.toLp_eq_toLp]
    filter_upwards [Lp.simpleFunc.coeFn_zero p μ G, h_toLp n] with a ha0 ha_toLp
    rw [ha0, ha_toLp]
    exact hx_nonneg n a
  have hx_tendsto :
      Tendsto (fun n : ℕ => eLpNorm ((x n : α → G) - (g : α → G)) p μ) atTop (𝓝 0) := by
    apply SimpleFunc.tendsto_approxOn_Lp_eLpNorm g_meas zero_mem hp_ne_top
    · have hg_nonneg : (0 : α → G) ≤ᵐ[μ] g := (Lp.coeFn_nonneg _).mpr g.2
      refine hg_nonneg.mono fun a ha => subset_closure ?_
      simpa using ha
    · simp_rw [sub_zero]; exact hg_memℒp.eLpNorm_lt_top
  refine
    ⟨fun n =>
      (coeSimpleFuncNonnegToLpNonneg p μ G) ⟨toLp (x n) (hx_memℒp n), hx_nonneg_Lp n⟩,
      fun n => mem_range_self _, ?_⟩
  suffices Tendsto (fun n : ℕ => (toLp (x n) (hx_memℒp n) : Lp G p μ)) atTop (𝓝 (g : Lp G p μ)) by
    rw [tendsto_iff_dist_tendsto_zero] at this ⊢
    simp_rw [Subtype.dist_eq]
    exact this
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    hp : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    g : Subtype fun g => LE.le 0 g
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    hg_memℒp : MeasureTheory.Memℒp (↑↑↑g) p μ
    zero_mem : Membership.mem (Inter.inter (Union.union (Set.range ↑↑↑g) (Singleto …
    this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Union.union (Set.range ↑ …
    g_meas : Measurable ↑↑↑g
    x : Nat → MeasureTheory.SimpleFunc α G := fun n => MeasureTheory.SimpleFunc.ap …
    hx_nonneg : ∀ (n : Nat), LE.le 0 (x n)
    hx_memℒp : ∀ (n : Nat), MeasureTheory.Memℒp (⇑(x n)) p μ
    h_toLp : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp …
    hx_nonneg_Lp : ∀ (n : Nat), LE.le 0 (MeasureTheory.Lp.simpleFunc.toLp (x n) ⋯)
    hx_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ⇑(x n)  …
    ⊢ Filter.Tendsto (fun n => ↑(MeasureTheory.Lp.simpleFunc.toLp (x n) ⋯)) Filter …
  -/
  rw [Lp.tendsto_Lp_iff_tendsto_ℒp']
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    G : Type u_7
    inst✝ : NormedLatticeAddCommGroup G
    hp : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    g : Subtype fun g => LE.le 0 g
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    hg_memℒp : MeasureTheory.Memℒp (↑↑↑g) p μ
    zero_mem : Membership.mem (Inter.inter (Union.union (Set.range ↑↑↑g) (Singleto …
    this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Union.union (Set.range ↑ …
    g_meas : Measurable ↑↑↑g
    x : Nat → MeasureTheory.SimpleFunc α G := fun n => MeasureTheory.SimpleFunc.ap …
    hx_nonneg : ∀ (n : Nat), LE.le 0 (x n)
    hx_memℒp : ∀ (n : Nat), MeasureTheory.Memℒp (⇑(x n)) p μ
    h_toLp : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp …
    hx_nonneg_Lp : ∀ (n : Nat), LE.le 0 (MeasureTheory.Lp.simpleFunc.toLp (x n) ⋯)
    hx_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ⇑(x n)  …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ↑↑↑(MeasureTheory. …
  -/
  refine Filter.Tendsto.congr (fun n => eLpNorm_congr_ae (EventuallyEq.sub ?_ ?_)) hx_tendsto
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      G : Type u_7
      inst✝ : NormedLatticeAddCommGroup G
      hp : Fact (LE.le 1 p)
      hp_ne_top : Ne p Top.top
      g : Subtype fun g => LE.le 0 g
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      hg_memℒp : MeasureTheory.Memℒp (↑↑↑g) p μ
      zero_mem : Membership.mem (Inter.inter (Union.union (Set.range ↑↑↑g) (Singleto …
      this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Union.union (Set.range ↑ …
      g_meas : Measurable ↑↑↑g
      x : Nat → MeasureTheory.SimpleFunc α G := fun n => MeasureTheory.SimpleFunc.ap …
      hx_nonneg : ∀ (n : Nat), LE.le 0 (x n)
      hx_memℒp : ∀ (n : Nat), MeasureTheory.Memℒp (⇑(x n)) p μ
      h_toLp : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp …
      hx_nonneg_Lp : ∀ (n : Nat), LE.le 0 (MeasureTheory.Lp.simpleFunc.toLp (x n) ⋯)
      hx_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ⇑(x n)  …
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(x n) ↑↑↑(MeasureTheory.Lp.simpleFunc.toL …
    -/
  · symm
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      G : Type u_7
      inst✝ : NormedLatticeAddCommGroup G
      hp : Fact (LE.le 1 p)
      hp_ne_top : Ne p Top.top
      g : Subtype fun g => LE.le 0 g
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      hg_memℒp : MeasureTheory.Memℒp (↑↑↑g) p μ
      zero_mem : Membership.mem (Inter.inter (Union.union (Set.range ↑↑↑g) (Singleto …
      this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Union.union (Set.range ↑ …
      g_meas : Measurable ↑↑↑g
      x : Nat → MeasureTheory.SimpleFunc α G := fun n => MeasureTheory.SimpleFunc.ap …
      hx_nonneg : ∀ (n : Nat), LE.le 0 (x n)
      hx_memℒp : ∀ (n : Nat), MeasureTheory.Memℒp (⇑(x n)) p μ
      h_toLp : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp …
      hx_nonneg_Lp : ∀ (n : Nat), LE.le 0 (MeasureTheory.Lp.simpleFunc.toLp (x n) ⋯)
      hx_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ⇑(x n)  …
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑(MeasureTheory.Lp.simpleFunc.toLp (x n) …
    -/
    rw [Lp.simpleFunc.toLp_eq_toLp]
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      G : Type u_7
      inst✝ : NormedLatticeAddCommGroup G
      hp : Fact (LE.le 1 p)
      hp_ne_top : Ne p Top.top
      g : Subtype fun g => LE.le 0 g
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      hg_memℒp : MeasureTheory.Memℒp (↑↑↑g) p μ
      zero_mem : Membership.mem (Inter.inter (Union.union (Set.range ↑↑↑g) (Singleto …
      this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Union.union (Set.range ↑ …
      g_meas : Measurable ↑↑↑g
      x : Nat → MeasureTheory.SimpleFunc α G := fun n => MeasureTheory.SimpleFunc.ap …
      hx_nonneg : ∀ (n : Nat), LE.le 0 (x n)
      hx_memℒp : ∀ (n : Nat), MeasureTheory.Memℒp (⇑(x n)) p μ
      h_toLp : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp …
      hx_nonneg_Lp : ∀ (n : Nat), LE.le 0 (MeasureTheory.Lp.simpleFunc.toLp (x n) ⋯)
      hx_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ⇑(x n)  …
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp ⇑(x n) ⋯) ⇑(x n)
    -/
    exact h_toLp n
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      G : Type u_7
      inst✝ : NormedLatticeAddCommGroup G
      hp : Fact (LE.le 1 p)
      hp_ne_top : Ne p Top.top
      g : Subtype fun g => LE.le 0 g
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      hg_memℒp : MeasureTheory.Memℒp (↑↑↑g) p μ
      zero_mem : Membership.mem (Inter.inter (Union.union (Set.range ↑↑↑g) (Singleto …
      this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Union.union (Set.range ↑ …
      g_meas : Measurable ↑↑↑g
      x : Nat → MeasureTheory.SimpleFunc α G := fun n => MeasureTheory.SimpleFunc.ap …
      hx_nonneg : ∀ (n : Nat), LE.le 0 (x n)
      hx_memℒp : ∀ (n : Nat), MeasureTheory.Memℒp (⇑(x n)) p μ
      h_toLp : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp …
      hx_nonneg_Lp : ∀ (n : Nat), LE.le 0 (MeasureTheory.Lp.simpleFunc.toLp (x n) ⋯)
      hx_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ⇑(x n)  …
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑g ↑↑↑g
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- To prove something for an arbitrary `Lp` function in a second countable Borel normed group, it
suffices to show that
* the property holds for (multiples of) characteristic functions;
* is closed under addition;
* the set of functions in `Lp` for which the property holds is closed.
-/
@[elab_as_elim]
theorem Lp.induction [_i : Fact (1 ≤ p)] (hp_ne_top : p ≠ ∞) (P : Lp E p μ → Prop)
    (h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : μ s < ∞),
      P (Lp.simpleFunc.indicatorConst p hs hμs.ne c))
    (h_add : ∀ ⦃f g⦄, ∀ hf : Memℒp f p μ, ∀ hg : Memℒp g p μ, Disjoint (support f) (support g) →
      P (hf.toLp f) → P (hg.toLp g) → P (hf.toLp f + hg.toLp g))
    (h_closed : IsClosed { f : Lp E p μ | P f }) : ∀ f : Lp E p μ, P f := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    _i : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    P : (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) → Prop
    h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
    h_add : ∀ ⦃f g : α → E⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
    h_closed : IsClosed (setOf fun f => P f)
    ⊢ ∀ (f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x), P f
  -/
  refine fun f => (Lp.simpleFunc.denseRange hp_ne_top).induction_on f h_closed ?_
  refine Lp.simpleFunc.induction (α := α) (E := E) (lt_of_lt_of_le zero_lt_one _i.elim).ne'
    hp_ne_top ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      _i : Fact (LE.le 1 p)
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) → Prop
      h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → E⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.top), P  …
    -/
  · exact fun c s => h_ind c
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      _i : Fact (LE.le 1 p)
      hp_ne_top : Ne p Top.top
      P : (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x) → Prop
      h_ind : ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.to …
      h_add : ∀ ⦃f g : α → E⦄ (hf : MeasureTheory.Memℒp f p μ) (hg : MeasureTheory.M …
      h_closed : IsClosed (setOf fun f => P f)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄ (hf : MeasureTheory.Memℒp (⇑f) p μ) ( …
    -/
  · exact fun f g hf hg => h_add hf hg
    /-
      🎉 no goals
    -/


/-- To prove something for an arbitrary `Memℒp` function in a second countable
Borel normed group, it suffices to show that
* the property holds for (multiples of) characteristic functions;
* is closed under addition;
* the set of functions in the `Lᵖ` space for which the property holds is closed.
* the property is closed under the almost-everywhere equal relation.

It is possible to make the hypotheses in the induction steps a bit stronger, and such conditions
can be added once we need them (for example in `h_add` it is only necessary to consider the sum of
a simple function with a multiple of a characteristic function and that the intersection
of their images is a subset of `{0}`).
-/
@[elab_as_elim]
theorem Memℒp.induction [_i : Fact (1 ≤ p)] (hp_ne_top : p ≠ ∞) (P : (α → E) → Prop)
    (h_ind : ∀ (c : E) ⦃s⦄, MeasurableSet s → μ s < ∞ → P (s.indicator fun _ => c))
    (h_add : ∀ ⦃f g : α → E⦄, Disjoint (support f) (support g) → Memℒp f p μ → Memℒp g p μ →
      P f → P g → P (f + g))
    (h_closed : IsClosed { f : Lp E p μ | P f })
    (h_ae : ∀ ⦃f g⦄, f =ᵐ[μ] g → Memℒp f p μ → P f → P g) :
    ∀ ⦃f : α → E⦄, Memℒp f p μ → P f := by
  have : ∀ f : SimpleFunc α E, Memℒp f p μ → P f := by
    apply SimpleFunc.induction
    · intro c s hs h
      by_cases hc : c = 0
      · subst hc; convert h_ind 0 MeasurableSet.empty (by simp) using 1; ext; simp [const]
      have hp_pos : p ≠ 0 := (lt_of_lt_of_le zero_lt_one _i.elim).ne'
      exact h_ind c hs (SimpleFunc.measure_lt_top_of_memℒp_indicator hp_pos hp_ne_top hc hs h)
    · intro f g hfg hf hg int_fg
      rw [SimpleFunc.coe_add,
        memℒp_add_of_disjoint hfg f.stronglyMeasurable g.stronglyMeasurable] at int_fg
      exact h_add hfg int_fg.1 int_fg.2 (hf int_fg.1) (hg int_fg.2)
  have : ∀ f : Lp.simpleFunc E p μ, P f := by
    intro f
    exact
      h_ae (Lp.simpleFunc.toSimpleFunc_eq_toFun f) (Lp.simpleFunc.memℒp f)
        (this (Lp.simpleFunc.toSimpleFunc f) (Lp.simpleFunc.memℒp f))
  have : ∀ f : Lp E p μ, P f := fun f =>
    (Lp.simpleFunc.denseRange hp_ne_top).induction_on f h_closed this
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    _i : Fact (LE.le 1 p)
    hp_ne_top : Ne p Top.top
    P : (α → E) → Prop
    h_ind : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_add : ∀ ⦃f g : α → E⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_closed : IsClosed (setOf fun f => P ↑↑f)
    h_ae : ∀ ⦃f g : α → E⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    this✝¹ : ∀ (f : MeasureTheory.SimpleFunc α E), MeasureTheory.Memℒp (⇑f) p μ →  …
    this✝ : ∀ (f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E  …
    this : ∀ (f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x), P ↑↑f
    ⊢ ∀ ⦃f : α → E⦄, MeasureTheory.Memℒp f p μ → P f
  -/
  exact fun f hf => h_ae hf.coeFn_toLp (Lp.memℒp _) (this (hf.toLp f))
  /-
    🎉 no goals
  -/


/-- If a set of ae strongly measurable functions is stable under addition and approximates
characteristic functions in `ℒp`, then it is dense in `ℒp`. -/
theorem Memℒp.induction_dense (hp_ne_top : p ≠ ∞) (P : (α → E) → Prop)
    (h0P :
      ∀ (c : E) ⦃s : Set α⦄,
        MeasurableSet s →
          μ s < ∞ →
            ∀ {ε : ℝ≥0∞}, ε ≠ 0 → ∃ g : α → E, eLpNorm (g - s.indicator fun _ => c) p μ ≤ ε ∧ P g)
    (h1P : ∀ f g, P f → P g → P (f + g)) (h2P : ∀ f, P f → AEStronglyMeasurable f μ) {f : α → E}
    (hf : Memℒp f p μ) {ε : ℝ≥0∞} (hε : ε ≠ 0) : ∃ g : α → E, eLpNorm (f - g) p μ ≤ ε ∧ P g := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    hp_ne_top : Ne p Top.top
    P : (α → E) → Prop
    h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
    h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
    h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub f g) p μ) ε) (P …
  -/
  rcases eq_or_ne p 0 with (rfl | hp_pos)
  · rcases h0P (0 : E) MeasurableSet.empty (by simp only [measure_empty, zero_lt_top])
        hε with ⟨g, _, Pg⟩
    /-
      case inl.intro.intro
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      μ : MeasureTheory.Measure α
      P : (α → E) → Prop
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f : α → E
      ε : ENNReal
      hε : Ne ε 0
      hp_ne_top : Ne 0 Top.top
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      hf : MeasureTheory.Memℒp f 0 μ
      g : α → E
      left✝ : LE.le (MeasureTheory.eLpNorm (HSub.hSub g (EmptyCollection.emptyCollec …
      Pg : P g
      ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub f g) 0 μ) ε) (P …
    -/
    exact ⟨g, by simp only [eLpNorm_exponent_zero, zero_le'], Pg⟩
    /-
      🎉 no goals
    -/
  suffices H : ∀ (f' : α →ₛ E) (δ : ℝ≥0∞) (hδ : δ ≠ 0), Memℒp f' p μ →
      ∃ g, eLpNorm (⇑f' - g) p μ ≤ δ ∧ P g by
    obtain ⟨η, ηpos, hη⟩ := exists_Lp_half E μ p hε
    rcases hf.exists_simpleFunc_eLpNorm_sub_lt hp_ne_top ηpos.ne' with ⟨f', hf', f'_mem⟩
    rcases H f' η ηpos.ne' f'_mem with ⟨g, hg, Pg⟩
    refine ⟨g, ?_, Pg⟩
    convert (hη _ _ (hf.aestronglyMeasurable.sub f'.aestronglyMeasurable)
          (f'.aestronglyMeasurable.sub (h2P g Pg)) hf'.le hg).le using 2
    simp only [sub_add_sub_cancel]
  /-
    case inr
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure α
    hp_ne_top : Ne p Top.top
    P : (α → E) → Prop
    h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
    h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
    h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ε : ENNReal
    hε : Ne ε 0
    hp_pos : Ne p 0
    ⊢ ∀ (f' : MeasureTheory.SimpleFunc α E) (δ : ENNReal), Ne δ 0 → MeasureTheory. …
  -/
  apply SimpleFunc.induction
    /-
      case inr.h_ind
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      ε : ENNReal
      hε : Ne ε 0
      hp_pos : Ne p 0
      ⊢ ∀ (c : E) {s : Set α} (hs : MeasurableSet s) (δ : ENNReal), Ne δ 0 → Measure …
    -/
  · intro c s hs ε εpos Hs
    /-
      case inr.h_ind
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      ε✝ : ENNReal
      hε : Ne ε✝ 0
      hp_pos : Ne p 0
      c : E
      s : Set α
      hs : MeasurableSet s
      ε : ENNReal
      εpos : Ne ε 0
      Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
      ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheor …
    -/
    rcases eq_or_ne c 0 with (rfl | hc)
    · rcases h0P (0 : E) MeasurableSet.empty (by simp only [measure_empty, zero_lt_top])
          εpos with ⟨g, hg, Pg⟩
      /-
        case inr.h_ind.inl.intro.intro
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_ne_top : Ne p Top.top
        P : (α → E) → Prop
        h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
        h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
        h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        ε✝ : ENNReal
        hε : Ne ε✝ 0
        hp_pos : Ne p 0
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        εpos : Ne ε 0
        Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        g : α → E
        hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub g (EmptyCollection.emptyCollectio …
        Pg : P g
        ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheor …
      -/
      rw [← eLpNorm_neg, neg_sub] at hg
      /-
        case inr.h_ind.inl.intro.intro
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_ne_top : Ne p Top.top
        P : (α → E) → Prop
        h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
        h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
        h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        ε✝ : ENNReal
        hε : Ne ε✝ 0
        hp_pos : Ne p 0
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        εpos : Ne ε 0
        Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        g : α → E
        hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub (EmptyCollection.emptyCollection. …
        Pg : P g
        ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheor …
      -/
      refine ⟨g, ?_, Pg⟩
      /-
        case inr.h_ind.inl.intro.intro
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_ne_top : Ne p Top.top
        P : (α → E) → Prop
        h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
        h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
        h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        ε✝ : ENNReal
        hε : Ne ε✝ 0
        hp_pos : Ne p 0
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        εpos : Ne ε 0
        Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        g : α → E
        hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub (EmptyCollection.emptyCollection. …
        Pg : P g
        ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheory.SimpleFunc.piecewis …
      -/
      convert hg
      /-
        case h.e'_3.h.e'_5.h.e'_5
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_ne_top : Ne p Top.top
        P : (α → E) → Prop
        h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
        h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
        h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        ε✝ : ENNReal
        hε : Ne ε✝ 0
        hp_pos : Ne p 0
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        εpos : Ne ε 0
        Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        g : α → E
        hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub (EmptyCollection.emptyCollection. …
        Pg : P g
        ⊢ Eq (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTheory.SimpleFunc.cons …
      -/
      ext x
      simp only [SimpleFunc.const_zero, SimpleFunc.coe_piecewise, SimpleFunc.coe_zero,
        piecewise_eq_indicator, indicator_zero', Pi.zero_apply, indicator_zero]
      /-
        case inr.h_ind.inr
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_ne_top : Ne p Top.top
        P : (α → E) → Prop
        h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
        h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
        h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        ε✝ : ENNReal
        hε : Ne ε✝ 0
        hp_pos : Ne p 0
        c : E
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        εpos : Ne ε 0
        Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        hc : Ne c 0
        ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheor …
      -/
    · have : μ s < ∞ := SimpleFunc.measure_lt_top_of_memℒp_indicator hp_pos hp_ne_top hc hs Hs
      /-
        case inr.h_ind.inr
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_ne_top : Ne p Top.top
        P : (α → E) → Prop
        h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
        h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
        h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        ε✝ : ENNReal
        hε : Ne ε✝ 0
        hp_pos : Ne p 0
        c : E
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        εpos : Ne ε 0
        Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        hc : Ne c 0
        this : LT.lt (μ s) Top.top
        ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheor …
      -/
      rcases h0P c hs this εpos with ⟨g, hg, Pg⟩
      /-
        case inr.h_ind.inr.intro.intro
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_ne_top : Ne p Top.top
        P : (α → E) → Prop
        h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
        h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
        h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        ε✝ : ENNReal
        hε : Ne ε✝ 0
        hp_pos : Ne p 0
        c : E
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        εpos : Ne ε 0
        Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        hc : Ne c 0
        this : LT.lt (μ s) Top.top
        g : α → E
        hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub g (s.indicator fun x => c)) p μ) ε
        Pg : P g
        ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheor …
      -/
      rw [← eLpNorm_neg, neg_sub] at hg
      /-
        case inr.h_ind.inr.intro.intro
        α : Type u_1
        E : Type u_4
        inst✝¹ : MeasurableSpace α
        inst✝ : NormedAddCommGroup E
        p : ENNReal
        μ : MeasureTheory.Measure α
        hp_ne_top : Ne p Top.top
        P : (α → E) → Prop
        h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
        h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
        h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        ε✝ : ENNReal
        hε : Ne ε✝ 0
        hp_pos : Ne p 0
        c : E
        s : Set α
        hs : MeasurableSet s
        ε : ENNReal
        εpos : Ne ε 0
        Hs : MeasureTheory.Memℒp (⇑(MeasureTheory.SimpleFunc.piecewise s hs (MeasureTh …
        hc : Ne c 0
        this : LT.lt (μ s) Top.top
        g : α → E
        hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub (s.indicator fun x => c) g) p μ) ε
        Pg : P g
        ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(MeasureTheor …
      -/
      exact ⟨g, hg, Pg⟩
      /-
        🎉 no goals
      -/
    /-
      case inr.h_add
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      ε : ENNReal
      hε : Ne ε 0
      hp_pos : Ne p 0
      ⊢ ∀ ⦃f g : MeasureTheory.SimpleFunc α E⦄, Disjoint (Function.support ⇑f) (Func …
    -/
  · intro f f' hff' hf hf' δ δpos int_ff'
    /-
      case inr.h_add
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f✝ : α → E
      hf✝ : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      hp_pos : Ne p 0
      f f' : MeasureTheory.SimpleFunc α E
      hff' : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f) p μ → Exists fun g =>  …
      hf' : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f') p μ → Exists fun g = …
      δ : ENNReal
      δpos : Ne δ 0
      int_ff' : MeasureTheory.Memℒp (⇑(HAdd.hAdd f f')) p μ
      ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(HAdd.hAdd f  …
    -/
    obtain ⟨η, ηpos, hη⟩ := exists_Lp_half E μ p δpos
    rw [SimpleFunc.coe_add,
      memℒp_add_of_disjoint hff' f.stronglyMeasurable f'.stronglyMeasurable] at int_ff'
    /-
      case inr.h_add.intro.intro
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f✝ : α → E
      hf✝ : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      hp_pos : Ne p 0
      f f' : MeasureTheory.SimpleFunc α E
      hff' : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f) p μ → Exists fun g =>  …
      hf' : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f') p μ → Exists fun g = …
      δ : ENNReal
      δpos : Ne δ 0
      int_ff' : And (MeasureTheory.Memℒp (⇑f) p μ) (MeasureTheory.Memℒp (⇑f') p μ)
      η : ENNReal
      ηpos : LT.lt 0 η
      hη : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(HAdd.hAdd f  …
    -/
    rcases hf η ηpos.ne' int_ff'.1 with ⟨g, hg, Pg⟩
    /-
      case inr.h_add.intro.intro.intro.intro
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f✝ : α → E
      hf✝ : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      hp_pos : Ne p 0
      f f' : MeasureTheory.SimpleFunc α E
      hff' : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f) p μ → Exists fun g =>  …
      hf' : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f') p μ → Exists fun g = …
      δ : ENNReal
      δpos : Ne δ 0
      int_ff' : And (MeasureTheory.Memℒp (⇑f) p μ) (MeasureTheory.Memℒp (⇑f') p μ)
      η : ENNReal
      ηpos : LT.lt 0 η
      hη : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      g : α → E
      hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑f) g) p μ) η
      Pg : P g
      ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(HAdd.hAdd f  …
    -/
    rcases hf' η ηpos.ne' int_ff'.2 with ⟨g', hg', Pg'⟩
    /-
      case inr.h_add.intro.intro.intro.intro.intro.intro
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f✝ : α → E
      hf✝ : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      hp_pos : Ne p 0
      f f' : MeasureTheory.SimpleFunc α E
      hff' : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f) p μ → Exists fun g =>  …
      hf' : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f') p μ → Exists fun g = …
      δ : ENNReal
      δpos : Ne δ 0
      int_ff' : And (MeasureTheory.Memℒp (⇑f) p μ) (MeasureTheory.Memℒp (⇑f') p μ)
      η : ENNReal
      ηpos : LT.lt 0 η
      hη : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      g : α → E
      hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑f) g) p μ) η
      Pg : P g
      g' : α → E
      hg' : LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑f') g') p μ) η
      Pg' : P g'
      ⊢ Exists fun g => And (LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑(HAdd.hAdd f  …
    -/
    refine ⟨g + g', ?_, h1P g g' Pg Pg'⟩
    convert (hη _ _ (f.aestronglyMeasurable.sub (h2P g Pg))
          (f'.aestronglyMeasurable.sub (h2P g' Pg')) hg hg').le using 2
    /-
      case h.e'_3.h.e'_5
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f✝ : α → E
      hf✝ : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      hp_pos : Ne p 0
      f f' : MeasureTheory.SimpleFunc α E
      hff' : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f) p μ → Exists fun g =>  …
      hf' : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f') p μ → Exists fun g = …
      δ : ENNReal
      δpos : Ne δ 0
      int_ff' : And (MeasureTheory.Memℒp (⇑f) p μ) (MeasureTheory.Memℒp (⇑f') p μ)
      η : ENNReal
      ηpos : LT.lt 0 η
      hη : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      g : α → E
      hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑f) g) p μ) η
      Pg : P g
      g' : α → E
      hg' : LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑f') g') p μ) η
      Pg' : P g'
      ⊢ Eq (HSub.hSub (⇑(HAdd.hAdd f f')) (HAdd.hAdd g g')) (HAdd.hAdd (HSub.hSub (⇑ …
    -/
    rw [SimpleFunc.coe_add]
    /-
      case h.e'_3.h.e'_5
      α : Type u_1
      E : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : NormedAddCommGroup E
      p : ENNReal
      μ : MeasureTheory.Measure α
      hp_ne_top : Ne p Top.top
      P : (α → E) → Prop
      h0P : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → ∀ {ε : EN …
      h1P : ∀ (f g : α → E), P f → P g → P (HAdd.hAdd f g)
      h2P : ∀ (f : α → E), P f → MeasureTheory.AEStronglyMeasurable f μ
      f✝ : α → E
      hf✝ : MeasureTheory.Memℒp f✝ p μ
      ε : ENNReal
      hε : Ne ε 0
      hp_pos : Ne p 0
      f f' : MeasureTheory.SimpleFunc α E
      hff' : Disjoint (Function.support ⇑f) (Function.support ⇑f')
      hf : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f) p μ → Exists fun g =>  …
      hf' : ∀ (δ : ENNReal), Ne δ 0 → MeasureTheory.Memℒp (⇑f') p μ → Exists fun g = …
      δ : ENNReal
      δpos : Ne δ 0
      int_ff' : And (MeasureTheory.Memℒp (⇑f) p μ) (MeasureTheory.Memℒp (⇑f') p μ)
      η : ENNReal
      ηpos : LT.lt 0 η
      hη : ∀ (f g : α → E), MeasureTheory.AEStronglyMeasurable f μ → MeasureTheory.A …
      g : α → E
      hg : LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑f) g) p μ) η
      Pg : P g
      g' : α → E
      hg' : LE.le (MeasureTheory.eLpNorm (HSub.hSub (⇑f') g') p μ) η
      Pg' : P g'
      ⊢ Eq (HSub.hSub (HAdd.hAdd ⇑f ⇑f') (HAdd.hAdd g g')) (HAdd.hAdd (HSub.hSub (⇑f …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


@[inherit_doc MeasureTheory.Lp.simpleFunc]
notation:25 α " →₁ₛ[" μ "] " E => @MeasureTheory.Lp.simpleFunc α E _ _ 1 μ


theorem L1.SimpleFunc.toLp_one_eq_toL1 (f : α →ₛ E) (hf : Integrable f μ) :
    (Lp.simpleFunc.toLp f (memℒp_one_iff_integrable.2 hf) : α →₁[μ] E) = hf.toL1 f :=
  rfl


protected theorem L1.SimpleFunc.integrable (f : α →₁ₛ[μ] E) :
    Integrable (Lp.simpleFunc.toSimpleFunc f) μ := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ MeasureTheory.Integrable (⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc f)) μ
  -/
  rw [← memℒp_one_iff_integrable]; exact Lp.simpleFunc.memℒp f
                                   /-
                                     🎉 no goals
                                   -/


/-- To prove something for an arbitrary integrable function in a normed group,
it suffices to show that
* the property holds for (multiples of) characteristic functions;
* is closed under addition;
* the set of functions in the `L¹` space for which the property holds is closed.
* the property is closed under the almost-everywhere equal relation.

It is possible to make the hypotheses in the induction steps a bit stronger, and such conditions
can be added once we need them (for example in `h_add` it is only necessary to consider the sum of
a simple function with a multiple of a characteristic function and that the intersection
of their images is a subset of `{0}`).
-/
@[elab_as_elim]
theorem Integrable.induction (P : (α → E) → Prop)
    (h_ind : ∀ (c : E) ⦃s⦄, MeasurableSet s → μ s < ∞ → P (s.indicator fun _ => c))
    (h_add :
      ∀ ⦃f g : α → E⦄,
        Disjoint (support f) (support g) → Integrable f μ → Integrable g μ → P f → P g → P (f + g))
    (h_closed : IsClosed { f : α →₁[μ] E | P f })
    (h_ae : ∀ ⦃f g⦄, f =ᵐ[μ] g → Integrable f μ → P f → P g) :
    ∀ ⦃f : α → E⦄, Integrable f μ → P f := by
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    P : (α → E) → Prop
    h_ind : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_add : ∀ ⦃f g : α → E⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_closed : IsClosed (setOf fun f => P ↑↑f)
    h_ae : ∀ ⦃f g : α → E⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    ⊢ ∀ ⦃f : α → E⦄, MeasureTheory.Integrable f μ → P f
  -/
  simp only [← memℒp_one_iff_integrable] at *
  /-
    α : Type u_1
    E : Type u_4
    inst✝¹ : MeasurableSpace α
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure α
    P : (α → E) → Prop
    h_ind : ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P (s.in …
    h_closed : IsClosed (setOf fun f => P ↑↑f)
    h_add : ∀ ⦃f g : α → E⦄, Disjoint (Function.support f) (Function.support g) →  …
    h_ae : ∀ ⦃f g : α → E⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory. …
    ⊢ ∀ ⦃f : α → E⦄, MeasureTheory.Memℒp f 1 μ → P f
  -/
  exact Memℒp.induction one_ne_top (P := P) h_ind h_add h_closed h_ae
  /-
    🎉 no goals
  -/


