/-- Differentiation under integral of `x ↦ ∫ t in a..b, F x t` at a given point `x₀`, assuming
`F x₀` is integrable, `x ↦ F x a` is locally Lipschitz on a ball around `x₀` for ae `a`
(with a ball radius independent of `a`) with integrable Lipschitz bound, and `F x` is ae-measurable
for `x` in a possibly smaller neighborhood of `x₀`. -/
nonrec theorem hasFDerivAt_integral_of_dominated_loc_of_lip
    {F : H → ℝ → E} {F' : ℝ → H →L[𝕜] E} {x₀ : H}
    (ε_pos : 0 < ε) (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) (μ.restrict (Ι a b)))
    (hF_int : IntervalIntegrable (F x₀) μ a b)
    (hF'_meas : AEStronglyMeasurable F' (μ.restrict (Ι a b)))
    (h_lip : ∀ᵐ t ∂μ, t ∈ Ι a b →
      LipschitzOnWith (Real.nnabs <| bound t) (fun x => F x t) (ball x₀ ε))
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_diff : ∀ᵐ t ∂μ, t ∈ Ι a b → HasFDerivAt (fun x => F x t) (F' t) x₀) :
    IntervalIntegrable F' μ a b ∧
      HasFDerivAt (fun x => ∫ t in a..b, F x t ∂μ) (∫ t in a..b, F' t ∂μ) x₀ := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    a b ε : Real
    bound : Real → Real
    F : H → Real → E
    F' : Real → ContinuousLinearMap (RingHom.id 𝕜) H E
    x₀ : H
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lip : Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → Lipschit …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → HasFDer …
    ⊢ And (IntervalIntegrable F' μ a b) (HasFDerivAt (fun x => intervalIntegral (f …
  -/
  rw [← ae_restrict_iff' measurableSet_uIoc] at h_lip h_diff
  /-
    𝕜 : Type u_1
    inst✝⁵ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    a b ε : Real
    bound : Real → Real
    F : H → Real → E
    F' : Real → ContinuousLinearMap (RingHom.id 𝕜) H E
    x₀ : H
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lip : Filter.Eventually (fun x => LipschitzOnWith (Real.nnabs (bound x)) (fu …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun x => HasFDerivAt (fun x_1 => F x_1 x) (F' x) x …
    ⊢ And (IntervalIntegrable F' μ a b) (HasFDerivAt (fun x => intervalIntegral (f …
  -/
  simp only [intervalIntegrable_iff] at hF_int bound_integrable ⊢
  /-
    𝕜 : Type u_1
    inst✝⁵ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    a b ε : Real
    bound : Real → Real
    F : H → Real → E
    F' : Real → ContinuousLinearMap (RingHom.id 𝕜) H E
    x₀ : H
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lip : Filter.Eventually (fun x => LipschitzOnWith (Real.nnabs (bound x)) (fu …
    h_diff : Filter.Eventually (fun x => HasFDerivAt (fun x_1 => F x_1 x) (F' x) x …
    hF_int : MeasureTheory.IntegrableOn (F x₀) (Set.uIoc a b) μ
    bound_integrable : MeasureTheory.IntegrableOn bound (Set.uIoc a b) μ
    ⊢ And (MeasureTheory.IntegrableOn F' (Set.uIoc a b) μ) (HasFDerivAt (fun x =>  …
  -/
  simp only [intervalIntegral_eq_integral_uIoc]
  have := hasFDerivAt_integral_of_dominated_loc_of_lip ε_pos hF_meas hF_int hF'_meas h_lip
    bound_integrable h_diff
  /-
    𝕜 : Type u_1
    inst✝⁵ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    a b ε : Real
    bound : Real → Real
    F : H → Real → E
    F' : Real → ContinuousLinearMap (RingHom.id 𝕜) H E
    x₀ : H
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lip : Filter.Eventually (fun x => LipschitzOnWith (Real.nnabs (bound x)) (fu …
    h_diff : Filter.Eventually (fun x => HasFDerivAt (fun x_1 => F x_1 x) (F' x) x …
    hF_int : MeasureTheory.IntegrableOn (F x₀) (Set.uIoc a b) μ
    bound_integrable : MeasureTheory.IntegrableOn bound (Set.uIoc a b) μ
    this : And (MeasureTheory.Integrable F' (μ.restrict (Set.uIoc a b))) (HasFDeri …
    ⊢ And (MeasureTheory.IntegrableOn F' (Set.uIoc a b) μ) (HasFDerivAt (fun x =>  …
  -/
  exact ⟨this.1, this.2.const_smul _⟩
  /-
    🎉 no goals
  -/


/-- Differentiation under integral of `x ↦ ∫ F x a` at a given point `x₀`, assuming
`F x₀` is integrable, `x ↦ F x a` is differentiable on a ball around `x₀` for ae `a` with
derivative norm uniformly bounded by an integrable function (the ball radius is independent of `a`),
and `F x` is ae-measurable for `x` in a possibly smaller neighborhood of `x₀`. -/
nonrec theorem hasFDerivAt_integral_of_dominated_of_fderiv_le
    {F : H → ℝ → E} {F' : H → ℝ → H →L[𝕜] E} {x₀ : H} (ε_pos : 0 < ε)
    (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) (μ.restrict (Ι a b)))
    (hF_int : IntervalIntegrable (F x₀) μ a b)
    (hF'_meas : AEStronglyMeasurable (F' x₀) (μ.restrict (Ι a b)))
    (h_bound : ∀ᵐ t ∂μ, t ∈ Ι a b → ∀ x ∈ ball x₀ ε, ‖F' x t‖ ≤ bound t)
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_diff : ∀ᵐ t ∂μ, t ∈ Ι a b → ∀ x ∈ ball x₀ ε, HasFDerivAt (fun x => F x t) (F' x t) x) :
    HasFDerivAt (fun x => ∫ t in a..b, F x t ∂μ) (∫ t in a..b, F' x₀ t ∂μ) x₀ := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    a b ε : Real
    bound : Real → Real
    F : H → Real → E
    F' : H → Real → ContinuousLinearMap (RingHom.id 𝕜) H E
    x₀ : H
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → ∀ (x : …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → ∀ (x :  …
    ⊢ HasFDerivAt (fun x => intervalIntegral (fun t => F x t) a b μ) (intervalInte …
  -/
  rw [← ae_restrict_iff' measurableSet_uIoc] at h_bound h_diff
  /-
    𝕜 : Type u_1
    inst✝⁵ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    a b ε : Real
    bound : Real → Real
    F : H → Real → E
    F' : H → Real → ContinuousLinearMap (RingHom.id 𝕜) H E
    x₀ : H
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : Filter.Eventually (fun x => ∀ (x_1 : H), Membership.mem (Metric.ball …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun x => ∀ (x_1 : H), Membership.mem (Metric.ball  …
    ⊢ HasFDerivAt (fun x => intervalIntegral (fun t => F x t) a b μ) (intervalInte …
  -/
  simp only [intervalIntegrable_iff] at hF_int bound_integrable
  /-
    𝕜 : Type u_1
    inst✝⁵ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    a b ε : Real
    bound : Real → Real
    F : H → Real → E
    F' : H → Real → ContinuousLinearMap (RingHom.id 𝕜) H E
    x₀ : H
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : Filter.Eventually (fun x => ∀ (x_1 : H), Membership.mem (Metric.ball …
    h_diff : Filter.Eventually (fun x => ∀ (x_1 : H), Membership.mem (Metric.ball  …
    hF_int : MeasureTheory.IntegrableOn (F x₀) (Set.uIoc a b) μ
    bound_integrable : MeasureTheory.IntegrableOn bound (Set.uIoc a b) μ
    ⊢ HasFDerivAt (fun x => intervalIntegral (fun t => F x t) a b μ) (intervalInte …
  -/
  simp only [intervalIntegral_eq_integral_uIoc]
  exact (hasFDerivAt_integral_of_dominated_of_fderiv_le ε_pos hF_meas hF_int hF'_meas h_bound
    bound_integrable h_diff).const_smul _


/-- Derivative under integral of `x ↦ ∫ F x a` at a given point `x₀ : 𝕜`, `𝕜 = ℝ` or `𝕜 = ℂ`,
assuming `F x₀` is integrable, `x ↦ F x a` is locally Lipschitz on a ball around `x₀` for ae `a`
(with ball radius independent of `a`) with integrable Lipschitz bound, and `F x` is
ae-measurable for `x` in a possibly smaller neighborhood of `x₀`. -/
nonrec theorem hasDerivAt_integral_of_dominated_loc_of_lip {F : 𝕜 → ℝ → E} {F' : ℝ → E} {x₀ : 𝕜}
    (ε_pos : 0 < ε) (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) (μ.restrict (Ι a b)))
    (hF_int : IntervalIntegrable (F x₀) μ a b)
    (hF'_meas : AEStronglyMeasurable F' (μ.restrict (Ι a b)))
    (h_lipsch : ∀ᵐ t ∂μ, t ∈ Ι a b →
      LipschitzOnWith (Real.nnabs <| bound t) (fun x => F x t) (ball x₀ ε))
    (bound_integrable : IntervalIntegrable (bound : ℝ → ℝ) μ a b)
    (h_diff : ∀ᵐ t ∂μ, t ∈ Ι a b → HasDerivAt (fun x => F x t) (F' t) x₀) :
    IntervalIntegrable F' μ a b ∧
      HasDerivAt (fun x => ∫ t in a..b, F x t ∂μ) (∫ t in a..b, F' t ∂μ) x₀ := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    a b ε : Real
    bound : Real → Real
    F : 𝕜 → Real → E
    F' : Real → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lipsch : Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → Lipsc …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → HasDeri …
    ⊢ And (IntervalIntegrable F' μ a b) (HasDerivAt (fun x => intervalIntegral (fu …
  -/
  rw [← ae_restrict_iff' measurableSet_uIoc] at h_lipsch h_diff
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    a b ε : Real
    bound : Real → Real
    F : 𝕜 → Real → E
    F' : Real → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lipsch : Filter.Eventually (fun x => LipschitzOnWith (Real.nnabs (bound x))  …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun x => HasDerivAt (fun x_1 => F x_1 x) (F' x) x₀ …
    ⊢ And (IntervalIntegrable F' μ a b) (HasDerivAt (fun x => intervalIntegral (fu …
  -/
  simp only [intervalIntegrable_iff] at hF_int bound_integrable ⊢
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    a b ε : Real
    bound : Real → Real
    F : 𝕜 → Real → E
    F' : Real → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lipsch : Filter.Eventually (fun x => LipschitzOnWith (Real.nnabs (bound x))  …
    h_diff : Filter.Eventually (fun x => HasDerivAt (fun x_1 => F x_1 x) (F' x) x₀ …
    hF_int : MeasureTheory.IntegrableOn (F x₀) (Set.uIoc a b) μ
    bound_integrable : MeasureTheory.IntegrableOn bound (Set.uIoc a b) μ
    ⊢ And (MeasureTheory.IntegrableOn F' (Set.uIoc a b) μ) (HasDerivAt (fun x => i …
  -/
  simp only [intervalIntegral_eq_integral_uIoc]
  have := hasDerivAt_integral_of_dominated_loc_of_lip ε_pos hF_meas hF_int hF'_meas h_lipsch
    bound_integrable h_diff
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    a b ε : Real
    bound : Real → Real
    F : 𝕜 → Real → E
    F' : Real → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lipsch : Filter.Eventually (fun x => LipschitzOnWith (Real.nnabs (bound x))  …
    h_diff : Filter.Eventually (fun x => HasDerivAt (fun x_1 => F x_1 x) (F' x) x₀ …
    hF_int : MeasureTheory.IntegrableOn (F x₀) (Set.uIoc a b) μ
    bound_integrable : MeasureTheory.IntegrableOn bound (Set.uIoc a b) μ
    this : And (MeasureTheory.Integrable F' (μ.restrict (Set.uIoc a b))) (HasDeriv …
    ⊢ And (MeasureTheory.IntegrableOn F' (Set.uIoc a b) μ) (HasDerivAt (fun x => H …
  -/
  exact ⟨this.1, this.2.const_smul _⟩
  /-
    🎉 no goals
  -/


/-- Derivative under integral of `x ↦ ∫ F x a` at a given point `x₀ : 𝕜`, `𝕜 = ℝ` or `𝕜 = ℂ`,
assuming `F x₀` is integrable, `x ↦ F x a` is differentiable on an interval around `x₀` for ae `a`
(with interval radius independent of `a`) with derivative uniformly bounded by an integrable
function, and `F x` is ae-measurable for `x` in a possibly smaller neighborhood of `x₀`. -/
nonrec theorem hasDerivAt_integral_of_dominated_loc_of_deriv_le
    {F : 𝕜 → ℝ → E} {F' : 𝕜 → ℝ → E} {x₀ : 𝕜}
    (ε_pos : 0 < ε) (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) (μ.restrict (Ι a b)))
    (hF_int : IntervalIntegrable (F x₀) μ a b)
    (hF'_meas : AEStronglyMeasurable (F' x₀) (μ.restrict (Ι a b)))
    (h_bound : ∀ᵐ t ∂μ, t ∈ Ι a b → ∀ x ∈ ball x₀ ε, ‖F' x t‖ ≤ bound t)
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_diff : ∀ᵐ t ∂μ, t ∈ Ι a b → ∀ x ∈ ball x₀ ε, HasDerivAt (fun x => F x t) (F' x t) x) :
    IntervalIntegrable (F' x₀) μ a b ∧
      HasDerivAt (fun x => ∫ t in a..b, F x t ∂μ) (∫ t in a..b, F' x₀ t ∂μ) x₀ := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    a b ε : Real
    bound : Real → Real
    F F' : 𝕜 → Real → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → ∀ (x : …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → ∀ (x :  …
    ⊢ And (IntervalIntegrable (F' x₀) μ a b) (HasDerivAt (fun x => intervalIntegra …
  -/
  rw [← ae_restrict_iff' measurableSet_uIoc] at h_bound h_diff
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    a b ε : Real
    bound : Real → Real
    F F' : 𝕜 → Real → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : Filter.Eventually (fun x => ∀ (x_1 : 𝕜), Membership.mem (Metric.ball …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun x => ∀ (x_1 : 𝕜), Membership.mem (Metric.ball  …
    ⊢ And (IntervalIntegrable (F' x₀) μ a b) (HasDerivAt (fun x => intervalIntegra …
  -/
  simp only [intervalIntegrable_iff] at hF_int bound_integrable ⊢
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    a b ε : Real
    bound : Real → Real
    F F' : 𝕜 → Real → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : Filter.Eventually (fun x => ∀ (x_1 : 𝕜), Membership.mem (Metric.ball …
    h_diff : Filter.Eventually (fun x => ∀ (x_1 : 𝕜), Membership.mem (Metric.ball  …
    hF_int : MeasureTheory.IntegrableOn (F x₀) (Set.uIoc a b) μ
    bound_integrable : MeasureTheory.IntegrableOn bound (Set.uIoc a b) μ
    ⊢ And (MeasureTheory.IntegrableOn (F' x₀) (Set.uIoc a b) μ) (HasDerivAt (fun x …
  -/
  simp only [intervalIntegral_eq_integral_uIoc]
  have := hasDerivAt_integral_of_dominated_loc_of_deriv_le ε_pos hF_meas hF_int hF'_meas h_bound
    bound_integrable h_diff
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    μ : MeasureTheory.Measure Real
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    a b ε : Real
    bound : Real → Real
    F F' : 𝕜 → Real → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : Filter.Eventually (fun x => ∀ (x_1 : 𝕜), Membership.mem (Metric.ball …
    h_diff : Filter.Eventually (fun x => ∀ (x_1 : 𝕜), Membership.mem (Metric.ball  …
    hF_int : MeasureTheory.IntegrableOn (F x₀) (Set.uIoc a b) μ
    bound_integrable : MeasureTheory.IntegrableOn bound (Set.uIoc a b) μ
    this : And (MeasureTheory.Integrable (F' x₀) (μ.restrict (Set.uIoc a b))) (Has …
    ⊢ And (MeasureTheory.IntegrableOn (F' x₀) (Set.uIoc a b) μ) (HasDerivAt (fun x …
  -/
  exact ⟨this.1, this.2.const_smul _⟩
  /-
    🎉 no goals
  -/


