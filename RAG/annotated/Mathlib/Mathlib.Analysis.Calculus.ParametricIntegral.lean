/-- Differentiation under integral of `x ↦ ∫ F x a` at a given point `x₀`, assuming `F x₀` is
integrable, `‖F x a - F x₀ a‖ ≤ bound a * ‖x - x₀‖` for `x` in a ball around `x₀` for ae `a` with
integrable Lipschitz bound `bound` (with a ball radius independent of `a`), and `F x` is
ae-measurable for `x` in the same ball. See `hasFDerivAt_integral_of_dominated_loc_of_lip` for a
slightly less general but usually more useful version. -/
theorem hasFDerivAt_integral_of_dominated_loc_of_lip' {F' : α → H →L[𝕜] E} (ε_pos : 0 < ε)
    (hF_meas : ∀ x ∈ ball x₀ ε, AEStronglyMeasurable (F x) μ) (hF_int : Integrable (F x₀) μ)
    (hF'_meas : AEStronglyMeasurable F' μ)
    (h_lipsch : ∀ᵐ a ∂μ, ∀ x ∈ ball x₀ ε, ‖F x a - F x₀ a‖ ≤ bound a * ‖x - x₀‖)
    (bound_integrable : Integrable (bound : α → ℝ) μ)
    (h_diff : ∀ᵐ a ∂μ, HasFDerivAt (F · a) (F' a) x₀) :
    Integrable F' μ ∧ HasFDerivAt (fun x ↦ ∫ a, F x a ∂μ) (∫ a, F' a ∂μ) x₀ := by
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
  have x₀_in : x₀ ∈ ball x₀ ε := mem_ball_self ε_pos
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
  have nneg : ∀ x, 0 ≤ ‖x - x₀‖⁻¹ := fun x ↦ inv_nonneg.mpr (norm_nonneg _)
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
    nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
  set b : α → ℝ := fun a ↦ |bound a|
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
    nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
    b : α → Real := fun a => abs (bound a)
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
  have b_int : Integrable b μ := bound_integrable.norm
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
    nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
    b : α → Real := fun a => abs (bound a)
    b_int : MeasureTheory.Integrable b μ
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
  have b_nonneg : ∀ a, 0 ≤ b a := fun a ↦ abs_nonneg _
  replace h_lipsch : ∀ᵐ a ∂μ, ∀ x ∈ ball x₀ ε, ‖F x a - F x₀ a‖ ≤ b a * ‖x - x₀‖ :=
    h_lipsch.mono fun a ha x hx ↦
      (ha x hx).trans <| mul_le_mul_of_nonneg_right (le_abs_self _) (norm_nonneg _)
  have hF_int' : ∀ x ∈ ball x₀ ε, Integrable (F x) μ := fun x x_in ↦ by
    have : ∀ᵐ a ∂μ, ‖F x₀ a - F x a‖ ≤ ε * b a := by
      simp only [norm_sub_rev (F x₀ _)]
      refine h_lipsch.mono fun a ha ↦ (ha x x_in).trans ?_
      rw [mul_comm ε]
      rw [mem_ball, dist_eq_norm] at x_in
      exact mul_le_mul_of_nonneg_left x_in.le (b_nonneg _)
    exact integrable_of_norm_sub_le (hF_meas x x_in) hF_int
      (bound_integrable.norm.const_mul ε) this
  have hF'_int : Integrable F' μ :=
    have : ∀ᵐ a ∂μ, ‖F' a‖ ≤ b a := by
      apply (h_diff.and h_lipsch).mono
      rintro a ⟨ha_diff, ha_lip⟩
      exact ha_diff.le_of_lip' (b_nonneg a) (mem_of_superset (ball_mem_nhds _ ε_pos) <| ha_lip)
    b_int.mono' hF'_meas this
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
    nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
    b : α → Real := fun a => abs (bound a)
    b_int : MeasureTheory.Integrable b μ
    b_nonneg : ∀ (a : α), LE.le 0 (b a)
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
    hF'_int : MeasureTheory.Integrable F' μ
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
  refine ⟨hF'_int, ?_⟩
  /- Discard the trivial case where `E` is not complete, as all integrals vanish. -/
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
    nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
    b : α → Real := fun a => abs (bound a)
    b_int : MeasureTheory.Integrable b μ
    b_nonneg : ∀ (a : α), LE.le 0 (b a)
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
    hF'_int : MeasureTheory.Integrable F' μ
    ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheor …
  -/
  by_cases hE : CompleteSpace E; swap
    /-
      case neg
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : Not (CompleteSpace E)
      ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheor …
    -/
  · rcases subsingleton_or_nontrivial H with hH|hH
      /-
        case neg.inl
        α : Type u_1
        inst✝⁶ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        𝕜 : Type u_2
        inst✝⁵ : RCLike 𝕜
        E : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_4
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        F : H → α → E
        x₀ : H
        bound : α → Real
        ε : Real
        F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
        ε_pos : LT.lt 0 ε
        hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
        hF_int : MeasureTheory.Integrable (F x₀) μ
        hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
        bound_integrable : MeasureTheory.Integrable bound μ
        h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
        x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
        nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
        b : α → Real := fun a => abs (bound a)
        b_int : MeasureTheory.Integrable b μ
        b_nonneg : ∀ (a : α), LE.le 0 (b a)
        h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
        hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
        hF'_int : MeasureTheory.Integrable F' μ
        hE : Not (CompleteSpace E)
        hH : Subsingleton H
        ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheor …
      -/
    · have : Subsingleton (H →L[𝕜] E) := inferInstance
      /-
        case neg.inl
        α : Type u_1
        inst✝⁶ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        𝕜 : Type u_2
        inst✝⁵ : RCLike 𝕜
        E : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_4
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        F : H → α → E
        x₀ : H
        bound : α → Real
        ε : Real
        F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
        ε_pos : LT.lt 0 ε
        hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
        hF_int : MeasureTheory.Integrable (F x₀) μ
        hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
        bound_integrable : MeasureTheory.Integrable bound μ
        h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
        x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
        nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
        b : α → Real := fun a => abs (bound a)
        b_int : MeasureTheory.Integrable b μ
        b_nonneg : ∀ (a : α), LE.le 0 (b a)
        h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
        hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
        hF'_int : MeasureTheory.Integrable F' μ
        hE : Not (CompleteSpace E)
        hH : Subsingleton H
        this : Subsingleton (ContinuousLinearMap (RingHom.id 𝕜) H E)
        ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheor …
      -/
      convert hasFDerivAt_of_subsingleton _ x₀
      /-
        🎉 no goals
      -/
    · have : ¬(CompleteSpace (H →L[𝕜] E)) := by
        simpa [SeparatingDual.completeSpace_continuousLinearMap_iff] using hE
      /-
        case neg.inr
        α : Type u_1
        inst✝⁶ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        𝕜 : Type u_2
        inst✝⁵ : RCLike 𝕜
        E : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_4
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        F : H → α → E
        x₀ : H
        bound : α → Real
        ε : Real
        F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
        ε_pos : LT.lt 0 ε
        hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
        hF_int : MeasureTheory.Integrable (F x₀) μ
        hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
        bound_integrable : MeasureTheory.Integrable bound μ
        h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
        x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
        nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
        b : α → Real := fun a => abs (bound a)
        b_int : MeasureTheory.Integrable b μ
        b_nonneg : ∀ (a : α), LE.le 0 (b a)
        h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
        hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
        hF'_int : MeasureTheory.Integrable F' μ
        hE : Not (CompleteSpace E)
        hH : Nontrivial H
        this : Not (CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) H E))
        ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheor …
      -/
      simp only [integral, hE, ↓reduceDIte, this]
      /-
        case neg.inr
        α : Type u_1
        inst✝⁶ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        𝕜 : Type u_2
        inst✝⁵ : RCLike 𝕜
        E : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : NormedSpace 𝕜 E
        H : Type u_4
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        F : H → α → E
        x₀ : H
        bound : α → Real
        ε : Real
        F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
        ε_pos : LT.lt 0 ε
        hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
        hF_int : MeasureTheory.Integrable (F x₀) μ
        hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
        bound_integrable : MeasureTheory.Integrable bound μ
        h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
        x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
        nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
        b : α → Real := fun a => abs (bound a)
        b_int : MeasureTheory.Integrable b μ
        b_nonneg : ∀ (a : α), LE.le 0 (b a)
        h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
        hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
        hF'_int : MeasureTheory.Integrable F' μ
        hE : Not (CompleteSpace E)
        hH : Nontrivial H
        this : Not (CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) H E))
        ⊢ HasFDerivAt (fun x => 0) 0 x₀
      -/
      exact hasFDerivAt_const 0 x₀
      /-
        🎉 no goals
      -/
  /-
    case pos
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
    nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
    b : α → Real := fun a => abs (bound a)
    b_int : MeasureTheory.Integrable b μ
    b_nonneg : ∀ (a : α), LE.le 0 (b a)
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
    hF'_int : MeasureTheory.Integrable F' μ
    hE : CompleteSpace E
    ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheor …
  -/
  have h_ball : ball x₀ ε ∈ 𝓝 x₀ := ball_mem_nhds x₀ ε_pos
  have : ∀ᶠ x in 𝓝 x₀, ‖x - x₀‖⁻¹ * ‖((∫ a, F x a ∂μ) - ∫ a, F x₀ a ∂μ) - (∫ a, F' a ∂μ) (x - x₀)‖ =
      ‖∫ a, ‖x - x₀‖⁻¹ • (F x a - F x₀ a - F' a (x - x₀)) ∂μ‖ := by
    apply mem_of_superset (ball_mem_nhds _ ε_pos)
    intro x x_in; simp only
    rw [Set.mem_setOf_eq, ← norm_smul_of_nonneg (nneg _), integral_smul, integral_sub, integral_sub,
      ← ContinuousLinearMap.integral_apply hF'_int]
    exacts [hF_int' x x_in, hF_int, (hF_int' x x_in).sub hF_int,
      hF'_int.apply_continuousLinearMap _]
  rw [hasFDerivAt_iff_tendsto, tendsto_congr' this, ← tendsto_zero_iff_norm_tendsto_zero, ←
    show (∫ a : α, ‖x₀ - x₀‖⁻¹ • (F x₀ a - F x₀ a - (F' a) (x₀ - x₀)) ∂μ) = 0 by simp]
  /-
    case pos
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
    nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
    b : α → Real := fun a => abs (bound a)
    b_int : MeasureTheory.Integrable b μ
    b_nonneg : ∀ (a : α), LE.le 0 (b a)
    h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
    hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
    hF'_int : MeasureTheory.Integrable F' μ
    hE : CompleteSpace E
    h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
    this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
    ⊢ Filter.Tendsto (fun x => MeasureTheory.integral μ fun a => HSMul.hSMul (Inv. …
  -/
  apply tendsto_integral_filter_of_dominated_convergence
    /-
      case pos.hF_meas
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      ⊢ Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fun a => HSM …
    -/
  · filter_upwards [h_ball] with _ x_in
    /-
      case h
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      a✝ : H
      x_in : Membership.mem (Metric.ball x₀ ε) a✝
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HSMul.hSMul (Inv.inv (Norm.norm …
    -/
    apply AEStronglyMeasurable.const_smul
    /-
      case h.hf
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      a✝ : H
      x_in : Membership.mem (Metric.ball x₀ ε) a✝
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HSub.hSub (HSub.hSub (F a✝ a) ( …
    -/
    exact ((hF_meas _ x_in).sub (hF_meas _ x₀_in)).sub (hF'_meas.apply_continuousLinearMap _)
    /-
      🎉 no goals
    -/
    /-
      case pos.h_bound
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm.norm (HS …
    -/
  · refine mem_of_superset h_ball fun x hx ↦ ?_
    /-
      case pos.h_bound
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      x : H
      hx : Membership.mem (Metric.ball x₀ ε) x
      ⊢ Membership.mem (setOf fun x => (fun n => Filter.Eventually (fun a => LE.le ( …
    -/
    apply (h_diff.and h_lipsch).mono
    /-
      case pos.h_bound
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      x : H
      hx : Membership.mem (Metric.ball x₀ ε) x
      ⊢ ∀ (x_1 : α), And (HasFDerivAt (fun x => F x x_1) (F' x_1) x₀) (∀ (x : H), Me …
    -/
    on_goal 1 => rintro a ⟨-, ha_bound⟩
    /-
      case pos.h_bound.intro
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      x : H
      hx : Membership.mem (Metric.ball x₀ ε) x
      a : α
      ha_bound : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → LE.le (Norm.norm ( …
      ⊢ LE.le (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm (HSub.hSub x x₀))) (HSub.h …
    -/
    show ‖‖x - x₀‖⁻¹ • (F x a - F x₀ a - F' a (x - x₀))‖ ≤ b a + ‖F' a‖
    /-
      case pos.h_bound.intro
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      x : H
      hx : Membership.mem (Metric.ball x₀ ε) x
      a : α
      ha_bound : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → LE.le (Norm.norm ( …
      ⊢ LE.le (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm (HSub.hSub x x₀))) (HSub.h …
    -/
    replace ha_bound : ‖F x a - F x₀ a‖ ≤ b a * ‖x - x₀‖ := ha_bound x hx
    calc
      ‖‖x - x₀‖⁻¹ • (F x a - F x₀ a - F' a (x - x₀))‖ =
          ‖‖x - x₀‖⁻¹ • (F x a - F x₀ a) - ‖x - x₀‖⁻¹ • F' a (x - x₀)‖ := by rw [smul_sub]
      _ ≤ ‖‖x - x₀‖⁻¹ • (F x a - F x₀ a)‖ + ‖‖x - x₀‖⁻¹ • F' a (x - x₀)‖ := norm_sub_le _ _
      _ = ‖x - x₀‖⁻¹ * ‖F x a - F x₀ a‖ + ‖x - x₀‖⁻¹ * ‖F' a (x - x₀)‖ := by
        rw [norm_smul_of_nonneg, norm_smul_of_nonneg] <;> exact nneg _
      _ ≤ ‖x - x₀‖⁻¹ * (b a * ‖x - x₀‖) + ‖x - x₀‖⁻¹ * (‖F' a‖ * ‖x - x₀‖) := by
        gcongr; exact (F' a).le_opNorm _
      _ ≤ b a + ‖F' a‖ := ?_
    /-
      case pos.h_bound.intro
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      x : H
      hx : Membership.mem (Metric.ball x₀ ε) x
      a : α
      ha_bound : LE.le (Norm.norm (HSub.hSub (F x a) (F x₀ a))) (HMul.hMul (b a) (No …
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub x x₀))) (HMul.hMu …
    -/
    simp only [← div_eq_inv_mul]
    /-
      case pos.h_bound.intro
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      x : H
      hx : Membership.mem (Metric.ball x₀ ε) x
      a : α
      ha_bound : LE.le (Norm.norm (HSub.hSub (F x a) (F x₀ a))) (HMul.hMul (b a) (No …
      ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (HMul.hMul (b a) (Norm.norm (HSub.hSub x x₀))) ( …
    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
    apply_rules [add_le_add, div_le_of_le_mul₀] <;> first | rfl | positivity
                                                    /-
                                                      🎉 no goals
                                                    -/
    /-
      case pos.bound_integrable
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      ⊢ MeasureTheory.Integrable (fun a => HAdd.hAdd (b a) (Norm.norm (F' a))) μ
    -/
  · exact b_int.add hF'_int.norm
    /-
      🎉 no goals
    -/
    /-
      case pos.h_lim
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv (N …
    -/
  · apply h_diff.mono
    /-
      case pos.h_lim
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      ⊢ ∀ (x : α), HasFDerivAt (fun x_1 => F x_1 x) (F' x) x₀ → Filter.Tendsto (fun  …
    -/
    intro a ha
    /-
      case pos.h_lim
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      a : α
      ha : HasFDerivAt (fun x => F x a) (F' a) x₀
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv (Norm.norm (HSub.hSub n x₀)))  …
    -/
    suffices Tendsto (fun x ↦ ‖x - x₀‖⁻¹ • (F x a - F x₀ a - F' a (x - x₀))) (𝓝 x₀) (𝓝 0) by simpa
    /-
      case pos.h_lim
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hSu …
      a : α
      ha : HasFDerivAt (fun x => F x a) (F' a) x₀
      ⊢ Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (Norm.norm (HSub.hSub x x₀)))  …
    -/
    rw [tendsto_zero_iff_norm_tendsto_zero]
    have : (fun x ↦ ‖x - x₀‖⁻¹ * ‖F x a - F x₀ a - F' a (x - x₀)‖) = fun x ↦
        ‖‖x - x₀‖⁻¹ • (F x a - F x₀ a - F' a (x - x₀))‖ := by
      ext x
      rw [norm_smul_of_nonneg (nneg _)]
    /-
      case pos.h_lim
      α : Type u_1
      inst✝⁶ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      F : H → α → E
      x₀ : H
      bound : α → Real
      ε : Real
      F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
      ε_pos : LT.lt 0 ε
      hF_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.AEStr …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
      x₀_in : Membership.mem (Metric.ball x₀ ε) x₀
      nneg : ∀ (x : H), LE.le 0 (Inv.inv (Norm.norm (HSub.hSub x x₀)))
      b : α → Real := fun a => abs (bound a)
      b_int : MeasureTheory.Integrable b μ
      b_nonneg : ∀ (a : α), LE.le 0 (b a)
      h_lipsch : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball  …
      hF_int' : ∀ (x : H), Membership.mem (Metric.ball x₀ ε) x → MeasureTheory.Integ …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : CompleteSpace E
      h_ball : Membership.mem (nhds x₀) (Metric.ball x₀ ε)
      this✝ : Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Norm.norm (HSub.hS …
      a : α
      ha : HasFDerivAt (fun x => F x a) (F' a) x₀
      this : Eq (fun x => HMul.hMul (Inv.inv (Norm.norm (HSub.hSub x x₀))) (Norm.nor …
      ⊢ Filter.Tendsto (fun x => Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm (HSub.hS …
    -/
    rwa [hasFDerivAt_iff_tendsto, this] at ha
    /-
      🎉 no goals
    -/


/-- Differentiation under integral of `x ↦ ∫ F x a` at a given point `x₀`, assuming
`F x₀` is integrable, `x ↦ F x a` is locally Lipschitz on a ball around `x₀` for ae `a`
(with a ball radius independent of `a`) with integrable Lipschitz bound, and `F x` is ae-measurable
for `x` in a possibly smaller neighborhood of `x₀`. -/
theorem hasFDerivAt_integral_of_dominated_loc_of_lip {F' : α → H →L[𝕜] E}
    (ε_pos : 0 < ε) (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) μ)
    (hF_int : Integrable (F x₀) μ) (hF'_meas : AEStronglyMeasurable F' μ)
    (h_lip : ∀ᵐ a ∂μ, LipschitzOnWith (Real.nnabs <| bound a) (F · a) (ball x₀ ε))
    (bound_integrable : Integrable (bound : α → ℝ) μ)
    (h_diff : ∀ᵐ a ∂μ, HasFDerivAt (F · a) (F' a) x₀) :
    Integrable F' μ ∧ HasFDerivAt (fun x ↦ ∫ a, F x a ∂μ) (∫ a, F' a ∂μ) x₀ := by
  obtain ⟨δ, δ_pos, hδ⟩ : ∃ δ > 0, ∀ x ∈ ball x₀ δ, AEStronglyMeasurable (F x) μ ∧ x ∈ ball x₀ ε :=
    eventually_nhds_iff_ball.mp (hF_meas.and (ball_mem_nhds x₀ ε_pos))
  /-
    case intro.intro
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lip : Filter.Eventually (fun a => LipschitzOnWith (Real.nnabs (bound a)) (fu …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    δ : Real
    δ_pos : GT.gt δ 0
    hδ : ∀ (x : H), Membership.mem (Metric.ball x₀ δ) x → And (MeasureTheory.AEStr …
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
  choose hδ_meas hδε using hδ
  replace h_lip : ∀ᵐ a : α ∂μ, ∀ x ∈ ball x₀ δ, ‖F x a - F x₀ a‖ ≤ |bound a| * ‖x - x₀‖ :=
    h_lip.mono fun a lip x hx ↦ lip.norm_sub_le (hδε x hx) (mem_ball_self ε_pos)
  /-
    case intro.intro
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    δ : Real
    δ_pos : GT.gt δ 0
    hδ_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ δ) x → MeasureTheory.AEStr …
    hδε : ∀ (x : H), Membership.mem (Metric.ball x₀ δ) x → Membership.mem (Metric. …
    h_lip : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball x₀  …
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
  replace bound_integrable := bound_integrable.norm
  /-
    case intro.intro
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (F' a) x₀) ( …
    δ : Real
    δ_pos : GT.gt δ 0
    hδ_meas : ∀ (x : H), Membership.mem (Metric.ball x₀ δ) x → MeasureTheory.AEStr …
    hδε : ∀ (x : H), Membership.mem (Metric.ball x₀ δ) x → Membership.mem (Metric. …
    h_lip : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball x₀  …
    bound_integrable : MeasureTheory.Integrable (fun a => Norm.norm (bound a)) μ
    ⊢ And (MeasureTheory.Integrable F' μ) (HasFDerivAt (fun x => MeasureTheory.int …
  -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  apply hasFDerivAt_integral_of_dominated_loc_of_lip' δ_pos <;> assumption
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- Differentiation under integral of `x ↦ ∫ x in a..b, F x t` at a given point `x₀ ∈ (a,b)`,
assuming `F x₀` is integrable on `(a,b)`, that `x ↦ F x t` is Lipschitz on a ball around `x₀`
for almost every `t` (with a ball radius independent of `t`) with integrable Lipschitz bound,
and `F x` is a.e.-measurable for `x` in a possibly smaller neighborhood of `x₀`. -/
theorem hasFDerivAt_integral_of_dominated_loc_of_lip_interval [NormedSpace ℝ H] {μ : Measure ℝ}
    {F : H → ℝ → E} {F' : ℝ → H →L[ℝ] E} {a b : ℝ} {bound : ℝ → ℝ} (ε_pos : 0 < ε)
    (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) <| μ.restrict (Ι a b))
    (hF_int : IntervalIntegrable (F x₀) μ a b)
    (hF'_meas : AEStronglyMeasurable F' <| μ.restrict (Ι a b))
    (h_lip : ∀ᵐ t ∂μ.restrict (Ι a b),
      LipschitzOnWith (Real.nnabs <| bound t) (F · t) (ball x₀ ε))
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_diff : ∀ᵐ t ∂μ.restrict (Ι a b), HasFDerivAt (F · t) (F' t) x₀) :
    IntervalIntegrable F' μ a b ∧
      HasFDerivAt (fun x ↦ ∫ t in a..b, F x t ∂μ) (∫ t in a..b, F' t ∂μ) x₀ := by
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    x₀ : H
    ε : Real
    inst✝ : NormedSpace Real H
    μ : MeasureTheory.Measure Real
    F : H → Real → E
    F' : Real → ContinuousLinearMap (RingHom.id Real) H E
    a b : Real
    bound : Real → Real
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.uIoc a b))
    h_lip : Filter.Eventually (fun t => LipschitzOnWith (Real.nnabs (bound t)) (fu …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun t => HasFDerivAt (fun x => F x t) (F' t) x₀) ( …
    ⊢ And (IntervalIntegrable F' μ a b) (HasFDerivAt (fun x => intervalIntegral (f …
  -/
  simp_rw [AEStronglyMeasurable.aestronglyMeasurable_uIoc_iff, eventually_and] at hF_meas hF'_meas
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    x₀ : H
    ε : Real
    inst✝ : NormedSpace Real H
    μ : MeasureTheory.Measure Real
    F : H → Real → E
    F' : Real → ContinuousLinearMap (RingHom.id Real) H E
    a b : Real
    bound : Real → Real
    ε_pos : LT.lt 0 ε
    hF_int : IntervalIntegrable (F x₀) μ a b
    h_lip : Filter.Eventually (fun t => LipschitzOnWith (Real.nnabs (bound t)) (fu …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun t => HasFDerivAt (fun x => F x t) (F' t) x₀) ( …
    hF'_meas : And (MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.Ioc a b …
    hF_meas : And (Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable  …
    ⊢ And (IntervalIntegrable F' μ a b) (HasFDerivAt (fun x => intervalIntegral (f …
  -/
  rw [ae_restrict_uIoc_iff] at h_lip h_diff
  have H₁ :=
    hasFDerivAt_integral_of_dominated_loc_of_lip ε_pos hF_meas.1 hF_int.1 hF'_meas.1 h_lip.1
      bound_integrable.1 h_diff.1
  have H₂ :=
    hasFDerivAt_integral_of_dominated_loc_of_lip ε_pos hF_meas.2 hF_int.2 hF'_meas.2 h_lip.2
      bound_integrable.2 h_diff.2
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    x₀ : H
    ε : Real
    inst✝ : NormedSpace Real H
    μ : MeasureTheory.Measure Real
    F : H → Real → E
    F' : Real → ContinuousLinearMap (RingHom.id Real) H E
    a b : Real
    bound : Real → Real
    ε_pos : LT.lt 0 ε
    hF_int : IntervalIntegrable (F x₀) μ a b
    h_lip : And (Filter.Eventually (fun x => LipschitzOnWith (Real.nnabs (bound x) …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : And (Filter.Eventually (fun x => HasFDerivAt (fun x_1 => F x_1 x) (F' …
    hF'_meas : And (MeasureTheory.AEStronglyMeasurable F' (μ.restrict (Set.Ioc a b …
    hF_meas : And (Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable  …
    H₁ : And (MeasureTheory.Integrable F' (μ.restrict (Set.Ioc a b))) (HasFDerivAt …
    H₂ : And (MeasureTheory.Integrable F' (μ.restrict (Set.Ioc b a))) (HasFDerivAt …
    ⊢ And (IntervalIntegrable F' μ a b) (HasFDerivAt (fun x => intervalIntegral (f …
  -/
  exact ⟨⟨H₁.1, H₂.1⟩, H₁.2.sub H₂.2⟩
  /-
    🎉 no goals
  -/


/-- Differentiation under integral of `x ↦ ∫ F x a` at a given point `x₀`, assuming
`F x₀` is integrable, `x ↦ F x a` is differentiable on a ball around `x₀` for ae `a` with
derivative norm uniformly bounded by an integrable function (the ball radius is independent of `a`),
and `F x` is ae-measurable for `x` in a possibly smaller neighborhood of `x₀`. -/
theorem hasFDerivAt_integral_of_dominated_of_fderiv_le {F' : H → α → H →L[𝕜] E} (ε_pos : 0 < ε)
    (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) μ) (hF_int : Integrable (F x₀) μ)
    (hF'_meas : AEStronglyMeasurable (F' x₀) μ)
    (h_bound : ∀ᵐ a ∂μ, ∀ x ∈ ball x₀ ε, ‖F' x a‖ ≤ bound a)
    (bound_integrable : Integrable (bound : α → ℝ) μ)
    (h_diff : ∀ᵐ a ∂μ, ∀ x ∈ ball x₀ ε, HasFDerivAt (F · a) (F' x a) x) :
    HasFDerivAt (fun x ↦ ∫ a, F x a ∂μ) (∫ a, F' x₀ a ∂μ) x₀ := by
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : H → α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) μ
    h_bound : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball x …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball x₀ …
    ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheor …
  -/
  letI : NormedSpace ℝ H := NormedSpace.restrictScalars ℝ 𝕜 H
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    F : H → α → E
    x₀ : H
    bound : α → Real
    ε : Real
    F' : H → α → ContinuousLinearMap (RingHom.id 𝕜) H E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) μ
    h_bound : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball x …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => ∀ (x : H), Membership.mem (Metric.ball x₀ …
    this : NormedSpace Real H := NormedSpace.restrictScalars Real 𝕜 H
    ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheor …
  -/
  have x₀_in : x₀ ∈ ball x₀ ε := mem_ball_self ε_pos
  have diff_x₀ : ∀ᵐ a ∂μ, HasFDerivAt (F · a) (F' x₀ a) x₀ :=
    h_diff.mono fun a ha ↦ ha x₀ x₀_in
  have : ∀ᵐ a ∂μ, LipschitzOnWith (Real.nnabs (bound a)) (F · a) (ball x₀ ε) := by
    apply (h_diff.and h_bound).mono
    rintro a ⟨ha_deriv, ha_bound⟩
    refine (convex_ball _ _).lipschitzOnWith_of_nnnorm_hasFDerivWithin_le
      (fun x x_in ↦ (ha_deriv x x_in).hasFDerivWithinAt) fun x x_in ↦ ?_
    rw [← NNReal.coe_le_coe, coe_nnnorm, Real.coe_nnabs]
    exact (ha_bound x x_in).trans (le_abs_self _)
  exact (hasFDerivAt_integral_of_dominated_loc_of_lip ε_pos hF_meas hF_int hF'_meas this
    bound_integrable diff_x₀).2


/-- Differentiation under integral of `x ↦ ∫ x in a..b, F x a` at a given point `x₀`, assuming
`F x₀` is integrable on `(a,b)`, `x ↦ F x a` is differentiable on a ball around `x₀` for ae `a` with
derivative norm uniformly bounded by an integrable function (the ball radius is independent of `a`),
and `F x` is ae-measurable for `x` in a possibly smaller neighborhood of `x₀`. -/
theorem hasFDerivAt_integral_of_dominated_of_fderiv_le'' [NormedSpace ℝ H] {μ : Measure ℝ}
    {F : H → ℝ → E} {F' : H → ℝ → H →L[ℝ] E} {a b : ℝ} {bound : ℝ → ℝ} (ε_pos : 0 < ε)
    (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) <| μ.restrict (Ι a b))
    (hF_int : IntervalIntegrable (F x₀) μ a b)
    (hF'_meas : AEStronglyMeasurable (F' x₀) <| μ.restrict (Ι a b))
    (h_bound : ∀ᵐ t ∂μ.restrict (Ι a b), ∀ x ∈ ball x₀ ε, ‖F' x t‖ ≤ bound t)
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_diff : ∀ᵐ t ∂μ.restrict (Ι a b), ∀ x ∈ ball x₀ ε, HasFDerivAt (F · t) (F' x t) x) :
    HasFDerivAt (fun x ↦ ∫ t in a..b, F x t ∂μ) (∫ t in a..b, F' x₀ t ∂μ) x₀ := by
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    x₀ : H
    ε : Real
    inst✝ : NormedSpace Real H
    μ : MeasureTheory.Measure Real
    F : H → Real → E
    F' : H → Real → ContinuousLinearMap (RingHom.id Real) H E
    a b : Real
    bound : Real → Real
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : Filter.Eventually (fun t => ∀ (x : H), Membership.mem (Metric.ball x …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : Filter.Eventually (fun t => ∀ (x : H), Membership.mem (Metric.ball x₀ …
    ⊢ HasFDerivAt (fun x => intervalIntegral (fun t => F x t) a b μ) (intervalInte …
  -/
  rw [ae_restrict_uIoc_iff] at h_diff h_bound
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    x₀ : H
    ε : Real
    inst✝ : NormedSpace Real H
    μ : MeasureTheory.Measure Real
    F : H → Real → E
    F' : H → Real → ContinuousLinearMap (RingHom.id Real) H E
    a b : Real
    bound : Real → Real
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : IntervalIntegrable (F x₀) μ a b
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) (μ.restrict (Set.uIoc a  …
    h_bound : And (Filter.Eventually (fun x => ∀ (x_1 : H), Membership.mem (Metric …
    bound_integrable : IntervalIntegrable bound μ a b
    h_diff : And (Filter.Eventually (fun x => ∀ (x_1 : H), Membership.mem (Metric. …
    ⊢ HasFDerivAt (fun x => intervalIntegral (fun t => F x t) a b μ) (intervalInte …
  -/
  simp_rw [AEStronglyMeasurable.aestronglyMeasurable_uIoc_iff, eventually_and] at hF_meas hF'_meas
  exact
    (hasFDerivAt_integral_of_dominated_of_fderiv_le ε_pos hF_meas.1 hF_int.1 hF'_meas.1 h_bound.1
          bound_integrable.1 h_diff.1).sub
      (hasFDerivAt_integral_of_dominated_of_fderiv_le ε_pos hF_meas.2 hF_int.2 hF'_meas.2 h_bound.2
        bound_integrable.2 h_diff.2)


/-- Derivative under integral of `x ↦ ∫ F x a` at a given point `x₀ : 𝕜`, `𝕜 = ℝ` or `𝕜 = ℂ`,
assuming `F x₀` is integrable, `x ↦ F x a` is locally Lipschitz on a ball around `x₀` for ae `a`
(with ball radius independent of `a`) with integrable Lipschitz bound, and `F x` is
ae-measurable for `x` in a possibly smaller neighborhood of `x₀`. -/
theorem hasDerivAt_integral_of_dominated_loc_of_lip {F' : α → E} (ε_pos : 0 < ε)
    (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) μ) (hF_int : Integrable (F x₀) μ)
    (hF'_meas : AEStronglyMeasurable F' μ)
    (h_lipsch : ∀ᵐ a ∂μ, LipschitzOnWith (Real.nnabs <| bound a) (F · a) (ball x₀ ε))
    (bound_integrable : Integrable (bound : α → ℝ) μ)
    (h_diff : ∀ᵐ a ∂μ, HasDerivAt (F · a) (F' a) x₀) :
    Integrable F' μ ∧ HasDerivAt (fun x ↦ ∫ a, F x a ∂μ) (∫ a, F' a ∂μ) x₀ := by
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    bound : α → Real
    ε : Real
    F : 𝕜 → α → E
    x₀ : 𝕜
    F' : α → E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => LipschitzOnWith (Real.nnabs (bound a))  …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => HasDerivAt (fun x => F x a) (F' a) x₀) (M …
    ⊢ And (MeasureTheory.Integrable F' μ) (HasDerivAt (fun x => MeasureTheory.inte …
  -/
  set L : E →L[𝕜] 𝕜 →L[𝕜] E := ContinuousLinearMap.smulRightL 𝕜 𝕜 E 1
  replace h_diff : ∀ᵐ a ∂μ, HasFDerivAt (F · a) (L (F' a)) x₀ :=
    h_diff.mono fun x hx ↦ hx.hasFDerivAt
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    bound : α → Real
    ε : Real
    F : 𝕜 → α → E
    x₀ : 𝕜
    F' : α → E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => LipschitzOnWith (Real.nnabs (bound a))  …
    bound_integrable : MeasureTheory.Integrable bound μ
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) 𝕜 …
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (L (F' a)) x …
    ⊢ And (MeasureTheory.Integrable F' μ) (HasDerivAt (fun x => MeasureTheory.inte …
  -/
  have hm : AEStronglyMeasurable (L ∘ F') μ := L.continuous.comp_aestronglyMeasurable hF'_meas
  cases'
    hasFDerivAt_integral_of_dominated_loc_of_lip ε_pos hF_meas hF_int hm h_lipsch bound_integrable
      h_diff with
    hF'_int key
  replace hF'_int : Integrable F' μ := by
    rw [← integrable_norm_iff hm] at hF'_int
    simpa only [L, (· ∘ ·), integrable_norm_iff, hF'_meas, one_mul, norm_one,
      ContinuousLinearMap.comp_apply, ContinuousLinearMap.coe_restrict_scalarsL',
      ContinuousLinearMap.norm_restrictScalars, ContinuousLinearMap.norm_smulRightL_apply] using
      hF'_int
  /-
    case intro
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    bound : α → Real
    ε : Real
    F : 𝕜 → α → E
    x₀ : 𝕜
    F' : α → E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => LipschitzOnWith (Real.nnabs (bound a))  …
    bound_integrable : MeasureTheory.Integrable bound μ
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) 𝕜 …
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (L (F' a)) x …
    hm : MeasureTheory.AEStronglyMeasurable (Function.comp (⇑L) F') μ
    key : HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureT …
    hF'_int : MeasureTheory.Integrable F' μ
    ⊢ And (MeasureTheory.Integrable F' μ) (HasDerivAt (fun x => MeasureTheory.inte …
  -/
  refine ⟨hF'_int, ?_⟩
  /-
    case intro
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    bound : α → Real
    ε : Real
    F : 𝕜 → α → E
    x₀ : 𝕜
    F' : α → E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => LipschitzOnWith (Real.nnabs (bound a))  …
    bound_integrable : MeasureTheory.Integrable bound μ
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) 𝕜 …
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (L (F' a)) x …
    hm : MeasureTheory.AEStronglyMeasurable (Function.comp (⇑L) F') μ
    key : HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureT …
    hF'_int : MeasureTheory.Integrable F' μ
    ⊢ HasDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheory …
  -/
  by_cases hE : CompleteSpace E; swap
    /-
      case neg
      α : Type u_1
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      𝕜 : Type u_2
      inst✝³ : RCLike 𝕜
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace 𝕜 E
      bound : α → Real
      ε : Real
      F : 𝕜 → α → E
      x₀ : 𝕜
      F' : α → E
      ε_pos : LT.lt 0 ε
      hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
      hF_int : MeasureTheory.Integrable (F x₀) μ
      hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
      h_lipsch : Filter.Eventually (fun a => LipschitzOnWith (Real.nnabs (bound a))  …
      bound_integrable : MeasureTheory.Integrable bound μ
      L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) 𝕜 …
      h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (L (F' a)) x …
      hm : MeasureTheory.AEStronglyMeasurable (Function.comp (⇑L) F') μ
      key : HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureT …
      hF'_int : MeasureTheory.Integrable F' μ
      hE : Not (CompleteSpace E)
      ⊢ HasDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheory …
    -/
  · simpa [integral, hE] using hasDerivAt_const x₀ 0
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    bound : α → Real
    ε : Real
    F : 𝕜 → α → E
    x₀ : 𝕜
    F' : α → E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => LipschitzOnWith (Real.nnabs (bound a))  …
    bound_integrable : MeasureTheory.Integrable bound μ
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) 𝕜 …
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (L (F' a)) x …
    hm : MeasureTheory.AEStronglyMeasurable (Function.comp (⇑L) F') μ
    key : HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureT …
    hF'_int : MeasureTheory.Integrable F' μ
    hE : CompleteSpace E
    ⊢ HasDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheory …
  -/
  simp_rw [hasDerivAt_iff_hasFDerivAt] at h_diff ⊢
  /-
    case pos
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    bound : α → Real
    ε : Real
    F : 𝕜 → α → E
    x₀ : 𝕜
    F' : α → E
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    hF'_meas : MeasureTheory.AEStronglyMeasurable F' μ
    h_lipsch : Filter.Eventually (fun a => LipschitzOnWith (Real.nnabs (bound a))  …
    bound_integrable : MeasureTheory.Integrable bound μ
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) 𝕜 …
    h_diff : Filter.Eventually (fun a => HasFDerivAt (fun x => F x a) (L (F' a)) x …
    hm : MeasureTheory.AEStronglyMeasurable (Function.comp (⇑L) F') μ
    key : HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureT …
    hF'_int : MeasureTheory.Integrable F' μ
    hE : CompleteSpace E
    ⊢ HasFDerivAt (fun x => MeasureTheory.integral μ fun a => F x a) (ContinuousLi …
  -/
  simpa only [(· ∘ ·), ContinuousLinearMap.integral_comp_comm _ hF'_int] using key
  /-
    🎉 no goals
  -/


/-- Derivative under integral of `x ↦ ∫ F x a` at a given point `x₀ : ℝ`, assuming
`F x₀` is integrable, `x ↦ F x a` is differentiable on an interval around `x₀` for ae `a`
(with interval radius independent of `a`) with derivative uniformly bounded by an integrable
function, and `F x` is ae-measurable for `x` in a possibly smaller neighborhood of `x₀`. -/
theorem hasDerivAt_integral_of_dominated_loc_of_deriv_le (ε_pos : 0 < ε)
    (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) μ) (hF_int : Integrable (F x₀) μ)
    {F' : 𝕜 → α → E} (hF'_meas : AEStronglyMeasurable (F' x₀) μ)
    (h_bound : ∀ᵐ a ∂μ, ∀ x ∈ ball x₀ ε, ‖F' x a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_diff : ∀ᵐ a ∂μ, ∀ x ∈ ball x₀ ε, HasDerivAt (F · a) (F' x a) x) :
    Integrable (F' x₀) μ ∧ HasDerivAt (fun n ↦ ∫ a, F n a ∂μ) (∫ a, F' x₀ a ∂μ) x₀ := by
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace 𝕜 E
    bound : α → Real
    ε : Real
    F : 𝕜 → α → E
    x₀ : 𝕜
    ε_pos : LT.lt 0 ε
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    hF_int : MeasureTheory.Integrable (F x₀) μ
    F' : 𝕜 → α → E
    hF'_meas : MeasureTheory.AEStronglyMeasurable (F' x₀) μ
    h_bound : Filter.Eventually (fun a => ∀ (x : 𝕜), Membership.mem (Metric.ball x …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_diff : Filter.Eventually (fun a => ∀ (x : 𝕜), Membership.mem (Metric.ball x₀ …
    ⊢ And (MeasureTheory.Integrable (F' x₀) μ) (HasDerivAt (fun n => MeasureTheory …
  -/
  have x₀_in : x₀ ∈ ball x₀ ε := mem_ball_self ε_pos
  have diff_x₀ : ∀ᵐ a ∂μ, HasDerivAt (F · a) (F' x₀ a) x₀ :=
    h_diff.mono fun a ha ↦ ha x₀ x₀_in
  have : ∀ᵐ a ∂μ, LipschitzOnWith (Real.nnabs (bound a)) (fun x : 𝕜 ↦ F x a) (ball x₀ ε) := by
    apply (h_diff.and h_bound).mono
    rintro a ⟨ha_deriv, ha_bound⟩
    refine (convex_ball _ _).lipschitzOnWith_of_nnnorm_hasDerivWithin_le
      (fun x x_in ↦ (ha_deriv x x_in).hasDerivWithinAt) fun x x_in ↦ ?_
    rw [← NNReal.coe_le_coe, coe_nnnorm, Real.coe_nnabs]
    exact (ha_bound x x_in).trans (le_abs_self _)
  exact
    hasDerivAt_integral_of_dominated_loc_of_lip ε_pos hF_meas hF_int hF'_meas this bound_integrable
      diff_x₀


