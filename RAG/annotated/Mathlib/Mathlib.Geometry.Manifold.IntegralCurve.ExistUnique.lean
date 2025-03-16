/-- Existence of local integral curves for a $C^1$ vector field at interior points of a smooth
manifold. -/
theorem exists_isIntegralCurveAt_of_contMDiffAt [CompleteSpace E]
    (hv : ContMDiffAt I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)) x₀)
    (hx : I.IsInteriorPoint x₀) :
    ∃ γ : ℝ → M, γ t₀ = x₀ ∧ IsIntegralCurveAt γ v t₀ := by
  -- express the differentiability of the vector field `v` in the local chart
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hv : ContMDiffAt I I.tangent 1 (fun x => { proj := x, snd := v x }) x₀
    hx : I.IsInteriorPoint x₀
    ⊢ Exists fun γ => And (Eq (γ t₀) x₀) (IsIntegralCurveAt γ v t₀)
  -/
  rw [contMDiffAt_iff] at hv
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hv : And (ContinuousAt (fun x => { proj := x, snd := v x }) x₀) (ContDiffWithi …
    hx : I.IsInteriorPoint x₀
    ⊢ Exists fun γ => And (Eq (γ t₀) x₀) (IsIntegralCurveAt γ v t₀)
  -/
  obtain ⟨_, hv⟩ := hv
  -- use Picard-Lindelöf theorem to extract a solution to the ODE in the local chart
  obtain ⟨f, hf1, hf2⟩ := exists_forall_hasDerivAt_Ioo_eq_of_contDiffAt t₀
    (hv.contDiffAt (range_mem_nhds_isInteriorPoint hx)).snd
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Exists fun ε => And (GT.gt ε 0) (∀ (t : Real), Membership.mem (Set.Ioo ( …
    ⊢ Exists fun γ => And (Eq (γ t₀) x₀) (IsIntegralCurveAt γ v t₀)
  -/
  simp_rw [← Real.ball_eq_Ioo, ← Metric.eventually_nhds_iff_ball] at hf2
  -- use continuity of `f` so that `f t` remains inside `interior (extChartAt I x₀).target`
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    ⊢ Exists fun γ => And (Eq (γ t₀) x₀) (IsIntegralCurveAt γ v t₀)
  -/
  have ⟨a, ha, hf2'⟩ := Metric.eventually_nhds_iff_ball.mp hf2
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    ⊢ Exists fun γ => And (Eq (γ t₀) x₀) (IsIntegralCurveAt γ v t₀)
  -/
  have hcont := (hf2' t₀ (Metric.mem_ball_self ha)).continuousAt
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ContinuousAt f t₀
    ⊢ Exists fun γ => And (Eq (γ t₀) x₀) (IsIntegralCurveAt γ v t₀)
  -/
  rw [continuousAt_def, hf1] at hcont
  have hnhds : f ⁻¹' (interior (extChartAt I x₀).target) ∈ 𝓝 t₀ :=
    hcont _ (isOpen_interior.mem_nhds ((I.isInteriorPoint_iff).mp hx))
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Membership.mem (nhds t₀) (Set.preimage f (interior (extChartAt I x₀).t …
    ⊢ Exists fun γ => And (Eq (γ t₀) x₀) (IsIntegralCurveAt γ v t₀)
  -/
  rw [← eventually_mem_nhds_iff] at hnhds
  -- obtain a neighbourhood `s` so that the above conditions both hold in `s`
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    ⊢ Exists fun γ => And (Eq (γ t₀) x₀) (IsIntegralCurveAt γ v t₀)
  -/
  obtain ⟨s, hs, haux⟩ := (hf2.and hnhds).exists_mem
  -- prove that `γ := (extChartAt I x₀).symm ∘ f` is a desired integral curve
  refine ⟨(extChartAt I x₀).symm ∘ f,
    Eq.symm (by rw [Function.comp_apply, hf1, PartialEquiv.left_inv _ (mem_extChartAt_source ..)]),
    isIntegralCurveAt_iff.mpr ⟨s, hs, ?_⟩⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    ⊢ IsIntegralCurveOn (Function.comp (↑(extChartAt I x₀).symm) f) v s
  -/
  intros t ht
  -- collect useful terms in convenient forms
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp (↑(extChartAt …
  -/
  let xₜ : M := (extChartAt I x₀).symm (f t) -- `xₜ := γ t`
  have h : HasDerivAt f (x := t) <| fderivWithin ℝ (extChartAt I x₀ ∘ (extChartAt I xₜ).symm)
    (range I) (extChartAt I xₜ xₜ) (v xₜ) := (haux t ht).1
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((fderivWithin Real (Function.comp ↑(extChartAt I x₀) ↑(extCh …
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp (↑(extChartAt …
  -/
  rw [← tangentCoordChange_def] at h
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((tangentCoordChange I xₜ x₀ xₜ) (v xₜ)) t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp (↑(extChartAt …
  -/
  have hf3 := mem_preimage.mp <| mem_of_mem_nhds (haux t ht).2
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((tangentCoordChange I xₜ x₀ xₜ) (v xₜ)) t
    hf3 : Membership.mem (interior (extChartAt I x₀).target) (f t)
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp (↑(extChartAt …
  -/
  have hf3' := mem_of_mem_of_subset hf3 interior_subset
  have hft1 := mem_preimage.mp <|
    mem_of_mem_of_subset hf3' (extChartAt I x₀).target_subset_preimage_source
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((tangentCoordChange I xₜ x₀ xₜ) (v xₜ)) t
    hf3 : Membership.mem (interior (extChartAt I x₀).target) (f t)
    hf3' : Membership.mem (extChartAt I x₀).target (f t)
    hft1 : Membership.mem (extChartAt I x₀).source (↑(extChartAt I x₀).symm (f t))
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (Function.comp (↑(extChartAt …
  -/
  have hft2 := mem_extChartAt_source (I := I) xₜ
  -- express the derivative of the integral curve in the local chart
  refine ⟨(continuousAt_extChartAt_symm'' hf3').comp h.continuousAt,
    HasDerivWithinAt.hasFDerivWithinAt ?_⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((tangentCoordChange I xₜ x₀ xₜ) (v xₜ)) t
    hf3 : Membership.mem (interior (extChartAt I x₀).target) (f t)
    hf3' : Membership.mem (extChartAt I x₀).target (f t)
    hft1 : Membership.mem (extChartAt I x₀).source (↑(extChartAt I x₀).symm (f t))
    hft2 : Membership.mem (extChartAt I xₜ).source xₜ
    ⊢ HasDerivWithinAt (writtenInExtChartAt (modelWithCornersSelf Real Real) I t ( …
  -/
  simp only [mfld_simps, hasDerivWithinAt_univ]
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((tangentCoordChange I xₜ x₀ xₜ) (v xₜ)) t
    hf3 : Membership.mem (interior (extChartAt I x₀).target) (f t)
    hf3' : Membership.mem (extChartAt I x₀).target (f t)
    hft1 : Membership.mem (extChartAt I x₀).source (↑(extChartAt I x₀).symm (f t))
    hft2 : Membership.mem (extChartAt I xₜ).source xₜ
    ⊢ HasDerivAt (Function.comp (Function.comp ↑I ↑(chartAt H (↑(chartAt H x₀).sym …
  -/
  show HasDerivAt ((extChartAt I xₜ ∘ (extChartAt I x₀).symm) ∘ f) (v xₜ) t
  -- express `v (γ t)` as `D⁻¹ D (v (γ t))`, where `D` is a change of coordinates, so we can use
  -- `HasFDerivAt.comp_hasDerivAt` on `h`
  rw [← tangentCoordChange_self (I := I) (x := xₜ) (z := xₜ) (v := v xₜ) hft2,
    ← tangentCoordChange_comp (x := x₀) ⟨⟨hft2, hft1⟩, hft2⟩]
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((tangentCoordChange I xₜ x₀ xₜ) (v xₜ)) t
    hf3 : Membership.mem (interior (extChartAt I x₀).target) (f t)
    hf3' : Membership.mem (extChartAt I x₀).target (f t)
    hft1 : Membership.mem (extChartAt I x₀).source (↑(extChartAt I x₀).symm (f t))
    hft2 : Membership.mem (extChartAt I xₜ).source xₜ
    ⊢ HasDerivAt (Function.comp (Function.comp ↑(extChartAt I xₜ) ↑(extChartAt I x …
  -/
  apply HasFDerivAt.comp_hasDerivAt _ _ h
  apply HasFDerivWithinAt.hasFDerivAt (s := range I) _ <|
    mem_nhds_iff.mpr ⟨interior (extChartAt I x₀).target,
      subset_trans interior_subset (extChartAt_target_subset_range ..),
      isOpen_interior, hf3⟩
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((tangentCoordChange I xₜ x₀ xₜ) (v xₜ)) t
    hf3 : Membership.mem (interior (extChartAt I x₀).target) (f t)
    hf3' : Membership.mem (extChartAt I x₀).target (f t)
    hft1 : Membership.mem (extChartAt I x₀).source (↑(extChartAt I x₀).symm (f t))
    hft2 : Membership.mem (extChartAt I xₜ).source xₜ
    ⊢ HasFDerivWithinAt (Function.comp ↑(extChartAt I xₜ) ↑(extChartAt I x₀).symm) …
  -/
  rw [← (extChartAt I x₀).right_inv hf3']
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    x₀ : M
    inst✝ : CompleteSpace E
    hx : I.IsInteriorPoint x₀
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) x₀
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    f : Real → E
    hf1 : Eq (f t₀) (↑(extChartAt I x₀) x₀)
    hf2 : Filter.Eventually (fun y => HasDerivAt f (Function.comp (↑(extChartAt I. …
    a : Real
    ha : GT.gt a 0
    hf2' : ∀ (y : Real), Membership.mem (Metric.ball t₀ a) y → HasDerivAt f (Funct …
    hcont : ∀ (A : Set E), Membership.mem (nhds (↑(extChartAt I x₀) x₀)) A → Membe …
    hnhds : Filter.Eventually (fun x' => Membership.mem (nhds x') (Set.preimage f  …
    s : Set Real
    hs : Membership.mem (nhds t₀) s
    haux : ∀ (y : Real), Membership.mem s y → And (HasDerivAt f (Function.comp (↑( …
    t : Real
    ht : Membership.mem s t
    xₜ : M := ↑(extChartAt I x₀).symm (f t)
    h : HasDerivAt f ((tangentCoordChange I xₜ x₀ xₜ) (v xₜ)) t
    hf3 : Membership.mem (interior (extChartAt I x₀).target) (f t)
    hf3' : Membership.mem (extChartAt I x₀).target (f t)
    hft1 : Membership.mem (extChartAt I x₀).source (↑(extChartAt I x₀).symm (f t))
    hft2 : Membership.mem (extChartAt I xₜ).source xₜ
    ⊢ HasFDerivWithinAt (Function.comp ↑(extChartAt I xₜ) ↑(extChartAt I x₀).symm) …
  -/
  exact hasFDerivWithinAt_tangentCoordChange ⟨hft1, hft2⟩
  /-
    🎉 no goals
  -/


/-- Existence of local integral curves for a $C^1$ vector field on a smooth manifold without
boundary. -/
lemma exists_isIntegralCurveAt_of_contMDiffAt_boundaryless
    [CompleteSpace E] [BoundarylessManifold I M]
    (hv : ContMDiffAt I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)) x₀) :
    ∃ γ : ℝ → M, γ t₀ = x₀ ∧ IsIntegralCurveAt γ v t₀ :=
  exists_isIntegralCurveAt_of_contMDiffAt t₀ hv BoundarylessManifold.isInteriorPoint


/-- Local integral curves are unique.

If a $C^1$ vector field `v` admits two local integral curves `γ γ' : ℝ → M` at `t₀` with
`γ t₀ = γ' t₀`, then `γ` and `γ'` agree on some open interval containing `t₀`. -/
theorem isIntegralCurveAt_eventuallyEq_of_contMDiffAt (hγt₀ : I.IsInteriorPoint (γ t₀))
    (hv : ContMDiffAt I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)) (γ t₀))
    (hγ : IsIntegralCurveAt γ v t₀) (hγ' : IsIntegralCurveAt γ' v t₀) (h : γ t₀ = γ' t₀) :
    γ =ᶠ[𝓝 t₀] γ' := by
  -- first define `v'` as the vector field expressed in the local chart around `γ t₀`
  -- this is basically what the function looks like when `hv` is unfolded
  set v' : E → E := fun x ↦
    tangentCoordChange I ((extChartAt I (γ t₀)).symm x) (γ t₀) ((extChartAt I (γ t₀)).symm x)
      (v ((extChartAt I (γ t₀)).symm x)) with hv'
  -- extract a set `s` on which `v'` is Lipschitz
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    hγt₀ : I.IsInteriorPoint (γ t₀)
    hv : ContMDiffAt I I.tangent 1 (fun x => { proj := x, snd := v x }) (γ t₀)
    hγ : IsIntegralCurveAt γ v t₀
    hγ' : IsIntegralCurveAt γ' v t₀
    h : Eq (γ t₀) (γ' t₀)
    v' : E → E := fun x => (tangentCoordChange I (↑(extChartAt I (γ t₀)).symm x) ( …
    hv' : Eq v' fun x => (tangentCoordChange I (↑(extChartAt I (γ t₀)).symm x) (γ  …
    ⊢ (nhds t₀).EventuallyEq γ γ'
  -/
  rw [contMDiffAt_iff] at hv
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    hγt₀ : I.IsInteriorPoint (γ t₀)
    hv : And (ContinuousAt (fun x => { proj := x, snd := v x }) (γ t₀)) (ContDiffW …
    hγ : IsIntegralCurveAt γ v t₀
    hγ' : IsIntegralCurveAt γ' v t₀
    h : Eq (γ t₀) (γ' t₀)
    v' : E → E := fun x => (tangentCoordChange I (↑(extChartAt I (γ t₀)).symm x) ( …
    hv' : Eq v' fun x => (tangentCoordChange I (↑(extChartAt I (γ t₀)).symm x) (γ  …
    ⊢ (nhds t₀).EventuallyEq γ γ'
  -/
  obtain ⟨_, hv⟩ := hv
  obtain ⟨K, s, hs, hlip⟩ : ∃ K, ∃ s ∈ 𝓝 _, LipschitzOnWith K v' s :=
    (hv.contDiffAt (range_mem_nhds_isInteriorPoint hγt₀)).snd.exists_lipschitzOnWith
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    hγt₀ : I.IsInteriorPoint (γ t₀)
    hγ : IsIntegralCurveAt γ v t₀
    hγ' : IsIntegralCurveAt γ' v t₀
    h : Eq (γ t₀) (γ' t₀)
    v' : E → E := fun x => (tangentCoordChange I (↑(extChartAt I (γ t₀)).symm x) ( …
    hv' : Eq v' fun x => (tangentCoordChange I (↑(extChartAt I (γ t₀)).symm x) (γ  …
    left✝ : ContinuousAt (fun x => { proj := x, snd := v x }) (γ t₀)
    hv : ContDiffWithinAt Real (↑1) (Function.comp (↑(extChartAt I.tangent { proj  …
    K : NNReal
    s : Set E
    hs : Membership.mem (nhds (↑(extChartAt I (γ t₀)) (γ t₀))) s
    hlip : LipschitzOnWith K v' s
    ⊢ (nhds t₀).EventuallyEq γ γ'
  -/
  have hlip (t : ℝ) : LipschitzOnWith K ((fun _ ↦ v') t) ((fun _ ↦ s) t) := hlip
  -- internal lemmas to reduce code duplication
  have hsrc {g} (hg : IsIntegralCurveAt g v t₀) :
    ∀ᶠ t in 𝓝 t₀, g ⁻¹' (extChartAt I (g t₀)).source ∈ 𝓝 t := eventually_mem_nhds_iff.mpr <|
      continuousAt_def.mp hg.continuousAt _ <| extChartAt_source_mem_nhds (g t₀)
  have hmem {g : ℝ → M} {t} (ht : g ⁻¹' (extChartAt I (g t₀)).source ∈ 𝓝 t) :
    g t ∈ (extChartAt I (g t₀)).source := mem_preimage.mp <| mem_of_mem_nhds ht
  have hdrv {g} (hg : IsIntegralCurveAt g v t₀) (h' : γ t₀ = g t₀) : ∀ᶠ t in 𝓝 t₀,
      HasDerivAt ((extChartAt I (g t₀)) ∘ g) ((fun _ ↦ v') t (((extChartAt I (g t₀)) ∘ g) t)) t ∧
      ((extChartAt I (g t₀)) ∘ g) t ∈ (fun _ ↦ s) t := by
    apply Filter.Eventually.and
    · apply (hsrc hg |>.and hg.eventually_hasDerivAt).mono
      rintro t ⟨ht1, ht2⟩
      rw [hv', h']
      apply ht2.congr_deriv
      congr <;>
      rw [Function.comp_apply, PartialEquiv.left_inv _ (hmem ht1)]
    · apply ((continuousAt_extChartAt (g t₀)).comp hg.continuousAt).preimage_mem_nhds
      rw [Function.comp_apply, ← h']
      exact hs
  have heq {g} (hg : IsIntegralCurveAt g v t₀) :
    g =ᶠ[𝓝 t₀] (extChartAt I (g t₀)).symm ∘ ↑(extChartAt I (g t₀)) ∘ g := by
    apply (hsrc hg).mono
    intros t ht
    rw [Function.comp_apply, Function.comp_apply, PartialEquiv.left_inv _ (hmem ht)]
  -- main proof
  suffices (extChartAt I (γ t₀)) ∘ γ =ᶠ[𝓝 t₀] (extChartAt I (γ' t₀)) ∘ γ' from
    (heq hγ).trans <| (this.fun_comp (extChartAt I (γ t₀)).symm).trans (h ▸ (heq hγ').symm)
  exact ODE_solution_unique_of_eventually (.of_forall hlip)
    (hdrv hγ rfl) (hdrv hγ' h) (by rw [Function.comp_apply, Function.comp_apply, h])


theorem isIntegralCurveAt_eventuallyEq_of_contMDiffAt_boundaryless [BoundarylessManifold I M]
    (hv : ContMDiffAt I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)) (γ t₀))
    (hγ : IsIntegralCurveAt γ v t₀) (hγ' : IsIntegralCurveAt γ' v t₀) (h : γ t₀ = γ' t₀) :
    γ =ᶠ[𝓝 t₀] γ' :=
  isIntegralCurveAt_eventuallyEq_of_contMDiffAt BoundarylessManifold.isInteriorPoint hv hγ hγ' h


/-- Integral curves are unique on open intervals.

If a $C^1$ vector field `v` admits two integral curves `γ γ' : ℝ → M` on some open interval
`Ioo a b`, and `γ t₀ = γ' t₀` for some `t ∈ Ioo a b`, then `γ` and `γ'` agree on `Ioo a b`. -/
theorem isIntegralCurveOn_Ioo_eqOn_of_contMDiff (ht₀ : t₀ ∈ Ioo a b)
    (hγt : ∀ t ∈ Ioo a b, I.IsInteriorPoint (γ t))
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)))
    (hγ : IsIntegralCurveOn γ v (Ioo a b)) (hγ' : IsIntegralCurveOn γ' v (Ioo a b))
    (h : γ t₀ = γ' t₀) : EqOn γ γ' (Ioo a b) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : T2Space M
    a b : Real
    ht₀ : Membership.mem (Set.Ioo a b) t₀
    hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
    hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
    h : Eq (γ t₀) (γ' t₀)
    ⊢ Set.EqOn γ γ' (Set.Ioo a b)
  -/
  set s := {t | γ t = γ' t} ∩ Ioo a b with hs
  -- since `Ioo a b` is connected, we get `s = Ioo a b` by showing that `s` is clopen in `Ioo a b`
  -- in the subtype topology (`s` is also non-empty by assumption)
  -- here we use a slightly weaker alternative theorem
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : T2Space M
    a b : Real
    ht₀ : Membership.mem (Set.Ioo a b) t₀
    hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
    hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
    h : Eq (γ t₀) (γ' t₀)
    s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
    hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
    ⊢ Set.EqOn γ γ' (Set.Ioo a b)
  -/
  suffices hsub : Ioo a b ⊆ s from fun t ht ↦ mem_setOf.mp ((subset_def ▸ hsub) t ht).1
  apply isPreconnected_Ioo.subset_of_closure_inter_subset (s := Ioo a b) (u := s) _
    ⟨t₀, ⟨ht₀, ⟨h, ht₀⟩⟩⟩
  · -- is this really the most convenient way to pass to subtype topology?
    -- TODO: shorten this when better API around subtype topology exists
    rw [hs, inter_comm, ← Subtype.image_preimage_val, inter_comm, ← Subtype.image_preimage_val,
      image_subset_image_iff Subtype.val_injective, preimage_setOf_eq]
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      ⊢ HasSubset.Subset (Set.preimage Subtype.val (closure (Set.image Subtype.val ( …
    -/
    intros t ht
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      t : Subtype fun x => Membership.mem (Set.Ioo a b) x
      ht : Membership.mem (Set.preimage Subtype.val (closure (Set.image Subtype.val  …
      ⊢ Membership.mem (setOf fun a_1 => Eq (γ ↑a_1) (γ' ↑a_1)) t
    -/
    rw [mem_preimage, ← closure_subtype] at ht
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      t : Subtype fun x => Membership.mem (Set.Ioo a b) x
      ht : Membership.mem (closure (setOf fun a_1 => Eq (γ ↑a_1) (γ' ↑a_1))) t
      ⊢ Membership.mem (setOf fun a_1 => Eq (γ ↑a_1) (γ' ↑a_1)) t
    -/
    revert ht t
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      ⊢ ∀ ⦃t : Subtype fun x => Membership.mem (Set.Ioo a b) x⦄, Membership.mem (clo …
    -/
    apply IsClosed.closure_subset (isClosed_eq _ _)
      /-
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        ⊢ Continuous fun y => γ ↑y
      -/
    · rw [continuous_iff_continuousAt]
      /-
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        ⊢ ∀ (x : Subtype fun x => Membership.mem (Set.Ioo a b) x), ContinuousAt (fun y …
      -/
      rintro ⟨_, ht⟩
      /-
        case mk
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        val✝ : Real
        ht : Membership.mem (Set.Ioo a b) val✝
        ⊢ ContinuousAt (fun y => γ ↑y) ⟨val✝, ht⟩
      -/
      apply ContinuousAt.comp _ continuousAt_subtype_val
      /-
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        val✝ : Real
        ht : Membership.mem (Set.Ioo a b) val✝
        ⊢ ContinuousAt γ ↑⟨val✝, ht⟩
      -/
      rw [Subtype.coe_mk]
      /-
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        val✝ : Real
        ht : Membership.mem (Set.Ioo a b) val✝
        ⊢ ContinuousAt γ val✝
      -/
      exact hγ.continuousAt ht
      /-
        🎉 no goals
      -/
      /-
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        ⊢ Continuous fun y => γ' ↑y
      -/
    · rw [continuous_iff_continuousAt]
      /-
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        ⊢ ∀ (x : Subtype fun x => Membership.mem (Set.Ioo a b) x), ContinuousAt (fun y …
      -/
      rintro ⟨_, ht⟩
      /-
        case mk
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        val✝ : Real
        ht : Membership.mem (Set.Ioo a b) val✝
        ⊢ ContinuousAt (fun y => γ' ↑y) ⟨val✝, ht⟩
      -/
      apply ContinuousAt.comp _ continuousAt_subtype_val
      /-
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        val✝ : Real
        ht : Membership.mem (Set.Ioo a b) val✝
        ⊢ ContinuousAt γ' ↑⟨val✝, ht⟩
      -/
      rw [Subtype.coe_mk]
      /-
        E : Type u_1
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace Real E
        H : Type u_2
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : ChartedSpace H M
        inst✝¹ : SmoothManifoldWithCorners I M
        γ γ' : Real → M
        v : (x : M) → TangentSpace I x
        t₀ : Real
        inst✝ : T2Space M
        a b : Real
        ht₀ : Membership.mem (Set.Ioo a b) t₀
        hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
        hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
        h : Eq (γ t₀) (γ' t₀)
        s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
        hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
        val✝ : Real
        ht : Membership.mem (Set.Ioo a b) val✝
        ⊢ ContinuousAt γ' val✝
      -/
      exact hγ'.continuousAt ht
      /-
        🎉 no goals
      -/
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      ⊢ IsOpen s
    -/
  · rw [isOpen_iff_mem_nhds]
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      ⊢ ∀ (x : Real), Membership.mem s x → Membership.mem (nhds x) s
    -/
    intro t₁ ht₁
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      t₁ : Real
      ht₁ : Membership.mem s t₁
      ⊢ Membership.mem (nhds t₁) s
    -/
    have hmem := Ioo_mem_nhds ht₁.2.1 ht₁.2.2
    have heq : γ =ᶠ[𝓝 t₁] γ' := isIntegralCurveAt_eventuallyEq_of_contMDiffAt
      (hγt _ ht₁.2) hv.contMDiffAt (hγ.isIntegralCurveAt hmem) (hγ'.isIntegralCurveAt hmem) ht₁.1
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      t₁ : Real
      ht₁ : Membership.mem s t₁
      hmem : Membership.mem (nhds t₁) (Set.Ioo a b)
      heq : (nhds t₁).EventuallyEq γ γ'
      ⊢ Membership.mem (nhds t₁) s
    -/
    apply (heq.and hmem).mono
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      H : Type u_2
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      inst✝¹ : SmoothManifoldWithCorners I M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      t₀ : Real
      inst✝ : T2Space M
      a b : Real
      ht₀ : Membership.mem (Set.Ioo a b) t₀
      hγt : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → I.IsInteriorPoint (γ t)
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a b)
      h : Eq (γ t₀) (γ' t₀)
      s : Set Real := Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b)
      hs : Eq s (Inter.inter (setOf fun t => Eq (γ t) (γ' t)) (Set.Ioo a b))
      t₁ : Real
      ht₁ : Membership.mem s t₁
      hmem : Membership.mem (nhds t₁) (Set.Ioo a b)
      heq : (nhds t₁).EventuallyEq γ γ'
      ⊢ ∀ (x : Real), And (Eq (γ x) (γ' x)) (And (LT.lt a x) (LT.lt x b)) → Inter.in …
    -/
    exact fun _ ht ↦ ht
    /-
      🎉 no goals
    -/


theorem isIntegralCurveOn_Ioo_eqOn_of_contMDiff_boundaryless [BoundarylessManifold I M]
    (ht₀ : t₀ ∈ Ioo a b)
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)))
    (hγ : IsIntegralCurveOn γ v (Ioo a b)) (hγ' : IsIntegralCurveOn γ' v (Ioo a b))
    (h : γ t₀ = γ' t₀) : EqOn γ γ' (Ioo a b) :=
  isIntegralCurveOn_Ioo_eqOn_of_contMDiff
    ht₀ (fun _ _ ↦ BoundarylessManifold.isInteriorPoint) hv hγ hγ' h


/-- Global integral curves are unique.

If a continuously differentiable vector field `v` admits two global integral curves
`γ γ' : ℝ → M`, and `γ t₀ = γ' t₀` for some `t₀`, then `γ` and `γ'` are equal. -/
theorem isIntegralCurve_eq_of_contMDiff (hγt : ∀ t, I.IsInteriorPoint (γ t))
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)))
    (hγ : IsIntegralCurve γ v) (hγ' : IsIntegralCurve γ' v) (h : γ t₀ = γ' t₀) : γ = γ' := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type u_2
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : T2Space M
    hγt : ∀ (t : Real), I.IsInteriorPoint (γ t)
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    hγ : IsIntegralCurve γ v
    hγ' : IsIntegralCurve γ' v
    h : Eq (γ t₀) (γ' t₀)
    ⊢ Eq γ γ'
  -/
  ext t
  obtain ⟨T, ht₀, ht⟩ : ∃ T, t ∈ Ioo (-T) T ∧ t₀ ∈ Ioo (-T) T := by
    obtain ⟨T, hT₁, hT₂⟩ := exists_abs_lt t
    obtain ⟨hT₂, hT₃⟩ := abs_lt.mp hT₂
    obtain ⟨S, hS₁, hS₂⟩ := exists_abs_lt t₀
    obtain ⟨hS₂, hS₃⟩ := abs_lt.mp hS₂
    exact ⟨T + S, by constructor <;> constructor <;> linarith⟩
  exact isIntegralCurveOn_Ioo_eqOn_of_contMDiff ht (fun t _ ↦ hγt t) hv
    ((hγ.isIntegralCurveOn _).mono (subset_univ _))
    ((hγ'.isIntegralCurveOn _).mono (subset_univ _)) h ht₀


theorem isIntegralCurve_Ioo_eq_of_contMDiff_boundaryless [BoundarylessManifold I M]
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)))
    (hγ : IsIntegralCurve γ v) (hγ' : IsIntegralCurve γ' v) (h : γ t₀ = γ' t₀) : γ = γ' :=
  isIntegralCurve_eq_of_contMDiff (fun _ ↦ BoundarylessManifold.isInteriorPoint) hv hγ hγ' h


/-- For a global integral curve `γ`, if it crosses itself at `a b : ℝ`, then it is periodic with
period `a - b`. -/
lemma IsIntegralCurve.periodic_of_eq [BoundarylessManifold I M]
    (hγ : IsIntegralCurve γ v)
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)))
    (heq : γ a = γ b) : Periodic γ (a - b) := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    a b : Real
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    heq : Eq (γ a) (γ b)
    ⊢ Function.Periodic γ (HSub.hSub a b)
  -/
  intro t
  apply congrFun <|
    isIntegralCurve_Ioo_eq_of_contMDiff_boundaryless (t₀ := b) hv (hγ.comp_add _) hγ _
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    a b : Real
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    heq : Eq (γ a) (γ b)
    t : Real
    ⊢ Eq (Function.comp γ (fun x => HAdd.hAdd x (HSub.hSub a b)) b) (γ b)
  -/
  rw [comp_apply, add_sub_cancel, heq]
  /-
    🎉 no goals
  -/


/-- A global integral curve is injective xor periodic with positive period. -/
lemma IsIntegralCurve.periodic_xor_injective [BoundarylessManifold I M]
    (hγ : IsIntegralCurve γ v)
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M))) :
    Xor' (∃ T > 0, Periodic γ T) (Injective γ) := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ⊢ Xor' (Exists fun T => And (GT.gt T 0) (Function.Periodic γ T)) (Function.Inj …
  -/
  rw [xor_iff_iff_not]
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ⊢ Iff (Exists fun T => And (GT.gt T 0) (Function.Periodic γ T)) (Not (Function …
  -/
  refine ⟨fun ⟨T, hT, hf⟩ ↦ hf.not_injective (ne_of_gt hT), ?_⟩
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ⊢ Not (Function.Injective γ) → Exists fun T => And (GT.gt T 0) (Function.Perio …
  -/
  intro h
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    h : Not (Function.Injective γ)
    ⊢ Exists fun T => And (GT.gt T 0) (Function.Periodic γ T)
  -/
  rw [Injective] at h
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    h : Not (∀ ⦃a₁ a₂ : Real⦄, Eq (γ a₁) (γ a₂) → Eq a₁ a₂)
    ⊢ Exists fun T => And (GT.gt T 0) (Function.Periodic γ T)
  -/
  push_neg at h
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    h : Exists fun ⦃a₁⦄ => Exists fun ⦃a₂⦄ => And (Eq (γ a₁) (γ a₂)) (Ne a₁ a₂)
    ⊢ Exists fun T => And (GT.gt T 0) (Function.Periodic γ T)
  -/
  obtain ⟨a, b, heq, hne⟩ := h
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    hγ : IsIntegralCurve γ v
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    a b : Real
    heq : Eq (γ a) (γ b)
    hne : Ne a b
    ⊢ Exists fun T => And (GT.gt T 0) (Function.Periodic γ T)
  -/
  refine ⟨|a - b|, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      inst✝¹ : T2Space M
      inst✝ : BoundarylessManifold I M
      hγ : IsIntegralCurve γ v
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b : Real
      heq : Eq (γ a) (γ b)
      hne : Ne a b
      ⊢ GT.gt (abs (HSub.hSub a b)) 0
    -/
  · rw [gt_iff_lt, abs_pos, sub_ne_zero]
    /-
      case intro.intro.intro.refine_1
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      inst✝¹ : T2Space M
      inst✝ : BoundarylessManifold I M
      hγ : IsIntegralCurve γ v
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b : Real
      heq : Eq (γ a) (γ b)
      hne : Ne a b
      ⊢ Ne a b
    -/
    exact hne
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      inst✝¹ : T2Space M
      inst✝ : BoundarylessManifold I M
      hγ : IsIntegralCurve γ v
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b : Real
      heq : Eq (γ a) (γ b)
      hne : Ne a b
      ⊢ Function.Periodic γ (abs (HSub.hSub a b))
    -/
  · by_cases hab : a - b < 0
      /-
        case pos
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        H : Type u_2
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        γ : Real → M
        v : (x : M) → TangentSpace I x
        inst✝¹ : T2Space M
        inst✝ : BoundarylessManifold I M
        hγ : IsIntegralCurve γ v
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        a b : Real
        heq : Eq (γ a) (γ b)
        hne : Ne a b
        hab : LT.lt (HSub.hSub a b) 0
        ⊢ Function.Periodic γ (abs (HSub.hSub a b))
      -/
    · rw [abs_of_neg hab, neg_sub]
      /-
        case pos
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        H : Type u_2
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        γ : Real → M
        v : (x : M) → TangentSpace I x
        inst✝¹ : T2Space M
        inst✝ : BoundarylessManifold I M
        hγ : IsIntegralCurve γ v
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        a b : Real
        heq : Eq (γ a) (γ b)
        hne : Ne a b
        hab : LT.lt (HSub.hSub a b) 0
        ⊢ Function.Periodic γ (HSub.hSub b a)
      -/
      exact hγ.periodic_of_eq hv heq.symm
      /-
        🎉 no goals
      -/
      /-
        case neg
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        H : Type u_2
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        γ : Real → M
        v : (x : M) → TangentSpace I x
        inst✝¹ : T2Space M
        inst✝ : BoundarylessManifold I M
        hγ : IsIntegralCurve γ v
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        a b : Real
        heq : Eq (γ a) (γ b)
        hne : Ne a b
        hab : Not (LT.lt (HSub.hSub a b) 0)
        ⊢ Function.Periodic γ (abs (HSub.hSub a b))
      -/
    · rw [not_lt] at hab
      /-
        case neg
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        H : Type u_2
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        γ : Real → M
        v : (x : M) → TangentSpace I x
        inst✝¹ : T2Space M
        inst✝ : BoundarylessManifold I M
        hγ : IsIntegralCurve γ v
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        a b : Real
        heq : Eq (γ a) (γ b)
        hne : Ne a b
        hab : LE.le 0 (HSub.hSub a b)
        ⊢ Function.Periodic γ (abs (HSub.hSub a b))
      -/
      rw [abs_of_nonneg hab]
      /-
        case neg
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        H : Type u_2
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_3
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        γ : Real → M
        v : (x : M) → TangentSpace I x
        inst✝¹ : T2Space M
        inst✝ : BoundarylessManifold I M
        hγ : IsIntegralCurve γ v
        hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
        a b : Real
        heq : Eq (γ a) (γ b)
        hne : Ne a b
        hab : LE.le 0 (HSub.hSub a b)
        ⊢ Function.Periodic γ (HSub.hSub a b)
      -/
      exact hγ.periodic_of_eq hv heq
      /-
        🎉 no goals
      -/

