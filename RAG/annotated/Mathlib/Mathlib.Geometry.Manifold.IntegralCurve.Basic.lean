/-- If `γ : ℝ → M` is $C^1$ on `s : Set ℝ` and `v` is a vector field on `M`,
`IsIntegralCurveOn γ v s` means `γ t` is tangent to `v (γ t)` for all `t ∈ s`. The value of `γ`
outside of `s` is irrelevant and considered junk. -/
def IsIntegralCurveOn (γ : ℝ → M) (v : (x : M) → TangentSpace I x) (s : Set ℝ) : Prop :=
  ∀ t ∈ s, HasMFDerivAt 𝓘(ℝ, ℝ) I γ t ((1 : ℝ →L[ℝ] ℝ).smulRight <| v (γ t))


/-- If `v` is a vector field on `M` and `t₀ : ℝ`, `IsIntegralCurveAt γ v t₀` means `γ : ℝ → M` is a
local integral curve of `v` in a neighbourhood containing `t₀`. The value of `γ` outside of this
interval is irrelevant and considered junk. -/
def IsIntegralCurveAt (γ : ℝ → M) (v : (x : M) → TangentSpace I x) (t₀ : ℝ) : Prop :=
  ∀ᶠ t in 𝓝 t₀, HasMFDerivAt 𝓘(ℝ, ℝ) I γ t ((1 : ℝ →L[ℝ] ℝ).smulRight <| v (γ t))


/-- If `v : M → TM` is a vector field on `M`, `IsIntegralCurve γ v` means `γ : ℝ → M` is a global
integral curve of `v`. That is, `γ t` is tangent to `v (γ t)` for all `t : ℝ`. -/
def IsIntegralCurve (γ : ℝ → M) (v : (x : M) → TangentSpace I x) : Prop :=
  ∀ t : ℝ, HasMFDerivAt 𝓘(ℝ, ℝ) I γ t ((1 : ℝ →L[ℝ] ℝ).smulRight (v (γ t)))


lemma IsIntegralCurve.isIntegralCurveOn (h : IsIntegralCurve γ v) (s : Set ℝ) :
    IsIntegralCurveOn γ v s := fun t _ ↦ h t


lemma isIntegralCurve_iff_isIntegralCurveOn : IsIntegralCurve γ v ↔ IsIntegralCurveOn γ v univ :=
  ⟨fun h ↦ h.isIntegralCurveOn _, fun h t ↦ h t (mem_univ _)⟩


lemma isIntegralCurveAt_iff :
    IsIntegralCurveAt γ v t₀ ↔ ∃ s ∈ 𝓝 t₀, IsIntegralCurveOn γ v s := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    ⊢ Iff (IsIntegralCurveAt γ v t₀) (Exists fun s => And (Membership.mem (nhds t₀ …
  -/
  simp_rw [IsIntegralCurveOn, ← Filter.eventually_iff_exists_mem, IsIntegralCurveAt]
  /-
    🎉 no goals
  -/


/-- `γ` is an integral curve for `v` at `t₀` iff `γ` is an integral curve on some interval
containing `t₀`. -/
lemma isIntegralCurveAt_iff' :
    IsIntegralCurveAt γ v t₀ ↔ ∃ ε > 0, IsIntegralCurveOn γ v (Metric.ball t₀ ε) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    ⊢ Iff (IsIntegralCurveAt γ v t₀) (Exists fun ε => And (GT.gt ε 0) (IsIntegralC …
  -/
  simp_rw [IsIntegralCurveOn, ← Metric.eventually_nhds_iff_ball, IsIntegralCurveAt]
  /-
    🎉 no goals
  -/


lemma IsIntegralCurve.isIntegralCurveAt (h : IsIntegralCurve γ v) (t : ℝ) :
    IsIntegralCurveAt γ v t := isIntegralCurveAt_iff.mpr ⟨univ, Filter.univ_mem, fun t _ ↦ h t⟩


lemma isIntegralCurve_iff_isIntegralCurveAt :
    IsIntegralCurve γ v ↔ ∀ t : ℝ, IsIntegralCurveAt γ v t :=
  ⟨fun h ↦ h.isIntegralCurveAt, fun h t ↦ by
    /-
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      h : ∀ (t : Real), IsIntegralCurveAt γ v t
      t : Real
      ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap.smu …
    -/
    obtain ⟨s, hs, h⟩ := isIntegralCurveAt_iff.mp (h t)
    /-
      case intro.intro
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type u_2
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      γ : Real → M
      v : (x : M) → TangentSpace I x
      h✝ : ∀ (t : Real), IsIntegralCurveAt γ v t
      t : Real
      s : Set Real
      hs : Membership.mem (nhds t) s
      h : IsIntegralCurveOn γ v s
      ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap.smu …
    -/
    exact h t (mem_of_mem_nhds hs)⟩
    /-
      🎉 no goals
    -/


lemma IsIntegralCurveOn.mono (h : IsIntegralCurveOn γ v s) (hs : s' ⊆ s) :
    IsIntegralCurveOn γ v s' := fun t ht ↦ h t (mem_of_mem_of_subset ht hs)


lemma IsIntegralCurveOn.of_union (h : IsIntegralCurveOn γ v s) (h' : IsIntegralCurveOn γ v s') :
    IsIntegralCurveOn γ v (s ∪ s') := fun _ ↦ fun | .inl ht => h _ ht | .inr ht => h' _ ht


lemma IsIntegralCurveAt.hasMFDerivAt (h : IsIntegralCurveAt γ v t₀) :
    HasMFDerivAt 𝓘(ℝ, ℝ) I γ t₀ ((1 : ℝ →L[ℝ] ℝ).smulRight (v (γ t₀))) :=
  have ⟨_, hs, h⟩ := isIntegralCurveAt_iff.mp h
  h t₀ (mem_of_mem_nhds hs)


lemma IsIntegralCurveOn.isIntegralCurveAt (h : IsIntegralCurveOn γ v s) (hs : s ∈ 𝓝 t₀) :
    IsIntegralCurveAt γ v t₀ := isIntegralCurveAt_iff.mpr ⟨s, hs, h⟩


/-- If `γ` is an integral curve at each `t ∈ s`, it is an integral curve on `s`. -/
lemma IsIntegralCurveAt.isIntegralCurveOn (h : ∀ t ∈ s, IsIntegralCurveAt γ v t) :
    IsIntegralCurveOn γ v s := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    h : ∀ (t : Real), Membership.mem s t → IsIntegralCurveAt γ v t
    ⊢ IsIntegralCurveOn γ v s
  -/
  intros t ht
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    h : ∀ (t : Real), Membership.mem s t → IsIntegralCurveAt γ v t
    t : Real
    ht : Membership.mem s t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap.smu …
  -/
  obtain ⟨s, hs, h⟩ := isIntegralCurveAt_iff.mp (h t ht)
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type u_2
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s✝ : Set Real
    h✝ : ∀ (t : Real), Membership.mem s✝ t → IsIntegralCurveAt γ v t
    t : Real
    ht : Membership.mem s✝ t
    s : Set Real
    hs : Membership.mem (nhds t) s
    h : IsIntegralCurveOn γ v s
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap.smu …
  -/
  exact h t (mem_of_mem_nhds hs)
  /-
    🎉 no goals
  -/


lemma isIntegralCurveOn_iff_isIntegralCurveAt (hs : IsOpen s) :
    IsIntegralCurveOn γ v s ↔ ∀ t ∈ s, IsIntegralCurveAt γ v t :=
  ⟨fun h _ ht ↦ h.isIntegralCurveAt (hs.mem_nhds ht), IsIntegralCurveAt.isIntegralCurveOn⟩


lemma IsIntegralCurveOn.continuousAt (hγ : IsIntegralCurveOn γ v s) (ht : t₀ ∈ s) :
    ContinuousAt γ t₀ := (hγ t₀ ht).1


lemma IsIntegralCurveOn.continuousOn (hγ : IsIntegralCurveOn γ v s) :
    ContinuousOn γ s := fun t ht ↦ (hγ t ht).1.continuousWithinAt


lemma IsIntegralCurveAt.continuousAt (hγ : IsIntegralCurveAt γ v t₀) :
    ContinuousAt γ t₀ :=
  have ⟨_, hs, hγ⟩ := isIntegralCurveAt_iff.mp hγ
  hγ.continuousAt <| mem_of_mem_nhds hs


lemma IsIntegralCurve.continuous (hγ : IsIntegralCurve γ v) : Continuous γ :=
  continuous_iff_continuousAt.mpr fun _ ↦ (hγ.isIntegralCurveOn univ).continuousAt (mem_univ _)


/-- If `γ` is an integral curve of a vector field `v`, then `γ t` is tangent to `v (γ t)` when
  expressed in the local chart around the initial point `γ t₀`. -/
lemma IsIntegralCurveOn.hasDerivAt (hγ : IsIntegralCurveOn γ v s) {t : ℝ} (ht : t ∈ s)
    (hsrc : γ t ∈ (extChartAt I (γ t₀)).source) :
    HasDerivAt ((extChartAt I (γ t₀)) ∘ γ)
      (tangentCoordChange I (γ t) (γ t₀) (γ t) (v (γ t))) t := by
  -- turn `HasDerivAt` into comp of `HasMFDerivAt`
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
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveOn γ v s
    t : Real
    ht : Membership.mem s t
    hsrc : Membership.mem (extChartAt I (γ t₀)).source (γ t)
    ⊢ HasDerivAt (Function.comp (↑(extChartAt I (γ t₀))) γ) ((tangentCoordChange I …
  -/
  have hsrc := extChartAt_source I (γ t₀) ▸ hsrc
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
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveOn γ v s
    t : Real
    ht : Membership.mem s t
    hsrc✝ : Membership.mem (extChartAt I (γ t₀)).source (γ t)
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    ⊢ HasDerivAt (Function.comp (↑(extChartAt I (γ t₀))) γ) ((tangentCoordChange I …
  -/
  rw [hasDerivAt_iff_hasFDerivAt, ← hasMFDerivAt_iff_hasFDerivAt]
  apply (HasMFDerivAt.comp t
    (hasMFDerivAt_extChartAt (I := I) hsrc) (hγ _ ht)).congr_mfderiv
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
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveOn γ v s
    t : Real
    ht : Membership.mem s t
    hsrc✝ : Membership.mem (extChartAt I (γ t₀)).source (γ t)
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    ⊢ Eq ((mfderiv I I (↑(chartAt H (γ t₀))) (γ t)).comp (ContinuousLinearMap.smul …
  -/
  rw [ContinuousLinearMap.ext_iff]
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
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveOn γ v s
    t : Real
    ht : Membership.mem s t
    hsrc✝ : Membership.mem (extChartAt I (γ t₀)).source (γ t)
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    ⊢ ∀ (x : TangentSpace (modelWithCornersSelf Real Real) t), Eq (((mfderiv I I ( …
  -/
  intro a
  rw [ContinuousLinearMap.comp_apply, ContinuousLinearMap.smulRight_apply, map_smul,
    ← ContinuousLinearMap.one_apply (R₁ := ℝ) a, ← ContinuousLinearMap.smulRight_apply,
    mfderiv_chartAt_eq_tangentCoordChange hsrc]
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
    γ : Real → M
    v : (x : M) → TangentSpace I x
    s : Set Real
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveOn γ v s
    t : Real
    ht : Membership.mem s t
    hsrc✝ : Membership.mem (extChartAt I (γ t₀)).source (γ t)
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    a : TangentSpace (modelWithCornersSelf Real Real) t
    ⊢ Eq ((ContinuousLinearMap.smulRight 1 ((tangentCoordChange I (γ t) (γ t₀) (γ  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma IsIntegralCurveAt.eventually_hasDerivAt (hγ : IsIntegralCurveAt γ v t₀) :
    ∀ᶠ t in 𝓝 t₀, HasDerivAt ((extChartAt I (γ t₀)) ∘ γ)
      (tangentCoordChange I (γ t) (γ t₀) (γ t) (v (γ t))) t := by
  apply eventually_mem_nhds_iff.mpr
    (hγ.continuousAt.preimage_mem_nhds (extChartAt_source_mem_nhds (I := I) _)) |>.and hγ |>.mono
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
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveAt γ v t₀
    ⊢ ∀ (x : Real), And (Membership.mem (nhds x) (Set.preimage γ (extChartAt I (γ  …
  -/
  rintro t ⟨ht1, ht2⟩
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveAt γ v t₀
    t : Real
    ht1 : Membership.mem (nhds t) (Set.preimage γ (extChartAt I (γ t₀)).source)
    ht2 : HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap …
    ⊢ HasDerivAt (Function.comp (↑(extChartAt I (γ t₀))) γ) ((tangentCoordChange I …
  -/
  have hsrc := mem_of_mem_nhds ht1
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveAt γ v t₀
    t : Real
    ht1 : Membership.mem (nhds t) (Set.preimage γ (extChartAt I (γ t₀)).source)
    ht2 : HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap …
    hsrc : Membership.mem (Set.preimage γ (extChartAt I (γ t₀)).source) t
    ⊢ HasDerivAt (Function.comp (↑(extChartAt I (γ t₀))) γ) ((tangentCoordChange I …
  -/
  rw [mem_preimage, extChartAt_source I (γ t₀)] at hsrc
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveAt γ v t₀
    t : Real
    ht1 : Membership.mem (nhds t) (Set.preimage γ (extChartAt I (γ t₀)).source)
    ht2 : HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap …
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    ⊢ HasDerivAt (Function.comp (↑(extChartAt I (γ t₀))) γ) ((tangentCoordChange I …
  -/
  rw [hasDerivAt_iff_hasFDerivAt, ← hasMFDerivAt_iff_hasFDerivAt]
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveAt γ v t₀
    t : Real
    ht1 : Membership.mem (nhds t) (Set.preimage γ (extChartAt I (γ t₀)).source)
    ht2 : HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap …
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) (modelWithCornersSelf Real E)  …
  -/
  apply (HasMFDerivAt.comp t (hasMFDerivAt_extChartAt (I := I) hsrc) ht2).congr_mfderiv
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveAt γ v t₀
    t : Real
    ht1 : Membership.mem (nhds t) (Set.preimage γ (extChartAt I (γ t₀)).source)
    ht2 : HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap …
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    ⊢ Eq ((mfderiv I I (↑(chartAt H (γ t₀))) (γ t)).comp (ContinuousLinearMap.smul …
  -/
  rw [ContinuousLinearMap.ext_iff]
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveAt γ v t₀
    t : Real
    ht1 : Membership.mem (nhds t) (Set.preimage γ (extChartAt I (γ t₀)).source)
    ht2 : HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap …
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    ⊢ ∀ (x : TangentSpace (modelWithCornersSelf Real Real) t), Eq (((mfderiv I I ( …
  -/
  intro a
  rw [ContinuousLinearMap.comp_apply, ContinuousLinearMap.smulRight_apply, map_smul,
    ← ContinuousLinearMap.one_apply (R₁ := ℝ) a, ← ContinuousLinearMap.smulRight_apply,
    mfderiv_chartAt_eq_tangentCoordChange hsrc]
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type u_2
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    γ : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : SmoothManifoldWithCorners I M
    hγ : IsIntegralCurveAt γ v t₀
    t : Real
    ht1 : Membership.mem (nhds t) (Set.preimage γ (extChartAt I (γ t₀)).source)
    ht2 : HasMFDerivAt (modelWithCornersSelf Real Real) I γ t (ContinuousLinearMap …
    hsrc : Membership.mem (chartAt H (γ t₀)).source (γ t)
    a : TangentSpace (modelWithCornersSelf Real Real) t
    ⊢ Eq ((ContinuousLinearMap.smulRight 1 ((tangentCoordChange I (γ t) (γ t₀) (γ  …
  -/
  rfl
  /-
    🎉 no goals
  -/

