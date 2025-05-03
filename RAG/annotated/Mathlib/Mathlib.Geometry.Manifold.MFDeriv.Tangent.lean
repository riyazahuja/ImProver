/-- The derivative of the chart at a base point is the chart of the tangent bundle, composed with
the identification between the tangent bundle of the model space and the product space. -/
theorem tangentMap_chart {p q : TangentBundle I M} (h : q.1 ∈ (chartAt H p.1).source) :
    tangentMap I I (chartAt H p.1) q =
      (TotalSpace.toProd _ _).symm
        ((chartAt (ModelProd H E) p : TangentBundle I M → ModelProd H E) q) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p q : TangentBundle I M
    h : Membership.mem (chartAt H p.proj).source q.proj
    ⊢ Eq (tangentMap I I (↑(chartAt H p.proj)) q) ((Bundle.TotalSpace.toProd H E). …
  -/
  dsimp [tangentMap]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p q : TangentBundle I M
    h : Membership.mem (chartAt H p.proj).source q.proj
    ⊢ Eq { proj := ↑(chartAt H p.proj) q.proj, snd := (mfderiv I I (↑(chartAt H p. …
  -/
  rw [MDifferentiableAt.mfderiv]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : SmoothManifoldWithCorners I M
      p q : TangentBundle I M
      h : Membership.mem (chartAt H p.proj).source q.proj
      ⊢ Eq { proj := ↑(chartAt H p.proj) q.proj, snd := (fderivWithin 𝕜 (writtenInEx …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : SmoothManifoldWithCorners I M
      p q : TangentBundle I M
      h : Membership.mem (chartAt H p.proj).source q.proj
      ⊢ MDifferentiableAt I I (↑(chartAt H p.proj)) q.proj
    -/
  · exact mdifferentiableAt_atlas (chart_mem_atlas _ _) h
    /-
      🎉 no goals
    -/


/-- The derivative of the inverse of the chart at a base point is the inverse of the chart of the
tangent bundle, composed with the identification between the tangent bundle of the model space and
the product space. -/
theorem tangentMap_chart_symm {p : TangentBundle I M} {q : TangentBundle I H}
    (h : q.1 ∈ (chartAt H p.1).target) :
    tangentMap I I (chartAt H p.1).symm q =
      (chartAt (ModelProd H E) p).symm (TotalSpace.toProd H E q) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p : TangentBundle I M
    q : TangentBundle I H
    h : Membership.mem (chartAt H p.proj).target q.proj
    ⊢ Eq (tangentMap I I (↑(chartAt H p.proj).symm) q) (↑(chartAt (ModelProd H E)  …
  -/
  dsimp only [tangentMap]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p : TangentBundle I M
    q : TangentBundle I H
    h : Membership.mem (chartAt H p.proj).target q.proj
    ⊢ Eq { proj := ↑(chartAt H p.proj).symm q.proj, snd := (mfderiv I I (↑(chartAt …
  -/
  rw [MDifferentiableAt.mfderiv (mdifferentiableAt_atlas_symm (chart_mem_atlas _ _) h)]
  simp only [ContinuousLinearMap.coe_coe, TangentBundle.chartAt, h, tangentBundleCore,
    mfld_simps, (· ∘ ·)]
  -- `simp` fails to apply `PartialEquiv.prod_symm` with `ModelProd`
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p : TangentBundle I M
    q : TangentBundle I H
    h : Membership.mem (chartAt H p.proj).target q.proj
    ⊢ Eq { proj := ↑(chartAt H p.proj).symm q.proj, snd := (fderivWithin 𝕜 (Functi …
  -/
  congr
  /-
    case e_snd.e_a.e_x.e_a
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p : TangentBundle I M
    q : TangentBundle I H
    h : Membership.mem (chartAt H p.proj).target q.proj
    ⊢ Eq q.proj (↑↑(achart H p.proj) (↑((chartAt H p.proj).prod (PartialHomeomorph …
  -/
  exact ((chartAt H (TotalSpace.proj p)).right_inv h).symm
  /-
    🎉 no goals
  -/


lemma mfderiv_chartAt_eq_tangentCoordChange {x y : M} (hsrc : x ∈ (chartAt H y).source) :
    mfderiv I I (chartAt H y) x = tangentCoordChange I x y x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    x y : M
    hsrc : Membership.mem (chartAt H y).source x
    ⊢ Eq (mfderiv I I (↑(chartAt H y)) x) (tangentCoordChange I x y x)
  -/
  have := mdifferentiableAt_atlas (I := I) (ChartedSpace.chart_mem_atlas _) hsrc
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    x y : M
    hsrc : Membership.mem (chartAt H y).source x
    this : MDifferentiableAt I I (↑(ChartedSpace.chartAt y)) x
    ⊢ Eq (mfderiv I I (↑(chartAt H y)) x) (tangentCoordChange I x y x)
  -/
  simp [mfderiv, if_pos this, Function.comp_assoc]
  /-
    🎉 no goals
  -/


/-- The preimage under the projection from the tangent bundle of a set with unique differential in
the basis also has unique differential. -/
theorem UniqueMDiffOn.tangentBundle_proj_preimage {s : Set M} (hs : UniqueMDiffOn I s) :
    UniqueMDiffOn I.tangent (π E (TangentSpace I) ⁻¹' s) :=
  hs.bundle_preimage _


/-- To write a linear map between tangent spaces in coordinates amounts to precomposing and
postcomposing it with derivatives of extended charts.
Concrete version of `inTangentCoordinates_eq`. -/
lemma inTangentCoordinates_eq_mfderiv_comp
    {N : Type*} {f : N → M} {g : N → M'}
    {ϕ : Π x : N, TangentSpace I (f x) →L[𝕜] TangentSpace I' (g x)} {x₀ : N} {x : N}
    (hx : f x ∈ (chartAt H (f x₀)).source) (hy : g x ∈ (chartAt H' (g x₀)).source) :
    inTangentCoordinates I I' f g ϕ x₀ x =
    (mfderiv I' 𝓘(𝕜, E') (extChartAt I' (g x₀)) (g x)) ∘L (ϕ x) ∘L
      (mfderivWithin 𝓘(𝕜, E) I (extChartAt I (f x₀)).symm (range I)
        (extChartAt I (f x₀) (f x))) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    inst✝⁶ : SmoothManifoldWithCorners I M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : SmoothManifoldWithCorners I' M'
    N : Type u_8
    f : N → M
    g : N → M'
    ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
    x₀ x : N
    hx : Membership.mem (chartAt H (f x₀)).source (f x)
    hy : Membership.mem (chartAt H' (g x₀)).source (g x)
    ⊢ Eq (inTangentCoordinates I I' f g ϕ x₀ x) ((mfderiv I' (modelWithCornersSelf …
  -/
  rw [inTangentCoordinates_eq _ _ _ hx hy, tangentBundleCore_coordChange]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    inst✝⁶ : SmoothManifoldWithCorners I M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : SmoothManifoldWithCorners I' M'
    N : Type u_8
    f : N → M
    g : N → M'
    ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
    x₀ x : N
    hx : Membership.mem (chartAt H (f x₀)).source (f x)
    hy : Membership.mem (chartAt H' (g x₀)).source (g x)
    ⊢ Eq ((fderivWithin 𝕜 (Function.comp ↑((↑(achart H' (g x₀))).extend I') ↑((↑(a …
  -/
  congr
  · have : MDifferentiableAt I' 𝓘(𝕜, E') (extChartAt I' (g x₀)) (g x) :=
      mdifferentiableAt_extChartAt hy
    /-
      case e_g
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      inst✝⁶ : SmoothManifoldWithCorners I M
      E' : Type u_5
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝³ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      N : Type u_8
      f : N → M
      g : N → M'
      ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
      x₀ x : N
      hx : Membership.mem (chartAt H (f x₀)).source (f x)
      hy : Membership.mem (chartAt H' (g x₀)).source (g x)
      this : MDifferentiableAt I' (modelWithCornersSelf 𝕜 E') (↑(extChartAt I' (g x₀ …
      ⊢ Eq (fderivWithin 𝕜 (Function.comp ↑((↑(achart H' (g x₀))).extend I') ↑((↑(ac …
    -/
    simp at this
    /-
      case e_g
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      inst✝⁶ : SmoothManifoldWithCorners I M
      E' : Type u_5
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝³ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      N : Type u_8
      f : N → M
      g : N → M'
      ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
      x₀ x : N
      hx : Membership.mem (chartAt H (f x₀)).source (f x)
      hy : Membership.mem (chartAt H' (g x₀)).source (g x)
      this : MDifferentiableAt I' (modelWithCornersSelf 𝕜 E') (Function.comp ↑I' ↑(c …
      ⊢ Eq (fderivWithin 𝕜 (Function.comp ↑((↑(achart H' (g x₀))).extend I') ↑((↑(ac …
    -/
    simp [mfderiv, this]
    /-
      🎉 no goals
    -/
    /-
      case e_f.e_f
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      inst✝⁶ : SmoothManifoldWithCorners I M
      E' : Type u_5
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝³ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      N : Type u_8
      f : N → M
      g : N → M'
      ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
      x₀ x : N
      hx : Membership.mem (chartAt H (f x₀)).source (f x)
      hy : Membership.mem (chartAt H' (g x₀)).source (g x)
      ⊢ Eq ((tangentBundleCore I M).coordChange (achart H (f x₀)) (achart H (f x)) ( …
    -/
  · simp only [mfderivWithin, writtenInExtChartAt, modelWithCornersSelf_coe, range_id, inter_univ]
    /-
      case e_f.e_f
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      inst✝⁶ : SmoothManifoldWithCorners I M
      E' : Type u_5
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝³ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      N : Type u_8
      f : N → M
      g : N → M'
      ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
      x₀ x : N
      hx : Membership.mem (chartAt H (f x₀)).source (f x)
      hy : Membership.mem (chartAt H' (g x₀)).source (g x)
      ⊢ Eq ((tangentBundleCore I M).coordChange (achart H (f x₀)) (achart H (f x)) ( …
    -/
    rw [if_pos]
      /-
        case e_f.e_f
        𝕜 : Type u_1
        inst✝¹² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁸ : TopologicalSpace M
        inst✝⁷ : ChartedSpace H M
        inst✝⁶ : SmoothManifoldWithCorners I M
        E' : Type u_5
        inst✝⁵ : NormedAddCommGroup E'
        inst✝⁴ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝³ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝² : TopologicalSpace M'
        inst✝¹ : ChartedSpace H' M'
        inst✝ : SmoothManifoldWithCorners I' M'
        N : Type u_8
        f : N → M
        g : N → M'
        ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
        x₀ x : N
        hx : Membership.mem (chartAt H (f x₀)).source (f x)
        hy : Membership.mem (chartAt H' (g x₀)).source (g x)
        ⊢ Eq ((tangentBundleCore I M).coordChange (achart H (f x₀)) (achart H (f x)) ( …
      -/
    · simp [Function.comp_def, PartialHomeomorph.left_inv (chartAt H (f x₀)) hx]
      /-
        🎉 no goals
      -/
      /-
        case e_f.e_f.hc
        𝕜 : Type u_1
        inst✝¹² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁸ : TopologicalSpace M
        inst✝⁷ : ChartedSpace H M
        inst✝⁶ : SmoothManifoldWithCorners I M
        E' : Type u_5
        inst✝⁵ : NormedAddCommGroup E'
        inst✝⁴ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝³ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝² : TopologicalSpace M'
        inst✝¹ : ChartedSpace H' M'
        inst✝ : SmoothManifoldWithCorners I' M'
        N : Type u_8
        f : N → M
        g : N → M'
        ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
        x₀ x : N
        hx : Membership.mem (chartAt H (f x₀)).source (f x)
        hy : Membership.mem (chartAt H' (g x₀)).source (g x)
        ⊢ MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I (f x₀)) …
      -/
    · apply mdifferentiableWithinAt_extChartAt_symm
      /-
        case e_f.e_f.hc.h
        𝕜 : Type u_1
        inst✝¹² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁸ : TopologicalSpace M
        inst✝⁷ : ChartedSpace H M
        inst✝⁶ : SmoothManifoldWithCorners I M
        E' : Type u_5
        inst✝⁵ : NormedAddCommGroup E'
        inst✝⁴ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝³ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝² : TopologicalSpace M'
        inst✝¹ : ChartedSpace H' M'
        inst✝ : SmoothManifoldWithCorners I' M'
        N : Type u_8
        f : N → M
        g : N → M'
        ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
        x₀ x : N
        hx : Membership.mem (chartAt H (f x₀)).source (f x)
        hy : Membership.mem (chartAt H' (g x₀)).source (g x)
        ⊢ Membership.mem (extChartAt I (f x₀)).target (↑(extChartAt I (f x₀)) (f x))
      -/
      apply (extChartAt I (f x₀)).map_source
      /-
        case e_f.e_f.hc.h
        𝕜 : Type u_1
        inst✝¹² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁸ : TopologicalSpace M
        inst✝⁷ : ChartedSpace H M
        inst✝⁶ : SmoothManifoldWithCorners I M
        E' : Type u_5
        inst✝⁵ : NormedAddCommGroup E'
        inst✝⁴ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝³ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝² : TopologicalSpace M'
        inst✝¹ : ChartedSpace H' M'
        inst✝ : SmoothManifoldWithCorners I' M'
        N : Type u_8
        f : N → M
        g : N → M'
        ϕ : (x : N) → ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I (f x)) (Tange …
        x₀ x : N
        hx : Membership.mem (chartAt H (f x₀)).source (f x)
        hy : Membership.mem (chartAt H' (g x₀)).source (g x)
        ⊢ Membership.mem (extChartAt I (f x₀)).source (f x)
      -/
      simpa using hx
      /-
        🎉 no goals
      -/


variable (I) in
/-- The canonical identification between the tangent bundle to the model space and the product,
as a diffeomorphism -/
def tangentBundleModelSpaceDiffeomorph (n : ℕ∞) :
    TangentBundle I H ≃ₘ^n⟮I.tangent, I.prod 𝓘(𝕜, E)⟯ ModelProd H E where
  __ := TotalSpace.toProd H E
  contMDiff_toFun := contMDiff_tangentBundleModelSpaceHomeomorph
  contMDiff_invFun := contMDiff_tangentBundleModelSpaceHomeomorph_symm

