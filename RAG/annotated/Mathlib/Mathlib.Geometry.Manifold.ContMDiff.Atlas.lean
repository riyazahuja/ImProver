theorem contMDiff_model : ContMDiff I 𝓘(𝕜, E) n I := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    ⊢ ContMDiff I (modelWithCornersSelf 𝕜 E) n ↑I
  -/
  intro x
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : H
    ⊢ ContMDiffAt I (modelWithCornersSelf 𝕜 E) n (↑I) x
  -/
  refine contMDiffAt_iff.mpr ⟨I.continuousAt, ?_⟩
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : H
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt (modelWithCornersSelf 𝕜 …
  -/
  simp only [mfld_simps]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : H
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp ↑I ↑I.symm) (Set.range ↑I) (↑I x)
  -/
  refine contDiffWithinAt_id.congr_of_eventuallyEq ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      n : ENat
      x : H
      ⊢ (nhdsWithin (↑I x) (Set.range ↑I)).EventuallyEq (Function.comp ↑I ↑I.symm) id
    -/
  · exact Filter.eventuallyEq_of_mem self_mem_nhdsWithin fun x₂ => I.right_inv
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : H
    ⊢ Eq (Function.comp (↑I) (↑I.symm) (↑I x)) (id (↑I x))
  -/
  simp_rw [Function.comp_apply, I.left_inv, Function.id_def]
  /-
    🎉 no goals
  -/


theorem contMDiffOn_model_symm : ContMDiffOn 𝓘(𝕜, E) I n I.symm (range I) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    ⊢ ContMDiffOn (modelWithCornersSelf 𝕜 E) I n (↑I.symm) (Set.range ↑I)
  -/
  rw [contMDiffOn_iff]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    ⊢ And (ContinuousOn (↑I.symm) (Set.range ↑I)) (∀ (x : E) (y : H), ContDiffOn 𝕜 …
  -/
  refine ⟨I.continuousOn_symm, fun x y => ?_⟩
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : E
    y : H
    ⊢ ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I y)) (Function.comp ↑I.symm  …
  -/
  simp only [mfld_simps]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : E
    y : H
    ⊢ ContDiffOn 𝕜 (↑n) (Function.comp ↑I ↑I.symm) (Set.range ↑I)
  -/
  exact contDiffOn_id.congr fun x' => I.right_inv
  /-
    🎉 no goals
  -/


/-- An atlas member is `C^n` for any `n`. -/
theorem contMDiffOn_of_mem_maximalAtlas (h : e ∈ maximalAtlas I M) : ContMDiffOn I I n e e.source :=
  ContMDiffOn.of_le ((contDiffWithinAt_localInvariantProp ⊤).liftPropOn_of_mem_maximalAtlas
      contDiffWithinAtProp_id h) le_top


/-- The inverse of an atlas member is `C^n` for any `n`. -/
theorem contMDiffOn_symm_of_mem_maximalAtlas (h : e ∈ maximalAtlas I M) :
    ContMDiffOn I I n e.symm e.target :=
  ContMDiffOn.of_le ((contDiffWithinAt_localInvariantProp ⊤).liftPropOn_symm_of_mem_maximalAtlas
      contDiffWithinAtProp_id h) le_top


theorem contMDiffAt_of_mem_maximalAtlas (h : e ∈ maximalAtlas I M) (hx : x ∈ e.source) :
    ContMDiffAt I I n e x :=
  (contMDiffOn_of_mem_maximalAtlas h).contMDiffAt <| e.open_source.mem_nhds hx


theorem contMDiffAt_symm_of_mem_maximalAtlas {x : H} (h : e ∈ maximalAtlas I M)
    (hx : x ∈ e.target) : ContMDiffAt I I n e.symm x :=
  (contMDiffOn_symm_of_mem_maximalAtlas h).contMDiffAt <| e.open_target.mem_nhds hx


theorem contMDiffOn_chart : ContMDiffOn I I n (chartAt H x) (chartAt H x).source :=
  contMDiffOn_of_mem_maximalAtlas <| chart_mem_maximalAtlas x


theorem contMDiffOn_chart_symm : ContMDiffOn I I n (chartAt H x).symm (chartAt H x).target :=
  contMDiffOn_symm_of_mem_maximalAtlas <| chart_mem_maximalAtlas x


theorem contMDiffAt_extend {x : M} (he : e ∈ maximalAtlas I M) (hx : x ∈ e.source) :
    ContMDiffAt I 𝓘(𝕜, E) n (e.extend I) x :=
  (contMDiff_model _).comp x <| contMDiffAt_of_mem_maximalAtlas he hx


theorem contMDiffAt_extChartAt' {x' : M} (h : x' ∈ (chartAt H x).source) :
    ContMDiffAt I 𝓘(𝕜, E) n (extChartAt I x) x' :=
  contMDiffAt_extend (chart_mem_maximalAtlas x) h


theorem contMDiffAt_extChartAt : ContMDiffAt I 𝓘(𝕜, E) n (extChartAt I x) x :=
  contMDiffAt_extChartAt' <| mem_chart_source H x


theorem contMDiffOn_extChartAt : ContMDiffOn I 𝓘(𝕜, E) n (extChartAt I x) (chartAt H x).source :=
  fun _x' hx' => (contMDiffAt_extChartAt' hx').contMDiffWithinAt


theorem contMDiffOn_extend_symm (he : e ∈ maximalAtlas I M) :
    ContMDiffOn 𝓘(𝕜, E) I n (e.extend I).symm (I '' e.target) := by
  refine (contMDiffOn_symm_of_mem_maximalAtlas he).comp
    (contMDiffOn_model_symm.mono <| image_subset_range _ _) ?_
  simp_rw [image_subset_iff, PartialEquiv.restr_coe_symm, I.toPartialEquiv_coe_symm,
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
                                                    e : PartialHomeomorph M H
                                                    n : ENat
                                                    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
                                                    ⊢ HasSubset.Subset e.target e.target
                                                  -/
    preimage_preimage, I.left_inv, preimage_id']; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem contMDiffOn_extChartAt_symm (x : M) :
    ContMDiffOn 𝓘(𝕜, E) I n (extChartAt I x).symm (extChartAt I x).target := by
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
    n : ENat
    x : M
    ⊢ ContMDiffOn (modelWithCornersSelf 𝕜 E) I n (↑(extChartAt I x).symm) (extChar …
  -/
  convert contMDiffOn_extend_symm (chart_mem_maximalAtlas (I := I) x)
  /-
    case h.e'_23
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
    n : ENat
    x : M
    ⊢ Eq (extChartAt I x).target (Set.image (↑I) (chartAt H x).target)
  -/
  rw [extChartAt_target, I.image_eq]
  /-
    🎉 no goals
  -/


theorem contMDiffWithinAt_extChartAt_symm_target
    (x : M) {y : E} (hy : y ∈ (extChartAt I x).target) :
    ContMDiffWithinAt 𝓘(𝕜, E) I n (extChartAt I x).symm (extChartAt I x).target y :=
  contMDiffOn_extChartAt_symm x y hy


theorem contMDiffWithinAt_extChartAt_symm_range
    (x : M) {y : E} (hy : y ∈ (extChartAt I x).target) :
    ContMDiffWithinAt 𝓘(𝕜, E) I n (extChartAt I x).symm (range I) y :=
  (contMDiffWithinAt_extChartAt_symm_target x hy).mono_of_mem_nhdsWithin
    (extChartAt_target_mem_nhdsWithin_of_mem hy)


/-- An element of `contDiffGroupoid ⊤ I` is `C^n` for any `n`. -/
theorem contMDiffOn_of_mem_contDiffGroupoid {e' : PartialHomeomorph H H}
    (h : e' ∈ contDiffGroupoid ∞ I) : ContMDiffOn I I n e' e'.source :=
  (contDiffWithinAt_localInvariantProp n).liftPropOn_of_mem_groupoid contDiffWithinAtProp_id h


theorem isLocalStructomorphOn_contDiffGroupoid_iff_aux {f : PartialHomeomorph M M'}
    (hf : LiftPropOn (contDiffGroupoid ∞ I).IsLocalStructomorphWithinAt f f.source) :
    ContMDiffOn I I ⊤ f f.source := by
  -- It suffices to show smoothness near each `x`
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    ⊢ ContMDiffOn I I Top.top (↑f) f.source
  -/
  apply contMDiffOn_of_locally_contMDiffOn
  /-
    case h
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    ⊢ ∀ (x : M), Membership.mem f.source x → Exists fun u => And (IsOpen u) (And ( …
  -/
  intro x hx
  /-
    case h
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    x : M
    hx : Membership.mem f.source x
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I To …
  -/
  let c := chartAt H x
  /-
    case h
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    x : M
    hx : Membership.mem f.source x
    c : PartialHomeomorph M H := chartAt H x
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I To …
  -/
  let c' := chartAt H (f x)
  /-
    case h
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    x : M
    hx : Membership.mem f.source x
    c : PartialHomeomorph M H := chartAt H x
    c' : PartialHomeomorph M' H := chartAt H (↑f x)
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I To …
  -/
  obtain ⟨-, hxf⟩ := hf x hx
  -- Since `f` is a local structomorph, it is locally equal to some transferred element `e` of
  -- the `contDiffGroupoid`.
  obtain
    ⟨e, he, he' : EqOn (c' ∘ f ∘ c.symm) e (c.symm ⁻¹' f.source ∩ e.source), hex :
      c x ∈ e.source⟩ :=
    hxf (by simp only [hx, mfld_simps])
  -- We choose a convenient set `s` in `M`.
  /-
    case h.mk.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    x : M
    hx : Membership.mem f.source x
    c : PartialHomeomorph M H := chartAt H x
    c' : PartialHomeomorph M' H := chartAt H (↑f x)
    hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
    e : PartialHomeomorph H H
    he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
    he' : Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑e) (Inter.in …
    hex : Membership.mem e.source (↑c x)
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I To …
  -/
  let s : Set M := (f.trans c').source ∩ ((c.trans e).trans c'.symm).source
  /-
    case h.mk.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    x : M
    hx : Membership.mem f.source x
    c : PartialHomeomorph M H := chartAt H x
    c' : PartialHomeomorph M' H := chartAt H (↑f x)
    hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
    e : PartialHomeomorph H H
    he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
    he' : Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑e) (Inter.in …
    hex : Membership.mem e.source (↑c x)
    s : Set M := Inter.inter (f.trans c').source ((c.trans e).trans c'.symm).source
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I To …
  -/
  refine ⟨s, (f.trans c').open_source.inter ((c.trans e).trans c'.symm).open_source, ?_, ?_⟩
    /-
      case h.mk.intro.intro.intro.refine_1
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
      x : M
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      c' : PartialHomeomorph M' H := chartAt H (↑f x)
      hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      he' : Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑e) (Inter.in …
      hex : Membership.mem e.source (↑c x)
      s : Set M := Inter.inter (f.trans c').source ((c.trans e).trans c'.symm).source
      ⊢ Membership.mem s x
    -/
  · simp only [s, mfld_simps]
    /-
      case h.mk.intro.intro.intro.refine_1
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
      x : M
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      c' : PartialHomeomorph M' H := chartAt H (↑f x)
      hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      he' : Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑e) (Inter.in …
      hex : Membership.mem e.source (↑c x)
      s : Set M := Inter.inter (f.trans c').source ((c.trans e).trans c'.symm).source
      ⊢ And (And (Membership.mem f.source x) (Membership.mem c'.source (↑f x))) (And …
    -/
                   /-
                     🎉 no goals
                   -/
    rw [← he'] <;> simp only [c, c', hx, hex, mfld_simps]
                   /-
                     🎉 no goals
                   -/
  -- We need to show `f` is `ContMDiffOn` the domain `s ∩ f.source`.  We show this in two
  -- steps: `f` is equal to `c'.symm ∘ e ∘ c` on that domain and that function is
  -- `ContMDiffOn` it.
  have H₁ : ContMDiffOn I I ⊤ (c'.symm ∘ e ∘ c) s := by
    have hc' : ContMDiffOn I I ⊤ c'.symm _ := contMDiffOn_chart_symm
    have he'' : ContMDiffOn I I ⊤ e _ := contMDiffOn_of_mem_contDiffGroupoid he
    have hc : ContMDiffOn I I ⊤ c _ := contMDiffOn_chart
    refine (hc'.comp' (he''.comp' hc)).mono ?_
    dsimp [s, c, c']
    mfld_set_tac
  have H₂ : EqOn f (c'.symm ∘ e ∘ c) s := by
    intro y hy
    simp only [s, mfld_simps] at hy
    have hy₁ : f y ∈ c'.source := by simp only [hy, mfld_simps]
    have hy₂ : y ∈ c.source := by simp only [hy, mfld_simps]
    have hy₃ : c y ∈ c.symm ⁻¹' f.source ∩ e.source := by simp only [hy, mfld_simps]
    calc
      f y = c'.symm (c' (f y)) := by rw [c'.left_inv hy₁]
      _ = c'.symm (c' (f (c.symm (c y)))) := by rw [c.left_inv hy₂]
      _ = c'.symm (e (c y)) := by rw [← he' hy₃]; rfl
  /-
    case h.mk.intro.intro.intro.refine_2
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    x : M
    hx : Membership.mem f.source x
    c : PartialHomeomorph M H := chartAt H x
    c' : PartialHomeomorph M' H := chartAt H (↑f x)
    hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
    e : PartialHomeomorph H H
    he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
    he' : Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑e) (Inter.in …
    hex : Membership.mem e.source (↑c x)
    s : Set M := Inter.inter (f.trans c').source ((c.trans e).trans c'.symm).source
    H₁ : ContMDiffOn I I Top.top (Function.comp (↑c'.symm) (Function.comp ↑e ↑c)) s
    H₂ : Set.EqOn (↑f) (Function.comp (↑c'.symm) (Function.comp ↑e ↑c)) s
    ⊢ ContMDiffOn I I Top.top (↑f) (Inter.inter f.source s)
  -/
  refine (H₁.congr H₂).mono ?_
  /-
    case h.mk.intro.intro.intro.refine_2
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    hf : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomor …
    x : M
    hx : Membership.mem f.source x
    c : PartialHomeomorph M H := chartAt H x
    c' : PartialHomeomorph M' H := chartAt H (↑f x)
    hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
    e : PartialHomeomorph H H
    he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
    he' : Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑e) (Inter.in …
    hex : Membership.mem e.source (↑c x)
    s : Set M := Inter.inter (f.trans c').source ((c.trans e).trans c'.symm).source
    H₁ : ContMDiffOn I I Top.top (Function.comp (↑c'.symm) (Function.comp ↑e ↑c)) s
    H₂ : Set.EqOn (↑f) (Function.comp (↑c'.symm) (Function.comp ↑e ↑c)) s
    ⊢ HasSubset.Subset (Inter.inter f.source s) s
  -/
  mfld_set_tac
  /-
    🎉 no goals
  -/


/-- Let `M` and `M'` be smooth manifolds with the same model-with-corners, `I`.  Then `f : M → M'`
is a local structomorphism for `I`, if and only if it is manifold-smooth on the domain of definition
in both directions. -/
theorem isLocalStructomorphOn_contDiffGroupoid_iff (f : PartialHomeomorph M M') :
    LiftPropOn (contDiffGroupoid ∞ I).IsLocalStructomorphWithinAt f f.source ↔
      ContMDiffOn I I ⊤ f f.source ∧ ContMDiffOn I I ⊤ f.symm f.target := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    M' : Type u_5
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H M'
    IsM' : SmoothManifoldWithCorners I M'
    f : PartialHomeomorph M M'
    ⊢ Iff (ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructom …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      ⊢ ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphW …
    -/
  · intro h
    refine ⟨isLocalStructomorphOn_contDiffGroupoid_iff_aux h,
      isLocalStructomorphOn_contDiffGroupoid_iff_aux ?_⟩
    -- todo: we can generalize this part of the proof to a lemma
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      ⊢ ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphW …
    -/
    intro X hX
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      X : M'
      hX : Membership.mem f.symm.source X
      ⊢ ChartedSpace.LiftPropWithinAt (contDiffGroupoid (↑Top.top) I).IsLocalStructo …
    -/
    let x := f.symm X
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      X : M'
      hX : Membership.mem f.symm.source X
      x : M := ↑f.symm X
      ⊢ ChartedSpace.LiftPropWithinAt (contDiffGroupoid (↑Top.top) I).IsLocalStructo …
    -/
    have hx : x ∈ f.source := f.symm.mapsTo hX
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      X : M'
      hX : Membership.mem f.symm.source X
      x : M := ↑f.symm X
      hx : Membership.mem f.source x
      ⊢ ChartedSpace.LiftPropWithinAt (contDiffGroupoid (↑Top.top) I).IsLocalStructo …
    -/
    let c := chartAt H x
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      X : M'
      hX : Membership.mem f.symm.source X
      x : M := ↑f.symm X
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      ⊢ ChartedSpace.LiftPropWithinAt (contDiffGroupoid (↑Top.top) I).IsLocalStructo …
    -/
    let c' := chartAt H X
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      X : M'
      hX : Membership.mem f.symm.source X
      x : M := ↑f.symm X
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      c' : PartialHomeomorph M' H := chartAt H X
      ⊢ ChartedSpace.LiftPropWithinAt (contDiffGroupoid (↑Top.top) I).IsLocalStructo …
    -/
    obtain ⟨-, hxf⟩ := h x hx
    /-
      case mp.mk
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      X : M'
      hX : Membership.mem f.symm.source X
      x : M := ↑f.symm X
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      c' : PartialHomeomorph M' H := chartAt H X
      hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
      ⊢ ChartedSpace.LiftPropWithinAt (contDiffGroupoid (↑Top.top) I).IsLocalStructo …
    -/
    refine ⟨(f.symm.continuousAt hX).continuousWithinAt, fun h2x => ?_⟩
    obtain ⟨e, he, h2e, hef, hex⟩ :
      ∃ e : PartialHomeomorph H H,
        e ∈ contDiffGroupoid ∞ I ∧
          e.source ⊆ (c.symm ≫ₕ f ≫ₕ c').source ∧
            EqOn (c' ∘ f ∘ c.symm) e e.source ∧ c x ∈ e.source := by
      have h1 : c' = chartAt H (f x) := by simp only [x, c', f.right_inv hX]
      have h2 : c' ∘ f ∘ c.symm = ⇑(c.symm ≫ₕ f ≫ₕ c') := rfl
      have hcx : c x ∈ c.symm ⁻¹' f.source := by simp only [c, hx, mfld_simps]
      rw [h2]
      rw [← h1, h2, PartialHomeomorph.isLocalStructomorphWithinAt_iff'] at hxf
      · exact hxf hcx
      · dsimp [x, c]; mfld_set_tac
      · apply Or.inl
        simp only [c, hx, h1, mfld_simps]
    have h2X : c' X = e (c (f.symm X)) := by
      rw [← hef hex]
      dsimp only [Function.comp_def]
      have hfX : f.symm X ∈ c.source := by simp only [c, x, hX, mfld_simps]
      rw [c.left_inv hfX, f.right_inv hX]
    have h3e : EqOn (c ∘ f.symm ∘ c'.symm) e.symm (c'.symm ⁻¹' f.target ∩ e.target) := by
      have h1 : EqOn (c.symm ≫ₕ f ≫ₕ c').symm e.symm (e.target ∩ e.target) := by
        apply EqOn.symm
        refine e.isImage_source_target.symm_eqOn_of_inter_eq_of_eqOn ?_ ?_
        · rw [inter_self, inter_eq_right.mpr h2e]
        · rw [inter_self]; exact hef.symm
      have h2 : e.target ⊆ (c.symm ≫ₕ f ≫ₕ c').target := by
        intro x hx; rw [← e.right_inv hx, ← hef (e.symm.mapsTo hx)]
        exact PartialHomeomorph.mapsTo _ (h2e <| e.symm.mapsTo hx)
      rw [inter_self] at h1
      rwa [inter_eq_right.mpr]
      refine h2.trans ?_
      mfld_set_tac
    /-
      case mp.mk.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      X : M'
      hX : Membership.mem f.symm.source X
      x : M := ↑f.symm X
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      c' : PartialHomeomorph M' H := chartAt H X
      hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
      h2x : Membership.mem (Set.preimage (↑(chartAt H X).symm) f.symm.source) (↑(cha …
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      h2e : HasSubset.Subset e.source (c.symm.trans (f.trans c')).source
      hef : Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑e) e.source
      hex : Membership.mem e.source (↑c x)
      h2X : Eq (↑c' X) (↑e (↑c (↑f.symm X)))
      h3e : Set.EqOn (Function.comp (↑c) (Function.comp ↑f.symm ↑c'.symm)) (↑e.symm) …
      ⊢ Exists fun e => And (Membership.mem (contDiffGroupoid (↑Top.top) I) e) (And  …
    -/
    refine ⟨e.symm, StructureGroupoid.symm _ he, h3e, ?_⟩
    /-
      case mp.mk.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h : ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) I).IsLocalStructomorp …
      X : M'
      hX : Membership.mem f.symm.source X
      x : M := ↑f.symm X
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      c' : PartialHomeomorph M' H := chartAt H X
      hxf : (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.co …
      h2x : Membership.mem (Set.preimage (↑(chartAt H X).symm) f.symm.source) (↑(cha …
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      h2e : HasSubset.Subset e.source (c.symm.trans (f.trans c')).source
      hef : Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑e) e.source
      hex : Membership.mem e.source (↑c x)
      h2X : Eq (↑c' X) (↑e (↑c (↑f.symm X)))
      h3e : Set.EqOn (Function.comp (↑c) (Function.comp ↑f.symm ↑c'.symm)) (↑e.symm) …
      ⊢ Membership.mem e.symm.source (↑(chartAt H X) X)
    -/
    rw [h2X]; exact e.mapsTo hex
              /-
                🎉 no goals
              -/
  · -- We now show the converse: a partial homeomorphism `f : M → M'` which is smooth in both
    -- directions is a local structomorphism.  We do this by proposing
    -- `((chart_at H x).symm.trans f).trans (chart_at H (f x))` as a candidate for a structomorphism
    -- of `H`.
    /-
      case mpr
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      ⊢ And (ContMDiffOn I I Top.top (↑f) f.source) (ContMDiffOn I I Top.top (↑f.sym …
    -/
    rintro ⟨h₁, h₂⟩ x hx
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h₁ : ContMDiffOn I I Top.top (↑f) f.source
      h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
      x : M
      hx : Membership.mem f.source x
      ⊢ ChartedSpace.LiftPropWithinAt (contDiffGroupoid (↑Top.top) I).IsLocalStructo …
    -/
    refine ⟨(h₁ x hx).continuousWithinAt, ?_⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h₁ : ContMDiffOn I I Top.top (↑f) f.source
      h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
      x : M
      hx : Membership.mem f.source x
      ⊢ (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.comp ( …
    -/
    let c := chartAt H x
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h₁ : ContMDiffOn I I Top.top (↑f) f.source
      h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
      x : M
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      ⊢ (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.comp ( …
    -/
    let c' := chartAt H (f x)
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h₁ : ContMDiffOn I I Top.top (↑f) f.source
      h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
      x : M
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      c' : PartialHomeomorph M' H := chartAt H (↑f x)
      ⊢ (contDiffGroupoid (↑Top.top) I).IsLocalStructomorphWithinAt (Function.comp ( …
    -/
    rintro (hx' : c x ∈ c.symm ⁻¹' f.source)
    -- propose `(c.symm.trans f).trans c'` as a candidate for a local structomorphism of `H`
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      M' : Type u_5
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H M'
      IsM' : SmoothManifoldWithCorners I M'
      f : PartialHomeomorph M M'
      h₁ : ContMDiffOn I I Top.top (↑f) f.source
      h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
      x : M
      hx : Membership.mem f.source x
      c : PartialHomeomorph M H := chartAt H x
      c' : PartialHomeomorph M' H := chartAt H (↑f x)
      hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
      ⊢ Exists fun e => And (Membership.mem (contDiffGroupoid (↑Top.top) I) e) (And  …
    -/
    refine ⟨(c.symm.trans f).trans c', ⟨?_, ?_⟩, (?_ : EqOn (c' ∘ f ∘ c.symm) _ _), ?_⟩
    · -- smoothness of the candidate local structomorphism in the forward direction
      /-
        case mpr.intro.refine_1
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H := chartAt H x
        c' : PartialHomeomorph M' H := chartAt H (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        ⊢ (contDiffPregroupoid (↑Top.top) I).property (↑((c.symm.trans f).trans c')) ( …
      -/
      intro y hy
      /-
        case mpr.intro.refine_1
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H := chartAt H x
        c' : PartialHomeomorph M' H := chartAt H (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        y : E
        hy : Membership.mem (Inter.inter (Set.preimage (↑I.symm) ((c.symm.trans f).tra …
        ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((c.symm.t …
      -/
      simp only [mfld_simps] at hy
      have H : ContMDiffWithinAt I I ⊤ f (f ≫ₕ c').source ((extChartAt I x).symm y) := by
        refine (h₁ ((extChartAt I x).symm y) ?_).mono ?_
        · simp only [c, hy, mfld_simps]
        · mfld_set_tac
      /-
        case mpr.intro.refine_1
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H✝ : Type u_3
        inst✝⁵ : TopologicalSpace H✝
        I : ModelWithCorners 𝕜 E H✝
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H✝ M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H✝ M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H✝ := chartAt H✝ x
        c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        y : E
        hy : And (And (And (Membership.mem c.target (↑I.symm y)) (Membership.mem f.sou …
        H : ContMDiffWithinAt I I Top.top (↑f) (f.trans c').source (↑(extChartAt I x). …
        ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((c.symm.t …
      -/
      have hy' : (extChartAt I x).symm y ∈ c.source := by simp only [c, hy, mfld_simps]
      have hy'' : f ((extChartAt I x).symm y) ∈ c'.source := by
        simp only [c, hy, mfld_simps]
      /-
        case mpr.intro.refine_1
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H✝ : Type u_3
        inst✝⁵ : TopologicalSpace H✝
        I : ModelWithCorners 𝕜 E H✝
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H✝ M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H✝ M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H✝ := chartAt H✝ x
        c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        y : E
        hy : And (And (And (Membership.mem c.target (↑I.symm y)) (Membership.mem f.sou …
        H : ContMDiffWithinAt I I Top.top (↑f) (f.trans c').source (↑(extChartAt I x). …
        hy' : Membership.mem c.source (↑(extChartAt I x).symm y)
        hy'' : Membership.mem c'.source (↑f (↑(extChartAt I x).symm y))
        ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((c.symm.t …
      -/
      rw [contMDiffWithinAt_iff_of_mem_source hy' hy''] at H
      /-
        case mpr.intro.refine_1
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H✝ : Type u_3
        inst✝⁵ : TopologicalSpace H✝
        I : ModelWithCorners 𝕜 E H✝
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H✝ M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H✝ M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H✝ := chartAt H✝ x
        c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        y : E
        hy : And (And (And (Membership.mem c.target (↑I.symm y)) (Membership.mem f.sou …
        H : And (ContinuousWithinAt (↑f) (f.trans c').source (↑(extChartAt I x).symm y …
        hy' : Membership.mem c.source (↑(extChartAt I x).symm y)
        hy'' : Membership.mem c'.source (↑f (↑(extChartAt I x).symm y))
        ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((c.symm.t …
      -/
      convert H.2.mono _
        /-
          case h.e'_12
          𝕜 : Type u_1
          inst✝⁸ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          H✝ : Type u_3
          inst✝⁵ : TopologicalSpace H✝
          I : ModelWithCorners 𝕜 E H✝
          M : Type u_4
          inst✝⁴ : TopologicalSpace M
          inst✝³ : ChartedSpace H✝ M
          inst✝² : SmoothManifoldWithCorners I M
          M' : Type u_5
          inst✝¹ : TopologicalSpace M'
          inst✝ : ChartedSpace H✝ M'
          IsM' : SmoothManifoldWithCorners I M'
          f : PartialHomeomorph M M'
          h₁ : ContMDiffOn I I Top.top (↑f) f.source
          h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
          x : M
          hx : Membership.mem f.source x
          c : PartialHomeomorph M H✝ := chartAt H✝ x
          c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
          hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
          y : E
          hy : And (And (And (Membership.mem c.target (↑I.symm y)) (Membership.mem f.sou …
          H : And (ContinuousWithinAt (↑f) (f.trans c').source (↑(extChartAt I x).symm y …
          hy' : Membership.mem c.source (↑(extChartAt I x).symm y)
          hy'' : Membership.mem c'.source (↑f (↑(extChartAt I x).symm y))
          ⊢ Eq y (↑(extChartAt I x) (↑(extChartAt I x).symm y))
        -/
      · simp only [c, hy, mfld_simps]
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.refine_1.convert_2
          𝕜 : Type u_1
          inst✝⁸ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          H✝ : Type u_3
          inst✝⁵ : TopologicalSpace H✝
          I : ModelWithCorners 𝕜 E H✝
          M : Type u_4
          inst✝⁴ : TopologicalSpace M
          inst✝³ : ChartedSpace H✝ M
          inst✝² : SmoothManifoldWithCorners I M
          M' : Type u_5
          inst✝¹ : TopologicalSpace M'
          inst✝ : ChartedSpace H✝ M'
          IsM' : SmoothManifoldWithCorners I M'
          f : PartialHomeomorph M M'
          h₁ : ContMDiffOn I I Top.top (↑f) f.source
          h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
          x : M
          hx : Membership.mem f.source x
          c : PartialHomeomorph M H✝ := chartAt H✝ x
          c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
          hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
          y : E
          hy : And (And (And (Membership.mem c.target (↑I.symm y)) (Membership.mem f.sou …
          H : And (ContinuousWithinAt (↑f) (f.trans c').source (↑(extChartAt I x).symm y …
          hy' : Membership.mem c.source (↑(extChartAt I x).symm y)
          hy'' : Membership.mem c'.source (↑f (↑(extChartAt I x).symm y))
          ⊢ HasSubset.Subset (Inter.inter (Set.preimage (↑I.symm) ((c.symm.trans f).tran …
        -/
      · dsimp [c, c']; mfld_set_tac
                       /-
                         🎉 no goals
                       -/
    · -- smoothness of the candidate local structomorphism in the reverse direction
      /-
        case mpr.intro.refine_2
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H := chartAt H x
        c' : PartialHomeomorph M' H := chartAt H (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        ⊢ (contDiffPregroupoid (↑Top.top) I).property (↑((c.symm.trans f).trans c').sy …
      -/
      intro y hy
      /-
        case mpr.intro.refine_2
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H := chartAt H x
        c' : PartialHomeomorph M' H := chartAt H (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        y : E
        hy : Membership.mem (Inter.inter (Set.preimage (↑I.symm) ((c.symm.trans f).tra …
        ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((c.symm.t …
      -/
      simp only [mfld_simps] at hy
      have H : ContMDiffWithinAt I I ⊤ f.symm (f.symm ≫ₕ c).source
          ((extChartAt I (f x)).symm y) := by
        refine (h₂ ((extChartAt I (f x)).symm y) ?_).mono ?_
        · simp only [c', hy, mfld_simps]
        · mfld_set_tac
      /-
        case mpr.intro.refine_2
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H✝ : Type u_3
        inst✝⁵ : TopologicalSpace H✝
        I : ModelWithCorners 𝕜 E H✝
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H✝ M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H✝ M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H✝ := chartAt H✝ x
        c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        y : E
        hy : And (And (Membership.mem c'.target (↑I.symm y)) (And (Membership.mem f.ta …
        H : ContMDiffWithinAt I I Top.top (↑f.symm) (f.symm.trans c).source (↑(extChar …
        ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((c.symm.t …
      -/
      have hy' : (extChartAt I (f x)).symm y ∈ c'.source := by simp only [c', hy, mfld_simps]
      have hy'' : f.symm ((extChartAt I (f x)).symm y) ∈ c.source := by
        simp only [c', hy, mfld_simps]
      /-
        case mpr.intro.refine_2
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H✝ : Type u_3
        inst✝⁵ : TopologicalSpace H✝
        I : ModelWithCorners 𝕜 E H✝
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H✝ M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H✝ M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H✝ := chartAt H✝ x
        c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        y : E
        hy : And (And (Membership.mem c'.target (↑I.symm y)) (And (Membership.mem f.ta …
        H : ContMDiffWithinAt I I Top.top (↑f.symm) (f.symm.trans c).source (↑(extChar …
        hy' : Membership.mem c'.source (↑(extChartAt I (↑f x)).symm y)
        hy'' : Membership.mem c.source (↑f.symm (↑(extChartAt I (↑f x)).symm y))
        ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((c.symm.t …
      -/
      rw [contMDiffWithinAt_iff_of_mem_source hy' hy''] at H
      /-
        case mpr.intro.refine_2
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H✝ : Type u_3
        inst✝⁵ : TopologicalSpace H✝
        I : ModelWithCorners 𝕜 E H✝
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H✝ M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H✝ M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H✝ := chartAt H✝ x
        c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        y : E
        hy : And (And (Membership.mem c'.target (↑I.symm y)) (And (Membership.mem f.ta …
        H : And (ContinuousWithinAt (↑f.symm) (f.symm.trans c).source (↑(extChartAt I  …
        hy' : Membership.mem c'.source (↑(extChartAt I (↑f x)).symm y)
        hy'' : Membership.mem c.source (↑f.symm (↑(extChartAt I (↑f x)).symm y))
        ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((c.symm.t …
      -/
      convert H.2.mono _
        /-
          case h.e'_12
          𝕜 : Type u_1
          inst✝⁸ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          H✝ : Type u_3
          inst✝⁵ : TopologicalSpace H✝
          I : ModelWithCorners 𝕜 E H✝
          M : Type u_4
          inst✝⁴ : TopologicalSpace M
          inst✝³ : ChartedSpace H✝ M
          inst✝² : SmoothManifoldWithCorners I M
          M' : Type u_5
          inst✝¹ : TopologicalSpace M'
          inst✝ : ChartedSpace H✝ M'
          IsM' : SmoothManifoldWithCorners I M'
          f : PartialHomeomorph M M'
          h₁ : ContMDiffOn I I Top.top (↑f) f.source
          h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
          x : M
          hx : Membership.mem f.source x
          c : PartialHomeomorph M H✝ := chartAt H✝ x
          c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
          hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
          y : E
          hy : And (And (Membership.mem c'.target (↑I.symm y)) (And (Membership.mem f.ta …
          H : And (ContinuousWithinAt (↑f.symm) (f.symm.trans c).source (↑(extChartAt I  …
          hy' : Membership.mem c'.source (↑(extChartAt I (↑f x)).symm y)
          hy'' : Membership.mem c.source (↑f.symm (↑(extChartAt I (↑f x)).symm y))
          ⊢ Eq y (↑(extChartAt I (↑f x)) (↑(extChartAt I (↑f x)).symm y))
        -/
      · simp only [c', hy, mfld_simps]
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.refine_2.convert_2
          𝕜 : Type u_1
          inst✝⁸ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          H✝ : Type u_3
          inst✝⁵ : TopologicalSpace H✝
          I : ModelWithCorners 𝕜 E H✝
          M : Type u_4
          inst✝⁴ : TopologicalSpace M
          inst✝³ : ChartedSpace H✝ M
          inst✝² : SmoothManifoldWithCorners I M
          M' : Type u_5
          inst✝¹ : TopologicalSpace M'
          inst✝ : ChartedSpace H✝ M'
          IsM' : SmoothManifoldWithCorners I M'
          f : PartialHomeomorph M M'
          h₁ : ContMDiffOn I I Top.top (↑f) f.source
          h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
          x : M
          hx : Membership.mem f.source x
          c : PartialHomeomorph M H✝ := chartAt H✝ x
          c' : PartialHomeomorph M' H✝ := chartAt H✝ (↑f x)
          hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
          y : E
          hy : And (And (Membership.mem c'.target (↑I.symm y)) (And (Membership.mem f.ta …
          H : And (ContinuousWithinAt (↑f.symm) (f.symm.trans c).source (↑(extChartAt I  …
          hy' : Membership.mem c'.source (↑(extChartAt I (↑f x)).symm y)
          hy'' : Membership.mem c.source (↑f.symm (↑(extChartAt I (↑f x)).symm y))
          ⊢ HasSubset.Subset (Inter.inter (Set.preimage (↑I.symm) ((c.symm.trans f).tran …
        -/
      · dsimp [c, c']; mfld_set_tac
                       /-
                         🎉 no goals
                       -/
    -- now check the candidate local structomorphism agrees with `f` where it is supposed to
      /-
        case mpr.intro.refine_3
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H := chartAt H x
        c' : PartialHomeomorph M' H := chartAt H (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        ⊢ Set.EqOn (Function.comp (↑c') (Function.comp ↑f ↑c.symm)) (↑((c.symm.trans f …
      -/
    · simp only [mfld_simps]; apply eqOn_refl
                              /-
                                🎉 no goals
                              -/
      /-
        case mpr.intro.refine_4
        𝕜 : Type u_1
        inst✝⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁴ : TopologicalSpace M
        inst✝³ : ChartedSpace H M
        inst✝² : SmoothManifoldWithCorners I M
        M' : Type u_5
        inst✝¹ : TopologicalSpace M'
        inst✝ : ChartedSpace H M'
        IsM' : SmoothManifoldWithCorners I M'
        f : PartialHomeomorph M M'
        h₁ : ContMDiffOn I I Top.top (↑f) f.source
        h₂ : ContMDiffOn I I Top.top (↑f.symm) f.target
        x : M
        hx : Membership.mem f.source x
        c : PartialHomeomorph M H := chartAt H x
        c' : PartialHomeomorph M' H := chartAt H (↑f x)
        hx' : Membership.mem (Set.preimage (↑c.symm) f.source) (↑c x)
        ⊢ Membership.mem ((c.symm.trans f).trans c').source (↑(chartAt H x) x)
      -/
    · simp only [c, c', hx', mfld_simps]
      /-
        🎉 no goals
      -/


