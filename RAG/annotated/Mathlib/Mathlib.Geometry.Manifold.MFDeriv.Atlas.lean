protected theorem hasMFDerivAt {x} : HasMFDerivAt I 𝓘(𝕜, E) I x (ContinuousLinearMap.id _ _) :=
  ⟨I.continuousAt, (hasFDerivWithinAt_id _ _).congr' I.rightInvOn (mem_range_self _)⟩


protected theorem hasMFDerivWithinAt {s x} :
    HasMFDerivWithinAt I 𝓘(𝕜, E) I s x (ContinuousLinearMap.id _ _) :=
  I.hasMFDerivAt.hasMFDerivWithinAt


protected theorem mdifferentiableWithinAt {s x} : MDifferentiableWithinAt I 𝓘(𝕜, E) I s x :=
  I.hasMFDerivWithinAt.mdifferentiableWithinAt


protected theorem mdifferentiableAt {x} : MDifferentiableAt I 𝓘(𝕜, E) I x :=
  I.hasMFDerivAt.mdifferentiableAt


protected theorem mdifferentiableOn {s} : MDifferentiableOn I 𝓘(𝕜, E) I s := fun _ _ =>
  I.mdifferentiableWithinAt


protected theorem mdifferentiable : MDifferentiable I 𝓘(𝕜, E) I := fun _ => I.mdifferentiableAt


theorem hasMFDerivWithinAt_symm {x} (hx : x ∈ range I) :
    HasMFDerivWithinAt 𝓘(𝕜, E) I I.symm (range I) x (ContinuousLinearMap.id _ _) :=
  ⟨I.continuousWithinAt_symm,
    (hasFDerivWithinAt_id _ _).congr' (fun _y hy => I.rightInvOn hy.1) ⟨hx, mem_range_self _⟩⟩


theorem mdifferentiableOn_symm : MDifferentiableOn 𝓘(𝕜, E) I I.symm (range I) := fun _x hx =>
  (I.hasMFDerivWithinAt_symm hx).mdifferentiableWithinAt


theorem mdifferentiableWithinAt_symm {z : E} (hz : z ∈ range I) :
    MDifferentiableWithinAt 𝓘(𝕜, E) I I.symm (range I) z :=
  I.mdifferentiableOn_symm z hz


theorem mdifferentiableAt_atlas (h : e ∈ atlas H M) {x : M} (hx : x ∈ e.source) :
    MDifferentiableAt I I e x := by
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
    h : Membership.mem (atlas H M) e
    x : M
    hx : Membership.mem e.source x
    ⊢ MDifferentiableAt I I (↑e) x
  -/
  rw [mdifferentiableAt_iff]
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
    h : Membership.mem (atlas H M) e
    x : M
    hx : Membership.mem e.source x
    ⊢ And (ContinuousAt (↑e) x) (DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I …
  -/
  refine ⟨(e.continuousOn x hx).continuousAt (e.open_source.mem_nhds hx), ?_⟩
  have mem :
    I ((chartAt H x : M → H) x) ∈ I.symm ⁻¹' ((chartAt H x).symm ≫ₕ e).source ∩ range I := by
    simp only [hx, mfld_simps]
  have : (chartAt H x).symm.trans e ∈ contDiffGroupoid ∞ I :=
    HasGroupoid.compatible (chart_mem_atlas H x) h
  have A :
    ContDiffOn 𝕜 ∞ (I ∘ (chartAt H x).symm.trans e ∘ I.symm)
      (I.symm ⁻¹' ((chartAt H x).symm.trans e).source ∩ range I) :=
    this.1
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
    h : Membership.mem (atlas H M) e
    x : M
    hx : Membership.mem e.source x
    mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) ((chartAt H x).symm. …
    this : Membership.mem (contDiffGroupoid (↑Top.top) I) ((chartAt H x).symm.tran …
    A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((chartAt H x) …
    ⊢ DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I x ↑e) (Set.range ↑I) (↑(ex …
  -/
  have B := A.differentiableOn (mod_cast le_top) (I ((chartAt H x : M → H) x)) mem
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
    h : Membership.mem (atlas H M) e
    x : M
    hx : Membership.mem e.source x
    mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) ((chartAt H x).symm. …
    this : Membership.mem (contDiffGroupoid (↑Top.top) I) ((chartAt H x).symm.tran …
    A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((chartAt H x) …
    B : DifferentiableWithinAt 𝕜 (Function.comp (↑I) (Function.comp ↑((chartAt H x …
    ⊢ DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I x ↑e) (Set.range ↑I) (↑(ex …
  -/
  simp only [mfld_simps] at B
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
    h : Membership.mem (atlas H M) e
    x : M
    hx : Membership.mem e.source x
    mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) ((chartAt H x).symm. …
    this : Membership.mem (contDiffGroupoid (↑Top.top) I) ((chartAt H x).symm.tran …
    A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((chartAt H x) …
    B : DifferentiableWithinAt 𝕜 (Function.comp (↑I) (Function.comp (Function.comp …
    ⊢ DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I x ↑e) (Set.range ↑I) (↑(ex …
  -/
  rw [inter_comm, differentiableWithinAt_inter] at B
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
      h : Membership.mem (atlas H M) e
      x : M
      hx : Membership.mem e.source x
      mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) ((chartAt H x).symm. …
      this : Membership.mem (contDiffGroupoid (↑Top.top) I) ((chartAt H x).symm.tran …
      A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((chartAt H x) …
      B : DifferentiableWithinAt 𝕜 (Function.comp (↑I) (Function.comp (Function.comp …
      ⊢ DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I x ↑e) (Set.range ↑I) (↑(ex …
    -/
  · simpa only [mfld_simps]
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
      e : PartialHomeomorph M H
      h : Membership.mem (atlas H M) e
      x : M
      hx : Membership.mem e.source x
      mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) ((chartAt H x).symm. …
      this : Membership.mem (contDiffGroupoid (↑Top.top) I) ((chartAt H x).symm.tran …
      A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑((chartAt H x) …
      B : DifferentiableWithinAt 𝕜 (Function.comp (↑I) (Function.comp (Function.comp …
      ⊢ Membership.mem (nhds (↑I (↑(chartAt H x) x))) (Inter.inter (Set.preimage (↑I …
    -/
  · apply IsOpen.mem_nhds ((PartialHomeomorph.open_source _).preimage I.continuous_symm) mem.1
    /-
      🎉 no goals
    -/


theorem mdifferentiableOn_atlas (h : e ∈ atlas H M) : MDifferentiableOn I I e e.source :=
  fun _x hx => (mdifferentiableAt_atlas h hx).mdifferentiableWithinAt


theorem mdifferentiableAt_atlas_symm (h : e ∈ atlas H M) {x : H} (hx : x ∈ e.target) :
    MDifferentiableAt I I e.symm x := by
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
    h : Membership.mem (atlas H M) e
    x : H
    hx : Membership.mem e.target x
    ⊢ MDifferentiableAt I I (↑e.symm) x
  -/
  rw [mdifferentiableAt_iff]
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
    h : Membership.mem (atlas H M) e
    x : H
    hx : Membership.mem e.target x
    ⊢ And (ContinuousAt (↑e.symm) x) (DifferentiableWithinAt 𝕜 (writtenInExtChartA …
  -/
  refine ⟨(e.continuousOn_symm x hx).continuousAt (e.open_target.mem_nhds hx), ?_⟩
  have mem : I x ∈ I.symm ⁻¹' (e.symm ≫ₕ chartAt H (e.symm x)).source ∩ range I := by
    simp only [hx, mfld_simps]
  have : e.symm.trans (chartAt H (e.symm x)) ∈ contDiffGroupoid ∞ I :=
    HasGroupoid.compatible h (chart_mem_atlas H _)
  have A :
    ContDiffOn 𝕜 ∞ (I ∘ e.symm.trans (chartAt H (e.symm x)) ∘ I.symm)
      (I.symm ⁻¹' (e.symm.trans (chartAt H (e.symm x))).source ∩ range I) :=
    this.1
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
    h : Membership.mem (atlas H M) e
    x : H
    hx : Membership.mem e.target x
    mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) (e.symm.trans (chart …
    this : Membership.mem (contDiffGroupoid (↑Top.top) I) (e.symm.trans (chartAt H …
    A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑(e.symm.trans  …
    ⊢ DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I x ↑e.symm) (Set.range ↑I)  …
  -/
  have B := A.differentiableOn (mod_cast le_top) (I x) mem
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
    h : Membership.mem (atlas H M) e
    x : H
    hx : Membership.mem e.target x
    mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) (e.symm.trans (chart …
    this : Membership.mem (contDiffGroupoid (↑Top.top) I) (e.symm.trans (chartAt H …
    A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑(e.symm.trans  …
    B : DifferentiableWithinAt 𝕜 (Function.comp (↑I) (Function.comp ↑(e.symm.trans …
    ⊢ DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I x ↑e.symm) (Set.range ↑I)  …
  -/
  simp only [mfld_simps] at B
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
    h : Membership.mem (atlas H M) e
    x : H
    hx : Membership.mem e.target x
    mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) (e.symm.trans (chart …
    this : Membership.mem (contDiffGroupoid (↑Top.top) I) (e.symm.trans (chartAt H …
    A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑(e.symm.trans  …
    B : DifferentiableWithinAt 𝕜 (Function.comp (↑I) (Function.comp (Function.comp …
    ⊢ DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I x ↑e.symm) (Set.range ↑I)  …
  -/
  rw [inter_comm, differentiableWithinAt_inter] at B
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
      h : Membership.mem (atlas H M) e
      x : H
      hx : Membership.mem e.target x
      mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) (e.symm.trans (chart …
      this : Membership.mem (contDiffGroupoid (↑Top.top) I) (e.symm.trans (chartAt H …
      A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑(e.symm.trans  …
      B : DifferentiableWithinAt 𝕜 (Function.comp (↑I) (Function.comp (Function.comp …
      ⊢ DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I x ↑e.symm) (Set.range ↑I)  …
    -/
  · simpa only [mfld_simps]
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
      e : PartialHomeomorph M H
      h : Membership.mem (atlas H M) e
      x : H
      hx : Membership.mem e.target x
      mem : Membership.mem (Inter.inter (Set.preimage (↑I.symm) (e.symm.trans (chart …
      this : Membership.mem (contDiffGroupoid (↑Top.top) I) (e.symm.trans (chartAt H …
      A : ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑(e.symm.trans  …
      B : DifferentiableWithinAt 𝕜 (Function.comp (↑I) (Function.comp (Function.comp …
      ⊢ Membership.mem (nhds (↑I x)) (Inter.inter (Set.preimage (↑I.symm) e.target)  …
    -/
  · apply IsOpen.mem_nhds ((PartialHomeomorph.open_source _).preimage I.continuous_symm) mem.1
    /-
      🎉 no goals
    -/


theorem mdifferentiableOn_atlas_symm (h : e ∈ atlas H M) : MDifferentiableOn I I e.symm e.target :=
  fun _x hx => (mdifferentiableAt_atlas_symm h hx).mdifferentiableWithinAt


theorem mdifferentiable_of_mem_atlas (h : e ∈ atlas H M) : e.MDifferentiable I I :=
  ⟨mdifferentiableOn_atlas h, mdifferentiableOn_atlas_symm h⟩


theorem mdifferentiable_chart (x : M) : (chartAt H x).MDifferentiable I I :=
  mdifferentiable_of_mem_atlas (chart_mem_atlas _ _)


nonrec theorem symm : e.symm.MDifferentiable I' I := he.symm


protected theorem mdifferentiableAt {x : M} (hx : x ∈ e.source) : MDifferentiableAt I I' e x :=
  (he.1 x hx).mdifferentiableAt (e.open_source.mem_nhds hx)


theorem mdifferentiableAt_symm {x : M'} (hx : x ∈ e.target) : MDifferentiableAt I' I e.symm x :=
  (he.2 x hx).mdifferentiableAt (e.open_target.mem_nhds hx)


theorem symm_comp_deriv {x : M} (hx : x ∈ e.source) :
    (mfderiv I' I e.symm (e x)).comp (mfderiv I I' e x) =
      ContinuousLinearMap.id 𝕜 (TangentSpace I x) := by
  have : mfderiv I I (e.symm ∘ e) x = (mfderiv I' I e.symm (e x)).comp (mfderiv I I' e x) :=
    mfderiv_comp x (he.mdifferentiableAt_symm (e.map_source hx)) (he.mdifferentiableAt hx)
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    e : PartialHomeomorph M M'
    he : PartialHomeomorph.MDifferentiable I I' e
    x : M
    hx : Membership.mem e.source x
    this : Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) ((mfderiv I' I (↑e.symm)  …
    ⊢ Eq ((mfderiv I' I (↑e.symm) (↑e x)).comp (mfderiv I I' (↑e) x)) (ContinuousL …
  -/
  rw [← this]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    e : PartialHomeomorph M M'
    he : PartialHomeomorph.MDifferentiable I I' e
    x : M
    hx : Membership.mem e.source x
    this : Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) ((mfderiv I' I (↑e.symm)  …
    ⊢ Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) (ContinuousLinearMap.id 𝕜 (Tan …
  -/
  have : mfderiv I I (_root_.id : M → M) x = ContinuousLinearMap.id _ _ := mfderiv_id
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    e : PartialHomeomorph M M'
    he : PartialHomeomorph.MDifferentiable I I' e
    x : M
    hx : Membership.mem e.source x
    this✝ : Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) ((mfderiv I' I (↑e.symm) …
    this : Eq (mfderiv I I id x) (ContinuousLinearMap.id 𝕜 (TangentSpace I x))
    ⊢ Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) (ContinuousLinearMap.id 𝕜 (Tan …
  -/
  rw [← this]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    e : PartialHomeomorph M M'
    he : PartialHomeomorph.MDifferentiable I I' e
    x : M
    hx : Membership.mem e.source x
    this✝ : Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) ((mfderiv I' I (↑e.symm) …
    this : Eq (mfderiv I I id x) (ContinuousLinearMap.id 𝕜 (TangentSpace I x))
    ⊢ Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) (mfderiv I I id x)
  -/
  apply Filter.EventuallyEq.mfderiv_eq
  /-
    case hL
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    e : PartialHomeomorph M M'
    he : PartialHomeomorph.MDifferentiable I I' e
    x : M
    hx : Membership.mem e.source x
    this✝ : Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) ((mfderiv I' I (↑e.symm) …
    this : Eq (mfderiv I I id x) (ContinuousLinearMap.id 𝕜 (TangentSpace I x))
    ⊢ (nhds x).EventuallyEq (Function.comp ↑e.symm ↑e) id
  -/
  have : e.source ∈ 𝓝 x := e.open_source.mem_nhds hx
  /-
    case hL
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    e : PartialHomeomorph M M'
    he : PartialHomeomorph.MDifferentiable I I' e
    x : M
    hx : Membership.mem e.source x
    this✝¹ : Eq (mfderiv I I (Function.comp ↑e.symm ↑e) x) ((mfderiv I' I (↑e.symm …
    this✝ : Eq (mfderiv I I id x) (ContinuousLinearMap.id 𝕜 (TangentSpace I x))
    this : Membership.mem (nhds x) e.source
    ⊢ (nhds x).EventuallyEq (Function.comp ↑e.symm ↑e) id
  -/
  exact Filter.mem_of_superset this (by mfld_set_tac)
  /-
    🎉 no goals
  -/


theorem comp_symm_deriv {x : M'} (hx : x ∈ e.target) :
    (mfderiv I I' e (e.symm x)).comp (mfderiv I' I e.symm x) =
      ContinuousLinearMap.id 𝕜 (TangentSpace I' x) :=
  he.symm.symm_comp_deriv hx


/-- The derivative of a differentiable partial homeomorphism, as a continuous linear equivalence
between the tangent spaces at `x` and `e x`. -/
protected def mfderiv (he : e.MDifferentiable I I') {x : M} (hx : x ∈ e.source) :
    TangentSpace I x ≃L[𝕜] TangentSpace I' (e x) :=
  { mfderiv I I' e x with
    invFun := mfderiv I' I e.symm (e x)
    continuous_toFun := (mfderiv I I' e x).cont
    continuous_invFun := (mfderiv I' I e.symm (e x)).cont
    left_inv := fun y => by
      /-
        𝕜 : Type u_1
        inst✝¹⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹⁴ : NormedAddCommGroup E
        inst✝¹³ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹² : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝¹¹ : TopologicalSpace M
        inst✝¹⁰ : ChartedSpace H M
        E' : Type u_5
        inst✝⁹ : NormedAddCommGroup E'
        inst✝⁸ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁷ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : ChartedSpace H' M'
        E'' : Type u_8
        inst✝⁴ : NormedAddCommGroup E''
        inst✝³ : NormedSpace 𝕜 E''
        H'' : Type u_9
        inst✝² : TopologicalSpace H''
        I'' : ModelWithCorners 𝕜 E'' H''
        M'' : Type u_10
        inst✝¹ : TopologicalSpace M''
        inst✝ : ChartedSpace H'' M''
        e : PartialHomeomorph M M'
        he✝ : PartialHomeomorph.MDifferentiable I I' e
        e' : PartialHomeomorph M' M''
        he : PartialHomeomorph.MDifferentiable I I' e
        x : M
        hx : Membership.mem e.source x
        y : TangentSpace I x
        ⊢ Eq ((mfderiv I' I (↑e.symm) (↑e x)) ((↑__src✝).toFun y)) y
      -/
      have : (ContinuousLinearMap.id _ _ : TangentSpace I x →L[𝕜] TangentSpace I x) y = y := rfl
      /-
        𝕜 : Type u_1
        inst✝¹⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹⁴ : NormedAddCommGroup E
        inst✝¹³ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹² : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝¹¹ : TopologicalSpace M
        inst✝¹⁰ : ChartedSpace H M
        E' : Type u_5
        inst✝⁹ : NormedAddCommGroup E'
        inst✝⁸ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁷ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : ChartedSpace H' M'
        E'' : Type u_8
        inst✝⁴ : NormedAddCommGroup E''
        inst✝³ : NormedSpace 𝕜 E''
        H'' : Type u_9
        inst✝² : TopologicalSpace H''
        I'' : ModelWithCorners 𝕜 E'' H''
        M'' : Type u_10
        inst✝¹ : TopologicalSpace M''
        inst✝ : ChartedSpace H'' M''
        e : PartialHomeomorph M M'
        he✝ : PartialHomeomorph.MDifferentiable I I' e
        e' : PartialHomeomorph M' M''
        he : PartialHomeomorph.MDifferentiable I I' e
        x : M
        hx : Membership.mem e.source x
        y : TangentSpace I x
        this : Eq ((ContinuousLinearMap.id 𝕜 (TangentSpace I x)) y) y
        ⊢ Eq ((mfderiv I' I (↑e.symm) (↑e x)) ((↑__src✝).toFun y)) y
      -/
      conv_rhs => rw [← this, ← he.symm_comp_deriv hx]
      /-
        𝕜 : Type u_1
        inst✝¹⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹⁴ : NormedAddCommGroup E
        inst✝¹³ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹² : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝¹¹ : TopologicalSpace M
        inst✝¹⁰ : ChartedSpace H M
        E' : Type u_5
        inst✝⁹ : NormedAddCommGroup E'
        inst✝⁸ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁷ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : ChartedSpace H' M'
        E'' : Type u_8
        inst✝⁴ : NormedAddCommGroup E''
        inst✝³ : NormedSpace 𝕜 E''
        H'' : Type u_9
        inst✝² : TopologicalSpace H''
        I'' : ModelWithCorners 𝕜 E'' H''
        M'' : Type u_10
        inst✝¹ : TopologicalSpace M''
        inst✝ : ChartedSpace H'' M''
        e : PartialHomeomorph M M'
        he✝ : PartialHomeomorph.MDifferentiable I I' e
        e' : PartialHomeomorph M' M''
        he : PartialHomeomorph.MDifferentiable I I' e
        x : M
        hx : Membership.mem e.source x
        y : TangentSpace I x
        this : Eq ((ContinuousLinearMap.id 𝕜 (TangentSpace I x)) y) y
        ⊢ Eq ((mfderiv I' I (↑e.symm) (↑e x)) ((↑__src✝).toFun y)) (((mfderiv I' I (↑e …
      -/
      rfl
      /-
        🎉 no goals
      -/
    right_inv := fun y => by
      have :
        (ContinuousLinearMap.id 𝕜 _ : TangentSpace I' (e x) →L[𝕜] TangentSpace I' (e x)) y = y :=
        rfl
      /-
        𝕜 : Type u_1
        inst✝¹⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹⁴ : NormedAddCommGroup E
        inst✝¹³ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹² : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝¹¹ : TopologicalSpace M
        inst✝¹⁰ : ChartedSpace H M
        E' : Type u_5
        inst✝⁹ : NormedAddCommGroup E'
        inst✝⁸ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁷ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : ChartedSpace H' M'
        E'' : Type u_8
        inst✝⁴ : NormedAddCommGroup E''
        inst✝³ : NormedSpace 𝕜 E''
        H'' : Type u_9
        inst✝² : TopologicalSpace H''
        I'' : ModelWithCorners 𝕜 E'' H''
        M'' : Type u_10
        inst✝¹ : TopologicalSpace M''
        inst✝ : ChartedSpace H'' M''
        e : PartialHomeomorph M M'
        he✝ : PartialHomeomorph.MDifferentiable I I' e
        e' : PartialHomeomorph M' M''
        he : PartialHomeomorph.MDifferentiable I I' e
        x : M
        hx : Membership.mem e.source x
        y : TangentSpace I' (↑e x)
        this : Eq ((ContinuousLinearMap.id 𝕜 (TangentSpace I' (↑e x))) y) y
        ⊢ Eq ((↑__src✝).toFun ((mfderiv I' I (↑e.symm) (↑e x)) y)) y
      -/
      conv_rhs => rw [← this, ← he.comp_symm_deriv (e.map_source hx)]
      /-
        𝕜 : Type u_1
        inst✝¹⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹⁴ : NormedAddCommGroup E
        inst✝¹³ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹² : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝¹¹ : TopologicalSpace M
        inst✝¹⁰ : ChartedSpace H M
        E' : Type u_5
        inst✝⁹ : NormedAddCommGroup E'
        inst✝⁸ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁷ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : ChartedSpace H' M'
        E'' : Type u_8
        inst✝⁴ : NormedAddCommGroup E''
        inst✝³ : NormedSpace 𝕜 E''
        H'' : Type u_9
        inst✝² : TopologicalSpace H''
        I'' : ModelWithCorners 𝕜 E'' H''
        M'' : Type u_10
        inst✝¹ : TopologicalSpace M''
        inst✝ : ChartedSpace H'' M''
        e : PartialHomeomorph M M'
        he✝ : PartialHomeomorph.MDifferentiable I I' e
        e' : PartialHomeomorph M' M''
        he : PartialHomeomorph.MDifferentiable I I' e
        x : M
        hx : Membership.mem e.source x
        y : TangentSpace I' (↑e x)
        this : Eq ((ContinuousLinearMap.id 𝕜 (TangentSpace I' (↑e x))) y) y
        ⊢ Eq ((↑__src✝).toFun ((mfderiv I' I (↑e.symm) (↑e x)) y)) (((mfderiv I I' (↑e …
      -/
      rw [e.left_inv hx]
      /-
        𝕜 : Type u_1
        inst✝¹⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹⁴ : NormedAddCommGroup E
        inst✝¹³ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝¹² : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝¹¹ : TopologicalSpace M
        inst✝¹⁰ : ChartedSpace H M
        E' : Type u_5
        inst✝⁹ : NormedAddCommGroup E'
        inst✝⁸ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁷ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝⁶ : TopologicalSpace M'
        inst✝⁵ : ChartedSpace H' M'
        E'' : Type u_8
        inst✝⁴ : NormedAddCommGroup E''
        inst✝³ : NormedSpace 𝕜 E''
        H'' : Type u_9
        inst✝² : TopologicalSpace H''
        I'' : ModelWithCorners 𝕜 E'' H''
        M'' : Type u_10
        inst✝¹ : TopologicalSpace M''
        inst✝ : ChartedSpace H'' M''
        e : PartialHomeomorph M M'
        he✝ : PartialHomeomorph.MDifferentiable I I' e
        e' : PartialHomeomorph M' M''
        he : PartialHomeomorph.MDifferentiable I I' e
        x : M
        hx : Membership.mem e.source x
        y : TangentSpace I' (↑e x)
        this : Eq ((ContinuousLinearMap.id 𝕜 (TangentSpace I' (↑e x))) y) y
        ⊢ Eq ((↑__src✝).toFun ((mfderiv I' I (↑e.symm) (↑e x)) y)) (((mfderiv I I' (↑e …
      -/
      rfl }
      /-
        🎉 no goals
      -/


theorem mfderiv_bijective {x : M} (hx : x ∈ e.source) : Function.Bijective (mfderiv I I' e x) :=
  (he.mfderiv hx).bijective


theorem mfderiv_injective {x : M} (hx : x ∈ e.source) : Function.Injective (mfderiv I I' e x) :=
  (he.mfderiv hx).injective


theorem mfderiv_surjective {x : M} (hx : x ∈ e.source) : Function.Surjective (mfderiv I I' e x) :=
  (he.mfderiv hx).surjective


theorem ker_mfderiv_eq_bot {x : M} (hx : x ∈ e.source) : LinearMap.ker (mfderiv I I' e x) = ⊥ :=
  (he.mfderiv hx).toLinearEquiv.ker


theorem range_mfderiv_eq_top {x : M} (hx : x ∈ e.source) : LinearMap.range (mfderiv I I' e x) = ⊤ :=
  (he.mfderiv hx).toLinearEquiv.range


theorem range_mfderiv_eq_univ {x : M} (hx : x ∈ e.source) : range (mfderiv I I' e x) = univ :=
  (he.mfderiv_surjective hx).range_eq


theorem trans (he' : e'.MDifferentiable I' I'') : (e.trans e').MDifferentiable I I'' := by
  /-
    𝕜 : Type u_1
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : ChartedSpace H M
    E' : Type u_5
    inst✝⁹ : NormedAddCommGroup E'
    inst✝⁸ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁷ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁶ : TopologicalSpace M'
    inst✝⁵ : ChartedSpace H' M'
    E'' : Type u_8
    inst✝⁴ : NormedAddCommGroup E''
    inst✝³ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝² : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝¹ : TopologicalSpace M''
    inst✝ : ChartedSpace H'' M''
    e : PartialHomeomorph M M'
    he : PartialHomeomorph.MDifferentiable I I' e
    e' : PartialHomeomorph M' M''
    he' : PartialHomeomorph.MDifferentiable I' I'' e'
    ⊢ PartialHomeomorph.MDifferentiable I I'' (e.trans e')
  -/
  constructor
    /-
      case left
      𝕜 : Type u_1
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace H M
      E' : Type u_5
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁷ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁶ : TopologicalSpace M'
      inst✝⁵ : ChartedSpace H' M'
      E'' : Type u_8
      inst✝⁴ : NormedAddCommGroup E''
      inst✝³ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝² : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝¹ : TopologicalSpace M''
      inst✝ : ChartedSpace H'' M''
      e : PartialHomeomorph M M'
      he : PartialHomeomorph.MDifferentiable I I' e
      e' : PartialHomeomorph M' M''
      he' : PartialHomeomorph.MDifferentiable I' I'' e'
      ⊢ MDifferentiableOn I I'' (↑(e.trans e')) (e.trans e').source
    -/
  · intro x hx
    /-
      case left
      𝕜 : Type u_1
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace H M
      E' : Type u_5
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁷ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁶ : TopologicalSpace M'
      inst✝⁵ : ChartedSpace H' M'
      E'' : Type u_8
      inst✝⁴ : NormedAddCommGroup E''
      inst✝³ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝² : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝¹ : TopologicalSpace M''
      inst✝ : ChartedSpace H'' M''
      e : PartialHomeomorph M M'
      he : PartialHomeomorph.MDifferentiable I I' e
      e' : PartialHomeomorph M' M''
      he' : PartialHomeomorph.MDifferentiable I' I'' e'
      x : M
      hx : Membership.mem (e.trans e').source x
      ⊢ MDifferentiableWithinAt I I'' (↑(e.trans e')) (e.trans e').source x
    -/
    simp only [mfld_simps] at hx
    exact
      ((he'.mdifferentiableAt hx.2).comp _ (he.mdifferentiableAt hx.1)).mdifferentiableWithinAt
    /-
      case right
      𝕜 : Type u_1
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace H M
      E' : Type u_5
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁷ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁶ : TopologicalSpace M'
      inst✝⁵ : ChartedSpace H' M'
      E'' : Type u_8
      inst✝⁴ : NormedAddCommGroup E''
      inst✝³ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝² : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝¹ : TopologicalSpace M''
      inst✝ : ChartedSpace H'' M''
      e : PartialHomeomorph M M'
      he : PartialHomeomorph.MDifferentiable I I' e
      e' : PartialHomeomorph M' M''
      he' : PartialHomeomorph.MDifferentiable I' I'' e'
      ⊢ MDifferentiableOn I'' I (↑(e.trans e').symm) (e.trans e').target
    -/
  · intro x hx
    /-
      case right
      𝕜 : Type u_1
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace H M
      E' : Type u_5
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁷ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁶ : TopologicalSpace M'
      inst✝⁵ : ChartedSpace H' M'
      E'' : Type u_8
      inst✝⁴ : NormedAddCommGroup E''
      inst✝³ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝² : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝¹ : TopologicalSpace M''
      inst✝ : ChartedSpace H'' M''
      e : PartialHomeomorph M M'
      he : PartialHomeomorph.MDifferentiable I I' e
      e' : PartialHomeomorph M' M''
      he' : PartialHomeomorph.MDifferentiable I' I'' e'
      x : M''
      hx : Membership.mem (e.trans e').target x
      ⊢ MDifferentiableWithinAt I'' I (↑(e.trans e').symm) (e.trans e').target x
    -/
    simp only [mfld_simps] at hx
    exact
      ((he.symm.mdifferentiableAt hx.2).comp _
          (he'.symm.mdifferentiableAt hx.1)).mdifferentiableWithinAt


theorem hasMFDerivAt_extChartAt (h : y ∈ (chartAt H x).source) :
    HasMFDerivAt I 𝓘(𝕜, E) (extChartAt I x) y (mfderiv I I (chartAt H x) y : _) :=
  I.hasMFDerivAt.comp y ((mdifferentiable_chart x).mdifferentiableAt h).hasMFDerivAt


theorem hasMFDerivWithinAt_extChartAt (h : y ∈ (chartAt H x).source) :
    HasMFDerivWithinAt I 𝓘(𝕜, E) (extChartAt I x) s y (mfderiv I I (chartAt H x) y : _) :=
  (hasMFDerivAt_extChartAt h).hasMFDerivWithinAt


theorem mdifferentiableAt_extChartAt (h : y ∈ (chartAt H x).source) :
    MDifferentiableAt I 𝓘(𝕜, E) (extChartAt I x) y :=
  (hasMFDerivAt_extChartAt h).mdifferentiableAt


theorem mdifferentiableOn_extChartAt :
    MDifferentiableOn I 𝓘(𝕜, E) (extChartAt I x) (chartAt H x).source := fun _y hy =>
  (hasMFDerivWithinAt_extChartAt hy).mdifferentiableWithinAt


theorem mdifferentiableWithinAt_extChartAt_symm (h : z ∈ (extChartAt I x).target) :
    MDifferentiableWithinAt 𝓘(𝕜, E) I (extChartAt I x).symm (range I) z := by
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
    x : M
    z : E
    h : Membership.mem (extChartAt I x).target z
    ⊢ MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm …
  -/
  have Z := I.mdifferentiableWithinAt_symm (extChartAt_target_subset_range x h)
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
    x : M
    z : E
    h : Membership.mem (extChartAt I x).target z
    Z : MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑I.symm) (Set.range  …
    ⊢ MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm …
  -/
  apply MDifferentiableAt.comp_mdifferentiableWithinAt (I' := I) _ _ Z
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
    x : M
    z : E
    h : Membership.mem (extChartAt I x).target z
    Z : MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑I.symm) (Set.range  …
    ⊢ MDifferentiableAt I I (chartAt H x).toPartialEquiv.2 (↑I.symm z)
  -/
  apply mdifferentiableAt_atlas_symm (ChartedSpace.chart_mem_atlas x)
  simp only [extChartAt, PartialHomeomorph.extend, PartialEquiv.trans_target,
    ModelWithCorners.target_eq, ModelWithCorners.toPartialEquiv_coe_symm, mem_inter_iff, mem_range,
    mem_preimage] at h
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
    x : M
    z : E
    Z : MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑I.symm) (Set.range  …
    h : And (Exists fun y => Eq (↑I y) z) (Membership.mem (chartAt H x).target (↑I …
    ⊢ Membership.mem (ChartedSpace.chartAt x).target (↑I.symm z)
  -/
  exact h.2
  /-
    🎉 no goals
  -/


theorem mdifferentiableOn_extChartAt_symm :
    MDifferentiableOn 𝓘(𝕜, E) I (extChartAt I x).symm (extChartAt I x).target := by
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
    x : M
    ⊢ MDifferentiableOn (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm) (ext …
  -/
  intro y hy
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm …
  -/
  exact (mdifferentiableWithinAt_extChartAt_symm hy).mono (extChartAt_target_subset_range x)
  /-
    🎉 no goals
  -/


/-- The composition of the derivative of `extChartAt` with the derivative of the inverse of
`extChartAt` gives the identity.
Version where the basepoint belongs to `(extChartAt I x).target`. -/
lemma mfderiv_extChartAt_comp_mfderivWithin_extChartAt_symm {x : M}
    {y : E} (hy : y ∈ (extChartAt I x).target) :
    (mfderiv I 𝓘(𝕜, E) (extChartAt I x) ((extChartAt I x).symm y)) ∘L
      (mfderivWithin 𝓘(𝕜, E) I (extChartAt I x).symm (range I) y) = ContinuousLinearMap.id _ _ := by
  have U : UniqueMDiffWithinAt 𝓘(𝕜, E) (range ↑I) y := by
    apply I.uniqueMDiffOn
    exact extChartAt_target_subset_range x hy
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
    ⊢ Eq ((mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChartAt  …
  -/
  have h'y : (extChartAt I x).symm y ∈ (extChartAt I x).source := (extChartAt I x).map_target hy
  have h''y : (extChartAt I x).symm y ∈ (chartAt H x).source := by
    rwa [← extChartAt_source (I := I)]
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
    h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
    h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
    ⊢ Eq ((mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChartAt  …
  -/
  rw [← mfderiv_comp_mfderivWithin]; rotate_left
    /-
      case hg
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      ⊢ MDifferentiableAt I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extCha …
    -/
  · apply mdifferentiableAt_extChartAt h''y
    /-
      🎉 no goals
    -/
    /-
      case hf
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      ⊢ MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm …
    -/
  · exact mdifferentiableWithinAt_extChartAt_symm hy
    /-
      🎉 no goals
    -/
    /-
      case hxs
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      ⊢ UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
    -/
  · exact U
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
    h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
    h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
    ⊢ Eq (mfderivWithin (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E) (Fun …
  -/
  rw [← mfderivWithin_id U]
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
    h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
    h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
    ⊢ Eq (mfderivWithin (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E) (Fun …
  -/
  apply Filter.EventuallyEq.mfderivWithin_eq U
    /-
      case hL
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      ⊢ (nhdsWithin y (Set.range ↑I)).EventuallyEq (Function.comp ↑(extChartAt I x)  …
    -/
  · filter_upwards [extChartAt_target_mem_nhdsWithin_of_mem hy] with z hz
    /-
      case h
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      z : E
      hz : Membership.mem (extChartAt I x).target z
      ⊢ Eq (Function.comp (↑(extChartAt I x)) (↑(extChartAt I x).symm) z) (id z)
    -/
    simp only [Function.comp_def, PartialEquiv.right_inv (extChartAt I x) hz, id_eq]
    /-
      🎉 no goals
    -/
    /-
      case hx
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      U : UniqueMDiffWithinAt (modelWithCornersSelf 𝕜 E) (Set.range ↑I) y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      ⊢ Eq (Function.comp (↑(extChartAt I x)) (↑(extChartAt I x).symm) y) (id y)
    -/
  · simp only [Function.comp_def, PartialEquiv.right_inv (extChartAt I x) hy, id_eq]
    /-
      🎉 no goals
    -/


/-- The composition of the derivative of `extChartAt` with the derivative of the inverse of
`extChartAt` gives the identity.
Version where the basepoint belongs to `(extChartAt I x).source`. -/
lemma mfderiv_extChartAt_comp_mfderivWithin_extChartAt_symm' {x : M}
    {y : M} (hy : y ∈ (extChartAt I x).source) :
    (mfderiv I 𝓘(𝕜, E) (extChartAt I x) y) ∘L
      (mfderivWithin 𝓘(𝕜, E) I (extChartAt I x).symm (range I) (extChartAt I x y))
    = ContinuousLinearMap.id _ _ := by
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
    hy : Membership.mem (extChartAt I x).source y
    ⊢ Eq ((mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) y).comp (mfder …
  -/
  have : y = (extChartAt I x).symm (extChartAt I x y) := ((extChartAt I x).left_inv hy).symm
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
    hy : Membership.mem (extChartAt I x).source y
    this : Eq y (↑(extChartAt I x).symm (↑(extChartAt I x) y))
    ⊢ Eq ((mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) y).comp (mfder …
  -/
  convert mfderiv_extChartAt_comp_mfderivWithin_extChartAt_symm ((extChartAt I x).map_source hy)
  /-
    🎉 no goals
  -/


/-- The composition of the derivative of the inverse of `extChartAt` with the derivative of
`extChartAt` gives the identity.
Version where the basepoint belongs to `(extChartAt I x).target`. -/
lemma mfderivWithin_extChartAt_symm_comp_mfderiv_extChartAt
    {y : E} (hy : y ∈ (extChartAt I x).target) :
    (mfderivWithin 𝓘(𝕜, E) I (extChartAt I x).symm (range I) y) ∘L
      (mfderiv I 𝓘(𝕜, E) (extChartAt I x) ((extChartAt I x).symm y))
      = ContinuousLinearMap.id _ _ := by
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Eq ((mfderivWithin (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm) (Se …
  -/
  have h'y : (extChartAt I x).symm y ∈ (extChartAt I x).source := (extChartAt I x).map_target hy
  have h''y : (extChartAt I x).symm y ∈ (chartAt H x).source := by
    rwa [← extChartAt_source (I := I)]
  have U' : UniqueMDiffWithinAt I (extChartAt I x).source ((extChartAt I x).symm y) :=
    (isOpen_extChartAt_source x).uniqueMDiffWithinAt h'y
  have : mfderiv I 𝓘(𝕜, E) (extChartAt I x) ((extChartAt I x).symm y)
      = mfderivWithin I 𝓘(𝕜, E) (extChartAt I x) (extChartAt I x).source
      ((extChartAt I x).symm y) := by
    rw [mfderivWithin_eq_mfderiv U']
    exact mdifferentiableAt_extChartAt h''y
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
    h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
    U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
    this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
    ⊢ Eq ((mfderivWithin (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm) (Se …
  -/
  rw [this, ← mfderivWithin_comp_of_eq]; rotate_left
    /-
      case hg
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      ⊢ MDifferentiableWithinAt (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm …
    -/
  · exact mdifferentiableWithinAt_extChartAt_symm hy
    /-
      🎉 no goals
    -/
    /-
      case hf
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      ⊢ MDifferentiableWithinAt I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (ex …
    -/
  · exact (mdifferentiableAt_extChartAt h''y).mdifferentiableWithinAt
    /-
      🎉 no goals
    -/
    /-
      case h
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      ⊢ HasSubset.Subset (extChartAt I x).source (Set.preimage (↑(extChartAt I x)) ( …
    -/
  · intro z hz
    /-
      case h
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      z : M
      hz : Membership.mem (extChartAt I x).source z
      ⊢ Membership.mem (Set.preimage (↑(extChartAt I x)) (Set.range ↑I)) z
    -/
    apply extChartAt_target_subset_range x
    /-
      case h.a
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      z : M
      hz : Membership.mem (extChartAt I x).source z
      ⊢ Membership.mem (extChartAt I x).target (↑(extChartAt I x) z)
    -/
    exact PartialEquiv.map_source (extChartAt I x) hz
    /-
      🎉 no goals
    -/
    /-
      case hxs
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      ⊢ UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
    -/
  · exact U'
    /-
      🎉 no goals
    -/
    /-
      case hy
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      ⊢ Eq (↑(extChartAt I x) (↑(extChartAt I x).symm y)) y
    -/
  · exact PartialEquiv.right_inv (extChartAt I x) hy
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
    h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
    U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
    this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
    ⊢ Eq (mfderivWithin I I (Function.comp ↑(extChartAt I x).symm ↑(extChartAt I x …
  -/
  rw [← mfderivWithin_id U']
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
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
    h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
    U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
    this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
    ⊢ Eq (mfderivWithin I I (Function.comp ↑(extChartAt I x).symm ↑(extChartAt I x …
  -/
  apply Filter.EventuallyEq.mfderivWithin_eq U'
    /-
      case hL
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      ⊢ (nhdsWithin (↑(extChartAt I x).symm y) (extChartAt I x).source).EventuallyEq …
    -/
  · filter_upwards [extChartAt_source_mem_nhdsWithin' h'y] with z hz
    /-
      case h
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      z : M
      hz : Membership.mem (extChartAt I x).source z
      ⊢ Eq (Function.comp (↑(extChartAt I x).symm) (↑(extChartAt I x)) z) (id z)
    -/
    simp only [Function.comp_def, PartialEquiv.left_inv (extChartAt I x) hz, id_eq]
    /-
      🎉 no goals
    -/
    /-
      case hx
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
      x : M
      y : E
      hy : Membership.mem (extChartAt I x).target y
      h'y : Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
      h''y : Membership.mem (chartAt H x).source (↑(extChartAt I x).symm y)
      U' : UniqueMDiffWithinAt I (extChartAt I x).source (↑(extChartAt I x).symm y)
      this : Eq (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChar …
      ⊢ Eq (Function.comp (↑(extChartAt I x).symm) (↑(extChartAt I x)) (↑(extChartAt …
    -/
  · simp only [Function.comp_def, PartialEquiv.right_inv (extChartAt I x) hy, id_eq]
    /-
      🎉 no goals
    -/


/-- The composition of the derivative of the inverse of `extChartAt` with the derivative of
`extChartAt` gives the identity.
Version where the basepoint belongs to `(extChartAt I x).source`. -/
lemma mfderivWithin_extChartAt_symm_comp_mfderiv_extChartAt'
    {y : M} (hy : y ∈ (extChartAt I x).source) :
    (mfderivWithin 𝓘(𝕜, E) I (extChartAt I x).symm (range I) (extChartAt I x y)) ∘L
      (mfderiv I 𝓘(𝕜, E) (extChartAt I x) y)
      = ContinuousLinearMap.id _ _ := by
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
    hy : Membership.mem (extChartAt I x).source y
    ⊢ Eq ((mfderivWithin (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm) (Se …
  -/
  have : y = (extChartAt I x).symm (extChartAt I x y) := ((extChartAt I x).left_inv hy).symm
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
    hy : Membership.mem (extChartAt I x).source y
    this : Eq y (↑(extChartAt I x).symm (↑(extChartAt I x) y))
    ⊢ Eq ((mfderivWithin (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x).symm) (Se …
  -/
  convert mfderivWithin_extChartAt_symm_comp_mfderiv_extChartAt ((extChartAt I x).map_source hy)
  /-
    🎉 no goals
  -/


lemma isInvertible_mfderivWithin_extChartAt_symm {y : E} (hy : y ∈ (extChartAt I x).target) :
    (mfderivWithin 𝓘(𝕜, E) I (extChartAt I x).symm (range I) y).IsInvertible :=
  ContinuousLinearMap.IsInvertible.of_inverse
    (mfderivWithin_extChartAt_symm_comp_mfderiv_extChartAt hy)
    (mfderiv_extChartAt_comp_mfderivWithin_extChartAt_symm hy)


lemma isInvertible_mfderiv_extChartAt {y : M} (hy : y ∈ (extChartAt I x).source) :
    (mfderiv I 𝓘(𝕜, E) (extChartAt I x) y).IsInvertible := by
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
    hy : Membership.mem (extChartAt I x).source y
    ⊢ (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) y).IsInvertible
  -/
  have h'y : extChartAt I x y ∈ (extChartAt I x).target := (extChartAt I x).map_source hy
  have Z := ContinuousLinearMap.IsInvertible.of_inverse
    (mfderiv_extChartAt_comp_mfderivWithin_extChartAt_symm h'y)
    (mfderivWithin_extChartAt_symm_comp_mfderiv_extChartAt h'y)
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
    hy : Membership.mem (extChartAt I x).source y
    h'y : Membership.mem (extChartAt I x).target (↑(extChartAt I x) y)
    Z : (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChartAt I  …
    ⊢ (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) y).IsInvertible
  -/
  have : (extChartAt I x).symm ((extChartAt I x) y) = y := (extChartAt I x).left_inv hy
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
    hy : Membership.mem (extChartAt I x).source y
    h'y : Membership.mem (extChartAt I x).target (↑(extChartAt I x) y)
    Z : (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) (↑(extChartAt I  …
    this : Eq (↑(extChartAt I x).symm (↑(extChartAt I x) y)) y
    ⊢ (mfderiv I (modelWithCornersSelf 𝕜 E) (↑(extChartAt I x)) y).IsInvertible
  -/
  rwa [this] at Z
  /-
    🎉 no goals
  -/


