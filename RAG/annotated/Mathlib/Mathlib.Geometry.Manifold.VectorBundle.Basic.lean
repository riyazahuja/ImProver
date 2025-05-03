/-- A fiber bundle `E` over a base `B` with model fiber `F` is naturally a charted space modelled on
`B × F`. -/
instance FiberBundle.chartedSpace' : ChartedSpace (B × F) (TotalSpace F E) where
  atlas := (fun e : Trivialization F (π F E) => e.toPartialHomeomorph) '' trivializationAtlas F E
  chartAt x := (trivializationAt F E x.proj).toPartialHomeomorph
  mem_chart_source x :=
    (trivializationAt F E x.proj).mem_source.mpr (mem_baseSet_trivializationAt F E x.proj)
  chart_mem_atlas _ := mem_image_of_mem _ (trivialization_mem_atlas F E _)


theorem FiberBundle.chartedSpace'_chartAt (x : TotalSpace F E) :
    chartAt (B × F) x = (trivializationAt F E x.proj).toPartialHomeomorph :=
  rfl

/- Porting note: In Lean 3, the next instance was inside a section with locally reducible
`ModelProd` and it used `ModelProd B F` as the intermediate space. Using `B × F` in the middle
gives the same instance.
-/
--attribute [local reducible] ModelProd


/-- Let `B` be a charted space modelled on `HB`.  Then a fiber bundle `E` over a base `B` with model
fiber `F` is naturally a charted space modelled on `HB.prod F`. -/
instance FiberBundle.chartedSpace : ChartedSpace (ModelProd HB F) (TotalSpace F E) :=
  ChartedSpace.comp _ (B × F) _


theorem FiberBundle.chartedSpace_chartAt (x : TotalSpace F E) :
    chartAt (ModelProd HB F) x =
      (trivializationAt F E x.proj).toPartialHomeomorph ≫ₕ
        (chartAt HB x.proj).prod (PartialHomeomorph.refl F) := by
  dsimp only [chartAt_comp, prodChartedSpace_chartAt, FiberBundle.chartedSpace'_chartAt,
    chartAt_self_eq]
  /-
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    HB : Type u_7
    inst✝³ : TopologicalSpace HB
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    ⊢ Eq ((FiberBundle.trivializationAt F E x.proj).trans ((chartAt HB (↑(FiberBun …
  -/
  rw [Trivialization.coe_coe, Trivialization.coe_fst' _ (mem_baseSet_trivializationAt F E x.proj)]
  /-
    🎉 no goals
  -/


theorem FiberBundle.chartedSpace_chartAt_symm_fst (x : TotalSpace F E) (y : ModelProd HB F)
    (hy : y ∈ (chartAt (ModelProd HB F) x).target) :
    ((chartAt (ModelProd HB F) x).symm y).proj = (chartAt HB x.proj).symm y.1 := by
  /-
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    HB : Type u_7
    inst✝³ : TopologicalSpace HB
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    y : ModelProd HB F
    hy : Membership.mem (chartAt (ModelProd HB F) x).target y
    ⊢ Eq (↑(chartAt (ModelProd HB F) x).symm y).proj (↑(chartAt HB x.proj).symm y.1)
  -/
  simp only [FiberBundle.chartedSpace_chartAt, mfld_simps] at hy ⊢
  /-
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    HB : Type u_7
    inst✝³ : TopologicalSpace HB
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    y : ModelProd HB F
    hy : And (Membership.mem (SProd.sprod (chartAt HB x.proj).target Set.univ) y)  …
    ⊢ Eq (↑(FiberBundle.trivializationAt F E x.proj).symm (↑((chartAt HB x.proj).p …
  -/
  exact (trivializationAt F E x.proj).proj_symm_apply hy.2
  /-
    🎉 no goals
  -/


protected theorem FiberBundle.extChartAt (x : TotalSpace F E) :
    extChartAt (IB.prod 𝓘(𝕜, F)) x =
      (trivializationAt F E x.proj).toPartialEquiv ≫
        (extChartAt IB x.proj).prod (PartialEquiv.refl F) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    ⊢ Eq (extChartAt (IB.prod (modelWithCornersSelf 𝕜 F)) x) ((FiberBundle.trivial …
  -/
  simp_rw [extChartAt, FiberBundle.chartedSpace_chartAt, extend]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    ⊢ Eq (((FiberBundle.trivializationAt F E x.proj).trans ((chartAt HB x.proj).pr …
  -/
  simp only [PartialEquiv.trans_assoc, mfld_simps]
  -- Porting note: should not be needed
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    ⊢ Eq ((FiberBundle.trivializationAt F E x.proj).trans (((chartAt HB x.proj).pr …
  -/
  rw [PartialEquiv.prod_trans, PartialEquiv.refl_trans]
  /-
    🎉 no goals
  -/


protected theorem FiberBundle.extChartAt_target (x : TotalSpace F E) :
    (extChartAt (IB.prod 𝓘(𝕜, F)) x).target =
      ((extChartAt IB x.proj).target ∩
        (extChartAt IB x.proj).symm ⁻¹' (trivializationAt F E x.proj).baseSet) ×ˢ univ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    ⊢ Eq (extChartAt (IB.prod (modelWithCornersSelf 𝕜 F)) x).target (SProd.sprod ( …
  -/
  rw [FiberBundle.extChartAt, PartialEquiv.trans_target, Trivialization.target_eq, inter_prod]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    ⊢ Eq (Inter.inter ((extChartAt IB x.proj).prod (PartialEquiv.refl F)).target ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem FiberBundle.writtenInExtChartAt_trivializationAt {x : TotalSpace F E} {y}
    (hy : y ∈ (extChartAt (IB.prod 𝓘(𝕜, F)) x).target) :
    writtenInExtChartAt (IB.prod 𝓘(𝕜, F)) (IB.prod 𝓘(𝕜, F)) x
      (trivializationAt F E x.proj) y = y :=
  writtenInExtChartAt_chartAt_comp _ hy


theorem FiberBundle.writtenInExtChartAt_trivializationAt_symm {x : TotalSpace F E} {y}
    (hy : y ∈ (extChartAt (IB.prod 𝓘(𝕜, F)) x).target) :
    writtenInExtChartAt (IB.prod 𝓘(𝕜, F)) (IB.prod 𝓘(𝕜, F)) (trivializationAt F E x.proj x)
      (trivializationAt F E x.proj).toPartialHomeomorph.symm y = y :=
  writtenInExtChartAt_chartAt_symm_comp _ hy


/-- Characterization of C^n functions into a smooth vector bundle. -/
theorem contMDiffWithinAt_totalSpace (f : M → TotalSpace F E) {s : Set M} {x₀ : M} :
    ContMDiffWithinAt IM (IB.prod 𝓘(𝕜, F)) n f s x₀ ↔
      ContMDiffWithinAt IM IB n (fun x => (f x).proj) s x₀ ∧
      ContMDiffWithinAt IM 𝓘(𝕜, F) n (fun x ↦ (trivializationAt F E (f x₀).proj (f x)).2) s x₀ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    ⊢ Iff (ContMDiffWithinAt IM (IB.prod (modelWithCornersSelf 𝕜 F)) n f s x₀) (An …
  -/
  simp (config := { singlePass := true }) only [contMDiffWithinAt_iff_target]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    ⊢ Iff (And (ContinuousWithinAt f s x₀) (ContMDiffWithinAt IM (modelWithCorners …
  -/
  rw [and_and_and_comm, ← FiberBundle.continuousWithinAt_totalSpace, and_congr_right_iff]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    ⊢ ContinuousWithinAt f s x₀ → Iff (ContMDiffWithinAt IM (modelWithCornersSelf  …
  -/
  intro hf
  simp_rw [modelWithCornersSelf_prod, FiberBundle.extChartAt, Function.comp_def,
    PartialEquiv.trans_apply, PartialEquiv.prod_coe, PartialEquiv.refl_coe,
    extChartAt_self_apply, modelWithCornersSelf_coe, Function.id_def, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    hf : ContinuousWithinAt f s x₀
    ⊢ Iff (ContMDiffWithinAt IM ((modelWithCornersSelf 𝕜 EB).prod (modelWithCorner …
  -/
  refine (contMDiffWithinAt_prod_iff _).trans (and_congr ?_ Iff.rfl)
  have h1 : (fun x => (f x).proj) ⁻¹' (trivializationAt F E (f x₀).proj).baseSet ∈ 𝓝[s] x₀ :=
    ((FiberBundle.continuous_proj F E).continuousWithinAt.comp hf (mapsTo_image f s))
      ((Trivialization.open_baseSet _).mem_nhds (mem_baseSet_trivializationAt F E _))
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    s : Set M
    x₀ : M
    hf : ContinuousWithinAt f s x₀
    h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
    ⊢ Iff (ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 EB) n (Function.comp Prod. …
  -/
  refine EventuallyEq.contMDiffWithinAt_iff (eventually_of_mem h1 fun x hx => ?_) ?_
    /-
      case refine_1
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      inst✝¹⁴ : NormedAddCommGroup F
      inst✝¹³ : NormedSpace 𝕜 F
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹¹ : (x : B) → TopologicalSpace (E x)
      EB : Type u_7
      inst✝¹⁰ : NormedAddCommGroup EB
      inst✝⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      EM : Type u_10
      inst✝⁷ : NormedAddCommGroup EM
      inst✝⁶ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝⁵ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace HM M
      n : ENat
      inst✝² : TopologicalSpace B
      inst✝¹ : ChartedSpace HB B
      inst✝ : FiberBundle F E
      f : M → Bundle.TotalSpace F E
      s : Set M
      x₀ : M
      hf : ContinuousWithinAt f s x₀
      h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
      x : M
      hx : Membership.mem (Set.preimage (fun x => (f x).proj) (FiberBundle.trivializ …
      ⊢ Eq (Function.comp Prod.fst (fun x => { fst := ↑(extChartAt IB (f x₀).proj) ( …
    -/
  · simp_rw [Function.comp, PartialHomeomorph.coe_coe, Trivialization.coe_coe]
    /-
      case refine_1
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      inst✝¹⁴ : NormedAddCommGroup F
      inst✝¹³ : NormedSpace 𝕜 F
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹¹ : (x : B) → TopologicalSpace (E x)
      EB : Type u_7
      inst✝¹⁰ : NormedAddCommGroup EB
      inst✝⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      EM : Type u_10
      inst✝⁷ : NormedAddCommGroup EM
      inst✝⁶ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝⁵ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace HM M
      n : ENat
      inst✝² : TopologicalSpace B
      inst✝¹ : ChartedSpace HB B
      inst✝ : FiberBundle F E
      f : M → Bundle.TotalSpace F E
      s : Set M
      x₀ : M
      hf : ContinuousWithinAt f s x₀
      h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
      x : M
      hx : Membership.mem (Set.preimage (fun x => (f x).proj) (FiberBundle.trivializ …
      ⊢ Eq (↑(extChartAt IB (f x₀).proj) (↑(FiberBundle.trivializationAt F E (f x₀). …
    -/
    rw [Trivialization.coe_fst']
    /-
      case refine_1.ex
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      inst✝¹⁴ : NormedAddCommGroup F
      inst✝¹³ : NormedSpace 𝕜 F
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹¹ : (x : B) → TopologicalSpace (E x)
      EB : Type u_7
      inst✝¹⁰ : NormedAddCommGroup EB
      inst✝⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      EM : Type u_10
      inst✝⁷ : NormedAddCommGroup EM
      inst✝⁶ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝⁵ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace HM M
      n : ENat
      inst✝² : TopologicalSpace B
      inst✝¹ : ChartedSpace HB B
      inst✝ : FiberBundle F E
      f : M → Bundle.TotalSpace F E
      s : Set M
      x₀ : M
      hf : ContinuousWithinAt f s x₀
      h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
      x : M
      hx : Membership.mem (Set.preimage (fun x => (f x).proj) (FiberBundle.trivializ …
      ⊢ Membership.mem (FiberBundle.trivializationAt F E (f x₀).proj).baseSet (f x). …
    -/
    exact hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      inst✝¹⁴ : NormedAddCommGroup F
      inst✝¹³ : NormedSpace 𝕜 F
      inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹¹ : (x : B) → TopologicalSpace (E x)
      EB : Type u_7
      inst✝¹⁰ : NormedAddCommGroup EB
      inst✝⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      EM : Type u_10
      inst✝⁷ : NormedAddCommGroup EM
      inst✝⁶ : NormedSpace 𝕜 EM
      HM : Type u_11
      inst✝⁵ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace HM M
      n : ENat
      inst✝² : TopologicalSpace B
      inst✝¹ : ChartedSpace HB B
      inst✝ : FiberBundle F E
      f : M → Bundle.TotalSpace F E
      s : Set M
      x₀ : M
      hf : ContinuousWithinAt f s x₀
      h1 : Membership.mem (nhdsWithin x₀ s) (Set.preimage (fun x => (f x).proj) (Fib …
      ⊢ Eq (Function.comp Prod.fst (fun x => { fst := ↑(extChartAt IB (f x₀).proj) ( …
    -/
  · simp only [mfld_simps]
    /-
      🎉 no goals
    -/


/-- Characterization of C^n functions into a smooth vector bundle. -/
theorem contMDiffAt_totalSpace (f : M → TotalSpace F E) (x₀ : M) :
    ContMDiffAt IM (IB.prod 𝓘(𝕜, F)) n f x₀ ↔
      ContMDiffAt IM IB n (fun x => (f x).proj) x₀ ∧
        ContMDiffAt IM 𝓘(𝕜, F) n (fun x => (trivializationAt F E (f x₀).proj (f x)).2) x₀ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    inst✝¹² : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝¹¹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝¹⁰ : NormedAddCommGroup EB
    inst✝⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    EM : Type u_10
    inst✝⁷ : NormedAddCommGroup EM
    inst✝⁶ : NormedSpace 𝕜 EM
    HM : Type u_11
    inst✝⁵ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace HM M
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    f : M → Bundle.TotalSpace F E
    x₀ : M
    ⊢ Iff (ContMDiffAt IM (IB.prod (modelWithCornersSelf 𝕜 F)) n f x₀) (And (ContM …
  -/
  simp_rw [← contMDiffWithinAt_univ]; exact contMDiffWithinAt_totalSpace f
                                      /-
                                        🎉 no goals
                                      -/


/-- Characterization of C^n sections within a set at a point of a smooth vector bundle. -/
theorem contMDiffWithinAt_section (s : ∀ x, E x) (a : Set B) (x₀ : B) :
    ContMDiffWithinAt IB (IB.prod 𝓘(𝕜, F)) n (fun x => TotalSpace.mk' F x (s x)) a x₀ ↔
      ContMDiffWithinAt IB 𝓘(𝕜, F) n (fun x ↦ (trivializationAt F E x₀ ⟨x, s x⟩).2) a x₀ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    s : (x : B) → E x
    a : Set B
    x₀ : B
    ⊢ Iff (ContMDiffWithinAt IB (IB.prod (modelWithCornersSelf 𝕜 F)) n (fun x => B …
  -/
  simp_rw [contMDiffWithinAt_totalSpace, and_iff_right_iff_imp]; intro; exact contMDiffWithinAt_id
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- Characterization of C^n sections of a smooth vector bundle. -/
theorem contMDiffAt_section (s : ∀ x, E x) (x₀ : B) :
    ContMDiffAt IB (IB.prod 𝓘(𝕜, F)) n (fun x => TotalSpace.mk' F x (s x)) x₀ ↔
      ContMDiffAt IB 𝓘(𝕜, F) n (fun x ↦ (trivializationAt F E x₀ ⟨x, s x⟩).2) x₀ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    s : (x : B) → E x
    x₀ : B
    ⊢ Iff (ContMDiffAt IB (IB.prod (modelWithCornersSelf 𝕜 F)) n (fun x => Bundle. …
  -/
  simp_rw [contMDiffAt_totalSpace, and_iff_right_iff_imp]; intro; exact contMDiffAt_id
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem contMDiff_proj : ContMDiff (IB.prod 𝓘(𝕜, F)) IB n (π F E) := fun x ↦ by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    ⊢ ContMDiffAt (IB.prod (modelWithCornersSelf 𝕜 F)) IB n Bundle.TotalSpace.proj x
  -/
  have : ContMDiffAt (IB.prod 𝓘(𝕜, F)) (IB.prod 𝓘(𝕜, F)) n id x := contMDiffAt_id
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    this : ContMDiffAt (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCor …
    ⊢ ContMDiffAt (IB.prod (modelWithCornersSelf 𝕜 F)) IB n Bundle.TotalSpace.proj x
  -/
  rw [contMDiffAt_totalSpace] at this
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    inst✝⁷ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁶ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁵ : NormedAddCommGroup EB
    inst✝⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    n : ENat
    inst✝² : TopologicalSpace B
    inst✝¹ : ChartedSpace HB B
    inst✝ : FiberBundle F E
    x : Bundle.TotalSpace F E
    this : And (ContMDiffAt (IB.prod (modelWithCornersSelf 𝕜 F)) IB n (fun x => (i …
    ⊢ ContMDiffAt (IB.prod (modelWithCornersSelf 𝕜 F)) IB n Bundle.TotalSpace.proj x
  -/
  exact this.1
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-21")] alias smooth_proj := contMDiff_proj


theorem contMDiffOn_proj {s : Set (TotalSpace F E)} :
    ContMDiffOn (IB.prod 𝓘(𝕜, F)) IB n (π F E) s :=
  (Bundle.contMDiff_proj E).contMDiffOn


@[deprecated (since := "2024-11-21")] alias smoothOn_proj := contMDiffOn_proj


theorem contMDiffAt_proj {p : TotalSpace F E} : ContMDiffAt (IB.prod 𝓘(𝕜, F)) IB n (π F E) p :=
  (Bundle.contMDiff_proj E).contMDiffAt


@[deprecated (since := "2024-11-21")] alias smoothAt_proj := contMDiffAt_proj


theorem contMDiffWithinAt_proj {s : Set (TotalSpace F E)} {p : TotalSpace F E} :
    ContMDiffWithinAt (IB.prod 𝓘(𝕜, F)) IB n (π F E) s p :=
  (Bundle.contMDiffAt_proj E).contMDiffWithinAt


@[deprecated (since := "2024-11-21")] alias smoothWithinAt_proj := contMDiffWithinAt_proj


theorem contMDiff_zeroSection : ContMDiff IB (IB.prod 𝓘(𝕜, F)) ⊤ (zeroSection F E) := fun x ↦ by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : NormedAddCommGroup F
    inst✝¹¹ : NormedSpace 𝕜 F
    inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁸ : NormedAddCommGroup EB
    inst✝⁷ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁶ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : ChartedSpace HB B
    inst✝³ : FiberBundle F E
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module 𝕜 (E x)
    inst✝ : VectorBundle 𝕜 F E
    x : B
    ⊢ ContMDiffAt IB (IB.prod (modelWithCornersSelf 𝕜 F)) Top.top (Bundle.zeroSect …
  -/
  unfold zeroSection
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : NormedAddCommGroup F
    inst✝¹¹ : NormedSpace 𝕜 F
    inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁸ : NormedAddCommGroup EB
    inst✝⁷ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁶ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : ChartedSpace HB B
    inst✝³ : FiberBundle F E
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module 𝕜 (E x)
    inst✝ : VectorBundle 𝕜 F E
    x : B
    ⊢ ContMDiffAt IB (IB.prod (modelWithCornersSelf 𝕜 F)) Top.top (fun x => { proj …
  -/
  rw [Bundle.contMDiffAt_section]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : NormedAddCommGroup F
    inst✝¹¹ : NormedSpace 𝕜 F
    inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁹ : (x : B) → TopologicalSpace (E x)
    EB : Type u_7
    inst✝⁸ : NormedAddCommGroup EB
    inst✝⁷ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁶ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : ChartedSpace HB B
    inst✝³ : FiberBundle F E
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module 𝕜 (E x)
    inst✝ : VectorBundle 𝕜 F E
    x : B
    ⊢ ContMDiffAt IB (modelWithCornersSelf 𝕜 F) Top.top (fun x_1 => (↑(FiberBundle …
  -/
  apply (contMDiffAt_const (c := 0)).congr_of_eventuallyEq
  filter_upwards [(trivializationAt F E x).open_baseSet.mem_nhds
    (mem_baseSet_trivializationAt F E x)] with y hy
    using congr_arg Prod.snd <| (trivializationAt F E x).zeroSection 𝕜 hy


@[deprecated (since := "2024-11-21")] alias smooth_zeroSection := contMDiff_zeroSection


variable (IB) in
/-- When `B` is a smooth manifold with corners with respect to a model `IB` and `E` is a
topological vector bundle over `B` with fibers isomorphic to `F`, then `SmoothVectorBundle F E IB`
registers that the bundle is smooth, in the sense of having smooth transition functions.
This is a mixin, not carrying any new data. -/
class SmoothVectorBundle : Prop where
  protected contMDiffOn_coordChangeL :
    ∀ (e e' : Trivialization F (π F E)) [MemTrivializationAtlas e] [MemTrivializationAtlas e'],
      ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun b : B => (e.coordChangeL 𝕜 e' b : F →L[𝕜] F))
        (e.baseSet ∩ e'.baseSet)


theorem contMDiffOn_coordChangeL :
    ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) n (fun b : B => (e.coordChangeL 𝕜 e' b : F →L[𝕜] F))
      (e.baseSet ∩ e'.baseSet) :=
  (SmoothVectorBundle.contMDiffOn_coordChangeL e e').of_le le_top


theorem contMDiffOn_symm_coordChangeL :
    ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) n (fun b : B => ((e.coordChangeL 𝕜 e' b).symm : F →L[𝕜] F))
      (e.baseSet ∩ e'.baseSet) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁶ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁵ : NormedAddCommGroup EB
    inst✝¹⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹² : TopologicalSpace B
    inst✝¹¹ : ChartedSpace HB B
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
  -/
  apply ContMDiffOn.of_le _ le_top
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁶ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁵ : NormedAddCommGroup EB
    inst✝¹⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹² : TopologicalSpace B
    inst✝¹¹ : ChartedSpace HB B
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
  -/
  rw [inter_comm]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁶ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁵ : NormedAddCommGroup EB
    inst✝¹⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹² : TopologicalSpace B
    inst✝¹¹ : ChartedSpace HB B
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
  -/
  refine (SmoothVectorBundle.contMDiffOn_coordChangeL e' e).congr fun b hb ↦ ?_
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁶ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁵ : NormedAddCommGroup EB
    inst✝¹⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹² : TopologicalSpace B
    inst✝¹¹ : ChartedSpace HB B
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    b : B
    hb : Membership.mem (Inter.inter e'.baseSet e.baseSet) b
    ⊢ Eq ↑(Trivialization.coordChangeL 𝕜 e e' b).symm ↑(Trivialization.coordChange …
  -/
  rw [e.symm_coordChangeL e' hb]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-21")] alias smoothOn_coordChangeL := contMDiffOn_coordChangeL

@[deprecated (since := "2024-11-21")]
alias smoothOn_symm_coordChangeL := contMDiffOn_symm_coordChangeL



theorem contMDiffAt_coordChangeL {x : B} (h : x ∈ e.baseSet) (h' : x ∈ e'.baseSet) :
    ContMDiffAt IB 𝓘(𝕜, F →L[𝕜] F) n (fun b : B => (e.coordChangeL 𝕜 e' b : F →L[𝕜] F)) x :=
  (contMDiffOn_coordChangeL e e').contMDiffAt <|
    (e.open_baseSet.inter e'.open_baseSet).mem_nhds ⟨h, h'⟩


@[deprecated (since := "2024-11-21")] alias smoothAt_coordChangeL := contMDiffAt_coordChangeL


protected theorem ContMDiffWithinAt.coordChangeL
    (hf : ContMDiffWithinAt IM IB n f s x) (he : f x ∈ e.baseSet) (he' : f x ∈ e'.baseSet) :
    ContMDiffWithinAt IM 𝓘(𝕜, F →L[𝕜] F) n (fun y ↦ (e.coordChangeL 𝕜 e' (f y) : F →L[𝕜] F)) s x :=
  (contMDiffAt_coordChangeL he he').comp_contMDiffWithinAt _ hf


protected nonrec theorem ContMDiffAt.coordChangeL
    (hf : ContMDiffAt IM IB n f x) (he : f x ∈ e.baseSet) (he' : f x ∈ e'.baseSet) :
    ContMDiffAt IM 𝓘(𝕜, F →L[𝕜] F) n (fun y ↦ (e.coordChangeL 𝕜 e' (f y) : F →L[𝕜] F)) x :=
  hf.coordChangeL he he'


protected theorem ContMDiffOn.coordChangeL
    (hf : ContMDiffOn IM IB n f s) (he : MapsTo f s e.baseSet) (he' : MapsTo f s e'.baseSet) :
    ContMDiffOn IM 𝓘(𝕜, F →L[𝕜] F) n (fun y ↦ (e.coordChangeL 𝕜 e' (f y) : F →L[𝕜] F)) s :=
  fun x hx ↦ (hf x hx).coordChangeL (he hx) (he' hx)


protected theorem ContMDiff.coordChangeL
    (hf : ContMDiff IM IB n f) (he : ∀ x, f x ∈ e.baseSet) (he' : ∀ x, f x ∈ e'.baseSet) :
    ContMDiff IM 𝓘(𝕜, F →L[𝕜] F) n (fun y ↦ (e.coordChangeL 𝕜 e' (f y) : F →L[𝕜] F)) := fun x ↦
  (hf x).coordChangeL (he x) (he' x)


@[deprecated (since := "2024-11-21")]
alias SmoothWithinAt.coordChangeL := ContMDiffWithinAt.coordChangeL


@[deprecated (since := "2024-11-21")]
alias SmoothAt.coordChangeL := ContMDiffAt.coordChangeL


@[deprecated (since := "2024-11-21")]
alias SmoothOn.coordChangeL := ContMDiffOn.coordChangeL


@[deprecated (since := "2024-11-21")]
alias Smooth.coordChangeL := ContMDiff.coordChangeL


protected theorem ContMDiffWithinAt.coordChange
    (hf : ContMDiffWithinAt IM IB n f s x) (hg : ContMDiffWithinAt IM 𝓘(𝕜, F) n g s x)
    (he : f x ∈ e.baseSet) (he' : f x ∈ e'.baseSet) :
    ContMDiffWithinAt IM 𝓘(𝕜, F) n (fun y ↦ e.coordChange e' (f y) (g y)) s x := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²¹ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝²⁰ : NormedAddCommGroup EB
    inst✝¹⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁵ : NormedAddCommGroup EM
    inst✝¹⁴ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹³ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹² : TopologicalSpace M
    inst✝¹¹ : ChartedSpace HM M
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    s : Set M
    f : M → B
    g : M → F
    x : M
    hf : ContMDiffWithinAt IM IB n f s x
    hg : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n g s x
    he : Membership.mem e.baseSet (f x)
    he' : Membership.mem e'.baseSet (f x)
    ⊢ ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n (fun y => e.coordChange e' …
  -/
  refine ((hf.coordChangeL he he').clm_apply hg).congr_of_eventuallyEq ?_ ?_
  · have : e.baseSet ∩ e'.baseSet ∈ 𝓝 (f x) :=
     (e.open_baseSet.inter e'.open_baseSet).mem_nhds ⟨he, he'⟩
    /-
      case refine_1
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : MemTrivializationAtlas e
      inst✝ : MemTrivializationAtlas e'
      s : Set M
      f : M → B
      g : M → F
      x : M
      hf : ContMDiffWithinAt IM IB n f s x
      hg : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n g s x
      he : Membership.mem e.baseSet (f x)
      he' : Membership.mem e'.baseSet (f x)
      this : Membership.mem (nhds (f x)) (Inter.inter e.baseSet e'.baseSet)
      ⊢ (nhdsWithin x s).EventuallyEq (fun y => e.coordChange e' (f y) (g y)) fun x  …
    -/
    filter_upwards [hf.continuousWithinAt this] with y hy
    /-
      case h
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : MemTrivializationAtlas e
      inst✝ : MemTrivializationAtlas e'
      s : Set M
      f : M → B
      g : M → F
      x : M
      hf : ContMDiffWithinAt IM IB n f s x
      hg : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n g s x
      he : Membership.mem e.baseSet (f x)
      he' : Membership.mem e'.baseSet (f x)
      this : Membership.mem (nhds (f x)) (Inter.inter e.baseSet e'.baseSet)
      y : M
      hy : Membership.mem (Set.preimage f (Inter.inter e.baseSet e'.baseSet)) y
      ⊢ Eq (e.coordChange e' (f y) (g y)) (↑(Trivialization.coordChangeL 𝕜 e e' (f y …
    -/
    exact (Trivialization.coordChangeL_apply' e e' hy (g y)).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : MemTrivializationAtlas e
      inst✝ : MemTrivializationAtlas e'
      s : Set M
      f : M → B
      g : M → F
      x : M
      hf : ContMDiffWithinAt IM IB n f s x
      hg : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n g s x
      he : Membership.mem e.baseSet (f x)
      he' : Membership.mem e'.baseSet (f x)
      ⊢ Eq (e.coordChange e' (f x) (g x)) (↑(Trivialization.coordChangeL 𝕜 e e' (f x …
    -/
  · exact (Trivialization.coordChangeL_apply' e e' ⟨he, he'⟩ (g x)).symm
    /-
      🎉 no goals
    -/


protected nonrec theorem ContMDiffAt.coordChange
    (hf : ContMDiffAt IM IB n f x) (hg : ContMDiffAt IM 𝓘(𝕜, F) n g x) (he : f x ∈ e.baseSet)
    (he' : f x ∈ e'.baseSet) :
    ContMDiffAt IM 𝓘(𝕜, F) n (fun y ↦ e.coordChange e' (f y) (g y)) x :=
  hf.coordChange hg he he'


protected theorem ContMDiffOn.coordChange (hf : ContMDiffOn IM IB n f s)
    (hg : ContMDiffOn IM 𝓘(𝕜, F) n g s) (he : MapsTo f s e.baseSet) (he' : MapsTo f s e'.baseSet) :
    ContMDiffOn IM 𝓘(𝕜, F) n (fun y ↦ e.coordChange e' (f y) (g y)) s := fun x hx ↦
  (hf x hx).coordChange (hg x hx) (he hx) (he' hx)


protected theorem ContMDiff.coordChange (hf : ContMDiff IM IB n f)
    (hg : ContMDiff IM 𝓘(𝕜, F) n g) (he : ∀ x, f x ∈ e.baseSet) (he' : ∀ x, f x ∈ e'.baseSet) :
    ContMDiff IM 𝓘(𝕜, F) n (fun y ↦ e.coordChange e' (f y) (g y)) := fun x ↦
  (hf x).coordChange (hg x) (he x) (he' x)


@[deprecated (since := "2024-11-21")]
alias SmoothWithinAt.coordChange := ContMDiffWithinAt.coordChange


@[deprecated (since := "2024-11-21")]
alias SmoothAt.coordChange := ContMDiffAt.coordChange


@[deprecated (since := "2024-11-21")]
alias SmoothOn.coordChange := ContMDiffOn.coordChange


@[deprecated (since := "2024-11-21")]
alias Smooth.coordChange := ContMDiff.coordChange


variable (IB) in
theorem Trivialization.contMDiffOn_symm_trans :
    ContMDiffOn (IB.prod 𝓘(𝕜, F)) (IB.prod 𝓘(𝕜, F)) n
      (e.toPartialHomeomorph.symm ≫ₕ e'.toPartialHomeomorph) (e.target ∩ e'.target) := by
  have Hmaps : MapsTo Prod.fst (e.target ∩ e'.target) (e.baseSet ∩ e'.baseSet) := fun x hx ↦
    ⟨e.mem_target.1 hx.1, e'.mem_target.1 hx.2⟩
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁶ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁵ : NormedAddCommGroup EB
    inst✝¹⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹² : TopologicalSpace B
    inst✝¹¹ : ChartedSpace HB B
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    Hmaps : Set.MapsTo Prod.fst (Inter.inter e.target e'.target) (Inter.inter e.ba …
    ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
  -/
  rw [mapsTo_inter] at Hmaps
  -- TODO: drop `congr` https://github.com/leanprover-community/mathlib4/issues/5473
  refine (contMDiffOn_fst.prod_mk
    (contMDiffOn_fst.coordChange contMDiffOn_snd Hmaps.1 Hmaps.2)).congr ?_
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁶ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁵ : NormedAddCommGroup EB
    inst✝¹⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹² : TopologicalSpace B
    inst✝¹¹ : ChartedSpace HB B
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    Hmaps : And (Set.MapsTo Prod.fst (Inter.inter e.target e'.target) e.baseSet) ( …
    ⊢ ∀ (y : Prod B F), Membership.mem (Inter.inter e.target e'.target) y → Eq (↑( …
  -/
  rintro ⟨b, x⟩ hb
  /-
    case mk
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁶ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁵ : NormedAddCommGroup EB
    inst✝¹⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹² : TopologicalSpace B
    inst✝¹¹ : ChartedSpace HB B
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    Hmaps : And (Set.MapsTo Prod.fst (Inter.inter e.target e'.target) e.baseSet) ( …
    b : B
    x : F
    hb : Membership.mem (Inter.inter e.target e'.target) { fst := b, snd := x }
    ⊢ Eq (↑(e.symm.trans e'.toPartialHomeomorph) { fst := b, snd := x }) { fst :=  …
  -/
  refine Prod.ext ?_ rfl
  have : (e.toPartialHomeomorph.symm (b, x)).1 ∈ e'.baseSet := by
    simp_all only [Trivialization.mem_target, mfld_simps]
  /-
    case mk
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁶ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁵ : NormedAddCommGroup EB
    inst✝¹⁴ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹³ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹² : TopologicalSpace B
    inst✝¹¹ : ChartedSpace HB B
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    Hmaps : And (Set.MapsTo Prod.fst (Inter.inter e.target e'.target) e.baseSet) ( …
    b : B
    x : F
    hb : Membership.mem (Inter.inter e.target e'.target) { fst := b, snd := x }
    this : Membership.mem e'.baseSet (↑e.symm { fst := b, snd := x }).proj
    ⊢ Eq (↑(e.symm.trans e'.toPartialHomeomorph) { fst := b, snd := x }).1 { fst : …
  -/
  exact (e'.coe_fst' this).trans (e.proj_symm_apply hb.1)
  /-
    🎉 no goals
  -/


theorem ContMDiffWithinAt.change_section_trivialization {f : M → TotalSpace F E}
    (hp : ContMDiffWithinAt IM IB n (π F E ∘ f) s x)
    (hf : ContMDiffWithinAt IM 𝓘(𝕜, F) n (fun y ↦ (e (f y)).2) s x)
    (he : f x ∈ e.source) (he' : f x ∈ e'.source) :
    ContMDiffWithinAt IM 𝓘(𝕜, F) n (fun y ↦ (e' (f y)).2) s x := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²¹ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝²⁰ : NormedAddCommGroup EB
    inst✝¹⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁵ : NormedAddCommGroup EM
    inst✝¹⁴ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹³ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹² : TopologicalSpace M
    inst✝¹¹ : ChartedSpace HM M
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    s : Set M
    x : M
    f : M → Bundle.TotalSpace F E
    hp : ContMDiffWithinAt IM IB n (Function.comp Bundle.TotalSpace.proj f) s x
    hf : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n (fun y => (↑e (f y)).2) …
    he : Membership.mem e.source (f x)
    he' : Membership.mem e'.source (f x)
    ⊢ ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n (fun y => (↑e' (f y)).2) s x
  -/
  rw [Trivialization.mem_source] at he he'
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²¹ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝²⁰ : NormedAddCommGroup EB
    inst✝¹⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁵ : NormedAddCommGroup EM
    inst✝¹⁴ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹³ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹² : TopologicalSpace M
    inst✝¹¹ : ChartedSpace HM M
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : MemTrivializationAtlas e
    inst✝ : MemTrivializationAtlas e'
    s : Set M
    x : M
    f : M → Bundle.TotalSpace F E
    hp : ContMDiffWithinAt IM IB n (Function.comp Bundle.TotalSpace.proj f) s x
    hf : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n (fun y => (↑e (f y)).2) …
    he : Membership.mem e.baseSet (f x).proj
    he' : Membership.mem e'.baseSet (f x).proj
    ⊢ ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n (fun y => (↑e' (f y)).2) s x
  -/
  refine (hp.coordChange hf he he').congr_of_eventuallyEq ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : MemTrivializationAtlas e
      inst✝ : MemTrivializationAtlas e'
      s : Set M
      x : M
      f : M → Bundle.TotalSpace F E
      hp : ContMDiffWithinAt IM IB n (Function.comp Bundle.TotalSpace.proj f) s x
      hf : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n (fun y => (↑e (f y)).2) …
      he : Membership.mem e.baseSet (f x).proj
      he' : Membership.mem e'.baseSet (f x).proj
      ⊢ (nhdsWithin x s).EventuallyEq (fun y => (↑e' (f y)).2) fun y => e.coordChang …
    -/
  · filter_upwards [hp.continuousWithinAt (e.open_baseSet.mem_nhds he)] with y hy
    /-
      case h
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : MemTrivializationAtlas e
      inst✝ : MemTrivializationAtlas e'
      s : Set M
      x : M
      f : M → Bundle.TotalSpace F E
      hp : ContMDiffWithinAt IM IB n (Function.comp Bundle.TotalSpace.proj f) s x
      hf : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n (fun y => (↑e (f y)).2) …
      he : Membership.mem e.baseSet (f x).proj
      he' : Membership.mem e'.baseSet (f x).proj
      y : M
      hy : Membership.mem (Set.preimage (Function.comp Bundle.TotalSpace.proj f) e.b …
      ⊢ Eq (↑e' (f y)).2 (e.coordChange e' (Function.comp Bundle.TotalSpace.proj f y …
    -/
    rw [Function.comp_apply, e.coordChange_apply_snd _ hy]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : MemTrivializationAtlas e
      inst✝ : MemTrivializationAtlas e'
      s : Set M
      x : M
      f : M → Bundle.TotalSpace F E
      hp : ContMDiffWithinAt IM IB n (Function.comp Bundle.TotalSpace.proj f) s x
      hf : ContMDiffWithinAt IM (modelWithCornersSelf 𝕜 F) n (fun y => (↑e (f y)).2) …
      he : Membership.mem e.baseSet (f x).proj
      he' : Membership.mem e'.baseSet (f x).proj
      ⊢ Eq (↑e' (f x)).2 (e.coordChange e' (Function.comp Bundle.TotalSpace.proj f x …
    -/
  · rw [Function.comp_apply, e.coordChange_apply_snd _ he]
    /-
      🎉 no goals
    -/


theorem Trivialization.contMDiffWithinAt_snd_comp_iff₂ {f : M → TotalSpace F E}
    (hp : ContMDiffWithinAt IM IB n (π F E ∘ f) s x)
    (he : f x ∈ e.source) (he' : f x ∈ e'.source) :
    ContMDiffWithinAt IM 𝓘(𝕜, F) n (fun y ↦ (e (f y)).2) s x ↔
      ContMDiffWithinAt IM 𝓘(𝕜, F) n (fun y ↦ (e' (f y)).2) s x :=
  ⟨(hp.change_section_trivialization · he he'), (hp.change_section_trivialization · he' he)⟩


variable [SmoothManifoldWithCorners IB B] in
/-- For a smooth vector bundle `E` over `B` with fiber modelled on `F`, the change-of-co-ordinates
between two trivializations `e`, `e'` for `E`, considered as charts to `B × F`, is smooth and
fiberwise linear. -/
instance SmoothFiberwiseLinear.hasGroupoid :
    HasGroupoid (TotalSpace F E) (smoothFiberwiseLinear B F IB) where
  compatible := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      inst✝ : SmoothManifoldWithCorners IB B
      ⊢ ∀ {e e' : PartialHomeomorph (Bundle.TotalSpace F E) (Prod B F)}, Membership. …
    -/
    rintro _ _ ⟨e, he, rfl⟩ ⟨e', he', rfl⟩
    /-
      case intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      inst✝ : SmoothManifoldWithCorners IB B
      e : Trivialization F Bundle.TotalSpace.proj
      he : Membership.mem (FiberBundle.trivializationAtlas F E) e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : Membership.mem (FiberBundle.trivializationAtlas F E) e'
      ⊢ Membership.mem (smoothFiberwiseLinear B F IB) (((fun e => e.toPartialHomeomo …
    -/
    haveI : MemTrivializationAtlas e := ⟨he⟩
    /-
      case intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      inst✝ : SmoothManifoldWithCorners IB B
      e : Trivialization F Bundle.TotalSpace.proj
      he : Membership.mem (FiberBundle.trivializationAtlas F E) e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : Membership.mem (FiberBundle.trivializationAtlas F E) e'
      this : MemTrivializationAtlas e
      ⊢ Membership.mem (smoothFiberwiseLinear B F IB) (((fun e => e.toPartialHomeomo …
    -/
    haveI : MemTrivializationAtlas e' := ⟨he'⟩
    /-
      case intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      inst✝ : SmoothManifoldWithCorners IB B
      e : Trivialization F Bundle.TotalSpace.proj
      he : Membership.mem (FiberBundle.trivializationAtlas F E) e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : Membership.mem (FiberBundle.trivializationAtlas F E) e'
      this✝ : MemTrivializationAtlas e
      this : MemTrivializationAtlas e'
      ⊢ Membership.mem (smoothFiberwiseLinear B F IB) (((fun e => e.toPartialHomeomo …
    -/
    rw [mem_smoothFiberwiseLinear_iff]
    refine ⟨_, _, e.open_baseSet.inter e'.open_baseSet, contMDiffOn_coordChangeL e e',
      contMDiffOn_symm_coordChangeL e e', ?_⟩
    /-
      case intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      inst✝ : SmoothManifoldWithCorners IB B
      e : Trivialization F Bundle.TotalSpace.proj
      he : Membership.mem (FiberBundle.trivializationAtlas F E) e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : Membership.mem (FiberBundle.trivializationAtlas F E) e'
      this✝ : MemTrivializationAtlas e
      this : MemTrivializationAtlas e'
      ⊢ (((fun e => e.toPartialHomeomorph) e).symm.trans ((fun e => e.toPartialHomeo …
    -/
    refine PartialHomeomorph.eqOnSourceSetoid.symm ⟨?_, ?_⟩
    · simp only [e.symm_trans_source_eq e', FiberwiseLinear.partialHomeomorph, trans_toPartialEquiv,
        symm_toPartialEquiv]
      /-
        case intro.intro.intro.intro.refine_2
        𝕜 : Type u_1
        B : Type u_2
        B' : Type u_3
        F : Type u_4
        M : Type u_5
        E : B → Type u_6
        inst✝²⁰ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝¹⁹ : NormedAddCommGroup EB
        inst✝¹⁸ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝¹⁷ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝¹⁶ : TopologicalSpace B
        inst✝¹⁵ : ChartedSpace HB B
        EM : Type u_9
        inst✝¹⁴ : NormedAddCommGroup EM
        inst✝¹³ : NormedSpace 𝕜 EM
        HM : Type u_10
        inst✝¹² : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        inst✝¹¹ : TopologicalSpace M
        inst✝¹⁰ : ChartedSpace HM M
        n : ENat
        inst✝⁹ : (x : B) → AddCommMonoid (E x)
        inst✝⁸ : (x : B) → Module 𝕜 (E x)
        inst✝⁷ : NormedAddCommGroup F
        inst✝⁶ : NormedSpace 𝕜 F
        inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝⁴ : (x : B) → TopologicalSpace (E x)
        inst✝³ : FiberBundle F E
        inst✝² : VectorBundle 𝕜 F E
        inst✝¹ : SmoothVectorBundle F E IB
        inst✝ : SmoothManifoldWithCorners IB B
        e : Trivialization F Bundle.TotalSpace.proj
        he : Membership.mem (FiberBundle.trivializationAtlas F E) e
        e' : Trivialization F Bundle.TotalSpace.proj
        he' : Membership.mem (FiberBundle.trivializationAtlas F E) e'
        this✝ : MemTrivializationAtlas e
        this : MemTrivializationAtlas e'
        ⊢ Set.EqOn (↑(FiberwiseLinear.partialHomeomorph (Trivialization.coordChangeL 𝕜 …
      -/
    · rintro ⟨b, v⟩ hb
      /-
        case intro.intro.intro.intro.refine_2.mk
        𝕜 : Type u_1
        B : Type u_2
        B' : Type u_3
        F : Type u_4
        M : Type u_5
        E : B → Type u_6
        inst✝²⁰ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝¹⁹ : NormedAddCommGroup EB
        inst✝¹⁸ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝¹⁷ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝¹⁶ : TopologicalSpace B
        inst✝¹⁵ : ChartedSpace HB B
        EM : Type u_9
        inst✝¹⁴ : NormedAddCommGroup EM
        inst✝¹³ : NormedSpace 𝕜 EM
        HM : Type u_10
        inst✝¹² : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        inst✝¹¹ : TopologicalSpace M
        inst✝¹⁰ : ChartedSpace HM M
        n : ENat
        inst✝⁹ : (x : B) → AddCommMonoid (E x)
        inst✝⁸ : (x : B) → Module 𝕜 (E x)
        inst✝⁷ : NormedAddCommGroup F
        inst✝⁶ : NormedSpace 𝕜 F
        inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝⁴ : (x : B) → TopologicalSpace (E x)
        inst✝³ : FiberBundle F E
        inst✝² : VectorBundle 𝕜 F E
        inst✝¹ : SmoothVectorBundle F E IB
        inst✝ : SmoothManifoldWithCorners IB B
        e : Trivialization F Bundle.TotalSpace.proj
        he : Membership.mem (FiberBundle.trivializationAtlas F E) e
        e' : Trivialization F Bundle.TotalSpace.proj
        he' : Membership.mem (FiberBundle.trivializationAtlas F E) e'
        this✝ : MemTrivializationAtlas e
        this : MemTrivializationAtlas e'
        b : B
        v : F
        hb : Membership.mem (FiberwiseLinear.partialHomeomorph (Trivialization.coordCh …
        ⊢ Eq (↑(FiberwiseLinear.partialHomeomorph (Trivialization.coordChangeL 𝕜 e e') …
      -/
      exact (e.apply_symm_apply_eq_coordChangeL e' hb.1 v).symm
      /-
        🎉 no goals
      -/


variable [SmoothManifoldWithCorners IB B] in
/-- A smooth vector bundle `E` is naturally a smooth manifold. -/
instance Bundle.TotalSpace.smoothManifoldWithCorners [SmoothManifoldWithCorners IB B] :
    SmoothManifoldWithCorners (IB.prod 𝓘(𝕜, F)) (TotalSpace F E) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    B' : Type u_3
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²¹ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝²⁰ : NormedAddCommGroup EB
    inst✝¹⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁵ : NormedAddCommGroup EM
    inst✝¹⁴ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹³ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹² : TopologicalSpace M
    inst✝¹¹ : ChartedSpace HM M
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
    ⊢ SmoothManifoldWithCorners (IB.prod (modelWithCornersSelf 𝕜 F)) (Bundle.Total …
  -/
  refine { StructureGroupoid.HasGroupoid.comp (smoothFiberwiseLinear B F IB) ?_ with }
  /-
    𝕜 : Type u_1
    B : Type u_2
    B' : Type u_3
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²¹ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝²⁰ : NormedAddCommGroup EB
    inst✝¹⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁵ : NormedAddCommGroup EM
    inst✝¹⁴ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹³ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹² : TopologicalSpace M
    inst✝¹¹ : ChartedSpace HM M
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
    ⊢ ∀ (e : PartialHomeomorph (Prod B F) (Prod B F)), Membership.mem (smoothFiber …
  -/
  intro e he
  /-
    𝕜 : Type u_1
    B : Type u_2
    B' : Type u_3
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²¹ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝²⁰ : NormedAddCommGroup EB
    inst✝¹⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁵ : NormedAddCommGroup EM
    inst✝¹⁴ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹³ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹² : TopologicalSpace M
    inst✝¹¹ : ChartedSpace HM M
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
    e : PartialHomeomorph (Prod B F) (Prod B F)
    he : Membership.mem (smoothFiberwiseLinear B F IB) e
    ⊢ ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) (IB.prod (modelWithCorn …
  -/
  rw [mem_smoothFiberwiseLinear_iff] at he
  /-
    𝕜 : Type u_1
    B : Type u_2
    B' : Type u_3
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²¹ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝²⁰ : NormedAddCommGroup EB
    inst✝¹⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁵ : NormedAddCommGroup EM
    inst✝¹⁴ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹³ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹² : TopologicalSpace M
    inst✝¹¹ : ChartedSpace HM M
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
    e : PartialHomeomorph (Prod B F) (Prod B F)
    he : Exists fun φ => Exists fun U => Exists fun hU => Exists fun hφ => Exists  …
    ⊢ ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) (IB.prod (modelWithCorn …
  -/
  obtain ⟨φ, U, hU, hφ, h2φ, heφ⟩ := he
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    B : Type u_2
    B' : Type u_3
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²¹ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝²⁰ : NormedAddCommGroup EB
    inst✝¹⁹ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁸ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁵ : NormedAddCommGroup EM
    inst✝¹⁴ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹³ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹² : TopologicalSpace M
    inst✝¹¹ : ChartedSpace HM M
    n : ENat
    inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
    inst✝⁹ : (x : B) → Module 𝕜 (E x)
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace 𝕜 F
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : (x : B) → TopologicalSpace (E x)
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜 F E
    inst✝² : SmoothVectorBundle F E IB
    inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
    e : PartialHomeomorph (Prod B F) (Prod B F)
    φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
    U : Set B
    hU : IsOpen U
    hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
    h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
    heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
    ⊢ ChartedSpace.LiftPropOn (contDiffGroupoid (↑Top.top) (IB.prod (modelWithCorn …
  -/
  rw [isLocalStructomorphOn_contDiffGroupoid_iff]
  refine ⟨ContMDiffOn.congr ?_ (EqOnSource.eqOn heφ),
      ContMDiffOn.congr ?_ (EqOnSource.eqOn (EqOnSource.symm' heφ))⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
    -/
  · rw [EqOnSource.source_eq heφ]
    /-
      case intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
    -/
    apply contMDiffOn_fst.prod_mk
    /-
      case intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (modelWithCornersSelf 𝕜 F)  …
    -/
    exact (hφ.comp contMDiffOn_fst <| prod_subset_preimage_fst _ _).clm_apply contMDiffOn_snd
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
    -/
  · rw [EqOnSource.target_eq heφ]
    /-
      case intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
    -/
    apply contMDiffOn_fst.prod_mk
    /-
      case intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²¹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝²⁰ : NormedAddCommGroup EB
      inst✝¹⁹ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁸ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁷ : TopologicalSpace B
      inst✝¹⁶ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁵ : NormedAddCommGroup EM
      inst✝¹⁴ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹³ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹² : TopologicalSpace M
      inst✝¹¹ : ChartedSpace HM M
      n : ENat
      inst✝¹⁰ : (x : B) → AddCommMonoid (E x)
      inst✝⁹ : (x : B) → Module 𝕜 (E x)
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedSpace 𝕜 F
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : (x : B) → TopologicalSpace (E x)
      inst✝⁴ : FiberBundle F E
      inst✝³ : VectorBundle 𝕜 F E
      inst✝² : SmoothVectorBundle F E IB
      inst✝¹ inst✝ : SmoothManifoldWithCorners IB B
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (modelWithCornersSelf 𝕜 F)  …
    -/
    exact (h2φ.comp contMDiffOn_fst <| prod_subset_preimage_fst _ _).clm_apply contMDiffOn_snd
    /-
      🎉 no goals
    -/


theorem Trivialization.contMDiffWithinAt_iff {f : M → TotalSpace F E} {s : Set M} {x₀ : M}
    (he : f x₀ ∈ e.source) :
    ContMDiffWithinAt IM (IB.prod 𝓘(𝕜, F)) n f s x₀ ↔
      ContMDiffWithinAt IM IB n (fun x => (f x).proj) s x₀ ∧
      ContMDiffWithinAt IM 𝓘(𝕜, F) n (fun x ↦ (e (f x)).2) s x₀ :=
  (contMDiffWithinAt_totalSpace _).trans <| and_congr_right fun h ↦
    Trivialization.contMDiffWithinAt_snd_comp_iff₂ h FiberBundle.mem_trivializationAt_proj_source he


theorem Trivialization.contMDiffAt_iff {f : M → TotalSpace F E} {x₀ : M} (he : f x₀ ∈ e.source) :
    ContMDiffAt IM (IB.prod 𝓘(𝕜, F)) n f x₀ ↔
      ContMDiffAt IM IB n (fun x => (f x).proj) x₀ ∧
      ContMDiffAt IM 𝓘(𝕜, F) n (fun x ↦ (e (f x)).2) x₀ :=
  e.contMDiffWithinAt_iff he


theorem Trivialization.contMDiffOn_iff {f : M → TotalSpace F E} {s : Set M}
    (he : MapsTo f s e.source) :
    ContMDiffOn IM (IB.prod 𝓘(𝕜, F)) n f s ↔
      ContMDiffOn IM IB n (fun x => (f x).proj) s ∧
      ContMDiffOn IM 𝓘(𝕜, F) n (fun x ↦ (e (f x)).2) s := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²⁰ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁹ : NormedAddCommGroup EB
    inst✝¹⁸ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁷ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁶ : TopologicalSpace B
    inst✝¹⁵ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁴ : NormedAddCommGroup EM
    inst✝¹³ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : ChartedSpace HM M
    n : ENat
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module 𝕜 (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    inst✝³ : FiberBundle F E
    inst✝² : VectorBundle 𝕜 F E
    inst✝¹ : SmoothVectorBundle F E IB
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : MemTrivializationAtlas e
    f : M → Bundle.TotalSpace F E
    s : Set M
    he : Set.MapsTo f s e.source
    ⊢ Iff (ContMDiffOn IM (IB.prod (modelWithCornersSelf 𝕜 F)) n f s) (And (ContMD …
  -/
  simp only [ContMDiffOn, ← forall_and]
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    M : Type u_5
    E : B → Type u_6
    inst✝²⁰ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁹ : NormedAddCommGroup EB
    inst✝¹⁸ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹⁷ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹⁶ : TopologicalSpace B
    inst✝¹⁵ : ChartedSpace HB B
    EM : Type u_9
    inst✝¹⁴ : NormedAddCommGroup EM
    inst✝¹³ : NormedSpace 𝕜 EM
    HM : Type u_10
    inst✝¹² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : ChartedSpace HM M
    n : ENat
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module 𝕜 (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    inst✝³ : FiberBundle F E
    inst✝² : VectorBundle 𝕜 F E
    inst✝¹ : SmoothVectorBundle F E IB
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : MemTrivializationAtlas e
    f : M → Bundle.TotalSpace F E
    s : Set M
    he : Set.MapsTo f s e.source
    ⊢ Iff (∀ (x : M), Membership.mem s x → ContMDiffWithinAt IM (IB.prod (modelWit …
  -/
  exact forall₂_congr fun x hx ↦ e.contMDiffWithinAt_iff (he hx)
  /-
    🎉 no goals
  -/


theorem Trivialization.contMDiff_iff {f : M → TotalSpace F E} (he : ∀ x, f x ∈ e.source) :
    ContMDiff IM (IB.prod 𝓘(𝕜, F)) n f ↔
      ContMDiff IM IB n (fun x => (f x).proj) ∧
      ContMDiff IM 𝓘(𝕜, F) n (fun x ↦ (e (f x)).2) :=
  (forall_congr' fun x ↦ e.contMDiffAt_iff (he x)).trans forall_and


@[deprecated (since := "2024-11-21")]
alias Trivialization.smoothWithinAt_iff := Trivialization.contMDiffWithinAt_iff


@[deprecated (since := "2024-11-21")]
alias Trivialization.smoothAt_iff := Trivialization.contMDiffAt_iff


@[deprecated (since := "2024-11-21")]
alias Trivialization.smoothOn_iff := Trivialization.contMDiffOn_iff


@[deprecated (since := "2024-11-21")]
alias Trivialization.smooth_iff := Trivialization.contMDiff_iff


theorem Trivialization.contMDiffOn (e : Trivialization F (π F E)) [MemTrivializationAtlas e] :
    ContMDiffOn (IB.prod 𝓘(𝕜, F)) (IB.prod 𝓘(𝕜, F)) ⊤ e e.source := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁴ : NormedAddCommGroup EB
    inst✝¹³ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹² : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : ChartedSpace HB B
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module 𝕜 (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    inst✝³ : FiberBundle F E
    inst✝² : VectorBundle 𝕜 F E
    inst✝¹ : SmoothVectorBundle F E IB
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : MemTrivializationAtlas e
    ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
  -/
  have : ContMDiffOn (IB.prod 𝓘(𝕜, F)) (IB.prod 𝓘(𝕜, F)) ⊤ id e.source := contMDiffOn_id
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁴ : NormedAddCommGroup EB
    inst✝¹³ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹² : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : ChartedSpace HB B
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module 𝕜 (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    inst✝³ : FiberBundle F E
    inst✝² : VectorBundle 𝕜 F E
    inst✝¹ : SmoothVectorBundle F E IB
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : MemTrivializationAtlas e
    this : ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCor …
    ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
  -/
  rw [e.contMDiffOn_iff (mapsTo_id _)] at this
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁴ : NormedAddCommGroup EB
    inst✝¹³ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹² : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : ChartedSpace HB B
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module 𝕜 (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    inst✝³ : FiberBundle F E
    inst✝² : VectorBundle 𝕜 F E
    inst✝¹ : SmoothVectorBundle F E IB
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : MemTrivializationAtlas e
    this : And (ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) IB Top.top (fun x …
    ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
  -/
  exact (this.1.prod_mk this.2).congr fun x hx ↦ (e.mk_proj_snd hx).symm
  /-
    🎉 no goals
  -/


theorem Trivialization.contMDiffOn_symm (e : Trivialization F (π F E)) [MemTrivializationAtlas e] :
    ContMDiffOn (IB.prod 𝓘(𝕜, F)) (IB.prod 𝓘(𝕜, F)) ⊤ e.toPartialHomeomorph.symm e.target := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁴ : NormedAddCommGroup EB
    inst✝¹³ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹² : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : ChartedSpace HB B
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module 𝕜 (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    inst✝³ : FiberBundle F E
    inst✝² : VectorBundle 𝕜 F E
    inst✝¹ : SmoothVectorBundle F E IB
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : MemTrivializationAtlas e
    ⊢ ContMDiffOn (IB.prod (modelWithCornersSelf 𝕜 F)) (IB.prod (modelWithCornersS …
  -/
  rw [e.contMDiffOn_iff e.toPartialHomeomorph.symm_mapsTo]
  refine ⟨contMDiffOn_fst.congr fun x hx ↦ e.proj_symm_apply hx,
    contMDiffOn_snd.congr fun x hx ↦ ?_⟩
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝¹⁴ : NormedAddCommGroup EB
    inst✝¹³ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝¹² : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : ChartedSpace HB B
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module 𝕜 (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : (x : B) → TopologicalSpace (E x)
    inst✝³ : FiberBundle F E
    inst✝² : VectorBundle 𝕜 F E
    inst✝¹ : SmoothVectorBundle F E IB
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : MemTrivializationAtlas e
    x : Prod B F
    hx : Membership.mem e.target x
    ⊢ Eq (↑e (↑e.symm x)).2 x.2
  -/
  rw [e.apply_symm_apply hx]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-21")] alias Trivialization.smoothOn := Trivialization.contMDiffOn


@[deprecated (since := "2024-11-21")]
alias Trivialization.smoothOn_symm := Trivialization.contMDiffOn_symm


/-- Mixin for a `VectorBundleCore` stating smoothness (of transition functions). -/
class IsSmooth (IB : ModelWithCorners 𝕜 EB HB) : Prop where
  contMDiffOn_coordChange :
    ∀ i j, ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (Z.coordChange i j) (Z.baseSet i ∩ Z.baseSet j)


theorem contMDiffOn_coordChange (IB : ModelWithCorners 𝕜 EB HB) [h : Z.IsSmooth IB] (i j : ι) :
    ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (Z.coordChange i j) (Z.baseSet i ∩ Z.baseSet j) :=
  h.1 i j


@[deprecated (since := "2024-11-21")]
alias smoothOn_coordChange := contMDiffOn_coordChange


/-- If a `VectorBundleCore` has the `IsSmooth` mixin, then the vector bundle constructed from it
is a smooth vector bundle. -/
instance smoothVectorBundle : SmoothVectorBundle F Z.Fiber IB where
  contMDiffOn_coordChangeL := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      ι : Type u_11
      Z : VectorBundleCore 𝕜 B F ι
      inst✝ : Z.IsSmooth IB
      ⊢ ∀ (e e' : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivialization …
    -/
    rintro - - ⟨i, rfl⟩ ⟨i', rfl⟩
    /-
      case mk.intro.mk.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      ι : Type u_11
      Z : VectorBundleCore 𝕜 B F ι
      inst✝ : Z.IsSmooth IB
      i i' : ι
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
    -/
    refine (Z.contMDiffOn_coordChange IB i i').congr fun b hb ↦ ?_
    /-
      case mk.intro.mk.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      ι : Type u_11
      Z : VectorBundleCore 𝕜 B F ι
      inst✝ : Z.IsSmooth IB
      i i' : ι
      b : B
      hb : Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet i')) b
      ⊢ Eq (↑(Trivialization.coordChangeL 𝕜 (Z.toFiberBundleCore.localTriv i) (Z.toF …
    -/
    ext v
    /-
      case mk.intro.mk.intro.h
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝²⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁹ : NormedAddCommGroup EB
      inst✝¹⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁶ : TopologicalSpace B
      inst✝¹⁵ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹⁴ : NormedAddCommGroup EM
      inst✝¹³ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹¹ : TopologicalSpace M
      inst✝¹⁰ : ChartedSpace HM M
      n : ENat
      inst✝⁹ : (x : B) → AddCommMonoid (E x)
      inst✝⁸ : (x : B) → Module 𝕜 (E x)
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace 𝕜 F
      inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁴ : (x : B) → TopologicalSpace (E x)
      inst✝³ : FiberBundle F E
      inst✝² : VectorBundle 𝕜 F E
      inst✝¹ : SmoothVectorBundle F E IB
      ι : Type u_11
      Z : VectorBundleCore 𝕜 B F ι
      inst✝ : Z.IsSmooth IB
      i i' : ι
      b : B
      hb : Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet i')) b
      v : F
      ⊢ Eq (↑(Trivialization.coordChangeL 𝕜 (Z.toFiberBundleCore.localTriv i) (Z.toF …
    -/
    exact Z.localTriv_coordChange_eq i i' hb v
    /-
      🎉 no goals
    -/


/-- A trivial vector bundle over a smooth manifold is a smooth vector bundle. -/
instance Bundle.Trivial.smoothVectorBundle : SmoothVectorBundle F (Bundle.Trivial B F) IB where
  contMDiffOn_coordChangeL := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁸ : NormedAddCommGroup EB
      inst✝¹⁷ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁶ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁵ : TopologicalSpace B
      inst✝¹⁴ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹³ : NormedAddCommGroup EM
      inst✝¹² : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹¹ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹⁰ : TopologicalSpace M
      inst✝⁹ : ChartedSpace HM M
      n : ENat
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module 𝕜 (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (x : B) → TopologicalSpace (E x)
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      ⊢ ∀ (e e' : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivialization …
    -/
    intro e e' he he'
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁸ : NormedAddCommGroup EB
      inst✝¹⁷ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁶ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁵ : TopologicalSpace B
      inst✝¹⁴ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹³ : NormedAddCommGroup EM
      inst✝¹² : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹¹ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹⁰ : TopologicalSpace M
      inst✝⁹ : ChartedSpace HM M
      n : ENat
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module 𝕜 (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (x : B) → TopologicalSpace (E x)
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      e e' : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      he' : MemTrivializationAtlas e'
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
    -/
    obtain rfl := Bundle.Trivial.eq_trivialization B F e
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁸ : NormedAddCommGroup EB
      inst✝¹⁷ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁶ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁵ : TopologicalSpace B
      inst✝¹⁴ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹³ : NormedAddCommGroup EM
      inst✝¹² : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹¹ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹⁰ : TopologicalSpace M
      inst✝⁹ : ChartedSpace HM M
      n : ENat
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module 𝕜 (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (x : B) → TopologicalSpace (E x)
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      he : MemTrivializationAtlas (Bundle.Trivial.trivialization B F)
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
    -/
    obtain rfl := Bundle.Trivial.eq_trivialization B F e'
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁸ : NormedAddCommGroup EB
      inst✝¹⁷ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁶ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁵ : TopologicalSpace B
      inst✝¹⁴ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹³ : NormedAddCommGroup EM
      inst✝¹² : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹¹ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹⁰ : TopologicalSpace M
      inst✝⁹ : ChartedSpace HM M
      n : ENat
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module 𝕜 (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (x : B) → TopologicalSpace (E x)
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      he he' : MemTrivializationAtlas (Bundle.Trivial.trivialization B F)
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
    -/
    simp_rw [Bundle.Trivial.trivialization.coordChangeL]
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝¹⁸ : NormedAddCommGroup EB
      inst✝¹⁷ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝¹⁶ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝¹⁵ : TopologicalSpace B
      inst✝¹⁴ : ChartedSpace HB B
      EM : Type u_9
      inst✝¹³ : NormedAddCommGroup EM
      inst✝¹² : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝¹¹ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝¹⁰ : TopologicalSpace M
      inst✝⁹ : ChartedSpace HM M
      n : ENat
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module 𝕜 (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (x : B) → TopologicalSpace (E x)
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      he he' : MemTrivializationAtlas (Bundle.Trivial.trivialization B F)
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
    -/
    exact contMDiff_const.contMDiffOn
    /-
      🎉 no goals
    -/


/-- The direct sum of two smooth vector bundles over the same base is a smooth vector bundle. -/
instance Bundle.Prod.smoothVectorBundle : SmoothVectorBundle (F₁ × F₂) (E₁ ×ᵇ E₂) IB where
  contMDiffOn_coordChangeL := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝³⁸ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝³⁷ : NormedAddCommGroup EB
      inst✝³⁶ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝³⁵ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝³⁴ : TopologicalSpace B
      inst✝³³ : ChartedSpace HB B
      EM : Type u_9
      inst✝³² : NormedAddCommGroup EM
      inst✝³¹ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝³⁰ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝²⁹ : TopologicalSpace M
      inst✝²⁸ : ChartedSpace HM M
      n : ENat
      inst✝²⁷ : (x : B) → AddCommMonoid (E x)
      inst✝²⁶ : (x : B) → Module 𝕜 (E x)
      inst✝²⁵ : NormedAddCommGroup F
      inst✝²⁴ : NormedSpace 𝕜 F
      inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝²² : (x : B) → TopologicalSpace (E x)
      inst✝²¹ : FiberBundle F E
      inst✝²⁰ : VectorBundle 𝕜 F E
      inst✝¹⁹ : SmoothVectorBundle F E IB
      F₁ : Type u_11
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜 F₁
      E₁ : B → Type u_12
      inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
      inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
      F₂ : Type u_13
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜 F₂
      E₂ : B → Type u_14
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
      inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : FiberBundle F₂ E₂
      inst✝⁴ : VectorBundle 𝕜 F₁ E₁
      inst✝³ : VectorBundle 𝕜 F₂ E₂
      inst✝² : SmoothVectorBundle F₁ E₁ IB
      inst✝¹ : SmoothVectorBundle F₂ E₂ IB
      inst✝ : SmoothManifoldWithCorners IB B
      ⊢ ∀ (e e' : Trivialization (Prod F₁ F₂) Bundle.TotalSpace.proj) [inst : MemTri …
    -/
    rintro _ _ ⟨e₁, e₂, i₁, i₂, rfl⟩ ⟨e₁', e₂', i₁', i₂', rfl⟩
    /-
      case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝³⁸ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝³⁷ : NormedAddCommGroup EB
      inst✝³⁶ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝³⁵ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝³⁴ : TopologicalSpace B
      inst✝³³ : ChartedSpace HB B
      EM : Type u_9
      inst✝³² : NormedAddCommGroup EM
      inst✝³¹ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝³⁰ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝²⁹ : TopologicalSpace M
      inst✝²⁸ : ChartedSpace HM M
      n : ENat
      inst✝²⁷ : (x : B) → AddCommMonoid (E x)
      inst✝²⁶ : (x : B) → Module 𝕜 (E x)
      inst✝²⁵ : NormedAddCommGroup F
      inst✝²⁴ : NormedSpace 𝕜 F
      inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝²² : (x : B) → TopologicalSpace (E x)
      inst✝²¹ : FiberBundle F E
      inst✝²⁰ : VectorBundle 𝕜 F E
      inst✝¹⁹ : SmoothVectorBundle F E IB
      F₁ : Type u_11
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜 F₁
      E₁ : B → Type u_12
      inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
      inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
      F₂ : Type u_13
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜 F₂
      E₂ : B → Type u_14
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
      inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : FiberBundle F₂ E₂
      inst✝⁴ : VectorBundle 𝕜 F₁ E₁
      inst✝³ : VectorBundle 𝕜 F₂ E₂
      inst✝² : SmoothVectorBundle F₁ E₁ IB
      inst✝¹ : SmoothVectorBundle F₂ E₂ IB
      inst✝ : SmoothManifoldWithCorners IB B
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      i₁ : MemTrivializationAtlas e₁
      i₂ : MemTrivializationAtlas e₂
      e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      i₁' : MemTrivializationAtlas e₁'
      i₂' : MemTrivializationAtlas e₂'
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) ( …
    -/
    refine ContMDiffOn.congr ?_ (e₁.coordChangeL_prod 𝕜 e₁' e₂ e₂')
    /-
      case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      M : Type u_5
      E : B → Type u_6
      inst✝³⁸ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝³⁷ : NormedAddCommGroup EB
      inst✝³⁶ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝³⁵ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝³⁴ : TopologicalSpace B
      inst✝³³ : ChartedSpace HB B
      EM : Type u_9
      inst✝³² : NormedAddCommGroup EM
      inst✝³¹ : NormedSpace 𝕜 EM
      HM : Type u_10
      inst✝³⁰ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      inst✝²⁹ : TopologicalSpace M
      inst✝²⁸ : ChartedSpace HM M
      n : ENat
      inst✝²⁷ : (x : B) → AddCommMonoid (E x)
      inst✝²⁶ : (x : B) → Module 𝕜 (E x)
      inst✝²⁵ : NormedAddCommGroup F
      inst✝²⁴ : NormedSpace 𝕜 F
      inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝²² : (x : B) → TopologicalSpace (E x)
      inst✝²¹ : FiberBundle F E
      inst✝²⁰ : VectorBundle 𝕜 F E
      inst✝¹⁹ : SmoothVectorBundle F E IB
      F₁ : Type u_11
      inst✝¹⁸ : NormedAddCommGroup F₁
      inst✝¹⁷ : NormedSpace 𝕜 F₁
      E₁ : B → Type u_12
      inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
      inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
      F₂ : Type u_13
      inst✝¹³ : NormedAddCommGroup F₂
      inst✝¹² : NormedSpace 𝕜 F₂
      E₂ : B → Type u_14
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
      inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
      inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
      inst✝⁶ : FiberBundle F₁ E₁
      inst✝⁵ : FiberBundle F₂ E₂
      inst✝⁴ : VectorBundle 𝕜 F₁ E₁
      inst✝³ : VectorBundle 𝕜 F₂ E₂
      inst✝² : SmoothVectorBundle F₁ E₁ IB
      inst✝¹ : SmoothVectorBundle F₂ E₂ IB
      inst✝ : SmoothManifoldWithCorners IB B
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      i₁ : MemTrivializationAtlas e₁
      i₂ : MemTrivializationAtlas e₂
      e₁' : Trivialization F₁ Bundle.TotalSpace.proj
      e₂' : Trivialization F₂ Bundle.TotalSpace.proj
      i₁' : MemTrivializationAtlas e₁'
      i₂' : MemTrivializationAtlas e₂'
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) ( …
    -/
    refine ContMDiffOn.clm_prodMap ?_ ?_
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_1
        𝕜 : Type u_1
        B : Type u_2
        B' : Type u_3
        F : Type u_4
        M : Type u_5
        E : B → Type u_6
        inst✝³⁸ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝³⁷ : NormedAddCommGroup EB
        inst✝³⁶ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝³⁵ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝³⁴ : TopologicalSpace B
        inst✝³³ : ChartedSpace HB B
        EM : Type u_9
        inst✝³² : NormedAddCommGroup EM
        inst✝³¹ : NormedSpace 𝕜 EM
        HM : Type u_10
        inst✝³⁰ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        inst✝²⁹ : TopologicalSpace M
        inst✝²⁸ : ChartedSpace HM M
        n : ENat
        inst✝²⁷ : (x : B) → AddCommMonoid (E x)
        inst✝²⁶ : (x : B) → Module 𝕜 (E x)
        inst✝²⁵ : NormedAddCommGroup F
        inst✝²⁴ : NormedSpace 𝕜 F
        inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝²² : (x : B) → TopologicalSpace (E x)
        inst✝²¹ : FiberBundle F E
        inst✝²⁰ : VectorBundle 𝕜 F E
        inst✝¹⁹ : SmoothVectorBundle F E IB
        F₁ : Type u_11
        inst✝¹⁸ : NormedAddCommGroup F₁
        inst✝¹⁷ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_12
        inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
        inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
        F₂ : Type u_13
        inst✝¹³ : NormedAddCommGroup F₂
        inst✝¹² : NormedSpace 𝕜 F₂
        E₂ : B → Type u_14
        inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
        inst✝⁶ : FiberBundle F₁ E₁
        inst✝⁵ : FiberBundle F₂ E₂
        inst✝⁴ : VectorBundle 𝕜 F₁ E₁
        inst✝³ : VectorBundle 𝕜 F₂ E₂
        inst✝² : SmoothVectorBundle F₁ E₁ IB
        inst✝¹ : SmoothVectorBundle F₂ E₂ IB
        inst✝ : SmoothManifoldWithCorners IB B
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        i₁ : MemTrivializationAtlas e₁
        i₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        i₁' : MemTrivializationAtlas e₁'
        i₂' : MemTrivializationAtlas e₂'
        ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
      -/
    · refine (contMDiffOn_coordChangeL e₁ e₁').mono ?_
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_1
        𝕜 : Type u_1
        B : Type u_2
        B' : Type u_3
        F : Type u_4
        M : Type u_5
        E : B → Type u_6
        inst✝³⁸ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝³⁷ : NormedAddCommGroup EB
        inst✝³⁶ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝³⁵ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝³⁴ : TopologicalSpace B
        inst✝³³ : ChartedSpace HB B
        EM : Type u_9
        inst✝³² : NormedAddCommGroup EM
        inst✝³¹ : NormedSpace 𝕜 EM
        HM : Type u_10
        inst✝³⁰ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        inst✝²⁹ : TopologicalSpace M
        inst✝²⁸ : ChartedSpace HM M
        n : ENat
        inst✝²⁷ : (x : B) → AddCommMonoid (E x)
        inst✝²⁶ : (x : B) → Module 𝕜 (E x)
        inst✝²⁵ : NormedAddCommGroup F
        inst✝²⁴ : NormedSpace 𝕜 F
        inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝²² : (x : B) → TopologicalSpace (E x)
        inst✝²¹ : FiberBundle F E
        inst✝²⁰ : VectorBundle 𝕜 F E
        inst✝¹⁹ : SmoothVectorBundle F E IB
        F₁ : Type u_11
        inst✝¹⁸ : NormedAddCommGroup F₁
        inst✝¹⁷ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_12
        inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
        inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
        F₂ : Type u_13
        inst✝¹³ : NormedAddCommGroup F₂
        inst✝¹² : NormedSpace 𝕜 F₂
        E₂ : B → Type u_14
        inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
        inst✝⁶ : FiberBundle F₁ E₁
        inst✝⁵ : FiberBundle F₂ E₂
        inst✝⁴ : VectorBundle 𝕜 F₁ E₁
        inst✝³ : VectorBundle 𝕜 F₂ E₂
        inst✝² : SmoothVectorBundle F₁ E₁ IB
        inst✝¹ : SmoothVectorBundle F₂ E₂ IB
        inst✝ : SmoothManifoldWithCorners IB B
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        i₁ : MemTrivializationAtlas e₁
        i₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        i₁' : MemTrivializationAtlas e₁'
        i₂' : MemTrivializationAtlas e₂'
        ⊢ HasSubset.Subset (Inter.inter (e₁.prod e₂).baseSet (e₁'.prod e₂').baseSet) ( …
      -/
      simp only [Trivialization.baseSet_prod, mfld_simps]
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_1
        𝕜 : Type u_1
        B : Type u_2
        B' : Type u_3
        F : Type u_4
        M : Type u_5
        E : B → Type u_6
        inst✝³⁸ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝³⁷ : NormedAddCommGroup EB
        inst✝³⁶ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝³⁵ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝³⁴ : TopologicalSpace B
        inst✝³³ : ChartedSpace HB B
        EM : Type u_9
        inst✝³² : NormedAddCommGroup EM
        inst✝³¹ : NormedSpace 𝕜 EM
        HM : Type u_10
        inst✝³⁰ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        inst✝²⁹ : TopologicalSpace M
        inst✝²⁸ : ChartedSpace HM M
        n : ENat
        inst✝²⁷ : (x : B) → AddCommMonoid (E x)
        inst✝²⁶ : (x : B) → Module 𝕜 (E x)
        inst✝²⁵ : NormedAddCommGroup F
        inst✝²⁴ : NormedSpace 𝕜 F
        inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝²² : (x : B) → TopologicalSpace (E x)
        inst✝²¹ : FiberBundle F E
        inst✝²⁰ : VectorBundle 𝕜 F E
        inst✝¹⁹ : SmoothVectorBundle F E IB
        F₁ : Type u_11
        inst✝¹⁸ : NormedAddCommGroup F₁
        inst✝¹⁷ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_12
        inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
        inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
        F₂ : Type u_13
        inst✝¹³ : NormedAddCommGroup F₂
        inst✝¹² : NormedSpace 𝕜 F₂
        E₂ : B → Type u_14
        inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
        inst✝⁶ : FiberBundle F₁ E₁
        inst✝⁵ : FiberBundle F₂ E₂
        inst✝⁴ : VectorBundle 𝕜 F₁ E₁
        inst✝³ : VectorBundle 𝕜 F₂ E₂
        inst✝² : SmoothVectorBundle F₁ E₁ IB
        inst✝¹ : SmoothVectorBundle F₂ E₂ IB
        inst✝ : SmoothManifoldWithCorners IB B
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        i₁ : MemTrivializationAtlas e₁
        i₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        i₁' : MemTrivializationAtlas e₁'
        i₂' : MemTrivializationAtlas e₂'
        ⊢ HasSubset.Subset (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.int …
      -/
      mfld_set_tac
      /-
        🎉 no goals
      -/
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_2
        𝕜 : Type u_1
        B : Type u_2
        B' : Type u_3
        F : Type u_4
        M : Type u_5
        E : B → Type u_6
        inst✝³⁸ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝³⁷ : NormedAddCommGroup EB
        inst✝³⁶ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝³⁵ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝³⁴ : TopologicalSpace B
        inst✝³³ : ChartedSpace HB B
        EM : Type u_9
        inst✝³² : NormedAddCommGroup EM
        inst✝³¹ : NormedSpace 𝕜 EM
        HM : Type u_10
        inst✝³⁰ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        inst✝²⁹ : TopologicalSpace M
        inst✝²⁸ : ChartedSpace HM M
        n : ENat
        inst✝²⁷ : (x : B) → AddCommMonoid (E x)
        inst✝²⁶ : (x : B) → Module 𝕜 (E x)
        inst✝²⁵ : NormedAddCommGroup F
        inst✝²⁴ : NormedSpace 𝕜 F
        inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝²² : (x : B) → TopologicalSpace (E x)
        inst✝²¹ : FiberBundle F E
        inst✝²⁰ : VectorBundle 𝕜 F E
        inst✝¹⁹ : SmoothVectorBundle F E IB
        F₁ : Type u_11
        inst✝¹⁸ : NormedAddCommGroup F₁
        inst✝¹⁷ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_12
        inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
        inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
        F₂ : Type u_13
        inst✝¹³ : NormedAddCommGroup F₂
        inst✝¹² : NormedSpace 𝕜 F₂
        E₂ : B → Type u_14
        inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
        inst✝⁶ : FiberBundle F₁ E₁
        inst✝⁵ : FiberBundle F₂ E₂
        inst✝⁴ : VectorBundle 𝕜 F₁ E₁
        inst✝³ : VectorBundle 𝕜 F₂ E₂
        inst✝² : SmoothVectorBundle F₁ E₁ IB
        inst✝¹ : SmoothVectorBundle F₂ E₂ IB
        inst✝ : SmoothManifoldWithCorners IB B
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        i₁ : MemTrivializationAtlas e₁
        i₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        i₁' : MemTrivializationAtlas e₁'
        i₂' : MemTrivializationAtlas e₂'
        ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
      -/
    · refine (contMDiffOn_coordChangeL e₂ e₂').mono ?_
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_2
        𝕜 : Type u_1
        B : Type u_2
        B' : Type u_3
        F : Type u_4
        M : Type u_5
        E : B → Type u_6
        inst✝³⁸ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝³⁷ : NormedAddCommGroup EB
        inst✝³⁶ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝³⁵ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝³⁴ : TopologicalSpace B
        inst✝³³ : ChartedSpace HB B
        EM : Type u_9
        inst✝³² : NormedAddCommGroup EM
        inst✝³¹ : NormedSpace 𝕜 EM
        HM : Type u_10
        inst✝³⁰ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        inst✝²⁹ : TopologicalSpace M
        inst✝²⁸ : ChartedSpace HM M
        n : ENat
        inst✝²⁷ : (x : B) → AddCommMonoid (E x)
        inst✝²⁶ : (x : B) → Module 𝕜 (E x)
        inst✝²⁵ : NormedAddCommGroup F
        inst✝²⁴ : NormedSpace 𝕜 F
        inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝²² : (x : B) → TopologicalSpace (E x)
        inst✝²¹ : FiberBundle F E
        inst✝²⁰ : VectorBundle 𝕜 F E
        inst✝¹⁹ : SmoothVectorBundle F E IB
        F₁ : Type u_11
        inst✝¹⁸ : NormedAddCommGroup F₁
        inst✝¹⁷ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_12
        inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
        inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
        F₂ : Type u_13
        inst✝¹³ : NormedAddCommGroup F₂
        inst✝¹² : NormedSpace 𝕜 F₂
        E₂ : B → Type u_14
        inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
        inst✝⁶ : FiberBundle F₁ E₁
        inst✝⁵ : FiberBundle F₂ E₂
        inst✝⁴ : VectorBundle 𝕜 F₁ E₁
        inst✝³ : VectorBundle 𝕜 F₂ E₂
        inst✝² : SmoothVectorBundle F₁ E₁ IB
        inst✝¹ : SmoothVectorBundle F₂ E₂ IB
        inst✝ : SmoothManifoldWithCorners IB B
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        i₁ : MemTrivializationAtlas e₁
        i₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        i₁' : MemTrivializationAtlas e₁'
        i₂' : MemTrivializationAtlas e₂'
        ⊢ HasSubset.Subset (Inter.inter (e₁.prod e₂).baseSet (e₁'.prod e₂').baseSet) ( …
      -/
      simp only [Trivialization.baseSet_prod, mfld_simps]
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_2
        𝕜 : Type u_1
        B : Type u_2
        B' : Type u_3
        F : Type u_4
        M : Type u_5
        E : B → Type u_6
        inst✝³⁸ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝³⁷ : NormedAddCommGroup EB
        inst✝³⁶ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝³⁵ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝³⁴ : TopologicalSpace B
        inst✝³³ : ChartedSpace HB B
        EM : Type u_9
        inst✝³² : NormedAddCommGroup EM
        inst✝³¹ : NormedSpace 𝕜 EM
        HM : Type u_10
        inst✝³⁰ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        inst✝²⁹ : TopologicalSpace M
        inst✝²⁸ : ChartedSpace HM M
        n : ENat
        inst✝²⁷ : (x : B) → AddCommMonoid (E x)
        inst✝²⁶ : (x : B) → Module 𝕜 (E x)
        inst✝²⁵ : NormedAddCommGroup F
        inst✝²⁴ : NormedSpace 𝕜 F
        inst✝²³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝²² : (x : B) → TopologicalSpace (E x)
        inst✝²¹ : FiberBundle F E
        inst✝²⁰ : VectorBundle 𝕜 F E
        inst✝¹⁹ : SmoothVectorBundle F E IB
        F₁ : Type u_11
        inst✝¹⁸ : NormedAddCommGroup F₁
        inst✝¹⁷ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_12
        inst✝¹⁶ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        inst✝¹⁵ : (x : B) → AddCommMonoid (E₁ x)
        inst✝¹⁴ : (x : B) → Module 𝕜 (E₁ x)
        F₂ : Type u_13
        inst✝¹³ : NormedAddCommGroup F₂
        inst✝¹² : NormedSpace 𝕜 F₂
        E₂ : B → Type u_14
        inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝¹⁰ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁹ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁸ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁷ : (x : B) → TopologicalSpace (E₂ x)
        inst✝⁶ : FiberBundle F₁ E₁
        inst✝⁵ : FiberBundle F₂ E₂
        inst✝⁴ : VectorBundle 𝕜 F₁ E₁
        inst✝³ : VectorBundle 𝕜 F₂ E₂
        inst✝² : SmoothVectorBundle F₁ E₁ IB
        inst✝¹ : SmoothVectorBundle F₂ E₂ IB
        inst✝ : SmoothManifoldWithCorners IB B
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        i₁ : MemTrivializationAtlas e₁
        i₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        i₁' : MemTrivializationAtlas e₁'
        i₂' : MemTrivializationAtlas e₂'
        ⊢ HasSubset.Subset (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.int …
      -/
      mfld_set_tac
      /-
        🎉 no goals
      -/


variable (IB) in
/-- Mixin for a `VectorPrebundle` stating smoothness of coordinate changes. -/
class IsSmooth (a : VectorPrebundle 𝕜 F E) : Prop where
  exists_smoothCoordChange :
    ∀ᵉ (e ∈ a.pretrivializationAtlas) (e' ∈ a.pretrivializationAtlas),
      ∃ f : B → F →L[𝕜] F,
        ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ f (e.baseSet ∩ e'.baseSet) ∧
          ∀ (b : B) (_ : b ∈ e.baseSet ∩ e'.baseSet) (v : F),
            f b v = (e' ⟨b, e.symm b v⟩).2


variable (IB) in
/-- A randomly chosen coordinate change on a `SmoothVectorPrebundle`, given by
  the field `exists_coordChange`. Note that `a.smoothCoordChange` need not be the same as
  `a.coordChange`. -/
noncomputable def smoothCoordChange (he : e ∈ a.pretrivializationAtlas)
    (he' : e' ∈ a.pretrivializationAtlas) (b : B) : F →L[𝕜] F :=
  Classical.choose (ha.exists_smoothCoordChange e he e' he') b


theorem contMDiffOn_smoothCoordChange (he : e ∈ a.pretrivializationAtlas)
    (he' : e' ∈ a.pretrivializationAtlas) :
    ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (a.smoothCoordChange IB he he') (e.baseSet ∩ e'.baseSet) :=
  (Classical.choose_spec (ha.exists_smoothCoordChange e he e' he')).1


@[deprecated (since := "2024-11-21")]
alias smoothOn_smoothCoordChange := contMDiffOn_smoothCoordChange


theorem smoothCoordChange_apply (he : e ∈ a.pretrivializationAtlas)
    (he' : e' ∈ a.pretrivializationAtlas) {b : B} (hb : b ∈ e.baseSet ∩ e'.baseSet) (v : F) :
    a.smoothCoordChange IB he he' b v = (e' ⟨b, e.symm b v⟩).2 :=
  (Classical.choose_spec (ha.exists_smoothCoordChange e he e' he')).2 b hb v


theorem mk_smoothCoordChange (he : e ∈ a.pretrivializationAtlas)
    (he' : e' ∈ a.pretrivializationAtlas) {b : B} (hb : b ∈ e.baseSet ∩ e'.baseSet) (v : F) :
    (b, a.smoothCoordChange IB he he' b v) = e' ⟨b, e.symm b v⟩ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_4
    E : B → Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    EB : Type u_7
    inst✝⁹ : NormedAddCommGroup EB
    inst✝⁸ : NormedSpace 𝕜 EB
    HB : Type u_8
    inst✝⁷ : TopologicalSpace HB
    IB : ModelWithCorners 𝕜 EB HB
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : ChartedSpace HB B
    inst✝⁴ : (x : B) → AddCommMonoid (E x)
    inst✝³ : (x : B) → Module 𝕜 (E x)
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : VectorPrebundle 𝕜 F E
    ha : VectorPrebundle.IsSmooth IB a
    e e' : Pretrivialization F Bundle.TotalSpace.proj
    he : Membership.mem a.pretrivializationAtlas e
    he' : Membership.mem a.pretrivializationAtlas e'
    b : B
    hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
    v : F
    ⊢ Eq { fst := b, snd := (VectorPrebundle.smoothCoordChange IB a he he' b) v }  …
  -/
  ext
    /-
      case fst
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      E : B → Type u_6
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝⁹ : NormedAddCommGroup EB
      inst✝⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : ChartedSpace HB B
      inst✝⁴ : (x : B) → AddCommMonoid (E x)
      inst✝³ : (x : B) → Module 𝕜 (E x)
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : (x : B) → TopologicalSpace (E x)
      a : VectorPrebundle 𝕜 F E
      ha : VectorPrebundle.IsSmooth IB a
      e e' : Pretrivialization F Bundle.TotalSpace.proj
      he : Membership.mem a.pretrivializationAtlas e
      he' : Membership.mem a.pretrivializationAtlas e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      v : F
      ⊢ Eq { fst := b, snd := (VectorPrebundle.smoothCoordChange IB a he he' b) v }. …
    -/
  · rw [e.mk_symm hb.1 v, e'.coe_fst', e.proj_symm_apply' hb.1]
    /-
      case fst
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      E : B → Type u_6
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝⁹ : NormedAddCommGroup EB
      inst✝⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : ChartedSpace HB B
      inst✝⁴ : (x : B) → AddCommMonoid (E x)
      inst✝³ : (x : B) → Module 𝕜 (E x)
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : (x : B) → TopologicalSpace (E x)
      a : VectorPrebundle 𝕜 F E
      ha : VectorPrebundle.IsSmooth IB a
      e e' : Pretrivialization F Bundle.TotalSpace.proj
      he : Membership.mem a.pretrivializationAtlas e
      he' : Membership.mem a.pretrivializationAtlas e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      v : F
      ⊢ Membership.mem e'.baseSet (↑e.symm { fst := b, snd := v }).proj
    -/
    rw [e.proj_symm_apply' hb.1]; exact hb.2
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case snd
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_4
      E : B → Type u_6
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      EB : Type u_7
      inst✝⁹ : NormedAddCommGroup EB
      inst✝⁸ : NormedSpace 𝕜 EB
      HB : Type u_8
      inst✝⁷ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : ChartedSpace HB B
      inst✝⁴ : (x : B) → AddCommMonoid (E x)
      inst✝³ : (x : B) → Module 𝕜 (E x)
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : (x : B) → TopologicalSpace (E x)
      a : VectorPrebundle 𝕜 F E
      ha : VectorPrebundle.IsSmooth IB a
      e e' : Pretrivialization F Bundle.TotalSpace.proj
      he : Membership.mem a.pretrivializationAtlas e
      he' : Membership.mem a.pretrivializationAtlas e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      v : F
      ⊢ Eq { fst := b, snd := (VectorPrebundle.smoothCoordChange IB a he he' b) v }. …
    -/
  · exact a.smoothCoordChange_apply he he' hb v
    /-
      🎉 no goals
    -/


variable (IB) in
/-- Make a `SmoothVectorBundle` from a `SmoothVectorPrebundle`. -/
theorem smoothVectorBundle : @SmoothVectorBundle
    _ _ F E _ _ _ _ _ _ IB _ _ _ _ _ _ a.totalSpaceTopology _ a.toFiberBundle a.toVectorBundle :=
  letI := a.totalSpaceTopology; letI := a.toFiberBundle; letI := a.toVectorBundle
  { contMDiffOn_coordChangeL := by
      /-
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_4
        E : B → Type u_6
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝⁹ : NormedAddCommGroup EB
        inst✝⁸ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝⁷ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝⁶ : TopologicalSpace B
        inst✝⁵ : ChartedSpace HB B
        inst✝⁴ : (x : B) → AddCommMonoid (E x)
        inst✝³ : (x : B) → Module 𝕜 (E x)
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle 𝕜 F E
        ha : VectorPrebundle.IsSmooth IB a
        this✝¹ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this✝ : FiberBundle F E := a.toFiberBundle
        this : VectorBundle 𝕜 F E := VectorPrebundle.toVectorBundle a
        ⊢ ∀ (e e' : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivialization …
      -/
      rintro _ _ ⟨e, he, rfl⟩ ⟨e', he', rfl⟩
      /-
        case mk.intro.intro.mk.intro.intro
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_4
        E : B → Type u_6
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝⁹ : NormedAddCommGroup EB
        inst✝⁸ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝⁷ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝⁶ : TopologicalSpace B
        inst✝⁵ : ChartedSpace HB B
        inst✝⁴ : (x : B) → AddCommMonoid (E x)
        inst✝³ : (x : B) → Module 𝕜 (E x)
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle 𝕜 F E
        ha : VectorPrebundle.IsSmooth IB a
        this✝¹ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this✝ : FiberBundle F E := a.toFiberBundle
        this : VectorBundle 𝕜 F E := VectorPrebundle.toVectorBundle a
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
      -/
      refine (a.contMDiffOn_smoothCoordChange he he').congr ?_
      /-
        case mk.intro.intro.mk.intro.intro
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_4
        E : B → Type u_6
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝⁹ : NormedAddCommGroup EB
        inst✝⁸ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝⁷ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝⁶ : TopologicalSpace B
        inst✝⁵ : ChartedSpace HB B
        inst✝⁴ : (x : B) → AddCommMonoid (E x)
        inst✝³ : (x : B) → Module 𝕜 (E x)
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle 𝕜 F E
        ha : VectorPrebundle.IsSmooth IB a
        this✝¹ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this✝ : FiberBundle F E := a.toFiberBundle
        this : VectorBundle 𝕜 F E := VectorPrebundle.toVectorBundle a
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        ⊢ ∀ (y : B), Membership.mem (Inter.inter e.baseSet e'.baseSet) y → Eq (↑(Trivi …
      -/
      intro b hb
      /-
        case mk.intro.intro.mk.intro.intro
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_4
        E : B → Type u_6
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝⁹ : NormedAddCommGroup EB
        inst✝⁸ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝⁷ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝⁶ : TopologicalSpace B
        inst✝⁵ : ChartedSpace HB B
        inst✝⁴ : (x : B) → AddCommMonoid (E x)
        inst✝³ : (x : B) → Module 𝕜 (E x)
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle 𝕜 F E
        ha : VectorPrebundle.IsSmooth IB a
        this✝¹ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this✝ : FiberBundle F E := a.toFiberBundle
        this : VectorBundle 𝕜 F E := VectorPrebundle.toVectorBundle a
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        b : B
        hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
        ⊢ Eq (↑(Trivialization.coordChangeL 𝕜 (a.toFiberPrebundle.trivializationOfMemP …
      -/
      ext v
      rw [a.smoothCoordChange_apply he he' hb v, ContinuousLinearEquiv.coe_coe,
        Trivialization.coordChangeL_apply]
      /-
        case mk.intro.intro.mk.intro.intro.h
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_4
        E : B → Type u_6
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        EB : Type u_7
        inst✝⁹ : NormedAddCommGroup EB
        inst✝⁸ : NormedSpace 𝕜 EB
        HB : Type u_8
        inst✝⁷ : TopologicalSpace HB
        IB : ModelWithCorners 𝕜 EB HB
        inst✝⁶ : TopologicalSpace B
        inst✝⁵ : ChartedSpace HB B
        inst✝⁴ : (x : B) → AddCommMonoid (E x)
        inst✝³ : (x : B) → Module 𝕜 (E x)
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle 𝕜 F E
        ha : VectorPrebundle.IsSmooth IB a
        this✝¹ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this✝ : FiberBundle F E := a.toFiberBundle
        this : VectorBundle 𝕜 F E := VectorPrebundle.toVectorBundle a
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        b : B
        hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
        v : F
        ⊢ Eq (↑(a.toFiberPrebundle.trivializationOfMemPretrivializationAtlas he') { pr …
      -/
      exacts [rfl, hb] }
      /-
        🎉 no goals
      -/


