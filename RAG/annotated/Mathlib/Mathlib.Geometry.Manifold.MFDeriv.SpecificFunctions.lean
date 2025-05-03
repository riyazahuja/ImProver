protected theorem hasMFDerivWithinAt : HasMFDerivWithinAt 𝓘(𝕜, E) 𝓘(𝕜, E') f s x f :=
  f.hasFDerivWithinAt.hasMFDerivWithinAt


protected theorem hasMFDerivAt : HasMFDerivAt 𝓘(𝕜, E) 𝓘(𝕜, E') f x f :=
  f.hasFDerivAt.hasMFDerivAt


protected theorem mdifferentiableWithinAt : MDifferentiableWithinAt 𝓘(𝕜, E) 𝓘(𝕜, E') f s x :=
  f.differentiableWithinAt.mdifferentiableWithinAt


protected theorem mdifferentiableOn : MDifferentiableOn 𝓘(𝕜, E) 𝓘(𝕜, E') f s :=
  f.differentiableOn.mdifferentiableOn


protected theorem mdifferentiableAt : MDifferentiableAt 𝓘(𝕜, E) 𝓘(𝕜, E') f x :=
  f.differentiableAt.mdifferentiableAt


protected theorem mdifferentiable : MDifferentiable 𝓘(𝕜, E) 𝓘(𝕜, E') f :=
  f.differentiable.mdifferentiable


theorem mfderiv_eq : mfderiv 𝓘(𝕜, E) 𝓘(𝕜, E') f x = f :=
  f.hasMFDerivAt.mfderiv


theorem mfderivWithin_eq (hs : UniqueMDiffWithinAt 𝓘(𝕜, E) s x) :
    mfderivWithin 𝓘(𝕜, E) 𝓘(𝕜, E') f s x = f :=
  f.hasMFDerivWithinAt.mfderivWithin hs


protected theorem hasMFDerivWithinAt : HasMFDerivWithinAt 𝓘(𝕜, E) 𝓘(𝕜, E') f s x (f : E →L[𝕜] E') :=
  f.hasFDerivWithinAt.hasMFDerivWithinAt


protected theorem hasMFDerivAt : HasMFDerivAt 𝓘(𝕜, E) 𝓘(𝕜, E') f x (f : E →L[𝕜] E') :=
  f.hasFDerivAt.hasMFDerivAt


theorem mfderiv_eq : mfderiv 𝓘(𝕜, E) 𝓘(𝕜, E') f x = (f : E →L[𝕜] E') :=
  f.hasMFDerivAt.mfderiv


theorem mfderivWithin_eq (hs : UniqueMDiffWithinAt 𝓘(𝕜, E) s x) :
    mfderivWithin 𝓘(𝕜, E) 𝓘(𝕜, E') f s x = (f : E →L[𝕜] E') :=
  f.hasMFDerivWithinAt.mfderivWithin hs


theorem hasMFDerivAt_id (x : M) :
    HasMFDerivAt I I (@id M) x (ContinuousLinearMap.id 𝕜 (TangentSpace I x)) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    x : M
    ⊢ HasMFDerivAt I I id x (ContinuousLinearMap.id 𝕜 (TangentSpace I x))
  -/
  refine ⟨continuousAt_id, ?_⟩
  have : ∀ᶠ y in 𝓝[range I] (extChartAt I x) x, (extChartAt I x ∘ (extChartAt I x).symm) y = y := by
    apply Filter.mem_of_superset (extChartAt_target_mem_nhdsWithin x)
    mfld_set_tac
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    x : M
    this : Filter.Eventually (fun y => Eq (Function.comp (↑(extChartAt I x)) (↑(ex …
    ⊢ HasFDerivWithinAt (writtenInExtChartAt I I x id) (ContinuousLinearMap.id 𝕜 ( …
  -/
  apply HasFDerivWithinAt.congr_of_eventuallyEq (hasFDerivWithinAt_id _ _) this
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    x : M
    this : Filter.Eventually (fun y => Eq (Function.comp (↑(extChartAt I x)) (↑(ex …
    ⊢ Eq (Function.comp (↑(extChartAt I x)) (↑(extChartAt I x).symm) (↑(extChartAt …
  -/
  simp only [mfld_simps]
  /-
    🎉 no goals
  -/


theorem hasMFDerivWithinAt_id (s : Set M) (x : M) :
    HasMFDerivWithinAt I I (@id M) s x (ContinuousLinearMap.id 𝕜 (TangentSpace I x)) :=
  (hasMFDerivAt_id x).hasMFDerivWithinAt


theorem mdifferentiableAt_id : MDifferentiableAt I I (@id M) x :=
  (hasMFDerivAt_id x).mdifferentiableAt


theorem mdifferentiableWithinAt_id : MDifferentiableWithinAt I I (@id M) s x :=
  mdifferentiableAt_id.mdifferentiableWithinAt


theorem mdifferentiable_id : MDifferentiable I I (@id M) := fun _ => mdifferentiableAt_id


theorem mdifferentiableOn_id : MDifferentiableOn I I (@id M) s :=
  mdifferentiable_id.mdifferentiableOn


@[simp, mfld_simps]
theorem mfderiv_id : mfderiv I I (@id M) x = ContinuousLinearMap.id 𝕜 (TangentSpace I x) :=
  HasMFDerivAt.mfderiv (hasMFDerivAt_id x)


theorem mfderivWithin_id (hxs : UniqueMDiffWithinAt I s x) :
    mfderivWithin I I (@id M) s x = ContinuousLinearMap.id 𝕜 (TangentSpace I x) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    x : M
    hxs : UniqueMDiffWithinAt I s x
    ⊢ Eq (mfderivWithin I I id s x) (ContinuousLinearMap.id 𝕜 (TangentSpace I x))
  -/
  rw [MDifferentiable.mfderivWithin mdifferentiableAt_id hxs]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    x : M
    hxs : UniqueMDiffWithinAt I s x
    ⊢ Eq (mfderiv I I id x) (ContinuousLinearMap.id 𝕜 (TangentSpace I x))
  -/
  exact mfderiv_id
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
                                                               /-
                                                                 𝕜 : Type u_1
                                                                 inst✝⁵ : NontriviallyNormedField 𝕜
                                                                 E : Type u_2
                                                                 inst✝⁴ : NormedAddCommGroup E
                                                                 inst✝³ : NormedSpace 𝕜 E
                                                                 H : Type u_3
                                                                 inst✝² : TopologicalSpace H
                                                                 I : ModelWithCorners 𝕜 E H
                                                                 M : Type u_4
                                                                 inst✝¹ : TopologicalSpace M
                                                                 inst✝ : ChartedSpace H M
                                                                 ⊢ Eq (tangentMap I I id) id
                                                               -/
theorem tangentMap_id : tangentMap I I (id : M → M) = id := by ext1 ⟨x, v⟩; simp [tangentMap]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem tangentMapWithin_id {p : TangentBundle I M} (hs : UniqueMDiffWithinAt I s p.proj) :
    tangentMapWithin I I (id : M → M) s p = p := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    p : TangentBundle I M
    hs : UniqueMDiffWithinAt I s p.proj
    ⊢ Eq (tangentMapWithin I I id s p) p
  -/
  simp only [tangentMapWithin, id]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    p : TangentBundle I M
    hs : UniqueMDiffWithinAt I s p.proj
    ⊢ Eq { proj := p.proj, snd := (mfderivWithin I I id s p.proj) p.snd } p
  -/
  rw [mfderivWithin_id]
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      s : Set M
      p : TangentBundle I M
      hs : UniqueMDiffWithinAt I s p.proj
      ⊢ Eq { proj := p.proj, snd := (ContinuousLinearMap.id 𝕜 (TangentSpace I p.proj …
    -/
  · rcases p with ⟨⟩; rfl
                      /-
                        🎉 no goals
                      -/
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      s : Set M
      p : TangentBundle I M
      hs : UniqueMDiffWithinAt I s p.proj
      ⊢ UniqueMDiffWithinAt I s p.proj
    -/
  · exact hs
    /-
      🎉 no goals
    -/


theorem hasMFDerivAt_const (c : M') (x : M) :
    HasMFDerivAt I I' (fun _ : M => c) x (0 : TangentSpace I x →L[𝕜] TangentSpace I' c) := by
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
    c : M'
    x : M
    ⊢ HasMFDerivAt I I' (fun x => c) x 0
  -/
  refine ⟨continuous_const.continuousAt, ?_⟩
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
    c : M'
    x : M
    ⊢ HasFDerivWithinAt (writtenInExtChartAt I I' x fun x => c) 0 (Set.range ↑I) ( …
  -/
  simp only [writtenInExtChartAt, Function.comp_def, hasFDerivWithinAt_const]
  /-
    🎉 no goals
  -/


theorem hasMFDerivWithinAt_const (c : M') (s : Set M) (x : M) :
    HasMFDerivWithinAt I I' (fun _ : M => c) s x (0 : TangentSpace I x →L[𝕜] TangentSpace I' c) :=
  (hasMFDerivAt_const c x).hasMFDerivWithinAt


theorem mdifferentiableAt_const : MDifferentiableAt I I' (fun _ : M => c) x :=
  (hasMFDerivAt_const c x).mdifferentiableAt


theorem mdifferentiableWithinAt_const : MDifferentiableWithinAt I I' (fun _ : M => c) s x :=
  mdifferentiableAt_const.mdifferentiableWithinAt


theorem mdifferentiable_const : MDifferentiable I I' fun _ : M => c := fun _ =>
  mdifferentiableAt_const


theorem mdifferentiableOn_const : MDifferentiableOn I I' (fun _ : M => c) s :=
  mdifferentiable_const.mdifferentiableOn


@[simp, mfld_simps]
theorem mfderiv_const :
    mfderiv I I' (fun _ : M => c) x = (0 : TangentSpace I x →L[𝕜] TangentSpace I' c) :=
  HasMFDerivAt.mfderiv (hasMFDerivAt_const c x)


theorem mfderivWithin_const :
    mfderivWithin I I' (fun _ : M => c) s x = (0 : TangentSpace I x →L[𝕜] TangentSpace I' c) :=
  (hasMFDerivWithinAt_const _ _ _).mfderivWithin_eq_zero


theorem hasMFDerivAt_fst (x : M × M') :
    HasMFDerivAt (I.prod I') I Prod.fst x
      (ContinuousLinearMap.fst 𝕜 (TangentSpace I x.1) (TangentSpace I' x.2)) := by
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
    x : Prod M M'
    ⊢ HasMFDerivAt (I.prod I') I Prod.fst x (ContinuousLinearMap.fst 𝕜 (TangentSpa …
  -/
  refine ⟨continuous_fst.continuousAt, ?_⟩
  have :
    ∀ᶠ y in 𝓝[range (I.prod I')] extChartAt (I.prod I') x x,
      (extChartAt I x.1 ∘ Prod.fst ∘ (extChartAt (I.prod I') x).symm) y = y.1 := by
    /- porting note: was
    apply Filter.mem_of_superset (extChartAt_target_mem_nhdsWithin (I.prod I') x)
    mfld_set_tac
    -/
    filter_upwards [extChartAt_target_mem_nhdsWithin x] with y hy
    rw [extChartAt_prod] at hy
    exact (extChartAt I x.1).right_inv hy.1
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
    x : Prod M M'
    this : Filter.Eventually (fun y => Eq (Function.comp (↑(extChartAt I x.1)) (Fu …
    ⊢ HasFDerivWithinAt (writtenInExtChartAt (I.prod I') I x Prod.fst) (Continuous …
  -/
  apply HasFDerivWithinAt.congr_of_eventuallyEq hasFDerivWithinAt_fst this
  -- Porting note: next line was `simp only [mfld_simps]`
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
    x : Prod M M'
    this : Filter.Eventually (fun y => Eq (Function.comp (↑(extChartAt I x.1)) (Fu …
    ⊢ Eq (Function.comp (↑(extChartAt I x.1)) (Function.comp Prod.fst ↑(extChartAt …
  -/
  exact (extChartAt I x.1).right_inv <| (extChartAt I x.1).map_source (mem_extChartAt_source _)
  /-
    🎉 no goals
  -/


theorem hasMFDerivWithinAt_fst (s : Set (M × M')) (x : M × M') :
    HasMFDerivWithinAt (I.prod I') I Prod.fst s x
      (ContinuousLinearMap.fst 𝕜 (TangentSpace I x.1) (TangentSpace I' x.2)) :=
  (hasMFDerivAt_fst x).hasMFDerivWithinAt


theorem mdifferentiableAt_fst {x : M × M'} : MDifferentiableAt (I.prod I') I Prod.fst x :=
  (hasMFDerivAt_fst x).mdifferentiableAt


theorem mdifferentiableWithinAt_fst {s : Set (M × M')} {x : M × M'} :
    MDifferentiableWithinAt (I.prod I') I Prod.fst s x :=
  mdifferentiableAt_fst.mdifferentiableWithinAt


theorem mdifferentiable_fst : MDifferentiable (I.prod I') I (Prod.fst : M × M' → M) := fun _ =>
  mdifferentiableAt_fst


theorem mdifferentiableOn_fst {s : Set (M × M')} : MDifferentiableOn (I.prod I') I Prod.fst s :=
  mdifferentiable_fst.mdifferentiableOn


@[simp, mfld_simps]
theorem mfderiv_fst {x : M × M'} :
    mfderiv (I.prod I') I Prod.fst x =
      ContinuousLinearMap.fst 𝕜 (TangentSpace I x.1) (TangentSpace I' x.2) :=
  (hasMFDerivAt_fst x).mfderiv


theorem mfderivWithin_fst {s : Set (M × M')} {x : M × M'}
    (hxs : UniqueMDiffWithinAt (I.prod I') s x) :
    mfderivWithin (I.prod I') I Prod.fst s x =
      ContinuousLinearMap.fst 𝕜 (TangentSpace I x.1) (TangentSpace I' x.2) := by
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
    s : Set (Prod M M')
    x : Prod M M'
    hxs : UniqueMDiffWithinAt (I.prod I') s x
    ⊢ Eq (mfderivWithin (I.prod I') I Prod.fst s x) (ContinuousLinearMap.fst 𝕜 (Ta …
  -/
  rw [MDifferentiable.mfderivWithin mdifferentiableAt_fst hxs]; exact mfderiv_fst
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp, mfld_simps]
theorem tangentMap_prod_fst {p : TangentBundle (I.prod I') (M × M')} :
    tangentMap (I.prod I') I Prod.fst p = ⟨p.proj.1, p.2.1⟩ := by
  -- Porting note: `rfl` wasn't needed
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
    p : TangentBundle (I.prod I') (Prod M M')
    ⊢ Eq (tangentMap (I.prod I') I Prod.fst p) { proj := p.proj.1, snd := p.snd.1 }
  -/
  simp [tangentMap]; rfl
                     /-
                       🎉 no goals
                     -/


theorem tangentMapWithin_prod_fst {s : Set (M × M')} {p : TangentBundle (I.prod I') (M × M')}
    (hs : UniqueMDiffWithinAt (I.prod I') s p.proj) :
    tangentMapWithin (I.prod I') I Prod.fst s p = ⟨p.proj.1, p.2.1⟩ := by
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
    s : Set (Prod M M')
    p : TangentBundle (I.prod I') (Prod M M')
    hs : UniqueMDiffWithinAt (I.prod I') s p.proj
    ⊢ Eq (tangentMapWithin (I.prod I') I Prod.fst s p) { proj := p.proj.1, snd :=  …
  -/
  simp only [tangentMapWithin]
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
    s : Set (Prod M M')
    p : TangentBundle (I.prod I') (Prod M M')
    hs : UniqueMDiffWithinAt (I.prod I') s p.proj
    ⊢ Eq { proj := p.proj.1, snd := (mfderivWithin (I.prod I') I Prod.fst s p.proj …
  -/
  rw [mfderivWithin_fst]
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
      s : Set (Prod M M')
      p : TangentBundle (I.prod I') (Prod M M')
      hs : UniqueMDiffWithinAt (I.prod I') s p.proj
      ⊢ Eq { proj := p.proj.1, snd := (ContinuousLinearMap.fst 𝕜 (TangentSpace I p.p …
    -/
  · rcases p with ⟨⟩; rfl
                      /-
                        🎉 no goals
                      -/
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
      s : Set (Prod M M')
      p : TangentBundle (I.prod I') (Prod M M')
      hs : UniqueMDiffWithinAt (I.prod I') s p.proj
      ⊢ UniqueMDiffWithinAt (I.prod I') s p.proj
    -/
  · exact hs
    /-
      🎉 no goals
    -/


theorem hasMFDerivAt_snd (x : M × M') :
    HasMFDerivAt (I.prod I') I' Prod.snd x
      (ContinuousLinearMap.snd 𝕜 (TangentSpace I x.1) (TangentSpace I' x.2)) := by
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
    x : Prod M M'
    ⊢ HasMFDerivAt (I.prod I') I' Prod.snd x (ContinuousLinearMap.snd 𝕜 (TangentSp …
  -/
  refine ⟨continuous_snd.continuousAt, ?_⟩
  have :
    ∀ᶠ y in 𝓝[range (I.prod I')] extChartAt (I.prod I') x x,
      (extChartAt I' x.2 ∘ Prod.snd ∘ (extChartAt (I.prod I') x).symm) y = y.2 := by
    /- porting note: was
    apply Filter.mem_of_superset (extChartAt_target_mem_nhdsWithin (I.prod I') x)
    mfld_set_tac
    -/
    filter_upwards [extChartAt_target_mem_nhdsWithin x] with y hy
    rw [extChartAt_prod] at hy
    exact (extChartAt I' x.2).right_inv hy.2
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
    x : Prod M M'
    this : Filter.Eventually (fun y => Eq (Function.comp (↑(extChartAt I' x.2)) (F …
    ⊢ HasFDerivWithinAt (writtenInExtChartAt (I.prod I') I' x Prod.snd) (Continuou …
  -/
  apply HasFDerivWithinAt.congr_of_eventuallyEq hasFDerivWithinAt_snd this
  -- Porting note: the next line was `simp only [mfld_simps]`
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
    x : Prod M M'
    this : Filter.Eventually (fun y => Eq (Function.comp (↑(extChartAt I' x.2)) (F …
    ⊢ Eq (Function.comp (↑(extChartAt I' x.2)) (Function.comp Prod.snd ↑(extChartA …
  -/
  exact (extChartAt I' x.2).right_inv <| (extChartAt I' x.2).map_source (mem_extChartAt_source _)
  /-
    🎉 no goals
  -/


theorem hasMFDerivWithinAt_snd (s : Set (M × M')) (x : M × M') :
    HasMFDerivWithinAt (I.prod I') I' Prod.snd s x
      (ContinuousLinearMap.snd 𝕜 (TangentSpace I x.1) (TangentSpace I' x.2)) :=
  (hasMFDerivAt_snd x).hasMFDerivWithinAt


theorem mdifferentiableAt_snd {x : M × M'} : MDifferentiableAt (I.prod I') I' Prod.snd x :=
  (hasMFDerivAt_snd x).mdifferentiableAt


theorem mdifferentiableWithinAt_snd {s : Set (M × M')} {x : M × M'} :
    MDifferentiableWithinAt (I.prod I') I' Prod.snd s x :=
  mdifferentiableAt_snd.mdifferentiableWithinAt


theorem mdifferentiable_snd : MDifferentiable (I.prod I') I' (Prod.snd : M × M' → M') := fun _ =>
  mdifferentiableAt_snd


theorem mdifferentiableOn_snd {s : Set (M × M')} : MDifferentiableOn (I.prod I') I' Prod.snd s :=
  mdifferentiable_snd.mdifferentiableOn


@[simp, mfld_simps]
theorem mfderiv_snd {x : M × M'} :
    mfderiv (I.prod I') I' Prod.snd x =
      ContinuousLinearMap.snd 𝕜 (TangentSpace I x.1) (TangentSpace I' x.2) :=
  (hasMFDerivAt_snd x).mfderiv


theorem mfderivWithin_snd {s : Set (M × M')} {x : M × M'}
    (hxs : UniqueMDiffWithinAt (I.prod I') s x) :
    mfderivWithin (I.prod I') I' Prod.snd s x =
      ContinuousLinearMap.snd 𝕜 (TangentSpace I x.1) (TangentSpace I' x.2) := by
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
    s : Set (Prod M M')
    x : Prod M M'
    hxs : UniqueMDiffWithinAt (I.prod I') s x
    ⊢ Eq (mfderivWithin (I.prod I') I' Prod.snd s x) (ContinuousLinearMap.snd 𝕜 (T …
  -/
  rw [MDifferentiable.mfderivWithin mdifferentiableAt_snd hxs]; exact mfderiv_snd
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem MDifferentiableWithinAt.fst {f : N → M × M'} {s : Set N} {x : N}
    (hf : MDifferentiableWithinAt J (I.prod I') f s x) :
    MDifferentiableWithinAt J I (fun x => (f x).1) s x :=
  mdifferentiableAt_fst.comp_mdifferentiableWithinAt x hf


theorem MDifferentiableAt.fst {f : N → M × M'} {x : N} (hf : MDifferentiableAt J (I.prod I') f x) :
    MDifferentiableAt J I (fun x => (f x).1) x :=
  mdifferentiableAt_fst.comp x hf


theorem MDifferentiable.fst {f : N → M × M'} (hf : MDifferentiable J (I.prod I') f) :
    MDifferentiable J I fun x => (f x).1 :=
  mdifferentiable_fst.comp hf


theorem MDifferentiableWithinAt.snd {f : N → M × M'} {s : Set N} {x : N}
    (hf : MDifferentiableWithinAt J (I.prod I') f s x) :
    MDifferentiableWithinAt J I' (fun x => (f x).2) s x :=
  mdifferentiableAt_snd.comp_mdifferentiableWithinAt x hf


theorem MDifferentiableAt.snd {f : N → M × M'} {x : N} (hf : MDifferentiableAt J (I.prod I') f x) :
    MDifferentiableAt J I' (fun x => (f x).2) x :=
  mdifferentiableAt_snd.comp x hf


theorem MDifferentiable.snd {f : N → M × M'} (hf : MDifferentiable J (I.prod I') f) :
    MDifferentiable J I' fun x => (f x).2 :=
  mdifferentiable_snd.comp hf


theorem mdifferentiableWithinAt_prod_iff (f : M → M' × N') :
    MDifferentiableWithinAt I (I'.prod J') f s x ↔
      MDifferentiableWithinAt I I' (Prod.fst ∘ f) s x
      ∧ MDifferentiableWithinAt I J' (Prod.snd ∘ f) s x :=
  ⟨fun h => ⟨h.fst, h.snd⟩, fun h => h.1.prod_mk h.2⟩


theorem mdifferentiableWithinAt_prod_module_iff (f : M → F₁ × F₂) :
    MDifferentiableWithinAt I 𝓘(𝕜, F₁ × F₂) f s x ↔
      MDifferentiableWithinAt I 𝓘(𝕜, F₁) (Prod.fst ∘ f) s x ∧
      MDifferentiableWithinAt I 𝓘(𝕜, F₂) (Prod.snd ∘ f) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_17
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_18
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    s : Set M
    x : M
    f : M → Prod F₁ F₂
    ⊢ Iff (MDifferentiableWithinAt I (modelWithCornersSelf 𝕜 (Prod F₁ F₂)) f s x)  …
  -/
  rw [modelWithCornersSelf_prod, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_17
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_18
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    s : Set M
    x : M
    f : M → Prod F₁ F₂
    ⊢ Iff (MDifferentiableWithinAt I ((modelWithCornersSelf 𝕜 F₁).prod (modelWithC …
  -/
  exact mdifferentiableWithinAt_prod_iff f
  /-
    🎉 no goals
  -/


theorem mdifferentiableAt_prod_iff (f : M → M' × N') :
    MDifferentiableAt I (I'.prod J') f x ↔
      MDifferentiableAt I I' (Prod.fst ∘ f) x ∧ MDifferentiableAt I J' (Prod.snd ∘ f) x := by
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
    F' : Type u_14
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_15
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_16
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    x : M
    f : M → Prod M' N'
    ⊢ Iff (MDifferentiableAt I (I'.prod J') f x) (And (MDifferentiableAt I I' (Fun …
  -/
  simp_rw [← mdifferentiableWithinAt_univ]; exact mdifferentiableWithinAt_prod_iff f
                                            /-
                                              🎉 no goals
                                            -/


theorem mdifferentiableAt_prod_module_iff (f : M → F₁ × F₂) :
    MDifferentiableAt I 𝓘(𝕜, F₁ × F₂) f x ↔
      MDifferentiableAt I 𝓘(𝕜, F₁) (Prod.fst ∘ f) x
      ∧ MDifferentiableAt I 𝓘(𝕜, F₂) (Prod.snd ∘ f) x := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_17
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_18
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    x : M
    f : M → Prod F₁ F₂
    ⊢ Iff (MDifferentiableAt I (modelWithCornersSelf 𝕜 (Prod F₁ F₂)) f x) (And (MD …
  -/
  rw [modelWithCornersSelf_prod, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_17
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_18
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    x : M
    f : M → Prod F₁ F₂
    ⊢ Iff (MDifferentiableAt I ((modelWithCornersSelf 𝕜 F₁).prod (modelWithCorners …
  -/
  exact mdifferentiableAt_prod_iff f
  /-
    🎉 no goals
  -/


theorem mdifferentiableOn_prod_iff (f : M → M' × N') :
    MDifferentiableOn I (I'.prod J') f s ↔
      MDifferentiableOn I I' (Prod.fst ∘ f) s ∧ MDifferentiableOn I J' (Prod.snd ∘ f) s :=
  ⟨fun h ↦ ⟨fun x hx ↦ ((mdifferentiableWithinAt_prod_iff f).1 (h x hx)).1,
      fun x hx ↦ ((mdifferentiableWithinAt_prod_iff f).1 (h x hx)).2⟩ ,
    fun h x hx ↦ (mdifferentiableWithinAt_prod_iff f).2 ⟨h.1 x hx, h.2 x hx⟩⟩


theorem mdifferentiableOn_prod_module_iff (f : M → F₁ × F₂) :
    MDifferentiableOn I 𝓘(𝕜, F₁ × F₂) f s ↔
      MDifferentiableOn I 𝓘(𝕜, F₁) (Prod.fst ∘ f) s
      ∧ MDifferentiableOn I 𝓘(𝕜, F₂) (Prod.snd ∘ f) s := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_17
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_18
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    s : Set M
    f : M → Prod F₁ F₂
    ⊢ Iff (MDifferentiableOn I (modelWithCornersSelf 𝕜 (Prod F₁ F₂)) f s) (And (MD …
  -/
  rw [modelWithCornersSelf_prod, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_17
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_18
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    s : Set M
    f : M → Prod F₁ F₂
    ⊢ Iff (MDifferentiableOn I ((modelWithCornersSelf 𝕜 F₁).prod (modelWithCorners …
  -/
  exact mdifferentiableOn_prod_iff f
  /-
    🎉 no goals
  -/


theorem mdifferentiable_prod_iff (f : M → M' × N') :
    MDifferentiable I (I'.prod J') f ↔
      MDifferentiable I I' (Prod.fst ∘ f) ∧ MDifferentiable I J' (Prod.snd ∘ f) :=
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
                                          F' : Type u_14
                                          inst✝⁴ : NormedAddCommGroup F'
                                          inst✝³ : NormedSpace 𝕜 F'
                                          G' : Type u_15
                                          inst✝² : TopologicalSpace G'
                                          J' : ModelWithCorners 𝕜 F' G'
                                          N' : Type u_16
                                          inst✝¹ : TopologicalSpace N'
                                          inst✝ : ChartedSpace G' N'
                                          f : M → Prod M' N'
                                          h : And (MDifferentiable I I' (Function.comp Prod.fst f)) (MDifferentiable I J …
                                          ⊢ MDifferentiable I (I'.prod J') f
                                        -/
  ⟨fun h => ⟨h.fst, h.snd⟩, fun h => by convert h.1.prod_mk h.2⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem mdifferentiable_prod_module_iff (f : M → F₁ × F₂) :
    MDifferentiable I 𝓘(𝕜, F₁ × F₂) f ↔
      MDifferentiable I 𝓘(𝕜, F₁) (Prod.fst ∘ f) ∧ MDifferentiable I 𝓘(𝕜, F₂) (Prod.snd ∘ f) := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_17
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_18
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    f : M → Prod F₁ F₂
    ⊢ Iff (MDifferentiable I (modelWithCornersSelf 𝕜 (Prod F₁ F₂)) f) (And (MDiffe …
  -/
  rw [modelWithCornersSelf_prod, ← chartedSpaceSelf_prod]
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    F₁ : Type u_17
    inst✝³ : NormedAddCommGroup F₁
    inst✝² : NormedSpace 𝕜 F₁
    F₂ : Type u_18
    inst✝¹ : NormedAddCommGroup F₂
    inst✝ : NormedSpace 𝕜 F₂
    f : M → Prod F₁ F₂
    ⊢ Iff (MDifferentiable I ((modelWithCornersSelf 𝕜 F₁).prod (modelWithCornersSe …
  -/
  exact mdifferentiable_prod_iff f
  /-
    🎉 no goals
  -/



/-- The product map of two `C^n` functions within a set at a point is `C^n`
within the product set at the product point. -/
theorem MDifferentiableWithinAt.prod_map' {p : M × N} (hf : MDifferentiableWithinAt I I' f s p.1)
    (hg : MDifferentiableWithinAt J J' g r p.2) :
    MDifferentiableWithinAt (I.prod J) (I'.prod J') (Prod.map f g) (s ×ˢ r) p :=
  (hf.comp p mdifferentiableWithinAt_fst (prod_subset_preimage_fst _ _)).prod_mk <|
    hg.comp p mdifferentiableWithinAt_snd (prod_subset_preimage_snd _ _)


theorem MDifferentiableWithinAt.prod_map (hf : MDifferentiableWithinAt I I' f s x)
    (hg : MDifferentiableWithinAt J J' g r y) :
    MDifferentiableWithinAt (I.prod J) (I'.prod J') (Prod.map f g) (s ×ˢ r) (x, y) :=
  MDifferentiableWithinAt.prod_map' hf hg


theorem MDifferentiableAt.prod_map
    (hf : MDifferentiableAt I I' f x) (hg : MDifferentiableAt J J' g y) :
    MDifferentiableAt (I.prod J) (I'.prod J') (Prod.map f g) (x, y) := by
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_11
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_12
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_13
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_14
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_15
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_16
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    x : M
    f : M → M'
    g : N → N'
    y : N
    hf : MDifferentiableAt I I' f x
    hg : MDifferentiableAt J J' g y
    ⊢ MDifferentiableAt (I.prod J) (I'.prod J') (Prod.map f g) { fst := x, snd :=  …
  -/
  rw [← mdifferentiableWithinAt_univ] at *
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_11
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_12
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_13
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_14
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_15
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_16
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    x : M
    f : M → M'
    g : N → N'
    y : N
    hf : MDifferentiableWithinAt I I' f Set.univ x
    hg : MDifferentiableWithinAt J J' g Set.univ y
    ⊢ MDifferentiableWithinAt (I.prod J) (I'.prod J') (Prod.map f g) Set.univ { fs …
  -/
  convert hf.prod_map hg
  /-
    case h.e'_22
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_11
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_12
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_13
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_14
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_15
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_16
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    x : M
    f : M → M'
    g : N → N'
    y : N
    hf : MDifferentiableWithinAt I I' f Set.univ x
    hg : MDifferentiableWithinAt J J' g Set.univ y
    ⊢ Eq Set.univ (SProd.sprod Set.univ Set.univ)
  -/
  exact univ_prod_univ.symm
  /-
    🎉 no goals
  -/


/-- Variant of `MDifferentiableAt.prod_map` in which the point in the product is given as `p`
instead of a pair `(x, y)`. -/
theorem MDifferentiableAt.prod_map' {p : M × N}
    (hf : MDifferentiableAt I I' f p.1) (hg : MDifferentiableAt J J' g p.2) :
    MDifferentiableAt (I.prod J) (I'.prod J') (Prod.map f g) p := by
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_11
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_12
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_13
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_14
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_15
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_16
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    g : N → N'
    p : Prod M N
    hf : MDifferentiableAt I I' f p.1
    hg : MDifferentiableAt J J' g p.2
    ⊢ MDifferentiableAt (I.prod J) (I'.prod J') (Prod.map f g) p
  -/
  rcases p with ⟨⟩
  /-
    case mk
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_11
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_12
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_13
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_14
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_15
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_16
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    g : N → N'
    fst✝ : M
    snd✝ : N
    hf : MDifferentiableAt I I' f { fst := fst✝, snd := snd✝ }.1
    hg : MDifferentiableAt J J' g { fst := fst✝, snd := snd✝ }.2
    ⊢ MDifferentiableAt (I.prod J) (I'.prod J') (Prod.map f g) { fst := fst✝, snd  …
  -/
  exact hf.prod_map hg
  /-
    🎉 no goals
  -/


theorem MDifferentiableOn.prod_map
    (hf : MDifferentiableOn I I' f s) (hg : MDifferentiableOn J J' g r) :
    MDifferentiableOn (I.prod J) (I'.prod J') (Prod.map f g) (s ×ˢ r) :=
  (hf.comp mdifferentiableOn_fst (prod_subset_preimage_fst _ _)).prod_mk <|
    hg.comp mdifferentiableOn_snd (prod_subset_preimage_snd _ _)


theorem MDifferentiable.prod_map (hf : MDifferentiable I I' f) (hg : MDifferentiable J J' g) :
    MDifferentiable (I.prod J) (I'.prod J') (Prod.map f g) := by
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_11
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_12
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_13
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_14
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_15
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_16
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    g : N → N'
    hf : MDifferentiable I I' f
    hg : MDifferentiable J J' g
    ⊢ MDifferentiable (I.prod J) (I'.prod J') (Prod.map f g)
  -/
  intro p
  /-
    𝕜 : Type u_1
    inst✝²⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁹ : NormedAddCommGroup E
    inst✝¹⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹⁶ : TopologicalSpace M
    inst✝¹⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝¹⁴ : NormedAddCommGroup E'
    inst✝¹³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝¹² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹¹ : TopologicalSpace M'
    inst✝¹⁰ : ChartedSpace H' M'
    F : Type u_11
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace 𝕜 F
    G : Type u_12
    inst✝⁷ : TopologicalSpace G
    J : ModelWithCorners 𝕜 F G
    N : Type u_13
    inst✝⁶ : TopologicalSpace N
    inst✝⁵ : ChartedSpace G N
    F' : Type u_14
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    G' : Type u_15
    inst✝² : TopologicalSpace G'
    J' : ModelWithCorners 𝕜 F' G'
    N' : Type u_16
    inst✝¹ : TopologicalSpace N'
    inst✝ : ChartedSpace G' N'
    f : M → M'
    g : N → N'
    hf : MDifferentiable I I' f
    hg : MDifferentiable J J' g
    p : Prod M N
    ⊢ MDifferentiableAt (I.prod J) (I'.prod J') (Prod.map f g) p
  -/
  exact (hf p.1).prod_map' (hg p.2)
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem tangentMap_prod_snd {p : TangentBundle (I.prod I') (M × M')} :
    tangentMap (I.prod I') I' Prod.snd p = ⟨p.proj.2, p.2.2⟩ := by
  -- Porting note: `rfl` wasn't needed
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
    p : TangentBundle (I.prod I') (Prod M M')
    ⊢ Eq (tangentMap (I.prod I') I' Prod.snd p) { proj := p.proj.2, snd := p.snd.2 }
  -/
  simp [tangentMap]; rfl
                     /-
                       🎉 no goals
                     -/


theorem tangentMapWithin_prod_snd {s : Set (M × M')} {p : TangentBundle (I.prod I') (M × M')}
    (hs : UniqueMDiffWithinAt (I.prod I') s p.proj) :
    tangentMapWithin (I.prod I') I' Prod.snd s p = ⟨p.proj.2, p.2.2⟩ := by
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
    s : Set (Prod M M')
    p : TangentBundle (I.prod I') (Prod M M')
    hs : UniqueMDiffWithinAt (I.prod I') s p.proj
    ⊢ Eq (tangentMapWithin (I.prod I') I' Prod.snd s p) { proj := p.proj.2, snd := …
  -/
  simp only [tangentMapWithin]
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
    s : Set (Prod M M')
    p : TangentBundle (I.prod I') (Prod M M')
    hs : UniqueMDiffWithinAt (I.prod I') s p.proj
    ⊢ Eq { proj := p.proj.2, snd := (mfderivWithin (I.prod I') I' Prod.snd s p.pro …
  -/
  rw [mfderivWithin_snd]
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
      s : Set (Prod M M')
      p : TangentBundle (I.prod I') (Prod M M')
      hs : UniqueMDiffWithinAt (I.prod I') s p.proj
      ⊢ Eq { proj := p.proj.2, snd := (ContinuousLinearMap.snd 𝕜 (TangentSpace I p.p …
    -/
  · rcases p with ⟨⟩; rfl
                      /-
                        🎉 no goals
                      -/
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
      s : Set (Prod M M')
      p : TangentBundle (I.prod I') (Prod M M')
      hs : UniqueMDiffWithinAt (I.prod I') s p.proj
      ⊢ UniqueMDiffWithinAt (I.prod I') s p.proj
    -/
  · exact hs
    /-
      🎉 no goals
    -/


theorem MDifferentiableAt.mfderiv_prod {f : M → M'} {g : M → M''} {x : M}
    (hf : MDifferentiableAt I I' f x) (hg : MDifferentiableAt I I'' g x) :
    mfderiv I (I'.prod I'') (fun x => (f x, g x)) x =
      (mfderiv I I' f x).prod (mfderiv I I'' g x) := by
  classical
  simp_rw [mfderiv, if_pos (hf.prod_mk hg), if_pos hf, if_pos hg]
  exact hf.differentiableWithinAt_writtenInExtChartAt.fderivWithin_prod
    hg.differentiableWithinAt_writtenInExtChartAt (I.uniqueDiffOn _ (mem_range_self _))


theorem mfderiv_prod_left {x₀ : M} {y₀ : M'} :
    mfderiv I (I.prod I') (fun x => (x, y₀)) x₀ =
      ContinuousLinearMap.inl 𝕜 (TangentSpace I x₀) (TangentSpace I' y₀) := by
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
    x₀ : M
    y₀ : M'
    ⊢ Eq (mfderiv I (I.prod I') (fun x => { fst := x, snd := y₀ }) x₀) (Continuous …
  -/
  refine (mdifferentiableAt_id.mfderiv_prod mdifferentiableAt_const).trans ?_
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
    x₀ : M
    y₀ : M'
    ⊢ Eq ((mfderiv I I id x₀).prod (mfderiv I I' (fun x => y₀) x₀)) (ContinuousLin …
  -/
  rw [mfderiv_id, mfderiv_const, ContinuousLinearMap.inl]
  /-
    🎉 no goals
  -/


theorem tangentMap_prod_left {p : TangentBundle I M} {y₀ : M'} :
    tangentMap I (I.prod I') (fun x => (x, y₀)) p = ⟨(p.1, y₀), (p.2, 0)⟩ := by
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
    p : TangentBundle I M
    y₀ : M'
    ⊢ Eq (tangentMap I (I.prod I') (fun x => { fst := x, snd := y₀ }) p) { proj := …
  -/
  simp only [tangentMap, mfderiv_prod_left, TotalSpace.mk_inj]
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
    p : TangentBundle I M
    y₀ : M'
    ⊢ Eq ((ContinuousLinearMap.inl 𝕜 (TangentSpace I p.proj) (TangentSpace I' y₀)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mfderiv_prod_right {x₀ : M} {y₀ : M'} :
    mfderiv I' (I.prod I') (fun y => (x₀, y)) y₀ =
      ContinuousLinearMap.inr 𝕜 (TangentSpace I x₀) (TangentSpace I' y₀) := by
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
    x₀ : M
    y₀ : M'
    ⊢ Eq (mfderiv I' (I.prod I') (fun y => { fst := x₀, snd := y }) y₀) (Continuou …
  -/
  refine (mdifferentiableAt_const.mfderiv_prod mdifferentiableAt_id).trans ?_
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
    x₀ : M
    y₀ : M'
    ⊢ Eq ((mfderiv I' I (fun x => x₀) y₀).prod (mfderiv I' I' id y₀)) (ContinuousL …
  -/
  rw [mfderiv_id, mfderiv_const, ContinuousLinearMap.inr]
  /-
    🎉 no goals
  -/


theorem tangentMap_prod_right {p : TangentBundle I' M'} {x₀ : M} :
    tangentMap I' (I.prod I') (fun y => (x₀, y)) p = ⟨(x₀, p.1), (0, p.2)⟩ := by
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
    p : TangentBundle I' M'
    x₀ : M
    ⊢ Eq (tangentMap I' (I.prod I') (fun y => { fst := x₀, snd := y }) p) { proj : …
  -/
  simp only [tangentMap, mfderiv_prod_right, TotalSpace.mk_inj]
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
    p : TangentBundle I' M'
    x₀ : M
    ⊢ Eq ((ContinuousLinearMap.inr 𝕜 (TangentSpace I x₀) (TangentSpace I' p.proj)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The total derivative of a function in two variables is the sum of the partial derivatives.
  Note that to state this (without casts) we need to be able to see through the definition of
  `TangentSpace`. -/
theorem mfderiv_prod_eq_add {f : M × M' → M''} {p : M × M'}
    (hf : MDifferentiableAt (I.prod I') I'' f p) :
    mfderiv (I.prod I') I'' f p =
        mfderiv (I.prod I') I'' (fun z : M × M' => f (z.1, p.2)) p +
          mfderiv (I.prod I') I'' (fun z : M × M' => f (p.1, z.2)) p := by
  erw [mfderiv_comp_of_eq hf (mdifferentiableAt_fst.prod_mk mdifferentiableAt_const) rfl,
    mfderiv_comp_of_eq hf (mdifferentiableAt_const.prod_mk mdifferentiableAt_snd) rfl,
    ← ContinuousLinearMap.comp_add,
    mdifferentiableAt_fst.mfderiv_prod mdifferentiableAt_const,
    mdifferentiableAt_const.mfderiv_prod mdifferentiableAt_snd, mfderiv_fst,
    mfderiv_snd, mfderiv_const, mfderiv_const]
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
    f : Prod M M' → M''
    p : Prod M M'
    hf : MDifferentiableAt (I.prod I') I'' f p
    ⊢ Eq (mfderiv (I.prod I') I'' f p) ((mfderiv (I.prod I') I'' f { fst := p.1, s …
  -/
  symm
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
    f : Prod M M' → M''
    p : Prod M M'
    hf : MDifferentiableAt (I.prod I') I'' f p
    ⊢ Eq ((mfderiv (I.prod I') I'' f { fst := p.1, snd := p.2 }).comp (HAdd.hAdd ( …
  -/
  convert ContinuousLinearMap.comp_id <| mfderiv (.prod I I') I'' f (p.1, p.2)
  /-
    case h.e'_2.h.e'_24
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
    f : Prod M M' → M''
    p : Prod M M'
    hf : MDifferentiableAt (I.prod I') I'' f p
    ⊢ Eq (HAdd.hAdd ((ContinuousLinearMap.fst 𝕜 (TangentSpace I p.1) (TangentSpace …
  -/
  exact ContinuousLinearMap.coprod_inl_inr
  /-
    🎉 no goals
  -/


/-- The total derivative of a function in two variables is the sum of the partial derivatives.
  Note that to state this (without casts) we need to be able to see through the definition of
  `TangentSpace`. Version in terms of the one-variable derivatives. -/
theorem mfderiv_prod_eq_add_comp {f : M × M' → M''} {p : M × M'}
    (hf : MDifferentiableAt (I.prod I') I'' f p) :
    mfderiv (I.prod I') I'' f p =
        (mfderiv I I'' (fun z : M => f (z, p.2)) p.1) ∘L (id (ContinuousLinearMap.fst 𝕜 E E') :
          (TangentSpace (I.prod I') p) →L[𝕜] (TangentSpace I p.1)) +
        (mfderiv I' I'' (fun z : M' => f (p.1, z)) p.2) ∘L (id (ContinuousLinearMap.snd 𝕜 E E') :
          (TangentSpace (I.prod I') p) →L[𝕜] (TangentSpace I' p.2)) := by
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
    f : Prod M M' → M''
    p : Prod M M'
    hf : MDifferentiableAt (I.prod I') I'' f p
    ⊢ Eq (mfderiv (I.prod I') I'' f p) (HAdd.hAdd ((mfderiv I I'' (fun z => f { fs …
  -/
  rw [mfderiv_prod_eq_add hf]
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
    f : Prod M M' → M''
    p : Prod M M'
    hf : MDifferentiableAt (I.prod I') I'' f p
    ⊢ Eq (HAdd.hAdd (mfderiv (I.prod I') I'' (fun z => f { fst := z.1, snd := p.2  …
  -/
  congr
    /-
      case e_a
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
      f : Prod M M' → M''
      p : Prod M M'
      hf : MDifferentiableAt (I.prod I') I'' f p
      ⊢ Eq (mfderiv (I.prod I') I'' (fun z => f { fst := z.1, snd := p.2 }) p) ((mfd …
    -/
  · have : (fun z : M × M' => f (z.1, p.2)) = (fun z : M => f (z, p.2)) ∘ Prod.fst := rfl
    /-
      case e_a
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
      f : Prod M M' → M''
      p : Prod M M'
      hf : MDifferentiableAt (I.prod I') I'' f p
      this : Eq (fun z => f { fst := z.1, snd := p.2 }) (Function.comp (fun z => f { …
      ⊢ Eq (mfderiv (I.prod I') I'' (fun z => f { fst := z.1, snd := p.2 }) p) ((mfd …
    -/
    rw [this, mfderiv_comp (I' := I)]
      /-
        case e_a
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
        f : Prod M M' → M''
        p : Prod M M'
        hf : MDifferentiableAt (I.prod I') I'' f p
        this : Eq (fun z => f { fst := z.1, snd := p.2 }) (Function.comp (fun z => f { …
        ⊢ Eq ((mfderiv I I'' (fun z => f { fst := z, snd := p.2 }) p.1).comp (mfderiv  …
      -/
    · simp only [mfderiv_fst, id_eq]
      /-
        case e_a
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
        f : Prod M M' → M''
        p : Prod M M'
        hf : MDifferentiableAt (I.prod I') I'' f p
        this : Eq (fun z => f { fst := z.1, snd := p.2 }) (Function.comp (fun z => f { …
        ⊢ Eq ((mfderiv I I'' (fun z => f { fst := z, snd := p.2 }) p.1).comp (Continuo …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case e_a.hg
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
        f : Prod M M' → M''
        p : Prod M M'
        hf : MDifferentiableAt (I.prod I') I'' f p
        this : Eq (fun z => f { fst := z.1, snd := p.2 }) (Function.comp (fun z => f { …
        ⊢ MDifferentiableAt I I'' (fun z => f { fst := z, snd := p.2 }) p.1
      -/
    · exact hf.comp _  (mdifferentiableAt_id.prod_mk mdifferentiableAt_const)
      /-
        🎉 no goals
      -/
      /-
        case e_a.hf
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
        f : Prod M M' → M''
        p : Prod M M'
        hf : MDifferentiableAt (I.prod I') I'' f p
        this : Eq (fun z => f { fst := z.1, snd := p.2 }) (Function.comp (fun z => f { …
        ⊢ MDifferentiableAt (I.prod I') I Prod.fst p
      -/
    · exact mdifferentiableAt_fst
      /-
        🎉 no goals
      -/
    /-
      case e_a
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
      f : Prod M M' → M''
      p : Prod M M'
      hf : MDifferentiableAt (I.prod I') I'' f p
      ⊢ Eq (mfderiv (I.prod I') I'' (fun z => f { fst := p.1, snd := z.2 }) p) ((mfd …
    -/
  · have : (fun z : M × M' => f (p.1, z.2)) = (fun z : M' => f (p.1, z)) ∘ Prod.snd := rfl
    /-
      case e_a
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
      f : Prod M M' → M''
      p : Prod M M'
      hf : MDifferentiableAt (I.prod I') I'' f p
      this : Eq (fun z => f { fst := p.1, snd := z.2 }) (Function.comp (fun z => f { …
      ⊢ Eq (mfderiv (I.prod I') I'' (fun z => f { fst := p.1, snd := z.2 }) p) ((mfd …
    -/
    rw [this, mfderiv_comp (I' := I')]
      /-
        case e_a
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
        f : Prod M M' → M''
        p : Prod M M'
        hf : MDifferentiableAt (I.prod I') I'' f p
        this : Eq (fun z => f { fst := p.1, snd := z.2 }) (Function.comp (fun z => f { …
        ⊢ Eq ((mfderiv I' I'' (fun z => f { fst := p.1, snd := z }) p.2).comp (mfderiv …
      -/
    · simp only [mfderiv_snd, id_eq]
      /-
        case e_a
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
        f : Prod M M' → M''
        p : Prod M M'
        hf : MDifferentiableAt (I.prod I') I'' f p
        this : Eq (fun z => f { fst := p.1, snd := z.2 }) (Function.comp (fun z => f { …
        ⊢ Eq ((mfderiv I' I'' (fun z => f { fst := p.1, snd := z }) p.2).comp (Continu …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case e_a.hg
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
        f : Prod M M' → M''
        p : Prod M M'
        hf : MDifferentiableAt (I.prod I') I'' f p
        this : Eq (fun z => f { fst := p.1, snd := z.2 }) (Function.comp (fun z => f { …
        ⊢ MDifferentiableAt I' I'' (fun z => f { fst := p.1, snd := z }) p.2
      -/
    · exact hf.comp _ (mdifferentiableAt_const.prod_mk mdifferentiableAt_id)
      /-
        🎉 no goals
      -/
      /-
        case e_a.hf
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
        f : Prod M M' → M''
        p : Prod M M'
        hf : MDifferentiableAt (I.prod I') I'' f p
        this : Eq (fun z => f { fst := p.1, snd := z.2 }) (Function.comp (fun z => f { …
        ⊢ MDifferentiableAt (I.prod I') I' Prod.snd p
      -/
    · exact mdifferentiableAt_snd
      /-
        🎉 no goals
      -/


/-- The total derivative of a function in two variables is the sum of the partial derivatives.
  Note that to state this (without casts) we need to be able to see through the definition of
  `TangentSpace`. Version in terms of the one-variable derivatives. -/
theorem mfderiv_prod_eq_add_apply {f : M × M' → M''} {p : M × M'} {v : TangentSpace (I.prod I') p}
    (hf : MDifferentiableAt (I.prod I') I'' f p) :
    mfderiv (I.prod I') I'' f p v =
      mfderiv I I'' (fun z : M => f (z, p.2)) p.1 v.1 +
      mfderiv I' I'' (fun z : M' => f (p.1, z)) p.2 v.2 := by
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
    f : Prod M M' → M''
    p : Prod M M'
    v : TangentSpace (I.prod I') p
    hf : MDifferentiableAt (I.prod I') I'' f p
    ⊢ Eq ((mfderiv (I.prod I') I'' f p) v) (HAdd.hAdd ((mfderiv I I'' (fun z => f  …
  -/
  rw [mfderiv_prod_eq_add_comp hf]
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
    f : Prod M M' → M''
    p : Prod M M'
    v : TangentSpace (I.prod I') p
    hf : MDifferentiableAt (I.prod I') I'' f p
    ⊢ Eq ((HAdd.hAdd ((mfderiv I I'' (fun z => f { fst := z, snd := p.2 }) p.1).co …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem HasMFDerivAt.add (hf : HasMFDerivAt I 𝓘(𝕜, E') f z f')
    (hg : HasMFDerivAt I 𝓘(𝕜, E') g z g') : HasMFDerivAt I 𝓘(𝕜, E') (f + g) z (f' + g') :=
  ⟨hf.1.add hg.1, hf.2.add hg.2⟩


theorem MDifferentiableAt.add (hf : MDifferentiableAt I 𝓘(𝕜, E') f z)
    (hg : MDifferentiableAt I 𝓘(𝕜, E') g z) : MDifferentiableAt I 𝓘(𝕜, E') (f + g) z :=
  (hf.hasMFDerivAt.add hg.hasMFDerivAt).mdifferentiableAt


theorem MDifferentiable.add (hf : MDifferentiable I 𝓘(𝕜, E') f)
    (hg : MDifferentiable I 𝓘(𝕜, E') g) : MDifferentiable I 𝓘(𝕜, E') (f + g) := fun x =>
  (hf x).add (hg x)

-- Porting note: forcing types using `by exact`

theorem mfderiv_add (hf : MDifferentiableAt I 𝓘(𝕜, E') f z)
    (hg : MDifferentiableAt I 𝓘(𝕜, E') g z) :
        /-
          𝕜 : Type u_1
          inst✝³³ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝³² : NormedAddCommGroup E
          inst✝³¹ : NormedSpace 𝕜 E
          H : Type u_3
          inst✝³⁰ : TopologicalSpace H
          I : ModelWithCorners 𝕜 E H
          M : Type u_4
          inst✝²⁹ : TopologicalSpace M
          inst✝²⁸ : ChartedSpace H M
          E' : Type u_5
          inst✝²⁷ : NormedAddCommGroup E'
          inst✝²⁶ : NormedSpace 𝕜 E'
          H' : Type u_6
          inst✝²⁵ : TopologicalSpace H'
          I' : ModelWithCorners 𝕜 E' H'
          M' : Type u_7
          inst✝²⁴ : TopologicalSpace M'
          inst✝²³ : ChartedSpace H' M'
          E'' : Type u_8
          inst✝²² : NormedAddCommGroup E''
          inst✝²¹ : NormedSpace 𝕜 E''
          H'' : Type u_9
          inst✝²⁰ : TopologicalSpace H''
          I'' : ModelWithCorners 𝕜 E'' H''
          M'' : Type u_10
          inst✝¹⁹ : TopologicalSpace M''
          inst✝¹⁸ : ChartedSpace H'' M''
          F : Type u_11
          inst✝¹⁷ : NormedAddCommGroup F
          inst✝¹⁶ : NormedSpace 𝕜 F
          G : Type u_12
          inst✝¹⁵ : TopologicalSpace G
          J : ModelWithCorners 𝕜 F G
          N : Type u_13
          inst✝¹⁴ : TopologicalSpace N
          inst✝¹³ : ChartedSpace G N
          F' : Type u_14
          inst✝¹² : NormedAddCommGroup F'
          inst✝¹¹ : NormedSpace 𝕜 F'
          G' : Type u_15
          inst✝¹⁰ : TopologicalSpace G'
          J' : ModelWithCorners 𝕜 F' G'
          N' : Type u_16
          inst✝⁹ : TopologicalSpace N'
          inst✝⁸ : ChartedSpace G' N'
          F₁ : Type u_17
          inst✝⁷ : NormedAddCommGroup F₁
          inst✝⁶ : NormedSpace 𝕜 F₁
          F₂ : Type u_18
          inst✝⁵ : NormedAddCommGroup F₂
          inst✝⁴ : NormedSpace 𝕜 F₂
          F₃ : Type u_19
          inst✝³ : NormedAddCommGroup F₃
          inst✝² : NormedSpace 𝕜 F₃
          F₄ : Type u_20
          inst✝¹ : NormedAddCommGroup F₄
          inst✝ : NormedSpace 𝕜 F₄
          s : Set M
          x z : M
          f g : M → E'
          f' g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
          hf : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') f z
          hg : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') g z
          ⊢ ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
        -/
    (by exact mfderiv I 𝓘(𝕜, E') (f + g) z : TangentSpace I z →L[𝕜] E') =
        /-
          🎉 no goals
        -/
          /-
            𝕜 : Type u_1
            inst✝³³ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝³² : NormedAddCommGroup E
            inst✝³¹ : NormedSpace 𝕜 E
            H : Type u_3
            inst✝³⁰ : TopologicalSpace H
            I : ModelWithCorners 𝕜 E H
            M : Type u_4
            inst✝²⁹ : TopologicalSpace M
            inst✝²⁸ : ChartedSpace H M
            E' : Type u_5
            inst✝²⁷ : NormedAddCommGroup E'
            inst✝²⁶ : NormedSpace 𝕜 E'
            H' : Type u_6
            inst✝²⁵ : TopologicalSpace H'
            I' : ModelWithCorners 𝕜 E' H'
            M' : Type u_7
            inst✝²⁴ : TopologicalSpace M'
            inst✝²³ : ChartedSpace H' M'
            E'' : Type u_8
            inst✝²² : NormedAddCommGroup E''
            inst✝²¹ : NormedSpace 𝕜 E''
            H'' : Type u_9
            inst✝²⁰ : TopologicalSpace H''
            I'' : ModelWithCorners 𝕜 E'' H''
            M'' : Type u_10
            inst✝¹⁹ : TopologicalSpace M''
            inst✝¹⁸ : ChartedSpace H'' M''
            F : Type u_11
            inst✝¹⁷ : NormedAddCommGroup F
            inst✝¹⁶ : NormedSpace 𝕜 F
            G : Type u_12
            inst✝¹⁵ : TopologicalSpace G
            J : ModelWithCorners 𝕜 F G
            N : Type u_13
            inst✝¹⁴ : TopologicalSpace N
            inst✝¹³ : ChartedSpace G N
            F' : Type u_14
            inst✝¹² : NormedAddCommGroup F'
            inst✝¹¹ : NormedSpace 𝕜 F'
            G' : Type u_15
            inst✝¹⁰ : TopologicalSpace G'
            J' : ModelWithCorners 𝕜 F' G'
            N' : Type u_16
            inst✝⁹ : TopologicalSpace N'
            inst✝⁸ : ChartedSpace G' N'
            F₁ : Type u_17
            inst✝⁷ : NormedAddCommGroup F₁
            inst✝⁶ : NormedSpace 𝕜 F₁
            F₂ : Type u_18
            inst✝⁵ : NormedAddCommGroup F₂
            inst✝⁴ : NormedSpace 𝕜 F₂
            F₃ : Type u_19
            inst✝³ : NormedAddCommGroup F₃
            inst✝² : NormedSpace 𝕜 F₃
            F₄ : Type u_20
            inst✝¹ : NormedAddCommGroup F₄
            inst✝ : NormedSpace 𝕜 F₄
            s : Set M
            x z : M
            f g : M → E'
            f' g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
            hf : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') f z
            hg : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') g z
            ⊢ ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
          -/
          /-
            🎉 no goals
          -/
      (by exact mfderiv I 𝓘(𝕜, E') f z) + (by exact mfderiv I 𝓘(𝕜, E') g z) :=
                                              /-
                                                🎉 no goals
                                              -/
  (hf.hasMFDerivAt.add hg.hasMFDerivAt).mfderiv


theorem HasMFDerivAt.const_smul (hf : HasMFDerivAt I 𝓘(𝕜, E') f z f') (s : 𝕜) :
    HasMFDerivAt I 𝓘(𝕜, E') (s • f) z (s • f') :=
  ⟨hf.1.const_smul s, hf.2.const_smul s⟩


theorem MDifferentiableAt.const_smul (hf : MDifferentiableAt I 𝓘(𝕜, E') f z) (s : 𝕜) :
    MDifferentiableAt I 𝓘(𝕜, E') (s • f) z :=
  (hf.hasMFDerivAt.const_smul s).mdifferentiableAt


theorem MDifferentiable.const_smul (s : 𝕜) (hf : MDifferentiable I 𝓘(𝕜, E') f) :
    MDifferentiable I 𝓘(𝕜, E') (s • f) := fun x => (hf x).const_smul s


theorem const_smul_mfderiv (hf : MDifferentiableAt I 𝓘(𝕜, E') f z) (s : 𝕜) :
    (mfderiv I 𝓘(𝕜, E') (s • f) z : TangentSpace I z →L[𝕜] E') =
      (s • mfderiv I 𝓘(𝕜, E') f z : TangentSpace I z →L[𝕜] E') :=
  (hf.hasMFDerivAt.const_smul s).mfderiv


theorem HasMFDerivAt.neg (hf : HasMFDerivAt I 𝓘(𝕜, E') f z f') :
    HasMFDerivAt I 𝓘(𝕜, E') (-f) z (-f') :=
  ⟨hf.1.neg, hf.2.neg⟩


theorem hasMFDerivAt_neg : HasMFDerivAt I 𝓘(𝕜, E') (-f) z (-f') ↔ HasMFDerivAt I 𝓘(𝕜, E') f z f' :=
                /-
                  𝕜 : Type u_1
                  inst✝⁷ : NontriviallyNormedField 𝕜
                  E : Type u_2
                  inst✝⁶ : NormedAddCommGroup E
                  inst✝⁵ : NormedSpace 𝕜 E
                  H : Type u_3
                  inst✝⁴ : TopologicalSpace H
                  I : ModelWithCorners 𝕜 E H
                  M : Type u_4
                  inst✝³ : TopologicalSpace M
                  inst✝² : ChartedSpace H M
                  E' : Type u_5
                  inst✝¹ : NormedAddCommGroup E'
                  inst✝ : NormedSpace 𝕜 E'
                  z : M
                  f : M → E'
                  f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
                  hf : HasMFDerivAt I (modelWithCornersSelf 𝕜 E') (Neg.neg f) z (Neg.neg f')
                  ⊢ HasMFDerivAt I (modelWithCornersSelf 𝕜 E') f z f'
                -/
                                   /-
                                     🎉 no goals
                                   -/
  ⟨fun hf => by convert hf.neg <;> rw [neg_neg], fun hf => hf.neg⟩
                                   /-
                                     🎉 no goals
                                   -/


theorem MDifferentiableAt.neg (hf : MDifferentiableAt I 𝓘(𝕜, E') f z) :
    MDifferentiableAt I 𝓘(𝕜, E') (-f) z :=
  hf.hasMFDerivAt.neg.mdifferentiableAt


theorem mdifferentiableAt_neg :
    MDifferentiableAt I 𝓘(𝕜, E') (-f) z ↔ MDifferentiableAt I 𝓘(𝕜, E') f z :=
                /-
                  𝕜 : Type u_1
                  inst✝⁷ : NontriviallyNormedField 𝕜
                  E : Type u_2
                  inst✝⁶ : NormedAddCommGroup E
                  inst✝⁵ : NormedSpace 𝕜 E
                  H : Type u_3
                  inst✝⁴ : TopologicalSpace H
                  I : ModelWithCorners 𝕜 E H
                  M : Type u_4
                  inst✝³ : TopologicalSpace M
                  inst✝² : ChartedSpace H M
                  E' : Type u_5
                  inst✝¹ : NormedAddCommGroup E'
                  inst✝ : NormedSpace 𝕜 E'
                  z : M
                  f : M → E'
                  hf : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') (Neg.neg f) z
                  ⊢ MDifferentiableAt I (modelWithCornersSelf 𝕜 E') f z
                -/
  ⟨fun hf => by convert hf.neg; rw [neg_neg], fun hf => hf.neg⟩
                                /-
                                  🎉 no goals
                                -/


theorem MDifferentiable.neg (hf : MDifferentiable I 𝓘(𝕜, E') f) : MDifferentiable I 𝓘(𝕜, E') (-f) :=
  fun x => (hf x).neg


theorem mfderiv_neg (f : M → E') (x : M) :
    (mfderiv I 𝓘(𝕜, E') (-f) x : TangentSpace I x →L[𝕜] E') =
      (-mfderiv I 𝓘(𝕜, E') f x : TangentSpace I x →L[𝕜] E') := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    E' : Type u_5
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : M → E'
    x : M
    ⊢ Eq (mfderiv I (modelWithCornersSelf 𝕜 E') (Neg.neg f) x) (Neg.neg (mfderiv I …
  -/
  simp_rw [mfderiv]
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    E' : Type u_5
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace 𝕜 E'
    f : M → E'
    x : M
    ⊢ Eq (ite (MDifferentiableAt I (modelWithCornersSelf 𝕜 E') (Neg.neg f) x) (fde …
  -/
  by_cases hf : MDifferentiableAt I 𝓘(𝕜, E') f x
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      E' : Type u_5
      inst✝¹ : NormedAddCommGroup E'
      inst✝ : NormedSpace 𝕜 E'
      f : M → E'
      x : M
      hf : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') f x
      ⊢ Eq (ite (MDifferentiableAt I (modelWithCornersSelf 𝕜 E') (Neg.neg f) x) (fde …
    -/
  · exact hf.hasMFDerivAt.neg.mfderiv
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      E' : Type u_5
      inst✝¹ : NormedAddCommGroup E'
      inst✝ : NormedSpace 𝕜 E'
      f : M → E'
      x : M
      hf : Not (MDifferentiableAt I (modelWithCornersSelf 𝕜 E') f x)
      ⊢ Eq (ite (MDifferentiableAt I (modelWithCornersSelf 𝕜 E') (Neg.neg f) x) (fde …
    -/
  · rw [if_neg hf]; rw [← mdifferentiableAt_neg] at hf; rw [if_neg hf, neg_zero]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem HasMFDerivAt.sub (hf : HasMFDerivAt I 𝓘(𝕜, E') f z f')
    (hg : HasMFDerivAt I 𝓘(𝕜, E') g z g') : HasMFDerivAt I 𝓘(𝕜, E') (f - g) z (f' - g') :=
  ⟨hf.1.sub hg.1, hf.2.sub hg.2⟩


theorem MDifferentiableAt.sub (hf : MDifferentiableAt I 𝓘(𝕜, E') f z)
    (hg : MDifferentiableAt I 𝓘(𝕜, E') g z) : MDifferentiableAt I 𝓘(𝕜, E') (f - g) z :=
  (hf.hasMFDerivAt.sub hg.hasMFDerivAt).mdifferentiableAt


theorem MDifferentiable.sub (hf : MDifferentiable I 𝓘(𝕜, E') f)
    (hg : MDifferentiable I 𝓘(𝕜, E') g) : MDifferentiable I 𝓘(𝕜, E') (f - g) := fun x =>
  (hf x).sub (hg x)


theorem mfderiv_sub (hf : MDifferentiableAt I 𝓘(𝕜, E') f z)
    (hg : MDifferentiableAt I 𝓘(𝕜, E') g z) :
        /-
          𝕜 : Type u_1
          inst✝³³ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝³² : NormedAddCommGroup E
          inst✝³¹ : NormedSpace 𝕜 E
          H : Type u_3
          inst✝³⁰ : TopologicalSpace H
          I : ModelWithCorners 𝕜 E H
          M : Type u_4
          inst✝²⁹ : TopologicalSpace M
          inst✝²⁸ : ChartedSpace H M
          E' : Type u_5
          inst✝²⁷ : NormedAddCommGroup E'
          inst✝²⁶ : NormedSpace 𝕜 E'
          H' : Type u_6
          inst✝²⁵ : TopologicalSpace H'
          I' : ModelWithCorners 𝕜 E' H'
          M' : Type u_7
          inst✝²⁴ : TopologicalSpace M'
          inst✝²³ : ChartedSpace H' M'
          E'' : Type u_8
          inst✝²² : NormedAddCommGroup E''
          inst✝²¹ : NormedSpace 𝕜 E''
          H'' : Type u_9
          inst✝²⁰ : TopologicalSpace H''
          I'' : ModelWithCorners 𝕜 E'' H''
          M'' : Type u_10
          inst✝¹⁹ : TopologicalSpace M''
          inst✝¹⁸ : ChartedSpace H'' M''
          F : Type u_11
          inst✝¹⁷ : NormedAddCommGroup F
          inst✝¹⁶ : NormedSpace 𝕜 F
          G : Type u_12
          inst✝¹⁵ : TopologicalSpace G
          J : ModelWithCorners 𝕜 F G
          N : Type u_13
          inst✝¹⁴ : TopologicalSpace N
          inst✝¹³ : ChartedSpace G N
          F' : Type u_14
          inst✝¹² : NormedAddCommGroup F'
          inst✝¹¹ : NormedSpace 𝕜 F'
          G' : Type u_15
          inst✝¹⁰ : TopologicalSpace G'
          J' : ModelWithCorners 𝕜 F' G'
          N' : Type u_16
          inst✝⁹ : TopologicalSpace N'
          inst✝⁸ : ChartedSpace G' N'
          F₁ : Type u_17
          inst✝⁷ : NormedAddCommGroup F₁
          inst✝⁶ : NormedSpace 𝕜 F₁
          F₂ : Type u_18
          inst✝⁵ : NormedAddCommGroup F₂
          inst✝⁴ : NormedSpace 𝕜 F₂
          F₃ : Type u_19
          inst✝³ : NormedAddCommGroup F₃
          inst✝² : NormedSpace 𝕜 F₃
          F₄ : Type u_20
          inst✝¹ : NormedAddCommGroup F₄
          inst✝ : NormedSpace 𝕜 F₄
          s : Set M
          x z : M
          f g : M → E'
          f' g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
          hf : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') f z
          hg : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') g z
          ⊢ ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
        -/
    (by exact mfderiv I 𝓘(𝕜, E') (f - g) z : TangentSpace I z →L[𝕜] E') =
        /-
          🎉 no goals
        -/
          /-
            𝕜 : Type u_1
            inst✝³³ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝³² : NormedAddCommGroup E
            inst✝³¹ : NormedSpace 𝕜 E
            H : Type u_3
            inst✝³⁰ : TopologicalSpace H
            I : ModelWithCorners 𝕜 E H
            M : Type u_4
            inst✝²⁹ : TopologicalSpace M
            inst✝²⁸ : ChartedSpace H M
            E' : Type u_5
            inst✝²⁷ : NormedAddCommGroup E'
            inst✝²⁶ : NormedSpace 𝕜 E'
            H' : Type u_6
            inst✝²⁵ : TopologicalSpace H'
            I' : ModelWithCorners 𝕜 E' H'
            M' : Type u_7
            inst✝²⁴ : TopologicalSpace M'
            inst✝²³ : ChartedSpace H' M'
            E'' : Type u_8
            inst✝²² : NormedAddCommGroup E''
            inst✝²¹ : NormedSpace 𝕜 E''
            H'' : Type u_9
            inst✝²⁰ : TopologicalSpace H''
            I'' : ModelWithCorners 𝕜 E'' H''
            M'' : Type u_10
            inst✝¹⁹ : TopologicalSpace M''
            inst✝¹⁸ : ChartedSpace H'' M''
            F : Type u_11
            inst✝¹⁷ : NormedAddCommGroup F
            inst✝¹⁶ : NormedSpace 𝕜 F
            G : Type u_12
            inst✝¹⁵ : TopologicalSpace G
            J : ModelWithCorners 𝕜 F G
            N : Type u_13
            inst✝¹⁴ : TopologicalSpace N
            inst✝¹³ : ChartedSpace G N
            F' : Type u_14
            inst✝¹² : NormedAddCommGroup F'
            inst✝¹¹ : NormedSpace 𝕜 F'
            G' : Type u_15
            inst✝¹⁰ : TopologicalSpace G'
            J' : ModelWithCorners 𝕜 F' G'
            N' : Type u_16
            inst✝⁹ : TopologicalSpace N'
            inst✝⁸ : ChartedSpace G' N'
            F₁ : Type u_17
            inst✝⁷ : NormedAddCommGroup F₁
            inst✝⁶ : NormedSpace 𝕜 F₁
            F₂ : Type u_18
            inst✝⁵ : NormedAddCommGroup F₂
            inst✝⁴ : NormedSpace 𝕜 F₂
            F₃ : Type u_19
            inst✝³ : NormedAddCommGroup F₃
            inst✝² : NormedSpace 𝕜 F₃
            F₄ : Type u_20
            inst✝¹ : NormedAddCommGroup F₄
            inst✝ : NormedSpace 𝕜 F₄
            s : Set M
            x z : M
            f g : M → E'
            f' g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
            hf : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') f z
            hg : MDifferentiableAt I (modelWithCornersSelf 𝕜 E') g z
            ⊢ ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) E'
          -/
          /-
            🎉 no goals
          -/
      (by exact mfderiv I 𝓘(𝕜, E') f z) - (by exact mfderiv I 𝓘(𝕜, E') g z) :=
                                              /-
                                                🎉 no goals
                                              -/
  (hf.hasMFDerivAt.sub hg.hasMFDerivAt).mfderiv


theorem HasMFDerivWithinAt.mul' (hp : HasMFDerivWithinAt I 𝓘(𝕜, F') p s z p')
    (hq : HasMFDerivWithinAt I 𝓘(𝕜, F') q s z q') :
    HasMFDerivWithinAt I 𝓘(𝕜, F') (p * q) s z (p z • q' + p'.smulRight (q z) : E →L[𝕜] F') :=
                     /-
                       𝕜 : Type u_1
                       inst✝⁷ : NontriviallyNormedField 𝕜
                       E : Type u_2
                       inst✝⁶ : NormedAddCommGroup E
                       inst✝⁵ : NormedSpace 𝕜 E
                       H : Type u_3
                       inst✝⁴ : TopologicalSpace H
                       I : ModelWithCorners 𝕜 E H
                       M : Type u_4
                       inst✝³ : TopologicalSpace M
                       inst✝² : ChartedSpace H M
                       s : Set M
                       z : M
                       F' : Type u_21
                       inst✝¹ : NormedRing F'
                       inst✝ : NormedAlgebra 𝕜 F'
                       p q : M → F'
                       p' q' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) F'
                       hp : HasMFDerivWithinAt I (modelWithCornersSelf 𝕜 F') p s z p'
                       hq : HasMFDerivWithinAt I (modelWithCornersSelf 𝕜 F') q s z q'
                       ⊢ HasFDerivWithinAt (writtenInExtChartAt I (modelWithCornersSelf 𝕜 F') z (HMul …
                     -/
  ⟨hp.1.mul hq.1, by simpa only [mfld_simps] using hp.2.mul' hq.2⟩
                     /-
                       🎉 no goals
                     -/


theorem HasMFDerivAt.mul' (hp : HasMFDerivAt I 𝓘(𝕜, F') p z p')
    (hq : HasMFDerivAt I 𝓘(𝕜, F') q z q') :
    HasMFDerivAt I 𝓘(𝕜, F') (p * q) z (p z • q' + p'.smulRight (q z) : E →L[𝕜] F') :=
  hasMFDerivWithinAt_univ.mp <| hp.hasMFDerivWithinAt.mul' hq.hasMFDerivWithinAt


theorem MDifferentiableWithinAt.mul (hp : MDifferentiableWithinAt I 𝓘(𝕜, F') p s z)
    (hq : MDifferentiableWithinAt I 𝓘(𝕜, F') q s z) :
    MDifferentiableWithinAt I 𝓘(𝕜, F') (p * q) s z :=
  (hp.hasMFDerivWithinAt.mul' hq.hasMFDerivWithinAt).mdifferentiableWithinAt


theorem MDifferentiableAt.mul (hp : MDifferentiableAt I 𝓘(𝕜, F') p z)
    (hq : MDifferentiableAt I 𝓘(𝕜, F') q z) : MDifferentiableAt I 𝓘(𝕜, F') (p * q) z :=
  (hp.hasMFDerivAt.mul' hq.hasMFDerivAt).mdifferentiableAt


theorem MDifferentiableOn.mul (hp : MDifferentiableOn I 𝓘(𝕜, F') p s)
    (hq : MDifferentiableOn I 𝓘(𝕜, F') q s) : MDifferentiableOn I 𝓘(𝕜, F') (p * q) s := fun x hx =>
  (hp x hx).mul <| hq x hx


theorem MDifferentiable.mul (hp : MDifferentiable I 𝓘(𝕜, F') p)
    (hq : MDifferentiable I 𝓘(𝕜, F') q) : MDifferentiable I 𝓘(𝕜, F') (p * q) := fun x =>
  (hp x).mul (hq x)


theorem HasMFDerivWithinAt.mul (hp : HasMFDerivWithinAt I 𝓘(𝕜, F') p s z p')
    (hq : HasMFDerivWithinAt I 𝓘(𝕜, F') q s z q') :
    HasMFDerivWithinAt I 𝓘(𝕜, F') (p * q) s z (p z • q' + q z • p' : E →L[𝕜] F') := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    s : Set M
    z : M
    F' : Type u_21
    inst✝¹ : NormedCommRing F'
    inst✝ : NormedAlgebra 𝕜 F'
    p q : M → F'
    p' q' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I z) F'
    hp : HasMFDerivWithinAt I (modelWithCornersSelf 𝕜 F') p s z p'
    hq : HasMFDerivWithinAt I (modelWithCornersSelf 𝕜 F') q s z q'
    ⊢ HasMFDerivWithinAt I (modelWithCornersSelf 𝕜 F') (HMul.hMul p q) s z (HAdd.h …
  -/
  convert hp.mul' hq; ext _; apply mul_comm
                             /-
                               🎉 no goals
                             -/


theorem HasMFDerivAt.mul (hp : HasMFDerivAt I 𝓘(𝕜, F') p z p')
    (hq : HasMFDerivAt I 𝓘(𝕜, F') q z q') :
    HasMFDerivAt I 𝓘(𝕜, F') (p * q) z (p z • q' + q z • p' : E →L[𝕜] F') :=
  hasMFDerivWithinAt_univ.mp <| hp.hasMFDerivWithinAt.mul hq.hasMFDerivWithinAt


