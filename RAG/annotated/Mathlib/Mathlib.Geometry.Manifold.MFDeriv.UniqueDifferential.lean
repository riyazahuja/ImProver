/-- If `s` has the unique differential property at `x`, `f` is differentiable within `s` at x` and
its derivative has dense range, then `f '' s` has the unique differential property at `f x`. -/
theorem UniqueMDiffWithinAt.image_denseRange (hs : UniqueMDiffWithinAt I s x)
    {f : M → M'} {f' : E →L[𝕜] E'} (hf : HasMFDerivWithinAt I I' f s x f')
    (hd : DenseRange f') : UniqueMDiffWithinAt I' (f '' s) (f x) := by
  /- Rewrite in coordinates, apply `HasFDerivWithinAt.uniqueDiffWithinAt`. -/
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
    s : Set M
    x : M
    hs : UniqueMDiffWithinAt I s x
    f : M → M'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    hf : HasMFDerivWithinAt I I' f s x f'
    hd : DenseRange ⇑f'
    ⊢ UniqueMDiffWithinAt I' (Set.image f s) (f x)
  -/
  have := hs.inter' <| hf.1 (extChartAt_source_mem_nhds (I := I') (f x))
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
    s : Set M
    x : M
    hs : UniqueMDiffWithinAt I s x
    f : M → M'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    hf : HasMFDerivWithinAt I I' f s x f'
    hd : DenseRange ⇑f'
    this : UniqueMDiffWithinAt I (Inter.inter s (Set.preimage f (extChartAt I' (f  …
    ⊢ UniqueMDiffWithinAt I' (Set.image f s) (f x)
  -/
  refine (((hf.2.mono ?sub1).uniqueDiffWithinAt this hd).mono ?sub2).congr_pt ?pt
  /-
    case sub1
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
    s : Set M
    x : M
    hs : UniqueMDiffWithinAt I s x
    f : M → M'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    hf : HasMFDerivWithinAt I I' f s x f'
    hd : DenseRange ⇑f'
    this : UniqueMDiffWithinAt I (Inter.inter s (Set.preimage f (extChartAt I' (f  …
    ⊢ HasSubset.Subset (Inter.inter (Set.preimage (↑(extChartAt I x).symm) (Inter. …
  -/
  case pt => simp only [mfld_simps]
  /-
    case sub1
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
    s : Set M
    x : M
    hs : UniqueMDiffWithinAt I s x
    f : M → M'
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    hf : HasMFDerivWithinAt I I' f s x f'
    hd : DenseRange ⇑f'
    this : UniqueMDiffWithinAt I (Inter.inter s (Set.preimage f (extChartAt I' (f  …
    ⊢ HasSubset.Subset (Inter.inter (Set.preimage (↑(extChartAt I x).symm) (Inter. …
  -/
  case sub1 => mfld_set_tac
  case sub2 =>
    rintro _ ⟨y, ⟨⟨hys, hfy⟩, -⟩, rfl⟩
    exact ⟨⟨_, hys, ((extChartAt I' (f x)).left_inv hfy).symm⟩, mem_range_self _⟩


/-- If `s` has the unique differential property, `f` is differentiable on `s` and its derivative
at every point of `s` has dense range, then `f '' s` has the unique differential property.
This version uses the `HasMFDerivWithinAt` predicate. -/
theorem UniqueMDiffOn.image_denseRange' (hs : UniqueMDiffOn I s) {f : M → M'}
    {f' : M → E →L[𝕜] E'} (hf : ∀ x ∈ s, HasMFDerivWithinAt I I' f s x (f' x))
    (hd : ∀ x ∈ s, DenseRange (f' x)) :
    UniqueMDiffOn I' (f '' s) :=
  forall_mem_image.2 fun x hx ↦ (hs x hx).image_denseRange (hf x hx) (hd x hx)


/-- If `s` has the unique differential property, `f` is differentiable on `s` and its derivative
at every point of `s` has dense range, then `f '' s` has the unique differential property. -/
theorem UniqueMDiffOn.image_denseRange (hs : UniqueMDiffOn I s) {f : M → M'}
    (hf : MDifferentiableOn I I' f s) (hd : ∀ x ∈ s, DenseRange (mfderivWithin I I' f s x)) :
    UniqueMDiffOn I' (f '' s) :=
  hs.image_denseRange' (fun x hx ↦ (hf x hx).hasMFDerivWithinAt) hd


protected theorem UniqueMDiffWithinAt.preimage_partialHomeomorph (hs : UniqueMDiffWithinAt I s x)
    {e : PartialHomeomorph M M'} (he : e.MDifferentiable I I') (hx : x ∈ e.source) :
    UniqueMDiffWithinAt I' (e.target ∩ e.symm ⁻¹' s) (e x) := by
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
    s : Set M
    x : M
    hs : UniqueMDiffWithinAt I s x
    e : PartialHomeomorph M M'
    he : PartialHomeomorph.MDifferentiable I I' e
    hx : Membership.mem e.source x
    ⊢ UniqueMDiffWithinAt I' (Inter.inter e.target (Set.preimage (↑e.symm) s)) (↑e …
  -/
  rw [← e.image_source_inter_eq', inter_comm]
  exact (hs.inter (e.open_source.mem_nhds hx)).image_denseRange
    (he.mdifferentiableAt hx).hasMFDerivAt.hasMFDerivWithinAt
    (he.mfderiv_surjective hx).denseRange


/-- If a set has the unique differential property, then its image under a local
diffeomorphism also has the unique differential property. -/
theorem UniqueMDiffOn.uniqueMDiffOn_preimage (hs : UniqueMDiffOn I s) {e : PartialHomeomorph M M'}
    (he : e.MDifferentiable I I') : UniqueMDiffOn I' (e.target ∩ e.symm ⁻¹' s) := fun _x hx ↦
  e.right_inv hx.1 ▸ (hs _ hx.2).preimage_partialHomeomorph he (e.map_target hx.1)


variable [SmoothManifoldWithCorners I M]  in
/-- If a set in a manifold has the unique derivative property, then its pullback by any extended
chart, in the vector space, also has the unique derivative property. -/
theorem UniqueMDiffOn.uniqueMDiffOn_target_inter (hs : UniqueMDiffOn I s) (x : M) :
    UniqueMDiffOn 𝓘(𝕜, E) ((extChartAt I x).target ∩ (extChartAt I x).symm ⁻¹' s) := by
  -- this is just a reformulation of `UniqueMDiffOn.uniqueMDiffOn_preimage`, using as `e`
  -- the local chart at `x`.
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
    s : Set M
    inst✝ : SmoothManifoldWithCorners I M
    hs : UniqueMDiffOn I s
    x : M
    ⊢ UniqueMDiffOn (modelWithCornersSelf 𝕜 E) (Inter.inter (extChartAt I x).targe …
  -/
  rw [← PartialEquiv.image_source_inter_eq', inter_comm, extChartAt_source]
  exact (hs.inter (chartAt H x).open_source).image_denseRange'
    (fun y hy ↦ hasMFDerivWithinAt_extChartAt hy.2)
    fun y hy ↦ ((mdifferentiable_chart _).mfderiv_surjective hy.2).denseRange


variable [SmoothManifoldWithCorners I M]  in
/-- If a set in a manifold has the unique derivative property, then its pullback by any extended
chart, in the vector space, also has the unique derivative property. -/
theorem UniqueMDiffOn.uniqueDiffOn_target_inter (hs : UniqueMDiffOn I s) (x : M) :
    UniqueDiffOn 𝕜 ((extChartAt I x).target ∩ (extChartAt I x).symm ⁻¹' s) :=
  (hs.uniqueMDiffOn_target_inter x).uniqueDiffOn


variable [SmoothManifoldWithCorners I M]  in
theorem UniqueMDiffOn.uniqueDiffWithinAt_range_inter (hs : UniqueMDiffOn I s) (x : M) (y : E)
    (hy : y ∈ (extChartAt I x).target ∩ (extChartAt I x).symm ⁻¹' s) :
    UniqueDiffWithinAt 𝕜 (range I ∩ (extChartAt I x).symm ⁻¹' s) y := by
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
    s : Set M
    inst✝ : SmoothManifoldWithCorners I M
    hs : UniqueMDiffOn I s
    x : M
    y : E
    hy : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
    ⊢ UniqueDiffWithinAt 𝕜 (Inter.inter (Set.range ↑I) (Set.preimage (↑(extChartAt …
  -/
  apply (hs.uniqueDiffOn_target_inter x y hy).mono
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
    s : Set M
    inst✝ : SmoothManifoldWithCorners I M
    hs : UniqueMDiffOn I s
    x : M
    y : E
    hy : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
    ⊢ HasSubset.Subset (Inter.inter (extChartAt I x).target (Set.preimage (↑(extCh …
  -/
  apply inter_subset_inter_left _ (extChartAt_target_subset_range x)
  /-
    🎉 no goals
  -/


variable [SmoothManifoldWithCorners I M]  in
/-- When considering functions between manifolds, this statement shows up often. It entails
the unique differential of the pullback in extended charts of the set where the function can
be read in the charts. -/
theorem UniqueMDiffOn.uniqueDiffOn_inter_preimage (hs : UniqueMDiffOn I s) (x : M) (y : M'')
    {f : M → M''} (hf : ContinuousOn f s) :
    UniqueDiffOn 𝕜
      ((extChartAt I x).target ∩ (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' y).source)) :=
  haveI : UniqueMDiffOn I (s ∩ f ⁻¹' (extChartAt I' y).source) := by
    /-
      𝕜 : Type u_1
      inst✝¹¹ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁸ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      E' : Type u_5
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝³ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M'' : Type u_8
      inst✝² : TopologicalSpace M''
      inst✝¹ : ChartedSpace H' M''
      s : Set M
      inst✝ : SmoothManifoldWithCorners I M
      hs : UniqueMDiffOn I s
      x : M
      y : M''
      f : M → M''
      hf : ContinuousOn f s
      ⊢ UniqueMDiffOn I (Inter.inter s (Set.preimage f (extChartAt I' y).source))
    -/
    intro z hz
    /-
      𝕜 : Type u_1
      inst✝¹¹ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁸ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      E' : Type u_5
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝³ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M'' : Type u_8
      inst✝² : TopologicalSpace M''
      inst✝¹ : ChartedSpace H' M''
      s : Set M
      inst✝ : SmoothManifoldWithCorners I M
      hs : UniqueMDiffOn I s
      x : M
      y : M''
      f : M → M''
      hf : ContinuousOn f s
      z : M
      hz : Membership.mem (Inter.inter s (Set.preimage f (extChartAt I' y).source)) z
      ⊢ UniqueMDiffWithinAt I (Inter.inter s (Set.preimage f (extChartAt I' y).sourc …
    -/
    apply (hs z hz.1).inter'
    /-
      𝕜 : Type u_1
      inst✝¹¹ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁸ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      E' : Type u_5
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝³ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M'' : Type u_8
      inst✝² : TopologicalSpace M''
      inst✝¹ : ChartedSpace H' M''
      s : Set M
      inst✝ : SmoothManifoldWithCorners I M
      hs : UniqueMDiffOn I s
      x : M
      y : M''
      f : M → M''
      hf : ContinuousOn f s
      z : M
      hz : Membership.mem (Inter.inter s (Set.preimage f (extChartAt I' y).source)) z
      ⊢ Membership.mem (nhdsWithin z s) (Set.preimage f (extChartAt I' y).source)
    -/
    apply (hf z hz.1).preimage_mem_nhdsWithin
    /-
      𝕜 : Type u_1
      inst✝¹¹ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁸ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      E' : Type u_5
      inst✝⁵ : NormedAddCommGroup E'
      inst✝⁴ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝³ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M'' : Type u_8
      inst✝² : TopologicalSpace M''
      inst✝¹ : ChartedSpace H' M''
      s : Set M
      inst✝ : SmoothManifoldWithCorners I M
      hs : UniqueMDiffOn I s
      x : M
      y : M''
      f : M → M''
      hf : ContinuousOn f s
      z : M
      hz : Membership.mem (Inter.inter s (Set.preimage f (extChartAt I' y).source)) z
      ⊢ Membership.mem (nhds (f z)) (extChartAt I' y).source
    -/
    exact (isOpen_extChartAt_source y).mem_nhds hz.2
    /-
      🎉 no goals
    -/
  this.uniqueDiffOn_target_inter _


private lemma UniqueMDiffWithinAt.bundle_preimage_aux {p : TotalSpace F Z}
    (hs : UniqueMDiffWithinAt I s p.proj) (h's : s ⊆ (trivializationAt F Z p.proj).baseSet) :
    UniqueMDiffWithinAt (I.prod 𝓘(𝕜, F)) (π F Z ⁻¹' s) p := by
  suffices ((extChartAt I p.proj).symm ⁻¹' s ∩ range I) ×ˢ univ ⊆
      (extChartAt (I.prod 𝓘(𝕜, F)) p).symm ⁻¹' (TotalSpace.proj ⁻¹' s) ∩ range (I.prod 𝓘(𝕜, F)) by
    let w := (extChartAt (I.prod 𝓘(𝕜, F)) p p).2
    have A : extChartAt (I.prod 𝓘(𝕜, F)) p p = (extChartAt I p.1 p.1, w) := by
      ext
      · simp [FiberBundle.chartedSpace_chartAt]
      · rfl
    simp only [UniqueMDiffWithinAt, A] at hs ⊢
    exact (hs.prod (uniqueDiffWithinAt_univ (x := w))).mono this
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
    s : Set M
    F : Type u_9
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    Z : M → Type u_10
    inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
    inst✝¹ : (b : M) → TopologicalSpace (Z b)
    inst✝ : FiberBundle F Z
    p : Bundle.TotalSpace F Z
    hs : UniqueMDiffWithinAt I s p.proj
    h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z p.proj).baseSet
    ⊢ HasSubset.Subset (SProd.sprod (Inter.inter (Set.preimage (↑(extChartAt I p.p …
  -/
  rcases p with ⟨x, v⟩
  /-
    case mk
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
    s : Set M
    F : Type u_9
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    Z : M → Type u_10
    inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
    inst✝¹ : (b : M) → TopologicalSpace (Z b)
    inst✝ : FiberBundle F Z
    x : M
    v : Z x
    hs : UniqueMDiffWithinAt I s { proj := x, snd := v }.proj
    h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z { proj := x, snd := …
    ⊢ HasSubset.Subset (SProd.sprod (Inter.inter (Set.preimage (↑(extChartAt I { p …
  -/
  dsimp
  /-
    case mk
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
    s : Set M
    F : Type u_9
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    Z : M → Type u_10
    inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
    inst✝¹ : (b : M) → TopologicalSpace (Z b)
    inst✝ : FiberBundle F Z
    x : M
    v : Z x
    hs : UniqueMDiffWithinAt I s { proj := x, snd := v }.proj
    h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z { proj := x, snd := …
    ⊢ HasSubset.Subset (SProd.sprod (Inter.inter (Set.preimage (Function.comp ↑(ch …
  -/
  rintro ⟨z, w⟩ ⟨hz, -⟩
  simp only [ModelWithCorners.target_eq, mem_inter_iff, mem_preimage, Function.comp_apply,
    mem_range] at hz
  simp only [FiberBundle.chartedSpace_chartAt, PartialHomeomorph.coe_trans_symm, mem_inter_iff,
    mem_preimage, Function.comp_apply, mem_range]
  /-
    case mk.mk.intro
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
    s : Set M
    F : Type u_9
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    Z : M → Type u_10
    inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
    inst✝¹ : (b : M) → TopologicalSpace (Z b)
    inst✝ : FiberBundle F Z
    x : M
    v : Z x
    hs : UniqueMDiffWithinAt I s { proj := x, snd := v }.proj
    h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z { proj := x, snd := …
    z : E
    w : F
    hz : And (Membership.mem s (↑(chartAt H x).symm (↑I.symm z))) (Exists fun y => …
    ⊢ And (Membership.mem s (↑(FiberBundle.trivializationAt F Z x).symm (↑((chartA …
  -/
  constructor
  · rw [PartialEquiv.prod_symm, PartialEquiv.refl_symm, PartialEquiv.prod_coe,
      ModelWithCorners.toPartialEquiv_coe_symm, PartialEquiv.refl_coe,
      PartialHomeomorph.prod_symm, PartialHomeomorph.refl_symm, PartialHomeomorph.prod_apply,
      PartialHomeomorph.refl_apply]
    /-
      case mk.mk.intro.left
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
      s : Set M
      F : Type u_9
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      Z : M → Type u_10
      inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
      inst✝¹ : (b : M) → TopologicalSpace (Z b)
      inst✝ : FiberBundle F Z
      x : M
      v : Z x
      hs : UniqueMDiffWithinAt I s { proj := x, snd := v }.proj
      h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z { proj := x, snd := …
      z : E
      w : F
      hz : And (Membership.mem s (↑(chartAt H x).symm (↑I.symm z))) (Exists fun y => …
      ⊢ Membership.mem s (↑(FiberBundle.trivializationAt F Z x).symm ((fun p => { fs …
    -/
    convert hz.1
    /-
      case h.e'_5
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
      s : Set M
      F : Type u_9
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      Z : M → Type u_10
      inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
      inst✝¹ : (b : M) → TopologicalSpace (Z b)
      inst✝ : FiberBundle F Z
      x : M
      v : Z x
      hs : UniqueMDiffWithinAt I s { proj := x, snd := v }.proj
      h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z { proj := x, snd := …
      z : E
      w : F
      hz : And (Membership.mem s (↑(chartAt H x).symm (↑I.symm z))) (Exists fun y => …
      ⊢ Eq (↑(FiberBundle.trivializationAt F Z x).symm ((fun p => { fst := ↑(chartAt …
    -/
    apply Trivialization.proj_symm_apply'
    /-
      case h.e'_5.hx
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
      s : Set M
      F : Type u_9
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      Z : M → Type u_10
      inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
      inst✝¹ : (b : M) → TopologicalSpace (Z b)
      inst✝ : FiberBundle F Z
      x : M
      v : Z x
      hs : UniqueMDiffWithinAt I s { proj := x, snd := v }.proj
      h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z { proj := x, snd := …
      z : E
      w : F
      hz : And (Membership.mem s (↑(chartAt H x).symm (↑I.symm z))) (Exists fun y => …
      ⊢ Membership.mem (FiberBundle.trivializationAt F Z x).baseSet (↑(chartAt H x). …
    -/
    exact h's hz.1
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.intro.right
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
      s : Set M
      F : Type u_9
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      Z : M → Type u_10
      inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
      inst✝¹ : (b : M) → TopologicalSpace (Z b)
      inst✝ : FiberBundle F Z
      x : M
      v : Z x
      hs : UniqueMDiffWithinAt I s { proj := x, snd := v }.proj
      h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z { proj := x, snd := …
      z : E
      w : F
      hz : And (Membership.mem s (↑(chartAt H x).symm (↑I.symm z))) (Exists fun y => …
      ⊢ Exists fun y => Eq (Prod.map (↑I) id y) { fst := z, snd := w }
    -/
  · rcases hz.2 with ⟨u, rfl⟩
    /-
      case mk.mk.intro.right.intro
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
      s : Set M
      F : Type u_9
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      Z : M → Type u_10
      inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
      inst✝¹ : (b : M) → TopologicalSpace (Z b)
      inst✝ : FiberBundle F Z
      x : M
      v : Z x
      hs : UniqueMDiffWithinAt I s { proj := x, snd := v }.proj
      h's : HasSubset.Subset s (FiberBundle.trivializationAt F Z { proj := x, snd := …
      w : F
      u : H
      hz : And (Membership.mem s (↑(chartAt H x).symm (↑I.symm (↑I u)))) (Exists fun …
      ⊢ Exists fun y => Eq (Prod.map (↑I) id y) { fst := ↑I u, snd := w }
    -/
    exact ⟨(u, w), rfl⟩
    /-
      🎉 no goals
    -/


/-- In a fiber bundle, the preimage under the projection of a set with unique differentials
in the base has unique differentials in the bundle. -/
theorem UniqueMDiffWithinAt.bundle_preimage {p : TotalSpace F Z}
    (hs : UniqueMDiffWithinAt I s p.proj) :
    UniqueMDiffWithinAt (I.prod 𝓘(𝕜, F)) (π F Z ⁻¹' s) p := by
  suffices UniqueMDiffWithinAt (I.prod 𝓘(𝕜, F))
    (π F Z ⁻¹' (s ∩ (trivializationAt F Z p.proj).baseSet)) p from this.mono (by simp)
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
    s : Set M
    F : Type u_9
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    Z : M → Type u_10
    inst✝² : TopologicalSpace (Bundle.TotalSpace F Z)
    inst✝¹ : (b : M) → TopologicalSpace (Z b)
    inst✝ : FiberBundle F Z
    p : Bundle.TotalSpace F Z
    hs : UniqueMDiffWithinAt I s p.proj
    ⊢ UniqueMDiffWithinAt (I.prod (modelWithCornersSelf 𝕜 F)) (Set.preimage Bundle …
  -/
  apply UniqueMDiffWithinAt.bundle_preimage_aux (hs.inter _) inter_subset_right
  exact IsOpen.mem_nhds (trivializationAt F Z p.proj).open_baseSet
    (FiberBundle.mem_baseSet_trivializationAt' p.proj)


@[deprecated (since := "2024-12-02")]
alias UniqueMDiffWithinAt.smooth_bundle_preimage := UniqueMDiffWithinAt.bundle_preimage


/-- In a fiber bundle, the preimage under the projection of a set with unique differentials
in the base has unique differentials in the bundle. Version with a point `⟨b, x⟩`. -/
theorem UniqueMDiffWithinAt.bundle_preimage' {b : M} (hs : UniqueMDiffWithinAt I s b)
    (x : Z b) : UniqueMDiffWithinAt (I.prod 𝓘(𝕜, F)) (π F Z ⁻¹' s) ⟨b, x⟩ :=
  hs.bundle_preimage (p := ⟨b, x⟩)


@[deprecated (since := "2024-12-02")]
alias UniqueMDiffWithinAt.smooth_bundle_preimage' := UniqueMDiffWithinAt.bundle_preimage'


/-- In a fiber bundle, the preimage under the projection of a set with unique differentials
in the base has unique differentials in the bundle. -/
theorem UniqueMDiffOn.bundle_preimage (hs : UniqueMDiffOn I s) :
    UniqueMDiffOn (I.prod 𝓘(𝕜, F)) (π F Z ⁻¹' s) := fun _p hp ↦
  (hs _ hp).bundle_preimage


@[deprecated (since := "2024-12-02")]
alias UniqueMDiffOn.smooth_bundle_preimage := UniqueMDiffOn.bundle_preimage

/- TODO: move me to `Mathlib.Geometry.Manifold.VectorBundle.MDifferentiable` once #19636 is in. -/

theorem Trivialization.mdifferentiable [SmoothVectorBundle F Z I]
    (e : Trivialization F (π F Z)) [MemTrivializationAtlas e] :
    e.MDifferentiable (I.prod 𝓘(𝕜, F)) (I.prod 𝓘(𝕜, F)) :=
  ⟨e.contMDiffOn.mdifferentiableOn le_top, e.contMDiffOn_symm.mdifferentiableOn le_top⟩


