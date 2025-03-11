theorem uniqueMDiffWithinAt_univ : UniqueMDiffWithinAt I univ x := by
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
    ⊢ UniqueMDiffWithinAt I Set.univ x
  -/
  unfold UniqueMDiffWithinAt
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
    ⊢ UniqueDiffWithinAt 𝕜 (Inter.inter (Set.preimage (↑(extChartAt I x).symm) Set …
  -/
  simp only [preimage_univ, univ_inter]
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
    ⊢ UniqueDiffWithinAt 𝕜 (Set.range ↑I) (↑(extChartAt I x) x)
  -/
  exact I.uniqueDiffOn _ (mem_range_self _)
  /-
    🎉 no goals
  -/


theorem uniqueMDiffWithinAt_iff_inter_range {s : Set M} {x : M} :
    UniqueMDiffWithinAt I s x ↔
      UniqueDiffWithinAt 𝕜 ((extChartAt I x).symm ⁻¹' s ∩ range I)
        ((extChartAt I x) x) := Iff.rfl


theorem uniqueMDiffWithinAt_iff {s : Set M} {x : M} :
    UniqueMDiffWithinAt I s x ↔
      UniqueDiffWithinAt 𝕜 ((extChartAt I x).symm ⁻¹' s ∩ (extChartAt I x).target)
        ((extChartAt I x) x) := by
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
    ⊢ Iff (UniqueMDiffWithinAt I s x) (UniqueDiffWithinAt 𝕜 (Inter.inter (Set.prei …
  -/
  apply uniqueDiffWithinAt_congr
  /-
    case st
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
    ⊢ Eq (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.preimage (↑(extChartA …
  -/
  rw [nhdsWithin_inter, nhdsWithin_inter, nhdsWithin_extChartAt_target_eq]
  /-
    🎉 no goals
  -/


nonrec theorem UniqueMDiffWithinAt.mono_nhds {s t : Set M} {x : M} (hs : UniqueMDiffWithinAt I s x)
    (ht : 𝓝[s] x ≤ 𝓝[t] x) : UniqueMDiffWithinAt I t x :=
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
                       s t : Set M
                       x : M
                       hs : UniqueMDiffWithinAt I s x
                       ht : LE.le (nhdsWithin x s) (nhdsWithin x t)
                       ⊢ LE.le (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.preimage (↑(extCha …
                     -/
  hs.mono_nhds <| by simpa only [← map_extChartAt_nhdsWithin] using Filter.map_mono ht
                     /-
                       🎉 no goals
                     -/


theorem UniqueMDiffWithinAt.mono_of_mem_nhdsWithin {s t : Set M} {x : M}
    (hs : UniqueMDiffWithinAt I s x) (ht : t ∈ 𝓝[s] x) : UniqueMDiffWithinAt I t x :=
  hs.mono_nhds (nhdsWithin_le_iff.2 ht)


@[deprecated (since := "2024-10-31")]
alias UniqueMDiffWithinAt.mono_of_mem := UniqueMDiffWithinAt.mono_of_mem_nhdsWithin


theorem UniqueMDiffWithinAt.mono (h : UniqueMDiffWithinAt I s x) (st : s ⊆ t) :
    UniqueMDiffWithinAt I t x :=
  UniqueDiffWithinAt.mono h <| inter_subset_inter (preimage_mono st) (Subset.refl _)


theorem UniqueMDiffWithinAt.inter' (hs : UniqueMDiffWithinAt I s x) (ht : t ∈ 𝓝[s] x) :
    UniqueMDiffWithinAt I (s ∩ t) x :=
  hs.mono_of_mem_nhdsWithin (Filter.inter_mem self_mem_nhdsWithin ht)


theorem UniqueMDiffWithinAt.inter (hs : UniqueMDiffWithinAt I s x) (ht : t ∈ 𝓝 x) :
    UniqueMDiffWithinAt I (s ∩ t) x :=
  hs.inter' (nhdsWithin_le_nhds ht)


theorem IsOpen.uniqueMDiffWithinAt (hs : IsOpen s) (xs : x ∈ s) : UniqueMDiffWithinAt I s x :=
  (uniqueMDiffWithinAt_univ I).mono_of_mem_nhdsWithin <| nhdsWithin_le_nhds <| hs.mem_nhds xs


theorem UniqueMDiffOn.inter (hs : UniqueMDiffOn I s) (ht : IsOpen t) : UniqueMDiffOn I (s ∩ t) :=
  fun _x hx => UniqueMDiffWithinAt.inter (hs _ hx.1) (ht.mem_nhds hx.2)


theorem IsOpen.uniqueMDiffOn (hs : IsOpen s) : UniqueMDiffOn I s :=
  fun _x hx => hs.uniqueMDiffWithinAt hx


theorem uniqueMDiffOn_univ : UniqueMDiffOn I (univ : Set M) :=
  isOpen_univ.uniqueMDiffOn


nonrec theorem UniqueMDiffWithinAt.prod {x : M} {y : M'} {s t} (hs : UniqueMDiffWithinAt I s x)
    (ht : UniqueMDiffWithinAt I' t y) : UniqueMDiffWithinAt (I.prod I') (s ×ˢ t) (x, y) := by
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
    x : M
    y : M'
    s : Set M
    t : Set M'
    hs : UniqueMDiffWithinAt I s x
    ht : UniqueMDiffWithinAt I' t y
    ⊢ UniqueMDiffWithinAt (I.prod I') (SProd.sprod s t) { fst := x, snd := y }
  -/
  refine (hs.prod ht).mono ?_
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
    x : M
    y : M'
    s : Set M
    t : Set M'
    hs : UniqueMDiffWithinAt I s x
    ht : UniqueMDiffWithinAt I' t y
    ⊢ HasSubset.Subset (SProd.sprod (Inter.inter (Set.preimage (↑(extChartAt I x). …
  -/
  rw [ModelWithCorners.range_prod, ← prod_inter_prod]
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
    x : M
    y : M'
    s : Set M
    t : Set M'
    hs : UniqueMDiffWithinAt I s x
    ht : UniqueMDiffWithinAt I' t y
    ⊢ HasSubset.Subset (Inter.inter (SProd.sprod (Set.preimage (↑(extChartAt I x). …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem UniqueMDiffOn.prod {s : Set M} {t : Set M'} (hs : UniqueMDiffOn I s)
    (ht : UniqueMDiffOn I' t) : UniqueMDiffOn (I.prod I') (s ×ˢ t) := fun x h ↦
  (hs x.1 h.1).prod (ht x.2 h.2)


theorem MDifferentiableWithinAt.mono (hst : s ⊆ t) (h : MDifferentiableWithinAt I I' f t x) :
    MDifferentiableWithinAt I I' f s x :=
  ⟨ContinuousWithinAt.mono h.1 hst, DifferentiableWithinAt.mono
    h.differentiableWithinAt_writtenInExtChartAt
    (inter_subset_inter_left _ (preimage_mono hst))⟩


theorem mdifferentiableWithinAt_univ :
    MDifferentiableWithinAt I I' f univ x ↔ MDifferentiableAt I I' f x := by
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
    f : M → M'
    x : M
    ⊢ Iff (MDifferentiableWithinAt I I' f Set.univ x) (MDifferentiableAt I I' f x)
  -/
  simp_rw [MDifferentiableWithinAt, MDifferentiableAt, ChartedSpace.LiftPropAt]
  /-
    🎉 no goals
  -/


theorem mdifferentiableWithinAt_inter (ht : t ∈ 𝓝 x) :
    MDifferentiableWithinAt I I' f (s ∩ t) x ↔ MDifferentiableWithinAt I I' f s x := by
  rw [MDifferentiableWithinAt, MDifferentiableWithinAt,
    differentiableWithinAt_localInvariantProp.liftPropWithinAt_inter ht]


theorem mdifferentiableWithinAt_inter' (ht : t ∈ 𝓝[s] x) :
    MDifferentiableWithinAt I I' f (s ∩ t) x ↔ MDifferentiableWithinAt I I' f s x := by
  rw [MDifferentiableWithinAt, MDifferentiableWithinAt,
    differentiableWithinAt_localInvariantProp.liftPropWithinAt_inter' ht]


theorem MDifferentiableAt.mdifferentiableWithinAt (h : MDifferentiableAt I I' f x) :
    MDifferentiableWithinAt I I' f s x :=
  MDifferentiableWithinAt.mono (subset_univ _) (mdifferentiableWithinAt_univ.2 h)


theorem MDifferentiableWithinAt.mdifferentiableAt (h : MDifferentiableWithinAt I I' f s x)
    (hs : s ∈ 𝓝 x) : MDifferentiableAt I I' f x := by
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
    f : M → M'
    x : M
    s : Set M
    h : MDifferentiableWithinAt I I' f s x
    hs : Membership.mem (nhds x) s
    ⊢ MDifferentiableAt I I' f x
  -/
  have : s = univ ∩ s := by rw [univ_inter]
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
    f : M → M'
    x : M
    s : Set M
    h : MDifferentiableWithinAt I I' f s x
    hs : Membership.mem (nhds x) s
    this : Eq s (Inter.inter Set.univ s)
    ⊢ MDifferentiableAt I I' f x
  -/
  rwa [this, mdifferentiableWithinAt_inter hs, mdifferentiableWithinAt_univ] at h
  /-
    🎉 no goals
  -/


theorem MDifferentiableOn.mono (h : MDifferentiableOn I I' f t) (st : s ⊆ t) :
    MDifferentiableOn I I' f s := fun x hx => (h x (st hx)).mono st


theorem mdifferentiableOn_univ : MDifferentiableOn I I' f univ ↔ MDifferentiable I I' f := by
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
    f : M → M'
    ⊢ Iff (MDifferentiableOn I I' f Set.univ) (MDifferentiable I I' f)
  -/
  simp only [MDifferentiableOn, mdifferentiableWithinAt_univ, mfld_simps]; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem MDifferentiableOn.mdifferentiableAt (h : MDifferentiableOn I I' f s) (hx : s ∈ 𝓝 x) :
    MDifferentiableAt I I' f x :=
  (h x (mem_of_mem_nhds hx)).mdifferentiableAt hx


theorem MDifferentiable.mdifferentiableOn (h : MDifferentiable I I' f) :
    MDifferentiableOn I I' f s :=
  (mdifferentiableOn_univ.2 h).mono (subset_univ _)


theorem mdifferentiableOn_of_locally_mdifferentiableOn
    (h : ∀ x ∈ s, ∃ u, IsOpen u ∧ x ∈ u ∧ MDifferentiableOn I I' f (s ∩ u)) :
    MDifferentiableOn I I' f s := by
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
    f : M → M'
    s : Set M
    h : ∀ (x : M), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    ⊢ MDifferentiableOn I I' f s
  -/
  intro x xs
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
    f : M → M'
    s : Set M
    h : ∀ (x : M), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : M
    xs : Membership.mem s x
    ⊢ MDifferentiableWithinAt I I' f s x
  -/
  rcases h x xs with ⟨t, t_open, xt, ht⟩
  /-
    case intro.intro.intro
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
    f : M → M'
    s : Set M
    h : ∀ (x : M), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : M
    xs : Membership.mem s x
    t : Set M
    t_open : IsOpen t
    xt : Membership.mem t x
    ht : MDifferentiableOn I I' f (Inter.inter s t)
    ⊢ MDifferentiableWithinAt I I' f s x
  -/
  exact (mdifferentiableWithinAt_inter (t_open.mem_nhds xt)).1 (ht x ⟨xs, xt⟩)
  /-
    🎉 no goals
  -/


theorem MDifferentiable.mdifferentiableAt (hf : MDifferentiable I I' f) :
    MDifferentiableAt I I' f x :=
  hf x


theorem mdifferentiableWithinAt_iff_target_inter {f : M → M'} {s : Set M} {x : M} :
    MDifferentiableWithinAt I I' f s x ↔
      ContinuousWithinAt f s x ∧
        DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I' x f)
          ((extChartAt I x).target ∩ (extChartAt I x).symm ⁻¹' s) ((extChartAt I x) x) := by
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
    f : M → M'
    s : Set M
    x : M
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (And (ContinuousWithinAt f s x) (Di …
  -/
  rw [mdifferentiableWithinAt_iff']
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
    f : M → M'
    s : Set M
    x : M
    ⊢ Iff (And (ContinuousWithinAt f s x) (DifferentiableWithinAt 𝕜 (writtenInExtC …
  -/
  refine and_congr Iff.rfl (exists_congr fun f' => ?_)
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
    f : M → M'
    s : Set M
    x : M
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    ⊢ Iff (HasFDerivWithinAt (writtenInExtChartAt I I' x f) f' (Inter.inter (Set.p …
  -/
  rw [inter_comm]
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
    f : M → M'
    s : Set M
    x : M
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    ⊢ Iff (HasFDerivWithinAt (writtenInExtChartAt I I' x f) f' (Inter.inter (Set.r …
  -/
  simp only [HasFDerivWithinAt, nhdsWithin_inter, nhdsWithin_extChartAt_target_eq]
  /-
    🎉 no goals
  -/


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in the corresponding extended chart. -/
theorem mdifferentiableWithinAt_iff :
    MDifferentiableWithinAt I I' f s x ↔
      ContinuousWithinAt f s x ∧
        DifferentiableWithinAt 𝕜 (extChartAt I' (f x) ∘ f ∘ (extChartAt I x).symm)
          ((extChartAt I x).symm ⁻¹' s ∩ range I) (extChartAt I x x) := by
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
    f : M → M'
    x : M
    s : Set M
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (And (ContinuousWithinAt f s x) (Di …
  -/
  simp_rw [MDifferentiableWithinAt, ChartedSpace.liftPropWithinAt_iff']; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in the corresponding extended chart. This form states smoothness of `f`
written in such a way that the set is restricted to lie within the domain/codomain of the
corresponding charts.
Even though this expression is more complicated than the one in `mdifferentiableWithinAt_iff`, it is
a smaller set, but their germs at `extChartAt I x x` are equal. It is sometimes useful to rewrite
using this in the goal.
-/
theorem mdifferentiableWithinAt_iff_target_inter' :
    MDifferentiableWithinAt I I' f s x ↔
      ContinuousWithinAt f s x ∧
        DifferentiableWithinAt 𝕜 (extChartAt I' (f x) ∘ f ∘ (extChartAt I x).symm)
          ((extChartAt I x).target ∩
            (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' (f x)).source))
          (extChartAt I x x) := by
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
    f : M → M'
    x : M
    s : Set M
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (And (ContinuousWithinAt f s x) (Di …
  -/
  simp only [MDifferentiableWithinAt, liftPropWithinAt_iff']
  exact and_congr_right fun hc => differentiableWithinAt_congr_nhds <|
    hc.nhdsWithin_extChartAt_symm_preimage_inter_range


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in the corresponding extended chart in the target. -/
theorem mdifferentiableWithinAt_iff_target :
    MDifferentiableWithinAt I I' f s x ↔
      ContinuousWithinAt f s x ∧
      MDifferentiableWithinAt I 𝓘(𝕜, E') (extChartAt I' (f x) ∘ f) s x := by
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
    f : M → M'
    x : M
    s : Set M
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (And (ContinuousWithinAt f s x) (MD …
  -/
  simp_rw [MDifferentiableWithinAt, liftPropWithinAt_iff', ← and_assoc]
  have cont :
    ContinuousWithinAt f s x ∧ ContinuousWithinAt (extChartAt I' (f x) ∘ f) s x ↔
        ContinuousWithinAt f s x :=
      and_iff_left_of_imp <| (continuousAt_extChartAt _).comp_continuousWithinAt
  simp_rw [cont, DifferentiableWithinAtProp, extChartAt, PartialHomeomorph.extend,
    PartialEquiv.coe_trans,
    ModelWithCorners.toPartialEquiv_coe, PartialHomeomorph.coe_coe, modelWithCornersSelf_coe,
    chartAt_self_eq, PartialHomeomorph.refl_apply]
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
    f : M → M'
    x : M
    s : Set M
    cont : Iff (And (ContinuousWithinAt f s x) (ContinuousWithinAt (Function.comp  …
    ⊢ Iff (And (ContinuousWithinAt f s x) (DifferentiableWithinAt 𝕜 (Function.comp …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mdifferentiableAt_iff_target {x : M} :
    MDifferentiableAt I I' f x ↔
      ContinuousAt f x ∧ MDifferentiableAt I 𝓘(𝕜, E') (extChartAt I' (f x) ∘ f) x := by
  rw [← mdifferentiableWithinAt_univ, ← mdifferentiableWithinAt_univ,
    mdifferentiableWithinAt_iff_target, continuousWithinAt_univ]


theorem mdifferentiableWithinAt_iff_source_of_mem_maximalAtlas
    [SmoothManifoldWithCorners I M] (he : e ∈ maximalAtlas I M) (hx : x ∈ e.source) :
    MDifferentiableWithinAt I I' f s x ↔
      MDifferentiableWithinAt 𝓘(𝕜, E) I' (f ∘ (e.extend I).symm) ((e.extend I).symm ⁻¹' s ∩ range I)
        (e.extend I x) := by
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    e : PartialHomeomorph M H
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hx : Membership.mem e.source x
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (MDifferentiableWithinAt (modelWith …
  -/
  have h2x := hx; rw [← e.extend_source (I := I)] at h2x
  simp_rw [MDifferentiableWithinAt,
    differentiableWithinAt_localInvariantProp.liftPropWithinAt_indep_chart_source he hx,
    StructureGroupoid.liftPropWithinAt_self_source,
    e.extend_symm_continuousWithinAt_comp_right_iff, differentiableWithinAtProp_self_source,
    DifferentiableWithinAtProp, Function.comp, e.left_inv hx, (e.extend I).left_inv h2x]
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    e : PartialHomeomorph M H
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hx : Membership.mem e.source x
    h2x : Membership.mem (e.extend I).source x
    ⊢ Iff (And (ContinuousWithinAt (Function.comp f ↑e.symm) (Set.preimage (↑e.sym …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mdifferentiableWithinAt_iff_source_of_mem_source
    [SmoothManifoldWithCorners I M] {x' : M} (hx' : x' ∈ (chartAt H x).source) :
    MDifferentiableWithinAt I I' f s x' ↔
      MDifferentiableWithinAt 𝓘(𝕜, E) I' (f ∘ (extChartAt I x).symm)
        ((extChartAt I x).symm ⁻¹' s ∩ range I) (extChartAt I x x') :=
  mdifferentiableWithinAt_iff_source_of_mem_maximalAtlas (chart_mem_maximalAtlas x) hx'


theorem mdifferentiableAt_iff_source_of_mem_source
    [SmoothManifoldWithCorners I M] {x' : M} (hx' : x' ∈ (chartAt H x).source) :
    MDifferentiableAt I I' f x' ↔
      MDifferentiableWithinAt 𝓘(𝕜, E) I' (f ∘ (extChartAt I x).symm) (range I)
        (extChartAt I x x') := by
  simp_rw [← mdifferentiableWithinAt_univ, mdifferentiableWithinAt_iff_source_of_mem_source hx',
    preimage_univ, univ_inter]


theorem mdifferentiableWithinAt_iff_target_of_mem_source
    [SmoothManifoldWithCorners I' M'] {x : M} {y : M'} (hy : f x ∈ (chartAt H' y).source) :
    MDifferentiableWithinAt I I' f s x ↔
      ContinuousWithinAt f s x ∧ MDifferentiableWithinAt I 𝓘(𝕜, E') (extChartAt I' y ∘ f) s x := by
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (And (ContinuousWithinAt f s x) (MD …
  -/
  simp_rw [MDifferentiableWithinAt]
  rw [differentiableWithinAt_localInvariantProp.liftPropWithinAt_indep_chart_target
      (chart_mem_maximalAtlas y) hy,
    and_congr_right]
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    ⊢ ContinuousWithinAt f s x → Iff (ChartedSpace.LiftPropWithinAt (Differentiabl …
  -/
  intro hf
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (ChartedSpace.LiftPropWithinAt (DifferentiableWithinAtProp I I') (Functi …
  -/
  simp_rw [StructureGroupoid.liftPropWithinAt_self_target]
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (And (ContinuousWithinAt (Function.comp (↑(chartAt H' y)) f) s x) (Diffe …
  -/
  simp_rw [((chartAt H' y).continuousAt hy).comp_continuousWithinAt hf]
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (And True (DifferentiableWithinAtProp I I' (Function.comp (Function.comp …
  -/
  rw [← extChartAt_source I'] at hy
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (extChartAt I' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (And True (DifferentiableWithinAtProp I I' (Function.comp (Function.comp …
  -/
  simp_rw [(continuousAt_extChartAt' hy).comp_continuousWithinAt hf]
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
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (extChartAt I' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (And True (DifferentiableWithinAtProp I I' (Function.comp (Function.comp …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mdifferentiableAt_iff_target_of_mem_source
    [SmoothManifoldWithCorners I' M'] {x : M} {y : M'} (hy : f x ∈ (chartAt H' y).source) :
    MDifferentiableAt I I' f x ↔
      ContinuousAt f x ∧ MDifferentiableAt I 𝓘(𝕜, E') (extChartAt I' y ∘ f) x := by
  rw [← mdifferentiableWithinAt_univ, mdifferentiableWithinAt_iff_target_of_mem_source hy,
    continuousWithinAt_univ, ← mdifferentiableWithinAt_univ]


theorem mdifferentiableWithinAt_iff_of_mem_maximalAtlas {x : M} (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I' M') (hx : x ∈ e.source) (hy : f x ∈ e'.source) :
    MDifferentiableWithinAt I I' f s x ↔
      ContinuousWithinAt f s x ∧
        DifferentiableWithinAt 𝕜 (e'.extend I' ∘ f ∘ (e.extend I).symm)
          ((e.extend I).symm ⁻¹' s ∩ range I) (e.extend I x) :=
  differentiableWithinAt_localInvariantProp.liftPropWithinAt_indep_chart he hx he' hy


/-- An alternative formulation of `mdifferentiableWithinAt_iff_of_mem_maximalAtlas`
  if the set if `s` lies in `e.source`. -/
theorem mdifferentiableWithinAt_iff_image {x : M} (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I' M') (hs : s ⊆ e.source) (hx : x ∈ e.source) (hy : f x ∈ e'.source) :
    MDifferentiableWithinAt I I' f s x ↔
      ContinuousWithinAt f s x ∧
        DifferentiableWithinAt 𝕜 (e'.extend I' ∘ f ∘ (e.extend I).symm) (e.extend I '' s)
          (e.extend I x) := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    hx : Membership.mem e.source x
    hy : Membership.mem e'.source (f x)
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (And (ContinuousWithinAt f s x) (Di …
  -/
  rw [mdifferentiableWithinAt_iff_of_mem_maximalAtlas he he' hx hy, and_congr_right_iff]
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    hx : Membership.mem e.source x
    hy : Membership.mem e'.source (f x)
    ⊢ ContinuousWithinAt f s x → Iff (DifferentiableWithinAt 𝕜 (Function.comp (↑(e …
  -/
  refine fun _ => differentiableWithinAt_congr_nhds ?_
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    hx : Membership.mem e.source x
    hy : Membership.mem e'.source (f x)
    x✝ : ContinuousWithinAt f s x
    ⊢ Eq (nhdsWithin (↑(e.extend I) x) (Inter.inter (Set.preimage (↑(e.extend I).s …
  -/
  simp_rw [nhdsWithin_eq_iff_eventuallyEq, e.extend_symm_preimage_inter_range_eventuallyEq hs hx]
  /-
    🎉 no goals
  -/


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in any chart containing that point. -/
theorem mdifferentiableWithinAt_iff_of_mem_source {x' : M} {y : M'} (hx : x' ∈ (chartAt H x).source)
    (hy : f x' ∈ (chartAt H' y).source) :
    MDifferentiableWithinAt I I' f s x' ↔
      ContinuousWithinAt f s x' ∧
        DifferentiableWithinAt 𝕜 (extChartAt I' y ∘ f ∘ (extChartAt I x).symm)
          ((extChartAt I x).symm ⁻¹' s ∩ range I) (extChartAt I x x') :=
  mdifferentiableWithinAt_iff_of_mem_maximalAtlas (chart_mem_maximalAtlas x)
    (chart_mem_maximalAtlas y) hx hy


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in any chart containing that point. Version requiring differentiability
in the target instead of `range I`. -/
theorem mdifferentiableWithinAt_iff_of_mem_source' {x' : M} {y : M'}
    (hx : x' ∈ (chartAt H x).source) (hy : f x' ∈ (chartAt H' y).source) :
    MDifferentiableWithinAt I I' f s x' ↔
      ContinuousWithinAt f s x' ∧
        DifferentiableWithinAt 𝕜 (extChartAt I' y ∘ f ∘ (extChartAt I x).symm)
          ((extChartAt I x).target ∩ (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' y).source))
          (extChartAt I x x') := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (chartAt H x).source x'
    hy : Membership.mem (chartAt H' y).source (f x')
    ⊢ Iff (MDifferentiableWithinAt I I' f s x') (And (ContinuousWithinAt f s x') ( …
  -/
  refine (mdifferentiableWithinAt_iff_of_mem_source hx hy).trans ?_
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (chartAt H x).source x'
    hy : Membership.mem (chartAt H' y).source (f x')
    ⊢ Iff (And (ContinuousWithinAt f s x') (DifferentiableWithinAt 𝕜 (Function.com …
  -/
  rw [← extChartAt_source I] at hx
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (extChartAt I x).source x'
    hy : Membership.mem (chartAt H' y).source (f x')
    ⊢ Iff (And (ContinuousWithinAt f s x') (DifferentiableWithinAt 𝕜 (Function.com …
  -/
  rw [← extChartAt_source I'] at hy
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (extChartAt I x).source x'
    hy : Membership.mem (extChartAt I' y).source (f x')
    ⊢ Iff (And (ContinuousWithinAt f s x') (DifferentiableWithinAt 𝕜 (Function.com …
  -/
  rw [and_congr_right_iff]
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (extChartAt I x).source x'
    hy : Membership.mem (extChartAt I' y).source (f x')
    ⊢ ContinuousWithinAt f s x' → Iff (DifferentiableWithinAt 𝕜 (Function.comp (↑( …
  -/
  set e := extChartAt I x; set e' := extChartAt I' (f x)
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hy : Membership.mem (extChartAt I' y).source (f x')
    e : PartialEquiv M E := extChartAt I x
    hx : Membership.mem e.source x'
    e' : PartialEquiv M' E' := extChartAt I' (f x)
    ⊢ ContinuousWithinAt f s x' → Iff (DifferentiableWithinAt 𝕜 (Function.comp (↑( …
  -/
  refine fun hc => differentiableWithinAt_congr_nhds ?_
  rw [← e.image_source_inter_eq', ← map_extChartAt_nhdsWithin_eq_image' hx,
    ← map_extChartAt_nhdsWithin' hx, inter_comm, nhdsWithin_inter_of_mem]
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hy : Membership.mem (extChartAt I' y).source (f x')
    e : PartialEquiv M E := extChartAt I x
    hx : Membership.mem e.source x'
    e' : PartialEquiv M' E' := extChartAt I' (f x)
    hc : ContinuousWithinAt f s x'
    ⊢ Membership.mem (nhdsWithin x' s) (Set.preimage f (extChartAt I' y).source)
  -/
  exact hc (extChartAt_source_mem_nhds' hy)
  /-
    🎉 no goals
  -/


theorem mdifferentiableAt_iff_of_mem_source {x' : M} {y : M'} (hx : x' ∈ (chartAt H x).source)
    (hy : f x' ∈ (chartAt H' y).source) :
    MDifferentiableAt I I' f x' ↔
      ContinuousAt f x' ∧
        DifferentiableWithinAt 𝕜 (extChartAt I' y ∘ f ∘ (extChartAt I x).symm) (range I)
          (extChartAt I x x') :=
  (mdifferentiableWithinAt_iff_of_mem_source hx hy).trans <| by
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      x : M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      x' : M
      y : M'
      hx : Membership.mem (chartAt H x).source x'
      hy : Membership.mem (chartAt H' y).source (f x')
      ⊢ Iff (And (ContinuousWithinAt f Set.univ x') (DifferentiableWithinAt 𝕜 (Funct …
    -/
    rw [continuousWithinAt_univ, preimage_univ, univ_inter]
    /-
      🎉 no goals
    -/


theorem mdifferentiableOn_iff_of_mem_maximalAtlas (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I' M') (hs : s ⊆ e.source) (h2s : MapsTo f s e'.source) :
    MDifferentiableOn I I' f s ↔
      ContinuousOn f s ∧
        DifferentiableOn 𝕜 (e'.extend I' ∘ f ∘ (e.extend I).symm) (e.extend I '' s) := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    h2s : Set.MapsTo f s e'.source
    ⊢ Iff (MDifferentiableOn I I' f s) (And (ContinuousOn f s) (DifferentiableOn 𝕜 …
  -/
  simp_rw [ContinuousOn, DifferentiableOn, Set.forall_mem_image, ← forall_and, MDifferentiableOn]
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    h2s : Set.MapsTo f s e'.source
    ⊢ Iff (∀ (x : M), Membership.mem s x → MDifferentiableWithinAt I I' f s x) (∀  …
  -/
  exact forall₂_congr fun x hx => mdifferentiableWithinAt_iff_image he he' hs (hs hx) (h2s hx)
  /-
    🎉 no goals
  -/


/-- Differentiability on a set is equivalent to differentiability in the extended charts. -/
theorem mdifferentiableOn_iff_of_mem_maximalAtlas' (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I' M') (hs : s ⊆ e.source) (h2s : MapsTo f s e'.source) :
    MDifferentiableOn I I' f s ↔
      DifferentiableOn 𝕜 (e'.extend I' ∘ f ∘ (e.extend I).symm) (e.extend I '' s) :=
  (mdifferentiableOn_iff_of_mem_maximalAtlas he he' hs h2s).trans <| and_iff_right_of_imp fun h ↦
    (e.continuousOn_writtenInExtend_iff hs h2s).1 h.continuousOn


/-- If the set where you want `f` to be smooth lies entirely in a single chart, and `f` maps it
  into a single chart, the smoothness of `f` on that set can be expressed by purely looking in
  these charts.
  Note: this lemma uses `extChartAt I x '' s` instead of `(extChartAt I x).symm ⁻¹' s` to ensure
  that this set lies in `(extChartAt I x).target`. -/
theorem mdifferentiableOn_iff_of_subset_source {x : M} {y : M'} (hs : s ⊆ (chartAt H x).source)
    (h2s : MapsTo f s (chartAt H' y).source) :
    MDifferentiableOn I I' f s ↔
      ContinuousOn f s ∧
        DifferentiableOn 𝕜 (extChartAt I' y ∘ f ∘ (extChartAt I x).symm) (extChartAt I x '' s) :=
  mdifferentiableOn_iff_of_mem_maximalAtlas (chart_mem_maximalAtlas x)
    (chart_mem_maximalAtlas y) hs h2s


/-- If the set where you want `f` to be smooth lies entirely in a single chart, and `f` maps it
  into a single chart, the smoothness of `f` on that set can be expressed by purely looking in
  these charts.
  Note: this lemma uses `extChartAt I x '' s` instead of `(extChartAt I x).symm ⁻¹' s` to ensure
  that this set lies in `(extChartAt I x).target`. -/
theorem mdifferentiableOn_iff_of_subset_source' {x : M} {y : M'} (hs : s ⊆ (extChartAt I x).source)
    (h2s : MapsTo f s (extChartAt I' y).source) :
    MDifferentiableOn I I' f s ↔
        DifferentiableOn 𝕜 (extChartAt I' y ∘ f ∘ (extChartAt I x).symm) (extChartAt I x '' s) := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hs : HasSubset.Subset s (extChartAt I x).source
    h2s : Set.MapsTo f s (extChartAt I' y).source
    ⊢ Iff (MDifferentiableOn I I' f s) (DifferentiableOn 𝕜 (Function.comp (↑(extCh …
  -/
  rw [extChartAt_source] at hs h2s
  exact mdifferentiableOn_iff_of_mem_maximalAtlas' (chart_mem_maximalAtlas x)
    (chart_mem_maximalAtlas y) hs h2s


/-- One can reformulate smoothness on a set as continuity on this set, and smoothness in any
extended chart. -/
theorem mdifferentiableOn_iff :
    MDifferentiableOn I I' f s ↔
      ContinuousOn f s ∧
        ∀ (x : M) (y : M'),
          DifferentiableOn 𝕜 (extChartAt I' y ∘ f ∘ (extChartAt I x).symm)
            ((extChartAt I x).target ∩
              (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' y).source)) := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (MDifferentiableOn I I' f s) (And (ContinuousOn f s) (∀ (x : M) (y : M') …
  -/
  constructor
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      ⊢ MDifferentiableOn I I' f s → And (ContinuousOn f s) (∀ (x : M) (y : M'), Dif …
    -/
  · intro h
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : MDifferentiableOn I I' f s
      ⊢ And (ContinuousOn f s) (∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.com …
    -/
    refine ⟨fun x hx => (h x hx).1, fun x y z hz => ?_⟩
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : MDifferentiableOn I I' f s
      x : M
      y : M'
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑(extChartAt I' y)) (Function.comp  …
    -/
    simp only [mfld_simps] at hz
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : MDifferentiableOn I I' f s
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑(extChartAt I' y)) (Function.comp  …
    -/
    let w := (extChartAt I x).symm z
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : MDifferentiableOn I I' f s
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑(extChartAt I' y)) (Function.comp  …
    -/
    have : w ∈ s := by simp only [w, hz, mfld_simps]
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : MDifferentiableOn I I' f s
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      this : Membership.mem s w
      ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑(extChartAt I' y)) (Function.comp  …
    -/
    specialize h w this
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      this : Membership.mem s w
      h : MDifferentiableWithinAt I I' f s w
      ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑(extChartAt I' y)) (Function.comp  …
    -/
    have w1 : w ∈ (chartAt H x).source := by simp only [w, hz, mfld_simps]
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      this : Membership.mem s w
      h : MDifferentiableWithinAt I I' f s w
      w1 : Membership.mem (chartAt H x).source w
      ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑(extChartAt I' y)) (Function.comp  …
    -/
    have w2 : f w ∈ (chartAt H' y).source := by simp only [w, hz, mfld_simps]
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      this : Membership.mem s w
      h : MDifferentiableWithinAt I I' f s w
      w1 : Membership.mem (chartAt H x).source w
      w2 : Membership.mem (chartAt H' y).source (f w)
      ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑(extChartAt I' y)) (Function.comp  …
    -/
    convert ((mdifferentiableWithinAt_iff_of_mem_source w1 w2).mp h).2.mono _
      /-
        case h.e'_13
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
        E' : Type u_5
        inst✝⁶ : NormedAddCommGroup E'
        inst✝⁵ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝³ : TopologicalSpace M'
        inst✝² : ChartedSpace H' M'
        f : M → M'
        s : Set M
        inst✝¹ : SmoothManifoldWithCorners I M
        inst✝ : SmoothManifoldWithCorners I' M'
        x : M
        y : M'
        z : E
        hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
        w : M := ↑(extChartAt I x).symm z
        this : Membership.mem s w
        h : MDifferentiableWithinAt I I' f s w
        w1 : Membership.mem (chartAt H x).source w
        w2 : Membership.mem (chartAt H' y).source (f w)
        ⊢ Eq z (↑(extChartAt I x) w)
      -/
    · simp only [w, hz, mfld_simps]
      /-
        🎉 no goals
      -/
      /-
        case mp.convert_2
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
        E' : Type u_5
        inst✝⁶ : NormedAddCommGroup E'
        inst✝⁵ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝³ : TopologicalSpace M'
        inst✝² : ChartedSpace H' M'
        f : M → M'
        s : Set M
        inst✝¹ : SmoothManifoldWithCorners I M
        inst✝ : SmoothManifoldWithCorners I' M'
        x : M
        y : M'
        z : E
        hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
        w : M := ↑(extChartAt I x).symm z
        this : Membership.mem s w
        h : MDifferentiableWithinAt I I' f s w
        w1 : Membership.mem (chartAt H x).source w
        w2 : Membership.mem (chartAt H' y).source (f w)
        ⊢ HasSubset.Subset (Inter.inter (extChartAt I x).target (Set.preimage (↑(extCh …
      -/
    · mfld_set_tac
      /-
        🎉 no goals
      -/
    /-
      case mpr
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      ⊢ And (ContinuousOn f s) (∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.com …
    -/
  · rintro ⟨hcont, hdiff⟩ x hx
    /-
      case mpr.intro
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑(extChartAt I' …
      x : M
      hx : Membership.mem s x
      ⊢ MDifferentiableWithinAt I I' f s x
    -/
    refine differentiableWithinAt_localInvariantProp.liftPropWithinAt_iff.mpr ?_
    /-
      case mpr.intro
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑(extChartAt I' …
      x : M
      hx : Membership.mem s x
      ⊢ And (ContinuousWithinAt f s x) (DifferentiableWithinAtProp I I' (Function.co …
    -/
    refine ⟨hcont x hx, ?_⟩
    /-
      case mpr.intro
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑(extChartAt I' …
      x : M
      hx : Membership.mem s x
      ⊢ DifferentiableWithinAtProp I I' (Function.comp (↑(chartAt H' (f x))) (Functi …
    -/
    dsimp [DifferentiableWithinAtProp]
    /-
      case mpr.intro
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑(extChartAt I' …
      x : M
      hx : Membership.mem s x
      ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp (Function.comp  …
    -/
    convert hdiff x (f x) (extChartAt I x x) (by simp only [hx, mfld_simps]) using 1
    /-
      case h.e'_12
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑(extChartAt I' …
      x : M
      hx : Membership.mem s x
      ⊢ Eq (Inter.inter (Inter.inter (Set.preimage (↑I.symm) (chartAt H x).target) ( …
    -/
    mfld_set_tac
    /-
      🎉 no goals
    -/


/-- One can reformulate smoothness on a set as continuity on this set, and smoothness in any
extended chart in the target. -/
theorem mdifferentiableOn_iff_target :
    MDifferentiableOn I I' f s ↔
      ContinuousOn f s ∧
        ∀ y : M', MDifferentiableOn I 𝓘(𝕜, E') (extChartAt I' y ∘ f)
          (s ∩ f ⁻¹' (extChartAt I' y).source) := by
  simp only [mdifferentiableOn_iff, ModelWithCorners.source_eq, chartAt_self_eq,
    PartialHomeomorph.refl_partialEquiv, PartialEquiv.refl_trans, extChartAt,
    PartialHomeomorph.extend, Set.preimage_univ, Set.inter_univ, and_congr_right_iff]
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ ContinuousOn f s → Iff (∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.com …
  -/
  intro h
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    h : ContinuousOn f s
    ⊢ Iff (∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑((chartAt H' y) …
  -/
  constructor
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      ⊢ (∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑((chartAt H' y).tra …
    -/
  · refine fun h' y => ⟨?_, fun x _ => h' x y⟩
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      h' : ∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑((chartAt H' y).t …
      y : M'
      ⊢ ContinuousOn (Function.comp (↑((chartAt H' y).trans I'.toPartialEquiv)) f) ( …
    -/
    have h'' : ContinuousOn _ univ := (ModelWithCorners.continuous I').continuousOn
    /-
      case mp
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      h' : ∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑((chartAt H' y).t …
      y : M'
      h'' : ContinuousOn (↑I') Set.univ
      ⊢ ContinuousOn (Function.comp (↑((chartAt H' y).trans I'.toPartialEquiv)) f) ( …
    -/
    convert (h''.comp_inter (chartAt H' y).continuousOn_toFun).comp_inter h
    /-
      case h.e'_6.h.e'_4.h.e'_4
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      h' : ∀ (x : M) (y : M'), DifferentiableOn 𝕜 (Function.comp (↑((chartAt H' y).t …
      y : M'
      h'' : ContinuousOn (↑I') Set.univ
      ⊢ Eq ((chartAt H' y).trans I'.toPartialEquiv).source (Inter.inter (chartAt H'  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      ⊢ (∀ (y : M'), And (ContinuousOn (Function.comp (↑((chartAt H' y).trans I'.toP …
    -/
  · exact fun h' x y => (h' y).2 x 0
    /-
      🎉 no goals
    -/


/-- One can reformulate smoothness as continuity and smoothness in any extended chart. -/
theorem mdifferentiable_iff :
    MDifferentiable I I' f ↔
      Continuous f ∧
        ∀ (x : M) (y : M'),
          DifferentiableOn 𝕜 (extChartAt I' y ∘ f ∘ (extChartAt I x).symm)
            ((extChartAt I x).target ∩
              (extChartAt I x).symm ⁻¹' (f ⁻¹' (extChartAt I' y).source)) := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (MDifferentiable I I' f) (And (Continuous f) (∀ (x : M) (y : M'), Differ …
  -/
  simp [← mdifferentiableOn_univ, mdifferentiableOn_iff, continuous_iff_continuousOn_univ]
  /-
    🎉 no goals
  -/


/-- One can reformulate smoothness as continuity and smoothness in any extended chart in the
target. -/
theorem mdifferentiable_iff_target :
    MDifferentiable I I' f ↔
      Continuous f ∧ ∀ y : M',
        MDifferentiableOn I 𝓘(𝕜, E') (extChartAt I' y ∘ f) (f ⁻¹' (extChartAt I' y).source) := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (MDifferentiable I I' f) (And (Continuous f) (∀ (y : M'), MDifferentiabl …
  -/
  rw [← mdifferentiableOn_univ, mdifferentiableOn_iff_target]
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (And (ContinuousOn f Set.univ) (∀ (y : M'), MDifferentiableOn I (modelWi …
  -/
  simp [continuous_iff_continuousOn_univ]
  /-
    🎉 no goals
  -/


theorem ContMDiffWithinAt.mdifferentiableWithinAt (hf : ContMDiffWithinAt I I' n f s x)
    (hn : 1 ≤ n) : MDifferentiableWithinAt I I' f s x := by
  suffices h : MDifferentiableWithinAt I I' f (s ∩ f ⁻¹' (extChartAt I' (f x)).source) x by
    rwa [mdifferentiableWithinAt_inter'] at h
    apply hf.1.preimage_mem_nhdsWithin
    exact extChartAt_source_mem_nhds (f x)
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
    f : M → M'
    x : M
    s : Set M
    n : ENat
    hf : ContMDiffWithinAt I I' n f s x
    hn : LE.le 1 n
    ⊢ MDifferentiableWithinAt I I' f (Inter.inter s (Set.preimage f (extChartAt I' …
  -/
  rw [mdifferentiableWithinAt_iff]
  exact ⟨hf.1.mono inter_subset_left, (hf.2.differentiableWithinAt (mod_cast hn)).mono
    (by mfld_set_tac)⟩


theorem ContMDiffAt.mdifferentiableAt (hf : ContMDiffAt I I' n f x) (hn : 1 ≤ n) :
    MDifferentiableAt I I' f x :=
  mdifferentiableWithinAt_univ.1 <| ContMDiffWithinAt.mdifferentiableWithinAt hf hn


theorem ContMDiff.mdifferentiableAt (hf : ContMDiff I I' n f) (hn : 1 ≤ n) :
    MDifferentiableAt I I' f x :=
  hf.contMDiffAt.mdifferentiableAt hn


theorem ContMDiff.mdifferentiableWithinAt (hf : ContMDiff I I' n f) (hn : 1 ≤ n) :
    MDifferentiableWithinAt I I' f s x :=
  (hf.contMDiffAt.mdifferentiableAt hn).mdifferentiableWithinAt


theorem ContMDiffOn.mdifferentiableOn (hf : ContMDiffOn I I' n f s) (hn : 1 ≤ n) :
    MDifferentiableOn I I' f s := fun x hx => (hf x hx).mdifferentiableWithinAt hn


@[deprecated (since := "2024-11-20")]
alias SmoothWithinAt.mdifferentiableWithinAt := ContMDiffWithinAt.mdifferentiableWithinAt


theorem ContMDiff.mdifferentiable (hf : ContMDiff I I' n f) (hn : 1 ≤ n) : MDifferentiable I I' f :=
  fun x => (hf x).mdifferentiableAt hn


@[deprecated (since := "2024-11-20")]
alias SmoothAt.mdifferentiableAt := ContMDiffAt.mdifferentiableAt


@[deprecated (since := "2024-11-20")]
alias SmoothOn.mdifferentiableOn := ContMDiffOn.mdifferentiableOn


@[deprecated (since := "2024-11-20")]
alias Smooth.mdifferentiable := ContMDiff.mdifferentiable


@[deprecated (since := "2024-11-20")]
alias Smooth.mdifferentiableAt := ContMDiff.mdifferentiableAt


theorem MDifferentiableOn.continuousOn (h : MDifferentiableOn I I' f s) : ContinuousOn f s :=
  fun x hx => (h x hx).continuousWithinAt


theorem MDifferentiable.continuous (h : MDifferentiable I I' f) : Continuous f :=
  continuous_iff_continuousAt.2 fun x => (h x).continuousAt


@[deprecated (since := "2024-11-20")]
alias Smooth.mdifferentiableWithinAt := ContMDiff.mdifferentiableWithinAt


theorem MDifferentiableWithinAt.prod_mk {f : M → M'} {g : M → M''}
    (hf : MDifferentiableWithinAt I I' f s x) (hg : MDifferentiableWithinAt I I'' g s x) :
    MDifferentiableWithinAt I (I'.prod I'') (fun x => (f x, g x)) s x :=
  ⟨hf.1.prod hg.1, hf.2.prod hg.2⟩


theorem MDifferentiableAt.prod_mk {f : M → M'} {g : M → M''} (hf : MDifferentiableAt I I' f x)
    (hg : MDifferentiableAt I I'' g x) :
    MDifferentiableAt I (I'.prod I'') (fun x => (f x, g x)) x :=
  ⟨hf.1.prod hg.1, hf.2.prod hg.2⟩


theorem MDifferentiableWithinAt.prod_mk_space {f : M → E'} {g : M → E''}
    (hf : MDifferentiableWithinAt I 𝓘(𝕜, E') f s x)
    (hg : MDifferentiableWithinAt I 𝓘(𝕜, E'') g s x) :
    MDifferentiableWithinAt I 𝓘(𝕜, E' × E'') (fun x => (f x, g x)) s x :=
  ⟨hf.1.prod hg.1, hf.2.prod hg.2⟩


theorem MDifferentiableAt.prod_mk_space {f : M → E'} {g : M → E''}
    (hf : MDifferentiableAt I 𝓘(𝕜, E') f x) (hg : MDifferentiableAt I 𝓘(𝕜, E'') g x) :
    MDifferentiableAt I 𝓘(𝕜, E' × E'') (fun x => (f x, g x)) x :=
  ⟨hf.1.prod hg.1, hf.2.prod hg.2⟩


theorem MDifferentiableOn.prod_mk {f : M → M'} {g : M → M''} (hf : MDifferentiableOn I I' f s)
    (hg : MDifferentiableOn I I'' g s) :
    MDifferentiableOn I (I'.prod I'') (fun x => (f x, g x)) s := fun x hx =>
  (hf x hx).prod_mk (hg x hx)


theorem MDifferentiable.prod_mk {f : M → M'} {g : M → M''} (hf : MDifferentiable I I' f)
    (hg : MDifferentiable I I'' g) : MDifferentiable I (I'.prod I'') fun x => (f x, g x) := fun x =>
  (hf x).prod_mk (hg x)


theorem MDifferentiableOn.prod_mk_space {f : M → E'} {g : M → E''}
    (hf : MDifferentiableOn I 𝓘(𝕜, E') f s) (hg : MDifferentiableOn I 𝓘(𝕜, E'') g s) :
    MDifferentiableOn I 𝓘(𝕜, E' × E'') (fun x => (f x, g x)) s := fun x hx =>
  (hf x hx).prod_mk_space (hg x hx)


theorem MDifferentiable.prod_mk_space {f : M → E'} {g : M → E''} (hf : MDifferentiable I 𝓘(𝕜, E') f)
    (hg : MDifferentiable I 𝓘(𝕜, E'') g) : MDifferentiable I 𝓘(𝕜, E' × E'') fun x => (f x, g x) :=
  fun x => (hf x).prod_mk_space (hg x)


theorem writtenInExtChartAt_comp (h : ContinuousWithinAt f s x) :
    {y | writtenInExtChartAt I I'' x (g ∘ f) y =
          (writtenInExtChartAt I' I'' (f x) g ∘ writtenInExtChartAt I I' x f) y} ∈
      𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] (extChartAt I x) x := by
  apply
    @Filter.mem_of_superset _ _ (f ∘ (extChartAt I x).symm ⁻¹' (extChartAt I' (f x)).source) _
      (extChartAt_preimage_mem_nhdsWithin
        (h.preimage_mem_nhdsWithin (extChartAt_source_mem_nhds _)))
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    h : ContinuousWithinAt f s x
    ⊢ HasSubset.Subset (Set.preimage (Function.comp f ↑(extChartAt I x).symm) (ext …
  -/
  mfld_set_tac
  /-
    🎉 no goals
  -/


/-- `UniqueMDiffWithinAt` achieves its goal: it implies the uniqueness of the derivative. -/
protected nonrec theorem UniqueMDiffWithinAt.eq (U : UniqueMDiffWithinAt I s x)
    (h : HasMFDerivWithinAt I I' f s x f') (h₁ : HasMFDerivWithinAt I I' f s x f₁') : f' = f₁' := by
  -- Porting note: didn't need `convert` because of finding instances by unification
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
    f : M → M'
    x : M
    s : Set M
    f' f₁' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I …
    U : UniqueMDiffWithinAt I s x
    h : HasMFDerivWithinAt I I' f s x f'
    h₁ : HasMFDerivWithinAt I I' f s x f₁'
    ⊢ Eq f' f₁'
  -/
  convert U.eq h.2 h₁.2
  /-
    🎉 no goals
  -/


protected theorem UniqueMDiffOn.eq (U : UniqueMDiffOn I s) (hx : x ∈ s)
    (h : HasMFDerivWithinAt I I' f s x f') (h₁ : HasMFDerivWithinAt I I' f s x f₁') : f' = f₁' :=
  UniqueMDiffWithinAt.eq (U _ hx) h h₁


theorem mfderivWithin_zero_of_not_mdifferentiableWithinAt
    (h : ¬MDifferentiableWithinAt I I' f s x) : mfderivWithin I I' f s x = 0 := by
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
    f : M → M'
    x : M
    s : Set M
    h : Not (MDifferentiableWithinAt I I' f s x)
    ⊢ Eq (mfderivWithin I I' f s x) 0
  -/
  simp only [mfderivWithin, h, if_neg, not_false_iff]
  /-
    🎉 no goals
  -/


theorem mfderiv_zero_of_not_mdifferentiableAt (h : ¬MDifferentiableAt I I' f x) :
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
                                 f : M → M'
                                 x : M
                                 h : Not (MDifferentiableAt I I' f x)
                                 ⊢ Eq (mfderiv I I' f x) 0
                               -/
    mfderiv I I' f x = 0 := by simp only [mfderiv, h, if_neg, not_false_iff]
                               /-
                                 🎉 no goals
                               -/


theorem HasMFDerivWithinAt.mono (h : HasMFDerivWithinAt I I' f t x f') (hst : s ⊆ t) :
    HasMFDerivWithinAt I I' f s x f' :=
  ⟨ContinuousWithinAt.mono h.1 hst,
    HasFDerivWithinAt.mono h.2 (inter_subset_inter (preimage_mono hst) (Subset.refl _))⟩


theorem HasMFDerivAt.hasMFDerivWithinAt (h : HasMFDerivAt I I' f x f') :
    HasMFDerivWithinAt I I' f s x f' :=
  ⟨ContinuousAt.continuousWithinAt h.1, HasFDerivWithinAt.mono h.2 inter_subset_right⟩


theorem HasMFDerivWithinAt.mdifferentiableWithinAt (h : HasMFDerivWithinAt I I' f s x f') :
    MDifferentiableWithinAt I I' f s x :=
  ⟨h.1, ⟨f', h.2⟩⟩


theorem HasMFDerivAt.mdifferentiableAt (h : HasMFDerivAt I I' f x f') :
    MDifferentiableAt I I' f x := by
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
    f : M → M'
    x : M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivAt I I' f x f'
    ⊢ MDifferentiableAt I I' f x
  -/
  rw [mdifferentiableAt_iff]
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
    f : M → M'
    x : M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivAt I I' f x f'
    ⊢ And (ContinuousAt f x) (DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I' x …
  -/
  exact ⟨h.1, ⟨f', h.2⟩⟩
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem hasMFDerivWithinAt_univ :
    HasMFDerivWithinAt I I' f univ x f' ↔ HasMFDerivAt I I' f x f' := by
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
    f : M → M'
    x : M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    ⊢ Iff (HasMFDerivWithinAt I I' f Set.univ x f') (HasMFDerivAt I I' f x f')
  -/
  simp only [HasMFDerivWithinAt, HasMFDerivAt, continuousWithinAt_univ, mfld_simps]
  /-
    🎉 no goals
  -/


theorem hasMFDerivAt_unique (h₀ : HasMFDerivAt I I' f x f₀') (h₁ : HasMFDerivAt I I' f x f₁') :
    f₀' = f₁' := by
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
    f : M → M'
    x : M
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace  …
    h₀ : HasMFDerivAt I I' f x f₀'
    h₁ : HasMFDerivAt I I' f x f₁'
    ⊢ Eq f₀' f₁'
  -/
  rw [← hasMFDerivWithinAt_univ] at h₀ h₁
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
    f : M → M'
    x : M
    f₀' f₁' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace  …
    h₀ : HasMFDerivWithinAt I I' f Set.univ x f₀'
    h₁ : HasMFDerivWithinAt I I' f Set.univ x f₁'
    ⊢ Eq f₀' f₁'
  -/
  exact (uniqueMDiffWithinAt_univ I).eq h₀ h₁
  /-
    🎉 no goals
  -/


theorem hasMFDerivWithinAt_inter' (h : t ∈ 𝓝[s] x) :
    HasMFDerivWithinAt I I' f (s ∩ t) x f' ↔ HasMFDerivWithinAt I I' f s x f' := by
  rw [HasMFDerivWithinAt, HasMFDerivWithinAt, extChartAt_preimage_inter_eq,
    hasFDerivWithinAt_inter', continuousWithinAt_inter' h]
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
    f : M → M'
    x : M
    s t : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : Membership.mem (nhdsWithin x s) t
    ⊢ Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.preimage  …
  -/
  exact extChartAt_preimage_mem_nhdsWithin h
  /-
    🎉 no goals
  -/


theorem hasMFDerivWithinAt_inter (h : t ∈ 𝓝 x) :
    HasMFDerivWithinAt I I' f (s ∩ t) x f' ↔ HasMFDerivWithinAt I I' f s x f' := by
  rw [HasMFDerivWithinAt, HasMFDerivWithinAt, extChartAt_preimage_inter_eq, hasFDerivWithinAt_inter,
    continuousWithinAt_inter h]
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
    f : M → M'
    x : M
    s t : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : Membership.mem (nhds x) t
    ⊢ Membership.mem (nhds (↑(extChartAt I x) x)) (Set.preimage (↑(extChartAt I x) …
  -/
  exact extChartAt_preimage_mem_nhds h
  /-
    🎉 no goals
  -/


theorem HasMFDerivWithinAt.union (hs : HasMFDerivWithinAt I I' f s x f')
    (ht : HasMFDerivWithinAt I I' f t x f') : HasMFDerivWithinAt I I' f (s ∪ t) x f' := by
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
    f : M → M'
    x : M
    s t : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    hs : HasMFDerivWithinAt I I' f s x f'
    ht : HasMFDerivWithinAt I I' f t x f'
    ⊢ HasMFDerivWithinAt I I' f (Union.union s t) x f'
  -/
  constructor
    /-
      case left
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
      f : M → M'
      x : M
      s t : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      hs : HasMFDerivWithinAt I I' f s x f'
      ht : HasMFDerivWithinAt I I' f t x f'
      ⊢ ContinuousWithinAt f (Union.union s t) x
    -/
  · exact ContinuousWithinAt.union hs.1 ht.1
    /-
      🎉 no goals
    -/
    /-
      case right
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
      f : M → M'
      x : M
      s t : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      hs : HasMFDerivWithinAt I I' f s x f'
      ht : HasMFDerivWithinAt I I' f t x f'
      ⊢ HasFDerivWithinAt (writtenInExtChartAt I I' x f) f' (Inter.inter (Set.preima …
    -/
  · convert HasFDerivWithinAt.union hs.2 ht.2 using 1
    /-
      case h.e'_13
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
      f : M → M'
      x : M
      s t : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      hs : HasMFDerivWithinAt I I' f s x f'
      ht : HasMFDerivWithinAt I I' f t x f'
      ⊢ Eq (Inter.inter (Set.preimage (↑(extChartAt I x).symm) (Union.union s t)) (S …
    -/
    simp only [union_inter_distrib_right, preimage_union]
    /-
      🎉 no goals
    -/


theorem HasMFDerivWithinAt.mono_of_mem_nhdsWithin
    (h : HasMFDerivWithinAt I I' f s x f') (ht : s ∈ 𝓝[t] x) :
    HasMFDerivWithinAt I I' f t x f' :=
  (hasMFDerivWithinAt_inter' ht).1 (h.mono inter_subset_right)


@[deprecated (since := "2024-10-31")]
alias HasMFDerivWithinAt.mono_of_mem := HasMFDerivWithinAt.mono_of_mem_nhdsWithin


theorem HasMFDerivWithinAt.hasMFDerivAt (h : HasMFDerivWithinAt I I' f s x f') (hs : s ∈ 𝓝 x) :
    HasMFDerivAt I I' f x f' := by
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
    f : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivWithinAt I I' f s x f'
    hs : Membership.mem (nhds x) s
    ⊢ HasMFDerivAt I I' f x f'
  -/
  rwa [← univ_inter s, hasMFDerivWithinAt_inter hs, hasMFDerivWithinAt_univ] at h
  /-
    🎉 no goals
  -/


theorem MDifferentiableWithinAt.hasMFDerivWithinAt (h : MDifferentiableWithinAt I I' f s x) :
    HasMFDerivWithinAt I I' f s x (mfderivWithin I I' f s x) := by
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
    f : M → M'
    x : M
    s : Set M
    h : MDifferentiableWithinAt I I' f s x
    ⊢ HasMFDerivWithinAt I I' f s x (mfderivWithin I I' f s x)
  -/
  refine ⟨h.1, ?_⟩
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
    f : M → M'
    x : M
    s : Set M
    h : MDifferentiableWithinAt I I' f s x
    ⊢ HasFDerivWithinAt (writtenInExtChartAt I I' x f) (mfderivWithin I I' f s x)  …
  -/
  simp only [mfderivWithin, h, if_pos, mfld_simps]
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
    f : M → M'
    x : M
    s : Set M
    h : MDifferentiableWithinAt I I' f s x
    ⊢ HasFDerivWithinAt (Function.comp (Function.comp ↑I' ↑(chartAt H' (f x))) (Fu …
  -/
  exact DifferentiableWithinAt.hasFDerivWithinAt h.2
  /-
    🎉 no goals
  -/


theorem mdifferentiableWithinAt_iff_exists_hasMFDerivWithinAt :
    MDifferentiableWithinAt I I' f s x ↔ ∃ f', HasMFDerivWithinAt I I' f s x f' := by
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
    f : M → M'
    x : M
    s : Set M
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (Exists fun f' => HasMFDerivWithinA …
  -/
  refine ⟨fun h ↦ ⟨mfderivWithin I I' f s x, h.hasMFDerivWithinAt⟩, ?_⟩
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
    f : M → M'
    x : M
    s : Set M
    ⊢ (Exists fun f' => HasMFDerivWithinAt I I' f s x f') → MDifferentiableWithinA …
  -/
  rintro ⟨f', hf'⟩
  /-
    case intro
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
    f : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    hf' : HasMFDerivWithinAt I I' f s x f'
    ⊢ MDifferentiableWithinAt I I' f s x
  -/
  exact hf'.mdifferentiableWithinAt
  /-
    🎉 no goals
  -/


theorem MDifferentiableWithinAt.mono_of_mem_nhdsWithin
    (h : MDifferentiableWithinAt I I' f s x) {t : Set M}
    (hst : s ∈ 𝓝[t] x) : MDifferentiableWithinAt I I' f t x :=
  (h.hasMFDerivWithinAt.mono_of_mem_nhdsWithin hst).mdifferentiableWithinAt


theorem MDifferentiableWithinAt.congr_nhds (h : MDifferentiableWithinAt I I' f s x) {t : Set M}
    (hst : 𝓝[s] x = 𝓝[t] x) : MDifferentiableWithinAt I I' f t x :=
  h.mono_of_mem_nhdsWithin <| hst ▸ self_mem_nhdsWithin


theorem mdifferentiableWithinAt_congr_nhds {t : Set M} (hst : 𝓝[s] x = 𝓝[t] x) :
    MDifferentiableWithinAt I I' f s x ↔ MDifferentiableWithinAt I I' f t x :=
  ⟨fun h => h.congr_nhds hst, fun h => h.congr_nhds hst.symm⟩


protected theorem MDifferentiableWithinAt.mfderivWithin (h : MDifferentiableWithinAt I I' f s x) :
    mfderivWithin I I' f s x =
      fderivWithin 𝕜 (writtenInExtChartAt I I' x f : _) ((extChartAt I x).symm ⁻¹' s ∩ range I)
        ((extChartAt I x) x) := by
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
    f : M → M'
    x : M
    s : Set M
    h : MDifferentiableWithinAt I I' f s x
    ⊢ Eq (mfderivWithin I I' f s x) (fderivWithin 𝕜 (writtenInExtChartAt I I' x f) …
  -/
  simp only [mfderivWithin, h, if_pos]
  /-
    🎉 no goals
  -/


theorem MDifferentiableAt.hasMFDerivAt (h : MDifferentiableAt I I' f x) :
    HasMFDerivAt I I' f x (mfderiv I I' f x) := by
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
    f : M → M'
    x : M
    h : MDifferentiableAt I I' f x
    ⊢ HasMFDerivAt I I' f x (mfderiv I I' f x)
  -/
  refine ⟨h.continuousAt, ?_⟩
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
    f : M → M'
    x : M
    h : MDifferentiableAt I I' f x
    ⊢ HasFDerivWithinAt (writtenInExtChartAt I I' x f) (mfderiv I I' f x) (Set.ran …
  -/
  simp only [mfderiv, h, if_pos, mfld_simps]
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
    f : M → M'
    x : M
    h : MDifferentiableAt I I' f x
    ⊢ HasFDerivWithinAt (Function.comp (Function.comp ↑I' ↑(chartAt H' (f x))) (Fu …
  -/
  exact DifferentiableWithinAt.hasFDerivWithinAt h.differentiableWithinAt_writtenInExtChartAt
  /-
    🎉 no goals
  -/


protected theorem MDifferentiableAt.mfderiv (h : MDifferentiableAt I I' f x) :
    mfderiv I I' f x =
      fderivWithin 𝕜 (writtenInExtChartAt I I' x f : _) (range I) ((extChartAt I x) x) := by
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
    f : M → M'
    x : M
    h : MDifferentiableAt I I' f x
    ⊢ Eq (mfderiv I I' f x) (fderivWithin 𝕜 (writtenInExtChartAt I I' x f) (Set.ra …
  -/
  simp only [mfderiv, h, if_pos]
  /-
    🎉 no goals
  -/


protected theorem HasMFDerivAt.mfderiv (h : HasMFDerivAt I I' f x f') : mfderiv I I' f x = f' :=
  (hasMFDerivAt_unique h h.mdifferentiableAt.hasMFDerivAt).symm


protected theorem HasMFDerivWithinAt.mfderivWithin (h : HasMFDerivWithinAt I I' f s x f')
    (hxs : UniqueMDiffWithinAt I s x) : mfderivWithin I I' f s x = f' := by
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
    f : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivWithinAt I I' f s x f'
    hxs : UniqueMDiffWithinAt I s x
    ⊢ Eq (mfderivWithin I I' f s x) f'
  -/
  ext
  /-
    case h
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
    f : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivWithinAt I I' f s x f'
    hxs : UniqueMDiffWithinAt I s x
    x✝ : TangentSpace I x
    ⊢ Eq ((mfderivWithin I I' f s x) x✝) (f' x✝)
  -/
  rw [hxs.eq h h.mdifferentiableWithinAt.hasMFDerivWithinAt]
  /-
    🎉 no goals
  -/


theorem HasMFDerivWithinAt.mfderivWithin_eq_zero (h : HasMFDerivWithinAt I I' f s x 0) :
    mfderivWithin I I' f s x = 0 := by
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
    f : M → M'
    x : M
    s : Set M
    h : HasMFDerivWithinAt I I' f s x 0
    ⊢ Eq (mfderivWithin I I' f s x) 0
  -/
  simp only [mfld_simps, mfderivWithin, h.mdifferentiableWithinAt, ↓reduceIte]
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
    f : M → M'
    x : M
    s : Set M
    h : HasMFDerivWithinAt I I' f s x 0
    ⊢ Eq (fderivWithin 𝕜 (Function.comp (Function.comp ↑I' ↑(chartAt H' (f x))) (F …
  -/
  simp only [HasMFDerivWithinAt, mfld_simps] at h
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
    f : M → M'
    x : M
    s : Set M
    h : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (Function.comp (Function …
    ⊢ Eq (fderivWithin 𝕜 (Function.comp (Function.comp ↑I' ↑(chartAt H' (f x))) (F …
  -/
  rw [fderivWithin, if_pos]
  /-
    case hc
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
    f : M → M'
    x : M
    s : Set M
    h : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (Function.comp (Function …
    ⊢ HasFDerivWithinAt (Function.comp (Function.comp ↑I' ↑(chartAt H' (f x))) (Fu …
  -/
  exact h.2
  /-
    🎉 no goals
  -/


theorem MDifferentiable.mfderivWithin (h : MDifferentiableAt I I' f x)
    (hxs : UniqueMDiffWithinAt I s x) : mfderivWithin I I' f s x = mfderiv I I' f x := by
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
    f : M → M'
    x : M
    s : Set M
    h : MDifferentiableAt I I' f x
    hxs : UniqueMDiffWithinAt I s x
    ⊢ Eq (_root_.mfderivWithin I I' f s x) (mfderiv I I' f x)
  -/
  apply HasMFDerivWithinAt.mfderivWithin _ hxs
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
    f : M → M'
    x : M
    s : Set M
    h : MDifferentiableAt I I' f x
    hxs : UniqueMDiffWithinAt I s x
    ⊢ HasMFDerivWithinAt I I' f s x (mfderiv I I' f x)
  -/
  exact h.hasMFDerivAt.hasMFDerivWithinAt
  /-
    🎉 no goals
  -/


theorem mfderivWithin_subset (st : s ⊆ t) (hs : UniqueMDiffWithinAt I s x)
    (h : MDifferentiableWithinAt I I' f t x) :
    mfderivWithin I I' f s x = mfderivWithin I I' f t x :=
  ((MDifferentiableWithinAt.hasMFDerivWithinAt h).mono st).mfderivWithin hs


@[simp, mfld_simps]
theorem mfderivWithin_univ : mfderivWithin I I' f univ = mfderiv I I' f := by
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
    f : M → M'
    ⊢ Eq (mfderivWithin I I' f Set.univ) (mfderiv I I' f)
  -/
  ext x : 1
  /-
    case h
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
    f : M → M'
    x : M
    ⊢ Eq (mfderivWithin I I' f Set.univ x) (mfderiv I I' f x)
  -/
  simp only [mfderivWithin, mfderiv, mfld_simps]
  /-
    case h
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
    f : M → M'
    x : M
    ⊢ Eq (ite (MDifferentiableWithinAt I I' f Set.univ x) (fderivWithin 𝕜 (Functio …
  -/
  rw [mdifferentiableWithinAt_univ]
  /-
    🎉 no goals
  -/


theorem mfderivWithin_inter (ht : t ∈ 𝓝 x) :
    mfderivWithin I I' f (s ∩ t) x = mfderivWithin I I' f s x := by
  rw [mfderivWithin, mfderivWithin, extChartAt_preimage_inter_eq, mdifferentiableWithinAt_inter ht,
    fderivWithin_inter (extChartAt_preimage_mem_nhds ht)]


theorem mfderivWithin_of_mem_nhds (h : s ∈ 𝓝 x) : mfderivWithin I I' f s x = mfderiv I I' f x := by
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
    f : M → M'
    x : M
    s : Set M
    h : Membership.mem (nhds x) s
    ⊢ Eq (mfderivWithin I I' f s x) (mfderiv I I' f x)
  -/
  rw [← mfderivWithin_univ, ← univ_inter s, mfderivWithin_inter h]
  /-
    🎉 no goals
  -/


lemma mfderivWithin_of_isOpen (hs : IsOpen s) (hx : x ∈ s) :
    mfderivWithin I I' f s x = mfderiv I I' f x :=
  mfderivWithin_of_mem_nhds (hs.mem_nhds hx)


theorem hasMFDerivWithinAt_insert {y : M} :
    HasMFDerivWithinAt I I' f (insert y s) x f' ↔ HasMFDerivWithinAt I I' f s x f' := by
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
    f : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    y : M
    ⊢ Iff (HasMFDerivWithinAt I I' f (Insert.insert y s) x f') (HasMFDerivWithinAt …
  -/
  have : T1Space M := I.t1Space M
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
    f : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    y : M
    this : T1Space M
    ⊢ Iff (HasMFDerivWithinAt I I' f (Insert.insert y s) x f') (HasMFDerivWithinAt …
  -/
  refine ⟨fun h => h.mono <| subset_insert y s, fun hf ↦ ?_⟩
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
    f : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    y : M
    this : T1Space M
    hf : HasMFDerivWithinAt I I' f s x f'
    ⊢ HasMFDerivWithinAt I I' f (Insert.insert y s) x f'
  -/
  rcases eq_or_ne x y with rfl | h
    /-
      case inl
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      this : T1Space M
      hf : HasMFDerivWithinAt I I' f s x f'
      ⊢ HasMFDerivWithinAt I I' f (Insert.insert x s) x f'
    -/
  · rw [HasMFDerivWithinAt] at hf ⊢
    /-
      case inl
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      this : T1Space M
      hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
      ⊢ And (ContinuousWithinAt f (Insert.insert x s) x) (HasFDerivWithinAt (written …
    -/
    refine ⟨hf.1.insert, ?_⟩
    have : (extChartAt I x).target ∈
        𝓝[(extChartAt I x).symm ⁻¹' insert x s ∩ range I] (extChartAt I x) x :=
      nhdsWithin_mono _ inter_subset_right (extChartAt_target_mem_nhdsWithin x)
    /-
      case inl
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      this✝ : T1Space M
      hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
      this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
      ⊢ HasFDerivWithinAt (writtenInExtChartAt I I' x f) f' (Inter.inter (Set.preima …
    -/
    rw [← hasFDerivWithinAt_inter' this]
    /-
      case inl
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      this✝ : T1Space M
      hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
      this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
      ⊢ HasFDerivWithinAt (writtenInExtChartAt I I' x f) f' (Inter.inter (Inter.inte …
    -/
    apply hf.2.insert.mono
    /-
      case inl
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      this✝ : T1Space M
      hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
      this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
      ⊢ HasSubset.Subset (Inter.inter (Inter.inter (Set.preimage (↑(extChartAt I x). …
    -/
    rintro z ⟨⟨hz, h2z⟩, h'z⟩
    /-
      case inl.intro.intro
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      this✝ : T1Space M
      hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
      this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
      z : E
      h'z : Membership.mem (extChartAt I x).target z
      hz : Membership.mem (Set.preimage (↑(extChartAt I x).symm) (Insert.insert x s) …
      h2z : Membership.mem (Set.range ↑I) z
      ⊢ Membership.mem (Insert.insert (↑(extChartAt I x) x) (Inter.inter (Set.preima …
    -/
    simp only [mem_inter_iff, mem_preimage, mem_insert_iff, mem_range] at hz h2z ⊢
    /-
      case inl.intro.intro
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      this✝ : T1Space M
      hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
      this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
      z : E
      h'z : Membership.mem (extChartAt I x).target z
      hz : Or (Eq (↑(extChartAt I x).symm z) x) (Membership.mem s (↑(extChartAt I x) …
      h2z : Exists fun y => Eq (↑I y) z
      ⊢ Or (Eq z (↑(extChartAt I x) x)) (And (Membership.mem s (↑(extChartAt I x).sy …
    -/
    rcases hz with xz | h'z
      /-
        case inl.intro.intro.inl
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
        f : M → M'
        x : M
        s : Set M
        f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
        this✝ : T1Space M
        hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
        this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
        z : E
        h'z : Membership.mem (extChartAt I x).target z
        h2z : Exists fun y => Eq (↑I y) z
        xz : Eq (↑(extChartAt I x).symm z) x
        ⊢ Or (Eq z (↑(extChartAt I x) x)) (And (Membership.mem s (↑(extChartAt I x).sy …
      -/
    · left
      /-
        case inl.intro.intro.inl.h
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
        f : M → M'
        x : M
        s : Set M
        f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
        this✝ : T1Space M
        hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
        this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
        z : E
        h'z : Membership.mem (extChartAt I x).target z
        h2z : Exists fun y => Eq (↑I y) z
        xz : Eq (↑(extChartAt I x).symm z) x
        ⊢ Eq z (↑(extChartAt I x) x)
      -/
      have : x ∈ (extChartAt I x).source := mem_extChartAt_source x
      /-
        case inl.intro.intro.inl.h
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
        f : M → M'
        x : M
        s : Set M
        f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
        this✝¹ : T1Space M
        hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
        this✝ : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.pre …
        z : E
        h'z : Membership.mem (extChartAt I x).target z
        h2z : Exists fun y => Eq (↑I y) z
        xz : Eq (↑(extChartAt I x).symm z) x
        this : Membership.mem (extChartAt I x).source x
        ⊢ Eq z (↑(extChartAt I x) x)
      -/
      exact (((extChartAt I x).eq_symm_apply this h'z).1 xz.symm).symm
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.intro.inr
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
        f : M → M'
        x : M
        s : Set M
        f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
        this✝ : T1Space M
        hf : And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt I  …
        this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
        z : E
        h'z✝ : Membership.mem (extChartAt I x).target z
        h2z : Exists fun y => Eq (↑I y) z
        h'z : Membership.mem s (↑(extChartAt I x).symm z)
        ⊢ Or (Eq z (↑(extChartAt I x) x)) (And (Membership.mem s (↑(extChartAt I x).sy …
      -/
    · exact Or.inr ⟨h'z, h2z⟩
      /-
        🎉 no goals
      -/
    /-
      case inr
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      y : M
      this : T1Space M
      hf : HasMFDerivWithinAt I I' f s x f'
      h : Ne x y
      ⊢ HasMFDerivWithinAt I I' f (Insert.insert y s) x f'
    -/
  · apply hf.mono_of_mem_nhdsWithin ?_
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
      f : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      y : M
      this : T1Space M
      hf : HasMFDerivWithinAt I I' f s x f'
      h : Ne x y
      ⊢ Membership.mem (nhdsWithin x (Insert.insert y s)) s
    -/
    simp_rw [nhdsWithin_insert_of_ne h, self_mem_nhdsWithin]
    /-
      🎉 no goals
    -/


alias ⟨HasMFDerivWithinAt.of_insert, HasMFDerivWithinAt.insert'⟩ := hasMFDerivWithinAt_insert


protected theorem HasMFDerivWithinAt.insert (h : HasMFDerivWithinAt I I' f s x f') :
    HasMFDerivWithinAt I I' f (insert x s) x f' :=
  h.insert'


theorem hasMFDerivWithinAt_diff_singleton (y : M) :
    HasMFDerivWithinAt I I' f (s \ {y}) x f' ↔ HasMFDerivWithinAt I I' f s x f' := by
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
    f : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    y : M
    ⊢ Iff (HasMFDerivWithinAt I I' f (SDiff.sdiff s (Singleton.singleton y)) x f') …
  -/
  rw [← hasMFDerivWithinAt_insert, insert_diff_singleton, hasMFDerivWithinAt_insert]
  /-
    🎉 no goals
  -/


theorem mfderivWithin_eq_mfderiv (hs : UniqueMDiffWithinAt I s x) (h : MDifferentiableAt I I' f x) :
    mfderivWithin I I' f s x = mfderiv I I' f x := by
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
    f : M → M'
    x : M
    s : Set M
    hs : UniqueMDiffWithinAt I s x
    h : MDifferentiableAt I I' f x
    ⊢ Eq (mfderivWithin I I' f s x) (mfderiv I I' f x)
  -/
  rw [← mfderivWithin_univ]
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
    f : M → M'
    x : M
    s : Set M
    hs : UniqueMDiffWithinAt I s x
    h : MDifferentiableAt I I' f x
    ⊢ Eq (mfderivWithin I I' f s x) (mfderivWithin I I' f Set.univ x)
  -/
  exact mfderivWithin_subset (subset_univ _) hs h.mdifferentiableWithinAt
  /-
    🎉 no goals
  -/


theorem mdifferentiableWithinAt_insert_self :
    MDifferentiableWithinAt I I' f (insert x s) x ↔ MDifferentiableWithinAt I I' f s x :=
  ⟨fun h ↦ h.mono (subset_insert x s), fun h ↦ h.hasMFDerivWithinAt.insert.mdifferentiableWithinAt⟩


theorem mdifferentiableWithinAt_insert {y : M} :
    MDifferentiableWithinAt I I' f (insert y s) x ↔ MDifferentiableWithinAt I I' f s x := by
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
    f : M → M'
    x : M
    s : Set M
    y : M
    ⊢ Iff (MDifferentiableWithinAt I I' f (Insert.insert y s) x) (MDifferentiableW …
  -/
  rcases eq_or_ne x y with (rfl | h)
    /-
      case inl
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
      f : M → M'
      x : M
      s : Set M
      ⊢ Iff (MDifferentiableWithinAt I I' f (Insert.insert x s) x) (MDifferentiableW …
    -/
  · exact mdifferentiableWithinAt_insert_self
    /-
      🎉 no goals
    -/
  /-
    case inr
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
    f : M → M'
    x : M
    s : Set M
    y : M
    h : Ne x y
    ⊢ Iff (MDifferentiableWithinAt I I' f (Insert.insert y s) x) (MDifferentiableW …
  -/
  have : T1Space M := I.t1Space M
  /-
    case inr
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
    f : M → M'
    x : M
    s : Set M
    y : M
    h : Ne x y
    this : T1Space M
    ⊢ Iff (MDifferentiableWithinAt I I' f (Insert.insert y s) x) (MDifferentiableW …
  -/
  apply mdifferentiableWithinAt_congr_nhds
  /-
    case inr.hst
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
    f : M → M'
    x : M
    s : Set M
    y : M
    h : Ne x y
    this : T1Space M
    ⊢ Eq (nhdsWithin x (Insert.insert y s)) (nhdsWithin x s)
  -/
  exact nhdsWithin_insert_of_ne h
  /-
    🎉 no goals
  -/


alias ⟨MDifferentiableWithinAt.of_insert, MDifferentiableWithinAt.insert'⟩ :=
mdifferentiableWithinAt_insert


protected theorem MDifferentiableWithinAt.insert (h : MDifferentiableWithinAt I I' f s x) :
    MDifferentiableWithinAt I I' f (insert x s) x :=
  h.insert'


theorem HasMFDerivWithinAt.continuousWithinAt (h : HasMFDerivWithinAt I I' f s x f') :
    ContinuousWithinAt f s x :=
  h.1


theorem HasMFDerivAt.continuousAt (h : HasMFDerivAt I I' f x f') : ContinuousAt f x :=
  h.1


theorem tangentMapWithin_subset {p : TangentBundle I M} (st : s ⊆ t)
    (hs : UniqueMDiffWithinAt I s p.1) (h : MDifferentiableWithinAt I I' f t p.1) :
    tangentMapWithin I I' f s p = tangentMapWithin I I' f t p := by
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
    f : M → M'
    s t : Set M
    p : TangentBundle I M
    st : HasSubset.Subset s t
    hs : UniqueMDiffWithinAt I s p.proj
    h : MDifferentiableWithinAt I I' f t p.proj
    ⊢ Eq (tangentMapWithin I I' f s p) (tangentMapWithin I I' f t p)
  -/
  simp only [tangentMapWithin, mfld_simps]
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
    f : M → M'
    s t : Set M
    p : TangentBundle I M
    st : HasSubset.Subset s t
    hs : UniqueMDiffWithinAt I s p.proj
    h : MDifferentiableWithinAt I I' f t p.proj
    ⊢ Eq ((mfderivWithin I I' f s p.proj) p.snd) ((mfderivWithin I I' f t p.proj)  …
  -/
  rw [mfderivWithin_subset st hs h]
  /-
    🎉 no goals
  -/


theorem tangentMapWithin_univ : tangentMapWithin I I' f univ = tangentMap I I' f := by
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
    f : M → M'
    ⊢ Eq (tangentMapWithin I I' f Set.univ) (tangentMap I I' f)
  -/
  ext p : 1
  /-
    case h
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
    f : M → M'
    p : TangentBundle I M
    ⊢ Eq (tangentMapWithin I I' f Set.univ p) (tangentMap I I' f p)
  -/
  simp only [tangentMapWithin, tangentMap, mfld_simps]
  /-
    🎉 no goals
  -/


theorem tangentMapWithin_eq_tangentMap {p : TangentBundle I M} (hs : UniqueMDiffWithinAt I s p.1)
    (h : MDifferentiableAt I I' f p.1) : tangentMapWithin I I' f s p = tangentMap I I' f p := by
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
    f : M → M'
    s : Set M
    p : TangentBundle I M
    hs : UniqueMDiffWithinAt I s p.proj
    h : MDifferentiableAt I I' f p.proj
    ⊢ Eq (tangentMapWithin I I' f s p) (tangentMap I I' f p)
  -/
  rw [← mdifferentiableWithinAt_univ] at h
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
    f : M → M'
    s : Set M
    p : TangentBundle I M
    hs : UniqueMDiffWithinAt I s p.proj
    h : MDifferentiableWithinAt I I' f Set.univ p.proj
    ⊢ Eq (tangentMapWithin I I' f s p) (tangentMap I I' f p)
  -/
  rw [← tangentMapWithin_univ]
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
    f : M → M'
    s : Set M
    p : TangentBundle I M
    hs : UniqueMDiffWithinAt I s p.proj
    h : MDifferentiableWithinAt I I' f Set.univ p.proj
    ⊢ Eq (tangentMapWithin I I' f s p) (tangentMapWithin I I' f Set.univ p)
  -/
  exact tangentMapWithin_subset (subset_univ _) hs h
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem tangentMapWithin_proj {p : TangentBundle I M} :
    (tangentMapWithin I I' f s p).proj = f p.proj :=
  rfl


@[simp, mfld_simps]
theorem tangentMap_proj {p : TangentBundle I M} : (tangentMap I I' f p).proj = f p.proj :=
  rfl


/-- If two sets coincide locally around `x`, except maybe at a point `y`, then their
preimage under `extChartAt x` coincide locally, except maybe at `extChartAt I x x`. -/
theorem preimage_extChartAt_eventuallyEq_compl_singleton (y : M) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    ((extChartAt I x).symm ⁻¹' s ∩ range I : Set E) =ᶠ[𝓝[{extChartAt I x x}ᶜ] (extChartAt I x x)]
    ((extChartAt I x).symm ⁻¹' t ∩ range I : Set E) := by
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
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ (nhdsWithin (↑(extChartAt I x) x) (HasCompl.compl (Singleton.singleton (↑(ex …
  -/
  have : T1Space M := I.t1Space M
  obtain ⟨u, u_mem, hu⟩ : ∃ u ∈ 𝓝 x, u ∩ {x}ᶜ ⊆ {y | (y ∈ s) = (y ∈ t)} :=
    mem_nhdsWithin_iff_exists_mem_nhds_inter.1 (nhdsWithin_compl_singleton_le x y h)
  /-
    case intro.intro
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
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    u : Set M
    u_mem : Membership.mem (nhds x) u
    hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
    ⊢ (nhdsWithin (↑(extChartAt I x) x) (HasCompl.compl (Singleton.singleton (↑(ex …
  -/
  rw [← extChartAt_to_inv (I:= I) x] at u_mem
  have B : (extChartAt I x).target ∪ (range I)ᶜ ∈ 𝓝 (extChartAt I x x) := by
    rw [← nhdsWithin_univ, ← union_compl_self (range I), nhdsWithin_union]
    apply Filter.union_mem_sup (extChartAt_target_mem_nhdsWithin x) self_mem_nhdsWithin
  apply mem_nhdsWithin_iff_exists_mem_nhds_inter.2
    ⟨_, Filter.inter_mem ((continuousAt_extChartAt_symm x).preimage_mem_nhds u_mem) B, ?_⟩
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
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    u : Set M
    u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
    hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
    B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
    ⊢ HasSubset.Subset (Inter.inter (Inter.inter (Set.preimage (↑(extChartAt I x). …
  -/
  rintro z ⟨hz, h'z⟩
  /-
    case intro
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
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    u : Set M
    u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
    hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
    B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
    z : E
    hz : Membership.mem (Inter.inter (Set.preimage (↑(extChartAt I x).symm) u) (Un …
    h'z : Membership.mem (HasCompl.compl (Singleton.singleton (↑(extChartAt I x) x …
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => Eq (Inter.inter (Set.preimage ( …
  -/
  simp only [eq_iff_iff, mem_setOf_eq]
  /-
    case intro
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
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    u : Set M
    u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
    hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
    B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
    z : E
    hz : Membership.mem (Inter.inter (Set.preimage (↑(extChartAt I x).symm) u) (Un …
    h'z : Membership.mem (HasCompl.compl (Singleton.singleton (↑(extChartAt I x) x …
    ⊢ Iff (Inter.inter (Set.preimage (↑(extChartAt I x).symm) s) (Set.range ↑I) z) …
  -/
  change z ∈ (extChartAt I x).symm ⁻¹' s ∩ range I ↔ z ∈ (extChartAt I x).symm ⁻¹' t ∩ range I
  /-
    case intro
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
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    u : Set M
    u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
    hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
    B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
    z : E
    hz : Membership.mem (Inter.inter (Set.preimage (↑(extChartAt I x).symm) u) (Un …
    h'z : Membership.mem (HasCompl.compl (Singleton.singleton (↑(extChartAt I x) x …
    ⊢ Iff (Membership.mem (Inter.inter (Set.preimage (↑(extChartAt I x).symm) s) ( …
  -/
  by_cases hIz : z ∈ range I
    /-
      case pos
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
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      u : Set M
      u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
      hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
      B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
      z : E
      hz : Membership.mem (Inter.inter (Set.preimage (↑(extChartAt I x).symm) u) (Un …
      h'z : Membership.mem (HasCompl.compl (Singleton.singleton (↑(extChartAt I x) x …
      hIz : Membership.mem (Set.range ↑I) z
      ⊢ Iff (Membership.mem (Inter.inter (Set.preimage (↑(extChartAt I x).symm) s) ( …
    -/
  · simp [-extChartAt, hIz] at hz ⊢
    /-
      case pos
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
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      u : Set M
      u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
      hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
      B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
      z : E
      h'z : Membership.mem (HasCompl.compl (Singleton.singleton (↑(extChartAt I x) x …
      hIz : Membership.mem (Set.range ↑I) z
      hz : And (Membership.mem u (↑(extChartAt I x).symm z)) (Membership.mem (extCha …
      ⊢ Iff (Membership.mem s (↑(extChartAt I x).symm z)) (Membership.mem t (↑(extCh …
    -/
    rw [← eq_iff_iff]
    /-
      case pos
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
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      u : Set M
      u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
      hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
      B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
      z : E
      h'z : Membership.mem (HasCompl.compl (Singleton.singleton (↑(extChartAt I x) x …
      hIz : Membership.mem (Set.range ↑I) z
      hz : And (Membership.mem u (↑(extChartAt I x).symm z)) (Membership.mem (extCha …
      ⊢ Eq (Membership.mem s (↑(extChartAt I x).symm z)) (Membership.mem t (↑(extCha …
    -/
    apply hu ⟨hz.1, ?_⟩
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
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      u : Set M
      u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
      hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
      B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
      z : E
      h'z : Membership.mem (HasCompl.compl (Singleton.singleton (↑(extChartAt I x) x …
      hIz : Membership.mem (Set.range ↑I) z
      hz : And (Membership.mem u (↑(extChartAt I x).symm z)) (Membership.mem (extCha …
      ⊢ Membership.mem (HasCompl.compl (Singleton.singleton x)) (↑(extChartAt I x).s …
    -/
    simp only [mem_compl_iff, mem_singleton_iff, ne_comm, ne_eq] at h'z ⊢
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
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      u : Set M
      u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
      hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
      B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
      z : E
      hIz : Membership.mem (Set.range ↑I) z
      hz : And (Membership.mem u (↑(extChartAt I x).symm z)) (Membership.mem (extCha …
      h'z : Not (Eq z (↑(extChartAt I x) x))
      ⊢ Not (Eq x (↑(extChartAt I x).symm z))
    -/
    rw [(extChartAt I x).eq_symm_apply (by simp) hz.2]
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
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      u : Set M
      u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
      hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
      B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
      z : E
      hIz : Membership.mem (Set.range ↑I) z
      hz : And (Membership.mem u (↑(extChartAt I x).symm z)) (Membership.mem (extCha …
      h'z : Not (Eq z (↑(extChartAt I x) x))
      ⊢ Not (Eq (↑(extChartAt I x) x) z)
    -/
    exact Ne.symm h'z
    /-
      🎉 no goals
    -/
    /-
      case neg
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
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      u : Set M
      u_mem : Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) u
      hu : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) …
      B : Membership.mem (nhds (↑(extChartAt I x) x)) (Union.union (extChartAt I x). …
      z : E
      hz : Membership.mem (Inter.inter (Set.preimage (↑(extChartAt I x).symm) u) (Un …
      h'z : Membership.mem (HasCompl.compl (Singleton.singleton (↑(extChartAt I x) x …
      hIz : Not (Membership.mem (Set.range ↑I) z)
      ⊢ Iff (Membership.mem (Inter.inter (Set.preimage (↑(extChartAt I x).symm) s) ( …
    -/
  · simp [hIz]
    /-
      🎉 no goals
    -/


/-- If two sets coincide locally, except maybe at a point, then it is equivalent to have a manifold
derivative within one or the other. -/
theorem hasMFDerivWithinAt_congr_set' (y : M) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    HasMFDerivWithinAt I I' f s x f' ↔ HasMFDerivWithinAt I I' f t x f' := by
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
    f : M → M'
    x : M
    s t : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ Iff (HasMFDerivWithinAt I I' f s x f') (HasMFDerivWithinAt I I' f t x f')
  -/
  have : T1Space M := I.t1Space M
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
    f : M → M'
    x : M
    s t : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    ⊢ Iff (HasMFDerivWithinAt I I' f s x f') (HasMFDerivWithinAt I I' f t x f')
  -/
  simp only [HasMFDerivWithinAt]
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
    f : M → M'
    x : M
    s t : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    ⊢ Iff (And (ContinuousWithinAt f s x) (HasFDerivWithinAt (writtenInExtChartAt  …
  -/
  refine and_congr ?_ ?_
    /-
      case refine_1
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
      f : M → M'
      x : M
      s t : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      ⊢ Iff (ContinuousWithinAt f s x) (ContinuousWithinAt f t x)
    -/
  · exact continuousWithinAt_congr_set' _ h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
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
      f : M → M'
      x : M
      s t : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      ⊢ Iff (HasFDerivWithinAt (writtenInExtChartAt I I' x f) f' (Inter.inter (Set.p …
    -/
  · apply hasFDerivWithinAt_congr_set' (extChartAt I x x)
    /-
      case refine_2
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
      f : M → M'
      x : M
      s t : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      this : T1Space M
      ⊢ (nhdsWithin (↑(extChartAt I x) x) (HasCompl.compl (Singleton.singleton (↑(ex …
    -/
    exact preimage_extChartAt_eventuallyEq_compl_singleton y h
    /-
      🎉 no goals
    -/


theorem hasMFDerivWithinAt_congr_set (h : s =ᶠ[𝓝 x] t) :
    HasMFDerivWithinAt I I' f s x f' ↔ HasMFDerivWithinAt I I' f t x f' :=
  hasMFDerivWithinAt_congr_set' x <| h.filter_mono inf_le_left


/-- If two sets coincide around a point (except possibly at a single point `y`), then it is
equivalent to be differentiable within one or the other set. -/
theorem mdifferentiableWithinAt_congr_set' (y : M) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    MDifferentiableWithinAt I I' f s x ↔ MDifferentiableWithinAt I I' f t x := by
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
    f : M → M'
    x : M
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (MDifferentiableWithinAt I I' f t x)
  -/
  simp only [mdifferentiableWithinAt_iff_exists_hasMFDerivWithinAt]
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
    f : M → M'
    x : M
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ Iff (Exists fun f' => HasMFDerivWithinAt I I' f s x f') (Exists fun f' => Ha …
  -/
  exact exists_congr fun _ => hasMFDerivWithinAt_congr_set' _ h
  /-
    🎉 no goals
  -/


theorem mdifferentiableWithinAt_congr_set (h : s =ᶠ[𝓝 x] t) :
    MDifferentiableWithinAt I I' f s x ↔ MDifferentiableWithinAt I I' f t x := by
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
    f : M → M'
    x : M
    s t : Set M
    h : (nhds x).EventuallyEq s t
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (MDifferentiableWithinAt I I' f t x)
  -/
  simp only [mdifferentiableWithinAt_iff_exists_hasMFDerivWithinAt]
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
    f : M → M'
    x : M
    s t : Set M
    h : (nhds x).EventuallyEq s t
    ⊢ Iff (Exists fun f' => HasMFDerivWithinAt I I' f s x f') (Exists fun f' => Ha …
  -/
  exact exists_congr fun _ => hasMFDerivWithinAt_congr_set h
  /-
    🎉 no goals
  -/


/-- If two sets coincide locally, except maybe at a point, then derivatives within these sets
are the same. -/
theorem mfderivWithin_congr_set' (y : M) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    mfderivWithin I I' f s x = mfderivWithin I I' f t x := by
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
    f : M → M'
    x : M
    s t : Set M
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ Eq (mfderivWithin I I' f s x) (mfderivWithin I I' f t x)
  -/
  by_cases hx : MDifferentiableWithinAt I I' f s x
    /-
      case pos
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
      f : M → M'
      x : M
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      hx : MDifferentiableWithinAt I I' f s x
      ⊢ Eq (mfderivWithin I I' f s x) (mfderivWithin I I' f t x)
    -/
  · simp only [mfderivWithin, hx, (mdifferentiableWithinAt_congr_set' y h).1 hx, ↓reduceIte]
    /-
      case pos
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
      f : M → M'
      x : M
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      hx : MDifferentiableWithinAt I I' f s x
      ⊢ Eq (fderivWithin 𝕜 (writtenInExtChartAt I I' x f) (Inter.inter (Set.preimage …
    -/
    apply fderivWithin_congr_set' (extChartAt I x x)
    /-
      case pos
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
      f : M → M'
      x : M
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      hx : MDifferentiableWithinAt I I' f s x
      ⊢ (nhdsWithin (↑(extChartAt I x) x) (HasCompl.compl (Singleton.singleton (↑(ex …
    -/
    exact preimage_extChartAt_eventuallyEq_compl_singleton y h
    /-
      🎉 no goals
    -/
    /-
      case neg
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
      f : M → M'
      x : M
      s t : Set M
      y : M
      h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
      hx : Not (MDifferentiableWithinAt I I' f s x)
      ⊢ Eq (mfderivWithin I I' f s x) (mfderivWithin I I' f t x)
    -/
  · simp [mfderivWithin, hx, ← mdifferentiableWithinAt_congr_set' y h]
    /-
      🎉 no goals
    -/


/-- If two sets coincide locally, then derivatives within these sets
are the same. -/
theorem mfderivWithin_congr_set (h : s =ᶠ[𝓝 x] t) :
    mfderivWithin I I' f s x = mfderivWithin I I' f t x :=
  mfderivWithin_congr_set' x <| h.filter_mono inf_le_left


/-- If two sets coincide locally, except maybe at a point, then derivatives within these sets
coincide locally. -/
theorem mfderivWithin_eventually_congr_set' (y : M) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    ∀ᶠ y in 𝓝 x, mfderivWithin I I' f s y = mfderivWithin I I' f t y :=
  (eventually_nhds_nhdsWithin.2 h).mono fun _ => mfderivWithin_congr_set' y


/-- If two sets coincide locally, then derivatives within these sets coincide locally. -/
theorem mfderivWithin_eventually_congr_set (h : s =ᶠ[𝓝 x] t) :
    ∀ᶠ y in 𝓝 x, mfderivWithin I I' f s y = mfderivWithin I I' f t y :=
  mfderivWithin_eventually_congr_set' x <| h.filter_mono inf_le_left


theorem HasMFDerivAt.congr_mfderiv (h : HasMFDerivAt I I' f x f') (h' : f' = f₁') :
    HasMFDerivAt I I' f x f₁' :=
  h' ▸ h


theorem HasMFDerivWithinAt.congr_mfderiv (h : HasMFDerivWithinAt I I' f s x f') (h' : f' = f₁') :
    HasMFDerivWithinAt I I' f s x f₁' :=
  h' ▸ h


theorem HasMFDerivWithinAt.congr_of_eventuallyEq (h : HasMFDerivWithinAt I I' f s x f')
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : HasMFDerivWithinAt I I' f₁ s x f' := by
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
    f f₁ : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivWithinAt I I' f s x f'
    h₁ : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ HasMFDerivWithinAt I I' f₁ s x f'
  -/
  refine ⟨ContinuousWithinAt.congr_of_eventuallyEq h.1 h₁ hx, ?_⟩
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
    f f₁ : M → M'
    x : M
    s : Set M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivWithinAt I I' f s x f'
    h₁ : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ HasFDerivWithinAt (writtenInExtChartAt I I' x f₁) f' (Inter.inter (Set.preim …
  -/
  apply HasFDerivWithinAt.congr_of_eventuallyEq h.2
  · have :
      (extChartAt I x).symm ⁻¹' {y | f₁ y = f y} ∈
        𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] (extChartAt I x) x :=
      extChartAt_preimage_mem_nhdsWithin h₁
    /-
      case h₁
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
      f f₁ : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      h : HasMFDerivWithinAt I I' f s x f'
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
      ⊢ (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.preimage (↑(extChartAt I …
    -/
    apply Filter.mem_of_superset this fun y => _
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
      f f₁ : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      h : HasMFDerivWithinAt I I' f s x f'
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      this : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.prei …
      ⊢ ∀ (y : E), Membership.mem (Set.preimage (↑(extChartAt I x).symm) (setOf fun  …
    -/
    simp +contextual only [hx, mfld_simps]
    /-
      🎉 no goals
    -/
    /-
      case hx
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
      f f₁ : M → M'
      x : M
      s : Set M
      f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
      h : HasMFDerivWithinAt I I' f s x f'
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      ⊢ Eq (writtenInExtChartAt I I' x f₁ (↑(extChartAt I x) x)) (writtenInExtChartA …
    -/
  · simp only [hx, mfld_simps]
    /-
      🎉 no goals
    -/


theorem HasMFDerivWithinAt.congr_mono (h : HasMFDerivWithinAt I I' f s x f')
    (ht : ∀ x ∈ t, f₁ x = f x) (hx : f₁ x = f x) (h₁ : t ⊆ s) : HasMFDerivWithinAt I I' f₁ t x f' :=
  (h.mono h₁).congr_of_eventuallyEq (Filter.mem_inf_of_right ht) hx


theorem HasMFDerivAt.congr_of_eventuallyEq (h : HasMFDerivAt I I' f x f') (h₁ : f₁ =ᶠ[𝓝 x] f) :
    HasMFDerivAt I I' f₁ x f' := by
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
    f f₁ : M → M'
    x : M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivAt I I' f x f'
    h₁ : (nhds x).EventuallyEq f₁ f
    ⊢ HasMFDerivAt I I' f₁ x f'
  -/
  rw [← hasMFDerivWithinAt_univ] at h ⊢
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
    f f₁ : M → M'
    x : M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivWithinAt I I' f Set.univ x f'
    h₁ : (nhds x).EventuallyEq f₁ f
    ⊢ HasMFDerivWithinAt I I' f₁ Set.univ x f'
  -/
  apply h.congr_of_eventuallyEq _ (mem_of_mem_nhds h₁ : _)
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
    f f₁ : M → M'
    x : M
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    h : HasMFDerivWithinAt I I' f Set.univ x f'
    h₁ : (nhds x).EventuallyEq f₁ f
    ⊢ (nhdsWithin x Set.univ).EventuallyEq f₁ f
  -/
  rwa [nhdsWithin_univ]
  /-
    🎉 no goals
  -/


theorem MDifferentiableWithinAt.congr_of_eventuallyEq (h : MDifferentiableWithinAt I I' f s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : MDifferentiableWithinAt I I' f₁ s x :=
  (h.hasMFDerivWithinAt.congr_of_eventuallyEq h₁ hx).mdifferentiableWithinAt


theorem MDifferentiableWithinAt.congr_of_eventuallyEq_of_mem
    (h : MDifferentiableWithinAt I I' f s x) (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s) :
    MDifferentiableWithinAt I I' f₁ s x :=
  h.congr_of_eventuallyEq h₁ (mem_of_mem_nhdsWithin hx h₁ :)


theorem MDifferentiableWithinAt.congr_of_eventuallyEq_insert
    (h : MDifferentiableWithinAt I I' f s x) (h₁ : f₁ =ᶠ[𝓝[insert x s] x] f) :
    MDifferentiableWithinAt I I' f₁ s x :=
  (h.insert.congr_of_eventuallyEq_of_mem h₁ (mem_insert x s)).of_insert


theorem Filter.EventuallyEq.mdifferentiableWithinAt_iff (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) :
    MDifferentiableWithinAt I I' f s x ↔ MDifferentiableWithinAt I I' f₁ s x := by
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
    f f₁ : M → M'
    x : M
    s : Set M
    h₁ : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (MDifferentiableWithinAt I I' f₁ s x)
  -/
  constructor
    /-
      case mp
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
      f f₁ : M → M'
      x : M
      s : Set M
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      ⊢ MDifferentiableWithinAt I I' f s x → MDifferentiableWithinAt I I' f₁ s x
    -/
  · intro h
    /-
      case mp
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
      f f₁ : M → M'
      x : M
      s : Set M
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : MDifferentiableWithinAt I I' f s x
      ⊢ MDifferentiableWithinAt I I' f₁ s x
    -/
    apply h.congr_of_eventuallyEq h₁ hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
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
      f f₁ : M → M'
      x : M
      s : Set M
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      ⊢ MDifferentiableWithinAt I I' f₁ s x → MDifferentiableWithinAt I I' f s x
    -/
  · intro h
    /-
      case mpr
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
      f f₁ : M → M'
      x : M
      s : Set M
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : MDifferentiableWithinAt I I' f₁ s x
      ⊢ MDifferentiableWithinAt I I' f s x
    -/
    apply h.congr_of_eventuallyEq _ hx.symm
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
      f f₁ : M → M'
      x : M
      s : Set M
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : MDifferentiableWithinAt I I' f₁ s x
      ⊢ (nhdsWithin x s).EventuallyEq f f₁
    -/
    apply h₁.mono
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
      f f₁ : M → M'
      x : M
      s : Set M
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : MDifferentiableWithinAt I I' f₁ s x
      ⊢ ∀ (x : M), Eq (f₁ x) (f x) → Eq (f x) (f₁ x)
    -/
    intro y
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
      f f₁ : M → M'
      x : M
      s : Set M
      h₁ : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : MDifferentiableWithinAt I I' f₁ s x
      y : M
      ⊢ Eq (f₁ y) (f y) → Eq (f y) (f₁ y)
    -/
    apply Eq.symm
    /-
      🎉 no goals
    -/


theorem MDifferentiableWithinAt.congr_mono (h : MDifferentiableWithinAt I I' f s x)
    (ht : ∀ x ∈ t, f₁ x = f x) (hx : f₁ x = f x) (h₁ : t ⊆ s) :
    MDifferentiableWithinAt I I' f₁ t x :=
  (HasMFDerivWithinAt.congr_mono h.hasMFDerivWithinAt ht hx h₁).mdifferentiableWithinAt


theorem MDifferentiableWithinAt.congr (h : MDifferentiableWithinAt I I' f s x)
    (ht : ∀ x ∈ s, f₁ x = f x) (hx : f₁ x = f x) : MDifferentiableWithinAt I I' f₁ s x :=
  (HasMFDerivWithinAt.congr_mono h.hasMFDerivWithinAt ht hx (Subset.refl _)).mdifferentiableWithinAt


theorem MDifferentiableOn.congr_mono (h : MDifferentiableOn I I' f s) (h' : ∀ x ∈ t, f₁ x = f x)
    (h₁ : t ⊆ s) : MDifferentiableOn I I' f₁ t := fun x hx =>
  (h x (h₁ hx)).congr_mono h' (h' x hx) h₁


theorem MDifferentiableAt.congr_of_eventuallyEq (h : MDifferentiableAt I I' f x)
    (hL : f₁ =ᶠ[𝓝 x] f) : MDifferentiableAt I I' f₁ x :=
  (h.hasMFDerivAt.congr_of_eventuallyEq hL).mdifferentiableAt


theorem MDifferentiableWithinAt.mfderivWithin_congr_mono (h : MDifferentiableWithinAt I I' f s x)
    (hs : ∀ x ∈ t, f₁ x = f x) (hx : f₁ x = f x) (hxt : UniqueMDiffWithinAt I t x) (h₁ : t ⊆ s) :
    mfderivWithin I I' f₁ t x = mfderivWithin I I' f s x :=
  (HasMFDerivWithinAt.congr_mono h.hasMFDerivWithinAt hs hx h₁).mfderivWithin hxt


theorem MDifferentiableWithinAt.mfderivWithin_mono (h : MDifferentiableWithinAt I I' f s x)
    (hxt : UniqueMDiffWithinAt I t x) (h₁ : t ⊆ s) :
    mfderivWithin I I' f t x = mfderivWithin I I' f s x :=
  h.mfderivWithin_congr_mono (fun _ _ ↦ rfl) rfl hxt h₁


theorem MDifferentiableWithinAt.mfderivWithin_mono_of_mem_nhdsWithin
    (h : MDifferentiableWithinAt I I' f s x) (hxt : UniqueMDiffWithinAt I t x) (h₁ : s ∈ 𝓝[t] x) :
    mfderivWithin I I' f t x = mfderivWithin I I' f s x :=
  (HasMFDerivWithinAt.mono_of_mem_nhdsWithin h.hasMFDerivWithinAt h₁).mfderivWithin hxt


theorem Filter.EventuallyEq.mfderivWithin_eq (hs : UniqueMDiffWithinAt I s x) (hL : f₁ =ᶠ[𝓝[s] x] f)
    (hx : f₁ x = f x) : mfderivWithin I I' f₁ s x = mfderivWithin I I' f s x := by
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
    f f₁ : M → M'
    x : M
    s : Set M
    hs : UniqueMDiffWithinAt I s x
    hL : (nhdsWithin x s).EventuallyEq f₁ f
    hx : Eq (f₁ x) (f x)
    ⊢ Eq (mfderivWithin I I' f₁ s x) (mfderivWithin I I' f s x)
  -/
  by_cases h : MDifferentiableWithinAt I I' f s x
    /-
      case pos
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
      f f₁ : M → M'
      x : M
      s : Set M
      hs : UniqueMDiffWithinAt I s x
      hL : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : MDifferentiableWithinAt I I' f s x
      ⊢ Eq (mfderivWithin I I' f₁ s x) (mfderivWithin I I' f s x)
    -/
  · exact (h.hasMFDerivWithinAt.congr_of_eventuallyEq hL hx).mfderivWithin hs
    /-
      🎉 no goals
    -/
    /-
      case neg
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
      f f₁ : M → M'
      x : M
      s : Set M
      hs : UniqueMDiffWithinAt I s x
      hL : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : Not (MDifferentiableWithinAt I I' f s x)
      ⊢ Eq (mfderivWithin I I' f₁ s x) (mfderivWithin I I' f s x)
    -/
  · unfold mfderivWithin
    /-
      case neg
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
      f f₁ : M → M'
      x : M
      s : Set M
      hs : UniqueMDiffWithinAt I s x
      hL : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : Not (MDifferentiableWithinAt I I' f s x)
      ⊢ Eq (ite (MDifferentiableWithinAt I I' f₁ s x) (fderivWithin 𝕜 (writtenInExtC …
    -/
    rw [if_neg h, if_neg]
    /-
      case neg.hnc
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
      f f₁ : M → M'
      x : M
      s : Set M
      hs : UniqueMDiffWithinAt I s x
      hL : (nhdsWithin x s).EventuallyEq f₁ f
      hx : Eq (f₁ x) (f x)
      h : Not (MDifferentiableWithinAt I I' f s x)
      ⊢ Not (MDifferentiableWithinAt I I' f₁ s x)
    -/
    rwa [← hL.mdifferentiableWithinAt_iff hx]
    /-
      🎉 no goals
    -/


theorem mfderivWithin_congr (hs : UniqueMDiffWithinAt I s x) (hL : ∀ x ∈ s, f₁ x = f x)
    (hx : f₁ x = f x) : mfderivWithin I I' f₁ s x = mfderivWithin I I' f s x :=
  Filter.EventuallyEq.mfderivWithin_eq hs (Filter.eventuallyEq_of_mem self_mem_nhdsWithin hL) hx


theorem tangentMapWithin_congr (h : ∀ x ∈ s, f x = f₁ x) (p : TangentBundle I M) (hp : p.1 ∈ s)
    (hs : UniqueMDiffWithinAt I s p.1) :
    tangentMapWithin I I' f s p = tangentMapWithin I I' f₁ s p := by
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
    f f₁ : M → M'
    s : Set M
    h : ∀ (x : M), Membership.mem s x → Eq (f x) (f₁ x)
    p : TangentBundle I M
    hp : Membership.mem s p.proj
    hs : UniqueMDiffWithinAt I s p.proj
    ⊢ Eq (tangentMapWithin I I' f s p) (tangentMapWithin I I' f₁ s p)
  -/
  refine TotalSpace.ext (h p.1 hp) ?_
  -- This used to be `simp only`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
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
    f f₁ : M → M'
    s : Set M
    h : ∀ (x : M), Membership.mem s x → Eq (f x) (f₁ x)
    p : TangentBundle I M
    hp : Membership.mem s p.proj
    hs : UniqueMDiffWithinAt I s p.proj
    ⊢ HEq (tangentMapWithin I I' f s p).snd (tangentMapWithin I I' f₁ s p).snd
  -/
  rw [tangentMapWithin, h p.1 hp, tangentMapWithin, mfderivWithin_congr hs h (h _ hp)]
  /-
    🎉 no goals
  -/


theorem Filter.EventuallyEq.mfderiv_eq (hL : f₁ =ᶠ[𝓝 x] f) :
    mfderiv I I' f₁ x = mfderiv I I' f x := by
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
    f f₁ : M → M'
    x : M
    hL : (nhds x).EventuallyEq f₁ f
    ⊢ Eq (mfderiv I I' f₁ x) (mfderiv I I' f x)
  -/
  have A : f₁ x = f x := (mem_of_mem_nhds hL : _)
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
    f f₁ : M → M'
    x : M
    hL : (nhds x).EventuallyEq f₁ f
    A : Eq (f₁ x) (f x)
    ⊢ Eq (mfderiv I I' f₁ x) (mfderiv I I' f x)
  -/
  rw [← mfderivWithin_univ, ← mfderivWithin_univ]
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
    f f₁ : M → M'
    x : M
    hL : (nhds x).EventuallyEq f₁ f
    A : Eq (f₁ x) (f x)
    ⊢ Eq (mfderivWithin I I' f₁ Set.univ x) (mfderivWithin I I' f Set.univ x)
  -/
  rw [← nhdsWithin_univ] at hL
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
    f f₁ : M → M'
    x : M
    hL : (nhdsWithin x Set.univ).EventuallyEq f₁ f
    A : Eq (f₁ x) (f x)
    ⊢ Eq (mfderivWithin I I' f₁ Set.univ x) (mfderivWithin I I' f Set.univ x)
  -/
  exact hL.mfderivWithin_eq (uniqueMDiffWithinAt_univ I) A
  /-
    🎉 no goals
  -/


/-- A congruence lemma for `mfderiv`, (ab)using the fact that `TangentSpace I' (f x)` is
definitionally equal to `E'`. -/
theorem mfderiv_congr_point {x' : M} (h : x = x') :
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
                                                                    f : M → M'
                                                                    x x' : M
                                                                    h : Eq x x'
                                                                    ⊢ Eq (mfderiv I I' f x) (mfderiv I I' f x')
                                                                  -/
    @Eq (E →L[𝕜] E') (mfderiv I I' f x) (mfderiv I I' f x') := by subst h; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- A congruence lemma for `mfderiv`, (ab)using the fact that `TangentSpace I' (f x)` is
definitionally equal to `E'`. -/
theorem mfderiv_congr {f' : M → M'} (h : f = f') :
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
                                                                    f : M → M'
                                                                    x : M
                                                                    f' : M → M'
                                                                    h : Eq f f'
                                                                    ⊢ Eq (mfderiv I I' f x) (mfderiv I I' f' x)
                                                                  -/
    @Eq (E →L[𝕜] E') (mfderiv I I' f x) (mfderiv I I' f' x) := by subst h; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem HasMFDerivWithinAt.comp (hg : HasMFDerivWithinAt I' I'' g u (f x) g')
    (hf : HasMFDerivWithinAt I I' f s x f') (hst : s ⊆ f ⁻¹' u) :
    HasMFDerivWithinAt I I'' (g ∘ f) s x (g'.comp f') := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I' (f x)) (TangentSpace  …
    hg : HasMFDerivWithinAt I' I'' g u (f x) g'
    hf : HasMFDerivWithinAt I I' f s x f'
    hst : HasSubset.Subset s (Set.preimage f u)
    ⊢ HasMFDerivWithinAt I I'' (Function.comp g f) s x (g'.comp f')
  -/
  refine ⟨ContinuousWithinAt.comp hg.1 hf.1 hst, ?_⟩
  have A :
    HasFDerivWithinAt (writtenInExtChartAt I' I'' (f x) g ∘ writtenInExtChartAt I I' x f)
      (ContinuousLinearMap.comp g' f' : E →L[𝕜] E'') ((extChartAt I x).symm ⁻¹' s ∩ range I)
      ((extChartAt I x) x) := by
    have :
      (extChartAt I x).symm ⁻¹' (f ⁻¹' (extChartAt I' (f x)).source) ∈
        𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] (extChartAt I x) x :=
      extChartAt_preimage_mem_nhdsWithin
        (hf.1.preimage_mem_nhdsWithin (extChartAt_source_mem_nhds _))
    unfold HasMFDerivWithinAt at *
    rw [← hasFDerivWithinAt_inter' this, ← extChartAt_preimage_inter_eq] at hf ⊢
    have : writtenInExtChartAt I I' x f ((extChartAt I x) x) = (extChartAt I' (f x)) (f x) := by
      simp only [mfld_simps]
    rw [← this] at hg
    apply HasFDerivWithinAt.comp ((extChartAt I x) x) hg.2 hf.2 _
    intro y hy
    simp only [mfld_simps] at hy
    have : f (((chartAt H x).symm : H → M) (I.symm y)) ∈ u := hst hy.1.1
    simp only [hy, this, mfld_simps]
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I' (f x)) (TangentSpace  …
    hg : HasMFDerivWithinAt I' I'' g u (f x) g'
    hf : HasMFDerivWithinAt I I' f s x f'
    hst : HasSubset.Subset s (Set.preimage f u)
    A : HasFDerivWithinAt (Function.comp (writtenInExtChartAt I' I'' (f x) g) (wri …
    ⊢ HasFDerivWithinAt (writtenInExtChartAt I I'' x (Function.comp g f)) (g'.comp …
  -/
  apply A.congr_of_eventuallyEq (writtenInExtChartAt_comp hf.1)
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I' (f x)) (TangentSpace  …
    hg : HasMFDerivWithinAt I' I'' g u (f x) g'
    hf : HasMFDerivWithinAt I I' f s x f'
    hst : HasSubset.Subset s (Set.preimage f u)
    A : HasFDerivWithinAt (Function.comp (writtenInExtChartAt I' I'' (f x) g) (wri …
    ⊢ Eq (writtenInExtChartAt I I'' x (Function.comp g f) (↑(extChartAt I x) x)) ( …
  -/
  simp only [mfld_simps]
  /-
    🎉 no goals
  -/


/-- The **chain rule for manifolds**. -/
theorem HasMFDerivAt.comp (hg : HasMFDerivAt I' I'' g (f x) g') (hf : HasMFDerivAt I I' f x f') :
    HasMFDerivAt I I'' (g ∘ f) x (g'.comp f') := by
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
    f : M → M'
    x : M
    g : M' → M''
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I' (f x)) (TangentSpace  …
    hg : HasMFDerivAt I' I'' g (f x) g'
    hf : HasMFDerivAt I I' f x f'
    ⊢ HasMFDerivAt I I'' (Function.comp g f) x (g'.comp f')
  -/
  rw [← hasMFDerivWithinAt_univ] at *
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
    f : M → M'
    x : M
    g : M' → M''
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I' (f x)) (TangentSpace  …
    hg : HasMFDerivWithinAt I' I'' g Set.univ (f x) g'
    hf : HasMFDerivWithinAt I I' f Set.univ x f'
    ⊢ HasMFDerivWithinAt I I'' (Function.comp g f) Set.univ x (g'.comp f')
  -/
  exact HasMFDerivWithinAt.comp x (hg.mono (subset_univ _)) hf subset_preimage_univ
  /-
    🎉 no goals
  -/


theorem HasMFDerivAt.comp_hasMFDerivWithinAt (hg : HasMFDerivAt I' I'' g (f x) g')
    (hf : HasMFDerivWithinAt I I' f s x f') :
    HasMFDerivWithinAt I I'' (g ∘ f) s x (g'.comp f') := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I' (f x)) (TangentSpace  …
    hg : HasMFDerivAt I' I'' g (f x) g'
    hf : HasMFDerivWithinAt I I' f s x f'
    ⊢ HasMFDerivWithinAt I I'' (Function.comp g f) s x (g'.comp f')
  -/
  rw [← hasMFDerivWithinAt_univ] at *
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    f' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I x) (TangentSpace I' (f …
    g' : ContinuousLinearMap (RingHom.id 𝕜) (TangentSpace I' (f x)) (TangentSpace  …
    hg : HasMFDerivWithinAt I' I'' g Set.univ (f x) g'
    hf : HasMFDerivWithinAt I I' f s x f'
    ⊢ HasMFDerivWithinAt I I'' (Function.comp g f) s x (g'.comp f')
  -/
  exact HasMFDerivWithinAt.comp x (hg.mono (subset_univ _)) hf subset_preimage_univ
  /-
    🎉 no goals
  -/


theorem MDifferentiableWithinAt.comp (hg : MDifferentiableWithinAt I' I'' g u (f x))
    (hf : MDifferentiableWithinAt I I' f s x) (h : s ⊆ f ⁻¹' u) :
    MDifferentiableWithinAt I I'' (g ∘ f) s x := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  rcases hf.2 with ⟨f', hf'⟩
  /-
    case intro
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    hf' : HasFDerivWithinAt (Function.comp (↑I') (Function.comp (Function.comp (↑( …
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  have F : HasMFDerivWithinAt I I' f s x f' := ⟨hf.1, hf'⟩
  /-
    case intro
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    hf' : HasFDerivWithinAt (Function.comp (↑I') (Function.comp (Function.comp (↑( …
    F : HasMFDerivWithinAt I I' f s x f'
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  rcases hg.2 with ⟨g', hg'⟩
  /-
    case intro.intro
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    hf' : HasFDerivWithinAt (Function.comp (↑I') (Function.comp (Function.comp (↑( …
    F : HasMFDerivWithinAt I I' f s x f'
    g' : ContinuousLinearMap (RingHom.id 𝕜) E' E''
    hg' : HasFDerivWithinAt (Function.comp (↑I'') (Function.comp (Function.comp (↑ …
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  have G : HasMFDerivWithinAt I' I'' g u (f x) g' := ⟨hg.1, hg'⟩
  /-
    case intro.intro
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E'
    hf' : HasFDerivWithinAt (Function.comp (↑I') (Function.comp (Function.comp (↑( …
    F : HasMFDerivWithinAt I I' f s x f'
    g' : ContinuousLinearMap (RingHom.id 𝕜) E' E''
    hg' : HasFDerivWithinAt (Function.comp (↑I'') (Function.comp (Function.comp (↑ …
    G : HasMFDerivWithinAt I' I'' g u (f x) g'
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  exact (HasMFDerivWithinAt.comp x G F h).mdifferentiableWithinAt
  /-
    🎉 no goals
  -/


theorem MDifferentiableWithinAt.comp_of_eq {y : M'} (hg : MDifferentiableWithinAt I' I'' g u y)
    (hf : MDifferentiableWithinAt I I' f s x) (h : s ⊆ f ⁻¹' u) (hy : f x = y) :
    MDifferentiableWithinAt I I'' (g ∘ f) s x := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    y : M'
    hg : MDifferentiableWithinAt I' I'' g u y
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    hy : Eq (f x) y
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  subst hy; exact hg.comp _ hf h
            /-
              🎉 no goals
            -/


theorem MDifferentiableWithinAt.comp_of_preimage_mem_nhdsWithin
    (hg : MDifferentiableWithinAt I' I'' g u (f x))
    (hf : MDifferentiableWithinAt I I' f s x) (h : f ⁻¹' u ∈ 𝓝[s] x) :
    MDifferentiableWithinAt I I'' (g ∘ f) s x :=
  (hg.comp _ (hf.mono inter_subset_right) inter_subset_left).mono_of_mem_nhdsWithin
    (Filter.inter_mem h self_mem_nhdsWithin)


theorem MDifferentiableWithinAt.comp_of_preimage_mem_nhdsWithin_of_eq {y : M'}
    (hg : MDifferentiableWithinAt I' I'' g u y)
    (hf : MDifferentiableWithinAt I I' f s x) (h : f ⁻¹' u ∈ 𝓝[s] x) (hy : f x = y) :
    MDifferentiableWithinAt I I'' (g ∘ f) s x := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    y : M'
    hg : MDifferentiableWithinAt I' I'' g u y
    hf : MDifferentiableWithinAt I I' f s x
    h : Membership.mem (nhdsWithin x s) (Set.preimage f u)
    hy : Eq (f x) y
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  subst hy; exact MDifferentiableWithinAt.comp_of_preimage_mem_nhdsWithin _ hg hf h
            /-
              🎉 no goals
            -/


theorem MDifferentiableAt.comp (hg : MDifferentiableAt I' I'' g (f x))
    (hf : MDifferentiableAt I I' f x) : MDifferentiableAt I I'' (g ∘ f) x :=
  (hg.hasMFDerivAt.comp x hf.hasMFDerivAt).mdifferentiableAt


theorem MDifferentiableAt.comp_of_eq {y : M'} (hg : MDifferentiableAt I' I'' g y)
    (hf : MDifferentiableAt I I' f x) (hy : f x = y) : MDifferentiableAt I I'' (g ∘ f) x := by
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
    f : M → M'
    x : M
    g : M' → M''
    y : M'
    hg : MDifferentiableAt I' I'' g y
    hf : MDifferentiableAt I I' f x
    hy : Eq (f x) y
    ⊢ MDifferentiableAt I I'' (Function.comp g f) x
  -/
  subst hy; exact hg.comp _ hf
            /-
              🎉 no goals
            -/


theorem MDifferentiableAt.comp_mdifferentiableWithinAt
    (hg : MDifferentiableAt I' I'' g (f x)) (hf : MDifferentiableWithinAt I I' f s x) :
    MDifferentiableWithinAt I I'' (g ∘ f) s x := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    hg : MDifferentiableAt I' I'' g (f x)
    hf : MDifferentiableWithinAt I I' f s x
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  rw [← mdifferentiableWithinAt_univ] at hg
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    hg : MDifferentiableWithinAt I' I'' g Set.univ (f x)
    hf : MDifferentiableWithinAt I I' f s x
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  exact hg.comp _ hf (by simp)
  /-
    🎉 no goals
  -/


theorem MDifferentiableAt.comp_mdifferentiableWithinAt_of_eq {y : M'}
    (hg : MDifferentiableAt I' I'' g y) (hf : MDifferentiableWithinAt I I' f s x) (hy : f x = y) :
    MDifferentiableWithinAt I I'' (g ∘ f) s x := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    y : M'
    hg : MDifferentiableAt I' I'' g y
    hf : MDifferentiableWithinAt I I' f s x
    hy : Eq (f x) y
    ⊢ MDifferentiableWithinAt I I'' (Function.comp g f) s x
  -/
  subst hy; exact hg.comp_mdifferentiableWithinAt _ hf
            /-
              🎉 no goals
            -/


theorem mfderivWithin_comp (hg : MDifferentiableWithinAt I' I'' g u (f x))
    (hf : MDifferentiableWithinAt I I' f s x) (h : s ⊆ f ⁻¹' u) (hxs : UniqueMDiffWithinAt I s x) :
    mfderivWithin I I'' (g ∘ f) s x =
      (mfderivWithin I' I'' g u (f x)).comp (mfderivWithin I I' f s x) := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    hxs : UniqueMDiffWithinAt I s x
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) s x) ((mfderivWithin I' I'' g u  …
  -/
  apply HasMFDerivWithinAt.mfderivWithin _ hxs
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    hxs : UniqueMDiffWithinAt I s x
    ⊢ HasMFDerivWithinAt I I'' (Function.comp g f) s x ((mfderivWithin I' I'' g u  …
  -/
  exact HasMFDerivWithinAt.comp x hg.hasMFDerivWithinAt hf.hasMFDerivWithinAt h
  /-
    🎉 no goals
  -/


theorem mfderivWithin_comp_of_eq {x : M} {y : M'} (hg : MDifferentiableWithinAt I' I'' g u y)
    (hf : MDifferentiableWithinAt I I' f s x) (h : s ⊆ f ⁻¹' u) (hxs : UniqueMDiffWithinAt I s x)
    (hy : f x = y) :
    mfderivWithin I I'' (g ∘ f) s x =
      (mfderivWithin I' I'' g u y).comp (mfderivWithin I I' f s x) := by
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
    f : M → M'
    s : Set M
    g : M' → M''
    u : Set M'
    x : M
    y : M'
    hg : MDifferentiableWithinAt I' I'' g u y
    hf : MDifferentiableWithinAt I I' f s x
    h : HasSubset.Subset s (Set.preimage f u)
    hxs : UniqueMDiffWithinAt I s x
    hy : Eq (f x) y
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) s x) ((mfderivWithin I' I'' g u  …
  -/
  subst hy; exact mfderivWithin_comp x hg hf h hxs
            /-
              🎉 no goals
            -/


theorem mfderivWithin_comp_of_preimage_mem_nhdsWithin
    (hg : MDifferentiableWithinAt I' I'' g u (f x))
    (hf : MDifferentiableWithinAt I I' f s x) (h : f ⁻¹' u ∈ 𝓝[s] x)
    (hxs : UniqueMDiffWithinAt I s x) :
    mfderivWithin I I'' (g ∘ f) s x =
      (mfderivWithin I' I'' g u (f x)).comp (mfderivWithin I I' f s x) := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : Membership.mem (nhdsWithin x s) (Set.preimage f u)
    hxs : UniqueMDiffWithinAt I s x
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) s x) ((mfderivWithin I' I'' g u  …
  -/
  have A : s ∩ f ⁻¹' u ∈ 𝓝[s] x := Filter.inter_mem self_mem_nhdsWithin h
  have B : mfderivWithin I I'' (g ∘ f) s x = mfderivWithin I I'' (g ∘ f) (s ∩ f ⁻¹' u) x := by
    apply MDifferentiableWithinAt.mfderivWithin_mono_of_mem_nhdsWithin _ hxs A
    exact hg.comp _ (hf.mono inter_subset_left) inter_subset_right
  have C : mfderivWithin I I' f s x = mfderivWithin I I' f (s ∩ f ⁻¹' u) x :=
    MDifferentiableWithinAt.mfderivWithin_mono_of_mem_nhdsWithin (hf.mono inter_subset_left) hxs A
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : Membership.mem (nhdsWithin x s) (Set.preimage f u)
    hxs : UniqueMDiffWithinAt I s x
    A : Membership.mem (nhdsWithin x s) (Inter.inter s (Set.preimage f u))
    B : Eq (mfderivWithin I I'' (Function.comp g f) s x) (mfderivWithin I I'' (Fun …
    C : Eq (mfderivWithin I I' f s x) (mfderivWithin I I' f (Inter.inter s (Set.pr …
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) s x) ((mfderivWithin I' I'' g u  …
  -/
  rw [B, C]
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    hg : MDifferentiableWithinAt I' I'' g u (f x)
    hf : MDifferentiableWithinAt I I' f s x
    h : Membership.mem (nhdsWithin x s) (Set.preimage f u)
    hxs : UniqueMDiffWithinAt I s x
    A : Membership.mem (nhdsWithin x s) (Inter.inter s (Set.preimage f u))
    B : Eq (mfderivWithin I I'' (Function.comp g f) s x) (mfderivWithin I I'' (Fun …
    C : Eq (mfderivWithin I I' f s x) (mfderivWithin I I' f (Inter.inter s (Set.pr …
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) (Inter.inter s (Set.preimage f u …
  -/
  exact mfderivWithin_comp _ hg (hf.mono inter_subset_left) inter_subset_right (hxs.inter' h)
  /-
    🎉 no goals
  -/


theorem mfderivWithin_comp_of_preimage_mem_nhdsWithin_of_eq {y : M'}
    (hg : MDifferentiableWithinAt I' I'' g u y)
    (hf : MDifferentiableWithinAt I I' f s x) (h : f ⁻¹' u ∈ 𝓝[s] x)
    (hxs : UniqueMDiffWithinAt I s x) (hy : f x = y) :
    mfderivWithin I I'' (g ∘ f) s x =
      (mfderivWithin I' I'' g u y).comp (mfderivWithin I I' f s x) := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    u : Set M'
    y : M'
    hg : MDifferentiableWithinAt I' I'' g u y
    hf : MDifferentiableWithinAt I I' f s x
    h : Membership.mem (nhdsWithin x s) (Set.preimage f u)
    hxs : UniqueMDiffWithinAt I s x
    hy : Eq (f x) y
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) s x) ((mfderivWithin I' I'' g u  …
  -/
  subst hy; exact mfderivWithin_comp_of_preimage_mem_nhdsWithin _ hg hf h hxs
            /-
              🎉 no goals
            -/


theorem mfderiv_comp_mfderivWithin (hg : MDifferentiableAt I' I'' g (f x))
    (hf : MDifferentiableWithinAt I I' f s x) (hxs : UniqueMDiffWithinAt I s x) :
    mfderivWithin I I'' (g ∘ f) s x =
      (mfderiv I' I'' g (f x)).comp (mfderivWithin I I' f s x) := by
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    hg : MDifferentiableAt I' I'' g (f x)
    hf : MDifferentiableWithinAt I I' f s x
    hxs : UniqueMDiffWithinAt I s x
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) s x) ((mfderiv I' I'' g (f x)).c …
  -/
  rw [← mfderivWithin_univ]
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
    f : M → M'
    x : M
    s : Set M
    g : M' → M''
    hg : MDifferentiableAt I' I'' g (f x)
    hf : MDifferentiableWithinAt I I' f s x
    hxs : UniqueMDiffWithinAt I s x
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) s x) ((mfderivWithin I' I'' g Se …
  -/
  exact mfderivWithin_comp _ hg.mdifferentiableWithinAt hf (by simp) hxs
  /-
    🎉 no goals
  -/


theorem mfderiv_comp_mfderivWithin_of_eq {x : M} {y : M'} (hg : MDifferentiableAt I' I'' g y)
    (hf : MDifferentiableWithinAt I I' f s x) (hxs : UniqueMDiffWithinAt I s x) (hy : f x = y) :
    mfderivWithin I I'' (g ∘ f) s x =
      (mfderiv I' I'' g y).comp (mfderivWithin I I' f s x) := by
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
    f : M → M'
    s : Set M
    g : M' → M''
    x : M
    y : M'
    hg : MDifferentiableAt I' I'' g y
    hf : MDifferentiableWithinAt I I' f s x
    hxs : UniqueMDiffWithinAt I s x
    hy : Eq (f x) y
    ⊢ Eq (mfderivWithin I I'' (Function.comp g f) s x) ((mfderiv I' I'' g y).comp  …
  -/
  subst hy; exact mfderiv_comp_mfderivWithin x hg hf hxs
            /-
              🎉 no goals
            -/


theorem mfderiv_comp (hg : MDifferentiableAt I' I'' g (f x)) (hf : MDifferentiableAt I I' f x) :
    mfderiv I I'' (g ∘ f) x = (mfderiv I' I'' g (f x)).comp (mfderiv I I' f x) := by
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
    f : M → M'
    x : M
    g : M' → M''
    hg : MDifferentiableAt I' I'' g (f x)
    hf : MDifferentiableAt I I' f x
    ⊢ Eq (mfderiv I I'' (Function.comp g f) x) ((mfderiv I' I'' g (f x)).comp (mfd …
  -/
  apply HasMFDerivAt.mfderiv
  /-
    case h
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
    f : M → M'
    x : M
    g : M' → M''
    hg : MDifferentiableAt I' I'' g (f x)
    hf : MDifferentiableAt I I' f x
    ⊢ HasMFDerivAt I I'' (Function.comp g f) x ((mfderiv I' I'' g (f x)).comp (mfd …
  -/
  exact HasMFDerivAt.comp x hg.hasMFDerivAt hf.hasMFDerivAt
  /-
    🎉 no goals
  -/


theorem mfderiv_comp_of_eq {x : M} {y : M'} (hg : MDifferentiableAt I' I'' g y)
    (hf : MDifferentiableAt I I' f x) (hy : f x = y) :
    mfderiv I I'' (g ∘ f) x = (mfderiv I' I'' g (f x)).comp (mfderiv I I' f x) := by
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
    f : M → M'
    g : M' → M''
    x : M
    y : M'
    hg : MDifferentiableAt I' I'' g y
    hf : MDifferentiableAt I I' f x
    hy : Eq (f x) y
    ⊢ Eq (mfderiv I I'' (Function.comp g f) x) ((mfderiv I' I'' g (f x)).comp (mfd …
  -/
  subst hy; exact mfderiv_comp x hg hf
            /-
              🎉 no goals
            -/


theorem mfderiv_comp_apply (hg : MDifferentiableAt I' I'' g (f x))
    (hf : MDifferentiableAt I I' f x) (v : TangentSpace I x) :
    mfderiv I I'' (g ∘ f) x v = (mfderiv I' I'' g (f x)) ((mfderiv I I' f x) v) := by
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
    f : M → M'
    x : M
    g : M' → M''
    hg : MDifferentiableAt I' I'' g (f x)
    hf : MDifferentiableAt I I' f x
    v : TangentSpace I x
    ⊢ Eq ((mfderiv I I'' (Function.comp g f) x) v) ((mfderiv I' I'' g (f x)) ((mfd …
  -/
  rw [mfderiv_comp _ hg hf]
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
    f : M → M'
    x : M
    g : M' → M''
    hg : MDifferentiableAt I' I'' g (f x)
    hf : MDifferentiableAt I I' f x
    v : TangentSpace I x
    ⊢ Eq (((mfderiv I' I'' g (f x)).comp (mfderiv I I' f x)) v) ((mfderiv I' I'' g …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mfderiv_comp_apply_of_eq {y : M'} (hg : MDifferentiableAt I' I'' g y)
    (hf : MDifferentiableAt I I' f x)  (hy : f x = y) (v : TangentSpace I x) :
    mfderiv I I'' (g ∘ f) x v = (mfderiv I' I'' g y) ((mfderiv I I' f x) v) := by
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
    f : M → M'
    x : M
    g : M' → M''
    y : M'
    hg : MDifferentiableAt I' I'' g y
    hf : MDifferentiableAt I I' f x
    hy : Eq (f x) y
    v : TangentSpace I x
    ⊢ Eq ((mfderiv I I'' (Function.comp g f) x) v) ((mfderiv I' I'' g y) ((mfderiv …
  -/
  subst hy; exact mfderiv_comp_apply _ hg hf v
            /-
              🎉 no goals
            -/


theorem MDifferentiableOn.comp (hg : MDifferentiableOn I' I'' g u) (hf : MDifferentiableOn I I' f s)
    (st : s ⊆ f ⁻¹' u) : MDifferentiableOn I I'' (g ∘ f) s := fun x hx =>
  MDifferentiableWithinAt.comp x (hg (f x) (st hx)) (hf x hx) st


theorem MDifferentiable.comp_mdifferentiableOn (hg : MDifferentiable I' I'' g)
    (hf : MDifferentiableOn I I' f s) : MDifferentiableOn I I'' (g ∘ f) s := by
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
    f : M → M'
    s : Set M
    g : M' → M''
    hg : MDifferentiable I' I'' g
    hf : MDifferentiableOn I I' f s
    ⊢ MDifferentiableOn I I'' (Function.comp g f) s
  -/
  rw [← mdifferentiableOn_univ] at hg
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
    f : M → M'
    s : Set M
    g : M' → M''
    hg : MDifferentiableOn I' I'' g Set.univ
    hf : MDifferentiableOn I I' f s
    ⊢ MDifferentiableOn I I'' (Function.comp g f) s
  -/
  exact hg.comp hf (by simp)
  /-
    🎉 no goals
  -/


theorem MDifferentiable.comp (hg : MDifferentiable I' I'' g) (hf : MDifferentiable I I' f) :
    MDifferentiable I I'' (g ∘ f) := fun x => MDifferentiableAt.comp x (hg (f x)) (hf x)


theorem tangentMapWithin_comp_at (p : TangentBundle I M)
    (hg : MDifferentiableWithinAt I' I'' g u (f p.1)) (hf : MDifferentiableWithinAt I I' f s p.1)
    (h : s ⊆ f ⁻¹' u) (hps : UniqueMDiffWithinAt I s p.1) :
    tangentMapWithin I I'' (g ∘ f) s p =
      tangentMapWithin I' I'' g u (tangentMapWithin I I' f s p) := by
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
    f : M → M'
    s : Set M
    g : M' → M''
    u : Set M'
    p : TangentBundle I M
    hg : MDifferentiableWithinAt I' I'' g u (f p.proj)
    hf : MDifferentiableWithinAt I I' f s p.proj
    h : HasSubset.Subset s (Set.preimage f u)
    hps : UniqueMDiffWithinAt I s p.proj
    ⊢ Eq (tangentMapWithin I I'' (Function.comp g f) s p) (tangentMapWithin I' I'' …
  -/
  simp only [tangentMapWithin, mfld_simps]
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
    f : M → M'
    s : Set M
    g : M' → M''
    u : Set M'
    p : TangentBundle I M
    hg : MDifferentiableWithinAt I' I'' g u (f p.proj)
    hf : MDifferentiableWithinAt I I' f s p.proj
    h : HasSubset.Subset s (Set.preimage f u)
    hps : UniqueMDiffWithinAt I s p.proj
    ⊢ Eq ((mfderivWithin I I'' (Function.comp g f) s p.proj) p.snd) ((mfderivWithi …
  -/
  rw [mfderivWithin_comp p.1 hg hf h hps]
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
    f : M → M'
    s : Set M
    g : M' → M''
    u : Set M'
    p : TangentBundle I M
    hg : MDifferentiableWithinAt I' I'' g u (f p.proj)
    hf : MDifferentiableWithinAt I I' f s p.proj
    h : HasSubset.Subset s (Set.preimage f u)
    hps : UniqueMDiffWithinAt I s p.proj
    ⊢ Eq (((mfderivWithin I' I'' g u (f p.proj)).comp (mfderivWithin I I' f s p.pr …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem tangentMap_comp_at (p : TangentBundle I M) (hg : MDifferentiableAt I' I'' g (f p.1))
    (hf : MDifferentiableAt I I' f p.1) :
    tangentMap I I'' (g ∘ f) p = tangentMap I' I'' g (tangentMap I I' f p) := by
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
    f : M → M'
    g : M' → M''
    p : TangentBundle I M
    hg : MDifferentiableAt I' I'' g (f p.proj)
    hf : MDifferentiableAt I I' f p.proj
    ⊢ Eq (tangentMap I I'' (Function.comp g f) p) (tangentMap I' I'' g (tangentMap …
  -/
  simp only [tangentMap, mfld_simps]
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
    f : M → M'
    g : M' → M''
    p : TangentBundle I M
    hg : MDifferentiableAt I' I'' g (f p.proj)
    hf : MDifferentiableAt I I' f p.proj
    ⊢ Eq ((mfderiv I I'' (Function.comp g f) p.proj) p.snd) ((mfderiv I' I'' g (f  …
  -/
  rw [mfderiv_comp p.1 hg hf]
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
    f : M → M'
    g : M' → M''
    p : TangentBundle I M
    hg : MDifferentiableAt I' I'' g (f p.proj)
    hf : MDifferentiableAt I I' f p.proj
    ⊢ Eq (((mfderiv I' I'' g (f p.proj)).comp (mfderiv I I' f p.proj)) p.snd) ((mf …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem tangentMap_comp (hg : MDifferentiable I' I'' g) (hf : MDifferentiable I I' f) :
    tangentMap I I'' (g ∘ f) = tangentMap I' I'' g ∘ tangentMap I I' f := by
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
    f : M → M'
    g : M' → M''
    hg : MDifferentiable I' I'' g
    hf : MDifferentiable I I' f
    ⊢ Eq (tangentMap I I'' (Function.comp g f)) (Function.comp (tangentMap I' I''  …
  -/
  ext p : 1; exact tangentMap_comp_at _ (hg _) (hf _)
             /-
               🎉 no goals
             -/


