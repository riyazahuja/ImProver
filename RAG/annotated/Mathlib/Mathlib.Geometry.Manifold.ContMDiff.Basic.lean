/-- The composition of `C^n` functions within domains at points is `C^n`. -/
theorem ContMDiffWithinAt.comp {t : Set M'} {g : M' → M''} (x : M)
    (hg : ContMDiffWithinAt I' I'' n g t (f x)) (hf : ContMDiffWithinAt I I' n f s x)
    (st : MapsTo f s t) : ContMDiffWithinAt I I'' n (g ∘ f) s x := by
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    s : Set M
    n : ENat
    t : Set M'
    g : M' → M''
    x : M
    hg : ContMDiffWithinAt I' I'' n g t (f x)
    hf : ContMDiffWithinAt I I' n f s x
    st : Set.MapsTo f s t
    ⊢ ContMDiffWithinAt I I'' n (Function.comp g f) s x
  -/
  rw [contMDiffWithinAt_iff] at hg hf ⊢
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    s : Set M
    n : ENat
    t : Set M'
    g : M' → M''
    x : M
    hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
    hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    st : Set.MapsTo f s t
    ⊢ And (ContinuousWithinAt (Function.comp g f) s x) (ContDiffWithinAt 𝕜 (↑n) (F …
  -/
  refine ⟨hg.1.comp hf.1 st, ?_⟩
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    s : Set M
    n : ENat
    t : Set M'
    g : M' → M''
    x : M
    hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
    hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    st : Set.MapsTo f s t
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I'' (Function.comp g f  …
  -/
  set e := extChartAt I x
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    s : Set M
    n : ENat
    t : Set M'
    g : M' → M''
    x : M
    hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
    st : Set.MapsTo f s t
    e : PartialEquiv M E := extChartAt I x
    hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I'' (Function.comp g f  …
  -/
  set e' := extChartAt I' (f x)
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    s : Set M
    n : ENat
    t : Set M'
    g : M' → M''
    x : M
    st : Set.MapsTo f s t
    e : PartialEquiv M E := extChartAt I x
    e' : PartialEquiv M' E' := extChartAt I' (f x)
    hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
    hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I'' (Function.comp g f  …
  -/
  have : e' (f x) = (writtenInExtChartAt I I' x f) (e x) := by simp only [e, e', mfld_simps]
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    s : Set M
    n : ENat
    t : Set M'
    g : M' → M''
    x : M
    st : Set.MapsTo f s t
    e : PartialEquiv M E := extChartAt I x
    e' : PartialEquiv M' E' := extChartAt I' (f x)
    hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
    hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
    this : Eq (↑e' (f x)) (writtenInExtChartAt I I' x f (↑e x))
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I'' (Function.comp g f  …
  -/
  rw [this] at hg
  have A : ∀ᶠ y in 𝓝[e.symm ⁻¹' s ∩ range I] e x, f (e.symm y) ∈ t ∧ f (e.symm y) ∈ e'.source := by
    simp only [e, ← map_extChartAt_nhdsWithin, eventually_map]
    filter_upwards [hf.1.tendsto (extChartAt_source_mem_nhds (I := I') (f x)),
      inter_mem_nhdsWithin s (extChartAt_source_mem_nhds (I := I) x)]
    rintro x' (hfx' : f x' ∈ e'.source) ⟨hx's, hx'⟩
    simp only [e, e.map_source hx', true_and, e.left_inv hx', st hx's, *]
  refine ((hg.2.comp _ (hf.2.mono inter_subset_right)
      ((mapsTo_preimage _ _).mono_left inter_subset_left)).mono_of_mem_nhdsWithin
      (inter_mem ?_ self_mem_nhdsWithin)).congr_of_eventuallyEq ?_ ?_
    /-
      case refine_1
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
      E' : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E'
      inst✝⁹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁷ : TopologicalSpace M'
      E'' : Type u_8
      inst✝⁶ : NormedAddCommGroup E''
      inst✝⁵ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝⁴ : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝³ : TopologicalSpace M''
      inst✝² : ChartedSpace H M
      inst✝¹ : ChartedSpace H' M'
      inst✝ : ChartedSpace H'' M''
      f : M → M'
      s : Set M
      n : ENat
      t : Set M'
      g : M' → M''
      x : M
      st : Set.MapsTo f s t
      e : PartialEquiv M E := extChartAt I x
      e' : PartialEquiv M' E' := extChartAt I' (f x)
      hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
      hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
      this : Eq (↑e' (f x)) (writtenInExtChartAt I I' x f (↑e x))
      A : Filter.Eventually (fun y => And (Membership.mem t (f (↑e.symm y))) (Member …
      ⊢ Membership.mem (nhdsWithin (↑e x) (Inter.inter (Set.preimage (↑e.symm) s) (S …
    -/
  · filter_upwards [A]
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
      E' : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E'
      inst✝⁹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁷ : TopologicalSpace M'
      E'' : Type u_8
      inst✝⁶ : NormedAddCommGroup E''
      inst✝⁵ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝⁴ : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝³ : TopologicalSpace M''
      inst✝² : ChartedSpace H M
      inst✝¹ : ChartedSpace H' M'
      inst✝ : ChartedSpace H'' M''
      f : M → M'
      s : Set M
      n : ENat
      t : Set M'
      g : M' → M''
      x : M
      st : Set.MapsTo f s t
      e : PartialEquiv M E := extChartAt I x
      e' : PartialEquiv M' E' := extChartAt I' (f x)
      hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
      hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
      this : Eq (↑e' (f x)) (writtenInExtChartAt I I' x f (↑e x))
      A : Filter.Eventually (fun y => And (Membership.mem t (f (↑e.symm y))) (Member …
      ⊢ ∀ (a : E), And (Membership.mem t (f (↑e.symm a))) (Membership.mem e'.source  …
    -/
    rintro x' ⟨ht, hfx'⟩
    simp only [*, e, e',mem_preimage, writtenInExtChartAt, (· ∘ ·), mem_inter_iff, e'.left_inv,
      true_and]
    /-
      case h.intro
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
      E' : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E'
      inst✝⁹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁷ : TopologicalSpace M'
      E'' : Type u_8
      inst✝⁶ : NormedAddCommGroup E''
      inst✝⁵ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝⁴ : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝³ : TopologicalSpace M''
      inst✝² : ChartedSpace H M
      inst✝¹ : ChartedSpace H' M'
      inst✝ : ChartedSpace H'' M''
      f : M → M'
      s : Set M
      n : ENat
      t : Set M'
      g : M' → M''
      x : M
      st : Set.MapsTo f s t
      e : PartialEquiv M E := extChartAt I x
      e' : PartialEquiv M' E' := extChartAt I' (f x)
      hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
      hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
      this : Eq (↑e' (f x)) (writtenInExtChartAt I I' x f (↑e x))
      A : Filter.Eventually (fun y => And (Membership.mem t (f (↑e.symm y))) (Member …
      x' : E
      ht : Membership.mem t (f (↑e.symm x'))
      hfx' : Membership.mem e'.source (f (↑e.symm x'))
      ⊢ Membership.mem (Set.range ↑I') (↑(extChartAt I' (f x)) (f (↑(extChartAt I x) …
    -/
    exact mem_range_self _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
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
      E' : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E'
      inst✝⁹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁷ : TopologicalSpace M'
      E'' : Type u_8
      inst✝⁶ : NormedAddCommGroup E''
      inst✝⁵ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝⁴ : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝³ : TopologicalSpace M''
      inst✝² : ChartedSpace H M
      inst✝¹ : ChartedSpace H' M'
      inst✝ : ChartedSpace H'' M''
      f : M → M'
      s : Set M
      n : ENat
      t : Set M'
      g : M' → M''
      x : M
      st : Set.MapsTo f s t
      e : PartialEquiv M E := extChartAt I x
      e' : PartialEquiv M' E' := extChartAt I' (f x)
      hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
      hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
      this : Eq (↑e' (f x)) (writtenInExtChartAt I I' x f (↑e x))
      A : Filter.Eventually (fun y => And (Membership.mem t (f (↑e.symm y))) (Member …
      ⊢ (nhdsWithin (↑e x) (Inter.inter (Set.preimage (↑e.symm) s) (Set.range ↑I))). …
    -/
  · filter_upwards [A]
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
      E' : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E'
      inst✝⁹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁷ : TopologicalSpace M'
      E'' : Type u_8
      inst✝⁶ : NormedAddCommGroup E''
      inst✝⁵ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝⁴ : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝³ : TopologicalSpace M''
      inst✝² : ChartedSpace H M
      inst✝¹ : ChartedSpace H' M'
      inst✝ : ChartedSpace H'' M''
      f : M → M'
      s : Set M
      n : ENat
      t : Set M'
      g : M' → M''
      x : M
      st : Set.MapsTo f s t
      e : PartialEquiv M E := extChartAt I x
      e' : PartialEquiv M' E' := extChartAt I' (f x)
      hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
      hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
      this : Eq (↑e' (f x)) (writtenInExtChartAt I I' x f (↑e x))
      A : Filter.Eventually (fun y => And (Membership.mem t (f (↑e.symm y))) (Member …
      ⊢ ∀ (a : E), And (Membership.mem t (f (↑e.symm a))) (Membership.mem e'.source  …
    -/
    rintro x' ⟨-, hfx'⟩
    /-
      case h.intro
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
      E' : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E'
      inst✝⁹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝⁷ : TopologicalSpace M'
      E'' : Type u_8
      inst✝⁶ : NormedAddCommGroup E''
      inst✝⁵ : NormedSpace 𝕜 E''
      H'' : Type u_9
      inst✝⁴ : TopologicalSpace H''
      I'' : ModelWithCorners 𝕜 E'' H''
      M'' : Type u_10
      inst✝³ : TopologicalSpace M''
      inst✝² : ChartedSpace H M
      inst✝¹ : ChartedSpace H' M'
      inst✝ : ChartedSpace H'' M''
      f : M → M'
      s : Set M
      n : ENat
      t : Set M'
      g : M' → M''
      x : M
      st : Set.MapsTo f s t
      e : PartialEquiv M E := extChartAt I x
      e' : PartialEquiv M' E' := extChartAt I' (f x)
      hg : And (ContinuousWithinAt g t (f x)) (ContDiffWithinAt 𝕜 (↑n) (Function.com …
      hf : And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑ …
      this : Eq (↑e' (f x)) (writtenInExtChartAt I I' x f (↑e x))
      A : Filter.Eventually (fun y => And (Membership.mem t (f (↑e.symm y))) (Member …
      x' : E
      hfx' : Membership.mem e'.source (f (↑e.symm x'))
      ⊢ Eq (Function.comp (↑(extChartAt I'' (Function.comp g f x))) (Function.comp ( …
    -/
    simp only [*, e, e', (· ∘ ·), writtenInExtChartAt, e'.left_inv]
    /-
      🎉 no goals
    -/
  · simp only [e, e', writtenInExtChartAt, (· ∘ ·), mem_extChartAt_source,
      e.left_inv, e'.left_inv]


/-- See note [comp_of_eq lemmas] -/
theorem ContMDiffWithinAt.comp_of_eq {t : Set M'} {g : M' → M''} {x : M} {y : M'}
    (hg : ContMDiffWithinAt I' I'' n g t y) (hf : ContMDiffWithinAt I I' n f s x)
    (st : MapsTo f s t) (hx : f x = y) : ContMDiffWithinAt I I'' n (g ∘ f) s x := by
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    s : Set M
    n : ENat
    t : Set M'
    g : M' → M''
    x : M
    y : M'
    hg : ContMDiffWithinAt I' I'' n g t y
    hf : ContMDiffWithinAt I I' n f s x
    st : Set.MapsTo f s t
    hx : Eq (f x) y
    ⊢ ContMDiffWithinAt I I'' n (Function.comp g f) s x
  -/
  subst hx; exact hg.comp x hf st
            /-
              🎉 no goals
            -/


@[deprecated (since := "2024-11-20")] alias SmoothWithinAt.comp := ContMDiffWithinAt.comp


/-- The composition of `C^n` functions on domains is `C^n`. -/
theorem ContMDiffOn.comp {t : Set M'} {g : M' → M''} (hg : ContMDiffOn I' I'' n g t)
    (hf : ContMDiffOn I I' n f s) (st : s ⊆ f ⁻¹' t) : ContMDiffOn I I'' n (g ∘ f) s := fun x hx =>
  (hg _ (st hx)).comp x (hf x hx) st


@[deprecated (since := "2024-11-20")] alias SmoothOn.comp := ContMDiffOn.comp


/-- The composition of `C^n` functions on domains is `C^n`. -/
theorem ContMDiffOn.comp' {t : Set M'} {g : M' → M''} (hg : ContMDiffOn I' I'' n g t)
    (hf : ContMDiffOn I I' n f s) : ContMDiffOn I I'' n (g ∘ f) (s ∩ f ⁻¹' t) :=
  hg.comp (hf.mono inter_subset_left) inter_subset_right


@[deprecated (since := "2024-11-20")] alias SmoothOn.comp' := ContMDiffOn.comp'


/-- The composition of `C^n` functions is `C^n`. -/
theorem ContMDiff.comp {g : M' → M''} (hg : ContMDiff I' I'' n g) (hf : ContMDiff I I' n f) :
    ContMDiff I I'' n (g ∘ f) := by
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    n : ENat
    g : M' → M''
    hg : ContMDiff I' I'' n g
    hf : ContMDiff I I' n f
    ⊢ ContMDiff I I'' n (Function.comp g f)
  -/
  rw [← contMDiffOn_univ] at hf hg ⊢
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    n : ENat
    g : M' → M''
    hg : ContMDiffOn I' I'' n g Set.univ
    hf : ContMDiffOn I I' n f Set.univ
    ⊢ ContMDiffOn I I'' n (Function.comp g f) Set.univ
  -/
  exact hg.comp hf subset_preimage_univ
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias Smooth.comp := ContMDiff.comp


/-- The composition of `C^n` functions within domains at points is `C^n`. -/
theorem ContMDiffWithinAt.comp' {t : Set M'} {g : M' → M''} (x : M)
    (hg : ContMDiffWithinAt I' I'' n g t (f x)) (hf : ContMDiffWithinAt I I' n f s x) :
    ContMDiffWithinAt I I'' n (g ∘ f) (s ∩ f ⁻¹' t) x :=
  hg.comp x (hf.mono inter_subset_left) inter_subset_right


@[deprecated (since := "2024-11-20")] alias SmoothWithinAt.comp' := ContMDiffWithinAt.comp'


/-- `g ∘ f` is `C^n` within `s` at `x` if `g` is `C^n` at `f x` and
`f` is `C^n` within `s` at `x`. -/
theorem ContMDiffAt.comp_contMDiffWithinAt {g : M' → M''} (x : M)
    (hg : ContMDiffAt I' I'' n g (f x)) (hf : ContMDiffWithinAt I I' n f s x) :
    ContMDiffWithinAt I I'' n (g ∘ f) s x :=
  hg.comp x hf (mapsTo_univ _ _)


@[deprecated (since := "2024-11-20")]
alias SmoothAt.comp_smoothWithinAt := ContMDiffAt.comp_contMDiffWithinAt


/-- The composition of `C^n` functions at points is `C^n`. -/
nonrec theorem ContMDiffAt.comp {g : M' → M''} (x : M) (hg : ContMDiffAt I' I'' n g (f x))
    (hf : ContMDiffAt I I' n f x) : ContMDiffAt I I'' n (g ∘ f) x :=
  hg.comp x hf (mapsTo_univ _ _)


/-- See note [comp_of_eq lemmas] -/
theorem ContMDiffAt.comp_of_eq {g : M' → M''} {x : M} {y : M'} (hg : ContMDiffAt I' I'' n g y)
    (hf : ContMDiffAt I I' n f x) (hx : f x = y) : ContMDiffAt I I'' n (g ∘ f) x := by
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
    E' : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E'
    inst✝⁹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁷ : TopologicalSpace M'
    E'' : Type u_8
    inst✝⁶ : NormedAddCommGroup E''
    inst✝⁵ : NormedSpace 𝕜 E''
    H'' : Type u_9
    inst✝⁴ : TopologicalSpace H''
    I'' : ModelWithCorners 𝕜 E'' H''
    M'' : Type u_10
    inst✝³ : TopologicalSpace M''
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H'' M''
    f : M → M'
    n : ENat
    g : M' → M''
    x : M
    y : M'
    hg : ContMDiffAt I' I'' n g y
    hf : ContMDiffAt I I' n f x
    hx : Eq (f x) y
    ⊢ ContMDiffAt I I'' n (Function.comp g f) x
  -/
  subst hx; exact hg.comp x hf
            /-
              🎉 no goals
            -/


@[deprecated (since := "2024-11-20")] alias SmoothAt.comp := ContMDiffAt.comp


theorem ContMDiff.comp_contMDiffOn {f : M → M'} {g : M' → M''} {s : Set M}
    (hg : ContMDiff I' I'' n g) (hf : ContMDiffOn I I' n f s) : ContMDiffOn I I'' n (g ∘ f) s :=
  hg.contMDiffOn.comp hf Set.subset_preimage_univ


@[deprecated (since := "2024-11-20")] alias Smooth.comp_smoothOn := ContMDiff.comp_contMDiffOn


theorem ContMDiffOn.comp_contMDiff {t : Set M'} {g : M' → M''} (hg : ContMDiffOn I' I'' n g t)
    (hf : ContMDiff I I' n f) (ht : ∀ x, f x ∈ t) : ContMDiff I I'' n (g ∘ f) :=
  contMDiffOn_univ.mp <| hg.comp hf.contMDiffOn fun x _ => ht x


@[deprecated (since := "2024-11-20")] alias SmoothOn.comp_smooth := ContMDiffOn.comp_contMDiff


theorem contMDiff_id : ContMDiff I I n (id : M → M) :=
  ContMDiff.of_le
    ((contDiffWithinAt_localInvariantProp ⊤).liftProp_id contDiffWithinAtProp_id) le_top


@[deprecated (since := "2024-11-20")] alias smooth_id := contMDiff_id


theorem contMDiffOn_id : ContMDiffOn I I n (id : M → M) s :=
  contMDiff_id.contMDiffOn


@[deprecated (since := "2024-11-20")] alias smoothOn_id := contMDiffOn_id


theorem contMDiffAt_id : ContMDiffAt I I n (id : M → M) x :=
  contMDiff_id.contMDiffAt


@[deprecated (since := "2024-11-20")] alias smoothAt_id := contMDiffAt_id


theorem contMDiffWithinAt_id : ContMDiffWithinAt I I n (id : M → M) s x :=
  contMDiffAt_id.contMDiffWithinAt


@[deprecated (since := "2024-11-20")] alias smoothWithinAt_id := contMDiffWithinAt_id


theorem contMDiff_const : ContMDiff I I' n fun _ : M => c := by
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
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    n : ENat
    c : M'
    ⊢ ContMDiff I I' n fun x => c
  -/
  intro x
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
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    n : ENat
    c : M'
    x : M
    ⊢ ContMDiffAt I I' n (fun x => c) x
  -/
  refine ⟨continuousWithinAt_const, ?_⟩
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
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    n : ENat
    c : M'
    x : M
    ⊢ ContDiffWithinAtProp I I' n (Function.comp (↑(chartAt H' c)) (Function.comp  …
  -/
  simp only [ContDiffWithinAtProp, Function.comp_def]
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
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    n : ENat
    c : M'
    x : M
    ⊢ ContDiffWithinAt 𝕜 (↑n) (fun x => ↑I' (↑(chartAt H' c) c)) (Inter.inter (Set …
  -/
  exact contDiffWithinAt_const
  /-
    🎉 no goals
  -/


@[to_additive]
theorem contMDiff_one [One M'] : ContMDiff I I' n (1 : M → M') := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    n : ENat
    inst✝ : One M'
    ⊢ ContMDiff I I' n 1
  -/
  simp only [Pi.one_def, contMDiff_const]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias smooth_const := contMDiff_const


@[deprecated (since := "2024-11-20")] alias smooth_one := contMDiff_one

@[deprecated (since := "2024-11-20")] alias smooth_zero := contMDiff_zero


theorem contMDiffOn_const : ContMDiffOn I I' n (fun _ : M => c) s :=
  contMDiff_const.contMDiffOn


@[to_additive]
theorem contMDiffOn_one [One M'] : ContMDiffOn I I' n (1 : M → M') s :=
  contMDiff_one.contMDiffOn


@[deprecated (since := "2024-11-20")] alias smoothOn_const := contMDiffOn_const


@[deprecated (since := "2024-11-20")] alias smoothOn_one := contMDiffOn_one

@[deprecated (since := "2024-11-20")] alias smoothOn_zero := contMDiffOn_zero



theorem contMDiffAt_const : ContMDiffAt I I' n (fun _ : M => c) x :=
  contMDiff_const.contMDiffAt


@[to_additive]
theorem contMDiffAt_one [One M'] : ContMDiffAt I I' n (1 : M → M') x :=
  contMDiff_one.contMDiffAt


@[deprecated (since := "2024-11-20")] alias smoothAt_const := contMDiffAt_const


@[deprecated (since := "2024-11-20")] alias smoothAt_one := contMDiffAt_one

@[deprecated (since := "2024-11-20")] alias smoothAt_zero := contMDiffAt_zero


theorem contMDiffWithinAt_const : ContMDiffWithinAt I I' n (fun _ : M => c) s x :=
  contMDiffAt_const.contMDiffWithinAt


@[to_additive]
theorem contMDiffWithinAt_one [One M'] : ContMDiffWithinAt I I' n (1 : M → M') s x :=
  contMDiffAt_const.contMDiffWithinAt


@[deprecated (since := "2024-11-20")] alias smoothWithinAt_const := contMDiffWithinAt_const


@[deprecated (since := "2024-11-20")] alias smoothWithinAt_one := contMDiffWithinAt_one

@[deprecated (since := "2024-11-20")] alias smoothWithinAt_zero := contMDiffWithinAt_zero


/-- `f` is continuously differentiable if it is cont. differentiable at
each `x ∈ mulTSupport f`. -/
@[to_additive "`f` is continuously differentiable if it is continuously
differentiable at each `x ∈ tsupport f`."]
theorem contMDiff_of_mulTSupport [One M'] {f : M → M'}
    (hf : ∀ x ∈ mulTSupport f, ContMDiffAt I I' n f x) : ContMDiff I I' n f := by
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    n : ENat
    inst✝ : One M'
    f : M → M'
    hf : ∀ (x : M), Membership.mem (mulTSupport f) x → ContMDiffAt I I' n f x
    ⊢ ContMDiff I I' n f
  -/
  intro x
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
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H M
    inst✝¹ : ChartedSpace H' M'
    n : ENat
    inst✝ : One M'
    f : M → M'
    hf : ∀ (x : M), Membership.mem (mulTSupport f) x → ContMDiffAt I I' n f x
    x : M
    ⊢ ContMDiffAt I I' n f x
  -/
  by_cases hx : x ∈ mulTSupport f
    /-
      case pos
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
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H M
      inst✝¹ : ChartedSpace H' M'
      n : ENat
      inst✝ : One M'
      f : M → M'
      hf : ∀ (x : M), Membership.mem (mulTSupport f) x → ContMDiffAt I I' n f x
      x : M
      hx : Membership.mem (mulTSupport f) x
      ⊢ ContMDiffAt I I' n f x
    -/
  · exact hf x hx
    /-
      🎉 no goals
    -/
  · exact ContMDiffAt.congr_of_eventuallyEq contMDiffAt_const
      (not_mem_mulTSupport_iff_eventuallyEq.1 hx)


@[to_additive contMDiffWithinAt_of_not_mem]
theorem contMDiffWithinAt_of_not_mem_mulTSupport {f : M → M'} [One M'] {x : M}
    (hx : x ∉ mulTSupport f) (n : ℕ∞) (s : Set M) : ContMDiffWithinAt I I' n f s x := by
  apply contMDiffWithinAt_const.congr_of_eventuallyEq
    (eventually_nhdsWithin_of_eventually_nhds <| not_mem_mulTSupport_iff_eventuallyEq.mp hx)
    (image_eq_one_of_nmem_mulTSupport hx)


/-- `f` is continuously differentiable at each point outside of its `mulTSupport`. -/
@[to_additive contMDiffAt_of_not_mem]
theorem contMDiffAt_of_not_mem_mulTSupport {f : M → M'} [One M'] {x : M}
    (hx : x ∉ mulTSupport f) (n : ℕ∞) : ContMDiffAt I I' n f x :=
  contMDiffWithinAt_of_not_mem_mulTSupport hx n univ



theorem contMDiffAt_subtype_iff {n : ℕ∞} {U : Opens M} {f : M → M'} {x : U} :
    ContMDiffAt I I' n (fun x : U ↦ f x) x ↔ ContMDiffAt I I' n f x :=
  ((contDiffWithinAt_localInvariantProp n).liftPropAt_iff_comp_subtype_val _ _).symm


@[deprecated (since := "2024-11-20")] alias contMdiffAt_subtype_iff := contMDiffAt_subtype_iff


theorem contMDiff_subtype_val {n : ℕ∞} {U : Opens M} : ContMDiff I I n (Subtype.val : U → M) :=
  fun _ ↦ contMDiffAt_subtype_iff.mpr contMDiffAt_id


@[to_additive]
theorem ContMDiff.extend_one [T2Space M] [One M'] {n : ℕ∞} {U : Opens M} {f : U → M'}
    (supp : HasCompactMulSupport f) (diff : ContMDiff I I' n f) :
    ContMDiff I I' n (Subtype.val.extend f 1) := fun x ↦ by
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
    E' : Type u_5
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁵ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁴ : TopologicalSpace M'
    inst✝³ : ChartedSpace H M
    inst✝² : ChartedSpace H' M'
    inst✝¹ : T2Space M
    inst✝ : One M'
    n : ENat
    U : TopologicalSpace.Opens M
    f : (Subtype fun x => Membership.mem U x) → M'
    supp : HasCompactMulSupport f
    diff : ContMDiff I I' n f
    x : M
    ⊢ ContMDiffAt I I' n (Function.extend Subtype.val f 1) x
  -/
  refine contMDiff_of_mulTSupport (fun x h ↦ ?_) _
  lift x to U using Subtype.coe_image_subset _ _
    (supp.mulTSupport_extend_one_subset continuous_subtype_val h)
  /-
    case intro
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
    E' : Type u_5
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁵ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁴ : TopologicalSpace M'
    inst✝³ : ChartedSpace H M
    inst✝² : ChartedSpace H' M'
    inst✝¹ : T2Space M
    inst✝ : One M'
    n : ENat
    U : TopologicalSpace.Opens M
    f : (Subtype fun x => Membership.mem U x) → M'
    supp : HasCompactMulSupport f
    diff : ContMDiff I I' n f
    x✝ : M
    x : Subtype fun x => Membership.mem U x
    h : Membership.mem (mulTSupport (Function.extend Subtype.val f 1)) ↑x
    ⊢ ContMDiffAt I I' n (Function.extend Subtype.val f 1) ↑x
  -/
  rw [← contMDiffAt_subtype_iff]
  /-
    case intro
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
    E' : Type u_5
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁵ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁴ : TopologicalSpace M'
    inst✝³ : ChartedSpace H M
    inst✝² : ChartedSpace H' M'
    inst✝¹ : T2Space M
    inst✝ : One M'
    n : ENat
    U : TopologicalSpace.Opens M
    f : (Subtype fun x => Membership.mem U x) → M'
    supp : HasCompactMulSupport f
    diff : ContMDiff I I' n f
    x✝ : M
    x : Subtype fun x => Membership.mem U x
    h : Membership.mem (mulTSupport (Function.extend Subtype.val f 1)) ↑x
    ⊢ ContMDiffAt I I' n (fun x => Function.extend Subtype.val f 1 ↑x) x
  -/
  simp_rw [← comp_def]
  /-
    case intro
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
    E' : Type u_5
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁵ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁴ : TopologicalSpace M'
    inst✝³ : ChartedSpace H M
    inst✝² : ChartedSpace H' M'
    inst✝¹ : T2Space M
    inst✝ : One M'
    n : ENat
    U : TopologicalSpace.Opens M
    f : (Subtype fun x => Membership.mem U x) → M'
    supp : HasCompactMulSupport f
    diff : ContMDiff I I' n f
    x✝ : M
    x : Subtype fun x => Membership.mem U x
    h : Membership.mem (mulTSupport (Function.extend Subtype.val f 1)) ↑x
    ⊢ ContMDiffAt I I' n (Function.comp (Function.extend Subtype.val f 1) Subtype. …
  -/
  rw [extend_comp Subtype.val_injective]
  /-
    case intro
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
    E' : Type u_5
    inst✝⁷ : NormedAddCommGroup E'
    inst✝⁶ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁵ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝⁴ : TopologicalSpace M'
    inst✝³ : ChartedSpace H M
    inst✝² : ChartedSpace H' M'
    inst✝¹ : T2Space M
    inst✝ : One M'
    n : ENat
    U : TopologicalSpace.Opens M
    f : (Subtype fun x => Membership.mem U x) → M'
    supp : HasCompactMulSupport f
    diff : ContMDiff I I' n f
    x✝ : M
    x : Subtype fun x => Membership.mem U x
    h : Membership.mem (mulTSupport (Function.extend Subtype.val f 1)) ↑x
    ⊢ ContMDiffAt I I' n f x
  -/
  exact diff.contMDiffAt
  /-
    🎉 no goals
  -/


theorem contMDiff_inclusion {n : ℕ∞} {U V : Opens M} (h : U ≤ V) :
    ContMDiff I I n (Opens.inclusion h : U → V) := by
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
    n : ENat
    U V : TopologicalSpace.Opens M
    h : LE.le U V
    ⊢ ContMDiff I I n (TopologicalSpace.Opens.inclusion h)
  -/
  rintro ⟨x, hx : x ∈ U⟩
  /-
    case mk
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
    n : ENat
    U V : TopologicalSpace.Opens M
    h : LE.le U V
    x : M
    hx : Membership.mem U x
    ⊢ ContMDiffAt I I n (TopologicalSpace.Opens.inclusion h) ⟨x, hx⟩
  -/
  apply (contDiffWithinAt_localInvariantProp n).liftProp_inclusion
  /-
    case mk.hQ
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
    n : ENat
    U V : TopologicalSpace.Opens M
    h : LE.le U V
    x : M
    hx : Membership.mem U x
    ⊢ ∀ (y : H), ContDiffWithinAtProp I I n id Set.univ y
  -/
  intro y
  /-
    case mk.hQ
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
    n : ENat
    U V : TopologicalSpace.Opens M
    h : LE.le U V
    x : M
    hx : Membership.mem U x
    y : H
    ⊢ ContDiffWithinAtProp I I n id Set.univ y
  -/
  dsimp only [ContDiffWithinAtProp, id_comp, preimage_univ]
  /-
    case mk.hQ
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
    n : ENat
    U V : TopologicalSpace.Opens M
    h : LE.le U V
    x : M
    hx : Membership.mem U x
    y : H
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp ↑I ↑I.symm) (Inter.inter Set.univ (Se …
  -/
  rw [Set.univ_inter]
  /-
    case mk.hQ
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
    n : ENat
    U V : TopologicalSpace.Opens M
    h : LE.le U V
    x : M
    hx : Membership.mem U x
    y : H
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp ↑I ↑I.symm) (Set.range ↑I) (↑I y)
  -/
  exact contDiffWithinAt_id.congr I.rightInvOn (congr_arg I (I.left_inv y))
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias smooth_subtype_iff := contMDiffAt_subtype_iff


@[deprecated (since := "2024-11-20")] alias smooth_subtype_val := contMDiff_subtype_val


@[deprecated (since := "2024-11-20")] alias Smooth.extend_one := ContMDiff.extend_one

@[deprecated (since := "2024-11-20")] alias Smooth.extend_zero := ContMDiff.extend_zero


@[deprecated (since := "2024-11-20")] alias smooth_inclusion := contMDiff_inclusion


/-- If the `ChartedSpace` structure on a manifold `M` is given by an open embedding `e : M → H`,
then `e` is smooth. -/
lemma contMDiff_isOpenEmbedding [Nonempty M] :
    haveI := h.singletonChartedSpace; ContMDiff I I n e := by
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    ⊢ ContMDiff I I n e
  -/
  haveI := h.singleton_smoothManifoldWithCorners (I := I)
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    ⊢ ContMDiff I I n e
  -/
  rw [@contMDiff_iff _ _ _ _ _ _ _ _ _ _ h.singletonChartedSpace]
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    ⊢ And (Continuous e) (∀ (x : M) (y : H), ContDiffOn 𝕜 (↑n) (Function.comp (↑(e …
  -/
  use h.continuous
  /-
    case right
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    ⊢ ∀ (x : M) (y : H), ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I y)) (Fun …
  -/
  intros x y
  -- show the function is actually the identity on the range of I ∘ e
  /-
    case right
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    x : M
    y : H
    ⊢ ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I y)) (Function.comp e ↑(extC …
  -/
  apply contDiffOn_id.congr
  /-
    case right
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    x : M
    y : H
    ⊢ ∀ (x_1 : E), Membership.mem (Inter.inter (extChartAt I x).target (Set.preima …
  -/
  intros z hz
  -- factorise into the chart `e` and the model `id`
  /-
    case right
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    x : M
    y : H
    z : E
    hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
    ⊢ Eq (Function.comp (↑(extChartAt I y)) (Function.comp e ↑(extChartAt I x).sym …
  -/
  simp only [mfld_simps]
  /-
    case right
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    x : M
    y : H
    z : E
    hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
    ⊢ Eq (↑I (e (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e h).symm (↑I.symm …
  -/
  rw [h.toPartialHomeomorph_right_inv]
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      x : M
      y : H
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      ⊢ Eq (↑I (↑I.symm z)) z
    -/
  · rw [I.right_inv]
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      x : M
      y : H
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      ⊢ Membership.mem (Set.range ↑I) z
    -/
    apply mem_of_subset_of_mem _ hz.1
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      x : M
      y : H
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      ⊢ HasSubset.Subset (extChartAt I x).target (Set.range ↑I)
    -/
    exact haveI := h.singletonChartedSpace; extChartAt_target_subset_range (I := I) x
    /-
      🎉 no goals
    -/
  · -- `hz` implies that `z ∈ range (I ∘ e)`
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      x : M
      y : H
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      ⊢ Membership.mem (Set.range e) (↑I.symm z)
    -/
    have := hz.1
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this✝ : SmoothManifoldWithCorners I M
      x : M
      y : H
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      this : Membership.mem (extChartAt I x).target z
      ⊢ Membership.mem (Set.range e) (↑I.symm z)
    -/
    rw [@extChartAt_target _ _ _ _ _ _ _ _ _ _ h.singletonChartedSpace] at this
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this✝ : SmoothManifoldWithCorners I M
      x : M
      y : H
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      this : Membership.mem (Inter.inter (Set.preimage (↑I.symm) (chartAt H x).targe …
      ⊢ Membership.mem (Set.range e) (↑I.symm z)
    -/
    have := this.1
    rw [mem_preimage, PartialHomeomorph.singletonChartedSpace_chartAt_eq,
      h.toPartialHomeomorph_target] at this
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this✝¹ : SmoothManifoldWithCorners I M
      x : M
      y : H
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      this✝ : Membership.mem (Inter.inter (Set.preimage (↑I.symm) (chartAt H x).targ …
      this : Membership.mem (Set.range e) (↑I.symm z)
      ⊢ Membership.mem (Set.range e) (↑I.symm z)
    -/
    exact this
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-18")]
alias contMDiff_openEmbedding := contMDiff_isOpenEmbedding


/-- If the `ChartedSpace` structure on a manifold `M` is given by an open embedding `e : M → H`,
then the inverse of `e` is smooth. -/
lemma contMDiffOn_isOpenEmbedding_symm [Nonempty M] :
    haveI := h.singletonChartedSpace; ContMDiffOn I I
      n (IsOpenEmbedding.toPartialHomeomorph e h).symm (range e) := by
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    ⊢ ContMDiffOn I I n (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e h).symm) …
  -/
  haveI := h.singleton_smoothManifoldWithCorners (I := I)
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    ⊢ ContMDiffOn I I n (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e h).symm) …
  -/
  rw [@contMDiffOn_iff]
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
    e : M → H
    h : Topology.IsOpenEmbedding e
    n : WithTop Nat
    inst✝ : Nonempty M
    this : SmoothManifoldWithCorners I M
    ⊢ And (ContinuousOn (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e h).symm) …
  -/
  constructor
    /-
      case left
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      ⊢ ContinuousOn (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e h).symm) (Set …
    -/
  · rw [← h.toPartialHomeomorph_target]
    /-
      case left
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      ⊢ ContinuousOn (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e h).symm) (Top …
    -/
    exact (h.toPartialHomeomorph e).continuousOn_symm
    /-
      🎉 no goals
    -/
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      ⊢ ∀ (x : H) (y : M), ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I y)) (Fun …
    -/
  · intros z hz
    -- show the function is actually the identity on the range of I ∘ e
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      z : H
      hz : M
      ⊢ ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I hz)) (Function.comp ↑(Topol …
    -/
    apply contDiffOn_id.congr
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      z : H
      hz : M
      ⊢ ∀ (x : E), Membership.mem (Inter.inter (extChartAt I z).target (Set.preimage …
    -/
    intros z hz
    -- factorise into the chart `e` and the model `id`
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this : SmoothManifoldWithCorners I M
      z✝ : H
      hz✝ : M
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I z✝).target (Set.preimage (↑(ext …
      ⊢ Eq (Function.comp (↑(extChartAt I hz✝)) (Function.comp ↑(Topology.IsOpenEmbe …
    -/
    simp only [mfld_simps]
    have : I.symm z ∈ range e := by
      rw [ModelWithCorners.symm, ← mem_preimage]
      exact hz.2.1
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this✝ : SmoothManifoldWithCorners I M
      z✝ : H
      hz✝ : M
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I z✝).target (Set.preimage (↑(ext …
      this : Membership.mem (Set.range e) (↑I.symm z)
      ⊢ Eq (↑I (e (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e h).symm (↑I.symm …
    -/
    rw [h.toPartialHomeomorph_right_inv e this]
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this✝ : SmoothManifoldWithCorners I M
      z✝ : H
      hz✝ : M
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I z✝).target (Set.preimage (↑(ext …
      this : Membership.mem (Set.range e) (↑I.symm z)
      ⊢ Eq (↑I (↑I.symm z)) z
    -/
    apply I.right_inv
    /-
      case right
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
      e : M → H
      h : Topology.IsOpenEmbedding e
      n : WithTop Nat
      inst✝ : Nonempty M
      this✝ : SmoothManifoldWithCorners I M
      z✝ : H
      hz✝ : M
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I z✝).target (Set.preimage (↑(ext …
      this : Membership.mem (Set.range e) (↑I.symm z)
      ⊢ Membership.mem (Set.range ↑I) z
    -/
    exact mem_of_subset_of_mem (extChartAt_target_subset_range _) hz.1
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-18")]
alias contMDiffOn_openEmbedding_symm := contMDiffOn_isOpenEmbedding_symm


/-- Let `M'` be a manifold whose chart structure is given by an open embedding `e'` into its model
space `H'`. Then the smoothness of `e' ∘ f : M → H'` implies the smoothness of `f`.

This is useful, for example, when `e' ∘ f = g ∘ e` for smooth maps `e : M → X` and `g : X → H'`. -/
lemma ContMDiff.of_comp_isOpenEmbedding {f : M → M'} (hf : ContMDiff I I' n (e' ∘ f)) :
    haveI := h'.singletonChartedSpace; ContMDiff I I' n f := by
  have : f = (h'.toPartialHomeomorph e').symm ∘ e' ∘ f := by
    ext
    rw [Function.comp_apply, Function.comp_apply, IsOpenEmbedding.toPartialHomeomorph_left_inv]
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
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    n : WithTop Nat
    inst✝¹ : ChartedSpace H M
    inst✝ : Nonempty M'
    e' : M' → H'
    h' : Topology.IsOpenEmbedding e'
    f : M → M'
    hf : ContMDiff I I' n (Function.comp e' f)
    this : Eq f (Function.comp (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e'  …
    ⊢ ContMDiff I I' n f
  -/
  rw [this]
  apply @ContMDiffOn.comp_contMDiff _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    h'.singletonChartedSpace _ _ (range e') _ (contMDiffOn_isOpenEmbedding_symm h') hf
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
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    n : WithTop Nat
    inst✝¹ : ChartedSpace H M
    inst✝ : Nonempty M'
    e' : M' → H'
    h' : Topology.IsOpenEmbedding e'
    f : M → M'
    hf : ContMDiff I I' n (Function.comp e' f)
    this : Eq f (Function.comp (↑(Topology.IsOpenEmbedding.toPartialHomeomorph e'  …
    ⊢ ∀ (x : M), Membership.mem (Set.range e') (Function.comp e' f x)
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias ContMDiff.of_comp_openEmbedding := ContMDiff.of_comp_isOpenEmbedding


