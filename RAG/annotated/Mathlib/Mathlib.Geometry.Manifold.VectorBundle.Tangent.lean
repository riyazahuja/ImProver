/-- Auxiliary lemma for tangent spaces: the derivative of a coordinate change between two charts is
  smooth on its source. -/
theorem contDiffOn_fderiv_coord_change (i j : atlas H M) :
    ContDiffOn 𝕜 ∞ (fderivWithin 𝕜 (j.1.extend I ∘ (i.1.extend I).symm) (range I))
      ((i.1.extend I).symm ≫ j.1.extend I).source := by
  have h : ((i.1.extend I).symm ≫ j.1.extend I).source ⊆ range I := by
    rw [i.1.extend_coord_change_source]; apply image_subset_range
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    i j : ↑(atlas H M)
    h : HasSubset.Subset (((↑i).extend I).symm.trans ((↑j).extend I)).source (Set. …
    ⊢ ContDiffOn 𝕜 (↑Top.top) (fderivWithin 𝕜 (Function.comp ↑((↑j).extend I) ↑((↑ …
  -/
  intro x hx
  refine (ContDiffWithinAt.fderivWithin_right ?_ I.uniqueDiffOn (n := ∞) (mod_cast le_top)
    <| h hx).mono h
  refine (PartialHomeomorph.contDiffOn_extend_coord_change (subset_maximalAtlas j.2)
    (subset_maximalAtlas i.2) x hx).mono_of_mem_nhdsWithin ?_
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    i j : ↑(atlas H M)
    h : HasSubset.Subset (((↑i).extend I).symm.trans ((↑j).extend I)).source (Set. …
    x : E
    hx : Membership.mem (((↑i).extend I).symm.trans ((↑j).extend I)).source x
    ⊢ Membership.mem (nhdsWithin x (Set.range ↑I)) (((↑i).extend I).symm.trans ((↑ …
  -/
  exact i.1.extend_coord_change_source_mem_nhdsWithin j.1 hx
  /-
    🎉 no goals
  -/


variable (I M) in
/-- Let `M` be a smooth manifold with corners with model `I` on `(E, H)`.
Then `tangentBundleCore I M` is the vector bundle core for the tangent bundle over `M`.
It is indexed by the atlas of `M`, with fiber `E` and its change of coordinates from the chart `i`
to the chart `j` at point `x : M` is the derivative of the composite
```
  I.symm   i.symm    j     I
E -----> H -----> M --> H --> E
```
within the set `range I ⊆ E` at `I (i x) : E`. -/
@[simps indexAt coordChange]
def tangentBundleCore : VectorBundleCore 𝕜 M E (atlas H M) where
  baseSet i := i.1.source
  isOpen_baseSet i := i.1.open_source
  indexAt := achart H
  mem_baseSet_at := mem_chart_source H
  coordChange i j x :=
    fderivWithin 𝕜 (j.1.extend I ∘ (i.1.extend I).symm) (range I) (i.1.extend I x)
  coordChange_self i x hx v := by
    /-
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedSpace 𝕜 E'
      H : Type u_4
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_5
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_6
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      inst✝⁵ : SmoothManifoldWithCorners I M
      M' : Type u_7
      inst✝⁴ : TopologicalSpace M'
      inst✝³ : ChartedSpace H' M'
      inst✝² : SmoothManifoldWithCorners I' M'
      F : Type u_8
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      i : ↑(atlas H M)
      x : M
      hx : Membership.mem ((fun i => (↑i).source) i) x
      v : E
      ⊢ Eq (((fun i j x => fderivWithin 𝕜 (Function.comp ↑((↑j).extend I) ↑((↑i).ext …
    -/
    simp only
    /-
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedSpace 𝕜 E'
      H : Type u_4
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_5
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_6
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      inst✝⁵ : SmoothManifoldWithCorners I M
      M' : Type u_7
      inst✝⁴ : TopologicalSpace M'
      inst✝³ : ChartedSpace H' M'
      inst✝² : SmoothManifoldWithCorners I' M'
      F : Type u_8
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      i : ↑(atlas H M)
      x : M
      hx : Membership.mem ((fun i => (↑i).source) i) x
      v : E
      ⊢ Eq ((fderivWithin 𝕜 (Function.comp ↑((↑i).extend I) ↑((↑i).extend I).symm) ( …
    -/
    rw [Filter.EventuallyEq.fderivWithin_eq, fderivWithin_id', ContinuousLinearMap.id_apply]
      /-
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i : ↑(atlas H M)
        x : M
        hx : Membership.mem ((fun i => (↑i).source) i) x
        v : E
        ⊢ UniqueDiffWithinAt 𝕜 (Set.range ↑I) (↑((↑i).extend I) x)
      -/
    · exact I.uniqueDiffWithinAt_image
      /-
        🎉 no goals
      -/
      /-
        case hs
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i : ↑(atlas H M)
        x : M
        hx : Membership.mem ((fun i => (↑i).source) i) x
        v : E
        ⊢ (nhdsWithin (↑((↑i).extend I) x) (Set.range ↑I)).EventuallyEq (Function.comp …
      -/
    · filter_upwards [i.1.extend_target_mem_nhdsWithin hx] with y hy
      /-
        case h
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i : ↑(atlas H M)
        x : M
        hx : Membership.mem ((fun i => (↑i).source) i) x
        v y : E
        hy : Membership.mem ((↑i).extend I).target y
        ⊢ Eq (Function.comp (↑((↑i).extend I)) (↑((↑i).extend I).symm) y) y
      -/
      exact (i.1.extend I).right_inv hy
      /-
        🎉 no goals
      -/
      /-
        case hx
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i : ↑(atlas H M)
        x : M
        hx : Membership.mem ((fun i => (↑i).source) i) x
        v : E
        ⊢ Eq (Function.comp (↑((↑i).extend I)) (↑((↑i).extend I).symm) (↑((↑i).extend  …
      -/
    · simp_rw [Function.comp_apply, i.1.extend_left_inv hx]
      /-
        🎉 no goals
      -/
  continuousOn_coordChange i j := by
    refine (contDiffOn_fderiv_coord_change i j).continuousOn.comp
      (i.1.continuousOn_extend.mono ?_) ?_
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j : ↑(atlas H M)
        ⊢ HasSubset.Subset (Inter.inter ((fun i => (↑i).source) i) ((fun i => (↑i).sou …
      -/
    · rw [i.1.extend_source]; exact inter_subset_left
                              /-
                                🎉 no goals
                              -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedSpace 𝕜 E'
      H : Type u_4
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_5
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_6
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      inst✝⁵ : SmoothManifoldWithCorners I M
      M' : Type u_7
      inst✝⁴ : TopologicalSpace M'
      inst✝³ : ChartedSpace H' M'
      inst✝² : SmoothManifoldWithCorners I' M'
      F : Type u_8
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      i j : ↑(atlas H M)
      ⊢ Set.MapsTo (↑((↑i).extend I)) (Inter.inter ((fun i => (↑i).source) i) ((fun  …
    -/
    simp_rw [← i.1.extend_image_source_inter, mapsTo_image]
    /-
      🎉 no goals
    -/
  coordChange_comp := by
    /-
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedSpace 𝕜 E'
      H : Type u_4
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_5
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_6
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      inst✝⁵ : SmoothManifoldWithCorners I M
      M' : Type u_7
      inst✝⁴ : TopologicalSpace M'
      inst✝³ : ChartedSpace H' M'
      inst✝² : SmoothManifoldWithCorners I' M'
      F : Type u_8
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      ⊢ ∀ (i j k : ↑(atlas H M)) (x : M), Membership.mem (Inter.inter (Inter.inter ( …
    -/
    rintro i j k x ⟨⟨hxi, hxj⟩, hxk⟩ v
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedSpace 𝕜 E'
      H : Type u_4
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_5
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_6
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      inst✝⁵ : SmoothManifoldWithCorners I M
      M' : Type u_7
      inst✝⁴ : TopologicalSpace M'
      inst✝³ : ChartedSpace H' M'
      inst✝² : SmoothManifoldWithCorners I' M'
      F : Type u_8
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      i j k : ↑(atlas H M)
      x : M
      hxk : Membership.mem ((fun i => (↑i).source) k) x
      hxi : Membership.mem ((fun i => (↑i).source) i) x
      hxj : Membership.mem ((fun i => (↑i).source) j) x
      v : E
      ⊢ Eq (((fun i j x => fderivWithin 𝕜 (Function.comp ↑((↑j).extend I) ↑((↑i).ext …
    -/
    rw [fderivWithin_fderivWithin, Filter.EventuallyEq.fderivWithin_eq]
      /-
        case intro.intro.hs
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j k : ↑(atlas H M)
        x : M
        hxk : Membership.mem ((fun i => (↑i).source) k) x
        hxi : Membership.mem ((fun i => (↑i).source) i) x
        hxj : Membership.mem ((fun i => (↑i).source) j) x
        v : E
        ⊢ (nhdsWithin (↑((↑i).extend I) x) (Set.range ↑I)).EventuallyEq (Function.comp …
      -/
    · have := i.1.extend_preimage_mem_nhds (I := I) hxi (j.1.extend_source_mem_nhds (I := I) hxj)
      /-
        case intro.intro.hs
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j k : ↑(atlas H M)
        x : M
        hxk : Membership.mem ((fun i => (↑i).source) k) x
        hxi : Membership.mem ((fun i => (↑i).source) i) x
        hxj : Membership.mem ((fun i => (↑i).source) j) x
        v : E
        this : Membership.mem (nhds (↑((↑i).extend I) x)) (Set.preimage (↑((↑i).extend …
        ⊢ (nhdsWithin (↑((↑i).extend I) x) (Set.range ↑I)).EventuallyEq (Function.comp …
      -/
      filter_upwards [nhdsWithin_le_nhds this] with y hy
      /-
        case h
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j k : ↑(atlas H M)
        x : M
        hxk : Membership.mem ((fun i => (↑i).source) k) x
        hxi : Membership.mem ((fun i => (↑i).source) i) x
        hxj : Membership.mem ((fun i => (↑i).source) j) x
        v : E
        this : Membership.mem (nhds (↑((↑i).extend I) x)) (Set.preimage (↑((↑i).extend …
        y : E
        hy : Membership.mem (Set.preimage (↑((↑i).extend I).symm) ((↑j).extend I).sour …
        ⊢ Eq (Function.comp (Function.comp ↑((↑k).extend I) ↑((↑j).extend I).symm) (Fu …
      -/
      simp_rw [Function.comp_apply, (j.1.extend I).left_inv hy]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.hx
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j k : ↑(atlas H M)
        x : M
        hxk : Membership.mem ((fun i => (↑i).source) k) x
        hxi : Membership.mem ((fun i => (↑i).source) i) x
        hxj : Membership.mem ((fun i => (↑i).source) j) x
        v : E
        ⊢ Eq (Function.comp (Function.comp ↑((↑k).extend I) ↑((↑j).extend I).symm) (Fu …
      -/
    · simp_rw [Function.comp_apply, i.1.extend_left_inv hxi, j.1.extend_left_inv hxj]
      /-
        🎉 no goals
      -/
    · exact (contDiffWithinAt_extend_coord_change' (subset_maximalAtlas k.2)
        (subset_maximalAtlas j.2) hxk hxj).differentiableWithinAt (mod_cast le_top)
    · exact (contDiffWithinAt_extend_coord_change' (subset_maximalAtlas j.2)
        (subset_maximalAtlas i.2) hxj hxi).differentiableWithinAt (mod_cast le_top)
      /-
        case intro.intro.h
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j k : ↑(atlas H M)
        x : M
        hxk : Membership.mem ((fun i => (↑i).source) k) x
        hxi : Membership.mem ((fun i => (↑i).source) i) x
        hxj : Membership.mem ((fun i => (↑i).source) j) x
        v : E
        ⊢ Set.MapsTo (Function.comp ↑((↑j).extend I) ↑((↑i).extend I).symm) (Set.range …
      -/
    · intro x _; exact mem_range_self _
                 /-
                   🎉 no goals
                 -/
      /-
        case intro.intro.hxs
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j k : ↑(atlas H M)
        x : M
        hxk : Membership.mem ((fun i => (↑i).source) k) x
        hxi : Membership.mem ((fun i => (↑i).source) i) x
        hxj : Membership.mem ((fun i => (↑i).source) j) x
        v : E
        ⊢ UniqueDiffWithinAt 𝕜 (Set.range ↑I) (↑((↑i).extend I) x)
      -/
    · exact I.uniqueDiffWithinAt_image
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.hy
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j k : ↑(atlas H M)
        x : M
        hxk : Membership.mem ((fun i => (↑i).source) k) x
        hxi : Membership.mem ((fun i => (↑i).source) i) x
        hxj : Membership.mem ((fun i => (↑i).source) j) x
        v : E
        ⊢ Eq (Function.comp (↑((↑j).extend I)) (↑((↑i).extend I).symm) (↑((↑i).extend  …
      -/
    · rw [Function.comp_apply, i.1.extend_left_inv hxi]
      /-
        🎉 no goals
      -/

-- Porting note: moved to a separate `simp high` lemma b/c `simp` can simplify the LHS

@[simp high]
theorem tangentBundleCore_baseSet (i) : (tangentBundleCore I M).baseSet i = i.1.source := rfl


theorem tangentBundleCore_coordChange_achart (x x' z : M) :
    (tangentBundleCore I M).coordChange (achart H x) (achart H x') z =
      fderivWithin 𝕜 (extChartAt I x' ∘ (extChartAt I x).symm) (range I) (extChartAt I x z) :=
  rfl


variable (I) in
/-- In a manifold `M`, given two preferred charts indexed by `x y : M`, `tangentCoordChange I x y`
is the family of derivatives of the corresponding change-of-coordinates map. It takes junk values
outside the intersection of the sources of the two charts.

Note that this definition takes advantage of the fact that `tangentBundleCore` has the same base
sets as the preferred charts of the base manifold. -/
abbrev tangentCoordChange (x y : M) : M → E →L[𝕜] E :=
  (tangentBundleCore I M).coordChange (achart H x) (achart H y)


lemma tangentCoordChange_def {x y z : M} : tangentCoordChange I x y z =
    fderivWithin 𝕜 (extChartAt I y ∘ (extChartAt I x).symm) (range I) (extChartAt I x z) := rfl


lemma tangentCoordChange_self {x z : M} {v : E} (h : z ∈ (extChartAt I x).source) :
    tangentCoordChange I x x z v = v := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    x z : M
    v : E
    h : Membership.mem (extChartAt I x).source z
    ⊢ Eq ((tangentCoordChange I x x z) v) v
  -/
  apply (tangentBundleCore I M).coordChange_self
  /-
    case a
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    x z : M
    v : E
    h : Membership.mem (extChartAt I x).source z
    ⊢ Membership.mem ((tangentBundleCore I M).baseSet (achart H x)) z
  -/
  rw [tangentBundleCore_baseSet, coe_achart, ← extChartAt_source I]
  /-
    case a
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    x z : M
    v : E
    h : Membership.mem (extChartAt I x).source z
    ⊢ Membership.mem (extChartAt I x).source z
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma tangentCoordChange_comp {w x y z : M} {v : E}
    (h : z ∈ (extChartAt I w).source ∩ (extChartAt I x).source ∩ (extChartAt I y).source) :
    tangentCoordChange I x y z (tangentCoordChange I w x z v) = tangentCoordChange I w y z v := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    w x y z : M
    v : E
    h : Membership.mem (Inter.inter (Inter.inter (extChartAt I w).source (extChart …
    ⊢ Eq ((tangentCoordChange I x y z) ((tangentCoordChange I w x z) v)) ((tangent …
  -/
  apply (tangentBundleCore I M).coordChange_comp
  /-
    case a
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    w x y z : M
    v : E
    h : Membership.mem (Inter.inter (Inter.inter (extChartAt I w).source (extChart …
    ⊢ Membership.mem (Inter.inter (Inter.inter ((tangentBundleCore I M).baseSet (a …
  -/
  simp only [tangentBundleCore_baseSet, coe_achart, ← extChartAt_source I]
  /-
    case a
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    w x y z : M
    v : E
    h : Membership.mem (Inter.inter (Inter.inter (extChartAt I w).source (extChart …
    ⊢ Membership.mem (Inter.inter (Inter.inter (extChartAt I w).source (extChartAt …
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma hasFDerivWithinAt_tangentCoordChange {x y z : M}
    (h : z ∈ (extChartAt I x).source ∩ (extChartAt I y).source) :
    HasFDerivWithinAt ((extChartAt I y) ∘ (extChartAt I x).symm) (tangentCoordChange I x y z)
      (range I) (extChartAt I x z) :=
  have h' : extChartAt I x z ∈ ((extChartAt I x).symm ≫ (extChartAt I y)).source := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_6
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : SmoothManifoldWithCorners I M
      x y z : M
      h : Membership.mem (Inter.inter (extChartAt I x).source (extChartAt I y).sourc …
      ⊢ Membership.mem ((extChartAt I x).symm.trans (extChartAt I y)).source (↑(extC …
    -/
    rw [PartialEquiv.trans_source'', PartialEquiv.symm_symm, PartialEquiv.symm_target]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_6
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : SmoothManifoldWithCorners I M
      x y z : M
      h : Membership.mem (Inter.inter (extChartAt I x).source (extChartAt I y).sourc …
      ⊢ Membership.mem (Set.image (↑(extChartAt I x)) (Inter.inter (extChartAt I x). …
    -/
    exact mem_image_of_mem _ h
    /-
      🎉 no goals
    -/
                                                                         /-
                                                                           𝕜 : Type u_1
                                                                           inst✝⁶ : NontriviallyNormedField 𝕜
                                                                           E : Type u_2
                                                                           inst✝⁵ : NormedAddCommGroup E
                                                                           inst✝⁴ : NormedSpace 𝕜 E
                                                                           H : Type u_4
                                                                           inst✝³ : TopologicalSpace H
                                                                           I : ModelWithCorners 𝕜 E H
                                                                           M : Type u_6
                                                                           inst✝² : TopologicalSpace M
                                                                           inst✝¹ : ChartedSpace H M
                                                                           inst✝ : SmoothManifoldWithCorners I M
                                                                           x y z : M
                                                                           h : Membership.mem (Inter.inter (extChartAt I x).source (extChartAt I y).sourc …
                                                                           h' : Membership.mem ((extChartAt I x).symm.trans (extChartAt I y)).source (↑(e …
                                                                           ⊢ LE.le 1 ↑Top.top
                                                                         -/
  ((contDiffWithinAt_ext_coord_change y x h').differentiableWithinAt (by simp)).hasFDerivWithinAt
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma continuousOn_tangentCoordChange (x y : M) : ContinuousOn (tangentCoordChange I x y)
    ((extChartAt I x).source ∩ (extChartAt I y).source) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    x y : M
    ⊢ ContinuousOn (tangentCoordChange I x y) (Inter.inter (extChartAt I x).source …
  -/
  convert (tangentBundleCore I M).continuousOn_coordChange (achart H x) (achart H y) <;>
  /-
    case h.e'_6.h.e'_3
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    x y : M
    ⊢ Eq (extChartAt I x).source ((tangentBundleCore I M).baseSet (achart H x))
  -/
  /-
    🎉 no goals
  -/
  simp only [tangentBundleCore_baseSet, coe_achart, ← extChartAt_source I]
  /-
    🎉 no goals
  -/


local notation "TM" => TangentBundle I M


instance : TopologicalSpace TM :=
  (tangentBundleCore I M).toTopologicalSpace


instance TangentSpace.fiberBundle : FiberBundle E (TangentSpace I : M → Type _) :=
  (tangentBundleCore I M).fiberBundle


instance TangentSpace.vectorBundle : VectorBundle 𝕜 E (TangentSpace I : M → Type _) :=
  (tangentBundleCore I M).vectorBundle


protected theorem chartAt (p : TM) :
    chartAt (ModelProd H E) p =
      ((tangentBundleCore I M).toFiberBundleCore.localTriv (achart H p.1)).toPartialHomeomorph ≫ₕ
        (chartAt H p.1).prod (PartialHomeomorph.refl E) :=
  rfl


theorem chartAt_toPartialEquiv (p : TM) :
    (chartAt (ModelProd H E) p).toPartialEquiv =
      (tangentBundleCore I M).toFiberBundleCore.localTrivAsPartialEquiv (achart H p.1) ≫
        (chartAt H p.1).toPartialEquiv.prod (PartialEquiv.refl E) :=
  rfl


theorem trivializationAt_eq_localTriv (x : M) :
    trivializationAt E (TangentSpace I) x =
      (tangentBundleCore I M).toFiberBundleCore.localTriv (achart H x) :=
  rfl


@[simp, mfld_simps]
theorem trivializationAt_source (x : M) :
    (trivializationAt E (TangentSpace I) x).source =
      π E (TangentSpace I) ⁻¹' (chartAt H x).source :=
  rfl


@[simp, mfld_simps]
theorem trivializationAt_target (x : M) :
    (trivializationAt E (TangentSpace I) x).target = (chartAt H x).source ×ˢ univ :=
  rfl


@[simp, mfld_simps]
theorem trivializationAt_baseSet (x : M) :
    (trivializationAt E (TangentSpace I) x).baseSet = (chartAt H x).source :=
  rfl


theorem trivializationAt_apply (x : M) (z : TM) :
    trivializationAt E (TangentSpace I) x z =
      (z.1, fderivWithin 𝕜 ((chartAt H x).extend I ∘ ((chartAt H z.1).extend I).symm) (range I)
        ((chartAt H z.1).extend I z.1) z.2) :=
  rfl


@[simp, mfld_simps]
theorem trivializationAt_fst (x : M) (z : TM) : (trivializationAt E (TangentSpace I) x z).1 = z.1 :=
  rfl


@[simp, mfld_simps]
theorem mem_chart_source_iff (p q : TM) :
    p ∈ (chartAt (ModelProd H E) q).source ↔ p.1 ∈ (chartAt H q.1).source := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p q : TangentBundle I M
    ⊢ Iff (Membership.mem (chartAt (ModelProd H E) q).source p) (Membership.mem (c …
  -/
  simp only [FiberBundle.chartedSpace_chartAt, mfld_simps]
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem mem_chart_target_iff (p : H × E) (q : TM) :
    p ∈ (chartAt (ModelProd H E) q).target ↔ p.1 ∈ (chartAt H q.1).target := by
  /- porting note: was
  simp +contextual only [FiberBundle.chartedSpace_chartAt,
    and_iff_left_iff_imp, mfld_simps]
  -/
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p : Prod H E
    q : TangentBundle I M
    ⊢ Iff (Membership.mem (chartAt (ModelProd H E) q).target p) (Membership.mem (c …
  -/
  simp only [FiberBundle.chartedSpace_chartAt, mfld_simps]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p : Prod H E
    q : TangentBundle I M
    ⊢ Iff (And (Membership.mem (chartAt H q.proj).target p.1) (Membership.mem (cha …
  -/
  rw [PartialEquiv.prod_symm]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_6
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    p : Prod H E
    q : TangentBundle I M
    ⊢ Iff (And (Membership.mem (chartAt H q.proj).target p.1) (Membership.mem (cha …
  -/
  simp +contextual only [and_iff_left_iff_imp, mfld_simps]
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem coe_chartAt_fst (p q : TM) : ((chartAt (ModelProd H E) q) p).1 = chartAt H q.1 p.1 :=
  rfl


@[simp, mfld_simps]
theorem coe_chartAt_symm_fst (p : H × E) (q : TM) :
    ((chartAt (ModelProd H E) q).symm p).1 = ((chartAt H q.1).symm : H → M) p.1 :=
  rfl


@[simp, mfld_simps]
theorem trivializationAt_continuousLinearMapAt {b₀ b : M}
    (hb : b ∈ (trivializationAt E (TangentSpace I) b₀).baseSet) :
    (trivializationAt E (TangentSpace I) b₀).continuousLinearMapAt 𝕜 b =
      (tangentBundleCore I M).coordChange (achart H b) (achart H b₀) b :=
  (tangentBundleCore I M).localTriv_continuousLinearMapAt hb


@[simp, mfld_simps]
theorem trivializationAt_symmL {b₀ b : M}
    (hb : b ∈ (trivializationAt E (TangentSpace I) b₀).baseSet) :
    (trivializationAt E (TangentSpace I) b₀).symmL 𝕜 b =
      (tangentBundleCore I M).coordChange (achart H b₀) (achart H b) b :=
  (tangentBundleCore I M).localTriv_symmL hb

-- Porting note: `simp` simplifies LHS to `.id _ _`

@[simp high, mfld_simps]
theorem coordChange_model_space (b b' x : F) :
    (tangentBundleCore 𝓘(𝕜, F) F).coordChange (achart F b) (achart F b') x = 1 := by
  simpa only [tangentBundleCore_coordChange, mfld_simps] using
    fderivWithin_id uniqueDiffWithinAt_univ

-- Porting note: `simp` simplifies LHS to `.id _ _`

@[simp high, mfld_simps]
theorem symmL_model_space (b b' : F) :
    (trivializationAt F (TangentSpace 𝓘(𝕜, F)) b).symmL 𝕜 b' = (1 : F →L[𝕜] F) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_8
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    b b' : F
    ⊢ Eq (Trivialization.symmL 𝕜 (FiberBundle.trivializationAt F (TangentSpace (mo …
  -/
  rw [TangentBundle.trivializationAt_symmL, coordChange_model_space]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_8
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    b b' : F
    ⊢ Membership.mem (FiberBundle.trivializationAt F (TangentSpace (modelWithCorne …
  -/
  apply mem_univ
  /-
    🎉 no goals
  -/

-- Porting note: `simp` simplifies LHS to `.id _ _`

@[simp high, mfld_simps]
theorem continuousLinearMapAt_model_space (b b' : F) :
    (trivializationAt F (TangentSpace 𝓘(𝕜, F)) b).continuousLinearMapAt 𝕜 b' = (1 : F →L[𝕜] F) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_8
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    b b' : F
    ⊢ Eq (Trivialization.continuousLinearMapAt 𝕜 (FiberBundle.trivializationAt F ( …
  -/
  rw [TangentBundle.trivializationAt_continuousLinearMapAt, coordChange_model_space]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_8
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    b b' : F
    ⊢ Membership.mem (FiberBundle.trivializationAt F (TangentSpace (modelWithCorne …
  -/
  apply mem_univ
  /-
    🎉 no goals
  -/


instance tangentBundleCore.isSmooth : (tangentBundleCore I M).IsSmooth I := by
  /-
    𝕜 : Type u_1
    inst✝¹⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹¹ : NormedAddCommGroup E'
    inst✝¹⁰ : NormedSpace 𝕜 E'
    H : Type u_4
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    H' : Type u_5
    inst✝⁸ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_6
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    inst✝⁵ : SmoothManifoldWithCorners I M
    M' : Type u_7
    inst✝⁴ : TopologicalSpace M'
    inst✝³ : ChartedSpace H' M'
    inst✝² : SmoothManifoldWithCorners I' M'
    F : Type u_8
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    ⊢ (tangentBundleCore I M).IsSmooth I
  -/
  refine ⟨fun i j => ?_⟩
  rw [contMDiffOn_iff_source_of_mem_maximalAtlas (subset_maximalAtlas i.2),
    contMDiffOn_iff_contDiffOn]
    /-
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedSpace 𝕜 E'
      H : Type u_4
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_5
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_6
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      inst✝⁵ : SmoothManifoldWithCorners I M
      M' : Type u_7
      inst✝⁴ : TopologicalSpace M'
      inst✝³ : ChartedSpace H' M'
      inst✝² : SmoothManifoldWithCorners I' M'
      F : Type u_8
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      i j : ↑(atlas H M)
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp ((tangentBundleCore I M).coordChange  …
    -/
  · refine ((contDiffOn_fderiv_coord_change (I := I) i j).congr fun x hx => ?_).mono ?_
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j : ↑(atlas H M)
        x : E
        hx : Membership.mem (((↑i).extend I).symm.trans ((↑j).extend I)).source x
        ⊢ Eq (Function.comp ((tangentBundleCore I M).coordChange i j) (↑((↑i).extend I …
      -/
    · rw [PartialEquiv.trans_source'] at hx
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j : ↑(atlas H M)
        x : E
        hx : Membership.mem (Inter.inter ((↑i).extend I).symm.source (Set.preimage (↑( …
        ⊢ Eq (Function.comp ((tangentBundleCore I M).coordChange i j) (↑((↑i).extend I …
      -/
      simp_rw [Function.comp_apply, tangentBundleCore_coordChange, (i.1.extend I).right_inv hx.1]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        i j : ↑(atlas H M)
        ⊢ HasSubset.Subset (Set.image (↑((↑i).extend I)) (Inter.inter ((tangentBundleC …
      -/
    · exact (i.1.extend_image_source_inter j.1).subset
      /-
        🎉 no goals
      -/
    /-
      𝕜 : Type u_1
      inst✝¹⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹¹ : NormedAddCommGroup E'
      inst✝¹⁰ : NormedSpace 𝕜 E'
      H : Type u_4
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_5
      inst✝⁸ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_6
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      inst✝⁵ : SmoothManifoldWithCorners I M
      M' : Type u_7
      inst✝⁴ : TopologicalSpace M'
      inst✝³ : ChartedSpace H' M'
      inst✝² : SmoothManifoldWithCorners I' M'
      F : Type u_8
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      i j : ↑(atlas H M)
      ⊢ HasSubset.Subset (Inter.inter ((tangentBundleCore I M).baseSet i) ((tangentB …
    -/
  · apply inter_subset_left
    /-
      🎉 no goals
    -/


instance TangentBundle.smoothVectorBundle : SmoothVectorBundle E (TangentSpace I : M → Type _) I :=
  (tangentBundleCore I M).smoothVectorBundle


@[simp, mfld_simps]
theorem trivializationAt_model_space_apply (p : TangentBundle I H) (x : H) :
    trivializationAt E (TangentSpace I) x p = (p.1, p.2) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    p : TangentBundle I H
    x : H
    ⊢ Eq (↑(FiberBundle.trivializationAt E (TangentSpace I) x) p) { fst := p.proj, …
  -/
  simp [TangentBundle.trivializationAt_apply]
  have : fderivWithin 𝕜 (↑I ∘ ↑I.symm) (range I) (I p.proj) =
      fderivWithin 𝕜 id (range I) (I p.proj) :=
    fderivWithin_congr' (fun y hy ↦ by simp [hy]) (mem_range_self p.proj)
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    p : TangentBundle I H
    x : H
    this : Eq (fderivWithin 𝕜 (Function.comp ↑I ↑I.symm) (Set.range ↑I) (↑I p.proj …
    ⊢ Eq ((fderivWithin 𝕜 (Function.comp ↑I ↑I.symm) (Set.range ↑I) (↑I p.proj)) p …
  -/
  simp [this, fderivWithin_id (ModelWithCorners.uniqueDiffWithinAt_image I)]
  /-
    🎉 no goals
  -/


/-- In the tangent bundle to the model space, the charts are just the canonical identification
between a product type and a sigma type, a.k.a. `TotalSpace.toProd`. -/
@[simp, mfld_simps]
theorem tangentBundle_model_space_chartAt (p : TangentBundle I H) :
    (chartAt (ModelProd H E) p).toPartialEquiv = (TotalSpace.toProd H E).toPartialEquiv := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    p : TangentBundle I H
    ⊢ Eq (chartAt (ModelProd H E) p).toPartialEquiv (Bundle.TotalSpace.toProd H E) …
  -/
  ext x : 1
    /-
      case h
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      p x : TangentBundle I H
      ⊢ Eq (↑(chartAt (ModelProd H E) p).toPartialEquiv x) (↑(Bundle.TotalSpace.toPr …
    -/
  · ext; · rfl
           /-
             🎉 no goals
           -/
    /-
      case h.h₂
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      p x : TangentBundle I H
      ⊢ Eq (↑(chartAt (ModelProd H E) p).toPartialEquiv x).2 (↑(Bundle.TotalSpace.to …
    -/
    exact (tangentBundleCore I H).coordChange_self (achart _ x.1) x.1 (mem_achart_source H x.1) x.2
    /-
      🎉 no goals
    -/
    /-
      case hsymm
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      p : TangentBundle I H
      x : ModelProd H E
      ⊢ Eq (↑(chartAt (ModelProd H E) p).symm x) (↑(Bundle.TotalSpace.toProd H E).to …
    -/
  · ext; · rfl
           /-
             🎉 no goals
           -/
    /-
      case hsymm.snd
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      p : TangentBundle I H
      x : ModelProd H E
      ⊢ HEq (↑(chartAt (ModelProd H E) p).symm x).snd (↑(Bundle.TotalSpace.toProd H  …
    -/
    apply heq_of_eq
    /-
      case hsymm.snd.h
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      p : TangentBundle I H
      x : ModelProd H E
      ⊢ Eq (↑(chartAt (ModelProd H E) p).symm x).snd (↑(Bundle.TotalSpace.toProd H E …
    -/
    exact (tangentBundleCore I H).coordChange_self (achart _ x.1) x.1 (mem_achart_source H x.1) x.2
    /-
      🎉 no goals
    -/
  simp_rw [TangentBundle.chartAt, FiberBundleCore.localTriv,
    FiberBundleCore.localTrivAsPartialEquiv, VectorBundleCore.toFiberBundleCore_baseSet,
    tangentBundleCore_baseSet]
  /-
    case hs
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    p : TangentBundle I H
    ⊢ Eq ({ toFun := fun p_1 => { fst := p_1.proj, snd := (tangentBundleCore I H). …
  -/
  simp only [mfld_simps]
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem tangentBundle_model_space_coe_chartAt (p : TangentBundle I H) :
    ⇑(chartAt (ModelProd H E) p) = TotalSpace.toProd H E := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    p : TangentBundle I H
    ⊢ Eq ↑(chartAt (ModelProd H E) p) ⇑(Bundle.TotalSpace.toProd H E)
  -/
  rw [← PartialHomeomorph.coe_coe, tangentBundle_model_space_chartAt]; rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp, mfld_simps]
theorem tangentBundle_model_space_coe_chartAt_symm (p : TangentBundle I H) :
    ((chartAt (ModelProd H E) p).symm : ModelProd H E → TangentBundle I H) =
      (TotalSpace.toProd H E).symm := by
  rw [← PartialHomeomorph.coe_coe, PartialHomeomorph.symm_toPartialEquiv,
                                        /-
                                          𝕜 : Type u_1
                                          inst✝³ : NontriviallyNormedField 𝕜
                                          E : Type u_2
                                          inst✝² : NormedAddCommGroup E
                                          inst✝¹ : NormedSpace 𝕜 E
                                          H : Type u_4
                                          inst✝ : TopologicalSpace H
                                          I : ModelWithCorners 𝕜 E H
                                          p : TangentBundle I H
                                          ⊢ Eq ↑(Bundle.TotalSpace.toProd H E).toPartialEquiv.symm ⇑(Bundle.TotalSpace.t …
                                        -/
    tangentBundle_model_space_chartAt]; rfl
                                        /-
                                          🎉 no goals
                                        -/


theorem tangentBundleCore_coordChange_model_space (x x' z : H) :
    (tangentBundleCore I H).coordChange (achart H x) (achart H x') z =
    ContinuousLinearMap.id 𝕜 E := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    x x' z : H
    ⊢ Eq ((tangentBundleCore I H).coordChange (achart H x) (achart H x') z) (Conti …
  -/
  ext v; exact (tangentBundleCore I H).coordChange_self (achart _ z) z (mem_univ _) v
         /-
           🎉 no goals
         -/


variable (I) in
/-- The canonical identification between the tangent bundle to the model space and the product,
as a homeomorphism. For the diffeomorphism version, see `tangentBundleModelSpaceDiffeomorph`. -/
def tangentBundleModelSpaceHomeomorph : TangentBundle I H ≃ₜ ModelProd H E :=
  { TotalSpace.toProd H E with
    continuous_toFun := by
      /-
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        ⊢ Continuous __src✝.toFun
      -/
      let p : TangentBundle I H := ⟨I.symm (0 : E), (0 : E)⟩
      have : Continuous (chartAt (ModelProd H E) p) := by
        rw [continuous_iff_continuousOn_univ]
        convert (chartAt (ModelProd H E) p).continuousOn
        simp only [TangentSpace.fiberBundle, mfld_simps]
      /-
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : TangentBundle I H := { proj := ↑I.symm 0, snd := 0 }
        this : Continuous ↑(chartAt (ModelProd H E) p)
        ⊢ Continuous __src✝.toFun
      -/
      simpa only [mfld_simps] using this
      /-
        🎉 no goals
      -/
    continuous_invFun := by
      /-
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        ⊢ Continuous __src✝.invFun
      -/
      let p : TangentBundle I H := ⟨I.symm (0 : E), (0 : E)⟩
      have : Continuous (chartAt (ModelProd H E) p).symm := by
        rw [continuous_iff_continuousOn_univ]
        convert (chartAt (ModelProd H E) p).symm.continuousOn
        simp only [mfld_simps]
      /-
        𝕜 : Type u_1
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹¹ : NormedAddCommGroup E'
        inst✝¹⁰ : NormedSpace 𝕜 E'
        H : Type u_4
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_5
        inst✝⁸ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M : Type u_6
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        inst✝⁵ : SmoothManifoldWithCorners I M
        M' : Type u_7
        inst✝⁴ : TopologicalSpace M'
        inst✝³ : ChartedSpace H' M'
        inst✝² : SmoothManifoldWithCorners I' M'
        F : Type u_8
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : TangentBundle I H := { proj := ↑I.symm 0, snd := 0 }
        this : Continuous ↑(chartAt (ModelProd H E) p).symm
        ⊢ Continuous __src✝.invFun
      -/
      simpa only [mfld_simps] using this }
      /-
        🎉 no goals
      -/


@[simp, mfld_simps]
theorem tangentBundleModelSpaceHomeomorph_coe :
    (tangentBundleModelSpaceHomeomorph I : TangentBundle I H → ModelProd H E) =
      TotalSpace.toProd H E :=
  rfl


@[simp, mfld_simps]
theorem tangentBundleModelSpaceHomeomorph_coe_symm :
    ((tangentBundleModelSpaceHomeomorph I).symm : ModelProd H E → TangentBundle I H) =
      (TotalSpace.toProd H E).symm :=
  rfl


theorem contMDiff_tangentBundleModelSpaceHomeomorph {n : ℕ∞} :
    ContMDiff I.tangent (I.prod 𝓘(𝕜, E)) n
    (tangentBundleModelSpaceHomeomorph I : TangentBundle I H → ModelProd H E) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    ⊢ ContMDiff I.tangent (I.prod (modelWithCornersSelf 𝕜 E)) n ⇑(tangentBundleMod …
  -/
  apply contMDiff_iff.2 ⟨Homeomorph.continuous _, fun x y ↦ ?_⟩
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : TangentBundle I H
    y : ModelProd H E
    ⊢ ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt (I.prod (modelWithCornersSelf …
  -/
  apply contDiffOn_id.congr
  simp only [mfld_simps, mem_range, TotalSpace.toProd, Equiv.coe_fn_symm_mk, forall_exists_index,
    Prod.forall, Prod.mk.injEq]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : TangentBundle I H
    y : ModelProd H E
    ⊢ ∀ (a b : E) (x : H), Eq (↑I x) a → And (Eq (↑I ({ toFun := fun x => { fst := …
  -/
  rintro a b x rfl
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x✝ : TangentBundle I H
    y : ModelProd H E
    b : E
    x : H
    ⊢ And (Eq (↑I ({ toFun := fun x => { fst := x.proj, snd := x.snd }, invFun :=  …
  -/
  simp [PartialEquiv.prod]
  /-
    🎉 no goals
  -/


theorem contMDiff_tangentBundleModelSpaceHomeomorph_symm {n : ℕ∞} :
    ContMDiff (I.prod 𝓘(𝕜, E)) I.tangent n
    ((tangentBundleModelSpaceHomeomorph I).symm : ModelProd H E → TangentBundle I H) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    ⊢ ContMDiff (I.prod (modelWithCornersSelf 𝕜 E)) I.tangent n ⇑(tangentBundleMod …
  -/
  apply contMDiff_iff.2 ⟨Homeomorph.continuous _, fun x y ↦ ?_⟩
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : ModelProd H E
    y : TangentBundle I H
    ⊢ ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I.tangent y)) (Function.comp  …
  -/
  apply contDiffOn_id.congr
  simp only [mfld_simps, mem_range, TotalSpace.toProd, Equiv.coe_fn_symm_mk, forall_exists_index,
    Prod.forall, Prod.mk.injEq]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : ModelProd H E
    y : TangentBundle I H
    ⊢ ∀ (a b : E) (x : H), Eq (↑I x) a → And (Eq (↑I ({ toFun := fun x => { fst := …
  -/
  rintro a b x rfl
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x✝ : ModelProd H E
    y : TangentBundle I H
    b : E
    x : H
    ⊢ And (Eq (↑I ({ toFun := fun x => { fst := x.proj, snd := x.snd }, invFun :=  …
  -/
  simp [PartialEquiv.prod]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x✝ : ModelProd H E
    y : TangentBundle I H
    b : E
    x : H
    ⊢ And (Eq (↑I ({ toFun := fun x => { fst := x.proj, snd := x.snd }, invFun :=  …
  -/
  exact ⟨rfl, rfl⟩
  /-
    🎉 no goals
  -/


variable (H I) in
/-- In the tangent bundle to the model space, the second projection is smooth. -/
lemma contMDiff_snd_tangentBundle_modelSpace {n : ℕ∞} :
    ContMDiff I.tangent 𝓘(𝕜, E) n (fun (p : TangentBundle I H) ↦ p.2) := by
  change ContMDiff I.tangent 𝓘(𝕜, E) n
    ((id Prod.snd : ModelProd H E → E) ∘ (tangentBundleModelSpaceHomeomorph I))
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_4
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    ⊢ ContMDiff I.tangent (modelWithCornersSelf 𝕜 E) n (Function.comp (id Prod.snd …
  -/
  apply ContMDiff.comp (I' := I.prod 𝓘(𝕜, E))
    /-
      case hg
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      n : ENat
      ⊢ ContMDiff (I.prod (modelWithCornersSelf 𝕜 E)) (modelWithCornersSelf 𝕜 E) n ( …
    -/
  · convert contMDiff_snd
    /-
      case h.e'_11.h
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      n : ENat
      e_10✝ : Eq (instTopologicalSpaceModelProd H E) instTopologicalSpaceProd
      ⊢ Eq (chartedSpaceSelf (ModelProd H E)) (prodChartedSpace H H E E)
    -/
    rw [chartedSpaceSelf_prod]
    /-
      case h.e'_11.h
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      n : ENat
      e_10✝ : Eq (instTopologicalSpaceModelProd H E) instTopologicalSpaceProd
      ⊢ Eq (chartedSpaceSelf (ModelProd H E)) (chartedSpaceSelf (Prod H E))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hf
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_4
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      n : ENat
      ⊢ ContMDiff I.tangent (I.prod (modelWithCornersSelf 𝕜 E)) n ⇑(tangentBundleMod …
    -/
  · exact contMDiff_tangentBundleModelSpaceHomeomorph
    /-
      🎉 no goals
    -/


/-- A vector field on a vector space is smooth in the manifold sense iff it is smooth in the vector
space sense-/
lemma contMDiffWithinAt_vectorSpace_iff_contDiffWithinAt
    {V : Π (x : E), TangentSpace 𝓘(𝕜, E) x} {n : ℕ∞} {s : Set E} {x : E} :
    ContMDiffWithinAt 𝓘(𝕜, E) 𝓘(𝕜, E).tangent n (fun x ↦ (V x : TangentBundle 𝓘(𝕜, E) E)) s x ↔
      ContDiffWithinAt 𝕜 n V s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V : (x : E) → TangentSpace (modelWithCornersSelf 𝕜 E) x
    n : ENat
    s : Set E
    x : E
    ⊢ Iff (ContMDiffWithinAt (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E) …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
  · exact ContMDiffWithinAt.contDiffWithinAt <|
      (contMDiff_snd_tangentBundle_modelSpace E 𝓘(𝕜, E)).contMDiffAt.comp_contMDiffWithinAt _ h
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      V : (x : E) → TangentSpace (modelWithCornersSelf 𝕜 E) x
      n : ENat
      s : Set E
      x : E
      h : ContDiffWithinAt 𝕜 (↑n) V s x
      ⊢ ContMDiffWithinAt (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E).tang …
    -/
  · apply (Bundle.contMDiffWithinAt_totalSpace _).2
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      V : (x : E) → TangentSpace (modelWithCornersSelf 𝕜 E) x
      n : ENat
      s : Set E
      x : E
      h : ContDiffWithinAt 𝕜 (↑n) V s x
      ⊢ And (ContMDiffWithinAt (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E) …
    -/
    refine ⟨contMDiffWithinAt_id, ?_⟩
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      V : (x : E) → TangentSpace (modelWithCornersSelf 𝕜 E) x
      n : ENat
      s : Set E
      x : E
      h : ContDiffWithinAt 𝕜 (↑n) V s x
      ⊢ ContMDiffWithinAt (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E) n (f …
    -/
    convert h.contMDiffWithinAt with y
    /-
      case h.e'_22.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      V : (x : E) → TangentSpace (modelWithCornersSelf 𝕜 E) x
      n : ENat
      s : Set E
      x : E
      h : ContDiffWithinAt 𝕜 (↑n) V s x
      y : E
      ⊢ Eq (↑(FiberBundle.trivializationAt E (TangentSpace (modelWithCornersSelf 𝕜 E …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- A vector field on a vector space is smooth in the manifold sense iff it is smooth in the vector
space sense-/
lemma contMDiffAt_vectorSpace_iff_contDiffAt
    {V : Π (x : E), TangentSpace 𝓘(𝕜, E) x} {n : ℕ∞} {x : E} :
    ContMDiffAt 𝓘(𝕜, E) 𝓘(𝕜, E).tangent n (fun x ↦ (V x : TangentBundle 𝓘(𝕜, E) E)) x ↔
      ContDiffAt 𝕜 n V x := by
  simp only [← contMDiffWithinAt_univ, ← contDiffWithinAt_univ,
    contMDiffWithinAt_vectorSpace_iff_contDiffWithinAt]


/-- A vector field on a vector space is smooth in the manifold sense iff it is smooth in the vector
space sense-/
lemma contMDiffOn_vectorSpace_iff_contDiffOn
    {V : Π (x : E), TangentSpace 𝓘(𝕜, E) x} {n : ℕ∞} {s : Set E} :
    ContMDiffOn 𝓘(𝕜, E) 𝓘(𝕜, E).tangent n (fun x ↦ (V x : TangentBundle 𝓘(𝕜, E) E)) s ↔
      ContDiffOn 𝕜 n V s := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V : (x : E) → TangentSpace (modelWithCornersSelf 𝕜 E) x
    n : ENat
    s : Set E
    ⊢ Iff (ContMDiffOn (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E).tange …
  -/
  simp only [ContMDiffOn, ContDiffOn, contMDiffWithinAt_vectorSpace_iff_contDiffWithinAt ]
  /-
    🎉 no goals
  -/


/-- A vector field on a vector space is smooth in the manifold sense iff it is smooth in the vector
space sense-/
lemma contMDiff_vectorSpace_iff_contDiff
    {V : Π (x : E), TangentSpace 𝓘(𝕜, E) x} {n : ℕ∞} :
    ContMDiff 𝓘(𝕜, E) 𝓘(𝕜, E).tangent n (fun x ↦ (V x : TangentBundle 𝓘(𝕜, E) E)) ↔
      ContDiff 𝕜 n V := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    V : (x : E) → TangentSpace (modelWithCornersSelf 𝕜 E) x
    n : ENat
    ⊢ Iff (ContMDiff (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E).tangent …
  -/
  simp only [← contMDiffOn_univ, ← contDiffOn_univ, contMDiffOn_vectorSpace_iff_contDiffOn]
  /-
    🎉 no goals
  -/


/-- The map `inCoordinates` for the tangent bundle is trivial on the model spaces -/
theorem inCoordinates_tangent_bundle_core_model_space (x₀ x : H) (y₀ y : H') (ϕ : E →L[𝕜] E') :
    inCoordinates E (TangentSpace I) E' (TangentSpace I') x₀ x y₀ y ϕ = ϕ := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    H : Type u_4
    inst✝¹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    H' : Type u_5
    inst✝ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    x₀ x : H
    y₀ y : H'
    ϕ : ContinuousLinearMap (RingHom.id 𝕜) E E'
    ⊢ Eq (ContinuousLinearMap.inCoordinates E (TangentSpace I) E' (TangentSpace I' …
  -/
                                              /-
                                                🎉 no goals
                                              -/
  erw [VectorBundleCore.inCoordinates_eq] <;> try trivial
                                              /-
                                                🎉 no goals
                                              -/
  simp_rw [tangentBundleCore_indexAt, tangentBundleCore_coordChange_model_space,
    ContinuousLinearMap.id_comp, ContinuousLinearMap.comp_id]


variable (I I') in
/-- When `ϕ x` is a continuous linear map that changes vectors in charts around `f x` to vectors
in charts around `g x`, `inTangentCoordinates I I' f g ϕ x₀ x` is a coordinate change of
this continuous linear map that makes sense from charts around `f x₀` to charts around `g x₀`
by composing it with appropriate coordinate changes.
Note that the type of `ϕ` is more accurately
`Π x : N, TangentSpace I (f x) →L[𝕜] TangentSpace I' (g x)`.
We are unfolding `TangentSpace` in this type so that Lean recognizes that the type of `ϕ` doesn't
actually depend on `f` or `g`.

This is the underlying function of the trivializations of the hom of (pullbacks of) tangent spaces.
-/
def inTangentCoordinates (f : N → M) (g : N → M') (ϕ : N → E →L[𝕜] E') : N → N → E →L[𝕜] E' :=
  fun x₀ x => inCoordinates E (TangentSpace I) E' (TangentSpace I') (f x₀) (f x) (g x₀) (g x) (ϕ x)


theorem inTangentCoordinates_model_space (f : N → H) (g : N → H') (ϕ : N → E →L[𝕜] E') (x₀ : N) :
    inTangentCoordinates I I' f g ϕ x₀ = ϕ := by
  simp (config := { unfoldPartialApp := true }) only [inTangentCoordinates,
    inCoordinates_tangent_bundle_core_model_space]


/-- To write a linear map between tangent spaces in coordinates amounts to precomposing and
postcomposing it with suitable coordinate changes. For a concrete version expressing the
change of coordinates as derivatives of extended charts,
see `inTangentCoordinates_eq_mfderiv_comp`. -/
theorem inTangentCoordinates_eq (f : N → M) (g : N → M') (ϕ : N → E →L[𝕜] E') {x₀ x : N}
    (hx : f x ∈ (chartAt H (f x₀)).source) (hy : g x ∈ (chartAt H' (g x₀)).source) :
    inTangentCoordinates I I' f g ϕ x₀ x =
      (tangentBundleCore I' M').coordChange (achart H' (g x)) (achart H' (g x₀)) (g x) ∘L
        ϕ x ∘L (tangentBundleCore I M).coordChange (achart H (f x₀)) (achart H (f x)) (f x) :=
  (tangentBundleCore I M).inCoordinates_eq (tangentBundleCore I' M') (ϕ x) hx hy


