/-- Let `X` and `Y` be R₁ topological spaces
with Borel σ-algebras and measures `μ` and `ν`, respectively.
Suppose that `μ` is inner regular for finite measure sets with respect to compact sets
and `ν` is a locally finite measure.
Let `f : α → C(X, Y)` be a family of continuous maps
that converges to a continuous map `g : C(X, Y)` in the compact-open topology along a filter `l`.
Suppose that `g` is a measure preserving map
and `f a` is a measure preserving map eventually along `l`.
Then for any finite measure measurable set `s`,
the preimages `f a ⁻¹' s` tend to the preimage `g ⁻¹' s` in measure.
More precisely, the measure of the symmetric difference of these two sets tends to zero. -/
theorem tendsto_measure_symmDiff_preimage_nhds_zero
    {l : Filter α} {f : α → C(X, Y)} {g : C(X, Y)} {s : Set Y} (hfg : Tendsto f l (𝓝 g))
    (hf : ∀ᶠ a in l, MeasurePreserving (f a) μ ν) (hg : MeasurePreserving g μ ν)
    (hs : NullMeasurableSet s ν) (hνs : ν s ≠ ∞) :
    Tendsto (fun a ↦ μ ((f a ⁻¹' s) ∆ (g ⁻¹' s))) l (𝓝 0) := by
  have : ν.InnerRegularCompactLTTop := by
    rw [← hg.map_eq]
    exact .map_of_continuous (map_continuous _)
  /-
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    s : Set Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    this : ν.InnerRegularCompactLTTop
    ⊢ Filter.Tendsto (fun a => μ (symmDiff (Set.preimage (⇑(f a)) s) (Set.preimage …
  -/
  rw [ENNReal.tendsto_nhds_zero]
  /-
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    s : Set Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    this : ν.InnerRegularCompactLTTop
    ⊢ ∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventually (fun x => LE.le (μ (symmDiff  …
  -/
  intro ε hε
  -- Without loss of generality, `s` is an open set.
  /-
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    s : Set Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    this : ν.InnerRegularCompactLTTop
    ε : ENNReal
    hε : GT.gt ε 0
    ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
  -/
  wlog hso : IsOpen s generalizing s ε
    /-
      case inr
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : R1Space X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : BorelSpace Y
      inst✝² : R1Space Y
      μ : MeasureTheory.Measure X
      ν : MeasureTheory.Measure Y
      inst✝¹ : μ.InnerRegularCompactLTTop
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
      l : Filter α
      f : α → ContinuousMap X Y
      g : ContinuousMap X Y
      s : Set Y
      hfg : Filter.Tendsto f l (nhds g)
      hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
      hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
      hs : MeasureTheory.NullMeasurableSet s ν
      hνs : Ne (ν s) Top.top
      this✝ : ν.InnerRegularCompactLTTop
      ε : ENNReal
      hε : GT.gt ε 0
      this : ∀ {s : Set Y}, MeasureTheory.NullMeasurableSet s ν → Ne (ν s) Top.top → …
      hso : Not (IsOpen s)
      ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
    -/
  · have H : 0 < ε / 3 := ENNReal.div_pos hε.ne' ENNReal.coe_ne_top
    -- Indeed, we can choose an open set `U` such that `ν (U ∆ s) < ε / 3`,
    -- apply the lemma to `U`, then use the triangle inequality for `μ (_ ∆ _)`.
    /-
      case inr
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : R1Space X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : BorelSpace Y
      inst✝² : R1Space Y
      μ : MeasureTheory.Measure X
      ν : MeasureTheory.Measure Y
      inst✝¹ : μ.InnerRegularCompactLTTop
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
      l : Filter α
      f : α → ContinuousMap X Y
      g : ContinuousMap X Y
      s : Set Y
      hfg : Filter.Tendsto f l (nhds g)
      hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
      hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
      hs : MeasureTheory.NullMeasurableSet s ν
      hνs : Ne (ν s) Top.top
      this✝ : ν.InnerRegularCompactLTTop
      ε : ENNReal
      hε : GT.gt ε 0
      this : ∀ {s : Set Y}, MeasureTheory.NullMeasurableSet s ν → Ne (ν s) Top.top → …
      hso : Not (IsOpen s)
      H : LT.lt 0 (HDiv.hDiv ε 3)
      ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
    -/
    rcases hs.exists_isOpen_symmDiff_lt hνs H.ne' with ⟨U, hUo, hU, hUs⟩
    /-
      case inr.intro.intro.intro
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : R1Space X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : BorelSpace Y
      inst✝² : R1Space Y
      μ : MeasureTheory.Measure X
      ν : MeasureTheory.Measure Y
      inst✝¹ : μ.InnerRegularCompactLTTop
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
      l : Filter α
      f : α → ContinuousMap X Y
      g : ContinuousMap X Y
      s : Set Y
      hfg : Filter.Tendsto f l (nhds g)
      hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
      hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
      hs : MeasureTheory.NullMeasurableSet s ν
      hνs : Ne (ν s) Top.top
      this✝ : ν.InnerRegularCompactLTTop
      ε : ENNReal
      hε : GT.gt ε 0
      this : ∀ {s : Set Y}, MeasureTheory.NullMeasurableSet s ν → Ne (ν s) Top.top → …
      hso : Not (IsOpen s)
      H : LT.lt 0 (HDiv.hDiv ε 3)
      U : Set Y
      hUo : IsOpen U
      hU : LT.lt (ν U) Top.top
      hUs : LT.lt (ν (symmDiff U s)) (HDiv.hDiv ε 3)
      ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
    -/
    have hmU : NullMeasurableSet U ν := hUo.measurableSet.nullMeasurableSet
    /-
      case inr.intro.intro.intro
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : R1Space X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : BorelSpace Y
      inst✝² : R1Space Y
      μ : MeasureTheory.Measure X
      ν : MeasureTheory.Measure Y
      inst✝¹ : μ.InnerRegularCompactLTTop
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
      l : Filter α
      f : α → ContinuousMap X Y
      g : ContinuousMap X Y
      s : Set Y
      hfg : Filter.Tendsto f l (nhds g)
      hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
      hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
      hs : MeasureTheory.NullMeasurableSet s ν
      hνs : Ne (ν s) Top.top
      this✝ : ν.InnerRegularCompactLTTop
      ε : ENNReal
      hε : GT.gt ε 0
      this : ∀ {s : Set Y}, MeasureTheory.NullMeasurableSet s ν → Ne (ν s) Top.top → …
      hso : Not (IsOpen s)
      H : LT.lt 0 (HDiv.hDiv ε 3)
      U : Set Y
      hUo : IsOpen U
      hU : LT.lt (ν U) Top.top
      hUs : LT.lt (ν (symmDiff U s)) (HDiv.hDiv ε 3)
      hmU : MeasureTheory.NullMeasurableSet U ν
      ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
    -/
    replace hUs := hUs.le
    /-
      case inr.intro.intro.intro
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : R1Space X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : BorelSpace Y
      inst✝² : R1Space Y
      μ : MeasureTheory.Measure X
      ν : MeasureTheory.Measure Y
      inst✝¹ : μ.InnerRegularCompactLTTop
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
      l : Filter α
      f : α → ContinuousMap X Y
      g : ContinuousMap X Y
      s : Set Y
      hfg : Filter.Tendsto f l (nhds g)
      hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
      hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
      hs : MeasureTheory.NullMeasurableSet s ν
      hνs : Ne (ν s) Top.top
      this✝ : ν.InnerRegularCompactLTTop
      ε : ENNReal
      hε : GT.gt ε 0
      this : ∀ {s : Set Y}, MeasureTheory.NullMeasurableSet s ν → Ne (ν s) Top.top → …
      hso : Not (IsOpen s)
      H : LT.lt 0 (HDiv.hDiv ε 3)
      U : Set Y
      hUo : IsOpen U
      hU : LT.lt (ν U) Top.top
      hmU : MeasureTheory.NullMeasurableSet U ν
      hUs : LE.le (ν (symmDiff U s)) (HDiv.hDiv ε 3)
      ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
    -/
    filter_upwards [hf, this hmU hU.ne _ H hUo] with a hfa ha
    calc
      μ ((f a ⁻¹' s) ∆ (g ⁻¹' s))
        ≤ μ ((f a ⁻¹' s) ∆ (f a ⁻¹' U)) + μ ((f a ⁻¹' U) ∆ (g ⁻¹' U))
          + μ ((g ⁻¹' U) ∆ (g ⁻¹' s)) := by
        refine (measure_symmDiff_le _ (g ⁻¹' U) _).trans ?_
        gcongr
        apply measure_symmDiff_le
      _ ≤ ε / 3 + ε / 3 + ε / 3 := by
        gcongr
        · rwa [← preimage_symmDiff, hfa.measure_preimage (hs.symmDiff hmU), symmDiff_comm]
        · rwa [← preimage_symmDiff, hg.measure_preimage (hmU.symmDiff hs)]
      _ = ε := by simp
  -- Take a compact closed subset `K ⊆ g ⁻¹' s` of almost full measure,
  -- `μ (g ⁻¹' s \ K) < ε / 2`.
  /-
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
  -/
  have hνs' : μ (g ⁻¹' s) ≠ ∞ := by rwa [hg.measure_preimage hs]
  obtain ⟨K, hKg, hKco, hKcl, hKμ⟩ :
      ∃ K, MapsTo g K s ∧ IsCompact K ∧ IsClosed K ∧ μ (g ⁻¹' s \ K) < ε / 2 :=
    (hg.measurable hso.measurableSet).exists_isCompact_isClosed_diff_lt hνs' <| by simp [hε.ne']
  /-
    case intro.intro.intro.intro
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    hνs' : Ne (μ (Set.preimage (⇑g) s)) Top.top
    K : Set X
    hKg : Set.MapsTo (⇑g) K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hKμ : LT.lt (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
    ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
  -/
  have hKm : NullMeasurableSet K μ := hKcl.nullMeasurableSet
  -- Take `a` such that `f a` is measure preserving and maps `K` to `s`.
  -- This is possible, because `K` is a compact set and `s` is an open set.
  /-
    case intro.intro.intro.intro
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    hνs' : Ne (μ (Set.preimage (⇑g) s)) Top.top
    K : Set X
    hKg : Set.MapsTo (⇑g) K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hKμ : LT.lt (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
    hKm : MeasureTheory.NullMeasurableSet K μ
    ⊢ Filter.Eventually (fun x => LE.le (μ (symmDiff (Set.preimage (⇑(f x)) s) (Se …
  -/
  filter_upwards [hf, ContinuousMap.tendsto_nhds_compactOpen.mp hfg K hKco s hso hKg] with a hfa ha
  -- Then each of the sets `g ⁻¹' s ∆ K = g ⁻¹' s \ K` and `f a ⁻¹' s ∆ K = f a ⁻¹' s \ K`
  -- have measure at most `ε / 2`, thus `f a ⁻¹' s ∆ g ⁻¹' s` has measure at most `ε`.
  /-
    case h
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    hνs' : Ne (μ (Set.preimage (⇑g) s)) Top.top
    K : Set X
    hKg : Set.MapsTo (⇑g) K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hKμ : LT.lt (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
    hKm : MeasureTheory.NullMeasurableSet K μ
    a : α
    hfa : MeasureTheory.MeasurePreserving (⇑(f a)) μ ν
    ha : Set.MapsTo (⇑(f a)) K s
    ⊢ LE.le (μ (symmDiff (Set.preimage (⇑(f a)) s) (Set.preimage (⇑g) s))) ε
  -/
  rw [← ENNReal.add_halves ε]
  /-
    case h
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    hνs' : Ne (μ (Set.preimage (⇑g) s)) Top.top
    K : Set X
    hKg : Set.MapsTo (⇑g) K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hKμ : LT.lt (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
    hKm : MeasureTheory.NullMeasurableSet K μ
    a : α
    hfa : MeasureTheory.MeasurePreserving (⇑(f a)) μ ν
    ha : Set.MapsTo (⇑(f a)) K s
    ⊢ LE.le (μ (symmDiff (Set.preimage (⇑(f a)) s) (Set.preimage (⇑g) s))) (HAdd.h …
  -/
  refine (measure_symmDiff_le _ K _).trans ?_
  /-
    case h
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    hνs' : Ne (μ (Set.preimage (⇑g) s)) Top.top
    K : Set X
    hKg : Set.MapsTo (⇑g) K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hKμ : LT.lt (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
    hKm : MeasureTheory.NullMeasurableSet K μ
    a : α
    hfa : MeasureTheory.MeasurePreserving (⇑(f a)) μ ν
    ha : Set.MapsTo (⇑(f a)) K s
    ⊢ LE.le (HAdd.hAdd (μ (symmDiff (Set.preimage (⇑(f a)) s) K)) (μ (symmDiff K ( …
  -/
  rw [symmDiff_of_ge ha.subset_preimage, symmDiff_of_le hKg.subset_preimage]
  /-
    case h
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    hνs' : Ne (μ (Set.preimage (⇑g) s)) Top.top
    K : Set X
    hKg : Set.MapsTo (⇑g) K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hKμ : LT.lt (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
    hKm : MeasureTheory.NullMeasurableSet K μ
    a : α
    hfa : MeasureTheory.MeasurePreserving (⇑(f a)) μ ν
    ha : Set.MapsTo (⇑(f a)) K s
    ⊢ LE.le (HAdd.hAdd (μ (SDiff.sdiff (Set.preimage (⇑(f a)) s) K)) (μ (SDiff.sdi …
  -/
  gcongr
  /-
    case h.h₁
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    hνs' : Ne (μ (Set.preimage (⇑g) s)) Top.top
    K : Set X
    hKg : Set.MapsTo (⇑g) K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hKμ : LT.lt (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
    hKm : MeasureTheory.NullMeasurableSet K μ
    a : α
    hfa : MeasureTheory.MeasurePreserving (⇑(f a)) μ ν
    ha : Set.MapsTo (⇑(f a)) K s
    ⊢ LE.le (μ (SDiff.sdiff (Set.preimage (⇑(f a)) s) K)) (HDiv.hDiv ε 2)
  -/
  have hK' : μ K ≠ ∞ := ne_top_of_le_ne_top hνs' <| measure_mono hKg.subset_preimage
  rw [measure_diff_le_iff_le_add hKm ha.subset_preimage hK', hfa.measure_preimage hs,
    ← hg.measure_preimage hs, ← measure_diff_le_iff_le_add hKm hKg.subset_preimage hK']
  /-
    case h.h₁
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    inst✝⁶ : R1Space X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : BorelSpace Y
    inst✝² : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    l : Filter α
    f : α → ContinuousMap X Y
    g : ContinuousMap X Y
    hfg : Filter.Tendsto f l (nhds g)
    hf : Filter.Eventually (fun a => MeasureTheory.MeasurePreserving (⇑(f a)) μ ν) l
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    this : ν.InnerRegularCompactLTTop
    s : Set Y
    hs : MeasureTheory.NullMeasurableSet s ν
    hνs : Ne (ν s) Top.top
    ε : ENNReal
    hε : GT.gt ε 0
    hso : IsOpen s
    hνs' : Ne (μ (Set.preimage (⇑g) s)) Top.top
    K : Set X
    hKg : Set.MapsTo (⇑g) K s
    hKco : IsCompact K
    hKcl : IsClosed K
    hKμ : LT.lt (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
    hKm : MeasureTheory.NullMeasurableSet K μ
    a : α
    hfa : MeasureTheory.MeasurePreserving (⇑(f a)) μ ν
    ha : Set.MapsTo (⇑(f a)) K s
    hK' : Ne (μ K) Top.top
    ⊢ LE.le (μ (SDiff.sdiff (Set.preimage (⇑g) s) K)) (HDiv.hDiv ε 2)
  -/
  exact hKμ.le
  /-
    🎉 no goals
  -/


/-- Let `f : Z → C(X, Y)` be a continuous (in the compact open topology) family
of continuous measure preserving maps.
Let `t : Set Y` be a null measurable set of finite measure.
Then for any `s`, the set of parameters `z`
such that the preimage of `t` under `f_z` is a.e. equal to `s`
is a closed set.

In particular, if `X = Y` and `s = t`,
then we see that the a.e. stabilizer of a set is a closed set. -/
theorem isClosed_setOf_preimage_ae_eq {f : Z → C(X, Y)} (hf : Continuous f)
    (hfm : ∀ z, MeasurePreserving (f z) μ ν) (s : Set X)
    {t : Set Y} (htm : NullMeasurableSet t ν) (ht : ν t ≠ ∞) :
    IsClosed {z | f z ⁻¹' t =ᵐ[μ] s} := by
  /-
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : BorelSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace Y
    inst✝⁴ : BorelSpace Y
    inst✝³ : R1Space Y
    inst✝² : TopologicalSpace Z
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    f : Z → ContinuousMap X Y
    hf : Continuous f
    hfm : ∀ (z : Z), MeasureTheory.MeasurePreserving (⇑(f z)) μ ν
    s : Set X
    t : Set Y
    htm : MeasureTheory.NullMeasurableSet t ν
    ht : Ne (ν t) Top.top
    ⊢ IsClosed (setOf fun z => (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑( …
  -/
  rw [← isOpen_compl_iff, isOpen_iff_mem_nhds]
  /-
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : BorelSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace Y
    inst✝⁴ : BorelSpace Y
    inst✝³ : R1Space Y
    inst✝² : TopologicalSpace Z
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    f : Z → ContinuousMap X Y
    hf : Continuous f
    hfm : ∀ (z : Z), MeasureTheory.MeasurePreserving (⇑(f z)) μ ν
    s : Set X
    t : Set Y
    htm : MeasureTheory.NullMeasurableSet t ν
    ht : Ne (ν t) Top.top
    ⊢ ∀ (x : Z), Membership.mem (HasCompl.compl (setOf fun z => (MeasureTheory.ae  …
  -/
  intro z hz
  replace hz : ∀ᶠ ε : ℝ≥0∞ in 𝓝 0, ε < μ ((f z ⁻¹' t) ∆ s) := by
    apply gt_mem_nhds
    rwa [pos_iff_ne_zero, ne_eq, measure_symmDiff_eq_zero_iff]
  filter_upwards [(tendsto_measure_symmDiff_preimage_nhds_zero (hf.tendsto z)
    (.of_forall hfm) (hfm z) htm ht).eventually hz] with w hw
  /-
    case h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : BorelSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace Y
    inst✝⁴ : BorelSpace Y
    inst✝³ : R1Space Y
    inst✝² : TopologicalSpace Z
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    f : Z → ContinuousMap X Y
    hf : Continuous f
    hfm : ∀ (z : Z), MeasureTheory.MeasurePreserving (⇑(f z)) μ ν
    s : Set X
    t : Set Y
    htm : MeasureTheory.NullMeasurableSet t ν
    ht : Ne (ν t) Top.top
    z : Z
    hz : Filter.Eventually (fun ε => LT.lt ε (μ (symmDiff (Set.preimage (⇑(f z)) t …
    w : Z
    hw : LT.lt (μ (symmDiff (Set.preimage (⇑(f w)) t) (Set.preimage (⇑(f z)) t)))  …
    ⊢ Membership.mem (HasCompl.compl (setOf fun z => (MeasureTheory.ae μ).Eventual …
  -/
  intro (hw' : f w ⁻¹' t =ᵐ[μ] s)
  /-
    case h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : BorelSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace Y
    inst✝⁴ : BorelSpace Y
    inst✝³ : R1Space Y
    inst✝² : TopologicalSpace Z
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    f : Z → ContinuousMap X Y
    hf : Continuous f
    hfm : ∀ (z : Z), MeasureTheory.MeasurePreserving (⇑(f z)) μ ν
    s : Set X
    t : Set Y
    htm : MeasureTheory.NullMeasurableSet t ν
    ht : Ne (ν t) Top.top
    z : Z
    hz : Filter.Eventually (fun ε => LT.lt ε (μ (symmDiff (Set.preimage (⇑(f z)) t …
    w : Z
    hw : LT.lt (μ (symmDiff (Set.preimage (⇑(f w)) t) (Set.preimage (⇑(f z)) t)))  …
    hw' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(f w)) t) s
    ⊢ False
  -/
  rw [measure_congr (hw'.symmDiff (ae_eq_refl _)), symmDiff_comm] at hw
  /-
    case h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : BorelSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace Y
    inst✝⁴ : BorelSpace Y
    inst✝³ : R1Space Y
    inst✝² : TopologicalSpace Z
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : μ.InnerRegularCompactLTTop
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ν
    f : Z → ContinuousMap X Y
    hf : Continuous f
    hfm : ∀ (z : Z), MeasureTheory.MeasurePreserving (⇑(f z)) μ ν
    s : Set X
    t : Set Y
    htm : MeasureTheory.NullMeasurableSet t ν
    ht : Ne (ν t) Top.top
    z : Z
    hz : Filter.Eventually (fun ε => LT.lt ε (μ (symmDiff (Set.preimage (⇑(f z)) t …
    w : Z
    hw : LT.lt (μ (symmDiff (Set.preimage (⇑(f z)) t) s)) (μ (symmDiff (Set.preima …
    hw' : (MeasureTheory.ae μ).EventuallyEq (Set.preimage (⇑(f w)) t) s
    ⊢ False
  -/
  exact hw.false
  /-
    🎉 no goals
  -/


