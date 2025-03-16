/-- Let `X` and `Y` be R₁ topological spaces
with Borel σ-algebras and measures `μ` and `ν`, respectively.
Suppose that `μ` is inner regular for finite measure sets with respect to compact sets
and `ν` is a locally finite measure.
Let `1 ≤ p < ∞` be an extended nonnegative real number.
Then the composition of a function `g : Lp E p ν`
and a measure preserving continuous function `f : C(X, Y)`
is continuous in both variables. -/
theorem compMeasurePreserving_continuous (hp : p ≠ ∞) :
    Continuous fun gf : Lp E p ν × {f : C(X, Y) // MeasurePreserving f μ ν} ↦
      compMeasurePreserving gf.2.1 gf.2.2 gf.1 := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : MeasurableSpace X
    inst✝⁹ : BorelSpace X
    inst✝⁸ : R1Space X
    inst✝⁷ : TopologicalSpace Y
    inst✝⁶ : MeasurableSpace Y
    inst✝⁵ : BorelSpace Y
    inst✝⁴ : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure ν
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    ⊢ Continuous fun gf => (MeasureTheory.Lp.compMeasurePreserving ⇑↑gf.2 ⋯) gf.1
  -/
  have hp₀ : p ≠ 0 := (one_pos.trans_le Fact.out).ne'
  refine continuous_prod_of_dense_continuous_lipschitzWith _ 1
    (MeasureTheory.Lp.simpleFunc.dense hp) ?_ fun f ↦ (isometry_compMeasurePreserving f.2).lipschitz
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : MeasurableSpace X
    inst✝⁹ : BorelSpace X
    inst✝⁸ : R1Space X
    inst✝⁷ : TopologicalSpace Y
    inst✝⁶ : MeasurableSpace Y
    inst✝⁵ : BorelSpace Y
    inst✝⁴ : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure ν
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    hp₀ : Ne p 0
    ⊢ ∀ (a : Subtype fun x => Membership.mem (MeasureTheory.Lp E p ν) x), Membersh …
  -/
  intro f hf
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : MeasurableSpace X
    inst✝⁹ : BorelSpace X
    inst✝⁸ : R1Space X
    inst✝⁷ : TopologicalSpace Y
    inst✝⁶ : MeasurableSpace Y
    inst✝⁵ : BorelSpace Y
    inst✝⁴ : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure ν
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    hp₀ : Ne p 0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p ν) x
    hf : Membership.mem (↑(MeasureTheory.Lp.simpleFunc E p ν)) f
    ⊢ Continuous fun y => (MeasureTheory.Lp.compMeasurePreserving ⇑↑{ fst := f, sn …
  -/
  lift f to Lp.simpleFunc E p ν using hf
  induction f using Lp.simpleFunc.induction hp₀ hp with
  | h_add hfp hgp _ ihf ihg => exact ihf.add ihg
  | @h_ind c s hs hνs =>
    dsimp only [Lp.simpleFunc.coe_indicatorConst, Lp.indicatorConstLp_compMeasurePreserving]
    refine continuous_indicatorConstLp_set hp fun f ↦ ?_
    apply tendsto_measure_symmDiff_preimage_nhds_zero continuousAt_subtype_val _ f.2
      hs.nullMeasurableSet hνs.ne
    exact .of_forall Subtype.property


theorem Filter.Tendsto.compMeasurePreservingLp {α : Type*} {l : Filter α}
    {f : α → Lp E p ν} {f₀ : Lp E p ν} {g : α → C(X, Y)} {g₀ : C(X, Y)}
    (hf : Tendsto f l (𝓝 f₀)) (hg : Tendsto g l (𝓝 g₀))
    (hgm : ∀ a, MeasurePreserving (g a) μ ν) (hgm₀ : MeasurePreserving g₀ μ ν) (hp : p ≠ ∞) :
    Tendsto (fun a ↦ Lp.compMeasurePreserving (g a) (hgm a) (f a)) l
      (𝓝 (Lp.compMeasurePreserving g₀ hgm₀ f₀)) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : MeasurableSpace X
    inst✝⁹ : BorelSpace X
    inst✝⁸ : R1Space X
    inst✝⁷ : TopologicalSpace Y
    inst✝⁶ : MeasurableSpace Y
    inst✝⁵ : BorelSpace Y
    inst✝⁴ : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure ν
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    α : Type u_4
    l : Filter α
    f : α → Subtype fun x => Membership.mem (MeasureTheory.Lp E p ν) x
    f₀ : Subtype fun x => Membership.mem (MeasureTheory.Lp E p ν) x
    g : α → ContinuousMap X Y
    g₀ : ContinuousMap X Y
    hf : Filter.Tendsto f l (nhds f₀)
    hg : Filter.Tendsto g l (nhds g₀)
    hgm : ∀ (a : α), MeasureTheory.MeasurePreserving (⇑(g a)) μ ν
    hgm₀ : MeasureTheory.MeasurePreserving (⇑g₀) μ ν
    hp : Ne p Top.top
    ⊢ Filter.Tendsto (fun a => (MeasureTheory.Lp.compMeasurePreserving ⇑(g a) ⋯) ( …
  -/
  have := (Lp.compMeasurePreserving_continuous μ ν E hp).tendsto ⟨f₀, g₀, hgm₀⟩
  replace hg : Tendsto (fun a ↦ ⟨g a, hgm a⟩ : α → {g : C(X, Y) // MeasurePreserving g μ ν})
      l (𝓝 ⟨g₀, hgm₀⟩) :=
    tendsto_subtype_rng.2 hg
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : MeasurableSpace X
    inst✝⁹ : BorelSpace X
    inst✝⁸ : R1Space X
    inst✝⁷ : TopologicalSpace Y
    inst✝⁶ : MeasurableSpace Y
    inst✝⁵ : BorelSpace Y
    inst✝⁴ : R1Space Y
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝³ : μ.InnerRegularCompactLTTop
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure ν
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    α : Type u_4
    l : Filter α
    f : α → Subtype fun x => Membership.mem (MeasureTheory.Lp E p ν) x
    f₀ : Subtype fun x => Membership.mem (MeasureTheory.Lp E p ν) x
    g : α → ContinuousMap X Y
    g₀ : ContinuousMap X Y
    hf : Filter.Tendsto f l (nhds f₀)
    hgm : ∀ (a : α), MeasureTheory.MeasurePreserving (⇑(g a)) μ ν
    hgm₀ : MeasureTheory.MeasurePreserving (⇑g₀) μ ν
    hp : Ne p Top.top
    this : Filter.Tendsto (fun gf => (MeasureTheory.Lp.compMeasurePreserving ⇑↑gf. …
    hg : Filter.Tendsto (fun a => ⟨g a, ⋯⟩) l (nhds ⟨g₀, hgm₀⟩)
    ⊢ Filter.Tendsto (fun a => (MeasureTheory.Lp.compMeasurePreserving ⇑(g a) ⋯) ( …
  -/
  convert this.comp (hf.prod_mk_nhds hg)
  /-
    🎉 no goals
  -/


theorem ContinuousWithinAt.compMeasurePreservingLp (hf : ContinuousWithinAt f s z)
    (hg : ContinuousWithinAt g s z) (hgm : ∀ z, MeasurePreserving (g z) μ ν) (hp : p ≠ ∞) :
    ContinuousWithinAt (fun z ↦ Lp.compMeasurePreserving (g z) (hgm z) (f z)) s z :=
  Tendsto.compMeasurePreservingLp hf hg _ _ hp


theorem ContinuousAt.compMeasurePreservingLp (hf : ContinuousAt f z)
    (hg : ContinuousAt g z) (hgm : ∀ z, MeasurePreserving (g z) μ ν) (hp : p ≠ ∞) :
    ContinuousAt (fun z ↦ Lp.compMeasurePreserving (g z) (hgm z) (f z)) z :=
  Tendsto.compMeasurePreservingLp hf hg _ _ hp


theorem ContinuousOn.compMeasurePreservingLp (hf : ContinuousOn f s)
    (hg : ContinuousOn g s) (hgm : ∀ z, MeasurePreserving (g z) μ ν) (hp : p ≠ ∞) :
    ContinuousOn (fun z ↦ Lp.compMeasurePreserving (g z) (hgm z) (f z)) s := fun z hz ↦
  (hf z hz).compMeasurePreservingLp (hg z hz) hgm hp


theorem Continuous.compMeasurePreservingLp (hf : Continuous f) (hg : Continuous g)
    (hgm : ∀ z, MeasurePreserving (g z) μ ν) (hp : p ≠ ∞) :
    Continuous (fun z ↦ Lp.compMeasurePreserving (g z) (hgm z) (f z)) :=
  continuous_iff_continuousAt.mpr fun _ ↦
    hf.continuousAt.compMeasurePreservingLp hg.continuousAt hgm hp

