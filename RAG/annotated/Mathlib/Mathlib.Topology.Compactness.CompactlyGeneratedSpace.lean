/--
The compactly generated topology on a topological space `X`. This is the finest topology
which makes all maps from compact Hausdorff spaces to `X`, which are continuous for the original
topology, continuous.

Note: this definition should be used with an explicit universe parameter `u` for the size of the
compact Hausdorff spaces mapping to `X`.
-/
def TopologicalSpace.compactlyGenerated (X : Type w) [TopologicalSpace X] : TopologicalSpace X :=
  let f : (Σ (i : (S : CompHaus.{u}) × C(S, X)), i.fst) → X := fun ⟨⟨_, i⟩, s⟩ ↦ i s
  coinduced f inferInstance


lemma continuous_from_compactlyGenerated [TopologicalSpace X] [t : TopologicalSpace Y] (f : X → Y)
    (h : ∀ (S : CompHaus.{u}) (g : C(S, X)), Continuous (f ∘ g)) :
        Continuous[compactlyGenerated.{u} X, t] f := by
  /-
    X : Type w
    Y : Type x
    inst✝ : TopologicalSpace X
    t : TopologicalSpace Y
    f : X → Y
    h : ∀ (S : CompHaus) (g : ContinuousMap (↑S.toTop) X), Continuous (Function.co …
    ⊢ Continuous f
  -/
  rw [continuous_coinduced_dom]
  /-
    X : Type w
    Y : Type x
    inst✝ : TopologicalSpace X
    t : TopologicalSpace Y
    f : X → Y
    h : ∀ (S : CompHaus) (g : ContinuousMap (↑S.toTop) X), Continuous (Function.co …
    ⊢ Continuous (Function.comp f fun x => TopologicalSpace.compactlyGenerated.mat …
  -/
  continuity
  /-
    🎉 no goals
  -/


/--
A topological space `X` is compactly generated if its topology is finer than (and thus equal to)
the compactly generated topology, i.e. it is coinduced by the continuous maps from compact
Hausdorff spaces to `X`.

This version includes an explicit universe parameter `u` which should always be specified. It is
intended for categorical purposes. See `CompactlyGeneratedSpace` for the version without this
parameter, intended for topological purposes.
-/
class UCompactlyGeneratedSpace (X : Type v) [t : TopologicalSpace X] : Prop where
  /-- The topology of `X` is finer than the compactly generated topology. -/
  le_compactlyGenerated : t ≤ compactlyGenerated.{u} X


lemma eq_compactlyGenerated [t : TopologicalSpace X] [UCompactlyGeneratedSpace.{u} X] :
    t = compactlyGenerated.{u} X := by
  /-
    X : Type w
    t : TopologicalSpace X
    inst✝ : UCompactlyGeneratedSpace X
    ⊢ Eq t (TopologicalSpace.compactlyGenerated X)
  -/
  apply le_antisymm
    /-
      case a
      X : Type w
      t : TopologicalSpace X
      inst✝ : UCompactlyGeneratedSpace X
      ⊢ LE.le t (TopologicalSpace.compactlyGenerated X)
    -/
  · exact UCompactlyGeneratedSpace.le_compactlyGenerated
    /-
      🎉 no goals
    -/
  · simp only [compactlyGenerated, ← continuous_iff_coinduced_le, continuous_sigma_iff,
      Sigma.forall]
    /-
      case a
      X : Type w
      t : TopologicalSpace X
      inst✝ : UCompactlyGeneratedSpace X
      ⊢ ∀ (a : CompHaus) (b : ContinuousMap (↑a.toTop) X), Continuous fun a_1 => b a_1
    -/
    exact fun S f ↦ f.2
    /-
      🎉 no goals
    -/


instance (X : Type v) [t : TopologicalSpace X] [DiscreteTopology X] :
    UCompactlyGeneratedSpace.{u} X where
  le_compactlyGenerated := by
    /-
      X✝ : Type w
      Y : Type x
      X : Type v
      t : TopologicalSpace X
      inst✝ : DiscreteTopology X
      ⊢ LE.le t (TopologicalSpace.compactlyGenerated X)
    -/
    rw [DiscreteTopology.eq_bot (t := t)]
    /-
      X✝ : Type w
      Y : Type x
      X : Type v
      t : TopologicalSpace X
      inst✝ : DiscreteTopology X
      ⊢ LE.le Bot.bot (TopologicalSpace.compactlyGenerated X)
    -/
    exact bot_le
    /-
      🎉 no goals
    -/


set_option linter.unusedVariables false in
/-- Let `f : X → Y`. Suppose that to prove that `f` is continuous, it suffices to show that
for every compact Hausdorff space `K` and every continuous map `g : K → X`, `f ∘ g` is continuous.
Then `X` is compactly generated. -/
lemma uCompactlyGeneratedSpace_of_continuous_maps [t : TopologicalSpace X]
    (h : ∀ {Y : Type w} [tY : TopologicalSpace Y] (f : X → Y),
      (∀ (S : CompHaus.{u}) (g : C(S, X)), Continuous (f ∘ g)) → Continuous f) :
        UCompactlyGeneratedSpace.{u} X where
  le_compactlyGenerated := by
    suffices Continuous[t, compactlyGenerated.{u} X] (id : X → X) by
      rwa [← continuous_id_iff_le]
    /-
      X : Type w
      t : TopologicalSpace X
      h : ∀ {Y : Type w} [tY : TopologicalSpace Y] (f : X → Y), (∀ (S : CompHaus) (g …
      ⊢ Continuous id
    -/
    apply h (tY := compactlyGenerated.{u} X)
    /-
      case a
      X : Type w
      t : TopologicalSpace X
      h : ∀ {Y : Type w} [tY : TopologicalSpace Y] (f : X → Y), (∀ (S : CompHaus) (g …
      ⊢ ∀ (S : CompHaus) (g : ContinuousMap (↑S.toTop) X), Continuous (Function.comp …
    -/
    intro S g
    /-
      case a
      X : Type w
      t : TopologicalSpace X
      h : ∀ {Y : Type w} [tY : TopologicalSpace Y] (f : X → Y), (∀ (S : CompHaus) (g …
      S : CompHaus
      g : ContinuousMap (↑S.toTop) X
      ⊢ Continuous (Function.comp id ⇑g)
    -/
    let f : (Σ (i : (T : CompHaus.{u}) × C(T, X)), i.fst) → X := fun ⟨⟨_, i⟩, s⟩ ↦ i s
    suffices ∀ (i : (T : CompHaus.{u}) × C(T, X)),
      Continuous[inferInstance, compactlyGenerated X] (fun (a : i.fst) ↦ f ⟨i, a⟩) from this ⟨S, g⟩
    /-
      case a
      X : Type w
      t : TopologicalSpace X
      h : ∀ {Y : Type w} [tY : TopologicalSpace Y] (f : X → Y), (∀ (S : CompHaus) (g …
      S : CompHaus
      g : ContinuousMap (↑S.toTop) X
      f : (Sigma fun i => ↑i.fst.toTop) → X := fun x => TopologicalSpace.compactlyGe …
      ⊢ ∀ (i : Sigma fun T => ContinuousMap (↑T.toTop) X), Continuous fun a => f ⟨i, …
    -/
    rw [← @continuous_sigma_iff]
    /-
      case a
      X : Type w
      t : TopologicalSpace X
      h : ∀ {Y : Type w} [tY : TopologicalSpace Y] (f : X → Y), (∀ (S : CompHaus) (g …
      S : CompHaus
      g : ContinuousMap (↑S.toTop) X
      f : (Sigma fun i => ↑i.fst.toTop) → X := fun x => TopologicalSpace.compactlyGe …
      ⊢ Continuous f
    -/
    apply continuous_coinduced_rng
    /-
      🎉 no goals
    -/


/-- If `X` is compactly generated, to prove that `f : X → Y` is continuous it is enough to show
that for every compact Hausdorff space `K` and every continuous map `g : K → X`,
`f ∘ g` is continuous. -/
lemma continuous_from_uCompactlyGeneratedSpace [UCompactlyGeneratedSpace.{u} X] (f : X → Y)
    (h : ∀ (S : CompHaus.{u}) (g : C(S, X)), Continuous (f ∘ g)) : Continuous f := by
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : UCompactlyGeneratedSpace X
    f : X → Y
    h : ∀ (S : CompHaus) (g : ContinuousMap (↑S.toTop) X), Continuous (Function.co …
    ⊢ Continuous f
  -/
  apply continuous_le_dom UCompactlyGeneratedSpace.le_compactlyGenerated
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : UCompactlyGeneratedSpace X
    f : X → Y
    h : ∀ (S : CompHaus) (g : ContinuousMap (↑S.toTop) X), Continuous (Function.co …
    ⊢ Continuous f
  -/
  exact continuous_from_compactlyGenerated f h
  /-
    🎉 no goals
  -/


/-- A topological space `X` is compactly generated if a set `s` is closed when `f ⁻¹' s` is
closed for every continuous map `f : K → X`, where `K` is compact Hausdorff. -/
theorem uCompactlyGeneratedSpace_of_isClosed
    (h : ∀ (s : Set X), (∀ (S : CompHaus.{u}) (f : C(S, X)), IsClosed (f ⁻¹' s)) → IsClosed s) :
    UCompactlyGeneratedSpace.{u} X :=
  uCompactlyGeneratedSpace_of_continuous_maps fun _ h' ↦
    continuous_iff_isClosed.2 fun _ hs ↦ h _ fun S g ↦ hs.preimage (h' S g)


/-- A topological space `X` is compactly generated if a set `s` is open when `f ⁻¹' s` is
open for every continuous map `f : K → X`, where `K` is compact Hausdorff. -/
theorem uCompactlyGeneratedSpace_of_isOpen
    (h : ∀ (s : Set X), (∀ (S : CompHaus.{u}) (f : C(S, X)), IsOpen (f ⁻¹' s)) → IsOpen s) :
    UCompactlyGeneratedSpace.{u} X :=
  uCompactlyGeneratedSpace_of_continuous_maps fun _ h' ↦
    continuous_def.2 fun _ hs ↦ h _ fun S g ↦ hs.preimage (h' S g)


/-- In a compactly generated space `X`, a set `s` is closed when `f ⁻¹' s` is
closed for every continuous map `f : K → X`, where `K` is compact Hausdorff. -/
theorem UCompactlyGeneratedSpace.isClosed [UCompactlyGeneratedSpace.{u} X] {s : Set X}
    (hs : ∀ (S : CompHaus.{u}) (f : C(S, X)), IsClosed (f ⁻¹' s)) : IsClosed s := by
  rw [eq_compactlyGenerated (X := X), TopologicalSpace.compactlyGenerated, isClosed_coinduced,
    isClosed_sigma_iff]
  /-
    X : Type w
    tX : TopologicalSpace X
    inst✝ : UCompactlyGeneratedSpace X
    s : Set X
    hs : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) X), IsClosed (Set.preimage …
    ⊢ ∀ (i : Sigma fun S => ContinuousMap (↑S.toTop) X), IsClosed (Set.preimage (S …
  -/
  exact fun ⟨S, f⟩ ↦ hs S f
  /-
    🎉 no goals
  -/


/-- In a compactly generated space `X`, a set `s` is open when `f ⁻¹' s` is
open for every continuous map `f : K → X`, where `K` is compact Hausdorff. -/
theorem UCompactlyGeneratedSpace.isOpen [UCompactlyGeneratedSpace.{u} X] {s : Set X}
    (hs : ∀ (S : CompHaus.{u}) (f : C(S, X)), IsOpen (f ⁻¹' s)) : IsOpen s := by
  rw [eq_compactlyGenerated (X := X), TopologicalSpace.compactlyGenerated, isOpen_coinduced,
    isOpen_sigma_iff]
  /-
    X : Type w
    tX : TopologicalSpace X
    inst✝ : UCompactlyGeneratedSpace X
    s : Set X
    hs : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) X), IsOpen (Set.preimage ( …
    ⊢ ∀ (i : Sigma fun S => ContinuousMap (↑S.toTop) X), IsOpen (Set.preimage (Sig …
  -/
  exact fun ⟨S, f⟩ ↦ hs S f
  /-
    🎉 no goals
  -/


/-- If the topology of `X` is coinduced by a continuous function whose domain is
compactly generated, then so is `X`. -/
theorem uCompactlyGeneratedSpace_of_coinduced
    [UCompactlyGeneratedSpace.{u} X] {f : X → Y} (hf : Continuous f) (ht : tY = coinduced f tX) :
    UCompactlyGeneratedSpace.{u} Y := by
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : UCompactlyGeneratedSpace X
    f : X → Y
    hf : Continuous f
    ht : Eq tY (TopologicalSpace.coinduced f tX)
    ⊢ UCompactlyGeneratedSpace Y
  -/
  refine uCompactlyGeneratedSpace_of_isClosed fun s h ↦ ?_
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : UCompactlyGeneratedSpace X
    f : X → Y
    hf : Continuous f
    ht : Eq tY (TopologicalSpace.coinduced f tX)
    s : Set Y
    h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) Y), IsClosed (Set.preimage  …
    ⊢ IsClosed s
  -/
  rw [ht, isClosed_coinduced]
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : UCompactlyGeneratedSpace X
    f : X → Y
    hf : Continuous f
    ht : Eq tY (TopologicalSpace.coinduced f tX)
    s : Set Y
    h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) Y), IsClosed (Set.preimage  …
    ⊢ IsClosed (Set.preimage f s)
  -/
  exact UCompactlyGeneratedSpace.isClosed fun _ ⟨g, hg⟩ ↦ h _ ⟨_, hf.comp hg⟩
  /-
    🎉 no goals
  -/


/-- The quotient of a compactly generated space is compactly generated. -/
instance {S : Setoid X} [UCompactlyGeneratedSpace.{u} X] :
    UCompactlyGeneratedSpace.{u} (Quotient S) :=
  uCompactlyGeneratedSpace_of_coinduced continuous_quotient_mk' rfl


/-- The sum of two compactly generated spaces is compactly generated. -/
instance [UCompactlyGeneratedSpace.{u} X] [UCompactlyGeneratedSpace.{v} Y] :
    UCompactlyGeneratedSpace.{max u v} (X ⊕ Y) := by
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝¹ : UCompactlyGeneratedSpace X
    inst✝ : UCompactlyGeneratedSpace Y
    ⊢ UCompactlyGeneratedSpace (Sum X Y)
  -/
  refine uCompactlyGeneratedSpace_of_isClosed fun s h ↦ isClosed_sum_iff.2 ⟨?_, ?_⟩
  all_goals
    refine UCompactlyGeneratedSpace.isClosed fun S ⟨f, hf⟩ ↦ ?_
    /-
      case refine_1
      X : Type w
      Y : Type x
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝¹ : UCompactlyGeneratedSpace X
      inst✝ : UCompactlyGeneratedSpace Y
      s : Set (Sum X Y)
      h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) (Sum X Y)), IsClosed (Set.p …
      S : CompHaus
      x✝ : ContinuousMap (↑S.toTop) X
      f : ↑S.toTop → X
      hf : Continuous f
      ⊢ IsClosed (Set.preimage (⇑{ toFun := f, continuous_toFun := hf }) (Set.preima …
    -/
  · let g : ULift.{v} S → X ⊕ Y := Sum.inl ∘ f ∘ ULift.down
    /-
      case refine_1
      X : Type w
      Y : Type x
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝¹ : UCompactlyGeneratedSpace X
      inst✝ : UCompactlyGeneratedSpace Y
      s : Set (Sum X Y)
      h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) (Sum X Y)), IsClosed (Set.p …
      S : CompHaus
      x✝ : ContinuousMap (↑S.toTop) X
      f : ↑S.toTop → X
      hf : Continuous f
      g : ULift.{v, u} ↑S.toTop → Sum X Y := Function.comp Sum.inl (Function.comp f  …
      ⊢ IsClosed (Set.preimage (⇑{ toFun := f, continuous_toFun := hf }) (Set.preima …
    -/
    have hg : Continuous g := continuous_inl.comp <| hf.comp continuous_uLift_down
    /-
      case refine_1
      X : Type w
      Y : Type x
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝¹ : UCompactlyGeneratedSpace X
      inst✝ : UCompactlyGeneratedSpace Y
      s : Set (Sum X Y)
      h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) (Sum X Y)), IsClosed (Set.p …
      S : CompHaus
      x✝ : ContinuousMap (↑S.toTop) X
      f : ↑S.toTop → X
      hf : Continuous f
      g : ULift.{v, u} ↑S.toTop → Sum X Y := Function.comp Sum.inl (Function.comp f  …
      hg : Continuous g
      ⊢ IsClosed (Set.preimage (⇑{ toFun := f, continuous_toFun := hf }) (Set.preima …
    -/
    exact (h (CompHaus.of (ULift.{v} S)) ⟨g, hg⟩).preimage continuous_uLift_up
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type w
      Y : Type x
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝¹ : UCompactlyGeneratedSpace X
      inst✝ : UCompactlyGeneratedSpace Y
      s : Set (Sum X Y)
      h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) (Sum X Y)), IsClosed (Set.p …
      S : CompHaus
      x✝ : ContinuousMap (↑S.toTop) Y
      f : ↑S.toTop → Y
      hf : Continuous f
      ⊢ IsClosed (Set.preimage (⇑{ toFun := f, continuous_toFun := hf }) (Set.preima …
    -/
  · let g : ULift.{u} S → X ⊕ Y := Sum.inr ∘ f ∘ ULift.down
    /-
      case refine_2
      X : Type w
      Y : Type x
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝¹ : UCompactlyGeneratedSpace X
      inst✝ : UCompactlyGeneratedSpace Y
      s : Set (Sum X Y)
      h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) (Sum X Y)), IsClosed (Set.p …
      S : CompHaus
      x✝ : ContinuousMap (↑S.toTop) Y
      f : ↑S.toTop → Y
      hf : Continuous f
      g : ULift.{u, v} ↑S.toTop → Sum X Y := Function.comp Sum.inr (Function.comp f  …
      ⊢ IsClosed (Set.preimage (⇑{ toFun := f, continuous_toFun := hf }) (Set.preima …
    -/
    have hg : Continuous g := continuous_inr.comp <| hf.comp continuous_uLift_down
    /-
      case refine_2
      X : Type w
      Y : Type x
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝¹ : UCompactlyGeneratedSpace X
      inst✝ : UCompactlyGeneratedSpace Y
      s : Set (Sum X Y)
      h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) (Sum X Y)), IsClosed (Set.p …
      S : CompHaus
      x✝ : ContinuousMap (↑S.toTop) Y
      f : ↑S.toTop → Y
      hf : Continuous f
      g : ULift.{u, v} ↑S.toTop → Sum X Y := Function.comp Sum.inr (Function.comp f  …
      hg : Continuous g
      ⊢ IsClosed (Set.preimage (⇑{ toFun := f, continuous_toFun := hf }) (Set.preima …
    -/
    exact (h (CompHaus.of (ULift.{u} S)) ⟨g, hg⟩).preimage continuous_uLift_up
    /-
      🎉 no goals
    -/


/-- The sigma type associated to a family of compactly generated spaces is compactly generated. -/
instance {ι : Type v} {X : ι → Type w} [∀ i, TopologicalSpace (X i)]
    [∀ i, UCompactlyGeneratedSpace.{u} (X i)] : UCompactlyGeneratedSpace.{u} (Σ i, X i) :=
  uCompactlyGeneratedSpace_of_isClosed fun _ h ↦ isClosed_sigma_iff.2 fun i ↦
    UCompactlyGeneratedSpace.isClosed fun S ⟨f, hf⟩ ↦
      h S ⟨Sigma.mk i ∘ f, continuous_sigmaMk.comp hf⟩


open OnePoint in
/-- A sequential space is compactly generated.

The proof is taken from <https://ncatlab.org/nlab/files/StricklandCGHWSpaces.pdf>,
Proposition 1.6. -/
instance (priority := 100) [SequentialSpace X] : UCompactlyGeneratedSpace.{u} X := by
  refine uCompactlyGeneratedSpace_of_isClosed fun s h ↦
    SequentialSpace.isClosed_of_seq _ fun u p hu hup ↦ ?_
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : SequentialSpace X
    s : Set X
    h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) X), IsClosed (Set.preimage  …
    u : Nat → X
    p : X
    hu : ∀ (n : Nat), Membership.mem s (u n)
    hup : Filter.Tendsto u Filter.atTop (nhds p)
    ⊢ Membership.mem s p
  -/
  let g : ULift.{u} (OnePoint ℕ) → X := (continuousMapMkNat u p hup) ∘ ULift.down
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : SequentialSpace X
    s : Set X
    h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) X), IsClosed (Set.preimage  …
    u : Nat → X
    p : X
    hu : ∀ (n : Nat), Membership.mem s (u n)
    hup : Filter.Tendsto u Filter.atTop (nhds p)
    g : ULift.{u, 0} (OnePoint Nat) → X := Function.comp (⇑(OnePoint.continuousMap …
    ⊢ Membership.mem s p
  -/
  change ULift.up ∞ ∈ g ⁻¹' s
  have : Filter.Tendsto (@OnePoint.some ℕ) Filter.atTop (𝓝 ∞) := by
    rw [← Nat.cofinite_eq_atTop, ← cocompact_eq_cofinite, ← coclosedCompact_eq_cocompact]
    exact tendsto_coe_infty
  /-
    X : Type w
    Y : Type x
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    inst✝ : SequentialSpace X
    s : Set X
    h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) X), IsClosed (Set.preimage  …
    u : Nat → X
    p : X
    hu : ∀ (n : Nat), Membership.mem s (u n)
    hup : Filter.Tendsto u Filter.atTop (nhds p)
    g : ULift.{u, 0} (OnePoint Nat) → X := Function.comp (⇑(OnePoint.continuousMap …
    this : Filter.Tendsto OnePoint.some Filter.atTop (nhds OnePoint.infty)
    ⊢ Membership.mem (Set.preimage g s) { down := OnePoint.infty }
  -/
  apply IsClosed.mem_of_tendsto _ ((continuous_uLift_up.tendsto ∞).comp this)
    /-
      X : Type w
      Y : Type x
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝ : SequentialSpace X
      s : Set X
      h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) X), IsClosed (Set.preimage  …
      u : Nat → X
      p : X
      hu : ∀ (n : Nat), Membership.mem s (u n)
      hup : Filter.Tendsto u Filter.atTop (nhds p)
      g : ULift.{u, 0} (OnePoint Nat) → X := Function.comp (⇑(OnePoint.continuousMap …
      this : Filter.Tendsto OnePoint.some Filter.atTop (nhds OnePoint.infty)
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.preimage g s) (Function.comp …
    -/
  · simp only [Function.comp_apply, mem_preimage, eventually_atTop, ge_iff_le]
    /-
      X : Type w
      Y : Type x
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      inst✝ : SequentialSpace X
      s : Set X
      h : ∀ (S : CompHaus) (f : ContinuousMap (↑S.toTop) X), IsClosed (Set.preimage  …
      u : Nat → X
      p : X
      hu : ∀ (n : Nat), Membership.mem s (u n)
      hup : Filter.Tendsto u Filter.atTop (nhds p)
      g : ULift.{u, 0} (OnePoint Nat) → X := Function.comp (⇑(OnePoint.continuousMap …
      this : Filter.Tendsto OnePoint.some Filter.atTop (nhds OnePoint.infty)
      ⊢ Exists fun a => ∀ (b : Nat), LE.le a b → Membership.mem s (g { down := ↑b })
    -/
    exact ⟨0, fun b _ ↦ hu b⟩
    /-
      🎉 no goals
    -/
  · exact h (CompHaus.of (ULift.{u} (OnePoint ℕ)))
      ⟨g, (continuousMapMkNat u p hup).continuous.comp continuous_uLift_down⟩


/--
A topological space `X` is compactly generated if its topology is finer than (and thus equal to)
the compactly generated topology, i.e. it is coinduced by the continuous maps from compact
Hausdorff spaces to `X`.

In this version, intended for topological purposes, the compact spaces are taken
in the same universe as `X`. See `UCompactlyGeneratedSpace` for a version with an explicit
universe parameter, intended for categorical purposes.
-/
abbrev CompactlyGeneratedSpace (X : Type u) [TopologicalSpace X] : Prop :=
  UCompactlyGeneratedSpace.{u} X


/-- If `X` is compactly generated, to prove that `f : X → Y` is continuous it is enough to show
that for every compact Hausdorff space `K` and every continuous map `g : K → X`,
`f ∘ g` is continuous. -/
lemma continuous_from_compactlyGeneratedSpace [CompactlyGeneratedSpace X] (f : X → Y)
    (h : ∀ (K : Type u) [TopologicalSpace K], [CompactSpace K] → [T2Space K] →
      (∀ g : K → X, Continuous g → Continuous (f ∘ g))) : Continuous f :=
  continuous_from_uCompactlyGeneratedSpace f fun K ⟨g, hg⟩ ↦ h K g hg


/-- Let `f : X → Y`. Suppose that to prove that `f` is continuous, it suffices to show that
for every compact Hausdorff space `K` and every continuous map `g : K → X`, `f ∘ g` is continuous.
Then `X` is compactly generated. -/
lemma compactlyGeneratedSpace_of_continuous_maps
    (h : ∀ {Y : Type u} [TopologicalSpace Y] (f : X → Y),
      (∀ (K : Type u) [TopologicalSpace K], [CompactSpace K] → [T2Space K] →
        (∀ g : K → X, Continuous g → Continuous (f ∘ g))) → Continuous f) :
    CompactlyGeneratedSpace X :=
  uCompactlyGeneratedSpace_of_continuous_maps fun f h' ↦ h f fun K _ _ _ g hg ↦
    h' (CompHaus.of K) ⟨g, hg⟩


/-- A topological space `X` is compactly generated if a set `s` is closed when `f ⁻¹' s` is
closed for every continuous map `f : K → X`, where `K` is compact Hausdorff. -/
theorem compactlyGeneratedSpace_of_isClosed
    (h : ∀ (s : Set X), (∀ (K : Type u) [TopologicalSpace K], [CompactSpace K] → [T2Space K] →
      ∀ (f : K → X), Continuous f → IsClosed (f ⁻¹' s)) → IsClosed s) :
    CompactlyGeneratedSpace X :=
  uCompactlyGeneratedSpace_of_isClosed fun s h' ↦ h s fun K _ _ _ f hf ↦ h' (CompHaus.of K) ⟨f, hf⟩


/-- In a compactly generated space `X`, a set `s` is closed when `f ⁻¹' s` is
closed for every continuous map `f : K → X`, where `K` is compact Hausdorff. -/
theorem CompactlyGeneratedSpace.isClosed' [CompactlyGeneratedSpace X] {s : Set X}
    (hs : ∀ (K : Type u) [TopologicalSpace K], [CompactSpace K] → [T2Space K] →
      ∀ (f : K → X), Continuous f → IsClosed (f ⁻¹' s)) : IsClosed s :=
  UCompactlyGeneratedSpace.isClosed fun S ⟨f, hf⟩ ↦ hs S f hf


/-- In a compactly generated space `X`, a set `s` is closed when `s ∩ K` is
closed for every compact set `K`. -/
theorem CompactlyGeneratedSpace.isClosed [CompactlyGeneratedSpace X] {s : Set X}
    (hs : ∀ ⦃K⦄, IsCompact K → IsClosed (s ∩ K)) : IsClosed s := by
   /-
     X : Type u
     inst✝¹ : TopologicalSpace X
     inst✝ : CompactlyGeneratedSpace X
     s : Set X
     hs : ∀ ⦃K : Set X⦄, IsCompact K → IsClosed (Inter.inter s K)
     ⊢ IsClosed s
   -/
   refine isClosed' fun K _ _ _ f hf ↦ ?_
   /-
     X : Type u
     inst✝¹ : TopologicalSpace X
     inst✝ : CompactlyGeneratedSpace X
     s : Set X
     hs : ∀ ⦃K : Set X⦄, IsCompact K → IsClosed (Inter.inter s K)
     K : Type u
     x✝² : TopologicalSpace K
     x✝¹ : CompactSpace K
     x✝ : T2Space K
     f : K → X
     hf : Continuous f
     ⊢ IsClosed (Set.preimage f s)
   -/
   rw [← Set.preimage_inter_range]
   /-
     X : Type u
     inst✝¹ : TopologicalSpace X
     inst✝ : CompactlyGeneratedSpace X
     s : Set X
     hs : ∀ ⦃K : Set X⦄, IsCompact K → IsClosed (Inter.inter s K)
     K : Type u
     x✝² : TopologicalSpace K
     x✝¹ : CompactSpace K
     x✝ : T2Space K
     f : K → X
     hf : Continuous f
     ⊢ IsClosed (Set.preimage f (Inter.inter s (Set.range f)))
   -/
   exact (hs (isCompact_range hf)).preimage hf
   /-
     🎉 no goals
   -/


/-- A topological space `X` is compactly generated if a set `s` is open when `f ⁻¹' s` is
open for every continuous map `f : K → X`, where `K` is compact Hausdorff. -/
theorem compactlyGeneratedSpace_of_isOpen
    (h : ∀ (s : Set X), (∀ (K : Type u) [TopologicalSpace K], [CompactSpace K] → [T2Space K] →
      ∀ (f : K → X), Continuous f → IsOpen (f ⁻¹' s)) → IsOpen s) :
    CompactlyGeneratedSpace X :=
  uCompactlyGeneratedSpace_of_isOpen fun s h' ↦ h s fun K _ _ _ f hf ↦ h' (CompHaus.of K) ⟨f, hf⟩


/-- In a compactly generated space `X`, a set `s` is open when `f ⁻¹' s` is
open for every continuous map `f : K → X`, where `K` is compact Hausdorff. -/
theorem CompactlyGeneratedSpace.isOpen' [CompactlyGeneratedSpace X] {s : Set X}
    (hs : ∀ (K : Type u) [TopologicalSpace K], [CompactSpace K] → [T2Space K] →
      ∀ (f : K → X), Continuous f → IsOpen (f ⁻¹' s)) : IsOpen s :=
  UCompactlyGeneratedSpace.isOpen fun S ⟨f, hf⟩ ↦ hs S f hf


/-- In a compactly generated space `X`, a set `s` is open when `s ∩ K` is
closed for every open set `K`. -/
theorem CompactlyGeneratedSpace.isOpen [CompactlyGeneratedSpace X] {s : Set X}
    (hs : ∀ ⦃K⦄, IsCompact K → IsOpen (s ∩ K)) : IsOpen s := by
   /-
     X : Type u
     inst✝¹ : TopologicalSpace X
     inst✝ : CompactlyGeneratedSpace X
     s : Set X
     hs : ∀ ⦃K : Set X⦄, IsCompact K → IsOpen (Inter.inter s K)
     ⊢ IsOpen s
   -/
   refine isOpen' fun K _ _ _ f hf ↦ ?_
   /-
     X : Type u
     inst✝¹ : TopologicalSpace X
     inst✝ : CompactlyGeneratedSpace X
     s : Set X
     hs : ∀ ⦃K : Set X⦄, IsCompact K → IsOpen (Inter.inter s K)
     K : Type u
     x✝² : TopologicalSpace K
     x✝¹ : CompactSpace K
     x✝ : T2Space K
     f : K → X
     hf : Continuous f
     ⊢ IsOpen (Set.preimage f s)
   -/
   rw [← Set.preimage_inter_range]
   /-
     X : Type u
     inst✝¹ : TopologicalSpace X
     inst✝ : CompactlyGeneratedSpace X
     s : Set X
     hs : ∀ ⦃K : Set X⦄, IsCompact K → IsOpen (Inter.inter s K)
     K : Type u
     x✝² : TopologicalSpace K
     x✝¹ : CompactSpace K
     x✝ : T2Space K
     f : K → X
     hf : Continuous f
     ⊢ IsOpen (Set.preimage f (Inter.inter s (Set.range f)))
   -/
   exact (hs (isCompact_range hf)).preimage hf
   /-
     🎉 no goals
   -/


/-- If the topology of `X` is coinduced by a continuous function whose domain is
compactly generated, then so is `X`. -/
theorem compactlyGeneratedSpace_of_coinduced
    {X : Type u} [tX : TopologicalSpace X] {Y : Type u} [tY : TopologicalSpace Y]
    [CompactlyGeneratedSpace X] {f : X → Y} (hf : Continuous f) (ht : tY = coinduced f tX) :
    CompactlyGeneratedSpace Y := uCompactlyGeneratedSpace_of_coinduced hf ht


/-- The sigma type associated to a family of compactly generated spaces is compactly generated. -/
instance {ι : Type u} {X : ι → Type v}
    [∀ i, TopologicalSpace (X i)] [∀ i, CompactlyGeneratedSpace (X i)] :
    CompactlyGeneratedSpace (Σ i, X i) := by
  refine compactlyGeneratedSpace_of_isClosed fun s h ↦ isClosed_sigma_iff.2 fun i ↦
    CompactlyGeneratedSpace.isClosed' fun K _ _ _ f hf ↦ ?_
  /-
    X✝ : Type u
    Y : Type v
    inst✝³ : TopologicalSpace X✝
    inst✝² : TopologicalSpace Y
    ι : Type u
    X : ι → Type v
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : ∀ (i : ι), CompactlyGeneratedSpace (X i)
    s : Set (Sigma fun i => X i)
    h : ∀ (K : Type (max u v)) [inst : TopologicalSpace K] [inst_1 : CompactSpace  …
    i : ι
    K : Type v
    x✝² : TopologicalSpace K
    x✝¹ : CompactSpace K
    x✝ : T2Space K
    f : K → X i
    hf : Continuous f
    ⊢ IsClosed (Set.preimage f (Set.preimage (Sigma.mk i) s))
  -/
  let g : ULift.{u} K → (Σ i, X i) := Sigma.mk i ∘ f ∘ ULift.down
  /-
    X✝ : Type u
    Y : Type v
    inst✝³ : TopologicalSpace X✝
    inst✝² : TopologicalSpace Y
    ι : Type u
    X : ι → Type v
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : ∀ (i : ι), CompactlyGeneratedSpace (X i)
    s : Set (Sigma fun i => X i)
    h : ∀ (K : Type (max u v)) [inst : TopologicalSpace K] [inst_1 : CompactSpace  …
    i : ι
    K : Type v
    x✝² : TopologicalSpace K
    x✝¹ : CompactSpace K
    x✝ : T2Space K
    f : K → X i
    hf : Continuous f
    g : ULift.{u, v} K → Sigma fun i => X i := Function.comp (Sigma.mk i) (Functio …
    ⊢ IsClosed (Set.preimage f (Set.preimage (Sigma.mk i) s))
  -/
  have hg : Continuous g := continuous_sigmaMk.comp <| hf.comp continuous_uLift_down
  /-
    X✝ : Type u
    Y : Type v
    inst✝³ : TopologicalSpace X✝
    inst✝² : TopologicalSpace Y
    ι : Type u
    X : ι → Type v
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : ∀ (i : ι), CompactlyGeneratedSpace (X i)
    s : Set (Sigma fun i => X i)
    h : ∀ (K : Type (max u v)) [inst : TopologicalSpace K] [inst_1 : CompactSpace  …
    i : ι
    K : Type v
    x✝² : TopologicalSpace K
    x✝¹ : CompactSpace K
    x✝ : T2Space K
    f : K → X i
    hf : Continuous f
    g : ULift.{u, v} K → Sigma fun i => X i := Function.comp (Sigma.mk i) (Functio …
    hg : Continuous g
    ⊢ IsClosed (Set.preimage f (Set.preimage (Sigma.mk i) s))
  -/
  exact (h _ g hg).preimage continuous_uLift_up
  /-
    🎉 no goals
  -/


theorem CompactlyGeneratedSpace.isClosed_iff_of_t2 [CompactlyGeneratedSpace X] (s : Set X) :
    IsClosed s ↔ ∀ ⦃K⦄, IsCompact K → IsClosed (s ∩ K) where
  mp hs _ hK := hs.inter hK.isClosed
  mpr := CompactlyGeneratedSpace.isClosed


/-- Let `s ⊆ X`. Suppose that `X` is Hausdorff, and that to prove that `s` is closed,
it suffices to show that for every compact set `K ⊆ X`, `s ∩ K` is closed.
Then `X` is compactly generated. -/
theorem compactlyGeneratedSpace_of_isClosed_of_t2
    (h : ∀ s, (∀ (K : Set X), IsCompact K → IsClosed (s ∩ K)) → IsClosed s) :
    CompactlyGeneratedSpace X := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    h : ∀ (s : Set X), (∀ (K : Set X), IsCompact K → IsClosed (Inter.inter s K)) → …
    ⊢ CompactlyGeneratedSpace X
  -/
  refine compactlyGeneratedSpace_of_isClosed fun s hs ↦ h s fun K hK ↦ ?_
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    h : ∀ (s : Set X), (∀ (K : Set X), IsCompact K → IsClosed (Inter.inter s K)) → …
    s : Set X
    hs : ∀ (K : Type u) [inst : TopologicalSpace K] [inst_1 : CompactSpace K] [ins …
    K : Set X
    hK : IsCompact K
    ⊢ IsClosed (Inter.inter s K)
  -/
  rw [Set.inter_comm, ← Subtype.image_preimage_coe]
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    h : ∀ (s : Set X), (∀ (K : Set X), IsCompact K → IsClosed (Inter.inter s K)) → …
    s : Set X
    hs : ∀ (K : Type u) [inst : TopologicalSpace K] [inst_1 : CompactSpace K] [ins …
    K : Set X
    hK : IsCompact K
    ⊢ IsClosed (Set.image Subtype.val (Set.preimage Subtype.val s))
  -/
  apply hK.isClosed.isClosedMap_subtype_val
  /-
    case a
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    h : ∀ (s : Set X), (∀ (K : Set X), IsCompact K → IsClosed (Inter.inter s K)) → …
    s : Set X
    hs : ∀ (K : Type u) [inst : TopologicalSpace K] [inst_1 : CompactSpace K] [ins …
    K : Set X
    hK : IsCompact K
    ⊢ IsClosed (Set.preimage Subtype.val s)
  -/
  have : CompactSpace ↑K := isCompact_iff_compactSpace.1 hK
  /-
    case a
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    h : ∀ (s : Set X), (∀ (K : Set X), IsCompact K → IsClosed (Inter.inter s K)) → …
    s : Set X
    hs : ∀ (K : Type u) [inst : TopologicalSpace K] [inst_1 : CompactSpace K] [ins …
    K : Set X
    hK : IsCompact K
    this : CompactSpace ↑K
    ⊢ IsClosed (Set.preimage Subtype.val s)
  -/
  exact hs _ Subtype.val continuous_subtype_val
  /-
    🎉 no goals
  -/


open scoped Set.Notation in
/-- Let `s ⊆ X`. Suppose that `X` is Hausdorff, and that to prove that `s` is open,
it suffices to show that for every compact set `K ⊆ X`, `s ∩ K` is open in `K`.
Then `X` is compactly generated. -/
theorem compactlyGeneratedSpace_of_isOpen_of_t2
    (h : ∀ s, (∀ (K : Set X), IsCompact K → IsOpen (K ↓∩ s)) → IsOpen s) :
    CompactlyGeneratedSpace X := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    h : ∀ (s : Set X), (∀ (K : Set X), IsCompact K → IsOpen (Set.preimage Subtype. …
    ⊢ CompactlyGeneratedSpace X
  -/
  refine compactlyGeneratedSpace_of_isOpen fun s hs ↦ h s fun K hK ↦ ?_
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    h : ∀ (s : Set X), (∀ (K : Set X), IsCompact K → IsOpen (Set.preimage Subtype. …
    s : Set X
    hs : ∀ (K : Type u) [inst : TopologicalSpace K] [inst_1 : CompactSpace K] [ins …
    K : Set X
    hK : IsCompact K
    ⊢ IsOpen (Set.preimage Subtype.val s)
  -/
  have : CompactSpace ↑K := isCompact_iff_compactSpace.1 hK
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    h : ∀ (s : Set X), (∀ (K : Set X), IsCompact K → IsOpen (Set.preimage Subtype. …
    s : Set X
    hs : ∀ (K : Type u) [inst : TopologicalSpace K] [inst_1 : CompactSpace K] [ins …
    K : Set X
    hK : IsCompact K
    this : CompactSpace ↑K
    ⊢ IsOpen (Set.preimage Subtype.val s)
  -/
  exact hs _ Subtype.val continuous_subtype_val
  /-
    🎉 no goals
  -/


/-- A Hausdorff and weakly locally compact space is compactly generated. -/
instance (priority := 100) [WeaklyLocallyCompactSpace X] :
    CompactlyGeneratedSpace X := by
  /-
    X : Type u
    Y : Type v
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : T2Space X
    inst✝ : WeaklyLocallyCompactSpace X
    ⊢ CompactlyGeneratedSpace X
  -/
  refine compactlyGeneratedSpace_of_isClosed_of_t2 fun s h ↦ ?_
  /-
    X : Type u
    Y : Type v
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : T2Space X
    inst✝ : WeaklyLocallyCompactSpace X
    s : Set X
    h : ∀ (K : Set X), IsCompact K → IsClosed (Inter.inter s K)
    ⊢ IsClosed s
  -/
  rw [isClosed_iff_forall_filter]
  /-
    X : Type u
    Y : Type v
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : T2Space X
    inst✝ : WeaklyLocallyCompactSpace X
    s : Set X
    h : ∀ (K : Set X), IsCompact K → IsClosed (Inter.inter s K)
    ⊢ ∀ (x : X) (F : Filter X), F.NeBot → LE.le F (Filter.principal s) → LE.le F ( …
  -/
  intro x ℱ hℱ₁ hℱ₂ hℱ₃
  /-
    X : Type u
    Y : Type v
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : T2Space X
    inst✝ : WeaklyLocallyCompactSpace X
    s : Set X
    h : ∀ (K : Set X), IsCompact K → IsClosed (Inter.inter s K)
    x : X
    ℱ : Filter X
    hℱ₁ : ℱ.NeBot
    hℱ₂ : LE.le ℱ (Filter.principal s)
    hℱ₃ : LE.le ℱ (nhds x)
    ⊢ Membership.mem s x
  -/
  rcases exists_compact_mem_nhds x with ⟨K, hK, K_mem⟩
  exact Set.mem_of_mem_inter_left <| isClosed_iff_forall_filter.1 (h _ hK) x ℱ hℱ₁
    (Filter.inf_principal ▸ le_inf hℱ₂ (le_trans hℱ₃ <| Filter.le_principal_iff.2 K_mem)) hℱ₃


