/-- A sequence of functions `u` is adapted to a filtration `f` if for all `i`,
`u i` is `f i`-measurable. -/
def Adapted (f : Filtration ι m) (u : ι → Ω → β) : Prop :=
  ∀ i : ι, StronglyMeasurable[f i] (u i)


@[to_additive]
protected theorem mul [Mul β] [ContinuousMul β] (hu : Adapted f u) (hv : Adapted f v) :
    Adapted f (u * v) := fun i => (hu i).mul (hv i)


@[to_additive]
protected theorem div [Div β] [ContinuousDiv β] (hu : Adapted f u) (hv : Adapted f v) :
    Adapted f (u / v) := fun i => (hu i).div (hv i)


@[to_additive]
protected theorem inv [Group β] [TopologicalGroup β] (hu : Adapted f u) :
    Adapted f u⁻¹ := fun i => (hu i).inv


protected theorem smul [SMul ℝ β] [ContinuousSMul ℝ β] (c : ℝ) (hu : Adapted f u) :
    Adapted f (c • u) := fun i => (hu i).const_smul c


protected theorem stronglyMeasurable {i : ι} (hf : Adapted f u) : StronglyMeasurable[m] (u i) :=
  (hf i).mono (f.le i)


theorem stronglyMeasurable_le {i j : ι} (hf : Adapted f u) (hij : i ≤ j) :
    StronglyMeasurable[f j] (u i) := (hf i).mono (f.mono hij)


theorem adapted_const (f : Filtration ι m) (x : β) : Adapted f fun _ _ => x := fun _ =>
  stronglyMeasurable_const


theorem adapted_zero [Zero β] (f : Filtration ι m) : Adapted f (0 : ι → Ω → β) := fun i =>
  @stronglyMeasurable_zero Ω β (f i) _ _


theorem Filtration.adapted_natural [MetrizableSpace β] [mβ : MeasurableSpace β] [BorelSpace β]
    {u : ι → Ω → β} (hum : ∀ i, StronglyMeasurable[m] (u i)) :
    Adapted (Filtration.natural u hum) u := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace.MetrizableSpace β
    mβ : MeasurableSpace β
    inst✝ : BorelSpace β
    u : ι → Ω → β
    hum : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    ⊢ MeasureTheory.Adapted (MeasureTheory.Filtration.natural u hum) u
  -/
  intro i
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace.MetrizableSpace β
    mβ : MeasurableSpace β
    inst✝ : BorelSpace β
    u : ι → Ω → β
    hum : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    i : ι
    ⊢ MeasureTheory.StronglyMeasurable (u i)
  -/
  refine StronglyMeasurable.mono ?_ (le_iSup₂_of_le i (le_refl i) le_rfl)
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace.MetrizableSpace β
    mβ : MeasurableSpace β
    inst✝ : BorelSpace β
    u : ι → Ω → β
    hum : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    i : ι
    ⊢ MeasureTheory.StronglyMeasurable (u i)
  -/
  rw [stronglyMeasurable_iff_measurable_separable]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder ι
    inst✝¹ : TopologicalSpace.MetrizableSpace β
    mβ : MeasurableSpace β
    inst✝ : BorelSpace β
    u : ι → Ω → β
    hum : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    i : ι
    ⊢ And (Measurable (u i)) (TopologicalSpace.IsSeparable (Set.range (u i)))
  -/
  exact ⟨measurable_iff_comap_le.2 le_rfl, (hum i).isSeparable_range⟩
  /-
    🎉 no goals
  -/


/-- Progressively measurable process. A sequence of functions `u` is said to be progressively
measurable with respect to a filtration `f` if at each point in time `i`, `u` restricted to
`Set.Iic i × Ω` is measurable with respect to the product `MeasurableSpace` structure where the
σ-algebra used for `Ω` is `f i`.
The usual definition uses the interval `[0,i]`, which we replace by `Set.Iic i`. We recover the
usual definition for index types `ℝ≥0` or `ℕ`. -/
def ProgMeasurable [MeasurableSpace ι] (f : Filtration ι m) (u : ι → Ω → β) : Prop :=
  ∀ i, StronglyMeasurable[Subtype.instMeasurableSpace.prod (f i)] fun p : Set.Iic i × Ω => u p.1 p.2


theorem progMeasurable_const [MeasurableSpace ι] (f : Filtration ι m) (b : β) :
    ProgMeasurable f (fun _ _ => b : ι → Ω → β) := fun i =>
  @stronglyMeasurable_const _ _ (Subtype.instMeasurableSpace.prod (f i)) _ _


protected theorem adapted (h : ProgMeasurable f u) : Adapted f u := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝² : TopologicalSpace β
    inst✝¹ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    inst✝ : MeasurableSpace ι
    h : MeasureTheory.ProgMeasurable f u
    ⊢ MeasureTheory.Adapted f u
  -/
  intro i
  have : u i = (fun p : Set.Iic i × Ω => u p.1 p.2) ∘ fun x => (⟨i, Set.mem_Iic.mpr le_rfl⟩, x) :=
    rfl
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝² : TopologicalSpace β
    inst✝¹ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    inst✝ : MeasurableSpace ι
    h : MeasureTheory.ProgMeasurable f u
    i : ι
    this : Eq (u i) (Function.comp (fun p => u (↑p.1) p.2) fun x => { fst := ⟨i, ⋯ …
    ⊢ MeasureTheory.StronglyMeasurable (u i)
  -/
  rw [this]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝² : TopologicalSpace β
    inst✝¹ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    inst✝ : MeasurableSpace ι
    h : MeasureTheory.ProgMeasurable f u
    i : ι
    this : Eq (u i) (Function.comp (fun p => u (↑p.1) p.2) fun x => { fst := ⟨i, ⋯ …
    ⊢ MeasureTheory.StronglyMeasurable (Function.comp (fun p => u (↑p.1) p.2) fun  …
  -/
  exact (h i).comp_measurable measurable_prod_mk_left
  /-
    🎉 no goals
  -/


protected theorem comp {t : ι → Ω → ι} [TopologicalSpace ι] [BorelSpace ι] [MetrizableSpace ι]
    (h : ProgMeasurable f u) (ht : ProgMeasurable f t) (ht_le : ∀ i ω, t i ω ≤ i) :
    ProgMeasurable f fun i ω => u (t i ω) ω := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    inst✝³ : MeasurableSpace ι
    t : ι → Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : BorelSpace ι
    inst✝ : TopologicalSpace.MetrizableSpace ι
    h : MeasureTheory.ProgMeasurable f u
    ht : MeasureTheory.ProgMeasurable f t
    ht_le : ∀ (i : ι) (ω : Ω), LE.le (t i ω) i
    ⊢ MeasureTheory.ProgMeasurable f fun i ω => u (t i ω) ω
  -/
  intro i
  have : (fun p : ↥(Set.Iic i) × Ω => u (t (p.fst : ι) p.snd) p.snd) =
    (fun p : ↥(Set.Iic i) × Ω => u (p.fst : ι) p.snd) ∘ fun p : ↥(Set.Iic i) × Ω =>
      (⟨t (p.fst : ι) p.snd, Set.mem_Iic.mpr ((ht_le _ _).trans p.fst.prop)⟩, p.snd) := rfl
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    inst✝³ : MeasurableSpace ι
    t : ι → Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : BorelSpace ι
    inst✝ : TopologicalSpace.MetrizableSpace ι
    h : MeasureTheory.ProgMeasurable f u
    ht : MeasureTheory.ProgMeasurable f t
    ht_le : ∀ (i : ι) (ω : Ω), LE.le (t i ω) i
    i : ι
    this : Eq (fun p => u (t (↑p.1) p.2) p.2) (Function.comp (fun p => u (↑p.1) p. …
    ⊢ MeasureTheory.StronglyMeasurable fun p => (fun i ω => u (t i ω) ω) (↑p.1) p.2
  -/
  rw [this]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    inst✝³ : MeasurableSpace ι
    t : ι → Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : BorelSpace ι
    inst✝ : TopologicalSpace.MetrizableSpace ι
    h : MeasureTheory.ProgMeasurable f u
    ht : MeasureTheory.ProgMeasurable f t
    ht_le : ∀ (i : ι) (ω : Ω), LE.le (t i ω) i
    i : ι
    this : Eq (fun p => u (t (↑p.1) p.2) p.2) (Function.comp (fun p => u (↑p.1) p. …
    ⊢ MeasureTheory.StronglyMeasurable (Function.comp (fun p => u (↑p.1) p.2) fun  …
  -/
  exact (h i).comp_measurable ((ht i).measurable.subtype_mk.prod_mk measurable_snd)
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem mul [Mul β] [ContinuousMul β] (hu : ProgMeasurable f u)
    (hv : ProgMeasurable f v) : ProgMeasurable f fun i ω => u i ω * v i ω := fun i =>
  (hu i).mul (hv i)


@[to_additive]
protected theorem finset_prod' {γ} [CommMonoid β] [ContinuousMul β] {U : γ → ι → Ω → β}
    {s : Finset γ} (h : ∀ c ∈ s, ProgMeasurable f (U c)) : ProgMeasurable f (∏ c ∈ s, U c) :=
  Finset.prod_induction U (ProgMeasurable f) (fun _ _ => ProgMeasurable.mul)
    (progMeasurable_const _ 1) h


@[to_additive]
protected theorem finset_prod {γ} [CommMonoid β] [ContinuousMul β] {U : γ → ι → Ω → β}
    {s : Finset γ} (h : ∀ c ∈ s, ProgMeasurable f (U c)) :
    ProgMeasurable f fun i a => ∏ c ∈ s, U c i a := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁴ : TopologicalSpace β
    inst✝³ : Preorder ι
    f : MeasureTheory.Filtration ι m
    inst✝² : MeasurableSpace ι
    γ : Type u_4
    inst✝¹ : CommMonoid β
    inst✝ : ContinuousMul β
    U : γ → ι → Ω → β
    s : Finset γ
    h : ∀ (c : γ), Membership.mem s c → MeasureTheory.ProgMeasurable f (U c)
    ⊢ MeasureTheory.ProgMeasurable f fun i a => s.prod fun c => U c i a
  -/
  convert ProgMeasurable.finset_prod' h using 1; ext (i a); simp only [Finset.prod_apply]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive]
protected theorem inv [Group β] [TopologicalGroup β] (hu : ProgMeasurable f u) :
    ProgMeasurable f fun i ω => (u i ω)⁻¹ := fun i => (hu i).inv


@[to_additive]
protected theorem div [Group β] [TopologicalGroup β] (hu : ProgMeasurable f u)
    (hv : ProgMeasurable f v) : ProgMeasurable f fun i ω => u i ω / v i ω := fun i =>
  (hu i).div (hv i)


theorem progMeasurable_of_tendsto' {γ} [MeasurableSpace ι] [PseudoMetrizableSpace β]
    (fltr : Filter γ) [fltr.NeBot] [fltr.IsCountablyGenerated] {U : γ → ι → Ω → β}
    (h : ∀ l, ProgMeasurable f (U l)) (h_tendsto : Tendsto U fltr (𝓝 u)) : ProgMeasurable f u := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    γ : Type u_4
    inst✝³ : MeasurableSpace ι
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    fltr : Filter γ
    inst✝¹ : fltr.NeBot
    inst✝ : fltr.IsCountablyGenerated
    U : γ → ι → Ω → β
    h : ∀ (l : γ), MeasureTheory.ProgMeasurable f (U l)
    h_tendsto : Filter.Tendsto U fltr (nhds u)
    ⊢ MeasureTheory.ProgMeasurable f u
  -/
  intro i
  apply @stronglyMeasurable_of_tendsto (Set.Iic i × Ω) β γ
    (MeasurableSpace.prod _ (f i)) _ _ fltr _ _ _ _ fun l => h l i
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    γ : Type u_4
    inst✝³ : MeasurableSpace ι
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    fltr : Filter γ
    inst✝¹ : fltr.NeBot
    inst✝ : fltr.IsCountablyGenerated
    U : γ → ι → Ω → β
    h : ∀ (l : γ), MeasureTheory.ProgMeasurable f (U l)
    h_tendsto : Filter.Tendsto U fltr (nhds u)
    i : ι
    ⊢ Filter.Tendsto (fun l p => U l (↑p.1) p.2) fltr (nhds fun p => u (↑p.1) p.2)
  -/
  rw [tendsto_pi_nhds] at h_tendsto ⊢
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    γ : Type u_4
    inst✝³ : MeasurableSpace ι
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    fltr : Filter γ
    inst✝¹ : fltr.NeBot
    inst✝ : fltr.IsCountablyGenerated
    U : γ → ι → Ω → β
    h : ∀ (l : γ), MeasureTheory.ProgMeasurable f (U l)
    h_tendsto : ∀ (x : ι), Filter.Tendsto (fun i => U i x) fltr (nhds (u x))
    i : ι
    ⊢ ∀ (x : Prod (↑(Set.Iic i)) Ω), Filter.Tendsto (fun i_1 => U i_1 (↑x.1) x.2)  …
  -/
  intro x
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    γ : Type u_4
    inst✝³ : MeasurableSpace ι
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    fltr : Filter γ
    inst✝¹ : fltr.NeBot
    inst✝ : fltr.IsCountablyGenerated
    U : γ → ι → Ω → β
    h : ∀ (l : γ), MeasureTheory.ProgMeasurable f (U l)
    h_tendsto : ∀ (x : ι), Filter.Tendsto (fun i => U i x) fltr (nhds (u x))
    i : ι
    x : Prod (↑(Set.Iic i)) Ω
    ⊢ Filter.Tendsto (fun i_1 => U i_1 (↑x.1) x.2) fltr (nhds (u (↑x.1) x.2))
  -/
  specialize h_tendsto x.fst
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    γ : Type u_4
    inst✝³ : MeasurableSpace ι
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    fltr : Filter γ
    inst✝¹ : fltr.NeBot
    inst✝ : fltr.IsCountablyGenerated
    U : γ → ι → Ω → β
    h : ∀ (l : γ), MeasureTheory.ProgMeasurable f (U l)
    i : ι
    x : Prod (↑(Set.Iic i)) Ω
    h_tendsto : Filter.Tendsto (fun i_1 => U i_1 ↑x.1) fltr (nhds (u ↑x.1))
    ⊢ Filter.Tendsto (fun i_1 => U i_1 (↑x.1) x.2) fltr (nhds (u (↑x.1) x.2))
  -/
  rw [tendsto_nhds] at h_tendsto ⊢
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Preorder ι
    u : ι → Ω → β
    f : MeasureTheory.Filtration ι m
    γ : Type u_4
    inst✝³ : MeasurableSpace ι
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    fltr : Filter γ
    inst✝¹ : fltr.NeBot
    inst✝ : fltr.IsCountablyGenerated
    U : γ → ι → Ω → β
    h : ∀ (l : γ), MeasureTheory.ProgMeasurable f (U l)
    i : ι
    x : Prod (↑(Set.Iic i)) Ω
    h_tendsto : ∀ (s : Set (Ω → β)), IsOpen s → Membership.mem s (u ↑x.1) → Member …
    ⊢ ∀ (s : Set β), IsOpen s → Membership.mem s (u (↑x.1) x.2) → Membership.mem f …
  -/
  exact fun s hs h_mem => h_tendsto {g | g x.snd ∈ s} (hs.preimage (continuous_apply x.snd)) h_mem
  /-
    🎉 no goals
  -/


theorem progMeasurable_of_tendsto [MeasurableSpace ι] [PseudoMetrizableSpace β] {U : ℕ → ι → Ω → β}
    (h : ∀ l, ProgMeasurable f (U l)) (h_tendsto : Tendsto U atTop (𝓝 u)) : ProgMeasurable f u :=
  progMeasurable_of_tendsto' atTop h h_tendsto


/-- A continuous and adapted process is progressively measurable. -/
theorem Adapted.progMeasurable_of_continuous [TopologicalSpace ι] [MetrizableSpace ι]
    [SecondCountableTopology ι] [MeasurableSpace ι] [OpensMeasurableSpace ι]
    [PseudoMetrizableSpace β] (h : Adapted f u) (hu_cont : ∀ ω, Continuous fun i => u i ω) :
    ProgMeasurable f u := fun i =>
  @stronglyMeasurable_uncurry_of_continuous_of_stronglyMeasurable _ _ (Set.Iic i) _ _ _ _ _ _ _
    (f i) _ (fun ω => (hu_cont ω).comp continuous_induced_dom) fun j => (h j).mono (f.mono j.prop)


/-- For filtrations indexed by a discrete order, `Adapted` and `ProgMeasurable` are equivalent.
This lemma provides `Adapted f u → ProgMeasurable f u`.
See `ProgMeasurable.adapted` for the reverse direction, which is true more generally. -/
theorem Adapted.progMeasurable_of_discrete [TopologicalSpace ι] [DiscreteTopology ι]
    [SecondCountableTopology ι] [MeasurableSpace ι] [OpensMeasurableSpace ι]
    [PseudoMetrizableSpace β] (h : Adapted f u) : ProgMeasurable f u :=
  h.progMeasurable_of_continuous fun _ => continuous_of_discreteTopology

-- this dot notation will make more sense once we have a more general definition for predictable

theorem Predictable.adapted {f : Filtration ℕ m} {u : ℕ → Ω → β} (hu : Adapted f fun n => u (n + 1))
    (hu0 : StronglyMeasurable[f 0] (u 0)) : Adapted f u := fun n =>
  match n with
  | 0 => hu0
  | n + 1 => (hu n).mono (f.mono n.le_succ)


