/-- Two functions defined on two (possibly different) measure spaces are identically distributed if
their image measures coincide. This only makes sense when the functions are ae measurable
(as otherwise the image measures are not defined), so we require this as well in the definition. -/
structure IdentDistrib (f : α → γ) (g : β → γ)
    (μ : Measure α := by volume_tac)
    (ν : Measure β := by volume_tac) : Prop where
  aemeasurable_fst : AEMeasurable f μ
  aemeasurable_snd : AEMeasurable g ν
  map_eq : Measure.map f μ = Measure.map g ν


protected theorem refl (hf : AEMeasurable f μ) : IdentDistrib f f μ μ :=
  { aemeasurable_fst := hf
    aemeasurable_snd := hf
    map_eq := rfl }


protected theorem symm (h : IdentDistrib f g μ ν) : IdentDistrib g f ν μ :=
  { aemeasurable_fst := h.aemeasurable_snd
    aemeasurable_snd := h.aemeasurable_fst
    map_eq := h.map_eq.symm }


protected theorem trans {ρ : Measure δ} {h : δ → γ} (h₁ : IdentDistrib f g μ ν)
    (h₂ : IdentDistrib g h ν ρ) : IdentDistrib f h μ ρ :=
  { aemeasurable_fst := h₁.aemeasurable_fst
    aemeasurable_snd := h₂.aemeasurable_snd
    map_eq := h₁.map_eq.trans h₂.map_eq }


protected theorem comp_of_aemeasurable {u : γ → δ} (h : IdentDistrib f g μ ν)
    (hu : AEMeasurable u (Measure.map f μ)) : IdentDistrib (u ∘ f) (u ∘ g) μ ν :=
  { aemeasurable_fst := hu.comp_aemeasurable h.aemeasurable_fst
                           /-
                             α : Type u_1
                             β : Type u_2
                             γ : Type u_3
                             δ : Type u_4
                             inst✝³ : MeasurableSpace α
                             inst✝² : MeasurableSpace β
                             inst✝¹ : MeasurableSpace γ
                             inst✝ : MeasurableSpace δ
                             μ : MeasureTheory.Measure α
                             ν : MeasureTheory.Measure β
                             f : α → γ
                             g : β → γ
                             u : γ → δ
                             h : ProbabilityTheory.IdentDistrib f g μ ν
                             hu : AEMeasurable u (MeasureTheory.Measure.map f μ)
                             ⊢ AEMeasurable (Function.comp u g) ν
                           -/
    aemeasurable_snd := by rw [h.map_eq] at hu; exact hu.comp_aemeasurable h.aemeasurable_snd
                                                /-
                                                  🎉 no goals
                                                -/
    map_eq := by
      rw [← AEMeasurable.map_map_of_aemeasurable hu h.aemeasurable_fst, ←
        AEMeasurable.map_map_of_aemeasurable _ h.aemeasurable_snd, h.map_eq]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        inst✝³ : MeasurableSpace α
        inst✝² : MeasurableSpace β
        inst✝¹ : MeasurableSpace γ
        inst✝ : MeasurableSpace δ
        μ : MeasureTheory.Measure α
        ν : MeasureTheory.Measure β
        f : α → γ
        g : β → γ
        u : γ → δ
        h : ProbabilityTheory.IdentDistrib f g μ ν
        hu : AEMeasurable u (MeasureTheory.Measure.map f μ)
        ⊢ AEMeasurable u (MeasureTheory.Measure.map g ν)
      -/
      rwa [← h.map_eq] }
      /-
        🎉 no goals
      -/


protected theorem comp {u : γ → δ} (h : IdentDistrib f g μ ν) (hu : Measurable u) :
    IdentDistrib (u ∘ f) (u ∘ g) μ ν :=
  h.comp_of_aemeasurable hu.aemeasurable


protected theorem of_ae_eq {g : α → γ} (hf : AEMeasurable f μ) (heq : f =ᵐ[μ] g) :
    IdentDistrib f g μ μ :=
  { aemeasurable_fst := hf
    aemeasurable_snd := hf.congr heq
    map_eq := Measure.map_congr heq }


lemma _root_.MeasureTheory.AEMeasurable.identDistrib_mk
    (hf : AEMeasurable f μ) : IdentDistrib f (hf.mk f) μ μ :=
  IdentDistrib.of_ae_eq hf hf.ae_eq_mk


lemma _root_.MeasureTheory.AEStronglyMeasurable.identDistrib_mk
    [TopologicalSpace γ] [PseudoMetrizableSpace γ] [BorelSpace γ]
    (hf : AEStronglyMeasurable f μ) : IdentDistrib f (hf.mk f) μ μ :=
  IdentDistrib.of_ae_eq hf.aemeasurable hf.ae_eq_mk


theorem measure_mem_eq (h : IdentDistrib f g μ ν) {s : Set γ} (hs : MeasurableSet s) :
    μ (f ⁻¹' s) = ν (g ⁻¹' s) := by
  rw [← Measure.map_apply_of_aemeasurable h.aemeasurable_fst hs, ←
    Measure.map_apply_of_aemeasurable h.aemeasurable_snd hs, h.map_eq]


alias measure_preimage_eq := measure_mem_eq


theorem ae_snd (h : IdentDistrib f g μ ν) {p : γ → Prop} (pmeas : MeasurableSet {x | p x})
    (hp : ∀ᵐ x ∂μ, p (f x)) : ∀ᵐ x ∂ν, p (g x) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    p : γ → Prop
    pmeas : MeasurableSet (setOf fun x => p x)
    hp : Filter.Eventually (fun x => p (f x)) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun x => p (g x)) (MeasureTheory.ae ν)
  -/
  apply (ae_map_iff h.aemeasurable_snd pmeas).1
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    p : γ → Prop
    pmeas : MeasurableSet (setOf fun x => p x)
    hp : Filter.Eventually (fun x => p (f x)) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun y => p y) (MeasureTheory.ae (MeasureTheory.Measure.ma …
  -/
  rw [← h.map_eq]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    p : γ → Prop
    pmeas : MeasurableSet (setOf fun x => p x)
    hp : Filter.Eventually (fun x => p (f x)) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun y => p y) (MeasureTheory.ae (MeasureTheory.Measure.ma …
  -/
  exact (ae_map_iff h.aemeasurable_fst pmeas).2 hp
  /-
    🎉 no goals
  -/


theorem ae_mem_snd (h : IdentDistrib f g μ ν) {t : Set γ} (tmeas : MeasurableSet t)
    (ht : ∀ᵐ x ∂μ, f x ∈ t) : ∀ᵐ x ∂ν, g x ∈ t :=
  h.ae_snd tmeas ht


/-- In a second countable topology, the first function in an identically distributed pair is a.e.
strongly measurable. So is the second function, but use `h.symm.aestronglyMeasurable_fst` as
`h.aestronglyMeasurable_snd` has a different meaning. -/
theorem aestronglyMeasurable_fst [TopologicalSpace γ] [MetrizableSpace γ] [OpensMeasurableSpace γ]
    [SecondCountableTopology γ] (h : IdentDistrib f g μ ν) : AEStronglyMeasurable f μ :=
  h.aemeasurable_fst.aestronglyMeasurable


/-- If `f` and `g` are identically distributed and `f` is a.e. strongly measurable, so is `g`. -/
theorem aestronglyMeasurable_snd [TopologicalSpace γ] [MetrizableSpace γ] [BorelSpace γ]
    (h : IdentDistrib f g μ ν) (hf : AEStronglyMeasurable f μ) : AEStronglyMeasurable g ν := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : TopologicalSpace.MetrizableSpace γ
    inst✝ : BorelSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable g ν
  -/
  refine aestronglyMeasurable_iff_aemeasurable_separable.2 ⟨h.aemeasurable_snd, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : TopologicalSpace.MetrizableSpace γ
    inst✝ : BorelSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
  -/
  rcases (aestronglyMeasurable_iff_aemeasurable_separable.1 hf).2 with ⟨t, t_sep, ht⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : TopologicalSpace.MetrizableSpace γ
    inst✝ : BorelSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.AEStronglyMeasurable f μ
    t : Set γ
    t_sep : TopologicalSpace.IsSeparable t
    ht : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
  -/
  refine ⟨closure t, t_sep.closure, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : TopologicalSpace.MetrizableSpace γ
    inst✝ : BorelSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.AEStronglyMeasurable f μ
    t : Set γ
    t_sep : TopologicalSpace.IsSeparable t
    ht : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun x => Membership.mem (closure t) (g x)) (MeasureTheory …
  -/
  apply h.ae_mem_snd isClosed_closure.measurableSet
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : TopologicalSpace.MetrizableSpace γ
    inst✝ : BorelSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.AEStronglyMeasurable f μ
    t : Set γ
    t_sep : TopologicalSpace.IsSeparable t
    ht : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun x => Membership.mem (closure t) (f x)) (MeasureTheory …
  -/
  filter_upwards [ht] with x hx using subset_closure hx
  /-
    🎉 no goals
  -/


theorem aestronglyMeasurable_iff [TopologicalSpace γ] [MetrizableSpace γ] [BorelSpace γ]
    (h : IdentDistrib f g μ ν) : AEStronglyMeasurable f μ ↔ AEStronglyMeasurable g ν :=
  ⟨fun hf => h.aestronglyMeasurable_snd hf, fun hg => h.symm.aestronglyMeasurable_snd hg⟩


theorem essSup_eq [ConditionallyCompleteLinearOrder γ] [TopologicalSpace γ] [OpensMeasurableSpace γ]
    [OrderClosedTopology γ] (h : IdentDistrib f g μ ν) : essSup f μ = essSup g ν := by
  have I : ∀ a, μ {x : α | a < f x} = ν {x : β | a < g x} := fun a =>
    h.measure_mem_eq measurableSet_Ioi
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝³ : ConditionallyCompleteLinearOrder γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : OpensMeasurableSpace γ
    inst✝ : OrderClosedTopology γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    I : ∀ (a : γ), Eq (μ (setOf fun x => LT.lt a (f x))) (ν (setOf fun x => LT.lt  …
    ⊢ Eq (essSup f μ) (essSup g ν)
  -/
  simp_rw [essSup_eq_sInf, I]
  /-
    🎉 no goals
  -/


theorem lintegral_eq {f : α → ℝ≥0∞} {g : β → ℝ≥0∞} (h : IdentDistrib f g μ ν) :
    ∫⁻ x, f x ∂μ = ∫⁻ x, g x ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → ENNReal
    g : β → ENNReal
    h : ProbabilityTheory.IdentDistrib f g μ ν
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f x) (MeasureTheory.lintegral ν fun x …
  -/
  change ∫⁻ x, id (f x) ∂μ = ∫⁻ x, id (g x) ∂ν
  rw [← lintegral_map' aemeasurable_id h.aemeasurable_fst, ←
    lintegral_map' aemeasurable_id h.aemeasurable_snd, h.map_eq]


theorem integral_eq [NormedAddCommGroup γ] [NormedSpace ℝ γ] [BorelSpace γ]
    (h : IdentDistrib f g μ ν) : ∫ x, f x ∂μ = ∫ x, g x ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝² : NormedAddCommGroup γ
    inst✝¹ : NormedSpace Real γ
    inst✝ : BorelSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral ν fun x = …
  -/
  by_cases hf : AEStronglyMeasurable f μ
  · have A : AEStronglyMeasurable id (Measure.map f μ) := by
      rw [aestronglyMeasurable_iff_aemeasurable_separable]
      rcases (aestronglyMeasurable_iff_aemeasurable_separable.1 hf).2 with ⟨t, t_sep, ht⟩
      refine ⟨aemeasurable_id, ⟨closure t, t_sep.closure, ?_⟩⟩
      rw [ae_map_iff h.aemeasurable_fst]
      · filter_upwards [ht] with x hx using subset_closure hx
      · exact isClosed_closure.measurableSet
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝² : NormedAddCommGroup γ
      inst✝¹ : NormedSpace Real γ
      inst✝ : BorelSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      hf : MeasureTheory.AEStronglyMeasurable f μ
      A : MeasureTheory.AEStronglyMeasurable id (MeasureTheory.Measure.map f μ)
      ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral ν fun x = …
    -/
    change ∫ x, id (f x) ∂μ = ∫ x, id (g x) ∂ν
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝² : NormedAddCommGroup γ
      inst✝¹ : NormedSpace Real γ
      inst✝ : BorelSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      hf : MeasureTheory.AEStronglyMeasurable f μ
      A : MeasureTheory.AEStronglyMeasurable id (MeasureTheory.Measure.map f μ)
      ⊢ Eq (MeasureTheory.integral μ fun x => id (f x)) (MeasureTheory.integral ν fu …
    -/
    rw [← integral_map h.aemeasurable_fst A]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝² : NormedAddCommGroup γ
      inst✝¹ : NormedSpace Real γ
      inst✝ : BorelSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      hf : MeasureTheory.AEStronglyMeasurable f μ
      A : MeasureTheory.AEStronglyMeasurable id (MeasureTheory.Measure.map f μ)
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map f μ) fun y => id y) (M …
    -/
    rw [h.map_eq] at A
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝² : NormedAddCommGroup γ
      inst✝¹ : NormedSpace Real γ
      inst✝ : BorelSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      hf : MeasureTheory.AEStronglyMeasurable f μ
      A : MeasureTheory.AEStronglyMeasurable id (MeasureTheory.Measure.map g ν)
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map f μ) fun y => id y) (M …
    -/
    rw [← integral_map h.aemeasurable_snd A, h.map_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝² : NormedAddCommGroup γ
      inst✝¹ : NormedSpace Real γ
      inst✝ : BorelSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      hf : Not (MeasureTheory.AEStronglyMeasurable f μ)
      ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral ν fun x = …
    -/
  · rw [integral_non_aestronglyMeasurable hf]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝² : NormedAddCommGroup γ
      inst✝¹ : NormedSpace Real γ
      inst✝ : BorelSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      hf : Not (MeasureTheory.AEStronglyMeasurable f μ)
      ⊢ Eq 0 (MeasureTheory.integral ν fun x => g x)
    -/
    rw [h.aestronglyMeasurable_iff] at hf
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝² : NormedAddCommGroup γ
      inst✝¹ : NormedSpace Real γ
      inst✝ : BorelSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      hf : Not (MeasureTheory.AEStronglyMeasurable g ν)
      ⊢ Eq 0 (MeasureTheory.integral ν fun x => g x)
    -/
    rw [integral_non_aestronglyMeasurable hf]
    /-
      🎉 no goals
    -/


theorem eLpNorm_eq [NormedAddCommGroup γ] [OpensMeasurableSpace γ] (h : IdentDistrib f g μ ν)
    (p : ℝ≥0∞) : eLpNorm f p μ = eLpNorm g p ν := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : OpensMeasurableSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    p : ENNReal
    ⊢ Eq (MeasureTheory.eLpNorm f p μ) (MeasureTheory.eLpNorm g p ν)
  -/
  by_cases h0 : p = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝¹ : NormedAddCommGroup γ
      inst✝ : OpensMeasurableSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      p : ENNReal
      h0 : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm f p μ) (MeasureTheory.eLpNorm g p ν)
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : OpensMeasurableSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    p : ENNReal
    h0 : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm f p μ) (MeasureTheory.eLpNorm g p ν)
  -/
  by_cases h_top : p = ∞
  · simp only [h_top, eLpNorm, eLpNormEssSup, ENNReal.top_ne_zero, eq_self_iff_true, if_true,
      if_false]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝¹ : NormedAddCommGroup γ
      inst✝ : OpensMeasurableSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      p : ENNReal
      h0 : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ Eq (essSup (fun x => ENorm.enorm (f x)) μ) (essSup (fun x => ENorm.enorm (g  …
    -/
    apply essSup_eq
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      f : α → γ
      g : β → γ
      inst✝¹ : NormedAddCommGroup γ
      inst✝ : OpensMeasurableSpace γ
      h : ProbabilityTheory.IdentDistrib f g μ ν
      p : ENNReal
      h0 : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ ProbabilityTheory.IdentDistrib (fun x => ENorm.enorm (f x)) (fun x => ENorm. …
    -/
    exact h.comp (measurable_coe_nnreal_ennreal.comp measurable_nnnorm)
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : OpensMeasurableSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    p : ENNReal
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm f p μ) (MeasureTheory.eLpNorm g p ν)
  -/
  simp only [eLpNorm_eq_eLpNorm' h0 h_top, eLpNorm', one_div]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : OpensMeasurableSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    p : ENNReal
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (ENorm.enorm (f  …
  -/
  congr 1
  /-
    case neg.e_a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : OpensMeasurableSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    p : ENNReal
    h0 : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (ENorm.enorm (f a)) p.toRea …
  -/
  apply lintegral_eq
  exact h.comp (Measurable.pow_const (measurable_coe_nnreal_ennreal.comp measurable_nnnorm)
    p.toReal)


@[deprecated (since := "2024-07-27")]
alias snorm_eq := eLpNorm_eq


theorem memℒp_snd [NormedAddCommGroup γ] [BorelSpace γ] {p : ℝ≥0∞} (h : IdentDistrib f g μ ν)
    (hf : Memℒp f p μ) : Memℒp g p ν := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : BorelSpace γ
    p : ENNReal
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.Memℒp f p μ
    ⊢ MeasureTheory.Memℒp g p ν
  -/
  refine ⟨h.aestronglyMeasurable_snd hf.aestronglyMeasurable, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : BorelSpace γ
    p : ENNReal
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.Memℒp f p μ
    ⊢ LT.lt (MeasureTheory.eLpNorm g p ν) Top.top
  -/
  rw [← h.eLpNorm_eq]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : BorelSpace γ
    p : ENNReal
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.Memℒp f p μ
    ⊢ LT.lt (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  exact hf.2
  /-
    🎉 no goals
  -/


theorem memℒp_iff [NormedAddCommGroup γ] [BorelSpace γ] {p : ℝ≥0∞} (h : IdentDistrib f g μ ν) :
    Memℒp f p μ ↔ Memℒp g p ν :=
  ⟨fun hf => h.memℒp_snd hf, fun hg => h.symm.memℒp_snd hg⟩


theorem integrable_snd [NormedAddCommGroup γ] [BorelSpace γ] (h : IdentDistrib f g μ ν)
    (hf : Integrable f μ) : Integrable g ν := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : BorelSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable g ν
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    inst✝² : MeasurableSpace γ
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → γ
    g : β → γ
    inst✝¹ : NormedAddCommGroup γ
    inst✝ : BorelSpace γ
    h : ProbabilityTheory.IdentDistrib f g μ ν
    hf : MeasureTheory.Memℒp f 1 μ
    ⊢ MeasureTheory.Memℒp g 1 ν
  -/
  exact h.memℒp_snd hf
  /-
    🎉 no goals
  -/


theorem integrable_iff [NormedAddCommGroup γ] [BorelSpace γ] (h : IdentDistrib f g μ ν) :
    Integrable f μ ↔ Integrable g ν :=
  ⟨fun hf => h.integrable_snd hf, fun hg => h.symm.integrable_snd hg⟩


protected theorem norm [NormedAddCommGroup γ] [BorelSpace γ] (h : IdentDistrib f g μ ν) :
    IdentDistrib (fun x => ‖f x‖) (fun x => ‖g x‖) μ ν :=
  h.comp measurable_norm


protected theorem nnnorm [NormedAddCommGroup γ] [BorelSpace γ] (h : IdentDistrib f g μ ν) :
    IdentDistrib (fun x => ‖f x‖₊) (fun x => ‖g x‖₊) μ ν :=
  h.comp measurable_nnnorm


protected theorem pow [Pow γ ℕ] [MeasurablePow γ ℕ] (h : IdentDistrib f g μ ν) {n : ℕ} :
    IdentDistrib (fun x => f x ^ n) (fun x => g x ^ n) μ ν :=
  h.comp (measurable_id.pow_const n)


protected theorem sq [Pow γ ℕ] [MeasurablePow γ ℕ] (h : IdentDistrib f g μ ν) :
    IdentDistrib (fun x => f x ^ 2) (fun x => g x ^ 2) μ ν :=
  h.comp (measurable_id.pow_const 2)


protected theorem coe_nnreal_ennreal {f : α → ℝ≥0} {g : β → ℝ≥0} (h : IdentDistrib f g μ ν) :
    IdentDistrib (fun x => (f x : ℝ≥0∞)) (fun x => (g x : ℝ≥0∞)) μ ν :=
  h.comp measurable_coe_nnreal_ennreal


@[to_additive]
theorem mul_const [Mul γ] [MeasurableMul γ] (h : IdentDistrib f g μ ν) (c : γ) :
    IdentDistrib (fun x => f x * c) (fun x => g x * c) μ ν :=
  h.comp (measurable_mul_const c)


@[to_additive]
theorem const_mul [Mul γ] [MeasurableMul γ] (h : IdentDistrib f g μ ν) (c : γ) :
    IdentDistrib (fun x => c * f x) (fun x => c * g x) μ ν :=
  h.comp (measurable_const_mul c)


@[to_additive]
theorem div_const [Div γ] [MeasurableDiv γ] (h : IdentDistrib f g μ ν) (c : γ) :
    IdentDistrib (fun x => f x / c) (fun x => g x / c) μ ν :=
  h.comp (MeasurableDiv.measurable_div_const c)


@[to_additive]
theorem const_div [Div γ] [MeasurableDiv γ] (h : IdentDistrib f g μ ν) (c : γ) :
    IdentDistrib (fun x => c / f x) (fun x => c / g x) μ ν :=
  h.comp (MeasurableDiv.measurable_const_div c)


@[to_additive]
lemma inv [Inv γ] [MeasurableInv γ] (h : IdentDistrib f g μ ν) :
    IdentDistrib f⁻¹ g⁻¹ μ ν := h.comp measurable_inv


theorem evariance_eq {f : α → ℝ} {g : β → ℝ} (h : IdentDistrib f g μ ν) :
    evariance f μ = evariance g ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → Real
    g : β → Real
    h : ProbabilityTheory.IdentDistrib f g μ ν
    ⊢ Eq (ProbabilityTheory.evariance f μ) (ProbabilityTheory.evariance g ν)
  -/
  convert (h.sub_const (∫ x, f x ∂μ)).nnnorm.coe_nnreal_ennreal.sq.lintegral_eq
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → Real
    g : β → Real
    h : ProbabilityTheory.IdentDistrib f g μ ν
    ⊢ Eq (ProbabilityTheory.evariance g ν) (MeasureTheory.lintegral ν fun x => HPo …
  -/
  rw [h.integral_eq]
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    f : α → Real
    g : β → Real
    h : ProbabilityTheory.IdentDistrib f g μ ν
    ⊢ Eq (ProbabilityTheory.evariance g ν) (MeasureTheory.lintegral ν fun x => HPo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem variance_eq {f : α → ℝ} {g : β → ℝ} (h : IdentDistrib f g μ ν) :
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝¹ : MeasurableSpace α
                                        inst✝ : MeasurableSpace β
                                        μ : MeasureTheory.Measure α
                                        ν : MeasureTheory.Measure β
                                        f : α → Real
                                        g : β → Real
                                        h : ProbabilityTheory.IdentDistrib f g μ ν
                                        ⊢ Eq (ProbabilityTheory.variance f μ) (ProbabilityTheory.variance g ν)
                                      -/
    variance f μ = variance g ν := by rw [variance, h.evariance_eq]; rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- This lemma is superseded by `Memℒp.uniformIntegrable_of_identDistrib` which only requires
`AEStronglyMeasurable`. -/
theorem Memℒp.uniformIntegrable_of_identDistrib_aux {ι : Type*} {f : ι → α → E} {j : ι} {p : ℝ≥0∞}
    (hp : 1 ≤ p) (hp' : p ≠ ∞) (hℒp : Memℒp (f j) p μ) (hfmeas : ∀ i, StronglyMeasurable (f i))
    (hf : ∀ i, IdentDistrib (f i) (f j) μ μ) : UniformIntegrable f p μ := by
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hfmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  refine uniformIntegrable_of' hp hp' hfmeas fun ε hε => ?_
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hfmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  by_cases hι : Nonempty ι
  /-
    case pos
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hfmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    ε : Real
    hε : LT.lt 0 ε
    hι : Nonempty ι
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  swap; · exact ⟨0, fun i => False.elim (hι <| Nonempty.intro i)⟩
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hfmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    ε : Real
    hε : LT.lt 0 ε
    hι : Nonempty ι
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  obtain ⟨C, hC₁, hC₂⟩ := hℒp.eLpNorm_indicator_norm_ge_pos_le (hfmeas _) hε
  /-
    case pos.intro.intro
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hfmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    ε : Real
    hε : LT.lt 0 ε
    hι : Nonempty ι
    C : Real
    hC₁ : LT.lt 0 C
    hC₂ : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C ↑(NNNorm.nnnorm (f …
    ⊢ Exists fun C => ∀ (i : ι), LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE. …
  -/
  refine ⟨⟨C, hC₁.le⟩, fun i => le_trans (le_of_eq ?_) hC₂⟩
  have : {x | (⟨C, hC₁.le⟩ : ℝ≥0) ≤ ‖f i x‖₊} = {x | C ≤ ‖f i x‖} := by
    ext x
    simp_rw [← norm_toNNReal]
    exact Real.le_toNNReal_iff_coe_le (norm_nonneg _)
  /-
    case pos.intro.intro
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hfmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    ε : Real
    hε : LT.lt 0 ε
    hι : Nonempty ι
    C : Real
    hC₁ : LT.lt 0 C
    hC₂ : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C ↑(NNNorm.nnnorm (f …
    i : ι
    this : Eq (setOf fun x => LE.le ⟨C, ⋯⟩ (NNNorm.nnnorm (f i x))) (setOf fun x = …
    ⊢ Eq (MeasureTheory.eLpNorm ((setOf fun x => LE.le ⟨C, ⋯⟩ (NNNorm.nnnorm (f i  …
  -/
  rw [this, ← eLpNorm_norm, ← eLpNorm_norm (Set.indicator _ _)]
  /-
    case pos.intro.intro
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hfmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    ε : Real
    hε : LT.lt 0 ε
    hι : Nonempty ι
    C : Real
    hC₁ : LT.lt 0 C
    hC₂ : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C ↑(NNNorm.nnnorm (f …
    i : ι
    this : Eq (setOf fun x => LE.le ⟨C, ⋯⟩ (NNNorm.nnnorm (f i x))) (setOf fun x = …
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => Norm.norm ((setOf fun x => LE.le C (Norm …
  -/
  simp_rw [norm_indicator_eq_indicator_norm, coe_nnnorm]
  /-
    case pos.intro.intro
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hfmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    ε : Real
    hε : LT.lt 0 ε
    hι : Nonempty ι
    C : Real
    hC₁ : LT.lt 0 C
    hC₂ : LE.le (MeasureTheory.eLpNorm ((setOf fun x => LE.le C ↑(NNNorm.nnnorm (f …
    i : ι
    this : Eq (setOf fun x => LE.le ⟨C, ⋯⟩ (NNNorm.nnnorm (f i x))) (setOf fun x = …
    ⊢ Eq (MeasureTheory.eLpNorm (fun x => (setOf fun x => LE.le C (Norm.norm (f i  …
  -/
  let F : E → ℝ := (fun x : E => if (⟨C, hC₁.le⟩ : ℝ≥0) ≤ ‖x‖₊ then ‖x‖ else 0)
  have F_meas : Measurable F := by
    apply measurable_norm.indicator (measurableSet_le measurable_const measurable_nnnorm)
  have : ∀ k, (fun x ↦ Set.indicator {x | C ≤ ‖f k x‖} (fun a ↦ ‖f k a‖) x) = F ∘ f k := by
    intro k
    ext x
    simp only [Set.indicator, Set.mem_setOf_eq]; norm_cast
  rw [this, this, ← eLpNorm_map_measure F_meas.aestronglyMeasurable (hf i).aemeasurable_fst,
    (hf i).map_eq, eLpNorm_map_measure F_meas.aestronglyMeasurable (hf j).aemeasurable_fst]


/-- A sequence of identically distributed Lᵖ functions is p-uniformly integrable. -/
theorem Memℒp.uniformIntegrable_of_identDistrib {ι : Type*} {f : ι → α → E} {j : ι} {p : ℝ≥0∞}
    (hp : 1 ≤ p) (hp' : p ≠ ∞) (hℒp : Memℒp (f j) p μ) (hf : ∀ i, IdentDistrib (f i) (f j) μ μ) :
    UniformIntegrable f p μ := by
  have hfmeas : ∀ i, AEStronglyMeasurable (f i) μ := fun i =>
    (hf i).aestronglyMeasurable_iff.2 hℒp.1
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    hfmeas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  set g : ι → α → E := fun i => (hfmeas i).choose
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    hfmeas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    g : ι → α → E := fun i => Exists.choose ⋯
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  have hgmeas : ∀ i, StronglyMeasurable (g i) := fun i => (Exists.choose_spec <| hfmeas i).1
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    hfmeas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    g : ι → α → E := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  have hgeq : ∀ i, g i =ᵐ[μ] f i := fun i => (Exists.choose_spec <| hfmeas i).2.symm
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ι : Type u_6
    f : ι → α → E
    j : ι
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    hℒp : MeasureTheory.Memℒp (f j) p μ
    hf : ∀ (i : ι), ProbabilityTheory.IdentDistrib (f i) (f j) μ μ
    hfmeas : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    g : ι → α → E := fun i => Exists.choose ⋯
    hgmeas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (g i)
    hgeq : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (g i) (f i)
    ⊢ MeasureTheory.UniformIntegrable f p μ
  -/
  have hgℒp : Memℒp (g j) p μ := hℒp.ae_eq (hgeq j).symm
  exact UniformIntegrable.ae_eq
    (Memℒp.uniformIntegrable_of_identDistrib_aux hp hp' hgℒp hgmeas fun i =>
      (IdentDistrib.of_ae_eq (hgmeas i).aemeasurable (hgeq i)).trans
        ((hf i).trans <| IdentDistrib.of_ae_eq (hfmeas j).aemeasurable (hgeq j).symm)) hgeq


/-- If `X` and `Y` are independent and `(X, Y)` and `(X', Y')` are identically distributed,
then `X'` and `Y'` are independent. -/
lemma indepFun_of_identDistrib_pair
    {μ : Measure γ} {μ' : Measure δ} [IsFiniteMeasure μ] [IsFiniteMeasure μ']
    {X : γ → α} {X' : δ → α} {Y : γ → β} {Y' : δ → β} (h_indep : IndepFun X Y μ)
    (h_ident : IdentDistrib (fun ω ↦ (X ω, Y ω)) (fun ω ↦ (X' ω, Y' ω)) μ μ') :
    IndepFun X' Y' μ' := by
  rw [indepFun_iff_map_prod_eq_prod_map_map _ _, ← h_ident.map_eq,
    (indepFun_iff_map_prod_eq_prod_map_map _ _).1 h_indep]
  · exact congr (congrArg Measure.prod <| (h_ident.comp measurable_fst).map_eq)
      (h_ident.comp measurable_snd).map_eq
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      inst✝² : MeasurableSpace δ
      μ : MeasureTheory.Measure γ
      μ' : MeasureTheory.Measure δ
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ'
      X : γ → α
      X' : δ → α
      Y : γ → β
      Y' : δ → β
      h_indep : ProbabilityTheory.IndepFun X Y μ
      h_ident : ProbabilityTheory.IdentDistrib (fun ω => { fst := X ω, snd := Y ω }) …
      ⊢ AEMeasurable X μ
    -/
  · exact measurable_fst.aemeasurable.comp_aemeasurable h_ident.aemeasurable_fst
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      inst✝² : MeasurableSpace δ
      μ : MeasureTheory.Measure γ
      μ' : MeasureTheory.Measure δ
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ'
      X : γ → α
      X' : δ → α
      Y : γ → β
      Y' : δ → β
      h_indep : ProbabilityTheory.IndepFun X Y μ
      h_ident : ProbabilityTheory.IdentDistrib (fun ω => { fst := X ω, snd := Y ω }) …
      ⊢ AEMeasurable Y μ
    -/
  · exact measurable_snd.aemeasurable.comp_aemeasurable h_ident.aemeasurable_fst
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      inst✝² : MeasurableSpace δ
      μ : MeasureTheory.Measure γ
      μ' : MeasureTheory.Measure δ
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ'
      X : γ → α
      X' : δ → α
      Y : γ → β
      Y' : δ → β
      h_indep : ProbabilityTheory.IndepFun X Y μ
      h_ident : ProbabilityTheory.IdentDistrib (fun ω => { fst := X ω, snd := Y ω }) …
      ⊢ AEMeasurable X' μ'
    -/
  · exact measurable_fst.aemeasurable.comp_aemeasurable h_ident.aemeasurable_snd
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      inst✝² : MeasurableSpace δ
      μ : MeasureTheory.Measure γ
      μ' : MeasureTheory.Measure δ
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ'
      X : γ → α
      X' : δ → α
      Y : γ → β
      Y' : δ → β
      h_indep : ProbabilityTheory.IndepFun X Y μ
      h_ident : ProbabilityTheory.IdentDistrib (fun ω => { fst := X ω, snd := Y ω }) …
      ⊢ AEMeasurable Y' μ'
    -/
  · exact measurable_snd.aemeasurable.comp_aemeasurable h_ident.aemeasurable_snd
    /-
      🎉 no goals
    -/


