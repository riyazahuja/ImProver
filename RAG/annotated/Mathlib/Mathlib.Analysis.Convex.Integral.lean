/-- If `μ` is a probability measure on `α`, `s` is a convex closed set in `E`, and `f` is an
integrable function sending `μ`-a.e. points to `s`, then the expected value of `f` belongs to `s`:
`∫ x, f x ∂μ ∈ s`. See also `Convex.sum_mem` for a finite sum version of this lemma. -/
theorem Convex.integral_mem [IsProbabilityMeasure μ] (hs : Convex ℝ s) (hsc : IsClosed s)
    (hf : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ) : (∫ x, f x ∂μ) ∈ s := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hs : Convex Real s
    hsc : IsClosed s
    hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    ⊢ Membership.mem s (MeasureTheory.integral μ fun x => f x)
  -/
  borelize E
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hs : Convex Real s
    hsc : IsClosed s
    hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ Membership.mem s (MeasureTheory.integral μ fun x => f x)
  -/
  rcases hfi.aestronglyMeasurable with ⟨g, hgm, hfg⟩
  haveI : SeparableSpace (range g ∩ s : Set E) :=
    (hgm.isSeparable_range.mono inter_subset_left).separableSpace
  obtain ⟨y₀, h₀⟩ : (range g ∩ s).Nonempty := by
    rcases (hf.and hfg).exists with ⟨x₀, h₀⟩
    exact ⟨f x₀, by simp only [h₀.2, mem_range_self], h₀.1⟩
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hs : Convex Real s
    hsc : IsClosed s
    hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : α → E
    hgm : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
    y₀ : E
    h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
    ⊢ Membership.mem s (MeasureTheory.integral μ fun x => f x)
  -/
  rw [integral_congr_ae hfg]; rw [integrable_congr hfg] at hfi
  have hg : ∀ᵐ x ∂μ, g x ∈ closure (range g ∩ s) := by
    filter_upwards [hfg.rw (fun _ y => y ∈ s) hf] with x hx
    apply subset_closure
    exact ⟨mem_range_self _, hx⟩
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hs : Convex Real s
    hsc : IsClosed s
    hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    g : α → E
    hfi : MeasureTheory.Integrable g μ
    hgm : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    this : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
    y₀ : E
    h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
    hg : Filter.Eventually (fun x => Membership.mem (closure (Inter.inter (Set.ran …
    ⊢ Membership.mem s (MeasureTheory.integral μ fun a => g a)
  -/
  set G : ℕ → SimpleFunc α E := SimpleFunc.approxOn _ hgm.measurable (range g ∩ s) y₀ h₀
  have : Tendsto (fun n => (G n).integral μ) atTop (𝓝 <| ∫ x, g x ∂μ) :=
    tendsto_integral_approxOn_of_measurable hfi _ hg _ (integrable_const _)
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hs : Convex Real s
    hsc : IsClosed s
    hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    this✝² : MeasurableSpace E := borel E
    this✝¹ : BorelSpace E
    g : α → E
    hfi : MeasureTheory.Integrable g μ
    hgm : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    this✝ : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
    y₀ : E
    h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
    hg : Filter.Eventually (fun x => Membership.mem (closure (Inter.inter (Set.ran …
    G : Nat → MeasureTheory.SimpleFunc α E := MeasureTheory.SimpleFunc.approxOn g  …
    this : Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (G n)) Fil …
    ⊢ Membership.mem s (MeasureTheory.integral μ fun a => g a)
  -/
  refine hsc.mem_of_tendsto this (Eventually.of_forall fun n => hs.sum_mem ?_ ?_ ?_)
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      s : Set E
      f : α → E
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      hs : Convex Real s
      hsc : IsClosed s
      hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
      this✝² : MeasurableSpace E := borel E
      this✝¹ : BorelSpace E
      g : α → E
      hfi : MeasureTheory.Integrable g μ
      hgm : MeasureTheory.StronglyMeasurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      this✝ : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
      y₀ : E
      h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
      hg : Filter.Eventually (fun x => Membership.mem (closure (Inter.inter (Set.ran …
      G : Nat → MeasureTheory.SimpleFunc α E := MeasureTheory.SimpleFunc.approxOn g  …
      this : Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (G n)) Fil …
      n : Nat
      ⊢ ∀ (i : E), Membership.mem (G n).range i → LE.le 0 (μ (Set.preimage (⇑(G n))  …
    -/
  · exact fun _ _ => ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/
  · rw [← ENNReal.toReal_sum, (G n).sum_range_measure_preimage_singleton, measure_univ,
      ENNReal.one_toReal]
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      s : Set E
      f : α → E
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      hs : Convex Real s
      hsc : IsClosed s
      hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
      this✝² : MeasurableSpace E := borel E
      this✝¹ : BorelSpace E
      g : α → E
      hfi : MeasureTheory.Integrable g μ
      hgm : MeasureTheory.StronglyMeasurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      this✝ : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
      y₀ : E
      h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
      hg : Filter.Eventually (fun x => Membership.mem (closure (Inter.inter (Set.ran …
      G : Nat → MeasureTheory.SimpleFunc α E := MeasureTheory.SimpleFunc.approxOn g  …
      this : Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (G n)) Fil …
      n : Nat
      ⊢ ∀ (a : E), Membership.mem (G n).range a → Ne (μ (Set.preimage (⇑(G n)) (Sing …
    -/
    exact fun _ _ => measure_ne_top _ _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      s : Set E
      f : α → E
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      hs : Convex Real s
      hsc : IsClosed s
      hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
      this✝² : MeasurableSpace E := borel E
      this✝¹ : BorelSpace E
      g : α → E
      hfi : MeasureTheory.Integrable g μ
      hgm : MeasureTheory.StronglyMeasurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      this✝ : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
      y₀ : E
      h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
      hg : Filter.Eventually (fun x => Membership.mem (closure (Inter.inter (Set.ran …
      G : Nat → MeasureTheory.SimpleFunc α E := MeasureTheory.SimpleFunc.approxOn g  …
      this : Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (G n)) Fil …
      n : Nat
      ⊢ ∀ (i : E), Membership.mem (G n).range i → Membership.mem s (↑(ContinuousLine …
    -/
  · simp only [SimpleFunc.mem_range, forall_mem_range]
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      s : Set E
      f : α → E
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      hs : Convex Real s
      hsc : IsClosed s
      hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
      this✝² : MeasurableSpace E := borel E
      this✝¹ : BorelSpace E
      g : α → E
      hfi : MeasureTheory.Integrable g μ
      hgm : MeasureTheory.StronglyMeasurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      this✝ : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
      y₀ : E
      h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
      hg : Filter.Eventually (fun x => Membership.mem (closure (Inter.inter (Set.ran …
      G : Nat → MeasureTheory.SimpleFunc α E := MeasureTheory.SimpleFunc.approxOn g  …
      this : Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (G n)) Fil …
      n : Nat
      ⊢ ∀ (i : α), Membership.mem s (↑(ContinuousLinearMap.id Real E) ((G n) i))
    -/
    intro x
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      s : Set E
      f : α → E
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      hs : Convex Real s
      hsc : IsClosed s
      hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
      this✝² : MeasurableSpace E := borel E
      this✝¹ : BorelSpace E
      g : α → E
      hfi : MeasureTheory.Integrable g μ
      hgm : MeasureTheory.StronglyMeasurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      this✝ : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
      y₀ : E
      h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
      hg : Filter.Eventually (fun x => Membership.mem (closure (Inter.inter (Set.ran …
      G : Nat → MeasureTheory.SimpleFunc α E := MeasureTheory.SimpleFunc.approxOn g  …
      this : Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (G n)) Fil …
      n : Nat
      x : α
      ⊢ Membership.mem s (↑(ContinuousLinearMap.id Real E) ((G n) x))
    -/
    apply (range g).inter_subset_right
    /-
      case intro.intro.intro.refine_3.a
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      s : Set E
      f : α → E
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      hs : Convex Real s
      hsc : IsClosed s
      hf : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
      this✝² : MeasurableSpace E := borel E
      this✝¹ : BorelSpace E
      g : α → E
      hfi : MeasureTheory.Integrable g μ
      hgm : MeasureTheory.StronglyMeasurable g
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      this✝ : TopologicalSpace.SeparableSpace ↑(Inter.inter (Set.range g) s)
      y₀ : E
      h₀ : Membership.mem (Inter.inter (Set.range g) s) y₀
      hg : Filter.Eventually (fun x => Membership.mem (closure (Inter.inter (Set.ran …
      G : Nat → MeasureTheory.SimpleFunc α E := MeasureTheory.SimpleFunc.approxOn g  …
      this : Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (G n)) Fil …
      n : Nat
      x : α
      ⊢ Membership.mem (Inter.inter (Set.range g) s) (↑(ContinuousLinearMap.id Real  …
    -/
    exact SimpleFunc.approxOn_mem hgm.measurable h₀ _ _
    /-
      🎉 no goals
    -/


/-- If `μ` is a non-zero finite measure on `α`, `s` is a convex closed set in `E`, and `f` is an
integrable function sending `μ`-a.e. points to `s`, then the average value of `f` belongs to `s`:
`⨍ x, f x ∂μ ∈ s`. See also `Convex.centerMass_mem` for a finite sum version of this lemma. -/
theorem Convex.average_mem [IsFiniteMeasure μ] [NeZero μ] (hs : Convex ℝ s) (hsc : IsClosed s)
    (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ) : (⨍ x, f x ∂μ) ∈ s :=
  hs.integral_mem hsc (ae_mono' smul_absolutelyContinuous hfs) hfi.to_average


/-- If `μ` is a non-zero finite measure on `α`, `s` is a convex closed set in `E`, and `f` is an
integrable function sending `μ`-a.e. points to `s`, then the average value of `f` belongs to `s`:
`⨍ x, f x ∂μ ∈ s`. See also `Convex.centerMass_mem` for a finite sum version of this lemma. -/
theorem Convex.set_average_mem (hs : Convex ℝ s) (hsc : IsClosed s) (h0 : μ t ≠ 0) (ht : μ t ≠ ∞)
    (hfs : ∀ᵐ x ∂μ.restrict t, f x ∈ s) (hfi : IntegrableOn f t μ) : (⨍ x in t, f x ∂μ) ∈ s :=
  have := Fact.mk ht.lt_top
  have := NeZero.mk h0
  hs.average_mem hsc hfs hfi


/-- If `μ` is a non-zero finite measure on `α`, `s` is a convex set in `E`, and `f` is an integrable
function sending `μ`-a.e. points to `s`, then the average value of `f` belongs to `closure s`:
`⨍ x, f x ∂μ ∈ s`. See also `Convex.centerMass_mem` for a finite sum version of this lemma. -/
theorem Convex.set_average_mem_closure (hs : Convex ℝ s) (h0 : μ t ≠ 0) (ht : μ t ≠ ∞)
    (hfs : ∀ᵐ x ∂μ.restrict t, f x ∈ s) (hfi : IntegrableOn f t μ) :
    (⨍ x in t, f x ∂μ) ∈ closure s :=
  hs.closure.set_average_mem isClosed_closure h0 ht (hfs.mono fun _ hx => subset_closure hx) hfi


theorem ConvexOn.average_mem_epigraph [IsFiniteMeasure μ] [NeZero μ] (hg : ConvexOn ℝ s g)
    (hgc : ContinuousOn g s) (hsc : IsClosed s) (hfs : ∀ᵐ x ∂μ, f x ∈ s)
    (hfi : Integrable f μ) (hgi : Integrable (g ∘ f) μ) :
    (⨍ x, f x ∂μ, ⨍ x, g (f x) ∂μ) ∈ {p : E × ℝ | p.1 ∈ s ∧ g p.1 ≤ p.2} := by
  have ht_mem : ∀ᵐ x ∂μ, (f x, g (f x)) ∈ {p : E × ℝ | p.1 ∈ s ∧ g p.1 ≤ p.2} :=
    hfs.mono fun x hx => ⟨hx, le_rfl⟩
  exact average_pair hfi hgi ▸
    hg.convex_epigraph.average_mem (hsc.epigraph hgc) ht_mem (hfi.prod_mk hgi)


theorem ConcaveOn.average_mem_hypograph [IsFiniteMeasure μ] [NeZero μ] (hg : ConcaveOn ℝ s g)
    (hgc : ContinuousOn g s) (hsc : IsClosed s) (hfs : ∀ᵐ x ∂μ, f x ∈ s)
    (hfi : Integrable f μ) (hgi : Integrable (g ∘ f) μ) :
    (⨍ x, f x ∂μ, ⨍ x, g (f x) ∂μ) ∈ {p : E × ℝ | p.1 ∈ s ∧ p.2 ≤ g p.1} := by
  simpa only [mem_setOf_eq, Pi.neg_apply, average_neg, neg_le_neg_iff] using
    hg.neg.average_mem_epigraph hgc.neg hsc hfs hfi hgi.neg


/-- **Jensen's inequality**: if a function `g : E → ℝ` is convex and continuous on a convex closed
set `s`, `μ` is a finite non-zero measure on `α`, and `f : α → E` is a function sending
`μ`-a.e. points to `s`, then the value of `g` at the average value of `f` is less than or equal to
the average value of `g ∘ f` provided that both `f` and `g ∘ f` are integrable. See also
`ConvexOn.map_centerMass_le` for a finite sum version of this lemma. -/
theorem ConvexOn.map_average_le [IsFiniteMeasure μ] [NeZero μ]
    (hg : ConvexOn ℝ s g) (hgc : ContinuousOn g s) (hsc : IsClosed s)
    (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ) (hgi : Integrable (g ∘ f) μ) :
    g (⨍ x, f x ∂μ) ≤ ⨍ x, g (f x) ∂μ :=
  (hg.average_mem_epigraph hgc hsc hfs hfi hgi).2


/-- **Jensen's inequality**: if a function `g : E → ℝ` is concave and continuous on a convex closed
set `s`, `μ` is a finite non-zero measure on `α`, and `f : α → E` is a function sending
`μ`-a.e. points to `s`, then the average value of `g ∘ f` is less than or equal to the value of `g`
at the average value of `f` provided that both `f` and `g ∘ f` are integrable. See also
`ConcaveOn.le_map_centerMass` for a finite sum version of this lemma. -/
theorem ConcaveOn.le_map_average [IsFiniteMeasure μ] [NeZero μ]
    (hg : ConcaveOn ℝ s g) (hgc : ContinuousOn g s) (hsc : IsClosed s)
    (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ) (hgi : Integrable (g ∘ f) μ) :
    (⨍ x, g (f x) ∂μ) ≤ g (⨍ x, f x ∂μ) :=
  (hg.average_mem_hypograph hgc hsc hfs hfi hgi).2


/-- **Jensen's inequality**: if a function `g : E → ℝ` is convex and continuous on a convex closed
set `s`, `μ` is a finite non-zero measure on `α`, and `f : α → E` is a function sending
`μ`-a.e. points of a set `t` to `s`, then the value of `g` at the average value of `f` over `t` is
less than or equal to the average value of `g ∘ f` over `t` provided that both `f` and `g ∘ f` are
integrable. -/
theorem ConvexOn.set_average_mem_epigraph (hg : ConvexOn ℝ s g) (hgc : ContinuousOn g s)
    (hsc : IsClosed s) (h0 : μ t ≠ 0) (ht : μ t ≠ ∞) (hfs : ∀ᵐ x ∂μ.restrict t, f x ∈ s)
    (hfi : IntegrableOn f t μ) (hgi : IntegrableOn (g ∘ f) t μ) :
    (⨍ x in t, f x ∂μ, ⨍ x in t, g (f x) ∂μ) ∈ {p : E × ℝ | p.1 ∈ s ∧ g p.1 ≤ p.2} :=
  have := Fact.mk ht.lt_top
  have := NeZero.mk h0
  hg.average_mem_epigraph hgc hsc hfs hfi hgi


/-- **Jensen's inequality**: if a function `g : E → ℝ` is concave and continuous on a convex closed
set `s`, `μ` is a finite non-zero measure on `α`, and `f : α → E` is a function sending
`μ`-a.e. points of a set `t` to `s`, then the average value of `g ∘ f` over `t` is less than or
equal to the value of `g` at the average value of `f` over `t` provided that both `f` and `g ∘ f`
are integrable. -/
theorem ConcaveOn.set_average_mem_hypograph (hg : ConcaveOn ℝ s g) (hgc : ContinuousOn g s)
    (hsc : IsClosed s) (h0 : μ t ≠ 0) (ht : μ t ≠ ∞) (hfs : ∀ᵐ x ∂μ.restrict t, f x ∈ s)
    (hfi : IntegrableOn f t μ) (hgi : IntegrableOn (g ∘ f) t μ) :
    (⨍ x in t, f x ∂μ, ⨍ x in t, g (f x) ∂μ) ∈ {p : E × ℝ | p.1 ∈ s ∧ p.2 ≤ g p.1} := by
  simpa only [mem_setOf_eq, Pi.neg_apply, average_neg, neg_le_neg_iff] using
    hg.neg.set_average_mem_epigraph hgc.neg hsc h0 ht hfs hfi hgi.neg


/-- **Jensen's inequality**: if a function `g : E → ℝ` is convex and continuous on a convex closed
set `s`, `μ` is a finite non-zero measure on `α`, and `f : α → E` is a function sending
`μ`-a.e. points of a set `t` to `s`, then the value of `g` at the average value of `f` over `t` is
less than or equal to the average value of `g ∘ f` over `t` provided that both `f` and `g ∘ f` are
integrable. -/
theorem ConvexOn.map_set_average_le (hg : ConvexOn ℝ s g) (hgc : ContinuousOn g s)
    (hsc : IsClosed s) (h0 : μ t ≠ 0) (ht : μ t ≠ ∞) (hfs : ∀ᵐ x ∂μ.restrict t, f x ∈ s)
    (hfi : IntegrableOn f t μ) (hgi : IntegrableOn (g ∘ f) t μ) :
    g (⨍ x in t, f x ∂μ) ≤ ⨍ x in t, g (f x) ∂μ :=
  (hg.set_average_mem_epigraph hgc hsc h0 ht hfs hfi hgi).2


/-- **Jensen's inequality**: if a function `g : E → ℝ` is concave and continuous on a convex closed
set `s`, `μ` is a finite non-zero measure on `α`, and `f : α → E` is a function sending
`μ`-a.e. points of a set `t` to `s`, then the average value of `g ∘ f` over `t` is less than or
equal to the value of `g` at the average value of `f` over `t` provided that both `f` and `g ∘ f`
are integrable. -/
theorem ConcaveOn.le_map_set_average (hg : ConcaveOn ℝ s g) (hgc : ContinuousOn g s)
    (hsc : IsClosed s) (h0 : μ t ≠ 0) (ht : μ t ≠ ∞) (hfs : ∀ᵐ x ∂μ.restrict t, f x ∈ s)
    (hfi : IntegrableOn f t μ) (hgi : IntegrableOn (g ∘ f) t μ) :
    (⨍ x in t, g (f x) ∂μ) ≤ g (⨍ x in t, f x ∂μ) :=
  (hg.set_average_mem_hypograph hgc hsc h0 ht hfs hfi hgi).2


/-- **Jensen's inequality**: if a function `g : E → ℝ` is convex and continuous on a convex closed
set `s`, `μ` is a probability measure on `α`, and `f : α → E` is a function sending `μ`-a.e.  points
to `s`, then the value of `g` at the expected value of `f` is less than or equal to the expected
value of `g ∘ f` provided that both `f` and `g ∘ f` are integrable. See also
`ConvexOn.map_centerMass_le` for a finite sum version of this lemma. -/
theorem ConvexOn.map_integral_le [IsProbabilityMeasure μ] (hg : ConvexOn ℝ s g)
    (hgc : ContinuousOn g s) (hsc : IsClosed s) (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ)
    (hgi : Integrable (g ∘ f) μ) : g (∫ x, f x ∂μ) ≤ ∫ x, g (f x) ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    g : E → Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hg : ConvexOn Real s g
    hgc : ContinuousOn g s
    hsc : IsClosed s
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    hgi : MeasureTheory.Integrable (Function.comp g f) μ
    ⊢ LE.le (g (MeasureTheory.integral μ fun x => f x)) (MeasureTheory.integral μ  …
  -/
  simpa only [average_eq_integral] using hg.map_average_le hgc hsc hfs hfi hgi
  /-
    🎉 no goals
  -/


/-- **Jensen's inequality**: if a function `g : E → ℝ` is concave and continuous on a convex closed
set `s`, `μ` is a probability measure on `α`, and `f : α → E` is a function sending `μ`-a.e.  points
to `s`, then the expected value of `g ∘ f` is less than or equal to the value of `g` at the expected
value of `f` provided that both `f` and `g ∘ f` are integrable. -/
theorem ConcaveOn.le_map_integral [IsProbabilityMeasure μ] (hg : ConcaveOn ℝ s g)
    (hgc : ContinuousOn g s) (hsc : IsClosed s) (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ)
    (hgi : Integrable (g ∘ f) μ) : (∫ x, g (f x) ∂μ) ≤ g (∫ x, f x ∂μ) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    g : E → Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hg : ConcaveOn Real s g
    hgc : ContinuousOn g s
    hsc : IsClosed s
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    hgi : MeasureTheory.Integrable (Function.comp g f) μ
    ⊢ LE.le (MeasureTheory.integral μ fun x => g (f x)) (g (MeasureTheory.integral …
  -/
  simpa only [average_eq_integral] using hg.le_map_average hgc hsc hfs hfi hgi
  /-
    🎉 no goals
  -/


/-- If `f : α → E` is an integrable function, then either it is a.e. equal to the constant
`⨍ x, f x ∂μ` or there exists a measurable set such that `μ t ≠ 0`, `μ tᶜ ≠ 0`, and the average
values of `f` over `t` and `tᶜ` are different. -/
theorem ae_eq_const_or_exists_average_ne_compl [IsFiniteMeasure μ] (hfi : Integrable f μ) :
    f =ᵐ[μ] const α (⨍ x, f x ∂μ) ∨
      ∃ t, MeasurableSet t ∧ μ t ≠ 0 ∧ μ tᶜ ≠ 0 ∧ (⨍ x in t, f x ∂μ) ≠ ⨍ x in tᶜ, f x ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfi : MeasureTheory.Integrable f μ
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  refine or_iff_not_imp_right.mpr fun H => ?_; push_neg at H
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfi : MeasureTheory.Integrable f μ
    H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.average …
  -/
  refine hfi.ae_eq_of_forall_setIntegral_eq _ _ (integrable_const _) fun t ht ht' => ?_; clear ht'
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfi : MeasureTheory.Integrable f μ
    H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (MeasureTheory.integ …
  -/
  simp only [const_apply, setIntegral_const]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfi : MeasureTheory.Integrable f μ
    H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
    t : Set α
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (HSMul.hSMul (μ t).t …
  -/
  by_cases h₀ : μ t = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfi : MeasureTheory.Integrable f μ
      H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
      t : Set α
      ht : MeasurableSet t
      h₀ : Eq (μ t) 0
      ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (HSMul.hSMul (μ t).t …
    -/
  · rw [restrict_eq_zero.2 h₀, integral_zero_measure, h₀, ENNReal.zero_toReal, zero_smul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfi : MeasureTheory.Integrable f μ
    H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
    t : Set α
    ht : MeasurableSet t
    h₀ : Not (Eq (μ t) 0)
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (HSMul.hSMul (μ t).t …
  -/
  by_cases h₀' : μ tᶜ = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfi : MeasureTheory.Integrable f μ
      H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
      t : Set α
      ht : MeasurableSet t
      h₀ : Not (Eq (μ t) 0)
      h₀' : Eq (μ (HasCompl.compl t)) 0
      ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (HSMul.hSMul (μ t).t …
    -/
  · rw [← ae_eq_univ] at h₀'
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfi : MeasureTheory.Integrable f μ
      H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
      t : Set α
      ht : MeasurableSet t
      h₀ : Not (Eq (μ t) 0)
      h₀' : (MeasureTheory.ae μ).EventuallyEq t Set.univ
      ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (HSMul.hSMul (μ t).t …
    -/
    rw [restrict_congr_set h₀', restrict_univ, measure_congr h₀', measure_smul_average]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfi : MeasureTheory.Integrable f μ
    H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
    t : Set α
    ht : MeasurableSet t
    h₀ : Not (Eq (μ t) 0)
    h₀' : Not (Eq (μ (HasCompl.compl t)) 0)
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (HSMul.hSMul (μ t).t …
  -/
  have := average_mem_openSegment_compl_self ht.nullMeasurableSet h₀ h₀' hfi
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfi : MeasureTheory.Integrable f μ
    H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
    t : Set α
    ht : MeasurableSet t
    h₀ : Not (Eq (μ t) 0)
    h₀' : Not (Eq (μ (HasCompl.compl t)) 0)
    this : Membership.mem (openSegment Real (MeasureTheory.average (μ.restrict t)  …
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (HSMul.hSMul (μ t).t …
  -/
  rw [← H t ht h₀ h₀', openSegment_same, mem_singleton_iff] at this
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfi : MeasureTheory.Integrable f μ
    H : ∀ (t : Set α), MeasurableSet t → Ne (μ t) 0 → Ne (μ (HasCompl.compl t)) 0  …
    t : Set α
    ht : MeasurableSet t
    h₀ : Not (Eq (μ t) 0)
    h₀' : Not (Eq (μ (HasCompl.compl t)) 0)
    this : Eq (MeasureTheory.average μ fun x => f x) (MeasureTheory.average (μ.res …
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (HSMul.hSMul (μ t).t …
  -/
  rw [this, measure_smul_setAverage _ (measure_ne_top μ _)]
  /-
    🎉 no goals
  -/


/-- If an integrable function `f : α → E` takes values in a convex set `s` and for some set `t` of
positive measure, the average value of `f` over `t` belongs to the interior of `s`, then the average
of `f` over the whole space belongs to the interior of `s`. -/
theorem Convex.average_mem_interior_of_set [IsFiniteMeasure μ] (hs : Convex ℝ s) (h0 : μ t ≠ 0)
    (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ) (ht : (⨍ x in t, f x ∂μ) ∈ interior s) :
    (⨍ x, f x ∂μ) ∈ interior s := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    t : Set α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hs : Convex Real s
    h0 : Ne (μ t) 0
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    ht : Membership.mem (interior s) (MeasureTheory.average (μ.restrict t) fun x = …
    ⊢ Membership.mem (interior s) (MeasureTheory.average μ fun x => f x)
  -/
  rw [← measure_toMeasurable] at h0; rw [← restrict_toMeasurable (measure_ne_top μ t)] at ht
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    t : Set α
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hs : Convex Real s
    h0 : Ne (μ (MeasureTheory.toMeasurable μ t)) 0
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    ht : Membership.mem (interior s) (MeasureTheory.average (μ.restrict (MeasureTh …
    ⊢ Membership.mem (interior s) (MeasureTheory.average μ fun x => f x)
  -/
  by_cases h0' : μ (toMeasurable μ t)ᶜ = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      s : Set E
      t : Set α
      f : α → E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hs : Convex Real s
      h0 : Ne (μ (MeasureTheory.toMeasurable μ t)) 0
      hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
      hfi : MeasureTheory.Integrable f μ
      ht : Membership.mem (interior s) (MeasureTheory.average (μ.restrict (MeasureTh …
      h0' : Eq (μ (HasCompl.compl (MeasureTheory.toMeasurable μ t))) 0
      ⊢ Membership.mem (interior s) (MeasureTheory.average μ fun x => f x)
    -/
  · rw [← ae_eq_univ] at h0'
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      s : Set E
      t : Set α
      f : α → E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hs : Convex Real s
      h0 : Ne (μ (MeasureTheory.toMeasurable μ t)) 0
      hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
      hfi : MeasureTheory.Integrable f μ
      ht : Membership.mem (interior s) (MeasureTheory.average (μ.restrict (MeasureTh …
      h0' : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.toMeasurable μ t) Set.u …
      ⊢ Membership.mem (interior s) (MeasureTheory.average μ fun x => f x)
    -/
    rwa [restrict_congr_set h0', restrict_univ] at ht
    /-
      🎉 no goals
    -/
  exact
    hs.openSegment_interior_closure_subset_interior ht
      (hs.set_average_mem_closure h0' (measure_ne_top _ _) (ae_restrict_of_ae hfs)
        hfi.integrableOn)
      (average_mem_openSegment_compl_self (measurableSet_toMeasurable μ t).nullMeasurableSet h0
        h0' hfi)


/-- If an integrable function `f : α → E` takes values in a strictly convex closed set `s`, then
either it is a.e. equal to its average value, or its average value belongs to the interior of
`s`. -/
theorem StrictConvex.ae_eq_const_or_average_mem_interior [IsFiniteMeasure μ] (hs : StrictConvex ℝ s)
    (hsc : IsClosed s) (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ) :
    f =ᵐ[μ] const α (⨍ x, f x ∂μ) ∨ (⨍ x, f x ∂μ) ∈ interior s := by
  have : ∀ {t}, μ t ≠ 0 → (⨍ x in t, f x ∂μ) ∈ s := fun ht =>
    hs.convex.set_average_mem hsc ht (measure_ne_top _ _) (ae_restrict_of_ae hfs) hfi.integrableOn
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hs : StrictConvex Real s
    hsc : IsClosed s
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    this : ∀ {t : Set α}, Ne (μ t) 0 → Membership.mem s (MeasureTheory.average (μ. …
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  refine (ae_eq_const_or_exists_average_ne_compl hfi).imp_right ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hs : StrictConvex Real s
    hsc : IsClosed s
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    this : ∀ {t : Set α}, Ne (μ t) 0 → Membership.mem s (MeasureTheory.average (μ. …
    ⊢ (Exists fun t => And (MeasurableSet t) (And (Ne (μ t) 0) (And (Ne (μ (HasCom …
  -/
  rintro ⟨t, hm, h₀, h₀', hne⟩
  exact
    hs.openSegment_subset (this h₀) (this h₀') hne
      (average_mem_openSegment_compl_self hm.nullMeasurableSet h₀ h₀' hfi)


/-- **Jensen's inequality**, strict version: if an integrable function `f : α → E` takes values in a
convex closed set `s`, and `g : E → ℝ` is continuous and strictly convex on `s`, then
either `f` is a.e. equal to its average value, or `g (⨍ x, f x ∂μ) < ⨍ x, g (f x) ∂μ`. -/
theorem StrictConvexOn.ae_eq_const_or_map_average_lt [IsFiniteMeasure μ] (hg : StrictConvexOn ℝ s g)
    (hgc : ContinuousOn g s) (hsc : IsClosed s) (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ)
    (hgi : Integrable (g ∘ f) μ) :
    f =ᵐ[μ] const α (⨍ x, f x ∂μ) ∨ g (⨍ x, f x ∂μ) < ⨍ x, g (f x) ∂μ := by
  have : ∀ {t}, μ t ≠ 0 → (⨍ x in t, f x ∂μ) ∈ s ∧ g (⨍ x in t, f x ∂μ) ≤ ⨍ x in t, g (f x) ∂μ :=
    fun ht =>
    hg.convexOn.set_average_mem_epigraph hgc hsc ht (measure_ne_top _ _) (ae_restrict_of_ae hfs)
      hfi.integrableOn hgi.integrableOn
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    g : E → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hg : StrictConvexOn Real s g
    hgc : ContinuousOn g s
    hsc : IsClosed s
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    hgi : MeasureTheory.Integrable (Function.comp g f) μ
    this : ∀ {t : Set α}, Ne (μ t) 0 → And (Membership.mem s (MeasureTheory.averag …
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  refine (ae_eq_const_or_exists_average_ne_compl hfi).imp_right ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    g : E → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hg : StrictConvexOn Real s g
    hgc : ContinuousOn g s
    hsc : IsClosed s
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    hgi : MeasureTheory.Integrable (Function.comp g f) μ
    this : ∀ {t : Set α}, Ne (μ t) 0 → And (Membership.mem s (MeasureTheory.averag …
    ⊢ (Exists fun t => And (MeasurableSet t) (And (Ne (μ t) 0) (And (Ne (μ (HasCom …
  -/
  rintro ⟨t, hm, h₀, h₀', hne⟩
  rcases average_mem_openSegment_compl_self hm.nullMeasurableSet h₀ h₀' (hfi.prod_mk hgi) with
    ⟨a, b, ha, hb, hab, h_avg⟩
  rw [average_pair hfi hgi, average_pair hfi.integrableOn hgi.integrableOn,
    average_pair hfi.integrableOn hgi.integrableOn, Prod.smul_mk,
    Prod.smul_mk, Prod.mk_add_mk, Prod.mk.inj_iff] at h_avg
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    g : E → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hg : StrictConvexOn Real s g
    hgc : ContinuousOn g s
    hsc : IsClosed s
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    hgi : MeasureTheory.Integrable (Function.comp g f) μ
    this : ∀ {t : Set α}, Ne (μ t) 0 → And (Membership.mem s (MeasureTheory.averag …
    t : Set α
    hm : MeasurableSet t
    h₀ : Ne (μ t) 0
    h₀' : Ne (μ (HasCompl.compl t)) 0
    hne : Ne (MeasureTheory.average (μ.restrict t) fun x => f x) (MeasureTheory.av …
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    h_avg : And (Eq (HAdd.hAdd (HSMul.hSMul a (MeasureTheory.average (μ.restrict t …
    ⊢ LT.lt (g (MeasureTheory.average μ fun x => f x)) (MeasureTheory.average μ fu …
  -/
  simp only [Function.comp] at h_avg
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    s : Set E
    f : α → E
    g : E → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hg : StrictConvexOn Real s g
    hgc : ContinuousOn g s
    hsc : IsClosed s
    hfs : Filter.Eventually (fun x => Membership.mem s (f x)) (MeasureTheory.ae μ)
    hfi : MeasureTheory.Integrable f μ
    hgi : MeasureTheory.Integrable (Function.comp g f) μ
    this : ∀ {t : Set α}, Ne (μ t) 0 → And (Membership.mem s (MeasureTheory.averag …
    t : Set α
    hm : MeasurableSet t
    h₀ : Ne (μ t) 0
    h₀' : Ne (μ (HasCompl.compl t)) 0
    hne : Ne (MeasureTheory.average (μ.restrict t) fun x => f x) (MeasureTheory.av …
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    h_avg : And (Eq (HAdd.hAdd (HSMul.hSMul a (MeasureTheory.average (μ.restrict t …
    ⊢ LT.lt (g (MeasureTheory.average μ fun x => f x)) (MeasureTheory.average μ fu …
  -/
  rw [← h_avg.1, ← h_avg.2]
  calc
    g ((a • ⨍ x in t, f x ∂μ) + b • ⨍ x in tᶜ, f x ∂μ) <
        a * g (⨍ x in t, f x ∂μ) + b * g (⨍ x in tᶜ, f x ∂μ) :=
      hg.2 (this h₀).1 (this h₀').1 hne ha hb hab
    _ ≤ (a * ⨍ x in t, g (f x) ∂μ) + b * ⨍ x in tᶜ, g (f x) ∂μ :=
      add_le_add (mul_le_mul_of_nonneg_left (this h₀).2 ha.le)
        (mul_le_mul_of_nonneg_left (this h₀').2 hb.le)


/-- **Jensen's inequality**, strict version: if an integrable function `f : α → E` takes values in a
convex closed set `s`, and `g : E → ℝ` is continuous and strictly concave on `s`, then
either `f` is a.e. equal to its average value, or `⨍ x, g (f x) ∂μ < g (⨍ x, f x ∂μ)`. -/
theorem StrictConcaveOn.ae_eq_const_or_lt_map_average [IsFiniteMeasure μ]
    (hg : StrictConcaveOn ℝ s g) (hgc : ContinuousOn g s) (hsc : IsClosed s)
    (hfs : ∀ᵐ x ∂μ, f x ∈ s) (hfi : Integrable f μ) (hgi : Integrable (g ∘ f) μ) :
    f =ᵐ[μ] const α (⨍ x, f x ∂μ) ∨ (⨍ x, g (f x) ∂μ) < g (⨍ x, f x ∂μ) := by
  simpa only [Pi.neg_apply, average_neg, neg_lt_neg_iff] using
    hg.neg.ae_eq_const_or_map_average_lt hgc.neg hsc hfs hfi hgi.neg


/-- If `E` is a strictly convex normed space and `f : α → E` is a function such that `‖f x‖ ≤ C`
a.e., then either this function is a.e. equal to its average value, or the norm of its average value
is strictly less than `C`. -/
theorem ae_eq_const_or_norm_average_lt_of_norm_le_const [StrictConvexSpace ℝ E]
    (h_le : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) : f =ᵐ[μ] const α (⨍ x, f x ∂μ) ∨ ‖⨍ x, f x ∂μ‖ < C := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    C : Real
    inst✝ : StrictConvexSpace Real E
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  rcases le_or_lt C 0 with hC0 | hC0
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      C : Real
      inst✝ : StrictConvexSpace Real E
      h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
      hC0 : LE.le C 0
      ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
    -/
  · have : f =ᵐ[μ] 0 := h_le.mono fun x hx => norm_le_zero_iff.1 (hx.trans hC0)
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      C : Real
      inst✝ : StrictConvexSpace Real E
      h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
      hC0 : LE.le C 0
      this : (MeasureTheory.ae μ).EventuallyEq f 0
      ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
    -/
    simp only [average_congr this, Pi.zero_apply, average_zero]
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      C : Real
      inst✝ : StrictConvexSpace Real E
      h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
      hC0 : LE.le C 0
      this : (MeasureTheory.ae μ).EventuallyEq f 0
      ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α 0)) (LT.lt (Norm.n …
    -/
    exact Or.inl this
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    C : Real
    inst✝ : StrictConvexSpace Real E
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    hC0 : LT.lt 0 C
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  by_cases hfi : Integrable f μ; swap
    /-
      case neg
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      μ : MeasureTheory.Measure α
      f : α → E
      C : Real
      inst✝ : StrictConvexSpace Real E
      h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
      hC0 : LT.lt 0 C
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
    -/
  · simp [average_eq, integral_undef hfi, hC0, ENNReal.toReal_pos_iff]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    C : Real
    inst✝ : StrictConvexSpace Real E
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    hC0 : LT.lt 0 C
    hfi : MeasureTheory.Integrable f μ
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  rcases (le_top : μ univ ≤ ∞).eq_or_lt with hμt | hμt; · simp [average_eq, hμt, hC0]
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    case pos.inr
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    C : Real
    inst✝ : StrictConvexSpace Real E
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    hC0 : LT.lt 0 C
    hfi : MeasureTheory.Integrable f μ
    hμt : LT.lt (μ Set.univ) Top.top
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  haveI : IsFiniteMeasure μ := ⟨hμt⟩
  /-
    case pos.inr
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    C : Real
    inst✝ : StrictConvexSpace Real E
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    hC0 : LT.lt 0 C
    hfi : MeasureTheory.Integrable f μ
    hμt : LT.lt (μ Set.univ) Top.top
    this : MeasureTheory.IsFiniteMeasure μ
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  replace h_le : ∀ᵐ x ∂μ, f x ∈ closedBall (0 : E) C := by simpa only [mem_closedBall_zero_iff]
  simpa only [interior_closedBall _ hC0.ne', mem_ball_zero_iff] using
    (strictConvex_closedBall ℝ (0 : E) C).ae_eq_const_or_average_mem_interior isClosed_ball h_le
      hfi


/-- If `E` is a strictly convex normed space and `f : α → E` is a function such that `‖f x‖ ≤ C`
a.e., then either this function is a.e. equal to its average value, or the norm of its integral is
strictly less than `(μ univ).toReal * C`. -/
theorem ae_eq_const_or_norm_integral_lt_of_norm_le_const [StrictConvexSpace ℝ E] [IsFiniteMeasure μ]
    (h_le : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) :
    f =ᵐ[μ] const α (⨍ x, f x ∂μ) ∨ ‖∫ x, f x ∂μ‖ < (μ univ).toReal * C := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    C : Real
    inst✝¹ : StrictConvexSpace Real E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  rcases eq_or_ne μ 0 with h₀ | h₀; · left; simp [h₀, EventuallyEq]
                                            /-
                                              🎉 no goals
                                            -/
  have hμ : 0 < (μ univ).toReal := by
    simp [ENNReal.toReal_pos_iff, pos_iff_ne_zero, h₀, measure_lt_top]
  /-
    case inr
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    μ : MeasureTheory.Measure α
    f : α → E
    C : Real
    inst✝¹ : StrictConvexSpace Real E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    h₀ : Ne μ 0
    hμ : LT.lt 0 (μ Set.univ).toReal
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq f (Function.const α (MeasureTheory.ave …
  -/
  refine (ae_eq_const_or_norm_average_lt_of_norm_le_const h_le).imp_right fun H => ?_
  rwa [average_eq, norm_smul, norm_inv, Real.norm_eq_abs, abs_of_pos hμ, ← div_eq_inv_mul,
    div_lt_iff₀' hμ] at H


/-- If `E` is a strictly convex normed space and `f : α → E` is a function such that `‖f x‖ ≤ C`
a.e. on a set `t` of finite measure, then either this function is a.e. equal to its average value on
`t`, or the norm of its integral over `t` is strictly less than `(μ t).toReal * C`. -/
theorem ae_eq_const_or_norm_setIntegral_lt_of_norm_le_const [StrictConvexSpace ℝ E] (ht : μ t ≠ ∞)
    (h_le : ∀ᵐ x ∂μ.restrict t, ‖f x‖ ≤ C) :
    f =ᵐ[μ.restrict t] const α (⨍ x in t, f x ∂μ) ∨ ‖∫ x in t, f x ∂μ‖ < (μ t).toReal * C := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    t : Set α
    f : α → E
    C : Real
    inst✝ : StrictConvexSpace Real E
    ht : Ne (μ t) Top.top
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    ⊢ Or ((MeasureTheory.ae (μ.restrict t)).EventuallyEq f (Function.const α (Meas …
  -/
  haveI := Fact.mk ht.lt_top
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    t : Set α
    f : α → E
    C : Real
    inst✝ : StrictConvexSpace Real E
    ht : Ne (μ t) Top.top
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    this : Fact (LT.lt (μ t) Top.top)
    ⊢ Or ((MeasureTheory.ae (μ.restrict t)).EventuallyEq f (Function.const α (Meas …
  -/
  rw [← restrict_apply_univ]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure α
    t : Set α
    f : α → E
    C : Real
    inst✝ : StrictConvexSpace Real E
    ht : Ne (μ t) Top.top
    h_le : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.a …
    this : Fact (LT.lt (μ t) Top.top)
    ⊢ Or ((MeasureTheory.ae (μ.restrict t)).EventuallyEq f (Function.const α (Meas …
  -/
  exact ae_eq_const_or_norm_integral_lt_of_norm_le_const h_le
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_const_or_norm_set_integral_lt_of_norm_le_const :=
  ae_eq_const_or_norm_setIntegral_lt_of_norm_le_const

