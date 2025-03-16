theorem measurableSet_integrable [SFinite ν] ⦃f : α → β → E⦄
    (hf : StronglyMeasurable (uncurry f)) : MeasurableSet {x | Integrable (f x) ν} := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.SFinite ν
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    ⊢ MeasurableSet (setOf fun x => MeasureTheory.Integrable (f x) ν)
  -/
  simp_rw [Integrable, hf.of_uncurry_left.aestronglyMeasurable, true_and]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.SFinite ν
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    ⊢ MeasurableSet (setOf fun x => MeasureTheory.HasFiniteIntegral (f x) ν)
  -/
  exact measurableSet_lt (Measurable.lintegral_prod_right hf.ennnorm) measurable_const
  /-
    🎉 no goals
  -/


/-- The Bochner integral is measurable. This shows that the integrand of (the right-hand-side of)
  Fubini's theorem is measurable.
  This version has `f` in curried form. -/
theorem MeasureTheory.StronglyMeasurable.integral_prod_right [SFinite ν] ⦃f : α → β → E⦄
    (hf : StronglyMeasurable (uncurry f)) : StronglyMeasurable fun x => ∫ y, f x y ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite ν
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral ν fun y =>  …
  -/
  by_cases hE : CompleteSpace E; swap; · simp [integral, hE, stronglyMeasurable_const]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite ν
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    hE : CompleteSpace E
    ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral ν fun y =>  …
  -/
  borelize E
  haveI : SeparableSpace (range (uncurry f) ∪ {0} : Set E) :=
    hf.separableSpace_range_union_singleton
  let s : ℕ → SimpleFunc (α × β) E :=
    SimpleFunc.approxOn _ hf.measurable (range (uncurry f) ∪ {0}) 0 (by simp)
  /-
    case pos
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite ν
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    hE : CompleteSpace E
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range (Function.uncu …
    s : Nat → MeasureTheory.SimpleFunc (Prod α β) E := MeasureTheory.SimpleFunc.ap …
    ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral ν fun y =>  …
  -/
  let s' : ℕ → α → SimpleFunc β E := fun n x => (s n).comp (Prod.mk x) measurable_prod_mk_left
  /-
    case pos
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite ν
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    hE : CompleteSpace E
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range (Function.uncu …
    s : Nat → MeasureTheory.SimpleFunc (Prod α β) E := MeasureTheory.SimpleFunc.ap …
    s' : Nat → α → MeasureTheory.SimpleFunc β E := fun n x => (s n).comp (Prod.mk  …
    ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral ν fun y =>  …
  -/
  let f' : ℕ → α → E := fun n => {x | Integrable (f x) ν}.indicator fun x => (s' n x).integral ν
  have hf' : ∀ n, StronglyMeasurable (f' n) := by
    intro n; refine StronglyMeasurable.indicator ?_ (measurableSet_integrable hf)
    have : ∀ x, ((s' n x).range.filter fun x => x ≠ 0) ⊆ (s n).range := by
      intro x; refine Finset.Subset.trans (Finset.filter_subset _ _) ?_; intro y
      simp_rw [SimpleFunc.mem_range]; rintro ⟨z, rfl⟩; exact ⟨(x, z), rfl⟩
    simp only [SimpleFunc.integral_eq_sum_of_subset (this _)]
    refine Finset.stronglyMeasurable_sum _ fun x _ => ?_
    refine (Measurable.ennreal_toReal ?_).stronglyMeasurable.smul_const _
    simp only [s', SimpleFunc.coe_comp, preimage_comp]
    apply measurable_measure_prod_mk_left
    exact (s n).measurableSet_fiber x
  have h2f' : Tendsto f' atTop (𝓝 fun x : α => ∫ y : β, f x y ∂ν) := by
    rw [tendsto_pi_nhds]; intro x
    by_cases hfx : Integrable (f x) ν
    · have (n) : Integrable (s' n x) ν := by
        apply (hfx.norm.add hfx.norm).mono' (s' n x).aestronglyMeasurable
        filter_upwards with y
        simp_rw [s', SimpleFunc.coe_comp]; exact SimpleFunc.norm_approxOn_zero_le _ _ (x, y) n
      simp only [f', hfx, SimpleFunc.integral_eq_integral _ (this _), indicator_of_mem,
        mem_setOf_eq]
      refine
        tendsto_integral_of_dominated_convergence (fun y => ‖f x y‖ + ‖f x y‖)
          (fun n => (s' n x).aestronglyMeasurable) (hfx.norm.add hfx.norm) ?_ ?_
      · refine fun n => Eventually.of_forall fun y =>
          SimpleFunc.norm_approxOn_zero_le ?_ ?_ (x, y) n
        -- Porting note: Lean 3 solved the following two subgoals on its own
        · exact hf.measurable
        · simp
      · refine Eventually.of_forall fun y => SimpleFunc.tendsto_approxOn ?_ ?_ ?_
        -- Porting note: Lean 3 solved the following two subgoals on its own
        · exact hf.measurable.of_uncurry_left
        · simp
        apply subset_closure
        simp [-uncurry_apply_pair]
    · simp [f', hfx, integral_undef]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite ν
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    hE : CompleteSpace E
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range (Function.uncu …
    s : Nat → MeasureTheory.SimpleFunc (Prod α β) E := MeasureTheory.SimpleFunc.ap …
    s' : Nat → α → MeasureTheory.SimpleFunc β E := fun n x => (s n).comp (Prod.mk  …
    f' : Nat → α → E := fun n => (setOf fun x => MeasureTheory.Integrable (f x) ν) …
    hf' : ∀ (n : Nat), MeasureTheory.StronglyMeasurable (f' n)
    h2f' : Filter.Tendsto f' Filter.atTop (nhds fun x => MeasureTheory.integral ν  …
    ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral ν fun y =>  …
  -/
  exact stronglyMeasurable_of_tendsto _ hf' h2f'
  /-
    🎉 no goals
  -/


/-- The Bochner integral is measurable. This shows that the integrand of (the right-hand-side of)
  Fubini's theorem is measurable. -/
theorem MeasureTheory.StronglyMeasurable.integral_prod_right' [SFinite ν] ⦃f : α × β → E⦄
    (hf : StronglyMeasurable f) : StronglyMeasurable fun x => ∫ y, f (x, y) ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite ν
    f : Prod α β → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral ν fun y =>  …
  -/
  rw [← uncurry_curry f] at hf; exact hf.integral_prod_right
                                /-
                                  🎉 no goals
                                -/


/-- The Bochner integral is measurable. This shows that the integrand of (the right-hand-side of)
  the symmetric version of Fubini's theorem is measurable.
  This version has `f` in curried form. -/
theorem MeasureTheory.StronglyMeasurable.integral_prod_left [SFinite μ] ⦃f : α → β → E⦄
    (hf : StronglyMeasurable (uncurry f)) : StronglyMeasurable fun y => ∫ x, f x y ∂μ :=
  (hf.comp_measurable measurable_swap).integral_prod_right'


/-- The Bochner integral is measurable. This shows that the integrand of (the right-hand-side of)
  the symmetric version of Fubini's theorem is measurable. -/
theorem MeasureTheory.StronglyMeasurable.integral_prod_left' [SFinite μ] ⦃f : α × β → E⦄
    (hf : StronglyMeasurable f) : StronglyMeasurable fun y => ∫ x, f (x, y) ∂μ :=
  (hf.comp_measurable measurable_swap).integral_prod_right'


theorem integrable_measure_prod_mk_left {s : Set (α × β)} (hs : MeasurableSet s)
    (h2s : (μ.prod ν) s ≠ ∞) : Integrable (fun x => (ν (Prod.mk x ⁻¹' s)).toReal) μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne ((μ.prod ν) s) Top.top
    ⊢ MeasureTheory.Integrable (fun x => (ν (Set.preimage (Prod.mk x) s)).toReal) μ
  -/
  refine ⟨(measurable_measure_prod_mk_left hs).ennreal_toReal.aemeasurable.aestronglyMeasurable, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne ((μ.prod ν) s) Top.top
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => (ν (Set.preimage (Prod.mk x) s)).t …
  -/
  simp_rw [hasFiniteIntegral_iff_nnnorm, ennnorm_eq_ofReal toReal_nonneg]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne ((μ.prod ν) s) Top.top
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (ν (Set.preimage (P …
  -/
  convert h2s.lt_top using 1
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne ((μ.prod ν) s) Top.top
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (ν (Set.preimage (Prod …
  -/
  rw [prod_apply hs]
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne ((μ.prod ν) s) Top.top
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (ν (Set.preimage (Prod …
  -/
  apply lintegral_congr_ae
  /-
    case h.e'_3.h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne ((μ.prod ν) s) Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => ENNReal.ofReal (ν (Set.preimage  …
  -/
  filter_upwards [ae_measure_lt_top hs h2s] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    h2s : Ne ((μ.prod ν) s) Top.top
    x : α
    hx : LT.lt (ν (Set.preimage (Prod.mk x) s)) Top.top
    ⊢ Eq (ENNReal.ofReal (ν (Set.preimage (Prod.mk x) s)).toReal) (ν (Set.preimage …
  -/
  rw [lt_top_iff_ne_top] at hx; simp [ofReal_toReal, hx]
                                /-
                                  🎉 no goals
                                -/


nonrec theorem MeasureTheory.AEStronglyMeasurable.prod_swap {γ : Type*} [TopologicalSpace γ]
    [SFinite μ] [SFinite ν] {f : β × α → γ} (hf : AEStronglyMeasurable f (ν.prod μ)) :
    AEStronglyMeasurable (fun z : α × β => f z.swap) (μ.prod ν) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    γ : Type u_4
    inst✝² : TopologicalSpace γ
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    f : Prod β α → γ
    hf : MeasureTheory.AEStronglyMeasurable f (ν.prod μ)
    ⊢ MeasureTheory.AEStronglyMeasurable (fun z => f z.swap) (μ.prod ν)
  -/
  rw [← prod_swap] at hf
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    γ : Type u_4
    inst✝² : TopologicalSpace γ
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    f : Prod β α → γ
    hf : MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.map Prod.swap …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun z => f z.swap) (μ.prod ν)
  -/
  exact hf.comp_measurable measurable_swap
  /-
    🎉 no goals
  -/


theorem MeasureTheory.AEStronglyMeasurable.fst {γ} [TopologicalSpace γ] [SFinite ν] {f : α → γ}
    (hf : AEStronglyMeasurable f μ) : AEStronglyMeasurable (fun z : α × β => f z.1) (μ.prod ν) :=
  hf.comp_quasiMeasurePreserving quasiMeasurePreserving_fst


theorem MeasureTheory.AEStronglyMeasurable.snd {γ} [TopologicalSpace γ] [SFinite ν] {f : β → γ}
    (hf : AEStronglyMeasurable f ν) : AEStronglyMeasurable (fun z : α × β => f z.2) (μ.prod ν) :=
  hf.comp_quasiMeasurePreserving quasiMeasurePreserving_snd


/-- The Bochner integral is a.e.-measurable.
  This shows that the integrand of (the right-hand-side of) Fubini's theorem is a.e.-measurable. -/
theorem MeasureTheory.AEStronglyMeasurable.integral_prod_right' [SFinite ν] [NormedSpace ℝ E]
    ⦃f : α × β → E⦄ (hf : AEStronglyMeasurable f (μ.prod ν)) :
    AEStronglyMeasurable (fun x => ∫ y, f (x, y) ∂ν) μ :=
  ⟨fun x => ∫ y, hf.mk f (x, y) ∂ν, hf.stronglyMeasurable_mk.integral_prod_right', by
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝² : NormedAddCommGroup E
      inst✝¹ : MeasureTheory.SFinite ν
      inst✝ : NormedSpace Real E
      f : Prod α β → E
      hf : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => MeasureTheory.integral ν fun y = …
    -/
    filter_upwards [ae_ae_of_ae_prod hf.ae_eq_mk] with _ hx using integral_congr_ae hx⟩
    /-
      🎉 no goals
    -/


theorem MeasureTheory.AEStronglyMeasurable.prod_mk_left {γ : Type*} [SFinite ν]
    [TopologicalSpace γ] {f : α × β → γ} (hf : AEStronglyMeasurable f (μ.prod ν)) :
    ∀ᵐ x ∂μ, AEStronglyMeasurable (fun y => f (x, y)) ν := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    γ : Type u_4
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : TopologicalSpace γ
    f : Prod α β → γ
    hf : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
    ⊢ Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fun y => f { …
  -/
  filter_upwards [ae_ae_of_ae_prod hf.ae_eq_mk] with x hx
  exact
    ⟨fun y => hf.mk f (x, y), hf.stronglyMeasurable_mk.comp_measurable measurable_prod_mk_left, hx⟩


theorem integrable_swap_iff [SFinite μ] {f : α × β → E} :
    Integrable (f ∘ Prod.swap) (ν.prod μ) ↔ Integrable f (μ.prod ν) :=
  measurePreserving_swap.integrable_comp_emb MeasurableEquiv.prodComm.measurableEmbedding


theorem Integrable.swap [SFinite μ] ⦃f : α × β → E⦄ (hf : Integrable f (μ.prod ν)) :
    Integrable (f ∘ Prod.swap) (ν.prod μ) :=
  integrable_swap_iff.2 hf


theorem hasFiniteIntegral_prod_iff ⦃f : α × β → E⦄ (h1f : StronglyMeasurable f) :
    HasFiniteIntegral f (μ.prod ν) ↔
      (∀ᵐ x ∂μ, HasFiniteIntegral (fun y => f (x, y)) ν) ∧
        HasFiniteIntegral (fun x => ∫ y, ‖f (x, y)‖ ∂ν) μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.SFinite ν
    f : Prod α β → E
    h1f : MeasureTheory.StronglyMeasurable f
    ⊢ Iff (MeasureTheory.HasFiniteIntegral f (μ.prod ν)) (And (Filter.Eventually ( …
  -/
  simp only [hasFiniteIntegral_iff_nnnorm, lintegral_prod_of_measurable _ h1f.ennnorm]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.SFinite ν
    f : Prod α β → E
    h1f : MeasureTheory.StronglyMeasurable f
    ⊢ Iff (LT.lt (MeasureTheory.lintegral μ fun x => MeasureTheory.lintegral ν fun …
  -/
  have (x) : ∀ᵐ y ∂ν, 0 ≤ ‖f (x, y)‖ := by filter_upwards with y using norm_nonneg _
  simp_rw [integral_eq_lintegral_of_nonneg_ae (this _)
      (h1f.norm.comp_measurable measurable_prod_mk_left).aestronglyMeasurable,
    ennnorm_eq_ofReal toReal_nonneg, ofReal_norm_eq_coe_nnnorm]
  -- this fact is probably too specialized to be its own lemma
  have : ∀ {p q r : Prop} (_ : r → p), (r ↔ p ∧ q) ↔ p → (r ↔ q) := fun {p q r} h1 => by
    rw [← and_congr_right_iff, and_iff_right_of_imp h1]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.SFinite ν
    f : Prod α β → E
    h1f : MeasureTheory.StronglyMeasurable f
    this✝ : ∀ (x : α), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
    this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
    ⊢ Iff (LT.lt (MeasureTheory.lintegral μ fun x => MeasureTheory.lintegral ν fun …
  -/
  rw [this]
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.StronglyMeasurable f
      this✝ : ∀ (x : α), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
      this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
      ⊢ Filter.Eventually (fun x => LT.lt (MeasureTheory.lintegral ν fun a => ↑(NNNo …
    -/
  · intro h2f; rw [lintegral_congr_ae]
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.StronglyMeasurable f
      this✝ : ∀ (x : α), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
      this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
      h2f : Filter.Eventually (fun x => LT.lt (MeasureTheory.lintegral ν fun a => ↑( …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => MeasureTheory.lintegral ν fun y  …
    -/
    filter_upwards [h2f] with x hx
    /-
      case h
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.StronglyMeasurable f
      this✝ : ∀ (x : α), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
      this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
      h2f : Filter.Eventually (fun x => LT.lt (MeasureTheory.lintegral ν fun a => ↑( …
      x : α
      hx : LT.lt (MeasureTheory.lintegral ν fun a => ↑(NNNorm.nnnorm (f { fst := x,  …
      ⊢ Eq (MeasureTheory.lintegral ν fun y => ↑(NNNorm.nnnorm (f { fst := x, snd := …
    -/
    rw [ofReal_toReal]; rw [← lt_top_iff_ne_top]; exact hx
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.StronglyMeasurable f
      this✝ : ∀ (x : α), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
      this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
      ⊢ LT.lt (MeasureTheory.lintegral μ fun x => MeasureTheory.lintegral ν fun y => …
    -/
  · intro h2f; refine ae_lt_top ?_ h2f.ne; exact h1f.ennnorm.lintegral_prod_right'
                                           /-
                                             🎉 no goals
                                           -/


theorem hasFiniteIntegral_prod_iff' ⦃f : α × β → E⦄ (h1f : AEStronglyMeasurable f (μ.prod ν)) :
    HasFiniteIntegral f (μ.prod ν) ↔
      (∀ᵐ x ∂μ, HasFiniteIntegral (fun y => f (x, y)) ν) ∧
        HasFiniteIntegral (fun x => ∫ y, ‖f (x, y)‖ ∂ν) μ := by
  rw [hasFiniteIntegral_congr h1f.ae_eq_mk,
    hasFiniteIntegral_prod_iff h1f.stronglyMeasurable_mk]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.SFinite ν
    f : Prod α β → E
    h1f : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
    ⊢ Iff (And (Filter.Eventually (fun x => MeasureTheory.HasFiniteIntegral (fun y …
  -/
  apply and_congr
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
      ⊢ Iff (Filter.Eventually (fun x => MeasureTheory.HasFiniteIntegral (fun y => M …
    -/
  · apply eventually_congr
    /-
      case h₁.h
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
      ⊢ Filter.Eventually (fun x => Iff (MeasureTheory.HasFiniteIntegral (fun y => M …
    -/
    filter_upwards [ae_ae_of_ae_prod h1f.ae_eq_mk.symm]
    /-
      case h
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
      ⊢ ∀ (a : α), Filter.Eventually (fun y => Eq (MeasureTheory.AEStronglyMeasurabl …
    -/
    intro x hx
    /-
      case h
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
      x : α
      hx : Filter.Eventually (fun y => Eq (MeasureTheory.AEStronglyMeasurable.mk f h …
      ⊢ Iff (MeasureTheory.HasFiniteIntegral (fun y => MeasureTheory.AEStronglyMeasu …
    -/
    exact hasFiniteIntegral_congr hx
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.SFinite ν
      f : Prod α β → E
      h1f : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
      ⊢ Iff (MeasureTheory.HasFiniteIntegral (fun x => MeasureTheory.integral ν fun  …
    -/
  · apply hasFiniteIntegral_congr
    filter_upwards [ae_ae_of_ae_prod h1f.ae_eq_mk.symm] with _ hx using
      integral_congr_ae (EventuallyEq.fun_comp hx _)


/-- A binary function is integrable if the function `y ↦ f (x, y)` is integrable for almost every
  `x` and the function `x ↦ ∫ ‖f (x, y)‖ dy` is integrable. -/
theorem integrable_prod_iff ⦃f : α × β → E⦄ (h1f : AEStronglyMeasurable f (μ.prod ν)) :
    Integrable f (μ.prod ν) ↔
      (∀ᵐ x ∂μ, Integrable (fun y => f (x, y)) ν) ∧ Integrable (fun x => ∫ y, ‖f (x, y)‖ ∂ν) μ := by
  simp [Integrable, h1f, hasFiniteIntegral_prod_iff', h1f.norm.integral_prod_right',
    h1f.prod_mk_left]


/-- A binary function is integrable if the function `x ↦ f (x, y)` is integrable for almost every
  `y` and the function `y ↦ ∫ ‖f (x, y)‖ dx` is integrable. -/
theorem integrable_prod_iff' [SFinite μ] ⦃f : α × β → E⦄
    (h1f : AEStronglyMeasurable f (μ.prod ν)) :
    Integrable f (μ.prod ν) ↔
      (∀ᵐ y ∂ν, Integrable (fun x => f (x, y)) μ) ∧ Integrable (fun y => ∫ x, ‖f (x, y)‖ ∂μ) ν := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → E
    h1f : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
    ⊢ Iff (MeasureTheory.Integrable f (μ.prod ν)) (And (Filter.Eventually (fun y = …
  -/
  convert integrable_prod_iff h1f.prod_swap using 1
  /-
    case h.e'_1.a
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → E
    h1f : MeasureTheory.AEStronglyMeasurable f (μ.prod ν)
    ⊢ Iff (MeasureTheory.Integrable f (μ.prod ν)) (MeasureTheory.Integrable (fun z …
  -/
  rw [funext fun _ => Function.comp_apply.symm, integrable_swap_iff]
  /-
    🎉 no goals
  -/


theorem Integrable.prod_left_ae [SFinite μ] ⦃f : α × β → E⦄ (hf : Integrable f (μ.prod ν)) :
    ∀ᵐ y ∂ν, Integrable (fun x => f (x, y)) μ :=
  ((integrable_prod_iff' hf.aestronglyMeasurable).mp hf).1


theorem Integrable.prod_right_ae [SFinite μ] ⦃f : α × β → E⦄ (hf : Integrable f (μ.prod ν)) :
    ∀ᵐ x ∂μ, Integrable (fun y => f (x, y)) ν :=
  hf.swap.prod_left_ae


theorem Integrable.integral_norm_prod_left ⦃f : α × β → E⦄ (hf : Integrable f (μ.prod ν)) :
    Integrable (fun x => ∫ y, ‖f (x, y)‖ ∂ν) μ :=
  ((integrable_prod_iff hf.aestronglyMeasurable).mp hf).2


theorem Integrable.integral_norm_prod_right [SFinite μ] ⦃f : α × β → E⦄
    (hf : Integrable f (μ.prod ν)) : Integrable (fun y => ∫ x, ‖f (x, y)‖ ∂μ) ν :=
  hf.swap.integral_norm_prod_left


theorem Integrable.prod_smul {𝕜 : Type*} [NontriviallyNormedField 𝕜] [NormedSpace 𝕜 E]
    {f : α → 𝕜} {g : β → E} (hf : Integrable f μ) (hg : Integrable g ν) :
    Integrable (fun z : α × β => f z.1 • g z.2) (μ.prod ν) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    𝕜 : Type u_4
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    f : α → 𝕜
    g : β → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g ν
    ⊢ MeasureTheory.Integrable (fun z => HSMul.hSMul (f z.1) (g z.2)) (μ.prod ν)
  -/
  refine (integrable_prod_iff ?_).2 ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      𝕜 : Type u_4
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedSpace 𝕜 E
      f : α → 𝕜
      g : β → E
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g ν
      ⊢ MeasureTheory.AEStronglyMeasurable (fun z => HSMul.hSMul (f z.1) (g z.2)) (μ …
    -/
  · exact hf.1.fst.smul hg.1.snd
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      𝕜 : Type u_4
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedSpace 𝕜 E
      f : α → 𝕜
      g : β → E
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g ν
      ⊢ Filter.Eventually (fun x => MeasureTheory.Integrable (fun y => HSMul.hSMul ( …
    -/
  · exact Eventually.of_forall fun x => hg.smul (f x)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      𝕜 : Type u_4
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedSpace 𝕜 E
      f : α → 𝕜
      g : β → E
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g ν
      ⊢ MeasureTheory.Integrable (fun x => MeasureTheory.integral ν fun y => Norm.no …
    -/
  · simpa only [norm_smul, integral_mul_left] using hf.norm.mul_const _
    /-
      🎉 no goals
    -/


theorem Integrable.prod_mul {L : Type*} [RCLike L] {f : α → L} {g : β → L} (hf : Integrable f μ)
    (hg : Integrable g ν) : Integrable (fun z : α × β => f z.1 * g z.2) (μ.prod ν) :=
  hf.prod_smul hg


theorem Integrable.integral_prod_left ⦃f : α × β → E⦄ (hf : Integrable f (μ.prod ν)) :
    Integrable (fun x => ∫ y, f (x, y) ∂ν) μ :=
  Integrable.mono hf.integral_norm_prod_left hf.aestronglyMeasurable.integral_prod_right' <|
    Eventually.of_forall fun x =>
      (norm_integral_le_integral_norm _).trans_eq <|
        (norm_of_nonneg <|
            integral_nonneg_of_ae <|
              Eventually.of_forall fun y => (norm_nonneg (f (x, y)) : _)).symm


theorem Integrable.integral_prod_right [SFinite μ] ⦃f : α × β → E⦄
    (hf : Integrable f (μ.prod ν)) : Integrable (fun y => ∫ x, f (x, y) ∂μ) ν :=
  hf.swap.integral_prod_left


theorem integral_prod_swap (f : α × β → E) :
    ∫ z, f z.swap ∂ν.prod μ = ∫ z, f z ∂μ.prod ν :=
  measurePreserving_swap.integral_comp MeasurableEquiv.prodComm.measurableEmbedding _


/-- Integrals commute with addition inside another integral. `F` can be any function. -/
theorem integral_fn_integral_add ⦃f g : α × β → E⦄ (F : E → E') (hf : Integrable f (μ.prod ν))
    (hg : Integrable g (μ.prod ν)) :
    (∫ x, F (∫ y, f (x, y) + g (x, y) ∂ν) ∂μ) =
      ∫ x, F ((∫ y, f (x, y) ∂ν) + ∫ y, g (x, y) ∂ν) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    E' : Type u_4
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod α β → E
    F : E → E'
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    ⊢ Eq (MeasureTheory.integral μ fun x => F (MeasureTheory.integral ν fun y => H …
  -/
  refine integral_congr_ae ?_
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    E' : Type u_4
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod α β → E
    F : E → E'
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => F (MeasureTheory.integral ν fun  …
  -/
  filter_upwards [hf.prod_right_ae, hg.prod_right_ae] with _ h2f h2g
  /-
    case h
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    E' : Type u_4
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod α β → E
    F : E → E'
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    a✝ : α
    h2f : MeasureTheory.Integrable (fun y => f { fst := a✝, snd := y }) ν
    h2g : MeasureTheory.Integrable (fun y => g { fst := a✝, snd := y }) ν
    ⊢ Eq (F (MeasureTheory.integral ν fun y => HAdd.hAdd (f { fst := a✝, snd := y  …
  -/
  simp [integral_add h2f h2g]
  /-
    🎉 no goals
  -/


/-- Integrals commute with subtraction inside another integral.
  `F` can be any measurable function. -/
theorem integral_fn_integral_sub ⦃f g : α × β → E⦄ (F : E → E') (hf : Integrable f (μ.prod ν))
    (hg : Integrable g (μ.prod ν)) :
    (∫ x, F (∫ y, f (x, y) - g (x, y) ∂ν) ∂μ) =
      ∫ x, F ((∫ y, f (x, y) ∂ν) - ∫ y, g (x, y) ∂ν) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    E' : Type u_4
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod α β → E
    F : E → E'
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    ⊢ Eq (MeasureTheory.integral μ fun x => F (MeasureTheory.integral ν fun y => H …
  -/
  refine integral_congr_ae ?_
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    E' : Type u_4
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod α β → E
    F : E → E'
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => F (MeasureTheory.integral ν fun  …
  -/
  filter_upwards [hf.prod_right_ae, hg.prod_right_ae] with _ h2f h2g
  /-
    case h
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    E' : Type u_4
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod α β → E
    F : E → E'
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    a✝ : α
    h2f : MeasureTheory.Integrable (fun y => f { fst := a✝, snd := y }) ν
    h2g : MeasureTheory.Integrable (fun y => g { fst := a✝, snd := y }) ν
    ⊢ Eq (F (MeasureTheory.integral ν fun y => HSub.hSub (f { fst := a✝, snd := y  …
  -/
  simp [integral_sub h2f h2g]
  /-
    🎉 no goals
  -/


/-- Integrals commute with subtraction inside a lower Lebesgue integral.
  `F` can be any function. -/
theorem lintegral_fn_integral_sub ⦃f g : α × β → E⦄ (F : E → ℝ≥0∞) (hf : Integrable f (μ.prod ν))
    (hg : Integrable g (μ.prod ν)) :
    (∫⁻ x, F (∫ y, f (x, y) - g (x, y) ∂ν) ∂μ) =
      ∫⁻ x, F ((∫ y, f (x, y) ∂ν) - ∫ y, g (x, y) ∂ν) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f g : Prod α β → E
    F : E → ENNReal
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    ⊢ Eq (MeasureTheory.lintegral μ fun x => F (MeasureTheory.integral ν fun y =>  …
  -/
  refine lintegral_congr_ae ?_
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f g : Prod α β → E
    F : E → ENNReal
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => F (MeasureTheory.integral ν fun  …
  -/
  filter_upwards [hf.prod_right_ae, hg.prod_right_ae] with _ h2f h2g
  /-
    case h
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f g : Prod α β → E
    F : E → ENNReal
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hg : MeasureTheory.Integrable g (μ.prod ν)
    a✝ : α
    h2f : MeasureTheory.Integrable (fun y => f { fst := a✝, snd := y }) ν
    h2g : MeasureTheory.Integrable (fun y => g { fst := a✝, snd := y }) ν
    ⊢ Eq (F (MeasureTheory.integral ν fun y => HSub.hSub (f { fst := a✝, snd := y  …
  -/
  simp [integral_sub h2f h2g]
  /-
    🎉 no goals
  -/


/-- Double integrals commute with addition. -/
theorem integral_integral_add ⦃f g : α × β → E⦄ (hf : Integrable f (μ.prod ν))
    (hg : Integrable g (μ.prod ν)) :
    (∫ x, ∫ y, f (x, y) + g (x, y) ∂ν ∂μ) = (∫ x, ∫ y, f (x, y) ∂ν ∂μ) + ∫ x, ∫ y, g (x, y) ∂ν ∂μ :=
  (integral_fn_integral_add id hf hg).trans <|
    integral_add hf.integral_prod_left hg.integral_prod_left


/-- Double integrals commute with addition. This is the version with `(f + g) (x, y)`
  (instead of `f (x, y) + g (x, y)`) in the LHS. -/
theorem integral_integral_add' ⦃f g : α × β → E⦄ (hf : Integrable f (μ.prod ν))
    (hg : Integrable g (μ.prod ν)) :
    (∫ x, ∫ y, (f + g) (x, y) ∂ν ∂μ) = (∫ x, ∫ y, f (x, y) ∂ν ∂μ) + ∫ x, ∫ y, g (x, y) ∂ν ∂μ :=
  integral_integral_add hf hg


/-- Double integrals commute with subtraction. -/
theorem integral_integral_sub ⦃f g : α × β → E⦄ (hf : Integrable f (μ.prod ν))
    (hg : Integrable g (μ.prod ν)) :
    (∫ x, ∫ y, f (x, y) - g (x, y) ∂ν ∂μ) = (∫ x, ∫ y, f (x, y) ∂ν ∂μ) - ∫ x, ∫ y, g (x, y) ∂ν ∂μ :=
  (integral_fn_integral_sub id hf hg).trans <|
    integral_sub hf.integral_prod_left hg.integral_prod_left


/-- Double integrals commute with subtraction. This is the version with `(f - g) (x, y)`
  (instead of `f (x, y) - g (x, y)`) in the LHS. -/
theorem integral_integral_sub' ⦃f g : α × β → E⦄ (hf : Integrable f (μ.prod ν))
    (hg : Integrable g (μ.prod ν)) :
    (∫ x, ∫ y, (f - g) (x, y) ∂ν ∂μ) = (∫ x, ∫ y, f (x, y) ∂ν ∂μ) - ∫ x, ∫ y, g (x, y) ∂ν ∂μ :=
  integral_integral_sub hf hg


/-- The map that sends an L¹-function `f : α × β → E` to `∫∫f` is continuous. -/
theorem continuous_integral_integral :
    Continuous fun f : α × β →₁[μ.prod ν] E => ∫ x, ∫ y, f (x, y) ∂ν ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Continuous fun f => MeasureTheory.integral μ fun x => MeasureTheory.integral …
  -/
  rw [continuous_iff_continuousAt]; intro g
  refine
    tendsto_integral_of_L1 _ (L1.integrable_coeFn g).integral_prod_left
      (Eventually.of_forall fun h => (L1.integrable_coeFn h).integral_prod_left) ?_
  simp_rw [←
    lintegral_fn_integral_sub (fun x => (‖x‖₊ : ℝ≥0∞)) (L1.integrable_coeFn _)
      (L1.integrable_coeFn g)]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν)) x
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm  …
  -/
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds _ (fun i => zero_le _) _
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν)) x
      ⊢ (Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν)) x) → ENNR …
    -/
  · exact fun i => ∫⁻ x, ∫⁻ y, ‖i (x, y) - g (x, y)‖₊ ∂ν ∂μ
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν)) x
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => MeasureTheory.li …
  -/
  swap; · exact fun i => lintegral_mono fun x => ennnorm_integral_le_lintegral_ennnorm _
          /-
            🎉 no goals
          -/
  show
    Tendsto (fun i : α × β →₁[μ.prod ν] E => ∫⁻ x, ∫⁻ y : β, ‖i (x, y) - g (x, y)‖₊ ∂ν ∂μ) (𝓝 g)
      (𝓝 0)
  have : ∀ i : α × β →₁[μ.prod ν] E, Measurable fun z => (‖i z - g z‖₊ : ℝ≥0∞) := fun i =>
    ((Lp.stronglyMeasurable i).sub (Lp.stronglyMeasurable g)).ennnorm
  -- Porting note: was
  -- simp_rw [← lintegral_prod_of_measurable _ (this _), ← L1.ofReal_norm_sub_eq_lintegral, ←
  --   ofReal_zero]
  conv =>
    congr
    ext
    rw [← lintegral_prod_of_measurable _ (this _), ← L1.ofReal_norm_sub_eq_lintegral]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν)) x
    this : ∀ (i : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν) …
    ⊢ Filter.Tendsto (fun x => ENNReal.ofReal (Norm.norm (HSub.hSub x g))) (nhds g …
  -/
  rw [← ofReal_zero]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν)) x
    this : ∀ (i : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν) …
    ⊢ Filter.Tendsto (fun x => ENNReal.ofReal (Norm.norm (HSub.hSub x g))) (nhds g …
  -/
  refine (continuous_ofReal.tendsto 0).comp ?_
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν)) x
    this : ∀ (i : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.prod ν) …
    ⊢ Filter.Tendsto (fun x => Norm.norm (HSub.hSub x g)) (nhds g) (nhds 0)
  -/
  rw [← tendsto_iff_norm_sub_tendsto_zero]; exact tendsto_id
                                            /-
                                              🎉 no goals
                                            -/


/-- **Fubini's Theorem**: For integrable functions on `α × β`,
  the Bochner integral of `f` is equal to the iterated Bochner integral.
  `integrable_prod_iff` can be useful to show that the function in question in integrable.
  `MeasureTheory.Integrable.integral_prod_right` is useful to show that the inner integral
  of the right-hand side is integrable. -/
theorem integral_prod (f : α × β → E) (hf : Integrable f (μ.prod ν)) :
    ∫ z, f z ∂μ.prod ν = ∫ x, ∫ y, f (x, y) ∂ν ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → E
    hf : MeasureTheory.Integrable f (μ.prod ν)
    ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => f z) (MeasureTheory.integral  …
  -/
  by_cases hE : CompleteSpace E; swap; · simp only [integral, dif_neg hE]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → E
    hf : MeasureTheory.Integrable f (μ.prod ν)
    hE : CompleteSpace E
    ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => f z) (MeasureTheory.integral  …
  -/
  revert f
  /-
    case pos
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    hE : CompleteSpace E
    ⊢ ∀ (f : Prod α β → E), MeasureTheory.Integrable f (μ.prod ν) → Eq (MeasureThe …
  -/
  apply Integrable.induction
    /-
      case pos.h_ind
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      hE : CompleteSpace E
      ⊢ ∀ (c : E) ⦃s : Set (Prod α β)⦄, MeasurableSet s → LT.lt ((μ.prod ν) s) Top.t …
    -/
  · intro c s hs h2s
    simp_rw [integral_indicator hs, ← indicator_comp_right, Function.comp_def,
      integral_indicator (measurable_prod_mk_left hs), setIntegral_const, integral_smul_const,
      integral_toReal (measurable_measure_prod_mk_left hs).aemeasurable
        (ae_measure_lt_top hs h2s.ne)]
    /-
      case pos.h_ind
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      hE : CompleteSpace E
      c : E
      s : Set (Prod α β)
      hs : MeasurableSet s
      h2s : LT.lt ((μ.prod ν) s) Top.top
      ⊢ Eq (HSMul.hSMul ((μ.prod ν) s).toReal c) (HSMul.hSMul (MeasureTheory.lintegr …
    -/
    rw [prod_apply hs]
    /-
      🎉 no goals
    -/
    /-
      case pos.h_add
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      hE : CompleteSpace E
      ⊢ ∀ ⦃f g : Prod α β → E⦄, Disjoint (Function.support f) (Function.support g) → …
    -/
  · rintro f g - i_f i_g hf hg
    /-
      case pos.h_add
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      hE : CompleteSpace E
      f g : Prod α β → E
      i_f : MeasureTheory.Integrable f (μ.prod ν)
      i_g : MeasureTheory.Integrable g (μ.prod ν)
      hf : Eq (MeasureTheory.integral (μ.prod ν) fun z => f z) (MeasureTheory.integr …
      hg : Eq (MeasureTheory.integral (μ.prod ν) fun z => g z) (MeasureTheory.integr …
      ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => HAdd.hAdd f g z) (MeasureTheo …
    -/
    simp_rw [integral_add' i_f i_g, integral_integral_add' i_f i_g, hf, hg]
    /-
      🎉 no goals
    -/
    /-
      case pos.h_closed
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      hE : CompleteSpace E
      ⊢ IsClosed (setOf fun f => Eq (MeasureTheory.integral (μ.prod ν) fun z => ↑↑f  …
    -/
  · exact isClosed_eq continuous_integral continuous_integral_integral
    /-
      🎉 no goals
    -/
    /-
      case pos.h_ae
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁵ : MeasurableSpace α
      inst✝⁴ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      hE : CompleteSpace E
      ⊢ ∀ ⦃f g : Prod α β → E⦄, (MeasureTheory.ae (μ.prod ν)).EventuallyEq f g → Mea …
    -/
  · rintro f g hfg - hf; convert hf using 1
      /-
        case h.e'_2
        α : Type u_1
        β : Type u_2
        E : Type u_3
        inst✝⁵ : MeasurableSpace α
        inst✝⁴ : MeasurableSpace β
        μ : MeasureTheory.Measure α
        ν : MeasureTheory.Measure β
        inst✝³ : NormedAddCommGroup E
        inst✝² : MeasureTheory.SFinite ν
        inst✝¹ : NormedSpace Real E
        inst✝ : MeasureTheory.SFinite μ
        hE : CompleteSpace E
        f g : Prod α β → E
        hfg : (MeasureTheory.ae (μ.prod ν)).EventuallyEq f g
        hf : Eq (MeasureTheory.integral (μ.prod ν) fun z => f z) (MeasureTheory.integr …
        ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => g z) (MeasureTheory.integral  …
      -/
    · exact integral_congr_ae hfg.symm
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        α : Type u_1
        β : Type u_2
        E : Type u_3
        inst✝⁵ : MeasurableSpace α
        inst✝⁴ : MeasurableSpace β
        μ : MeasureTheory.Measure α
        ν : MeasureTheory.Measure β
        inst✝³ : NormedAddCommGroup E
        inst✝² : MeasureTheory.SFinite ν
        inst✝¹ : NormedSpace Real E
        inst✝ : MeasureTheory.SFinite μ
        hE : CompleteSpace E
        f g : Prod α β → E
        hfg : (MeasureTheory.ae (μ.prod ν)).EventuallyEq f g
        hf : Eq (MeasureTheory.integral (μ.prod ν) fun z => f z) (MeasureTheory.integr …
        ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.integral ν fun y => g {  …
      -/
    · apply integral_congr_ae
      /-
        case h.e'_3.h
        α : Type u_1
        β : Type u_2
        E : Type u_3
        inst✝⁵ : MeasurableSpace α
        inst✝⁴ : MeasurableSpace β
        μ : MeasureTheory.Measure α
        ν : MeasureTheory.Measure β
        inst✝³ : NormedAddCommGroup E
        inst✝² : MeasureTheory.SFinite ν
        inst✝¹ : NormedSpace Real E
        inst✝ : MeasureTheory.SFinite μ
        hE : CompleteSpace E
        f g : Prod α β → E
        hfg : (MeasureTheory.ae (μ.prod ν)).EventuallyEq f g
        hf : Eq (MeasureTheory.integral (μ.prod ν) fun z => f z) (MeasureTheory.integr …
        ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => MeasureTheory.integral ν fun y = …
      -/
      filter_upwards [ae_ae_of_ae_prod hfg] with x hfgx using integral_congr_ae (ae_eq_symm hfgx)
      /-
        🎉 no goals
      -/


/-- Symmetric version of **Fubini's Theorem**: For integrable functions on `α × β`,
  the Bochner integral of `f` is equal to the iterated Bochner integral.
  This version has the integrals on the right-hand side in the other order. -/
theorem integral_prod_symm (f : α × β → E) (hf : Integrable f (μ.prod ν)) :
    ∫ z, f z ∂μ.prod ν = ∫ y, ∫ x, f (x, y) ∂μ ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → E
    hf : MeasureTheory.Integrable f (μ.prod ν)
    ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => f z) (MeasureTheory.integral  …
  -/
  rw [← integral_prod_swap f]; exact integral_prod _ hf.swap
                               /-
                                 🎉 no goals
                               -/


/-- Reversed version of **Fubini's Theorem**. -/
theorem integral_integral {f : α → β → E} (hf : Integrable (uncurry f) (μ.prod ν)) :
    ∫ x, ∫ y, f x y ∂ν ∂μ = ∫ z, f z.1 z.2 ∂μ.prod ν :=
  (integral_prod _ hf).symm


/-- Reversed version of **Fubini's Theorem** (symmetric version). -/
theorem integral_integral_symm {f : α → β → E} (hf : Integrable (uncurry f) (μ.prod ν)) :
    ∫ x, ∫ y, f x y ∂ν ∂μ = ∫ z, f z.2 z.1 ∂ν.prod μ :=
  (integral_prod_symm _ hf.swap).symm


/-- Change the order of Bochner integration. -/
theorem integral_integral_swap ⦃f : α → β → E⦄ (hf : Integrable (uncurry f) (μ.prod ν)) :
    ∫ x, ∫ y, f x y ∂ν ∂μ = ∫ y, ∫ x, f x y ∂μ ∂ν :=
  (integral_integral hf).trans (integral_prod_symm _ hf)


/-- **Fubini's Theorem** for set integrals. -/
theorem setIntegral_prod (f : α × β → E) {s : Set α} {t : Set β}
    (hf : IntegrableOn f (s ×ˢ t) (μ.prod ν)) :
    ∫ z in s ×ˢ t, f z ∂μ.prod ν = ∫ x in s, ∫ y in t, f (x, y) ∂ν ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → E
    s : Set α
    t : Set β
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) (μ.prod ν)
    ⊢ Eq (MeasureTheory.integral ((μ.prod ν).restrict (SProd.sprod s t)) fun z =>  …
  -/
  simp only [← Measure.prod_restrict s t, IntegrableOn] at hf ⊢
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : Prod α β → E
    s : Set α
    t : Set β
    hf : MeasureTheory.Integrable f ((μ.restrict s).prod (ν.restrict t))
    ⊢ Eq (MeasureTheory.integral ((μ.restrict s).prod (ν.restrict t)) fun z => f z …
  -/
  exact integral_prod f hf
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")] alias set_integral_prod := setIntegral_prod


theorem integral_prod_smul {𝕜 : Type*} [RCLike 𝕜] [NormedSpace 𝕜 E] (f : α → 𝕜) (g : β → E) :
    ∫ z, f z.1 • g z.2 ∂μ.prod ν = (∫ x, f x ∂μ) • ∫ y, g y ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    𝕜 : Type u_5
    inst✝¹ : RCLike 𝕜
    inst✝ : NormedSpace 𝕜 E
    f : α → 𝕜
    g : β → E
    ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => HSMul.hSMul (f z.1) (g z.2))  …
  -/
  by_cases hE : CompleteSpace E; swap; · simp [integral, hE]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    𝕜 : Type u_5
    inst✝¹ : RCLike 𝕜
    inst✝ : NormedSpace 𝕜 E
    f : α → 𝕜
    g : β → E
    hE : CompleteSpace E
    ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => HSMul.hSMul (f z.1) (g z.2))  …
  -/
  by_cases h : Integrable (fun z : α × β => f z.1 • g z.2) (μ.prod ν)
    /-
      case pos
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁷ : MeasurableSpace α
      inst✝⁶ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : MeasureTheory.SFinite ν
      inst✝³ : NormedSpace Real E
      inst✝² : MeasureTheory.SFinite μ
      𝕜 : Type u_5
      inst✝¹ : RCLike 𝕜
      inst✝ : NormedSpace 𝕜 E
      f : α → 𝕜
      g : β → E
      hE : CompleteSpace E
      h : MeasureTheory.Integrable (fun z => HSMul.hSMul (f z.1) (g z.2)) (μ.prod ν)
      ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => HSMul.hSMul (f z.1) (g z.2))  …
    -/
  · rw [integral_prod _ h]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁷ : MeasurableSpace α
      inst✝⁶ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : MeasureTheory.SFinite ν
      inst✝³ : NormedSpace Real E
      inst✝² : MeasureTheory.SFinite μ
      𝕜 : Type u_5
      inst✝¹ : RCLike 𝕜
      inst✝ : NormedSpace 𝕜 E
      f : α → 𝕜
      g : β → E
      hE : CompleteSpace E
      h : MeasureTheory.Integrable (fun z => HSMul.hSMul (f z.1) (g z.2)) (μ.prod ν)
      ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.integral ν fun y => HSMu …
    -/
    simp_rw [integral_smul, integral_smul_const]
    /-
      🎉 no goals
    -/
  have H : ¬Integrable f μ ∨ ¬Integrable g ν := by
    contrapose! h
    exact h.1.prod_smul h.2
  /-
    case neg
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : NormedSpace Real E
    inst✝² : MeasureTheory.SFinite μ
    𝕜 : Type u_5
    inst✝¹ : RCLike 𝕜
    inst✝ : NormedSpace 𝕜 E
    f : α → 𝕜
    g : β → E
    hE : CompleteSpace E
    h : Not (MeasureTheory.Integrable (fun z => HSMul.hSMul (f z.1) (g z.2)) (μ.pr …
    H : Or (Not (MeasureTheory.Integrable f μ)) (Not (MeasureTheory.Integrable g ν))
    ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => HSMul.hSMul (f z.1) (g z.2))  …
  -/
                        /-
                          🎉 no goals
                        -/
  cases' H with H H <;> simp [integral_undef h, integral_undef H]
                        /-
                          🎉 no goals
                        -/


theorem integral_prod_mul {L : Type*} [RCLike L] (f : α → L) (g : β → L) :
    ∫ z, f z.1 * g z.2 ∂μ.prod ν = (∫ x, f x ∂μ) * ∫ y, g y ∂ν :=
  integral_prod_smul f g


theorem setIntegral_prod_mul {L : Type*} [RCLike L] (f : α → L) (g : β → L) (s : Set α)
    (t : Set β) :
    ∫ z in s ×ˢ t, f z.1 * g z.2 ∂μ.prod ν = (∫ x in s, f x ∂μ) * ∫ y in t, g y ∂ν := by
  -- Porting note: added
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    L : Type u_5
    inst✝ : RCLike L
    f : α → L
    g : β → L
    s : Set α
    t : Set β
    ⊢ Eq (MeasureTheory.integral ((μ.prod ν).restrict (SProd.sprod s t)) fun z =>  …
  -/
  rw [← Measure.prod_restrict s t]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : MeasureTheory.SFinite μ
    L : Type u_5
    inst✝ : RCLike L
    f : α → L
    g : β → L
    s : Set α
    t : Set β
    ⊢ Eq (MeasureTheory.integral ((μ.restrict s).prod (ν.restrict t)) fun z => HMu …
  -/
  apply integral_prod_mul
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")] alias set_integral_prod_mul := setIntegral_prod_mul


theorem integral_fun_snd (f : β → E) : ∫ z, f z.2 ∂μ.prod ν = (μ univ).toReal • ∫ y, f y ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : β → E
    ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => f z.2) (HSMul.hSMul (μ Set.un …
  -/
  simpa using integral_prod_smul (1 : α → ℝ) f
  /-
    🎉 no goals
  -/


theorem integral_fun_fst (f : α → E) : ∫ z, f z.1 ∂μ.prod ν = (ν univ).toReal • ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : α → E
    ⊢ Eq (MeasureTheory.integral (μ.prod ν) fun z => f z.1) (HSMul.hSMul (ν Set.un …
  -/
  rw [← integral_prod_swap]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : α → E
    ⊢ Eq (MeasureTheory.integral (ν.prod μ) fun z => f z.swap.1) (HSMul.hSMul (ν S …
  -/
  apply integral_fun_snd
  /-
    🎉 no goals
  -/


/-- A version of *Fubini theorem* for continuous functions with compact support: one may swap
the order of integration with respect to locally finite measures. One does not assume that the
measures are σ-finite, contrary to the usual Fubini theorem. -/
lemma integral_integral_swap_of_hasCompactSupport
    {f : X → Y → E} (hf : Continuous f.uncurry) (h'f : HasCompactSupport f.uncurry)
    {μ : Measure X} {ν : Measure Y} [IsFiniteMeasureOnCompacts μ] [IsFiniteMeasureOnCompacts ν] :
    ∫ x, (∫ y, f x y ∂ν) ∂μ = ∫ y, (∫ x, f x y ∂μ) ∂ν := by
  /-
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    X : Type u_5
    Y : Type u_6
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : OpensMeasurableSpace X
    inst✝² : OpensMeasurableSpace Y
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    h'f : HasCompactSupport (Function.uncurry f)
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.integral ν fun y => f x  …
  -/
  let U := Prod.fst '' (tsupport f.uncurry)
  /-
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    X : Type u_5
    Y : Type u_6
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : OpensMeasurableSpace X
    inst✝² : OpensMeasurableSpace Y
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    h'f : HasCompactSupport (Function.uncurry f)
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    U : Set X := Set.image Prod.fst (tsupport (Function.uncurry f))
    ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.integral ν fun y => f x  …
  -/
  have : Fact (μ U < ∞) := ⟨(IsCompact.image h'f continuous_fst).measure_lt_top⟩
  /-
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    X : Type u_5
    Y : Type u_6
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : OpensMeasurableSpace X
    inst✝² : OpensMeasurableSpace Y
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    h'f : HasCompactSupport (Function.uncurry f)
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    U : Set X := Set.image Prod.fst (tsupport (Function.uncurry f))
    this : Fact (LT.lt (μ U) Top.top)
    ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.integral ν fun y => f x  …
  -/
  let V := Prod.snd '' (tsupport f.uncurry)
  /-
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    X : Type u_5
    Y : Type u_6
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : OpensMeasurableSpace X
    inst✝² : OpensMeasurableSpace Y
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    h'f : HasCompactSupport (Function.uncurry f)
    μ : MeasureTheory.Measure X
    ν : MeasureTheory.Measure Y
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts ν
    U : Set X := Set.image Prod.fst (tsupport (Function.uncurry f))
    this : Fact (LT.lt (μ U) Top.top)
    V : Set Y := Set.image Prod.snd (tsupport (Function.uncurry f))
    ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.integral ν fun y => f x  …
  -/
  have : Fact (ν V < ∞) := ⟨(IsCompact.image h'f continuous_snd).measure_lt_top⟩
  calc
  ∫ x, (∫ y, f x y ∂ν) ∂μ = ∫ x, (∫ y in V, f x y ∂ν) ∂μ := by
    congr 1 with x
    apply (setIntegral_eq_integral_of_forall_compl_eq_zero (fun y hy ↦ ?_)).symm
    contrapose! hy
    have : (x, y) ∈ Function.support f.uncurry := hy
    exact mem_image_of_mem _ (subset_tsupport _ this)
  _ = ∫ x in U, (∫ y in V, f x y ∂ν) ∂μ := by
    apply (setIntegral_eq_integral_of_forall_compl_eq_zero (fun x hx ↦ ?_)).symm
    have : ∀ y, f x y = 0 := by
      intro y
      contrapose! hx
      have : (x, y) ∈ Function.support f.uncurry := hx
      exact mem_image_of_mem _ (subset_tsupport _ this)
    simp [this]
  _ = ∫ y in V, (∫ x in U, f x y ∂μ) ∂ν := by
    apply integral_integral_swap
    apply (integrableOn_iff_integrable_of_support_subset (subset_tsupport f.uncurry)).mp
    refine ⟨(h'f.stronglyMeasurable_of_prod hf).aestronglyMeasurable, ?_⟩
    obtain ⟨C, hC⟩ : ∃ C, ∀ p, ‖f.uncurry p‖ ≤ C := hf.bounded_above_of_compact_support h'f
    exact hasFiniteIntegral_of_bounded (C := C) (Eventually.of_forall hC)
  _ = ∫ y, (∫ x in U, f x y ∂μ) ∂ν := by
    apply setIntegral_eq_integral_of_forall_compl_eq_zero (fun y hy ↦ ?_)
    have : ∀ x, f x y = 0 := by
      intro x
      contrapose! hy
      have : (x, y) ∈ Function.support f.uncurry := hy
      exact mem_image_of_mem _ (subset_tsupport _ this)
    simp [this]
  _ = ∫ y, (∫ x, f x y ∂μ) ∂ν := by
    congr 1 with y
    apply setIntegral_eq_integral_of_forall_compl_eq_zero (fun x hx ↦ ?_)
    contrapose! hx
    have : (x, y) ∈ Function.support f.uncurry := hx
    exact mem_image_of_mem _ (subset_tsupport _ this)


