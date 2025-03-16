theorem ProbabilityTheory.measurableSet_kernel_integrable ⦃f : α → β → E⦄
    (hf : StronglyMeasurable (uncurry f)) :
    MeasurableSet {x | Integrable (f x) (κ x)} := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    ⊢ MeasurableSet (setOf fun x => MeasureTheory.Integrable (f x) (κ x))
  -/
  simp_rw [Integrable, hf.of_uncurry_left.aestronglyMeasurable, true_and]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry f)
    ⊢ MeasurableSet (setOf fun x => MeasureTheory.HasFiniteIntegral (f x) (κ x))
  -/
  exact measurableSet_lt (Measurable.lintegral_kernel_prod_right hf.ennnorm) measurable_const
  /-
    🎉 no goals
  -/


theorem StronglyMeasurable.integral_kernel_prod_right ⦃f : α → β → E⦄
    (hf : StronglyMeasurable (uncurry f)) : StronglyMeasurable fun x => ∫ y, f x y ∂κ x := by
  classical
  by_cases hE : CompleteSpace E; swap
  · simp [integral, hE, stronglyMeasurable_const]
  borelize E
  haveI : TopologicalSpace.SeparableSpace (range (uncurry f) ∪ {0} : Set E) :=
    hf.separableSpace_range_union_singleton
  let s : ℕ → SimpleFunc (α × β) E :=
    SimpleFunc.approxOn _ hf.measurable (range (uncurry f) ∪ {0}) 0 (by simp)
  let s' : ℕ → α → SimpleFunc β E := fun n x => (s n).comp (Prod.mk x) measurable_prod_mk_left
  let f' : ℕ → α → E := fun n =>
    {x | Integrable (f x) (κ x)}.indicator fun x => (s' n x).integral (κ x)
  have hf' : ∀ n, StronglyMeasurable (f' n) := by
    intro n; refine StronglyMeasurable.indicator ?_ (measurableSet_kernel_integrable hf)
    have : ∀ x, ((s' n x).range.filter fun x => x ≠ 0) ⊆ (s n).range := by
      intro x; refine Finset.Subset.trans (Finset.filter_subset _ _) ?_; intro y
      simp_rw [SimpleFunc.mem_range]; rintro ⟨z, rfl⟩; exact ⟨(x, z), rfl⟩
    simp only [SimpleFunc.integral_eq_sum_of_subset (this _)]
    refine Finset.stronglyMeasurable_sum _ fun x _ => ?_
    refine (Measurable.ennreal_toReal ?_).stronglyMeasurable.smul_const _
    simp only [s', SimpleFunc.coe_comp, preimage_comp]
    apply Kernel.measurable_kernel_prod_mk_left
    exact (s n).measurableSet_fiber x
  have h2f' : Tendsto f' atTop (𝓝 fun x : α => ∫ y : β, f x y ∂κ x) := by
    rw [tendsto_pi_nhds]; intro x
    by_cases hfx : Integrable (f x) (κ x)
    · have (n) : Integrable (s' n x) (κ x) := by
        apply (hfx.norm.add hfx.norm).mono' (s' n x).aestronglyMeasurable
        filter_upwards with y
        simp_rw [s', SimpleFunc.coe_comp]; exact SimpleFunc.norm_approxOn_zero_le _ _ (x, y) n
      simp only [f',  hfx, SimpleFunc.integral_eq_integral _ (this _), indicator_of_mem,
        mem_setOf_eq]
      refine
        tendsto_integral_of_dominated_convergence (fun y => ‖f x y‖ + ‖f x y‖)
          (fun n => (s' n x).aestronglyMeasurable) (hfx.norm.add hfx.norm) ?_ ?_
      · -- Porting note: was
        -- exact fun n => Eventually.of_forall fun y =>
        --   SimpleFunc.norm_approxOn_zero_le _ _ (x, y) n
        exact fun n => Eventually.of_forall fun y =>
          SimpleFunc.norm_approxOn_zero_le hf.measurable (by simp) (x, y) n
      · refine Eventually.of_forall fun y => SimpleFunc.tendsto_approxOn hf.measurable (by simp) ?_
        apply subset_closure
        simp [-uncurry_apply_pair]
    · simp [f', hfx, integral_undef]
  exact stronglyMeasurable_of_tendsto _ hf' h2f'


theorem StronglyMeasurable.integral_kernel_prod_right' ⦃f : α × β → E⦄ (hf : StronglyMeasurable f) :
    StronglyMeasurable fun x => ∫ y, f (x, y) ∂κ x := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : NormedSpace Real E
    f : Prod α β → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral (κ x) fun y …
  -/
  rw [← uncurry_curry f] at hf
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : NormedSpace Real E
    f : Prod α β → E
    hf : MeasureTheory.StronglyMeasurable (Function.uncurry (Function.curry f))
    ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral (κ x) fun y …
  -/
  exact hf.integral_kernel_prod_right
  /-
    🎉 no goals
  -/


theorem StronglyMeasurable.integral_kernel_prod_right'' {f : β × γ → E}
    (hf : StronglyMeasurable f) : StronglyMeasurable fun x => ∫ y, f (x, y) ∂η (a, x) := by
  change
    StronglyMeasurable
      ((fun x => ∫ y, (fun u : (α × β) × γ => f (u.1.2, u.2)) (x, y) ∂η x) ∘ fun x => (a, x))
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    a : α
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    inst✝ : NormedSpace Real E
    f : Prod β γ → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable (Function.comp (fun x => MeasureTheory.inte …
  -/
  apply StronglyMeasurable.comp_measurable _ (measurable_prod_mk_left (m := mα))
  · have := MeasureTheory.StronglyMeasurable.integral_kernel_prod_right' (κ := η)
      (hf.comp_measurable (measurable_fst.snd.prod_mk measurable_snd))
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      a : α
      E : Type u_4
      inst✝² : NormedAddCommGroup E
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      inst✝ : NormedSpace Real E
      f : Prod β γ → E
      hf : MeasureTheory.StronglyMeasurable f
      this : MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral (η x)  …
      ⊢ MeasureTheory.StronglyMeasurable fun x => MeasureTheory.integral (η x) fun y …
    -/
    simpa using this
    /-
      🎉 no goals
    -/


theorem StronglyMeasurable.integral_kernel_prod_left ⦃f : β → α → E⦄
    (hf : StronglyMeasurable (uncurry f)) : StronglyMeasurable fun y => ∫ x, f x y ∂κ y :=
  (hf.comp_measurable measurable_swap).integral_kernel_prod_right'


theorem StronglyMeasurable.integral_kernel_prod_left' ⦃f : β × α → E⦄ (hf : StronglyMeasurable f) :
    StronglyMeasurable fun y => ∫ x, f (x, y) ∂κ y :=
  (hf.comp_measurable measurable_swap).integral_kernel_prod_right'


theorem StronglyMeasurable.integral_kernel_prod_left'' {f : γ × β → E} (hf : StronglyMeasurable f) :
    StronglyMeasurable fun y => ∫ x, f (x, y) ∂η (a, y) := by
  change
    StronglyMeasurable
      ((fun y => ∫ x, (fun u : γ × α × β => f (u.1, u.2.2)) (x, y) ∂η y) ∘ fun x => (a, x))
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    a : α
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    inst✝ : NormedSpace Real E
    f : Prod γ β → E
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable (Function.comp (fun y => MeasureTheory.inte …
  -/
  apply StronglyMeasurable.comp_measurable _ (measurable_prod_mk_left (m := mα))
  · have := MeasureTheory.StronglyMeasurable.integral_kernel_prod_left' (κ := η)
      (hf.comp_measurable (measurable_fst.prod_mk measurable_snd.snd))
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      a : α
      E : Type u_4
      inst✝² : NormedAddCommGroup E
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      inst✝ : NormedSpace Real E
      f : Prod γ β → E
      hf : MeasureTheory.StronglyMeasurable f
      this : MeasureTheory.StronglyMeasurable fun y => MeasureTheory.integral (η y)  …
      ⊢ MeasureTheory.StronglyMeasurable fun y => MeasureTheory.integral (η y) fun x …
    -/
    simpa using this
    /-
      🎉 no goals
    -/


