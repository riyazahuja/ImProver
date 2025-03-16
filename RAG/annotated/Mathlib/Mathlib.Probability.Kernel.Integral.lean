lemma IsFiniteKernel.integrable (μ : Measure α) [IsFiniteMeasure μ]
    (κ : Kernel α β) [IsFiniteKernel κ] {s : Set β} (hs : MeasurableSet s) :
    Integrable (fun x ↦ (κ x s).toReal) μ := by
  refine Integrable.mono' (integrable_const (IsFiniteKernel.bound κ).toReal)
    ((κ.measurable_coe hs).ennreal_toReal.aestronglyMeasurable)
    (ae_of_all μ fun x ↦ ?_)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    s : Set β
    hs : MeasurableSet s
    x : α
    ⊢ LE.le (Norm.norm ((κ x) s).toReal) (ProbabilityTheory.IsFiniteKernel.bound κ …
  -/
  rw [Real.norm_eq_abs, abs_of_nonneg ENNReal.toReal_nonneg]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    s : Set β
    hs : MeasurableSet s
    x : α
    ⊢ LE.le ((κ x) s).toReal (ProbabilityTheory.IsFiniteKernel.bound κ).toReal
  -/
  exact ENNReal.toReal_mono (IsFiniteKernel.bound_ne_top _) (Kernel.measure_le_bound _ _ _)
  /-
    🎉 no goals
  -/


lemma IsMarkovKernel.integrable (μ : Measure α) [IsFiniteMeasure μ]
    (κ : Kernel α β) [IsMarkovKernel κ] {s : Set β} (hs : MeasurableSet s) :
    Integrable (fun x => (κ x s).toReal) μ :=
  IsFiniteKernel.integrable μ κ hs


lemma integral_congr_ae₂ {f g : α → β → E} {μ : Measure α} (h : ∀ᵐ a ∂μ, f a =ᵐ[κ a] g a) :
    ∫ a, ∫ b, f a b ∂(κ a) ∂μ = ∫ a, ∫ b, g a b ∂(κ a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → β → E
    μ : MeasureTheory.Measure α
    h : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq (f a) (g …
    ⊢ Eq (MeasureTheory.integral μ fun a => MeasureTheory.integral (κ a) fun b =>  …
  -/
  apply integral_congr_ae
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → β → E
    μ : MeasureTheory.Measure α
    h : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq (f a) (g …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => MeasureTheory.integral (κ a) fun …
  -/
  filter_upwards [h] with _ ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → β → E
    μ : MeasureTheory.Measure α
    h : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq (f a) (g …
    a✝ : α
    ha : (MeasureTheory.ae (κ a✝)).EventuallyEq (f a✝) (g a✝)
    ⊢ Eq (MeasureTheory.integral (κ a✝) fun b => f a✝ b) (MeasureTheory.integral ( …
  -/
  apply integral_congr_ae
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → β → E
    μ : MeasureTheory.Measure α
    h : Filter.Eventually (fun a => (MeasureTheory.ae (κ a)).EventuallyEq (f a) (g …
    a✝ : α
    ha : (MeasureTheory.ae (κ a✝)).EventuallyEq (f a✝) (g a✝)
    ⊢ (MeasureTheory.ae (κ a✝)).EventuallyEq (f a✝) (g a✝)
  -/
  filter_upwards [ha] with _ hb using hb
  /-
    🎉 no goals
  -/


theorem integral_deterministic' (hg : Measurable g) (hf : StronglyMeasurable f) :
    ∫ x, f x ∂deterministic g hg a = f (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    f : β → E
    a : α
    inst✝ : CompleteSpace E
    g : α → β
    hg : Measurable g
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (MeasureTheory.integral ((ProbabilityTheory.Kernel.deterministic g hg) a) …
  -/
  rw [deterministic_apply, integral_dirac' _ _ hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_deterministic [MeasurableSingletonClass β] (hg : Measurable g) :
    ∫ x, f x ∂deterministic g hg a = f (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : β → E
    a : α
    inst✝¹ : CompleteSpace E
    g : α → β
    inst✝ : MeasurableSingletonClass β
    hg : Measurable g
    ⊢ Eq (MeasureTheory.integral ((ProbabilityTheory.Kernel.deterministic g hg) a) …
  -/
  rw [deterministic_apply, integral_dirac _ (g a)]
  /-
    🎉 no goals
  -/


theorem setIntegral_deterministic' (hg : Measurable g)
    (hf : StronglyMeasurable f) {s : Set β} (hs : MeasurableSet s) [Decidable (g a ∈ s)] :
    ∫ x in s, f x ∂deterministic g hg a = if g a ∈ s then f (g a) else 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : β → E
    a : α
    inst✝¹ : CompleteSpace E
    g : α → β
    hg : Measurable g
    hf : MeasureTheory.StronglyMeasurable f
    s : Set β
    hs : MeasurableSet s
    inst✝ : Decidable (Membership.mem s (g a))
    ⊢ Eq (MeasureTheory.integral (((ProbabilityTheory.Kernel.deterministic g hg) a …
  -/
  rw [deterministic_apply, setIntegral_dirac' hf _ hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_deterministic' := setIntegral_deterministic'


@[simp]
theorem setIntegral_deterministic [MeasurableSingletonClass β] (hg : Measurable g)
    (s : Set β) [Decidable (g a ∈ s)] :
    ∫ x in s, f x ∂deterministic g hg a = if g a ∈ s then f (g a) else 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    f : β → E
    a : α
    inst✝² : CompleteSpace E
    g : α → β
    inst✝¹ : MeasurableSingletonClass β
    hg : Measurable g
    s : Set β
    inst✝ : Decidable (Membership.mem s (g a))
    ⊢ Eq (MeasureTheory.integral (((ProbabilityTheory.Kernel.deterministic g hg) a …
  -/
  rw [deterministic_apply, setIntegral_dirac f _ s]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_deterministic := setIntegral_deterministic


@[simp]
theorem integral_const {μ : Measure β} : ∫ x, f x ∂const α μ a = ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : β → E
    a : α
    μ : MeasureTheory.Measure β
    ⊢ Eq (MeasureTheory.integral ((ProbabilityTheory.Kernel.const α μ) a) fun x => …
  -/
  rw [const_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem setIntegral_const {μ : Measure β} {s : Set β} :
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          mα : MeasurableSpace α
                                                          mβ : MeasurableSpace β
                                                          E : Type u_3
                                                          inst✝¹ : NormedAddCommGroup E
                                                          inst✝ : NormedSpace Real E
                                                          f : β → E
                                                          a : α
                                                          μ : MeasureTheory.Measure β
                                                          s : Set β
                                                          ⊢ Eq (MeasureTheory.integral (((ProbabilityTheory.Kernel.const α μ) a).restric …
                                                        -/
    ∫ x in s, f x ∂const α μ a = ∫ x in s, f x ∂μ := by rw [const_apply]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[deprecated (since := "2024-04-17")]
alias set_integral_const := setIntegral_const


@[simp]
theorem integral_restrict (hs : MeasurableSet s) :
    ∫ x, f x ∂κ.restrict hs a = ∫ x in s, f x ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : β → E
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral ((κ.restrict hs) a) fun x => f x) (MeasureTheory. …
  -/
  rw [restrict_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem setIntegral_restrict (hs : MeasurableSet s) (t : Set β) :
    ∫ x in t, f x ∂κ.restrict hs a = ∫ x in t ∩ s, f x ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : β → E
    a : α
    s : Set β
    hs : MeasurableSet s
    t : Set β
    ⊢ Eq (MeasureTheory.integral (((κ.restrict hs) a).restrict t) fun x => f x) (M …
  -/
  rw [restrict_apply, Measure.restrict_restrict' hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_restrict := setIntegral_restrict


theorem integral_piecewise (a : α) (g : β → E) :
    ∫ b, g b ∂piecewise hs κ η a = if a ∈ s then ∫ b, g b ∂κ a else ∫ b, g b ∂η a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝ : DecidablePred fun x => Membership.mem s x
    a : α
    g : β → E
    ⊢ Eq (MeasureTheory.integral ((ProbabilityTheory.Kernel.piecewise hs κ η) a) f …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  simp_rw [piecewise_apply]; split_ifs <;> rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem setIntegral_piecewise (a : α) (g : β → E) (t : Set β) :
    ∫ b in t, g b ∂piecewise hs κ η a =
      if a ∈ s then ∫ b in t, g b ∂κ a else ∫ b in t, g b ∂η a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝ : DecidablePred fun x => Membership.mem s x
    a : α
    g : β → E
    t : Set β
    ⊢ Eq (MeasureTheory.integral (((ProbabilityTheory.Kernel.piecewise hs κ η) a). …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  simp_rw [piecewise_apply]; split_ifs <;> rfl
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-04-17")]
alias set_integral_piecewise := setIntegral_piecewise


