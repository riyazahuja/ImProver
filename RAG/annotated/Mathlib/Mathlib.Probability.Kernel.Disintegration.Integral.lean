lemma lintegral_condKernel_mem (a : α) {s : Set (β × Ω)} (hs : MeasurableSet s) :
    ∫⁻ x, Kernel.condKernel κ (a, x) {y | (x, y) ∈ s} ∂(Kernel.fst κ a) = κ a s := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    a : α
    s : Set (Prod β Ω)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ.fst a) fun x => (κ.condKernel { fst := a, snd …
  -/
  conv_rhs => rw [← κ.disintegrate κ.condKernel]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    a : α
    s : Set (Prod β Ω)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (κ.fst a) fun x => (κ.condKernel { fst := a, snd …
  -/
  simp_rw [Kernel.compProd_apply hs]
  /-
    🎉 no goals
  -/


lemma setLIntegral_condKernel_eq_measure_prod (a : α) {s : Set β} (hs : MeasurableSet s)
    {t : Set Ω} (ht : MeasurableSet t) :
    ∫⁻ b in s, Kernel.condKernel κ (a, b) t ∂(Kernel.fst κ a) = κ a (s ×ˢ t) := by
  have : κ a (s ×ˢ t) = (Kernel.fst κ ⊗ₖ Kernel.condKernel κ) a (s ×ˢ t) := by
    congr; exact (κ.disintegrate _).symm
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    a : α
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    this : Eq ((κ a) (SProd.sprod s t)) (((κ.fst.compProd κ.condKernel) a) (SProd. …
    ⊢ Eq (MeasureTheory.lintegral ((κ.fst a).restrict s) fun b => (κ.condKernel {  …
  -/
  rw [this, Kernel.compProd_apply (hs.prod ht)]
  classical
  have : ∀ b, Kernel.condKernel κ (a, b) {c | (b, c) ∈ s ×ˢ t}
      = s.indicator (fun b ↦ Kernel.condKernel κ (a, b) t) b := by
    intro b
    by_cases hb : b ∈ s <;> simp [hb]
  simp_rw [this]
  rw [lintegral_indicator hs]


@[deprecated (since := "2024-06-29")]
alias set_lintegral_condKernel_eq_measure_prod := setLIntegral_condKernel_eq_measure_prod


lemma lintegral_condKernel (hf : Measurable f) (a : α) :
    ∫⁻ b, ∫⁻ ω, f (b, ω) ∂(Kernel.condKernel κ (a, b)) ∂(Kernel.fst κ a) = ∫⁻ x, f x ∂(κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    a : α
    ⊢ Eq (MeasureTheory.lintegral (κ.fst a) fun b => MeasureTheory.lintegral (κ.co …
  -/
  conv_rhs => rw [← κ.disintegrate κ.condKernel]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    a : α
    ⊢ Eq (MeasureTheory.lintegral (κ.fst a) fun b => MeasureTheory.lintegral (κ.co …
  -/
  rw [Kernel.lintegral_compProd _ _ _ hf]
  /-
    🎉 no goals
  -/


lemma setLIntegral_condKernel (hf : Measurable f) (a : α) {s : Set β}
    (hs : MeasurableSet s) {t : Set Ω} (ht : MeasurableSet t) :
    ∫⁻ b in s, ∫⁻ ω in t, f (b, ω) ∂(Kernel.condKernel κ (a, b)) ∂(Kernel.fst κ a)
      = ∫⁻ x in s ×ˢ t, f x ∂(κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    a : α
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral ((κ.fst a).restrict s) fun b => MeasureTheory.li …
  -/
  conv_rhs => rw [← κ.disintegrate κ.condKernel]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    a : α
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral ((κ.fst a).restrict s) fun b => MeasureTheory.li …
  -/
  rw [Kernel.setLIntegral_compProd _ _ _ hf hs ht]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_condKernel := setLIntegral_condKernel


lemma setLIntegral_condKernel_univ_right (hf : Measurable f) (a : α) {s : Set β}
    (hs : MeasurableSet s) :
    ∫⁻ b in s, ∫⁻ ω, f (b, ω) ∂(Kernel.condKernel κ (a, b)) ∂(Kernel.fst κ a)
      = ∫⁻ x in s ×ˢ Set.univ, f x ∂(κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((κ.fst a).restrict s) fun b => MeasureTheory.li …
  -/
  rw [← setLIntegral_condKernel hf a hs MeasurableSet.univ]; simp_rw [Measure.restrict_univ]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_condKernel_univ_right := setLIntegral_condKernel_univ_right


lemma setLIntegral_condKernel_univ_left (hf : Measurable f) (a : α) {t : Set Ω}
    (ht : MeasurableSet t) :
    ∫⁻ b, ∫⁻ ω in t, f (b, ω) ∂(Kernel.condKernel κ (a, b)) ∂(Kernel.fst κ a)
      = ∫⁻ x in Set.univ ×ˢ t, f x ∂(κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    a : α
    t : Set Ω
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (κ.fst a) fun b => MeasureTheory.lintegral ((κ.c …
  -/
  rw [← setLIntegral_condKernel hf a MeasurableSet.univ ht]; simp_rw [Measure.restrict_univ]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_condKernel_univ_left := setLIntegral_condKernel_univ_left


lemma _root_.MeasureTheory.AEStronglyMeasurable.integral_kernel_condKernel (a : α)
    (hf : AEStronglyMeasurable f (κ a)) :
    AEStronglyMeasurable (fun x ↦ ∫ y, f (x, y) ∂(Kernel.condKernel κ (a, x)))
      (Kernel.fst κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    hf : MeasureTheory.AEStronglyMeasurable f (κ a)
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral (κ.condK …
  -/
  rw [← κ.disintegrate κ.condKernel] at hf
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    hf : MeasureTheory.AEStronglyMeasurable f ((κ.fst.compProd κ.condKernel) a)
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral (κ.condK …
  -/
  exact AEStronglyMeasurable.integral_kernel_compProd hf
  /-
    🎉 no goals
  -/


lemma integral_condKernel (a : α) (hf : Integrable f (κ a)) :
    ∫ b, ∫ ω, f (b, ω) ∂(Kernel.condKernel κ (a, b)) ∂(Kernel.fst κ a) = ∫ x, f x ∂(κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    hf : MeasureTheory.Integrable f (κ a)
    ⊢ Eq (MeasureTheory.integral (κ.fst a) fun b => MeasureTheory.integral (κ.cond …
  -/
  conv_rhs => rw [← κ.disintegrate κ.condKernel]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    hf : MeasureTheory.Integrable f (κ a)
    ⊢ Eq (MeasureTheory.integral (κ.fst a) fun b => MeasureTheory.integral (κ.cond …
  -/
  rw [← κ.disintegrate κ.condKernel] at hf
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    hf : MeasureTheory.Integrable f ((κ.fst.compProd κ.condKernel) a)
    ⊢ Eq (MeasureTheory.integral (κ.fst a) fun b => MeasureTheory.integral (κ.cond …
  -/
  rw [integral_compProd hf]
  /-
    🎉 no goals
  -/


lemma setIntegral_condKernel (a : α) {s : Set β} (hs : MeasurableSet s)
    {t : Set Ω} (ht : MeasurableSet t) (hf : IntegrableOn f (s ×ˢ t) (κ a)) :
    ∫ b in s, ∫ ω in t, f (b, ω) ∂(Kernel.condKernel κ (a, b)) ∂(Kernel.fst κ a)
      = ∫ x in s ×ˢ t, f x ∂(κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) (κ a)
    ⊢ Eq (MeasureTheory.integral ((κ.fst a).restrict s) fun b => MeasureTheory.int …
  -/
  conv_rhs => rw [← κ.disintegrate κ.condKernel]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) (κ a)
    ⊢ Eq (MeasureTheory.integral ((κ.fst a).restrict s) fun b => MeasureTheory.int …
  -/
  rw [← κ.disintegrate κ.condKernel] at hf
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) ((κ.fst.compProd κ.condKer …
    ⊢ Eq (MeasureTheory.integral ((κ.fst a).restrict s) fun b => MeasureTheory.int …
  -/
  rw [setIntegral_compProd hs ht hf]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_condKernel := setIntegral_condKernel


lemma setIntegral_condKernel_univ_right (a : α) {s : Set β} (hs : MeasurableSet s)
    (hf : IntegrableOn f (s ×ˢ Set.univ) (κ a)) :
    ∫ b in s, ∫ ω, f (b, ω) ∂(Kernel.condKernel κ (a, b)) ∂(Kernel.fst κ a)
      = ∫ x in s ×ˢ Set.univ, f x ∂(κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    s : Set β
    hs : MeasurableSet s
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s Set.univ) (κ a)
    ⊢ Eq (MeasureTheory.integral ((κ.fst a).restrict s) fun b => MeasureTheory.int …
  -/
  rw [← setIntegral_condKernel a hs MeasurableSet.univ hf]; simp_rw [Measure.restrict_univ]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[deprecated (since := "2024-04-17")]
alias set_integral_condKernel_univ_right := setIntegral_condKernel_univ_right


lemma setIntegral_condKernel_univ_left (a : α) {t : Set Ω} (ht : MeasurableSet t)
    (hf : IntegrableOn f (Set.univ ×ˢ t) (κ a)) :
    ∫ b, ∫ ω in t, f (b, ω) ∂(Kernel.condKernel κ (a, b)) ∂(Kernel.fst κ a)
      = ∫ x in Set.univ ×ˢ t, f x ∂(κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace Ω
    inst✝⁵ : StandardBorelSpace Ω
    inst✝⁴ : Nonempty Ω
    inst✝³ : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝² : ProbabilityTheory.IsFiniteKernel κ
    E : Type u_4
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : α
    t : Set Ω
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod Set.univ t) (κ a)
    ⊢ Eq (MeasureTheory.integral (κ.fst a) fun b => MeasureTheory.integral ((κ.con …
  -/
  rw [← setIntegral_condKernel a MeasurableSet.univ ht hf]; simp_rw [Measure.restrict_univ]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[deprecated (since := "2024-04-17")]
alias set_integral_condKernel_univ_left := setIntegral_condKernel_univ_left


lemma lintegral_condKernel_mem {s : Set (β × Ω)} (hs : MeasurableSet s) :
    ∫⁻ x, ρ.condKernel x {y | (x, y) ∈ s} ∂ρ.fst = ρ s := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set (Prod β Ω)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ρ.fst fun x => (ρ.condKernel x) (setOf fun y =>  …
  -/
  conv_rhs => rw [← ρ.disintegrate ρ.condKernel]
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set (Prod β Ω)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ρ.fst fun x => (ρ.condKernel x) (setOf fun y =>  …
  -/
  simp_rw [compProd_apply hs]
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set (Prod β Ω)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ρ.fst fun x => (ρ.condKernel x) (setOf fun y =>  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma setLIntegral_condKernel_eq_measure_prod {s : Set β} (hs : MeasurableSet s) {t : Set Ω}
    (ht : MeasurableSet t) :
    ∫⁻ b in s, ρ.condKernel b t ∂ρ.fst = ρ (s ×ˢ t) := by
  have : ρ (s ×ˢ t) = (ρ.fst ⊗ₘ ρ.condKernel) (s ×ˢ t) := by
    congr; exact (ρ.disintegrate _).symm
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    this : Eq (ρ (SProd.sprod s t)) ((ρ.fst.compProd ρ.condKernel) (SProd.sprod s  …
    ⊢ Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun b => (ρ.condKernel b) t)  …
  -/
  rw [this, compProd_apply (hs.prod ht)]
  classical
  have : ∀ b, ρ.condKernel b (Prod.mk b ⁻¹' s ×ˢ t)
      = s.indicator (fun b ↦ ρ.condKernel b t) b := by
    intro b
    by_cases hb : b ∈ s <;> simp [hb]
  simp_rw [this]
  rw [lintegral_indicator hs]


lemma lintegral_condKernel (hf : Measurable f) :
    ∫⁻ b, ∫⁻ ω, f (b, ω) ∂(ρ.condKernel b) ∂ρ.fst = ∫⁻ x, f x ∂ρ := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral ρ.fst fun b => MeasureTheory.lintegral (ρ.condKe …
  -/
  conv_rhs => rw [← ρ.disintegrate ρ.condKernel]
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral ρ.fst fun b => MeasureTheory.lintegral (ρ.condKe …
  -/
  rw [lintegral_compProd hf]
  /-
    🎉 no goals
  -/


lemma setLIntegral_condKernel (hf : Measurable f) {s : Set β}
    (hs : MeasurableSet s) {t : Set Ω} (ht : MeasurableSet t) :
    ∫⁻ b in s, ∫⁻ ω in t, f (b, ω) ∂(ρ.condKernel b) ∂ρ.fst
      = ∫⁻ x in s ×ˢ t, f x ∂ρ := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun b => MeasureTheory.linteg …
  -/
  conv_rhs => rw [← ρ.disintegrate ρ.condKernel]
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun b => MeasureTheory.linteg …
  -/
  rw [setLIntegral_compProd hf hs ht]
  /-
    🎉 no goals
  -/


lemma setLIntegral_condKernel_univ_right (hf : Measurable f) {s : Set β}
    (hs : MeasurableSet s) :
    ∫⁻ b in s, ∫⁻ ω, f (b, ω) ∂(ρ.condKernel b) ∂ρ.fst
      = ∫⁻ x in s ×ˢ Set.univ, f x ∂ρ := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun b => MeasureTheory.linteg …
  -/
  rw [← setLIntegral_condKernel hf hs MeasurableSet.univ]; simp_rw [Measure.restrict_univ]
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma setLIntegral_condKernel_univ_left (hf : Measurable f) {t : Set Ω}
    (ht : MeasurableSet t) :
    ∫⁻ b, ∫⁻ ω in t, f (b, ω) ∂(ρ.condKernel b) ∂ρ.fst
      = ∫⁻ x in Set.univ ×ˢ t, f x ∂ρ := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod β Ω → ENNReal
    hf : Measurable f
    t : Set Ω
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral ρ.fst fun b => MeasureTheory.lintegral ((ρ.condK …
  -/
  rw [← setLIntegral_condKernel hf MeasurableSet.univ ht]; simp_rw [Measure.restrict_univ]
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma _root_.MeasureTheory.AEStronglyMeasurable.integral_condKernel
    (hf : AEStronglyMeasurable f ρ) :
    AEStronglyMeasurable (fun x ↦ ∫ y, f (x, y) ∂ρ.condKernel x) ρ.fst := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hf : MeasureTheory.AEStronglyMeasurable f ρ
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral (ρ.condK …
  -/
  rw [← ρ.disintegrate ρ.condKernel] at hf
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hf : MeasureTheory.AEStronglyMeasurable f (ρ.fst.compProd ρ.condKernel)
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => MeasureTheory.integral (ρ.condK …
  -/
  exact AEStronglyMeasurable.integral_kernel_compProd hf
  /-
    🎉 no goals
  -/


lemma integral_condKernel (hf : Integrable f ρ) :
    ∫ b, ∫ ω, f (b, ω) ∂(ρ.condKernel b) ∂ρ.fst = ∫ x, f x ∂ρ := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hf : MeasureTheory.Integrable f ρ
    ⊢ Eq (MeasureTheory.integral ρ.fst fun b => MeasureTheory.integral (ρ.condKern …
  -/
  conv_rhs => rw [← ρ.disintegrate ρ.condKernel]
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hf : MeasureTheory.Integrable f ρ
    ⊢ Eq (MeasureTheory.integral ρ.fst fun b => MeasureTheory.integral (ρ.condKern …
  -/
  rw [← ρ.disintegrate ρ.condKernel] at hf
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hf : MeasureTheory.Integrable f (ρ.fst.compProd ρ.condKernel)
    ⊢ Eq (MeasureTheory.integral ρ.fst fun b => MeasureTheory.integral (ρ.condKern …
  -/
  rw [integral_compProd hf]
  /-
    🎉 no goals
  -/


lemma setIntegral_condKernel {s : Set β} (hs : MeasurableSet s)
    {t : Set Ω} (ht : MeasurableSet t) (hf : IntegrableOn f (s ×ˢ t) ρ) :
    ∫ b in s, ∫ ω in t, f (b, ω) ∂(ρ.condKernel b) ∂ρ.fst = ∫ x in s ×ˢ t, f x ∂ρ := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) ρ
    ⊢ Eq (MeasureTheory.integral (ρ.fst.restrict s) fun b => MeasureTheory.integra …
  -/
  conv_rhs => rw [← ρ.disintegrate ρ.condKernel]
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) ρ
    ⊢ Eq (MeasureTheory.integral (ρ.fst.restrict s) fun b => MeasureTheory.integra …
  -/
  rw [← ρ.disintegrate ρ.condKernel] at hf
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set β
    hs : MeasurableSet s
    t : Set Ω
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) (ρ.fst.compProd ρ.condKern …
    ⊢ Eq (MeasureTheory.integral (ρ.fst.restrict s) fun b => MeasureTheory.integra …
  -/
  rw [setIntegral_compProd hs ht hf]
  /-
    🎉 no goals
  -/


lemma setIntegral_condKernel_univ_right {s : Set β} (hs : MeasurableSet s)
    (hf : IntegrableOn f (s ×ˢ Set.univ) ρ) :
    ∫ b in s, ∫ ω, f (b, ω) ∂(ρ.condKernel b) ∂ρ.fst = ∫ x in s ×ˢ Set.univ, f x ∂ρ := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set β
    hs : MeasurableSet s
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s Set.univ) ρ
    ⊢ Eq (MeasureTheory.integral (ρ.fst.restrict s) fun b => MeasureTheory.integra …
  -/
  rw [← setIntegral_condKernel hs MeasurableSet.univ hf]; simp_rw [Measure.restrict_univ]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma setIntegral_condKernel_univ_left {t : Set Ω} (ht : MeasurableSet t)
    (hf : IntegrableOn f (Set.univ ×ˢ t) ρ) :
    ∫ b, ∫ ω in t, f (b, ω) ∂(ρ.condKernel b) ∂ρ.fst = ∫ x in Set.univ ×ˢ t, f x ∂ρ := by
  /-
    β : Type u_1
    Ω : Type u_2
    mβ : MeasurableSpace β
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod β Ω)
    inst✝² : MeasureTheory.IsFiniteMeasure ρ
    E : Type u_3
    f : Prod β Ω → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Set Ω
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod Set.univ t) ρ
    ⊢ Eq (MeasureTheory.integral ρ.fst fun b => MeasureTheory.integral ((ρ.condKer …
  -/
  rw [← setIntegral_condKernel MeasurableSet.univ ht hf]; simp_rw [Measure.restrict_univ]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem AEStronglyMeasurable.ae_integrable_condKernel_iff {f : α × Ω → F}
    (hf : AEStronglyMeasurable f ρ) :
    (∀ᵐ a ∂ρ.fst, Integrable (fun ω ↦ f (a, ω)) (ρ.condKernel a)) ∧
      Integrable (fun a ↦ ∫ ω, ‖f (a, ω)‖ ∂ρ.condKernel a) ρ.fst ↔ Integrable f ρ := by
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf : MeasureTheory.AEStronglyMeasurable f ρ
    ⊢ Iff (And (Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f { …
  -/
  rw [← ρ.disintegrate ρ.condKernel] at hf
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf : MeasureTheory.AEStronglyMeasurable f (ρ.fst.compProd ρ.condKernel)
    ⊢ Iff (And (Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f { …
  -/
  conv_rhs => rw [← ρ.disintegrate ρ.condKernel]
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf : MeasureTheory.AEStronglyMeasurable f (ρ.fst.compProd ρ.condKernel)
    ⊢ Iff (And (Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f { …
  -/
  rw [Measure.integrable_compProd_iff hf]
  /-
    🎉 no goals
  -/


theorem Integrable.condKernel_ae {f : α × Ω → F} (hf_int : Integrable f ρ) :
    ∀ᵐ a ∂ρ.fst, Integrable (fun ω ↦ f (a, ω)) (ρ.condKernel a) := by
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf_int : MeasureTheory.Integrable f ρ
    ⊢ Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f { fst := a, …
  -/
  have hf_ae : AEStronglyMeasurable f ρ := hf_int.1
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf_int : MeasureTheory.Integrable f ρ
    hf_ae : MeasureTheory.AEStronglyMeasurable f ρ
    ⊢ Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f { fst := a, …
  -/
  rw [← hf_ae.ae_integrable_condKernel_iff] at hf_int
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf_int : And (Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f …
    hf_ae : MeasureTheory.AEStronglyMeasurable f ρ
    ⊢ Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f { fst := a, …
  -/
  exact hf_int.1
  /-
    🎉 no goals
  -/


theorem Integrable.integral_norm_condKernel {f : α × Ω → F} (hf_int : Integrable f ρ) :
    Integrable (fun x ↦ ∫ y, ‖f (x, y)‖ ∂ρ.condKernel x) ρ.fst := by
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf_int : MeasureTheory.Integrable f ρ
    ⊢ MeasureTheory.Integrable (fun x => MeasureTheory.integral (ρ.condKernel x) f …
  -/
  have hf_ae : AEStronglyMeasurable f ρ := hf_int.1
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf_int : MeasureTheory.Integrable f ρ
    hf_ae : MeasureTheory.AEStronglyMeasurable f ρ
    ⊢ MeasureTheory.Integrable (fun x => MeasureTheory.integral (ρ.condKernel x) f …
  -/
  rw [← hf_ae.ae_integrable_condKernel_iff] at hf_int
  /-
    α : Type u_1
    Ω : Type u_2
    F : Type u_4
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    inst✝¹ : NormedAddCommGroup F
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → F
    hf_int : And (Filter.Eventually (fun a => MeasureTheory.Integrable (fun ω => f …
    hf_ae : MeasureTheory.AEStronglyMeasurable f ρ
    ⊢ MeasureTheory.Integrable (fun x => MeasureTheory.integral (ρ.condKernel x) f …
  -/
  exact hf_int.2
  /-
    🎉 no goals
  -/


theorem Integrable.norm_integral_condKernel {f : α × Ω → E} (hf_int : Integrable f ρ) :
    Integrable (fun x ↦ ‖∫ y, f (x, y) ∂ρ.condKernel x‖) ρ.fst := by
  /-
    α : Type u_1
    Ω : Type u_2
    E : Type u_3
    mα : MeasurableSpace α
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → E
    hf_int : MeasureTheory.Integrable f ρ
    ⊢ MeasureTheory.Integrable (fun x => Norm.norm (MeasureTheory.integral (ρ.cond …
  -/
  refine hf_int.integral_norm_condKernel.mono hf_int.1.integral_condKernel.norm ?_
  /-
    α : Type u_1
    Ω : Type u_2
    E : Type u_3
    mα : MeasurableSpace α
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → E
    hf_int : MeasureTheory.Integrable f ρ
    ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (Norm.norm (MeasureTheory.integ …
  -/
  refine Filter.Eventually.of_forall fun x ↦ ?_
  /-
    α : Type u_1
    Ω : Type u_2
    E : Type u_3
    mα : MeasurableSpace α
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → E
    hf_int : MeasureTheory.Integrable f ρ
    x : α
    ⊢ LE.le (Norm.norm (Norm.norm (MeasureTheory.integral (ρ.condKernel x) fun y = …
  -/
  rw [norm_norm]
  /-
    α : Type u_1
    Ω : Type u_2
    E : Type u_3
    mα : MeasurableSpace α
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → E
    hf_int : MeasureTheory.Integrable f ρ
    x : α
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (ρ.condKernel x) fun y => f { fst : …
  -/
  refine (norm_integral_le_integral_norm _).trans_eq (Real.norm_of_nonneg ?_).symm
  /-
    α : Type u_1
    Ω : Type u_2
    E : Type u_3
    mα : MeasurableSpace α
    inst✝⁵ : MeasurableSpace Ω
    inst✝⁴ : StandardBorelSpace Ω
    inst✝³ : Nonempty Ω
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    f : Prod α Ω → E
    hf_int : MeasureTheory.Integrable f ρ
    x : α
    ⊢ LE.le 0 (MeasureTheory.integral (ρ.condKernel x) fun a => Norm.norm (f { fst …
  -/
  exact integral_nonneg_of_ae (Filter.Eventually.of_forall fun y ↦ norm_nonneg _)
  /-
    🎉 no goals
  -/


theorem Integrable.integral_condKernel {f : α × Ω → E} (hf_int : Integrable f ρ) :
    Integrable (fun x ↦ ∫ y, f (x, y) ∂ρ.condKernel x) ρ.fst :=
  (integrable_norm_iff hf_int.1.integral_condKernel).mp hf_int.norm_integral_condKernel


