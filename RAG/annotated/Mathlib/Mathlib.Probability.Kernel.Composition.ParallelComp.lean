/-- Parallel product of two kernels. -/
noncomputable
def parallelComp (κ : Kernel α β) (η : Kernel γ δ) : Kernel (α × γ) (β × δ) :=
  (prodMkRight γ κ) ×ₖ (prodMkLeft α η)


@[inherit_doc]
scoped[ProbabilityTheory] infixl:100 " ∥ₖ " => ProbabilityTheory.Kernel.parallelComp


lemma parallelComp_apply (κ : Kernel α β) [IsSFiniteKernel κ]
    (η : Kernel γ δ) [IsSFiniteKernel η] (x : α × γ) :
    (κ ∥ₖ η) x = (κ x.1).prod (η x.2) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    x : Prod α γ
    ⊢ Eq ((κ.parallelComp η) x) ((κ x.1).prod (η x.2))
  -/
  rw [parallelComp, prod_apply, prodMkRight_apply, prodMkLeft_apply]
  /-
    🎉 no goals
  -/


lemma lintegral_parallelComp (κ : Kernel α β) [IsSFiniteKernel κ]
    (η : Kernel γ δ) [IsSFiniteKernel η]
    (ac : α × γ) {g : β × δ → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ bd, g bd ∂(κ ∥ₖ η) ac = ∫⁻ b, ∫⁻ d, g (b, d) ∂η ac.2 ∂κ ac.1 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ac : Prod α γ
    g : Prod β δ → ENNReal
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral ((κ.parallelComp η) ac) fun bd => g bd) (Measure …
  -/
  rw [parallelComp, lintegral_prod _ _ _ hg]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ac : Prod α γ
    g : Prod β δ → ENNReal
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral ((ProbabilityTheory.Kernel.prodMkRight γ κ) ac)  …
  -/
  simp
  /-
    🎉 no goals
  -/


instance (κ : Kernel α β) (η : Kernel γ δ) : IsSFiniteKernel (κ ∥ₖ η) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    η : ProbabilityTheory.Kernel γ δ
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.parallelComp η)
  -/
  rw [parallelComp]; infer_instance
                     /-
                       🎉 no goals
                     -/


instance (κ : Kernel α β) [IsFiniteKernel κ] (η : Kernel γ δ) [IsFiniteKernel η] :
    IsFiniteKernel (κ ∥ₖ η) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    η : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    ⊢ ProbabilityTheory.IsFiniteKernel (κ.parallelComp η)
  -/
  rw [parallelComp]; infer_instance
                     /-
                       🎉 no goals
                     -/


instance (κ : Kernel α β) [IsMarkovKernel κ] (η : Kernel γ δ) [IsMarkovKernel η] :
    IsMarkovKernel (κ ∥ₖ η) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
    η : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsMarkovKernel η
    ⊢ ProbabilityTheory.IsMarkovKernel (κ.parallelComp η)
  -/
  rw [parallelComp]; infer_instance
                     /-
                       🎉 no goals
                     -/


instance (κ : Kernel α β) [IsZeroOrMarkovKernel κ] (η : Kernel γ δ) [IsZeroOrMarkovKernel η] :
    IsZeroOrMarkovKernel (κ ∥ₖ η) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsZeroOrMarkovKernel κ
    η : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsZeroOrMarkovKernel η
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (κ.parallelComp η)
  -/
  rw [parallelComp]; infer_instance
                     /-
                       🎉 no goals
                     -/


lemma parallelComp_comp_copy (κ : Kernel α β) [IsSFiniteKernel κ]
    (η : Kernel α γ) [IsSFiniteKernel η] :
    (κ ∥ₖ η) ∘ₖ (copy α) = κ ×ₖ η := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq ((κ.parallelComp η).comp (ProbabilityTheory.Kernel.copy α)) (κ.prod η)
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq ((((κ.parallelComp η).comp (ProbabilityTheory.Kernel.copy α)) a) s) (((κ. …
  -/
  simp_rw [prod_apply, comp_apply, copy_apply, Measure.bind_apply hs (Kernel.measurable _)]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.dirac { fst := a, snd :=  …
  -/
  rw [lintegral_dirac']
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (((κ.parallelComp η) { fst := a, snd := a }) s) (((κ a).prod (η a)) s)
  -/
  swap; · exact Kernel.measurable_coe _ hs
          /-
            🎉 no goals
          -/
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel α γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    ⊢ Eq (((κ.parallelComp η) { fst := a, snd := a }) s) (((κ a).prod (η a)) s)
  -/
  rw [parallelComp_apply]
  /-
    🎉 no goals
  -/


lemma swap_parallelComp {κ : Kernel α β} [IsSFiniteKernel κ]
    {η : Kernel γ δ} [IsSFiniteKernel η] :
    (swap β δ) ∘ₖ (κ ∥ₖ η) = (η ∥ₖ κ) ∘ₖ (swap α γ) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq ((ProbabilityTheory.Kernel.swap β δ).comp (κ.parallelComp η)) ((η.paralle …
  -/
  rw [parallelComp, swap_prod, parallelComp]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    mδ : MeasurableSpace δ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel γ δ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq ((ProbabilityTheory.Kernel.prodMkLeft α η).prod (ProbabilityTheory.Kernel …
  -/
  ext ac s hs
  rw [comp_apply, swap_apply, Measure.bind_apply hs (Kernel.measurable _),
    lintegral_dirac' _ (Kernel.measurable_coe _ hs), prod_apply, prod_apply, prodMkLeft_apply,
    prodMkLeft_apply, prodMkRight_apply, prodMkRight_apply, Prod.fst_swap, Prod.snd_swap]


/-- For a deterministic kernel, copying then applying the kernel to the two copies is the same
as first applying the kernel then copying. -/
lemma deterministic_comp_copy {f : α → β} (hf : Measurable f) :
    (Kernel.deterministic f hf ∥ₖ Kernel.deterministic f hf) ∘ₖ Kernel.copy α
      = Kernel.copy β ∘ₖ Kernel.deterministic f hf := by
  simp_rw [Kernel.parallelComp_comp_copy, Kernel.deterministic_prod_deterministic,
    Kernel.copy, Kernel.deterministic_comp_deterministic, Function.comp_def]


