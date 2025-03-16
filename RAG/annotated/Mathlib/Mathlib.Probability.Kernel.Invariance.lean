@[simp]
theorem bind_add (μ ν : Measure α) (κ : Kernel α β) : (μ + ν).bind κ = μ.bind κ + ν.bind κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq ((HAdd.hAdd μ ν).bind ⇑κ) (HAdd.hAdd (μ.bind ⇑κ) (ν.bind ⇑κ))
  -/
  ext1 s hs
  rw [Measure.bind_apply hs (Kernel.measurable _), lintegral_add_measure, Measure.coe_add,
    Pi.add_apply, Measure.bind_apply hs (Kernel.measurable _),
    Measure.bind_apply hs (Kernel.measurable _)]


@[simp]
theorem bind_smul (κ : Kernel α β) (μ : Measure α) (r : ℝ≥0∞) : (r • μ).bind κ = r • μ.bind κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    μ : MeasureTheory.Measure α
    r : ENNReal
    ⊢ Eq ((HSMul.hSMul r μ).bind ⇑κ) (HSMul.hSMul r (μ.bind ⇑κ))
  -/
  ext1 s hs
  rw [Measure.bind_apply hs (Kernel.measurable _), lintegral_smul_measure, Measure.coe_smul,
    Pi.smul_apply, Measure.bind_apply hs (Kernel.measurable _), smul_eq_mul]


theorem const_bind_eq_comp_const (κ : Kernel α β) (μ : Measure α) :
    const α (μ.bind κ) = κ ∘ₖ const α μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    μ : MeasureTheory.Measure α
    ⊢ Eq (ProbabilityTheory.Kernel.const α (μ.bind ⇑κ)) (κ.comp (ProbabilityTheory …
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    μ : MeasureTheory.Measure α
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.const α (μ.bind ⇑κ)) a) s) (((κ.comp (Probabi …
  -/
  simp_rw [comp_apply' _ _ _ hs, const_apply, Measure.bind_apply hs (Kernel.measurable _)]
  /-
    🎉 no goals
  -/


theorem comp_const_apply_eq_bind (κ : Kernel α β) (μ : Measure α) (a : α) :
    (κ ∘ₖ const α μ) a = μ.bind κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    μ : MeasureTheory.Measure α
    a : α
    ⊢ Eq ((κ.comp (ProbabilityTheory.Kernel.const α μ)) a) (μ.bind ⇑κ)
  -/
  rw [← const_apply (μ.bind κ) a, const_bind_eq_comp_const κ μ]
  /-
    🎉 no goals
  -/


/-- A measure `μ` is invariant with respect to the kernel `κ` if the push-forward measure of `μ`
along `κ` equals `μ`. -/
def Invariant (κ : Kernel α α) (μ : Measure α) : Prop :=
  μ.bind κ = μ


theorem Invariant.def (hκ : Invariant κ μ) : μ.bind κ = μ :=
  hκ


theorem Invariant.comp_const (hκ : Invariant κ μ) : κ ∘ₖ const α μ = const α μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    κ : ProbabilityTheory.Kernel α α
    μ : MeasureTheory.Measure α
    hκ : κ.Invariant μ
    ⊢ Eq (κ.comp (ProbabilityTheory.Kernel.const α μ)) (ProbabilityTheory.Kernel.c …
  -/
  rw [← const_bind_eq_comp_const κ μ, hκ.def]
  /-
    🎉 no goals
  -/


theorem Invariant.comp [IsSFiniteKernel κ] (hκ : Invariant κ μ) (hη : Invariant η μ) :
    Invariant (κ ∘ₖ η) μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    κ η : ProbabilityTheory.Kernel α α
    μ : MeasureTheory.Measure α
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : κ.Invariant μ
    hη : η.Invariant μ
    ⊢ (κ.comp η).Invariant μ
  -/
  cases' isEmpty_or_nonempty α with _ hα
    /-
      case inl
      α : Type u_1
      mα : MeasurableSpace α
      κ η : ProbabilityTheory.Kernel α α
      μ : MeasureTheory.Measure α
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      hκ : κ.Invariant μ
      hη : η.Invariant μ
      h✝ : IsEmpty α
      ⊢ (κ.comp η).Invariant μ
    -/
  · exact Subsingleton.elim _ _
    /-
      🎉 no goals
    -/
  · simp_rw [Invariant, ← comp_const_apply_eq_bind (κ ∘ₖ η) μ hα.some, comp_assoc, hη.comp_const,
      hκ.comp_const, const_apply]


