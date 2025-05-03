/-- Kernel which to `a` associates the dirac measure at `f a`. This is a Markov kernel. -/
noncomputable def deterministic (f : α → β) (hf : Measurable f) : Kernel α β where
  toFun a := Measure.dirac (f a)
  measurable' := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      f : α → β
      hf : Measurable f
      ⊢ Measurable fun a => MeasureTheory.Measure.dirac (f a)
    -/
    refine Measure.measurable_of_measurable_coe _ fun s hs => ?_
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      f : α → β
      hf : Measurable f
      s : Set β
      hs : MeasurableSet s
      ⊢ Measurable fun b => (MeasureTheory.Measure.dirac (f b)) s
    -/
    simp_rw [Measure.dirac_apply' _ hs]
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      f : α → β
      hf : Measurable f
      s : Set β
      hs : MeasurableSet s
      ⊢ Measurable fun b => s.indicator 1 (f b)
    -/
    exact measurable_one.indicator (hf hs)
    /-
      🎉 no goals
    -/


theorem deterministic_apply {f : α → β} (hf : Measurable f) (a : α) :
    deterministic f hf a = Measure.dirac (f a) :=
  rfl


theorem deterministic_apply' {f : α → β} (hf : Measurable f) (a : α) {s : Set β}
    (hs : MeasurableSet s) : deterministic f hf a s = s.indicator (fun _ => 1) (f a) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.deterministic f hf) a) s) (s.indicator (fun x …
  -/
  rw [deterministic]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (({ toFun := fun a => MeasureTheory.Measure.dirac (f a), measurable' := ⋯ …
  -/
  change Measure.dirac (f a) s = s.indicator 1 (f a)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : Measurable f
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.dirac (f a)) s) (s.indicator 1 (f a))
  -/
  simp_rw [Measure.dirac_apply' _ hs]
  /-
    🎉 no goals
  -/


instance isMarkovKernel_deterministic {f : α → β} (hf : Measurable f) :
    IsMarkovKernel (deterministic f hf) :=
               /-
                 α : Type u_1
                 β : Type u_2
                 ι : Type u_3
                 mα : MeasurableSpace α
                 mβ : MeasurableSpace β
                 κ : ProbabilityTheory.Kernel α β
                 f : α → β
                 hf : Measurable f
                 a : α
                 ⊢ MeasureTheory.IsProbabilityMeasure ((ProbabilityTheory.Kernel.deterministic  …
               -/
  ⟨fun a => by rw [deterministic_apply hf]; infer_instance⟩
                                            /-
                                              🎉 no goals
                                            -/


theorem lintegral_deterministic' {f : β → ℝ≥0∞} {g : α → β} {a : α} (hg : Measurable g)
    (hf : Measurable f) : ∫⁻ x, f x ∂deterministic g hg a = f (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    a : α
    hg : Measurable g
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral ((ProbabilityTheory.Kernel.deterministic g hg) a …
  -/
  rw [deterministic_apply, lintegral_dirac' _ hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem lintegral_deterministic {f : β → ℝ≥0∞} {g : α → β} {a : α} (hg : Measurable g)
    [MeasurableSingletonClass β] : ∫⁻ x, f x ∂deterministic g hg a = f (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    a : α
    hg : Measurable g
    inst✝ : MeasurableSingletonClass β
    ⊢ Eq (MeasureTheory.lintegral ((ProbabilityTheory.Kernel.deterministic g hg) a …
  -/
  rw [deterministic_apply, lintegral_dirac (g a) f]
  /-
    🎉 no goals
  -/


theorem setLIntegral_deterministic' {f : β → ℝ≥0∞} {g : α → β} {a : α} (hg : Measurable g)
    (hf : Measurable f) {s : Set β} (hs : MeasurableSet s) [Decidable (g a ∈ s)] :
    ∫⁻ x in s, f x ∂deterministic g hg a = if g a ∈ s then f (g a) else 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    a : α
    hg : Measurable g
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    inst✝ : Decidable (Membership.mem s (g a))
    ⊢ Eq (MeasureTheory.lintegral (((ProbabilityTheory.Kernel.deterministic g hg)  …
  -/
  rw [deterministic_apply, setLIntegral_dirac' hf hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_deterministic' := setLIntegral_deterministic'


@[simp]
theorem setLIntegral_deterministic {f : β → ℝ≥0∞} {g : α → β} {a : α} (hg : Measurable g)
    [MeasurableSingletonClass β] (s : Set β) [Decidable (g a ∈ s)] :
    ∫⁻ x in s, f x ∂deterministic g hg a = if g a ∈ s then f (g a) else 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    a : α
    hg : Measurable g
    inst✝¹ : MeasurableSingletonClass β
    s : Set β
    inst✝ : Decidable (Membership.mem s (g a))
    ⊢ Eq (MeasureTheory.lintegral (((ProbabilityTheory.Kernel.deterministic g hg)  …
  -/
  rw [deterministic_apply, setLIntegral_dirac f s]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_deterministic := setLIntegral_deterministic


/-- The identity kernel, that maps `x : α` to the Dirac measure at `x`. -/
protected noncomputable
def id : Kernel α α := Kernel.deterministic id measurable_id


                                                         /-
                                                           α : Type u_1
                                                           β : Type u_2
                                                           ι : Type u_3
                                                           mα : MeasurableSpace α
                                                           mβ : MeasurableSpace β
                                                           κ : ProbabilityTheory.Kernel α β
                                                           ⊢ ProbabilityTheory.IsMarkovKernel ProbabilityTheory.Kernel.id
                                                         -/
instance : IsMarkovKernel (Kernel.id : Kernel α α) := by rw [Kernel.id]; infer_instance
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma id_apply (a : α) : Kernel.id a = Measure.dirac a := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    a : α
    ⊢ Eq (ProbabilityTheory.Kernel.id a) (MeasureTheory.Measure.dirac a)
  -/
  rw [Kernel.id, deterministic_apply, id_def]
  /-
    🎉 no goals
  -/


/-- The deterministic kernel that maps `x : α` to the Dirac measure at `(x, x) : α × α`. -/
noncomputable
def copy (α : Type*) [MeasurableSpace α] : Kernel α (α × α) :=
  Kernel.deterministic (fun x ↦ (x, x)) (measurable_id.prod measurable_id)


                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           ι : Type u_3
                                           mα : MeasurableSpace α
                                           mβ : MeasurableSpace β
                                           κ : ProbabilityTheory.Kernel α β
                                           ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.copy α)
                                         -/
instance : IsMarkovKernel (copy α) := by rw [copy]; infer_instance
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                                 /-
                                                                   α : Type u_1
                                                                   mα : MeasurableSpace α
                                                                   a : α
                                                                   ⊢ Eq ((ProbabilityTheory.Kernel.copy α) a) (MeasureTheory.Measure.dirac { fst  …
                                                                 -/
lemma copy_apply (a : α) : copy α a = Measure.dirac (a, a) := by simp [copy, deterministic_apply]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- The Markov kernel to the `Unit` type. -/
noncomputable
def discard (α : Type*) [MeasurableSpace α] : Kernel α Unit :=
  Kernel.deterministic (fun _ ↦ ()) measurable_const


                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              ι : Type u_3
                                              mα : MeasurableSpace α
                                              mβ : MeasurableSpace β
                                              κ : ProbabilityTheory.Kernel α β
                                              ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.discard α)
                                            -/
instance : IsMarkovKernel (discard α) := by rw [discard]; infer_instance
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
lemma discard_apply (a : α) : discard α a = Measure.dirac () := deterministic_apply _ _


/-- The deterministic kernel that maps `(x, y)` to the Dirac measure at `(y, x)`. -/
noncomputable
def swap (α β : Type*) [MeasurableSpace α] [MeasurableSpace β] : Kernel (α × β) (β × α) :=
  Kernel.deterministic Prod.swap measurable_swap


                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             ι : Type u_3
                                             mα : MeasurableSpace α
                                             mβ : MeasurableSpace β
                                             κ : ProbabilityTheory.Kernel α β
                                             ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.swap α β)
                                           -/
instance : IsMarkovKernel (swap α β) := by rw [swap]; infer_instance
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- See `swap_apply'` for a fully applied version of this lemma. -/
lemma swap_apply (ab : α × β) : swap α β ab = Measure.dirac ab.swap := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    ab : Prod α β
    ⊢ Eq ((ProbabilityTheory.Kernel.swap α β) ab) (MeasureTheory.Measure.dirac ab. …
  -/
  rw [swap, deterministic_apply]
  /-
    🎉 no goals
  -/


/-- See `swap_apply` for a partially applied version of this lemma. -/
lemma swap_apply' (ab : α × β) {s : Set (β × α)} (hs : MeasurableSet s) :
    swap α β ab s = s.indicator 1 ab.swap := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    ab : Prod α β
    s : Set (Prod β α)
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.swap α β) ab) s) (s.indicator 1 ab.swap)
  -/
  rw [swap_apply, Measure.dirac_apply' _ hs]
  /-
    🎉 no goals
  -/


/-- Constant kernel, which always returns the same measure. -/
def const (α : Type*) {β : Type*} [MeasurableSpace α] {_ : MeasurableSpace β} (μβ : Measure β) :
    Kernel α β where
  toFun _ := μβ
  measurable' := measurable_const


@[simp]
theorem const_apply (μβ : Measure β) (a : α) : const α μβ a = μβ :=
  rfl


@[simp]
lemma const_zero : const α (0 : Measure β) = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    ⊢ Eq (ProbabilityTheory.Kernel.const α 0) 0
  -/
  ext x s _; simp [const_apply]
             /-
               🎉 no goals
             -/


lemma const_add (β : Type*) [MeasurableSpace β] (μ ν : Measure α) :
                                                  /-
                                                    α : Type u_1
                                                    mα : MeasurableSpace α
                                                    β : Type u_4
                                                    inst✝ : MeasurableSpace β
                                                    μ ν : MeasureTheory.Measure α
                                                    ⊢ Eq (ProbabilityTheory.Kernel.const β (HAdd.hAdd μ ν)) (HAdd.hAdd (Probabilit …
                                                  -/
    const β (μ + ν) = const β μ + const β ν := by ext; simp
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma sum_const [Countable ι] (μ : ι → Measure β) :
    Kernel.sum (fun n ↦ const α (μ n)) = const α (Measure.sum μ) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure β
    ⊢ Eq (ProbabilityTheory.Kernel.sum fun n => ProbabilityTheory.Kernel.const α ( …
  -/
  ext x s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure β
    x : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.sum fun n => ProbabilityTheory.Kernel.const α …
  -/
  rw [const_apply, Measure.sum_apply _ hs, Kernel.sum_apply' _ _ hs]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    μ : ι → MeasureTheory.Measure β
    x : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (tsum fun n => ((ProbabilityTheory.Kernel.const α (μ n)) x) s) (tsum fun  …
  -/
  simp only [const_apply]
  /-
    🎉 no goals
  -/


instance const.instIsFiniteKernel {μβ : Measure β} [IsFiniteMeasure μβ] :
    IsFiniteKernel (const α μβ) :=
  ⟨⟨μβ Set.univ, measure_lt_top _ _, fun _ => le_rfl⟩⟩


instance const.instIsSFiniteKernel {μβ : Measure β} [SFinite μβ] :
    IsSFiniteKernel (const α μβ) :=
                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  ι : Type u_3
                                                                  mα : MeasurableSpace α
                                                                  mβ : MeasurableSpace β
                                                                  κ : ProbabilityTheory.Kernel α β
                                                                  μβ : MeasureTheory.Measure β
                                                                  inst✝ : MeasureTheory.SFinite μβ
                                                                  ⊢ Eq (ProbabilityTheory.Kernel.const α μβ) (ProbabilityTheory.Kernel.sum fun n …
                                                                -/
  ⟨fun n ↦ const α (sfiniteSeq μβ n), fun n ↦ inferInstance, by rw [sum_const, sum_sfiniteSeq]⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


instance const.instIsMarkovKernel {μβ : Measure β} [hμβ : IsProbabilityMeasure μβ] :
    IsMarkovKernel (const α μβ) :=
  ⟨fun _ => hμβ⟩


instance const.instIsZeroOrMarkovKernel {μβ : Measure β} [hμβ : IsZeroOrProbabilityMeasure μβ] :
    IsZeroOrMarkovKernel (const α μβ) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    μβ : MeasureTheory.Measure β
    hμβ : MeasureTheory.IsZeroOrProbabilityMeasure μβ
    ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (ProbabilityTheory.Kernel.const α μβ)
  -/
  rcases eq_zero_or_isProbabilityMeasure μβ with rfl | h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      hμβ : MeasureTheory.IsZeroOrProbabilityMeasure 0
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (ProbabilityTheory.Kernel.const α 0)
    -/
  · simp only [const_zero]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      hμβ : MeasureTheory.IsZeroOrProbabilityMeasure 0
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      μβ : MeasureTheory.Measure β
      hμβ : MeasureTheory.IsZeroOrProbabilityMeasure μβ
      h : MeasureTheory.IsProbabilityMeasure μβ
      ⊢ ProbabilityTheory.IsZeroOrMarkovKernel (ProbabilityTheory.Kernel.const α μβ)
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


lemma isSFiniteKernel_const [Nonempty α] {μβ : Measure β} :
    IsSFiniteKernel (const α μβ) ↔ SFinite μβ :=
  ⟨fun h ↦ h.sFinite (Classical.arbitrary α), fun _ ↦ inferInstance⟩


@[simp]
theorem lintegral_const {f : β → ℝ≥0∞} {μ : Measure β} {a : α} :
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  mα : MeasurableSpace α
                                                  mβ : MeasurableSpace β
                                                  f : β → ENNReal
                                                  μ : MeasureTheory.Measure β
                                                  a : α
                                                  ⊢ Eq (MeasureTheory.lintegral ((ProbabilityTheory.Kernel.const α μ) a) fun x = …
                                                -/
    ∫⁻ x, f x ∂const α μ a = ∫⁻ x, f x ∂μ := by rw [const_apply]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem setLIntegral_const {f : β → ℝ≥0∞} {μ : Measure β} {a : α} {s : Set β} :
                                                          /-
                                                            α : Type u_1
                                                            β : Type u_2
                                                            mα : MeasurableSpace α
                                                            mβ : MeasurableSpace β
                                                            f : β → ENNReal
                                                            μ : MeasureTheory.Measure β
                                                            a : α
                                                            s : Set β
                                                            ⊢ Eq (MeasureTheory.lintegral (((ProbabilityTheory.Kernel.const α μ) a).restri …
                                                          -/
    ∫⁻ x in s, f x ∂const α μ a = ∫⁻ x in s, f x ∂μ := by rw [const_apply]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_const := setLIntegral_const


/-- In a countable space with measurable singletons, every function `α → MeasureTheory.Measure β`
defines a kernel. -/
def ofFunOfCountable [MeasurableSpace α] {_ : MeasurableSpace β} [Countable α]
    [MeasurableSingletonClass α] (f : α → Measure β) : Kernel α β where
  toFun := f
  measurable' := measurable_of_countable f


/-- Kernel given by the restriction of the measures in the image of a kernel to a set. -/
protected noncomputable def restrict (κ : Kernel α β) (hs : MeasurableSet s) : Kernel α β where
  toFun a := (κ a).restrict s
  measurable' := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ : ProbabilityTheory.Kernel α β
      s t : Set β
      κ : ProbabilityTheory.Kernel α β
      hs : MeasurableSet s
      ⊢ Measurable fun a => (κ a).restrict s
    -/
    refine Measure.measurable_of_measurable_coe _ fun t ht => ?_
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ : ProbabilityTheory.Kernel α β
      s t✝ : Set β
      κ : ProbabilityTheory.Kernel α β
      hs : MeasurableSet s
      t : Set β
      ht : MeasurableSet t
      ⊢ Measurable fun b => ((κ b).restrict s) t
    -/
    simp_rw [Measure.restrict_apply ht]
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ : ProbabilityTheory.Kernel α β
      s t✝ : Set β
      κ : ProbabilityTheory.Kernel α β
      hs : MeasurableSet s
      t : Set β
      ht : MeasurableSet t
      ⊢ Measurable fun b => (κ b) (Inter.inter t s)
    -/
    exact Kernel.measurable_coe κ (ht.inter hs)
    /-
      🎉 no goals
    -/


theorem restrict_apply (κ : Kernel α β) (hs : MeasurableSet s) (a : α) :
    κ.restrict hs a = (κ a).restrict s :=
  rfl


theorem restrict_apply' (κ : Kernel α β) (hs : MeasurableSet s) (a : α) (ht : MeasurableSet t) :
    κ.restrict hs a t = (κ a) (t ∩ s) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    s t : Set β
    κ : ProbabilityTheory.Kernel α β
    hs : MeasurableSet s
    a : α
    ht : MeasurableSet t
    ⊢ Eq (((κ.restrict hs) a) t) ((κ a) (Inter.inter t s))
  -/
  rw [restrict_apply κ hs a, Measure.restrict_apply ht]
  /-
    🎉 no goals
  -/


@[simp]
theorem restrict_univ : κ.restrict MeasurableSet.univ = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq (κ.restrict ⋯) κ
  -/
  ext1 a
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    a : α
    ⊢ Eq ((κ.restrict ⋯) a) (κ a)
  -/
  rw [Kernel.restrict_apply, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem lintegral_restrict (κ : Kernel α β) (hs : MeasurableSet s) (a : α) (f : β → ℝ≥0∞) :
                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             mα : MeasurableSpace α
                                                             mβ : MeasurableSpace β
                                                             s : Set β
                                                             κ : ProbabilityTheory.Kernel α β
                                                             hs : MeasurableSet s
                                                             a : α
                                                             f : β → ENNReal
                                                             ⊢ Eq (MeasureTheory.lintegral ((κ.restrict hs) a) fun b => f b) (MeasureTheory …
                                                           -/
    ∫⁻ b, f b ∂κ.restrict hs a = ∫⁻ b in s, f b ∂κ a := by rw [restrict_apply]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem setLIntegral_restrict (κ : Kernel α β) (hs : MeasurableSet s) (a : α) (f : β → ℝ≥0∞)
    (t : Set β) : ∫⁻ b in t, f b ∂κ.restrict hs a = ∫⁻ b in t ∩ s, f b ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set β
    κ : ProbabilityTheory.Kernel α β
    hs : MeasurableSet s
    a : α
    f : β → ENNReal
    t : Set β
    ⊢ Eq (MeasureTheory.lintegral (((κ.restrict hs) a).restrict t) fun b => f b) ( …
  -/
  rw [restrict_apply, Measure.restrict_restrict' hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_restrict := setLIntegral_restrict



instance IsFiniteKernel.restrict (κ : Kernel α β) [IsFiniteKernel κ] (hs : MeasurableSet s) :
    IsFiniteKernel (κ.restrict hs) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    s t : Set β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hs : MeasurableSet s
    ⊢ ProbabilityTheory.IsFiniteKernel (κ.restrict hs)
  -/
  refine ⟨⟨IsFiniteKernel.bound κ, IsFiniteKernel.bound_lt_top κ, fun a => ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    s t : Set β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hs : MeasurableSet s
    a : α
    ⊢ LE.le (((κ.restrict hs) a) Set.univ) (ProbabilityTheory.IsFiniteKernel.bound …
  -/
  rw [restrict_apply' κ hs a MeasurableSet.univ]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    s t : Set β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hs : MeasurableSet s
    a : α
    ⊢ LE.le ((κ a) (Inter.inter Set.univ s)) (ProbabilityTheory.IsFiniteKernel.bou …
  -/
  exact measure_le_bound κ a _
  /-
    🎉 no goals
  -/


instance IsSFiniteKernel.restrict (κ : Kernel α β) [IsSFiniteKernel κ] (hs : MeasurableSet s) :
    IsSFiniteKernel (κ.restrict hs) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    s t : Set β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hs : MeasurableSet s
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.restrict hs)
  -/
  refine ⟨⟨fun n => Kernel.restrict (seq κ n) hs, inferInstance, ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    s t : Set β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hs : MeasurableSet s
    ⊢ Eq (κ.restrict hs) (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).restrict …
  -/
  ext1 a
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    s t : Set β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hs : MeasurableSet s
    a : α
    ⊢ Eq ((κ.restrict hs) a) ((ProbabilityTheory.Kernel.sum fun n => (κ.seq n).res …
  -/
  simp_rw [sum_apply, restrict_apply, ← Measure.restrict_sum _ hs, ← sum_apply, kernel_sum_seq]
  /-
    🎉 no goals
  -/


/-- Kernel with value `(κ a).comap f`, for a measurable embedding `f`. That is, for a measurable set
`t : Set β`, `ProbabilityTheory.Kernel.comapRight κ hf a t = κ a (f '' t)`. -/
noncomputable def comapRight (κ : Kernel α β) (hf : MeasurableEmbedding f) : Kernel α γ where
  toFun a := (κ a).comap f
  measurable' := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ : ProbabilityTheory.Kernel α β
      γ : Type u_4
      mγ : MeasurableSpace γ
      f : γ → β
      κ : ProbabilityTheory.Kernel α β
      hf : MeasurableEmbedding f
      ⊢ Measurable fun a => MeasureTheory.Measure.comap f (κ a)
    -/
    refine Measure.measurable_measure.mpr fun t ht => ?_
    have : (fun a => Measure.comap f (κ a) t) = fun a => κ a (f '' t) := by
      ext1 a
      rw [Measure.comap_apply _ hf.injective _ _ ht]
      exact fun s' hs' ↦ hf.measurableSet_image.mpr hs'
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ : ProbabilityTheory.Kernel α β
      γ : Type u_4
      mγ : MeasurableSpace γ
      f : γ → β
      κ : ProbabilityTheory.Kernel α β
      hf : MeasurableEmbedding f
      t : Set γ
      ht : MeasurableSet t
      this : Eq (fun a => (MeasureTheory.Measure.comap f (κ a)) t) fun a => (κ a) (S …
      ⊢ Measurable fun b => (MeasureTheory.Measure.comap f (κ b)) t
    -/
    rw [this]
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ✝ : ProbabilityTheory.Kernel α β
      γ : Type u_4
      mγ : MeasurableSpace γ
      f : γ → β
      κ : ProbabilityTheory.Kernel α β
      hf : MeasurableEmbedding f
      t : Set γ
      ht : MeasurableSet t
      this : Eq (fun a => (MeasureTheory.Measure.comap f (κ a)) t) fun a => (κ a) (S …
      ⊢ Measurable fun a => (κ a) (Set.image f t)
    -/
    exact Kernel.measurable_coe _ (hf.measurableSet_image.mpr ht)
    /-
      🎉 no goals
    -/


theorem comapRight_apply (κ : Kernel α β) (hf : MeasurableEmbedding f) (a : α) :
    comapRight κ hf a = Measure.comap f (κ a) :=
  rfl


theorem comapRight_apply' (κ : Kernel α β) (hf : MeasurableEmbedding f) (a : α) {t : Set γ}
    (ht : MeasurableSet t) : comapRight κ hf a t = κ a (f '' t) := by
  rw [comapRight_apply,
    Measure.comap_apply _ hf.injective (fun s => hf.measurableSet_image.mpr) _ ht]


@[simp]
lemma comapRight_id (κ : Kernel α β) : comapRight κ MeasurableEmbedding.id = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq (κ.comapRight ⋯) κ
  -/
  ext _ _ hs; rw [comapRight_apply' _ _ _ hs]; simp
                                               /-
                                                 🎉 no goals
                                               -/


theorem IsMarkovKernel.comapRight (κ : Kernel α β) (hf : MeasurableEmbedding f)
    (hκ : ∀ a, κ a (Set.range f) = 1) : IsMarkovKernel (comapRight κ hf) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    hf : MeasurableEmbedding f
    hκ : ∀ (a : α), Eq ((κ a) (Set.range f)) 1
    ⊢ ProbabilityTheory.IsMarkovKernel (κ.comapRight hf)
  -/
  refine ⟨fun a => ⟨?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    hf : MeasurableEmbedding f
    hκ : ∀ (a : α), Eq ((κ a) (Set.range f)) 1
    a : α
    ⊢ Eq (((κ.comapRight hf) a) Set.univ) 1
  -/
  rw [comapRight_apply' κ hf a MeasurableSet.univ]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    hf : MeasurableEmbedding f
    hκ : ∀ (a : α), Eq ((κ a) (Set.range f)) 1
    a : α
    ⊢ Eq ((κ a) (Set.image f Set.univ)) 1
  -/
  simp only [Set.image_univ, Subtype.range_coe_subtype, Set.setOf_mem_eq]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    hf : MeasurableEmbedding f
    hκ : ∀ (a : α), Eq ((κ a) (Set.range f)) 1
    a : α
    ⊢ Eq ((κ a) (Set.range f)) 1
  -/
  exact hκ a
  /-
    🎉 no goals
  -/


instance IsFiniteKernel.comapRight (κ : Kernel α β) [IsFiniteKernel κ]
    (hf : MeasurableEmbedding f) : IsFiniteKernel (comapRight κ hf) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : MeasurableEmbedding f
    ⊢ ProbabilityTheory.IsFiniteKernel (κ.comapRight hf)
  -/
  refine ⟨⟨IsFiniteKernel.bound κ, IsFiniteKernel.bound_lt_top κ, fun a => ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : MeasurableEmbedding f
    a : α
    ⊢ LE.le (((κ.comapRight hf) a) Set.univ) (ProbabilityTheory.IsFiniteKernel.bou …
  -/
  rw [comapRight_apply' κ hf a .univ]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : MeasurableEmbedding f
    a : α
    ⊢ LE.le ((κ a) (Set.image f Set.univ)) (ProbabilityTheory.IsFiniteKernel.bound …
  -/
  exact measure_le_bound κ a _
  /-
    🎉 no goals
  -/


protected instance IsSFiniteKernel.comapRight (κ : Kernel α β) [IsSFiniteKernel κ]
    (hf : MeasurableEmbedding f) : IsSFiniteKernel (comapRight κ hf) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hf : MeasurableEmbedding f
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.comapRight hf)
  -/
  refine ⟨⟨fun n => comapRight (seq κ n) hf, inferInstance, ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hf : MeasurableEmbedding f
    ⊢ Eq (κ.comapRight hf) (ProbabilityTheory.Kernel.sum fun n => (κ.seq n).comapR …
  -/
  ext1 a
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hf : MeasurableEmbedding f
    a : α
    ⊢ Eq ((κ.comapRight hf) a) ((ProbabilityTheory.Kernel.sum fun n => (κ.seq n).c …
  -/
  rw [sum_apply]
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hf : MeasurableEmbedding f
    a : α
    ⊢ Eq ((κ.comapRight hf) a) (MeasureTheory.Measure.sum fun n => ((κ.seq n).coma …
  -/
  simp_rw [comapRight_apply _ hf]
  have :
    (Measure.sum fun n => Measure.comap f (seq κ n a)) =
      Measure.comap f (Measure.sum fun n => seq κ n a) := by
    ext1 t ht
    rw [Measure.comap_apply _ hf.injective (fun s' => hf.measurableSet_image.mpr) _ ht,
      Measure.sum_apply _ ht, Measure.sum_apply _ (hf.measurableSet_image.mpr ht)]
    congr with n : 1
    rw [Measure.comap_apply _ hf.injective (fun s' => hf.measurableSet_image.mpr) _ ht]
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ✝ : ProbabilityTheory.Kernel α β
    γ : Type u_4
    mγ : MeasurableSpace γ
    f : γ → β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hf : MeasurableEmbedding f
    a : α
    this : Eq (MeasureTheory.Measure.sum fun n => MeasureTheory.Measure.comap f (( …
    ⊢ Eq (MeasureTheory.Measure.comap f (κ a)) (MeasureTheory.Measure.sum fun n => …
  -/
  rw [this, measure_sum_seq]
  /-
    🎉 no goals
  -/


/-- `ProbabilityTheory.Kernel.piecewise hs κ η` is the kernel equal to `κ` on the measurable set `s`
and to `η` on its complement. -/
def piecewise (hs : MeasurableSet s) (κ η : Kernel α β) : Kernel α β where
  toFun a := if a ∈ s then κ a else η a
  measurable' := κ.measurable.piecewise hs η.measurable


theorem piecewise_apply (a : α) : piecewise hs κ η a = if a ∈ s then κ a else η a :=
  rfl


theorem piecewise_apply' (a : α) (t : Set β) :
    piecewise hs κ η a t = if a ∈ s then κ a t else η a t := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝ : DecidablePred fun x => Membership.mem s x
    a : α
    t : Set β
    ⊢ Eq (((ProbabilityTheory.Kernel.piecewise hs κ η) a) t) (ite (Membership.mem  …
  -/
                                      /-
                                        🎉 no goals
                                      -/
  rw [piecewise_apply]; split_ifs <;> rfl
                                      /-
                                        🎉 no goals
                                      -/


instance IsMarkovKernel.piecewise [IsMarkovKernel κ] [IsMarkovKernel η] :
    IsMarkovKernel (piecewise hs κ η) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
    inst✝ : ProbabilityTheory.IsMarkovKernel η
    ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.piecewise hs κ η)
  -/
  refine ⟨fun a => ⟨?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsMarkovKernel κ
    inst✝ : ProbabilityTheory.IsMarkovKernel η
    a : α
    ⊢ Eq (((ProbabilityTheory.Kernel.piecewise hs κ η) a) Set.univ) 1
  -/
  rw [piecewise_apply', measure_univ, measure_univ, ite_self]
  /-
    🎉 no goals
  -/


instance IsFiniteKernel.piecewise [IsFiniteKernel κ] [IsFiniteKernel η] :
    IsFiniteKernel (piecewise hs κ η) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    ⊢ ProbabilityTheory.IsFiniteKernel (ProbabilityTheory.Kernel.piecewise hs κ η)
  -/
  refine ⟨⟨max (IsFiniteKernel.bound κ) (IsFiniteKernel.bound η), ?_, fun a => ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ η : ProbabilityTheory.Kernel α β
      s : Set α
      hs : MeasurableSet s
      inst✝² : DecidablePred fun x => Membership.mem s x
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel η
      ⊢ LT.lt (Max.max (ProbabilityTheory.IsFiniteKernel.bound κ) (ProbabilityTheory …
    -/
  · exact max_lt (IsFiniteKernel.bound_lt_top κ) (IsFiniteKernel.bound_lt_top η)
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le (((ProbabilityTheory.Kernel.piecewise hs κ η) a) Set.univ) (Max.max (P …
  -/
  rw [piecewise_apply']
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel η
    a : α
    ⊢ LE.le (ite (Membership.mem s a) ((κ a) Set.univ) ((η a) Set.univ)) (Max.max  …
  -/
  exact (ite_le_sup _ _ _).trans (sup_le_sup (measure_le_bound _ _ _) (measure_le_bound _ _ _))
  /-
    🎉 no goals
  -/


protected instance IsSFiniteKernel.piecewise [IsSFiniteKernel κ] [IsSFiniteKernel η] :
    IsSFiniteKernel (piecewise hs κ η) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.piecewise hs κ η)
  -/
  refine ⟨⟨fun n => piecewise hs (seq κ n) (seq η n), inferInstance, ?_⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (ProbabilityTheory.Kernel.piecewise hs κ η) (ProbabilityTheory.Kernel.sum …
  -/
  ext1 a
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    ⊢ Eq ((ProbabilityTheory.Kernel.piecewise hs κ η) a) ((ProbabilityTheory.Kerne …
  -/
  simp_rw [sum_apply, Kernel.piecewise_apply]
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝² : DecidablePred fun x => Membership.mem s x
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    ⊢ Eq (ite (Membership.mem s a) (κ a) (η a)) (MeasureTheory.Measure.sum fun n = …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> exact (measure_sum_seq _ a).symm
                /-
                  🎉 no goals
                -/


theorem lintegral_piecewise (a : α) (g : β → ℝ≥0∞) :
    ∫⁻ b, g b ∂piecewise hs κ η a = if a ∈ s then ∫⁻ b, g b ∂κ a else ∫⁻ b, g b ∂η a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝ : DecidablePred fun x => Membership.mem s x
    a : α
    g : β → ENNReal
    ⊢ Eq (MeasureTheory.lintegral ((ProbabilityTheory.Kernel.piecewise hs κ η) a)  …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  simp_rw [piecewise_apply]; split_ifs <;> rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem setLIntegral_piecewise (a : α) (g : β → ℝ≥0∞) (t : Set β) :
    ∫⁻ b in t, g b ∂piecewise hs κ η a =
      if a ∈ s then ∫⁻ b in t, g b ∂κ a else ∫⁻ b in t, g b ∂η a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    s : Set α
    hs : MeasurableSet s
    inst✝ : DecidablePred fun x => Membership.mem s x
    a : α
    g : β → ENNReal
    t : Set β
    ⊢ Eq (MeasureTheory.lintegral (((ProbabilityTheory.Kernel.piecewise hs κ η) a) …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  simp_rw [piecewise_apply]; split_ifs <;> rfl
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_piecewise := setLIntegral_piecewise


lemma exists_ae_eq_isMarkovKernel {μ : Measure α}
    (h : ∀ᵐ a ∂μ, IsProbabilityMeasure (κ a)) (h' : μ ≠ 0) :
    ∃ (η : Kernel α β), (κ =ᵐ[μ] η) ∧ IsMarkovKernel η := by
  classical
  obtain ⟨s, s_meas, μs, hs⟩ : ∃ s, MeasurableSet s ∧ μ s = 0
      ∧ ∀ a ∉ s, IsProbabilityMeasure (κ a) := by
    refine ⟨toMeasurable μ {a | ¬ IsProbabilityMeasure (κ a)}, measurableSet_toMeasurable _ _,
      by simpa [measure_toMeasurable] using h, ?_⟩
    intro a ha
    contrapose! ha
    exact subset_toMeasurable _ _ ha
  obtain ⟨a, ha⟩ : sᶜ.Nonempty := by
    contrapose! h'; simpa [μs, h'] using measure_univ_le_add_compl s (μ := μ)
  refine ⟨Kernel.piecewise s_meas (Kernel.const _ (κ a)) κ, ?_, ?_⟩
  · filter_upwards [measure_zero_iff_ae_nmem.1 μs] with b hb
    simp [hb, piecewise]
  · refine ⟨fun b ↦ ?_⟩
    by_cases hb : b ∈ s
    · simpa [hb, piecewise] using hs _ ha
    · simpa [hb, piecewise] using hs _ hb


