/-- The composition-product of a measure and a kernel. -/
noncomputable
def compProd (μ : Measure α) (κ : Kernel α β) : Measure (α × β) :=
  (Kernel.const Unit μ ⊗ₖ Kernel.prodMkLeft Unit κ) ()


@[inherit_doc]
scoped[ProbabilityTheory] infixl:100 " ⊗ₘ " => MeasureTheory.Measure.compProd


lemma compProd_of_not_sfinite (μ : Measure α) (κ : Kernel α β) (h : ¬ SFinite μ) :
    μ ⊗ₘ κ = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    h : Not (MeasureTheory.SFinite μ)
    ⊢ Eq (μ.compProd κ) 0
  -/
  rw [compProd, Kernel.compProd_of_not_isSFiniteKernel_left, Kernel.zero_apply]
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    h : Not (MeasureTheory.SFinite μ)
    ⊢ Not (ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.const Unit  …
  -/
  rwa [Kernel.isSFiniteKernel_const]
  /-
    🎉 no goals
  -/


lemma compProd_of_not_isSFiniteKernel (μ : Measure α) (κ : Kernel α β) (h : ¬ IsSFiniteKernel κ) :
    μ ⊗ₘ κ = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    h : Not (ProbabilityTheory.IsSFiniteKernel κ)
    ⊢ Eq (μ.compProd κ) 0
  -/
  rw [compProd, Kernel.compProd_of_not_isSFiniteKernel_right, Kernel.zero_apply]
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    h : Not (ProbabilityTheory.IsSFiniteKernel κ)
    ⊢ Not (ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.prodMkLeft  …
  -/
  rwa [Kernel.isSFiniteKernel_prodMkLeft_unit]
  /-
    🎉 no goals
  -/


                                                                                   /-
                                                                                     α : Type u_1
                                                                                     β : Type u_2
                                                                                     mα : MeasurableSpace α
                                                                                     mβ : MeasurableSpace β
                                                                                     κ : ProbabilityTheory.Kernel α β
                                                                                     ⊢ Eq (MeasureTheory.Measure.compProd 0 κ) 0
                                                                                   -/
@[simp] lemma compProd_zero_left (κ : Kernel α β) : (0 : Measure α) ⊗ₘ κ = 0 := by simp [compProd]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/

                                                                                    /-
                                                                                      α : Type u_1
                                                                                      β : Type u_2
                                                                                      mα : MeasurableSpace α
                                                                                      mβ : MeasurableSpace β
                                                                                      μ : MeasureTheory.Measure α
                                                                                      ⊢ Eq (μ.compProd 0) 0
                                                                                    -/
@[simp] lemma compProd_zero_right (μ : Measure α) : μ ⊗ₘ (0 : Kernel α β) = 0 := by simp [compProd]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


lemma compProd_apply [SFinite μ] [IsSFiniteKernel κ] {s : Set (α × β)} (hs : MeasurableSet s) :
    (μ ⊗ₘ κ) s = ∫⁻ a, κ a (Prod.mk a ⁻¹' s) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq ((μ.compProd κ) s) (MeasureTheory.lintegral μ fun a => (κ a) (Set.preimag …
  -/
  simp_rw [compProd, Kernel.compProd_apply hs, Kernel.const_apply, Kernel.prodMkLeft_apply']
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral μ fun b => (κ b) (setOf fun c => Membership.mem  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma compProd_apply_univ [SFinite μ] [IsMarkovKernel κ] : (μ ⊗ₘ κ) univ = μ univ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ Eq ((μ.compProd κ) Set.univ) (μ Set.univ)
  -/
  rw [compProd_apply MeasurableSet.univ]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ Eq (MeasureTheory.lintegral μ fun a => (κ a) (Set.preimage (Prod.mk a) Set.u …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma compProd_apply_prod [SFinite μ] [IsSFiniteKernel κ]
    {s : Set α} {t : Set β} (hs : MeasurableSet s) (ht : MeasurableSet t) :
    (μ ⊗ₘ κ) (s ×ˢ t) = ∫⁻ a in s, κ a t ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    s : Set α
    t : Set β
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ Eq ((μ.compProd κ) (SProd.sprod s t)) (MeasureTheory.lintegral (μ.restrict s …
  -/
  rw [compProd_apply (hs.prod ht), ← lintegral_indicator hs]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    s : Set α
    t : Set β
    hs : MeasurableSet s
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral μ fun a => (κ a) (Set.preimage (Prod.mk a) (SPro …
  -/
  congr with a
  classical
  rw [indicator_apply]
  split_ifs with ha <;> simp [ha]


lemma compProd_congr [IsSFiniteKernel κ] [IsSFiniteKernel η]
    (h : κ =ᵐ[μ] η) : μ ⊗ₘ κ = μ ⊗ₘ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
    ⊢ Eq (μ.compProd κ) (μ.compProd η)
  -/
  by_cases hμ : SFinite μ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      hμ : MeasureTheory.SFinite μ
      ⊢ Eq (μ.compProd κ) (μ.compProd η)
    -/
  · ext s hs
    have : (fun a ↦ κ a (Prod.mk a ⁻¹' s)) =ᵐ[μ] fun a ↦ η a (Prod.mk a ⁻¹' s) := by
      filter_upwards [h] with a ha using by rw [ha]
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      hμ : MeasureTheory.SFinite μ
      s : Set (Prod α β)
      hs : MeasurableSet s
      this : (MeasureTheory.ae μ).EventuallyEq (fun a => (κ a) (Set.preimage (Prod.m …
      ⊢ Eq ((μ.compProd κ) s) ((μ.compProd η) s)
    -/
    rw [compProd_apply hs, lintegral_congr_ae this, compProd_apply hs]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (MeasureTheory.ae μ).EventuallyEq ⇑κ ⇑η
      hμ : Not (MeasureTheory.SFinite μ)
      ⊢ Eq (μ.compProd κ) (μ.compProd η)
    -/
  · simp [compProd_of_not_sfinite _ _ hμ]
    /-
      🎉 no goals
    -/


lemma ae_compProd_of_ae_ae {p : α × β → Prop}
    (hp : MeasurableSet {x | p x}) (h : ∀ᵐ a ∂μ, ∀ᵐ b ∂(κ a), p (a, b)) :
    ∀ᵐ x ∂(μ ⊗ₘ κ), p x :=
  Kernel.ae_compProd_of_ae_ae hp h


lemma ae_ae_of_ae_compProd [SFinite μ] [IsSFiniteKernel κ] {p : α × β → Prop}
    (h : ∀ᵐ x ∂(μ ⊗ₘ κ), p x) :
    ∀ᵐ a ∂μ, ∀ᵐ b ∂κ a, p (a, b) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    p : Prod α β → Prop
    h : Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.compProd κ))
    ⊢ Filter.Eventually (fun a => Filter.Eventually (fun b => p { fst := a, snd := …
  -/
  convert Kernel.ae_ae_of_ae_compProd h -- Much faster with `convert`
  /-
    🎉 no goals
  -/


lemma ae_compProd_iff [SFinite μ] [IsSFiniteKernel κ] {p : α × β → Prop}
    (hp : MeasurableSet {x | p x}) :
    (∀ᵐ x ∂(μ ⊗ₘ κ), p x) ↔ ∀ᵐ a ∂μ, ∀ᵐ b ∂(κ a), p (a, b) :=
  Kernel.ae_compProd_iff hp


/-- The composition product of a measure and a constant kernel is the product between the two
measures. -/
@[simp]
lemma compProd_const {ν : Measure β} [SFinite μ] [SFinite ν] :
    μ ⊗ₘ (Kernel.const α ν) = μ.prod ν := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    ⊢ Eq (μ.compProd (ProbabilityTheory.Kernel.const α ν)) (μ.prod ν)
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq ((μ.compProd (ProbabilityTheory.Kernel.const α ν)) s) ((μ.prod ν) s)
  -/
  simp_rw [compProd_apply hs, prod_apply hs, Kernel.const_apply]
  /-
    🎉 no goals
  -/


lemma compProd_add_left (μ ν : Measure α) [SFinite μ] [SFinite ν] (κ : Kernel α β) :
    (μ + ν) ⊗ₘ κ = μ ⊗ₘ κ + ν ⊗ₘ κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    κ : ProbabilityTheory.Kernel α β
    ⊢ Eq ((HAdd.hAdd μ ν).compProd κ) (HAdd.hAdd (μ.compProd κ) (ν.compProd κ))
  -/
  by_cases hκ : IsSFiniteKernel κ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : MeasureTheory.SFinite ν
      κ : ProbabilityTheory.Kernel α β
      hκ : ProbabilityTheory.IsSFiniteKernel κ
      ⊢ Eq ((HAdd.hAdd μ ν).compProd κ) (HAdd.hAdd (μ.compProd κ) (ν.compProd κ))
    -/
  · simp_rw [Measure.compProd, Kernel.const_add, Kernel.compProd_add_left, Kernel.add_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : MeasureTheory.SFinite ν
      κ : ProbabilityTheory.Kernel α β
      hκ : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ Eq ((HAdd.hAdd μ ν).compProd κ) (HAdd.hAdd (μ.compProd κ) (ν.compProd κ))
    -/
  · simp [compProd_of_not_isSFiniteKernel _ _ hκ]
    /-
      🎉 no goals
    -/


lemma compProd_add_right (μ : Measure α) (κ η : Kernel α β)
    [IsSFiniteKernel κ] [IsSFiniteKernel η] :
    μ ⊗ₘ (κ + η) = μ ⊗ₘ κ + μ ⊗ₘ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (μ.compProd (HAdd.hAdd κ η)) (HAdd.hAdd (μ.compProd κ) (μ.compProd η))
  -/
  by_cases hμ : SFinite μ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hμ : MeasureTheory.SFinite μ
      ⊢ Eq (μ.compProd (HAdd.hAdd κ η)) (HAdd.hAdd (μ.compProd κ) (μ.compProd η))
    -/
  · simp_rw [Measure.compProd, Kernel.prodMkLeft_add, Kernel.compProd_add_right, Kernel.add_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hμ : Not (MeasureTheory.SFinite μ)
      ⊢ Eq (μ.compProd (HAdd.hAdd κ η)) (HAdd.hAdd (μ.compProd κ) (μ.compProd η))
    -/
  · simp [compProd_of_not_sfinite _ _ hμ]
    /-
      🎉 no goals
    -/


@[simp]
lemma fst_compProd (μ : Measure α) [SFinite μ] (κ : Kernel α β) [IsMarkovKernel κ] :
    (μ ⊗ₘ κ).fst = μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ Eq (μ.compProd κ).fst μ
  -/
  ext s
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    s : Set α
    a✝ : MeasurableSet s
    ⊢ Eq ((μ.compProd κ).fst s) (μ s)
  -/
  rw [compProd, Measure.fst, ← Kernel.fst_apply, Kernel.fst_compProd, Kernel.const_apply]
  /-
    🎉 no goals
  -/


lemma compProd_smul_left (a : ℝ≥0∞) [SFinite μ] [IsSFiniteKernel κ] :
    (a • μ) ⊗ₘ κ = a • (μ ⊗ₘ κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    a : ENNReal
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq ((HSMul.hSMul a μ).compProd κ) (HSMul.hSMul a (μ.compProd κ))
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    a : ENNReal
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq (((HSMul.hSMul a μ).compProd κ) s) ((HSMul.hSMul a (μ.compProd κ)) s)
  -/
  simp only [compProd_apply hs, lintegral_smul_measure, smul_apply, smul_eq_mul]
  /-
    🎉 no goals
  -/


lemma lintegral_compProd [SFinite μ] [IsSFiniteKernel κ]
    {f : α × β → ℝ≥0∞} (hf : Measurable f) :
    ∫⁻ x, f x ∂(μ ⊗ₘ κ) = ∫⁻ a, ∫⁻ b, f (a, b) ∂(κ a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : Prod α β → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral (μ.compProd κ) fun x => f x) (MeasureTheory.lint …
  -/
  rw [compProd, Kernel.lintegral_compProd _ _ _ hf]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : Prod α β → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral ((ProbabilityTheory.Kernel.const Unit μ) Unit.un …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma setLIntegral_compProd [SFinite μ] [IsSFiniteKernel κ]
    {f : α × β → ℝ≥0∞} (hf : Measurable f)
    {s : Set α} (hs : MeasurableSet s) {t : Set β} (ht : MeasurableSet t) :
    ∫⁻ x in s ×ˢ t, f x ∂(μ ⊗ₘ κ) = ∫⁻ a in s, ∫⁻ b in t, f (a, b) ∂(κ a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : Prod α β → ENNReal
    hf : Measurable f
    s : Set α
    hs : MeasurableSet s
    t : Set β
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral ((μ.compProd κ).restrict (SProd.sprod s t)) fun  …
  -/
  rw [compProd, Kernel.setLIntegral_compProd _ _ _ hf hs ht]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : Prod α β → ENNReal
    hf : Measurable f
    s : Set α
    hs : MeasurableSet s
    t : Set β
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (((ProbabilityTheory.Kernel.const Unit μ) Unit.u …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_compProd := setLIntegral_compProd


lemma integrable_compProd_iff [SFinite μ] [IsSFiniteKernel κ] {E : Type*} [NormedAddCommGroup E]
    {f : α × β → E} (hf : AEStronglyMeasurable f (μ ⊗ₘ κ)) :
    Integrable f (μ ⊗ₘ κ) ↔
      (∀ᵐ x ∂μ, Integrable (fun y => f (x, y)) (κ x)) ∧
        Integrable (fun x => ∫ y, ‖f (x, y)‖ ∂(κ x)) μ := by
  simp_rw [Measure.compProd, ProbabilityTheory.integrable_compProd_iff hf, Kernel.prodMkLeft_apply,
    Kernel.const_apply]


lemma integral_compProd [SFinite μ] [IsSFiniteKernel κ] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    {f : α × β → E} (hf : Integrable f (μ ⊗ₘ κ)) :
    ∫ x, f x ∂(μ ⊗ₘ κ) = ∫ a, ∫ b, f (a, b) ∂(κ a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Prod α β → E
    hf : MeasureTheory.Integrable f (μ.compProd κ)
    ⊢ Eq (MeasureTheory.integral (μ.compProd κ) fun x => f x) (MeasureTheory.integ …
  -/
  rw [compProd, ProbabilityTheory.integral_compProd hf]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Prod α β → E
    hf : MeasureTheory.Integrable f (μ.compProd κ)
    ⊢ Eq (MeasureTheory.integral ((ProbabilityTheory.Kernel.const Unit μ) Unit.uni …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma setIntegral_compProd [SFinite μ] [IsSFiniteKernel κ] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    {s : Set α} (hs : MeasurableSet s) {t : Set β} (ht : MeasurableSet t)
    {f : α × β → E} (hf : IntegrableOn f (s ×ˢ t) (μ ⊗ₘ κ))  :
    ∫ x in s ×ˢ t, f x ∂(μ ⊗ₘ κ) = ∫ a in s, ∫ b in t, f (a, b) ∂(κ a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set α
    hs : MeasurableSet s
    t : Set β
    ht : MeasurableSet t
    f : Prod α β → E
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) (μ.compProd κ)
    ⊢ Eq (MeasureTheory.integral ((μ.compProd κ).restrict (SProd.sprod s t)) fun x …
  -/
  rw [compProd, ProbabilityTheory.setIntegral_compProd hs ht hf]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set α
    hs : MeasurableSet s
    t : Set β
    ht : MeasurableSet t
    f : Prod α β → E
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) (μ.compProd κ)
    ⊢ Eq (MeasureTheory.integral (((ProbabilityTheory.Kernel.const Unit μ) Unit.un …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_compProd := setIntegral_compProd


lemma dirac_compProd_apply [MeasurableSingletonClass α] {a : α} [IsSFiniteKernel κ]
    {s : Set (α × β)} (hs : MeasurableSet s) :
    (Measure.dirac a ⊗ₘ κ) s = κ a (Prod.mk a ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasurableSingletonClass α
    a : α
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    s : Set (Prod α β)
    hs : MeasurableSet s
    ⊢ Eq (((MeasureTheory.Measure.dirac a).compProd κ) s) ((κ a) (Set.preimage (Pr …
  -/
  rw [compProd_apply hs, lintegral_dirac]
  /-
    🎉 no goals
  -/


lemma dirac_unit_compProd (κ : Kernel Unit β) [IsSFiniteKernel κ] :
    Measure.dirac () ⊗ₘ κ = (κ ()).map (Prod.mk ()) := by
  /-
    β : Type u_2
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel Unit β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq ((MeasureTheory.Measure.dirac Unit.unit).compProd κ) (MeasureTheory.Measu …
  -/
  ext s hs; rw [dirac_compProd_apply hs, Measure.map_apply measurable_prod_mk_left hs]
            /-
              🎉 no goals
            -/


lemma dirac_unit_compProd_const (μ : Measure β) [IsFiniteMeasure μ] :
    Measure.dirac () ⊗ₘ Kernel.const Unit μ = μ.map (Prod.mk ()) := by
  /-
    β : Type u_2
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq ((MeasureTheory.Measure.dirac Unit.unit).compProd (ProbabilityTheory.Kern …
  -/
  rw [dirac_unit_compProd, Kernel.const_apply]
  /-
    🎉 no goals
  -/


lemma snd_dirac_unit_compProd_const (μ : Measure β) [IsFiniteMeasure μ] :
                                                            /-
                                                              β : Type u_2
                                                              mβ : MeasurableSpace β
                                                              μ : MeasureTheory.Measure β
                                                              inst✝ : MeasureTheory.IsFiniteMeasure μ
                                                              ⊢ Eq ((MeasureTheory.Measure.dirac Unit.unit).compProd (ProbabilityTheory.Kern …
                                                            -/
    snd (Measure.dirac () ⊗ₘ Kernel.const Unit μ) = μ := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    mα : MeasurableSpace α
                                    mβ : MeasurableSpace β
                                    μ ν : MeasureTheory.Measure α
                                    κ η : ProbabilityTheory.Kernel α β
                                    ⊢ MeasureTheory.SFinite (μ.compProd κ)
                                  -/
instance : SFinite (μ ⊗ₘ κ) := by rw [compProd]; infer_instance
                                                 /-
                                                   🎉 no goals
                                                 -/


instance [IsFiniteMeasure μ] [IsFiniteKernel κ] : IsFiniteMeasure (μ ⊗ₘ κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ MeasureTheory.IsFiniteMeasure (μ.compProd κ)
  -/
  rw [compProd]; infer_instance
                 /-
                   🎉 no goals
                 -/


instance [IsProbabilityMeasure μ] [IsMarkovKernel κ] : IsProbabilityMeasure (μ ⊗ₘ κ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : ProbabilityTheory.IsMarkovKernel κ
    ⊢ MeasureTheory.IsProbabilityMeasure (μ.compProd κ)
  -/
  rw [compProd]; infer_instance
                 /-
                   🎉 no goals
                 -/


lemma AbsolutelyContinuous.compProd_left [SFinite ν] (hμν : μ ≪ ν) (κ : Kernel α β) :
    μ ⊗ₘ κ ≪ ν ⊗ₘ κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite ν
    hμν : μ.AbsolutelyContinuous ν
    κ : ProbabilityTheory.Kernel α β
    ⊢ (μ.compProd κ).AbsolutelyContinuous (ν.compProd κ)
  -/
  by_cases hκ : IsSFiniteKernel κ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite ν
      hμν : μ.AbsolutelyContinuous ν
      κ : ProbabilityTheory.Kernel α β
      hκ : ProbabilityTheory.IsSFiniteKernel κ
      ⊢ (μ.compProd κ).AbsolutelyContinuous (ν.compProd κ)
    -/
  · have : SFinite μ := sFinite_of_absolutelyContinuous hμν
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite ν
      hμν : μ.AbsolutelyContinuous ν
      κ : ProbabilityTheory.Kernel α β
      hκ : ProbabilityTheory.IsSFiniteKernel κ
      this : MeasureTheory.SFinite μ
      ⊢ (μ.compProd κ).AbsolutelyContinuous (ν.compProd κ)
    -/
    refine Measure.AbsolutelyContinuous.mk fun s hs hs_zero ↦ ?_
    rw [Measure.compProd_apply hs, lintegral_eq_zero_iff (Kernel.measurable_kernel_prod_mk_left hs)]
      at hs_zero ⊢
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite ν
      hμν : μ.AbsolutelyContinuous ν
      κ : ProbabilityTheory.Kernel α β
      hκ : ProbabilityTheory.IsSFiniteKernel κ
      this : MeasureTheory.SFinite μ
      s : Set (Prod α β)
      hs : MeasurableSet s
      hs_zero : (MeasureTheory.ae ν).EventuallyEq (fun a => (κ a) (Set.preimage (Pro …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => (κ a) (Set.preimage (Prod.mk a)  …
    -/
    exact hμν.ae_eq hs_zero
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite ν
      hμν : μ.AbsolutelyContinuous ν
      κ : ProbabilityTheory.Kernel α β
      hκ : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ (μ.compProd κ).AbsolutelyContinuous (ν.compProd κ)
    -/
  · simp [compProd_of_not_isSFiniteKernel _ _ hκ]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-11")]
alias absolutelyContinuous_compProd_left := AbsolutelyContinuous.compProd_left


lemma AbsolutelyContinuous.compProd_right [SFinite μ] [IsSFiniteKernel η]
    (hκη : ∀ᵐ a ∂μ, κ a ≪ η a) :
    μ ⊗ₘ κ ≪ μ ⊗ₘ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hκη : Filter.Eventually (fun a => (κ a).AbsolutelyContinuous (η a)) (MeasureTh …
    ⊢ (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
  -/
  by_cases hκ : IsSFiniteKernel κ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hκη : Filter.Eventually (fun a => (κ a).AbsolutelyContinuous (η a)) (MeasureTh …
      hκ : ProbabilityTheory.IsSFiniteKernel κ
      ⊢ (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
    -/
  · refine Measure.AbsolutelyContinuous.mk fun s hs hs_zero ↦ ?_
    rw [Measure.compProd_apply hs, lintegral_eq_zero_iff (Kernel.measurable_kernel_prod_mk_left hs)]
      at hs_zero ⊢
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hκη : Filter.Eventually (fun a => (κ a).AbsolutelyContinuous (η a)) (MeasureTh …
      hκ : ProbabilityTheory.IsSFiniteKernel κ
      s : Set (Prod α β)
      hs : MeasurableSet s
      hs_zero : (MeasureTheory.ae μ).EventuallyEq (fun a => (η a) (Set.preimage (Pro …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => (κ a) (Set.preimage (Prod.mk a)  …
    -/
    filter_upwards [hs_zero, hκη] with a ha_zero ha_ac using ha_ac ha_zero
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hκη : Filter.Eventually (fun a => (κ a).AbsolutelyContinuous (η a)) (MeasureTh …
      hκ : Not (ProbabilityTheory.IsSFiniteKernel κ)
      ⊢ (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
    -/
  · simp [compProd_of_not_isSFiniteKernel _ _ hκ]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-11")]
alias absolutelyContinuous_compProd_right := AbsolutelyContinuous.compProd_right


lemma AbsolutelyContinuous.compProd [SFinite ν] [IsSFiniteKernel η]
    (hμν : μ ≪ ν) (hκη : ∀ᵐ a ∂μ, κ a ≪ η a) :
    μ ⊗ₘ κ ≪ ν ⊗ₘ η :=
  have : SFinite μ := sFinite_of_absolutelyContinuous hμν
  (Measure.AbsolutelyContinuous.compProd_right hκη).trans (hμν.compProd_left _)


@[deprecated (since := "2024-12-11")]
alias absolutelyContinuous_compProd := AbsolutelyContinuous.compProd


lemma absolutelyContinuous_of_compProd [SFinite μ] [IsSFiniteKernel κ] [h_zero : ∀ a, NeZero (κ a)]
    (h : μ ⊗ₘ κ ≪ ν ⊗ₘ η) :
    μ ≪ ν := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    ⊢ μ.AbsolutelyContinuous ν
  -/
  refine Measure.AbsolutelyContinuous.mk (fun s hs hs0 ↦ ?_)
  have h1 : (ν ⊗ₘ η) (s ×ˢ univ) = 0 := by
    by_cases hν : SFinite ν
    swap; · simp [compProd_of_not_sfinite _ _ hν]
    by_cases hη : IsSFiniteKernel η
    swap; · simp [compProd_of_not_isSFiniteKernel _ _ hη]
    rw [Measure.compProd_apply_prod hs MeasurableSet.univ]
    exact setLIntegral_measure_zero _ _ hs0
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    ⊢ Eq (μ s) 0
  -/
  have h2 : (μ ⊗ₘ κ) (s ×ˢ univ) = 0 := h h1
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    h2 : Eq ((μ.compProd κ) (SProd.sprod s Set.univ)) 0
    ⊢ Eq (μ s) 0
  -/
  rw [Measure.compProd_apply_prod hs MeasurableSet.univ, lintegral_eq_zero_iff] at h2
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    h2 : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun a => (κ a) Set.univ) 0
    ⊢ Eq (μ s) 0
  -/
  swap; · exact Kernel.measurable_coe _ MeasurableSet.univ
          /-
            🎉 no goals
          -/
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    h2 : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun a => (κ a) Set.univ) 0
    ⊢ Eq (μ s) 0
  -/
  by_contra hμs
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    h2 : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun a => (κ a) Set.univ) 0
    hμs : Not (Eq (μ s) 0)
    ⊢ False
  -/
  have : Filter.NeBot (ae (μ.restrict s)) := by simp [hμs]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    h2 : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun a => (κ a) Set.univ) 0
    hμs : Not (Eq (μ s) 0)
    this : (MeasureTheory.ae (μ.restrict s)).NeBot
    ⊢ False
  -/
  obtain ⟨a, ha⟩ : ∃ a, κ a univ = 0 := h2.exists
  /-
    case intro
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    h2 : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun a => (κ a) Set.univ) 0
    hμs : Not (Eq (μ s) 0)
    this : (MeasureTheory.ae (μ.restrict s)).NeBot
    a : α
    ha : Eq ((κ a) Set.univ) 0
    ⊢ False
  -/
  refine absurd ha ?_
  /-
    case intro
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    h2 : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun a => (κ a) Set.univ) 0
    hμs : Not (Eq (μ s) 0)
    this : (MeasureTheory.ae (μ.restrict s)).NeBot
    a : α
    ha : Eq ((κ a) Set.univ) 0
    ⊢ Not (Eq ((κ a) Set.univ) 0)
  -/
  simp only [Measure.measure_univ_eq_zero]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    h_zero : ∀ (a : α), NeZero (κ a)
    h : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    s : Set α
    hs : MeasurableSet s
    hs0 : Eq (ν s) 0
    h1 : Eq ((ν.compProd η) (SProd.sprod s Set.univ)) 0
    h2 : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun a => (κ a) Set.univ) 0
    hμs : Not (Eq (μ s) 0)
    this : (MeasureTheory.ae (μ.restrict s)).NeBot
    a : α
    ha : Eq ((κ a) Set.univ) 0
    ⊢ Not (Eq (κ a) 0)
  -/
  exact (h_zero a).out
  /-
    🎉 no goals
  -/


lemma absolutelyContinuous_compProd_left_iff [SFinite μ] [SFinite ν]
    [IsFiniteKernel κ] [∀ a, NeZero (κ a)] :
    μ ⊗ₘ κ ≪ ν ⊗ₘ κ ↔ μ ≪ ν :=
  ⟨absolutelyContinuous_of_compProd, fun h ↦ h.compProd_left κ⟩


lemma AbsolutelyContinuous.compProd_of_compProd [SFinite ν] [IsSFiniteKernel η]
    (hμν : μ ≪ ν) (hκη : μ ⊗ₘ κ ≪ μ ⊗ₘ η) :
    μ ⊗ₘ κ ≪ ν ⊗ₘ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hμν : μ.AbsolutelyContinuous ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
    ⊢ (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
  -/
  by_cases hμ : SFinite μ
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hμν : μ.AbsolutelyContinuous ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
    hμ : MeasureTheory.SFinite μ
    ⊢ (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
  -/
  swap; · rw [compProd_of_not_sfinite _ _ hμ]; simp
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hμν : μ.AbsolutelyContinuous ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
    hμ : MeasureTheory.SFinite μ
    ⊢ (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
  -/
  refine AbsolutelyContinuous.mk fun s hs hs_zero ↦ ?_
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hμν : μ.AbsolutelyContinuous ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
    hμ : MeasureTheory.SFinite μ
    s : Set (Prod α β)
    hs : MeasurableSet s
    hs_zero : Eq ((ν.compProd η) s) 0
    ⊢ Eq ((μ.compProd κ) s) 0
  -/
  suffices (μ ⊗ₘ η) s = 0 from hκη this
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hμν : μ.AbsolutelyContinuous ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
    hμ : MeasureTheory.SFinite μ
    s : Set (Prod α β)
    hs : MeasurableSet s
    hs_zero : Eq ((ν.compProd η) s) 0
    ⊢ Eq ((μ.compProd η) s) 0
  -/
  rw [measure_zero_iff_ae_nmem, ae_compProd_iff hs.compl] at hs_zero ⊢
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hμν : μ.AbsolutelyContinuous ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
    hμ : MeasureTheory.SFinite μ
    s : Set (Prod α β)
    hs : MeasurableSet s
    hs_zero : Filter.Eventually (fun a => Filter.Eventually (fun b => Not (Members …
    ⊢ Filter.Eventually (fun a => Filter.Eventually (fun b => Not (Membership.mem  …
  -/
  exact hμν.ae_le hs_zero
  /-
    🎉 no goals
  -/


lemma MutuallySingular.compProd_of_left (hμν : μ ⟂ₘ ν) (κ η : Kernel α β) :
    μ ⊗ₘ κ ⟂ₘ ν ⊗ₘ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  by_cases hμ : SFinite μ
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  swap; · rw [compProd_of_not_sfinite _ _ hμ]; simp
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  by_cases hν : SFinite ν
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  swap; · rw [compProd_of_not_sfinite _ _ hν]; simp
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  by_cases hκ : IsSFiniteKernel κ
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  swap; · rw [compProd_of_not_isSFiniteKernel _ _ hκ]; simp
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  by_cases hη : IsSFiniteKernel η
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    hη : ProbabilityTheory.IsSFiniteKernel η
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  swap; · rw [compProd_of_not_isSFiniteKernel _ _ hη]; simp
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    hη : ProbabilityTheory.IsSFiniteKernel η
    ⊢ (μ.compProd κ).MutuallySingular (ν.compProd η)
  -/
  refine ⟨hμν.nullSet ×ˢ univ, hμν.measurableSet_nullSet.prod .univ, ?_⟩
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    hη : ProbabilityTheory.IsSFiniteKernel η
    ⊢ And (Eq ((μ.compProd κ) (SProd.sprod hμν.nullSet Set.univ)) 0) (Eq ((ν.compP …
  -/
  rw [compProd_apply_prod hμν.measurableSet_nullSet .univ, compl_prod_eq_union]
  simp only [MutuallySingular.restrict_nullSet, lintegral_zero_measure, compl_univ,
    prod_empty, union_empty, true_and]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    hη : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq ((ν.compProd η) (SProd.sprod (HasCompl.compl hμν.nullSet) Set.univ)) 0
  -/
  rw [compProd_apply_prod hμν.measurableSet_nullSet.compl .univ]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    hμν : μ.MutuallySingular ν
    κ η : ProbabilityTheory.Kernel α β
    hμ : MeasureTheory.SFinite μ
    hν : MeasureTheory.SFinite ν
    hκ : ProbabilityTheory.IsSFiniteKernel κ
    hη : ProbabilityTheory.IsSFiniteKernel η
    ⊢ Eq (MeasureTheory.lintegral (ν.restrict (HasCompl.compl hμν.nullSet)) fun a  …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma mutuallySingular_of_mutuallySingular_compProd {ξ : Measure α}
    [SFinite μ] [SFinite ν] [IsSFiniteKernel κ] [IsSFiniteKernel η]
    (h : μ ⊗ₘ κ ⟂ₘ ν ⊗ₘ η) (hμ : ξ ≪ μ) (hν : ξ ≪ ν) :
    ∀ᵐ x ∂ξ, κ x ⟂ₘ η x := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    ξ : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    h : (μ.compProd κ).MutuallySingular (ν.compProd η)
    hμ : ξ.AbsolutelyContinuous μ
    hν : ξ.AbsolutelyContinuous ν
    ⊢ Filter.Eventually (fun x => (κ x).MutuallySingular (η x)) (MeasureTheory.ae ξ)
  -/
  have hs : MeasurableSet h.nullSet := h.measurableSet_nullSet
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    ξ : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    h : (μ.compProd κ).MutuallySingular (ν.compProd η)
    hμ : ξ.AbsolutelyContinuous μ
    hν : ξ.AbsolutelyContinuous ν
    hs : MeasurableSet h.nullSet
    ⊢ Filter.Eventually (fun x => (κ x).MutuallySingular (η x)) (MeasureTheory.ae ξ)
  -/
  have hμ_zero : (μ ⊗ₘ κ) h.nullSet = 0 := h.measure_nullSet
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    ξ : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    h : (μ.compProd κ).MutuallySingular (ν.compProd η)
    hμ : ξ.AbsolutelyContinuous μ
    hν : ξ.AbsolutelyContinuous ν
    hs : MeasurableSet h.nullSet
    hμ_zero : Eq ((μ.compProd κ) h.nullSet) 0
    ⊢ Filter.Eventually (fun x => (κ x).MutuallySingular (η x)) (MeasureTheory.ae ξ)
  -/
  have hν_zero : (ν ⊗ₘ η) h.nullSetᶜ = 0 := h.measure_compl_nullSet
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    ξ : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasureTheory.SFinite ν
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    h : (μ.compProd κ).MutuallySingular (ν.compProd η)
    hμ : ξ.AbsolutelyContinuous μ
    hν : ξ.AbsolutelyContinuous ν
    hs : MeasurableSet h.nullSet
    hμ_zero : Eq ((μ.compProd κ) h.nullSet) 0
    hν_zero : Eq ((ν.compProd η) (HasCompl.compl h.nullSet)) 0
    ⊢ Filter.Eventually (fun x => (κ x).MutuallySingular (η x)) (MeasureTheory.ae ξ)
  -/
  rw [compProd_apply, lintegral_eq_zero_iff'] at hμ_zero hν_zero
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      ξ : MeasureTheory.Measure α
      inst✝³ : MeasureTheory.SFinite μ
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (μ.compProd κ).MutuallySingular (ν.compProd η)
      hμ : ξ.AbsolutelyContinuous μ
      hν : ξ.AbsolutelyContinuous ν
      hs : MeasurableSet h.nullSet
      hμ_zero : (MeasureTheory.ae μ).EventuallyEq (fun a => (κ a) (Set.preimage (Pro …
      hν_zero : (MeasureTheory.ae ν).EventuallyEq (fun a => (η a) (Set.preimage (Pro …
      ⊢ Filter.Eventually (fun x => (κ x).MutuallySingular (η x)) (MeasureTheory.ae ξ)
    -/
  · filter_upwards [hμ hμ_zero, hν hν_zero] with x hxμ hxν
    /-
      case h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      ξ : MeasureTheory.Measure α
      inst✝³ : MeasureTheory.SFinite μ
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (μ.compProd κ).MutuallySingular (ν.compProd η)
      hμ : ξ.AbsolutelyContinuous μ
      hν : ξ.AbsolutelyContinuous ν
      hs : MeasurableSet h.nullSet
      hμ_zero : (MeasureTheory.ae μ).EventuallyEq (fun a => (κ a) (Set.preimage (Pro …
      hν_zero : (MeasureTheory.ae ν).EventuallyEq (fun a => (η a) (Set.preimage (Pro …
      x : α
      hxμ : Eq ((κ x) (Set.preimage (Prod.mk x) h.nullSet)) (0 x)
      hxν : Eq ((η x) (Set.preimage (Prod.mk x) (HasCompl.compl h.nullSet))) (0 x)
      ⊢ (κ x).MutuallySingular (η x)
    -/
    exact ⟨Prod.mk x ⁻¹' h.nullSet, measurable_prod_mk_left hs, ⟨hxμ, hxν⟩⟩
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      ξ : MeasureTheory.Measure α
      inst✝³ : MeasureTheory.SFinite μ
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (μ.compProd κ).MutuallySingular (ν.compProd η)
      hμ : ξ.AbsolutelyContinuous μ
      hν : ξ.AbsolutelyContinuous ν
      hs : MeasurableSet h.nullSet
      hμ_zero : (MeasureTheory.ae μ).EventuallyEq (fun a => (κ a) (Set.preimage (Pro …
      hν_zero : Eq (MeasureTheory.lintegral ν fun a => (η a) (Set.preimage (Prod.mk  …
      ⊢ AEMeasurable (fun a => (η a) (Set.preimage (Prod.mk a) (HasCompl.compl h.nul …
    -/
  · exact (Kernel.measurable_kernel_prod_mk_left hs.compl).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      ξ : MeasureTheory.Measure α
      inst✝³ : MeasureTheory.SFinite μ
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (μ.compProd κ).MutuallySingular (ν.compProd η)
      hμ : ξ.AbsolutelyContinuous μ
      hν : ξ.AbsolutelyContinuous ν
      hs : MeasurableSet h.nullSet
      hμ_zero : Eq (MeasureTheory.lintegral μ fun a => (κ a) (Set.preimage (Prod.mk  …
      hν_zero : Eq (MeasureTheory.lintegral ν fun a => (η a) (Set.preimage (Prod.mk  …
      ⊢ AEMeasurable (fun a => (κ a) (Set.preimage (Prod.mk a) h.nullSet)) μ
    -/
  · exact (Kernel.measurable_kernel_prod_mk_left hs).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      ξ : MeasureTheory.Measure α
      inst✝³ : MeasureTheory.SFinite μ
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (μ.compProd κ).MutuallySingular (ν.compProd η)
      hμ : ξ.AbsolutelyContinuous μ
      hν : ξ.AbsolutelyContinuous ν
      hs : MeasurableSet h.nullSet
      hμ_zero : Eq (MeasureTheory.lintegral μ fun a => (κ a) (Set.preimage (Prod.mk  …
      hν_zero : Eq ((ν.compProd η) (HasCompl.compl h.nullSet)) 0
      ⊢ MeasurableSet (HasCompl.compl h.nullSet)
    -/
  · exact hs.compl
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      ξ : MeasureTheory.Measure α
      inst✝³ : MeasureTheory.SFinite μ
      inst✝² : MeasureTheory.SFinite ν
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      h : (μ.compProd κ).MutuallySingular (ν.compProd η)
      hμ : ξ.AbsolutelyContinuous μ
      hν : ξ.AbsolutelyContinuous ν
      hs : MeasurableSet h.nullSet
      hμ_zero : Eq ((μ.compProd κ) h.nullSet) 0
      hν_zero : Eq ((ν.compProd η) (HasCompl.compl h.nullSet)) 0
      ⊢ MeasurableSet h.nullSet
    -/
  · exact hs
    /-
      🎉 no goals
    -/


lemma mutuallySingular_compProd_left_iff [SFinite μ] [SigmaFinite ν]
    [IsSFiniteKernel κ] [hκ : ∀ x, NeZero (κ x)] :
    μ ⊗ₘ κ ⟂ₘ ν ⊗ₘ κ ↔ μ ⟂ₘ ν := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : ∀ (x : α), NeZero (κ x)
    ⊢ Iff ((μ.compProd κ).MutuallySingular (ν.compProd κ)) (μ.MutuallySingular ν)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.compProd_of_left _ _⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : ∀ (x : α), NeZero (κ x)
    h : (μ.compProd κ).MutuallySingular (ν.compProd κ)
    ⊢ μ.MutuallySingular ν
  -/
  rw [← withDensity_rnDeriv_eq_zero]
  have hh := mutuallySingular_of_mutuallySingular_compProd h ?_ ?_
    (ξ := ν.withDensity (μ.rnDeriv ν))
  /-
    case refine_3
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : ∀ (x : α), NeZero (κ x)
    h : (μ.compProd κ).MutuallySingular (ν.compProd κ)
    hh : Filter.Eventually (fun x => (κ x).MutuallySingular (κ x)) (MeasureTheory. …
    ⊢ Eq (ν.withDensity (μ.rnDeriv ν)) 0
  -/
  rotate_left
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ : ProbabilityTheory.Kernel α β
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      hκ : ∀ (x : α), NeZero (κ x)
      h : (μ.compProd κ).MutuallySingular (ν.compProd κ)
      ⊢ (ν.withDensity (μ.rnDeriv ν)).AbsolutelyContinuous μ
    -/
  · exact absolutelyContinuous_of_le (μ.withDensity_rnDeriv_le ν)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ : ProbabilityTheory.Kernel α β
      inst✝² : MeasureTheory.SFinite μ
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      hκ : ∀ (x : α), NeZero (κ x)
      h : (μ.compProd κ).MutuallySingular (ν.compProd κ)
      ⊢ (ν.withDensity (μ.rnDeriv ν)).AbsolutelyContinuous ν
    -/
  · exact withDensity_absolutelyContinuous _ _
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : ∀ (x : α), NeZero (κ x)
    h : (μ.compProd κ).MutuallySingular (ν.compProd κ)
    hh : Filter.Eventually (fun x => (κ x).MutuallySingular (κ x)) (MeasureTheory. …
    ⊢ Eq (ν.withDensity (μ.rnDeriv ν)) 0
  -/
  simp_rw [MutuallySingular.self_iff, (hκ _).ne] at hh
  /-
    case refine_3
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ : ProbabilityTheory.Kernel α β
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hκ : ∀ (x : α), NeZero (κ x)
    h : (μ.compProd κ).MutuallySingular (ν.compProd κ)
    hh : Filter.Eventually (fun x => False) (MeasureTheory.ae (ν.withDensity (μ.rn …
    ⊢ Eq (ν.withDensity (μ.rnDeriv ν)) 0
  -/
  exact ae_eq_bot.mp (Filter.eventually_false_iff_eq_bot.mp hh)
  /-
    🎉 no goals
  -/


lemma AbsolutelyContinuous.mutuallySingular_compProd_iff [SigmaFinite μ] [SigmaFinite ν]
    (hμν : μ ≪ ν) :
    μ ⊗ₘ κ ⟂ₘ ν ⊗ₘ η ↔ μ ⊗ₘ κ ⟂ₘ μ ⊗ₘ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Iff ((μ.compProd κ).MutuallySingular (ν.compProd η)) ((μ.compProd κ).Mutuall …
  -/
  conv_lhs => rw [ν.haveLebesgueDecomposition_add μ]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Iff ((μ.compProd κ).MutuallySingular ((HAdd.hAdd (ν.singularPart μ) (μ.withD …
  -/
  rw [compProd_add_left, MutuallySingular.add_right_iff]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Iff (And ((μ.compProd κ).MutuallySingular ((ν.singularPart μ).compProd η)) ( …
  -/
  simp only [(mutuallySingular_singularPart ν μ).symm.compProd_of_left κ η, true_and]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Iff ((μ.compProd κ).MutuallySingular ((μ.withDensity (ν.rnDeriv μ)).compProd …
  -/
  refine ⟨fun h ↦ h.mono_ac .rfl ?_, fun h ↦ h.mono_ac .rfl ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      h : (μ.compProd κ).MutuallySingular ((μ.withDensity (ν.rnDeriv μ)).compProd η)
      ⊢ (μ.compProd η).AbsolutelyContinuous ((μ.withDensity (ν.rnDeriv μ)).compProd η)
    -/
  · exact (absolutelyContinuous_withDensity_rnDeriv hμν).compProd_left _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      h : (μ.compProd κ).MutuallySingular (μ.compProd η)
      ⊢ ((μ.withDensity (ν.rnDeriv μ)).compProd η).AbsolutelyContinuous (μ.compProd η)
    -/
  · exact (withDensity_absolutelyContinuous μ (ν.rnDeriv μ)).compProd_left _
    /-
      🎉 no goals
    -/


lemma mutuallySingular_compProd_iff [SigmaFinite μ] [SigmaFinite ν] :
    μ ⊗ₘ κ ⟂ₘ ν ⊗ₘ η ↔ ∀ ξ, SFinite ξ → ξ ≪ μ → ξ ≪ ν → ξ ⊗ₘ κ ⟂ₘ ξ ⊗ₘ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ Iff ((μ.compProd κ).MutuallySingular (ν.compProd η)) (∀ (ξ : MeasureTheory.M …
  -/
  conv_lhs => rw [μ.haveLebesgueDecomposition_add ν]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ Iff (((HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν))).compProd  …
  -/
  rw [compProd_add_left, MutuallySingular.add_left_iff]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ Iff (And (((μ.singularPart ν).compProd κ).MutuallySingular (ν.compProd η)) ( …
  -/
  simp only [(mutuallySingular_singularPart μ ν).compProd_of_left κ η, true_and]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ Iff (((ν.withDensity (μ.rnDeriv ν)).compProd κ).MutuallySingular (ν.compProd …
  -/
  rw [(withDensity_absolutelyContinuous ν (μ.rnDeriv ν)).mutuallySingular_compProd_iff]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ Iff (((ν.withDensity (μ.rnDeriv ν)).compProd κ).MutuallySingular ((ν.withDen …
  -/
  refine ⟨fun h ξ hξ hξμ hξν ↦ ?_, fun h ↦ ?_⟩
  · exact h.mono_ac ((hξμ.withDensity_rnDeriv hξν).compProd_left _)
      ((hξμ.withDensity_rnDeriv hξν).compProd_left _)
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      μ ν : MeasureTheory.Measure α
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      h : ∀ (ξ : MeasureTheory.Measure α), MeasureTheory.SFinite ξ → ξ.AbsolutelyCon …
      ⊢ ((ν.withDensity (μ.rnDeriv ν)).compProd κ).MutuallySingular ((ν.withDensity  …
    -/
  · refine h _ ?_ ?_ ?_
      /-
        case refine_2.refine_1
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        μ ν : MeasureTheory.Measure α
        κ η : ProbabilityTheory.Kernel α β
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        h : ∀ (ξ : MeasureTheory.Measure α), MeasureTheory.SFinite ξ → ξ.AbsolutelyCon …
        ⊢ MeasureTheory.SFinite (ν.withDensity (μ.rnDeriv ν))
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        μ ν : MeasureTheory.Measure α
        κ η : ProbabilityTheory.Kernel α β
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        h : ∀ (ξ : MeasureTheory.Measure α), MeasureTheory.SFinite ξ → ξ.AbsolutelyCon …
        ⊢ (ν.withDensity (μ.rnDeriv ν)).AbsolutelyContinuous μ
      -/
    · exact absolutelyContinuous_of_le (withDensity_rnDeriv_le _ _)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_3
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        μ ν : MeasureTheory.Measure α
        κ η : ProbabilityTheory.Kernel α β
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        h : ∀ (ξ : MeasureTheory.Measure α), MeasureTheory.SFinite ξ → ξ.AbsolutelyCon …
        ⊢ (ν.withDensity (μ.rnDeriv ν)).AbsolutelyContinuous ν
      -/
    · exact withDensity_absolutelyContinuous ν (μ.rnDeriv ν)
      /-
        🎉 no goals
      -/


lemma absolutelyContinuous_compProd_of_compProd [SigmaFinite μ] [SigmaFinite ν]
    (hκη : μ ⊗ₘ κ ≪ ν ⊗ₘ η) :
    μ ⊗ₘ κ ≪ μ ⊗ₘ η := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (ν.compProd η)
    ⊢ (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
  -/
  rw [ν.haveLebesgueDecomposition_add μ, compProd_add_left, add_comm] at hκη
  have h := absolutelyContinuous_of_add_of_mutuallySingular hκη
    ((mutuallySingular_singularPart _ _).symm.compProd_of_left _ _)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (HAdd.hAdd ((μ.withDensity (ν.rnDeri …
    h : (μ.compProd κ).AbsolutelyContinuous ((μ.withDensity (ν.rnDeriv μ)).compPro …
    ⊢ (μ.compProd κ).AbsolutelyContinuous (μ.compProd η)
  -/
  refine h.trans (AbsolutelyContinuous.compProd_left ?_ _)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    μ ν : MeasureTheory.Measure α
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hκη : (μ.compProd κ).AbsolutelyContinuous (HAdd.hAdd ((μ.withDensity (ν.rnDeri …
    h : (μ.compProd κ).AbsolutelyContinuous ((μ.withDensity (ν.rnDeriv μ)).compPro …
    ⊢ (μ.withDensity (ν.rnDeriv μ)).AbsolutelyContinuous μ
  -/
  exact withDensity_absolutelyContinuous _ _
  /-
    🎉 no goals
  -/


lemma absolutelyContinuous_compProd_iff
    [SigmaFinite μ] [SigmaFinite ν] [IsSFiniteKernel κ] [IsSFiniteKernel η] [∀ x, NeZero (κ x)] :
    μ ⊗ₘ κ ≪ ν ⊗ₘ η ↔ μ ≪ ν ∧ μ ⊗ₘ κ ≪ μ ⊗ₘ η :=
  ⟨fun h ↦ ⟨absolutelyContinuous_of_compProd h, absolutelyContinuous_compProd_of_compProd h⟩,
    fun h ↦ h.1.compProd_of_compProd h.2⟩


