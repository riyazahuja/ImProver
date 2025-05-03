theorem hasFiniteIntegral_prod_mk_left (a : α) {s : Set (β × γ)} (h2s : (κ ⊗ₖ η) a s ≠ ∞) :
    HasFiniteIntegral (fun b => (η (a, b) (Prod.mk b ⁻¹' s)).toReal) (κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    h2s : Ne (((κ.compProd η) a) s) Top.top
    ⊢ MeasureTheory.HasFiniteIntegral (fun b => ((η { fst := a, snd := b }) (Set.p …
  -/
  let t := toMeasurable ((κ ⊗ₖ η) a) s
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    h2s : Ne (((κ.compProd η) a) s) Top.top
    t : Set (Prod β γ) := MeasureTheory.toMeasurable ((κ.compProd η) a) s
    ⊢ MeasureTheory.HasFiniteIntegral (fun b => ((η { fst := a, snd := b }) (Set.p …
  -/
  simp_rw [hasFiniteIntegral_iff_nnnorm, ennnorm_eq_ofReal toReal_nonneg]
  calc
    ∫⁻ b, ENNReal.ofReal (η (a, b) (Prod.mk b ⁻¹' s)).toReal ∂κ a
    _ ≤ ∫⁻ b, η (a, b) (Prod.mk b ⁻¹' t) ∂κ a := by
      refine lintegral_mono_ae ?_
      filter_upwards [ae_kernel_lt_top a h2s] with b hb
      rw [ofReal_toReal hb.ne]
      exact measure_mono (preimage_mono (subset_toMeasurable _ _))
    _ ≤ (κ ⊗ₖ η) a t := le_compProd_apply _ _ _ _
    _ = (κ ⊗ₖ η) a s := measure_toMeasurable s
    _ < ⊤ := h2s.lt_top


theorem integrable_kernel_prod_mk_left (a : α) {s : Set (β × γ)} (hs : MeasurableSet s)
    (h2s : (κ ⊗ₖ η) a s ≠ ∞) : Integrable (fun b => (η (a, b) (Prod.mk b ⁻¹' s)).toReal) (κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    s : Set (Prod β γ)
    hs : MeasurableSet s
    h2s : Ne (((κ.compProd η) a) s) Top.top
    ⊢ MeasureTheory.Integrable (fun b => ((η { fst := a, snd := b }) (Set.preimage …
  -/
  constructor
    /-
      case left
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      s : Set (Prod β γ)
      hs : MeasurableSet s
      h2s : Ne (((κ.compProd η) a) s) Top.top
      ⊢ MeasureTheory.AEStronglyMeasurable (fun b => ((η { fst := a, snd := b }) (Se …
    -/
  · exact (measurable_kernel_prod_mk_left' hs a).ennreal_toReal.aestronglyMeasurable
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      s : Set (Prod β γ)
      hs : MeasurableSet s
      h2s : Ne (((κ.compProd η) a) s) Top.top
      ⊢ MeasureTheory.HasFiniteIntegral (fun b => ((η { fst := a, snd := b }) (Set.p …
    -/
  · exact hasFiniteIntegral_prod_mk_left a h2s
    /-
      🎉 no goals
    -/


theorem _root_.MeasureTheory.AEStronglyMeasurable.integral_kernel_compProd [NormedSpace ℝ E]
    ⦃f : β × γ → E⦄ (hf : AEStronglyMeasurable f ((κ ⊗ₖ η) a)) :
    AEStronglyMeasurable (fun x => ∫ y, f (x, y) ∂η (a, x)) (κ a) :=
  ⟨fun x => ∫ y, hf.mk f (x, y) ∂η (a, x), hf.stronglyMeasurable_mk.integral_kernel_prod_right'', by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f : Prod β γ → E
      hf : MeasureTheory.AEStronglyMeasurable f ((κ.compProd η) a)
      ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun x => MeasureTheory.integral (η {  …
    -/
    filter_upwards [ae_ae_of_ae_compProd hf.ae_eq_mk] with _ hx using integral_congr_ae hx⟩
    /-
      🎉 no goals
    -/


theorem _root_.MeasureTheory.AEStronglyMeasurable.compProd_mk_left {δ : Type*} [TopologicalSpace δ]
    {f : β × γ → δ} (hf : AEStronglyMeasurable f ((κ ⊗ₖ η) a)) :
    ∀ᵐ x ∂κ a, AEStronglyMeasurable (fun y => f (x, y)) (η (a, x)) := by
  filter_upwards [ae_ae_of_ae_compProd hf.ae_eq_mk] with x hx using
    ⟨fun y => hf.mk f (x, y), hf.stronglyMeasurable_mk.comp_measurable measurable_prod_mk_left, hx⟩


theorem hasFiniteIntegral_compProd_iff ⦃f : β × γ → E⦄ (h1f : StronglyMeasurable f) :
    HasFiniteIntegral f ((κ ⊗ₖ η) a) ↔
      (∀ᵐ x ∂κ a, HasFiniteIntegral (fun y => f (x, y)) (η (a, x))) ∧
        HasFiniteIntegral (fun x => ∫ y, ‖f (x, y)‖ ∂η (a, x)) (κ a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝² : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → E
    h1f : MeasureTheory.StronglyMeasurable f
    ⊢ Iff (MeasureTheory.HasFiniteIntegral f ((κ.compProd η) a)) (And (Filter.Even …
  -/
  simp only [hasFiniteIntegral_iff_nnnorm]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝² : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → E
    h1f : MeasureTheory.StronglyMeasurable f
    ⊢ Iff (LT.lt (MeasureTheory.lintegral ((κ.compProd η) a) fun a => ↑(NNNorm.nnn …
  -/
  rw [Kernel.lintegral_compProd _ _ _ h1f.ennnorm]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝² : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → E
    h1f : MeasureTheory.StronglyMeasurable f
    ⊢ Iff (LT.lt (MeasureTheory.lintegral (κ a) fun b => MeasureTheory.lintegral ( …
  -/
  have : ∀ x, ∀ᵐ y ∂η (a, x), 0 ≤ ‖f (x, y)‖ := fun x => Eventually.of_forall fun y => norm_nonneg _
  simp_rw [integral_eq_lintegral_of_nonneg_ae (this _)
      (h1f.norm.comp_measurable measurable_prod_mk_left).aestronglyMeasurable,
    ennnorm_eq_ofReal toReal_nonneg, ofReal_norm_eq_coe_nnnorm]
  have : ∀ {p q r : Prop} (_ : r → p), (r ↔ p ∧ q) ↔ p → (r ↔ q) := fun {p q r} h1 => by
    rw [← and_congr_right_iff, and_iff_right_of_imp h1]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝² : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → E
    h1f : MeasureTheory.StronglyMeasurable f
    this✝ : ∀ (x : β), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
    this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
    ⊢ Iff (LT.lt (MeasureTheory.lintegral (κ a) fun b => MeasureTheory.lintegral ( …
  -/
  rw [this]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝² : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Prod β γ → E
      h1f : MeasureTheory.StronglyMeasurable f
      this✝ : ∀ (x : β), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
      this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
      ⊢ Filter.Eventually (fun x => LT.lt (MeasureTheory.lintegral (η { fst := a, sn …
    -/
  · intro h2f; rw [lintegral_congr_ae]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝² : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Prod β γ → E
      h1f : MeasureTheory.StronglyMeasurable f
      this✝ : ∀ (x : β), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
      this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
      h2f : Filter.Eventually (fun x => LT.lt (MeasureTheory.lintegral (η { fst := a …
      ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun b => MeasureTheory.lintegral (η { …
    -/
    filter_upwards [h2f] with x hx
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝² : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Prod β γ → E
      h1f : MeasureTheory.StronglyMeasurable f
      this✝ : ∀ (x : β), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
      this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
      h2f : Filter.Eventually (fun x => LT.lt (MeasureTheory.lintegral (η { fst := a …
      x : β
      hx : LT.lt (MeasureTheory.lintegral (η { fst := a, snd := x }) fun a => ↑(NNNo …
      ⊢ Eq (MeasureTheory.lintegral (η { fst := a, snd := x }) fun c => ↑(NNNorm.nnn …
    -/
    rw [ofReal_toReal]; rw [← lt_top_iff_ne_top]; exact hx
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝² : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Prod β γ → E
      h1f : MeasureTheory.StronglyMeasurable f
      this✝ : ∀ (x : β), Filter.Eventually (fun y => LE.le 0 (Norm.norm (f { fst :=  …
      this : ∀ {p q r : Prop}, (r → p) → Iff (Iff r (And p q)) (p → Iff r q)
      ⊢ LT.lt (MeasureTheory.lintegral (κ a) fun b => MeasureTheory.lintegral (η { f …
    -/
  · intro h2f; refine ae_lt_top ?_ h2f.ne; exact h1f.ennnorm.lintegral_kernel_prod_right''
                                           /-
                                             🎉 no goals
                                           -/


theorem hasFiniteIntegral_compProd_iff' ⦃f : β × γ → E⦄
    (h1f : AEStronglyMeasurable f ((κ ⊗ₖ η) a)) :
    HasFiniteIntegral f ((κ ⊗ₖ η) a) ↔
      (∀ᵐ x ∂κ a, HasFiniteIntegral (fun y => f (x, y)) (η (a, x))) ∧
        HasFiniteIntegral (fun x => ∫ y, ‖f (x, y)‖ ∂η (a, x)) (κ a) := by
  rw [hasFiniteIntegral_congr h1f.ae_eq_mk,
    hasFiniteIntegral_compProd_iff h1f.stronglyMeasurable_mk]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝² : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    f : Prod β γ → E
    h1f : MeasureTheory.AEStronglyMeasurable f ((κ.compProd η) a)
    ⊢ Iff (And (Filter.Eventually (fun x => MeasureTheory.HasFiniteIntegral (fun y …
  -/
  apply and_congr
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝² : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Prod β γ → E
      h1f : MeasureTheory.AEStronglyMeasurable f ((κ.compProd η) a)
      ⊢ Iff (Filter.Eventually (fun x => MeasureTheory.HasFiniteIntegral (fun y => M …
    -/
  · apply eventually_congr
    filter_upwards [ae_ae_of_ae_compProd h1f.ae_eq_mk.symm] with x hx using
      hasFiniteIntegral_congr hx
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝² : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      f : Prod β γ → E
      h1f : MeasureTheory.AEStronglyMeasurable f ((κ.compProd η) a)
      ⊢ Iff (MeasureTheory.HasFiniteIntegral (fun x => MeasureTheory.integral (η { f …
    -/
  · apply hasFiniteIntegral_congr
    filter_upwards [ae_ae_of_ae_compProd h1f.ae_eq_mk.symm] with _ hx using
      integral_congr_ae (EventuallyEq.fun_comp hx _)


theorem integrable_compProd_iff ⦃f : β × γ → E⦄ (hf : AEStronglyMeasurable f ((κ ⊗ₖ η) a)) :
    Integrable f ((κ ⊗ₖ η) a) ↔
      (∀ᵐ x ∂κ a, Integrable (fun y => f (x, y)) (η (a, x))) ∧
        Integrable (fun x => ∫ y, ‖f (x, y)‖ ∂η (a, x)) (κ a) := by
  simp only [Integrable, hasFiniteIntegral_compProd_iff' hf, hf.norm.integral_kernel_compProd,
    hf, hf.compProd_mk_left, eventually_and, true_and]


theorem _root_.MeasureTheory.Integrable.compProd_mk_left_ae ⦃f : β × γ → E⦄
    (hf : Integrable f ((κ ⊗ₖ η) a)) : ∀ᵐ x ∂κ a, Integrable (fun y => f (x, y)) (η (a, x)) :=
  ((integrable_compProd_iff hf.aestronglyMeasurable).mp hf).1


theorem _root_.MeasureTheory.Integrable.integral_norm_compProd ⦃f : β × γ → E⦄
    (hf : Integrable f ((κ ⊗ₖ η) a)) : Integrable (fun x => ∫ y, ‖f (x, y)‖ ∂η (a, x)) (κ a) :=
  ((integrable_compProd_iff hf.aestronglyMeasurable).mp hf).2


theorem _root_.MeasureTheory.Integrable.integral_compProd [NormedSpace ℝ E]
    ⦃f : β × γ → E⦄ (hf : Integrable f ((κ ⊗ₖ η) a)) :
    Integrable (fun x => ∫ y, f (x, y) ∂η (a, x)) (κ a) :=
  Integrable.mono hf.integral_norm_compProd hf.aestronglyMeasurable.integral_kernel_compProd <|
    Eventually.of_forall fun x =>
      (norm_integral_le_integral_norm _).trans_eq <|
        (norm_of_nonneg <|
            integral_nonneg_of_ae <|
              Eventually.of_forall fun y => (norm_nonneg (f (x, y)) : _)).symm


theorem Kernel.integral_fn_integral_add ⦃f g : β × γ → E⦄ (F : E → E')
    (hf : Integrable f ((κ ⊗ₖ η) a)) (hg : Integrable g ((κ ⊗ₖ η) a)) :
    ∫ x, F (∫ y, f (x, y) + g (x, y) ∂η (a, x)) ∂κ a =
      ∫ x, F (∫ y, f (x, y) ∂η (a, x) + ∫ y, g (x, y) ∂η (a, x)) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁵ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝⁴ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝³ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝² : NormedSpace Real E
    E' : Type u_5
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod β γ → E
    F : E → E'
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    ⊢ Eq (MeasureTheory.integral (κ a) fun x => F (MeasureTheory.integral (η { fst …
  -/
  refine integral_congr_ae ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁵ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝⁴ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝³ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝² : NormedSpace Real E
    E' : Type u_5
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod β γ → E
    F : E → E'
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun x => F (MeasureTheory.integral (η …
  -/
  filter_upwards [hf.compProd_mk_left_ae, hg.compProd_mk_left_ae] with _ h2f h2g
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁵ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝⁴ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝³ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝² : NormedSpace Real E
    E' : Type u_5
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod β γ → E
    F : E → E'
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    a✝ : β
    h2f : MeasureTheory.Integrable (fun y => f { fst := a✝, snd := y }) (η { fst : …
    h2g : MeasureTheory.Integrable (fun y => g { fst := a✝, snd := y }) (η { fst : …
    ⊢ Eq (F (MeasureTheory.integral (η { fst := a, snd := a✝ }) fun y => HAdd.hAdd …
  -/
  simp [integral_add h2f h2g]
  /-
    🎉 no goals
  -/


theorem Kernel.integral_fn_integral_sub ⦃f g : β × γ → E⦄ (F : E → E')
    (hf : Integrable f ((κ ⊗ₖ η) a)) (hg : Integrable g ((κ ⊗ₖ η) a)) :
    ∫ x, F (∫ y, f (x, y) - g (x, y) ∂η (a, x)) ∂κ a =
      ∫ x, F (∫ y, f (x, y) ∂η (a, x) - ∫ y, g (x, y) ∂η (a, x)) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁵ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝⁴ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝³ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝² : NormedSpace Real E
    E' : Type u_5
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod β γ → E
    F : E → E'
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    ⊢ Eq (MeasureTheory.integral (κ a) fun x => F (MeasureTheory.integral (η { fst …
  -/
  refine integral_congr_ae ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁵ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝⁴ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝³ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝² : NormedSpace Real E
    E' : Type u_5
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod β γ → E
    F : E → E'
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun x => F (MeasureTheory.integral (η …
  -/
  filter_upwards [hf.compProd_mk_left_ae, hg.compProd_mk_left_ae] with _ h2f h2g
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁵ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝⁴ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝³ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝² : NormedSpace Real E
    E' : Type u_5
    inst✝¹ : NormedAddCommGroup E'
    inst✝ : NormedSpace Real E'
    f g : Prod β γ → E
    F : E → E'
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    a✝ : β
    h2f : MeasureTheory.Integrable (fun y => f { fst := a✝, snd := y }) (η { fst : …
    h2g : MeasureTheory.Integrable (fun y => g { fst := a✝, snd := y }) (η { fst : …
    ⊢ Eq (F (MeasureTheory.integral (η { fst := a, snd := a✝ }) fun y => HSub.hSub …
  -/
  simp [integral_sub h2f h2g]
  /-
    🎉 no goals
  -/


theorem Kernel.lintegral_fn_integral_sub ⦃f g : β × γ → E⦄ (F : E → ℝ≥0∞)
    (hf : Integrable f ((κ ⊗ₖ η) a)) (hg : Integrable g ((κ ⊗ₖ η) a)) :
    ∫⁻ x, F (∫ y, f (x, y) - g (x, y) ∂η (a, x)) ∂κ a =
      ∫⁻ x, F (∫ y, f (x, y) ∂η (a, x) - ∫ y, g (x, y) ∂η (a, x)) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    f g : Prod β γ → E
    F : E → ENNReal
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun x => F (MeasureTheory.integral (η { fs …
  -/
  refine lintegral_congr_ae ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    f g : Prod β γ → E
    F : E → ENNReal
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (fun x => F (MeasureTheory.integral (η …
  -/
  filter_upwards [hf.compProd_mk_left_ae, hg.compProd_mk_left_ae] with _ h2f h2g
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    f g : Prod β γ → E
    F : E → ENNReal
    hf : MeasureTheory.Integrable f ((κ.compProd η) a)
    hg : MeasureTheory.Integrable g ((κ.compProd η) a)
    a✝ : β
    h2f : MeasureTheory.Integrable (fun y => f { fst := a✝, snd := y }) (η { fst : …
    h2g : MeasureTheory.Integrable (fun y => g { fst := a✝, snd := y }) (η { fst : …
    ⊢ Eq (F (MeasureTheory.integral (η { fst := a, snd := a✝ }) fun y => HSub.hSub …
  -/
  simp [integral_sub h2f h2g]
  /-
    🎉 no goals
  -/


theorem Kernel.integral_integral_add ⦃f g : β × γ → E⦄ (hf : Integrable f ((κ ⊗ₖ η) a))
    (hg : Integrable g ((κ ⊗ₖ η) a)) :
    ∫ x, ∫ y, f (x, y) + g (x, y) ∂η (a, x) ∂κ a =
      ∫ x, ∫ y, f (x, y) ∂η (a, x) ∂κ a + ∫ x, ∫ y, g (x, y) ∂η (a, x) ∂κ a :=
  (Kernel.integral_fn_integral_add id hf hg).trans <|
    integral_add hf.integral_compProd hg.integral_compProd


theorem Kernel.integral_integral_add' ⦃f g : β × γ → E⦄ (hf : Integrable f ((κ ⊗ₖ η) a))
    (hg : Integrable g ((κ ⊗ₖ η) a)) :
    ∫ x, ∫ y, (f + g) (x, y) ∂η (a, x) ∂κ a =
      ∫ x, ∫ y, f (x, y) ∂η (a, x) ∂κ a + ∫ x, ∫ y, g (x, y) ∂η (a, x) ∂κ a :=
  Kernel.integral_integral_add hf hg


theorem Kernel.integral_integral_sub ⦃f g : β × γ → E⦄ (hf : Integrable f ((κ ⊗ₖ η) a))
    (hg : Integrable g ((κ ⊗ₖ η) a)) :
    ∫ x, ∫ y, f (x, y) - g (x, y) ∂η (a, x) ∂κ a =
      ∫ x, ∫ y, f (x, y) ∂η (a, x) ∂κ a - ∫ x, ∫ y, g (x, y) ∂η (a, x) ∂κ a :=
  (Kernel.integral_fn_integral_sub id hf hg).trans <|
    integral_sub hf.integral_compProd hg.integral_compProd


theorem Kernel.integral_integral_sub' ⦃f g : β × γ → E⦄ (hf : Integrable f ((κ ⊗ₖ η) a))
    (hg : Integrable g ((κ ⊗ₖ η) a)) :
    ∫ x, ∫ y, (f - g) (x, y) ∂η (a, x) ∂κ a =
      ∫ x, ∫ y, f (x, y) ∂η (a, x) ∂κ a - ∫ x, ∫ y, g (x, y) ∂η (a, x) ∂κ a :=
  Kernel.integral_integral_sub hf hg

-- Porting note: couldn't get the `→₁[]` syntax to work

theorem Kernel.continuous_integral_integral :
    -- Continuous fun f : α × β →₁[(κ ⊗ₖ η) a] E => ∫ x, ∫ y, f (x, y) ∂η (a, x) ∂κ a := by
    Continuous fun f : (MeasureTheory.Lp (α := β × γ) E 1 (((κ ⊗ₖ η) a) : Measure (β × γ))) =>
        ∫ x, ∫ y, f (x, y) ∂η (a, x) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    ⊢ Continuous fun f => MeasureTheory.integral (κ a) fun x => MeasureTheory.inte …
  -/
  rw [continuous_iff_continuousAt]; intro g
  refine
    tendsto_integral_of_L1 _ (L1.integrable_coeFn g).integral_compProd
      (Eventually.of_forall fun h => (L1.integrable_coeFn h).integral_compProd) ?_
  simp_rw [←
    Kernel.lintegral_fn_integral_sub (fun x => (‖x‖₊ : ℝ≥0∞)) (L1.integrable_coeFn _)
      (L1.integrable_coeFn g)]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compProd η) a)) x
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral (κ a) fun x => ↑(NNNorm.nnn …
  -/
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds _ (fun i => zero_le _) _
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compProd η) a)) x
      ⊢ (Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compProd η) a)) x …
    -/
  · exact fun i => ∫⁻ x, ∫⁻ y, ‖i (x, y) - g (x, y)‖₊ ∂η (a, x) ∂κ a
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compProd η) a)) x
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral (κ a) fun x => MeasureTheor …
  -/
  swap; · exact fun i => lintegral_mono fun x => ennnorm_integral_le_lintegral_ennnorm _
          /-
            🎉 no goals
          -/
  show
    Tendsto
      (fun i : β × γ →₁[(κ ⊗ₖ η) a] E => ∫⁻ x, ∫⁻ y : γ, ‖i (x, y) - g (x, y)‖₊ ∂η (a, x) ∂κ a)
      (𝓝 g) (𝓝 0)
  have : ∀ i : (MeasureTheory.Lp (α := β × γ) E 1 (((κ ⊗ₖ η) a) : Measure (β × γ))),
      Measurable fun z => (‖i z - g z‖₊ : ℝ≥0∞) := fun i =>
    ((Lp.stronglyMeasurable i).sub (Lp.stronglyMeasurable g)).ennnorm
  simp_rw [← Kernel.lintegral_compProd _ _ _ (this _), ← L1.ofReal_norm_sub_eq_lintegral, ←
    ofReal_zero]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compProd η) a)) x
    this : ∀ (i : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compPr …
    ⊢ Filter.Tendsto (fun i => ENNReal.ofReal (Norm.norm (HSub.hSub i g))) (nhds g …
  -/
  refine (continuous_ofReal.tendsto 0).comp ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compProd η) a)) x
    this : ∀ (i : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compPr …
    ⊢ Filter.Tendsto (fun i => Norm.norm (HSub.hSub i g)) (nhds g) (nhds 0)
  -/
  rw [← tendsto_iff_norm_sub_tendsto_zero]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compProd η) a)) x
    this : ∀ (i : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 ((κ.compPr …
    ⊢ Filter.Tendsto (fun i => i) (nhds g) (nhds g)
  -/
  exact tendsto_id
  /-
    🎉 no goals
  -/


theorem integral_compProd :
    ∀ {f : β × γ → E} (_ : Integrable f ((κ ⊗ₖ η) a)),
      ∫ z, f z ∂(κ ⊗ₖ η) a = ∫ x, ∫ y, f (x, y) ∂η (a, x) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    ⊢ ∀ {f : Prod β γ → E}, MeasureTheory.Integrable f ((κ.compProd η) a) → Eq (Me …
  -/
  by_cases hE : CompleteSpace E; swap
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : Not (CompleteSpace E)
      ⊢ MeasureTheory.Integrable f✝ ((κ.compProd η) a) → Eq (MeasureTheory.integral  …
    -/
  · simp [integral, hE]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    f✝ : Prod β γ → E
    hE : CompleteSpace E
    ⊢ MeasureTheory.Integrable f✝ ((κ.compProd η) a) → Eq (MeasureTheory.integral  …
  -/
  apply Integrable.induction
    /-
      case pos.h_ind
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      ⊢ ∀ (c : E) ⦃s : Set (Prod β γ)⦄, MeasurableSet s → LT.lt (((κ.compProd η) a)  …
    -/
  · intro c s hs h2s
    simp_rw [integral_indicator hs, ← indicator_comp_right, Function.comp_def,
      integral_indicator (measurable_prod_mk_left hs), MeasureTheory.setIntegral_const,
      integral_smul_const]
    /-
      case pos.h_ind
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      c : E
      s : Set (Prod β γ)
      hs : MeasurableSet s
      h2s : LT.lt (((κ.compProd η) a) s) Top.top
      ⊢ Eq (HSMul.hSMul (((κ.compProd η) a) s).toReal c) (HSMul.hSMul (MeasureTheory …
    -/
    congr 1
    /-
      case pos.h_ind.e_a
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      c : E
      s : Set (Prod β γ)
      hs : MeasurableSet s
      h2s : LT.lt (((κ.compProd η) a) s) Top.top
      ⊢ Eq (((κ.compProd η) a) s).toReal (MeasureTheory.integral (κ a) fun x => ((η  …
    -/
    rw [integral_toReal]
    /-
      case pos.h_ind.e_a
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      c : E
      s : Set (Prod β γ)
      hs : MeasurableSet s
      h2s : LT.lt (((κ.compProd η) a) s) Top.top
      ⊢ Eq (((κ.compProd η) a) s).toReal (MeasureTheory.lintegral (κ a) fun a_1 => ( …
    -/
    rotate_left
      /-
        case pos.h_ind.e_a.hfm
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        E : Type u_4
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝³ : NormedAddCommGroup E
        κ : ProbabilityTheory.Kernel α β
        inst✝² : ProbabilityTheory.IsSFiniteKernel κ
        η : ProbabilityTheory.Kernel (Prod α β) γ
        inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
        a : α
        inst✝ : NormedSpace Real E
        f✝ : Prod β γ → E
        hE : CompleteSpace E
        c : E
        s : Set (Prod β γ)
        hs : MeasurableSet s
        h2s : LT.lt (((κ.compProd η) a) s) Top.top
        ⊢ AEMeasurable (fun x => (η { fst := a, snd := x }) (Set.preimage (Prod.mk x)  …
      -/
    · exact (Kernel.measurable_kernel_prod_mk_left' hs _).aemeasurable
      /-
        🎉 no goals
      -/
      /-
        case pos.h_ind.e_a.hf
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        E : Type u_4
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝³ : NormedAddCommGroup E
        κ : ProbabilityTheory.Kernel α β
        inst✝² : ProbabilityTheory.IsSFiniteKernel κ
        η : ProbabilityTheory.Kernel (Prod α β) γ
        inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
        a : α
        inst✝ : NormedSpace Real E
        f✝ : Prod β γ → E
        hE : CompleteSpace E
        c : E
        s : Set (Prod β γ)
        hs : MeasurableSet s
        h2s : LT.lt (((κ.compProd η) a) s) Top.top
        ⊢ Filter.Eventually (fun x => LT.lt ((η { fst := a, snd := x }) (Set.preimage  …
      -/
    · exact ae_kernel_lt_top a h2s.ne
      /-
        🎉 no goals
      -/
    /-
      case pos.h_ind.e_a
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      c : E
      s : Set (Prod β γ)
      hs : MeasurableSet s
      h2s : LT.lt (((κ.compProd η) a) s) Top.top
      ⊢ Eq (((κ.compProd η) a) s).toReal (MeasureTheory.lintegral (κ a) fun a_1 => ( …
    -/
    rw [Kernel.compProd_apply hs]
    /-
      case pos.h_ind.e_a
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      c : E
      s : Set (Prod β γ)
      hs : MeasurableSet s
      h2s : LT.lt (((κ.compProd η) a) s) Top.top
      ⊢ Eq (MeasureTheory.lintegral (κ a) fun b => (η { fst := a, snd := b }) (setOf …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case pos.h_add
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      ⊢ ∀ ⦃f g : Prod β γ → E⦄, Disjoint (Function.support f) (Function.support g) → …
    -/
  · intro f g _ i_f i_g hf hg
    /-
      case pos.h_add
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      f g : Prod β γ → E
      a✝ : Disjoint (Function.support f) (Function.support g)
      i_f : MeasureTheory.Integrable f ((κ.compProd η) a)
      i_g : MeasureTheory.Integrable g ((κ.compProd η) a)
      hf : Eq (MeasureTheory.integral ((κ.compProd η) a) fun z => f z) (MeasureTheor …
      hg : Eq (MeasureTheory.integral ((κ.compProd η) a) fun z => g z) (MeasureTheor …
      ⊢ Eq (MeasureTheory.integral ((κ.compProd η) a) fun z => HAdd.hAdd f g z) (Mea …
    -/
    simp_rw [integral_add' i_f i_g, Kernel.integral_integral_add' i_f i_g, hf, hg]
    /-
      🎉 no goals
    -/
    /-
      case pos.h_closed
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      ⊢ IsClosed (setOf fun f => Eq (MeasureTheory.integral ((κ.compProd η) a) fun z …
    -/
  · exact isClosed_eq continuous_integral Kernel.continuous_integral_integral
    /-
      🎉 no goals
    -/
    /-
      case pos.h_ae
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      ⊢ ∀ ⦃f g : Prod β γ → E⦄, (MeasureTheory.ae ((κ.compProd η) a)).EventuallyEq f …
    -/
  · intro f g hfg _ hf
    /-
      case pos.h_ae
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f✝ : Prod β γ → E
      hE : CompleteSpace E
      f g : Prod β γ → E
      hfg : (MeasureTheory.ae ((κ.compProd η) a)).EventuallyEq f g
      a✝ : MeasureTheory.Integrable f ((κ.compProd η) a)
      hf : Eq (MeasureTheory.integral ((κ.compProd η) a) fun z => f z) (MeasureTheor …
      ⊢ Eq (MeasureTheory.integral ((κ.compProd η) a) fun z => g z) (MeasureTheory.i …
    -/
    convert hf using 1
      /-
        case h.e'_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        E : Type u_4
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝³ : NormedAddCommGroup E
        κ : ProbabilityTheory.Kernel α β
        inst✝² : ProbabilityTheory.IsSFiniteKernel κ
        η : ProbabilityTheory.Kernel (Prod α β) γ
        inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
        a : α
        inst✝ : NormedSpace Real E
        f✝ : Prod β γ → E
        hE : CompleteSpace E
        f g : Prod β γ → E
        hfg : (MeasureTheory.ae ((κ.compProd η) a)).EventuallyEq f g
        a✝ : MeasureTheory.Integrable f ((κ.compProd η) a)
        hf : Eq (MeasureTheory.integral ((κ.compProd η) a) fun z => f z) (MeasureTheor …
        ⊢ Eq (MeasureTheory.integral ((κ.compProd η) a) fun z => g z) (MeasureTheory.i …
      -/
    · exact integral_congr_ae hfg.symm
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        E : Type u_4
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        mγ : MeasurableSpace γ
        inst✝³ : NormedAddCommGroup E
        κ : ProbabilityTheory.Kernel α β
        inst✝² : ProbabilityTheory.IsSFiniteKernel κ
        η : ProbabilityTheory.Kernel (Prod α β) γ
        inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
        a : α
        inst✝ : NormedSpace Real E
        f✝ : Prod β γ → E
        hE : CompleteSpace E
        f g : Prod β γ → E
        hfg : (MeasureTheory.ae ((κ.compProd η) a)).EventuallyEq f g
        a✝ : MeasureTheory.Integrable f ((κ.compProd η) a)
        hf : Eq (MeasureTheory.integral ((κ.compProd η) a) fun z => f z) (MeasureTheor …
        ⊢ Eq (MeasureTheory.integral (κ a) fun x => MeasureTheory.integral (η { fst := …
      -/
    · apply integral_congr_ae
      filter_upwards [ae_ae_of_ae_compProd hfg] with x hfgx using
        integral_congr_ae (ae_eq_symm hfgx)


theorem setIntegral_compProd {f : β × γ → E} {s : Set β} {t : Set γ} (hs : MeasurableSet s)
    (ht : MeasurableSet t) (hf : IntegrableOn f (s ×ˢ t) ((κ ⊗ₖ η) a)) :
    ∫ z in s ×ˢ t, f z ∂(κ ⊗ₖ η) a = ∫ x in s, ∫ y in t, f (x, y) ∂η (a, x) ∂κ a := by
  -- Porting note: `compProd_restrict` needed some explicit arguments
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    f : Prod β γ → E
    s : Set β
    t : Set γ
    hs : MeasurableSet s
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) ((κ.compProd η) a)
    ⊢ Eq (MeasureTheory.integral (((κ.compProd η) a).restrict (SProd.sprod s t)) f …
  -/
  rw [← Kernel.restrict_apply (κ ⊗ₖ η) (hs.prod ht), ← compProd_restrict hs ht, integral_compProd]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f : Prod β γ → E
      s : Set β
      t : Set γ
      hs : MeasurableSet s
      ht : MeasurableSet t
      hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) ((κ.compProd η) a)
      ⊢ Eq (MeasureTheory.integral ((κ.restrict hs) a) fun x => MeasureTheory.integr …
    -/
  · simp_rw [Kernel.restrict_apply]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      E : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝³ : NormedAddCommGroup E
      κ : ProbabilityTheory.Kernel α β
      inst✝² : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) γ
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
      a : α
      inst✝ : NormedSpace Real E
      f : Prod β γ → E
      s : Set β
      t : Set γ
      hs : MeasurableSet s
      ht : MeasurableSet t
      hf : MeasureTheory.IntegrableOn f (SProd.sprod s t) ((κ.compProd η) a)
      ⊢ MeasureTheory.Integrable f (((κ.restrict hs).compProd (η.restrict ht)) a)
    -/
  · rw [compProd_restrict, Kernel.restrict_apply]; exact hf
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-04-17")]
alias set_integral_compProd := setIntegral_compProd


theorem setIntegral_compProd_univ_right (f : β × γ → E) {s : Set β} (hs : MeasurableSet s)
    (hf : IntegrableOn f (s ×ˢ univ) ((κ ⊗ₖ η) a)) :
    ∫ z in s ×ˢ univ, f z ∂(κ ⊗ₖ η) a = ∫ x in s, ∫ y, f (x, y) ∂η (a, x) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    f : Prod β γ → E
    s : Set β
    hs : MeasurableSet s
    hf : MeasureTheory.IntegrableOn f (SProd.sprod s Set.univ) ((κ.compProd η) a)
    ⊢ Eq (MeasureTheory.integral (((κ.compProd η) a).restrict (SProd.sprod s Set.u …
  -/
  simp_rw [setIntegral_compProd hs MeasurableSet.univ hf, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_compProd_univ_right := setIntegral_compProd_univ_right


theorem setIntegral_compProd_univ_left (f : β × γ → E) {t : Set γ} (ht : MeasurableSet t)
    (hf : IntegrableOn f (univ ×ˢ t) ((κ ⊗ₖ η) a)) :
    ∫ z in univ ×ˢ t, f z ∂(κ ⊗ₖ η) a = ∫ x, ∫ y in t, f (x, y) ∂η (a, x) ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    E : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝³ : NormedAddCommGroup E
    κ : ProbabilityTheory.Kernel α β
    inst✝² : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) γ
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel η
    a : α
    inst✝ : NormedSpace Real E
    f : Prod β γ → E
    t : Set γ
    ht : MeasurableSet t
    hf : MeasureTheory.IntegrableOn f (SProd.sprod Set.univ t) ((κ.compProd η) a)
    ⊢ Eq (MeasureTheory.integral (((κ.compProd η) a).restrict (SProd.sprod Set.uni …
  -/
  simp_rw [setIntegral_compProd MeasurableSet.univ ht hf, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_compProd_univ_left := setIntegral_compProd_univ_left


