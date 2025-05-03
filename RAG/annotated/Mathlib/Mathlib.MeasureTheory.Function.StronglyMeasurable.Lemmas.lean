theorem MeasureTheory.AEStronglyMeasurable.comp_measurePreserving
    {γ : Type*} {_ : MeasurableSpace γ} {_ : MeasurableSpace α} {f : γ → α} {μ : Measure γ}
    {ν : Measure α} (hg : AEStronglyMeasurable g ν) (hf : MeasurePreserving f μ ν) :
    AEStronglyMeasurable (g ∘ f) μ :=
  hg.comp_quasiMeasurePreserving hf.quasiMeasurePreserving


theorem MeasureTheory.MeasurePreserving.aestronglyMeasurable_comp_iff {β : Type*}
    {f : α → β} {mα : MeasurableSpace α} {μa : Measure α} {mβ : MeasurableSpace β} {μb : Measure β}
    (hf : MeasurePreserving f μa μb) (h₂ : MeasurableEmbedding f) {g : β → γ} :
    AEStronglyMeasurable (g ∘ f) μa ↔ AEStronglyMeasurable g μb := by
  /-
    α : Type u_1
    γ : Type u_3
    inst✝ : TopologicalSpace γ
    β : Type u_4
    f : α → β
    mα : MeasurableSpace α
    μa : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    μb : MeasureTheory.Measure β
    hf : MeasureTheory.MeasurePreserving f μa μb
    h₂ : MeasurableEmbedding f
    g : β → γ
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable (Function.comp g f) μa) (MeasureTheo …
  -/
  rw [← hf.map_eq, h₂.aestronglyMeasurable_map_iff]
  /-
    🎉 no goals
  -/


theorem aestronglyMeasurable_smul_const_iff {f : α → 𝕜} {c : E} (hc : c ≠ 0) :
    AEStronglyMeasurable (fun x => f x • c) μ ↔ AEStronglyMeasurable f μ :=
  (isClosedEmbedding_smul_left hc).isEmbedding.aestronglyMeasurable_comp_iff


theorem StronglyMeasurable.apply_continuousLinearMap
    {_m : MeasurableSpace α} {φ : α → F →L[𝕜] E} (hφ : StronglyMeasurable φ) (v : F) :
    StronglyMeasurable fun a => φ a v :=
  (ContinuousLinearMap.apply 𝕜 E v).continuous.comp_stronglyMeasurable hφ


@[measurability]
theorem MeasureTheory.AEStronglyMeasurable.apply_continuousLinearMap {φ : α → F →L[𝕜] E}
    (hφ : AEStronglyMeasurable φ μ) (v : F) :
    AEStronglyMeasurable (fun a => φ a v) μ :=
  (ContinuousLinearMap.apply 𝕜 E v).continuous.comp_aestronglyMeasurable hφ


theorem ContinuousLinearMap.aestronglyMeasurable_comp₂ (L : E →L[𝕜] F →L[𝕜] G) {f : α → E}
    {g : α → F} (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    AEStronglyMeasurable (fun x => L (f x) (g x)) μ :=
  L.continuous₂.comp_aestronglyMeasurable₂ hf hg


theorem aestronglyMeasurable_withDensity_iff {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] {f : α → ℝ≥0} (hf : Measurable f) {g : α → E} :
    AEStronglyMeasurable g (μ.withDensity fun x => (f x : ℝ≥0∞)) ↔
      AEStronglyMeasurable (fun x => (f x : ℝ) • g x) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → NNReal
    hf : Measurable f
    g : α → E
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable g (μ.withDensity fun x => ↑(f x))) ( …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g : α → E
      ⊢ MeasureTheory.AEStronglyMeasurable g (μ.withDensity fun x => ↑(f x)) → Measu …
    -/
  · rintro ⟨g', g'meas, hg'⟩
    /-
      case mp.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
    -/
    have A : MeasurableSet { x : α | f x ≠ 0 } := (hf (measurableSet_singleton 0)).compl
    /-
      case mp.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
      A : MeasurableSet (setOf fun x => Ne (f x) 0)
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
    -/
    refine ⟨fun x => (f x : ℝ) • g' x, hf.coe_nnreal_real.stronglyMeasurable.smul g'meas, ?_⟩
    /-
      case mp.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
      A : MeasurableSet (setOf fun x => Ne (f x) 0)
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) fun  …
    -/
    apply @ae_of_ae_restrict_of_ae_restrict_compl _ _ _ { x | f x ≠ 0 }
      /-
        case mp.intro.intro.ht
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : MeasureTheory.StronglyMeasurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        ⊢ Filter.Eventually (fun x => Eq ((fun x => HSMul.hSMul (↑(f x)) (g x)) x) ((f …
      -/
    · rw [EventuallyEq, ae_withDensity_iff hf.coe_nnreal_ennreal] at hg'
      /-
        case mp.intro.intro.ht
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : MeasureTheory.StronglyMeasurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        ⊢ Filter.Eventually (fun x => Eq ((fun x => HSMul.hSMul (↑(f x)) (g x)) x) ((f …
      -/
      rw [ae_restrict_iff' A]
      /-
        case mp.intro.intro.ht
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : MeasureTheory.StronglyMeasurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        ⊢ Filter.Eventually (fun x => Membership.mem (setOf fun x => Ne (f x) 0) x → E …
      -/
      filter_upwards [hg'] with a ha h'a
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : MeasureTheory.StronglyMeasurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        a : α
        ha : Ne (↑(f a)) 0 → Eq (g a) (g' a)
        h'a : Ne (f a) 0
        ⊢ Eq (HSMul.hSMul (↑(f a)) (g a)) (HSMul.hSMul (↑(f a)) (g' a))
      -/
      have : (f a : ℝ≥0∞) ≠ 0 := by simpa only [Ne, ENNReal.coe_eq_zero] using h'a
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : MeasureTheory.StronglyMeasurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        a : α
        ha : Ne (↑(f a)) 0 → Eq (g a) (g' a)
        h'a : Ne (f a) 0
        this : Ne (↑(f a)) 0
        ⊢ Eq (HSMul.hSMul (↑(f a)) (g a)) (HSMul.hSMul (↑(f a)) (g' a))
      -/
      rw [ha this]
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.htc
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : MeasureTheory.StronglyMeasurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        ⊢ Filter.Eventually (fun x => Eq ((fun x => HSMul.hSMul (↑(f x)) (g x)) x) ((f …
      -/
    · filter_upwards [ae_restrict_mem A.compl] with x hx
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : MeasureTheory.StronglyMeasurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        x : α
        hx : Membership.mem (HasCompl.compl (setOf fun x => Ne (f x) 0)) x
        ⊢ Eq (HSMul.hSMul (↑(f x)) (g x)) (HSMul.hSMul (↑(f x)) (g' x))
      -/
      simp only [Classical.not_not, mem_setOf_eq, mem_compl_iff] at hx
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_4
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : MeasureTheory.StronglyMeasurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        x : α
        hx : Eq (f x) 0
        ⊢ Eq (HSMul.hSMul (↑(f x)) (g x)) (HSMul.hSMul (↑(f x)) (g' x))
      -/
      simp [hx]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g : α → E
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ → …
    -/
  · rintro ⟨g', g'meas, hg'⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      ⊢ MeasureTheory.AEStronglyMeasurable g (μ.withDensity fun x => ↑(f x))
    -/
    refine ⟨fun x => (f x : ℝ)⁻¹ • g' x, hf.coe_nnreal_real.inv.stronglyMeasurable.smul g'meas, ?_⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      ⊢ (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g fun x => H …
    -/
    rw [EventuallyEq, ae_withDensity_iff hf.coe_nnreal_ennreal]
    /-
      case mpr.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      ⊢ Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (HSMul.hSMul (Inv.inv ↑ …
    -/
    filter_upwards [hg'] with x hx h'x
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      x : α
      hx : Eq (HSMul.hSMul (↑(f x)) (g x)) (g' x)
      h'x : Ne (↑(f x)) 0
      ⊢ Eq (g x) (HSMul.hSMul (Inv.inv ↑(f x)) (g' x))
    -/
    rw [← hx, smul_smul, inv_mul_cancel₀, one_smul]
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      x : α
      hx : Eq (HSMul.hSMul (↑(f x)) (g x)) (g' x)
      h'x : Ne (↑(f x)) 0
      ⊢ Ne (↑(f x)) 0
    -/
    simp only [Ne, ENNReal.coe_eq_zero] at h'x
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_4
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : MeasureTheory.StronglyMeasurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      x : α
      hx : Eq (HSMul.hSMul (↑(f x)) (g x)) (g' x)
      h'x : Not (Eq (f x) 0)
      ⊢ Ne (↑(f x)) 0
    -/
    simpa only [NNReal.coe_eq_zero, Ne] using h'x
    /-
      🎉 no goals
    -/

