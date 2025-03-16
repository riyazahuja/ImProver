theorem aemeasurable_withDensity_iff {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [SecondCountableTopology E] [MeasurableSpace E] [BorelSpace E] {f : α → ℝ≥0}
    (hf : Measurable f) {g : α → E} :
    AEMeasurable g (μ.withDensity fun x => (f x : ℝ≥0∞)) ↔
      AEMeasurable (fun x => (f x : ℝ) • g x) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : SecondCountableTopology E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    f : α → NNReal
    hf : Measurable f
    g : α → E
    ⊢ Iff (AEMeasurable g (μ.withDensity fun x => ↑(f x))) (AEMeasurable (fun x => …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g : α → E
      ⊢ AEMeasurable g (μ.withDensity fun x => ↑(f x)) → AEMeasurable (fun x => HSMu …
    -/
  · rintro ⟨g', g'meas, hg'⟩
    /-
      case mp.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
      ⊢ AEMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
    -/
    have A : MeasurableSet { x : α | f x ≠ 0 } := (hf (measurableSet_singleton 0)).compl
    /-
      case mp.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
      A : MeasurableSet (setOf fun x => Ne (f x) 0)
      ⊢ AEMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
    -/
    refine ⟨fun x => (f x : ℝ) • g' x, hf.coe_nnreal_real.smul g'meas, ?_⟩
    /-
      case mp.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
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
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
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
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
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
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        ⊢ Filter.Eventually (fun x => Membership.mem (setOf fun x => Ne (f x) 0) x → E …
      -/
      filter_upwards [hg']
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
        hg' : Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (g' x)) (MeasureThe …
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        ⊢ ∀ (a : α), (Ne (↑(f a)) 0 → Eq (g a) (g' a)) → Ne (f a) 0 → Eq (HSMul.hSMul  …
      -/
      intro a ha h'a
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
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
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
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
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        ⊢ Filter.Eventually (fun x => Eq ((fun x => HSMul.hSMul (↑(f x)) (g x)) x) ((f …
      -/
    · filter_upwards [ae_restrict_mem A.compl]
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
        hg' : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g g'
        A : MeasurableSet (setOf fun x => Ne (f x) 0)
        ⊢ ∀ (a : α), Membership.mem (HasCompl.compl (setOf fun x => Ne (f x) 0)) a → E …
      -/
      intro x hx
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
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
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : SecondCountableTopology E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        f : α → NNReal
        hf : Measurable f
        g g' : α → E
        g'meas : Measurable g'
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
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g : α → E
      ⊢ AEMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ → AEMeasurable g (μ.wit …
    -/
  · rintro ⟨g', g'meas, hg'⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      ⊢ AEMeasurable g (μ.withDensity fun x => ↑(f x))
    -/
    refine ⟨fun x => (f x : ℝ)⁻¹ • g' x, hf.coe_nnreal_real.inv.smul g'meas, ?_⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      ⊢ (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq g fun x => H …
    -/
    rw [EventuallyEq, ae_withDensity_iff hf.coe_nnreal_ennreal]
    /-
      case mpr.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      ⊢ Filter.Eventually (fun x => Ne (↑(f x)) 0 → Eq (g x) (HSMul.hSMul (Inv.inv ↑ …
    -/
    filter_upwards [hg']
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
      hg' : (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(f x)) (g x)) g'
      ⊢ ∀ (a : α), Eq (HSMul.hSMul (↑(f a)) (g a)) (g' a) → Ne (↑(f a)) 0 → Eq (g a) …
    -/
    intro x hx h'x
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
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
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
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
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : SecondCountableTopology E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      f : α → NNReal
      hf : Measurable f
      g g' : α → E
      g'meas : Measurable g'
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

