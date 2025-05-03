/-- If `m₁, m₂` are independent σ-algebras and `f` is `m₁`-measurable, then `𝔼[f | m₂] = 𝔼[f]`
almost everywhere. -/
theorem condexp_indep_eq (hle₁ : m₁ ≤ m) (hle₂ : m₂ ≤ m) [SigmaFinite (μ.trim hle₂)]
    (hf : StronglyMeasurable[m₁] f) (hindp : Indep m₁ m₂ μ) : μ[f|m₂] =ᵐ[μ] fun _ => μ[f] := by
  /-
    Ω : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    m₁ m₂ m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → E
    hle₁ : LE.le m₁ m
    hle₂ : LE.le m₂ m
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
    hf : MeasureTheory.StronglyMeasurable f
    hindp : ProbabilityTheory.Indep m₁ m₂ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m₂ μ f) fun x => Me …
  -/
  by_cases hfint : Integrable f μ
  /-
    case pos
    Ω : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    m₁ m₂ m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → E
    hle₁ : LE.le m₁ m
    hle₂ : LE.le m₂ m
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
    hf : MeasureTheory.StronglyMeasurable f
    hindp : ProbabilityTheory.Indep m₁ m₂ μ
    hfint : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp m₂ μ f) fun x => Me …
  -/
  swap; · rw [condexp_undef hfint, integral_undef hfint]; rfl
                                                          /-
                                                            🎉 no goals
                                                          -/
  refine (ae_eq_condexp_of_forall_setIntegral_eq hle₂ hfint
    (fun s _ hs => integrableOn_const.2 (Or.inr hs)) (fun s hms hs => ?_)
      stronglyMeasurable_const.aeStronglyMeasurable').symm
  /-
    case pos
    Ω : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    m₁ m₂ m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → E
    hle₁ : LE.le m₁ m
    hle₂ : LE.le m₂ m
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
    hf : MeasureTheory.StronglyMeasurable f
    hindp : ProbabilityTheory.Indep m₁ m₂ μ
    hfint : MeasureTheory.Integrable f μ
    s : Set Ω
    hms : MeasurableSet s
    hs : LT.lt (μ s) Top.top
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => MeasureTheory.integral μ  …
  -/
  rw [setIntegral_const]
  /-
    case pos
    Ω : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    m₁ m₂ m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → E
    hle₁ : LE.le m₁ m
    hle₂ : LE.le m₂ m
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
    hf : MeasureTheory.StronglyMeasurable f
    hindp : ProbabilityTheory.Indep m₁ m₂ μ
    hfint : MeasureTheory.Integrable f μ
    s : Set Ω
    hms : MeasurableSet s
    hs : LT.lt (μ s) Top.top
    ⊢ Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.integral μ fun x => f x)) (Measu …
  -/
  rw [← memℒp_one_iff_integrable] at hfint
  /-
    case pos
    Ω : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    m₁ m₂ m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Ω → E
    hle₁ : LE.le m₁ m
    hle₂ : LE.le m₂ m
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
    hf : MeasureTheory.StronglyMeasurable f
    hindp : ProbabilityTheory.Indep m₁ m₂ μ
    hfint : MeasureTheory.Memℒp f 1 μ
    s : Set Ω
    hms : MeasurableSet s
    hs : LT.lt (μ s) Top.top
    ⊢ Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.integral μ fun x => f x)) (Measu …
  -/
  refine Memℒp.induction_stronglyMeasurable hle₁ ENNReal.one_ne_top _ ?_ ?_ ?_ ?_ hfint ?_
    /-
      case pos.refine_1
      Ω : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      m₁ m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Ω → E
      hle₁ : LE.le m₁ m
      hle₂ : LE.le m₂ m
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
      hf : MeasureTheory.StronglyMeasurable f
      hindp : ProbabilityTheory.Indep m₁ m₂ μ
      hfint : MeasureTheory.Memℒp f 1 μ
      s : Set Ω
      hms : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.AEStronglyMeasurable' m₁ f μ
    -/
  · exact ⟨f, hf, EventuallyEq.rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case pos.refine_2
      Ω : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      m₁ m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Ω → E
      hle₁ : LE.le m₁ m
      hle₂ : LE.le m₂ m
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
      hf : MeasureTheory.StronglyMeasurable f
      hindp : ProbabilityTheory.Indep m₁ m₂ μ
      hfint : MeasureTheory.Memℒp f 1 μ
      s : Set Ω
      hms : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      ⊢ ∀ (c : E) ⦃s_1 : Set Ω⦄, MeasurableSet s_1 → LT.lt (μ s_1) Top.top → Eq (HSM …
    -/
  · intro c t hmt _
    /-
      case pos.refine_2
      Ω : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      m₁ m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Ω → E
      hle₁ : LE.le m₁ m
      hle₂ : LE.le m₂ m
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
      hf : MeasureTheory.StronglyMeasurable f
      hindp : ProbabilityTheory.Indep m₁ m₂ μ
      hfint : MeasureTheory.Memℒp f 1 μ
      s : Set Ω
      hms : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      c : E
      t : Set Ω
      hmt : MeasurableSet t
      a✝ : LT.lt (μ t) Top.top
      ⊢ Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.integral μ fun x => t.indicator  …
    -/
    rw [Indep_iff] at hindp
    rw [integral_indicator (hle₁ _ hmt), setIntegral_const, smul_smul, ← ENNReal.toReal_mul,
      mul_comm, ← hindp _ _ hmt hms, setIntegral_indicator (hle₁ _ hmt), setIntegral_const,
      Set.inter_comm]
    /-
      case pos.refine_3
      Ω : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      m₁ m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Ω → E
      hle₁ : LE.le m₁ m
      hle₂ : LE.le m₂ m
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
      hf : MeasureTheory.StronglyMeasurable f
      hindp : ProbabilityTheory.Indep m₁ m₂ μ
      hfint : MeasureTheory.Memℒp f 1 μ
      s : Set Ω
      hms : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      ⊢ ∀ ⦃f g : Ω → E⦄, Disjoint (Function.support f) (Function.support g) → Measur …
    -/
  · intro u v _ huint hvint hu hv hu_eq hv_eq
    /-
      case pos.refine_3
      Ω : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      m₁ m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Ω → E
      hle₁ : LE.le m₁ m
      hle₂ : LE.le m₂ m
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
      hf : MeasureTheory.StronglyMeasurable f
      hindp : ProbabilityTheory.Indep m₁ m₂ μ
      hfint : MeasureTheory.Memℒp f 1 μ
      s : Set Ω
      hms : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      u v : Ω → E
      a✝ : Disjoint (Function.support u) (Function.support v)
      huint : MeasureTheory.Memℒp u 1 μ
      hvint : MeasureTheory.Memℒp v 1 μ
      hu : MeasureTheory.StronglyMeasurable u
      hv : MeasureTheory.StronglyMeasurable v
      hu_eq : Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.integral μ fun x => u x))  …
      hv_eq : Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.integral μ fun x => v x))  …
      ⊢ Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.integral μ fun x => HAdd.hAdd u  …
    -/
    rw [memℒp_one_iff_integrable] at huint hvint
    rw [integral_add' huint hvint, smul_add, hu_eq, hv_eq,
      integral_add' huint.integrableOn hvint.integrableOn]
  · have heq₁ : (fun f : lpMeas E ℝ m₁ 1 μ => ∫ x, (f : Ω → E) x ∂μ) =
        (fun f : Lp E 1 μ => ∫ x, f x ∂μ) ∘ Submodule.subtypeL _ := by
      refine funext fun f => integral_congr_ae ?_
      simp_rw [Submodule.coe_subtypeL', Submodule.coe_subtype]; norm_cast
    have heq₂ : (fun f : lpMeas E ℝ m₁ 1 μ => ∫ x in s, (f : Ω → E) x ∂μ) =
        (fun f : Lp E 1 μ => ∫ x in s, f x ∂μ) ∘ Submodule.subtypeL _ := by
      refine funext fun f => integral_congr_ae (ae_restrict_of_ae ?_)
      simp_rw [Submodule.coe_subtypeL', Submodule.coe_subtype]
      exact Eventually.of_forall fun _ => (by trivial)
    /-
      case pos.refine_4
      Ω : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      m₁ m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Ω → E
      hle₁ : LE.le m₁ m
      hle₂ : LE.le m₂ m
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
      hf : MeasureTheory.StronglyMeasurable f
      hindp : ProbabilityTheory.Indep m₁ m₂ μ
      hfint : MeasureTheory.Memℒp f 1 μ
      s : Set Ω
      hms : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      heq₁ : Eq (fun f => MeasureTheory.integral μ fun x => ↑↑↑f x) (Function.comp ( …
      heq₂ : Eq (fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑f x) (Fu …
      ⊢ IsClosed (setOf fun f => Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.integra …
    -/
    refine isClosed_eq (Continuous.const_smul ?_ _) ?_
      /-
        case pos.refine_4.refine_1
        Ω : Type u_1
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : CompleteSpace E
        m₁ m₂ m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Ω → E
        hle₁ : LE.le m₁ m
        hle₂ : LE.le m₂ m
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
        hf : MeasureTheory.StronglyMeasurable f
        hindp : ProbabilityTheory.Indep m₁ m₂ μ
        hfint : MeasureTheory.Memℒp f 1 μ
        s : Set Ω
        hms : MeasurableSet s
        hs : LT.lt (μ s) Top.top
        heq₁ : Eq (fun f => MeasureTheory.integral μ fun x => ↑↑↑f x) (Function.comp ( …
        heq₂ : Eq (fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑f x) (Fu …
        ⊢ Continuous fun f => MeasureTheory.integral μ fun x => ↑↑↑f x
      -/
    · rw [heq₁]
      /-
        case pos.refine_4.refine_1
        Ω : Type u_1
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : CompleteSpace E
        m₁ m₂ m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Ω → E
        hle₁ : LE.le m₁ m
        hle₂ : LE.le m₂ m
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
        hf : MeasureTheory.StronglyMeasurable f
        hindp : ProbabilityTheory.Indep m₁ m₂ μ
        hfint : MeasureTheory.Memℒp f 1 μ
        s : Set Ω
        hms : MeasurableSet s
        hs : LT.lt (μ s) Top.top
        heq₁ : Eq (fun f => MeasureTheory.integral μ fun x => ↑↑↑f x) (Function.comp ( …
        heq₂ : Eq (fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑f x) (Fu …
        ⊢ Continuous (Function.comp (fun f => MeasureTheory.integral μ fun x => ↑↑f x) …
      -/
      exact continuous_integral.comp (ContinuousLinearMap.continuous _)
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_4.refine_2
        Ω : Type u_1
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : CompleteSpace E
        m₁ m₂ m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Ω → E
        hle₁ : LE.le m₁ m
        hle₂ : LE.le m₂ m
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
        hf : MeasureTheory.StronglyMeasurable f
        hindp : ProbabilityTheory.Indep m₁ m₂ μ
        hfint : MeasureTheory.Memℒp f 1 μ
        s : Set Ω
        hms : MeasurableSet s
        hs : LT.lt (μ s) Top.top
        heq₁ : Eq (fun f => MeasureTheory.integral μ fun x => ↑↑↑f x) (Function.comp ( …
        heq₂ : Eq (fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑f x) (Fu …
        ⊢ Continuous fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑f x
      -/
    · rw [heq₂]
      /-
        case pos.refine_4.refine_2
        Ω : Type u_1
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : CompleteSpace E
        m₁ m₂ m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        f : Ω → E
        hle₁ : LE.le m₁ m
        hle₂ : LE.le m₂ m
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
        hf : MeasureTheory.StronglyMeasurable f
        hindp : ProbabilityTheory.Indep m₁ m₂ μ
        hfint : MeasureTheory.Memℒp f 1 μ
        s : Set Ω
        hms : MeasurableSet s
        hs : LT.lt (μ s) Top.top
        heq₁ : Eq (fun f => MeasureTheory.integral μ fun x => ↑↑↑f x) (Function.comp ( …
        heq₂ : Eq (fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑f x) (Fu …
        ⊢ Continuous (Function.comp (fun f => MeasureTheory.integral (μ.restrict s) fu …
      -/
      exact (continuous_setIntegral _).comp (ContinuousLinearMap.continuous _)
      /-
        🎉 no goals
      -/
    /-
      case pos.refine_5
      Ω : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      m₁ m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Ω → E
      hle₁ : LE.le m₁ m
      hle₂ : LE.le m₂ m
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
      hf : MeasureTheory.StronglyMeasurable f
      hindp : ProbabilityTheory.Indep m₁ m₂ μ
      hfint : MeasureTheory.Memℒp f 1 μ
      s : Set Ω
      hms : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      ⊢ ∀ ⦃f g : Ω → E⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory.Memℒp …
    -/
  · intro u v huv _ hueq
    rwa [← integral_congr_ae huv, ←
      (setIntegral_congr_ae (hle₂ _ hms) _ : ∫ x in s, u x ∂μ = ∫ x in s, v x ∂μ)]
    /-
      Ω : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      m₁ m₂ m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Ω → E
      hle₁ : LE.le m₁ m
      hle₂ : LE.le m₂ m
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hle₂)
      hf : MeasureTheory.StronglyMeasurable f
      hindp : ProbabilityTheory.Indep m₁ m₂ μ
      hfint : MeasureTheory.Memℒp f 1 μ
      s : Set Ω
      hms : MeasurableSet s
      hs : LT.lt (μ s) Top.top
      u v : Ω → E
      huv : (MeasureTheory.ae μ).EventuallyEq u v
      a✝ : MeasureTheory.Memℒp u 1 μ
      hueq : Eq (HSMul.hSMul (μ s).toReal (MeasureTheory.integral μ fun x => u x)) ( …
      ⊢ Filter.Eventually (fun x => Membership.mem s x → Eq (u x) (v x)) (MeasureThe …
    -/
    filter_upwards [huv] with x hx _ using hx
    /-
      🎉 no goals
    -/


