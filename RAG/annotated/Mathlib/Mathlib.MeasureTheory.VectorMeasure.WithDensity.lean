open Classical in
/-- Given a measure `μ` and an integrable function `f`, `μ.withDensityᵥ f` is
the vector measure which maps the set `s` to `∫ₛ f ∂μ`. -/
def Measure.withDensityᵥ {m : MeasurableSpace α} (μ : Measure α) (f : α → E) : VectorMeasure α E :=
  if hf : Integrable f μ then
    { measureOf' := fun s => if MeasurableSet s then ∫ x in s, f x ∂μ else 0
                   /-
                     α : Type u_1
                     m✝ : MeasurableSpace α
                     μ✝ : MeasureTheory.Measure α
                     E : Type u_2
                     inst✝¹ : NormedAddCommGroup E
                     inst✝ : NormedSpace Real E
                     m : MeasurableSpace α
                     μ : MeasureTheory.Measure α
                     f : α → E
                     hf : MeasureTheory.Integrable f μ
                     ⊢ Eq ((fun s => ite (MeasurableSet s) (MeasureTheory.integral (μ.restrict s) f …
                   -/
      empty' := by simp
                   /-
                     🎉 no goals
                   -/
      not_measurable' := fun _ hs => if_neg hs
      m_iUnion' := fun s hs₁ hs₂ => by
        /-
          α : Type u_1
          m✝ : MeasurableSpace α
          μ✝ : MeasureTheory.Measure α
          E : Type u_2
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : α → E
          hf : MeasureTheory.Integrable f μ
          s : Nat → Set α
          hs₁ : ∀ (i : Nat), MeasurableSet (s i)
          hs₂ : Pairwise (Function.onFun Disjoint s)
          ⊢ HasSum (fun i => (fun s => ite (MeasurableSet s) (MeasureTheory.integral (μ. …
        -/
        dsimp only
        /-
          α : Type u_1
          m✝ : MeasurableSpace α
          μ✝ : MeasureTheory.Measure α
          E : Type u_2
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : α → E
          hf : MeasureTheory.Integrable f μ
          s : Nat → Set α
          hs₁ : ∀ (i : Nat), MeasurableSet (s i)
          hs₂ : Pairwise (Function.onFun Disjoint s)
          ⊢ HasSum (fun i => ite (MeasurableSet (s i)) (MeasureTheory.integral (μ.restri …
        -/
        convert hasSum_integral_iUnion hs₁ hs₂ hf.integrableOn with n
          /-
            case h.e'_5.h
            α : Type u_1
            m✝ : MeasurableSpace α
            μ✝ : MeasureTheory.Measure α
            E : Type u_2
            inst✝¹ : NormedAddCommGroup E
            inst✝ : NormedSpace Real E
            m : MeasurableSpace α
            μ : MeasureTheory.Measure α
            f : α → E
            hf : MeasureTheory.Integrable f μ
            s : Nat → Set α
            hs₁ : ∀ (i : Nat), MeasurableSet (s i)
            hs₂ : Pairwise (Function.onFun Disjoint s)
            n : Nat
            ⊢ Eq (ite (MeasurableSet (s n)) (MeasureTheory.integral (μ.restrict (s n)) fun …
          -/
        · rw [if_pos (hs₁ n)]
          /-
            🎉 no goals
          -/
          /-
            case h.e'_6
            α : Type u_1
            m✝ : MeasurableSpace α
            μ✝ : MeasureTheory.Measure α
            E : Type u_2
            inst✝¹ : NormedAddCommGroup E
            inst✝ : NormedSpace Real E
            m : MeasurableSpace α
            μ : MeasureTheory.Measure α
            f : α → E
            hf : MeasureTheory.Integrable f μ
            s : Nat → Set α
            hs₁ : ∀ (i : Nat), MeasurableSet (s i)
            hs₂ : Pairwise (Function.onFun Disjoint s)
            ⊢ Eq (ite (MeasurableSet (Set.iUnion fun i => s i)) (MeasureTheory.integral (μ …
          -/
        · rw [if_pos (MeasurableSet.iUnion hs₁)] }
          /-
            🎉 no goals
          -/
  else 0


theorem withDensityᵥ_apply (hf : Integrable f μ) {s : Set α} (hs : MeasurableSet s) :
                                                /-
                                                  α : Type u_1
                                                  m : MeasurableSpace α
                                                  μ : MeasureTheory.Measure α
                                                  E : Type u_2
                                                  inst✝¹ : NormedAddCommGroup E
                                                  inst✝ : NormedSpace Real E
                                                  f : α → E
                                                  hf : MeasureTheory.Integrable f μ
                                                  s : Set α
                                                  hs : MeasurableSet s
                                                  ⊢ Eq (↑(μ.withDensityᵥ f) s) (MeasureTheory.integral (μ.restrict s) fun x => f …
                                                -/
    μ.withDensityᵥ f s = ∫ x in s, f x ∂μ := by rw [withDensityᵥ, dif_pos hf]; exact dif_pos hs
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem withDensityᵥ_zero : μ.withDensityᵥ (0 : α → E) = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    ⊢ Eq (μ.withDensityᵥ 0) 0
  -/
  ext1 s hs; erw [withDensityᵥ_apply (integrable_zero α E μ) hs]; simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem withDensityᵥ_neg : μ.withDensityᵥ (-f) = -μ.withDensityᵥ f := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → E
    ⊢ Eq (μ.withDensityᵥ (Neg.neg f)) (Neg.neg (μ.withDensityᵥ f))
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (μ.withDensityᵥ (Neg.neg f)) (Neg.neg (μ.withDensityᵥ f))
    -/
  · ext1 i hi
    rw [VectorMeasure.neg_apply, withDensityᵥ_apply hf hi, ← integral_neg,
      withDensityᵥ_apply hf.neg hi]
    /-
      case pos.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → E
      hf : MeasureTheory.Integrable f μ
      i : Set α
      hi : MeasurableSet i
      ⊢ Eq (MeasureTheory.integral (μ.restrict i) fun x => Neg.neg f x) (MeasureTheo …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (μ.withDensityᵥ (Neg.neg f)) (Neg.neg (μ.withDensityᵥ f))
    -/
  · rw [withDensityᵥ, withDensityᵥ, dif_neg hf, dif_neg, neg_zero]
    /-
      case neg.hnc
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Not (MeasureTheory.Integrable (Neg.neg f) μ)
    -/
    rwa [integrable_neg_iff]
    /-
      🎉 no goals
    -/


theorem withDensityᵥ_neg' : (μ.withDensityᵥ fun x => -f x) = -μ.withDensityᵥ f :=
  withDensityᵥ_neg


@[simp]
theorem withDensityᵥ_add (hf : Integrable f μ) (hg : Integrable g μ) :
    μ.withDensityᵥ (f + g) = μ.withDensityᵥ f + μ.withDensityᵥ g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (μ.withDensityᵥ (HAdd.hAdd f g)) (HAdd.hAdd (μ.withDensityᵥ f) (μ.withDen …
  -/
  ext1 i hi
  rw [withDensityᵥ_apply (hf.add hg) hi, VectorMeasure.add_apply, withDensityᵥ_apply hf hi,
    withDensityᵥ_apply hg hi]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (MeasureTheory.integral (μ.restrict i) fun x => HAdd.hAdd f g x) (HAdd.hA …
  -/
  simp_rw [Pi.add_apply]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (MeasureTheory.integral (μ.restrict i) fun x => HAdd.hAdd (f x) (g x)) (H …
  -/
  rw [integral_add] <;> rw [← integrableOn_univ]
    /-
      case h.hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : α → E
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      i : Set α
      hi : MeasurableSet i
      ⊢ MeasureTheory.IntegrableOn f Set.univ (μ.restrict i)
    -/
  · exact hf.integrableOn.restrict MeasurableSet.univ
    /-
      🎉 no goals
    -/
    /-
      case h.hg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : α → E
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      i : Set α
      hi : MeasurableSet i
      ⊢ MeasureTheory.IntegrableOn g Set.univ (μ.restrict i)
    -/
  · exact hg.integrableOn.restrict MeasurableSet.univ
    /-
      🎉 no goals
    -/


theorem withDensityᵥ_add' (hf : Integrable f μ) (hg : Integrable g μ) :
    (μ.withDensityᵥ fun x => f x + g x) = μ.withDensityᵥ f + μ.withDensityᵥ g :=
  withDensityᵥ_add hf hg


@[simp]
theorem withDensityᵥ_sub (hf : Integrable f μ) (hg : Integrable g μ) :
    μ.withDensityᵥ (f - g) = μ.withDensityᵥ f - μ.withDensityᵥ g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (μ.withDensityᵥ (HSub.hSub f g)) (HSub.hSub (μ.withDensityᵥ f) (μ.withDen …
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg, withDensityᵥ_add hf hg.neg, withDensityᵥ_neg]
  /-
    🎉 no goals
  -/


theorem withDensityᵥ_sub' (hf : Integrable f μ) (hg : Integrable g μ) :
    (μ.withDensityᵥ fun x => f x - g x) = μ.withDensityᵥ f - μ.withDensityᵥ g :=
  withDensityᵥ_sub hf hg


@[simp]
theorem withDensityᵥ_smul {𝕜 : Type*} [NontriviallyNormedField 𝕜] [NormedSpace 𝕜 E]
    [SMulCommClass ℝ 𝕜 E] (f : α → E) (r : 𝕜) : μ.withDensityᵥ (r • f) = r • μ.withDensityᵥ f := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    𝕜 : Type u_3
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : SMulCommClass Real 𝕜 E
    f : α → E
    r : 𝕜
    ⊢ Eq (μ.withDensityᵥ (HSMul.hSMul r f)) (HSMul.hSMul r (μ.withDensityᵥ f))
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      𝕜 : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : SMulCommClass Real 𝕜 E
      f : α → E
      r : 𝕜
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (μ.withDensityᵥ (HSMul.hSMul r f)) (HSMul.hSMul r (μ.withDensityᵥ f))
    -/
  · ext1 i hi
    rw [withDensityᵥ_apply (hf.smul r) hi, VectorMeasure.smul_apply, withDensityᵥ_apply hf hi, ←
      integral_smul r f]
    /-
      case pos.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      𝕜 : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : SMulCommClass Real 𝕜 E
      f : α → E
      r : 𝕜
      hf : MeasureTheory.Integrable f μ
      i : Set α
      hi : MeasurableSet i
      ⊢ Eq (MeasureTheory.integral (μ.restrict i) fun x => HSMul.hSMul r f x) (Measu …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      𝕜 : Type u_3
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : SMulCommClass Real 𝕜 E
      f : α → E
      r : 𝕜
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (μ.withDensityᵥ (HSMul.hSMul r f)) (HSMul.hSMul r (μ.withDensityᵥ f))
    -/
  · by_cases hr : r = 0
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        𝕜 : Type u_3
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : SMulCommClass Real 𝕜 E
        f : α → E
        r : 𝕜
        hf : Not (MeasureTheory.Integrable f μ)
        hr : Eq r 0
        ⊢ Eq (μ.withDensityᵥ (HSMul.hSMul r f)) (HSMul.hSMul r (μ.withDensityᵥ f))
      -/
    · rw [hr, zero_smul, zero_smul, withDensityᵥ_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        𝕜 : Type u_3
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : SMulCommClass Real 𝕜 E
        f : α → E
        r : 𝕜
        hf : Not (MeasureTheory.Integrable f μ)
        hr : Not (Eq r 0)
        ⊢ Eq (μ.withDensityᵥ (HSMul.hSMul r f)) (HSMul.hSMul r (μ.withDensityᵥ f))
      -/
    · rw [withDensityᵥ, withDensityᵥ, dif_neg hf, dif_neg, smul_zero]
      /-
        case neg.hnc
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        𝕜 : Type u_3
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : SMulCommClass Real 𝕜 E
        f : α → E
        r : 𝕜
        hf : Not (MeasureTheory.Integrable f μ)
        hr : Not (Eq r 0)
        ⊢ Not (MeasureTheory.Integrable (HSMul.hSMul r f) μ)
      -/
      rwa [integrable_smul_iff hr f]
      /-
        🎉 no goals
      -/


theorem withDensityᵥ_smul' {𝕜 : Type*} [NontriviallyNormedField 𝕜] [NormedSpace 𝕜 E]
    [SMulCommClass ℝ 𝕜 E] (f : α → E) (r : 𝕜) :
    (μ.withDensityᵥ fun x => r • f x) = r • μ.withDensityᵥ f :=
  withDensityᵥ_smul f r


theorem withDensityᵥ_smul_eq_withDensityᵥ_withDensity {f : α → ℝ≥0} {g : α → E}
    (hf : AEMeasurable f μ) (hfg : Integrable (f • g) μ) :
    μ.withDensityᵥ (f • g) = (μ.withDensity (fun x ↦ f x)).withDensityᵥ g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → NNReal
    g : α → E
    hf : AEMeasurable f μ
    hfg : MeasureTheory.Integrable (HSMul.hSMul f g) μ
    ⊢ Eq (μ.withDensityᵥ (HSMul.hSMul f g)) ((μ.withDensity fun x => ↑(f x)).withD …
  -/
  ext s hs
  rw [withDensityᵥ_apply hfg hs,
    withDensityᵥ_apply ((integrable_withDensity_iff_integrable_smul₀ hf).mpr hfg) hs,
    setIntegral_withDensity_eq_setIntegral_smul₀ hf.restrict _ hs]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → NNReal
    g : α → E
    hf : AEMeasurable f μ
    hfg : MeasureTheory.Integrable (HSMul.hSMul f g) μ
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul f g x) (Measu …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem withDensityᵥ_smul_eq_withDensityᵥ_withDensity' {f : α → ℝ≥0∞} {g : α → E}
    (hf : AEMeasurable f μ) (hflt : ∀ᵐ x ∂μ, f x < ∞)
    (hfg : Integrable (fun x ↦ (f x).toReal • g x) μ) :
    μ.withDensityᵥ (fun x ↦ (f x).toReal • g x) = (μ.withDensity f).withDensityᵥ g := by
  rw [← withDensity_congr_ae (coe_toNNReal_ae_eq hflt),
    ← withDensityᵥ_smul_eq_withDensityᵥ_withDensity hf.ennreal_toNNReal hfg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → ENNReal
    g : α → E
    hf : AEMeasurable f μ
    hflt : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    hfg : MeasureTheory.Integrable (fun x => HSMul.hSMul (f x).toReal (g x)) μ
    ⊢ Eq (μ.withDensityᵥ fun x => HSMul.hSMul (f x).toReal (g x)) (μ.withDensityᵥ  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Measure.withDensityᵥ_absolutelyContinuous (μ : Measure α) (f : α → ℝ) :
    μ.withDensityᵥ f ≪ᵥ μ.toENNRealVectorMeasure := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ⊢ (μ.withDensityᵥ f).AbsolutelyContinuous μ.toENNRealVectorMeasure
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ (μ.withDensityᵥ f).AbsolutelyContinuous μ.toENNRealVectorMeasure
    -/
  · refine VectorMeasure.AbsolutelyContinuous.mk fun i hi₁ hi₂ => ?_
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : Eq (↑μ.toENNRealVectorMeasure i) 0
      ⊢ Eq (↑(μ.withDensityᵥ f) i) 0
    -/
    rw [toENNRealVectorMeasure_apply_measurable hi₁] at hi₂
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : Eq (μ i) 0
      ⊢ Eq (↑(μ.withDensityᵥ f) i) 0
    -/
    rw [withDensityᵥ_apply hf hi₁, Measure.restrict_zero_set hi₂, integral_zero_measure]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ (μ.withDensityᵥ f).AbsolutelyContinuous μ.toENNRealVectorMeasure
    -/
  · rw [withDensityᵥ, dif_neg hf]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ MeasureTheory.VectorMeasure.AbsolutelyContinuous 0 μ.toENNRealVectorMeasure
    -/
    exact VectorMeasure.AbsolutelyContinuous.zero _
    /-
      🎉 no goals
    -/


/-- Having the same density implies the underlying functions are equal almost everywhere. -/
theorem Integrable.ae_eq_of_withDensityᵥ_eq [CompleteSpace E] {f g : α → E} (hf : Integrable f μ)
    (hg : Integrable g μ) (hfg : μ.withDensityᵥ f = μ.withDensityᵥ g) : f =ᵐ[μ] g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : Eq (μ.withDensityᵥ f) (μ.withDensityᵥ g)
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  refine hf.ae_eq_of_forall_setIntegral_eq f g hg fun i hi _ => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : Eq (μ.withDensityᵥ f) (μ.withDensityᵥ g)
    i : Set α
    hi : MeasurableSet i
    x✝ : LT.lt (μ i) Top.top
    ⊢ Eq (MeasureTheory.integral (μ.restrict i) fun x => f x) (MeasureTheory.integ …
  -/
  rw [← withDensityᵥ_apply hf hi, hfg, withDensityᵥ_apply hg hi]
  /-
    🎉 no goals
  -/


theorem WithDensityᵥEq.congr_ae {f g : α → E} (h : f =ᵐ[μ] g) :
    μ.withDensityᵥ f = μ.withDensityᵥ g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : α → E
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Eq (μ.withDensityᵥ f) (μ.withDensityᵥ g)
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : α → E
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (μ.withDensityᵥ f) (μ.withDensityᵥ g)
    -/
  · ext i hi
    /-
      case pos.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : α → E
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : MeasureTheory.Integrable f μ
      i : Set α
      hi : MeasurableSet i
      ⊢ Eq (↑(μ.withDensityᵥ f) i) (↑(μ.withDensityᵥ g) i)
    -/
    rw [withDensityᵥ_apply hf hi, withDensityᵥ_apply (hf.congr h) hi]
    /-
      case pos.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : α → E
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : MeasureTheory.Integrable f μ
      i : Set α
      hi : MeasurableSet i
      ⊢ Eq (MeasureTheory.integral (μ.restrict i) fun x => f x) (MeasureTheory.integ …
    -/
    exact integral_congr_ae (ae_restrict_of_ae h)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : α → E
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (μ.withDensityᵥ f) (μ.withDensityᵥ g)
    -/
  · have hg : ¬Integrable g μ := by intro hg; exact hf (hg.congr h.symm)
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : α → E
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hf : Not (MeasureTheory.Integrable f μ)
      hg : Not (MeasureTheory.Integrable g μ)
      ⊢ Eq (μ.withDensityᵥ f) (μ.withDensityᵥ g)
    -/
    rw [withDensityᵥ, withDensityᵥ, dif_neg hf, dif_neg hg]
    /-
      🎉 no goals
    -/


theorem Integrable.withDensityᵥ_eq_iff [CompleteSpace E]
    {f g : α → E} (hf : Integrable f μ) (hg : Integrable g μ) :
    μ.withDensityᵥ f = μ.withDensityᵥ g ↔ f =ᵐ[μ] g :=
  ⟨fun hfg => hf.ae_eq_of_withDensityᵥ_eq hg hfg, fun h => WithDensityᵥEq.congr_ae h⟩


theorem withDensityᵥ_toReal {f : α → ℝ≥0∞} (hfm : AEMeasurable f μ) (hf : (∫⁻ x, f x ∂μ) ≠ ∞) :
    (μ.withDensityᵥ fun x => (f x).toReal) =
      @toSignedMeasure α _ (μ.withDensity f) (isFiniteMeasure_withDensity hf) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : AEMeasurable f μ
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ⊢ Eq (μ.withDensityᵥ fun x => (f x).toReal) (μ.withDensity f).toSignedMeasure
  -/
  have hfi := integrable_toReal_of_lintegral_ne_top hfm hf
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : AEMeasurable f μ
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hfi : MeasureTheory.Integrable (fun x => (f x).toReal) μ
    ⊢ Eq (μ.withDensityᵥ fun x => (f x).toReal) (μ.withDensity f).toSignedMeasure
  -/
  haveI := isFiniteMeasure_withDensity hf
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : AEMeasurable f μ
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hfi : MeasureTheory.Integrable (fun x => (f x).toReal) μ
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity f)
    ⊢ Eq (μ.withDensityᵥ fun x => (f x).toReal) (μ.withDensity f).toSignedMeasure
  -/
  ext i hi
  rw [withDensityᵥ_apply hfi hi, toSignedMeasure_apply_measurable hi, withDensity_apply _ hi,
    integral_toReal hfm.restrict]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : AEMeasurable f μ
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hfi : MeasureTheory.Integrable (fun x => (f x).toReal) μ
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity f)
    i : Set α
    hi : MeasurableSet i
    ⊢ Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae (μ.restri …
  -/
  refine ae_lt_top' hfm.restrict (ne_top_of_le_ne_top hf ?_)
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : AEMeasurable f μ
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hfi : MeasureTheory.Integrable (fun x => (f x).toReal) μ
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity f)
    i : Set α
    hi : MeasurableSet i
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict i) fun x => f x) (MeasureTheory.l …
  -/
  conv_rhs => rw [← setLIntegral_univ]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : AEMeasurable f μ
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hfi : MeasureTheory.Integrable (fun x => (f x).toReal) μ
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity f)
    i : Set α
    hi : MeasurableSet i
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict i) fun x => f x) (MeasureTheory.l …
  -/
  exact lintegral_mono_set (Set.subset_univ _)
  /-
    🎉 no goals
  -/


theorem withDensityᵥ_eq_withDensity_pos_part_sub_withDensity_neg_part {f : α → ℝ}
    (hfi : Integrable f μ) :
    μ.withDensityᵥ f =
      @toSignedMeasure α _ (μ.withDensity fun x => ENNReal.ofReal <| f x)
          (isFiniteMeasure_withDensity_ofReal hfi.2) -
        @toSignedMeasure α _ (μ.withDensity fun x => ENNReal.ofReal <| -f x)
          (isFiniteMeasure_withDensity_ofReal hfi.neg.2) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfi : MeasureTheory.Integrable f μ
    ⊢ Eq (μ.withDensityᵥ f) (HSub.hSub (μ.withDensity fun x => ENNReal.ofReal (f x …
  -/
  haveI := isFiniteMeasure_withDensity_ofReal hfi.2
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfi : MeasureTheory.Integrable f μ
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal (f …
    ⊢ Eq (μ.withDensityᵥ f) (HSub.hSub (μ.withDensity fun x => ENNReal.ofReal (f x …
  -/
  haveI := isFiniteMeasure_withDensity_ofReal hfi.neg.2
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfi : MeasureTheory.Integrable f μ
    this✝ : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal ( …
    this : MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal (N …
    ⊢ Eq (μ.withDensityᵥ f) (HSub.hSub (μ.withDensity fun x => ENNReal.ofReal (f x …
  -/
  ext i hi
  rw [withDensityᵥ_apply hfi hi,
    integral_eq_lintegral_pos_part_sub_lintegral_neg_part hfi.integrableOn,
    VectorMeasure.sub_apply, toSignedMeasure_apply_measurable hi,
    toSignedMeasure_apply_measurable hi, withDensity_apply _ hi, withDensity_apply _ hi]


theorem Integrable.withDensityᵥ_trim_eq_integral {m m0 : MeasurableSpace α} {μ : Measure α}
    (hm : m ≤ m0) {f : α → ℝ} (hf : Integrable f μ) {i : Set α} (hi : MeasurableSet[m] i) :
    (μ.withDensityᵥ f).trim hm i = ∫ x in i, f x ∂μ := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (↑((μ.withDensityᵥ f).trim hm) i) (MeasureTheory.integral (μ.restrict i)  …
  -/
  rw [VectorMeasure.trim_measurableSet_eq hm hi, withDensityᵥ_apply hf (hm _ hi)]
  /-
    🎉 no goals
  -/


theorem Integrable.withDensityᵥ_trim_absolutelyContinuous {m m0 : MeasurableSpace α} {μ : Measure α}
    (hm : m ≤ m0) (hfi : Integrable f μ) :
    (μ.withDensityᵥ f).trim hm ≪ᵥ (μ.trim hm).toENNRealVectorMeasure := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hfi : MeasureTheory.Integrable f μ
    ⊢ ((μ.withDensityᵥ f).trim hm).AbsolutelyContinuous (μ.trim hm).toENNRealVecto …
  -/
  refine VectorMeasure.AbsolutelyContinuous.mk fun j hj₁ hj₂ => ?_
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hfi : MeasureTheory.Integrable f μ
    j : Set α
    hj₁ : MeasurableSet j
    hj₂ : Eq (↑(μ.trim hm).toENNRealVectorMeasure j) 0
    ⊢ Eq (↑((μ.withDensityᵥ f).trim hm) j) 0
  -/
  rw [Measure.toENNRealVectorMeasure_apply_measurable hj₁, trim_measurableSet_eq hm hj₁] at hj₂
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hfi : MeasureTheory.Integrable f μ
    j : Set α
    hj₁ : MeasurableSet j
    hj₂ : Eq (μ j) 0
    ⊢ Eq (↑((μ.withDensityᵥ f).trim hm) j) 0
  -/
  rw [VectorMeasure.trim_measurableSet_eq hm hj₁, withDensityᵥ_apply hfi (hm _ hj₁)]
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    hfi : MeasureTheory.Integrable f μ
    j : Set α
    hj₁ : MeasurableSet j
    hj₂ : Eq (μ j) 0
    ⊢ Eq (MeasureTheory.integral (μ.restrict j) fun x => f x) 0
  -/
  simp only [Measure.restrict_eq_zero.mpr hj₂, integral_zero_measure]
  /-
    🎉 no goals
  -/


