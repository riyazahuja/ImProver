theorem withDensity_rnDeriv_eq (μ ν : Measure α) [HaveLebesgueDecomposition μ ν] (h : μ ≪ ν) :
    ν.withDensity (rnDeriv μ ν) = μ := by
  suffices μ.singularPart ν = 0 by
    conv_rhs => rw [haveLebesgueDecomposition_add μ ν, this, zero_add]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h : μ.AbsolutelyContinuous ν
    ⊢ Eq (μ.singularPart ν) 0
  -/
  suffices μ.singularPart ν Set.univ = 0 by simpa using this
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h : μ.AbsolutelyContinuous ν
    ⊢ Eq ((μ.singularPart ν) Set.univ) 0
  -/
  have h_sing := mutuallySingular_singularPart μ ν
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h : μ.AbsolutelyContinuous ν
    h_sing : (μ.singularPart ν).MutuallySingular ν
    ⊢ Eq ((μ.singularPart ν) Set.univ) 0
  -/
  rw [← measure_add_measure_compl h_sing.measurableSet_nullSet]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h : μ.AbsolutelyContinuous ν
    h_sing : (μ.singularPart ν).MutuallySingular ν
    ⊢ Eq (HAdd.hAdd ((μ.singularPart ν) h_sing.nullSet) ((μ.singularPart ν) (HasCo …
  -/
  simp only [MutuallySingular.measure_nullSet, zero_add]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h : μ.AbsolutelyContinuous ν
    h_sing : (μ.singularPart ν).MutuallySingular ν
    ⊢ Eq ((μ.singularPart ν) (HasCompl.compl h_sing.nullSet)) 0
  -/
  refine le_antisymm ?_ (zero_le _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h : μ.AbsolutelyContinuous ν
    h_sing : (μ.singularPart ν).MutuallySingular ν
    ⊢ LE.le ((μ.singularPart ν) (HasCompl.compl h_sing.nullSet)) 0
  -/
  refine (singularPart_le μ ν ?_ ).trans_eq ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    h : μ.AbsolutelyContinuous ν
    h_sing : (μ.singularPart ν).MutuallySingular ν
    ⊢ Eq (μ (HasCompl.compl h_sing.nullSet)) 0
  -/
  exact h h_sing.measure_compl_nullSet
  /-
    🎉 no goals
  -/


/-- **The Radon-Nikodym theorem**: Given two measures `μ` and `ν`, if
`HaveLebesgueDecomposition μ ν`, then `μ` is absolutely continuous to `ν` if and only if
`ν.withDensity (rnDeriv μ ν) = μ`. -/
theorem absolutelyContinuous_iff_withDensity_rnDeriv_eq
    [HaveLebesgueDecomposition μ ν] : μ ≪ ν ↔ ν.withDensity (rnDeriv μ ν) = μ :=
  ⟨withDensity_rnDeriv_eq μ ν, fun h => h ▸ withDensity_absolutelyContinuous _ _⟩


lemma rnDeriv_pos [HaveLebesgueDecomposition μ ν] (hμν : μ ≪ ν) :
    ∀ᵐ x ∂μ, 0 < μ.rnDeriv ν x := by
  rw [← Measure.withDensity_rnDeriv_eq _ _  hμν,
    ae_withDensity_iff (Measure.measurable_rnDeriv _ _), Measure.withDensity_rnDeriv_eq _ _  hμν]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Filter.Eventually (fun x => Ne (μ.rnDeriv ν x) 0 → LT.lt 0 (μ.rnDeriv ν x))  …
  -/
  exact ae_of_all _ (fun x hx ↦ lt_of_le_of_ne (zero_le _) hx.symm)
  /-
    🎉 no goals
  -/


lemma rnDeriv_pos' [HaveLebesgueDecomposition ν μ] [SigmaFinite μ] (hμν : μ ≪ ν) :
    ∀ᵐ x ∂μ, 0 < ν.rnDeriv μ x := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : ν.HaveLebesgueDecomposition μ
    inst✝ : MeasureTheory.SigmaFinite μ
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Filter.Eventually (fun x => LT.lt 0 (ν.rnDeriv μ x)) (MeasureTheory.ae μ)
  -/
  refine (absolutelyContinuous_withDensity_rnDeriv hμν).ae_le ?_
  filter_upwards [Measure.rnDeriv_pos (withDensity_absolutelyContinuous μ (ν.rnDeriv μ)),
    (withDensity_absolutelyContinuous μ (ν.rnDeriv μ)).ae_le
    (Measure.rnDeriv_withDensity μ (Measure.measurable_rnDeriv ν μ))] with x hx hx2
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : ν.HaveLebesgueDecomposition μ
    inst✝ : MeasureTheory.SigmaFinite μ
    hμν : μ.AbsolutelyContinuous ν
    x : α
    hx : LT.lt 0 ((μ.withDensity (ν.rnDeriv μ)).rnDeriv μ x)
    hx2 : Eq ((μ.withDensity (ν.rnDeriv μ)).rnDeriv μ x) (ν.rnDeriv μ x)
    ⊢ LT.lt 0 (ν.rnDeriv μ x)
  -/
  rwa [← hx2]
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `rnDeriv_withDensity_left`. -/
lemma rnDeriv_withDensity_withDensity_rnDeriv_left (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν]
    (hf_ne_top : ∀ᵐ x ∂μ, f x ≠ ∞) :
    ((ν.withDensity (μ.rnDeriv ν)).withDensity f).rnDeriv ν =ᵐ[ν] (μ.withDensity f).rnDeriv ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae ν).EventuallyEq (((ν.withDensity (μ.rnDeriv ν)).withDensit …
  -/
  conv_rhs => rw [μ.haveLebesgueDecomposition_add ν, add_comm, withDensity_add_measure]
  have : SigmaFinite ((μ.singularPart ν).withDensity f) :=
    SigmaFinite.withDensity_of_ne_top (ae_mono (Measure.singularPart_le _ _) hf_ne_top)
  have : SigmaFinite ((ν.withDensity (μ.rnDeriv ν)).withDensity f) :=
    SigmaFinite.withDensity_of_ne_top (ae_mono (Measure.withDensity_rnDeriv_le _ _) hf_ne_top)
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    this✝ : MeasureTheory.SigmaFinite ((μ.singularPart ν).withDensity f)
    this : MeasureTheory.SigmaFinite ((ν.withDensity (μ.rnDeriv ν)).withDensity f)
    ⊢ (MeasureTheory.ae ν).EventuallyEq (((ν.withDensity (μ.rnDeriv ν)).withDensit …
  -/
  exact (rnDeriv_add_of_mutuallySingular _ _ _ (mutuallySingular_singularPart μ ν).withDensity).symm
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `rnDeriv_withDensity_right`. -/
lemma rnDeriv_withDensity_withDensity_rnDeriv_right (μ ν : Measure α) [SigmaFinite μ]
    [SigmaFinite ν] (hf : AEMeasurable f ν) (hf_ne_zero : ∀ᵐ x ∂ν, f x ≠ 0)
    (hf_ne_top : ∀ᵐ x ∂ν, f x ≠ ∞) :
    (ν.withDensity (μ.rnDeriv ν)).rnDeriv (ν.withDensity f) =ᵐ[ν] μ.rnDeriv (ν.withDensity f) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv (ν. …
  -/
  conv_rhs => rw [μ.haveLebesgueDecomposition_add ν, add_comm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv (ν. …
  -/
  have hν_ac : ν ≪ ν.withDensity f := withDensity_absolutelyContinuous' hf hf_ne_zero
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    hν_ac : ν.AbsolutelyContinuous (ν.withDensity f)
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv (ν. …
  -/
  refine hν_ac.ae_eq ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    hν_ac : ν.AbsolutelyContinuous (ν.withDensity f)
    ⊢ (MeasureTheory.ae (ν.withDensity f)).EventuallyEq ((ν.withDensity (μ.rnDeriv …
  -/
  have : SigmaFinite (ν.withDensity f) := SigmaFinite.withDensity_of_ne_top hf_ne_top
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    hν_ac : ν.AbsolutelyContinuous (ν.withDensity f)
    this : MeasureTheory.SigmaFinite (ν.withDensity f)
    ⊢ (MeasureTheory.ae (ν.withDensity f)).EventuallyEq ((ν.withDensity (μ.rnDeriv …
  -/
  refine (rnDeriv_add_of_mutuallySingular _ _ _ ?_).symm
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    hν_ac : ν.AbsolutelyContinuous (ν.withDensity f)
    this : MeasureTheory.SigmaFinite (ν.withDensity f)
    ⊢ (μ.singularPart ν).MutuallySingular (ν.withDensity f)
  -/
  exact ((mutuallySingular_singularPart μ ν).symm.withDensity).symm
  /-
    🎉 no goals
  -/


lemma rnDeriv_withDensity_left_of_absolutelyContinuous {ν : Measure α} [SigmaFinite μ]
    [SigmaFinite ν] (hμν : μ ≪ ν) (hf : AEMeasurable f ν) :
    (μ.withDensity f).rnDeriv ν =ᵐ[ν] fun x ↦ f x * μ.rnDeriv ν x := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hf : AEMeasurable f ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.withDensity f).rnDeriv ν) fun x => HMu …
  -/
  refine (Measure.eq_rnDeriv₀ ?_ Measure.MutuallySingular.zero_left ?_).symm
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      ⊢ AEMeasurable (fun x => HMul.hMul (f x) (μ.rnDeriv ν x)) ν
    -/
  · exact hf.mul (Measure.measurable_rnDeriv _ _).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      ⊢ Eq (μ.withDensity f) (HAdd.hAdd 0 (ν.withDensity fun x => HMul.hMul (f x) (μ …
    -/
  · ext1 s hs
    /-
      case refine_2.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq ((μ.withDensity f) s) ((HAdd.hAdd 0 (ν.withDensity fun x => HMul.hMul (f  …
    -/
    rw [zero_add, withDensity_apply _ hs, withDensity_apply _ hs]
    /-
      case refine_2.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory.lint …
    -/
    conv_lhs => rw [← Measure.withDensity_rnDeriv_eq _ _ hμν]
    /-
      case refine_2.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral ((ν.withDensity (μ.rnDeriv ν)).restrict s) fun a …
    -/
    rw [setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀ _ _ _ hs]
      /-
        case refine_2.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        s : Set α
        hs : MeasurableSet s
        ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun a => HMul.hMul (μ.rnDeriv ν)  …
      -/
    · congr with x
      /-
        case refine_2.h.e_f.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        s : Set α
        hs : MeasurableSet s
        x : α
        ⊢ Eq (HMul.hMul (μ.rnDeriv ν) f x) (HMul.hMul (f x) (μ.rnDeriv ν x))
      -/
      rw [mul_comm]
      /-
        case refine_2.h.e_f.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        s : Set α
        hs : MeasurableSet s
        x : α
        ⊢ Eq (HMul.hMul f (μ.rnDeriv ν) x) (HMul.hMul (f x) (μ.rnDeriv ν x))
      -/
      simp only [Pi.mul_apply]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        s : Set α
        hs : MeasurableSet s
        ⊢ Filter.Eventually (fun x => LT.lt (μ.rnDeriv ν x) Top.top) (MeasureTheory.ae …
      -/
    · refine ae_restrict_of_ae ?_
      /-
        case refine_2.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        s : Set α
        hs : MeasurableSet s
        ⊢ Filter.Eventually (fun x => LT.lt (μ.rnDeriv ν x) Top.top) (MeasureTheory.ae …
      -/
      exact Measure.rnDeriv_lt_top _ _
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SigmaFinite μ
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        s : Set α
        hs : MeasurableSet s
        ⊢ AEMeasurable (μ.rnDeriv ν) (ν.restrict s)
      -/
    · exact (Measure.measurable_rnDeriv _ _).aemeasurable
      /-
        🎉 no goals
      -/


lemma rnDeriv_withDensity_left {μ ν : Measure α} [SigmaFinite μ] [SigmaFinite ν]
    (hfν : AEMeasurable f ν) (hf_ne_top : ∀ᵐ x ∂μ, f x ≠ ∞) :
    (μ.withDensity f).rnDeriv ν =ᵐ[ν] fun x ↦ f x * μ.rnDeriv ν x := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.withDensity f).rnDeriv ν) fun x => HMu …
  -/
  let μ' := ν.withDensity (μ.rnDeriv ν)
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    μ' : MeasureTheory.Measure α := ν.withDensity (μ.rnDeriv ν)
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.withDensity f).rnDeriv ν) fun x => HMu …
  -/
  have hμ'ν : μ' ≪ ν := withDensity_absolutelyContinuous _ _
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    μ' : MeasureTheory.Measure α := ν.withDensity (μ.rnDeriv ν)
    hμ'ν : μ'.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.withDensity f).rnDeriv ν) fun x => HMu …
  -/
  have h := rnDeriv_withDensity_left_of_absolutelyContinuous hμ'ν hfν
  have h1 : μ'.rnDeriv ν =ᵐ[ν] μ.rnDeriv ν :=
    Measure.rnDeriv_withDensity _ (Measure.measurable_rnDeriv _ _)
  have h2 : (μ'.withDensity f).rnDeriv ν =ᵐ[ν] (μ.withDensity f).rnDeriv ν := by
    exact rnDeriv_withDensity_withDensity_rnDeriv_left μ ν hf_ne_top
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    μ' : MeasureTheory.Measure α := ν.withDensity (μ.rnDeriv ν)
    hμ'ν : μ'.AbsolutelyContinuous ν
    h : (MeasureTheory.ae ν).EventuallyEq ((μ'.withDensity f).rnDeriv ν) fun x =>  …
    h1 : (MeasureTheory.ae ν).EventuallyEq (μ'.rnDeriv ν) (μ.rnDeriv ν)
    h2 : (MeasureTheory.ae ν).EventuallyEq ((μ'.withDensity f).rnDeriv ν) ((μ.with …
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.withDensity f).rnDeriv ν) fun x => HMu …
  -/
  filter_upwards [h, h1, h2] with x hx hx1 hx2
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    μ' : MeasureTheory.Measure α := ν.withDensity (μ.rnDeriv ν)
    hμ'ν : μ'.AbsolutelyContinuous ν
    h : (MeasureTheory.ae ν).EventuallyEq ((μ'.withDensity f).rnDeriv ν) fun x =>  …
    h1 : (MeasureTheory.ae ν).EventuallyEq (μ'.rnDeriv ν) (μ.rnDeriv ν)
    h2 : (MeasureTheory.ae ν).EventuallyEq ((μ'.withDensity f).rnDeriv ν) ((μ.with …
    x : α
    hx : Eq ((μ'.withDensity f).rnDeriv ν x) (HMul.hMul (f x) (μ'.rnDeriv ν x))
    hx1 : Eq (μ'.rnDeriv ν x) (μ.rnDeriv ν x)
    hx2 : Eq ((μ'.withDensity f).rnDeriv ν x) ((μ.withDensity f).rnDeriv ν x)
    ⊢ Eq ((μ.withDensity f).rnDeriv ν x) (HMul.hMul (f x) (μ.rnDeriv ν x))
  -/
  rw [← hx2, hx, hx1]
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `rnDeriv_withDensity_right`. -/
lemma rnDeriv_withDensity_right_of_absolutelyContinuous {ν : Measure α}
    [HaveLebesgueDecomposition μ ν] [SigmaFinite ν] (hμν : μ ≪ ν) (hf : AEMeasurable f ν)
    (hf_ne_zero : ∀ᵐ x ∂ν, f x ≠ 0) (hf_ne_top : ∀ᵐ x ∂ν, f x ≠ ∞) :
    μ.rnDeriv (ν.withDensity f) =ᵐ[ν] fun x ↦ (f x)⁻¹ * μ.rnDeriv ν x := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (ν.withDensity f)) fun x => HMu …
  -/
  have : SigmaFinite (ν.withDensity f) := SigmaFinite.withDensity_of_ne_top hf_ne_top
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    this : MeasureTheory.SigmaFinite (ν.withDensity f)
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (ν.withDensity f)) fun x => HMu …
  -/
  refine (withDensity_absolutelyContinuous' hf hf_ne_zero).ae_eq ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    this : MeasureTheory.SigmaFinite (ν.withDensity f)
    ⊢ (MeasureTheory.ae (ν.withDensity f)).EventuallyEq (μ.rnDeriv (ν.withDensity  …
  -/
  refine (Measure.eq_rnDeriv₀ (ν := ν.withDensity f) ?_ Measure.MutuallySingular.zero_left ?_).symm
  · exact (hf.inv.mono_ac (withDensity_absolutelyContinuous _ _)).mul
      (Measure.measurable_rnDeriv _ _).aemeasurable
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : μ.HaveLebesgueDecomposition ν
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
      hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
      this : MeasureTheory.SigmaFinite (ν.withDensity f)
      ⊢ Eq μ (HAdd.hAdd 0 ((ν.withDensity f).withDensity fun x => HMul.hMul (Inv.inv …
    -/
  · ext1 s hs
    /-
      case refine_2.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : μ.HaveLebesgueDecomposition ν
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
      hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
      this : MeasureTheory.SigmaFinite (ν.withDensity f)
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (μ s) ((HAdd.hAdd 0 ((ν.withDensity f).withDensity fun x => HMul.hMul (In …
    -/
    conv_lhs => rw [← Measure.withDensity_rnDeriv_eq _ _ hμν]
    /-
      case refine_2.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : μ.HaveLebesgueDecomposition ν
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
      hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
      this : MeasureTheory.SigmaFinite (ν.withDensity f)
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq ((ν.withDensity (μ.rnDeriv ν)) s) ((HAdd.hAdd 0 ((ν.withDensity f).withDe …
    -/
    rw [zero_add, withDensity_apply _ hs, withDensity_apply _ hs]
    /-
      case refine_2.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ν : MeasureTheory.Measure α
      inst✝¹ : μ.HaveLebesgueDecomposition ν
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      hf : AEMeasurable f ν
      hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
      hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
      this : MeasureTheory.SigmaFinite (ν.withDensity f)
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun a => μ.rnDeriv ν a) (MeasureT …
    -/
    rw [setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀ _ _ _ hs]
      /-
        case refine_2.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : μ.HaveLebesgueDecomposition ν
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
        hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
        this : MeasureTheory.SigmaFinite (ν.withDensity f)
        s : Set α
        hs : MeasurableSet s
        ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun a => μ.rnDeriv ν a) (MeasureT …
      -/
    · simp only [Pi.mul_apply]
      have : (fun a ↦ f a * ((f a)⁻¹ * μ.rnDeriv ν a)) =ᵐ[ν] μ.rnDeriv ν := by
        filter_upwards [hf_ne_zero, hf_ne_top] with x hx1 hx2
        simp [← mul_assoc, ENNReal.mul_inv_cancel, hx1, hx2]
      /-
        case refine_2.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : μ.HaveLebesgueDecomposition ν
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
        hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
        this✝ : MeasureTheory.SigmaFinite (ν.withDensity f)
        s : Set α
        hs : MeasurableSet s
        this : (MeasureTheory.ae ν).EventuallyEq (fun a => HMul.hMul (f a) (HMul.hMul  …
        ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun a => μ.rnDeriv ν a) (MeasureT …
      -/
      rw [lintegral_congr_ae (ae_restrict_of_ae this)]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : μ.HaveLebesgueDecomposition ν
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
        hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
        this : MeasureTheory.SigmaFinite (ν.withDensity f)
        s : Set α
        hs : MeasurableSet s
        ⊢ Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae (ν.restri …
      -/
    · refine ae_restrict_of_ae ?_
      /-
        case refine_2.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : μ.HaveLebesgueDecomposition ν
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
        hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
        this : MeasureTheory.SigmaFinite (ν.withDensity f)
        s : Set α
        hs : MeasurableSet s
        ⊢ Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae ν)
      -/
      filter_upwards [hf_ne_top] with x hx using hx.lt_top
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        ν : MeasureTheory.Measure α
        inst✝¹ : μ.HaveLebesgueDecomposition ν
        inst✝ : MeasureTheory.SigmaFinite ν
        hμν : μ.AbsolutelyContinuous ν
        hf : AEMeasurable f ν
        hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
        hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
        this : MeasureTheory.SigmaFinite (ν.withDensity f)
        s : Set α
        hs : MeasurableSet s
        ⊢ AEMeasurable f (ν.restrict s)
      -/
    · exact hf.restrict
      /-
        🎉 no goals
      -/


lemma rnDeriv_withDensity_right (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν]
    (hf : AEMeasurable f ν) (hf_ne_zero : ∀ᵐ x ∂ν, f x ≠ 0) (hf_ne_top : ∀ᵐ x ∂ν, f x ≠ ∞) :
    μ.rnDeriv (ν.withDensity f) =ᵐ[ν] fun x ↦ (f x)⁻¹ * μ.rnDeriv ν x := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (ν.withDensity f)) fun x => HMu …
  -/
  let μ' := ν.withDensity (μ.rnDeriv ν)
  have h₁ : μ'.rnDeriv (ν.withDensity f) =ᵐ[ν] μ.rnDeriv (ν.withDensity f) :=
    rnDeriv_withDensity_withDensity_rnDeriv_right μ ν hf hf_ne_zero hf_ne_top
  have h₂ : μ.rnDeriv ν =ᵐ[ν] μ'.rnDeriv ν :=
    (Measure.rnDeriv_withDensity _ (Measure.measurable_rnDeriv _ _)).symm
  have hμ' := rnDeriv_withDensity_right_of_absolutelyContinuous
    (withDensity_absolutelyContinuous ν (μ.rnDeriv ν)) hf hf_ne_zero hf_ne_top
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    μ' : MeasureTheory.Measure α := ν.withDensity (μ.rnDeriv ν)
    h₁ : (MeasureTheory.ae ν).EventuallyEq (μ'.rnDeriv (ν.withDensity f)) (μ.rnDer …
    h₂ : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν) (μ'.rnDeriv ν)
    hμ' : (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv …
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (ν.withDensity f)) fun x => HMu …
  -/
  filter_upwards [h₁, h₂, hμ'] with x hx₁ hx₂ hx_eq
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : AEMeasurable f ν
    hf_ne_zero : Filter.Eventually (fun x => Ne (f x) 0) (MeasureTheory.ae ν)
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae ν)
    μ' : MeasureTheory.Measure α := ν.withDensity (μ.rnDeriv ν)
    h₁ : (MeasureTheory.ae ν).EventuallyEq (μ'.rnDeriv (ν.withDensity f)) (μ.rnDer …
    h₂ : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν) (μ'.rnDeriv ν)
    hμ' : (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv …
    x : α
    hx₁ : Eq (μ'.rnDeriv (ν.withDensity f) x) (μ.rnDeriv (ν.withDensity f) x)
    hx₂ : Eq (μ.rnDeriv ν x) (μ'.rnDeriv ν x)
    hx_eq : Eq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv (ν.withDensity f) x) (HMul.h …
    ⊢ Eq (μ.rnDeriv (ν.withDensity f) x) (HMul.hMul (Inv.inv (f x)) (μ.rnDeriv ν x))
  -/
  rw [← hx₁, hx₂, hx_eq]
  /-
    🎉 no goals
  -/


lemma rnDeriv_eq_zero_of_mutuallySingular {ν' : Measure α} [HaveLebesgueDecomposition μ ν']
    [SigmaFinite ν'] (h : μ ⟂ₘ ν) (hνν' : ν ≪ ν') :
    μ.rnDeriv ν' =ᵐ[ν] 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν'
    inst✝ : MeasureTheory.SigmaFinite ν'
    h : μ.MutuallySingular ν
    hνν' : ν.AbsolutelyContinuous ν'
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν') 0
  -/
  let t := h.nullSet
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν'
    inst✝ : MeasureTheory.SigmaFinite ν'
    h : μ.MutuallySingular ν
    hνν' : ν.AbsolutelyContinuous ν'
    t : Set α := h.nullSet
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν') 0
  -/
  have ht : MeasurableSet t := h.measurableSet_nullSet
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν'
    inst✝ : MeasureTheory.SigmaFinite ν'
    h : μ.MutuallySingular ν
    hνν' : ν.AbsolutelyContinuous ν'
    t : Set α := h.nullSet
    ht : MeasurableSet t
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν') 0
  -/
  refine ae_of_ae_restrict_of_ae_restrict_compl t ?_ (by simp [t])
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν'
    inst✝ : MeasureTheory.SigmaFinite ν'
    h : μ.MutuallySingular ν
    hνν' : ν.AbsolutelyContinuous ν'
    t : Set α := h.nullSet
    ht : MeasurableSet t
    ⊢ Filter.Eventually (fun x => Eq (μ.rnDeriv ν' x) (0 x)) (MeasureTheory.ae (ν. …
  -/
  change μ.rnDeriv ν' =ᵐ[ν.restrict t] 0
  have : μ.rnDeriv ν' =ᵐ[ν.restrict t] (μ.restrict t).rnDeriv ν' := by
    have h : (μ.restrict t).rnDeriv ν' =ᵐ[ν] t.indicator (μ.rnDeriv ν') :=
      hνν'.ae_le (rnDeriv_restrict μ ν' ht)
    rw [Filter.EventuallyEq, ae_restrict_iff' ht]
    filter_upwards [h] with x hx hxt
    rw [hx, Set.indicator_of_mem hxt]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν'
    inst✝ : MeasureTheory.SigmaFinite ν'
    h : μ.MutuallySingular ν
    hνν' : ν.AbsolutelyContinuous ν'
    t : Set α := h.nullSet
    ht : MeasurableSet t
    this : (MeasureTheory.ae (ν.restrict t)).EventuallyEq (μ.rnDeriv ν') ((μ.restr …
    ⊢ (MeasureTheory.ae (ν.restrict t)).EventuallyEq (μ.rnDeriv ν') 0
  -/
  refine this.trans ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν'
    inst✝ : MeasureTheory.SigmaFinite ν'
    h : μ.MutuallySingular ν
    hνν' : ν.AbsolutelyContinuous ν'
    t : Set α := h.nullSet
    ht : MeasurableSet t
    this : (MeasureTheory.ae (ν.restrict t)).EventuallyEq (μ.rnDeriv ν') ((μ.restr …
    ⊢ (MeasureTheory.ae (ν.restrict t)).EventuallyEq ((μ.restrict t).rnDeriv ν') 0
  -/
  simp only [t, MutuallySingular.restrict_nullSet]
  suffices (0 : Measure α).rnDeriv ν' =ᵐ[ν'] 0 by
    have h_ac' : ν.restrict t ≪ ν' := restrict_le_self.absolutelyContinuous.trans hνν'
    exact h_ac'.ae_le this
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν'
    inst✝ : MeasureTheory.SigmaFinite ν'
    h : μ.MutuallySingular ν
    hνν' : ν.AbsolutelyContinuous ν'
    t : Set α := h.nullSet
    ht : MeasurableSet t
    this : (MeasureTheory.ae (ν.restrict t)).EventuallyEq (μ.rnDeriv ν') ((μ.restr …
    ⊢ (MeasureTheory.ae ν').EventuallyEq (MeasureTheory.Measure.rnDeriv 0 ν') 0
  -/
  exact rnDeriv_zero _
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `rnDeriv_add_right_of_mutuallySingular`. -/
lemma rnDeriv_add_right_of_absolutelyContinuous_of_mutuallySingular {ν' : Measure α}
    [HaveLebesgueDecomposition μ ν] [HaveLebesgueDecomposition μ (ν + ν')] [SigmaFinite ν]
    (hμν : μ ≪ ν) (hνν' : ν ⟂ₘ ν') :
    μ.rnDeriv (ν + ν') =ᵐ[ν] μ.rnDeriv ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : μ.HaveLebesgueDecomposition ν
    inst✝¹ : μ.HaveLebesgueDecomposition (HAdd.hAdd ν ν')
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hνν' : ν.MutuallySingular ν'
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HAdd.hAdd ν ν')) (μ.rnDeriv ν)
  -/
  let t := hνν'.nullSet
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : μ.HaveLebesgueDecomposition ν
    inst✝¹ : μ.HaveLebesgueDecomposition (HAdd.hAdd ν ν')
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hνν' : ν.MutuallySingular ν'
    t : Set α := hνν'.nullSet
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HAdd.hAdd ν ν')) (μ.rnDeriv ν)
  -/
  have ht : MeasurableSet t := hνν'.measurableSet_nullSet
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : μ.HaveLebesgueDecomposition ν
    inst✝¹ : μ.HaveLebesgueDecomposition (HAdd.hAdd ν ν')
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hνν' : ν.MutuallySingular ν'
    t : Set α := hνν'.nullSet
    ht : MeasurableSet t
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HAdd.hAdd ν ν')) (μ.rnDeriv ν)
  -/
  refine ae_of_ae_restrict_of_ae_restrict_compl t (by simp [t]) ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : μ.HaveLebesgueDecomposition ν
    inst✝¹ : μ.HaveLebesgueDecomposition (HAdd.hAdd ν ν')
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hνν' : ν.MutuallySingular ν'
    t : Set α := hνν'.nullSet
    ht : MeasurableSet t
    ⊢ Filter.Eventually (fun x => Eq (μ.rnDeriv (HAdd.hAdd ν ν') x) (μ.rnDeriv ν x …
  -/
  change μ.rnDeriv (ν + ν') =ᵐ[ν.restrict tᶜ] μ.rnDeriv ν
  rw [← withDensity_eq_iff_of_sigmaFinite (μ := ν.restrict tᶜ)
    (Measure.measurable_rnDeriv _ _).aemeasurable (Measure.measurable_rnDeriv _ _).aemeasurable]
  have : (ν.restrict tᶜ).withDensity (μ.rnDeriv (ν + ν'))
      = ((ν + ν').restrict tᶜ).withDensity (μ.rnDeriv (ν + ν')) := by simp [t]
  rw [this, ← restrict_withDensity ht.compl, ← restrict_withDensity ht.compl,
      Measure.withDensity_rnDeriv_eq _ _ (hμν.add_right ν'), Measure.withDensity_rnDeriv_eq _ _ hμν]


/-- Auxiliary lemma for `rnDeriv_add_right_of_mutuallySingular`. -/
lemma rnDeriv_add_right_of_mutuallySingular' {ν' : Measure α}
    [SigmaFinite μ] [SigmaFinite ν] [SigmaFinite ν']
    (hμν' : μ ⟂ₘ ν') (hνν' : ν ⟂ₘ ν') :
    μ.rnDeriv (ν + ν') =ᵐ[ν] μ.rnDeriv ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HAdd.hAdd ν ν')) (μ.rnDeriv ν)
  -/
  have h_ac : ν ≪ ν + ν' := Measure.AbsolutelyContinuous.rfl.add_right _
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HAdd.hAdd ν ν')) (μ.rnDeriv ν)
  -/
  rw [haveLebesgueDecomposition_add μ ν]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν) (ν.withDens …
  -/
  have h₁ := rnDeriv_add' (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)) (ν + ν')
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν) (ν.withDens …
  -/
  have h₂ := rnDeriv_add' (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)) ν
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν) (ν.withD …
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν) (ν.withDens …
  -/
  refine (Filter.EventuallyEq.trans (h_ac.ae_le h₁) ?_).trans h₂.symm
  have h₃ := rnDeriv_add_right_of_absolutelyContinuous_of_mutuallySingular
    (withDensity_absolutelyContinuous ν (μ.rnDeriv ν)) hνν'
  have h₄ : (μ.singularPart ν).rnDeriv (ν + ν') =ᵐ[ν] 0 := by
    refine h_ac.ae_eq ?_
    simp only [rnDeriv_eq_zero, MutuallySingular.add_right_iff]
    exact ⟨mutuallySingular_singularPart μ ν, hμν'.singularPart ν⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν) (ν.withD …
    h₃ : (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv  …
    h₄ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv (HAdd.hAdd  …
    ⊢ (MeasureTheory.ae ν).EventuallyEq (HAdd.hAdd ((μ.singularPart ν).rnDeriv (HA …
  -/
  have h₅ : (μ.singularPart ν).rnDeriv ν =ᵐ[ν] 0 := rnDeriv_singularPart μ ν
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν) (ν.withD …
    h₃ : (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv  …
    h₄ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv (HAdd.hAdd  …
    h₅ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv ν) 0
    ⊢ (MeasureTheory.ae ν).EventuallyEq (HAdd.hAdd ((μ.singularPart ν).rnDeriv (HA …
  -/
  filter_upwards [h₃, h₄, h₅] with x hx₃ hx₄ hx₅
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν) (ν.withD …
    h₃ : (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv  …
    h₄ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv (HAdd.hAdd  …
    h₅ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv ν) 0
    x : α
    hx₃ : Eq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv (HAdd.hAdd ν ν') x) ((ν.withDe …
    hx₄ : Eq ((μ.singularPart ν).rnDeriv (HAdd.hAdd ν ν') x) (0 x)
    hx₅ : Eq ((μ.singularPart ν).rnDeriv ν x) (0 x)
    ⊢ Eq (HAdd.hAdd ((μ.singularPart ν).rnDeriv (HAdd.hAdd ν ν')) ((ν.withDensity  …
  -/
  simp only [Pi.add_apply]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hμν' : μ.MutuallySingular ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν) (ν.withD …
    h₃ : (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv  …
    h₄ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv (HAdd.hAdd  …
    h₅ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv ν) 0
    x : α
    hx₃ : Eq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv (HAdd.hAdd ν ν') x) ((ν.withDe …
    hx₄ : Eq ((μ.singularPart ν).rnDeriv (HAdd.hAdd ν ν') x) (0 x)
    hx₅ : Eq ((μ.singularPart ν).rnDeriv ν x) (0 x)
    ⊢ Eq (HAdd.hAdd ((μ.singularPart ν).rnDeriv (HAdd.hAdd ν ν') x) ((ν.withDensit …
  -/
  rw [hx₃, hx₄, hx₅]
  /-
    🎉 no goals
  -/


lemma rnDeriv_add_right_of_mutuallySingular {ν' : Measure α}
    [SigmaFinite μ] [SigmaFinite ν] [SigmaFinite ν'] (hνν' : ν ⟂ₘ ν') :
    μ.rnDeriv (ν + ν') =ᵐ[ν] μ.rnDeriv ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hνν' : ν.MutuallySingular ν'
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HAdd.hAdd ν ν')) (μ.rnDeriv ν)
  -/
  have h_ac : ν ≪ ν + ν' := Measure.AbsolutelyContinuous.rfl.add_right _
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    ⊢ (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HAdd.hAdd ν ν')) (μ.rnDeriv ν)
  -/
  rw [haveLebesgueDecomposition_add μ ν']
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν') (ν'.withDe …
  -/
  have h₁ := rnDeriv_add' (μ.singularPart ν') (ν'.withDensity (μ.rnDeriv ν')) (ν + ν')
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν') (ν'.withDe …
  -/
  have h₂ := rnDeriv_add' (μ.singularPart ν') (ν'.withDensity (μ.rnDeriv ν')) ν
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν') (ν'.wit …
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν') (ν'.withDe …
  -/
  refine (Filter.EventuallyEq.trans (h_ac.ae_le h₁) ?_).trans h₂.symm
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν') (ν'.wit …
    ⊢ (MeasureTheory.ae ν).EventuallyEq (HAdd.hAdd ((μ.singularPart ν').rnDeriv (H …
  -/
  have h₃ := rnDeriv_add_right_of_mutuallySingular' (?_ : μ.singularPart ν' ⟂ₘ ν') hνν'
  · have h₄ : (ν'.withDensity (rnDeriv μ ν')).rnDeriv (ν + ν') =ᵐ[ν] 0 := by
      refine rnDeriv_eq_zero_of_mutuallySingular ?_ h_ac
      exact hνν'.symm.withDensity
    have h₅ : (ν'.withDensity (rnDeriv μ ν')).rnDeriv ν =ᵐ[ν] 0 := by
      rw [rnDeriv_eq_zero]
      exact hνν'.symm.withDensity
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν ν' : MeasureTheory.Measure α
      inst✝² : MeasureTheory.SigmaFinite μ
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite ν'
      hνν' : ν.MutuallySingular ν'
      h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
      h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
      h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν') (ν'.wit …
      h₃ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν').rnDeriv (HAdd.hAdd …
      h₄ : (MeasureTheory.ae ν).EventuallyEq ((ν'.withDensity (μ.rnDeriv ν')).rnDeri …
      h₅ : (MeasureTheory.ae ν).EventuallyEq ((ν'.withDensity (μ.rnDeriv ν')).rnDeri …
      ⊢ (MeasureTheory.ae ν).EventuallyEq (HAdd.hAdd ((μ.singularPart ν').rnDeriv (H …
    -/
    filter_upwards [h₃, h₄, h₅] with x hx₃ hx₄ hx₅
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ ν ν' : MeasureTheory.Measure α
      inst✝² : MeasureTheory.SigmaFinite μ
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite ν'
      hνν' : ν.MutuallySingular ν'
      h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
      h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
      h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν') (ν'.wit …
      h₃ : (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν').rnDeriv (HAdd.hAdd …
      h₄ : (MeasureTheory.ae ν).EventuallyEq ((ν'.withDensity (μ.rnDeriv ν')).rnDeri …
      h₅ : (MeasureTheory.ae ν).EventuallyEq ((ν'.withDensity (μ.rnDeriv ν')).rnDeri …
      x : α
      hx₃ : Eq ((μ.singularPart ν').rnDeriv (HAdd.hAdd ν ν') x) ((μ.singularPart ν') …
      hx₄ : Eq ((ν'.withDensity (μ.rnDeriv ν')).rnDeriv (HAdd.hAdd ν ν') x) (0 x)
      hx₅ : Eq ((ν'.withDensity (μ.rnDeriv ν')).rnDeriv ν x) (0 x)
      ⊢ Eq (HAdd.hAdd ((μ.singularPart ν').rnDeriv (HAdd.hAdd ν ν')) ((ν'.withDensit …
    -/
    rw [Pi.add_apply, Pi.add_apply, hx₃, hx₄, hx₅]
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    α : Type u_1
    m : MeasurableSpace α
    μ ν ν' : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite ν'
    hνν' : ν.MutuallySingular ν'
    h_ac : ν.AbsolutelyContinuous (HAdd.hAdd ν ν')
    h₁ : (MeasureTheory.ae (HAdd.hAdd ν ν')).EventuallyEq ((HAdd.hAdd (μ.singularP …
    h₂ : (MeasureTheory.ae ν).EventuallyEq ((HAdd.hAdd (μ.singularPart ν') (ν'.wit …
    ⊢ (μ.singularPart ν').MutuallySingular ν'
  -/
  exact mutuallySingular_singularPart μ ν'
  /-
    🎉 no goals
  -/


lemma rnDeriv_withDensity_rnDeriv [SigmaFinite μ] [SigmaFinite ν] (hμν : μ ≪ ν) :
    μ.rnDeriv (μ.withDensity (ν.rnDeriv μ)) =ᵐ[μ] μ.rnDeriv ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.ae μ).EventuallyEq (μ.rnDeriv (μ.withDensity (ν.rnDeriv μ)))  …
  -/
  conv_rhs => rw [ν.haveLebesgueDecomposition_add μ, add_comm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.ae μ).EventuallyEq (μ.rnDeriv (μ.withDensity (ν.rnDeriv μ)))  …
  -/
  refine (absolutelyContinuous_withDensity_rnDeriv hμν).ae_eq ?_
  exact (rnDeriv_add_right_of_mutuallySingular
    (Measure.mutuallySingular_singularPart ν μ).symm.withDensity).symm


/-- Auxiliary lemma for `inv_rnDeriv`. -/
lemma inv_rnDeriv_aux [HaveLebesgueDecomposition μ ν] [HaveLebesgueDecomposition ν μ]
    [SigmaFinite μ] (hμν : μ ≪ ν) (hνμ : ν ≪ μ) :
    (μ.rnDeriv ν)⁻¹ =ᵐ[μ] ν.rnDeriv μ := by
  suffices μ.withDensity (μ.rnDeriv ν)⁻¹ = μ.withDensity (ν.rnDeriv μ) by
    calc (μ.rnDeriv ν)⁻¹ =ᵐ[μ] (μ.withDensity (μ.rnDeriv ν)⁻¹).rnDeriv μ :=
          (rnDeriv_withDensity _ (measurable_rnDeriv _ _).inv).symm
    _ = (μ.withDensity (ν.rnDeriv μ)).rnDeriv μ := by rw [this]
    _ =ᵐ[μ] ν.rnDeriv μ := rnDeriv_withDensity _ (measurable_rnDeriv _ _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝² : μ.HaveLebesgueDecomposition ν
    inst✝¹ : ν.HaveLebesgueDecomposition μ
    inst✝ : MeasureTheory.SigmaFinite μ
    hμν : μ.AbsolutelyContinuous ν
    hνμ : ν.AbsolutelyContinuous μ
    ⊢ Eq (μ.withDensity (Inv.inv (μ.rnDeriv ν))) (μ.withDensity (ν.rnDeriv μ))
  -/
  rw [withDensity_rnDeriv_eq _ _ hνμ, ← withDensity_rnDeriv_eq _ _ hμν]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝² : μ.HaveLebesgueDecomposition ν
    inst✝¹ : ν.HaveLebesgueDecomposition μ
    inst✝ : MeasureTheory.SigmaFinite μ
    hμν : μ.AbsolutelyContinuous ν
    hνμ : ν.AbsolutelyContinuous μ
    ⊢ Eq ((ν.withDensity (μ.rnDeriv ν)).withDensity (Inv.inv ((ν.withDensity (μ.rn …
  -/
  conv in ((ν.withDensity (μ.rnDeriv ν)).rnDeriv ν)⁻¹ => rw [withDensity_rnDeriv_eq _ _ hμν]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝² : μ.HaveLebesgueDecomposition ν
    inst✝¹ : ν.HaveLebesgueDecomposition μ
    inst✝ : MeasureTheory.SigmaFinite μ
    hμν : μ.AbsolutelyContinuous ν
    hνμ : ν.AbsolutelyContinuous μ
    ⊢ Eq ((ν.withDensity (μ.rnDeriv ν)).withDensity (Inv.inv (μ.rnDeriv ν))) ν
  -/
  change (ν.withDensity (μ.rnDeriv ν)).withDensity (fun x ↦ (μ.rnDeriv ν x)⁻¹) = ν
  rw [withDensity_inv_same (measurable_rnDeriv _ _)
    (by filter_upwards [hνμ.ae_le (rnDeriv_pos hμν)] with x hx using hx.ne')
    (rnDeriv_ne_top _ _)]


lemma inv_rnDeriv [SigmaFinite μ] [SigmaFinite ν] (hμν : μ ≪ ν) :
    (μ.rnDeriv ν)⁻¹ =ᵐ[μ] ν.rnDeriv μ := by
  suffices (μ.rnDeriv ν)⁻¹ =ᵐ[μ] (μ.rnDeriv (μ.withDensity (ν.rnDeriv μ)))⁻¹
      ∧ ν.rnDeriv μ =ᵐ[μ] (μ.withDensity (ν.rnDeriv μ)).rnDeriv μ by
    refine (this.1.trans (Filter.EventuallyEq.trans ?_ this.2.symm))
    exact Measure.inv_rnDeriv_aux (absolutelyContinuous_withDensity_rnDeriv hμν)
      (withDensity_absolutelyContinuous _ _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ And ((MeasureTheory.ae μ).EventuallyEq (Inv.inv (μ.rnDeriv ν)) (Inv.inv (μ.r …
  -/
  constructor
    /-
      case left
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Inv.inv (μ.rnDeriv ν)) (Inv.inv (μ.rnDeri …
    -/
  · filter_upwards [rnDeriv_withDensity_rnDeriv hμν] with x hx
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      x : α
      hx : Eq (μ.rnDeriv (μ.withDensity (ν.rnDeriv μ)) x) (μ.rnDeriv ν x)
      ⊢ Eq (Inv.inv (μ.rnDeriv ν) x) (Inv.inv (μ.rnDeriv (μ.withDensity (ν.rnDeriv μ …
    -/
    simp only [Pi.inv_apply, inv_inj]
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      x : α
      hx : Eq (μ.rnDeriv (μ.withDensity (ν.rnDeriv μ)) x) (μ.rnDeriv ν x)
      ⊢ Eq (μ.rnDeriv ν x) (μ.rnDeriv (μ.withDensity (ν.rnDeriv μ)) x)
    -/
    exact hx.symm
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hμν : μ.AbsolutelyContinuous ν
      ⊢ (MeasureTheory.ae μ).EventuallyEq (ν.rnDeriv μ) ((μ.withDensity (ν.rnDeriv μ …
    -/
  · exact (Measure.rnDeriv_withDensity μ (Measure.measurable_rnDeriv ν μ)).symm
    /-
      🎉 no goals
    -/


lemma inv_rnDeriv' [SigmaFinite μ] [SigmaFinite ν] (hμν : μ ≪ ν) :
    (ν.rnDeriv μ)⁻¹ =ᵐ[μ] μ.rnDeriv ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Inv.inv (ν.rnDeriv μ)) (μ.rnDeriv ν)
  -/
  filter_upwards [inv_rnDeriv hμν] with x hx; simp only [Pi.inv_apply, ← hx, inv_inv]
                                              /-
                                                🎉 no goals
                                              -/


lemma setLIntegral_rnDeriv_le (s : Set α) :
    ∫⁻ x in s, μ.rnDeriv ν x ∂ν ≤ μ s :=
  (withDensity_apply_le _ _).trans (Measure.le_iff'.1 (withDensity_rnDeriv_le μ ν) s)


@[deprecated (since := "2024-06-29")]
alias set_lintegral_rnDeriv_le := setLIntegral_rnDeriv_le


lemma lintegral_rnDeriv_le : ∫⁻ x, μ.rnDeriv ν x ∂ν ≤ μ Set.univ :=
  (setLIntegral_univ _).symm ▸ Measure.setLIntegral_rnDeriv_le Set.univ


lemma setLIntegral_rnDeriv' [HaveLebesgueDecomposition μ ν] (hμν : μ ≪ ν) {s : Set α}
    (hs : MeasurableSet s) :
    ∫⁻ x in s, μ.rnDeriv ν x ∂ν = μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun x => μ.rnDeriv ν x) (μ s)
  -/
  rw [← withDensity_apply _ hs, Measure.withDensity_rnDeriv_eq _ _ hμν]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_rnDeriv' := setLIntegral_rnDeriv'


lemma setLIntegral_rnDeriv [HaveLebesgueDecomposition μ ν] [SFinite ν]
    (hμν : μ ≪ ν) (s : Set α) :
    ∫⁻ x in s, μ.rnDeriv ν x ∂ν = μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SFinite ν
    hμν : μ.AbsolutelyContinuous ν
    s : Set α
    ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun x => μ.rnDeriv ν x) (μ s)
  -/
  rw [← withDensity_apply' _ s, Measure.withDensity_rnDeriv_eq _ _ hμν]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_rnDeriv := setLIntegral_rnDeriv


lemma lintegral_rnDeriv [HaveLebesgueDecomposition μ ν] (hμν : μ ≪ ν) :
    ∫⁻ x, μ.rnDeriv ν x ∂ν = μ Set.univ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Eq (MeasureTheory.lintegral ν fun x => μ.rnDeriv ν x) (μ Set.univ)
  -/
  rw [← setLIntegral_univ, setLIntegral_rnDeriv' hμν MeasurableSet.univ]
  /-
    🎉 no goals
  -/


lemma integrableOn_toReal_rnDeriv {s : Set α} (hμs : μ s ≠ ∞) :
    IntegrableOn (fun x ↦ (μ.rnDeriv ν x).toReal) s ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hμs : Ne (μ s) Top.top
    ⊢ MeasureTheory.IntegrableOn (fun x => (μ.rnDeriv ν x).toReal) s ν
  -/
  refine integrable_toReal_of_lintegral_ne_top (Measure.measurable_rnDeriv _ _).aemeasurable ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hμs : Ne (μ s) Top.top
    ⊢ Ne (MeasureTheory.lintegral (ν.restrict s) fun x => μ.rnDeriv ν x) Top.top
  -/
  exact ((setLIntegral_rnDeriv_le _).trans_lt hμs.lt_top).ne
  /-
    🎉 no goals
  -/


lemma setIntegral_toReal_rnDeriv_eq_withDensity' [SigmaFinite μ]
    {s : Set α} (hs : MeasurableSet s) :
    ∫ x in s, (μ.rnDeriv ν x).toReal ∂ν = (ν.withDensity (μ.rnDeriv ν) s).toReal := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (ν.restrict s) fun x => (μ.rnDeriv ν x).toReal) ( …
  -/
  rw [integral_toReal (Measure.measurable_rnDeriv _ _).aemeasurable]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun a => μ.rnDeriv ν a).toReal (( …
    -/
  · rw [ENNReal.toReal_eq_toReal_iff, ← withDensity_apply _ hs]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      s : Set α
      hs : MeasurableSet s
      ⊢ Or (Eq ((ν.withDensity (μ.rnDeriv ν)) s) ((ν.withDensity (μ.rnDeriv ν)) s))  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      s : Set α
      hs : MeasurableSet s
      ⊢ Filter.Eventually (fun x => LT.lt (μ.rnDeriv ν x) Top.top) (MeasureTheory.ae …
    -/
  · exact ae_restrict_of_ae (Measure.rnDeriv_lt_top _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_toReal_rnDeriv_eq_withDensity' := setIntegral_toReal_rnDeriv_eq_withDensity'


lemma setIntegral_toReal_rnDeriv_eq_withDensity [SigmaFinite μ] [SFinite ν] (s : Set α) :
    ∫ x in s, (μ.rnDeriv ν x).toReal ∂ν = (ν.withDensity (μ.rnDeriv ν) s).toReal := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    s : Set α
    ⊢ Eq (MeasureTheory.integral (ν.restrict s) fun x => (μ.rnDeriv ν x).toReal) ( …
  -/
  rw [integral_toReal (Measure.measurable_rnDeriv _ _).aemeasurable]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SFinite ν
      s : Set α
      ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun a => μ.rnDeriv ν a).toReal (( …
    -/
  · rw [ENNReal.toReal_eq_toReal_iff, ← withDensity_apply' _ s]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SFinite ν
      s : Set α
      ⊢ Or (Eq ((ν.withDensity (μ.rnDeriv ν)) s) ((ν.withDensity (μ.rnDeriv ν)) s))  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SFinite ν
      s : Set α
      ⊢ Filter.Eventually (fun x => LT.lt (μ.rnDeriv ν x) Top.top) (MeasureTheory.ae …
    -/
  · exact ae_restrict_of_ae (Measure.rnDeriv_lt_top _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_toReal_rnDeriv_eq_withDensity := setIntegral_toReal_rnDeriv_eq_withDensity


lemma setIntegral_toReal_rnDeriv_le [SigmaFinite μ] {s : Set α} (hμs : μ s ≠ ∞) :
    ∫ x in s, (μ.rnDeriv ν x).toReal ∂ν ≤ (μ s).toReal := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hμs : Ne (μ s) Top.top
    ⊢ LE.le (MeasureTheory.integral (ν.restrict s) fun x => (μ.rnDeriv ν x).toReal …
  -/
  set t := toMeasurable μ s with ht
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hμs : Ne (μ s) Top.top
    t : Set α := MeasureTheory.toMeasurable μ s
    ht : Eq t (MeasureTheory.toMeasurable μ s)
    ⊢ LE.le (MeasureTheory.integral (ν.restrict s) fun x => (μ.rnDeriv ν x).toReal …
  -/
  have ht_m : MeasurableSet t := measurableSet_toMeasurable μ s
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    hμs : Ne (μ s) Top.top
    t : Set α := MeasureTheory.toMeasurable μ s
    ht : Eq t (MeasureTheory.toMeasurable μ s)
    ht_m : MeasurableSet t
    ⊢ LE.le (MeasureTheory.integral (ν.restrict s) fun x => (μ.rnDeriv ν x).toReal …
  -/
  have hμt : μ t ≠ ∞ := by rwa [ht, measure_toMeasurable s]
  calc ∫ x in s, (μ.rnDeriv ν x).toReal ∂ν
    ≤ ∫ x in t, (μ.rnDeriv ν x).toReal ∂ν := by
        refine setIntegral_mono_set ?_ ?_ (HasSubset.Subset.eventuallyLE (subset_toMeasurable _ _))
        · exact integrableOn_toReal_rnDeriv hμt
        · exact ae_of_all _ (by simp)
  _ = (withDensity ν (rnDeriv μ ν) t).toReal := setIntegral_toReal_rnDeriv_eq_withDensity' ht_m
  _ ≤ (μ t).toReal := by
        gcongr
        · exact hμt
        · apply withDensity_rnDeriv_le
  _ = (μ s).toReal := by rw [measure_toMeasurable s]


@[deprecated (since := "2024-04-17")]
alias set_integral_toReal_rnDeriv_le := setIntegral_toReal_rnDeriv_le


lemma setIntegral_toReal_rnDeriv' [SigmaFinite μ] [HaveLebesgueDecomposition μ ν]
    (hμν : μ ≪ ν) {s : Set α} (hs : MeasurableSet s) :
    ∫ x in s, (μ.rnDeriv ν x).toReal ∂ν = (μ s).toReal := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (ν.restrict s) fun x => (μ.rnDeriv ν x).toReal) ( …
  -/
  rw [setIntegral_toReal_rnDeriv_eq_withDensity' hs, Measure.withDensity_rnDeriv_eq _ _ hμν]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_toReal_rnDeriv' := setIntegral_toReal_rnDeriv'


lemma setIntegral_toReal_rnDeriv [SigmaFinite μ] [SigmaFinite ν] (hμν : μ ≪ ν) (s : Set α) :
    ∫ x in s, (μ.rnDeriv ν x).toReal ∂ν = (μ s).toReal := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    s : Set α
    ⊢ Eq (MeasureTheory.integral (ν.restrict s) fun x => (μ.rnDeriv ν x).toReal) ( …
  -/
  rw [setIntegral_toReal_rnDeriv_eq_withDensity s, Measure.withDensity_rnDeriv_eq _ _ hμν]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_toReal_rnDeriv := setIntegral_toReal_rnDeriv


lemma integral_toReal_rnDeriv [SigmaFinite μ] [SigmaFinite ν] (hμν : μ ≪ ν) :
    ∫ x, (μ.rnDeriv ν x).toReal ∂ν = (μ Set.univ).toReal := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Eq (MeasureTheory.integral ν fun x => (μ.rnDeriv ν x).toReal) (μ Set.univ).t …
  -/
  rw [← setIntegral_univ, setIntegral_toReal_rnDeriv hμν Set.univ]
  /-
    🎉 no goals
  -/


lemma integral_toReal_rnDeriv' [IsFiniteMeasure μ] [SigmaFinite ν] :
    ∫ x, (μ.rnDeriv ν x).toReal ∂ν = (μ Set.univ).toReal - (μ.singularPart ν Set.univ).toReal := by
  rw [← ENNReal.toReal_sub_of_le (μ.singularPart_le ν Set.univ) (measure_ne_top _ _),
    ← Measure.sub_apply .univ (Measure.singularPart_le μ ν), Measure.measure_sub_singularPart,
    ← Measure.setIntegral_toReal_rnDeriv_eq_withDensity, setIntegral_univ]


lemma rnDeriv_mul_rnDeriv {κ : Measure α} [SigmaFinite μ] [SigmaFinite ν] [SigmaFinite κ]
    (hμν : μ ≪ ν) :
    μ.rnDeriv ν * ν.rnDeriv κ =ᵐ[κ] μ.rnDeriv κ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν κ : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite κ
    hμν : μ.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.ae κ).EventuallyEq (HMul.hMul (μ.rnDeriv ν) (ν.rnDeriv κ)) (μ …
  -/
  refine (rnDeriv_withDensity_left ?_ ?_).symm.trans ?_
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ ν κ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.SigmaFinite μ
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite κ
      hμν : μ.AbsolutelyContinuous ν
      ⊢ AEMeasurable (μ.rnDeriv ν) κ
    -/
  · exact (Measure.measurable_rnDeriv _ _).aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν κ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.SigmaFinite μ
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite κ
      hμν : μ.AbsolutelyContinuous ν
      ⊢ Filter.Eventually (fun x => Ne (μ.rnDeriv ν x) Top.top) (MeasureTheory.ae ν)
    -/
  · exact rnDeriv_ne_top _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ ν κ : MeasureTheory.Measure α
      inst✝² : MeasureTheory.SigmaFinite μ
      inst✝¹ : MeasureTheory.SigmaFinite ν
      inst✝ : MeasureTheory.SigmaFinite κ
      hμν : μ.AbsolutelyContinuous ν
      ⊢ (MeasureTheory.ae κ).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv κ)  …
    -/
  · rw [Measure.withDensity_rnDeriv_eq _ _ hμν]
    /-
      🎉 no goals
    -/


lemma rnDeriv_mul_rnDeriv' {κ : Measure α} [SigmaFinite μ] [SigmaFinite ν] [SigmaFinite κ]
    (hνκ : ν ≪ κ) :
    μ.rnDeriv ν * ν.rnDeriv κ =ᵐ[ν] μ.rnDeriv κ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν κ : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite κ
    hνκ : ν.AbsolutelyContinuous κ
    ⊢ (MeasureTheory.ae ν).EventuallyEq (HMul.hMul (μ.rnDeriv ν) (ν.rnDeriv κ)) (μ …
  -/
  obtain ⟨h_meas, h_sing, hμν⟩ := Measure.haveLebesgueDecomposition_spec μ ν
  filter_upwards [hνκ <| Measure.rnDeriv_add' (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)) κ,
    hνκ <| Measure.rnDeriv_withDensity_left_of_absolutelyContinuous hνκ h_meas.aemeasurable,
    Measure.rnDeriv_eq_zero_of_mutuallySingular h_sing hνκ] with x hx1 hx2 hx3
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ ν κ : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite κ
    hνκ : ν.AbsolutelyContinuous κ
    h_meas : Measurable (μ.rnDeriv ν)
    h_sing : (μ.singularPart ν).MutuallySingular ν
    hμν : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    x : α
    hx1 : Eq ((HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν))).rnDeriv …
    hx2 : Eq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv κ x) (HMul.hMul (μ.rnDeriv ν x …
    hx3 : Eq ((μ.singularPart ν).rnDeriv κ x) (0 x)
    ⊢ Eq (HMul.hMul (μ.rnDeriv ν) (ν.rnDeriv κ) x) (μ.rnDeriv κ x)
  -/
  nth_rw 2 [hμν]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ ν κ : MeasureTheory.Measure α
    inst✝² : MeasureTheory.SigmaFinite μ
    inst✝¹ : MeasureTheory.SigmaFinite ν
    inst✝ : MeasureTheory.SigmaFinite κ
    hνκ : ν.AbsolutelyContinuous κ
    h_meas : Measurable (μ.rnDeriv ν)
    h_sing : (μ.singularPart ν).MutuallySingular ν
    hμν : Eq μ (HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν)))
    x : α
    hx1 : Eq ((HAdd.hAdd (μ.singularPart ν) (ν.withDensity (μ.rnDeriv ν))).rnDeriv …
    hx2 : Eq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv κ x) (HMul.hMul (μ.rnDeriv ν x …
    hx3 : Eq ((μ.singularPart ν).rnDeriv κ x) (0 x)
    ⊢ Eq (HMul.hMul (μ.rnDeriv ν) (ν.rnDeriv κ) x) ((HAdd.hAdd (μ.singularPart ν)  …
  -/
  rw [hx1, Pi.add_apply, hx2, Pi.mul_apply, hx3, Pi.zero_apply, zero_add]
  /-
    🎉 no goals
  -/


lemma rnDeriv_le_one_of_le (hμν : μ ≤ ν) [SigmaFinite ν] : μ.rnDeriv ν ≤ᵐ[ν] 1 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : LE.le μ ν
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ (MeasureTheory.ae ν).EventuallyLE (μ.rnDeriv ν) 1
  -/
  refine ae_le_of_forall_setLIntegral_le_of_sigmaFinite (μ.measurable_rnDeriv ν) fun s _ _ ↦ ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : LE.le μ ν
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set α
    x✝¹ : MeasurableSet s
    x✝ : LT.lt (ν s) Top.top
    ⊢ LE.le (MeasureTheory.lintegral (ν.restrict s) fun x => μ.rnDeriv ν x) (Measu …
  -/
  simp only [Pi.one_apply, MeasureTheory.setLIntegral_one]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : LE.le μ ν
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set α
    x✝¹ : MeasurableSet s
    x✝ : LT.lt (ν s) Top.top
    ⊢ LE.le (MeasureTheory.lintegral (ν.restrict s) fun x => μ.rnDeriv ν x) (ν s)
  -/
  exact (Measure.setLIntegral_rnDeriv_le s).trans (hμν s)
  /-
    🎉 no goals
  -/


lemma rnDeriv_le_one_iff_le [HaveLebesgueDecomposition μ ν] [SigmaFinite ν] (hμν : μ ≪ ν) :
    μ.rnDeriv ν ≤ᵐ[ν] 1 ↔ μ ≤ ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Iff ((MeasureTheory.ae ν).EventuallyLE (μ.rnDeriv ν) 1) (LE.le μ ν)
  -/
  refine ⟨fun h s ↦ ?_, fun h ↦ rnDeriv_le_one_of_le h⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    h : (MeasureTheory.ae ν).EventuallyLE (μ.rnDeriv ν) 1
    s : Set α
    ⊢ LE.le (μ s) (ν s)
  -/
  rw [← withDensity_rnDeriv_eq _ _ hμν, withDensity_apply', ← setLIntegral_one]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    h : (MeasureTheory.ae ν).EventuallyLE (μ.rnDeriv ν) 1
    s : Set α
    ⊢ LE.le (MeasureTheory.lintegral (ν.restrict s) fun a => μ.rnDeriv ν a) (Measu …
  -/
  exact setLIntegral_mono_ae aemeasurable_const (h.mono fun _ hh _ ↦ hh)
  /-
    🎉 no goals
  -/


lemma rnDeriv_eq_one_iff_eq [HaveLebesgueDecomposition μ ν] [SigmaFinite ν] (hμν : μ ≪ ν) :
    μ.rnDeriv ν =ᵐ[ν] 1 ↔ μ = ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Iff ((MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν) 1) (Eq μ ν)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h ▸ ν.rnDeriv_self⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv ν) 1
    ⊢ Eq μ ν
  -/
  rw [← withDensity_rnDeriv_eq _ _ hμν, withDensity_congr_ae h, withDensity_one]
  /-
    🎉 no goals
  -/


lemma _root_.MeasurableEmbedding.rnDeriv_map_aux (hf : MeasurableEmbedding f)
    (hμν : μ ≪ ν) [SigmaFinite μ] [SigmaFinite ν] :
    (fun x ↦ (μ.map f).rnDeriv (ν.map f) (f x)) =ᵐ[ν] μ.rnDeriv ν := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => (MeasureTheory.Measure.map f μ). …
  -/
  refine ae_eq_of_forall_setLIntegral_eq_of_sigmaFinite ?_ ?_ (fun s _ _ ↦ ?_)
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      mβ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      hμν : μ.AbsolutelyContinuous ν
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      ⊢ Measurable fun x => (MeasureTheory.Measure.map f μ).rnDeriv (MeasureTheory.M …
    -/
  · exact (Measure.measurable_rnDeriv _ _).comp hf.measurable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      mβ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      hμν : μ.AbsolutelyContinuous ν
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      ⊢ Measurable (μ.rnDeriv ν)
    -/
  · exact Measure.measurable_rnDeriv _ _
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set α
    x✝¹ : MeasurableSet s
    x✝ : LT.lt (ν s) Top.top
    ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun x => (MeasureTheory.Measure.m …
  -/
  rw [← hf.lintegral_map, Measure.setLIntegral_rnDeriv hμν]
  /-
    case refine_3
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set α
    x✝¹ : MeasurableSet s
    x✝ : LT.lt (ν s) Top.top
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map f (ν.restrict s)) fun …
  -/
  have hs_eq : s = f ⁻¹' (f '' s) := by rw [hf.injective.preimage_image]
  /-
    case refine_3
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set α
    x✝¹ : MeasurableSet s
    x✝ : LT.lt (ν s) Top.top
    hs_eq : Eq s (Set.preimage f (Set.image f s))
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map f (ν.restrict s)) fun …
  -/
  have : SigmaFinite (ν.map f) := hf.sigmaFinite_map
  rw [hs_eq, ← hf.restrict_map, Measure.setLIntegral_rnDeriv (hf.absolutelyContinuous_map hμν),
    hf.map_apply]


lemma _root_.MeasurableEmbedding.rnDeriv_map (hf : MeasurableEmbedding f)
    (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν] :
    (fun x ↦ (μ.map f).rnDeriv (ν.map f) (f x)) =ᵐ[ν] μ.rnDeriv ν := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => (MeasureTheory.Measure.map f μ). …
  -/
  rw [μ.haveLebesgueDecomposition_add ν, Measure.map_add _ _ hf.measurable]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory.Measur …
  -/
  have : SigmaFinite (map f ν) := hf.sigmaFinite_map
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory.Measur …
  -/
  have : SigmaFinite (map f (μ.singularPart ν)) := hf.sigmaFinite_map
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
    this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart  …
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory.Measur …
  -/
  have : SigmaFinite (map f (ν.withDensity (μ.rnDeriv ν))) := hf.sigmaFinite_map
  have h_add := Measure.rnDeriv_add' ((μ.singularPart ν).map f)
    ((ν.withDensity (μ.rnDeriv ν)).map f) (ν.map f)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    this✝¹ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
    this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart …
    this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (ν.withDensity ( …
    h_add : (MeasureTheory.ae (MeasureTheory.Measure.map f ν)).EventuallyEq ((HAdd …
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory.Measur …
  -/
  rw [Filter.EventuallyEq, hf.ae_map_iff, ← Filter.EventuallyEq] at h_add
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    this✝¹ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
    this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart …
    this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (ν.withDensity ( …
    h_add : (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory. …
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory.Measur …
  -/
  refine h_add.trans ((Measure.rnDeriv_add' _ _ _).trans ?_).symm
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    this✝¹ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
    this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart …
    this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (ν.withDensity ( …
    h_add : (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory. …
    ⊢ (MeasureTheory.ae ν).EventuallyEq (HAdd.hAdd ((μ.singularPart ν).rnDeriv ν)  …
  -/
  refine Filter.EventuallyEq.add ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      this✝¹ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
      this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart …
      this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (ν.withDensity ( …
      h_add : (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory. …
      ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.singularPart ν).rnDeriv ν) fun x => (M …
    -/
  · refine (Measure.rnDeriv_singularPart μ ν).trans ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      this✝¹ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
      this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart …
      this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (ν.withDensity ( …
      h_add : (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory. …
      ⊢ (MeasureTheory.ae ν).EventuallyEq 0 fun x => (MeasureTheory.Measure.map f (μ …
    -/
    symm
    suffices (fun x ↦ ((μ.singularPart ν).map f).rnDeriv (ν.map f) x) =ᵐ[ν.map f] 0 by
      rw [Filter.EventuallyEq, hf.ae_map_iff] at this
      exact this
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      this✝¹ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
      this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart …
      this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (ν.withDensity ( …
      h_add : (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory. …
      ⊢ (MeasureTheory.ae (MeasureTheory.Measure.map f ν)).EventuallyEq (fun x => (M …
    -/
    refine Measure.rnDeriv_eq_zero_of_mutuallySingular ?_ Measure.AbsolutelyContinuous.rfl
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      this✝¹ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
      this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart …
      this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (ν.withDensity ( …
      h_add : (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory. …
      ⊢ (MeasureTheory.Measure.map f (μ.singularPart ν)).MutuallySingular (MeasureTh …
    -/
    exact hf.mutuallySingular_map (μ.mutuallySingular_singularPart ν)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      this✝¹ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f ν)
      this✝ : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (μ.singularPart …
      this : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f (ν.withDensity ( …
      h_add : (MeasureTheory.ae ν).EventuallyEq (fun x => (HAdd.hAdd (MeasureTheory. …
      ⊢ (MeasureTheory.ae ν).EventuallyEq ((ν.withDensity (μ.rnDeriv ν)).rnDeriv ν)  …
    -/
  · exact (hf.rnDeriv_map_aux (withDensity_absolutelyContinuous _ _)).symm
    /-
      🎉 no goals
    -/


lemma _root_.MeasurableEmbedding.map_withDensity_rnDeriv (hf : MeasurableEmbedding f)
    (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν] :
    (ν.withDensity (μ.rnDeriv ν)).map f = (ν.map f).withDensity ((μ.map f).rnDeriv (ν.map f)) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ Eq (MeasureTheory.Measure.map f (ν.withDensity (μ.rnDeriv ν))) ((MeasureTheo …
  -/
  ext s hs
  rw [hf.map_apply, withDensity_apply _ (hf.measurable hs), withDensity_apply _ hs,
    setLIntegral_map hs (Measure.measurable_rnDeriv _ _) hf.measurable]
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (ν.restrict (Set.preimage f s)) fun a => μ.rnDer …
  -/
  refine setLIntegral_congr_fun (hf.measurable hs) ?_
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    s : Set β
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.preimage f s) x → Eq (μ.rnDe …
  -/
  filter_upwards [hf.rnDeriv_map μ ν] with a ha _ using ha.symm
  /-
    🎉 no goals
  -/


lemma _root_.MeasurableEmbedding.singularPart_map (hf : MeasurableEmbedding f)
    (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν] :
    (μ.map f).singularPart (ν.map f) = (μ.singularPart ν).map f := by
  have h_add : μ.map f = (μ.singularPart ν).map f
      + (ν.map f).withDensity ((μ.map f).rnDeriv (ν.map f)) := by
    conv_lhs => rw [μ.haveLebesgueDecomposition_add ν]
    rw [Measure.map_add _ _ hf.measurable, ← hf.map_withDensity_rnDeriv μ ν]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    h_add : Eq (MeasureTheory.Measure.map f μ) (HAdd.hAdd (MeasureTheory.Measure.m …
    ⊢ Eq ((MeasureTheory.Measure.map f μ).singularPart (MeasureTheory.Measure.map  …
  -/
  refine (Measure.eq_singularPart (Measure.measurable_rnDeriv _ _) ?_ h_add).symm
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    h_add : Eq (MeasureTheory.Measure.map f μ) (HAdd.hAdd (MeasureTheory.Measure.m …
    ⊢ (MeasureTheory.Measure.map f (μ.singularPart ν)).MutuallySingular (MeasureTh …
  -/
  exact hf.mutuallySingular_map (μ.mutuallySingular_singularPart ν)
  /-
    🎉 no goals
  -/


theorem withDensityᵥ_rnDeriv_eq (s : SignedMeasure α) (μ : Measure α) [SigmaFinite μ]
    (h : s ≪ᵥ μ.toENNRealVectorMeasure) : μ.withDensityᵥ (s.rnDeriv μ) = s := by
  rw [absolutelyContinuous_ennreal_iff, (_ : μ.toENNRealVectorMeasure.ennrealToMeasure = μ),
    totalVariation_absolutelyContinuous_iff] at h
    /-
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      h : And (s.toJordanDecomposition.posPart.AbsolutelyContinuous μ) (s.toJordanDe …
      ⊢ Eq (μ.withDensityᵥ (s.rnDeriv μ)) s
    -/
  · ext1 i hi
    rw [withDensityᵥ_apply (integrable_rnDeriv _ _) hi, rnDeriv_def, integral_sub,
      setIntegral_toReal_rnDeriv h.1 i, setIntegral_toReal_rnDeriv h.2 i]
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        h : And (s.toJordanDecomposition.posPart.AbsolutelyContinuous μ) (s.toJordanDe …
        i : Set α
        hi : MeasurableSet i
        ⊢ Eq (HSub.hSub (s.toJordanDecomposition.posPart i).toReal (s.toJordanDecompos …
      -/
    · conv_rhs => rw [← s.toSignedMeasure_toJordanDecomposition]
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        h : And (s.toJordanDecomposition.posPart.AbsolutelyContinuous μ) (s.toJordanDe …
        i : Set α
        hi : MeasurableSet i
        ⊢ Eq (HSub.hSub (s.toJordanDecomposition.posPart i).toReal (s.toJordanDecompos …
      -/
      erw [VectorMeasure.sub_apply]
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        h : And (s.toJordanDecomposition.posPart.AbsolutelyContinuous μ) (s.toJordanDe …
        i : Set α
        hi : MeasurableSet i
        ⊢ Eq (HSub.hSub (s.toJordanDecomposition.posPart i).toReal (s.toJordanDecompos …
      -/
      rw [toSignedMeasure_apply_measurable hi, toSignedMeasure_apply_measurable hi]
      /-
        🎉 no goals
      -/
    all_goals
      rw [← integrableOn_univ]
      refine IntegrableOn.restrict ?_ MeasurableSet.univ
      refine ⟨?_, hasFiniteIntegral_toReal_of_lintegral_ne_top ?_⟩
      · apply Measurable.aestronglyMeasurable (by fun_prop)
      · rw [setLIntegral_univ]
        exact (lintegral_rnDeriv_lt_top _ _).ne
    /-
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      h : s.totalVariation.AbsolutelyContinuous μ.toENNRealVectorMeasure.ennrealToMe …
      ⊢ Eq μ.toENNRealVectorMeasure.ennrealToMeasure μ
    -/
  · exact equivMeasure.right_inv μ
    /-
      🎉 no goals
    -/


/-- The Radon-Nikodym theorem for signed measures. -/
theorem absolutelyContinuous_iff_withDensityᵥ_rnDeriv_eq (s : SignedMeasure α) (μ : Measure α)
    [SigmaFinite μ] : s ≪ᵥ μ.toENNRealVectorMeasure ↔ μ.withDensityᵥ (s.rnDeriv μ) = s :=
  ⟨withDensityᵥ_rnDeriv_eq s μ, fun h => h ▸ withDensityᵥ_absolutelyContinuous _ _⟩


theorem lintegral_rnDeriv_mul [HaveLebesgueDecomposition μ ν] (hμν : μ ≪ ν) {f : α → ℝ≥0∞}
    (hf : AEMeasurable f ν) : ∫⁻ x, μ.rnDeriv ν x * f x ∂ν = ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    f : α → ENNReal
    hf : AEMeasurable f ν
    ⊢ Eq (MeasureTheory.lintegral ν fun x => HMul.hMul (μ.rnDeriv ν x) (f x)) (Mea …
  -/
  nth_rw 2 [← withDensity_rnDeriv_eq μ ν hμν]
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    f : α → ENNReal
    hf : AEMeasurable f ν
    ⊢ Eq (MeasureTheory.lintegral ν fun x => HMul.hMul (μ.rnDeriv ν x) (f x)) (Mea …
  -/
  rw [lintegral_withDensity_eq_lintegral_mul₀ (measurable_rnDeriv μ ν).aemeasurable hf]
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    f : α → ENNReal
    hf : AEMeasurable f ν
    ⊢ Eq (MeasureTheory.lintegral ν fun x => HMul.hMul (μ.rnDeriv ν x) (f x)) (Mea …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma setLIntegral_rnDeriv_mul [HaveLebesgueDecomposition μ ν] (hμν : μ ≪ ν) {f : α → ℝ≥0∞}
    (hf : AEMeasurable f ν) {s : Set α} (hs : MeasurableSet s) :
    ∫⁻ x in s, μ.rnDeriv ν x * f x ∂ν = ∫⁻ x in s, f x ∂μ := by
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    f : α → ENNReal
    hf : AEMeasurable f ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun x => HMul.hMul (μ.rnDeriv ν x …
  -/
  nth_rw 2 [← Measure.withDensity_rnDeriv_eq μ ν hμν]
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    f : α → ENNReal
    hf : AEMeasurable f ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun x => HMul.hMul (μ.rnDeriv ν x …
  -/
  rw [setLIntegral_withDensity_eq_lintegral_mul₀ (measurable_rnDeriv μ ν).aemeasurable hf hs]
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    f : α → ENNReal
    hf : AEMeasurable f ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral (ν.restrict s) fun x => HMul.hMul (μ.rnDeriv ν x …
  -/
  simp only [Pi.mul_apply]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_rnDeriv_mul := setLIntegral_rnDeriv_mul


theorem integrable_rnDeriv_smul_iff (hμν : μ ≪ ν) :
    Integrable (fun x ↦ (μ.rnDeriv ν x).toReal • f x) ν ↔ Integrable f μ := by
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Iff (MeasureTheory.Integrable (fun x => HSMul.hSMul (μ.rnDeriv ν x).toReal ( …
  -/
  nth_rw 2 [← withDensity_rnDeriv_eq μ ν hμν]
  rw [← integrable_withDensity_iff_integrable_smul' (E := E)
    (measurable_rnDeriv μ ν) (rnDeriv_lt_top μ ν)]


theorem withDensityᵥ_rnDeriv_smul (hμν : μ ≪ ν) (hf : Integrable f μ) :
    ν.withDensityᵥ (fun x ↦ (rnDeriv μ ν x).toReal • f x) = μ.withDensityᵥ f := by
  rw [withDensityᵥ_smul_eq_withDensityᵥ_withDensity' (measurable_rnDeriv μ ν).aemeasurable
    (rnDeriv_lt_top μ ν) ((integrable_rnDeriv_smul_iff hμν).mpr hf), withDensity_rnDeriv_eq μ ν hμν]


theorem integral_rnDeriv_smul (hμν : μ ≪ ν) :
    ∫ x, (μ.rnDeriv ν x).toReal • f x ∂ν = ∫ x, f x ∂μ := by
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Eq (MeasureTheory.integral ν fun x => HSMul.hSMul (μ.rnDeriv ν x).toReal (f  …
  -/
  by_cases hf : Integrable f μ
  · rw [← setIntegral_univ, ← withDensityᵥ_apply ((integrable_rnDeriv_smul_iff hμν).mpr hf) .univ,
      ← setIntegral_univ, ← withDensityᵥ_apply hf .univ, withDensityᵥ_rnDeriv_smul hμν hf]
    /-
      case neg
      α : Type u_3
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      E : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : μ.HaveLebesgueDecomposition ν
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → E
      hμν : μ.AbsolutelyContinuous ν
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.integral ν fun x => HSMul.hSMul (μ.rnDeriv ν x).toReal (f  …
    -/
  · rw [integral_undef hf, integral_undef]
    /-
      case neg
      α : Type u_3
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      E : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : μ.HaveLebesgueDecomposition ν
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → E
      hμν : μ.AbsolutelyContinuous ν
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Not (MeasureTheory.Integrable (fun x => HSMul.hSMul (μ.rnDeriv ν x).toReal ( …
    -/
    contrapose! hf
    /-
      case neg
      α : Type u_3
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      E : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : μ.HaveLebesgueDecomposition ν
      inst✝ : MeasureTheory.SigmaFinite μ
      f : α → E
      hμν : μ.AbsolutelyContinuous ν
      hf : MeasureTheory.Integrable (fun x => HSMul.hSMul (μ.rnDeriv ν x).toReal (f  …
      ⊢ MeasureTheory.Integrable f μ
    -/
    exact (integrable_rnDeriv_smul_iff hμν).mp hf
    /-
      🎉 no goals
    -/


lemma setIntegral_rnDeriv_smul (hμν : μ ≪ ν) {s : Set α} (hs : MeasurableSet s) :
    ∫ x in s, (μ.rnDeriv ν x).toReal • f x ∂ν = ∫ x in s, f x ∂μ := by
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : μ.HaveLebesgueDecomposition ν
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hμν : μ.AbsolutelyContinuous ν
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (ν.restrict s) fun x => HSMul.hSMul (μ.rnDeriv ν  …
  -/
  simp_rw [← integral_indicator hs, Set.indicator_smul, integral_rnDeriv_smul hμν]
  /-
    🎉 no goals
  -/


