/-- Log-Likelihood Ratio between two measures. -/
noncomputable def llr (μ ν : Measure α) (x : α) : ℝ := log (μ.rnDeriv ν x).toReal


lemma llr_def (μ ν : Measure α) : llr μ ν = fun x ↦ log (μ.rnDeriv ν x).toReal := rfl


lemma exp_llr (μ ν : Measure α) [SigmaFinite μ] :
    (fun x ↦ exp (llr μ ν x))
      =ᵐ[ν] fun x ↦ if μ.rnDeriv ν x = 0 then 1 else (μ.rnDeriv ν x).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => Real.exp (MeasureTheory.llr μ ν  …
  -/
  filter_upwards [Measure.rnDeriv_lt_top μ ν] with x hx
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    x : α
    hx : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (Real.exp (MeasureTheory.llr μ ν x)) (ite (Eq (μ.rnDeriv ν x) 0) 1 (μ.rnD …
  -/
  by_cases h_zero : μ.rnDeriv ν x = 0
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      x : α
      hx : LT.lt (μ.rnDeriv ν x) Top.top
      h_zero : Eq (μ.rnDeriv ν x) 0
      ⊢ Eq (Real.exp (MeasureTheory.llr μ ν x)) (ite (Eq (μ.rnDeriv ν x) 0) 1 (μ.rnD …
    -/
  · simp only [llr, h_zero, ENNReal.zero_toReal, log_zero, exp_zero, ite_true]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      x : α
      hx : LT.lt (μ.rnDeriv ν x) Top.top
      h_zero : Not (Eq (μ.rnDeriv ν x) 0)
      ⊢ Eq (Real.exp (MeasureTheory.llr μ ν x)) (ite (Eq (μ.rnDeriv ν x) 0) 1 (μ.rnD …
    -/
  · rw [llr, exp_log, if_neg h_zero]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      x : α
      hx : LT.lt (μ.rnDeriv ν x) Top.top
      h_zero : Not (Eq (μ.rnDeriv ν x) 0)
      ⊢ LT.lt 0 (μ.rnDeriv ν x).toReal
    -/
    exact ENNReal.toReal_pos h_zero hx.ne
    /-
      🎉 no goals
    -/


lemma exp_llr_of_ac (μ ν : Measure α) [SigmaFinite μ] [Measure.HaveLebesgueDecomposition μ ν]
    (hμν : μ ≪ ν) :
    (fun x ↦ exp (llr μ ν x)) =ᵐ[μ] fun x ↦ (μ.rnDeriv ν x).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => Real.exp (MeasureTheory.llr μ ν  …
  -/
  filter_upwards [hμν.ae_le (exp_llr μ ν), Measure.rnDeriv_pos hμν] with x hx_eq hx_pos
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    x : α
    hx_eq : Eq (Real.exp (MeasureTheory.llr μ ν x)) (ite (Eq (μ.rnDeriv ν x) 0) 1  …
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    ⊢ Eq (Real.exp (MeasureTheory.llr μ ν x)) (μ.rnDeriv ν x).toReal
  -/
  rw [hx_eq, if_neg hx_pos.ne']
  /-
    🎉 no goals
  -/


lemma exp_llr_of_ac' (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν] (hμν : ν ≪ μ) :
    (fun x ↦ exp (llr μ ν x)) =ᵐ[ν] fun x ↦ (μ.rnDeriv ν x).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : ν.AbsolutelyContinuous μ
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => Real.exp (MeasureTheory.llr μ ν  …
  -/
  filter_upwards [exp_llr μ ν, Measure.rnDeriv_pos' hμν] with x hx hx_pos
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : ν.AbsolutelyContinuous μ
    x : α
    hx : Eq (Real.exp (MeasureTheory.llr μ ν x)) (ite (Eq (μ.rnDeriv ν x) 0) 1 (μ. …
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    ⊢ Eq (Real.exp (MeasureTheory.llr μ ν x)) (μ.rnDeriv ν x).toReal
  -/
  rwa [if_neg hx_pos.ne'] at hx
  /-
    🎉 no goals
  -/


lemma neg_llr [SigmaFinite μ] [SigmaFinite ν] (hμν : μ ≪ ν) :
    - llr μ ν =ᵐ[μ] llr ν μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Neg.neg (MeasureTheory.llr μ ν)) (Measure …
  -/
  filter_upwards [Measure.inv_rnDeriv hμν] with x hx
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    x : α
    hx : Eq (Inv.inv (μ.rnDeriv ν) x) (ν.rnDeriv μ x)
    ⊢ Eq (Neg.neg (MeasureTheory.llr μ ν) x) (MeasureTheory.llr ν μ x)
  -/
  rw [Pi.neg_apply, llr, llr, ← log_inv, ← ENNReal.toReal_inv]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    x : α
    hx : Eq (Inv.inv (μ.rnDeriv ν) x) (ν.rnDeriv μ x)
    ⊢ Eq (Real.log (Inv.inv (μ.rnDeriv ν x)).toReal) (Real.log (ν.rnDeriv μ x).toR …
  -/
  congr
  /-
    🎉 no goals
  -/


lemma exp_neg_llr [SigmaFinite μ] [SigmaFinite ν] (hμν : μ ≪ ν) :
    (fun x ↦ exp (- llr μ ν x)) =ᵐ[μ] fun x ↦ (ν.rnDeriv μ x).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => Real.exp (Neg.neg (MeasureTheory …
  -/
  filter_upwards [neg_llr hμν, exp_llr_of_ac' ν μ hμν] with x hx hx_exp_log
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    x : α
    hx : Eq (Neg.neg (MeasureTheory.llr μ ν) x) (MeasureTheory.llr ν μ x)
    hx_exp_log : Eq (Real.exp (MeasureTheory.llr ν μ x)) (ν.rnDeriv μ x).toReal
    ⊢ Eq (Real.exp (Neg.neg (MeasureTheory.llr μ ν x))) (ν.rnDeriv μ x).toReal
  -/
  rw [Pi.neg_apply] at hx
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    x : α
    hx : Eq (Neg.neg (MeasureTheory.llr μ ν x)) (MeasureTheory.llr ν μ x)
    hx_exp_log : Eq (Real.exp (MeasureTheory.llr ν μ x)) (ν.rnDeriv μ x).toReal
    ⊢ Eq (Real.exp (Neg.neg (MeasureTheory.llr μ ν x))) (ν.rnDeriv μ x).toReal
  -/
  rw [hx, hx_exp_log]
  /-
    🎉 no goals
  -/


lemma exp_neg_llr' [SigmaFinite μ] [SigmaFinite ν] (hμν : ν ≪ μ) :
    (fun x ↦ exp (- llr μ ν x)) =ᵐ[ν] fun x ↦ (ν.rnDeriv μ x).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : ν.AbsolutelyContinuous μ
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => Real.exp (Neg.neg (MeasureTheory …
  -/
  filter_upwards [neg_llr hμν, exp_llr_of_ac ν μ hμν] with x hx hx_exp_log
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : ν.AbsolutelyContinuous μ
    x : α
    hx : Eq (Neg.neg (MeasureTheory.llr ν μ) x) (MeasureTheory.llr μ ν x)
    hx_exp_log : Eq (Real.exp (MeasureTheory.llr ν μ x)) (ν.rnDeriv μ x).toReal
    ⊢ Eq (Real.exp (Neg.neg (MeasureTheory.llr μ ν x))) (ν.rnDeriv μ x).toReal
  -/
  rw [Pi.neg_apply, neg_eq_iff_eq_neg] at hx
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : ν.AbsolutelyContinuous μ
    x : α
    hx : Eq (MeasureTheory.llr ν μ x) (Neg.neg (MeasureTheory.llr μ ν x))
    hx_exp_log : Eq (Real.exp (MeasureTheory.llr ν μ x)) (ν.rnDeriv μ x).toReal
    ⊢ Eq (Real.exp (Neg.neg (MeasureTheory.llr μ ν x))) (ν.rnDeriv μ x).toReal
  -/
  rw [← hx, hx_exp_log]
  /-
    🎉 no goals
  -/


@[measurability]
lemma measurable_llr (μ ν : Measure α) : Measurable (llr μ ν) :=
  (Measure.measurable_rnDeriv μ ν).ennreal_toReal.log


@[measurability]
lemma stronglyMeasurable_llr (μ ν : Measure α) : StronglyMeasurable (llr μ ν) :=
  (measurable_llr μ ν).stronglyMeasurable


lemma llr_smul_left [IsFiniteMeasure μ] [Measure.HaveLebesgueDecomposition μ ν]
    (hμν : μ ≪ ν) (c : ℝ≥0∞) (hc : c ≠ 0) (hc_ne_top : c ≠ ∞) :
    llr (c • μ) ν =ᵐ[μ] fun x ↦ llr μ ν x + log c.toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.llr (HSMul.hSMul c μ) ν) fu …
  -/
  simp only [llr, llr_def]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => Real.log ((HSMul.hSMul c μ).rnDe …
  -/
  have h := Measure.rnDeriv_smul_left_of_ne_top μ ν hc_ne_top
  filter_upwards [hμν.ae_le h, Measure.rnDeriv_pos hμν, hμν.ae_le (Measure.rnDeriv_lt_top μ ν)]
    with x hx_eq hx_pos hx_ne_top
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
    x : α
    hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (Real.log ((HSMul.hSMul c μ).rnDeriv ν x).toReal) (HAdd.hAdd (Real.log (μ …
  -/
  rw [hx_eq]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
    x : α
    hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (Real.log (HSMul.hSMul c (μ.rnDeriv ν) x).toReal) (HAdd.hAdd (Real.log (μ …
  -/
  simp only [Pi.smul_apply, smul_eq_mul, ENNReal.toReal_mul]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
    x : α
    hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (Real.log (HMul.hMul c.toReal (μ.rnDeriv ν x).toReal)) (HAdd.hAdd (Real.l …
  -/
  rw [log_mul]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
    x : α
    hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (HAdd.hAdd (Real.log c.toReal) (Real.log (μ.rnDeriv ν x).toReal)) (HAdd.h …
  -/
  rotate_left
    /-
      case h.hx
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : μ.HaveLebesgueDecomposition ν
      hμν : μ.AbsolutelyContinuous ν
      c : ENNReal
      hc : Ne c 0
      hc_ne_top : Ne c Top.top
      h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
      x : α
      hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
      hx_pos : LT.lt 0 (μ.rnDeriv ν x)
      hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
      ⊢ Ne c.toReal 0
    -/
  · rw [ENNReal.toReal_ne_zero]
    /-
      case h.hx
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : μ.HaveLebesgueDecomposition ν
      hμν : μ.AbsolutelyContinuous ν
      c : ENNReal
      hc : Ne c 0
      hc_ne_top : Ne c Top.top
      h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
      x : α
      hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
      hx_pos : LT.lt 0 (μ.rnDeriv ν x)
      hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
      ⊢ And (Ne c 0) (Ne c Top.top)
    -/
    simp [hc, hc_ne_top]
    /-
      🎉 no goals
    -/
    /-
      case h.hy
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : μ.HaveLebesgueDecomposition ν
      hμν : μ.AbsolutelyContinuous ν
      c : ENNReal
      hc : Ne c 0
      hc_ne_top : Ne c Top.top
      h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
      x : α
      hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
      hx_pos : LT.lt 0 (μ.rnDeriv ν x)
      hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
      ⊢ Ne (μ.rnDeriv ν x).toReal 0
    -/
  · rw [ENNReal.toReal_ne_zero]
    /-
      case h.hy
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : μ.HaveLebesgueDecomposition ν
      hμν : μ.AbsolutelyContinuous ν
      c : ENNReal
      hc : Ne c 0
      hc_ne_top : Ne c Top.top
      h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
      x : α
      hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
      hx_pos : LT.lt 0 (μ.rnDeriv ν x)
      hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
      ⊢ And (Ne (μ.rnDeriv ν x) 0) (Ne (μ.rnDeriv ν x) Top.top)
    -/
    simp [hx_pos.ne', hx_ne_top.ne]
    /-
      🎉 no goals
    -/
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq ((HSMul.hSMul c μ).rnDeriv ν) (HSMul.hSM …
    x : α
    hx_eq : Eq ((HSMul.hSMul c μ).rnDeriv ν x) (HSMul.hSMul c (μ.rnDeriv ν) x)
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (HAdd.hAdd (Real.log c.toReal) (Real.log (μ.rnDeriv ν x).toReal)) (HAdd.h …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma llr_smul_right [IsFiniteMeasure μ] [Measure.HaveLebesgueDecomposition μ ν]
    (hμν : μ ≪ ν) (c : ℝ≥0∞) (hc : c ≠ 0) (hc_ne_top : c ≠ ∞) :
    llr μ (c • ν) =ᵐ[μ] fun x ↦ llr μ ν x - log c.toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.llr μ (HSMul.hSMul c ν)) fu …
  -/
  simp only [llr, llr_def]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => Real.log (μ.rnDeriv (HSMul.hSMul …
  -/
  have h := Measure.rnDeriv_smul_right_of_ne_top μ ν hc hc_ne_top
  filter_upwards [hμν.ae_le h, Measure.rnDeriv_pos hμν, hμν.ae_le (Measure.rnDeriv_lt_top μ ν)]
    with x hx_eq hx_pos hx_ne_top
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
    x : α
    hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (Real.log (μ.rnDeriv (HSMul.hSMul c ν) x).toReal) (HSub.hSub (Real.log (μ …
  -/
  rw [hx_eq]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
    x : α
    hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (Real.log (HSMul.hSMul (Inv.inv c) (μ.rnDeriv ν) x).toReal) (HSub.hSub (R …
  -/
  simp only [Pi.smul_apply, smul_eq_mul, ENNReal.toReal_mul]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
    x : α
    hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (Real.log (HMul.hMul (Inv.inv c).toReal (μ.rnDeriv ν x).toReal)) (HSub.hS …
  -/
  rw [log_mul]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
    x : α
    hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (HAdd.hAdd (Real.log (Inv.inv c).toReal) (Real.log (μ.rnDeriv ν x).toReal …
  -/
  rotate_left
    /-
      case h.hx
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : μ.HaveLebesgueDecomposition ν
      hμν : μ.AbsolutelyContinuous ν
      c : ENNReal
      hc : Ne c 0
      hc_ne_top : Ne c Top.top
      h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
      x : α
      hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
      hx_pos : LT.lt 0 (μ.rnDeriv ν x)
      hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
      ⊢ Ne (Inv.inv c).toReal 0
    -/
  · rw [ENNReal.toReal_ne_zero]
    /-
      case h.hx
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : μ.HaveLebesgueDecomposition ν
      hμν : μ.AbsolutelyContinuous ν
      c : ENNReal
      hc : Ne c 0
      hc_ne_top : Ne c Top.top
      h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
      x : α
      hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
      hx_pos : LT.lt 0 (μ.rnDeriv ν x)
      hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
      ⊢ And (Ne (Inv.inv c) 0) (Ne (Inv.inv c) Top.top)
    -/
    simp [hc, hc_ne_top]
    /-
      🎉 no goals
    -/
    /-
      case h.hy
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : μ.HaveLebesgueDecomposition ν
      hμν : μ.AbsolutelyContinuous ν
      c : ENNReal
      hc : Ne c 0
      hc_ne_top : Ne c Top.top
      h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
      x : α
      hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
      hx_pos : LT.lt 0 (μ.rnDeriv ν x)
      hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
      ⊢ Ne (μ.rnDeriv ν x).toReal 0
    -/
  · rw [ENNReal.toReal_ne_zero]
    /-
      case h.hy
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : μ.HaveLebesgueDecomposition ν
      hμν : μ.AbsolutelyContinuous ν
      c : ENNReal
      hc : Ne c 0
      hc_ne_top : Ne c Top.top
      h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
      x : α
      hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
      hx_pos : LT.lt 0 (μ.rnDeriv ν x)
      hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
      ⊢ And (Ne (μ.rnDeriv ν x) 0) (Ne (μ.rnDeriv ν x) Top.top)
    -/
    simp [hx_pos.ne', hx_ne_top.ne]
    /-
      🎉 no goals
    -/
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
    x : α
    hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (HAdd.hAdd (Real.log (Inv.inv c).toReal) (Real.log (μ.rnDeriv ν x).toReal …
  -/
  rw [ENNReal.toReal_inv, log_inv]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : μ.HaveLebesgueDecomposition ν
    hμν : μ.AbsolutelyContinuous ν
    c : ENNReal
    hc : Ne c 0
    hc_ne_top : Ne c Top.top
    h : (MeasureTheory.ae ν).EventuallyEq (μ.rnDeriv (HSMul.hSMul c ν)) (HSMul.hSM …
    x : α
    hx_eq : Eq (μ.rnDeriv (HSMul.hSMul c ν) x) (HSMul.hSMul (Inv.inv c) (μ.rnDeriv …
    hx_pos : LT.lt 0 (μ.rnDeriv ν x)
    hx_ne_top : LT.lt (μ.rnDeriv ν x) Top.top
    ⊢ Eq (HAdd.hAdd (Neg.neg (Real.log c.toReal)) (Real.log (μ.rnDeriv ν x).toReal …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma llr_tilted_left [SigmaFinite μ] [SigmaFinite ν] (hμν : μ ≪ ν)
    (hf : Integrable (fun x ↦ exp (f x)) μ) (hfν : AEMeasurable f ν) :
    (llr (μ.tilted f) ν) =ᵐ[μ] fun x ↦ f x - log (∫ z, exp (f z) ∂μ) + llr μ ν x := by
  cases eq_zero_or_neZero μ with
  | inl hμ =>
    simp only [hμ, ae_zero, Filter.EventuallyEq]; exact Filter.eventually_bot
  | inr h0 =>
    filter_upwards [hμν.ae_le (toReal_rnDeriv_tilted_left μ hfν), Measure.rnDeriv_pos hμν,
      hμν.ae_le (Measure.rnDeriv_lt_top μ ν)] with x hx hx_pos hx_lt_top
    rw [llr, hx, log_mul, div_eq_mul_inv, log_mul (exp_pos _).ne', log_exp, log_inv, llr,
      ← sub_eq_add_neg]
    · simp only [ne_eq, inv_eq_zero]
      exact (integral_exp_pos hf).ne'
    · simp only [ne_eq, div_eq_zero_iff]
      push_neg
      exact ⟨(exp_pos _).ne', (integral_exp_pos hf).ne'⟩
    · simp [ENNReal.toReal_eq_zero_iff, hx_lt_top.ne, hx_pos.ne']


lemma integrable_llr_tilted_left [IsFiniteMeasure μ] [SigmaFinite ν]
    (hμν : μ ≪ ν) (hf : Integrable f μ) (h_int : Integrable (llr μ ν) μ)
    (hfμ : Integrable (fun x ↦ exp (f x)) μ) (hfν : AEMeasurable f ν) :
    Integrable (llr (μ.tilted f) ν) μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → Real
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hf : MeasureTheory.Integrable f μ
    h_int : MeasureTheory.Integrable (MeasureTheory.llr μ ν) μ
    hfμ : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    hfν : AEMeasurable f ν
    ⊢ MeasureTheory.Integrable (MeasureTheory.llr (μ.tilted f) ν) μ
  -/
  rw [integrable_congr (llr_tilted_left hμν hfμ hfν)]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → Real
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hf : MeasureTheory.Integrable f μ
    h_int : MeasureTheory.Integrable (MeasureTheory.llr μ ν) μ
    hfμ : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    hfν : AEMeasurable f ν
    ⊢ MeasureTheory.Integrable (fun x => HAdd.hAdd (HSub.hSub (f x) (Real.log (Mea …
  -/
  exact Integrable.add (hf.sub (integrable_const _)) h_int
  /-
    🎉 no goals
  -/


lemma integral_llr_tilted_left [IsProbabilityMeasure μ] [SigmaFinite ν]
    (hμν : μ ≪ ν) (hf : Integrable f μ) (h_int : Integrable (llr μ ν) μ)
    (hfμ : Integrable (fun x ↦ exp (f x)) μ) (hfν : AEMeasurable f ν) :
    ∫ x, llr (μ.tilted f) ν x ∂μ = ∫ x, llr μ ν x ∂μ + ∫ x, f x ∂μ - log (∫ x, exp (f x) ∂μ) := by
  calc ∫ x, llr (μ.tilted f) ν x ∂μ
    = ∫ x, f x - log (∫ x, exp (f x) ∂μ) + llr μ ν x ∂μ :=
        integral_congr_ae (llr_tilted_left hμν hfμ hfν)
  _ = ∫ x, f x ∂μ - log (∫ x, exp (f x) ∂μ) + ∫ x, llr μ ν x ∂μ := by
        rw [integral_add ?_ h_int]
        swap; · exact hf.sub (integrable_const _)
        rw [integral_sub hf (integrable_const _)]
        simp only [integral_const, measure_univ, ENNReal.one_toReal, smul_eq_mul, one_mul]
  _ = ∫ x, llr μ ν x ∂μ + ∫ x, f x ∂μ - log (∫ x, exp (f x) ∂μ) := by abel


lemma llr_tilted_right [SigmaFinite μ] [SigmaFinite ν]
    (hμν : μ ≪ ν) (hf : Integrable (fun x ↦ exp (f x)) ν) :
    (llr μ (ν.tilted f)) =ᵐ[μ] fun x ↦ - f x + log (∫ z, exp (f z) ∂ν) + llr μ ν x := by
  cases eq_zero_or_neZero ν with
  | inl h =>
    have hμ : μ = 0 := by ext s _; exact hμν (by simp [h])
    simp only [hμ, ae_zero, Filter.EventuallyEq]; exact Filter.eventually_bot
  | inr h0 =>
    filter_upwards [hμν.ae_le (toReal_rnDeriv_tilted_right μ ν hf), Measure.rnDeriv_pos hμν,
      hμν.ae_le (Measure.rnDeriv_lt_top μ ν)] with x hx hx_pos hx_lt_top
    rw [llr, hx, log_mul, log_mul (exp_pos _).ne', log_exp, llr]
    · exact (integral_exp_pos hf).ne'
    · refine (mul_pos (exp_pos _) (integral_exp_pos hf)).ne'
    · simp [ENNReal.toReal_eq_zero_iff, hx_lt_top.ne, hx_pos.ne']


lemma integrable_llr_tilted_right [IsFiniteMeasure μ] [SigmaFinite ν]
    (hμν : μ ≪ ν) (hfμ : Integrable f μ)
    (h_int : Integrable (llr μ ν) μ) (hfν : Integrable (fun x ↦ exp (f x)) ν) :
    Integrable (llr μ (ν.tilted f)) μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → Real
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hfμ : MeasureTheory.Integrable f μ
    h_int : MeasureTheory.Integrable (MeasureTheory.llr μ ν) μ
    hfν : MeasureTheory.Integrable (fun x => Real.exp (f x)) ν
    ⊢ MeasureTheory.Integrable (MeasureTheory.llr μ (ν.tilted f)) μ
  -/
  rw [integrable_congr (llr_tilted_right hμν hfν)]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → Real
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hμν : μ.AbsolutelyContinuous ν
    hfμ : MeasureTheory.Integrable f μ
    h_int : MeasureTheory.Integrable (MeasureTheory.llr μ ν) μ
    hfν : MeasureTheory.Integrable (fun x => Real.exp (f x)) ν
    ⊢ MeasureTheory.Integrable (fun x => HAdd.hAdd (HAdd.hAdd (Neg.neg (f x)) (Rea …
  -/
  exact Integrable.add (hfμ.neg.add (integrable_const _)) h_int
  /-
    🎉 no goals
  -/


lemma integral_llr_tilted_right [IsProbabilityMeasure μ] [SigmaFinite ν]
    (hμν : μ ≪ ν) (hfμ : Integrable f μ) (hfν : Integrable (fun x ↦ exp (f x)) ν)
    (h_int : Integrable (llr μ ν) μ) :
    ∫ x, llr μ (ν.tilted f) x ∂μ = ∫ x, llr μ ν x ∂μ - ∫ x, f x ∂μ + log (∫ x, exp (f x) ∂ν) := by
  calc ∫ x, llr μ (ν.tilted f) x ∂μ
    = ∫ x, - f x + log (∫ x, exp (f x) ∂ν) + llr μ ν x ∂μ :=
        integral_congr_ae (llr_tilted_right hμν hfν)
  _ = - ∫ x, f x ∂μ + log (∫ x, exp (f x) ∂ν) + ∫ x, llr μ ν x ∂μ := by
        rw [← integral_neg, integral_add ?_ h_int]
        swap; · exact hfμ.neg.add (integrable_const _)
        rw [integral_add ?_ (integrable_const _)]
        swap; · exact hfμ.neg
        simp only [integral_const, measure_univ, ENNReal.one_toReal, smul_eq_mul, one_mul]
  _ = ∫ x, llr μ ν x ∂μ - ∫ x, f x ∂μ + log (∫ x, exp (f x) ∂ν) := by abel


