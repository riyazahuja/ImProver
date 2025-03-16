lemma ENNReal.finStronglyMeasurable_of_measurable (hf : ∫⁻ x, f x ∂μ ≠ ∞)
    (hf_meas : Measurable f) :
    FinStronglyMeasurable f μ :=
  ⟨SimpleFunc.eapprox f, measure_support_eapprox_lt_top hf_meas hf,
    SimpleFunc.tendsto_eapprox hf_meas⟩


lemma ENNReal.aefinStronglyMeasurable_of_aemeasurable (hf : ∫⁻ x, f x ∂μ ≠ ∞)
    (hf_meas : AEMeasurable f μ) :
    AEFinStronglyMeasurable f μ := by
  refine ⟨hf_meas.mk f, ENNReal.finStronglyMeasurable_of_measurable ?_ hf_meas.measurable_mk,
    hf_meas.ae_eq_mk⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hf_meas : AEMeasurable f μ
    ⊢ Ne (MeasureTheory.lintegral μ fun x => AEMeasurable.mk f hf_meas x) Top.top
  -/
  rwa [lintegral_congr_ae hf_meas.ae_eq_mk.symm]
  /-
    🎉 no goals
  -/

