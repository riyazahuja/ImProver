/-- Measurable equivalence between `ℂ` and `ℝ² = Fin 2 → ℝ`. -/
def measurableEquivPi : ℂ ≃ᵐ (Fin 2 → ℝ) :=
  basisOneI.equivFun.toContinuousLinearEquiv.toHomeomorph.toMeasurableEquiv


@[simp]
theorem measurableEquivPi_apply (a : ℂ) :
    measurableEquivPi a = ![a.re, a.im] := rfl


@[simp]
theorem measurableEquivPi_symm_apply (p : (Fin 2) → ℝ) :
    measurableEquivPi.symm p = (p 0) + (p 1) * I := rfl


/-- Measurable equivalence between `ℂ` and `ℝ × ℝ`. -/
def measurableEquivRealProd : ℂ ≃ᵐ ℝ × ℝ :=
  equivRealProdCLM.toHomeomorph.toMeasurableEquiv


@[simp]
theorem measurableEquivRealProd_apply (a : ℂ) : measurableEquivRealProd a = (a.re, a.im) := rfl


@[simp]
theorem measurableEquivRealProd_symm_apply (p : ℝ × ℝ) :
    measurableEquivRealProd.symm p = {re := p.1, im := p.2} := rfl


                                     /-
                                       ⊢ MeasureTheory.Measure Complex
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
theorem volume_preserving_equiv_pi : MeasurePreserving measurableEquivPi := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    ⊢ MeasureTheory.MeasurePreserving (⇑Complex.measurableEquivPi) MeasureTheory.M …
  -/
  convert (measurableEquivPi.symm.measurable.measurePreserving volume).symm
  rw [← addHaarMeasure_eq_volume_pi, ← Basis.parallelepiped_basisFun, ← Basis.addHaar,
    measurableEquivPi, Homeomorph.toMeasurableEquiv_symm_coe,
    ContinuousLinearEquiv.symm_toHomeomorph, ContinuousLinearEquiv.coe_toHomeomorph,
    Basis.map_addHaar, eq_comm]
  /-
    case h.e'_6
    ⊢ Eq ((Pi.basisFun Real (Fin 2)).map Complex.basisOneI.equivFun.toContinuousLi …
  -/
  exact (Basis.addHaar_eq_iff _ _).mpr Complex.orthonormalBasisOneI.volume_parallelepiped
  /-
    🎉 no goals
  -/


                                            /-
                                              ⊢ MeasureTheory.Measure Complex
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
theorem volume_preserving_equiv_real_prod : MeasurePreserving measurableEquivRealProd :=
                                            /-
                                              🎉 no goals
                                            -/
  (volume_preserving_finTwoArrow ℝ).comp volume_preserving_equiv_pi


