/-- Translating a function by left-multiplication does not change its Lebesgue integral
with respect to a left-invariant measure. -/
@[to_additive
      "Translating a function by left-addition does not change its Lebesgue integral with
      respect to a left-invariant measure."]
theorem lintegral_mul_left_eq_self [IsMulLeftInvariant μ] (f : G → ℝ≥0∞) (g : G) :
    (∫⁻ x, f (g * x) ∂μ) = ∫⁻ x, f x ∂μ := by
  /-
    G : Type u_1
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulLeftInvariant
    f : G → ENNReal
    g : G
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f (HMul.hMul g x)) (MeasureTheory.lin …
  -/
  convert (lintegral_map_equiv f <| MeasurableEquiv.mulLeft g).symm
  /-
    case h.e'_3.h.e'_3
    G : Type u_1
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulLeftInvariant
    f : G → ENNReal
    g : G
    ⊢ Eq μ (MeasureTheory.Measure.map (⇑(MeasurableEquiv.mulLeft g)) μ)
  -/
  simp [map_mul_left_eq_self μ g]
  /-
    🎉 no goals
  -/


/-- Translating a function by right-multiplication does not change its Lebesgue integral
with respect to a right-invariant measure. -/
@[to_additive
      "Translating a function by right-addition does not change its Lebesgue integral with
      respect to a right-invariant measure."]
theorem lintegral_mul_right_eq_self [IsMulRightInvariant μ] (f : G → ℝ≥0∞) (g : G) :
    (∫⁻ x, f (x * g) ∂μ) = ∫⁻ x, f x ∂μ := by
  /-
    G : Type u_1
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulRightInvariant
    f : G → ENNReal
    g : G
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f (HMul.hMul x g)) (MeasureTheory.lin …
  -/
  convert (lintegral_map_equiv f <| MeasurableEquiv.mulRight g).symm using 1
  /-
    case h.e'_3
    G : Type u_1
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulRightInvariant
    f : G → ENNReal
    g : G
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f x) (MeasureTheory.lintegral (Measur …
  -/
  simp [map_mul_right_eq_self μ g]
  /-
    🎉 no goals
  -/


@[to_additive] -- Porting note: was `@[simp]`
theorem lintegral_div_right_eq_self [IsMulRightInvariant μ] (f : G → ℝ≥0∞) (g : G) :
    (∫⁻ x, f (x / g) ∂μ) = ∫⁻ x, f x ∂μ := by
  /-
    G : Type u_1
    inst✝³ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulRightInvariant
    f : G → ENNReal
    g : G
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f (HDiv.hDiv x g)) (MeasureTheory.lin …
  -/
  simp_rw [div_eq_mul_inv, lintegral_mul_right_eq_self f g⁻¹]
  /-
    🎉 no goals
  -/


/-- For nonzero regular left invariant measures, the integral of a continuous nonnegative function
  `f` is 0 iff `f` is 0. -/
@[to_additive
      "For nonzero regular left invariant measures, the integral of a continuous nonnegative
      function `f` is 0 iff `f` is 0."]
theorem lintegral_eq_zero_of_isMulLeftInvariant [Regular μ] [NeZero μ] {f : G → ℝ≥0∞}
    (hf : Continuous f) : ∫⁻ x, f x ∂μ = 0 ↔ f = 0 := by
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalGroup G
    inst✝³ : BorelSpace G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : μ.Regular
    inst✝ : NeZero μ
    f : G → ENNReal
    hf : Continuous f
    ⊢ Iff (Eq (MeasureTheory.lintegral μ fun x => f x) 0) (Eq f 0)
  -/
  rw [lintegral_eq_zero_iff hf.measurable, hf.ae_eq_iff_eq μ continuous_zero]
  /-
    🎉 no goals
  -/


