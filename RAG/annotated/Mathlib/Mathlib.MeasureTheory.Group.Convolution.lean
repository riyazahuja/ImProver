/-- Multiplicative convolution of measures. -/
@[to_additive conv "Additive convolution of measures."]
noncomputable def mconv (μ : Measure M) (ν : Measure M) :
    Measure M := Measure.map (fun x : M × M ↦ x.1 * x.2) (μ.prod ν)


/-- Scoped notation for the multiplicative convolution of measures. -/
scoped[MeasureTheory] infix:80 " ∗ " => MeasureTheory.Measure.mconv


/-- Scoped notation for the additive convolution of measures. -/
scoped[MeasureTheory] infix:80 " ∗ " => MeasureTheory.Measure.conv


/-- Convolution of the dirac measure at 1 with a measure μ returns μ. -/
@[to_additive (attr := simp)]
theorem dirac_one_mconv [MeasurableMul₂ M] (μ : Measure M) [SFinite μ] :
    (Measure.dirac 1) ∗ μ = μ := by
  /-
    M : Type u_1
    inst✝³ : Monoid M
    inst✝² : MeasurableSpace M
    inst✝¹ : MeasurableMul₂ M
    μ : MeasureTheory.Measure M
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq ((MeasureTheory.Measure.dirac 1).mconv μ) μ
  -/
  unfold mconv
  /-
    M : Type u_1
    inst✝³ : Monoid M
    inst✝² : MeasurableSpace M
    inst✝¹ : MeasurableMul₂ M
    μ : MeasureTheory.Measure M
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HMul.hMul x.1 x.2) ((MeasureTheory.M …
  -/
  rw [MeasureTheory.Measure.dirac_prod, map_map (by fun_prop)]
    /-
      M : Type u_1
      inst✝³ : Monoid M
      inst✝² : MeasurableSpace M
      inst✝¹ : MeasurableMul₂ M
      μ : MeasureTheory.Measure M
      inst✝ : MeasureTheory.SFinite μ
      ⊢ Eq (MeasureTheory.Measure.map (Function.comp (fun x => HMul.hMul x.1 x.2) (P …
    -/
  · simp only [Function.comp_def, one_mul, map_id']
    /-
      🎉 no goals
    -/
  /-
    M : Type u_1
    inst✝³ : Monoid M
    inst✝² : MeasurableSpace M
    inst✝¹ : MeasurableMul₂ M
    μ : MeasureTheory.Measure M
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Measurable (Prod.mk 1)
  -/
  fun_prop
  /-
    🎉 no goals
  -/


/-- Convolution of a measure μ with the dirac measure at 1 returns μ. -/
@[to_additive (attr := simp)]
theorem mconv_dirac_one [MeasurableMul₂ M]
    (μ : Measure M) [SFinite μ] : μ ∗ (Measure.dirac 1) = μ := by
  /-
    M : Type u_1
    inst✝³ : Monoid M
    inst✝² : MeasurableSpace M
    inst✝¹ : MeasurableMul₂ M
    μ : MeasureTheory.Measure M
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq (μ.mconv (MeasureTheory.Measure.dirac 1)) μ
  -/
  unfold mconv
  /-
    M : Type u_1
    inst✝³ : Monoid M
    inst✝² : MeasurableSpace M
    inst✝¹ : MeasurableMul₂ M
    μ : MeasureTheory.Measure M
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HMul.hMul x.1 x.2) (μ.prod (MeasureT …
  -/
  rw [MeasureTheory.Measure.prod_dirac, map_map (by fun_prop)]
    /-
      M : Type u_1
      inst✝³ : Monoid M
      inst✝² : MeasurableSpace M
      inst✝¹ : MeasurableMul₂ M
      μ : MeasureTheory.Measure M
      inst✝ : MeasureTheory.SFinite μ
      ⊢ Eq (MeasureTheory.Measure.map (Function.comp (fun x => HMul.hMul x.1 x.2) fu …
    -/
  · simp only [Function.comp_def, mul_one, map_id']
    /-
      🎉 no goals
    -/
  /-
    M : Type u_1
    inst✝³ : Monoid M
    inst✝² : MeasurableSpace M
    inst✝¹ : MeasurableMul₂ M
    μ : MeasureTheory.Measure M
    inst✝ : MeasureTheory.SFinite μ
    ⊢ Measurable fun x => { fst := x, snd := 1 }
  -/
  fun_prop
  /-
    🎉 no goals
  -/


/-- Convolution of the zero measure with a measure μ returns the zero measure. -/
@[to_additive (attr := simp) conv_zero]
theorem mconv_zero (μ : Measure M) : (0 : Measure M) ∗ μ = (0 : Measure M) := by
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    inst✝ : MeasurableSpace M
    μ : MeasureTheory.Measure M
    ⊢ Eq (MeasureTheory.Measure.mconv 0 μ) 0
  -/
  unfold mconv
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    inst✝ : MeasurableSpace M
    μ : MeasureTheory.Measure M
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HMul.hMul x.1 x.2) (MeasureTheory.Me …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Convolution of a measure μ with the zero measure returns the zero measure. -/
@[to_additive (attr := simp) zero_conv]
theorem zero_mconv (μ : Measure M) : μ ∗ (0 : Measure M) = (0 : Measure M) := by
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    inst✝ : MeasurableSpace M
    μ : MeasureTheory.Measure M
    ⊢ Eq (μ.mconv 0) 0
  -/
  unfold mconv
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    inst✝ : MeasurableSpace M
    μ : MeasureTheory.Measure M
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HMul.hMul x.1 x.2) (μ.prod 0)) 0
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive conv_add]
theorem mconv_add [MeasurableMul₂ M] (μ : Measure M) (ν : Measure M) (ρ : Measure M) [SFinite μ]
    [SFinite ν] [SFinite ρ] : μ ∗ (ν + ρ) = μ ∗ ν + μ ∗ ρ := by
  /-
    M : Type u_1
    inst✝⁵ : Monoid M
    inst✝⁴ : MeasurableSpace M
    inst✝³ : MeasurableMul₂ M
    μ ν ρ : MeasureTheory.Measure M
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ Eq (μ.mconv (HAdd.hAdd ν ρ)) (HAdd.hAdd (μ.mconv ν) (μ.mconv ρ))
  -/
  unfold mconv
  /-
    M : Type u_1
    inst✝⁵ : Monoid M
    inst✝⁴ : MeasurableSpace M
    inst✝³ : MeasurableMul₂ M
    μ ν ρ : MeasureTheory.Measure M
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HMul.hMul x.1 x.2) (μ.prod (HAdd.hAd …
  -/
  rw [prod_add, map_add]
  /-
    case hf
    M : Type u_1
    inst✝⁵ : Monoid M
    inst✝⁴ : MeasurableSpace M
    inst✝³ : MeasurableMul₂ M
    μ ν ρ : MeasureTheory.Measure M
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ Measurable fun x => HMul.hMul x.1 x.2
  -/
  fun_prop
  /-
    🎉 no goals
  -/


@[to_additive add_conv]
theorem add_mconv [MeasurableMul₂ M] (μ : Measure M) (ν : Measure M) (ρ : Measure M) [SFinite μ]
    [SFinite ν] [SFinite ρ] : (μ + ν) ∗ ρ = μ ∗ ρ + ν ∗ ρ := by
  /-
    M : Type u_1
    inst✝⁵ : Monoid M
    inst✝⁴ : MeasurableSpace M
    inst✝³ : MeasurableMul₂ M
    μ ν ρ : MeasureTheory.Measure M
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ Eq ((HAdd.hAdd μ ν).mconv ρ) (HAdd.hAdd (μ.mconv ρ) (ν.mconv ρ))
  -/
  unfold mconv
  /-
    M : Type u_1
    inst✝⁵ : Monoid M
    inst✝⁴ : MeasurableSpace M
    inst✝³ : MeasurableMul₂ M
    μ ν ρ : MeasureTheory.Measure M
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HMul.hMul x.1 x.2) ((HAdd.hAdd μ ν). …
  -/
  rw [add_prod, map_add]
  /-
    case hf
    M : Type u_1
    inst✝⁵ : Monoid M
    inst✝⁴ : MeasurableSpace M
    inst✝³ : MeasurableMul₂ M
    μ ν ρ : MeasureTheory.Measure M
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasureTheory.SFinite ν
    inst✝ : MeasureTheory.SFinite ρ
    ⊢ Measurable fun x => HMul.hMul x.1 x.2
  -/
  fun_prop
  /-
    🎉 no goals
  -/


/-- To get commutativity, we need the underlying multiplication to be commutative. -/
@[to_additive conv_comm]
theorem mconv_comm {M : Type*} [CommMonoid M] [MeasurableSpace M] [MeasurableMul₂ M] (μ : Measure M)
    (ν : Measure M) [SFinite μ] [SFinite ν] : μ ∗ ν = ν ∗ μ := by
  /-
    M : Type u_2
    inst✝⁴ : CommMonoid M
    inst✝³ : MeasurableSpace M
    inst✝² : MeasurableMul₂ M
    μ ν : MeasureTheory.Measure M
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    ⊢ Eq (μ.mconv ν) (ν.mconv μ)
  -/
  unfold mconv
  /-
    M : Type u_2
    inst✝⁴ : CommMonoid M
    inst✝³ : MeasurableSpace M
    inst✝² : MeasurableMul₂ M
    μ ν : MeasureTheory.Measure M
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HMul.hMul x.1 x.2) (μ.prod ν)) (Meas …
  -/
  rw [← prod_swap, map_map (by fun_prop)]
    /-
      M : Type u_2
      inst✝⁴ : CommMonoid M
      inst✝³ : MeasurableSpace M
      inst✝² : MeasurableMul₂ M
      μ ν : MeasureTheory.Measure M
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : MeasureTheory.SFinite ν
      ⊢ Eq (MeasureTheory.Measure.map (Function.comp (fun x => HMul.hMul x.1 x.2) Pr …
    -/
  · simp [Function.comp_def, mul_comm]
    /-
      🎉 no goals
    -/
  /-
    M : Type u_2
    inst✝⁴ : CommMonoid M
    inst✝³ : MeasurableSpace M
    inst✝² : MeasurableMul₂ M
    μ ν : MeasureTheory.Measure M
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    ⊢ Measurable Prod.swap
  -/
  fun_prop
  /-
    🎉 no goals
  -/


/-- Convolution of SFinite maps is SFinite. -/
@[to_additive sfinite_conv_of_sfinite]
instance sfinite_mconv_of_sfinite (μ : Measure M) (ν : Measure M) [SFinite μ] [SFinite ν] :
    SFinite (μ ∗ ν) := inferInstanceAs <| SFinite ((μ.prod ν).map fun (x : M × M) ↦ x.1 * x.2)


@[to_additive finite_of_finite_conv]
instance finite_of_finite_mconv (μ : Measure M) (ν : Measure M) [IsFiniteMeasure μ]
    [IsFiniteMeasure ν] : IsFiniteMeasure (μ ∗ ν) := by
  have h : (μ ∗ ν) Set.univ < ⊤ := by
    unfold mconv
    exact IsFiniteMeasure.measure_univ_lt_top
  /-
    M : Type u_1
    inst✝³ : Monoid M
    inst✝² : MeasurableSpace M
    μ ν : MeasureTheory.Measure M
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h : LT.lt ((μ.mconv ν) Set.univ) Top.top
    ⊢ MeasureTheory.IsFiniteMeasure (μ.mconv ν)
  -/
  exact {measure_univ_lt_top := h}
  /-
    🎉 no goals
  -/


@[to_additive probabilitymeasure_of_probabilitymeasures_conv]
instance probabilitymeasure_of_probabilitymeasures_mconv (μ : Measure M) (ν : Measure M)
    [MeasurableMul₂ M] [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    IsProbabilityMeasure (μ ∗ ν) :=
                                             /-
                                               M : Type u_1
                                               inst✝⁴ : Monoid M
                                               inst✝³ : MeasurableSpace M
                                               μ ν : MeasureTheory.Measure M
                                               inst✝² : MeasurableMul₂ M
                                               inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
                                               inst✝ : MeasureTheory.IsProbabilityMeasure ν
                                               ⊢ AEMeasurable (fun x => HMul.hMul x.1 x.2) (μ.prod ν)
                                             -/
  MeasureTheory.isProbabilityMeasure_map (by fun_prop)
                                             /-
                                               🎉 no goals
                                             -/


