/-- The map `(x, y) ↦ (x, xy)` as a `MeasurableEquiv`. -/
@[to_additive "The map `(x, y) ↦ (x, x + y)` as a `MeasurableEquiv`."]
protected def MeasurableEquiv.shearMulRight [MeasurableInv G] : G × G ≃ᵐ G × G :=
  { Equiv.prodShear (Equiv.refl _) Equiv.mulLeft with
    measurable_toFun := measurable_fst.prod_mk measurable_mul
    measurable_invFun := measurable_fst.prod_mk <| measurable_fst.inv.mul measurable_snd }


/-- The map `(x, y) ↦ (x, y / x)` as a `MeasurableEquiv` with as inverse `(x, y) ↦ (x, yx)` -/
@[to_additive
"The map `(x, y) ↦ (x, y - x)` as a `MeasurableEquiv` with as inverse `(x, y) ↦ (x, y + x)`."]
protected def MeasurableEquiv.shearDivRight [MeasurableInv G] : G × G ≃ᵐ G × G :=
  { Equiv.prodShear (Equiv.refl _) Equiv.divRight with
    measurable_toFun := measurable_fst.prod_mk <| measurable_snd.div measurable_fst
    measurable_invFun := measurable_fst.prod_mk <| measurable_snd.mul measurable_fst }


/-- The multiplicative shear mapping `(x, y) ↦ (x, xy)` preserves the measure `μ × ν`.
This condition is part of the definition of a measurable group in [Halmos, §59].
There, the map in this lemma is called `S`. -/
@[to_additive measurePreserving_prod_add
" The shear mapping `(x, y) ↦ (x, x + y)` preserves the measure `μ × ν`. "]
theorem measurePreserving_prod_mul [IsMulLeftInvariant ν] :
    MeasurePreserving (fun z : G × G => (z.1, z.1 * z.2)) (μ.prod ν) (μ.prod ν) :=
  (MeasurePreserving.id μ).skew_product measurable_mul <|
    Filter.Eventually.of_forall <| map_mul_left_eq_self ν


/-- The map `(x, y) ↦ (y, yx)` sends the measure `μ × ν` to `ν × μ`.
This is the map `SR` in [Halmos, §59].
`S` is the map `(x, y) ↦ (x, xy)` and `R` is `Prod.swap`. -/
@[to_additive measurePreserving_prod_add_swap
" The map `(x, y) ↦ (y, y + x)` sends the measure `μ × ν` to `ν × μ`. "]
theorem measurePreserving_prod_mul_swap [IsMulLeftInvariant μ] :
    MeasurePreserving (fun z : G × G => (z.2, z.2 * z.1)) (μ.prod ν) (ν.prod μ) :=
  (measurePreserving_prod_mul ν μ).comp measurePreserving_swap


@[to_additive]
theorem measurable_measure_mul_right (hs : MeasurableSet s) :
    Measurable fun x => μ ((fun y => y * x) ⁻¹' s) := by
  suffices
    Measurable fun y =>
      μ ((fun x => (x, y)) ⁻¹' ((fun z : G × G => ((1 : G), z.1 * z.2)) ⁻¹' univ ×ˢ s))
    by convert this using 1; ext1 x; congr 1 with y : 1; simp
  /-
    G : Type u_1
    inst✝³ : MeasurableSpace G
    inst✝² : Group G
    inst✝¹ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝ : MeasureTheory.SFinite μ
    s : Set G
    hs : MeasurableSet s
    ⊢ Measurable fun y => μ (Set.preimage (fun x => { fst := x, snd := y }) (Set.p …
  -/
  apply measurable_measure_prod_mk_right
  /-
    case hs
    G : Type u_1
    inst✝³ : MeasurableSpace G
    inst✝² : Group G
    inst✝¹ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝ : MeasureTheory.SFinite μ
    s : Set G
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.preimage (fun z => { fst := 1, snd := HMul.hMul z.1 z.2 } …
  -/
  apply measurable_const.prod_mk measurable_mul (MeasurableSet.univ.prod hs)
  /-
    G : Type u_1
    inst✝³ : MeasurableSpace G
    inst✝² : Group G
    inst✝¹ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝ : MeasureTheory.SFinite μ
    s : Set G
    hs : MeasurableSet s
    ⊢ MeasurableSpace G
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The map `(x, y) ↦ (x, x⁻¹y)` is measure-preserving.
This is the function `S⁻¹` in [Halmos, §59],
where `S` is the map `(x, y) ↦ (x, xy)`. -/
@[to_additive measurePreserving_prod_neg_add
"The map `(x, y) ↦ (x, - x + y)` is measure-preserving."]
theorem measurePreserving_prod_inv_mul [IsMulLeftInvariant ν] :
    MeasurePreserving (fun z : G × G => (z.1, z.1⁻¹ * z.2)) (μ.prod ν) (μ.prod ν) :=
  (measurePreserving_prod_mul μ ν).symm <| MeasurableEquiv.shearMulRight G


/-- The map `(x, y) ↦ (y, y⁻¹x)` sends `μ × ν` to `ν × μ`.
This is the function `S⁻¹R` in [Halmos, §59],
where `S` is the map `(x, y) ↦ (x, xy)` and `R` is `Prod.swap`. -/
@[to_additive measurePreserving_prod_neg_add_swap
"The map `(x, y) ↦ (y, - y + x)` sends `μ × ν` to `ν × μ`."]
theorem measurePreserving_prod_inv_mul_swap :
    MeasurePreserving (fun z : G × G => (z.2, z.2⁻¹ * z.1)) (μ.prod ν) (ν.prod μ) :=
  (measurePreserving_prod_inv_mul ν μ).comp measurePreserving_swap


/-- The map `(x, y) ↦ (yx, x⁻¹)` is measure-preserving.
This is the function `S⁻¹RSR` in [Halmos, §59],
where `S` is the map `(x, y) ↦ (x, xy)` and `R` is `Prod.swap`. -/
@[to_additive measurePreserving_add_prod_neg
"The map `(x, y) ↦ (y + x, - x)` is measure-preserving."]
theorem measurePreserving_mul_prod_inv [IsMulLeftInvariant ν] :
    MeasurePreserving (fun z : G × G => (z.2 * z.1, z.1⁻¹)) (μ.prod ν) (μ.prod ν) := by
  convert (measurePreserving_prod_inv_mul_swap ν μ).comp (measurePreserving_prod_mul_swap μ ν)
    using 1
  /-
    case h.e'_5
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    ⊢ Eq (fun z => { fst := HMul.hMul z.2 z.1, snd := Inv.inv z.1 }) (Function.com …
  -/
  ext1 ⟨x, y⟩
  /-
    case h.e'_5.h.mk
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    x y : G
    ⊢ Eq { fst := HMul.hMul { fst := x, snd := y }.2 { fst := x, snd := y }.1, snd …
  -/
  simp_rw [Function.comp_apply, mul_inv_rev, inv_mul_cancel_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem quasiMeasurePreserving_inv : QuasiMeasurePreserving (Inv.inv : G → G) μ μ := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving Inv.inv μ μ
  -/
  refine ⟨measurable_inv, AbsolutelyContinuous.mk fun s hsm hμs => ?_⟩
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    s : Set G
    hsm : MeasurableSet s
    hμs : Eq (μ s) 0
    ⊢ Eq ((MeasureTheory.Measure.map Inv.inv μ) s) 0
  -/
  rw [map_apply measurable_inv hsm, inv_preimage]
  have hf : Measurable fun z : G × G => (z.2 * z.1, z.1⁻¹) :=
    (measurable_snd.mul measurable_fst).prod_mk measurable_fst.inv
  suffices map (fun z : G × G => (z.2 * z.1, z.1⁻¹)) (μ.prod μ) (s⁻¹ ×ˢ s⁻¹) = 0 by
    simpa only [(measurePreserving_mul_prod_inv μ μ).map_eq, prod_prod, mul_eq_zero (M₀ := ℝ≥0∞),
      or_self_iff] using this
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    s : Set G
    hsm : MeasurableSet s
    hμs : Eq (μ s) 0
    hf : Measurable fun z => { fst := HMul.hMul z.2 z.1, snd := Inv.inv z.1 }
    ⊢ Eq ((MeasureTheory.Measure.map (fun z => { fst := HMul.hMul z.2 z.1, snd :=  …
  -/
  have hsm' : MeasurableSet (s⁻¹ ×ˢ s⁻¹) := hsm.inv.prod hsm.inv
  simp_rw [map_apply hf hsm', prod_apply_symm (μ := μ) (ν := μ) (hf hsm'), preimage_preimage,
    mk_preimage_prod, inv_preimage, inv_inv, measure_mono_null inter_subset_right hμs,
    lintegral_zero]


@[to_additive (attr := simp)]
theorem measure_inv_null : μ s⁻¹ = 0 ↔ μ s = 0 := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    s : Set G
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    ⊢ Iff (Eq (μ (Inv.inv s)) 0) (Eq (μ s) 0)
  -/
  refine ⟨fun hs => ?_, (quasiMeasurePreserving_inv μ).preimage_null⟩
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    s : Set G
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    hs : Eq (μ (Inv.inv s)) 0
    ⊢ Eq (μ s) 0
  -/
  rw [← inv_inv s]
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    s : Set G
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    hs : Eq (μ (Inv.inv s)) 0
    ⊢ Eq (μ (Inv.inv (Inv.inv s))) 0
  -/
  exact (quasiMeasurePreserving_inv μ).preimage_null hs
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem inv_ae : (ae μ)⁻¹ = ae μ := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    ⊢ Eq (Inv.inv (MeasureTheory.ae μ)) (MeasureTheory.ae μ)
  -/
  refine le_antisymm (quasiMeasurePreserving_inv μ).tendsto_ae ?_
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    ⊢ LE.le (MeasureTheory.ae μ) (Inv.inv (MeasureTheory.ae μ))
  -/
  nth_rewrite 1 [← inv_inv (ae μ)]
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    ⊢ LE.le (Inv.inv (Inv.inv (MeasureTheory.ae μ))) (Inv.inv (MeasureTheory.ae μ))
  -/
  exact Filter.map_mono (quasiMeasurePreserving_inv μ).tendsto_ae
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem eventuallyConst_inv_set_ae :
    EventuallyConst (s⁻¹ : Set G) (ae μ) ↔ EventuallyConst s (ae μ) := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    s : Set G
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    ⊢ Iff (Filter.EventuallyConst (Inv.inv s) (MeasureTheory.ae μ)) (Filter.Eventu …
  -/
  rw [← inv_preimage, eventuallyConst_preimage, Filter.map_inv, inv_ae]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inv_absolutelyContinuous : μ.inv ≪ μ :=
  (quasiMeasurePreserving_inv μ).absolutelyContinuous


@[to_additive]
theorem absolutelyContinuous_inv : μ ≪ μ.inv := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    ⊢ μ.AbsolutelyContinuous μ.inv
  -/
  refine AbsolutelyContinuous.mk fun s _ => ?_
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    s : Set G
    x✝ : MeasurableSet s
    ⊢ Eq (μ.inv s) 0 → Eq (μ s) 0
  -/
  simp_rw [inv_apply μ s, measure_inv_null, imp_self]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem lintegral_lintegral_mul_inv [IsMulLeftInvariant ν] (f : G → G → ℝ≥0∞)
    (hf : AEMeasurable (uncurry f) (μ.prod ν)) :
    (∫⁻ x, ∫⁻ y, f (y * x) x⁻¹ ∂ν ∂μ) = ∫⁻ x, ∫⁻ y, f x y ∂ν ∂μ := by
  have h : Measurable fun z : G × G => (z.2 * z.1, z.1⁻¹) :=
    (measurable_snd.mul measurable_fst).prod_mk measurable_fst.inv
  have h2f : AEMeasurable (uncurry fun x y => f (y * x) x⁻¹) (μ.prod ν) :=
    hf.comp_quasiMeasurePreserving (measurePreserving_mul_prod_inv μ ν).quasiMeasurePreserving
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    f : G → G → ENNReal
    hf : AEMeasurable (Function.uncurry f) (μ.prod ν)
    h : Measurable fun z => { fst := HMul.hMul z.2 z.1, snd := Inv.inv z.1 }
    h2f : AEMeasurable (Function.uncurry fun x y => f (HMul.hMul y x) (Inv.inv x)) …
    ⊢ Eq (MeasureTheory.lintegral μ fun x => MeasureTheory.lintegral ν fun y => f  …
  -/
  simp_rw [lintegral_lintegral h2f, lintegral_lintegral hf]
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    f : G → G → ENNReal
    hf : AEMeasurable (Function.uncurry f) (μ.prod ν)
    h : Measurable fun z => { fst := HMul.hMul z.2 z.1, snd := Inv.inv z.1 }
    h2f : AEMeasurable (Function.uncurry fun x y => f (HMul.hMul y x) (Inv.inv x)) …
    ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f (HMul.hMul z.2 z.1) (Inv.i …
  -/
  conv_rhs => rw [← (measurePreserving_mul_prod_inv μ ν).map_eq]
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    f : G → G → ENNReal
    hf : AEMeasurable (Function.uncurry f) (μ.prod ν)
    h : Measurable fun z => { fst := HMul.hMul z.2 z.1, snd := Inv.inv z.1 }
    h2f : AEMeasurable (Function.uncurry fun x y => f (HMul.hMul y x) (Inv.inv x)) …
    ⊢ Eq (MeasureTheory.lintegral (μ.prod ν) fun z => f (HMul.hMul z.2 z.1) (Inv.i …
  -/
  symm
  exact
    lintegral_map' (hf.mono' (measurePreserving_mul_prod_inv μ ν).map_eq.absolutelyContinuous)
      h.aemeasurable


@[to_additive]
theorem measure_mul_right_null (y : G) : μ ((fun x => x * y) ⁻¹' s) = 0 ↔ μ s = 0 :=
  calc
    μ ((fun x => x * y) ⁻¹' s) = 0 ↔ μ ((fun x => y⁻¹ * x) ⁻¹' s⁻¹)⁻¹ = 0 := by
      /-
        G : Type u_1
        inst✝⁵ : MeasurableSpace G
        inst✝⁴ : Group G
        inst✝³ : MeasurableMul₂ G
        μ : MeasureTheory.Measure G
        inst✝² : MeasureTheory.SFinite μ
        s : Set G
        inst✝¹ : MeasurableInv G
        inst✝ : μ.IsMulLeftInvariant
        y : G
        ⊢ Iff (Eq (μ (Set.preimage (fun x => HMul.hMul x y) s)) 0) (Eq (μ (Inv.inv (Se …
      -/
      simp_rw [← inv_preimage, preimage_preimage, mul_inv_rev, inv_inv]
      /-
        🎉 no goals
      -/
                      /-
                        G : Type u_1
                        inst✝⁵ : MeasurableSpace G
                        inst✝⁴ : Group G
                        inst✝³ : MeasurableMul₂ G
                        μ : MeasureTheory.Measure G
                        inst✝² : MeasureTheory.SFinite μ
                        s : Set G
                        inst✝¹ : MeasurableInv G
                        inst✝ : μ.IsMulLeftInvariant
                        y : G
                        ⊢ Iff (Eq (μ (Inv.inv (Set.preimage (fun x => HMul.hMul (Inv.inv y) x) (Inv.in …
                      -/
    _ ↔ μ s = 0 := by simp only [measure_inv_null μ, measure_preimage_mul]
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem measure_mul_right_ne_zero (h2s : μ s ≠ 0) (y : G) : μ ((fun x => x * y) ⁻¹' s) ≠ 0 :=
  (not_congr (measure_mul_right_null μ y)).mpr h2s


@[to_additive]
theorem absolutelyContinuous_map_mul_right (g : G) : μ ≪ map (· * g) μ := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    ⊢ μ.AbsolutelyContinuous (MeasureTheory.Measure.map (fun x => HMul.hMul x g) μ)
  -/
  refine AbsolutelyContinuous.mk fun s hs => ?_
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    s : Set G
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.map (fun x => HMul.hMul x g) μ) s) 0 → Eq (μ s) 0
  -/
  rw [map_apply (measurable_mul_const g) hs, measure_mul_right_null]; exact id
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[to_additive]
theorem absolutelyContinuous_map_div_left (g : G) : μ ≪ map (fun h => g / h) μ := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    ⊢ μ.AbsolutelyContinuous (MeasureTheory.Measure.map (fun h => HDiv.hDiv g h) μ)
  -/
  simp_rw [div_eq_mul_inv]
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    ⊢ μ.AbsolutelyContinuous (MeasureTheory.Measure.map (fun h => HMul.hMul g (Inv …
  -/
  erw [← map_map (measurable_const_mul g) measurable_inv]
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    ⊢ μ.AbsolutelyContinuous (MeasureTheory.Measure.map (fun x => HMul.hMul g x) ( …
  -/
  conv_lhs => rw [← map_mul_left_eq_self μ g]
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    ⊢ (MeasureTheory.Measure.map (fun x => HMul.hMul g x) μ).AbsolutelyContinuous  …
  -/
  exact (absolutelyContinuous_inv μ).map (measurable_const_mul g)
  /-
    🎉 no goals
  -/


/-- This is the computation performed in the proof of [Halmos, §60 Th. A]. -/
@[to_additive "This is the computation performed in the proof of [Halmos, §60 Th. A]."]
theorem measure_mul_lintegral_eq [IsMulLeftInvariant ν] (sm : MeasurableSet s) (f : G → ℝ≥0∞)
    (hf : Measurable f) : (μ s * ∫⁻ y, f y ∂ν) = ∫⁻ x, ν ((fun z => z * x) ⁻¹' s) * f x⁻¹ ∂μ := by
  rw [← setLIntegral_one, ← lintegral_indicator sm,
    ← lintegral_lintegral_mul (measurable_const.indicator sm).aemeasurable hf.aemeasurable,
    ← lintegral_lintegral_mul_inv μ ν]
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    s : Set G
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    sm : MeasurableSet s
    f : G → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral μ fun x => MeasureTheory.lintegral ν fun y => HM …
  -/
  swap
  · exact (((measurable_const.indicator sm).comp measurable_fst).mul
      (hf.comp measurable_snd)).aemeasurable
  have ms :
    ∀ x : G, Measurable fun y => ((fun z => z * x) ⁻¹' s).indicator (fun _ => (1 : ℝ≥0∞)) y :=
    fun x => measurable_const.indicator (measurable_mul_const _ sm)
  have : ∀ x y, s.indicator (fun _ : G => (1 : ℝ≥0∞)) (y * x) =
      ((fun z => z * x) ⁻¹' s).indicator (fun b : G => 1) y := by
    intro x y; symm; convert indicator_comp_right (M := ℝ≥0∞) fun y => y * x using 2; ext1; rfl
  simp_rw [this, lintegral_mul_const _ (ms _), lintegral_indicator (measurable_mul_const _ sm),
    setLIntegral_one]


/-- Any two nonzero left-invariant measures are absolutely continuous w.r.t. each other. -/
@[to_additive
" Any two nonzero left-invariant measures are absolutely continuous w.r.t. each other. "]
theorem absolutelyContinuous_of_isMulLeftInvariant [IsMulLeftInvariant ν] (hν : ν ≠ 0) : μ ≪ ν := by
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    hν : Ne ν 0
    ⊢ μ.AbsolutelyContinuous ν
  -/
  refine AbsolutelyContinuous.mk fun s sm hνs => ?_
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    hν : Ne ν 0
    s : Set G
    sm : MeasurableSet s
    hνs : Eq (ν s) 0
    ⊢ Eq (μ s) 0
  -/
  have h1 := measure_mul_lintegral_eq μ ν sm 1 measurable_one
  simp_rw [Pi.one_apply, lintegral_one, mul_one, (measure_mul_right_null ν _).mpr hνs,
    lintegral_zero, mul_eq_zero (M₀ := ℝ≥0∞), measure_univ_eq_zero.not.mpr hν, or_false] at h1
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulLeftInvariant
    inst✝ : ν.IsMulLeftInvariant
    hν : Ne ν 0
    s : Set G
    sm : MeasurableSet s
    hνs : Eq (ν s) 0
    h1 : Eq (μ s) 0
    ⊢ Eq (μ s) 0
  -/
  exact h1
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ae_measure_preimage_mul_right_lt_top (hμs : μ' s ≠ ∞) :
    ∀ᵐ x ∂μ', ν' ((· * x) ⁻¹' s) < ∞ := by
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    hμs : Ne (μ' s) Top.top
    ⊢ Filter.Eventually (fun x => LT.lt (ν' (Set.preimage (fun x_1 => HMul.hMul x_ …
  -/
  wlog sm : MeasurableSet s generalizing s
  · filter_upwards [this ((measure_toMeasurable _).trans_ne hμs) (measurableSet_toMeasurable ..)]
      with x hx using lt_of_le_of_lt (by gcongr; apply subset_toMeasurable) hx
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s✝ : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    hμs : Ne (μ' s) Top.top
    sm : MeasurableSet s
    ⊢ Filter.Eventually (fun x => LT.lt (ν' (Set.preimage (fun x_1 => HMul.hMul x_ …
  -/
  refine ae_of_forall_measure_lt_top_ae_restrict' ν'.inv _ ?_
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s✝ : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    hμs : Ne (μ' s) Top.top
    sm : MeasurableSet s
    ⊢ ∀ (s_1 : Set G), MeasurableSet s_1 → LT.lt (μ' s_1) Top.top → LT.lt (ν'.inv  …
  -/
  intro A hA _ h3A
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s✝ : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    hμs : Ne (μ' s) Top.top
    sm : MeasurableSet s
    A : Set G
    hA : MeasurableSet A
    a✝ : LT.lt (μ' A) Top.top
    h3A : LT.lt (ν'.inv A) Top.top
    ⊢ Filter.Eventually (fun x => LT.lt (ν' (Set.preimage (fun x_1 => HMul.hMul x_ …
  -/
  simp only [ν'.inv_apply] at h3A
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s✝ : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    hμs : Ne (μ' s) Top.top
    sm : MeasurableSet s
    A : Set G
    hA : MeasurableSet A
    a✝ : LT.lt (μ' A) Top.top
    h3A : LT.lt (ν' (Inv.inv A)) Top.top
    ⊢ Filter.Eventually (fun x => LT.lt (ν' (Set.preimage (fun x_1 => HMul.hMul x_ …
  -/
  apply ae_lt_top (measurable_measure_mul_right ν' sm)
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s✝ : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    hμs : Ne (μ' s) Top.top
    sm : MeasurableSet s
    A : Set G
    hA : MeasurableSet A
    a✝ : LT.lt (μ' A) Top.top
    h3A : LT.lt (ν' (Inv.inv A)) Top.top
    ⊢ Ne (MeasureTheory.lintegral (μ'.restrict A) fun x => ν' (Set.preimage (fun y …
  -/
  have h1 := measure_mul_lintegral_eq μ' ν' sm (A⁻¹.indicator 1) (measurable_one.indicator hA.inv)
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s✝ : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    hμs : Ne (μ' s) Top.top
    sm : MeasurableSet s
    A : Set G
    hA : MeasurableSet A
    a✝ : LT.lt (μ' A) Top.top
    h3A : LT.lt (ν' (Inv.inv A)) Top.top
    h1 : Eq (HMul.hMul (μ' s) (MeasureTheory.lintegral ν' fun y => (Inv.inv A).ind …
    ⊢ Ne (MeasureTheory.lintegral (μ'.restrict A) fun x => ν' (Set.preimage (fun y …
  -/
  rw [lintegral_indicator hA.inv] at h1
  simp_rw [Pi.one_apply, setLIntegral_one, ← image_inv_eq_inv, indicator_image inv_injective,
    image_inv_eq_inv, ← indicator_mul_right _ fun x => ν' ((· * x) ⁻¹' s), Function.comp,
    Pi.one_apply, mul_one] at h1
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s✝ : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    hμs : Ne (μ' s) Top.top
    sm : MeasurableSet s
    A : Set G
    hA : MeasurableSet A
    a✝ : LT.lt (μ' A) Top.top
    h3A : LT.lt (ν' (Inv.inv A)) Top.top
    h1 : Eq (HMul.hMul (μ' s) (ν' (Inv.inv A))) (MeasureTheory.lintegral μ' fun x  …
    ⊢ Ne (MeasureTheory.lintegral (μ'.restrict A) fun x => ν' (Set.preimage (fun y …
  -/
  rw [← lintegral_indicator hA, ← h1]
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s✝ : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    hμs : Ne (μ' s) Top.top
    sm : MeasurableSet s
    A : Set G
    hA : MeasurableSet A
    a✝ : LT.lt (μ' A) Top.top
    h3A : LT.lt (ν' (Inv.inv A)) Top.top
    h1 : Eq (HMul.hMul (μ' s) (ν' (Inv.inv A))) (MeasureTheory.lintegral μ' fun x  …
    ⊢ Ne (HMul.hMul (μ' s) (ν' (Inv.inv A))) Top.top
  -/
  exact ENNReal.mul_ne_top hμs h3A.ne
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ae_measure_preimage_mul_right_lt_top_of_ne_zero (h2s : ν' s ≠ 0) (h3s : ν' s ≠ ∞) :
    ∀ᵐ x ∂μ', ν' ((fun y => y * x) ⁻¹' s) < ∞ := by
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    ⊢ Filter.Eventually (fun x => LT.lt (ν' (Set.preimage (fun y => HMul.hMul y x) …
  -/
  refine (ae_measure_preimage_mul_right_lt_top ν' ν' h3s).filter_mono ?_
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    ⊢ LE.le (MeasureTheory.ae μ') (MeasureTheory.ae ν')
  -/
  refine (absolutelyContinuous_of_isMulLeftInvariant μ' ν' ?_).ae_le
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    ⊢ Ne ν' 0
  -/
  refine mt ?_ h2s
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    ⊢ Eq ν' 0 → Eq (ν' s) 0
  -/
  intro hν
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    hν : Eq ν' 0
    ⊢ Eq (ν' s) 0
  -/
  rw [hν, Measure.coe_zero, Pi.zero_apply]
  /-
    🎉 no goals
  -/


/-- A technical lemma relating two different measures. This is basically [Halmos, §60 Th. A].
  Note that if `f` is the characteristic function of a measurable set `t` this states that
  `μ t = c * μ s` for a constant `c` that does not depend on `μ`.

  Note: There is a gap in the last step of the proof in [Halmos].
  In the last line, the equality `g(x⁻¹)ν(sx⁻¹) = f(x)` holds if we can prove that
  `0 < ν(sx⁻¹) < ∞`. The first inequality follows from §59, Th. D, but the second inequality is
  not justified. We prove this inequality for almost all `x` in
  `MeasureTheory.ae_measure_preimage_mul_right_lt_top_of_ne_zero`. -/
@[to_additive
"A technical lemma relating two different measures. This is basically [Halmos, §60 Th. A]. Note that
if `f` is the characteristic function of a measurable set `t` this states that `μ t = c * μ s` for a
constant `c` that does not depend on `μ`.

Note: There is a gap in the last step of the proof in [Halmos]. In the last line, the equality
`g(-x) + ν(s - x) = f(x)` holds if we can prove that `0 < ν(s - x) < ∞`. The first inequality
follows from §59, Th. D, but the second inequality is not justified. We prove this inequality for
almost all `x` in `MeasureTheory.ae_measure_preimage_add_right_lt_top_of_ne_zero`."]
theorem measure_lintegral_div_measure (sm : MeasurableSet s) (h2s : ν' s ≠ 0) (h3s : ν' s ≠ ∞)
    (f : G → ℝ≥0∞) (hf : Measurable f) :
    (μ' s * ∫⁻ y, f y⁻¹ / ν' ((· * y⁻¹) ⁻¹' s) ∂ν') = ∫⁻ x, f x ∂μ' := by
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    sm : MeasurableSet s
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    f : G → ENNReal
    hf : Measurable f
    ⊢ Eq (HMul.hMul (μ' s) (MeasureTheory.lintegral ν' fun y => HDiv.hDiv (f (Inv. …
  -/
  set g := fun y => f y⁻¹ / ν' ((fun x => x * y⁻¹) ⁻¹' s)
  have hg : Measurable g :=
    (hf.comp measurable_inv).div ((measurable_measure_mul_right ν' sm).comp measurable_inv)
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    sm : MeasurableSet s
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    f : G → ENNReal
    hf : Measurable f
    g : G → ENNReal := fun y => HDiv.hDiv (f (Inv.inv y)) (ν' (Set.preimage (fun x …
    hg : Measurable g
    ⊢ Eq (HMul.hMul (μ' s) (MeasureTheory.lintegral ν' g)) (MeasureTheory.lintegra …
  -/
  simp_rw [measure_mul_lintegral_eq μ' ν' sm g hg, g, inv_inv]
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    sm : MeasurableSet s
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    f : G → ENNReal
    hf : Measurable f
    g : G → ENNReal := fun y => HDiv.hDiv (f (Inv.inv y)) (ν' (Set.preimage (fun x …
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral μ' fun x => HMul.hMul (ν' (Set.preimage (fun x_1 …
  -/
  refine lintegral_congr_ae ?_
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    sm : MeasurableSet s
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    f : G → ENNReal
    hf : Measurable f
    g : G → ENNReal := fun y => HDiv.hDiv (f (Inv.inv y)) (ν' (Set.preimage (fun x …
    hg : Measurable g
    ⊢ (MeasureTheory.ae μ').EventuallyEq (fun x => HMul.hMul (ν' (Set.preimage (fu …
  -/
  refine (ae_measure_preimage_mul_right_lt_top_of_ne_zero μ' ν' h2s h3s).mono fun x hx => ?_
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    sm : MeasurableSet s
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    f : G → ENNReal
    hf : Measurable f
    g : G → ENNReal := fun y => HDiv.hDiv (f (Inv.inv y)) (ν' (Set.preimage (fun x …
    hg : Measurable g
    x : G
    hx : LT.lt (ν' (Set.preimage (fun y => HMul.hMul y x) s)) Top.top
    ⊢ Eq ((fun x => HMul.hMul (ν' (Set.preimage (fun x_1 => HMul.hMul x_1 x) s)) ( …
  -/
  simp_rw [ENNReal.mul_div_cancel (measure_mul_right_ne_zero ν' h2s _) hx.ne]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem measure_mul_measure_eq (s t : Set G) (h2s : ν' s ≠ 0) (h3s : ν' s ≠ ∞) :
    μ' s * ν' t = ν' s * μ' t := by
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s t : Set G
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    ⊢ Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMul (ν' s) (μ' t))
  -/
  wlog hs : MeasurableSet s generalizing s
    /-
      case inr
      G : Type u_1
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : Group G
      inst✝⁵ : MeasurableMul₂ G
      inst✝⁴ : MeasurableInv G
      μ' ν' : MeasureTheory.Measure G
      inst✝³ : MeasureTheory.SigmaFinite μ'
      inst✝² : MeasureTheory.SigmaFinite ν'
      inst✝¹ : μ'.IsMulLeftInvariant
      inst✝ : ν'.IsMulLeftInvariant
      s t : Set G
      h2s : Ne (ν' s) 0
      h3s : Ne (ν' s) Top.top
      this : ∀ (s : Set G), Ne (ν' s) 0 → Ne (ν' s) Top.top → MeasurableSet s → Eq ( …
      hs : Not (MeasurableSet s)
      ⊢ Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMul (ν' s) (μ' t))
    -/
  · rcases exists_measurable_superset₂ μ' ν' s with ⟨s', -, hm, hμ, hν⟩
    /-
      case inr.intro.intro.intro.intro
      G : Type u_1
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : Group G
      inst✝⁵ : MeasurableMul₂ G
      inst✝⁴ : MeasurableInv G
      μ' ν' : MeasureTheory.Measure G
      inst✝³ : MeasureTheory.SigmaFinite μ'
      inst✝² : MeasureTheory.SigmaFinite ν'
      inst✝¹ : μ'.IsMulLeftInvariant
      inst✝ : ν'.IsMulLeftInvariant
      s t : Set G
      h2s : Ne (ν' s) 0
      h3s : Ne (ν' s) Top.top
      this : ∀ (s : Set G), Ne (ν' s) 0 → Ne (ν' s) Top.top → MeasurableSet s → Eq ( …
      hs : Not (MeasurableSet s)
      s' : Set G
      hm : MeasurableSet s'
      hμ : Eq (μ' s') (μ' s)
      hν : Eq (ν' s') (ν' s)
      ⊢ Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMul (ν' s) (μ' t))
    -/
                                        /-
                                          🎉 no goals
                                        -/
    rw [← hμ, ← hν, this s' _ _ hm] <;> rwa [hν]
                                        /-
                                          🎉 no goals
                                        -/
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    t s : Set G
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    hs : MeasurableSet s
    ⊢ Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMul (ν' s) (μ' t))
  -/
  wlog ht : MeasurableSet t generalizing t
    /-
      case inr
      G : Type u_1
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : Group G
      inst✝⁵ : MeasurableMul₂ G
      inst✝⁴ : MeasurableInv G
      μ' ν' : MeasureTheory.Measure G
      inst✝³ : MeasureTheory.SigmaFinite μ'
      inst✝² : MeasureTheory.SigmaFinite ν'
      inst✝¹ : μ'.IsMulLeftInvariant
      inst✝ : ν'.IsMulLeftInvariant
      t s : Set G
      h2s : Ne (ν' s) 0
      h3s : Ne (ν' s) Top.top
      hs : MeasurableSet s
      this : ∀ (t : Set G), MeasurableSet t → Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMu …
      ht : Not (MeasurableSet t)
      ⊢ Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMul (ν' s) (μ' t))
    -/
  · rcases exists_measurable_superset₂ μ' ν' t with ⟨t', -, hm, hμ, hν⟩
    /-
      case inr.intro.intro.intro.intro
      G : Type u_1
      inst✝⁷ : MeasurableSpace G
      inst✝⁶ : Group G
      inst✝⁵ : MeasurableMul₂ G
      inst✝⁴ : MeasurableInv G
      μ' ν' : MeasureTheory.Measure G
      inst✝³ : MeasureTheory.SigmaFinite μ'
      inst✝² : MeasureTheory.SigmaFinite ν'
      inst✝¹ : μ'.IsMulLeftInvariant
      inst✝ : ν'.IsMulLeftInvariant
      t s : Set G
      h2s : Ne (ν' s) 0
      h3s : Ne (ν' s) Top.top
      hs : MeasurableSet s
      this : ∀ (t : Set G), MeasurableSet t → Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMu …
      ht : Not (MeasurableSet t)
      t' : Set G
      hm : MeasurableSet t'
      hμ : Eq (μ' t') (μ' t)
      hν : Eq (ν' t') (ν' t)
      ⊢ Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMul (ν' s) (μ' t))
    -/
    rw [← hμ, ← hν, this _ hm]
    /-
      🎉 no goals
    -/
  have h1 := measure_lintegral_div_measure ν' ν' hs h2s h3s (t.indicator fun _ => 1)
    (measurable_const.indicator ht)
  have h2 := measure_lintegral_div_measure μ' ν' hs h2s h3s (t.indicator fun _ => 1)
    (measurable_const.indicator ht)
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    hs : MeasurableSet s
    t : Set G
    ht : MeasurableSet t
    h1 : Eq (HMul.hMul (ν' s) (MeasureTheory.lintegral ν' fun y => HDiv.hDiv (t.in …
    h2 : Eq (HMul.hMul (μ' s) (MeasureTheory.lintegral ν' fun y => HDiv.hDiv (t.in …
    ⊢ Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMul (ν' s) (μ' t))
  -/
  rw [lintegral_indicator ht, setLIntegral_one] at h1 h2
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    s : Set G
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    hs : MeasurableSet s
    t : Set G
    ht : MeasurableSet t
    h1 : Eq (HMul.hMul (ν' s) (MeasureTheory.lintegral ν' fun y => HDiv.hDiv (t.in …
    h2 : Eq (HMul.hMul (μ' s) (MeasureTheory.lintegral ν' fun y => HDiv.hDiv (t.in …
    ⊢ Eq (HMul.hMul (μ' s) (ν' t)) (HMul.hMul (ν' s) (μ' t))
  -/
  rw [← h1, mul_left_comm, h2]
  /-
    🎉 no goals
  -/


/-- Left invariant Borel measures on a measurable group are unique (up to a scalar). -/
@[to_additive
" Left invariant Borel measures on an additive measurable group are unique (up to a scalar). "]
theorem measure_eq_div_smul (h2s : ν' s ≠ 0) (h3s : ν' s ≠ ∞) :
    μ' = (μ' s / ν' s) • ν' := by
  /-
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    s : Set G
    inst✝⁴ : MeasurableInv G
    μ' ν' : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SigmaFinite μ'
    inst✝² : MeasureTheory.SigmaFinite ν'
    inst✝¹ : μ'.IsMulLeftInvariant
    inst✝ : ν'.IsMulLeftInvariant
    h2s : Ne (ν' s) 0
    h3s : Ne (ν' s) Top.top
    ⊢ Eq μ' (HSMul.hSMul (HDiv.hDiv (μ' s) (ν' s)) ν')
  -/
  ext1 t -
  rw [smul_apply, smul_eq_mul, mul_comm, ← mul_div_assoc, mul_comm,
    measure_mul_measure_eq μ' ν' s t h2s h3s, mul_div_assoc, ENNReal.mul_div_cancel h2s h3s]


@[to_additive measurePreserving_prod_add_right]
theorem measurePreserving_prod_mul_right [IsMulRightInvariant ν] :
    MeasurePreserving (fun z : G × G => (z.1, z.2 * z.1)) (μ.prod ν) (μ.prod ν) :=
  MeasurePreserving.skew_product (g := fun x y => y * x) (MeasurePreserving.id μ)
    (measurable_snd.mul measurable_fst) <| Filter.Eventually.of_forall <| map_mul_right_eq_self ν


/-- The map `(x, y) ↦ (y, xy)` sends the measure `μ × ν` to `ν × μ`. -/
@[to_additive measurePreserving_prod_add_swap_right
" The map `(x, y) ↦ (y, x + y)` sends the measure `μ × ν` to `ν × μ`. "]
theorem measurePreserving_prod_mul_swap_right [IsMulRightInvariant μ] :
    MeasurePreserving (fun z : G × G => (z.2, z.1 * z.2)) (μ.prod ν) (ν.prod μ) :=
  (measurePreserving_prod_mul_right ν μ).comp measurePreserving_swap


/-- The map `(x, y) ↦ (xy, y)` preserves the measure `μ × ν`. -/
@[to_additive measurePreserving_add_prod
" The map `(x, y) ↦ (x + y, y)` preserves the measure `μ × ν`. "]
theorem measurePreserving_mul_prod [IsMulRightInvariant μ] :
    MeasurePreserving (fun z : G × G => (z.1 * z.2, z.2)) (μ.prod ν) (μ.prod ν) :=
  measurePreserving_swap.comp (measurePreserving_prod_mul_swap_right μ ν)


/-- The map `(x, y) ↦ (x, y / x)` is measure-preserving. -/
@[to_additive measurePreserving_prod_sub "The map `(x, y) ↦ (x, y - x)` is measure-preserving."]
theorem measurePreserving_prod_div [IsMulRightInvariant ν] :
    MeasurePreserving (fun z : G × G => (z.1, z.2 / z.1)) (μ.prod ν) (μ.prod ν) :=
  (measurePreserving_prod_mul_right μ ν).symm (MeasurableEquiv.shearDivRight G).symm


/-- The map `(x, y) ↦ (y, x / y)` sends `μ × ν` to `ν × μ`. -/
@[to_additive measurePreserving_prod_sub_swap
      "The map `(x, y) ↦ (y, x - y)` sends `μ × ν` to `ν × μ`."]
theorem measurePreserving_prod_div_swap [IsMulRightInvariant μ] :
    MeasurePreserving (fun z : G × G => (z.2, z.1 / z.2)) (μ.prod ν) (ν.prod μ) :=
  (measurePreserving_prod_div ν μ).comp measurePreserving_swap


/-- The map `(x, y) ↦ (x / y, y)` preserves the measure `μ × ν`. -/
@[to_additive measurePreserving_sub_prod
" The map `(x, y) ↦ (x - y, y)` preserves the measure `μ × ν`. "]
theorem measurePreserving_div_prod [IsMulRightInvariant μ] :
    MeasurePreserving (fun z : G × G => (z.1 / z.2, z.2)) (μ.prod ν) (μ.prod ν) :=
  measurePreserving_swap.comp (measurePreserving_prod_div_swap μ ν)


/-- The map `(x, y) ↦ (xy, x⁻¹)` is measure-preserving. -/
@[to_additive measurePreserving_add_prod_neg_right
"The map `(x, y) ↦ (x + y, - x)` is measure-preserving."]
theorem measurePreserving_mul_prod_inv_right [IsMulRightInvariant μ] [IsMulRightInvariant ν] :
    MeasurePreserving (fun z : G × G => (z.1 * z.2, z.1⁻¹)) (μ.prod ν) (μ.prod ν) := by
  convert (measurePreserving_prod_div_swap ν μ).comp (measurePreserving_prod_mul_swap_right μ ν)
    using 1
  /-
    case h.e'_5
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulRightInvariant
    inst✝ : ν.IsMulRightInvariant
    ⊢ Eq (fun z => { fst := HMul.hMul z.1 z.2, snd := Inv.inv z.1 }) (Function.com …
  -/
  ext1 ⟨x, y⟩
  /-
    case h.e'_5.h.mk
    G : Type u_1
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : Group G
    inst✝⁵ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝⁴ : MeasureTheory.SFinite ν
    inst✝³ : MeasureTheory.SFinite μ
    inst✝² : MeasurableInv G
    inst✝¹ : μ.IsMulRightInvariant
    inst✝ : ν.IsMulRightInvariant
    x y : G
    ⊢ Eq { fst := HMul.hMul { fst := x, snd := y }.1 { fst := x, snd := y }.2, snd …
  -/
  simp_rw [Function.comp_apply, div_mul_eq_div_div_swap, div_self', one_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem quasiMeasurePreserving_inv_of_right_invariant [IsMulRightInvariant μ] :
    QuasiMeasurePreserving (Inv.inv : G → G) μ μ := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulRightInvariant
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving Inv.inv μ μ
  -/
  rw [← μ.inv_inv]
  exact
    (quasiMeasurePreserving_inv μ.inv).mono (inv_absolutelyContinuous μ.inv)
      (absolutelyContinuous_inv μ.inv)


@[to_additive]
theorem quasiMeasurePreserving_div_left [IsMulLeftInvariant μ] (g : G) :
    QuasiMeasurePreserving (fun h : G => g / h) μ μ := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HDiv.hDiv g h) μ μ
  -/
  simp_rw [div_eq_mul_inv]
  exact
    (measurePreserving_mul_left μ g).quasiMeasurePreserving.comp (quasiMeasurePreserving_inv μ)


@[to_additive]
theorem quasiMeasurePreserving_div_left_of_right_invariant [IsMulRightInvariant μ] (g : G) :
    QuasiMeasurePreserving (fun h : G => g / h) μ μ := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulRightInvariant
    g : G
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HDiv.hDiv g h) μ μ
  -/
  rw [← μ.inv_inv]
  exact
    (quasiMeasurePreserving_div_left μ.inv g).mono (inv_absolutelyContinuous μ.inv)
      (absolutelyContinuous_inv μ.inv)


@[to_additive]
theorem quasiMeasurePreserving_div_of_right_invariant [IsMulRightInvariant μ] :
    QuasiMeasurePreserving (fun p : G × G => p.1 / p.2) (μ.prod ν) μ := by
  /-
    G : Type u_1
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : Group G
    inst✝⁴ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SFinite ν
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulRightInvariant
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun p => HDiv.hDiv p.1 p.2) (μ …
  -/
  refine QuasiMeasurePreserving.prod_of_left measurable_div (Eventually.of_forall fun y => ?_)
  /-
    G : Type u_1
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : Group G
    inst✝⁴ : MeasurableMul₂ G
    μ ν : MeasureTheory.Measure G
    inst✝³ : MeasureTheory.SFinite ν
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulRightInvariant
    y : G
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HDiv.hDiv { fst := x, …
  -/
  exact (measurePreserving_div_right μ y).quasiMeasurePreserving
  /-
    🎉 no goals
  -/


@[to_additive]
theorem quasiMeasurePreserving_div [IsMulLeftInvariant μ] :
    QuasiMeasurePreserving (fun p : G × G => p.1 / p.2) (μ.prod ν) μ :=
  (quasiMeasurePreserving_div_of_right_invariant μ.inv ν).mono
    ((absolutelyContinuous_inv μ).prod AbsolutelyContinuous.rfl) (inv_absolutelyContinuous μ)


/-- A *left*-invariant measure is quasi-preserved by *right*-multiplication.
This should not be confused with `(measurePreserving_mul_right μ g).quasiMeasurePreserving`. -/
@[to_additive
"A *left*-invariant measure is quasi-preserved by *right*-addition.
This should not be confused with `(measurePreserving_add_right μ g).quasiMeasurePreserving`. "]
theorem quasiMeasurePreserving_mul_right [IsMulLeftInvariant μ] (g : G) :
    QuasiMeasurePreserving (fun h : G => h * g) μ μ := by
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HMul.hMul h g) μ μ
  -/
  refine ⟨measurable_mul_const g, AbsolutelyContinuous.mk fun s hs => ?_⟩
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulLeftInvariant
    g : G
    s : Set G
    hs : MeasurableSet s
    ⊢ Eq (μ s) 0 → Eq ((MeasureTheory.Measure.map (fun h => HMul.hMul h g) μ) s) 0
  -/
  rw [map_apply (measurable_mul_const g) hs, measure_mul_right_null]; exact id
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- A *right*-invariant measure is quasi-preserved by *left*-multiplication.
This should not be confused with `(measurePreserving_mul_left μ g).quasiMeasurePreserving`. -/
@[to_additive
"A *right*-invariant measure is quasi-preserved by *left*-addition.
This should not be confused with `(measurePreserving_add_left μ g).quasiMeasurePreserving`. "]
theorem quasiMeasurePreserving_mul_left [IsMulRightInvariant μ] (g : G) :
    QuasiMeasurePreserving (fun h : G => g * h) μ μ := by
  have :=
    (quasiMeasurePreserving_mul_right μ.inv g⁻¹).mono (inv_absolutelyContinuous μ.inv)
      (absolutelyContinuous_inv μ.inv)
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulRightInvariant
    g : G
    this : MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HMul.hMul h (Inv …
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HMul.hMul g h) μ μ
  -/
  rw [μ.inv_inv] at this
  have :=
    (quasiMeasurePreserving_inv_of_right_invariant μ).comp
      (this.comp (quasiMeasurePreserving_inv_of_right_invariant μ))
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulRightInvariant
    g : G
    this✝ : MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HMul.hMul h (In …
    this : MeasureTheory.Measure.QuasiMeasurePreserving (Function.comp Inv.inv (Fu …
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HMul.hMul g h) μ μ
  -/
  simp_rw [Function.comp_def, mul_inv_rev, inv_inv] at this
  /-
    G : Type u_1
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul₂ G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.SFinite μ
    inst✝¹ : MeasurableInv G
    inst✝ : μ.IsMulRightInvariant
    g : G
    this✝ : MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HMul.hMul h (In …
    this : MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HMul.hMul g x) μ μ
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun h => HMul.hMul g h) μ μ
  -/
  exact this
  /-
    🎉 no goals
  -/


