local notation "dim" => Module.finrank ℝ


/-- If `μ` is an additive Haar measure on a normed space `E`,
then `μ.toSphere` is the measure on the unit sphere in `E`
such that `μ.toSphere s = Module.finrank ℝ E • μ (Set.Ioo (0 : ℝ) 1 • s)`. -/
def toSphere (μ : Measure E) : Measure (sphere (0 : E) 1) :=
  dim E • ((μ.comap (Subtype.val ∘ (homeomorphUnitSphereProd E).symm)).restrict
    (univ ×ˢ Iio ⟨1, mem_Ioi.2 one_pos⟩)).fst


theorem toSphere_apply_aux (s : Set (sphere (0 : E) 1)) (r : Ioi (0 : ℝ)) :
    μ ((↑) '' (homeomorphUnitSphereProd E ⁻¹' s ×ˢ Iio r)) = μ (Ioo (0 : ℝ) r • ((↑) '' s)) := by
  rw [← image2_smul, image2_image_right, ← Homeomorph.image_symm, image_image,
    ← image_subtype_val_Ioi_Iio, image2_image_left, image2_swap, ← image_prod]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    s : Set ↑(Metric.sphere 0 1)
    r : ↑(Set.Ioi 0)
    ⊢ Eq (μ (Set.image (fun x => ↑((homeomorphUnitSphereProd E).symm x)) (SProd.sp …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toSphere_apply' {s : Set (sphere (0 : E) 1)} (hs : MeasurableSet s) :
    μ.toSphere s = dim E * μ (Ioo (0 : ℝ) 1 • ((↑) '' s)) := by
  rw [toSphere, smul_apply, fst_apply hs, restrict_apply (measurable_fst hs),
    ((MeasurableEmbedding.subtype_coe (measurableSet_singleton _).compl).comp
      (Homeomorph.measurableEmbedding _)).comap_apply,
    image_comp, Homeomorph.image_symm, univ_prod, ← Set.prod_eq, nsmul_eq_mul, toSphere_apply_aux]


theorem toSphere_apply_univ' : μ.toSphere univ = dim E * μ (ball 0 1 \ {0}) := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝ : BorelSpace E
    ⊢ Eq (μ.toSphere Set.univ) (HMul.hMul (↑(Module.finrank Real E)) (μ (SDiff.sdi …
  -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
  rw [μ.toSphere_apply' .univ, image_univ, Subtype.range_coe, Ioo_smul_sphere_zero] <;> simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[simp]
theorem toSphere_apply_univ : μ.toSphere univ = dim E * μ (ball 0 1) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Eq (μ.toSphere Set.univ) (HMul.hMul (↑(Module.finrank Real E)) (μ (Metric.ba …
  -/
  nontriviality E
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    a✝ : Nontrivial E
    ⊢ Eq (μ.toSphere Set.univ) (HMul.hMul (↑(Module.finrank Real E)) (μ (Metric.ba …
  -/
  rw [toSphere_apply_univ', measure_diff_null (measure_singleton _)]
  /-
    🎉 no goals
  -/


instance : IsFiniteMeasure μ.toSphere where
  measure_univ_lt_top := by
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      ⊢ LT.lt (μ.toSphere Set.univ) Top.top
    -/
    rw [toSphere_apply_univ']
    exact ENNReal.mul_lt_top (ENNReal.natCast_lt_top _) <|
      measure_ball_lt_top.trans_le' <| measure_mono diff_subset


/-- The measure on `(0, +∞)` that has density `(· ^ n)` with respect to the Lebesgue measure. -/
def volumeIoiPow (n : ℕ) : Measure (Ioi (0 : ℝ)) :=
  .withDensity (.comap Subtype.val volume) fun r ↦ .ofReal (r.1 ^ n)


lemma volumeIoiPow_apply_Iio (n : ℕ) (x : Ioi (0 : ℝ)) :
    volumeIoiPow n (Iio x) = ENNReal.ofReal (x.1 ^ (n + 1) / (n + 1)) := by
  /-
    n : Nat
    x : ↑(Set.Ioi 0)
    ⊢ Eq ((MeasureTheory.Measure.volumeIoiPow n) (Set.Iio x)) (ENNReal.ofReal (HDi …
  -/
  have hr₀ : 0 ≤ x.1 := le_of_lt x.2
  rw [volumeIoiPow, withDensity_apply _ measurableSet_Iio,
    setLIntegral_subtype measurableSet_Ioi _ fun a : ℝ ↦ .ofReal (a ^ n),
    image_subtype_val_Ioi_Iio, restrict_congr_set Ioo_ae_eq_Ioc,
    ← ofReal_integral_eq_lintegral_ofReal (intervalIntegrable_pow _).1, ← integral_of_le hr₀]
    /-
      n : Nat
      x : ↑(Set.Ioi 0)
      hr₀ : LE.le 0 ↑x
      ⊢ Eq (ENNReal.ofReal (intervalIntegral (fun x => HPow.hPow x n) 0 (↑x) Measure …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      x : ↑(Set.Ioi 0)
      hr₀ : LE.le 0 ↑x
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc 0 ↑x) …
    -/
  · filter_upwards [ae_restrict_mem measurableSet_Ioc] with y hy
    /-
      case h
      n : Nat
      x : ↑(Set.Ioi 0)
      hr₀ : LE.le 0 ↑x
      y : Real
      hy : Membership.mem (Set.Ioc 0 ↑x) y
      ⊢ LE.le (0 y) (HPow.hPow y n)
    -/
    exact pow_nonneg hy.1.le _
    /-
      🎉 no goals
    -/


/-- The intervals `(0, k + 1)` have finite measure `MeasureTheory.Measure.volumeIoiPow _`
and cover the whole open ray `(0, +∞)`. -/
def finiteSpanningSetsIn_volumeIoiPow_range_Iio (n : ℕ) :
    FiniteSpanningSetsIn (volumeIoiPow n) (range Iio) where
  set k := Iio ⟨k + 1, mem_Ioi.2 k.cast_add_one_pos⟩
  set_mem _ := mem_range_self _
                 /-
                   E : Type u_1
                   inst✝⁵ : NormedAddCommGroup E
                   inst✝⁴ : NormedSpace Real E
                   inst✝³ : MeasurableSpace E
                   μ : MeasureTheory.Measure E
                   inst✝² : BorelSpace E
                   inst✝¹ : FiniteDimensional Real E
                   inst✝ : μ.IsAddHaarMeasure
                   n k : Nat
                   ⊢ LT.lt ((MeasureTheory.Measure.volumeIoiPow n) ((fun k => Set.Iio ⟨HAdd.hAdd  …
                 -/
  finite k := by simp [volumeIoiPow_apply_Iio]
                 /-
                   🎉 no goals
                 -/
  spanning := iUnion_eq_univ_iff.2 fun x ↦ ⟨⌊x.1⌋₊, Nat.lt_floor_add_one x.1⟩


instance (n : ℕ) : SigmaFinite (volumeIoiPow n) :=
  (finiteSpanningSetsIn_volumeIoiPow_range_Iio n).sigmaFinite


/-- The homeomorphism `homeomorphUnitSphereProd E` sends an additive Haar measure `μ`
to the product of `μ.toSphere` and `MeasureTheory.Measure.volumeIoiPow (dim E - 1)`,
where `dim E = Module.finrank ℝ E` is the dimension of `E`. -/
theorem measurePreserving_homeomorphUnitSphereProd :
    MeasurePreserving (homeomorphUnitSphereProd E) (μ.comap (↑))
      (μ.toSphere.prod (volumeIoiPow (dim E - 1))) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    ⊢ MeasureTheory.MeasurePreserving (⇑(homeomorphUnitSphereProd E)) (MeasureTheo …
  -/
  nontriviality E
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    a✝ : Nontrivial E
    ⊢ MeasureTheory.MeasurePreserving (⇑(homeomorphUnitSphereProd E)) (MeasureTheo …
  -/
  refine ⟨(homeomorphUnitSphereProd E).measurable, .symm ?_⟩
  refine prod_eq_generateFrom generateFrom_measurableSet
    ((borel_eq_generateFrom_Iio _).symm.trans BorelSpace.measurable_eq.symm)
    isPiSystem_measurableSet isPiSystem_Iio
    μ.toSphere.toFiniteSpanningSetsIn (finiteSpanningSetsIn_volumeIoiPow_range_Iio _)
    fun s hs ↦ forall_mem_range.2 fun r ↦ ?_
  have : Ioo (0 : ℝ) r = r.1 • Ioo (0 : ℝ) 1 := by
    rw [LinearOrderedField.smul_Ioo r.2.out, smul_zero, smul_eq_mul, mul_one]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    a✝ : Nontrivial E
    s : Set ↑(Metric.sphere 0 1)
    hs : Membership.mem (setOf fun s => MeasurableSet s) s
    r : ↑(Set.Ioi 0)
    this : Eq (Set.Ioo 0 ↑r) (HSMul.hSMul (↑r) (Set.Ioo 0 1))
    ⊢ Eq ((MeasureTheory.Measure.map (⇑(homeomorphUnitSphereProd E)) (MeasureTheor …
  -/
  have hpos : 0 < dim E := Module.finrank_pos
  rw [(Homeomorph.measurableEmbedding _).map_apply, toSphere_apply' _ hs, volumeIoiPow_apply_Iio,
    comap_subtype_coe_apply (measurableSet_singleton _).compl, toSphere_apply_aux, this,
    smul_assoc, μ.addHaar_smul_of_nonneg r.2.out.le, Nat.sub_add_cancel hpos, Nat.cast_pred hpos,
    sub_add_cancel, mul_right_comm, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul, mul_div_cancel₀]
  /-
    case hb
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    a✝ : Nontrivial E
    s : Set ↑(Metric.sphere 0 1)
    hs : Membership.mem (setOf fun s => MeasurableSet s) s
    r : ↑(Set.Ioi 0)
    this : Eq (Set.Ioo 0 ↑r) (HSMul.hSMul (↑r) (Set.Ioo 0 1))
    hpos : LT.lt 0 (Module.finrank Real E)
    ⊢ Ne (↑(Module.finrank Real E)) 0
  -/
  exacts [(Nat.cast_pos.2 hpos).ne', Nat.cast_nonneg _]
  /-
    🎉 no goals
  -/


lemma integral_fun_norm_addHaar (f : ℝ → F) :
    ∫ x, f (‖x‖) ∂μ = dim E • (μ (ball 0 1)).toReal • ∫ y in Ioi (0 : ℝ), y ^ (dim E - 1) • f y :=
  calc
    ∫ x, f (‖x‖) ∂μ = ∫ x : ({(0)}ᶜ : Set E), f (‖x.1‖) ∂(μ.comap (↑)) := by
      rw [integral_subtype_comap (measurableSet_singleton _).compl fun x ↦ f (‖x‖),
        restrict_compl_singleton]
    _ = ∫ x : sphere (0 : E) 1 × Ioi (0 : ℝ), f x.2 ∂μ.toSphere.prod (.volumeIoiPow (dim E - 1)) :=
      μ.measurePreserving_homeomorphUnitSphereProd.integral_comp (Homeomorph.measurableEmbedding _)
        (f ∘ Subtype.val ∘ Prod.snd)
    _ = (μ.toSphere univ).toReal • ∫ x : Ioi (0 : ℝ), f x ∂.volumeIoiPow (dim E - 1) :=
      integral_fun_snd (f ∘ Subtype.val)
    _ = _ := by
      /-
        E : Type u_1
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        inst✝⁶ : MeasurableSpace E
        F : Type u_2
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace Real F
        inst✝³ : Nontrivial E
        μ : MeasureTheory.Measure E
        inst✝² : FiniteDimensional Real E
        inst✝¹ : BorelSpace E
        inst✝ : μ.IsAddHaarMeasure
        f : Real → F
        ⊢ Eq (HSMul.hSMul (μ.toSphere Set.univ).toReal (MeasureTheory.integral (Measur …
      -/
      simp only [Measure.volumeIoiPow, ENNReal.ofReal]
      rw [integral_withDensity_eq_integral_smul, μ.toSphere_apply_univ,
        ENNReal.toReal_mul, ENNReal.toReal_nat, ← nsmul_eq_mul, smul_assoc,
        integral_subtype_comap measurableSet_Ioi fun a ↦ Real.toNNReal (a ^ (dim E - 1)) • f a,
        setIntegral_congr_fun measurableSet_Ioi fun x hx ↦ ?_]
        /-
          E : Type u_1
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          inst✝⁶ : MeasurableSpace E
          F : Type u_2
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace Real F
          inst✝³ : Nontrivial E
          μ : MeasureTheory.Measure E
          inst✝² : FiniteDimensional Real E
          inst✝¹ : BorelSpace E
          inst✝ : μ.IsAddHaarMeasure
          f : Real → F
          x : Real
          hx : Membership.mem (Set.Ioi 0) x
          ⊢ Eq (HSMul.hSMul (HPow.hPow x (HSub.hSub (Module.finrank Real E) 1)).toNNReal …
        -/
      · rw [NNReal.smul_def, Real.coe_toNNReal _ (pow_nonneg hx.out.le _)]
        /-
          🎉 no goals
        -/
        /-
          case f_meas
          E : Type u_1
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          inst✝⁶ : MeasurableSpace E
          F : Type u_2
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace Real F
          inst✝³ : Nontrivial E
          μ : MeasureTheory.Measure E
          inst✝² : FiniteDimensional Real E
          inst✝¹ : BorelSpace E
          inst✝ : μ.IsAddHaarMeasure
          f : Real → F
          ⊢ Measurable fun r => (HPow.hPow (↑r) (HSub.hSub (Module.finrank Real E) 1)).t …
        -/
      · exact (measurable_subtype_coe.pow_const _).real_toNNReal
        /-
          🎉 no goals
        -/


