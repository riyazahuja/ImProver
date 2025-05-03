theorem volume_regionBetween_eq_integral' [SigmaFinite μ] (f_int : IntegrableOn f s μ)
    (g_int : IntegrableOn g s μ) (hs : MeasurableSet s) (hfg : f ≤ᵐ[μ.restrict s] g) :
    μ.prod volume (regionBetween f g s) = ENNReal.ofReal (∫ y in s, (g - f) y ∂μ) := by
  have h : g - f =ᵐ[μ.restrict s] fun x => Real.toNNReal (g x - f x) :=
    hfg.mono fun x hx => (Real.coe_toNNReal _ <| sub_nonneg.2 hx).symm
  rw [volume_regionBetween_eq_lintegral f_int.aemeasurable g_int.aemeasurable hs,
    integral_congr_ae h, lintegral_congr_ae,
    lintegral_coe_eq_integral _ ((integrable_congr h).mp (g_int.sub f_int))]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    s : Set α
    inst✝ : MeasureTheory.SigmaFinite μ
    f_int : MeasureTheory.IntegrableOn f s μ
    g_int : MeasureTheory.IntegrableOn g s μ
    hs : MeasurableSet s
    hfg : (MeasureTheory.ae (μ.restrict s)).EventuallyLE f g
    h : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (HSub.hSub g f) fun x => ↑( …
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun y => ENNReal.ofReal (HSu …
  -/
  dsimp only
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    s : Set α
    inst✝ : MeasureTheory.SigmaFinite μ
    f_int : MeasureTheory.IntegrableOn f s μ
    g_int : MeasureTheory.IntegrableOn g s μ
    hs : MeasurableSet s
    hfg : (MeasureTheory.ae (μ.restrict s)).EventuallyLE f g
    h : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (HSub.hSub g f) fun x => ↑( …
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun y => ENNReal.ofReal (HSu …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If two functions are integrable on a measurable set, and one function is less than
    or equal to the other on that set, then the volume of the region
    between the two functions can be represented as an integral. -/
theorem volume_regionBetween_eq_integral [SigmaFinite μ] (f_int : IntegrableOn f s μ)
    (g_int : IntegrableOn g s μ) (hs : MeasurableSet s) (hfg : ∀ x ∈ s, f x ≤ g x) :
    μ.prod volume (regionBetween f g s) = ENNReal.ofReal (∫ y in s, (g - f) y ∂μ) :=
  volume_regionBetween_eq_integral' f_int g_int hs
    ((ae_restrict_iff' hs).mpr (Eventually.of_forall hfg))


/-- If the sequence with `n`-th term the sup norm of `fun x ↦ f (x + n)` on the interval `Icc 0 1`,
for `n ∈ ℤ`, is summable, then `f` is integrable on `ℝ`. -/
theorem Real.integrable_of_summable_norm_Icc {E : Type*} [NormedAddCommGroup E] {f : C(ℝ, E)}
    (hf : Summable fun n : ℤ => ‖(f.comp <| ContinuousMap.addRight n).restrict (Icc 0 1)‖) :
    /-
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      hf : Summable fun n => Norm.norm (ContinuousMap.restrict (Set.Icc 0 1) (f.comp …
      ⊢ MeasureTheory.Measure Real
    -/
    Integrable f := by
    /-
      🎉 no goals
    -/
  refine integrable_of_summable_norm_restrict (.of_nonneg_of_le
    (fun n : ℤ => mul_nonneg (norm_nonneg
      (f.restrict (⟨Icc (n : ℝ) ((n : ℝ) + 1), isCompact_Icc⟩ : Compacts ℝ)))
        ENNReal.toReal_nonneg) (fun n => ?_) hf) ?_
  · simp only [Compacts.coe_mk, Real.volume_Icc, add_sub_cancel_left,
      ENNReal.toReal_ofReal zero_le_one, mul_one, norm_le _ (norm_nonneg _)]
    /-
      case refine_1
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      hf : Summable fun n => Norm.norm (ContinuousMap.restrict (Set.Icc 0 1) (f.comp …
      n : Int
      ⊢ ∀ (x : ↑(Set.Icc (↑n) (HAdd.hAdd (↑n) 1))), LE.le (Norm.norm ((ContinuousMap …
    -/
    intro x
    have := ((f.comp <| ContinuousMap.addRight n).restrict (Icc 0 1)).norm_coe_le_norm
        ⟨x - n, ⟨sub_nonneg.mpr x.2.1, sub_le_iff_le_add'.mpr x.2.2⟩⟩
    simpa only [ContinuousMap.restrict_apply, comp_apply, coe_addRight, Subtype.coe_mk,
      sub_add_cancel] using this
    /-
      case refine_2
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : ContinuousMap Real E
      hf : Summable fun n => Norm.norm (ContinuousMap.restrict (Set.Icc 0 1) (f.comp …
      ⊢ Eq (Set.iUnion fun i => ↑{ carrier := Set.Icc (↑i) (HAdd.hAdd (↑i) 1), isCom …
    -/
  · exact iUnion_Icc_intCast ℝ
    /-
      🎉 no goals
    -/


theorem integral_comp_neg_Iic {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (c : ℝ) (f : ℝ → E) : (∫ x in Iic c, f (-x)) = ∫ x in Ioi (-c), f x := by
  have A : MeasurableEmbedding fun x : ℝ => -x :=
    (Homeomorph.neg ℝ).isClosedEmbedding.measurableEmbedding
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    c : Real
    f : Real → E
    A : MeasurableEmbedding fun x => Neg.neg x
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have := MeasurableEmbedding.setIntegral_map (μ := volume) A f (Ici (-c))
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    c : Real
    f : Real → E
    A : MeasurableEmbedding fun x => Neg.neg x
    this : Eq (MeasureTheory.integral ((MeasureTheory.Measure.map (fun x => Neg.ne …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [Measure.map_neg_eq_self (volume : Measure ℝ)] at this
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    c : Real
    f : Real → E
    A : MeasurableEmbedding fun x => Neg.neg x
    this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simp_rw [← integral_Ici_eq_integral_Ioi, this, neg_preimage, neg_Ici, neg_neg]
  /-
    🎉 no goals
  -/

/- @[simp] Porting note: Linter complains it does not apply to itself. Although it does apply to
itself, it does not apply when `f` is more complicated -/

theorem integral_comp_neg_Ioi {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (c : ℝ) (f : ℝ → E) : (∫ x in Ioi c, f (-x)) = ∫ x in Iic (-c), f x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    c : Real
    f : Real → E
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [← neg_neg c, ← integral_comp_neg_Iic]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    c : Real
    f : Real → E
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simp only [neg_neg]
  /-
    🎉 no goals
  -/


theorem integral_comp_abs {f : ℝ → ℝ} :
    ∫ x, f |x| = 2 * ∫ x in Ioi (0 : ℝ), f x := by
  have eq : ∫ (x : ℝ) in Ioi 0, f |x| = ∫ (x : ℝ) in Ioi 0, f x := by
    refine setIntegral_congr_fun measurableSet_Ioi (fun _ hx => ?_)
    rw [abs_eq_self.mpr (le_of_lt (by exact hx))]
  /-
    f : Real → Real
    eq : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (S …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => f (abs …
  -/
  by_cases hf : IntegrableOn (fun x => f |x|) (Ioi 0)
  · have int_Iic : IntegrableOn (fun x ↦ f |x|) (Iic 0) := by
      rw [← Measure.map_neg_eq_self (volume : Measure ℝ)]
      let m : MeasurableEmbedding fun x : ℝ => -x := (Homeomorph.neg ℝ).measurableEmbedding
      rw [m.integrableOn_map_iff]
      simp_rw [Function.comp_def, abs_neg, neg_preimage, neg_Iic, neg_zero]
      exact integrableOn_Ici_iff_integrableOn_Ioi.mpr hf
    calc
      _ = (∫ x in Iic 0, f |x|) + ∫ x in Ioi 0, f |x| := by
        rw [← setIntegral_union (Iic_disjoint_Ioi le_rfl) measurableSet_Ioi int_Iic hf,
          Iic_union_Ioi, restrict_univ]
      _ = 2 * ∫ x in Ioi 0, f x := by
        rw [two_mul, eq]
        congr! 1
        rw [← neg_zero, ← integral_comp_neg_Iic, neg_zero]
        refine setIntegral_congr_fun measurableSet_Iic (fun _ hx => ?_)
        rw [abs_eq_neg_self.mpr (by exact hx)]
  · have : ¬ Integrable (fun x => f |x|) := by
      contrapose! hf
      exact hf.integrableOn
    /-
      case neg
      f : Real → Real
      eq : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (S …
      hf : Not (MeasureTheory.IntegrableOn (fun x => f (abs x)) (Set.Ioi 0) MeasureT …
      this : Not (MeasureTheory.Integrable (fun x => f (abs x)) MeasureTheory.Measur …
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => f (abs …
    -/
    rw [← eq, integral_undef hf, integral_undef this, mul_zero]
    /-
      🎉 no goals
    -/

