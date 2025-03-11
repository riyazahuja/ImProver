theorem measure_Icc_lt_top (μ : Measure (ι → ℝ)) [IsLocallyFiniteMeasure μ] : μ (Box.Icc I) < ∞ :=
  show μ (Icc I.lower I.upper) < ∞ from I.isCompact_Icc.measure_lt_top


theorem measure_coe_lt_top (μ : Measure (ι → ℝ)) [IsLocallyFiniteMeasure μ] : μ I < ∞ :=
  (measure_mono <| coe_subset_Icc).trans_lt (I.measure_Icc_lt_top μ)


theorem measurableSet_coe : MeasurableSet (I : Set (ι → ℝ)) := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Countable ι
    ⊢ MeasurableSet ↑I
  -/
  rw [coe_eq_pi]
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Countable ι
    ⊢ MeasurableSet (Set.univ.pi fun i => Set.Ioc (I.lower i) (I.upper i))
  -/
  exact MeasurableSet.univ_pi fun i => measurableSet_Ioc
  /-
    🎉 no goals
  -/


theorem measurableSet_Icc : MeasurableSet (Box.Icc I) :=
  _root_.measurableSet_Icc


theorem measurableSet_Ioo : MeasurableSet (Box.Ioo I) :=
  MeasurableSet.univ_pi fun _ => _root_.measurableSet_Ioo


theorem coe_ae_eq_Icc : (I : Set (ι → ℝ)) =ᵐ[volume] Box.Icc I := by
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (↑I) (BoxI …
  -/
  rw [coe_eq_pi]
  /-
    ι : Type u_1
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Set.univ. …
  -/
  exact Measure.univ_pi_Ioc_ae_eq_Icc
  /-
    🎉 no goals
  -/


theorem Ioo_ae_eq_Icc : Box.Ioo I =ᵐ[volume] Box.Icc I :=
  Measure.univ_pi_Ioo_ae_eq_Icc


theorem Prepartition.measure_iUnion_toReal [Finite ι] {I : Box ι} (π : Prepartition I)
    (μ : Measure (ι → ℝ)) [IsLocallyFiniteMeasure μ] :
    (μ π.iUnion).toReal = ∑ J ∈ π.boxes, (μ J).toReal := by
  /-
    ι : Type u_1
    inst✝¹ : Finite ι
    I : BoxIntegral.Box ι
    π : BoxIntegral.Prepartition I
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ Eq (μ π.iUnion).toReal (π.boxes.sum fun J => (μ ↑J).toReal)
  -/
  erw [← ENNReal.toReal_sum, π.iUnion_def, measure_biUnion_finset π.pairwiseDisjoint]
  /-
    ι : Type u_1
    inst✝¹ : Finite ι
    I : BoxIntegral.Box ι
    π : BoxIntegral.Prepartition I
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ ∀ (b : BoxIntegral.Box ι), Membership.mem π.boxes b → MeasurableSet ↑b
  -/
  exacts [fun J _ => J.measurableSet_coe, fun J _ => (J.measure_coe_lt_top μ).ne]
  /-
    🎉 no goals
  -/


/-- If `μ` is a locally finite measure on `ℝⁿ`, then `fun J ↦ (μ J).toReal` is a box-additive
function. -/
@[simps]
def toBoxAdditive [Finite ι] (μ : Measure (ι → ℝ)) [IsLocallyFiniteMeasure μ] : ι →ᵇᵃ[⊤] ℝ where
  toFun J := (μ J).toReal
                                      /-
                                        ι : Type u_1
                                        inst✝¹ : Finite ι
                                        μ : MeasureTheory.Measure (ι → Real)
                                        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
                                        J : BoxIntegral.Box ι
                                        x✝ : LE.le (↑J) Top.top
                                        π : BoxIntegral.Prepartition J
                                        hπ : π.IsPartition
                                        ⊢ Eq (π.boxes.sum fun Ji => (fun J => (μ ↑J).toReal) Ji) ((fun J => (μ ↑J).toR …
                                      -/
  sum_partition_boxes' J _ π hπ := by rw [← π.measure_iUnion_toReal, hπ.iUnion_eq]
                                      /-
                                        🎉 no goals
                                      -/


theorem volume_apply (I : Box ι) :
    (volume : Measure (ι → ℝ)).toBoxAdditive I = ∏ i, (I.upper i - I.lower i) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    ⊢ Eq (MeasureTheory.MeasureSpace.volume.toBoxAdditive I) (Finset.univ.prod fun …
  -/
  rw [Measure.toBoxAdditive_apply, coe_eq_pi, Real.volume_pi_Ioc_toReal I.lower_le_upper]
  /-
    🎉 no goals
  -/


@[simp]
theorem volume_apply' (I : Box ι) :
    ((volume : Measure (ι → ℝ)) I).toReal = ∏ i, (I.upper i - I.lower i) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    ⊢ Eq (MeasureTheory.MeasureSpace.volume ↑I).toReal (Finset.univ.prod fun i =>  …
  -/
  rw [coe_eq_pi, Real.volume_pi_Ioc_toReal I.lower_le_upper]
  /-
    🎉 no goals
  -/


theorem volume_face_mul {n} (i : Fin (n + 1)) (I : Box (Fin (n + 1))) :
    (∏ j, ((I.face i).upper j - (I.face i).lower j)) * (I.upper i - I.lower i) =
      ∏ j, (I.upper j - I.lower j) := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    ⊢ Eq (HMul.hMul (Finset.univ.prod fun j => HSub.hSub ((I.face i).upper j) ((I. …
  -/
  simp only [face_lower, face_upper, (· ∘ ·), Fin.prod_univ_succAbove _ i, mul_comm]
  /-
    🎉 no goals
  -/


/-- Box-additive map sending each box `I` to the continuous linear endomorphism
`x ↦ (volume I).toReal • x`. -/
protected def volume {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] : ι →ᵇᵃ E →L[ℝ] E :=
  (volume : Measure (ι → ℝ)).toBoxAdditive.toSMul


theorem volume_apply {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] (I : Box ι) (x : E) :
    BoxAdditiveMap.volume I x = (∏ j, (I.upper j - I.lower j)) • x := by
  /-
    ι : Type u_1
    inst✝² : Fintype ι
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    I : BoxIntegral.Box ι
    x : E
    ⊢ Eq ((BoxIntegral.BoxAdditiveMap.volume I) x) (HSMul.hSMul (Finset.univ.prod  …
  -/
  rw [BoxAdditiveMap.volume, toSMul_apply]
  /-
    ι : Type u_1
    inst✝² : Fintype ι
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    I : BoxIntegral.Box ι
    x : E
    ⊢ Eq (HSMul.hSMul (MeasureTheory.MeasureSpace.volume.toBoxAdditive I) x) (HSMu …
  -/
  exact congr_arg₂ (· • ·) I.volume_apply rfl
  /-
    🎉 no goals
  -/


