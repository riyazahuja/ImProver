@[measurability]
protected theorem AddCircle.measurable_mk' {a : ℝ} :
    Measurable (β := AddCircle a) ((↑) : ℝ → AddCircle a) :=
  Continuous.measurable <| AddCircle.continuous_mk' a


theorem isAddFundamentalDomain_Ioc {T : ℝ} (hT : 0 < T) (t : ℝ)
    (μ : Measure ℝ := by volume_tac) :
    IsAddFundamentalDomain (AddSubgroup.zmultiples T) (Ioc t (t + T)) μ := by
  /-
    T : Real
    hT : LT.lt 0 T
    t : Real
    μ : autoParam (MeasureTheory.Measure Real) _auto✝
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem (AddSu …
  -/
  refine IsAddFundamentalDomain.mk' nullMeasurableSet_Ioc fun x => ?_
  have : Bijective (codRestrict (fun n : ℤ => n • T) (AddSubgroup.zmultiples T) _) :=
    (Equiv.ofInjective (fun n : ℤ => n • T) (zsmul_left_strictMono hT).injective).bijective
  /-
    T : Real
    hT : LT.lt 0 T
    t : Real
    μ : autoParam (MeasureTheory.Measure Real) _auto✝
    x : Real
    this : Function.Bijective (Set.codRestrict (fun n => HSMul.hSMul n T) ↑(AddSub …
    ⊢ ExistsUnique fun g => Membership.mem (Set.Ioc t (HAdd.hAdd t T)) (HVAdd.hVAd …
  -/
  refine this.existsUnique_iff.2 ?_
  /-
    T : Real
    hT : LT.lt 0 T
    t : Real
    μ : autoParam (MeasureTheory.Measure Real) _auto✝
    x : Real
    this : Function.Bijective (Set.codRestrict (fun n => HSMul.hSMul n T) ↑(AddSub …
    ⊢ ExistsUnique fun x_1 => Membership.mem (Set.Ioc t (HAdd.hAdd t T)) (HVAdd.hV …
  -/
  simpa only [add_comm x] using existsUnique_add_zsmul_mem_Ioc hT x t
  /-
    🎉 no goals
  -/


theorem isAddFundamentalDomain_Ioc' {T : ℝ} (hT : 0 < T) (t : ℝ) (μ : Measure ℝ := by volume_tac) :
    IsAddFundamentalDomain (AddSubgroup.op <| .zmultiples T) (Ioc t (t + T)) μ := by
  /-
    T : Real
    hT : LT.lt 0 T
    t : Real
    μ : autoParam (MeasureTheory.Measure Real) _auto✝
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem (AddSu …
  -/
  refine IsAddFundamentalDomain.mk' nullMeasurableSet_Ioc fun x => ?_
  have : Bijective (codRestrict (fun n : ℤ => n • T) (AddSubgroup.zmultiples T) _) :=
    (Equiv.ofInjective (fun n : ℤ => n • T) (zsmul_left_strictMono hT).injective).bijective
  /-
    T : Real
    hT : LT.lt 0 T
    t : Real
    μ : autoParam (MeasureTheory.Measure Real) _auto✝
    x : Real
    this : Function.Bijective (Set.codRestrict (fun n => HSMul.hSMul n T) ↑(AddSub …
    ⊢ ExistsUnique fun g => Membership.mem (Set.Ioc t (HAdd.hAdd t T)) (HVAdd.hVAd …
  -/
  refine (AddSubgroup.equivOp _).bijective.comp this |>.existsUnique_iff.2 ?_
  /-
    T : Real
    hT : LT.lt 0 T
    t : Real
    μ : autoParam (MeasureTheory.Measure Real) _auto✝
    x : Real
    this : Function.Bijective (Set.codRestrict (fun n => HSMul.hSMul n T) ↑(AddSub …
    ⊢ ExistsUnique fun x_1 => Membership.mem (Set.Ioc t (HAdd.hAdd t T)) (HVAdd.hV …
  -/
  simpa using existsUnique_add_zsmul_mem_Ioc hT x t
  /-
    🎉 no goals
  -/


/-- Equip the "additive circle" `ℝ ⧸ (ℤ ∙ T)` with, as a standard measure, the Haar measure of total
mass `T` -/
noncomputable instance measureSpace : MeasureSpace (AddCircle T) :=
  { QuotientAddGroup.measurableSpace _ with volume := ENNReal.ofReal T • addHaarMeasure ⊤ }


@[simp]
protected theorem measure_univ : volume (Set.univ : Set (AddCircle T)) = ENNReal.ofReal T := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ Eq (MeasureTheory.MeasureSpace.volume Set.univ) (ENNReal.ofReal T)
  -/
  dsimp [volume]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ Eq (HMul.hMul (ENNReal.ofReal T) ((MeasureTheory.Measure.addHaarMeasure Top. …
  -/
  rw [← PositiveCompacts.coe_top]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ Eq (HMul.hMul (ENNReal.ofReal T) ((MeasureTheory.Measure.addHaarMeasure Top. …
  -/
  simp [addHaarMeasure_self (G := AddCircle T), -PositiveCompacts.coe_top]
  /-
    🎉 no goals
  -/


instance : IsAddHaarMeasure (volume : Measure (AddCircle T)) :=
                              /-
                                T : Real
                                hT : Fact (LT.lt 0 T)
                                ⊢ Ne (ENNReal.ofReal T) 0
                              -/
  IsAddHaarMeasure.smul _ (by simp [hT.out]) ENNReal.ofReal_ne_top
                              /-
                                🎉 no goals
                              -/


instance isFiniteMeasure : IsFiniteMeasure (volume : Measure (AddCircle T)) where
                            /-
                              T : Real
                              hT : Fact (LT.lt 0 T)
                              ⊢ LT.lt (MeasureTheory.MeasureSpace.volume Set.univ) Top.top
                            -/
  measure_univ_lt_top := by simp
                            /-
                              🎉 no goals
                            -/


           /-
             T : Real
             hT : Fact (LT.lt 0 T)
             ⊢ MeasureTheory.Measure Real
           -/
instance : HasAddFundamentalDomain (AddSubgroup.op <| .zmultiples T) ℝ where
           /-
             🎉 no goals
           -/
                                                  /-
                                                    T : Real
                                                    hT : Fact (LT.lt 0 T)
                                                    ⊢ MeasureTheory.Measure Real
                                                  -/
  ExistsIsAddFundamentalDomain := ⟨Ioc 0 (0 + T), isAddFundamentalDomain_Ioc' Fact.out 0⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


instance : AddQuotientMeasureEqMeasurePreimage volume (volume : Measure (AddCircle T)) := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ MeasureTheory.AddQuotientMeasureEqMeasurePreimage MeasureTheory.MeasureSpace …
  -/
  apply MeasureTheory.leftInvariantIsAddQuotientMeasureEqMeasurePreimage
  /-
    case h
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ Eq (MeasureTheory.addCovolume (Subtype fun x => Membership.mem (AddSubgroup. …
  -/
  simp [(isAddFundamentalDomain_Ioc' hT.out 0).covolume_eq_volume, AddCircle.measure_univ]
  /-
    🎉 no goals
  -/


/-- The covering map from `ℝ` to the "additive circle" `ℝ ⧸ (ℤ ∙ T)` is measure-preserving,
considered with respect to the standard measure (defined to be the Haar measure of total mass `T`)
on the additive circle, and with respect to the restriction of Lebsegue measure on `ℝ` to an
interval (t, t + T]. -/
protected theorem measurePreserving_mk (t : ℝ) :
    /-
      T : Real
      hT : Fact (LT.lt 0 T)
      t : Real
      ⊢ MeasureTheory.Measure (AddCircle T)
    -/
    MeasurePreserving (β := AddCircle T) ((↑) : ℝ → AddCircle T)
    /-
      🎉 no goals
    -/
      (volume.restrict (Ioc t (t + T))) :=
  measurePreserving_quotientAddGroup_mk_of_AddQuotientMeasureEqMeasurePreimage
                               /-
                                 T : Real
                                 hT : Fact (LT.lt 0 T)
                                 t : Real
                                 ⊢ MeasureTheory.Measure Real
                               -/
    volume (𝓕 := Ioc t (t+T)) (isAddFundamentalDomain_Ioc' hT.out _) _
                               /-
                                 🎉 no goals
                               -/


lemma add_projection_respects_measure (t : ℝ) {U : Set (AddCircle T)} (meas_U : MeasurableSet U) :
    volume U = volume (QuotientAddGroup.mk ⁻¹' U ∩ (Ioc t (t + T))) :=
   /-
     T : Real
     hT : Fact (LT.lt 0 T)
     t : Real
     U : Set (AddCircle T)
     meas_U : MeasurableSet U
     ⊢ MeasureTheory.Measure Real
   -/
  (isAddFundamentalDomain_Ioc' hT.out _).addProjection_respects_measure_apply
   /-
     🎉 no goals
   -/
    (volume : Measure (AddCircle T)) meas_U


theorem volume_closedBall {x : AddCircle T} (ε : ℝ) :
    volume (Metric.closedBall x ε) = ENNReal.ofReal (min T (2 * ε)) := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x : AddCircle T
    ε : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x ε)) (ENNReal.ofRe …
  -/
  have hT' : |T| = T := abs_eq_self.mpr hT.out.le
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x : AddCircle T
    ε : Real
    hT' : Eq (abs T) T
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall x ε)) (ENNReal.ofRe …
  -/
  let I := Ioc (-(T / 2)) (T / 2)
  have h₁ : ε < T / 2 → Metric.closedBall (0 : ℝ) ε ∩ I = Metric.closedBall (0 : ℝ) ε := by
    intro hε
    rw [inter_eq_left, Real.closedBall_eq_Icc, zero_sub, zero_add]
    rintro y ⟨hy₁, hy₂⟩; constructor <;> linarith
  have h₂ : (↑) ⁻¹' Metric.closedBall (0 : AddCircle T) ε ∩ I =
      if ε < T / 2 then Metric.closedBall (0 : ℝ) ε else I := by
    conv_rhs => rw [← if_ctx_congr (Iff.rfl : ε < T / 2 ↔ ε < T / 2) h₁ fun _ => rfl, ← hT']
    apply coe_real_preimage_closedBall_inter_eq
    simpa only [hT', Real.closedBall_eq_Icc, zero_add, zero_sub] using Ioc_subset_Icc_self
  rw [addHaar_closedBall_center, add_projection_respects_measure T (-(T/2))
    measurableSet_closedBall, (by linarith : -(T / 2) + T = T / 2), h₂]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x : AddCircle T
    ε : Real
    hT' : Eq (abs T) T
    I : Set Real := Set.Ioc (Neg.neg (HDiv.hDiv T 2)) (HDiv.hDiv T 2)
    h₁ : LT.lt ε (HDiv.hDiv T 2) → Eq (Inter.inter (Metric.closedBall 0 ε) I) (Met …
    h₂ : Eq (Inter.inter (Set.preimage QuotientAddGroup.mk (Metric.closedBall 0 ε) …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (ite (LT.lt ε (HDiv.hDiv T 2)) (Metric …
  -/
  by_cases hε : ε < T / 2
    /-
      case pos
      T : Real
      hT : Fact (LT.lt 0 T)
      x : AddCircle T
      ε : Real
      hT' : Eq (abs T) T
      I : Set Real := Set.Ioc (Neg.neg (HDiv.hDiv T 2)) (HDiv.hDiv T 2)
      h₁ : LT.lt ε (HDiv.hDiv T 2) → Eq (Inter.inter (Metric.closedBall 0 ε) I) (Met …
      h₂ : Eq (Inter.inter (Set.preimage QuotientAddGroup.mk (Metric.closedBall 0 ε) …
      hε : LT.lt ε (HDiv.hDiv T 2)
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (ite (LT.lt ε (HDiv.hDiv T 2)) (Metric …
    -/
  · simp [hε, min_eq_right (by linarith : 2 * ε ≤ T)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      T : Real
      hT : Fact (LT.lt 0 T)
      x : AddCircle T
      ε : Real
      hT' : Eq (abs T) T
      I : Set Real := Set.Ioc (Neg.neg (HDiv.hDiv T 2)) (HDiv.hDiv T 2)
      h₁ : LT.lt ε (HDiv.hDiv T 2) → Eq (Inter.inter (Metric.closedBall 0 ε) I) (Met …
      h₂ : Eq (Inter.inter (Set.preimage QuotientAddGroup.mk (Metric.closedBall 0 ε) …
      hε : Not (LT.lt ε (HDiv.hDiv T 2))
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (ite (LT.lt ε (HDiv.hDiv T 2)) (Metric …
    -/
  · simp [I, hε, min_eq_left (by linarith : T ≤ 2 * ε)]
    /-
      🎉 no goals
    -/


instance : IsUnifLocDoublingMeasure (volume : Measure (AddCircle T)) := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ IsUnifLocDoublingMeasure MeasureTheory.MeasureSpace.volume
  -/
  refine ⟨⟨Real.toNNReal 2, Filter.Eventually.of_forall fun ε x => ?_⟩⟩
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ε : Real
    x : AddCircle T
    ⊢ LE.le (MeasureTheory.MeasureSpace.volume (Metric.closedBall x (HMul.hMul 2 ε …
  -/
  simp only [volume_closedBall]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ε : Real
    x : AddCircle T
    ⊢ LE.le (ENNReal.ofReal (Min.min T (HMul.hMul 2 (HMul.hMul 2 ε)))) (HMul.hMul  …
  -/
  erw [← ENNReal.ofReal_mul zero_le_two]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ε : Real
    x : AddCircle T
    ⊢ LE.le (ENNReal.ofReal (Min.min T (HMul.hMul 2 (HMul.hMul 2 ε)))) (ENNReal.of …
  -/
  apply ENNReal.ofReal_le_ofReal
  /-
    case h
    T : Real
    hT : Fact (LT.lt 0 T)
    ε : Real
    x : AddCircle T
    ⊢ LE.le (Min.min T (HMul.hMul 2 (HMul.hMul 2 ε))) (HMul.hMul 2 (Min.min T (HMu …
  -/
  rw [mul_min_of_nonneg _ _ (zero_le_two : (0 : ℝ) ≤ 2)]
  /-
    case h
    T : Real
    hT : Fact (LT.lt 0 T)
    ε : Real
    x : AddCircle T
    ⊢ LE.le (Min.min T (HMul.hMul 2 (HMul.hMul 2 ε))) (Min.min (HMul.hMul 2 T) (HM …
  -/
  exact min_le_min (by linarith [hT.out]) (le_refl _)
  /-
    🎉 no goals
  -/


/-- The isomorphism `AddCircle T ≃ Ioc a (a + T)` whose inverse is the natural quotient map,
  as an equivalence of measurable spaces. -/
noncomputable def measurableEquivIoc (a : ℝ) : AddCircle T ≃ᵐ Ioc a (a + T) where
  toEquiv := equivIoc T a
  measurable_toFun := measurable_of_measurable_on_compl_singleton _
    (continuousOn_iff_continuous_restrict.mp <| continuousOn_of_forall_continuousAt fun _x hx =>
      continuousAt_equivIoc T a hx).measurable
  measurable_invFun := AddCircle.measurable_mk'.comp measurable_subtype_coe


/-- The isomorphism `AddCircle T ≃ Ico a (a + T)` whose inverse is the natural quotient map,
  as an equivalence of measurable spaces. -/
noncomputable def measurableEquivIco (a : ℝ) : AddCircle T ≃ᵐ Ico a (a + T) where
  toEquiv := equivIco T a
  measurable_toFun := measurable_of_measurable_on_compl_singleton _
    (continuousOn_iff_continuous_restrict.mp <| continuousOn_of_forall_continuousAt fun _x hx =>
      continuousAt_equivIco T a hx).measurable
  measurable_invFun := AddCircle.measurable_mk'.comp measurable_subtype_coe


attribute [local instance] Subtype.measureSpace in
/-- The lower integral of a function over `AddCircle T` is equal to the lower integral over an
interval (t, t + T] in `ℝ` of its lift to `ℝ`. -/
protected theorem lintegral_preimage (t : ℝ) (f : AddCircle T → ℝ≥0∞) :
    (∫⁻ a in Ioc t (t + T), f a) = ∫⁻ b : AddCircle T, f b := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    t : Real
    f : AddCircle T → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  have m : MeasurableSet (Ioc t (t + T)) := measurableSet_Ioc
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    t : Real
    f : AddCircle T → ENNReal
    m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  have := lintegral_map_equiv (μ := volume) f (measurableEquivIoc T t).symm
  simp only [measurableEquivIoc, equivIoc, QuotientAddGroup.equivIocMod, MeasurableEquiv.symm_mk,
    MeasurableEquiv.coe_mk, Equiv.coe_fn_symm_mk] at this
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    t : Real
    f : AddCircle T → ENNReal
    m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
    this : Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (fun x => ↑↑x) M …
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  rw [← (AddCircle.measurePreserving_mk T t).map_eq]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    t : Real
    f : AddCircle T → ENNReal
    m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
    this : Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (fun x => ↑↑x) M …
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  convert this.symm using 1
    /-
      case h.e'_2
      T : Real
      hT : Fact (LT.lt 0 T)
      t : Real
      f : AddCircle T → ENNReal
      m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
      this : Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (fun x => ↑↑x) M …
      ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
    -/
  · rw [← map_comap_subtype_coe m _]
    /-
      case h.e'_2
      T : Real
      hT : Fact (LT.lt 0 T)
      t : Real
      f : AddCircle T → ENNReal
      m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
      this : Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (fun x => ↑↑x) M …
      ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map Subtype.val (MeasureT …
    -/
    exact MeasurableEmbedding.lintegral_map (MeasurableEmbedding.subtype_coe m) _
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      T : Real
      hT : Fact (LT.lt 0 T)
      t : Real
      f : AddCircle T → ENNReal
      m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
      this : Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (fun x => ↑↑x) M …
      ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map QuotientAddGroup.mk ( …
    -/
  · congr 1
    have : ((↑) : Ioc t (t + T) → AddCircle T) = ((↑) : ℝ → AddCircle T) ∘ ((↑) : _ → ℝ) := by
      ext1 x; rfl
    /-
      case h.e'_3.e_μ
      T : Real
      hT : Fact (LT.lt 0 T)
      t : Real
      f : AddCircle T → ENNReal
      m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
      this✝ : Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (fun x => ↑↑x)  …
      this : Eq (fun x => ↑↑x) (Function.comp QuotientAddGroup.mk Subtype.val)
      ⊢ Eq (MeasureTheory.Measure.map QuotientAddGroup.mk (MeasureTheory.MeasureSpac …
    -/
    simp_rw [this]
    /-
      case h.e'_3.e_μ
      T : Real
      hT : Fact (LT.lt 0 T)
      t : Real
      f : AddCircle T → ENNReal
      m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
      this✝ : Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (fun x => ↑↑x)  …
      this : Eq (fun x => ↑↑x) (Function.comp QuotientAddGroup.mk Subtype.val)
      ⊢ Eq (MeasureTheory.Measure.map QuotientAddGroup.mk (MeasureTheory.MeasureSpac …
    -/
    rw [← map_map AddCircle.measurable_mk' measurable_subtype_coe, ← map_comap_subtype_coe m]
    /-
      case h.e'_3.e_μ
      T : Real
      hT : Fact (LT.lt 0 T)
      t : Real
      f : AddCircle T → ENNReal
      m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
      this✝ : Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (fun x => ↑↑x)  …
      this : Eq (fun x => ↑↑x) (Function.comp QuotientAddGroup.mk Subtype.val)
      ⊢ Eq (MeasureTheory.Measure.map QuotientAddGroup.mk (MeasureTheory.Measure.map …
    -/
    rfl
    /-
      🎉 no goals
    -/


attribute [local instance] Subtype.measureSpace in
/-- The integral of an almost-everywhere strongly measurable function over `AddCircle T` is equal
to the integral over an interval (t, t + T] in `ℝ` of its lift to `ℝ`. -/
protected theorem integral_preimage (t : ℝ) (f : AddCircle T → E) :
    (∫ a in Ioc t (t + T), f a) = ∫ b : AddCircle T, f b := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    f : AddCircle T → E
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have m : MeasurableSet (Ioc t (t + T)) := measurableSet_Ioc
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    f : AddCircle T → E
    m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have := integral_map_equiv (μ := volume) (measurableEquivIoc T t).symm f
  simp only [measurableEquivIoc, equivIoc, QuotientAddGroup.equivIocMod, MeasurableEquiv.symm_mk,
    MeasurableEquiv.coe_mk, Equiv.coe_fn_symm_mk] at this
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    f : AddCircle T → E
    m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
    this : Eq (MeasureTheory.integral (MeasureTheory.Measure.map (fun x => ↑↑x) Me …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [← (AddCircle.measurePreserving_mk T t).map_eq, ← integral_subtype m, ← this]
  have : ((↑) : Ioc t (t + T) → AddCircle T) = ((↑) : ℝ → AddCircle T) ∘ ((↑) : _ → ℝ) := by
    ext1 x; rfl
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    f : AddCircle T → E
    m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
    this✝ : Eq (MeasureTheory.integral (MeasureTheory.Measure.map (fun x => ↑↑x) M …
    this : Eq (fun x => ↑↑x) (Function.comp QuotientAddGroup.mk Subtype.val)
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map (fun x => ↑↑x) Measure …
  -/
  simp_rw [this]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    f : AddCircle T → E
    m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
    this✝ : Eq (MeasureTheory.integral (MeasureTheory.Measure.map (fun x => ↑↑x) M …
    this : Eq (fun x => ↑↑x) (Function.comp QuotientAddGroup.mk Subtype.val)
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map (Function.comp Quotien …
  -/
  rw [← map_map AddCircle.measurable_mk' measurable_subtype_coe, ← map_comap_subtype_coe m]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    f : AddCircle T → E
    m : MeasurableSet (Set.Ioc t (HAdd.hAdd t T))
    this✝ : Eq (MeasureTheory.integral (MeasureTheory.Measure.map (fun x => ↑↑x) M …
    this : Eq (fun x => ↑↑x) (Function.comp QuotientAddGroup.mk Subtype.val)
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map QuotientAddGroup.mk (M …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The integral of an almost-everywhere strongly measurable function over `AddCircle T` is equal
to the integral over an interval (t, t + T] in `ℝ` of its lift to `ℝ`. -/
protected theorem intervalIntegral_preimage (t : ℝ) (f : AddCircle T → E) :
    ∫ a in t..t + T, f a = ∫ b : AddCircle T, f b := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    f : AddCircle T → E
    ⊢ Eq (intervalIntegral (fun a => f ↑a) t (HAdd.hAdd t T) MeasureTheory.Measure …
  -/
  rw [integral_of_le, AddCircle.integral_preimage T t f]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    f : AddCircle T → E
    ⊢ LE.le t (HAdd.hAdd t T)
  -/
  linarith [hT.out]
  /-
    🎉 no goals
  -/


                                                                                 /-
                                                                                   ⊢ Eq (MeasureTheory.MeasureSpace.volume Set.univ) 1
                                                                                 -/
protected theorem measure_univ : volume (Set.univ : Set UnitAddCircle) = 1 := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- The covering map from `ℝ` to the "unit additive circle" `ℝ ⧸ ℤ` is measure-preserving,
considered with respect to the standard measure (defined to be the Haar measure of total mass 1)
on the additive circle, and with respect to the restriction of Lebsegue measure on `ℝ` to an
interval (t, t + 1]. -/
protected theorem measurePreserving_mk (t : ℝ) :
    /-
      t : Real
      ⊢ MeasureTheory.Measure UnitAddCircle
    -/
    MeasurePreserving (β := UnitAddCircle) ((↑) : ℝ → UnitAddCircle)
    /-
      🎉 no goals
    -/
      (volume.restrict (Ioc t (t + 1))) :=
  AddCircle.measurePreserving_mk 1 t


/-- The integral of a measurable function over `UnitAddCircle` is equal to the integral over an
interval (t, t + 1] in `ℝ` of its lift to `ℝ`. -/
protected theorem lintegral_preimage (t : ℝ) (f : UnitAddCircle → ℝ≥0∞) :
    (∫⁻ a in Ioc t (t + 1), f a) = ∫⁻ b : UnitAddCircle, f b :=
  AddCircle.lintegral_preimage 1 t f


/-- The integral of an almost-everywhere strongly measurable function over `UnitAddCircle` is
equal to the integral over an interval (t, t + 1] in `ℝ` of its lift to `ℝ`. -/
protected theorem integral_preimage (t : ℝ) (f : UnitAddCircle → E) :
    (∫ a in Ioc t (t + 1), f a) = ∫ b : UnitAddCircle, f b :=
  AddCircle.integral_preimage 1 t f


/-- The integral of an almost-everywhere strongly measurable function over `UnitAddCircle` is
equal to the integral over an interval (t, t + 1] in `ℝ` of its lift to `ℝ`. -/
protected theorem intervalIntegral_preimage (t : ℝ) (f : UnitAddCircle → E) :
    ∫ a in t..t + 1, f a = ∫ b : UnitAddCircle, f b :=
  AddCircle.intervalIntegral_preimage 1 t f


/-- An auxiliary lemma for a more general `Function.Periodic.intervalIntegral_add_eq`. -/
theorem intervalIntegral_add_eq_of_pos (hf : Periodic f T) (hT : 0 < T) (t s : ℝ) :
    ∫ x in t..t + T, f x = ∫ x in s..s + T, f x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    T : Real
    hf : Function.Periodic f T
    hT : LT.lt 0 T
    t s : Real
    ⊢ Eq (intervalIntegral (fun x => f x) t (HAdd.hAdd t T) MeasureTheory.MeasureS …
  -/
  simp only [integral_of_le, hT.le, le_add_iff_nonneg_right]
  haveI : VAddInvariantMeasure (AddSubgroup.zmultiples T) ℝ volume :=
    ⟨fun c s _ => measure_preimage_add _ _ _⟩
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    T : Real
    hf : Function.Periodic f T
    hT : LT.lt 0 T
    t s : Real
    this : MeasureTheory.VAddInvariantMeasure (Subtype fun x => Membership.mem (Ad …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  apply IsAddFundamentalDomain.setIntegral_eq (G := AddSubgroup.zmultiples T)
  /-
    case hs
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    T : Real
    hf : Function.Periodic f T
    hT : LT.lt 0 T
    t s : Real
    this : MeasureTheory.VAddInvariantMeasure (Subtype fun x => Membership.mem (Ad …
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem (AddSu …
  -/
  exacts [isAddFundamentalDomain_Ioc hT t, isAddFundamentalDomain_Ioc hT s, hf.map_vadd_zmultiples]
  /-
    🎉 no goals
  -/


/-- If `f` is a periodic function with period `T`, then its integral over `[t, t + T]` does not
depend on `t`. -/
theorem intervalIntegral_add_eq (hf : Periodic f T) (t s : ℝ) :
    ∫ x in t..t + T, f x = ∫ x in s..s + T, f x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    T : Real
    hf : Function.Periodic f T
    t s : Real
    ⊢ Eq (intervalIntegral (fun x => f x) t (HAdd.hAdd t T) MeasureTheory.MeasureS …
  -/
  rcases lt_trichotomy (0 : ℝ) T with (hT | rfl | hT)
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      T : Real
      hf : Function.Periodic f T
      t s : Real
      hT : LT.lt 0 T
      ⊢ Eq (intervalIntegral (fun x => f x) t (HAdd.hAdd t T) MeasureTheory.MeasureS …
    -/
  · exact hf.intervalIntegral_add_eq_of_pos hT t s
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      t s : Real
      hf : Function.Periodic f 0
      ⊢ Eq (intervalIntegral (fun x => f x) t (HAdd.hAdd t 0) MeasureTheory.MeasureS …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      T : Real
      hf : Function.Periodic f T
      t s : Real
      hT : LT.lt T 0
      ⊢ Eq (intervalIntegral (fun x => f x) t (HAdd.hAdd t T) MeasureTheory.MeasureS …
    -/
  · rw [← neg_inj, ← integral_symm, ← integral_symm]
    simpa only [← sub_eq_add_neg, add_sub_cancel_right] using
      hf.neg.intervalIntegral_add_eq_of_pos (neg_pos.2 hT) (t + T) (s + T)


/-- If `f` is an integrable periodic function with period `T`, then its integral over `[t, s + T]`
is the sum of its integrals over the intervals `[t, s]` and `[t, t + T]`. -/
theorem intervalIntegral_add_eq_add (hf : Periodic f T) (t s : ℝ)
    (h_int : ∀ t₁ t₂, IntervalIntegrable f MeasureSpace.volume t₁ t₂) :
    ∫ x in t..s + T, f x = (∫ x in t..s, f x) + ∫ x in t..t + T, f x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    T : Real
    hf : Function.Periodic f T
    t s : Real
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable f MeasureTheory.MeasureSpace.volu …
    ⊢ Eq (intervalIntegral (fun x => f x) t (HAdd.hAdd s T) MeasureTheory.MeasureS …
  -/
  rw [hf.intervalIntegral_add_eq t s, integral_add_adjacent_intervals (h_int t s) (h_int s _)]
  /-
    🎉 no goals
  -/


/-- If `f` is an integrable periodic function with period `T`, and `n` is an integer, then its
integral over `[t, t + n • T]` is `n` times its integral over `[t, t + T]`. -/
theorem intervalIntegral_add_zsmul_eq (hf : Periodic f T) (n : ℤ) (t : ℝ)
    (h_int : ∀ t₁ t₂, IntervalIntegrable f MeasureSpace.volume t₁ t₂) :
    ∫ x in t..t + n • T, f x = n • ∫ x in t..t + T, f x := by
  -- Reduce to the case `b = 0`
  suffices (∫ x in (0)..(n • T), f x) = n • ∫ x in (0)..T, f x by
    simp only [hf.intervalIntegral_add_eq t 0, (hf.zsmul n).intervalIntegral_add_eq t 0, zero_add,
      this]
  -- First prove it for natural numbers
  have : ∀ m : ℕ, (∫ x in (0)..m • T, f x) = m • ∫ x in (0)..T, f x := fun m ↦ by
    induction' m with m ih
    · simp
    · simp only [succ_nsmul, hf.intervalIntegral_add_eq_add 0 (m • T) h_int, ih, zero_add]
  -- Then prove it for all integers
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    T : Real
    hf : Function.Periodic f T
    n : Int
    t : Real
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable f MeasureTheory.MeasureSpace.volu …
    this : ∀ (m : Nat), Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul m T) Me …
    ⊢ Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul n T) MeasureTheory.Measur …
  -/
  cases' n with n n
    /-
      case ofNat
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      T : Real
      hf : Function.Periodic f T
      t : Real
      h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable f MeasureTheory.MeasureSpace.volu …
      this : ∀ (m : Nat), Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul m T) Me …
      n : Nat
      ⊢ Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul (Int.ofNat n) T) MeasureT …
    -/
  · simp [← this n]
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      T : Real
      hf : Function.Periodic f T
      t : Real
      h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable f MeasureTheory.MeasureSpace.volu …
      this : ∀ (m : Nat), Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul m T) Me …
      n : Nat
      ⊢ Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul (Int.negSucc n) T) Measur …
    -/
  · conv_rhs => rw [negSucc_zsmul]
    /-
      case negSucc
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      T : Real
      hf : Function.Periodic f T
      t : Real
      h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable f MeasureTheory.MeasureSpace.volu …
      this : ∀ (m : Nat), Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul m T) Me …
      n : Nat
      ⊢ Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul (Int.negSucc n) T) Measur …
    -/
    have h₀ : Int.negSucc n • T + (n + 1) • T = 0 := by simp; linarith
    /-
      case negSucc
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      T : Real
      hf : Function.Periodic f T
      t : Real
      h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable f MeasureTheory.MeasureSpace.volu …
      this : ∀ (m : Nat), Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul m T) Me …
      n : Nat
      h₀ : Eq (HAdd.hAdd (HSMul.hSMul (Int.negSucc n) T) (HSMul.hSMul (HAdd.hAdd n 1 …
      ⊢ Eq (intervalIntegral (fun x => f x) 0 (HSMul.hSMul (Int.negSucc n) T) Measur …
    -/
    rw [integral_symm, ← (hf.nsmul (n + 1)).funext, neg_inj]
    simp_rw [integral_comp_add_right, h₀, zero_add, this (n + 1), add_comm T,
      hf.intervalIntegral_add_eq ((n + 1) • T) 0, zero_add]


/-- If `g : ℝ → ℝ` is periodic with period `T > 0`, then for any `t : ℝ`, the function
`t ↦ ∫ x in 0..t, g x` is bounded below by `t ↦ X + ⌊t/T⌋ • Y` for appropriate constants `X` and
`Y`. -/
theorem sInf_add_zsmul_le_integral_of_pos (hT : 0 < T) (t : ℝ) :
    (sInf ((fun t => ∫ x in (0)..t, g x) '' Icc 0 T) + ⌊t / T⌋ • ∫ x in (0)..T, g x) ≤
      ∫ x in (0)..t, g x := by
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    hT : LT.lt 0 T
    t : Real
    ⊢ LE.le (HAdd.hAdd (InfSet.sInf (Set.image (fun t => intervalIntegral (fun x = …
  -/
  let ε := Int.fract (t / T) * T
  conv_rhs =>
    rw [← Int.fract_div_mul_self_add_zsmul_eq T t (by linarith), ←
      integral_add_adjacent_intervals (h_int 0 ε) (h_int _ _)]
  rw [hg.intervalIntegral_add_zsmul_eq ⌊t / T⌋ ε h_int, hg.intervalIntegral_add_eq ε 0, zero_add,
    add_le_add_iff_right]
  exact (continuous_primitive h_int 0).continuousOn.sInf_image_Icc_le <|
    mem_Icc_of_Ico (Int.fract_div_mul_self_mem_Ico T t hT)


/-- If `g : ℝ → ℝ` is periodic with period `T > 0`, then for any `t : ℝ`, the function
`t ↦ ∫ x in 0..t, g x` is bounded above by `t ↦ X + ⌊t/T⌋ • Y` for appropriate constants `X` and
`Y`. -/
theorem integral_le_sSup_add_zsmul_of_pos (hT : 0 < T) (t : ℝ) :
    (∫ x in (0)..t, g x) ≤
      sSup ((fun t => ∫ x in (0)..t, g x) '' Icc 0 T) + ⌊t / T⌋ • ∫ x in (0)..T, g x := by
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    hT : LT.lt 0 T
    t : Real
    ⊢ LE.le (intervalIntegral (fun x => g x) 0 t MeasureTheory.MeasureSpace.volume …
  -/
  let ε := Int.fract (t / T) * T
  conv_lhs =>
    rw [← Int.fract_div_mul_self_add_zsmul_eq T t (by linarith), ←
      integral_add_adjacent_intervals (h_int 0 ε) (h_int _ _)]
  rw [hg.intervalIntegral_add_zsmul_eq ⌊t / T⌋ ε h_int, hg.intervalIntegral_add_eq ε 0, zero_add,
    add_le_add_iff_right]
  exact (continuous_primitive h_int 0).continuousOn.le_sSup_image_Icc
    (mem_Icc_of_Ico (Int.fract_div_mul_self_mem_Ico T t hT))


/-- If `g : ℝ → ℝ` is periodic with period `T > 0` and `0 < ∫ x in 0..T, g x`, then
`t ↦ ∫ x in 0..t, g x` tends to `∞` as `t` tends to `∞`. -/
theorem tendsto_atTop_intervalIntegral_of_pos (h₀ : 0 < ∫ x in (0)..T, g x) (hT : 0 < T) :
    Tendsto (fun t => ∫ x in (0)..t, g x) atTop atTop := by
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    h₀ : LT.lt 0 (intervalIntegral (fun x => g x) 0 T MeasureTheory.MeasureSpace.v …
    hT : LT.lt 0 T
    ⊢ Filter.Tendsto (fun t => intervalIntegral (fun x => g x) 0 t MeasureTheory.M …
  -/
  apply tendsto_atTop_mono (hg.sInf_add_zsmul_le_integral_of_pos h_int hT)
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    h₀ : LT.lt 0 (intervalIntegral (fun x => g x) 0 T MeasureTheory.MeasureSpace.v …
    hT : LT.lt 0 T
    ⊢ Filter.Tendsto (fun n => HAdd.hAdd (InfSet.sInf (Set.image (fun t => interva …
  -/
  apply atTop.tendsto_atTop_add_const_left (sInf <| (fun t => ∫ x in (0)..t, g x) '' Icc 0 T)
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    h₀ : LT.lt 0 (intervalIntegral (fun x => g x) 0 T MeasureTheory.MeasureSpace.v …
    hT : LT.lt 0 T
    ⊢ Filter.Tendsto (fun x => HSMul.hSMul (Int.floor (HDiv.hDiv x T)) (intervalIn …
  -/
  apply Tendsto.atTop_zsmul_const h₀
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    h₀ : LT.lt 0 (intervalIntegral (fun x => g x) 0 T MeasureTheory.MeasureSpace.v …
    hT : LT.lt 0 T
    ⊢ Filter.Tendsto (fun x => Int.floor (HDiv.hDiv x T)) Filter.atTop Filter.atTop
  -/
  exact tendsto_floor_atTop.comp (tendsto_id.atTop_mul_const (inv_pos.mpr hT))
  /-
    🎉 no goals
  -/


/-- If `g : ℝ → ℝ` is periodic with period `T > 0` and `0 < ∫ x in 0..T, g x`, then
`t ↦ ∫ x in 0..t, g x` tends to `-∞` as `t` tends to `-∞`. -/
theorem tendsto_atBot_intervalIntegral_of_pos (h₀ : 0 < ∫ x in (0)..T, g x) (hT : 0 < T) :
    Tendsto (fun t => ∫ x in (0)..t, g x) atBot atBot := by
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    h₀ : LT.lt 0 (intervalIntegral (fun x => g x) 0 T MeasureTheory.MeasureSpace.v …
    hT : LT.lt 0 T
    ⊢ Filter.Tendsto (fun t => intervalIntegral (fun x => g x) 0 t MeasureTheory.M …
  -/
  apply tendsto_atBot_mono (hg.integral_le_sSup_add_zsmul_of_pos h_int hT)
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    h₀ : LT.lt 0 (intervalIntegral (fun x => g x) 0 T MeasureTheory.MeasureSpace.v …
    hT : LT.lt 0 T
    ⊢ Filter.Tendsto (fun n => HAdd.hAdd (SupSet.sSup (Set.image (fun t => interva …
  -/
  apply atBot.tendsto_atBot_add_const_left (sSup <| (fun t => ∫ x in (0)..t, g x) '' Icc 0 T)
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    h₀ : LT.lt 0 (intervalIntegral (fun x => g x) 0 T MeasureTheory.MeasureSpace.v …
    hT : LT.lt 0 T
    ⊢ Filter.Tendsto (fun x => HSMul.hSMul (Int.floor (HDiv.hDiv x T)) (intervalIn …
  -/
  apply Tendsto.atBot_zsmul_const h₀
  /-
    T : Real
    g : Real → Real
    hg : Function.Periodic g T
    h_int : ∀ (t₁ t₂ : Real), IntervalIntegrable g MeasureTheory.MeasureSpace.volu …
    h₀ : LT.lt 0 (intervalIntegral (fun x => g x) 0 T MeasureTheory.MeasureSpace.v …
    hT : LT.lt 0 T
    ⊢ Filter.Tendsto (fun x => Int.floor (HDiv.hDiv x T)) Filter.atBot Filter.atBot
  -/
  exact tendsto_floor_atBot.comp (tendsto_id.atBot_mul_const (inv_pos.mpr hT))
  /-
    🎉 no goals
  -/


/-- If `g : ℝ → ℝ` is periodic with period `T > 0` and `∀ x, 0 < g x`, then `t ↦ ∫ x in 0..t, g x`
tends to `∞` as `t` tends to `∞`. -/
theorem tendsto_atTop_intervalIntegral_of_pos' (h₀ : ∀ x, 0 < g x) (hT : 0 < T) :
    Tendsto (fun t => ∫ x in (0)..t, g x) atTop atTop :=
  hg.tendsto_atTop_intervalIntegral_of_pos h_int (intervalIntegral_pos_of_pos (h_int 0 T) h₀ hT) hT


/-- If `g : ℝ → ℝ` is periodic with period `T > 0` and `∀ x, 0 < g x`, then `t ↦ ∫ x in 0..t, g x`
tends to `-∞` as `t` tends to `-∞`. -/
theorem tendsto_atBot_intervalIntegral_of_pos' (h₀ : ∀ x, 0 < g x) (hT : 0 < T) :
    Tendsto (fun t => ∫ x in (0)..t, g x) atBot atBot :=
  hg.tendsto_atBot_intervalIntegral_of_pos h_int (intervalIntegral_pos_of_pos (h_int 0 T) h₀ hT) hT


