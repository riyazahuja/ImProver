/-- Measure on `α` such that for a measurable set `s`, `ρ.IicSnd r s = ρ (s ×ˢ Iic r)`. -/
noncomputable def IicSnd (r : ℝ) : Measure α :=
  (ρ.restrict (univ ×ˢ Iic r)).fst


theorem IicSnd_apply (r : ℝ) {s : Set α} (hs : MeasurableSet s) :
    ρ.IicSnd r s = ρ (s ×ˢ Iic r) := by
  rw [IicSnd, fst_apply hs, restrict_apply' (MeasurableSet.univ.prod measurableSet_Iic),
    univ_prod, Set.prod_eq]


theorem IicSnd_univ (r : ℝ) : ρ.IicSnd r univ = ρ (univ ×ˢ Iic r) :=
  IicSnd_apply ρ r MeasurableSet.univ


@[gcongr]
theorem IicSnd_mono {r r' : ℝ} (h_le : r ≤ r') : ρ.IicSnd r ≤ ρ.IicSnd r' := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    r r' : Real
    h_le : LE.le r r'
    ⊢ LE.le (ρ.IicSnd r) (ρ.IicSnd r')
  -/
  unfold IicSnd; gcongr
                 /-
                   🎉 no goals
                 -/


theorem IicSnd_le_fst (r : ℝ) : ρ.IicSnd r ≤ ρ.fst :=
  fst_mono restrict_le_self


theorem IicSnd_ac_fst (r : ℝ) : ρ.IicSnd r ≪ ρ.fst :=
  Measure.absolutelyContinuous_of_le (IicSnd_le_fst ρ r)


theorem IsFiniteMeasure.IicSnd {ρ : Measure (α × ℝ)} [IsFiniteMeasure ρ] (r : ℝ) :
    IsFiniteMeasure (ρ.IicSnd r) :=
  isFiniteMeasure_of_le _ (IicSnd_le_fst ρ _)


theorem iInf_IicSnd_gt (t : ℚ) {s : Set α} (hs : MeasurableSet s) [IsFiniteMeasure ρ] :
    ⨅ r : { r' : ℚ // t < r' }, ρ.IicSnd r s = ρ.IicSnd t s := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    t : Rat
    s : Set α
    hs : MeasurableSet s
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ Eq (iInf fun r => (ρ.IicSnd ↑↑r) s) ((ρ.IicSnd ↑t) s)
  -/
  simp_rw [ρ.IicSnd_apply _ hs, Measure.iInf_rat_gt_prod_Iic hs]
  /-
    🎉 no goals
  -/


theorem tendsto_IicSnd_atTop {s : Set α} (hs : MeasurableSet s) :
    Tendsto (fun r : ℚ ↦ ρ.IicSnd r s) atTop (𝓝 (ρ.fst s)) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    s : Set α
    hs : MeasurableSet s
    ⊢ Filter.Tendsto (fun r => (ρ.IicSnd ↑r) s) Filter.atTop (nhds (ρ.fst s))
  -/
  simp_rw [ρ.IicSnd_apply _ hs, fst_apply hs, ← prod_univ]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    s : Set α
    hs : MeasurableSet s
    ⊢ Filter.Tendsto (fun r => ρ (SProd.sprod s (Set.Iic ↑r))) Filter.atTop (nhds  …
  -/
  rw [← Real.iUnion_Iic_rat, prod_iUnion]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    s : Set α
    hs : MeasurableSet s
    ⊢ Filter.Tendsto (fun r => ρ (SProd.sprod s (Set.Iic ↑r))) Filter.atTop (nhds  …
  -/
  apply tendsto_measure_iUnion_atTop
  /-
    case hm
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    s : Set α
    hs : MeasurableSet s
    ⊢ Monotone fun r => SProd.sprod s (Set.Iic ↑r)
  -/
  exact monotone_const.set_prod Rat.cast_mono.Iic
  /-
    🎉 no goals
  -/


theorem tendsto_IicSnd_atBot [IsFiniteMeasure ρ] {s : Set α} (hs : MeasurableSet s) :
    Tendsto (fun r : ℚ ↦ ρ.IicSnd r s) atBot (𝓝 0) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set α
    hs : MeasurableSet s
    ⊢ Filter.Tendsto (fun r => (ρ.IicSnd ↑r) s) Filter.atBot (nhds 0)
  -/
  simp_rw [ρ.IicSnd_apply _ hs]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set α
    hs : MeasurableSet s
    ⊢ Filter.Tendsto (fun r => ρ (SProd.sprod s (Set.Iic ↑r))) Filter.atBot (nhds 0)
  -/
  have h_empty : ρ (s ×ˢ ∅) = 0 := by simp only [prod_empty, measure_empty]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set α
    hs : MeasurableSet s
    h_empty : Eq (ρ (SProd.sprod s EmptyCollection.emptyCollection)) 0
    ⊢ Filter.Tendsto (fun r => ρ (SProd.sprod s (Set.Iic ↑r))) Filter.atBot (nhds 0)
  -/
  rw [← h_empty, ← Real.iInter_Iic_rat, prod_iInter]
  suffices h_neg :
      Tendsto (fun r : ℚ ↦ ρ (s ×ˢ Iic ↑(-r))) atTop (𝓝 (ρ (⋂ r : ℚ, s ×ˢ Iic ↑(-r)))) by
    have h_inter_eq : ⋂ r : ℚ, s ×ˢ Iic ↑(-r) = ⋂ r : ℚ, s ×ˢ Iic (r : ℝ) := by
      ext1 x
      simp only [Rat.cast_eq_id, id, mem_iInter, mem_prod, mem_Iic]
      refine ⟨fun h i ↦ ⟨(h i).1, ?_⟩, fun h i ↦ ⟨(h i).1, ?_⟩⟩ <;> have h' := h (-i)
      · rw [neg_neg] at h'; exact h'.2
      · exact h'.2
    rw [h_inter_eq] at h_neg
    have h_fun_eq : (fun r : ℚ ↦ ρ (s ×ˢ Iic (r : ℝ))) = fun r : ℚ ↦ ρ (s ×ˢ Iic ↑(- -r)) := by
      simp_rw [neg_neg]
    rw [h_fun_eq]
    exact h_neg.comp tendsto_neg_atBot_atTop
  refine tendsto_measure_iInter_atTop (fun q ↦ (hs.prod measurableSet_Iic).nullMeasurableSet)
    ?_ ⟨0, measure_ne_top ρ _⟩
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set α
    hs : MeasurableSet s
    h_empty : Eq (ρ (SProd.sprod s EmptyCollection.emptyCollection)) 0
    ⊢ Antitone fun r => SProd.sprod s (Set.Iic ↑(Neg.neg r))
  -/
  refine fun q r hqr ↦ Set.prod_mono subset_rfl fun x hx ↦ ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set α
    hs : MeasurableSet s
    h_empty : Eq (ρ (SProd.sprod s EmptyCollection.emptyCollection)) 0
    q r : Rat
    hqr : LE.le q r
    x : Real
    hx : Membership.mem (Set.Iic ↑(Neg.neg r)) x
    ⊢ Membership.mem (Set.Iic ↑(Neg.neg q)) x
  -/
  simp only [Rat.cast_neg, mem_Iic] at hx ⊢
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set α
    hs : MeasurableSet s
    h_empty : Eq (ρ (SProd.sprod s EmptyCollection.emptyCollection)) 0
    q r : Rat
    hqr : LE.le q r
    x : Real
    hx : LE.le x (Neg.neg ↑r)
    ⊢ LE.le x (Neg.neg ↑q)
  -/
  refine hx.trans (neg_le_neg ?_)
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set α
    hs : MeasurableSet s
    h_empty : Eq (ρ (SProd.sprod s EmptyCollection.emptyCollection)) 0
    q r : Rat
    hqr : LE.le q r
    x : Real
    hx : LE.le x (Neg.neg ↑r)
    ⊢ LE.le ↑q ↑r
  -/
  exact mod_cast hqr
  /-
    🎉 no goals
  -/


/-- `preCDF` is the Radon-Nikodym derivative of `ρ.IicSnd` with respect to `ρ.fst` at each
`r : ℚ`. This function `ℚ → α → ℝ≥0∞` is such that for almost all `a : α`, the function `ℚ → ℝ≥0∞`
satisfies the properties of a cdf (monotone with limit 0 at -∞ and 1 at +∞, right-continuous).

We define this function on `ℚ` and not `ℝ` because `ℚ` is countable, which allows us to prove
properties of the form `∀ᵐ a ∂ρ.fst, ∀ q, P (preCDF q a)`, instead of the weaker
`∀ q, ∀ᵐ a ∂ρ.fst, P (preCDF q a)`. -/
noncomputable def preCDF (ρ : Measure (α × ℝ)) (r : ℚ) : α → ℝ≥0∞ :=
  Measure.rnDeriv (ρ.IicSnd r) ρ.fst


theorem measurable_preCDF {ρ : Measure (α × ℝ)} {r : ℚ} : Measurable (preCDF ρ r) :=
  Measure.measurable_rnDeriv _ _


lemma measurable_preCDF' {ρ : Measure (α × ℝ)} :
    Measurable fun a r ↦ (preCDF ρ r a).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    ⊢ Measurable fun a r => (ProbabilityTheory.preCDF ρ r a).toReal
  -/
  rw [measurable_pi_iff]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    ⊢ ∀ (a : Rat), Measurable fun x => (ProbabilityTheory.preCDF ρ a x).toReal
  -/
  exact fun _ ↦ measurable_preCDF.ennreal_toReal
  /-
    🎉 no goals
  -/


theorem withDensity_preCDF (ρ : Measure (α × ℝ)) (r : ℚ) [IsFiniteMeasure ρ] :
    ρ.fst.withDensity (preCDF ρ r) = ρ.IicSnd r :=
  Measure.absolutelyContinuous_iff_withDensity_rnDeriv_eq.mp (Measure.IicSnd_ac_fst ρ r)


theorem setLIntegral_preCDF_fst (ρ : Measure (α × ℝ)) (r : ℚ) {s : Set α} (hs : MeasurableSet s)
    [IsFiniteMeasure ρ] : ∫⁻ x in s, preCDF ρ r x ∂ρ.fst = ρ.IicSnd r s := by
  have : ∀ r, ∫⁻ x in s, preCDF ρ r x ∂ρ.fst = ∫⁻ x in s, (preCDF ρ r * 1) x ∂ρ.fst := by
    simp only [mul_one, eq_self_iff_true, forall_const]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    r : Rat
    s : Set α
    hs : MeasurableSet s
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    this : ∀ (r : Rat), Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun x => Pr …
    ⊢ Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun x => ProbabilityTheory.pr …
  -/
  rw [this, ← setLIntegral_withDensity_eq_setLIntegral_mul _ measurable_preCDF _ hs]
  · simp only [withDensity_preCDF ρ r, Pi.one_apply, lintegral_one, Measure.restrict_apply,
      MeasurableSet.univ, univ_inter]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      r : Rat
      s : Set α
      hs : MeasurableSet s
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      this : ∀ (r : Rat), Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun x => Pr …
      ⊢ Measurable 1
    -/
  · rw [(_ : (1 : α → ℝ≥0∞) = fun _ ↦ 1)]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      r : Rat
      s : Set α
      hs : MeasurableSet s
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      this : ∀ (r : Rat), Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun x => Pr …
      ⊢ Measurable fun x => 1
    -/
    exacts [measurable_const, rfl]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_preCDF_fst := setLIntegral_preCDF_fst


lemma lintegral_preCDF_fst (ρ : Measure (α × ℝ)) (r : ℚ) [IsFiniteMeasure ρ] :
    ∫⁻ x, preCDF ρ r x ∂ρ.fst = ρ.IicSnd r univ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    r : Rat
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ Eq (MeasureTheory.lintegral ρ.fst fun x => ProbabilityTheory.preCDF ρ r x) ( …
  -/
  rw [← setLIntegral_univ, setLIntegral_preCDF_fst ρ r MeasurableSet.univ]
  /-
    🎉 no goals
  -/


theorem monotone_preCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] :
    ∀ᵐ a ∂ρ.fst, Monotone fun r ↦ preCDF ρ r a := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ Filter.Eventually (fun a => Monotone fun r => ProbabilityTheory.preCDF ρ r a …
  -/
  simp_rw [Monotone, ae_all_iff]
  refine fun r r' hrr' ↦ ae_le_of_forall_setLIntegral_le_of_sigmaFinite measurable_preCDF
    fun s hs _ ↦ ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r r' : Rat
    hrr' : LE.le r r'
    s : Set α
    hs : MeasurableSet s
    x✝ : LT.lt (ρ.fst s) Top.top
    ⊢ LE.le (MeasureTheory.lintegral (ρ.fst.restrict s) fun x => ProbabilityTheory …
  -/
  rw [setLIntegral_preCDF_fst ρ r hs, setLIntegral_preCDF_fst ρ r' hs]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r r' : Rat
    hrr' : LE.le r r'
    s : Set α
    hs : MeasurableSet s
    x✝ : LT.lt (ρ.fst s) Top.top
    ⊢ LE.le ((ρ.IicSnd ↑r) s) ((ρ.IicSnd ↑r') s)
  -/
  exact Measure.IicSnd_mono ρ (mod_cast hrr') s
  /-
    🎉 no goals
  -/


theorem preCDF_le_one (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] :
    ∀ᵐ a ∂ρ.fst, ∀ r, preCDF ρ r a ≤ 1 := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ Filter.Eventually (fun a => ∀ (r : Rat), LE.le (ProbabilityTheory.preCDF ρ r …
  -/
  rw [ae_all_iff]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ ∀ (i : Rat), Filter.Eventually (fun a => LE.le (ProbabilityTheory.preCDF ρ i …
  -/
  refine fun r ↦ ae_le_of_forall_setLIntegral_le_of_sigmaFinite measurable_preCDF fun s hs _ ↦ ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r : Rat
    s : Set α
    hs : MeasurableSet s
    x✝ : LT.lt (ρ.fst s) Top.top
    ⊢ LE.le (MeasureTheory.lintegral (ρ.fst.restrict s) fun x => ProbabilityTheory …
  -/
  rw [setLIntegral_preCDF_fst ρ r hs]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r : Rat
    s : Set α
    hs : MeasurableSet s
    x✝ : LT.lt (ρ.fst s) Top.top
    ⊢ LE.le ((ρ.IicSnd ↑r) s) (MeasureTheory.lintegral (ρ.fst.restrict s) fun x => …
  -/
  simp only [Pi.one_apply, lintegral_one, Measure.restrict_apply, MeasurableSet.univ, univ_inter]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r : Rat
    s : Set α
    hs : MeasurableSet s
    x✝ : LT.lt (ρ.fst s) Top.top
    ⊢ LE.le ((ρ.IicSnd ↑r) s) (ρ.fst s)
  -/
  exact Measure.IicSnd_le_fst ρ r s
  /-
    🎉 no goals
  -/


lemma setIntegral_preCDF_fst (ρ : Measure (α × ℝ)) (r : ℚ) {s : Set α} (hs : MeasurableSet s)
    [IsFiniteMeasure ρ] :
    ∫ x in s, (preCDF ρ r x).toReal ∂ρ.fst = (ρ.IicSnd r s).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    r : Rat
    s : Set α
    hs : MeasurableSet s
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ Eq (MeasureTheory.integral (ρ.fst.restrict s) fun x => (ProbabilityTheory.pr …
  -/
  rw [integral_toReal]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      r : Rat
      s : Set α
      hs : MeasurableSet s
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      ⊢ Eq (MeasureTheory.lintegral (ρ.fst.restrict s) fun a => ProbabilityTheory.pr …
    -/
  · rw [setLIntegral_preCDF_fst _ _ hs]
    /-
      🎉 no goals
    -/
    /-
      case hfm
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      r : Rat
      s : Set α
      hs : MeasurableSet s
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      ⊢ AEMeasurable (ProbabilityTheory.preCDF ρ r) (ρ.fst.restrict s)
    -/
  · exact measurable_preCDF.aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      r : Rat
      s : Set α
      hs : MeasurableSet s
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      ⊢ Filter.Eventually (fun x => LT.lt (ProbabilityTheory.preCDF ρ r x) Top.top)  …
    -/
  · refine ae_restrict_of_ae ?_
    /-
      case hf
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      r : Rat
      s : Set α
      hs : MeasurableSet s
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      ⊢ Filter.Eventually (fun x => LT.lt (ProbabilityTheory.preCDF ρ r x) Top.top)  …
    -/
    filter_upwards [preCDF_le_one ρ] with a ha
    /-
      case h
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      r : Rat
      s : Set α
      hs : MeasurableSet s
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : α
      ha : ∀ (r : Rat), LE.le (ProbabilityTheory.preCDF ρ r a) 1
      ⊢ LT.lt (ProbabilityTheory.preCDF ρ r a) Top.top
    -/
    exact (ha r).trans_lt ENNReal.one_lt_top
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_preCDF_fst := setIntegral_preCDF_fst


lemma integral_preCDF_fst (ρ : Measure (α × ℝ)) (r : ℚ) [IsFiniteMeasure ρ] :
    ∫ x, (preCDF ρ r x).toReal ∂ρ.fst = (ρ.IicSnd r univ).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    r : Rat
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ Eq (MeasureTheory.integral ρ.fst fun x => (ProbabilityTheory.preCDF ρ r x).t …
  -/
  rw [← setIntegral_univ, setIntegral_preCDF_fst ρ _ MeasurableSet.univ]
  /-
    🎉 no goals
  -/


lemma integrable_preCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] (x : ℚ) :
    Integrable (fun a ↦ (preCDF ρ x a).toReal) ρ.fst := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    x : Rat
    ⊢ MeasureTheory.Integrable (fun a => (ProbabilityTheory.preCDF ρ x a).toReal)  …
  -/
  refine integrable_of_forall_fin_meas_le _ (measure_lt_top ρ.fst univ) ?_ fun t _ _ ↦ ?_
    /-
      case refine_1
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      x : Rat
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => (ProbabilityTheory.preCDF ρ x a …
    -/
  · exact measurable_preCDF.ennreal_toReal.aestronglyMeasurable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      x : Rat
      t : Set α
      x✝¹ : MeasurableSet t
      x✝ : Ne (ρ.fst t) Top.top
      ⊢ LE.le (MeasureTheory.lintegral (ρ.fst.restrict t) fun x_1 => ↑(NNNorm.nnnorm …
    -/
  · simp_rw [← ofReal_norm_eq_coe_nnnorm, Real.norm_of_nonneg ENNReal.toReal_nonneg]
    /-
      case refine_2
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      x : Rat
      t : Set α
      x✝¹ : MeasurableSet t
      x✝ : Ne (ρ.fst t) Top.top
      ⊢ LE.le (MeasureTheory.lintegral (ρ.fst.restrict t) fun x_1 => ENNReal.ofReal  …
    -/
    rw [← lintegral_one]
    /-
      case refine_2
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      x : Rat
      t : Set α
      x✝¹ : MeasurableSet t
      x✝ : Ne (ρ.fst t) Top.top
      ⊢ LE.le (MeasureTheory.lintegral (ρ.fst.restrict t) fun x_1 => ENNReal.ofReal  …
    -/
    refine (setLIntegral_le_lintegral _ _).trans (lintegral_mono_ae ?_)
    /-
      case refine_2
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      x : Rat
      t : Set α
      x✝¹ : MeasurableSet t
      x✝ : Ne (ρ.fst t) Top.top
      ⊢ Filter.Eventually (fun a => LE.le (ENNReal.ofReal (ProbabilityTheory.preCDF  …
    -/
    filter_upwards [preCDF_le_one ρ] with a ha using ENNReal.ofReal_toReal_le.trans (ha _)
    /-
      🎉 no goals
    -/


lemma isRatCondKernelCDFAux_preCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] :
    IsRatCondKernelCDFAux (fun p r ↦ (preCDF ρ r p.2).toReal)
      (Kernel.const Unit ρ) (Kernel.const Unit ρ.fst) where
  measurable := measurable_preCDF'.comp measurable_snd
  mono' a r r' hrr' := by
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      r r' : Rat
      hrr' : LE.le r r'
      ⊢ Filter.Eventually (fun c => LE.le (ProbabilityTheory.preCDF ρ r { fst := a,  …
    -/
    filter_upwards [monotone_preCDF ρ, preCDF_le_one ρ] with a h₁ h₂
    /-
      case h
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a✝ : Unit
      r r' : Rat
      hrr' : LE.le r r'
      a : α
      h₁ : Monotone fun r => ProbabilityTheory.preCDF ρ r a
      h₂ : ∀ (r : Rat), LE.le (ProbabilityTheory.preCDF ρ r a) 1
      ⊢ LE.le (ProbabilityTheory.preCDF ρ r a).toReal (ProbabilityTheory.preCDF ρ r' …
    -/
    exact ENNReal.toReal_mono ((h₂ _).trans_lt ENNReal.one_lt_top).ne (h₁ hrr')
    /-
      🎉 no goals
    -/
                    /-
                      α : Type u_1
                      mα : MeasurableSpace α
                      ρ : MeasureTheory.Measure (Prod α Real)
                      inst✝ : MeasureTheory.IsFiniteMeasure ρ
                      x✝ : Unit
                      q : Rat
                      ⊢ Filter.Eventually (fun c => LE.le 0 (ProbabilityTheory.preCDF ρ q { fst := x …
                    -/
  nonneg' _ q := by simp
                    /-
                      🎉 no goals
                    -/
  le_one' a q := by
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      q : Rat
      ⊢ Filter.Eventually (fun c => LE.le (ProbabilityTheory.preCDF ρ q { fst := a,  …
    -/
    simp only [Kernel.const_apply, forall_const]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      q : Rat
      ⊢ Filter.Eventually (fun c => LE.le (ProbabilityTheory.preCDF ρ q c).toReal 1) …
    -/
    filter_upwards [preCDF_le_one ρ] with a ha
    /-
      case h
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a✝ : Unit
      q : Rat
      a : α
      ha : ∀ (r : Rat), LE.le (ProbabilityTheory.preCDF ρ r a) 1
      ⊢ LE.le (ProbabilityTheory.preCDF ρ q a).toReal 1
    -/
    refine ENNReal.toReal_le_of_le_ofReal zero_le_one ?_
    /-
      case h
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a✝ : Unit
      q : Rat
      a : α
      ha : ∀ (r : Rat), LE.le (ProbabilityTheory.preCDF ρ r a) 1
      ⊢ LE.le (ProbabilityTheory.preCDF ρ q a) (ENNReal.ofReal 1)
    -/
    simp [ha]
    /-
      🎉 no goals
    -/
  tendsto_integral_of_antitone a s _ hs_tendsto := by
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      s : Nat → Rat
      x✝ : Antitone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
      ⊢ Filter.Tendsto (fun m => MeasureTheory.integral ((ProbabilityTheory.Kernel.c …
    -/
    simp_rw [Kernel.const_apply, integral_preCDF_fst ρ]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      s : Nat → Rat
      x✝ : Antitone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
      ⊢ Filter.Tendsto (fun m => ((ρ.IicSnd ↑(s m)) Set.univ).toReal) Filter.atTop ( …
    -/
    have h := ρ.tendsto_IicSnd_atBot MeasurableSet.univ
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      s : Nat → Rat
      x✝ : Antitone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
      h : Filter.Tendsto (fun r => (ρ.IicSnd ↑r) Set.univ) Filter.atBot (nhds 0)
      ⊢ Filter.Tendsto (fun m => ((ρ.IicSnd ↑(s m)) Set.univ).toReal) Filter.atTop ( …
    -/
    rw [← ENNReal.zero_toReal]
    have h0 : Tendsto ENNReal.toReal (𝓝 0) (𝓝 0) :=
      ENNReal.continuousAt_toReal ENNReal.zero_ne_top
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      s : Nat → Rat
      x✝ : Antitone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
      h : Filter.Tendsto (fun r => (ρ.IicSnd ↑r) Set.univ) Filter.atBot (nhds 0)
      h0 : Filter.Tendsto ENNReal.toReal (nhds 0) (nhds 0)
      ⊢ Filter.Tendsto (fun m => ((ρ.IicSnd ↑(s m)) Set.univ).toReal) Filter.atTop ( …
    -/
    exact h0.comp (h.comp hs_tendsto)
    /-
      🎉 no goals
    -/
  tendsto_integral_of_monotone a s _ hs_tendsto := by
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      s : Nat → Rat
      x✝ : Monotone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
      ⊢ Filter.Tendsto (fun m => MeasureTheory.integral ((ProbabilityTheory.Kernel.c …
    -/
    simp_rw [Kernel.const_apply, integral_preCDF_fst ρ]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      s : Nat → Rat
      x✝ : Monotone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
      ⊢ Filter.Tendsto (fun m => ((ρ.IicSnd ↑(s m)) Set.univ).toReal) Filter.atTop ( …
    -/
    have h := ρ.tendsto_IicSnd_atTop MeasurableSet.univ
    have h0 : Tendsto ENNReal.toReal (𝓝 (ρ.fst univ)) (𝓝 (ρ.fst univ).toReal) :=
      ENNReal.continuousAt_toReal (measure_ne_top _ _)
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      a : Unit
      s : Nat → Rat
      x✝ : Monotone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
      h : Filter.Tendsto (fun r => (ρ.IicSnd ↑r) Set.univ) Filter.atTop (nhds (ρ.fst …
      h0 : Filter.Tendsto ENNReal.toReal (nhds (ρ.fst Set.univ)) (nhds (ρ.fst Set.un …
      ⊢ Filter.Tendsto (fun m => ((ρ.IicSnd ↑(s m)) Set.univ).toReal) Filter.atTop ( …
    -/
    exact h0.comp (h.comp hs_tendsto)
    /-
      🎉 no goals
    -/
  integrable _ q := integrable_preCDF ρ q
  setIntegral a s hs q := by rw [Kernel.const_apply, Kernel.const_apply,
    setIntegral_preCDF_fst _ _ hs, Measure.IicSnd_apply _ _ hs]


lemma isRatCondKernelCDF_preCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] :
    IsRatCondKernelCDF (fun p r ↦ (preCDF ρ r p.2).toReal)
      (Kernel.const Unit ρ) (Kernel.const Unit ρ.fst) :=
  (isRatCondKernelCDFAux_preCDF ρ).isRatCondKernelCDF


/-- Conditional cdf of the measure given the value on `α`, as a Stieltjes function. -/
noncomputable def condCDF (ρ : Measure (α × ℝ)) (a : α) : StieltjesFunction :=
  stieltjesOfMeasurableRat (fun a r ↦ (preCDF ρ r a).toReal) measurable_preCDF' a


lemma condCDF_eq_stieltjesOfMeasurableRat_unit_prod (ρ : Measure (α × ℝ)) (a : α) :
    condCDF ρ a = stieltjesOfMeasurableRat (fun (p : Unit × α) r ↦ (preCDF ρ r p.2).toReal)
      (measurable_preCDF'.comp measurable_snd) ((), a) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    a : α
    ⊢ Eq (ProbabilityTheory.condCDF ρ a) (ProbabilityTheory.stieltjesOfMeasurableR …
  -/
  ext x
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    a : α
    x : Real
    ⊢ Eq (↑(ProbabilityTheory.condCDF ρ a) x) (↑(ProbabilityTheory.stieltjesOfMeas …
  -/
  rw [condCDF, ← stieltjesOfMeasurableRat_unit_prod]
  /-
    🎉 no goals
  -/


lemma isCondKernelCDF_condCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] :
    IsCondKernelCDF (fun p : Unit × α ↦ condCDF ρ p.2) (Kernel.const Unit ρ)
      (Kernel.const Unit ρ.fst) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ ProbabilityTheory.IsCondKernelCDF (fun p => ProbabilityTheory.condCDF ρ p.2) …
  -/
  simp_rw [condCDF_eq_stieltjesOfMeasurableRat_unit_prod ρ]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ ProbabilityTheory.IsCondKernelCDF (fun p => ProbabilityTheory.stieltjesOfMea …
  -/
  exact isCondKernelCDF_stieltjesOfMeasurableRat (isRatCondKernelCDF_preCDF ρ)
  /-
    🎉 no goals
  -/


/-- The conditional cdf is non-negative for all `a : α`. -/
theorem condCDF_nonneg (ρ : Measure (α × ℝ)) (a : α) (r : ℝ) : 0 ≤ condCDF ρ a r :=
  stieltjesOfMeasurableRat_nonneg _ a r


/-- The conditional cdf is lower or equal to 1 for all `a : α`. -/
theorem condCDF_le_one (ρ : Measure (α × ℝ)) (a : α) (x : ℝ) : condCDF ρ a x ≤ 1 :=
  stieltjesOfMeasurableRat_le_one _ _ _


/-- The conditional cdf tends to 0 at -∞ for all `a : α`. -/
theorem tendsto_condCDF_atBot (ρ : Measure (α × ℝ)) (a : α) :
    Tendsto (condCDF ρ a) atBot (𝓝 0) := tendsto_stieltjesOfMeasurableRat_atBot _ _


/-- The conditional cdf tends to 1 at +∞ for all `a : α`. -/
theorem tendsto_condCDF_atTop (ρ : Measure (α × ℝ)) (a : α) :
    Tendsto (condCDF ρ a) atTop (𝓝 1) := tendsto_stieltjesOfMeasurableRat_atTop _ _


theorem condCDF_ae_eq (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] (r : ℚ) :
    (fun a ↦ condCDF ρ a r) =ᵐ[ρ.fst] fun a ↦ (preCDF ρ r a).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r : Rat
    ⊢ (MeasureTheory.ae ρ.fst).EventuallyEq (fun a => ↑(ProbabilityTheory.condCDF  …
  -/
  simp_rw [condCDF_eq_stieltjesOfMeasurableRat_unit_prod ρ]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r : Rat
    ⊢ (MeasureTheory.ae ρ.fst).EventuallyEq (fun a => ↑(ProbabilityTheory.stieltje …
  -/
  exact stieltjesOfMeasurableRat_ae_eq (isRatCondKernelCDF_preCDF ρ) () r
  /-
    🎉 no goals
  -/


theorem ofReal_condCDF_ae_eq (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] (r : ℚ) :
    (fun a ↦ ENNReal.ofReal (condCDF ρ a r)) =ᵐ[ρ.fst] preCDF ρ r := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r : Rat
    ⊢ (MeasureTheory.ae ρ.fst).EventuallyEq (fun a => ENNReal.ofReal (↑(Probabilit …
  -/
  filter_upwards [condCDF_ae_eq ρ r, preCDF_le_one ρ] with a ha ha_le_one
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r : Rat
    a : α
    ha : Eq (↑(ProbabilityTheory.condCDF ρ a) ↑r) (ProbabilityTheory.preCDF ρ r a) …
    ha_le_one : ∀ (r : Rat), LE.le (ProbabilityTheory.preCDF ρ r a) 1
    ⊢ Eq (ENNReal.ofReal (↑(ProbabilityTheory.condCDF ρ a) ↑r)) (ProbabilityTheory …
  -/
  rw [ha, ENNReal.ofReal_toReal]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    r : Rat
    a : α
    ha : Eq (↑(ProbabilityTheory.condCDF ρ a) ↑r) (ProbabilityTheory.preCDF ρ r a) …
    ha_le_one : ∀ (r : Rat), LE.le (ProbabilityTheory.preCDF ρ r a) 1
    ⊢ Ne (ProbabilityTheory.preCDF ρ r a) Top.top
  -/
  exact ((ha_le_one r).trans_lt ENNReal.one_lt_top).ne
  /-
    🎉 no goals
  -/


/-- The conditional cdf is a measurable function of `a : α` for all `x : ℝ`. -/
theorem measurable_condCDF (ρ : Measure (α × ℝ)) (x : ℝ) : Measurable fun a ↦ condCDF ρ a x :=
  measurable_stieltjesOfMeasurableRat _ _


/-- The conditional cdf is a strongly measurable function of `a : α` for all `x : ℝ`. -/
theorem stronglyMeasurable_condCDF (ρ : Measure (α × ℝ)) (x : ℝ) :
    StronglyMeasurable fun a ↦ condCDF ρ a x := stronglyMeasurable_stieltjesOfMeasurableRat _ _


theorem setLIntegral_condCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] (x : ℝ) {s : Set α}
    (hs : MeasurableSet s) :
    ∫⁻ a in s, ENNReal.ofReal (condCDF ρ a x) ∂ρ.fst = ρ (s ×ˢ Iic x) :=
  (isCondKernelCDF_condCDF ρ).setLIntegral () hs x


@[deprecated (since := "2024-06-29")]
alias set_lintegral_condCDF := setLIntegral_condCDF


theorem lintegral_condCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] (x : ℝ) :
    ∫⁻ a, ENNReal.ofReal (condCDF ρ a x) ∂ρ.fst = ρ (univ ×ˢ Iic x) :=
  (isCondKernelCDF_condCDF ρ).lintegral () x


theorem integrable_condCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] (x : ℝ) :
    Integrable (fun a ↦ condCDF ρ a x) ρ.fst :=
  (isCondKernelCDF_condCDF ρ).integrable () x


theorem setIntegral_condCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] (x : ℝ) {s : Set α}
    (hs : MeasurableSet s) : ∫ a in s, condCDF ρ a x ∂ρ.fst = (ρ (s ×ˢ Iic x)).toReal :=
  (isCondKernelCDF_condCDF ρ).setIntegral () hs x


@[deprecated (since := "2024-04-17")]
alias set_integral_condCDF := setIntegral_condCDF


theorem integral_condCDF (ρ : Measure (α × ℝ)) [IsFiniteMeasure ρ] (x : ℝ) :
    ∫ a, condCDF ρ a x ∂ρ.fst = (ρ (univ ×ˢ Iic x)).toReal :=
  (isCondKernelCDF_condCDF ρ).integral () x


theorem measure_condCDF_Iic (ρ : Measure (α × ℝ)) (a : α) (x : ℝ) :
    (condCDF ρ a).measure (Iic x) = ENNReal.ofReal (condCDF ρ a x) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    a : α
    x : Real
    ⊢ Eq ((ProbabilityTheory.condCDF ρ a).measure (Set.Iic x)) (ENNReal.ofReal (↑( …
  -/
  rw [← sub_zero (condCDF ρ a x)]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    a : α
    x : Real
    ⊢ Eq ((ProbabilityTheory.condCDF ρ a).measure (Set.Iic x)) (ENNReal.ofReal (HS …
  -/
  exact (condCDF ρ a).measure_Iic (tendsto_condCDF_atBot ρ a) _
  /-
    🎉 no goals
  -/


theorem measure_condCDF_univ (ρ : Measure (α × ℝ)) (a : α) : (condCDF ρ a).measure univ = 1 := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    a : α
    ⊢ Eq ((ProbabilityTheory.condCDF ρ a).measure Set.univ) 1
  -/
  rw [← ENNReal.ofReal_one, ← sub_zero (1 : ℝ)]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    a : α
    ⊢ Eq ((ProbabilityTheory.condCDF ρ a).measure Set.univ) (ENNReal.ofReal (HSub. …
  -/
  exact StieltjesFunction.measure_univ _ (tendsto_condCDF_atBot ρ a) (tendsto_condCDF_atTop ρ a)
  /-
    🎉 no goals
  -/


instance instIsProbabilityMeasureCondCDF (ρ : Measure (α × ℝ)) (a : α) :
    IsProbabilityMeasure (condCDF ρ a).measure :=
  ⟨measure_condCDF_univ ρ a⟩


/-- The function `a ↦ (condCDF ρ a).measure` is measurable. -/
theorem measurable_measure_condCDF (ρ : Measure (α × ℝ)) :
    Measurable fun a => (condCDF ρ a).measure :=
  .measure_of_isPiSystem_of_isProbabilityMeasure (borel_eq_generateFrom_Iic ℝ) isPiSystem_Iic <| by
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      ⊢ ∀ (s : Set Real), Membership.mem (Set.range Set.Iic) s → Measurable fun a => …
    -/
    simp_rw [forall_mem_range, measure_condCDF_Iic]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      ⊢ ∀ (i : Real), Measurable fun a => ENNReal.ofReal (↑(ProbabilityTheory.condCD …
    -/
    exact fun u ↦ (measurable_condCDF ρ u).ennreal_ofReal
    /-
      🎉 no goals
    -/


