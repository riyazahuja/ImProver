/-- Given a set `s`, return the continuous linear map `fun x => (μ s).toReal • x`. The extension
of that set function through `setToL1` gives the Bochner integral of L1 functions. -/
def weightedSMul {_ : MeasurableSpace α} (μ : Measure α) (s : Set α) : F →L[ℝ] F :=
  (μ s).toReal • ContinuousLinearMap.id ℝ F


theorem weightedSMul_apply {m : MeasurableSpace α} (μ : Measure α) (s : Set α) (x : F) :
                                                /-
                                                  α : Type u_1
                                                  F : Type u_3
                                                  inst✝¹ : NormedAddCommGroup F
                                                  inst✝ : NormedSpace Real F
                                                  m : MeasurableSpace α
                                                  μ : MeasureTheory.Measure α
                                                  s : Set α
                                                  x : F
                                                  ⊢ Eq ((MeasureTheory.weightedSMul μ s) x) (HSMul.hSMul (μ s).toReal x)
                                                -/
    weightedSMul μ s x = (μ s).toReal • x := by simp [weightedSMul]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem weightedSMul_zero_measure {m : MeasurableSpace α} :
                                                                 /-
                                                                   α : Type u_1
                                                                   F : Type u_3
                                                                   inst✝¹ : NormedAddCommGroup F
                                                                   inst✝ : NormedSpace Real F
                                                                   m : MeasurableSpace α
                                                                   ⊢ Eq (MeasureTheory.weightedSMul 0) 0
                                                                 -/
    weightedSMul (0 : Measure α) = (0 : Set α → F →L[ℝ] F) := by ext1; simp [weightedSMul]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem weightedSMul_empty {m : MeasurableSpace α} (μ : Measure α) :
                                             /-
                                               α : Type u_1
                                               F : Type u_3
                                               inst✝¹ : NormedAddCommGroup F
                                               inst✝ : NormedSpace Real F
                                               m : MeasurableSpace α
                                               μ : MeasureTheory.Measure α
                                               ⊢ Eq (MeasureTheory.weightedSMul μ EmptyCollection.emptyCollection) 0
                                             -/
    weightedSMul μ ∅ = (0 : F →L[ℝ] F) := by ext1 x; rw [weightedSMul_apply]; simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem weightedSMul_add_measure {m : MeasurableSpace α} (μ ν : Measure α) {s : Set α}
    (hμs : μ s ≠ ∞) (hνs : ν s ≠ ∞) :
    (weightedSMul (μ + ν) s : F →L[ℝ] F) = weightedSMul μ s + weightedSMul ν s := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hμs : Ne (μ s) Top.top
    hνs : Ne (ν s) Top.top
    ⊢ Eq (MeasureTheory.weightedSMul (HAdd.hAdd μ ν) s) (HAdd.hAdd (MeasureTheory. …
  -/
  ext1 x
  /-
    case h
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hμs : Ne (μ s) Top.top
    hνs : Ne (ν s) Top.top
    x : F
    ⊢ Eq ((MeasureTheory.weightedSMul (HAdd.hAdd μ ν) s) x) ((HAdd.hAdd (MeasureTh …
  -/
  push_cast
  /-
    case h
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hμs : Ne (μ s) Top.top
    hνs : Ne (ν s) Top.top
    x : F
    ⊢ Eq ((MeasureTheory.weightedSMul (HAdd.hAdd μ ν) s) x) (HAdd.hAdd (⇑(MeasureT …
  -/
  simp_rw [Pi.add_apply, weightedSMul_apply]
  /-
    case h
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hμs : Ne (μ s) Top.top
    hνs : Ne (ν s) Top.top
    x : F
    ⊢ Eq (HSMul.hSMul ((HAdd.hAdd μ ν) s).toReal x) (HAdd.hAdd (HSMul.hSMul (μ s). …
  -/
  push_cast
  /-
    case h
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hμs : Ne (μ s) Top.top
    hνs : Ne (ν s) Top.top
    x : F
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (⇑μ) (⇑ν) s).toReal x) (HAdd.hAdd (HSMul.hSMul (μ …
  -/
  rw [Pi.add_apply, ENNReal.toReal_add hμs hνs, add_smul]
  /-
    🎉 no goals
  -/


theorem weightedSMul_smul_measure {m : MeasurableSpace α} (μ : Measure α) (c : ℝ≥0∞) {s : Set α} :
    (weightedSMul (c • μ) s : F →L[ℝ] F) = c.toReal • weightedSMul μ s := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    s : Set α
    ⊢ Eq (MeasureTheory.weightedSMul (HSMul.hSMul c μ) s) (HSMul.hSMul c.toReal (M …
  -/
  ext1 x
  /-
    case h
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    s : Set α
    x : F
    ⊢ Eq ((MeasureTheory.weightedSMul (HSMul.hSMul c μ) s) x) ((HSMul.hSMul c.toRe …
  -/
  push_cast
  /-
    case h
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    s : Set α
    x : F
    ⊢ Eq ((MeasureTheory.weightedSMul (HSMul.hSMul c μ) s) x) (HSMul.hSMul c.toRea …
  -/
  simp_rw [Pi.smul_apply, weightedSMul_apply]
  /-
    case h
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    s : Set α
    x : F
    ⊢ Eq (HSMul.hSMul ((HSMul.hSMul c μ) s).toReal x) (HSMul.hSMul c.toReal (HSMul …
  -/
  push_cast
  /-
    case h
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    s : Set α
    x : F
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul c (⇑μ) s).toReal x) (HSMul.hSMul c.toReal (HSMu …
  -/
  simp_rw [Pi.smul_apply, smul_eq_mul, toReal_mul, smul_smul]
  /-
    🎉 no goals
  -/


theorem weightedSMul_congr (s t : Set α) (hst : μ s = μ t) :
    (weightedSMul μ s : F →L[ℝ] F) = weightedSMul μ t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hst : Eq (μ s) (μ t)
    ⊢ Eq (MeasureTheory.weightedSMul μ s) (MeasureTheory.weightedSMul μ t)
  -/
  ext1 x; simp_rw [weightedSMul_apply]; congr 2
                                        /-
                                          🎉 no goals
                                        -/


theorem weightedSMul_null {s : Set α} (h_zero : μ s = 0) : (weightedSMul μ s : F →L[ℝ] F) = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    h_zero : Eq (μ s) 0
    ⊢ Eq (MeasureTheory.weightedSMul μ s) 0
  -/
  ext1 x; rw [weightedSMul_apply, h_zero]; simp
                                           /-
                                             🎉 no goals
                                           -/


theorem weightedSMul_union' (s t : Set α) (ht : MeasurableSet t) (hs_finite : μ s ≠ ∞)
    (ht_finite : μ t ≠ ∞) (hdisj : Disjoint s t) :
    (weightedSMul μ (s ∪ t) : F →L[ℝ] F) = weightedSMul μ s + weightedSMul μ t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    ht : MeasurableSet t
    hs_finite : Ne (μ s) Top.top
    ht_finite : Ne (μ t) Top.top
    hdisj : Disjoint s t
    ⊢ Eq (MeasureTheory.weightedSMul μ (Union.union s t)) (HAdd.hAdd (MeasureTheor …
  -/
  ext1 x
  simp_rw [add_apply, weightedSMul_apply, measure_union hdisj ht,
    ENNReal.toReal_add hs_finite ht_finite, add_smul]


@[nolint unusedArguments]
theorem weightedSMul_union (s t : Set α) (_hs : MeasurableSet s) (ht : MeasurableSet t)
    (hs_finite : μ s ≠ ∞) (ht_finite : μ t ≠ ∞) (hdisj : Disjoint s t) :
    (weightedSMul μ (s ∪ t) : F →L[ℝ] F) = weightedSMul μ s + weightedSMul μ t :=
  weightedSMul_union' s t ht hs_finite ht_finite hdisj


theorem weightedSMul_smul [NormedField 𝕜] [NormedSpace 𝕜 F] [SMulCommClass ℝ 𝕜 F] (c : 𝕜)
    (s : Set α) (x : F) : weightedSMul μ s (c • x) = c • weightedSMul μ s x := by
  /-
    α : Type u_1
    F : Type u_3
    𝕜 : Type u_4
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : SMulCommClass Real 𝕜 F
    c : 𝕜
    s : Set α
    x : F
    ⊢ Eq ((MeasureTheory.weightedSMul μ s) (HSMul.hSMul c x)) (HSMul.hSMul c ((Mea …
  -/
  simp_rw [weightedSMul_apply, smul_comm]
  /-
    🎉 no goals
  -/


theorem norm_weightedSMul_le (s : Set α) : ‖(weightedSMul μ s : F →L[ℝ] F)‖ ≤ (μ s).toReal :=
  calc
    ‖(weightedSMul μ s : F →L[ℝ] F)‖ = ‖(μ s).toReal‖ * ‖ContinuousLinearMap.id ℝ F‖ :=
      norm_smul (μ s).toReal (ContinuousLinearMap.id ℝ F)
    _ ≤ ‖(μ s).toReal‖ :=
      ((mul_le_mul_of_nonneg_left norm_id_le (norm_nonneg _)).trans (mul_one _).le)
    _ = abs (μ s).toReal := Real.norm_eq_abs _
    _ = (μ s).toReal := abs_eq_self.mpr ENNReal.toReal_nonneg


theorem dominatedFinMeasAdditive_weightedSMul {_ : MeasurableSpace α} (μ : Measure α) :
    DominatedFinMeasAdditive μ (weightedSMul μ : Set α → F →L[ℝ] F) 1 :=
  ⟨weightedSMul_union, fun s _ _ => (norm_weightedSMul_le s).trans (one_mul _).symm.le⟩


theorem weightedSMul_nonneg (s : Set α) (x : ℝ) (hx : 0 ≤ x) : 0 ≤ weightedSMul μ s x := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    x : Real
    hx : LE.le 0 x
    ⊢ LE.le 0 ((MeasureTheory.weightedSMul μ s) x)
  -/
  simp only [weightedSMul, Algebra.id.smul_eq_mul, coe_smul', _root_.id, coe_id', Pi.smul_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    x : Real
    hx : LE.le 0 x
    ⊢ LE.le 0 (HMul.hMul (μ s).toReal x)
  -/
  exact mul_nonneg toReal_nonneg hx
  /-
    🎉 no goals
  -/


local infixr:25 " →ₛ " => SimpleFunc


/-- Positive part of a simple function. -/
def posPart (f : α →ₛ E) : α →ₛ E :=
  f.map fun b => max b 0


/-- Negative part of a simple function. -/
def negPart [Neg E] (f : α →ₛ E) : α →ₛ E :=
  posPart (-f)


theorem posPart_map_norm (f : α →ₛ ℝ) : (posPart f).map norm = posPart f := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α Real
    ⊢ Eq (MeasureTheory.SimpleFunc.map Norm.norm f.posPart) f.posPart
  -/
  ext; rw [map_apply, Real.norm_eq_abs, abs_of_nonneg]; exact le_max_right _ _
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem negPart_map_norm (f : α →ₛ ℝ) : (negPart f).map norm = negPart f := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α Real
    ⊢ Eq (MeasureTheory.SimpleFunc.map Norm.norm f.negPart) f.negPart
  -/
  rw [negPart]; exact posPart_map_norm _
                /-
                  🎉 no goals
                -/


theorem posPart_sub_negPart (f : α →ₛ ℝ) : f.posPart - f.negPart = f := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α Real
    ⊢ Eq (HSub.hSub f.posPart f.negPart) f
  -/
  simp only [posPart, negPart]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α Real
    ⊢ Eq (HSub.hSub (MeasureTheory.SimpleFunc.map (fun b => Max.max b 0) f) (Measu …
  -/
  ext a
  /-
    case H
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α Real
    a : α
    ⊢ Eq ((HSub.hSub (MeasureTheory.SimpleFunc.map (fun b => Max.max b 0) f) (Meas …
  -/
  rw [coe_sub]
  /-
    case H
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α Real
    a : α
    ⊢ Eq (HSub.hSub (⇑(MeasureTheory.SimpleFunc.map (fun b => Max.max b 0) f)) (⇑( …
  -/
  exact max_zero_sub_eq_self (f a)
  /-
    🎉 no goals
  -/


/-- Bochner integral of simple functions whose codomain is a real `NormedSpace`.
This is equal to `∑ x ∈ f.range, (μ (f ⁻¹' {x})).toReal • x` (see `integral_eq`). -/
def integral {_ : MeasurableSpace α} (μ : Measure α) (f : α →ₛ F) : F :=
  f.setToSimpleFunc (weightedSMul μ)


theorem integral_def {_ : MeasurableSpace α} (μ : Measure α) (f : α →ₛ F) :
    f.integral μ = f.setToSimpleFunc (weightedSMul μ) := rfl


theorem integral_eq {m : MeasurableSpace α} (μ : Measure α) (f : α →ₛ F) :
    f.integral μ = ∑ x ∈ f.range, (μ (f ⁻¹' {x})).toReal • x := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α F
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ f) (f.range.sum fun x => HSMul.hSMul …
  -/
  simp [integral, setToSimpleFunc, weightedSMul_apply]
  /-
    🎉 no goals
  -/


theorem integral_eq_sum_filter [DecidablePred fun x : F => x ≠ 0] {m : MeasurableSpace α}
    (f : α →ₛ F) (μ : Measure α) :
    f.integral μ = ∑ x ∈ {x ∈ f.range | x ≠ 0}, (μ (f ⁻¹' {x})).toReal • x := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : DecidablePred fun x => Ne x 0
    m : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α F
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ f) ((Finset.filter (fun x => Ne x 0) …
  -/
  simp_rw [integral_def, setToSimpleFunc_eq_sum_filter, weightedSMul_apply]
  /-
    🎉 no goals
  -/


/-- The Bochner integral is equal to a sum over any set that includes `f.range` (except `0`). -/
theorem integral_eq_sum_of_subset [DecidablePred fun x : F => x ≠ 0] {f : α →ₛ F} {s : Finset F}
    (hs : {x ∈ f.range | x ≠ 0} ⊆ s) :
    f.integral μ = ∑ x ∈ s, (μ (f ⁻¹' {x})).toReal • x := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Ne x 0
    f : MeasureTheory.SimpleFunc α F
    s : Finset F
    hs : HasSubset.Subset (Finset.filter (fun x => Ne x 0) f.range) s
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ f) (s.sum fun x => HSMul.hSMul (μ (S …
  -/
  rw [SimpleFunc.integral_eq_sum_filter, Finset.sum_subset hs]
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Ne x 0
    f : MeasureTheory.SimpleFunc α F
    s : Finset F
    hs : HasSubset.Subset (Finset.filter (fun x => Ne x 0) f.range) s
    ⊢ ∀ (x : F), Membership.mem s x → Not (Membership.mem (Finset.filter (fun x => …
  -/
  rintro x - hx; rw [Finset.mem_filter, not_and_or, Ne, Classical.not_not] at hx
  -- Porting note: reordered for clarity
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Ne x 0
    f : MeasureTheory.SimpleFunc α F
    s : Finset F
    hs : HasSubset.Subset (Finset.filter (fun x => Ne x 0) f.range) s
    x : F
    hx : Or (Not (Membership.mem f.range x)) (Eq x 0)
    ⊢ Eq (HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.singleton x))).toReal x) 0
  -/
  rcases hx.symm with (rfl | hx)
    /-
      case inl
      α : Type u_1
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : DecidablePred fun x => Ne x 0
      f : MeasureTheory.SimpleFunc α F
      s : Finset F
      hs : HasSubset.Subset (Finset.filter (fun x => Ne x 0) f.range) s
      hx : Or (Not (Membership.mem f.range 0)) (Eq 0 0)
      ⊢ Eq (HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.singleton 0))).toReal 0) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Ne x 0
    f : MeasureTheory.SimpleFunc α F
    s : Finset F
    hs : HasSubset.Subset (Finset.filter (fun x => Ne x 0) f.range) s
    x : F
    hx✝ : Or (Not (Membership.mem f.range x)) (Eq x 0)
    hx : Not (Membership.mem f.range x)
    ⊢ Eq (HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.singleton x))).toReal x) 0
  -/
  rw [SimpleFunc.mem_range] at hx
  -- Porting note: added
  /-
    case inr
    α : Type u_1
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Ne x 0
    f : MeasureTheory.SimpleFunc α F
    s : Finset F
    hs : HasSubset.Subset (Finset.filter (fun x => Ne x 0) f.range) s
    x : F
    hx✝ : Or (Not (Membership.mem f.range x)) (Eq x 0)
    hx : Not (Membership.mem (Set.range ⇑f) x)
    ⊢ Eq (HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.singleton x))).toReal x) 0
  -/
  simp only [Set.mem_range, not_exists] at hx
  /-
    case inr
    α : Type u_1
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : DecidablePred fun x => Ne x 0
    f : MeasureTheory.SimpleFunc α F
    s : Finset F
    hs : HasSubset.Subset (Finset.filter (fun x => Ne x 0) f.range) s
    x : F
    hx✝ : Or (Not (Membership.mem f.range x)) (Eq x 0)
    hx : ∀ (x_1 : α), Not (Eq (f x_1) x)
    ⊢ Eq (HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.singleton x))).toReal x) 0
  -/
                             /-
                               🎉 no goals
                             -/
  rw [preimage_eq_empty] <;> simp [Set.disjoint_singleton_left, hx]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem integral_const {m : MeasurableSpace α} (μ : Measure α) (y : F) :
    (const α y).integral μ = (μ univ).toReal • y := by
  classical
  calc
    (const α y).integral μ = ∑ z ∈ {y}, (μ (const α y ⁻¹' {z})).toReal • z :=
      integral_eq_sum_of_subset <| (filter_subset _ _).trans (range_const_subset _ _)
    _ = (μ univ).toReal • y := by simp [Set.preimage] -- Porting note: added `Set.preimage`


@[simp]
theorem integral_piecewise_zero {m : MeasurableSpace α} (f : α →ₛ F) (μ : Measure α) {s : Set α}
    (hs : MeasurableSet s) : (piecewise s hs f 0).integral μ = f.integral (μ.restrict s) := by
  classical
  refine (integral_eq_sum_of_subset ?_).trans
      ((sum_congr rfl fun y hy => ?_).trans (integral_eq_sum_filter _ _).symm)
  · intro y hy
    simp only [mem_filter, mem_range, coe_piecewise, coe_zero, piecewise_eq_indicator,
      mem_range_indicator] at *
    rcases hy with ⟨⟨rfl, -⟩ | ⟨x, -, rfl⟩, h₀⟩
    exacts [(h₀ rfl).elim, ⟨Set.mem_range_self _, h₀⟩]
  · dsimp
    rw [Set.piecewise_eq_indicator, indicator_preimage_of_not_mem,
      Measure.restrict_apply (f.measurableSet_preimage _)]
    exact fun h₀ => (mem_filter.1 hy).2 (Eq.symm h₀)


/-- Calculate the integral of `g ∘ f : α →ₛ F`, where `f` is an integrable function from `α` to `E`
    and `g` is a function from `E` to `F`. We require `g 0 = 0` so that `g ∘ f` is integrable. -/
theorem map_integral (f : α →ₛ E) (g : E → F) (hf : Integrable f μ) (hg : g 0 = 0) :
    (f.map g).integral μ = ∑ x ∈ f.range, ENNReal.toReal (μ (f ⁻¹' {x})) • g x :=
  map_setToSimpleFunc _ weightedSMul_union hf hg


/-- `SimpleFunc.integral` and `SimpleFunc.lintegral` agree when the integrand has type
    `α →ₛ ℝ≥0∞`. But since `ℝ≥0∞` is not a `NormedSpace`, we need some form of coercion.
    See `integral_eq_lintegral` for a simpler version. -/
theorem integral_eq_lintegral' {f : α →ₛ E} {g : E → ℝ≥0∞} (hf : Integrable f μ) (hg0 : g 0 = 0)
    (ht : ∀ b, g b ≠ ∞) :
    (f.map (ENNReal.toReal ∘ g)).integral μ = ENNReal.toReal (∫⁻ a, g (f a) ∂μ) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    g : E → ENNReal
    hf : MeasureTheory.Integrable (⇑f) μ
    hg0 : Eq (g 0) 0
    ht : ∀ (b : E), Ne (g b) Top.top
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ (MeasureTheory.SimpleFunc.map (Funct …
  -/
  have hf' : f.FinMeasSupp μ := integrable_iff_finMeasSupp.1 hf
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    g : E → ENNReal
    hf : MeasureTheory.Integrable (⇑f) μ
    hg0 : Eq (g 0) 0
    ht : ∀ (b : E), Ne (g b) Top.top
    hf' : f.FinMeasSupp μ
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ (MeasureTheory.SimpleFunc.map (Funct …
  -/
  simp only [← map_apply g f, lintegral_eq_lintegral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    g : E → ENNReal
    hf : MeasureTheory.Integrable (⇑f) μ
    hg0 : Eq (g 0) 0
    ht : ∀ (b : E), Ne (g b) Top.top
    hf' : f.FinMeasSupp μ
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ (MeasureTheory.SimpleFunc.map (Funct …
  -/
  rw [map_integral f _ hf, map_lintegral, ENNReal.toReal_sum]
    /-
      α : Type u_1
      E : Type u_2
      inst✝ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α E
      g : E → ENNReal
      hf : MeasureTheory.Integrable (⇑f) μ
      hg0 : Eq (g 0) 0
      ht : ∀ (b : E), Ne (g b) Top.top
      hf' : f.FinMeasSupp μ
      ⊢ Eq (f.range.sum fun x => HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.single …
    -/
  · refine Finset.sum_congr rfl fun b _ => ?_
    -- Porting note: added `Function.comp_apply`
    /-
      α : Type u_1
      E : Type u_2
      inst✝ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α E
      g : E → ENNReal
      hf : MeasureTheory.Integrable (⇑f) μ
      hg0 : Eq (g 0) 0
      ht : ∀ (b : E), Ne (g b) Top.top
      hf' : f.FinMeasSupp μ
      b : E
      x✝ : Membership.mem f.range b
      ⊢ Eq (HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.singleton b))).toReal (Func …
    -/
    rw [smul_eq_mul, toReal_mul, mul_comm, Function.comp_apply]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      E : Type u_2
      inst✝ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α E
      g : E → ENNReal
      hf : MeasureTheory.Integrable (⇑f) μ
      hg0 : Eq (g 0) 0
      ht : ∀ (b : E), Ne (g b) Top.top
      hf' : f.FinMeasSupp μ
      ⊢ ∀ (a : E), Membership.mem f.range a → Ne (HMul.hMul (g a) (μ (Set.preimage ( …
    -/
  · rintro a -
    /-
      α : Type u_1
      E : Type u_2
      inst✝ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α E
      g : E → ENNReal
      hf : MeasureTheory.Integrable (⇑f) μ
      hg0 : Eq (g 0) 0
      ht : ∀ (b : E), Ne (g b) Top.top
      hf' : f.FinMeasSupp μ
      a : E
      ⊢ Ne (HMul.hMul (g a) (μ (Set.preimage (⇑f) (Singleton.singleton a)))) Top.top
    -/
    by_cases a0 : a = 0
      /-
        case pos
        α : Type u_1
        E : Type u_2
        inst✝ : NormedAddCommGroup E
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : MeasureTheory.SimpleFunc α E
        g : E → ENNReal
        hf : MeasureTheory.Integrable (⇑f) μ
        hg0 : Eq (g 0) 0
        ht : ∀ (b : E), Ne (g b) Top.top
        hf' : f.FinMeasSupp μ
        a : E
        a0 : Eq a 0
        ⊢ Ne (HMul.hMul (g a) (μ (Set.preimage (⇑f) (Singleton.singleton a)))) Top.top
      -/
    · rw [a0, hg0, zero_mul]; exact WithTop.zero_ne_top
                              /-
                                🎉 no goals
                              -/
      /-
        case neg
        α : Type u_1
        E : Type u_2
        inst✝ : NormedAddCommGroup E
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : MeasureTheory.SimpleFunc α E
        g : E → ENNReal
        hf : MeasureTheory.Integrable (⇑f) μ
        hg0 : Eq (g 0) 0
        ht : ∀ (b : E), Ne (g b) Top.top
        hf' : f.FinMeasSupp μ
        a : E
        a0 : Not (Eq a 0)
        ⊢ Ne (HMul.hMul (g a) (μ (Set.preimage (⇑f) (Singleton.singleton a)))) Top.top
      -/
    · apply mul_ne_top (ht a) (hf'.meas_preimage_singleton_ne_zero a0).ne
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      E : Type u_2
      inst✝ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α E
      g : E → ENNReal
      hf : MeasureTheory.Integrable (⇑f) μ
      hg0 : Eq (g 0) 0
      ht : ∀ (b : E), Ne (g b) Top.top
      hf' : f.FinMeasSupp μ
      ⊢ Eq (Function.comp ENNReal.toReal g 0) 0
    -/
  · simp [hg0]
    /-
      🎉 no goals
    -/


theorem integral_congr {f g : α →ₛ E} (hf : Integrable f μ) (h : f =ᵐ[μ] g) :
    f.integral μ = g.integral μ :=
  setToSimpleFunc_congr (weightedSMul μ) (fun _ _ => weightedSMul_null) weightedSMul_union hf h


/-- `SimpleFunc.bintegral` and `SimpleFunc.integral` agree when the integrand has type
    `α →ₛ ℝ≥0∞`. But since `ℝ≥0∞` is not a `NormedSpace`, we need some form of coercion. -/
theorem integral_eq_lintegral {f : α →ₛ ℝ} (hf : Integrable f μ) (h_pos : 0 ≤ᵐ[μ] f) :
    f.integral μ = ENNReal.toReal (∫⁻ a, ENNReal.ofReal (f a) ∂μ) := by
  have : f =ᵐ[μ] f.map (ENNReal.toReal ∘ ENNReal.ofReal) :=
    h_pos.mono fun a h => (ENNReal.toReal_ofReal h).symm
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α Real
    hf : MeasureTheory.Integrable (⇑f) μ
    h_pos : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
    this : (MeasureTheory.ae μ).EventuallyEq ⇑f ⇑(MeasureTheory.SimpleFunc.map (Fu …
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ f) (MeasureTheory.lintegral μ fun a  …
  -/
  rw [← integral_eq_lintegral' hf]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α Real
    hf : MeasureTheory.Integrable (⇑f) μ
    h_pos : (MeasureTheory.ae μ).EventuallyLE 0 ⇑f
    this : (MeasureTheory.ae μ).EventuallyEq ⇑f ⇑(MeasureTheory.SimpleFunc.map (Fu …
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ f) (MeasureTheory.SimpleFunc.integra …
  -/
  exacts [integral_congr hf this, ENNReal.ofReal_zero, fun b => ENNReal.ofReal_ne_top]
  /-
    🎉 no goals
  -/


theorem integral_add {f g : α →ₛ E} (hf : Integrable f μ) (hg : Integrable g μ) :
    integral μ (f + g) = integral μ f + integral μ g :=
  setToSimpleFunc_add _ weightedSMul_union hf hg


theorem integral_neg {f : α →ₛ E} (hf : Integrable f μ) : integral μ (-f) = -integral μ f :=
  setToSimpleFunc_neg _ weightedSMul_union hf


theorem integral_sub {f g : α →ₛ E} (hf : Integrable f μ) (hg : Integrable g μ) :
    integral μ (f - g) = integral μ f - integral μ g :=
  setToSimpleFunc_sub _ weightedSMul_union hf hg


theorem integral_smul (c : 𝕜) {f : α →ₛ E} (hf : Integrable f μ) :
    integral μ (c • f) = c • integral μ f :=
  setToSimpleFunc_smul _ weightedSMul_union weightedSMul_smul c hf


theorem norm_setToSimpleFunc_le_integral_norm (T : Set α → E →L[ℝ] F) {C : ℝ}
    (hT_norm : ∀ s, MeasurableSet s → μ s < ∞ → ‖T s‖ ≤ C * (μ s).toReal) {f : α →ₛ E}
    (hf : Integrable f μ) : ‖f.setToSimpleFunc T‖ ≤ C * (f.map norm).integral μ :=
  calc
    ‖f.setToSimpleFunc T‖ ≤ C * ∑ x ∈ f.range, ENNReal.toReal (μ (f ⁻¹' {x})) * ‖x‖ :=
      norm_setToSimpleFunc_le_sum_mul_norm_of_integrable T hT_norm f hf
    _ = C * (f.map norm).integral μ := by
      /-
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedSpace Real E
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        C : Real
        hT_norm : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (Norm.n …
        f : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        ⊢ Eq (HMul.hMul C (f.range.sum fun x => HMul.hMul (μ (Set.preimage (⇑f) (Singl …
      -/
      rw [map_integral f norm hf norm_zero]; simp_rw [smul_eq_mul]
                                             /-
                                               🎉 no goals
                                             -/


theorem norm_integral_le_integral_norm (f : α →ₛ E) (hf : Integrable f μ) :
    ‖f.integral μ‖ ≤ (f.map norm).integral μ := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    ⊢ LE.le (Norm.norm (MeasureTheory.SimpleFunc.integral μ f)) (MeasureTheory.Sim …
  -/
  refine (norm_setToSimpleFunc_le_integral_norm _ (fun s _ _ => ?_) hf).trans (one_mul _).le
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    s : Set α
    x✝¹ : MeasurableSet s
    x✝ : LT.lt (μ s) Top.top
    ⊢ LE.le (Norm.norm (MeasureTheory.weightedSMul μ s)) (HMul.hMul 1 (μ s).toReal)
  -/
  exact (norm_weightedSMul_le s).trans (one_mul _).symm.le
  /-
    🎉 no goals
  -/


theorem integral_add_measure {ν} (f : α →ₛ E) (hf : Integrable f (μ + ν)) :
    f.integral (μ + ν) = f.integral μ + f.integral ν := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    ν : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) (HAdd.hAdd μ ν)
    ⊢ Eq (MeasureTheory.SimpleFunc.integral (HAdd.hAdd μ ν) f) (HAdd.hAdd (Measure …
  -/
  simp_rw [integral_def]
  refine setToSimpleFunc_add_left'
    (weightedSMul μ) (weightedSMul ν) (weightedSMul (μ + ν)) (fun s _ hμνs => ?_) hf
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    ν : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) (HAdd.hAdd μ ν)
    s : Set α
    x✝ : MeasurableSet s
    hμνs : LT.lt ((HAdd.hAdd μ ν) s) Top.top
    ⊢ Eq (MeasureTheory.weightedSMul (HAdd.hAdd μ ν) s) (HAdd.hAdd (MeasureTheory. …
  -/
  rw [lt_top_iff_ne_top, Measure.coe_add, Pi.add_apply, ENNReal.add_ne_top] at hμνs
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    ν : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) (HAdd.hAdd μ ν)
    s : Set α
    x✝ : MeasurableSet s
    hμνs : And (Ne (μ s) Top.top) (Ne (ν s) Top.top)
    ⊢ Eq (MeasureTheory.weightedSMul (HAdd.hAdd μ ν) s) (HAdd.hAdd (MeasureTheory. …
  -/
  rw [weightedSMul_add_measure _ _ hμνs.1 hμνs.2]
  /-
    🎉 no goals
  -/


theorem norm_eq_integral (f : α →₁ₛ[μ] E) : ‖f‖ = ((toSimpleFunc f).map norm).integral μ := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (Norm.norm f) (MeasureTheory.SimpleFunc.integral μ (MeasureTheory.SimpleF …
  -/
  rw [norm_eq_sum_mul f, (toSimpleFunc f).map_integral norm (SimpleFunc.integrable f) norm_zero]
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).range.sum fun x => HMul.hMu …
  -/
  simp_rw [smul_eq_mul]
  /-
    🎉 no goals
  -/


/-- Positive part of a simple function in L1 space. -/
nonrec def posPart (f : α →₁ₛ[μ] ℝ) : α →₁ₛ[μ] ℝ :=
  ⟨Lp.posPart (f : α →₁[μ] ℝ), by
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      𝕜 : Type u_4
      inst✝ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ⊢ Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) (MeasureTheory.Lp.posP …
    -/
    rcases f with ⟨f, s, hsf⟩
    /-
      case mk.intro
      α : Type u_1
      E : Type u_2
      F : Type u_3
      𝕜 : Type u_4
      inst✝ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x
      s : MeasureTheory.SimpleFunc α Real
      hsf : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
      ⊢ Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) (MeasureTheory.Lp.posP …
    -/
    use s.posPart
    simp only [Subtype.coe_mk, Lp.coe_posPart, ← hsf, AEEqFun.posPart_mk,
      SimpleFunc.coe_map, mk_eq_mk]
    -- Porting note: added
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      𝕜 : Type u_4
      inst✝ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x
      s : MeasureTheory.SimpleFunc α Real
      hsf : Eq (MeasureTheory.AEEqFun.mk ⇑s ⋯) ↑f
      ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑s.posPart fun x => Max.max (s x) 0
    -/
    simp [SimpleFunc.posPart, Function.comp_def, EventuallyEq.rfl] ⟩
    /-
      🎉 no goals
    -/


/-- Negative part of a simple function in L1 space. -/
def negPart (f : α →₁ₛ[μ] ℝ) : α →₁ₛ[μ] ℝ :=
  posPart (-f)


@[norm_cast]
theorem coe_posPart (f : α →₁ₛ[μ] ℝ) : (posPart f : α →₁[μ] ℝ) = Lp.posPart (f : α →₁[μ] ℝ) := rfl


@[norm_cast]
theorem coe_negPart (f : α →₁ₛ[μ] ℝ) : (negPart f : α →₁[μ] ℝ) = Lp.negPart (f : α →₁[μ] ℝ) := rfl


/-- The Bochner integral over simple functions in L1 space. -/
def integral (f : α →₁ₛ[μ] E) : E :=
  (toSimpleFunc f).integral μ


theorem integral_eq_integral (f : α →₁ₛ[μ] E) : integral f = (toSimpleFunc f).integral μ := rfl


nonrec theorem integral_eq_lintegral {f : α →₁ₛ[μ] ℝ} (h_pos : 0 ≤ᵐ[μ] toSimpleFunc f) :
    integral f = ENNReal.toReal (∫⁻ a, ENNReal.ofReal ((toSimpleFunc f) a) ∂μ) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    h_pos : (MeasureTheory.ae μ).EventuallyLE 0 ⇑(MeasureTheory.Lp.simpleFunc.toSi …
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.integral f) (MeasureTheory.lintegral μ fun a …
  -/
  rw [integral, SimpleFunc.integral_eq_lintegral (SimpleFunc.integrable f) h_pos]
  /-
    🎉 no goals
  -/


theorem integral_eq_setToL1S (f : α →₁ₛ[μ] E) : integral f = setToL1S (weightedSMul μ) f := rfl


nonrec theorem integral_congr {f g : α →₁ₛ[μ] E} (h : toSimpleFunc f =ᵐ[μ] toSimpleFunc g) :
    integral f = integral g :=
  SimpleFunc.integral_congr (SimpleFunc.integrable f) h


theorem integral_add (f g : α →₁ₛ[μ] E) : integral (f + g) = integral f + integral g :=
  setToL1S_add _ (fun _ _ => weightedSMul_null) weightedSMul_union _ _


theorem integral_smul (c : 𝕜) (f : α →₁ₛ[μ] E) : integral (c • f) = c • integral f :=
  setToL1S_smul _ (fun _ _ => weightedSMul_null) weightedSMul_union weightedSMul_smul c f


theorem norm_integral_le_norm (f : α →₁ₛ[μ] E) : ‖integral f‖ ≤ ‖f‖ := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ LE.le (Norm.norm (MeasureTheory.L1.SimpleFunc.integral f)) (Norm.norm f)
  -/
  rw [integral, norm_eq_integral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ LE.le (Norm.norm (MeasureTheory.SimpleFunc.integral μ (MeasureTheory.Lp.simp …
  -/
  exact (toSimpleFunc f).norm_integral_le_integral_norm (SimpleFunc.integrable f)
  /-
    🎉 no goals
  -/


/-- The Bochner integral over simple functions in L1 space as a continuous linear map. -/
def integralCLM' : (α →₁ₛ[μ] E) →L[𝕜] E :=
  LinearMap.mkContinuous ⟨⟨integral, integral_add⟩, integral_smul⟩ 1 fun f =>
                                             /-
                                               α : Type u_1
                                               E : Type u_2
                                               F : Type u_3
                                               𝕜 : Type u_4
                                               inst✝⁴ : NormedAddCommGroup E
                                               m : MeasurableSpace α
                                               μ : MeasureTheory.Measure α
                                               inst✝³ : NormedField 𝕜
                                               inst✝² : NormedSpace 𝕜 E
                                               inst✝¹ : NormedSpace Real E
                                               inst✝ : SMulCommClass Real 𝕜 E
                                               f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
                                               ⊢ LE.le (Norm.norm f) (HMul.hMul 1 (Norm.norm f))
                                             -/
    le_trans (norm_integral_le_norm _) <| by rw [one_mul]
                                             /-
                                               🎉 no goals
                                             -/


/-- The Bochner integral over simple functions in L1 space as a continuous linear map over ℝ. -/
def integralCLM : (α →₁ₛ[μ] E) →L[ℝ] E :=
  integralCLM' α E ℝ μ


local notation "Integral" => integralCLM α E μ


theorem norm_Integral_le_one : ‖Integral‖ ≤ 1 :=
  -- Porting note: Old proof was `LinearMap.mkContinuous_norm_le _ zero_le_one _`
  LinearMap.mkContinuous_norm_le _ zero_le_one (fun f => by
    /-
      α : Type u_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedSpace Real E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
      ⊢ LE.le (Norm.norm ({ toFun := MeasureTheory.L1.SimpleFunc.integral, map_add'  …
    -/
    rw [one_mul]
    /-
      α : Type u_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedSpace Real E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
      ⊢ LE.le (Norm.norm ({ toFun := MeasureTheory.L1.SimpleFunc.integral, map_add'  …
    -/
    exact norm_integral_le_norm f)
    /-
      🎉 no goals
    -/


theorem posPart_toSimpleFunc (f : α →₁ₛ[μ] ℝ) :
    toSimpleFunc (posPart f) =ᵐ[μ] (toSimpleFunc f).posPart := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  have eq : ∀ a, (toSimpleFunc f).posPart a = max ((toSimpleFunc f) a) 0 := fun a => rfl
  have ae_eq : ∀ᵐ a ∂μ, toSimpleFunc (posPart f) a = max ((toSimpleFunc f) a) 0 := by
    filter_upwards [toSimpleFunc_eq_toFun (posPart f), Lp.coeFn_posPart (f : α →₁[μ] ℝ),
      toSimpleFunc_eq_toFun f] with _ _ h₂ h₃
    convert h₂ using 1
    -- Porting note: added
    rw [h₃]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    eq : ∀ (a : α), Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).posPart a) (M …
    ae_eq : Filter.Eventually (fun a => Eq ((MeasureTheory.Lp.simpleFunc.toSimpleF …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  refine ae_eq.mono fun a h => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    eq : ∀ (a : α), Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).posPart a) (M …
    ae_eq : Filter.Eventually (fun a => Eq ((MeasureTheory.Lp.simpleFunc.toSimpleF …
    a : α
    h : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (MeasureTheory.L1.SimpleFunc …
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (MeasureTheory.L1.SimpleFunc.p …
  -/
  rw [h, eq]
  /-
    🎉 no goals
  -/


theorem negPart_toSimpleFunc (f : α →₁ₛ[μ] ℝ) :
    toSimpleFunc (negPart f) =ᵐ[μ] (toSimpleFunc f).negPart := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  rw [SimpleFunc.negPart, MeasureTheory.SimpleFunc.negPart]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  filter_upwards [posPart_toSimpleFunc (-f), neg_toSimpleFunc f]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    ⊢ ∀ (a : α), Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (MeasureTheory.L1.S …
  -/
  intro a h₁ h₂
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    a : α
    h₁ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (MeasureTheory.L1.SimpleFun …
    h₂ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (Neg.neg f)) a) (Neg.neg (⇑ …
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (MeasureTheory.L1.SimpleFunc.p …
  -/
  rw [h₁]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    a : α
    h₁ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (MeasureTheory.L1.SimpleFun …
    h₂ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (Neg.neg f)) a) (Neg.neg (⇑ …
    ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (Neg.neg f)).posPart a) ((Neg. …
  -/
  show max _ _ = max _ _
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    a : α
    h₁ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (MeasureTheory.L1.SimpleFun …
    h₂ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (Neg.neg f)) a) (Neg.neg (⇑ …
    ⊢ Eq (Max.max ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (Neg.neg f)) a) 0) (M …
  -/
  rw [h₂]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    a : α
    h₁ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (MeasureTheory.L1.SimpleFun …
    h₂ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc (Neg.neg f)) a) (Neg.neg (⇑ …
    ⊢ Eq (Max.max (Neg.neg (⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc f)) a) 0) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem integral_eq_norm_posPart_sub (f : α →₁ₛ[μ] ℝ) : integral f = ‖posPart f‖ - ‖negPart f‖ := by
  -- Convert things in `L¹` to their `SimpleFunc` counterpart
  have ae_eq₁ : (toSimpleFunc f).posPart =ᵐ[μ] (toSimpleFunc (posPart f)).map norm := by
    filter_upwards [posPart_toSimpleFunc f] with _ h
    rw [SimpleFunc.map_apply, h]
    conv_lhs => rw [← SimpleFunc.posPart_map_norm, SimpleFunc.map_apply]
  -- Convert things in `L¹` to their `SimpleFunc` counterpart
  have ae_eq₂ : (toSimpleFunc f).negPart =ᵐ[μ] (toSimpleFunc (negPart f)).map norm := by
    filter_upwards [negPart_toSimpleFunc f] with _ h
    rw [SimpleFunc.map_apply, h]
    conv_lhs => rw [← SimpleFunc.negPart_map_norm, SimpleFunc.map_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
    ae_eq₁ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
    ae_eq₂ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.integral f) (HSub.hSub (Norm.norm (MeasureTh …
  -/
  rw [integral, norm_eq_integral, norm_eq_integral, ← SimpleFunc.integral_sub]
  · show (toSimpleFunc f).integral μ =
      ((toSimpleFunc (posPart f)).map norm - (toSimpleFunc (negPart f)).map norm).integral μ
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ae_eq₁ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ae_eq₂ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ⊢ Eq (MeasureTheory.SimpleFunc.integral μ (MeasureTheory.Lp.simpleFunc.toSimpl …
    -/
    apply MeasureTheory.SimpleFunc.integral_congr (SimpleFunc.integrable f)
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ae_eq₁ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ae_eq₂ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
    -/
    filter_upwards [ae_eq₁, ae_eq₂] with _ h₁ h₂
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ae_eq₁ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ae_eq₂ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      a✝ : α
      h₁ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).posPart a✝) ((MeasureThe …
      h₂ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).negPart a✝) ((MeasureThe …
      ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f) a✝) ((HSub.hSub (MeasureThe …
    -/
    rw [SimpleFunc.sub_apply, ← h₁, ← h₂]
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ae_eq₁ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ae_eq₂ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      a✝ : α
      h₁ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).posPart a✝) ((MeasureThe …
      h₂ : Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).negPart a✝) ((MeasureThe …
      ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f) a✝) (HSub.hSub ((MeasureThe …
    -/
    exact DFunLike.congr_fun (toSimpleFunc f).posPart_sub_negPart.symm _
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ae_eq₁ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ae_eq₂ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ⊢ MeasureTheory.Integrable (⇑(MeasureTheory.SimpleFunc.map Norm.norm (MeasureT …
    -/
  · exact (SimpleFunc.integrable f).pos_part.congr ae_eq₁
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ae_eq₁ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ae_eq₂ : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSim …
      ⊢ MeasureTheory.Integrable (⇑(MeasureTheory.SimpleFunc.map Norm.norm (MeasureT …
    -/
  · exact (SimpleFunc.integrable f).neg_part.congr ae_eq₂
    /-
      🎉 no goals
    -/


local notation "Integral" => @integralCLM α E _ _ _ _ _ μ _


/-- The Bochner integral in L1 space as a continuous linear map. -/
nonrec def integralCLM' : (α →₁[μ] E) →L[𝕜] E :=
  (integralCLM' α E 𝕜 μ).extend (coeToLp α E 𝕜) (simpleFunc.denseRange one_ne_top)
    simpleFunc.isUniformInducing


/-- The Bochner integral in L1 space as a continuous linear map over ℝ. -/
def integralCLM : (α →₁[μ] E) →L[ℝ] E :=
  integralCLM' ℝ

-- Porting note: added `(E := E)` in several places below.

/-- The Bochner integral in L1 space -/
irreducible_def integral : (α →₁[μ] E) → E :=
  integralCLM (E := E)


theorem integral_eq (f : α →₁[μ] E) : integral f = integralCLM (E := E) f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integral f) (MeasureTheory.L1.integralCLM f)
  -/
  simp only [integral]
  /-
    🎉 no goals
  -/


theorem integral_eq_setToL1 (f : α →₁[μ] E) :
    integral f = setToL1 (E := E) (dominatedFinMeasAdditive_weightedSMul μ) f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integral f) ((MeasureTheory.L1.setToL1 ⋯) f)
  -/
  simp only [integral]; rfl
                        /-
                          🎉 no goals
                        -/


@[norm_cast]
theorem SimpleFunc.integral_L1_eq_integral (f : α →₁ₛ[μ] E) :
    L1.integral (f : α →₁[μ] E) = SimpleFunc.integral f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integral ↑f) (MeasureTheory.L1.SimpleFunc.integral f)
  -/
  simp only [integral, L1.integral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integralCLM ↑f) (MeasureTheory.SimpleFunc.integral μ (M …
  -/
  exact setToL1_eq_setToL1SCLM (dominatedFinMeasAdditive_weightedSMul μ) f
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_zero : integral (0 : α →₁[μ] E) = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    ⊢ Eq (MeasureTheory.L1.integral 0) 0
  -/
  simp only [integral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    ⊢ Eq (MeasureTheory.L1.integralCLM 0) 0
  -/
  exact map_zero integralCLM
  /-
    🎉 no goals
  -/


@[integral_simps]
theorem integral_add (f g : α →₁[μ] E) : integral (f + g) = integral f + integral g := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integral (HAdd.hAdd f g)) (HAdd.hAdd (MeasureTheory.L1. …
  -/
  simp only [integral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integralCLM (HAdd.hAdd f g)) (HAdd.hAdd (MeasureTheory. …
  -/
  exact map_add integralCLM f g
  /-
    🎉 no goals
  -/


@[integral_simps]
theorem integral_neg (f : α →₁[μ] E) : integral (-f) = -integral f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integral (Neg.neg f)) (Neg.neg (MeasureTheory.L1.integr …
  -/
  simp only [integral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integralCLM (Neg.neg f)) (Neg.neg (MeasureTheory.L1.int …
  -/
  exact map_neg integralCLM f
  /-
    🎉 no goals
  -/


@[integral_simps]
theorem integral_sub (f g : α →₁[μ] E) : integral (f - g) = integral f - integral g := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integral (HSub.hSub f g)) (HSub.hSub (MeasureTheory.L1. …
  -/
  simp only [integral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integralCLM (HSub.hSub f g)) (HSub.hSub (MeasureTheory. …
  -/
  exact map_sub integralCLM f g
  /-
    🎉 no goals
  -/


@[integral_simps]
theorem integral_smul (c : 𝕜) (f : α →₁[μ] E) : integral (c • f) = c • integral f := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝⁵ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : CompleteSpace E
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integral (HSMul.hSMul c f)) (HSMul.hSMul c (MeasureTheo …
  -/
  simp only [integral]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝⁵ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : CompleteSpace E
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integralCLM (HSMul.hSMul c f)) (HSMul.hSMul c (MeasureT …
  -/
  show (integralCLM' (E := E) 𝕜) (c • f) = c • (integralCLM' (E := E) 𝕜) f
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝⁵ : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : CompleteSpace E
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.integralCLM' 𝕜) (HSMul.hSMul c f)) (HSMul.hSMul c ((Me …
  -/
  exact _root_.map_smul (integralCLM' (E := E) 𝕜) c f
  /-
    🎉 no goals
  -/


local notation "Integral" => @integralCLM α E _ _ μ _ _


local notation "sIntegral" => @SimpleFunc.integralCLM α E _ _ μ _


theorem norm_Integral_le_one : ‖integralCLM (α := α) (E := E) (μ := μ)‖ ≤ 1 :=
  norm_setToL1_le (dominatedFinMeasAdditive_weightedSMul μ) zero_le_one


theorem nnnorm_Integral_le_one : ‖integralCLM (α := α) (E := E) (μ := μ)‖₊ ≤ 1 :=
  norm_Integral_le_one


theorem norm_integral_le (f : α →₁[μ] E) : ‖integral f‖ ≤ ‖f‖ :=
  calc
                                                  /-
                                                    α : Type u_1
                                                    E : Type u_2
                                                    inst✝² : NormedAddCommGroup E
                                                    m : MeasurableSpace α
                                                    μ : MeasureTheory.Measure α
                                                    inst✝¹ : NormedSpace Real E
                                                    inst✝ : CompleteSpace E
                                                    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
                                                    ⊢ Eq (Norm.norm (MeasureTheory.L1.integral f)) (Norm.norm (MeasureTheory.L1.in …
                                                  -/
    ‖integral f‖ = ‖integralCLM (E := E) f‖ := by simp only [integral]
                                                  /-
                                                    🎉 no goals
                                                  -/
    _ ≤ ‖integralCLM (α := α) (E := E) (μ := μ)‖ * ‖f‖ := le_opNorm _ _
    _ ≤ 1 * ‖f‖ := mul_le_mul_of_nonneg_right norm_Integral_le_one <| norm_nonneg _
    _ = ‖f‖ := one_mul _


theorem nnnorm_integral_le (f : α →₁[μ] E) : ‖integral f‖₊ ≤ ‖f‖₊ :=
  norm_integral_le f


@[continuity]
theorem continuous_integral : Continuous fun f : α →₁[μ] E => integral f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    ⊢ Continuous fun f => MeasureTheory.L1.integral f
  -/
  simp only [integral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    ⊢ Continuous fun f => MeasureTheory.L1.integralCLM f
  -/
  exact L1.integralCLM.continuous
  /-
    🎉 no goals
  -/


theorem integral_eq_norm_posPart_sub (f : α →₁[μ] ℝ) :
    integral f = ‖Lp.posPart f‖ - ‖Lp.negPart f‖ := by
  -- Use `isClosed_property` and `isClosed_eq`
  refine @isClosed_property _ _ _ ((↑) : (α →₁ₛ[μ] ℝ) → α →₁[μ] ℝ)
      (fun f : α →₁[μ] ℝ => integral f = ‖Lp.posPart f‖ - ‖Lp.negPart f‖)
      (simpleFunc.denseRange one_ne_top) (isClosed_eq ?_ ?_) ?_ f
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x
      ⊢ Continuous MeasureTheory.L1.integral
    -/
  · simp only [integral]
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x
      ⊢ Continuous ⇑MeasureTheory.L1.integralCLM
    -/
    exact cont _
    /-
      🎉 no goals
    -/
  · refine Continuous.sub (continuous_norm.comp Lp.continuous_posPart)
      (continuous_norm.comp Lp.continuous_negPart)
  -- Show that the property holds for all simple functions in the `L¹` space.
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x
      ⊢ ∀ (a : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ …
    -/
  · intro s
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x
      s : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ⊢ Eq (MeasureTheory.L1.integral ↑s) (HSub.hSub (Norm.norm (MeasureTheory.Lp.po …
    -/
    norm_cast
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x
      s : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc Real 1 μ) x
      ⊢ Eq (MeasureTheory.L1.SimpleFunc.integral s) (HSub.hSub (Norm.norm (MeasureTh …
    -/
    exact SimpleFunc.integral_eq_norm_posPart_sub _
    /-
      🎉 no goals
    -/


open Classical in
/-- The Bochner integral -/
irreducible_def integral {_ : MeasurableSpace α} (μ : Measure α) (f : α → G) : G :=
  if _ : CompleteSpace G then
    if hf : Integrable f μ then L1.integral (hf.toL1 f) else 0
  else 0


@[inherit_doc MeasureTheory.integral]
notation3 "∫ "(...)", "r:60:(scoped f => f)" ∂"μ:70 => integral μ r


@[inherit_doc MeasureTheory.integral]
notation3 "∫ "(...)", "r:60:(scoped f => integral volume f) => r


@[inherit_doc MeasureTheory.integral]
notation3 "∫ "(...)" in "s", "r:60:(scoped f => f)" ∂"μ:70 => integral (Measure.restrict μ s) r


@[inherit_doc MeasureTheory.integral]
notation3 "∫ "(...)" in "s", "r:60:(scoped f => integral (Measure.restrict volume s) f) => r


theorem integral_eq (f : α → E) (hf : Integrable f μ) : ∫ a, f a ∂μ = L1.integral (hf.toL1 f) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → E
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.L1.integral (Measu …
  -/
  simp [integral, hE, hf]
  /-
    🎉 no goals
  -/


theorem integral_eq_setToFun (f : α → E) :
    ∫ a, f a ∂μ = setToFun μ (weightedSMul μ) (dominatedFinMeasAdditive_weightedSMul μ) f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → E
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.setToFun μ (Measur …
  -/
  simp only [integral, hE, L1.integral]; rfl
                                         /-
                                           🎉 no goals
                                         -/


theorem L1.integral_eq_integral (f : α →₁[μ] E) : L1.integral f = ∫ a, f a ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integral f) (MeasureTheory.integral μ fun a => ↑↑f a)
  -/
  simp only [integral, L1.integral, integral_eq_setToFun]
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.integralCLM f) (MeasureTheory.setToFun μ (MeasureTheory …
  -/
  exact (L1.setToFun_eq_setToL1 (dominatedFinMeasAdditive_weightedSMul μ) f).symm
  /-
    🎉 no goals
  -/


theorem integral_undef {f : α → G} (h : ¬Integrable f μ) : ∫ a, f a ∂μ = 0 := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    h : Not (MeasureTheory.Integrable f μ)
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) 0
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      h : Not (MeasureTheory.Integrable f μ)
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun a => f a) 0
    -/
  · simp [integral, hG, h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      h : Not (MeasureTheory.Integrable f μ)
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => f a) 0
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


theorem Integrable.of_integral_ne_zero {f : α → G} (h : ∫ a, f a ∂μ ≠ 0) : Integrable f μ :=
  Not.imp_symm integral_undef h


theorem integral_non_aestronglyMeasurable {f : α → G} (h : ¬AEStronglyMeasurable f μ) :
    ∫ a, f a ∂μ = 0 :=
  integral_undef <| not_and_of_not_left _ h


@[simp]
theorem integral_zero : ∫ _ : α, (0 : G) ∂μ = 0 := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.integral μ fun x => 0) 0
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun x => 0) 0
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun x => 0) μ) (fun  …
    -/
    exact setToFun_zero (dominatedFinMeasAdditive_weightedSMul μ)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun x => 0) 0
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


@[simp]
theorem integral_zero' : integral μ (0 : α → G) = 0 :=
  integral_zero α G


theorem integrable_of_integral_eq_one {f : α → ℝ} (h : ∫ x, f x ∂μ = 1) : Integrable f μ :=
  .of_integral_ne_zero <| h ▸ one_ne_zero


theorem integral_add {f g : α → G} (hf : Integrable f μ) (hg : Integrable g μ) :
    ∫ a, f a + g a ∂μ = ∫ a, f a ∂μ + ∫ a, g a ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → G
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (MeasureTheory.integral μ fun a => HAdd.hAdd (f a) (g a)) (HAdd.hAdd (Mea …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun a => HAdd.hAdd (f a) (g a)) (HAdd.hAdd (Mea …
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun a => HAdd.hAdd ( …
    -/
    exact setToFun_add (dominatedFinMeasAdditive_weightedSMul μ) hf hg
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => HAdd.hAdd (f a) (g a)) (HAdd.hAdd (Mea …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


theorem integral_add' {f g : α → G} (hf : Integrable f μ) (hg : Integrable g μ) :
    ∫ a, (f + g) a ∂μ = ∫ a, f a ∂μ + ∫ a, g a ∂μ :=
  integral_add hf hg


theorem integral_finset_sum {ι} (s : Finset ι) {f : ι → α → G} (hf : ∀ i ∈ s, Integrable (f i) μ) :
    ∫ a, ∑ i ∈ s, f i a ∂μ = ∑ i ∈ s, ∫ a, f i a ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    s : Finset ι
    f : ι → α → G
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (f i) μ
    ⊢ Eq (MeasureTheory.integral μ fun a => s.sum fun i => f i a) (s.sum fun i =>  …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_6
      s : Finset ι
      f : ι → α → G
      hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (f i) μ
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun a => s.sum fun i => f i a) (s.sum fun i =>  …
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_6
      s : Finset ι
      f : ι → α → G
      hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (f i) μ
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun a => s.sum fun i …
    -/
    exact setToFun_finset_sum (dominatedFinMeasAdditive_weightedSMul _) s hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_6
      s : Finset ι
      f : ι → α → G
      hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (f i) μ
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => s.sum fun i => f i a) (s.sum fun i =>  …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


@[integral_simps]
theorem integral_neg (f : α → G) : ∫ a, -f a ∂μ = -∫ a, f a ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    ⊢ Eq (MeasureTheory.integral μ fun a => Neg.neg (f a)) (Neg.neg (MeasureTheory …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun a => Neg.neg (f a)) (Neg.neg (MeasureTheory …
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun a => Neg.neg (f  …
    -/
    exact setToFun_neg (dominatedFinMeasAdditive_weightedSMul μ) f
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => Neg.neg (f a)) (Neg.neg (MeasureTheory …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


theorem integral_neg' (f : α → G) : ∫ a, (-f) a ∂μ = -∫ a, f a ∂μ :=
  integral_neg f


theorem integral_sub {f g : α → G} (hf : Integrable f μ) (hg : Integrable g μ) :
    ∫ a, f a - g a ∂μ = ∫ a, f a ∂μ - ∫ a, g a ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → G
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (MeasureTheory.integral μ fun a => HSub.hSub (f a) (g a)) (HSub.hSub (Mea …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun a => HSub.hSub (f a) (g a)) (HSub.hSub (Mea …
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun a => HSub.hSub ( …
    -/
    exact setToFun_sub (dominatedFinMeasAdditive_weightedSMul μ) hf hg
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => HSub.hSub (f a) (g a)) (HSub.hSub (Mea …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


theorem integral_sub' {f g : α → G} (hf : Integrable f μ) (hg : Integrable g μ) :
    ∫ a, (f - g) a ∂μ = ∫ a, f a ∂μ - ∫ a, g a ∂μ :=
  integral_sub hf hg


@[integral_simps]
theorem integral_smul [NormedSpace 𝕜 G] [SMulCommClass ℝ 𝕜 G] (c : 𝕜) (f : α → G) :
    ∫ a, c • f a ∂μ = c • ∫ a, f a ∂μ := by
  /-
    α : Type u_1
    𝕜 : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    G : Type u_5
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : SMulCommClass Real 𝕜 G
    c : 𝕜
    f : α → G
    ⊢ Eq (MeasureTheory.integral μ fun a => HSMul.hSMul c (f a)) (HSMul.hSMul c (M …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      𝕜 : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : SMulCommClass Real 𝕜 G
      c : 𝕜
      f : α → G
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun a => HSMul.hSMul c (f a)) (HSMul.hSMul c (M …
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      𝕜 : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : SMulCommClass Real 𝕜 G
      c : 𝕜
      f : α → G
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun a => HSMul.hSMul …
    -/
    exact setToFun_smul (dominatedFinMeasAdditive_weightedSMul μ) weightedSMul_smul c f
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      𝕜 : Type u_4
      inst✝⁴ : NontriviallyNormedField 𝕜
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : SMulCommClass Real 𝕜 G
      c : 𝕜
      f : α → G
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => HSMul.hSMul c (f a)) (HSMul.hSMul c (M …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


theorem integral_mul_left {L : Type*} [RCLike L] (r : L) (f : α → L) :
    ∫ a, r * f a ∂μ = r * ∫ a, f a ∂μ :=
  integral_smul r f


theorem integral_mul_right {L : Type*} [RCLike L] (r : L) (f : α → L) :
    ∫ a, f a * r ∂μ = (∫ a, f a ∂μ) * r := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    L : Type u_6
    inst✝ : RCLike L
    r : L
    f : α → L
    ⊢ Eq (MeasureTheory.integral μ fun a => HMul.hMul (f a) r) (HMul.hMul (Measure …
  -/
  simp only [mul_comm]; exact integral_mul_left r f
                        /-
                          🎉 no goals
                        -/


theorem integral_div {L : Type*} [RCLike L] (r : L) (f : α → L) :
    ∫ a, f a / r ∂μ = (∫ a, f a ∂μ) / r := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    L : Type u_6
    inst✝ : RCLike L
    r : L
    f : α → L
    ⊢ Eq (MeasureTheory.integral μ fun a => HDiv.hDiv (f a) r) (HDiv.hDiv (Measure …
  -/
  simpa only [← div_eq_mul_inv] using integral_mul_right r⁻¹ f
  /-
    🎉 no goals
  -/


theorem integral_congr_ae {f g : α → G} (h : f =ᵐ[μ] g) : ∫ a, f a ∂μ = ∫ a, g a ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → G
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun a = …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun a = …
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun a => f a) μ) (fu …
    -/
    exact setToFun_congr_ae (dominatedFinMeasAdditive_weightedSMul μ) h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → G
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun a = …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


lemma integral_congr_ae₂ {β : Type*} {_ : MeasurableSpace β} {ν : Measure β} {f g : α → β → G}
    (h : ∀ᵐ a ∂μ, f a =ᵐ[ν] g a) :
    ∫ a, ∫ b, f a b ∂ν ∂μ = ∫ a, ∫ b, g a b ∂ν ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    x✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f g : α → β → G
    h : Filter.Eventually (fun a => (MeasureTheory.ae ν).EventuallyEq (f a) (g a)) …
    ⊢ Eq (MeasureTheory.integral μ fun a => MeasureTheory.integral ν fun b => f a  …
  -/
  apply integral_congr_ae
  /-
    case h
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    x✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f g : α → β → G
    h : Filter.Eventually (fun a => (MeasureTheory.ae ν).EventuallyEq (f a) (g a)) …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => MeasureTheory.integral ν fun b = …
  -/
  filter_upwards [h] with _ ha
  /-
    case h
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    x✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f g : α → β → G
    h : Filter.Eventually (fun a => (MeasureTheory.ae ν).EventuallyEq (f a) (g a)) …
    a✝ : α
    ha : (MeasureTheory.ae ν).EventuallyEq (f a✝) (g a✝)
    ⊢ Eq (MeasureTheory.integral ν fun b => f a✝ b) (MeasureTheory.integral ν fun  …
  -/
  apply integral_congr_ae
  /-
    case h.h
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    x✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f g : α → β → G
    h : Filter.Eventually (fun a => (MeasureTheory.ae ν).EventuallyEq (f a) (g a)) …
    a✝ : α
    ha : (MeasureTheory.ae ν).EventuallyEq (f a✝) (g a✝)
    ⊢ (MeasureTheory.ae ν).EventuallyEq (f a✝) (g a✝)
  -/
  filter_upwards [ha] with _ hb using hb
  /-
    🎉 no goals
  -/

-- Porting note: `nolint simpNF` added because simplify fails on left-hand side

@[simp, nolint simpNF]
theorem L1.integral_of_fun_eq_integral {f : α → G} (hf : Integrable f μ) :
    ∫ a, (hf.toL1 f) a ∂μ = ∫ a, f a ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun a => ↑↑(MeasureTheory.Integrable.toL1 f hf) …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hf : MeasureTheory.Integrable f μ
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral μ fun a => ↑↑(MeasureTheory.Integrable.toL1 f hf) …
    -/
  · simp only [MeasureTheory.integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hf : MeasureTheory.Integrable f μ
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun a => ↑↑(MeasureT …
    -/
    exact setToFun_toL1 (dominatedFinMeasAdditive_weightedSMul μ) hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hf : MeasureTheory.Integrable f μ
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => ↑↑(MeasureTheory.Integrable.toL1 f hf) …
    -/
  · simp [MeasureTheory.integral, hG]
    /-
      🎉 no goals
    -/


@[continuity]
theorem continuous_integral : Continuous fun f : α →₁[μ] G => ∫ a, f a ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Continuous fun f => MeasureTheory.integral μ fun a => ↑↑f a
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hG : CompleteSpace G
      ⊢ Continuous fun f => MeasureTheory.integral μ fun a => ↑↑f a
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hG : CompleteSpace G
      ⊢ Continuous fun f => dite True (fun h => dite (MeasureTheory.Integrable (fun  …
    -/
    exact continuous_setToFun (dominatedFinMeasAdditive_weightedSMul μ)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hG : Not (CompleteSpace G)
      ⊢ Continuous fun f => MeasureTheory.integral μ fun a => ↑↑f a
    -/
  · simp [integral, hG, continuous_const]
    /-
      🎉 no goals
    -/


theorem norm_integral_le_lintegral_norm (f : α → G) :
    ‖∫ a, f a ∂μ‖ ≤ ENNReal.toReal (∫⁻ a, ENNReal.ofReal ‖f a‖ ∂μ) := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.lin …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hG : CompleteSpace G
      ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.lin …
    -/
  · by_cases hf : Integrable f μ
      /-
        case pos
        α : Type u_1
        G : Type u_5
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace Real G
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → G
        hG : CompleteSpace G
        hf : MeasureTheory.Integrable f μ
        ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.lin …
      -/
    · rw [integral_eq f hf, ← Integrable.norm_toL1_eq_lintegral_norm f hf]
      /-
        case pos
        α : Type u_1
        G : Type u_5
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace Real G
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → G
        hG : CompleteSpace G
        hf : MeasureTheory.Integrable f μ
        ⊢ LE.le (Norm.norm (MeasureTheory.L1.integral (MeasureTheory.Integrable.toL1 f …
      -/
      exact L1.norm_integral_le _
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        G : Type u_5
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace Real G
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → G
        hG : CompleteSpace G
        hf : Not (MeasureTheory.Integrable f μ)
        ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.lin …
      -/
    · rw [integral_undef hf, norm_zero]; exact toReal_nonneg
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hG : Not (CompleteSpace G)
      ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.lin …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


theorem ennnorm_integral_le_lintegral_ennnorm (f : α → G) :
    (‖∫ a, f a ∂μ‖₊ : ℝ≥0∞) ≤ ∫⁻ a, ‖f a‖₊ ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    ⊢ LE.le (↑(NNNorm.nnnorm (MeasureTheory.integral μ fun a => f a))) (MeasureThe …
  -/
  simp_rw [← ofReal_norm_eq_coe_nnnorm]
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    ⊢ LE.le (ENNReal.ofReal (Norm.norm (MeasureTheory.integral μ fun a => f a))) ( …
  -/
  apply ENNReal.ofReal_le_of_le_toReal
  /-
    case h
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.lin …
  -/
  exact norm_integral_le_lintegral_norm f
  /-
    🎉 no goals
  -/


theorem integral_eq_zero_of_ae {f : α → G} (hf : f =ᵐ[μ] 0) : ∫ a, f a ∂μ = 0 := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    hf : (MeasureTheory.ae μ).EventuallyEq f 0
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) 0
  -/
  simp [integral_congr_ae hf, integral_zero]
  /-
    🎉 no goals
  -/


/-- If `f` has finite integral, then `∫ x in s, f x ∂μ` is absolutely continuous in `s`: it tends
to zero as `μ s` tends to zero. -/
theorem HasFiniteIntegral.tendsto_setIntegral_nhds_zero {ι} {f : α → G}
    (hf : HasFiniteIntegral f μ) {l : Filter ι} {s : ι → Set α} (hs : Tendsto (μ ∘ s) l (𝓝 0)) :
    Tendsto (fun i => ∫ x in s i, f x ∂μ) l (𝓝 0) := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hf : MeasureTheory.HasFiniteIntegral f μ
    l : Filter ι
    s : ι → Set α
    hs : Filter.Tendsto (Function.comp (⇑μ) s) l (nhds 0)
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero]
  simp_rw [← coe_nnnorm, ← NNReal.coe_zero, NNReal.tendsto_coe, ← ENNReal.tendsto_coe,
    ENNReal.coe_zero]
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (tendsto_setLIntegral_zero (ne_of_lt hf) hs) (fun i => zero_le _)
    fun i => ennnorm_integral_le_lintegral_ennnorm _


@[deprecated (since := "2024-04-17")]
alias HasFiniteIntegral.tendsto_set_integral_nhds_zero :=
  HasFiniteIntegral.tendsto_setIntegral_nhds_zero


/-- If `f` is integrable, then `∫ x in s, f x ∂μ` is absolutely continuous in `s`: it tends
to zero as `μ s` tends to zero. -/
theorem Integrable.tendsto_setIntegral_nhds_zero {ι} {f : α → G} (hf : Integrable f μ)
    {l : Filter ι} {s : ι → Set α} (hs : Tendsto (μ ∘ s) l (𝓝 0)) :
    Tendsto (fun i => ∫ x in s i, f x ∂μ) l (𝓝 0) :=
  hf.2.tendsto_setIntegral_nhds_zero hs


@[deprecated (since := "2024-04-17")]
alias Integrable.tendsto_set_integral_nhds_zero :=
  Integrable.tendsto_setIntegral_nhds_zero


/-- If `F i → f` in `L1`, then `∫ x, F i x ∂μ → ∫ x, f x ∂μ`. -/
theorem tendsto_integral_of_L1 {ι} (f : α → G) (hfi : Integrable f μ) {F : ι → α → G} {l : Filter ι}
    (hFi : ∀ᶠ i in l, Integrable (F i) μ)
    (hF : Tendsto (fun i => ∫⁻ x, ‖F i x - f x‖₊ ∂μ) l (𝓝 0)) :
    Tendsto (fun i => ∫ x, F i x ∂μ) l (𝓝 <| ∫ x, f x ∂μ) := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hfi : MeasureTheory.Integrable f μ
    F : ι → α → G
    l : Filter ι
    hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
    hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun x => F i x) l (nhds (M …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_6
      f : α → G
      hfi : MeasureTheory.Integrable f μ
      F : ι → α → G
      l : Filter ι
      hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
      hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
      hG : CompleteSpace G
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun x => F i x) l (nhds (M …
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_6
      f : α → G
      hfi : MeasureTheory.Integrable f μ
      F : ι → α → G
      l : Filter ι
      hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
      hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
      hG : CompleteSpace G
      ⊢ Filter.Tendsto (fun i => dite True (fun h => dite (MeasureTheory.Integrable  …
    -/
    exact tendsto_setToFun_of_L1 (dominatedFinMeasAdditive_weightedSMul μ) f hfi hFi hF
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_6
      f : α → G
      hfi : MeasureTheory.Integrable f μ
      F : ι → α → G
      l : Filter ι
      hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
      hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
      hG : Not (CompleteSpace G)
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun x => F i x) l (nhds (M …
    -/
  · simp [integral, hG, tendsto_const_nhds]
    /-
      🎉 no goals
    -/


/-- If `F i → f` in `L1`, then `∫ x, F i x ∂μ → ∫ x, f x ∂μ`. -/
lemma tendsto_integral_of_L1' {ι} (f : α → G) (hfi : Integrable f μ) {F : ι → α → G} {l : Filter ι}
    (hFi : ∀ᶠ i in l, Integrable (F i) μ) (hF : Tendsto (fun i ↦ eLpNorm (F i - f) 1 μ) l (𝓝 0)) :
    Tendsto (fun i ↦ ∫ x, F i x ∂μ) l (𝓝 (∫ x, f x ∂μ)) := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hfi : MeasureTheory.Integrable f μ
    F : ι → α → G
    l : Filter ι
    hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
    hF : Filter.Tendsto (fun i => MeasureTheory.eLpNorm (HSub.hSub (F i) f) 1 μ) l …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun x => F i x) l (nhds (M …
  -/
  refine tendsto_integral_of_L1 f hfi hFi ?_
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hfi : MeasureTheory.Integrable f μ
    F : ι → α → G
    l : Filter ι
    hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
    hF : Filter.Tendsto (fun i => MeasureTheory.eLpNorm (HSub.hSub (F i) f) 1 μ) l …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm  …
  -/
  simp_rw [eLpNorm_one_eq_lintegral_nnnorm, Pi.sub_apply] at hF
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hfi : MeasureTheory.Integrable f μ
    F : ι → α → G
    l : Filter ι
    hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
    hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm  …
  -/
  exact hF
  /-
    🎉 no goals
  -/


/-- If `F i → f` in `L1`, then `∫ x in s, F i x ∂μ → ∫ x in s, f x ∂μ`. -/
lemma tendsto_setIntegral_of_L1 {ι} (f : α → G) (hfi : Integrable f μ) {F : ι → α → G}
    {l : Filter ι}
    (hFi : ∀ᶠ i in l, Integrable (F i) μ) (hF : Tendsto (fun i ↦ ∫⁻ x, ‖F i x - f x‖₊ ∂μ) l (𝓝 0))
    (s : Set α) :
    Tendsto (fun i ↦ ∫ x in s, F i x ∂μ) l (𝓝 (∫ x in s, f x ∂μ)) := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hfi : MeasureTheory.Integrable f μ
    F : ι → α → G
    l : Filter ι
    hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
    hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
    s : Set α
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict s) fun x => F i  …
  -/
  refine tendsto_integral_of_L1 f hfi.restrict ?_ ?_
    /-
      case refine_1
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_6
      f : α → G
      hfi : MeasureTheory.Integrable f μ
      F : ι → α → G
      l : Filter ι
      hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
      hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
      s : Set α
      ⊢ Filter.Eventually (fun i => MeasureTheory.Integrable (F i) (μ.restrict s)) l
    -/
  · filter_upwards [hFi] with i hi using hi.restrict
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_6
      f : α → G
      hfi : MeasureTheory.Integrable f μ
      F : ι → α → G
      l : Filter ι
      hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
      hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
      s : Set α
      ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict s) fun x => ↑(N …
    -/
  · simp_rw [← eLpNorm_one_eq_lintegral_nnnorm] at hF ⊢
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hF (fun _ ↦ zero_le')
      (fun _ ↦ eLpNorm_mono_measure _ Measure.restrict_le_self)


@[deprecated (since := "2024-04-17")]
alias tendsto_set_integral_of_L1 := tendsto_setIntegral_of_L1


/-- If `F i → f` in `L1`, then `∫ x in s, F i x ∂μ → ∫ x in s, f x ∂μ`. -/
lemma tendsto_setIntegral_of_L1' {ι} (f : α → G) (hfi : Integrable f μ) {F : ι → α → G}
    {l : Filter ι}
    (hFi : ∀ᶠ i in l, Integrable (F i) μ) (hF : Tendsto (fun i ↦ eLpNorm (F i - f) 1 μ) l (𝓝 0))
    (s : Set α) :
    Tendsto (fun i ↦ ∫ x in s, F i x ∂μ) l (𝓝 (∫ x in s, f x ∂μ)) := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hfi : MeasureTheory.Integrable f μ
    F : ι → α → G
    l : Filter ι
    hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
    hF : Filter.Tendsto (fun i => MeasureTheory.eLpNorm (HSub.hSub (F i) f) 1 μ) l …
    s : Set α
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict s) fun x => F i  …
  -/
  refine tendsto_setIntegral_of_L1 f hfi hFi ?_ s
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hfi : MeasureTheory.Integrable f μ
    F : ι → α → G
    l : Filter ι
    hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
    hF : Filter.Tendsto (fun i => MeasureTheory.eLpNorm (HSub.hSub (F i) f) 1 μ) l …
    s : Set α
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm  …
  -/
  simp_rw [eLpNorm_one_eq_lintegral_nnnorm, Pi.sub_apply] at hF
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_6
    f : α → G
    hfi : MeasureTheory.Integrable f μ
    F : ι → α → G
    l : Filter ι
    hFi : Filter.Eventually (fun i => MeasureTheory.Integrable (F i) μ) l
    s : Set α
    hF : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnno …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm  …
  -/
  exact hF
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias tendsto_set_integral_of_L1' := tendsto_setIntegral_of_L1'


theorem continuousWithinAt_of_dominated {F : X → α → G} {x₀ : X} {bound : α → ℝ} {s : Set X}
    (hF_meas : ∀ᶠ x in 𝓝[s] x₀, AEStronglyMeasurable (F x) μ)
    (h_bound : ∀ᶠ x in 𝓝[s] x₀, ∀ᵐ a ∂μ, ‖F x a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_cont : ∀ᵐ a ∂μ, ContinuousWithinAt (fun x => F x a) s x₀) :
    ContinuousWithinAt (fun x => ∫ a, F x a ∂μ) s x₀ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    X : Type u_6
    inst✝¹ : TopologicalSpace X
    inst✝ : FirstCountableTopology X
    F : X → α → G
    x₀ : X
    bound : α → Real
    s : Set X
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    h_bound : Filter.Eventually (fun x => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_cont : Filter.Eventually (fun a => ContinuousWithinAt (fun x => F x a) s x₀) …
    ⊢ ContinuousWithinAt (fun x => MeasureTheory.integral μ fun a => F x a) s x₀
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      X : Type u_6
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      F : X → α → G
      x₀ : X
      bound : α → Real
      s : Set X
      hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousWithinAt (fun x => F x a) s x₀) …
      hG : CompleteSpace G
      ⊢ ContinuousWithinAt (fun x => MeasureTheory.integral μ fun a => F x a) s x₀
    -/
  · simp only [integral, hG, L1.integral]
    exact continuousWithinAt_setToFun_of_dominated (dominatedFinMeasAdditive_weightedSMul μ)
      hF_meas h_bound bound_integrable h_cont
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      X : Type u_6
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      F : X → α → G
      x₀ : X
      bound : α → Real
      s : Set X
      hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousWithinAt (fun x => F x a) s x₀) …
      hG : Not (CompleteSpace G)
      ⊢ ContinuousWithinAt (fun x => MeasureTheory.integral μ fun a => F x a) s x₀
    -/
  · simp [integral, hG, continuousWithinAt_const]
    /-
      🎉 no goals
    -/


theorem continuousAt_of_dominated {F : X → α → G} {x₀ : X} {bound : α → ℝ}
    (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) μ)
    (h_bound : ∀ᶠ x in 𝓝 x₀, ∀ᵐ a ∂μ, ‖F x a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_cont : ∀ᵐ a ∂μ, ContinuousAt (fun x => F x a) x₀) :
    ContinuousAt (fun x => ∫ a, F x a ∂μ) x₀ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    X : Type u_6
    inst✝¹ : TopologicalSpace X
    inst✝ : FirstCountableTopology X
    F : X → α → G
    x₀ : X
    bound : α → Real
    hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
    h_bound : Filter.Eventually (fun x => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_cont : Filter.Eventually (fun a => ContinuousAt (fun x => F x a) x₀) (Measur …
    ⊢ ContinuousAt (fun x => MeasureTheory.integral μ fun a => F x a) x₀
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      X : Type u_6
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      F : X → α → G
      x₀ : X
      bound : α → Real
      hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousAt (fun x => F x a) x₀) (Measur …
      hG : CompleteSpace G
      ⊢ ContinuousAt (fun x => MeasureTheory.integral μ fun a => F x a) x₀
    -/
  · simp only [integral, hG, L1.integral]
    exact continuousAt_setToFun_of_dominated (dominatedFinMeasAdditive_weightedSMul μ)
      hF_meas h_bound bound_integrable h_cont
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      X : Type u_6
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      F : X → α → G
      x₀ : X
      bound : α → Real
      hF_meas : Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (F x) …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousAt (fun x => F x a) x₀) (Measur …
      hG : Not (CompleteSpace G)
      ⊢ ContinuousAt (fun x => MeasureTheory.integral μ fun a => F x a) x₀
    -/
  · simp [integral, hG, continuousAt_const]
    /-
      🎉 no goals
    -/


theorem continuousOn_of_dominated {F : X → α → G} {bound : α → ℝ} {s : Set X}
    (hF_meas : ∀ x ∈ s, AEStronglyMeasurable (F x) μ)
    (h_bound : ∀ x ∈ s, ∀ᵐ a ∂μ, ‖F x a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_cont : ∀ᵐ a ∂μ, ContinuousOn (fun x => F x a) s) :
    ContinuousOn (fun x => ∫ a, F x a ∂μ) s := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    X : Type u_6
    inst✝¹ : TopologicalSpace X
    inst✝ : FirstCountableTopology X
    F : X → α → G
    bound : α → Real
    s : Set X
    hF_meas : ∀ (x : X), Membership.mem s x → MeasureTheory.AEStronglyMeasurable ( …
    h_bound : ∀ (x : X), Membership.mem s x → Filter.Eventually (fun a => LE.le (N …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_cont : Filter.Eventually (fun a => ContinuousOn (fun x => F x a) s) (Measure …
    ⊢ ContinuousOn (fun x => MeasureTheory.integral μ fun a => F x a) s
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      X : Type u_6
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      F : X → α → G
      bound : α → Real
      s : Set X
      hF_meas : ∀ (x : X), Membership.mem s x → MeasureTheory.AEStronglyMeasurable ( …
      h_bound : ∀ (x : X), Membership.mem s x → Filter.Eventually (fun a => LE.le (N …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousOn (fun x => F x a) s) (Measure …
      hG : CompleteSpace G
      ⊢ ContinuousOn (fun x => MeasureTheory.integral μ fun a => F x a) s
    -/
  · simp only [integral, hG, L1.integral]
    exact continuousOn_setToFun_of_dominated (dominatedFinMeasAdditive_weightedSMul μ)
      hF_meas h_bound bound_integrable h_cont
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      X : Type u_6
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      F : X → α → G
      bound : α → Real
      s : Set X
      hF_meas : ∀ (x : X), Membership.mem s x → MeasureTheory.AEStronglyMeasurable ( …
      h_bound : ∀ (x : X), Membership.mem s x → Filter.Eventually (fun a => LE.le (N …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousOn (fun x => F x a) s) (Measure …
      hG : Not (CompleteSpace G)
      ⊢ ContinuousOn (fun x => MeasureTheory.integral μ fun a => F x a) s
    -/
  · simp [integral, hG, continuousOn_const]
    /-
      🎉 no goals
    -/


theorem continuous_of_dominated {F : X → α → G} {bound : α → ℝ}
    (hF_meas : ∀ x, AEStronglyMeasurable (F x) μ) (h_bound : ∀ x, ∀ᵐ a ∂μ, ‖F x a‖ ≤ bound a)
    (bound_integrable : Integrable bound μ) (h_cont : ∀ᵐ a ∂μ, Continuous fun x => F x a) :
    Continuous fun x => ∫ a, F x a ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    X : Type u_6
    inst✝¹ : TopologicalSpace X
    inst✝ : FirstCountableTopology X
    F : X → α → G
    bound : α → Real
    hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) μ
    h_bound : ∀ (x : X), Filter.Eventually (fun a => LE.le (Norm.norm (F x a)) (bo …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_cont : Filter.Eventually (fun a => Continuous fun x => F x a) (MeasureTheory …
    ⊢ Continuous fun x => MeasureTheory.integral μ fun a => F x a
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      X : Type u_6
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      F : X → α → G
      bound : α → Real
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) μ
      h_bound : ∀ (x : X), Filter.Eventually (fun a => LE.le (Norm.norm (F x a)) (bo …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => Continuous fun x => F x a) (MeasureTheory …
      hG : CompleteSpace G
      ⊢ Continuous fun x => MeasureTheory.integral μ fun a => F x a
    -/
  · simp only [integral, hG, L1.integral]
    exact continuous_setToFun_of_dominated (dominatedFinMeasAdditive_weightedSMul μ)
      hF_meas h_bound bound_integrable h_cont
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      X : Type u_6
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      F : X → α → G
      bound : α → Real
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) μ
      h_bound : ∀ (x : X), Filter.Eventually (fun a => LE.le (Norm.norm (F x a)) (bo …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => Continuous fun x => F x a) (MeasureTheory …
      hG : Not (CompleteSpace G)
      ⊢ Continuous fun x => MeasureTheory.integral μ fun a => F x a
    -/
  · simp [integral, hG, continuous_const]
    /-
      🎉 no goals
    -/


/-- The Bochner integral of a real-valued function `f : α → ℝ` is the difference between the
  integral of the positive part of `f` and the integral of the negative part of `f`. -/
theorem integral_eq_lintegral_pos_part_sub_lintegral_neg_part {f : α → ℝ} (hf : Integrable f μ) :
    ∫ a, f a ∂μ =
      ENNReal.toReal (∫⁻ a, .ofReal (f a) ∂μ) - ENNReal.toReal (∫⁻ a, .ofReal (-f a) ∂μ) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (HSub.hSub (MeasureTheory.lintegr …
  -/
  let f₁ := hf.toL1 f
  -- Go to the `L¹` space
  have eq₁ : ENNReal.toReal (∫⁻ a, ENNReal.ofReal (f a) ∂μ) = ‖Lp.posPart f₁‖ := by
    rw [L1.norm_def]
    congr 1
    apply lintegral_congr_ae
    filter_upwards [Lp.coeFn_posPart f₁, hf.coeFn_toL1] with _ h₁ h₂
    rw [h₁, h₂, ENNReal.ofReal]
    congr 1
    apply NNReal.eq
    rw [Real.nnnorm_of_nonneg (le_max_right _ _)]
    rw [Real.coe_toNNReal', NNReal.coe_mk]
  -- Go to the `L¹` space
  have eq₂ : ENNReal.toReal (∫⁻ a, ENNReal.ofReal (-f a) ∂μ) = ‖Lp.negPart f₁‖ := by
    rw [L1.norm_def]
    congr 1
    apply lintegral_congr_ae
    filter_upwards [Lp.coeFn_negPart f₁, hf.coeFn_toL1] with _ h₁ h₂
    rw [h₁, h₂, ENNReal.ofReal]
    congr 1
    apply NNReal.eq
    simp only [Real.coe_toNNReal', coe_nnnorm, nnnorm_neg]
    rw [Real.norm_of_nonpos (min_le_right _ _), ← max_neg_neg, neg_zero]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    f₁ : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x := MeasureT …
    eq₁ : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)).toReal (Nor …
    eq₂ : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (Neg.neg (f a))).t …
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (HSub.hSub (MeasureTheory.lintegr …
  -/
  rw [eq₁, eq₂, integral, dif_pos, dif_pos]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    f₁ : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 1 μ) x := MeasureT …
    eq₁ : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)).toReal (Nor …
    eq₂ : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (Neg.neg (f a))).t …
    ⊢ Eq (MeasureTheory.L1.integral (MeasureTheory.Integrable.toL1 (fun a => f a)  …
  -/
  exact L1.integral_eq_norm_posPart_sub _
  /-
    🎉 no goals
  -/


theorem integral_eq_lintegral_of_nonneg_ae {f : α → ℝ} (hf : 0 ≤ᵐ[μ] f)
    (hfm : AEStronglyMeasurable f μ) :
    ∫ a, f a ∂μ = ENNReal.toReal (∫⁻ a, ENNReal.ofReal (f a) ∂μ) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.lintegral μ fun a  …
  -/
  by_cases hfi : Integrable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfm : MeasureTheory.AEStronglyMeasurable f μ
      hfi : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.lintegral μ fun a  …
    -/
  · rw [integral_eq_lintegral_pos_part_sub_lintegral_neg_part hfi]
    have h_min : ∫⁻ a, ENNReal.ofReal (-f a) ∂μ = 0 := by
      rw [lintegral_eq_zero_iff']
      · refine hf.mono ?_
        simp only [Pi.zero_apply]
        intro a h
        simp only [h, neg_nonpos, ofReal_eq_zero]
      · exact measurable_ofReal.comp_aemeasurable hfm.aemeasurable.neg
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfm : MeasureTheory.AEStronglyMeasurable f μ
      hfi : MeasureTheory.Integrable f μ
      h_min : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (Neg.neg (f a))) 0
      ⊢ Eq (HSub.hSub (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)).toRe …
    -/
    rw [h_min, zero_toReal, _root_.sub_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfm : MeasureTheory.AEStronglyMeasurable f μ
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.lintegral μ fun a  …
    -/
  · rw [integral_undef hfi]
    simp_rw [Integrable, hfm, hasFiniteIntegral_iff_norm, lt_top_iff_ne_top, Ne, true_and,
      Classical.not_not] at hfi
    have : ∫⁻ a : α, ENNReal.ofReal (f a) ∂μ = ∫⁻ a, ENNReal.ofReal ‖f a‖ ∂μ := by
      refine lintegral_congr_ae (hf.mono fun a h => ?_)
      dsimp only
      rw [Real.norm_eq_abs, abs_of_nonneg h]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfm : MeasureTheory.AEStronglyMeasurable f μ
      hfi : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (Norm.norm (f a))) …
      this : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)) (MeasureTh …
      ⊢ Eq 0 (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)).toReal
    -/
    rw [this, hfi]; rfl
                    /-
                      🎉 no goals
                    -/


theorem integral_norm_eq_lintegral_nnnorm {P : Type*} [NormedAddCommGroup P] {f : α → P}
    (hf : AEStronglyMeasurable f μ) : ∫ x, ‖f x‖ ∂μ = ENNReal.toReal (∫⁻ x, ‖f x‖₊ ∂μ) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    P : Type u_7
    inst✝ : NormedAddCommGroup P
    f : α → P
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Eq (MeasureTheory.integral μ fun x => Norm.norm (f x)) (MeasureTheory.linteg …
  -/
  rw [integral_eq_lintegral_of_nonneg_ae _ hf.norm]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      P : Type u_7
      inst✝ : NormedAddCommGroup P
      f : α → P
      hf : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (Norm.norm (f a))).toR …
    -/
  · simp_rw [ofReal_norm_eq_coe_nnnorm]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      P : Type u_7
      inst✝ : NormedAddCommGroup P
      f : α → P
      hf : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => Norm.norm (f x)
    -/
  · filter_upwards; simp_rw [Pi.zero_apply, norm_nonneg, imp_true_iff]
                    /-
                      🎉 no goals
                    -/


theorem ofReal_integral_norm_eq_lintegral_nnnorm {P : Type*} [NormedAddCommGroup P] {f : α → P}
    (hf : Integrable f μ) : ENNReal.ofReal (∫ x, ‖f x‖ ∂μ) = ∫⁻ x, ‖f x‖₊ ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    P : Type u_7
    inst✝ : NormedAddCommGroup P
    f : α → P
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral μ fun x => Norm.norm (f x))) (Mea …
  -/
  rw [integral_norm_eq_lintegral_nnnorm hf.aestronglyMeasurable, ENNReal.ofReal_toReal]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    P : Type u_7
    inst✝ : NormedAddCommGroup P
    f : α → P
    hf : MeasureTheory.Integrable f μ
    ⊢ Ne (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm (f x))) Top.top
  -/
  exact lt_top_iff_ne_top.mp (hasFiniteIntegral_iff_nnnorm.mpr hf.2)
  /-
    🎉 no goals
  -/


theorem integral_eq_integral_pos_part_sub_integral_neg_part {f : α → ℝ} (hf : Integrable f μ) :
    ∫ a, f a ∂μ = ∫ a, (Real.toNNReal (f a) : ℝ) ∂μ - ∫ a, (Real.toNNReal (-f a) : ℝ) ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (HSub.hSub (MeasureTheory.integra …
  -/
  rw [← integral_sub hf.real_toNNReal]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun a = …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.Integrable (fun a => ↑(Neg.neg (f a)).toNNReal) μ
    -/
  · exact hf.neg.real_toNNReal
    /-
      🎉 no goals
    -/


theorem integral_nonneg_of_ae {f : α → ℝ} (hf : 0 ≤ᵐ[μ] f) : 0 ≤ ∫ a, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ LE.le 0 (MeasureTheory.integral μ fun a => f a)
  -/
  have A : CompleteSpace ℝ := by infer_instance
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    A : CompleteSpace Real
    ⊢ LE.le 0 (MeasureTheory.integral μ fun a => f a)
  -/
  simp only [integral_def, A, L1.integral_def, dite_true]
  exact setToFun_nonneg (dominatedFinMeasAdditive_weightedSMul μ)
    (fun s _ _ => weightedSMul_nonneg s) hf


theorem lintegral_coe_eq_integral (f : α → ℝ≥0) (hfi : Integrable (fun x => (f x : ℝ)) μ) :
    ∫⁻ a, f a ∂μ = ENNReal.ofReal (∫ a, f a ∂μ) := by
  simp_rw [integral_eq_lintegral_of_nonneg_ae (Eventually.of_forall fun x => (f x).coe_nonneg)
      hfi.aestronglyMeasurable, ← ENNReal.coe_nnreal_eq]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    hfi : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑(f a)) (ENNReal.ofReal (MeasureTheor …
  -/
  rw [ENNReal.ofReal_toReal]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    hfi : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ⊢ Ne (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
  -/
  rw [← lt_top_iff_ne_top]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    hfi : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
  -/
  convert hfi.hasFiniteIntegral
  -- Porting note: `convert` no longer unfolds `HasFiniteIntegral`
  /-
    case a
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    hfi : MeasureTheory.Integrable (fun x => ↑(f x)) μ
    ⊢ Iff (LT.lt (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top) (MeasureTheo …
  -/
  simp_rw [hasFiniteIntegral_iff_nnnorm, NNReal.nnnorm_eq]
  /-
    🎉 no goals
  -/


theorem ofReal_integral_eq_lintegral_ofReal {f : α → ℝ} (hfi : Integrable f μ) (f_nn : 0 ≤ᵐ[μ] f) :
    ENNReal.ofReal (∫ x, f x ∂μ) = ∫⁻ x, ENNReal.ofReal (f x) ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfi : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (MeasureTheory.l …
  -/
  have : f =ᵐ[μ] (‖f ·‖) := f_nn.mono fun _x hx ↦ (abs_of_nonneg hx).symm
  simp_rw [integral_congr_ae this, ofReal_integral_norm_eq_lintegral_nnnorm hfi,
    ← ofReal_norm_eq_coe_nnnorm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfi : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    this : (MeasureTheory.ae μ).EventuallyEq f fun x => Norm.norm (f x)
    ⊢ Eq (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (Norm.norm (f x))) (Me …
  -/
  exact lintegral_congr_ae (this.symm.fun_comp ENNReal.ofReal)
  /-
    🎉 no goals
  -/


theorem integral_toReal {f : α → ℝ≥0∞} (hfm : AEMeasurable f μ) (hf : ∀ᵐ x ∂μ, f x < ∞) :
    ∫ a, (f a).toReal ∂μ = (∫⁻ a, f a ∂μ).toReal := by
  rw [integral_eq_lintegral_of_nonneg_ae _ hfm.ennreal_toReal.aestronglyMeasurable,
    lintegral_congr_ae (ofReal_toReal_ae_eq hf)]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : AEMeasurable f μ
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => (f x).toReal
  -/
  exact Eventually.of_forall fun x => ENNReal.toReal_nonneg
  /-
    🎉 no goals
  -/


theorem lintegral_coe_le_coe_iff_integral_le {f : α → ℝ≥0} (hfi : Integrable (fun x => (f x : ℝ)) μ)
    {b : ℝ≥0} : ∫⁻ a, f a ∂μ ≤ b ↔ ∫ a, (f a : ℝ) ∂μ ≤ b := by
  rw [lintegral_coe_eq_integral f hfi, ENNReal.ofReal, ENNReal.coe_le_coe,
    Real.toNNReal_le_iff_le_coe]


theorem integral_coe_le_of_lintegral_coe_le {f : α → ℝ≥0} {b : ℝ≥0} (h : ∫⁻ a, f a ∂μ ≤ b) :
    ∫ a, (f a : ℝ) ∂μ ≤ b := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    b : NNReal
    h : LE.le (MeasureTheory.lintegral μ fun a => ↑(f a)) ↑b
    ⊢ LE.le (MeasureTheory.integral μ fun a => ↑(f a)) ↑b
  -/
  by_cases hf : Integrable (fun a => (f a : ℝ)) μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      b : NNReal
      h : LE.le (MeasureTheory.lintegral μ fun a => ↑(f a)) ↑b
      hf : MeasureTheory.Integrable (fun a => ↑(f a)) μ
      ⊢ LE.le (MeasureTheory.integral μ fun a => ↑(f a)) ↑b
    -/
  · exact (lintegral_coe_le_coe_iff_integral_le hf).1 h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → NNReal
      b : NNReal
      h : LE.le (MeasureTheory.lintegral μ fun a => ↑(f a)) ↑b
      hf : Not (MeasureTheory.Integrable (fun a => ↑(f a)) μ)
      ⊢ LE.le (MeasureTheory.integral μ fun a => ↑(f a)) ↑b
    -/
  · rw [integral_undef hf]; exact b.2
                            /-
                              🎉 no goals
                            -/


theorem integral_nonneg {f : α → ℝ} (hf : 0 ≤ f) : 0 ≤ ∫ a, f a ∂μ :=
  integral_nonneg_of_ae <| Eventually.of_forall hf


theorem integral_nonpos_of_ae {f : α → ℝ} (hf : f ≤ᵐ[μ] 0) : ∫ a, f a ∂μ ≤ 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : (MeasureTheory.ae μ).EventuallyLE f 0
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) 0
  -/
  have hf : 0 ≤ᵐ[μ] -f := hf.mono fun a h => by rwa [Pi.neg_apply, Pi.zero_apply, neg_nonneg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf✝ : (MeasureTheory.ae μ).EventuallyLE f 0
    hf : (MeasureTheory.ae μ).EventuallyLE 0 (Neg.neg f)
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) 0
  -/
  have : 0 ≤ ∫ a, -f a ∂μ := integral_nonneg_of_ae hf
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf✝ : (MeasureTheory.ae μ).EventuallyLE f 0
    hf : (MeasureTheory.ae μ).EventuallyLE 0 (Neg.neg f)
    this : LE.le 0 (MeasureTheory.integral μ fun a => Neg.neg (f a))
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) 0
  -/
  rwa [integral_neg, neg_nonneg] at this
  /-
    🎉 no goals
  -/


theorem integral_nonpos {f : α → ℝ} (hf : f ≤ 0) : ∫ a, f a ∂μ ≤ 0 :=
  integral_nonpos_of_ae <| Eventually.of_forall hf


theorem integral_eq_zero_iff_of_nonneg_ae {f : α → ℝ} (hf : 0 ≤ᵐ[μ] f) (hfi : Integrable f μ) :
    ∫ x, f x ∂μ = 0 ↔ f =ᵐ[μ] 0 := by
  simp_rw [integral_eq_lintegral_of_nonneg_ae hf hfi.1, ENNReal.toReal_eq_zero_iff,
    ← ENNReal.not_lt_top, ← hasFiniteIntegral_iff_ofReal hf, hfi.2, not_true_eq_false, or_false]
  -- Porting note: split into parts, to make `rw` and `simp` work
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    hfi : MeasureTheory.Integrable f μ
    ⊢ Iff (Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)) 0) ((Measu …
  -/
  rw [lintegral_eq_zero_iff']
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (fun a => ENNReal.ofReal (f a)) 0) (( …
    -/
  · rw [← hf.le_iff_eq, Filter.EventuallyEq, Filter.EventuallyLE]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      ⊢ Iff (Filter.Eventually (fun x => Eq (ENNReal.ofReal (f x)) (0 x)) (MeasureTh …
    -/
    simp only [Pi.zero_apply, ofReal_eq_zero]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      ⊢ AEMeasurable (fun a => ENNReal.ofReal (f a)) μ
    -/
  · exact (ENNReal.measurable_ofReal.comp_aemeasurable hfi.1.aemeasurable)
    /-
      🎉 no goals
    -/


theorem integral_eq_zero_iff_of_nonneg {f : α → ℝ} (hf : 0 ≤ f) (hfi : Integrable f μ) :
    ∫ x, f x ∂μ = 0 ↔ f =ᵐ[μ] 0 :=
  integral_eq_zero_iff_of_nonneg_ae (Eventually.of_forall hf) hfi


lemma integral_eq_iff_of_ae_le {f g : α → ℝ}
    (hf : Integrable f μ) (hg : Integrable g μ) (hfg : f ≤ᵐ[μ] g) :
    ∫ a, f a ∂μ = ∫ a, g a ∂μ ↔ f =ᵐ[μ] g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ Iff (Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fu …
  -/
  refine ⟨fun h_le ↦ EventuallyEq.symm ?_, fun h ↦ integral_congr_ae h⟩
  rw [← sub_ae_eq_zero,
    ← integral_eq_zero_iff_of_nonneg_ae ((sub_nonneg_ae _ _).mpr hfg) (hg.sub hf)]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    h_le : Eq (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fu …
    ⊢ Eq (MeasureTheory.integral μ fun x => HSub.hSub g f x) 0
  -/
  simpa [Pi.sub_apply, integral_sub hg hf, sub_eq_zero, eq_comm]
  /-
    🎉 no goals
  -/


theorem integral_pos_iff_support_of_nonneg_ae {f : α → ℝ} (hf : 0 ≤ᵐ[μ] f) (hfi : Integrable f μ) :
    (0 < ∫ x, f x ∂μ) ↔ 0 < μ (Function.support f) := by
  simp_rw [(integral_nonneg_of_ae hf).lt_iff_ne, pos_iff_ne_zero, Ne, @eq_comm ℝ 0,
    integral_eq_zero_iff_of_nonneg_ae hf hfi, Filter.EventuallyEq, ae_iff, Pi.zero_apply,
    Function.support]


theorem integral_pos_iff_support_of_nonneg {f : α → ℝ} (hf : 0 ≤ f) (hfi : Integrable f μ) :
    (0 < ∫ x, f x ∂μ) ↔ 0 < μ (Function.support f) :=
  integral_pos_iff_support_of_nonneg_ae (Eventually.of_forall hf) hfi


lemma integral_exp_pos {μ : Measure α} {f : α → ℝ} [hμ : NeZero μ]
    (hf : Integrable (fun x ↦ Real.exp (f x)) μ) :
    0 < ∫ x, Real.exp (f x) ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hμ : NeZero μ
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    ⊢ LT.lt 0 (MeasureTheory.integral μ fun x => Real.exp (f x))
  -/
  rw [integral_pos_iff_support_of_nonneg (fun x ↦ (Real.exp_pos _).le) hf]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hμ : NeZero μ
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    ⊢ LT.lt 0 (μ (Function.support fun x => Real.exp (f x)))
  -/
  suffices (Function.support fun x ↦ Real.exp (f x)) = Set.univ by simp [this, hμ.out]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hμ : NeZero μ
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    ⊢ Eq (Function.support fun x => Real.exp (f x)) Set.univ
  -/
  ext1 x
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hμ : NeZero μ
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    x : α
    ⊢ Iff (Membership.mem (Function.support fun x => Real.exp (f x)) x) (Membershi …
  -/
  simp only [Function.mem_support, ne_eq, (Real.exp_pos _).ne', not_false_eq_true, Set.mem_univ]
  /-
    🎉 no goals
  -/


/-- Monotone convergence theorem for real-valued functions and Bochner integrals -/
lemma integral_tendsto_of_tendsto_of_monotone {μ : Measure α} {f : ℕ → α → ℝ} {F : α → ℝ}
    (hf : ∀ n, Integrable (f n) μ) (hF : Integrable F μ) (h_mono : ∀ᵐ x ∂μ, Monotone fun n ↦ f n x)
    (h_tendsto : ∀ᵐ x ∂μ, Tendsto (fun n ↦ f n x) atTop (𝓝 (F x))) :
    Tendsto (fun n ↦ ∫ x, f n x ∂μ) atTop (𝓝 (∫ x, F x ∂μ)) := by
  -- switch from the Bochner to the Lebesgue integral
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => f n x) Filter.atT …
  -/
  let f' := fun n x ↦ f n x - f 0 x
  have hf'_nonneg : ∀ᵐ x ∂μ, ∀ n, 0 ≤ f' n x := by
    filter_upwards [h_mono] with a ha n
    simp [f', ha (zero_le n)]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
    hf'_nonneg : Filter.Eventually (fun x => ∀ (n : Nat), LE.le 0 (f' n x)) (Measu …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => f n x) Filter.atT …
  -/
  have hf'_meas : ∀ n, Integrable (f' n) μ := fun n ↦ (hf n).sub (hf 0)
  suffices Tendsto (fun n ↦ ∫ x, f' n x ∂μ) atTop (𝓝 (∫ x, (F - f 0) x ∂μ)) by
    simp_rw [f', integral_sub (hf _) (hf _), integral_sub' hF (hf 0),
      tendsto_sub_const_iff] at this
    exact this
  have hF_ge : 0 ≤ᵐ[μ] fun x ↦ (F - f 0) x := by
    filter_upwards [h_tendsto, h_mono] with x hx_tendsto hx_mono
    simp only [Pi.zero_apply, Pi.sub_apply, sub_nonneg]
    exact ge_of_tendsto' hx_tendsto (fun n ↦ hx_mono (zero_le _))
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
    hf'_nonneg : Filter.Eventually (fun x => ∀ (n : Nat), LE.le 0 (f' n x)) (Measu …
    hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
    hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => f' n x) Filter.at …
  -/
  rw [ae_all_iff] at hf'_nonneg
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
    hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
    hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
    hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => f' n x) Filter.at …
  -/
  simp_rw [integral_eq_lintegral_of_nonneg_ae (hf'_nonneg _) (hf'_meas _).1]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
    hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
    hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
    hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral μ fun a => ENNReal.ofReal  …
  -/
  rw [integral_eq_lintegral_of_nonneg_ae hF_ge (hF.1.sub (hf 0).1)]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
    hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
    hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
    hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral μ fun a => ENNReal.ofReal  …
  -/
  have h_cont := ENNReal.continuousAt_toReal (x := ∫⁻ a, ENNReal.ofReal ((F - f 0) a) ∂μ) ?_
  /-
    case refine_2
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
    hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
    hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
    hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
    h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral μ fun a => ENNReal.ofReal  …
  -/
  swap
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      ⊢ Ne (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HSub.hSub F (f 0) a)) …
    -/
  · rw [← ofReal_integral_eq_lintegral_ofReal (hF.sub (hf 0)) hF_ge]
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      ⊢ Ne (ENNReal.ofReal (MeasureTheory.integral μ fun x => HSub.hSub F (f 0) x))  …
    -/
    exact ENNReal.ofReal_ne_top
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
    hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
    hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
    hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
    h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.lintegral μ fun a => ENNReal.ofReal  …
  -/
  refine h_cont.tendsto.comp ?_
  -- use the result for the Lebesgue integral
  /-
    case refine_2
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
    hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
    hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
    hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
    h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun a => ENNReal.ofReal ( …
  -/
  refine lintegral_tendsto_of_tendsto_of_monotone ?_ ?_ ?_
    /-
      case refine_2.refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      ⊢ ∀ (n : Nat), AEMeasurable (fun a => ENNReal.ofReal (f' n a)) μ
    -/
  · exact fun n ↦ ((hf n).sub (hf 0)).aemeasurable.ennreal_ofReal
    /-
      🎉 no goals
    -/
    /-
      case refine_2.refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      ⊢ Filter.Eventually (fun x => Monotone fun n => ENNReal.ofReal (f' n x)) (Meas …
    -/
  · filter_upwards [h_mono] with x hx n m hnm
    /-
      case h
      α : Type u_1
      m✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      x : α
      hx : Monotone fun n => f n x
      n m : Nat
      hnm : LE.le n m
      ⊢ LE.le ((fun n => ENNReal.ofReal (f' n x)) n) ((fun n => ENNReal.ofReal (f' n …
    -/
    refine ENNReal.ofReal_le_ofReal ?_
    /-
      case h
      α : Type u_1
      m✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      x : α
      hx : Monotone fun n => f n x
      n m : Nat
      hnm : LE.le n m
      ⊢ LE.le (f' n x) (f' m x)
    -/
    simp only [f', tsub_le_iff_right, sub_add_cancel]
    /-
      case h
      α : Type u_1
      m✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      x : α
      hx : Monotone fun n => f n x
      n m : Nat
      hnm : LE.le n m
      ⊢ LE.le (f n x) (f m x)
    -/
    exact hx hnm
    /-
      🎉 no goals
    -/
    /-
      case refine_2.refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => ENNReal.ofReal (f' n x) …
    -/
  · filter_upwards [h_tendsto] with x hx
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      x : α
      hx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (F x))
      ⊢ Filter.Tendsto (fun n => ENNReal.ofReal (f' n x)) Filter.atTop (nhds (ENNRea …
    -/
    refine (ENNReal.continuous_ofReal.tendsto _).comp ?_
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      x : α
      hx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (F x))
      ⊢ Filter.Tendsto (fun n => f' n x) Filter.atTop (nhds (HSub.hSub F (f 0) x))
    -/
    simp only [Pi.sub_apply]
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      f' : Nat → α → Real := fun n x => HSub.hSub (f n x) (f 0 x)
      hf'_nonneg : ∀ (i : Nat), Filter.Eventually (fun a => LE.le 0 (f' i a)) (Measu …
      hf'_meas : ∀ (n : Nat), MeasureTheory.Integrable (f' n) μ
      hF_ge : (MeasureTheory.ae μ).EventuallyLE 0 fun x => HSub.hSub F (f 0) x
      h_cont : ContinuousAt ENNReal.toReal (MeasureTheory.lintegral μ fun a => ENNRe …
      x : α
      hx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (F x))
      ⊢ Filter.Tendsto (fun n => f' n x) Filter.atTop (nhds (HSub.hSub (F x) (f 0 x)))
    -/
    exact Tendsto.sub hx tendsto_const_nhds
    /-
      🎉 no goals
    -/


/-- Monotone convergence theorem for real-valued functions and Bochner integrals -/
lemma integral_tendsto_of_tendsto_of_antitone {μ : Measure α} {f : ℕ → α → ℝ} {F : α → ℝ}
    (hf : ∀ n, Integrable (f n) μ) (hF : Integrable F μ) (h_mono : ∀ᵐ x ∂μ, Antitone fun n ↦ f n x)
    (h_tendsto : ∀ᵐ x ∂μ, Tendsto (fun n ↦ f n x) atTop (𝓝 (F x))) :
    Tendsto (fun n ↦ ∫ x, f n x ∂μ) atTop (𝓝 (∫ x, F x ∂μ)) := by
  suffices Tendsto (fun n ↦ ∫ x, -f n x ∂μ) atTop (𝓝 (∫ x, -F x ∂μ)) by
    suffices Tendsto (fun n ↦ ∫ x, - -f n x ∂μ) atTop (𝓝 (∫ x, - -F x ∂μ)) by
      simpa [neg_neg] using this
    convert this.neg <;> rw [integral_neg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF : MeasureTheory.Integrable F μ
    h_mono : Filter.Eventually (fun x => Antitone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => Neg.neg (f n x))  …
  -/
  refine integral_tendsto_of_tendsto_of_monotone (fun n ↦ (hf n).neg) hF.neg ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Antitone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      ⊢ Filter.Eventually (fun x => Monotone fun n => Neg.neg (f n x)) (MeasureTheor …
    -/
  · filter_upwards [h_mono] with x hx n m hnm using neg_le_neg_iff.mpr <| hx hnm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF : MeasureTheory.Integrable F μ
      h_mono : Filter.Eventually (fun x => Antitone fun n => f n x) (MeasureTheory.a …
      h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
      ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => Neg.neg (f n x)) Filter …
    -/
  · filter_upwards [h_tendsto] with x hx using hx.neg
    /-
      🎉 no goals
    -/


/-- If a monotone sequence of functions has an upper bound and the sequence of integrals of these
functions tends to the integral of the upper bound, then the sequence of functions converges
almost everywhere to the upper bound. -/
lemma tendsto_of_integral_tendsto_of_monotone {μ : Measure α} {f : ℕ → α → ℝ} {F : α → ℝ}
    (hf_int : ∀ n, Integrable (f n) μ) (hF_int : Integrable F μ)
    (hf_tendsto : Tendsto (fun i ↦ ∫ a, f i a ∂μ) atTop (𝓝 (∫ a, F a ∂μ)))
    (hf_mono : ∀ᵐ a ∂μ, Monotone (fun i ↦ f i a))
    (hf_bound : ∀ᵐ a ∂μ, ∀ i, f i a ≤ F a) :
    ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) atTop (𝓝 (F a)) := by
  -- reduce to the `ℝ≥0∞` case
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  let f' : ℕ → α → ℝ≥0∞ := fun n a ↦ ENNReal.ofReal (f n a - f 0 a)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
    f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  let F' : α → ℝ≥0∞ := fun a ↦ ENNReal.ofReal (F a - f 0 a)
  have hf'_int_eq : ∀ i, ∫⁻ a, f' i a ∂μ = ENNReal.ofReal (∫ a, f i a ∂μ - ∫ a, f 0 a ∂μ) := by
    intro i
    unfold f'
    rw [← ofReal_integral_eq_lintegral_ofReal, integral_sub (hf_int i) (hf_int 0)]
    · exact (hf_int i).sub (hf_int 0)
    · filter_upwards [hf_mono] with a h_mono
      simp [h_mono (zero_le i)]
  have hF'_int_eq : ∫⁻ a, F' a ∂μ = ENNReal.ofReal (∫ a, F a ∂μ - ∫ a, f 0 a ∂μ) := by
    unfold F'
    rw [← ofReal_integral_eq_lintegral_ofReal, integral_sub hF_int (hf_int 0)]
    · exact hF_int.sub (hf_int 0)
    · filter_upwards [hf_bound] with a h_bound
      simp [h_bound 0]
  have h_tendsto : Tendsto (fun i ↦ ∫⁻ a, f' i a ∂μ) atTop (𝓝 (∫⁻ a, F' a ∂μ)) := by
    simp_rw [hf'_int_eq, hF'_int_eq]
    refine (ENNReal.continuous_ofReal.tendsto _).comp ?_
    rwa [tendsto_sub_const_iff]
  have h_mono : ∀ᵐ a ∂μ, Monotone (fun i ↦ f' i a) := by
    filter_upwards [hf_mono] with a ha_mono i j hij
    refine ENNReal.ofReal_le_ofReal ?_
    simp [ha_mono hij]
  have h_bound : ∀ᵐ a ∂μ, ∀ i, f' i a ≤ F' a := by
    filter_upwards [hf_bound] with a ha_bound i
    refine ENNReal.ofReal_le_ofReal ?_
    simp only [tsub_le_iff_right, sub_add_cancel, ha_bound i]
  -- use the corresponding lemma for `ℝ≥0∞`
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
    f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
    F' : α → ENNReal := fun a => ENNReal.ofReal (HSub.hSub (F a) (f 0 a))
    hf'_int_eq : ∀ (i : Nat), Eq (MeasureTheory.lintegral μ fun a => f' i a) (ENNR …
    hF'_int_eq : Eq (MeasureTheory.lintegral μ fun a => F' a) (ENNReal.ofReal (HSu …
    h_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f' i a …
    h_mono : Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (Mea …
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  have h := tendsto_of_lintegral_tendsto_of_monotone ?_ h_tendsto h_mono h_bound ?_
  /-
    case refine_3
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
    f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
    F' : α → ENNReal := fun a => ENNReal.ofReal (HSub.hSub (F a) (f 0 a))
    hf'_int_eq : ∀ (i : Nat), Eq (MeasureTheory.lintegral μ fun a => f' i a) (ENNR …
    hF'_int_eq : Eq (MeasureTheory.lintegral μ fun a => F' a) (ENNReal.ofReal (HSu …
    h_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f' i a …
    h_mono : Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (Mea …
    h : Filter.Eventually (fun a => Filter.Tendsto (fun i => f' i a) Filter.atTop  …
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  rotate_left
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF_int : MeasureTheory.Integrable F μ
      hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
      hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
      hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
      f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
      F' : α → ENNReal := fun a => ENNReal.ofReal (HSub.hSub (F a) (f 0 a))
      hf'_int_eq : ∀ (i : Nat), Eq (MeasureTheory.lintegral μ fun a => f' i a) (ENNR …
      hF'_int_eq : Eq (MeasureTheory.lintegral μ fun a => F' a) (ENNReal.ofReal (HSu …
      h_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f' i a …
      h_mono : Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory. …
      h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (Mea …
      ⊢ AEMeasurable F' μ
    -/
  · exact (hF_int.1.aemeasurable.sub (hf_int 0).1.aemeasurable).ennreal_ofReal
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF_int : MeasureTheory.Integrable F μ
      hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
      hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
      hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
      f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
      F' : α → ENNReal := fun a => ENNReal.ofReal (HSub.hSub (F a) (f 0 a))
      hf'_int_eq : ∀ (i : Nat), Eq (MeasureTheory.lintegral μ fun a => f' i a) (ENNR …
      hF'_int_eq : Eq (MeasureTheory.lintegral μ fun a => F' a) (ENNReal.ofReal (HSu …
      h_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f' i a …
      h_mono : Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory. …
      h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (Mea …
      ⊢ Ne (MeasureTheory.lintegral μ fun a => F' a) Top.top
    -/
  · exact ((lintegral_ofReal_le_lintegral_nnnorm _).trans_lt (hF_int.sub (hf_int 0)).2).ne
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
    f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
    F' : α → ENNReal := fun a => ENNReal.ofReal (HSub.hSub (F a) (f 0 a))
    hf'_int_eq : ∀ (i : Nat), Eq (MeasureTheory.lintegral μ fun a => f' i a) (ENNR …
    hF'_int_eq : Eq (MeasureTheory.lintegral μ fun a => F' a) (ENNReal.ofReal (HSu …
    h_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f' i a …
    h_mono : Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (Mea …
    h : Filter.Eventually (fun a => Filter.Tendsto (fun i => f' i a) Filter.atTop  …
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  filter_upwards [h, hf_mono, hf_bound] with a ha ha_mono ha_bound
  have h1 : (fun i ↦ f i a) = fun i ↦ (f' i a).toReal + f 0 a := by
    unfold f'
    ext i
    rw [ENNReal.toReal_ofReal]
    · abel
    · simp [ha_mono (zero_le i)]
  have h2 : F a = (F' a).toReal + f 0 a := by
    unfold F'
    rw [ENNReal.toReal_ofReal]
    · abel
    · simp [ha_bound 0]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
    f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
    F' : α → ENNReal := fun a => ENNReal.ofReal (HSub.hSub (F a) (f 0 a))
    hf'_int_eq : ∀ (i : Nat), Eq (MeasureTheory.lintegral μ fun a => f' i a) (ENNR …
    hF'_int_eq : Eq (MeasureTheory.lintegral μ fun a => F' a) (ENNReal.ofReal (HSu …
    h_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f' i a …
    h_mono : Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (Mea …
    h : Filter.Eventually (fun a => Filter.Tendsto (fun i => f' i a) Filter.atTop  …
    a : α
    ha : Filter.Tendsto (fun i => f' i a) Filter.atTop (nhds (F' a))
    ha_mono : Monotone fun i => f i a
    ha_bound : ∀ (i : Nat), LE.le (f i a) (F a)
    h1 : Eq (fun i => f i a) fun i => HAdd.hAdd (f' i a).toReal (f 0 a)
    h2 : Eq (F a) (HAdd.hAdd (F' a).toReal (f 0 a))
    ⊢ Filter.Tendsto (fun i => f i a) Filter.atTop (nhds (F a))
  -/
  rw [h1, h2]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
    f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
    F' : α → ENNReal := fun a => ENNReal.ofReal (HSub.hSub (F a) (f 0 a))
    hf'_int_eq : ∀ (i : Nat), Eq (MeasureTheory.lintegral μ fun a => f' i a) (ENNR …
    hF'_int_eq : Eq (MeasureTheory.lintegral μ fun a => F' a) (ENNReal.ofReal (HSu …
    h_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f' i a …
    h_mono : Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (Mea …
    h : Filter.Eventually (fun a => Filter.Tendsto (fun i => f' i a) Filter.atTop  …
    a : α
    ha : Filter.Tendsto (fun i => f' i a) Filter.atTop (nhds (F' a))
    ha_mono : Monotone fun i => f i a
    ha_bound : ∀ (i : Nat), LE.le (f i a) (F a)
    h1 : Eq (fun i => f i a) fun i => HAdd.hAdd (f' i a).toReal (f 0 a)
    h2 : Eq (F a) (HAdd.hAdd (F' a).toReal (f 0 a))
    ⊢ Filter.Tendsto (fun i => HAdd.hAdd (f' i a).toReal (f 0 a)) Filter.atTop (nh …
  -/
  refine Filter.Tendsto.add ?_ tendsto_const_nhds
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Meas …
    f' : Nat → α → ENNReal := fun n a => ENNReal.ofReal (HSub.hSub (f n a) (f 0 a))
    F' : α → ENNReal := fun a => ENNReal.ofReal (HSub.hSub (F a) (f 0 a))
    hf'_int_eq : ∀ (i : Nat), Eq (MeasureTheory.lintegral μ fun a => f' i a) (ENNR …
    hF'_int_eq : Eq (MeasureTheory.lintegral μ fun a => F' a) (ENNReal.ofReal (HSu …
    h_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f' i a …
    h_mono : Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (Mea …
    h : Filter.Eventually (fun a => Filter.Tendsto (fun i => f' i a) Filter.atTop  …
    a : α
    ha : Filter.Tendsto (fun i => f' i a) Filter.atTop (nhds (F' a))
    ha_mono : Monotone fun i => f i a
    ha_bound : ∀ (i : Nat), LE.le (f i a) (F a)
    h1 : Eq (fun i => f i a) fun i => HAdd.hAdd (f' i a).toReal (f 0 a)
    h2 : Eq (F a) (HAdd.hAdd (F' a).toReal (f 0 a))
    ⊢ Filter.Tendsto (fun i => (f' i a).toReal) Filter.atTop (nhds (F' a).toReal)
  -/
  exact (ENNReal.continuousAt_toReal ENNReal.ofReal_ne_top).tendsto.comp ha
  /-
    🎉 no goals
  -/


/-- If an antitone sequence of functions has a lower bound and the sequence of integrals of these
functions tends to the integral of the lower bound, then the sequence of functions converges
almost everywhere to the lower bound. -/
lemma tendsto_of_integral_tendsto_of_antitone {μ : Measure α} {f : ℕ → α → ℝ} {F : α → ℝ}
    (hf_int : ∀ n, Integrable (f n) μ) (hF_int : Integrable F μ)
    (hf_tendsto : Tendsto (fun i ↦ ∫ a, f i a ∂μ) atTop (𝓝 (∫ a, F a ∂μ)))
    (hf_mono : ∀ᵐ a ∂μ, Antitone (fun i ↦ f i a))
    (hf_bound : ∀ᵐ a ∂μ, ∀ i, F a ≤ f i a) :
    ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) atTop (𝓝 (F a)) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  let f' : ℕ → α → ℝ := fun i a ↦ - f i a
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
    f' : Nat → α → Real := fun i a => Neg.neg (f i a)
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  let F' : α → ℝ := fun a ↦ - F a
  suffices ∀ᵐ a ∂μ, Tendsto (fun i ↦ f' i a) atTop (𝓝 (F' a)) by
    filter_upwards [this] with a ha_tendsto
    convert ha_tendsto.neg
    · simp [f']
    · simp [F']
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → Real
    F : α → Real
    hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hF_int : MeasureTheory.Integrable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
    hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
    hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
    f' : Nat → α → Real := fun i a => Neg.neg (f i a)
    F' : α → Real := fun a => Neg.neg (F a)
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f' i a) Filter.atTop (n …
  -/
  refine tendsto_of_integral_tendsto_of_monotone (fun n ↦ (hf_int n).neg) hF_int.neg ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF_int : MeasureTheory.Integrable F μ
      hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
      hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
      hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
      f' : Nat → α → Real := fun i a => Neg.neg (f i a)
      F' : α → Real := fun a => Neg.neg (F a)
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f' i a) Filter.at …
    -/
  · convert hf_tendsto.neg
      /-
        case h.e'_3.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : Nat → α → Real
        F : α → Real
        hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
        hF_int : MeasureTheory.Integrable F μ
        hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
        hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
        hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
        f' : Nat → α → Real := fun i a => Neg.neg (f i a)
        F' : α → Real := fun a => Neg.neg (F a)
        x✝ : Nat
        ⊢ Eq (MeasureTheory.integral μ fun a => f' x✝ a) (Neg.neg (MeasureTheory.integ …
      -/
    · rw [integral_neg]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h.e'_3
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : Nat → α → Real
        F : α → Real
        hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
        hF_int : MeasureTheory.Integrable F μ
        hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
        hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
        hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
        f' : Nat → α → Real := fun i a => Neg.neg (f i a)
        F' : α → Real := fun a => Neg.neg (F a)
        ⊢ Eq (MeasureTheory.integral μ fun a => F' a) (Neg.neg (MeasureTheory.integral …
      -/
    · rw [integral_neg]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF_int : MeasureTheory.Integrable F μ
      hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
      hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
      hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
      f' : Nat → α → Real := fun i a => Neg.neg (f i a)
      F' : α → Real := fun a => Neg.neg (F a)
      ⊢ Filter.Eventually (fun a => Monotone fun i => f' i a) (MeasureTheory.ae μ)
    -/
  · filter_upwards [hf_mono] with a ha i j hij
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF_int : MeasureTheory.Integrable F μ
      hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
      hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
      hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
      f' : Nat → α → Real := fun i a => Neg.neg (f i a)
      F' : α → Real := fun a => Neg.neg (F a)
      a : α
      ha : Antitone fun i => f i a
      i j : Nat
      hij : LE.le i j
      ⊢ LE.le ((fun i => f' i a) i) ((fun i => f' i a) j)
    -/
    simp [f', ha hij]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF_int : MeasureTheory.Integrable F μ
      hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
      hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
      hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
      f' : Nat → α → Real := fun i a => Neg.neg (f i a)
      F' : α → Real := fun a => Neg.neg (F a)
      ⊢ Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f' i a) (F' a)) (MeasureTheo …
    -/
  · filter_upwards [hf_bound] with a ha i
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → Real
      F : α → Real
      hf_int : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hF_int : MeasureTheory.Integrable F μ
      hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => f i a) …
      hf_mono : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory. …
      hf_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (F a) (f i a)) (Meas …
      f' : Nat → α → Real := fun i a => Neg.neg (f i a)
      F' : α → Real := fun a => Neg.neg (F a)
      a : α
      ha : ∀ (i : Nat), LE.le (F a) (f i a)
      i : Nat
      ⊢ LE.le (f' i a) (F' a)
    -/
    simp [f', F', ha i]
    /-
      🎉 no goals
    -/


theorem L1.norm_eq_integral_norm (f : α →₁[μ] H) : ‖f‖ = ∫ a, ‖f a‖ ∂μ := by
  simp only [eLpNorm, eLpNorm'_eq_lintegral_nnnorm, ENNReal.one_toReal, ENNReal.rpow_one,
    Lp.norm_def, if_false, ENNReal.one_ne_top, one_ne_zero, _root_.div_one]
  rw [integral_eq_lintegral_of_nonneg_ae (Eventually.of_forall (by simp [norm_nonneg]))
      (Lp.aestronglyMeasurable f).norm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp H 1 μ) x
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑f a))).toReal (Mea …
  -/
  simp [ofReal_norm_eq_coe_nnnorm]
  /-
    🎉 no goals
  -/


theorem L1.dist_eq_integral_dist (f g : α →₁[μ] H) : dist f g = ∫ a, dist (f a) (g a) ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp H 1 μ) x
    ⊢ Eq (Dist.dist f g) (MeasureTheory.integral μ fun a => Dist.dist (↑↑f a) (↑↑g …
  -/
  simp only [dist_eq_norm, L1.norm_eq_integral_norm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp H 1 μ) x
    ⊢ Eq (MeasureTheory.integral μ fun a => Norm.norm (↑↑(HSub.hSub f g) a)) (Meas …
  -/
  exact integral_congr_ae <| (Lp.coeFn_sub _ _).fun_comp norm
  /-
    🎉 no goals
  -/


theorem L1.norm_of_fun_eq_integral_norm {f : α → H} (hf : Integrable f μ) :
    ‖hf.toL1 f‖ = ∫ a, ‖f a‖ ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f : α → H
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (Norm.norm (MeasureTheory.Integrable.toL1 f hf)) (MeasureTheory.integral  …
  -/
  rw [L1.norm_eq_integral_norm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f : α → H
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun a => Norm.norm (↑↑(MeasureTheory.Integrable …
  -/
  exact integral_congr_ae <| hf.coeFn_toL1.fun_comp _
  /-
    🎉 no goals
  -/


theorem Memℒp.eLpNorm_eq_integral_rpow_norm {f : α → H} {p : ℝ≥0∞} (hp1 : p ≠ 0) (hp2 : p ≠ ∞)
    (hf : Memℒp f p μ) :
    eLpNorm f p μ = ENNReal.ofReal ((∫ a, ‖f a‖ ^ p.toReal ∂μ) ^ p.toReal⁻¹) := by
  have A : ∫⁻ a : α, ENNReal.ofReal (‖f a‖ ^ p.toReal) ∂μ = ∫⁻ a : α, ‖f a‖₊ ^ p.toReal ∂μ := by
    simp_rw [← ofReal_rpow_of_nonneg (norm_nonneg _) toReal_nonneg, ofReal_norm_eq_coe_nnnorm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f : α → H
    p : ENNReal
    hp1 : Ne p 0
    hp2 : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    A : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (Norm.nor …
    ⊢ Eq (MeasureTheory.eLpNorm f p μ) (ENNReal.ofReal (HPow.hPow (MeasureTheory.i …
  -/
  simp only [eLpNorm_eq_lintegral_rpow_nnnorm hp1 hp2, one_div]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f : α → H
    p : ENNReal
    hp1 : Ne p 0
    hp2 : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    A : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (Norm.nor …
    ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnnorm …
  -/
  rw [integral_eq_lintegral_of_nonneg_ae]; rotate_left
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      H : Type u_7
      inst✝ : NormedAddCommGroup H
      f : α → H
      p : ENNReal
      hp1 : Ne p 0
      hp2 : Ne p Top.top
      hf : MeasureTheory.Memℒp f p μ
      A : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (Norm.nor …
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun a => HPow.hPow (Norm.norm (f a)) p.t …
    -/
  · exact ae_of_all _ fun x => by positivity
    /-
      🎉 no goals
    -/
    /-
      case hfm
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      H : Type u_7
      inst✝ : NormedAddCommGroup H
      f : α → H
      p : ENNReal
      hp1 : Ne p 0
      hp2 : Ne p Top.top
      hf : MeasureTheory.Memℒp f p μ
      A : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (Norm.nor …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HPow.hPow (Norm.norm (f a)) p.t …
    -/
  · exact (hf.aestronglyMeasurable.norm.aemeasurable.pow_const _).aestronglyMeasurable
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f : α → H
    p : ENNReal
    hp1 : Ne p 0
    hp2 : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    A : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (Norm.nor …
    ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnnorm …
  -/
  rw [A, ← ofReal_rpow_of_nonneg toReal_nonneg (inv_nonneg.2 toReal_nonneg), ofReal_toReal]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    H : Type u_7
    inst✝ : NormedAddCommGroup H
    f : α → H
    p : ENNReal
    hp1 : Ne p 0
    hp2 : Ne p Top.top
    hf : MeasureTheory.Memℒp f p μ
    A : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (Norm.nor …
    ⊢ Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f a))) p. …
  -/
  exact (lintegral_rpow_nnnorm_lt_top_of_eLpNorm_lt_top hp1 hp2 hf.2).ne
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_eq_integral_rpow_norm := Memℒp.eLpNorm_eq_integral_rpow_norm


theorem integral_mono_ae {f g : α → ℝ} (hf : Integrable f μ) (hg : Integrable g μ) (h : f ≤ᵐ[μ] g) :
    ∫ a, f a ∂μ ≤ ∫ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    h : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun  …
  -/
  have A : CompleteSpace ℝ := by infer_instance
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    h : (MeasureTheory.ae μ).EventuallyLE f g
    A : CompleteSpace Real
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun  …
  -/
  simp only [integral, A, L1.integral]
  exact setToFun_mono (dominatedFinMeasAdditive_weightedSMul μ)
    (fun s _ _ => weightedSMul_nonneg s) hf hg h


@[gcongr, mono]
theorem integral_mono {f g : α → ℝ} (hf : Integrable f μ) (hg : Integrable g μ) (h : f ≤ g) :
    ∫ a, f a ∂μ ≤ ∫ a, g a ∂μ :=
  integral_mono_ae hf hg <| Eventually.of_forall h


theorem integral_mono_of_nonneg {f g : α → ℝ} (hf : 0 ≤ᵐ[μ] f) (hgi : Integrable g μ)
    (h : f ≤ᵐ[μ] g) : ∫ a, f a ∂μ ≤ ∫ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    hgi : MeasureTheory.Integrable g μ
    h : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun  …
  -/
  by_cases hfm : AEStronglyMeasurable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hgi : MeasureTheory.Integrable g μ
      h : (MeasureTheory.ae μ).EventuallyLE f g
      hfm : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ LE.le (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun  …
    -/
  · refine integral_mono_ae ⟨hfm, ?_⟩ hgi h
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hgi : MeasureTheory.Integrable g μ
      h : (MeasureTheory.ae μ).EventuallyLE f g
      hfm : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ MeasureTheory.HasFiniteIntegral f μ
    -/
    refine hgi.hasFiniteIntegral.mono <| h.mp <| hf.mono fun x hf hfg => ?_
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      hf✝ : (MeasureTheory.ae μ).EventuallyLE 0 f
      hgi : MeasureTheory.Integrable g μ
      h : (MeasureTheory.ae μ).EventuallyLE f g
      hfm : MeasureTheory.AEStronglyMeasurable f μ
      x : α
      hf : LE.le (0 x) (f x)
      hfg : LE.le (f x) (g x)
      ⊢ LE.le (Norm.norm (f x)) (Norm.norm (g x))
    -/
    simpa [abs_of_nonneg hf, abs_of_nonneg (le_trans hf hfg)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hgi : MeasureTheory.Integrable g μ
      h : (MeasureTheory.ae μ).EventuallyLE f g
      hfm : Not (MeasureTheory.AEStronglyMeasurable f μ)
      ⊢ LE.le (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral μ fun  …
    -/
  · rw [integral_non_aestronglyMeasurable hfm]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hgi : MeasureTheory.Integrable g μ
      h : (MeasureTheory.ae μ).EventuallyLE f g
      hfm : Not (MeasureTheory.AEStronglyMeasurable f μ)
      ⊢ LE.le 0 (MeasureTheory.integral μ fun a => g a)
    -/
    exact integral_nonneg_of_ae (hf.trans h)
    /-
      🎉 no goals
    -/


theorem integral_mono_measure {f : α → ℝ} {ν} (hle : μ ≤ ν) (hf : 0 ≤ᵐ[ν] f)
    (hfi : Integrable f ν) : ∫ a, f a ∂μ ≤ ∫ a, f a ∂ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    hle : LE.le μ ν
    hf : (MeasureTheory.ae ν).EventuallyLE 0 f
    hfi : MeasureTheory.Integrable f ν
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral ν fun  …
  -/
  have hfi' : Integrable f μ := hfi.mono_measure hle
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    hle : LE.le μ ν
    hf : (MeasureTheory.ae ν).EventuallyLE 0 f
    hfi : MeasureTheory.Integrable f ν
    hfi' : MeasureTheory.Integrable f μ
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral ν fun  …
  -/
  have hf' : 0 ≤ᵐ[μ] f := hle.absolutelyContinuous hf
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    hle : LE.le μ ν
    hf : (MeasureTheory.ae ν).EventuallyLE 0 f
    hfi : MeasureTheory.Integrable f ν
    hfi' : MeasureTheory.Integrable f μ
    hf' : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ LE.le (MeasureTheory.integral μ fun a => f a) (MeasureTheory.integral ν fun  …
  -/
  rw [integral_eq_lintegral_of_nonneg_ae hf' hfi'.1, integral_eq_lintegral_of_nonneg_ae hf hfi.1]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    hle : LE.le μ ν
    hf : (MeasureTheory.ae ν).EventuallyLE 0 f
    hfi : MeasureTheory.Integrable f ν
    hfi' : MeasureTheory.Integrable f μ
    hf' : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)).toReal (Meas …
  -/
  refine ENNReal.toReal_mono ?_ (lintegral_mono' hle le_rfl)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    hle : LE.le μ ν
    hf : (MeasureTheory.ae ν).EventuallyLE 0 f
    hfi : MeasureTheory.Integrable f ν
    hfi' : MeasureTheory.Integrable f μ
    hf' : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Ne (MeasureTheory.lintegral ν fun a => ENNReal.ofReal (f a)) Top.top
  -/
  exact ((hasFiniteIntegral_iff_ofReal hf).1 hfi.2).ne
  /-
    🎉 no goals
  -/


theorem norm_integral_le_integral_norm (f : α → G) : ‖∫ a, f a ∂μ‖ ≤ ∫ a, ‖f a‖ ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.int …
  -/
  have le_ae : ∀ᵐ a ∂μ, 0 ≤ ‖f a‖ := Eventually.of_forall fun a => norm_nonneg _
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    le_ae : Filter.Eventually (fun a => LE.le 0 (Norm.norm (f a))) (MeasureTheory. …
    ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.int …
  -/
  by_cases h : AEStronglyMeasurable f μ
  · calc
      ‖∫ a, f a ∂μ‖ ≤ ENNReal.toReal (∫⁻ a, ENNReal.ofReal ‖f a‖ ∂μ) :=
        norm_integral_le_lintegral_norm _
      _ = ∫ a, ‖f a‖ ∂μ := (integral_eq_lintegral_of_nonneg_ae le_ae <| h.norm).symm
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      le_ae : Filter.Eventually (fun a => LE.le 0 (Norm.norm (f a))) (MeasureTheory. …
      h : Not (MeasureTheory.AEStronglyMeasurable f μ)
      ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun a => f a)) (MeasureTheory.int …
    -/
  · rw [integral_non_aestronglyMeasurable h, norm_zero]
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      le_ae : Filter.Eventually (fun a => LE.le 0 (Norm.norm (f a))) (MeasureTheory. …
      h : Not (MeasureTheory.AEStronglyMeasurable f μ)
      ⊢ LE.le 0 (MeasureTheory.integral μ fun a => Norm.norm (f a))
    -/
    exact integral_nonneg_of_ae le_ae
    /-
      🎉 no goals
    -/


theorem norm_integral_le_of_norm_le {f : α → G} {g : α → ℝ} (hg : Integrable g μ)
    (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ g x) : ‖∫ x, f x ∂μ‖ ≤ ∫ x, g x ∂μ :=
  calc
    ‖∫ x, f x ∂μ‖ ≤ ∫ x, ‖f x‖ ∂μ := norm_integral_le_integral_norm f
    _ ≤ ∫ x, g x ∂μ := integral_mono_of_nonneg (Eventually.of_forall fun _ => norm_nonneg _) hg h


theorem SimpleFunc.integral_eq_integral (f : α →ₛ E) (hfi : Integrable f μ) :
    f.integral μ = ∫ x, f x ∂μ := by
  rw [MeasureTheory.integral_eq f hfi, ← L1.SimpleFunc.toLp_one_eq_toL1,
    L1.SimpleFunc.integral_L1_eq_integral, L1.SimpleFunc.integral_eq_integral]
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    hfi : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (MeasureTheory.SimpleFunc.integral μ f) (MeasureTheory.SimpleFunc.integra …
  -/
  exact SimpleFunc.integral_congr hfi (Lp.simpleFunc.toSimpleFunc_toLp _ _).symm
  /-
    🎉 no goals
  -/


theorem SimpleFunc.integral_eq_sum (f : α →ₛ E) (hfi : Integrable f μ) :
    ∫ x, f x ∂μ = ∑ x ∈ f.range, ENNReal.toReal (μ (f ⁻¹' {x})) • x := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α E
    hfi : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (f.range.sum fun x => HSMul.hSMul …
  -/
  rw [← f.integral_eq_integral hfi, SimpleFunc.integral, ← SimpleFunc.integral_eq]; rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[simp]
theorem integral_const (c : E) : ∫ _ : α, c ∂μ = (μ univ).toReal • c := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : E
    ⊢ Eq (MeasureTheory.integral μ fun x => c) (HSMul.hSMul (μ Set.univ).toReal c)
  -/
  cases' (@le_top _ _ _ (μ univ)).lt_or_eq with hμ hμ
    /-
      case inl
      α : Type u_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      hE : CompleteSpace E
      inst✝ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      c : E
      hμ : LT.lt (μ Set.univ) Top.top
      ⊢ Eq (MeasureTheory.integral μ fun x => c) (HSMul.hSMul (μ Set.univ).toReal c)
    -/
  · haveI : IsFiniteMeasure μ := ⟨hμ⟩
    /-
      case inl
      α : Type u_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      hE : CompleteSpace E
      inst✝ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      c : E
      hμ : LT.lt (μ Set.univ) Top.top
      this : MeasureTheory.IsFiniteMeasure μ
      ⊢ Eq (MeasureTheory.integral μ fun x => c) (HSMul.hSMul (μ Set.univ).toReal c)
    -/
    simp only [integral, hE, L1.integral]
    /-
      case inl
      α : Type u_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      hE : CompleteSpace E
      inst✝ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      c : E
      hμ : LT.lt (μ Set.univ) Top.top
      this : MeasureTheory.IsFiniteMeasure μ
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun x => c) μ) (fun  …
    -/
    exact setToFun_const (dominatedFinMeasAdditive_weightedSMul _) _
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      hE : CompleteSpace E
      inst✝ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      c : E
      hμ : Eq (μ Set.univ) Top.top
      ⊢ Eq (MeasureTheory.integral μ fun x => c) (HSMul.hSMul (μ Set.univ).toReal c)
    -/
  · by_cases hc : c = 0
      /-
        case pos
        α : Type u_1
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        hE : CompleteSpace E
        inst✝ : NormedSpace Real E
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        c : E
        hμ : Eq (μ Set.univ) Top.top
        hc : Eq c 0
        ⊢ Eq (MeasureTheory.integral μ fun x => c) (HSMul.hSMul (μ Set.univ).toReal c)
      -/
    · simp [hc, integral_zero]
      /-
        🎉 no goals
      -/
    · have : ¬Integrable (fun _ : α => c) μ := by
        simp only [integrable_const_iff, not_or]
        exact ⟨hc, hμ.not_lt⟩
      /-
        case neg
        α : Type u_1
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        hE : CompleteSpace E
        inst✝ : NormedSpace Real E
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        c : E
        hμ : Eq (μ Set.univ) Top.top
        hc : Not (Eq c 0)
        this : Not (MeasureTheory.Integrable (fun x => c) μ)
        ⊢ Eq (MeasureTheory.integral μ fun x => c) (HSMul.hSMul (μ Set.univ).toReal c)
      -/
      simp [integral_undef, *]
      /-
        🎉 no goals
      -/


theorem norm_integral_le_of_norm_le_const [IsFiniteMeasure μ] {f : α → G} {C : ℝ}
    (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) : ‖∫ x, f x ∂μ‖ ≤ C * (μ univ).toReal :=
  calc
    ‖∫ x, f x ∂μ‖ ≤ ∫ _, C ∂μ := norm_integral_le_of_norm_le (integrable_const C) h
                                  /-
                                    α : Type u_1
                                    G : Type u_5
                                    inst✝² : NormedAddCommGroup G
                                    inst✝¹ : NormedSpace Real G
                                    m : MeasurableSpace α
                                    μ : MeasureTheory.Measure α
                                    inst✝ : MeasureTheory.IsFiniteMeasure μ
                                    f : α → G
                                    C : Real
                                    h : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae μ)
                                    ⊢ Eq (MeasureTheory.integral μ fun x => C) (HMul.hMul C (μ Set.univ).toReal)
                                  -/
    _ = C * (μ univ).toReal := by rw [integral_const, smul_eq_mul, mul_comm]
                                  /-
                                    🎉 no goals
                                  -/


theorem tendsto_integral_approxOn_of_measurable [MeasurableSpace E] [BorelSpace E] {f : α → E}
    {s : Set E} [SeparableSpace s] (hfi : Integrable f μ) (hfm : Measurable f)
    (hs : ∀ᵐ x ∂μ, f x ∈ closure s) {y₀ : E} (h₀ : y₀ ∈ s) (h₀i : Integrable (fun _ => y₀) μ) :
    Tendsto (fun n => (SimpleFunc.approxOn f hfm s y₀ h₀ n).integral μ)
      atTop (𝓝 <| ∫ x, f x ∂μ) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝³ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    f : α → E
    s : Set E
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hfi : MeasureTheory.Integrable f μ
    hfm : Measurable f
    hs : Filter.Eventually (fun x => Membership.mem (closure s) (f x)) (MeasureThe …
    y₀ : E
    h₀ : Membership.mem s y₀
    h₀i : MeasureTheory.Integrable (fun x => y₀) μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (MeasureTheory. …
  -/
  have hfi' := SimpleFunc.integrable_approxOn hfm hfi h₀ h₀i
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝³ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    f : α → E
    s : Set E
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hfi : MeasureTheory.Integrable f μ
    hfm : Measurable f
    hs : Filter.Eventually (fun x => Membership.mem (closure s) (f x)) (MeasureThe …
    y₀ : E
    h₀ : Membership.mem s y₀
    h₀i : MeasureTheory.Integrable (fun x => y₀) μ
    hfi' : ∀ (n : Nat), MeasureTheory.Integrable (⇑(MeasureTheory.SimpleFunc.appro …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (MeasureTheory. …
  -/
  simp only [SimpleFunc.integral_eq_integral _ (hfi' _), integral, hE, L1.integral]
  exact tendsto_setToFun_approxOn_of_measurable (dominatedFinMeasAdditive_weightedSMul μ)
    hfi hfm hs h₀ h₀i


theorem tendsto_integral_approxOn_of_measurable_of_range_subset [MeasurableSpace E] [BorelSpace E]
    {f : α → E} (fmeas : Measurable f) (hf : Integrable f μ) (s : Set E) [SeparableSpace s]
    (hs : range f ∪ {0} ⊆ s) :
                                                                 /-
                                                                   α : Type u_1
                                                                   E : Type u_2
                                                                   F : Type u_3
                                                                   𝕜 : Type u_4
                                                                   inst✝¹² : NormedAddCommGroup E
                                                                   hE : CompleteSpace E
                                                                   inst✝¹¹ : NontriviallyNormedField 𝕜
                                                                   inst✝¹⁰ : NormedAddCommGroup F
                                                                   inst✝⁹ : NormedSpace Real F
                                                                   inst✝⁸ : CompleteSpace F
                                                                   G : Type u_5
                                                                   inst✝⁷ : NormedAddCommGroup G
                                                                   inst✝⁶ : NormedSpace Real G
                                                                   inst✝⁵ : NormedSpace Real E
                                                                   f✝ : α → E
                                                                   m : MeasurableSpace α
                                                                   μ : MeasureTheory.Measure α
                                                                   X : Type u_6
                                                                   inst✝⁴ : TopologicalSpace X
                                                                   inst✝³ : FirstCountableTopology X
                                                                   inst✝² : MeasurableSpace E
                                                                   inst✝¹ : BorelSpace E
                                                                   f : α → E
                                                                   fmeas : Measurable f
                                                                   hf : MeasureTheory.Integrable f μ
                                                                   s : Set E
                                                                   inst✝ : TopologicalSpace.SeparableSpace ↑s
                                                                   hs : HasSubset.Subset (Union.union (Set.range f) (Singleton.singleton 0)) s
                                                                   n : Nat
                                                                   ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) 0
                                                                 -/
    Tendsto (fun n => (SimpleFunc.approxOn f fmeas s 0 (hs <| by simp) n).integral μ) atTop
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      (𝓝 <| ∫ x, f x ∂μ) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝³ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    f : α → E
    fmeas : Measurable f
    hf : MeasureTheory.Integrable f μ
    s : Set E
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hs : HasSubset.Subset (Union.union (Set.range f) (Singleton.singleton 0)) s
    ⊢ Filter.Tendsto (fun n => MeasureTheory.SimpleFunc.integral μ (MeasureTheory. …
  -/
  apply tendsto_integral_approxOn_of_measurable hf fmeas _ _ (integrable_zero _ _ _)
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝³ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    f : α → E
    fmeas : Measurable f
    hf : MeasureTheory.Integrable f μ
    s : Set E
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hs : HasSubset.Subset (Union.union (Set.range f) (Singleton.singleton 0)) s
    ⊢ Filter.Eventually (fun x => Membership.mem (closure s) (f x)) (MeasureTheory …
  -/
  exact Eventually.of_forall fun x => subset_closure (hs (Set.mem_union_left _ (mem_range_self _)))
  /-
    🎉 no goals
  -/

-- We redeclare `E` here to temporarily avoid
-- the `[CompleteSpace E]` and `[NormedSpace ℝ E]` instances.

theorem tendsto_integral_norm_approxOn_sub
    {E : Type*} [NormedAddCommGroup E] [MeasurableSpace E] [BorelSpace E] {f : α → E}
    (fmeas : Measurable f) (hf : Integrable f μ) [SeparableSpace (range f ∪ {0} : Set E)] :
                                                                             /-
                                                                               α : Type u_1
                                                                               E✝ : Type u_2
                                                                               F : Type u_3
                                                                               𝕜 : Type u_4
                                                                               inst✝¹³ : NormedAddCommGroup E✝
                                                                               hE : CompleteSpace E✝
                                                                               inst✝¹² : NontriviallyNormedField 𝕜
                                                                               inst✝¹¹ : NormedAddCommGroup F
                                                                               inst✝¹⁰ : NormedSpace Real F
                                                                               inst✝⁹ : CompleteSpace F
                                                                               G : Type u_5
                                                                               inst✝⁸ : NormedAddCommGroup G
                                                                               inst✝⁷ : NormedSpace Real G
                                                                               inst✝⁶ : NormedSpace Real E✝
                                                                               f✝ : α → E✝
                                                                               m : MeasurableSpace α
                                                                               μ : MeasureTheory.Measure α
                                                                               X : Type u_6
                                                                               inst✝⁵ : TopologicalSpace X
                                                                               inst✝⁴ : FirstCountableTopology X
                                                                               E : Type u_7
                                                                               inst✝³ : NormedAddCommGroup E
                                                                               inst✝² : MeasurableSpace E
                                                                               inst✝¹ : BorelSpace E
                                                                               f : α → E
                                                                               fmeas : Measurable f
                                                                               hf : MeasureTheory.Integrable f μ
                                                                               inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
                                                                               n : Nat
                                                                               x : α
                                                                               ⊢ Membership.mem (Union.union (Set.range f) (Singleton.singleton 0)) 0
                                                                             -/
    Tendsto (fun n ↦ ∫ x, ‖SimpleFunc.approxOn f fmeas (range f ∪ {0}) 0 (by simp) n x - f x‖ ∂μ)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
      atTop (𝓝 0) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_7
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    f : α → E
    fmeas : Measurable f
    hf : MeasureTheory.Integrable f μ
    inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => Norm.norm (HSub.h …
  -/
  convert (tendsto_toReal zero_ne_top).comp (tendsto_approxOn_range_L1_nnnorm fmeas hf) with n
  /-
    case h.e'_3.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_7
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    f : α → E
    fmeas : Measurable f
    hf : MeasureTheory.Integrable f μ
    inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
    n : Nat
    ⊢ Eq (MeasureTheory.integral μ fun x => Norm.norm (HSub.hSub ((MeasureTheory.S …
  -/
  rw [integral_norm_eq_lintegral_nnnorm]
    /-
      case h.e'_3.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      f : α → E
      fmeas : Measurable f
      hf : MeasureTheory.Integrable f μ
      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
      n : Nat
      ⊢ Eq (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm (HSub.hSub ((MeasureT …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      f : α → E
      fmeas : Measurable f
      hf : MeasureTheory.Integrable f μ
      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
      n : Nat
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HSub.hSub ((MeasureTheory.Simpl …
    -/
  · apply (SimpleFunc.aestronglyMeasurable _).sub
    /-
      case h.e'_3.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      f : α → E
      fmeas : Measurable f
      hf : MeasureTheory.Integrable f μ
      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
      n : Nat
      ⊢ MeasureTheory.AEStronglyMeasurable f μ
    -/
    apply (stronglyMeasurable_iff_measurable_separable.2 ⟨fmeas, ?_⟩ ).aestronglyMeasurable
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝³ : NormedAddCommGroup E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      f : α → E
      fmeas : Measurable f
      hf : MeasureTheory.Integrable f μ
      inst✝ : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton …
      n : Nat
      ⊢ TopologicalSpace.IsSeparable (Set.range f)
    -/
    exact .mono (.of_subtype (range f ∪ {0})) subset_union_left
    /-
      🎉 no goals
    -/


theorem integral_add_measure {f : α → G} (hμ : Integrable f μ) (hν : Integrable f ν) :
    ∫ x, f x ∂(μ + ν) = ∫ x, f x ∂μ + ∫ x, f x ∂ν := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → G
    hμ : MeasureTheory.Integrable f μ
    hν : MeasureTheory.Integrable f ν
    ⊢ Eq (MeasureTheory.integral (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (Measure …
  -/
  by_cases hG : CompleteSpace G; swap
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      f : α → G
      hμ : MeasureTheory.Integrable f μ
      hν : MeasureTheory.Integrable f ν
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (Measure …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → G
    hμ : MeasureTheory.Integrable f μ
    hν : MeasureTheory.Integrable f ν
    hG : CompleteSpace G
    ⊢ Eq (MeasureTheory.integral (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (Measure …
  -/
  have hfi := hμ.add_measure hν
  /-
    case pos
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → G
    hμ : MeasureTheory.Integrable f μ
    hν : MeasureTheory.Integrable f ν
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (HAdd.hAdd μ ν)
    ⊢ Eq (MeasureTheory.integral (HAdd.hAdd μ ν) fun x => f x) (HAdd.hAdd (Measure …
  -/
  simp_rw [integral_eq_setToFun]
  have hμ_dfma : DominatedFinMeasAdditive (μ + ν) (weightedSMul μ : Set α → G →L[ℝ] G) 1 :=
    DominatedFinMeasAdditive.add_measure_right μ ν (dominatedFinMeasAdditive_weightedSMul μ)
      zero_le_one
  have hν_dfma : DominatedFinMeasAdditive (μ + ν) (weightedSMul ν : Set α → G →L[ℝ] G) 1 :=
    DominatedFinMeasAdditive.add_measure_left μ ν (dominatedFinMeasAdditive_weightedSMul ν)
      zero_le_one
  rw [← setToFun_congr_measure_of_add_right hμ_dfma
        (dominatedFinMeasAdditive_weightedSMul μ) f hfi,
    ← setToFun_congr_measure_of_add_left hν_dfma (dominatedFinMeasAdditive_weightedSMul ν) f hfi]
  /-
    case pos
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → G
    hμ : MeasureTheory.Integrable f μ
    hν : MeasureTheory.Integrable f ν
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (HAdd.hAdd μ ν)
    hμ_dfma : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ ν) (MeasureTheor …
    hν_dfma : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ ν) (MeasureTheor …
    ⊢ Eq (MeasureTheory.setToFun (HAdd.hAdd μ ν) (MeasureTheory.weightedSMul (HAdd …
  -/
  refine setToFun_add_left' _ _ _ (fun s _ hμνs => ?_) f
  /-
    case pos
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → G
    hμ : MeasureTheory.Integrable f μ
    hν : MeasureTheory.Integrable f ν
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (HAdd.hAdd μ ν)
    hμ_dfma : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ ν) (MeasureTheor …
    hν_dfma : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ ν) (MeasureTheor …
    s : Set α
    x✝ : MeasurableSet s
    hμνs : LT.lt ((HAdd.hAdd μ ν) s) Top.top
    ⊢ Eq (MeasureTheory.weightedSMul (HAdd.hAdd μ ν) s) (HAdd.hAdd (MeasureTheory. …
  -/
  rw [Measure.coe_add, Pi.add_apply, add_lt_top] at hμνs
  rw [weightedSMul, weightedSMul, weightedSMul, ← add_smul, Measure.coe_add, Pi.add_apply,
  toReal_add hμνs.1.ne hμνs.2.ne]


@[simp]
theorem integral_zero_measure {m : MeasurableSpace α} (f : α → G) :
    (∫ x, f x ∂(0 : Measure α)) = 0 := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    f : α → G
    ⊢ Eq (MeasureTheory.integral 0 fun x => f x) 0
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      f : α → G
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral 0 fun x => f x) 0
    -/
  · simp only [integral, hG, L1.integral]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      f : α → G
      hG : CompleteSpace G
      ⊢ Eq (dite True (fun h => dite (MeasureTheory.Integrable (fun x => f x) 0) (fu …
    -/
    exact setToFun_measure_zero (dominatedFinMeasAdditive_weightedSMul _) rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      f : α → G
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral 0 fun x => f x) 0
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


@[simp]
theorem setIntegral_zero_measure (f : α → G) {μ : Measure α} {s : Set α} (hs : μ s = 0) :
    ∫ x in s, f x ∂μ = 0 := Measure.restrict_eq_zero.mpr hs ▸ integral_zero_measure f


lemma integral_of_isEmpty [IsEmpty α] {f : α → G} : ∫ x, f x ∂μ = 0 :=
    μ.eq_zero_of_isEmpty ▸ integral_zero_measure _


theorem integral_finset_sum_measure {ι} {m : MeasurableSpace α} {f : α → G} {μ : ι → Measure α}
    {s : Finset ι} (hf : ∀ i ∈ s, Integrable f (μ i)) :
    ∫ a, f a ∂(∑ i ∈ s, μ i) = ∑ i ∈ s, ∫ a, f a ∂μ i := by
  induction s using Finset.cons_induction_on with
  | h₁ => simp
  | h₂ h ih =>
    rw [Finset.forall_mem_cons] at hf
    rw [Finset.sum_cons, Finset.sum_cons, ← ih hf.2]
    exact integral_add_measure hf.1 (integrable_finset_sum_measure.2 hf.2)


theorem nndist_integral_add_measure_le_lintegral
    {f : α → G} (h₁ : Integrable f μ) (h₂ : Integrable f ν) :
    (nndist (∫ x, f x ∂μ) (∫ x, f x ∂(μ + ν)) : ℝ≥0∞) ≤ ∫⁻ x, ‖f x‖₊ ∂ν := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → G
    h₁ : MeasureTheory.Integrable f μ
    h₂ : MeasureTheory.Integrable f ν
    ⊢ LE.le (↑(NNDist.nndist (MeasureTheory.integral μ fun x => f x) (MeasureTheor …
  -/
  rw [integral_add_measure h₁ h₂, nndist_comm, nndist_eq_nnnorm, add_sub_cancel_left]
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : α → G
    h₁ : MeasureTheory.Integrable f μ
    h₂ : MeasureTheory.Integrable f ν
    ⊢ LE.le (↑(NNNorm.nnnorm (MeasureTheory.integral ν fun x => f x))) (MeasureThe …
  -/
  exact ennnorm_integral_le_lintegral_ennnorm _
  /-
    🎉 no goals
  -/


theorem hasSum_integral_measure {ι} {m : MeasurableSpace α} {f : α → G} {μ : ι → Measure α}
    (hf : Integrable f (Measure.sum μ)) :
    HasSum (fun i => ∫ a, f a ∂μ i) (∫ a, f a ∂Measure.sum μ) := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum μ)
    ⊢ HasSum (fun i => MeasureTheory.integral (μ i) fun a => f a) (MeasureTheory.i …
  -/
  have hfi : ∀ i, Integrable f (μ i) := fun i => hf.mono_measure (Measure.le_sum _ _)
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum μ)
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ⊢ HasSum (fun i => MeasureTheory.integral (μ i) fun a => f a) (MeasureTheory.i …
  -/
  simp only [HasSum, ← integral_finset_sum_measure fun i _ => hfi i]
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum μ)
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ⊢ Filter.Tendsto (fun s => MeasureTheory.integral (s.sum fun i => μ i) fun a = …
  -/
  refine Metric.nhds_basis_ball.tendsto_right_iff.mpr fun ε ε0 => ?_
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum μ)
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => Membership.mem (Metric.ball (MeasureTheory.integ …
  -/
  lift ε to ℝ≥0 using ε0.le
  /-
    case intro
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum μ)
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    ⊢ Filter.Eventually (fun x => Membership.mem (Metric.ball (MeasureTheory.integ …
  -/
  have hf_lt : (∫⁻ x, ‖f x‖₊ ∂Measure.sum μ) < ∞ := hf.2
  have hmem : ∀ᶠ y in 𝓝 (∫⁻ x, ‖f x‖₊ ∂Measure.sum μ), (∫⁻ x, ‖f x‖₊ ∂Measure.sum μ) < y + ε := by
    refine tendsto_id.add tendsto_const_nhds (lt_mem_nhds (α := ℝ≥0∞) <| ENNReal.lt_add_right ?_ ?_)
    exacts [hf_lt.ne, ENNReal.coe_ne_zero.2 (NNReal.coe_ne_zero.1 ε0.ne')]
  /-
    case intro
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum μ)
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hf_lt : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x =>  …
    hmem : Filter.Eventually (fun y => LT.lt (MeasureTheory.lintegral (MeasureTheo …
    ⊢ Filter.Eventually (fun x => Membership.mem (Metric.ball (MeasureTheory.integ …
  -/
  refine ((hasSum_lintegral_measure (fun x => ‖f x‖₊) μ).eventually hmem).mono fun s hs => ?_
  obtain ⟨ν, hν⟩ : ∃ ν, (∑ i ∈ s, μ i) + ν = Measure.sum μ := by
    refine ⟨Measure.sum fun i : ↥(sᶜ : Set ι) => μ i, ?_⟩
    simpa only [← Measure.sum_coe_finset] using Measure.sum_add_sum_compl (s : Set ι) μ
  /-
    case intro.intro
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum μ)
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hf_lt : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x =>  …
    hmem : Filter.Eventually (fun y => LT.lt (MeasureTheory.lintegral (MeasureTheo …
    s : Finset ι
    hs : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x => ↑(N …
    ν : MeasureTheory.Measure α
    hν : Eq (HAdd.hAdd (s.sum fun i => μ i) ν) (MeasureTheory.Measure.sum μ)
    ⊢ Membership.mem (Metric.ball (MeasureTheory.integral (MeasureTheory.Measure.s …
  -/
  rw [Metric.mem_ball, ← coe_nndist, NNReal.coe_lt_coe, ← ENNReal.coe_lt_coe, ← hν]
  /-
    case intro.intro
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum μ)
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hf_lt : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x =>  …
    hmem : Filter.Eventually (fun y => LT.lt (MeasureTheory.lintegral (MeasureTheo …
    s : Finset ι
    hs : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x => ↑(N …
    ν : MeasureTheory.Measure α
    hν : Eq (HAdd.hAdd (s.sum fun i => μ i) ν) (MeasureTheory.Measure.sum μ)
    ⊢ LT.lt ↑(NNDist.nndist (MeasureTheory.integral (s.sum fun i => μ i) fun a =>  …
  -/
  rw [← hν, integrable_add_measure] at hf
  /-
    case intro.intro
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hf_lt : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x =>  …
    hmem : Filter.Eventually (fun y => LT.lt (MeasureTheory.lintegral (MeasureTheo …
    s : Finset ι
    hs : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x => ↑(N …
    ν : MeasureTheory.Measure α
    hf : And (MeasureTheory.Integrable f (s.sum fun i => μ i)) (MeasureTheory.Inte …
    hν : Eq (HAdd.hAdd (s.sum fun i => μ i) ν) (MeasureTheory.Measure.sum μ)
    ⊢ LT.lt ↑(NNDist.nndist (MeasureTheory.integral (s.sum fun i => μ i) fun a =>  …
  -/
  refine (nndist_integral_add_measure_le_lintegral hf.1 hf.2).trans_lt ?_
  /-
    case intro.intro
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hf_lt : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x =>  …
    hmem : Filter.Eventually (fun y => LT.lt (MeasureTheory.lintegral (MeasureTheo …
    s : Finset ι
    hs : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x => ↑(N …
    ν : MeasureTheory.Measure α
    hf : And (MeasureTheory.Integrable f (s.sum fun i => μ i)) (MeasureTheory.Inte …
    hν : Eq (HAdd.hAdd (s.sum fun i => μ i) ν) (MeasureTheory.Measure.sum μ)
    ⊢ LT.lt (MeasureTheory.lintegral ν fun x => ↑(NNNorm.nnnorm (f x))) ↑ε
  -/
  rw [← hν, lintegral_add_measure, lintegral_finset_sum_measure] at hs
  /-
    case intro.intro
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    ι : Type u_7
    m : MeasurableSpace α
    f : α → G
    μ : ι → MeasureTheory.Measure α
    hfi : ∀ (i : ι), MeasureTheory.Integrable f (μ i)
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    hf_lt : LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.sum μ) fun x =>  …
    hmem : Filter.Eventually (fun y => LT.lt (MeasureTheory.lintegral (MeasureTheo …
    s : Finset ι
    ν : MeasureTheory.Measure α
    hs : LT.lt (HAdd.hAdd (s.sum fun i => MeasureTheory.lintegral (μ i) fun a => ↑ …
    hf : And (MeasureTheory.Integrable f (s.sum fun i => μ i)) (MeasureTheory.Inte …
    hν : Eq (HAdd.hAdd (s.sum fun i => μ i) ν) (MeasureTheory.Measure.sum μ)
    ⊢ LT.lt (MeasureTheory.lintegral ν fun x => ↑(NNNorm.nnnorm (f x))) ↑ε
  -/
  exact lt_of_add_lt_add_left hs
  /-
    🎉 no goals
  -/


theorem integral_sum_measure {ι} {_ : MeasurableSpace α} {f : α → G} {μ : ι → Measure α}
    (hf : Integrable f (Measure.sum μ)) : ∫ a, f a ∂Measure.sum μ = ∑' i, ∫ a, f a ∂μ i :=
  (hasSum_integral_measure hf).tsum_eq.symm


@[simp]
theorem integral_smul_measure (f : α → G) (c : ℝ≥0∞) :
    ∫ x, f x ∂c • μ = c.toReal • ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    c : ENNReal
    ⊢ Eq (MeasureTheory.integral (HSMul.hSMul c μ) fun x => f x) (HSMul.hSMul c.to …
  -/
  by_cases hG : CompleteSpace G; swap
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      c : ENNReal
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral (HSMul.hSMul c μ) fun x => f x) (HSMul.hSMul c.to …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/
  -- First we consider the “degenerate” case `c = ∞`
  /-
    case pos
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    c : ENNReal
    hG : CompleteSpace G
    ⊢ Eq (MeasureTheory.integral (HSMul.hSMul c μ) fun x => f x) (HSMul.hSMul c.to …
  -/
  rcases eq_or_ne c ∞ with (rfl | hc)
    /-
      case pos.inl
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → G
      hG : CompleteSpace G
      ⊢ Eq (MeasureTheory.integral (HSMul.hSMul Top.top μ) fun x => f x) (HSMul.hSMu …
    -/
  · rw [ENNReal.top_toReal, zero_smul, integral_eq_setToFun, setToFun_top_smul_measure]
    /-
      🎉 no goals
    -/
  -- Main case: `c ≠ ∞`
  /-
    case pos.inr
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    c : ENNReal
    hG : CompleteSpace G
    hc : Ne c Top.top
    ⊢ Eq (MeasureTheory.integral (HSMul.hSMul c μ) fun x => f x) (HSMul.hSMul c.to …
  -/
  simp_rw [integral_eq_setToFun, ← setToFun_smul_left]
  have hdfma : DominatedFinMeasAdditive μ (weightedSMul (c • μ) : Set α → G →L[ℝ] G) c.toReal :=
    mul_one c.toReal ▸ (dominatedFinMeasAdditive_weightedSMul (c • μ)).of_smul_measure c hc
  /-
    case pos.inr
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    c : ENNReal
    hG : CompleteSpace G
    hc : Ne c Top.top
    hdfma : MeasureTheory.DominatedFinMeasAdditive μ (MeasureTheory.weightedSMul ( …
    ⊢ Eq (MeasureTheory.setToFun (HSMul.hSMul c μ) (MeasureTheory.weightedSMul (HS …
  -/
  have hdfma_smul := dominatedFinMeasAdditive_weightedSMul (F := G) (c • μ)
  /-
    case pos.inr
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    c : ENNReal
    hG : CompleteSpace G
    hc : Ne c Top.top
    hdfma : MeasureTheory.DominatedFinMeasAdditive μ (MeasureTheory.weightedSMul ( …
    hdfma_smul : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) (Measure …
    ⊢ Eq (MeasureTheory.setToFun (HSMul.hSMul c μ) (MeasureTheory.weightedSMul (HS …
  -/
  rw [← setToFun_congr_smul_measure c hc hdfma hdfma_smul f]
  /-
    case pos.inr
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → G
    c : ENNReal
    hG : CompleteSpace G
    hc : Ne c Top.top
    hdfma : MeasureTheory.DominatedFinMeasAdditive μ (MeasureTheory.weightedSMul ( …
    hdfma_smul : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) (Measure …
    ⊢ Eq (MeasureTheory.setToFun μ (MeasureTheory.weightedSMul (HSMul.hSMul c μ))  …
  -/
  exact setToFun_congr_left' _ _ (fun s _ _ => weightedSMul_smul_measure μ c) f
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_smul_nnreal_measure (f : α → G) (c : ℝ≥0) :
    ∫ x, f x ∂(c • μ) = c • ∫ x, f x ∂μ :=
  integral_smul_measure f (c : ℝ≥0∞)


theorem integral_map_of_stronglyMeasurable {β} [MeasurableSpace β] {φ : α → β} (hφ : Measurable φ)
    {f : β → G} (hfm : StronglyMeasurable f) : ∫ y, f y ∂Measure.map φ μ = ∫ x, f (φ x) ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map φ μ) fun y => f y) (Me …
  -/
  by_cases hG : CompleteSpace G; swap
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : MeasurableSpace β
      φ : α → β
      hφ : Measurable φ
      f : β → G
      hfm : MeasureTheory.StronglyMeasurable f
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map φ μ) fun y => f y) (Me …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map φ μ) fun y => f y) (Me …
  -/
  by_cases hfi : Integrable f (Measure.map φ μ); swap
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : MeasurableSpace β
      φ : α → β
      hφ : Measurable φ
      f : β → G
      hfm : MeasureTheory.StronglyMeasurable f
      hG : CompleteSpace G
      hfi : Not (MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ))
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map φ μ) fun y => f y) (Me …
    -/
  · rw [integral_undef hfi, integral_undef]
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      inst✝ : MeasurableSpace β
      φ : α → β
      hφ : Measurable φ
      f : β → G
      hfm : MeasureTheory.StronglyMeasurable f
      hG : CompleteSpace G
      hfi : Not (MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ))
      ⊢ Not (MeasureTheory.Integrable (fun x => f (φ x)) μ)
    -/
    exact fun hfφ => hfi ((integrable_map_measure hfm.aestronglyMeasurable hφ.aemeasurable).2 hfφ)
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ)
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map φ μ) fun y => f y) (Me …
  -/
  borelize G
  /-
    case pos
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ)
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map φ μ) fun y => f y) (Me …
  -/
  have : SeparableSpace (range f ∪ {0} : Set G) := hfm.separableSpace_range_union_singleton
  refine tendsto_nhds_unique
    (tendsto_integral_approxOn_of_measurable_of_range_subset hfm.measurable hfi _ Subset.rfl) ?_
  convert tendsto_integral_approxOn_of_measurable_of_range_subset (hfm.measurable.comp hφ)
    ((integrable_map_measure hfm.aestronglyMeasurable hφ.aemeasurable).1 hfi) (range f ∪ {0})
    (by simp [insert_subset_insert, Set.range_comp_subset_range]) using 1
  /-
    case h.e'_3
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ)
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    ⊢ Eq (fun n => MeasureTheory.SimpleFunc.integral (MeasureTheory.Measure.map φ  …
  -/
  ext1 i
  simp only [SimpleFunc.approxOn_comp, SimpleFunc.integral_eq, Measure.map_apply, hφ,
    SimpleFunc.measurableSet_preimage, ← preimage_comp, SimpleFunc.coe_comp]
  /-
    case h.e'_3.h
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ)
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    i : Nat
    ⊢ Eq ((MeasureTheory.SimpleFunc.approxOn f ⋯ (Union.union (Set.range f) (Singl …
  -/
  refine (Finset.sum_subset (SimpleFunc.range_comp_subset_range _ hφ) fun y _ hy => ?_).symm
  /-
    case h.e'_3.h
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ)
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    i : Nat
    y : G
    x✝ : Membership.mem (MeasureTheory.SimpleFunc.approxOn f ⋯ (Union.union (Set.r …
    hy : Not (Membership.mem ((MeasureTheory.SimpleFunc.approxOn f ⋯ (Union.union  …
    ⊢ Eq (HSMul.hSMul (μ (Set.preimage (Function.comp (⇑(MeasureTheory.SimpleFunc. …
  -/
  rw [SimpleFunc.mem_range, ← Set.preimage_singleton_eq_empty, SimpleFunc.coe_comp] at hy
  /-
    case h.e'_3.h
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ)
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    i : Nat
    y : G
    x✝ : Membership.mem (MeasureTheory.SimpleFunc.approxOn f ⋯ (Union.union (Set.r …
    hy : Eq (Set.preimage (Function.comp (⇑(MeasureTheory.SimpleFunc.approxOn f ⋯  …
    ⊢ Eq (HSMul.hSMul (μ (Set.preimage (Function.comp (⇑(MeasureTheory.SimpleFunc. …
  -/
  rw [hy]
  /-
    case h.e'_3.h
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : MeasurableSpace β
    φ : α → β
    hφ : Measurable φ
    f : β → G
    hfm : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    hfi : MeasureTheory.Integrable f (MeasureTheory.Measure.map φ μ)
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    i : Nat
    y : G
    x✝ : Membership.mem (MeasureTheory.SimpleFunc.approxOn f ⋯ (Union.union (Set.r …
    hy : Eq (Set.preimage (Function.comp (⇑(MeasureTheory.SimpleFunc.approxOn f ⋯  …
    ⊢ Eq (HSMul.hSMul (μ EmptyCollection.emptyCollection).toReal y) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem integral_map {β} [MeasurableSpace β] {φ : α → β} (hφ : AEMeasurable φ μ) {f : β → G}
    (hfm : AEStronglyMeasurable f (Measure.map φ μ)) :
    ∫ y, f y ∂Measure.map φ μ = ∫ x, f (φ x) ∂μ :=
  let g := hfm.mk f
  calc
    ∫ y, f y ∂Measure.map φ μ = ∫ y, g y ∂Measure.map φ μ := integral_congr_ae hfm.ae_eq_mk
                                                /-
                                                  α : Type u_1
                                                  G : Type u_5
                                                  inst✝² : NormedAddCommGroup G
                                                  inst✝¹ : NormedSpace Real G
                                                  m : MeasurableSpace α
                                                  μ : MeasureTheory.Measure α
                                                  β : Type u_7
                                                  inst✝ : MeasurableSpace β
                                                  φ : α → β
                                                  hφ : AEMeasurable φ μ
                                                  f : β → G
                                                  hfm : MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.map φ μ)
                                                  g : β → G := MeasureTheory.AEStronglyMeasurable.mk f hfm
                                                  ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map φ μ) fun y => g y) (Me …
                                                -/
    _ = ∫ y, g y ∂Measure.map (hφ.mk φ) μ := by congr 1; exact Measure.map_congr hφ.ae_eq_mk
                                                         /-
                                                           🎉 no goals
                                                         -/
    _ = ∫ x, g (hφ.mk φ x) ∂μ :=
      (integral_map_of_stronglyMeasurable hφ.measurable_mk hfm.stronglyMeasurable_mk)
    _ = ∫ x, g (φ x) ∂μ := integral_congr_ae (hφ.ae_eq_mk.symm.fun_comp _)
    _ = ∫ x, f (φ x) ∂μ := integral_congr_ae <| ae_eq_comp hφ hfm.ae_eq_mk.symm


theorem _root_.MeasurableEmbedding.integral_map {β} {_ : MeasurableSpace β} {f : α → β}
    (hf : MeasurableEmbedding f) (g : β → G) : ∫ y, g y ∂Measure.map f μ = ∫ x, g (f x) ∂μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    x✝ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    g : β → G
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map f μ) fun y => g y) (Me …
  -/
  by_cases hgm : AEStronglyMeasurable g (Measure.map f μ)
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      x✝ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      g : β → G
      hgm : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map f μ) fun y => g y) (Me …
    -/
  · exact MeasureTheory.integral_map hf.measurable.aemeasurable hgm
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      x✝ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      g : β → G
      hgm : Not (MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ))
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map f μ) fun y => g y) (Me …
    -/
  · rw [integral_non_aestronglyMeasurable hgm, integral_non_aestronglyMeasurable]
    /-
      case neg
      α : Type u_1
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_7
      x✝ : MeasurableSpace β
      f : α → β
      hf : MeasurableEmbedding f
      g : β → G
      hgm : Not (MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ))
      ⊢ Not (MeasureTheory.AEStronglyMeasurable (fun x => g (f x)) μ)
    -/
    exact fun hgf => hgm (hf.aestronglyMeasurable_map_iff.2 hgf)
    /-
      🎉 no goals
    -/


theorem _root_.Topology.IsClosedEmbedding.integral_map {β} [TopologicalSpace α] [BorelSpace α]
    [TopologicalSpace β] [MeasurableSpace β] [BorelSpace β] {φ : α → β} (hφ : IsClosedEmbedding φ)
    (f : β → G) : ∫ y, f y ∂Measure.map φ μ = ∫ x, f (φ x) ∂μ :=
  hφ.measurableEmbedding.integral_map _


@[deprecated (since := "2024-10-20")]
alias _root_.ClosedEmbedding.integral_map := IsClosedEmbedding.integral_map


theorem integral_map_equiv {β} [MeasurableSpace β] (e : α ≃ᵐ β) (f : β → G) :
    ∫ y, f y ∂Measure.map e μ = ∫ x, f (e x) ∂μ :=
  e.measurableEmbedding.integral_map f


theorem MeasurePreserving.integral_comp {β} {_ : MeasurableSpace β} {f : α → β} {ν}
    (h₁ : MeasurePreserving f μ ν) (h₂ : MeasurableEmbedding f) (g : β → G) :
    ∫ x, g (f x) ∂μ = ∫ y, g y ∂ν :=
  h₁.map_eq ▸ (h₂.integral_map g).symm


theorem MeasurePreserving.integral_comp' {β} [MeasurableSpace β] {ν} {f : α ≃ᵐ β}
    (h : MeasurePreserving f μ ν) (g : β → G) :
    ∫ x, g (f x) ∂μ = ∫ y, g y ∂ν := MeasurePreserving.integral_comp h f.measurableEmbedding _


theorem integral_subtype_comap {α} [MeasurableSpace α] {μ : Measure α} {s : Set α}
    (hs : MeasurableSet s) (f : α → G) :
    ∫ x : s, f (x : α) ∂(Measure.comap Subtype.val μ) = ∫ x in s, f x ∂μ := by
  /-
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    α : Type u_7
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → G
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.comap Subtype.val μ) fun x …
  -/
  rw [← map_comap_subtype_coe hs]
  /-
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    α : Type u_7
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → G
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.comap Subtype.val μ) fun x …
  -/
  exact ((MeasurableEmbedding.subtype_coe hs).integral_map _).symm
  /-
    🎉 no goals
  -/


attribute [local instance] Measure.Subtype.measureSpace in
theorem integral_subtype {α} [MeasureSpace α] {s : Set α} (hs : MeasurableSet s) (f : α → G) :
    ∫ x : s, f x = ∫ x in s, f x := integral_subtype_comap hs f


@[simp]
theorem integral_dirac' [MeasurableSpace α] (f : α → E) (a : α) (hfm : StronglyMeasurable f) :
    ∫ x, f x ∂Measure.dirac a = f a := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasurableSpace α
    f : α → E
    a : α
    hfm : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.dirac a) fun x => f x) (f a)
  -/
  borelize E
  calc
    ∫ x, f x ∂Measure.dirac a = ∫ _, f a ∂Measure.dirac a :=
      integral_congr_ae <| ae_eq_dirac' hfm.measurable
    _ = f a := by simp [Measure.dirac_apply_of_mem]


@[simp]
theorem integral_dirac [MeasurableSpace α] [MeasurableSingletonClass α] (f : α → E) (a : α) :
    ∫ x, f x ∂Measure.dirac a = f a :=
  calc
    ∫ x, f x ∂Measure.dirac a = ∫ _, f a ∂Measure.dirac a := integral_congr_ae <| ae_eq_dirac f
                  /-
                    α : Type u_1
                    E : Type u_2
                    inst✝³ : NormedAddCommGroup E
                    hE : CompleteSpace E
                    inst✝² : NormedSpace Real E
                    inst✝¹ : MeasurableSpace α
                    inst✝ : MeasurableSingletonClass α
                    f : α → E
                    a : α
                    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.dirac a) fun x => f a) (f a)
                  -/
    _ = f a := by simp [Measure.dirac_apply_of_mem]
                  /-
                    🎉 no goals
                  -/


theorem setIntegral_dirac' {mα : MeasurableSpace α} {f : α → E} (hf : StronglyMeasurable f) (a : α)
    {s : Set α} (hs : MeasurableSet s) [Decidable (a ∈ s)] :
    ∫ x in s, f x ∂Measure.dirac a = if a ∈ s then f a else 0 := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    mα : MeasurableSpace α
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    a : α
    s : Set α
    hs : MeasurableSet s
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (MeasureTheory.integral ((MeasureTheory.Measure.dirac a).restrict s) fun  …
  -/
  rw [restrict_dirac' hs]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    mα : MeasurableSpace α
    f : α → E
    hf : MeasureTheory.StronglyMeasurable f
    a : α
    s : Set α
    hs : MeasurableSet s
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (MeasureTheory.integral (ite (Membership.mem s a) (MeasureTheory.Measure. …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      hE : CompleteSpace E
      inst✝¹ : NormedSpace Real E
      mα : MeasurableSpace α
      f : α → E
      hf : MeasureTheory.StronglyMeasurable f
      a : α
      s : Set α
      hs : MeasurableSet s
      inst✝ : Decidable (Membership.mem s a)
      h✝ : Membership.mem s a
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.dirac a) fun x => f x) (f a)
    -/
  · exact integral_dirac' _ _ hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      hE : CompleteSpace E
      inst✝¹ : NormedSpace Real E
      mα : MeasurableSpace α
      f : α → E
      hf : MeasureTheory.StronglyMeasurable f
      a : α
      s : Set α
      hs : MeasurableSet s
      inst✝ : Decidable (Membership.mem s a)
      h✝ : Not (Membership.mem s a)
      ⊢ Eq (MeasureTheory.integral 0 fun x => f x) 0
    -/
  · exact integral_zero_measure _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_dirac' := setIntegral_dirac'


theorem setIntegral_dirac [MeasurableSpace α] [MeasurableSingletonClass α] (f : α → E) (a : α)
    (s : Set α) [Decidable (a ∈ s)] :
    ∫ x in s, f x ∂Measure.dirac a = if a ∈ s then f a else 0 := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝³ : NormedSpace Real E
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSingletonClass α
    f : α → E
    a : α
    s : Set α
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (MeasureTheory.integral ((MeasureTheory.Measure.dirac a).restrict s) fun  …
  -/
  rw [restrict_dirac]
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝³ : NormedSpace Real E
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSingletonClass α
    f : α → E
    a : α
    s : Set α
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (MeasureTheory.integral (ite (Membership.mem s a) (MeasureTheory.Measure. …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      hE : CompleteSpace E
      inst✝³ : NormedSpace Real E
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSingletonClass α
      f : α → E
      a : α
      s : Set α
      inst✝ : Decidable (Membership.mem s a)
      h✝ : Membership.mem s a
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.dirac a) fun x => f x) (f a)
    -/
  · exact integral_dirac _ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      hE : CompleteSpace E
      inst✝³ : NormedSpace Real E
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSingletonClass α
      f : α → E
      a : α
      s : Set α
      inst✝ : Decidable (Membership.mem s a)
      h✝ : Not (Membership.mem s a)
      ⊢ Eq (MeasureTheory.integral 0 fun x => f x) 0
    -/
  · exact integral_zero_measure _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_dirac := setIntegral_dirac


/-- **Markov's inequality** also known as **Chebyshev's first inequality**. -/
theorem mul_meas_ge_le_integral_of_nonneg {f : α → ℝ} (hf_nonneg : 0 ≤ᵐ[μ] f)
    (hf_int : Integrable f μ) (ε : ℝ) : ε * (μ { x | ε ≤ f x }).toReal ≤ ∫ x, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
    hf_int : MeasureTheory.Integrable f μ
    ε : Real
    ⊢ LE.le (HMul.hMul ε (μ (setOf fun x => LE.le ε (f x))).toReal) (MeasureTheory …
  -/
  cases' eq_top_or_lt_top (μ {x | ε ≤ f x}) with hμ hμ
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
      hf_int : MeasureTheory.Integrable f μ
      ε : Real
      hμ : Eq (μ (setOf fun x => LE.le ε (f x))) Top.top
      ⊢ LE.le (HMul.hMul ε (μ (setOf fun x => LE.le ε (f x))).toReal) (MeasureTheory …
    -/
  · simpa [hμ] using integral_nonneg_of_ae hf_nonneg
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
      hf_int : MeasureTheory.Integrable f μ
      ε : Real
      hμ : LT.lt (μ (setOf fun x => LE.le ε (f x))) Top.top
      ⊢ LE.le (HMul.hMul ε (μ (setOf fun x => LE.le ε (f x))).toReal) (MeasureTheory …
    -/
  · have := Fact.mk hμ
    calc
      ε * (μ { x | ε ≤ f x }).toReal = ∫ _ in {x | ε ≤ f x}, ε ∂μ := by simp [mul_comm]
      _ ≤ ∫ x in {x | ε ≤ f x}, f x ∂μ :=
        integral_mono_ae (integrable_const _) (hf_int.mono_measure μ.restrict_le_self) <|
          ae_restrict_mem₀ <| hf_int.aemeasurable.nullMeasurable measurableSet_Ici
      _ ≤ _ := integral_mono_measure μ.restrict_le_self hf_nonneg hf_int


/-- Hölder's inequality for the integral of a product of norms. The integral of the product of two
norms of functions is bounded by the product of their `ℒp` and `ℒq` seminorms when `p` and `q` are
conjugate exponents. -/
theorem integral_mul_norm_le_Lp_mul_Lq {E} [NormedAddCommGroup E] {f g : α → E} {p q : ℝ}
    (hpq : p.IsConjExponent q) (hf : Memℒp f (ENNReal.ofReal p) μ)
    (hg : Memℒp g (ENNReal.ofReal q) μ) :
    ∫ a, ‖f a‖ * ‖g a‖ ∂μ ≤ (∫ a, ‖f a‖ ^ p ∂μ) ^ (1 / p) * (∫ a, ‖g a‖ ^ q ∂μ) ^ (1 / q) := by
  -- translate the Bochner integrals into Lebesgue integrals.
  rw [integral_eq_lintegral_of_nonneg_ae, integral_eq_lintegral_of_nonneg_ae,
    integral_eq_lintegral_of_nonneg_ae]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f g : α → E
    p q : Real
    hpq : p.IsConjExponent q
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Norm.no …
  -/
  rotate_left
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f g : α → E
      p q : Real
      hpq : p.IsConjExponent q
      hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
      hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun a => HPow.hPow (Norm.norm (g a)) q
    -/
  · exact Eventually.of_forall fun x => Real.rpow_nonneg (norm_nonneg _) _
    /-
      🎉 no goals
    -/
    /-
      case hfm
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f g : α → E
      p q : Real
      hpq : p.IsConjExponent q
      hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
      hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HPow.hPow (Norm.norm (g a)) q) μ
    -/
  · exact (hg.1.norm.aemeasurable.pow aemeasurable_const).aestronglyMeasurable
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f g : α → E
      p q : Real
      hpq : p.IsConjExponent q
      hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
      hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun a => HPow.hPow (Norm.norm (f a)) p
    -/
  · exact Eventually.of_forall fun x => Real.rpow_nonneg (norm_nonneg _) _
    /-
      🎉 no goals
    -/
    /-
      case hfm
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f g : α → E
      p q : Real
      hpq : p.IsConjExponent q
      hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
      hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HPow.hPow (Norm.norm (f a)) p) μ
    -/
  · exact (hf.1.norm.aemeasurable.pow aemeasurable_const).aestronglyMeasurable
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f g : α → E
      p q : Real
      hpq : p.IsConjExponent q
      hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
      hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun a => HMul.hMul (Norm.norm (f a)) (No …
    -/
  · exact Eventually.of_forall fun x => mul_nonneg (norm_nonneg _) (norm_nonneg _)
    /-
      🎉 no goals
    -/
    /-
      case hfm
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f g : α → E
      p q : Real
      hpq : p.IsConjExponent q
      hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
      hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HMul.hMul (Norm.norm (f a)) (No …
    -/
  · exact hf.1.norm.mul hg.1.norm
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f g : α → E
    p q : Real
    hpq : p.IsConjExponent q
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Norm.no …
  -/
  rw [ENNReal.toReal_rpow, ENNReal.toReal_rpow, ← ENNReal.toReal_mul]
  -- replace norms by nnnorm
  have h_left : ∫⁻ a, ENNReal.ofReal (‖f a‖ * ‖g a‖) ∂μ =
      ∫⁻ a, ((fun x => (‖f x‖₊ : ℝ≥0∞)) * fun x => (‖g x‖₊ : ℝ≥0∞)) a ∂μ := by
    simp_rw [Pi.mul_apply, ← ofReal_norm_eq_coe_nnnorm, ENNReal.ofReal_mul (norm_nonneg _)]
  have h_right_f : ∫⁻ a, ENNReal.ofReal (‖f a‖ ^ p) ∂μ = ∫⁻ a, (‖f a‖₊ : ℝ≥0∞) ^ p ∂μ := by
    refine lintegral_congr fun x => ?_
    rw [← ofReal_norm_eq_coe_nnnorm, ENNReal.ofReal_rpow_of_nonneg (norm_nonneg _) hpq.nonneg]
  have h_right_g : ∫⁻ a, ENNReal.ofReal (‖g a‖ ^ q) ∂μ = ∫⁻ a, (‖g a‖₊ : ℝ≥0∞) ^ q ∂μ := by
    refine lintegral_congr fun x => ?_
    rw [← ofReal_norm_eq_coe_nnnorm, ENNReal.ofReal_rpow_of_nonneg (norm_nonneg _) hpq.symm.nonneg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f g : α → E
    p q : Real
    hpq : p.IsConjExponent q
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
    h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
    h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
    h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Norm.no …
  -/
  rw [h_left, h_right_f, h_right_g]
  -- we can now apply `ENNReal.lintegral_mul_le_Lp_mul_Lq` (up to the `toReal` application)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_7
    inst✝ : NormedAddCommGroup E
    f g : α → E
    p q : Real
    hpq : p.IsConjExponent q
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
    h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
    h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
    h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul (fun x => ↑(NNNorm.nnnor …
  -/
  refine ENNReal.toReal_mono ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_7
      inst✝ : NormedAddCommGroup E
      f g : α → E
      p q : Real
      hpq : p.IsConjExponent q
      hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
      hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
      h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
      h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
      h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
      ⊢ Ne (HMul.hMul (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NN …
    -/
  · refine ENNReal.mul_ne_top ?_ ?_
      /-
        case refine_1.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_7
        inst✝ : NormedAddCommGroup E
        f g : α → E
        p q : Real
        hpq : p.IsConjExponent q
        hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
        hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
        h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
        h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
        h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
        ⊢ Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm …
      -/
    · convert hf.eLpNorm_ne_top
      /-
        case h.e'_2
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_7
        inst✝ : NormedAddCommGroup E
        f g : α → E
        p q : Real
        hpq : p.IsConjExponent q
        hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
        hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
        h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
        h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
        h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
        ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm …
      -/
      rw [eLpNorm_eq_lintegral_rpow_nnnorm]
        /-
          case h.e'_2
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          E : Type u_7
          inst✝ : NormedAddCommGroup E
          f g : α → E
          p q : Real
          hpq : p.IsConjExponent q
          hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
          hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
          h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
          h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm …
        -/
      · rw [ENNReal.toReal_ofReal hpq.nonneg]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2.hp_ne_zero
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          E : Type u_7
          inst✝ : NormedAddCommGroup E
          f g : α → E
          p q : Real
          hpq : p.IsConjExponent q
          hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
          hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
          h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
          h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          ⊢ Ne (ENNReal.ofReal p) 0
        -/
      · rw [Ne, ENNReal.ofReal_eq_zero, not_le]
        /-
          case h.e'_2.hp_ne_zero
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          E : Type u_7
          inst✝ : NormedAddCommGroup E
          f g : α → E
          p q : Real
          hpq : p.IsConjExponent q
          hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
          hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
          h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
          h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          ⊢ LT.lt 0 p
        -/
        exact hpq.pos
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2.hp_ne_top
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          E : Type u_7
          inst✝ : NormedAddCommGroup E
          f g : α → E
          p q : Real
          hpq : p.IsConjExponent q
          hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
          hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
          h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
          h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          ⊢ Ne (ENNReal.ofReal p) Top.top
        -/
      · exact ENNReal.coe_ne_top
        /-
          🎉 no goals
        -/
      /-
        case refine_1.refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_7
        inst✝ : NormedAddCommGroup E
        f g : α → E
        p q : Real
        hpq : p.IsConjExponent q
        hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
        hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
        h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
        h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
        h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
        ⊢ Ne (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm …
      -/
    · convert hg.eLpNorm_ne_top
      /-
        case h.e'_2
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_7
        inst✝ : NormedAddCommGroup E
        f g : α → E
        p q : Real
        hpq : p.IsConjExponent q
        hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
        hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
        h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
        h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
        h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
        ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm …
      -/
      rw [eLpNorm_eq_lintegral_rpow_nnnorm]
        /-
          case h.e'_2
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          E : Type u_7
          inst✝ : NormedAddCommGroup E
          f g : α → E
          p q : Real
          hpq : p.IsConjExponent q
          hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
          hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
          h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
          h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm …
        -/
      · rw [ENNReal.toReal_ofReal hpq.symm.nonneg]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2.hp_ne_zero
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          E : Type u_7
          inst✝ : NormedAddCommGroup E
          f g : α → E
          p q : Real
          hpq : p.IsConjExponent q
          hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
          hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
          h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
          h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          ⊢ Ne (ENNReal.ofReal q) 0
        -/
      · rw [Ne, ENNReal.ofReal_eq_zero, not_le]
        /-
          case h.e'_2.hp_ne_zero
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          E : Type u_7
          inst✝ : NormedAddCommGroup E
          f g : α → E
          p q : Real
          hpq : p.IsConjExponent q
          hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
          hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
          h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
          h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          ⊢ LT.lt 0 q
        -/
        exact hpq.symm.pos
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2.hp_ne_top
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          E : Type u_7
          inst✝ : NormedAddCommGroup E
          f g : α → E
          p q : Real
          hpq : p.IsConjExponent q
          hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
          hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
          h_left : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HMul.hMul (Nor …
          h_right_f : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          h_right_g : Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow ( …
          ⊢ Ne (ENNReal.ofReal q) Top.top
        -/
      · exact ENNReal.coe_ne_top
        /-
          🎉 no goals
        -/
  · exact ENNReal.lintegral_mul_le_Lp_mul_Lq μ hpq hf.1.nnnorm.aemeasurable.coe_nnreal_ennreal
      hg.1.nnnorm.aemeasurable.coe_nnreal_ennreal


/-- Hölder's inequality for functions `α → ℝ`. The integral of the product of two nonnegative
functions is bounded by the product of their `ℒp` and `ℒq` seminorms when `p` and `q` are conjugate
exponents. -/
theorem integral_mul_le_Lp_mul_Lq_of_nonneg {p q : ℝ} (hpq : p.IsConjExponent q) {f g : α → ℝ}
    (hf_nonneg : 0 ≤ᵐ[μ] f) (hg_nonneg : 0 ≤ᵐ[μ] g) (hf : Memℒp f (ENNReal.ofReal p) μ)
    (hg : Memℒp g (ENNReal.ofReal q) μ) :
    ∫ a, f a * g a ∂μ ≤ (∫ a, f a ^ p ∂μ) ^ (1 / p) * (∫ a, g a ^ q ∂μ) ^ (1 / q) := by
  have h_left : ∫ a, f a * g a ∂μ = ∫ a, ‖f a‖ * ‖g a‖ ∂μ := by
    refine integral_congr_ae ?_
    filter_upwards [hf_nonneg, hg_nonneg] with x hxf hxg
    rw [Real.norm_of_nonneg hxf, Real.norm_of_nonneg hxg]
  have h_right_f : ∫ a, f a ^ p ∂μ = ∫ a, ‖f a‖ ^ p ∂μ := by
    refine integral_congr_ae ?_
    filter_upwards [hf_nonneg] with x hxf
    rw [Real.norm_of_nonneg hxf]
  have h_right_g : ∫ a, g a ^ q ∂μ = ∫ a, ‖g a‖ ^ q ∂μ := by
    refine integral_congr_ae ?_
    filter_upwards [hg_nonneg] with x hxg
    rw [Real.norm_of_nonneg hxg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → Real
    hf_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
    hg_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 g
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
    h_left : Eq (MeasureTheory.integral μ fun a => HMul.hMul (f a) (g a)) (Measure …
    h_right_f : Eq (MeasureTheory.integral μ fun a => HPow.hPow (f a) p) (MeasureT …
    h_right_g : Eq (MeasureTheory.integral μ fun a => HPow.hPow (g a) q) (MeasureT …
    ⊢ LE.le (MeasureTheory.integral μ fun a => HMul.hMul (f a) (g a)) (HMul.hMul ( …
  -/
  rw [h_left, h_right_f, h_right_g]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p q : Real
    hpq : p.IsConjExponent q
    f g : α → Real
    hf_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
    hg_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 g
    hf : MeasureTheory.Memℒp f (ENNReal.ofReal p) μ
    hg : MeasureTheory.Memℒp g (ENNReal.ofReal q) μ
    h_left : Eq (MeasureTheory.integral μ fun a => HMul.hMul (f a) (g a)) (Measure …
    h_right_f : Eq (MeasureTheory.integral μ fun a => HPow.hPow (f a) p) (MeasureT …
    h_right_g : Eq (MeasureTheory.integral μ fun a => HPow.hPow (g a) q) (MeasureT …
    ⊢ LE.le (MeasureTheory.integral μ fun a => HMul.hMul (Norm.norm (f a)) (Norm.n …
  -/
  exact integral_mul_norm_le_Lp_mul_Lq hpq hf hg
  /-
    🎉 no goals
  -/


theorem integral_countable' [Countable α] [MeasurableSingletonClass α] {μ : Measure α}
    {f : α → E} (hf : Integrable f μ) :
    ∫ a, f a ∂μ = ∑' a, (μ {a}).toReal • f a := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝² : NormedSpace Real E
    m : MeasurableSpace α
    inst✝¹ : Countable α
    inst✝ : MeasurableSingletonClass α
    μ : MeasureTheory.Measure α
    f : α → E
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (tsum fun a => HSMul.hSMul (μ (Si …
  -/
  rw [← Measure.sum_smul_dirac μ] at hf
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝² : NormedSpace Real E
    m : MeasurableSpace α
    inst✝¹ : Countable α
    inst✝ : MeasurableSingletonClass α
    μ : MeasureTheory.Measure α
    f : α → E
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum fun a => HSMul.hSMu …
    ⊢ Eq (MeasureTheory.integral μ fun a => f a) (tsum fun a => HSMul.hSMul (μ (Si …
  -/
  rw [← Measure.sum_smul_dirac μ, integral_sum_measure hf]
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝² : NormedSpace Real E
    m : MeasurableSpace α
    inst✝¹ : Countable α
    inst✝ : MeasurableSingletonClass α
    μ : MeasureTheory.Measure α
    f : α → E
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum fun a => HSMul.hSMu …
    ⊢ Eq (tsum fun i => MeasureTheory.integral (HSMul.hSMul (μ (Singleton.singleto …
  -/
  congr 1 with a : 1
  /-
    case e_f.h
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝² : NormedSpace Real E
    m : MeasurableSpace α
    inst✝¹ : Countable α
    inst✝ : MeasurableSingletonClass α
    μ : MeasureTheory.Measure α
    f : α → E
    hf : MeasureTheory.Integrable f (MeasureTheory.Measure.sum fun a => HSMul.hSMu …
    a : α
    ⊢ Eq (MeasureTheory.integral (HSMul.hSMul (μ (Singleton.singleton a)) (Measure …
  -/
  rw [integral_smul_measure, integral_dirac, Measure.sum_smul_dirac]
  /-
    🎉 no goals
  -/


theorem integral_singleton' {μ : Measure α} {f : α → E} (hf : StronglyMeasurable f) (a : α) :
    ∫ a in {a}, f a ∂μ = (μ {a}).toReal • f a := by
  simp only [Measure.restrict_singleton, integral_smul_measure, integral_dirac' f a hf, smul_eq_mul,
    mul_comm]


theorem integral_singleton [MeasurableSingletonClass α] {μ : Measure α} (f : α → E) (a : α) :
    ∫ a in {a}, f a ∂μ = (μ {a}).toReal • f a := by
  simp only [Measure.restrict_singleton, integral_smul_measure, integral_dirac, smul_eq_mul,
    mul_comm]


theorem integral_countable [MeasurableSingletonClass α] (f : α → E) {s : Set α} (hs : s.Countable)
    (hf : IntegrableOn f s μ) :
    ∫ a in s, f a ∂μ = ∑' a : s, (μ {(a : α)}).toReal • f a := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    s : Set α
    hs : s.Countable
    hf : MeasureTheory.IntegrableOn f s μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun a => f a) (tsum fun a => HSMul …
  -/
  have hi : Countable { x // x ∈ s } := Iff.mpr countable_coe_iff hs
  have hf' : Integrable (fun (x : s) => f x) (Measure.comap Subtype.val μ) := by
    rw [IntegrableOn, ← map_comap_subtype_coe, integrable_map_measure] at hf
    · apply hf
    · exact Integrable.aestronglyMeasurable hf
    · exact Measurable.aemeasurable measurable_subtype_coe
    · exact Countable.measurableSet hs
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    s : Set α
    hs : s.Countable
    hf : MeasureTheory.IntegrableOn f s μ
    hi : Countable (Subtype fun x => Membership.mem s x)
    hf' : MeasureTheory.Integrable (fun x => f ↑x) (MeasureTheory.Measure.comap Su …
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun a => f a) (tsum fun a => HSMul …
  -/
  rw [← integral_subtype_comap hs.measurableSet, integral_countable' hf']
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    s : Set α
    hs : s.Countable
    hf : MeasureTheory.IntegrableOn f s μ
    hi : Countable (Subtype fun x => Membership.mem s x)
    hf' : MeasureTheory.Integrable (fun x => f ↑x) (MeasureTheory.Measure.comap Su …
    ⊢ Eq (tsum fun a => HSMul.hSMul ((MeasureTheory.Measure.comap Subtype.val μ) ( …
  -/
  congr 1 with a : 1
  rw [Measure.comap_apply Subtype.val Subtype.coe_injective
    (fun s' hs' => MeasurableSet.subtype_image (Countable.measurableSet hs) hs') _
    (MeasurableSet.singleton a)]
  /-
    case e_f.h
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSingletonClass α
    f : α → E
    s : Set α
    hs : s.Countable
    hf : MeasureTheory.IntegrableOn f s μ
    hi : Countable (Subtype fun x => Membership.mem s x)
    hf' : MeasureTheory.Integrable (fun x => f ↑x) (MeasureTheory.Measure.comap Su …
    a : ↑s
    ⊢ Eq (HSMul.hSMul (μ (Set.image Subtype.val (Singleton.singleton a))).toReal ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem integral_finset [MeasurableSingletonClass α] (s : Finset α) (f : α → E)
    (hf : IntegrableOn f s μ) :
    ∫ x in s, f x ∂μ = ∑ x ∈ s, (μ {x}).toReal • f x := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSingletonClass α
    s : Finset α
    f : α → E
    hf : MeasureTheory.IntegrableOn f (↑s) μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict ↑s) fun x => f x) (s.sum fun x => HSM …
  -/
  rw [integral_countable _ s.countable_toSet hf, ← Finset.tsum_subtype']
  /-
    🎉 no goals
  -/


theorem integral_fintype [MeasurableSingletonClass α] [Fintype α] (f : α → E)
    (hf : Integrable f μ) :
    ∫ x, f x ∂μ = ∑ x, (μ {x}).toReal • f x := by
  -- NB: Integrable f does not follow from Fintype, because the measure itself could be non-finite
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝² : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSingletonClass α
    inst✝ : Fintype α
    f : α → E
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (Finset.univ.sum fun x => HSMul.h …
  -/
  rw [← integral_finset .univ, Finset.coe_univ, Measure.restrict_univ]
  /-
    case hf
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    hE : CompleteSpace E
    inst✝² : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSingletonClass α
    inst✝ : Fintype α
    f : α → E
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.IntegrableOn f (↑Finset.univ) μ
  -/
  simp [Finset.coe_univ, Measure.restrict_univ, hf]
  /-
    🎉 no goals
  -/


theorem integral_unique [Unique α] (f : α → E) : ∫ x, f x ∂μ = (μ univ).toReal • f default :=
  calc
                                          /-
                                            α : Type u_1
                                            E : Type u_2
                                            inst✝² : NormedAddCommGroup E
                                            hE : CompleteSpace E
                                            inst✝¹ : NormedSpace Real E
                                            m : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            inst✝ : Unique α
                                            f : α → E
                                            ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral μ fun x = …
                                          -/
    ∫ x, f x ∂μ = ∫ _, f default ∂μ := by congr with x; congr; exact Unique.uniq _ x
                                                               /-
                                                                 🎉 no goals
                                                               -/
                                          /-
                                            α : Type u_1
                                            E : Type u_2
                                            inst✝² : NormedAddCommGroup E
                                            hE : CompleteSpace E
                                            inst✝¹ : NormedSpace Real E
                                            m : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            inst✝ : Unique α
                                            f : α → E
                                            ⊢ Eq (MeasureTheory.integral μ fun x => f Inhabited.default) (HSMul.hSMul (μ S …
                                          -/
    _ = (μ univ).toReal • f default := by rw [integral_const]
                                          /-
                                            🎉 no goals
                                          -/


theorem integral_pos_of_integrable_nonneg_nonzero [TopologicalSpace α] [Measure.IsOpenPosMeasure μ]
    {f : α → ℝ} {x : α} (f_cont : Continuous f) (f_int : Integrable f μ) (f_nonneg : 0 ≤ f)
    (f_x : f x ≠ 0) : 0 < ∫ x, f x ∂μ :=
  (integral_pos_iff_support_of_nonneg f_nonneg f_int).2
    (IsOpen.measure_pos μ f_cont.isOpen_support ⟨x, f_x⟩)


/-- Simple function seen as simple function of a larger `MeasurableSpace`. -/
def SimpleFunc.toLargerSpace (hm : m ≤ m0) (f : @SimpleFunc β m γ) : SimpleFunc β γ :=
  ⟨@SimpleFunc.toFun β m γ f, fun x => hm _ (@SimpleFunc.measurableSet_fiber β γ m f x),
    @SimpleFunc.finite_range β γ m f⟩


theorem SimpleFunc.coe_toLargerSpace_eq (hm : m ≤ m0) (f : @SimpleFunc β m γ) :
    ⇑(f.toLargerSpace hm) = f := rfl


theorem integral_simpleFunc_larger_space (hm : m ≤ m0) (f : @SimpleFunc β m F)
    (hf_int : Integrable f μ) :
    ∫ x, f x ∂μ = ∑ x ∈ @SimpleFunc.range β F m f, ENNReal.toReal (μ (f ⁻¹' {x})) • x := by
  /-
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (f.range.sum fun x => HSMul.hSMul …
  -/
  simp_rw [← f.coe_toLargerSpace_eq hm]
  /-
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (MeasureTheory.integral μ fun x => (MeasureTheory.SimpleFunc.toLargerSpac …
  -/
  have hf_int : Integrable (f.toLargerSpace hm) μ := by rwa [SimpleFunc.coe_toLargerSpace_eq]
  /-
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int✝ : MeasureTheory.Integrable (⇑f) μ
    hf_int : MeasureTheory.Integrable (⇑(MeasureTheory.SimpleFunc.toLargerSpace hm …
    ⊢ Eq (MeasureTheory.integral μ fun x => (MeasureTheory.SimpleFunc.toLargerSpac …
  -/
  rw [SimpleFunc.integral_eq_sum _ hf_int]
  /-
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int✝ : MeasureTheory.Integrable (⇑f) μ
    hf_int : MeasureTheory.Integrable (⇑(MeasureTheory.SimpleFunc.toLargerSpace hm …
    ⊢ Eq ((MeasureTheory.SimpleFunc.toLargerSpace hm f).range.sum fun x => HSMul.h …
  -/
  congr 1
  /-
    🎉 no goals
  -/


theorem integral_trim_simpleFunc (hm : m ≤ m0) (f : @SimpleFunc β m F) (hf_int : Integrable f μ) :
    ∫ x, f x ∂μ = ∫ x, f x ∂μ.trim hm := by
  /-
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  have hf : StronglyMeasurable[m] f := @SimpleFunc.stronglyMeasurable β F m _ f
  /-
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int : MeasureTheory.Integrable (⇑f) μ
    hf : MeasureTheory.StronglyMeasurable ⇑f
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  have hf_int_m := hf_int.trim hm hf
  rw [integral_simpleFunc_larger_space (le_refl m) f hf_int_m,
    integral_simpleFunc_larger_space hm f hf_int]
  /-
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int : MeasureTheory.Integrable (⇑f) μ
    hf : MeasureTheory.StronglyMeasurable ⇑f
    hf_int_m : MeasureTheory.Integrable (⇑f) (μ.trim hm)
    ⊢ Eq (f.range.sum fun x => HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.single …
  -/
  congr with x
  /-
    case e_f.h
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int : MeasureTheory.Integrable (⇑f) μ
    hf : MeasureTheory.StronglyMeasurable ⇑f
    hf_int_m : MeasureTheory.Integrable (⇑f) (μ.trim hm)
    x : F
    ⊢ Eq (HSMul.hSMul (μ (Set.preimage (⇑f) (Singleton.singleton x))).toReal x) (H …
  -/
  congr 2
  /-
    case e_f.h.e_a.e_a
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : MeasureTheory.SimpleFunc β F
    hf_int : MeasureTheory.Integrable (⇑f) μ
    hf : MeasureTheory.StronglyMeasurable ⇑f
    hf_int_m : MeasureTheory.Integrable (⇑f) (μ.trim hm)
    x : F
    ⊢ Eq (μ (Set.preimage (⇑f) (Singleton.singleton x))) ((μ.trim hm) (Set.preimag …
  -/
  exact (trim_measurableSet_eq hm (@SimpleFunc.measurableSet_fiber β F m f x)).symm
  /-
    🎉 no goals
  -/


theorem integral_trim (hm : m ≤ m0) {f : β → G} (hf : StronglyMeasurable[m] f) :
    ∫ x, f x ∂μ = ∫ x, f x ∂μ.trim hm := by
  /-
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  by_cases hG : CompleteSpace G; swap
    /-
      case neg
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      β : Type u_6
      m m0 : MeasurableSpace β
      μ : MeasureTheory.Measure β
      hm : LE.le m m0
      f : β → G
      hf : MeasureTheory.StronglyMeasurable f
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/
  /-
    case pos
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  borelize G
  /-
    case pos
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  by_cases hf_int : Integrable f μ
  /-
    case pos
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    hf_int : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  swap
  · have hf_int_m : ¬Integrable f (μ.trim hm) := fun hf_int_m =>
      hf_int (integrable_of_integrable_trim hm hf_int_m)
    /-
      case neg
      G : Type u_5
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      β : Type u_6
      m m0 : MeasurableSpace β
      μ : MeasureTheory.Measure β
      hm : LE.le m m0
      f : β → G
      hf : MeasureTheory.StronglyMeasurable f
      hG : CompleteSpace G
      this✝¹ : MeasurableSpace G := borel G
      this✝ : BorelSpace G
      hf_int : Not (MeasureTheory.Integrable f μ)
      hf_int_m : Not (MeasureTheory.Integrable f (μ.trim hm))
      ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
    -/
    rw [integral_undef hf_int, integral_undef hf_int_m]
    /-
      🎉 no goals
    -/
  /-
    case pos
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    hf_int : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  haveI : SeparableSpace (range f ∪ {0} : Set G) := hf.separableSpace_range_union_singleton
  /-
    case pos
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    hf_int : MeasureTheory.Integrable f μ
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  let f_seq := @SimpleFunc.approxOn G β _ _ _ m _ hf.measurable (range f ∪ {0}) 0 (by simp) _
  have hf_seq_meas : ∀ n, StronglyMeasurable[m] (f_seq n) := fun n =>
    @SimpleFunc.stronglyMeasurable β G m _ (f_seq n)
  have hf_seq_int : ∀ n, Integrable (f_seq n) μ :=
    SimpleFunc.integrable_approxOn_range (hf.mono hm).measurable hf_int
  have hf_seq_int_m : ∀ n, Integrable (f_seq n) (μ.trim hm) := fun n =>
    (hf_seq_int n).trim hm (hf_seq_meas n)
  have hf_seq_eq : ∀ n, ∫ x, f_seq n x ∂μ = ∫ x, f_seq n x ∂μ.trim hm := fun n =>
    integral_trim_simpleFunc hm (f_seq n) (hf_seq_int n)
  have h_lim_1 : atTop.Tendsto (fun n => ∫ x, f_seq n x ∂μ) (𝓝 (∫ x, f x ∂μ)) := by
    refine tendsto_integral_of_L1 f hf_int (Eventually.of_forall hf_seq_int) ?_
    exact SimpleFunc.tendsto_approxOn_range_L1_nnnorm (hf.mono hm).measurable hf_int
  have h_lim_2 : atTop.Tendsto (fun n => ∫ x, f_seq n x ∂μ) (𝓝 (∫ x, f x ∂μ.trim hm)) := by
    simp_rw [hf_seq_eq]
    refine @tendsto_integral_of_L1 β G _ _ m (μ.trim hm) _ f (hf_int.trim hm hf) _ _
      (Eventually.of_forall hf_seq_int_m) ?_
    exact @SimpleFunc.tendsto_approxOn_range_L1_nnnorm β G m _ _ _ f _ _ hf.measurable
      (hf_int.trim hm hf)
  /-
    case pos
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.StronglyMeasurable f
    hG : CompleteSpace G
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    hf_int : MeasureTheory.Integrable f μ
    this : TopologicalSpace.SeparableSpace ↑(Union.union (Set.range f) (Singleton. …
    f_seq : Nat → MeasureTheory.SimpleFunc β G := MeasureTheory.SimpleFunc.approxO …
    hf_seq_meas : ∀ (n : Nat), MeasureTheory.StronglyMeasurable ⇑(f_seq n)
    hf_seq_int : ∀ (n : Nat), MeasureTheory.Integrable (⇑(f_seq n)) μ
    hf_seq_int_m : ∀ (n : Nat), MeasureTheory.Integrable (⇑(f_seq n)) (μ.trim hm)
    hf_seq_eq : ∀ (n : Nat), Eq (MeasureTheory.integral μ fun x => (f_seq n) x) (M …
    h_lim_1 : Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => (f_seq n) …
    h_lim_2 : Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => (f_seq n) …
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  exact tendsto_nhds_unique h_lim_1 h_lim_2
  /-
    🎉 no goals
  -/


theorem integral_trim_ae (hm : m ≤ m0) {f : β → G} (hf : AEStronglyMeasurable f (μ.trim hm)) :
    ∫ x, f x ∂μ = ∫ x, f x ∂μ.trim hm := by
  /-
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.AEStronglyMeasurable f (μ.trim hm)
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.trim h …
  -/
  rw [integral_congr_ae (ae_eq_of_ae_eq_trim hf.ae_eq_mk), integral_congr_ae hf.ae_eq_mk]
  /-
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    β : Type u_6
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    hm : LE.le m m0
    f : β → G
    hf : MeasureTheory.AEStronglyMeasurable f (μ.trim hm)
    ⊢ Eq (MeasureTheory.integral μ fun a => MeasureTheory.AEStronglyMeasurable.mk  …
  -/
  exact integral_trim hm hf.stronglyMeasurable_mk
  /-
    🎉 no goals
  -/


theorem ae_eq_trim_of_stronglyMeasurable [TopologicalSpace γ] [MetrizableSpace γ] (hm : m ≤ m0)
    {f g : β → γ} (hf : StronglyMeasurable[m] f) (hg : StronglyMeasurable[m] g)
    (hfg : f =ᵐ[μ] g) : f =ᵐ[μ.trim hm] g := by
  /-
    β : Type u_6
    γ : Type u_7
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace.MetrizableSpace γ
    hm : LE.le m m0
    f g : β → γ
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq f g
  -/
  rwa [EventuallyEq, ae_iff, trim_measurableSet_eq hm]
  /-
    β : Type u_6
    γ : Type u_7
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace.MetrizableSpace γ
    hm : LE.le m m0
    f g : β → γ
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ MeasurableSet (setOf fun a => Not (Eq (f a) (g a)))
  -/
  exact (hf.measurableSet_eq_fun hg).compl
  /-
    🎉 no goals
  -/


theorem ae_eq_trim_iff [TopologicalSpace γ] [MetrizableSpace γ] (hm : m ≤ m0) {f g : β → γ}
    (hf : StronglyMeasurable[m] f) (hg : StronglyMeasurable[m] g) :
    f =ᵐ[μ.trim hm] g ↔ f =ᵐ[μ] g :=
  ⟨ae_eq_of_ae_eq_trim, ae_eq_trim_of_stronglyMeasurable hm hf hg⟩


theorem ae_le_trim_of_stronglyMeasurable [LinearOrder γ] [TopologicalSpace γ]
    [OrderClosedTopology γ] [PseudoMetrizableSpace γ] (hm : m ≤ m0) {f g : β → γ}
    (hf : StronglyMeasurable[m] f) (hg : StronglyMeasurable[m] g) (hfg : f ≤ᵐ[μ] g) :
    f ≤ᵐ[μ.trim hm] g := by
  /-
    β : Type u_6
    γ : Type u_7
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    inst✝³ : LinearOrder γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : OrderClosedTopology γ
    inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
    hm : LE.le m m0
    f g : β → γ
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyLE f g
  -/
  rwa [EventuallyLE, ae_iff, trim_measurableSet_eq hm]
  /-
    β : Type u_6
    γ : Type u_7
    m m0 : MeasurableSpace β
    μ : MeasureTheory.Measure β
    inst✝³ : LinearOrder γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : OrderClosedTopology γ
    inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
    hm : LE.le m m0
    f g : β → γ
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ MeasurableSet (setOf fun a => Not (LE.le (f a) (g a)))
  -/
  exact (hf.measurableSet_le hg).compl
  /-
    🎉 no goals
  -/


theorem ae_le_trim_iff [LinearOrder γ] [TopologicalSpace γ] [OrderClosedTopology γ]
    [PseudoMetrizableSpace γ] (hm : m ≤ m0) {f g : β → γ} (hf : StronglyMeasurable[m] f)
    (hg : StronglyMeasurable[m] g) : f ≤ᵐ[μ.trim hm] g ↔ f ≤ᵐ[μ] g :=
  ⟨ae_le_of_ae_le_trim, ae_le_trim_of_stronglyMeasurable hm hf hg⟩


theorem eLpNorm_one_le_of_le {r : ℝ≥0} (hfint : Integrable f μ) (hfint' : 0 ≤ ∫ x, f x ∂μ)
    (hf : ∀ᵐ ω ∂μ, f ω ≤ r) : eLpNorm f 1 μ ≤ 2 * μ Set.univ * r := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : NNReal
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
    hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
    ⊢ LE.le (MeasureTheory.eLpNorm f 1 μ) (HMul.hMul (HMul.hMul 2 (μ Set.univ)) ↑r)
  -/
  by_cases hr : r = 0
  · suffices f =ᵐ[μ] 0 by
      rw [eLpNorm_congr_ae this, eLpNorm_zero, hr, ENNReal.coe_zero, mul_zero]
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      r : NNReal
      hfint : MeasureTheory.Integrable f μ
      hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
      hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
      hr : Eq r 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
    -/
    rw [hr] at hf
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      r : NNReal
      hfint : MeasureTheory.Integrable f μ
      hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
      hf : Filter.Eventually (fun ω => LE.le (f ω) ↑0) (MeasureTheory.ae μ)
      hr : Eq r 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
    -/
    norm_cast at hf
    -- Porting note: two lines above were
    --rw [hr, Nonneg.coe_zero] at hf
    have hnegf : ∫ x, -f x ∂μ = 0 := by
      rw [integral_neg, neg_eq_zero]
      exact le_antisymm (integral_nonpos_of_ae hf) hfint'
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      r : NNReal
      hfint : MeasureTheory.Integrable f μ
      hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
      hr : Eq r 0
      hf : Filter.Eventually (fun ω => LE.le (f ω) 0) (MeasureTheory.ae μ)
      hnegf : Eq (MeasureTheory.integral μ fun x => Neg.neg (f x)) 0
      ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
    -/
    have := (integral_eq_zero_iff_of_nonneg_ae ?_ hfint.neg).1 hnegf
      /-
        case pos.refine_2
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        r : NNReal
        hfint : MeasureTheory.Integrable f μ
        hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
        hr : Eq r 0
        hf : Filter.Eventually (fun ω => LE.le (f ω) 0) (MeasureTheory.ae μ)
        hnegf : Eq (MeasureTheory.integral μ fun x => Neg.neg (f x)) 0
        this : (MeasureTheory.ae μ).EventuallyEq (Neg.neg f) 0
        ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
      -/
    · filter_upwards [this] with ω hω
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        r : NNReal
        hfint : MeasureTheory.Integrable f μ
        hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
        hr : Eq r 0
        hf : Filter.Eventually (fun ω => LE.le (f ω) 0) (MeasureTheory.ae μ)
        hnegf : Eq (MeasureTheory.integral μ fun x => Neg.neg (f x)) 0
        this : (MeasureTheory.ae μ).EventuallyEq (Neg.neg f) 0
        ω : α
        hω : Eq (Neg.neg f ω) (0 ω)
        ⊢ Eq (f ω) (0 ω)
      -/
      rwa [Pi.neg_apply, Pi.zero_apply, neg_eq_zero] at hω
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_1
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        r : NNReal
        hfint : MeasureTheory.Integrable f μ
        hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
        hr : Eq r 0
        hf : Filter.Eventually (fun ω => LE.le (f ω) 0) (MeasureTheory.ae μ)
        hnegf : Eq (MeasureTheory.integral μ fun x => Neg.neg (f x)) 0
        ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (Neg.neg f)
      -/
    · filter_upwards [hf] with ω hω
      /-
        case h
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        r : NNReal
        hfint : MeasureTheory.Integrable f μ
        hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
        hr : Eq r 0
        hf : Filter.Eventually (fun ω => LE.le (f ω) 0) (MeasureTheory.ae μ)
        hnegf : Eq (MeasureTheory.integral μ fun x => Neg.neg (f x)) 0
        ω : α
        hω : LE.le (f ω) 0
        ⊢ LE.le (0 ω) (Neg.neg f ω)
      -/
      rwa [Pi.zero_apply, Pi.neg_apply, Right.nonneg_neg_iff]
      /-
        🎉 no goals
      -/
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : NNReal
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
    hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
    hr : Not (Eq r 0)
    ⊢ LE.le (MeasureTheory.eLpNorm f 1 μ) (HMul.hMul (HMul.hMul 2 (μ Set.univ)) ↑r)
  -/
  by_cases hμ : IsFiniteMeasure μ
  /-
    case pos
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : NNReal
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
    hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
    hr : Not (Eq r 0)
    hμ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LE.le (MeasureTheory.eLpNorm f 1 μ) (HMul.hMul (HMul.hMul 2 (μ Set.univ)) ↑r)
  -/
  swap
  · have : μ Set.univ = ∞ := by
      by_contra hμ'
      exact hμ (IsFiniteMeasure.mk <| lt_top_iff_ne_top.2 hμ')
    /-
      case neg
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      r : NNReal
      hfint : MeasureTheory.Integrable f μ
      hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
      hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
      hr : Not (Eq r 0)
      hμ : Not (MeasureTheory.IsFiniteMeasure μ)
      this : Eq (μ Set.univ) Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm f 1 μ) (HMul.hMul (HMul.hMul 2 (μ Set.univ)) ↑r)
    -/
    rw [this, ENNReal.mul_top', if_neg, ENNReal.top_mul', if_neg]
      /-
        case neg
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        r : NNReal
        hfint : MeasureTheory.Integrable f μ
        hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
        hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
        hr : Not (Eq r 0)
        hμ : Not (MeasureTheory.IsFiniteMeasure μ)
        this : Eq (μ Set.univ) Top.top
        ⊢ LE.le (MeasureTheory.eLpNorm f 1 μ) Top.top
      -/
    · exact le_top
      /-
        🎉 no goals
      -/
      /-
        case neg.hnc
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        r : NNReal
        hfint : MeasureTheory.Integrable f μ
        hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
        hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
        hr : Not (Eq r 0)
        hμ : Not (MeasureTheory.IsFiniteMeasure μ)
        this : Eq (μ Set.univ) Top.top
        ⊢ Not (Eq (↑r) 0)
      -/
    · simp [hr]
      /-
        🎉 no goals
      -/
      /-
        case neg.hnc
        α : Type u_1
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        r : NNReal
        hfint : MeasureTheory.Integrable f μ
        hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
        hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
        hr : Not (Eq r 0)
        hμ : Not (MeasureTheory.IsFiniteMeasure μ)
        this : Eq (μ Set.univ) Top.top
        ⊢ Not (Eq 2 0)
      -/
    · norm_num
      /-
        🎉 no goals
      -/
  /-
    case pos
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : NNReal
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
    hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
    hr : Not (Eq r 0)
    hμ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LE.le (MeasureTheory.eLpNorm f 1 μ) (HMul.hMul (HMul.hMul 2 (μ Set.univ)) ↑r)
  -/
  haveI := hμ
  /-
    case pos
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : NNReal
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
    hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
    hr : Not (Eq r 0)
    hμ this : MeasureTheory.IsFiniteMeasure μ
    ⊢ LE.le (MeasureTheory.eLpNorm f 1 μ) (HMul.hMul (HMul.hMul 2 (μ Set.univ)) ↑r)
  -/
  rw [integral_eq_integral_pos_part_sub_integral_neg_part hfint, sub_nonneg] at hfint'
  have hposbdd : ∫ ω, max (f ω) 0 ∂μ ≤ (μ Set.univ).toReal • (r : ℝ) := by
    rw [← integral_const]
    refine integral_mono_ae hfint.real_toNNReal (integrable_const (r : ℝ)) ?_
    filter_upwards [hf] with ω hω using Real.toNNReal_le_iff_le_coe.2 hω
  rw [Memℒp.eLpNorm_eq_integral_rpow_norm one_ne_zero ENNReal.one_ne_top
      (memℒp_one_iff_integrable.2 hfint),
    ENNReal.ofReal_le_iff_le_toReal
      (ENNReal.mul_ne_top (ENNReal.mul_ne_top ENNReal.two_ne_top <| @measure_ne_top _ _ _ hμ _)
        ENNReal.coe_ne_top)]
  simp_rw [ENNReal.one_toReal, _root_.inv_one, Real.rpow_one, Real.norm_eq_abs, ←
    max_zero_add_max_neg_zero_eq_abs_self, ← Real.coe_toNNReal']
  /-
    case pos
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : NNReal
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le (MeasureTheory.integral μ fun a => ↑(Neg.neg (f a)).toNNReal) ( …
    hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
    hr : Not (Eq r 0)
    hμ this : MeasureTheory.IsFiniteMeasure μ
    hposbdd : LE.le (MeasureTheory.integral μ fun ω => Max.max (f ω) 0) (HSMul.hSM …
    ⊢ LE.le (MeasureTheory.integral μ fun a => HAdd.hAdd ↑(f a).toNNReal ↑(Neg.neg …
  -/
  rw [integral_add hfint.real_toNNReal]
  · simp only [Real.coe_toNNReal', ENNReal.toReal_mul, ENNReal.one_toReal, ENNReal.coe_toReal,
      Left.nonneg_neg_iff, Left.neg_nonpos_iff, toReal_ofNat] at hfint' ⊢
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      r : NNReal
      hfint : MeasureTheory.Integrable f μ
      hfint' : LE.le (MeasureTheory.integral μ fun a => Max.max (Neg.neg (f a)) 0) ( …
      hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
      hr : Not (Eq r 0)
      hμ this : MeasureTheory.IsFiniteMeasure μ
      hposbdd : LE.le (MeasureTheory.integral μ fun ω => Max.max (f ω) 0) (HSMul.hSM …
      ⊢ LE.le (HAdd.hAdd (MeasureTheory.integral μ fun a => Max.max (f a) 0) (Measur …
    -/
    refine (add_le_add_left hfint' _).trans ?_
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      r : NNReal
      hfint : MeasureTheory.Integrable f μ
      hfint' : LE.le (MeasureTheory.integral μ fun a => Max.max (Neg.neg (f a)) 0) ( …
      hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
      hr : Not (Eq r 0)
      hμ this : MeasureTheory.IsFiniteMeasure μ
      hposbdd : LE.le (MeasureTheory.integral μ fun ω => Max.max (f ω) 0) (HSMul.hSM …
      ⊢ LE.le (HAdd.hAdd (MeasureTheory.integral μ fun a => Max.max (f a) 0) (Measur …
    -/
    rwa [← two_mul, mul_assoc, mul_le_mul_left (two_pos : (0 : ℝ) < 2)]
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      r : NNReal
      hfint : MeasureTheory.Integrable f μ
      hfint' : LE.le (MeasureTheory.integral μ fun a => ↑(Neg.neg (f a)).toNNReal) ( …
      hf : Filter.Eventually (fun ω => LE.le (f ω) ↑r) (MeasureTheory.ae μ)
      hr : Not (Eq r 0)
      hμ this : MeasureTheory.IsFiniteMeasure μ
      hposbdd : LE.le (MeasureTheory.integral μ fun ω => Max.max (f ω) 0) (HSMul.hSM …
      ⊢ MeasureTheory.Integrable (fun a => ↑(Neg.neg (f a)).toNNReal) μ
    -/
  · exact hfint.neg.sup (integrable_zero _ _ μ)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_one_le_of_le := eLpNorm_one_le_of_le


theorem eLpNorm_one_le_of_le' {r : ℝ} (hfint : Integrable f μ) (hfint' : 0 ≤ ∫ x, f x ∂μ)
    (hf : ∀ᵐ ω ∂μ, f ω ≤ r) : eLpNorm f 1 μ ≤ 2 * μ Set.univ * ENNReal.ofReal r := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : Real
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
    hf : Filter.Eventually (fun ω => LE.le (f ω) r) (MeasureTheory.ae μ)
    ⊢ LE.le (MeasureTheory.eLpNorm f 1 μ) (HMul.hMul (HMul.hMul 2 (μ Set.univ)) (E …
  -/
  refine eLpNorm_one_le_of_le hfint hfint' ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : Real
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
    hf : Filter.Eventually (fun ω => LE.le (f ω) r) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun ω => LE.le (f ω) ↑r.toNNReal) (MeasureTheory.ae μ)
  -/
  simp only [Real.coe_toNNReal', le_max_iff]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    r : Real
    hfint : MeasureTheory.Integrable f μ
    hfint' : LE.le 0 (MeasureTheory.integral μ fun x => f x)
    hf : Filter.Eventually (fun ω => LE.le (f ω) r) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun ω => Or (LE.le (f ω) r) (LE.le (f ω) 0)) (MeasureTheo …
  -/
  filter_upwards [hf] with ω hω using Or.inl hω
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_one_le_of_le' := eLpNorm_one_le_of_le'


attribute [local instance] monadLiftOptionMetaM in
/-- Positivity extension for integrals.

This extension only proves non-negativity, strict positivity is more delicate for integration and
requires more assumptions. -/
@[positivity MeasureTheory.integral _ _]
def evalIntegral : PositivityExt where eval {u α} zα pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(@MeasureTheory.integral $i ℝ _ $inst2 _ _ $f) =>
    let i : Q($i) ← mkFreshExprMVarQ q($i) .syntheticOpaque
    have body : Q(ℝ) := .betaRev f #[i]
    let rbody ← core zα pα body
    let pbody ← rbody.toNonneg
    let pr : Q(∀ x, 0 ≤ $f x) ← mkLambdaFVars #[i] pbody
    assertInstancesCommute
    return .nonnegative q(integral_nonneg $pr)
  | _ => throwError "not MeasureTheory.integral"


