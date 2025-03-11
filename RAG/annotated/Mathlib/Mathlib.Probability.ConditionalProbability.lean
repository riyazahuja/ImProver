variable (μ) in
/-- The conditional probability measure of measure `μ` on set `s` is `μ` restricted to `s`
and scaled by the inverse of `μ s` (to make it a probability measure):
`(μ s)⁻¹ • μ.restrict s`. -/
def cond (s : Set Ω) : Measure Ω :=
  (μ s)⁻¹ • μ.restrict s


@[inherit_doc] scoped notation:max μ "[|" s "]" => ProbabilityTheory.cond μ s

@[inherit_doc cond] scoped notation3:max μ "[" t " | " s "]" => ProbabilityTheory.cond μ s t


/-- The conditional probability measure of measure `μ` on `{ω | X ω ∈ s}`.

It is `μ` restricted to `{ω | X ω ∈ s}` and scaled by the inverse of `μ {ω | X ω ∈ s}`
(to make it a probability measure): `(μ {ω | X ω ∈ s})⁻¹ • μ.restrict {ω | X ω ∈ s}`. -/
scoped notation:max μ "[|" X " in " s "]" => μ[|X ⁻¹' s]


/-- The conditional probability measure of measure `μ` on set `{ω | X ω = x}`.

It is `μ` restricted to `{ω | X ω = x}` and scaled by the inverse of `μ {ω | X ω = x}`
(to make it a probability measure): `(μ {ω | X ω = x})⁻¹ • μ.restrict {ω | X ω = x}`. -/
scoped notation:max μ "[" s " | "  X " in " t "]" => μ[s | X ⁻¹' t]


/-- The conditional probability measure of measure `μ` on `{ω | X ω = x}`.

It is `μ` restricted to `{ω | X ω = x}` and scaled by the inverse of `μ {ω | X ω = x}`
(to make it a probability measure): `(μ {ω | X ω = x})⁻¹ • μ.restrict {ω | X ω = x}`. -/
scoped notation:max μ "[|" X " ← " x "]" => μ[|X in {x}]


/-- The conditional probability measure of measure `μ` on set `{ω | X ω = x}`.

It is `μ` restricted to `{ω | X ω = x}` and scaled by the inverse of `μ {ω | X ω = x}`
(to make it a probability measure): `(μ {ω | X ω = x})⁻¹ • μ.restrict {ω | X ω = x}`. -/
scoped notation:max μ "[" s " | "  X " ← " x "]" => μ[s | X in {x}]


/-- The conditional probability measure of any measure on any set of finite positive measure
is a probability measure. -/
theorem cond_isProbabilityMeasure_of_finite (hcs : μ s ≠ 0) (hs : μ s ≠ ∞) :
    IsProbabilityMeasure μ[|s] :=
  ⟨by
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Set Ω
      hcs : Ne (μ s) 0
      hs : Ne (μ s) Top.top
      ⊢ Eq ((ProbabilityTheory.cond μ s) Set.univ) 1
    -/
    unfold ProbabilityTheory.cond
    simp only [Measure.coe_smul, Pi.smul_apply, MeasurableSet.univ, Measure.restrict_apply,
      Set.univ_inter, smul_eq_mul]
    /-
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Set Ω
      hcs : Ne (μ s) 0
      hs : Ne (μ s) Top.top
      ⊢ Eq (HMul.hMul (Inv.inv (μ s)) (μ s)) 1
    -/
    exact ENNReal.inv_mul_cancel hcs hs⟩
    /-
      🎉 no goals
    -/


/-- The conditional probability measure of any finite measure on any set of positive measure
is a probability measure. -/
theorem cond_isProbabilityMeasure [IsFiniteMeasure μ] (hcs : μ s ≠ 0) :
    IsProbabilityMeasure μ[|s] := cond_isProbabilityMeasure_of_finite hcs (measure_ne_top μ s)


instance : IsZeroOrProbabilityMeasure μ[|s] := by
  /-
    Ω : Type u_1
    Ω' : Type u_2
    α : Type u_3
    m : MeasurableSpace Ω
    m' : MeasurableSpace Ω'
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (ProbabilityTheory.cond μ s)
  -/
  constructor
  simp only [cond, Measure.coe_smul, Pi.smul_apply, MeasurableSet.univ, Measure.restrict_apply,
    univ_inter, smul_eq_mul, ← ENNReal.div_eq_inv_mul]
  /-
    case measure_univ
    Ω : Type u_1
    Ω' : Type u_2
    α : Type u_3
    m : MeasurableSpace Ω
    m' : MeasurableSpace Ω'
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    ⊢ Or (Eq (HDiv.hDiv (μ s) (μ s)) 0) (Eq (HDiv.hDiv (μ s) (μ s)) 1)
  -/
  rcases eq_or_ne (μ s) 0 with h | h
    /-
      case measure_univ.inl
      Ω : Type u_1
      Ω' : Type u_2
      α : Type u_3
      m : MeasurableSpace Ω
      m' : MeasurableSpace Ω'
      μ : MeasureTheory.Measure Ω
      s t : Set Ω
      h : Eq (μ s) 0
      ⊢ Or (Eq (HDiv.hDiv (μ s) (μ s)) 0) (Eq (HDiv.hDiv (μ s) (μ s)) 1)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case measure_univ.inr
    Ω : Type u_1
    Ω' : Type u_2
    α : Type u_3
    m : MeasurableSpace Ω
    m' : MeasurableSpace Ω'
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    h : Ne (μ s) 0
    ⊢ Or (Eq (HDiv.hDiv (μ s) (μ s)) 0) (Eq (HDiv.hDiv (μ s) (μ s)) 1)
  -/
  rcases eq_or_ne (μ s) ∞ with h' | h'
    /-
      case measure_univ.inr.inl
      Ω : Type u_1
      Ω' : Type u_2
      α : Type u_3
      m : MeasurableSpace Ω
      m' : MeasurableSpace Ω'
      μ : MeasureTheory.Measure Ω
      s t : Set Ω
      h : Ne (μ s) 0
      h' : Eq (μ s) Top.top
      ⊢ Or (Eq (HDiv.hDiv (μ s) (μ s)) 0) (Eq (HDiv.hDiv (μ s) (μ s)) 1)
    -/
  · simp [h']
    /-
      🎉 no goals
    -/
  /-
    case measure_univ.inr.inr
    Ω : Type u_1
    Ω' : Type u_2
    α : Type u_3
    m : MeasurableSpace Ω
    m' : MeasurableSpace Ω'
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    h : Ne (μ s) 0
    h' : Ne (μ s) Top.top
    ⊢ Or (Eq (HDiv.hDiv (μ s) (μ s)) 0) (Eq (HDiv.hDiv (μ s) (μ s)) 1)
  -/
  simp [ENNReal.div_self h h']
  /-
    🎉 no goals
  -/


variable (μ) in
theorem cond_toMeasurable_eq :
    μ[|(toMeasurable μ s)] = μ[|s] := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Set Ω
    ⊢ Eq (ProbabilityTheory.cond μ (MeasureTheory.toMeasurable μ s)) (ProbabilityT …
  -/
  unfold cond
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Set Ω
    ⊢ Eq (HSMul.hSMul (Inv.inv (μ (MeasureTheory.toMeasurable μ s))) (μ.restrict ( …
  -/
  by_cases hnt : μ s = ∞
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Set Ω
      hnt : Eq (μ s) Top.top
      ⊢ Eq (HSMul.hSMul (Inv.inv (μ (MeasureTheory.toMeasurable μ s))) (μ.restrict ( …
    -/
  · simp [hnt]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Set Ω
      hnt : Not (Eq (μ s) Top.top)
      ⊢ Eq (HSMul.hSMul (Inv.inv (μ (MeasureTheory.toMeasurable μ s))) (μ.restrict ( …
    -/
  · simp [Measure.restrict_toMeasurable hnt]
    /-
      🎉 no goals
    -/


lemma cond_absolutelyContinuous : μ[|s] ≪ μ :=
  smul_absolutelyContinuous.trans restrict_le_self.absolutelyContinuous


lemma absolutelyContinuous_cond_univ [IsFiniteMeasure μ] : μ ≪ μ[|univ] := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ μ.AbsolutelyContinuous (ProbabilityTheory.cond μ Set.univ)
  -/
  rw [cond, restrict_univ]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ μ.AbsolutelyContinuous (HSMul.hSMul (Inv.inv (μ Set.univ)) μ)
  -/
  refine absolutelyContinuous_smul ?_
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Ne (Inv.inv (μ Set.univ)) 0
  -/
  simp [measure_ne_top]
  /-
    🎉 no goals
  -/


variable (μ) in
                                           /-
                                             Ω : Type u_1
                                             m : MeasurableSpace Ω
                                             μ : MeasureTheory.Measure Ω
                                             ⊢ Eq (ProbabilityTheory.cond μ EmptyCollection.emptyCollection) 0
                                           -/
@[simp] lemma cond_empty : μ[|∅] = 0 := by simp [cond]
                                           /-
                                             🎉 no goals
                                           -/


variable (μ) in
@[simp] lemma cond_univ [IsProbabilityMeasure μ] : μ[|Set.univ] = μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq (ProbabilityTheory.cond μ Set.univ) μ
  -/
  simp [cond, measure_univ, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   Ω : Type u_1
                                                                   m : MeasurableSpace Ω
                                                                   μ : MeasureTheory.Measure Ω
                                                                   s : Set Ω
                                                                   ⊢ Iff (Eq (ProbabilityTheory.cond μ s) 0) (Or (Eq (μ s) Top.top) (Eq (μ s) 0))
                                                                 -/
@[simp] lemma cond_eq_zero : μ[|s] = 0 ↔ μ s = ∞ ∨ μ s = 0 := by simp [cond]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                     /-
                                                                       Ω : Type u_1
                                                                       m : MeasurableSpace Ω
                                                                       μ : MeasureTheory.Measure Ω
                                                                       s : Set Ω
                                                                       hμs : Eq (μ s) 0
                                                                       ⊢ Eq (ProbabilityTheory.cond μ s) 0
                                                                     -/
lemma cond_eq_zero_of_meas_eq_zero (hμs : μ s = 0) : μ[|s] = 0 := by simp [hμs]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- The axiomatic definition of conditional probability derived from a measure-theoretic one. -/
theorem cond_apply (hms : MeasurableSet s) (μ : Measure Ω) (t : Set Ω) :
    μ[t|s] = (μ s)⁻¹ * μ (s ∩ t) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    s : Set Ω
    hms : MeasurableSet s
    μ : MeasureTheory.Measure Ω
    t : Set Ω
    ⊢ Eq ((ProbabilityTheory.cond μ s) t) (HMul.hMul (Inv.inv (μ s)) (μ (Inter.int …
  -/
  rw [cond, Measure.smul_apply, Measure.restrict_apply' hms, Set.inter_comm, smul_eq_mul]
  /-
    🎉 no goals
  -/


theorem cond_apply' (ht : MeasurableSet t) (μ : Measure Ω) : μ[t|s] = (μ s)⁻¹ * μ (s ∩ t) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    s t : Set Ω
    ht : MeasurableSet t
    μ : MeasureTheory.Measure Ω
    ⊢ Eq ((ProbabilityTheory.cond μ s) t) (HMul.hMul (Inv.inv (μ s)) (μ (Inter.int …
  -/
  rw [cond, Measure.smul_apply, Measure.restrict_apply ht, Set.inter_comm, smul_eq_mul]
  /-
    🎉 no goals
  -/


@[simp] lemma cond_apply_self (hs₀ : μ s ≠ 0) (hs : μ s ≠ ∞) : μ[s|s] = 1 := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Set Ω
    hs₀ : Ne (μ s) 0
    hs : Ne (μ s) Top.top
    ⊢ Eq ((ProbabilityTheory.cond μ s) s) 1
  -/
  simpa [cond] using ENNReal.inv_mul_cancel hs₀ hs
  /-
    🎉 no goals
  -/


theorem cond_inter_self (hms : MeasurableSet s) (t : Set Ω) (μ : Measure Ω) :
    μ[s ∩ t|s] = μ[t|s] := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    s : Set Ω
    hms : MeasurableSet s
    t : Set Ω
    μ : MeasureTheory.Measure Ω
    ⊢ Eq ((ProbabilityTheory.cond μ s) (Inter.inter s t)) ((ProbabilityTheory.cond …
  -/
  rw [cond_apply hms, ← Set.inter_assoc, Set.inter_self, ← cond_apply hms]
  /-
    🎉 no goals
  -/


theorem inter_pos_of_cond_ne_zero (hms : MeasurableSet s) (hcst : μ[t|s] ≠ 0) : 0 < μ (s ∩ t) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    hms : MeasurableSet s
    hcst : Ne ((ProbabilityTheory.cond μ s) t) 0
    ⊢ LT.lt 0 (μ (Inter.inter s t))
  -/
  refine pos_iff_ne_zero.mpr (right_ne_zero_of_mul (a := (μ s)⁻¹) ?_)
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    hms : MeasurableSet s
    hcst : Ne ((ProbabilityTheory.cond μ s) t) 0
    ⊢ Ne (HMul.hMul (Inv.inv (μ s)) (μ (Inter.inter s t))) 0
  -/
  convert hcst
  /-
    case h.e'_2
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    hms : MeasurableSet s
    hcst : Ne ((ProbabilityTheory.cond μ s) t) 0
    ⊢ Eq (HMul.hMul (Inv.inv (μ s)) (μ (Inter.inter s t))) ((ProbabilityTheory.con …
  -/
  simp [hms, Set.inter_comm, cond]
  /-
    🎉 no goals
  -/


lemma cond_pos_of_inter_ne_zero [IsFiniteMeasure μ] (hms : MeasurableSet s) (hci : μ (s ∩ t) ≠ 0) :
    0 < μ[t | s] := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hms : MeasurableSet s
    hci : Ne (μ (Inter.inter s t)) 0
    ⊢ LT.lt 0 ((ProbabilityTheory.cond μ s) t)
  -/
  rw [cond_apply hms]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hms : MeasurableSet s
    hci : Ne (μ (Inter.inter s t)) 0
    ⊢ LT.lt 0 (HMul.hMul (Inv.inv (μ s)) (μ (Inter.inter s t)))
  -/
  refine ENNReal.mul_pos ?_ hci
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hms : MeasurableSet s
    hci : Ne (μ (Inter.inter s t)) 0
    ⊢ Ne (Inv.inv (μ s)) 0
  -/
  exact ENNReal.inv_ne_zero.mpr (measure_ne_top _ _)
  /-
    🎉 no goals
  -/


lemma cond_cond_eq_cond_inter' (hms : MeasurableSet s) (hmt : MeasurableSet t) (hcs : μ s ≠ ∞) :
    μ[|s][|t] = μ[|s ∩ t] := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    hms : MeasurableSet s
    hmt : MeasurableSet t
    hcs : Ne (μ s) Top.top
    ⊢ Eq (ProbabilityTheory.cond (ProbabilityTheory.cond μ s) t) (ProbabilityTheor …
  -/
  ext u
  /-
    case h
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    hms : MeasurableSet s
    hmt : MeasurableSet t
    hcs : Ne (μ s) Top.top
    u : Set Ω
    a✝ : MeasurableSet u
    ⊢ Eq ((ProbabilityTheory.cond (ProbabilityTheory.cond μ s) t) u) ((Probability …
  -/
  rw [cond_apply hmt, cond_apply hms, cond_apply hms, cond_apply (hms.inter hmt)]
  /-
    case h
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s t : Set Ω
    hms : MeasurableSet s
    hmt : MeasurableSet t
    hcs : Ne (μ s) Top.top
    u : Set Ω
    a✝ : MeasurableSet u
    ⊢ Eq (HMul.hMul (Inv.inv (HMul.hMul (Inv.inv (μ s)) (μ (Inter.inter s t)))) (H …
  -/
  obtain hst | hst := eq_or_ne (μ (s ∩ t)) 0
    /-
      case h.inl
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s t : Set Ω
      hms : MeasurableSet s
      hmt : MeasurableSet t
      hcs : Ne (μ s) Top.top
      u : Set Ω
      a✝ : MeasurableSet u
      hst : Eq (μ (Inter.inter s t)) 0
      ⊢ Eq (HMul.hMul (Inv.inv (HMul.hMul (Inv.inv (μ s)) (μ (Inter.inter s t)))) (H …
    -/
  · have : μ (s ∩ t ∩ u) = 0 := measure_mono_null Set.inter_subset_left hst
    /-
      case h.inl
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s t : Set Ω
      hms : MeasurableSet s
      hmt : MeasurableSet t
      hcs : Ne (μ s) Top.top
      u : Set Ω
      a✝ : MeasurableSet u
      hst : Eq (μ (Inter.inter s t)) 0
      this : Eq (μ (Inter.inter (Inter.inter s t) u)) 0
      ⊢ Eq (HMul.hMul (Inv.inv (HMul.hMul (Inv.inv (μ s)) (μ (Inter.inter s t)))) (H …
    -/
    simp [this, ← Set.inter_assoc]
    /-
      🎉 no goals
    -/
  · have hcs' : μ s ≠ 0 :=
      (measure_pos_of_superset Set.inter_subset_left hst).ne'
    simp [*, ← mul_assoc, ← Set.inter_assoc, ENNReal.mul_inv, ENNReal.mul_inv_cancel,
      mul_right_comm _ _ (μ s)⁻¹]


/-- Conditioning first on `s` and then on `t` results in the same measure as conditioning
on `s ∩ t`. -/
theorem cond_cond_eq_cond_inter (hms : MeasurableSet s) (hmt : MeasurableSet t) (μ : Measure Ω)
    [IsFiniteMeasure μ] : μ[|s][|t] = μ[|s ∩ t] :=
  cond_cond_eq_cond_inter' hms hmt (measure_ne_top μ s)


theorem cond_mul_eq_inter' (hms : MeasurableSet s) (hcs' : μ s ≠ ∞) (t : Set Ω) :
    μ[t|s] * μ s = μ (s ∩ t) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Set Ω
    hms : MeasurableSet s
    hcs' : Ne (μ s) Top.top
    t : Set Ω
    ⊢ Eq (HMul.hMul ((ProbabilityTheory.cond μ s) t) (μ s)) (μ (Inter.inter s t))
  -/
  obtain hcs | hcs := eq_or_ne (μ s) 0
    /-
      case inl
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Set Ω
      hms : MeasurableSet s
      hcs' : Ne (μ s) Top.top
      t : Set Ω
      hcs : Eq (μ s) 0
      ⊢ Eq (HMul.hMul ((ProbabilityTheory.cond μ s) t) (μ s)) (μ (Inter.inter s t))
    -/
  · simp [hcs, measure_inter_null_of_null_left]
    /-
      🎉 no goals
    -/
    /-
      case inr
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Set Ω
      hms : MeasurableSet s
      hcs' : Ne (μ s) Top.top
      t : Set Ω
      hcs : Ne (μ s) 0
      ⊢ Eq (HMul.hMul ((ProbabilityTheory.cond μ s) t) (μ s)) (μ (Inter.inter s t))
    -/
  · rw [cond_apply hms, mul_comm, ← mul_assoc, ENNReal.mul_inv_cancel hcs hcs', one_mul]
    /-
      🎉 no goals
    -/


theorem cond_mul_eq_inter (hms : MeasurableSet s) (t : Set Ω) (μ : Measure Ω) [IsFiniteMeasure μ] :
    μ[t|s] * μ s = μ (s ∩ t) := cond_mul_eq_inter' hms (measure_ne_top _ s) t


/-- A version of the law of total probability. -/
theorem cond_add_cond_compl_eq (hms : MeasurableSet s) (μ : Measure Ω) [IsFiniteMeasure μ] :
    μ[t|s] * μ s + μ[t|sᶜ] * μ sᶜ = μ t := by
  rw [cond_mul_eq_inter hms, cond_mul_eq_inter hms.compl, Set.inter_comm _ t,
    Set.inter_comm _ t]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    s t : Set Ω
    hms : MeasurableSet s
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (HAdd.hAdd (μ (Inter.inter t s)) (μ (Inter.inter t (HasCompl.compl s))))  …
  -/
  exact measure_inter_add_diff t hms
  /-
    🎉 no goals
  -/


/-- **Bayes' Theorem** -/
theorem cond_eq_inv_mul_cond_mul (hms : MeasurableSet s) (hmt : MeasurableSet t) (μ : Measure Ω)
    [IsFiniteMeasure μ] : μ[t|s] = (μ s)⁻¹ * μ[s|t] * μ t := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    s t : Set Ω
    hms : MeasurableSet s
    hmt : MeasurableSet t
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq ((ProbabilityTheory.cond μ s) t) (HMul.hMul (HMul.hMul (Inv.inv (μ s)) (( …
  -/
  rw [mul_assoc, cond_mul_eq_inter hmt s, Set.inter_comm, cond_apply hms]
  /-
    🎉 no goals
  -/


lemma comap_cond {i : Ω' → Ω} (hi : MeasurableEmbedding i) (hi' : ∀ᵐ ω ∂μ, ω ∈ range i)
    (hs : MeasurableSet s) : comap i μ[|s] = (comap i μ)[|i in s] := by
  /-
    Ω : Type u_1
    Ω' : Type u_2
    m : MeasurableSpace Ω
    m' : MeasurableSpace Ω'
    μ : MeasureTheory.Measure Ω
    s : Set Ω
    i : Ω' → Ω
    hi : MeasurableEmbedding i
    hi' : Filter.Eventually (fun ω => Membership.mem (Set.range i) ω) (MeasureTheo …
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.Measure.comap i (ProbabilityTheory.cond μ s)) (Probability …
  -/
  ext t ht
  /-
    case h
    Ω : Type u_1
    Ω' : Type u_2
    m : MeasurableSpace Ω
    m' : MeasurableSpace Ω'
    μ : MeasureTheory.Measure Ω
    s : Set Ω
    i : Ω' → Ω
    hi : MeasurableEmbedding i
    hi' : Filter.Eventually (fun ω => Membership.mem (Set.range i) ω) (MeasureTheo …
    hs : MeasurableSet s
    t : Set Ω'
    ht : MeasurableSet t
    ⊢ Eq ((MeasureTheory.Measure.comap i (ProbabilityTheory.cond μ s)) t) ((Probab …
  -/
  change μ (range i)ᶜ = 0 at hi'
  rw [cond_apply, comap_apply, cond_apply, comap_apply, comap_apply, image_inter,
    image_preimage_eq_inter_range, inter_right_comm, measure_inter_conull hi',
    measure_inter_conull hi']
  all_goals first
  | exact hi.injective
  | exact hi.measurableSet_image'
  | exact hs
  | exact ht
  | exact hi.measurable hs
  | exact (hi.measurable hs).inter ht


/-- The **law of total probability** for a random variable taking finitely many values: a measure
`μ` can be expressed as a linear combination of its conditional measures `μ[|X ← x]` on fibers of a
random variable `X` valued in a fintype. -/
lemma sum_meas_smul_cond_fiber {X : Ω → α} (hX : Measurable X) (μ : Measure Ω) [IsFiniteMeasure μ] :
    ∑ x, μ (X ⁻¹' {x}) • μ[|X ← x] = μ := by
  /-
    Ω : Type u_1
    α : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : Fintype α
    inst✝² : MeasurableSpace α
    inst✝¹ : DiscreteMeasurableSpace α
    X : Ω → α
    hX : Measurable X
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (μ (Set.preimage X (Singleton.singl …
  -/
  ext E hE
  calc
    _ = ∑ x, μ (X ⁻¹' {x} ∩ E) := by
      simp only [Measure.coe_finset_sum, Measure.coe_smul, Finset.sum_apply,
        Pi.smul_apply, smul_eq_mul]
      simp_rw [mul_comm (μ _), cond_mul_eq_inter (hX (.singleton _))]
    _ = _ := by
      have : ⋃ x ∈ Finset.univ, X ⁻¹' {x} ∩ E = E := by ext; simp
      rw [← measure_biUnion_finset _ fun _ _ ↦ (hX (.singleton _)).inter hE, this]
      aesop (add simp [PairwiseDisjoint, Set.Pairwise, Function.onFun, disjoint_left])


