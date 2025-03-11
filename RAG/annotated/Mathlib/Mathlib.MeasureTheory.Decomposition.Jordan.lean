/-- A Jordan decomposition of a measurable space is a pair of mutually singular,
finite measures. -/
@[ext]
structure JordanDecomposition (α : Type*) [MeasurableSpace α] where
  (posPart negPart : Measure α)
  [posPart_finite : IsFiniteMeasure posPart]
  [negPart_finite : IsFiniteMeasure negPart]
  mutuallySingular : posPart ⟂ₘ negPart


instance instZero : Zero (JordanDecomposition α) where zero := ⟨0, 0, MutuallySingular.zero_right⟩


instance instInhabited : Inhabited (JordanDecomposition α) where default := 0


instance instInvolutiveNeg : InvolutiveNeg (JordanDecomposition α) where
  neg j := ⟨j.negPart, j.posPart, j.mutuallySingular.symm⟩
  neg_neg _ := JordanDecomposition.ext rfl rfl


instance instSMul : SMul ℝ≥0 (JordanDecomposition α) where
  smul r j :=
    ⟨r • j.posPart, r • j.negPart,
      MutuallySingular.smul _ (MutuallySingular.smul _ j.mutuallySingular.symm).symm⟩


instance instSMulReal : SMul ℝ (JordanDecomposition α) where
  smul r j := if 0 ≤ r then r.toNNReal • j else -((-r).toNNReal • j)


@[simp]
theorem zero_posPart : (0 : JordanDecomposition α).posPart = 0 :=
  rfl


@[simp]
theorem zero_negPart : (0 : JordanDecomposition α).negPart = 0 :=
  rfl


@[simp]
theorem neg_posPart : (-j).posPart = j.negPart :=
  rfl


@[simp]
theorem neg_negPart : (-j).negPart = j.posPart :=
  rfl


@[simp]
theorem smul_posPart (r : ℝ≥0) : (r • j).posPart = r • j.posPart :=
  rfl


@[simp]
theorem smul_negPart (r : ℝ≥0) : (r • j).negPart = r • j.negPart :=
  rfl


theorem real_smul_def (r : ℝ) (j : JordanDecomposition α) :
    r • j = if 0 ≤ r then r.toNNReal • j else -((-r).toNNReal • j) :=
  rfl


@[simp]
theorem coe_smul (r : ℝ≥0) : (r : ℝ) • j = r • j := by
  -- Porting note: replaced `show`
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    r : NNReal
    ⊢ Eq (HSMul.hSMul (↑r) j) (HSMul.hSMul r j)
  -/
  rw [real_smul_def, if_pos (NNReal.coe_nonneg r), Real.toNNReal_coe]
  /-
    🎉 no goals
  -/


theorem real_smul_nonneg (r : ℝ) (hr : 0 ≤ r) : r • j = r.toNNReal • j :=
  dif_pos hr


theorem real_smul_neg (r : ℝ) (hr : r < 0) : r • j = -((-r).toNNReal • j) :=
  dif_neg (not_le.2 hr)


theorem real_smul_posPart_nonneg (r : ℝ) (hr : 0 ≤ r) :
    (r • j).posPart = r.toNNReal • j.posPart := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (HSMul.hSMul r j).posPart (HSMul.hSMul r.toNNReal j.posPart)
  -/
  rw [real_smul_def, ← smul_posPart, if_pos hr]
  /-
    🎉 no goals
  -/


theorem real_smul_negPart_nonneg (r : ℝ) (hr : 0 ≤ r) :
    (r • j).negPart = r.toNNReal • j.negPart := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (HSMul.hSMul r j).negPart (HSMul.hSMul r.toNNReal j.negPart)
  -/
  rw [real_smul_def, ← smul_negPart, if_pos hr]
  /-
    🎉 no goals
  -/


theorem real_smul_posPart_neg (r : ℝ) (hr : r < 0) :
    (r • j).posPart = (-r).toNNReal • j.negPart := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    r : Real
    hr : LT.lt r 0
    ⊢ Eq (HSMul.hSMul r j).posPart (HSMul.hSMul (Neg.neg r).toNNReal j.negPart)
  -/
  rw [real_smul_def, ← smul_negPart, if_neg (not_le.2 hr), neg_posPart]
  /-
    🎉 no goals
  -/


theorem real_smul_negPart_neg (r : ℝ) (hr : r < 0) :
    (r • j).negPart = (-r).toNNReal • j.posPart := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    r : Real
    hr : LT.lt r 0
    ⊢ Eq (HSMul.hSMul r j).negPart (HSMul.hSMul (Neg.neg r).toNNReal j.posPart)
  -/
  rw [real_smul_def, ← smul_posPart, if_neg (not_le.2 hr), neg_negPart]
  /-
    🎉 no goals
  -/


/-- The signed measure associated with a Jordan decomposition. -/
def toSignedMeasure : SignedMeasure α :=
  j.posPart.toSignedMeasure - j.negPart.toSignedMeasure


theorem toSignedMeasure_zero : (0 : JordanDecomposition α).toSignedMeasure = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    ⊢ Eq (MeasureTheory.JordanDecomposition.toSignedMeasure 0) 0
  -/
  ext1 i hi
  -- Porting note: replaced `erw` by adding further lemmas
  rw [toSignedMeasure, toSignedMeasure_sub_apply hi, zero_posPart, zero_negPart, sub_self,
    VectorMeasure.coe_zero, Pi.zero_apply]


theorem toSignedMeasure_neg : (-j).toSignedMeasure = -j.toSignedMeasure := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    ⊢ Eq (Neg.neg j).toSignedMeasure (Neg.neg j.toSignedMeasure)
  -/
  ext1 i hi
  -- Porting note: removed `rfl` after the `rw` by adding further steps.
  rw [neg_apply, toSignedMeasure, toSignedMeasure, toSignedMeasure_sub_apply hi,
    toSignedMeasure_sub_apply hi, neg_sub, neg_posPart, neg_negPart]


theorem toSignedMeasure_smul (r : ℝ≥0) : (r • j).toSignedMeasure = r • j.toSignedMeasure := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    r : NNReal
    ⊢ Eq (HSMul.hSMul r j).toSignedMeasure (HSMul.hSMul r j.toSignedMeasure)
  -/
  ext1 i hi
  rw [VectorMeasure.smul_apply, toSignedMeasure, toSignedMeasure,
    toSignedMeasure_sub_apply hi, toSignedMeasure_sub_apply hi, smul_sub, smul_posPart,
    smul_negPart, ← ENNReal.toReal_smul, ← ENNReal.toReal_smul, Measure.smul_apply,
    Measure.smul_apply]


/-- A Jordan decomposition provides a Hahn decomposition. -/
theorem exists_compl_positive_negative :
    ∃ S : Set α,
      MeasurableSet S ∧
        j.toSignedMeasure ≤[S] 0 ∧
          0 ≤[Sᶜ] j.toSignedMeasure ∧ j.posPart S = 0 ∧ j.negPart Sᶜ = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    ⊢ Exists fun S => And (MeasurableSet S) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  obtain ⟨S, hS₁, hS₂, hS₃⟩ := j.mutuallySingular
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    j : MeasureTheory.JordanDecomposition α
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : Eq (j.posPart S) 0
    hS₃ : Eq (j.negPart (HasCompl.compl S)) 0
    ⊢ Exists fun S => And (MeasurableSet S) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  refine ⟨S, hS₁, ?_, ?_, hS₂, hS₃⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      j : MeasureTheory.JordanDecomposition α
      S : Set α
      hS₁ : MeasurableSet S
      hS₂ : Eq (j.posPart S) 0
      hS₃ : Eq (j.negPart (HasCompl.compl S)) 0
      ⊢ LE.le (MeasureTheory.VectorMeasure.restrict j.toSignedMeasure S) (MeasureThe …
    -/
  · refine restrict_le_restrict_of_subset_le _ _ fun A hA hA₁ => ?_
    rw [toSignedMeasure, toSignedMeasure_sub_apply hA,
      show j.posPart A = 0 from nonpos_iff_eq_zero.1 (hS₂ ▸ measure_mono hA₁), ENNReal.zero_toReal,
      zero_sub, neg_le, zero_apply, neg_zero]
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      j : MeasureTheory.JordanDecomposition α
      S : Set α
      hS₁ : MeasurableSet S
      hS₂ : Eq (j.posPart S) 0
      hS₃ : Eq (j.negPart (HasCompl.compl S)) 0
      A : Set α
      hA : MeasurableSet A
      hA₁ : HasSubset.Subset A S
      ⊢ LE.le 0 (j.negPart A).toReal
    -/
    exact ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      j : MeasureTheory.JordanDecomposition α
      S : Set α
      hS₁ : MeasurableSet S
      hS₂ : Eq (j.posPart S) 0
      hS₃ : Eq (j.negPart (HasCompl.compl S)) 0
      ⊢ LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl S)) (MeasureTh …
    -/
  · refine restrict_le_restrict_of_subset_le _ _ fun A hA hA₁ => ?_
    rw [toSignedMeasure, toSignedMeasure_sub_apply hA,
      show j.negPart A = 0 from nonpos_iff_eq_zero.1 (hS₃ ▸ measure_mono hA₁), ENNReal.zero_toReal,
      sub_zero]
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      j : MeasureTheory.JordanDecomposition α
      S : Set α
      hS₁ : MeasurableSet S
      hS₂ : Eq (j.posPart S) 0
      hS₃ : Eq (j.negPart (HasCompl.compl S)) 0
      A : Set α
      hA : MeasurableSet A
      hA₁ : HasSubset.Subset A (HasCompl.compl S)
      ⊢ LE.le (↑0 A) (j.posPart A).toReal
    -/
    exact ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/


/-- Given a signed measure `s`, `s.toJordanDecomposition` is the Jordan decomposition `j`,
such that `s = j.toSignedMeasure`. This property is known as the Jordan decomposition
theorem, and is shown by
`MeasureTheory.SignedMeasure.toSignedMeasure_toJordanDecomposition`. -/
def toJordanDecomposition (s : SignedMeasure α) : JordanDecomposition α :=
  let i := s.exists_compl_positive_negative.choose
  let hi := s.exists_compl_positive_negative.choose_spec
  { posPart := s.toMeasureOfZeroLE i hi.1 hi.2.1
    negPart := s.toMeasureOfLEZero iᶜ hi.1.compl hi.2.2
    posPart_finite := inferInstance
    negPart_finite := inferInstance
    mutuallySingular := by
      /-
        α : Type u_1
        inst✝ : MeasurableSpace α
        s✝ s : MeasureTheory.SignedMeasure α
        i : Set α := ⋯.choose
        hi : And (MeasurableSet ⋯.choose) (And (LE.le (MeasureTheory.VectorMeasure.res …
        ⊢ (s.toMeasureOfZeroLE i ⋯ ⋯).MutuallySingular (s.toMeasureOfLEZero (HasCompl. …
      -/
      refine ⟨iᶜ, hi.1.compl, ?_, ?_⟩
      -- Porting note: added `NNReal.eq_iff`
        /-
          case refine_1
          α : Type u_1
          inst✝ : MeasurableSpace α
          s✝ s : MeasureTheory.SignedMeasure α
          i : Set α := ⋯.choose
          hi : And (MeasurableSet ⋯.choose) (And (LE.le (MeasureTheory.VectorMeasure.res …
          ⊢ Eq ((s.toMeasureOfZeroLE i ⋯ ⋯) (HasCompl.compl i)) 0
        -/
      · rw [toMeasureOfZeroLE_apply _ _ hi.1 hi.1.compl]; simp [NNReal.eq_iff]
                                                          /-
                                                            🎉 no goals
                                                          -/
        /-
          case refine_2
          α : Type u_1
          inst✝ : MeasurableSpace α
          s✝ s : MeasureTheory.SignedMeasure α
          i : Set α := ⋯.choose
          hi : And (MeasurableSet ⋯.choose) (And (LE.le (MeasureTheory.VectorMeasure.res …
          ⊢ Eq ((s.toMeasureOfLEZero (HasCompl.compl i) ⋯ ⋯) (HasCompl.compl (HasCompl.c …
        -/
      · rw [toMeasureOfLEZero_apply _ _ hi.1.compl hi.1.compl.compl]; simp [NNReal.eq_iff] }
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem toJordanDecomposition_spec (s : SignedMeasure α) :
    ∃ (i : Set α) (hi₁ : MeasurableSet i) (hi₂ : 0 ≤[i] s) (hi₃ : s ≤[iᶜ] 0),
      s.toJordanDecomposition.posPart = s.toMeasureOfZeroLE i hi₁ hi₂ ∧
        s.toJordanDecomposition.negPart = s.toMeasureOfLEZero iᶜ hi₁.compl hi₃ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    ⊢ Exists fun i => Exists fun hi₁ => Exists fun hi₂ => Exists fun hi₃ => And (E …
  -/
  set i := s.exists_compl_positive_negative.choose
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α := ⋯.choose
    ⊢ Exists fun i => Exists fun hi₁ => Exists fun hi₂ => Exists fun hi₃ => And (E …
  -/
  obtain ⟨hi₁, hi₂, hi₃⟩ := s.exists_compl_positive_negative.choose_spec
  /-
    case intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α := ⋯.choose
    hi₁ : MeasurableSet ⋯.choose
    hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 ⋯.choose) (MeasureTheory.V …
    hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl ⋯.choose)) …
    ⊢ Exists fun i => Exists fun hi₁ => Exists fun hi₂ => Exists fun hi₃ => And (E …
  -/
  exact ⟨i, hi₁, hi₂, hi₃, rfl, rfl⟩
  /-
    🎉 no goals
  -/


/-- **The Jordan decomposition theorem**: Given a signed measure `s`, there exists a pair of
mutually singular measures `μ` and `ν` such that `s = μ - ν`. In this case, the measures `μ`
and `ν` are given by `s.toJordanDecomposition.posPart` and
`s.toJordanDecomposition.negPart` respectively.

Note that we use `MeasureTheory.JordanDecomposition.toSignedMeasure` to represent the
signed measure corresponding to
`s.toJordanDecomposition.posPart - s.toJordanDecomposition.negPart`. -/
@[simp]
theorem toSignedMeasure_toJordanDecomposition (s : SignedMeasure α) :
    s.toJordanDecomposition.toSignedMeasure = s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    ⊢ Eq s.toJordanDecomposition.toSignedMeasure s
  -/
  obtain ⟨i, hi₁, hi₂, hi₃, hμ, hν⟩ := s.toJordanDecomposition_spec
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
    hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
    hμ : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
    hν : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl i …
    ⊢ Eq s.toJordanDecomposition.toSignedMeasure s
  -/
  simp only [JordanDecomposition.toSignedMeasure, hμ, hν]
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
    hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
    hμ : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
    hν : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl i …
    ⊢ Eq (HSub.hSub (s.toMeasureOfZeroLE i hi₁ hi₂).toSignedMeasure (s.toMeasureOf …
  -/
  ext k hk
  rw [toSignedMeasure_sub_apply hk, toMeasureOfZeroLE_apply _ hi₂ hi₁ hk,
    toMeasureOfLEZero_apply _ hi₃ hi₁.compl hk]
  /-
    case intro.intro.intro.intro.intro.h
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
    hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
    hμ : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
    hν : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl i …
    k : Set α
    hk : MeasurableSet k
    ⊢ Eq (HSub.hSub (↑⟨↑s (Inter.inter i k), ⋯⟩).toReal (↑⟨Neg.neg (↑s (Inter.inte …
  -/
  simp only [ENNReal.coe_toReal, NNReal.coe_mk, ENNReal.some_eq_coe, sub_neg_eq_add]
  rw [← of_union _ (MeasurableSet.inter hi₁ hk) (MeasurableSet.inter hi₁.compl hk),
    Set.inter_comm i, Set.inter_comm iᶜ, Set.inter_union_compl _ _]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
    hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
    hμ : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
    hν : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl i …
    k : Set α
    hk : MeasurableSet k
    ⊢ Disjoint (Inter.inter i k) (Inter.inter (HasCompl.compl i) k)
  -/
  exact (disjoint_compl_right.inf_left _).inf_right _
  /-
    🎉 no goals
  -/


/-- A subset `v` of a null-set `w` has zero measure if `w` is a subset of a positive set `u`. -/
theorem subset_positive_null_set (hu : MeasurableSet u) (hv : MeasurableSet v)
    (hw : MeasurableSet w) (hsu : 0 ≤[u] s) (hw₁ : s w = 0) (hw₂ : w ⊆ u) (hwt : v ⊆ w) :
    s v = 0 := by
  have : s v + s (w \ v) = 0 := by
    rw [← hw₁, ← of_union Set.disjoint_sdiff_right hv (hw.diff hv), Set.union_diff_self,
      Set.union_eq_self_of_subset_left hwt]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) (MeasureTheory.VectorMe …
    hw₁ : Eq (↑s w) 0
    hw₂ : HasSubset.Subset w u
    hwt : HasSubset.Subset v w
    this : Eq (HAdd.hAdd (↑s v) (↑s (SDiff.sdiff w v))) 0
    ⊢ Eq (↑s v) 0
  -/
  have h₁ := nonneg_of_zero_le_restrict _ (restrict_le_restrict_subset _ _ hu hsu (hwt.trans hw₂))
  have h₂ : 0 ≤ s (w \ v) :=
    nonneg_of_zero_le_restrict _
      (restrict_le_restrict_subset _ _ hu hsu (diff_subset.trans hw₂))
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) (MeasureTheory.VectorMe …
    hw₁ : Eq (↑s w) 0
    hw₂ : HasSubset.Subset w u
    hwt : HasSubset.Subset v w
    this : Eq (HAdd.hAdd (↑s v) (↑s (SDiff.sdiff w v))) 0
    h₁ : LE.le 0 (↑s v)
    h₂ : LE.le 0 (↑s (SDiff.sdiff w v))
    ⊢ Eq (↑s v) 0
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- A subset `v` of a null-set `w` has zero measure if `w` is a subset of a negative set `u`. -/
theorem subset_negative_null_set (hu : MeasurableSet u) (hv : MeasurableSet v)
    (hw : MeasurableSet w) (hsu : s ≤[u] 0) (hw₁ : s w = 0) (hw₂ : w ⊆ u) (hwt : v ⊆ w) :
    s v = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict s u) (MeasureTheory.VectorMe …
    hw₁ : Eq (↑s w) 0
    hw₂ : HasSubset.Subset w u
    hwt : HasSubset.Subset v w
    ⊢ Eq (↑s v) 0
  -/
  rw [← s.neg_le_neg_iff _ hu, neg_zero] at hsu
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hw₁ : Eq (↑s w) 0
    hw₂ : HasSubset.Subset w u
    hwt : HasSubset.Subset v w
    ⊢ Eq (↑s v) 0
  -/
  have := subset_positive_null_set hu hv hw hsu
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hw₁ : Eq (↑s w) 0
    hw₂ : HasSubset.Subset w u
    hwt : HasSubset.Subset v w
    this : Eq (↑(Neg.neg s) w) 0 → HasSubset.Subset w u → HasSubset.Subset v w → E …
    ⊢ Eq (↑s v) 0
  -/
  simp only [Pi.neg_apply, neg_eq_zero, coe_neg] at this
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hw₁ : Eq (↑s w) 0
    hw₂ : HasSubset.Subset w u
    hwt : HasSubset.Subset v w
    this : Eq (↑s w) 0 → HasSubset.Subset w u → HasSubset.Subset v w → Eq (↑s v) 0
    ⊢ Eq (↑s v) 0
  -/
  exact this hw₁ hw₂ hwt
  /-
    🎉 no goals
  -/


/-- If the symmetric difference of two positive sets is a null-set, then so are the differences
between the two sets. -/
theorem of_diff_eq_zero_of_symmDiff_eq_zero_positive (hu : MeasurableSet u) (hv : MeasurableSet v)
    (hsu : 0 ≤[u] s) (hsv : 0 ≤[v] s) (hs : s (u ∆ v) = 0) : s (u \ v) = 0 ∧ s (v \ u) = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) (MeasureTheory.VectorMe …
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict 0 v) (MeasureTheory.VectorMe …
    hs : Eq (↑s (symmDiff u v)) 0
    ⊢ And (Eq (↑s (SDiff.sdiff u v)) 0) (Eq (↑s (SDiff.sdiff v u)) 0)
  -/
  rw [restrict_le_restrict_iff] at hsu hsv
  on_goal 1 =>
    have a := hsu (hu.diff hv) diff_subset
    have b := hsv (hv.diff hu) diff_subset
    erw [of_union (Set.disjoint_of_subset_left diff_subset disjoint_sdiff_self_right)
        (hu.diff hv) (hv.diff hu)] at hs
    rw [zero_apply] at a b
    constructor
  /-
    case left
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hsu : ∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j u → LE.le (↑0 j) (↑s …
    hsv : ∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j v → LE.le (↑0 j) (↑s …
    hs : Eq (HAdd.hAdd (↑s (SDiff.sdiff u v)) (↑s (SDiff.sdiff v u))) 0
    a : LE.le 0 (↑s (SDiff.sdiff u v))
    b : LE.le 0 (↑s (SDiff.sdiff v u))
    ⊢ Eq (↑s (SDiff.sdiff u v)) 0
  -/
  all_goals first | linarith | assumption
  /-
    🎉 no goals
  -/


/-- If the symmetric difference of two negative sets is a null-set, then so are the differences
between the two sets. -/
theorem of_diff_eq_zero_of_symmDiff_eq_zero_negative (hu : MeasurableSet u) (hv : MeasurableSet v)
    (hsu : s ≤[u] 0) (hsv : s ≤[v] 0) (hs : s (u ∆ v) = 0) : s (u \ v) = 0 ∧ s (v \ u) = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict s u) (MeasureTheory.VectorMe …
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict s v) (MeasureTheory.VectorMe …
    hs : Eq (↑s (symmDiff u v)) 0
    ⊢ And (Eq (↑s (SDiff.sdiff u v)) 0) (Eq (↑s (SDiff.sdiff v u)) 0)
  -/
  rw [← s.neg_le_neg_iff _ hu, neg_zero] at hsu
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict s v) (MeasureTheory.VectorMe …
    hs : Eq (↑s (symmDiff u v)) 0
    ⊢ And (Eq (↑s (SDiff.sdiff u v)) 0) (Eq (↑s (SDiff.sdiff v u)) 0)
  -/
  rw [← s.neg_le_neg_iff _ hv, neg_zero] at hsv
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict 0 v) ((Neg.neg s).restrict v)
    hs : Eq (↑s (symmDiff u v)) 0
    ⊢ And (Eq (↑s (SDiff.sdiff u v)) 0) (Eq (↑s (SDiff.sdiff v u)) 0)
  -/
  have := of_diff_eq_zero_of_symmDiff_eq_zero_positive hu hv hsu hsv
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict 0 v) ((Neg.neg s).restrict v)
    hs : Eq (↑s (symmDiff u v)) 0
    this : Eq (↑(Neg.neg s) (symmDiff u v)) 0 → And (Eq (↑(Neg.neg s) (SDiff.sdiff …
    ⊢ And (Eq (↑s (SDiff.sdiff u v)) 0) (Eq (↑s (SDiff.sdiff v u)) 0)
  -/
  simp only [Pi.neg_apply, neg_eq_zero, coe_neg] at this
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict 0 v) ((Neg.neg s).restrict v)
    hs : Eq (↑s (symmDiff u v)) 0
    this : Eq (↑s (symmDiff u v)) 0 → And (Eq (↑s (SDiff.sdiff u v)) 0) (Eq (↑s (S …
    ⊢ And (Eq (↑s (SDiff.sdiff u v)) 0) (Eq (↑s (SDiff.sdiff v u)) 0)
  -/
  exact this hs
  /-
    🎉 no goals
  -/


theorem of_inter_eq_of_symmDiff_eq_zero_positive (hu : MeasurableSet u) (hv : MeasurableSet v)
    (hw : MeasurableSet w) (hsu : 0 ≤[u] s) (hsv : 0 ≤[v] s) (hs : s (u ∆ v) = 0) :
    s (w ∩ u) = s (w ∩ v) := by
  have hwuv : s ((w ∩ u) ∆ (w ∩ v)) = 0 := by
    refine
      subset_positive_null_set (hu.union hv) ((hw.inter hu).symmDiff (hw.inter hv))
        (hu.symmDiff hv) (restrict_le_restrict_union _ _ hu hsu hv hsv) hs
        Set.symmDiff_subset_union ?_
    rw [← Set.inter_symmDiff_distrib_left]
    exact Set.inter_subset_right
  obtain ⟨huv, hvu⟩ :=
    of_diff_eq_zero_of_symmDiff_eq_zero_positive (hw.inter hu) (hw.inter hv)
      (restrict_le_restrict_subset _ _ hu hsu (w.inter_subset_right))
      (restrict_le_restrict_subset _ _ hv hsv (w.inter_subset_right)) hwuv
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) (MeasureTheory.VectorMe …
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict 0 v) (MeasureTheory.VectorMe …
    hs : Eq (↑s (symmDiff u v)) 0
    hwuv : Eq (↑s (symmDiff (Inter.inter w u) (Inter.inter w v))) 0
    huv : Eq (↑s (SDiff.sdiff (Inter.inter w u) (Inter.inter w v))) 0
    hvu : Eq (↑s (SDiff.sdiff (Inter.inter w v) (Inter.inter w u))) 0
    ⊢ Eq (↑s (Inter.inter w u)) (↑s (Inter.inter w v))
  -/
  rw [← of_diff_of_diff_eq_zero (hw.inter hu) (hw.inter hv) hvu, huv, zero_add]
  /-
    🎉 no goals
  -/


theorem of_inter_eq_of_symmDiff_eq_zero_negative (hu : MeasurableSet u) (hv : MeasurableSet v)
    (hw : MeasurableSet w) (hsu : s ≤[u] 0) (hsv : s ≤[v] 0) (hs : s (u ∆ v) = 0) :
    s (w ∩ u) = s (w ∩ v) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict s u) (MeasureTheory.VectorMe …
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict s v) (MeasureTheory.VectorMe …
    hs : Eq (↑s (symmDiff u v)) 0
    ⊢ Eq (↑s (Inter.inter w u)) (↑s (Inter.inter w v))
  -/
  rw [← s.neg_le_neg_iff _ hu, neg_zero] at hsu
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict s v) (MeasureTheory.VectorMe …
    hs : Eq (↑s (symmDiff u v)) 0
    ⊢ Eq (↑s (Inter.inter w u)) (↑s (Inter.inter w v))
  -/
  rw [← s.neg_le_neg_iff _ hv, neg_zero] at hsv
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict 0 v) ((Neg.neg s).restrict v)
    hs : Eq (↑s (symmDiff u v)) 0
    ⊢ Eq (↑s (Inter.inter w u)) (↑s (Inter.inter w v))
  -/
  have := of_inter_eq_of_symmDiff_eq_zero_positive hu hv hw hsu hsv
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict 0 v) ((Neg.neg s).restrict v)
    hs : Eq (↑s (symmDiff u v)) 0
    this : Eq (↑(Neg.neg s) (symmDiff u v)) 0 → Eq (↑(Neg.neg s) (Inter.inter w u) …
    ⊢ Eq (↑s (Inter.inter w u)) (↑s (Inter.inter w v))
  -/
  simp only [Pi.neg_apply, neg_inj, neg_eq_zero, coe_neg] at this
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    u v w : Set α
    hu : MeasurableSet u
    hv : MeasurableSet v
    hw : MeasurableSet w
    hsu : LE.le (MeasureTheory.VectorMeasure.restrict 0 u) ((Neg.neg s).restrict u)
    hsv : LE.le (MeasureTheory.VectorMeasure.restrict 0 v) ((Neg.neg s).restrict v)
    hs : Eq (↑s (symmDiff u v)) 0
    this : Eq (↑s (symmDiff u v)) 0 → Eq (↑s (Inter.inter w u)) (↑s (Inter.inter w …
    ⊢ Eq (↑s (Inter.inter w u)) (↑s (Inter.inter w v))
  -/
  exact this hs
  /-
    🎉 no goals
  -/


private theorem eq_of_posPart_eq_posPart {j₁ j₂ : JordanDecomposition α}
    (hj : j₁.posPart = j₂.posPart) (hj' : j₁.toSignedMeasure = j₂.toSignedMeasure) : j₁ = j₂ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j₁ j₂ : MeasureTheory.JordanDecomposition α
    hj : Eq j₁.posPart j₂.posPart
    hj' : Eq j₁.toSignedMeasure j₂.toSignedMeasure
    ⊢ Eq j₁ j₂
  -/
  ext1
    /-
      case posPart
      α : Type u_1
      inst✝ : MeasurableSpace α
      j₁ j₂ : MeasureTheory.JordanDecomposition α
      hj : Eq j₁.posPart j₂.posPart
      hj' : Eq j₁.toSignedMeasure j₂.toSignedMeasure
      ⊢ Eq j₁.posPart j₂.posPart
    -/
  · exact hj
    /-
      🎉 no goals
    -/
    /-
      case negPart
      α : Type u_1
      inst✝ : MeasurableSpace α
      j₁ j₂ : MeasureTheory.JordanDecomposition α
      hj : Eq j₁.posPart j₂.posPart
      hj' : Eq j₁.toSignedMeasure j₂.toSignedMeasure
      ⊢ Eq j₁.negPart j₂.negPart
    -/
  · rw [← toSignedMeasure_eq_toSignedMeasure_iff]
    -- Porting note: golfed
    /-
      case negPart
      α : Type u_1
      inst✝ : MeasurableSpace α
      j₁ j₂ : MeasureTheory.JordanDecomposition α
      hj : Eq j₁.posPart j₂.posPart
      hj' : Eq j₁.toSignedMeasure j₂.toSignedMeasure
      ⊢ Eq j₁.negPart.toSignedMeasure j₂.negPart.toSignedMeasure
    -/
    unfold toSignedMeasure at hj'
    /-
      case negPart
      α : Type u_1
      inst✝ : MeasurableSpace α
      j₁ j₂ : MeasureTheory.JordanDecomposition α
      hj : Eq j₁.posPart j₂.posPart
      hj' : Eq (HSub.hSub j₁.posPart.toSignedMeasure j₁.negPart.toSignedMeasure) (HS …
      ⊢ Eq j₁.negPart.toSignedMeasure j₂.negPart.toSignedMeasure
    -/
    simp_rw [hj, sub_right_inj] at hj'
    /-
      case negPart
      α : Type u_1
      inst✝ : MeasurableSpace α
      j₁ j₂ : MeasureTheory.JordanDecomposition α
      hj : Eq j₁.posPart j₂.posPart
      hj' : Eq j₁.negPart.toSignedMeasure j₂.negPart.toSignedMeasure
      ⊢ Eq j₁.negPart.toSignedMeasure j₂.negPart.toSignedMeasure
    -/
    exact hj'
    /-
      🎉 no goals
    -/


/-- The Jordan decomposition of a signed measure is unique. -/
theorem toSignedMeasure_injective : Injective <| @JordanDecomposition.toSignedMeasure α _ := by
  /- The main idea is that two Jordan decompositions of a signed measure provide two
    Hahn decompositions for that measure. Then, from `of_symmDiff_compl_positive_negative`,
    the symmetric difference of the two Hahn decompositions has measure zero, thus, allowing us to
    show the equality of the underlying measures of the Jordan decompositions. -/
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    ⊢ Function.Injective MeasureTheory.JordanDecomposition.toSignedMeasure
  -/
  intro j₁ j₂ hj
  -- obtain the two Hahn decompositions from the Jordan decompositions
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    j₁ j₂ : MeasureTheory.JordanDecomposition α
    hj : Eq j₁.toSignedMeasure j₂.toSignedMeasure
    ⊢ Eq j₁ j₂
  -/
  obtain ⟨S, hS₁, hS₂, hS₃, hS₄, hS₅⟩ := j₁.exists_compl_positive_negative
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    j₁ j₂ : MeasureTheory.JordanDecomposition α
    hj : Eq j₁.toSignedMeasure j₂.toSignedMeasure
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure S) (Measu …
    hS₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl S)) (Measu …
    hS₄ : Eq (j₁.posPart S) 0
    hS₅ : Eq (j₁.negPart (HasCompl.compl S)) 0
    ⊢ Eq j₁ j₂
  -/
  obtain ⟨T, hT₁, hT₂, hT₃, hT₄, hT₅⟩ := j₂.exists_compl_positive_negative
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    j₁ j₂ : MeasureTheory.JordanDecomposition α
    hj : Eq j₁.toSignedMeasure j₂.toSignedMeasure
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure S) (Measu …
    hS₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl S)) (Measu …
    hS₄ : Eq (j₁.posPart S) 0
    hS₅ : Eq (j₁.negPart (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₂.toSignedMeasure T) (Measu …
    hT₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl T)) (Measu …
    hT₄ : Eq (j₂.posPart T) 0
    hT₅ : Eq (j₂.negPart (HasCompl.compl T)) 0
    ⊢ Eq j₁ j₂
  -/
  rw [← hj] at hT₂ hT₃
  -- the symmetric differences of the two Hahn decompositions have measure zero
  obtain ⟨hST₁, -⟩ :=
    of_symmDiff_compl_positive_negative hS₁.compl hT₁.compl ⟨hS₃, (compl_compl S).symm ▸ hS₂⟩
      ⟨hT₃, (compl_compl T).symm ▸ hT₂⟩
  -- it suffices to show the Jordan decompositions have the same positive parts
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    j₁ j₂ : MeasureTheory.JordanDecomposition α
    hj : Eq j₁.toSignedMeasure j₂.toSignedMeasure
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure S) (Measu …
    hS₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl S)) (Measu …
    hS₄ : Eq (j₁.posPart S) 0
    hS₅ : Eq (j₁.negPart (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure T) (Measu …
    hT₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl T)) (Measu …
    hT₄ : Eq (j₂.posPart T) 0
    hT₅ : Eq (j₂.negPart (HasCompl.compl T)) 0
    hST₁ : Eq (↑j₁.toSignedMeasure (symmDiff (HasCompl.compl S) (HasCompl.compl T) …
    ⊢ Eq j₁ j₂
  -/
  refine eq_of_posPart_eq_posPart ?_ hj
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    j₁ j₂ : MeasureTheory.JordanDecomposition α
    hj : Eq j₁.toSignedMeasure j₂.toSignedMeasure
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure S) (Measu …
    hS₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl S)) (Measu …
    hS₄ : Eq (j₁.posPart S) 0
    hS₅ : Eq (j₁.negPart (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure T) (Measu …
    hT₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl T)) (Measu …
    hT₄ : Eq (j₂.posPart T) 0
    hT₅ : Eq (j₂.negPart (HasCompl.compl T)) 0
    hST₁ : Eq (↑j₁.toSignedMeasure (symmDiff (HasCompl.compl S) (HasCompl.compl T) …
    ⊢ Eq j₁.posPart j₂.posPart
  -/
  ext1 i hi
  -- we see that the positive parts of the two Jordan decompositions are equal to their
  -- associated signed measures restricted on their associated Hahn decompositions
  have hμ₁ : (j₁.posPart i).toReal = j₁.toSignedMeasure (i ∩ Sᶜ) := by
    rw [toSignedMeasure, toSignedMeasure_sub_apply (hi.inter hS₁.compl),
      show j₁.negPart (i ∩ Sᶜ) = 0 from
        nonpos_iff_eq_zero.1 (hS₅ ▸ measure_mono Set.inter_subset_right),
      ENNReal.zero_toReal, sub_zero]
    conv_lhs => rw [← Set.inter_union_compl i S]
    rw [measure_union,
      show j₁.posPart (i ∩ S) = 0 from
        nonpos_iff_eq_zero.1 (hS₄ ▸ measure_mono Set.inter_subset_right),
      zero_add]
    · refine
        Set.disjoint_of_subset_left Set.inter_subset_right
          (Set.disjoint_of_subset_right Set.inter_subset_right disjoint_compl_right)
    · exact hi.inter hS₁.compl
  have hμ₂ : (j₂.posPart i).toReal = j₂.toSignedMeasure (i ∩ Tᶜ) := by
    rw [toSignedMeasure, toSignedMeasure_sub_apply (hi.inter hT₁.compl),
      show j₂.negPart (i ∩ Tᶜ) = 0 from
        nonpos_iff_eq_zero.1 (hT₅ ▸ measure_mono Set.inter_subset_right),
      ENNReal.zero_toReal, sub_zero]
    conv_lhs => rw [← Set.inter_union_compl i T]
    rw [measure_union,
      show j₂.posPart (i ∩ T) = 0 from
        nonpos_iff_eq_zero.1 (hT₄ ▸ measure_mono Set.inter_subset_right),
      zero_add]
    · exact
        Set.disjoint_of_subset_left Set.inter_subset_right
          (Set.disjoint_of_subset_right Set.inter_subset_right disjoint_compl_right)
    · exact hi.inter hT₁.compl
  -- since the two signed measures associated with the Jordan decompositions are the same,
  -- and the symmetric difference of the Hahn decompositions have measure zero, the result follows
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.h
    α : Type u_1
    inst✝ : MeasurableSpace α
    j₁ j₂ : MeasureTheory.JordanDecomposition α
    hj : Eq j₁.toSignedMeasure j₂.toSignedMeasure
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure S) (Measu …
    hS₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl S)) (Measu …
    hS₄ : Eq (j₁.posPart S) 0
    hS₅ : Eq (j₁.negPart (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure T) (Measu …
    hT₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl T)) (Measu …
    hT₄ : Eq (j₂.posPart T) 0
    hT₅ : Eq (j₂.negPart (HasCompl.compl T)) 0
    hST₁ : Eq (↑j₁.toSignedMeasure (symmDiff (HasCompl.compl S) (HasCompl.compl T) …
    i : Set α
    hi : MeasurableSet i
    hμ₁ : Eq (j₁.posPart i).toReal (↑j₁.toSignedMeasure (Inter.inter i (HasCompl.c …
    hμ₂ : Eq (j₂.posPart i).toReal (↑j₂.toSignedMeasure (Inter.inter i (HasCompl.c …
    ⊢ Eq (j₁.posPart i) (j₂.posPart i)
  -/
  rw [← ENNReal.toReal_eq_toReal (measure_ne_top _ _) (measure_ne_top _ _), hμ₁, hμ₂, ← hj]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.h
    α : Type u_1
    inst✝ : MeasurableSpace α
    j₁ j₂ : MeasureTheory.JordanDecomposition α
    hj : Eq j₁.toSignedMeasure j₂.toSignedMeasure
    S : Set α
    hS₁ : MeasurableSet S
    hS₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure S) (Measu …
    hS₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl S)) (Measu …
    hS₄ : Eq (j₁.posPart S) 0
    hS₅ : Eq (j₁.negPart (HasCompl.compl S)) 0
    T : Set α
    hT₁ : MeasurableSet T
    hT₂ : LE.le (MeasureTheory.VectorMeasure.restrict j₁.toSignedMeasure T) (Measu …
    hT₃ : LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl T)) (Measu …
    hT₄ : Eq (j₂.posPart T) 0
    hT₅ : Eq (j₂.negPart (HasCompl.compl T)) 0
    hST₁ : Eq (↑j₁.toSignedMeasure (symmDiff (HasCompl.compl S) (HasCompl.compl T) …
    i : Set α
    hi : MeasurableSet i
    hμ₁ : Eq (j₁.posPart i).toReal (↑j₁.toSignedMeasure (Inter.inter i (HasCompl.c …
    hμ₂ : Eq (j₂.posPart i).toReal (↑j₂.toSignedMeasure (Inter.inter i (HasCompl.c …
    ⊢ Eq (↑j₁.toSignedMeasure (Inter.inter i (HasCompl.compl S))) (↑j₁.toSignedMea …
  -/
  exact of_inter_eq_of_symmDiff_eq_zero_positive hS₁.compl hT₁.compl hi hS₃ hT₃ hST₁
  /-
    🎉 no goals
  -/


@[simp]
theorem toJordanDecomposition_toSignedMeasure (j : JordanDecomposition α) :
    j.toSignedMeasure.toJordanDecomposition = j :=
                                                                                /-
                                                                                  α : Type u_1
                                                                                  inst✝ : MeasurableSpace α
                                                                                  j : MeasureTheory.JordanDecomposition α
                                                                                  ⊢ Eq j.toSignedMeasure j.toSignedMeasure.toJordanDecomposition.toSignedMeasure
                                                                                -/
  (@toSignedMeasure_injective _ _ j j.toSignedMeasure.toJordanDecomposition (by simp)).symm
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- `MeasureTheory.SignedMeasure.toJordanDecomposition` and
`MeasureTheory.JordanDecomposition.toSignedMeasure` form an `Equiv`. -/
@[simps apply symm_apply]
def toJordanDecompositionEquiv (α : Type*) [MeasurableSpace α] :
    SignedMeasure α ≃ JordanDecomposition α where
  toFun := toJordanDecomposition
  invFun := toSignedMeasure
  left_inv := toSignedMeasure_toJordanDecomposition
  right_inv := toJordanDecomposition_toSignedMeasure


theorem toJordanDecomposition_zero : (0 : SignedMeasure α).toJordanDecomposition = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    ⊢ Eq (MeasureTheory.SignedMeasure.toJordanDecomposition 0) 0
  -/
  apply toSignedMeasure_injective
  /-
    case a
    α : Type u_1
    inst✝ : MeasurableSpace α
    ⊢ Eq (MeasureTheory.SignedMeasure.toJordanDecomposition 0).toSignedMeasure (Me …
  -/
  simp [toSignedMeasure_zero]
  /-
    🎉 no goals
  -/


theorem toJordanDecomposition_neg (s : SignedMeasure α) :
    (-s).toJordanDecomposition = -s.toJordanDecomposition := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    ⊢ Eq (Neg.neg s).toJordanDecomposition (Neg.neg s.toJordanDecomposition)
  -/
  apply toSignedMeasure_injective
  /-
    case a
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    ⊢ Eq (Neg.neg s).toJordanDecomposition.toSignedMeasure (Neg.neg s.toJordanDeco …
  -/
  simp [toSignedMeasure_neg]
  /-
    🎉 no goals
  -/


theorem toJordanDecomposition_smul (s : SignedMeasure α) (r : ℝ≥0) :
    (r • s).toJordanDecomposition = r • s.toJordanDecomposition := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    r : NNReal
    ⊢ Eq (HSMul.hSMul r s).toJordanDecomposition (HSMul.hSMul r s.toJordanDecompos …
  -/
  apply toSignedMeasure_injective
  /-
    case a
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    r : NNReal
    ⊢ Eq (HSMul.hSMul r s).toJordanDecomposition.toSignedMeasure (HSMul.hSMul r s. …
  -/
  simp [toSignedMeasure_smul]
  /-
    🎉 no goals
  -/


private theorem toJordanDecomposition_smul_real_nonneg (s : SignedMeasure α) (r : ℝ)
    (hr : 0 ≤ r) : (r • s).toJordanDecomposition = r • s.toJordanDecomposition := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (HSMul.hSMul r s).toJordanDecomposition (HSMul.hSMul r s.toJordanDecompos …
  -/
  lift r to ℝ≥0 using hr
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    r : NNReal
    ⊢ Eq (HSMul.hSMul (↑r) s).toJordanDecomposition (HSMul.hSMul (↑r) s.toJordanDe …
  -/
  rw [JordanDecomposition.coe_smul, ← toJordanDecomposition_smul]
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    r : NNReal
    ⊢ Eq (HSMul.hSMul (↑r) s).toJordanDecomposition (HSMul.hSMul r s).toJordanDeco …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toJordanDecomposition_smul_real (s : SignedMeasure α) (r : ℝ) :
    (r • s).toJordanDecomposition = r • s.toJordanDecomposition := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    r : Real
    ⊢ Eq (HSMul.hSMul r s).toJordanDecomposition (HSMul.hSMul r s.toJordanDecompos …
  -/
  by_cases hr : 0 ≤ r
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      r : Real
      hr : LE.le 0 r
      ⊢ Eq (HSMul.hSMul r s).toJordanDecomposition (HSMul.hSMul r s.toJordanDecompos …
    -/
  · exact toJordanDecomposition_smul_real_nonneg s r hr
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      r : Real
      hr : Not (LE.le 0 r)
      ⊢ Eq (HSMul.hSMul r s).toJordanDecomposition (HSMul.hSMul r s.toJordanDecompos …
    -/
  · ext1
    · rw [real_smul_posPart_neg _ _ (not_le.1 hr),
        show r • s = -(-r • s) by rw [neg_smul, neg_neg], toJordanDecomposition_neg, neg_posPart,
        toJordanDecomposition_smul_real_nonneg, ← smul_negPart, real_smul_nonneg]
      /-
        case neg.posPart.hr
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        r : Real
        hr : Not (LE.le 0 r)
        ⊢ LE.le 0 (Neg.neg r)
      -/
      all_goals exact Left.nonneg_neg_iff.2 (le_of_lt (not_le.1 hr))
      /-
        🎉 no goals
      -/
    · rw [real_smul_negPart_neg _ _ (not_le.1 hr),
        show r • s = -(-r • s) by rw [neg_smul, neg_neg], toJordanDecomposition_neg, neg_negPart,
        toJordanDecomposition_smul_real_nonneg, ← smul_posPart, real_smul_nonneg]
      /-
        case neg.negPart.hr
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        r : Real
        hr : Not (LE.le 0 r)
        ⊢ LE.le 0 (Neg.neg r)
      -/
      all_goals exact Left.nonneg_neg_iff.2 (le_of_lt (not_le.1 hr))
      /-
        🎉 no goals
      -/


theorem toJordanDecomposition_eq {s : SignedMeasure α} {j : JordanDecomposition α}
    (h : s = j.toSignedMeasure) : s.toJordanDecomposition = j := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    j : MeasureTheory.JordanDecomposition α
    h : Eq s j.toSignedMeasure
    ⊢ Eq s.toJordanDecomposition j
  -/
  rw [h, toJordanDecomposition_toSignedMeasure]
  /-
    🎉 no goals
  -/


/-- The total variation of a signed measure. -/
def totalVariation (s : SignedMeasure α) : Measure α :=
  s.toJordanDecomposition.posPart + s.toJordanDecomposition.negPart


theorem totalVariation_zero : (0 : SignedMeasure α).totalVariation = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    ⊢ Eq (MeasureTheory.SignedMeasure.totalVariation 0) 0
  -/
  simp [totalVariation, toJordanDecomposition_zero]
  /-
    🎉 no goals
  -/


theorem totalVariation_neg (s : SignedMeasure α) : (-s).totalVariation = s.totalVariation := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    ⊢ Eq (Neg.neg s).totalVariation s.totalVariation
  -/
  simp [totalVariation, toJordanDecomposition_neg, add_comm]
  /-
    🎉 no goals
  -/


theorem null_of_totalVariation_zero (s : SignedMeasure α) {i : Set α}
    (hs : s.totalVariation i = 0) : s i = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hs : Eq (s.totalVariation i) 0
    ⊢ Eq (↑s i) 0
  -/
  rw [totalVariation, Measure.coe_add, Pi.add_apply, add_eq_zero] at hs
  rw [← toSignedMeasure_toJordanDecomposition s, toSignedMeasure, VectorMeasure.coe_sub,
    Pi.sub_apply, Measure.toSignedMeasure_apply, Measure.toSignedMeasure_apply]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hs : And (Eq (s.toJordanDecomposition.posPart i) 0) (Eq (s.toJordanDecompositi …
    ⊢ Eq (HSub.hSub (ite (MeasurableSet i) (s.toJordanDecomposition.posPart i).toR …
  -/
  by_cases hi : MeasurableSet i
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hs : And (Eq (s.toJordanDecomposition.posPart i) 0) (Eq (s.toJordanDecompositi …
      hi : MeasurableSet i
      ⊢ Eq (HSub.hSub (ite (MeasurableSet i) (s.toJordanDecomposition.posPart i).toR …
    -/
  · rw [if_pos hi, if_pos hi]; simp [hs.1, hs.2]
                               /-
                                 🎉 no goals
                               -/
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hs : And (Eq (s.toJordanDecomposition.posPart i) 0) (Eq (s.toJordanDecompositi …
      hi : Not (MeasurableSet i)
      ⊢ Eq (HSub.hSub (ite (MeasurableSet i) (s.toJordanDecomposition.posPart i).toR …
    -/
  · simp [if_neg hi]
    /-
      🎉 no goals
    -/


theorem absolutelyContinuous_ennreal_iff (s : SignedMeasure α) (μ : VectorMeasure α ℝ≥0∞) :
    s ≪ᵥ μ ↔ s.totalVariation ≪ μ.ennrealToMeasure := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.VectorMeasure α ENNReal
    ⊢ Iff (MeasureTheory.VectorMeasure.AbsolutelyContinuous s μ) (s.totalVariation …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : MeasureTheory.VectorMeasure.AbsolutelyContinuous s μ
      ⊢ s.totalVariation.AbsolutelyContinuous μ.ennrealToMeasure
    -/
  · refine Measure.AbsolutelyContinuous.mk fun S hS₁ hS₂ => ?_
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : MeasureTheory.VectorMeasure.AbsolutelyContinuous s μ
      S : Set α
      hS₁ : MeasurableSet S
      hS₂ : Eq (μ.ennrealToMeasure S) 0
      ⊢ Eq (s.totalVariation S) 0
    -/
    obtain ⟨i, hi₁, hi₂, hi₃, hpos, hneg⟩ := s.toJordanDecomposition_spec
    rw [totalVariation, Measure.add_apply, hpos, hneg, toMeasureOfZeroLE_apply _ _ _ hS₁,
      toMeasureOfLEZero_apply _ _ _ hS₁]
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : MeasureTheory.VectorMeasure.AbsolutelyContinuous s μ
      S : Set α
      hS₁ : MeasurableSet S
      hS₂ : Eq (μ.ennrealToMeasure S) 0
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
      hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
      hpos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
      hneg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl …
      ⊢ Eq (HAdd.hAdd ↑⟨↑s (Inter.inter i S), ⋯⟩ ↑⟨Neg.neg (↑s (Inter.inter (HasComp …
    -/
    rw [← VectorMeasure.AbsolutelyContinuous.ennrealToMeasure] at h
    -- Porting note: added `NNReal.eq_iff`
    simp [h (measure_mono_null (i.inter_subset_right) hS₂),
      h (measure_mono_null (iᶜ.inter_subset_right) hS₂), NNReal.eq_iff]
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : s.totalVariation.AbsolutelyContinuous μ.ennrealToMeasure
      ⊢ MeasureTheory.VectorMeasure.AbsolutelyContinuous s μ
    -/
  · refine VectorMeasure.AbsolutelyContinuous.mk fun S hS₁ hS₂ => ?_
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : s.totalVariation.AbsolutelyContinuous μ.ennrealToMeasure
      S : Set α
      hS₁ : MeasurableSet S
      hS₂ : Eq (↑μ S) 0
      ⊢ Eq (↑s S) 0
    -/
    rw [← VectorMeasure.ennrealToMeasure_apply hS₁] at hS₂
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : s.totalVariation.AbsolutelyContinuous μ.ennrealToMeasure
      S : Set α
      hS₁ : MeasurableSet S
      hS₂ : Eq (μ.ennrealToMeasure S) 0
      ⊢ Eq (↑s S) 0
    -/
    exact null_of_totalVariation_zero s (h hS₂)
    /-
      🎉 no goals
    -/


theorem totalVariation_absolutelyContinuous_iff (s : SignedMeasure α) (μ : Measure α) :
    s.totalVariation ≪ μ ↔
      s.toJordanDecomposition.posPart ≪ μ ∧ s.toJordanDecomposition.negPart ≪ μ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.Measure α
    ⊢ Iff (s.totalVariation.AbsolutelyContinuous μ) (And (s.toJordanDecomposition. …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      h : s.totalVariation.AbsolutelyContinuous μ
      ⊢ And (s.toJordanDecomposition.posPart.AbsolutelyContinuous μ) (s.toJordanDeco …
    -/
  · constructor
    all_goals
      refine Measure.AbsolutelyContinuous.mk fun S _ hS₂ => ?_
      have := h hS₂
      rw [totalVariation, Measure.add_apply, add_eq_zero] at this
    /-
      case mp.left
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      h : s.totalVariation.AbsolutelyContinuous μ
      S : Set α
      x✝ : MeasurableSet S
      hS₂ : Eq (μ S) 0
      this : And (Eq (s.toJordanDecomposition.posPart S) 0) (Eq (s.toJordanDecomposi …
      ⊢ Eq (s.toJordanDecomposition.posPart S) 0
    -/
    exacts [this.1, this.2]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      h : And (s.toJordanDecomposition.posPart.AbsolutelyContinuous μ) (s.toJordanDe …
      ⊢ s.totalVariation.AbsolutelyContinuous μ
    -/
  · refine Measure.AbsolutelyContinuous.mk fun S _ hS₂ => ?_
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.Measure α
      h : And (s.toJordanDecomposition.posPart.AbsolutelyContinuous μ) (s.toJordanDe …
      S : Set α
      x✝ : MeasurableSet S
      hS₂ : Eq (μ S) 0
      ⊢ Eq (s.totalVariation S) 0
    -/
    rw [totalVariation, Measure.add_apply, h.1 hS₂, h.2 hS₂, add_zero]
    /-
      🎉 no goals
    -/

-- TODO: Generalize to vector measures once total variation on vector measures is defined

theorem mutuallySingular_iff (s t : SignedMeasure α) :
    s ⟂ᵥ t ↔ s.totalVariation ⟂ₘ t.totalVariation := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s t : MeasureTheory.SignedMeasure α
    ⊢ Iff (MeasureTheory.VectorMeasure.MutuallySingular s t) (s.totalVariation.Mut …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      ⊢ MeasureTheory.VectorMeasure.MutuallySingular s t → s.totalVariation.Mutually …
    -/
  · rintro ⟨u, hmeas, hu₁, hu₂⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      u : Set α
      hmeas : MeasurableSet u
      hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
      hu₂ : ∀ (t_1 : Set α), HasSubset.Subset t_1 (HasCompl.compl u) → Eq (↑t t_1) 0
      ⊢ s.totalVariation.MutuallySingular t.totalVariation
    -/
    obtain ⟨i, hi₁, hi₂, hi₃, hipos, hineg⟩ := s.toJordanDecomposition_spec
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      u : Set α
      hmeas : MeasurableSet u
      hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
      hu₂ : ∀ (t_1 : Set α), HasSubset.Subset t_1 (HasCompl.compl u) → Eq (↑t t_1) 0
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
      hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
      hipos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
      hineg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.comp …
      ⊢ s.totalVariation.MutuallySingular t.totalVariation
    -/
    obtain ⟨j, hj₁, hj₂, hj₃, hjpos, hjneg⟩ := t.toJordanDecomposition_spec
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      α : Type u_1
      inst✝ : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      u : Set α
      hmeas : MeasurableSet u
      hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
      hu₂ : ∀ (t_1 : Set α), HasSubset.Subset t_1 (HasCompl.compl u) → Eq (↑t t_1) 0
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
      hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
      hipos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
      hineg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.comp …
      j : Set α
      hj₁ : MeasurableSet j
      hj₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 j) (MeasureTheory.VectorMe …
      hj₃ : LE.le (MeasureTheory.VectorMeasure.restrict t (HasCompl.compl j)) (Measu …
      hjpos : Eq t.toJordanDecomposition.posPart (t.toMeasureOfZeroLE j hj₁ hj₂)
      hjneg : Eq t.toJordanDecomposition.negPart (t.toMeasureOfLEZero (HasCompl.comp …
      ⊢ s.totalVariation.MutuallySingular t.totalVariation
    -/
    refine ⟨u, hmeas, ?_, ?_⟩
    · rw [totalVariation, Measure.add_apply, hipos, hineg, toMeasureOfZeroLE_apply _ _ _ hmeas,
        toMeasureOfLEZero_apply _ _ _ hmeas]
      -- Porting note: added `NNReal.eq_iff`
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        α : Type u_1
        inst✝ : MeasurableSpace α
        s t : MeasureTheory.SignedMeasure α
        u : Set α
        hmeas : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
        hu₂ : ∀ (t_1 : Set α), HasSubset.Subset t_1 (HasCompl.compl u) → Eq (↑t t_1) 0
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
        hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
        hipos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
        hineg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.comp …
        j : Set α
        hj₁ : MeasurableSet j
        hj₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 j) (MeasureTheory.VectorMe …
        hj₃ : LE.le (MeasureTheory.VectorMeasure.restrict t (HasCompl.compl j)) (Measu …
        hjpos : Eq t.toJordanDecomposition.posPart (t.toMeasureOfZeroLE j hj₁ hj₂)
        hjneg : Eq t.toJordanDecomposition.negPart (t.toMeasureOfLEZero (HasCompl.comp …
        ⊢ Eq (HAdd.hAdd ↑⟨↑s (Inter.inter i u), ⋯⟩ ↑⟨Neg.neg (↑s (Inter.inter (HasComp …
      -/
      simp [hu₁ _ Set.inter_subset_right, NNReal.eq_iff]
      /-
        🎉 no goals
      -/
    · rw [totalVariation, Measure.add_apply, hjpos, hjneg,
        toMeasureOfZeroLE_apply _ _ _ hmeas.compl,
        toMeasureOfLEZero_apply _ _ _ hmeas.compl]
      -- Porting note: added `NNReal.eq_iff`
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
        α : Type u_1
        inst✝ : MeasurableSpace α
        s t : MeasureTheory.SignedMeasure α
        u : Set α
        hmeas : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
        hu₂ : ∀ (t_1 : Set α), HasSubset.Subset t_1 (HasCompl.compl u) → Eq (↑t t_1) 0
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
        hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
        hipos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
        hineg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.comp …
        j : Set α
        hj₁ : MeasurableSet j
        hj₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 j) (MeasureTheory.VectorMe …
        hj₃ : LE.le (MeasureTheory.VectorMeasure.restrict t (HasCompl.compl j)) (Measu …
        hjpos : Eq t.toJordanDecomposition.posPart (t.toMeasureOfZeroLE j hj₁ hj₂)
        hjneg : Eq t.toJordanDecomposition.negPart (t.toMeasureOfLEZero (HasCompl.comp …
        ⊢ Eq (HAdd.hAdd ↑⟨↑t (Inter.inter j (HasCompl.compl u)), ⋯⟩ ↑⟨Neg.neg (↑t (Int …
      -/
      simp [hu₂ _ Set.inter_subset_right, NNReal.eq_iff]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      s t : MeasureTheory.SignedMeasure α
      ⊢ s.totalVariation.MutuallySingular t.totalVariation → MeasureTheory.VectorMea …
    -/
  · rintro ⟨u, hmeas, hu₁, hu₂⟩
    exact
      ⟨u, hmeas, fun t htu => null_of_totalVariation_zero _ (measure_mono_null htu hu₁),
        fun t htv => null_of_totalVariation_zero _ (measure_mono_null htv hu₂)⟩


theorem mutuallySingular_ennreal_iff (s : SignedMeasure α) (μ : VectorMeasure α ℝ≥0∞) :
    s ⟂ᵥ μ ↔ s.totalVariation ⟂ₘ μ.ennrealToMeasure := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    μ : MeasureTheory.VectorMeasure α ENNReal
    ⊢ Iff (MeasureTheory.VectorMeasure.MutuallySingular s μ) (s.totalVariation.Mut …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      ⊢ MeasureTheory.VectorMeasure.MutuallySingular s μ → s.totalVariation.Mutually …
    -/
  · rintro ⟨u, hmeas, hu₁, hu₂⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      u : Set α
      hmeas : MeasurableSet u
      hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
      hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑μ t) 0
      ⊢ s.totalVariation.MutuallySingular μ.ennrealToMeasure
    -/
    obtain ⟨i, hi₁, hi₂, hi₃, hpos, hneg⟩ := s.toJordanDecomposition_spec
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      u : Set α
      hmeas : MeasurableSet u
      hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
      hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑μ t) 0
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
      hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
      hpos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
      hneg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl …
      ⊢ s.totalVariation.MutuallySingular μ.ennrealToMeasure
    -/
    refine ⟨u, hmeas, ?_, ?_⟩
    · rw [totalVariation, Measure.add_apply, hpos, hneg, toMeasureOfZeroLE_apply _ _ _ hmeas,
        toMeasureOfLEZero_apply _ _ _ hmeas]
      -- Porting note: added `NNReal.eq_iff`
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.VectorMeasure α ENNReal
        u : Set α
        hmeas : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
        hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑μ t) 0
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
        hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
        hpos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
        hneg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl …
        ⊢ Eq (HAdd.hAdd ↑⟨↑s (Inter.inter i u), ⋯⟩ ↑⟨Neg.neg (↑s (Inter.inter (HasComp …
      -/
      simp [hu₁ _ Set.inter_subset_right, NNReal.eq_iff]
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.VectorMeasure α ENNReal
        u : Set α
        hmeas : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
        hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑μ t) 0
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
        hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
        hpos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
        hneg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl …
        ⊢ Eq (μ.ennrealToMeasure (HasCompl.compl u)) 0
      -/
    · rw [VectorMeasure.ennrealToMeasure_apply hmeas.compl]
      /-
        case mp.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        μ : MeasureTheory.VectorMeasure α ENNReal
        u : Set α
        hmeas : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑s t) 0
        hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑μ t) 0
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
        hi₃ : LE.le (MeasureTheory.VectorMeasure.restrict s (HasCompl.compl i)) (Measu …
        hpos : Eq s.toJordanDecomposition.posPart (s.toMeasureOfZeroLE i hi₁ hi₂)
        hneg : Eq s.toJordanDecomposition.negPart (s.toMeasureOfLEZero (HasCompl.compl …
        ⊢ Eq (↑μ (HasCompl.compl u)) 0
      -/
      exact hu₂ _ (Set.Subset.refl _)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      ⊢ s.totalVariation.MutuallySingular μ.ennrealToMeasure → MeasureTheory.VectorM …
    -/
  · rintro ⟨u, hmeas, hu₁, hu₂⟩
    refine
      VectorMeasure.MutuallySingular.mk u hmeas
        (fun t htu _ => null_of_totalVariation_zero _ (measure_mono_null htu hu₁)) fun t htv hmt =>
        ?_
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      u : Set α
      hmeas : MeasurableSet u
      hu₁ : Eq (s.totalVariation u) 0
      hu₂ : Eq (μ.ennrealToMeasure (HasCompl.compl u)) 0
      t : Set α
      htv : HasSubset.Subset t (HasCompl.compl u)
      hmt : MeasurableSet t
      ⊢ Eq (↑μ t) 0
    -/
    rw [← VectorMeasure.ennrealToMeasure_apply hmt]
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      μ : MeasureTheory.VectorMeasure α ENNReal
      u : Set α
      hmeas : MeasurableSet u
      hu₁ : Eq (s.totalVariation u) 0
      hu₂ : Eq (μ.ennrealToMeasure (HasCompl.compl u)) 0
      t : Set α
      htv : HasSubset.Subset t (HasCompl.compl u)
      hmt : MeasurableSet t
      ⊢ Eq (μ.ennrealToMeasure t) 0
    -/
    exact measure_mono_null htv hu₂
    /-
      🎉 no goals
    -/


theorem totalVariation_mutuallySingular_iff (s : SignedMeasure α) (μ : Measure α) :
    s.totalVariation ⟂ₘ μ ↔
      s.toJordanDecomposition.posPart ⟂ₘ μ ∧ s.toJordanDecomposition.negPart ⟂ₘ μ :=
  Measure.MutuallySingular.add_left_iff


