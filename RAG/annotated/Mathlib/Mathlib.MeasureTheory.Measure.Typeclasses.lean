/-- A measure `μ` is called finite if `μ univ < ∞`. -/
class IsFiniteMeasure (μ : Measure α) : Prop where
  measure_univ_lt_top : μ univ < ∞


theorem not_isFiniteMeasure_iff : ¬IsFiniteMeasure μ ↔ μ Set.univ = ∞ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Iff (Not (MeasureTheory.IsFiniteMeasure μ)) (Eq (μ Set.univ) Top.top)
  -/
  refine ⟨fun h => ?_, fun h => fun h' => h'.measure_univ_lt_top.ne h⟩
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    h : Not (MeasureTheory.IsFiniteMeasure μ)
    ⊢ Eq (μ Set.univ) Top.top
  -/
  by_contra h'
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    h : Not (MeasureTheory.IsFiniteMeasure μ)
    h' : Not (Eq (μ Set.univ) Top.top)
    ⊢ False
  -/
  exact h ⟨lt_top_iff_ne_top.mpr h'⟩
  /-
    🎉 no goals
  -/


instance Restrict.isFiniteMeasure (μ : Measure α) [hs : Fact (μ s < ∞)] :
    IsFiniteMeasure (μ.restrict s) :=
      /-
        α : Type u_1
        β : Type u_2
        δ : Type u_3
        ι : Type u_4
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace β
        μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
        s t : Set α
        μ : MeasureTheory.Measure α
        hs : Fact (LT.lt (μ s) Top.top)
        ⊢ LT.lt ((μ.restrict s) Set.univ) Top.top
      -/
  ⟨by simpa using hs.elim⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem measure_lt_top (μ : Measure α) [IsFiniteMeasure μ] (s : Set α) : μ s < ∞ :=
  (measure_mono (subset_univ s)).trans_lt IsFiniteMeasure.measure_univ_lt_top


instance isFiniteMeasureRestrict (μ : Measure α) (s : Set α) [h : IsFiniteMeasure μ] :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            δ : Type u_3
                                            ι : Type u_4
                                            m0 : MeasurableSpace α
                                            inst✝ : MeasurableSpace β
                                            μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
                                            s✝ t : Set α
                                            μ : MeasureTheory.Measure α
                                            s : Set α
                                            h : MeasureTheory.IsFiniteMeasure μ
                                            ⊢ LT.lt ((μ.restrict s) Set.univ) Top.top
                                          -/
    IsFiniteMeasure (μ.restrict s) := ⟨by simp⟩
                                          /-
                                            🎉 no goals
                                          -/


@[simp, aesop (rule_sets := [finiteness]) safe apply]
theorem measure_ne_top (μ : Measure α) [IsFiniteMeasure μ] (s : Set α) : μ s ≠ ∞ :=
  ne_of_lt (measure_lt_top μ s)


theorem measure_compl_le_add_of_le_add [IsFiniteMeasure μ] (hs : MeasurableSet s)
    (ht : MeasurableSet t) {ε : ℝ≥0∞} (h : μ s ≤ μ t + ε) : μ tᶜ ≤ μ sᶜ + ε := by
  rw [measure_compl ht (measure_ne_top μ _), measure_compl hs (measure_ne_top μ _),
    tsub_le_iff_right]
  calc
    μ univ = μ univ - μ s + μ s := (tsub_add_cancel_of_le <| measure_mono s.subset_univ).symm
    _ ≤ μ univ - μ s + (μ t + ε) := add_le_add_left h _
    _ = _ := by rw [add_right_comm, add_assoc]


theorem measure_compl_le_add_iff [IsFiniteMeasure μ] (hs : MeasurableSet s) (ht : MeasurableSet t)
    {ε : ℝ≥0∞} : μ sᶜ ≤ μ tᶜ + ε ↔ μ t ≤ μ s + ε :=
  ⟨fun h => compl_compl s ▸ compl_compl t ▸ measure_compl_le_add_of_le_add hs.compl ht.compl h,
    measure_compl_le_add_of_le_add ht hs⟩


/-- The measure of the whole space with respect to a finite measure, considered as `ℝ≥0`. -/
def measureUnivNNReal (μ : Measure α) : ℝ≥0 :=
  (μ univ).toNNReal


@[simp]
theorem coe_measureUnivNNReal (μ : Measure α) [IsFiniteMeasure μ] :
    ↑(measureUnivNNReal μ) = μ univ :=
  ENNReal.coe_toNNReal (measure_ne_top μ univ)


instance isFiniteMeasureZero : IsFiniteMeasure (0 : Measure α) :=
      /-
        α : Type u_1
        β : Type u_2
        δ : Type u_3
        ι : Type u_4
        m0 : MeasurableSpace α
        inst✝ : MeasurableSpace β
        μ ν ν₁ ν₂ : MeasureTheory.Measure α
        s t : Set α
        ⊢ LT.lt (0 Set.univ) Top.top
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


instance (priority := 50) isFiniteMeasureOfIsEmpty [IsEmpty α] : IsFiniteMeasure μ := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝ : IsEmpty α
    ⊢ MeasureTheory.IsFiniteMeasure μ
  -/
  rw [eq_zero_of_isEmpty μ]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝ : IsEmpty α
    ⊢ MeasureTheory.IsFiniteMeasure 0
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem measureUnivNNReal_zero : measureUnivNNReal (0 : Measure α) = 0 :=
  rfl


instance isFiniteMeasureAdd [IsFiniteMeasure μ] [IsFiniteMeasure ν] : IsFiniteMeasure (μ + ν) where
  measure_univ_lt_top := by
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      ⊢ LT.lt ((HAdd.hAdd μ ν) Set.univ) Top.top
    -/
    rw [Measure.coe_add, Pi.add_apply, ENNReal.add_lt_top]
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      ⊢ And (LT.lt (μ Set.univ) Top.top) (LT.lt (ν Set.univ) Top.top)
    -/
    exact ⟨measure_lt_top _ _, measure_lt_top _ _⟩
    /-
      🎉 no goals
    -/


instance isFiniteMeasureSMulNNReal [IsFiniteMeasure μ] {r : ℝ≥0} : IsFiniteMeasure (r • μ) where
  measure_univ_lt_top := ENNReal.mul_lt_top ENNReal.coe_lt_top (measure_lt_top _ _)


instance IsFiniteMeasure.average : IsFiniteMeasure ((μ univ)⁻¹ • μ) where
  measure_univ_lt_top := by
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      ⊢ LT.lt ((HSMul.hSMul (Inv.inv (μ Set.univ)) μ) Set.univ) Top.top
    -/
    rw [smul_apply, smul_eq_mul, ← ENNReal.div_eq_inv_mul]
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      ⊢ LT.lt (HDiv.hDiv (μ Set.univ) (μ Set.univ)) Top.top
    -/
    exact ENNReal.div_self_le_one.trans_lt ENNReal.one_lt_top
    /-
      🎉 no goals
    -/


instance isFiniteMeasureSMulOfNNRealTower {R} [SMul R ℝ≥0] [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0 ℝ≥0∞]
    [IsScalarTower R ℝ≥0∞ ℝ≥0∞] [IsFiniteMeasure μ] {r : R} : IsFiniteMeasure (r • μ) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝⁵ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    R : Type u_5
    inst✝⁴ : SMul R NNReal
    inst✝³ : SMul R ENNReal
    inst✝² : IsScalarTower R NNReal ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    r : R
    ⊢ MeasureTheory.IsFiniteMeasure (HSMul.hSMul r μ)
  -/
  rw [← smul_one_smul ℝ≥0 r μ]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝⁵ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    R : Type u_5
    inst✝⁴ : SMul R NNReal
    inst✝³ : SMul R ENNReal
    inst✝² : IsScalarTower R NNReal ENNReal
    inst✝¹ : IsScalarTower R ENNReal ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    r : R
    ⊢ MeasureTheory.IsFiniteMeasure (HSMul.hSMul (HSMul.hSMul r 1) μ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem isFiniteMeasure_of_le (μ : Measure α) [IsFiniteMeasure μ] (h : ν ≤ μ) : IsFiniteMeasure ν :=
  { measure_univ_lt_top := (h Set.univ).trans_lt (measure_lt_top _ _) }


@[instance]
theorem Measure.isFiniteMeasure_map {m : MeasurableSpace α} (μ : Measure α) [IsFiniteMeasure μ]
    (f : α → β) : IsFiniteMeasure (μ.map f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → β
    ⊢ MeasureTheory.IsFiniteMeasure (MeasureTheory.Measure.map f μ)
  -/
  by_cases hf : AEMeasurable f μ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → β
      hf : AEMeasurable f μ
      ⊢ MeasureTheory.IsFiniteMeasure (MeasureTheory.Measure.map f μ)
    -/
  · constructor
    /-
      case pos.measure_univ_lt_top
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → β
      hf : AEMeasurable f μ
      ⊢ LT.lt ((MeasureTheory.Measure.map f μ) Set.univ) Top.top
    -/
    rw [map_apply_of_aemeasurable hf MeasurableSet.univ]
    /-
      case pos.measure_univ_lt_top
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → β
      hf : AEMeasurable f μ
      ⊢ LT.lt (μ (Set.preimage f Set.univ)) Top.top
    -/
    exact measure_lt_top μ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → β
      hf : Not (AEMeasurable f μ)
      ⊢ MeasureTheory.IsFiniteMeasure (MeasureTheory.Measure.map f μ)
    -/
  · rw [map_of_not_aemeasurable hf]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → β
      hf : Not (AEMeasurable f μ)
      ⊢ MeasureTheory.IsFiniteMeasure 0
    -/
    exact MeasureTheory.isFiniteMeasureZero
    /-
      🎉 no goals
    -/


instance IsFiniteMeasure_comap (f : β → α) [IsFiniteMeasure μ] : IsFiniteMeasure (μ.comap f) where
  measure_univ_lt_top := by
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      f : β → α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      ⊢ LT.lt ((MeasureTheory.Measure.comap f μ) Set.univ) Top.top
    -/
    by_cases hf : Injective f ∧ ∀ s, MeasurableSet s → NullMeasurableSet (f '' s) μ
      /-
        case pos
        α : Type u_1
        β : Type u_2
        δ : Type u_3
        ι : Type u_4
        m0 : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        μ ν ν₁ ν₂ : MeasureTheory.Measure α
        s t : Set α
        f : β → α
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : And (Function.Injective f) (∀ (s : Set β), MeasurableSet s → MeasureTheor …
        ⊢ LT.lt ((MeasureTheory.Measure.comap f μ) Set.univ) Top.top
      -/
    · rw [Measure.comap_apply₀ _ _ hf.1 hf.2 MeasurableSet.univ.nullMeasurableSet]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        δ : Type u_3
        ι : Type u_4
        m0 : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        μ ν ν₁ ν₂ : MeasureTheory.Measure α
        s t : Set α
        f : β → α
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : And (Function.Injective f) (∀ (s : Set β), MeasurableSet s → MeasureTheor …
        ⊢ LT.lt (μ (Set.image f Set.univ)) Top.top
      -/
      exact measure_lt_top μ _
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        δ : Type u_3
        ι : Type u_4
        m0 : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        μ ν ν₁ ν₂ : MeasureTheory.Measure α
        s t : Set α
        f : β → α
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : Not (And (Function.Injective f) (∀ (s : Set β), MeasurableSet s → Measure …
        ⊢ LT.lt ((MeasureTheory.Measure.comap f μ) Set.univ) Top.top
      -/
    · rw [Measure.comap, dif_neg hf]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        δ : Type u_3
        ι : Type u_4
        m0 : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        μ ν ν₁ ν₂ : MeasureTheory.Measure α
        s t : Set α
        f : β → α
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : Not (And (Function.Injective f) (∀ (s : Set β), MeasurableSet s → Measure …
        ⊢ LT.lt (0 Set.univ) Top.top
      -/
      exact zero_lt_top
      /-
        🎉 no goals
      -/


@[simp]
theorem measureUnivNNReal_eq_zero [IsFiniteMeasure μ] : measureUnivNNReal μ = 0 ↔ μ = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Iff (Eq (MeasureTheory.measureUnivNNReal μ) 0) (Eq μ 0)
  -/
  rw [← MeasureTheory.Measure.measure_univ_eq_zero, ← coe_measureUnivNNReal]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Iff (Eq (MeasureTheory.measureUnivNNReal μ) 0) (Eq (↑(MeasureTheory.measureU …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem measureUnivNNReal_pos [IsFiniteMeasure μ] (hμ : μ ≠ 0) : 0 < measureUnivNNReal μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : Ne μ 0
    ⊢ LT.lt 0 (MeasureTheory.measureUnivNNReal μ)
  -/
  contrapose! hμ
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμ : LE.le (MeasureTheory.measureUnivNNReal μ) 0
    ⊢ Eq μ 0
  -/
  simpa [measureUnivNNReal_eq_zero, Nat.le_zero] using hμ
  /-
    🎉 no goals
  -/


/-- `le_of_add_le_add_left` is normally applicable to `OrderedCancelAddCommMonoid`,
but it holds for measures with the additional assumption that μ is finite. -/
theorem Measure.le_of_add_le_add_left [IsFiniteMeasure μ] (A2 : μ + ν₁ ≤ μ + ν₂) : ν₁ ≤ ν₂ :=
  fun S => ENNReal.le_of_add_le_add_left (MeasureTheory.measure_ne_top μ S) (A2 S)


theorem summable_measure_toReal [hμ : IsFiniteMeasure μ] {f : ℕ → Set α}
    (hf₁ : ∀ i : ℕ, MeasurableSet (f i)) (hf₂ : Pairwise (Disjoint on f)) :
    Summable fun x => (μ (f x)).toReal := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Set α
    hf₁ : ∀ (i : Nat), MeasurableSet (f i)
    hf₂ : Pairwise (Function.onFun Disjoint f)
    ⊢ Summable fun x => (μ (f x)).toReal
  -/
  apply ENNReal.summable_toReal
  /-
    case hsum
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Set α
    hf₁ : ∀ (i : Nat), MeasurableSet (f i)
    hf₂ : Pairwise (Function.onFun Disjoint f)
    ⊢ Ne (tsum fun x => μ (f x)) Top.top
  -/
  rw [← MeasureTheory.measure_iUnion hf₂ hf₁]
  /-
    case hsum
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hμ : MeasureTheory.IsFiniteMeasure μ
    f : Nat → Set α
    hf₁ : ∀ (i : Nat), MeasurableSet (f i)
    hf₂ : Pairwise (Function.onFun Disjoint f)
    ⊢ Ne (μ (Set.iUnion fun i => f i)) Top.top
  -/
  exact ne_of_lt (measure_lt_top _ _)
  /-
    🎉 no goals
  -/


theorem ae_eq_univ_iff_measure_eq [IsFiniteMeasure μ] (hs : NullMeasurableSet s μ) :
    s =ᵐ[μ] univ ↔ μ s = μ univ :=
  ⟨measure_congr, fun h ↦
    ae_eq_of_subset_of_measure_ge (subset_univ _) h.ge hs (measure_ne_top _ _)⟩


theorem ae_iff_measure_eq [IsFiniteMeasure μ] {p : α → Prop}
    (hp : NullMeasurableSet { a | p a } μ) : (∀ᵐ a ∂μ, p a) ↔ μ { a | p a } = μ univ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    p : α → Prop
    hp : MeasureTheory.NullMeasurableSet (setOf fun a => p a) μ
    ⊢ Iff (Filter.Eventually (fun a => p a) (MeasureTheory.ae μ)) (Eq (μ (setOf fu …
  -/
  rw [← ae_eq_univ_iff_measure_eq hp, eventuallyEq_univ, eventually_iff]
  /-
    🎉 no goals
  -/


theorem ae_mem_iff_measure_eq [IsFiniteMeasure μ] {s : Set α} (hs : NullMeasurableSet s μ) :
    (∀ᵐ a ∂μ, a ∈ s) ↔ μ s = μ univ :=
  ae_iff_measure_eq hs


lemma tendsto_measure_biUnion_Ici_zero_of_pairwise_disjoint
    {X : Type*} [MeasurableSpace X] {μ : Measure X} [IsFiniteMeasure μ]
    {Es : ℕ → Set X} (Es_mble : ∀ i, NullMeasurableSet (Es i) μ)
    (Es_disj : Pairwise fun n m ↦ Disjoint (Es n) (Es m)) :
    Tendsto (μ ∘ fun n ↦ ⋃ i ≥ n, Es i) atTop (𝓝 0) := by
  have decr : Antitone fun n ↦ ⋃ i ≥ n, Es i :=
    fun n m hnm ↦ biUnion_mono (fun _ hi ↦ le_trans hnm hi) (fun _ _ ↦ subset_rfl)
  have nothing : ⋂ n, ⋃ i ≥ n, Es i = ∅ := by
    apply subset_antisymm _ (empty_subset _)
    intro x hx
    simp only [mem_iInter, mem_iUnion, exists_prop] at hx
    obtain ⟨j, _, x_in_Es_j⟩ := hx 0
    obtain ⟨k, k_gt_j, x_in_Es_k⟩ := hx (j+1)
    have oops := (Es_disj (Nat.ne_of_lt k_gt_j)).ne_of_mem x_in_Es_j x_in_Es_k
    contradiction
  have key := tendsto_measure_iInter_atTop (μ := μ) (fun n ↦ by measurability)
    decr ⟨0, measure_ne_top _ _⟩
  /-
    X : Type u_5
    inst✝¹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    Es : Nat → Set X
    Es_mble : ∀ (i : Nat), MeasureTheory.NullMeasurableSet (Es i) μ
    Es_disj : Pairwise fun n m => Disjoint (Es n) (Es m)
    decr : Antitone fun n => Set.iUnion fun i => Set.iUnion fun h => Es i
    nothing : Eq (Set.iInter fun n => Set.iUnion fun i => Set.iUnion fun h => Es i …
    key : Filter.Tendsto (Function.comp ⇑μ fun n => Set.iUnion fun i => Set.iUnion …
    ⊢ Filter.Tendsto (Function.comp ⇑μ fun n => Set.iUnion fun i => Set.iUnion fun …
  -/
  simp only [nothing, measure_empty] at key
  /-
    X : Type u_5
    inst✝¹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    Es : Nat → Set X
    Es_mble : ∀ (i : Nat), MeasureTheory.NullMeasurableSet (Es i) μ
    Es_disj : Pairwise fun n m => Disjoint (Es n) (Es m)
    decr : Antitone fun n => Set.iUnion fun i => Set.iUnion fun h => Es i
    nothing : Eq (Set.iInter fun n => Set.iUnion fun i => Set.iUnion fun h => Es i …
    key : Filter.Tendsto (Function.comp ⇑μ fun n => Set.iUnion fun i => Set.iUnion …
    ⊢ Filter.Tendsto (Function.comp ⇑μ fun n => Set.iUnion fun i => Set.iUnion fun …
  -/
  convert key
  /-
    🎉 no goals
  -/


theorem abs_toReal_measure_sub_le_measure_symmDiff'
    (hs : NullMeasurableSet s μ) (ht : NullMeasurableSet t μ) (hs' : μ s ≠ ∞) (ht' : μ t ≠ ∞) :
    |(μ s).toReal - (μ t).toReal| ≤ (μ (s ∆ t)).toReal := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    hs' : Ne (μ s) Top.top
    ht' : Ne (μ t) Top.top
    ⊢ LE.le (abs (HSub.hSub (μ s).toReal (μ t).toReal)) (μ (symmDiff s t)).toReal
  -/
  have hst : μ (s \ t) ≠ ∞ := (measure_lt_top_of_subset diff_subset hs').ne
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    hs' : Ne (μ s) Top.top
    ht' : Ne (μ t) Top.top
    hst : Ne (μ (SDiff.sdiff s t)) Top.top
    ⊢ LE.le (abs (HSub.hSub (μ s).toReal (μ t).toReal)) (μ (symmDiff s t)).toReal
  -/
  have hts : μ (t \ s) ≠ ∞ := (measure_lt_top_of_subset diff_subset ht').ne
  suffices (μ s).toReal - (μ t).toReal = (μ (s \ t)).toReal - (μ (t \ s)).toReal by
    rw [this, measure_symmDiff_eq hs ht, ENNReal.toReal_add hst hts]
    convert abs_sub (μ (s \ t)).toReal (μ (t \ s)).toReal <;> simp
  rw [measure_diff' s ht ht', measure_diff' t hs hs',
    ENNReal.toReal_sub_of_le measure_le_measure_union_right (measure_union_ne_top hs' ht'),
    ENNReal.toReal_sub_of_le measure_le_measure_union_right (measure_union_ne_top ht' hs'),
    union_comm t s]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    ht : MeasureTheory.NullMeasurableSet t μ
    hs' : Ne (μ s) Top.top
    ht' : Ne (μ t) Top.top
    hst : Ne (μ (SDiff.sdiff s t)) Top.top
    hts : Ne (μ (SDiff.sdiff t s)) Top.top
    ⊢ Eq (HSub.hSub (μ s).toReal (μ t).toReal) (HSub.hSub (HSub.hSub (μ (Union.uni …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem abs_toReal_measure_sub_le_measure_symmDiff [IsFiniteMeasure μ]
    (hs : NullMeasurableSet s μ) (ht : NullMeasurableSet t μ) :
    |(μ s).toReal - (μ t).toReal| ≤ (μ (s ∆ t)).toReal :=
  abs_toReal_measure_sub_le_measure_symmDiff' hs ht (measure_ne_top μ s) (measure_ne_top μ t)


instance {s : Finset ι} {μ : ι → Measure α} [∀ i, IsFiniteMeasure (μ i)] :
                                                                   /-
                                                                     α : Type u_1
                                                                     β : Type u_2
                                                                     δ : Type u_3
                                                                     ι : Type u_4
                                                                     m0 : MeasurableSpace α
                                                                     inst✝¹ : MeasurableSpace β
                                                                     μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
                                                                     s✝ t : Set α
                                                                     s : Finset ι
                                                                     μ : ι → MeasureTheory.Measure α
                                                                     inst✝ : ∀ (i : ι), MeasureTheory.IsFiniteMeasure (μ i)
                                                                     ⊢ LT.lt ((s.sum fun i => μ i) Set.univ) Top.top
                                                                   -/
    IsFiniteMeasure (∑ i ∈ s, μ i) where measure_univ_lt_top := by simp [measure_lt_top]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance [Finite ι] {μ : ι → Measure α} [∀ i, IsFiniteMeasure (μ i)] :
    IsFiniteMeasure (.sum μ) where
  measure_univ_lt_top := by
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      inst✝¹ : Finite ι
      μ : ι → MeasureTheory.Measure α
      inst✝ : ∀ (i : ι), MeasureTheory.IsFiniteMeasure (μ i)
      ⊢ LT.lt ((MeasureTheory.Measure.sum μ) Set.univ) Top.top
    -/
    cases nonempty_fintype ι
    /-
      case intro
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      inst✝¹ : Finite ι
      μ : ι → MeasureTheory.Measure α
      inst✝ : ∀ (i : ι), MeasureTheory.IsFiniteMeasure (μ i)
      val✝ : Fintype ι
      ⊢ LT.lt ((MeasureTheory.Measure.sum μ) Set.univ) Top.top
    -/
    simp [measure_lt_top]
    /-
      🎉 no goals
    -/


/-- A measure `μ` is zero or a probability measure if `μ univ = 0` or `μ univ = 1`. This class
of measures appears naturally when conditioning on events, and many results which are true for
probability measures hold more generally over this class. -/
class IsZeroOrProbabilityMeasure (μ : Measure α) : Prop where
  measure_univ : μ univ = 0 ∨ μ univ = 1


lemma isZeroOrProbabilityMeasure_iff : IsZeroOrProbabilityMeasure μ ↔ μ univ = 0 ∨ μ univ = 1 :=
  ⟨fun _ ↦ IsZeroOrProbabilityMeasure.measure_univ, IsZeroOrProbabilityMeasure.mk⟩


lemma prob_le_one {μ : Measure α} [IsZeroOrProbabilityMeasure μ] {s : Set α} : μ s ≤ 1 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    s : Set α
    ⊢ LE.le (μ s) 1
  -/
  apply (measure_mono (subset_univ _)).trans
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    s : Set α
    ⊢ LE.le (μ Set.univ) 1
  -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  rcases IsZeroOrProbabilityMeasure.measure_univ (μ := μ) with h | h <;> simp [h]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem one_le_prob_iff {μ : Measure α} [IsZeroOrProbabilityMeasure μ] : 1 ≤ μ s ↔ μ s = 1 :=
  ⟨fun h => le_antisymm prob_le_one h, fun h => h ▸ le_refl _⟩


instance (priority := 100) IsZeroOrProbabilityMeasure.toIsFiniteMeasure (μ : Measure α)
    [IsZeroOrProbabilityMeasure μ] : IsFiniteMeasure μ :=
  ⟨prob_le_one.trans_lt one_lt_top⟩


instance : IsZeroOrProbabilityMeasure (0 : Measure α) :=
  ⟨Or.inl rfl⟩


/-- A measure `μ` is called a probability measure if `μ univ = 1`. -/
class IsProbabilityMeasure (μ : Measure α) : Prop where
  measure_univ : μ univ = 1


lemma isProbabilityMeasure_iff : IsProbabilityMeasure μ ↔ μ univ = 1 :=
  ⟨fun _ ↦ measure_univ, IsProbabilityMeasure.mk⟩


instance (priority := 100) (μ : Measure α) [IsProbabilityMeasure μ] :
    IsZeroOrProbabilityMeasure μ :=
  ⟨Or.inr measure_univ⟩


theorem IsProbabilityMeasure.ne_zero (μ : Measure α) [IsProbabilityMeasure μ] : μ ≠ 0 :=
                                  /-
                                    α : Type u_1
                                    m0 : MeasurableSpace α
                                    μ : MeasureTheory.Measure α
                                    inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                    ⊢ Not (Eq (μ Set.univ) 0)
                                  -/
  mt measure_univ_eq_zero.2 <| by simp [measure_univ]
                                  /-
                                    🎉 no goals
                                  -/


instance (priority := 100) IsProbabilityMeasure.neZero (μ : Measure α) [IsProbabilityMeasure μ] :
    NeZero μ := ⟨IsProbabilityMeasure.ne_zero μ⟩

-- Porting note: no longer an `instance` because `inferInstance` can find it now

theorem IsProbabilityMeasure.ae_neBot [IsProbabilityMeasure μ] : NeBot (ae μ) := inferInstance


theorem prob_add_prob_compl [IsProbabilityMeasure μ] (h : MeasurableSet s) : μ s + μ sᶜ = 1 :=
  (measure_add_measure_compl h).trans measure_univ

-- Porting note: made an `instance`, using `NeZero`

instance isProbabilityMeasureSMul [IsFiniteMeasure μ] [NeZero μ] :
    IsProbabilityMeasure ((μ univ)⁻¹ • μ) :=
  ⟨ENNReal.inv_mul_cancel (NeZero.ne (μ univ)) (measure_ne_top _ _)⟩


theorem isProbabilityMeasure_map {f : α → β} (hf : AEMeasurable f μ) :
    IsProbabilityMeasure (map f μ) :=
      /-
        α : Type u_1
        β : Type u_2
        m0 : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsProbabilityMeasure μ
        f : α → β
        hf : AEMeasurable f μ
        ⊢ Eq ((MeasureTheory.Measure.map f μ) Set.univ) 1
      -/
  ⟨by simp [map_apply_of_aemeasurable, hf]⟩
      /-
        🎉 no goals
      -/


instance IsProbabilityMeasure_comap_equiv (f : β ≃ᵐ α) : IsProbabilityMeasure (μ.comap f) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    p : α → Prop
    f✝ : β → α
    f : MeasurableEquiv β α
    ⊢ MeasureTheory.IsProbabilityMeasure (MeasureTheory.Measure.comap (⇑f) μ)
  -/
  rw [← MeasurableEquiv.map_symm]; exact isProbabilityMeasure_map f.symm.measurable.aemeasurable
                                   /-
                                     🎉 no goals
                                   -/


/-- Note that this is not quite as useful as it looks because the measure takes values in `ℝ≥0∞`.
Thus the subtraction appearing is the truncated subtraction of `ℝ≥0∞`, rather than the
better-behaved subtraction of `ℝ`. -/
lemma prob_compl_eq_one_sub₀ (h : NullMeasurableSet s μ) : μ sᶜ = 1 - μ s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    h : MeasureTheory.NullMeasurableSet s μ
    ⊢ Eq (μ (HasCompl.compl s)) (HSub.hSub 1 (μ s))
  -/
  rw [measure_compl₀ h (measure_ne_top _ _), measure_univ]
  /-
    🎉 no goals
  -/


/-- Note that this is not quite as useful as it looks because the measure takes values in `ℝ≥0∞`.
Thus the subtraction appearing is the truncated subtraction of `ℝ≥0∞`, rather than the
better-behaved subtraction of `ℝ`. -/
theorem prob_compl_eq_one_sub (hs : MeasurableSet s) : μ sᶜ = 1 - μ s :=
  prob_compl_eq_one_sub₀ hs.nullMeasurableSet


@[simp] lemma prob_compl_eq_zero_iff₀ (hs : NullMeasurableSet s μ) : μ sᶜ = 0 ↔ μ s = 1 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ Iff (Eq (μ (HasCompl.compl s)) 0) (Eq (μ s) 1)
  -/
  rw [prob_compl_eq_one_sub₀ hs, tsub_eq_zero_iff_le, one_le_prob_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma prob_compl_eq_zero_iff (hs : MeasurableSet s) : μ sᶜ = 0 ↔ μ s = 1 :=
  prob_compl_eq_zero_iff₀ hs.nullMeasurableSet


@[simp] lemma prob_compl_eq_one_iff₀ (hs : NullMeasurableSet s μ) : μ sᶜ = 1 ↔ μ s = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ Iff (Eq (μ (HasCompl.compl s)) 1) (Eq (μ s) 0)
  -/
  rw [← prob_compl_eq_zero_iff₀ hs.compl, compl_compl]
  /-
    🎉 no goals
  -/


@[simp] lemma prob_compl_eq_one_iff (hs : MeasurableSet s) : μ sᶜ = 1 ↔ μ s = 0 :=
  prob_compl_eq_one_iff₀ hs.nullMeasurableSet


lemma mem_ae_iff_prob_eq_one₀ (hs : NullMeasurableSet s μ) : s ∈ ae μ ↔ μ s = 1 :=
  mem_ae_iff.trans <| prob_compl_eq_zero_iff₀ hs


lemma mem_ae_iff_prob_eq_one (hs : MeasurableSet s) : s ∈ ae μ ↔ μ s = 1 :=
  mem_ae_iff.trans <| prob_compl_eq_zero_iff hs


lemma ae_iff_prob_eq_one (hp : Measurable p) : (∀ᵐ a ∂μ, p a) ↔ μ {a | p a} = 1 :=
  mem_ae_iff_prob_eq_one hp.setOf


lemma isProbabilityMeasure_comap (hf : Injective f) (hf' : ∀ᵐ a ∂μ, a ∈ range f)
    (hf'' : ∀ s, MeasurableSet s → MeasurableSet (f '' s)) :
    IsProbabilityMeasure (μ.comap f) where
  measure_univ := by
    rw [comap_apply _ hf hf'' _ MeasurableSet.univ,
      ← mem_ae_iff_prob_eq_one (hf'' _ MeasurableSet.univ)]
    /-
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      f : β → α
      hf : Function.Injective f
      hf' : Filter.Eventually (fun a => Membership.mem (Set.range f) a) (MeasureTheo …
      hf'' : ∀ (s : Set β), MeasurableSet s → MeasurableSet (Set.image f s)
      ⊢ Membership.mem (MeasureTheory.ae μ) (Set.image f Set.univ)
    -/
    simpa
    /-
      🎉 no goals
    -/


protected lemma _root_.MeasurableEmbedding.isProbabilityMeasure_comap (hf : MeasurableEmbedding f)
    (hf' : ∀ᵐ a ∂μ, a ∈ range f) : IsProbabilityMeasure (μ.comap f) :=
  isProbabilityMeasure_comap hf.injective hf' hf.measurableSet_image'


instance isProbabilityMeasure_map_up :
    IsProbabilityMeasure (μ.map ULift.up) := isProbabilityMeasure_map measurable_up.aemeasurable


instance isProbabilityMeasure_comap_down : IsProbabilityMeasure (μ.comap ULift.down) :=
  MeasurableEquiv.ulift.measurableEmbedding.isProbabilityMeasure_comap <| ae_of_all _ <| by
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      inst✝ : MeasureTheory.IsProbabilityMeasure μ
      p : α → Prop
      f : β → α
      ⊢ ∀ (a : α), Membership.mem (Set.range ⇑MeasurableEquiv.ulift) a
    -/
    simp [Function.Surjective.range_eq <| EquivLike.surjective _]
    /-
      🎉 no goals
    -/


instance isZeroOrProbabilityMeasureSMul :
    IsZeroOrProbabilityMeasure ((μ univ)⁻¹ • μ) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (HSMul.hSMul (Inv.inv (μ Set.univ)) …
  -/
  rcases eq_zero_or_neZero μ with rfl | h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (HSMul.hSMul (Inv.inv (0 Set.univ)) …
    -/
  · simp; infer_instance
          /-
            🎉 no goals
          -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    h : NeZero μ
    ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (HSMul.hSMul (Inv.inv (μ Set.univ)) …
  -/
  rcases eq_top_or_lt_top (μ univ) with h | h
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      h✝ : NeZero μ
      h : Eq (μ Set.univ) Top.top
      ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (HSMul.hSMul (Inv.inv (μ Set.univ)) …
    -/
  · simp [h]; infer_instance
              /-
                🎉 no goals
              -/
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    h✝ : NeZero μ
    h : LT.lt (μ Set.univ) Top.top
    ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (HSMul.hSMul (Inv.inv (μ Set.univ)) …
  -/
  have : IsFiniteMeasure μ := ⟨h⟩
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    h✝ : NeZero μ
    h : LT.lt (μ Set.univ) Top.top
    this : MeasureTheory.IsFiniteMeasure μ
    ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (HSMul.hSMul (Inv.inv (μ Set.univ)) …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


variable (μ) in
lemma eq_zero_or_isProbabilityMeasure : μ = 0 ∨ IsProbabilityMeasure μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    ⊢ Or (Eq μ 0) (MeasureTheory.IsProbabilityMeasure μ)
  -/
  rcases IsZeroOrProbabilityMeasure.measure_univ (μ := μ) with h | h
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      h : Eq (μ Set.univ) 0
      ⊢ Or (Eq μ 0) (MeasureTheory.IsProbabilityMeasure μ)
    -/
  · apply Or.inl (measure_univ_eq_zero.mp h)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      h : Eq (μ Set.univ) 1
      ⊢ Or (Eq μ 0) (MeasureTheory.IsProbabilityMeasure μ)
    -/
  · exact Or.inr ⟨h⟩
    /-
      🎉 no goals
    -/


instance {f : α → β} : IsZeroOrProbabilityMeasure (map f μ) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    p : α → Prop
    f✝ : β → α
    f : α → β
    ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (MeasureTheory.Measure.map f μ)
  -/
  by_cases hf : AEMeasurable f μ
    /-
      case pos
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      p : α → Prop
      f✝ : β → α
      f : α → β
      hf : AEMeasurable f μ
      ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (MeasureTheory.Measure.map f μ)
    -/
  · simpa [isZeroOrProbabilityMeasure_iff, hf] using IsZeroOrProbabilityMeasure.measure_univ
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      p : α → Prop
      f✝ : β → α
      f : α → β
      hf : Not (AEMeasurable f μ)
      ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (MeasureTheory.Measure.map f μ)
    -/
  · simp [isZeroOrProbabilityMeasure_iff, hf]
    /-
      🎉 no goals
    -/


lemma prob_compl_lt_one_sub_of_lt_prob {p : ℝ≥0∞} (hμs : p < μ s) (s_mble : MeasurableSet s) :
    μ sᶜ < 1 - p := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    p : ENNReal
    hμs : LT.lt p (μ s)
    s_mble : MeasurableSet s
    ⊢ LT.lt (μ (HasCompl.compl s)) (HSub.hSub 1 p)
  -/
  rcases eq_zero_or_isProbabilityMeasure μ with rfl | h
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      s : Set α
      p : ENNReal
      s_mble : MeasurableSet s
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure 0
      hμs : LT.lt p (0 s)
      ⊢ LT.lt (0 (HasCompl.compl s)) (HSub.hSub 1 p)
    -/
  · simp at hμs
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      p : ENNReal
      hμs : LT.lt p (μ s)
      s_mble : MeasurableSet s
      h : MeasureTheory.IsProbabilityMeasure μ
      ⊢ LT.lt (μ (HasCompl.compl s)) (HSub.hSub 1 p)
    -/
  · rw [prob_compl_eq_one_sub s_mble]
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      p : ENNReal
      hμs : LT.lt p (μ s)
      s_mble : MeasurableSet s
      h : MeasureTheory.IsProbabilityMeasure μ
      ⊢ LT.lt (HSub.hSub 1 (μ s)) (HSub.hSub 1 p)
    -/
    apply ENNReal.sub_lt_of_sub_lt prob_le_one (Or.inl one_ne_top)
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      p : ENNReal
      hμs : LT.lt p (μ s)
      s_mble : MeasurableSet s
      h : MeasureTheory.IsProbabilityMeasure μ
      ⊢ LT.lt (HSub.hSub 1 (HSub.hSub 1 p)) (μ s)
    -/
    convert hμs
    /-
      case h.e'_3
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      p : ENNReal
      hμs : LT.lt p (μ s)
      s_mble : MeasurableSet s
      h : MeasureTheory.IsProbabilityMeasure μ
      ⊢ Eq (HSub.hSub 1 (HSub.hSub 1 p)) p
    -/
    exact ENNReal.sub_sub_cancel one_ne_top (lt_of_lt_of_le hμs prob_le_one).le
    /-
      🎉 no goals
    -/


lemma prob_compl_le_one_sub_of_le_prob {p : ℝ≥0∞} (hμs : p ≤ μ s) (s_mble : MeasurableSet s) :
    μ sᶜ ≤ 1 - p := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
    p : ENNReal
    hμs : LE.le p (μ s)
    s_mble : MeasurableSet s
    ⊢ LE.le (μ (HasCompl.compl s)) (HSub.hSub 1 p)
  -/
  rcases eq_zero_or_isProbabilityMeasure μ with rfl | h
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      s : Set α
      p : ENNReal
      s_mble : MeasurableSet s
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure 0
      hμs : LE.le p (0 s)
      ⊢ LE.le (0 (HasCompl.compl s)) (HSub.hSub 1 p)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝ : MeasureTheory.IsZeroOrProbabilityMeasure μ
      p : ENNReal
      hμs : LE.le p (μ s)
      s_mble : MeasurableSet s
      h : MeasureTheory.IsProbabilityMeasure μ
      ⊢ LE.le (μ (HasCompl.compl s)) (HSub.hSub 1 p)
    -/
  · simpa [prob_compl_eq_one_sub s_mble] using tsub_le_tsub_left hμs 1
    /-
      🎉 no goals
    -/


/-- Measure `μ` *has no atoms* if the measure of each singleton is zero.

NB: Wikipedia assumes that for any measurable set `s` with positive `μ`-measure,
there exists a measurable `t ⊆ s` such that `0 < μ t < μ s`. While this implies `μ {x} = 0`,
the converse is not true. -/
class NoAtoms {m0 : MeasurableSpace α} (μ : Measure α) : Prop where
  measure_singleton : ∀ x, μ {x} = 0


theorem _root_.Set.Subsingleton.measure_zero (hs : s.Subsingleton) (μ : Measure α) [NoAtoms μ] :
    μ s = 0 :=
  hs.induction_on (p := fun s => μ s = 0) measure_empty measure_singleton


theorem Measure.restrict_singleton' {a : α} : μ.restrict {a} = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.NoAtoms μ
    a : α
    ⊢ Eq (μ.restrict (Singleton.singleton a)) 0
  -/
  simp only [measure_singleton, Measure.restrict_eq_zero]
  /-
    🎉 no goals
  -/


instance Measure.restrict.instNoAtoms (s : Set α) : NoAtoms (μ.restrict s) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t : Set α
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set α
    ⊢ MeasureTheory.NoAtoms (μ.restrict s)
  -/
  refine ⟨fun x => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t : Set α
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set α
    x : α
    ⊢ Eq ((μ.restrict s) (Singleton.singleton x)) 0
  -/
  obtain ⟨t, hxt, ht1, ht2⟩ := exists_measurable_superset_of_null (measure_singleton x : μ {x} = 0)
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t✝ : Set α
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set α
    x : α
    t : Set α
    hxt : HasSubset.Subset (Singleton.singleton x) t
    ht1 : MeasurableSet t
    ht2 : Eq (μ t) 0
    ⊢ Eq ((μ.restrict s) (Singleton.singleton x)) 0
  -/
  apply measure_mono_null hxt
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t✝ : Set α
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set α
    x : α
    t : Set α
    hxt : HasSubset.Subset (Singleton.singleton x) t
    ht1 : MeasurableSet t
    ht2 : Eq (μ t) 0
    ⊢ Eq ((μ.restrict s) t) 0
  -/
  rw [Measure.restrict_apply ht1]
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t✝ : Set α
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set α
    x : α
    t : Set α
    hxt : HasSubset.Subset (Singleton.singleton x) t
    ht1 : MeasurableSet t
    ht2 : Eq (μ t) 0
    ⊢ Eq (μ (Inter.inter t s)) 0
  -/
  apply measure_mono_null inter_subset_left ht2
  /-
    🎉 no goals
  -/


theorem _root_.Set.Countable.measure_zero (h : s.Countable) (μ : Measure α) [NoAtoms μ] :
    μ s = 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    s : Set α
    h : s.Countable
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.NoAtoms μ
    ⊢ Eq (μ s) 0
  -/
  rw [← biUnion_of_singleton s, measure_biUnion_null_iff h]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    s : Set α
    h : s.Countable
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.NoAtoms μ
    ⊢ ∀ (i : α), Membership.mem s i → Eq (μ (Singleton.singleton i)) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem _root_.Set.Countable.ae_not_mem (h : s.Countable) (μ : Measure α) [NoAtoms μ] :
    ∀ᵐ x ∂μ, x ∉ s := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    s : Set α
    h : s.Countable
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.NoAtoms μ
    ⊢ Filter.Eventually (fun x => Not (Membership.mem s x)) (MeasureTheory.ae μ)
  -/
  simpa only [ae_iff, Classical.not_not] using h.measure_zero μ
  /-
    🎉 no goals
  -/


lemma _root_.Set.Countable.measure_restrict_compl (h : s.Countable) (μ : Measure α) [NoAtoms μ] :
    μ.restrict sᶜ = μ :=
  restrict_eq_self_of_ae_mem <| h.ae_not_mem μ


@[simp]
lemma restrict_compl_singleton (a : α) : μ.restrict ({a}ᶜ) = μ :=
  (countable_singleton _).measure_restrict_compl μ


theorem _root_.Set.Finite.measure_zero (h : s.Finite) (μ : Measure α) [NoAtoms μ] : μ s = 0 :=
  h.countable.measure_zero μ


theorem _root_.Finset.measure_zero (s : Finset α) (μ : Measure α) [NoAtoms μ] : μ s = 0 :=
  s.finite_toSet.measure_zero μ


theorem insert_ae_eq_self (a : α) (s : Set α) : (insert a s : Set α) =ᵐ[μ] s :=
  union_ae_eq_right.2 <| measure_mono_null diff_subset (measure_singleton _)


theorem Iio_ae_eq_Iic : Iio a =ᵐ[μ] Iic a :=
  Iio_ae_eq_Iic' (measure_singleton a)


theorem Ioi_ae_eq_Ici : Ioi a =ᵐ[μ] Ici a :=
  Ioi_ae_eq_Ici' (measure_singleton a)


theorem Ioo_ae_eq_Ioc : Ioo a b =ᵐ[μ] Ioc a b :=
  Ioo_ae_eq_Ioc' (measure_singleton b)


theorem Ioc_ae_eq_Icc : Ioc a b =ᵐ[μ] Icc a b :=
  Ioc_ae_eq_Icc' (measure_singleton a)


theorem Ioo_ae_eq_Ico : Ioo a b =ᵐ[μ] Ico a b :=
  Ioo_ae_eq_Ico' (measure_singleton a)


theorem Ioo_ae_eq_Icc : Ioo a b =ᵐ[μ] Icc a b :=
  Ioo_ae_eq_Icc' (measure_singleton a) (measure_singleton b)


theorem Ico_ae_eq_Icc : Ico a b =ᵐ[μ] Icc a b :=
  Ico_ae_eq_Icc' (measure_singleton b)


theorem Ico_ae_eq_Ioc : Ico a b =ᵐ[μ] Ioc a b :=
  Ico_ae_eq_Ioc' (measure_singleton a) (measure_singleton b)


theorem restrict_Iio_eq_restrict_Iic : μ.restrict (Iio a) = μ.restrict (Iic a) :=
  restrict_congr_set Iio_ae_eq_Iic


theorem restrict_Ioi_eq_restrict_Ici : μ.restrict (Ioi a) = μ.restrict (Ici a) :=
  restrict_congr_set Ioi_ae_eq_Ici


theorem restrict_Ioo_eq_restrict_Ioc : μ.restrict (Ioo a b) = μ.restrict (Ioc a b) :=
  restrict_congr_set Ioo_ae_eq_Ioc


theorem restrict_Ioc_eq_restrict_Icc : μ.restrict (Ioc a b) = μ.restrict (Icc a b) :=
  restrict_congr_set Ioc_ae_eq_Icc


theorem restrict_Ioo_eq_restrict_Ico : μ.restrict (Ioo a b) = μ.restrict (Ico a b) :=
  restrict_congr_set Ioo_ae_eq_Ico


theorem restrict_Ioo_eq_restrict_Icc : μ.restrict (Ioo a b) = μ.restrict (Icc a b) :=
  restrict_congr_set Ioo_ae_eq_Icc


theorem restrict_Ico_eq_restrict_Icc : μ.restrict (Ico a b) = μ.restrict (Icc a b) :=
  restrict_congr_set Ico_ae_eq_Icc


theorem restrict_Ico_eq_restrict_Ioc : μ.restrict (Ico a b) = μ.restrict (Ioc a b) :=
  restrict_congr_set Ico_ae_eq_Ioc


theorem uIoc_ae_eq_interval [LinearOrder α] {a b : α} : Ι a b =ᵐ[μ] [[a, b]] :=
  Ioc_ae_eq_Icc


theorem ite_ae_eq_of_measure_zero {γ} (f : α → γ) (g : α → γ) (s : Set α) [DecidablePred (· ∈ s)]
    (hs_zero : μ s = 0) :
    (fun x => ite (x ∈ s) (f x) (g x)) =ᵐ[μ] g := by
  have h_ss : sᶜ ⊆ { a : α | ite (a ∈ s) (f a) (g a) = g a } := fun x hx => by
    simp [(Set.mem_compl_iff _ _).mp hx]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    γ : Type u_5
    f g : α → γ
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs_zero : Eq (μ s) 0
    h_ss : HasSubset.Subset (HasCompl.compl s) (setOf fun a => Eq (ite (Membership …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ite (Membership.mem s x) (f x) ( …
  -/
  refine measure_mono_null ?_ hs_zero
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    γ : Type u_5
    f g : α → γ
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs_zero : Eq (μ s) 0
    h_ss : HasSubset.Subset (HasCompl.compl s) (setOf fun a => Eq (ite (Membership …
    ⊢ HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq ((fun x => ite …
  -/
  conv_rhs => rw [← compl_compl s]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    γ : Type u_5
    f g : α → γ
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs_zero : Eq (μ s) 0
    h_ss : HasSubset.Subset (HasCompl.compl s) (setOf fun a => Eq (ite (Membership …
    ⊢ HasSubset.Subset (HasCompl.compl (setOf fun x => (fun x => Eq ((fun x => ite …
  -/
  rwa [Set.compl_subset_compl]
  /-
    🎉 no goals
  -/


theorem ite_ae_eq_of_measure_compl_zero {γ} (f : α → γ) (g : α → γ)
    (s : Set α) [DecidablePred (· ∈ s)] (hs_zero : μ sᶜ = 0) :
    (fun x => ite (x ∈ s) (f x) (g x)) =ᵐ[μ] f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    γ : Type u_5
    f g : α → γ
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs_zero : Eq (μ (HasCompl.compl s)) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ite (Membership.mem s x) (f x) ( …
  -/
  rw [← mem_ae_iff] at hs_zero
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    γ : Type u_5
    f g : α → γ
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs_zero : Membership.mem (MeasureTheory.ae μ) s
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ite (Membership.mem s x) (f x) ( …
  -/
  filter_upwards [hs_zero]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    γ : Type u_5
    f g : α → γ
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs_zero : Membership.mem (MeasureTheory.ae μ) s
    ⊢ ∀ (a : α), Membership.mem s a → Eq (ite (Membership.mem s a) (f a) (g a)) (f …
  -/
  intros
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    γ : Type u_5
    f g : α → γ
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs_zero : Membership.mem (MeasureTheory.ae μ) s
    a✝¹ : α
    a✝ : Membership.mem s a✝¹
    ⊢ Eq (ite (Membership.mem s a✝¹) (f a✝¹) (g a✝¹)) (f a✝¹)
  -/
  split_ifs
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    γ : Type u_5
    f g : α → γ
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs_zero : Membership.mem (MeasureTheory.ae μ) s
    a✝¹ : α
    a✝ : Membership.mem s a✝¹
    ⊢ Eq (f a✝¹) (f a✝¹)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A measure is called finite at filter `f` if it is finite at some set `s ∈ f`.
Equivalently, it is eventually finite at `s` in `f.small_sets`. -/
def FiniteAtFilter {_m0 : MeasurableSpace α} (μ : Measure α) (f : Filter α) : Prop :=
  ∃ s ∈ f, μ s < ∞


theorem finiteAtFilter_of_finite {_m0 : MeasurableSpace α} (μ : Measure α) [IsFiniteMeasure μ]
    (f : Filter α) : μ.FiniteAtFilter f :=
  ⟨univ, univ_mem, measure_lt_top μ univ⟩


theorem FiniteAtFilter.exists_mem_basis {f : Filter α} (hμ : FiniteAtFilter μ f) {p : ι → Prop}
    {s : ι → Set α} (hf : f.HasBasis p s) : ∃ i, p i ∧ μ (s i) < ∞ :=
  (hf.exists_iff fun {_s _t} hst ht => (measure_mono hst).trans_lt ht).1 hμ


theorem finiteAtBot {m0 : MeasurableSpace α} (μ : Measure α) : μ.FiniteAtFilter ⊥ :=
                  /-
                    α : Type u_1
                    m0 : MeasurableSpace α
                    μ : MeasureTheory.Measure α
                    ⊢ LT.lt (μ EmptyCollection.emptyCollection) Top.top
                  -/
  ⟨∅, mem_bot, by simp only [measure_empty, zero_lt_top]⟩
                  /-
                    🎉 no goals
                  -/


/-- `μ` has finite spanning sets in `C` if there is a countable sequence of sets in `C` that have
  finite measures. This structure is a type, which is useful if we want to record extra properties
  about the sets, such as that they are monotone.
  `SigmaFinite` is defined in terms of this: `μ` is σ-finite if there exists a sequence of
  finite spanning sets in the collection of all measurable sets. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure FiniteSpanningSetsIn {m0 : MeasurableSpace α} (μ : Measure α) (C : Set (Set α)) where
  protected set : ℕ → Set α
  protected set_mem : ∀ i, set i ∈ C
  protected finite : ∀ i, μ (set i) < ∞
  protected spanning : ⋃ i, set i = univ


/-- A measure is called s-finite if it is a countable sum of finite measures. -/
class SFinite (μ : Measure α) : Prop where
  out' : ∃ m : ℕ → Measure α, (∀ n, IsFiniteMeasure (m n)) ∧ μ = Measure.sum m


/-- A sequence of finite measures such that `μ = sum (sfiniteSeq μ)` (see `sum_sfiniteSeq`). -/
noncomputable def sfiniteSeq (μ : Measure α) [h : SFinite μ] : ℕ → Measure α := h.1.choose


@[deprecated (since := "2024-10-11")] alias sFiniteSeq := sfiniteSeq


instance isFiniteMeasure_sfiniteSeq [h : SFinite μ] (n : ℕ) : IsFiniteMeasure (sfiniteSeq μ n) :=
  h.1.choose_spec.1 n


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-10-11")]
instance isFiniteMeasure_sFiniteSeq [SFinite μ] (n : ℕ) : IsFiniteMeasure (sFiniteSeq μ n) :=
  isFiniteMeasure_sfiniteSeq n


lemma sum_sfiniteSeq (μ : Measure α) [h : SFinite μ] : sum (sfiniteSeq μ) = μ :=
  h.1.choose_spec.2.symm


@[deprecated (since := "2024-10-11")] alias sum_sFiniteSeq := sum_sfiniteSeq


lemma sfiniteSeq_le (μ : Measure α) [SFinite μ] (n : ℕ) : sfiniteSeq μ n ≤ μ :=
  (le_sum _ n).trans (sum_sfiniteSeq μ).le


@[deprecated (since := "2024-10-11")] alias sFiniteSeq_le := sfiniteSeq_le


                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      δ : Type u_3
                                                                      ι : Type u_4
                                                                      m0 : MeasurableSpace α
                                                                      inst✝ : MeasurableSpace β
                                                                      μ ν ν₁ ν₂ : MeasureTheory.Measure α
                                                                      s t : Set α
                                                                      ⊢ Eq 0 (MeasureTheory.Measure.sum fun x => 0)
                                                                    -/
instance : SFinite (0 : Measure α) := ⟨fun _ ↦ 0, inferInstance, by rw [Measure.sum_zero]⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
lemma sfiniteSeq_zero (n : ℕ) : sfiniteSeq (0 : Measure α) n = 0 :=
  bot_unique <| sfiniteSeq_le _ _


@[deprecated (since := "2024-10-11")] alias sFiniteSeq_zero := sfiniteSeq_zero


/-- A countable sum of finite measures is s-finite.
This lemma is superseded by the instance below. -/
lemma sfinite_sum_of_countable [Countable ι]
    (m : ι → Measure α) [∀ n, IsFiniteMeasure (m n)] : SFinite (Measure.sum m) := by
  classical
  obtain ⟨f, hf⟩ : ∃ f : ι → ℕ, Function.Injective f := Countable.exists_injective_nat ι
  refine ⟨_, fun n ↦ ?_, (sum_extend_zero hf m).symm⟩
  rcases em (n ∈ range f) with ⟨i, rfl⟩ | hn
  · rw [hf.extend_apply]
    infer_instance
  · rw [Function.extend_apply' _ _ _ hn, Pi.zero_apply]
    infer_instance


instance [Countable ι] (m : ι → Measure α) [∀ n, SFinite (m n)] : SFinite (Measure.sum m) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : Countable ι
    m : ι → MeasureTheory.Measure α
    inst✝ : ∀ (n : ι), MeasureTheory.SFinite (m n)
    ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.sum m)
  -/
  change SFinite (Measure.sum (fun i ↦ m i))
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : Countable ι
    m : ι → MeasureTheory.Measure α
    inst✝ : ∀ (n : ι), MeasureTheory.SFinite (m n)
    ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.sum fun i => m i)
  -/
  simp_rw [← sum_sfiniteSeq (m _), Measure.sum_sum]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : Countable ι
    m : ι → MeasureTheory.Measure α
    inst✝ : ∀ (n : ι), MeasureTheory.SFinite (m n)
    ⊢ MeasureTheory.SFinite (MeasureTheory.Measure.sum fun p => MeasureTheory.sfin …
  -/
  apply sfinite_sum_of_countable
  /-
    🎉 no goals
  -/


instance [SFinite μ] [SFinite ν] : SFinite (μ + ν) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    ⊢ MeasureTheory.SFinite (HAdd.hAdd μ ν)
  -/
  have : ∀ b : Bool, SFinite (cond b μ ν) := by simp [*]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : MeasureTheory.SFinite ν
    this : ∀ (b : Bool), MeasureTheory.SFinite (cond b μ ν)
    ⊢ MeasureTheory.SFinite (HAdd.hAdd μ ν)
  -/
  simpa using inferInstanceAs (SFinite (.sum (cond · μ ν)))
  /-
    🎉 no goals
  -/


instance [SFinite μ] (s : Set α) : SFinite (μ.restrict s) :=
  ⟨fun n ↦ (sfiniteSeq μ n).restrict s, fun n ↦ inferInstance,
       /-
         α : Type u_1
         β : Type u_2
         δ : Type u_3
         ι : Type u_4
         m0 : MeasurableSpace α
         inst✝¹ : MeasurableSpace β
         μ ν ν₁ ν₂ : MeasureTheory.Measure α
         s✝ t : Set α
         inst✝ : MeasureTheory.SFinite μ
         s : Set α
         ⊢ Eq (μ.restrict s) (MeasureTheory.Measure.sum fun n => (MeasureTheory.sfinite …
       -/
    by rw [← restrict_sum_of_countable, sum_sfiniteSeq]⟩
       /-
         🎉 no goals
       -/


variable (μ) in
/-- For an s-finite measure `μ`, there exists a finite measure `ν`
such that each of `μ` and `ν` is absolutely continuous with respect to the other.
-/
theorem exists_isFiniteMeasure_absolutelyContinuous [SFinite μ] :
    ∃ ν : Measure α, IsFiniteMeasure ν ∧ μ ≪ ν ∧ ν ≪ μ := by
  rcases ENNReal.exists_pos_tsum_mul_lt_of_countable top_ne_zero (sfiniteSeq μ · univ)
    fun _ ↦ measure_ne_top _ _ with ⟨c, hc₀, hc⟩
  have {s : Set α} : sum (fun n ↦ c n • sfiniteSeq μ n) s = 0 ↔ μ s = 0 := by
    conv_rhs => rw [← sum_sfiniteSeq μ, sum_apply_of_countable]
    simp [(hc₀ _).ne']
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    c : Nat → NNReal
    hc₀ : ∀ (i : Nat), LT.lt 0 (c i)
    hc : LT.lt (tsum fun i => HMul.hMul ((MeasureTheory.sfiniteSeq μ i) Set.univ)  …
    this : ∀ {s : Set α}, Iff (Eq ((MeasureTheory.Measure.sum fun n => HSMul.hSMul …
    ⊢ Exists fun ν => And (MeasureTheory.IsFiniteMeasure ν) (And (μ.AbsolutelyCont …
  -/
  refine ⟨.sum fun n ↦ c n • sfiniteSeq μ n, ⟨?_⟩, fun _ ↦ this.1, fun _ ↦ this.2⟩
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    c : Nat → NNReal
    hc₀ : ∀ (i : Nat), LT.lt 0 (c i)
    hc : LT.lt (tsum fun i => HMul.hMul ((MeasureTheory.sfiniteSeq μ i) Set.univ)  …
    this : ∀ {s : Set α}, Iff (Eq ((MeasureTheory.Measure.sum fun n => HSMul.hSMul …
    ⊢ LT.lt ((MeasureTheory.Measure.sum fun n => HSMul.hSMul (c n) (MeasureTheory. …
  -/
  simpa [mul_comm] using hc
  /-
    🎉 no goals
  -/


variable (μ) in
@[deprecated exists_isFiniteMeasure_absolutelyContinuous (since := "2024-08-25")]
theorem exists_absolutelyContinuous_isFiniteMeasure [SFinite μ] :
    ∃ ν : Measure α, IsFiniteMeasure ν ∧ μ ≪ ν :=
  let ⟨ν, hfin, h, _⟩ := exists_isFiniteMeasure_absolutelyContinuous μ; ⟨ν, hfin, h⟩


/-- A measure `μ` is called σ-finite if there is a countable collection of sets
 `{ A i | i ∈ ℕ }` such that `μ (A i) < ∞` and `⋃ i, A i = s`. -/
class SigmaFinite {m0 : MeasurableSpace α} (μ : Measure α) : Prop where
  out' : Nonempty (μ.FiniteSpanningSetsIn univ)


theorem sigmaFinite_iff : SigmaFinite μ ↔ Nonempty (μ.FiniteSpanningSetsIn univ) :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


theorem SigmaFinite.out (h : SigmaFinite μ) : Nonempty (μ.FiniteSpanningSetsIn univ) :=
  h.1


/-- If `μ` is σ-finite it has finite spanning sets in the collection of all measurable sets. -/
def Measure.toFiniteSpanningSetsIn (μ : Measure α) [h : SigmaFinite μ] :
    μ.FiniteSpanningSetsIn { s | MeasurableSet s } where
  set n := toMeasurable μ (h.out.some.set n)
  set_mem _ := measurableSet_toMeasurable _ _
  finite n := by
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      μ : MeasureTheory.Measure α
      h : MeasureTheory.SigmaFinite μ
      n : Nat
      ⊢ LT.lt (μ ((fun n => MeasureTheory.toMeasurable μ (⋯.some.set n)) n)) Top.top
    -/
    rw [measure_toMeasurable]
    /-
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι : Type u_4
      m0 : MeasurableSpace α
      inst✝ : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      μ : MeasureTheory.Measure α
      h : MeasureTheory.SigmaFinite μ
      n : Nat
      ⊢ LT.lt (μ (⋯.some.set n)) Top.top
    -/
    exact h.out.some.finite n
    /-
      🎉 no goals
    -/
  spanning := eq_univ_of_subset (iUnion_mono fun _ => subset_toMeasurable _ _) h.out.some.spanning


/-- A noncomputable way to get a monotone collection of sets that span `univ` and have finite
  measure using `Classical.choose`. This definition satisfies monotonicity in addition to all other
  properties in `SigmaFinite`. -/
def spanningSets (μ : Measure α) [SigmaFinite μ] (i : ℕ) : Set α :=
  Accumulate μ.toFiniteSpanningSetsIn.set i


theorem monotone_spanningSets (μ : Measure α) [SigmaFinite μ] : Monotone (spanningSets μ) :=
  monotone_accumulate


@[gcongr]
lemma spanningSets_mono [SigmaFinite μ] {m n : ℕ} (hmn : m ≤ n) :
    spanningSets μ m ⊆ spanningSets μ n := monotone_spanningSets _ hmn


theorem measurableSet_spanningSets (μ : Measure α) [SigmaFinite μ] (i : ℕ) :
    MeasurableSet (spanningSets μ i) :=
  MeasurableSet.iUnion fun j => MeasurableSet.iUnion fun _ => μ.toFiniteSpanningSetsIn.set_mem j


@[deprecated (since := "2024-10-16")] alias measurable_spanningSets := measurableSet_spanningSets


theorem measure_spanningSets_lt_top (μ : Measure α) [SigmaFinite μ] (i : ℕ) :
    μ (spanningSets μ i) < ∞ :=
  measure_biUnion_lt_top (finite_le_nat i) fun j _ => μ.toFiniteSpanningSetsIn.finite j


@[simp]
theorem iUnion_spanningSets (μ : Measure α) [SigmaFinite μ] : ⋃ i : ℕ, spanningSets μ i = univ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ Eq (Set.iUnion fun i => MeasureTheory.spanningSets μ i) Set.univ
  -/
  simp_rw [spanningSets, iUnion_accumulate, μ.toFiniteSpanningSetsIn.spanning]
  /-
    🎉 no goals
  -/


theorem isCountablySpanning_spanningSets (μ : Measure α) [SigmaFinite μ] :
    IsCountablySpanning (range (spanningSets μ)) :=
  ⟨spanningSets μ, mem_range_self, iUnion_spanningSets μ⟩


open scoped Classical in
/-- `spanningSetsIndex μ x` is the least `n : ℕ` such that `x ∈ spanningSets μ n`. -/
noncomputable def spanningSetsIndex (μ : Measure α) [SigmaFinite μ] (x : α) : ℕ :=
  Nat.find <| iUnion_eq_univ_iff.1 (iUnion_spanningSets μ) x


open scoped Classical in
theorem measurableSet_spanningSetsIndex (μ : Measure α) [SigmaFinite μ] :
    Measurable (spanningSetsIndex μ) :=
  measurable_find _ <| measurableSet_spanningSets μ


open scoped Classical in
theorem preimage_spanningSetsIndex_singleton (μ : Measure α) [SigmaFinite μ] (n : ℕ) :
    spanningSetsIndex μ ⁻¹' {n} = disjointed (spanningSets μ) n :=
  preimage_find_eq_disjointed _ _ _


theorem spanningSetsIndex_eq_iff (μ : Measure α) [SigmaFinite μ] {x : α} {n : ℕ} :
    spanningSetsIndex μ x = n ↔ x ∈ disjointed (spanningSets μ) n := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    x : α
    n : Nat
    ⊢ Iff (Eq (MeasureTheory.spanningSetsIndex μ x) n) (Membership.mem (disjointed …
  -/
  convert Set.ext_iff.1 (preimage_spanningSetsIndex_singleton μ n) x
  /-
    🎉 no goals
  -/


theorem mem_disjointed_spanningSetsIndex (μ : Measure α) [SigmaFinite μ] (x : α) :
    x ∈ disjointed (spanningSets μ) (spanningSetsIndex μ x) :=
  (spanningSetsIndex_eq_iff μ).1 rfl


theorem mem_spanningSetsIndex (μ : Measure α) [SigmaFinite μ] (x : α) :
    x ∈ spanningSets μ (spanningSetsIndex μ x) :=
  disjointed_subset _ _ (mem_disjointed_spanningSetsIndex μ x)


theorem mem_spanningSets_of_index_le (μ : Measure α) [SigmaFinite μ] (x : α) {n : ℕ}
    (hn : spanningSetsIndex μ x ≤ n) : x ∈ spanningSets μ n :=
  monotone_spanningSets μ hn (mem_spanningSetsIndex μ x)


theorem eventually_mem_spanningSets (μ : Measure α) [SigmaFinite μ] (x : α) :
    ∀ᶠ n in atTop, x ∈ spanningSets μ n :=
  eventually_atTop.2 ⟨spanningSetsIndex μ x, fun _ => mem_spanningSets_of_index_le μ x⟩


theorem sum_restrict_disjointed_spanningSets (μ ν : Measure α) [SigmaFinite ν] :
    sum (fun n ↦ μ.restrict (disjointed (spanningSets ν) n)) = μ := by
  rw [← restrict_iUnion (disjoint_disjointed _)
      (MeasurableSet.disjointed (measurableSet_spanningSets _)),
    iUnion_disjointed, iUnion_spanningSets, restrict_univ]


instance (priority := 100) [SigmaFinite μ] : SFinite μ := by
  have : ∀ n, Fact (μ (disjointed (spanningSets μ) n) < ∞) :=
    fun n ↦ ⟨(measure_mono (disjointed_subset _ _)).trans_lt (measure_spanningSets_lt_top μ n)⟩
  exact ⟨⟨fun n ↦ μ.restrict (disjointed (spanningSets μ) n), fun n ↦ by infer_instance,
    (sum_restrict_disjointed_spanningSets μ μ).symm⟩⟩


/-- A set in a σ-finite space has zero measure if and only if its intersection with
all members of the countable family of finite measure spanning sets has zero measure. -/
theorem forall_measure_inter_spanningSets_eq_zero [MeasurableSpace α] {μ : Measure α}
    [SigmaFinite μ] (s : Set α) : (∀ n, μ (s ∩ spanningSets μ n) = 0) ↔ μ s = 0 := by
  nth_rw 2 [show s = ⋃ n, s ∩ spanningSets μ n by
      rw [← inter_iUnion, iUnion_spanningSets, inter_univ] ]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    ⊢ Iff (∀ (n : Nat), Eq (μ (Inter.inter s (MeasureTheory.spanningSets μ n))) 0) …
  -/
  rw [measure_iUnion_null_iff]
  /-
    🎉 no goals
  -/


/-- A set in a σ-finite space has positive measure if and only if its intersection with
some member of the countable family of finite measure spanning sets has positive measure. -/
theorem exists_measure_inter_spanningSets_pos [MeasurableSpace α] {μ : Measure α} [SigmaFinite μ]
    (s : Set α) : (∃ n, 0 < μ (s ∩ spanningSets μ n)) ↔ 0 < μ s := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    ⊢ Iff (Exists fun n => LT.lt 0 (μ (Inter.inter s (MeasureTheory.spanningSets μ …
  -/
  rw [← not_iff_not]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    ⊢ Iff (Not (Exists fun n => LT.lt 0 (μ (Inter.inter s (MeasureTheory.spanningS …
  -/
  simp only [not_exists, not_lt, nonpos_iff_eq_zero]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    ⊢ Iff (∀ (x : Nat), Eq (μ (Inter.inter s (MeasureTheory.spanningSets μ x))) 0) …
  -/
  exact forall_measure_inter_spanningSets_eq_zero s
  /-
    🎉 no goals
  -/


/-- If the union of a.e.-disjoint null-measurable sets has finite measure, then there are only
finitely many members of the union whose measure exceeds any given positive number. -/
theorem finite_const_le_meas_of_disjoint_iUnion₀ {ι : Type*} [MeasurableSpace α] (μ : Measure α)
    {ε : ℝ≥0∞} (ε_pos : 0 < ε) {As : ι → Set α} (As_mble : ∀ i : ι, NullMeasurableSet (As i) μ)
    (As_disj : Pairwise (AEDisjoint μ on As)) (Union_As_finite : μ (⋃ i, As i) ≠ ∞) :
    Set.Finite { i : ι | ε ≤ μ (As i) } :=
  ENNReal.finite_const_le_of_tsum_ne_top
    (ne_top_of_le_ne_top Union_As_finite (tsum_meas_le_meas_iUnion_of_disjoint₀ μ As_mble As_disj))
    ε_pos.ne'


/-- If the union of disjoint measurable sets has finite measure, then there are only
finitely many members of the union whose measure exceeds any given positive number. -/
theorem finite_const_le_meas_of_disjoint_iUnion {ι : Type*} [MeasurableSpace α] (μ : Measure α)
    {ε : ℝ≥0∞} (ε_pos : 0 < ε) {As : ι → Set α} (As_mble : ∀ i : ι, MeasurableSet (As i))
    (As_disj : Pairwise (Disjoint on As)) (Union_As_finite : μ (⋃ i, As i) ≠ ∞) :
    Set.Finite { i : ι | ε ≤ μ (As i) } :=
  finite_const_le_meas_of_disjoint_iUnion₀ μ ε_pos (fun i ↦ (As_mble i).nullMeasurableSet)
    (fun _ _ h ↦ Disjoint.aedisjoint (As_disj h)) Union_As_finite


/-- If all elements of an infinite set have measure uniformly separated from zero,
then the set has infinite measure. -/
theorem _root_.Set.Infinite.meas_eq_top [MeasurableSingletonClass α]
    {s : Set α} (hs : s.Infinite) (h' : ∃ ε, ε ≠ 0 ∧ ∀ x ∈ s, ε ≤ μ {x}) : μ s = ∞ := top_unique <|
  let ⟨ε, hne, hε⟩ := h'; have := hs.to_subtype
  calc
    ∞ = ∑' _ : s, ε := (ENNReal.tsum_const_eq_top_of_ne_zero hne).symm
    _ ≤ ∑' x : s, μ {x.1} := ENNReal.tsum_le_tsum fun x ↦ hε x x.2
    _ ≤ μ (⋃ x : s, {x.1}) := tsum_meas_le_meas_iUnion_of_disjoint _
                                                           /-
                                                             α : Type u_1
                                                             m0 : MeasurableSpace α
                                                             μ : MeasureTheory.Measure α
                                                             inst✝ : MeasurableSingletonClass α
                                                             s : Set α
                                                             hs : s.Infinite
                                                             h' : Exists fun ε => And (Ne ε 0) (∀ (x : α), Membership.mem s x → LE.le ε (μ  …
                                                             ε : ENNReal
                                                             hne✝ : Ne ε 0
                                                             hε : ∀ (x : α), Membership.mem s x → LE.le ε (μ (Singleton.singleton x))
                                                             this : Infinite ↑s
                                                             x y : ↑s
                                                             hne : Ne x y
                                                             ⊢ Function.onFun Disjoint (fun x => Singleton.singleton ↑x) x y
                                                           -/
      (fun _ ↦ MeasurableSet.singleton _) fun x y hne ↦ by simpa [Subtype.val_inj]
                                                           /-
                                                             🎉 no goals
                                                           -/
                  /-
                    α : Type u_1
                    m0 : MeasurableSpace α
                    μ : MeasureTheory.Measure α
                    inst✝ : MeasurableSingletonClass α
                    s : Set α
                    hs : s.Infinite
                    h' : Exists fun ε => And (Ne ε 0) (∀ (x : α), Membership.mem s x → LE.le ε (μ  …
                    ε : ENNReal
                    hne : Ne ε 0
                    hε : ∀ (x : α), Membership.mem s x → LE.le ε (μ (Singleton.singleton x))
                    this : Infinite ↑s
                    ⊢ Eq (μ (Set.iUnion fun x => Singleton.singleton ↑x)) (μ s)
                  -/
    _ = μ s := by simp
                  /-
                    🎉 no goals
                  -/


/-- If the union of a.e.-disjoint null-measurable sets has finite measure, then there are only
countably many members of the union whose measure is positive. -/
theorem countable_meas_pos_of_disjoint_of_meas_iUnion_ne_top₀ {ι : Type*} {_ : MeasurableSpace α}
    (μ : Measure α) {As : ι → Set α} (As_mble : ∀ i : ι, NullMeasurableSet (As i) μ)
    (As_disj : Pairwise (AEDisjoint μ on As)) (Union_As_finite : μ (⋃ i, As i) ≠ ∞) :
    Set.Countable { i : ι | 0 < μ (As i) } := by
  /-
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    Union_As_finite : Ne (μ (Set.iUnion fun i => As i)) Top.top
    ⊢ (setOf fun i => LT.lt 0 (μ (As i))).Countable
  -/
  set posmeas := { i : ι | 0 < μ (As i) } with posmeas_def
  rcases exists_seq_strictAnti_tendsto' (zero_lt_one : (0 : ℝ≥0∞) < 1) with
    ⟨as, _, as_mem, as_lim⟩
  /-
    case intro.intro.intro
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    Union_As_finite : Ne (μ (Set.iUnion fun i => As i)) Top.top
    posmeas : Set ι := setOf fun i => LT.lt 0 (μ (As i))
    posmeas_def : Eq posmeas (setOf fun i => LT.lt 0 (μ (As i)))
    as : Nat → ENNReal
    left✝ : StrictAnti as
    as_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (as n)
    as_lim : Filter.Tendsto as Filter.atTop (nhds 0)
    ⊢ posmeas.Countable
  -/
  set fairmeas := fun n : ℕ => { i : ι | as n ≤ μ (As i) }
  have countable_union : posmeas = ⋃ n, fairmeas n := by
    have fairmeas_eq : ∀ n, fairmeas n = (fun i => μ (As i)) ⁻¹' Ici (as n) := fun n => by
      simp only [fairmeas]
      rfl
    simpa only [fairmeas_eq, posmeas_def, ← preimage_iUnion,
      iUnion_Ici_eq_Ioi_of_lt_of_tendsto (fun n => (as_mem n).1) as_lim]
  /-
    case intro.intro.intro
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    Union_As_finite : Ne (μ (Set.iUnion fun i => As i)) Top.top
    posmeas : Set ι := setOf fun i => LT.lt 0 (μ (As i))
    posmeas_def : Eq posmeas (setOf fun i => LT.lt 0 (μ (As i)))
    as : Nat → ENNReal
    left✝ : StrictAnti as
    as_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (as n)
    as_lim : Filter.Tendsto as Filter.atTop (nhds 0)
    fairmeas : Nat → Set ι := fun n => setOf fun i => LE.le (as n) (μ (As i))
    countable_union : Eq posmeas (Set.iUnion fun n => fairmeas n)
    ⊢ posmeas.Countable
  -/
  rw [countable_union]
  /-
    case intro.intro.intro
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    Union_As_finite : Ne (μ (Set.iUnion fun i => As i)) Top.top
    posmeas : Set ι := setOf fun i => LT.lt 0 (μ (As i))
    posmeas_def : Eq posmeas (setOf fun i => LT.lt 0 (μ (As i)))
    as : Nat → ENNReal
    left✝ : StrictAnti as
    as_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (as n)
    as_lim : Filter.Tendsto as Filter.atTop (nhds 0)
    fairmeas : Nat → Set ι := fun n => setOf fun i => LE.le (as n) (μ (As i))
    countable_union : Eq posmeas (Set.iUnion fun n => fairmeas n)
    ⊢ (Set.iUnion fun n => fairmeas n).Countable
  -/
  refine countable_iUnion fun n => Finite.countable ?_
  /-
    case intro.intro.intro
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    Union_As_finite : Ne (μ (Set.iUnion fun i => As i)) Top.top
    posmeas : Set ι := setOf fun i => LT.lt 0 (μ (As i))
    posmeas_def : Eq posmeas (setOf fun i => LT.lt 0 (μ (As i)))
    as : Nat → ENNReal
    left✝ : StrictAnti as
    as_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (as n)
    as_lim : Filter.Tendsto as Filter.atTop (nhds 0)
    fairmeas : Nat → Set ι := fun n => setOf fun i => LE.le (as n) (μ (As i))
    countable_union : Eq posmeas (Set.iUnion fun n => fairmeas n)
    n : Nat
    ⊢ (fairmeas n).Finite
  -/
  exact finite_const_le_meas_of_disjoint_iUnion₀ μ (as_mem n).1 As_mble As_disj Union_As_finite
  /-
    🎉 no goals
  -/


/-- If the union of disjoint measurable sets has finite measure, then there are only
countably many members of the union whose measure is positive. -/
theorem countable_meas_pos_of_disjoint_of_meas_iUnion_ne_top {ι : Type*} {_ : MeasurableSpace α}
    (μ : Measure α) {As : ι → Set α} (As_mble : ∀ i : ι, MeasurableSet (As i))
    (As_disj : Pairwise (Disjoint on As)) (Union_As_finite : μ (⋃ i, As i) ≠ ∞) :
    Set.Countable { i : ι | 0 < μ (As i) } :=
  countable_meas_pos_of_disjoint_of_meas_iUnion_ne_top₀ μ (fun i ↦ (As_mble i).nullMeasurableSet)
    ((fun _ _ h ↦ Disjoint.aedisjoint (As_disj h))) Union_As_finite


/-- In an s-finite space, among disjoint null-measurable sets, only countably many can have positive
measure. -/
theorem countable_meas_pos_of_disjoint_iUnion₀ {ι : Type*} {_ : MeasurableSpace α} {μ : Measure α}
    [SFinite μ] {As : ι → Set α} (As_mble : ∀ i : ι, NullMeasurableSet (As i) μ)
    (As_disj : Pairwise (AEDisjoint μ on As)) :
    Set.Countable { i : ι | 0 < μ (As i) } := by
  /-
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) As)
    ⊢ (setOf fun i => LT.lt 0 (μ (As i))).Countable
  -/
  rw [← sum_sfiniteSeq μ] at As_disj As_mble ⊢
  have obs : { i : ι | 0 < sum (sfiniteSeq μ) (As i) }
      ⊆ ⋃ n, { i : ι | 0 < sfiniteSeq μ n (As i) } := by
    intro i hi
    by_contra con
    simp only [mem_iUnion, mem_setOf_eq, not_exists, not_lt, nonpos_iff_eq_zero] at *
    rw [sum_apply₀] at hi
    · simp_rw [con] at hi
      simp at hi
    · exact As_mble i
  /-
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) (MeasureTheory.Mea …
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint (MeasureTheory.Me …
    obs : HasSubset.Subset (setOf fun i => LT.lt 0 ((MeasureTheory.Measure.sum (Me …
    ⊢ (setOf fun i => LT.lt 0 ((MeasureTheory.Measure.sum (MeasureTheory.sfiniteSe …
  -/
  apply Countable.mono obs
  /-
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) (MeasureTheory.Mea …
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint (MeasureTheory.Me …
    obs : HasSubset.Subset (setOf fun i => LT.lt 0 ((MeasureTheory.Measure.sum (Me …
    ⊢ (Set.iUnion fun n => setOf fun i => LT.lt 0 ((MeasureTheory.sfiniteSeq μ n)  …
  -/
  refine countable_iUnion fun n ↦ ?_
  /-
    α : Type u_1
    ι : Type u_5
    x✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    As : ι → Set α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) (MeasureTheory.Mea …
    As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint (MeasureTheory.Me …
    obs : HasSubset.Subset (setOf fun i => LT.lt 0 ((MeasureTheory.Measure.sum (Me …
    n : Nat
    ⊢ (setOf fun i => LT.lt 0 ((MeasureTheory.sfiniteSeq μ n) (As i))).Countable
  -/
  apply countable_meas_pos_of_disjoint_of_meas_iUnion_ne_top₀
    /-
      case As_mble
      α : Type u_1
      ι : Type u_5
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      As : ι → Set α
      As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) (MeasureTheory.Mea …
      As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint (MeasureTheory.Me …
      obs : HasSubset.Subset (setOf fun i => LT.lt 0 ((MeasureTheory.Measure.sum (Me …
      n : Nat
      ⊢ ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) (MeasureTheory.sfiniteSeq  …
    -/
  · exact fun i ↦ (As_mble i).mono (le_sum _ _)
    /-
      🎉 no goals
    -/
    /-
      case As_disj
      α : Type u_1
      ι : Type u_5
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      As : ι → Set α
      As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) (MeasureTheory.Mea …
      As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint (MeasureTheory.Me …
      obs : HasSubset.Subset (setOf fun i => LT.lt 0 ((MeasureTheory.Measure.sum (Me …
      n : Nat
      ⊢ Pairwise (Function.onFun (MeasureTheory.AEDisjoint (MeasureTheory.sfiniteSeq …
    -/
  · exact fun i j hij ↦ AEDisjoint.of_le (As_disj hij) (le_sum _ _)
    /-
      🎉 no goals
    -/
    /-
      case Union_As_finite
      α : Type u_1
      ι : Type u_5
      x✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      As : ι → Set α
      As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) (MeasureTheory.Mea …
      As_disj : Pairwise (Function.onFun (MeasureTheory.AEDisjoint (MeasureTheory.Me …
      obs : HasSubset.Subset (setOf fun i => LT.lt 0 ((MeasureTheory.Measure.sum (Me …
      n : Nat
      ⊢ Ne ((MeasureTheory.sfiniteSeq μ n) (Set.iUnion fun i => As i)) Top.top
    -/
  · exact measure_ne_top _ (⋃ i, As i)
    /-
      🎉 no goals
    -/


/-- In an s-finite space, among disjoint measurable sets, only countably many can have positive
measure. -/
theorem countable_meas_pos_of_disjoint_iUnion {ι : Type*} {_ : MeasurableSpace α} {μ : Measure α}
    [SFinite μ] {As : ι → Set α} (As_mble : ∀ i : ι, MeasurableSet (As i))
    (As_disj : Pairwise (Disjoint on As)) : Set.Countable { i : ι | 0 < μ (As i) } :=
  countable_meas_pos_of_disjoint_iUnion₀ (fun i ↦ (As_mble i).nullMeasurableSet)
    ((fun _ _ h ↦ Disjoint.aedisjoint (As_disj h)))


theorem countable_meas_level_set_pos₀ {α β : Type*} {_ : MeasurableSpace α} {μ : Measure α}
    [SFinite μ] [MeasurableSpace β] [MeasurableSingletonClass β] {g : α → β}
    (g_mble : NullMeasurable g μ) : Set.Countable { t : β | 0 < μ { a : α | g a = t } } := by
  have level_sets_disjoint : Pairwise (Disjoint on fun t : β => { a : α | g a = t }) :=
    fun s t hst => Disjoint.preimage g (disjoint_singleton.mpr hst)
  exact Measure.countable_meas_pos_of_disjoint_iUnion₀
    (fun b => g_mble (‹MeasurableSingletonClass β›.measurableSet_singleton b))
    ((fun _ _ h ↦ Disjoint.aedisjoint (level_sets_disjoint h)))


theorem countable_meas_level_set_pos {α β : Type*} {_ : MeasurableSpace α} {μ : Measure α}
    [SFinite μ] [MeasurableSpace β] [MeasurableSingletonClass β] {g : α → β}
    (g_mble : Measurable g) : Set.Countable { t : β | 0 < μ { a : α | g a = t } } :=
  countable_meas_level_set_pos₀ g_mble.nullMeasurable


/-- If a measure `μ` is the sum of a countable family `mₙ`, and a set `t` has finite measure for
each `mₙ`, then its measurable superset `toMeasurable μ t` (which has the same measure as `t`)
satisfies, for any measurable set `s`, the equality `μ (toMeasurable μ t ∩ s) = μ (t ∩ s)`. -/
theorem measure_toMeasurable_inter_of_sum {s : Set α} (hs : MeasurableSet s) {t : Set α}
    {m : ℕ → Measure α} (hv : ∀ n, m n t ≠ ∞) (hμ : μ = sum m) :
    μ (toMeasurable μ t ∩ s) = μ (t ∩ s) := by
  -- we show that there is a measurable superset of `t` satisfying the conclusion for any
  -- measurable set `s`. It is built for each measure `mₙ` using `toMeasurable`
  -- (which is well behaved for finite measure sets thanks to `measure_toMeasurable_inter`), and
  -- then taking the intersection over `n`.
  have A : ∃ t', t' ⊇ t ∧ MeasurableSet t' ∧ ∀ u, MeasurableSet u → μ (t' ∩ u) = μ (t ∩ u) := by
    let w n := toMeasurable (m n) t
    have T : t ⊆ ⋂ n, w n := subset_iInter (fun i ↦ subset_toMeasurable (m i) t)
    have M : MeasurableSet (⋂ n, w n) :=
      MeasurableSet.iInter (fun i ↦ measurableSet_toMeasurable (m i) t)
    refine ⟨⋂ n, w n, T, M, fun u hu ↦ ?_⟩
    refine le_antisymm ?_ (by gcongr)
    rw [hμ, sum_apply _ (M.inter hu)]
    apply le_trans _ (le_sum_apply _ _)
    apply ENNReal.tsum_le_tsum (fun i ↦ ?_)
    calc
    m i ((⋂ n, w n) ∩ u) ≤ m i (w i ∩ u) := by gcongr; apply iInter_subset
    _ = m i (t ∩ u) := measure_toMeasurable_inter hu (hv i)
  -- thanks to the definition of `toMeasurable`, the previous property will also be shared
  -- by `toMeasurable μ t`, which is enough to conclude the proof.
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    t : Set α
    m : Nat → MeasureTheory.Measure α
    hv : ∀ (n : Nat), Ne ((m n) t) Top.top
    hμ : Eq μ (MeasureTheory.Measure.sum m)
    A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
    ⊢ Eq (μ (Inter.inter (MeasureTheory.toMeasurable μ t) s)) (μ (Inter.inter t s))
  -/
  rw [toMeasurable]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    t : Set α
    m : Nat → MeasureTheory.Measure α
    hv : ∀ (n : Nat), Ne ((m n) t) Top.top
    hμ : Eq μ (MeasureTheory.Measure.sum m)
    A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
    ⊢ Eq (μ (Inter.inter (dite (Exists fun t_1 => And (Superset t_1 t) (And (Measu …
  -/
  split_ifs with ht
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      m : Nat → MeasureTheory.Measure α
      hv : ∀ (n : Nat), Ne ((m n) t) Top.top
      hμ : Eq μ (MeasureTheory.Measure.sum m)
      A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
      ht : Exists fun t_1 => And (Superset t_1 t) (And (MeasurableSet t_1) ((Measure …
      ⊢ Eq (μ (Inter.inter ht.choose s)) (μ (Inter.inter t s))
    -/
  · apply measure_congr
    /-
      case pos.H
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      m : Nat → MeasureTheory.Measure α
      hv : ∀ (n : Nat), Ne ((m n) t) Top.top
      hμ : Eq μ (MeasureTheory.Measure.sum m)
      A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
      ht : Exists fun t_1 => And (Superset t_1 t) (And (MeasurableSet t_1) ((Measure …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Inter.inter ht.choose s) (Inter.inter t s)
    -/
    exact ae_eq_set_inter ht.choose_spec.2.2 (ae_eq_refl _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      m : Nat → MeasureTheory.Measure α
      hv : ∀ (n : Nat), Ne ((m n) t) Top.top
      hμ : Eq μ (MeasureTheory.Measure.sum m)
      A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
      ht : Not (Exists fun t_1 => And (Superset t_1 t) (And (MeasurableSet t_1) ((Me …
      ⊢ Eq (μ (Inter.inter A.choose s)) (μ (Inter.inter t s))
    -/
  · exact A.choose_spec.2.2 s hs
    /-
      🎉 no goals
    -/


/-- If a set `t` is covered by a countable family of finite measure sets, then its measurable
superset `toMeasurable μ t` (which has the same measure as `t`) satisfies,
for any measurable set `s`, the equality `μ (toMeasurable μ t ∩ s) = μ (t ∩ s)`. -/
theorem measure_toMeasurable_inter_of_cover {s : Set α} (hs : MeasurableSet s) {t : Set α}
    {v : ℕ → Set α} (hv : t ⊆ ⋃ n, v n) (h'v : ∀ n, μ (t ∩ v n) ≠ ∞) :
    μ (toMeasurable μ t ∩ s) = μ (t ∩ s) := by
  -- we show that there is a measurable superset of `t` satisfying the conclusion for any
  -- measurable set `s`. It is built on each member of a spanning family using `toMeasurable`
  -- (which is well behaved for finite measure sets thanks to `measure_toMeasurable_inter`), and
  -- the desired property passes to the union.
  have A : ∃ t', t' ⊇ t ∧ MeasurableSet t' ∧ ∀ u, MeasurableSet u → μ (t' ∩ u) = μ (t ∩ u) := by
    let w n := toMeasurable μ (t ∩ v n)
    have hw : ∀ n, μ (w n) < ∞ := by
      intro n
      simp_rw [w, measure_toMeasurable]
      exact (h'v n).lt_top
    set t' := ⋃ n, toMeasurable μ (t ∩ disjointed w n) with ht'
    have tt' : t ⊆ t' :=
      calc
        t ⊆ ⋃ n, t ∩ disjointed w n := by
          rw [← inter_iUnion, iUnion_disjointed, inter_iUnion]
          intro x hx
          rcases mem_iUnion.1 (hv hx) with ⟨n, hn⟩
          refine mem_iUnion.2 ⟨n, ?_⟩
          have : x ∈ t ∩ v n := ⟨hx, hn⟩
          exact ⟨hx, subset_toMeasurable μ _ this⟩
        _ ⊆ ⋃ n, toMeasurable μ (t ∩ disjointed w n) :=
          iUnion_mono fun n => subset_toMeasurable _ _
    refine ⟨t', tt', MeasurableSet.iUnion fun n => measurableSet_toMeasurable μ _, fun u hu => ?_⟩
    apply le_antisymm _ (by gcongr)
    calc
      μ (t' ∩ u) ≤ ∑' n, μ (toMeasurable μ (t ∩ disjointed w n) ∩ u) := by
        rw [ht', iUnion_inter]
        exact measure_iUnion_le _
      _ = ∑' n, μ (t ∩ disjointed w n ∩ u) := by
        congr 1
        ext1 n
        apply measure_toMeasurable_inter hu
        apply ne_of_lt
        calc
          μ (t ∩ disjointed w n) ≤ μ (t ∩ w n) := by
            gcongr
            exact disjointed_le w n
          _ ≤ μ (w n) := measure_mono inter_subset_right
          _ < ∞ := hw n
      _ = ∑' n, μ.restrict (t ∩ u) (disjointed w n) := by
        congr 1
        ext1 n
        rw [restrict_apply, inter_comm t _, inter_assoc]
        refine MeasurableSet.disjointed (fun n => ?_) n
        exact measurableSet_toMeasurable _ _
      _ = μ.restrict (t ∩ u) (⋃ n, disjointed w n) := by
        rw [measure_iUnion]
        · exact disjoint_disjointed _
        · intro i
          refine MeasurableSet.disjointed (fun n => ?_) i
          exact measurableSet_toMeasurable _ _
      _ ≤ μ.restrict (t ∩ u) univ := measure_mono (subset_univ _)
      _ = μ (t ∩ u) := by rw [restrict_apply MeasurableSet.univ, univ_inter]
  -- thanks to the definition of `toMeasurable`, the previous property will also be shared
  -- by `toMeasurable μ t`, which is enough to conclude the proof.
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    t : Set α
    v : Nat → Set α
    hv : HasSubset.Subset t (Set.iUnion fun n => v n)
    h'v : ∀ (n : Nat), Ne (μ (Inter.inter t (v n))) Top.top
    A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
    ⊢ Eq (μ (Inter.inter (MeasureTheory.toMeasurable μ t) s)) (μ (Inter.inter t s))
  -/
  rw [toMeasurable]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    t : Set α
    v : Nat → Set α
    hv : HasSubset.Subset t (Set.iUnion fun n => v n)
    h'v : ∀ (n : Nat), Ne (μ (Inter.inter t (v n))) Top.top
    A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
    ⊢ Eq (μ (Inter.inter (dite (Exists fun t_1 => And (Superset t_1 t) (And (Measu …
  -/
  split_ifs with ht
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      v : Nat → Set α
      hv : HasSubset.Subset t (Set.iUnion fun n => v n)
      h'v : ∀ (n : Nat), Ne (μ (Inter.inter t (v n))) Top.top
      A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
      ht : Exists fun t_1 => And (Superset t_1 t) (And (MeasurableSet t_1) ((Measure …
      ⊢ Eq (μ (Inter.inter ht.choose s)) (μ (Inter.inter t s))
    -/
  · apply measure_congr
    /-
      case pos.H
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      v : Nat → Set α
      hv : HasSubset.Subset t (Set.iUnion fun n => v n)
      h'v : ∀ (n : Nat), Ne (μ (Inter.inter t (v n))) Top.top
      A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
      ht : Exists fun t_1 => And (Superset t_1 t) (And (MeasurableSet t_1) ((Measure …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (Inter.inter ht.choose s) (Inter.inter t s)
    -/
    exact ae_eq_set_inter ht.choose_spec.2.2 (ae_eq_refl _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      v : Nat → Set α
      hv : HasSubset.Subset t (Set.iUnion fun n => v n)
      h'v : ∀ (n : Nat), Ne (μ (Inter.inter t (v n))) Top.top
      A : Exists fun t' => And (Superset t' t) (And (MeasurableSet t') (∀ (u : Set α …
      ht : Not (Exists fun t_1 => And (Superset t_1 t) (And (MeasurableSet t_1) ((Me …
      ⊢ Eq (μ (Inter.inter A.choose s)) (μ (Inter.inter t s))
    -/
  · exact A.choose_spec.2.2 s hs
    /-
      🎉 no goals
    -/


theorem restrict_toMeasurable_of_cover {s : Set α} {v : ℕ → Set α} (hv : s ⊆ ⋃ n, v n)
    (h'v : ∀ n, μ (s ∩ v n) ≠ ∞) : μ.restrict (toMeasurable μ s) = μ.restrict s :=
  ext fun t ht => by
    /-
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      v : Nat → Set α
      hv : HasSubset.Subset s (Set.iUnion fun n => v n)
      h'v : ∀ (n : Nat), Ne (μ (Inter.inter s (v n))) Top.top
      t : Set α
      ht : MeasurableSet t
      ⊢ Eq ((μ.restrict (MeasureTheory.toMeasurable μ s)) t) ((μ.restrict s) t)
    -/
    simp only [restrict_apply ht, inter_comm t, measure_toMeasurable_inter_of_cover ht hv h'v]
    /-
      🎉 no goals
    -/


/-- The measurable superset `toMeasurable μ t` of `t` (which has the same measure as `t`)
satisfies, for any measurable set `s`, the equality `μ (toMeasurable μ t ∩ s) = μ (t ∩ s)`.
This only holds when `μ` is s-finite -- for example for σ-finite measures. For a version without
this assumption (but requiring that `t` has finite measure), see `measure_toMeasurable_inter`. -/
theorem measure_toMeasurable_inter_of_sFinite [SFinite μ] {s : Set α} (hs : MeasurableSet s)
    (t : Set α) : μ (toMeasurable μ t ∩ s) = μ (t ∩ s) :=
  measure_toMeasurable_inter_of_sum hs (fun _ ↦ measure_ne_top _ t) (sum_sfiniteSeq μ).symm


@[simp]
theorem restrict_toMeasurable_of_sFinite [SFinite μ] (s : Set α) :
    μ.restrict (toMeasurable μ s) = μ.restrict s :=
  ext fun t ht => by
    rw [restrict_apply ht, inter_comm t, measure_toMeasurable_inter_of_sFinite ht,
      restrict_apply ht, inter_comm t]


/-- Auxiliary lemma for `iSup_restrict_spanningSets`. -/
theorem iSup_restrict_spanningSets_of_measurableSet [SigmaFinite μ] (hs : MeasurableSet s) :
    ⨆ i, μ.restrict (spanningSets μ i) s = μ s :=
  calc
    ⨆ i, μ.restrict (spanningSets μ i) s = μ.restrict (⋃ i, spanningSets μ i) s :=
      (restrict_iUnion_apply_eq_iSup (monotone_spanningSets μ).directed_le hs).symm
                  /-
                    α : Type u_1
                    m0 : MeasurableSpace α
                    μ : MeasureTheory.Measure α
                    s : Set α
                    inst✝ : MeasureTheory.SigmaFinite μ
                    hs : MeasurableSet s
                    ⊢ Eq ((μ.restrict (Set.iUnion fun i => MeasureTheory.spanningSets μ i)) s) (μ s)
                  -/
    _ = μ s := by rw [iUnion_spanningSets, restrict_univ]
                  /-
                    🎉 no goals
                  -/


theorem iSup_restrict_spanningSets [SigmaFinite μ] (s : Set α) :
    ⨆ i, μ.restrict (spanningSets μ i) s = μ s := by
  rw [← measure_toMeasurable s,
    ← iSup_restrict_spanningSets_of_measurableSet (measurableSet_toMeasurable _ _)]
  simp_rw [restrict_apply' (measurableSet_spanningSets μ _), Set.inter_comm s,
    ← restrict_apply (measurableSet_spanningSets μ _), ← restrict_toMeasurable_of_sFinite s,
    restrict_apply (measurableSet_spanningSets μ _), Set.inter_comm _ (toMeasurable μ s)]


/-- In a σ-finite space, any measurable set of measure `> r` contains a measurable subset of
finite measure `> r`. -/
theorem exists_subset_measure_lt_top [SigmaFinite μ] {r : ℝ≥0∞} (hs : MeasurableSet s)
    (h's : r < μ s) : ∃ t, MeasurableSet t ∧ t ⊆ s ∧ r < μ t ∧ μ t < ∞ := by
  rw [← iSup_restrict_spanningSets,
    @lt_iSup_iff _ _ _ r fun i : ℕ => μ.restrict (spanningSets μ i) s] at h's
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.SigmaFinite μ
    r : ENNReal
    hs : MeasurableSet s
    h's : Exists fun i => LT.lt r ((μ.restrict (MeasureTheory.spanningSets μ i)) s)
    ⊢ Exists fun t => And (MeasurableSet t) (And (HasSubset.Subset t s) (And (LT.l …
  -/
  rcases h's with ⟨n, hn⟩
  /-
    case intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.SigmaFinite μ
    r : ENNReal
    hs : MeasurableSet s
    n : Nat
    hn : LT.lt r ((μ.restrict (MeasureTheory.spanningSets μ n)) s)
    ⊢ Exists fun t => And (MeasurableSet t) (And (HasSubset.Subset t s) (And (LT.l …
  -/
  simp only [restrict_apply hs] at hn
  refine
    ⟨s ∩ spanningSets μ n, hs.inter (measurableSet_spanningSets _ _), inter_subset_left, hn, ?_⟩
  /-
    case intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.SigmaFinite μ
    r : ENNReal
    hs : MeasurableSet s
    n : Nat
    hn : LT.lt r (μ (Inter.inter s (MeasureTheory.spanningSets μ n)))
    ⊢ LT.lt (μ (Inter.inter s (MeasureTheory.spanningSets μ n))) Top.top
  -/
  exact (measure_mono inter_subset_right).trans_lt (measure_spanningSets_lt_top _ _)
  /-
    🎉 no goals
  -/


/-- If `μ` has finite spanning sets in `C` and `C ∩ {s | μ s < ∞} ⊆ D` then `μ` has finite spanning
sets in `D`. -/
protected def mono' (h : μ.FiniteSpanningSetsIn C) (hC : C ∩ { s | μ s < ∞ } ⊆ D) :
    μ.FiniteSpanningSetsIn D :=
  ⟨h.set, fun i => hC ⟨h.set_mem i, h.finite i⟩, h.finite, h.spanning⟩


/-- If `μ` has finite spanning sets in `C` and `C ⊆ D` then `μ` has finite spanning sets in `D`. -/
protected def mono (h : μ.FiniteSpanningSetsIn C) (hC : C ⊆ D) : μ.FiniteSpanningSetsIn D :=
  h.mono' fun _s hs => hC hs.1


/-- If `μ` has finite spanning sets in the collection of measurable sets `C`, then `μ` is σ-finite.
-/
protected theorem sigmaFinite (h : μ.FiniteSpanningSetsIn C) : SigmaFinite μ :=
  ⟨⟨h.mono <| subset_univ C⟩⟩


/-- An extensionality for measures. It is `ext_of_generateFrom_of_iUnion` formulated in terms of
`FiniteSpanningSetsIn`. -/
protected theorem ext {ν : Measure α} {C : Set (Set α)} (hA : ‹_› = generateFrom C)
    (hC : IsPiSystem C) (h : μ.FiniteSpanningSetsIn C) (h_eq : ∀ s ∈ C, μ s = ν s) : μ = ν :=
  ext_of_generateFrom_of_iUnion C _ hA hC h.spanning h.set_mem (fun i => (h.finite i).ne) h_eq


protected theorem isCountablySpanning (h : μ.FiniteSpanningSetsIn C) : IsCountablySpanning C :=
  ⟨h.set, h.set_mem, h.spanning⟩


theorem sigmaFinite_of_countable {S : Set (Set α)} (hc : S.Countable) (hμ : ∀ s ∈ S, μ s < ∞)
    (hU : ⋃₀ S = univ) : SigmaFinite μ := by
  obtain ⟨s, hμ, hs⟩ : ∃ s : ℕ → Set α, (∀ n, μ (s n) < ∞) ∧ ⋃ n, s n = univ :=
    (@exists_seq_cover_iff_countable _ (fun x => μ x < ∞) ⟨∅, by simp⟩).2 ⟨S, hc, hμ, hU⟩
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    S : Set (Set α)
    hc : S.Countable
    hμ✝ : ∀ (s : Set α), Membership.mem S s → LT.lt (μ s) Top.top
    hU : Eq S.sUnion Set.univ
    s : Nat → Set α
    hμ : ∀ (n : Nat), LT.lt (μ (s n)) Top.top
    hs : Eq (Set.iUnion fun n => s n) Set.univ
    ⊢ MeasureTheory.SigmaFinite μ
  -/
  exact ⟨⟨⟨fun n => s n, fun _ => trivial, hμ, hs⟩⟩⟩
  /-
    🎉 no goals
  -/


/-- Given measures `μ`, `ν` where `ν ≤ μ`, `FiniteSpanningSetsIn.ofLe` provides the induced
`FiniteSpanningSet` with respect to `ν` from a `FiniteSpanningSet` with respect to `μ`. -/
def FiniteSpanningSetsIn.ofLE (h : ν ≤ μ) {C : Set (Set α)} (S : μ.FiniteSpanningSetsIn C) :
    ν.FiniteSpanningSetsIn C where
  set := S.set
  set_mem := S.set_mem
  finite n := lt_of_le_of_lt (le_iff'.1 h _) (S.finite n)
  spanning := S.spanning


theorem sigmaFinite_of_le (μ : Measure α) [hs : SigmaFinite μ] (h : ν ≤ μ) : SigmaFinite ν :=
  ⟨hs.out.map <| FiniteSpanningSetsIn.ofLE h⟩


@[simp] lemma add_right_inj (μ ν₁ ν₂ : Measure α) [SigmaFinite μ] :
    μ + ν₁ = μ + ν₂ ↔ ν₁ = ν₂ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ Iff (Eq (HAdd.hAdd μ ν₁) (HAdd.hAdd μ ν₂)) (Eq ν₁ ν₂)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ by rw [h]⟩
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (HAdd.hAdd μ ν₁) (HAdd.hAdd μ ν₂)
    ⊢ Eq ν₁ ν₂
  -/
  rw [ext_iff_of_iUnion_eq_univ (iUnion_spanningSets μ)]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (HAdd.hAdd μ ν₁) (HAdd.hAdd μ ν₂)
    ⊢ ∀ (i : Nat), Eq (ν₁.restrict (MeasureTheory.spanningSets μ i)) (ν₂.restrict  …
  -/
  intro i
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (HAdd.hAdd μ ν₁) (HAdd.hAdd μ ν₂)
    i : Nat
    ⊢ Eq (ν₁.restrict (MeasureTheory.spanningSets μ i)) (ν₂.restrict (MeasureTheor …
  -/
  ext s hs
  rw [← ENNReal.add_right_inj (measure_mono s.inter_subset_right |>.trans_lt <|
    measure_spanningSets_lt_top μ i).ne]
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    h : Eq (HAdd.hAdd μ ν₁) (HAdd.hAdd μ ν₂)
    i : Nat
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (HAdd.hAdd (μ (Inter.inter s (MeasureTheory.spanningSets μ i))) ((ν₁.rest …
  -/
  simp only [ext_iff', coe_add, Pi.add_apply] at h
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν₁ ν₂ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    i : Nat
    s : Set α
    hs : MeasurableSet s
    h : ∀ (s : Set α), Eq (HAdd.hAdd (μ s) (ν₁ s)) (HAdd.hAdd (μ s) (ν₂ s))
    ⊢ Eq (HAdd.hAdd (μ (Inter.inter s (MeasureTheory.spanningSets μ i))) ((ν₁.rest …
  -/
  simp [hs, h]
  /-
    🎉 no goals
  -/


@[simp] lemma add_left_inj (μ ν₁ ν₂ : Measure α) [SigmaFinite μ] :
                                    /-
                                      α : Type u_1
                                      m0 : MeasurableSpace α
                                      μ ν₁ ν₂ : MeasureTheory.Measure α
                                      inst✝ : MeasureTheory.SigmaFinite μ
                                      ⊢ Iff (Eq (HAdd.hAdd ν₁ μ) (HAdd.hAdd ν₂ μ)) (Eq ν₁ ν₂)
                                    -/
    ν₁ + μ = ν₂ + μ ↔ ν₁ = ν₂ := by rw [add_comm _ μ, add_comm _ μ, μ.add_right_inj]
                                    /-
                                      🎉 no goals
                                    -/


/-- Every finite measure is σ-finite. -/
instance (priority := 100) IsFiniteMeasure.toSigmaFinite {_m0 : MeasurableSpace α} (μ : Measure α)
    [IsFiniteMeasure μ] : SigmaFinite μ :=
  ⟨⟨⟨fun _ => univ, fun _ => trivial, fun _ => measure_lt_top μ _, iUnion_const _⟩⟩⟩


theorem sigmaFinite_bot_iff (μ : @Measure α ⊥) : SigmaFinite μ ↔ IsFiniteMeasure μ := by
  refine
    ⟨fun h => ⟨?_⟩, fun h => by
      haveI := h
      infer_instance⟩
  /-
    α : Type u_1
    μ : MeasureTheory.Measure α
    h : MeasureTheory.SigmaFinite μ
    ⊢ LT.lt (μ Set.univ) Top.top
  -/
  haveI : SigmaFinite μ := h
  /-
    α : Type u_1
    μ : MeasureTheory.Measure α
    h this : MeasureTheory.SigmaFinite μ
    ⊢ LT.lt (μ Set.univ) Top.top
  -/
  let s := spanningSets μ
  /-
    α : Type u_1
    μ : MeasureTheory.Measure α
    h this : MeasureTheory.SigmaFinite μ
    s : Nat → Set α := MeasureTheory.spanningSets μ
    ⊢ LT.lt (μ Set.univ) Top.top
  -/
  have hs_univ : ⋃ i, s i = Set.univ := iUnion_spanningSets μ
  /-
    α : Type u_1
    μ : MeasureTheory.Measure α
    h this : MeasureTheory.SigmaFinite μ
    s : Nat → Set α := MeasureTheory.spanningSets μ
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    ⊢ LT.lt (μ Set.univ) Top.top
  -/
  have hs_meas : ∀ i, MeasurableSet[⊥] (s i) := measurableSet_spanningSets μ
  /-
    α : Type u_1
    μ : MeasureTheory.Measure α
    h this : MeasureTheory.SigmaFinite μ
    s : Nat → Set α := MeasureTheory.spanningSets μ
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    hs_meas : ∀ (i : Nat), MeasurableSet (s i)
    ⊢ LT.lt (μ Set.univ) Top.top
  -/
  simp_rw [MeasurableSpace.measurableSet_bot_iff] at hs_meas
  /-
    α : Type u_1
    μ : MeasureTheory.Measure α
    h this : MeasureTheory.SigmaFinite μ
    s : Nat → Set α := MeasureTheory.spanningSets μ
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    hs_meas : ∀ (i : Nat), Or (Eq (s i) EmptyCollection.emptyCollection) (Eq (s i) …
    ⊢ LT.lt (μ Set.univ) Top.top
  -/
  by_cases h_univ_empty : (Set.univ : Set α) = ∅
    /-
      case pos
      α : Type u_1
      μ : MeasureTheory.Measure α
      h this : MeasureTheory.SigmaFinite μ
      s : Nat → Set α := MeasureTheory.spanningSets μ
      hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
      hs_meas : ∀ (i : Nat), Or (Eq (s i) EmptyCollection.emptyCollection) (Eq (s i) …
      h_univ_empty : Eq Set.univ EmptyCollection.emptyCollection
      ⊢ LT.lt (μ Set.univ) Top.top
    -/
  · rw [h_univ_empty, measure_empty]
    /-
      case pos
      α : Type u_1
      μ : MeasureTheory.Measure α
      h this : MeasureTheory.SigmaFinite μ
      s : Nat → Set α := MeasureTheory.spanningSets μ
      hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
      hs_meas : ∀ (i : Nat), Or (Eq (s i) EmptyCollection.emptyCollection) (Eq (s i) …
      h_univ_empty : Eq Set.univ EmptyCollection.emptyCollection
      ⊢ LT.lt 0 Top.top
    -/
    exact ENNReal.zero_ne_top.lt_top
    /-
      🎉 no goals
    -/
  obtain ⟨i, hsi⟩ : ∃ i, s i = Set.univ := by
    by_contra! h_not_univ
    have h_empty : ∀ i, s i = ∅ := by simpa [h_not_univ] using hs_meas
    simp only [h_empty, iUnion_empty] at hs_univ
    exact h_univ_empty hs_univ.symm
  /-
    case neg.intro
    α : Type u_1
    μ : MeasureTheory.Measure α
    h this : MeasureTheory.SigmaFinite μ
    s : Nat → Set α := MeasureTheory.spanningSets μ
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    hs_meas : ∀ (i : Nat), Or (Eq (s i) EmptyCollection.emptyCollection) (Eq (s i) …
    h_univ_empty : Not (Eq Set.univ EmptyCollection.emptyCollection)
    i : Nat
    hsi : Eq (s i) Set.univ
    ⊢ LT.lt (μ Set.univ) Top.top
  -/
  rw [← hsi]
  /-
    case neg.intro
    α : Type u_1
    μ : MeasureTheory.Measure α
    h this : MeasureTheory.SigmaFinite μ
    s : Nat → Set α := MeasureTheory.spanningSets μ
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    hs_meas : ∀ (i : Nat), Or (Eq (s i) EmptyCollection.emptyCollection) (Eq (s i) …
    h_univ_empty : Not (Eq Set.univ EmptyCollection.emptyCollection)
    i : Nat
    hsi : Eq (s i) Set.univ
    ⊢ LT.lt (μ (s i)) Top.top
  -/
  exact measure_spanningSets_lt_top μ i
  /-
    🎉 no goals
  -/


instance Restrict.sigmaFinite (μ : Measure α) [SigmaFinite μ] (s : Set α) :
    SigmaFinite (μ.restrict s) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t : Set α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    ⊢ MeasureTheory.SigmaFinite (μ.restrict s)
  -/
  refine ⟨⟨⟨spanningSets μ, fun _ => trivial, fun i => ?_, iUnion_spanningSets μ⟩⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t : Set α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    i : Nat
    ⊢ LT.lt ((μ.restrict s) (MeasureTheory.spanningSets μ i)) Top.top
  -/
  rw [Measure.restrict_apply (measurableSet_spanningSets μ i)]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t : Set α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    s : Set α
    i : Nat
    ⊢ LT.lt (μ (Inter.inter (MeasureTheory.spanningSets μ i) s)) Top.top
  -/
  exact (measure_mono inter_subset_left).trans_lt (measure_spanningSets_lt_top μ i)
  /-
    🎉 no goals
  -/


instance sum.sigmaFinite {ι} [Finite ι] (μ : ι → Measure α) [∀ i, SigmaFinite (μ i)] :
    SigmaFinite (sum μ) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι✝ : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    ι : Type u_5
    inst✝¹ : Finite ι
    μ : ι → MeasureTheory.Measure α
    inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite (μ i)
    ⊢ MeasureTheory.SigmaFinite (MeasureTheory.Measure.sum μ)
  -/
  cases nonempty_fintype ι
  have : ∀ n, MeasurableSet (⋂ i : ι, spanningSets (μ i) n) := fun n =>
    MeasurableSet.iInter fun i => measurableSet_spanningSets (μ i) n
  /-
    case intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι✝ : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    ι : Type u_5
    inst✝¹ : Finite ι
    μ : ι → MeasureTheory.Measure α
    inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite (μ i)
    val✝ : Fintype ι
    this : ∀ (n : Nat), MeasurableSet (Set.iInter fun i => MeasureTheory.spanningS …
    ⊢ MeasureTheory.SigmaFinite (MeasureTheory.Measure.sum μ)
  -/
  refine ⟨⟨⟨fun n => ⋂ i, spanningSets (μ i) n, fun _ => trivial, fun n => ?_, ?_⟩⟩⟩
    /-
      case intro.refine_1
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι✝ : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      ι : Type u_5
      inst✝¹ : Finite ι
      μ : ι → MeasureTheory.Measure α
      inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite (μ i)
      val✝ : Fintype ι
      this : ∀ (n : Nat), MeasurableSet (Set.iInter fun i => MeasureTheory.spanningS …
      n : Nat
      ⊢ LT.lt ((MeasureTheory.Measure.sum μ) ((fun n => Set.iInter fun i => MeasureT …
    -/
  · rw [sum_apply _ (this n), tsum_fintype, ENNReal.sum_lt_top]
    /-
      case intro.refine_1
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι✝ : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      ι : Type u_5
      inst✝¹ : Finite ι
      μ : ι → MeasureTheory.Measure α
      inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite (μ i)
      val✝ : Fintype ι
      this : ∀ (n : Nat), MeasurableSet (Set.iInter fun i => MeasureTheory.spanningS …
      n : Nat
      ⊢ ∀ (a : ι), Membership.mem Finset.univ a → LT.lt ((μ a) (Set.iInter fun i =>  …
    -/
    rintro i -
    /-
      case intro.refine_1
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι✝ : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      ι : Type u_5
      inst✝¹ : Finite ι
      μ : ι → MeasureTheory.Measure α
      inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite (μ i)
      val✝ : Fintype ι
      this : ∀ (n : Nat), MeasurableSet (Set.iInter fun i => MeasureTheory.spanningS …
      n : Nat
      i : ι
      ⊢ LT.lt ((μ i) (Set.iInter fun i => MeasureTheory.spanningSets (μ i) n)) Top.top
    -/
    exact (measure_mono <| iInter_subset _ i).trans_lt (measure_spanningSets_lt_top (μ i) n)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι✝ : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      ι : Type u_5
      inst✝¹ : Finite ι
      μ : ι → MeasureTheory.Measure α
      inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite (μ i)
      val✝ : Fintype ι
      this : ∀ (n : Nat), MeasurableSet (Set.iInter fun i => MeasureTheory.spanningS …
      ⊢ Eq (Set.iUnion fun i => (fun n => Set.iInter fun i => MeasureTheory.spanning …
    -/
  · rw [iUnion_iInter_of_monotone]
      /-
        case intro.refine_2
        α : Type u_1
        β : Type u_2
        δ : Type u_3
        ι✝ : Type u_4
        m0 : MeasurableSpace α
        inst✝² : MeasurableSpace β
        μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
        s t : Set α
        ι : Type u_5
        inst✝¹ : Finite ι
        μ : ι → MeasureTheory.Measure α
        inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite (μ i)
        val✝ : Fintype ι
        this : ∀ (n : Nat), MeasurableSet (Set.iInter fun i => MeasureTheory.spanningS …
        ⊢ Eq (Set.iInter fun i => Set.iUnion fun j => MeasureTheory.spanningSets (μ i) …
      -/
    · simp_rw [iUnion_spanningSets, iInter_univ]
      /-
        🎉 no goals
      -/
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      δ : Type u_3
      ι✝ : Type u_4
      m0 : MeasurableSpace α
      inst✝² : MeasurableSpace β
      μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
      s t : Set α
      ι : Type u_5
      inst✝¹ : Finite ι
      μ : ι → MeasureTheory.Measure α
      inst✝ : ∀ (i : ι), MeasureTheory.SigmaFinite (μ i)
      val✝ : Fintype ι
      this : ∀ (n : Nat), MeasurableSet (Set.iInter fun i => MeasureTheory.spanningS …
      ⊢ ∀ (i : ι), Monotone (MeasureTheory.spanningSets (μ i))
    -/
    exact fun i => monotone_spanningSets (μ i)
    /-
      🎉 no goals
    -/


instance Add.sigmaFinite (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν] :
    SigmaFinite (μ + ν) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν✝ ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ MeasureTheory.SigmaFinite (HAdd.hAdd μ ν)
  -/
  rw [← sum_cond]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν✝ ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    ⊢ MeasureTheory.SigmaFinite (MeasureTheory.Measure.sum fun b => cond b μ ν)
  -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  refine @sum.sigmaFinite _ _ _ _ _ (Bool.rec ?_ ?_) <;> simpa
                                                         /-
                                                           🎉 no goals
                                                         -/


instance SMul.sigmaFinite {μ : Measure α} [SigmaFinite μ] (c : ℝ≥0) :
    MeasureTheory.SigmaFinite (c • μ) where
  out' :=
  ⟨{  set := spanningSets μ
      set_mem := fun _ ↦ trivial
      finite := by
        /-
          α : Type u_1
          β : Type u_2
          δ : Type u_3
          ι : Type u_4
          m0 : MeasurableSpace α
          inst✝¹ : MeasurableSpace β
          μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
          s t : Set α
          μ : MeasureTheory.Measure α
          inst✝ : MeasureTheory.SigmaFinite μ
          c : NNReal
          ⊢ ∀ (i : Nat), LT.lt ((HSMul.hSMul c μ) (MeasureTheory.spanningSets μ i)) Top. …
        -/
        intro i
        /-
          α : Type u_1
          β : Type u_2
          δ : Type u_3
          ι : Type u_4
          m0 : MeasurableSpace α
          inst✝¹ : MeasurableSpace β
          μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
          s t : Set α
          μ : MeasureTheory.Measure α
          inst✝ : MeasureTheory.SigmaFinite μ
          c : NNReal
          i : Nat
          ⊢ LT.lt ((HSMul.hSMul c μ) (MeasureTheory.spanningSets μ i)) Top.top
        -/
        simp only [Measure.coe_smul, Pi.smul_apply, nnreal_smul_coe_apply]
        /-
          α : Type u_1
          β : Type u_2
          δ : Type u_3
          ι : Type u_4
          m0 : MeasurableSpace α
          inst✝¹ : MeasurableSpace β
          μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
          s t : Set α
          μ : MeasureTheory.Measure α
          inst✝ : MeasureTheory.SigmaFinite μ
          c : NNReal
          i : Nat
          ⊢ LT.lt (HMul.hMul (↑c) (μ (MeasureTheory.spanningSets μ i))) Top.top
        -/
        exact ENNReal.mul_lt_top ENNReal.coe_lt_top (measure_spanningSets_lt_top μ i)
        /-
          🎉 no goals
        -/
      spanning := iUnion_spanningSets μ }⟩


instance [SigmaFinite (μ.restrict s)] [SigmaFinite (μ.restrict t)] :
    SigmaFinite (μ.restrict (s ∪ t)) := sigmaFinite_of_le _ (restrict_union_le _ _)


instance [SigmaFinite (μ.restrict s)] : SigmaFinite (μ.restrict (s ∩ t)) :=
  sigmaFinite_of_le (μ.restrict s) (restrict_mono_ae (ae_of_all _ Set.inter_subset_left))


instance [SigmaFinite (μ.restrict t)] : SigmaFinite (μ.restrict (s ∩ t)) :=
  sigmaFinite_of_le (μ.restrict t) (restrict_mono_ae (ae_of_all _ Set.inter_subset_right))


theorem SigmaFinite.of_map (μ : Measure α) {f : α → β} (hf : AEMeasurable f μ)
    (h : SigmaFinite (μ.map f)) : SigmaFinite μ :=
  ⟨⟨⟨fun n => f ⁻¹' spanningSets (μ.map f) n, fun _ => trivial, fun n => by
        simp only [← map_apply_of_aemeasurable hf, measurableSet_spanningSets,
          measure_spanningSets_lt_top],
           /-
             α : Type u_1
             β : Type u_2
             m0 : MeasurableSpace α
             inst✝ : MeasurableSpace β
             μ : MeasureTheory.Measure α
             f : α → β
             hf : AEMeasurable f μ
             h : MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f μ)
             ⊢ Eq (Set.iUnion fun i => (fun n => Set.preimage f (MeasureTheory.spanningSets …
           -/
        by rw [← preimage_iUnion, iUnion_spanningSets, preimage_univ]⟩⟩⟩
           /-
             🎉 no goals
           -/


lemma _root_.MeasurableEmbedding.sigmaFinite_map {f : α → β} (hf : MeasurableEmbedding f)
    [SigmaFinite μ] :
    SigmaFinite (μ.map f) := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    μ : MeasureTheory.Measure α
    f : α → β
    hf : MeasurableEmbedding f
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ MeasureTheory.SigmaFinite (MeasureTheory.Measure.map f μ)
  -/
  refine ⟨fun n ↦ f '' (spanningSets μ n) ∪ (Set.range f)ᶜ, by simp, fun n ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f : α → β
      hf : MeasurableEmbedding f
      inst✝ : MeasureTheory.SigmaFinite μ
      n : Nat
      ⊢ LT.lt ((MeasureTheory.Measure.map f μ) ((fun n => Union.union (Set.image f ( …
    -/
  · rw [hf.map_apply, Set.preimage_union]
    simp only [Set.preimage_compl, Set.preimage_range, Set.compl_univ, Set.union_empty,
      Set.preimage_image_eq _ hf.injective]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      μ : MeasureTheory.Measure α
      f : α → β
      hf : MeasurableEmbedding f
      inst✝ : MeasureTheory.SigmaFinite μ
      n : Nat
      ⊢ LT.lt (μ (MeasureTheory.spanningSets μ n)) Top.top
    -/
    exact measure_spanningSets_lt_top μ n
    /-
      🎉 no goals
    -/
  · rw [← Set.iUnion_union, ← Set.image_iUnion, iUnion_spanningSets,
      Set.image_univ, Set.union_compl_self]


theorem _root_.MeasurableEquiv.sigmaFinite_map (f : α ≃ᵐ β) [SigmaFinite μ] :
    SigmaFinite (μ.map f) := f.measurableEmbedding.sigmaFinite_map


/-- Similar to `ae_of_forall_measure_lt_top_ae_restrict`, but where you additionally get the
  hypothesis that another σ-finite measure has finite values on `s`. -/
theorem ae_of_forall_measure_lt_top_ae_restrict' {μ : Measure α} (ν : Measure α) [SigmaFinite μ]
    [SigmaFinite ν] (P : α → Prop)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → ν s < ∞ → ∀ᵐ x ∂μ.restrict s, P x) : ∀ᵐ x ∂μ, P x := by
  have : ∀ n, ∀ᵐ x ∂μ, x ∈ spanningSets (μ + ν) n → P x := by
    intro n
    have := h
      (spanningSets (μ + ν) n) (measurableSet_spanningSets _ _)
      ((self_le_add_right _ _).trans_lt (measure_spanningSets_lt_top (μ + ν) _))
      ((self_le_add_left _ _).trans_lt (measure_spanningSets_lt_top (μ + ν) _))
    exact (ae_restrict_iff' (measurableSet_spanningSets _ _)).mp this
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    P : α → Prop
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LT.lt (ν s) Top.top …
    this : ∀ (n : Nat), Filter.Eventually (fun x => Membership.mem (MeasureTheory. …
    ⊢ Filter.Eventually (fun x => P x) (MeasureTheory.ae μ)
  -/
  filter_upwards [ae_all_iff.2 this] with _ hx using hx _ (mem_spanningSetsIndex _ _)
  /-
    🎉 no goals
  -/


/-- To prove something for almost all `x` w.r.t. a σ-finite measure, it is sufficient to show that
  this holds almost everywhere in sets where the measure has finite value. -/
theorem ae_of_forall_measure_lt_top_ae_restrict {μ : Measure α} [SigmaFinite μ] (P : α → Prop)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → ∀ᵐ x ∂μ.restrict s, P x) : ∀ᵐ x ∂μ, P x :=
  ae_of_forall_measure_lt_top_ae_restrict' μ P fun s hs h2s _ => h s hs h2s


/-- A measure is called locally finite if it is finite in some neighborhood of each point. -/
class IsLocallyFiniteMeasure [TopologicalSpace α] (μ : Measure α) : Prop where
  finiteAtNhds : ∀ x, μ.FiniteAtFilter (𝓝 x)

-- see Note [lower instance priority]

instance (priority := 100) IsFiniteMeasure.toIsLocallyFiniteMeasure [TopologicalSpace α]
    (μ : Measure α) [IsFiniteMeasure μ] : IsLocallyFiniteMeasure μ :=
  ⟨fun _ => finiteAtFilter_of_finite _ _⟩


theorem Measure.finiteAt_nhds [TopologicalSpace α] (μ : Measure α) [IsLocallyFiniteMeasure μ]
    (x : α) : μ.FiniteAtFilter (𝓝 x) :=
  IsLocallyFiniteMeasure.finiteAtNhds x


theorem Measure.smul_finite (μ : Measure α) [IsFiniteMeasure μ] {c : ℝ≥0∞} (hc : c ≠ ∞) :
    IsFiniteMeasure (c • μ) := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : ENNReal
    hc : Ne c Top.top
    ⊢ MeasureTheory.IsFiniteMeasure (HSMul.hSMul c μ)
  -/
  lift c to ℝ≥0 using hc
  /-
    case intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : NNReal
    ⊢ MeasureTheory.IsFiniteMeasure (HSMul.hSMul (↑c) μ)
  -/
  exact MeasureTheory.isFiniteMeasureSMulNNReal
  /-
    🎉 no goals
  -/


theorem Measure.exists_isOpen_measure_lt_top [TopologicalSpace α] (μ : Measure α)
    [IsLocallyFiniteMeasure μ] (x : α) : ∃ s : Set α, x ∈ s ∧ IsOpen s ∧ μ s < ∞ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    x : α
    ⊢ Exists fun s => And (Membership.mem s x) (And (IsOpen s) (LT.lt (μ s) Top.to …
  -/
  simpa only [and_assoc] using (μ.finiteAt_nhds x).exists_mem_basis (nhds_basis_opens x)
  /-
    🎉 no goals
  -/


instance isLocallyFiniteMeasureSMulNNReal [TopologicalSpace α] (μ : Measure α)
    [IsLocallyFiniteMeasure μ] (c : ℝ≥0) : IsLocallyFiniteMeasure (c • μ) := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    c : NNReal
    ⊢ MeasureTheory.IsLocallyFiniteMeasure (HSMul.hSMul c μ)
  -/
  refine ⟨fun x => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    c : NNReal
    x : α
    ⊢ (HSMul.hSMul c μ).FiniteAtFilter (nhds x)
  -/
  rcases μ.exists_isOpen_measure_lt_top x with ⟨o, xo, o_open, μo⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    c : NNReal
    x : α
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    ⊢ (HSMul.hSMul c μ).FiniteAtFilter (nhds x)
  -/
  refine ⟨o, o_open.mem_nhds xo, ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    c : NNReal
    x : α
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    ⊢ LT.lt ((HSMul.hSMul c μ) o) Top.top
  -/
  apply ENNReal.mul_lt_top _ μo
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝² : MeasurableSpace β
    μ✝ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    c : NNReal
    x : α
    o : Set α
    xo : Membership.mem o x
    o_open : IsOpen o
    μo : LT.lt (μ o) Top.top
    ⊢ LT.lt (↑ENNReal.ofNNRealHom.toMonoidWithZeroHom c) Top.top
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem Measure.isTopologicalBasis_isOpen_lt_top [TopologicalSpace α]
    (μ : Measure α) [IsLocallyFiniteMeasure μ] :
    TopologicalSpace.IsTopologicalBasis { s | IsOpen s ∧ μ s < ∞ } := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ TopologicalSpace.IsTopologicalBasis (setOf fun s => And (IsOpen s) (LT.lt (μ …
  -/
  refine TopologicalSpace.isTopologicalBasis_of_isOpen_of_nhds (fun s hs => hs.1) ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ ∀ (a : α) (u : Set α), Membership.mem u a → IsOpen u → Exists fun v => And ( …
  -/
  intro x s xs hs
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    x : α
    s : Set α
    xs : Membership.mem s x
    hs : IsOpen s
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => And (IsOpen s) (LT.lt (μ …
  -/
  rcases μ.exists_isOpen_measure_lt_top x with ⟨v, xv, hv, μv⟩
  /-
    case intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    x : α
    s : Set α
    xs : Membership.mem s x
    hs : IsOpen s
    v : Set α
    xv : Membership.mem v x
    hv : IsOpen v
    μv : LT.lt (μ v) Top.top
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => And (IsOpen s) (LT.lt (μ …
  -/
  refine ⟨v ∩ s, ⟨hv.inter hs, lt_of_le_of_lt ?_ μv⟩, ⟨xv, xs⟩, inter_subset_right⟩
  /-
    case intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    inst✝¹ : TopologicalSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    x : α
    s : Set α
    xs : Membership.mem s x
    hs : IsOpen s
    v : Set α
    xv : Membership.mem v x
    hv : IsOpen v
    μv : LT.lt (μ v) Top.top
    ⊢ LE.le (μ (Inter.inter v s)) (μ v)
  -/
  exact measure_mono inter_subset_left
  /-
    🎉 no goals
  -/


/-- A measure `μ` is finite on compacts if any compact set `K` satisfies `μ K < ∞`. -/
class IsFiniteMeasureOnCompacts [TopologicalSpace α] (μ : Measure α) : Prop where
  protected lt_top_of_isCompact : ∀ ⦃K : Set α⦄, IsCompact K → μ K < ∞


/-- A compact subset has finite measure for a measure which is finite on compacts. -/
theorem _root_.IsCompact.measure_lt_top [TopologicalSpace α] {μ : Measure α}
    [IsFiniteMeasureOnCompacts μ] ⦃K : Set α⦄ (hK : IsCompact K) : μ K < ∞ :=
  IsFiniteMeasureOnCompacts.lt_top_of_isCompact hK


/-- A compact subset has finite measure for a measure which is finite on compacts. -/
theorem _root_.IsCompact.measure_ne_top [TopologicalSpace α] {μ : Measure α}
    [IsFiniteMeasureOnCompacts μ] ⦃K : Set α⦄ (hK : IsCompact K) : μ K ≠ ∞ :=
  hK.measure_lt_top.ne


/-- A bounded subset has finite measure for a measure which is finite on compact sets, in a
proper space. -/
theorem _root_.Bornology.IsBounded.measure_lt_top [PseudoMetricSpace α] [ProperSpace α]
    {μ : Measure α} [IsFiniteMeasureOnCompacts μ] ⦃s : Set α⦄ (hs : Bornology.IsBounded s) :
    μ s < ∞ :=
  calc
    μ s ≤ μ (closure s) := measure_mono subset_closure
    _ < ∞ := (Metric.isCompact_of_isClosed_isBounded isClosed_closure hs.closure).measure_lt_top


theorem measure_closedBall_lt_top [PseudoMetricSpace α] [ProperSpace α] {μ : Measure α}
    [IsFiniteMeasureOnCompacts μ] {x : α} {r : ℝ} : μ (Metric.closedBall x r) < ∞ :=
  Metric.isBounded_closedBall.measure_lt_top


theorem measure_ball_lt_top [PseudoMetricSpace α] [ProperSpace α] {μ : Measure α}
    [IsFiniteMeasureOnCompacts μ] {x : α} {r : ℝ} : μ (Metric.ball x r) < ∞ :=
  Metric.isBounded_ball.measure_lt_top


protected theorem IsFiniteMeasureOnCompacts.smul [TopologicalSpace α] (μ : Measure α)
    [IsFiniteMeasureOnCompacts μ] {c : ℝ≥0∞} (hc : c ≠ ∞) : IsFiniteMeasureOnCompacts (c • μ) :=
  ⟨fun _K hK => ENNReal.mul_lt_top hc.lt_top hK.measure_lt_top⟩


instance IsFiniteMeasureOnCompacts.smul_nnreal [TopologicalSpace α] (μ : Measure α)
    [IsFiniteMeasureOnCompacts μ] (c : ℝ≥0) : IsFiniteMeasureOnCompacts (c • μ) :=
  IsFiniteMeasureOnCompacts.smul μ coe_ne_top


instance instIsFiniteMeasureOnCompactsRestrict [TopologicalSpace α] {μ : Measure α}
    [IsFiniteMeasureOnCompacts μ] {s : Set α} : IsFiniteMeasureOnCompacts (μ.restrict s) :=
  ⟨fun _k hk ↦ (restrict_apply_le _ _).trans_lt hk.measure_lt_top⟩


instance (priority := 100) CompactSpace.isFiniteMeasure [TopologicalSpace α] [CompactSpace α]
    [IsFiniteMeasureOnCompacts μ] : IsFiniteMeasure μ :=
  ⟨IsFiniteMeasureOnCompacts.lt_top_of_isCompact isCompact_univ⟩


instance (priority := 100) SigmaFinite.of_isFiniteMeasureOnCompacts [TopologicalSpace α]
    [SigmaCompactSpace α] (μ : Measure α) [IsFiniteMeasureOnCompacts μ] : SigmaFinite μ :=
  ⟨⟨{   set := compactCovering α
        set_mem := fun _ => trivial
        finite := fun n => (isCompact_compactCovering α n).measure_lt_top
        spanning := iUnion_compactCovering α }⟩⟩

-- see Note [lower instance priority]

instance (priority := 100) sigmaFinite_of_locallyFinite [TopologicalSpace α]
    [SecondCountableTopology α] [IsLocallyFiniteMeasure μ] : SigmaFinite μ := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s t : Set α
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ MeasureTheory.SigmaFinite μ
  -/
  choose s hsx hsμ using μ.finiteAt_nhds
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t : Set α
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : α → Set α
    hsx : ∀ (x : α), Membership.mem (nhds x) (s x)
    hsμ : ∀ (x : α), LT.lt (μ (s x)) Top.top
    ⊢ MeasureTheory.SigmaFinite μ
  -/
  rcases TopologicalSpace.countable_cover_nhds hsx with ⟨t, htc, htU⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t✝ : Set α
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : α → Set α
    hsx : ∀ (x : α), Membership.mem (nhds x) (s x)
    hsμ : ∀ (x : α), LT.lt (μ (s x)) Top.top
    t : Set α
    htc : t.Countable
    htU : Eq (Set.iUnion fun x => Set.iUnion fun h => s x) Set.univ
    ⊢ MeasureTheory.SigmaFinite μ
  -/
  refine Measure.sigmaFinite_of_countable (htc.image s) (forall_mem_image.2 fun x _ => hsμ x) ?_
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    m0 : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    μ ν ν₁ ν₂ : MeasureTheory.Measure α
    s✝ t✝ : Set α
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : α → Set α
    hsx : ∀ (x : α), Membership.mem (nhds x) (s x)
    hsμ : ∀ (x : α), LT.lt (μ (s x)) Top.top
    t : Set α
    htc : t.Countable
    htU : Eq (Set.iUnion fun x => Set.iUnion fun h => s x) Set.univ
    ⊢ Eq (Set.image s t).sUnion Set.univ
  -/
  rwa [sUnion_image]
  /-
    🎉 no goals
  -/


/-- A measure which is finite on compact sets in a locally compact space is locally finite. -/
instance (priority := 100) isLocallyFiniteMeasure_of_isFiniteMeasureOnCompacts [TopologicalSpace α]
    [WeaklyLocallyCompactSpace α] [IsFiniteMeasureOnCompacts μ] : IsLocallyFiniteMeasure μ :=
  ⟨fun x ↦
    let ⟨K, K_compact, K_mem⟩ := exists_compact_mem_nhds x
    ⟨K, K_mem, K_compact.measure_lt_top⟩⟩


theorem exists_pos_measure_of_cover [Countable ι] {U : ι → Set α} (hU : ⋃ i, U i = univ)
    (hμ : μ ≠ 0) : ∃ i, 0 < μ (U i) := by
  /-
    α : Type u_1
    ι : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    U : ι → Set α
    hU : Eq (Set.iUnion fun i => U i) Set.univ
    hμ : Ne μ 0
    ⊢ Exists fun i => LT.lt 0 (μ (U i))
  -/
  contrapose! hμ with H
  /-
    α : Type u_1
    ι : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    U : ι → Set α
    hU : Eq (Set.iUnion fun i => U i) Set.univ
    H : ∀ (i : ι), LE.le (μ (U i)) 0
    ⊢ Eq μ 0
  -/
  rw [← measure_univ_eq_zero, ← hU]
  /-
    α : Type u_1
    ι : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    U : ι → Set α
    hU : Eq (Set.iUnion fun i => U i) Set.univ
    H : ∀ (i : ι), LE.le (μ (U i)) 0
    ⊢ Eq (μ (Set.iUnion fun i => U i)) 0
  -/
  exact measure_iUnion_null fun i => nonpos_iff_eq_zero.1 (H i)
  /-
    🎉 no goals
  -/


theorem exists_pos_preimage_ball [PseudoMetricSpace δ] (f : α → δ) (x : δ) (hμ : μ ≠ 0) :
    ∃ n : ℕ, 0 < μ (f ⁻¹' Metric.ball x n) :=
                                  /-
                                    α : Type u_1
                                    δ : Type u_3
                                    m0 : MeasurableSpace α
                                    μ : MeasureTheory.Measure α
                                    inst✝ : PseudoMetricSpace δ
                                    f : α → δ
                                    x : δ
                                    hμ : Ne μ 0
                                    ⊢ Eq (Set.iUnion fun i => Set.preimage f (Metric.ball x ↑i)) Set.univ
                                  -/
  exists_pos_measure_of_cover (by rw [← preimage_iUnion, Metric.iUnion_ball_nat, preimage_univ]) hμ
                                  /-
                                    🎉 no goals
                                  -/


theorem exists_pos_ball [PseudoMetricSpace α] (x : α) (hμ : μ ≠ 0) :
    ∃ n : ℕ, 0 < μ (Metric.ball x n) :=
  exists_pos_preimage_ball id x hμ


/-- If a set has zero measure in a neighborhood of each of its points, then it has zero measure
in a second-countable space. -/
@[deprecated (since := "2024-05-14")]
alias null_of_locally_null := measure_null_of_locally_null


theorem exists_ne_forall_mem_nhds_pos_measure_preimage {β} [TopologicalSpace β] [T1Space β]
    [SecondCountableTopology β] [Nonempty β] {f : α → β} (h : ∀ b, ∃ᵐ x ∂μ, f x ≠ b) :
    ∃ a b : β, a ≠ b ∧ (∀ s ∈ 𝓝 a, 0 < μ (f ⁻¹' s)) ∧ ∀ t ∈ 𝓝 b, 0 < μ (f ⁻¹' t) := by
  -- We use an `OuterMeasure` so that the proof works without `Measurable f`
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    h : ∀ (b : β), Filter.Frequently (fun x => Ne (f x) b) (MeasureTheory.ae μ)
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  set m : OuterMeasure β := OuterMeasure.map f μ.toOuterMeasure
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    h : ∀ (b : β), Filter.Frequently (fun x => Ne (f x) b) (MeasureTheory.ae μ)
    m : MeasureTheory.OuterMeasure β := (MeasureTheory.OuterMeasure.map f) μ.toOut …
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  replace h : ∀ b : β, m {b}ᶜ ≠ 0 := fun b => not_eventually.mpr (h b)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    m : MeasureTheory.OuterMeasure β := (MeasureTheory.OuterMeasure.map f) μ.toOut …
    h : ∀ (b : β), Ne (m (HasCompl.compl (Singleton.singleton b))) 0
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  inhabit β
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    m : MeasureTheory.OuterMeasure β := (MeasureTheory.OuterMeasure.map f) μ.toOut …
    h : ∀ (b : β), Ne (m (HasCompl.compl (Singleton.singleton b))) 0
    inhabited_h : Inhabited β
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  have : m univ ≠ 0 := ne_bot_of_le_ne_bot (h default) (measure_mono <| subset_univ _)
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    m : MeasureTheory.OuterMeasure β := (MeasureTheory.OuterMeasure.map f) μ.toOut …
    h : ∀ (b : β), Ne (m (HasCompl.compl (Singleton.singleton b))) 0
    inhabited_h : Inhabited β
    this : Ne (m Set.univ) 0
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  rcases exists_mem_forall_mem_nhdsWithin_pos_measure this with ⟨b, -, hb⟩
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    m : MeasureTheory.OuterMeasure β := (MeasureTheory.OuterMeasure.map f) μ.toOut …
    h : ∀ (b : β), Ne (m (HasCompl.compl (Singleton.singleton b))) 0
    inhabited_h : Inhabited β
    this : Ne (m Set.univ) 0
    b : β
    hb : ∀ (t : Set β), Membership.mem (nhdsWithin b Set.univ) t → LT.lt 0 (m t)
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  simp only [nhdsWithin_univ] at hb
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    m : MeasureTheory.OuterMeasure β := (MeasureTheory.OuterMeasure.map f) μ.toOut …
    h : ∀ (b : β), Ne (m (HasCompl.compl (Singleton.singleton b))) 0
    inhabited_h : Inhabited β
    this : Ne (m Set.univ) 0
    b : β
    hb : ∀ (t : Set β), Membership.mem (nhds b) t → LT.lt 0 (m t)
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  rcases exists_mem_forall_mem_nhdsWithin_pos_measure (h b) with ⟨a, hab : a ≠ b, ha⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    m : MeasureTheory.OuterMeasure β := (MeasureTheory.OuterMeasure.map f) μ.toOut …
    h : ∀ (b : β), Ne (m (HasCompl.compl (Singleton.singleton b))) 0
    inhabited_h : Inhabited β
    this : Ne (m Set.univ) 0
    b : β
    hb : ∀ (t : Set β), Membership.mem (nhds b) t → LT.lt 0 (m t)
    a : β
    hab : Ne a b
    ha : ∀ (t : Set β), Membership.mem (nhdsWithin a (HasCompl.compl (Singleton.si …
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  simp only [isOpen_compl_singleton.nhdsWithin_eq hab] at ha
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝³ : TopologicalSpace β
    inst✝² : T1Space β
    inst✝¹ : SecondCountableTopology β
    inst✝ : Nonempty β
    f : α → β
    m : MeasureTheory.OuterMeasure β := (MeasureTheory.OuterMeasure.map f) μ.toOut …
    h : ∀ (b : β), Ne (m (HasCompl.compl (Singleton.singleton b))) 0
    inhabited_h : Inhabited β
    this : Ne (m Set.univ) 0
    b : β
    hb : ∀ (t : Set β), Membership.mem (nhds b) t → LT.lt 0 (m t)
    a : β
    hab : Ne a b
    ha : ∀ (t : Set β), Membership.mem (nhds a) t → LT.lt 0 (m t)
    ⊢ Exists fun a => Exists fun b => And (Ne a b) (And (∀ (s : Set β), Membership …
  -/
  exact ⟨a, b, hab, ha, hb⟩
  /-
    🎉 no goals
  -/


/-- If two finite measures give the same mass to the whole space and coincide on a π-system made
of measurable sets, then they coincide on all sets in the σ-algebra generated by the π-system. -/
theorem ext_on_measurableSpace_of_generate_finite {α} (m₀ : MeasurableSpace α) {μ ν : Measure α}
    [IsFiniteMeasure μ] (C : Set (Set α)) (hμν : ∀ s ∈ C, μ s = ν s) {m : MeasurableSpace α}
    (h : m ≤ m₀) (hA : m = MeasurableSpace.generateFrom C) (hC : IsPiSystem C)
    (h_univ : μ Set.univ = ν Set.univ) {s : Set α} (hs : MeasurableSet[m] s) : μ s = ν s := by
  haveI : IsFiniteMeasure ν := by
    constructor
    rw [← h_univ]
    apply IsFiniteMeasure.measure_univ_lt_top
  induction s, hs using induction_on_inter hA hC with
  | empty => simp
  | basic t ht => exact hμν t ht
  | compl t htm iht =>
    rw [measure_compl (h t htm) (measure_ne_top _ _), measure_compl (h t htm) (measure_ne_top _ _),
      iht, h_univ]
  | iUnion f hfd hfm ihf =>
    simp [measure_iUnion, hfd, h _ (hfm _), ihf]


/-- Two finite measures are equal if they are equal on the π-system generating the σ-algebra
  (and `univ`). -/
theorem ext_of_generate_finite (C : Set (Set α)) (hA : m0 = generateFrom C) (hC : IsPiSystem C)
    [IsFiniteMeasure μ] (hμν : ∀ s ∈ C, μ s = ν s) (h_univ : μ univ = ν univ) : μ = ν :=
  Measure.ext fun _s hs =>
    ext_on_measurableSpace_of_generate_finite m0 C hμν le_rfl hA hC h_univ hs


/-- Given `S : μ.FiniteSpanningSetsIn {s | MeasurableSet s}`,
`FiniteSpanningSetsIn.disjointed` provides a `FiniteSpanningSetsIn {s | MeasurableSet s}`
such that its underlying sets are pairwise disjoint. -/
protected def FiniteSpanningSetsIn.disjointed {μ : Measure α}
    (S : μ.FiniteSpanningSetsIn { s | MeasurableSet s }) :
    μ.FiniteSpanningSetsIn { s | MeasurableSet s } :=
  ⟨disjointed S.set, MeasurableSet.disjointed S.set_mem, fun n =>
    lt_of_le_of_lt (measure_mono (disjointed_subset S.set n)) (S.finite _),
    S.spanning ▸ iUnion_disjointed⟩


theorem FiniteSpanningSetsIn.disjointed_set_eq {μ : Measure α}
    (S : μ.FiniteSpanningSetsIn { s | MeasurableSet s }) : S.disjointed.set = disjointed S.set :=
  rfl


theorem exists_eq_disjoint_finiteSpanningSetsIn (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν] :
    ∃ (S : μ.FiniteSpanningSetsIn { s | MeasurableSet s })
      (T : ν.FiniteSpanningSetsIn { s | MeasurableSet s }),
      S.set = T.set ∧ Pairwise (Disjoint on S.set) :=
  let S := (μ + ν).toFiniteSpanningSetsIn.disjointed
  ⟨S.ofLE (Measure.le_add_right le_rfl), S.ofLE (Measure.le_add_left le_rfl), rfl,
    disjoint_disjointed _⟩


theorem filter_mono (h : f ≤ g) : μ.FiniteAtFilter g → μ.FiniteAtFilter f := fun ⟨s, hs, hμ⟩ =>
  ⟨s, h hs, hμ⟩


theorem inf_of_left (h : μ.FiniteAtFilter f) : μ.FiniteAtFilter (f ⊓ g) :=
  h.filter_mono inf_le_left


theorem inf_of_right (h : μ.FiniteAtFilter g) : μ.FiniteAtFilter (f ⊓ g) :=
  h.filter_mono inf_le_right


@[simp]
theorem inf_ae_iff : μ.FiniteAtFilter (f ⊓ ae μ) ↔ μ.FiniteAtFilter f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Filter α
    ⊢ Iff (μ.FiniteAtFilter (Min.min f (MeasureTheory.ae μ))) (μ.FiniteAtFilter f)
  -/
  refine ⟨?_, fun h => h.filter_mono inf_le_left⟩
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Filter α
    ⊢ μ.FiniteAtFilter (Min.min f (MeasureTheory.ae μ)) → μ.FiniteAtFilter f
  -/
  rintro ⟨s, ⟨t, ht, u, hu, rfl⟩, hμ⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Filter α
    t : Set α
    ht : Membership.mem f t
    u : Set α
    hu : Membership.mem (MeasureTheory.ae μ) u
    hμ : LT.lt (μ (Inter.inter t u)) Top.top
    ⊢ μ.FiniteAtFilter f
  -/
  suffices μ t ≤ μ (t ∩ u) from ⟨t, ht, this.trans_lt hμ⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Filter α
    t : Set α
    ht : Membership.mem f t
    u : Set α
    hu : Membership.mem (MeasureTheory.ae μ) u
    hμ : LT.lt (μ (Inter.inter t u)) Top.top
    ⊢ LE.le (μ t) (μ (Inter.inter t u))
  -/
  exact measure_mono_ae (mem_of_superset hu fun x hu ht => ⟨ht, hu⟩)
  /-
    🎉 no goals
  -/


alias ⟨of_inf_ae, _⟩ := inf_ae_iff


theorem filter_mono_ae (h : f ⊓ (ae μ) ≤ g) (hg : μ.FiniteAtFilter g) : μ.FiniteAtFilter f :=
  inf_ae_iff.1 (hg.filter_mono h)


protected theorem measure_mono (h : μ ≤ ν) : ν.FiniteAtFilter f → μ.FiniteAtFilter f :=
  fun ⟨s, hs, hν⟩ => ⟨s, hs, (Measure.le_iff'.1 h s).trans_lt hν⟩


@[mono]
protected theorem mono (hf : f ≤ g) (hμ : μ ≤ ν) : ν.FiniteAtFilter g → μ.FiniteAtFilter f :=
  fun h => (h.filter_mono hf).measure_mono hμ


protected theorem eventually (h : μ.FiniteAtFilter f) : ∀ᶠ s in f.smallSets, μ s < ∞ :=
  (eventually_smallSets' fun _s _t hst ht => (measure_mono hst).trans_lt ht).2 h


theorem filterSup : μ.FiniteAtFilter f → μ.FiniteAtFilter g → μ.FiniteAtFilter (f ⊔ g) :=
  fun ⟨s, hsf, hsμ⟩ ⟨t, htg, htμ⟩ =>
  ⟨s ∪ t, union_mem_sup hsf htg, (measure_union_le s t).trans_lt (ENNReal.add_lt_top.2 ⟨hsμ, htμ⟩)⟩


theorem finiteAt_nhdsWithin [TopologicalSpace α] {_m0 : MeasurableSpace α} (μ : Measure α)
    [IsLocallyFiniteMeasure μ] (x : α) (s : Set α) : μ.FiniteAtFilter (𝓝[s] x) :=
  (finiteAt_nhds μ x).inf_of_left


@[simp]
theorem finiteAt_principal : μ.FiniteAtFilter (𝓟 s) ↔ μ s < ∞ :=
  ⟨fun ⟨_t, ht, hμ⟩ => (measure_mono ht).trans_lt hμ, fun h => ⟨s, mem_principal_self s, h⟩⟩


theorem isLocallyFiniteMeasure_of_le [TopologicalSpace α] {_m : MeasurableSpace α} {μ ν : Measure α}
    [H : IsLocallyFiniteMeasure μ] (h : ν ≤ μ) : IsLocallyFiniteMeasure ν :=
  let F := H.finiteAtNhds
  ⟨fun x => (F x).measure_mono h⟩


/-- If `s` is a compact set and `μ` is finite at `𝓝 x` for every `x ∈ s`, then `s` admits an open
superset of finite measure. -/
theorem exists_open_superset_measure_lt_top' (h : IsCompact s)
    (hμ : ∀ x ∈ s, μ.FiniteAtFilter (𝓝 x)) : ∃ U ⊇ s, IsOpen U ∧ μ U < ∞ := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    h : IsCompact s
    hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhds x)
    ⊢ Exists fun U => And (Superset U s) (And (IsOpen U) (LT.lt (μ U) Top.top))
  -/
  refine IsCompact.induction_on h ?_ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      h : IsCompact s
      hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhds x)
      ⊢ Exists fun U => And (Superset U EmptyCollection.emptyCollection) (And (IsOpe …
    -/
  · use ∅
    /-
      case h
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      h : IsCompact s
      hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhds x)
      ⊢ And (Superset EmptyCollection.emptyCollection EmptyCollection.emptyCollectio …
    -/
    simp [Superset]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      h : IsCompact s
      hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhds x)
      ⊢ ∀ ⦃s t : Set α⦄, HasSubset.Subset s t → (Exists fun U => And (Superset U t)  …
    -/
  · rintro s t hst ⟨U, htU, hUo, hU⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s✝ : Set α
      h : IsCompact s✝
      hμ : ∀ (x : α), Membership.mem s✝ x → μ.FiniteAtFilter (nhds x)
      s t : Set α
      hst : HasSubset.Subset s t
      U : Set α
      htU : Superset U t
      hUo : IsOpen U
      hU : LT.lt (μ U) Top.top
      ⊢ Exists fun U => And (Superset U s) (And (IsOpen U) (LT.lt (μ U) Top.top))
    -/
    exact ⟨U, hst.trans htU, hUo, hU⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      h : IsCompact s
      hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhds x)
      ⊢ ∀ ⦃s t : Set α⦄, (Exists fun U => And (Superset U s) (And (IsOpen U) (LT.lt  …
    -/
  · rintro s t ⟨U, hsU, hUo, hU⟩ ⟨V, htV, hVo, hV⟩
    refine
      ⟨U ∪ V, union_subset_union hsU htV, hUo.union hVo,
        (measure_union_le _ _).trans_lt <| ENNReal.add_lt_top.2 ⟨hU, hV⟩⟩
    /-
      case refine_4
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      h : IsCompact s
      hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhds x)
      ⊢ ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhdsWit …
    -/
  · intro x hx
    /-
      case refine_4
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      h : IsCompact s
      hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhds x)
      x : α
      hx : Membership.mem s x
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Exists fun U => And …
    -/
    rcases (hμ x hx).exists_mem_basis (nhds_basis_opens _) with ⟨U, ⟨hx, hUo⟩, hU⟩
    /-
      case refine_4.intro.intro.intro
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      h : IsCompact s
      hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhds x)
      x : α
      hx✝ : Membership.mem s x
      U : Set α
      hU : LT.lt (μ U) Top.top
      hx : Membership.mem U x
      hUo : IsOpen U
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Exists fun U => And …
    -/
    exact ⟨U, nhdsWithin_le_nhds (hUo.mem_nhds hx), U, Subset.rfl, hUo, hU⟩
    /-
      🎉 no goals
    -/


/-- If `s` is a compact set and `μ` is a locally finite measure, then `s` admits an open superset of
finite measure. -/
theorem exists_open_superset_measure_lt_top (h : IsCompact s) (μ : Measure α)
    [IsLocallyFiniteMeasure μ] : ∃ U ⊇ s, IsOpen U ∧ μ U < ∞ :=
  h.exists_open_superset_measure_lt_top' fun x _ => μ.finiteAt_nhds x


theorem measure_lt_top_of_nhdsWithin (h : IsCompact s) (hμ : ∀ x ∈ s, μ.FiniteAtFilter (𝓝[s] x)) :
    μ s < ∞ :=
                               /-
                                 α : Type u_1
                                 inst✝¹ : TopologicalSpace α
                                 inst✝ : MeasurableSpace α
                                 μ : MeasureTheory.Measure α
                                 s : Set α
                                 h : IsCompact s
                                 hμ : ∀ (x : α), Membership.mem s x → μ.FiniteAtFilter (nhdsWithin x s)
                                 ⊢ LT.lt (μ EmptyCollection.emptyCollection) Top.top
                               -/
  IsCompact.induction_on h (by simp) (fun _ _ hst ht => (measure_mono hst).trans_lt ht)
                               /-
                                 🎉 no goals
                               -/
    (fun s t hs ht => (measure_union_le s t).trans_lt (ENNReal.add_lt_top.2 ⟨hs, ht⟩)) hμ


theorem measure_zero_of_nhdsWithin (hs : IsCompact s) :
    (∀ a ∈ s, ∃ t ∈ 𝓝[s] a, μ t = 0) → μ s = 0 := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : IsCompact s
    ⊢ (∀ (a : α), Membership.mem s a → Exists fun t => And (Membership.mem (nhdsWi …
  -/
  simpa only [← compl_mem_ae_iff] using hs.compl_mem_sets_of_nhdsWithin
  /-
    🎉 no goals
  -/


instance (priority := 100) isFiniteMeasureOnCompacts_of_isLocallyFiniteMeasure [TopologicalSpace α]
    {_ : MeasurableSpace α} {μ : Measure α} [IsLocallyFiniteMeasure μ] :
    IsFiniteMeasureOnCompacts μ :=
  ⟨fun _s hs => hs.measure_lt_top_of_nhdsWithin fun _ _ => μ.finiteAt_nhdsWithin _ _⟩


theorem isFiniteMeasure_iff_isFiniteMeasureOnCompacts_of_compactSpace [TopologicalSpace α]
    [MeasurableSpace α] {μ : Measure α} [CompactSpace α] :
    IsFiniteMeasure μ ↔ IsFiniteMeasureOnCompacts μ := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompactSpace α
    ⊢ Iff (MeasureTheory.IsFiniteMeasure μ) (MeasureTheory.IsFiniteMeasureOnCompac …
  -/
  constructor <;> intros
    /-
      case mp
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompactSpace α
      a✝ : MeasureTheory.IsFiniteMeasure μ
      ⊢ MeasureTheory.IsFiniteMeasureOnCompacts μ
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompactSpace α
      a✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      ⊢ MeasureTheory.IsFiniteMeasure μ
    -/
  · exact CompactSpace.isFiniteMeasure
    /-
      🎉 no goals
    -/


/-- Compact covering of a `σ`-compact topological space as
`MeasureTheory.Measure.FiniteSpanningSetsIn`. -/
def MeasureTheory.Measure.finiteSpanningSetsInCompact [TopologicalSpace α] [SigmaCompactSpace α]
    {_ : MeasurableSpace α} (μ : Measure α) [IsLocallyFiniteMeasure μ] :
    μ.FiniteSpanningSetsIn { K | IsCompact K } where
  set := compactCovering α
  set_mem := isCompact_compactCovering α
  finite n := (isCompact_compactCovering α n).measure_lt_top
  spanning := iUnion_compactCovering α


/-- A locally finite measure on a `σ`-compact topological space admits a finite spanning sequence
of open sets. -/
def MeasureTheory.Measure.finiteSpanningSetsInOpen [TopologicalSpace α] [SigmaCompactSpace α]
    {_ : MeasurableSpace α} (μ : Measure α) [IsLocallyFiniteMeasure μ] :
    μ.FiniteSpanningSetsIn { K | IsOpen K } where
  set n := ((isCompact_compactCovering α n).exists_open_superset_measure_lt_top μ).choose
  set_mem n :=
    ((isCompact_compactCovering α n).exists_open_superset_measure_lt_top μ).choose_spec.2.1
  finite n :=
    ((isCompact_compactCovering α n).exists_open_superset_measure_lt_top μ).choose_spec.2.2
  spanning :=
    eq_univ_of_subset
      (iUnion_mono fun n =>
        ((isCompact_compactCovering α n).exists_open_superset_measure_lt_top μ).choose_spec.1)
      (iUnion_compactCovering α)


/-- A locally finite measure on a second countable topological space admits a finite spanning
sequence of open sets. -/
noncomputable irreducible_def MeasureTheory.Measure.finiteSpanningSetsInOpen' [TopologicalSpace α]
  [SecondCountableTopology α] {m : MeasurableSpace α} (μ : Measure α) [IsLocallyFiniteMeasure μ] :
  μ.FiniteSpanningSetsIn { K | IsOpen K } := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K)
  -/
  suffices H : Nonempty (μ.FiniteSpanningSetsIn { K | IsOpen K }) from H.some
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ Nonempty (μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K))
  -/
  cases isEmpty_or_nonempty α
  · exact
      ⟨{  set := fun _ => ∅
          set_mem := fun _ => by simp
          finite := fun _ => by simp
          spanning := by simp [eq_iff_true_of_subsingleton] }⟩
  /-
    case inr
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    h✝ : Nonempty α
    ⊢ Nonempty (μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K))
  -/
  inhabit α
  /-
    case inr
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    ⊢ Nonempty (μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K))
  -/
  let S : Set (Set α) := { s | IsOpen s ∧ μ s < ∞ }
  obtain ⟨T, T_count, TS, hT⟩ : ∃ T : Set (Set α), T.Countable ∧ T ⊆ S ∧ ⋃₀ T = ⋃₀ S :=
    isOpen_sUnion_countable S fun s hs => hs.1
  /-
    case inr.intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    S : Set (Set α) := setOf fun s => And (IsOpen s) (LT.lt (μ s) Top.top)
    T : Set (Set α)
    T_count : T.Countable
    TS : HasSubset.Subset T S
    hT : Eq T.sUnion S.sUnion
    ⊢ Nonempty (μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K))
  -/
  rw [μ.isTopologicalBasis_isOpen_lt_top.sUnion_eq] at hT
  have T_ne : T.Nonempty := by
    by_contra h'T
    rw [not_nonempty_iff_eq_empty.1 h'T, sUnion_empty] at hT
    simpa only [← hT] using mem_univ (default : α)
  /-
    case inr.intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    S : Set (Set α) := setOf fun s => And (IsOpen s) (LT.lt (μ s) Top.top)
    T : Set (Set α)
    T_count : T.Countable
    TS : HasSubset.Subset T S
    hT : Eq T.sUnion Set.univ
    T_ne : T.Nonempty
    ⊢ Nonempty (μ.FiniteSpanningSetsIn (setOf fun K => IsOpen K))
  -/
  obtain ⟨f, hf⟩ : ∃ f : ℕ → Set α, T = range f := T_count.exists_eq_range T_ne
  have fS : ∀ n, f n ∈ S := by
    intro n
    apply TS
    rw [hf]
    exact mem_range_self n
  refine
    ⟨{  set := f
        set_mem := fun n => (fS n).1
        finite := fun n => (fS n).2
        spanning := ?_ }⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    S : Set (Set α) := setOf fun s => And (IsOpen s) (LT.lt (μ s) Top.top)
    T : Set (Set α)
    T_count : T.Countable
    TS : HasSubset.Subset T S
    hT : Eq T.sUnion Set.univ
    T_ne : T.Nonempty
    f : Nat → Set α
    hf : Eq T (Set.range f)
    fS : ∀ (n : Nat), Membership.mem S (f n)
    ⊢ Eq (Set.iUnion fun i => f i) Set.univ
  -/
  refine eq_univ_of_forall fun x => ?_
  obtain ⟨t, tT, xt⟩ : ∃ t : Set α, t ∈ range f ∧ x ∈ t := by
    have : x ∈ ⋃₀ T := by simp only [hT, mem_univ]
    simpa only [mem_sUnion, exists_prop, ← hf]
  /-
    case inr.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    S : Set (Set α) := setOf fun s => And (IsOpen s) (LT.lt (μ s) Top.top)
    T : Set (Set α)
    T_count : T.Countable
    TS : HasSubset.Subset T S
    hT : Eq T.sUnion Set.univ
    T_ne : T.Nonempty
    f : Nat → Set α
    hf : Eq T (Set.range f)
    fS : ∀ (n : Nat), Membership.mem S (f n)
    x : α
    t : Set α
    tT : Membership.mem (Set.range f) t
    xt : Membership.mem t x
    ⊢ Membership.mem (Set.iUnion fun i => f i) x
  -/
  obtain ⟨n, rfl⟩ : ∃ n : ℕ, f n = t := by simpa only using tT
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    δ : Type u_3
    ι : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : SecondCountableTopology α
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    h✝ : Nonempty α
    inhabited_h : Inhabited α
    S : Set (Set α) := setOf fun s => And (IsOpen s) (LT.lt (μ s) Top.top)
    T : Set (Set α)
    T_count : T.Countable
    TS : HasSubset.Subset T S
    hT : Eq T.sUnion Set.univ
    T_ne : T.Nonempty
    f : Nat → Set α
    hf : Eq T (Set.range f)
    fS : ∀ (n : Nat), Membership.mem S (f n)
    x : α
    n : Nat
    tT : Membership.mem (Set.range f) (f n)
    xt : Membership.mem (f n) x
    ⊢ Membership.mem (Set.iUnion fun i => f i) x
  -/
  exact mem_iUnion_of_mem _ xt
  /-
    🎉 no goals
  -/


theorem measure_Icc_lt_top : μ (Icc a b) < ∞ :=
  isCompact_Icc.measure_lt_top


theorem measure_Ico_lt_top : μ (Ico a b) < ∞ :=
  (measure_mono Ico_subset_Icc_self).trans_lt measure_Icc_lt_top


theorem measure_Ioc_lt_top : μ (Ioc a b) < ∞ :=
  (measure_mono Ioc_subset_Icc_self).trans_lt measure_Icc_lt_top


theorem measure_Ioo_lt_top : μ (Ioo a b) < ∞ :=
  (measure_mono Ioo_subset_Icc_self).trans_lt measure_Icc_lt_top


