/-- A stopping time with respect to some filtration `f` is a function
`τ` such that for all `i`, the preimage of `{j | j ≤ i}` along `τ` is measurable
with respect to `f i`.

Intuitively, the stopping time `τ` describes some stopping rule such that at time
`i`, we may determine it with the information we have at time `i`. -/
def IsStoppingTime [Preorder ι] (f : Filtration ι m) (τ : Ω → ι) :=
  ∀ i : ι, MeasurableSet[f i] <| {ω | τ ω ≤ i}


theorem isStoppingTime_const [Preorder ι] (f : Filtration ι m) (i : ι) :
                                               /-
                                                 Ω : Type u_1
                                                 ι : Type u_3
                                                 m : MeasurableSpace Ω
                                                 inst✝ : Preorder ι
                                                 f : MeasureTheory.Filtration ι m
                                                 i j : ι
                                                 ⊢ MeasurableSet (setOf fun ω => LE.le ((fun x => i) ω) j)
                                               -/
    IsStoppingTime f fun _ => i := fun j => by simp only [MeasurableSet.const]
                                               /-
                                                 🎉 no goals
                                               -/


protected theorem IsStoppingTime.measurableSet_le (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[f i] {ω | τ ω ≤ i} :=
  hτ i


theorem IsStoppingTime.measurableSet_lt_of_pred [PredOrder ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[f i] {ω | τ ω < i} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : PredOrder ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  by_cases hi_min : IsMin i
    /-
      case pos
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : PredOrder ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      hi_min : IsMin i
      ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
    -/
  · suffices {ω : Ω | τ ω < i} = ∅ by rw [this]; exact @MeasurableSet.empty _ (f i)
    /-
      case pos
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : PredOrder ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      hi_min : IsMin i
      ⊢ Eq (setOf fun ω => LT.lt (τ ω) i) EmptyCollection.emptyCollection
    -/
    ext1 ω
    /-
      case pos.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : PredOrder ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      hi_min : IsMin i
      ω : Ω
      ⊢ Iff (Membership.mem (setOf fun ω => LT.lt (τ ω) i) ω) (Membership.mem EmptyC …
    -/
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    /-
      case pos.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : PredOrder ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      hi_min : IsMin i
      ω : Ω
      ⊢ Not (LT.lt (τ ω) i)
    -/
    rw [isMin_iff_forall_not_lt] at hi_min
    /-
      case pos.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : PredOrder ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      hi_min : ∀ (b : ι), Not (LT.lt b i)
      ω : Ω
      ⊢ Not (LT.lt (τ ω) i)
    -/
    exact hi_min (τ ω)
    /-
      🎉 no goals
    -/
  /-
    case neg
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : PredOrder ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    hi_min : Not (IsMin i)
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  have : {ω : Ω | τ ω < i} = τ ⁻¹' Set.Iic (pred i) := by ext; simp [Iic_pred_of_not_isMin hi_min]
  /-
    case neg
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : PredOrder ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    hi_min : Not (IsMin i)
    this : Eq (setOf fun ω => LT.lt (τ ω) i) (Set.preimage τ (Set.Iic (Order.pred  …
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  rw [this]
  /-
    case neg
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : PredOrder ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    hi_min : Not (IsMin i)
    this : Eq (setOf fun ω => LT.lt (τ ω) i) (Set.preimage τ (Set.Iic (Order.pred  …
    ⊢ MeasurableSet (Set.preimage τ (Set.Iic (Order.pred i)))
  -/
  exact f.mono (pred_le i) _ (hτ.measurableSet_le <| pred i)
  /-
    🎉 no goals
  -/


protected theorem measurableSet_eq_of_countable_range (hτ : IsStoppingTime f τ)
    (h_countable : (Set.range τ).Countable) (i : ι) : MeasurableSet[f i] {ω | τ ω = i} := by
  have : {ω | τ ω = i} = {ω | τ ω ≤ i} \ ⋃ (j ∈ Set.range τ) (_ : j < i), {ω | τ ω ≤ j} := by
    ext1 a
    simp only [Set.mem_setOf_eq, Set.mem_range, Set.iUnion_exists, Set.iUnion_iUnion_eq',
      Set.mem_diff, Set.mem_iUnion, exists_prop, not_exists, not_and, not_le]
    constructor <;> intro h
    · simp only [h, lt_iff_le_not_le, le_refl, and_imp, imp_self, imp_true_iff, and_self_iff]
    · exact h.1.eq_or_lt.resolve_right fun h_lt => h.2 a h_lt le_rfl
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : PartialOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => Eq (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ ω) …
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) i)
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : PartialOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => Eq (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ ω) …
    ⊢ MeasurableSet (SDiff.sdiff (setOf fun ω => LE.le (τ ω) i) (Set.iUnion fun j  …
  -/
  refine (hτ.measurableSet_le i).diff ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : PartialOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => Eq (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ ω) …
    ⊢ MeasurableSet (Set.iUnion fun j => Set.iUnion fun h => Set.iUnion fun x => s …
  -/
  refine MeasurableSet.biUnion h_countable fun j _ => ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : PartialOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => Eq (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ ω) …
    j : ι
    x✝ : Membership.mem (Set.range τ) j
    ⊢ MeasurableSet (Set.iUnion fun x => setOf fun ω => LE.le (τ ω) j)
  -/
  rw [Set.iUnion_eq_if]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : PartialOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => Eq (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ ω) …
    j : ι
    x✝ : Membership.mem (Set.range τ) j
    ⊢ MeasurableSet (ite (LT.lt j i) (setOf fun ω => LE.le (τ ω) j) EmptyCollectio …
  -/
  split_ifs with hji
    /-
      case pos
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : PartialOrder ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      hτ : MeasureTheory.IsStoppingTime f τ
      h_countable : (Set.range τ).Countable
      i : ι
      this : Eq (setOf fun ω => Eq (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ ω) …
      j : ι
      x✝ : Membership.mem (Set.range τ) j
      hji : LT.lt j i
      ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) j)
    -/
  · exact f.mono hji.le _ (hτ.measurableSet_le j)
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : PartialOrder ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      hτ : MeasureTheory.IsStoppingTime f τ
      h_countable : (Set.range τ).Countable
      i : ι
      this : Eq (setOf fun ω => Eq (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ ω) …
      j : ι
      x✝ : Membership.mem (Set.range τ) j
      hji : Not (LT.lt j i)
      ⊢ MeasurableSet EmptyCollection.emptyCollection
    -/
  · exact @MeasurableSet.empty _ (f i)
    /-
      🎉 no goals
    -/


protected theorem measurableSet_eq_of_countable [Countable ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[f i] {ω | τ ω = i} :=
  hτ.measurableSet_eq_of_countable_range (Set.to_countable _) i


protected theorem measurableSet_lt_of_countable_range (hτ : IsStoppingTime f τ)
    (h_countable : (Set.range τ).Countable) (i : ι) : MeasurableSet[f i] {ω | τ ω < i} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : PartialOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  have : {ω | τ ω < i} = {ω | τ ω ≤ i} \ {ω | τ ω = i} := by ext1 ω; simp [lt_iff_le_and_ne]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : PartialOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => LT.lt (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ …
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : PartialOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => LT.lt (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ …
    ⊢ MeasurableSet (SDiff.sdiff (setOf fun ω => LE.le (τ ω) i) (setOf fun ω => Eq …
  -/
  exact (hτ.measurableSet_le i).diff (hτ.measurableSet_eq_of_countable_range h_countable i)
  /-
    🎉 no goals
  -/


protected theorem measurableSet_lt_of_countable [Countable ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[f i] {ω | τ ω < i} :=
  hτ.measurableSet_lt_of_countable_range (Set.to_countable _) i


protected theorem measurableSet_ge_of_countable_range {ι} [LinearOrder ι] {τ : Ω → ι}
    {f : Filtration ι m} (hτ : IsStoppingTime f τ) (h_countable : (Set.range τ).Countable) (i : ι) :
    MeasurableSet[f i] {ω | i ≤ τ ω} := by
  have : {ω | i ≤ τ ω} = {ω | τ ω < i}ᶜ := by
    ext1 ω; simp only [Set.mem_setOf_eq, Set.mem_compl_iff, not_lt]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    ι : Type u_4
    inst✝ : LinearOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => LE.le i (τ ω)) (HasCompl.compl (setOf fun ω => LT.lt …
    ⊢ MeasurableSet (setOf fun ω => LE.le i (τ ω))
  -/
  rw [this]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    ι : Type u_4
    inst✝ : LinearOrder ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => LE.le i (τ ω)) (HasCompl.compl (setOf fun ω => LT.lt …
    ⊢ MeasurableSet (HasCompl.compl (setOf fun ω => LT.lt (τ ω) i))
  -/
  exact (hτ.measurableSet_lt_of_countable_range h_countable i).compl
  /-
    🎉 no goals
  -/


protected theorem measurableSet_ge_of_countable {ι} [LinearOrder ι] {τ : Ω → ι} {f : Filtration ι m}
    [Countable ι] (hτ : IsStoppingTime f τ) (i : ι) : MeasurableSet[f i] {ω | i ≤ τ ω} :=
  hτ.measurableSet_ge_of_countable_range (Set.to_countable _) i


theorem IsStoppingTime.measurableSet_gt (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[f i] {ω | i < τ ω} := by
  have : {ω | i < τ ω} = {ω | τ ω ≤ i}ᶜ := by
    ext1 ω; simp only [Set.mem_setOf_eq, Set.mem_compl_iff, not_le]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LT.lt i (τ ω)) (HasCompl.compl (setOf fun ω => LE.le …
    ⊢ MeasurableSet (setOf fun ω => LT.lt i (τ ω))
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LT.lt i (τ ω)) (HasCompl.compl (setOf fun ω => LE.le …
    ⊢ MeasurableSet (HasCompl.compl (setOf fun ω => LE.le (τ ω) i))
  -/
  exact (hτ.measurableSet_le i).compl
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `MeasureTheory.IsStoppingTime.measurableSet_lt`. -/
theorem IsStoppingTime.measurableSet_lt_of_isLUB (hτ : IsStoppingTime f τ) (i : ι)
    (h_lub : IsLUB (Set.Iio i) i) : MeasurableSet[f i] {ω | τ ω < i} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    h_lub : IsLUB (Set.Iio i) i
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  by_cases hi_min : IsMin i
    /-
      case pos
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      h_lub : IsLUB (Set.Iio i) i
      hi_min : IsMin i
      ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
    -/
  · suffices {ω | τ ω < i} = ∅ by rw [this]; exact @MeasurableSet.empty _ (f i)
    /-
      case pos
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      h_lub : IsLUB (Set.Iio i) i
      hi_min : IsMin i
      ⊢ Eq (setOf fun ω => LT.lt (τ ω) i) EmptyCollection.emptyCollection
    -/
    ext1 ω
    /-
      case pos.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      h_lub : IsLUB (Set.Iio i) i
      hi_min : IsMin i
      ω : Ω
      ⊢ Iff (Membership.mem (setOf fun ω => LT.lt (τ ω) i) ω) (Membership.mem EmptyC …
    -/
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    /-
      case pos.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      h_lub : IsLUB (Set.Iio i) i
      hi_min : IsMin i
      ω : Ω
      ⊢ Not (LT.lt (τ ω) i)
    -/
    exact isMin_iff_forall_not_lt.mp hi_min (τ ω)
    /-
      🎉 no goals
    -/
  obtain ⟨seq, -, -, h_tendsto, h_bound⟩ :
      ∃ seq : ℕ → ι, Monotone seq ∧ (∀ j, seq j ≤ i) ∧ Tendsto seq atTop (𝓝 i) ∧ ∀ j, seq j < i :=
    h_lub.exists_seq_monotone_tendsto (not_isMin_iff.mp hi_min)
  have h_Ioi_eq_Union : Set.Iio i = ⋃ j, {k | k ≤ seq j} := by
    ext1 k
    simp only [Set.mem_Iio, Set.mem_iUnion, Set.mem_setOf_eq]
    refine ⟨fun hk_lt_i => ?_, fun h_exists_k_le_seq => ?_⟩
    · rw [tendsto_atTop'] at h_tendsto
      have h_nhds : Set.Ici k ∈ 𝓝 i :=
        mem_nhds_iff.mpr ⟨Set.Ioi k, Set.Ioi_subset_Ici le_rfl, isOpen_Ioi, hk_lt_i⟩
      obtain ⟨a, ha⟩ : ∃ a : ℕ, ∀ b : ℕ, b ≥ a → k ≤ seq b := h_tendsto (Set.Ici k) h_nhds
      exact ⟨a, ha a le_rfl⟩
    · obtain ⟨j, hk_seq_j⟩ := h_exists_k_le_seq
      exact hk_seq_j.trans_lt (h_bound j)
  have h_lt_eq_preimage : {ω | τ ω < i} = τ ⁻¹' Set.Iio i := by
    ext1 ω; simp only [Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Iio]
  /-
    case neg.intro.intro.intro.intro
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    h_lub : IsLUB (Set.Iio i) i
    hi_min : Not (IsMin i)
    seq : Nat → ι
    h_tendsto : Filter.Tendsto seq Filter.atTop (nhds i)
    h_bound : ∀ (j : Nat), LT.lt (seq j) i
    h_Ioi_eq_Union : Eq (Set.Iio i) (Set.iUnion fun j => setOf fun k => LE.le k (s …
    h_lt_eq_preimage : Eq (setOf fun ω => LT.lt (τ ω) i) (Set.preimage τ (Set.Iio  …
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  rw [h_lt_eq_preimage, h_Ioi_eq_Union]
  /-
    case neg.intro.intro.intro.intro
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    h_lub : IsLUB (Set.Iio i) i
    hi_min : Not (IsMin i)
    seq : Nat → ι
    h_tendsto : Filter.Tendsto seq Filter.atTop (nhds i)
    h_bound : ∀ (j : Nat), LT.lt (seq j) i
    h_Ioi_eq_Union : Eq (Set.Iio i) (Set.iUnion fun j => setOf fun k => LE.le k (s …
    h_lt_eq_preimage : Eq (setOf fun ω => LT.lt (τ ω) i) (Set.preimage τ (Set.Iio  …
    ⊢ MeasurableSet (Set.preimage τ (Set.iUnion fun j => setOf fun k => LE.le k (s …
  -/
  simp only [Set.preimage_iUnion, Set.preimage_setOf_eq]
  /-
    case neg.intro.intro.intro.intro
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    h_lub : IsLUB (Set.Iio i) i
    hi_min : Not (IsMin i)
    seq : Nat → ι
    h_tendsto : Filter.Tendsto seq Filter.atTop (nhds i)
    h_bound : ∀ (j : Nat), LT.lt (seq j) i
    h_Ioi_eq_Union : Eq (Set.Iio i) (Set.iUnion fun j => setOf fun k => LE.le k (s …
    h_lt_eq_preimage : Eq (setOf fun ω => LT.lt (τ ω) i) (Set.preimage τ (Set.Iio  …
    ⊢ MeasurableSet (Set.iUnion fun i => setOf fun a => LE.le (τ a) (seq i))
  -/
  exact MeasurableSet.iUnion fun n => f.mono (h_bound n).le _ (hτ.measurableSet_le (seq n))
  /-
    🎉 no goals
  -/


theorem IsStoppingTime.measurableSet_lt (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[f i] {ω | τ ω < i} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  obtain ⟨i', hi'_lub⟩ : ∃ i', IsLUB (Set.Iio i) i' := exists_lub_Iio i
  /-
    case intro
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i i' : ι
    hi'_lub : IsLUB (Set.Iio i) i'
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  cases' lub_Iio_eq_self_or_Iio_eq_Iic i hi'_lub with hi'_eq_i h_Iio_eq_Iic
    /-
      case intro.inl
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i i' : ι
      hi'_lub : IsLUB (Set.Iio i) i'
      hi'_eq_i : Eq i' i
      ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
    -/
  · rw [← hi'_eq_i] at hi'_lub ⊢
    /-
      case intro.inl
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i i' : ι
      hi'_lub : IsLUB (Set.Iio i') i'
      hi'_eq_i : Eq i' i
      ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i')
    -/
    exact hτ.measurableSet_lt_of_isLUB i' hi'_lub
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i i' : ι
      hi'_lub : IsLUB (Set.Iio i) i'
      h_Iio_eq_Iic : Eq (Set.Iio i) (Set.Iic i')
      ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
    -/
  · have h_lt_eq_preimage : {ω : Ω | τ ω < i} = τ ⁻¹' Set.Iio i := rfl
    /-
      case intro.inr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i i' : ι
      hi'_lub : IsLUB (Set.Iio i) i'
      h_Iio_eq_Iic : Eq (Set.Iio i) (Set.Iic i')
      h_lt_eq_preimage : Eq (setOf fun ω => LT.lt (τ ω) i) (Set.preimage τ (Set.Iio  …
      ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
    -/
    rw [h_lt_eq_preimage, h_Iio_eq_Iic]
    /-
      case intro.inr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝² : TopologicalSpace ι
      inst✝¹ : OrderTopology ι
      inst✝ : FirstCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i i' : ι
      hi'_lub : IsLUB (Set.Iio i) i'
      h_Iio_eq_Iic : Eq (Set.Iio i) (Set.Iic i')
      h_lt_eq_preimage : Eq (setOf fun ω => LT.lt (τ ω) i) (Set.preimage τ (Set.Iio  …
      ⊢ MeasurableSet (Set.preimage τ (Set.Iic i'))
    -/
    exact f.mono (lub_Iio_le i hi'_lub) _ (hτ.measurableSet_le i')
    /-
      🎉 no goals
    -/


theorem IsStoppingTime.measurableSet_ge (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[f i] {ω | i ≤ τ ω} := by
  have : {ω | i ≤ τ ω} = {ω | τ ω < i}ᶜ := by
    ext1 ω; simp only [Set.mem_setOf_eq, Set.mem_compl_iff, not_lt]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LE.le i (τ ω)) (HasCompl.compl (setOf fun ω => LT.lt …
    ⊢ MeasurableSet (setOf fun ω => LE.le i (τ ω))
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LE.le i (τ ω)) (HasCompl.compl (setOf fun ω => LT.lt …
    ⊢ MeasurableSet (HasCompl.compl (setOf fun ω => LT.lt (τ ω) i))
  -/
  exact (hτ.measurableSet_lt i).compl
  /-
    🎉 no goals
  -/


theorem IsStoppingTime.measurableSet_eq (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[f i] {ω | τ ω = i} := by
  have : {ω | τ ω = i} = {ω | τ ω ≤ i} ∩ {ω | τ ω ≥ i} := by
    ext1 ω; simp only [Set.mem_setOf_eq, Set.mem_inter_iff, le_antisymm_iff]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => Eq (τ ω) i) (Inter.inter (setOf fun ω => LE.le (τ ω) …
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) i)
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => Eq (τ ω) i) (Inter.inter (setOf fun ω => LE.le (τ ω) …
    ⊢ MeasurableSet (Inter.inter (setOf fun ω => LE.le (τ ω) i) (setOf fun ω => GE …
  -/
  exact (hτ.measurableSet_le i).inter (hτ.measurableSet_ge i)
  /-
    🎉 no goals
  -/


theorem IsStoppingTime.measurableSet_eq_le (hτ : IsStoppingTime f τ) {i j : ι} (hle : i ≤ j) :
    MeasurableSet[f j] {ω | τ ω = i} :=
  f.mono hle _ <| hτ.measurableSet_eq i


theorem IsStoppingTime.measurableSet_lt_le (hτ : IsStoppingTime f τ) {i j : ι} (hle : i ≤ j) :
    MeasurableSet[f j] {ω | τ ω < i} :=
  f.mono hle _ <| hτ.measurableSet_lt i


theorem isStoppingTime_of_measurableSet_eq [Preorder ι] [Countable ι] {f : Filtration ι m}
    {τ : Ω → ι} (hτ : ∀ i, MeasurableSet[f i] {ω | τ ω = i}) : IsStoppingTime f τ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    inst✝ : Countable ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : ∀ (i : ι), MeasurableSet (setOf fun ω => Eq (τ ω) i)
    ⊢ MeasureTheory.IsStoppingTime f τ
  -/
  intro i
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    inst✝ : Countable ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : ∀ (i : ι), MeasurableSet (setOf fun ω => Eq (τ ω) i)
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) i)
  -/
  rw [show {ω | τ ω ≤ i} = ⋃ k ≤ i, {ω | τ ω = k} by ext; simp]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    inst✝ : Countable ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : ∀ (i : ι), MeasurableSet (setOf fun ω => Eq (τ ω) i)
    i : ι
    ⊢ MeasurableSet (Set.iUnion fun k => Set.iUnion fun h => setOf fun ω => Eq (τ  …
  -/
  refine MeasurableSet.biUnion (Set.to_countable _) fun k hk => ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    inst✝ : Countable ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : ∀ (i : ι), MeasurableSet (setOf fun ω => Eq (τ ω) i)
    i k : ι
    hk : Membership.mem (fun k => Preorder.toLE.1 k i) k
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) k)
  -/
  exact f.mono hk _ (hτ k)
  /-
    🎉 no goals
  -/


protected theorem max [LinearOrder ι] {f : Filtration ι m} {τ π : Ω → ι} (hτ : IsStoppingTime f τ)
    (hπ : IsStoppingTime f π) : IsStoppingTime f fun ω => max (τ ω) (π ω) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ MeasureTheory.IsStoppingTime f fun ω => Max.max (τ ω) (π ω)
  -/
  intro i
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LE.le ((fun ω => Max.max (τ ω) (π ω)) ω) i)
  -/
  simp_rw [max_le_iff, Set.setOf_and]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : ι
    ⊢ MeasurableSet (Inter.inter (setOf fun a => LE.le (τ a) i) (setOf fun a => LE …
  -/
  exact (hτ i).inter (hπ i)
  /-
    🎉 no goals
  -/


protected theorem max_const [LinearOrder ι] {f : Filtration ι m} {τ : Ω → ι}
    (hτ : IsStoppingTime f τ) (i : ι) : IsStoppingTime f fun ω => max (τ ω) i :=
  hτ.max (isStoppingTime_const f i)


protected theorem min [LinearOrder ι] {f : Filtration ι m} {τ π : Ω → ι} (hτ : IsStoppingTime f τ)
    (hπ : IsStoppingTime f π) : IsStoppingTime f fun ω => min (τ ω) (π ω) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ MeasureTheory.IsStoppingTime f fun ω => Min.min (τ ω) (π ω)
  -/
  intro i
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LE.le ((fun ω => Min.min (τ ω) (π ω)) ω) i)
  -/
  simp_rw [min_le_iff, Set.setOf_or]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : ι
    ⊢ MeasurableSet (Union.union (setOf fun a => LE.le (τ a) i) (setOf fun a => LE …
  -/
  exact (hτ i).union (hπ i)
  /-
    🎉 no goals
  -/


protected theorem min_const [LinearOrder ι] {f : Filtration ι m} {τ : Ω → ι}
    (hτ : IsStoppingTime f τ) (i : ι) : IsStoppingTime f fun ω => min (τ ω) i :=
  hτ.min (isStoppingTime_const f i)


theorem add_const [AddGroup ι] [Preorder ι] [AddRightMono ι]
    [AddLeftMono ι] {f : Filtration ι m} {τ : Ω → ι} (hτ : IsStoppingTime f τ)
    {i : ι} (hi : 0 ≤ i) : IsStoppingTime f fun ω => τ ω + i := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : AddGroup ι
    inst✝² : Preorder ι
    inst✝¹ : AddRightMono ι
    inst✝ : AddLeftMono ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    hi : LE.le 0 i
    ⊢ MeasureTheory.IsStoppingTime f fun ω => HAdd.hAdd (τ ω) i
  -/
  intro j
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : AddGroup ι
    inst✝² : Preorder ι
    inst✝¹ : AddRightMono ι
    inst✝ : AddLeftMono ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    hi : LE.le 0 i
    j : ι
    ⊢ MeasurableSet (setOf fun ω => LE.le ((fun ω => HAdd.hAdd (τ ω) i) ω) j)
  -/
  simp_rw [← le_sub_iff_add_le]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : AddGroup ι
    inst✝² : Preorder ι
    inst✝¹ : AddRightMono ι
    inst✝ : AddLeftMono ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    hi : LE.le 0 i
    j : ι
    ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) (HSub.hSub j i))
  -/
  exact f.mono (sub_le_self j hi) _ (hτ (j - i))
  /-
    🎉 no goals
  -/


theorem add_const_nat {f : Filtration ℕ m} {τ : Ω → ℕ} (hτ : IsStoppingTime f τ) {i : ℕ} :
    IsStoppingTime f fun ω => τ ω + i := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    i : Nat
    ⊢ MeasureTheory.IsStoppingTime f fun ω => HAdd.hAdd (τ ω) i
  -/
  refine isStoppingTime_of_measurableSet_eq fun j => ?_
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    i j : Nat
    ⊢ MeasurableSet (setOf fun ω => Eq (HAdd.hAdd (τ ω) i) j)
  -/
  by_cases hij : i ≤ j
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      f : MeasureTheory.Filtration Nat m
      τ : Ω → Nat
      hτ : MeasureTheory.IsStoppingTime f τ
      i j : Nat
      hij : LE.le i j
      ⊢ MeasurableSet (setOf fun ω => Eq (HAdd.hAdd (τ ω) i) j)
    -/
  · simp_rw [eq_comm, ← Nat.sub_eq_iff_eq_add hij, eq_comm]
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      f : MeasureTheory.Filtration Nat m
      τ : Ω → Nat
      hτ : MeasureTheory.IsStoppingTime f τ
      i j : Nat
      hij : LE.le i j
      ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) (HSub.hSub j i))
    -/
    exact f.mono (j.sub_le i) _ (hτ.measurableSet_eq (j - i))
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      f : MeasureTheory.Filtration Nat m
      τ : Ω → Nat
      hτ : MeasureTheory.IsStoppingTime f τ
      i j : Nat
      hij : Not (LE.le i j)
      ⊢ MeasurableSet (setOf fun ω => Eq (HAdd.hAdd (τ ω) i) j)
    -/
  · rw [not_le] at hij
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      f : MeasureTheory.Filtration Nat m
      τ : Ω → Nat
      hτ : MeasureTheory.IsStoppingTime f τ
      i j : Nat
      hij : LT.lt j i
      ⊢ MeasurableSet (setOf fun ω => Eq (HAdd.hAdd (τ ω) i) j)
    -/
    convert @MeasurableSet.empty _ (f.1 j)
    /-
      case h.e'_3
      Ω : Type u_1
      m : MeasurableSpace Ω
      f : MeasureTheory.Filtration Nat m
      τ : Ω → Nat
      hτ : MeasureTheory.IsStoppingTime f τ
      i j : Nat
      hij : LT.lt j i
      ⊢ Eq (setOf fun ω => Eq (HAdd.hAdd (τ ω) i) j) EmptyCollection.emptyCollection
    -/
    ext ω
    /-
      case h.e'_3.h
      Ω : Type u_1
      m : MeasurableSpace Ω
      f : MeasureTheory.Filtration Nat m
      τ : Ω → Nat
      hτ : MeasureTheory.IsStoppingTime f τ
      i j : Nat
      hij : LT.lt j i
      ω : Ω
      ⊢ Iff (Membership.mem (setOf fun ω => Eq (HAdd.hAdd (τ ω) i) j) ω) (Membership …
    -/
    simp only [Set.mem_empty_iff_false, iff_false, Set.mem_setOf]
    /-
      case h.e'_3.h
      Ω : Type u_1
      m : MeasurableSpace Ω
      f : MeasureTheory.Filtration Nat m
      τ : Ω → Nat
      hτ : MeasureTheory.IsStoppingTime f τ
      i j : Nat
      hij : LT.lt j i
      ω : Ω
      ⊢ Not (Eq (HAdd.hAdd (τ ω) i) j)
    -/
    omega
    /-
      🎉 no goals
    -/

-- generalize to certain countable type?

theorem add {f : Filtration ℕ m} {τ π : Ω → ℕ} (hτ : IsStoppingTime f τ) (hπ : IsStoppingTime f π) :
    IsStoppingTime f (τ + π) := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ π : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ MeasureTheory.IsStoppingTime f (HAdd.hAdd τ π)
  -/
  intro i
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ π : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : Nat
    ⊢ MeasurableSet (setOf fun ω => LE.le (HAdd.hAdd τ π ω) i)
  -/
  rw [(_ : {ω | (τ + π) ω ≤ i} = ⋃ k ≤ i, {ω | π ω = k} ∩ {ω | τ ω + k ≤ i})]
  · exact MeasurableSet.iUnion fun k =>
      MeasurableSet.iUnion fun hk => (hπ.measurableSet_eq_le hk).inter (hτ.add_const_nat i)
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ π : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : Nat
    ⊢ Eq (setOf fun ω => LE.le (HAdd.hAdd τ π ω) i) (Set.iUnion fun k => Set.iUnio …
  -/
  ext ω
  /-
    case h
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ π : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : Nat
    ω : Ω
    ⊢ Iff (Membership.mem (setOf fun ω => LE.le (HAdd.hAdd τ π ω) i) ω) (Membershi …
  -/
  simp only [Pi.add_apply, Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, exists_prop]
  /-
    case h
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ π : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : Nat
    ω : Ω
    ⊢ Iff (LE.le (HAdd.hAdd (τ ω) (π ω)) i) (Exists fun i_1 => And (LE.le i_1 i) ( …
  -/
  refine ⟨fun h => ⟨π ω, by omega, rfl, h⟩, ?_⟩
  /-
    case h
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ π : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : Nat
    ω : Ω
    ⊢ (Exists fun i_1 => And (LE.le i_1 i) (And (Eq (π ω) i_1) (LE.le (HAdd.hAdd ( …
  -/
  rintro ⟨j, hj, rfl, h⟩
  /-
    case h.intro.intro.intro
    Ω : Type u_1
    m : MeasurableSpace Ω
    f : MeasureTheory.Filtration Nat m
    τ π : Ω → Nat
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    i : Nat
    ω : Ω
    hj : LE.le (π ω) i
    h : LE.le (HAdd.hAdd (τ ω) (π ω)) i
    ⊢ LE.le (HAdd.hAdd (τ ω) (π ω)) i
  -/
  assumption
  /-
    🎉 no goals
  -/


/-- The associated σ-algebra with a stopping time. -/
protected def measurableSpace (hτ : IsStoppingTime f τ) : MeasurableSpace Ω where
  MeasurableSet' s := ∀ i : ι, MeasurableSet[f i] (s ∩ {ω | τ ω ≤ i})
  measurableSet_empty i := (Set.empty_inter {ω | τ ω ≤ i}).symm ▸ @MeasurableSet.empty _ (f i)
  measurableSet_compl s hs i := by
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      hs : (fun s => ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le ( …
      i : ι
      ⊢ MeasurableSet (Inter.inter (HasCompl.compl s) (setOf fun ω => LE.le (τ ω) i))
    -/
    rw [(_ : sᶜ ∩ {ω | τ ω ≤ i} = (sᶜ ∪ {ω | τ ω ≤ i}ᶜ) ∩ {ω | τ ω ≤ i})]
      /-
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ π : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        hs : (fun s => ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le ( …
        i : ι
        ⊢ MeasurableSet (Inter.inter (Union.union (HasCompl.compl s) (HasCompl.compl ( …
      -/
    · refine MeasurableSet.inter ?_ ?_
        /-
          case refine_1
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ π : Ω → ι
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          hs : (fun s => ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le ( …
          i : ι
          ⊢ MeasurableSet (Union.union (HasCompl.compl s) (HasCompl.compl (setOf fun ω = …
        -/
      · rw [← Set.compl_inter]
        /-
          case refine_1
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ π : Ω → ι
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          hs : (fun s => ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le ( …
          i : ι
          ⊢ MeasurableSet (HasCompl.compl (Inter.inter s (setOf fun ω => LE.le (τ ω) i)))
        -/
        exact (hs i).compl
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ π : Ω → ι
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          hs : (fun s => ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le ( …
          i : ι
          ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) i)
        -/
      · exact hτ i
        /-
          🎉 no goals
        -/
      /-
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ π : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        hs : (fun s => ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le ( …
        i : ι
        ⊢ Eq (Inter.inter (HasCompl.compl s) (setOf fun ω => LE.le (τ ω) i)) (Inter.in …
      -/
    · rw [Set.union_inter_distrib_right]
      /-
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ π : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        hs : (fun s => ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le ( …
        i : ι
        ⊢ Eq (Inter.inter (HasCompl.compl s) (setOf fun ω => LE.le (τ ω) i)) (Union.un …
      -/
      simp only [Set.compl_inter_self, Set.union_empty]
      /-
        🎉 no goals
      -/
  measurableSet_iUnion s hs i := by
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Nat → Set Ω
      hs : ∀ (i : Nat), (fun s => ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun …
      i : ι
      ⊢ MeasurableSet (Inter.inter (Set.iUnion fun i => s i) (setOf fun ω => LE.le ( …
    -/
    rw [forall_swap] at hs
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Nat → Set Ω
      hs : ∀ (y : ι) (x : Nat), MeasurableSet (Inter.inter (s x) (setOf fun ω => LE. …
      i : ι
      ⊢ MeasurableSet (Inter.inter (Set.iUnion fun i => s i) (setOf fun ω => LE.le ( …
    -/
    rw [Set.iUnion_inter]
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Nat → Set Ω
      hs : ∀ (y : ι) (x : Nat), MeasurableSet (Inter.inter (s x) (setOf fun ω => LE. …
      i : ι
      ⊢ MeasurableSet (Set.iUnion fun i_1 => Inter.inter (s i_1) (setOf fun ω => LE. …
    -/
    exact MeasurableSet.iUnion (hs i)
    /-
      🎉 no goals
    -/


protected theorem measurableSet (hτ : IsStoppingTime f τ) (s : Set Ω) :
    MeasurableSet[hτ.measurableSpace] s ↔ ∀ i : ι, MeasurableSet[f i] (s ∩ {ω | τ ω ≤ i}) :=
  Iff.rfl


theorem measurableSpace_mono (hτ : IsStoppingTime f τ) (hπ : IsStoppingTime f π) (hle : τ ≤ π) :
    hτ.measurableSpace ≤ hπ.measurableSpace := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    hle : LE.le τ π
    ⊢ LE.le hτ.measurableSpace hπ.measurableSpace
  -/
  intro s hs i
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    hle : LE.le τ π
    s : Set Ω
    hs : MeasurableSet s
    i : ι
    ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le (π ω) i))
  -/
  rw [(_ : s ∩ {ω | π ω ≤ i} = s ∩ {ω | τ ω ≤ i} ∩ {ω | π ω ≤ i})]
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      hle : LE.le τ π
      s : Set Ω
      hs : MeasurableSet s
      i : ι
      ⊢ MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) i)) (s …
    -/
  · exact (hs i).inter (hπ i)
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      hle : LE.le τ π
      s : Set Ω
      hs : MeasurableSet s
      i : ι
      ⊢ Eq (Inter.inter s (setOf fun ω => LE.le (π ω) i)) (Inter.inter (Inter.inter  …
    -/
  · ext
    /-
      case h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      hle : LE.le τ π
      s : Set Ω
      hs : MeasurableSet s
      i : ι
      x✝ : Ω
      ⊢ Iff (Membership.mem (Inter.inter s (setOf fun ω => LE.le (π ω) i)) x✝) (Memb …
    -/
    simp only [Set.mem_inter_iff, iff_self_and, and_congr_left_iff, Set.mem_setOf_eq]
    /-
      case h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      hle : LE.le τ π
      s : Set Ω
      hs : MeasurableSet s
      i : ι
      x✝ : Ω
      ⊢ LE.le (π x✝) i → Membership.mem s x✝ → LE.le (τ x✝) i
    -/
    intro hle' _
    /-
      case h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      hle : LE.le τ π
      s : Set Ω
      hs : MeasurableSet s
      i : ι
      x✝ : Ω
      hle' : LE.le (π x✝) i
      a✝ : Membership.mem s x✝
      ⊢ LE.le (τ x✝) i
    -/
    exact le_trans (hle _) hle'
    /-
      🎉 no goals
    -/


theorem measurableSpace_le_of_countable [Countable ι] (hτ : IsStoppingTime f τ) :
    hτ.measurableSpace ≤ m := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : Countable ι
    hτ : MeasureTheory.IsStoppingTime f τ
    ⊢ LE.le hτ.measurableSpace m
  -/
  intro s hs
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : Countable ι
    hτ : MeasureTheory.IsStoppingTime f τ
    s : Set Ω
    hs : MeasurableSet s
    ⊢ MeasurableSet s
  -/
  change ∀ i, MeasurableSet[f i] (s ∩ {ω | τ ω ≤ i}) at hs
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : Countable ι
    hτ : MeasureTheory.IsStoppingTime f τ
    s : Set Ω
    hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
    ⊢ MeasurableSet s
  -/
  rw [(_ : s = ⋃ i, s ∩ {ω | τ ω ≤ i})]
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : Countable ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      ⊢ MeasurableSet (Set.iUnion fun i => Inter.inter s (setOf fun ω => LE.le (τ ω) …
    -/
  · exact MeasurableSet.iUnion fun i => f.le i _ (hs i)
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : Countable ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      ⊢ Eq s (Set.iUnion fun i => Inter.inter s (setOf fun ω => LE.le (τ ω) i))
    -/
  · ext ω; constructor <;> rw [Set.mem_iUnion]
      /-
        case h.mp
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝¹ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        inst✝ : Countable ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
        ω : Ω
        ⊢ Membership.mem s ω → Exists fun i => Membership.mem (Inter.inter s (setOf fu …
      -/
    · exact fun hx => ⟨τ ω, hx, le_rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mpr
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝¹ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        inst✝ : Countable ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
        ω : Ω
        ⊢ (Exists fun i => Membership.mem (Inter.inter s (setOf fun ω => LE.le (τ ω) i …
      -/
    · rintro ⟨_, hx, _⟩
      /-
        case h.mpr.intro.intro
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝¹ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        inst✝ : Countable ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
        ω : Ω
        w✝ : ι
        hx : Membership.mem s ω
        right✝ : Membership.mem (setOf fun ω => LE.le (τ ω) w✝) ω
        ⊢ Membership.mem s ω
      -/
      exact hx
      /-
        🎉 no goals
      -/


theorem measurableSpace_le [IsCountablyGenerated (atTop : Filter ι)] [IsDirected ι (· ≤ ·)]
    (hτ : IsStoppingTime f τ) : hτ.measurableSpace ≤ m := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝² : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝¹ : Filter.atTop.IsCountablyGenerated
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    hτ : MeasureTheory.IsStoppingTime f τ
    ⊢ LE.le hτ.measurableSpace m
  -/
  intro s hs
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝² : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝¹ : Filter.atTop.IsCountablyGenerated
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    hτ : MeasureTheory.IsStoppingTime f τ
    s : Set Ω
    hs : MeasurableSet s
    ⊢ MeasurableSet s
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝² : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝¹ : Filter.atTop.IsCountablyGenerated
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      hs : MeasurableSet s
      h✝ : IsEmpty ι
      ⊢ MeasurableSet s
    -/
  · haveI : IsEmpty Ω := ⟨fun ω => IsEmpty.false (τ ω)⟩
    /-
      case inl
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝² : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝¹ : Filter.atTop.IsCountablyGenerated
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      hs : MeasurableSet s
      h✝ : IsEmpty ι
      this : IsEmpty Ω
      ⊢ MeasurableSet s
    -/
    apply Subsingleton.measurableSet
    /-
      🎉 no goals
    -/
    /-
      case inr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝² : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝¹ : Filter.atTop.IsCountablyGenerated
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      hs : MeasurableSet s
      h✝ : Nonempty ι
      ⊢ MeasurableSet s
    -/
  · change ∀ i, MeasurableSet[f i] (s ∩ {ω | τ ω ≤ i}) at hs
    /-
      case inr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝² : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝¹ : Filter.atTop.IsCountablyGenerated
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      h✝ : Nonempty ι
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      ⊢ MeasurableSet s
    -/
    obtain ⟨seq : ℕ → ι, h_seq_tendsto⟩ := (atTop : Filter ι).exists_seq_tendsto
    /-
      case inr.intro
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝² : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝¹ : Filter.atTop.IsCountablyGenerated
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      h✝ : Nonempty ι
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      seq : Nat → ι
      h_seq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
      ⊢ MeasurableSet s
    -/
    rw [(_ : s = ⋃ n, s ∩ {ω | τ ω ≤ seq n})]
      /-
        case inr.intro
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝² : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        inst✝¹ : Filter.atTop.IsCountablyGenerated
        inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        h✝ : Nonempty ι
        hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
        seq : Nat → ι
        h_seq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
        ⊢ MeasurableSet (Set.iUnion fun n => Inter.inter s (setOf fun ω => LE.le (τ ω) …
      -/
    · exact MeasurableSet.iUnion fun i => f.le (seq i) _ (hs (seq i))
      /-
        🎉 no goals
      -/
      /-
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝² : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        inst✝¹ : Filter.atTop.IsCountablyGenerated
        inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        h✝ : Nonempty ι
        hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
        seq : Nat → ι
        h_seq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
        ⊢ Eq s (Set.iUnion fun n => Inter.inter s (setOf fun ω => LE.le (τ ω) (seq n)))
      -/
    · ext ω; constructor <;> rw [Set.mem_iUnion]
        /-
          case h.mp
          Ω : Type u_1
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝² : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ : Ω → ι
          inst✝¹ : Filter.atTop.IsCountablyGenerated
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          h✝ : Nonempty ι
          hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
          seq : Nat → ι
          h_seq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
          ω : Ω
          ⊢ Membership.mem s ω → Exists fun i => Membership.mem (Inter.inter s (setOf fu …
        -/
      · intro hx
        /-
          case h.mp
          Ω : Type u_1
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝² : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ : Ω → ι
          inst✝¹ : Filter.atTop.IsCountablyGenerated
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          h✝ : Nonempty ι
          hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
          seq : Nat → ι
          h_seq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
          ω : Ω
          hx : Membership.mem s ω
          ⊢ Exists fun i => Membership.mem (Inter.inter s (setOf fun ω => LE.le (τ ω) (s …
        -/
        suffices ∃ i, τ ω ≤ seq i from ⟨this.choose, hx, this.choose_spec⟩
        /-
          case h.mp
          Ω : Type u_1
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝² : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ : Ω → ι
          inst✝¹ : Filter.atTop.IsCountablyGenerated
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          h✝ : Nonempty ι
          hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
          seq : Nat → ι
          h_seq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
          ω : Ω
          hx : Membership.mem s ω
          ⊢ Exists fun i => LE.le (τ ω) (seq i)
        -/
        rw [tendsto_atTop] at h_seq_tendsto
        /-
          case h.mp
          Ω : Type u_1
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝² : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ : Ω → ι
          inst✝¹ : Filter.atTop.IsCountablyGenerated
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          h✝ : Nonempty ι
          hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
          seq : Nat → ι
          h_seq_tendsto : ∀ (b : ι), Filter.Eventually (fun a => LE.le b (seq a)) Filter …
          ω : Ω
          hx : Membership.mem s ω
          ⊢ Exists fun i => LE.le (τ ω) (seq i)
        -/
        exact (h_seq_tendsto (τ ω)).exists
        /-
          🎉 no goals
        -/
        /-
          case h.mpr
          Ω : Type u_1
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝² : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ : Ω → ι
          inst✝¹ : Filter.atTop.IsCountablyGenerated
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          h✝ : Nonempty ι
          hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
          seq : Nat → ι
          h_seq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
          ω : Ω
          ⊢ (Exists fun i => Membership.mem (Inter.inter s (setOf fun ω => LE.le (τ ω) ( …
        -/
      · rintro ⟨_, hx, _⟩
        /-
          case h.mpr.intro.intro
          Ω : Type u_1
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝² : Preorder ι
          f : MeasureTheory.Filtration ι m
          τ : Ω → ι
          inst✝¹ : Filter.atTop.IsCountablyGenerated
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          hτ : MeasureTheory.IsStoppingTime f τ
          s : Set Ω
          h✝ : Nonempty ι
          hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
          seq : Nat → ι
          h_seq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
          ω : Ω
          w✝ : Nat
          hx : Membership.mem s ω
          right✝ : Membership.mem (setOf fun ω => LE.le (τ ω) (seq w✝)) ω
          ⊢ Membership.mem s ω
        -/
        exact hx
        /-
          🎉 no goals
        -/


@[deprecated (since := "2024-12-25")] alias measurableSpace_le' := measurableSpace_le


@[simp]
theorem measurableSpace_const (f : Filtration ι m) (i : ι) :
    (isStoppingTime_const f i).measurableSpace = f i := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : Preorder ι
    f : MeasureTheory.Filtration ι m
    i : ι
    ⊢ Eq ⋯.measurableSpace (↑f i)
  -/
  ext1 s
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : Preorder ι
    f : MeasureTheory.Filtration ι m
    i : ι
    s : Set Ω
    ⊢ Iff (MeasurableSet s) (MeasurableSet s)
  -/
  change MeasurableSet[(isStoppingTime_const f i).measurableSpace] s ↔ MeasurableSet[f i] s
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : Preorder ι
    f : MeasureTheory.Filtration ι m
    i : ι
    s : Set Ω
    ⊢ Iff (MeasurableSet s) (MeasurableSet s)
  -/
  rw [IsStoppingTime.measurableSet]
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : Preorder ι
    f : MeasureTheory.Filtration ι m
    i : ι
    s : Set Ω
    ⊢ Iff (∀ (i_1 : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le i i_1)) …
  -/
  constructor <;> intro h
    /-
      case h.mp
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      i : ι
      s : Set Ω
      h : ∀ (i_1 : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le i i_1))
      ⊢ MeasurableSet s
    -/
  · specialize h i
    /-
      case h.mp
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      i : ι
      s : Set Ω
      h : MeasurableSet (Inter.inter s (setOf fun ω => LE.le i i))
      ⊢ MeasurableSet s
    -/
    simpa only [le_refl, Set.setOf_true, Set.inter_univ] using h
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      i : ι
      s : Set Ω
      h : MeasurableSet s
      ⊢ ∀ (i_1 : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le i i_1))
    -/
  · intro j
    /-
      case h.mpr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      i : ι
      s : Set Ω
      h : MeasurableSet s
      j : ι
      ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le i j))
    -/
    by_cases hij : i ≤ j
      /-
        case pos
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        i : ι
        s : Set Ω
        h : MeasurableSet s
        j : ι
        hij : LE.le i j
        ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le i j))
      -/
    · simp only [hij, Set.setOf_true, Set.inter_univ]
      /-
        case pos
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        i : ι
        s : Set Ω
        h : MeasurableSet s
        j : ι
        hij : LE.le i j
        ⊢ MeasurableSet s
      -/
      exact f.mono hij _ h
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        i : ι
        s : Set Ω
        h : MeasurableSet s
        j : ι
        hij : Not (LE.le i j)
        ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le i j))
      -/
    · simp only [hij, Set.setOf_false, Set.inter_empty, @MeasurableSet.empty _ (f.1 j)]
      /-
        🎉 no goals
      -/


theorem measurableSet_inter_eq_iff (hτ : IsStoppingTime f τ) (s : Set Ω) (i : ι) :
    MeasurableSet[hτ.measurableSpace] (s ∩ {ω | τ ω = i}) ↔
      MeasurableSet[f i] (s ∩ {ω | τ ω = i}) := by
  have : ∀ j, {ω : Ω | τ ω = i} ∩ {ω : Ω | τ ω ≤ j} = {ω : Ω | τ ω = i} ∩ {_ω | i ≤ j} := by
    intro j
    ext1 ω
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq, and_congr_right_iff]
    intro hxi
    rw [hxi]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : Preorder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    s : Set Ω
    i : ι
    this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
    ⊢ Iff (MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))) (MeasurableS …
  -/
  constructor <;> intro h
    /-
      case mp
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      i : ι
      this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
      h : MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
      ⊢ MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
    -/
  · specialize h i
    /-
      case mp
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      i : ι
      this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
      h : MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω => Eq (τ ω) i)) (se …
      ⊢ MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
    -/
    simpa only [Set.inter_assoc, this, le_refl, Set.setOf_true, Set.inter_univ] using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      i : ι
      this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
      h : MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
      ⊢ MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
    -/
  · intro j
    /-
      case mpr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      i : ι
      this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
      h : MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
      j : ι
      ⊢ MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω => Eq (τ ω) i)) (setO …
    -/
    rw [Set.inter_assoc, this]
    /-
      case mpr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      s : Set Ω
      i : ι
      this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
      h : MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
      j : ι
      ⊢ MeasurableSet (Inter.inter s (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf …
    -/
    by_cases hij : i ≤ j
      /-
        case pos
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        i : ι
        this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
        h : MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
        j : ι
        hij : LE.le i j
        ⊢ MeasurableSet (Inter.inter s (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf …
      -/
    · simp only [hij, Set.setOf_true, Set.inter_univ]
      /-
        case pos
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        i : ι
        this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
        h : MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
        j : ι
        hij : LE.le i j
        ⊢ MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
      -/
      exact f.mono hij _ h
      /-
        🎉 no goals
      -/
      /-
        case neg
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        s : Set Ω
        i : ι
        this : ∀ (j : ι), Eq (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf fun ω =>  …
        h : MeasurableSet (Inter.inter s (setOf fun ω => Eq (τ ω) i))
        j : ι
        hij : Not (LE.le i j)
        ⊢ MeasurableSet (Inter.inter s (Inter.inter (setOf fun ω => Eq (τ ω) i) (setOf …
      -/
    · simp [hij]
      /-
        🎉 no goals
      -/


theorem measurableSpace_le_of_le_const (hτ : IsStoppingTime f τ) {i : ι} (hτ_le : ∀ ω, τ ω ≤ i) :
    hτ.measurableSpace ≤ f i :=
  (measurableSpace_mono hτ _ hτ_le).trans (measurableSpace_const _ _).le


theorem measurableSpace_le_of_le (hτ : IsStoppingTime f τ) {n : ι} (hτ_le : ∀ ω, τ ω ≤ n) :
    hτ.measurableSpace ≤ m :=
  (hτ.measurableSpace_le_of_le_const hτ_le).trans (f.le n)


theorem le_measurableSpace_of_const_le (hτ : IsStoppingTime f τ) {i : ι} (hτ_le : ∀ ω, i ≤ τ ω) :
    f i ≤ hτ.measurableSpace :=
  (measurableSpace_const _ _).symm.le.trans (measurableSpace_mono _ hτ hτ_le)


instance sigmaFinite_stopping_time {ι} [SemilatticeSup ι] [OrderBot ι]
    [(Filter.atTop : Filter ι).IsCountablyGenerated] {μ : Measure Ω} {f : Filtration ι m}
    {τ : Ω → ι} [SigmaFiniteFiltration μ f] (hτ : IsStoppingTime f τ) :
    SigmaFinite (μ.trim hτ.measurableSpace_le) := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι✝ : Type u_3
    m : MeasurableSpace Ω
    ι : Type u_4
    inst✝³ : SemilatticeSup ι
    inst✝² : OrderBot ι
    inst✝¹ : Filter.atTop.IsCountablyGenerated
    μ : MeasureTheory.Measure Ω
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ f
    hτ : MeasureTheory.IsStoppingTime f τ
    ⊢ MeasureTheory.SigmaFinite (μ.trim ⋯)
  -/
  refine @sigmaFiniteTrim_mono _ _ ?_ _ _ _ ?_ ?_
    /-
      case refine_1
      Ω : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      m : MeasurableSpace Ω
      ι : Type u_4
      inst✝³ : SemilatticeSup ι
      inst✝² : OrderBot ι
      inst✝¹ : Filter.atTop.IsCountablyGenerated
      μ : MeasureTheory.Measure Ω
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ f
      hτ : MeasureTheory.IsStoppingTime f τ
      ⊢ MeasurableSpace Ω
    -/
  · exact f ⊥
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Ω : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      m : MeasurableSpace Ω
      ι : Type u_4
      inst✝³ : SemilatticeSup ι
      inst✝² : OrderBot ι
      inst✝¹ : Filter.atTop.IsCountablyGenerated
      μ : MeasureTheory.Measure Ω
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ f
      hτ : MeasureTheory.IsStoppingTime f τ
      ⊢ LE.le (↑f Bot.bot) hτ.measurableSpace
    -/
  · exact hτ.le_measurableSpace_of_const_le fun _ => bot_le
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      Ω : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      m : MeasurableSpace Ω
      ι : Type u_4
      inst✝³ : SemilatticeSup ι
      inst✝² : OrderBot ι
      inst✝¹ : Filter.atTop.IsCountablyGenerated
      μ : MeasureTheory.Measure Ω
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ f
      hτ : MeasureTheory.IsStoppingTime f τ
      ⊢ MeasureTheory.SigmaFinite (μ.trim ⋯)
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


instance sigmaFinite_stopping_time_of_le {ι} [SemilatticeSup ι] [OrderBot ι] {μ : Measure Ω}
    {f : Filtration ι m} {τ : Ω → ι} [SigmaFiniteFiltration μ f] (hτ : IsStoppingTime f τ) {n : ι}
    (hτ_le : ∀ ω, τ ω ≤ n) : SigmaFinite (μ.trim (hτ.measurableSpace_le_of_le hτ_le)) := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι✝ : Type u_3
    m : MeasurableSpace Ω
    ι : Type u_4
    inst✝² : SemilatticeSup ι
    inst✝¹ : OrderBot ι
    μ : MeasureTheory.Measure Ω
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ f
    hτ : MeasureTheory.IsStoppingTime f τ
    n : ι
    hτ_le : ∀ (ω : Ω), LE.le (τ ω) n
    ⊢ MeasureTheory.SigmaFinite (μ.trim ⋯)
  -/
  refine @sigmaFiniteTrim_mono _ _ ?_ _ _ _ ?_ ?_
    /-
      case refine_1
      Ω : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      m : MeasurableSpace Ω
      ι : Type u_4
      inst✝² : SemilatticeSup ι
      inst✝¹ : OrderBot ι
      μ : MeasureTheory.Measure Ω
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ f
      hτ : MeasureTheory.IsStoppingTime f τ
      n : ι
      hτ_le : ∀ (ω : Ω), LE.le (τ ω) n
      ⊢ MeasurableSpace Ω
    -/
  · exact f ⊥
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Ω : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      m : MeasurableSpace Ω
      ι : Type u_4
      inst✝² : SemilatticeSup ι
      inst✝¹ : OrderBot ι
      μ : MeasureTheory.Measure Ω
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ f
      hτ : MeasureTheory.IsStoppingTime f τ
      n : ι
      hτ_le : ∀ (ω : Ω), LE.le (τ ω) n
      ⊢ LE.le (↑f Bot.bot) hτ.measurableSpace
    -/
  · exact hτ.le_measurableSpace_of_const_le fun _ => bot_le
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      Ω : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      m : MeasurableSpace Ω
      ι : Type u_4
      inst✝² : SemilatticeSup ι
      inst✝¹ : OrderBot ι
      μ : MeasureTheory.Measure Ω
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ f
      hτ : MeasureTheory.IsStoppingTime f τ
      n : ι
      hτ_le : ∀ (ω : Ω), LE.le (τ ω) n
      ⊢ MeasureTheory.SigmaFinite (μ.trim ⋯)
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


protected theorem measurableSet_le' (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω ≤ i} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) i)
  -/
  intro j
  have : {ω : Ω | τ ω ≤ i} ∩ {ω : Ω | τ ω ≤ j} = {ω : Ω | τ ω ≤ min i j} := by
    ext1 ω; simp only [Set.mem_inter_iff, Set.mem_setOf_eq, le_min_iff]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i j : ι
    this : Eq (Inter.inter (setOf fun ω => LE.le (τ ω) i) (setOf fun ω => LE.le (τ …
    ⊢ MeasurableSet (Inter.inter (setOf fun ω => LE.le (τ ω) i) (setOf fun ω => LE …
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i j : ι
    this : Eq (Inter.inter (setOf fun ω => LE.le (τ ω) i) (setOf fun ω => LE.le (τ …
    ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) (Min.min i j))
  -/
  exact f.mono (min_le_right i j) _ (hτ _)
  /-
    🎉 no goals
  -/


protected theorem measurableSet_gt' (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | i < τ ω} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    ⊢ MeasurableSet (setOf fun ω => LT.lt i (τ ω))
  -/
  have : {ω : Ω | i < τ ω} = {ω : Ω | τ ω ≤ i}ᶜ := by ext1 ω; simp
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LT.lt i (τ ω)) (HasCompl.compl (setOf fun ω => LE.le …
    ⊢ MeasurableSet (setOf fun ω => LT.lt i (τ ω))
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LT.lt i (τ ω)) (HasCompl.compl (setOf fun ω => LE.le …
    ⊢ MeasurableSet (HasCompl.compl (setOf fun ω => LE.le (τ ω) i))
  -/
  exact (hτ.measurableSet_le' i).compl
  /-
    🎉 no goals
  -/


protected theorem measurableSet_eq' [TopologicalSpace ι] [OrderTopology ι]
    [FirstCountableTopology ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω = i} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) i)
  -/
  rw [← Set.univ_inter {ω | τ ω = i}, measurableSet_inter_eq_iff, Set.univ_inter]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) i)
  -/
  exact hτ.measurableSet_eq i
  /-
    🎉 no goals
  -/


protected theorem measurableSet_ge' [TopologicalSpace ι] [OrderTopology ι]
    [FirstCountableTopology ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | i ≤ τ ω} := by
  have : {ω | i ≤ τ ω} = {ω | τ ω = i} ∪ {ω | i < τ ω} := by
    ext1 ω
    simp only [le_iff_lt_or_eq, Set.mem_setOf_eq, Set.mem_union]
    rw [@eq_comm _ i, or_comm]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LE.le i (τ ω)) (Union.union (setOf fun ω => Eq (τ ω) …
    ⊢ MeasurableSet (setOf fun ω => LE.le i (τ ω))
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LE.le i (τ ω)) (Union.union (setOf fun ω => Eq (τ ω) …
    ⊢ MeasurableSet (Union.union (setOf fun ω => Eq (τ ω) i) (setOf fun ω => LT.lt …
  -/
  exact (hτ.measurableSet_eq' i).union (hτ.measurableSet_gt' i)
  /-
    🎉 no goals
  -/


protected theorem measurableSet_lt' [TopologicalSpace ι] [OrderTopology ι]
    [FirstCountableTopology ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω < i} := by
  have : {ω | τ ω < i} = {ω | τ ω ≤ i} \ {ω | τ ω = i} := by
    ext1 ω
    simp only [lt_iff_le_and_ne, Set.mem_setOf_eq, Set.mem_diff]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LT.lt (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ …
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝³ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    inst✝² : TopologicalSpace ι
    inst✝¹ : OrderTopology ι
    inst✝ : FirstCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    this : Eq (setOf fun ω => LT.lt (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ …
    ⊢ MeasurableSet (SDiff.sdiff (setOf fun ω => LE.le (τ ω) i) (setOf fun ω => Eq …
  -/
  exact (hτ.measurableSet_le' i).diff (hτ.measurableSet_eq' i)
  /-
    🎉 no goals
  -/


protected theorem measurableSet_eq_of_countable_range' (hτ : IsStoppingTime f τ)
    (h_countable : (Set.range τ).Countable) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω = i} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) i)
  -/
  rw [← Set.univ_inter {ω | τ ω = i}, measurableSet_inter_eq_iff, Set.univ_inter]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) i)
  -/
  exact hτ.measurableSet_eq_of_countable_range h_countable i
  /-
    🎉 no goals
  -/


protected theorem measurableSet_eq_of_countable' [Countable ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω = i} :=
  hτ.measurableSet_eq_of_countable_range' (Set.to_countable _) i


protected theorem measurableSet_ge_of_countable_range' (hτ : IsStoppingTime f τ)
    (h_countable : (Set.range τ).Countable) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | i ≤ τ ω} := by
  have : {ω | i ≤ τ ω} = {ω | τ ω = i} ∪ {ω | i < τ ω} := by
    ext1 ω
    simp only [le_iff_lt_or_eq, Set.mem_setOf_eq, Set.mem_union]
    rw [@eq_comm _ i, or_comm]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => LE.le i (τ ω)) (Union.union (setOf fun ω => Eq (τ ω) …
    ⊢ MeasurableSet (setOf fun ω => LE.le i (τ ω))
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => LE.le i (τ ω)) (Union.union (setOf fun ω => Eq (τ ω) …
    ⊢ MeasurableSet (Union.union (setOf fun ω => Eq (τ ω) i) (setOf fun ω => LT.lt …
  -/
  exact (hτ.measurableSet_eq_of_countable_range' h_countable i).union (hτ.measurableSet_gt' i)
  /-
    🎉 no goals
  -/


protected theorem measurableSet_ge_of_countable' [Countable ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | i ≤ τ ω} :=
  hτ.measurableSet_ge_of_countable_range' (Set.to_countable _) i


protected theorem measurableSet_lt_of_countable_range' (hτ : IsStoppingTime f τ)
    (h_countable : (Set.range τ).Countable) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω < i} := by
  have : {ω | τ ω < i} = {ω | τ ω ≤ i} \ {ω | τ ω = i} := by
    ext1 ω
    simp only [lt_iff_le_and_ne, Set.mem_setOf_eq, Set.mem_diff]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => LT.lt (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ …
    ⊢ MeasurableSet (setOf fun ω => LT.lt (τ ω) i)
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    i : ι
    this : Eq (setOf fun ω => LT.lt (τ ω) i) (SDiff.sdiff (setOf fun ω => LE.le (τ …
    ⊢ MeasurableSet (SDiff.sdiff (setOf fun ω => LE.le (τ ω) i) (setOf fun ω => Eq …
  -/
  exact (hτ.measurableSet_le' i).diff (hτ.measurableSet_eq_of_countable_range' h_countable i)
  /-
    🎉 no goals
  -/


protected theorem measurableSet_lt_of_countable' [Countable ι] (hτ : IsStoppingTime f τ) (i : ι) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω < i} :=
  hτ.measurableSet_lt_of_countable_range' (Set.to_countable _) i


protected theorem measurableSpace_le_of_countable_range (hτ : IsStoppingTime f τ)
    (h_countable : (Set.range τ).Countable) : hτ.measurableSpace ≤ m := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    ⊢ LE.le hτ.measurableSpace m
  -/
  intro s hs
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    s : Set Ω
    hs : MeasurableSet s
    ⊢ MeasurableSet s
  -/
  change ∀ i, MeasurableSet[f i] (s ∩ {ω | τ ω ≤ i}) at hs
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    h_countable : (Set.range τ).Countable
    s : Set Ω
    hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
    ⊢ MeasurableSet s
  -/
  rw [(_ : s = ⋃ i ∈ Set.range τ, s ∩ {ω | τ ω ≤ i})]
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      h_countable : (Set.range τ).Countable
      s : Set Ω
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      ⊢ MeasurableSet (Set.iUnion fun i => Set.iUnion fun h => Inter.inter s (setOf  …
    -/
  · exact MeasurableSet.biUnion h_countable fun i _ => f.le i _ (hs i)
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      h_countable : (Set.range τ).Countable
      s : Set Ω
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      ⊢ Eq s (Set.iUnion fun i => Set.iUnion fun h => Inter.inter s (setOf fun ω =>  …
    -/
  · ext ω
    /-
      case h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      h_countable : (Set.range τ).Countable
      s : Set Ω
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      ω : Ω
      ⊢ Iff (Membership.mem s ω) (Membership.mem (Set.iUnion fun i => Set.iUnion fun …
    -/
    constructor <;> rw [Set.mem_iUnion]
      /-
        case h.mp
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : LinearOrder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        h_countable : (Set.range τ).Countable
        s : Set Ω
        hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
        ω : Ω
        ⊢ Membership.mem s ω → Exists fun i => Membership.mem (Set.iUnion fun h => Int …
      -/
    · exact fun hx => ⟨τ ω, by simpa using hx⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mpr
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : LinearOrder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        h_countable : (Set.range τ).Countable
        s : Set Ω
        hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
        ω : Ω
        ⊢ (Exists fun i => Membership.mem (Set.iUnion fun h => Inter.inter s (setOf fu …
      -/
    · rintro ⟨i, hx⟩
      simp only [Set.mem_range, Set.iUnion_exists, Set.mem_iUnion, Set.mem_inter_iff,
        Set.mem_setOf_eq, exists_prop, exists_and_right] at hx
      /-
        case h.mpr.intro
        Ω : Type u_1
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : LinearOrder ι
        f : MeasureTheory.Filtration ι m
        τ : Ω → ι
        hτ : MeasureTheory.IsStoppingTime f τ
        h_countable : (Set.range τ).Countable
        s : Set Ω
        hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
        ω : Ω
        i : ι
        hx : And (Exists fun y => Eq (τ y) i) (And (Membership.mem s ω) (LE.le (τ ω) i))
        ⊢ Membership.mem s ω
      -/
      exact hx.2.1
      /-
        🎉 no goals
      -/


protected theorem measurable [TopologicalSpace ι] [MeasurableSpace ι] [BorelSpace ι]
    [OrderTopology ι] [SecondCountableTopology ι] (hτ : IsStoppingTime f τ) :
    Measurable[hτ.measurableSpace] τ :=
  @measurable_of_Iic ι Ω _ _ _ hτ.measurableSpace _ _ _ _ fun i => hτ.measurableSet_le' i


protected theorem measurable_of_le [TopologicalSpace ι] [MeasurableSpace ι] [BorelSpace ι]
    [OrderTopology ι] [SecondCountableTopology ι] (hτ : IsStoppingTime f τ) {i : ι}
    (hτ_le : ∀ ω, τ ω ≤ i) : Measurable[f i] τ :=
  hτ.measurable.mono (measurableSpace_le_of_le_const _ hτ_le) le_rfl


theorem measurableSpace_min (hτ : IsStoppingTime f τ) (hπ : IsStoppingTime f π) :
    (hτ.min hπ).measurableSpace = hτ.measurableSpace ⊓ hπ.measurableSpace := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ Eq ⋯.measurableSpace (Min.min hτ.measurableSpace hπ.measurableSpace)
  -/
  refine le_antisymm ?_ ?_
  · exact le_inf (measurableSpace_mono _ hτ fun _ => min_le_left _ _)
      (measurableSpace_mono _ hπ fun _ => min_le_right _ _)
    /-
      case refine_2
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      ⊢ LE.le (Min.min hτ.measurableSpace hπ.measurableSpace) ⋯.measurableSpace
    -/
  · intro s
    change MeasurableSet[hτ.measurableSpace] s ∧ MeasurableSet[hπ.measurableSpace] s →
      MeasurableSet[(hτ.min hπ).measurableSpace] s
    /-
      case refine_2
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      ⊢ And (MeasurableSet s) (MeasurableSet s) → MeasurableSet s
    -/
    simp_rw [IsStoppingTime.measurableSet]
    have : ∀ i, {ω | min (τ ω) (π ω) ≤ i} = {ω | τ ω ≤ i} ∪ {ω | π ω ≤ i} := by
      intro i; ext1 ω; simp
    /-
      case refine_2
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      this : ∀ (i : ι), Eq (setOf fun ω => LE.le (Min.min (τ ω) (π ω)) i) (Union.uni …
      ⊢ And (∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i)) …
    -/
    simp_rw [this, Set.inter_union_distrib_left]
    /-
      case refine_2
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      this : ∀ (i : ι), Eq (setOf fun ω => LE.le (Min.min (τ ω) (π ω)) i) (Union.uni …
      ⊢ And (∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i)) …
    -/
    exact fun h i => (h.left i).union (h.right i)
    /-
      🎉 no goals
    -/


theorem measurableSet_min_iff (hτ : IsStoppingTime f τ) (hπ : IsStoppingTime f π) (s : Set Ω) :
    MeasurableSet[(hτ.min hπ).measurableSpace] s ↔
      MeasurableSet[hτ.measurableSpace] s ∧ MeasurableSet[hπ.measurableSpace] s := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    s : Set Ω
    ⊢ Iff (MeasurableSet s) (And (MeasurableSet s) (MeasurableSet s))
  -/
  rw [measurableSpace_min hτ hπ]; rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem measurableSpace_min_const (hτ : IsStoppingTime f τ) {i : ι} :
    (hτ.min_const i).measurableSpace = hτ.measurableSpace ⊓ f i := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    ⊢ Eq ⋯.measurableSpace (Min.min hτ.measurableSpace (↑f i))
  -/
  rw [hτ.measurableSpace_min (isStoppingTime_const _ i), measurableSpace_const]
  /-
    🎉 no goals
  -/


theorem measurableSet_min_const_iff (hτ : IsStoppingTime f τ) (s : Set Ω) {i : ι} :
    MeasurableSet[(hτ.min_const i).measurableSpace] s ↔
      MeasurableSet[hτ.measurableSpace] s ∧ MeasurableSet[f i] s := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    s : Set Ω
    i : ι
    ⊢ Iff (MeasurableSet s) (And (MeasurableSet s) (MeasurableSet s))
  -/
  rw [measurableSpace_min_const hτ]; apply MeasurableSpace.measurableSet_inf
                                     /-
                                       🎉 no goals
                                     -/


theorem measurableSet_inter_le [TopologicalSpace ι] [SecondCountableTopology ι] [OrderTopology ι]
    [MeasurableSpace ι] [BorelSpace ι] (hτ : IsStoppingTime f τ) (hπ : IsStoppingTime f π)
    (s : Set Ω) (hs : MeasurableSet[hτ.measurableSpace] s) :
    MeasurableSet[(hτ.min hπ).measurableSpace] (s ∩ {ω | τ ω ≤ π ω}) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    s : Set Ω
    hs : MeasurableSet s
    ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))
  -/
  simp_rw [IsStoppingTime.measurableSet] at hs ⊢
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    s : Set Ω
    hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
    ⊢ ∀ (i : ι), MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω => LE.le ( …
  -/
  intro i
  have : s ∩ {ω | τ ω ≤ π ω} ∩ {ω | min (τ ω) (π ω) ≤ i} =
      s ∩ {ω | τ ω ≤ i} ∩ {ω | min (τ ω) (π ω) ≤ i} ∩
        {ω | min (τ ω) i ≤ min (min (τ ω) (π ω)) i} := by
    ext1 ω
    simp only [min_le_iff, Set.mem_inter_iff, Set.mem_setOf_eq, le_min_iff, le_refl, true_and,
      true_or]
    by_cases hτi : τ ω ≤ i
    · simp only [hτi, true_or, and_true, and_congr_right_iff]
      intro
      constructor <;> intro h
      · exact Or.inl h
      · cases' h with h h
        · exact h
        · exact hτi.trans h
    simp only [hτi, false_or, and_false, false_and, iff_false, not_and, not_le, and_imp]
    refine fun _ hτ_le_π => lt_of_lt_of_le ?_ hτ_le_π
    rw [← not_le]
    exact hτi
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    s : Set Ω
    hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
    i : ι
    this : Eq (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω))) (set …
    ⊢ MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)) …
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    s : Set Ω
    hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
    i : ι
    this : Eq (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω))) (set …
    ⊢ MeasurableSet (Inter.inter (Inter.inter (Inter.inter s (setOf fun ω => LE.le …
  -/
  refine ((hs i).inter ((hτ.min hπ) i)).inter ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    s : Set Ω
    hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
    i : ι
    this : Eq (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω))) (set …
    ⊢ MeasurableSet (setOf fun ω => LE.le (Min.min (τ ω) i) (Min.min (Min.min (τ ω …
  -/
  apply @measurableSet_le _ _ _ _ _ (Filtration.seq f i) _ _ _ _ _ ?_ ?_
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : SecondCountableTopology ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSpace ι
      inst✝ : BorelSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      i : ι
      this : Eq (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω))) (set …
      ⊢ Measurable fun a => Min.min (τ a) i
    -/
  · exact (hτ.min_const i).measurable_of_le fun _ => min_le_right _ _
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : SecondCountableTopology ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSpace ι
      inst✝ : BorelSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      hs : ∀ (i : ι), MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
      i : ι
      this : Eq (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω))) (set …
      ⊢ Measurable fun a => Min.min (Min.min (τ a) (π a)) i
    -/
  · exact ((hτ.min hπ).min_const i).measurable_of_le fun _ => min_le_right _ _
    /-
      🎉 no goals
    -/


theorem measurableSet_inter_le_iff [TopologicalSpace ι] [SecondCountableTopology ι]
    [OrderTopology ι] [MeasurableSpace ι] [BorelSpace ι] (hτ : IsStoppingTime f τ)
    (hπ : IsStoppingTime f π) (s : Set Ω) :
    MeasurableSet[hτ.measurableSpace] (s ∩ {ω | τ ω ≤ π ω}) ↔
      MeasurableSet[(hτ.min hπ).measurableSpace] (s ∩ {ω | τ ω ≤ π ω}) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    s : Set Ω
    ⊢ Iff (MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))) (Meas …
  -/
  constructor <;> intro h
  · have : s ∩ {ω | τ ω ≤ π ω} = s ∩ {ω | τ ω ≤ π ω} ∩ {ω | τ ω ≤ π ω} := by
      rw [Set.inter_assoc, Set.inter_self]
    /-
      case mp
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : SecondCountableTopology ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSpace ι
      inst✝ : BorelSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      h : MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))
      this : Eq (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω))) (Inter.inter (Int …
      ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))
    -/
    rw [this]
    /-
      case mp
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : SecondCountableTopology ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSpace ι
      inst✝ : BorelSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      h : MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))
      this : Eq (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω))) (Inter.inter (Int …
      ⊢ MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)) …
    -/
    exact measurableSet_inter_le _ hπ _ h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : SecondCountableTopology ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSpace ι
      inst✝ : BorelSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      h : MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))
      ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))
    -/
  · rw [measurableSet_min_iff hτ hπ] at h
    /-
      case mpr
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : SecondCountableTopology ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSpace ι
      inst✝ : BorelSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      s : Set Ω
      h : And (MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))) (Me …
      ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) (π ω)))
    -/
    exact h.1
    /-
      🎉 no goals
    -/


theorem measurableSet_inter_le_const_iff (hτ : IsStoppingTime f τ) (s : Set Ω) (i : ι) :
    MeasurableSet[hτ.measurableSpace] (s ∩ {ω | τ ω ≤ i}) ↔
      MeasurableSet[(hτ.min_const i).measurableSpace] (s ∩ {ω | τ ω ≤ i}) := by
  rw [IsStoppingTime.measurableSet_min_iff hτ (isStoppingTime_const _ i),
    IsStoppingTime.measurableSpace_const, IsStoppingTime.measurableSet]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    s : Set Ω
    i : ι
    ⊢ Iff (∀ (i_1 : ι), MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω =>  …
  -/
  refine ⟨fun h => ⟨h, ?_⟩, fun h j => h.1 j⟩
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    s : Set Ω
    i : ι
    h : ∀ (i_1 : ι), MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω => LE. …
    ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
  -/
  specialize h i
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ : Ω → ι
    hτ : MeasureTheory.IsStoppingTime f τ
    s : Set Ω
    i : ι
    h : MeasurableSet (Inter.inter (Inter.inter s (setOf fun ω => LE.le (τ ω) i))  …
    ⊢ MeasurableSet (Inter.inter s (setOf fun ω => LE.le (τ ω) i))
  -/
  rwa [Set.inter_assoc, Set.inter_self] at h
  /-
    🎉 no goals
  -/


theorem measurableSet_le_stopping_time [TopologicalSpace ι] [SecondCountableTopology ι]
    [OrderTopology ι] [MeasurableSpace ι] [BorelSpace ι] (hτ : IsStoppingTime f τ)
    (hπ : IsStoppingTime f π) : MeasurableSet[hτ.measurableSpace] {ω | τ ω ≤ π ω} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) (π ω))
  -/
  rw [hτ.measurableSet]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ ∀ (i : ι), MeasurableSet (Inter.inter (setOf fun ω => LE.le (τ ω) (π ω)) (se …
  -/
  intro j
  have : {ω | τ ω ≤ π ω} ∩ {ω | τ ω ≤ j} = {ω | min (τ ω) j ≤ min (π ω) j} ∩ {ω | τ ω ≤ j} := by
    ext1 ω
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq, min_le_iff, le_min_iff, le_refl,
      and_congr_left_iff]
    intro h
    simp only [h, or_self_iff, and_true]
    rw [Iff.comm, or_iff_left_iff_imp]
    exact h.trans
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    j : ι
    this : Eq (Inter.inter (setOf fun ω => LE.le (τ ω) (π ω)) (setOf fun ω => LE.l …
    ⊢ MeasurableSet (Inter.inter (setOf fun ω => LE.le (τ ω) (π ω)) (setOf fun ω = …
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    j : ι
    this : Eq (Inter.inter (setOf fun ω => LE.le (τ ω) (π ω)) (setOf fun ω => LE.l …
    ⊢ MeasurableSet (Inter.inter (setOf fun ω => LE.le (Min.min (τ ω) j) (Min.min  …
  -/
  refine MeasurableSet.inter ?_ (hτ.measurableSet_le j)
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    j : ι
    this : Eq (Inter.inter (setOf fun ω => LE.le (τ ω) (π ω)) (setOf fun ω => LE.l …
    ⊢ MeasurableSet (setOf fun ω => LE.le (Min.min (τ ω) j) (Min.min (π ω) j))
  -/
  apply @measurableSet_le _ _ _ _ _ (Filtration.seq f j) _ _ _ _ _ ?_ ?_
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : SecondCountableTopology ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSpace ι
      inst✝ : BorelSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      j : ι
      this : Eq (Inter.inter (setOf fun ω => LE.le (τ ω) (π ω)) (setOf fun ω => LE.l …
      ⊢ Measurable fun a => Min.min (τ a) j
    -/
  · exact (hτ.min_const j).measurable_of_le fun _ => min_le_right _ _
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : SecondCountableTopology ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSpace ι
      inst✝ : BorelSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      j : ι
      this : Eq (Inter.inter (setOf fun ω => LE.le (τ ω) (π ω)) (setOf fun ω => LE.l …
      ⊢ Measurable fun a => Min.min (π a) j
    -/
  · exact (hπ.min_const j).measurable_of_le fun _ => min_le_right _ _
    /-
      🎉 no goals
    -/


theorem measurableSet_stopping_time_le [TopologicalSpace ι] [SecondCountableTopology ι]
    [OrderTopology ι] [MeasurableSpace ι] [BorelSpace ι] (hτ : IsStoppingTime f τ)
    (hπ : IsStoppingTime f π) : MeasurableSet[hπ.measurableSpace] {ω | τ ω ≤ π ω} := by
  suffices MeasurableSet[(hτ.min hπ).measurableSpace] {ω : Ω | τ ω ≤ π ω} by
    rw [measurableSet_min_iff hτ hπ] at this; exact this.2
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) (π ω))
  -/
  rw [← Set.univ_inter {ω : Ω | τ ω ≤ π ω}, ← hτ.measurableSet_inter_le_iff hπ, Set.univ_inter]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : SecondCountableTopology ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSpace ι
    inst✝ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ MeasurableSet (setOf fun ω => LE.le (τ ω) (π ω))
  -/
  exact measurableSet_le_stopping_time hτ hπ
  /-
    🎉 no goals
  -/


theorem measurableSet_eq_stopping_time [AddGroup ι] [TopologicalSpace ι] [MeasurableSpace ι]
    [BorelSpace ι] [OrderTopology ι] [MeasurableSingletonClass ι] [SecondCountableTopology ι]
    [MeasurableSub₂ ι] (hτ : IsStoppingTime f τ) (hπ : IsStoppingTime f π) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω = π ω} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁸ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁷ : AddGroup ι
    inst✝⁶ : TopologicalSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : MeasurableSingletonClass ι
    inst✝¹ : SecondCountableTopology ι
    inst✝ : MeasurableSub₂ ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) (π ω))
  -/
  rw [hτ.measurableSet]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁸ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁷ : AddGroup ι
    inst✝⁶ : TopologicalSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : MeasurableSingletonClass ι
    inst✝¹ : SecondCountableTopology ι
    inst✝ : MeasurableSub₂ ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ ∀ (i : ι), MeasurableSet (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf …
  -/
  intro j
  have : {ω | τ ω = π ω} ∩ {ω | τ ω ≤ j} =
      {ω | min (τ ω) j = min (π ω) j} ∩ {ω | τ ω ≤ j} ∩ {ω | π ω ≤ j} := by
    ext1 ω
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq]
    refine ⟨fun h => ⟨⟨?_, h.2⟩, ?_⟩, fun h => ⟨?_, h.1.2⟩⟩
    · rw [h.1]
    · rw [← h.1]; exact h.2
    · cases' h with h' hσ_le
      cases' h' with h_eq hτ_le
      rwa [min_eq_left hτ_le, min_eq_left hσ_le] at h_eq
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁸ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁷ : AddGroup ι
    inst✝⁶ : TopologicalSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : MeasurableSingletonClass ι
    inst✝¹ : SecondCountableTopology ι
    inst✝ : MeasurableSub₂ ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    j : ι
    this : Eq (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => LE.le ( …
    ⊢ MeasurableSet (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => L …
  -/
  rw [this]
  refine
    MeasurableSet.inter (MeasurableSet.inter ?_ (hτ.measurableSet_le j)) (hπ.measurableSet_le j)
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁸ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁷ : AddGroup ι
    inst✝⁶ : TopologicalSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : MeasurableSingletonClass ι
    inst✝¹ : SecondCountableTopology ι
    inst✝ : MeasurableSub₂ ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    j : ι
    this : Eq (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => LE.le ( …
    ⊢ MeasurableSet (setOf fun ω => Eq (Min.min (τ ω) j) (Min.min (π ω) j))
  -/
  apply measurableSet_eq_fun
    /-
      case hf
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁸ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁷ : AddGroup ι
      inst✝⁶ : TopologicalSpace ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : BorelSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : MeasurableSingletonClass ι
      inst✝¹ : SecondCountableTopology ι
      inst✝ : MeasurableSub₂ ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      j : ι
      this : Eq (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => LE.le ( …
      ⊢ Measurable fun x => Min.min (τ x) j
    -/
  · exact (hτ.min_const j).measurable_of_le fun _ => min_le_right _ _
    /-
      🎉 no goals
    -/
    /-
      case hg
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁸ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁷ : AddGroup ι
      inst✝⁶ : TopologicalSpace ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : BorelSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : MeasurableSingletonClass ι
      inst✝¹ : SecondCountableTopology ι
      inst✝ : MeasurableSub₂ ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      j : ι
      this : Eq (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => LE.le ( …
      ⊢ Measurable fun x => Min.min (π x) j
    -/
  · exact (hπ.min_const j).measurable_of_le fun _ => min_le_right _ _
    /-
      🎉 no goals
    -/


theorem measurableSet_eq_stopping_time_of_countable [Countable ι] [TopologicalSpace ι]
    [MeasurableSpace ι] [BorelSpace ι] [OrderTopology ι] [MeasurableSingletonClass ι]
    [SecondCountableTopology ι] (hτ : IsStoppingTime f τ) (hπ : IsStoppingTime f π) :
    MeasurableSet[hτ.measurableSpace] {ω | τ ω = π ω} := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁷ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁶ : Countable ι
    inst✝⁵ : TopologicalSpace ι
    inst✝⁴ : MeasurableSpace ι
    inst✝³ : BorelSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSingletonClass ι
    inst✝ : SecondCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) (π ω))
  -/
  rw [hτ.measurableSet]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁷ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁶ : Countable ι
    inst✝⁵ : TopologicalSpace ι
    inst✝⁴ : MeasurableSpace ι
    inst✝³ : BorelSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSingletonClass ι
    inst✝ : SecondCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    ⊢ ∀ (i : ι), MeasurableSet (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf …
  -/
  intro j
  have : {ω | τ ω = π ω} ∩ {ω | τ ω ≤ j} =
      {ω | min (τ ω) j = min (π ω) j} ∩ {ω | τ ω ≤ j} ∩ {ω | π ω ≤ j} := by
    ext1 ω
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq]
    refine ⟨fun h => ⟨⟨?_, h.2⟩, ?_⟩, fun h => ⟨?_, h.1.2⟩⟩
    · rw [h.1]
    · rw [← h.1]; exact h.2
    · cases' h with h' hπ_le
      cases' h' with h_eq hτ_le
      rwa [min_eq_left hτ_le, min_eq_left hπ_le] at h_eq
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁷ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁶ : Countable ι
    inst✝⁵ : TopologicalSpace ι
    inst✝⁴ : MeasurableSpace ι
    inst✝³ : BorelSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSingletonClass ι
    inst✝ : SecondCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    j : ι
    this : Eq (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => LE.le ( …
    ⊢ MeasurableSet (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => L …
  -/
  rw [this]
  refine
    MeasurableSet.inter (MeasurableSet.inter ?_ (hτ.measurableSet_le j)) (hπ.measurableSet_le j)
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁷ : LinearOrder ι
    f : MeasureTheory.Filtration ι m
    τ π : Ω → ι
    inst✝⁶ : Countable ι
    inst✝⁵ : TopologicalSpace ι
    inst✝⁴ : MeasurableSpace ι
    inst✝³ : BorelSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : MeasurableSingletonClass ι
    inst✝ : SecondCountableTopology ι
    hτ : MeasureTheory.IsStoppingTime f τ
    hπ : MeasureTheory.IsStoppingTime f π
    j : ι
    this : Eq (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => LE.le ( …
    ⊢ MeasurableSet (setOf fun ω => Eq (Min.min (τ ω) j) (Min.min (π ω) j))
  -/
  apply measurableSet_eq_fun_of_countable
    /-
      case hf
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁷ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁶ : Countable ι
      inst✝⁵ : TopologicalSpace ι
      inst✝⁴ : MeasurableSpace ι
      inst✝³ : BorelSpace ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSingletonClass ι
      inst✝ : SecondCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      j : ι
      this : Eq (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => LE.le ( …
      ⊢ Measurable fun x => Min.min (τ x) j
    -/
  · exact (hτ.min_const j).measurable_of_le fun _ => min_le_right _ _
    /-
      🎉 no goals
    -/
    /-
      case hg
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁷ : LinearOrder ι
      f : MeasureTheory.Filtration ι m
      τ π : Ω → ι
      inst✝⁶ : Countable ι
      inst✝⁵ : TopologicalSpace ι
      inst✝⁴ : MeasurableSpace ι
      inst✝³ : BorelSpace ι
      inst✝² : OrderTopology ι
      inst✝¹ : MeasurableSingletonClass ι
      inst✝ : SecondCountableTopology ι
      hτ : MeasureTheory.IsStoppingTime f τ
      hπ : MeasureTheory.IsStoppingTime f π
      j : ι
      this : Eq (Inter.inter (setOf fun ω => Eq (τ ω) (π ω)) (setOf fun ω => LE.le ( …
      ⊢ Measurable fun x => Min.min (π x) j
    -/
  · exact (hπ.min_const j).measurable_of_le fun _ => min_le_right _ _
    /-
      🎉 no goals
    -/


/-- Given a map `u : ι → Ω → E`, its stopped value with respect to the stopping
time `τ` is the map `x ↦ u (τ ω) ω`. -/
def stoppedValue (u : ι → Ω → β) (τ : Ω → ι) : Ω → β := fun ω => u (τ ω) ω


theorem stoppedValue_const (u : ι → Ω → β) (i : ι) : (stoppedValue u fun _ => i) = u i :=
  rfl


/-- Given a map `u : ι → Ω → E`, the stopped process with respect to `τ` is `u i ω` if
`i ≤ τ ω`, and `u (τ ω) ω` otherwise.

Intuitively, the stopped process stops evolving once the stopping time has occurred. -/
def stoppedProcess (u : ι → Ω → β) (τ : Ω → ι) : ι → Ω → β := fun i ω => u (min i (τ ω)) ω


theorem stoppedProcess_eq_stoppedValue {u : ι → Ω → β} {τ : Ω → ι} :
    stoppedProcess u τ = fun i => stoppedValue u fun ω => min i (τ ω) :=
  rfl


theorem stoppedValue_stoppedProcess {u : ι → Ω → β} {τ σ : Ω → ι} :
    stoppedValue (stoppedProcess u τ) σ = stoppedValue u fun ω => min (σ ω) (τ ω) :=
  rfl


theorem stoppedProcess_eq_of_le {u : ι → Ω → β} {τ : Ω → ι} {i : ι} {ω : Ω} (h : i ≤ τ ω) :
                                         /-
                                           Ω : Type u_1
                                           β : Type u_2
                                           ι : Type u_3
                                           inst✝ : LinearOrder ι
                                           u : ι → Ω → β
                                           τ : Ω → ι
                                           i : ι
                                           ω : Ω
                                           h : LE.le i (τ ω)
                                           ⊢ Eq (MeasureTheory.stoppedProcess u τ i ω) (u i ω)
                                         -/
    stoppedProcess u τ i ω = u i ω := by simp [stoppedProcess, min_eq_left h]
                                         /-
                                           🎉 no goals
                                         -/


theorem stoppedProcess_eq_of_ge {u : ι → Ω → β} {τ : Ω → ι} {i : ι} {ω : Ω} (h : τ ω ≤ i) :
                                             /-
                                               Ω : Type u_1
                                               β : Type u_2
                                               ι : Type u_3
                                               inst✝ : LinearOrder ι
                                               u : ι → Ω → β
                                               τ : Ω → ι
                                               i : ι
                                               ω : Ω
                                               h : LE.le (τ ω) i
                                               ⊢ Eq (MeasureTheory.stoppedProcess u τ i ω) (u (τ ω) ω)
                                             -/
    stoppedProcess u τ i ω = u (τ ω) ω := by simp [stoppedProcess, min_eq_right h]
                                             /-
                                               🎉 no goals
                                             -/


theorem progMeasurable_min_stopping_time [MetrizableSpace ι] (hτ : IsStoppingTime f τ) :
    ProgMeasurable f fun i ω => min i (τ ω) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝ : TopologicalSpace.MetrizableSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    ⊢ MeasureTheory.ProgMeasurable f fun i ω => Min.min i (τ ω)
  -/
  intro i
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝ : TopologicalSpace.MetrizableSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    ⊢ MeasureTheory.StronglyMeasurable fun p => (fun i ω => Min.min i (τ ω)) (↑p.1 …
  -/
  let m_prod : MeasurableSpace (Set.Iic i × Ω) := Subtype.instMeasurableSpace.prod (f i)
  let m_set : ∀ t : Set (Set.Iic i × Ω), MeasurableSpace t := fun _ =>
    @Subtype.instMeasurableSpace (Set.Iic i × Ω) _ m_prod
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝ : TopologicalSpace.MetrizableSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
    m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
    ⊢ MeasureTheory.StronglyMeasurable fun p => (fun i ω => Min.min i (τ ω)) (↑p.1 …
  -/
  let s := {p : Set.Iic i × Ω | τ p.2 ≤ i}
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝ : TopologicalSpace.MetrizableSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
    m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
    s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
    ⊢ MeasureTheory.StronglyMeasurable fun p => (fun i ω => Min.min i (τ ω)) (↑p.1 …
  -/
  have hs : MeasurableSet[m_prod] s := @measurable_snd (Set.Iic i) Ω _ (f i) _ (hτ i)
  have h_meas_fst : ∀ t : Set (Set.Iic i × Ω),
      Measurable[m_set t] fun x : t => ((x : Set.Iic i × Ω).fst : ι) :=
    fun t => (@measurable_subtype_coe (Set.Iic i × Ω) m_prod _).fst.subtype_val
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝ : TopologicalSpace.MetrizableSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
    m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
    s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
    hs : MeasurableSet s
    h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
    ⊢ MeasureTheory.StronglyMeasurable fun p => (fun i ω => Min.min i (τ ω)) (↑p.1 …
  -/
  apply Measurable.stronglyMeasurable
  /-
    case hf
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝ : TopologicalSpace.MetrizableSpace ι
    hτ : MeasureTheory.IsStoppingTime f τ
    i : ι
    m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
    m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
    s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
    hs : MeasurableSet s
    h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
    ⊢ Measurable fun p => (fun i ω => Min.min i (τ ω)) (↑p.1) p.2
  -/
  refine measurable_of_restrict_of_restrict_compl hs ?_ ?_
    /-
      case hf.refine_1
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      ⊢ Measurable (s.restrict fun p => (fun i ω => Min.min i (τ ω)) (↑p.1) p.2)
    -/
  · refine @Measurable.min _ _ _ _ _ (m_set s) _ _ _ _ _ (h_meas_fst s) ?_
    /-
      case hf.refine_1
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      ⊢ Measurable fun a => τ (↑a).2
    -/
    refine @measurable_of_Iic ι s _ _ _ (m_set s) _ _ _ _ fun j => ?_
    have h_set_eq : (fun x : s => τ (x : Set.Iic i × Ω).snd) ⁻¹' Set.Iic j =
        (fun x : s => (x : Set.Iic i × Ω).snd) ⁻¹' {ω | τ ω ≤ min i j} := by
      ext1 ω
      simp only [Set.mem_preimage, Set.mem_Iic, iff_and_self, le_min_iff, Set.mem_setOf_eq]
      exact fun _ => ω.prop
    /-
      case hf.refine_1
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      j : ι
      h_set_eq : Eq (Set.preimage (fun x => τ (↑x).2) (Set.Iic j)) (Set.preimage (fu …
      ⊢ MeasurableSet (Set.preimage (fun a => τ (↑a).2) (Set.Iic j))
    -/
    rw [h_set_eq]
    suffices h_meas : @Measurable _ _ (m_set s) (f i) fun x : s ↦ (x : Set.Iic i × Ω).snd from
      h_meas (f.mono (min_le_left _ _) _ (hτ.measurableSet_le (min i j)))
    /-
      case hf.refine_1
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      j : ι
      h_set_eq : Eq (Set.preimage (fun x => τ (↑x).2) (Set.Iic j)) (Set.preimage (fu …
      ⊢ Measurable fun x => (↑x).2
    -/
    exact measurable_snd.comp (@measurable_subtype_coe _ m_prod _)
    /-
      🎉 no goals
    -/
    /-
      case hf.refine_2
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      ⊢ Measurable ((HasCompl.compl s).restrict fun p => (fun i ω => Min.min i (τ ω) …
    -/
  · letI sc := sᶜ
    suffices h_min_eq_left :
      (fun x : sc => min (↑(x : Set.Iic i × Ω).fst) (τ (x : Set.Iic i × Ω).snd)) = fun x : sc =>
        ↑(x : Set.Iic i × Ω).fst by
      simp +unfoldPartialApp only [sc, Set.restrict, h_min_eq_left]
      exact h_meas_fst _
    /-
      case hf.refine_2
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      sc : Set (Prod (↑(Set.Iic i)) Ω) := HasCompl.compl s
      ⊢ Eq (fun x => Min.min (↑(↑x).1) (τ (↑x).2)) fun x => ↑(↑x).1
    -/
    ext1 ω
    /-
      case hf.refine_2.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      sc : Set (Prod (↑(Set.Iic i)) Ω) := HasCompl.compl s
      ω : ↑sc
      ⊢ Eq (Min.min (↑(↑ω).1) (τ (↑ω).2)) ↑(↑ω).1
    -/
    rw [min_eq_left]
    /-
      case hf.refine_2.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      sc : Set (Prod (↑(Set.Iic i)) Ω) := HasCompl.compl s
      ω : ↑sc
      ⊢ LE.le (↑(↑ω).1) (τ (↑ω).2)
    -/
    have hx_fst_le : ↑(ω : Set.Iic i × Ω).fst ≤ i := (ω : Set.Iic i × Ω).fst.prop
    /-
      case hf.refine_2.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      sc : Set (Prod (↑(Set.Iic i)) Ω) := HasCompl.compl s
      ω : ↑sc
      hx_fst_le : LE.le (↑(↑ω).1) i
      ⊢ LE.le (↑(↑ω).1) (τ (↑ω).2)
    -/
    refine hx_fst_le.trans (le_of_lt ?_)
    /-
      case hf.refine_2.h
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      sc : Set (Prod (↑(Set.Iic i)) Ω) := HasCompl.compl s
      ω : ↑sc
      hx_fst_le : LE.le (↑(↑ω).1) i
      ⊢ LT.lt i (τ (↑ω).2)
    -/
    convert ω.prop
    /-
      case a
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁶ : LinearOrder ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : SecondCountableTopology ι
      inst✝¹ : BorelSpace ι
      τ : Ω → ι
      f : MeasureTheory.Filtration ι m
      inst✝ : TopologicalSpace.MetrizableSpace ι
      hτ : MeasureTheory.IsStoppingTime f τ
      i : ι
      m_prod : MeasurableSpace (Prod (↑(Set.Iic i)) Ω) := Subtype.instMeasurableSpac …
      m_set : (t : Set (Prod (↑(Set.Iic i)) Ω)) → MeasurableSpace ↑t := fun x => Sub …
      s : Set (Prod (↑(Set.Iic i)) Ω) := setOf fun p => LE.le (τ p.2) i
      hs : MeasurableSet s
      h_meas_fst : ∀ (t : Set (Prod (↑(Set.Iic i)) Ω)), Measurable fun x => ↑(↑x).1
      sc : Set (Prod (↑(Set.Iic i)) Ω) := HasCompl.compl s
      ω : ↑sc
      hx_fst_le : LE.le (↑(↑ω).1) i
      ⊢ Iff (LT.lt i (τ (↑ω).2)) (Membership.mem sc ↑ω)
    -/
    simp only [sc, s, not_le, Set.mem_compl_iff, Set.mem_setOf_eq]
    /-
      🎉 no goals
    -/


theorem ProgMeasurable.stoppedProcess [MetrizableSpace ι] (h : ProgMeasurable f u)
    (hτ : IsStoppingTime f τ) : ProgMeasurable f (stoppedProcess u τ) :=
  h.comp (progMeasurable_min_stopping_time hτ) fun _ _ => min_le_left _ _


theorem ProgMeasurable.adapted_stoppedProcess [MetrizableSpace ι] (h : ProgMeasurable f u)
    (hτ : IsStoppingTime f τ) : Adapted f (MeasureTheory.stoppedProcess u τ) :=
  (h.stoppedProcess hτ).adapted


theorem ProgMeasurable.stronglyMeasurable_stoppedProcess [MetrizableSpace ι]
    (hu : ProgMeasurable f u) (hτ : IsStoppingTime f τ) (i : ι) :
    StronglyMeasurable (MeasureTheory.stoppedProcess u τ i) :=
  (hu.adapted_stoppedProcess hτ i).mono (f.le _)


theorem stronglyMeasurable_stoppedValue_of_le (h : ProgMeasurable f u) (hτ : IsStoppingTime f τ)
    {n : ι} (hτ_le : ∀ ω, τ ω ≤ n) : StronglyMeasurable[f n] (stoppedValue u τ) := by
  have : stoppedValue u τ =
      (fun p : Set.Iic n × Ω => u (↑p.fst) p.snd) ∘ fun ω => (⟨τ ω, hτ_le ω⟩, ω) := by
    ext1 ω; simp only [stoppedValue, Function.comp_apply, Subtype.coe_mk]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    inst✝ : TopologicalSpace β
    u : ι → Ω → β
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    h : MeasureTheory.ProgMeasurable f u
    hτ : MeasureTheory.IsStoppingTime f τ
    n : ι
    hτ_le : ∀ (ω : Ω), LE.le (τ ω) n
    this : Eq (MeasureTheory.stoppedValue u τ) (Function.comp (fun p => u (↑p.1) p …
    ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.stoppedValue u τ)
  -/
  rw [this]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    inst✝ : TopologicalSpace β
    u : ι → Ω → β
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    h : MeasureTheory.ProgMeasurable f u
    hτ : MeasureTheory.IsStoppingTime f τ
    n : ι
    hτ_le : ∀ (ω : Ω), LE.le (τ ω) n
    this : Eq (MeasureTheory.stoppedValue u τ) (Function.comp (fun p => u (↑p.1) p …
    ⊢ MeasureTheory.StronglyMeasurable (Function.comp (fun p => u (↑p.1) p.2) fun  …
  -/
  refine StronglyMeasurable.comp_measurable (h n) ?_
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    inst✝ : TopologicalSpace β
    u : ι → Ω → β
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    h : MeasureTheory.ProgMeasurable f u
    hτ : MeasureTheory.IsStoppingTime f τ
    n : ι
    hτ_le : ∀ (ω : Ω), LE.le (τ ω) n
    this : Eq (MeasureTheory.stoppedValue u τ) (Function.comp (fun p => u (↑p.1) p …
    ⊢ Measurable fun ω => { fst := ⟨τ ω, ⋯⟩, snd := ω }
  -/
  exact (hτ.measurable_of_le hτ_le).subtype_mk.prod_mk measurable_id
  /-
    🎉 no goals
  -/


theorem measurable_stoppedValue [MetrizableSpace β] [MeasurableSpace β] [BorelSpace β]
    (hf_prog : ProgMeasurable f u) (hτ : IsStoppingTime f τ) :
    Measurable[hτ.measurableSpace] (stoppedValue u τ) := by
  have h_str_meas : ∀ i, StronglyMeasurable[f i] (stoppedValue u fun ω => min (τ ω) i) := fun i =>
    stronglyMeasurable_stoppedValue_of_le hf_prog (hτ.min_const i) fun _ => min_le_right _ _
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁹ : LinearOrder ι
    inst✝⁸ : MeasurableSpace ι
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : OrderTopology ι
    inst✝⁵ : SecondCountableTopology ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : TopologicalSpace β
    u : ι → Ω → β
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝² : TopologicalSpace.MetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    hf_prog : MeasureTheory.ProgMeasurable f u
    hτ : MeasureTheory.IsStoppingTime f τ
    h_str_meas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (MeasureTheory.stoppe …
    ⊢ Measurable (MeasureTheory.stoppedValue u τ)
  -/
  intro t ht i
  suffices stoppedValue u τ ⁻¹' t ∩ {ω : Ω | τ ω ≤ i} =
      (stoppedValue u fun ω => min (τ ω) i) ⁻¹' t ∩ {ω : Ω | τ ω ≤ i} by
    rw [this]; exact ((h_str_meas i).measurable ht).inter (hτ.measurableSet_le i)
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁹ : LinearOrder ι
    inst✝⁸ : MeasurableSpace ι
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : OrderTopology ι
    inst✝⁵ : SecondCountableTopology ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : TopologicalSpace β
    u : ι → Ω → β
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝² : TopologicalSpace.MetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    hf_prog : MeasureTheory.ProgMeasurable f u
    hτ : MeasureTheory.IsStoppingTime f τ
    h_str_meas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (MeasureTheory.stoppe …
    t : Set β
    ht : MeasurableSet t
    i : ι
    ⊢ Eq (Inter.inter (Set.preimage (MeasureTheory.stoppedValue u τ) t) (setOf fun …
  -/
  ext1 ω
  simp only [stoppedValue, Set.mem_inter_iff, Set.mem_preimage, Set.mem_setOf_eq,
    and_congr_left_iff]
  /-
    case h
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁹ : LinearOrder ι
    inst✝⁸ : MeasurableSpace ι
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : OrderTopology ι
    inst✝⁵ : SecondCountableTopology ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : TopologicalSpace β
    u : ι → Ω → β
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝² : TopologicalSpace.MetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    hf_prog : MeasureTheory.ProgMeasurable f u
    hτ : MeasureTheory.IsStoppingTime f τ
    h_str_meas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (MeasureTheory.stoppe …
    t : Set β
    ht : MeasurableSet t
    i : ι
    ω : Ω
    ⊢ LE.le (τ ω) i → Iff (Membership.mem t (u (τ ω) ω)) (Membership.mem t (u (Min …
  -/
  intro h
  /-
    case h
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁹ : LinearOrder ι
    inst✝⁸ : MeasurableSpace ι
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : OrderTopology ι
    inst✝⁵ : SecondCountableTopology ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : TopologicalSpace β
    u : ι → Ω → β
    τ : Ω → ι
    f : MeasureTheory.Filtration ι m
    inst✝² : TopologicalSpace.MetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    hf_prog : MeasureTheory.ProgMeasurable f u
    hτ : MeasureTheory.IsStoppingTime f τ
    h_str_meas : ∀ (i : ι), MeasureTheory.StronglyMeasurable (MeasureTheory.stoppe …
    t : Set β
    ht : MeasurableSet t
    i : ι
    ω : Ω
    h : LE.le (τ ω) i
    ⊢ Iff (Membership.mem t (u (τ ω) ω)) (Membership.mem t (u (Min.min (τ ω) i) ω))
  -/
  rw [min_eq_left h]
  /-
    🎉 no goals
  -/


theorem stoppedValue_eq_of_mem_finset [AddCommMonoid E] {s : Finset ι} (hbdd : ∀ ω, τ ω ∈ s) :
    stoppedValue u τ = ∑ i ∈ s, Set.indicator {ω | τ ω = i} (u i) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝ : AddCommMonoid E
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    ⊢ Eq (MeasureTheory.stoppedValue u τ) (s.sum fun i => (setOf fun ω => Eq (τ ω) …
  -/
  ext y
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝ : AddCommMonoid E
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    y : Ω
    ⊢ Eq (MeasureTheory.stoppedValue u τ y) (s.sum (fun i => (setOf fun ω => Eq (τ …
  -/
  rw [stoppedValue, Finset.sum_apply, Finset.sum_indicator_eq_sum_filter]
  suffices Finset.filter (fun i => y ∈ {ω : Ω | τ ω = i}) s = ({τ y} : Finset ι) by
    rw [this, Finset.sum_singleton]
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝ : AddCommMonoid E
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    y : Ω
    ⊢ Eq (Finset.filter (fun i => Membership.mem (setOf fun ω => Eq (τ ω) i) y) s) …
  -/
  ext1 ω
  /-
    case h.h
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝ : AddCommMonoid E
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    y : Ω
    ω : ι
    ⊢ Iff (Membership.mem (Finset.filter (fun i => Membership.mem (setOf fun ω =>  …
  -/
  simp only [Set.mem_setOf_eq, Finset.mem_filter, Finset.mem_singleton]
  /-
    case h.h
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝ : AddCommMonoid E
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    y : Ω
    ω : ι
    ⊢ Iff (And (Membership.mem s ω) (Eq (τ y) ω)) (Eq ω (τ y))
  -/
  constructor <;> intro h
    /-
      case h.h.mp
      Ω : Type u_1
      ι : Type u_3
      τ : Ω → ι
      E : Type u_4
      u : ι → Ω → E
      inst✝ : AddCommMonoid E
      s : Finset ι
      hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
      y : Ω
      ω : ι
      h : And (Membership.mem s ω) (Eq (τ y) ω)
      ⊢ Eq ω (τ y)
    -/
  · exact h.2.symm
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      Ω : Type u_1
      ι : Type u_3
      τ : Ω → ι
      E : Type u_4
      u : ι → Ω → E
      inst✝ : AddCommMonoid E
      s : Finset ι
      hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
      y : Ω
      ω : ι
      h : Eq ω (τ y)
      ⊢ And (Membership.mem s ω) (Eq (τ y) ω)
    -/
  · refine ⟨?_, h.symm⟩; rw [h]; exact hbdd y
                                 /-
                                   🎉 no goals
                                 -/


theorem stoppedValue_eq' [Preorder ι] [LocallyFiniteOrderBot ι] [AddCommMonoid E] {N : ι}
    (hbdd : ∀ ω, τ ω ≤ N) :
    stoppedValue u τ = ∑ i ∈ Finset.Iic N, Set.indicator {ω | τ ω = i} (u i) :=
  stoppedValue_eq_of_mem_finset fun ω => Finset.mem_Iic.mpr (hbdd ω)


theorem stoppedProcess_eq_of_mem_finset [LinearOrder ι] [AddCommMonoid E] {s : Finset ι} (n : ι)
    (hbdd : ∀ ω, τ ω < n → τ ω ∈ s) : stoppedProcess u τ n = Set.indicator {a | n ≤ τ a} (u n) +
      ∑ i ∈ s.filter (· < n), Set.indicator {ω | τ ω = i} (u i) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝¹ : LinearOrder ι
    inst✝ : AddCommMonoid E
    s : Finset ι
    n : ι
    hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
    ⊢ Eq (MeasureTheory.stoppedProcess u τ n) (HAdd.hAdd ((setOf fun a => LE.le n  …
  -/
  ext ω
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝¹ : LinearOrder ι
    inst✝ : AddCommMonoid E
    s : Finset ι
    n : ι
    hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
    ω : Ω
    ⊢ Eq (MeasureTheory.stoppedProcess u τ n ω) (HAdd.hAdd ((setOf fun a => LE.le  …
  -/
  rw [Pi.add_apply, Finset.sum_apply]
  /-
    case h
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝¹ : LinearOrder ι
    inst✝ : AddCommMonoid E
    s : Finset ι
    n : ι
    hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
    ω : Ω
    ⊢ Eq (MeasureTheory.stoppedProcess u τ n ω) (HAdd.hAdd ((setOf fun a => LE.le  …
  -/
  rcases le_or_lt n (τ ω) with h | h
    /-
      case h.inl
      Ω : Type u_1
      ι : Type u_3
      τ : Ω → ι
      E : Type u_4
      u : ι → Ω → E
      inst✝¹ : LinearOrder ι
      inst✝ : AddCommMonoid E
      s : Finset ι
      n : ι
      hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
      ω : Ω
      h : LE.le n (τ ω)
      ⊢ Eq (MeasureTheory.stoppedProcess u τ n ω) (HAdd.hAdd ((setOf fun a => LE.le  …
    -/
  · rw [stoppedProcess_eq_of_le h, Set.indicator_of_mem, Finset.sum_eq_zero, add_zero]
      /-
        case h.inl
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LE.le n (τ ω)
        ⊢ ∀ (x : ι), Membership.mem (Finset.filter (fun x => LT.lt x n) s) x → Eq ((se …
      -/
    · intro m hm
      /-
        case h.inl
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LE.le n (τ ω)
        m : ι
        hm : Membership.mem (Finset.filter (fun x => LT.lt x n) s) m
        ⊢ Eq ((setOf fun ω => Eq (τ ω) m).indicator (u m) ω) 0
      -/
      refine Set.indicator_of_not_mem ?_ _
      /-
        case h.inl
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LE.le n (τ ω)
        m : ι
        hm : Membership.mem (Finset.filter (fun x => LT.lt x n) s) m
        ⊢ Not (Membership.mem (setOf fun ω => Eq (τ ω) m) ω)
      -/
      rw [Finset.mem_filter] at hm
      /-
        case h.inl
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LE.le n (τ ω)
        m : ι
        hm : And (Membership.mem s m) (LT.lt m n)
        ⊢ Not (Membership.mem (setOf fun ω => Eq (τ ω) m) ω)
      -/
      exact (hm.2.trans_le h).ne'
      /-
        🎉 no goals
      -/
      /-
        case h.inl.h
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LE.le n (τ ω)
        ⊢ Membership.mem (setOf fun a => LE.le n (τ a)) ω
      -/
    · exact h
      /-
        🎉 no goals
      -/
    /-
      case h.inr
      Ω : Type u_1
      ι : Type u_3
      τ : Ω → ι
      E : Type u_4
      u : ι → Ω → E
      inst✝¹ : LinearOrder ι
      inst✝ : AddCommMonoid E
      s : Finset ι
      n : ι
      hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
      ω : Ω
      h : LT.lt (τ ω) n
      ⊢ Eq (MeasureTheory.stoppedProcess u τ n ω) (HAdd.hAdd ((setOf fun a => LE.le  …
    -/
  · rw [stoppedProcess_eq_of_ge (le_of_lt h), Finset.sum_eq_single_of_mem (τ ω)]
      /-
        case h.inr
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LT.lt (τ ω) n
        ⊢ Eq (u (τ ω) ω) (HAdd.hAdd ((setOf fun a => LE.le n (τ a)).indicator (u n) ω) …
      -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    · rw [Set.indicator_of_not_mem, zero_add, Set.indicator_of_mem] <;> rw [Set.mem_setOf]
      /-
        case h.inr.h
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LT.lt (τ ω) n
        ⊢ Not (LE.le n (τ ω))
      -/
      exact not_le.2 h
      /-
        🎉 no goals
      -/
      /-
        case h.inr.h
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LT.lt (τ ω) n
        ⊢ Membership.mem (Finset.filter (fun x => LT.lt x n) s) (τ ω)
      -/
    · rw [Finset.mem_filter]
      /-
        case h.inr.h
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LT.lt (τ ω) n
        ⊢ And (Membership.mem s (τ ω)) (LT.lt (τ ω) n)
      -/
      exact ⟨hbdd ω h, h⟩
      /-
        🎉 no goals
      -/
      /-
        case h.inr.h₀
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LT.lt (τ ω) n
        ⊢ ∀ (b : ι), Membership.mem (Finset.filter (fun x => LT.lt x n) s) b → Ne b (τ …
      -/
    · intro b _ hneq
      /-
        case h.inr.h₀
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LT.lt (τ ω) n
        b : ι
        a✝ : Membership.mem (Finset.filter (fun x => LT.lt x n) s) b
        hneq : Ne b (τ ω)
        ⊢ Eq ((setOf fun ω => Eq (τ ω) b).indicator (u b) ω) 0
      -/
      rw [Set.indicator_of_not_mem]
      /-
        case h.inr.h₀.h
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LT.lt (τ ω) n
        b : ι
        a✝ : Membership.mem (Finset.filter (fun x => LT.lt x n) s) b
        hneq : Ne b (τ ω)
        ⊢ Not (Membership.mem (setOf fun ω => Eq (τ ω) b) ω)
      -/
      rw [Set.mem_setOf]
      /-
        case h.inr.h₀.h
        Ω : Type u_1
        ι : Type u_3
        τ : Ω → ι
        E : Type u_4
        u : ι → Ω → E
        inst✝¹ : LinearOrder ι
        inst✝ : AddCommMonoid E
        s : Finset ι
        n : ι
        hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
        ω : Ω
        h : LT.lt (τ ω) n
        b : ι
        a✝ : Membership.mem (Finset.filter (fun x => LT.lt x n) s) b
        hneq : Ne b (τ ω)
        ⊢ Not (Eq (τ ω) b)
      -/
      exact hneq.symm
      /-
        🎉 no goals
      -/


theorem stoppedProcess_eq'' [LinearOrder ι] [LocallyFiniteOrderBot ι] [AddCommMonoid E] (n : ι) :
    stoppedProcess u τ n = Set.indicator {a | n ≤ τ a} (u n) +
      ∑ i ∈ Finset.Iio n, Set.indicator {ω | τ ω = i} (u i) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : AddCommMonoid E
    n : ι
    ⊢ Eq (MeasureTheory.stoppedProcess u τ n) (HAdd.hAdd ((setOf fun a => LE.le n  …
  -/
  have h_mem : ∀ ω, τ ω < n → τ ω ∈ Finset.Iio n := fun ω h => Finset.mem_Iio.mpr h
  /-
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : AddCommMonoid E
    n : ι
    h_mem : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem (Finset.Iio n) (τ ω)
    ⊢ Eq (MeasureTheory.stoppedProcess u τ n) (HAdd.hAdd ((setOf fun a => LE.le n  …
  -/
  rw [stoppedProcess_eq_of_mem_finset n h_mem]
  /-
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : AddCommMonoid E
    n : ι
    h_mem : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem (Finset.Iio n) (τ ω)
    ⊢ Eq (HAdd.hAdd ((setOf fun a => LE.le n (τ a)).indicator (u n)) ((Finset.filt …
  -/
  congr with i
  /-
    case e_a.e_s.h
    Ω : Type u_1
    ι : Type u_3
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : AddCommMonoid E
    n : ι
    h_mem : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem (Finset.Iio n) (τ ω)
    i : ι
    ⊢ Iff (Membership.mem (Finset.filter (fun x => LT.lt x n) (Finset.Iio n)) i) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem memℒp_stoppedValue_of_mem_finset (hτ : IsStoppingTime ℱ τ) (hu : ∀ n, Memℒp (u n) p μ)
    {s : Finset ι} (hbdd : ∀ ω, τ ω ∈ s) : Memℒp (stoppedValue u τ) p μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    p : ENNReal
    u : ι → Ω → E
    inst✝¹ : PartialOrder ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    ⊢ MeasureTheory.Memℒp (MeasureTheory.stoppedValue u τ) p μ
  -/
  rw [stoppedValue_eq_of_mem_finset hbdd]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    p : ENNReal
    u : ι → Ω → E
    inst✝¹ : PartialOrder ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    ⊢ MeasureTheory.Memℒp (s.sum fun i => (setOf fun ω => Eq (τ ω) i).indicator (u …
  -/
  refine memℒp_finset_sum' _ fun i _ => Memℒp.indicator ?_ (hu i)
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    p : ENNReal
    u : ι → Ω → E
    inst✝¹ : PartialOrder ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    i : ι
    x✝ : Membership.mem s i
    ⊢ MeasurableSet (setOf fun ω => Eq (τ ω) i)
  -/
  refine ℱ.le i {a : Ω | τ a = i} (hτ.measurableSet_eq_of_countable_range ?_ i)
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    p : ENNReal
    u : ι → Ω → E
    inst✝¹ : PartialOrder ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    i : ι
    x✝ : Membership.mem s i
    ⊢ (Set.range τ).Countable
  -/
  refine ((Finset.finite_toSet s).subset fun ω hω => ?_).countable
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    p : ENNReal
    u : ι → Ω → E
    inst✝¹ : PartialOrder ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    i : ι
    x✝ : Membership.mem s i
    ω : ι
    hω : Membership.mem (Set.range τ) ω
    ⊢ Membership.mem (↑s) ω
  -/
  obtain ⟨y, rfl⟩ := hω
  /-
    case intro
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    p : ENNReal
    u : ι → Ω → E
    inst✝¹ : PartialOrder ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    i : ι
    x✝ : Membership.mem s i
    y : Ω
    ⊢ Membership.mem (↑s) (τ y)
  -/
  exact hbdd y
  /-
    🎉 no goals
  -/


theorem memℒp_stoppedValue [LocallyFiniteOrderBot ι] (hτ : IsStoppingTime ℱ τ)
    (hu : ∀ n, Memℒp (u n) p μ) {N : ι} (hbdd : ∀ ω, τ ω ≤ N) : Memℒp (stoppedValue u τ) p μ :=
  memℒp_stoppedValue_of_mem_finset hτ hu fun ω => Finset.mem_Iic.mpr (hbdd ω)


theorem integrable_stoppedValue_of_mem_finset (hτ : IsStoppingTime ℱ τ)
    (hu : ∀ n, Integrable (u n) μ) {s : Finset ι} (hbdd : ∀ ω, τ ω ∈ s) :
    Integrable (stoppedValue u τ) μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝¹ : PartialOrder ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Integrable (u n) μ
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    ⊢ MeasureTheory.Integrable (MeasureTheory.stoppedValue u τ) μ
  -/
  simp_rw [← memℒp_one_iff_integrable] at hu ⊢
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝¹ : PartialOrder ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    s : Finset ι
    hbdd : ∀ (ω : Ω), Membership.mem s (τ ω)
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) 1 μ
    ⊢ MeasureTheory.Memℒp (MeasureTheory.stoppedValue u τ) 1 μ
  -/
  exact memℒp_stoppedValue_of_mem_finset hτ hu hbdd
  /-
    🎉 no goals
  -/


theorem integrable_stoppedValue [LocallyFiniteOrderBot ι] (hτ : IsStoppingTime ℱ τ)
    (hu : ∀ n, Integrable (u n) μ) {N : ι} (hbdd : ∀ ω, τ ω ≤ N) :
    Integrable (stoppedValue u τ) μ :=
  integrable_stoppedValue_of_mem_finset hτ hu fun ω => Finset.mem_Iic.mpr (hbdd ω)


theorem memℒp_stoppedProcess_of_mem_finset (hτ : IsStoppingTime ℱ τ) (hu : ∀ n, Memℒp (u n) p μ)
    (n : ι) {s : Finset ι} (hbdd : ∀ ω, τ ω < n → τ ω ∈ s) : Memℒp (stoppedProcess u τ n) p μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    p : ENNReal
    u : ι → Ω → E
    inst✝⁴ : LinearOrder ι
    inst✝³ : TopologicalSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
    n : ι
    s : Finset ι
    hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
    ⊢ MeasureTheory.Memℒp (MeasureTheory.stoppedProcess u τ n) p μ
  -/
  rw [stoppedProcess_eq_of_mem_finset n hbdd]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    p : ENNReal
    u : ι → Ω → E
    inst✝⁴ : LinearOrder ι
    inst✝³ : TopologicalSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
    n : ι
    s : Finset ι
    hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
    ⊢ MeasureTheory.Memℒp (HAdd.hAdd ((setOf fun a => LE.le n (τ a)).indicator (u  …
  -/
  refine Memℒp.add ?_ ?_
    /-
      case refine_1
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      τ : Ω → ι
      E : Type u_4
      p : ENNReal
      u : ι → Ω → E
      inst✝⁴ : LinearOrder ι
      inst✝³ : TopologicalSpace ι
      inst✝² : OrderTopology ι
      inst✝¹ : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝ : NormedAddCommGroup E
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
      n : ι
      s : Finset ι
      hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
      ⊢ MeasureTheory.Memℒp ((setOf fun a => LE.le n (τ a)).indicator (u n)) p μ
    -/
  · exact Memℒp.indicator (ℱ.le n {a : Ω | n ≤ τ a} (hτ.measurableSet_ge n)) (hu n)
    /-
      🎉 no goals
    -/
  · suffices Memℒp (fun ω => ∑ i ∈ s.filter (· < n), {a : Ω | τ a = i}.indicator (u i) ω) p μ by
      convert this using 1; ext1 ω; simp only [Finset.sum_apply]
    /-
      case refine_2
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      τ : Ω → ι
      E : Type u_4
      p : ENNReal
      u : ι → Ω → E
      inst✝⁴ : LinearOrder ι
      inst✝³ : TopologicalSpace ι
      inst✝² : OrderTopology ι
      inst✝¹ : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝ : NormedAddCommGroup E
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
      n : ι
      s : Finset ι
      hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
      ⊢ MeasureTheory.Memℒp (fun ω => (Finset.filter (fun x => LT.lt x n) s).sum fun …
    -/
    refine memℒp_finset_sum _ fun i _ => Memℒp.indicator ?_ (hu i)
    /-
      case refine_2
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      τ : Ω → ι
      E : Type u_4
      p : ENNReal
      u : ι → Ω → E
      inst✝⁴ : LinearOrder ι
      inst✝³ : TopologicalSpace ι
      inst✝² : OrderTopology ι
      inst✝¹ : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝ : NormedAddCommGroup E
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) p μ
      n : ι
      s : Finset ι
      hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
      i : ι
      x✝ : Membership.mem (Finset.filter (fun x => LT.lt x n) s) i
      ⊢ MeasurableSet (setOf fun a => Eq (τ a) i)
    -/
    exact ℱ.le i {a : Ω | τ a = i} (hτ.measurableSet_eq i)
    /-
      🎉 no goals
    -/


theorem memℒp_stoppedProcess [LocallyFiniteOrderBot ι] (hτ : IsStoppingTime ℱ τ)
    (hu : ∀ n, Memℒp (u n) p μ) (n : ι) : Memℒp (stoppedProcess u τ n) p μ :=
  memℒp_stoppedProcess_of_mem_finset hτ hu n fun _ h => Finset.mem_Iio.mpr h


theorem integrable_stoppedProcess_of_mem_finset (hτ : IsStoppingTime ℱ τ)
    (hu : ∀ n, Integrable (u n) μ) (n : ι) {s : Finset ι} (hbdd : ∀ ω, τ ω < n → τ ω ∈ s) :
    Integrable (stoppedProcess u τ n) μ := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝⁴ : LinearOrder ι
    inst✝³ : TopologicalSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hu : ∀ (n : ι), MeasureTheory.Integrable (u n) μ
    n : ι
    s : Finset ι
    hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
    ⊢ MeasureTheory.Integrable (MeasureTheory.stoppedProcess u τ n) μ
  -/
  simp_rw [← memℒp_one_iff_integrable] at hu ⊢
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    τ : Ω → ι
    E : Type u_4
    u : ι → Ω → E
    inst✝⁴ : LinearOrder ι
    inst✝³ : TopologicalSpace ι
    inst✝² : OrderTopology ι
    inst✝¹ : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝ : NormedAddCommGroup E
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    n : ι
    s : Finset ι
    hbdd : ∀ (ω : Ω), LT.lt (τ ω) n → Membership.mem s (τ ω)
    hu : ∀ (n : ι), MeasureTheory.Memℒp (u n) 1 μ
    ⊢ MeasureTheory.Memℒp (MeasureTheory.stoppedProcess u τ n) 1 μ
  -/
  exact memℒp_stoppedProcess_of_mem_finset hτ hu n hbdd
  /-
    🎉 no goals
  -/


theorem integrable_stoppedProcess [LocallyFiniteOrderBot ι] (hτ : IsStoppingTime ℱ τ)
    (hu : ∀ n, Integrable (u n) μ) (n : ι) : Integrable (stoppedProcess u τ n) μ :=
  integrable_stoppedProcess_of_mem_finset hτ hu n fun _ h => Finset.mem_Iio.mpr h


/-- The stopped process of an adapted process with continuous paths is adapted. -/
theorem Adapted.stoppedProcess [MetrizableSpace ι] (hu : Adapted f u)
    (hu_cont : ∀ ω, Continuous fun i => u i ω) (hτ : IsStoppingTime f τ) :
    Adapted f (stoppedProcess u τ) :=
  ((hu.progMeasurable_of_continuous hu_cont).stoppedProcess hτ).adapted


/-- If the indexing order has the discrete topology, then the stopped process of an adapted process
is adapted. -/
theorem Adapted.stoppedProcess_of_discrete [DiscreteTopology ι] (hu : Adapted f u)
    (hτ : IsStoppingTime f τ) : Adapted f (MeasureTheory.stoppedProcess u τ) :=
  (hu.progMeasurable_of_discrete.stoppedProcess hτ).adapted


theorem Adapted.stronglyMeasurable_stoppedProcess [MetrizableSpace ι] (hu : Adapted f u)
    (hu_cont : ∀ ω, Continuous fun i => u i ω) (hτ : IsStoppingTime f τ) (n : ι) :
    StronglyMeasurable (MeasureTheory.stoppedProcess u τ n) :=
  (hu.progMeasurable_of_continuous hu_cont).stronglyMeasurable_stoppedProcess hτ n


theorem Adapted.stronglyMeasurable_stoppedProcess_of_discrete [DiscreteTopology ι]
    (hu : Adapted f u) (hτ : IsStoppingTime f τ) (n : ι) :
    StronglyMeasurable (MeasureTheory.stoppedProcess u τ n) :=
  hu.progMeasurable_of_discrete.stronglyMeasurable_stoppedProcess hτ n


theorem stoppedValue_sub_eq_sum [AddCommGroup β] (hle : τ ≤ π) :
    stoppedValue u π - stoppedValue u τ = fun ω =>
      (∑ i ∈ Finset.Ico (τ ω) (π ω), (u (i + 1) - u i)) ω := by
  /-
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    ⊢ Eq (HSub.hSub (MeasureTheory.stoppedValue u π) (MeasureTheory.stoppedValue u …
  -/
  ext ω
  /-
    case h
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    ω : Ω
    ⊢ Eq (HSub.hSub (MeasureTheory.stoppedValue u π) (MeasureTheory.stoppedValue u …
  -/
  rw [Finset.sum_Ico_eq_sub _ (hle ω), Finset.sum_range_sub, Finset.sum_range_sub]
  /-
    case h
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    ω : Ω
    ⊢ Eq (HSub.hSub (MeasureTheory.stoppedValue u π) (MeasureTheory.stoppedValue u …
  -/
  simp [stoppedValue]
  /-
    🎉 no goals
  -/


theorem stoppedValue_sub_eq_sum' [AddCommGroup β] (hle : τ ≤ π) {N : ℕ} (hbdd : ∀ ω, π ω ≤ N) :
    stoppedValue u π - stoppedValue u τ = fun ω =>
      (∑ i ∈ Finset.range (N + 1), Set.indicator {ω | τ ω ≤ i ∧ i < π ω} (u (i + 1) - u i)) ω := by
  /-
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    N : Nat
    hbdd : ∀ (ω : Ω), LE.le (π ω) N
    ⊢ Eq (HSub.hSub (MeasureTheory.stoppedValue u π) (MeasureTheory.stoppedValue u …
  -/
  rw [stoppedValue_sub_eq_sum hle]
  /-
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    N : Nat
    hbdd : ∀ (ω : Ω), LE.le (π ω) N
    ⊢ Eq (fun ω => (Finset.Ico (τ ω) (π ω)).sum (fun i => HSub.hSub (u (HAdd.hAdd  …
  -/
  ext ω
  /-
    case h
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    N : Nat
    hbdd : ∀ (ω : Ω), LE.le (π ω) N
    ω : Ω
    ⊢ Eq ((Finset.Ico (τ ω) (π ω)).sum (fun i => HSub.hSub (u (HAdd.hAdd i 1)) (u  …
  -/
  simp only [Finset.sum_apply, Finset.sum_indicator_eq_sum_filter]
  /-
    case h
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    N : Nat
    hbdd : ∀ (ω : Ω), LE.le (π ω) N
    ω : Ω
    ⊢ Eq ((Finset.Ico (τ ω) (π ω)).sum fun c => HSub.hSub (u (HAdd.hAdd c 1)) (u c …
  -/
  refine Finset.sum_congr ?_ fun _ _ => rfl
  /-
    case h
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    N : Nat
    hbdd : ∀ (ω : Ω), LE.le (π ω) N
    ω : Ω
    ⊢ Eq (Finset.Ico (τ ω) (π ω)) (Finset.filter (fun i => Membership.mem (setOf f …
  -/
  ext i
  /-
    case h.h
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    N : Nat
    hbdd : ∀ (ω : Ω), LE.le (π ω) N
    ω : Ω
    i : Nat
    ⊢ Iff (Membership.mem (Finset.Ico (τ ω) (π ω)) i) (Membership.mem (Finset.filt …
  -/
  simp only [Finset.mem_filter, Set.mem_setOf_eq, Finset.mem_range, Finset.mem_Ico]
  /-
    case h.h
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ π : Ω → Nat
    inst✝ : AddCommGroup β
    hle : LE.le τ π
    N : Nat
    hbdd : ∀ (ω : Ω), LE.le (π ω) N
    ω : Ω
    i : Nat
    ⊢ Iff (And (LE.le (τ ω) i) (LT.lt i (π ω))) (And (LT.lt i (HAdd.hAdd N 1)) (An …
  -/
  exact ⟨fun h => ⟨lt_trans h.2 (Nat.lt_succ_iff.2 <| hbdd _), h⟩, fun h => h.2⟩
  /-
    🎉 no goals
  -/


theorem stoppedValue_eq {N : ℕ} (hbdd : ∀ ω, τ ω ≤ N) : stoppedValue u τ = fun x =>
    (∑ i ∈ Finset.range (N + 1), Set.indicator {ω | τ ω = i} (u i)) x :=
  stoppedValue_eq_of_mem_finset fun ω => Finset.mem_range_succ_iff.mpr (hbdd ω)


theorem stoppedProcess_eq (n : ℕ) : stoppedProcess u τ n = Set.indicator {a | n ≤ τ a} (u n) +
    ∑ i ∈ Finset.range n, Set.indicator {ω | τ ω = i} (u i) := by
  /-
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ : Ω → Nat
    inst✝ : AddCommMonoid β
    n : Nat
    ⊢ Eq (MeasureTheory.stoppedProcess u τ n) (HAdd.hAdd ((setOf fun a => LE.le n  …
  -/
  rw [stoppedProcess_eq'' n]
  /-
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ : Ω → Nat
    inst✝ : AddCommMonoid β
    n : Nat
    ⊢ Eq (HAdd.hAdd ((setOf fun a => LE.le n (τ a)).indicator (u n)) ((Finset.Iio  …
  -/
  congr with i
  /-
    case e_a.e_s.h
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ : Ω → Nat
    inst✝ : AddCommMonoid β
    n i : Nat
    ⊢ Iff (Membership.mem (Finset.Iio n) i) (Membership.mem (Finset.range n) i)
  -/
  rw [Finset.mem_Iio, Finset.mem_range]
  /-
    🎉 no goals
  -/


theorem stoppedProcess_eq' (n : ℕ) : stoppedProcess u τ n = Set.indicator {a | n + 1 ≤ τ a} (u n) +
    ∑ i ∈ Finset.range (n + 1), Set.indicator {a | τ a = i} (u i) := by
  have : {a | n ≤ τ a}.indicator (u n) =
      {a | n + 1 ≤ τ a}.indicator (u n) + {a | τ a = n}.indicator (u n) := by
    ext x
    rw [add_comm, Pi.add_apply, ← Set.indicator_union_of_not_mem_inter]
    · simp_rw [@eq_comm _ _ n, @le_iff_eq_or_lt _ _ n, Nat.succ_le_iff, Set.setOf_or]
    · rintro ⟨h₁, h₂⟩
      rw [Set.mem_setOf] at h₁ h₂
      exact (Nat.succ_le_iff.1 h₂).ne h₁.symm
  /-
    Ω : Type u_1
    β : Type u_2
    u : Nat → Ω → β
    τ : Ω → Nat
    inst✝ : AddCommMonoid β
    n : Nat
    this : Eq ((setOf fun a => LE.le n (τ a)).indicator (u n)) (HAdd.hAdd ((setOf  …
    ⊢ Eq (MeasureTheory.stoppedProcess u τ n) (HAdd.hAdd ((setOf fun a => LE.le (H …
  -/
  rw [stoppedProcess_eq, this, Finset.sum_range_succ_comm, ← add_assoc]
  /-
    🎉 no goals
  -/


/-- Given stopping times `τ` and `η` which are bounded below, `Set.piecewise s τ η` is also
a stopping time with respect to the same filtration. -/
theorem IsStoppingTime.piecewise_of_le (hτ_st : IsStoppingTime 𝒢 τ) (hη_st : IsStoppingTime 𝒢 η)
    (hτ : ∀ ω, i ≤ τ ω) (hη : ∀ ω, i ≤ η ω) (hs : MeasurableSet[𝒢 i] s) :
    IsStoppingTime 𝒢 (s.piecewise τ η) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    𝒢 : MeasureTheory.Filtration ι m
    τ η : Ω → ι
    i : ι
    s : Set Ω
    inst✝ : DecidablePred fun x => Membership.mem s x
    hτ_st : MeasureTheory.IsStoppingTime 𝒢 τ
    hη_st : MeasureTheory.IsStoppingTime 𝒢 η
    hτ : ∀ (ω : Ω), LE.le i (τ ω)
    hη : ∀ (ω : Ω), LE.le i (η ω)
    hs : MeasurableSet s
    ⊢ MeasureTheory.IsStoppingTime 𝒢 (s.piecewise τ η)
  -/
  intro n
  have : {ω | s.piecewise τ η ω ≤ n} = s ∩ {ω | τ ω ≤ n} ∪ sᶜ ∩ {ω | η ω ≤ n} := by
    ext1 ω
    simp only [Set.piecewise, Set.mem_inter_iff, Set.mem_setOf_eq, and_congr_right_iff]
    by_cases hx : ω ∈ s <;> simp [hx]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    𝒢 : MeasureTheory.Filtration ι m
    τ η : Ω → ι
    i : ι
    s : Set Ω
    inst✝ : DecidablePred fun x => Membership.mem s x
    hτ_st : MeasureTheory.IsStoppingTime 𝒢 τ
    hη_st : MeasureTheory.IsStoppingTime 𝒢 η
    hτ : ∀ (ω : Ω), LE.le i (τ ω)
    hη : ∀ (ω : Ω), LE.le i (η ω)
    hs : MeasurableSet s
    n : ι
    this : Eq (setOf fun ω => LE.le (s.piecewise τ η ω) n) (Union.union (Inter.int …
    ⊢ MeasurableSet (setOf fun ω => LE.le (s.piecewise τ η ω) n)
  -/
  rw [this]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹ : Preorder ι
    𝒢 : MeasureTheory.Filtration ι m
    τ η : Ω → ι
    i : ι
    s : Set Ω
    inst✝ : DecidablePred fun x => Membership.mem s x
    hτ_st : MeasureTheory.IsStoppingTime 𝒢 τ
    hη_st : MeasureTheory.IsStoppingTime 𝒢 η
    hτ : ∀ (ω : Ω), LE.le i (τ ω)
    hη : ∀ (ω : Ω), LE.le i (η ω)
    hs : MeasurableSet s
    n : ι
    this : Eq (setOf fun ω => LE.le (s.piecewise τ η ω) n) (Union.union (Inter.int …
    ⊢ MeasurableSet (Union.union (Inter.inter s (setOf fun ω => LE.le (τ ω) n)) (I …
  -/
  by_cases hin : i ≤ n
    /-
      case pos
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      𝒢 : MeasureTheory.Filtration ι m
      τ η : Ω → ι
      i : ι
      s : Set Ω
      inst✝ : DecidablePred fun x => Membership.mem s x
      hτ_st : MeasureTheory.IsStoppingTime 𝒢 τ
      hη_st : MeasureTheory.IsStoppingTime 𝒢 η
      hτ : ∀ (ω : Ω), LE.le i (τ ω)
      hη : ∀ (ω : Ω), LE.le i (η ω)
      hs : MeasurableSet s
      n : ι
      this : Eq (setOf fun ω => LE.le (s.piecewise τ η ω) n) (Union.union (Inter.int …
      hin : LE.le i n
      ⊢ MeasurableSet (Union.union (Inter.inter s (setOf fun ω => LE.le (τ ω) n)) (I …
    -/
  · have hs_n : MeasurableSet[𝒢 n] s := 𝒢.mono hin _ hs
    /-
      case pos
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      𝒢 : MeasureTheory.Filtration ι m
      τ η : Ω → ι
      i : ι
      s : Set Ω
      inst✝ : DecidablePred fun x => Membership.mem s x
      hτ_st : MeasureTheory.IsStoppingTime 𝒢 τ
      hη_st : MeasureTheory.IsStoppingTime 𝒢 η
      hτ : ∀ (ω : Ω), LE.le i (τ ω)
      hη : ∀ (ω : Ω), LE.le i (η ω)
      hs : MeasurableSet s
      n : ι
      this : Eq (setOf fun ω => LE.le (s.piecewise τ η ω) n) (Union.union (Inter.int …
      hin : LE.le i n
      hs_n : MeasurableSet s
      ⊢ MeasurableSet (Union.union (Inter.inter s (setOf fun ω => LE.le (τ ω) n)) (I …
    -/
    exact (hs_n.inter (hτ_st n)).union (hs_n.compl.inter (hη_st n))
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      𝒢 : MeasureTheory.Filtration ι m
      τ η : Ω → ι
      i : ι
      s : Set Ω
      inst✝ : DecidablePred fun x => Membership.mem s x
      hτ_st : MeasureTheory.IsStoppingTime 𝒢 τ
      hη_st : MeasureTheory.IsStoppingTime 𝒢 η
      hτ : ∀ (ω : Ω), LE.le i (τ ω)
      hη : ∀ (ω : Ω), LE.le i (η ω)
      hs : MeasurableSet s
      n : ι
      this : Eq (setOf fun ω => LE.le (s.piecewise τ η ω) n) (Union.union (Inter.int …
      hin : Not (LE.le i n)
      ⊢ MeasurableSet (Union.union (Inter.inter s (setOf fun ω => LE.le (τ ω) n)) (I …
    -/
  · have hτn : ∀ ω, ¬τ ω ≤ n := fun ω hτn => hin ((hτ ω).trans hτn)
    /-
      case neg
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      𝒢 : MeasureTheory.Filtration ι m
      τ η : Ω → ι
      i : ι
      s : Set Ω
      inst✝ : DecidablePred fun x => Membership.mem s x
      hτ_st : MeasureTheory.IsStoppingTime 𝒢 τ
      hη_st : MeasureTheory.IsStoppingTime 𝒢 η
      hτ : ∀ (ω : Ω), LE.le i (τ ω)
      hη : ∀ (ω : Ω), LE.le i (η ω)
      hs : MeasurableSet s
      n : ι
      this : Eq (setOf fun ω => LE.le (s.piecewise τ η ω) n) (Union.union (Inter.int …
      hin : Not (LE.le i n)
      hτn : ∀ (ω : Ω), Not (LE.le (τ ω) n)
      ⊢ MeasurableSet (Union.union (Inter.inter s (setOf fun ω => LE.le (τ ω) n)) (I …
    -/
    have hηn : ∀ ω, ¬η ω ≤ n := fun ω hηn => hin ((hη ω).trans hηn)
    /-
      case neg
      Ω : Type u_1
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝¹ : Preorder ι
      𝒢 : MeasureTheory.Filtration ι m
      τ η : Ω → ι
      i : ι
      s : Set Ω
      inst✝ : DecidablePred fun x => Membership.mem s x
      hτ_st : MeasureTheory.IsStoppingTime 𝒢 τ
      hη_st : MeasureTheory.IsStoppingTime 𝒢 η
      hτ : ∀ (ω : Ω), LE.le i (τ ω)
      hη : ∀ (ω : Ω), LE.le i (η ω)
      hs : MeasurableSet s
      n : ι
      this : Eq (setOf fun ω => LE.le (s.piecewise τ η ω) n) (Union.union (Inter.int …
      hin : Not (LE.le i n)
      hτn : ∀ (ω : Ω), Not (LE.le (τ ω) n)
      hηn : ∀ (ω : Ω), Not (LE.le (η ω) n)
      ⊢ MeasurableSet (Union.union (Inter.inter s (setOf fun ω => LE.le (τ ω) n)) (I …
    -/
    simp [hτn, hηn, @MeasurableSet.empty _ _]
    /-
      🎉 no goals
    -/


theorem isStoppingTime_piecewise_const (hij : i ≤ j) (hs : MeasurableSet[𝒢 i] s) :
    IsStoppingTime 𝒢 (s.piecewise (fun _ => i) fun _ => j) :=
  (isStoppingTime_const 𝒢 i).piecewise_of_le (isStoppingTime_const 𝒢 j) (fun _ => le_rfl)
    (fun _ => hij) hs


theorem stoppedValue_piecewise_const {ι' : Type*} {i j : ι'} {f : ι' → Ω → ℝ} :
    stoppedValue f (s.piecewise (fun _ => i) fun _ => j) = s.piecewise (f i) (f j) := by
  /-
    Ω : Type u_1
    s : Set Ω
    inst✝ : DecidablePred fun x => Membership.mem s x
    ι' : Type u_4
    i j : ι'
    f : ι' → Ω → Real
    ⊢ Eq (MeasureTheory.stoppedValue f (s.piecewise (fun x => i) fun x => j)) (s.p …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  ext ω; rw [stoppedValue]; by_cases hx : ω ∈ s <;> simp [hx]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem stoppedValue_piecewise_const' {ι' : Type*} {i j : ι'} {f : ι' → Ω → ℝ} :
    stoppedValue f (s.piecewise (fun _ => i) fun _ => j) =
    s.indicator (f i) + sᶜ.indicator (f j) := by
  /-
    Ω : Type u_1
    s : Set Ω
    inst✝ : DecidablePred fun x => Membership.mem s x
    ι' : Type u_4
    i j : ι'
    f : ι' → Ω → Real
    ⊢ Eq (MeasureTheory.stoppedValue f (s.piecewise (fun x => i) fun x => j)) (HAd …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  ext ω; rw [stoppedValue]; by_cases hx : ω ∈ s <;> simp [hx]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem condexp_stopping_time_ae_eq_restrict_eq_of_countable_range [SigmaFiniteFiltration μ ℱ]
    (hτ : IsStoppingTime ℱ τ) (h_countable : (Set.range τ).Countable)
    [SigmaFinite (μ.trim (hτ.measurableSpace_le_of_countable_range h_countable))] (i : ι) :
    μ[f|hτ.measurableSpace] =ᵐ[μ.restrict {x | τ x = i}] μ[f|ℱ i] := by
  refine condexp_ae_eq_restrict_of_measurableSpace_eq_on
    (hτ.measurableSpace_le_of_countable_range h_countable) (ℱ.le i)
    (hτ.measurableSet_eq_of_countable_range' h_countable i) fun t => ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m
    τ : Ω → ι
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : Ω → E
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    h_countable : (Set.range τ).Countable
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    i : ι
    t : Set Ω
    ⊢ Iff (MeasurableSet (Inter.inter (setOf fun x => Eq (τ x) i) t)) (MeasurableS …
  -/
  rw [Set.inter_comm _ t, IsStoppingTime.measurableSet_inter_eq_iff]
  /-
    🎉 no goals
  -/


theorem condexp_stopping_time_ae_eq_restrict_eq_of_countable [Countable ι]
    [SigmaFiniteFiltration μ ℱ] (hτ : IsStoppingTime ℱ τ)
    [SigmaFinite (μ.trim hτ.measurableSpace_le_of_countable)] (i : ι) :
    μ[f|hτ.measurableSpace] =ᵐ[μ.restrict {x | τ x = i}] μ[f|ℱ i] :=
  condexp_stopping_time_ae_eq_restrict_eq_of_countable_range hτ (Set.to_countable _) i


theorem condexp_min_stopping_time_ae_eq_restrict_le_const (hτ : IsStoppingTime ℱ τ) (i : ι)
    [SigmaFinite (μ.trim (hτ.min_const i).measurableSpace_le)] :
    μ[f|(hτ.min_const i).measurableSpace] =ᵐ[μ.restrict {x | τ x ≤ i}] μ[f|hτ.measurableSpace] := by
  have : SigmaFinite (μ.trim hτ.measurableSpace_le) :=
    haveI h_le : (hτ.min_const i).measurableSpace ≤ hτ.measurableSpace := by
      rw [IsStoppingTime.measurableSpace_min_const]
      exact inf_le_left
    sigmaFiniteTrim_mono _ h_le
  refine (condexp_ae_eq_restrict_of_measurableSpace_eq_on hτ.measurableSpace_le
    (hτ.min_const i).measurableSpace_le (hτ.measurableSet_le' i) fun t => ?_).symm
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : LinearOrder ι
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m
    τ : Ω → ι
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    f : Ω → E
    inst✝¹ : Filter.atTop.IsCountablyGenerated
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    i : ι
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    this : MeasureTheory.SigmaFinite (μ.trim ⋯)
    t : Set Ω
    ⊢ Iff (MeasurableSet (Inter.inter (setOf fun ω => LE.le (τ ω) i) t)) (Measurab …
  -/
  rw [Set.inter_comm _ t, hτ.measurableSet_inter_le_const_iff]
  /-
    🎉 no goals
  -/


theorem condexp_stopping_time_ae_eq_restrict_eq [FirstCountableTopology ι]
    [SigmaFiniteFiltration μ ℱ] (hτ : IsStoppingTime ℱ τ)
    [SigmaFinite (μ.trim hτ.measurableSpace_le)] (i : ι) :
    μ[f|hτ.measurableSpace] =ᵐ[μ.restrict {x | τ x = i}] μ[f|ℱ i] := by
  refine condexp_ae_eq_restrict_of_measurableSpace_eq_on hτ.measurableSpace_le (ℱ.le i)
    (hτ.measurableSet_eq' i) fun t => ?_
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁹ : LinearOrder ι
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m
    τ : Ω → ι
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    f : Ω → E
    inst✝⁵ : Filter.atTop.IsCountablyGenerated
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    i : ι
    t : Set Ω
    ⊢ Iff (MeasurableSet (Inter.inter (setOf fun x => Eq (τ x) i) t)) (MeasurableS …
  -/
  rw [Set.inter_comm _ t, IsStoppingTime.measurableSet_inter_eq_iff]
  /-
    🎉 no goals
  -/


theorem condexp_min_stopping_time_ae_eq_restrict_le [MeasurableSpace ι] [SecondCountableTopology ι]
    [BorelSpace ι] (hτ : IsStoppingTime ℱ τ) (hσ : IsStoppingTime ℱ σ)
    [SigmaFinite (μ.trim (hτ.min hσ).measurableSpace_le)] :
    μ[f|(hτ.min hσ).measurableSpace] =ᵐ[μ.restrict {x | τ x ≤ σ x}] μ[f|hτ.measurableSpace] := by
  have : SigmaFinite (μ.trim hτ.measurableSpace_le) :=
    haveI h_le : (hτ.min hσ).measurableSpace ≤ hτ.measurableSpace := by
      rw [IsStoppingTime.measurableSpace_min]
      · exact inf_le_left
      · simp_all only
    sigmaFiniteTrim_mono _ h_le
  refine (condexp_ae_eq_restrict_of_measurableSpace_eq_on hτ.measurableSpace_le
    (hτ.min hσ).measurableSpace_le (hτ.measurableSet_le_stopping_time hσ) fun t => ?_).symm
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝¹⁰ : LinearOrder ι
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration ι m
    τ σ : Ω → ι
    E : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    inst✝⁷ : CompleteSpace E
    f : Ω → E
    inst✝⁶ : Filter.atTop.IsCountablyGenerated
    inst✝⁵ : TopologicalSpace ι
    inst✝⁴ : OrderTopology ι
    inst✝³ : MeasurableSpace ι
    inst✝² : SecondCountableTopology ι
    inst✝¹ : BorelSpace ι
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hσ : MeasureTheory.IsStoppingTime ℱ σ
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    this : MeasureTheory.SigmaFinite (μ.trim ⋯)
    t : Set Ω
    ⊢ Iff (MeasurableSet (Inter.inter (setOf fun ω => LE.le (τ ω) (σ ω)) t)) (Meas …
  -/
  rw [Set.inter_comm _ t, IsStoppingTime.measurableSet_inter_le_iff]; simp_all only
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


