/-- A set `s` is Carathéodory-measurable for an outer measure `m` if for all sets `t` we have
  `m t = m (t ∩ s) + m (t \ s)`. -/
def IsCaratheodory (s : Set α) : Prop :=
  ∀ t, m t = m (t ∩ s) + m (t \ s)


theorem isCaratheodory_iff_le' {s : Set α} :
    IsCaratheodory m s ↔ ∀ t, m (t ∩ s) + m (t \ s) ≤ m t :=
  forall_congr' fun _ => le_antisymm_iff.trans <| and_iff_right <| measure_le_inter_add_diff _ _ _


@[simp]
                                                        /-
                                                          α : Type u
                                                          m : MeasureTheory.OuterMeasure α
                                                          ⊢ m.IsCaratheodory EmptyCollection.emptyCollection
                                                        -/
theorem isCaratheodory_empty : IsCaratheodory m ∅ := by simp [IsCaratheodory, m.empty, diff_empty]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem isCaratheodory_compl : IsCaratheodory m s₁ → IsCaratheodory m s₁ᶜ := by
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s₁ : Set α
    ⊢ m.IsCaratheodory s₁ → m.IsCaratheodory (HasCompl.compl s₁)
  -/
  simp [IsCaratheodory, diff_eq, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem isCaratheodory_compl_iff : IsCaratheodory m sᶜ ↔ IsCaratheodory m s :=
               /-
                 α : Type u
                 m : MeasureTheory.OuterMeasure α
                 s : Set α
                 h : m.IsCaratheodory (HasCompl.compl s)
                 ⊢ m.IsCaratheodory s
               -/
  ⟨fun h => by simpa using isCaratheodory_compl m h, isCaratheodory_compl m⟩
               /-
                 🎉 no goals
               -/


theorem isCaratheodory_union (h₁ : IsCaratheodory m s₁) (h₂ : IsCaratheodory m s₂) :
    IsCaratheodory m (s₁ ∪ s₂) := fun t => by
  rw [h₁ t, h₂ (t ∩ s₁), h₂ (t \ s₁), h₁ (t ∩ (s₁ ∪ s₂)), inter_diff_assoc _ _ s₁,
    Set.inter_assoc _ _ s₁, inter_eq_self_of_subset_right Set.subset_union_left,
    union_diff_left, h₂ (t ∩ s₁)]
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s₁ s₂ : Set α
    h₁ : m.IsCaratheodory s₁
    h₂ : m.IsCaratheodory s₂
    t : Set α
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (m (Inter.inter (Inter.inter t s₁) s₂)) (m (SDiff.s …
  -/
  simp [diff_eq, add_assoc]
  /-
    🎉 no goals
  -/


theorem measure_inter_union (h : s₁ ∩ s₂ ⊆ ∅) (h₁ : IsCaratheodory m s₁) {t : Set α} :
    m (t ∩ (s₁ ∪ s₂)) = m (t ∩ s₁) + m (t ∩ s₂) := by
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s₁ s₂ : Set α
    h : HasSubset.Subset (Inter.inter s₁ s₂) EmptyCollection.emptyCollection
    h₁ : m.IsCaratheodory s₁
    t : Set α
    ⊢ Eq (m (Inter.inter t (Union.union s₁ s₂))) (HAdd.hAdd (m (Inter.inter t s₁)) …
  -/
  rw [h₁, Set.inter_assoc, Set.union_inter_cancel_left, inter_diff_assoc, union_diff_cancel_left h]
  /-
    🎉 no goals
  -/


theorem isCaratheodory_iUnion_lt {s : ℕ → Set α} :
    ∀ {n : ℕ}, (∀ i < n, IsCaratheodory m (s i)) → IsCaratheodory m (⋃ i < n, s i)
               /-
                 α : Type u
                 m : MeasureTheory.OuterMeasure α
                 s : Nat → Set α
                 x✝ : ∀ (i : Nat), LT.lt i 0 → m.IsCaratheodory (s i)
                 ⊢ m.IsCaratheodory (Set.iUnion fun i => Set.iUnion fun h => s i)
               -/
  | 0, _ => by simp [Nat.not_lt_zero]
               /-
                 🎉 no goals
               -/
  | n + 1, h => by
    /-
      α : Type u
      m : MeasureTheory.OuterMeasure α
      s : Nat → Set α
      n : Nat
      h : ∀ (i : Nat), LT.lt i (HAdd.hAdd n 1) → m.IsCaratheodory (s i)
      ⊢ m.IsCaratheodory (Set.iUnion fun i => Set.iUnion fun h => s i)
    -/
    rw [biUnion_lt_succ]
    exact isCaratheodory_union m
            (isCaratheodory_iUnion_lt fun i hi => h i <| lt_of_lt_of_le hi <| Nat.le_succ _)
            (h n (le_refl (n + 1)))


theorem isCaratheodory_inter (h₁ : IsCaratheodory m s₁) (h₂ : IsCaratheodory m s₂) :
    IsCaratheodory m (s₁ ∩ s₂) := by
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s₁ s₂ : Set α
    h₁ : m.IsCaratheodory s₁
    h₂ : m.IsCaratheodory s₂
    ⊢ m.IsCaratheodory (Inter.inter s₁ s₂)
  -/
  rw [← isCaratheodory_compl_iff, Set.compl_inter]
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s₁ s₂ : Set α
    h₁ : m.IsCaratheodory s₁
    h₂ : m.IsCaratheodory s₂
    ⊢ m.IsCaratheodory (Union.union (HasCompl.compl s₁) (HasCompl.compl s₂))
  -/
  exact isCaratheodory_union _ (isCaratheodory_compl _ h₁) (isCaratheodory_compl _ h₂)
  /-
    🎉 no goals
  -/


lemma isCaratheodory_diff (h₁ : IsCaratheodory m s₁) (h₂ : IsCaratheodory m s₂) :
    IsCaratheodory m (s₁ \ s₂) := m.isCaratheodory_inter h₁ (m.isCaratheodory_compl h₂)


lemma isCaratheodory_partialSups {s : ℕ → Set α} (h : ∀ i, m.IsCaratheodory (s i)) (i : ℕ) :
    m.IsCaratheodory (partialSups s i) := by
  induction i with
  | zero => exact h 0
  | succ i hi => exact m.isCaratheodory_union hi (h (i + 1))


lemma isCaratheodory_disjointed {s : ℕ → Set α} (h : ∀ i, m.IsCaratheodory (s i)) (i : ℕ) :
    m.IsCaratheodory (disjointed s i) := by
  induction i with
  | zero => exact h 0
  | succ i _ => exact m.isCaratheodory_diff (h (i + 1)) (m.isCaratheodory_partialSups h i)


theorem isCaratheodory_sum {s : ℕ → Set α} (h : ∀ i, IsCaratheodory m (s i))
    (hd : Pairwise (Disjoint on s)) {t : Set α} :
    ∀ {n}, (∑ i ∈ Finset.range n, m (t ∩ s i)) = m (t ∩ ⋃ i < n, s i)
            /-
              α : Type u
              m : MeasureTheory.OuterMeasure α
              s : Nat → Set α
              h : ∀ (i : Nat), m.IsCaratheodory (s i)
              hd : Pairwise (Function.onFun Disjoint s)
              t : Set α
              ⊢ Eq ((Finset.range 0).sum fun i => m (Inter.inter t (s i))) (m (Inter.inter t …
            -/
  | 0 => by simp [Nat.not_lt_zero, m.empty]
            /-
              🎉 no goals
            -/
  | Nat.succ n => by
    rw [biUnion_lt_succ, Finset.sum_range_succ, Set.union_comm, isCaratheodory_sum h hd,
      m.measure_inter_union _ (h n), add_comm]
    /-
      α : Type u
      m : MeasureTheory.OuterMeasure α
      s : Nat → Set α
      h : ∀ (i : Nat), m.IsCaratheodory (s i)
      hd : Pairwise (Function.onFun Disjoint s)
      t : Set α
      n : Nat
      ⊢ HasSubset.Subset (Inter.inter (s n) (Set.iUnion fun k => Set.iUnion fun h => …
    -/
    intro a
    /-
      α : Type u
      m : MeasureTheory.OuterMeasure α
      s : Nat → Set α
      h : ∀ (i : Nat), m.IsCaratheodory (s i)
      hd : Pairwise (Function.onFun Disjoint s)
      t : Set α
      n : Nat
      a : α
      ⊢ Membership.mem (Inter.inter (s n) (Set.iUnion fun k => Set.iUnion fun h => s …
    -/
    simpa using fun (h₁ : a ∈ s n) i (hi : i < n) h₂ => (hd (ne_of_gt hi)).le_bot ⟨h₁, h₂⟩
    /-
      🎉 no goals
    -/


/-- Use `isCaratheodory_iUnion` instead, which does not require the disjoint assumption. -/
theorem isCaratheodory_iUnion_of_disjoint {s : ℕ → Set α} (h : ∀ i, IsCaratheodory m (s i))
    (hd : Pairwise (Disjoint on s)) : IsCaratheodory m (⋃ i, s i) := by
      /-
        α : Type u
        m : MeasureTheory.OuterMeasure α
        s : Nat → Set α
        h : ∀ (i : Nat), m.IsCaratheodory (s i)
        hd : Pairwise (Function.onFun Disjoint s)
        ⊢ m.IsCaratheodory (Set.iUnion fun i => s i)
      -/
      apply (isCaratheodory_iff_le' m).mpr
      /-
        α : Type u
        m : MeasureTheory.OuterMeasure α
        s : Nat → Set α
        h : ∀ (i : Nat), m.IsCaratheodory (s i)
        hd : Pairwise (Function.onFun Disjoint s)
        ⊢ ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t (Set.iUnion fun i => s i)) …
      -/
      intro t
      have hp : m (t ∩ ⋃ i, s i) ≤ ⨆ n, m (t ∩ ⋃ i < n, s i) := by
        convert measure_iUnion_le (μ := m) fun i => t ∩ s i using 1
        · simp [inter_iUnion]
        · simp [ENNReal.tsum_eq_iSup_nat, isCaratheodory_sum m h hd]
      /-
        α : Type u
        m : MeasureTheory.OuterMeasure α
        s : Nat → Set α
        h : ∀ (i : Nat), m.IsCaratheodory (s i)
        hd : Pairwise (Function.onFun Disjoint s)
        t : Set α
        hp : LE.le (m (Inter.inter t (Set.iUnion fun i => s i))) (iSup fun n => m (Int …
        ⊢ LE.le (HAdd.hAdd (m (Inter.inter t (Set.iUnion fun i => s i))) (m (SDiff.sdi …
      -/
      refine le_trans (add_le_add_right hp _) ?_
      /-
        α : Type u
        m : MeasureTheory.OuterMeasure α
        s : Nat → Set α
        h : ∀ (i : Nat), m.IsCaratheodory (s i)
        hd : Pairwise (Function.onFun Disjoint s)
        t : Set α
        hp : LE.le (m (Inter.inter t (Set.iUnion fun i => s i))) (iSup fun n => m (Int …
        ⊢ LE.le (HAdd.hAdd (iSup fun n => m (Inter.inter t (Set.iUnion fun i => Set.iU …
      -/
      rw [ENNReal.iSup_add]
      refine iSup_le fun n => le_trans (add_le_add_left ?_ _)
        (ge_of_eq (isCaratheodory_iUnion_lt m (fun i _ => h i) _))
      /-
        α : Type u
        m : MeasureTheory.OuterMeasure α
        s : Nat → Set α
        h : ∀ (i : Nat), m.IsCaratheodory (s i)
        hd : Pairwise (Function.onFun Disjoint s)
        t : Set α
        hp : LE.le (m (Inter.inter t (Set.iUnion fun i => s i))) (iSup fun n => m (Int …
        n : Nat
        ⊢ LE.le (m (SDiff.sdiff t (Set.iUnion fun i => s i))) (m (SDiff.sdiff t (Set.i …
      -/
      refine m.mono (diff_subset_diff_right ?_)
      /-
        α : Type u
        m : MeasureTheory.OuterMeasure α
        s : Nat → Set α
        h : ∀ (i : Nat), m.IsCaratheodory (s i)
        hd : Pairwise (Function.onFun Disjoint s)
        t : Set α
        hp : LE.le (m (Inter.inter t (Set.iUnion fun i => s i))) (iSup fun n => m (Int …
        n : Nat
        ⊢ HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => s i) (Set.iUnion f …
      -/
      exact iUnion₂_subset fun i _ => subset_iUnion _ i
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-07-29")]
alias isCaratheodory_iUnion_nat := isCaratheodory_iUnion_of_disjoint


lemma isCaratheodory_iUnion {s : ℕ → Set α} (h : ∀ i, m.IsCaratheodory (s i)) :
    m.IsCaratheodory (⋃ i, s i) := by
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s : Nat → Set α
    h : ∀ (i : Nat), m.IsCaratheodory (s i)
    ⊢ m.IsCaratheodory (Set.iUnion fun i => s i)
  -/
  rw [← iUnion_disjointed]
  exact m.isCaratheodory_iUnion_of_disjoint (m.isCaratheodory_disjointed h)
    (disjoint_disjointed _)


theorem f_iUnion {s : ℕ → Set α} (h : ∀ i, IsCaratheodory m (s i)) (hd : Pairwise (Disjoint on s)) :
    m (⋃ i, s i) = ∑' i, m (s i) := by
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s : Nat → Set α
    h : ∀ (i : Nat), m.IsCaratheodory (s i)
    hd : Pairwise (Function.onFun Disjoint s)
    ⊢ Eq (m (Set.iUnion fun i => s i)) (tsum fun i => m (s i))
  -/
  refine le_antisymm (measure_iUnion_le s) ?_
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s : Nat → Set α
    h : ∀ (i : Nat), m.IsCaratheodory (s i)
    hd : Pairwise (Function.onFun Disjoint s)
    ⊢ LE.le (tsum fun i => m (s i)) (m (Set.iUnion fun i => s i))
  -/
  rw [ENNReal.tsum_eq_iSup_nat]
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s : Nat → Set α
    h : ∀ (i : Nat), m.IsCaratheodory (s i)
    hd : Pairwise (Function.onFun Disjoint s)
    ⊢ LE.le (iSup fun i => (Finset.range i).sum fun a => m (s a)) (m (Set.iUnion f …
  -/
  refine iSup_le fun n => ?_
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s : Nat → Set α
    h : ∀ (i : Nat), m.IsCaratheodory (s i)
    hd : Pairwise (Function.onFun Disjoint s)
    n : Nat
    ⊢ LE.le ((Finset.range n).sum fun a => m (s a)) (m (Set.iUnion fun i => s i))
  -/
  have := @isCaratheodory_sum _ m _ h hd univ n
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s : Nat → Set α
    h : ∀ (i : Nat), m.IsCaratheodory (s i)
    hd : Pairwise (Function.onFun Disjoint s)
    n : Nat
    this : Eq ((Finset.range n).sum fun i => m (Inter.inter Set.univ (s i))) (m (I …
    ⊢ LE.le ((Finset.range n).sum fun a => m (s a)) (m (Set.iUnion fun i => s i))
  -/
  simp only [inter_comm, inter_univ, univ_inter] at this; simp only [this]
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s : Nat → Set α
    h : ∀ (i : Nat), m.IsCaratheodory (s i)
    hd : Pairwise (Function.onFun Disjoint s)
    n : Nat
    this : Eq ((Finset.range n).sum fun x => m (s x)) (m (Set.iUnion fun i => Set. …
    ⊢ LE.le (m (Set.iUnion fun i => Set.iUnion fun h => s i)) (m (Set.iUnion fun i …
  -/
  exact m.mono (iUnion₂_subset fun i _ => subset_iUnion _ i)
  /-
    🎉 no goals
  -/


/-- The Carathéodory-measurable sets for an outer measure `m` form a Dynkin system. -/
def caratheodoryDynkin : MeasurableSpace.DynkinSystem α where
  Has := IsCaratheodory m
  has_empty := isCaratheodory_empty m
  has_compl s := isCaratheodory_compl m s
                               /-
                                 α : Type u
                                 m : MeasureTheory.OuterMeasure α
                                 s s₁ s₂ : Set α
                                 f✝ : Nat → Set α
                                 x✝ : Pairwise (Function.onFun Disjoint f✝)
                                 hf : ∀ (i : Nat), m.IsCaratheodory (f✝ i)
                                 hn : Set α
                                 ⊢ Eq (m hn) (HAdd.hAdd (m (Inter.inter hn (Set.iUnion fun i => f✝ i))) (m (SDi …
                               -/
  has_iUnion_nat _ hf hn := by apply isCaratheodory_iUnion m hf
                               /-
                                 🎉 no goals
                               -/


/-- Given an outer measure `μ`, the Carathéodory-measurable space is
  defined such that `s` is measurable if `∀t, μ t = μ (t ∩ s) + μ (t \ s)`. -/
protected def caratheodory : MeasurableSpace α := by
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s s₁ s₂ : Set α
    ⊢ MeasurableSpace α
  -/
  apply MeasurableSpace.DynkinSystem.toMeasurableSpace (caratheodoryDynkin m)
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s s₁ s₂ : Set α
    ⊢ ∀ (s₁ s₂ : Set α), m.caratheodoryDynkin.Has s₁ → m.caratheodoryDynkin.Has s₂ …
  -/
  intro s₁ s₂
  /-
    α : Type u
    m : MeasureTheory.OuterMeasure α
    s s₁✝ s₂✝ s₁ s₂ : Set α
    ⊢ m.caratheodoryDynkin.Has s₁ → m.caratheodoryDynkin.Has s₂ → m.caratheodoryDy …
  -/
  apply isCaratheodory_inter
  /-
    🎉 no goals
  -/


theorem isCaratheodory_iff {s : Set α} :
    MeasurableSet[OuterMeasure.caratheodory m] s ↔ ∀ t, m t = m (t ∩ s) + m (t \ s) :=
  Iff.rfl


theorem isCaratheodory_iff_le {s : Set α} :
    MeasurableSet[OuterMeasure.caratheodory m] s ↔ ∀ t, m (t ∩ s) + m (t \ s) ≤ m t :=
  isCaratheodory_iff_le' m


protected theorem iUnion_eq_of_caratheodory {s : ℕ → Set α}
    (h : ∀ i, MeasurableSet[OuterMeasure.caratheodory m] (s i)) (hd : Pairwise (Disjoint on s)) :
    m (⋃ i, s i) = ∑' i, m (s i) :=
  f_iUnion m h hd


theorem ofFunction_caratheodory {m : Set α → ℝ≥0∞} {s : Set α} {h₀ : m ∅ = 0}
    (hs : ∀ t, m (t ∩ s) + m (t \ s) ≤ m t) :
    MeasurableSet[(OuterMeasure.ofFunction m h₀).caratheodory] s := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    s : Set α
    h₀ : Eq (m EmptyCollection.emptyCollection) 0
    hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
    ⊢ MeasurableSet s
  -/
  apply (isCaratheodory_iff_le _).mpr
  /-
    α : Type u_1
    m : Set α → ENNReal
    s : Set α
    h₀ : Eq (m EmptyCollection.emptyCollection) 0
    hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
    ⊢ ∀ (t : Set α), LE.le (HAdd.hAdd ((MeasureTheory.OuterMeasure.ofFunction m h₀ …
  -/
  refine fun t => le_iInf fun f => le_iInf fun hf => ?_
  refine
    le_trans
      (add_le_add ((iInf_le_of_le fun i => f i ∩ s) <| iInf_le _ ?_)
        ((iInf_le_of_le fun i => f i \ s) <| iInf_le _ ?_))
      ?_
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      h₀ : Eq (m EmptyCollection.emptyCollection) 0
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      f : Nat → Set α
      hf : HasSubset.Subset t (Set.iUnion fun i => f i)
      ⊢ HasSubset.Subset (Inter.inter t s) (Set.iUnion fun i => (fun i => Inter.inte …
    -/
  · rw [← iUnion_inter]
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      h₀ : Eq (m EmptyCollection.emptyCollection) 0
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      f : Nat → Set α
      hf : HasSubset.Subset t (Set.iUnion fun i => f i)
      ⊢ HasSubset.Subset (Inter.inter t s) (Inter.inter (Set.iUnion fun i => f i) s)
    -/
    exact inter_subset_inter_left _ hf
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      h₀ : Eq (m EmptyCollection.emptyCollection) 0
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      f : Nat → Set α
      hf : HasSubset.Subset t (Set.iUnion fun i => f i)
      ⊢ HasSubset.Subset (SDiff.sdiff t s) (Set.iUnion fun i => (fun i => SDiff.sdif …
    -/
  · rw [← iUnion_diff]
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      h₀ : Eq (m EmptyCollection.emptyCollection) 0
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      f : Nat → Set α
      hf : HasSubset.Subset t (Set.iUnion fun i => f i)
      ⊢ HasSubset.Subset (SDiff.sdiff t s) (SDiff.sdiff (Set.iUnion fun i => f i) s)
    -/
    exact diff_subset_diff_left hf
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      h₀ : Eq (m EmptyCollection.emptyCollection) 0
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      f : Nat → Set α
      hf : HasSubset.Subset t (Set.iUnion fun i => f i)
      ⊢ LE.le (HAdd.hAdd (tsum fun i => m ((fun i => Inter.inter (f i) s) i)) (tsum  …
    -/
  · rw [← ENNReal.tsum_add]
    /-
      case refine_3
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      h₀ : Eq (m EmptyCollection.emptyCollection) 0
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      f : Nat → Set α
      hf : HasSubset.Subset t (Set.iUnion fun i => f i)
      ⊢ LE.le (tsum fun a => HAdd.hAdd (m ((fun i => Inter.inter (f i) s) a)) (m ((f …
    -/
    exact ENNReal.tsum_le_tsum fun i => hs _
    /-
      🎉 no goals
    -/


theorem boundedBy_caratheodory {m : Set α → ℝ≥0∞} {s : Set α}
    (hs : ∀ t, m (t ∩ s) + m (t \ s) ≤ m t) : MeasurableSet[(boundedBy m).caratheodory] s := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    s : Set α
    hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
    ⊢ MeasurableSet s
  -/
  apply ofFunction_caratheodory; intro t
  /-
    case hs
    α : Type u_1
    m : Set α → ENNReal
    s : Set α
    hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
    t : Set α
    ⊢ LE.le (HAdd.hAdd (iSup fun x => m (Inter.inter t s)) (iSup fun x => m (SDiff …
  -/
  rcases t.eq_empty_or_nonempty with h | h
    /-
      case hs.inl
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      h : Eq t EmptyCollection.emptyCollection
      ⊢ LE.le (HAdd.hAdd (iSup fun x => m (Inter.inter t s)) (iSup fun x => m (SDiff …
    -/
  · simp [h, Set.not_nonempty_empty]
    /-
      🎉 no goals
    -/
    /-
      case hs.inr
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      h : t.Nonempty
      ⊢ LE.le (HAdd.hAdd (iSup fun x => m (Inter.inter t s)) (iSup fun x => m (SDiff …
    -/
  · convert le_trans _ (hs t)
      /-
        case h.e'_4
        α : Type u_1
        m : Set α → ENNReal
        s : Set α
        hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
        t : Set α
        h : t.Nonempty
        ⊢ Eq (iSup fun x => m t) (m t)
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
    /-
      case hs.inr.convert_2
      α : Type u_1
      m : Set α → ENNReal
      s : Set α
      hs : ∀ (t : Set α), LE.le (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s …
      t : Set α
      h : t.Nonempty
      ⊢ LE.le (HAdd.hAdd (iSup fun x => m (Inter.inter t s)) (iSup fun x => m (SDiff …
    -/
    exact add_le_add iSup_const_le iSup_const_le
    /-
      🎉 no goals
    -/


@[simp]
theorem zero_caratheodory : (0 : OuterMeasure α).caratheodory = ⊤ :=
  top_unique fun _ _ _ => (add_zero _).symm


theorem top_caratheodory : (⊤ : OuterMeasure α).caratheodory = ⊤ :=
  top_unique fun s _ =>
    (isCaratheodory_iff_le _).2 fun t =>
                                                /-
                                                  α : Type u_1
                                                  s : Set α
                                                  x✝ : MeasurableSet s
                                                  t : Set α
                                                  ht : Eq t EmptyCollection.emptyCollection
                                                  ⊢ LE.le (HAdd.hAdd (Top.top (Inter.inter t s)) (Top.top (SDiff.sdiff t s))) (T …
                                                -/
      t.eq_empty_or_nonempty.elim (fun ht => by simp [ht]) fun ht => by
                                                /-
                                                  🎉 no goals
                                                -/
        /-
          α : Type u_1
          s : Set α
          x✝ : MeasurableSet s
          t : Set α
          ht : t.Nonempty
          ⊢ LE.le (HAdd.hAdd (Top.top (Inter.inter t s)) (Top.top (SDiff.sdiff t s))) (T …
        -/
        simp only [ht, top_apply, le_top]
        /-
          🎉 no goals
        -/


theorem le_add_caratheodory (m₁ m₂ : OuterMeasure α) :
    m₁.caratheodory ⊓ m₂.caratheodory ≤ (m₁ + m₂ : OuterMeasure α).caratheodory :=
                           /-
                             α : Type u_1
                             m₁ m₂ : MeasureTheory.OuterMeasure α
                             s : Set α
                             x✝ : MeasurableSet s
                             t : Set α
                             hs₁ : Membership.mem ((fun m => setOf fun t => MeasurableSet t) m₁.caratheodor …
                             hs₂ : Membership.mem ((fun m => setOf fun t => MeasurableSet t) m₂.caratheodor …
                             ⊢ Eq ((HAdd.hAdd m₁ m₂) t) (HAdd.hAdd ((HAdd.hAdd m₁ m₂) (Inter.inter t s)) (( …
                           -/
  fun s ⟨hs₁, hs₂⟩ t => by simp [hs₁ t, hs₂ t, add_left_comm, add_assoc]
                           /-
                             🎉 no goals
                           -/


theorem le_sum_caratheodory {ι} (m : ι → OuterMeasure α) :
    ⨅ i, (m i).caratheodory ≤ (sum m).caratheodory := fun s h t => by
  /-
    α : Type u_1
    ι : Type u_2
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    h : MeasurableSet s
    t : Set α
    ⊢ Eq ((MeasureTheory.OuterMeasure.sum m) t) (HAdd.hAdd ((MeasureTheory.OuterMe …
  -/
  simp [fun i => MeasurableSpace.measurableSet_iInf.1 h i t, ENNReal.tsum_add]
  /-
    🎉 no goals
  -/


theorem le_smul_caratheodory (a : ℝ≥0∞) (m : OuterMeasure α) :
    m.caratheodory ≤ (a • m).caratheodory := fun s h t => by
      /-
        α : Type u_1
        a : ENNReal
        m : MeasureTheory.OuterMeasure α
        s : Set α
        h : MeasurableSet s
        t : Set α
        ⊢ Eq ((HSMul.hSMul a m) t) (HAdd.hAdd ((HSMul.hSMul a m) (Inter.inter t s)) (( …
      -/
      simp only [smul_apply, smul_eq_mul]
      /-
        α : Type u_1
        a : ENNReal
        m : MeasureTheory.OuterMeasure α
        s : Set α
        h : MeasurableSet s
        t : Set α
        ⊢ Eq (HMul.hMul a (m t)) (HAdd.hAdd (HMul.hMul a (m (Inter.inter t s))) (HMul. …
      -/
      rw [(isCaratheodory_iff m).mp h t]
      /-
        α : Type u_1
        a : ENNReal
        m : MeasureTheory.OuterMeasure α
        s : Set α
        h : MeasurableSet s
        t : Set α
        ⊢ Eq (HMul.hMul a (HAdd.hAdd (m (Inter.inter t s)) (m (SDiff.sdiff t s)))) (HA …
      -/
      simp [mul_add]
      /-
        🎉 no goals
      -/


@[simp]
theorem dirac_caratheodory (a : α) : (dirac a).caratheodory = ⊤ :=
  top_unique fun s _ t => by
    /-
      α : Type u_1
      a : α
      s : Set α
      x✝ : MeasurableSet s
      t : Set α
      ⊢ Eq ((MeasureTheory.OuterMeasure.dirac a) t) (HAdd.hAdd ((MeasureTheory.Outer …
    -/
    by_cases ht : a ∈ t; swap; · simp [ht]
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case pos
      α : Type u_1
      a : α
      s : Set α
      x✝ : MeasurableSet s
      t : Set α
      ht : Membership.mem t a
      ⊢ Eq ((MeasureTheory.OuterMeasure.dirac a) t) (HAdd.hAdd ((MeasureTheory.Outer …
    -/
                            /-
                              🎉 no goals
                            -/
    by_cases hs : a ∈ s <;> simp [*]
                            /-
                              🎉 no goals
                            -/


