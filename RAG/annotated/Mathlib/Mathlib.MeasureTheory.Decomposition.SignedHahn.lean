/-- Given the set `i` and the natural number `n`, `ExistsOneDivLT s i j` is the property that
there exists a measurable set `k ⊆ i` such that `1 / (n + 1) < s k`. -/
private def ExistsOneDivLT (s : SignedMeasure α) (i : Set α) (n : ℕ) : Prop :=
  ∃ k : Set α, k ⊆ i ∧ MeasurableSet k ∧ (1 / (n + 1) : ℝ) < s k


private theorem existsNatOneDivLTMeasure_of_not_negative (hi : ¬s ≤[i] 0) :
    ∃ n : ℕ, ExistsOneDivLT s i n :=
  let ⟨k, hj₁, hj₂, hj⟩ := exists_pos_measure_of_not_restrict_le_zero s hi
  let ⟨n, hn⟩ := exists_nat_one_div_lt hj
  ⟨n, k, hj₂, hj₁, hn⟩


/-- Given the set `i`, if `i` is not negative, `findExistsOneDivLT s i` is the
least natural number `n` such that `ExistsOneDivLT s i n`, otherwise, it returns 0. -/
private def findExistsOneDivLT (s : SignedMeasure α) (i : Set α) : ℕ :=
  if hi : ¬s ≤[i] 0 then Nat.find (existsNatOneDivLTMeasure_of_not_negative hi) else 0


private theorem findExistsOneDivLT_spec (hi : ¬s ≤[i] 0) :
    ExistsOneDivLT s i (findExistsOneDivLT s i) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
    ⊢ MeasureTheory.SignedMeasure.ExistsOneDivLT s i (MeasureTheory.SignedMeasure. …
  -/
  rw [findExistsOneDivLT, dif_pos hi]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
    ⊢ MeasureTheory.SignedMeasure.ExistsOneDivLT s i (Nat.find ⋯)
  -/
  convert Nat.find_spec (existsNatOneDivLTMeasure_of_not_negative hi)
  /-
    🎉 no goals
  -/


private theorem findExistsOneDivLT_min (hi : ¬s ≤[i] 0) {m : ℕ}
    (hm : m < findExistsOneDivLT s i) : ¬ExistsOneDivLT s i m := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
    m : Nat
    hm : LT.lt m (MeasureTheory.SignedMeasure.findExistsOneDivLT s i)
    ⊢ Not (MeasureTheory.SignedMeasure.ExistsOneDivLT s i m)
  -/
  rw [findExistsOneDivLT, dif_pos hi] at hm
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
    m : Nat
    hm : LT.lt m (Nat.find ⋯)
    ⊢ Not (MeasureTheory.SignedMeasure.ExistsOneDivLT s i m)
  -/
  exact Nat.find_min _ hm
  /-
    🎉 no goals
  -/


/-- Given the set `i`, if `i` is not negative, `someExistsOneDivLT` chooses the set
`k` from `ExistsOneDivLT s i (findExistsOneDivLT s i)`, otherwise, it returns the
empty set. -/
private def someExistsOneDivLT (s : SignedMeasure α) (i : Set α) : Set α :=
  if hi : ¬s ≤[i] 0 then Classical.choose (findExistsOneDivLT_spec hi) else ∅


private theorem someExistsOneDivLT_spec (hi : ¬s ≤[i] 0) :
    someExistsOneDivLT s i ⊆ i ∧
      MeasurableSet (someExistsOneDivLT s i) ∧
        (1 / (findExistsOneDivLT s i + 1) : ℝ) < s (someExistsOneDivLT s i) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
    ⊢ And (HasSubset.Subset (MeasureTheory.SignedMeasure.someExistsOneDivLT s i) i …
  -/
  rw [someExistsOneDivLT, dif_pos hi]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
    ⊢ And (HasSubset.Subset (Classical.choose ⋯) i) (And (MeasurableSet (Classical …
  -/
  exact Classical.choose_spec (findExistsOneDivLT_spec hi)
  /-
    🎉 no goals
  -/


private theorem someExistsOneDivLT_subset : someExistsOneDivLT s i ⊆ i := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    ⊢ HasSubset.Subset (MeasureTheory.SignedMeasure.someExistsOneDivLT s i) i
  -/
  by_cases hi : ¬s ≤[i] 0
  · exact
      let ⟨h, _⟩ := someExistsOneDivLT_spec hi
      h
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi : Not (Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory …
      ⊢ HasSubset.Subset (MeasureTheory.SignedMeasure.someExistsOneDivLT s i) i
    -/
  · rw [someExistsOneDivLT, dif_neg hi]
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi : Not (Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory …
      ⊢ HasSubset.Subset EmptyCollection.emptyCollection i
    -/
    exact Set.empty_subset _
    /-
      🎉 no goals
    -/


private theorem someExistsOneDivLT_subset' : someExistsOneDivLT s (i \ j) ⊆ i :=
  someExistsOneDivLT_subset.trans Set.diff_subset


private theorem someExistsOneDivLT_measurableSet : MeasurableSet (someExistsOneDivLT s i) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    ⊢ MeasurableSet (MeasureTheory.SignedMeasure.someExistsOneDivLT s i)
  -/
  by_cases hi : ¬s ≤[i] 0
  · exact
      let ⟨_, h, _⟩ := someExistsOneDivLT_spec hi
      h
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi : Not (Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory …
      ⊢ MeasurableSet (MeasureTheory.SignedMeasure.someExistsOneDivLT s i)
    -/
  · rw [someExistsOneDivLT, dif_neg hi]
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi : Not (Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory …
      ⊢ MeasurableSet EmptyCollection.emptyCollection
    -/
    exact MeasurableSet.empty
    /-
      🎉 no goals
    -/


private theorem someExistsOneDivLT_lt (hi : ¬s ≤[i] 0) :
    (1 / (findExistsOneDivLT s i + 1) : ℝ) < s (someExistsOneDivLT s i) :=
  let ⟨_, _, h⟩ := someExistsOneDivLT_spec hi
  h


/-- Given the set `i`, `restrictNonposSeq s i` is the sequence of sets defined inductively where
`restrictNonposSeq s i 0 = someExistsOneDivLT s (i \ ∅)` and
`restrictNonposSeq s i (n + 1) = someExistsOneDivLT s (i \ ⋃ k ≤ n, restrictNonposSeq k)`.

For each `n : ℕ`,`s (restrictNonposSeq s i n)` is close to maximal among all subsets of
`i \ ⋃ k ≤ n, restrictNonposSeq s i k`. -/
private def restrictNonposSeq (s : SignedMeasure α) (i : Set α) : ℕ → Set α
  | 0 => someExistsOneDivLT s (i \ ∅) -- I used `i \ ∅` instead of `i` to simplify some proofs
  | n + 1 =>
    someExistsOneDivLT s
      (i \
        ⋃ (k) (H : k ≤ n),
          have : k < n + 1 := Nat.lt_succ_iff.mpr H
          restrictNonposSeq s i k)


private theorem restrictNonposSeq_succ (n : ℕ) :
    restrictNonposSeq s i n.succ = someExistsOneDivLT s (i \ ⋃ k ≤ n, restrictNonposSeq s i k) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n : Nat
    ⊢ Eq (MeasureTheory.SignedMeasure.restrictNonposSeq s i n.succ) (MeasureTheory …
  -/
  rw [restrictNonposSeq]
  /-
    🎉 no goals
  -/


private theorem restrictNonposSeq_subset (n : ℕ) : restrictNonposSeq s i n ⊆ i := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n : Nat
    ⊢ HasSubset.Subset (MeasureTheory.SignedMeasure.restrictNonposSeq s i n) i
  -/
                                        /-
                                          🎉 no goals
                                        -/
  cases n <;> · rw [restrictNonposSeq]; exact someExistsOneDivLT_subset'
                                        /-
                                          🎉 no goals
                                        -/


private theorem restrictNonposSeq_lt (n : ℕ) (hn : ¬s ≤[i \ ⋃ k ≤ n, restrictNonposSeq s i k] 0) :
    (1 / (findExistsOneDivLT s (i \ ⋃ k ≤ n, restrictNonposSeq s i k) + 1) : ℝ) <
      s (restrictNonposSeq s i n.succ) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n : Nat
    hn : Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iU …
    ⊢ LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑(MeasureTheory.SignedMeasure.findExistsOneDi …
  -/
  rw [restrictNonposSeq_succ]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n : Nat
    hn : Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iU …
    ⊢ LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑(MeasureTheory.SignedMeasure.findExistsOneDi …
  -/
  apply someExistsOneDivLT_lt hn
  /-
    🎉 no goals
  -/


private theorem measure_of_restrictNonposSeq (hi₂ : ¬s ≤[i] 0) (n : ℕ)
    (hn : ¬s ≤[i \ ⋃ k < n, restrictNonposSeq s i k] 0) : 0 < s (restrictNonposSeq s i n) := by
  cases n with
  | zero =>
    rw [restrictNonposSeq]; rw [← @Set.diff_empty _ i] at hi₂
    rcases someExistsOneDivLT_spec hi₂ with ⟨_, _, h⟩
    exact lt_trans Nat.one_div_pos_of_nat h
  | succ n =>
    rw [restrictNonposSeq_succ]
    have h₁ : ¬s ≤[i \ ⋃ (k : ℕ) (_ : k ≤ n), restrictNonposSeq s i k] 0 := by
      refine mt (restrict_le_zero_subset _ ?_ (by simp [Nat.lt_succ_iff])) hn
      convert measurable_of_not_restrict_le_zero _ hn using 3
      exact funext fun x => by rw [Nat.lt_succ_iff]
    rcases someExistsOneDivLT_spec h₁ with ⟨_, _, h⟩
    exact lt_trans Nat.one_div_pos_of_nat h


private theorem restrictNonposSeq_measurableSet (n : ℕ) :
    MeasurableSet (restrictNonposSeq s i n) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n : Nat
    ⊢ MeasurableSet (MeasureTheory.SignedMeasure.restrictNonposSeq s i n)
  -/
  cases n <;>
      /-
        case zero
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        ⊢ MeasurableSet (MeasureTheory.SignedMeasure.restrictNonposSeq s i 0)
      -/
      /-
        case zero
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        ⊢ MeasurableSet (MeasureTheory.SignedMeasure.someExistsOneDivLT s (SDiff.sdiff …
      -/
      /-
        🎉 no goals
      -/
      /-
        case succ
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        n✝ : Nat
        ⊢ MeasurableSet (MeasureTheory.SignedMeasure.someExistsOneDivLT s (SDiff.sdiff …
      -/
      exact someExistsOneDivLT_measurableSet
      /-
        🎉 no goals
      -/


private theorem restrictNonposSeq_disjoint' {n m : ℕ} (h : n < m) :
    restrictNonposSeq s i n ∩ restrictNonposSeq s i m = ∅ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n m : Nat
    h : LT.lt n m
    ⊢ Eq (Inter.inter (MeasureTheory.SignedMeasure.restrictNonposSeq s i n) (Measu …
  -/
  rw [Set.eq_empty_iff_forall_not_mem]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n m : Nat
    h : LT.lt n m
    ⊢ ∀ (x : α), Not (Membership.mem (Inter.inter (MeasureTheory.SignedMeasure.res …
  -/
  rintro x ⟨hx₁, hx₂⟩
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n m : Nat
    h : LT.lt n m
    x : α
    hx₁ : Membership.mem (MeasureTheory.SignedMeasure.restrictNonposSeq s i n) x
    hx₂ : Membership.mem (MeasureTheory.SignedMeasure.restrictNonposSeq s i m) x
    ⊢ False
  -/
  cases m; · omega
             /-
               🎉 no goals
             -/
    /-
      case intro.succ
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      n : Nat
      x : α
      hx₁ : Membership.mem (MeasureTheory.SignedMeasure.restrictNonposSeq s i n) x
      n✝ : Nat
      h : LT.lt n (HAdd.hAdd n✝ 1)
      hx₂ : Membership.mem (MeasureTheory.SignedMeasure.restrictNonposSeq s i (HAdd. …
      ⊢ False
    -/
  · rw [restrictNonposSeq] at hx₂
    exact
      (someExistsOneDivLT_subset hx₂).2
        (Set.mem_iUnion.2 ⟨n, Set.mem_iUnion.2 ⟨Nat.lt_succ_iff.mp h, hx₁⟩⟩)


private theorem restrictNonposSeq_disjoint : Pairwise (Disjoint on restrictNonposSeq s i) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    ⊢ Pairwise (Function.onFun Disjoint (MeasureTheory.SignedMeasure.restrictNonpo …
  -/
  intro n m h
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n m : Nat
    h : Ne n m
    ⊢ Function.onFun Disjoint (MeasureTheory.SignedMeasure.restrictNonposSeq s i)  …
  -/
  rw [Function.onFun, Set.disjoint_iff_inter_eq_empty]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    n m : Nat
    h : Ne n m
    ⊢ Eq (Inter.inter (MeasureTheory.SignedMeasure.restrictNonposSeq s i n) (Measu …
  -/
  rcases lt_or_gt_of_ne h with (h | h)
    /-
      case inl
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      n m : Nat
      h✝ : Ne n m
      h : LT.lt n m
      ⊢ Eq (Inter.inter (MeasureTheory.SignedMeasure.restrictNonposSeq s i n) (Measu …
    -/
  · rw [restrictNonposSeq_disjoint' h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      n m : Nat
      h✝ : Ne n m
      h : GT.gt n m
      ⊢ Eq (Inter.inter (MeasureTheory.SignedMeasure.restrictNonposSeq s i n) (Measu …
    -/
  · rw [Set.inter_comm, restrictNonposSeq_disjoint' h]
    /-
      🎉 no goals
    -/


private theorem exists_subset_restrict_nonpos' (hi₁ : MeasurableSet i) (hi₂ : s i < 0)
    (hn : ¬∀ n : ℕ, ¬s ≤[i \ ⋃ l < n, restrictNonposSeq s i l] 0) :
    ∃ j : Set α, MeasurableSet j ∧ j ⊆ i ∧ s ≤[j] 0 ∧ s j < 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LT.lt (↑s i) 0
    hn : Not (∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDi …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  by_cases h : s ≤[i] 0
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      hn : Not (∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDi …
      h : LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.VectorMeas …
      ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
    -/
  · exact ⟨i, hi₁, Set.Subset.refl _, h, hi₂⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LT.lt (↑s i) 0
    hn : Not (∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDi …
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  push_neg at hn
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LT.lt (↑s i) 0
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  set k := Nat.find hn
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LT.lt (↑s i) 0
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
    k : Nat := Nat.find hn
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  have hk₂ : s ≤[i \ ⋃ l < k, restrictNonposSeq s i l] 0 := Nat.find_spec hn
  have hmeas : MeasurableSet (⋃ (l : ℕ) (_ : l < k), restrictNonposSeq s i l) :=
    MeasurableSet.iUnion fun _ => MeasurableSet.iUnion fun _ => restrictNonposSeq_measurableSet _
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LT.lt (↑s i) 0
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
    k : Nat := Nat.find hn
    hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
    hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  refine ⟨i \ ⋃ l < k, restrictNonposSeq s i l, hi₁.diff hmeas, Set.diff_subset, hk₂, ?_⟩
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LT.lt (↑s i) 0
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
    k : Nat := Nat.find hn
    hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
    hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
    ⊢ LT.lt (↑s (SDiff.sdiff i (Set.iUnion fun l => Set.iUnion fun h => MeasureThe …
  -/
  rw [of_diff hmeas hi₁, s.of_disjoint_iUnion]
  · have h₁ : ∀ l < k, 0 ≤ s (restrictNonposSeq s i l) := by
      intro l hl
      refine le_of_lt (measure_of_restrictNonposSeq h _ ?_)
      refine mt (restrict_le_zero_subset _ (hi₁.diff ?_) (Set.Subset.refl _)) (Nat.find_min hn hl)
      exact
        MeasurableSet.iUnion fun _ =>
          MeasurableSet.iUnion fun _ => restrictNonposSeq_measurableSet _
    suffices 0 ≤ ∑' l : ℕ, s (⋃ _ : l < k, restrictNonposSeq s i l) by
      rw [sub_neg]
      exact lt_of_lt_of_le hi₂ this
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
      ⊢ LE.le 0 (tsum fun l => ↑s (Set.iUnion fun x => MeasureTheory.SignedMeasure.r …
    -/
    refine tsum_nonneg ?_
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
      ⊢ ∀ (i_1 : Nat), LE.le 0 (↑s (Set.iUnion fun x => MeasureTheory.SignedMeasure. …
    -/
    intro l; by_cases h : l < k
      /-
        case pos
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LT.lt (↑s i) 0
        h✝ : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
        hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
        k : Nat := Nat.find hn
        hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
        hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
        h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
        l : Nat
        h : LT.lt l k
        ⊢ LE.le 0 (↑s (Set.iUnion fun x => MeasureTheory.SignedMeasure.restrictNonposS …
      -/
    · convert h₁ _ h
      /-
        case h.e'_4.h.e'_7
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LT.lt (↑s i) 0
        h✝ : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
        hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
        k : Nat := Nat.find hn
        hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
        hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
        h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
        l : Nat
        h : LT.lt l k
        ⊢ Eq (Set.iUnion fun x => MeasureTheory.SignedMeasure.restrictNonposSeq s i l) …
      -/
      ext x
      /-
        case h.e'_4.h.e'_7.h
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LT.lt (↑s i) 0
        h✝ : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
        hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
        k : Nat := Nat.find hn
        hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
        hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
        h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
        l : Nat
        h : LT.lt l k
        x : α
        ⊢ Iff (Membership.mem (Set.iUnion fun x => MeasureTheory.SignedMeasure.restric …
      -/
      rw [Set.mem_iUnion, exists_prop, and_iff_right_iff_imp]
      /-
        case h.e'_4.h.e'_7.h
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LT.lt (↑s i) 0
        h✝ : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
        hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
        k : Nat := Nat.find hn
        hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
        hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
        h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
        l : Nat
        h : LT.lt l k
        x : α
        ⊢ Membership.mem (MeasureTheory.SignedMeasure.restrictNonposSeq s i l) x → LT. …
      -/
      exact fun _ => h
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LT.lt (↑s i) 0
        h✝ : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
        hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
        k : Nat := Nat.find hn
        hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
        hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
        h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
        l : Nat
        h : Not (LT.lt l k)
        ⊢ LE.le 0 (↑s (Set.iUnion fun x => MeasureTheory.SignedMeasure.restrictNonposS …
      -/
    · convert le_of_eq s.empty.symm
      /-
        case h.e'_4.h.e'_7
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LT.lt (↑s i) 0
        h✝ : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
        hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
        k : Nat := Nat.find hn
        hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
        hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
        h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
        l : Nat
        h : Not (LT.lt l k)
        ⊢ Eq (Set.iUnion fun x => MeasureTheory.SignedMeasure.restrictNonposSeq s i l) …
      -/
      ext; simp only [exists_prop, Set.mem_empty_iff_false, Set.mem_iUnion, not_and, iff_false]
      /-
        case h.e'_4.h.e'_7.h
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LT.lt (↑s i) 0
        h✝ : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vect …
        hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
        k : Nat := Nat.find hn
        hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
        hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
        h₁ : ∀ (l : Nat), LT.lt l k → LE.le 0 (↑s (MeasureTheory.SignedMeasure.restric …
        l : Nat
        h : Not (LT.lt l k)
        x✝ : α
        ⊢ LT.lt l k → Not (Membership.mem (MeasureTheory.SignedMeasure.restrictNonposS …
      -/
      exact fun h' => False.elim (h h')
      /-
        🎉 no goals
      -/
    /-
      case neg.hm
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      ⊢ ∀ (i_1 : Nat), MeasurableSet (Set.iUnion fun x => MeasureTheory.SignedMeasur …
    -/
  · intro; exact MeasurableSet.iUnion fun _ => restrictNonposSeq_measurableSet _
           /-
             🎉 no goals
           -/
    /-
      case neg.hd
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      ⊢ Pairwise (Function.onFun Disjoint fun l => Set.iUnion fun x => MeasureTheory …
    -/
  · intro a b hab
    /-
      case neg.hd
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      a b : Nat
      hab : Ne a b
      ⊢ Function.onFun Disjoint (fun l => Set.iUnion fun x => MeasureTheory.SignedMe …
    -/
    refine Set.disjoint_iUnion_left.mpr fun _ => ?_
    /-
      case neg.hd
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      a b : Nat
      hab : Ne a b
      x✝ : LT.lt a k
      ⊢ Disjoint (MeasureTheory.SignedMeasure.restrictNonposSeq s i a) ((fun l => Se …
    -/
    refine Set.disjoint_iUnion_right.mpr fun _ => ?_
    /-
      case neg.hd
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      a b : Nat
      hab : Ne a b
      x✝¹ : LT.lt a k
      x✝ : LT.lt b k
      ⊢ Disjoint (MeasureTheory.SignedMeasure.restrictNonposSeq s i a) (MeasureTheor …
    -/
    exact restrictNonposSeq_disjoint hab
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      ⊢ HasSubset.Subset (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.Sign …
    -/
  · apply Set.iUnion_subset
    /-
      case neg.h
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      ⊢ ∀ (i_1 : Nat), HasSubset.Subset (Set.iUnion fun x => MeasureTheory.SignedMea …
    -/
    intro a x
    /-
      case neg.h
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      a : Nat
      x : α
      ⊢ Membership.mem (Set.iUnion fun x => MeasureTheory.SignedMeasure.restrictNonp …
    -/
    simp only [and_imp, exists_prop, Set.mem_iUnion]
    /-
      case neg.h
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      a : Nat
      x : α
      ⊢ LT.lt a k → Membership.mem (MeasureTheory.SignedMeasure.restrictNonposSeq s  …
    -/
    intro _ hx
    /-
      case neg.h
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LT.lt (↑s i) 0
      h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
      hn : Exists fun n => LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdif …
      k : Nat := Nat.find hn
      hk₂ : LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sdiff i (Set.iUnion …
      hmeas : MeasurableSet (Set.iUnion fun l => Set.iUnion fun x => MeasureTheory.S …
      a : Nat
      x : α
      a✝ : LT.lt a k
      hx : Membership.mem (MeasureTheory.SignedMeasure.restrictNonposSeq s i a) x
      ⊢ Membership.mem i x
    -/
    exact restrictNonposSeq_subset _ hx
    /-
      🎉 no goals
    -/


/-- A measurable set of negative measure has a negative subset of negative measure. -/
theorem exists_subset_restrict_nonpos (hi : s i < 0) :
    ∃ j : Set α, MeasurableSet j ∧ j ⊆ i ∧ s ≤[j] 0 ∧ s j < 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  have hi₁ : MeasurableSet i := by_contradiction fun h => ne_of_lt hi <| s.not_measurable h
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  by_cases h : s ≤[i] 0; · exact ⟨i, hi₁, Set.Subset.refl _, h, hi⟩
                           /-
                             🎉 no goals
                           -/
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  by_cases hn : ∀ n : ℕ, ¬s ≤[i \ ⋃ l < n, restrictNonposSeq s i l] 0
  /-
    case pos
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  swap; · exact exists_subset_restrict_nonpos' hi₁ hi hn
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  set A := i \ ⋃ l, restrictNonposSeq s i l with hA
  /-
    case pos
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    A : Set α := SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.re …
    hA : Eq A (SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.rest …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  set bdd : ℕ → ℕ := fun n => findExistsOneDivLT s (i \ ⋃ k ≤ n, restrictNonposSeq s i k)
  have hn' : ∀ n : ℕ, ¬s ≤[i \ ⋃ l ≤ n, restrictNonposSeq s i l] 0 := by
    intro n
    convert hn (n + 1) using 5 <;>
      · ext l
        simp only [exists_prop, Set.mem_iUnion, and_congr_left_iff]
        exact fun _ => Nat.lt_succ_iff.symm
  have h₁ : s i = s A + ∑' l, s (restrictNonposSeq s i l) := by
    rw [hA, ← s.of_disjoint_iUnion, add_comm, of_add_of_diff]
    · exact MeasurableSet.iUnion fun _ => restrictNonposSeq_measurableSet _
    exacts [hi₁, Set.iUnion_subset fun _ => restrictNonposSeq_subset _, fun _ =>
      restrictNonposSeq_measurableSet _, restrictNonposSeq_disjoint]
  have h₂ : s A ≤ s i := by
    rw [h₁]
    apply le_add_of_nonneg_right
    exact tsum_nonneg fun n => le_of_lt (measure_of_restrictNonposSeq h _ (hn n))
  have h₃' : Summable fun n => (1 / (bdd n + 1) : ℝ) := by
    have : Summable fun l => s (restrictNonposSeq s i l) :=
      HasSum.summable
        (s.m_iUnion (fun _ => restrictNonposSeq_measurableSet _) restrictNonposSeq_disjoint)
    refine .of_nonneg_of_le (fun n => ?_) (fun n => ?_)
        (this.comp_injective Nat.succ_injective)
    · exact le_of_lt Nat.one_div_pos_of_nat
    · exact le_of_lt (restrictNonposSeq_lt n (hn' n))
  have h₃ : Tendsto (fun n => (bdd n : ℝ) + 1) atTop atTop := by
    simp only [one_div] at h₃'
    exact Summable.tendsto_atTop_of_pos h₃' fun n => Nat.cast_add_one_pos (bdd n)
  have h₄ : Tendsto (fun n => (bdd n : ℝ)) atTop atTop := by
    convert atTop.tendsto_atTop_add_const_right (-1) h₃; simp
  have A_meas : MeasurableSet A :=
    hi₁.diff (MeasurableSet.iUnion fun _ => restrictNonposSeq_measurableSet _)
  /-
    case pos
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    A : Set α := SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.re …
    hA : Eq A (SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.rest …
    bdd : Nat → Nat := fun n => MeasureTheory.SignedMeasure.findExistsOneDivLT s ( …
    hn' : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.s …
    h₁ : Eq (↑s i) (HAdd.hAdd (↑s A) (tsum fun l => ↑s (MeasureTheory.SignedMeasur …
    h₂ : LE.le (↑s A) (↑s i)
    h₃' : Summable fun n => HDiv.hDiv 1 (HAdd.hAdd (↑(bdd n)) 1)
    h₃ : Filter.Tendsto (fun n => HAdd.hAdd (↑(bdd n)) 1) Filter.atTop Filter.atTop
    h₄ : Filter.Tendsto (fun n => ↑(bdd n)) Filter.atTop Filter.atTop
    A_meas : MeasurableSet A
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (And (LE.l …
  -/
  refine ⟨A, A_meas, Set.diff_subset, ?_, h₂.trans_lt hi⟩
  /-
    case pos
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    A : Set α := SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.re …
    hA : Eq A (SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.rest …
    bdd : Nat → Nat := fun n => MeasureTheory.SignedMeasure.findExistsOneDivLT s ( …
    hn' : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.s …
    h₁ : Eq (↑s i) (HAdd.hAdd (↑s A) (tsum fun l => ↑s (MeasureTheory.SignedMeasur …
    h₂ : LE.le (↑s A) (↑s i)
    h₃' : Summable fun n => HDiv.hDiv 1 (HAdd.hAdd (↑(bdd n)) 1)
    h₃ : Filter.Tendsto (fun n => HAdd.hAdd (↑(bdd n)) 1) Filter.atTop Filter.atTop
    h₄ : Filter.Tendsto (fun n => ↑(bdd n)) Filter.atTop Filter.atTop
    A_meas : MeasurableSet A
    ⊢ LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMeasur …
  -/
  by_contra hnn
  /-
    case pos
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    A : Set α := SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.re …
    hA : Eq A (SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.rest …
    bdd : Nat → Nat := fun n => MeasureTheory.SignedMeasure.findExistsOneDivLT s ( …
    hn' : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.s …
    h₁ : Eq (↑s i) (HAdd.hAdd (↑s A) (tsum fun l => ↑s (MeasureTheory.SignedMeasur …
    h₂ : LE.le (↑s A) (↑s i)
    h₃' : Summable fun n => HDiv.hDiv 1 (HAdd.hAdd (↑(bdd n)) 1)
    h₃ : Filter.Tendsto (fun n => HAdd.hAdd (↑(bdd n)) 1) Filter.atTop Filter.atTop
    h₄ : Filter.Tendsto (fun n => ↑(bdd n)) Filter.atTop Filter.atTop
    A_meas : MeasurableSet A
    hnn : Not (LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.Vec …
    ⊢ False
  -/
  rw [restrict_le_restrict_iff _ _ A_meas] at hnn; push_neg at hnn
  /-
    case pos
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    A : Set α := SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.re …
    hA : Eq A (SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.rest …
    bdd : Nat → Nat := fun n => MeasureTheory.SignedMeasure.findExistsOneDivLT s ( …
    hn' : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.s …
    h₁ : Eq (↑s i) (HAdd.hAdd (↑s A) (tsum fun l => ↑s (MeasureTheory.SignedMeasur …
    h₂ : LE.le (↑s A) (↑s i)
    h₃' : Summable fun n => HDiv.hDiv 1 (HAdd.hAdd (↑(bdd n)) 1)
    h₃ : Filter.Tendsto (fun n => HAdd.hAdd (↑(bdd n)) 1) Filter.atTop Filter.atTop
    h₄ : Filter.Tendsto (fun n => ↑(bdd n)) Filter.atTop Filter.atTop
    A_meas : MeasurableSet A
    hnn : Exists fun ⦃j⦄ => And (MeasurableSet j) (And (HasSubset.Subset j A) (LT. …
    ⊢ False
  -/
  obtain ⟨E, hE₁, hE₂, hE₃⟩ := hnn
  have : ∃ k, 1 ≤ bdd k ∧ 1 / (bdd k : ℝ) < s E := by
    rw [tendsto_atTop_atTop] at h₄
    obtain ⟨k, hk⟩ := h₄ (max (1 / s E + 1) 1)
    refine ⟨k, ?_, ?_⟩
    · have hle := le_of_max_le_right (hk k le_rfl)
      norm_cast at hle
    · have : 1 / s E < bdd k := by
        linarith only [le_of_max_le_left (hk k le_rfl)]
      rw [one_div] at this ⊢
      exact inv_lt_of_inv_lt₀ hE₃ this
  /-
    case pos.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    A : Set α := SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.re …
    hA : Eq A (SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.rest …
    bdd : Nat → Nat := fun n => MeasureTheory.SignedMeasure.findExistsOneDivLT s ( …
    hn' : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.s …
    h₁ : Eq (↑s i) (HAdd.hAdd (↑s A) (tsum fun l => ↑s (MeasureTheory.SignedMeasur …
    h₂ : LE.le (↑s A) (↑s i)
    h₃' : Summable fun n => HDiv.hDiv 1 (HAdd.hAdd (↑(bdd n)) 1)
    h₃ : Filter.Tendsto (fun n => HAdd.hAdd (↑(bdd n)) 1) Filter.atTop Filter.atTop
    h₄ : Filter.Tendsto (fun n => ↑(bdd n)) Filter.atTop Filter.atTop
    A_meas : MeasurableSet A
    E : Set α
    hE₁ : MeasurableSet E
    hE₂ : HasSubset.Subset E A
    hE₃ : LT.lt (↑0 E) (↑s E)
    this : Exists fun k => And (LE.le 1 (bdd k)) (LT.lt (HDiv.hDiv 1 ↑(bdd k)) (↑s …
    ⊢ False
  -/
  obtain ⟨k, hk₁, hk₂⟩ := this
  have hA' : A ⊆ i \ ⋃ l ≤ k, restrictNonposSeq s i l := by
    apply Set.diff_subset_diff_right
    intro x; simp only [Set.mem_iUnion]
    rintro ⟨n, _, hn₂⟩
    exact ⟨n, hn₂⟩
  refine
    findExistsOneDivLT_min (hn' k) (Nat.sub_lt hk₁ Nat.zero_lt_one)
      ⟨E, Set.Subset.trans hE₂ hA', hE₁, ?_⟩
  /-
    case pos.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    A : Set α := SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.re …
    hA : Eq A (SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.rest …
    bdd : Nat → Nat := fun n => MeasureTheory.SignedMeasure.findExistsOneDivLT s ( …
    hn' : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.s …
    h₁ : Eq (↑s i) (HAdd.hAdd (↑s A) (tsum fun l => ↑s (MeasureTheory.SignedMeasur …
    h₂ : LE.le (↑s A) (↑s i)
    h₃' : Summable fun n => HDiv.hDiv 1 (HAdd.hAdd (↑(bdd n)) 1)
    h₃ : Filter.Tendsto (fun n => HAdd.hAdd (↑(bdd n)) 1) Filter.atTop Filter.atTop
    h₄ : Filter.Tendsto (fun n => ↑(bdd n)) Filter.atTop Filter.atTop
    A_meas : MeasurableSet A
    E : Set α
    hE₁ : MeasurableSet E
    hE₂ : HasSubset.Subset E A
    hE₃ : LT.lt (↑0 E) (↑s E)
    k : Nat
    hk₁ : LE.le 1 (bdd k)
    hk₂ : LT.lt (HDiv.hDiv 1 ↑(bdd k)) (↑s E)
    hA' : HasSubset.Subset A (SDiff.sdiff i (Set.iUnion fun l => Set.iUnion fun h  …
    ⊢ LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑(HSub.hSub (MeasureTheory.SignedMeasure.find …
  -/
  convert hk₂; norm_cast
  /-
    case h.e'_3.h.e'_6
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi : LT.lt (↑s i) 0
    hi₁ : MeasurableSet i
    h : Not (LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.Vecto …
    hn : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.sd …
    A : Set α := SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.re …
    hA : Eq A (SDiff.sdiff i (Set.iUnion fun l => MeasureTheory.SignedMeasure.rest …
    bdd : Nat → Nat := fun n => MeasureTheory.SignedMeasure.findExistsOneDivLT s ( …
    hn' : ∀ (n : Nat), Not (LE.le (MeasureTheory.VectorMeasure.restrict s (SDiff.s …
    h₁ : Eq (↑s i) (HAdd.hAdd (↑s A) (tsum fun l => ↑s (MeasureTheory.SignedMeasur …
    h₂ : LE.le (↑s A) (↑s i)
    h₃' : Summable fun n => HDiv.hDiv 1 (HAdd.hAdd (↑(bdd n)) 1)
    h₃ : Filter.Tendsto (fun n => HAdd.hAdd (↑(bdd n)) 1) Filter.atTop Filter.atTop
    h₄ : Filter.Tendsto (fun n => ↑(bdd n)) Filter.atTop Filter.atTop
    A_meas : MeasurableSet A
    E : Set α
    hE₁ : MeasurableSet E
    hE₂ : HasSubset.Subset E A
    hE₃ : LT.lt (↑0 E) (↑s E)
    k : Nat
    hk₁ : LE.le 1 (bdd k)
    hk₂ : LT.lt (HDiv.hDiv 1 ↑(bdd k)) (↑s E)
    hA' : HasSubset.Subset A (SDiff.sdiff i (Set.iUnion fun l => Set.iUnion fun h  …
    ⊢ Eq (HAdd.hAdd (HSub.hSub (MeasureTheory.SignedMeasure.findExistsOneDivLT s ( …
  -/
  exact tsub_add_cancel_of_le hk₁
  /-
    🎉 no goals
  -/


/-- The set of measures of the set of measurable negative sets. -/
def measureOfNegatives (s : SignedMeasure α) : Set ℝ :=
  s '' { B | MeasurableSet B ∧ s ≤[B] 0 }


theorem zero_mem_measureOfNegatives : (0 : ℝ) ∈ s.measureOfNegatives :=
  ⟨∅, ⟨MeasurableSet.empty, le_restrict_empty _ _⟩, s.empty⟩


theorem bddBelow_measureOfNegatives : BddBelow s.measureOfNegatives := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    ⊢ BddBelow s.measureOfNegatives
  -/
  simp_rw [BddBelow, Set.Nonempty, mem_lowerBounds]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    ⊢ Exists fun x => ∀ (x_1 : Real), Membership.mem s.measureOfNegatives x_1 → LE …
  -/
  by_contra! h
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    h : ∀ (x : Real), Exists fun x_1 => And (Membership.mem s.measureOfNegatives x …
    ⊢ False
  -/
  have h' : ∀ n : ℕ, ∃ y : ℝ, y ∈ s.measureOfNegatives ∧ y < -n := fun n => h (-n)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    h : ∀ (x : Real), Exists fun x_1 => And (Membership.mem s.measureOfNegatives x …
    h' : ∀ (n : Nat), Exists fun y => And (Membership.mem s.measureOfNegatives y)  …
    ⊢ False
  -/
  choose f hf using h'
  have hf' : ∀ n : ℕ, ∃ B, MeasurableSet B ∧ s ≤[B] 0 ∧ s B < -n := by
    intro n
    rcases hf n with ⟨⟨B, ⟨hB₁, hBr⟩, hB₂⟩, hlt⟩
    exact ⟨B, hB₁, hBr, hB₂.symm ▸ hlt⟩
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    h : ∀ (x : Real), Exists fun x_1 => And (Membership.mem s.measureOfNegatives x …
    f : Nat → Real
    hf : ∀ (n : Nat), And (Membership.mem s.measureOfNegatives (f n)) (LT.lt (f n) …
    hf' : ∀ (n : Nat), Exists fun B => And (MeasurableSet B) (And (LE.le (MeasureT …
    ⊢ False
  -/
  choose B hmeas hr h_lt using hf'
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    h : ∀ (x : Real), Exists fun x_1 => And (Membership.mem s.measureOfNegatives x …
    f : Nat → Real
    hf : ∀ (n : Nat), And (Membership.mem s.measureOfNegatives (f n)) (LT.lt (f n) …
    B : Nat → Set α
    hmeas : ∀ (n : Nat), MeasurableSet (B n)
    hr : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measur …
    h_lt : ∀ (n : Nat), LT.lt (↑s (B n)) (Neg.neg ↑n)
    ⊢ False
  -/
  set A := ⋃ n, B n with hA
  have hfalse : ∀ n : ℕ, s A ≤ -n := by
    intro n
    refine le_trans ?_ (le_of_lt (h_lt _))
    rw [hA, ← Set.diff_union_of_subset (Set.subset_iUnion _ n),
      of_union Set.disjoint_sdiff_left _ (hmeas n)]
    · refine add_le_of_nonpos_left ?_
      have : s ≤[A] 0 := restrict_le_restrict_iUnion _ _ hmeas hr
      refine nonpos_of_restrict_le_zero _ (restrict_le_zero_subset _ ?_ Set.diff_subset this)
      exact MeasurableSet.iUnion hmeas
    · exact (MeasurableSet.iUnion hmeas).diff (hmeas n)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    h : ∀ (x : Real), Exists fun x_1 => And (Membership.mem s.measureOfNegatives x …
    f : Nat → Real
    hf : ∀ (n : Nat), And (Membership.mem s.measureOfNegatives (f n)) (LT.lt (f n) …
    B : Nat → Set α
    hmeas : ∀ (n : Nat), MeasurableSet (B n)
    hr : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measur …
    h_lt : ∀ (n : Nat), LT.lt (↑s (B n)) (Neg.neg ↑n)
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hfalse : ∀ (n : Nat), LE.le (↑s A) (Neg.neg ↑n)
    ⊢ False
  -/
  rcases exists_nat_gt (-s A) with ⟨n, hn⟩
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    h : ∀ (x : Real), Exists fun x_1 => And (Membership.mem s.measureOfNegatives x …
    f : Nat → Real
    hf : ∀ (n : Nat), And (Membership.mem s.measureOfNegatives (f n)) (LT.lt (f n) …
    B : Nat → Set α
    hmeas : ∀ (n : Nat), MeasurableSet (B n)
    hr : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measur …
    h_lt : ∀ (n : Nat), LT.lt (↑s (B n)) (Neg.neg ↑n)
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hfalse : ∀ (n : Nat), LE.le (↑s A) (Neg.neg ↑n)
    n : Nat
    hn : LT.lt (Neg.neg (↑s A)) ↑n
    ⊢ False
  -/
  exact lt_irrefl _ ((neg_lt.1 hn).trans_le (hfalse n))
  /-
    🎉 no goals
  -/


/-- Alternative formulation of `MeasureTheory.SignedMeasure.exists_isCompl_positive_negative`
(the Hahn decomposition theorem) using set complements. -/
theorem exists_compl_positive_negative (s : SignedMeasure α) :
    ∃ i : Set α, MeasurableSet i ∧ 0 ≤[i] s ∧ s ≤[iᶜ] 0 := by
  obtain ⟨f, _, hf₂, hf₁⟩ :=
    exists_seq_tendsto_sInf ⟨0, @zero_mem_measureOfNegatives _ _ s⟩ bddBelow_measureOfNegatives
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    hf₁ : ∀ (n : Nat), Membership.mem s.measureOfNegatives (f n)
    ⊢ Exists fun i => And (MeasurableSet i) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  choose B hB using hf₁
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    ⊢ Exists fun i => And (MeasurableSet i) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  have hB₁ : ∀ n, MeasurableSet (B n) := fun n => (hB n).1.1
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    ⊢ Exists fun i => And (MeasurableSet i) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  have hB₂ : ∀ n, s ≤[B n] 0 := fun n => (hB n).1.2
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    ⊢ Exists fun i => And (MeasurableSet i) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  set A := ⋃ n, B n with hA
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    ⊢ Exists fun i => And (MeasurableSet i) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  have hA₁ : MeasurableSet A := MeasurableSet.iUnion hB₁
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hA₁ : MeasurableSet A
    ⊢ Exists fun i => And (MeasurableSet i) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  have hA₂ : s ≤[A] 0 := restrict_le_restrict_iUnion _ _ hB₁ hB₂
  have hA₃ : s A = sInf s.measureOfNegatives := by
    apply le_antisymm
    · refine le_of_tendsto_of_tendsto tendsto_const_nhds hf₂ (Eventually.of_forall fun n => ?_)
      rw [← (hB n).2, hA, ← Set.diff_union_of_subset (Set.subset_iUnion _ n),
        of_union Set.disjoint_sdiff_left _ (hB₁ n)]
      · refine add_le_of_nonpos_left ?_
        have : s ≤[A] 0 :=
          restrict_le_restrict_iUnion _ _ hB₁ fun m =>
            let ⟨_, h⟩ := (hB m).1
            h
        refine
          nonpos_of_restrict_le_zero _ (restrict_le_zero_subset _ ?_ Set.diff_subset this)
        exact MeasurableSet.iUnion hB₁
      · exact (MeasurableSet.iUnion hB₁).diff (hB₁ n)
    · exact csInf_le bddBelow_measureOfNegatives ⟨A, ⟨hA₁, hA₂⟩, rfl⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hA₁ : MeasurableSet A
    hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
    hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
    ⊢ Exists fun i => And (MeasurableSet i) (And (LE.le (MeasureTheory.VectorMeasu …
  -/
  refine ⟨Aᶜ, hA₁.compl, ?_, (compl_compl A).symm ▸ hA₂⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hA₁ : MeasurableSet A
    hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
    hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
    ⊢ LE.le (MeasureTheory.VectorMeasure.restrict 0 (HasCompl.compl A)) (MeasureTh …
  -/
  rw [restrict_le_restrict_iff _ _ hA₁.compl]
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hA₁ : MeasurableSet A
    hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
    hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
    ⊢ ∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j (HasCompl.compl A) → LE. …
  -/
  intro C _ hC₁
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hA₁ : MeasurableSet A
    hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
    hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
    C : Set α
    a✝ : MeasurableSet C
    hC₁ : HasSubset.Subset C (HasCompl.compl A)
    ⊢ LE.le (↑0 C) (↑s C)
  -/
  by_contra! hC₂
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hA₁ : MeasurableSet A
    hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
    hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
    C : Set α
    a✝ : MeasurableSet C
    hC₁ : HasSubset.Subset C (HasCompl.compl A)
    hC₂ : LT.lt (↑s C) (↑0 C)
    ⊢ False
  -/
  rcases exists_subset_restrict_nonpos hC₂ with ⟨D, hD₁, hD, hD₂, hD₃⟩
  have : s (A ∪ D) < sInf s.measureOfNegatives := by
    rw [← hA₃,
      of_union (Set.disjoint_of_subset_right (Set.Subset.trans hD hC₁) disjoint_compl_right) hA₁
        hD₁]
    linarith
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hA₁ : MeasurableSet A
    hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
    hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
    C : Set α
    a✝ : MeasurableSet C
    hC₁ : HasSubset.Subset C (HasCompl.compl A)
    hC₂ : LT.lt (↑s C) (↑0 C)
    D : Set α
    hD₁ : MeasurableSet D
    hD : HasSubset.Subset D C
    hD₂ : LE.le (MeasureTheory.VectorMeasure.restrict s D) (MeasureTheory.VectorMe …
    hD₃ : LT.lt (↑s D) 0
    this : LT.lt (↑s (Union.union A D)) (InfSet.sInf s.measureOfNegatives)
    ⊢ False
  -/
  refine not_le.2 this ?_
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    f : Nat → Real
    left✝ : Antitone f
    hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
    B : Nat → Set α
    hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
    hB₁ : ∀ (n : Nat), MeasurableSet (B n)
    hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
    A : Set α := Set.iUnion fun n => B n
    hA : Eq A (Set.iUnion fun n => B n)
    hA₁ : MeasurableSet A
    hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
    hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
    C : Set α
    a✝ : MeasurableSet C
    hC₁ : HasSubset.Subset C (HasCompl.compl A)
    hC₂ : LT.lt (↑s C) (↑0 C)
    D : Set α
    hD₁ : MeasurableSet D
    hD : HasSubset.Subset D C
    hD₂ : LE.le (MeasureTheory.VectorMeasure.restrict s D) (MeasureTheory.VectorMe …
    hD₃ : LT.lt (↑s D) 0
    this : LT.lt (↑s (Union.union A D)) (InfSet.sInf s.measureOfNegatives)
    ⊢ LE.le (InfSet.sInf s.measureOfNegatives) (↑s (Union.union A D))
  -/
  refine csInf_le bddBelow_measureOfNegatives ⟨A ∪ D, ⟨?_, ?_⟩, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      f : Nat → Real
      left✝ : Antitone f
      hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
      B : Nat → Set α
      hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
      hB₁ : ∀ (n : Nat), MeasurableSet (B n)
      hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
      A : Set α := Set.iUnion fun n => B n
      hA : Eq A (Set.iUnion fun n => B n)
      hA₁ : MeasurableSet A
      hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
      hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
      C : Set α
      a✝ : MeasurableSet C
      hC₁ : HasSubset.Subset C (HasCompl.compl A)
      hC₂ : LT.lt (↑s C) (↑0 C)
      D : Set α
      hD₁ : MeasurableSet D
      hD : HasSubset.Subset D C
      hD₂ : LE.le (MeasureTheory.VectorMeasure.restrict s D) (MeasureTheory.VectorMe …
      hD₃ : LT.lt (↑s D) 0
      this : LT.lt (↑s (Union.union A D)) (InfSet.sInf s.measureOfNegatives)
      ⊢ MeasurableSet (Union.union A D)
    -/
  · exact hA₁.union hD₁
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      f : Nat → Real
      left✝ : Antitone f
      hf₂ : Filter.Tendsto f Filter.atTop (nhds (InfSet.sInf s.measureOfNegatives))
      B : Nat → Set α
      hB : ∀ (n : Nat), And (Membership.mem (setOf fun B => And (MeasurableSet B) (L …
      hB₁ : ∀ (n : Nat), MeasurableSet (B n)
      hB₂ : ∀ (n : Nat), LE.le (MeasureTheory.VectorMeasure.restrict s (B n)) (Measu …
      A : Set α := Set.iUnion fun n => B n
      hA : Eq A (Set.iUnion fun n => B n)
      hA₁ : MeasurableSet A
      hA₂ : LE.le (MeasureTheory.VectorMeasure.restrict s A) (MeasureTheory.VectorMe …
      hA₃ : Eq (↑s A) (InfSet.sInf s.measureOfNegatives)
      C : Set α
      a✝ : MeasurableSet C
      hC₁ : HasSubset.Subset C (HasCompl.compl A)
      hC₂ : LT.lt (↑s C) (↑0 C)
      D : Set α
      hD₁ : MeasurableSet D
      hD : HasSubset.Subset D C
      hD₂ : LE.le (MeasureTheory.VectorMeasure.restrict s D) (MeasureTheory.VectorMe …
      hD₃ : LT.lt (↑s D) 0
      this : LT.lt (↑s (Union.union A D)) (InfSet.sInf s.measureOfNegatives)
      ⊢ LE.le (MeasureTheory.VectorMeasure.restrict s (Union.union A D)) (MeasureThe …
    -/
  · exact restrict_le_restrict_union _ _ hA₁ hA₂ hD₁ hD₂
    /-
      🎉 no goals
    -/


/-- **The Hahn decomposition theorem**: Given a signed measure `s`, there exist
complement measurable sets `i` and `j` such that `i` is positive, `j` is negative. -/
theorem exists_isCompl_positive_negative (s : SignedMeasure α) :
    ∃ i j : Set α, MeasurableSet i ∧ 0 ≤[i] s ∧ MeasurableSet j ∧ s ≤[j] 0 ∧ IsCompl i j :=
  let ⟨i, hi₁, hi₂, hi₃⟩ := exists_compl_positive_negative s
  ⟨i, iᶜ, hi₁, hi₂, hi₁.compl, hi₃, isCompl_compl⟩


open scoped symmDiff in
/-- The symmetric difference of two Hahn decompositions has measure zero. -/
theorem of_symmDiff_compl_positive_negative {s : SignedMeasure α} {i j : Set α}
    (hi : MeasurableSet i) (hj : MeasurableSet j) (hi' : 0 ≤[i] s ∧ s ≤[iᶜ] 0)
    (hj' : 0 ≤[j] s ∧ s ≤[jᶜ] 0) : s (i ∆ j) = 0 ∧ s (iᶜ ∆ jᶜ) = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i j : Set α
    hi : MeasurableSet i
    hj : MeasurableSet j
    hi' : And (LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.Vec …
    hj' : And (LE.le (MeasureTheory.VectorMeasure.restrict 0 j) (MeasureTheory.Vec …
    ⊢ And (Eq (↑s (symmDiff i j)) 0) (Eq (↑s (symmDiff (HasCompl.compl i) (HasComp …
  -/
  rw [restrict_le_restrict_iff s 0, restrict_le_restrict_iff 0 s] at hi' hj'
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i j : Set α
    hi : MeasurableSet i
    hj : MeasurableSet j
    hi' : And (∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑0 j …
    hj' : And (∀ ⦃j_1 : Set α⦄, MeasurableSet j_1 → HasSubset.Subset j_1 j → LE.le …
    ⊢ And (Eq (↑s (symmDiff i j)) 0) (Eq (↑s (symmDiff (HasCompl.compl i) (HasComp …
  -/
  constructor
  · rw [Set.symmDiff_def, Set.diff_eq_compl_inter, Set.diff_eq_compl_inter, of_union,
      le_antisymm (hi'.2 (hi.compl.inter hj) Set.inter_subset_left)
        (hj'.1 (hi.compl.inter hj) Set.inter_subset_right),
      le_antisymm (hj'.2 (hj.compl.inter hi) Set.inter_subset_left)
        (hi'.1 (hj.compl.inter hi) Set.inter_subset_right),
      zero_apply, zero_apply, zero_add]
    · exact
        Set.disjoint_of_subset_left Set.inter_subset_left
          (Set.disjoint_of_subset_right Set.inter_subset_right
            (disjoint_comm.1 (IsCompl.disjoint isCompl_compl)))
      /-
        case left.hA
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i j : Set α
        hi : MeasurableSet i
        hj : MeasurableSet j
        hi' : And (∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑0 j …
        hj' : And (∀ ⦃j_1 : Set α⦄, MeasurableSet j_1 → HasSubset.Subset j_1 j → LE.le …
        ⊢ MeasurableSet (Inter.inter (HasCompl.compl j) i)
      -/
    · exact hj.compl.inter hi
      /-
        🎉 no goals
      -/
      /-
        case left.hB
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i j : Set α
        hi : MeasurableSet i
        hj : MeasurableSet j
        hi' : And (∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑0 j …
        hj' : And (∀ ⦃j_1 : Set α⦄, MeasurableSet j_1 → HasSubset.Subset j_1 j → LE.le …
        ⊢ MeasurableSet (Inter.inter (HasCompl.compl i) j)
      -/
    · exact hi.compl.inter hj
      /-
        🎉 no goals
      -/
  · rw [Set.symmDiff_def, Set.diff_eq_compl_inter, Set.diff_eq_compl_inter, compl_compl,
      compl_compl, of_union,
      le_antisymm (hi'.2 (hj.inter hi.compl) Set.inter_subset_right)
        (hj'.1 (hj.inter hi.compl) Set.inter_subset_left),
      le_antisymm (hj'.2 (hi.inter hj.compl) Set.inter_subset_right)
        (hi'.1 (hi.inter hj.compl) Set.inter_subset_left),
      zero_apply, zero_apply, zero_add]
    · exact
        Set.disjoint_of_subset_left Set.inter_subset_left
          (Set.disjoint_of_subset_right Set.inter_subset_right
            (IsCompl.disjoint isCompl_compl))
      /-
        case right.hA
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i j : Set α
        hi : MeasurableSet i
        hj : MeasurableSet j
        hi' : And (∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑0 j …
        hj' : And (∀ ⦃j_1 : Set α⦄, MeasurableSet j_1 → HasSubset.Subset j_1 j → LE.le …
        ⊢ MeasurableSet (Inter.inter j (HasCompl.compl i))
      -/
    · exact hj.inter hi.compl
      /-
        🎉 no goals
      -/
      /-
        case right.hB
        α : Type u_1
        inst✝ : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i j : Set α
        hi : MeasurableSet i
        hj : MeasurableSet j
        hi' : And (∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑0 j …
        hj' : And (∀ ⦃j_1 : Set α⦄, MeasurableSet j_1 → HasSubset.Subset j_1 j → LE.le …
        ⊢ MeasurableSet (Inter.inter i (HasCompl.compl j))
      -/
    · exact hi.inter hj.compl
      /-
        🎉 no goals
      -/
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i j : Set α
    hi : MeasurableSet i
    hj : MeasurableSet j
    hi' : And (∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑0 j …
    hj' : And (LE.le (MeasureTheory.VectorMeasure.restrict 0 j) (MeasureTheory.Vec …
    ⊢ MeasurableSet j
  -/
  all_goals measurability
  /-
    🎉 no goals
  -/


