/-- **Hahn decomposition theorem** -/
theorem hahn_decomposition (μ ν : Measure α) [IsFiniteMeasure μ] [IsFiniteMeasure ν] :
    ∃ s, MeasurableSet s ∧ (∀ t, MeasurableSet t → t ⊆ s → ν t ≤ μ t) ∧
      ∀ t, MeasurableSet t → t ⊆ sᶜ → μ t ≤ ν t := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  let d : Set α → ℝ := fun s => ((μ s).toNNReal : ℝ) - (ν s).toNNReal
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  let c : Set ℝ := d '' { s | MeasurableSet s }
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  let γ : ℝ := sSup c
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have hμ : ∀ s, μ s ≠ ∞ := measure_ne_top μ
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have hν : ∀ s, ν s ≠ ∞ := measure_ne_top ν
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have to_nnreal_μ : ∀ s, ((μ s).toNNReal : ℝ≥0∞) = μ s := fun s => ENNReal.coe_toNNReal <| hμ _
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have to_nnreal_ν : ∀ s, ((ν s).toNNReal : ℝ≥0∞) = ν s := fun s => ENNReal.coe_toNNReal <| hν _
  have d_split s t (ht : MeasurableSet t) : d s = d (s \ t) + d (s ∩ t) := by
    dsimp only [d]
    rw [← measure_inter_add_diff s ht, ← measure_inter_add_diff s ht,
      ENNReal.toNNReal_add (hμ _) (hμ _), ENNReal.toNNReal_add (hν _) (hν _), NNReal.coe_add,
      NNReal.coe_add]
    simp only [sub_eq_add_neg, neg_add]
    abel
  have d_Union (s : ℕ → Set α) (hm : Monotone s) :
    Tendsto (fun n => d (s n)) atTop (𝓝 (d (⋃ n, s n))) := by
    refine Tendsto.sub ?_ ?_ <;>
      refine NNReal.tendsto_coe.2 <| (ENNReal.tendsto_toNNReal ?_).comp <|
        tendsto_measure_iUnion_atTop hm
    · exact hμ _
    · exact hν _
  have d_Inter (s : ℕ → Set α) (hs : ∀ n, MeasurableSet (s n)) (hm : ∀ n m, n ≤ m → s m ⊆ s n) :
        Tendsto (fun n => d (s n)) atTop (𝓝 (d (⋂ n, s n))) := by
    refine Tendsto.sub ?_ ?_ <;>
      refine NNReal.tendsto_coe.2 <| (ENNReal.tendsto_toNNReal <| ?_).comp <|
        tendsto_measure_iInter_atTop (fun n ↦ (hs n).nullMeasurableSet) hm ?_
    exacts [hμ _, ⟨0, hμ _⟩, hν _, ⟨0, hν _⟩]
  have bdd_c : BddAbove c := by
    use (μ univ).toNNReal
    rintro r ⟨s, _hs, rfl⟩
    refine le_trans (sub_le_self _ <| NNReal.coe_nonneg _) ?_
    rw [NNReal.coe_le_coe, ← ENNReal.coe_le_coe, to_nnreal_μ, to_nnreal_μ]
    exact measure_mono (subset_univ _)
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have c_nonempty : c.Nonempty := Nonempty.image _ ⟨_, MeasurableSet.empty⟩
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have d_le_γ : ∀ s, MeasurableSet s → d s ≤ γ := fun s hs => le_csSup bdd_c ⟨s, hs, rfl⟩
  have (n : ℕ) : ∃ s : Set α, MeasurableSet s ∧ γ - (1 / 2) ^ n < d s := by
    have : γ - (1 / 2) ^ n < γ := sub_lt_self γ (pow_pos (half_pos zero_lt_one) n)
    rcases exists_lt_of_lt_csSup c_nonempty this with ⟨r, ⟨s, hs, rfl⟩, hlt⟩
    exact ⟨s, hs, hlt⟩
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
    this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  rcases Classical.axiom_of_choice this with ⟨e, he⟩
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
    this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
    e : Nat → Set α
    he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  change ℕ → Set α at e
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
    this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
    e : Nat → Set α
    he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have he₁ : ∀ n, MeasurableSet (e n) := fun n => (he n).1
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
    this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
    e : Nat → Set α
    he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
    he₁ : ∀ (n : Nat), MeasurableSet (e n)
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have he₂ : ∀ n, γ - (1 / 2) ^ n < d (e n) := fun n => (he n).2
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
    this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
    e : Nat → Set α
    he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
    he₁ : ∀ (n : Nat), MeasurableSet (e n)
    he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  let f : ℕ → ℕ → Set α := fun n m => (Finset.Ico n (m + 1)).inf e
  have hf n m : MeasurableSet (f n m) := by
    simp only [f, Finset.inf_eq_iInf]
    exact MeasurableSet.biInter (to_countable _) fun i _ => he₁ _
  have f_subset_f {a b c d} (hab : a ≤ b) (hcd : c ≤ d) : f a d ⊆ f b c := by
    simp_rw [f, Finset.inf_eq_iInf]
    exact biInter_subset_biInter_left (Finset.Ico_subset_Ico hab <| Nat.succ_le_succ hcd)
  have f_succ n m (hnm : n ≤ m) : f n (m + 1) = f n m ∩ e (m + 1) := by
    have : n ≤ m + 1 := le_of_lt (Nat.succ_le_succ hnm)
    simp_rw [f, Nat.Ico_succ_right_eq_insert_Ico this, Finset.inf_insert, Set.inter_comm]
    rfl
  have le_d_f n m (h : m ≤ n) : γ - 2 * (1 / 2) ^ m + (1 / 2) ^ n ≤ d (f m n) := by
    refine Nat.le_induction ?_ ?_ n h
    · have := he₂ m
      simp_rw [f, Nat.Ico_succ_singleton, Finset.inf_singleton]
      linarith
    · intro n (hmn : m ≤ n) ih
      have : γ + (γ - 2 * (1 / 2) ^ m + (1 / 2) ^ (n + 1)) ≤ γ + d (f m (n + 1)) := by
        calc
          γ + (γ - 2 * (1 / 2) ^ m + (1 / 2) ^ (n + 1)) =
              γ + (γ - 2 * (1 / 2) ^ m + ((1 / 2) ^ n - (1 / 2) ^ (n + 1))) := by
            rw [pow_succ, mul_one_div, _root_.sub_half]
          _ = γ - (1 / 2) ^ (n + 1) + (γ - 2 * (1 / 2) ^ m + (1 / 2) ^ n) := by
            simp only [sub_eq_add_neg]; abel
          _ ≤ d (e (n + 1)) + d (f m n) := add_le_add (le_of_lt <| he₂ _) ih
          _ ≤ d (e (n + 1)) + d (f m n \ e (n + 1)) + d (f m (n + 1)) := by
            rw [f_succ _ _ hmn, d_split (f m n) (e (n + 1)) (he₁ _), add_assoc]
          _ = d (e (n + 1) ∪ f m n) + d (f m (n + 1)) := by
            rw [d_split (e (n + 1) ∪ f m n) (e (n + 1)), union_diff_left, union_inter_cancel_left]
            · abel
            · exact he₁ _
          _ ≤ γ + d (f m (n + 1)) := add_le_add_right (d_le_γ _ <| (he₁ _).union (hf _ _)) _
      exact (add_le_add_iff_left γ).1 this
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
    this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
    e : Nat → Set α
    he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
    he₁ : ∀ (n : Nat), MeasurableSet (e n)
    he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
    f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
    hf : ∀ (n m : Nat), MeasurableSet (f n m)
    f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
    f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
    le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  let s := ⋃ m, ⋂ n, f m n
  have γ_le_d_s : γ ≤ d s := by
    have hγ : Tendsto (fun m : ℕ => γ - 2 * (1 / 2) ^ m) atTop (𝓝 γ) := by
      suffices Tendsto (fun m : ℕ => γ - 2 * (1 / 2) ^ m) atTop (𝓝 (γ - 2 * 0)) by
        simpa only [mul_zero, tsub_zero]
      exact
        tendsto_const_nhds.sub <|
          tendsto_const_nhds.mul <|
            tendsto_pow_atTop_nhds_zero_of_lt_one (le_of_lt <| half_pos <| zero_lt_one)
              (half_lt_self zero_lt_one)
    have hd : Tendsto (fun m => d (⋂ n, f m n)) atTop (𝓝 (d (⋃ m, ⋂ n, f m n))) := by
      refine d_Union _ ?_
      exact fun n m hnm =>
        subset_iInter fun i => Subset.trans (iInter_subset (f n) i) <| f_subset_f hnm <| le_rfl
    refine le_of_tendsto_of_tendsto' hγ hd fun m => ?_
    have : Tendsto (fun n => d (f m n)) atTop (𝓝 (d (⋂ n, f m n))) := by
      refine d_Inter _ ?_ ?_
      · intro n
        exact hf _ _
      · intro n m hnm
        exact f_subset_f le_rfl hnm
    refine ge_of_tendsto this (eventually_atTop.2 ⟨m, fun n hmn => ?_⟩)
    change γ - 2 * (1 / 2) ^ m ≤ d (f m n)
    refine le_trans ?_ (le_d_f _ _ hmn)
    exact le_add_of_le_of_nonneg le_rfl (pow_nonneg (le_of_lt <| half_pos <| zero_lt_one) _)
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
    this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
    e : Nat → Set α
    he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
    he₁ : ∀ (n : Nat), MeasurableSet (e n)
    he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
    f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
    hf : ∀ (n m : Nat), MeasurableSet (f n m)
    f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
    f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
    le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
    s : Set α := Set.iUnion fun m => Set.iInter fun n => f m n
    γ_le_d_s : LE.le γ (d s)
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  have hs : MeasurableSet s := MeasurableSet.iUnion fun n => MeasurableSet.iInter fun m => hf _ _
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
    c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
    γ : Real := SupSet.sSup c
    hμ : ∀ (s : Set α), Ne (μ s) Top.top
    hν : ∀ (s : Set α), Ne (ν s) Top.top
    to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
    to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
    d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
    d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
    d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
    bdd_c : BddAbove c
    c_nonempty : c.Nonempty
    d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
    this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
    e : Nat → Set α
    he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
    he₁ : ∀ (n : Nat), MeasurableSet (e n)
    he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
    f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
    hf : ∀ (n m : Nat), MeasurableSet (f n m)
    f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
    f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
    le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
    s : Set α := Set.iUnion fun m => Set.iInter fun n => f m n
    γ_le_d_s : LE.le γ (d s)
    hs : MeasurableSet s
    ⊢ Exists fun s => And (MeasurableSet s) (And (∀ (t : Set α), MeasurableSet t → …
  -/
  refine ⟨s, hs, ?_, ?_⟩
    /-
      case intro.refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
      c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
      γ : Real := SupSet.sSup c
      hμ : ∀ (s : Set α), Ne (μ s) Top.top
      hν : ∀ (s : Set α), Ne (ν s) Top.top
      to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
      to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
      d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
      d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
      d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
      bdd_c : BddAbove c
      c_nonempty : c.Nonempty
      d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
      this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
      e : Nat → Set α
      he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
      he₁ : ∀ (n : Nat), MeasurableSet (e n)
      he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
      f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
      hf : ∀ (n m : Nat), MeasurableSet (f n m)
      f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
      f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
      le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
      s : Set α := Set.iUnion fun m => Set.iInter fun n => f m n
      γ_le_d_s : LE.le γ (d s)
      hs : MeasurableSet s
      ⊢ ∀ (t : Set α), MeasurableSet t → HasSubset.Subset t s → LE.le (ν t) (μ t)
    -/
  · intro t ht hts
    have : 0 ≤ d t :=
      (add_le_add_iff_left γ).1 <|
        calc
          γ + 0 ≤ d s := by rw [add_zero]; exact γ_le_d_s
          _ = d (s \ t) + d t := by rw [d_split s _ ht, inter_eq_self_of_subset_right hts]
          _ ≤ γ + d t := add_le_add (d_le_γ _ (hs.diff ht)) le_rfl

    /-
      case intro.refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
      c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
      γ : Real := SupSet.sSup c
      hμ : ∀ (s : Set α), Ne (μ s) Top.top
      hν : ∀ (s : Set α), Ne (ν s) Top.top
      to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
      to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
      d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
      d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
      d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
      bdd_c : BddAbove c
      c_nonempty : c.Nonempty
      d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
      this✝ : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ …
      e : Nat → Set α
      he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
      he₁ : ∀ (n : Nat), MeasurableSet (e n)
      he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
      f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
      hf : ∀ (n m : Nat), MeasurableSet (f n m)
      f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
      f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
      le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
      s : Set α := Set.iUnion fun m => Set.iInter fun n => f m n
      γ_le_d_s : LE.le γ (d s)
      hs : MeasurableSet s
      t : Set α
      ht : MeasurableSet t
      hts : HasSubset.Subset t s
      this : LE.le 0 (d t)
      ⊢ LE.le (ν t) (μ t)
    -/
    rw [← to_nnreal_μ, ← to_nnreal_ν, ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
    /-
      case intro.refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
      c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
      γ : Real := SupSet.sSup c
      hμ : ∀ (s : Set α), Ne (μ s) Top.top
      hν : ∀ (s : Set α), Ne (ν s) Top.top
      to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
      to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
      d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
      d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
      d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
      bdd_c : BddAbove c
      c_nonempty : c.Nonempty
      d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
      this✝ : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ …
      e : Nat → Set α
      he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
      he₁ : ∀ (n : Nat), MeasurableSet (e n)
      he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
      f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
      hf : ∀ (n m : Nat), MeasurableSet (f n m)
      f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
      f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
      le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
      s : Set α := Set.iUnion fun m => Set.iInter fun n => f m n
      γ_le_d_s : LE.le γ (d s)
      hs : MeasurableSet s
      t : Set α
      ht : MeasurableSet t
      hts : HasSubset.Subset t s
      this : LE.le 0 (d t)
      ⊢ LE.le ↑(ν t).toNNReal ↑(μ t).toNNReal
    -/
    simpa only [d, le_sub_iff_add_le, zero_add] using this
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
      c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
      γ : Real := SupSet.sSup c
      hμ : ∀ (s : Set α), Ne (μ s) Top.top
      hν : ∀ (s : Set α), Ne (ν s) Top.top
      to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
      to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
      d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
      d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
      d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
      bdd_c : BddAbove c
      c_nonempty : c.Nonempty
      d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
      this : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ  …
      e : Nat → Set α
      he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
      he₁ : ∀ (n : Nat), MeasurableSet (e n)
      he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
      f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
      hf : ∀ (n m : Nat), MeasurableSet (f n m)
      f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
      f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
      le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
      s : Set α := Set.iUnion fun m => Set.iInter fun n => f m n
      γ_le_d_s : LE.le γ (d s)
      hs : MeasurableSet s
      ⊢ ∀ (t : Set α), MeasurableSet t → HasSubset.Subset t (HasCompl.compl s) → LE. …
    -/
  · intro t ht hts
    have : d t ≤ 0 :=
      (add_le_add_iff_left γ).1 <|
        calc
          γ + d t ≤ d s + d t := by gcongr
          _ = d (s ∪ t) := by
            rw [d_split (s ∪ t) _ ht, union_diff_right, union_inter_cancel_right,
              (subset_compl_iff_disjoint_left.1 hts).sdiff_eq_left]
          _ ≤ γ + 0 := by rw [add_zero]; exact d_le_γ _ (hs.union ht)

    /-
      case intro.refine_2
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
      c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
      γ : Real := SupSet.sSup c
      hμ : ∀ (s : Set α), Ne (μ s) Top.top
      hν : ∀ (s : Set α), Ne (ν s) Top.top
      to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
      to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
      d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
      d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
      d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
      bdd_c : BddAbove c
      c_nonempty : c.Nonempty
      d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
      this✝ : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ …
      e : Nat → Set α
      he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
      he₁ : ∀ (n : Nat), MeasurableSet (e n)
      he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
      f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
      hf : ∀ (n m : Nat), MeasurableSet (f n m)
      f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
      f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
      le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
      s : Set α := Set.iUnion fun m => Set.iInter fun n => f m n
      γ_le_d_s : LE.le γ (d s)
      hs : MeasurableSet s
      t : Set α
      ht : MeasurableSet t
      hts : HasSubset.Subset t (HasCompl.compl s)
      this : LE.le (d t) 0
      ⊢ LE.le (μ t) (ν t)
    -/
    rw [← to_nnreal_μ, ← to_nnreal_ν, ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
    /-
      case intro.refine_2
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      d : Set α → Real := fun s => HSub.hSub ↑(μ s).toNNReal ↑(ν s).toNNReal
      c : Set Real := Set.image d (setOf fun s => MeasurableSet s)
      γ : Real := SupSet.sSup c
      hμ : ∀ (s : Set α), Ne (μ s) Top.top
      hν : ∀ (s : Set α), Ne (ν s) Top.top
      to_nnreal_μ : ∀ (s : Set α), Eq (↑(μ s).toNNReal) (μ s)
      to_nnreal_ν : ∀ (s : Set α), Eq (↑(ν s).toNNReal) (ν s)
      d_split : ∀ (s t : Set α), MeasurableSet t → Eq (d s) (HAdd.hAdd (d (SDiff.sdi …
      d_Union : ∀ (s : Nat → Set α), Monotone s → Filter.Tendsto (fun n => d (s n))  …
      d_Inter : ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n m :  …
      bdd_c : BddAbove c
      c_nonempty : c.Nonempty
      d_le_γ : ∀ (s : Set α), MeasurableSet s → LE.le (d s) γ
      this✝ : ∀ (n : Nat), Exists fun s => And (MeasurableSet s) (LT.lt (HSub.hSub γ …
      e : Nat → Set α
      he : ∀ (x : Nat), And (MeasurableSet (e x)) (LT.lt (HSub.hSub γ (HPow.hPow (1  …
      he₁ : ∀ (n : Nat), MeasurableSet (e n)
      he₂ : ∀ (n : Nat), LT.lt (HSub.hSub γ (HPow.hPow (1 / 2) n)) (d (e n))
      f : Nat → Nat → Set α := fun n m => (Finset.Ico n (HAdd.hAdd m 1)).inf e
      hf : ∀ (n m : Nat), MeasurableSet (f n m)
      f_subset_f : ∀ {a b c d : Nat}, LE.le a b → LE.le c d → HasSubset.Subset (f a  …
      f_succ : ∀ (n m : Nat), LE.le n m → Eq (f n (HAdd.hAdd m 1)) (Inter.inter (f n …
      le_d_f : ∀ (n m : Nat), LE.le m n → LE.le (HAdd.hAdd (HSub.hSub γ (HMul.hMul 2 …
      s : Set α := Set.iUnion fun m => Set.iInter fun n => f m n
      γ_le_d_s : LE.le γ (d s)
      hs : MeasurableSet s
      t : Set α
      ht : MeasurableSet t
      hts : HasSubset.Subset t (HasCompl.compl s)
      this : LE.le (d t) 0
      ⊢ LE.le ↑(μ t).toNNReal ↑(ν t).toNNReal
    -/
    simpa only [d, sub_le_iff_le_add, zero_add] using this
    /-
      🎉 no goals
    -/


