local infixr:25 " →ₛ " => SimpleFunc


/-- A set function is `FinMeasAdditive` if its value on the union of two disjoint measurable
sets with finite measure is the sum of its values on each set. -/
def FinMeasAdditive {β} [AddMonoid β] {_ : MeasurableSpace α} (μ : Measure α) (T : Set α → β) :
    Prop :=
  ∀ s t, MeasurableSet s → MeasurableSet t → μ s ≠ ∞ → μ t ≠ ∞ → Disjoint s t →
    T (s ∪ t) = T s + T t


                                                                            /-
                                                                              α : Type u_1
                                                                              m : MeasurableSpace α
                                                                              μ : MeasureTheory.Measure α
                                                                              β : Type u_7
                                                                              inst✝ : AddCommMonoid β
                                                                              x✝⁶ x✝⁵ : Set α
                                                                              x✝⁴ : MeasurableSet x✝⁶
                                                                              x✝³ : MeasurableSet x✝⁵
                                                                              x✝² : Ne (μ x✝⁶) Top.top
                                                                              x✝¹ : Ne (μ x✝⁵) Top.top
                                                                              x✝ : Disjoint x✝⁶ x✝⁵
                                                                              ⊢ Eq (0 (Union.union x✝⁶ x✝⁵)) (HAdd.hAdd (0 x✝⁶) (0 x✝⁵))
                                                                            -/
theorem zero : FinMeasAdditive μ (0 : Set α → β) := fun _ _ _ _ _ _ _ => by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem add (hT : FinMeasAdditive μ T) (hT' : FinMeasAdditive μ T') :
    FinMeasAdditive μ (T + T') := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T T' : Set α → β
    hT : MeasureTheory.FinMeasAdditive μ T
    hT' : MeasureTheory.FinMeasAdditive μ T'
    ⊢ MeasureTheory.FinMeasAdditive μ (HAdd.hAdd T T')
  -/
  intro s t hs ht hμs hμt hst
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T T' : Set α → β
    hT : MeasureTheory.FinMeasAdditive μ T
    hT' : MeasureTheory.FinMeasAdditive μ T'
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    ⊢ Eq (HAdd.hAdd T T' (Union.union s t)) (HAdd.hAdd (HAdd.hAdd T T' s) (HAdd.hA …
  -/
  simp only [hT s t hs ht hμs hμt hst, hT' s t hs ht hμs hμt hst, Pi.add_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T T' : Set α → β
    hT : MeasureTheory.FinMeasAdditive μ T
    hT' : MeasureTheory.FinMeasAdditive μ T'
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (T s) (T t)) (HAdd.hAdd (T' s) (T' t))) (HAdd.hAdd  …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem smul [Monoid 𝕜] [DistribMulAction 𝕜 β] (hT : FinMeasAdditive μ T) (c : 𝕜) :
    FinMeasAdditive μ fun s => c • T s := fun s t hs ht hμs hμt hst => by
  /-
    α : Type u_1
    𝕜 : Type u_6
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝² : AddCommMonoid β
    T : Set α → β
    inst✝¹ : Monoid 𝕜
    inst✝ : DistribMulAction 𝕜 β
    hT : MeasureTheory.FinMeasAdditive μ T
    c : 𝕜
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    ⊢ Eq ((fun s => HSMul.hSMul c (T s)) (Union.union s t)) (HAdd.hAdd ((fun s =>  …
  -/
  simp [hT s t hs ht hμs hμt hst]
  /-
    🎉 no goals
  -/


theorem of_eq_top_imp_eq_top {μ' : Measure α} (h : ∀ s, MeasurableSet s → μ s = ∞ → μ' s = ∞)
    (hT : FinMeasAdditive μ T) : FinMeasAdditive μ' T := fun s t hs ht hμ's hμ't hst =>
  hT s t hs ht (mt (h s hs) hμ's) (mt (h t ht) hμ't) hst


theorem of_smul_measure (c : ℝ≥0∞) (hc_ne_top : c ≠ ∞) (hT : FinMeasAdditive (c • μ) T) :
    FinMeasAdditive μ T := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T : Set α → β
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.FinMeasAdditive (HSMul.hSMul c μ) T
    ⊢ MeasureTheory.FinMeasAdditive μ T
  -/
  refine of_eq_top_imp_eq_top (fun s _ hμs => ?_) hT
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T : Set α → β
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.FinMeasAdditive (HSMul.hSMul c μ) T
    s : Set α
    x✝ : MeasurableSet s
    hμs : Eq ((HSMul.hSMul c μ) s) Top.top
    ⊢ Eq (μ s) Top.top
  -/
  rw [Measure.smul_apply, smul_eq_mul, ENNReal.mul_eq_top] at hμs
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T : Set α → β
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.FinMeasAdditive (HSMul.hSMul c μ) T
    s : Set α
    x✝ : MeasurableSet s
    hμs : Or (And (Ne c 0) (Eq (μ s) Top.top)) (And (Eq c Top.top) (Ne (μ s) 0))
    ⊢ Eq (μ s) Top.top
  -/
  simp only [hc_ne_top, or_false, Ne, false_and] at hμs
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T : Set α → β
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.FinMeasAdditive (HSMul.hSMul c μ) T
    s : Set α
    x✝ : MeasurableSet s
    hμs : And (Not (Eq c 0)) (Eq (μ s) Top.top)
    ⊢ Eq (μ s) Top.top
  -/
  exact hμs.2
  /-
    🎉 no goals
  -/


theorem smul_measure (c : ℝ≥0∞) (hc_ne_zero : c ≠ 0) (hT : FinMeasAdditive μ T) :
    FinMeasAdditive (c • μ) T := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T : Set α → β
    c : ENNReal
    hc_ne_zero : Ne c 0
    hT : MeasureTheory.FinMeasAdditive μ T
    ⊢ MeasureTheory.FinMeasAdditive (HSMul.hSMul c μ) T
  -/
  refine of_eq_top_imp_eq_top (fun s _ hμs => ?_) hT
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T : Set α → β
    c : ENNReal
    hc_ne_zero : Ne c 0
    hT : MeasureTheory.FinMeasAdditive μ T
    s : Set α
    x✝ : MeasurableSet s
    hμs : Eq (μ s) Top.top
    ⊢ Eq ((HSMul.hSMul c μ) s) Top.top
  -/
  rw [Measure.smul_apply, smul_eq_mul, ENNReal.mul_eq_top]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T : Set α → β
    c : ENNReal
    hc_ne_zero : Ne c 0
    hT : MeasureTheory.FinMeasAdditive μ T
    s : Set α
    x✝ : MeasurableSet s
    hμs : Eq (μ s) Top.top
    ⊢ Or (And (Ne c 0) (Eq (μ s) Top.top)) (And (Eq c Top.top) (Ne (μ s) 0))
  -/
  simp only [hc_ne_zero, true_and, Ne, not_false_iff]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : AddCommMonoid β
    T : Set α → β
    c : ENNReal
    hc_ne_zero : Ne c 0
    hT : MeasureTheory.FinMeasAdditive μ T
    s : Set α
    x✝ : MeasurableSet s
    hμs : Eq (μ s) Top.top
    ⊢ Or (Eq (μ s) Top.top) (And (Eq c Top.top) (Not (Eq (μ s) 0)))
  -/
  exact Or.inl hμs
  /-
    🎉 no goals
  -/


theorem smul_measure_iff (c : ℝ≥0∞) (hc_ne_zero : c ≠ 0) (hc_ne_top : c ≠ ∞) :
    FinMeasAdditive (c • μ) T ↔ FinMeasAdditive μ T :=
  ⟨fun hT => of_smul_measure c hc_ne_top hT, fun hT => smul_measure c hc_ne_zero hT⟩


theorem map_empty_eq_zero {β} [AddCancelMonoid β] {T : Set α → β} (hT : FinMeasAdditive μ T) :
    T ∅ = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_8
    inst✝ : AddCancelMonoid β
    T : Set α → β
    hT : MeasureTheory.FinMeasAdditive μ T
    ⊢ Eq (T EmptyCollection.emptyCollection) 0
  -/
  have h_empty : μ ∅ ≠ ∞ := (measure_empty.le.trans_lt ENNReal.coe_lt_top).ne
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_8
    inst✝ : AddCancelMonoid β
    T : Set α → β
    hT : MeasureTheory.FinMeasAdditive μ T
    h_empty : Ne (μ EmptyCollection.emptyCollection) Top.top
    ⊢ Eq (T EmptyCollection.emptyCollection) 0
  -/
  specialize hT ∅ ∅ MeasurableSet.empty MeasurableSet.empty h_empty h_empty (disjoint_empty _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_8
    inst✝ : AddCancelMonoid β
    T : Set α → β
    h_empty : Ne (μ EmptyCollection.emptyCollection) Top.top
    hT : Eq (T (Union.union EmptyCollection.emptyCollection EmptyCollection.emptyC …
    ⊢ Eq (T EmptyCollection.emptyCollection) 0
  -/
  rw [Set.union_empty] at hT
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_8
    inst✝ : AddCancelMonoid β
    T : Set α → β
    h_empty : Ne (μ EmptyCollection.emptyCollection) Top.top
    hT : Eq (T EmptyCollection.emptyCollection) (HAdd.hAdd (T EmptyCollection.empt …
    ⊢ Eq (T EmptyCollection.emptyCollection) 0
  -/
  nth_rw 1 [← add_zero (T ∅)] at hT
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_8
    inst✝ : AddCancelMonoid β
    T : Set α → β
    h_empty : Ne (μ EmptyCollection.emptyCollection) Top.top
    hT : Eq (HAdd.hAdd (T EmptyCollection.emptyCollection) 0) (HAdd.hAdd (T EmptyC …
    ⊢ Eq (T EmptyCollection.emptyCollection) 0
  -/
  exact (add_left_cancel hT).symm
  /-
    🎉 no goals
  -/


theorem map_iUnion_fin_meas_set_eq_sum (T : Set α → β) (T_empty : T ∅ = 0)
    (h_add : FinMeasAdditive μ T) {ι} (S : ι → Set α) (sι : Finset ι)
    (hS_meas : ∀ i, MeasurableSet (S i)) (hSp : ∀ i ∈ sι, μ (S i) ≠ ∞)
    (h_disj : ∀ᵉ (i ∈ sι) (j ∈ sι), i ≠ j → Disjoint (S i) (S j)) :
    T (⋃ i ∈ sι, S i) = ∑ i ∈ sι, T (S i) := by
  classical
  revert hSp h_disj
  refine Finset.induction_on sι ?_ ?_
  · simp only [Finset.not_mem_empty, IsEmpty.forall_iff, iUnion_false, iUnion_empty, sum_empty,
      forall₂_true_iff, imp_true_iff, forall_true_left, not_false_iff, T_empty]
  intro a s has h hps h_disj
  rw [Finset.sum_insert has, ← h]
  swap; · exact fun i hi => hps i (Finset.mem_insert_of_mem hi)
  swap
  · exact fun i hi j hj hij =>
      h_disj i (Finset.mem_insert_of_mem hi) j (Finset.mem_insert_of_mem hj) hij
  rw [←
    h_add (S a) (⋃ i ∈ s, S i) (hS_meas a) (measurableSet_biUnion _ fun i _ => hS_meas i)
      (hps a (Finset.mem_insert_self a s))]
  · congr; convert Finset.iSup_insert a s S
  · exact (measure_biUnion_lt_top s.finite_toSet fun i hi ↦
      (hps i <| Finset.mem_insert_of_mem hi).lt_top).ne
  · simp_rw [Set.disjoint_iUnion_right]
    intro i hi
    refine h_disj a (Finset.mem_insert_self a s) i (Finset.mem_insert_of_mem hi) fun hai ↦ ?_
    rw [← hai] at hi
    exact has hi


/-- A `FinMeasAdditive` set function whose norm on every set is less than the measure of the
set (up to a multiplicative constant). -/
def DominatedFinMeasAdditive {β} [SeminormedAddCommGroup β] {_ : MeasurableSpace α} (μ : Measure α)
    (T : Set α → β) (C : ℝ) : Prop :=
  FinMeasAdditive μ T ∧ ∀ s, MeasurableSet s → μ s < ∞ → ‖T s‖ ≤ C * (μ s).toReal


theorem zero {m : MeasurableSpace α} (μ : Measure α) (hC : 0 ≤ C) :
    DominatedFinMeasAdditive μ (0 : Set α → β) C := by
  /-
    α : Type u_1
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    C : Real
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hC : LE.le 0 C
    ⊢ MeasureTheory.DominatedFinMeasAdditive μ 0 C
  -/
  refine ⟨FinMeasAdditive.zero, fun s _ _ => ?_⟩
  /-
    α : Type u_1
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    C : Real
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hC : LE.le 0 C
    s : Set α
    x✝¹ : MeasurableSet s
    x✝ : LT.lt (μ s) Top.top
    ⊢ LE.le (Norm.norm (0 s)) (HMul.hMul C (μ s).toReal)
  -/
  rw [Pi.zero_apply, norm_zero]
  /-
    α : Type u_1
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    C : Real
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hC : LE.le 0 C
    s : Set α
    x✝¹ : MeasurableSet s
    x✝ : LT.lt (μ s) Top.top
    ⊢ LE.le 0 (HMul.hMul C (μ s).toReal)
  -/
  exact mul_nonneg hC toReal_nonneg
  /-
    🎉 no goals
  -/


theorem eq_zero_of_measure_zero {β : Type*} [NormedAddCommGroup β] {T : Set α → β} {C : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) {s : Set α} (hs : MeasurableSet s) (hs_zero : μ s = 0) :
    T s = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_8
    inst✝ : NormedAddCommGroup β
    T : Set α → β
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (μ s) 0
    ⊢ Eq (T s) 0
  -/
  refine norm_eq_zero.mp ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_8
    inst✝ : NormedAddCommGroup β
    T : Set α → β
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (μ s) 0
    ⊢ Eq (Norm.norm (T s)) 0
  -/
  refine ((hT.2 s hs (by simp [hs_zero])).trans (le_of_eq ?_)).antisymm (norm_nonneg _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_8
    inst✝ : NormedAddCommGroup β
    T : Set α → β
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hs_zero : Eq (μ s) 0
    ⊢ Eq (HMul.hMul C (μ s).toReal) 0
  -/
  rw [hs_zero, ENNReal.zero_toReal, mul_zero]
  /-
    🎉 no goals
  -/


theorem eq_zero {β : Type*} [NormedAddCommGroup β] {T : Set α → β} {C : ℝ} {_ : MeasurableSpace α}
    (hT : DominatedFinMeasAdditive (0 : Measure α) T C) {s : Set α} (hs : MeasurableSet s) :
    T s = 0 :=
                                    /-
                                      α : Type u_1
                                      β : Type u_8
                                      inst✝ : NormedAddCommGroup β
                                      T : Set α → β
                                      C : Real
                                      x✝ : MeasurableSpace α
                                      hT : MeasureTheory.DominatedFinMeasAdditive 0 T C
                                      s : Set α
                                      hs : MeasurableSet s
                                      ⊢ Eq (0 s) 0
                                    -/
  eq_zero_of_measure_zero hT hs (by simp only [Measure.coe_zero, Pi.zero_apply])
                                    /-
                                      🎉 no goals
                                    -/


theorem add (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C') :
    DominatedFinMeasAdditive μ (T + T') (C + C') := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T T' : Set α → β
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    ⊢ MeasureTheory.DominatedFinMeasAdditive μ (HAdd.hAdd T T') (HAdd.hAdd C C')
  -/
  refine ⟨hT.1.add hT'.1, fun s hs hμs => ?_⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T T' : Set α → β
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le (Norm.norm (HAdd.hAdd T T' s)) (HMul.hMul (HAdd.hAdd C C') (μ s).toReal)
  -/
  rw [Pi.add_apply, add_mul]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T T' : Set α → β
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le (Norm.norm (HAdd.hAdd (T s) (T' s))) (HAdd.hAdd (HMul.hMul C (μ s).toR …
  -/
  exact (norm_add_le _ _).trans (add_le_add (hT.2 s hs hμs) (hT'.2 s hs hμs))
  /-
    🎉 no goals
  -/


theorem smul [NormedField 𝕜] [NormedSpace 𝕜 β] (hT : DominatedFinMeasAdditive μ T C) (c : 𝕜) :
    DominatedFinMeasAdditive μ (fun s => c • T s) (‖c‖ * C) := by
  /-
    α : Type u_1
    𝕜 : Type u_6
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝² : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 β
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : 𝕜
    ⊢ MeasureTheory.DominatedFinMeasAdditive μ (fun s => HSMul.hSMul c (T s)) (HMu …
  -/
  refine ⟨hT.1.smul c, fun s hs hμs => ?_⟩
  /-
    α : Type u_1
    𝕜 : Type u_6
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝² : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 β
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : 𝕜
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le (Norm.norm ((fun s => HSMul.hSMul c (T s)) s)) (HMul.hMul (HMul.hMul ( …
  -/
  dsimp only
  /-
    α : Type u_1
    𝕜 : Type u_6
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝² : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 β
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : 𝕜
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le (Norm.norm (HSMul.hSMul c (T s))) (HMul.hMul (HMul.hMul (Norm.norm c)  …
  -/
  rw [norm_smul, mul_assoc]
  /-
    α : Type u_1
    𝕜 : Type u_6
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝² : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 β
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : 𝕜
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm (T s))) (HMul.hMul (Norm.norm c) ( …
  -/
  exact mul_le_mul le_rfl (hT.2 s hs hμs) (norm_nonneg _) (norm_nonneg _)
  /-
    🎉 no goals
  -/


theorem of_measure_le {μ' : Measure α} (h : μ ≤ μ') (hT : DominatedFinMeasAdditive μ T C)
    (hC : 0 ≤ C) : DominatedFinMeasAdditive μ' T C := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    μ' : MeasureTheory.Measure α
    h : LE.le μ μ'
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hC : LE.le 0 C
    ⊢ MeasureTheory.DominatedFinMeasAdditive μ' T C
  -/
  have h' : ∀ s, μ s = ∞ → μ' s = ∞ := fun s hs ↦ top_unique <| hs.symm.trans_le (h _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    μ' : MeasureTheory.Measure α
    h : LE.le μ μ'
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hC : LE.le 0 C
    h' : ∀ (s : Set α), Eq (μ s) Top.top → Eq (μ' s) Top.top
    ⊢ MeasureTheory.DominatedFinMeasAdditive μ' T C
  -/
  refine ⟨hT.1.of_eq_top_imp_eq_top fun s _ ↦ h' s, fun s hs hμ's ↦ ?_⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    μ' : MeasureTheory.Measure α
    h : LE.le μ μ'
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hC : LE.le 0 C
    h' : ∀ (s : Set α), Eq (μ s) Top.top → Eq (μ' s) Top.top
    s : Set α
    hs : MeasurableSet s
    hμ's : LT.lt (μ' s) Top.top
    ⊢ LE.le (Norm.norm (T s)) (HMul.hMul C (μ' s).toReal)
  -/
  have hμs : μ s < ∞ := (h s).trans_lt hμ's
  calc
    ‖T s‖ ≤ C * (μ s).toReal := hT.2 s hs hμs
    _ ≤ C * (μ' s).toReal := by gcongr; exacts [hμ's.ne, h _]


theorem add_measure_right {_ : MeasurableSpace α} (μ ν : Measure α)
    (hT : DominatedFinMeasAdditive μ T C) (hC : 0 ≤ C) : DominatedFinMeasAdditive (μ + ν) T C :=
  of_measure_le (Measure.le_add_right le_rfl) hT hC


theorem add_measure_left {_ : MeasurableSpace α} (μ ν : Measure α)
    (hT : DominatedFinMeasAdditive ν T C) (hC : 0 ≤ C) : DominatedFinMeasAdditive (μ + ν) T C :=
  of_measure_le (Measure.le_add_left le_rfl) hT hC


theorem of_smul_measure (c : ℝ≥0∞) (hc_ne_top : c ≠ ∞) (hT : DominatedFinMeasAdditive (c • μ) T C) :
    DominatedFinMeasAdditive μ T (c.toReal * C) := by
  have h : ∀ s, MeasurableSet s → c • μ s = ∞ → μ s = ∞ := by
    intro s _ hcμs
    simp only [hc_ne_top, Algebra.id.smul_eq_mul, ENNReal.mul_eq_top, or_false, Ne,
      false_and] at hcμs
    exact hcμs.2
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C
    h : ∀ (s : Set α), MeasurableSet s → Eq (HSMul.hSMul c (μ s)) Top.top → Eq (μ  …
    ⊢ MeasureTheory.DominatedFinMeasAdditive μ T (HMul.hMul c.toReal C)
  -/
  refine ⟨hT.1.of_eq_top_imp_eq_top (μ := c • μ) h, fun s hs hμs => ?_⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C
    h : ∀ (s : Set α), MeasurableSet s → Eq (HSMul.hSMul c (μ s)) Top.top → Eq (μ  …
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le (Norm.norm (T s)) (HMul.hMul (HMul.hMul c.toReal C) (μ s).toReal)
  -/
  have hcμs : c • μ s ≠ ∞ := mt (h s hs) hμs.ne
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C
    h : ∀ (s : Set α), MeasurableSet s → Eq (HSMul.hSMul c (μ s)) Top.top → Eq (μ  …
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hcμs : Ne (HSMul.hSMul c (μ s)) Top.top
    ⊢ LE.le (Norm.norm (T s)) (HMul.hMul (HMul.hMul c.toReal C) (μ s).toReal)
  -/
  rw [smul_eq_mul] at hcμs
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C
    h : ∀ (s : Set α), MeasurableSet s → Eq (HSMul.hSMul c (μ s)) Top.top → Eq (μ  …
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hcμs : Ne (HMul.hMul c (μ s)) Top.top
    ⊢ LE.le (Norm.norm (T s)) (HMul.hMul (HMul.hMul c.toReal C) (μ s).toReal)
  -/
  simp_rw [DominatedFinMeasAdditive, Measure.smul_apply, smul_eq_mul, toReal_mul] at hT
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    c : ENNReal
    hc_ne_top : Ne c Top.top
    h : ∀ (s : Set α), MeasurableSet s → Eq (HSMul.hSMul c (μ s)) Top.top → Eq (μ  …
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hcμs : Ne (HMul.hMul c (μ s)) Top.top
    hT : And (MeasureTheory.FinMeasAdditive (HSMul.hSMul c μ) T) (∀ (s : Set α), M …
    ⊢ LE.le (Norm.norm (T s)) (HMul.hMul (HMul.hMul c.toReal C) (μ s).toReal)
  -/
  refine (hT.2 s hs hcμs.lt_top).trans (le_of_eq ?_)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_7
    inst✝ : SeminormedAddCommGroup β
    T : Set α → β
    C : Real
    c : ENNReal
    hc_ne_top : Ne c Top.top
    h : ∀ (s : Set α), MeasurableSet s → Eq (HSMul.hSMul c (μ s)) Top.top → Eq (μ  …
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    hcμs : Ne (HMul.hMul c (μ s)) Top.top
    hT : And (MeasureTheory.FinMeasAdditive (HSMul.hSMul c μ) T) (∀ (s : Set α), M …
    ⊢ Eq (HMul.hMul C (HMul.hMul c.toReal (μ s).toReal)) (HMul.hMul (HMul.hMul c.t …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem of_measure_le_smul {μ' : Measure α} (c : ℝ≥0∞) (hc : c ≠ ∞) (h : μ ≤ c • μ')
    (hT : DominatedFinMeasAdditive μ T C) (hC : 0 ≤ C) :
    DominatedFinMeasAdditive μ' T (c.toReal * C) :=
  (hT.of_measure_le h hC).of_smul_measure c hc


/-- Extend `Set α → (F →L[ℝ] F')` to `(α →ₛ F) → F'`. -/
def setToSimpleFunc {_ : MeasurableSpace α} (T : Set α → F →L[ℝ] F') (f : α →ₛ F) : F' :=
  ∑ x ∈ f.range, T (f ⁻¹' {x}) x


@[simp]
theorem setToSimpleFunc_zero {m : MeasurableSpace α} (f : α →ₛ F) :
                                                         /-
                                                           α : Type u_1
                                                           F : Type u_3
                                                           F' : Type u_4
                                                           inst✝³ : NormedAddCommGroup F
                                                           inst✝² : NormedSpace Real F
                                                           inst✝¹ : NormedAddCommGroup F'
                                                           inst✝ : NormedSpace Real F'
                                                           m : MeasurableSpace α
                                                           f : MeasureTheory.SimpleFunc α F
                                                           ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc 0 f) 0
                                                         -/
    setToSimpleFunc (0 : Set α → F →L[ℝ] F') f = 0 := by simp [setToSimpleFunc]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem setToSimpleFunc_zero' {T : Set α → E →L[ℝ] F'}
    (h_zero : ∀ s, MeasurableSet s → μ s < ∞ → T s = 0) (f : α →ₛ E) (hf : Integrable f μ) :
    setToSimpleFunc T f = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F' : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
    h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T f) 0
  -/
  simp_rw [setToSimpleFunc]
  /-
    α : Type u_1
    E : Type u_2
    F' : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
    h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (f.range.sum fun x => (T (Set.preimage (⇑f) (Singleton.singleton x))) x) 0
  -/
  refine sum_eq_zero fun x _ => ?_
  /-
    α : Type u_1
    E : Type u_2
    F' : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
    h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    x : E
    x✝ : Membership.mem f.range x
    ⊢ Eq ((T (Set.preimage (⇑f) (Singleton.singleton x))) x) 0
  -/
  by_cases hx0 : x = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F' : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F'
      inst✝ : NormedSpace Real F'
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
      h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      x : E
      x✝ : Membership.mem f.range x
      hx0 : Eq x 0
      ⊢ Eq ((T (Set.preimage (⇑f) (Singleton.singleton x))) x) 0
    -/
  · simp [hx0]
    /-
      🎉 no goals
    -/
  rw [h_zero (f ⁻¹' ({x} : Set E)) (measurableSet_fiber _ _)
      (measure_preimage_lt_top_of_integrable f hf hx0),
    ContinuousLinearMap.zero_apply]


@[simp]
theorem setToSimpleFunc_zero_apply {m : MeasurableSpace α} (T : Set α → F →L[ℝ] F') :
    setToSimpleFunc T (0 : α →ₛ F) = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T 0) 0
  -/
                                  /-
                                    🎉 no goals
                                  -/
  cases isEmpty_or_nonempty α <;> simp [setToSimpleFunc]
                                  /-
                                    🎉 no goals
                                  -/


theorem setToSimpleFunc_eq_sum_filter [DecidablePred fun x ↦ x ≠ (0 : F)]
    {m : MeasurableSpace α} (T : Set α → F →L[ℝ] F') (f : α →ₛ F) :
    setToSimpleFunc T f = ∑ x ∈ f.range with x ≠ 0, T (f ⁻¹' {x}) x := by
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : DecidablePred fun x => Ne x 0
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    f : MeasureTheory.SimpleFunc α F
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T f) ((Finset.filter (fun x =>  …
  -/
  symm
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : DecidablePred fun x => Ne x 0
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    f : MeasureTheory.SimpleFunc α F
    ⊢ Eq ((Finset.filter (fun x => Ne x 0) f.range).sum fun x => (T (Set.preimage  …
  -/
  refine sum_filter_of_ne fun x _ => mt fun hx0 => ?_
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : DecidablePred fun x => Ne x 0
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    f : MeasureTheory.SimpleFunc α F
    x : F
    x✝ : Membership.mem f.range x
    hx0 : Eq x 0
    ⊢ Eq ((T (Set.preimage (⇑f) (Singleton.singleton x))) x) 0
  -/
  rw [hx0]
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedAddCommGroup F'
    inst✝¹ : NormedSpace Real F'
    inst✝ : DecidablePred fun x => Ne x 0
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    f : MeasureTheory.SimpleFunc α F
    x : F
    x✝ : Membership.mem f.range x
    hx0 : Eq x 0
    ⊢ Eq ((T (Set.preimage (⇑f) (Singleton.singleton 0))) 0) 0
  -/
  exact ContinuousLinearMap.map_zero _
  /-
    🎉 no goals
  -/


theorem map_setToSimpleFunc (T : Set α → F →L[ℝ] F') (h_add : FinMeasAdditive μ T) {f : α →ₛ G}
    (hf : Integrable f μ) {g : G → F} (hg : g 0 = 0) :
    (f.map g).setToSimpleFunc T = ∑ x ∈ f.range, T (f ⁻¹' {x}) (g x) := by
  classical
  have T_empty : T ∅ = 0 := h_add.map_empty_eq_zero
  have hfp : ∀ x ∈ f.range, x ≠ 0 → μ (f ⁻¹' {x}) ≠ ∞ := fun x _ hx0 =>
    (measure_preimage_lt_top_of_integrable f hf hx0).ne
  simp only [setToSimpleFunc, range_map]
  refine Finset.sum_image' _ fun b hb => ?_
  rcases mem_range.1 hb with ⟨a, rfl⟩
  by_cases h0 : g (f a) = 0
  · simp_rw [h0]
    rw [ContinuousLinearMap.map_zero, Finset.sum_eq_zero fun x hx => ?_]
    rw [mem_filter] at hx
    rw [hx.2, ContinuousLinearMap.map_zero]
  have h_left_eq :
    T (map g f ⁻¹' {g (f a)}) (g (f a))
      = T (f ⁻¹' ({b ∈ f.range | g b = g (f a)} : Finset _)) (g (f a)) := by
    congr; rw [map_preimage_singleton]
  rw [h_left_eq]
  have h_left_eq' :
    T (f ⁻¹' ({b ∈ f.range | g b = g (f a)} : Finset _)) (g (f a))
      = T (⋃ y ∈ {b ∈ f.range | g b = g (f a)}, f ⁻¹' {y}) (g (f a)) := by
    congr; rw [← Finset.set_biUnion_preimage_singleton]
  rw [h_left_eq']
  rw [h_add.map_iUnion_fin_meas_set_eq_sum T T_empty]
  · simp only [sum_apply, ContinuousLinearMap.coe_sum']
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [mem_filter] at hx
    rw [hx.2]
  · exact fun i => measurableSet_fiber _ _
  · intro i hi
    rw [mem_filter] at hi
    refine hfp i hi.1 fun hi0 => ?_
    rw [hi0, hg] at hi
    exact h0 hi.2.symm
  · intro i _j hi _ hij
    rw [Set.disjoint_iff]
    intro x hx
    rw [Set.mem_inter_iff, Set.mem_preimage, Set.mem_preimage, Set.mem_singleton_iff,
      Set.mem_singleton_iff] at hx
    rw [← hx.1, ← hx.2] at hij
    exact absurd rfl hij


theorem setToSimpleFunc_congr' (T : Set α → E →L[ℝ] F) (h_add : FinMeasAdditive μ T) {f g : α →ₛ E}
    (hf : Integrable f μ) (hg : Integrable g μ)
    (h : Pairwise fun x y => T (f ⁻¹' {x} ∩ g ⁻¹' {y}) = 0) :
    f.setToSimpleFunc T = g.setToSimpleFunc T :=
  show ((pair f g).map Prod.fst).setToSimpleFunc T = ((pair f g).map Prod.snd).setToSimpleFunc T by
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      h_add : MeasureTheory.FinMeasAdditive μ T
      f g : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      hg : MeasureTheory.Integrable (⇑g) μ
      h : Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singl …
      ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.SimpleFunc.map …
    -/
    have h_pair : Integrable (f.pair g) μ := integrable_pair hf hg
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      h_add : MeasureTheory.FinMeasAdditive μ T
      f g : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      hg : MeasureTheory.Integrable (⇑g) μ
      h : Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singl …
      h_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
      ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.SimpleFunc.map …
    -/
    rw [map_setToSimpleFunc T h_add h_pair Prod.fst_zero]
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      h_add : MeasureTheory.FinMeasAdditive μ T
      f g : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      hg : MeasureTheory.Integrable (⇑g) μ
      h : Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singl …
      h_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
      ⊢ Eq ((f.pair g).range.sum fun x => (T (Set.preimage (⇑(f.pair g)) (Singleton. …
    -/
    rw [map_setToSimpleFunc T h_add h_pair Prod.snd_zero]
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      h_add : MeasureTheory.FinMeasAdditive μ T
      f g : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      hg : MeasureTheory.Integrable (⇑g) μ
      h : Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singl …
      h_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
      ⊢ Eq ((f.pair g).range.sum fun x => (T (Set.preimage (⇑(f.pair g)) (Singleton. …
    -/
    refine Finset.sum_congr rfl fun p hp => ?_
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      h_add : MeasureTheory.FinMeasAdditive μ T
      f g : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      hg : MeasureTheory.Integrable (⇑g) μ
      h : Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singl …
      h_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
      p : Prod E E
      hp : Membership.mem (f.pair g).range p
      ⊢ Eq ((T (Set.preimage (⇑(f.pair g)) (Singleton.singleton p))) p.1) ((T (Set.p …
    -/
    rcases mem_range.1 hp with ⟨a, rfl⟩
    /-
      case intro
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      h_add : MeasureTheory.FinMeasAdditive μ T
      f g : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      hg : MeasureTheory.Integrable (⇑g) μ
      h : Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singl …
      h_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
      a : α
      hp : Membership.mem (f.pair g).range ((f.pair g) a)
      ⊢ Eq ((T (Set.preimage (⇑(f.pair g)) (Singleton.singleton ((f.pair g) a)))) (( …
    -/
    by_cases eq : f a = g a
      /-
        case pos
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        h_add : MeasureTheory.FinMeasAdditive μ T
        f g : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        hg : MeasureTheory.Integrable (⇑g) μ
        h : Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singl …
        h_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
        a : α
        hp : Membership.mem (f.pair g).range ((f.pair g) a)
        eq : Eq (f a) (g a)
        ⊢ Eq ((T (Set.preimage (⇑(f.pair g)) (Singleton.singleton ((f.pair g) a)))) (( …
      -/
    · dsimp only [pair_apply]; rw [eq]
                               /-
                                 🎉 no goals
                               -/
    · have : T (pair f g ⁻¹' {(f a, g a)}) = 0 := by
        have h_eq : T ((⇑(f.pair g)) ⁻¹' {(f a, g a)}) = T (f ⁻¹' {f a} ∩ g ⁻¹' {g a}) := by
          congr; rw [pair_preimage_singleton f g]
        rw [h_eq]
        exact h eq
      /-
        case neg
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        h_add : MeasureTheory.FinMeasAdditive μ T
        f g : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        hg : MeasureTheory.Integrable (⇑g) μ
        h : Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singl …
        h_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
        a : α
        hp : Membership.mem (f.pair g).range ((f.pair g) a)
        eq : Not (Eq (f a) (g a))
        this : Eq (T (Set.preimage (⇑(f.pair g)) (Singleton.singleton { fst := f a, sn …
        ⊢ Eq ((T (Set.preimage (⇑(f.pair g)) (Singleton.singleton ((f.pair g) a)))) (( …
      -/
      simp only [this, ContinuousLinearMap.zero_apply, pair_apply]
      /-
        🎉 no goals
      -/


theorem setToSimpleFunc_congr (T : Set α → E →L[ℝ] F)
    (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0) (h_add : FinMeasAdditive μ T) {f g : α →ₛ E}
    (hf : Integrable f μ) (h : f =ᵐ[μ] g) : f.setToSimpleFunc T = g.setToSimpleFunc T := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    h : (MeasureTheory.ae μ).EventuallyEq ⇑f ⇑g
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T f) (MeasureTheory.SimpleFunc. …
  -/
  refine setToSimpleFunc_congr' T h_add hf ((integrable_congr h).mp hf) ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    h : (MeasureTheory.ae μ).EventuallyEq ⇑f ⇑g
    ⊢ Pairwise fun x y => Eq (T (Inter.inter (Set.preimage (⇑f) (Singleton.singlet …
  -/
  refine fun x y hxy => h_zero _ ((measurableSet_fiber f x).inter (measurableSet_fiber g y)) ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    h : (MeasureTheory.ae μ).EventuallyEq ⇑f ⇑g
    x y : E
    hxy : Ne x y
    ⊢ Eq (μ (Inter.inter (Set.preimage (⇑f) (Singleton.singleton x)) (Set.preimage …
  -/
  rw [EventuallyEq, ae_iff] at h
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    h : Eq (μ (setOf fun a => Not (Eq (f a) (g a)))) 0
    x y : E
    hxy : Ne x y
    ⊢ Eq (μ (Inter.inter (Set.preimage (⇑f) (Singleton.singleton x)) (Set.preimage …
  -/
  refine measure_mono_null (fun z => ?_) h
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    h : Eq (μ (setOf fun a => Not (Eq (f a) (g a)))) 0
    x y : E
    hxy : Ne x y
    z : α
    ⊢ Membership.mem (Inter.inter (Set.preimage (⇑f) (Singleton.singleton x)) (Set …
  -/
  simp_rw [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_singleton_iff]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    h : Eq (μ (setOf fun a => Not (Eq (f a) (g a)))) 0
    x y : E
    hxy : Ne x y
    z : α
    ⊢ And (Eq (f z) x) (Eq (g z) y) → Not (Eq (f z) (g z))
  -/
  intro h
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    h✝ : Eq (μ (setOf fun a => Not (Eq (f a) (g a)))) 0
    x y : E
    hxy : Ne x y
    z : α
    h : And (Eq (f z) x) (Eq (g z) y)
    ⊢ Not (Eq (f z) (g z))
  -/
  rwa [h.1, h.2]
  /-
    🎉 no goals
  -/


theorem setToSimpleFunc_congr_left (T T' : Set α → E →L[ℝ] F)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → T s = T' s) (f : α →ₛ E) (hf : Integrable f μ) :
    setToSimpleFunc T f = setToSimpleFunc T' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T f) (MeasureTheory.SimpleFunc. …
  -/
  simp_rw [setToSimpleFunc]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    ⊢ Eq (f.range.sum fun x => (T (Set.preimage (⇑f) (Singleton.singleton x))) x)  …
  -/
  refine sum_congr rfl fun x _ => ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    x : E
    x✝ : Membership.mem f.range x
    ⊢ Eq ((T (Set.preimage (⇑f) (Singleton.singleton x))) x) ((T' (Set.preimage (⇑ …
  -/
  by_cases hx0 : x = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      x : E
      x✝ : Membership.mem f.range x
      hx0 : Eq x 0
      ⊢ Eq ((T (Set.preimage (⇑f) (Singleton.singleton x))) x) ((T' (Set.preimage (⇑ …
    -/
  · simp [hx0]
    /-
      🎉 no goals
    -/
  · rw [h (f ⁻¹' {x}) (SimpleFunc.measurableSet_fiber _ _)
        (SimpleFunc.measure_preimage_lt_top_of_integrable _ hf hx0)]


theorem setToSimpleFunc_add_left {m : MeasurableSpace α} (T T' : Set α → F →L[ℝ] F') {f : α →ₛ F} :
    setToSimpleFunc (T + T') f = setToSimpleFunc T f + setToSimpleFunc T' f := by
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    m : MeasurableSpace α
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    f : MeasureTheory.SimpleFunc α F
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc (HAdd.hAdd T T') f) (HAdd.hAdd  …
  -/
  simp_rw [setToSimpleFunc, Pi.add_apply]
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    m : MeasurableSpace α
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    f : MeasureTheory.SimpleFunc α F
    ⊢ Eq (f.range.sum fun x => (HAdd.hAdd (T (Set.preimage (⇑f) (Singleton.singlet …
  -/
  push_cast
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    m : MeasurableSpace α
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    f : MeasureTheory.SimpleFunc α F
    ⊢ Eq (f.range.sum fun x => HAdd.hAdd (⇑(T (Set.preimage (⇑f) (Singleton.single …
  -/
  simp_rw [Pi.add_apply, sum_add_distrib]
  /-
    🎉 no goals
  -/


theorem setToSimpleFunc_add_left' (T T' T'' : Set α → E →L[ℝ] F)
    (h_add : ∀ s, MeasurableSet s → μ s < ∞ → T'' s = T s + T' s) {f : α →ₛ E}
    (hf : Integrable f μ) : setToSimpleFunc T'' f = setToSimpleFunc T f + setToSimpleFunc T' f := by
  classical
  simp_rw [setToSimpleFunc_eq_sum_filter]
  suffices
    ∀ x ∈ {x ∈ f.range | x ≠ 0}, T'' (f ⁻¹' {x}) = T (f ⁻¹' {x}) + T' (f ⁻¹' {x}) by
    rw [← sum_add_distrib]
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [this x hx]
    push_cast
    rw [Pi.add_apply]
  intro x hx
  refine
    h_add (f ⁻¹' {x}) (measurableSet_preimage _ _) (measure_preimage_lt_top_of_integrable _ hf ?_)
  rw [mem_filter] at hx
  exact hx.2


theorem setToSimpleFunc_smul_left {m : MeasurableSpace α} (T : Set α → F →L[ℝ] F') (c : ℝ)
    (f : α →ₛ F) : setToSimpleFunc (fun s => c • T s) f = c • setToSimpleFunc T f := by
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    c : Real
    f : MeasureTheory.SimpleFunc α F
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc (fun s => HSMul.hSMul c (T s))  …
  -/
  simp_rw [setToSimpleFunc, ContinuousLinearMap.smul_apply, smul_sum]
  /-
    🎉 no goals
  -/


theorem setToSimpleFunc_smul_left' (T T' : Set α → E →L[ℝ] F') (c : ℝ)
    (h_smul : ∀ s, MeasurableSet s → μ s < ∞ → T' s = c • T s) {f : α →ₛ E} (hf : Integrable f μ) :
    setToSimpleFunc T' f = c • setToSimpleFunc T f := by
  classical
  simp_rw [setToSimpleFunc_eq_sum_filter]
  suffices ∀ x ∈ {x ∈ f.range | x ≠ 0}, T' (f ⁻¹' {x}) = c • T (f ⁻¹' {x}) by
    rw [smul_sum]
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [this x hx]
    rfl
  intro x hx
  refine
    h_smul (f ⁻¹' {x}) (measurableSet_preimage _ _) (measure_preimage_lt_top_of_integrable _ hf ?_)
  rw [mem_filter] at hx
  exact hx.2


theorem setToSimpleFunc_add (T : Set α → E →L[ℝ] F) (h_add : FinMeasAdditive μ T) {f g : α →ₛ E}
    (hf : Integrable f μ) (hg : Integrable g μ) :
    setToSimpleFunc T (f + g) = setToSimpleFunc T f + setToSimpleFunc T g :=
  have hp_pair : Integrable (f.pair g) μ := integrable_pair hf hg
  calc
    setToSimpleFunc T (f + g) = ∑ x ∈ (pair f g).range, T (pair f g ⁻¹' {x}) (x.fst + x.snd) := by
      /-
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        h_add : MeasureTheory.FinMeasAdditive μ T
        f g : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        hg : MeasureTheory.Integrable (⇑g) μ
        hp_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
        ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (HAdd.hAdd f g)) ((f.pair g). …
      -/
      rw [add_eq_map₂, map_setToSimpleFunc T h_add hp_pair]; simp
                                                             /-
                                                               🎉 no goals
                                                             -/
    _ = ∑ x ∈ (pair f g).range, (T (pair f g ⁻¹' {x}) x.fst + T (pair f g ⁻¹' {x}) x.snd) :=
      (Finset.sum_congr rfl fun _ _ => ContinuousLinearMap.map_add _ _ _)
    _ = (∑ x ∈ (pair f g).range, T (pair f g ⁻¹' {x}) x.fst) +
          ∑ x ∈ (pair f g).range, T (pair f g ⁻¹' {x}) x.snd := by
      /-
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        h_add : MeasureTheory.FinMeasAdditive μ T
        f g : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        hg : MeasureTheory.Integrable (⇑g) μ
        hp_pair : MeasureTheory.Integrable (⇑(f.pair g)) μ
        ⊢ Eq ((f.pair g).range.sum fun x => HAdd.hAdd ((T (Set.preimage (⇑(f.pair g))  …
      -/
      rw [Finset.sum_add_distrib]
      /-
        🎉 no goals
      -/
    _ = ((pair f g).map Prod.fst).setToSimpleFunc T +
          ((pair f g).map Prod.snd).setToSimpleFunc T := by
      rw [map_setToSimpleFunc T h_add hp_pair Prod.snd_zero,
        map_setToSimpleFunc T h_add hp_pair Prod.fst_zero]


theorem setToSimpleFunc_neg (T : Set α → E →L[ℝ] F) (h_add : FinMeasAdditive μ T) {f : α →ₛ E}
    (hf : Integrable f μ) : setToSimpleFunc T (-f) = -setToSimpleFunc T f :=
  calc
    setToSimpleFunc T (-f) = setToSimpleFunc T (f.map Neg.neg) := rfl
    _ = -setToSimpleFunc T f := by
      /-
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        h_add : MeasureTheory.FinMeasAdditive μ T
        f : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.SimpleFunc.map …
      -/
      rw [map_setToSimpleFunc T h_add hf neg_zero, setToSimpleFunc, ← sum_neg_distrib]
      /-
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        h_add : MeasureTheory.FinMeasAdditive μ T
        f : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        ⊢ Eq (f.range.sum fun x => (T (Set.preimage (⇑f) (Singleton.singleton x))) (Ne …
      -/
      exact Finset.sum_congr rfl fun x _ => ContinuousLinearMap.map_neg _ _
      /-
        🎉 no goals
      -/


theorem setToSimpleFunc_sub (T : Set α → E →L[ℝ] F) (h_add : FinMeasAdditive μ T) {f g : α →ₛ E}
    (hf : Integrable f μ) (hg : Integrable g μ) :
    setToSimpleFunc T (f - g) = setToSimpleFunc T f - setToSimpleFunc T g := by
  rw [sub_eq_add_neg, setToSimpleFunc_add T h_add hf, setToSimpleFunc_neg T h_add hg,
    sub_eq_add_neg]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    hg : MeasureTheory.Integrable (⇑g) μ
    ⊢ MeasureTheory.Integrable (⇑(Neg.neg g)) μ
  -/
  rw [integrable_iff] at hg ⊢
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    hg : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑g) (Singleton.singleton y))) …
    ⊢ ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑(Neg.neg g)) (Singleton.singlet …
  -/
  intro x hx_ne
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    hg : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑g) (Singleton.singleton y))) …
    x : E
    hx_ne : Ne x 0
    ⊢ LT.lt (μ (Set.preimage (⇑(Neg.neg g)) (Singleton.singleton x))) Top.top
  -/
  change μ (Neg.neg ∘ g ⁻¹' {x}) < ∞
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    hg : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑g) (Singleton.singleton y))) …
    x : E
    hx_ne : Ne x 0
    ⊢ LT.lt (μ (Set.preimage (Function.comp Neg.neg ⇑g) (Singleton.singleton x)))  …
  -/
  rw [preimage_comp, neg_preimage, Set.neg_singleton]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    hg : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑g) (Singleton.singleton y))) …
    x : E
    hx_ne : Ne x 0
    ⊢ LT.lt (μ (Set.preimage (⇑g) (Singleton.singleton (Neg.neg x)))) Top.top
  -/
  refine hg (-x) ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    hg : ∀ (y : E), Ne y 0 → LT.lt (μ (Set.preimage (⇑g) (Singleton.singleton y))) …
    x : E
    hx_ne : Ne x 0
    ⊢ Ne (Neg.neg x) 0
  -/
  simp [hx_ne]
  /-
    🎉 no goals
  -/


theorem setToSimpleFunc_smul_real (T : Set α → E →L[ℝ] F) (h_add : FinMeasAdditive μ T) (c : ℝ)
    {f : α →ₛ E} (hf : Integrable f μ) : setToSimpleFunc T (c • f) = c • setToSimpleFunc T f :=
  calc
    setToSimpleFunc T (c • f) = ∑ x ∈ f.range, T (f ⁻¹' {x}) (c • x) := by
      /-
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        h_add : MeasureTheory.FinMeasAdditive μ T
        c : Real
        f : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (HSMul.hSMul c f)) (f.range.s …
      -/
      rw [smul_eq_map c f, map_setToSimpleFunc T h_add hf]; dsimp only; rw [smul_zero]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    _ = ∑ x ∈ f.range, c • T (f ⁻¹' {x}) x :=
                                          /-
                                            α : Type u_1
                                            E : Type u_2
                                            F : Type u_3
                                            inst✝³ : NormedAddCommGroup E
                                            inst✝² : NormedSpace Real E
                                            inst✝¹ : NormedAddCommGroup F
                                            inst✝ : NormedSpace Real F
                                            m : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            T : Set α → ContinuousLinearMap (RingHom.id Real) E F
                                            h_add : MeasureTheory.FinMeasAdditive μ T
                                            c : Real
                                            f : MeasureTheory.SimpleFunc α E
                                            hf : MeasureTheory.Integrable (⇑f) μ
                                            b : E
                                            x✝ : Membership.mem f.range b
                                            ⊢ Eq ((T (Set.preimage (⇑f) (Singleton.singleton b))) (HSMul.hSMul c b)) (HSMu …
                                          -/
      (Finset.sum_congr rfl fun b _ => by rw [ContinuousLinearMap.map_smul (T (f ⁻¹' {b})) c b])
                                          /-
                                            🎉 no goals
                                          -/
                                      /-
                                        α : Type u_1
                                        E : Type u_2
                                        F : Type u_3
                                        inst✝³ : NormedAddCommGroup E
                                        inst✝² : NormedSpace Real E
                                        inst✝¹ : NormedAddCommGroup F
                                        inst✝ : NormedSpace Real F
                                        m : MeasurableSpace α
                                        μ : MeasureTheory.Measure α
                                        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
                                        h_add : MeasureTheory.FinMeasAdditive μ T
                                        c : Real
                                        f : MeasureTheory.SimpleFunc α E
                                        hf : MeasureTheory.Integrable (⇑f) μ
                                        ⊢ Eq (f.range.sum fun x => HSMul.hSMul c ((T (Set.preimage (⇑f) (Singleton.sin …
                                      -/
    _ = c • setToSimpleFunc T f := by simp only [setToSimpleFunc, smul_sum, smul_smul, mul_comm]
                                      /-
                                        🎉 no goals
                                      -/


theorem setToSimpleFunc_smul {E} [NormedAddCommGroup E] [NormedField 𝕜] [NormedSpace 𝕜 E]
    [NormedSpace ℝ E] [NormedSpace 𝕜 F] (T : Set α → E →L[ℝ] F) (h_add : FinMeasAdditive μ T)
    (h_smul : ∀ c : 𝕜, ∀ s x, T s (c • x) = c • T s x) (c : 𝕜) {f : α →ₛ E} (hf : Integrable f μ) :
    setToSimpleFunc T (c • f) = c • setToSimpleFunc T f :=
  calc
    setToSimpleFunc T (c • f) = ∑ x ∈ f.range, T (f ⁻¹' {x}) (c • x) := by
      /-
        α : Type u_1
        F : Type u_3
        𝕜 : Type u_6
        inst✝⁶ : NormedAddCommGroup F
        inst✝⁵ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_7
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedField 𝕜
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedSpace Real E
        inst✝ : NormedSpace 𝕜 F
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        h_add : MeasureTheory.FinMeasAdditive μ T
        h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
        c : 𝕜
        f : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (HSMul.hSMul c f)) (f.range.s …
      -/
      rw [smul_eq_map c f, map_setToSimpleFunc T h_add hf]; dsimp only; rw [smul_zero]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                                                                 /-
                                                                                   α : Type u_1
                                                                                   F : Type u_3
                                                                                   𝕜 : Type u_6
                                                                                   inst✝⁶ : NormedAddCommGroup F
                                                                                   inst✝⁵ : NormedSpace Real F
                                                                                   m : MeasurableSpace α
                                                                                   μ : MeasureTheory.Measure α
                                                                                   E : Type u_7
                                                                                   inst✝⁴ : NormedAddCommGroup E
                                                                                   inst✝³ : NormedField 𝕜
                                                                                   inst✝² : NormedSpace 𝕜 E
                                                                                   inst✝¹ : NormedSpace Real E
                                                                                   inst✝ : NormedSpace 𝕜 F
                                                                                   T : Set α → ContinuousLinearMap (RingHom.id Real) E F
                                                                                   h_add : MeasureTheory.FinMeasAdditive μ T
                                                                                   h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
                                                                                   c : 𝕜
                                                                                   f : MeasureTheory.SimpleFunc α E
                                                                                   hf : MeasureTheory.Integrable (⇑f) μ
                                                                                   b : E
                                                                                   x✝ : Membership.mem f.range b
                                                                                   ⊢ Eq ((T (Set.preimage (⇑f) (Singleton.singleton b))) (HSMul.hSMul c b)) (HSMu …
                                                                                 -/
    _ = ∑ x ∈ f.range, c • T (f ⁻¹' {x}) x := Finset.sum_congr rfl fun b _ => by rw [h_smul]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                                      /-
                                        α : Type u_1
                                        F : Type u_3
                                        𝕜 : Type u_6
                                        inst✝⁶ : NormedAddCommGroup F
                                        inst✝⁵ : NormedSpace Real F
                                        m : MeasurableSpace α
                                        μ : MeasureTheory.Measure α
                                        E : Type u_7
                                        inst✝⁴ : NormedAddCommGroup E
                                        inst✝³ : NormedField 𝕜
                                        inst✝² : NormedSpace 𝕜 E
                                        inst✝¹ : NormedSpace Real E
                                        inst✝ : NormedSpace 𝕜 F
                                        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
                                        h_add : MeasureTheory.FinMeasAdditive μ T
                                        h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
                                        c : 𝕜
                                        f : MeasureTheory.SimpleFunc α E
                                        hf : MeasureTheory.Integrable (⇑f) μ
                                        ⊢ Eq (f.range.sum fun x => HSMul.hSMul c ((T (Set.preimage (⇑f) (Singleton.sin …
                                      -/
    _ = c • setToSimpleFunc T f := by simp only [setToSimpleFunc, smul_sum, smul_smul, mul_comm]
                                      /-
                                        🎉 no goals
                                      -/


theorem setToSimpleFunc_mono_left {m : MeasurableSpace α} (T T' : Set α → F →L[ℝ] G'')
    (hTT' : ∀ s x, T s x ≤ T' s x) (f : α →ₛ F) : setToSimpleFunc T f ≤ setToSimpleFunc T' f := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    G'' : Type u_8
    inst✝¹ : NormedLatticeAddCommGroup G''
    inst✝ : NormedSpace Real G''
    m : MeasurableSpace α
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) F G''
    hTT' : ∀ (s : Set α) (x : F), LE.le ((T s) x) ((T' s) x)
    f : MeasureTheory.SimpleFunc α F
    ⊢ LE.le (MeasureTheory.SimpleFunc.setToSimpleFunc T f) (MeasureTheory.SimpleFu …
  -/
  simp_rw [setToSimpleFunc]; exact sum_le_sum fun i _ => hTT' _ i
                             /-
                               🎉 no goals
                             -/


theorem setToSimpleFunc_mono_left' (T T' : Set α → E →L[ℝ] G'')
    (hTT' : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, T s x ≤ T' s x) (f : α →ₛ E)
    (hf : Integrable f μ) : setToSimpleFunc T f ≤ setToSimpleFunc T' f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G'' : Type u_8
    inst✝¹ : NormedLatticeAddCommGroup G''
    inst✝ : NormedSpace Real G''
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E G''
    hTT' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : E), LE.le …
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    ⊢ LE.le (MeasureTheory.SimpleFunc.setToSimpleFunc T f) (MeasureTheory.SimpleFu …
  -/
  refine sum_le_sum fun i _ => ?_
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G'' : Type u_8
    inst✝¹ : NormedLatticeAddCommGroup G''
    inst✝ : NormedSpace Real G''
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E G''
    hTT' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : E), LE.le …
    f : MeasureTheory.SimpleFunc α E
    hf : MeasureTheory.Integrable (⇑f) μ
    i : E
    x✝ : Membership.mem f.range i
    ⊢ LE.le ((T (Set.preimage (⇑f) (Singleton.singleton i))) i) ((T' (Set.preimage …
  -/
  by_cases h0 : i = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G'' : Type u_8
      inst✝¹ : NormedLatticeAddCommGroup G''
      inst✝ : NormedSpace Real G''
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E G''
      hTT' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : E), LE.le …
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      i : E
      x✝ : Membership.mem f.range i
      h0 : Eq i 0
      ⊢ LE.le ((T (Set.preimage (⇑f) (Singleton.singleton i))) i) ((T' (Set.preimage …
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G'' : Type u_8
      inst✝¹ : NormedLatticeAddCommGroup G''
      inst✝ : NormedSpace Real G''
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E G''
      hTT' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : E), LE.le …
      f : MeasureTheory.SimpleFunc α E
      hf : MeasureTheory.Integrable (⇑f) μ
      i : E
      x✝ : Membership.mem f.range i
      h0 : Not (Eq i 0)
      ⊢ LE.le ((T (Set.preimage (⇑f) (Singleton.singleton i))) i) ((T' (Set.preimage …
    -/
  · exact hTT' _ (measurableSet_fiber _ _) (measure_preimage_lt_top_of_integrable _ hf h0) i
    /-
      🎉 no goals
    -/


theorem setToSimpleFunc_nonneg {m : MeasurableSpace α} (T : Set α → G' →L[ℝ] G'')
    (hT_nonneg : ∀ s x, 0 ≤ x → 0 ≤ T s x) (f : α →ₛ G') (hf : 0 ≤ f) :
    0 ≤ setToSimpleFunc T f := by
  /-
    α : Type u_1
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α) (x : G'), LE.le 0 x → LE.le 0 ((T s) x)
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    ⊢ LE.le 0 (MeasureTheory.SimpleFunc.setToSimpleFunc T f)
  -/
  refine sum_nonneg fun i hi => hT_nonneg _ i ?_
  /-
    α : Type u_1
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α) (x : G'), LE.le 0 x → LE.le 0 ((T s) x)
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    i : G'
    hi : Membership.mem f.range i
    ⊢ LE.le 0 i
  -/
  rw [mem_range] at hi
  /-
    α : Type u_1
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α) (x : G'), LE.le 0 x → LE.le 0 ((T s) x)
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    i : G'
    hi : Membership.mem (Set.range ⇑f) i
    ⊢ LE.le 0 i
  -/
  obtain ⟨y, hy⟩ := Set.mem_range.mp hi
  /-
    case intro
    α : Type u_1
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α) (x : G'), LE.le 0 x → LE.le 0 ((T s) x)
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    i : G'
    hi : Membership.mem (Set.range ⇑f) i
    y : α
    hy : Eq (f y) i
    ⊢ LE.le 0 i
  -/
  rw [← hy]
  /-
    case intro
    α : Type u_1
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α) (x : G'), LE.le 0 x → LE.le 0 ((T s) x)
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    i : G'
    hi : Membership.mem (Set.range ⇑f) i
    y : α
    hy : Eq (f y) i
    ⊢ LE.le 0 (f y)
  -/
  refine le_trans ?_ (hf y)
  /-
    case intro
    α : Type u_1
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    m : MeasurableSpace α
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α) (x : G'), LE.le 0 x → LE.le 0 ((T s) x)
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    i : G'
    hi : Membership.mem (Set.range ⇑f) i
    y : α
    hy : Eq (f y) i
    ⊢ LE.le 0 (0 y)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem setToSimpleFunc_nonneg' (T : Set α → G' →L[ℝ] G'')
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) (f : α →ₛ G') (hf : 0 ≤ f)
    (hfi : Integrable f μ) : 0 ≤ setToSimpleFunc T f := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    hfi : MeasureTheory.Integrable (⇑f) μ
    ⊢ LE.le 0 (MeasureTheory.SimpleFunc.setToSimpleFunc T f)
  -/
  refine sum_nonneg fun i hi => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    hfi : MeasureTheory.Integrable (⇑f) μ
    i : G'
    hi : Membership.mem f.range i
    ⊢ LE.le 0 ((T (Set.preimage (⇑f) (Singleton.singleton i))) i)
  -/
  by_cases h0 : i = 0
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝³ : NormedLatticeAddCommGroup G''
      inst✝² : NormedSpace Real G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : MeasureTheory.SimpleFunc α G'
      hf : LE.le 0 f
      hfi : MeasureTheory.Integrable (⇑f) μ
      i : G'
      hi : Membership.mem f.range i
      h0 : Eq i 0
      ⊢ LE.le 0 ((T (Set.preimage (⇑f) (Singleton.singleton i))) i)
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  refine
    hT_nonneg _ (measurableSet_fiber _ _) (measure_preimage_lt_top_of_integrable _ hfi h0) i ?_
  /-
    case neg
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    hfi : MeasureTheory.Integrable (⇑f) μ
    i : G'
    hi : Membership.mem f.range i
    h0 : Not (Eq i 0)
    ⊢ LE.le 0 i
  -/
  rw [mem_range] at hi
  /-
    case neg
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    hfi : MeasureTheory.Integrable (⇑f) μ
    i : G'
    hi : Membership.mem (Set.range ⇑f) i
    h0 : Not (Eq i 0)
    ⊢ LE.le 0 i
  -/
  obtain ⟨y, hy⟩ := Set.mem_range.mp hi
  /-
    case neg.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    hfi : MeasureTheory.Integrable (⇑f) μ
    i : G'
    hi : Membership.mem (Set.range ⇑f) i
    h0 : Not (Eq i 0)
    y : α
    hy : Eq (f y) i
    ⊢ LE.le 0 i
  -/
  rw [← hy]
  /-
    case neg.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f : MeasureTheory.SimpleFunc α G'
    hf : LE.le 0 f
    hfi : MeasureTheory.Integrable (⇑f) μ
    i : G'
    hi : Membership.mem (Set.range ⇑f) i
    h0 : Not (Eq i 0)
    y : α
    hy : Eq (f y) i
    ⊢ LE.le 0 (f y)
  -/
  convert hf y
  /-
    🎉 no goals
  -/


theorem setToSimpleFunc_mono {T : Set α → G' →L[ℝ] G''} (h_add : FinMeasAdditive μ T)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f g : α →ₛ G'}
    (hfi : Integrable f μ) (hgi : Integrable g μ) (hfg : f ≤ g) :
    setToSimpleFunc T f ≤ setToSimpleFunc T g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : MeasureTheory.SimpleFunc α G'
    hfi : MeasureTheory.Integrable (⇑f) μ
    hgi : MeasureTheory.Integrable (⇑g) μ
    hfg : LE.le f g
    ⊢ LE.le (MeasureTheory.SimpleFunc.setToSimpleFunc T f) (MeasureTheory.SimpleFu …
  -/
  rw [← sub_nonneg, ← setToSimpleFunc_sub T h_add hgi hfi]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : MeasureTheory.SimpleFunc α G'
    hfi : MeasureTheory.Integrable (⇑f) μ
    hgi : MeasureTheory.Integrable (⇑g) μ
    hfg : LE.le f g
    ⊢ LE.le 0 (MeasureTheory.SimpleFunc.setToSimpleFunc T (HSub.hSub g f))
  -/
  refine setToSimpleFunc_nonneg' T hT_nonneg _ ?_ (hgi.sub hfi)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : MeasureTheory.SimpleFunc α G'
    hfi : MeasureTheory.Integrable (⇑f) μ
    hgi : MeasureTheory.Integrable (⇑g) μ
    hfg : LE.le f g
    ⊢ LE.le 0 (HSub.hSub g f)
  -/
  intro x
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : MeasureTheory.SimpleFunc α G'
    hfi : MeasureTheory.Integrable (⇑f) μ
    hgi : MeasureTheory.Integrable (⇑g) μ
    hfg : LE.le f g
    x : α
    ⊢ LE.le (0 x) ((HSub.hSub g f) x)
  -/
  simp only [coe_sub, sub_nonneg, coe_zero, Pi.zero_apply, Pi.sub_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G''
    inst✝² : NormedSpace Real G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : MeasureTheory.SimpleFunc α G'
    hfi : MeasureTheory.Integrable (⇑f) μ
    hgi : MeasureTheory.Integrable (⇑g) μ
    hfg : LE.le f g
    x : α
    ⊢ LE.le (f x) (g x)
  -/
  exact hfg x
  /-
    🎉 no goals
  -/


theorem norm_setToSimpleFunc_le_sum_opNorm {m : MeasurableSpace α} (T : Set α → F' →L[ℝ] F)
    (f : α →ₛ F') : ‖f.setToSimpleFunc T‖ ≤ ∑ x ∈ f.range, ‖T (f ⁻¹' {x})‖ * ‖x‖ :=
  calc
    ‖∑ x ∈ f.range, T (f ⁻¹' {x}) x‖ ≤ ∑ x ∈ f.range, ‖T (f ⁻¹' {x}) x‖ := norm_sum_le _ _
    _ ≤ ∑ x ∈ f.range, ‖T (f ⁻¹' {x})‖ * ‖x‖ := by
      /-
        α : Type u_1
        F : Type u_3
        F' : Type u_4
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace Real F
        inst✝¹ : NormedAddCommGroup F'
        inst✝ : NormedSpace Real F'
        m : MeasurableSpace α
        T : Set α → ContinuousLinearMap (RingHom.id Real) F' F
        f : MeasureTheory.SimpleFunc α F'
        ⊢ LE.le (f.range.sum fun x => Norm.norm ((T (Set.preimage (⇑f) (Singleton.sing …
      -/
      refine Finset.sum_le_sum fun b _ => ?_; simp_rw [ContinuousLinearMap.le_opNorm]
                                              /-
                                                🎉 no goals
                                              -/


@[deprecated (since := "2024-02-02")]
alias norm_setToSimpleFunc_le_sum_op_norm := norm_setToSimpleFunc_le_sum_opNorm


theorem norm_setToSimpleFunc_le_sum_mul_norm (T : Set α → F →L[ℝ] F') {C : ℝ}
    (hT_norm : ∀ s, MeasurableSet s → ‖T s‖ ≤ C * (μ s).toReal) (f : α →ₛ F) :
    ‖f.setToSimpleFunc T‖ ≤ C * ∑ x ∈ f.range, (μ (f ⁻¹' {x})).toReal * ‖x‖ :=
  calc
    ‖f.setToSimpleFunc T‖ ≤ ∑ x ∈ f.range, ‖T (f ⁻¹' {x})‖ * ‖x‖ :=
      norm_setToSimpleFunc_le_sum_opNorm T f
    _ ≤ ∑ x ∈ f.range, C * (μ (f ⁻¹' {x})).toReal * ‖x‖ := by
      /-
        α : Type u_1
        F : Type u_3
        F' : Type u_4
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace Real F
        inst✝¹ : NormedAddCommGroup F'
        inst✝ : NormedSpace Real F'
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
        C : Real
        hT_norm : ∀ (s : Set α), MeasurableSet s → LE.le (Norm.norm (T s)) (HMul.hMul  …
        f : MeasureTheory.SimpleFunc α F
        ⊢ LE.le (f.range.sum fun x => HMul.hMul (Norm.norm (T (Set.preimage (⇑f) (Sing …
      -/
      gcongr
      /-
        case h.h
        α : Type u_1
        F : Type u_3
        F' : Type u_4
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace Real F
        inst✝¹ : NormedAddCommGroup F'
        inst✝ : NormedSpace Real F'
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
        C : Real
        hT_norm : ∀ (s : Set α), MeasurableSet s → LE.le (Norm.norm (T s)) (HMul.hMul  …
        f : MeasureTheory.SimpleFunc α F
        i✝ : F
        a✝ : Membership.mem f.range i✝
        ⊢ LE.le (Norm.norm (T (Set.preimage (⇑f) (Singleton.singleton i✝)))) (HMul.hMu …
      -/
      exact hT_norm _ <| SimpleFunc.measurableSet_fiber _ _
      /-
        🎉 no goals
      -/
                                                              /-
                                                                α : Type u_1
                                                                F : Type u_3
                                                                F' : Type u_4
                                                                inst✝³ : NormedAddCommGroup F
                                                                inst✝² : NormedSpace Real F
                                                                inst✝¹ : NormedAddCommGroup F'
                                                                inst✝ : NormedSpace Real F'
                                                                m : MeasurableSpace α
                                                                μ : MeasureTheory.Measure α
                                                                T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
                                                                C : Real
                                                                hT_norm : ∀ (s : Set α), MeasurableSet s → LE.le (Norm.norm (T s)) (HMul.hMul  …
                                                                f : MeasureTheory.SimpleFunc α F
                                                                ⊢ LE.le (f.range.sum fun x => HMul.hMul (HMul.hMul C (μ (Set.preimage (⇑f) (Si …
                                                              -/
    _ ≤ C * ∑ x ∈ f.range, (μ (f ⁻¹' {x})).toReal * ‖x‖ := by simp_rw [mul_sum, ← mul_assoc]; rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


theorem norm_setToSimpleFunc_le_sum_mul_norm_of_integrable (T : Set α → E →L[ℝ] F') {C : ℝ}
    (hT_norm : ∀ s, MeasurableSet s → μ s < ∞ → ‖T s‖ ≤ C * (μ s).toReal) (f : α →ₛ E)
    (hf : Integrable f μ) :
    ‖f.setToSimpleFunc T‖ ≤ C * ∑ x ∈ f.range, (μ (f ⁻¹' {x})).toReal * ‖x‖ :=
  calc
    ‖f.setToSimpleFunc T‖ ≤ ∑ x ∈ f.range, ‖T (f ⁻¹' {x})‖ * ‖x‖ :=
      norm_setToSimpleFunc_le_sum_opNorm T f
    _ ≤ ∑ x ∈ f.range, C * (μ (f ⁻¹' {x})).toReal * ‖x‖ := by
      /-
        α : Type u_1
        E : Type u_2
        F' : Type u_4
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F'
        inst✝ : NormedSpace Real F'
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
        C : Real
        hT_norm : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (Norm.n …
        f : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        ⊢ LE.le (f.range.sum fun x => HMul.hMul (Norm.norm (T (Set.preimage (⇑f) (Sing …
      -/
      refine Finset.sum_le_sum fun b hb => ?_
      /-
        α : Type u_1
        E : Type u_2
        F' : Type u_4
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F'
        inst✝ : NormedSpace Real F'
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
        C : Real
        hT_norm : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (Norm.n …
        f : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        b : E
        hb : Membership.mem f.range b
        ⊢ LE.le (HMul.hMul (Norm.norm (T (Set.preimage (⇑f) (Singleton.singleton b)))) …
      -/
      obtain rfl | hb := eq_or_ne b 0
        /-
          case inl
          α : Type u_1
          E : Type u_2
          F' : Type u_4
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace Real E
          inst✝¹ : NormedAddCommGroup F'
          inst✝ : NormedSpace Real F'
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
          C : Real
          hT_norm : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (Norm.n …
          f : MeasureTheory.SimpleFunc α E
          hf : MeasureTheory.Integrable (⇑f) μ
          hb : Membership.mem f.range 0
          ⊢ LE.le (HMul.hMul (Norm.norm (T (Set.preimage (⇑f) (Singleton.singleton 0)))) …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u_1
        E : Type u_2
        F' : Type u_4
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : NormedAddCommGroup F'
        inst✝ : NormedSpace Real F'
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
        C : Real
        hT_norm : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (Norm.n …
        f : MeasureTheory.SimpleFunc α E
        hf : MeasureTheory.Integrable (⇑f) μ
        b : E
        hb✝ : Membership.mem f.range b
        hb : Ne b 0
        ⊢ LE.le (HMul.hMul (Norm.norm (T (Set.preimage (⇑f) (Singleton.singleton b)))) …
      -/
      gcongr
      exact hT_norm _ (SimpleFunc.measurableSet_fiber _ _) <|
        SimpleFunc.measure_preimage_lt_top_of_integrable _ hf hb
                                                              /-
                                                                α : Type u_1
                                                                E : Type u_2
                                                                F' : Type u_4
                                                                inst✝³ : NormedAddCommGroup E
                                                                inst✝² : NormedSpace Real E
                                                                inst✝¹ : NormedAddCommGroup F'
                                                                inst✝ : NormedSpace Real F'
                                                                m : MeasurableSpace α
                                                                μ : MeasureTheory.Measure α
                                                                T : Set α → ContinuousLinearMap (RingHom.id Real) E F'
                                                                C : Real
                                                                hT_norm : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (Norm.n …
                                                                f : MeasureTheory.SimpleFunc α E
                                                                hf : MeasureTheory.Integrable (⇑f) μ
                                                                ⊢ LE.le (f.range.sum fun x => HMul.hMul (HMul.hMul C (μ (Set.preimage (⇑f) (Si …
                                                              -/
    _ ≤ C * ∑ x ∈ f.range, (μ (f ⁻¹' {x})).toReal * ‖x‖ := by simp_rw [mul_sum, ← mul_assoc]; rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


theorem setToSimpleFunc_indicator (T : Set α → F →L[ℝ] F') (hT_empty : T ∅ = 0)
    {m : MeasurableSpace α} {s : Set α} (hs : MeasurableSet s) (x : F) :
    SimpleFunc.setToSimpleFunc T
        (SimpleFunc.piecewise s hs (SimpleFunc.const α x) (SimpleFunc.const α 0)) =
      T s x := by
  classical
  obtain rfl | hs_empty := s.eq_empty_or_nonempty
  · simp only [hT_empty, ContinuousLinearMap.zero_apply, piecewise_empty, const_zero,
      setToSimpleFunc_zero_apply]
  simp_rw [setToSimpleFunc]
  obtain rfl | hs_univ := eq_or_ne s univ
  · haveI hα := hs_empty.to_type
    simp [← Function.const_def]
  rw [range_indicator hs hs_empty hs_univ]
  by_cases hx0 : x = 0
  · simp_rw [hx0]; simp
  rw [sum_insert]
  swap; · rw [Finset.mem_singleton]; exact hx0
  rw [sum_singleton, (T _).map_zero, add_zero]
  congr
  simp only [coe_piecewise, piecewise_eq_indicator, coe_const, Function.const_zero,
    piecewise_eq_indicator]
  rw [indicator_preimage, ← Function.const_def, preimage_const_of_mem]
  swap; · exact Set.mem_singleton x
  rw [← Function.const_zero, ← Function.const_def, preimage_const_of_not_mem]
  swap; · rw [Set.mem_singleton_iff]; exact Ne.symm hx0
  simp


theorem setToSimpleFunc_const' [Nonempty α] (T : Set α → F →L[ℝ] F') (x : F)
    {m : MeasurableSpace α} : SimpleFunc.setToSimpleFunc T (SimpleFunc.const α x) = T univ x := by
  simp only [setToSimpleFunc, range_const, Set.mem_singleton, preimage_const_of_mem,
    sum_singleton, ← Function.const_def, coe_const]


theorem setToSimpleFunc_const (T : Set α → F →L[ℝ] F') (hT_empty : T ∅ = 0) (x : F)
    {m : MeasurableSpace α} : SimpleFunc.setToSimpleFunc T (SimpleFunc.const α x) = T univ x := by
  /-
    α : Type u_1
    F : Type u_3
    F' : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup F'
    inst✝ : NormedSpace Real F'
    T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
    hT_empty : Eq (T EmptyCollection.emptyCollection) 0
    x : F
    m : MeasurableSpace α
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.SimpleFunc.con …
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      F : Type u_3
      F' : Type u_4
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      inst✝¹ : NormedAddCommGroup F'
      inst✝ : NormedSpace Real F'
      T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
      hT_empty : Eq (T EmptyCollection.emptyCollection) 0
      x : F
      m : MeasurableSpace α
      h✝ : IsEmpty α
      ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.SimpleFunc.con …
    -/
  · have h_univ_empty : (univ : Set α) = ∅ := Subsingleton.elim _ _
    /-
      case inl
      α : Type u_1
      F : Type u_3
      F' : Type u_4
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      inst✝¹ : NormedAddCommGroup F'
      inst✝ : NormedSpace Real F'
      T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
      hT_empty : Eq (T EmptyCollection.emptyCollection) 0
      x : F
      m : MeasurableSpace α
      h✝ : IsEmpty α
      h_univ_empty : Eq Set.univ EmptyCollection.emptyCollection
      ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.SimpleFunc.con …
    -/
    rw [h_univ_empty, hT_empty]
    simp only [setToSimpleFunc, ContinuousLinearMap.zero_apply, sum_empty,
      range_eq_empty_of_isEmpty]
    /-
      case inr
      α : Type u_1
      F : Type u_3
      F' : Type u_4
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      inst✝¹ : NormedAddCommGroup F'
      inst✝ : NormedSpace Real F'
      T : Set α → ContinuousLinearMap (RingHom.id Real) F F'
      hT_empty : Eq (T EmptyCollection.emptyCollection) 0
      x : F
      m : MeasurableSpace α
      h✝ : Nonempty α
      ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.SimpleFunc.con …
    -/
  · exact setToSimpleFunc_const' T x
    /-
      🎉 no goals
    -/


theorem norm_eq_sum_mul (f : α →₁ₛ[μ] G) :
    ‖f‖ = ∑ x ∈ (toSimpleFunc f).range, (μ (toSimpleFunc f ⁻¹' {x})).toReal * ‖x‖ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
    ⊢ Eq (Norm.norm f) ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).range.sum fun …
  -/
  rw [norm_toSimpleFunc, eLpNorm_one_eq_lintegral_nnnorm]
  /-
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
    ⊢ Eq (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((MeasureTheory.Lp.si …
  -/
  have h_eq := SimpleFunc.map_apply (fun x => (‖x‖₊ : ℝ≥0∞)) (toSimpleFunc f)
  /-
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
    h_eq : ∀ (a : α), Eq ((MeasureTheory.SimpleFunc.map (fun x => ↑(NNNorm.nnnorm  …
    ⊢ Eq (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm ((MeasureTheory.Lp.si …
  -/
  simp_rw [← h_eq]
  /-
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
    h_eq : ∀ (a : α), Eq ((MeasureTheory.SimpleFunc.map (fun x => ↑(NNNorm.nnnorm  …
    ⊢ Eq (MeasureTheory.lintegral μ fun x => (MeasureTheory.SimpleFunc.map (fun x  …
  -/
  rw [SimpleFunc.lintegral_eq_lintegral, SimpleFunc.map_lintegral, ENNReal.toReal_sum]
    /-
      α : Type u_1
      G : Type u_5
      inst✝ : NormedAddCommGroup G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
      h_eq : ∀ (a : α), Eq ((MeasureTheory.SimpleFunc.map (fun x => ↑(NNNorm.nnnorm  …
      ⊢ Eq ((MeasureTheory.Lp.simpleFunc.toSimpleFunc f).range.sum fun a => (HMul.hM …
    -/
  · congr
    /-
      case e_f
      α : Type u_1
      G : Type u_5
      inst✝ : NormedAddCommGroup G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
      h_eq : ∀ (a : α), Eq ((MeasureTheory.SimpleFunc.map (fun x => ↑(NNNorm.nnnorm  …
      ⊢ Eq (fun a => (HMul.hMul (↑(NNNorm.nnnorm a)) (μ (Set.preimage (⇑(MeasureTheo …
    -/
    ext1 x
    rw [ENNReal.toReal_mul, mul_comm, ← ofReal_norm_eq_coe_nnnorm,
      ENNReal.toReal_ofReal (norm_nonneg _)]
    /-
      α : Type u_1
      G : Type u_5
      inst✝ : NormedAddCommGroup G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
      h_eq : ∀ (a : α), Eq ((MeasureTheory.SimpleFunc.map (fun x => ↑(NNNorm.nnnorm  …
      ⊢ ∀ (a : G), Membership.mem (MeasureTheory.Lp.simpleFunc.toSimpleFunc f).range …
    -/
  · intro x _
    /-
      α : Type u_1
      G : Type u_5
      inst✝ : NormedAddCommGroup G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
      h_eq : ∀ (a : α), Eq ((MeasureTheory.SimpleFunc.map (fun x => ↑(NNNorm.nnnorm  …
      x : G
      a✝ : Membership.mem (MeasureTheory.Lp.simpleFunc.toSimpleFunc f).range x
      ⊢ Ne (HMul.hMul (↑(NNNorm.nnnorm x)) (μ (Set.preimage (⇑(MeasureTheory.Lp.simp …
    -/
    by_cases hx0 : x = 0
      /-
        case pos
        α : Type u_1
        G : Type u_5
        inst✝ : NormedAddCommGroup G
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G 1 μ) x
        h_eq : ∀ (a : α), Eq ((MeasureTheory.SimpleFunc.map (fun x => ↑(NNNorm.nnnorm  …
        x : G
        a✝ : Membership.mem (MeasureTheory.Lp.simpleFunc.toSimpleFunc f).range x
        hx0 : Eq x 0
        ⊢ Ne (HMul.hMul (↑(NNNorm.nnnorm x)) (μ (Set.preimage (⇑(MeasureTheory.Lp.simp …
      -/
    · rw [hx0]; simp
                /-
                  🎉 no goals
                -/
    · exact
        ENNReal.mul_ne_top ENNReal.coe_ne_top
          (SimpleFunc.measure_preimage_lt_top_of_integrable _ (SimpleFunc.integrable f) hx0).ne


/-- Extend `Set α → (E →L[ℝ] F')` to `(α →₁ₛ[μ] E) → F'`. -/
def setToL1S (T : Set α → E →L[ℝ] F) (f : α →₁ₛ[μ] E) : F :=
  (toSimpleFunc f).setToSimpleFunc T


theorem setToL1S_eq_setToSimpleFunc (T : Set α → E →L[ℝ] F) (f : α →₁ₛ[μ] E) :
    setToL1S T f = (toSimpleFunc f).setToSimpleFunc T :=
  rfl


@[simp]
theorem setToL1S_zero_left (f : α →₁ₛ[μ] E) : setToL1S (0 : Set α → E →L[ℝ] F) f = 0 :=
  SimpleFunc.setToSimpleFunc_zero _


theorem setToL1S_zero_left' {T : Set α → E →L[ℝ] F}
    (h_zero : ∀ s, MeasurableSet s → μ s < ∞ → T s = 0) (f : α →₁ₛ[μ] E) : setToL1S T f = 0 :=
  SimpleFunc.setToSimpleFunc_zero' h_zero _ (SimpleFunc.integrable f)


theorem setToL1S_congr (T : Set α → E →L[ℝ] F) (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0)
    (h_add : FinMeasAdditive μ T) {f g : α →₁ₛ[μ] E} (h : toSimpleFunc f =ᵐ[μ] toSimpleFunc g) :
    setToL1S T f = setToL1S T g :=
  SimpleFunc.setToSimpleFunc_congr T h_zero h_add (SimpleFunc.integrable f) h


theorem setToL1S_congr_left (T T' : Set α → E →L[ℝ] F)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → T s = T' s) (f : α →₁ₛ[μ] E) :
    setToL1S T f = setToL1S T' f :=
  SimpleFunc.setToSimpleFunc_congr_left T T' h (simpleFunc.toSimpleFunc f) (SimpleFunc.integrable f)


/-- `setToL1S` does not change if we replace the measure `μ` by `μ'` with `μ ≪ μ'`. The statement
uses two functions `f` and `f'` because they have to belong to different types, but morally these
are the same function (we have `f =ᵐ[μ] f'`). -/
theorem setToL1S_congr_measure {μ' : Measure α} (T : Set α → E →L[ℝ] F)
    (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0) (h_add : FinMeasAdditive μ T) (hμ : μ ≪ μ')
    (f : α →₁ₛ[μ] E) (f' : α →₁ₛ[μ'] E) (h : (f : α → E) =ᵐ[μ] f') :
    setToL1S T f = setToL1S T f' := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hμ : μ.AbsolutelyContinuous μ'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    f' : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ') x
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑↑f ↑↑↑f'
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.setToL1S T f) (MeasureTheory.L1.SimpleFunc.s …
  -/
  refine SimpleFunc.setToSimpleFunc_congr T h_zero h_add (SimpleFunc.integrable f) ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hμ : μ.AbsolutelyContinuous μ'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    f' : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ') x
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑↑f ↑↑↑f'
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  refine (toSimpleFunc_eq_toFun f).trans ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hμ : μ.AbsolutelyContinuous μ'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    f' : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ') x
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑↑f ↑↑↑f'
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑f ⇑(MeasureTheory.Lp.simpleFunc.toSimpl …
  -/
  suffices (f' : α → E) =ᵐ[μ] simpleFunc.toSimpleFunc f' from h.trans this
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hμ : μ.AbsolutelyContinuous μ'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    f' : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ') x
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑↑f ↑↑↑f'
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑f' ⇑(MeasureTheory.Lp.simpleFunc.toSimp …
  -/
  have goal' : (f' : α → E) =ᵐ[μ'] simpleFunc.toSimpleFunc f' := (toSimpleFunc_eq_toFun f').symm
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hμ : μ.AbsolutelyContinuous μ'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    f' : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ') x
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑↑f ↑↑↑f'
    goal' : (MeasureTheory.ae μ').EventuallyEq ↑↑↑f' ⇑(MeasureTheory.Lp.simpleFunc …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑f' ⇑(MeasureTheory.Lp.simpleFunc.toSimp …
  -/
  exact hμ.ae_eq goal'
  /-
    🎉 no goals
  -/


theorem setToL1S_add_left (T T' : Set α → E →L[ℝ] F) (f : α →₁ₛ[μ] E) :
    setToL1S (T + T') f = setToL1S T f + setToL1S T' f :=
  SimpleFunc.setToSimpleFunc_add_left T T'


theorem setToL1S_add_left' (T T' T'' : Set α → E →L[ℝ] F)
    (h_add : ∀ s, MeasurableSet s → μ s < ∞ → T'' s = T s + T' s) (f : α →₁ₛ[μ] E) :
    setToL1S T'' f = setToL1S T f + setToL1S T' f :=
  SimpleFunc.setToSimpleFunc_add_left' T T' T'' h_add (SimpleFunc.integrable f)


theorem setToL1S_smul_left (T : Set α → E →L[ℝ] F) (c : ℝ) (f : α →₁ₛ[μ] E) :
    setToL1S (fun s => c • T s) f = c • setToL1S T f :=
  SimpleFunc.setToSimpleFunc_smul_left T c _


theorem setToL1S_smul_left' (T T' : Set α → E →L[ℝ] F) (c : ℝ)
    (h_smul : ∀ s, MeasurableSet s → μ s < ∞ → T' s = c • T s) (f : α →₁ₛ[μ] E) :
    setToL1S T' f = c • setToL1S T f :=
  SimpleFunc.setToSimpleFunc_smul_left' T T' c h_smul (SimpleFunc.integrable f)


theorem setToL1S_add (T : Set α → E →L[ℝ] F) (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0)
    (h_add : FinMeasAdditive μ T) (f g : α →₁ₛ[μ] E) :
    setToL1S T (f + g) = setToL1S T f + setToL1S T g := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.setToL1S T (HAdd.hAdd f g)) (HAdd.hAdd (Meas …
  -/
  simp_rw [setToL1S]
  rw [← SimpleFunc.setToSimpleFunc_add T h_add (SimpleFunc.integrable f)
      (SimpleFunc.integrable g)]
  exact
    SimpleFunc.setToSimpleFunc_congr T h_zero h_add (SimpleFunc.integrable _)
      (add_toSimpleFunc f g)


theorem setToL1S_neg {T : Set α → E →L[ℝ] F} (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0)
    (h_add : FinMeasAdditive μ T) (f : α →₁ₛ[μ] E) : setToL1S T (-f) = -setToL1S T f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.setToL1S T (Neg.neg f)) (Neg.neg (MeasureThe …
  -/
  simp_rw [setToL1S]
  have : simpleFunc.toSimpleFunc (-f) =ᵐ[μ] ⇑(-simpleFunc.toSimpleFunc f) :=
    neg_toSimpleFunc f
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    this : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpl …
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.Lp.simpleFunc. …
  -/
  rw [SimpleFunc.setToSimpleFunc_congr T h_zero h_add (SimpleFunc.integrable _) this]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    this : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpl …
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (Neg.neg (MeasureTheory.Lp.si …
  -/
  exact SimpleFunc.setToSimpleFunc_neg T h_add (SimpleFunc.integrable f)
  /-
    🎉 no goals
  -/


theorem setToL1S_sub {T : Set α → E →L[ℝ] F} (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0)
    (h_add : FinMeasAdditive μ T) (f g : α →₁ₛ[μ] E) :
    setToL1S T (f - g) = setToL1S T f - setToL1S T g := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.setToL1S T (HSub.hSub f g)) (HSub.hSub (Meas …
  -/
  rw [sub_eq_add_neg, setToL1S_add T h_zero h_add, setToL1S_neg h_zero h_add, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem setToL1S_smul_real (T : Set α → E →L[ℝ] F)
    (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0) (h_add : FinMeasAdditive μ T) (c : ℝ)
    (f : α →₁ₛ[μ] E) : setToL1S T (c • f) = c • setToL1S T f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    c : Real
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.setToL1S T (HSMul.hSMul c f)) (HSMul.hSMul c …
  -/
  simp_rw [setToL1S]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    c : Real
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.Lp.simpleFunc. …
  -/
  rw [← SimpleFunc.setToSimpleFunc_smul_real T h_add c (SimpleFunc.integrable f)]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    c : Real
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.Lp.simpleFunc. …
  -/
  refine SimpleFunc.setToSimpleFunc_congr T h_zero h_add (SimpleFunc.integrable _) ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    c : Real
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  exact smul_toSimpleFunc c f
  /-
    🎉 no goals
  -/


theorem setToL1S_smul {E} [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedSpace 𝕜 E]
    [NormedSpace 𝕜 F] (T : Set α → E →L[ℝ] F) (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0)
    (h_add : FinMeasAdditive μ T) (h_smul : ∀ c : 𝕜, ∀ s x, T s (c • x) = c • T s x) (c : 𝕜)
    (f : α →₁ₛ[μ] E) : setToL1S T (c • f) = c • setToL1S T f := by
  /-
    α : Type u_1
    F : Type u_3
    𝕜 : Type u_6
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedField 𝕜
    E : Type u_7
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.setToL1S T (HSMul.hSMul c f)) (HSMul.hSMul c …
  -/
  simp_rw [setToL1S]
  /-
    α : Type u_1
    F : Type u_3
    𝕜 : Type u_6
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedField 𝕜
    E : Type u_7
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.Lp.simpleFunc. …
  -/
  rw [← SimpleFunc.setToSimpleFunc_smul T h_add h_smul c (SimpleFunc.integrable f)]
  /-
    α : Type u_1
    F : Type u_3
    𝕜 : Type u_6
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedField 𝕜
    E : Type u_7
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.Lp.simpleFunc. …
  -/
  refine SimpleFunc.setToSimpleFunc_congr T h_zero h_add (SimpleFunc.integrable _) ?_
  /-
    α : Type u_1
    F : Type u_3
    𝕜 : Type u_6
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedField 𝕜
    E : Type u_7
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  exact smul_toSimpleFunc c f
  /-
    🎉 no goals
  -/


theorem norm_setToL1S_le (T : Set α → E →L[ℝ] F) {C : ℝ}
    (hT_norm : ∀ s, MeasurableSet s → μ s < ∞ → ‖T s‖ ≤ C * (μ s).toReal) (f : α →₁ₛ[μ] E) :
    ‖setToL1S T f‖ ≤ C * ‖f‖ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT_norm : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (Norm.n …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ LE.le (Norm.norm (MeasureTheory.L1.SimpleFunc.setToL1S T f)) (HMul.hMul C (N …
  -/
  rw [setToL1S, norm_eq_sum_mul f]
  exact
    SimpleFunc.norm_setToSimpleFunc_le_sum_mul_norm_of_integrable T hT_norm _
      (SimpleFunc.integrable f)


theorem setToL1S_indicatorConst {T : Set α → E →L[ℝ] F} {s : Set α}
    (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0) (h_add : FinMeasAdditive μ T)
    (hs : MeasurableSet s) (hμs : μ s < ∞) (x : E) :
    setToL1S T (simpleFunc.indicatorConst 1 hs hμs.ne x) = T s x := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    s : Set α
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    x : E
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.setToL1S T (MeasureTheory.Lp.simpleFunc.indi …
  -/
  have h_empty : T ∅ = 0 := h_zero _ MeasurableSet.empty measure_empty
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    s : Set α
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    x : E
    h_empty : Eq (T EmptyCollection.emptyCollection) 0
    ⊢ Eq (MeasureTheory.L1.SimpleFunc.setToL1S T (MeasureTheory.Lp.simpleFunc.indi …
  -/
  rw [setToL1S_eq_setToSimpleFunc]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    s : Set α
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    x : E
    h_empty : Eq (T EmptyCollection.emptyCollection) 0
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.Lp.simpleFunc. …
  -/
  refine Eq.trans ?_ (SimpleFunc.setToSimpleFunc_indicator T h_empty hs x)
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    s : Set α
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    x : E
    h_empty : Eq (T EmptyCollection.emptyCollection) 0
    ⊢ Eq (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.Lp.simpleFunc. …
  -/
  refine SimpleFunc.setToSimpleFunc_congr T h_zero h_add (SimpleFunc.integrable _) ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    s : Set α
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    x : E
    h_empty : Eq (T EmptyCollection.emptyCollection) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpleFunc …
  -/
  exact toSimpleFunc_indicatorConst hs hμs.ne x
  /-
    🎉 no goals
  -/


theorem setToL1S_const [IsFiniteMeasure μ] {T : Set α → E →L[ℝ] F}
    (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0) (h_add : FinMeasAdditive μ T) (x : E) :
    setToL1S T (simpleFunc.indicatorConst 1 MeasurableSet.univ (measure_ne_top μ _) x) = T univ x :=
  setToL1S_indicatorConst h_zero h_add MeasurableSet.univ (measure_lt_top _ _) x


theorem setToL1S_mono_left {T T' : Set α → E →L[ℝ] G''} (hTT' : ∀ s x, T s x ≤ T' s x)
    (f : α →₁ₛ[μ] E) : setToL1S T f ≤ setToL1S T' f :=
  SimpleFunc.setToSimpleFunc_mono_left T T' hTT' _


theorem setToL1S_mono_left' {T T' : Set α → E →L[ℝ] G''}
    (hTT' : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, T s x ≤ T' s x) (f : α →₁ₛ[μ] E) :
    setToL1S T f ≤ setToL1S T' f :=
  SimpleFunc.setToSimpleFunc_mono_left' T T' hTT' _ (SimpleFunc.integrable f)


theorem setToL1S_nonneg (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0)
    (h_add : FinMeasAdditive μ T)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f : α →₁ₛ[μ] G''}
    (hf : 0 ≤ f) : 0 ≤ setToL1S T f := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G'' : Type u_7
    G' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G'
    inst✝² : NormedSpace Real G'
    inst✝¹ : NormedLatticeAddCommGroup G''
    inst✝ : NormedSpace Real G''
    T : Set α → ContinuousLinearMap (RingHom.id Real) G'' G'
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'') …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G'' 1 μ) x
    hf : LE.le 0 f
    ⊢ LE.le 0 (MeasureTheory.L1.SimpleFunc.setToL1S T f)
  -/
  simp_rw [setToL1S]
  obtain ⟨f', hf', hff'⟩ : ∃ f' : α →ₛ G'', 0 ≤ f' ∧ simpleFunc.toSimpleFunc f =ᵐ[μ] f' := by
    obtain ⟨f'', hf'', hff''⟩ := exists_simpleFunc_nonneg_ae_eq hf
    exact ⟨f'', hf'', (Lp.simpleFunc.toSimpleFunc_eq_toFun f).trans hff''⟩
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G'' : Type u_7
    G' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G'
    inst✝² : NormedSpace Real G'
    inst✝¹ : NormedLatticeAddCommGroup G''
    inst✝ : NormedSpace Real G''
    T : Set α → ContinuousLinearMap (RingHom.id Real) G'' G'
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'') …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G'' 1 μ) x
    hf : LE.le 0 f
    f' : MeasureTheory.SimpleFunc α G''
    hf' : LE.le 0 f'
    hff' : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.Lp.simpleFunc.toSimpl …
    ⊢ LE.le 0 (MeasureTheory.SimpleFunc.setToSimpleFunc T (MeasureTheory.Lp.simple …
  -/
  rw [SimpleFunc.setToSimpleFunc_congr _ h_zero h_add (SimpleFunc.integrable _) hff']
  exact
    SimpleFunc.setToSimpleFunc_nonneg' T hT_nonneg _ hf' ((SimpleFunc.integrable f).congr hff')


theorem setToL1S_mono (h_zero : ∀ s, MeasurableSet s → μ s = 0 → T s = 0)
    (h_add : FinMeasAdditive μ T)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f g : α →₁ₛ[μ] G''}
    (hfg : f ≤ g) : setToL1S T f ≤ setToL1S T g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G'' : Type u_7
    G' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G'
    inst✝² : NormedSpace Real G'
    inst✝¹ : NormedLatticeAddCommGroup G''
    inst✝ : NormedSpace Real G''
    T : Set α → ContinuousLinearMap (RingHom.id Real) G'' G'
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'') …
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G'' 1 μ) x
    hfg : LE.le f g
    ⊢ LE.le (MeasureTheory.L1.SimpleFunc.setToL1S T f) (MeasureTheory.L1.SimpleFun …
  -/
  rw [← sub_nonneg] at hfg ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G'' : Type u_7
    G' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G'
    inst✝² : NormedSpace Real G'
    inst✝¹ : NormedLatticeAddCommGroup G''
    inst✝ : NormedSpace Real G''
    T : Set α → ContinuousLinearMap (RingHom.id Real) G'' G'
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'') …
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G'' 1 μ) x
    hfg : LE.le 0 (HSub.hSub g f)
    ⊢ LE.le 0 (HSub.hSub (MeasureTheory.L1.SimpleFunc.setToL1S T g) (MeasureTheory …
  -/
  rw [← setToL1S_sub h_zero h_add]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G'' : Type u_7
    G' : Type u_8
    inst✝³ : NormedLatticeAddCommGroup G'
    inst✝² : NormedSpace Real G'
    inst✝¹ : NormedLatticeAddCommGroup G''
    inst✝ : NormedSpace Real G''
    T : Set α → ContinuousLinearMap (RingHom.id Real) G'' G'
    h_zero : ∀ (s : Set α), MeasurableSet s → Eq (μ s) 0 → Eq (T s) 0
    h_add : MeasureTheory.FinMeasAdditive μ T
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'') …
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc G'' 1 μ) x
    hfg : LE.le 0 (HSub.hSub g f)
    ⊢ LE.le 0 (MeasureTheory.L1.SimpleFunc.setToL1S T (HSub.hSub g f))
  -/
  exact setToL1S_nonneg h_zero h_add hT_nonneg hfg
  /-
    🎉 no goals
  -/


/-- Extend `Set α → E →L[ℝ] F` to `(α →₁ₛ[μ] E) →L[𝕜] F`. -/
def setToL1SCLM' {T : Set α → E →L[ℝ] F} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C)
    (h_smul : ∀ c : 𝕜, ∀ s x, T s (c • x) = c • T s x) : (α →₁ₛ[μ] E) →L[𝕜] F :=
  LinearMap.mkContinuous
    ⟨⟨setToL1S T, setToL1S_add T (fun _ => hT.eq_zero_of_measure_zero) hT.1⟩,
      setToL1S_smul T (fun _ => hT.eq_zero_of_measure_zero) hT.1 h_smul⟩
    C fun f => norm_setToL1S_le T hT.2 f


/-- Extend `Set α → E →L[ℝ] F` to `(α →₁ₛ[μ] E) →L[ℝ] F`. -/
def setToL1SCLM {T : Set α → E →L[ℝ] F} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C) :
    (α →₁ₛ[μ] E) →L[ℝ] F :=
  LinearMap.mkContinuous
    ⟨⟨setToL1S T, setToL1S_add T (fun _ => hT.eq_zero_of_measure_zero) hT.1⟩,
      setToL1S_smul_real T (fun _ => hT.eq_zero_of_measure_zero) hT.1⟩
    C fun f => norm_setToL1S_le T hT.2 f


@[simp]
theorem setToL1SCLM_zero_left (hT : DominatedFinMeasAdditive μ (0 : Set α → E →L[ℝ] F) C)
    (f : α →₁ₛ[μ] E) : setToL1SCLM α E μ hT f = 0 :=
  setToL1S_zero_left _


theorem setToL1SCLM_zero_left' (hT : DominatedFinMeasAdditive μ T C)
    (h_zero : ∀ s, MeasurableSet s → μ s < ∞ → T s = 0) (f : α →₁ₛ[μ] E) :
    setToL1SCLM α E μ hT f = 0 :=
  setToL1S_zero_left' h_zero f


theorem setToL1SCLM_congr_left (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (h : T = T') (f : α →₁ₛ[μ] E) :
    setToL1SCLM α E μ hT f = setToL1SCLM α E μ hT' f :=
                                            /-
                                              α : Type u_1
                                              E : Type u_2
                                              F : Type u_3
                                              inst✝³ : NormedAddCommGroup E
                                              inst✝² : NormedSpace Real E
                                              inst✝¹ : NormedAddCommGroup F
                                              inst✝ : NormedSpace Real F
                                              m : MeasurableSpace α
                                              μ : MeasureTheory.Measure α
                                              T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
                                              C C' : Real
                                              hT : MeasureTheory.DominatedFinMeasAdditive μ T C
                                              hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
                                              h : Eq T T'
                                              f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
                                              x✝² : Set α
                                              x✝¹ : MeasurableSet x✝²
                                              x✝ : LT.lt (μ x✝²) Top.top
                                              ⊢ Eq (T x✝²) (T' x✝²)
                                            -/
  setToL1S_congr_left T T' (fun _ _ _ => by rw [h]) f
                                            /-
                                              🎉 no goals
                                            -/


theorem setToL1SCLM_congr_left' (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (h : ∀ s, MeasurableSet s → μ s < ∞ → T s = T' s)
    (f : α →₁ₛ[μ] E) : setToL1SCLM α E μ hT f = setToL1SCLM α E μ hT' f :=
  setToL1S_congr_left T T' h f


theorem setToL1SCLM_congr_measure {μ' : Measure α} (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ' T C') (hμ : μ ≪ μ') (f : α →₁ₛ[μ] E) (f' : α →₁ₛ[μ'] E)
    (h : (f : α → E) =ᵐ[μ] f') : setToL1SCLM α E μ hT f = setToL1SCLM α E μ' hT' f' :=
  setToL1S_congr_measure T (fun _ => hT.eq_zero_of_measure_zero) hT.1 hμ _ _ h


theorem setToL1SCLM_add_left (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (f : α →₁ₛ[μ] E) :
    setToL1SCLM α E μ (hT.add hT') f = setToL1SCLM α E μ hT f + setToL1SCLM α E μ hT' f :=
  setToL1S_add_left T T' f


theorem setToL1SCLM_add_left' (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (hT'' : DominatedFinMeasAdditive μ T'' C'')
    (h_add : ∀ s, MeasurableSet s → μ s < ∞ → T'' s = T s + T' s) (f : α →₁ₛ[μ] E) :
    setToL1SCLM α E μ hT'' f = setToL1SCLM α E μ hT f + setToL1SCLM α E μ hT' f :=
  setToL1S_add_left' T T' T'' h_add f


theorem setToL1SCLM_smul_left (c : ℝ) (hT : DominatedFinMeasAdditive μ T C) (f : α →₁ₛ[μ] E) :
    setToL1SCLM α E μ (hT.smul c) f = c • setToL1SCLM α E μ hT f :=
  setToL1S_smul_left T c f


theorem setToL1SCLM_smul_left' (c : ℝ) (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C')
    (h_smul : ∀ s, MeasurableSet s → μ s < ∞ → T' s = c • T s) (f : α →₁ₛ[μ] E) :
    setToL1SCLM α E μ hT' f = c • setToL1SCLM α E μ hT f :=
  setToL1S_smul_left' T T' c h_smul f


theorem norm_setToL1SCLM_le {T : Set α → E →L[ℝ] F} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C)
    (hC : 0 ≤ C) : ‖setToL1SCLM α E μ hT‖ ≤ C :=
  LinearMap.mkContinuous_norm_le _ hC _


theorem norm_setToL1SCLM_le' {T : Set α → E →L[ℝ] F} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C) :
    ‖setToL1SCLM α E μ hT‖ ≤ max C 0 :=
  LinearMap.mkContinuous_norm_le' _ _


theorem setToL1SCLM_const [IsFiniteMeasure μ] {T : Set α → E →L[ℝ] F} {C : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (x : E) :
    setToL1SCLM α E μ hT (simpleFunc.indicatorConst 1 MeasurableSet.univ (measure_ne_top μ _) x) =
      T univ x :=
  setToL1S_const (fun _ => hT.eq_zero_of_measure_zero) hT.1 x


theorem setToL1SCLM_mono_left {T T' : Set α → E →L[ℝ] G''} {C C' : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C')
    (hTT' : ∀ s x, T s x ≤ T' s x) (f : α →₁ₛ[μ] E) :
    setToL1SCLM α E μ hT f ≤ setToL1SCLM α E μ hT' f :=
  SimpleFunc.setToSimpleFunc_mono_left T T' hTT' _


theorem setToL1SCLM_mono_left' {T T' : Set α → E →L[ℝ] G''} {C C' : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C')
    (hTT' : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, T s x ≤ T' s x) (f : α →₁ₛ[μ] E) :
    setToL1SCLM α E μ hT f ≤ setToL1SCLM α E μ hT' f :=
  SimpleFunc.setToSimpleFunc_mono_left' T T' hTT' _ (SimpleFunc.integrable f)


theorem setToL1SCLM_nonneg {T : Set α → G' →L[ℝ] G''} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f : α →₁ₛ[μ] G'}
    (hf : 0 ≤ f) : 0 ≤ setToL1SCLM α G' μ hT f :=
  setToL1S_nonneg (fun _ => hT.eq_zero_of_measure_zero) hT.1 hT_nonneg hf


theorem setToL1SCLM_mono {T : Set α → G' →L[ℝ] G''} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f g : α →₁ₛ[μ] G'}
    (hfg : f ≤ g) : setToL1SCLM α G' μ hT f ≤ setToL1SCLM α G' μ hT g :=
  setToL1S_mono (fun _ => hT.eq_zero_of_measure_zero) hT.1 hT_nonneg hfg


/-- Extend `set α → (E →L[ℝ] F)` to `(α →₁[μ] E) →L[𝕜] F`. -/
def setToL1' (hT : DominatedFinMeasAdditive μ T C)
    (h_smul : ∀ c : 𝕜, ∀ s x, T s (c • x) = c • T s x) : (α →₁[μ] E) →L[𝕜] F :=
  (setToL1SCLM' α E 𝕜 μ hT h_smul).extend (coeToLp α E 𝕜) (simpleFunc.denseRange one_ne_top)
    simpleFunc.isUniformInducing


/-- Extend `Set α → E →L[ℝ] F` to `(α →₁[μ] E) →L[ℝ] F`. -/
def setToL1 (hT : DominatedFinMeasAdditive μ T C) : (α →₁[μ] E) →L[ℝ] F :=
  (setToL1SCLM α E μ hT).extend (coeToLp α E ℝ) (simpleFunc.denseRange one_ne_top)
    simpleFunc.isUniformInducing


theorem setToL1_eq_setToL1SCLM (hT : DominatedFinMeasAdditive μ T C) (f : α →₁ₛ[μ] E) :
    setToL1 hT f = setToL1SCLM α E μ hT f :=
  uniformly_extend_of_ind simpleFunc.isUniformInducing (simpleFunc.denseRange one_ne_top)
    (setToL1SCLM α E μ hT).uniformContinuous _


theorem setToL1_eq_setToL1' (hT : DominatedFinMeasAdditive μ T C)
    (h_smul : ∀ c : 𝕜, ∀ s x, T s (c • x) = c • T s x) (f : α →₁[μ] E) :
    setToL1 hT f = setToL1' 𝕜 hT h_smul f :=
  rfl


@[simp]
theorem setToL1_zero_left (hT : DominatedFinMeasAdditive μ (0 : Set α → E →L[ℝ] F) C)
    (f : α →₁[μ] E) : setToL1 hT f = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ 0 C
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) f) 0
  -/
  suffices setToL1 hT = 0 by rw [this]; simp
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ 0 C
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.setToL1 hT) 0
  -/
  refine ContinuousLinearMap.extend_unique (setToL1SCLM α E μ hT) _ _ _ _ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ 0 C
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (ContinuousLinearMap.comp 0 (MeasureTheory.Lp.simpleFunc.coeToLp α E Real …
  -/
  ext1 f
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ 0 C
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq ((ContinuousLinearMap.comp 0 (MeasureTheory.Lp.simpleFunc.coeToLp α E Rea …
  -/
  rw [setToL1SCLM_zero_left hT f, ContinuousLinearMap.zero_comp, ContinuousLinearMap.zero_apply]
  /-
    🎉 no goals
  -/


theorem setToL1_zero_left' (hT : DominatedFinMeasAdditive μ T C)
    (h_zero : ∀ s, MeasurableSet s → μ s < ∞ → T s = 0) (f : α →₁[μ] E) : setToL1 hT f = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) f) 0
  -/
  suffices setToL1 hT = 0 by rw [this]; simp
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.setToL1 hT) 0
  -/
  refine ContinuousLinearMap.extend_unique (setToL1SCLM α E μ hT) _ _ _ _ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (ContinuousLinearMap.comp 0 (MeasureTheory.Lp.simpleFunc.coeToLp α E Real …
  -/
  ext1 f
  rw [setToL1SCLM_zero_left' hT h_zero f, ContinuousLinearMap.zero_comp,
    ContinuousLinearMap.zero_apply]


theorem setToL1_congr_left (T T' : Set α → E →L[ℝ] F) {C C' : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C') (h : T = T')
    (f : α →₁[μ] E) : setToL1 hT f = setToL1 hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : Eq T T'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) f) ((MeasureTheory.L1.setToL1 hT') f)
  -/
  suffices setToL1 hT = setToL1 hT' by rw [this]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : Eq T T'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.setToL1 hT) (MeasureTheory.L1.setToL1 hT')
  -/
  refine ContinuousLinearMap.extend_unique (setToL1SCLM α E μ hT) _ _ _ _ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : Eq T T'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT').comp (MeasureTheory.Lp.simpleFunc.coeToLp …
  -/
  ext1 f
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : Eq T T'
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (((MeasureTheory.L1.setToL1 hT').comp (MeasureTheory.Lp.simpleFunc.coeToL …
  -/
  suffices setToL1 hT' f = setToL1SCLM α E μ hT f by rw [← this]; rfl
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : Eq T T'
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT') ↑f) ((MeasureTheory.L1.SimpleFunc.setToL1 …
  -/
  rw [setToL1_eq_setToL1SCLM]
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : Eq T T'
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.SimpleFunc.setToL1SCLM α E μ hT') f) ((MeasureTheory.L …
  -/
  exact setToL1SCLM_congr_left hT' hT h.symm f
  /-
    🎉 no goals
  -/


theorem setToL1_congr_left' (T T' : Set α → E →L[ℝ] F) {C C' : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C')
    (h : ∀ s, MeasurableSet s → μ s < ∞ → T s = T' s) (f : α →₁[μ] E) :
    setToL1 hT f = setToL1 hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) f) ((MeasureTheory.L1.setToL1 hT') f)
  -/
  suffices setToL1 hT = setToL1 hT' by rw [this]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.setToL1 hT) (MeasureTheory.L1.setToL1 hT')
  -/
  refine ContinuousLinearMap.extend_unique (setToL1SCLM α E μ hT) _ _ _ _ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT').comp (MeasureTheory.Lp.simpleFunc.coeToLp …
  -/
  ext1 f
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (((MeasureTheory.L1.setToL1 hT').comp (MeasureTheory.Lp.simpleFunc.coeToL …
  -/
  suffices setToL1 hT' f = setToL1SCLM α E μ hT f by rw [← this]; rfl
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT') ↑f) ((MeasureTheory.L1.SimpleFunc.setToL1 …
  -/
  rw [setToL1_eq_setToL1SCLM]
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.SimpleFunc.setToL1SCLM α E μ hT') f) ((MeasureTheory.L …
  -/
  exact (setToL1SCLM_congr_left' hT hT' h f).symm
  /-
    🎉 no goals
  -/


theorem setToL1_add_left (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (f : α →₁[μ] E) :
    setToL1 (hT.add hT') f = setToL1 hT f + setToL1 hT' f := by
  suffices setToL1 (hT.add hT') = setToL1 hT + setToL1 hT' by
    rw [this, ContinuousLinearMap.add_apply]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.setToL1 ⋯) (HAdd.hAdd (MeasureTheory.L1.setToL1 hT) (Me …
  -/
  refine ContinuousLinearMap.extend_unique (setToL1SCLM α E μ (hT.add hT')) _ _ _ _ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((HAdd.hAdd (MeasureTheory.L1.setToL1 hT) (MeasureTheory.L1.setToL1 hT')) …
  -/
  ext1 f
  suffices setToL1 hT f + setToL1 hT' f = setToL1SCLM α E μ (hT.add hT') f by
    rw [← this]; rfl
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (HAdd.hAdd ((MeasureTheory.L1.setToL1 hT) ↑f) ((MeasureTheory.L1.setToL1  …
  -/
  rw [setToL1_eq_setToL1SCLM, setToL1_eq_setToL1SCLM, setToL1SCLM_add_left hT hT']
  /-
    🎉 no goals
  -/


theorem setToL1_add_left' (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (hT'' : DominatedFinMeasAdditive μ T'' C'')
    (h_add : ∀ s, MeasurableSet s → μ s < ∞ → T'' s = T s + T' s) (f : α →₁[μ] E) :
    setToL1 hT'' f = setToL1 hT f + setToL1 hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' T'' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' C'' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    hT'' : MeasureTheory.DominatedFinMeasAdditive μ T'' C''
    h_add : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T'' s) (HAd …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT'') f) (HAdd.hAdd ((MeasureTheory.L1.setToL1 …
  -/
  suffices setToL1 hT'' = setToL1 hT + setToL1 hT' by rw [this, ContinuousLinearMap.add_apply]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' T'' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' C'' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    hT'' : MeasureTheory.DominatedFinMeasAdditive μ T'' C''
    h_add : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T'' s) (HAd …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.setToL1 hT'') (HAdd.hAdd (MeasureTheory.L1.setToL1 hT)  …
  -/
  refine ContinuousLinearMap.extend_unique (setToL1SCLM α E μ hT'') _ _ _ _ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' T'' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' C'' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    hT'' : MeasureTheory.DominatedFinMeasAdditive μ T'' C''
    h_add : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T'' s) (HAd …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((HAdd.hAdd (MeasureTheory.L1.setToL1 hT) (MeasureTheory.L1.setToL1 hT')) …
  -/
  ext1 f
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' T'' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' C'' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    hT'' : MeasureTheory.DominatedFinMeasAdditive μ T'' C''
    h_add : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T'' s) (HAd …
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (((HAdd.hAdd (MeasureTheory.L1.setToL1 hT) (MeasureTheory.L1.setToL1 hT') …
  -/
  suffices setToL1 hT f + setToL1 hT' f = setToL1SCLM α E μ hT'' f by rw [← this]; congr
  rw [setToL1_eq_setToL1SCLM, setToL1_eq_setToL1SCLM,
    setToL1SCLM_add_left' hT hT' hT'' h_add]


theorem setToL1_smul_left (hT : DominatedFinMeasAdditive μ T C) (c : ℝ) (f : α →₁[μ] E) :
    setToL1 (hT.smul c) f = c • setToL1 hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : Real
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 ⋯) f) (HSMul.hSMul c ((MeasureTheory.L1.setToL …
  -/
  suffices setToL1 (hT.smul c) = c • setToL1 hT by rw [this, ContinuousLinearMap.smul_apply]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : Real
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.setToL1 ⋯) (HSMul.hSMul c (MeasureTheory.L1.setToL1 hT))
  -/
  refine ContinuousLinearMap.extend_unique (setToL1SCLM α E μ (hT.smul c)) _ _ _ _ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : Real
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((HSMul.hSMul c (MeasureTheory.L1.setToL1 hT)).comp (MeasureTheory.Lp.sim …
  -/
  ext1 f
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : Real
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (((HSMul.hSMul c (MeasureTheory.L1.setToL1 hT)).comp (MeasureTheory.Lp.si …
  -/
  suffices c • setToL1 hT f = setToL1SCLM α E μ (hT.smul c) f by rw [← this]; congr
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : Real
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (HSMul.hSMul c ((MeasureTheory.L1.setToL1 hT) ↑f)) ((MeasureTheory.L1.Sim …
  -/
  rw [setToL1_eq_setToL1SCLM, setToL1SCLM_smul_left c hT]
  /-
    🎉 no goals
  -/


theorem setToL1_smul_left' (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (c : ℝ)
    (h_smul : ∀ s, MeasurableSet s → μ s < ∞ → T' s = c • T s) (f : α →₁[μ] E) :
    setToL1 hT' f = c • setToL1 hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    c : Real
    h_smul : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T' s) (HSM …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT') f) (HSMul.hSMul c ((MeasureTheory.L1.setT …
  -/
  suffices setToL1 hT' = c • setToL1 hT by rw [this, ContinuousLinearMap.smul_apply]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    c : Real
    h_smul : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T' s) (HSM …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.L1.setToL1 hT') (HSMul.hSMul c (MeasureTheory.L1.setToL1 h …
  -/
  refine ContinuousLinearMap.extend_unique (setToL1SCLM α E μ hT') _ _ _ _ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    c : Real
    h_smul : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T' s) (HSM …
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((HSMul.hSMul c (MeasureTheory.L1.setToL1 hT)).comp (MeasureTheory.Lp.sim …
  -/
  ext1 f
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    c : Real
    h_smul : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T' s) (HSM …
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (((HSMul.hSMul c (MeasureTheory.L1.setToL1 hT)).comp (MeasureTheory.Lp.si …
  -/
  suffices c • setToL1 hT f = setToL1SCLM α E μ hT' f by rw [← this]; congr
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    c : Real
    h_smul : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T' s) (HSM …
    f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
    ⊢ Eq (HSMul.hSMul c ((MeasureTheory.L1.setToL1 hT) ↑f)) ((MeasureTheory.L1.Sim …
  -/
  rw [setToL1_eq_setToL1SCLM, setToL1SCLM_smul_left' c hT hT' h_smul]
  /-
    🎉 no goals
  -/


theorem setToL1_smul (hT : DominatedFinMeasAdditive μ T C)
    (h_smul : ∀ c : 𝕜, ∀ s x, T s (c • x) = c • T s x) (c : 𝕜) (f : α →₁[μ] E) :
    setToL1 hT (c • f) = c • setToL1 hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    𝕜 : Type u_6
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) (HSMul.hSMul c f)) (HSMul.hSMul c ((Measur …
  -/
  rw [setToL1_eq_setToL1' hT h_smul, setToL1_eq_setToL1' hT h_smul]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    𝕜 : Type u_6
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq ((MeasureTheory.L1.setToL1' 𝕜 hT h_smul) (HSMul.hSMul c f)) (HSMul.hSMul  …
  -/
  exact ContinuousLinearMap.map_smul _ _ _
  /-
    🎉 no goals
  -/


theorem setToL1_simpleFunc_indicatorConst (hT : DominatedFinMeasAdditive μ T C) {s : Set α}
    (hs : MeasurableSet s) (hμs : μ s < ∞) (x : E) :
    setToL1 hT (simpleFunc.indicatorConst 1 hs hμs.ne x) = T s x := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    x : E
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) ↑(MeasureTheory.Lp.simpleFunc.indicatorCon …
  -/
  rw [setToL1_eq_setToL1SCLM]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    x : E
    ⊢ Eq ((MeasureTheory.L1.SimpleFunc.setToL1SCLM α E μ hT) (MeasureTheory.Lp.sim …
  -/
  exact setToL1S_indicatorConst (fun s => hT.eq_zero_of_measure_zero) hT.1 hs hμs x
  /-
    🎉 no goals
  -/


theorem setToL1_indicatorConstLp (hT : DominatedFinMeasAdditive μ T C) {s : Set α}
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : E) :
    setToL1 hT (indicatorConstLp 1 hs hμs x) = T s x := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) (MeasureTheory.indicatorConstLp 1 hs hμs x …
  -/
  rw [← Lp.simpleFunc.coe_indicatorConst hs hμs x]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) ↑(MeasureTheory.Lp.simpleFunc.indicatorCon …
  -/
  exact setToL1_simpleFunc_indicatorConst hT hs hμs.lt_top x
  /-
    🎉 no goals
  -/


theorem setToL1_const [IsFiniteMeasure μ] (hT : DominatedFinMeasAdditive μ T C) (x : E) :
    setToL1 hT (indicatorConstLp 1 MeasurableSet.univ (measure_ne_top _ _) x) = T univ x :=
  setToL1_indicatorConstLp hT MeasurableSet.univ (measure_ne_top _ _) x


theorem setToL1_mono_left' {T T' : Set α → E →L[ℝ] G''} {C C' : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C')
    (hTT' : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, T s x ≤ T' s x) (f : α →₁[μ] E) :
    setToL1 hT f ≤ setToL1 hT' f := by
  induction f using Lp.induction (hp_ne_top := one_ne_top) with
  | @h_ind c s hs hμs =>
    rw [setToL1_simpleFunc_indicatorConst hT hs hμs, setToL1_simpleFunc_indicatorConst hT' hs hμs]
    exact hTT' s hs hμs c
  | @h_add f g hf hg _ hf_le hg_le =>
    rw [(setToL1 hT).map_add, (setToL1 hT').map_add]
    exact add_le_add hf_le hg_le
  | h_closed => exact isClosed_le (setToL1 hT).continuous (setToL1 hT').continuous


theorem setToL1_mono_left {T T' : Set α → E →L[ℝ] G''} {C C' : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C')
    (hTT' : ∀ s x, T s x ≤ T' s x) (f : α →₁[μ] E) : setToL1 hT f ≤ setToL1 hT' f :=
  setToL1_mono_left' hT hT' (fun s _ _ x => hTT' s x) f


theorem setToL1_nonneg {T : Set α → G' →L[ℝ] G''} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f : α →₁[μ] G'}
    (hf : 0 ≤ f) : 0 ≤ setToL1 hT f := by
  suffices ∀ f : { g : α →₁[μ] G' // 0 ≤ g }, 0 ≤ setToL1 hT f from
    this (⟨f, hf⟩ : { g : α →₁[μ] G' // 0 ≤ g })
  refine fun g =>
    @isClosed_property { g : α →₁ₛ[μ] G' // 0 ≤ g } { g : α →₁[μ] G' // 0 ≤ g } _ _
      (fun g => 0 ≤ setToL1 hT g)
      (denseRange_coeSimpleFuncNonnegToLpNonneg 1 μ G' one_ne_top) ?_ ?_ g
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp G' 1 μ) x
      hf : LE.le 0 f
      g : Subtype fun g => LE.le 0 g
      ⊢ IsClosed (setOf fun x => (fun g => LE.le 0 ((MeasureTheory.L1.setToL1 hT) ↑g …
    -/
  · exact isClosed_le continuous_zero ((setToL1 hT).continuous.comp continuous_induced_dom)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp G' 1 μ) x
      hf : LE.le 0 f
      g : Subtype fun g => LE.le 0 g
      ⊢ ∀ (a : Subtype fun g => LE.le 0 g), (fun g => LE.le 0 ((MeasureTheory.L1.set …
    -/
  · intro g
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp G' 1 μ) x
      hf : LE.le 0 f
      g✝ : Subtype fun g => LE.le 0 g
      g : Subtype fun g => LE.le 0 g
      ⊢ LE.le 0 ((MeasureTheory.L1.setToL1 hT) ↑(MeasureTheory.Lp.simpleFunc.coeSimp …
    -/
    have : (coeSimpleFuncNonnegToLpNonneg 1 μ G' g : α →₁[μ] G') = (g : α →₁ₛ[μ] G') := rfl
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp G' 1 μ) x
      hf : LE.le 0 f
      g✝ : Subtype fun g => LE.le 0 g
      g : Subtype fun g => LE.le 0 g
      this : Eq ↑(MeasureTheory.Lp.simpleFunc.coeSimpleFuncNonnegToLpNonneg 1 μ G' g …
      ⊢ LE.le 0 ((MeasureTheory.L1.setToL1 hT) ↑(MeasureTheory.Lp.simpleFunc.coeSimp …
    -/
    rw [this, setToL1_eq_setToL1SCLM]
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp G' 1 μ) x
      hf : LE.le 0 f
      g✝ : Subtype fun g => LE.le 0 g
      g : Subtype fun g => LE.le 0 g
      this : Eq ↑(MeasureTheory.Lp.simpleFunc.coeSimpleFuncNonnegToLpNonneg 1 μ G' g …
      ⊢ LE.le 0 ((MeasureTheory.L1.SimpleFunc.setToL1SCLM α G' μ hT) ↑g)
    -/
    exact setToL1S_nonneg (fun s => hT.eq_zero_of_measure_zero) hT.1 hT_nonneg g.2
    /-
      🎉 no goals
    -/


theorem setToL1_mono {T : Set α → G' →L[ℝ] G''} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f g : α →₁[μ] G'}
    (hfg : f ≤ g) : setToL1 hT f ≤ setToL1 hT g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝⁴ : NormedLatticeAddCommGroup G''
    inst✝³ : NormedSpace Real G''
    inst✝² : CompleteSpace G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp G' 1 μ) x
    hfg : LE.le f g
    ⊢ LE.le ((MeasureTheory.L1.setToL1 hT) f) ((MeasureTheory.L1.setToL1 hT) g)
  -/
  rw [← sub_nonneg] at hfg ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝⁴ : NormedLatticeAddCommGroup G''
    inst✝³ : NormedSpace Real G''
    inst✝² : CompleteSpace G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp G' 1 μ) x
    hfg : LE.le 0 (HSub.hSub g f)
    ⊢ LE.le 0 (HSub.hSub ((MeasureTheory.L1.setToL1 hT) g) ((MeasureTheory.L1.setT …
  -/
  rw [← (setToL1 hT).map_sub]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝⁴ : NormedLatticeAddCommGroup G''
    inst✝³ : NormedSpace Real G''
    inst✝² : CompleteSpace G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp G' 1 μ) x
    hfg : LE.le 0 (HSub.hSub g f)
    ⊢ LE.le 0 ((MeasureTheory.L1.setToL1 hT) (HSub.hSub g f))
  -/
  exact setToL1_nonneg hT hT_nonneg hfg
  /-
    🎉 no goals
  -/


theorem norm_setToL1_le_norm_setToL1SCLM (hT : DominatedFinMeasAdditive μ T C) :
    ‖setToL1 hT‖ ≤ ‖setToL1SCLM α E μ hT‖ :=
  calc
    ‖setToL1 hT‖ ≤ (1 : ℝ≥0) * ‖setToL1SCLM α E μ hT‖ := by
      refine
        ContinuousLinearMap.opNorm_extend_le (setToL1SCLM α E μ hT) (coeToLp α E ℝ)
          (simpleFunc.denseRange one_ne_top) fun x => le_of_eq ?_
      /-
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : CompleteSpace F
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        C : Real
        hT : MeasureTheory.DominatedFinMeasAdditive μ T C
        x : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
        ⊢ Eq (Norm.norm x) (HMul.hMul (↑1) (Norm.norm ((MeasureTheory.Lp.simpleFunc.co …
      -/
      rw [NNReal.coe_one, one_mul]
      /-
        α : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : CompleteSpace F
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        C : Real
        hT : MeasureTheory.DominatedFinMeasAdditive μ T C
        x : Subtype fun x => Membership.mem (MeasureTheory.Lp.simpleFunc E 1 μ) x
        ⊢ Eq (Norm.norm x) (Norm.norm ((MeasureTheory.Lp.simpleFunc.coeToLp α E Real)  …
      -/
      rfl
      /-
        🎉 no goals
      -/
                                     /-
                                       α : Type u_1
                                       E : Type u_2
                                       F : Type u_3
                                       inst✝⁴ : NormedAddCommGroup E
                                       inst✝³ : NormedSpace Real E
                                       inst✝² : NormedAddCommGroup F
                                       inst✝¹ : NormedSpace Real F
                                       m : MeasurableSpace α
                                       μ : MeasureTheory.Measure α
                                       inst✝ : CompleteSpace F
                                       T : Set α → ContinuousLinearMap (RingHom.id Real) E F
                                       C : Real
                                       hT : MeasureTheory.DominatedFinMeasAdditive μ T C
                                       ⊢ Eq (HMul.hMul (↑1) (Norm.norm (MeasureTheory.L1.SimpleFunc.setToL1SCLM α E μ …
                                     -/
    _ = ‖setToL1SCLM α E μ hT‖ := by rw [NNReal.coe_one, one_mul]
                                     /-
                                       🎉 no goals
                                     -/


theorem norm_setToL1_le_mul_norm (hT : DominatedFinMeasAdditive μ T C) (hC : 0 ≤ C)
    (f : α →₁[μ] E) : ‖setToL1 hT f‖ ≤ C * ‖f‖ :=
  calc
    ‖setToL1 hT f‖ ≤ ‖setToL1SCLM α E μ hT‖ * ‖f‖ :=
      ContinuousLinearMap.le_of_opNorm_le _ (norm_setToL1_le_norm_setToL1SCLM hT) _
    _ ≤ C * ‖f‖ := mul_le_mul (norm_setToL1SCLM_le hT hC) le_rfl (norm_nonneg _) hC


theorem norm_setToL1_le_mul_norm' (hT : DominatedFinMeasAdditive μ T C) (f : α →₁[μ] E) :
    ‖setToL1 hT f‖ ≤ max C 0 * ‖f‖ :=
  calc
    ‖setToL1 hT f‖ ≤ ‖setToL1SCLM α E μ hT‖ * ‖f‖ :=
      ContinuousLinearMap.le_of_opNorm_le _ (norm_setToL1_le_norm_setToL1SCLM hT) _
    _ ≤ max C 0 * ‖f‖ :=
      mul_le_mul (norm_setToL1SCLM_le' hT) le_rfl (norm_nonneg _) (le_max_right _ _)


theorem norm_setToL1_le (hT : DominatedFinMeasAdditive μ T C) (hC : 0 ≤ C) : ‖setToL1 hT‖ ≤ C :=
  ContinuousLinearMap.opNorm_le_bound _ hC (norm_setToL1_le_mul_norm hT hC)


theorem norm_setToL1_le' (hT : DominatedFinMeasAdditive μ T C) : ‖setToL1 hT‖ ≤ max C 0 :=
  ContinuousLinearMap.opNorm_le_bound _ (le_max_right _ _) (norm_setToL1_le_mul_norm' hT)


theorem setToL1_lipschitz (hT : DominatedFinMeasAdditive μ T C) :
    LipschitzWith (Real.toNNReal C) (setToL1 hT) :=
  (setToL1 hT).lipschitz.weaken (norm_setToL1_le' hT)


/-- If `fs i → f` in `L1`, then `setToL1 hT (fs i) → setToL1 hT f`. -/
theorem tendsto_setToL1 (hT : DominatedFinMeasAdditive μ T C) (f : α →₁[μ] E) {ι}
    (fs : ι → α →₁[μ] E) {l : Filter ι} (hfs : Tendsto fs l (𝓝 f)) :
    Tendsto (fun i => setToL1 hT (fs i)) l (𝓝 <| setToL1 hT f) :=
  ((setToL1 hT).continuous.tendsto _).comp hfs


open Classical in
/-- Extend `T : Set α → E →L[ℝ] F` to `(α → E) → F` (for integrable functions `α → E`). We set it to
0 if the function is not integrable. -/
def setToFun (hT : DominatedFinMeasAdditive μ T C) (f : α → E) : F :=
  if hf : Integrable f μ then L1.setToL1 hT (hf.toL1 f) else 0


theorem setToFun_eq (hT : DominatedFinMeasAdditive μ T C) (hf : Integrable f μ) :
    setToFun μ T hT f = L1.setToL1 hT (hf.toL1 f) :=
  dif_pos hf


theorem L1.setToFun_eq_setToL1 (hT : DominatedFinMeasAdditive μ T C) (f : α →₁[μ] E) :
    setToFun μ T hT f = L1.setToL1 hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ Eq (MeasureTheory.setToFun μ T hT ↑↑f) ((MeasureTheory.L1.setToL1 hT) f)
  -/
  rw [setToFun_eq hT (L1.integrable_coeFn f), Integrable.toL1_coeFn]
  /-
    🎉 no goals
  -/


theorem setToFun_undef (hT : DominatedFinMeasAdditive μ T C) (hf : ¬Integrable f μ) :
    setToFun μ T hT f = 0 :=
  dif_neg hf


theorem setToFun_non_aEStronglyMeasurable (hT : DominatedFinMeasAdditive μ T C)
    (hf : ¬AEStronglyMeasurable f μ) : setToFun μ T hT f = 0 :=
  setToFun_undef hT (not_and_of_not_left _ hf)


theorem setToFun_congr_left (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (h : T = T') (f : α → E) :
    setToFun μ T hT f = setToFun μ T' hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : Eq T T'
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      h : Eq T T'
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
    -/
  · simp_rw [setToFun_eq _ hf, L1.setToL1_congr_left T T' hT hT' h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      h : Eq T T'
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
    -/
  · simp_rw [setToFun_undef _ hf]
    /-
      🎉 no goals
    -/


theorem setToFun_congr_left' (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (h : ∀ s, MeasurableSet s → μ s < ∞ → T s = T' s)
    (f : α → E) : setToFun μ T hT f = setToFun μ T' hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
    -/
  · simp_rw [setToFun_eq _ hf, L1.setToL1_congr_left' T T' hT hT' h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) (T' s)
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
    -/
  · simp_rw [setToFun_undef _ hf]
    /-
      🎉 no goals
    -/


theorem setToFun_add_left (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (f : α → E) :
    setToFun μ (T + T') (hT.add hT') f = setToFun μ T hT f + setToFun μ T' hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ (HAdd.hAdd T T') ⋯ f) (HAdd.hAdd (MeasureTheory …
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ (HAdd.hAdd T T') ⋯ f) (HAdd.hAdd (MeasureTheory …
    -/
  · simp_rw [setToFun_eq _ hf, L1.setToL1_add_left hT hT']
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ (HAdd.hAdd T T') ⋯ f) (HAdd.hAdd (MeasureTheory …
    -/
  · simp_rw [setToFun_undef _ hf, add_zero]
    /-
      🎉 no goals
    -/


theorem setToFun_add_left' (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (hT'' : DominatedFinMeasAdditive μ T'' C'')
    (h_add : ∀ s, MeasurableSet s → μ s < ∞ → T'' s = T s + T' s) (f : α → E) :
    setToFun μ T'' hT'' f = setToFun μ T hT f + setToFun μ T' hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' T'' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' C'' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    hT'' : MeasureTheory.DominatedFinMeasAdditive μ T'' C''
    h_add : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T'' s) (HAd …
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ T'' hT'' f) (HAdd.hAdd (MeasureTheory.setToFun  …
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' T'' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' C'' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      hT'' : MeasureTheory.DominatedFinMeasAdditive μ T'' C''
      h_add : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T'' s) (HAd …
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ T'' hT'' f) (HAdd.hAdd (MeasureTheory.setToFun  …
    -/
  · simp_rw [setToFun_eq _ hf, L1.setToL1_add_left' hT hT' hT'' h_add]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' T'' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' C'' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      hT'' : MeasureTheory.DominatedFinMeasAdditive μ T'' C''
      h_add : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T'' s) (HAd …
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ T'' hT'' f) (HAdd.hAdd (MeasureTheory.setToFun  …
    -/
  · simp_rw [setToFun_undef _ hf, add_zero]
    /-
      🎉 no goals
    -/


theorem setToFun_smul_left (hT : DominatedFinMeasAdditive μ T C) (c : ℝ) (f : α → E) :
    setToFun μ (fun s => c • T s) (hT.smul c) f = c • setToFun μ T hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    c : Real
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ (fun s => HSMul.hSMul c (T s)) ⋯ f) (HSMul.hSMu …
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      c : Real
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ (fun s => HSMul.hSMul c (T s)) ⋯ f) (HSMul.hSMu …
    -/
  · simp_rw [setToFun_eq _ hf, L1.setToL1_smul_left hT c]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      c : Real
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ (fun s => HSMul.hSMul c (T s)) ⋯ f) (HSMul.hSMu …
    -/
  · simp_rw [setToFun_undef _ hf, smul_zero]
    /-
      🎉 no goals
    -/


theorem setToFun_smul_left' (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ T' C') (c : ℝ)
    (h_smul : ∀ s, MeasurableSet s → μ s < ∞ → T' s = c • T s) (f : α → E) :
    setToFun μ T' hT' f = c • setToFun μ T hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    c : Real
    h_smul : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T' s) (HSM …
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ T' hT' f) (HSMul.hSMul c (MeasureTheory.setToFu …
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      c : Real
      h_smul : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T' s) (HSM …
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ T' hT' f) (HSMul.hSMul c (MeasureTheory.setToFu …
    -/
  · simp_rw [setToFun_eq _ hf, L1.setToL1_smul_left' hT hT' c h_smul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      c : Real
      h_smul : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T' s) (HSM …
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ T' hT' f) (HSMul.hSMul c (MeasureTheory.setToFu …
    -/
  · simp_rw [setToFun_undef _ hf, smul_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem setToFun_zero (hT : DominatedFinMeasAdditive μ T C) : setToFun μ T hT (0 : α → E) = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ⊢ Eq (MeasureTheory.setToFun μ T hT 0) 0
  -/
  erw [setToFun_eq hT (integrable_zero _ _ _), Integrable.toL1_zero, ContinuousLinearMap.map_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem setToFun_zero_left {hT : DominatedFinMeasAdditive μ (0 : Set α → E →L[ℝ] F) C} :
    setToFun μ 0 hT f = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    C : Real
    f : α → E
    hT : MeasureTheory.DominatedFinMeasAdditive μ 0 C
    ⊢ Eq (MeasureTheory.setToFun μ 0 hT f) 0
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      C : Real
      f : α → E
      hT : MeasureTheory.DominatedFinMeasAdditive μ 0 C
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ 0 hT f) 0
    -/
  · rw [setToFun_eq hT hf]; exact L1.setToL1_zero_left hT _
                            /-
                              🎉 no goals
                            -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      C : Real
      f : α → E
      hT : MeasureTheory.DominatedFinMeasAdditive μ 0 C
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ 0 hT f) 0
    -/
  · exact setToFun_undef hT hf
    /-
      🎉 no goals
    -/


theorem setToFun_zero_left' (hT : DominatedFinMeasAdditive μ T C)
    (h_zero : ∀ s, MeasurableSet s → μ s < ∞ → T s = 0) : setToFun μ T hT f = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    f : α → E
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) 0
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      f : α → E
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) 0
    -/
  · rw [setToFun_eq hT hf]; exact L1.setToL1_zero_left' hT h_zero _
                            /-
                              🎉 no goals
                            -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      f : α → E
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      h_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) 0
    -/
  · exact setToFun_undef hT hf
    /-
      🎉 no goals
    -/


theorem setToFun_add (hT : DominatedFinMeasAdditive μ T C) (hf : Integrable f μ)
    (hg : Integrable g μ) : setToFun μ T hT (f + g) = setToFun μ T hT f + setToFun μ T hT g := by
  rw [setToFun_eq hT (hf.add hg), setToFun_eq hT hf, setToFun_eq hT hg, Integrable.toL1_add,
    (L1.setToL1 hT).map_add]


theorem setToFun_finset_sum' (hT : DominatedFinMeasAdditive μ T C) {ι} (s : Finset ι)
    {f : ι → α → E} (hf : ∀ i ∈ s, Integrable (f i) μ) :
    setToFun μ T hT (∑ i ∈ s, f i) = ∑ i ∈ s, setToFun μ T hT (f i) := by
  classical
  revert hf
  refine Finset.induction_on s ?_ ?_
  · intro _
    simp only [setToFun_zero, Finset.sum_empty]
  · intro i s his ih hf
    simp only [his, Finset.sum_insert, not_false_iff]
    rw [setToFun_add hT (hf i (Finset.mem_insert_self i s)) _]
    · rw [ih fun i hi => hf i (Finset.mem_insert_of_mem hi)]
    · convert integrable_finset_sum s fun i hi => hf i (Finset.mem_insert_of_mem hi) with x
      simp


theorem setToFun_finset_sum (hT : DominatedFinMeasAdditive μ T C) {ι} (s : Finset ι) {f : ι → α → E}
    (hf : ∀ i ∈ s, Integrable (f i) μ) :
    (setToFun μ T hT fun a => ∑ i ∈ s, f i a) = ∑ i ∈ s, setToFun μ T hT (f i) := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ι : Type u_7
    s : Finset ι
    f : ι → α → E
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (f i) μ
    ⊢ Eq (MeasureTheory.setToFun μ T hT fun a => s.sum fun i => f i a) (s.sum fun  …
  -/
  convert setToFun_finset_sum' hT s hf with a; simp
                                               /-
                                                 🎉 no goals
                                               -/


theorem setToFun_neg (hT : DominatedFinMeasAdditive μ T C) (f : α → E) :
    setToFun μ T hT (-f) = -setToFun μ T hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ T hT (Neg.neg f)) (Neg.neg (MeasureTheory.setTo …
  -/
  by_cases hf : Integrable f μ
  · rw [setToFun_eq hT hf, setToFun_eq hT hf.neg, Integrable.toL1_neg,
      (L1.setToL1 hT).map_neg]
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ T hT (Neg.neg f)) (Neg.neg (MeasureTheory.setTo …
    -/
  · rw [setToFun_undef hT hf, setToFun_undef hT, neg_zero]
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Not (MeasureTheory.Integrable (Neg.neg f) μ)
    -/
    rwa [← integrable_neg_iff] at hf
    /-
      🎉 no goals
    -/


theorem setToFun_sub (hT : DominatedFinMeasAdditive μ T C) (hf : Integrable f μ)
    (hg : Integrable g μ) : setToFun μ T hT (f - g) = setToFun μ T hT f - setToFun μ T hT g := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    f g : α → E
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (MeasureTheory.setToFun μ T hT (HSub.hSub f g)) (HSub.hSub (MeasureTheory …
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg, setToFun_add hT hf hg.neg, setToFun_neg hT g]
  /-
    🎉 no goals
  -/


theorem setToFun_smul [NontriviallyNormedField 𝕜] [NormedSpace 𝕜 E] [NormedSpace 𝕜 F]
    (hT : DominatedFinMeasAdditive μ T C) (h_smul : ∀ c : 𝕜, ∀ s x, T s (c • x) = c • T s x) (c : 𝕜)
    (f : α → E) : setToFun μ T hT (c • f) = c • setToFun μ T hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    𝕜 : Type u_6
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
    c : 𝕜
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ T hT (HSMul.hSMul c f)) (HSMul.hSMul c (Measure …
  -/
  by_cases hf : Integrable f μ
  · rw [setToFun_eq hT hf, setToFun_eq hT, Integrable.toL1_smul',
      L1.setToL1_smul hT h_smul c _]
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      𝕜 : Type u_6
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
      c : 𝕜
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ T hT (HSMul.hSMul c f)) (HSMul.hSMul c (Measure …
    -/
  · by_cases hr : c = 0
      /-
        case pos
        α : Type u_1
        E : Type u_2
        F : Type u_3
        𝕜 : Type u_6
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝³ : CompleteSpace F
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        C : Real
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : NormedSpace 𝕜 F
        hT : MeasureTheory.DominatedFinMeasAdditive μ T C
        h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
        c : 𝕜
        f : α → E
        hf : Not (MeasureTheory.Integrable f μ)
        hr : Eq c 0
        ⊢ Eq (MeasureTheory.setToFun μ T hT (HSMul.hSMul c f)) (HSMul.hSMul c (Measure …
      -/
    · rw [hr]; simp
               /-
                 🎉 no goals
               -/
      /-
        case neg
        α : Type u_1
        E : Type u_2
        F : Type u_3
        𝕜 : Type u_6
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝³ : CompleteSpace F
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        C : Real
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : NormedSpace 𝕜 F
        hT : MeasureTheory.DominatedFinMeasAdditive μ T C
        h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
        c : 𝕜
        f : α → E
        hf : Not (MeasureTheory.Integrable f μ)
        hr : Not (Eq c 0)
        ⊢ Eq (MeasureTheory.setToFun μ T hT (HSMul.hSMul c f)) (HSMul.hSMul c (Measure …
      -/
    · have hf' : ¬Integrable (c • f) μ := by rwa [integrable_smul_iff hr f]
      /-
        case neg
        α : Type u_1
        E : Type u_2
        F : Type u_3
        𝕜 : Type u_6
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace Real F
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝³ : CompleteSpace F
        T : Set α → ContinuousLinearMap (RingHom.id Real) E F
        C : Real
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : NormedSpace 𝕜 F
        hT : MeasureTheory.DominatedFinMeasAdditive μ T C
        h_smul : ∀ (c : 𝕜) (s : Set α) (x : E), Eq ((T s) (HSMul.hSMul c x)) (HSMul.hS …
        c : 𝕜
        f : α → E
        hf : Not (MeasureTheory.Integrable f μ)
        hr : Not (Eq c 0)
        hf' : Not (MeasureTheory.Integrable (HSMul.hSMul c f) μ)
        ⊢ Eq (MeasureTheory.setToFun μ T hT (HSMul.hSMul c f)) (HSMul.hSMul c (Measure …
      -/
      rw [setToFun_undef hT hf, setToFun_undef hT hf', smul_zero]
      /-
        🎉 no goals
      -/


theorem setToFun_congr_ae (hT : DominatedFinMeasAdditive μ T C) (h : f =ᵐ[μ] g) :
    setToFun μ T hT f = setToFun μ T hT g := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    f g : α → E
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T hT g)
  -/
  by_cases hfi : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      f g : α → E
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hfi : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T hT g)
    -/
  · have hgi : Integrable g μ := hfi.congr h
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      f g : α → E
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hfi : MeasureTheory.Integrable f μ
      hgi : MeasureTheory.Integrable g μ
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T hT g)
    -/
    rw [setToFun_eq hT hfi, setToFun_eq hT hgi, (Integrable.toL1_eq_toL1_iff f g hfi hgi).2 h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      f g : α → E
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T hT g)
    -/
  · have hgi : ¬Integrable g μ := by rw [integrable_congr h] at hfi; exact hfi
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      f g : α → E
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      h : (MeasureTheory.ae μ).EventuallyEq f g
      hfi : Not (MeasureTheory.Integrable f μ)
      hgi : Not (MeasureTheory.Integrable g μ)
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T hT g)
    -/
    rw [setToFun_undef hT hfi, setToFun_undef hT hgi]
    /-
      🎉 no goals
    -/


theorem setToFun_measure_zero (hT : DominatedFinMeasAdditive μ T C) (h : μ = 0) :
    setToFun μ T hT f = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    f : α → E
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h : Eq μ 0
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) 0
  -/
  have : f =ᵐ[μ] 0 := by simp [h, EventuallyEq]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    f : α → E
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    h : Eq μ 0
    this : (MeasureTheory.ae μ).EventuallyEq f 0
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) 0
  -/
  rw [setToFun_congr_ae hT this, setToFun_zero]
  /-
    🎉 no goals
  -/


theorem setToFun_measure_zero' (hT : DominatedFinMeasAdditive μ T C)
    (h : ∀ s, MeasurableSet s → μ s < ∞ → μ s = 0) : setToFun μ T hT f = 0 :=
  setToFun_zero_left' hT fun s hs hμs => hT.eq_zero_of_measure_zero hs (h s hs hμs)


theorem setToFun_toL1 (hT : DominatedFinMeasAdditive μ T C) (hf : Integrable f μ) :
    setToFun μ T hT (hf.toL1 f) = setToFun μ T hT f :=
  setToFun_congr_ae hT hf.coeFn_toL1


theorem setToFun_indicator_const (hT : DominatedFinMeasAdditive μ T C) {s : Set α}
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : E) :
    setToFun μ T hT (s.indicator fun _ => x) = T s x := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    ⊢ Eq (MeasureTheory.setToFun μ T hT (s.indicator fun x_1 => x)) ((T s) x)
  -/
  rw [setToFun_congr_ae hT (@indicatorConstLp_coeFn _ _ _ 1 _ _ _ hs hμs x).symm]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    ⊢ Eq (MeasureTheory.setToFun μ T hT ↑↑(MeasureTheory.indicatorConstLp 1 hs hμs …
  -/
  rw [L1.setToFun_eq_setToL1 hT]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    ⊢ Eq ((MeasureTheory.L1.setToL1 hT) (MeasureTheory.indicatorConstLp 1 hs hμs x …
  -/
  exact L1.setToL1_indicatorConstLp hT hs hμs x
  /-
    🎉 no goals
  -/


theorem setToFun_const [IsFiniteMeasure μ] (hT : DominatedFinMeasAdditive μ T C) (x : E) :
    (setToFun μ T hT fun _ => x) = T univ x := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    x : E
    ⊢ Eq (MeasureTheory.setToFun μ T hT fun x_1 => x) ((T Set.univ) x)
  -/
  have : (fun _ : α => x) = Set.indicator univ fun _ => x := (indicator_univ _).symm
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    x : E
    this : Eq (fun x_1 => x) (Set.univ.indicator fun x_1 => x)
    ⊢ Eq (MeasureTheory.setToFun μ T hT fun x_1 => x) ((T Set.univ) x)
  -/
  rw [this]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    x : E
    this : Eq (fun x_1 => x) (Set.univ.indicator fun x_1 => x)
    ⊢ Eq (MeasureTheory.setToFun μ T hT (Set.univ.indicator fun x_1 => x)) ((T Set …
  -/
  exact setToFun_indicator_const hT MeasurableSet.univ (measure_ne_top _ _) x
  /-
    🎉 no goals
  -/


theorem setToFun_mono_left' {T T' : Set α → E →L[ℝ] G''} {C C' : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C')
    (hTT' : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, T s x ≤ T' s x) (f : α → E) :
    setToFun μ T hT f ≤ setToFun μ T' hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G'' : Type u_8
    inst✝² : NormedLatticeAddCommGroup G''
    inst✝¹ : NormedSpace Real G''
    inst✝ : CompleteSpace G''
    T T' : Set α → ContinuousLinearMap (RingHom.id Real) E G''
    C C' : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
    hTT' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : E), LE.le …
    f : α → E
    ⊢ LE.le (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G'' : Type u_8
      inst✝² : NormedLatticeAddCommGroup G''
      inst✝¹ : NormedSpace Real G''
      inst✝ : CompleteSpace G''
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E G''
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      hTT' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : E), LE.le …
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ LE.le (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
    -/
  · simp_rw [setToFun_eq _ hf]; exact L1.setToL1_mono_left' hT hT' hTT' _
                                /-
                                  🎉 no goals
                                -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G'' : Type u_8
      inst✝² : NormedLatticeAddCommGroup G''
      inst✝¹ : NormedSpace Real G''
      inst✝ : CompleteSpace G''
      T T' : Set α → ContinuousLinearMap (RingHom.id Real) E G''
      C C' : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ T' C'
      hTT' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : E), LE.le …
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ LE.le (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T' hT' f)
    -/
  · simp_rw [setToFun_undef _ hf]; rfl
                                   /-
                                     🎉 no goals
                                   -/


theorem setToFun_mono_left {T T' : Set α → E →L[ℝ] G''} {C C' : ℝ}
    (hT : DominatedFinMeasAdditive μ T C) (hT' : DominatedFinMeasAdditive μ T' C')
    (hTT' : ∀ s x, T s x ≤ T' s x) (f : α →₁[μ] E) : setToFun μ T hT f ≤ setToFun μ T' hT' f :=
  setToFun_mono_left' hT hT' (fun s _ _ x => hTT' s x) f


theorem setToFun_nonneg {T : Set α → G' →L[ℝ] G''} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f : α → G'}
    (hf : 0 ≤ᵐ[μ] f) : 0 ≤ setToFun μ T hT f := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝⁴ : NormedLatticeAddCommGroup G''
    inst✝³ : NormedSpace Real G''
    inst✝² : CompleteSpace G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f : α → G'
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ LE.le 0 (MeasureTheory.setToFun μ T hT f)
  -/
  by_cases hfi : Integrable f μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      ⊢ LE.le 0 (MeasureTheory.setToFun μ T hT f)
    -/
  · simp_rw [setToFun_eq _ hfi]
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      ⊢ LE.le 0 ((MeasureTheory.L1.setToL1 hT) (MeasureTheory.Integrable.toL1 f hfi))
    -/
    refine L1.setToL1_nonneg hT hT_nonneg ?_
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      ⊢ LE.le 0 (MeasureTheory.Integrable.toL1 f hfi)
    -/
    rw [← Lp.coeFn_le]
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑0 ↑↑(MeasureTheory.Integrable.toL1 f hfi)
    -/
    have h0 := Lp.coeFn_zero G' 1 μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      h0 : (MeasureTheory.ae μ).EventuallyEq (↑↑0) 0
      ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑0 ↑↑(MeasureTheory.Integrable.toL1 f hfi)
    -/
    have h := Integrable.coeFn_toL1 hfi
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      h0 : (MeasureTheory.ae μ).EventuallyEq (↑↑0) 0
      h : (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.Integrable.toL1 f hfi) …
      ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑0 ↑↑(MeasureTheory.Integrable.toL1 f hfi)
    -/
    filter_upwards [h0, h, hf] with _ h0a ha hfa
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      h0 : (MeasureTheory.ae μ).EventuallyEq (↑↑0) 0
      h : (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.Integrable.toL1 f hfi) …
      a✝ : α
      h0a : Eq (↑↑0 a✝) (0 a✝)
      ha : Eq (↑↑(MeasureTheory.Integrable.toL1 f hfi) a✝) (f a✝)
      hfa : LE.le (0 a✝) (f a✝)
      ⊢ LE.le (↑↑0 a✝) (↑↑(MeasureTheory.Integrable.toL1 f hfi) a✝)
    -/
    rw [h0a, ha]
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : MeasureTheory.Integrable f μ
      h0 : (MeasureTheory.ae μ).EventuallyEq (↑↑0) 0
      h : (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.Integrable.toL1 f hfi) …
      a✝ : α
      h0a : Eq (↑↑0 a✝) (0 a✝)
      ha : Eq (↑↑(MeasureTheory.Integrable.toL1 f hfi) a✝) (f a✝)
      hfa : LE.le (0 a✝) (f a✝)
      ⊢ LE.le (0 a✝) (f a✝)
    -/
    exact hfa
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      G' : Type u_7
      G'' : Type u_8
      inst✝⁴ : NormedLatticeAddCommGroup G''
      inst✝³ : NormedSpace Real G''
      inst✝² : CompleteSpace G''
      inst✝¹ : NormedLatticeAddCommGroup G'
      inst✝ : NormedSpace Real G'
      T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
      f : α → G'
      hf : (MeasureTheory.ae μ).EventuallyLE 0 f
      hfi : Not (MeasureTheory.Integrable f μ)
      ⊢ LE.le 0 (MeasureTheory.setToFun μ T hT f)
    -/
  · simp_rw [setToFun_undef _ hfi]; rfl
                                    /-
                                      🎉 no goals
                                    -/


theorem setToFun_mono {T : Set α → G' →L[ℝ] G''} {C : ℝ} (hT : DominatedFinMeasAdditive μ T C)
    (hT_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x, 0 ≤ x → 0 ≤ T s x) {f g : α → G'}
    (hf : Integrable f μ) (hg : Integrable g μ) (hfg : f ≤ᵐ[μ] g) :
    setToFun μ T hT f ≤ setToFun μ T hT g := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝⁴ : NormedLatticeAddCommGroup G''
    inst✝³ : NormedSpace Real G''
    inst✝² : CompleteSpace G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : α → G'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ LE.le (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ T hT g)
  -/
  rw [← sub_nonneg, ← setToFun_sub hT hg hf]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝⁴ : NormedLatticeAddCommGroup G''
    inst✝³ : NormedSpace Real G''
    inst✝² : CompleteSpace G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : α → G'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ LE.le 0 (MeasureTheory.setToFun μ T hT (HSub.hSub g f))
  -/
  refine setToFun_nonneg hT hT_nonneg (hfg.mono fun a ha => ?_)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝⁴ : NormedLatticeAddCommGroup G''
    inst✝³ : NormedSpace Real G''
    inst✝² : CompleteSpace G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : α → G'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    a : α
    ha : LE.le (f a) (g a)
    ⊢ LE.le (0 a) (HSub.hSub g f a)
  -/
  rw [Pi.sub_apply, Pi.zero_apply, sub_nonneg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    G' : Type u_7
    G'' : Type u_8
    inst✝⁴ : NormedLatticeAddCommGroup G''
    inst✝³ : NormedSpace Real G''
    inst✝² : CompleteSpace G''
    inst✝¹ : NormedLatticeAddCommGroup G'
    inst✝ : NormedSpace Real G'
    T : Set α → ContinuousLinearMap (RingHom.id Real) G' G''
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : G'), …
    f g : α → G'
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    a : α
    ha : LE.le (f a) (g a)
    ⊢ LE.le (f a) (g a)
  -/
  exact ha
  /-
    🎉 no goals
  -/


@[continuity]
theorem continuous_setToFun (hT : DominatedFinMeasAdditive μ T C) :
    Continuous fun f : α →₁[μ] E => setToFun μ T hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ⊢ Continuous fun f => MeasureTheory.setToFun μ T hT ↑↑f
  -/
  simp_rw [L1.setToFun_eq_setToL1 hT]; exact ContinuousLinearMap.continuous _
                                       /-
                                         🎉 no goals
                                       -/


/-- If `F i → f` in `L1`, then `setToFun μ T hT (F i) → setToFun μ T hT f`. -/
theorem tendsto_setToFun_of_L1 (hT : DominatedFinMeasAdditive μ T C) {ι} (f : α → E)
    (hfi : Integrable f μ) {fs : ι → α → E} {l : Filter ι} (hfsi : ∀ᶠ i in l, Integrable (fs i) μ)
    (hfs : Tendsto (fun i => ∫⁻ x, ‖fs i x - f x‖₊ ∂μ) l (𝓝 0)) :
    Tendsto (fun i => setToFun μ T hT (fs i)) l (𝓝 <| setToFun μ T hT f) := by
  classical
    let f_lp := hfi.toL1 f
    let F_lp i := if hFi : Integrable (fs i) μ then hFi.toL1 (fs i) else 0
    have tendsto_L1 : Tendsto F_lp l (𝓝 f_lp) := by
      rw [Lp.tendsto_Lp_iff_tendsto_ℒp']
      simp_rw [eLpNorm_one_eq_lintegral_nnnorm, Pi.sub_apply]
      refine (tendsto_congr' ?_).mp hfs
      filter_upwards [hfsi] with i hi
      refine lintegral_congr_ae ?_
      filter_upwards [hi.coeFn_toL1, hfi.coeFn_toL1] with x hxi hxf
      simp_rw [F_lp, dif_pos hi, hxi, f_lp, hxf]
    suffices Tendsto (fun i => setToFun μ T hT (F_lp i)) l (𝓝 (setToFun μ T hT f)) by
      refine (tendsto_congr' ?_).mp this
      filter_upwards [hfsi] with i hi
      suffices h_ae_eq : F_lp i =ᵐ[μ] fs i from setToFun_congr_ae hT h_ae_eq
      simp_rw [F_lp, dif_pos hi]
      exact hi.coeFn_toL1
    rw [setToFun_congr_ae hT hfi.coeFn_toL1.symm]
    exact ((continuous_setToFun hT).tendsto f_lp).comp tendsto_L1


theorem tendsto_setToFun_approxOn_of_measurable (hT : DominatedFinMeasAdditive μ T C)
    [MeasurableSpace E] [BorelSpace E] {f : α → E} {s : Set E} [SeparableSpace s]
    (hfi : Integrable f μ) (hfm : Measurable f) (hs : ∀ᵐ x ∂μ, f x ∈ closure s) {y₀ : E}
    (h₀ : y₀ ∈ s) (h₀i : Integrable (fun _ => y₀) μ) :
    Tendsto (fun n => setToFun μ T hT (SimpleFunc.approxOn f hfm s y₀ h₀ n)) atTop
      (𝓝 <| setToFun μ T hT f) :=
  tendsto_setToFun_of_L1 hT _ hfi
    (Eventually.of_forall (SimpleFunc.integrable_approxOn hfm hfi h₀ h₀i))
    (SimpleFunc.tendsto_approxOn_L1_nnnorm hfm _ hs (hfi.sub h₀i).2)


theorem tendsto_setToFun_approxOn_of_measurable_of_range_subset
    (hT : DominatedFinMeasAdditive μ T C) [MeasurableSpace E] [BorelSpace E] {f : α → E}
    (fmeas : Measurable f) (hf : Integrable f μ) (s : Set E) [SeparableSpace s]
    (hs : range f ∪ {0} ⊆ s) :
                                                                                 /-
                                                                                   α : Type u_1
                                                                                   E : Type u_2
                                                                                   F : Type u_3
                                                                                   F' : Type u_4
                                                                                   G : Type u_5
                                                                                   𝕜 : Type u_6
                                                                                   p : ENNReal
                                                                                   inst✝¹⁰ : NormedAddCommGroup E
                                                                                   inst✝⁹ : NormedSpace Real E
                                                                                   inst✝⁸ : NormedAddCommGroup F
                                                                                   inst✝⁷ : NormedSpace Real F
                                                                                   inst✝⁶ : NormedAddCommGroup F'
                                                                                   inst✝⁵ : NormedSpace Real F'
                                                                                   inst✝⁴ : NormedAddCommGroup G
                                                                                   m : MeasurableSpace α
                                                                                   μ : MeasureTheory.Measure α
                                                                                   inst✝³ : CompleteSpace F
                                                                                   T T' T'' : Set α → ContinuousLinearMap (RingHom.id Real) E F
                                                                                   C C' C'' : Real
                                                                                   f✝ g : α → E
                                                                                   hT : MeasureTheory.DominatedFinMeasAdditive μ T C
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
    Tendsto (fun n => setToFun μ T hT (SimpleFunc.approxOn f fmeas s 0 (hs <| by simp) n)) atTop
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
      (𝓝 <| setToFun μ T hT f) := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    f : α → E
    fmeas : Measurable f
    hf : MeasureTheory.Integrable f μ
    s : Set E
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hs : HasSubset.Subset (Union.union (Set.range f) (Singleton.singleton 0)) s
    ⊢ Filter.Tendsto (fun n => MeasureTheory.setToFun μ T hT ⇑(MeasureTheory.Simpl …
  -/
  refine tendsto_setToFun_approxOn_of_measurable hT hf fmeas ?_ _ (integrable_zero _ _ _)
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
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


/-- Auxiliary lemma for `setToFun_congr_measure`: the function sending `f : α →₁[μ] G` to
`f : α →₁[μ'] G` is continuous when `μ' ≤ c' • μ` for `c' ≠ ∞`. -/
theorem continuous_L1_toL1 {μ' : Measure α} (c' : ℝ≥0∞) (hc' : c' ≠ ∞) (hμ'_le : μ' ≤ c' • μ) :
    Continuous fun f : α →₁[μ] G =>
      (Integrable.of_measure_le_smul c' hc' hμ'_le (L1.integrable_coeFn f)).toL1 f := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    ⊢ Continuous fun f => MeasureTheory.Integrable.toL1 ↑↑f ⋯
  -/
  by_cases hc'0 : c' = 0
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝ : NormedAddCommGroup G
      m : MeasurableSpace α
      μ μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hc'0 : Eq c' 0
      ⊢ Continuous fun f => MeasureTheory.Integrable.toL1 ↑↑f ⋯
    -/
  · have hμ'0 : μ' = 0 := by rw [← Measure.nonpos_iff_eq_zero']; refine hμ'_le.trans ?_; simp [hc'0]
    have h_im_zero :
      (fun f : α →₁[μ] G =>
          (Integrable.of_measure_le_smul c' hc' hμ'_le (L1.integrable_coeFn f)).toL1 f) =
        0 := by
      ext1 f; ext1; simp_rw [hμ'0]; simp only [ae_zero, EventuallyEq, eventually_bot]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝ : NormedAddCommGroup G
      m : MeasurableSpace α
      μ μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hc'0 : Eq c' 0
      hμ'0 : Eq μ' 0
      h_im_zero : Eq (fun f => MeasureTheory.Integrable.toL1 ↑↑f ⋯) 0
      ⊢ Continuous fun f => MeasureTheory.Integrable.toL1 ↑↑f ⋯
    -/
    rw [h_im_zero]
    /-
      case pos
      α : Type u_1
      G : Type u_5
      inst✝ : NormedAddCommGroup G
      m : MeasurableSpace α
      μ μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hc'0 : Eq c' 0
      hμ'0 : Eq μ' 0
      h_im_zero : Eq (fun f => MeasureTheory.Integrable.toL1 ↑↑f ⋯) 0
      ⊢ Continuous 0
    -/
    exact continuous_zero
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hc'0 : Not (Eq c' 0)
    ⊢ Continuous fun f => MeasureTheory.Integrable.toL1 ↑↑f ⋯
  -/
  rw [Metric.continuous_iff]
  /-
    case neg
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hc'0 : Not (Eq c' 0)
    ⊢ ∀ (b : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x) (ε : Real …
  -/
  intro f ε hε_pos
  /-
    case neg
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hc'0 : Not (Eq c' 0)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    ε : Real
    hε_pos : GT.gt ε 0
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (a : Subtype fun x => Membership.mem (Mea …
  -/
  use ε / 2 / c'.toReal
  /-
    case h
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hc'0 : Not (Eq c' 0)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    ε : Real
    hε_pos : GT.gt ε 0
    ⊢ And (GT.gt (HDiv.hDiv (HDiv.hDiv ε 2) c'.toReal) 0) (∀ (a : Subtype fun x => …
  -/
  refine ⟨div_pos (half_pos hε_pos) (toReal_pos hc'0 hc'), ?_⟩
  /-
    case h
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hc'0 : Not (Eq c' 0)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    ε : Real
    hε_pos : GT.gt ε 0
    ⊢ ∀ (a : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x), LT.lt (D …
  -/
  intro g hfg
  /-
    case h
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hc'0 : Not (Eq c' 0)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    ε : Real
    hε_pos : GT.gt ε 0
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    hfg : LT.lt (Dist.dist g f) (HDiv.hDiv (HDiv.hDiv ε 2) c'.toReal)
    ⊢ LT.lt (Dist.dist (MeasureTheory.Integrable.toL1 ↑↑g ⋯) (MeasureTheory.Integr …
  -/
  rw [Lp.dist_def] at hfg ⊢
  /-
    case h
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hc'0 : Not (Eq c' 0)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    ε : Real
    hε_pos : GT.gt ε 0
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    hfg : LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑g ↑↑f) 1 μ).toReal (HDiv.hDiv  …
    ⊢ LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑(MeasureTheory.Integrable.toL1 ↑↑g …
  -/
  let h_int := fun f' : α →₁[μ] G => (L1.integrable_coeFn f').of_measure_le_smul c' hc' hμ'_le
  have :
    eLpNorm (⇑(Integrable.toL1 g (h_int g)) - ⇑(Integrable.toL1 f (h_int f))) 1 μ' =
      eLpNorm (⇑g - ⇑f) 1 μ' :=
    eLpNorm_congr_ae ((Integrable.coeFn_toL1 _).sub (Integrable.coeFn_toL1 _))
  /-
    case h
    α : Type u_1
    G : Type u_5
    inst✝ : NormedAddCommGroup G
    m : MeasurableSpace α
    μ μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hc'0 : Not (Eq c' 0)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    ε : Real
    hε_pos : GT.gt ε 0
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x
    hfg : LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑g ↑↑f) 1 μ).toReal (HDiv.hDiv  …
    h_int : ∀ (f' : Subtype fun x => Membership.mem (MeasureTheory.Lp G 1 μ) x), M …
    this : Eq (MeasureTheory.eLpNorm (HSub.hSub ↑↑(MeasureTheory.Integrable.toL1 ↑ …
    ⊢ LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑(MeasureTheory.Integrable.toL1 ↑↑g …
  -/
  rw [this]
  have h_eLpNorm_ne_top : eLpNorm (⇑g - ⇑f) 1 μ ≠ ∞ := by
    rw [← eLpNorm_congr_ae (Lp.coeFn_sub _ _)]; exact Lp.eLpNorm_ne_top _
  calc
    (eLpNorm (⇑g - ⇑f) 1 μ').toReal ≤ (c' * eLpNorm (⇑g - ⇑f) 1 μ).toReal := by
      refine toReal_mono (ENNReal.mul_ne_top hc' h_eLpNorm_ne_top) ?_
      refine (eLpNorm_mono_measure (⇑g - ⇑f) hμ'_le).trans_eq ?_
      rw [eLpNorm_smul_measure_of_ne_zero hc'0, smul_eq_mul]
      simp
    _ = c'.toReal * (eLpNorm (⇑g - ⇑f) 1 μ).toReal := toReal_mul
    _ ≤ c'.toReal * (ε / 2 / c'.toReal) := by gcongr
    _ = ε / 2 := by
      refine mul_div_cancel₀ (ε / 2) ?_; rw [Ne, toReal_eq_zero_iff]; simp [hc', hc'0]
    _ < ε := half_lt_self hε_pos


theorem setToFun_congr_measure_of_integrable {μ' : Measure α} (c' : ℝ≥0∞) (hc' : c' ≠ ∞)
    (hμ'_le : μ' ≤ c' • μ) (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ' T C') (f : α → E) (hfμ : Integrable f μ) :
    setToFun μ T hT f = setToFun μ' T hT' f := by
  -- integrability for `μ` implies integrability for `μ'`.
  have h_int : ∀ g : α → E, Integrable g μ → Integrable g μ' := fun g hg =>
    Integrable.of_measure_le_smul c' hc' hμ'_le hg
  -- We use `Integrable.induction`
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    c' : ENNReal
    hc' : Ne c' Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
    f : α → E
    hfμ : MeasureTheory.Integrable f μ
    h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ' T hT' f)
  -/
  apply hfμ.induction (P := fun f => setToFun μ T hT f = setToFun μ' T hT' f)
    /-
      case h_ind
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      ⊢ ∀ (c : E) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → (fun f => Eq  …
    -/
  · intro c s hs hμs
    have hμ's : μ' s ≠ ∞ := by
      refine ((hμ'_le s).trans_lt ?_).ne
      rw [Measure.smul_apply, smul_eq_mul]
      exact ENNReal.mul_lt_top hc'.lt_top hμs
    /-
      case h_ind
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      c : E
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      hμ's : Ne (μ' s) Top.top
      ⊢ Eq (MeasureTheory.setToFun μ T hT (s.indicator fun x => c)) (MeasureTheory.s …
    -/
    rw [setToFun_indicator_const hT hs hμs.ne, setToFun_indicator_const hT' hs hμ's]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      ⊢ ∀ ⦃f g : α → E⦄, Disjoint (Function.support f) (Function.support g) → Measur …
    -/
  · intro f₂ g₂ _ hf₂ hg₂ h_eq_f h_eq_g
    /-
      case h_add
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      f₂ g₂ : α → E
      a✝ : Disjoint (Function.support f₂) (Function.support g₂)
      hf₂ : MeasureTheory.Integrable f₂ μ
      hg₂ : MeasureTheory.Integrable g₂ μ
      h_eq_f : Eq (MeasureTheory.setToFun μ T hT f₂) (MeasureTheory.setToFun μ' T hT …
      h_eq_g : Eq (MeasureTheory.setToFun μ T hT g₂) (MeasureTheory.setToFun μ' T hT …
      ⊢ Eq (MeasureTheory.setToFun μ T hT (HAdd.hAdd f₂ g₂)) (MeasureTheory.setToFun …
    -/
    rw [setToFun_add hT hf₂ hg₂, setToFun_add hT' (h_int f₂ hf₂) (h_int g₂ hg₂), h_eq_f, h_eq_g]
    /-
      🎉 no goals
    -/
    /-
      case h_closed
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      ⊢ IsClosed (setOf fun f => (fun f => Eq (MeasureTheory.setToFun μ T hT f) (Mea …
    -/
  · refine isClosed_eq (continuous_setToFun hT) ?_
    have :
      (fun f : α →₁[μ] E => setToFun μ' T hT' f) = fun f : α →₁[μ] E =>
        setToFun μ' T hT' ((h_int f (L1.integrable_coeFn f)).toL1 f) := by
      ext1 f; exact setToFun_congr_ae hT' (Integrable.coeFn_toL1 _).symm
    /-
      case h_closed
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      this : Eq (fun f => MeasureTheory.setToFun μ' T hT' ↑↑f) fun f => MeasureTheor …
      ⊢ Continuous fun f => MeasureTheory.setToFun μ' T hT' ↑↑f
    -/
    rw [this]
    /-
      case h_closed
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      this : Eq (fun f => MeasureTheory.setToFun μ' T hT' ↑↑f) fun f => MeasureTheor …
      ⊢ Continuous fun f => MeasureTheory.setToFun μ' T hT' ↑↑(MeasureTheory.Integra …
    -/
    exact (continuous_setToFun hT').comp (continuous_L1_toL1 c' hc' hμ'_le)
    /-
      🎉 no goals
    -/
    /-
      case h_ae
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      ⊢ ∀ ⦃f g : α → E⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory.Integ …
    -/
  · intro f₂ g₂ hfg _ hf_eq
    /-
      case h_ae
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      f₂ g₂ : α → E
      hfg : (MeasureTheory.ae μ).EventuallyEq f₂ g₂
      a✝ : MeasureTheory.Integrable f₂ μ
      hf_eq : Eq (MeasureTheory.setToFun μ T hT f₂) (MeasureTheory.setToFun μ' T hT' …
      ⊢ Eq (MeasureTheory.setToFun μ T hT g₂) (MeasureTheory.setToFun μ' T hT' g₂)
    -/
    have hfg' : f₂ =ᵐ[μ'] g₂ := (Measure.absolutelyContinuous_of_le_smul hμ'_le).ae_eq hfg
    /-
      case h_ae
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c' : ENNReal
      hc' : Ne c' Top.top
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hfμ : MeasureTheory.Integrable f μ
      h_int : ∀ (g : α → E), MeasureTheory.Integrable g μ → MeasureTheory.Integrable …
      f₂ g₂ : α → E
      hfg : (MeasureTheory.ae μ).EventuallyEq f₂ g₂
      a✝ : MeasureTheory.Integrable f₂ μ
      hf_eq : Eq (MeasureTheory.setToFun μ T hT f₂) (MeasureTheory.setToFun μ' T hT' …
      hfg' : (MeasureTheory.ae μ').EventuallyEq f₂ g₂
      ⊢ Eq (MeasureTheory.setToFun μ T hT g₂) (MeasureTheory.setToFun μ' T hT' g₂)
    -/
    rw [← setToFun_congr_ae hT hfg, hf_eq, setToFun_congr_ae hT' hfg']
    /-
      🎉 no goals
    -/


theorem setToFun_congr_measure {μ' : Measure α} (c c' : ℝ≥0∞) (hc : c ≠ ∞) (hc' : c' ≠ ∞)
    (hμ_le : μ ≤ c • μ') (hμ'_le : μ' ≤ c' • μ) (hT : DominatedFinMeasAdditive μ T C)
    (hT' : DominatedFinMeasAdditive μ' T C') (f : α → E) :
    setToFun μ T hT f = setToFun μ' T hT' f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    c c' : ENNReal
    hc : Ne c Top.top
    hc' : Ne c' Top.top
    hμ_le : LE.le μ (HSMul.hSMul c μ')
    hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ' T hT' f)
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c c' : ENNReal
      hc : Ne c Top.top
      hc' : Ne c' Top.top
      hμ_le : LE.le μ (HSMul.hSMul c μ')
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ' T hT' f)
    -/
  · exact setToFun_congr_measure_of_integrable c' hc' hμ'_le hT hT' f hf
    /-
      🎉 no goals
    -/
  · -- if `f` is not integrable, both `setToFun` are 0.
    have h_int : ∀ g : α → E, ¬Integrable g μ → ¬Integrable g μ' := fun g =>
      mt fun h => h.of_measure_le_smul _ hc hμ_le
    /-
      case neg
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      μ' : MeasureTheory.Measure α
      c c' : ENNReal
      hc : Ne c Top.top
      hc' : Ne c' Top.top
      hμ_le : LE.le μ (HSMul.hSMul c μ')
      hμ'_le : LE.le μ' (HSMul.hSMul c' μ)
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT' : MeasureTheory.DominatedFinMeasAdditive μ' T C'
      f : α → E
      hf : Not (MeasureTheory.Integrable f μ)
      h_int : ∀ (g : α → E), Not (MeasureTheory.Integrable g μ) → Not (MeasureTheory …
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun μ' T hT' f)
    -/
    simp_rw [setToFun_undef _ hf, setToFun_undef _ (h_int f hf)]
    /-
      🎉 no goals
    -/


theorem setToFun_congr_measure_of_add_right {μ' : Measure α}
    (hT_add : DominatedFinMeasAdditive (μ + μ') T C') (hT : DominatedFinMeasAdditive μ T C)
    (f : α → E) (hf : Integrable f (μ + μ')) :
    setToFun (μ + μ') T hT_add f = setToFun μ T hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    hT_add : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ μ') T C'
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    f : α → E
    hf : MeasureTheory.Integrable f (HAdd.hAdd μ μ')
    ⊢ Eq (MeasureTheory.setToFun (HAdd.hAdd μ μ') T hT_add f) (MeasureTheory.setTo …
  -/
  refine setToFun_congr_measure_of_integrable 1 one_ne_top ?_ hT_add hT f hf
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    hT_add : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ μ') T C'
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    f : α → E
    hf : MeasureTheory.Integrable f (HAdd.hAdd μ μ')
    ⊢ LE.le μ (HSMul.hSMul 1 (HAdd.hAdd μ μ'))
  -/
  rw [one_smul]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    hT_add : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ μ') T C'
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    f : α → E
    hf : MeasureTheory.Integrable f (HAdd.hAdd μ μ')
    ⊢ LE.le μ (HAdd.hAdd μ μ')
  -/
  nth_rw 1 [← add_zero μ]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    hT_add : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ μ') T C'
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    f : α → E
    hf : MeasureTheory.Integrable f (HAdd.hAdd μ μ')
    ⊢ LE.le (HAdd.hAdd μ 0) (HAdd.hAdd μ μ')
  -/
  exact add_le_add le_rfl bot_le
  /-
    🎉 no goals
  -/


theorem setToFun_congr_measure_of_add_left {μ' : Measure α}
    (hT_add : DominatedFinMeasAdditive (μ + μ') T C') (hT : DominatedFinMeasAdditive μ' T C)
    (f : α → E) (hf : Integrable f (μ + μ')) :
    setToFun (μ + μ') T hT_add f = setToFun μ' T hT f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    hT_add : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ μ') T C'
    hT : MeasureTheory.DominatedFinMeasAdditive μ' T C
    f : α → E
    hf : MeasureTheory.Integrable f (HAdd.hAdd μ μ')
    ⊢ Eq (MeasureTheory.setToFun (HAdd.hAdd μ μ') T hT_add f) (MeasureTheory.setTo …
  -/
  refine setToFun_congr_measure_of_integrable 1 one_ne_top ?_ hT_add hT f hf
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    hT_add : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ μ') T C'
    hT : MeasureTheory.DominatedFinMeasAdditive μ' T C
    f : α → E
    hf : MeasureTheory.Integrable f (HAdd.hAdd μ μ')
    ⊢ LE.le μ' (HSMul.hSMul 1 (HAdd.hAdd μ μ'))
  -/
  rw [one_smul]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    hT_add : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ μ') T C'
    hT : MeasureTheory.DominatedFinMeasAdditive μ' T C
    f : α → E
    hf : MeasureTheory.Integrable f (HAdd.hAdd μ μ')
    ⊢ LE.le μ' (HAdd.hAdd μ μ')
  -/
  nth_rw 1 [← zero_add μ']
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    μ' : MeasureTheory.Measure α
    hT_add : MeasureTheory.DominatedFinMeasAdditive (HAdd.hAdd μ μ') T C'
    hT : MeasureTheory.DominatedFinMeasAdditive μ' T C
    f : α → E
    hf : MeasureTheory.Integrable f (HAdd.hAdd μ μ')
    ⊢ LE.le (HAdd.hAdd 0 μ') (HAdd.hAdd μ μ')
  -/
  exact add_le_add bot_le le_rfl
  /-
    🎉 no goals
  -/


theorem setToFun_top_smul_measure (hT : DominatedFinMeasAdditive (∞ • μ) T C) (f : α → E) :
    setToFun (∞ • μ) T hT f = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul Top.top μ) T C
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun (HSMul.hSMul Top.top μ) T hT f) 0
  -/
  refine setToFun_measure_zero' hT fun s _ hμs => ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul Top.top μ) T C
    f : α → E
    s : Set α
    x✝ : MeasurableSet s
    hμs : LT.lt ((HSMul.hSMul Top.top μ) s) Top.top
    ⊢ Eq ((HSMul.hSMul Top.top μ) s) 0
  -/
  rw [lt_top_iff_ne_top] at hμs
  simp only [true_and, Measure.smul_apply, ENNReal.mul_eq_top, eq_self_iff_true,
    top_ne_zero, Ne, not_false_iff, not_or, Classical.not_not, smul_eq_mul] at hμs
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul Top.top μ) T C
    f : α → E
    s : Set α
    x✝ : MeasurableSet s
    hμs : And (Not (Eq (μ s) Top.top)) (Eq (μ s) 0)
    ⊢ Eq ((HSMul.hSMul Top.top μ) s) 0
  -/
  simp only [hμs.right, Measure.smul_apply, mul_zero, smul_eq_mul]
  /-
    🎉 no goals
  -/


theorem setToFun_congr_smul_measure (c : ℝ≥0∞) (hc_ne_top : c ≠ ∞)
    (hT : DominatedFinMeasAdditive μ T C) (hT_smul : DominatedFinMeasAdditive (c • μ) T C')
    (f : α → E) : setToFun μ T hT f = setToFun (c • μ) T hT_smul f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_smul : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C'
    f : α → E
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun (HSMul.hSMul c  …
  -/
  by_cases hc0 : c = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      c : ENNReal
      hc_ne_top : Ne c Top.top
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_smul : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C'
      f : α → E
      hc0 : Eq c 0
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun (HSMul.hSMul c  …
    -/
  · simp [hc0] at hT_smul
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      c : ENNReal
      hc_ne_top : Ne c Top.top
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_smul✝ : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C'
      f : α → E
      hc0 : Eq c 0
      hT_smul : MeasureTheory.DominatedFinMeasAdditive 0 T C'
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun (HSMul.hSMul c  …
    -/
    have h : ∀ s, MeasurableSet s → μ s < ∞ → T s = 0 := fun s hs _ => hT_smul.eq_zero hs
    /-
      case pos
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      c : ENNReal
      hc_ne_top : Ne c Top.top
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_smul✝ : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C'
      f : α → E
      hc0 : Eq c 0
      hT_smul : MeasureTheory.DominatedFinMeasAdditive 0 T C'
      h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
      ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun (HSMul.hSMul c  …
    -/
    rw [setToFun_zero_left' _ h, setToFun_measure_zero]
    /-
      case pos.h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      c : ENNReal
      hc_ne_top : Ne c Top.top
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_smul✝ : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C'
      f : α → E
      hc0 : Eq c 0
      hT_smul : MeasureTheory.DominatedFinMeasAdditive 0 T C'
      h : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (T s) 0
      ⊢ Eq (HSMul.hSMul c μ) 0
    -/
    simp [hc0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C C' : Real
    c : ENNReal
    hc_ne_top : Ne c Top.top
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hT_smul : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C'
    f : α → E
    hc0 : Not (Eq c 0)
    ⊢ Eq (MeasureTheory.setToFun μ T hT f) (MeasureTheory.setToFun (HSMul.hSMul c  …
  -/
  refine setToFun_congr_measure c⁻¹ c ?_ hc_ne_top (le_of_eq ?_) le_rfl hT hT_smul f
    /-
      case neg.refine_1
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      c : ENNReal
      hc_ne_top : Ne c Top.top
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_smul : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C'
      f : α → E
      hc0 : Not (Eq c 0)
      ⊢ Ne (Inv.inv c) Top.top
    -/
  · simp [hc0]
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C C' : Real
      c : ENNReal
      hc_ne_top : Ne c Top.top
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      hT_smul : MeasureTheory.DominatedFinMeasAdditive (HSMul.hSMul c μ) T C'
      f : α → E
      hc0 : Not (Eq c 0)
      ⊢ Eq μ (HSMul.hSMul (Inv.inv c) (HSMul.hSMul c μ))
    -/
  · rw [smul_smul, ENNReal.inv_mul_cancel hc0 hc_ne_top, one_smul]
    /-
      🎉 no goals
    -/


theorem norm_setToFun_le_mul_norm (hT : DominatedFinMeasAdditive μ T C) (f : α →₁[μ] E)
    (hC : 0 ≤ C) : ‖setToFun μ T hT f‖ ≤ C * ‖f‖ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    hC : LE.le 0 C
    ⊢ LE.le (Norm.norm (MeasureTheory.setToFun μ T hT ↑↑f)) (HMul.hMul C (Norm.nor …
  -/
  rw [L1.setToFun_eq_setToL1]; exact L1.norm_setToL1_le_mul_norm hT hC f
                               /-
                                 🎉 no goals
                               -/


theorem norm_setToFun_le_mul_norm' (hT : DominatedFinMeasAdditive μ T C) (f : α →₁[μ] E) :
    ‖setToFun μ T hT f‖ ≤ max C 0 * ‖f‖ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x
    ⊢ LE.le (Norm.norm (MeasureTheory.setToFun μ T hT ↑↑f)) (HMul.hMul (Max.max C  …
  -/
  rw [L1.setToFun_eq_setToL1]; exact L1.norm_setToL1_le_mul_norm' hT f
                               /-
                                 🎉 no goals
                               -/


theorem norm_setToFun_le (hT : DominatedFinMeasAdditive μ T C) (hf : Integrable f μ) (hC : 0 ≤ C) :
    ‖setToFun μ T hT f‖ ≤ C * ‖hf.toL1 f‖ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    f : α → E
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hf : MeasureTheory.Integrable f μ
    hC : LE.le 0 C
    ⊢ LE.le (Norm.norm (MeasureTheory.setToFun μ T hT f)) (HMul.hMul C (Norm.norm  …
  -/
  rw [setToFun_eq hT hf]; exact L1.norm_setToL1_le_mul_norm hT hC _
                          /-
                            🎉 no goals
                          -/


theorem norm_setToFun_le' (hT : DominatedFinMeasAdditive μ T C) (hf : Integrable f μ) :
    ‖setToFun μ T hT f‖ ≤ max C 0 * ‖hf.toL1 f‖ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    f : α → E
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    hf : MeasureTheory.Integrable f μ
    ⊢ LE.le (Norm.norm (MeasureTheory.setToFun μ T hT f)) (HMul.hMul (Max.max C 0) …
  -/
  rw [setToFun_eq hT hf]; exact L1.norm_setToL1_le_mul_norm' hT _
                          /-
                            🎉 no goals
                          -/


/-- Lebesgue dominated convergence theorem provides sufficient conditions under which almost
  everywhere convergence of a sequence of functions implies the convergence of their image by
  `setToFun`.
  We could weaken the condition `bound_integrable` to require `HasFiniteIntegral bound μ` instead
  (i.e. not requiring that `bound` is measurable), but in all applications proving integrability
  is easier. -/
theorem tendsto_setToFun_of_dominated_convergence (hT : DominatedFinMeasAdditive μ T C)
    {fs : ℕ → α → E} {f : α → E} (bound : α → ℝ)
    (fs_measurable : ∀ n, AEStronglyMeasurable (fs n) μ) (bound_integrable : Integrable bound μ)
    (h_bound : ∀ n, ∀ᵐ a ∂μ, ‖fs n a‖ ≤ bound a)
    (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => fs n a) atTop (𝓝 (f a))) :
    Tendsto (fun n => setToFun μ T hT (fs n)) atTop (𝓝 <| setToFun μ T hT f) := by
  -- `f` is a.e.-measurable, since it is the a.e.-pointwise limit of a.e.-measurable functions.
  have f_measurable : AEStronglyMeasurable f μ :=
    aestronglyMeasurable_of_tendsto_ae _ fs_measurable h_lim
  -- all functions we consider are integrable
  have fs_int : ∀ n, Integrable (fs n) μ := fun n =>
    bound_integrable.mono' (fs_measurable n) (h_bound _)
  have f_int : Integrable f μ :=
    ⟨f_measurable,
      hasFiniteIntegral_of_dominated_convergence bound_integrable.hasFiniteIntegral h_bound
        h_lim⟩
  -- it suffices to prove the result for the corresponding L1 functions
  suffices
    Tendsto (fun n => L1.setToL1 hT ((fs_int n).toL1 (fs n))) atTop
      (𝓝 (L1.setToL1 hT (f_int.toL1 f))) by
    convert this with n
    · exact setToFun_eq hT (fs_int n)
    · exact setToFun_eq hT f_int
  -- the convergence of setToL1 follows from the convergence of the L1 functions
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.L1.setToL1 hT) (MeasureTheory.Integr …
  -/
  refine L1.tendsto_setToL1 hT _ _ ?_
  -- up to some rewriting, what we need to prove is `h_lim`
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.Integrable.toL1 (fs n) ⋯) Filter.atTo …
  -/
  rw [tendsto_iff_norm_sub_tendsto_zero]
  have lintegral_norm_tendsto_zero :
    Tendsto (fun n => ENNReal.toReal <| ∫⁻ a, ENNReal.ofReal ‖fs n a - f a‖ ∂μ) atTop (𝓝 0) :=
    (tendsto_toReal zero_ne_top).comp
      (tendsto_lintegral_norm_of_dominated_convergence fs_measurable
        bound_integrable.hasFiniteIntegral h_bound h_lim)
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    lintegral_norm_tendsto_zero : Filter.Tendsto (fun n => (MeasureTheory.lintegra …
    ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (MeasureTheory.Integrable.toL1 …
  -/
  convert lintegral_norm_tendsto_zero with n
  /-
    case h.e'_3.h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    lintegral_norm_tendsto_zero : Filter.Tendsto (fun n => (MeasureTheory.lintegra …
    n : Nat
    ⊢ Eq (Norm.norm (HSub.hSub (MeasureTheory.Integrable.toL1 (fs n) ⋯) (MeasureTh …
  -/
  rw [L1.norm_def]
  /-
    case h.e'_3.h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    lintegral_norm_tendsto_zero : Filter.Tendsto (fun n => (MeasureTheory.lintegra …
    n : Nat
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑(HSub.hSub (Measur …
  -/
  congr 1
  /-
    case h.e'_3.h.e_a
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    lintegral_norm_tendsto_zero : Filter.Tendsto (fun n => (MeasureTheory.lintegra …
    n : Nat
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑(HSub.hSub (Measur …
  -/
  refine lintegral_congr_ae ?_
  /-
    case h.e'_3.h.e_a
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    lintegral_norm_tendsto_zero : Filter.Tendsto (fun n => (MeasureTheory.lintegra …
    n : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => ↑(NNNorm.nnnorm (↑↑(HSub.hSub (M …
  -/
  rw [← Integrable.toL1_sub]
  /-
    case h.e'_3.h.e_a
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    lintegral_norm_tendsto_zero : Filter.Tendsto (fun n => (MeasureTheory.lintegra …
    n : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => ↑(NNNorm.nnnorm (↑↑(MeasureTheor …
  -/
  refine ((fs_int n).sub f_int).coeFn_toL1.mono fun x hx => ?_
  /-
    case h.e'_3.h.e_a
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    lintegral_norm_tendsto_zero : Filter.Tendsto (fun n => (MeasureTheory.lintegra …
    n : Nat
    x : α
    hx : Eq (↑↑(MeasureTheory.Integrable.toL1 (HSub.hSub (fs n) f) ⋯) x) (HSub.hSu …
    ⊢ Eq ((fun a => ↑(NNNorm.nnnorm (↑↑(MeasureTheory.Integrable.toL1 (HSub.hSub ( …
  -/
  dsimp only
  /-
    case h.e'_3.h.e_a
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : Nat → α → E
    f : α → E
    bound : α → Real
    fs_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs n a))  …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) Filter.at …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    fs_int : ∀ (n : Nat), MeasureTheory.Integrable (fs n) μ
    f_int : MeasureTheory.Integrable f μ
    lintegral_norm_tendsto_zero : Filter.Tendsto (fun n => (MeasureTheory.lintegra …
    n : Nat
    x : α
    hx : Eq (↑↑(MeasureTheory.Integrable.toL1 (HSub.hSub (fs n) f) ⋯) x) (HSub.hSu …
    ⊢ Eq (↑(NNNorm.nnnorm (↑↑(MeasureTheory.Integrable.toL1 (HSub.hSub (fs n) f) ⋯ …
  -/
  rw [hx, ofReal_norm_eq_coe_nnnorm, Pi.sub_apply]
  /-
    🎉 no goals
  -/


/-- Lebesgue dominated convergence theorem for filters with a countable basis -/
theorem tendsto_setToFun_filter_of_dominated_convergence (hT : DominatedFinMeasAdditive μ T C) {ι}
    {l : Filter ι} [l.IsCountablyGenerated] {fs : ι → α → E} {f : α → E} (bound : α → ℝ)
    (hfs_meas : ∀ᶠ n in l, AEStronglyMeasurable (fs n) μ)
    (h_bound : ∀ᶠ n in l, ∀ᵐ a ∂μ, ‖fs n a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => fs n a) l (𝓝 (f a))) :
    Tendsto (fun n => setToFun μ T hT (fs n)) l (𝓝 <| setToFun μ T hT f) := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ι : Type u_7
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    fs : ι → α → E
    f : α → E
    bound : α → Real
    hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.setToFun μ T hT (fs n)) l (nhds (Meas …
  -/
  rw [tendsto_iff_seq_tendsto]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ι : Type u_7
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    fs : ι → α → E
    f : α → E
    bound : α → Real
    hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
    ⊢ ∀ (x : Nat → ι), Filter.Tendsto x Filter.atTop l → Filter.Tendsto (Function. …
  -/
  intro x xl
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ι : Type u_7
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    fs : ι → α → E
    f : α → E
    bound : α → Real
    hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    ⊢ Filter.Tendsto (Function.comp (fun n => MeasureTheory.setToFun μ T hT (fs n) …
  -/
  have hxl : ∀ s ∈ l, ∃ a, ∀ b ≥ a, x b ∈ s := by rwa [tendsto_atTop'] at xl
  have h :
    { x : ι | (fun n => AEStronglyMeasurable (fs n) μ) x } ∩
        { x : ι | (fun n => ∀ᵐ a ∂μ, ‖fs n a‖ ≤ bound a) x } ∈ l :=
    inter_mem hfs_meas h_bound
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ι : Type u_7
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    fs : ι → α → E
    f : α → E
    bound : α → Real
    hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
    h : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AESt …
    ⊢ Filter.Tendsto (Function.comp (fun n => MeasureTheory.setToFun μ T hT (fs n) …
  -/
  obtain ⟨k, h⟩ := hxl _ h
  /-
    case intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ι : Type u_7
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    fs : ι → α → E
    f : α → E
    bound : α → Real
    hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
    h✝ : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AES …
    k : Nat
    h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
    ⊢ Filter.Tendsto (Function.comp (fun n => MeasureTheory.setToFun μ T hT (fs n) …
  -/
  rw [← tendsto_add_atTop_iff_nat k]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    ι : Type u_7
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    fs : ι → α → E
    f : α → E
    bound : α → Real
    hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
    h✝ : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AES …
    k : Nat
    h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
    ⊢ Filter.Tendsto (fun n => Function.comp (fun n => MeasureTheory.setToFun μ T  …
  -/
  refine tendsto_setToFun_of_dominated_convergence hT bound ?_ bound_integrable ?_ ?_
    /-
      case intro.refine_1
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      ι : Type u_7
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      fs : ι → α → E
      f : α → E
      bound : α → Real
      hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      h✝ : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AES …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fs (x (HAdd.hAdd n k))) μ
    -/
  · exact fun n => (h _ (self_le_add_left _ _)).1
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      ι : Type u_7
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      fs : ι → α → E
      f : α → E
      bound : α → Real
      hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      h✝ : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AES …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (fs (x (HAdd.hAdd  …
    -/
  · exact fun n => (h _ (self_le_add_left _ _)).2
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      ι : Type u_7
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      fs : ι → α → E
      f : α → E
      bound : α → Real
      hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      h✝ : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AES …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => fs (x (HAdd.hAdd n k))  …
    -/
  · filter_upwards [h_lim]
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      ι : Type u_7
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      fs : ι → α → E
      f : α → E
      bound : α → Real
      hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      h✝ : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AES …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ ∀ (a : α), Filter.Tendsto (fun n => fs n a) l (nhds (f a)) → Filter.Tendsto  …
    -/
    refine fun a h_lin => @Tendsto.comp _ _ _ (fun n => x (n + k)) (fun n => fs n a) _ _ _ h_lin ?_
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      ι : Type u_7
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      fs : ι → α → E
      f : α → E
      bound : α → Real
      hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      h✝ : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AES …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      a : α
      h_lin : Filter.Tendsto (fun n => fs n a) l (nhds (f a))
      ⊢ Filter.Tendsto (fun n => x (HAdd.hAdd n k)) Filter.atTop l
    -/
    rw [tendsto_add_atTop_iff_nat]
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      ι : Type u_7
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      fs : ι → α → E
      f : α → E
      bound : α → Real
      hfs_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fs  …
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => fs n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      h✝ : Membership.mem l (Inter.inter (setOf fun x => (fun n => MeasureTheory.AES …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      a : α
      h_lin : Filter.Tendsto (fun n => fs n a) l (nhds (f a))
      ⊢ Filter.Tendsto x Filter.atTop l
    -/
    assumption
    /-
      🎉 no goals
    -/


theorem continuousWithinAt_setToFun_of_dominated (hT : DominatedFinMeasAdditive μ T C)
    {fs : X → α → E} {x₀ : X} {bound : α → ℝ} {s : Set X}
    (hfs_meas : ∀ᶠ x in 𝓝[s] x₀, AEStronglyMeasurable (fs x) μ)
    (h_bound : ∀ᶠ x in 𝓝[s] x₀, ∀ᵐ a ∂μ, ‖fs x a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_cont : ∀ᵐ a ∂μ, ContinuousWithinAt (fun x => fs x a) s x₀) :
    ContinuousWithinAt (fun x => setToFun μ T hT (fs x)) s x₀ :=
  tendsto_setToFun_filter_of_dominated_convergence hT bound ‹_› ‹_› ‹_› ‹_›


theorem continuousAt_setToFun_of_dominated (hT : DominatedFinMeasAdditive μ T C) {fs : X → α → E}
    {x₀ : X} {bound : α → ℝ} (hfs_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (fs x) μ)
    (h_bound : ∀ᶠ x in 𝓝 x₀, ∀ᵐ a ∂μ, ‖fs x a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_cont : ∀ᵐ a ∂μ, ContinuousAt (fun x => fs x a) x₀) :
    ContinuousAt (fun x => setToFun μ T hT (fs x)) x₀ :=
  tendsto_setToFun_filter_of_dominated_convergence hT bound ‹_› ‹_› ‹_› ‹_›


theorem continuousOn_setToFun_of_dominated (hT : DominatedFinMeasAdditive μ T C) {fs : X → α → E}
    {bound : α → ℝ} {s : Set X} (hfs_meas : ∀ x ∈ s, AEStronglyMeasurable (fs x) μ)
    (h_bound : ∀ x ∈ s, ∀ᵐ a ∂μ, ‖fs x a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_cont : ∀ᵐ a ∂μ, ContinuousOn (fun x => fs x a) s) :
    ContinuousOn (fun x => setToFun μ T hT (fs x)) s := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    X : Type u_7
    inst✝¹ : TopologicalSpace X
    inst✝ : FirstCountableTopology X
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : X → α → E
    bound : α → Real
    s : Set X
    hfs_meas : ∀ (x : X), Membership.mem s x → MeasureTheory.AEStronglyMeasurable  …
    h_bound : ∀ (x : X), Membership.mem s x → Filter.Eventually (fun a => LE.le (N …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_cont : Filter.Eventually (fun a => ContinuousOn (fun x => fs x a) s) (Measur …
    ⊢ ContinuousOn (fun x => MeasureTheory.setToFun μ T hT (fs x)) s
  -/
  intro x hx
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : CompleteSpace F
    T : Set α → ContinuousLinearMap (RingHom.id Real) E F
    C : Real
    X : Type u_7
    inst✝¹ : TopologicalSpace X
    inst✝ : FirstCountableTopology X
    hT : MeasureTheory.DominatedFinMeasAdditive μ T C
    fs : X → α → E
    bound : α → Real
    s : Set X
    hfs_meas : ∀ (x : X), Membership.mem s x → MeasureTheory.AEStronglyMeasurable  …
    h_bound : ∀ (x : X), Membership.mem s x → Filter.Eventually (fun a => LE.le (N …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_cont : Filter.Eventually (fun a => ContinuousOn (fun x => fs x a) s) (Measur …
    x : X
    hx : Membership.mem s x
    ⊢ ContinuousWithinAt (fun x => MeasureTheory.setToFun μ T hT (fs x)) s x
  -/
  refine continuousWithinAt_setToFun_of_dominated hT ?_ ?_ bound_integrable ?_
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      X : Type u_7
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      fs : X → α → E
      bound : α → Real
      s : Set X
      hfs_meas : ∀ (x : X), Membership.mem s x → MeasureTheory.AEStronglyMeasurable  …
      h_bound : ∀ (x : X), Membership.mem s x → Filter.Eventually (fun a => LE.le (N …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousOn (fun x => fs x a) s) (Measur …
      x : X
      hx : Membership.mem s x
      ⊢ Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (fs x) μ) (nh …
    -/
  · filter_upwards [self_mem_nhdsWithin] with x hx using hfs_meas x hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      X : Type u_7
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      fs : X → α → E
      bound : α → Real
      s : Set X
      hfs_meas : ∀ (x : X), Membership.mem s x → MeasureTheory.AEStronglyMeasurable  …
      h_bound : ∀ (x : X), Membership.mem s x → Filter.Eventually (fun a => LE.le (N …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousOn (fun x => fs x a) s) (Measur …
      x : X
      hx : Membership.mem s x
      ⊢ Filter.Eventually (fun x => Filter.Eventually (fun a => LE.le (Norm.norm (fs …
    -/
  · filter_upwards [self_mem_nhdsWithin] with x hx using h_bound x hx
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : CompleteSpace F
      T : Set α → ContinuousLinearMap (RingHom.id Real) E F
      C : Real
      X : Type u_7
      inst✝¹ : TopologicalSpace X
      inst✝ : FirstCountableTopology X
      hT : MeasureTheory.DominatedFinMeasAdditive μ T C
      fs : X → α → E
      bound : α → Real
      s : Set X
      hfs_meas : ∀ (x : X), Membership.mem s x → MeasureTheory.AEStronglyMeasurable  …
      h_bound : ∀ (x : X), Membership.mem s x → Filter.Eventually (fun a => LE.le (N …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_cont : Filter.Eventually (fun a => ContinuousOn (fun x => fs x a) s) (Measur …
      x : X
      hx : Membership.mem s x
      ⊢ Filter.Eventually (fun a => ContinuousWithinAt (fun x => fs x a) s x) (Measu …
    -/
  · filter_upwards [h_cont] with a ha using ha x hx
    /-
      🎉 no goals
    -/


theorem continuous_setToFun_of_dominated (hT : DominatedFinMeasAdditive μ T C) {fs : X → α → E}
    {bound : α → ℝ} (hfs_meas : ∀ x, AEStronglyMeasurable (fs x) μ)
    (h_bound : ∀ x, ∀ᵐ a ∂μ, ‖fs x a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_cont : ∀ᵐ a ∂μ, Continuous fun x => fs x a) : Continuous fun x => setToFun μ T hT (fs x) :=
  continuous_iff_continuousAt.mpr fun _ =>
    continuousAt_setToFun_of_dominated hT (Eventually.of_forall hfs_meas)
        (Eventually.of_forall h_bound) ‹_› <|
      h_cont.mono fun _ => Continuous.continuousAt


