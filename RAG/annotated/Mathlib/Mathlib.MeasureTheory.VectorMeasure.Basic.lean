/-- A vector measure on a measurable space `α` is a σ-additive `M`-valued function (for some `M`
an add monoid) such that the empty set and non-measurable sets are mapped to zero. -/
structure VectorMeasure (α : Type*) [MeasurableSpace α] (M : Type*) [AddCommMonoid M]
    [TopologicalSpace M] where
  measureOf' : Set α → M
  empty' : measureOf' ∅ = 0
  not_measurable' ⦃i : Set α⦄ : ¬MeasurableSet i → measureOf' i = 0
  m_iUnion' ⦃f : ℕ → Set α⦄ : (∀ i, MeasurableSet (f i)) → Pairwise (Disjoint on f) →
    HasSum (fun i => measureOf' (f i)) (measureOf' (⋃ i, f i))


/-- A `SignedMeasure` is an `ℝ`-vector measure. -/
abbrev SignedMeasure (α : Type*) [MeasurableSpace α] :=
  VectorMeasure α ℝ


instance instCoeFun : CoeFun (VectorMeasure α M) fun _ => Set α → M :=
  ⟨VectorMeasure.measureOf'⟩


@[simp]
theorem empty (v : VectorMeasure α M) : v ∅ = 0 :=
  v.empty'


theorem not_measurable (v : VectorMeasure α M) {i : Set α} (hi : ¬MeasurableSet i) : v i = 0 :=
  v.not_measurable' hi


theorem m_iUnion (v : VectorMeasure α M) {f : ℕ → Set α} (hf₁ : ∀ i, MeasurableSet (f i))
    (hf₂ : Pairwise (Disjoint on f)) : HasSum (fun i => v (f i)) (v (⋃ i, f i)) :=
  v.m_iUnion' hf₁ hf₂


theorem coe_injective : @Function.Injective (VectorMeasure α M) (Set α → M) (⇑) := fun v w h => by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v w : MeasureTheory.VectorMeasure α M
    h : Eq ↑v ↑w
    ⊢ Eq v w
  -/
  cases v
  /-
    case mk
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    w : MeasureTheory.VectorMeasure α M
    measureOf'✝ : Set α → M
    empty'✝ : Eq (measureOf'✝ EmptyCollection.emptyCollection) 0
    not_measurable'✝ : ∀ ⦃i : Set α⦄, Not (MeasurableSet i) → Eq (measureOf'✝ i) 0
    m_iUnion'✝ : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), MeasurableSet (f i)) → Pairwis …
    h : Eq ↑{ measureOf' := measureOf'✝, empty' := empty'✝, not_measurable' := not …
    ⊢ Eq { measureOf' := measureOf'✝, empty' := empty'✝, not_measurable' := not_me …
  -/
  cases w
  /-
    case mk.mk
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    measureOf'✝¹ : Set α → M
    empty'✝¹ : Eq (measureOf'✝¹ EmptyCollection.emptyCollection) 0
    not_measurable'✝¹ : ∀ ⦃i : Set α⦄, Not (MeasurableSet i) → Eq (measureOf'✝¹ i) 0
    m_iUnion'✝¹ : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), MeasurableSet (f i)) → Pairwi …
    measureOf'✝ : Set α → M
    empty'✝ : Eq (measureOf'✝ EmptyCollection.emptyCollection) 0
    not_measurable'✝ : ∀ ⦃i : Set α⦄, Not (MeasurableSet i) → Eq (measureOf'✝ i) 0
    m_iUnion'✝ : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), MeasurableSet (f i)) → Pairwis …
    h : Eq ↑{ measureOf' := measureOf'✝¹, empty' := empty'✝¹, not_measurable' := n …
    ⊢ Eq { measureOf' := measureOf'✝¹, empty' := empty'✝¹, not_measurable' := not_ …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem ext_iff' (v w : VectorMeasure α M) : v = w ↔ ∀ i : Set α, v i = w i := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v w : MeasureTheory.VectorMeasure α M
    ⊢ Iff (Eq v w) (∀ (i : Set α), Eq (↑v i) (↑w i))
  -/
  rw [← coe_injective.eq_iff, funext_iff]
  /-
    🎉 no goals
  -/


theorem ext_iff (v w : VectorMeasure α M) : v = w ↔ ∀ i : Set α, MeasurableSet i → v i = w i := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v w : MeasureTheory.VectorMeasure α M
    ⊢ Iff (Eq v w) (∀ (i : Set α), MeasurableSet i → Eq (↑v i) (↑w i))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v w : MeasureTheory.VectorMeasure α M
      ⊢ Eq v w → ∀ (i : Set α), MeasurableSet i → Eq (↑v i) (↑w i)
    -/
  · rintro rfl _ _
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      i✝ : Set α
      a✝ : MeasurableSet i✝
      ⊢ Eq (↑v i✝) (↑v i✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v w : MeasureTheory.VectorMeasure α M
      ⊢ (∀ (i : Set α), MeasurableSet i → Eq (↑v i) (↑w i)) → Eq v w
    -/
  · rw [ext_iff']
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v w : MeasureTheory.VectorMeasure α M
      ⊢ (∀ (i : Set α), MeasurableSet i → Eq (↑v i) (↑w i)) → ∀ (i : Set α), Eq (↑v  …
    -/
    intro h i
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v w : MeasureTheory.VectorMeasure α M
      h : ∀ (i : Set α), MeasurableSet i → Eq (↑v i) (↑w i)
      i : Set α
      ⊢ Eq (↑v i) (↑w i)
    -/
    by_cases hi : MeasurableSet i
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_3
        inst✝¹ : AddCommMonoid M
        inst✝ : TopologicalSpace M
        v w : MeasureTheory.VectorMeasure α M
        h : ∀ (i : Set α), MeasurableSet i → Eq (↑v i) (↑w i)
        i : Set α
        hi : MeasurableSet i
        ⊢ Eq (↑v i) (↑w i)
      -/
    · exact h i hi
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_3
        inst✝¹ : AddCommMonoid M
        inst✝ : TopologicalSpace M
        v w : MeasureTheory.VectorMeasure α M
        h : ∀ (i : Set α), MeasurableSet i → Eq (↑v i) (↑w i)
        i : Set α
        hi : Not (MeasurableSet i)
        ⊢ Eq (↑v i) (↑w i)
      -/
    · simp_rw [not_measurable _ hi]
      /-
        🎉 no goals
      -/


@[ext]
theorem ext {s t : VectorMeasure α M} (h : ∀ i : Set α, MeasurableSet i → s i = t i) : s = t :=
  (ext_iff s t).2 h


theorem hasSum_of_disjoint_iUnion (hm : ∀ i, MeasurableSet (f i)) (hd : Pairwise (Disjoint on f)) :
    HasSum (fun i => v (f i)) (v (⋃ i, f i)) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : Countable β
    v : MeasureTheory.VectorMeasure α M
    f : β → Set α
    hm : ∀ (i : β), MeasurableSet (f i)
    hd : Pairwise (Function.onFun Disjoint f)
    ⊢ HasSum (fun i => ↑v (f i)) (↑v (Set.iUnion fun i => f i))
  -/
  rcases Countable.exists_injective_nat β with ⟨e, he⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : Countable β
    v : MeasureTheory.VectorMeasure α M
    f : β → Set α
    hm : ∀ (i : β), MeasurableSet (f i)
    hd : Pairwise (Function.onFun Disjoint f)
    e : β → Nat
    he : Function.Injective e
    ⊢ HasSum (fun i => ↑v (f i)) (↑v (Set.iUnion fun i => f i))
  -/
  rw [← hasSum_extend_zero he]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : Countable β
    v : MeasureTheory.VectorMeasure α M
    f : β → Set α
    hm : ∀ (i : β), MeasurableSet (f i)
    hd : Pairwise (Function.onFun Disjoint f)
    e : β → Nat
    he : Function.Injective e
    ⊢ HasSum (Function.extend e (fun i => ↑v (f i)) 0) (↑v (Set.iUnion fun i => f  …
  -/
  convert m_iUnion v (f := Function.extend e f fun _ ↦ ∅) _ _
    /-
      case h.e'_5.h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : Countable β
      v : MeasureTheory.VectorMeasure α M
      f : β → Set α
      hm : ∀ (i : β), MeasurableSet (f i)
      hd : Pairwise (Function.onFun Disjoint f)
      e : β → Nat
      he : Function.Injective e
      x✝ : Nat
      ⊢ Eq (Function.extend e (fun i => ↑v (f i)) 0 x✝) (↑v (Function.extend e f (fu …
    -/
  · simp only [Pi.zero_def, Function.apply_extend v, Function.comp_def, empty]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6.h.e'_7
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : Countable β
      v : MeasureTheory.VectorMeasure α M
      f : β → Set α
      hm : ∀ (i : β), MeasurableSet (f i)
      hd : Pairwise (Function.onFun Disjoint f)
      e : β → Nat
      he : Function.Injective e
      ⊢ Eq (Set.iUnion fun i => f i) (Set.iUnion fun i => Function.extend e f (fun x …
    -/
  · exact (iSup_extend_bot he _).symm
    /-
      🎉 no goals
    -/
    /-
      case intro.convert_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : Countable β
      v : MeasureTheory.VectorMeasure α M
      f : β → Set α
      hm : ∀ (i : β), MeasurableSet (f i)
      hd : Pairwise (Function.onFun Disjoint f)
      e : β → Nat
      he : Function.Injective e
      ⊢ ∀ (i : Nat), MeasurableSet (Function.extend e f (fun x => EmptyCollection.em …
    -/
  · simp [Function.apply_extend MeasurableSet, Function.comp_def, hm]
    /-
      🎉 no goals
    -/
    /-
      case intro.convert_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : Countable β
      v : MeasureTheory.VectorMeasure α M
      f : β → Set α
      hm : ∀ (i : β), MeasurableSet (f i)
      hd : Pairwise (Function.onFun Disjoint f)
      e : β → Nat
      he : Function.Injective e
      ⊢ Pairwise (Function.onFun Disjoint (Function.extend e f fun x => EmptyCollect …
    -/
  · exact hd.disjoint_extend_bot (he.factorsThrough _)
    /-
      🎉 no goals
    -/


theorem of_disjoint_iUnion (hm : ∀ i, MeasurableSet (f i)) (hd : Pairwise (Disjoint on f)) :
    v (⋃ i, f i) = ∑' i, v (f i) :=
  (hasSum_of_disjoint_iUnion hm hd).tsum_eq.symm


@[deprecated of_disjoint_iUnion (since := "2024-09-15")]
theorem of_disjoint_iUnion_nat (v : VectorMeasure α M) {f : ℕ → Set α}
    (hf₁ : ∀ i, MeasurableSet (f i)) (hf₂ : Pairwise (Disjoint on f)) :
    v (⋃ i, f i) = ∑' i, v (f i) :=
  of_disjoint_iUnion hf₁ hf₂


theorem of_union {A B : Set α} (h : Disjoint A B) (hA : MeasurableSet A) (hB : MeasurableSet B) :
    v (A ∪ B) = v A + v B := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    inst✝ : T2Space M
    A B : Set α
    h : Disjoint A B
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ Eq (↑v (Union.union A B)) (HAdd.hAdd (↑v A) (↑v B))
  -/
  rw [Set.union_eq_iUnion, of_disjoint_iUnion, tsum_fintype, Fintype.sum_bool, cond, cond]
  /-
    case hm
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    inst✝ : T2Space M
    A B : Set α
    h : Disjoint A B
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ ∀ (i : Bool), MeasurableSet (cond i A B)
  -/
  exacts [fun b => Bool.casesOn b hB hA, pairwise_disjoint_on_bool.2 h]
  /-
    🎉 no goals
  -/


theorem of_add_of_diff {A B : Set α} (hA : MeasurableSet A) (hB : MeasurableSet B) (h : A ⊆ B) :
    v A + v (B \ A) = v B := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    inst✝ : T2Space M
    A B : Set α
    hA : MeasurableSet A
    hB : MeasurableSet B
    h : HasSubset.Subset A B
    ⊢ Eq (HAdd.hAdd (↑v A) (↑v (SDiff.sdiff B A))) (↑v B)
  -/
  rw [← of_union (@Set.disjoint_sdiff_right _ A B) hA (hB.diff hA), Set.union_diff_cancel h]
  /-
    🎉 no goals
  -/


theorem of_diff {M : Type*} [AddCommGroup M] [TopologicalSpace M] [T2Space M]
    {v : VectorMeasure α M} {A B : Set α} (hA : MeasurableSet A) (hB : MeasurableSet B)
    (h : A ⊆ B) : v (B \ A) = v B - v A := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : T2Space M
    v : MeasureTheory.VectorMeasure α M
    A B : Set α
    hA : MeasurableSet A
    hB : MeasurableSet B
    h : HasSubset.Subset A B
    ⊢ Eq (↑v (SDiff.sdiff B A)) (HSub.hSub (↑v B) (↑v A))
  -/
  rw [← of_add_of_diff hA hB h, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


theorem of_diff_of_diff_eq_zero {A B : Set α} (hA : MeasurableSet A) (hB : MeasurableSet B)
    (h' : v (B \ A) = 0) : v (A \ B) + v B = v A := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    inst✝ : T2Space M
    A B : Set α
    hA : MeasurableSet A
    hB : MeasurableSet B
    h' : Eq (↑v (SDiff.sdiff B A)) 0
    ⊢ Eq (HAdd.hAdd (↑v (SDiff.sdiff A B)) (↑v B)) (↑v A)
  -/
  symm
  calc
    v A = v (A \ B ∪ A ∩ B) := by simp only [Set.diff_union_inter]
    _ = v (A \ B) + v (A ∩ B) := by
      rw [of_union]
      · rw [disjoint_comm]
        exact Set.disjoint_of_subset_left A.inter_subset_right disjoint_sdiff_self_right
      · exact hA.diff hB
      · exact hA.inter hB
    _ = v (A \ B) + v (A ∩ B ∪ B \ A) := by
      rw [of_union, h', add_zero]
      · exact Set.disjoint_of_subset_left A.inter_subset_left disjoint_sdiff_self_right
      · exact hA.inter hB
      · exact hB.diff hA
    _ = v (A \ B) + v B := by rw [Set.union_comm, Set.inter_comm, Set.diff_union_inter]


theorem of_iUnion_nonneg {M : Type*} [TopologicalSpace M] [OrderedAddCommMonoid M]
    [OrderClosedTopology M] {v : VectorMeasure α M} (hf₁ : ∀ i, MeasurableSet (f i))
    (hf₂ : Pairwise (Disjoint on f)) (hf₃ : ∀ i, 0 ≤ v (f i)) : 0 ≤ v (⋃ i, f i) :=
  (v.of_disjoint_iUnion hf₁ hf₂).symm ▸ tsum_nonneg hf₃


theorem of_iUnion_nonpos {M : Type*} [TopologicalSpace M] [OrderedAddCommMonoid M]
    [OrderClosedTopology M] {v : VectorMeasure α M} (hf₁ : ∀ i, MeasurableSet (f i))
    (hf₂ : Pairwise (Disjoint on f)) (hf₃ : ∀ i, v (f i) ≤ 0) : v (⋃ i, f i) ≤ 0 :=
  (v.of_disjoint_iUnion hf₁ hf₂).symm ▸ tsum_nonpos hf₃


theorem of_nonneg_disjoint_union_eq_zero {s : SignedMeasure α} {A B : Set α} (h : Disjoint A B)
    (hA₁ : MeasurableSet A) (hB₁ : MeasurableSet B) (hA₂ : 0 ≤ s A) (hB₂ : 0 ≤ s B)
    (hAB : s (A ∪ B) = 0) : s A = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    A B : Set α
    h : Disjoint A B
    hA₁ : MeasurableSet A
    hB₁ : MeasurableSet B
    hA₂ : LE.le 0 (↑s A)
    hB₂ : LE.le 0 (↑s B)
    hAB : Eq (↑s (Union.union A B)) 0
    ⊢ Eq (↑s A) 0
  -/
  rw [of_union h hA₁ hB₁] at hAB
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    A B : Set α
    h : Disjoint A B
    hA₁ : MeasurableSet A
    hB₁ : MeasurableSet B
    hA₂ : LE.le 0 (↑s A)
    hB₂ : LE.le 0 (↑s B)
    hAB : Eq (HAdd.hAdd (↑s A) (↑s B)) 0
    ⊢ Eq (↑s A) 0
  -/
  linarith
  /-
    🎉 no goals
  -/


theorem of_nonpos_disjoint_union_eq_zero {s : SignedMeasure α} {A B : Set α} (h : Disjoint A B)
    (hA₁ : MeasurableSet A) (hB₁ : MeasurableSet B) (hA₂ : s A ≤ 0) (hB₂ : s B ≤ 0)
    (hAB : s (A ∪ B) = 0) : s A = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    A B : Set α
    h : Disjoint A B
    hA₁ : MeasurableSet A
    hB₁ : MeasurableSet B
    hA₂ : LE.le (↑s A) 0
    hB₂ : LE.le (↑s B) 0
    hAB : Eq (↑s (Union.union A B)) 0
    ⊢ Eq (↑s A) 0
  -/
  rw [of_union h hA₁ hB₁] at hAB
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    A B : Set α
    h : Disjoint A B
    hA₁ : MeasurableSet A
    hB₁ : MeasurableSet B
    hA₂ : LE.le (↑s A) 0
    hB₂ : LE.le (↑s B) 0
    hAB : Eq (HAdd.hAdd (↑s A) (↑s B)) 0
    ⊢ Eq (↑s A) 0
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- Given a real number `r` and a signed measure `s`, `smul r s` is the signed
measure corresponding to the function `r • s`. -/
def smul (r : R) (v : VectorMeasure α M) : VectorMeasure α M where
  measureOf' := r • ⇑v
               /-
                 α : Type u_1
                 β : Type u_2
                 m : MeasurableSpace α
                 M : Type u_3
                 inst✝⁴ : AddCommMonoid M
                 inst✝³ : TopologicalSpace M
                 R : Type u_4
                 inst✝² : Semiring R
                 inst✝¹ : DistribMulAction R M
                 inst✝ : ContinuousConstSMul R M
                 r : R
                 v : MeasureTheory.VectorMeasure α M
                 ⊢ Eq (HSMul.hSMul r (↑v) EmptyCollection.emptyCollection) 0
               -/
  empty' := by rw [Pi.smul_apply, empty, smul_zero]
               /-
                 🎉 no goals
               -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               m : MeasurableSpace α
                               M : Type u_3
                               inst✝⁴ : AddCommMonoid M
                               inst✝³ : TopologicalSpace M
                               R : Type u_4
                               inst✝² : Semiring R
                               inst✝¹ : DistribMulAction R M
                               inst✝ : ContinuousConstSMul R M
                               r : R
                               v : MeasureTheory.VectorMeasure α M
                               x✝ : Set α
                               hi : Not (MeasurableSet x✝)
                               ⊢ Eq (HSMul.hSMul r (↑v) x✝) 0
                             -/
  not_measurable' _ hi := by rw [Pi.smul_apply, v.not_measurable hi, smul_zero]
                             /-
                               🎉 no goals
                             -/
                            /-
                              α : Type u_1
                              β : Type u_2
                              m : MeasurableSpace α
                              M : Type u_3
                              inst✝⁴ : AddCommMonoid M
                              inst✝³ : TopologicalSpace M
                              R : Type u_4
                              inst✝² : Semiring R
                              inst✝¹ : DistribMulAction R M
                              inst✝ : ContinuousConstSMul R M
                              r : R
                              v : MeasureTheory.VectorMeasure α M
                              x✝ : Nat → Set α
                              hf₁ : ∀ (i : Nat), MeasurableSet (x✝ i)
                              hf₂ : Pairwise (Function.onFun Disjoint x✝)
                              ⊢ HasSum (fun i => HSMul.hSMul r (↑v) (x✝ i)) (HSMul.hSMul r (↑v) (Set.iUnion  …
                            -/
  m_iUnion' _ hf₁ hf₂ := by exact HasSum.const_smul _ (v.m_iUnion hf₁ hf₂)
                            /-
                              🎉 no goals
                            -/


instance instSMul : SMul R (VectorMeasure α M) :=
  ⟨smul⟩


@[simp]
theorem coe_smul (r : R) (v : VectorMeasure α M) : ⇑(r • v) = r • ⇑v := rfl


theorem smul_apply (r : R) (v : VectorMeasure α M) (i : Set α) : (r • v) i = r • v i := rfl


instance instZero : Zero (VectorMeasure α M) :=
  ⟨⟨0, rfl, fun _ _ => rfl, fun _ _ _ => hasSum_zero⟩⟩


instance instInhabited : Inhabited (VectorMeasure α M) :=
  ⟨0⟩


@[simp]
theorem coe_zero : ⇑(0 : VectorMeasure α M) = 0 := rfl


theorem zero_apply (i : Set α) : (0 : VectorMeasure α M) i = 0 := rfl


/-- The sum of two vector measure is a vector measure. -/
def add (v w : VectorMeasure α M) : VectorMeasure α M where
  measureOf' := v + w
               /-
                 α : Type u_1
                 β : Type u_2
                 m : MeasurableSpace α
                 M : Type u_3
                 inst✝² : AddCommMonoid M
                 inst✝¹ : TopologicalSpace M
                 inst✝ : ContinuousAdd M
                 v w : MeasureTheory.VectorMeasure α M
                 ⊢ Eq (HAdd.hAdd (↑v) (↑w) EmptyCollection.emptyCollection) 0
               -/
  empty' := by simp
               /-
                 🎉 no goals
               -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               m : MeasurableSpace α
                               M : Type u_3
                               inst✝² : AddCommMonoid M
                               inst✝¹ : TopologicalSpace M
                               inst✝ : ContinuousAdd M
                               v w : MeasureTheory.VectorMeasure α M
                               x✝ : Set α
                               hi : Not (MeasurableSet x✝)
                               ⊢ Eq (HAdd.hAdd (↑v) (↑w) x✝) 0
                             -/
  not_measurable' _ hi := by rw [Pi.add_apply, v.not_measurable hi, w.not_measurable hi, add_zero]
                             /-
                               🎉 no goals
                             -/
  m_iUnion' _ hf₁ hf₂ := HasSum.add (v.m_iUnion hf₁ hf₂) (w.m_iUnion hf₁ hf₂)


instance instAdd : Add (VectorMeasure α M) :=
  ⟨add⟩


@[simp]
theorem coe_add (v w : VectorMeasure α M) : ⇑(v + w) = v + w := rfl


theorem add_apply (v w : VectorMeasure α M) (i : Set α) : (v + w) i = v i + w i := rfl


instance instAddCommMonoid : AddCommMonoid (VectorMeasure α M) :=
  Function.Injective.addCommMonoid _ coe_injective coe_zero coe_add fun _ _ => coe_smul _ _


/-- `(⇑)` is an `AddMonoidHom`. -/
@[simps]
def coeFnAddMonoidHom : VectorMeasure α M →+ Set α → M where
  toFun := (⇑)
  map_zero' := coe_zero
  map_add' := coe_add


/-- The negative of a vector measure is a vector measure. -/
def neg (v : VectorMeasure α M) : VectorMeasure α M where
  measureOf' := -v
               /-
                 α : Type u_1
                 β : Type u_2
                 m : MeasurableSpace α
                 M : Type u_3
                 inst✝² : AddCommGroup M
                 inst✝¹ : TopologicalSpace M
                 inst✝ : TopologicalAddGroup M
                 v : MeasureTheory.VectorMeasure α M
                 ⊢ Eq (Neg.neg (↑v) EmptyCollection.emptyCollection) 0
               -/
  empty' := by simp
               /-
                 🎉 no goals
               -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               m : MeasurableSpace α
                               M : Type u_3
                               inst✝² : AddCommGroup M
                               inst✝¹ : TopologicalSpace M
                               inst✝ : TopologicalAddGroup M
                               v : MeasureTheory.VectorMeasure α M
                               x✝ : Set α
                               hi : Not (MeasurableSet x✝)
                               ⊢ Eq (Neg.neg (↑v) x✝) 0
                             -/
  not_measurable' _ hi := by rw [Pi.neg_apply, neg_eq_zero, v.not_measurable hi]
                             /-
                               🎉 no goals
                             -/
  m_iUnion' _ hf₁ hf₂ := HasSum.neg <| v.m_iUnion hf₁ hf₂


instance instNeg : Neg (VectorMeasure α M) :=
  ⟨neg⟩


@[simp]
theorem coe_neg (v : VectorMeasure α M) : ⇑(-v) = -v := rfl


theorem neg_apply (v : VectorMeasure α M) (i : Set α) : (-v) i = -v i := rfl


/-- The difference of two vector measure is a vector measure. -/
def sub (v w : VectorMeasure α M) : VectorMeasure α M where
  measureOf' := v - w
               /-
                 α : Type u_1
                 β : Type u_2
                 m : MeasurableSpace α
                 M : Type u_3
                 inst✝² : AddCommGroup M
                 inst✝¹ : TopologicalSpace M
                 inst✝ : TopologicalAddGroup M
                 v w : MeasureTheory.VectorMeasure α M
                 ⊢ Eq (HSub.hSub (↑v) (↑w) EmptyCollection.emptyCollection) 0
               -/
  empty' := by simp
               /-
                 🎉 no goals
               -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               m : MeasurableSpace α
                               M : Type u_3
                               inst✝² : AddCommGroup M
                               inst✝¹ : TopologicalSpace M
                               inst✝ : TopologicalAddGroup M
                               v w : MeasureTheory.VectorMeasure α M
                               x✝ : Set α
                               hi : Not (MeasurableSet x✝)
                               ⊢ Eq (HSub.hSub (↑v) (↑w) x✝) 0
                             -/
  not_measurable' _ hi := by rw [Pi.sub_apply, v.not_measurable hi, w.not_measurable hi, sub_zero]
                             /-
                               🎉 no goals
                             -/
  m_iUnion' _ hf₁ hf₂ := HasSum.sub (v.m_iUnion hf₁ hf₂) (w.m_iUnion hf₁ hf₂)


instance instSub : Sub (VectorMeasure α M) :=
  ⟨sub⟩


@[simp]
theorem coe_sub (v w : VectorMeasure α M) : ⇑(v - w) = v - w := rfl


theorem sub_apply (v w : VectorMeasure α M) (i : Set α) : (v - w) i = v i - w i := rfl


instance instAddCommGroup : AddCommGroup (VectorMeasure α M) :=
  Function.Injective.addCommGroup _ coe_injective coe_zero coe_add coe_neg coe_sub
    (fun _ _ => coe_smul _ _) fun _ _ => coe_smul _ _


instance instDistribMulAction [ContinuousAdd M] : DistribMulAction R (VectorMeasure α M) :=
  Function.Injective.distribMulAction coeFnAddMonoidHom coe_injective coe_smul


instance instModule [ContinuousAdd M] : Module R (VectorMeasure α M) :=
  Function.Injective.module R coeFnAddMonoidHom coe_injective coe_smul


open Classical in
/-- A finite measure coerced into a real function is a signed measure. -/
@[simps]
def toSignedMeasure (μ : Measure α) [hμ : IsFiniteMeasure μ] : SignedMeasure α where
  measureOf' := fun s : Set α => if MeasurableSet s then (μ s).toReal else 0
               /-
                 α : Type u_1
                 β : Type u_2
                 m : MeasurableSpace α
                 μ : MeasureTheory.Measure α
                 hμ : MeasureTheory.IsFiniteMeasure μ
                 ⊢ Eq ((fun s => ite (MeasurableSet s) (μ s).toReal 0) EmptyCollection.emptyCol …
               -/
  empty' := by simp [μ.empty]
               /-
                 🎉 no goals
               -/
  not_measurable' _ hi := if_neg hi
  m_iUnion' f hf₁ hf₂ := by
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hμ : MeasureTheory.IsFiniteMeasure μ
      f : Nat → Set α
      hf₁ : ∀ (i : Nat), MeasurableSet (f i)
      hf₂ : Pairwise (Function.onFun Disjoint f)
      ⊢ HasSum (fun i => (fun s => ite (MeasurableSet s) (μ s).toReal 0) (f i)) ((fu …
    -/
    simp only [*, MeasurableSet.iUnion hf₁, if_true, measure_iUnion hf₂ hf₁]
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hμ : MeasureTheory.IsFiniteMeasure μ
      f : Nat → Set α
      hf₁ : ∀ (i : Nat), MeasurableSet (f i)
      hf₂ : Pairwise (Function.onFun Disjoint f)
      ⊢ HasSum (fun i => (μ (f i)).toReal) (tsum fun i => μ (f i)).toReal
    -/
    rw [ENNReal.tsum_toReal_eq]
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hμ : MeasureTheory.IsFiniteMeasure μ
      f : Nat → Set α
      hf₁ : ∀ (i : Nat), MeasurableSet (f i)
      hf₂ : Pairwise (Function.onFun Disjoint f)
      ⊢ HasSum (fun i => (μ (f i)).toReal) (tsum fun a => (μ (f a)).toReal)
    -/
    exacts [(summable_measure_toReal hf₁ hf₂).hasSum, fun _ ↦ measure_ne_top _ _]
    /-
      🎉 no goals
    -/


theorem toSignedMeasure_apply_measurable {μ : Measure α} [IsFiniteMeasure μ] {i : Set α}
    (hi : MeasurableSet i) : μ.toSignedMeasure i = (μ i).toReal :=
  if_pos hi

-- Without this lemma, `singularPart_neg` in `MeasureTheory.Decomposition.Lebesgue` is
-- extremely slow

theorem toSignedMeasure_congr {μ ν : Measure α} [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (h : μ = ν) : μ.toSignedMeasure = ν.toSignedMeasure := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h : Eq μ ν
    ⊢ Eq μ.toSignedMeasure ν.toSignedMeasure
  -/
  congr
  /-
    🎉 no goals
  -/


theorem toSignedMeasure_eq_toSignedMeasure_iff {μ ν : Measure α} [IsFiniteMeasure μ]
    [IsFiniteMeasure ν] : μ.toSignedMeasure = ν.toSignedMeasure ↔ μ = ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Iff (Eq μ.toSignedMeasure ν.toSignedMeasure) (Eq μ ν)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      h : Eq μ.toSignedMeasure ν.toSignedMeasure
      ⊢ Eq μ ν
    -/
  · ext1 i hi
    /-
      case refine_1.h
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      h : Eq μ.toSignedMeasure ν.toSignedMeasure
      i : Set α
      hi : MeasurableSet i
      ⊢ Eq (μ i) (ν i)
    -/
    have : μ.toSignedMeasure i = ν.toSignedMeasure i := by rw [h]
    rwa [toSignedMeasure_apply_measurable hi, toSignedMeasure_apply_measurable hi,
        ENNReal.toReal_eq_toReal] at this
          /-
            case refine_1.h.ha
            α : Type u_1
            m : MeasurableSpace α
            μ ν : MeasureTheory.Measure α
            inst✝¹ : MeasureTheory.IsFiniteMeasure μ
            inst✝ : MeasureTheory.IsFiniteMeasure ν
            h : Eq μ.toSignedMeasure ν.toSignedMeasure
            i : Set α
            hi : MeasurableSet i
            this : Eq (μ i).toReal (ν i).toReal
            ⊢ Ne (μ i) Top.top
          -/
          /-
            🎉 no goals
          -/
      <;> exact measure_ne_top _ _
          /-
            🎉 no goals
          -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.IsFiniteMeasure μ
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      h : Eq μ ν
      ⊢ Eq μ.toSignedMeasure ν.toSignedMeasure
    -/
  · congr
    /-
      🎉 no goals
    -/


@[simp]
theorem toSignedMeasure_zero : (0 : Measure α).toSignedMeasure = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ⊢ Eq (MeasureTheory.Measure.toSignedMeasure 0) 0
  -/
  ext i
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    i : Set α
    a✝ : MeasurableSet i
    ⊢ Eq (↑(MeasureTheory.Measure.toSignedMeasure 0) i) (↑0 i)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toSignedMeasure_add (μ ν : Measure α) [IsFiniteMeasure μ] [IsFiniteMeasure ν] :
    (μ + ν).toSignedMeasure = μ.toSignedMeasure + ν.toSignedMeasure := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Eq (HAdd.hAdd μ ν).toSignedMeasure (HAdd.hAdd μ.toSignedMeasure ν.toSignedMe …
  -/
  ext i hi
  rw [toSignedMeasure_apply_measurable hi, add_apply,
    ENNReal.toReal_add (ne_of_lt (measure_lt_top _ _)) (ne_of_lt (measure_lt_top _ _)),
    VectorMeasure.add_apply, toSignedMeasure_apply_measurable hi,
    toSignedMeasure_apply_measurable hi]


@[simp]
theorem toSignedMeasure_smul (μ : Measure α) [IsFiniteMeasure μ] (r : ℝ≥0) :
    (r • μ).toSignedMeasure = r • μ.toSignedMeasure := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    r : NNReal
    ⊢ Eq (HSMul.hSMul r μ).toSignedMeasure (HSMul.hSMul r μ.toSignedMeasure)
  -/
  ext i hi
  rw [toSignedMeasure_apply_measurable hi, VectorMeasure.smul_apply,
    toSignedMeasure_apply_measurable hi, coe_smul, Pi.smul_apply, ENNReal.toReal_smul]


open Classical in
/-- A measure is a vector measure over `ℝ≥0∞`. -/
@[simps]
def toENNRealVectorMeasure (μ : Measure α) : VectorMeasure α ℝ≥0∞ where
  measureOf' := fun i : Set α => if MeasurableSet i then μ i else 0
               /-
                 α : Type u_1
                 β : Type u_2
                 m : MeasurableSpace α
                 μ : MeasureTheory.Measure α
                 ⊢ Eq ((fun i => ite (MeasurableSet i) (μ i) 0) EmptyCollection.emptyCollection …
               -/
  empty' := by simp [μ.empty]
               /-
                 🎉 no goals
               -/
  not_measurable' _ hi := if_neg hi
  m_iUnion' _ hf₁ hf₂ := by
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      x✝ : Nat → Set α
      hf₁ : ∀ (i : Nat), MeasurableSet (x✝ i)
      hf₂ : Pairwise (Function.onFun Disjoint x✝)
      ⊢ HasSum (fun i => (fun i => ite (MeasurableSet i) (μ i) 0) (x✝ i)) ((fun i => …
    -/
    simp only
    rw [Summable.hasSum_iff ENNReal.summable, if_pos (MeasurableSet.iUnion hf₁),
      MeasureTheory.measure_iUnion hf₂ hf₁]
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      x✝ : Nat → Set α
      hf₁ : ∀ (i : Nat), MeasurableSet (x✝ i)
      hf₂ : Pairwise (Function.onFun Disjoint x✝)
      ⊢ Eq (tsum fun b => ite (MeasurableSet (x✝ b)) (μ (x✝ b)) 0) (tsum fun i => μ  …
    -/
    exact tsum_congr fun n => if_pos (hf₁ n)
    /-
      🎉 no goals
    -/


theorem toENNRealVectorMeasure_apply_measurable {μ : Measure α} {i : Set α} (hi : MeasurableSet i) :
    μ.toENNRealVectorMeasure i = μ i :=
  if_pos hi


@[simp]
theorem toENNRealVectorMeasure_zero : (0 : Measure α).toENNRealVectorMeasure = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    ⊢ Eq (MeasureTheory.Measure.toENNRealVectorMeasure 0) 0
  -/
  ext i
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    i : Set α
    a✝ : MeasurableSet i
    ⊢ Eq (↑(MeasureTheory.Measure.toENNRealVectorMeasure 0) i) (↑0 i)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toENNRealVectorMeasure_add (μ ν : Measure α) :
    (μ + ν).toENNRealVectorMeasure = μ.toENNRealVectorMeasure + ν.toENNRealVectorMeasure := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ Eq (HAdd.hAdd μ ν).toENNRealVectorMeasure (HAdd.hAdd μ.toENNRealVectorMeasur …
  -/
  refine MeasureTheory.VectorMeasure.ext fun i hi => ?_
  rw [toENNRealVectorMeasure_apply_measurable hi, add_apply, VectorMeasure.add_apply,
    toENNRealVectorMeasure_apply_measurable hi, toENNRealVectorMeasure_apply_measurable hi]


theorem toSignedMeasure_sub_apply {μ ν : Measure α} [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    {i : Set α} (hi : MeasurableSet i) :
    (μ.toSignedMeasure - ν.toSignedMeasure) i = (μ i).toReal - (ν i).toReal := by
  rw [VectorMeasure.sub_apply, toSignedMeasure_apply_measurable hi,
    Measure.toSignedMeasure_apply_measurable hi]


/-- A vector measure over `ℝ≥0∞` is a measure. -/
def ennrealToMeasure {_ : MeasurableSpace α} (v : VectorMeasure α ℝ≥0∞) : Measure α :=
  ofMeasurable (fun s _ => v s) v.empty fun _ hf₁ hf₂ => v.of_disjoint_iUnion hf₁ hf₂


theorem ennrealToMeasure_apply {m : MeasurableSpace α} {v : VectorMeasure α ℝ≥0∞} {s : Set α}
    (hs : MeasurableSet s) : ennrealToMeasure v s = v s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    v : MeasureTheory.VectorMeasure α ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (v.ennrealToMeasure s) (↑v s)
  -/
  rw [ennrealToMeasure, ofMeasurable_apply _ hs]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.MeasureTheory.Measure.toENNRealVectorMeasure_ennrealToMeasure
    (μ : VectorMeasure α ℝ≥0∞) :
    toENNRealVectorMeasure (ennrealToMeasure μ) = μ := ext fun s hs => by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.VectorMeasure α ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (↑μ.ennrealToMeasure.toENNRealVectorMeasure s) (↑μ s)
  -/
  rw [toENNRealVectorMeasure_apply_measurable hs, ennrealToMeasure_apply hs]
  /-
    🎉 no goals
  -/


@[simp]
theorem ennrealToMeasure_toENNRealVectorMeasure (μ : Measure α) :
    ennrealToMeasure (toENNRealVectorMeasure μ) = μ := Measure.ext fun s hs => by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (μ.toENNRealVectorMeasure.ennrealToMeasure s) (μ s)
  -/
  rw [ennrealToMeasure_apply hs, toENNRealVectorMeasure_apply_measurable hs]
  /-
    🎉 no goals
  -/


/-- The equiv between `VectorMeasure α ℝ≥0∞` and `Measure α` formed by
`MeasureTheory.VectorMeasure.ennrealToMeasure` and
`MeasureTheory.Measure.toENNRealVectorMeasure`. -/
@[simps]
def equivMeasure [MeasurableSpace α] : VectorMeasure α ℝ≥0∞ ≃ Measure α where
  toFun := ennrealToMeasure
  invFun := toENNRealVectorMeasure
  left_inv := toENNRealVectorMeasure_ennrealToMeasure
  right_inv := ennrealToMeasure_toENNRealVectorMeasure


open Classical in
/-- The pushforward of a vector measure along a function. -/
def map (v : VectorMeasure α M) (f : α → β) : VectorMeasure β M :=
  if hf : Measurable f then
    { measureOf' := fun s => if MeasurableSet s then v (f ⁻¹' s) else 0
                   /-
                     α : Type u_1
                     β : Type u_2
                     m inst✝³ : MeasurableSpace α
                     inst✝² : MeasurableSpace β
                     M : Type u_3
                     inst✝¹ : AddCommMonoid M
                     inst✝ : TopologicalSpace M
                     v✝ v : MeasureTheory.VectorMeasure α M
                     f : α → β
                     hf : Measurable f
                     ⊢ Eq ((fun s => ite (MeasurableSet s) (↑v (Set.preimage f s)) 0) EmptyCollecti …
                   -/
      empty' := by simp
                   /-
                     🎉 no goals
                   -/
      not_measurable' := fun _ hi => if_neg hi
      m_iUnion' := by
        /-
          α : Type u_1
          β : Type u_2
          m inst✝³ : MeasurableSpace α
          inst✝² : MeasurableSpace β
          M : Type u_3
          inst✝¹ : AddCommMonoid M
          inst✝ : TopologicalSpace M
          v✝ v : MeasureTheory.VectorMeasure α M
          f : α → β
          hf : Measurable f
          ⊢ ∀ ⦃f_1 : Nat → Set β⦄, (∀ (i : Nat), MeasurableSet (f_1 i)) → Pairwise (Func …
        -/
        intro g hg₁ hg₂
        /-
          α : Type u_1
          β : Type u_2
          m inst✝³ : MeasurableSpace α
          inst✝² : MeasurableSpace β
          M : Type u_3
          inst✝¹ : AddCommMonoid M
          inst✝ : TopologicalSpace M
          v✝ v : MeasureTheory.VectorMeasure α M
          f : α → β
          hf : Measurable f
          g : Nat → Set β
          hg₁ : ∀ (i : Nat), MeasurableSet (g i)
          hg₂ : Pairwise (Function.onFun Disjoint g)
          ⊢ HasSum (fun i => (fun s => ite (MeasurableSet s) (↑v (Set.preimage f s)) 0)  …
        -/
        simp only
        /-
          α : Type u_1
          β : Type u_2
          m inst✝³ : MeasurableSpace α
          inst✝² : MeasurableSpace β
          M : Type u_3
          inst✝¹ : AddCommMonoid M
          inst✝ : TopologicalSpace M
          v✝ v : MeasureTheory.VectorMeasure α M
          f : α → β
          hf : Measurable f
          g : Nat → Set β
          hg₁ : ∀ (i : Nat), MeasurableSet (g i)
          hg₂ : Pairwise (Function.onFun Disjoint g)
          ⊢ HasSum (fun i => ite (MeasurableSet (g i)) (↑v (Set.preimage f (g i))) 0) (i …
        -/
        convert v.m_iUnion (fun i => hf (hg₁ i)) fun i j hij => (hg₂ hij).preimage _
          /-
            case h.e'_5.h
            α : Type u_1
            β : Type u_2
            m inst✝³ : MeasurableSpace α
            inst✝² : MeasurableSpace β
            M : Type u_3
            inst✝¹ : AddCommMonoid M
            inst✝ : TopologicalSpace M
            v✝ v : MeasureTheory.VectorMeasure α M
            f : α → β
            hf : Measurable f
            g : Nat → Set β
            hg₁ : ∀ (i : Nat), MeasurableSet (g i)
            hg₂ : Pairwise (Function.onFun Disjoint g)
            x✝ : Nat
            ⊢ Eq (ite (MeasurableSet (g x✝)) (↑v (Set.preimage f (g x✝))) 0) (↑v (Set.prei …
          -/
        · rw [if_pos (hg₁ _)]
          /-
            🎉 no goals
          -/
          /-
            case h.e'_6
            α : Type u_1
            β : Type u_2
            m inst✝³ : MeasurableSpace α
            inst✝² : MeasurableSpace β
            M : Type u_3
            inst✝¹ : AddCommMonoid M
            inst✝ : TopologicalSpace M
            v✝ v : MeasureTheory.VectorMeasure α M
            f : α → β
            hf : Measurable f
            g : Nat → Set β
            hg₁ : ∀ (i : Nat), MeasurableSet (g i)
            hg₂ : Pairwise (Function.onFun Disjoint g)
            ⊢ Eq (ite (MeasurableSet (Set.iUnion fun i => g i)) (↑v (Set.preimage f (Set.i …
          -/
        · rw [Set.preimage_iUnion, if_pos (MeasurableSet.iUnion hg₁)] }
          /-
            🎉 no goals
          -/
  else 0


theorem map_not_measurable {f : α → β} (hf : ¬Measurable f) : v.map f = 0 :=
  dif_neg hf


theorem map_apply {f : α → β} (hf : Measurable f) {s : Set β} (hs : MeasurableSet s) :
    v.map f s = v (f ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (↑(v.map f) s) (↑v (Set.preimage f s))
  -/
  rw [map, dif_pos hf]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    f : α → β
    hf : Measurable f
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (↑{ measureOf' := fun s => ite (MeasurableSet s) (↑v (Set.preimage f s))  …
  -/
  exact if_pos hs
  /-
    🎉 no goals
  -/


@[simp]
theorem map_id : v.map id = v :=
                     /-
                       α : Type u_1
                       inst✝² : MeasurableSpace α
                       M : Type u_3
                       inst✝¹ : AddCommMonoid M
                       inst✝ : TopologicalSpace M
                       v : MeasureTheory.VectorMeasure α M
                       i : Set α
                       hi : MeasurableSet i
                       ⊢ Eq (↑(v.map id) i) (↑v i)
                     -/
  ext fun i hi => by rw [map_apply v measurable_id hi, Set.preimage_id]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem map_zero (f : α → β) : (0 : VectorMeasure α M).map f = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    f : α → β
    ⊢ Eq (MeasureTheory.VectorMeasure.map 0 f) 0
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      f : α → β
      hf : Measurable f
      ⊢ Eq (MeasureTheory.VectorMeasure.map 0 f) 0
    -/
  · ext i hi
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      f : α → β
      hf : Measurable f
      i : Set β
      hi : MeasurableSet i
      ⊢ Eq (↑(MeasureTheory.VectorMeasure.map 0 f) i) (↑0 i)
    -/
    rw [map_apply _ hf hi, zero_apply, zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      f : α → β
      hf : Not (Measurable f)
      ⊢ Eq (MeasureTheory.VectorMeasure.map 0 f) 0
    -/
  · exact dif_neg hf
    /-
      🎉 no goals
    -/


/-- Given a vector measure `v` on `M` and a continuous `AddMonoidHom` `f : M → N`, `f ∘ v` is a
vector measure on `N`. -/
def mapRange (v : VectorMeasure α M) (f : M →+ N) (hf : Continuous f) : VectorMeasure α N where
  measureOf' s := f (v s)
               /-
                 α : Type u_1
                 β : Type u_2
                 m inst✝⁵ : MeasurableSpace α
                 inst✝⁴ : MeasurableSpace β
                 M : Type u_3
                 inst✝³ : AddCommMonoid M
                 inst✝² : TopologicalSpace M
                 v✝ : MeasureTheory.VectorMeasure α M
                 N : Type u_4
                 inst✝¹ : AddCommMonoid N
                 inst✝ : TopologicalSpace N
                 v : MeasureTheory.VectorMeasure α M
                 f : AddMonoidHom M N
                 hf : Continuous ⇑f
                 ⊢ Eq ((fun s => f (↑v s)) EmptyCollection.emptyCollection) 0
               -/
  empty' := by simp only; rw [empty, AddMonoidHom.map_zero]
                          /-
                            🎉 no goals
                          -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               m inst✝⁵ : MeasurableSpace α
                               inst✝⁴ : MeasurableSpace β
                               M : Type u_3
                               inst✝³ : AddCommMonoid M
                               inst✝² : TopologicalSpace M
                               v✝ : MeasureTheory.VectorMeasure α M
                               N : Type u_4
                               inst✝¹ : AddCommMonoid N
                               inst✝ : TopologicalSpace N
                               v : MeasureTheory.VectorMeasure α M
                               f : AddMonoidHom M N
                               hf : Continuous ⇑f
                               i : Set α
                               hi : Not (MeasurableSet i)
                               ⊢ Eq ((fun s => f (↑v s)) i) 0
                             -/
  not_measurable' i hi := by simp only; rw [not_measurable v hi, AddMonoidHom.map_zero]
                                        /-
                                          🎉 no goals
                                        -/
  m_iUnion' _ hg₁ hg₂ := HasSum.map (v.m_iUnion hg₁ hg₂) f hf


@[simp]
theorem mapRange_apply {f : M →+ N} (hf : Continuous f) {s : Set α} : v.mapRange f hf s = f (v s) :=
  rfl


@[simp]
theorem mapRange_id : v.mapRange (AddMonoidHom.id M) continuous_id = v := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    ⊢ Eq (v.mapRange (AddMonoidHom.id M) ⋯) v
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝² : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    i✝ : Set α
    a✝ : MeasurableSet i✝
    ⊢ Eq (↑(v.mapRange (AddMonoidHom.id M) ⋯) i✝) (↑v i✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mapRange_zero {f : M →+ N} (hf : Continuous f) :
    mapRange (0 : VectorMeasure α M) f hf = 0 := by
  /-
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : TopologicalSpace M
    N : Type u_4
    inst✝¹ : AddCommMonoid N
    inst✝ : TopologicalSpace N
    f : AddMonoidHom M N
    hf : Continuous ⇑f
    ⊢ Eq (MeasureTheory.VectorMeasure.mapRange 0 f hf) 0
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝⁴ : MeasurableSpace α
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : TopologicalSpace M
    N : Type u_4
    inst✝¹ : AddCommMonoid N
    inst✝ : TopologicalSpace N
    f : AddMonoidHom M N
    hf : Continuous ⇑f
    i✝ : Set α
    a✝ : MeasurableSet i✝
    ⊢ Eq (↑(MeasureTheory.VectorMeasure.mapRange 0 f hf) i✝) (↑0 i✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem mapRange_add {v w : VectorMeasure α M} {f : M →+ N} (hf : Continuous f) :
    (v + w).mapRange f hf = v.mapRange f hf + w.mapRange f hf := by
  /-
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    M : Type u_3
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : TopologicalSpace M
    N : Type u_4
    inst✝³ : AddCommMonoid N
    inst✝² : TopologicalSpace N
    inst✝¹ : ContinuousAdd M
    inst✝ : ContinuousAdd N
    v w : MeasureTheory.VectorMeasure α M
    f : AddMonoidHom M N
    hf : Continuous ⇑f
    ⊢ Eq ((HAdd.hAdd v w).mapRange f hf) (HAdd.hAdd (v.mapRange f hf) (w.mapRange  …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝⁶ : MeasurableSpace α
    M : Type u_3
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : TopologicalSpace M
    N : Type u_4
    inst✝³ : AddCommMonoid N
    inst✝² : TopologicalSpace N
    inst✝¹ : ContinuousAdd M
    inst✝ : ContinuousAdd N
    v w : MeasureTheory.VectorMeasure α M
    f : AddMonoidHom M N
    hf : Continuous ⇑f
    i✝ : Set α
    a✝ : MeasurableSet i✝
    ⊢ Eq (↑((HAdd.hAdd v w).mapRange f hf) i✝) (↑(HAdd.hAdd (v.mapRange f hf) (w.m …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a continuous `AddMonoidHom` `f : M → N`, `mapRangeHom` is the `AddMonoidHom` mapping the
vector measure `v` on `M` to the vector measure `f ∘ v` on `N`. -/
def mapRangeHom (f : M →+ N) (hf : Continuous f) : VectorMeasure α M →+ VectorMeasure α N where
  toFun v := v.mapRange f hf
  map_zero' := mapRange_zero hf
  map_add' _ _ := mapRange_add hf


/-- Given a continuous linear map `f : M → N`, `mapRangeₗ` is the linear map mapping the
vector measure `v` on `M` to the vector measure `f ∘ v` on `N`. -/
def mapRangeₗ (f : M →ₗ[R] N) (hf : Continuous f) : VectorMeasure α M →ₗ[R] VectorMeasure α N where
  toFun v := v.mapRange f.toAddMonoidHom hf
  map_add' _ _ := mapRange_add hf
  map_smul' := by
    /-
      α : Type u_1
      β : Type u_2
      m inst✝¹² : MeasurableSpace α
      inst✝¹¹ : MeasurableSpace β
      M : Type u_3
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      N : Type u_4
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : TopologicalSpace N
      R : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : ContinuousAdd M
      inst✝² : ContinuousAdd N
      inst✝¹ : ContinuousConstSMul R M
      inst✝ : ContinuousConstSMul R N
      f : LinearMap (RingHom.id R) M N
      hf : Continuous ⇑f
      ⊢ ∀ (m : R) (x : MeasureTheory.VectorMeasure α M), Eq ({ toFun := fun v => v.m …
    -/
    intros
    /-
      α : Type u_1
      β : Type u_2
      m inst✝¹² : MeasurableSpace α
      inst✝¹¹ : MeasurableSpace β
      M : Type u_3
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      N : Type u_4
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : TopologicalSpace N
      R : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : ContinuousAdd M
      inst✝² : ContinuousAdd N
      inst✝¹ : ContinuousConstSMul R M
      inst✝ : ContinuousConstSMul R N
      f : LinearMap (RingHom.id R) M N
      hf : Continuous ⇑f
      m✝ : R
      x✝ : MeasureTheory.VectorMeasure α M
      ⊢ Eq ({ toFun := fun v => v.mapRange f.toAddMonoidHom hf, map_add' := ⋯ }.toFu …
    -/
    ext
    /-
      case h
      α : Type u_1
      β : Type u_2
      m inst✝¹² : MeasurableSpace α
      inst✝¹¹ : MeasurableSpace β
      M : Type u_3
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      N : Type u_4
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : TopologicalSpace N
      R : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : ContinuousAdd M
      inst✝² : ContinuousAdd N
      inst✝¹ : ContinuousConstSMul R M
      inst✝ : ContinuousConstSMul R N
      f : LinearMap (RingHom.id R) M N
      hf : Continuous ⇑f
      m✝ : R
      x✝ : MeasureTheory.VectorMeasure α M
      i✝ : Set α
      a✝ : MeasurableSet i✝
      ⊢ Eq (↑({ toFun := fun v => v.mapRange f.toAddMonoidHom hf, map_add' := ⋯ }.to …
    -/
    simp
    /-
      🎉 no goals
    -/


open Classical in
/-- The restriction of a vector measure on some set. -/
def restrict (v : VectorMeasure α M) (i : Set α) : VectorMeasure α M :=
  if hi : MeasurableSet i then
    { measureOf' := fun s => if MeasurableSet s then v (s ∩ i) else 0
                   /-
                     α : Type u_1
                     β : Type u_2
                     m inst✝³ : MeasurableSpace α
                     inst✝² : MeasurableSpace β
                     M : Type u_3
                     inst✝¹ : AddCommMonoid M
                     inst✝ : TopologicalSpace M
                     v✝ v : MeasureTheory.VectorMeasure α M
                     i : Set α
                     hi : MeasurableSet i
                     ⊢ Eq ((fun s => ite (MeasurableSet s) (↑v (Inter.inter s i)) 0) EmptyCollectio …
                   -/
      empty' := by simp
                   /-
                     🎉 no goals
                   -/
      not_measurable' := fun _ hi => if_neg hi
      m_iUnion' := by
        /-
          α : Type u_1
          β : Type u_2
          m inst✝³ : MeasurableSpace α
          inst✝² : MeasurableSpace β
          M : Type u_3
          inst✝¹ : AddCommMonoid M
          inst✝ : TopologicalSpace M
          v✝ v : MeasureTheory.VectorMeasure α M
          i : Set α
          hi : MeasurableSet i
          ⊢ ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), MeasurableSet (f i)) → Pairwise (Function …
        -/
        intro f hf₁ hf₂
        /-
          α : Type u_1
          β : Type u_2
          m inst✝³ : MeasurableSpace α
          inst✝² : MeasurableSpace β
          M : Type u_3
          inst✝¹ : AddCommMonoid M
          inst✝ : TopologicalSpace M
          v✝ v : MeasureTheory.VectorMeasure α M
          i : Set α
          hi : MeasurableSet i
          f : Nat → Set α
          hf₁ : ∀ (i : Nat), MeasurableSet (f i)
          hf₂ : Pairwise (Function.onFun Disjoint f)
          ⊢ HasSum (fun i_1 => (fun s => ite (MeasurableSet s) (↑v (Inter.inter s i)) 0) …
        -/
        simp only
        convert v.m_iUnion (fun n => (hf₁ n).inter hi)
            (hf₂.mono fun i j => Disjoint.mono inf_le_left inf_le_left)
          /-
            case h.e'_5.h
            α : Type u_1
            β : Type u_2
            m inst✝³ : MeasurableSpace α
            inst✝² : MeasurableSpace β
            M : Type u_3
            inst✝¹ : AddCommMonoid M
            inst✝ : TopologicalSpace M
            v✝ v : MeasureTheory.VectorMeasure α M
            i : Set α
            hi : MeasurableSet i
            f : Nat → Set α
            hf₁ : ∀ (i : Nat), MeasurableSet (f i)
            hf₂ : Pairwise (Function.onFun Disjoint f)
            x✝ : Nat
            ⊢ Eq (ite (MeasurableSet (f x✝)) (↑v (Inter.inter (f x✝) i)) 0) (↑v (Inter.int …
          -/
        · rw [if_pos (hf₁ _)]
          /-
            🎉 no goals
          -/
          /-
            case h.e'_6
            α : Type u_1
            β : Type u_2
            m inst✝³ : MeasurableSpace α
            inst✝² : MeasurableSpace β
            M : Type u_3
            inst✝¹ : AddCommMonoid M
            inst✝ : TopologicalSpace M
            v✝ v : MeasureTheory.VectorMeasure α M
            i : Set α
            hi : MeasurableSet i
            f : Nat → Set α
            hf₁ : ∀ (i : Nat), MeasurableSet (f i)
            hf₂ : Pairwise (Function.onFun Disjoint f)
            ⊢ Eq (ite (MeasurableSet (Set.iUnion fun i => f i)) (↑v (Inter.inter (Set.iUni …
          -/
        · rw [Set.iUnion_inter, if_pos (MeasurableSet.iUnion hf₁)] }
          /-
            🎉 no goals
          -/
  else 0


theorem restrict_not_measurable {i : Set α} (hi : ¬MeasurableSet i) : v.restrict i = 0 :=
  dif_neg hi


theorem restrict_apply {i : Set α} (hi : MeasurableSet i) {j : Set α} (hj : MeasurableSet j) :
    v.restrict i j = v (j ∩ i) := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    j : Set α
    hj : MeasurableSet j
    ⊢ Eq (↑(v.restrict i) j) (↑v (Inter.inter j i))
  -/
  rw [restrict, dif_pos hi]
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    j : Set α
    hj : MeasurableSet j
    ⊢ Eq (↑{ measureOf' := fun s => ite (MeasurableSet s) (↑v (Inter.inter s i)) 0 …
  -/
  exact if_pos hj
  /-
    🎉 no goals
  -/


theorem restrict_eq_self {i : Set α} (hi : MeasurableSet i) {j : Set α} (hj : MeasurableSet j)
    (hij : j ⊆ i) : v.restrict i j = v j := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    j : Set α
    hj : MeasurableSet j
    hij : HasSubset.Subset j i
    ⊢ Eq (↑(v.restrict i) j) (↑v j)
  -/
  rw [restrict_apply v hi hj, Set.inter_eq_left.2 hij]
  /-
    🎉 no goals
  -/


@[simp]
theorem restrict_empty : v.restrict ∅ = 0 :=
  ext fun i hi => by
    /-
      α : Type u_1
      inst✝² : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      hi : MeasurableSet i
      ⊢ Eq (↑(v.restrict EmptyCollection.emptyCollection) i) (↑0 i)
    -/
    rw [restrict_apply v MeasurableSet.empty hi, Set.inter_empty, v.empty, zero_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem restrict_univ : v.restrict Set.univ = v :=
                     /-
                       α : Type u_1
                       inst✝² : MeasurableSpace α
                       M : Type u_3
                       inst✝¹ : AddCommMonoid M
                       inst✝ : TopologicalSpace M
                       v : MeasureTheory.VectorMeasure α M
                       i : Set α
                       hi : MeasurableSet i
                       ⊢ Eq (↑(v.restrict Set.univ) i) (↑v i)
                     -/
  ext fun i hi => by rw [restrict_apply v MeasurableSet.univ hi, Set.inter_univ]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem restrict_zero {i : Set α} : (0 : VectorMeasure α M).restrict i = 0 := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    i : Set α
    ⊢ Eq (MeasureTheory.VectorMeasure.restrict 0 i) 0
  -/
  by_cases hi : MeasurableSet i
    /-
      case pos
      α : Type u_1
      inst✝² : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      i : Set α
      hi : MeasurableSet i
      ⊢ Eq (MeasureTheory.VectorMeasure.restrict 0 i) 0
    -/
  · ext j hj
    /-
      case pos.h
      α : Type u_1
      inst✝² : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      i : Set α
      hi : MeasurableSet i
      j : Set α
      hj : MeasurableSet j
      ⊢ Eq (↑(MeasureTheory.VectorMeasure.restrict 0 i) j) (↑0 j)
    -/
    rw [restrict_apply 0 hi hj, zero_apply, zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : MeasurableSpace α
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      i : Set α
      hi : Not (MeasurableSet i)
      ⊢ Eq (MeasureTheory.VectorMeasure.restrict 0 i) 0
    -/
  · exact dif_neg hi
    /-
      🎉 no goals
    -/


theorem map_add (v w : VectorMeasure α M) (f : α → β) : (v + w).map f = v.map f + w.map f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : MeasurableSpace β
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousAdd M
    v w : MeasureTheory.VectorMeasure α M
    f : α → β
    ⊢ Eq ((HAdd.hAdd v w).map f) (HAdd.hAdd (v.map f) (w.map f))
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : ContinuousAdd M
      v w : MeasureTheory.VectorMeasure α M
      f : α → β
      hf : Measurable f
      ⊢ Eq ((HAdd.hAdd v w).map f) (HAdd.hAdd (v.map f) (w.map f))
    -/
  · ext i hi
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : ContinuousAdd M
      v w : MeasureTheory.VectorMeasure α M
      f : α → β
      hf : Measurable f
      i : Set β
      hi : MeasurableSet i
      ⊢ Eq (↑((HAdd.hAdd v w).map f) i) (↑(HAdd.hAdd (v.map f) (w.map f)) i)
    -/
    simp [map_apply _ hf hi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : ContinuousAdd M
      v w : MeasureTheory.VectorMeasure α M
      f : α → β
      hf : Not (Measurable f)
      ⊢ Eq ((HAdd.hAdd v w).map f) (HAdd.hAdd (v.map f) (w.map f))
    -/
  · simp [map, dif_neg hf]
    /-
      🎉 no goals
    -/


/-- `VectorMeasure.map` as an additive monoid homomorphism. -/
@[simps]
def mapGm (f : α → β) : VectorMeasure α M →+ VectorMeasure β M where
  toFun v := v.map f
  map_zero' := map_zero f
  map_add' _ _ := map_add _ _ f


theorem restrict_add (v w : VectorMeasure α M) (i : Set α) :
    (v + w).restrict i = v.restrict i + w.restrict i := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousAdd M
    v w : MeasureTheory.VectorMeasure α M
    i : Set α
    ⊢ Eq ((HAdd.hAdd v w).restrict i) (HAdd.hAdd (v.restrict i) (w.restrict i))
  -/
  by_cases hi : MeasurableSet i
    /-
      case pos
      α : Type u_1
      inst✝³ : MeasurableSpace α
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : ContinuousAdd M
      v w : MeasureTheory.VectorMeasure α M
      i : Set α
      hi : MeasurableSet i
      ⊢ Eq ((HAdd.hAdd v w).restrict i) (HAdd.hAdd (v.restrict i) (w.restrict i))
    -/
  · ext j hj
    /-
      case pos.h
      α : Type u_1
      inst✝³ : MeasurableSpace α
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : ContinuousAdd M
      v w : MeasureTheory.VectorMeasure α M
      i : Set α
      hi : MeasurableSet i
      j : Set α
      hj : MeasurableSet j
      ⊢ Eq (↑((HAdd.hAdd v w).restrict i) j) (↑(HAdd.hAdd (v.restrict i) (w.restrict …
    -/
    simp [restrict_apply _ hi hj]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝³ : MeasurableSpace α
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : TopologicalSpace M
      inst✝ : ContinuousAdd M
      v w : MeasureTheory.VectorMeasure α M
      i : Set α
      hi : Not (MeasurableSet i)
      ⊢ Eq ((HAdd.hAdd v w).restrict i) (HAdd.hAdd (v.restrict i) (w.restrict i))
    -/
  · simp [restrict_not_measurable _ hi]
    /-
      🎉 no goals
    -/


/-- `VectorMeasure.restrict` as an additive monoid homomorphism. -/
@[simps]
def restrictGm (i : Set α) : VectorMeasure α M →+ VectorMeasure α M where
  toFun v := v.restrict i
  map_zero' := restrict_zero
  map_add' _ _ := restrict_add _ _ i


@[simp]
theorem map_smul {v : VectorMeasure α M} {f : α → β} (c : R) : (c • v).map f = c • v.map f := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝⁵ : MeasurableSpace β
    M : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : TopologicalSpace M
    R : Type u_4
    inst✝² : Semiring R
    inst✝¹ : DistribMulAction R M
    inst✝ : ContinuousConstSMul R M
    v : MeasureTheory.VectorMeasure α M
    f : α → β
    c : R
    ⊢ Eq ((HSMul.hSMul c v).map f) (HSMul.hSMul c (v.map f))
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝⁵ : MeasurableSpace β
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      f : α → β
      c : R
      hf : Measurable f
      ⊢ Eq ((HSMul.hSMul c v).map f) (HSMul.hSMul c (v.map f))
    -/
  · ext i hi
    /-
      case pos.h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝⁵ : MeasurableSpace β
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      f : α → β
      c : R
      hf : Measurable f
      i : Set β
      hi : MeasurableSet i
      ⊢ Eq (↑((HSMul.hSMul c v).map f) i) (↑(HSMul.hSMul c (v.map f)) i)
    -/
    simp [map_apply _ hf hi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝⁵ : MeasurableSpace β
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      f : α → β
      c : R
      hf : Not (Measurable f)
      ⊢ Eq ((HSMul.hSMul c v).map f) (HSMul.hSMul c (v.map f))
    -/
  · simp only [map, dif_neg hf]
    -- `smul_zero` does not work since we do not require `ContinuousAdd`
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝⁵ : MeasurableSpace β
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      f : α → β
      c : R
      hf : Not (Measurable f)
      ⊢ Eq 0 (HSMul.hSMul c 0)
    -/
    ext i
    /-
      case neg.h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝⁵ : MeasurableSpace β
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      f : α → β
      c : R
      hf : Not (Measurable f)
      i : Set β
      a✝ : MeasurableSet i
      ⊢ Eq (↑0 i) (↑(HSMul.hSMul c 0) i)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem restrict_smul {v : VectorMeasure α M} {i : Set α} (c : R) :
    (c • v).restrict i = c • v.restrict i := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝⁴ : AddCommMonoid M
    inst✝³ : TopologicalSpace M
    R : Type u_4
    inst✝² : Semiring R
    inst✝¹ : DistribMulAction R M
    inst✝ : ContinuousConstSMul R M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    c : R
    ⊢ Eq ((HSMul.hSMul c v).restrict i) (HSMul.hSMul c (v.restrict i))
  -/
  by_cases hi : MeasurableSet i
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      c : R
      hi : MeasurableSet i
      ⊢ Eq ((HSMul.hSMul c v).restrict i) (HSMul.hSMul c (v.restrict i))
    -/
  · ext j hj
    /-
      case pos.h
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      c : R
      hi : MeasurableSet i
      j : Set α
      hj : MeasurableSet j
      ⊢ Eq (↑((HSMul.hSMul c v).restrict i) j) (↑(HSMul.hSMul c (v.restrict i)) j)
    -/
    simp [restrict_apply _ hi hj]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      c : R
      hi : Not (MeasurableSet i)
      ⊢ Eq ((HSMul.hSMul c v).restrict i) (HSMul.hSMul c (v.restrict i))
    -/
  · simp only [restrict_not_measurable _ hi]
    -- `smul_zero` does not work since we do not require `ContinuousAdd`
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      c : R
      hi : Not (MeasurableSet i)
      ⊢ Eq 0 (HSMul.hSMul c 0)
    -/
    ext j
    /-
      case neg.h
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      R : Type u_4
      inst✝² : Semiring R
      inst✝¹ : DistribMulAction R M
      inst✝ : ContinuousConstSMul R M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      c : R
      hi : Not (MeasurableSet i)
      j : Set α
      a✝ : MeasurableSet j
      ⊢ Eq (↑0 j) (↑(HSMul.hSMul c 0) j)
    -/
    simp
    /-
      🎉 no goals
    -/


/-- `VectorMeasure.map` as a linear map. -/
@[simps]
def mapₗ (f : α → β) : VectorMeasure α M →ₗ[R] VectorMeasure β M where
  toFun v := v.map f
  map_add' _ _ := map_add _ _ f
  map_smul' _ _ := map_smul _


/-- `VectorMeasure.restrict` as an additive monoid homomorphism. -/
@[simps]
def restrictₗ (i : Set α) : VectorMeasure α M →ₗ[R] VectorMeasure α M where
  toFun v := v.restrict i
  map_add' _ _ := restrict_add _ _ i
  map_smul' _ _ := restrict_smul _


/-- Vector measures over a partially ordered monoid is partially ordered.

This definition is consistent with `Measure.instPartialOrder`. -/
instance instPartialOrder : PartialOrder (VectorMeasure α M) where
  le v w := ∀ i, MeasurableSet i → v i ≤ w i
  le_refl _ _ _ := le_rfl
  le_trans _ _ _ h₁ h₂ i hi := le_trans (h₁ i hi) (h₂ i hi)
  le_antisymm _ _ h₁ h₂ := ext fun i hi => le_antisymm (h₁ i hi) (h₂ i hi)


theorem le_iff : v ≤ w ↔ ∀ i, MeasurableSet i → v i ≤ w i := Iff.rfl


theorem le_iff' : v ≤ w ↔ ∀ i, v i ≤ w i := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid M
    inst✝ : PartialOrder M
    v w : MeasureTheory.VectorMeasure α M
    ⊢ Iff (LE.le v w) (∀ (i : Set α), LE.le (↑v i) (↑w i))
  -/
  refine ⟨fun h i => ?_, fun h i _ => h i⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid M
    inst✝ : PartialOrder M
    v w : MeasureTheory.VectorMeasure α M
    h : LE.le v w
    i : Set α
    ⊢ LE.le (↑v i) (↑w i)
  -/
  by_cases hi : MeasurableSet i
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      h : LE.le v w
      i : Set α
      hi : MeasurableSet i
      ⊢ LE.le (↑v i) (↑w i)
    -/
  · exact h i hi
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      h : LE.le v w
      i : Set α
      hi : Not (MeasurableSet i)
      ⊢ LE.le (↑v i) (↑w i)
    -/
  · rw [v.not_measurable hi, w.not_measurable hi]
    /-
      🎉 no goals
    -/


scoped[MeasureTheory]
  notation3:50 v " ≤[" i:50 "] " w:50 =>
    MeasureTheory.VectorMeasure.restrict v i ≤ MeasureTheory.VectorMeasure.restrict w i


theorem restrict_le_restrict_iff {i : Set α} (hi : MeasurableSet i) :
    v ≤[i] w ↔ ∀ ⦃j⦄, MeasurableSet j → j ⊆ i → v j ≤ w j :=
  ⟨fun h j hj₁ hj₂ => restrict_eq_self v hi hj₁ hj₂ ▸ restrict_eq_self w hi hj₁ hj₂ ▸ h j hj₁,
    fun h => le_iff.1 fun _ hj =>
      (restrict_apply v hi hj).symm ▸ (restrict_apply w hi hj).symm ▸
      h (hj.inter hi) Set.inter_subset_right⟩


theorem subset_le_of_restrict_le_restrict {i : Set α} (hi : MeasurableSet i) (hi₂ : v ≤[i] w)
    {j : Set α} (hj : j ⊆ i) : v j ≤ w j := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid M
    inst✝ : PartialOrder M
    v w : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    hi₂ : LE.le (v.restrict i) (w.restrict i)
    j : Set α
    hj : HasSubset.Subset j i
    ⊢ LE.le (↑v j) (↑w j)
  -/
  by_cases hj₁ : MeasurableSet j
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      i : Set α
      hi : MeasurableSet i
      hi₂ : LE.le (v.restrict i) (w.restrict i)
      j : Set α
      hj : HasSubset.Subset j i
      hj₁ : MeasurableSet j
      ⊢ LE.le (↑v j) (↑w j)
    -/
  · exact (restrict_le_restrict_iff _ _ hi).1 hi₂ hj₁ hj
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      i : Set α
      hi : MeasurableSet i
      hi₂ : LE.le (v.restrict i) (w.restrict i)
      j : Set α
      hj : HasSubset.Subset j i
      hj₁ : Not (MeasurableSet j)
      ⊢ LE.le (↑v j) (↑w j)
    -/
  · rw [v.not_measurable hj₁, w.not_measurable hj₁]
    /-
      🎉 no goals
    -/


theorem restrict_le_restrict_of_subset_le {i : Set α}
    (h : ∀ ⦃j⦄, MeasurableSet j → j ⊆ i → v j ≤ w j) : v ≤[i] w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid M
    inst✝ : PartialOrder M
    v w : MeasureTheory.VectorMeasure α M
    i : Set α
    h : ∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑v j) (↑w j)
    ⊢ LE.le (v.restrict i) (w.restrict i)
  -/
  by_cases hi : MeasurableSet i
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      i : Set α
      h : ∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑v j) (↑w j)
      hi : MeasurableSet i
      ⊢ LE.le (v.restrict i) (w.restrict i)
    -/
  · exact (restrict_le_restrict_iff _ _ hi).2 h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      i : Set α
      h : ∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑v j) (↑w j)
      hi : Not (MeasurableSet i)
      ⊢ LE.le (v.restrict i) (w.restrict i)
    -/
  · rw [restrict_not_measurable v hi, restrict_not_measurable w hi]
    /-
      🎉 no goals
    -/


theorem restrict_le_restrict_subset {i j : Set α} (hi₁ : MeasurableSet i) (hi₂ : v ≤[i] w)
    (hij : j ⊆ i) : v ≤[j] w :=
  restrict_le_restrict_of_subset_le v w fun _ _ hk₂ =>
    subset_le_of_restrict_le_restrict v w hi₁ hi₂ (Set.Subset.trans hk₂ hij)


theorem le_restrict_empty : v ≤[∅] w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid M
    inst✝ : PartialOrder M
    v w : MeasureTheory.VectorMeasure α M
    ⊢ LE.le (v.restrict EmptyCollection.emptyCollection) (w.restrict EmptyCollecti …
  -/
  intro j _
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid M
    inst✝ : PartialOrder M
    v w : MeasureTheory.VectorMeasure α M
    j : Set α
    a✝ : MeasurableSet j
    ⊢ LE.le (↑(v.restrict EmptyCollection.emptyCollection) j) (↑(w.restrict EmptyC …
  -/
  rw [restrict_empty, restrict_empty]
  /-
    🎉 no goals
  -/


theorem le_restrict_univ_iff_le : v ≤[Set.univ] w ↔ v ≤ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid M
    inst✝ : PartialOrder M
    v w : MeasureTheory.VectorMeasure α M
    ⊢ Iff (LE.le (v.restrict Set.univ) (w.restrict Set.univ)) (LE.le v w)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      ⊢ LE.le (v.restrict Set.univ) (w.restrict Set.univ) → LE.le v w
    -/
  · intro h s hs
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      h : LE.le (v.restrict Set.univ) (w.restrict Set.univ)
      s : Set α
      hs : MeasurableSet s
      ⊢ LE.le (↑v s) (↑w s)
    -/
    have := h s hs
    rwa [restrict_apply _ MeasurableSet.univ hs, Set.inter_univ,
      restrict_apply _ MeasurableSet.univ hs, Set.inter_univ] at this
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      ⊢ LE.le v w → LE.le (v.restrict Set.univ) (w.restrict Set.univ)
    -/
  · intro h s hs
    rw [restrict_apply _ MeasurableSet.univ hs, Set.inter_univ,
      restrict_apply _ MeasurableSet.univ hs, Set.inter_univ]
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid M
      inst✝ : PartialOrder M
      v w : MeasureTheory.VectorMeasure α M
      h : LE.le v w
      s : Set α
      hs : MeasurableSet s
      ⊢ LE.le (↑v s) (↑w s)
    -/
    exact h s hs
    /-
      🎉 no goals
    -/


nonrec theorem neg_le_neg {i : Set α} (hi : MeasurableSet i) (h : v ≤[i] w) : -w ≤[i] -v := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommGroup M
    inst✝ : TopologicalAddGroup M
    v w : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    h : LE.le (v.restrict i) (w.restrict i)
    ⊢ LE.le ((Neg.neg w).restrict i) ((Neg.neg v).restrict i)
  -/
  intro j hj₁
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommGroup M
    inst✝ : TopologicalAddGroup M
    v w : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    h : LE.le (v.restrict i) (w.restrict i)
    j : Set α
    hj₁ : MeasurableSet j
    ⊢ LE.le (↑((Neg.neg w).restrict i) j) (↑((Neg.neg v).restrict i) j)
  -/
  rw [restrict_apply _ hi hj₁, restrict_apply _ hi hj₁, neg_apply, neg_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommGroup M
    inst✝ : TopologicalAddGroup M
    v w : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    h : LE.le (v.restrict i) (w.restrict i)
    j : Set α
    hj₁ : MeasurableSet j
    ⊢ LE.le (Neg.neg (↑w (Inter.inter j i))) (Neg.neg (↑v (Inter.inter j i)))
  -/
  refine neg_le_neg ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommGroup M
    inst✝ : TopologicalAddGroup M
    v w : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    h : LE.le (v.restrict i) (w.restrict i)
    j : Set α
    hj₁ : MeasurableSet j
    ⊢ LE.le (↑v (Inter.inter j i)) (↑w (Inter.inter j i))
  -/
  rw [← restrict_apply _ hi hj₁, ← restrict_apply _ hi hj₁]
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommGroup M
    inst✝ : TopologicalAddGroup M
    v w : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    h : LE.le (v.restrict i) (w.restrict i)
    j : Set α
    hj₁ : MeasurableSet j
    ⊢ LE.le (↑(v.restrict i) j) (↑(w.restrict i) j)
  -/
  exact h j hj₁
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_le_neg_iff {i : Set α} (hi : MeasurableSet i) : -w ≤[i] -v ↔ v ≤[i] w :=
  ⟨fun h => neg_neg v ▸ neg_neg w ▸ neg_le_neg _ _ hi h, fun h => neg_le_neg _ _ hi h⟩


theorem restrict_le_restrict_iUnion {f : ℕ → Set α} (hf₁ : ∀ n, MeasurableSet (f n))
    (hf₂ : ∀ n, v ≤[f n] w) : v ≤[⋃ n, f n] w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : OrderClosedTopology M
    v w : MeasureTheory.VectorMeasure α M
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
    ⊢ LE.le (v.restrict (Set.iUnion fun n => f n)) (w.restrict (Set.iUnion fun n = …
  -/
  refine restrict_le_restrict_of_subset_le v w fun a ha₁ ha₂ => ?_
  have ha₃ : ⋃ n, a ∩ disjointed f n = a := by
    rwa [← Set.inter_iUnion, iUnion_disjointed, Set.inter_eq_left]
  have ha₄ : Pairwise (Disjoint on fun n => a ∩ disjointed f n) :=
    (disjoint_disjointed _).mono fun i j => Disjoint.mono inf_le_right inf_le_right
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : OrderClosedTopology M
    v w : MeasureTheory.VectorMeasure α M
    f : Nat → Set α
    hf₁ : ∀ (n : Nat), MeasurableSet (f n)
    hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
    a : Set α
    ha₁ : MeasurableSet a
    ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
    ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
    ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
    ⊢ LE.le (↑v a) (↑w a)
  -/
  rw [← ha₃, v.of_disjoint_iUnion _ ha₄, w.of_disjoint_iUnion _ ha₄]
    /-
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : OrderedAddCommMonoid M
      inst✝ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      f : Nat → Set α
      hf₁ : ∀ (n : Nat), MeasurableSet (f n)
      hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
      a : Set α
      ha₁ : MeasurableSet a
      ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
      ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
      ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
      ⊢ LE.le (tsum fun i => ↑v (Inter.inter a (disjointed f i))) (tsum fun i => ↑w  …
    -/
  · refine tsum_le_tsum (fun n => (restrict_le_restrict_iff v w (hf₁ n)).1 (hf₂ n) ?_ ?_) ?_ ?_
      /-
        case refine_1
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_3
        inst✝² : TopologicalSpace M
        inst✝¹ : OrderedAddCommMonoid M
        inst✝ : OrderClosedTopology M
        v w : MeasureTheory.VectorMeasure α M
        f : Nat → Set α
        hf₁ : ∀ (n : Nat), MeasurableSet (f n)
        hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
        a : Set α
        ha₁ : MeasurableSet a
        ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
        ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
        ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
        n : Nat
        ⊢ MeasurableSet (Inter.inter a (disjointed f n))
      -/
    · exact ha₁.inter (MeasurableSet.disjointed hf₁ n)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_3
        inst✝² : TopologicalSpace M
        inst✝¹ : OrderedAddCommMonoid M
        inst✝ : OrderClosedTopology M
        v w : MeasureTheory.VectorMeasure α M
        f : Nat → Set α
        hf₁ : ∀ (n : Nat), MeasurableSet (f n)
        hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
        a : Set α
        ha₁ : MeasurableSet a
        ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
        ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
        ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
        n : Nat
        ⊢ HasSubset.Subset (Inter.inter a (disjointed f n)) (f n)
      -/
    · exact Set.Subset.trans Set.inter_subset_right (disjointed_subset _ _)
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_3
        inst✝² : TopologicalSpace M
        inst✝¹ : OrderedAddCommMonoid M
        inst✝ : OrderClosedTopology M
        v w : MeasureTheory.VectorMeasure α M
        f : Nat → Set α
        hf₁ : ∀ (n : Nat), MeasurableSet (f n)
        hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
        a : Set α
        ha₁ : MeasurableSet a
        ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
        ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
        ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
        ⊢ Summable fun i => ↑v (Inter.inter a (disjointed f i))
      -/
    · refine (v.m_iUnion (fun n => ?_) ?_).summable
        /-
          case refine_3.refine_1
          α : Type u_1
          m : MeasurableSpace α
          M : Type u_3
          inst✝² : TopologicalSpace M
          inst✝¹ : OrderedAddCommMonoid M
          inst✝ : OrderClosedTopology M
          v w : MeasureTheory.VectorMeasure α M
          f : Nat → Set α
          hf₁ : ∀ (n : Nat), MeasurableSet (f n)
          hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
          a : Set α
          ha₁ : MeasurableSet a
          ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
          ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
          ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
          n : Nat
          ⊢ MeasurableSet (Inter.inter a (disjointed f n))
        -/
      · exact ha₁.inter (MeasurableSet.disjointed hf₁ n)
        /-
          🎉 no goals
        -/
        /-
          case refine_3.refine_2
          α : Type u_1
          m : MeasurableSpace α
          M : Type u_3
          inst✝² : TopologicalSpace M
          inst✝¹ : OrderedAddCommMonoid M
          inst✝ : OrderClosedTopology M
          v w : MeasureTheory.VectorMeasure α M
          f : Nat → Set α
          hf₁ : ∀ (n : Nat), MeasurableSet (f n)
          hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
          a : Set α
          ha₁ : MeasurableSet a
          ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
          ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
          ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
          ⊢ Pairwise (Function.onFun Disjoint fun i => Inter.inter a (disjointed f i))
        -/
      · exact (disjoint_disjointed _).mono fun i j => Disjoint.mono inf_le_right inf_le_right
        /-
          🎉 no goals
        -/
      /-
        case refine_4
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_3
        inst✝² : TopologicalSpace M
        inst✝¹ : OrderedAddCommMonoid M
        inst✝ : OrderClosedTopology M
        v w : MeasureTheory.VectorMeasure α M
        f : Nat → Set α
        hf₁ : ∀ (n : Nat), MeasurableSet (f n)
        hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
        a : Set α
        ha₁ : MeasurableSet a
        ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
        ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
        ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
        ⊢ Summable fun i => ↑w (Inter.inter a (disjointed f i))
      -/
    · refine (w.m_iUnion (fun n => ?_) ?_).summable
        /-
          case refine_4.refine_1
          α : Type u_1
          m : MeasurableSpace α
          M : Type u_3
          inst✝² : TopologicalSpace M
          inst✝¹ : OrderedAddCommMonoid M
          inst✝ : OrderClosedTopology M
          v w : MeasureTheory.VectorMeasure α M
          f : Nat → Set α
          hf₁ : ∀ (n : Nat), MeasurableSet (f n)
          hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
          a : Set α
          ha₁ : MeasurableSet a
          ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
          ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
          ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
          n : Nat
          ⊢ MeasurableSet (Inter.inter a (disjointed f n))
        -/
      · exact ha₁.inter (MeasurableSet.disjointed hf₁ n)
        /-
          🎉 no goals
        -/
        /-
          case refine_4.refine_2
          α : Type u_1
          m : MeasurableSpace α
          M : Type u_3
          inst✝² : TopologicalSpace M
          inst✝¹ : OrderedAddCommMonoid M
          inst✝ : OrderClosedTopology M
          v w : MeasureTheory.VectorMeasure α M
          f : Nat → Set α
          hf₁ : ∀ (n : Nat), MeasurableSet (f n)
          hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
          a : Set α
          ha₁ : MeasurableSet a
          ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
          ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
          ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
          ⊢ Pairwise (Function.onFun Disjoint fun i => Inter.inter a (disjointed f i))
        -/
      · exact (disjoint_disjointed _).mono fun i j => Disjoint.mono inf_le_right inf_le_right
        /-
          🎉 no goals
        -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : OrderedAddCommMonoid M
      inst✝ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      f : Nat → Set α
      hf₁ : ∀ (n : Nat), MeasurableSet (f n)
      hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
      a : Set α
      ha₁ : MeasurableSet a
      ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
      ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
      ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
      ⊢ ∀ (i : Nat), MeasurableSet (Inter.inter a (disjointed f i))
    -/
  · intro n
    /-
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : OrderedAddCommMonoid M
      inst✝ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      f : Nat → Set α
      hf₁ : ∀ (n : Nat), MeasurableSet (f n)
      hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
      a : Set α
      ha₁ : MeasurableSet a
      ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
      ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
      ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
      n : Nat
      ⊢ MeasurableSet (Inter.inter a (disjointed f n))
    -/
    exact ha₁.inter (MeasurableSet.disjointed hf₁ n)
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : OrderedAddCommMonoid M
      inst✝ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      f : Nat → Set α
      hf₁ : ∀ (n : Nat), MeasurableSet (f n)
      hf₂ : ∀ (n : Nat), LE.le (v.restrict (f n)) (w.restrict (f n))
      a : Set α
      ha₁ : MeasurableSet a
      ha₂ : HasSubset.Subset a (Set.iUnion fun n => f n)
      ha₃ : Eq (Set.iUnion fun n => Inter.inter a (disjointed f n)) a
      ha₄ : Pairwise (Function.onFun Disjoint fun n => Inter.inter a (disjointed f n))
      ⊢ ∀ (i : Nat), MeasurableSet (Inter.inter a (disjointed f i))
    -/
  · exact fun n => ha₁.inter (MeasurableSet.disjointed hf₁ n)
    /-
      🎉 no goals
    -/


theorem restrict_le_restrict_countable_iUnion [Countable β] {f : β → Set α}
    (hf₁ : ∀ b, MeasurableSet (f b)) (hf₂ : ∀ b, v ≤[f b] w) : v ≤[⋃ b, f b] w := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : OrderedAddCommMonoid M
    inst✝¹ : OrderClosedTopology M
    v w : MeasureTheory.VectorMeasure α M
    inst✝ : Countable β
    f : β → Set α
    hf₁ : ∀ (b : β), MeasurableSet (f b)
    hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
    ⊢ LE.le (v.restrict (Set.iUnion fun b => f b)) (w.restrict (Set.iUnion fun b = …
  -/
  cases nonempty_encodable β
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : OrderedAddCommMonoid M
    inst✝¹ : OrderClosedTopology M
    v w : MeasureTheory.VectorMeasure α M
    inst✝ : Countable β
    f : β → Set α
    hf₁ : ∀ (b : β), MeasurableSet (f b)
    hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
    val✝ : Encodable β
    ⊢ LE.le (v.restrict (Set.iUnion fun b => f b)) (w.restrict (Set.iUnion fun b = …
  -/
  rw [← Encodable.iUnion_decode₂]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    M : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : OrderedAddCommMonoid M
    inst✝¹ : OrderClosedTopology M
    v w : MeasureTheory.VectorMeasure α M
    inst✝ : Countable β
    f : β → Set α
    hf₁ : ∀ (b : β), MeasurableSet (f b)
    hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
    val✝ : Encodable β
    ⊢ LE.le (v.restrict (Set.iUnion fun i => Set.iUnion fun b => Set.iUnion fun h  …
  -/
  refine restrict_le_restrict_iUnion v w ?_ ?_
    /-
      case intro.refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : OrderedAddCommMonoid M
      inst✝¹ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      inst✝ : Countable β
      f : β → Set α
      hf₁ : ∀ (b : β), MeasurableSet (f b)
      hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
      val✝ : Encodable β
      ⊢ ∀ (n : Nat), MeasurableSet (Set.iUnion fun b => Set.iUnion fun h => f b)
    -/
  · intro n
    /-
      case intro.refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : OrderedAddCommMonoid M
      inst✝¹ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      inst✝ : Countable β
      f : β → Set α
      hf₁ : ∀ (b : β), MeasurableSet (f b)
      hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
      val✝ : Encodable β
      n : Nat
      ⊢ MeasurableSet (Set.iUnion fun b => Set.iUnion fun h => f b)
    -/
    measurability
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : OrderedAddCommMonoid M
      inst✝¹ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      inst✝ : Countable β
      f : β → Set α
      hf₁ : ∀ (b : β), MeasurableSet (f b)
      hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
      val✝ : Encodable β
      ⊢ ∀ (n : Nat), LE.le (v.restrict (Set.iUnion fun b => Set.iUnion fun h => f b) …
    -/
  · intro n
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_3
      inst✝³ : TopologicalSpace M
      inst✝² : OrderedAddCommMonoid M
      inst✝¹ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      inst✝ : Countable β
      f : β → Set α
      hf₁ : ∀ (b : β), MeasurableSet (f b)
      hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
      val✝ : Encodable β
      n : Nat
      ⊢ LE.le (v.restrict (Set.iUnion fun b => Set.iUnion fun h => f b)) (w.restrict …
    -/
    cases' Encodable.decode₂ β n with b
      /-
        case intro.refine_2.none
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : OrderedAddCommMonoid M
        inst✝¹ : OrderClosedTopology M
        v w : MeasureTheory.VectorMeasure α M
        inst✝ : Countable β
        f : β → Set α
        hf₁ : ∀ (b : β), MeasurableSet (f b)
        hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
        val✝ : Encodable β
        n : Nat
        ⊢ LE.le (v.restrict (Set.iUnion fun b => Set.iUnion fun h => f b)) (w.restrict …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_2.some
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        M : Type u_3
        inst✝³ : TopologicalSpace M
        inst✝² : OrderedAddCommMonoid M
        inst✝¹ : OrderClosedTopology M
        v w : MeasureTheory.VectorMeasure α M
        inst✝ : Countable β
        f : β → Set α
        hf₁ : ∀ (b : β), MeasurableSet (f b)
        hf₂ : ∀ (b : β), LE.le (v.restrict (f b)) (w.restrict (f b))
        val✝ : Encodable β
        n : Nat
        b : β
        ⊢ LE.le (v.restrict (Set.iUnion fun b_1 => Set.iUnion fun h => f b_1)) (w.rest …
      -/
    · simp [hf₂ b]
      /-
        🎉 no goals
      -/


theorem restrict_le_restrict_union (hi₁ : MeasurableSet i) (hi₂ : v ≤[i] w) (hj₁ : MeasurableSet j)
    (hj₂ : v ≤[j] w) : v ≤[i ∪ j] w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : OrderClosedTopology M
    v w : MeasureTheory.VectorMeasure α M
    i j : Set α
    hi₁ : MeasurableSet i
    hi₂ : LE.le (v.restrict i) (w.restrict i)
    hj₁ : MeasurableSet j
    hj₂ : LE.le (v.restrict j) (w.restrict j)
    ⊢ LE.le (v.restrict (Union.union i j)) (w.restrict (Union.union i j))
  -/
  rw [Set.union_eq_iUnion]
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝² : TopologicalSpace M
    inst✝¹ : OrderedAddCommMonoid M
    inst✝ : OrderClosedTopology M
    v w : MeasureTheory.VectorMeasure α M
    i j : Set α
    hi₁ : MeasurableSet i
    hi₂ : LE.le (v.restrict i) (w.restrict i)
    hj₁ : MeasurableSet j
    hj₂ : LE.le (v.restrict j) (w.restrict j)
    ⊢ LE.le (v.restrict (Set.iUnion fun b => cond b i j)) (w.restrict (Set.iUnion  …
  -/
  refine restrict_le_restrict_countable_iUnion v w ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : OrderedAddCommMonoid M
      inst✝ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      i j : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (v.restrict i) (w.restrict i)
      hj₁ : MeasurableSet j
      hj₂ : LE.le (v.restrict j) (w.restrict j)
      ⊢ ∀ (b : Bool), MeasurableSet (cond b i j)
    -/
  · measurability
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝² : TopologicalSpace M
      inst✝¹ : OrderedAddCommMonoid M
      inst✝ : OrderClosedTopology M
      v w : MeasureTheory.VectorMeasure α M
      i j : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (v.restrict i) (w.restrict i)
      hj₁ : MeasurableSet j
      hj₂ : LE.le (v.restrict j) (w.restrict j)
      ⊢ ∀ (b : Bool), LE.le (v.restrict (cond b i j)) (w.restrict (cond b i j))
    -/
                       /-
                         🎉 no goals
                       -/
  · rintro (_ | _) <;> simpa
                       /-
                         🎉 no goals
                       -/


theorem nonneg_of_zero_le_restrict (hi₂ : 0 ≤[i] v) : 0 ≤ v i := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : OrderedAddCommMonoid M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (v.restrict i)
    ⊢ LE.le 0 (↑v i)
  -/
  by_cases hi₁ : MeasurableSet i
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : OrderedAddCommMonoid M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (v.restrict i)
      hi₁ : MeasurableSet i
      ⊢ LE.le 0 (↑v i)
    -/
  · exact (restrict_le_restrict_iff _ _ hi₁).1 hi₂ hi₁ Set.Subset.rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : OrderedAddCommMonoid M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (v.restrict i)
      hi₁ : Not (MeasurableSet i)
      ⊢ LE.le 0 (↑v i)
    -/
  · rw [v.not_measurable hi₁]
    /-
      🎉 no goals
    -/


theorem nonpos_of_restrict_le_zero (hi₂ : v ≤[i] 0) : v i ≤ 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : OrderedAddCommMonoid M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi₂ : LE.le (v.restrict i) (MeasureTheory.VectorMeasure.restrict 0 i)
    ⊢ LE.le (↑v i) 0
  -/
  by_cases hi₁ : MeasurableSet i
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : OrderedAddCommMonoid M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      hi₂ : LE.le (v.restrict i) (MeasureTheory.VectorMeasure.restrict 0 i)
      hi₁ : MeasurableSet i
      ⊢ LE.le (↑v i) 0
    -/
  · exact (restrict_le_restrict_iff _ _ hi₁).1 hi₂ hi₁ Set.Subset.rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_3
      inst✝¹ : TopologicalSpace M
      inst✝ : OrderedAddCommMonoid M
      v : MeasureTheory.VectorMeasure α M
      i : Set α
      hi₂ : LE.le (v.restrict i) (MeasureTheory.VectorMeasure.restrict 0 i)
      hi₁ : Not (MeasurableSet i)
      ⊢ LE.le (↑v i) 0
    -/
  · rw [v.not_measurable hi₁]
    /-
      🎉 no goals
    -/


theorem zero_le_restrict_not_measurable (hi : ¬MeasurableSet i) : 0 ≤[i] v := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : OrderedAddCommMonoid M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : Not (MeasurableSet i)
    ⊢ LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (v.restrict i)
  -/
  rw [restrict_zero, restrict_not_measurable _ hi]
  /-
    🎉 no goals
  -/


theorem restrict_le_zero_of_not_measurable (hi : ¬MeasurableSet i) : v ≤[i] 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : OrderedAddCommMonoid M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : Not (MeasurableSet i)
    ⊢ LE.le (v.restrict i) (MeasureTheory.VectorMeasure.restrict 0 i)
  -/
  rw [restrict_zero, restrict_not_measurable _ hi]
  /-
    🎉 no goals
  -/


theorem measurable_of_not_zero_le_restrict (hi : ¬0 ≤[i] v) : MeasurableSet i :=
  Not.imp_symm (zero_le_restrict_not_measurable _) hi


theorem measurable_of_not_restrict_le_zero (hi : ¬v ≤[i] 0) : MeasurableSet i :=
  Not.imp_symm (restrict_le_zero_of_not_measurable _) hi


theorem zero_le_restrict_subset (hi₁ : MeasurableSet i) (hij : j ⊆ i) (hi₂ : 0 ≤[i] v) : 0 ≤[j] v :=
  restrict_le_restrict_of_subset_le _ _ fun _ hk₁ hk₂ =>
    (restrict_le_restrict_iff _ _ hi₁).1 hi₂ hk₁ (Set.Subset.trans hk₂ hij)


theorem restrict_le_zero_subset (hi₁ : MeasurableSet i) (hij : j ⊆ i) (hi₂ : v ≤[i] 0) : v ≤[j] 0 :=
  restrict_le_restrict_of_subset_le _ _ fun _ hk₁ hk₂ =>
    (restrict_le_restrict_iff _ _ hi₁).1 hi₂ hk₁ (Set.Subset.trans hk₂ hij)


theorem exists_pos_measure_of_not_restrict_le_zero (hi : ¬v ≤[i] 0) :
    ∃ j : Set α, MeasurableSet j ∧ j ⊆ i ∧ 0 < v j := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : LinearOrderedAddCommMonoid M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : Not (LE.le (v.restrict i) (MeasureTheory.VectorMeasure.restrict 0 i))
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (LT.lt 0 ( …
  -/
  have hi₁ : MeasurableSet i := measurable_of_not_restrict_le_zero _ hi
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : LinearOrderedAddCommMonoid M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : Not (LE.le (v.restrict i) (MeasureTheory.VectorMeasure.restrict 0 i))
    hi₁ : MeasurableSet i
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (LT.lt 0 ( …
  -/
  rw [restrict_le_restrict_iff _ _ hi₁] at hi
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : LinearOrderedAddCommMonoid M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : Not (∀ ⦃j : Set α⦄, MeasurableSet j → HasSubset.Subset j i → LE.le (↑v j) …
    hi₁ : MeasurableSet i
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (LT.lt 0 ( …
  -/
  push_neg at hi
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_3
    inst✝¹ : TopologicalSpace M
    inst✝ : LinearOrderedAddCommMonoid M
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi₁ : MeasurableSet i
    hi : Exists fun ⦃j⦄ => And (MeasurableSet j) (And (HasSubset.Subset j i) (LT.l …
    ⊢ Exists fun j => And (MeasurableSet j) (And (HasSubset.Subset j i) (LT.lt 0 ( …
  -/
  exact hi
  /-
    🎉 no goals
  -/


instance instAddLeftMono : AddLeftMono (VectorMeasure α M) :=
  ⟨fun _ _ _ h i hi => add_le_add_left (h i hi) _⟩


/-- A vector measure `v` is absolutely continuous with respect to a measure `μ` if for all sets
`s`, `μ s = 0`, we have `v s = 0`. -/
def AbsolutelyContinuous (v : VectorMeasure α M) (w : VectorMeasure α N) :=
  ∀ ⦃s : Set α⦄, w s = 0 → v s = 0


@[inherit_doc VectorMeasure.AbsolutelyContinuous]
scoped[MeasureTheory] infixl:50 " ≪ᵥ " => MeasureTheory.VectorMeasure.AbsolutelyContinuous


theorem mk (h : ∀ ⦃s : Set α⦄, MeasurableSet s → w s = 0 → v s = 0) : v ≪ᵥ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝³ : AddCommMonoid M
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid N
    inst✝ : TopologicalSpace N
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : ∀ ⦃s : Set α⦄, MeasurableSet s → Eq (↑w s) 0 → Eq (↑v s) 0
    ⊢ v.AbsolutelyContinuous w
  -/
  intro s hs
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝³ : AddCommMonoid M
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid N
    inst✝ : TopologicalSpace N
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : ∀ ⦃s : Set α⦄, MeasurableSet s → Eq (↑w s) 0 → Eq (↑v s) 0
    s : Set α
    hs : Eq (↑w s) 0
    ⊢ Eq (↑v s) 0
  -/
  by_cases hmeas : MeasurableSet s
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝³ : AddCommMonoid M
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid N
      inst✝ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      h : ∀ ⦃s : Set α⦄, MeasurableSet s → Eq (↑w s) 0 → Eq (↑v s) 0
      s : Set α
      hs : Eq (↑w s) 0
      hmeas : MeasurableSet s
      ⊢ Eq (↑v s) 0
    -/
  · exact h hmeas hs
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝³ : AddCommMonoid M
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid N
      inst✝ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      h : ∀ ⦃s : Set α⦄, MeasurableSet s → Eq (↑w s) 0 → Eq (↑v s) 0
      s : Set α
      hs : Eq (↑w s) 0
      hmeas : Not (MeasurableSet s)
      ⊢ Eq (↑v s) 0
    -/
  · exact not_measurable v hmeas
    /-
      🎉 no goals
    -/


theorem eq {w : VectorMeasure α M} (h : v = w) : v ≪ᵥ w :=
  fun _ hs => h.symm ▸ hs


@[refl]
theorem refl (v : VectorMeasure α M) : v ≪ᵥ v :=
  eq rfl


@[trans]
theorem trans {u : VectorMeasure α L} {v : VectorMeasure α M} {w : VectorMeasure α N} (huv : u ≪ᵥ v)
    (hvw : v ≪ᵥ w) : u ≪ᵥ w :=
  fun _ hs => huv <| hvw hs


theorem zero (v : VectorMeasure α N) : (0 : VectorMeasure α M) ≪ᵥ v :=
  fun s _ => VectorMeasure.zero_apply s


theorem neg_left {M : Type*} [AddCommGroup M] [TopologicalSpace M] [TopologicalAddGroup M]
    {v : VectorMeasure α M} {w : VectorMeasure α N} (h : v ≪ᵥ w) : -v ≪ᵥ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    N : Type u_5
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    M : Type u_6
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : v.AbsolutelyContinuous w
    ⊢ (Neg.neg v).AbsolutelyContinuous w
  -/
  intro s hs
  /-
    α : Type u_1
    m : MeasurableSpace α
    N : Type u_5
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    M : Type u_6
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : v.AbsolutelyContinuous w
    s : Set α
    hs : Eq (↑w s) 0
    ⊢ Eq (↑(Neg.neg v) s) 0
  -/
  rw [neg_apply, h hs, neg_zero]
  /-
    🎉 no goals
  -/


theorem neg_right {N : Type*} [AddCommGroup N] [TopologicalSpace N] [TopologicalAddGroup N]
    {v : VectorMeasure α M} {w : VectorMeasure α N} (h : v ≪ᵥ w) : v ≪ᵥ -w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝⁴ : AddCommMonoid M
    inst✝³ : TopologicalSpace M
    N : Type u_6
    inst✝² : AddCommGroup N
    inst✝¹ : TopologicalSpace N
    inst✝ : TopologicalAddGroup N
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : v.AbsolutelyContinuous w
    ⊢ v.AbsolutelyContinuous (Neg.neg w)
  -/
  intro s hs
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝⁴ : AddCommMonoid M
    inst✝³ : TopologicalSpace M
    N : Type u_6
    inst✝² : AddCommGroup N
    inst✝¹ : TopologicalSpace N
    inst✝ : TopologicalAddGroup N
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : v.AbsolutelyContinuous w
    s : Set α
    hs : Eq (↑(Neg.neg w) s) 0
    ⊢ Eq (↑v s) 0
  -/
  rw [neg_apply, neg_eq_zero] at hs
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝⁴ : AddCommMonoid M
    inst✝³ : TopologicalSpace M
    N : Type u_6
    inst✝² : AddCommGroup N
    inst✝¹ : TopologicalSpace N
    inst✝ : TopologicalAddGroup N
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : v.AbsolutelyContinuous w
    s : Set α
    hs : Eq (↑w s) 0
    ⊢ Eq (↑v s) 0
  -/
  exact h hs
  /-
    🎉 no goals
  -/


theorem add [ContinuousAdd M] {v₁ v₂ : VectorMeasure α M} {w : VectorMeasure α N} (hv₁ : v₁ ≪ᵥ w)
    (hv₂ : v₂ ≪ᵥ w) : v₁ + v₂ ≪ᵥ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝⁴ : AddCommMonoid M
    inst✝³ : TopologicalSpace M
    inst✝² : AddCommMonoid N
    inst✝¹ : TopologicalSpace N
    inst✝ : ContinuousAdd M
    v₁ v₂ : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    hv₁ : v₁.AbsolutelyContinuous w
    hv₂ : v₂.AbsolutelyContinuous w
    ⊢ (HAdd.hAdd v₁ v₂).AbsolutelyContinuous w
  -/
  intro s hs
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝⁴ : AddCommMonoid M
    inst✝³ : TopologicalSpace M
    inst✝² : AddCommMonoid N
    inst✝¹ : TopologicalSpace N
    inst✝ : ContinuousAdd M
    v₁ v₂ : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    hv₁ : v₁.AbsolutelyContinuous w
    hv₂ : v₂.AbsolutelyContinuous w
    s : Set α
    hs : Eq (↑w s) 0
    ⊢ Eq (↑(HAdd.hAdd v₁ v₂) s) 0
  -/
  rw [add_apply, hv₁ hs, hv₂ hs, zero_add]
  /-
    🎉 no goals
  -/


theorem sub {M : Type*} [AddCommGroup M] [TopologicalSpace M] [TopologicalAddGroup M]
    {v₁ v₂ : VectorMeasure α M} {w : VectorMeasure α N} (hv₁ : v₁ ≪ᵥ w) (hv₂ : v₂ ≪ᵥ w) :
    v₁ - v₂ ≪ᵥ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    N : Type u_5
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    M : Type u_6
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    v₁ v₂ : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    hv₁ : v₁.AbsolutelyContinuous w
    hv₂ : v₂.AbsolutelyContinuous w
    ⊢ (HSub.hSub v₁ v₂).AbsolutelyContinuous w
  -/
  intro s hs
  /-
    α : Type u_1
    m : MeasurableSpace α
    N : Type u_5
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    M : Type u_6
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    v₁ v₂ : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    hv₁ : v₁.AbsolutelyContinuous w
    hv₂ : v₂.AbsolutelyContinuous w
    s : Set α
    hs : Eq (↑w s) 0
    ⊢ Eq (↑(HSub.hSub v₁ v₂) s) 0
  -/
  rw [sub_apply, hv₁ hs, hv₂ hs, zero_sub, neg_zero]
  /-
    🎉 no goals
  -/


theorem smul {R : Type*} [Semiring R] [DistribMulAction R M] [ContinuousConstSMul R M] {r : R}
    {v : VectorMeasure α M} {w : VectorMeasure α N} (h : v ≪ᵥ w) : r • v ≪ᵥ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    R : Type u_6
    inst✝² : Semiring R
    inst✝¹ : DistribMulAction R M
    inst✝ : ContinuousConstSMul R M
    r : R
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : v.AbsolutelyContinuous w
    ⊢ (HSMul.hSMul r v).AbsolutelyContinuous w
  -/
  intro s hs
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    R : Type u_6
    inst✝² : Semiring R
    inst✝¹ : DistribMulAction R M
    inst✝ : ContinuousConstSMul R M
    r : R
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : v.AbsolutelyContinuous w
    s : Set α
    hs : Eq (↑w s) 0
    ⊢ Eq (↑(HSMul.hSMul r v) s) 0
  -/
  rw [smul_apply, h hs, smul_zero]
  /-
    🎉 no goals
  -/


theorem map [MeasureSpace β] (h : v ≪ᵥ w) (f : α → β) : v.map f ≪ᵥ w.map f := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝⁴ : AddCommMonoid M
    inst✝³ : TopologicalSpace M
    inst✝² : AddCommMonoid N
    inst✝¹ : TopologicalSpace N
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    inst✝ : MeasureTheory.MeasureSpace β
    h : v.AbsolutelyContinuous w
    f : α → β
    ⊢ (v.map f).AbsolutelyContinuous (w.map f)
  -/
  by_cases hf : Measurable f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      inst✝² : AddCommMonoid N
      inst✝¹ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      inst✝ : MeasureTheory.MeasureSpace β
      h : v.AbsolutelyContinuous w
      f : α → β
      hf : Measurable f
      ⊢ (v.map f).AbsolutelyContinuous (w.map f)
    -/
  · refine mk fun s hs hws => ?_
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      inst✝² : AddCommMonoid N
      inst✝¹ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      inst✝ : MeasureTheory.MeasureSpace β
      h : v.AbsolutelyContinuous w
      f : α → β
      hf : Measurable f
      s : Set β
      hs : MeasurableSet s
      hws : Eq (↑(w.map f) s) 0
      ⊢ Eq (↑(v.map f) s) 0
    -/
    rw [map_apply _ hf hs] at hws ⊢
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      inst✝² : AddCommMonoid N
      inst✝¹ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      inst✝ : MeasureTheory.MeasureSpace β
      h : v.AbsolutelyContinuous w
      f : α → β
      hf : Measurable f
      s : Set β
      hs : MeasurableSet s
      hws : Eq (↑w (Set.preimage f s)) 0
      ⊢ Eq (↑v (Set.preimage f s)) 0
    -/
    exact h hws
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      inst✝² : AddCommMonoid N
      inst✝¹ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      inst✝ : MeasureTheory.MeasureSpace β
      h : v.AbsolutelyContinuous w
      f : α → β
      hf : Not (Measurable f)
      ⊢ (v.map f).AbsolutelyContinuous (w.map f)
    -/
  · intro s _
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝⁴ : AddCommMonoid M
      inst✝³ : TopologicalSpace M
      inst✝² : AddCommMonoid N
      inst✝¹ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      inst✝ : MeasureTheory.MeasureSpace β
      h : v.AbsolutelyContinuous w
      f : α → β
      hf : Not (Measurable f)
      s : Set β
      a✝ : Eq (↑(w.map f) s) 0
      ⊢ Eq (↑(v.map f) s) 0
    -/
    rw [map_not_measurable v hf, zero_apply]
    /-
      🎉 no goals
    -/


theorem ennrealToMeasure {μ : VectorMeasure α ℝ≥0∞} :
    (∀ ⦃s : Set α⦄, μ.ennrealToMeasure s = 0 → v s = 0) ↔ v ≪ᵥ μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    v : MeasureTheory.VectorMeasure α M
    μ : MeasureTheory.VectorMeasure α ENNReal
    ⊢ Iff (∀ ⦃s : Set α⦄, Eq (μ.ennrealToMeasure s) 0 → Eq (↑v s) 0) (v.Absolutely …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : ∀ ⦃s : Set α⦄, Eq (μ.ennrealToMeasure s) 0 → Eq (↑v s) 0
      ⊢ v.AbsolutelyContinuous μ
    -/
  · refine mk fun s hmeas hs => h ?_
    /-
      case mp
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : ∀ ⦃s : Set α⦄, Eq (μ.ennrealToMeasure s) 0 → Eq (↑v s) 0
      s : Set α
      hmeas : MeasurableSet s
      hs : Eq (↑μ s) 0
      ⊢ Eq (μ.ennrealToMeasure s) 0
    -/
    rw [← hs, ennrealToMeasure_apply hmeas]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : v.AbsolutelyContinuous μ
      ⊢ ∀ ⦃s : Set α⦄, Eq (μ.ennrealToMeasure s) 0 → Eq (↑v s) 0
    -/
  · intro s hs
    /-
      case mpr
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : TopologicalSpace M
      v : MeasureTheory.VectorMeasure α M
      μ : MeasureTheory.VectorMeasure α ENNReal
      h : v.AbsolutelyContinuous μ
      s : Set α
      hs : Eq (μ.ennrealToMeasure s) 0
      ⊢ Eq (↑v s) 0
    -/
    by_cases hmeas : MeasurableSet s
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_4
        inst✝¹ : AddCommMonoid M
        inst✝ : TopologicalSpace M
        v : MeasureTheory.VectorMeasure α M
        μ : MeasureTheory.VectorMeasure α ENNReal
        h : v.AbsolutelyContinuous μ
        s : Set α
        hs : Eq (μ.ennrealToMeasure s) 0
        hmeas : MeasurableSet s
        ⊢ Eq (↑v s) 0
      -/
    · rw [ennrealToMeasure_apply hmeas] at hs
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_4
        inst✝¹ : AddCommMonoid M
        inst✝ : TopologicalSpace M
        v : MeasureTheory.VectorMeasure α M
        μ : MeasureTheory.VectorMeasure α ENNReal
        h : v.AbsolutelyContinuous μ
        s : Set α
        hs : Eq (↑μ s) 0
        hmeas : MeasurableSet s
        ⊢ Eq (↑v s) 0
      -/
      exact h hs
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_4
        inst✝¹ : AddCommMonoid M
        inst✝ : TopologicalSpace M
        v : MeasureTheory.VectorMeasure α M
        μ : MeasureTheory.VectorMeasure α ENNReal
        h : v.AbsolutelyContinuous μ
        s : Set α
        hs : Eq (μ.ennrealToMeasure s) 0
        hmeas : Not (MeasurableSet s)
        ⊢ Eq (↑v s) 0
      -/
    · exact not_measurable v hmeas
      /-
        🎉 no goals
      -/


/-- Two vector measures `v` and `w` are said to be mutually singular if there exists a measurable
set `s`, such that for all `t ⊆ s`, `v t = 0` and for all `t ⊆ sᶜ`, `w t = 0`.

We note that we do not require the measurability of `t` in the definition since this makes it easier
to use. This is equivalent to the definition which requires measurability. To prove
`MutuallySingular` with the measurability condition, use
`MeasureTheory.VectorMeasure.MutuallySingular.mk`. -/
def MutuallySingular (v : VectorMeasure α M) (w : VectorMeasure α N) : Prop :=
  ∃ s : Set α, MeasurableSet s ∧ (∀ t ⊆ s, v t = 0) ∧ ∀ t ⊆ sᶜ, w t = 0


@[inherit_doc VectorMeasure.MutuallySingular]
scoped[MeasureTheory] infixl:60 " ⟂ᵥ " => MeasureTheory.VectorMeasure.MutuallySingular


theorem mk (s : Set α) (hs : MeasurableSet s) (h₁ : ∀ t ⊆ s, MeasurableSet t → v t = 0)
    (h₂ : ∀ t ⊆ sᶜ, MeasurableSet t → w t = 0) : v ⟂ᵥ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝³ : AddCommMonoid M
    inst✝² : TopologicalSpace M
    inst✝¹ : AddCommMonoid N
    inst✝ : TopologicalSpace N
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    s : Set α
    hs : MeasurableSet s
    h₁ : ∀ (t : Set α), HasSubset.Subset t s → MeasurableSet t → Eq (↑v t) 0
    h₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl s) → MeasurableSet t →  …
    ⊢ v.MutuallySingular w
  -/
  refine ⟨s, hs, fun t hst => ?_, fun t hst => ?_⟩ <;> by_cases ht : MeasurableSet t
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝³ : AddCommMonoid M
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid N
      inst✝ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      s : Set α
      hs : MeasurableSet s
      h₁ : ∀ (t : Set α), HasSubset.Subset t s → MeasurableSet t → Eq (↑v t) 0
      h₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl s) → MeasurableSet t →  …
      t : Set α
      hst : HasSubset.Subset t s
      ht : MeasurableSet t
      ⊢ Eq (↑v t) 0
    -/
  · exact h₁ t hst ht
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝³ : AddCommMonoid M
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid N
      inst✝ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      s : Set α
      hs : MeasurableSet s
      h₁ : ∀ (t : Set α), HasSubset.Subset t s → MeasurableSet t → Eq (↑v t) 0
      h₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl s) → MeasurableSet t →  …
      t : Set α
      hst : HasSubset.Subset t s
      ht : Not (MeasurableSet t)
      ⊢ Eq (↑v t) 0
    -/
  · exact not_measurable v ht
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝³ : AddCommMonoid M
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid N
      inst✝ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      s : Set α
      hs : MeasurableSet s
      h₁ : ∀ (t : Set α), HasSubset.Subset t s → MeasurableSet t → Eq (↑v t) 0
      h₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl s) → MeasurableSet t →  …
      t : Set α
      hst : HasSubset.Subset t (HasCompl.compl s)
      ht : MeasurableSet t
      ⊢ Eq (↑w t) 0
    -/
  · exact h₂ t hst ht
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝³ : AddCommMonoid M
      inst✝² : TopologicalSpace M
      inst✝¹ : AddCommMonoid N
      inst✝ : TopologicalSpace N
      v : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      s : Set α
      hs : MeasurableSet s
      h₁ : ∀ (t : Set α), HasSubset.Subset t s → MeasurableSet t → Eq (↑v t) 0
      h₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl s) → MeasurableSet t →  …
      t : Set α
      hst : HasSubset.Subset t (HasCompl.compl s)
      ht : Not (MeasurableSet t)
      ⊢ Eq (↑w t) 0
    -/
  · exact not_measurable w ht
    /-
      🎉 no goals
    -/


theorem symm (h : v ⟂ᵥ w) : w ⟂ᵥ v :=
  let ⟨s, hmeas, hs₁, hs₂⟩ := h
  ⟨sᶜ, hmeas.compl, hs₂, fun t ht => hs₁ _ (compl_compl s ▸ ht : t ⊆ s)⟩


theorem zero_right : v ⟂ᵥ (0 : VectorMeasure α N) :=
  ⟨∅, MeasurableSet.empty, fun _ ht => (Set.subset_empty_iff.1 ht).symm ▸ v.empty,
    fun _ _ => zero_apply _⟩


theorem zero_left : (0 : VectorMeasure α M) ⟂ᵥ w :=
  zero_right.symm


theorem add_left [T2Space N] [ContinuousAdd M] (h₁ : v₁ ⟂ᵥ w) (h₂ : v₂ ⟂ᵥ w) : v₁ + v₂ ⟂ᵥ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid N
    inst✝² : TopologicalSpace N
    v₁ v₂ : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    inst✝¹ : T2Space N
    inst✝ : ContinuousAdd M
    h₁ : v₁.MutuallySingular w
    h₂ : v₂.MutuallySingular w
    ⊢ (HAdd.hAdd v₁ v₂).MutuallySingular w
  -/
  obtain ⟨u, hmu, hu₁, hu₂⟩ := h₁
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid N
    inst✝² : TopologicalSpace N
    v₁ v₂ : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    inst✝¹ : T2Space N
    inst✝ : ContinuousAdd M
    h₂ : v₂.MutuallySingular w
    u : Set α
    hmu : MeasurableSet u
    hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
    hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
    ⊢ (HAdd.hAdd v₁ v₂).MutuallySingular w
  -/
  obtain ⟨v, hmv, hv₁, hv₂⟩ := h₂
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    N : Type u_5
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid N
    inst✝² : TopologicalSpace N
    v₁ v₂ : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    inst✝¹ : T2Space N
    inst✝ : ContinuousAdd M
    u : Set α
    hmu : MeasurableSet u
    hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
    hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
    v : Set α
    hmv : MeasurableSet v
    hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
    hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
    ⊢ (HAdd.hAdd v₁ v₂).MutuallySingular w
  -/
  refine mk (u ∩ v) (hmu.inter hmv) (fun t ht _ => ?_) fun t ht hmt => ?_
  · rw [add_apply, hu₁ _ (Set.subset_inter_iff.1 ht).1, hv₁ _ (Set.subset_inter_iff.1 ht).2,
      zero_add]
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      m : MeasurableSpace α
      M : Type u_4
      N : Type u_5
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : TopologicalSpace M
      inst✝³ : AddCommMonoid N
      inst✝² : TopologicalSpace N
      v₁ v₂ : MeasureTheory.VectorMeasure α M
      w : MeasureTheory.VectorMeasure α N
      inst✝¹ : T2Space N
      inst✝ : ContinuousAdd M
      u : Set α
      hmu : MeasurableSet u
      hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
      hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
      v : Set α
      hmv : MeasurableSet v
      hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
      hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
      t : Set α
      ht : HasSubset.Subset t (HasCompl.compl (Inter.inter u v))
      hmt : MeasurableSet t
      ⊢ Eq (↑w t) 0
    -/
  · rw [Set.compl_inter] at ht
    rw [(_ : t = uᶜ ∩ t ∪ vᶜ \ uᶜ ∩ t),
      of_union _ (hmu.compl.inter hmt) ((hmv.compl.diff hmu.compl).inter hmt), hu₂, hv₂, add_zero]
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.a
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_4
        N : Type u_5
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : TopologicalSpace M
        inst✝³ : AddCommMonoid N
        inst✝² : TopologicalSpace N
        v₁ v₂ : MeasureTheory.VectorMeasure α M
        w : MeasureTheory.VectorMeasure α N
        inst✝¹ : T2Space N
        inst✝ : ContinuousAdd M
        u : Set α
        hmu : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
        hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
        v : Set α
        hmv : MeasurableSet v
        hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
        hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
        t : Set α
        ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
        hmt : MeasurableSet t
        ⊢ HasSubset.Subset (Inter.inter (SDiff.sdiff (HasCompl.compl v) (HasCompl.comp …
      -/
    · exact Set.Subset.trans Set.inter_subset_left diff_subset
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.a
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_4
        N : Type u_5
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : TopologicalSpace M
        inst✝³ : AddCommMonoid N
        inst✝² : TopologicalSpace N
        v₁ v₂ : MeasureTheory.VectorMeasure α M
        w : MeasureTheory.VectorMeasure α N
        inst✝¹ : T2Space N
        inst✝ : ContinuousAdd M
        u : Set α
        hmu : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
        hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
        v : Set α
        hmv : MeasurableSet v
        hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
        hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
        t : Set α
        ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
        hmt : MeasurableSet t
        ⊢ HasSubset.Subset (Inter.inter (HasCompl.compl u) t) (HasCompl.compl u)
      -/
    · exact Set.inter_subset_left
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_4
        N : Type u_5
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : TopologicalSpace M
        inst✝³ : AddCommMonoid N
        inst✝² : TopologicalSpace N
        v₁ v₂ : MeasureTheory.VectorMeasure α M
        w : MeasureTheory.VectorMeasure α N
        inst✝¹ : T2Space N
        inst✝ : ContinuousAdd M
        u : Set α
        hmu : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
        hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
        v : Set α
        hmv : MeasurableSet v
        hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
        hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
        t : Set α
        ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
        hmt : MeasurableSet t
        ⊢ Disjoint (Inter.inter (HasCompl.compl u) t) (Inter.inter (SDiff.sdiff (HasCo …
      -/
    · exact disjoint_sdiff_self_right.mono Set.inter_subset_left Set.inter_subset_left
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        m : MeasurableSpace α
        M : Type u_4
        N : Type u_5
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : TopologicalSpace M
        inst✝³ : AddCommMonoid N
        inst✝² : TopologicalSpace N
        v₁ v₂ : MeasureTheory.VectorMeasure α M
        w : MeasureTheory.VectorMeasure α N
        inst✝¹ : T2Space N
        inst✝ : ContinuousAdd M
        u : Set α
        hmu : MeasurableSet u
        hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
        hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
        v : Set α
        hmv : MeasurableSet v
        hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
        hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
        t : Set α
        ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
        hmt : MeasurableSet t
        ⊢ Eq t (Union.union (Inter.inter (HasCompl.compl u) t) (Inter.inter (SDiff.sdi …
      -/
    · apply Set.Subset.antisymm <;> intro x hx
        /-
          case h₁
          α : Type u_1
          m : MeasurableSpace α
          M : Type u_4
          N : Type u_5
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : TopologicalSpace M
          inst✝³ : AddCommMonoid N
          inst✝² : TopologicalSpace N
          v₁ v₂ : MeasureTheory.VectorMeasure α M
          w : MeasureTheory.VectorMeasure α N
          inst✝¹ : T2Space N
          inst✝ : ContinuousAdd M
          u : Set α
          hmu : MeasurableSet u
          hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
          hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
          v : Set α
          hmv : MeasurableSet v
          hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
          hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
          t : Set α
          ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
          hmt : MeasurableSet t
          x : α
          hx : Membership.mem t x
          ⊢ Membership.mem (Union.union (Inter.inter (HasCompl.compl u) t) (Inter.inter  …
        -/
      · by_cases hxu' : x ∈ uᶜ
          /-
            case pos
            α : Type u_1
            m : MeasurableSpace α
            M : Type u_4
            N : Type u_5
            inst✝⁵ : AddCommMonoid M
            inst✝⁴ : TopologicalSpace M
            inst✝³ : AddCommMonoid N
            inst✝² : TopologicalSpace N
            v₁ v₂ : MeasureTheory.VectorMeasure α M
            w : MeasureTheory.VectorMeasure α N
            inst✝¹ : T2Space N
            inst✝ : ContinuousAdd M
            u : Set α
            hmu : MeasurableSet u
            hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
            hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
            v : Set α
            hmv : MeasurableSet v
            hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
            hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
            t : Set α
            ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
            hmt : MeasurableSet t
            x : α
            hx : Membership.mem t x
            hxu' : Membership.mem (HasCompl.compl u) x
            ⊢ Membership.mem (Union.union (Inter.inter (HasCompl.compl u) t) (Inter.inter  …
          -/
        · exact Or.inl ⟨hxu', hx⟩
          /-
            🎉 no goals
          -/
        /-
          case neg
          α : Type u_1
          m : MeasurableSpace α
          M : Type u_4
          N : Type u_5
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : TopologicalSpace M
          inst✝³ : AddCommMonoid N
          inst✝² : TopologicalSpace N
          v₁ v₂ : MeasureTheory.VectorMeasure α M
          w : MeasureTheory.VectorMeasure α N
          inst✝¹ : T2Space N
          inst✝ : ContinuousAdd M
          u : Set α
          hmu : MeasurableSet u
          hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
          hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
          v : Set α
          hmv : MeasurableSet v
          hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
          hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
          t : Set α
          ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
          hmt : MeasurableSet t
          x : α
          hx : Membership.mem t x
          hxu' : Not (Membership.mem (HasCompl.compl u) x)
          ⊢ Membership.mem (Union.union (Inter.inter (HasCompl.compl u) t) (Inter.inter  …
        -/
        rcases ht hx with (hxu | hxv)
        /-
          case neg.inl
          α : Type u_1
          m : MeasurableSpace α
          M : Type u_4
          N : Type u_5
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : TopologicalSpace M
          inst✝³ : AddCommMonoid N
          inst✝² : TopologicalSpace N
          v₁ v₂ : MeasureTheory.VectorMeasure α M
          w : MeasureTheory.VectorMeasure α N
          inst✝¹ : T2Space N
          inst✝ : ContinuousAdd M
          u : Set α
          hmu : MeasurableSet u
          hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
          hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
          v : Set α
          hmv : MeasurableSet v
          hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
          hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
          t : Set α
          ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
          hmt : MeasurableSet t
          x : α
          hx : Membership.mem t x
          hxu' : Not (Membership.mem (HasCompl.compl u) x)
          hxu : Membership.mem (HasCompl.compl u) x
          ⊢ Membership.mem (Union.union (Inter.inter (HasCompl.compl u) t) (Inter.inter  …
        -/
        exacts [False.elim (hxu' hxu), Or.inr ⟨⟨hxv, hxu'⟩, hx⟩]
        /-
          🎉 no goals
        -/
        /-
          case h₂
          α : Type u_1
          m : MeasurableSpace α
          M : Type u_4
          N : Type u_5
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : TopologicalSpace M
          inst✝³ : AddCommMonoid N
          inst✝² : TopologicalSpace N
          v₁ v₂ : MeasureTheory.VectorMeasure α M
          w : MeasureTheory.VectorMeasure α N
          inst✝¹ : T2Space N
          inst✝ : ContinuousAdd M
          u : Set α
          hmu : MeasurableSet u
          hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v₁ t) 0
          hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
          v : Set α
          hmv : MeasurableSet v
          hv₁ : ∀ (t : Set α), HasSubset.Subset t v → Eq (↑v₂ t) 0
          hv₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl v) → Eq (↑w t) 0
          t : Set α
          ht : HasSubset.Subset t (Union.union (HasCompl.compl u) (HasCompl.compl v))
          hmt : MeasurableSet t
          x : α
          hx : Membership.mem (Union.union (Inter.inter (HasCompl.compl u) t) (Inter.int …
          ⊢ Membership.mem t x
        -/
                                 /-
                                   🎉 no goals
                                 -/
      · cases' hx with hx hx <;> exact hx.2
                                 /-
                                   🎉 no goals
                                 -/


theorem add_right [T2Space M] [ContinuousAdd N] (h₁ : v ⟂ᵥ w₁) (h₂ : v ⟂ᵥ w₂) : v ⟂ᵥ w₁ + w₂ :=
  (add_left h₁.symm h₂.symm).symm


theorem smul_right {R : Type*} [Semiring R] [DistribMulAction R N] [ContinuousConstSMul R N]
    (r : R) (h : v ⟂ᵥ w) : v ⟂ᵥ r • w :=
  let ⟨s, hmeas, hs₁, hs₂⟩ := h
                                 /-
                                   α : Type u_1
                                   m : MeasurableSpace α
                                   M : Type u_4
                                   N : Type u_5
                                   inst✝⁶ : AddCommMonoid M
                                   inst✝⁵ : TopologicalSpace M
                                   inst✝⁴ : AddCommMonoid N
                                   inst✝³ : TopologicalSpace N
                                   v : MeasureTheory.VectorMeasure α M
                                   w : MeasureTheory.VectorMeasure α N
                                   R : Type u_6
                                   inst✝² : Semiring R
                                   inst✝¹ : DistribMulAction R N
                                   inst✝ : ContinuousConstSMul R N
                                   r : R
                                   h : v.MutuallySingular w
                                   s : Set α
                                   hmeas : MeasurableSet s
                                   hs₁ : ∀ (t : Set α), HasSubset.Subset t s → Eq (↑v t) 0
                                   hs₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl s) → Eq (↑w t) 0
                                   t : Set α
                                   ht : HasSubset.Subset t (HasCompl.compl s)
                                   ⊢ Eq (↑(HSMul.hSMul r w) t) 0
                                 -/
  ⟨s, hmeas, hs₁, fun t ht => by simp only [coe_smul, Pi.smul_apply, hs₂ t ht, smul_zero]⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem smul_left {R : Type*} [Semiring R] [DistribMulAction R M] [ContinuousConstSMul R M] (r : R)
    (h : v ⟂ᵥ w) : r • v ⟂ᵥ w :=
  (smul_right r h.symm).symm


theorem neg_left {M : Type*} [AddCommGroup M] [TopologicalSpace M] [TopologicalAddGroup M]
    {v : VectorMeasure α M} {w : VectorMeasure α N} (h : v ⟂ᵥ w) : -v ⟂ᵥ w := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    N : Type u_5
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    M : Type u_6
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    h : v.MutuallySingular w
    ⊢ (Neg.neg v).MutuallySingular w
  -/
  obtain ⟨u, hmu, hu₁, hu₂⟩ := h
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    N : Type u_5
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    M : Type u_6
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    u : Set α
    hmu : MeasurableSet u
    hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v t) 0
    hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
    ⊢ (Neg.neg v).MutuallySingular w
  -/
  refine ⟨u, hmu, fun s hs => ?_, hu₂⟩
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    N : Type u_5
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    M : Type u_6
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    u : Set α
    hmu : MeasurableSet u
    hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v t) 0
    hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
    s : Set α
    hs : HasSubset.Subset s u
    ⊢ Eq (↑(Neg.neg v) s) 0
  -/
  rw [neg_apply v s, neg_eq_zero]
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    N : Type u_5
    inst✝⁴ : AddCommMonoid N
    inst✝³ : TopologicalSpace N
    M : Type u_6
    inst✝² : AddCommGroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    v : MeasureTheory.VectorMeasure α M
    w : MeasureTheory.VectorMeasure α N
    u : Set α
    hmu : MeasurableSet u
    hu₁ : ∀ (t : Set α), HasSubset.Subset t u → Eq (↑v t) 0
    hu₂ : ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl u) → Eq (↑w t) 0
    s : Set α
    hs : HasSubset.Subset s u
    ⊢ Eq (↑v s) 0
  -/
  exact hu₁ s hs
  /-
    🎉 no goals
  -/


theorem neg_right {N : Type*} [AddCommGroup N] [TopologicalSpace N] [TopologicalAddGroup N]
    {v : VectorMeasure α M} {w : VectorMeasure α N} (h : v ⟂ᵥ w) : v ⟂ᵥ -w :=
  h.symm.neg_left.symm


@[simp]
theorem neg_left_iff {M : Type*} [AddCommGroup M] [TopologicalSpace M] [TopologicalAddGroup M]
    {v : VectorMeasure α M} {w : VectorMeasure α N} : -v ⟂ᵥ w ↔ v ⟂ᵥ w :=
  ⟨fun h => neg_neg v ▸ h.neg_left, neg_left⟩


@[simp]
theorem neg_right_iff {N : Type*} [AddCommGroup N] [TopologicalSpace N] [TopologicalAddGroup N]
    {v : VectorMeasure α M} {w : VectorMeasure α N} : v ⟂ᵥ -w ↔ v ⟂ᵥ w :=
  ⟨fun h => neg_neg w ▸ h.neg_right, neg_right⟩


open Classical in
/-- Restriction of a vector measure onto a sub-σ-algebra. -/
@[simps]
def trim {m n : MeasurableSpace α} (v : VectorMeasure α M) (hle : m ≤ n) :
    @VectorMeasure α m M _ _ :=
  @VectorMeasure.mk α m M _ _
    (fun i => if MeasurableSet[m] i then v i else 0)
        /-
          α : Type u_1
          β : Type u_2
          m✝ : MeasurableSpace α
          L : Type u_3
          M : Type u_4
          N : Type u_5
          inst✝⁵ : AddCommMonoid L
          inst✝⁴ : TopologicalSpace L
          inst✝³ : AddCommMonoid M
          inst✝² : TopologicalSpace M
          inst✝¹ : AddCommMonoid N
          inst✝ : TopologicalSpace N
          m n : MeasurableSpace α
          v : MeasureTheory.VectorMeasure α M
          hle : LE.le m n
          ⊢ Eq ((fun i => ite (MeasurableSet i) (↑v i) 0) EmptyCollection.emptyCollectio …
        -/
    (by dsimp only; rw [if_pos (@MeasurableSet.empty _ m), v.empty])
                    /-
                      🎉 no goals
                    -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      m✝ : MeasurableSpace α
                      L : Type u_3
                      M : Type u_4
                      N : Type u_5
                      inst✝⁵ : AddCommMonoid L
                      inst✝⁴ : TopologicalSpace L
                      inst✝³ : AddCommMonoid M
                      inst✝² : TopologicalSpace M
                      inst✝¹ : AddCommMonoid N
                      inst✝ : TopologicalSpace N
                      m n : MeasurableSpace α
                      v : MeasureTheory.VectorMeasure α M
                      hle : LE.le m n
                      i : Set α
                      hi : Not (MeasurableSet i)
                      ⊢ Eq ((fun i => ite (MeasurableSet i) (↑v i) 0) i) 0
                    -/
    (fun i hi => by dsimp only; rw [if_neg hi])
                                /-
                                  🎉 no goals
                                -/
    (fun f hf₁ hf₂ => by
      /-
        α : Type u_1
        β : Type u_2
        m✝ : MeasurableSpace α
        L : Type u_3
        M : Type u_4
        N : Type u_5
        inst✝⁵ : AddCommMonoid L
        inst✝⁴ : TopologicalSpace L
        inst✝³ : AddCommMonoid M
        inst✝² : TopologicalSpace M
        inst✝¹ : AddCommMonoid N
        inst✝ : TopologicalSpace N
        m n : MeasurableSpace α
        v : MeasureTheory.VectorMeasure α M
        hle : LE.le m n
        f : Nat → Set α
        hf₁ : ∀ (i : Nat), MeasurableSet (f i)
        hf₂ : Pairwise (Function.onFun Disjoint f)
        ⊢ HasSum (fun i => (fun i => ite (MeasurableSet i) (↑v i) 0) (f i)) ((fun i => …
      -/
      dsimp only
      /-
        α : Type u_1
        β : Type u_2
        m✝ : MeasurableSpace α
        L : Type u_3
        M : Type u_4
        N : Type u_5
        inst✝⁵ : AddCommMonoid L
        inst✝⁴ : TopologicalSpace L
        inst✝³ : AddCommMonoid M
        inst✝² : TopologicalSpace M
        inst✝¹ : AddCommMonoid N
        inst✝ : TopologicalSpace N
        m n : MeasurableSpace α
        v : MeasureTheory.VectorMeasure α M
        hle : LE.le m n
        f : Nat → Set α
        hf₁ : ∀ (i : Nat), MeasurableSet (f i)
        hf₂ : Pairwise (Function.onFun Disjoint f)
        ⊢ HasSum (fun i => ite (MeasurableSet (f i)) (↑v (f i)) 0) (ite (MeasurableSet …
      -/
      have hf₁' : ∀ k, MeasurableSet[n] (f k) := fun k => hle _ (hf₁ k)
      /-
        α : Type u_1
        β : Type u_2
        m✝ : MeasurableSpace α
        L : Type u_3
        M : Type u_4
        N : Type u_5
        inst✝⁵ : AddCommMonoid L
        inst✝⁴ : TopologicalSpace L
        inst✝³ : AddCommMonoid M
        inst✝² : TopologicalSpace M
        inst✝¹ : AddCommMonoid N
        inst✝ : TopologicalSpace N
        m n : MeasurableSpace α
        v : MeasureTheory.VectorMeasure α M
        hle : LE.le m n
        f : Nat → Set α
        hf₁ : ∀ (i : Nat), MeasurableSet (f i)
        hf₂ : Pairwise (Function.onFun Disjoint f)
        hf₁' : ∀ (k : Nat), MeasurableSet (f k)
        ⊢ HasSum (fun i => ite (MeasurableSet (f i)) (↑v (f i)) 0) (ite (MeasurableSet …
      -/
      convert v.m_iUnion hf₁' hf₂ using 1
        /-
          case h.e'_5
          α : Type u_1
          β : Type u_2
          m✝ : MeasurableSpace α
          L : Type u_3
          M : Type u_4
          N : Type u_5
          inst✝⁵ : AddCommMonoid L
          inst✝⁴ : TopologicalSpace L
          inst✝³ : AddCommMonoid M
          inst✝² : TopologicalSpace M
          inst✝¹ : AddCommMonoid N
          inst✝ : TopologicalSpace N
          m n : MeasurableSpace α
          v : MeasureTheory.VectorMeasure α M
          hle : LE.le m n
          f : Nat → Set α
          hf₁ : ∀ (i : Nat), MeasurableSet (f i)
          hf₂ : Pairwise (Function.onFun Disjoint f)
          hf₁' : ∀ (k : Nat), MeasurableSet (f k)
          ⊢ Eq (fun i => ite (MeasurableSet (f i)) (↑v (f i)) 0) fun i => ↑v (f i)
        -/
      · ext n
        /-
          case h.e'_5.h
          α : Type u_1
          β : Type u_2
          m✝ : MeasurableSpace α
          L : Type u_3
          M : Type u_4
          N : Type u_5
          inst✝⁵ : AddCommMonoid L
          inst✝⁴ : TopologicalSpace L
          inst✝³ : AddCommMonoid M
          inst✝² : TopologicalSpace M
          inst✝¹ : AddCommMonoid N
          inst✝ : TopologicalSpace N
          m n✝ : MeasurableSpace α
          v : MeasureTheory.VectorMeasure α M
          hle : LE.le m n✝
          f : Nat → Set α
          hf₁ : ∀ (i : Nat), MeasurableSet (f i)
          hf₂ : Pairwise (Function.onFun Disjoint f)
          hf₁' : ∀ (k : Nat), MeasurableSet (f k)
          n : Nat
          ⊢ Eq (ite (MeasurableSet (f n)) (↑v (f n)) 0) (↑v (f n))
        -/
        rw [if_pos (hf₁ n)]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_6
          α : Type u_1
          β : Type u_2
          m✝ : MeasurableSpace α
          L : Type u_3
          M : Type u_4
          N : Type u_5
          inst✝⁵ : AddCommMonoid L
          inst✝⁴ : TopologicalSpace L
          inst✝³ : AddCommMonoid M
          inst✝² : TopologicalSpace M
          inst✝¹ : AddCommMonoid N
          inst✝ : TopologicalSpace N
          m n : MeasurableSpace α
          v : MeasureTheory.VectorMeasure α M
          hle : LE.le m n
          f : Nat → Set α
          hf₁ : ∀ (i : Nat), MeasurableSet (f i)
          hf₂ : Pairwise (Function.onFun Disjoint f)
          hf₁' : ∀ (k : Nat), MeasurableSet (f k)
          ⊢ Eq (ite (MeasurableSet (Set.iUnion fun i => f i)) (↑v (Set.iUnion fun i => f …
        -/
      · rw [if_pos (@MeasurableSet.iUnion _ _ m _ _ hf₁)])
        /-
          🎉 no goals
        -/


theorem trim_eq_self : v.trim le_rfl = v := by
  /-
    α : Type u_1
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    n : MeasurableSpace α
    v : MeasureTheory.VectorMeasure α M
    ⊢ Eq (v.trim ⋯) v
  -/
  ext i hi
  /-
    case h
    α : Type u_1
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    n : MeasurableSpace α
    v : MeasureTheory.VectorMeasure α M
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (↑(v.trim ⋯) i) (↑v i)
  -/
  exact if_pos hi
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_trim (hle : m ≤ n) : (0 : VectorMeasure α M).trim hle = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    n : MeasurableSpace α
    hle : LE.le m n
    ⊢ Eq (MeasureTheory.VectorMeasure.trim 0 hle) 0
  -/
  ext i hi
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    n : MeasurableSpace α
    hle : LE.le m n
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (↑(MeasureTheory.VectorMeasure.trim 0 hle) i) (↑0 i)
  -/
  exact if_pos hi
  /-
    🎉 no goals
  -/


theorem trim_measurableSet_eq (hle : m ≤ n) {i : Set α} (hi : MeasurableSet[m] i) :
    v.trim hle i = v i :=
  if_pos hi


theorem restrict_trim (hle : m ≤ n) {i : Set α} (hi : MeasurableSet[m] i) :
    @VectorMeasure.restrict α m M _ _ (v.trim hle) i = (v.restrict i).trim hle := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    n : MeasurableSpace α
    v : MeasureTheory.VectorMeasure α M
    hle : LE.le m n
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq ((v.trim hle).restrict i) ((v.restrict i).trim hle)
  -/
  ext j hj
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    n : MeasurableSpace α
    v : MeasureTheory.VectorMeasure α M
    hle : LE.le m n
    i : Set α
    hi : MeasurableSet i
    j : Set α
    hj : MeasurableSet j
    ⊢ Eq (↑((v.trim hle).restrict i) j) (↑((v.restrict i).trim hle) j)
  -/
  rw [@restrict_apply _ m, trim_measurableSet_eq hle hj, restrict_apply, trim_measurableSet_eq]
  /-
    case h.hi
    α : Type u_1
    m : MeasurableSpace α
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : TopologicalSpace M
    n : MeasurableSpace α
    v : MeasureTheory.VectorMeasure α M
    hle : LE.le m n
    i : Set α
    hi : MeasurableSet i
    j : Set α
    hj : MeasurableSet j
    ⊢ MeasurableSet (Inter.inter j i)
  -/
  all_goals measurability
  /-
    🎉 no goals
  -/


/-- The underlying function for `SignedMeasure.toMeasureOfZeroLE`. -/
def toMeasureOfZeroLE' (s : SignedMeasure α) (i : Set α) (hi : 0 ≤[i] s) (j : Set α)
    (hj : MeasurableSet j) : ℝ≥0∞ :=
                                                   /-
                                                     α : Type u_1
                                                     β : Type u_2
                                                     m : MeasurableSpace α
                                                     s : MeasureTheory.SignedMeasure α
                                                     i : Set α
                                                     hi : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMea …
                                                     j : Set α
                                                     hj : MeasurableSet j
                                                     ⊢ LE.le 0 (↑(MeasureTheory.VectorMeasure.restrict 0 i) j)
                                                   -/
  ((↑) : ℝ≥0 → ℝ≥0∞) ⟨s.restrict i j, le_trans (by simp) (hi j hj)⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Given a signed measure `s` and a positive measurable set `i`, `toMeasureOfZeroLE`
provides the measure, mapping measurable sets `j` to `s (i ∩ j)`. -/
def toMeasureOfZeroLE (s : SignedMeasure α) (i : Set α) (hi₁ : MeasurableSet i) (hi₂ : 0 ≤[i] s) :
    Measure α := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i : Set α
    hi₁ : MeasurableSet i
    hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
    ⊢ MeasureTheory.Measure α
  -/
  refine Measure.ofMeasurable (s.toMeasureOfZeroLE' i hi₂) ?_ ?_
  · simp_rw [toMeasureOfZeroLE', s.restrict_apply hi₁ MeasurableSet.empty, Set.empty_inter i,
      s.empty]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
      ⊢ Eq (↑⟨0, ⋯⟩) 0
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
      ⊢ ∀ ⦃f : Nat → Set α⦄ (h : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Functi …
    -/
  · intro f hf₁ hf₂
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
      f : Nat → Set α
      hf₁ : ∀ (i : Nat), MeasurableSet (f i)
      hf₂ : Pairwise (Function.onFun Disjoint f)
      ⊢ Eq (s.toMeasureOfZeroLE' i hi₂ (Set.iUnion fun i => f i) ⋯) (tsum fun i_1 => …
    -/
    have h₁ : ∀ n, MeasurableSet (i ∩ f n) := fun n => hi₁.inter (hf₁ n)
    have h₂ : Pairwise (Disjoint on fun n : ℕ => i ∩ f n) := by
      intro n m hnm
      exact ((hf₂ hnm).inf_left' i).inf_right' i
    simp only [toMeasureOfZeroLE', s.restrict_apply hi₁ (MeasurableSet.iUnion hf₁), Set.inter_comm,
      Set.inter_iUnion, s.of_disjoint_iUnion h₁ h₂, ENNReal.some_eq_coe, id]
    have h : ∀ n, 0 ≤ s (i ∩ f n) := fun n =>
      s.nonneg_of_zero_le_restrict (s.zero_le_restrict_subset hi₁ Set.inter_subset_left hi₂)
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i : Set α
      hi₁ : MeasurableSet i
      hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
      f : Nat → Set α
      hf₁ : ∀ (i : Nat), MeasurableSet (f i)
      hf₂ : Pairwise (Function.onFun Disjoint f)
      h₁ : ∀ (n : Nat), MeasurableSet (Inter.inter i (f n))
      h₂ : Pairwise (Function.onFun Disjoint fun n => Inter.inter i (f n))
      h : ∀ (n : Nat), LE.le 0 (↑s (Inter.inter i (f n)))
      ⊢ Eq (↑⟨tsum fun i_1 => ↑s (Inter.inter i (f i_1)), ⋯⟩) (tsum fun i_1 => ↑⟨↑(M …
    -/
    rw [NNReal.coe_tsum_of_nonneg h, ENNReal.coe_tsum]
      /-
        case refine_2
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
        f : Nat → Set α
        hf₁ : ∀ (i : Nat), MeasurableSet (f i)
        hf₂ : Pairwise (Function.onFun Disjoint f)
        h₁ : ∀ (n : Nat), MeasurableSet (Inter.inter i (f n))
        h₂ : Pairwise (Function.onFun Disjoint fun n => Inter.inter i (f n))
        h : ∀ (n : Nat), LE.le 0 (↑s (Inter.inter i (f n)))
        ⊢ Eq (tsum fun a => ↑⟨↑s (Inter.inter i (f a)), ⋯⟩) (tsum fun i_1 => ↑⟨↑(Measu …
      -/
    · refine tsum_congr fun n => ?_
      /-
        case refine_2
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
        f : Nat → Set α
        hf₁ : ∀ (i : Nat), MeasurableSet (f i)
        hf₂ : Pairwise (Function.onFun Disjoint f)
        h₁ : ∀ (n : Nat), MeasurableSet (Inter.inter i (f n))
        h₂ : Pairwise (Function.onFun Disjoint fun n => Inter.inter i (f n))
        h : ∀ (n : Nat), LE.le 0 (↑s (Inter.inter i (f n)))
        n : Nat
        ⊢ Eq ↑⟨↑s (Inter.inter i (f n)), ⋯⟩ ↑⟨↑(MeasureTheory.VectorMeasure.restrict s …
      -/
      simp_rw [s.restrict_apply hi₁ (hf₁ n), Set.inter_comm]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        s : MeasureTheory.SignedMeasure α
        i : Set α
        hi₁ : MeasurableSet i
        hi₂ : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMe …
        f : Nat → Set α
        hf₁ : ∀ (i : Nat), MeasurableSet (f i)
        hf₂ : Pairwise (Function.onFun Disjoint f)
        h₁ : ∀ (n : Nat), MeasurableSet (Inter.inter i (f n))
        h₂ : Pairwise (Function.onFun Disjoint fun n => Inter.inter i (f n))
        h : ∀ (n : Nat), LE.le 0 (↑s (Inter.inter i (f n)))
        ⊢ Summable fun n => ⟨↑s (Inter.inter i (f n)), ⋯⟩
      -/
    · exact (NNReal.summable_mk h).2 (s.m_iUnion h₁ h₂).summable
      /-
        🎉 no goals
      -/


theorem toMeasureOfZeroLE_apply (hi : 0 ≤[i] s) (hi₁ : MeasurableSet i) (hj₁ : MeasurableSet j) :
    s.toMeasureOfZeroLE i hi₁ hi j = ((↑) : ℝ≥0 → ℝ≥0∞) ⟨s (i ∩ j), nonneg_of_zero_le_restrict
      s (zero_le_restrict_subset s hi₁ Set.inter_subset_left hi)⟩ := by
  simp_rw [toMeasureOfZeroLE, Measure.ofMeasurable_apply _ hj₁, toMeasureOfZeroLE',
    s.restrict_apply hi₁ hj₁, Set.inter_comm]


/-- Given a signed measure `s` and a negative measurable set `i`, `toMeasureOfLEZero`
provides the measure, mapping measurable sets `j` to `-s (i ∩ j)`. -/
def toMeasureOfLEZero (s : SignedMeasure α) (i : Set α) (hi₁ : MeasurableSet i) (hi₂ : s ≤[i] 0) :
    Measure α :=
  toMeasureOfZeroLE (-s) i hi₁ <| @neg_zero (VectorMeasure α ℝ) _ ▸ neg_le_neg _ _ hi₁ hi₂


theorem toMeasureOfLEZero_apply (hi : s ≤[i] 0) (hi₁ : MeasurableSet i) (hj₁ : MeasurableSet j) :
    s.toMeasureOfLEZero i hi₁ hi j = ((↑) : ℝ≥0 → ℝ≥0∞) ⟨-s (i ∩ j), neg_apply s (i ∩ j) ▸
      nonneg_of_zero_le_restrict _ (zero_le_restrict_subset _ hi₁ Set.inter_subset_left
      (@neg_zero (VectorMeasure α ℝ) _ ▸ neg_le_neg _ _ hi₁ hi))⟩ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    i j : Set α
    hi : LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.VectorMea …
    hi₁ : MeasurableSet i
    hj₁ : MeasurableSet j
    ⊢ Eq ((s.toMeasureOfLEZero i hi₁ hi) j) ↑⟨Neg.neg (↑s (Inter.inter i j)), ⋯⟩
  -/
  erw [toMeasureOfZeroLE_apply]
    /-
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i j : Set α
      hi : LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.VectorMea …
      hi₁ : MeasurableSet i
      hj₁ : MeasurableSet j
      ⊢ Eq ↑⟨↑(Neg.neg s) (Inter.inter i j), ⋯⟩ ↑⟨Neg.neg (↑s (Inter.inter i j)), ⋯⟩
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hj₁
      α : Type u_1
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i j : Set α
      hi : LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.VectorMea …
      hi₁ : MeasurableSet i
      hj₁ : MeasurableSet j
      ⊢ MeasurableSet j
    -/
  · assumption
    /-
      🎉 no goals
    -/


/-- `SignedMeasure.toMeasureOfZeroLE` is a finite measure. -/
instance toMeasureOfZeroLE_finite (hi : 0 ≤[i] s) (hi₁ : MeasurableSet i) :
    IsFiniteMeasure (s.toMeasureOfZeroLE i hi₁ hi) where
  measure_univ_lt_top := by
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i j : Set α
      hi : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMea …
      hi₁ : MeasurableSet i
      ⊢ LT.lt ((s.toMeasureOfZeroLE i hi₁ hi) Set.univ) Top.top
    -/
    rw [toMeasureOfZeroLE_apply s hi hi₁ MeasurableSet.univ]
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i j : Set α
      hi : LE.le (MeasureTheory.VectorMeasure.restrict 0 i) (MeasureTheory.VectorMea …
      hi₁ : MeasurableSet i
      ⊢ LT.lt (↑⟨↑s (Inter.inter i Set.univ), ⋯⟩) Top.top
    -/
    exact ENNReal.coe_lt_top
    /-
      🎉 no goals
    -/


/-- `SignedMeasure.toMeasureOfLEZero` is a finite measure. -/
instance toMeasureOfLEZero_finite (hi : s ≤[i] 0) (hi₁ : MeasurableSet i) :
    IsFiniteMeasure (s.toMeasureOfLEZero i hi₁ hi) where
  measure_univ_lt_top := by
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i j : Set α
      hi : LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.VectorMea …
      hi₁ : MeasurableSet i
      ⊢ LT.lt ((s.toMeasureOfLEZero i hi₁ hi) Set.univ) Top.top
    -/
    rw [toMeasureOfLEZero_apply s hi hi₁ MeasurableSet.univ]
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      s : MeasureTheory.SignedMeasure α
      i j : Set α
      hi : LE.le (MeasureTheory.VectorMeasure.restrict s i) (MeasureTheory.VectorMea …
      hi₁ : MeasurableSet i
      ⊢ LT.lt (↑⟨Neg.neg (↑s (Inter.inter i Set.univ)), ⋯⟩) Top.top
    -/
    exact ENNReal.coe_lt_top
    /-
      🎉 no goals
    -/


theorem toMeasureOfZeroLE_toSignedMeasure (hs : 0 ≤[Set.univ] s) :
    (s.toMeasureOfZeroLE Set.univ MeasurableSet.univ hs).toSignedMeasure = s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    hs : LE.le (MeasureTheory.VectorMeasure.restrict 0 Set.univ) (MeasureTheory.Ve …
    ⊢ Eq (s.toMeasureOfZeroLE Set.univ ⋯ hs).toSignedMeasure s
  -/
  ext i hi
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    hs : LE.le (MeasureTheory.VectorMeasure.restrict 0 Set.univ) (MeasureTheory.Ve …
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (↑(s.toMeasureOfZeroLE Set.univ ⋯ hs).toSignedMeasure i) (↑s i)
  -/
  simp [hi, toMeasureOfZeroLE_apply _ _ _ hi]
  /-
    🎉 no goals
  -/


theorem toMeasureOfLEZero_toSignedMeasure (hs : s ≤[Set.univ] 0) :
    (s.toMeasureOfLEZero Set.univ MeasurableSet.univ hs).toSignedMeasure = -s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    hs : LE.le (MeasureTheory.VectorMeasure.restrict s Set.univ) (MeasureTheory.Ve …
    ⊢ Eq (s.toMeasureOfLEZero Set.univ ⋯ hs).toSignedMeasure (Neg.neg s)
  -/
  ext i hi
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    s : MeasureTheory.SignedMeasure α
    hs : LE.le (MeasureTheory.VectorMeasure.restrict s Set.univ) (MeasureTheory.Ve …
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq (↑(s.toMeasureOfLEZero Set.univ ⋯ hs).toSignedMeasure i) (↑(Neg.neg s) i)
  -/
  simp [hi, toMeasureOfLEZero_apply _ _ _ hi]
  /-
    🎉 no goals
  -/


theorem zero_le_toSignedMeasure : 0 ≤ μ.toSignedMeasure := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LE.le 0 μ.toSignedMeasure
  -/
  rw [← le_restrict_univ_iff_le]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LE.le (MeasureTheory.VectorMeasure.restrict 0 Set.univ) (MeasureTheory.Vecto …
  -/
  refine restrict_le_restrict_of_subset_le _ _ fun j hj₁ _ => ?_
  simp only [Measure.toSignedMeasure_apply_measurable hj₁, coe_zero, Pi.zero_apply,
    ENNReal.toReal_nonneg, VectorMeasure.coe_zero]


theorem toSignedMeasure_toMeasureOfZeroLE :
    μ.toSignedMeasure.toMeasureOfZeroLE Set.univ MeasurableSet.univ
      ((le_restrict_univ_iff_le _ _).2 (zero_le_toSignedMeasure μ)) = μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ Eq (μ.toSignedMeasure.toMeasureOfZeroLE Set.univ ⋯ ⋯) μ
  -/
  refine Measure.ext fun i hi => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    i : Set α
    hi : MeasurableSet i
    ⊢ Eq ((μ.toSignedMeasure.toMeasureOfZeroLE Set.univ ⋯ ⋯) i) (μ i)
  -/
  lift μ i to ℝ≥0 using (measure_lt_top _ _).ne with m hm
  /-
    case intro
    α : Type u_1
    m✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    i : Set α
    hi : MeasurableSet i
    m : NNReal
    hm : Eq (↑m) (μ i)
    ⊢ Eq ((μ.toSignedMeasure.toMeasureOfZeroLE Set.univ ⋯ ⋯) i) ↑m
  -/
  rw [SignedMeasure.toMeasureOfZeroLE_apply _ _ _ hi, ENNReal.coe_inj]
  /-
    case intro
    α : Type u_1
    m✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    i : Set α
    hi : MeasurableSet i
    m : NNReal
    hm : Eq (↑m) (μ i)
    ⊢ Eq ⟨↑μ.toSignedMeasure (Inter.inter Set.univ i), ⋯⟩ m
  -/
  congr
  /-
    case intro.e_val
    α : Type u_1
    m✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    i : Set α
    hi : MeasurableSet i
    m : NNReal
    hm : Eq (↑m) (μ i)
    ⊢ Eq (↑μ.toSignedMeasure (Inter.inter Set.univ i)) ↑m
  -/
  simp [hi, ← hm]
  /-
    🎉 no goals
  -/


