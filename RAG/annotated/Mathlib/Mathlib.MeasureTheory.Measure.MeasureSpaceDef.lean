/-- A measure is defined to be an outer measure that is countably additive on
measurable sets, with the additional assumption that the outer measure is the canonical
extension of the restricted measure. -/
structure Measure (α : Type*) [MeasurableSpace α] extends OuterMeasure α where
  m_iUnion ⦃f : ℕ → Set α⦄ : (∀ i, MeasurableSet (f i)) → Pairwise (Disjoint on f) →
    toOuterMeasure (⋃ i, f i) = ∑' i, toOuterMeasure (f i)
  trim_le : toOuterMeasure.trim ≤ toOuterMeasure


/-- Notation for `Measure` with respect to a non-standard σ-algebra in the domain. -/
scoped notation "Measure[" mα "]" α:arg => @Measure α mα


theorem Measure.toOuterMeasure_injective [MeasurableSpace α] :
    Injective (toOuterMeasure : Measure α → OuterMeasure α)
  | ⟨_, _, _⟩, ⟨_, _, _⟩, rfl => rfl


instance Measure.instFunLike [MeasurableSpace α] : FunLike (Measure α) (Set α) ℝ≥0∞ where
  coe μ := μ.toOuterMeasure
  coe_injective' | ⟨_, _, _⟩, ⟨_, _, _⟩, h => toOuterMeasure_injective <| DFunLike.coe_injective h



instance Measure.instOuterMeasureClass [MeasurableSpace α] : OuterMeasureClass (Measure α) α where
  measure_empty m := measure_empty (μ := m.toOuterMeasure)
  measure_iUnion_nat_le m := m.iUnion_nat
  measure_mono m := m.mono


theorem trimmed (μ : Measure α) : μ.toOuterMeasure.trim = μ.toOuterMeasure :=
  le_antisymm μ.trim_le μ.1.le_trim


/-- Obtain a measure by giving a countably additive function that sends `∅` to `0`. -/
def ofMeasurable (m : ∀ s : Set α, MeasurableSet s → ℝ≥0∞) (m0 : m ∅ MeasurableSet.empty = 0)
    (mU :
      ∀ ⦃f : ℕ → Set α⦄ (h : ∀ i, MeasurableSet (f i)),
        Pairwise (Disjoint on f) → m (⋃ i, f i) (MeasurableSet.iUnion h) = ∑' i, m (f i) (h i)) :
    Measure α :=
  { toOuterMeasure := inducedOuterMeasure m _ m0
    m_iUnion := fun f hf hd =>
      show inducedOuterMeasure m _ m0 (iUnion f) = ∑' i, inducedOuterMeasure m _ m0 (f i) by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          δ : Type u_4
          ι : Sort u_5
          inst✝ : MeasurableSpace α
          μ μ₁ μ₂ : MeasureTheory.Measure α
          s s₁ s₂ t : Set α
          m : (s : Set α) → MeasurableSet s → ENNReal
          m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
          mU : ∀ ⦃f : Nat → Set α⦄ (h : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fun …
          f : Nat → Set α
          hf : ∀ (i : Nat), MeasurableSet (f i)
          hd : Pairwise (Function.onFun Disjoint f)
          ⊢ Eq ((MeasureTheory.inducedOuterMeasure m ⋯ m0) (Set.iUnion f)) (tsum fun i = …
        -/
        rw [inducedOuterMeasure_eq m0 mU, mU hf hd]
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          δ : Type u_4
          ι : Sort u_5
          inst✝ : MeasurableSpace α
          μ μ₁ μ₂ : MeasureTheory.Measure α
          s s₁ s₂ t : Set α
          m : (s : Set α) → MeasurableSet s → ENNReal
          m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
          mU : ∀ ⦃f : Nat → Set α⦄ (h : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fun …
          f : Nat → Set α
          hf : ∀ (i : Nat), MeasurableSet (f i)
          hd : Pairwise (Function.onFun Disjoint f)
          ⊢ Eq (tsum fun i => m (f i) ⋯) (tsum fun i => (MeasureTheory.inducedOuterMeasu …
        -/
        congr; funext n; rw [inducedOuterMeasure_eq m0 mU]
                         /-
                           🎉 no goals
                         -/
    trim_le := le_inducedOuterMeasure.2 fun s hs ↦ by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ι : Sort u_5
        inst✝ : MeasurableSpace α
        μ μ₁ μ₂ : MeasureTheory.Measure α
        s✝ s₁ s₂ t : Set α
        m : (s : Set α) → MeasurableSet s → ENNReal
        m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
        mU : ∀ ⦃f : Nat → Set α⦄ (h : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fun …
        s : Set α
        hs : MeasurableSet s
        ⊢ LE.le ((MeasureTheory.inducedOuterMeasure m ⋯ m0).trim s) (m s hs)
      -/
      rw [OuterMeasure.trim_eq _ hs, inducedOuterMeasure_eq m0 mU hs] }
      /-
        🎉 no goals
      -/


theorem ofMeasurable_apply {m : ∀ s : Set α, MeasurableSet s → ℝ≥0∞}
    {m0 : m ∅ MeasurableSet.empty = 0}
    {mU :
      ∀ ⦃f : ℕ → Set α⦄ (h : ∀ i, MeasurableSet (f i)),
        Pairwise (Disjoint on f) → m (⋃ i, f i) (MeasurableSet.iUnion h) = ∑' i, m (f i) (h i)}
    (s : Set α) (hs : MeasurableSet s) : ofMeasurable m m0 mU s = m s hs :=
  inducedOuterMeasure_eq m0 mU hs


@[ext]
theorem ext (h : ∀ s, MeasurableSet s → μ₁ s = μ₂ s) : μ₁ = μ₂ :=
  toOuterMeasure_injective <| by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ₁ μ₂ : MeasureTheory.Measure α
    h : ∀ (s : Set α), MeasurableSet s → Eq (μ₁ s) (μ₂ s)
    ⊢ Eq μ₁.toOuterMeasure μ₂.toOuterMeasure
  -/
  rw [← trimmed, OuterMeasure.trim_congr (h _), trimmed]
  /-
    🎉 no goals
  -/


theorem ext_iff' : μ₁ = μ₂ ↔ ∀ s, μ₁ s = μ₂ s :=
      /-
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ₁ μ₂ : MeasureTheory.Measure α
        ⊢ Eq μ₁ μ₂ → ∀ (s : Set α), Eq (μ₁ s) (μ₂ s)
      -/
  ⟨by rintro rfl s; rfl, fun h ↦ Measure.ext (fun s _ ↦ h s)⟩
                    /-
                      🎉 no goals
                    -/


theorem outerMeasure_le_iff {m : OuterMeasure α} : m ≤ μ.1 ↔ ∀ s, MeasurableSet s → m s ≤ μ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    m : MeasureTheory.OuterMeasure α
    ⊢ Iff (LE.le m μ.toOuterMeasure) (∀ (s : Set α), MeasurableSet s → LE.le (m s) …
  -/
  simpa only [μ.trimmed] using OuterMeasure.le_trim_iff (m₂ := μ.1)
  /-
    🎉 no goals
  -/


@[simp] theorem Measure.coe_toOuterMeasure (μ : Measure α) : ⇑μ.toOuterMeasure = μ := rfl


theorem Measure.toOuterMeasure_apply (μ : Measure α) (s : Set α) :
    μ.toOuterMeasure s = μ s :=
  rfl


theorem measure_eq_trim (s : Set α) : μ s = μ.toOuterMeasure.trim s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Eq (μ s) (μ.trim s)
  -/
  rw [μ.trimmed, μ.coe_toOuterMeasure]
  /-
    🎉 no goals
  -/


theorem measure_eq_iInf (s : Set α) : μ s = ⨅ (t) (_ : s ⊆ t) (_ : MeasurableSet t), μ t := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Eq (μ s) (iInf fun t => iInf fun x => iInf fun x => μ t)
  -/
  rw [measure_eq_trim, OuterMeasure.trim_eq_iInf, μ.coe_toOuterMeasure]
  /-
    🎉 no goals
  -/


/-- A variant of `measure_eq_iInf` which has a single `iInf`. This is useful when applying a
  lemma next that only works for non-empty infima, in which case you can use
  `nonempty_measurable_superset`. -/
theorem measure_eq_iInf' (μ : Measure α) (s : Set α) :
    μ s = ⨅ t : { t // s ⊆ t ∧ MeasurableSet t }, μ t := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Eq (μ s) (iInf fun t => μ ↑t)
  -/
  simp_rw [iInf_subtype, iInf_and, ← measure_eq_iInf]
  /-
    🎉 no goals
  -/


theorem measure_eq_inducedOuterMeasure :
    μ s = inducedOuterMeasure (fun s _ => μ s) MeasurableSet.empty μ.empty s :=
  measure_eq_trim _


theorem toOuterMeasure_eq_inducedOuterMeasure :
    μ.toOuterMeasure = inducedOuterMeasure (fun s _ => μ s) MeasurableSet.empty μ.empty :=
  μ.trimmed.symm


theorem measure_eq_extend (hs : MeasurableSet s) :
    μ s = extend (fun t (_ht : MeasurableSet t) => μ t) s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (μ s) (MeasureTheory.extend (fun t _ht => μ t) s)
  -/
  rw [extend_eq]
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    ⊢ MeasurableSet s
  -/
  exact hs
  /-
    🎉 no goals
  -/


theorem nonempty_of_measure_ne_zero (h : μ s ≠ 0) : s.Nonempty :=
  nonempty_iff_ne_empty.2 fun h' => h <| h'.symm ▸ measure_empty


theorem measure_mono_top (h : s₁ ⊆ s₂) (h₁ : μ s₁ = ∞) : μ s₂ = ∞ :=
  top_unique <| h₁ ▸ measure_mono h


@[simp, mono]
theorem measure_le_measure_union_left : μ s ≤ μ (s ∪ t) := μ.mono subset_union_left


@[simp, mono]
theorem measure_le_measure_union_right : μ t ≤ μ (s ∪ t) := μ.mono subset_union_right


/-- For every set there exists a measurable superset of the same measure. -/
theorem exists_measurable_superset (μ : Measure α) (s : Set α) :
    ∃ t, s ⊆ t ∧ MeasurableSet t ∧ μ t = μ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (μ t)  …
  -/
  simpa only [← measure_eq_trim] using μ.toOuterMeasure.exists_measurable_superset_eq_trim s
  /-
    🎉 no goals
  -/


/-- For every set `s` and a countable collection of measures `μ i` there exists a measurable
superset `t ⊇ s` such that each measure `μ i` takes the same value on `s` and `t`. -/
theorem exists_measurable_superset_forall_eq [Countable ι] (μ : ι → Measure α) (s : Set α) :
    ∃ t, s ⊆ t ∧ MeasurableSet t ∧ ∀ i, μ i t = μ i s := by
  simpa only [← measure_eq_trim] using
    OuterMeasure.exists_measurable_superset_forall_eq_trim (fun i => (μ i).toOuterMeasure) s


theorem exists_measurable_superset₂ (μ ν : Measure α) (s : Set α) :
    ∃ t, s ⊆ t ∧ MeasurableSet t ∧ μ t = μ s ∧ ν t = ν s := by
  simpa only [Bool.forall_bool.trans and_comm] using
    exists_measurable_superset_forall_eq (fun b => cond b μ ν) s


theorem exists_measurable_superset_of_null (h : μ s = 0) : ∃ t, s ⊆ t ∧ MeasurableSet t ∧ μ t = 0 :=
  h ▸ exists_measurable_superset μ s


theorem exists_measurable_superset_iff_measure_eq_zero :
    (∃ t, s ⊆ t ∧ MeasurableSet t ∧ μ t = 0) ↔ μ s = 0 :=
  ⟨fun ⟨_t, hst, _, ht⟩ => measure_mono_null hst ht, exists_measurable_superset_of_null⟩


theorem measure_biUnion_lt_top {s : Set β} {f : β → Set α} (hs : s.Finite)
    (hfin : ∀ i ∈ s, μ (f i) < ∞) : μ (⋃ i ∈ s, f i) < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set β
    f : β → Set α
    hs : s.Finite
    hfin : ∀ (i : β), Membership.mem s i → LT.lt (μ (f i)) Top.top
    ⊢ LT.lt (μ (Set.iUnion fun i => Set.iUnion fun h => f i)) Top.top
  -/
  convert (measure_biUnion_finset_le (μ := μ) hs.toFinset f).trans_lt _ using 3
    /-
      case h.e'_3.h.e'_6.h.e'_3
      α : Type u_1
      β : Type u_2
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set β
      f : β → Set α
      hs : s.Finite
      hfin : ∀ (i : β), Membership.mem s i → LT.lt (μ (f i)) Top.top
      ⊢ Eq (fun i => Set.iUnion fun h => f i) fun i => Set.iUnion fun h => f i
    -/
  · ext
    /-
      case h.e'_3.h.e'_6.h.e'_3.h.h
      α : Type u_1
      β : Type u_2
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set β
      f : β → Set α
      hs : s.Finite
      hfin : ∀ (i : β), Membership.mem s i → LT.lt (μ (f i)) Top.top
      x✝¹ : β
      x✝ : α
      ⊢ Iff (Membership.mem (Set.iUnion fun h => f x✝¹) x✝) (Membership.mem (Set.iUn …
    -/
    rw [Finite.mem_toFinset]
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      α : Type u_1
      β : Type u_2
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set β
      f : β → Set α
      hs : s.Finite
      hfin : ∀ (i : β), Membership.mem s i → LT.lt (μ (f i)) Top.top
      ⊢ LT.lt (hs.toFinset.sum fun i => μ (f i)) Top.top
    -/
  · simpa only [ENNReal.sum_lt_top, Finite.mem_toFinset]
    /-
      🎉 no goals
    -/


theorem measure_union_lt_top (hs : μ s < ∞) (ht : μ t < ∞) : μ (s ∪ t) < ∞ :=
  (measure_union_le s t).trans_lt (ENNReal.add_lt_top.mpr ⟨hs, ht⟩)


@[simp]
theorem measure_union_lt_top_iff : μ (s ∪ t) < ∞ ↔ μ s < ∞ ∧ μ t < ∞ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    ⊢ Iff (LT.lt (μ (Union.union s t)) Top.top) (And (LT.lt (μ s) Top.top) (LT.lt  …
  -/
  refine ⟨fun h => ⟨?_, ?_⟩, fun h => measure_union_lt_top h.1 h.2⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : Set α
      h : LT.lt (μ (Union.union s t)) Top.top
      ⊢ LT.lt (μ s) Top.top
    -/
  · exact (measure_mono Set.subset_union_left).trans_lt h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s t : Set α
      h : LT.lt (μ (Union.union s t)) Top.top
      ⊢ LT.lt (μ t) Top.top
    -/
  · exact (measure_mono Set.subset_union_right).trans_lt h
    /-
      🎉 no goals
    -/


theorem measure_union_ne_top (hs : μ s ≠ ∞) (ht : μ t ≠ ∞) : μ (s ∪ t) ≠ ∞ :=
  (measure_union_lt_top hs.lt_top ht.lt_top).ne


open scoped symmDiff in
theorem measure_symmDiff_ne_top (hs : μ s ≠ ∞) (ht : μ t ≠ ∞) : μ (s ∆ t) ≠ ∞ :=
  ne_top_of_le_ne_top (measure_union_ne_top hs ht) <| measure_mono symmDiff_subset_union


@[simp]
theorem measure_union_eq_top_iff : μ (s ∪ t) = ∞ ↔ μ s = ∞ ∨ μ t = ∞ :=
                      /-
                        α : Type u_1
                        inst✝ : MeasurableSpace α
                        μ : MeasureTheory.Measure α
                        s t : Set α
                        ⊢ Iff (Not (Eq (μ (Union.union s t)) Top.top)) (Not (Or (Eq (μ s) Top.top) (Eq …
                      -/
  not_iff_not.1 <| by simp only [← lt_top_iff_ne_top, ← Ne.eq_def, not_or, measure_union_lt_top_iff]
                      /-
                        🎉 no goals
                      -/


theorem exists_measure_pos_of_not_measure_iUnion_null [Countable ι] {s : ι → Set α}
    (hs : μ (⋃ n, s n) ≠ 0) : ∃ n, 0 < μ (s n) := by
  /-
    α : Type u_1
    ι : Sort u_5
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hs : Ne (μ (Set.iUnion fun n => s n)) 0
    ⊢ Exists fun n => LT.lt 0 (μ (s n))
  -/
  contrapose! hs
  /-
    α : Type u_1
    ι : Sort u_5
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable ι
    s : ι → Set α
    hs : ∀ (n : ι), LE.le (μ (s n)) 0
    ⊢ Eq (μ (Set.iUnion fun n => s n)) 0
  -/
  exact measure_iUnion_null fun n => nonpos_iff_eq_zero.1 (hs n)
  /-
    🎉 no goals
  -/


theorem measure_lt_top_of_subset (hst : t ⊆ s) (hs : μ s ≠ ∞) : μ t < ∞ :=
  lt_of_le_of_lt (μ.mono hst) hs.lt_top


theorem measure_inter_lt_top_of_left_ne_top (hs_finite : μ s ≠ ∞) : μ (s ∩ t) < ∞ :=
  measure_lt_top_of_subset inter_subset_left hs_finite


theorem measure_inter_lt_top_of_right_ne_top (ht_finite : μ t ≠ ∞) : μ (s ∩ t) < ∞ :=
  measure_lt_top_of_subset inter_subset_right ht_finite


theorem measure_inter_null_of_null_right (S : Set α) {T : Set α} (h : μ T = 0) : μ (S ∩ T) = 0 :=
  measure_mono_null inter_subset_right h


theorem measure_inter_null_of_null_left {S : Set α} (T : Set α) (h : μ S = 0) : μ (S ∩ T) = 0 :=
  measure_mono_null inter_subset_left h


/-- Given a predicate on `β` and `Set α` where both `α` and `β` are measurable spaces, if the
predicate holds for almost every `x : β` and
- `∅ : Set α`
- a family of sets generating the σ-algebra of `α`
Moreover, if for almost every `x : β`, the predicate is closed under complements and countable
disjoint unions, then the predicate holds for almost every `x : β` and all measurable sets of `α`.

This is an AE version of `MeasurableSpace.induction_on_inter` where the condition is dependent
on a measurable space `β`. -/
theorem _root_.MeasurableSpace.ae_induction_on_inter
    {α β : Type*} [MeasurableSpace β] {μ : Measure β}
    {C : β → Set α → Prop} {s : Set (Set α)} [m : MeasurableSpace α]
    (h_eq : m = MeasurableSpace.generateFrom s)
    (h_inter : IsPiSystem s) (h_empty : ∀ᵐ x ∂μ, C x ∅) (h_basic : ∀ᵐ x ∂μ, ∀ t ∈ s, C x t)
    (h_compl : ∀ᵐ x ∂μ, ∀ t, MeasurableSet t → C x t → C x tᶜ)
    (h_union : ∀ᵐ x ∂μ, ∀ f : ℕ → Set α,
        Pairwise (Disjoint on f) → (∀ i, MeasurableSet (f i)) → (∀ i, C x (f i)) → C x (⋃ i, f i)) :
    ∀ᵐ x ∂μ, ∀ ⦃t⦄, MeasurableSet t → C x t := by
  filter_upwards [h_empty, h_basic, h_compl, h_union] with x hx_empty hx_basic hx_compl hx_union
    using MeasurableSpace.induction_on_inter (C := fun t _ ↦ C x t)
      h_eq h_inter hx_empty hx_basic hx_compl hx_union


open Classical in
/-- A measurable set `t ⊇ s` such that `μ t = μ s`. It even satisfies `μ (t ∩ u) = μ (s ∩ u)` for
any measurable set `u` if `μ s ≠ ∞`, see `measure_toMeasurable_inter`.
(This property holds without the assumption `μ s ≠ ∞` when the space is s-finite -- for example
σ-finite), see `measure_toMeasurable_inter_of_sFinite`).
If `s` is a null measurable set, then
we also have `t =ᵐ[μ] s`, see `NullMeasurableSet.toMeasurable_ae_eq`.
This notion is sometimes called a "measurable hull" in the literature. -/
irreducible_def toMeasurable (μ : Measure α) (s : Set α) : Set α :=
  if h : ∃ t, t ⊇ s ∧ MeasurableSet t ∧ t =ᵐ[μ] s then h.choose else
    if h' : ∃ t, t ⊇ s ∧ MeasurableSet t ∧
      ∀ u, MeasurableSet u → μ (t ∩ u) = μ (s ∩ u) then h'.choose
    else (exists_measurable_superset μ s).choose


theorem subset_toMeasurable (μ : Measure α) (s : Set α) : s ⊆ toMeasurable μ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ HasSubset.Subset s (MeasureTheory.toMeasurable μ s)
  -/
  rw [toMeasurable_def]; split_ifs with hs h's
  /-
    case pos
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Exists fun t => And (Superset t s) (And (MeasurableSet t) ((MeasureTheory …
    ⊢ HasSubset.Subset s hs.choose
  -/
  exacts [hs.choose_spec.1, h's.choose_spec.1, (exists_measurable_superset μ s).choose_spec.1]
  /-
    🎉 no goals
  -/


theorem ae_le_toMeasurable : s ≤ᵐ[μ] toMeasurable μ s :=
  HasSubset.Subset.eventuallyLE (subset_toMeasurable _ _)


@[simp]
theorem measurableSet_toMeasurable (μ : Measure α) (s : Set α) :
    MeasurableSet (toMeasurable μ s) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ MeasurableSet (MeasureTheory.toMeasurable μ s)
  -/
  rw [toMeasurable_def]; split_ifs with hs h's
  exacts [hs.choose_spec.2.1, h's.choose_spec.2.1,
          (exists_measurable_superset μ s).choose_spec.2.1]


@[simp]
theorem measure_toMeasurable (s : Set α) : μ (toMeasurable μ s) = μ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Eq (μ (MeasureTheory.toMeasurable μ s)) (μ s)
  -/
  rw [toMeasurable_def]; split_ifs with hs h's
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : Exists fun t => And (Superset t s) (And (MeasurableSet t) ((MeasureTheory …
      ⊢ Eq (μ hs.choose) (μ s)
    -/
  · exact measure_congr hs.choose_spec.2.2
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : Not (Exists fun t => And (Superset t s) (And (MeasurableSet t) ((MeasureT …
      h's : Exists fun t => And (Superset t s) (And (MeasurableSet t) (∀ (u : Set α) …
      ⊢ Eq (μ h's.choose) (μ s)
    -/
  · simpa only [inter_univ] using h's.choose_spec.2.2 univ MeasurableSet.univ
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : Not (Exists fun t => And (Superset t s) (And (MeasurableSet t) ((MeasureT …
      h's : Not (Exists fun t => And (Superset t s) (And (MeasurableSet t) (∀ (u : S …
      ⊢ Eq (μ ⋯.choose) (μ s)
    -/
  · exact (exists_measurable_superset μ s).choose_spec.2.2
    /-
      🎉 no goals
    -/


/-- A measure space is a measurable space equipped with a
  measure, referred to as `volume`. -/
class MeasureSpace (α : Type*) extends MeasurableSpace α where
  volume : Measure α


/-- `∀ᵐ a, p a` means that `p a` for a.e. `a`, i.e. `p` holds true away from a null set.

This is notation for `Filter.Eventually P (MeasureTheory.ae MeasureSpace.volume)`. -/
notation3 "∀ᵐ "(...)", "r:(scoped P =>
  Filter.Eventually P <| MeasureTheory.ae MeasureTheory.MeasureSpace.volume) => r


/-- `∃ᵐ a, p a` means that `p` holds frequently, i.e. on a set of positive measure,
w.r.t. the volume measure.

This is notation for `Filter.Frequently P (MeasureTheory.ae MeasureSpace.volume)`. -/
notation3 "∃ᵐ "(...)", "r:(scoped P =>
  Filter.Frequently P <| MeasureTheory.ae MeasureTheory.MeasureSpace.volume) => r


/-- The tactic `exact volume`, to be used in optional (`autoParam`) arguments. -/
macro "volume_tac" : tactic =>
  `(tactic| (first | exact MeasureTheory.MeasureSpace.volume))


/-- A function is almost everywhere measurable if it coincides almost everywhere with a measurable
function. -/
@[fun_prop]
def AEMeasurable {_m : MeasurableSpace α} (f : α → β) (μ : Measure α := by volume_tac) : Prop :=
  ∃ g : α → β, Measurable g ∧ f =ᵐ[μ] g


@[fun_prop, aesop unsafe 30% apply (rule_sets := [Measurable])]
theorem Measurable.aemeasurable (h : Measurable f) : AEMeasurable f μ :=
  ⟨f, h, ae_eq_refl f⟩


lemma of_discrete [DiscreteMeasurableSpace α] : AEMeasurable f μ :=
  Measurable.of_discrete.aemeasurable


/-- Given an almost everywhere measurable function `f`, associate to it a measurable function
that coincides with it almost everywhere. `f` is explicit in the definition to make sure that
it shows in pretty-printing. -/
def mk (f : α → β) (h : AEMeasurable f μ) : α → β :=
  Classical.choose h


@[measurability]
theorem measurable_mk (h : AEMeasurable f μ) : Measurable (h.mk f) :=
  (Classical.choose_spec h).1


theorem ae_eq_mk (h : AEMeasurable f μ) : f =ᵐ[μ] h.mk f :=
  (Classical.choose_spec h).2


theorem congr (hf : AEMeasurable f μ) (h : f =ᵐ[μ] g) : AEMeasurable g μ :=
  ⟨hf.mk f, hf.measurable_mk, h.symm.trans hf.ae_eq_mk⟩


theorem aemeasurable_congr (h : f =ᵐ[μ] g) : AEMeasurable f μ ↔ AEMeasurable g μ :=
  ⟨fun hf => AEMeasurable.congr hf h, fun hg => AEMeasurable.congr hg h.symm⟩


@[simp, fun_prop, measurability]
theorem aemeasurable_const {b : β} : AEMeasurable (fun _a : α => b) μ :=
  measurable_const.aemeasurable


@[measurability]
theorem aemeasurable_id : AEMeasurable id μ :=
  measurable_id.aemeasurable


@[measurability]
theorem aemeasurable_id' : AEMeasurable (fun x => x) μ :=
  measurable_id.aemeasurable


theorem Measurable.comp_aemeasurable [MeasurableSpace δ] {f : α → δ} {g : δ → β} (hg : Measurable g)
    (hf : AEMeasurable f μ) : AEMeasurable (g ∘ f) μ :=
  ⟨g ∘ hf.mk f, hg.comp hf.measurable_mk, EventuallyEq.fun_comp hf.ae_eq_mk _⟩


@[fun_prop, measurability]
theorem Measurable.comp_aemeasurable' [MeasurableSpace δ] {f : α → δ} {g : δ → β}
    (hg : Measurable g) (hf : AEMeasurable f μ) : AEMeasurable (fun x ↦ g (f x)) μ :=
  Measurable.comp_aemeasurable hg hf


