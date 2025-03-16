open Classical in
/-- A measurable set such that `μ.restrict (μ.sigmaFiniteSetWRT ν)` is sigma-finite and for all
measurable sets `t ⊆ sᶜ`, either `ν t = 0` or `μ t = ∞`. -/
def Measure.sigmaFiniteSetWRT (μ ν : Measure α) : Set α :=
  if h : ∃ s : Set α, MeasurableSet s ∧ SigmaFinite (μ.restrict s)
    ∧ (∀ t, t ⊆ sᶜ → ν t ≠ 0 → μ t = ∞)
  then h.choose
  else ∅


@[measurability]
lemma measurableSet_sigmaFiniteSetWRT :
    MeasurableSet (μ.sigmaFiniteSetWRT ν) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ MeasurableSet (μ.sigmaFiniteSetWRT ν)
  -/
  rw [Measure.sigmaFiniteSetWRT]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    ⊢ MeasurableSet (dite (Exists fun s => And (MeasurableSet s) (And (MeasureTheo …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Exists fun s => And (MeasurableSet s) (And (MeasureTheory.SigmaFinite (μ.r …
      ⊢ MeasurableSet h.choose
    -/
  · exact h.choose_spec.1
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      h : Not (Exists fun s => And (MeasurableSet s) (And (MeasureTheory.SigmaFinite …
      ⊢ MeasurableSet EmptyCollection.emptyCollection
    -/
  · exact MeasurableSet.empty
    /-
      🎉 no goals
    -/


instance : SigmaFinite (μ.restrict (μ.sigmaFiniteSetWRT ν)) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    ⊢ MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetWRT ν))
  -/
  rw [Measure.sigmaFiniteSetWRT]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    ⊢ MeasureTheory.SigmaFinite (μ.restrict (dite (Exists fun s => And (Measurable …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s t : Set α
      h : Exists fun s => And (MeasurableSet s) (And (MeasureTheory.SigmaFinite (μ.r …
      ⊢ MeasureTheory.SigmaFinite (μ.restrict h.choose)
    -/
  · exact h.choose_spec.2.1
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s t : Set α
      h : Not (Exists fun s => And (MeasurableSet s) (And (MeasureTheory.SigmaFinite …
      ⊢ MeasureTheory.SigmaFinite (μ.restrict EmptyCollection.emptyCollection)
    -/
  · rw [Measure.restrict_empty]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s t : Set α
      h : Not (Exists fun s => And (MeasurableSet s) (And (MeasureTheory.SigmaFinite …
      ⊢ MeasureTheory.SigmaFinite 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- Let `C` be the supremum of `ν s` over all measurable sets `s` such that `μ.restrict s` is
sigma-finite. `C` is finite since `ν` is a finite measure. Then there exists a measurable set `t`
with `μ.restrict t` sigma-finite such that `ν t ≥ C - 1/n`. -/
lemma exists_isSigmaFiniteSet_measure_ge (μ ν : Measure α) [IsFiniteMeasure ν] (n : ℕ) :
    ∃ t, MeasurableSet t ∧ SigmaFinite (μ.restrict t)
      ∧ (⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s) - 1/n ≤ ν t := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    n : Nat
    ⊢ Exists fun t => And (MeasurableSet t) (And (MeasureTheory.SigmaFinite (μ.res …
  -/
  by_cases hC_lt : 1/n < ⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s
  · have h_lt_top : ⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s < ∞ := by
      refine (?_ : ⨆ (s) (_ : MeasurableSet s)
        (_ : SigmaFinite (μ.restrict s)), ν s ≤ ν Set.univ).trans_lt (measure_lt_top _ _)
      refine iSup_le (fun s ↦ ?_)
      exact iSup_le (fun _ ↦ iSup_le (fun _ ↦ measure_mono (Set.subset_univ s)))
    obtain ⟨t, ht⟩ := exists_lt_of_lt_ciSup
      (ENNReal.sub_lt_self h_lt_top.ne (ne_zero_of_lt hC_lt) (by simp) :
          (⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s) - 1/n
        < ⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s)
    have ht_meas : MeasurableSet t := by
      by_contra h_not_mem
      simp only [h_not_mem] at ht
      simp at ht
    have ht_mem : SigmaFinite (μ.restrict t) := by
      by_contra h_not_mem
      simp only [h_not_mem] at ht
      simp at ht
    /-
      case pos.intro
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      n : Nat
      hC_lt : LT.lt (HDiv.hDiv 1 ↑n) (iSup fun s => iSup fun x => iSup fun x => ν s)
      h_lt_top : LT.lt (iSup fun s => iSup fun x => iSup fun x => ν s) Top.top
      t : Set α
      ht : LT.lt (HSub.hSub (iSup fun s => iSup fun x => iSup fun x => ν s) (HDiv.hD …
      ht_meas : MeasurableSet t
      ht_mem : MeasureTheory.SigmaFinite (μ.restrict t)
      ⊢ Exists fun t => And (MeasurableSet t) (And (MeasureTheory.SigmaFinite (μ.res …
    -/
    refine ⟨t, ht_meas, ht_mem, ?_⟩
    /-
      case pos.intro
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      n : Nat
      hC_lt : LT.lt (HDiv.hDiv 1 ↑n) (iSup fun s => iSup fun x => iSup fun x => ν s)
      h_lt_top : LT.lt (iSup fun s => iSup fun x => iSup fun x => ν s) Top.top
      t : Set α
      ht : LT.lt (HSub.hSub (iSup fun s => iSup fun x => iSup fun x => ν s) (HDiv.hD …
      ht_meas : MeasurableSet t
      ht_mem : MeasureTheory.SigmaFinite (μ.restrict t)
      ⊢ LE.le (HSub.hSub (iSup fun s => iSup fun x => iSup fun x => ν s) (HDiv.hDiv  …
    -/
    simp only [ht_meas, ht_mem, iSup_true] at ht
    /-
      case pos.intro
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      n : Nat
      hC_lt : LT.lt (HDiv.hDiv 1 ↑n) (iSup fun s => iSup fun x => iSup fun x => ν s)
      h_lt_top : LT.lt (iSup fun s => iSup fun x => iSup fun x => ν s) Top.top
      t : Set α
      ht_meas : MeasurableSet t
      ht_mem : MeasureTheory.SigmaFinite (μ.restrict t)
      ht : LT.lt (HSub.hSub (iSup fun s => iSup fun x => iSup fun x => ν s) (HDiv.hD …
      ⊢ LE.le (HSub.hSub (iSup fun s => iSup fun x => iSup fun x => ν s) (HDiv.hDiv  …
    -/
    exact ht.le
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      n : Nat
      hC_lt : Not (LT.lt (HDiv.hDiv 1 ↑n) (iSup fun s => iSup fun x => iSup fun x => …
      ⊢ Exists fun t => And (MeasurableSet t) (And (MeasureTheory.SigmaFinite (μ.res …
    -/
  · refine ⟨∅, MeasurableSet.empty, by rw [Measure.restrict_empty]; infer_instance, ?_⟩
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      n : Nat
      hC_lt : Not (LT.lt (HDiv.hDiv 1 ↑n) (iSup fun s => iSup fun x => iSup fun x => …
      ⊢ LE.le (HSub.hSub (iSup fun s => iSup fun x => iSup fun x => ν s) (HDiv.hDiv  …
    -/
    rw [tsub_eq_zero_of_le (not_lt.mp hC_lt)]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      n : Nat
      hC_lt : Not (LT.lt (HDiv.hDiv 1 ↑n) (iSup fun s => iSup fun x => iSup fun x => …
      ⊢ LE.le 0 (ν EmptyCollection.emptyCollection)
    -/
    exact zero_le'
    /-
      🎉 no goals
    -/


/-- A measurable set such that `μ.restrict (μ.sigmaFiniteSetGE ν n)` is sigma-finite and
for `C` the supremum of `ν s` over all measurable sets `s` with `μ.restrict s` sigma-finite,
`ν (μ.sigmaFiniteSetGE ν n) ≥ C - 1/n`. -/
def Measure.sigmaFiniteSetGE (μ ν : Measure α) [IsFiniteMeasure ν] (n : ℕ) : Set α :=
  (exists_isSigmaFiniteSet_measure_ge μ ν n).choose


lemma measurableSet_sigmaFiniteSetGE [IsFiniteMeasure ν] (n : ℕ) :
    MeasurableSet (μ.sigmaFiniteSetGE ν n) :=
  (exists_isSigmaFiniteSet_measure_ge μ ν n).choose_spec.1


lemma sigmaFinite_restrict_sigmaFiniteSetGE (μ ν : Measure α) [IsFiniteMeasure ν] (n : ℕ) :
    SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE ν n)) :=
  (exists_isSigmaFiniteSet_measure_ge μ ν n).choose_spec.2.1


lemma measure_sigmaFiniteSetGE_le (μ ν : Measure α) [IsFiniteMeasure ν] (n : ℕ) :
    ν (μ.sigmaFiniteSetGE ν n)
      ≤ ⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s := by
  refine (le_iSup (f := fun s ↦ _)
    (sigmaFinite_restrict_sigmaFiniteSetGE μ ν n)).trans ?_
  exact le_iSup₂ (f := fun s _ ↦ ⨆ (_ : SigmaFinite (μ.restrict s)), ν s) (μ.sigmaFiniteSetGE ν n)
    (measurableSet_sigmaFiniteSetGE n)


lemma measure_sigmaFiniteSetGE_ge (μ ν : Measure α) [IsFiniteMeasure ν] (n : ℕ) :
    (⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s) - 1/n
      ≤ ν (μ.sigmaFiniteSetGE ν n) :=
  (exists_isSigmaFiniteSet_measure_ge μ ν n).choose_spec.2.2


lemma tendsto_measure_sigmaFiniteSetGE (μ ν : Measure α) [IsFiniteMeasure ν] :
    Tendsto (fun n ↦ ν (μ.sigmaFiniteSetGE ν n)) atTop
      (𝓝 (⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s)) := by
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le ?_
    tendsto_const_nhds (measure_sigmaFiniteSetGE_ge μ ν) (measure_sigmaFiniteSetGE_le μ ν)
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Filter.Tendsto (fun i => HSub.hSub (iSup fun s => iSup fun x => iSup fun x = …
  -/
  nth_rewrite 2 [← tsub_zero (⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s)]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Filter.Tendsto (fun i => HSub.hSub (iSup fun s => iSup fun x => iSup fun x = …
  -/
  refine ENNReal.Tendsto.sub tendsto_const_nhds ?_ (Or.inr ENNReal.zero_ne_top)
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Filter.Tendsto (fun i => HDiv.hDiv 1 ↑i) Filter.atTop (nhds 0)
  -/
  simp only [one_div]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Filter.Tendsto (fun i => Inv.inv ↑i) Filter.atTop (nhds 0)
  -/
  exact ENNReal.tendsto_inv_nat_nhds_zero
  /-
    🎉 no goals
  -/


/-- A measurable set such that `μ.restrict (μ.sigmaFiniteSetWRT' ν)` is sigma-finite and
`ν (μ.sigmaFiniteSetWRT' ν)` has maximal measure among such sets. -/
def Measure.sigmaFiniteSetWRT' (μ ν : Measure α) [IsFiniteMeasure ν] : Set α :=
  ⋃ n, μ.sigmaFiniteSetGE ν n


lemma measurableSet_sigmaFiniteSetWRT' [IsFiniteMeasure ν] :
    MeasurableSet (μ.sigmaFiniteSetWRT' ν) :=
  MeasurableSet.iUnion measurableSet_sigmaFiniteSetGE


lemma sigmaFinite_restrict_sigmaFiniteSetWRT' (μ ν : Measure α) [IsFiniteMeasure ν] :
    SigmaFinite (μ.restrict (μ.sigmaFiniteSetWRT' ν)) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetWRT' ν))
  -/
  have := sigmaFinite_restrict_sigmaFiniteSetGE μ ν
  let f : ℕ × ℕ → Set α := fun p : ℕ × ℕ ↦ (μ.sigmaFiniteSetWRT' ν)ᶜ
    ∪ (spanningSets (μ.restrict (μ.sigmaFiniteSetGE ν p.1)) p.2 ∩ (μ.sigmaFiniteSetGE ν p.1))
  suffices (μ.restrict (μ.sigmaFiniteSetWRT' ν)).FiniteSpanningSetsIn (Set.range f) from
    this.sigmaFinite
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
    f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
    ⊢ (μ.restrict (μ.sigmaFiniteSetWRT' ν)).FiniteSpanningSetsIn (Set.range f)
  -/
  let e : ℕ ≃ ℕ × ℕ := Nat.pairEquiv.symm
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
    f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
    e : Equiv Nat (Prod Nat Nat) := Nat.pairEquiv.symm
    ⊢ (μ.restrict (μ.sigmaFiniteSetWRT' ν)).FiniteSpanningSetsIn (Set.range f)
  -/
  refine ⟨fun n ↦ f (e n), fun _ ↦ by simp, fun n ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
      f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
      e : Equiv Nat (Prod Nat Nat) := Nat.pairEquiv.symm
      n : Nat
      ⊢ LT.lt ((μ.restrict (μ.sigmaFiniteSetWRT' ν)) ((fun n => f (e n)) n)) Top.top
    -/
  · simp only [Nat.pairEquiv_symm_apply, gt_iff_lt, measure_union_lt_top_iff, f, e]
    rw [Measure.restrict_apply' measurableSet_sigmaFiniteSetWRT', Set.compl_inter_self,
      Measure.restrict_apply' measurableSet_sigmaFiniteSetWRT']
    /-
      case refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
      f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
      e : Equiv Nat (Prod Nat Nat) := Nat.pairEquiv.symm
      n : Nat
      ⊢ And (LT.lt (μ EmptyCollection.emptyCollection) Top.top) (LT.lt (μ (Inter.int …
    -/
    simp only [measure_empty, ENNReal.zero_lt_top, true_and]
    /-
      case refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
      f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
      e : Equiv Nat (Prod Nat Nat) := Nat.pairEquiv.symm
      n : Nat
      ⊢ LT.lt (μ (Inter.inter (Inter.inter (MeasureTheory.spanningSets (μ.restrict ( …
    -/
    refine (measure_mono Set.inter_subset_left).trans_lt ?_
    /-
      case refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
      f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
      e : Equiv Nat (Prod Nat Nat) := Nat.pairEquiv.symm
      n : Nat
      ⊢ LT.lt (μ (Inter.inter (MeasureTheory.spanningSets (μ.restrict (μ.sigmaFinite …
    -/
    rw [← Measure.restrict_apply' (measurableSet_sigmaFiniteSetGE _)]
    /-
      case refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
      f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
      e : Equiv Nat (Prod Nat Nat) := Nat.pairEquiv.symm
      n : Nat
      ⊢ LT.lt ((μ.restrict (μ.sigmaFiniteSetGE ν (Nat.unpair n).1)) (MeasureTheory.s …
    -/
    exact measure_spanningSets_lt_top _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
      f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
      e : Equiv Nat (Prod Nat Nat) := Nat.pairEquiv.symm
      ⊢ Eq (Set.iUnion fun i => (fun n => f (e n)) i) Set.univ
    -/
  · simp only [Nat.pairEquiv_symm_apply, f, e]
    /-
      case refine_2
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure ν
      this : ∀ (n : Nat), MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetGE  …
      f : Prod Nat Nat → Set α := fun p => Union.union (HasCompl.compl (μ.sigmaFinit …
      e : Equiv Nat (Prod Nat Nat) := Nat.pairEquiv.symm
      ⊢ Eq (Set.iUnion fun i => Union.union (HasCompl.compl (μ.sigmaFiniteSetWRT' ν) …
    -/
    rw [← Set.union_iUnion]
    suffices ⋃ n, (spanningSets (μ.restrict (μ.sigmaFiniteSetGE ν (Nat.unpair n).1)) n.unpair.2
        ∩ μ.sigmaFiniteSetGE ν n.unpair.1) = μ.sigmaFiniteSetWRT' ν by
      rw [this, Set.compl_union_self]
    calc ⋃ n, (spanningSets (μ.restrict (μ.sigmaFiniteSetGE ν (Nat.unpair n).1)) n.unpair.2
        ∩ μ.sigmaFiniteSetGE ν n.unpair.1)
      = ⋃ n, ⋃ m, (spanningSets (μ.restrict (μ.sigmaFiniteSetGE ν n)) m
            ∩ μ.sigmaFiniteSetGE ν n) :=
          Set.iUnion_unpair (fun n m ↦ spanningSets (μ.restrict (μ.sigmaFiniteSetGE ν n)) m
            ∩ μ.sigmaFiniteSetGE ν n)
    _ = ⋃ n, μ.sigmaFiniteSetGE ν n := by
        refine Set.iUnion_congr (fun n ↦ ?_)
        rw [← Set.iUnion_inter, iUnion_spanningSets, Set.univ_inter]
    _ = μ.sigmaFiniteSetWRT' ν := rfl


/-- `μ.sigmaFiniteSetWRT' ν` has maximal `ν`-measure among all measurable sets `s` with sigma-finite
`μ.restrict s`. -/
lemma measure_sigmaFiniteSetWRT' (μ ν : Measure α) [IsFiniteMeasure ν] :
    ν (μ.sigmaFiniteSetWRT' ν)
      = ⨆ (s) (_ : MeasurableSet s) (_ : SigmaFinite (μ.restrict s)), ν s := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Eq (ν (μ.sigmaFiniteSetWRT' ν)) (iSup fun s => iSup fun x => iSup fun x => ν …
  -/
  apply le_antisymm
  · refine (le_iSup (f := fun _ ↦ _)
      (sigmaFinite_restrict_sigmaFiniteSetWRT' μ ν)).trans ?_
    exact le_iSup₂ (f := fun s _ ↦ ⨆ (_ : SigmaFinite (μ.restrict s)), ν s) (μ.sigmaFiniteSetWRT' ν)
      measurableSet_sigmaFiniteSetWRT'
  · exact le_of_tendsto' (tendsto_measure_sigmaFiniteSetGE μ ν)
      (fun _ ↦ measure_mono (Set.subset_iUnion _ _))


/-- Auxiliary lemma for `measure_eq_top_of_subset_compl_sigmaFiniteSetWRT'`. -/
lemma measure_eq_top_of_subset_compl_sigmaFiniteSetWRT'_of_measurableSet [IsFiniteMeasure ν]
    (hs : MeasurableSet s) (hs_subset : s ⊆ (μ.sigmaFiniteSetWRT' ν)ᶜ) (hνs : ν s ≠ 0) :
    μ s = ∞ := by
  suffices ¬ SigmaFinite (μ.restrict s) by
    by_contra h
    have h_lt_top : Fact (μ s < ∞) := ⟨Ne.lt_top h⟩
    exact this inferInstance
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    hs : MeasurableSet s
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hνs : Ne (ν s) 0
    ⊢ Not (MeasureTheory.SigmaFinite (μ.restrict s))
  -/
  intro hsσ
  have h_lt : ν (μ.sigmaFiniteSetWRT' ν) < ν (μ.sigmaFiniteSetWRT' ν ∪ s) := by
    rw [measure_union _ hs]
    · exact ENNReal.lt_add_right (measure_ne_top _ _) hνs
    · exact disjoint_compl_right.mono_right hs_subset
  have h_le : ν (μ.sigmaFiniteSetWRT' ν ∪ s) ≤ ν (μ.sigmaFiniteSetWRT' ν) := by
    conv_rhs => rw [measure_sigmaFiniteSetWRT']
    refine (le_iSup
      (f := fun (_ : SigmaFinite (μ.restrict (μ.sigmaFiniteSetWRT' ν ∪ s))) ↦ _) ?_).trans ?_
    · have := sigmaFinite_restrict_sigmaFiniteSetWRT' μ ν
      infer_instance
    · exact le_iSup₂ (f := fun s _ ↦ ⨆ (_ : SigmaFinite (μ.restrict _)), ν s)
        (μ.sigmaFiniteSetWRT' ν ∪ s) (measurableSet_sigmaFiniteSetWRT'.union hs)
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    hs : MeasurableSet s
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hνs : Ne (ν s) 0
    hsσ : MeasureTheory.SigmaFinite (μ.restrict s)
    h_lt : LT.lt (ν (μ.sigmaFiniteSetWRT' ν)) (ν (Union.union (μ.sigmaFiniteSetWRT …
    h_le : LE.le (ν (Union.union (μ.sigmaFiniteSetWRT' ν) s)) (ν (μ.sigmaFiniteSet …
    ⊢ False
  -/
  exact h_lt.not_le h_le
  /-
    🎉 no goals
  -/


/-- For all sets `s` in `(μ.sigmaFiniteSetWRT ν)ᶜ`, if `ν s ≠ 0` then `μ s = ∞`. -/
lemma measure_eq_top_of_subset_compl_sigmaFiniteSetWRT' [IsFiniteMeasure ν]
    (hs_subset : s ⊆ (μ.sigmaFiniteSetWRT' ν)ᶜ) (hνs : ν s ≠ 0) :
    μ s = ∞ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hνs : Ne (ν s) 0
    ⊢ Eq (μ s) Top.top
  -/
  rw [measure_eq_iInf]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hνs : Ne (ν s) 0
    ⊢ Eq (iInf fun t => iInf fun x => iInf fun x => μ t) Top.top
  -/
  simp_rw [iInf_eq_top]
  suffices ∀ t, t ⊆ (μ.sigmaFiniteSetWRT' ν)ᶜ → s ⊆ t → MeasurableSet t → μ t = ∞ by
    intro t hts ht
    suffices μ (t ∩ (μ.sigmaFiniteSetWRT' ν)ᶜ) = ∞ from
      measure_mono_top Set.inter_subset_left this
    have hs_subset_t : s ⊆ t ∩ (μ.sigmaFiniteSetWRT' ν)ᶜ := Set.subset_inter hts hs_subset
    exact this (t ∩ (μ.sigmaFiniteSetWRT' ν)ᶜ) Set.inter_subset_right hs_subset_t
      (ht.inter measurableSet_sigmaFiniteSetWRT'.compl)
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hνs : Ne (ν s) 0
    ⊢ ∀ (t : Set α), HasSubset.Subset t (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))  …
  -/
  intro t ht_subset hst ht
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hνs : Ne (ν s) 0
    t : Set α
    ht_subset : HasSubset.Subset t (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hst : HasSubset.Subset s t
    ht : MeasurableSet t
    ⊢ Eq (μ t) Top.top
  -/
  refine measure_eq_top_of_subset_compl_sigmaFiniteSetWRT'_of_measurableSet ht ht_subset ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hνs : Ne (ν s) 0
    t : Set α
    ht_subset : HasSubset.Subset t (HasCompl.compl (μ.sigmaFiniteSetWRT' ν))
    hst : HasSubset.Subset s t
    ht : MeasurableSet t
    ⊢ Ne (ν t) 0
  -/
  exact fun hνt ↦ hνs (measure_mono_null hst hνt)
  /-
    🎉 no goals
  -/


/-- For all sets `s` in `(μ.sigmaFiniteSetWRT ν)ᶜ`, if `ν s ≠ 0` then `μ s = ∞`. -/
lemma measure_eq_top_of_subset_compl_sigmaFiniteSetWRT [SFinite ν]
    (hs_subset : s ⊆ (μ.sigmaFiniteSetWRT ν)ᶜ) (hνs : ν s ≠ 0) :
    μ s = ∞ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.SFinite ν
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT ν))
    hνs : Ne (ν s) 0
    ⊢ Eq (μ s) Top.top
  -/
  have ⟨ν', hν', hνν', _⟩ := exists_isFiniteMeasure_absolutelyContinuous ν
  have h : ∃ s : Set α, MeasurableSet s ∧ SigmaFinite (μ.restrict s)
      ∧ (∀ t ⊆ sᶜ, ν t ≠ 0 → μ t = ∞) := by
    refine ⟨μ.sigmaFiniteSetWRT' ν', measurableSet_sigmaFiniteSetWRT',
      sigmaFinite_restrict_sigmaFiniteSetWRT' _ _,
      fun t ht_subset hνt ↦ measure_eq_top_of_subset_compl_sigmaFiniteSetWRT' ht_subset ?_⟩
    exact fun hν't ↦ hνt (hνν' hν't)
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.SFinite ν
    hs_subset : HasSubset.Subset s (HasCompl.compl (μ.sigmaFiniteSetWRT ν))
    hνs : Ne (ν s) 0
    ν' : MeasureTheory.Measure α
    hν' : MeasureTheory.IsFiniteMeasure ν'
    hνν' : ν.AbsolutelyContinuous ν'
    right✝ : ν'.AbsolutelyContinuous ν
    h : Exists fun s => And (MeasurableSet s) (And (MeasureTheory.SigmaFinite (μ.r …
    ⊢ Eq (μ s) Top.top
  -/
  rw [Measure.sigmaFiniteSetWRT, dif_pos h] at hs_subset
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.SFinite ν
    hνs : Ne (ν s) 0
    ν' : MeasureTheory.Measure α
    hν' : MeasureTheory.IsFiniteMeasure ν'
    hνν' : ν.AbsolutelyContinuous ν'
    right✝ : ν'.AbsolutelyContinuous ν
    h : Exists fun s => And (MeasurableSet s) (And (MeasureTheory.SigmaFinite (μ.r …
    hs_subset : HasSubset.Subset s (HasCompl.compl h.choose)
    ⊢ Eq (μ s) Top.top
  -/
  exact h.choose_spec.2.2 s hs_subset hνs
  /-
    🎉 no goals
  -/


lemma restrict_compl_sigmaFiniteSetWRT [SFinite ν] (hμν : μ ≪ ν) :
    μ.restrict (μ.sigmaFiniteSetWRT ν)ᶜ = ∞ • ν.restrict (μ.sigmaFiniteSetWRT ν)ᶜ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite ν
    hμν : μ.AbsolutelyContinuous ν
    ⊢ Eq (μ.restrict (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) (HSMul.hSMul Top.to …
  -/
  ext s
  rw [Measure.restrict_apply' measurableSet_sigmaFiniteSetWRT.compl,
    Measure.smul_apply, smul_eq_mul,
    Measure.restrict_apply' measurableSet_sigmaFiniteSetWRT.compl]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite ν
    hμν : μ.AbsolutelyContinuous ν
    s : Set α
    a✝ : MeasurableSet s
    ⊢ Eq (μ (Inter.inter s (HasCompl.compl (μ.sigmaFiniteSetWRT ν)))) (HMul.hMul T …
  -/
  by_cases hνs : ν (s ∩ (μ.sigmaFiniteSetWRT ν)ᶜ) = 0
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite ν
      hμν : μ.AbsolutelyContinuous ν
      s : Set α
      a✝ : MeasurableSet s
      hνs : Eq (ν (Inter.inter s (HasCompl.compl (μ.sigmaFiniteSetWRT ν)))) 0
      ⊢ Eq (μ (Inter.inter s (HasCompl.compl (μ.sigmaFiniteSetWRT ν)))) (HMul.hMul T …
    -/
  · rw [hνs, mul_zero]
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite ν
      hμν : μ.AbsolutelyContinuous ν
      s : Set α
      a✝ : MeasurableSet s
      hνs : Eq (ν (Inter.inter s (HasCompl.compl (μ.sigmaFiniteSetWRT ν)))) 0
      ⊢ Eq (μ (Inter.inter s (HasCompl.compl (μ.sigmaFiniteSetWRT ν)))) 0
    -/
    exact hμν hνs
    /-
      🎉 no goals
    -/
  · rw [ENNReal.top_mul hνs, measure_eq_top_of_subset_compl_sigmaFiniteSetWRT
      Set.inter_subset_right hνs]


@[simp]
lemma measure_compl_sigmaFiniteSetWRT (hμν : μ ≪ ν) [SigmaFinite μ] [SFinite ν] :
    ν (μ.sigmaFiniteSetWRT ν)ᶜ = 0 := by
  have h : ν (μ.sigmaFiniteSetWRT ν)ᶜ ≠ 0 → μ (μ.sigmaFiniteSetWRT ν)ᶜ = ∞ :=
    measure_eq_top_of_subset_compl_sigmaFiniteSetWRT subset_rfl
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    ⊢ Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0
  -/
  by_contra h0
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    h0 : Not (Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0)
    ⊢ False
  -/
  refine ENNReal.top_ne_zero ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    h0 : Not (Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0)
    ⊢ Eq Top.top 0
  -/
  rw [← h h0, ← Measure.iSup_restrict_spanningSets]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    h0 : Not (Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0)
    ⊢ Eq (iSup fun i => (μ.restrict (MeasureTheory.spanningSets μ i)) (HasCompl.co …
  -/
  simp_rw [Measure.restrict_apply' (measurableSet_spanningSets μ _), ENNReal.iSup_eq_zero]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    h0 : Not (Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0)
    ⊢ ∀ (i : Nat), Eq (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) (Me …
  -/
  intro i
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    h0 : Not (Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0)
    i : Nat
    ⊢ Eq (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) (MeasureTheory.s …
  -/
  by_contra h_ne_zero
  have h_zero_top := measure_eq_top_of_subset_compl_sigmaFiniteSetWRT
    (Set.inter_subset_left : (μ.sigmaFiniteSetWRT ν)ᶜ ∩ spanningSets μ i ⊆ _) ?_
  /-
    case refine_2
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    h0 : Not (Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0)
    i : Nat
    h_ne_zero : Not (Eq (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) ( …
    h_zero_top : Eq (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) (Meas …
    ⊢ False
  -/
  swap; · exact fun h ↦ h_ne_zero (hμν h)
          /-
            🎉 no goals
          -/
  /-
    case refine_2
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    h0 : Not (Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0)
    i : Nat
    h_ne_zero : Not (Eq (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) ( …
    h_zero_top : Eq (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) (Meas …
    ⊢ False
  -/
  refine absurd h_zero_top (ne_of_lt ?_)
  /-
    case refine_2
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : μ.AbsolutelyContinuous ν
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SFinite ν
    h : Ne (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0 → Eq (μ (HasCompl.compl  …
    h0 : Not (Eq (ν (HasCompl.compl (μ.sigmaFiniteSetWRT ν))) 0)
    i : Nat
    h_ne_zero : Not (Eq (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) ( …
    h_zero_top : Eq (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) (Meas …
    ⊢ LT.lt (μ (Inter.inter (HasCompl.compl (μ.sigmaFiniteSetWRT ν)) (MeasureTheor …
  -/
  exact (measure_mono Set.inter_subset_right).trans_lt (measure_spanningSets_lt_top μ i)
  /-
    🎉 no goals
  -/


/-- A measurable set such that `μ.restrict μ.sigmaFiniteSet` is sigma-finite,
  and for all measurable sets `s ⊆ μ.sigmaFiniteSetᶜ`, either `μ s = 0` or `μ s = ∞`. -/
def Measure.sigmaFiniteSet (μ : Measure α) : Set α := μ.sigmaFiniteSetWRT μ


@[measurability]
lemma measurableSet_sigmaFiniteSet : MeasurableSet μ.sigmaFiniteSet :=
  measurableSet_sigmaFiniteSetWRT


lemma measure_eq_zero_or_top_of_subset_compl_sigmaFiniteSet [SFinite μ]
    (ht_subset : t ⊆ μ.sigmaFiniteSetᶜ) :
    μ t = 0 ∨ μ t = ∞ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Set α
    inst✝ : MeasureTheory.SFinite μ
    ht_subset : HasSubset.Subset t (HasCompl.compl μ.sigmaFiniteSet)
    ⊢ Or (Eq (μ t) 0) (Eq (μ t) Top.top)
  -/
  rw [or_iff_not_imp_left]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Set α
    inst✝ : MeasureTheory.SFinite μ
    ht_subset : HasSubset.Subset t (HasCompl.compl μ.sigmaFiniteSet)
    ⊢ Not (Eq (μ t) 0) → Eq (μ t) Top.top
  -/
  exact measure_eq_top_of_subset_compl_sigmaFiniteSetWRT ht_subset
  /-
    🎉 no goals
  -/


/-- The measure `μ.restrict μ.sigmaFiniteSetᶜ` takes only two values: 0 and ∞ . -/
lemma restrict_compl_sigmaFiniteSet_eq_zero_or_top (μ : Measure α) [SFinite μ] (s : Set α) :
    μ.restrict μ.sigmaFiniteSetᶜ s = 0 ∨ μ.restrict μ.sigmaFiniteSetᶜ s = ∞ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    ⊢ Or (Eq ((μ.restrict (HasCompl.compl μ.sigmaFiniteSet)) s) 0) (Eq ((μ.restric …
  -/
  rw [Measure.restrict_apply' measurableSet_sigmaFiniteSet.compl]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    s : Set α
    ⊢ Or (Eq (μ (Inter.inter s (HasCompl.compl μ.sigmaFiniteSet))) 0) (Eq (μ (Inte …
  -/
  exact measure_eq_zero_or_top_of_subset_compl_sigmaFiniteSet Set.inter_subset_right
  /-
    🎉 no goals
  -/


/-- The restriction of an s-finite measure `μ` to `μ.sigmaFiniteSet` is sigma-finite. -/
instance : SigmaFinite (μ.restrict μ.sigmaFiniteSet) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    ⊢ MeasureTheory.SigmaFinite (μ.restrict μ.sigmaFiniteSet)
  -/
  rw [Measure.sigmaFiniteSet]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s t : Set α
    ⊢ MeasureTheory.SigmaFinite (μ.restrict (μ.sigmaFiniteSetWRT μ))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma sigmaFinite_of_measure_compl_sigmaFiniteSet_eq_zero (h : μ μ.sigmaFiniteSetᶜ = 0) :
    SigmaFinite μ := by
  rw [← Measure.restrict_add_restrict_compl (μ := μ) (measurableSet_sigmaFiniteSet (μ := μ)),
    Measure.restrict_eq_zero.mpr h, add_zero]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    h : Eq (μ (HasCompl.compl μ.sigmaFiniteSet)) 0
    ⊢ MeasureTheory.SigmaFinite (μ.restrict μ.sigmaFiniteSet)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
lemma measure_compl_sigmaFiniteSet (μ : Measure α) [SigmaFinite μ] : μ μ.sigmaFiniteSetᶜ = 0 :=
  measure_compl_sigmaFiniteSetWRT Measure.AbsolutelyContinuous.rfl


/-- An s-finite measure `μ` is sigma-finite iff `μ μ.sigmaFiniteSetᶜ = 0`. -/
lemma measure_compl_sigmaFiniteSet_eq_zero_iff_sigmaFinite (μ : Measure α) :
    μ μ.sigmaFiniteSetᶜ = 0 ↔ SigmaFinite μ :=
  ⟨sigmaFinite_of_measure_compl_sigmaFiniteSet_eq_zero, fun _ ↦ measure_compl_sigmaFiniteSet μ⟩


