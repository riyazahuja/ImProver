/-- We can trivially extend a function defined on a subclass of objects (with codomain `ℝ≥0∞`)
  to all objects by defining it to be `∞` on the objects not in the class. -/
def extend (s : α) : ℝ≥0∞ :=
  ⨅ h : P s, m s h


                                                               /-
                                                                 α : Type u_1
                                                                 P : α → Prop
                                                                 m : (s : α) → P s → ENNReal
                                                                 s : α
                                                                 h : P s
                                                                 ⊢ Eq (MeasureTheory.extend m s) (m s h)
                                                               -/
theorem extend_eq {s : α} (h : P s) : extend m s = m s h := by simp [extend, h]
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                                /-
                                                                  α : Type u_1
                                                                  P : α → Prop
                                                                  m : (s : α) → P s → ENNReal
                                                                  s : α
                                                                  h : Not (P s)
                                                                  ⊢ Eq (MeasureTheory.extend m s) Top.top
                                                                -/
theorem extend_eq_top {s : α} (h : ¬P s) : extend m s = ∞ := by simp [extend, h]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem smul_extend {R} [Zero R] [SMulWithZero R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    [NoZeroSMulDivisors R ℝ≥0∞] {c : R} (hc : c ≠ 0) :
    c • extend m = extend fun s h => c • m s h := by
  classical
  ext1 s
  dsimp [extend]
  by_cases h : P s
  · simp [h]
  · simp [h, ENNReal.smul_top, hc]


theorem le_extend {s : α} (h : P s) : m s h ≤ extend m s := by
  /-
    α : Type u_1
    P : α → Prop
    m : (s : α) → P s → ENNReal
    s : α
    h : P s
    ⊢ LE.le (m s h) (MeasureTheory.extend m s)
  -/
  simp only [extend, le_iInf_iff]
  /-
    α : Type u_1
    P : α → Prop
    m : (s : α) → P s → ENNReal
    s : α
    h : P s
    ⊢ ∀ (i : P s), LE.le (m s h) (m s i)
  -/
  intro
  /-
    α : Type u_1
    P : α → Prop
    m : (s : α) → P s → ENNReal
    s : α
    h i✝ : P s
    ⊢ LE.le (m s h) (m s i✝)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- TODO: why this is a bad `congr` lemma?

theorem extend_congr {β : Type*} {Pb : β → Prop} {mb : ∀ s : β, Pb s → ℝ≥0∞} {sa : α} {sb : β}
    (hP : P sa ↔ Pb sb) (hm : ∀ (ha : P sa) (hb : Pb sb), m sa ha = mb sb hb) :
    extend m sa = extend mb sb :=
  iInf_congr_Prop hP fun _h => hm _ _


@[simp]
theorem extend_top {α : Type*} {P : α → Prop} : extend (fun _ _ => ∞ : ∀ s : α, P s → ℝ≥0∞) = ⊤ :=
  funext fun _ => iInf_eq_top.mpr fun _ => rfl


theorem extend_iUnion_nat {f : ℕ → Set α} (hm : ∀ i, P (f i))
    (mU : m (⋃ i, f i) (PU hm) = ∑' i, m (f i) (hm i)) :
    extend m (⋃ i, f i) = ∑' i, extend m (f i) :=
  (extend_eq _ _).trans <|
    mU.trans <| by
      /-
        α : Type u_1
        P : Set α → Prop
        m : (s : Set α) → P s → ENNReal
        PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
        f : Nat → Set α
        hm : ∀ (i : Nat), P (f i)
        mU : Eq (m (Set.iUnion fun i => f i) ⋯) (tsum fun i => m (f i) ⋯)
        ⊢ Eq (tsum fun i => m (f i) ⋯) (tsum fun i => MeasureTheory.extend m (f i))
      -/
      congr with i
      /-
        case e_f.h
        α : Type u_1
        P : Set α → Prop
        m : (s : Set α) → P s → ENNReal
        PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
        f : Nat → Set α
        hm : ∀ (i : Nat), P (f i)
        mU : Eq (m (Set.iUnion fun i => f i) ⋯) (tsum fun i => m (f i) ⋯)
        i : Nat
        ⊢ Eq (m (f i) ⋯) (MeasureTheory.extend m (f i))
      -/
      rw [extend_eq]
      /-
        🎉 no goals
      -/


include P0 m0 in
theorem extend_empty : extend m ∅ = 0 :=
  (extend_eq _ P0).trans m0


include PU msU in
theorem extend_iUnion_le_tsum_nat' (s : ℕ → Set α) :
    extend m (⋃ i, s i) ≤ ∑' i, extend m (s i) := by
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
    s : Nat → Set α
    ⊢ LE.le (MeasureTheory.extend m (Set.iUnion fun i => s i)) (tsum fun i => Meas …
  -/
  by_cases h : ∀ i, P (s i)
    /-
      case pos
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      s : Nat → Set α
      h : ∀ (i : Nat), P (s i)
      ⊢ LE.le (MeasureTheory.extend m (Set.iUnion fun i => s i)) (tsum fun i => Meas …
    -/
  · rw [extend_eq _ (PU h), congr_arg tsum _]
      /-
        case pos
        α : Type u_1
        P : Set α → Prop
        m : (s : Set α) → P s → ENNReal
        PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
        msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
        s : Nat → Set α
        h : ∀ (i : Nat), P (s i)
        ⊢ LE.le (m (Set.iUnion fun i => s i) ⋯) (tsum ?m.9112)
      -/
    · apply msU h
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      s : Nat → Set α
      h : ∀ (i : Nat), P (s i)
      ⊢ Eq (fun i => MeasureTheory.extend m (s i)) fun i => m (s i) ⋯
    -/
    funext i
    /-
      case h
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      s : Nat → Set α
      h : ∀ (i : Nat), P (s i)
      i : Nat
      ⊢ Eq (MeasureTheory.extend m (s i)) (m (s i) ⋯)
    -/
    apply extend_eq _ (h i)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      s : Nat → Set α
      h : Not (∀ (i : Nat), P (s i))
      ⊢ LE.le (MeasureTheory.extend m (Set.iUnion fun i => s i)) (tsum fun i => Meas …
    -/
  · cases' not_forall.1 h with i hi
    /-
      case neg.intro
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      s : Nat → Set α
      h : Not (∀ (i : Nat), P (s i))
      i : Nat
      hi : Not (P (s i))
      ⊢ LE.le (MeasureTheory.extend m (Set.iUnion fun i => s i)) (tsum fun i => Meas …
    -/
    exact le_trans (le_iInf fun h => hi.elim h) (ENNReal.le_tsum i)
    /-
      🎉 no goals
    -/


include m_mono in
theorem extend_mono' ⦃s₁ s₂ : Set α⦄ (h₁ : P s₁) (hs : s₁ ⊆ s₂) : extend m s₁ ≤ extend m s₂ := by
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s₁ s₂ : Set α
    h₁ : P s₁
    hs : HasSubset.Subset s₁ s₂
    ⊢ LE.le (MeasureTheory.extend m s₁) (MeasureTheory.extend m s₂)
  -/
  refine le_iInf ?_
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s₁ s₂ : Set α
    h₁ : P s₁
    hs : HasSubset.Subset s₁ s₂
    ⊢ ∀ (i : P s₂), LE.le (MeasureTheory.extend m s₁) (m s₂ i)
  -/
  intro h₂
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s₁ s₂ : Set α
    h₁ : P s₁
    hs : HasSubset.Subset s₁ s₂
    h₂ : P s₂
    ⊢ LE.le (MeasureTheory.extend m s₁) (m s₂ h₂)
  -/
  rw [extend_eq m h₁]
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s₁ s₂ : Set α
    h₁ : P s₁
    hs : HasSubset.Subset s₁ s₂
    h₂ : P s₂
    ⊢ LE.le (m s₁ h₁) (m s₂ h₂)
  -/
  exact m_mono h₁ h₂ hs
  /-
    🎉 no goals
  -/


include P0 m0 PU mU in
theorem extend_iUnion {β} [Countable β] {f : β → Set α} (hd : Pairwise (Disjoint on f))
    (hm : ∀ i, P (f i)) : extend m (⋃ i, f i) = ∑' i, extend m (f i) := by
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), Pairwise (Function.onFun …
    β : Type u_2
    inst✝ : Countable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    hm : ∀ (i : β), P (f i)
    ⊢ Eq (MeasureTheory.extend m (Set.iUnion fun i => f i)) (tsum fun i => Measure …
  -/
  cases nonempty_encodable β
  /-
    case intro
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), Pairwise (Function.onFun …
    β : Type u_2
    inst✝ : Countable β
    f : β → Set α
    hd : Pairwise (Function.onFun Disjoint f)
    hm : ∀ (i : β), P (f i)
    val✝ : Encodable β
    ⊢ Eq (MeasureTheory.extend m (Set.iUnion fun i => f i)) (tsum fun i => Measure …
  -/
  rw [← Encodable.iUnion_decode₂, ← tsum_iUnion_decode₂]
  · exact
      extend_iUnion_nat PU (fun n => Encodable.iUnion_decode₂_cases P0 hm)
        (mU _ (Encodable.iUnion_decode₂_disjoint_on hd))
    /-
      case intro.m0
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), Pairwise (Function.onFun …
      β : Type u_2
      inst✝ : Countable β
      f : β → Set α
      hd : Pairwise (Function.onFun Disjoint f)
      hm : ∀ (i : β), P (f i)
      val✝ : Encodable β
      ⊢ Eq (MeasureTheory.extend m EmptyCollection.emptyCollection) 0
    -/
  · exact extend_empty P0 m0
    /-
      🎉 no goals
    -/


include P0 m0 PU mU in
theorem extend_union {s₁ s₂ : Set α} (hd : Disjoint s₁ s₂) (h₁ : P s₁) (h₂ : P s₂) :
    extend m (s₁ ∪ s₂) = extend m s₁ + extend m s₂ := by
  rw [union_eq_iUnion,
    extend_iUnion P0 m0 PU mU (pairwise_disjoint_on_bool.2 hd) (Bool.forall_bool.2 ⟨h₂, h₁⟩),
    tsum_fintype]
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), Pairwise (Function.onFun …
    s₁ s₂ : Set α
    hd : Disjoint s₁ s₂
    h₁ : P s₁
    h₂ : P s₂
    ⊢ Eq (Finset.univ.sum fun b => MeasureTheory.extend m (cond b s₁ s₂)) (HAdd.hA …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given an arbitrary function on a subset of sets, we can define the outer measure corresponding
  to it (this is the unique maximal outer measure that is at most `m` on the domain of `m`). -/
def inducedOuterMeasure : OuterMeasure α :=
  OuterMeasure.ofFunction (extend m) (extend_empty P0 m0)


theorem le_inducedOuterMeasure {μ : OuterMeasure α} :
    μ ≤ inducedOuterMeasure m P0 m0 ↔ ∀ (s) (hs : P s), μ s ≤ m s hs :=
  le_ofFunction.trans <| forall_congr' fun _s => le_iInf_iff


/-- If `P u` is `False` for any set `u` that has nonempty intersection both with `s` and `t`, then
`μ (s ∪ t) = μ s + μ t`, where `μ = inducedOuterMeasure m P0 m0`.

E.g., if `α` is an (e)metric space and `P u = diam u < r`, then this lemma implies that
`μ (s ∪ t) = μ s + μ t` on any two sets such that `r ≤ edist x y` for all `x ∈ s` and `y ∈ t`. -/
theorem inducedOuterMeasure_union_of_false_of_nonempty_inter {s t : Set α}
    (h : ∀ u, (s ∩ u).Nonempty → (t ∩ u).Nonempty → ¬P u) :
    inducedOuterMeasure m P0 m0 (s ∪ t) =
      inducedOuterMeasure m P0 m0 s + inducedOuterMeasure m P0 m0 t :=
  ofFunction_union_of_top_of_nonempty_inter fun u hsu htu => @iInf_of_empty _ _ _ ⟨h u hsu htu⟩ _


theorem inducedOuterMeasure_eq_extend' {s : Set α} (hs : P s) :
    inducedOuterMeasure m P0 m0 s = extend m s :=
  ofFunction_eq s (fun _t => extend_mono' m_mono hs) (extend_iUnion_le_tsum_nat' PU msU)


theorem inducedOuterMeasure_eq' {s : Set α} (hs : P s) : inducedOuterMeasure m P0 m0 s = m s hs :=
  (inducedOuterMeasure_eq_extend' PU msU m_mono hs).trans <| extend_eq _ _


theorem inducedOuterMeasure_eq_iInf (s : Set α) :
    inducedOuterMeasure m P0 m0 s = ⨅ (t : Set α) (ht : P t) (_ : s ⊆ t), m t ht := by
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s : Set α
    ⊢ Eq ((MeasureTheory.inducedOuterMeasure m P0 m0) s) (iInf fun t => iInf fun h …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      ⊢ LE.le ((MeasureTheory.inducedOuterMeasure m P0 m0) s) (iInf fun t => iInf fu …
    -/
  · simp only [le_iInf_iff]
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      ⊢ ∀ (i : Set α) (i_1 : P i), HasSubset.Subset s i → LE.le ((MeasureTheory.indu …
    -/
    intro t ht hs
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s t : Set α
      ht : P t
      hs : HasSubset.Subset s t
      ⊢ LE.le ((MeasureTheory.inducedOuterMeasure m P0 m0) s) (m t ht)
    -/
    refine le_trans (measure_mono hs) ?_
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s t : Set α
      ht : P t
      hs : HasSubset.Subset s t
      ⊢ LE.le ((MeasureTheory.inducedOuterMeasure m P0 m0) t) (m t ht)
    -/
    exact le_of_eq (inducedOuterMeasure_eq' _ msU m_mono _)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      ⊢ LE.le (iInf fun t => iInf fun ht => iInf fun x => m t ht) ((MeasureTheory.in …
    -/
  · refine le_iInf ?_
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      ⊢ ∀ (i : Nat → Set α), LE.le (iInf fun t => iInf fun ht => iInf fun x => m t h …
    -/
    intro f
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      f : Nat → Set α
      ⊢ LE.le (iInf fun t => iInf fun ht => iInf fun x => m t ht) (iInf fun x => tsu …
    -/
    refine le_iInf ?_
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      f : Nat → Set α
      ⊢ HasSubset.Subset s (Set.iUnion fun i => f i) → LE.le (iInf fun t => iInf fun …
    -/
    intro hf
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      f : Nat → Set α
      hf : HasSubset.Subset s (Set.iUnion fun i => f i)
      ⊢ LE.le (iInf fun t => iInf fun ht => iInf fun x => m t ht) (tsum fun i => Mea …
    -/
    refine le_trans ?_ (extend_iUnion_le_tsum_nat' _ msU _)
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      f : Nat → Set α
      hf : HasSubset.Subset s (Set.iUnion fun i => f i)
      ⊢ LE.le (iInf fun t => iInf fun ht => iInf fun x => m t ht) (MeasureTheory.ext …
    -/
    refine le_iInf ?_
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      f : Nat → Set α
      hf : HasSubset.Subset s (Set.iUnion fun i => f i)
      ⊢ ∀ (i : P (Set.iUnion fun i => f i)), LE.le (iInf fun t => iInf fun ht => iIn …
    -/
    intro h2f
    /-
      case a
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      f : Nat → Set α
      hf : HasSubset.Subset s (Set.iUnion fun i => f i)
      h2f : P (Set.iUnion fun i => f i)
      ⊢ LE.le (iInf fun t => iInf fun ht => iInf fun x => m t ht) (m (Set.iUnion fun …
    -/
    exact iInf_le_of_le _ (iInf_le_of_le h2f <| iInf_le _ hf)
    /-
      🎉 no goals
    -/


theorem inducedOuterMeasure_preimage (f : α ≃ α) (Pm : ∀ s : Set α, P (f ⁻¹' s) ↔ P s)
    (mm : ∀ (s : Set α) (hs : P s), m (f ⁻¹' s) ((Pm _).mpr hs) = m s hs) {A : Set α} :
    inducedOuterMeasure m P0 m0 (f ⁻¹' A) = inducedOuterMeasure m P0 m0 A := by
    /-
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      f : Equiv α α
      Pm : ∀ (s : Set α), Iff (P (Set.preimage (⇑f) s)) (P s)
      mm : ∀ (s : Set α) (hs : P s), Eq (m (Set.preimage (⇑f) s) ⋯) (m s hs)
      A : Set α
      ⊢ Eq ((MeasureTheory.inducedOuterMeasure m P0 m0) (Set.preimage (⇑f) A)) ((Mea …
    -/
    rw [inducedOuterMeasure_eq_iInf _ msU m_mono, inducedOuterMeasure_eq_iInf _ msU m_mono]; symm
    /-
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      f : Equiv α α
      Pm : ∀ (s : Set α), Iff (P (Set.preimage (⇑f) s)) (P s)
      mm : ∀ (s : Set α) (hs : P s), Eq (m (Set.preimage (⇑f) s) ⋯) (m s hs)
      A : Set α
      ⊢ Eq (iInf fun t => iInf fun ht => iInf fun x => m t ht) (iInf fun t => iInf f …
    -/
    refine f.injective.preimage_surjective.iInf_congr (preimage f) fun s => ?_
    /-
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      f : Equiv α α
      Pm : ∀ (s : Set α), Iff (P (Set.preimage (⇑f) s)) (P s)
      mm : ∀ (s : Set α) (hs : P s), Eq (m (Set.preimage (⇑f) s) ⋯) (m s hs)
      A s : Set α
      ⊢ Eq (iInf fun ht => iInf fun x => m (Set.preimage (⇑f) s) ht) (iInf fun ht => …
    -/
    refine iInf_congr_Prop (Pm s) ?_; intro hs
    /-
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      f : Equiv α α
      Pm : ∀ (s : Set α), Iff (P (Set.preimage (⇑f) s)) (P s)
      mm : ∀ (s : Set α) (hs : P s), Eq (m (Set.preimage (⇑f) s) ⋯) (m s hs)
      A s : Set α
      hs : P s
      ⊢ Eq (iInf fun x => m (Set.preimage (⇑f) s) ⋯) (iInf fun x => m s hs)
    -/
    refine iInf_congr_Prop f.surjective.preimage_subset_preimage_iff ?_
    /-
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      f : Equiv α α
      Pm : ∀ (s : Set α), Iff (P (Set.preimage (⇑f) s)) (P s)
      mm : ∀ (s : Set α) (hs : P s), Eq (m (Set.preimage (⇑f) s) ⋯) (m s hs)
      A s : Set α
      hs : P s
      ⊢ HasSubset.Subset A s → Eq (m (Set.preimage (⇑f) s) ⋯) (m s hs)
    -/
    intro _; exact mm s hs
             /-
               🎉 no goals
             -/


theorem inducedOuterMeasure_exists_set {s : Set α} (hs : inducedOuterMeasure m P0 m0 s ≠ ∞)
    {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ t : Set α,
      P t ∧ s ⊆ t ∧ inducedOuterMeasure m P0 m0 t ≤ inducedOuterMeasure m P0 m0 s + ε := by
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s : Set α
    hs : Ne ((MeasureTheory.inducedOuterMeasure m P0 m0) s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun t => And (P t) (And (HasSubset.Subset s t) (LE.le ((MeasureTheory …
  -/
  have h := ENNReal.lt_add_right hs hε
  conv at h =>
    lhs
    rw [inducedOuterMeasure_eq_iInf _ msU m_mono]
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s : Set α
    hs : Ne ((MeasureTheory.inducedOuterMeasure m P0 m0) s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    h : LT.lt (iInf fun t => iInf fun ht => iInf fun x => m t ht) (HAdd.hAdd ((Mea …
    ⊢ Exists fun t => And (P t) (And (HasSubset.Subset s t) (LE.le ((MeasureTheory …
  -/
  simp only [iInf_lt_iff] at h
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s : Set α
    hs : Ne ((MeasureTheory.inducedOuterMeasure m P0 m0) s) Top.top
    ε : ENNReal
    hε : Ne ε 0
    h : Exists fun i => Exists fun h => Exists fun i_1 => LT.lt (m i ⋯) (HAdd.hAdd …
    ⊢ Exists fun t => And (P t) (And (HasSubset.Subset s t) (LE.le ((MeasureTheory …
  -/
  rcases h with ⟨t, h1t, h2t, h3t⟩
  exact
    ⟨t, h1t, h2t, le_trans (le_of_eq <| inducedOuterMeasure_eq' _ msU m_mono h1t) (le_of_lt h3t)⟩


/-- To test whether `s` is Carathéodory-measurable we only need to check the sets `t` for which
  `P t` holds. See `ofFunction_caratheodory` for another way to show the Carathéodory-measurability
  of `s`.
-/
theorem inducedOuterMeasure_caratheodory (s : Set α) :
    MeasurableSet[(inducedOuterMeasure m P0 m0).caratheodory] s ↔
      ∀ t : Set α,
        P t →
          inducedOuterMeasure m P0 m0 (t ∩ s) + inducedOuterMeasure m P0 m0 (t \ s) ≤
            inducedOuterMeasure m P0 m0 t := by
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s : Set α
    ⊢ Iff (MeasurableSet s) (∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory …
  -/
  rw [isCaratheodory_iff_le]
  /-
    α : Type u_1
    P : Set α → Prop
    m : (s : Set α) → P s → ENNReal
    P0 : P EmptyCollection.emptyCollection
    m0 : Eq (m EmptyCollection.emptyCollection P0) 0
    PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
    msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
    m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
    s : Set α
    ⊢ Iff (∀ (t : Set α), LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      ⊢ (∀ (t : Set α), LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0 …
    -/
  · intro h t _ht
    /-
      case mp
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m …
      t : Set α
      _ht : P t
      ⊢ LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) (Inter.inter t …
    -/
    exact h t
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      ⊢ (∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m …
    -/
  · intro h u
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u : Set α
      ⊢ LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) (Inter.inter u …
    -/
    conv_rhs => rw [inducedOuterMeasure_eq_iInf _ msU m_mono]
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u : Set α
      ⊢ LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) (Inter.inter u …
    -/
    refine le_iInf ?_
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u : Set α
      ⊢ ∀ (i : Set α), LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) …
    -/
    intro t
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u t : Set α
      ⊢ LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) (Inter.inter u …
    -/
    refine le_iInf ?_
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u t : Set α
      ⊢ ∀ (i : P t), LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) ( …
    -/
    intro ht
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u t : Set α
      ht : P t
      ⊢ LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) (Inter.inter u …
    -/
    refine le_iInf ?_
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u t : Set α
      ht : P t
      ⊢ HasSubset.Subset u t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
    -/
    intro h2t
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u t : Set α
      ht : P t
      h2t : HasSubset.Subset u t
      ⊢ LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) (Inter.inter u …
    -/
    refine le_trans ?_ ((h t ht).trans_eq <| inducedOuterMeasure_eq' _ msU m_mono ht)
    /-
      case mpr
      α : Type u_1
      P : Set α → Prop
      m : (s : Set α) → P s → ENNReal
      P0 : P EmptyCollection.emptyCollection
      m0 : Eq (m EmptyCollection.emptyCollection P0) 0
      PU : ∀ ⦃f : Nat → Set α⦄, (∀ (i : Nat), P (f i)) → P (Set.iUnion fun i => f i)
      msU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), P (f i)), LE.le (m (Set.iUnion fu …
      m_mono : ∀ ⦃s₁ s₂ : Set α⦄ (hs₁ : P s₁) (hs₂ : P s₂), HasSubset.Subset s₁ s₂ → …
      s : Set α
      h : ∀ (t : Set α), P t → LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure  …
      u t : Set α
      ht : P t
      h2t : HasSubset.Subset u t
      ⊢ LE.le (HAdd.hAdd ((MeasureTheory.inducedOuterMeasure m P0 m0) (Inter.inter u …
    -/
    gcongr
    /-
      🎉 no goals
    -/


theorem extend_mono {s₁ s₂ : Set α} (h₁ : MeasurableSet s₁) (hs : s₁ ⊆ s₂) :
    extend m s₁ ≤ extend m s₂ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    hs : HasSubset.Subset s₁ s₂
    ⊢ LE.le (MeasureTheory.extend m s₁) (MeasureTheory.extend m s₂)
  -/
  refine le_iInf ?_; intro h₂
  have :=
    extend_union MeasurableSet.empty m0 MeasurableSet.iUnion mU disjoint_sdiff_self_right h₁
      (h₂.diff h₁)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    hs : HasSubset.Subset s₁ s₂
    h₂ : MeasurableSet s₂
    this : Eq (MeasureTheory.extend m (Union.union s₁ (SDiff.sdiff s₂ s₁))) (HAdd. …
    ⊢ LE.le (MeasureTheory.extend m s₁) (m s₂ h₂)
  -/
  rw [union_diff_cancel hs] at this
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    hs : HasSubset.Subset s₁ s₂
    h₂ : MeasurableSet s₂
    this : Eq (MeasureTheory.extend m s₂) (HAdd.hAdd (MeasureTheory.extend m s₁) ( …
    ⊢ LE.le (MeasureTheory.extend m s₁) (m s₂ h₂)
  -/
  rw [← extend_eq m]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    hs : HasSubset.Subset s₁ s₂
    h₂ : MeasurableSet s₂
    this : Eq (MeasureTheory.extend m s₂) (HAdd.hAdd (MeasureTheory.extend m s₁) ( …
    ⊢ LE.le (MeasureTheory.extend m s₁) (MeasureTheory.extend m s₂)
  -/
  exact le_iff_exists_add.2 ⟨_, this⟩
  /-
    🎉 no goals
  -/


theorem extend_iUnion_le_tsum_nat : ∀ s : ℕ → Set α,
    extend m (⋃ i, s i) ≤ ∑' i, extend m (s i) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    ⊢ ∀ (s : Nat → Set α), LE.le (MeasureTheory.extend m (Set.iUnion fun i => s i) …
  -/
  refine extend_iUnion_le_tsum_nat' MeasurableSet.iUnion ?_; intro f h
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    f : Nat → Set α
    h : ∀ (i : Nat), MeasurableSet (f i)
    ⊢ LE.le (m (Set.iUnion fun i => f i) ⋯) (tsum fun i => m (f i) ⋯)
  -/
  simp (config := { singlePass := true }) only [iUnion_disjointed.symm]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    f : Nat → Set α
    h : ∀ (i : Nat), MeasurableSet (f i)
    ⊢ LE.le (m (Set.iUnion fun n => disjointed f n) ⋯) (tsum fun i => m (f i) ⋯)
  -/
  rw [mU (MeasurableSet.disjointed h) (disjoint_disjointed _)]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    f : Nat → Set α
    h : ∀ (i : Nat), MeasurableSet (f i)
    ⊢ LE.le (tsum fun i => m (disjointed f i) ⋯) (tsum fun i => m (f i) ⋯)
  -/
  refine ENNReal.tsum_le_tsum fun i => ?_
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    f : Nat → Set α
    h : ∀ (i : Nat), MeasurableSet (f i)
    i : Nat
    ⊢ LE.le (m (disjointed f i) ⋯) (m (f i) ⋯)
  -/
  rw [← extend_eq m, ← extend_eq m]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : (s : Set α) → MeasurableSet s → ENNReal
    m0 : Eq (m EmptyCollection.emptyCollection ⋯) 0
    mU : ∀ ⦃f : Nat → Set α⦄ (hm : ∀ (i : Nat), MeasurableSet (f i)), Pairwise (Fu …
    f : Nat → Set α
    h : ∀ (i : Nat), MeasurableSet (f i)
    i : Nat
    ⊢ LE.le (MeasureTheory.extend m (disjointed f i)) (MeasureTheory.extend m (f i))
  -/
  exact extend_mono m0 mU (MeasurableSet.disjointed h _) (disjointed_le f _)
  /-
    🎉 no goals
  -/


theorem inducedOuterMeasure_eq_extend {s : Set α} (hs : MeasurableSet s) :
    inducedOuterMeasure m MeasurableSet.empty m0 s = extend m s :=
  ofFunction_eq s (fun _t => extend_mono m0 mU hs) (extend_iUnion_le_tsum_nat m0 mU)


theorem inducedOuterMeasure_eq {s : Set α} (hs : MeasurableSet s) :
    inducedOuterMeasure m MeasurableSet.empty m0 s = m s hs :=
  (inducedOuterMeasure_eq_extend m0 mU hs).trans <| extend_eq _ _


/-- Given an outer measure `m` we can forget its value on non-measurable sets, and then consider
  `m.trim`, the unique maximal outer measure less than that function. -/
def trim : OuterMeasure α :=
  inducedOuterMeasure (P := MeasurableSet) (fun s _ => m s) .empty m.empty


theorem le_trim_iff {m₁ m₂ : OuterMeasure α} :
    m₁ ≤ m₂.trim ↔ ∀ s, MeasurableSet s → m₁ s ≤ m₂ s :=
  le_inducedOuterMeasure


theorem le_trim : m ≤ m.trim := le_trim_iff.2 fun _ _ ↦ le_rfl


@[simp] -- Porting note: added `simp`
theorem trim_eq {s : Set α} (hs : MeasurableSet s) : m.trim s = m s :=
  inducedOuterMeasure_eq' MeasurableSet.iUnion (fun f _hf => measure_iUnion_le f)
    (fun _ _ _ _ h => measure_mono h) hs


theorem trim_congr {m₁ m₂ : OuterMeasure α} (H : ∀ {s : Set α}, MeasurableSet s → m₁ s = m₂ s) :
    m₁.trim = m₂.trim := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m₁ m₂ : MeasureTheory.OuterMeasure α
    H : ∀ {s : Set α}, MeasurableSet s → Eq (m₁ s) (m₂ s)
    ⊢ Eq m₁.trim m₂.trim
  -/
  simp +contextual only [trim, H]
  /-
    🎉 no goals
  -/


@[mono]
theorem trim_mono : Monotone (trim : OuterMeasure α → OuterMeasure α) := fun _m₁ _m₂ H _s =>
  iInf₂_mono fun _f _hs => ENNReal.tsum_le_tsum fun _b => iInf_mono fun _hf => H _


/-- `OuterMeasure.trim` is antitone in the σ-algebra. -/
theorem trim_anti_measurableSpace {α} (m : OuterMeasure α) {m0 m1 : MeasurableSpace α}
    (h : m0 ≤ m1) : @trim _ m1 m ≤ @trim _ m0 m := by
  /-
    α : Type u_2
    m : MeasureTheory.OuterMeasure α
    m0 m1 : MeasurableSpace α
    h : LE.le m0 m1
    ⊢ LE.le m.trim m.trim
  -/
  simp only [le_trim_iff]
  /-
    α : Type u_2
    m : MeasureTheory.OuterMeasure α
    m0 m1 : MeasurableSpace α
    h : LE.le m0 m1
    ⊢ ∀ (s : Set α), MeasurableSet s → LE.le (m.trim s) (m s)
  -/
  intro s hs
  /-
    α : Type u_2
    m : MeasureTheory.OuterMeasure α
    m0 m1 : MeasurableSpace α
    h : LE.le m0 m1
    s : Set α
    hs : MeasurableSet s
    ⊢ LE.le (m.trim s) (m s)
  -/
  rw [trim_eq _ (h s hs)]
  /-
    🎉 no goals
  -/


theorem trim_le_trim_iff {m₁ m₂ : OuterMeasure α} :
    m₁.trim ≤ m₂.trim ↔ ∀ s, MeasurableSet s → m₁ s ≤ m₂ s :=
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : MeasurableSpace α
                                                      m₁ m₂ : MeasureTheory.OuterMeasure α
                                                      s : Set α
                                                      hs : MeasurableSet s
                                                      ⊢ Iff (LE.le (m₁.trim s) (m₂ s)) (LE.le (m₁ s) (m₂ s))
                                                    -/
  le_trim_iff.trans <| forall₂_congr fun s hs => by rw [trim_eq _ hs]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem trim_eq_trim_iff {m₁ m₂ : OuterMeasure α} :
    m₁.trim = m₂.trim ↔ ∀ s, MeasurableSet s → m₁ s = m₂ s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m₁ m₂ : MeasureTheory.OuterMeasure α
    ⊢ Iff (Eq m₁.trim m₂.trim) (∀ (s : Set α), MeasurableSet s → Eq (m₁ s) (m₂ s))
  -/
  simp only [le_antisymm_iff, trim_le_trim_iff, forall_and]
  /-
    🎉 no goals
  -/


theorem trim_eq_iInf (s : Set α) : m.trim s = ⨅ (t) (_ : s ⊆ t) (_ : MeasurableSet t), m t := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Eq (m.trim s) (iInf fun t => iInf fun x => iInf fun x => m t)
  -/
  simp (config := { singlePass := true }) only [iInf_comm]
  exact
    inducedOuterMeasure_eq_iInf MeasurableSet.iUnion (fun f _ => measure_iUnion_le f)
      (fun _ _ _ _ h => measure_mono h) s


theorem trim_eq_iInf' (s : Set α) : m.trim s = ⨅ t : { t // s ⊆ t ∧ MeasurableSet t }, m t := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Eq (m.trim s) (iInf fun t => m ↑t)
  -/
  simp [iInf_subtype, iInf_and, trim_eq_iInf]
  /-
    🎉 no goals
  -/


theorem trim_trim (m : OuterMeasure α) : m.trim.trim = m.trim :=
  trim_eq_trim_iff.2 fun _s => m.trim_eq


@[simp]
theorem trim_top : (⊤ : OuterMeasure α).trim = ⊤ :=
  top_unique <| le_trim _


@[simp]
theorem trim_zero : (0 : OuterMeasure α).trim = 0 :=
  ext fun s =>
    le_antisymm
      ((measure_mono (subset_univ s)).trans_eq <| trim_eq _ MeasurableSet.univ)
      (zero_le _)


theorem trim_sum_ge {ι} (m : ι → OuterMeasure α) : (sum fun i => (m i).trim) ≤ (sum m).trim :=
  fun s => by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    ι : Type u_2
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ LE.le ((MeasureTheory.OuterMeasure.sum fun i => (m i).trim) s) ((MeasureTheo …
  -/
  simp only [sum_apply, trim_eq_iInf, le_iInf_iff]
  exact fun t st ht =>
    ENNReal.tsum_le_tsum fun i => iInf_le_of_le t <| iInf_le_of_le st <| iInf_le _ ht


theorem exists_measurable_superset_eq_trim (m : OuterMeasure α) (s : Set α) :
    ∃ t, s ⊆ t ∧ MeasurableSet t ∧ m t = m.trim s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
  -/
  simp only [trim_eq_iInf]; set ms := ⨅ (t : Set α) (_ : s ⊆ t) (_ : MeasurableSet t), m t
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.OuterMeasure α
    s : Set α
    ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
  -/
  by_cases hs : ms = ∞
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.OuterMeasure α
      s : Set α
      ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
      hs : Eq ms Top.top
      ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
    -/
  · simp only [hs]
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.OuterMeasure α
      s : Set α
      ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
      hs : Eq ms Top.top
      ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
    -/
    simp only [iInf_eq_top, ms] at hs
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.OuterMeasure α
      s : Set α
      ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
      hs : ∀ (i : Set α), HasSubset.Subset s i → MeasurableSet i → Eq (m i) Top.top
      ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
    -/
    exact ⟨univ, subset_univ s, MeasurableSet.univ, hs _ (subset_univ s) MeasurableSet.univ⟩
    /-
      🎉 no goals
    -/
  · have : ∀ r > ms, ∃ t, s ⊆ t ∧ MeasurableSet t ∧ m t < r := by
      intro r hs
      have : ∃t, MeasurableSet t ∧ s ⊆ t ∧ m t < r := by simpa [ms, iInf_lt_iff] using hs
      rcases this with ⟨t, hmt, hin, hlt⟩
      exists t
    have : ∀ n : ℕ, ∃ t, s ⊆ t ∧ MeasurableSet t ∧ m t < ms + (n : ℝ≥0∞)⁻¹ := by
      intro n
      refine this _ (ENNReal.lt_add_right hs ?_)
      simp
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.OuterMeasure α
      s : Set α
      ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
      hs : Not (Eq ms Top.top)
      this✝ : ∀ (r : ENNReal), GT.gt r ms → Exists fun t => And (HasSubset.Subset s  …
      this : ∀ (n : Nat), Exists fun t => And (HasSubset.Subset s t) (And (Measurabl …
      ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
    -/
    choose t hsub hm hm' using this
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.OuterMeasure α
      s : Set α
      ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
      hs : Not (Eq ms Top.top)
      this : ∀ (r : ENNReal), GT.gt r ms → Exists fun t => And (HasSubset.Subset s t …
      t : Nat → Set α
      hsub : ∀ (n : Nat), HasSubset.Subset s (t n)
      hm : ∀ (n : Nat), MeasurableSet (t n)
      hm' : ∀ (n : Nat), LT.lt (m (t n)) (HAdd.hAdd ms (Inv.inv ↑n))
      ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
    -/
    refine ⟨⋂ n, t n, subset_iInter hsub, MeasurableSet.iInter hm, ?_⟩
    have : Tendsto (fun n : ℕ => ms + (n : ℝ≥0∞)⁻¹) atTop (𝓝 (ms + 0)) :=
      tendsto_const_nhds.add ENNReal.tendsto_inv_nat_nhds_zero
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.OuterMeasure α
      s : Set α
      ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
      hs : Not (Eq ms Top.top)
      this✝ : ∀ (r : ENNReal), GT.gt r ms → Exists fun t => And (HasSubset.Subset s  …
      t : Nat → Set α
      hsub : ∀ (n : Nat), HasSubset.Subset s (t n)
      hm : ∀ (n : Nat), MeasurableSet (t n)
      hm' : ∀ (n : Nat), LT.lt (m (t n)) (HAdd.hAdd ms (Inv.inv ↑n))
      this : Filter.Tendsto (fun n => HAdd.hAdd ms (Inv.inv ↑n)) Filter.atTop (nhds  …
      ⊢ Eq (m (Set.iInter fun n => t n)) ms
    -/
    rw [add_zero] at this
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      m : MeasureTheory.OuterMeasure α
      s : Set α
      ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
      hs : Not (Eq ms Top.top)
      this✝ : ∀ (r : ENNReal), GT.gt r ms → Exists fun t => And (HasSubset.Subset s  …
      t : Nat → Set α
      hsub : ∀ (n : Nat), HasSubset.Subset s (t n)
      hm : ∀ (n : Nat), MeasurableSet (t n)
      hm' : ∀ (n : Nat), LT.lt (m (t n)) (HAdd.hAdd ms (Inv.inv ↑n))
      this : Filter.Tendsto (fun n => HAdd.hAdd ms (Inv.inv ↑n)) Filter.atTop (nhds  …
      ⊢ Eq (m (Set.iInter fun n => t n)) ms
    -/
    refine le_antisymm (ge_of_tendsto' this fun n => ?_) ?_
      /-
        case neg.refine_1
        α : Type u_1
        inst✝ : MeasurableSpace α
        m : MeasureTheory.OuterMeasure α
        s : Set α
        ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
        hs : Not (Eq ms Top.top)
        this✝ : ∀ (r : ENNReal), GT.gt r ms → Exists fun t => And (HasSubset.Subset s  …
        t : Nat → Set α
        hsub : ∀ (n : Nat), HasSubset.Subset s (t n)
        hm : ∀ (n : Nat), MeasurableSet (t n)
        hm' : ∀ (n : Nat), LT.lt (m (t n)) (HAdd.hAdd ms (Inv.inv ↑n))
        this : Filter.Tendsto (fun n => HAdd.hAdd ms (Inv.inv ↑n)) Filter.atTop (nhds  …
        n : Nat
        ⊢ LE.le (m (Set.iInter fun n => t n)) (HAdd.hAdd ms (Inv.inv ↑n))
      -/
    · exact le_trans (measure_mono <| iInter_subset t n) (hm' n).le
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        α : Type u_1
        inst✝ : MeasurableSpace α
        m : MeasureTheory.OuterMeasure α
        s : Set α
        ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
        hs : Not (Eq ms Top.top)
        this✝ : ∀ (r : ENNReal), GT.gt r ms → Exists fun t => And (HasSubset.Subset s  …
        t : Nat → Set α
        hsub : ∀ (n : Nat), HasSubset.Subset s (t n)
        hm : ∀ (n : Nat), MeasurableSet (t n)
        hm' : ∀ (n : Nat), LT.lt (m (t n)) (HAdd.hAdd ms (Inv.inv ↑n))
        this : Filter.Tendsto (fun n => HAdd.hAdd ms (Inv.inv ↑n)) Filter.atTop (nhds  …
        ⊢ LE.le ms (m (Set.iInter fun n => t n))
      -/
    · refine iInf_le_of_le (⋂ n, t n) ?_
      /-
        case neg.refine_2
        α : Type u_1
        inst✝ : MeasurableSpace α
        m : MeasureTheory.OuterMeasure α
        s : Set α
        ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
        hs : Not (Eq ms Top.top)
        this✝ : ∀ (r : ENNReal), GT.gt r ms → Exists fun t => And (HasSubset.Subset s  …
        t : Nat → Set α
        hsub : ∀ (n : Nat), HasSubset.Subset s (t n)
        hm : ∀ (n : Nat), MeasurableSet (t n)
        hm' : ∀ (n : Nat), LT.lt (m (t n)) (HAdd.hAdd ms (Inv.inv ↑n))
        this : Filter.Tendsto (fun n => HAdd.hAdd ms (Inv.inv ↑n)) Filter.atTop (nhds  …
        ⊢ LE.le (iInf fun x => iInf fun x => m (Set.iInter fun n => t n)) (m (Set.iInt …
      -/
      refine iInf_le_of_le (subset_iInter hsub) ?_
      /-
        case neg.refine_2
        α : Type u_1
        inst✝ : MeasurableSpace α
        m : MeasureTheory.OuterMeasure α
        s : Set α
        ms : ENNReal := iInf fun t => iInf fun x => iInf fun x => m t
        hs : Not (Eq ms Top.top)
        this✝ : ∀ (r : ENNReal), GT.gt r ms → Exists fun t => And (HasSubset.Subset s  …
        t : Nat → Set α
        hsub : ∀ (n : Nat), HasSubset.Subset s (t n)
        hm : ∀ (n : Nat), MeasurableSet (t n)
        hm' : ∀ (n : Nat), LT.lt (m (t n)) (HAdd.hAdd ms (Inv.inv ↑n))
        this : Filter.Tendsto (fun n => HAdd.hAdd ms (Inv.inv ↑n)) Filter.atTop (nhds  …
        ⊢ LE.le (iInf fun x => m (Set.iInter fun n => t n)) (m (Set.iInter fun n => t  …
      -/
      exact iInf_le _ (MeasurableSet.iInter hm)
      /-
        🎉 no goals
      -/


theorem exists_measurable_superset_of_trim_eq_zero {m : OuterMeasure α} {s : Set α}
    (h : m.trim s = 0) : ∃ t, s ⊆ t ∧ MeasurableSet t ∧ m t = 0 := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.OuterMeasure α
    s : Set α
    h : Eq (m.trim s) 0
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
  -/
  rcases exists_measurable_superset_eq_trim m s with ⟨t, hst, ht, hm⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    m : MeasureTheory.OuterMeasure α
    s : Set α
    h : Eq (m.trim s) 0
    t : Set α
    hst : HasSubset.Subset s t
    ht : MeasurableSet t
    hm : Eq (m t) (m.trim s)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (Eq (m t)  …
  -/
  exact ⟨t, hst, ht, h ▸ hm⟩
  /-
    🎉 no goals
  -/


/-- If `μ i` is a countable family of outer measures, then for every set `s` there exists
a measurable set `t ⊇ s` such that `μ i t = (μ i).trim s` for all `i`. -/
theorem exists_measurable_superset_forall_eq_trim {ι} [Countable ι] (μ : ι → OuterMeasure α)
    (s : Set α) : ∃ t, s ⊆ t ∧ MeasurableSet t ∧ ∀ i, μ i t = (μ i).trim s := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (∀ (i : ι) …
  -/
  choose t hst ht hμt using fun i => (μ i).exists_measurable_superset_eq_trim s
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    s : Set α
    t : ι → Set α
    hst : ∀ (i : ι), HasSubset.Subset s (t i)
    ht : ∀ (i : ι), MeasurableSet (t i)
    hμt : ∀ (i : ι), Eq ((μ i) (t i)) ((μ i).trim s)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (∀ (i : ι) …
  -/
  replace hst := subset_iInter hst
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    s : Set α
    t : ι → Set α
    ht : ∀ (i : ι), MeasurableSet (t i)
    hμt : ∀ (i : ι), Eq ((μ i) (t i)) ((μ i).trim s)
    hst : HasSubset.Subset s (Set.iInter fun i => t i)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (∀ (i : ι) …
  -/
  replace ht := MeasurableSet.iInter ht
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    s : Set α
    t : ι → Set α
    hμt : ∀ (i : ι), Eq ((μ i) (t i)) ((μ i).trim s)
    hst : HasSubset.Subset s (Set.iInter fun i => t i)
    ht : MeasurableSet (Set.iInter fun b => t b)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (MeasurableSet t) (∀ (i : ι) …
  -/
  refine ⟨⋂ i, t i, hst, ht, fun i => le_antisymm ?_ ?_⟩
  /-
    case refine_1
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    s : Set α
    t : ι → Set α
    hμt : ∀ (i : ι), Eq ((μ i) (t i)) ((μ i).trim s)
    hst : HasSubset.Subset s (Set.iInter fun i => t i)
    ht : MeasurableSet (Set.iInter fun b => t b)
    i : ι
    ⊢ LE.le ((μ i) (Set.iInter fun i => t i)) ((μ i).trim s)
  -/
  exacts [hμt i ▸ (μ i).mono (iInter_subset _ _), (measure_mono hst).trans_eq ((μ i).trim_eq ht)]
  /-
    🎉 no goals
  -/


/-- If `m₁ s = op (m₂ s) (m₃ s)` for all `s`, then the same is true for `m₁.trim`, `m₂.trim`,
and `m₃ s`. -/
theorem trim_binop {m₁ m₂ m₃ : OuterMeasure α} {op : ℝ≥0∞ → ℝ≥0∞ → ℝ≥0∞}
    (h : ∀ s, m₁ s = op (m₂ s) (m₃ s)) (s : Set α) : m₁.trim s = op (m₂.trim s) (m₃.trim s) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    m₁ m₂ m₃ : MeasureTheory.OuterMeasure α
    op : ENNReal → ENNReal → ENNReal
    h : ∀ (s : Set α), Eq (m₁ s) (op (m₂ s) (m₃ s))
    s : Set α
    ⊢ Eq (m₁.trim s) (op (m₂.trim s) (m₃.trim s))
  -/
  rcases exists_measurable_superset_forall_eq_trim ![m₁, m₂, m₃] s with ⟨t, _hst, _ht, htm⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    m₁ m₂ m₃ : MeasureTheory.OuterMeasure α
    op : ENNReal → ENNReal → ENNReal
    h : ∀ (s : Set α), Eq (m₁ s) (op (m₂ s) (m₃ s))
    s t : Set α
    _hst : HasSubset.Subset s t
    _ht : MeasurableSet t
    htm : ∀ (i : Fin (Nat.succ 0).succ.succ), Eq ((Matrix.vecCons m₁ (Matrix.vecCo …
    ⊢ Eq (m₁.trim s) (op (m₂.trim s) (m₃.trim s))
  -/
  simp only [Fin.forall_iff_succ, Matrix.cons_val_zero, Matrix.cons_val_succ] at htm
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    m₁ m₂ m₃ : MeasureTheory.OuterMeasure α
    op : ENNReal → ENNReal → ENNReal
    h : ∀ (s : Set α), Eq (m₁ s) (op (m₂ s) (m₃ s))
    s t : Set α
    _hst : HasSubset.Subset s t
    _ht : MeasurableSet t
    htm : And (Eq (m₁ t) (m₁.trim s)) (And (Eq (m₂ t) (m₂.trim s)) (And (Eq (m₃ t) …
    ⊢ Eq (m₁.trim s) (op (m₂.trim s) (m₃.trim s))
  -/
  rw [← htm.1, ← htm.2.1, ← htm.2.2.1, h]
  /-
    🎉 no goals
  -/


/-- If `m₁ s = op (m₂ s)` for all `s`, then the same is true for `m₁.trim` and `m₂.trim`. -/
theorem trim_op {m₁ m₂ : OuterMeasure α} {op : ℝ≥0∞ → ℝ≥0∞} (h : ∀ s, m₁ s = op (m₂ s))
    (s : Set α) : m₁.trim s = op (m₂.trim s) :=
  @trim_binop α _ m₁ m₂ 0 (fun a _b => op a) h s


/-- `trim` is additive. -/
theorem trim_add (m₁ m₂ : OuterMeasure α) : (m₁ + m₂).trim = m₁.trim + m₂.trim :=
  ext <| trim_binop (add_apply m₁ m₂)


/-- `trim` respects scalar multiplication. -/
theorem trim_smul {R : Type*} [SMul R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞] (c : R)
    (m : OuterMeasure α) : (c • m).trim = c • m.trim :=
  ext <| trim_op (smul_apply c m)


/-- `trim` sends the supremum of two outer measures to the supremum of the trimmed measures. -/
theorem trim_sup (m₁ m₂ : OuterMeasure α) : (m₁ ⊔ m₂).trim = m₁.trim ⊔ m₂.trim :=
  ext fun s => (trim_binop (sup_apply m₁ m₂) s).trans (sup_apply _ _ _).symm


/-- `trim` sends the supremum of a countable family of outer measures to the supremum
of the trimmed measures. -/
theorem trim_iSup {ι} [Countable ι] (μ : ι → OuterMeasure α) :
    trim (⨆ i, μ i) = ⨆ i, trim (μ i) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    ⊢ Eq (iSup fun i => μ i).trim (iSup fun i => (μ i).trim)
  -/
  simp_rw [← @iSup_plift_down _ ι]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    ⊢ Eq (iSup fun i => μ i.down).trim (iSup fun i => (μ i.down).trim)
  -/
  ext1 s
  obtain ⟨t, _, _, hμt⟩ :=
    exists_measurable_superset_forall_eq_trim
      (Option.elim' (⨆ i, μ (PLift.down i)) (μ ∘ PLift.down)) s
  /-
    case h.intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    s t : Set α
    left✝¹ : HasSubset.Subset s t
    left✝ : MeasurableSet t
    hμt : ∀ (i : Option (PLift ι)), Eq ((Option.elim' (iSup fun i => μ i.down) (Fu …
    ⊢ Eq ((iSup fun i => μ i.down).trim s) ((iSup fun i => (μ i.down).trim) s)
  -/
  simp only [Option.forall, Option.elim'] at hμt
  /-
    case h.intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    s t : Set α
    left✝¹ : HasSubset.Subset s t
    left✝ : MeasurableSet t
    hμt : And (Eq ((iSup fun i => μ i.down) t) ((iSup fun i => μ i.down).trim s))  …
    ⊢ Eq ((iSup fun i => μ i.down).trim s) ((iSup fun i => (μ i.down).trim) s)
  -/
  simp only [iSup_apply, ← hμt.1]
  /-
    case h.intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    ι : Sort u_2
    inst✝ : Countable ι
    μ : ι → MeasureTheory.OuterMeasure α
    s t : Set α
    left✝¹ : HasSubset.Subset s t
    left✝ : MeasurableSet t
    hμt : And (Eq ((iSup fun i => μ i.down) t) ((iSup fun i => μ i.down).trim s))  …
    ⊢ Eq (iSup fun i => (μ i.down) t) (iSup fun i => (μ i.down).trim s)
  -/
  exact iSup_congr hμt.2
  /-
    🎉 no goals
  -/


/-- The trimmed property of a measure μ states that `μ.toOuterMeasure.trim = μ.toOuterMeasure`.
This theorem shows that a restricted trimmed outer measure is a trimmed outer measure. -/
theorem restrict_trim {μ : OuterMeasure α} {s : Set α} (hs : MeasurableSet s) :
    (restrict s μ).trim = restrict s μ.trim := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.OuterMeasure α
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.OuterMeasure.restrict s) μ).trim ((MeasureTheory.OuterMea …
  -/
  refine le_antisymm (fun t => ?_) (le_trim_iff.2 fun t ht => ?_)
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.OuterMeasure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      ⊢ LE.le (((MeasureTheory.OuterMeasure.restrict s) μ).trim t) (((MeasureTheory. …
    -/
  · rw [restrict_apply]
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.OuterMeasure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      ⊢ LE.le (((MeasureTheory.OuterMeasure.restrict s) μ).trim t) (μ.trim (Inter.in …
    -/
    rcases μ.exists_measurable_superset_eq_trim (t ∩ s) with ⟨t', htt', ht', hμt'⟩
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.OuterMeasure α
      s : Set α
      hs : MeasurableSet s
      t t' : Set α
      htt' : HasSubset.Subset (Inter.inter t s) t'
      ht' : MeasurableSet t'
      hμt' : Eq (μ t') (μ.trim (Inter.inter t s))
      ⊢ LE.le (((MeasureTheory.OuterMeasure.restrict s) μ).trim t) (μ.trim (Inter.in …
    -/
    rw [← hμt']
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.OuterMeasure α
      s : Set α
      hs : MeasurableSet s
      t t' : Set α
      htt' : HasSubset.Subset (Inter.inter t s) t'
      ht' : MeasurableSet t'
      hμt' : Eq (μ t') (μ.trim (Inter.inter t s))
      ⊢ LE.le (((MeasureTheory.OuterMeasure.restrict s) μ).trim t) (μ t')
    -/
    rw [inter_subset] at htt'
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.OuterMeasure α
      s : Set α
      hs : MeasurableSet s
      t t' : Set α
      htt' : HasSubset.Subset t (Union.union (HasCompl.compl s) t')
      ht' : MeasurableSet t'
      hμt' : Eq (μ t') (μ.trim (Inter.inter t s))
      ⊢ LE.le (((MeasureTheory.OuterMeasure.restrict s) μ).trim t) (μ t')
    -/
    refine (measure_mono htt').trans ?_
    rw [trim_eq _ (hs.compl.union ht'), restrict_apply, union_inter_distrib_right, compl_inter_self,
      Set.empty_union]
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.OuterMeasure α
      s : Set α
      hs : MeasurableSet s
      t t' : Set α
      htt' : HasSubset.Subset t (Union.union (HasCompl.compl s) t')
      ht' : MeasurableSet t'
      hμt' : Eq (μ t') (μ.trim (Inter.inter t s))
      ⊢ LE.le (μ (Inter.inter t' s)) (μ t')
    -/
    exact measure_mono inter_subset_left
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.OuterMeasure α
      s : Set α
      hs : MeasurableSet s
      t : Set α
      ht : MeasurableSet t
      ⊢ LE.le (((MeasureTheory.OuterMeasure.restrict s) μ.trim) t) (((MeasureTheory. …
    -/
  · rw [restrict_apply, trim_eq _ (ht.inter hs), restrict_apply]
    /-
      🎉 no goals
    -/


