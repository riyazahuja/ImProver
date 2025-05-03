/-- If a function `f : α → β` is such that the level sets `{f < p}` and `{q < f}` have measurable
supersets which are disjoint up to measure zero when `p < q`, then `f` is almost-everywhere
measurable. It is even enough to have this for `p` and `q` in a countable dense set. -/
theorem MeasureTheory.aemeasurable_of_exist_almost_disjoint_supersets {α : Type*}
    {m : MeasurableSpace α} (μ : Measure α) {β : Type*} [CompleteLinearOrder β] [DenselyOrdered β]
    [TopologicalSpace β] [OrderTopology β] [SecondCountableTopology β] [MeasurableSpace β]
    [BorelSpace β] (s : Set β) (s_count : s.Countable) (s_dense : Dense s) (f : α → β)
    (h : ∀ p ∈ s, ∀ q ∈ s, p < q → ∃ u v, MeasurableSet u ∧ MeasurableSet v ∧
      { x | f x < p } ⊆ u ∧ { x | q < f x } ⊆ v ∧ μ (u ∩ v) = 0) :
    AEMeasurable f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝⁶ : CompleteLinearOrder β
    inst✝⁵ : DenselyOrdered β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    inst✝² : SecondCountableTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    s : Set β
    s_count : s.Countable
    s_dense : Dense s
    f : α → β
    h : ∀ (p : β), Membership.mem s p → ∀ (q : β), Membership.mem s q → LT.lt p q  …
    ⊢ AEMeasurable f μ
  -/
  haveI : Encodable s := s_count.toEncodable
  have h' : ∀ p q, ∃ u v, MeasurableSet u ∧ MeasurableSet v ∧
      { x | f x < p } ⊆ u ∧ { x | q < f x } ⊆ v ∧ (p ∈ s → q ∈ s → p < q → μ (u ∩ v) = 0) := by
    intro p q
    by_cases H : p ∈ s ∧ q ∈ s ∧ p < q
    · rcases h p H.1 q H.2.1 H.2.2 with ⟨u, v, hu, hv, h'u, h'v, hμ⟩
      exact ⟨u, v, hu, hv, h'u, h'v, fun _ _ _ => hμ⟩
    · refine
        ⟨univ, univ, MeasurableSet.univ, MeasurableSet.univ, subset_univ _, subset_univ _,
          fun ps qs pq => ?_⟩
      simp only [not_and] at H
      exact (H ps qs pq).elim
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝⁶ : CompleteLinearOrder β
    inst✝⁵ : DenselyOrdered β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    inst✝² : SecondCountableTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    s : Set β
    s_count : s.Countable
    s_dense : Dense s
    f : α → β
    h : ∀ (p : β), Membership.mem s p → ∀ (q : β), Membership.mem s q → LT.lt p q  …
    this : Encodable ↑s
    h' : ∀ (p q : β), Exists fun u => Exists fun v => And (MeasurableSet u) (And ( …
    ⊢ AEMeasurable f μ
  -/
  choose! u v huv using h'
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝⁶ : CompleteLinearOrder β
    inst✝⁵ : DenselyOrdered β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    inst✝² : SecondCountableTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    s : Set β
    s_count : s.Countable
    s_dense : Dense s
    f : α → β
    h : ∀ (p : β), Membership.mem s p → ∀ (q : β), Membership.mem s q → LT.lt p q  …
    this : Encodable ↑s
    u v : β → β → Set α
    huv : ∀ (p q : β), And (MeasurableSet (u p q)) (And (MeasurableSet (v p q)) (A …
    ⊢ AEMeasurable f μ
  -/
  let u' : β → Set α := fun p => ⋂ q ∈ s ∩ Ioi p, u p q
  have u'_meas : ∀ i, MeasurableSet (u' i) := by
    intro i
    exact MeasurableSet.biInter (s_count.mono inter_subset_left) fun b _ => (huv i b).1
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝⁶ : CompleteLinearOrder β
    inst✝⁵ : DenselyOrdered β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    inst✝² : SecondCountableTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    s : Set β
    s_count : s.Countable
    s_dense : Dense s
    f : α → β
    h : ∀ (p : β), Membership.mem s p → ∀ (q : β), Membership.mem s q → LT.lt p q  …
    this : Encodable ↑s
    u v : β → β → Set α
    huv : ∀ (p q : β), And (MeasurableSet (u p q)) (And (MeasurableSet (v p q)) (A …
    u' : β → Set α := fun p => Set.iInter fun q => Set.iInter fun h => u p q
    u'_meas : ∀ (i : β), MeasurableSet (u' i)
    ⊢ AEMeasurable f μ
  -/
  let f' : α → β := fun x => ⨅ i : s, piecewise (u' i) (fun _ => (i : β)) (fun _ => (⊤ : β)) x
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝⁶ : CompleteLinearOrder β
    inst✝⁵ : DenselyOrdered β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    inst✝² : SecondCountableTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    s : Set β
    s_count : s.Countable
    s_dense : Dense s
    f : α → β
    h : ∀ (p : β), Membership.mem s p → ∀ (q : β), Membership.mem s q → LT.lt p q  …
    this : Encodable ↑s
    u v : β → β → Set α
    huv : ∀ (p q : β), And (MeasurableSet (u p q)) (And (MeasurableSet (v p q)) (A …
    u' : β → Set α := fun p => Set.iInter fun q => Set.iInter fun h => u p q
    u'_meas : ∀ (i : β), MeasurableSet (u' i)
    f' : α → β := fun x => iInf fun i => (u' ↑i).piecewise (fun x => ↑i) (fun x => …
    ⊢ AEMeasurable f μ
  -/
  have f'_meas : Measurable f' := by fun_prop (disch := aesop)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝⁶ : CompleteLinearOrder β
    inst✝⁵ : DenselyOrdered β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    inst✝² : SecondCountableTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    s : Set β
    s_count : s.Countable
    s_dense : Dense s
    f : α → β
    h : ∀ (p : β), Membership.mem s p → ∀ (q : β), Membership.mem s q → LT.lt p q  …
    this : Encodable ↑s
    u v : β → β → Set α
    huv : ∀ (p q : β), And (MeasurableSet (u p q)) (And (MeasurableSet (v p q)) (A …
    u' : β → Set α := fun p => Set.iInter fun q => Set.iInter fun h => u p q
    u'_meas : ∀ (i : β), MeasurableSet (u' i)
    f' : α → β := fun x => iInf fun i => (u' ↑i).piecewise (fun x => ↑i) (fun x => …
    f'_meas : Measurable f'
    ⊢ AEMeasurable f μ
  -/
  let t := ⋃ (p : s) (q : ↥(s ∩ Ioi p)), u' p ∩ v p q
  have μt : μ t ≤ 0 :=
    calc
      μ t ≤ ∑' (p : s) (q : ↥(s ∩ Ioi p)), μ (u' p ∩ v p q) := by
        refine (measure_iUnion_le _).trans ?_
        refine ENNReal.tsum_le_tsum fun p => ?_
        haveI := (s_count.mono (s.inter_subset_left (t := Ioi ↑p))).to_subtype
        apply measure_iUnion_le
      _ ≤ ∑' (p : s) (q : ↥(s ∩ Ioi p)), μ (u p q ∩ v p q) := by
        gcongr with p q
        exact biInter_subset_of_mem q.2
      _ = ∑' (p : s) (_ : ↥(s ∩ Ioi p)), (0 : ℝ≥0∞) := by
        congr
        ext1 p
        congr
        ext1 q
        exact (huv p q).2.2.2.2 p.2 q.2.1 q.2.2
      _ = 0 := by simp only [tsum_zero]
  have ff' : ∀ᵐ x ∂μ, f x = f' x := by
    have : ∀ᵐ x ∂μ, x ∉ t := by
      have : μ t = 0 := le_antisymm μt bot_le
      change μ _ = 0
      convert this
      ext y
      simp only [not_exists, exists_prop, mem_setOf_eq, mem_compl_iff, not_not_mem]
    filter_upwards [this] with x hx
    apply (iInf_eq_of_forall_ge_of_forall_gt_exists_lt _ _).symm
    · intro i
      by_cases H : x ∈ u' i
      swap
      · simp only [H, le_top, not_false_iff, piecewise_eq_of_not_mem]
      simp only [H, piecewise_eq_of_mem]
      contrapose! hx
      obtain ⟨r, ⟨xr, rq⟩, rs⟩ : ∃ r, r ∈ Ioo (i : β) (f x) ∩ s :=
        dense_iff_inter_open.1 s_dense (Ioo i (f x)) isOpen_Ioo (nonempty_Ioo.2 hx)
      have A : x ∈ v i r := (huv i r).2.2.2.1 rq
      refine mem_iUnion.2 ⟨i, ?_⟩
      refine mem_iUnion.2 ⟨⟨r, ⟨rs, xr⟩⟩, ?_⟩
      exact ⟨H, A⟩
    · intro q hq
      obtain ⟨r, ⟨xr, rq⟩, rs⟩ : ∃ r, r ∈ Ioo (f x) q ∩ s :=
        dense_iff_inter_open.1 s_dense (Ioo (f x) q) isOpen_Ioo (nonempty_Ioo.2 hq)
      refine ⟨⟨r, rs⟩, ?_⟩
      have A : x ∈ u' r := mem_biInter fun i _ => (huv r i).2.2.1 xr
      simp only [A, rq, piecewise_eq_of_mem, Subtype.coe_mk]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_2
    inst✝⁶ : CompleteLinearOrder β
    inst✝⁵ : DenselyOrdered β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    inst✝² : SecondCountableTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    s : Set β
    s_count : s.Countable
    s_dense : Dense s
    f : α → β
    h : ∀ (p : β), Membership.mem s p → ∀ (q : β), Membership.mem s q → LT.lt p q  …
    this : Encodable ↑s
    u v : β → β → Set α
    huv : ∀ (p q : β), And (MeasurableSet (u p q)) (And (MeasurableSet (v p q)) (A …
    u' : β → Set α := fun p => Set.iInter fun q => Set.iInter fun h => u p q
    u'_meas : ∀ (i : β), MeasurableSet (u' i)
    f' : α → β := fun x => iInf fun i => (u' ↑i).piecewise (fun x => ↑i) (fun x => …
    f'_meas : Measurable f'
    t : Set α := Set.iUnion fun p => Set.iUnion fun q => Inter.inter (u' ↑p) (v ↑p …
    μt : LE.le (μ t) 0
    ff' : Filter.Eventually (fun x => Eq (f x) (f' x)) (MeasureTheory.ae μ)
    ⊢ AEMeasurable f μ
  -/
  exact ⟨f', f'_meas, ff'⟩
  /-
    🎉 no goals
  -/


/-- If a function `f : α → ℝ≥0∞` is such that the level sets `{f < p}` and `{q < f}` have measurable
supersets which are disjoint up to measure zero when `p` and `q` are finite numbers satisfying
`p < q`, then `f` is almost-everywhere measurable. -/
theorem ENNReal.aemeasurable_of_exist_almost_disjoint_supersets {α : Type*} {m : MeasurableSpace α}
    (μ : Measure α) (f : α → ℝ≥0∞)
    (h : ∀ (p : ℝ≥0) (q : ℝ≥0), p < q →
      ∃ u v, MeasurableSet u ∧ MeasurableSet v ∧
        { x | f x < p } ⊆ u ∧ { x | (q : ℝ≥0∞) < f x } ⊆ v ∧ μ (u ∩ v) = 0) :
    AEMeasurable f μ := by
  obtain ⟨s, s_count, s_dense, _, s_top⟩ :
    ∃ s : Set ℝ≥0∞, s.Countable ∧ Dense s ∧ 0 ∉ s ∧ ∞ ∉ s :=
    ENNReal.exists_countable_dense_no_zero_top
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : ∀ (p q : NNReal), LT.lt p q → Exists fun u => Exists fun v => And (Measura …
    s : Set ENNReal
    s_count : s.Countable
    s_dense : Dense s
    left✝ : Not (Membership.mem s 0)
    s_top : Not (Membership.mem s Top.top)
    ⊢ AEMeasurable f μ
  -/
  have I : ∀ x ∈ s, x ≠ ∞ := fun x xs hx => s_top (hx ▸ xs)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : ∀ (p q : NNReal), LT.lt p q → Exists fun u => Exists fun v => And (Measura …
    s : Set ENNReal
    s_count : s.Countable
    s_dense : Dense s
    left✝ : Not (Membership.mem s 0)
    s_top : Not (Membership.mem s Top.top)
    I : ∀ (x : ENNReal), Membership.mem s x → Ne x Top.top
    ⊢ AEMeasurable f μ
  -/
  apply MeasureTheory.aemeasurable_of_exist_almost_disjoint_supersets μ s s_count s_dense _
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : ∀ (p q : NNReal), LT.lt p q → Exists fun u => Exists fun v => And (Measura …
    s : Set ENNReal
    s_count : s.Countable
    s_dense : Dense s
    left✝ : Not (Membership.mem s 0)
    s_top : Not (Membership.mem s Top.top)
    I : ∀ (x : ENNReal), Membership.mem s x → Ne x Top.top
    ⊢ ∀ (p : ENNReal), Membership.mem s p → ∀ (q : ENNReal), Membership.mem s q →  …
  -/
  rintro p hp q hq hpq
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : ∀ (p q : NNReal), LT.lt p q → Exists fun u => Exists fun v => And (Measura …
    s : Set ENNReal
    s_count : s.Countable
    s_dense : Dense s
    left✝ : Not (Membership.mem s 0)
    s_top : Not (Membership.mem s Top.top)
    I : ∀ (x : ENNReal), Membership.mem s x → Ne x Top.top
    p : ENNReal
    hp : Membership.mem s p
    q : ENNReal
    hq : Membership.mem s q
    hpq : LT.lt p q
    ⊢ Exists fun u => Exists fun v => And (MeasurableSet u) (And (MeasurableSet v) …
  -/
  lift p to ℝ≥0 using I p hp
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : ∀ (p q : NNReal), LT.lt p q → Exists fun u => Exists fun v => And (Measura …
    s : Set ENNReal
    s_count : s.Countable
    s_dense : Dense s
    left✝ : Not (Membership.mem s 0)
    s_top : Not (Membership.mem s Top.top)
    I : ∀ (x : ENNReal), Membership.mem s x → Ne x Top.top
    q : ENNReal
    hq : Membership.mem s q
    p : NNReal
    hp : Membership.mem s ↑p
    hpq : LT.lt (↑p) q
    ⊢ Exists fun u => Exists fun v => And (MeasurableSet u) (And (MeasurableSet v) …
  -/
  lift q to ℝ≥0 using I q hq
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : ∀ (p q : NNReal), LT.lt p q → Exists fun u => Exists fun v => And (Measura …
    s : Set ENNReal
    s_count : s.Countable
    s_dense : Dense s
    left✝ : Not (Membership.mem s 0)
    s_top : Not (Membership.mem s Top.top)
    I : ∀ (x : ENNReal), Membership.mem s x → Ne x Top.top
    p : NNReal
    hp : Membership.mem s ↑p
    q : NNReal
    hq : Membership.mem s ↑q
    hpq : LT.lt ↑p ↑q
    ⊢ Exists fun u => Exists fun v => And (MeasurableSet u) (And (MeasurableSet v) …
  -/
  exact h p q (ENNReal.coe_lt_coe.1 hpq)
  /-
    🎉 no goals
  -/

