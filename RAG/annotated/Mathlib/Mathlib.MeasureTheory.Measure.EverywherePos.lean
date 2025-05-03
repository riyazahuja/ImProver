/-- A set `s` is *everywhere positive* (also called *self-supporting*) with respect to a
measure `μ` if it has positive measure around each of its points, i.e., if all neighborhoods `n`
of points of `s` satisfy `μ (s ∩ n) > 0`. -/
def IsEverywherePos (μ : Measure α) (s : Set α) : Prop :=
  ∀ x ∈ s, ∀ n ∈ 𝓝[s] x, 0 < μ n


/-- * The everywhere positive subset of a set is the subset made of those points all of whose
neighborhoods have positive measure inside the set. -/
def everywherePosSubset (μ : Measure α) (s : Set α) : Set α :=
  {x | x ∈ s ∧ ∀ n ∈ 𝓝[s] x, 0 < μ n}


lemma everywherePosSubset_subset (μ : Measure α) (s : Set α) : μ.everywherePosSubset s ⊆ s :=
  fun _x hx ↦ hx.1


/-- The everywhere positive subset of a set is obtained by removing an open set. -/
lemma exists_isOpen_everywherePosSubset_eq_diff (μ : Measure α) (s : Set α) :
    ∃ u, IsOpen u ∧ μ.everywherePosSubset s = s \ u := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ Exists fun u => And (IsOpen u) (Eq (μ.everywherePosSubset s) (SDiff.sdiff s  …
  -/
  refine ⟨{x | ∃ n ∈ 𝓝[s] x, μ n = 0}, ?_, by ext x; simp [everywherePosSubset, zero_lt_iff]⟩
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ IsOpen (setOf fun x => Exists fun n => And (Membership.mem (nhdsWithin x s)  …
  -/
  rw [isOpen_iff_mem_nhds]
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    ⊢ ∀ (x : α), Membership.mem (setOf fun x => Exists fun n => And (Membership.me …
  -/
  intro x ⟨n, ns, hx⟩
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    x : α
    n : Set α
    ns : Membership.mem (nhdsWithin x s) n
    hx : Eq (μ n) 0
    ⊢ Membership.mem (nhds x) (setOf fun x => Exists fun n => And (Membership.mem  …
  -/
  rcases mem_nhdsWithin_iff_exists_mem_nhds_inter.1 ns with ⟨v, vx, hv⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    x : α
    n : Set α
    ns : Membership.mem (nhdsWithin x s) n
    hx : Eq (μ n) 0
    v : Set α
    vx : Membership.mem (nhds x) v
    hv : HasSubset.Subset (Inter.inter v s) n
    ⊢ Membership.mem (nhds x) (setOf fun x => Exists fun n => And (Membership.mem  …
  -/
  rcases mem_nhds_iff.1 vx with ⟨w, wv, w_open, xw⟩
  have A : w ⊆ {x | ∃ n ∈ 𝓝[s] x, μ n = 0} := by
    intro y yw
    refine ⟨s ∩ w, inter_mem_nhdsWithin _ (w_open.mem_nhds yw), measure_mono_null ?_ hx⟩
    rw [inter_comm]
    exact (inter_subset_inter_left _ wv).trans hv
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    x : α
    n : Set α
    ns : Membership.mem (nhdsWithin x s) n
    hx : Eq (μ n) 0
    v : Set α
    vx : Membership.mem (nhds x) v
    hv : HasSubset.Subset (Inter.inter v s) n
    w : Set α
    wv : HasSubset.Subset w v
    w_open : IsOpen w
    xw : Membership.mem w x
    A : HasSubset.Subset w (setOf fun x => Exists fun n => And (Membership.mem (nh …
    ⊢ Membership.mem (nhds x) (setOf fun x => Exists fun n => And (Membership.mem  …
  -/
  have B : w ∈ 𝓝 x := w_open.mem_nhds xw
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    x : α
    n : Set α
    ns : Membership.mem (nhdsWithin x s) n
    hx : Eq (μ n) 0
    v : Set α
    vx : Membership.mem (nhds x) v
    hv : HasSubset.Subset (Inter.inter v s) n
    w : Set α
    wv : HasSubset.Subset w v
    w_open : IsOpen w
    xw : Membership.mem w x
    A : HasSubset.Subset w (setOf fun x => Exists fun n => And (Membership.mem (nh …
    B : Membership.mem (nhds x) w
    ⊢ Membership.mem (nhds x) (setOf fun x => Exists fun n => And (Membership.mem  …
  -/
  exact mem_of_superset B A
  /-
    🎉 no goals
  -/


protected lemma _root_.MeasurableSet.everywherePosSubset [OpensMeasurableSpace α]
    (hs : MeasurableSet s) :
    MeasurableSet (μ.everywherePosSubset s) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : OpensMeasurableSpace α
    hs : MeasurableSet s
    ⊢ MeasurableSet (μ.everywherePosSubset s)
  -/
  rcases exists_isOpen_everywherePosSubset_eq_diff μ s with ⟨u, u_open, hu⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : OpensMeasurableSpace α
    hs : MeasurableSet s
    u : Set α
    u_open : IsOpen u
    hu : Eq (μ.everywherePosSubset s) (SDiff.sdiff s u)
    ⊢ MeasurableSet (μ.everywherePosSubset s)
  -/
  rw [hu]
  /-
    case intro.intro
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : OpensMeasurableSpace α
    hs : MeasurableSet s
    u : Set α
    u_open : IsOpen u
    hu : Eq (μ.everywherePosSubset s) (SDiff.sdiff s u)
    ⊢ MeasurableSet (SDiff.sdiff s u)
  -/
  exact hs.diff u_open.measurableSet
  /-
    🎉 no goals
  -/


protected lemma _root_.IsClosed.everywherePosSubset (hs : IsClosed s) :
    IsClosed (μ.everywherePosSubset s) := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : IsClosed s
    ⊢ IsClosed (μ.everywherePosSubset s)
  -/
  rcases exists_isOpen_everywherePosSubset_eq_diff μ s with ⟨u, u_open, hu⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : IsClosed s
    u : Set α
    u_open : IsOpen u
    hu : Eq (μ.everywherePosSubset s) (SDiff.sdiff s u)
    ⊢ IsClosed (μ.everywherePosSubset s)
  -/
  rw [hu]
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : IsClosed s
    u : Set α
    u_open : IsOpen u
    hu : Eq (μ.everywherePosSubset s) (SDiff.sdiff s u)
    ⊢ IsClosed (SDiff.sdiff s u)
  -/
  exact hs.sdiff u_open
  /-
    🎉 no goals
  -/


protected lemma _root_.IsCompact.everywherePosSubset (hs : IsCompact s) :
    IsCompact (μ.everywherePosSubset s) := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : IsCompact s
    ⊢ IsCompact (μ.everywherePosSubset s)
  -/
  rcases exists_isOpen_everywherePosSubset_eq_diff μ s with ⟨u, u_open, hu⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : IsCompact s
    u : Set α
    u_open : IsOpen u
    hu : Eq (μ.everywherePosSubset s) (SDiff.sdiff s u)
    ⊢ IsCompact (μ.everywherePosSubset s)
  -/
  rw [hu]
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : IsCompact s
    u : Set α
    u_open : IsOpen u
    hu : Eq (μ.everywherePosSubset s) (SDiff.sdiff s u)
    ⊢ IsCompact (SDiff.sdiff s u)
  -/
  exact hs.diff u_open
  /-
    🎉 no goals
  -/


/-- Any compact set contained in `s \ μ.everywherePosSubset s` has zero measure. -/
lemma measure_eq_zero_of_subset_diff_everywherePosSubset
    (hk : IsCompact k) (h'k : k ⊆ s \ μ.everywherePosSubset s) : μ k = 0 := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s k : Set α
    hk : IsCompact k
    h'k : HasSubset.Subset k (SDiff.sdiff s (μ.everywherePosSubset s))
    ⊢ Eq (μ k) 0
  -/
  apply hk.induction_on (p := fun t ↦ μ t = 0)
    /-
      case he
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s k : Set α
      hk : IsCompact k
      h'k : HasSubset.Subset k (SDiff.sdiff s (μ.everywherePosSubset s))
      ⊢ Eq (μ EmptyCollection.emptyCollection) 0
    -/
  · exact measure_empty
    /-
      🎉 no goals
    -/
    /-
      case hmono
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s k : Set α
      hk : IsCompact k
      h'k : HasSubset.Subset k (SDiff.sdiff s (μ.everywherePosSubset s))
      ⊢ ∀ ⦃s t : Set α⦄, HasSubset.Subset s t → Eq (μ t) 0 → Eq (μ s) 0
    -/
  · exact fun s t hst ht ↦ measure_mono_null hst ht
    /-
      🎉 no goals
    -/
    /-
      case hunion
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s k : Set α
      hk : IsCompact k
      h'k : HasSubset.Subset k (SDiff.sdiff s (μ.everywherePosSubset s))
      ⊢ ∀ ⦃s t : Set α⦄, Eq (μ s) 0 → Eq (μ t) 0 → Eq (μ (Union.union s t)) 0
    -/
  · exact fun s t hs ht ↦ measure_union_null hs ht
    /-
      🎉 no goals
    -/
    /-
      case hnhds
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s k : Set α
      hk : IsCompact k
      h'k : HasSubset.Subset k (SDiff.sdiff s (μ.everywherePosSubset s))
      ⊢ ∀ (x : α), Membership.mem k x → Exists fun t => And (Membership.mem (nhdsWit …
    -/
  · intro x hx
    obtain ⟨u, ux, hu⟩ : ∃ u ∈ 𝓝[s] x, μ u = 0 := by
      simpa [everywherePosSubset, (h'k hx).1] using (h'k hx).2
    /-
      case hnhds.intro.intro
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s k : Set α
      hk : IsCompact k
      h'k : HasSubset.Subset k (SDiff.sdiff s (μ.everywherePosSubset s))
      x : α
      hx : Membership.mem k x
      u : Set α
      ux : Membership.mem (nhdsWithin x s) u
      hu : Eq (μ u) 0
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x k) t) (Eq (μ t) 0)
    -/
    exact ⟨u, nhdsWithin_mono x (h'k.trans diff_subset) ux, hu⟩
    /-
      🎉 no goals
    -/


/-- In a space with an inner regular measure, any measurable set coincides almost everywhere with
its everywhere positive subset. -/
lemma everywherePosSubset_ae_eq [OpensMeasurableSpace α] [InnerRegular μ] (hs : MeasurableSet s) :
    μ.everywherePosSubset s =ᵐ[μ] s := by
  simp only [ae_eq_set, diff_eq_empty.mpr (everywherePosSubset_subset μ s), measure_empty,
    true_and, (hs.diff hs.everywherePosSubset).measure_eq_iSup_isCompact, ENNReal.iSup_eq_zero]
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegular
    hs : MeasurableSet s
    ⊢ ∀ (i : Set α), HasSubset.Subset i (SDiff.sdiff s (μ.everywherePosSubset s))  …
  -/
  intro k hk h'k
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegular
    hs : MeasurableSet s
    k : Set α
    hk : HasSubset.Subset k (SDiff.sdiff s (μ.everywherePosSubset s))
    h'k : IsCompact k
    ⊢ Eq (μ k) 0
  -/
  exact measure_eq_zero_of_subset_diff_everywherePosSubset h'k hk
  /-
    🎉 no goals
  -/


/-- In a space with an inner regular measure for finite measure sets, any measurable set of finite
measure coincides almost everywhere with its everywhere positive subset. -/
lemma everywherePosSubset_ae_eq_of_measure_ne_top
    [OpensMeasurableSpace α] [InnerRegularCompactLTTop μ] (hs : MeasurableSet s) (h's : μ s ≠ ∞) :
    μ.everywherePosSubset s =ᵐ[μ] s := by
  have A : μ (s \ μ.everywherePosSubset s) ≠ ∞ :=
    ((measure_mono diff_subset).trans_lt h's.lt_top).ne
  simp only [ae_eq_set, diff_eq_empty.mpr (everywherePosSubset_subset μ s), measure_empty,
    true_and, (hs.diff hs.everywherePosSubset).measure_eq_iSup_isCompact_of_ne_top A,
    ENNReal.iSup_eq_zero]
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegularCompactLTTop
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    A : Ne (μ (SDiff.sdiff s (μ.everywherePosSubset s))) Top.top
    ⊢ ∀ (i : Set α), HasSubset.Subset i (SDiff.sdiff s (μ.everywherePosSubset s))  …
  -/
  intro k hk h'k
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegularCompactLTTop
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    A : Ne (μ (SDiff.sdiff s (μ.everywherePosSubset s))) Top.top
    k : Set α
    hk : HasSubset.Subset k (SDiff.sdiff s (μ.everywherePosSubset s))
    h'k : IsCompact k
    ⊢ Eq (μ k) 0
  -/
  exact measure_eq_zero_of_subset_diff_everywherePosSubset h'k hk
  /-
    🎉 no goals
  -/


/-- In a space with an inner regular measure, the everywhere positive subset of a measurable set
is itself everywhere positive. This is not obvious as `μ.everywherePosSubset s` is defined as
the points whose neighborhoods intersect `s` along positive measure subsets, but this does not
say they also intersect `μ.everywherePosSubset s` along positive measure subsets. -/
lemma isEverywherePos_everywherePosSubset
    [OpensMeasurableSpace α] [InnerRegular μ] (hs : MeasurableSet s) :
    μ.IsEverywherePos (μ.everywherePosSubset s) := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegular
    hs : MeasurableSet s
    ⊢ μ.IsEverywherePos (μ.everywherePosSubset s)
  -/
  intro x hx n hn
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegular
    hs : MeasurableSet s
    x : α
    hx : Membership.mem (μ.everywherePosSubset s) x
    n : Set α
    hn : Membership.mem (nhdsWithin x (μ.everywherePosSubset s)) n
    ⊢ LT.lt 0 (μ n)
  -/
  rcases mem_nhdsWithin_iff_exists_mem_nhds_inter.1 hn with ⟨u, u_mem, hu⟩
  have A : 0 < μ (u ∩ s) := by
    have : u ∩ s ∈ 𝓝[s] x := by rw [inter_comm]; exact inter_mem_nhdsWithin s u_mem
    exact hx.2 _ this
  have B : (u ∩ μ.everywherePosSubset s : Set α) =ᵐ[μ] (u ∩ s : Set α) :=
    ae_eq_set_inter (ae_eq_refl _) (everywherePosSubset_ae_eq hs)
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegular
    hs : MeasurableSet s
    x : α
    hx : Membership.mem (μ.everywherePosSubset s) x
    n : Set α
    hn : Membership.mem (nhdsWithin x (μ.everywherePosSubset s)) n
    u : Set α
    u_mem : Membership.mem (nhds x) u
    hu : HasSubset.Subset (Inter.inter u (μ.everywherePosSubset s)) n
    A : LT.lt 0 (μ (Inter.inter u s))
    B : (MeasureTheory.ae μ).EventuallyEq (Inter.inter u (μ.everywherePosSubset s) …
    ⊢ LT.lt 0 (μ n)
  -/
  rw [← B.measure_eq] at A
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegular
    hs : MeasurableSet s
    x : α
    hx : Membership.mem (μ.everywherePosSubset s) x
    n : Set α
    hn : Membership.mem (nhdsWithin x (μ.everywherePosSubset s)) n
    u : Set α
    u_mem : Membership.mem (nhds x) u
    hu : HasSubset.Subset (Inter.inter u (μ.everywherePosSubset s)) n
    A : LT.lt 0 (μ (Inter.inter u (μ.everywherePosSubset s)))
    B : (MeasureTheory.ae μ).EventuallyEq (Inter.inter u (μ.everywherePosSubset s) …
    ⊢ LT.lt 0 (μ n)
  -/
  exact A.trans_le (measure_mono hu)
  /-
    🎉 no goals
  -/


/-- In a space with an inner regular measure for finite measure sets, the everywhere positive subset
of a measurable set of finite measure is itself everywhere positive. This is not obvious as
`μ.everywherePosSubset s` is defined as the points whose neighborhoods intersect `s` along positive
measure subsets, but this does not say they also intersect `μ.everywherePosSubset s` along positive
measure subsets. -/
lemma isEverywherePos_everywherePosSubset_of_measure_ne_top
    [OpensMeasurableSpace α] [InnerRegularCompactLTTop μ] (hs : MeasurableSet s) (h's : μ s ≠ ∞) :
    μ.IsEverywherePos (μ.everywherePosSubset s) := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegularCompactLTTop
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    ⊢ μ.IsEverywherePos (μ.everywherePosSubset s)
  -/
  intro x hx n hn
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegularCompactLTTop
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    x : α
    hx : Membership.mem (μ.everywherePosSubset s) x
    n : Set α
    hn : Membership.mem (nhdsWithin x (μ.everywherePosSubset s)) n
    ⊢ LT.lt 0 (μ n)
  -/
  rcases mem_nhdsWithin_iff_exists_mem_nhds_inter.1 hn with ⟨u, u_mem, hu⟩
  have A : 0 < μ (u ∩ s) := by
    have : u ∩ s ∈ 𝓝[s] x := by rw [inter_comm]; exact inter_mem_nhdsWithin s u_mem
    exact hx.2 _ this
  have B : (u ∩ μ.everywherePosSubset s : Set α) =ᵐ[μ] (u ∩ s : Set α) :=
    ae_eq_set_inter (ae_eq_refl _) (everywherePosSubset_ae_eq_of_measure_ne_top hs h's)
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegularCompactLTTop
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    x : α
    hx : Membership.mem (μ.everywherePosSubset s) x
    n : Set α
    hn : Membership.mem (nhdsWithin x (μ.everywherePosSubset s)) n
    u : Set α
    u_mem : Membership.mem (nhds x) u
    hu : HasSubset.Subset (Inter.inter u (μ.everywherePosSubset s)) n
    A : LT.lt 0 (μ (Inter.inter u s))
    B : (MeasureTheory.ae μ).EventuallyEq (Inter.inter u (μ.everywherePosSubset s) …
    ⊢ LT.lt 0 (μ n)
  -/
  rw [← B.measure_eq] at A
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : OpensMeasurableSpace α
    inst✝ : μ.InnerRegularCompactLTTop
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    x : α
    hx : Membership.mem (μ.everywherePosSubset s) x
    n : Set α
    hn : Membership.mem (nhdsWithin x (μ.everywherePosSubset s)) n
    u : Set α
    u_mem : Membership.mem (nhds x) u
    hu : HasSubset.Subset (Inter.inter u (μ.everywherePosSubset s)) n
    A : LT.lt 0 (μ (Inter.inter u (μ.everywherePosSubset s)))
    B : (MeasureTheory.ae μ).EventuallyEq (Inter.inter u (μ.everywherePosSubset s) …
    ⊢ LT.lt 0 (μ n)
  -/
  exact A.trans_le (measure_mono hu)
  /-
    🎉 no goals
  -/


lemma IsEverywherePos.smul_measure (hs : IsEverywherePos μ s) {c : ℝ≥0∞} (hc : c ≠ 0) :
    IsEverywherePos (c • μ) s :=
                     /-
                       α : Type u_1
                       inst✝¹ : TopologicalSpace α
                       inst✝ : MeasurableSpace α
                       μ : MeasureTheory.Measure α
                       s : Set α
                       hs : μ.IsEverywherePos s
                       c : ENNReal
                       hc : Ne c 0
                       x : α
                       hx : Membership.mem s x
                       n : Set α
                       hn : Membership.mem (nhdsWithin x s) n
                       ⊢ LT.lt 0 ((HSMul.hSMul c μ) n)
                     -/
  fun x hx n hn ↦ by simpa [hc.bot_lt, hs x hx n hn] using hc.bot_lt
                     /-
                       🎉 no goals
                     -/


lemma IsEverywherePos.smul_measure_nnreal (hs : IsEverywherePos μ s) {c : ℝ≥0} (hc : c ≠ 0) :
    IsEverywherePos (c • μ) s :=
                      /-
                        α : Type u_1
                        inst✝¹ : TopologicalSpace α
                        inst✝ : MeasurableSpace α
                        μ : MeasureTheory.Measure α
                        s : Set α
                        hs : μ.IsEverywherePos s
                        c : NNReal
                        hc : Ne c 0
                        ⊢ Ne (↑ENNReal.ofNNRealHom.toMonoidWithZeroHom c) 0
                      -/
  hs.smul_measure (by simpa using hc)
                      /-
                        🎉 no goals
                      -/


/-- If two measures coincide locally, then a set which is everywhere positive for the former is
also everywhere positive for the latter. -/
lemma IsEverywherePos.of_forall_exists_nhds_eq (hs : IsEverywherePos μ s)
    (h : ∀ x ∈ s, ∃ t ∈ 𝓝 x, ∀ u ⊆ t, ν u = μ u) : IsEverywherePos ν s := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : μ.IsEverywherePos s
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    ⊢ ν.IsEverywherePos s
  -/
  intro x hx n hn
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : μ.IsEverywherePos s
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    x : α
    hx : Membership.mem s x
    n : Set α
    hn : Membership.mem (nhdsWithin x s) n
    ⊢ LT.lt 0 (ν n)
  -/
  rcases h x hx with ⟨t, t_mem, ht⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : μ.IsEverywherePos s
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    x : α
    hx : Membership.mem s x
    n : Set α
    hn : Membership.mem (nhdsWithin x s) n
    t : Set α
    t_mem : Membership.mem (nhds x) t
    ht : ∀ (u : Set α), HasSubset.Subset u t → Eq (ν u) (μ u)
    ⊢ LT.lt 0 (ν n)
  -/
  refine lt_of_lt_of_le ?_ (measure_mono (inter_subset_left (t := t)))
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : μ.IsEverywherePos s
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    x : α
    hx : Membership.mem s x
    n : Set α
    hn : Membership.mem (nhdsWithin x s) n
    t : Set α
    t_mem : Membership.mem (nhds x) t
    ht : ∀ (u : Set α), HasSubset.Subset u t → Eq (ν u) (μ u)
    ⊢ LT.lt 0 (ν (Inter.inter n t))
  -/
  rw [ht (n ∩ t) inter_subset_right]
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    hs : μ.IsEverywherePos s
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    x : α
    hx : Membership.mem s x
    n : Set α
    hn : Membership.mem (nhdsWithin x s) n
    t : Set α
    t_mem : Membership.mem (nhds x) t
    ht : ∀ (u : Set α), HasSubset.Subset u t → Eq (ν u) (μ u)
    ⊢ LT.lt 0 (μ (Inter.inter n t))
  -/
  exact hs x hx _ (inter_mem hn (mem_nhdsWithin_of_mem_nhds t_mem))
  /-
    🎉 no goals
  -/


/-- If two measures coincide locally, then a set is everywhere positive for the former iff it is
everywhere positive for the latter. -/
lemma isEverywherePos_iff_of_forall_exists_nhds_eq (h : ∀ x ∈ s, ∃ t ∈ 𝓝 x, ∀ u ⊆ t, ν u = μ u) :
    IsEverywherePos ν s ↔ IsEverywherePos μ s := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    ⊢ Iff (ν.IsEverywherePos s) (μ.IsEverywherePos s)
  -/
  refine ⟨fun H ↦ H.of_forall_exists_nhds_eq ?_, fun H ↦ H.of_forall_exists_nhds_eq h⟩
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    H : ν.IsEverywherePos s
    ⊢ ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds x) …
  -/
  intro x hx
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    H : ν.IsEverywherePos s
    x : α
    hx : Membership.mem s x
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (∀ (u : Set α), HasSubset.Su …
  -/
  rcases h x hx with ⟨t, ht, h't⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Exists fun t => And (Membership.mem (nhds  …
    H : ν.IsEverywherePos s
    x : α
    hx : Membership.mem s x
    t : Set α
    ht : Membership.mem (nhds x) t
    h't : ∀ (u : Set α), HasSubset.Subset u t → Eq (ν u) (μ u)
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (∀ (u : Set α), HasSubset.Su …
  -/
  exact ⟨t, ht, fun u hu ↦ (h't u hu).symm⟩
  /-
    🎉 no goals
  -/


/-- An open set is everywhere positive for a measure which is positive on open sets. -/
lemma _root_.IsOpen.isEverywherePos [IsOpenPosMeasure μ] (hs : IsOpen s) : IsEverywherePos μ s := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : μ.IsOpenPosMeasure
    hs : IsOpen s
    ⊢ μ.IsEverywherePos s
  -/
  intro x xs n hn
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : μ.IsOpenPosMeasure
    hs : IsOpen s
    x : α
    xs : Membership.mem s x
    n : Set α
    hn : Membership.mem (nhdsWithin x s) n
    ⊢ LT.lt 0 (μ n)
  -/
  rcases mem_nhdsWithin.1 hn with ⟨u, u_open, xu, hu⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : μ.IsOpenPosMeasure
    hs : IsOpen s
    x : α
    xs : Membership.mem s x
    n : Set α
    hn : Membership.mem (nhdsWithin x s) n
    u : Set α
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : HasSubset.Subset (Inter.inter u s) n
    ⊢ LT.lt 0 (μ n)
  -/
  apply lt_of_lt_of_le _ (measure_mono hu)
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : μ.IsOpenPosMeasure
    hs : IsOpen s
    x : α
    xs : Membership.mem s x
    n : Set α
    hn : Membership.mem (nhdsWithin x s) n
    u : Set α
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : HasSubset.Subset (Inter.inter u s) n
    ⊢ LT.lt 0 (μ (Inter.inter u s))
  -/
  exact (u_open.inter hs).measure_pos μ ⟨x, ⟨xu, xs⟩⟩
  /-
    🎉 no goals
  -/


/-- If a compact closed set is everywhere positive with respect to a left-invariant measure on a
topological group, then it is a Gδ set. This is nontrivial, as there is no second-countability or
metrizability assumption in the statement, so a general compact closed set has no reason to be
a countable intersection of open sets. -/
@[to_additive]
lemma IsEverywherePos.IsGdelta_of_isMulLeftInvariant
    {k : Set G} (h : μ.IsEverywherePos k) (hk : IsCompact k) (h'k : IsClosed k) :
    IsGδ k := by
  /- Consider a decreasing sequence of open neighborhoods `Vₙ` of the identity, such that `g k \ k`
  has small measure for all `g ∈ Vₙ`. We claim that `k = ⋂ Vₙ k`, which proves
  the lemma as the sets on the right are open. The inclusion `⊆` is trivial.
  Let us show the converse. Take `x` in the intersection. For each `n`, write `x = vₙ yₙ` with
  `vₙ ∈ Vₙ` and `yₙ ∈ k`. Let `z ∈ k` be a cluster value of `yₙ`, by compactness. As multiplication
  by `vₙ = x yₙ⁻¹ ∈ Vₙ` changes the measure of `k` by very little, passing to the limit we get
  `μ (x z⁻¹ k \ k) = 0`. By invariance of the measure under `z x ⁻¹`, we get `μ (k \ z x⁻¹ k) = 0`.
  Assume `x ∉ k`. Then `z ∈ k \ z x⁻¹ k`. Even more, this set is a neighborhood of `z` within `k`
  (as `z x⁻¹ k` is closed), and it has zero measure. This contradicts the fact that `k` has
  positive measure around the point `z`. -/
  obtain ⟨u, -, u_mem, u_lim⟩ : ∃ u, StrictAnti u ∧ (∀ (n : ℕ), u n ∈ Ioo 0 1)
    ∧ Tendsto u atTop (𝓝 0) := exists_seq_strictAnti_tendsto' (zero_lt_one : (0 : ℝ≥0∞) < 1)
  have : ∀ n, ∃ (W : Set G), IsOpen W ∧ 1 ∈ W ∧ ∀ g ∈ W * W, μ ((g • k) \ k) < u n :=
    fun n ↦ exists_open_nhds_one_mul_subset
      (eventually_nhds_one_measure_smul_diff_lt hk h'k (u_mem n).1.ne')
  /-
    case intro.intro.intro
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    k : Set G
    h : μ.IsEverywherePos k
    hk : IsCompact k
    h'k : IsClosed k
    u : Nat → ENNReal
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    this : ∀ (n : Nat), Exists fun W => And (IsOpen W) (And (Membership.mem W 1) ( …
    ⊢ IsGδ k
  -/
  choose W W_open mem_W hW using this
  /-
    case intro.intro.intro
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    k : Set G
    h : μ.IsEverywherePos k
    hk : IsCompact k
    h'k : IsClosed k
    u : Nat → ENNReal
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    W : Nat → Set G
    W_open : ∀ (n : Nat), IsOpen (W n)
    mem_W : ∀ (n : Nat), Membership.mem (W n) 1
    hW : ∀ (n : Nat) (g : G), Membership.mem (HMul.hMul (W n) (W n)) g → LT.lt (μ  …
    ⊢ IsGδ k
  -/
  let V n := ⋂ i ∈ Finset.range n, W i
  suffices ⋂ n, V n * k ⊆ k by
    replace : k = ⋂ n, V n * k := by
      apply Subset.antisymm (subset_iInter_iff.2 (fun n ↦ ?_)) this
      exact subset_mul_right k (by simp [V, mem_W])
    rw [this]
    refine .iInter_of_isOpen fun n ↦ ?_
    exact .mul_right (isOpen_biInter_finset (fun i _hi ↦ W_open i))
  /-
    case intro.intro.intro
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    k : Set G
    h : μ.IsEverywherePos k
    hk : IsCompact k
    h'k : IsClosed k
    u : Nat → ENNReal
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    W : Nat → Set G
    W_open : ∀ (n : Nat), IsOpen (W n)
    mem_W : ∀ (n : Nat), Membership.mem (W n) 1
    hW : ∀ (n : Nat) (g : G), Membership.mem (HMul.hMul (W n) (W n)) g → LT.lt (μ  …
    V : Nat → Set G := fun n => Set.iInter fun i => Set.iInter fun h => W i
    ⊢ HasSubset.Subset (Set.iInter fun n => HMul.hMul (V n) k) k
  -/
  intro x hx
  /-
    case intro.intro.intro
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    k : Set G
    h : μ.IsEverywherePos k
    hk : IsCompact k
    h'k : IsClosed k
    u : Nat → ENNReal
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    W : Nat → Set G
    W_open : ∀ (n : Nat), IsOpen (W n)
    mem_W : ∀ (n : Nat), Membership.mem (W n) 1
    hW : ∀ (n : Nat) (g : G), Membership.mem (HMul.hMul (W n) (W n)) g → LT.lt (μ  …
    V : Nat → Set G := fun n => Set.iInter fun i => Set.iInter fun h => W i
    x : G
    hx : Membership.mem (Set.iInter fun n => HMul.hMul (V n) k) x
    ⊢ Membership.mem k x
  -/
  choose v hv y hy hvy using mem_iInter.1 hx
  /-
    case intro.intro.intro
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    k : Set G
    h : μ.IsEverywherePos k
    hk : IsCompact k
    h'k : IsClosed k
    u : Nat → ENNReal
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    W : Nat → Set G
    W_open : ∀ (n : Nat), IsOpen (W n)
    mem_W : ∀ (n : Nat), Membership.mem (W n) 1
    hW : ∀ (n : Nat) (g : G), Membership.mem (HMul.hMul (W n) (W n)) g → LT.lt (μ  …
    V : Nat → Set G := fun n => Set.iInter fun i => Set.iInter fun h => W i
    x : G
    hx : Membership.mem (Set.iInter fun n => HMul.hMul (V n) k) x
    v : Nat → G
    hv : ∀ (i : Nat), Membership.mem (V i) (v i)
    y : Nat → G
    hy : ∀ (i : Nat), Membership.mem k (y i)
    hvy : ∀ (i : Nat), Eq ((fun x1 x2 => HMul.hMul x1 x2) (v i) (y i)) x
    ⊢ Membership.mem k x
  -/
  obtain ⟨z, zk, hz⟩ : ∃ z ∈ k, MapClusterPt z atTop y := hk.exists_mapClusterPt (by simp [hy])
  have A n : μ (((x * z ⁻¹) • k) \ k) ≤ u n := by
    apply le_of_lt (hW _ _ ?_)
    have : W n * {z} ∈ 𝓝 z := (IsOpen.mul_right (W_open n)).mem_nhds (by simp [mem_W])
    obtain ⟨i, hi, ni⟩ : ∃ i, y i ∈ W n * {z} ∧ n < i :=
      ((mapClusterPt_iff.1 hz _ this).and_eventually (eventually_gt_atTop n)).exists
    refine ⟨x * (y i) ⁻¹, ?_, y i * z⁻¹, by simpa using hi, by group⟩
    have I : V i ⊆ W n := iInter₂_subset n (by simp [ni])
    have J : x * (y i) ⁻¹ ∈ V i := by simpa [← hvy i] using hv i
    exact I J
  have B : μ (((x * z ⁻¹) • k) \ k) = 0 :=
    le_antisymm (ge_of_tendsto u_lim (Eventually.of_forall A)) bot_le
  have C : μ (k \ (z * x⁻¹) • k) = 0 := by
    have : μ ((z * x⁻¹) • (((x * z ⁻¹) • k) \ k)) = 0 := by rwa [measure_smul]
    rw [← this, smul_set_sdiff, smul_smul]
    group
    simp
  /-
    case intro.intro.intro.intro.intro
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    k : Set G
    h : μ.IsEverywherePos k
    hk : IsCompact k
    h'k : IsClosed k
    u : Nat → ENNReal
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    W : Nat → Set G
    W_open : ∀ (n : Nat), IsOpen (W n)
    mem_W : ∀ (n : Nat), Membership.mem (W n) 1
    hW : ∀ (n : Nat) (g : G), Membership.mem (HMul.hMul (W n) (W n)) g → LT.lt (μ  …
    V : Nat → Set G := fun n => Set.iInter fun i => Set.iInter fun h => W i
    x : G
    hx : Membership.mem (Set.iInter fun n => HMul.hMul (V n) k) x
    v : Nat → G
    hv : ∀ (i : Nat), Membership.mem (V i) (v i)
    y : Nat → G
    hy : ∀ (i : Nat), Membership.mem k (y i)
    hvy : ∀ (i : Nat), Eq ((fun x1 x2 => HMul.hMul x1 x2) (v i) (y i)) x
    z : G
    zk : Membership.mem k z
    hz : MapClusterPt z Filter.atTop y
    A : ∀ (n : Nat), LE.le (μ (SDiff.sdiff (HSMul.hSMul (HMul.hMul x (Inv.inv z))  …
    B : Eq (μ (SDiff.sdiff (HSMul.hSMul (HMul.hMul x (Inv.inv z)) k) k)) 0
    C : Eq (μ (SDiff.sdiff k (HSMul.hSMul (HMul.hMul z (Inv.inv x)) k))) 0
    ⊢ Membership.mem k x
  -/
  by_contra H
  have : k ∩ ((z * x⁻¹) • k)ᶜ ∈ 𝓝[k] z := by
    apply inter_mem_nhdsWithin k
    apply IsOpen.mem_nhds (by simpa using h'k.smul _)
    simp only [mem_compl_iff]
    contrapose! H
    simpa [mem_smul_set_iff_inv_smul_mem] using H
  /-
    case intro.intro.intro.intro.intro
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    k : Set G
    h : μ.IsEverywherePos k
    hk : IsCompact k
    h'k : IsClosed k
    u : Nat → ENNReal
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    W : Nat → Set G
    W_open : ∀ (n : Nat), IsOpen (W n)
    mem_W : ∀ (n : Nat), Membership.mem (W n) 1
    hW : ∀ (n : Nat) (g : G), Membership.mem (HMul.hMul (W n) (W n)) g → LT.lt (μ  …
    V : Nat → Set G := fun n => Set.iInter fun i => Set.iInter fun h => W i
    x : G
    hx : Membership.mem (Set.iInter fun n => HMul.hMul (V n) k) x
    v : Nat → G
    hv : ∀ (i : Nat), Membership.mem (V i) (v i)
    y : Nat → G
    hy : ∀ (i : Nat), Membership.mem k (y i)
    hvy : ∀ (i : Nat), Eq ((fun x1 x2 => HMul.hMul x1 x2) (v i) (y i)) x
    z : G
    zk : Membership.mem k z
    hz : MapClusterPt z Filter.atTop y
    A : ∀ (n : Nat), LE.le (μ (SDiff.sdiff (HSMul.hSMul (HMul.hMul x (Inv.inv z))  …
    B : Eq (μ (SDiff.sdiff (HSMul.hSMul (HMul.hMul x (Inv.inv z)) k) k)) 0
    C : Eq (μ (SDiff.sdiff k (HSMul.hSMul (HMul.hMul z (Inv.inv x)) k))) 0
    H : Not (Membership.mem k x)
    this : Membership.mem (nhdsWithin z k) (Inter.inter k (HasCompl.compl (HSMul.h …
    ⊢ False
  -/
  have : 0 < μ (k \ ((z * x⁻¹) • k)) := h z zk _ this
  /-
    case intro.intro.intro.intro.intro
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    k : Set G
    h : μ.IsEverywherePos k
    hk : IsCompact k
    h'k : IsClosed k
    u : Nat → ENNReal
    u_mem : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    W : Nat → Set G
    W_open : ∀ (n : Nat), IsOpen (W n)
    mem_W : ∀ (n : Nat), Membership.mem (W n) 1
    hW : ∀ (n : Nat) (g : G), Membership.mem (HMul.hMul (W n) (W n)) g → LT.lt (μ  …
    V : Nat → Set G := fun n => Set.iInter fun i => Set.iInter fun h => W i
    x : G
    hx : Membership.mem (Set.iInter fun n => HMul.hMul (V n) k) x
    v : Nat → G
    hv : ∀ (i : Nat), Membership.mem (V i) (v i)
    y : Nat → G
    hy : ∀ (i : Nat), Membership.mem k (y i)
    hvy : ∀ (i : Nat), Eq ((fun x1 x2 => HMul.hMul x1 x2) (v i) (y i)) x
    z : G
    zk : Membership.mem k z
    hz : MapClusterPt z Filter.atTop y
    A : ∀ (n : Nat), LE.le (μ (SDiff.sdiff (HSMul.hSMul (HMul.hMul x (Inv.inv z))  …
    B : Eq (μ (SDiff.sdiff (HSMul.hSMul (HMul.hMul x (Inv.inv z)) k) k)) 0
    C : Eq (μ (SDiff.sdiff k (HSMul.hSMul (HMul.hMul z (Inv.inv x)) k))) 0
    H : Not (Membership.mem k x)
    this✝ : Membership.mem (nhdsWithin z k) (Inter.inter k (HasCompl.compl (HSMul. …
    this : LT.lt 0 (μ (SDiff.sdiff k (HSMul.hSMul (HMul.hMul z (Inv.inv x)) k)))
    ⊢ False
  -/
  exact lt_irrefl _ (C.le.trans_lt this)
  /-
    🎉 no goals
  -/


/-- **Halmos' theorem: Haar measure is completion regular.** More precisely, any finite measure
set can be approximated from inside by a level set of a continuous function with compact support. -/
@[to_additive innerRegularWRT_preimage_one_hasCompactSupport_measure_ne_top_of_addGroup]
theorem innerRegularWRT_preimage_one_hasCompactSupport_measure_ne_top_of_group :
    InnerRegularWRT μ (fun s ↦ ∃ (f : G → ℝ), Continuous f ∧ HasCompactSupport f ∧ s = f ⁻¹' {1})
    (fun s ↦ MeasurableSet s ∧ μ s ≠ ∞) := by
  /- First, approximate a measurable set from inside by a compact closed set `K`. Then notice that
  the everywhere positive subset of `K` is a Gδ,
  by Lemma `IsEverywherePos.IsGdelta_of_isMulLeftInvariant`, and therefore the level set of a
  continuous compactly supported function. Moreover, it has the same measure as `K`. -/
  /-
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    ⊢ μ.InnerRegularWRT (fun s => Exists fun f => And (Continuous f) (And (HasComp …
  -/
  apply InnerRegularWRT.trans _ innerRegularWRT_isCompact_isClosed_measure_ne_top_of_group
  /-
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    ⊢ μ.InnerRegularWRT (fun s => Exists fun f => And (Continuous f) (And (HasComp …
  -/
  intro K ⟨K_comp, K_closed⟩ r hr
  /-
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    K : Set G
    K_comp : IsCompact K
    K_closed : IsClosed K
    r : ENNReal
    hr : LT.lt r (μ K)
    ⊢ Exists fun K_1 => And (HasSubset.Subset K_1 K) (And ((fun s => Exists fun f  …
  -/
  let L := μ.everywherePosSubset K
  /-
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    K : Set G
    K_comp : IsCompact K
    K_closed : IsClosed K
    r : ENNReal
    hr : LT.lt r (μ K)
    L : Set G := μ.everywherePosSubset K
    ⊢ Exists fun K_1 => And (HasSubset.Subset K_1 K) (And ((fun s => Exists fun f  …
  -/
  have L_comp : IsCompact L := K_comp.everywherePosSubset
  /-
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    K : Set G
    K_comp : IsCompact K
    K_closed : IsClosed K
    r : ENNReal
    hr : LT.lt r (μ K)
    L : Set G := μ.everywherePosSubset K
    L_comp : IsCompact L
    ⊢ Exists fun K_1 => And (HasSubset.Subset K_1 K) (And ((fun s => Exists fun f  …
  -/
  have L_closed : IsClosed L := K_closed.everywherePosSubset
  /-
    G : Type u_2
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : LocallyCompactSpace G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : μ.IsMulLeftInvariant
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : μ.InnerRegularCompactLTTop
    K : Set G
    K_comp : IsCompact K
    K_closed : IsClosed K
    r : ENNReal
    hr : LT.lt r (μ K)
    L : Set G := μ.everywherePosSubset K
    L_comp : IsCompact L
    L_closed : IsClosed L
    ⊢ Exists fun K_1 => And (HasSubset.Subset K_1 K) (And ((fun s => Exists fun f  …
  -/
  refine ⟨L, everywherePosSubset_subset μ K, ?_, ?_⟩
  · have : μ.IsEverywherePos L :=
      isEverywherePos_everywherePosSubset_of_measure_ne_top K_closed.measurableSet
      K_comp.measure_lt_top.ne
    /-
      case refine_1
      G : Type u_2
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : TopologicalGroup G
      inst✝⁵ : LocallyCompactSpace G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      μ : MeasureTheory.Measure G
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝ : μ.InnerRegularCompactLTTop
      K : Set G
      K_comp : IsCompact K
      K_closed : IsClosed K
      r : ENNReal
      hr : LT.lt r (μ K)
      L : Set G := μ.everywherePosSubset K
      L_comp : IsCompact L
      L_closed : IsClosed L
      this : μ.IsEverywherePos L
      ⊢ (fun s => Exists fun f => And (Continuous f) (And (HasCompactSupport f) (Eq  …
    -/
    have L_Gδ : IsGδ L := this.IsGdelta_of_isMulLeftInvariant L_comp L_closed
    obtain ⟨⟨f, f_cont⟩, Lf, -, f_comp, -⟩ : ∃ f : C(G, ℝ), L = f ⁻¹' {1} ∧ EqOn f 0 ∅
        ∧ HasCompactSupport f ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 :=
      exists_continuous_one_zero_of_isCompact_of_isGδ L_comp L_Gδ isClosed_empty
        (disjoint_empty L)
    /-
      case refine_1.intro.mk.intro.intro.intro
      G : Type u_2
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : TopologicalGroup G
      inst✝⁵ : LocallyCompactSpace G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      μ : MeasureTheory.Measure G
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝ : μ.InnerRegularCompactLTTop
      K : Set G
      K_comp : IsCompact K
      K_closed : IsClosed K
      r : ENNReal
      hr : LT.lt r (μ K)
      L : Set G := μ.everywherePosSubset K
      L_comp : IsCompact L
      L_closed : IsClosed L
      this : μ.IsEverywherePos L
      L_Gδ : IsGδ L
      f : G → Real
      f_cont : Continuous f
      Lf : Eq L (Set.preimage (⇑{ toFun := f, continuous_toFun := f_cont }) (Singlet …
      f_comp : HasCompactSupport ⇑{ toFun := f, continuous_toFun := f_cont }
      ⊢ (fun s => Exists fun f => And (Continuous f) (And (HasCompactSupport f) (Eq  …
    -/
    exact ⟨f, f_cont, f_comp, Lf⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_2
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : TopologicalGroup G
      inst✝⁵ : LocallyCompactSpace G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      μ : MeasureTheory.Measure G
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝ : μ.InnerRegularCompactLTTop
      K : Set G
      K_comp : IsCompact K
      K_closed : IsClosed K
      r : ENNReal
      hr : LT.lt r (μ K)
      L : Set G := μ.everywherePosSubset K
      L_comp : IsCompact L
      L_closed : IsClosed L
      ⊢ LT.lt r (μ L)
    -/
  · convert hr using 1
    /-
      case h.e'_4
      G : Type u_2
      inst✝⁸ : Group G
      inst✝⁷ : TopologicalSpace G
      inst✝⁶ : TopologicalGroup G
      inst✝⁵ : LocallyCompactSpace G
      inst✝⁴ : MeasurableSpace G
      inst✝³ : BorelSpace G
      μ : MeasureTheory.Measure G
      inst✝² : μ.IsMulLeftInvariant
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝ : μ.InnerRegularCompactLTTop
      K : Set G
      K_comp : IsCompact K
      K_closed : IsClosed K
      r : ENNReal
      hr : LT.lt r (μ K)
      L : Set G := μ.everywherePosSubset K
      L_comp : IsCompact L
      L_closed : IsClosed L
      ⊢ Eq (μ L) (μ K)
    -/
    apply measure_congr
    exact everywherePosSubset_ae_eq_of_measure_ne_top K_closed.measurableSet
      K_comp.measure_lt_top.ne


