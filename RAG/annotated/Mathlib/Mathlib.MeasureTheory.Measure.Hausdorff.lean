/-- We say that an outer measure `μ` in an (e)metric space is *metric* if `μ (s ∪ t) = μ s + μ t`
for any two metric separated sets `s`, `t`. -/
def IsMetric (μ : OuterMeasure X) : Prop :=
  ∀ s t : Set X, IsMetricSeparated s t → μ (s ∪ t) = μ s + μ t


/-- A metric outer measure is additive on a finite set of pairwise metric separated sets. -/
theorem finset_iUnion_of_pairwise_separated (hm : IsMetric μ) {I : Finset ι} {s : ι → Set X}
    (hI : ∀ i ∈ I, ∀ j ∈ I, i ≠ j → IsMetricSeparated (s i) (s j)) :
    μ (⋃ i ∈ I, s i) = ∑ i ∈ I, μ (s i) := by
  classical
  induction' I using Finset.induction_on with i I hiI ihI hI
  · simp
  simp only [Finset.mem_insert] at hI
  rw [Finset.set_biUnion_insert, hm, ihI, Finset.sum_insert hiI]
  exacts [fun i hi j hj hij => hI i (Or.inr hi) j (Or.inr hj) hij,
    IsMetricSeparated.finset_iUnion_right fun j hj =>
      hI i (Or.inl rfl) j (Or.inr hj) (ne_of_mem_of_not_mem hj hiI).symm]


/-- Caratheodory theorem. If `m` is a metric outer measure, then every Borel measurable set `t` is
Caratheodory measurable: for any (not necessarily measurable) set `s` we have
`μ (s ∩ t) + μ (s \ t) = μ s`. -/
theorem borel_le_caratheodory (hm : IsMetric μ) : borel X ≤ μ.caratheodory := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    μ : MeasureTheory.OuterMeasure X
    hm : μ.IsMetric
    ⊢ LE.le (borel X) μ.caratheodory
  -/
  rw [borel_eq_generateFrom_isClosed]
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    μ : MeasureTheory.OuterMeasure X
    hm : μ.IsMetric
    ⊢ LE.le (MeasurableSpace.generateFrom (setOf fun s => IsClosed s)) μ.caratheod …
  -/
  refine MeasurableSpace.generateFrom_le fun t ht => μ.isCaratheodory_iff_le.2 fun s => ?_
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    μ : MeasureTheory.OuterMeasure X
    hm : μ.IsMetric
    t : Set X
    ht : Membership.mem (setOf fun s => IsClosed s) t
    s : Set X
    ⊢ LE.le (HAdd.hAdd (μ (Inter.inter s t)) (μ (SDiff.sdiff s t))) (μ s)
  -/
  set S : ℕ → Set X := fun n => {x ∈ s | (↑n)⁻¹ ≤ infEdist x t}
  have Ssep (n) : IsMetricSeparated (S n) t :=
    ⟨n⁻¹, ENNReal.inv_ne_zero.2 (ENNReal.natCast_ne_top _),
      fun x hx y hy ↦ hx.2.trans <| infEdist_le_edist_of_mem hy⟩
  have Ssep' : ∀ n, IsMetricSeparated (S n) (s ∩ t) := fun n =>
    (Ssep n).mono Subset.rfl inter_subset_right
  have S_sub : ∀ n, S n ⊆ s \ t := fun n =>
    subset_inter inter_subset_left (Ssep n).subset_compl_right
  have hSs : ∀ n, μ (s ∩ t) + μ (S n) ≤ μ s := fun n =>
    calc
      μ (s ∩ t) + μ (S n) = μ (s ∩ t ∪ S n) := Eq.symm <| hm _ _ <| (Ssep' n).symm
      _ ≤ μ (s ∩ t ∪ s \ t) := μ.mono <| union_subset_union_right _ <| S_sub n
      _ = μ s := by rw [inter_union_diff]
  have iUnion_S : ⋃ n, S n = s \ t := by
    refine Subset.antisymm (iUnion_subset S_sub) ?_
    rintro x ⟨hxs, hxt⟩
    rw [mem_iff_infEdist_zero_of_closed ht] at hxt
    rcases ENNReal.exists_inv_nat_lt hxt with ⟨n, hn⟩
    exact mem_iUnion.2 ⟨n, hxs, hn.le⟩
  /- Now we have `∀ n, μ (s ∩ t) + μ (S n) ≤ μ s` and we need to prove
    `μ (s ∩ t) + μ (⋃ n, S n) ≤ μ s`. We can't pass to the limit because
    `μ` is only an outer measure. -/
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    μ : MeasureTheory.OuterMeasure X
    hm : μ.IsMetric
    t : Set X
    ht : Membership.mem (setOf fun s => IsClosed s) t
    s : Set X
    S : Nat → Set X := fun n => setOf fun x => And (Membership.mem s x) (LE.le (In …
    Ssep : ∀ (n : Nat), IsMetricSeparated (S n) t
    Ssep' : ∀ (n : Nat), IsMetricSeparated (S n) (Inter.inter s t)
    S_sub : ∀ (n : Nat), HasSubset.Subset (S n) (SDiff.sdiff s t)
    hSs : ∀ (n : Nat), LE.le (HAdd.hAdd (μ (Inter.inter s t)) (μ (S n))) (μ s)
    iUnion_S : Eq (Set.iUnion fun n => S n) (SDiff.sdiff s t)
    ⊢ LE.le (HAdd.hAdd (μ (Inter.inter s t)) (μ (SDiff.sdiff s t))) (μ s)
  -/
  by_cases htop : μ (s \ t) = ∞
    /-
      case pos
      X : Type u_2
      inst✝ : EMetricSpace X
      μ : MeasureTheory.OuterMeasure X
      hm : μ.IsMetric
      t : Set X
      ht : Membership.mem (setOf fun s => IsClosed s) t
      s : Set X
      S : Nat → Set X := fun n => setOf fun x => And (Membership.mem s x) (LE.le (In …
      Ssep : ∀ (n : Nat), IsMetricSeparated (S n) t
      Ssep' : ∀ (n : Nat), IsMetricSeparated (S n) (Inter.inter s t)
      S_sub : ∀ (n : Nat), HasSubset.Subset (S n) (SDiff.sdiff s t)
      hSs : ∀ (n : Nat), LE.le (HAdd.hAdd (μ (Inter.inter s t)) (μ (S n))) (μ s)
      iUnion_S : Eq (Set.iUnion fun n => S n) (SDiff.sdiff s t)
      htop : Eq (μ (SDiff.sdiff s t)) Top.top
      ⊢ LE.le (HAdd.hAdd (μ (Inter.inter s t)) (μ (SDiff.sdiff s t))) (μ s)
    -/
  · rw [htop, add_top, ← htop]
    /-
      case pos
      X : Type u_2
      inst✝ : EMetricSpace X
      μ : MeasureTheory.OuterMeasure X
      hm : μ.IsMetric
      t : Set X
      ht : Membership.mem (setOf fun s => IsClosed s) t
      s : Set X
      S : Nat → Set X := fun n => setOf fun x => And (Membership.mem s x) (LE.le (In …
      Ssep : ∀ (n : Nat), IsMetricSeparated (S n) t
      Ssep' : ∀ (n : Nat), IsMetricSeparated (S n) (Inter.inter s t)
      S_sub : ∀ (n : Nat), HasSubset.Subset (S n) (SDiff.sdiff s t)
      hSs : ∀ (n : Nat), LE.le (HAdd.hAdd (μ (Inter.inter s t)) (μ (S n))) (μ s)
      iUnion_S : Eq (Set.iUnion fun n => S n) (SDiff.sdiff s t)
      htop : Eq (μ (SDiff.sdiff s t)) Top.top
      ⊢ LE.le (μ (SDiff.sdiff s t)) (μ s)
    -/
    exact μ.mono diff_subset
    /-
      🎉 no goals
    -/
  suffices μ (⋃ n, S n) ≤ ⨆ n, μ (S n) by calc
    μ (s ∩ t) + μ (s \ t) = μ (s ∩ t) + μ (⋃ n, S n) := by rw [iUnion_S]
    _ ≤ μ (s ∩ t) + ⨆ n, μ (S n) := by gcongr
    _ = ⨆ n, μ (s ∩ t) + μ (S n) := ENNReal.add_iSup ..
    _ ≤ μ s := iSup_le hSs
  /- It suffices to show that `∑' k, μ (S (k + 1) \ S k) ≠ ∞`. Indeed, if we have this,
    then for all `N` we have `μ (⋃ n, S n) ≤ μ (S N) + ∑' k, m (S (N + k + 1) \ S (N + k))`
    and the second term tends to zero, see `OuterMeasure.iUnion_nat_of_monotone_of_tsum_ne_top`
    for details. -/
  have : ∀ n, S n ⊆ S (n + 1) := fun n x hx =>
    ⟨hx.1, le_trans (ENNReal.inv_le_inv.2 <| Nat.cast_le.2 n.le_succ) hx.2⟩
  classical -- Porting note: Added this to get the next tactic to work
  refine (μ.iUnion_nat_of_monotone_of_tsum_ne_top this ?_).le; clear this
  /- While the sets `S (k + 1) \ S k` are not pairwise metric separated, the sets in each
    subsequence `S (2 * k + 1) \ S (2 * k)` and `S (2 * k + 2) \ S (2 * k)` are metric separated,
    so `m` is additive on each of those sequences. -/
  rw [← tsum_even_add_odd ENNReal.summable ENNReal.summable, ENNReal.add_ne_top]
  suffices ∀ a, (∑' k : ℕ, μ (S (2 * k + 1 + a) \ S (2 * k + a))) ≠ ∞ from
    ⟨by simpa using this 0, by simpa using this 1⟩
  refine fun r => ne_top_of_le_ne_top htop ?_
  rw [← iUnion_S, ENNReal.tsum_eq_iSup_nat, iSup_le_iff]
  intro n
  rw [← hm.finset_iUnion_of_pairwise_separated]
  · exact μ.mono (iUnion_subset fun i => iUnion_subset fun _ x hx => mem_iUnion.2 ⟨_, hx.1⟩)
  suffices ∀ i j, i < j → IsMetricSeparated (S (2 * i + 1 + r)) (s \ S (2 * j + r)) from
    fun i _ j _ hij => hij.lt_or_lt.elim
      (fun h => (this i j h).mono inter_subset_left fun x hx => by exact ⟨hx.1.1, hx.2⟩)
      fun h => (this j i h).symm.mono (fun x hx => by exact ⟨hx.1.1, hx.2⟩) inter_subset_left
  intro i j hj
  have A : ((↑(2 * j + r))⁻¹ : ℝ≥0∞) < (↑(2 * i + 1 + r))⁻¹ := by
    rw [ENNReal.inv_lt_inv, Nat.cast_lt]; omega
  refine ⟨(↑(2 * i + 1 + r))⁻¹ - (↑(2 * j + r))⁻¹, by simpa [tsub_eq_zero_iff_le] using A,
    fun x hx y hy => ?_⟩
  have : infEdist y t < (↑(2 * j + r))⁻¹ := not_le.1 fun hle => hy.2 ⟨hy.1, hle⟩
  rcases infEdist_lt_iff.mp this with ⟨z, hzt, hyz⟩
  have hxz : (↑(2 * i + 1 + r))⁻¹ ≤ edist x z := le_infEdist.1 hx.2 _ hzt
  apply ENNReal.le_of_add_le_add_right hyz.ne_top
  refine le_trans ?_ (edist_triangle _ _ _)
  refine (add_le_add le_rfl hyz.le).trans (Eq.trans_le ?_ hxz)
  rw [tsub_add_cancel_of_le A.le]


theorem le_caratheodory [MeasurableSpace X] [BorelSpace X] (hm : IsMetric μ) :
    ‹MeasurableSpace X› ≤ μ.caratheodory := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    μ : MeasureTheory.OuterMeasure X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    hm : μ.IsMetric
    ⊢ LE.le inst✝¹ μ.caratheodory
  -/
  rw [BorelSpace.measurable_eq (α := X)]
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    μ : MeasureTheory.OuterMeasure X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    hm : μ.IsMetric
    ⊢ LE.le (borel X) μ.caratheodory
  -/
  exact hm.borel_le_caratheodory
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `OuterMeasure.mkMetric'`: given a function on sets
`m : Set X → ℝ≥0∞`, returns the maximal outer measure `μ` such that `μ s ≤ m s`
for any set `s` of diameter at most `r`. -/
def mkMetric'.pre (m : Set X → ℝ≥0∞) (r : ℝ≥0∞) : OuterMeasure X :=
  boundedBy <| extend fun s (_ : diam s ≤ r) => m s


/-- Given a function `m : Set X → ℝ≥0∞`, `mkMetric' m` is the supremum of `mkMetric'.pre m r`
over `r > 0`. Equivalently, it is the limit of `mkMetric'.pre m r` as `r` tends to zero from
the right. -/
def mkMetric' (m : Set X → ℝ≥0∞) : OuterMeasure X :=
  ⨆ r > 0, mkMetric'.pre m r


/-- Given a function `m : ℝ≥0∞ → ℝ≥0∞` and `r > 0`, let `μ r` be the maximal outer measure such that
`μ s ≤ m (EMetric.diam s)` whenever `EMetric.diam s < r`. Then `mkMetric m = ⨆ r > 0, μ r`. -/
def mkMetric (m : ℝ≥0∞ → ℝ≥0∞) : OuterMeasure X :=
  mkMetric' fun s => m (diam s)


theorem le_pre : μ ≤ pre m r ↔ ∀ s : Set X, diam s ≤ r → μ s ≤ m s := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    r : ENNReal
    μ : MeasureTheory.OuterMeasure X
    ⊢ Iff (LE.le μ (MeasureTheory.OuterMeasure.mkMetric'.pre m r)) (∀ (s : Set X), …
  -/
  simp only [pre, le_boundedBy, extend, le_iInf_iff]
  /-
    🎉 no goals
  -/


theorem pre_le (hs : diam s ≤ r) : pre m r s ≤ m s :=
  (boundedBy_le _).trans <| iInf_le _ hs


theorem mono_pre (m : Set X → ℝ≥0∞) {r r' : ℝ≥0∞} (h : r ≤ r') : pre m r' ≤ pre m r :=
  le_pre.2 fun _ hs => pre_le (hs.trans h)


theorem mono_pre_nat (m : Set X → ℝ≥0∞) : Monotone fun k : ℕ => pre m k⁻¹ :=
                                                           /-
                                                             X : Type u_2
                                                             inst✝ : EMetricSpace X
                                                             m : Set X → ENNReal
                                                             k l : Nat
                                                             h : LE.le k l
                                                             x✝ : Set X
                                                             hs : LE.le (EMetric.diam x✝) (Inv.inv ↑l)
                                                             ⊢ LE.le (Inv.inv ↑l) (Inv.inv ↑k)
                                                           -/
  fun k l h => le_pre.2 fun _ hs => pre_le (hs.trans <| by simpa)
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem tendsto_pre (m : Set X → ℝ≥0∞) (s : Set X) :
    Tendsto (fun r => pre m r s) (𝓝[>] 0) (𝓝 <| mkMetric' m s) := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s : Set X
    ⊢ Filter.Tendsto (fun r => (MeasureTheory.OuterMeasure.mkMetric'.pre m r) s) ( …
  -/
  rw [← map_coe_Ioi_atBot, tendsto_map'_iff]
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s : Set X
    ⊢ Filter.Tendsto (Function.comp (fun r => (MeasureTheory.OuterMeasure.mkMetric …
  -/
  simp only [mkMetric', OuterMeasure.iSup_apply, iSup_subtype']
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s : Set X
    ⊢ Filter.Tendsto (Function.comp (fun r => (MeasureTheory.OuterMeasure.mkMetric …
  -/
  exact tendsto_atBot_iSup fun r r' hr => mono_pre _ hr _
  /-
    🎉 no goals
  -/


theorem tendsto_pre_nat (m : Set X → ℝ≥0∞) (s : Set X) :
    Tendsto (fun n : ℕ => pre m n⁻¹ s) atTop (𝓝 <| mkMetric' m s) := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s : Set X
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.OuterMeasure.mkMetric'.pre m (Inv.in …
  -/
  refine (tendsto_pre m s).comp (tendsto_inf.2 ⟨ENNReal.tendsto_inv_nat_nhds_zero, ?_⟩)
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s : Set X
    ⊢ Filter.Tendsto (fun n => Inv.inv ↑n) Filter.atTop (Filter.principal (Set.Ioi …
  -/
  refine tendsto_principal.2 (Eventually.of_forall fun n => ?_)
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s : Set X
    n : Nat
    ⊢ Membership.mem (Set.Ioi 0) (Inv.inv ↑n)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem eq_iSup_nat (m : Set X → ℝ≥0∞) : mkMetric' m = ⨆ n : ℕ, mkMetric'.pre m n⁻¹ := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    ⊢ Eq (MeasureTheory.OuterMeasure.mkMetric' m) (iSup fun n => MeasureTheory.Out …
  -/
  ext1 s
  /-
    case h
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s : Set X
    ⊢ Eq ((MeasureTheory.OuterMeasure.mkMetric' m) s) ((iSup fun n => MeasureTheor …
  -/
  rw [iSup_apply]
  refine tendsto_nhds_unique (mkMetric'.tendsto_pre_nat m s)
    (tendsto_atTop_iSup fun k l hkl => mkMetric'.mono_pre_nat m hkl s)


/-- `MeasureTheory.OuterMeasure.mkMetric'.pre m r` is a trimmed measure provided that
`m (closure s) = m s` for any set `s`. -/
theorem trim_pre [MeasurableSpace X] [OpensMeasurableSpace X] (m : Set X → ℝ≥0∞)
    (hcl : ∀ s, m (closure s) = m s) (r : ℝ≥0∞) : (pre m r).trim = pre m r := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : OpensMeasurableSpace X
    m : Set X → ENNReal
    hcl : ∀ (s : Set X), Eq (m (closure s)) (m s)
    r : ENNReal
    ⊢ Eq (MeasureTheory.OuterMeasure.mkMetric'.pre m r).trim (MeasureTheory.OuterM …
  -/
  refine le_antisymm (le_pre.2 fun s hs => ?_) (le_trim _)
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : OpensMeasurableSpace X
    m : Set X → ENNReal
    hcl : ∀ (s : Set X), Eq (m (closure s)) (m s)
    r : ENNReal
    s : Set X
    hs : LE.le (EMetric.diam s) r
    ⊢ LE.le ((MeasureTheory.OuterMeasure.mkMetric'.pre m r).trim s) (m s)
  -/
  rw [trim_eq_iInf]
  refine iInf_le_of_le (closure s) <| iInf_le_of_le subset_closure <|
    iInf_le_of_le measurableSet_closure ((pre_le ?_).trans_eq (hcl _))
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : OpensMeasurableSpace X
    m : Set X → ENNReal
    hcl : ∀ (s : Set X), Eq (m (closure s)) (m s)
    r : ENNReal
    s : Set X
    hs : LE.le (EMetric.diam s) r
    ⊢ LE.le (EMetric.diam (closure s)) r
  -/
  rwa [diam_closure]
  /-
    🎉 no goals
  -/


/-- An outer measure constructed using `OuterMeasure.mkMetric'` is a metric outer measure. -/
theorem mkMetric'_isMetric (m : Set X → ℝ≥0∞) : (mkMetric' m).IsMetric := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    ⊢ (MeasureTheory.OuterMeasure.mkMetric' m).IsMetric
  -/
  rintro s t ⟨r, r0, hr⟩
  refine tendsto_nhds_unique_of_eventuallyEq
    (mkMetric'.tendsto_pre _ _) ((mkMetric'.tendsto_pre _ _).add (mkMetric'.tendsto_pre _ _)) ?_
  /-
    case intro.intro
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s t : Set X
    r : ENNReal
    r0 : Ne r 0
    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
    ⊢ (nhdsWithin 0 (Set.Ioi 0)).EventuallyEq (fun r => (MeasureTheory.OuterMeasur …
  -/
  rw [← pos_iff_ne_zero] at r0
  /-
    case intro.intro
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s t : Set X
    r : ENNReal
    r0 : LT.lt 0 r
    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
    ⊢ (nhdsWithin 0 (Set.Ioi 0)).EventuallyEq (fun r => (MeasureTheory.OuterMeasur …
  -/
  filter_upwards [Ioo_mem_nhdsGT r0]
  /-
    case h
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s t : Set X
    r : ENNReal
    r0 : LT.lt 0 r
    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
    ⊢ ∀ (a : ENNReal), Membership.mem (Set.Ioo 0 r) a → Eq ((MeasureTheory.OuterMe …
  -/
  rintro ε ⟨_, εr⟩
  /-
    case h.intro
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s t : Set X
    r : ENNReal
    r0 : LT.lt 0 r
    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
    ε : ENNReal
    left✝ : LT.lt 0 ε
    εr : LT.lt ε r
    ⊢ Eq ((MeasureTheory.OuterMeasure.mkMetric'.pre m ε) (Union.union s t)) (HAdd. …
  -/
  refine boundedBy_union_of_top_of_nonempty_inter ?_
  /-
    case h.intro
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s t : Set X
    r : ENNReal
    r0 : LT.lt 0 r
    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
    ε : ENNReal
    left✝ : LT.lt 0 ε
    εr : LT.lt ε r
    ⊢ ∀ (u : Set X), (Inter.inter s u).Nonempty → (Inter.inter t u).Nonempty → Eq  …
  -/
  rintro u ⟨x, hxs, hxu⟩ ⟨y, hyt, hyu⟩
  /-
    case h.intro.intro.intro.intro.intro
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s t : Set X
    r : ENNReal
    r0 : LT.lt 0 r
    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
    ε : ENNReal
    left✝ : LT.lt 0 ε
    εr : LT.lt ε r
    u : Set X
    x : X
    hxs : Membership.mem s x
    hxu : Membership.mem u x
    y : X
    hyt : Membership.mem t y
    hyu : Membership.mem u y
    ⊢ Eq (MeasureTheory.extend (fun s x => m s) u) Top.top
  -/
  have : ε < diam u := εr.trans_le ((hr x hxs y hyt).trans <| edist_le_diam_of_mem hxu hyu)
  /-
    case h.intro.intro.intro.intro.intro
    X : Type u_2
    inst✝ : EMetricSpace X
    m : Set X → ENNReal
    s t : Set X
    r : ENNReal
    r0 : LT.lt 0 r
    hr : ∀ (x : X), Membership.mem s x → ∀ (y : X), Membership.mem t y → LE.le r ( …
    ε : ENNReal
    left✝ : LT.lt 0 ε
    εr : LT.lt ε r
    u : Set X
    x : X
    hxs : Membership.mem s x
    hxu : Membership.mem u x
    y : X
    hyt : Membership.mem t y
    hyu : Membership.mem u y
    this : LT.lt ε (EMetric.diam u)
    ⊢ Eq (MeasureTheory.extend (fun s x => m s) u) Top.top
  -/
  exact iInf_eq_top.2 fun h => (this.not_le h).elim
  /-
    🎉 no goals
  -/


/-- If `c ∉ {0, ∞}` and `m₁ d ≤ c * m₂ d` for `d < ε` for some `ε > 0`
(we use `≤ᶠ[𝓝[≥] 0]` to state this), then `mkMetric m₁ hm₁ ≤ c • mkMetric m₂ hm₂`. -/
theorem mkMetric_mono_smul {m₁ m₂ : ℝ≥0∞ → ℝ≥0∞} {c : ℝ≥0∞} (hc : c ≠ ∞) (h0 : c ≠ 0)
    (hle : m₁ ≤ᶠ[𝓝[≥] 0] c • m₂) : (mkMetric m₁ : OuterMeasure X) ≤ c • mkMetric m₂ := by
  classical
  rcases (mem_nhdsGE_iff_exists_Ico_subset' zero_lt_one).1 hle with ⟨r, hr0, hr⟩
  refine fun s =>
    le_of_tendsto_of_tendsto (mkMetric'.tendsto_pre _ s)
      (ENNReal.Tendsto.const_mul (mkMetric'.tendsto_pre _ s) (Or.inr hc))
      (mem_of_superset (Ioo_mem_nhdsGT hr0) fun r' hr' => ?_)
  simp only [mem_setOf_eq, mkMetric'.pre, RingHom.id_apply]
  rw [← smul_eq_mul, ← smul_apply, smul_boundedBy hc]
  refine le_boundedBy.2 (fun t => (boundedBy_le _).trans ?_) _
  simp only [smul_eq_mul, Pi.smul_apply, extend, iInf_eq_if]
  split_ifs with ht
  · apply hr
    exact ⟨zero_le _, ht.trans_lt hr'.2⟩
  · simp [h0]


@[simp]
theorem mkMetric_top : (mkMetric (fun _ => ∞ : ℝ≥0∞ → ℝ≥0∞) : OuterMeasure X) = ⊤ := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    ⊢ Eq (MeasureTheory.OuterMeasure.mkMetric fun x => Top.top) Top.top
  -/
  simp_rw [mkMetric, mkMetric', mkMetric'.pre, extend_top, boundedBy_top, eq_top_iff]
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    ⊢ LE.le Top.top (iSup fun r => iSup fun x => Top.top)
  -/
  rw [le_iSup_iff]
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    ⊢ ∀ (b : MeasureTheory.OuterMeasure X), (∀ (i : ENNReal), LE.le (iSup fun x => …
  -/
  intro b hb
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    b : MeasureTheory.OuterMeasure X
    hb : ∀ (i : ENNReal), LE.le (iSup fun x => Top.top) b
    ⊢ LE.le Top.top b
  -/
  simpa using hb ⊤
  /-
    🎉 no goals
  -/


/-- If `m₁ d ≤ m₂ d` for `d < ε` for some `ε > 0` (we use `≤ᶠ[𝓝[≥] 0]` to state this), then
`mkMetric m₁ hm₁ ≤ mkMetric m₂ hm₂`. -/
theorem mkMetric_mono {m₁ m₂ : ℝ≥0∞ → ℝ≥0∞} (hle : m₁ ≤ᶠ[𝓝[≥] 0] m₂) :
    (mkMetric m₁ : OuterMeasure X) ≤ mkMetric m₂ := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m₁ m₂ : ENNReal → ENNReal
    hle : (nhdsWithin 0 (Set.Ici 0)).EventuallyLE m₁ m₂
    ⊢ LE.le (MeasureTheory.OuterMeasure.mkMetric m₁) (MeasureTheory.OuterMeasure.m …
  -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  convert @mkMetric_mono_smul X _ _ m₂ _ ENNReal.one_ne_top one_ne_zero _ <;> simp [*]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem isometry_comap_mkMetric (m : ℝ≥0∞ → ℝ≥0∞) {f : X → Y} (hf : Isometry f)
    (H : Monotone m ∨ Surjective f) : comap f (mkMetric m) = mkMetric m := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    m : ENNReal → ENNReal
    f : X → Y
    hf : Isometry f
    H : Or (Monotone m) (Function.Surjective f)
    ⊢ Eq ((MeasureTheory.OuterMeasure.comap f) (MeasureTheory.OuterMeasure.mkMetri …
  -/
  simp only [mkMetric, mkMetric', mkMetric'.pre, inducedOuterMeasure, comap_iSup]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    m : ENNReal → ENNReal
    f : X → Y
    hf : Isometry f
    H : Or (Monotone m) (Function.Surjective f)
    ⊢ Eq (iSup fun i => iSup fun i_1 => (MeasureTheory.OuterMeasure.comap f) (Meas …
  -/
  refine surjective_id.iSup_congr id fun ε => surjective_id.iSup_congr id fun hε => ?_
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    m : ENNReal → ENNReal
    f : X → Y
    hf : Isometry f
    H : Or (Monotone m) (Function.Surjective f)
    ε : ENNReal
    hε : GT.gt (id ε) 0
    ⊢ Eq ((MeasureTheory.OuterMeasure.comap f) (MeasureTheory.OuterMeasure.bounded …
  -/
  rw [comap_boundedBy _ (H.imp _ id)]
    /-
      X : Type u_2
      Y : Type u_3
      inst✝¹ : EMetricSpace X
      inst✝ : EMetricSpace Y
      m : ENNReal → ENNReal
      f : X → Y
      hf : Isometry f
      H : Or (Monotone m) (Function.Surjective f)
      ε : ENNReal
      hε : GT.gt (id ε) 0
      ⊢ Eq (MeasureTheory.OuterMeasure.boundedBy fun s => MeasureTheory.extend (fun  …
    -/
  · congr with s : 1
    /-
      case e_m.h
      X : Type u_2
      Y : Type u_3
      inst✝¹ : EMetricSpace X
      inst✝ : EMetricSpace Y
      m : ENNReal → ENNReal
      f : X → Y
      hf : Isometry f
      H : Or (Monotone m) (Function.Surjective f)
      ε : ENNReal
      hε : GT.gt (id ε) 0
      s : Set X
      ⊢ Eq (MeasureTheory.extend (fun s x => m (EMetric.diam s)) (Set.image f s)) (M …
    -/
    apply extend_congr
      /-
        case e_m.h.hP
        X : Type u_2
        Y : Type u_3
        inst✝¹ : EMetricSpace X
        inst✝ : EMetricSpace Y
        m : ENNReal → ENNReal
        f : X → Y
        hf : Isometry f
        H : Or (Monotone m) (Function.Surjective f)
        ε : ENNReal
        hε : GT.gt (id ε) 0
        s : Set X
        ⊢ Iff (LE.le (EMetric.diam (Set.image f s)) ε) (LE.le (EMetric.diam s) (id ε))
      -/
    · simp [hf.ediam_image]
      /-
        🎉 no goals
      -/
      /-
        case e_m.h.hm
        X : Type u_2
        Y : Type u_3
        inst✝¹ : EMetricSpace X
        inst✝ : EMetricSpace Y
        m : ENNReal → ENNReal
        f : X → Y
        hf : Isometry f
        H : Or (Monotone m) (Function.Surjective f)
        ε : ENNReal
        hε : GT.gt (id ε) 0
        s : Set X
        ⊢ LE.le (EMetric.diam (Set.image f s)) ε → LE.le (EMetric.diam s) (id ε) → Eq  …
      -/
    · intros; simp [hf.injective.subsingleton_image_iff, hf.ediam_image]
              /-
                🎉 no goals
              -/
    /-
      X : Type u_2
      Y : Type u_3
      inst✝¹ : EMetricSpace X
      inst✝ : EMetricSpace Y
      m : ENNReal → ENNReal
      f : X → Y
      hf : Isometry f
      H : Or (Monotone m) (Function.Surjective f)
      ε : ENNReal
      hε : GT.gt (id ε) 0
      ⊢ Monotone m → Monotone fun s => MeasureTheory.extend (fun s x => m (EMetric.d …
    -/
  · intro h_mono s t hst
    /-
      X : Type u_2
      Y : Type u_3
      inst✝¹ : EMetricSpace X
      inst✝ : EMetricSpace Y
      m : ENNReal → ENNReal
      f : X → Y
      hf : Isometry f
      H : Or (Monotone m) (Function.Surjective f)
      ε : ENNReal
      hε : GT.gt (id ε) 0
      h_mono : Monotone m
      s t : Subtype fun s => s.Nonempty
      hst : LE.le s t
      ⊢ LE.le ((fun s => MeasureTheory.extend (fun s x => m (EMetric.diam s)) ↑s) s) …
    -/
    simp only [extend, le_iInf_iff]
    /-
      X : Type u_2
      Y : Type u_3
      inst✝¹ : EMetricSpace X
      inst✝ : EMetricSpace Y
      m : ENNReal → ENNReal
      f : X → Y
      hf : Isometry f
      H : Or (Monotone m) (Function.Surjective f)
      ε : ENNReal
      hε : GT.gt (id ε) 0
      h_mono : Monotone m
      s t : Subtype fun s => s.Nonempty
      hst : LE.le s t
      ⊢ LE.le (EMetric.diam ↑t) ε → LE.le (iInf fun x => m (EMetric.diam ↑s)) (m (EM …
    -/
    intro ht
    /-
      X : Type u_2
      Y : Type u_3
      inst✝¹ : EMetricSpace X
      inst✝ : EMetricSpace Y
      m : ENNReal → ENNReal
      f : X → Y
      hf : Isometry f
      H : Or (Monotone m) (Function.Surjective f)
      ε : ENNReal
      hε : GT.gt (id ε) 0
      h_mono : Monotone m
      s t : Subtype fun s => s.Nonempty
      hst : LE.le s t
      ht : LE.le (EMetric.diam ↑t) ε
      ⊢ LE.le (iInf fun x => m (EMetric.diam ↑s)) (m (EMetric.diam ↑t))
    -/
    apply le_trans _ (h_mono (diam_mono hst))
    /-
      X : Type u_2
      Y : Type u_3
      inst✝¹ : EMetricSpace X
      inst✝ : EMetricSpace Y
      m : ENNReal → ENNReal
      f : X → Y
      hf : Isometry f
      H : Or (Monotone m) (Function.Surjective f)
      ε : ENNReal
      hε : GT.gt (id ε) 0
      h_mono : Monotone m
      s t : Subtype fun s => s.Nonempty
      hst : LE.le s t
      ht : LE.le (EMetric.diam ↑t) ε
      ⊢ LE.le (iInf fun x => m (EMetric.diam ↑s)) (m (EMetric.diam ((fun a => ↑a) s)))
    -/
    simp only [(diam_mono hst).trans ht, le_refl, ciInf_pos]
    /-
      🎉 no goals
    -/


theorem mkMetric_smul (m : ℝ≥0∞ → ℝ≥0∞) {c : ℝ≥0∞} (hc : c ≠ ∞) (hc' : c ≠ 0) :
    (mkMetric (c • m) : OuterMeasure X) = c • mkMetric m := by
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : ENNReal → ENNReal
    c : ENNReal
    hc : Ne c Top.top
    hc' : Ne c 0
    ⊢ Eq (MeasureTheory.OuterMeasure.mkMetric (HSMul.hSMul c m)) (HSMul.hSMul c (M …
  -/
  simp only [mkMetric, mkMetric', mkMetric'.pre, inducedOuterMeasure, ENNReal.smul_iSup]
  /-
    X : Type u_2
    inst✝ : EMetricSpace X
    m : ENNReal → ENNReal
    c : ENNReal
    hc : Ne c Top.top
    hc' : Ne c 0
    ⊢ Eq (iSup fun r => iSup fun x => MeasureTheory.OuterMeasure.boundedBy (Measur …
  -/
  simp_rw [smul_iSup, smul_boundedBy hc, smul_extend _ hc', Pi.smul_apply]
  /-
    🎉 no goals
  -/


theorem mkMetric_nnreal_smul (m : ℝ≥0∞ → ℝ≥0∞) {c : ℝ≥0} (hc : c ≠ 0) :
    (mkMetric (c • m) : OuterMeasure X) = c • mkMetric m := by
  rw [ENNReal.smul_def, ENNReal.smul_def,
    mkMetric_smul m ENNReal.coe_ne_top (ENNReal.coe_ne_zero.mpr hc)]


theorem isometry_map_mkMetric (m : ℝ≥0∞ → ℝ≥0∞) {f : X → Y} (hf : Isometry f)
    (H : Monotone m ∨ Surjective f) : map f (mkMetric m) = restrict (range f) (mkMetric m) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    m : ENNReal → ENNReal
    f : X → Y
    hf : Isometry f
    H : Or (Monotone m) (Function.Surjective f)
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) (MeasureTheory.OuterMeasure.mkMetric  …
  -/
  rw [← isometry_comap_mkMetric _ hf H, map_comap]
  /-
    🎉 no goals
  -/


theorem isometryEquiv_comap_mkMetric (m : ℝ≥0∞ → ℝ≥0∞) (f : X ≃ᵢ Y) :
    comap f (mkMetric m) = mkMetric m :=
  isometry_comap_mkMetric _ f.isometry (Or.inr f.surjective)


theorem isometryEquiv_map_mkMetric (m : ℝ≥0∞ → ℝ≥0∞) (f : X ≃ᵢ Y) :
    map f (mkMetric m) = mkMetric m := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : EMetricSpace X
    inst✝ : EMetricSpace Y
    m : ENNReal → ENNReal
    f : IsometryEquiv X Y
    ⊢ Eq ((MeasureTheory.OuterMeasure.map ⇑f) (MeasureTheory.OuterMeasure.mkMetric …
  -/
  rw [← isometryEquiv_comap_mkMetric _ f, map_comap_of_surjective f.surjective]
  /-
    🎉 no goals
  -/


theorem trim_mkMetric [MeasurableSpace X] [BorelSpace X] (m : ℝ≥0∞ → ℝ≥0∞) :
    (mkMetric m : OuterMeasure X).trim = mkMetric m := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m : ENNReal → ENNReal
    ⊢ Eq (MeasureTheory.OuterMeasure.mkMetric m).trim (MeasureTheory.OuterMeasure. …
  -/
  simp only [mkMetric, mkMetric'.eq_iSup_nat, trim_iSup]
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m : ENNReal → ENNReal
    ⊢ Eq (iSup fun i => (MeasureTheory.OuterMeasure.mkMetric'.pre (fun s => m (EMe …
  -/
  congr 1 with n : 1
  /-
    case e_s.h
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m : ENNReal → ENNReal
    n : Nat
    ⊢ Eq (MeasureTheory.OuterMeasure.mkMetric'.pre (fun s => m (EMetric.diam s)) ( …
  -/
  refine mkMetric'.trim_pre _ (fun s => ?_) _
  /-
    case e_s.h
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m : ENNReal → ENNReal
    n : Nat
    s : Set X
    ⊢ Eq (m (EMetric.diam (closure s))) (m (EMetric.diam s))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem le_mkMetric (m : ℝ≥0∞ → ℝ≥0∞) (μ : OuterMeasure X) (r : ℝ≥0∞) (h0 : 0 < r)
    (hr : ∀ s, diam s ≤ r → μ s ≤ m (diam s)) : μ ≤ mkMetric m :=
  le_iSup₂_of_le r h0 <| mkMetric'.le_pre.2 fun _ hs => hr _ hs


/-- Given a function `m : Set X → ℝ≥0∞`, `mkMetric' m` is the supremum of `μ r`
over `r > 0`, where `μ r` is the maximal outer measure `μ` such that `μ s ≤ m s`
for all `s`. While each `μ r` is an *outer* measure, the supremum is a measure. -/
def mkMetric' (m : Set X → ℝ≥0∞) : Measure X :=
  (OuterMeasure.mkMetric' m).toMeasure (OuterMeasure.mkMetric'_isMetric _).le_caratheodory


/-- Given a function `m : ℝ≥0∞ → ℝ≥0∞`, `mkMetric m` is the supremum of `μ r` over `r > 0`, where
`μ r` is the maximal outer measure `μ` such that `μ s ≤ m s` for all sets `s` that contain at least
two points. While each `mkMetric'.pre` is an *outer* measure, the supremum is a measure. -/
def mkMetric (m : ℝ≥0∞ → ℝ≥0∞) : Measure X :=
  (OuterMeasure.mkMetric m).toMeasure (OuterMeasure.mkMetric'_isMetric _).le_caratheodory


@[simp]
theorem mkMetric'_toOuterMeasure (m : Set X → ℝ≥0∞) :
    (mkMetric' m).toOuterMeasure = (OuterMeasure.mkMetric' m).trim :=
  rfl


@[simp]
theorem mkMetric_toOuterMeasure (m : ℝ≥0∞ → ℝ≥0∞) :
    (mkMetric m : Measure X).toOuterMeasure = OuterMeasure.mkMetric m :=
  OuterMeasure.trim_mkMetric m


theorem OuterMeasure.coe_mkMetric [MeasurableSpace X] [BorelSpace X] (m : ℝ≥0∞ → ℝ≥0∞) :
    ⇑(OuterMeasure.mkMetric m : OuterMeasure X) = Measure.mkMetric m := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m : ENNReal → ENNReal
    ⊢ Eq ⇑(MeasureTheory.OuterMeasure.mkMetric m) ⇑(MeasureTheory.Measure.mkMetric …
  -/
  rw [← Measure.mkMetric_toOuterMeasure, Measure.coe_toOuterMeasure]
  /-
    🎉 no goals
  -/


/-- If `c ∉ {0, ∞}` and `m₁ d ≤ c * m₂ d` for `d < ε` for some `ε > 0`
(we use `≤ᶠ[𝓝[≥] 0]` to state this), then `mkMetric m₁ hm₁ ≤ c • mkMetric m₂ hm₂`. -/
theorem mkMetric_mono_smul {m₁ m₂ : ℝ≥0∞ → ℝ≥0∞} {c : ℝ≥0∞} (hc : c ≠ ∞) (h0 : c ≠ 0)
    (hle : m₁ ≤ᶠ[𝓝[≥] 0] c • m₂) : (mkMetric m₁ : Measure X) ≤ c • mkMetric m₂ := fun s ↦ by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m₁ m₂ : ENNReal → ENNReal
    c : ENNReal
    hc : Ne c Top.top
    h0 : Ne c 0
    hle : (nhdsWithin 0 (Set.Ici 0)).EventuallyLE m₁ (HSMul.hSMul c m₂)
    s : Set X
    ⊢ LE.le ((MeasureTheory.Measure.mkMetric m₁) s) ((HSMul.hSMul c (MeasureTheory …
  -/
  rw [← OuterMeasure.coe_mkMetric, coe_smul, ← OuterMeasure.coe_mkMetric]
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m₁ m₂ : ENNReal → ENNReal
    c : ENNReal
    hc : Ne c Top.top
    h0 : Ne c 0
    hle : (nhdsWithin 0 (Set.Ici 0)).EventuallyLE m₁ (HSMul.hSMul c m₂)
    s : Set X
    ⊢ LE.le ((MeasureTheory.OuterMeasure.mkMetric m₁) s) (HSMul.hSMul c (⇑(Measure …
  -/
  exact OuterMeasure.mkMetric_mono_smul hc h0 hle s
  /-
    🎉 no goals
  -/


@[simp]
theorem mkMetric_top : (mkMetric (fun _ => ∞ : ℝ≥0∞ → ℝ≥0∞) : Measure X) = ⊤ := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    ⊢ Eq (MeasureTheory.Measure.mkMetric fun x => Top.top) Top.top
  -/
  apply toOuterMeasure_injective
  /-
    case a
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    ⊢ Eq (MeasureTheory.Measure.mkMetric fun x => Top.top).toOuterMeasure Top.top. …
  -/
  rw [mkMetric_toOuterMeasure, OuterMeasure.mkMetric_top, toOuterMeasure_top]
  /-
    🎉 no goals
  -/


/-- If `m₁ d ≤ m₂ d` for `d < ε` for some `ε > 0` (we use `≤ᶠ[𝓝[≥] 0]` to state this), then
`mkMetric m₁ hm₁ ≤ mkMetric m₂ hm₂`. -/
theorem mkMetric_mono {m₁ m₂ : ℝ≥0∞ → ℝ≥0∞} (hle : m₁ ≤ᶠ[𝓝[≥] 0] m₂) :
    (mkMetric m₁ : Measure X) ≤ mkMetric m₂ := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m₁ m₂ : ENNReal → ENNReal
    hle : (nhdsWithin 0 (Set.Ici 0)).EventuallyLE m₁ m₂
    ⊢ LE.le (MeasureTheory.Measure.mkMetric m₁) (MeasureTheory.Measure.mkMetric m₂)
  -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
  convert @mkMetric_mono_smul X _ _ _ _ m₂ _ ENNReal.one_ne_top one_ne_zero _ <;> simp [*]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- A formula for `MeasureTheory.Measure.mkMetric`. -/
theorem mkMetric_apply (m : ℝ≥0∞ → ℝ≥0∞) (s : Set X) :
    mkMetric m s =
      ⨆ (r : ℝ≥0∞) (_ : 0 < r),
        ⨅ (t : ℕ → Set X) (_ : s ⊆ iUnion t) (_ : ∀ n, diam (t n) ≤ r),
          ∑' n, ⨆ _ : (t n).Nonempty, m (diam (t n)) := by
  classical
  -- We mostly unfold the definitions but we need to switch the order of `∑'` and `⨅`
  simp only [← OuterMeasure.coe_mkMetric, OuterMeasure.mkMetric, OuterMeasure.mkMetric',
    OuterMeasure.iSup_apply, OuterMeasure.mkMetric'.pre, OuterMeasure.boundedBy_apply, extend]
  refine
    surjective_id.iSup_congr (id) fun r =>
      iSup_congr_Prop Iff.rfl fun _ =>
        surjective_id.iInf_congr _ fun t => iInf_congr_Prop Iff.rfl fun ht => ?_
  dsimp
  by_cases htr : ∀ n, diam (t n) ≤ r
  · rw [iInf_eq_if, if_pos htr]
    congr 1 with n : 1
    simp only [iInf_eq_if, htr n, id, if_true, iSup_and']
  · rw [iInf_eq_if, if_neg htr]
    push_neg at htr; rcases htr with ⟨n, hn⟩
    refine ENNReal.tsum_eq_top_of_eq_top ⟨n, ?_⟩
    rw [iSup_eq_if, if_pos, iInf_eq_if, if_neg]
    · exact hn.not_le
    rcases diam_pos_iff.1 ((zero_le r).trans_lt hn) with ⟨x, hx, -⟩
    exact ⟨x, hx⟩


theorem le_mkMetric (m : ℝ≥0∞ → ℝ≥0∞) (μ : Measure X) (ε : ℝ≥0∞) (h₀ : 0 < ε)
    (h : ∀ s : Set X, diam s ≤ ε → μ s ≤ m (diam s)) : μ ≤ mkMetric m := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m : ENNReal → ENNReal
    μ : MeasureTheory.Measure X
    ε : ENNReal
    h₀ : LT.lt 0 ε
    h : ∀ (s : Set X), LE.le (EMetric.diam s) ε → LE.le (μ s) (m (EMetric.diam s))
    ⊢ LE.le μ (MeasureTheory.Measure.mkMetric m)
  -/
  rw [← toOuterMeasure_le, mkMetric_toOuterMeasure]
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    m : ENNReal → ENNReal
    μ : MeasureTheory.Measure X
    ε : ENNReal
    h₀ : LT.lt 0 ε
    h : ∀ (s : Set X), LE.le (EMetric.diam s) ε → LE.le (μ s) (m (EMetric.diam s))
    ⊢ LE.le μ.toOuterMeasure (MeasureTheory.OuterMeasure.mkMetric m)
  -/
  exact OuterMeasure.le_mkMetric m μ.toOuterMeasure ε h₀ h
  /-
    🎉 no goals
  -/


/-- To bound the Hausdorff measure (or, more generally, for a measure defined using
`MeasureTheory.Measure.mkMetric`) of a set, one may use coverings with maximum diameter tending to
`0`, indexed by any sequence of countable types. -/
theorem mkMetric_le_liminf_tsum {β : Type*} {ι : β → Type*} [∀ n, Countable (ι n)] (s : Set X)
    {l : Filter β} (r : β → ℝ≥0∞) (hr : Tendsto r l (𝓝 0)) (t : ∀ n : β, ι n → Set X)
    (ht : ∀ᶠ n in l, ∀ i, diam (t n i) ≤ r n) (hst : ∀ᶠ n in l, s ⊆ ⋃ i, t n i) (m : ℝ≥0∞ → ℝ≥0∞) :
    mkMetric m s ≤ liminf (fun n => ∑' i, m (diam (t n i))) l := by
  /-
    X : Type u_2
    inst✝³ : EMetricSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    β : Type u_4
    ι : β → Type u_5
    inst✝ : ∀ (n : β), Countable (ι n)
    s : Set X
    l : Filter β
    r : β → ENNReal
    hr : Filter.Tendsto r l (nhds 0)
    t : (n : β) → ι n → Set X
    ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
    hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
    m : ENNReal → ENNReal
    ⊢ LE.le ((MeasureTheory.Measure.mkMetric m) s) (Filter.liminf (fun n => tsum f …
  -/
  haveI : ∀ n, Encodable (ι n) := fun n => Encodable.ofCountable _
  /-
    X : Type u_2
    inst✝³ : EMetricSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    β : Type u_4
    ι : β → Type u_5
    inst✝ : ∀ (n : β), Countable (ι n)
    s : Set X
    l : Filter β
    r : β → ENNReal
    hr : Filter.Tendsto r l (nhds 0)
    t : (n : β) → ι n → Set X
    ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
    hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
    m : ENNReal → ENNReal
    this : (n : β) → Encodable (ι n)
    ⊢ LE.le ((MeasureTheory.Measure.mkMetric m) s) (Filter.liminf (fun n => tsum f …
  -/
  simp only [mkMetric_apply]
  /-
    X : Type u_2
    inst✝³ : EMetricSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    β : Type u_4
    ι : β → Type u_5
    inst✝ : ∀ (n : β), Countable (ι n)
    s : Set X
    l : Filter β
    r : β → ENNReal
    hr : Filter.Tendsto r l (nhds 0)
    t : (n : β) → ι n → Set X
    ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
    hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
    m : ENNReal → ENNReal
    this : (n : β) → Encodable (ι n)
    ⊢ LE.le (iSup fun r => iSup fun x => iInf fun t => iInf fun x => iInf fun x => …
  -/
  refine iSup₂_le fun ε hε => ?_
  /-
    X : Type u_2
    inst✝³ : EMetricSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    β : Type u_4
    ι : β → Type u_5
    inst✝ : ∀ (n : β), Countable (ι n)
    s : Set X
    l : Filter β
    r : β → ENNReal
    hr : Filter.Tendsto r l (nhds 0)
    t : (n : β) → ι n → Set X
    ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
    hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
    m : ENNReal → ENNReal
    this : (n : β) → Encodable (ι n)
    ε : ENNReal
    hε : LT.lt 0 ε
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => tsum fun n => iSup fun x => …
  -/
  refine le_of_forall_le_of_dense fun c hc => ?_
  rcases ((frequently_lt_of_liminf_lt (by isBoundedDefault) hc).and_eventually
        ((hr.eventually (gt_mem_nhds hε)).and (ht.and hst))).exists with
    ⟨n, hn, hrn, htn, hstn⟩
  /-
    case intro.intro.intro.intro
    X : Type u_2
    inst✝³ : EMetricSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    β : Type u_4
    ι : β → Type u_5
    inst✝ : ∀ (n : β), Countable (ι n)
    s : Set X
    l : Filter β
    r : β → ENNReal
    hr : Filter.Tendsto r l (nhds 0)
    t : (n : β) → ι n → Set X
    ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
    hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
    m : ENNReal → ENNReal
    this : (n : β) → Encodable (ι n)
    ε : ENNReal
    hε : LT.lt 0 ε
    c : ENNReal
    hc : LT.lt (Filter.liminf (fun n => tsum fun i => m (EMetric.diam (t n i))) l) c
    n : β
    hn : LT.lt (tsum fun i => m (EMetric.diam (t n i))) c
    hrn : LT.lt (r n) ε
    htn : ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r n)
    hstn : HasSubset.Subset s (Set.iUnion fun i => t n i)
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => tsum fun n => iSup fun x => …
  -/
  set u : ℕ → Set X := fun j => ⋃ b ∈ decode₂ (ι n) j, t n b
  /-
    case intro.intro.intro.intro
    X : Type u_2
    inst✝³ : EMetricSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    β : Type u_4
    ι : β → Type u_5
    inst✝ : ∀ (n : β), Countable (ι n)
    s : Set X
    l : Filter β
    r : β → ENNReal
    hr : Filter.Tendsto r l (nhds 0)
    t : (n : β) → ι n → Set X
    ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
    hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
    m : ENNReal → ENNReal
    this : (n : β) → Encodable (ι n)
    ε : ENNReal
    hε : LT.lt 0 ε
    c : ENNReal
    hc : LT.lt (Filter.liminf (fun n => tsum fun i => m (EMetric.diam (t n i))) l) c
    n : β
    hn : LT.lt (tsum fun i => m (EMetric.diam (t n i))) c
    hrn : LT.lt (r n) ε
    htn : ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r n)
    hstn : HasSubset.Subset s (Set.iUnion fun i => t n i)
    u : Nat → Set X := fun j => Set.iUnion fun b => Set.iUnion fun h => t n b
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => tsum fun n => iSup fun x => …
  -/
  refine iInf₂_le_of_le u (by rwa [iUnion_decode₂]) ?_
  /-
    case intro.intro.intro.intro
    X : Type u_2
    inst✝³ : EMetricSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : BorelSpace X
    β : Type u_4
    ι : β → Type u_5
    inst✝ : ∀ (n : β), Countable (ι n)
    s : Set X
    l : Filter β
    r : β → ENNReal
    hr : Filter.Tendsto r l (nhds 0)
    t : (n : β) → ι n → Set X
    ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
    hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
    m : ENNReal → ENNReal
    this : (n : β) → Encodable (ι n)
    ε : ENNReal
    hε : LT.lt 0 ε
    c : ENNReal
    hc : LT.lt (Filter.liminf (fun n => tsum fun i => m (EMetric.diam (t n i))) l) c
    n : β
    hn : LT.lt (tsum fun i => m (EMetric.diam (t n i))) c
    hrn : LT.lt (r n) ε
    htn : ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r n)
    hstn : HasSubset.Subset s (Set.iUnion fun i => t n i)
    u : Nat → Set X := fun j => Set.iUnion fun b => Set.iUnion fun h => t n b
    ⊢ LE.le (iInf fun x => tsum fun n => iSup fun x => m (EMetric.diam (u n))) c
  -/
  refine iInf_le_of_le (fun j => ?_) ?_
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_2
      inst✝³ : EMetricSpace X
      inst✝² : MeasurableSpace X
      inst✝¹ : BorelSpace X
      β : Type u_4
      ι : β → Type u_5
      inst✝ : ∀ (n : β), Countable (ι n)
      s : Set X
      l : Filter β
      r : β → ENNReal
      hr : Filter.Tendsto r l (nhds 0)
      t : (n : β) → ι n → Set X
      ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
      hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
      m : ENNReal → ENNReal
      this : (n : β) → Encodable (ι n)
      ε : ENNReal
      hε : LT.lt 0 ε
      c : ENNReal
      hc : LT.lt (Filter.liminf (fun n => tsum fun i => m (EMetric.diam (t n i))) l) c
      n : β
      hn : LT.lt (tsum fun i => m (EMetric.diam (t n i))) c
      hrn : LT.lt (r n) ε
      htn : ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r n)
      hstn : HasSubset.Subset s (Set.iUnion fun i => t n i)
      u : Nat → Set X := fun j => Set.iUnion fun b => Set.iUnion fun h => t n b
      j : Nat
      ⊢ LE.le (EMetric.diam (u j)) ε
    -/
  · rw [EMetric.diam_iUnion_mem_option]
    /-
      case intro.intro.intro.intro.refine_1
      X : Type u_2
      inst✝³ : EMetricSpace X
      inst✝² : MeasurableSpace X
      inst✝¹ : BorelSpace X
      β : Type u_4
      ι : β → Type u_5
      inst✝ : ∀ (n : β), Countable (ι n)
      s : Set X
      l : Filter β
      r : β → ENNReal
      hr : Filter.Tendsto r l (nhds 0)
      t : (n : β) → ι n → Set X
      ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
      hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
      m : ENNReal → ENNReal
      this : (n : β) → Encodable (ι n)
      ε : ENNReal
      hε : LT.lt 0 ε
      c : ENNReal
      hc : LT.lt (Filter.liminf (fun n => tsum fun i => m (EMetric.diam (t n i))) l) c
      n : β
      hn : LT.lt (tsum fun i => m (EMetric.diam (t n i))) c
      hrn : LT.lt (r n) ε
      htn : ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r n)
      hstn : HasSubset.Subset s (Set.iUnion fun i => t n i)
      u : Nat → Set X := fun j => Set.iUnion fun b => Set.iUnion fun h => t n b
      j : Nat
      ⊢ LE.le (iSup fun i => iSup fun h => EMetric.diam (t n i)) ε
    -/
    exact iSup₂_le fun _ _ => (htn _).trans hrn.le
    /-
      🎉 no goals
    -/
  · calc
      (∑' j : ℕ, ⨆ _ : (u j).Nonempty, m (diam (u j))) = _ :=
        tsum_iUnion_decode₂ (fun t : Set X => ⨆ _ : t.Nonempty, m (diam t)) (by simp) _
      _ ≤ ∑' i : ι n, m (diam (t n i)) := ENNReal.tsum_le_tsum fun b => iSup_le fun _ => le_rfl
      _ ≤ c := hn.le


/-- To bound the Hausdorff measure (or, more generally, for a measure defined using
`MeasureTheory.Measure.mkMetric`) of a set, one may use coverings with maximum diameter tending to
`0`, indexed by any sequence of finite types. -/
theorem mkMetric_le_liminf_sum {β : Type*} {ι : β → Type*} [hι : ∀ n, Fintype (ι n)] (s : Set X)
    {l : Filter β} (r : β → ℝ≥0∞) (hr : Tendsto r l (𝓝 0)) (t : ∀ n : β, ι n → Set X)
    (ht : ∀ᶠ n in l, ∀ i, diam (t n i) ≤ r n) (hst : ∀ᶠ n in l, s ⊆ ⋃ i, t n i) (m : ℝ≥0∞ → ℝ≥0∞) :
    mkMetric m s ≤ liminf (fun n => ∑ i, m (diam (t n i))) l := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    β : Type u_4
    ι : β → Type u_5
    hι : (n : β) → Fintype (ι n)
    s : Set X
    l : Filter β
    r : β → ENNReal
    hr : Filter.Tendsto r l (nhds 0)
    t : (n : β) → ι n → Set X
    ht : Filter.Eventually (fun n => ∀ (i : ι n), LE.le (EMetric.diam (t n i)) (r  …
    hst : Filter.Eventually (fun n => HasSubset.Subset s (Set.iUnion fun i => t n  …
    m : ENNReal → ENNReal
    ⊢ LE.le ((MeasureTheory.Measure.mkMetric m) s) (Filter.liminf (fun n => Finset …
  -/
  simpa only [tsum_fintype] using mkMetric_le_liminf_tsum s r hr t ht hst m
  /-
    🎉 no goals
  -/


/-- Hausdorff measure on an (e)metric space. -/
def hausdorffMeasure (d : ℝ) : Measure X :=
  mkMetric fun r => r ^ d


scoped[MeasureTheory] notation "μH[" d "]" => MeasureTheory.Measure.hausdorffMeasure d


theorem le_hausdorffMeasure (d : ℝ) (μ : Measure X) (ε : ℝ≥0∞) (h₀ : 0 < ε)
    (h : ∀ s : Set X, diam s ≤ ε → μ s ≤ diam s ^ d) : μ ≤ μH[d] :=
  le_mkMetric _ μ ε h₀ h


/-- A formula for `μH[d] s`. -/
theorem hausdorffMeasure_apply (d : ℝ) (s : Set X) :
    μH[d] s =
      ⨆ (r : ℝ≥0∞) (_ : 0 < r),
        ⨅ (t : ℕ → Set X) (_ : s ⊆ ⋃ n, t n) (_ : ∀ n, diam (t n) ≤ r),
          ∑' n, ⨆ _ : (t n).Nonempty, diam (t n) ^ d :=
  mkMetric_apply _ _


/-- To bound the Hausdorff measure of a set, one may use coverings with maximum diameter tending
to `0`, indexed by any sequence of countable types. -/
theorem hausdorffMeasure_le_liminf_tsum {β : Type*} {ι : β → Type*} [∀ n, Countable (ι n)]
    (d : ℝ) (s : Set X) {l : Filter β} (r : β → ℝ≥0∞) (hr : Tendsto r l (𝓝 0))
    (t : ∀ n : β, ι n → Set X) (ht : ∀ᶠ n in l, ∀ i, diam (t n i) ≤ r n)
    (hst : ∀ᶠ n in l, s ⊆ ⋃ i, t n i) : μH[d] s ≤ liminf (fun n => ∑' i, diam (t n i) ^ d) l :=
  mkMetric_le_liminf_tsum s r hr t ht hst _


/-- To bound the Hausdorff measure of a set, one may use coverings with maximum diameter tending
to `0`, indexed by any sequence of finite types. -/
theorem hausdorffMeasure_le_liminf_sum {β : Type*} {ι : β → Type*} [∀ n, Fintype (ι n)]
    (d : ℝ) (s : Set X) {l : Filter β} (r : β → ℝ≥0∞) (hr : Tendsto r l (𝓝 0))
    (t : ∀ n : β, ι n → Set X) (ht : ∀ᶠ n in l, ∀ i, diam (t n i) ≤ r n)
    (hst : ∀ᶠ n in l, s ⊆ ⋃ i, t n i) : μH[d] s ≤ liminf (fun n => ∑ i, diam (t n i) ^ d) l :=
  mkMetric_le_liminf_sum s r hr t ht hst _


/-- If `d₁ < d₂`, then for any set `s` we have either `μH[d₂] s = 0`, or `μH[d₁] s = ∞`. -/
theorem hausdorffMeasure_zero_or_top {d₁ d₂ : ℝ} (h : d₁ < d₂) (s : Set X) :
    μH[d₂] s = 0 ∨ μH[d₁] s = ∞ := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h : LT.lt d₁ d₂
    s : Set X
    ⊢ Or (Eq ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Eq ((MeasureTheor …
  -/
  by_contra! H
  suffices ∀ c : ℝ≥0, c ≠ 0 → μH[d₂] s ≤ c * μH[d₁] s by
    rcases ENNReal.exists_nnreal_pos_mul_lt H.2 H.1 with ⟨c, hc0, hc⟩
    exact hc.not_le (this c (pos_iff_ne_zero.1 hc0))
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h : LT.lt d₁ d₂
    s : Set X
    H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
    ⊢ ∀ (c : NNReal), Ne c 0 → LE.le ((MeasureTheory.Measure.hausdorffMeasure d₂)  …
  -/
  intro c hc
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h : LT.lt d₁ d₂
    s : Set X
    H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
    c : NNReal
    hc : Ne c 0
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d₂) s) (HMul.hMul (↑c) ((Meas …
  -/
  refine le_iff'.1 (mkMetric_mono_smul ENNReal.coe_ne_top (mod_cast hc) ?_) s
  have : 0 < ((c : ℝ≥0∞) ^ (d₂ - d₁)⁻¹) := by
    rw [← ENNReal.coe_rpow_of_ne_zero hc, pos_iff_ne_zero, Ne, ENNReal.coe_eq_zero,
      NNReal.rpow_eq_zero_iff]
    exact mt And.left hc
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h : LT.lt d₁ d₂
    s : Set X
    H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
    c : NNReal
    hc : Ne c 0
    this : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
    ⊢ (nhdsWithin 0 (Set.Ici 0)).EventuallyLE (fun r => HPow.hPow r d₂) (HSMul.hSM …
  -/
  filter_upwards [Ico_mem_nhdsGE this]
  /-
    case h
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h : LT.lt d₁ d₂
    s : Set X
    H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
    c : NNReal
    hc : Ne c 0
    this : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
    ⊢ ∀ (a : ENNReal), Membership.mem (Set.Ico 0 (HPow.hPow (↑c) (Inv.inv (HSub.hS …
  -/
  rintro r ⟨hr₀, hrc⟩
  /-
    case h.intro
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h : LT.lt d₁ d₂
    s : Set X
    H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
    c : NNReal
    hc : Ne c 0
    this : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
    r : ENNReal
    hr₀ : LE.le 0 r
    hrc : LT.lt r (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
    ⊢ LE.le (HPow.hPow r d₂) (HSMul.hSMul (↑c) (fun r => HPow.hPow r d₁) r)
  -/
  lift r to ℝ≥0 using ne_top_of_lt hrc
  rw [Pi.smul_apply, smul_eq_mul,
    ← ENNReal.div_le_iff_le_mul (Or.inr ENNReal.coe_ne_top) (Or.inr <| mt ENNReal.coe_eq_zero.1 hc)]
  /-
    case h.intro.intro
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h : LT.lt d₁ d₂
    s : Set X
    H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
    c : NNReal
    hc : Ne c 0
    this : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
    r : NNReal
    hr₀ : LE.le 0 ↑r
    hrc : LT.lt (↑r) (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (↑r) d₂) (HPow.hPow (↑r) d₁)) ↑c
  -/
  rcases eq_or_ne r 0 with (rfl | hr₀)
    /-
      case h.intro.intro.inl
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d₁ d₂ : Real
      h : LT.lt d₁ d₂
      s : Set X
      H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
      c : NNReal
      hc : Ne c 0
      this : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      hr₀ : LE.le 0 ↑0
      hrc : LT.lt (↑0) (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      ⊢ LE.le (HDiv.hDiv (HPow.hPow (↑0) d₂) (HPow.hPow (↑0) d₁)) ↑c
    -/
  · rcases lt_or_le 0 d₂ with (h₂ | h₂)
      /-
        case h.intro.intro.inl.inl
        X : Type u_2
        inst✝² : EMetricSpace X
        inst✝¹ : MeasurableSpace X
        inst✝ : BorelSpace X
        d₁ d₂ : Real
        h : LT.lt d₁ d₂
        s : Set X
        H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
        c : NNReal
        hc : Ne c 0
        this : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
        hr₀ : LE.le 0 ↑0
        hrc : LT.lt (↑0) (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
        h₂ : LT.lt 0 d₂
        ⊢ LE.le (HDiv.hDiv (HPow.hPow (↑0) d₂) (HPow.hPow (↑0) d₁)) ↑c
      -/
    · simp only [h₂, ENNReal.zero_rpow_of_pos, zero_le, ENNReal.zero_div, ENNReal.coe_zero]
      /-
        🎉 no goals
      -/
    · simp only [h.trans_le h₂, ENNReal.div_top, zero_le, ENNReal.zero_rpow_of_neg,
        ENNReal.coe_zero]
    /-
      case h.intro.intro.inr
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d₁ d₂ : Real
      h : LT.lt d₁ d₂
      s : Set X
      H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
      c : NNReal
      hc : Ne c 0
      this : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      r : NNReal
      hr₀✝ : LE.le 0 ↑r
      hrc : LT.lt (↑r) (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      hr₀ : Ne r 0
      ⊢ LE.le (HDiv.hDiv (HPow.hPow (↑r) d₂) (HPow.hPow (↑r) d₁)) ↑c
    -/
  · have : (r : ℝ≥0∞) ≠ 0 := by simpa only [ENNReal.coe_eq_zero, Ne] using hr₀
    /-
      case h.intro.intro.inr
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d₁ d₂ : Real
      h : LT.lt d₁ d₂
      s : Set X
      H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
      c : NNReal
      hc : Ne c 0
      this✝ : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      r : NNReal
      hr₀✝ : LE.le 0 ↑r
      hrc : LT.lt (↑r) (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      hr₀ : Ne r 0
      this : Ne (↑r) 0
      ⊢ LE.le (HDiv.hDiv (HPow.hPow (↑r) d₂) (HPow.hPow (↑r) d₁)) ↑c
    -/
    rw [← ENNReal.rpow_sub _ _ this ENNReal.coe_ne_top]
    /-
      case h.intro.intro.inr
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d₁ d₂ : Real
      h : LT.lt d₁ d₂
      s : Set X
      H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
      c : NNReal
      hc : Ne c 0
      this✝ : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      r : NNReal
      hr₀✝ : LE.le 0 ↑r
      hrc : LT.lt (↑r) (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      hr₀ : Ne r 0
      this : Ne (↑r) 0
      ⊢ LE.le (HPow.hPow (↑r) (HSub.hSub d₂ d₁)) ↑c
    -/
    refine (ENNReal.rpow_lt_rpow hrc (sub_pos.2 h)).le.trans ?_
    /-
      case h.intro.intro.inr
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d₁ d₂ : Real
      h : LT.lt d₁ d₂
      s : Set X
      H : And (Ne ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0) (Ne ((MeasureTh …
      c : NNReal
      hc : Ne c 0
      this✝ : LT.lt 0 (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      r : NNReal
      hr₀✝ : LE.le 0 ↑r
      hrc : LT.lt (↑r) (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁)))
      hr₀ : Ne r 0
      this : Ne (↑r) 0
      ⊢ LE.le (HPow.hPow (HPow.hPow (↑c) (Inv.inv (HSub.hSub d₂ d₁))) (HSub.hSub d₂  …
    -/
    rw [← ENNReal.rpow_mul, inv_mul_cancel₀ (sub_pos.2 h).ne', ENNReal.rpow_one]
    /-
      🎉 no goals
    -/


/-- Hausdorff measure `μH[d] s` is monotone in `d`. -/
theorem hausdorffMeasure_mono {d₁ d₂ : ℝ} (h : d₁ ≤ d₂) (s : Set X) : μH[d₂] s ≤ μH[d₁] s := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h : LE.le d₁ d₂
    s : Set X
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d₂) s) ((MeasureTheory.Measur …
  -/
  rcases h.eq_or_lt with (rfl | h); · exact le_rfl
                                      /-
                                        🎉 no goals
                                      -/
  /-
    case inr
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d₁ d₂ : Real
    h✝ : LE.le d₁ d₂
    s : Set X
    h : LT.lt d₁ d₂
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d₂) s) ((MeasureTheory.Measur …
  -/
  cases' hausdorffMeasure_zero_or_top h s with hs hs
    /-
      case inr.inl
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d₁ d₂ : Real
      h✝ : LE.le d₁ d₂
      s : Set X
      h : LT.lt d₁ d₂
      hs : Eq ((MeasureTheory.Measure.hausdorffMeasure d₂) s) 0
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d₂) s) ((MeasureTheory.Measur …
    -/
  · rw [hs]; exact zero_le _
             /-
               🎉 no goals
             -/
    /-
      case inr.inr
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d₁ d₂ : Real
      h✝ : LE.le d₁ d₂
      s : Set X
      h : LT.lt d₁ d₂
      hs : Eq ((MeasureTheory.Measure.hausdorffMeasure d₁) s) Top.top
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d₂) s) ((MeasureTheory.Measur …
    -/
  · rw [hs]; exact le_top
             /-
               🎉 no goals
             -/


theorem noAtoms_hausdorff {d : ℝ} (hd : 0 < d) : NoAtoms (hausdorffMeasure d : Measure X) := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d : Real
    hd : LT.lt 0 d
    ⊢ MeasureTheory.NoAtoms (MeasureTheory.Measure.hausdorffMeasure d)
  -/
  refine ⟨fun x => ?_⟩
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d : Real
    hd : LT.lt 0 d
    x : X
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton x)) 0
  -/
  rw [← nonpos_iff_eq_zero, hausdorffMeasure_apply]
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    d : Real
    hd : LT.lt 0 d
    x : X
    ⊢ LE.le (iSup fun r => iSup fun x_1 => iInf fun t => iInf fun x => iInf fun x  …
  -/
  refine iSup₂_le fun ε _ => iInf₂_le_of_le (fun _ => {x}) ?_ <| iInf_le_of_le (fun _ => ?_) ?_
    /-
      case refine_1
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d : Real
      hd : LT.lt 0 d
      x : X
      ε : ENNReal
      x✝ : LT.lt 0 ε
      ⊢ HasSubset.Subset (Singleton.singleton x) (Set.iUnion fun n => (fun x_1 => Si …
    -/
  · exact subset_iUnion (fun _ => {x} : ℕ → Set X) 0
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d : Real
      hd : LT.lt 0 d
      x : X
      ε : ENNReal
      x✝¹ : LT.lt 0 ε
      x✝ : Nat
      ⊢ LE.le (EMetric.diam ((fun x_1 => Singleton.singleton x) x✝)) ε
    -/
  · simp only [EMetric.diam_singleton, zero_le]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d : Real
      hd : LT.lt 0 d
      x : X
      ε : ENNReal
      x✝ : LT.lt 0 ε
      ⊢ LE.le (tsum fun n => iSup fun x_1 => HPow.hPow (EMetric.diam ((fun x_2 => Si …
    -/
  · simp [hd]
    /-
      🎉 no goals
    -/


@[simp]
theorem hausdorffMeasure_zero_singleton (x : X) : μH[0] ({x} : Set X) = 1 := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    x : X
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton x)) 1
  -/
  apply le_antisymm
    /-
      case a
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      x : X
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton x)) 1
    -/
  · let r : ℕ → ℝ≥0∞ := fun _ => 0
    /-
      case a
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      x : X
      r : Nat → ENNReal := fun x => 0
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton x)) 1
    -/
    let t : ℕ → Unit → Set X := fun _ _ => {x}
    have ht : ∀ᶠ n in atTop, ∀ i, diam (t n i) ≤ r n := by
      simp only [t, r, imp_true_iff, eq_self_iff_true, diam_singleton, eventually_atTop,
        nonpos_iff_eq_zero, exists_const]
    /-
      case a
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      x : X
      r : Nat → ENNReal := fun x => 0
      t : Nat → Unit → Set X := fun x_1 x_2 => Singleton.singleton x
      ht : Filter.Eventually (fun n => ∀ (i : Unit), LE.le (EMetric.diam (t n i)) (r …
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton x)) 1
    -/
    simpa [t, liminf_const] using hausdorffMeasure_le_liminf_sum 0 {x} r tendsto_const_nhds t ht
    /-
      🎉 no goals
    -/
    /-
      case a
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      x : X
      ⊢ LE.le 1 ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton x))
    -/
  · rw [hausdorffMeasure_apply]
    suffices
      (1 : ℝ≥0∞) ≤
        ⨅ (t : ℕ → Set X) (_ : {x} ⊆ ⋃ n, t n) (_ : ∀ n, diam (t n) ≤ 1),
          ∑' n, ⨆ _ : (t n).Nonempty, diam (t n) ^ (0 : ℝ) by
      apply le_trans this _
      convert le_iSup₂ (α := ℝ≥0∞) (1 : ℝ≥0∞) zero_lt_one
      rfl
    /-
      case a
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      x : X
      ⊢ LE.le 1 (iInf fun t => iInf fun x => iInf fun x => tsum fun n => iSup fun x  …
    -/
    simp only [ENNReal.rpow_zero, le_iInf_iff]
    /-
      case a
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      x : X
      ⊢ ∀ (i : Nat → Set X), HasSubset.Subset (Singleton.singleton x) (Set.iUnion fu …
    -/
    intro t hst _
    /-
      case a
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      x : X
      t : Nat → Set X
      hst : HasSubset.Subset (Singleton.singleton x) (Set.iUnion fun n => t n)
      i✝ : ∀ (n : Nat), LE.le (EMetric.diam (t n)) 1
      ⊢ LE.le 1 (tsum fun n => iSup fun x => 1)
    -/
    rcases mem_iUnion.1 (hst (mem_singleton x)) with ⟨m, hm⟩
    /-
      case a.intro
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      x : X
      t : Nat → Set X
      hst : HasSubset.Subset (Singleton.singleton x) (Set.iUnion fun n => t n)
      i✝ : ∀ (n : Nat), LE.le (EMetric.diam (t n)) 1
      m : Nat
      hm : Membership.mem (t m) x
      ⊢ LE.le 1 (tsum fun n => iSup fun x => 1)
    -/
    have A : (t m).Nonempty := ⟨x, hm⟩
    calc
      (1 : ℝ≥0∞) = ⨆ h : (t m).Nonempty, 1 := by simp only [A, ciSup_pos]
      _ ≤ ∑' n, ⨆ h : (t n).Nonempty, 1 := ENNReal.le_tsum _


theorem one_le_hausdorffMeasure_zero_of_nonempty {s : Set X} (h : s.Nonempty) : 1 ≤ μH[0] s := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    h : s.Nonempty
    ⊢ LE.le 1 ((MeasureTheory.Measure.hausdorffMeasure 0) s)
  -/
  rcases h with ⟨x, hx⟩
  calc
    (1 : ℝ≥0∞) = μH[0] ({x} : Set X) := (hausdorffMeasure_zero_singleton x).symm
    _ ≤ μH[0] s := measure_mono (singleton_subset_iff.2 hx)


theorem hausdorffMeasure_le_one_of_subsingleton {s : Set X} (hs : s.Subsingleton) {d : ℝ}
    (hd : 0 ≤ d) : μH[d] s ≤ 1 := by
  /-
    X : Type u_2
    inst✝² : EMetricSpace X
    inst✝¹ : MeasurableSpace X
    inst✝ : BorelSpace X
    s : Set X
    hs : s.Subsingleton
    d : Real
    hd : LE.le 0 d
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) s) 1
  -/
  rcases eq_empty_or_nonempty s with (rfl | ⟨x, hx⟩)
    /-
      case inl
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      d : Real
      hd : LE.le 0 d
      hs : EmptyCollection.emptyCollection.Subsingleton
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) EmptyCollection.emptyColle …
    -/
  · simp only [measure_empty, zero_le]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      s : Set X
      hs : s.Subsingleton
      d : Real
      hd : LE.le 0 d
      x : X
      hx : Membership.mem s x
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) s) 1
    -/
  · rw [(subsingleton_iff_singleton hx).1 hs]
    /-
      case inr.intro
      X : Type u_2
      inst✝² : EMetricSpace X
      inst✝¹ : MeasurableSpace X
      inst✝ : BorelSpace X
      s : Set X
      hs : s.Subsingleton
      d : Real
      hd : LE.le 0 d
      x : X
      hx : Membership.mem s x
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton x)) 1
    -/
    rcases eq_or_lt_of_le hd with (rfl | dpos)
      /-
        case inr.intro.inl
        X : Type u_2
        inst✝² : EMetricSpace X
        inst✝¹ : MeasurableSpace X
        inst✝ : BorelSpace X
        s : Set X
        hs : s.Subsingleton
        x : X
        hx : Membership.mem s x
        hd : LE.le 0 0
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton x)) 1
      -/
    · simp only [le_refl, hausdorffMeasure_zero_singleton]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.inr
        X : Type u_2
        inst✝² : EMetricSpace X
        inst✝¹ : MeasurableSpace X
        inst✝ : BorelSpace X
        s : Set X
        hs : s.Subsingleton
        d : Real
        hd : LE.le 0 d
        x : X
        hx : Membership.mem s x
        dpos : LT.lt 0 d
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton x)) 1
      -/
    · haveI := noAtoms_hausdorff X dpos
      /-
        case inr.intro.inr
        X : Type u_2
        inst✝² : EMetricSpace X
        inst✝¹ : MeasurableSpace X
        inst✝ : BorelSpace X
        s : Set X
        hs : s.Subsingleton
        d : Real
        hd : LE.le 0 d
        x : X
        hx : Membership.mem s x
        dpos : LT.lt 0 d
        this : MeasureTheory.NoAtoms (MeasureTheory.Measure.hausdorffMeasure d)
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton x)) 1
      -/
      simp only [zero_le, measure_singleton]
      /-
        🎉 no goals
      -/


/-- If `f : X → Y` is Hölder continuous on `s` with a positive exponent `r`, then
`μH[d] (f '' s) ≤ C ^ d * μH[r * d] s`. -/
theorem hausdorffMeasure_image_le (h : HolderOnWith C r f s) (hr : 0 < r) {d : ℝ} (hd : 0 ≤ d) :
    μH[d] (f '' s) ≤ (C : ℝ≥0∞) ^ d * μH[r * d] s := by
  -- We start with the trivial case `C = 0`
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    C r : NNReal
    f : X → Y
    s : Set X
    h : HolderOnWith C r f s
    hr : LT.lt 0 r
    d : Real
    hd : LE.le 0 d
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image f s)) (HMul.hMu …
  -/
  rcases (zero_le C).eq_or_lt with (rfl | hC0)
    /-
      case inl
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      r : NNReal
      f : X → Y
      s : Set X
      hr : LT.lt 0 r
      d : Real
      hd : LE.le 0 d
      h : HolderOnWith 0 r f s
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image f s)) (HMul.hMu …
    -/
  · rcases eq_empty_or_nonempty s with (rfl | ⟨x, hx⟩)
      /-
        case inl.inl
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        r : NNReal
        f : X → Y
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        h : HolderOnWith 0 r f EmptyCollection.emptyCollection
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image f EmptyCollecti …
      -/
    · simp only [measure_empty, nonpos_iff_eq_zero, mul_zero, image_empty]
      /-
        🎉 no goals
      -/
    have : f '' s = {f x} :=
      have : (f '' s).Subsingleton := by simpa [diam_eq_zero_iff] using h.ediam_image_le
      (subsingleton_iff_singleton (mem_image_of_mem f hx)).1 this
    /-
      case inl.inr.intro
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      r : NNReal
      f : X → Y
      s : Set X
      hr : LT.lt 0 r
      d : Real
      hd : LE.le 0 d
      h : HolderOnWith 0 r f s
      x : X
      hx : Membership.mem s x
      this : Eq (Set.image f s) (Singleton.singleton (f x))
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image f s)) (HMul.hMu …
    -/
    rw [this]
    /-
      case inl.inr.intro
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      r : NNReal
      f : X → Y
      s : Set X
      hr : LT.lt 0 r
      d : Real
      hd : LE.le 0 d
      h : HolderOnWith 0 r f s
      x : X
      hx : Membership.mem s x
      this : Eq (Set.image f s) (Singleton.singleton (f x))
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton (f x) …
    -/
    rcases eq_or_lt_of_le hd with (rfl | h'd)
      /-
        case inl.inr.intro.inl
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        r : NNReal
        f : X → Y
        s : Set X
        hr : LT.lt 0 r
        h : HolderOnWith 0 r f s
        x : X
        hx : Membership.mem s x
        this : Eq (Set.image f s) (Singleton.singleton (f x))
        hd : LE.le 0 0
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton (f x) …
      -/
    · simp only [ENNReal.rpow_zero, one_mul, mul_zero]
      /-
        case inl.inr.intro.inl
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        r : NNReal
        f : X → Y
        s : Set X
        hr : LT.lt 0 r
        h : HolderOnWith 0 r f s
        x : X
        hx : Membership.mem s x
        this : Eq (Set.image f s) (Singleton.singleton (f x))
        hd : LE.le 0 0
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton (f x) …
      -/
      rw [hausdorffMeasure_zero_singleton]
      /-
        case inl.inr.intro.inl
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        r : NNReal
        f : X → Y
        s : Set X
        hr : LT.lt 0 r
        h : HolderOnWith 0 r f s
        x : X
        hx : Membership.mem s x
        this : Eq (Set.image f s) (Singleton.singleton (f x))
        hd : LE.le 0 0
        ⊢ LE.le 1 ((MeasureTheory.Measure.hausdorffMeasure 0) s)
      -/
      exact one_le_hausdorffMeasure_zero_of_nonempty ⟨x, hx⟩
      /-
        🎉 no goals
      -/
      /-
        case inl.inr.intro.inr
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        r : NNReal
        f : X → Y
        s : Set X
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        h : HolderOnWith 0 r f s
        x : X
        hx : Membership.mem s x
        this : Eq (Set.image f s) (Singleton.singleton (f x))
        h'd : LT.lt 0 d
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton (f x) …
      -/
    · haveI := noAtoms_hausdorff Y h'd
      /-
        case inl.inr.intro.inr
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        r : NNReal
        f : X → Y
        s : Set X
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        h : HolderOnWith 0 r f s
        x : X
        hx : Membership.mem s x
        this✝ : Eq (Set.image f s) (Singleton.singleton (f x))
        h'd : LT.lt 0 d
        this : MeasureTheory.NoAtoms (MeasureTheory.Measure.hausdorffMeasure d)
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton (f x) …
      -/
      simp only [zero_le, measure_singleton]
      /-
        🎉 no goals
      -/
  -- Now assume `C ≠ 0`
    /-
      case inr
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      C r : NNReal
      f : X → Y
      s : Set X
      h : HolderOnWith C r f s
      hr : LT.lt 0 r
      d : Real
      hd : LE.le 0 d
      hC0 : LT.lt 0 C
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image f s)) (HMul.hMu …
    -/
  · have hCd0 : (C : ℝ≥0∞) ^ d ≠ 0 := by simp [hC0.ne']
    /-
      case inr
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      C r : NNReal
      f : X → Y
      s : Set X
      h : HolderOnWith C r f s
      hr : LT.lt 0 r
      d : Real
      hd : LE.le 0 d
      hC0 : LT.lt 0 C
      hCd0 : Ne (HPow.hPow (↑C) d) 0
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image f s)) (HMul.hMu …
    -/
    have hCd : (C : ℝ≥0∞) ^ d ≠ ∞ := by simp [hd]
    simp only [hausdorffMeasure_apply, ENNReal.mul_iSup, ENNReal.mul_iInf_of_ne hCd0 hCd,
      ← ENNReal.tsum_mul_left]
    /-
      case inr
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      C r : NNReal
      f : X → Y
      s : Set X
      h : HolderOnWith C r f s
      hr : LT.lt 0 r
      d : Real
      hd : LE.le 0 d
      hC0 : LT.lt 0 C
      hCd0 : Ne (HPow.hPow (↑C) d) 0
      hCd : Ne (HPow.hPow (↑C) d) Top.top
      ⊢ LE.le (iSup fun r => iSup fun x => iInf fun t => iInf fun x => iInf fun x => …
    -/
    refine iSup_le fun R => iSup_le fun hR => ?_
    have : Tendsto (fun d : ℝ≥0∞ => (C : ℝ≥0∞) * d ^ (r : ℝ)) (𝓝 0) (𝓝 0) :=
      ENNReal.tendsto_const_mul_rpow_nhds_zero_of_pos ENNReal.coe_ne_top hr
    rcases ENNReal.nhds_zero_basis_Iic.eventually_iff.1 (this.eventually (gt_mem_nhds hR)) with
      ⟨δ, δ0, H⟩
    refine le_iSup₂_of_le δ δ0 <| iInf₂_mono' fun t hst ↦
      ⟨fun n => f '' (t n ∩ s), ?_, iInf_mono' fun htδ ↦
        ⟨fun n => (h.ediam_image_inter_le (t n)).trans (H (htδ n)).le, ?_⟩⟩
      /-
        case inr.intro.intro.refine_1
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        C r : NNReal
        f : X → Y
        s : Set X
        h : HolderOnWith C r f s
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        hC0 : LT.lt 0 C
        hCd0 : Ne (HPow.hPow (↑C) d) 0
        hCd : Ne (HPow.hPow (↑C) d) Top.top
        R : ENNReal
        hR : LT.lt 0 R
        this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
        δ : ENNReal
        δ0 : LT.lt 0 δ
        H : ∀ ⦃x : ENNReal⦄, Membership.mem (Set.Iic δ) x → LT.lt (HMul.hMul (↑C) (HPo …
        t : Nat → Set X
        hst : HasSubset.Subset s (Set.iUnion fun n => t n)
        ⊢ HasSubset.Subset (Set.image f s) (Set.iUnion fun n => (fun n => Set.image f  …
      -/
    · rw [← image_iUnion, ← iUnion_inter]
      /-
        case inr.intro.intro.refine_1
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        C r : NNReal
        f : X → Y
        s : Set X
        h : HolderOnWith C r f s
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        hC0 : LT.lt 0 C
        hCd0 : Ne (HPow.hPow (↑C) d) 0
        hCd : Ne (HPow.hPow (↑C) d) Top.top
        R : ENNReal
        hR : LT.lt 0 R
        this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
        δ : ENNReal
        δ0 : LT.lt 0 δ
        H : ∀ ⦃x : ENNReal⦄, Membership.mem (Set.Iic δ) x → LT.lt (HMul.hMul (↑C) (HPo …
        t : Nat → Set X
        hst : HasSubset.Subset s (Set.iUnion fun n => t n)
        ⊢ HasSubset.Subset (Set.image f s) (Set.image f (Inter.inter (Set.iUnion fun i …
      -/
      exact image_subset _ (subset_inter hst Subset.rfl)
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.refine_2
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        C r : NNReal
        f : X → Y
        s : Set X
        h : HolderOnWith C r f s
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        hC0 : LT.lt 0 C
        hCd0 : Ne (HPow.hPow (↑C) d) 0
        hCd : Ne (HPow.hPow (↑C) d) Top.top
        R : ENNReal
        hR : LT.lt 0 R
        this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
        δ : ENNReal
        δ0 : LT.lt 0 δ
        H : ∀ ⦃x : ENNReal⦄, Membership.mem (Set.Iic δ) x → LT.lt (HMul.hMul (↑C) (HPo …
        t : Nat → Set X
        hst : HasSubset.Subset s (Set.iUnion fun n => t n)
        htδ : ∀ (n : Nat), LE.le (EMetric.diam (t n)) δ
        ⊢ LE.le (tsum fun n => iSup fun x => HPow.hPow (EMetric.diam ((fun n => Set.im …
      -/
    · refine ENNReal.tsum_le_tsum fun n => ?_
      /-
        case inr.intro.intro.refine_2
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        C r : NNReal
        f : X → Y
        s : Set X
        h : HolderOnWith C r f s
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        hC0 : LT.lt 0 C
        hCd0 : Ne (HPow.hPow (↑C) d) 0
        hCd : Ne (HPow.hPow (↑C) d) Top.top
        R : ENNReal
        hR : LT.lt 0 R
        this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
        δ : ENNReal
        δ0 : LT.lt 0 δ
        H : ∀ ⦃x : ENNReal⦄, Membership.mem (Set.Iic δ) x → LT.lt (HMul.hMul (↑C) (HPo …
        t : Nat → Set X
        hst : HasSubset.Subset s (Set.iUnion fun n => t n)
        htδ : ∀ (n : Nat), LE.le (EMetric.diam (t n)) δ
        n : Nat
        ⊢ LE.le (iSup fun x => HPow.hPow (EMetric.diam ((fun n => Set.image f (Inter.i …
      -/
      simp only [iSup_le_iff, image_nonempty]
      /-
        case inr.intro.intro.refine_2
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        C r : NNReal
        f : X → Y
        s : Set X
        h : HolderOnWith C r f s
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        hC0 : LT.lt 0 C
        hCd0 : Ne (HPow.hPow (↑C) d) 0
        hCd : Ne (HPow.hPow (↑C) d) Top.top
        R : ENNReal
        hR : LT.lt 0 R
        this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
        δ : ENNReal
        δ0 : LT.lt 0 δ
        H : ∀ ⦃x : ENNReal⦄, Membership.mem (Set.Iic δ) x → LT.lt (HMul.hMul (↑C) (HPo …
        t : Nat → Set X
        hst : HasSubset.Subset s (Set.iUnion fun n => t n)
        htδ : ∀ (n : Nat), LE.le (EMetric.diam (t n)) δ
        n : Nat
        ⊢ (Inter.inter (t n) s).Nonempty → LE.le (HPow.hPow (EMetric.diam (Set.image f …
      -/
      intro hft
      /-
        case inr.intro.intro.refine_2
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        C r : NNReal
        f : X → Y
        s : Set X
        h : HolderOnWith C r f s
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        hC0 : LT.lt 0 C
        hCd0 : Ne (HPow.hPow (↑C) d) 0
        hCd : Ne (HPow.hPow (↑C) d) Top.top
        R : ENNReal
        hR : LT.lt 0 R
        this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
        δ : ENNReal
        δ0 : LT.lt 0 δ
        H : ∀ ⦃x : ENNReal⦄, Membership.mem (Set.Iic δ) x → LT.lt (HMul.hMul (↑C) (HPo …
        t : Nat → Set X
        hst : HasSubset.Subset s (Set.iUnion fun n => t n)
        htδ : ∀ (n : Nat), LE.le (EMetric.diam (t n)) δ
        n : Nat
        hft : (Inter.inter (t n) s).Nonempty
        ⊢ LE.le (HPow.hPow (EMetric.diam (Set.image f (Inter.inter (t n) s))) d) (iSup …
      -/
      simp only [Nonempty.mono ((t n).inter_subset_left) hft, ciSup_pos]
      /-
        case inr.intro.intro.refine_2
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        C r : NNReal
        f : X → Y
        s : Set X
        h : HolderOnWith C r f s
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        hC0 : LT.lt 0 C
        hCd0 : Ne (HPow.hPow (↑C) d) 0
        hCd : Ne (HPow.hPow (↑C) d) Top.top
        R : ENNReal
        hR : LT.lt 0 R
        this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
        δ : ENNReal
        δ0 : LT.lt 0 δ
        H : ∀ ⦃x : ENNReal⦄, Membership.mem (Set.Iic δ) x → LT.lt (HMul.hMul (↑C) (HPo …
        t : Nat → Set X
        hst : HasSubset.Subset s (Set.iUnion fun n => t n)
        htδ : ∀ (n : Nat), LE.le (EMetric.diam (t n)) δ
        n : Nat
        hft : (Inter.inter (t n) s).Nonempty
        ⊢ LE.le (HPow.hPow (EMetric.diam (Set.image f (Inter.inter (t n) s))) d) (HMul …
      -/
      rw [ENNReal.rpow_mul, ← ENNReal.mul_rpow_of_nonneg _ _ hd]
      /-
        case inr.intro.intro.refine_2
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        C r : NNReal
        f : X → Y
        s : Set X
        h : HolderOnWith C r f s
        hr : LT.lt 0 r
        d : Real
        hd : LE.le 0 d
        hC0 : LT.lt 0 C
        hCd0 : Ne (HPow.hPow (↑C) d) 0
        hCd : Ne (HPow.hPow (↑C) d) Top.top
        R : ENNReal
        hR : LT.lt 0 R
        this : Filter.Tendsto (fun d => HMul.hMul (↑C) (HPow.hPow d ↑r)) (nhds 0) (nhd …
        δ : ENNReal
        δ0 : LT.lt 0 δ
        H : ∀ ⦃x : ENNReal⦄, Membership.mem (Set.Iic δ) x → LT.lt (HMul.hMul (↑C) (HPo …
        t : Nat → Set X
        hst : HasSubset.Subset s (Set.iUnion fun n => t n)
        htδ : ∀ (n : Nat), LE.le (EMetric.diam (t n)) δ
        n : Nat
        hft : (Inter.inter (t n) s).Nonempty
        ⊢ LE.le (HPow.hPow (EMetric.diam (Set.image f (Inter.inter (t n) s))) d) (HPow …
      -/
      exact ENNReal.rpow_le_rpow (h.ediam_image_inter_le _) hd
      /-
        🎉 no goals
      -/


/-- If `f : X → Y` is `K`-Lipschitz on `s`, then `μH[d] (f '' s) ≤ K ^ d * μH[d] s`. -/
theorem hausdorffMeasure_image_le (h : LipschitzOnWith K f s) {d : ℝ} (hd : 0 ≤ d) :
    μH[d] (f '' s) ≤ (K : ℝ≥0∞) ^ d * μH[d] s := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    K : NNReal
    f : X → Y
    s : Set X
    h : LipschitzOnWith K f s
    d : Real
    hd : LE.le 0 d
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image f s)) (HMul.hMu …
  -/
  simpa only [NNReal.coe_one, one_mul] using h.holderOnWith.hausdorffMeasure_image_le zero_lt_one hd
  /-
    🎉 no goals
  -/


/-- If `f` is a `K`-Lipschitz map, then it increases the Hausdorff `d`-measures of sets at most
by the factor of `K ^ d`. -/
theorem hausdorffMeasure_image_le (h : LipschitzWith K f) {d : ℝ} (hd : 0 ≤ d) (s : Set X) :
    μH[d] (f '' s) ≤ (K : ℝ≥0∞) ^ d * μH[d] s :=
  h.lipschitzOnWith.hausdorffMeasure_image_le hd


theorem MeasureTheory.Measure.hausdorffMeasure_smul₀ {𝕜 E : Type*} [NormedAddCommGroup E]
    [NormedField 𝕜] [NormedSpace 𝕜 E] [MeasurableSpace E] [BorelSpace E] {d : ℝ} (hd : 0 ≤ d)
    {r : 𝕜} (hr : r ≠ 0) (s : Set E) : μH[d] (r • s) = ‖r‖₊ ^ d • μH[d] s := by
  have {r : 𝕜} (s : Set E) : μH[d] (r • s) ≤ ‖r‖₊ ^ d • μH[d] s := by
    simpa [ENNReal.coe_rpow_of_nonneg, hd]
      using (lipschitzWith_smul r).hausdorffMeasure_image_le hd s
  /-
    𝕜 : Type u_4
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    d : Real
    hd : LE.le 0 d
    r : 𝕜
    hr : Ne r 0
    s : Set E
    this : ∀ {r : 𝕜} (s : Set E), LE.le ((MeasureTheory.Measure.hausdorffMeasure d …
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure d) (HSMul.hSMul r s)) (HSMul.hSM …
  -/
  refine le_antisymm (this s) ?_
  /-
    𝕜 : Type u_4
    E : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    d : Real
    hd : LE.le 0 d
    r : 𝕜
    hr : Ne r 0
    s : Set E
    this : ∀ {r : 𝕜} (s : Set E), LE.le ((MeasureTheory.Measure.hausdorffMeasure d …
    ⊢ LE.le (HSMul.hSMul (HPow.hPow (NNNorm.nnnorm r) d) ((MeasureTheory.Measure.h …
  -/
  rw [← le_inv_smul_iff_of_pos]
    /-
      𝕜 : Type u_4
      E : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      d : Real
      hd : LE.le 0 d
      r : 𝕜
      hr : Ne r 0
      s : Set E
      this : ∀ {r : 𝕜} (s : Set E), LE.le ((MeasureTheory.Measure.hausdorffMeasure d …
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) s) (HSMul.hSMul (Inv.inv ( …
    -/
  · dsimp
    /-
      𝕜 : Type u_4
      E : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      d : Real
      hd : LE.le 0 d
      r : 𝕜
      hr : Ne r 0
      s : Set E
      this : ∀ {r : 𝕜} (s : Set E), LE.le ((MeasureTheory.Measure.hausdorffMeasure d …
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) s) (HMul.hMul (↑(Inv.inv ( …
    -/
    rw [← NNReal.inv_rpow, ← nnnorm_inv]
      /-
        𝕜 : Type u_4
        E : Type u_5
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedField 𝕜
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        d : Real
        hd : LE.le 0 d
        r : 𝕜
        hr : Ne r 0
        s : Set E
        this : ∀ {r : 𝕜} (s : Set E), LE.le ((MeasureTheory.Measure.hausdorffMeasure d …
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) s) (HMul.hMul (↑(HPow.hPow …
      -/
    · refine Eq.trans_le ?_ (this (r • s))
      /-
        𝕜 : Type u_4
        E : Type u_5
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedField 𝕜
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        d : Real
        hd : LE.le 0 d
        r : 𝕜
        hr : Ne r 0
        s : Set E
        this : ∀ {r : 𝕜} (s : Set E), LE.le ((MeasureTheory.Measure.hausdorffMeasure d …
        ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure d) s) ((MeasureTheory.Measure.ha …
      -/
      rw [inv_smul_smul₀ hr]
      /-
        🎉 no goals
      -/
    /-
      𝕜 : Type u_4
      E : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      d : Real
      hd : LE.le 0 d
      r : 𝕜
      hr : Ne r 0
      s : Set E
      this : ∀ {r : 𝕜} (s : Set E), LE.le ((MeasureTheory.Measure.hausdorffMeasure d …
      ⊢ LT.lt 0 (HPow.hPow (NNNorm.nnnorm r) d)
    -/
  · simp [pos_iff_ne_zero, hr]
    /-
      🎉 no goals
    -/


theorem hausdorffMeasure_preimage_le (hf : AntilipschitzWith K f) (hd : 0 ≤ d) (s : Set Y) :
    μH[d] (f ⁻¹' s) ≤ (K : ℝ≥0∞) ^ d * μH[d] s := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    K : NNReal
    d : Real
    hf : AntilipschitzWith K f
    hd : LE.le 0 d
    s : Set Y
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage f s)) (HMul. …
  -/
  rcases eq_or_ne K 0 with (rfl | h0)
    /-
      case inl
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      f : X → Y
      d : Real
      hd : LE.le 0 d
      s : Set Y
      hf : AntilipschitzWith 0 f
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage f s)) (HMul. …
    -/
  · rcases eq_empty_or_nonempty (f ⁻¹' s) with (hs | ⟨x, hx⟩)
      /-
        case inl.inl
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        f : X → Y
        d : Real
        hd : LE.le 0 d
        s : Set Y
        hf : AntilipschitzWith 0 f
        hs : Eq (Set.preimage f s) EmptyCollection.emptyCollection
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage f s)) (HMul. …
      -/
    · simp only [hs, measure_empty, zero_le]
      /-
        🎉 no goals
      -/
    have : f ⁻¹' s = {x} := by
      haveI : Subsingleton X := hf.subsingleton
      have : (f ⁻¹' s).Subsingleton := subsingleton_univ.anti (subset_univ _)
      exact (subsingleton_iff_singleton hx).1 this
    /-
      case inl.inr.intro
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      f : X → Y
      d : Real
      hd : LE.le 0 d
      s : Set Y
      hf : AntilipschitzWith 0 f
      x : X
      hx : Membership.mem (Set.preimage f s) x
      this : Eq (Set.preimage f s) (Singleton.singleton x)
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage f s)) (HMul. …
    -/
    rw [this]
    /-
      case inl.inr.intro
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      f : X → Y
      d : Real
      hd : LE.le 0 d
      s : Set Y
      hf : AntilipschitzWith 0 f
      x : X
      hx : Membership.mem (Set.preimage f s) x
      this : Eq (Set.preimage f s) (Singleton.singleton x)
      ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton x)) ( …
    -/
    rcases eq_or_lt_of_le hd with (rfl | h'd)
      /-
        case inl.inr.intro.inl
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        f : X → Y
        s : Set Y
        hf : AntilipschitzWith 0 f
        x : X
        hx : Membership.mem (Set.preimage f s) x
        this : Eq (Set.preimage f s) (Singleton.singleton x)
        hd : LE.le 0 0
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton x)) ( …
      -/
    · simp only [ENNReal.rpow_zero, one_mul, mul_zero]
      /-
        case inl.inr.intro.inl
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        f : X → Y
        s : Set Y
        hf : AntilipschitzWith 0 f
        x : X
        hx : Membership.mem (Set.preimage f s) x
        this : Eq (Set.preimage f s) (Singleton.singleton x)
        hd : LE.le 0 0
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure 0) (Singleton.singleton x)) ( …
      -/
      rw [hausdorffMeasure_zero_singleton]
      /-
        case inl.inr.intro.inl
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        f : X → Y
        s : Set Y
        hf : AntilipschitzWith 0 f
        x : X
        hx : Membership.mem (Set.preimage f s) x
        this : Eq (Set.preimage f s) (Singleton.singleton x)
        hd : LE.le 0 0
        ⊢ LE.le 1 ((MeasureTheory.Measure.hausdorffMeasure 0) s)
      -/
      exact one_le_hausdorffMeasure_zero_of_nonempty ⟨f x, hx⟩
      /-
        🎉 no goals
      -/
      /-
        case inl.inr.intro.inr
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        f : X → Y
        d : Real
        hd : LE.le 0 d
        s : Set Y
        hf : AntilipschitzWith 0 f
        x : X
        hx : Membership.mem (Set.preimage f s) x
        this : Eq (Set.preimage f s) (Singleton.singleton x)
        h'd : LT.lt 0 d
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton x)) ( …
      -/
    · haveI := noAtoms_hausdorff X h'd
      /-
        case inl.inr.intro.inr
        X : Type u_2
        Y : Type u_3
        inst✝⁵ : EMetricSpace X
        inst✝⁴ : EMetricSpace Y
        inst✝³ : MeasurableSpace X
        inst✝² : BorelSpace X
        inst✝¹ : MeasurableSpace Y
        inst✝ : BorelSpace Y
        f : X → Y
        d : Real
        hd : LE.le 0 d
        s : Set Y
        hf : AntilipschitzWith 0 f
        x : X
        hx : Membership.mem (Set.preimage f s) x
        this✝ : Eq (Set.preimage f s) (Singleton.singleton x)
        h'd : LT.lt 0 d
        this : MeasureTheory.NoAtoms (MeasureTheory.Measure.hausdorffMeasure d)
        ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Singleton.singleton x)) ( …
      -/
      simp only [zero_le, measure_singleton]
      /-
        🎉 no goals
      -/
  /-
    case inr
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    K : NNReal
    d : Real
    hf : AntilipschitzWith K f
    hd : LE.le 0 d
    s : Set Y
    h0 : Ne K 0
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage f s)) (HMul. …
  -/
  have hKd0 : (K : ℝ≥0∞) ^ d ≠ 0 := by simp [h0]
  /-
    case inr
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    K : NNReal
    d : Real
    hf : AntilipschitzWith K f
    hd : LE.le 0 d
    s : Set Y
    h0 : Ne K 0
    hKd0 : Ne (HPow.hPow (↑K) d) 0
    ⊢ LE.le ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage f s)) (HMul. …
  -/
  have hKd : (K : ℝ≥0∞) ^ d ≠ ∞ := by simp [hd]
  simp only [hausdorffMeasure_apply, ENNReal.mul_iSup, ENNReal.mul_iInf_of_ne hKd0 hKd,
    ← ENNReal.tsum_mul_left]
  /-
    case inr
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    K : NNReal
    d : Real
    hf : AntilipschitzWith K f
    hd : LE.le 0 d
    s : Set Y
    h0 : Ne K 0
    hKd0 : Ne (HPow.hPow (↑K) d) 0
    hKd : Ne (HPow.hPow (↑K) d) Top.top
    ⊢ LE.le (iSup fun r => iSup fun x => iInf fun t => iInf fun x => iInf fun x => …
  -/
  refine iSup₂_le fun ε ε0 => ?_
  /-
    case inr
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    K : NNReal
    d : Real
    hf : AntilipschitzWith K f
    hd : LE.le 0 d
    s : Set Y
    h0 : Ne K 0
    hKd0 : Ne (HPow.hPow (↑K) d) 0
    hKd : Ne (HPow.hPow (↑K) d) Top.top
    ε : ENNReal
    ε0 : LT.lt 0 ε
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => tsum fun n => iSup fun x => …
  -/
  refine le_iSup₂_of_le (ε / K) (by simp [ε0.ne']) ?_
  /-
    case inr
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    K : NNReal
    d : Real
    hf : AntilipschitzWith K f
    hd : LE.le 0 d
    s : Set Y
    h0 : Ne K 0
    hKd0 : Ne (HPow.hPow (↑K) d) 0
    hKd : Ne (HPow.hPow (↑K) d) Top.top
    ε : ENNReal
    ε0 : LT.lt 0 ε
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => tsum fun n => iSup fun x => …
  -/
  refine le_iInf₂ fun t hst => le_iInf fun htε => ?_
  /-
    case inr
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    K : NNReal
    d : Real
    hf : AntilipschitzWith K f
    hd : LE.le 0 d
    s : Set Y
    h0 : Ne K 0
    hKd0 : Ne (HPow.hPow (↑K) d) 0
    hKd : Ne (HPow.hPow (↑K) d) Top.top
    ε : ENNReal
    ε0 : LT.lt 0 ε
    t : Nat → Set Y
    hst : HasSubset.Subset s (Set.iUnion fun n => t n)
    htε : ∀ (n : Nat), LE.le (EMetric.diam (t n)) (HDiv.hDiv ε ↑K)
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => tsum fun n => iSup fun x => …
  -/
  replace hst : f ⁻¹' s ⊆ _ := preimage_mono hst; rw [preimage_iUnion] at hst
  /-
    case inr
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    K : NNReal
    d : Real
    hf : AntilipschitzWith K f
    hd : LE.le 0 d
    s : Set Y
    h0 : Ne K 0
    hKd0 : Ne (HPow.hPow (↑K) d) 0
    hKd : Ne (HPow.hPow (↑K) d) Top.top
    ε : ENNReal
    ε0 : LT.lt 0 ε
    t : Nat → Set Y
    htε : ∀ (n : Nat), LE.le (EMetric.diam (t n)) (HDiv.hDiv ε ↑K)
    hst : HasSubset.Subset (Set.preimage f s) (Set.iUnion fun i => Set.preimage f  …
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => tsum fun n => iSup fun x => …
  -/
  refine iInf₂_le_of_le _ hst (iInf_le_of_le (fun n => ?_) ?_)
    /-
      case inr.refine_1
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      f : X → Y
      K : NNReal
      d : Real
      hf : AntilipschitzWith K f
      hd : LE.le 0 d
      s : Set Y
      h0 : Ne K 0
      hKd0 : Ne (HPow.hPow (↑K) d) 0
      hKd : Ne (HPow.hPow (↑K) d) Top.top
      ε : ENNReal
      ε0 : LT.lt 0 ε
      t : Nat → Set Y
      htε : ∀ (n : Nat), LE.le (EMetric.diam (t n)) (HDiv.hDiv ε ↑K)
      hst : HasSubset.Subset (Set.preimage f s) (Set.iUnion fun i => Set.preimage f  …
      n : Nat
      ⊢ LE.le (EMetric.diam ((fun n => Set.preimage f (t n)) n)) ε
    -/
  · exact (hf.ediam_preimage_le _).trans (ENNReal.mul_le_of_le_div' <| htε n)
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      f : X → Y
      K : NNReal
      d : Real
      hf : AntilipschitzWith K f
      hd : LE.le 0 d
      s : Set Y
      h0 : Ne K 0
      hKd0 : Ne (HPow.hPow (↑K) d) 0
      hKd : Ne (HPow.hPow (↑K) d) Top.top
      ε : ENNReal
      ε0 : LT.lt 0 ε
      t : Nat → Set Y
      htε : ∀ (n : Nat), LE.le (EMetric.diam (t n)) (HDiv.hDiv ε ↑K)
      hst : HasSubset.Subset (Set.preimage f s) (Set.iUnion fun i => Set.preimage f  …
      ⊢ LE.le (tsum fun n => iSup fun x => HPow.hPow (EMetric.diam ((fun n => Set.pr …
    -/
  · refine ENNReal.tsum_le_tsum fun n => iSup_le_iff.2 fun hft => ?_
    /-
      case inr.refine_2
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      f : X → Y
      K : NNReal
      d : Real
      hf : AntilipschitzWith K f
      hd : LE.le 0 d
      s : Set Y
      h0 : Ne K 0
      hKd0 : Ne (HPow.hPow (↑K) d) 0
      hKd : Ne (HPow.hPow (↑K) d) Top.top
      ε : ENNReal
      ε0 : LT.lt 0 ε
      t : Nat → Set Y
      htε : ∀ (n : Nat), LE.le (EMetric.diam (t n)) (HDiv.hDiv ε ↑K)
      hst : HasSubset.Subset (Set.preimage f s) (Set.iUnion fun i => Set.preimage f  …
      n : Nat
      hft : ((fun n => Set.preimage f (t n)) n).Nonempty
      ⊢ LE.le (HPow.hPow (EMetric.diam ((fun n => Set.preimage f (t n)) n)) d) (iSup …
    -/
    simp only [nonempty_of_nonempty_preimage hft, ciSup_pos]
    /-
      case inr.refine_2
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      f : X → Y
      K : NNReal
      d : Real
      hf : AntilipschitzWith K f
      hd : LE.le 0 d
      s : Set Y
      h0 : Ne K 0
      hKd0 : Ne (HPow.hPow (↑K) d) 0
      hKd : Ne (HPow.hPow (↑K) d) Top.top
      ε : ENNReal
      ε0 : LT.lt 0 ε
      t : Nat → Set Y
      htε : ∀ (n : Nat), LE.le (EMetric.diam (t n)) (HDiv.hDiv ε ↑K)
      hst : HasSubset.Subset (Set.preimage f s) (Set.iUnion fun i => Set.preimage f  …
      n : Nat
      hft : ((fun n => Set.preimage f (t n)) n).Nonempty
      ⊢ LE.le (HPow.hPow (EMetric.diam (Set.preimage f (t n))) d) (HMul.hMul (HPow.h …
    -/
    rw [← ENNReal.mul_rpow_of_nonneg _ _ hd]
    /-
      case inr.refine_2
      X : Type u_2
      Y : Type u_3
      inst✝⁵ : EMetricSpace X
      inst✝⁴ : EMetricSpace Y
      inst✝³ : MeasurableSpace X
      inst✝² : BorelSpace X
      inst✝¹ : MeasurableSpace Y
      inst✝ : BorelSpace Y
      f : X → Y
      K : NNReal
      d : Real
      hf : AntilipschitzWith K f
      hd : LE.le 0 d
      s : Set Y
      h0 : Ne K 0
      hKd0 : Ne (HPow.hPow (↑K) d) 0
      hKd : Ne (HPow.hPow (↑K) d) Top.top
      ε : ENNReal
      ε0 : LT.lt 0 ε
      t : Nat → Set Y
      htε : ∀ (n : Nat), LE.le (EMetric.diam (t n)) (HDiv.hDiv ε ↑K)
      hst : HasSubset.Subset (Set.preimage f s) (Set.iUnion fun i => Set.preimage f  …
      n : Nat
      hft : ((fun n => Set.preimage f (t n)) n).Nonempty
      ⊢ LE.le (HPow.hPow (EMetric.diam (Set.preimage f (t n))) d) (HPow.hPow (HMul.h …
    -/
    exact ENNReal.rpow_le_rpow (hf.ediam_preimage_le _) hd
    /-
      🎉 no goals
    -/


theorem le_hausdorffMeasure_image (hf : AntilipschitzWith K f) (hd : 0 ≤ d) (s : Set X) :
    μH[d] s ≤ (K : ℝ≥0∞) ^ d * μH[d] (f '' s) :=
  calc
    μH[d] s ≤ μH[d] (f ⁻¹' (f '' s)) := measure_mono (subset_preimage_image _ _)
    _ ≤ (K : ℝ≥0∞) ^ d * μH[d] (f '' s) := hf.hausdorffMeasure_preimage_le hd (f '' s)


theorem hausdorffMeasure_image (hf : Isometry f) (hd : 0 ≤ d ∨ Surjective f) (s : Set X) :
    μH[d] (f '' s) = μH[d] s := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    d : Real
    hf : Isometry f
    hd : Or (LE.le 0 d) (Function.Surjective f)
    s : Set X
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image f s)) ((MeasureThe …
  -/
  simp only [hausdorffMeasure, ← OuterMeasure.coe_mkMetric, ← OuterMeasure.comap_apply]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    d : Real
    hf : Isometry f
    hd : Or (LE.le 0 d) (Function.Surjective f)
    s : Set X
    ⊢ Eq (((MeasureTheory.OuterMeasure.comap f) (MeasureTheory.OuterMeasure.mkMetr …
  -/
  rw [OuterMeasure.isometry_comap_mkMetric _ hf (hd.imp_left _)]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    d : Real
    hf : Isometry f
    hd : Or (LE.le 0 d) (Function.Surjective f)
    s : Set X
    ⊢ LE.le 0 d → Monotone fun r => HPow.hPow r d
  -/
  exact ENNReal.monotone_rpow_of_nonneg
  /-
    🎉 no goals
  -/


theorem hausdorffMeasure_preimage (hf : Isometry f) (hd : 0 ≤ d ∨ Surjective f) (s : Set Y) :
    μH[d] (f ⁻¹' s) = μH[d] (s ∩ range f) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    d : Real
    hf : Isometry f
    hd : Or (LE.le 0 d) (Function.Surjective f)
    s : Set Y
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage f s)) ((Measure …
  -/
  rw [← hf.hausdorffMeasure_image hd, image_preimage_eq_inter_range]
  /-
    🎉 no goals
  -/


theorem map_hausdorffMeasure (hf : Isometry f) (hd : 0 ≤ d ∨ Surjective f) :
    Measure.map f μH[d] = μH[d].restrict (range f) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    f : X → Y
    d : Real
    hf : Isometry f
    hd : Or (LE.le 0 d) (Function.Surjective f)
    ⊢ Eq (MeasureTheory.Measure.map f (MeasureTheory.Measure.hausdorffMeasure d))  …
  -/
  ext1 s hs
  rw [map_apply hf.continuous.measurable hs, Measure.restrict_apply hs,
    hf.hausdorffMeasure_preimage hd]


@[simp]
theorem hausdorffMeasure_image (e : X ≃ᵢ Y) (d : ℝ) (s : Set X) : μH[d] (e '' s) = μH[d] s :=
  e.isometry.hausdorffMeasure_image (Or.inr e.surjective) s


@[simp]
theorem hausdorffMeasure_preimage (e : X ≃ᵢ Y) (d : ℝ) (s : Set Y) : μH[d] (e ⁻¹' s) = μH[d] s := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    e : IsometryEquiv X Y
    d : Real
    s : Set Y
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage (⇑e) s)) ((Meas …
  -/
  rw [← e.image_symm, e.symm.hausdorffMeasure_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_hausdorffMeasure (e : X ≃ᵢ Y) (d : ℝ) : Measure.map e μH[d] = μH[d] := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁵ : EMetricSpace X
    inst✝⁴ : EMetricSpace Y
    inst✝³ : MeasurableSpace X
    inst✝² : BorelSpace X
    inst✝¹ : MeasurableSpace Y
    inst✝ : BorelSpace Y
    e : IsometryEquiv X Y
    d : Real
    ⊢ Eq (MeasureTheory.Measure.map (⇑e) (MeasureTheory.Measure.hausdorffMeasure d …
  -/
  rw [e.isometry.map_hausdorffMeasure (Or.inr e.surjective), e.surjective.range_eq, restrict_univ]
  /-
    🎉 no goals
  -/


theorem measurePreserving_hausdorffMeasure (e : X ≃ᵢ Y) (d : ℝ) : MeasurePreserving e μH[d] μH[d] :=
  ⟨e.continuous.measurable, map_hausdorffMeasure _ _⟩


@[to_additive]
theorem hausdorffMeasure_smul {α : Type*} [SMul α X] [IsometricSMul α X] {d : ℝ} (c : α)
    (h : 0 ≤ d ∨ Surjective (c • · : X → X)) (s : Set X) : μH[d] (c • s) = μH[d] s :=
  (isometry_smul X c).hausdorffMeasure_image h _


@[to_additive]
instance {d : ℝ} [Group X] [IsometricSMul X X] : IsMulLeftInvariant (μH[d] : Measure X) where
  map_mul_left_eq_self x := (IsometryEquiv.constSMul x).map_hausdorffMeasure _


@[to_additive]
instance {d : ℝ} [Group X] [IsometricSMul Xᵐᵒᵖ X] : IsMulRightInvariant (μH[d] : Measure X) where
  map_mul_right_eq_self x := (IsometryEquiv.constSMul (MulOpposite.op x)).map_hausdorffMeasure _


/-- In the space `ι → ℝ`, the Hausdorff measure coincides exactly with the Lebesgue measure. -/
@[simp]
theorem hausdorffMeasure_pi_real {ι : Type*} [Fintype ι] :
    (μH[Fintype.card ι] : Measure (ι → ℝ)) = volume := by
  classical
  -- it suffices to check that the two measures coincide on products of rational intervals
  refine (pi_eq_generateFrom (fun _ => Real.borel_eq_generateFrom_Ioo_rat.symm)
    (fun _ => Real.isPiSystem_Ioo_rat) (fun _ => Real.finiteSpanningSetsInIooRat _) ?_).symm
  simp only [mem_iUnion, mem_singleton_iff]
  -- fix such a product `s` of rational intervals, of the form `Π (a i, b i)`.
  intro s hs
  choose a b H using hs
  obtain rfl : s = fun i => Ioo (α := ℝ) (a i) (b i) := funext fun i => (H i).2
  replace H := fun i => (H i).1
  apply le_antisymm _
  -- first check that `volume s ≤ μH s`
  · have Hle : volume ≤ (μH[Fintype.card ι] : Measure (ι → ℝ)) := by
      refine le_hausdorffMeasure _ _ ∞ ENNReal.coe_lt_top fun s _ => ?_
      rw [ENNReal.rpow_natCast]
      exact Real.volume_pi_le_diam_pow s
    rw [← volume_pi_pi fun i => Ioo (a i : ℝ) (b i)]
    exact Measure.le_iff'.1 Hle _
  /- For the other inequality `μH s ≤ volume s`, we use a covering of `s` by sets of small diameter
    `1/n`, namely cubes with left-most point of the form `a i + f i / n` with `f i` ranging between
    `0` and `⌈(b i - a i) * n⌉`. Their number is asymptotic to `n^d * Π (b i - a i)`. -/
  have I : ∀ i, 0 ≤ (b i : ℝ) - a i := fun i => by
    simpa only [sub_nonneg, Rat.cast_le] using (H i).le
  let γ := fun n : ℕ => ∀ i : ι, Fin ⌈((b i : ℝ) - a i) * n⌉₊
  let t : ∀ n : ℕ, γ n → Set (ι → ℝ) := fun n f =>
    Set.pi univ fun i => Icc (a i + f i / n) (a i + (f i + 1) / n)
  have A : Tendsto (fun n : ℕ => 1 / (n : ℝ≥0∞)) atTop (𝓝 0) := by
    simp only [one_div, ENNReal.tendsto_inv_nat_nhds_zero]
  have B : ∀ᶠ n in atTop, ∀ i : γ n, diam (t n i) ≤ 1 / n := by
    refine eventually_atTop.2 ⟨1, fun n hn => ?_⟩
    intro f
    refine diam_pi_le_of_le fun b => ?_
    simp only [Real.ediam_Icc, add_div, ENNReal.ofReal_div_of_pos (Nat.cast_pos.mpr hn), le_refl,
      add_sub_add_left_eq_sub, add_sub_cancel_left, ENNReal.ofReal_one, ENNReal.ofReal_natCast]
  have C : ∀ᶠ n in atTop, (Set.pi univ fun i : ι => Ioo (a i : ℝ) (b i)) ⊆ ⋃ i : γ n, t n i := by
    refine eventually_atTop.2 ⟨1, fun n hn => ?_⟩
    have npos : (0 : ℝ) < n := Nat.cast_pos.2 hn
    intro x hx
    simp only [mem_Ioo, mem_univ_pi] at hx
    simp only [t, mem_iUnion, mem_Ioo, mem_univ_pi]
    let f : γ n := fun i =>
      ⟨⌊(x i - a i) * n⌋₊, by
        apply Nat.floor_lt_ceil_of_lt_of_pos
        · refine (mul_lt_mul_right npos).2 ?_
          simp only [(hx i).right, sub_lt_sub_iff_right]
        · refine mul_pos ?_ npos
          simpa only [Rat.cast_lt, sub_pos] using H i⟩
    refine ⟨f, fun i => ⟨?_, ?_⟩⟩
    · calc
        (a i : ℝ) + ⌊(x i - a i) * n⌋₊ / n ≤ (a i : ℝ) + (x i - a i) * n / n := by
          gcongr
          exact Nat.floor_le (mul_nonneg (sub_nonneg.2 (hx i).1.le) npos.le)
        _ = x i := by field_simp [npos.ne']
    · calc
        x i = (a i : ℝ) + (x i - a i) * n / n := by field_simp [npos.ne']
        _ ≤ (a i : ℝ) + (⌊(x i - a i) * n⌋₊ + 1) / n := by
          gcongr
          exact (Nat.lt_floor_add_one _).le
  calc
    μH[Fintype.card ι] (Set.pi univ fun i : ι => Ioo (a i : ℝ) (b i)) ≤
        liminf (fun n : ℕ => ∑ i : γ n, diam (t n i) ^ ((Fintype.card ι) : ℝ)) atTop :=
      hausdorffMeasure_le_liminf_sum _ (Set.pi univ fun i => Ioo (a i : ℝ) (b i))
        (fun n : ℕ => 1 / (n : ℝ≥0∞)) A t B C
    _ ≤ liminf (fun n : ℕ => ∑ i : γ n, (1 / (n : ℝ≥0∞)) ^ Fintype.card ι) atTop := by
      refine liminf_le_liminf ?_ ?_
      · filter_upwards [B] with _ hn
        apply Finset.sum_le_sum fun i _ => _
        simp only [ENNReal.rpow_natCast]
        intros i _
        exact pow_le_pow_left' (hn i) _
      · isBoundedDefault
    _ = liminf (fun n : ℕ => ∏ i : ι, (⌈((b i : ℝ) - a i) * n⌉₊ : ℝ≥0∞) / n) atTop := by
      simp only [γ, Finset.card_univ, Nat.cast_prod, one_mul, Fintype.card_fin, Finset.sum_const,
        nsmul_eq_mul, Fintype.card_pi, div_eq_mul_inv, Finset.prod_mul_distrib, Finset.prod_const]
    _ = ∏ i : ι, volume (Ioo (a i : ℝ) (b i)) := by
      simp only [Real.volume_Ioo]
      apply Tendsto.liminf_eq
      refine ENNReal.tendsto_finset_prod_of_ne_top _ (fun i _ => ?_) fun i _ => ?_
      · apply
          Tendsto.congr' _
            ((ENNReal.continuous_ofReal.tendsto _).comp
              ((tendsto_nat_ceil_mul_div_atTop (I i)).comp tendsto_natCast_atTop_atTop))
        apply eventually_atTop.2 ⟨1, fun n hn => _⟩
        intros n hn
        simp only [ENNReal.ofReal_div_of_pos (Nat.cast_pos.mpr hn), comp_apply,
          ENNReal.ofReal_natCast]
      · simp only [ENNReal.ofReal_ne_top, Ne, not_false_iff]


instance isAddHaarMeasure_hausdorffMeasure {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    [MeasurableSpace E] [BorelSpace E] :
    IsAddHaarMeasure (G := E) μH[finrank ℝ E] where
  lt_top_of_isCompact K hK := by
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      K : Set E
      hK : IsCompact K
      ⊢ LT.lt ((MeasureTheory.Measure.hausdorffMeasure ↑(Module.finrank Real E)) K)  …
    -/
    set e : E ≃L[ℝ] Fin (finrank ℝ E) → ℝ := ContinuousLinearEquiv.ofFinrankEq (by simp)
    suffices μH[finrank ℝ E] (e '' K) < ⊤ by
      rw [← e.symm_image_image K]
      apply lt_of_le_of_lt <| e.symm.lipschitz.hausdorffMeasure_image_le (by simp) (e '' K)
      rw [ENNReal.rpow_natCast]
      exact ENNReal.mul_lt_top (ENNReal.pow_lt_top ENNReal.coe_lt_top _) this
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      K : Set E
      hK : IsCompact K
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ LT.lt ((MeasureTheory.Measure.hausdorffMeasure ↑(Module.finrank Real E)) (Se …
    -/
    conv_lhs => congr; congr; rw [← Fintype.card_fin (finrank ℝ E)]
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      K : Set E
      hK : IsCompact K
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ LT.lt ((MeasureTheory.Measure.hausdorffMeasure ↑(Fintype.card (Fin (Module.f …
    -/
    rw [hausdorffMeasure_pi_real]
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      K : Set E
      hK : IsCompact K
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ LT.lt (MeasureTheory.MeasureSpace.volume (Set.image (⇑e) K)) Top.top
    -/
    exact (hK.image e.continuous).measure_lt_top
    /-
      🎉 no goals
    -/
  open_pos U hU hU' := by
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      U : Set E
      hU : IsOpen U
      hU' : U.Nonempty
      ⊢ Ne ((MeasureTheory.Measure.hausdorffMeasure ↑(Module.finrank Real E)) U) 0
    -/
    set e : E ≃L[ℝ] Fin (finrank ℝ E) → ℝ := ContinuousLinearEquiv.ofFinrankEq (by simp)
    suffices 0 < μH[finrank ℝ E] (e '' U) from
      (ENNReal.mul_pos_iff.mp (lt_of_lt_of_le this <|
        e.lipschitz.hausdorffMeasure_image_le (by simp) _)).2.ne'
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      U : Set E
      hU : IsOpen U
      hU' : U.Nonempty
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ LT.lt 0 ((MeasureTheory.Measure.hausdorffMeasure ↑(Module.finrank Real E)) ( …
    -/
    conv_rhs => congr; congr; rw [← Fintype.card_fin (finrank ℝ E)]
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      U : Set E
      hU : IsOpen U
      hU' : U.Nonempty
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ LT.lt 0 ((MeasureTheory.Measure.hausdorffMeasure ↑(Fintype.card (Fin (Module …
    -/
    rw [hausdorffMeasure_pi_real]
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      U : Set E
      hU : IsOpen U
      hU' : U.Nonempty
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ LT.lt 0 (MeasureTheory.MeasureSpace.volume (Set.image (⇑e) U))
    -/
    apply (e.isOpenMap U hU).measure_pos (μ := volume)
    /-
      ι : Type u_1
      X : Type u_2
      Y : Type u_3
      inst✝¹⁰ : EMetricSpace X
      inst✝⁹ : EMetricSpace Y
      inst✝⁸ : MeasurableSpace X
      inst✝⁷ : BorelSpace X
      inst✝⁶ : MeasurableSpace Y
      inst✝⁵ : BorelSpace Y
      E : Type u_4
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      U : Set E
      hU : IsOpen U
      hU' : U.Nonempty
      e : ContinuousLinearEquiv (RingHom.id Real) E (Fin (Module.finrank Real E) → R …
      ⊢ (Set.image (⇑e) U).Nonempty
    -/
    simpa
    /-
      🎉 no goals
    -/


theorem hausdorffMeasure_measurePreserving_funUnique [Unique ι]
    [SecondCountableTopology X] (d : ℝ) :
    MeasurePreserving (MeasurableEquiv.funUnique ι X) μH[d] μH[d] :=
  (IsometryEquiv.funUnique ι X).measurePreserving_hausdorffMeasure _


theorem hausdorffMeasure_measurePreserving_piFinTwo (α : Fin 2 → Type*)
    [∀ i, MeasurableSpace (α i)] [∀ i, EMetricSpace (α i)] [∀ i, BorelSpace (α i)]
    [∀ i, SecondCountableTopology (α i)] (d : ℝ) :
    MeasurePreserving (MeasurableEquiv.piFinTwo α) μH[d] μH[d] :=
  (IsometryEquiv.piFinTwo α).measurePreserving_hausdorffMeasure _


/-- In the space `ℝ`, the Hausdorff measure coincides exactly with the Lebesgue measure. -/
@[simp]
theorem hausdorffMeasure_real : (μH[1] : Measure ℝ) = volume := by
  rw [← (volume_preserving_funUnique Unit ℝ).map_eq,
    ← (hausdorffMeasure_measurePreserving_funUnique Unit ℝ 1).map_eq,
    ← hausdorffMeasure_pi_real, Fintype.card_unit, Nat.cast_one]


/-- In the space `ℝ × ℝ`, the Hausdorff measure coincides exactly with the Lebesgue measure. -/
@[simp]
theorem hausdorffMeasure_prod_real : (μH[2] : Measure (ℝ × ℝ)) = volume := by
  rw [← (volume_preserving_piFinTwo fun _ => ℝ).map_eq,
    ← (hausdorffMeasure_measurePreserving_piFinTwo (fun _ => ℝ) _).map_eq,
    ← hausdorffMeasure_pi_real, Fintype.card_fin, Nat.cast_two]


theorem hausdorffMeasure_smul_right_image [NormedAddCommGroup E] [NormedSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] (v : E) (s : Set ℝ) :
    μH[1] ((fun r => r • v) '' s) = ‖v‖₊ • μH[1] s := by
  /-
    E : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    v : E
    s : Set Real
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 1) (Set.image (fun r => HSMul.hS …
  -/
  obtain rfl | hv := eq_or_ne v 0
    /-
      case inl
      E : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set Real
      ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 1) (Set.image (fun r => HSMul.hS …
    -/
  · haveI := noAtoms_hausdorff E one_pos
    /-
      case inl
      E : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set Real
      this : MeasureTheory.NoAtoms (MeasureTheory.Measure.hausdorffMeasure 1)
      ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 1) (Set.image (fun r => HSMul.hS …
    -/
    obtain rfl | hs := s.eq_empty_or_nonempty
      /-
        case inl.inl
        E : Type u_5
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace Real E
        inst✝¹ : MeasurableSpace E
        inst✝ : BorelSpace E
        this : MeasureTheory.NoAtoms (MeasureTheory.Measure.hausdorffMeasure 1)
        ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 1) (Set.image (fun r => HSMul.hS …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      E : Type u_5
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set Real
      this : MeasureTheory.NoAtoms (MeasureTheory.Measure.hausdorffMeasure 1)
      hs : s.Nonempty
      ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 1) (Set.image (fun r => HSMul.hS …
    -/
    simp [hs]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    v : E
    s : Set Real
    hv : Ne v 0
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 1) (Set.image (fun r => HSMul.hS …
  -/
  have hn : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  -- break lineMap into pieces
  suffices
      μH[1] ((‖v‖ • ·) '' (LinearMap.toSpanSingleton ℝ E (‖v‖⁻¹ • v) '' s)) = ‖v‖₊ • μH[1] s by
    simpa only [Set.image_image, smul_comm (norm _), inv_smul_smul₀ hn,
      LinearMap.toSpanSingleton_apply] using this
  have iso_smul : Isometry (LinearMap.toSpanSingleton ℝ E (‖v‖⁻¹ • v)) := by
    refine AddMonoidHomClass.isometry_of_norm _ fun x => (norm_smul _ _).trans ?_
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn, mul_one, LinearMap.id_apply]
  rw [Set.image_smul, Measure.hausdorffMeasure_smul₀ zero_le_one hn, nnnorm_norm,
      NNReal.rpow_one, iso_smul.hausdorffMeasure_image (Or.inl <| zero_le_one' ℝ)]


/-- Scaling by `c` around `x` scales the measure by `‖c‖₊ ^ d`. -/
theorem hausdorffMeasure_homothety_image {d : ℝ} (hd : 0 ≤ d) (x : P) {c : 𝕜} (hc : c ≠ 0)
    (s : Set P) : μH[d] (AffineMap.homothety x c '' s) = ‖c‖₊ ^ d • μH[d] s := by
  suffices
    μH[d] (IsometryEquiv.vaddConst x '' ((c • ·) '' ((IsometryEquiv.vaddConst x).symm '' s))) =
      ‖c‖₊ ^ d • μH[d] s by
    simpa only [Set.image_image]
  /-
    𝕜 : Type u_4
    E : Type u_5
    P : Type u_6
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : MeasurableSpace P
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor E P
    inst✝ : BorelSpace P
    d : Real
    hd : LE.le 0 d
    x : P
    c : 𝕜
    hc : Ne c 0
    s : Set P
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure d) (Set.image (⇑(IsometryEquiv.v …
  -/
  borelize E
  rw [IsometryEquiv.hausdorffMeasure_image, Set.image_smul, Measure.hausdorffMeasure_smul₀ hd hc,
    IsometryEquiv.hausdorffMeasure_image]


theorem hausdorffMeasure_homothety_preimage {d : ℝ} (hd : 0 ≤ d) (x : P) {c : 𝕜} (hc : c ≠ 0)
    (s : Set P) : μH[d] (AffineMap.homothety x c ⁻¹' s) = ‖c‖₊⁻¹ ^ d • μH[d] s := by
  /-
    𝕜 : Type u_4
    E : Type u_5
    P : Type u_6
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : MeasurableSpace P
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor E P
    inst✝ : BorelSpace P
    d : Real
    hd : LE.le 0 d
    x : P
    c : 𝕜
    hc : Ne c 0
    s : Set P
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure d) (Set.preimage (⇑(AffineMap.ho …
  -/
  change μH[d] (AffineEquiv.homothetyUnitsMulHom x (Units.mk0 c hc) ⁻¹' s) = _
  rw [← AffineEquiv.image_symm, AffineEquiv.coe_homothetyUnitsMulHom_apply_symm,
    hausdorffMeasure_homothety_image hd x (_ : 𝕜ˣ).isUnit.ne_zero, Units.val_inv_eq_inv_val,
    Units.val_mk0, nnnorm_inv]


/-- Mapping a set of reals along a line segment scales the measure by the length of a segment.

This is an auxiliary result used to prove `hausdorffMeasure_affineSegment`. -/
theorem hausdorffMeasure_lineMap_image (x y : P) (s : Set ℝ) :
    μH[1] (AffineMap.lineMap x y '' s) = nndist x y • μH[1] s := by
  suffices μH[1] (IsometryEquiv.vaddConst x '' ((· • (y -ᵥ x)) '' s)) = nndist x y • μH[1] s by
    simpa only [Set.image_image]
  /-
    E : Type u_5
    P : Type u_6
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace P
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor E P
    inst✝ : BorelSpace P
    x y : P
    s : Set Real
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 1) (Set.image (⇑(IsometryEquiv.v …
  -/
  borelize E
  rw [IsometryEquiv.hausdorffMeasure_image, hausdorffMeasure_smul_right_image,
    nndist_eq_nnnorm_vsub' E]


/-- The measure of a segment is the distance between its endpoints. -/
@[simp]
theorem hausdorffMeasure_affineSegment (x y : P) : μH[1] (affineSegment ℝ x y) = edist x y := by
  rw [affineSegment, hausdorffMeasure_lineMap_image, hausdorffMeasure_real, Real.volume_Icc,
    sub_zero, ENNReal.ofReal_one, ← Algebra.algebraMap_eq_smul_one]
  /-
    E : Type u_5
    P : Type u_6
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace P
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor E P
    inst✝ : BorelSpace P
    x y : P
    ⊢ Eq ((algebraMap NNReal ENNReal) (NNDist.nndist x y)) (EDist.edist x y)
  -/
  exact (edist_nndist _ _).symm
  /-
    🎉 no goals
  -/


/-- The measure of a segment is the distance between its endpoints. -/
@[simp]
theorem hausdorffMeasure_segment {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] (x y : E) : μH[1] (segment ℝ x y) = edist x y := by
  /-
    E : Type u_7
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    x y : E
    ⊢ Eq ((MeasureTheory.Measure.hausdorffMeasure 1) (segment Real x y)) (EDist.ed …
  -/
  rw [← affineSegment_eq_segment, hausdorffMeasure_affineSegment]
  /-
    🎉 no goals
  -/


