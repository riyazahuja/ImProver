/-- Given a sequence of functions `f` and a function `g`, `notConvergentSeq f g n j` is the
set of elements such that `f k x` and `g x` are separated by at least `1 / (n + 1)` for some
`k ≥ j`.

This definition is useful for Egorov's theorem. -/
def notConvergentSeq [Preorder ι] (f : ι → α → β) (g : α → β) (n : ℕ) (j : ι) : Set α :=
  ⋃ (k) (_ : j ≤ k), { x | 1 / (n + 1 : ℝ) < dist (f k x) (g x) }


theorem mem_notConvergentSeq_iff [Preorder ι] {x : α} :
    x ∈ notConvergentSeq f g n j ↔ ∃ k ≥ j, 1 / (n + 1 : ℝ) < dist (f k x) (g x) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝¹ : MetricSpace β
    n : Nat
    j : ι
    f : ι → α → β
    g : α → β
    inst✝ : Preorder ι
    x : α
    ⊢ Iff (Membership.mem (MeasureTheory.Egorov.notConvergentSeq f g n j) x) (Exis …
  -/
  simp_rw [notConvergentSeq, Set.mem_iUnion, exists_prop, mem_setOf]
  /-
    🎉 no goals
  -/


theorem notConvergentSeq_antitone [Preorder ι] : Antitone (notConvergentSeq f g n) :=
  fun _ _ hjk => Set.iUnion₂_mono' fun l hl => ⟨l, le_trans hjk hl, Set.Subset.rfl⟩


theorem measure_inter_notConvergentSeq_eq_zero [SemilatticeSup ι] [Nonempty ι]
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) (n : ℕ) :
    μ (s ∩ ⋂ j, notConvergentSeq f g n j) = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    n : Nat
    ⊢ Eq (μ (Inter.inter s (Set.iInter fun j => MeasureTheory.Egorov.notConvergent …
  -/
  simp_rw [Metric.tendsto_atTop, ae_iff] at hfg
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    n : Nat
    hfg : Eq (μ (setOf fun a => Not (Membership.mem s a → ∀ (ε : Real), GT.gt ε 0  …
    ⊢ Eq (μ (Inter.inter s (Set.iInter fun j => MeasureTheory.Egorov.notConvergent …
  -/
  rw [← nonpos_iff_eq_zero, ← hfg]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    n : Nat
    hfg : Eq (μ (setOf fun a => Not (Membership.mem s a → ∀ (ε : Real), GT.gt ε 0  …
    ⊢ LE.le (μ (Inter.inter s (Set.iInter fun j => MeasureTheory.Egorov.notConverg …
  -/
  refine measure_mono fun x => ?_
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    n : Nat
    hfg : Eq (μ (setOf fun a => Not (Membership.mem s a → ∀ (ε : Real), GT.gt ε 0  …
    x : α
    ⊢ Membership.mem (Inter.inter s (Set.iInter fun j => MeasureTheory.Egorov.notC …
  -/
  simp only [Set.mem_inter_iff, Set.mem_iInter, mem_notConvergentSeq_iff]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    n : Nat
    hfg : Eq (μ (setOf fun a => Not (Membership.mem s a → ∀ (ε : Real), GT.gt ε 0  …
    x : α
    ⊢ And (Membership.mem s x) (∀ (i : ι), Exists fun k => And (GE.ge k i) (LT.lt  …
  -/
  push_neg
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    n : Nat
    hfg : Eq (μ (setOf fun a => Not (Membership.mem s a → ∀ (ε : Real), GT.gt ε 0  …
    x : α
    ⊢ And (Membership.mem s x) (∀ (i : ι), Exists fun k => And (GE.ge k i) (LT.lt  …
  -/
  rintro ⟨hmem, hx⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    n : Nat
    hfg : Eq (μ (setOf fun a => Not (Membership.mem s a → ∀ (ε : Real), GT.gt ε 0  …
    x : α
    hmem : Membership.mem s x
    hx : ∀ (i : ι), Exists fun k => And (GE.ge k i) (LT.lt (HDiv.hDiv 1 (HAdd.hAdd …
    ⊢ Membership.mem (setOf fun a => And (Membership.mem s a) (Exists fun ε => And …
  -/
  refine ⟨hmem, 1 / (n + 1 : ℝ), Nat.one_div_pos_of_nat, fun N => ?_⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    n : Nat
    hfg : Eq (μ (setOf fun a => Not (Membership.mem s a → ∀ (ε : Real), GT.gt ε 0  …
    x : α
    hmem : Membership.mem s x
    hx : ∀ (i : ι), Exists fun k => And (GE.ge k i) (LT.lt (HDiv.hDiv 1 (HAdd.hAdd …
    N : ι
    ⊢ Exists fun n_1 => And (GE.ge n_1 N) (LE.le (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))  …
  -/
  obtain ⟨n, hn₁, hn₂⟩ := hx N
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Nonempty ι
    n✝ : Nat
    hfg : Eq (μ (setOf fun a => Not (Membership.mem s a → ∀ (ε : Real), GT.gt ε 0  …
    x : α
    hmem : Membership.mem s x
    hx : ∀ (i : ι), Exists fun k => And (GE.ge k i) (LT.lt (HDiv.hDiv 1 (HAdd.hAdd …
    N n : ι
    hn₁ : GE.ge n N
    hn₂ : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑n✝) 1)) (Dist.dist (f n x) (g x))
    ⊢ Exists fun n => And (GE.ge n N) (LE.le (HDiv.hDiv 1 (HAdd.hAdd (↑n✝) 1)) (Di …
  -/
  exact ⟨n, hn₁, hn₂.le⟩
  /-
    🎉 no goals
  -/


theorem notConvergentSeq_measurableSet [Preorder ι] [Countable ι]
    (hf : ∀ n, StronglyMeasurable[m] (f n)) (hg : StronglyMeasurable g) :
    MeasurableSet (notConvergentSeq f g n j) :=
  MeasurableSet.iUnion fun k =>
    MeasurableSet.iUnion fun _ =>
      StronglyMeasurable.measurableSet_lt stronglyMeasurable_const <| (hf k).dist hg


theorem measure_notConvergentSeq_tendsto_zero [SemilatticeSup ι] [Countable ι]
    (hf : ∀ n, StronglyMeasurable (f n)) (hg : StronglyMeasurable g) (hsm : MeasurableSet s)
    (hs : μ s ≠ ∞) (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) (n : ℕ) :
    Tendsto (fun j => μ (s ∩ notConvergentSeq f g n j)) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Countable ι
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    n : Nat
    ⊢ Filter.Tendsto (fun j => μ (Inter.inter s (MeasureTheory.Egorov.notConvergen …
  -/
  cases' isEmpty_or_nonempty ι with h h
  · have : (fun j => μ (s ∩ notConvergentSeq f g n j)) = fun j => 0 := by
      simp only [eq_iff_true_of_subsingleton]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      inst✝² : MetricSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      f : ι → α → β
      g : α → β
      inst✝¹ : SemilatticeSup ι
      inst✝ : Countable ι
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hsm : MeasurableSet s
      hs : Ne (μ s) Top.top
      hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
      n : Nat
      h : IsEmpty ι
      this : Eq (fun j => μ (Inter.inter s (MeasureTheory.Egorov.notConvergentSeq f  …
      ⊢ Filter.Tendsto (fun j => μ (Inter.inter s (MeasureTheory.Egorov.notConvergen …
    -/
    rw [this]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      inst✝² : MetricSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      f : ι → α → β
      g : α → β
      inst✝¹ : SemilatticeSup ι
      inst✝ : Countable ι
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hsm : MeasurableSet s
      hs : Ne (μ s) Top.top
      hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
      n : Nat
      h : IsEmpty ι
      this : Eq (fun j => μ (Inter.inter s (MeasureTheory.Egorov.notConvergentSeq f  …
      ⊢ Filter.Tendsto (fun j => 0) Filter.atTop (nhds 0)
    -/
    exact tendsto_const_nhds
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝² : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    f : ι → α → β
    g : α → β
    inst✝¹ : SemilatticeSup ι
    inst✝ : Countable ι
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    n : Nat
    h : Nonempty ι
    ⊢ Filter.Tendsto (fun j => μ (Inter.inter s (MeasureTheory.Egorov.notConvergen …
  -/
  rw [← measure_inter_notConvergentSeq_eq_zero hfg n, Set.inter_iInter]
  refine tendsto_measure_iInter_atTop
    (fun n ↦ (hsm.inter <| notConvergentSeq_measurableSet hf hg).nullMeasurableSet)
    (fun k l hkl => Set.inter_subset_inter_right _ <| notConvergentSeq_antitone hkl)
    ⟨h.some, ne_top_of_le_ne_top hs (measure_mono Set.inter_subset_left)⟩


theorem exists_notConvergentSeq_lt (hε : 0 < ε) (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hsm : MeasurableSet s) (hs : μ s ≠ ∞)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) (n : ℕ) :
    ∃ j : ι, μ (s ∩ notConvergentSeq f g n j) ≤ ENNReal.ofReal (ε * 2⁻¹ ^ n) := by
  have ⟨N, hN⟩ := (ENNReal.tendsto_atTop ENNReal.zero_ne_top).1
    (measure_notConvergentSeq_tendsto_zero hf hg hsm hs hfg n) (ENNReal.ofReal (ε * 2⁻¹ ^ n)) (by
      rw [gt_iff_lt, ENNReal.ofReal_pos]
      exact mul_pos hε (pow_pos (by norm_num) n))
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    n : Nat
    N : ι
    hN : ∀ (n_1 : ι), GE.ge n_1 N → Membership.mem (Set.Icc (HSub.hSub 0 (ENNReal. …
    ⊢ Exists fun j => LE.le (μ (Inter.inter s (MeasureTheory.Egorov.notConvergentS …
  -/
  rw [zero_add] at hN
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    n : Nat
    N : ι
    hN : ∀ (n_1 : ι), GE.ge n_1 N → Membership.mem (Set.Icc (HSub.hSub 0 (ENNReal. …
    ⊢ Exists fun j => LE.le (μ (Inter.inter s (MeasureTheory.Egorov.notConvergentS …
  -/
  exact ⟨N, (hN N le_rfl).2⟩
  /-
    🎉 no goals
  -/


/-- Given some `ε > 0`, `notConvergentSeqLTIndex` provides the index such that
`notConvergentSeq` (intersected with a set of finite measure) has measure less than
`ε * 2⁻¹ ^ n`.

This definition is useful for Egorov's theorem. -/
def notConvergentSeqLTIndex (hε : 0 < ε) (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hsm : MeasurableSet s) (hs : μ s ≠ ∞)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) (n : ℕ) : ι :=
  Classical.choose <| exists_notConvergentSeq_lt hε hf hg hsm hs hfg n


theorem notConvergentSeqLTIndex_spec (hε : 0 < ε) (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hsm : MeasurableSet s) (hs : μ s ≠ ∞)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) (n : ℕ) :
    μ (s ∩ notConvergentSeq f g n (notConvergentSeqLTIndex hε hf hg hsm hs hfg n)) ≤
      ENNReal.ofReal (ε * 2⁻¹ ^ n) :=
  Classical.choose_spec <| exists_notConvergentSeq_lt hε hf hg hsm hs hfg n


/-- Given some `ε > 0`, `iUnionNotConvergentSeq` is the union of `notConvergentSeq` with
specific indices such that `iUnionNotConvergentSeq` has measure less equal than `ε`.

This definition is useful for Egorov's theorem. -/
def iUnionNotConvergentSeq (hε : 0 < ε) (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hsm : MeasurableSet s) (hs : μ s ≠ ∞)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) : Set α :=
  ⋃ n, s ∩ notConvergentSeq f g n (notConvergentSeqLTIndex (half_pos hε) hf hg hsm hs hfg n)


theorem iUnionNotConvergentSeq_measurableSet (hε : 0 < ε) (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hsm : MeasurableSet s) (hs : μ s ≠ ∞)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    MeasurableSet <| iUnionNotConvergentSeq hε hf hg hsm hs hfg :=
  MeasurableSet.iUnion fun _ => hsm.inter <| notConvergentSeq_measurableSet hf hg


theorem measure_iUnionNotConvergentSeq (hε : 0 < ε) (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hsm : MeasurableSet s) (hs : μ s ≠ ∞)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    μ (iUnionNotConvergentSeq hε hf hg hsm hs hfg) ≤ ENNReal.ofReal ε := by
  refine le_trans (measure_iUnion_le _) (le_trans
    (ENNReal.tsum_le_tsum <| notConvergentSeqLTIndex_spec (half_pos hε) hf hg hsm hs hfg) ?_)
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    ⊢ LE.le (tsum fun a => ENNReal.ofReal (HMul.hMul (HDiv.hDiv ε 2) (HPow.hPow (I …
  -/
  simp_rw [ENNReal.ofReal_mul (half_pos hε).le]
  rw [ENNReal.tsum_mul_left, ← ENNReal.ofReal_tsum_of_nonneg, inv_eq_one_div, tsum_geometric_two,
    ← ENNReal.ofReal_mul (half_pos hε).le, div_mul_cancel₀ ε two_ne_zero]
    /-
      case hf_nonneg
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      inst✝³ : MetricSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ε : Real
      f : ι → α → β
      g : α → β
      inst✝² : SemilatticeSup ι
      inst✝¹ : Nonempty ι
      inst✝ : Countable ι
      hε : LT.lt 0 ε
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hsm : MeasurableSet s
      hs : Ne (μ s) Top.top
      hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
      ⊢ ∀ (n : Nat), LE.le 0 (HPow.hPow (Inv.inv 2) n)
    -/
  · intro n; positivity
             /-
               🎉 no goals
             -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      inst✝³ : MetricSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ε : Real
      f : ι → α → β
      g : α → β
      inst✝² : SemilatticeSup ι
      inst✝¹ : Nonempty ι
      inst✝ : Countable ι
      hε : LT.lt 0 ε
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hsm : MeasurableSet s
      hs : Ne (μ s) Top.top
      hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
      ⊢ Summable (HPow.hPow (Inv.inv 2))
    -/
  · rw [inv_eq_one_div]
    /-
      case hf
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace α
      inst✝³ : MetricSpace β
      μ : MeasureTheory.Measure α
      s : Set α
      ε : Real
      f : ι → α → β
      g : α → β
      inst✝² : SemilatticeSup ι
      inst✝¹ : Nonempty ι
      inst✝ : Countable ι
      hε : LT.lt 0 ε
      hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
      hg : MeasureTheory.StronglyMeasurable g
      hsm : MeasurableSet s
      hs : Ne (μ s) Top.top
      hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
      ⊢ Summable (HPow.hPow (1 / 2))
    -/
    exact summable_geometric_two
    /-
      🎉 no goals
    -/


theorem iUnionNotConvergentSeq_subset (hε : 0 < ε) (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hsm : MeasurableSet s) (hs : μ s ≠ ∞)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    iUnionNotConvergentSeq hε hf hg hsm hs hfg ⊆ s := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    ⊢ HasSubset.Subset (MeasureTheory.Egorov.iUnionNotConvergentSeq hε hf hg hsm h …
  -/
  rw [iUnionNotConvergentSeq, ← Set.inter_iUnion]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    ⊢ HasSubset.Subset (Inter.inter s (Set.iUnion fun i => MeasureTheory.Egorov.no …
  -/
  exact Set.inter_subset_left
  /-
    🎉 no goals
  -/


theorem tendstoUniformlyOn_diff_iUnionNotConvergentSeq (hε : 0 < ε)
    (hf : ∀ n, StronglyMeasurable (f n)) (hg : StronglyMeasurable g) (hsm : MeasurableSet s)
    (hs : μ s ≠ ∞) (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) :
    TendstoUniformlyOn f g atTop (s \ Egorov.iUnionNotConvergentSeq hε hf hg hsm hs hfg) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    ⊢ TendstoUniformlyOn f g Filter.atTop (SDiff.sdiff s (MeasureTheory.Egorov.iUn …
  -/
  rw [Metric.tendstoUniformlyOn_iff]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    ⊢ ∀ (ε_1 : Real), GT.gt ε_1 0 → Filter.Eventually (fun n => ∀ (x : α), Members …
  -/
  intro δ hδ
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem (SDiff.sdiff s (Measur …
  -/
  obtain ⟨N, hN⟩ := exists_nat_one_div_lt hδ
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑N) 1)) δ
    ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem (SDiff.sdiff s (Measur …
  -/
  rw [eventually_atTop]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑N) 1)) δ
    ⊢ Exists fun a => ∀ (b : ι), GE.ge b a → ∀ (x : α), Membership.mem (SDiff.sdif …
  -/
  refine ⟨Egorov.notConvergentSeqLTIndex (half_pos hε) hf hg hsm hs hfg N, fun n hn x hx => ?_⟩
  simp only [Set.mem_diff, Egorov.iUnionNotConvergentSeq, not_exists, Set.mem_iUnion,
    Set.mem_inter_iff, not_and, exists_and_left] at hx
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑N) 1)) δ
    n : ι
    hn : GE.ge n (MeasureTheory.Egorov.notConvergentSeqLTIndex ⋯ hf hg hsm hs hfg N)
    x : α
    hx : And (Membership.mem s x) (Membership.mem s x → ∀ (x_1 : Nat), Not (Member …
    ⊢ LT.lt (Dist.dist (g x) (f n x)) δ
  -/
  obtain ⟨hxs, hx⟩ := hx
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑N) 1)) δ
    n : ι
    hn : GE.ge n (MeasureTheory.Egorov.notConvergentSeqLTIndex ⋯ hf hg hsm hs hfg N)
    x : α
    hxs : Membership.mem s x
    hx : Membership.mem s x → ∀ (x_1 : Nat), Not (Membership.mem (MeasureTheory.Eg …
    ⊢ LT.lt (Dist.dist (g x) (f n x)) δ
  -/
  specialize hx hxs N
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑N) 1)) δ
    n : ι
    hn : GE.ge n (MeasureTheory.Egorov.notConvergentSeqLTIndex ⋯ hf hg hsm hs hfg N)
    x : α
    hxs : Membership.mem s x
    hx : Not (Membership.mem (MeasureTheory.Egorov.notConvergentSeq f g N (Measure …
    ⊢ LT.lt (Dist.dist (g x) (f n x)) δ
  -/
  rw [Egorov.mem_notConvergentSeq_iff] at hx
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑N) 1)) δ
    n : ι
    hn : GE.ge n (MeasureTheory.Egorov.notConvergentSeqLTIndex ⋯ hf hg hsm hs hfg N)
    x : α
    hxs : Membership.mem s x
    hx : Not (Exists fun k => And (GE.ge k (MeasureTheory.Egorov.notConvergentSeqL …
    ⊢ LT.lt (Dist.dist (g x) (f n x)) δ
  -/
  push_neg at hx
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑N) 1)) δ
    n : ι
    hn : GE.ge n (MeasureTheory.Egorov.notConvergentSeqLTIndex ⋯ hf hg hsm hs hfg N)
    x : α
    hxs : Membership.mem s x
    hx : ∀ (k : ι), GE.ge k (MeasureTheory.Egorov.notConvergentSeqLTIndex ⋯ hf hg  …
    ⊢ LT.lt (Dist.dist (g x) (f n x)) δ
  -/
  rw [dist_comm]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝³ : MetricSpace β
    μ : MeasureTheory.Measure α
    s : Set α
    ε : Real
    f : ι → α → β
    g : α → β
    inst✝² : SemilatticeSup ι
    inst✝¹ : Nonempty ι
    inst✝ : Countable ι
    hε : LT.lt 0 ε
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hsm : MeasurableSet s
    hs : Ne (μ s) Top.top
    hfg : Filter.Eventually (fun x => Membership.mem s x → Filter.Tendsto (fun n = …
    δ : Real
    hδ : GT.gt δ 0
    N : Nat
    hN : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑N) 1)) δ
    n : ι
    hn : GE.ge n (MeasureTheory.Egorov.notConvergentSeqLTIndex ⋯ hf hg hsm hs hfg N)
    x : α
    hxs : Membership.mem s x
    hx : ∀ (k : ι), GE.ge k (MeasureTheory.Egorov.notConvergentSeqLTIndex ⋯ hf hg  …
    ⊢ LT.lt (Dist.dist (f n x) (g x)) δ
  -/
  exact lt_of_le_of_lt (hx n hn) hN
  /-
    🎉 no goals
  -/


/-- **Egorov's theorem**: If `f : ι → α → β` is a sequence of strongly measurable functions that
converges to `g : α → β` almost everywhere on a measurable set `s` of finite measure,
then for all `ε > 0`, there exists a subset `t ⊆ s` such that `μ t ≤ ε` and `f` converges to `g`
uniformly on `s \ t`. We require the index type `ι` to be countable, and usually `ι = ℕ`.

In other words, a sequence of almost everywhere convergent functions converges uniformly except on
an arbitrarily small set. -/
theorem tendstoUniformlyOn_of_ae_tendsto (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hsm : MeasurableSet s) (hs : μ s ≠ ∞)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → Tendsto (fun n => f n x) atTop (𝓝 (g x))) {ε : ℝ} (hε : 0 < ε) :
    ∃ t ⊆ s, MeasurableSet t ∧ μ t ≤ ENNReal.ofReal ε ∧ TendstoUniformlyOn f g atTop (s \ t) :=
  ⟨Egorov.iUnionNotConvergentSeq hε hf hg hsm hs hfg,
    Egorov.iUnionNotConvergentSeq_subset hε hf hg hsm hs hfg,
    Egorov.iUnionNotConvergentSeq_measurableSet hε hf hg hsm hs hfg,
    Egorov.measure_iUnionNotConvergentSeq hε hf hg hsm hs hfg,
    Egorov.tendstoUniformlyOn_diff_iUnionNotConvergentSeq hε hf hg hsm hs hfg⟩


/-- Egorov's theorem for finite measure spaces. -/
theorem tendstoUniformlyOn_of_ae_tendsto' [IsFiniteMeasure μ] (hf : ∀ n, StronglyMeasurable (f n))
    (hg : StronglyMeasurable g) (hfg : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) {ε : ℝ}
    (hε : 0 < ε) :
    ∃ t, MeasurableSet t ∧ μ t ≤ ENNReal.ofReal ε ∧ TendstoUniformlyOn f g atTop tᶜ := by
  have ⟨t, _, ht, htendsto⟩ := tendstoUniformlyOn_of_ae_tendsto hf hg MeasurableSet.univ
    (measure_ne_top μ Set.univ) (by filter_upwards [hfg] with _ htendsto _ using htendsto) hε
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝⁴ : MetricSpace β
    μ : MeasureTheory.Measure α
    inst✝³ : SemilatticeSup ι
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    f : ι → α → β
    g : α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    t : Set α
    left✝ : HasSubset.Subset t Set.univ
    ht : MeasurableSet t
    htendsto : And (LE.le (μ t) (ENNReal.ofReal ε)) (TendstoUniformlyOn f g Filter …
    ⊢ Exists fun t => And (MeasurableSet t) (And (LE.le (μ t) (ENNReal.ofReal ε))  …
  -/
  refine ⟨_, ht, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace α
    inst✝⁴ : MetricSpace β
    μ : MeasureTheory.Measure α
    inst✝³ : SemilatticeSup ι
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    f : ι → α → β
    g : α → β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : ∀ (n : ι), MeasureTheory.StronglyMeasurable (f n)
    hg : MeasureTheory.StronglyMeasurable g
    hfg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop …
    ε : Real
    hε : LT.lt 0 ε
    t : Set α
    left✝ : HasSubset.Subset t Set.univ
    ht : MeasurableSet t
    htendsto : And (LE.le (μ t) (ENNReal.ofReal ε)) (TendstoUniformlyOn f g Filter …
    ⊢ And (LE.le (μ t) (ENNReal.ofReal ε)) (TendstoUniformlyOn f g Filter.atTop (H …
  -/
  rwa [Set.compl_eq_univ_diff]
  /-
    🎉 no goals
  -/


