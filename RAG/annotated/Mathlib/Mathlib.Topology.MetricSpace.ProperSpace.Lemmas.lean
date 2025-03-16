/-- If a nonempty ball in a proper space includes a closed set `s`, then there exists a nonempty
ball with the same center and a strictly smaller radius that includes `s`. -/
theorem exists_pos_lt_subset_ball (hr : 0 < r) (hs : IsClosed s) (h : s ⊆ ball x r) :
    ∃ r' ∈ Ioo 0 r, s ⊆ ball x r' := by
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    x : α
    r : Real
    s : Set α
    hr : LT.lt 0 r
    hs : IsClosed s
    h : HasSubset.Subset s (Metric.ball x r)
    ⊢ Exists fun r' => And (Membership.mem (Set.Ioo 0 r) r') (HasSubset.Subset s ( …
  -/
  rcases eq_empty_or_nonempty s with (rfl | hne)
    /-
      case inl
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      x : α
      r : Real
      hr : LT.lt 0 r
      hs : IsClosed EmptyCollection.emptyCollection
      h : HasSubset.Subset EmptyCollection.emptyCollection (Metric.ball x r)
      ⊢ Exists fun r' => And (Membership.mem (Set.Ioo 0 r) r') (HasSubset.Subset Emp …
    -/
  · exact ⟨r / 2, ⟨half_pos hr, half_lt_self hr⟩, empty_subset _⟩
    /-
      🎉 no goals
    -/
  have : IsCompact s :=
    (isCompact_closedBall x r).of_isClosed_subset hs (h.trans ball_subset_closedBall)
  obtain ⟨y, hys, hy⟩ : ∃ y ∈ s, s ⊆ closedBall x (dist y x) :=
    this.exists_isMaxOn (β := α) (α := ℝ) hne (continuous_id.dist continuous_const).continuousOn
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    x : α
    r : Real
    s : Set α
    hr : LT.lt 0 r
    hs : IsClosed s
    h : HasSubset.Subset s (Metric.ball x r)
    hne : s.Nonempty
    this : IsCompact s
    y : α
    hys : Membership.mem s y
    hy : HasSubset.Subset s (Metric.closedBall x (Dist.dist y x))
    ⊢ Exists fun r' => And (Membership.mem (Set.Ioo 0 r) r') (HasSubset.Subset s ( …
  -/
  have hyr : dist y x < r := h hys
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    x : α
    r : Real
    s : Set α
    hr : LT.lt 0 r
    hs : IsClosed s
    h : HasSubset.Subset s (Metric.ball x r)
    hne : s.Nonempty
    this : IsCompact s
    y : α
    hys : Membership.mem s y
    hy : HasSubset.Subset s (Metric.closedBall x (Dist.dist y x))
    hyr : LT.lt (Dist.dist y x) r
    ⊢ Exists fun r' => And (Membership.mem (Set.Ioo 0 r) r') (HasSubset.Subset s ( …
  -/
  rcases exists_between hyr with ⟨r', hyr', hrr'⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    x : α
    r : Real
    s : Set α
    hr : LT.lt 0 r
    hs : IsClosed s
    h : HasSubset.Subset s (Metric.ball x r)
    hne : s.Nonempty
    this : IsCompact s
    y : α
    hys : Membership.mem s y
    hy : HasSubset.Subset s (Metric.closedBall x (Dist.dist y x))
    hyr : LT.lt (Dist.dist y x) r
    r' : Real
    hyr' : LT.lt (Dist.dist y x) r'
    hrr' : LT.lt r' r
    ⊢ Exists fun r' => And (Membership.mem (Set.Ioo 0 r) r') (HasSubset.Subset s ( …
  -/
  exact ⟨r', ⟨dist_nonneg.trans_lt hyr', hrr'⟩, hy.trans <| closedBall_subset_ball hyr'⟩
  /-
    🎉 no goals
  -/


/-- If a ball in a proper space includes a closed set `s`, then there exists a ball with the same
center and a strictly smaller radius that includes `s`. -/
theorem exists_lt_subset_ball (hs : IsClosed s) (h : s ⊆ ball x r) : ∃ r' < r, s ⊆ ball x r' := by
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    x : α
    r : Real
    s : Set α
    hs : IsClosed s
    h : HasSubset.Subset s (Metric.ball x r)
    ⊢ Exists fun r' => And (LT.lt r' r) (HasSubset.Subset s (Metric.ball x r'))
  -/
  rcases le_or_lt r 0 with hr | hr
    /-
      case inl
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      x : α
      r : Real
      s : Set α
      hs : IsClosed s
      h : HasSubset.Subset s (Metric.ball x r)
      hr : LE.le r 0
      ⊢ Exists fun r' => And (LT.lt r' r) (HasSubset.Subset s (Metric.ball x r'))
    -/
  · rw [ball_eq_empty.2 hr, subset_empty_iff] at h
    /-
      case inl
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      x : α
      r : Real
      s : Set α
      hs : IsClosed s
      h : Eq s EmptyCollection.emptyCollection
      hr : LE.le r 0
      ⊢ Exists fun r' => And (LT.lt r' r) (HasSubset.Subset s (Metric.ball x r'))
    -/
    subst s
    /-
      case inl
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      x : α
      r : Real
      hr : LE.le r 0
      hs : IsClosed EmptyCollection.emptyCollection
      ⊢ Exists fun r' => And (LT.lt r' r) (HasSubset.Subset EmptyCollection.emptyCol …
    -/
    exact (exists_lt r).imp fun r' hr' => ⟨hr', empty_subset _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      x : α
      r : Real
      s : Set α
      hs : IsClosed s
      h : HasSubset.Subset s (Metric.ball x r)
      hr : LT.lt 0 r
      ⊢ Exists fun r' => And (LT.lt r' r) (HasSubset.Subset s (Metric.ball x r'))
    -/
  · exact (exists_pos_lt_subset_ball hr hs h).imp fun r' hr' => ⟨hr'.1.2, hr'.2⟩
    /-
      🎉 no goals
    -/


theorem Metric.exists_isLocalMin_mem_ball [TopologicalSpace β]
    [ConditionallyCompleteLinearOrder β] [OrderTopology β] {f : α → β} {a z : α} {r : ℝ}
    (hf : ContinuousOn f (closedBall a r)) (hz : z ∈ closedBall a r)
    (hf1 : ∀ z' ∈ sphere a r, f z < f z') : ∃ z ∈ ball a r, IsLocalMin f z := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : PseudoMetricSpace α
    inst✝³ : ProperSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : ConditionallyCompleteLinearOrder β
    inst✝ : OrderTopology β
    f : α → β
    a z : α
    r : Real
    hf : ContinuousOn f (Metric.closedBall a r)
    hz : Membership.mem (Metric.closedBall a r) z
    hf1 : ∀ (z' : α), Membership.mem (Metric.sphere a r) z' → LT.lt (f z) (f z')
    ⊢ Exists fun z => And (Membership.mem (Metric.ball a r) z) (IsLocalMin f z)
  -/
  simp_rw [← closedBall_diff_ball] at hf1
  exact (isCompact_closedBall a r).exists_isLocalMin_mem_open ball_subset_closedBall hf hz hf1
    isOpen_ball

