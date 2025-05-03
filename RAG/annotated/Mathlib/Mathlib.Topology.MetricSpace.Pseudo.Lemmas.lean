instance : OrderTopology ℝ :=
  orderTopology_of_nhds_abs fun x => by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝ : PseudoMetricSpace α
      x : Real
      ⊢ Eq (nhds x) (iInf fun r => iInf fun h => Filter.principal (setOf fun b => LT …
    -/
    simp only [nhds_basis_ball.eq_biInf, ball, Real.dist_eq, abs_sub_comm]
    /-
      🎉 no goals
    -/


lemma Real.singleton_eq_inter_Icc (b : ℝ) : {b} = ⋂ (r > 0), Icc (b - r) (b + r) := by
  /-
    b : Real
    ⊢ Eq (Singleton.singleton b) (Set.iInter fun r => Set.iInter fun h => Set.Icc  …
  -/
  simp [Icc_eq_closedBall, biInter_basis_nhds Metric.nhds_basis_closedBall]
  /-
    🎉 no goals
  -/


/-- Special case of the sandwich lemma; see `tendsto_of_tendsto_of_tendsto_of_le_of_le'` for the
general case. -/
lemma squeeze_zero' {α} {f g : α → ℝ} {t₀ : Filter α} (hf : ∀ᶠ t in t₀, 0 ≤ f t)
    (hft : ∀ᶠ t in t₀, f t ≤ g t) (g0 : Tendsto g t₀ (𝓝 0)) : Tendsto f t₀ (𝓝 0) :=
  tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds g0 hf hft


/-- Special case of the sandwich lemma; see `tendsto_of_tendsto_of_tendsto_of_le_of_le`
and `tendsto_of_tendsto_of_tendsto_of_le_of_le'` for the general case. -/
lemma squeeze_zero {α} {f g : α → ℝ} {t₀ : Filter α} (hf : ∀ t, 0 ≤ f t) (hft : ∀ t, f t ≤ g t)
    (g0 : Tendsto g t₀ (𝓝 0)) : Tendsto f t₀ (𝓝 0) :=
  squeeze_zero' (Eventually.of_forall hf) (Eventually.of_forall hft) g0


/-- If `u` is a neighborhood of `x`, then for small enough `r`, the closed ball
`Metric.closedBall x r` is contained in `u`. -/
lemma eventually_closedBall_subset {x : α} {u : Set α} (hu : u ∈ 𝓝 x) :
    ∀ᶠ r in 𝓝 (0 : ℝ), closedBall x r ⊆ u := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    u : Set α
    hu : Membership.mem (nhds x) u
    ⊢ Filter.Eventually (fun r => HasSubset.Subset (Metric.closedBall x r) u) (nhd …
  -/
  obtain ⟨ε, εpos, hε⟩ : ∃ ε, 0 < ε ∧ closedBall x ε ⊆ u := nhds_basis_closedBall.mem_iff.1 hu
  /-
    case intro.intro
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    u : Set α
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    ⊢ Filter.Eventually (fun r => HasSubset.Subset (Metric.closedBall x r) u) (nhd …
  -/
  have : Iic ε ∈ 𝓝 (0 : ℝ) := Iic_mem_nhds εpos
  /-
    case intro.intro
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    u : Set α
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    this : Membership.mem (nhds 0) (Set.Iic ε)
    ⊢ Filter.Eventually (fun r => HasSubset.Subset (Metric.closedBall x r) u) (nhd …
  -/
  filter_upwards [this] with _ hr using Subset.trans (closedBall_subset_closedBall hr) hε
  /-
    🎉 no goals
  -/


lemma tendsto_closedBall_smallSets (x : α) : Tendsto (closedBall x) (𝓝 0) (𝓝 x).smallSets :=
  tendsto_smallSets_iff.2 fun _ ↦ eventually_closedBall_subset


lemma isClosed_ball : IsClosed (closedBall x ε) :=
  isClosed_le (continuous_id.dist continuous_const) continuous_const


lemma isClosed_sphere : IsClosed (sphere x ε) :=
  isClosed_eq (continuous_id.dist continuous_const) continuous_const


@[simp]
lemma closure_closedBall : closure (closedBall x ε) = closedBall x ε :=
  isClosed_ball.closure_eq


@[simp]
lemma closure_sphere : closure (sphere x ε) = sphere x ε :=
  isClosed_sphere.closure_eq


lemma closure_ball_subset_closedBall : closure (ball x ε) ⊆ closedBall x ε :=
  closure_minimal ball_subset_closedBall isClosed_ball


lemma frontier_ball_subset_sphere : frontier (ball x ε) ⊆ sphere x ε :=
  frontier_lt_subset_eq (continuous_id.dist continuous_const) continuous_const


lemma frontier_closedBall_subset_sphere : frontier (closedBall x ε) ⊆ sphere x ε :=
  frontier_le_subset_eq (continuous_id.dist continuous_const) continuous_const


lemma closedBall_zero' (x : α) : closedBall x 0 = closure {x} :=
  Subset.antisymm
    (fun _y hy =>
      mem_closure_iff.2 fun _ε ε0 => ⟨x, mem_singleton x, (mem_closedBall.1 hy).trans_lt ε0⟩)
    (closure_minimal (singleton_subset_iff.2 (dist_self x).le) isClosed_ball)


lemma eventually_isCompact_closedBall [WeaklyLocallyCompactSpace α] (x : α) :
    ∀ᶠ r in 𝓝 (0 : ℝ), IsCompact (closedBall x r) := by
  /-
    α : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : WeaklyLocallyCompactSpace α
    x : α
    ⊢ Filter.Eventually (fun r => IsCompact (Metric.closedBall x r)) (nhds 0)
  -/
  rcases exists_compact_mem_nhds x with ⟨s, s_compact, hs⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : WeaklyLocallyCompactSpace α
    x : α
    s : Set α
    s_compact : IsCompact s
    hs : Membership.mem (nhds x) s
    ⊢ Filter.Eventually (fun r => IsCompact (Metric.closedBall x r)) (nhds 0)
  -/
  filter_upwards [eventually_closedBall_subset hs] with r hr
  /-
    case h
    α : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : WeaklyLocallyCompactSpace α
    x : α
    s : Set α
    s_compact : IsCompact s
    hs : Membership.mem (nhds x) s
    r : Real
    hr : HasSubset.Subset (Metric.closedBall x r) s
    ⊢ IsCompact (Metric.closedBall x r)
  -/
  exact IsCompact.of_isClosed_subset s_compact isClosed_ball hr
  /-
    🎉 no goals
  -/


lemma exists_isCompact_closedBall [WeaklyLocallyCompactSpace α] (x : α) :
    ∃ r, 0 < r ∧ IsCompact (closedBall x r) := by
  have : ∀ᶠ r in 𝓝[>] 0, IsCompact (closedBall x r) :=
    eventually_nhdsWithin_of_eventually_nhds (eventually_isCompact_closedBall x)
  /-
    α : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : WeaklyLocallyCompactSpace α
    x : α
    this : Filter.Eventually (fun r => IsCompact (Metric.closedBall x r)) (nhdsWit …
    ⊢ Exists fun r => And (LT.lt 0 r) (IsCompact (Metric.closedBall x r))
  -/
  simpa only [and_comm] using (this.and self_mem_nhdsWithin).exists
  /-
    🎉 no goals
  -/


theorem biInter_gt_closedBall (x : α) (r : ℝ) : ⋂ r' > r, closedBall x r' = closedBall x r := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    ⊢ Eq (Set.iInter fun r' => Set.iInter fun h => Metric.closedBall x r') (Metric …
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    x✝ : α
    ⊢ Iff (Membership.mem (Set.iInter fun r' => Set.iInter fun h => Metric.closedB …
  -/
  simp [forall_gt_ge_iff]
  /-
    🎉 no goals
  -/


theorem biInter_gt_ball (x : α) (r : ℝ) : ⋂ r' > r, ball x r' = closedBall x r := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    ⊢ Eq (Set.iInter fun r' => Set.iInter fun h => Metric.ball x r') (Metric.close …
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    x✝ : α
    ⊢ Iff (Membership.mem (Set.iInter fun r' => Set.iInter fun h => Metric.ball x  …
  -/
  simp [forall_lt_iff_le']
  /-
    🎉 no goals
  -/


theorem biUnion_lt_ball (x : α) (r : ℝ) : ⋃ r' < r, ball x r' = ball x r := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    ⊢ Eq (Set.iUnion fun r' => Set.iUnion fun h => Metric.ball x r') (Metric.ball  …
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    x✝ : α
    ⊢ Iff (Membership.mem (Set.iUnion fun r' => Set.iUnion fun h => Metric.ball x  …
  -/
  rw [← not_iff_not]
  /-
    case h
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    x✝ : α
    ⊢ Iff (Not (Membership.mem (Set.iUnion fun r' => Set.iUnion fun h => Metric.ba …
  -/
  simp [forall_lt_le_iff]
  /-
    🎉 no goals
  -/


theorem biUnion_lt_closedBall (x : α) (r : ℝ) : ⋃ r' < r, closedBall x r' = ball x r := by
  /-
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    ⊢ Eq (Set.iUnion fun r' => Set.iUnion fun h => Metric.closedBall x r') (Metric …
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    x✝ : α
    ⊢ Iff (Membership.mem (Set.iUnion fun r' => Set.iUnion fun h => Metric.closedB …
  -/
  rw [← not_iff_not]
  /-
    case h
    α : Type u_2
    inst✝ : PseudoMetricSpace α
    x : α
    r : Real
    x✝ : α
    ⊢ Iff (Not (Membership.mem (Set.iUnion fun r' => Set.iUnion fun h => Metric.cl …
  -/
  simp [forall_lt_iff_le]
  /-
    🎉 no goals
  -/


