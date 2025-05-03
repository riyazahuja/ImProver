/-- Closed balls are bounded -/
theorem isBounded_closedBall : IsBounded (closedBall x r) :=
  isBounded_iff.2 ⟨r + r, fun y hy z hz =>
    calc dist y z ≤ dist y x + dist z x := dist_triangle_right _ _ _
    _ ≤ r + r := add_le_add hy hz⟩


/-- Open balls are bounded -/
theorem isBounded_ball : IsBounded (ball x r) :=
  isBounded_closedBall.subset ball_subset_closedBall


/-- Spheres are bounded -/
theorem isBounded_sphere : IsBounded (sphere x r) :=
  isBounded_closedBall.subset sphere_subset_closedBall


/-- Given a point, a bounded subset is included in some ball around this point -/
theorem isBounded_iff_subset_closedBall (c : α) : IsBounded s ↔ ∃ r, s ⊆ closedBall c r :=
  ⟨fun h ↦ (isBounded_iff.1 (h.insert c)).imp fun _r hr _x hx ↦ hr (.inr hx) (mem_insert _ _),
    fun ⟨_r, hr⟩ ↦ isBounded_closedBall.subset hr⟩


theorem _root_.Bornology.IsBounded.subset_closedBall (h : IsBounded s) (c : α) :
    ∃ r, s ⊆ closedBall c r :=
  (isBounded_iff_subset_closedBall c).1 h


theorem _root_.Bornology.IsBounded.subset_ball_lt (h : IsBounded s) (a : ℝ) (c : α) :
    ∃ r, a < r ∧ s ⊆ ball c r :=
  let ⟨r, hr⟩ := h.subset_closedBall c
  ⟨max r a + 1, (le_max_right _ _).trans_lt (lt_add_one _), hr.trans <| closedBall_subset_ball <|
    (le_max_left _ _).trans_lt (lt_add_one _)⟩


theorem _root_.Bornology.IsBounded.subset_ball (h : IsBounded s) (c : α) : ∃ r, s ⊆ ball c r :=
  (h.subset_ball_lt 0 c).imp fun _ ↦ And.right


theorem isBounded_iff_subset_ball (c : α) : IsBounded s ↔ ∃ r, s ⊆ ball c r :=
  ⟨(IsBounded.subset_ball · c), fun ⟨_r, hr⟩ ↦ isBounded_ball.subset hr⟩


theorem _root_.Bornology.IsBounded.subset_closedBall_lt (h : IsBounded s) (a : ℝ) (c : α) :
    ∃ r, a < r ∧ s ⊆ closedBall c r :=
  let ⟨r, har, hr⟩ := h.subset_ball_lt a c
  ⟨r, har, hr.trans ball_subset_closedBall⟩


theorem isBounded_closure_of_isBounded (h : IsBounded s) : IsBounded (closure s) :=
  let ⟨C, h⟩ := isBounded_iff.1 h
  isBounded_iff.2 ⟨C, fun _a ha _b hb => isClosed_Iic.closure_subset <|
    map_mem_closure₂ continuous_dist ha hb h⟩


protected theorem _root_.Bornology.IsBounded.closure (h : IsBounded s) : IsBounded (closure s) :=
  isBounded_closure_of_isBounded h


@[simp]
theorem isBounded_closure_iff : IsBounded (closure s) ↔ IsBounded s :=
  ⟨fun h => h.subset subset_closure, fun h => h.closure⟩


theorem hasBasis_cobounded_compl_closedBall (c : α) :
    (cobounded α).HasBasis (fun _ ↦ True) (fun r ↦ (closedBall c r)ᶜ) :=
                                                                                     /-
                                                                                       α : Type u
                                                                                       inst✝ : PseudoMetricSpace α
                                                                                       c : α
                                                                                       x✝ : Set α
                                                                                       ⊢ Iff (Exists fun r => HasSubset.Subset x✝ (Metric.closedBall c r)) (Exists fu …
                                                                                     -/
  ⟨compl_surjective.forall.2 fun _ ↦ (isBounded_iff_subset_closedBall c).trans <| by simp⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem hasBasis_cobounded_compl_ball (c : α) :
    (cobounded α).HasBasis (fun _ ↦ True) (fun r ↦ (ball c r)ᶜ) :=
                                                                               /-
                                                                                 α : Type u
                                                                                 inst✝ : PseudoMetricSpace α
                                                                                 c : α
                                                                                 x✝ : Set α
                                                                                 ⊢ Iff (Exists fun r => HasSubset.Subset x✝ (Metric.ball c r)) (Exists fun i => …
                                                                               -/
  ⟨compl_surjective.forall.2 fun _ ↦ (isBounded_iff_subset_ball c).trans <| by simp⟩
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem comap_dist_right_atTop (c : α) : comap (dist · c) atTop = cobounded α :=
  (atTop_basis.comap _).eq_of_same_basis <| by
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      c : α
      ⊢ (Bornology.cobounded α).HasBasis (fun x => True) fun i => Set.preimage (fun  …
    -/
    simpa only [compl_def, mem_ball, not_lt] using hasBasis_cobounded_compl_ball c
    /-
      🎉 no goals
    -/


@[simp]
theorem comap_dist_left_atTop (c : α) : comap (dist c) atTop = cobounded α := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    c : α
    ⊢ Eq (Filter.comap (Dist.dist c) Filter.atTop) (Bornology.cobounded α)
  -/
  simpa only [dist_comm _ c] using comap_dist_right_atTop c
  /-
    🎉 no goals
  -/


@[simp]
theorem tendsto_dist_right_atTop_iff (c : α) {f : β → α} {l : Filter β} :
    Tendsto (fun x ↦ dist (f x) c) l atTop ↔ Tendsto f l (cobounded α) := by
  /-
    α : Type u
    β : Type v
    inst✝ : PseudoMetricSpace α
    c : α
    f : β → α
    l : Filter β
    ⊢ Iff (Filter.Tendsto (fun x => Dist.dist (f x) c) l Filter.atTop) (Filter.Ten …
  -/
  rw [← comap_dist_right_atTop c, tendsto_comap_iff, Function.comp_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem tendsto_dist_left_atTop_iff (c : α) {f : β → α} {l : Filter β} :
    Tendsto (fun x ↦ dist c (f x)) l atTop ↔ Tendsto f l (cobounded α) := by
  /-
    α : Type u
    β : Type v
    inst✝ : PseudoMetricSpace α
    c : α
    f : β → α
    l : Filter β
    ⊢ Iff (Filter.Tendsto (fun x => Dist.dist c (f x)) l Filter.atTop) (Filter.Ten …
  -/
  simp only [dist_comm c, tendsto_dist_right_atTop_iff]
  /-
    🎉 no goals
  -/


theorem tendsto_dist_right_cobounded_atTop (c : α) : Tendsto (dist · c) (cobounded α) atTop :=
  tendsto_iff_comap.2 (comap_dist_right_atTop c).ge


theorem tendsto_dist_left_cobounded_atTop (c : α) : Tendsto (dist c) (cobounded α) atTop :=
  tendsto_iff_comap.2 (comap_dist_left_atTop c).ge


/-- A totally bounded set is bounded -/
theorem _root_.TotallyBounded.isBounded {s : Set α} (h : TotallyBounded s) : IsBounded s :=
  -- We cover the totally bounded set by finitely many balls of radius 1,
  -- and then argue that a finite union of bounded sets is bounded
  let ⟨_t, fint, subs⟩ := (totallyBounded_iff.mp h) 1 zero_lt_one
  ((isBounded_biUnion fint).2 fun _ _ => isBounded_ball).subset subs


/-- A compact set is bounded -/
theorem _root_.IsCompact.isBounded {s : Set α} (h : IsCompact s) : IsBounded s :=
  -- A compact set is totally bounded, thus bounded
  h.totallyBounded.isBounded


theorem cobounded_le_cocompact : cobounded α ≤ cocompact α :=
  hasBasis_cocompact.ge_iff.2 fun _s hs ↦ hs.isBounded


theorem isCobounded_iff_closedBall_compl_subset {s : Set α} (c : α) :
    IsCobounded s ↔ ∃ (r : ℝ), (Metric.closedBall c r)ᶜ ⊆ s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    c : α
    ⊢ Iff (Bornology.IsCobounded s) (Exists fun r => HasSubset.Subset (HasCompl.co …
  -/
  rw [← isBounded_compl_iff, isBounded_iff_subset_closedBall c]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    c : α
    ⊢ Iff (Exists fun r => HasSubset.Subset (HasCompl.compl s) (Metric.closedBall  …
  -/
  apply exists_congr
  /-
    case h
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    c : α
    ⊢ ∀ (a : Real), Iff (HasSubset.Subset (HasCompl.compl s) (Metric.closedBall c  …
  -/
  intro r
  /-
    case h
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    c : α
    r : Real
    ⊢ Iff (HasSubset.Subset (HasCompl.compl s) (Metric.closedBall c r)) (HasSubset …
  -/
  rw [compl_subset_comm]
  /-
    🎉 no goals
  -/


theorem _root_.Bornology.IsCobounded.closedBall_compl_subset {s : Set α} (hs : IsCobounded s)
    (c : α) : ∃ (r : ℝ), (Metric.closedBall c r)ᶜ ⊆ s :=
  (isCobounded_iff_closedBall_compl_subset c).mp hs


theorem closedBall_compl_subset_of_mem_cocompact {s : Set α} (hs : s ∈ cocompact α) (c : α) :
    ∃ (r : ℝ), (Metric.closedBall c r)ᶜ ⊆ s :=
  IsCobounded.closedBall_compl_subset (cobounded_le_cocompact hs) c


theorem mem_cocompact_of_closedBall_compl_subset [ProperSpace α] (c : α)
    (h : ∃ r, (closedBall c r)ᶜ ⊆ s) : s ∈ cocompact α := by
  /-
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    c : α
    h : Exists fun r => HasSubset.Subset (HasCompl.compl (Metric.closedBall c r)) s
    ⊢ Membership.mem (Filter.cocompact α) s
  -/
  rcases h with ⟨r, h⟩
  /-
    case intro
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    c : α
    r : Real
    h : HasSubset.Subset (HasCompl.compl (Metric.closedBall c r)) s
    ⊢ Membership.mem (Filter.cocompact α) s
  -/
  rw [Filter.mem_cocompact]
  /-
    case intro
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    c : α
    r : Real
    h : HasSubset.Subset (HasCompl.compl (Metric.closedBall c r)) s
    ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t) s)
  -/
  exact ⟨closedBall c r, isCompact_closedBall c r, h⟩
  /-
    🎉 no goals
  -/


theorem mem_cocompact_iff_closedBall_compl_subset [ProperSpace α] (c : α) :
    s ∈ cocompact α ↔ ∃ r, (closedBall c r)ᶜ ⊆ s :=
  ⟨(closedBall_compl_subset_of_mem_cocompact · _), mem_cocompact_of_closedBall_compl_subset _⟩


/-- Characterization of the boundedness of the range of a function -/
theorem isBounded_range_iff {f : β → α} : IsBounded (range f) ↔ ∃ C, ∀ x y, dist (f x) (f y) ≤ C :=
                            /-
                              α : Type u
                              β : Type v
                              inst✝ : PseudoMetricSpace α
                              f : β → α
                              ⊢ Iff (Exists fun C => ∀ ⦃x : α⦄, Membership.mem (Set.range f) x → ∀ ⦃y : α⦄,  …
                            -/
  isBounded_iff.trans <| by simp only [forall_mem_range]
                            /-
                              🎉 no goals
                            -/


theorem isBounded_image_iff {f : β → α} {s : Set β} :
    IsBounded (f '' s) ↔ ∃ C, ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ C :=
                            /-
                              α : Type u
                              β : Type v
                              inst✝ : PseudoMetricSpace α
                              f : β → α
                              s : Set β
                              ⊢ Iff (Exists fun C => ∀ ⦃x : α⦄, Membership.mem (Set.image f s) x → ∀ ⦃y : α⦄ …
                            -/
  isBounded_iff.trans <| by simp only [forall_mem_image]
                            /-
                              🎉 no goals
                            -/


theorem isBounded_range_of_tendsto_cofinite_uniformity {f : β → α}
    (hf : Tendsto (Prod.map f f) (.cofinite ×ˢ .cofinite) (𝓤 α)) : IsBounded (range f) := by
  rcases (hasBasis_cofinite.prod_self.tendsto_iff uniformity_basis_dist).1 hf 1 zero_lt_one with
    ⟨s, hsf, hs1⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝ : PseudoMetricSpace α
    f : β → α
    hf : Filter.Tendsto (Prod.map f f) (SProd.sprod Filter.cofinite Filter.cofinit …
    s : Set β
    hsf : s.Finite
    hs1 : ∀ (x : Prod β β), Membership.mem (SProd.sprod (HasCompl.compl s) (HasCom …
    ⊢ Bornology.IsBounded (Set.range f)
  -/
  rw [← image_union_image_compl_eq_range]
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝ : PseudoMetricSpace α
    f : β → α
    hf : Filter.Tendsto (Prod.map f f) (SProd.sprod Filter.cofinite Filter.cofinit …
    s : Set β
    hsf : s.Finite
    hs1 : ∀ (x : Prod β β), Membership.mem (SProd.sprod (HasCompl.compl s) (HasCom …
    ⊢ Bornology.IsBounded (Union.union (Set.image f ?m.50402) (Set.image f (HasCom …
  -/
  refine (hsf.image f).isBounded.union (isBounded_image_iff.2 ⟨1, fun x hx y hy ↦ ?_⟩)
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝ : PseudoMetricSpace α
    f : β → α
    hf : Filter.Tendsto (Prod.map f f) (SProd.sprod Filter.cofinite Filter.cofinit …
    s : Set β
    hsf : s.Finite
    hs1 : ∀ (x : Prod β β), Membership.mem (SProd.sprod (HasCompl.compl s) (HasCom …
    x : β
    hx : Membership.mem (HasCompl.compl s) x
    y : β
    hy : Membership.mem (HasCompl.compl s) y
    ⊢ LE.le (Dist.dist (f x) (f y)) 1
  -/
  exact le_of_lt (hs1 (x, y) ⟨hx, hy⟩)
  /-
    🎉 no goals
  -/


theorem isBounded_range_of_cauchy_map_cofinite {f : β → α} (hf : Cauchy (map f cofinite)) :
    IsBounded (range f) :=
  isBounded_range_of_tendsto_cofinite_uniformity <| (cauchy_map_iff.1 hf).2


theorem _root_.CauchySeq.isBounded_range {f : ℕ → α} (hf : CauchySeq f) : IsBounded (range f) :=
                                               /-
                                                 α : Type u
                                                 inst✝ : PseudoMetricSpace α
                                                 f : Nat → α
                                                 hf : CauchySeq f
                                                 ⊢ Cauchy (Filter.map f Filter.cofinite)
                                               -/
  isBounded_range_of_cauchy_map_cofinite <| by rwa [Nat.cofinite_eq_atTop]
                                               /-
                                                 🎉 no goals
                                               -/


theorem isBounded_range_of_tendsto_cofinite {f : β → α} {a : α} (hf : Tendsto f cofinite (𝓝 a)) :
    IsBounded (range f) :=
  isBounded_range_of_tendsto_cofinite_uniformity <|
    (hf.prod_map hf).mono_right <| nhds_prod_eq.symm.trans_le (nhds_le_uniformity a)


/-- In a compact space, all sets are bounded -/
theorem isBounded_of_compactSpace [CompactSpace α] : IsBounded s :=
  isCompact_univ.isBounded.subset (subset_univ _)


theorem isBounded_range_of_tendsto (u : ℕ → α) {x : α} (hu : Tendsto u atTop (𝓝 x)) :
    IsBounded (range u) :=
  hu.cauchySeq.isBounded_range


theorem disjoint_nhds_cobounded (x : α) : Disjoint (𝓝 x) (cobounded α) :=
  disjoint_of_disjoint_of_mem disjoint_compl_right (ball_mem_nhds _ one_pos) isBounded_ball


theorem disjoint_cobounded_nhds (x : α) : Disjoint (cobounded α) (𝓝 x) :=
  (disjoint_nhds_cobounded x).symm


theorem disjoint_nhdsSet_cobounded {s : Set α} (hs : IsCompact s) : Disjoint (𝓝ˢ s) (cobounded α) :=
  hs.disjoint_nhdsSet_left.2 fun _ _ ↦ disjoint_nhds_cobounded _


theorem disjoint_cobounded_nhdsSet {s : Set α} (hs : IsCompact s) : Disjoint (cobounded α) (𝓝ˢ s) :=
  (disjoint_nhdsSet_cobounded hs).symm


theorem exists_isBounded_image_of_tendsto {α β : Type*} [PseudoMetricSpace β]
    {l : Filter α} {f : α → β} {x : β} (hf : Tendsto f l (𝓝 x)) :
    ∃ s ∈ l, IsBounded (f '' s) :=
  (l.basis_sets.map f).disjoint_iff_left.mp <| (disjoint_nhds_cobounded x).mono_left hf


/-- If a function is continuous within a set `s` at every point of a compact set `k`, then it is
bounded on some open neighborhood of `k` in `s`. -/
theorem exists_isOpen_isBounded_image_inter_of_isCompact_of_forall_continuousWithinAt
    [TopologicalSpace β] {k s : Set β} {f : β → α} (hk : IsCompact k)
    (hf : ∀ x ∈ k, ContinuousWithinAt f s x) :
    ∃ t, k ⊆ t ∧ IsOpen t ∧ IsBounded (f '' (t ∩ s)) := by
  have : Disjoint (𝓝ˢ k ⊓ 𝓟 s) (comap f (cobounded α)) := by
    rw [disjoint_assoc, inf_comm, hk.disjoint_nhdsSet_left]
    exact fun x hx ↦ disjoint_left_comm.2 <|
      tendsto_comap.disjoint (disjoint_cobounded_nhds _) (hf x hx)
  rcases ((((hasBasis_nhdsSet _).inf_principal _)).disjoint_iff ((basis_sets _).comap _)).1 this
    with ⟨U, ⟨hUo, hkU⟩, t, ht, hd⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    k s : Set β
    f : β → α
    hk : IsCompact k
    hf : ∀ (x : β), Membership.mem k x → ContinuousWithinAt f s x
    this : Disjoint (Min.min (nhdsSet k) (Filter.principal s)) (Filter.comap f (Bo …
    U : Set β
    hUo : IsOpen U
    hkU : HasSubset.Subset k U
    t : Set α
    ht : Membership.mem (Bornology.cobounded α) t
    hd : Disjoint (Inter.inter U s) (Set.preimage f (id t))
    ⊢ Exists fun t => And (HasSubset.Subset k t) (And (IsOpen t) (Bornology.IsBoun …
  -/
  refine ⟨U, hkU, hUo, (isBounded_compl_iff.2 ht).subset ?_⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    k s : Set β
    f : β → α
    hk : IsCompact k
    hf : ∀ (x : β), Membership.mem k x → ContinuousWithinAt f s x
    this : Disjoint (Min.min (nhdsSet k) (Filter.principal s)) (Filter.comap f (Bo …
    U : Set β
    hUo : IsOpen U
    hkU : HasSubset.Subset k U
    t : Set α
    ht : Membership.mem (Bornology.cobounded α) t
    hd : Disjoint (Inter.inter U s) (Set.preimage f (id t))
    ⊢ HasSubset.Subset (Set.image f (Inter.inter U s)) (HasCompl.compl t)
  -/
  rwa [image_subset_iff, preimage_compl, subset_compl_iff_disjoint_right]
  /-
    🎉 no goals
  -/


/-- If a function is continuous at every point of a compact set `k`, then it is bounded on
some open neighborhood of `k`. -/
theorem exists_isOpen_isBounded_image_of_isCompact_of_forall_continuousAt [TopologicalSpace β]
    {k : Set β} {f : β → α} (hk : IsCompact k) (hf : ∀ x ∈ k, ContinuousAt f x) :
    ∃ t, k ⊆ t ∧ IsOpen t ∧ IsBounded (f '' t) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    k : Set β
    f : β → α
    hk : IsCompact k
    hf : ∀ (x : β), Membership.mem k x → ContinuousAt f x
    ⊢ Exists fun t => And (HasSubset.Subset k t) (And (IsOpen t) (Bornology.IsBoun …
  -/
  simp_rw [← continuousWithinAt_univ] at hf
  simpa only [inter_univ] using
    exists_isOpen_isBounded_image_inter_of_isCompact_of_forall_continuousWithinAt hk hf


/-- If a function is continuous on a set `s` containing a compact set `k`, then it is bounded on
some open neighborhood of `k` in `s`. -/
theorem exists_isOpen_isBounded_image_inter_of_isCompact_of_continuousOn [TopologicalSpace β]
    {k s : Set β} {f : β → α} (hk : IsCompact k) (hks : k ⊆ s) (hf : ContinuousOn f s) :
    ∃ t, k ⊆ t ∧ IsOpen t ∧ IsBounded (f '' (t ∩ s)) :=
  exists_isOpen_isBounded_image_inter_of_isCompact_of_forall_continuousWithinAt hk fun x hx =>
    hf x (hks hx)


/-- If a function is continuous on a neighborhood of a compact set `k`, then it is bounded on
some open neighborhood of `k`. -/
theorem exists_isOpen_isBounded_image_of_isCompact_of_continuousOn [TopologicalSpace β]
    {k s : Set β} {f : β → α} (hk : IsCompact k) (hs : IsOpen s) (hks : k ⊆ s)
    (hf : ContinuousOn f s) : ∃ t, k ⊆ t ∧ IsOpen t ∧ IsBounded (f '' t) :=
  exists_isOpen_isBounded_image_of_isCompact_of_forall_continuousAt hk fun _x hx =>
    hf.continuousAt (hs.mem_nhds (hks hx))


/-- The **Heine–Borel theorem**: In a proper space, a closed bounded set is compact. -/
theorem isCompact_of_isClosed_isBounded [ProperSpace α] (hc : IsClosed s) (hb : IsBounded s) :
    IsCompact s := by
  /-
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : ProperSpace α
    hc : IsClosed s
    hb : Bornology.IsBounded s
    ⊢ IsCompact s
  -/
  rcases eq_empty_or_nonempty s with (rfl | ⟨x, -⟩)
    /-
      case inl
      α : Type u
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      hc : IsClosed EmptyCollection.emptyCollection
      hb : Bornology.IsBounded EmptyCollection.emptyCollection
      ⊢ IsCompact EmptyCollection.emptyCollection
    -/
  · exact isCompact_empty
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u
      inst✝¹ : PseudoMetricSpace α
      s : Set α
      inst✝ : ProperSpace α
      hc : IsClosed s
      hb : Bornology.IsBounded s
      x : α
      ⊢ IsCompact s
    -/
  · rcases hb.subset_closedBall x with ⟨r, hr⟩
    /-
      case inr.intro.intro
      α : Type u
      inst✝¹ : PseudoMetricSpace α
      s : Set α
      inst✝ : ProperSpace α
      hc : IsClosed s
      hb : Bornology.IsBounded s
      x : α
      r : Real
      hr : HasSubset.Subset s (Metric.closedBall x r)
      ⊢ IsCompact s
    -/
    exact (isCompact_closedBall x r).of_isClosed_subset hc hr
    /-
      🎉 no goals
    -/


/-- The **Heine–Borel theorem**: In a proper space, the closure of a bounded set is compact. -/
theorem _root_.Bornology.IsBounded.isCompact_closure [ProperSpace α] (h : IsBounded s) :
    IsCompact (closure s) :=
  isCompact_of_isClosed_isBounded isClosed_closure h.closure

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: assume `[MetricSpace α]`
-- instead of `[PseudoMetricSpace α] [T2Space α]`

/-- The **Heine–Borel theorem**:
In a proper Hausdorff space, a set is compact if and only if it is closed and bounded. -/
theorem isCompact_iff_isClosed_bounded [T2Space α] [ProperSpace α] :
    IsCompact s ↔ IsClosed s ∧ IsBounded s :=
  ⟨fun h => ⟨h.isClosed, h.isBounded⟩, fun h => isCompact_of_isClosed_isBounded h.1 h.2⟩


theorem compactSpace_iff_isBounded_univ [ProperSpace α] :
    CompactSpace α ↔ IsBounded (univ : Set α) :=
  ⟨@isBounded_of_compactSpace α _ _, fun hb => ⟨isCompact_of_isClosed_isBounded isClosed_univ hb⟩⟩


theorem _root_.totallyBounded_Icc (a b : α) : TotallyBounded (Icc a b) :=
  isCompact_Icc.totallyBounded


theorem _root_.totallyBounded_Ico (a b : α) : TotallyBounded (Ico a b) :=
  (totallyBounded_Icc a b).subset Ico_subset_Icc_self


theorem _root_.totallyBounded_Ioc (a b : α) : TotallyBounded (Ioc a b) :=
  (totallyBounded_Icc a b).subset Ioc_subset_Icc_self


theorem _root_.totallyBounded_Ioo (a b : α) : TotallyBounded (Ioo a b) :=
  (totallyBounded_Icc a b).subset Ioo_subset_Icc_self


theorem isBounded_Icc (a b : α) : IsBounded (Icc a b) :=
  (totallyBounded_Icc a b).isBounded


theorem isBounded_Ico (a b : α) : IsBounded (Ico a b) :=
  (totallyBounded_Ico a b).isBounded


theorem isBounded_Ioc (a b : α) : IsBounded (Ioc a b) :=
  (totallyBounded_Ioc a b).isBounded


theorem isBounded_Ioo (a b : α) : IsBounded (Ioo a b) :=
  (totallyBounded_Ioo a b).isBounded


/-- In a pseudo metric space with a conditionally complete linear order such that the order and the
    metric structure give the same topology, any order-bounded set is metric-bounded. -/
theorem isBounded_of_bddAbove_of_bddBelow {s : Set α} (h₁ : BddAbove s) (h₂ : BddBelow s) :
    IsBounded s :=
  let ⟨u, hu⟩ := h₁
  let ⟨l, hl⟩ := h₂
  (isBounded_Icc l u).subset (fun _x hx => mem_Icc.mpr ⟨hl hx, hu hx⟩)


/-- The diameter of a set in a metric space. To get controllable behavior even when the diameter
should be infinite, we express it in terms of the `EMetric.diam` -/
noncomputable def diam (s : Set α) : ℝ :=
  ENNReal.toReal (EMetric.diam s)


/-- The diameter of a set is always nonnegative -/
theorem diam_nonneg : 0 ≤ diam s :=
  ENNReal.toReal_nonneg


theorem diam_subsingleton (hs : s.Subsingleton) : diam s = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : s.Subsingleton
    ⊢ Eq (Metric.diam s) 0
  -/
  simp only [diam, EMetric.diam_subsingleton hs, ENNReal.zero_toReal]
  /-
    🎉 no goals
  -/


/-- The empty set has zero diameter -/
@[simp]
theorem diam_empty : diam (∅ : Set α) = 0 :=
  diam_subsingleton subsingleton_empty


/-- A singleton has zero diameter -/
@[simp]
theorem diam_singleton : diam ({x} : Set α) = 0 :=
  diam_subsingleton subsingleton_singleton


@[to_additive (attr := simp)]
theorem diam_one [One α] : diam (1 : Set α) = 0 :=
  diam_singleton

-- Does not work as a simp-lemma, since {x, y} reduces to (insert y {x})

theorem diam_pair : diam ({x, y} : Set α) = dist x y := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Eq (Metric.diam (Insert.insert x (Singleton.singleton y))) (Dist.dist x y)
  -/
  simp only [diam, EMetric.diam_pair, dist_edist]
  /-
    🎉 no goals
  -/

-- Does not work as a simp-lemma, since {x, y, z} reduces to (insert z (insert y {x}))

theorem diam_triple :
    Metric.diam ({x, y, z} : Set α) = max (max (dist x y) (dist x z)) (dist y z) := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y z : α
    ⊢ Eq (Metric.diam (Insert.insert x (Insert.insert y (Singleton.singleton z)))) …
  -/
  simp only [Metric.diam, EMetric.diam_triple, dist_edist]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y z : α
    ⊢ Eq (Max.max (Max.max (EDist.edist x y) (EDist.edist x z)) (EDist.edist y z)) …
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  rw [ENNReal.toReal_max, ENNReal.toReal_max] <;> apply_rules [ne_of_lt, edist_lt_top, max_lt]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- If the distance between any two points in a set is bounded by some constant `C`,
then `ENNReal.ofReal C` bounds the emetric diameter of this set. -/
theorem ediam_le_of_forall_dist_le {C : ℝ} (h : ∀ x ∈ s, ∀ y ∈ s, dist x y ≤ C) :
    EMetric.diam s ≤ ENNReal.ofReal C :=
  EMetric.diam_le fun x hx y hy => (edist_dist x y).symm ▸ ENNReal.ofReal_le_ofReal (h x hx y hy)


/-- If the distance between any two points in a set is bounded by some non-negative constant,
this constant bounds the diameter. -/
theorem diam_le_of_forall_dist_le {C : ℝ} (h₀ : 0 ≤ C) (h : ∀ x ∈ s, ∀ y ∈ s, dist x y ≤ C) :
    diam s ≤ C :=
  ENNReal.toReal_le_of_le_ofReal h₀ (ediam_le_of_forall_dist_le h)


/-- If the distance between any two points in a nonempty set is bounded by some constant,
this constant bounds the diameter. -/
theorem diam_le_of_forall_dist_le_of_nonempty (hs : s.Nonempty) {C : ℝ}
    (h : ∀ x ∈ s, ∀ y ∈ s, dist x y ≤ C) : diam s ≤ C :=
  have h₀ : 0 ≤ C :=
    let ⟨x, hx⟩ := hs
    le_trans dist_nonneg (h x hx x hx)
  diam_le_of_forall_dist_le h₀ h


/-- The distance between two points in a set is controlled by the diameter of the set. -/
theorem dist_le_diam_of_mem' (h : EMetric.diam s ≠ ⊤) (hx : x ∈ s) (hy : y ∈ s) :
    dist x y ≤ diam s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Ne (EMetric.diam s) Top.top
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le (Dist.dist x y) (Metric.diam s)
  -/
  rw [diam, dist_edist]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    h : Ne (EMetric.diam s) Top.top
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le (EDist.edist x y).toReal (EMetric.diam s).toReal
  -/
  exact ENNReal.toReal_mono h <| EMetric.edist_le_diam_of_mem hx hy
  /-
    🎉 no goals
  -/


/-- Characterize the boundedness of a set in terms of the finiteness of its emetric.diameter. -/
theorem isBounded_iff_ediam_ne_top : IsBounded s ↔ EMetric.diam s ≠ ⊤ :=
  isBounded_iff.trans <| Iff.intro
    (fun ⟨_C, hC⟩ => ne_top_of_le_ne_top ENNReal.ofReal_ne_top <| ediam_le_of_forall_dist_le hC)
    fun h => ⟨diam s, fun _x hx _y hy => dist_le_diam_of_mem' h hx hy⟩


alias ⟨_root_.Bornology.IsBounded.ediam_ne_top, _⟩ := isBounded_iff_ediam_ne_top


theorem ediam_eq_top_iff_unbounded : EMetric.diam s = ⊤ ↔ ¬IsBounded s :=
  isBounded_iff_ediam_ne_top.not_left.symm


theorem ediam_univ_eq_top_iff_noncompact [ProperSpace α] :
    EMetric.diam (univ : Set α) = ∞ ↔ NoncompactSpace α := by
  rw [← not_compactSpace_iff, compactSpace_iff_isBounded_univ, isBounded_iff_ediam_ne_top,
    Classical.not_not]


@[simp]
theorem ediam_univ_of_noncompact [ProperSpace α] [NoncompactSpace α] :
    EMetric.diam (univ : Set α) = ∞ :=
  ediam_univ_eq_top_iff_noncompact.mpr ‹_›


@[simp]
theorem diam_univ_of_noncompact [ProperSpace α] [NoncompactSpace α] : diam (univ : Set α) = 0 := by
  /-
    α : Type u
    inst✝² : PseudoMetricSpace α
    inst✝¹ : ProperSpace α
    inst✝ : NoncompactSpace α
    ⊢ Eq (Metric.diam Set.univ) 0
  -/
  simp [diam]
  /-
    🎉 no goals
  -/


/-- The distance between two points in a set is controlled by the diameter of the set. -/
theorem dist_le_diam_of_mem (h : IsBounded s) (hx : x ∈ s) (hy : y ∈ s) : dist x y ≤ diam s :=
  dist_le_diam_of_mem' h.ediam_ne_top hx hy


theorem ediam_of_unbounded (h : ¬IsBounded s) : EMetric.diam s = ∞ := ediam_eq_top_iff_unbounded.2 h


/-- An unbounded set has zero diameter. If you would prefer to get the value ∞, use `EMetric.diam`.
This lemma makes it possible to avoid side conditions in some situations -/
theorem diam_eq_zero_of_unbounded (h : ¬IsBounded s) : diam s = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    h : Not (Bornology.IsBounded s)
    ⊢ Eq (Metric.diam s) 0
  -/
  rw [diam, ediam_of_unbounded h, ENNReal.top_toReal]
  /-
    🎉 no goals
  -/


/-- If `s ⊆ t`, then the diameter of `s` is bounded by that of `t`, provided `t` is bounded. -/
theorem diam_mono {s t : Set α} (h : s ⊆ t) (ht : IsBounded t) : diam s ≤ diam t :=
  ENNReal.toReal_mono ht.ediam_ne_top <| EMetric.diam_mono h


/-- The diameter of a union is controlled by the sum of the diameters, and the distance between
any two points in each of the sets. This lemma is true without any side condition, since it is
obviously true if `s ∪ t` is unbounded. -/
theorem diam_union {t : Set α} (xs : x ∈ s) (yt : y ∈ t) :
    diam (s ∪ t) ≤ diam s + dist x y + diam t := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x y : α
    t : Set α
    xs : Membership.mem s x
    yt : Membership.mem t y
    ⊢ LE.le (Metric.diam (Union.union s t)) (HAdd.hAdd (HAdd.hAdd (Metric.diam s)  …
  -/
  simp only [diam, dist_edist]
  refine (ENNReal.toReal_le_add' (EMetric.diam_union xs yt) ?_ ?_).trans
    (add_le_add_right ENNReal.toReal_add_le _)
    /-
      case refine_1
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x y : α
      t : Set α
      xs : Membership.mem s x
      yt : Membership.mem t y
      ⊢ Eq (HAdd.hAdd (EMetric.diam s) (EDist.edist x y)) Top.top → Eq (EMetric.diam …
    -/
  · simp only [ENNReal.add_eq_top, edist_ne_top, or_false]
    /-
      case refine_1
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x y : α
      t : Set α
      xs : Membership.mem s x
      yt : Membership.mem t y
      ⊢ Eq (EMetric.diam s) Top.top → Eq (EMetric.diam (Union.union s t)) Top.top
    -/
    exact fun h ↦ top_unique <| h ▸ EMetric.diam_mono subset_union_left
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x y : α
      t : Set α
      xs : Membership.mem s x
      yt : Membership.mem t y
      ⊢ Eq (EMetric.diam t) Top.top → Eq (EMetric.diam (Union.union s t)) Top.top
    -/
  · exact fun h ↦ top_unique <| h ▸ EMetric.diam_mono subset_union_right
    /-
      🎉 no goals
    -/


/-- If two sets intersect, the diameter of the union is bounded by the sum of the diameters. -/
theorem diam_union' {t : Set α} (h : (s ∩ t).Nonempty) : diam (s ∪ t) ≤ diam s + diam t := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    h : (Inter.inter s t).Nonempty
    ⊢ LE.le (Metric.diam (Union.union s t)) (HAdd.hAdd (Metric.diam s) (Metric.dia …
  -/
  rcases h with ⟨x, ⟨xs, xt⟩⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s t : Set α
    x : α
    xs : Membership.mem s x
    xt : Membership.mem t x
    ⊢ LE.le (Metric.diam (Union.union s t)) (HAdd.hAdd (Metric.diam s) (Metric.dia …
  -/
  simpa using diam_union xs xt
  /-
    🎉 no goals
  -/


theorem diam_le_of_subset_closedBall {r : ℝ} (hr : 0 ≤ r) (h : s ⊆ closedBall x r) :
    diam s ≤ 2 * r :=
  diam_le_of_forall_dist_le (mul_nonneg zero_le_two hr) fun a ha b hb =>
    calc
      dist a b ≤ dist a x + dist b x := dist_triangle_right _ _ _
      _ ≤ r + r := add_le_add (h ha) (h hb)
                      /-
                        α : Type u
                        inst✝ : PseudoMetricSpace α
                        s : Set α
                        x : α
                        r : Real
                        hr : LE.le 0 r
                        h : HasSubset.Subset s (Metric.closedBall x r)
                        a : α
                        ha : Membership.mem s a
                        b : α
                        hb : Membership.mem s b
                        ⊢ Eq (HAdd.hAdd r r) (HMul.hMul 2 r)
                      -/
      _ = 2 * r := by simp [mul_two, mul_comm]
                      /-
                        🎉 no goals
                      -/


/-- The diameter of a closed ball of radius `r` is at most `2 r`. -/
theorem diam_closedBall {r : ℝ} (h : 0 ≤ r) : diam (closedBall x r) ≤ 2 * r :=
  diam_le_of_subset_closedBall h Subset.rfl


/-- The diameter of a ball of radius `r` is at most `2 r`. -/
theorem diam_ball {r : ℝ} (h : 0 ≤ r) : diam (ball x r) ≤ 2 * r :=
  diam_le_of_subset_closedBall h ball_subset_closedBall


/-- If a family of complete sets with diameter tending to `0` is such that each finite intersection
is nonempty, then the total intersection is also nonempty. -/
theorem _root_.IsComplete.nonempty_iInter_of_nonempty_biInter {s : ℕ → Set α}
    (h0 : IsComplete (s 0)) (hs : ∀ n, IsClosed (s n)) (h's : ∀ n, IsBounded (s n))
    (h : ∀ N, (⋂ n ≤ N, s n).Nonempty) (h' : Tendsto (fun n => diam (s n)) atTop (𝓝 0)) :
    (⋂ n, s n).Nonempty := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Nat → Set α
    h0 : IsComplete (s 0)
    hs : ∀ (n : Nat), IsClosed (s n)
    h's : ∀ (n : Nat), Bornology.IsBounded (s n)
    h : ∀ (N : Nat), (Set.iInter fun n => Set.iInter fun h => s n).Nonempty
    h' : Filter.Tendsto (fun n => Metric.diam (s n)) Filter.atTop (nhds 0)
    ⊢ (Set.iInter fun n => s n).Nonempty
  -/
  let u N := (h N).some
  have I : ∀ n N, n ≤ N → u N ∈ s n := by
    intro n N hn
    apply mem_of_subset_of_mem _ (h N).choose_spec
    intro x hx
    simp only [mem_iInter] at hx
    exact hx n hn
  have : CauchySeq u := by
    apply cauchySeq_of_le_tendsto_0 _ _ h'
    intro m n N hm hn
    exact dist_le_diam_of_mem (h's N) (I _ _ hm) (I _ _ hn)
  obtain ⟨x, -, xlim⟩ : ∃ x ∈ s 0, Tendsto (fun n : ℕ => u n) atTop (𝓝 x) :=
    cauchySeq_tendsto_of_isComplete h0 (fun n => I 0 n (zero_le _)) this
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Nat → Set α
    h0 : IsComplete (s 0)
    hs : ∀ (n : Nat), IsClosed (s n)
    h's : ∀ (n : Nat), Bornology.IsBounded (s n)
    h : ∀ (N : Nat), (Set.iInter fun n => Set.iInter fun h => s n).Nonempty
    h' : Filter.Tendsto (fun n => Metric.diam (s n)) Filter.atTop (nhds 0)
    u : Nat → α := fun N => ⋯.some
    I : ∀ (n N : Nat), LE.le n N → Membership.mem (s n) (u N)
    this : CauchySeq u
    x : α
    xlim : Filter.Tendsto (fun n => u n) Filter.atTop (nhds x)
    ⊢ (Set.iInter fun n => s n).Nonempty
  -/
  refine ⟨x, mem_iInter.2 fun n => ?_⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Nat → Set α
    h0 : IsComplete (s 0)
    hs : ∀ (n : Nat), IsClosed (s n)
    h's : ∀ (n : Nat), Bornology.IsBounded (s n)
    h : ∀ (N : Nat), (Set.iInter fun n => Set.iInter fun h => s n).Nonempty
    h' : Filter.Tendsto (fun n => Metric.diam (s n)) Filter.atTop (nhds 0)
    u : Nat → α := fun N => ⋯.some
    I : ∀ (n N : Nat), LE.le n N → Membership.mem (s n) (u N)
    this : CauchySeq u
    x : α
    xlim : Filter.Tendsto (fun n => u n) Filter.atTop (nhds x)
    n : Nat
    ⊢ Membership.mem (s n) x
  -/
  apply (hs n).mem_of_tendsto xlim
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Nat → Set α
    h0 : IsComplete (s 0)
    hs : ∀ (n : Nat), IsClosed (s n)
    h's : ∀ (n : Nat), Bornology.IsBounded (s n)
    h : ∀ (N : Nat), (Set.iInter fun n => Set.iInter fun h => s n).Nonempty
    h' : Filter.Tendsto (fun n => Metric.diam (s n)) Filter.atTop (nhds 0)
    u : Nat → α := fun N => ⋯.some
    I : ∀ (n N : Nat), LE.le n N → Membership.mem (s n) (u N)
    this : CauchySeq u
    x : α
    xlim : Filter.Tendsto (fun n => u n) Filter.atTop (nhds x)
    n : Nat
    ⊢ Filter.Eventually (fun x => Membership.mem (s n) (u x)) Filter.atTop
  -/
  filter_upwards [Ici_mem_atTop n] with p hp
  /-
    case h
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Nat → Set α
    h0 : IsComplete (s 0)
    hs : ∀ (n : Nat), IsClosed (s n)
    h's : ∀ (n : Nat), Bornology.IsBounded (s n)
    h : ∀ (N : Nat), (Set.iInter fun n => Set.iInter fun h => s n).Nonempty
    h' : Filter.Tendsto (fun n => Metric.diam (s n)) Filter.atTop (nhds 0)
    u : Nat → α := fun N => ⋯.some
    I : ∀ (n N : Nat), LE.le n N → Membership.mem (s n) (u N)
    this : CauchySeq u
    x : α
    xlim : Filter.Tendsto (fun n => u n) Filter.atTop (nhds x)
    n p : Nat
    hp : Membership.mem (Set.Ici n) p
    ⊢ Membership.mem (s n) (u p)
  -/
  exact I n p hp
  /-
    🎉 no goals
  -/


/-- In a complete space, if a family of closed sets with diameter tending to `0` is such that each
finite intersection is nonempty, then the total intersection is also nonempty. -/
theorem nonempty_iInter_of_nonempty_biInter [CompleteSpace α] {s : ℕ → Set α}
    (hs : ∀ n, IsClosed (s n)) (h's : ∀ n, IsBounded (s n)) (h : ∀ N, (⋂ n ≤ N, s n).Nonempty)
    (h' : Tendsto (fun n => diam (s n)) atTop (𝓝 0)) : (⋂ n, s n).Nonempty :=
  (hs 0).isComplete.nonempty_iInter_of_nonempty_biInter hs h's h h'


/-- Extension for the `positivity` tactic: the diameter of a set is always nonnegative. -/
@[positivity Metric.diam _]
def evalDiam : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(@Metric.diam _ $inst $s) =>
    assertInstancesCommute
    pure (.nonnegative q(Metric.diam_nonneg))
  | _, _, _ => throwError "not ‖ · ‖"


theorem Metric.cobounded_eq_cocompact [ProperSpace α] : cobounded α = cocompact α := by
  /-
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    ⊢ Eq (Bornology.cobounded α) (Filter.cocompact α)
  -/
  nontriviality α; inhabit α
  exact cobounded_le_cocompact.antisymm <| (hasBasis_cobounded_compl_closedBall default).ge_iff.2
    fun _ _ ↦ (isCompact_closedBall _ _).compl_mem_cocompact


theorem tendsto_dist_right_cocompact_atTop [ProperSpace α] (x : α) :
    Tendsto (dist · x) (cocompact α) atTop :=
  (tendsto_dist_right_cobounded_atTop x).mono_left cobounded_eq_cocompact.ge


theorem tendsto_dist_left_cocompact_atTop [ProperSpace α] (x : α) :
    Tendsto (dist x) (cocompact α) atTop :=
  (tendsto_dist_left_cobounded_atTop x).mono_left cobounded_eq_cocompact.ge


theorem comap_dist_left_atTop_eq_cocompact [ProperSpace α] (x : α) :
                                             /-
                                               α : Type u
                                               inst✝¹ : PseudoMetricSpace α
                                               inst✝ : ProperSpace α
                                               x : α
                                               ⊢ Eq (Filter.comap (Dist.dist x) Filter.atTop) (Filter.cocompact α)
                                             -/
    comap (dist x) atTop = cocompact α := by simp [cobounded_eq_cocompact]
                                             /-
                                               🎉 no goals
                                             -/


theorem tendsto_cocompact_of_tendsto_dist_comp_atTop {f : β → α} {l : Filter β} (x : α)
    (h : Tendsto (fun y => dist (f y) x) l atTop) : Tendsto f l (cocompact α) :=
  ((tendsto_dist_right_atTop_iff _).1 h).mono_right cobounded_le_cocompact


theorem Metric.finite_isBounded_inter_isClosed [ProperSpace α] {K s : Set α} [DiscreteTopology s]
    (hK : IsBounded K) (hs : IsClosed s) : Set.Finite (K ∩ s) := by
  /-
    α : Type u
    inst✝² : PseudoMetricSpace α
    inst✝¹ : ProperSpace α
    K s : Set α
    inst✝ : DiscreteTopology ↑s
    hK : Bornology.IsBounded K
    hs : IsClosed s
    ⊢ (Inter.inter K s).Finite
  -/
  refine Set.Finite.subset (IsCompact.finite ?_ ?_) (Set.inter_subset_inter_left s subset_closure)
    /-
      case refine_1
      α : Type u
      inst✝² : PseudoMetricSpace α
      inst✝¹ : ProperSpace α
      K s : Set α
      inst✝ : DiscreteTopology ↑s
      hK : Bornology.IsBounded K
      hs : IsClosed s
      ⊢ IsCompact (Inter.inter (closure K) s)
    -/
  · exact hK.isCompact_closure.inter_right hs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      inst✝² : PseudoMetricSpace α
      inst✝¹ : ProperSpace α
      K s : Set α
      inst✝ : DiscreteTopology ↑s
      hK : Bornology.IsBounded K
      hs : IsClosed s
      ⊢ DiscreteTopology ↑(Inter.inter (closure K) s)
    -/
  · exact DiscreteTopology.of_subset inferInstance Set.inter_subset_right
    /-
      🎉 no goals
    -/

