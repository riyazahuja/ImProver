/-- The triangle (polygon) inequality for sequences of points; `Finset.Ico` version. -/
theorem dist_le_Ico_sum_dist (f : ℕ → α) {m n} (h : m ≤ n) :
    dist (f m) (f n) ≤ ∑ i ∈ Finset.Ico m n, dist (f i) (f (i + 1)) := by
  induction n, h using Nat.le_induction with
  | base => rw [Finset.Ico_self, Finset.sum_empty, dist_self]
  | succ n hle ihn =>
    calc
      dist (f m) (f (n + 1)) ≤ dist (f m) (f n) + dist (f n) (f (n + 1)) := dist_triangle _ _ _
      _ ≤ (∑ i ∈ Finset.Ico m n, _) + _ := add_le_add ihn le_rfl
      _ = ∑ i ∈ Finset.Ico m (n + 1), _ := by
      { rw [Nat.Ico_succ_right_eq_insert_Ico hle, Finset.sum_insert, add_comm]; simp }


/-- The triangle (polygon) inequality for sequences of points; `Finset.range` version. -/
theorem dist_le_range_sum_dist (f : ℕ → α) (n : ℕ) :
    dist (f 0) (f n) ≤ ∑ i ∈ Finset.range n, dist (f i) (f (i + 1)) :=
  Nat.Ico_zero_eq_range ▸ dist_le_Ico_sum_dist f (Nat.zero_le n)


/-- A version of `dist_le_Ico_sum_dist` with each intermediate distance replaced
with an upper estimate. -/
theorem dist_le_Ico_sum_of_dist_le {f : ℕ → α} {m n} (hmn : m ≤ n) {d : ℕ → ℝ}
    (hd : ∀ {k}, m ≤ k → k < n → dist (f k) (f (k + 1)) ≤ d k) :
    dist (f m) (f n) ≤ ∑ i ∈ Finset.Ico m n, d i :=
  le_trans (dist_le_Ico_sum_dist f hmn) <|
    Finset.sum_le_sum fun _k hk => hd (Finset.mem_Ico.1 hk).1 (Finset.mem_Ico.1 hk).2


/-- A version of `dist_le_range_sum_dist` with each intermediate distance replaced
with an upper estimate. -/
theorem dist_le_range_sum_of_dist_le {f : ℕ → α} (n : ℕ) {d : ℕ → ℝ}
    (hd : ∀ {k}, k < n → dist (f k) (f (k + 1)) ≤ d k) :
    dist (f 0) (f n) ≤ ∑ i ∈ Finset.range n, d i :=
  Nat.Ico_zero_eq_range ▸ dist_le_Ico_sum_of_dist_le (zero_le n) fun _ => hd


nonrec theorem isUniformInducing_iff [PseudoMetricSpace β] {f : α → β} :
    IsUniformInducing f ↔ UniformContinuous f ∧
      ∀ δ > 0, ∃ ε > 0, ∀ {a b : α}, dist (f a) (f b) < ε → dist a b < δ :=
  isUniformInducing_iff'.trans <| Iff.rfl.and <|
    ((uniformity_basis_dist.comap _).le_basis_iff uniformity_basis_dist).trans <| by
      /-
        α : Type u
        β : Type v
        inst✝¹ : PseudoMetricSpace α
        inst✝ : PseudoMetricSpace β
        f : α → β
        ⊢ Iff (∀ (i' : Real), LT.lt 0 i' → Exists fun i => And (LT.lt 0 i) (HasSubset. …
      -/
      simp only [subset_def, Prod.forall, gt_iff_lt, preimage_setOf_eq, Prod.map_apply, mem_setOf]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-05")]
alias uniformInducing_iff := isUniformInducing_iff


nonrec theorem isUniformEmbedding_iff [PseudoMetricSpace β] {f : α → β} :
    IsUniformEmbedding f ↔ Function.Injective f ∧ UniformContinuous f ∧
      ∀ δ > 0, ∃ ε > 0, ∀ {a b : α}, dist (f a) (f b) < ε → dist a b < δ := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    ⊢ Iff (IsUniformEmbedding f) (And (Function.Injective f) (And (UniformContinuo …
  -/
  rw [isUniformEmbedding_iff, and_comm, isUniformInducing_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_iff := isUniformEmbedding_iff


/-- If a map between pseudometric spaces is a uniform embedding then the distance between `f x`
and `f y` is controlled in terms of the distance between `x` and `y`. -/
theorem controlled_of_isUniformEmbedding [PseudoMetricSpace β] {f : α → β}
    (h : IsUniformEmbedding f) :
    (∀ ε > 0, ∃ δ > 0, ∀ {a b : α}, dist a b < δ → dist (f a) (f b) < ε) ∧
      ∀ δ > 0, ∃ ε > 0, ∀ {a b : α}, dist (f a) (f b) < ε → dist a b < δ :=
  ⟨uniformContinuous_iff.1 h.uniformContinuous, (isUniformEmbedding_iff.1 h).2.2⟩


@[deprecated (since := "2024-10-01")]
alias controlled_of_uniformEmbedding := controlled_of_isUniformEmbedding


theorem totallyBounded_iff {s : Set α} :
    TotallyBounded s ↔ ∀ ε > 0, ∃ t : Set α, t.Finite ∧ s ⊆ ⋃ y ∈ t, ball y ε :=
  uniformity_basis_dist.totallyBounded_iff


/-- A pseudometric space is totally bounded if one can reconstruct up to any ε>0 any element of the
space from finitely many data. -/
theorem totallyBounded_of_finite_discretization {s : Set α}
    (H : ∀ ε > (0 : ℝ),
        ∃ (β : Type u) (_ : Fintype β) (F : s → β), ∀ x y, F x = F y → dist (x : α) y < ε) :
    TotallyBounded s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    ⊢ TotallyBounded s
  -/
  rcases s.eq_empty_or_nonempty with hs | hs
    /-
      case inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
      hs : Eq s EmptyCollection.emptyCollection
      ⊢ TotallyBounded s
    -/
  · rw [hs]
    /-
      case inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
      hs : Eq s EmptyCollection.emptyCollection
      ⊢ TotallyBounded EmptyCollection.emptyCollection
    -/
    exact totallyBounded_empty
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    hs : s.Nonempty
    ⊢ TotallyBounded s
  -/
  rcases hs with ⟨x0, hx0⟩
  /-
    case inr.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    ⊢ TotallyBounded s
  -/
  haveI : Inhabited s := ⟨⟨x0, hx0⟩⟩
  /-
    case inr.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    this : Inhabited ↑s
    ⊢ TotallyBounded s
  -/
  refine totallyBounded_iff.2 fun ε ε0 => ?_
  /-
    case inr.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    this : Inhabited ↑s
    ε : Real
    ε0 : GT.gt ε 0
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset s (Set.iUnion fun y => Set.iU …
  -/
  rcases H ε ε0 with ⟨β, fβ, F, hF⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    this : Inhabited ↑s
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u
    fβ : Fintype β
    F : ↑s → β
    hF : ∀ (x y : ↑s), Eq (F x) (F y) → LT.lt (Dist.dist ↑x ↑y) ε
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset s (Set.iUnion fun y => Set.iU …
  -/
  let Finv := Function.invFun F
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    this : Inhabited ↑s
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u
    fβ : Fintype β
    F : ↑s → β
    hF : ∀ (x y : ↑s), Eq (F x) (F y) → LT.lt (Dist.dist ↑x ↑y) ε
    Finv : β → ↑s := Function.invFun F
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset s (Set.iUnion fun y => Set.iU …
  -/
  refine ⟨range (Subtype.val ∘ Finv), finite_range _, fun x xs => ?_⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    this : Inhabited ↑s
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u
    fβ : Fintype β
    F : ↑s → β
    hF : ∀ (x y : ↑s), Eq (F x) (F y) → LT.lt (Dist.dist ↑x ↑y) ε
    Finv : β → ↑s := Function.invFun F
    x : α
    xs : Membership.mem s x
    ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y ε) x
  -/
  let x' := Finv (F ⟨x, xs⟩)
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    this : Inhabited ↑s
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u
    fβ : Fintype β
    F : ↑s → β
    hF : ∀ (x y : ↑s), Eq (F x) (F y) → LT.lt (Dist.dist ↑x ↑y) ε
    Finv : β → ↑s := Function.invFun F
    x : α
    xs : Membership.mem s x
    x' : ↑s := Finv (F ⟨x, xs⟩)
    ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y ε) x
  -/
  have : F x' = F ⟨x, xs⟩ := Function.invFun_eq ⟨⟨x, xs⟩, rfl⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    this✝ : Inhabited ↑s
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u
    fβ : Fintype β
    F : ↑s → β
    hF : ∀ (x y : ↑s), Eq (F x) (F y) → LT.lt (Dist.dist ↑x ↑y) ε
    Finv : β → ↑s := Function.invFun F
    x : α
    xs : Membership.mem s x
    x' : ↑s := Finv (F ⟨x, xs⟩)
    this : Eq (F x') (F ⟨x, xs⟩)
    ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y ε) x
  -/
  simp only [Set.mem_iUnion, Set.mem_range]
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    x0 : α
    hx0 : Membership.mem s x0
    this✝ : Inhabited ↑s
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u
    fβ : Fintype β
    F : ↑s → β
    hF : ∀ (x y : ↑s), Eq (F x) (F y) → LT.lt (Dist.dist ↑x ↑y) ε
    Finv : β → ↑s := Function.invFun F
    x : α
    xs : Membership.mem s x
    x' : ↑s := Finv (F ⟨x, xs⟩)
    this : Eq (F x') (F ⟨x, xs⟩)
    ⊢ Exists fun i => Exists fun i_1 => Membership.mem (Metric.ball i ε) x
  -/
  exact ⟨_, ⟨F ⟨x, xs⟩, rfl⟩, hF _ _ this.symm⟩
  /-
    🎉 no goals
  -/


theorem finite_approx_of_totallyBounded {s : Set α} (hs : TotallyBounded s) :
    ∀ ε > 0, ∃ t, t ⊆ s ∧ Set.Finite t ∧ s ⊆ ⋃ y ∈ t, ball y ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : TotallyBounded s
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun t => And (HasSubset.Subset t s) (And t. …
  -/
  intro ε ε_pos
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : TotallyBounded s
    ε : Real
    ε_pos : GT.gt ε 0
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (HasSubset.Subset s …
  -/
  rw [totallyBounded_iff_subset] at hs
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : ∀ (d : Set (Prod α α)), Membership.mem (uniformity α) d → Exists fun t => …
    ε : Real
    ε_pos : GT.gt ε 0
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (HasSubset.Subset s …
  -/
  exact hs _ (dist_mem_uniformity ε_pos)
  /-
    🎉 no goals
  -/


/-- Expressing uniform convergence using `dist` -/
theorem tendstoUniformlyOnFilter_iff {F : ι → β → α} {f : β → α} {p : Filter ι} {p' : Filter β} :
    TendstoUniformlyOnFilter F f p p' ↔
      ∀ ε > 0, ∀ᶠ n : ι × β in p ×ˢ p', dist (f n.snd) (F n.fst n.snd) < ε := by
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝ : PseudoMetricSpace α
    F : ι → β → α
    f : β → α
    p : Filter ι
    p' : Filter β
    ⊢ Iff (TendstoUniformlyOnFilter F f p p') (∀ (ε : Real), GT.gt ε 0 → Filter.Ev …
  -/
  refine ⟨fun H ε hε => H _ (dist_mem_uniformity hε), fun H u hu => ?_⟩
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝ : PseudoMetricSpace α
    F : ι → β → α
    f : β → α
    p : Filter ι
    p' : Filter β
    H : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist (f  …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    ⊢ Filter.Eventually (fun n => Membership.mem u { fst := f n.2, snd := F n.1 n. …
  -/
  rcases mem_uniformity_dist.1 hu with ⟨ε, εpos, hε⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝ : PseudoMetricSpace α
    F : ι → β → α
    f : β → α
    p : Filter ι
    p' : Filter β
    H : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => LT.lt (Dist.dist (f  …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    ε : Real
    εpos : GT.gt ε 0
    hε : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd : …
    ⊢ Filter.Eventually (fun n => Membership.mem u { fst := f n.2, snd := F n.1 n. …
  -/
  exact (H ε εpos).mono fun n hn => hε hn
  /-
    🎉 no goals
  -/


/-- Expressing locally uniform convergence on a set using `dist`. -/
theorem tendstoLocallyUniformlyOn_iff [TopologicalSpace β] {F : ι → β → α} {f : β → α}
    {p : Filter ι} {s : Set β} :
    TendstoLocallyUniformlyOn F f p s ↔
      ∀ ε > 0, ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, ∀ᶠ n in p, ∀ y ∈ t, dist (f y) (F n y) < ε := by
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    ⊢ Iff (TendstoLocallyUniformlyOn F f p s) (∀ (ε : Real), GT.gt ε 0 → ∀ (x : β) …
  -/
  refine ⟨fun H ε hε => H _ (dist_mem_uniformity hε), fun H u hu x hx => ?_⟩
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : Real), GT.gt ε 0 → ∀ (x : β), Membership.mem s x → Exists fun t =>  …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    x : β
    hx : Membership.mem s x
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  rcases mem_uniformity_dist.1 hu with ⟨ε, εpos, hε⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : Real), GT.gt ε 0 → ∀ (x : β), Membership.mem s x → Exists fun t =>  …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    x : β
    hx : Membership.mem s x
    ε : Real
    εpos : GT.gt ε 0
    hε : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd : …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  rcases H ε εpos x hx with ⟨t, ht, Ht⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : Real), GT.gt ε 0 → ∀ (x : β), Membership.mem s x → Exists fun t =>  …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    x : β
    hx : Membership.mem s x
    ε : Real
    εpos : GT.gt ε 0
    hε : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd : …
    t : Set β
    ht : Membership.mem (nhdsWithin x s) t
    Ht : Filter.Eventually (fun n => ∀ (y : β), Membership.mem t y → LT.lt (Dist.d …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  exact ⟨t, ht, Ht.mono fun n hs x hx => hε (hs x hx)⟩
  /-
    🎉 no goals
  -/


/-- Expressing uniform convergence on a set using `dist`. -/
theorem tendstoUniformlyOn_iff {F : ι → β → α} {f : β → α} {p : Filter ι} {s : Set β} :
    TendstoUniformlyOn F f p s ↔ ∀ ε > 0, ∀ᶠ n in p, ∀ x ∈ s, dist (f x) (F n x) < ε := by
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝ : PseudoMetricSpace α
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    ⊢ Iff (TendstoUniformlyOn F f p s) (∀ (ε : Real), GT.gt ε 0 → Filter.Eventuall …
  -/
  refine ⟨fun H ε hε => H _ (dist_mem_uniformity hε), fun H u hu => ?_⟩
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝ : PseudoMetricSpace α
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : β), Membershi …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    ⊢ Filter.Eventually (fun n => ∀ (x : β), Membership.mem s x → Membership.mem u …
  -/
  rcases mem_uniformity_dist.1 hu with ⟨ε, εpos, hε⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝ : PseudoMetricSpace α
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : β), Membershi …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    ε : Real
    εpos : GT.gt ε 0
    hε : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd : …
    ⊢ Filter.Eventually (fun n => ∀ (x : β), Membership.mem s x → Membership.mem u …
  -/
  exact (H ε εpos).mono fun n hs x hx => hε (hs x hx)
  /-
    🎉 no goals
  -/


/-- Expressing locally uniform convergence using `dist`. -/
theorem tendstoLocallyUniformly_iff [TopologicalSpace β] {F : ι → β → α} {f : β → α}
    {p : Filter ι} :
    TendstoLocallyUniformly F f p ↔
      ∀ ε > 0, ∀ x : β, ∃ t ∈ 𝓝 x, ∀ᶠ n in p, ∀ y ∈ t, dist (f y) (F n y) < ε := by
  simp only [← tendstoLocallyUniformlyOn_univ, tendstoLocallyUniformlyOn_iff, nhdsWithin_univ,
    mem_univ, forall_const, exists_prop]


/-- Expressing uniform convergence using `dist`. -/
theorem tendstoUniformly_iff {F : ι → β → α} {f : β → α} {p : Filter ι} :
    TendstoUniformly F f p ↔ ∀ ε > 0, ∀ᶠ n in p, ∀ x, dist (f x) (F n x) < ε := by
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝ : PseudoMetricSpace α
    F : ι → β → α
    f : β → α
    p : Filter ι
    ⊢ Iff (TendstoUniformly F f p) (∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (f …
  -/
  rw [← tendstoUniformlyOn_univ, tendstoUniformlyOn_iff]
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    inst✝ : PseudoMetricSpace α
    F : ι → β → α
    f : β → α
    p : Filter ι
    ⊢ Iff (∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : β), Member …
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem cauchy_iff {f : Filter α} :
    Cauchy f ↔ NeBot f ∧ ∀ ε > 0, ∃ t ∈ f, ∀ x ∈ t, ∀ y ∈ t, dist x y < ε :=
  uniformity_basis_dist.cauchy_iff


/-- Given a point `x` in a discrete subset `s` of a pseudometric space, there is an open ball
centered at `x` and intersecting `s` only at `x`. -/
theorem exists_ball_inter_eq_singleton_of_mem_discrete [DiscreteTopology s] {x : α} (hx : x ∈ s) :
    ∃ ε > 0, Metric.ball x ε ∩ s = {x} :=
  nhds_basis_ball.exists_inter_eq_singleton_of_mem_discrete hx


/-- Given a point `x` in a discrete subset `s` of a pseudometric space, there is a closed ball
of positive radius centered at `x` and intersecting `s` only at `x`. -/
theorem exists_closedBall_inter_eq_singleton_of_discrete [DiscreteTopology s] {x : α} (hx : x ∈ s) :
    ∃ ε > 0, Metric.closedBall x ε ∩ s = {x} :=
  nhds_basis_closedBall.exists_inter_eq_singleton_of_mem_discrete hx


theorem Metric.inseparable_iff_nndist {x y : α} : Inseparable x y ↔ nndist x y = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Iff (Inseparable x y) (Eq (NNDist.nndist x y) 0)
  -/
  rw [EMetric.inseparable_iff, edist_nndist, ENNReal.coe_eq_zero]
  /-
    🎉 no goals
  -/


alias ⟨Inseparable.nndist_eq_zero, _⟩ := Metric.inseparable_iff_nndist


theorem Metric.inseparable_iff {x y : α} : Inseparable x y ↔ dist x y = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Iff (Inseparable x y) (Eq (Dist.dist x y) 0)
  -/
  rw [Metric.inseparable_iff_nndist, dist_nndist, NNReal.coe_eq_zero]
  /-
    🎉 no goals
  -/


alias ⟨Inseparable.dist_eq_zero, _⟩ := Metric.inseparable_iff


/-- A weaker version of `tendsto_nhds_unique` for `PseudoMetricSpace`. -/
theorem tendsto_nhds_unique_dist {f : β → α} {l : Filter β} {x y : α} [NeBot l]
    (ha : Tendsto f l (𝓝 x)) (hb : Tendsto f l (𝓝 y)) : dist x y = 0 :=
  (tendsto_nhds_unique_inseparable ha hb).dist_eq_zero


theorem cauchySeq_iff_tendsto_dist_atTop_0 [Nonempty β] [SemilatticeSup β] {u : β → α} :
    CauchySeq u ↔ Tendsto (fun n : β × β => dist (u n.1) (u n.2)) atTop (𝓝 0) := by
  rw [cauchySeq_iff_tendsto, Metric.uniformity_eq_comap_nhds_zero, tendsto_comap_iff,
    Function.comp_def]
  /-
    α : Type u
    β : Type v
    inst✝² : PseudoMetricSpace α
    inst✝¹ : Nonempty β
    inst✝ : SemilatticeSup β
    u : β → α
    ⊢ Iff (Filter.Tendsto (fun x => Dist.dist (Prod.map u u x).1 (Prod.map u u x). …
  -/
  simp_rw [Prod.map_fst, Prod.map_snd]
  /-
    🎉 no goals
  -/


/-- The preimage of a separable set by an inducing map is separable. -/
protected lemma IsInducing.isSeparable_preimage {f : β → α} [TopologicalSpace β]
    (hf : IsInducing f) {s : Set α} (hs : IsSeparable s) : IsSeparable (f ⁻¹' s) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    f : β → α
    inst✝ : TopologicalSpace β
    hf : Topology.IsInducing f
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    ⊢ TopologicalSpace.IsSeparable (Set.preimage f s)
  -/
  have : SeparableSpace s := hs.separableSpace
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    f : β → α
    inst✝ : TopologicalSpace β
    hf : Topology.IsInducing f
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    this : TopologicalSpace.SeparableSpace ↑s
    ⊢ TopologicalSpace.IsSeparable (Set.preimage f s)
  -/
  have : SecondCountableTopology s := UniformSpace.secondCountable_of_separable _
  have : IsInducing ((mapsTo_preimage f s).restrict _ _ _) :=
    (hf.comp IsInducing.subtypeVal).codRestrict _
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    f : β → α
    inst✝ : TopologicalSpace β
    hf : Topology.IsInducing f
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    this✝¹ : TopologicalSpace.SeparableSpace ↑s
    this✝ : SecondCountableTopology ↑s
    this : Topology.IsInducing (Set.MapsTo.restrict f (Set.preimage f s) s ⋯)
    ⊢ TopologicalSpace.IsSeparable (Set.preimage f s)
  -/
  have := this.secondCountableTopology
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    f : β → α
    inst✝ : TopologicalSpace β
    hf : Topology.IsInducing f
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    this✝² : TopologicalSpace.SeparableSpace ↑s
    this✝¹ : SecondCountableTopology ↑s
    this✝ : Topology.IsInducing (Set.MapsTo.restrict f (Set.preimage f s) s ⋯)
    this : SecondCountableTopology ↑(Set.preimage f s)
    ⊢ TopologicalSpace.IsSeparable (Set.preimage f s)
  -/
  exact .of_subtype _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias _root_.Inducing.isSeparable_preimage := IsInducing.isSeparable_preimage


protected theorem IsEmbedding.isSeparable_preimage {f : β → α} [TopologicalSpace β]
    (hf : IsEmbedding f) {s : Set α} (hs : IsSeparable s) : IsSeparable (f ⁻¹' s) :=
  hf.isInducing.isSeparable_preimage hs


@[deprecated (since := "2024-10-26")]
alias _root_.Embedding.isSeparable_preimage := IsEmbedding.isSeparable_preimage


/-- A compact set is separable. -/
theorem IsCompact.isSeparable {s : Set α} (hs : IsCompact s) : IsSeparable s :=
  haveI : CompactSpace s := isCompact_iff_compactSpace.mp hs
  .of_subtype s


/-- A pseudometric space is second countable if, for every `ε > 0`, there is a countable set which
is `ε`-dense. -/
theorem secondCountable_of_almost_dense_set
    (H : ∀ ε > (0 : ℝ), ∃ s : Set α, s.Countable ∧ ∀ x, ∃ y ∈ s, dist x y ≤ ε) :
    SecondCountableTopology α := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun s => And s.Countable (∀ (x : α), Exis …
    ⊢ SecondCountableTopology α
  -/
  refine EMetric.secondCountable_of_almost_dense_set fun ε ε0 => ?_
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun s => And s.Countable (∀ (x : α), Exis …
    ε : ENNReal
    ε0 : GT.gt ε 0
    ⊢ Exists fun t => And t.Countable (Eq (Set.iUnion fun x => Set.iUnion fun h => …
  -/
  rcases ENNReal.lt_iff_exists_nnreal_btwn.1 ε0 with ⟨ε', ε'0, ε'ε⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun s => And s.Countable (∀ (x : α), Exis …
    ε : ENNReal
    ε0 : GT.gt ε 0
    ε' : NNReal
    ε'0 : LT.lt 0 ↑ε'
    ε'ε : LT.lt (↑ε') ε
    ⊢ Exists fun t => And t.Countable (Eq (Set.iUnion fun x => Set.iUnion fun h => …
  -/
  choose s hsc y hys hyx using H ε' (mod_cast ε'0)
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun s => And s.Countable (∀ (x : α), Exis …
    ε : ENNReal
    ε0 : GT.gt ε 0
    ε' : NNReal
    ε'0 : LT.lt 0 ↑ε'
    ε'ε : LT.lt (↑ε') ε
    s : Set α
    hsc : s.Countable
    y : α → α
    hys : ∀ (x : α), Membership.mem s (y x)
    hyx : ∀ (x : α), LE.le (Dist.dist x (y x)) ↑ε'
    ⊢ Exists fun t => And t.Countable (Eq (Set.iUnion fun x => Set.iUnion fun h => …
  -/
  refine ⟨s, hsc, iUnion₂_eq_univ_iff.2 fun x => ⟨y x, hys _, le_trans ?_ ε'ε.le⟩⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun s => And s.Countable (∀ (x : α), Exis …
    ε : ENNReal
    ε0 : GT.gt ε 0
    ε' : NNReal
    ε'0 : LT.lt 0 ↑ε'
    ε'ε : LT.lt (↑ε') ε
    s : Set α
    hsc : s.Countable
    y : α → α
    hys : ∀ (x : α), Membership.mem s (y x)
    hyx : ∀ (x : α), LE.le (Dist.dist x (y x)) ↑ε'
    x : α
    ⊢ LE.le (EDist.edist x (y x)) ↑ε'
  -/
  exact mod_cast hyx x
  /-
    🎉 no goals
  -/


