/-- The triangle (polygon) inequality for sequences of points; `Finset.Ico` version. -/
theorem edist_le_Ico_sum_edist (f : ℕ → α) {m n} (h : m ≤ n) :
    edist (f m) (f n) ≤ ∑ i ∈ Finset.Ico m n, edist (f i) (f (i + 1)) := by
  induction n, h using Nat.le_induction with
  | base => rw [Finset.Ico_self, Finset.sum_empty, edist_self]
  | succ n hle ihn =>
    calc
      edist (f m) (f (n + 1)) ≤ edist (f m) (f n) + edist (f n) (f (n + 1)) := edist_triangle _ _ _
      _ ≤ (∑ i ∈ Finset.Ico m n, _) + _ := add_le_add ihn le_rfl
      _ = ∑ i ∈ Finset.Ico m (n + 1), _ := by
      { rw [Nat.Ico_succ_right_eq_insert_Ico hle, Finset.sum_insert, add_comm]; simp }


/-- The triangle (polygon) inequality for sequences of points; `Finset.range` version. -/
theorem edist_le_range_sum_edist (f : ℕ → α) (n : ℕ) :
    edist (f 0) (f n) ≤ ∑ i ∈ Finset.range n, edist (f i) (f (i + 1)) :=
  Nat.Ico_zero_eq_range ▸ edist_le_Ico_sum_edist f (Nat.zero_le n)


/-- A version of `edist_le_Ico_sum_edist` with each intermediate distance replaced
with an upper estimate. -/
theorem edist_le_Ico_sum_of_edist_le {f : ℕ → α} {m n} (hmn : m ≤ n) {d : ℕ → ℝ≥0∞}
    (hd : ∀ {k}, m ≤ k → k < n → edist (f k) (f (k + 1)) ≤ d k) :
    edist (f m) (f n) ≤ ∑ i ∈ Finset.Ico m n, d i :=
  le_trans (edist_le_Ico_sum_edist f hmn) <|
    Finset.sum_le_sum fun _k hk => hd (Finset.mem_Ico.1 hk).1 (Finset.mem_Ico.1 hk).2


/-- A version of `edist_le_range_sum_edist` with each intermediate distance replaced
with an upper estimate. -/
theorem edist_le_range_sum_of_edist_le {f : ℕ → α} (n : ℕ) {d : ℕ → ℝ≥0∞}
    (hd : ∀ {k}, k < n → edist (f k) (f (k + 1)) ≤ d k) :
    edist (f 0) (f n) ≤ ∑ i ∈ Finset.range n, d i :=
  Nat.Ico_zero_eq_range ▸ edist_le_Ico_sum_of_edist_le (zero_le n) fun _ => hd


theorem isUniformInducing_iff [PseudoEMetricSpace β] {f : α → β} :
    IsUniformInducing f ↔ UniformContinuous f ∧
      ∀ δ > 0, ∃ ε > 0, ∀ {a b : α}, edist (f a) (f b) < ε → edist a b < δ :=
  isUniformInducing_iff'.trans <| Iff.rfl.and <|
    ((uniformity_basis_edist.comap _).le_basis_iff uniformity_basis_edist).trans <| by
      /-
        α : Type u
        β : Type v
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : PseudoEMetricSpace β
        f : α → β
        ⊢ Iff (∀ (i' : ENNReal), LT.lt 0 i' → Exists fun i => And (LT.lt 0 i) (HasSubs …
      -/
      simp only [subset_def, Prod.forall]; rfl
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-10-05")]
alias uniformInducing_iff := isUniformInducing_iff


/-- ε-δ characterization of uniform embeddings on pseudoemetric spaces -/
nonrec theorem isUniformEmbedding_iff [PseudoEMetricSpace β] {f : α → β} :
    IsUniformEmbedding f ↔ Function.Injective f ∧ UniformContinuous f ∧
      ∀ δ > 0, ∃ ε > 0, ∀ {a b : α}, edist (f a) (f b) < ε → edist a b < δ :=
  (isUniformEmbedding_iff _).trans <| and_comm.trans <| Iff.rfl.and isUniformInducing_iff


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_iff := isUniformEmbedding_iff


/-- If a map between pseudoemetric spaces is a uniform embedding then the edistance between `f x`
and `f y` is controlled in terms of the distance between `x` and `y`.

In fact, this lemma holds for a `IsUniformInducing` map.
TODO: generalize? -/
theorem controlled_of_isUniformEmbedding [PseudoEMetricSpace β] {f : α → β}
    (h : IsUniformEmbedding f) :
    (∀ ε > 0, ∃ δ > 0, ∀ {a b : α}, edist a b < δ → edist (f a) (f b) < ε) ∧
      ∀ δ > 0, ∃ ε > 0, ∀ {a b : α}, edist (f a) (f b) < ε → edist a b < δ :=
  ⟨uniformContinuous_iff.1 h.uniformContinuous, (isUniformEmbedding_iff.1 h).2.2⟩


@[deprecated (since := "2024-10-01")]
alias controlled_of_uniformEmbedding := controlled_of_isUniformEmbedding


/-- ε-δ characterization of Cauchy sequences on pseudoemetric spaces -/
protected theorem cauchy_iff {f : Filter α} :
    Cauchy f ↔ f ≠ ⊥ ∧ ∀ ε > 0, ∃ t ∈ f, ∀ x, x ∈ t → ∀ y, y ∈ t → edist x y < ε := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f : Filter α
    ⊢ Iff (Cauchy f) (And (Ne f Bot.bot) (∀ (ε : ENNReal), GT.gt ε 0 → Exists fun  …
  -/
  rw [← neBot_iff]; exact uniformity_basis_edist.cauchy_iff
                    /-
                      🎉 no goals
                    -/


/-- A very useful criterion to show that a space is complete is to show that all sequences
which satisfy a bound of the form `edist (u n) (u m) < B N` for all `n m ≥ N` are
converging. This is often applied for `B N = 2^{-N}`, i.e., with a very fast convergence to
`0`, which makes it possible to use arguments of converging series, while this is impossible
to do in general for arbitrary Cauchy sequences. -/
theorem complete_of_convergent_controlled_sequences (B : ℕ → ℝ≥0∞) (hB : ∀ n, 0 < B n)
    (H : ∀ u : ℕ → α, (∀ N n m : ℕ, N ≤ n → N ≤ m → edist (u n) (u m) < B N) →
      ∃ x, Tendsto u atTop (𝓝 x)) :
    CompleteSpace α :=
  UniformSpace.complete_of_convergent_controlled_sequences
    (fun n => { p : α × α | edist p.1 p.2 < B n }) (fun n => edist_mem_uniformity <| hB n) H


/-- A sequentially complete pseudoemetric space is complete. -/
theorem complete_of_cauchySeq_tendsto :
    (∀ u : ℕ → α, CauchySeq u → ∃ a, Tendsto u atTop (𝓝 a)) → CompleteSpace α :=
  UniformSpace.complete_of_cauchySeq_tendsto


/-- Expressing locally uniform convergence on a set using `edist`. -/
theorem tendstoLocallyUniformlyOn_iff {ι : Type*} [TopologicalSpace β] {F : ι → β → α} {f : β → α}
    {p : Filter ι} {s : Set β} :
    TendstoLocallyUniformlyOn F f p s ↔
      ∀ ε > 0, ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, ∀ᶠ n in p, ∀ y ∈ t, edist (f y) (F n y) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    ι : Type u_2
    inst✝ : TopologicalSpace β
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    ⊢ Iff (TendstoLocallyUniformlyOn F f p s) (∀ (ε : ENNReal), GT.gt ε 0 → ∀ (x : …
  -/
  refine ⟨fun H ε hε => H _ (edist_mem_uniformity hε), fun H u hu x hx => ?_⟩
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    ι : Type u_2
    inst✝ : TopologicalSpace β
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : ENNReal), GT.gt ε 0 → ∀ (x : β), Membership.mem s x → Exists fun t  …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    x : β
    hx : Membership.mem s x
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  rcases mem_uniformity_edist.1 hu with ⟨ε, εpos, hε⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    ι : Type u_2
    inst✝ : TopologicalSpace β
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : ENNReal), GT.gt ε 0 → ∀ (x : β), Membership.mem s x → Exists fun t  …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    x : β
    hx : Membership.mem s x
    ε : ENNReal
    εpos : GT.gt ε 0
    hε : ∀ {a b : α}, LT.lt (EDist.edist a b) ε → Membership.mem u { fst := a, snd …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  rcases H ε εpos x hx with ⟨t, ht, Ht⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    ι : Type u_2
    inst✝ : TopologicalSpace β
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : ENNReal), GT.gt ε 0 → ∀ (x : β), Membership.mem s x → Exists fun t  …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    x : β
    hx : Membership.mem s x
    ε : ENNReal
    εpos : GT.gt ε 0
    hε : ∀ {a b : α}, LT.lt (EDist.edist a b) ε → Membership.mem u { fst := a, snd …
    t : Set β
    ht : Membership.mem (nhdsWithin x s) t
    Ht : Filter.Eventually (fun n => ∀ (y : β), Membership.mem t y → LT.lt (EDist. …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  exact ⟨t, ht, Ht.mono fun n hs x hx => hε (hs x hx)⟩
  /-
    🎉 no goals
  -/


/-- Expressing uniform convergence on a set using `edist`. -/
theorem tendstoUniformlyOn_iff {ι : Type*} {F : ι → β → α} {f : β → α} {p : Filter ι} {s : Set β} :
    TendstoUniformlyOn F f p s ↔ ∀ ε > 0, ∀ᶠ n in p, ∀ x ∈ s, edist (f x) (F n x) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝ : PseudoEMetricSpace α
    ι : Type u_2
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    ⊢ Iff (TendstoUniformlyOn F f p s) (∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventu …
  -/
  refine ⟨fun H ε hε => H _ (edist_mem_uniformity hε), fun H u hu => ?_⟩
  /-
    α : Type u
    β : Type v
    inst✝ : PseudoEMetricSpace α
    ι : Type u_2
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : β), Member …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    ⊢ Filter.Eventually (fun n => ∀ (x : β), Membership.mem s x → Membership.mem u …
  -/
  rcases mem_uniformity_edist.1 hu with ⟨ε, εpos, hε⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝ : PseudoEMetricSpace α
    ι : Type u_2
    F : ι → β → α
    f : β → α
    p : Filter ι
    s : Set β
    H : ∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : β), Member …
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    ε : ENNReal
    εpos : GT.gt ε 0
    hε : ∀ {a b : α}, LT.lt (EDist.edist a b) ε → Membership.mem u { fst := a, snd …
    ⊢ Filter.Eventually (fun n => ∀ (x : β), Membership.mem s x → Membership.mem u …
  -/
  exact (H ε εpos).mono fun n hs x hx => hε (hs x hx)
  /-
    🎉 no goals
  -/


/-- Expressing locally uniform convergence using `edist`. -/
theorem tendstoLocallyUniformly_iff {ι : Type*} [TopologicalSpace β] {F : ι → β → α} {f : β → α}
    {p : Filter ι} :
    TendstoLocallyUniformly F f p ↔
      ∀ ε > 0, ∀ x : β, ∃ t ∈ 𝓝 x, ∀ᶠ n in p, ∀ y ∈ t, edist (f y) (F n y) < ε := by
  simp only [← tendstoLocallyUniformlyOn_univ, tendstoLocallyUniformlyOn_iff, mem_univ,
    forall_const, exists_prop, nhdsWithin_univ]


/-- Expressing uniform convergence using `edist`. -/
theorem tendstoUniformly_iff {ι : Type*} {F : ι → β → α} {f : β → α} {p : Filter ι} :
    TendstoUniformly F f p ↔ ∀ ε > 0, ∀ᶠ n in p, ∀ x, edist (f x) (F n x) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝ : PseudoEMetricSpace α
    ι : Type u_2
    F : ι → β → α
    f : β → α
    p : Filter ι
    ⊢ Iff (TendstoUniformly F f p) (∀ (ε : ENNReal), GT.gt ε 0 → Filter.Eventually …
  -/
  simp only [← tendstoUniformlyOn_univ, tendstoUniformlyOn_iff, mem_univ, forall_const]
  /-
    🎉 no goals
  -/


theorem inseparable_iff : Inseparable x y ↔ edist x y = 0 := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    ⊢ Iff (Inseparable x y) (Eq (EDist.edist x y) 0)
  -/
  simp [inseparable_iff_mem_closure, mem_closure_iff, edist_comm, forall_lt_iff_le']
  /-
    🎉 no goals
  -/


alias ⟨_root_.Inseparable.edist_eq_zero, _⟩ := EMetric.inseparable_iff

-- see Note [nolint_ge]

/-- In a pseudoemetric space, Cauchy sequences are characterized by the fact that, eventually,
the pseudoedistance between its elements is arbitrarily small -/
theorem cauchySeq_iff [Nonempty β] [SemilatticeSup β] {u : β → α} :
    CauchySeq u ↔ ∀ ε > 0, ∃ N, ∀ m, N ≤ m → ∀ n, N ≤ n → edist (u m) (u n) < ε :=
  uniformity_basis_edist.cauchySeq_iff


/-- A variation around the emetric characterization of Cauchy sequences -/
theorem cauchySeq_iff' [Nonempty β] [SemilatticeSup β] {u : β → α} :
    CauchySeq u ↔ ∀ ε > (0 : ℝ≥0∞), ∃ N, ∀ n ≥ N, edist (u n) (u N) < ε :=
  uniformity_basis_edist.cauchySeq_iff'


/-- A variation of the emetric characterization of Cauchy sequences that deals with
`ℝ≥0` upper bounds. -/
theorem cauchySeq_iff_NNReal [Nonempty β] [SemilatticeSup β] {u : β → α} :
    CauchySeq u ↔ ∀ ε : ℝ≥0, 0 < ε → ∃ N, ∀ n, N ≤ n → edist (u n) (u N) < ε :=
  uniformity_basis_edist_nnreal.cauchySeq_iff'


theorem totallyBounded_iff {s : Set α} :
    TotallyBounded s ↔ ∀ ε > 0, ∃ t : Set α, t.Finite ∧ s ⊆ ⋃ y ∈ t, ball y ε :=
  ⟨fun H _ε ε0 => H _ (edist_mem_uniformity ε0), fun H _r ru =>
    let ⟨ε, ε0, hε⟩ := mem_uniformity_edist.1 ru
    let ⟨t, ft, h⟩ := H ε ε0
    ⟨t, ft, h.trans <| iUnion₂_mono fun _ _ _ => hε⟩⟩


theorem totallyBounded_iff' {s : Set α} :
    TotallyBounded s ↔ ∀ ε > 0, ∃ t, t ⊆ s ∧ Set.Finite t ∧ s ⊆ ⋃ y ∈ t, ball y ε :=
  ⟨fun H _ε ε0 => (totallyBounded_iff_subset.1 H) _ (edist_mem_uniformity ε0), fun H _r ru =>
    let ⟨ε, ε0, hε⟩ := mem_uniformity_edist.1 ru
    let ⟨t, _, ft, h⟩ := H ε ε0
    ⟨t, ft, h.trans <| iUnion₂_mono fun _ _ _ => hε⟩⟩


/-- A compact set in a pseudo emetric space is separable, i.e., it is a subset of the closure of a
countable set. -/
theorem subset_countable_closure_of_compact {s : Set α} (hs : IsCompact s) :
    ∃ t, t ⊆ s ∧ t.Countable ∧ s ⊆ closure t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : IsCompact s
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (HasSubset.Subse …
  -/
  refine subset_countable_closure_of_almost_dense_set s fun ε hε => ?_
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : IsCompact s
    ε : ENNReal
    hε : GT.gt ε 0
    ⊢ Exists fun t => And t.Countable (HasSubset.Subset s (Set.iUnion fun x => Set …
  -/
  rcases totallyBounded_iff'.1 hs.totallyBounded ε hε with ⟨t, -, htf, hst⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : IsCompact s
    ε : ENNReal
    hε : GT.gt ε 0
    t : Set α
    htf : t.Finite
    hst : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball …
    ⊢ Exists fun t => And t.Countable (HasSubset.Subset s (Set.iUnion fun x => Set …
  -/
  exact ⟨t, htf.countable, hst.trans <| iUnion₂_mono fun _ _ => ball_subset_closedBall⟩
  /-
    🎉 no goals
  -/


/-- A sigma compact pseudo emetric space has second countable topology. -/
instance (priority := 90) secondCountable_of_sigmaCompact [SigmaCompactSpace α] :
    SecondCountableTopology α := by
  /-
    α : Type u
    β : Type v
    X : Type u_1
    inst✝¹ : PseudoEMetricSpace α
    x y z : α
    ε ε₁ ε₂ : ENNReal
    s t : Set α
    inst✝ : SigmaCompactSpace α
    ⊢ SecondCountableTopology α
  -/
  suffices SeparableSpace α by exact UniformSpace.secondCountable_of_separable α
  choose T _ hTc hsubT using fun n =>
    subset_countable_closure_of_compact (isCompact_compactCovering α n)
  /-
    α : Type u
    β : Type v
    X : Type u_1
    inst✝¹ : PseudoEMetricSpace α
    x y z : α
    ε ε₁ ε₂ : ENNReal
    s t : Set α
    inst✝ : SigmaCompactSpace α
    T : Nat → Set α
    h✝ : ∀ (n : Nat), HasSubset.Subset (T n) (compactCovering α n)
    hTc : ∀ (n : Nat), (T n).Countable
    hsubT : ∀ (n : Nat), HasSubset.Subset (compactCovering α n) (closure (T n))
    ⊢ TopologicalSpace.SeparableSpace α
  -/
  refine ⟨⟨⋃ n, T n, countable_iUnion hTc, fun x => ?_⟩⟩
  /-
    α : Type u
    β : Type v
    X : Type u_1
    inst✝¹ : PseudoEMetricSpace α
    x✝ y z : α
    ε ε₁ ε₂ : ENNReal
    s t : Set α
    inst✝ : SigmaCompactSpace α
    T : Nat → Set α
    h✝ : ∀ (n : Nat), HasSubset.Subset (T n) (compactCovering α n)
    hTc : ∀ (n : Nat), (T n).Countable
    hsubT : ∀ (n : Nat), HasSubset.Subset (compactCovering α n) (closure (T n))
    x : α
    ⊢ Membership.mem (closure (Set.iUnion fun n => T n)) x
  -/
  rcases iUnion_eq_univ_iff.1 (iUnion_compactCovering α) x with ⟨n, hn⟩
  /-
    case intro
    α : Type u
    β : Type v
    X : Type u_1
    inst✝¹ : PseudoEMetricSpace α
    x✝ y z : α
    ε ε₁ ε₂ : ENNReal
    s t : Set α
    inst✝ : SigmaCompactSpace α
    T : Nat → Set α
    h✝ : ∀ (n : Nat), HasSubset.Subset (T n) (compactCovering α n)
    hTc : ∀ (n : Nat), (T n).Countable
    hsubT : ∀ (n : Nat), HasSubset.Subset (compactCovering α n) (closure (T n))
    x : α
    n : Nat
    hn : Membership.mem (compactCovering α n) x
    ⊢ Membership.mem (closure (Set.iUnion fun n => T n)) x
  -/
  exact closure_mono (subset_iUnion _ n) (hsubT _ hn)
  /-
    🎉 no goals
  -/


theorem secondCountable_of_almost_dense_set
    (hs : ∀ ε > 0, ∃ t : Set α, t.Countable ∧ ⋃ x ∈ t, closedBall x ε = univ) :
    SecondCountableTopology α := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (Eq (Set.iUn …
    ⊢ SecondCountableTopology α
  -/
  suffices SeparableSpace α from UniformSpace.secondCountable_of_separable α
  have : ∀ ε > 0, ∃ t : Set α, Set.Countable t ∧ univ ⊆ ⋃ x ∈ t, closedBall x ε := by
    simpa only [univ_subset_iff] using hs
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (Eq (Set.iUn …
    this : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset …
    ⊢ TopologicalSpace.SeparableSpace α
  -/
  rcases subset_countable_closure_of_almost_dense_set (univ : Set α) this with ⟨t, -, htc, ht⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (Eq (Set.iUn …
    this : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset …
    t : Set α
    htc : t.Countable
    ht : HasSubset.Subset Set.univ (closure t)
    ⊢ TopologicalSpace.SeparableSpace α
  -/
  exact ⟨⟨t, htc, fun x => ht (mem_univ x)⟩⟩
  /-
    🎉 no goals
  -/


/-- An emetric space is separated -/
instance (priority := 100) EMetricSpace.instT0Space : T0Space γ where
  t0 _ _ h := eq_of_edist_eq_zero <| inseparable_iff.1 h


/-- A map between emetric spaces is a uniform embedding if and only if the edistance between `f x`
and `f y` is controlled in terms of the distance between `x` and `y` and conversely. -/
theorem EMetric.isUniformEmbedding_iff' [EMetricSpace β] {f : γ → β} :
    IsUniformEmbedding f ↔
      (∀ ε > 0, ∃ δ > 0, ∀ {a b : γ}, edist a b < δ → edist (f a) (f b) < ε) ∧
        ∀ δ > 0, ∃ ε > 0, ∀ {a b : γ}, edist (f a) (f b) < ε → edist a b < δ := by
  /-
    β : Type v
    γ : Type w
    inst✝¹ : EMetricSpace γ
    inst✝ : EMetricSpace β
    f : γ → β
    ⊢ Iff (IsUniformEmbedding f) (And (∀ (ε : ENNReal), GT.gt ε 0 → Exists fun δ = …
  -/
  rw [isUniformEmbedding_iff_isUniformInducing, isUniformInducing_iff, uniformContinuous_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias EMetric.uniformEmbedding_iff' := EMetric.isUniformEmbedding_iff'


/-- If a `PseudoEMetricSpace` is a T₀ space, then it is an `EMetricSpace`. -/
-- Porting note: made `reducible`;
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: make it an instance?
abbrev EMetricSpace.ofT0PseudoEMetricSpace (α : Type*) [PseudoEMetricSpace α] [T0Space α] :
    EMetricSpace α :=
  { ‹PseudoEMetricSpace α› with
    eq_of_edist_eq_zero := fun h => (EMetric.inseparable_iff.2 h).eq }


/-- The product of two emetric spaces, with the max distance, is an extended
metric spaces. We make sure that the uniform structure thus constructed is the one
corresponding to the product of uniform spaces, to avoid diamond problems. -/
instance Prod.emetricSpaceMax [EMetricSpace β] : EMetricSpace (γ × β) :=
  .ofT0PseudoEMetricSpace _


/-- A compact set in an emetric space is separable, i.e., it is the closure of a countable set. -/
theorem countable_closure_of_compact {s : Set γ} (hs : IsCompact s) :
    ∃ t, t ⊆ s ∧ t.Countable ∧ s = closure t := by
  /-
    γ : Type w
    inst✝ : EMetricSpace γ
    s : Set γ
    hs : IsCompact s
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (Eq s (closure t …
  -/
  rcases subset_countable_closure_of_compact hs with ⟨t, hts, htc, hsub⟩
  /-
    case intro.intro.intro
    γ : Type w
    inst✝ : EMetricSpace γ
    s : Set γ
    hs : IsCompact s
    t : Set γ
    hts : HasSubset.Subset t s
    htc : t.Countable
    hsub : HasSubset.Subset s (closure t)
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (Eq s (closure t …
  -/
  exact ⟨t, hts, htc, hsub.antisymm (closure_minimal hts hs.isClosed)⟩
  /-
    🎉 no goals
  -/


instance [PseudoEMetricSpace X] : EDist (SeparationQuotient X) where
  edist := SeparationQuotient.lift₂ edist fun _ _ _ _ hx hy =>
    edist_congr (EMetric.inseparable_iff.1 hx) (EMetric.inseparable_iff.1 hy)


@[simp] theorem SeparationQuotient.edist_mk [PseudoEMetricSpace X] (x y : X) :
    edist (mk x) (mk y) = edist x y :=
  rfl


open SeparationQuotient in
instance [PseudoEMetricSpace X] : EMetricSpace (SeparationQuotient X) :=
  @EMetricSpace.ofT0PseudoEMetricSpace (SeparationQuotient X)
    { edist_self := surjective_mk.forall.2 edist_self,
      edist_comm := surjective_mk.forall₂.2 edist_comm,
      edist_triangle := surjective_mk.forall₃.2 edist_triangle,
      toUniformSpace := inferInstance,
      uniformity_edist := comap_injective (surjective_mk.prodMap surjective_mk) <| by
        /-
          α : Type u
          β : Type v
          X : Type u_1
          inst✝² : PseudoEMetricSpace α
          γ : Type w
          inst✝¹ : EMetricSpace γ
          inst✝ : PseudoEMetricSpace X
          ⊢ Eq (Filter.comap (Prod.map SeparationQuotient.mk SeparationQuotient.mk) (uni …
        -/
        simp [comap_mk_uniformity, PseudoEMetricSpace.uniformity_edist] } _
        /-
          🎉 no goals
        -/

