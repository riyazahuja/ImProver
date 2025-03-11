theorem subset_seqClosure {s : Set X} : s ⊆ seqClosure s := fun p hp =>
  ⟨const ℕ p, fun _ => hp, tendsto_const_nhds⟩


/-- The sequential closure of a set is contained in the closure of that set.
The converse is not true. -/
theorem seqClosure_subset_closure {s : Set X} : seqClosure s ⊆ closure s := fun _p ⟨_x, xM, xp⟩ =>
  mem_closure_of_tendsto xp (univ_mem' xM)


/-- The sequential closure of a sequentially closed set is the set itself. -/
theorem IsSeqClosed.seqClosure_eq {s : Set X} (hs : IsSeqClosed s) : seqClosure s = s :=
  Subset.antisymm (fun _p ⟨_x, hx, hp⟩ => hs hx hp) subset_seqClosure


/-- If a set is equal to its sequential closure, then it is sequentially closed. -/
theorem isSeqClosed_of_seqClosure_eq {s : Set X} (hs : seqClosure s = s) : IsSeqClosed s :=
  fun x _p hxs hxp => hs ▸ ⟨x, hxs, hxp⟩


/-- A set is sequentially closed iff it is equal to its sequential closure. -/
theorem isSeqClosed_iff {s : Set X} : IsSeqClosed s ↔ seqClosure s = s :=
  ⟨IsSeqClosed.seqClosure_eq, isSeqClosed_of_seqClosure_eq⟩


/-- A set is sequentially closed if it is closed. -/
protected theorem IsClosed.isSeqClosed {s : Set X} (hc : IsClosed s) : IsSeqClosed s :=
  fun _u _x hu hx => hc.mem_of_tendsto hx (Eventually.of_forall hu)


theorem seqClosure_eq_closure [FrechetUrysohnSpace X] (s : Set X) : seqClosure s = closure s :=
  seqClosure_subset_closure.antisymm <| FrechetUrysohnSpace.closure_subset_seqClosure s


/-- In a Fréchet-Urysohn space, a point belongs to the closure of a set iff it is a limit
of a sequence taking values in this set. -/
theorem mem_closure_iff_seq_limit [FrechetUrysohnSpace X] {s : Set X} {a : X} :
    a ∈ closure s ↔ ∃ x : ℕ → X, (∀ n : ℕ, x n ∈ s) ∧ Tendsto x atTop (𝓝 a) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FrechetUrysohnSpace X
    s : Set X
    a : X
    ⊢ Iff (Membership.mem (closure s) a) (Exists fun x => And (∀ (n : Nat), Member …
  -/
  rw [← seqClosure_eq_closure]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FrechetUrysohnSpace X
    s : Set X
    a : X
    ⊢ Iff (Membership.mem (seqClosure s) a) (Exists fun x => And (∀ (n : Nat), Mem …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If the domain of a function `f : α → β` is a Fréchet-Urysohn space, then convergence
is equivalent to sequential convergence. See also `Filter.tendsto_iff_seq_tendsto` for a version
that works for any pair of filters assuming that the filter in the domain is countably generated.

This property is equivalent to the definition of `FrechetUrysohnSpace`, see
`FrechetUrysohnSpace.of_seq_tendsto_imp_tendsto`. -/
theorem tendsto_nhds_iff_seq_tendsto [FrechetUrysohnSpace X] {f : X → Y} {a : X} {b : Y} :
    Tendsto f (𝓝 a) (𝓝 b) ↔ ∀ u : ℕ → X, Tendsto u atTop (𝓝 a) → Tendsto (f ∘ u) atTop (𝓝 b) := by
  refine
    ⟨fun hf u hu => hf.comp hu, fun h =>
      ((nhds_basis_closeds _).tendsto_iff (nhds_basis_closeds _)).2 ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace X
    f : X → Y
    a : X
    b : Y
    h : ∀ (u : Nat → X), Filter.Tendsto u Filter.atTop (nhds a) → Filter.Tendsto ( …
    ⊢ ∀ (ib : Set Y), And (Not (Membership.mem ib b)) (IsClosed ib) → Exists fun i …
  -/
  rintro s ⟨hbs, hsc⟩
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace X
    f : X → Y
    a : X
    b : Y
    h : ∀ (u : Nat → X), Filter.Tendsto u Filter.atTop (nhds a) → Filter.Tendsto ( …
    s : Set Y
    hbs : Not (Membership.mem s b)
    hsc : IsClosed s
    ⊢ Exists fun ia => And (And (Not (Membership.mem ia a)) (IsClosed ia)) (∀ (x : …
  -/
  refine ⟨closure (f ⁻¹' s), ⟨mt ?_ hbs, isClosed_closure⟩, fun x => mt fun hx => subset_closure hx⟩
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace X
    f : X → Y
    a : X
    b : Y
    h : ∀ (u : Nat → X), Filter.Tendsto u Filter.atTop (nhds a) → Filter.Tendsto ( …
    s : Set Y
    hbs : Not (Membership.mem s b)
    hsc : IsClosed s
    ⊢ Membership.mem (closure (Set.preimage f s)) a → Membership.mem s b
  -/
  rw [← seqClosure_eq_closure]
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace X
    f : X → Y
    a : X
    b : Y
    h : ∀ (u : Nat → X), Filter.Tendsto u Filter.atTop (nhds a) → Filter.Tendsto ( …
    s : Set Y
    hbs : Not (Membership.mem s b)
    hsc : IsClosed s
    ⊢ Membership.mem (seqClosure (Set.preimage f s)) a → Membership.mem s b
  -/
  rintro ⟨u, hus, hu⟩
  /-
    case intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace X
    f : X → Y
    a : X
    b : Y
    h : ∀ (u : Nat → X), Filter.Tendsto u Filter.atTop (nhds a) → Filter.Tendsto ( …
    s : Set Y
    hbs : Not (Membership.mem s b)
    hsc : IsClosed s
    u : Nat → X
    hus : ∀ (n : Nat), Membership.mem (Set.preimage f s) (u n)
    hu : Filter.Tendsto u Filter.atTop (nhds a)
    ⊢ Membership.mem s b
  -/
  exact hsc.mem_of_tendsto (h u hu) (Eventually.of_forall hus)
  /-
    🎉 no goals
  -/


/-- An alternative construction for `FrechetUrysohnSpace`: if sequential convergence implies
convergence, then the space is a Fréchet-Urysohn space. -/
theorem FrechetUrysohnSpace.of_seq_tendsto_imp_tendsto
    (h : ∀ (f : X → Prop) (a : X),
      (∀ u : ℕ → X, Tendsto u atTop (𝓝 a) → Tendsto (f ∘ u) atTop (𝓝 (f a))) → ContinuousAt f a) :
    FrechetUrysohnSpace X := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    h : ∀ (f : X → Prop) (a : X), (∀ (u : Nat → X), Filter.Tendsto u Filter.atTop  …
    ⊢ FrechetUrysohnSpace X
  -/
  refine ⟨fun s x hcx => ?_⟩
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    h : ∀ (f : X → Prop) (a : X), (∀ (u : Nat → X), Filter.Tendsto u Filter.atTop  …
    s : Set X
    x : X
    hcx : Membership.mem (closure s) x
    ⊢ Membership.mem (seqClosure s) x
  -/
  by_cases hx : x ∈ s
    /-
      case pos
      X : Type u_1
      inst✝ : TopologicalSpace X
      h : ∀ (f : X → Prop) (a : X), (∀ (u : Nat → X), Filter.Tendsto u Filter.atTop  …
      s : Set X
      x : X
      hcx : Membership.mem (closure s) x
      hx : Membership.mem s x
      ⊢ Membership.mem (seqClosure s) x
    -/
  · exact subset_seqClosure hx
    /-
      🎉 no goals
    -/
  · obtain ⟨u, hux, hus⟩ : ∃ u : ℕ → X, Tendsto u atTop (𝓝 x) ∧ ∃ᶠ x in atTop, u x ∈ s := by
      simpa only [ContinuousAt, hx, tendsto_nhds_true, (· ∘ ·), ← not_frequently, exists_prop,
        ← mem_closure_iff_frequently, hcx, imp_false, not_forall, not_not, not_false_eq_true,
        not_true_eq_false] using h (· ∉ s) x
    /-
      case neg.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      h : ∀ (f : X → Prop) (a : X), (∀ (u : Nat → X), Filter.Tendsto u Filter.atTop  …
      s : Set X
      x : X
      hcx : Membership.mem (closure s) x
      hx : Not (Membership.mem s x)
      u : Nat → X
      hux : Filter.Tendsto u Filter.atTop (nhds x)
      hus : Filter.Frequently (fun x => Membership.mem s (u x)) Filter.atTop
      ⊢ Membership.mem (seqClosure s) x
    -/
    rcases extraction_of_frequently_atTop hus with ⟨φ, φ_mono, hφ⟩
    /-
      case neg.intro.intro.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      h : ∀ (f : X → Prop) (a : X), (∀ (u : Nat → X), Filter.Tendsto u Filter.atTop  …
      s : Set X
      x : X
      hcx : Membership.mem (closure s) x
      hx : Not (Membership.mem s x)
      u : Nat → X
      hux : Filter.Tendsto u Filter.atTop (nhds x)
      hus : Filter.Frequently (fun x => Membership.mem s (u x)) Filter.atTop
      φ : Nat → Nat
      φ_mono : StrictMono φ
      hφ : ∀ (n : Nat), Membership.mem s (u (φ n))
      ⊢ Membership.mem (seqClosure s) x
    -/
    exact ⟨u ∘ φ, hφ, hux.comp φ_mono.tendsto_atTop⟩
    /-
      🎉 no goals
    -/

-- see Note [lower instance priority]

/-- Every first-countable space is a Fréchet-Urysohn space. -/
instance (priority := 100) FirstCountableTopology.frechetUrysohnSpace
    [FirstCountableTopology X] : FrechetUrysohnSpace X :=
  FrechetUrysohnSpace.of_seq_tendsto_imp_tendsto fun _ _ => tendsto_iff_seq_tendsto.2

-- see Note [lower instance priority]

/-- Every Fréchet-Urysohn space is a sequential space. -/
instance (priority := 100) FrechetUrysohnSpace.to_sequentialSpace [FrechetUrysohnSpace X] :
    SequentialSpace X :=
                  /-
                    X : Type u_1
                    Y : Type u_2
                    inst✝² : TopologicalSpace X
                    inst✝¹ : TopologicalSpace Y
                    inst✝ : FrechetUrysohnSpace X
                    s : Set X
                    hs : IsSeqClosed s
                    ⊢ IsClosed s
                  -/
  ⟨fun s hs => by rw [← closure_eq_iff_isClosed, ← seqClosure_eq_closure, hs.seqClosure_eq]⟩
                  /-
                    🎉 no goals
                  -/


theorem Topology.IsInducing.frechetUrysohnSpace [FrechetUrysohnSpace Y] {f : X → Y}
    (hf : IsInducing f) : FrechetUrysohnSpace X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    ⊢ FrechetUrysohnSpace X
  -/
  refine ⟨fun s x hx ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    s : Set X
    x : X
    hx : Membership.mem (closure s) x
    ⊢ Membership.mem (seqClosure s) x
  -/
  rw [hf.closure_eq_preimage_closure_image, mem_preimage, mem_closure_iff_seq_limit] at hx
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    s : Set X
    x : X
    hx : Exists fun x_1 => And (∀ (n : Nat), Membership.mem (Set.image f s) (x_1 n …
    ⊢ Membership.mem (seqClosure s) x
  -/
  rcases hx with ⟨u, hus, hu⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    s : Set X
    x : X
    u : Nat → Y
    hus : ∀ (n : Nat), Membership.mem (Set.image f s) (u n)
    hu : Filter.Tendsto u Filter.atTop (nhds (f x))
    ⊢ Membership.mem (seqClosure s) x
  -/
  choose v hv hvu using hus
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    s : Set X
    x : X
    u : Nat → Y
    hu : Filter.Tendsto u Filter.atTop (nhds (f x))
    v : Nat → X
    hv : ∀ (n : Nat), Membership.mem s (v n)
    hvu : ∀ (n : Nat), Eq (f (v n)) (u n)
    ⊢ Membership.mem (seqClosure s) x
  -/
  refine ⟨v, hv, ?_⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : FrechetUrysohnSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    s : Set X
    x : X
    u : Nat → Y
    hu : Filter.Tendsto u Filter.atTop (nhds (f x))
    v : Nat → X
    hv : ∀ (n : Nat), Membership.mem s (v n)
    hvu : ∀ (n : Nat), Eq (f (v n)) (u n)
    ⊢ Filter.Tendsto v Filter.atTop (nhds x)
  -/
  simpa only [hf.tendsto_nhds_iff, Function.comp_def, hvu]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias Inducing.frechetUrysohnSpace := IsInducing.frechetUrysohnSpace


/-- Subtype of a Fréchet-Urysohn space is a Fréchet-Urysohn space. -/
instance Subtype.instFrechetUrysohnSpace [FrechetUrysohnSpace X] {p : X → Prop} :
    FrechetUrysohnSpace (Subtype p) :=
  IsInducing.subtypeVal.frechetUrysohnSpace


/-- In a sequential space, a set is closed iff it's sequentially closed. -/
theorem isSeqClosed_iff_isClosed [SequentialSpace X] {M : Set X} : IsSeqClosed M ↔ IsClosed M :=
  ⟨IsSeqClosed.isClosed, IsClosed.isSeqClosed⟩


/-- The preimage of a sequentially closed set under a sequentially continuous map is sequentially
closed. -/
theorem IsSeqClosed.preimage {f : X → Y} {s : Set Y} (hs : IsSeqClosed s) (hf : SeqContinuous f) :
    IsSeqClosed (f ⁻¹' s) := fun _x _p hx hp => hs hx (hf hp)

-- A continuous function is sequentially continuous.

protected theorem Continuous.seqContinuous {f : X → Y} (hf : Continuous f) : SeqContinuous f :=
  fun _x p hx => (hf.tendsto p).comp hx


/-- A sequentially continuous function defined on a sequential space is continuous. -/
protected theorem SeqContinuous.continuous [SequentialSpace X] {f : X → Y} (hf : SeqContinuous f) :
    Continuous f :=
  continuous_iff_isClosed.mpr fun _s hs => (hs.isSeqClosed.preimage hf).isClosed


/-- If the domain of a function is a sequential space, then continuity of this function is
equivalent to its sequential continuity. -/
theorem continuous_iff_seqContinuous [SequentialSpace X] {f : X → Y} :
    Continuous f ↔ SeqContinuous f :=
  ⟨Continuous.seqContinuous, SeqContinuous.continuous⟩


theorem SequentialSpace.coinduced [SequentialSpace X] {Y} (f : X → Y) :
    @SequentialSpace Y (.coinduced f ‹_›) :=
  letI : TopologicalSpace Y := .coinduced f ‹_›
  ⟨fun _ hs ↦ isClosed_coinduced.2 (hs.preimage continuous_coinduced_rng.seqContinuous).isClosed⟩


protected theorem SequentialSpace.iSup {X} {ι : Sort*} {t : ι → TopologicalSpace X}
    (h : ∀ i, @SequentialSpace X (t i)) : @SequentialSpace X (⨆ i, t i) := by
  /-
    X : Type u_4
    ι : Sort u_3
    t : ι → TopologicalSpace X
    h : ∀ (i : ι), SequentialSpace X
    ⊢ SequentialSpace X
  -/
  letI : TopologicalSpace X := ⨆ i, t i
  /-
    X : Type u_4
    ι : Sort u_3
    t : ι → TopologicalSpace X
    h : ∀ (i : ι), SequentialSpace X
    this : TopologicalSpace X := iSup fun i => t i
    ⊢ SequentialSpace X
  -/
  refine ⟨fun s hs ↦ isClosed_iSup_iff.2 fun i ↦ ?_⟩
  /-
    X : Type u_4
    ι : Sort u_3
    t : ι → TopologicalSpace X
    h : ∀ (i : ι), SequentialSpace X
    this : TopologicalSpace X := iSup fun i => t i
    s : Set X
    hs : IsSeqClosed s
    i : ι
    ⊢ IsClosed s
  -/
  letI := t i
  /-
    X : Type u_4
    ι : Sort u_3
    t : ι → TopologicalSpace X
    h : ∀ (i : ι), SequentialSpace X
    this✝ : TopologicalSpace X := iSup fun i => t i
    s : Set X
    hs : IsSeqClosed s
    i : ι
    this : TopologicalSpace X := t i
    ⊢ IsClosed s
  -/
  exact IsSeqClosed.isClosed fun u x hus hux ↦ hs hus <| hux.mono_right <| nhds_mono <| le_iSup _ _
  /-
    🎉 no goals
  -/


protected theorem SequentialSpace.sup {X} {t₁ t₂ : TopologicalSpace X}
    (h₁ : @SequentialSpace X t₁) (h₂ : @SequentialSpace X t₂) :
    @SequentialSpace X (t₁ ⊔ t₂) := by
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : SequentialSpace X
    h₂ : SequentialSpace X
    ⊢ SequentialSpace X
  -/
  rw [sup_eq_iSup]
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : SequentialSpace X
    h₂ : SequentialSpace X
    ⊢ SequentialSpace X
  -/
  exact .iSup <| Bool.forall_bool.2 ⟨h₂, h₁⟩
  /-
    🎉 no goals
  -/


lemma Topology.IsQuotientMap.sequentialSpace [SequentialSpace X] {f : X → Y}
    (hf : IsQuotientMap f) : SequentialSpace Y := hf.2.symm ▸ .coinduced f


@[deprecated (since := "2024-10-22")]
alias QuotientMap.sequentialSpace := IsQuotientMap.sequentialSpace


/-- The quotient of a sequential space is a sequential space. -/
instance Quotient.instSequentialSpace [SequentialSpace X] {s : Setoid X} :
    SequentialSpace (Quotient s) :=
  isQuotientMap_quot_mk.sequentialSpace


/-- The sum (disjoint union) of two sequential spaces is a sequential space. -/
instance Sum.instSequentialSpace [SequentialSpace X] [SequentialSpace Y] :
    SequentialSpace (X ⊕ Y) :=
  .sup (.coinduced Sum.inl) (.coinduced Sum.inr)


/-- The disjoint union of an indexed family of sequential spaces is a sequential space. -/
instance Sigma.instSequentialSpace {ι : Type*} {X : ι → Type*}
    [∀ i, TopologicalSpace (X i)] [∀ i, SequentialSpace (X i)] : SequentialSpace (Σ i, X i) :=
  .iSup fun _ ↦ .coinduced _


theorem IsSeqCompact.subseq_of_frequently_in {s : Set X} (hs : IsSeqCompact s) {x : ℕ → X}
    (hx : ∃ᶠ n in atTop, x n ∈ s) :
    ∃ a ∈ s, ∃ φ : ℕ → ℕ, StrictMono φ ∧ Tendsto (x ∘ φ) atTop (𝓝 a) :=
  let ⟨ψ, hψ, huψ⟩ := extraction_of_frequently_atTop hx
  let ⟨a, a_in, φ, hφ, h⟩ := hs huψ
  ⟨a, a_in, ψ ∘ φ, hψ.comp hφ, h⟩


theorem SeqCompactSpace.tendsto_subseq [SeqCompactSpace X] (x : ℕ → X) :
    ∃ (a : X) (φ : ℕ → ℕ), StrictMono φ ∧ Tendsto (x ∘ φ) atTop (𝓝 a) :=
  let ⟨a, _, φ, mono, h⟩ := isSeqCompact_univ fun n => mem_univ (x n)
  ⟨a, φ, mono, h⟩


protected theorem IsCompact.isSeqCompact {s : Set X} (hs : IsCompact s) : IsSeqCompact s :=
  fun _x x_in =>
  let ⟨a, a_in, ha⟩ := hs (tendsto_principal.mpr (Eventually.of_forall x_in))
  ⟨a, a_in, tendsto_subseq ha⟩


theorem IsCompact.tendsto_subseq' {s : Set X} {x : ℕ → X} (hs : IsCompact s)
    (hx : ∃ᶠ n in atTop, x n ∈ s) :
    ∃ a ∈ s, ∃ φ : ℕ → ℕ, StrictMono φ ∧ Tendsto (x ∘ φ) atTop (𝓝 a) :=
  hs.isSeqCompact.subseq_of_frequently_in hx


theorem IsCompact.tendsto_subseq {s : Set X} {x : ℕ → X} (hs : IsCompact s) (hx : ∀ n, x n ∈ s) :
    ∃ a ∈ s, ∃ φ : ℕ → ℕ, StrictMono φ ∧ Tendsto (x ∘ φ) atTop (𝓝 a) :=
  hs.isSeqCompact hx

-- see Note [lower instance priority]

instance (priority := 100) FirstCountableTopology.seq_compact_of_compact [CompactSpace X] :
    SeqCompactSpace X :=
  ⟨isCompact_univ.isSeqCompact⟩


theorem CompactSpace.tendsto_subseq [CompactSpace X] (x : ℕ → X) :
    ∃ (a : _) (φ : ℕ → ℕ), StrictMono φ ∧ Tendsto (x ∘ φ) atTop (𝓝 a) :=
  SeqCompactSpace.tendsto_subseq x


/-- Sequential compactness of sets is preserved under sequentially continuous functions. -/
theorem IsSeqCompact.image (f_cont : SeqContinuous f) {K : Set X} (K_cpt : IsSeqCompact K) :
    IsSeqCompact (f '' K) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    f_cont : SeqContinuous f
    K : Set X
    K_cpt : IsSeqCompact K
    ⊢ IsSeqCompact (Set.image f K)
  -/
  intro ys ys_in_fK
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    f_cont : SeqContinuous f
    K : Set X
    K_cpt : IsSeqCompact K
    ys : Nat → Y
    ys_in_fK : ∀ (n : Nat), Membership.mem (Set.image f K) (ys n)
    ⊢ Exists fun a => And (Membership.mem (Set.image f K) a) (Exists fun φ => And  …
  -/
  choose xs xs_in_K fxs_eq_ys using ys_in_fK
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    f_cont : SeqContinuous f
    K : Set X
    K_cpt : IsSeqCompact K
    ys : Nat → Y
    xs : Nat → X
    xs_in_K : ∀ (n : Nat), Membership.mem K (xs n)
    fxs_eq_ys : ∀ (n : Nat), Eq (f (xs n)) (ys n)
    ⊢ Exists fun a => And (Membership.mem (Set.image f K) a) (Exists fun φ => And  …
  -/
  obtain ⟨a, a_in_K, phi, phi_mono, xs_phi_lim⟩ := K_cpt xs_in_K
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    f_cont : SeqContinuous f
    K : Set X
    K_cpt : IsSeqCompact K
    ys : Nat → Y
    xs : Nat → X
    xs_in_K : ∀ (n : Nat), Membership.mem K (xs n)
    fxs_eq_ys : ∀ (n : Nat), Eq (f (xs n)) (ys n)
    a : X
    a_in_K : Membership.mem K a
    phi : Nat → Nat
    phi_mono : StrictMono phi
    xs_phi_lim : Filter.Tendsto (Function.comp xs phi) Filter.atTop (nhds a)
    ⊢ Exists fun a => And (Membership.mem (Set.image f K) a) (Exists fun φ => And  …
  -/
  refine ⟨f a, mem_image_of_mem f a_in_K, phi, phi_mono, ?_⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    f_cont : SeqContinuous f
    K : Set X
    K_cpt : IsSeqCompact K
    ys : Nat → Y
    xs : Nat → X
    xs_in_K : ∀ (n : Nat), Membership.mem K (xs n)
    fxs_eq_ys : ∀ (n : Nat), Eq (f (xs n)) (ys n)
    a : X
    a_in_K : Membership.mem K a
    phi : Nat → Nat
    phi_mono : StrictMono phi
    xs_phi_lim : Filter.Tendsto (Function.comp xs phi) Filter.atTop (nhds a)
    ⊢ Filter.Tendsto (Function.comp ys phi) Filter.atTop (nhds (f a))
  -/
  exact (f_cont xs_phi_lim).congr fun x ↦ fxs_eq_ys (phi x)
  /-
    🎉 no goals
  -/


/-- The range of sequentially continuous function on a sequentially compact space is sequentially
compact. -/
theorem IsSeqCompact.range [SeqCompactSpace X] (f_cont : SeqContinuous f) :
    IsSeqCompact (Set.range f) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    inst✝ : SeqCompactSpace X
    f_cont : SeqContinuous f
    ⊢ IsSeqCompact (Set.range f)
  -/
  simpa using isSeqCompact_univ.image f_cont
  /-
    🎉 no goals
  -/


theorem IsSeqCompact.exists_tendsto_of_frequently_mem (hs : IsSeqCompact s) {u : ℕ → X}
    (hu : ∃ᶠ n in atTop, u n ∈ s) (huc : CauchySeq u) : ∃ x ∈ s, Tendsto u atTop (𝓝 x) :=
  let ⟨x, hxs, _φ, φ_mono, hx⟩ := hs.subseq_of_frequently_in hu
  ⟨x, hxs, tendsto_nhds_of_cauchySeq_of_subseq huc φ_mono.tendsto_atTop hx⟩


theorem IsSeqCompact.exists_tendsto (hs : IsSeqCompact s) {u : ℕ → X} (hu : ∀ n, u n ∈ s)
    (huc : CauchySeq u) : ∃ x ∈ s, Tendsto u atTop (𝓝 x) :=
  hs.exists_tendsto_of_frequently_mem (Frequently.of_forall hu) huc


/-- A sequentially compact set in a uniform space is totally bounded. -/
protected theorem IsSeqCompact.totallyBounded (h : IsSeqCompact s) : TotallyBounded s := by
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    s : Set X
    h : IsSeqCompact s
    ⊢ TotallyBounded s
  -/
  intro V V_in
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    s : Set X
    h : IsSeqCompact s
    V : Set (Prod X X)
    V_in : Membership.mem (uniformity X) V
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset s (Set.iUnion fun y => Set.iU …
  -/
  unfold IsSeqCompact at h
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    s : Set X
    h : ∀ ⦃x : Nat → X⦄, (∀ (n : Nat), Membership.mem s (x n)) → Exists fun a => A …
    V : Set (Prod X X)
    V_in : Membership.mem (uniformity X) V
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset s (Set.iUnion fun y => Set.iU …
  -/
  contrapose! h
  obtain ⟨u, u_in, hu⟩ : ∃ u : ℕ → X, (∀ n, u n ∈ s) ∧ ∀ n m, m < n → u m ∉ ball (u n) V := by
    simp only [not_subset, mem_iUnion₂, not_exists, exists_prop] at h
    simpa only [forall_and, forall_mem_image, not_and] using seq_of_forall_finite_exists h
  /-
    case intro.intro
    X : Type u_1
    inst✝ : UniformSpace X
    s : Set X
    V : Set (Prod X X)
    V_in : Membership.mem (uniformity X) V
    h : ∀ (t : Set X), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y => Set …
    u : Nat → X
    u_in : ∀ (n : Nat), Membership.mem s (u n)
    hu : ∀ (n m : Nat), LT.lt m n → Not (Membership.mem (UniformSpace.ball (u n) V …
    ⊢ Exists fun ⦃x⦄ => And (∀ (n : Nat), Membership.mem s (x n)) (∀ (a : X), Memb …
  -/
  refine ⟨u, u_in, fun x _ φ hφ huφ => ?_⟩
  obtain ⟨N, hN⟩ : ∃ N, ∀ p q, p ≥ N → q ≥ N → (u (φ p), u (φ q)) ∈ V :=
    huφ.cauchySeq.mem_entourage V_in
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝ : UniformSpace X
    s : Set X
    V : Set (Prod X X)
    V_in : Membership.mem (uniformity X) V
    h : ∀ (t : Set X), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y => Set …
    u : Nat → X
    u_in : ∀ (n : Nat), Membership.mem s (u n)
    hu : ∀ (n m : Nat), LT.lt m n → Not (Membership.mem (UniformSpace.ball (u n) V …
    x : X
    x✝ : Membership.mem s x
    φ : Nat → Nat
    hφ : StrictMono φ
    huφ : Filter.Tendsto (Function.comp u φ) Filter.atTop (nhds x)
    N : Nat
    hN : ∀ (p q : Nat), GE.ge p N → GE.ge q N → Membership.mem V { fst := u (φ p), …
    ⊢ False
  -/
  exact hu (φ <| N + 1) (φ N) (hφ <| Nat.lt_add_one N) (hN (N + 1) N N.le_succ le_rfl)
  /-
    🎉 no goals
  -/


/-- A sequentially compact set in a uniform set with countably generated uniformity filter
is complete. -/
protected theorem IsSeqCompact.isComplete (hs : IsSeqCompact s) : IsComplete s := fun l hl hls => by
  /-
    X : Type u_1
    inst✝¹ : UniformSpace X
    s : Set X
    inst✝ : (uniformity X).IsCountablyGenerated
    hs : IsSeqCompact s
    l : Filter X
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    ⊢ Exists fun x => And (Membership.mem s x) (LE.le l (nhds x))
  -/
  have := hl.1
  /-
    X : Type u_1
    inst✝¹ : UniformSpace X
    s : Set X
    inst✝ : (uniformity X).IsCountablyGenerated
    hs : IsSeqCompact s
    l : Filter X
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    ⊢ Exists fun x => And (Membership.mem s x) (LE.le l (nhds x))
  -/
  rcases exists_antitone_basis (𝓤 X) with ⟨V, hV⟩
  /-
    case intro
    X : Type u_1
    inst✝¹ : UniformSpace X
    s : Set X
    inst✝ : (uniformity X).IsCountablyGenerated
    hs : IsSeqCompact s
    l : Filter X
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    V : Nat → Set (Prod X X)
    hV : (uniformity X).HasAntitoneBasis V
    ⊢ Exists fun x => And (Membership.mem s x) (LE.le l (nhds x))
  -/
  choose W hW hWV using fun n => comp_mem_uniformity_sets (hV.mem n)
  have hWV' : ∀ n, W n ⊆ V n := fun n ⟨x, y⟩ hx =>
    @hWV n (x, y) ⟨x, refl_mem_uniformity <| hW _, hx⟩
  obtain ⟨t, ht_anti, htl, htW, hts⟩ :
      ∃ t : ℕ → Set X, Antitone t ∧ (∀ n, t n ∈ l) ∧ (∀ n, t n ×ˢ t n ⊆ W n) ∧ ∀ n, t n ⊆ s := by
    have : ∀ n, ∃ t ∈ l, t ×ˢ t ⊆ W n ∧ t ⊆ s := by
      rw [le_principal_iff] at hls
      have : ∀ n, W n ∩ s ×ˢ s ∈ l ×ˢ l := fun n => inter_mem (hl.2 (hW n)) (prod_mem_prod hls hls)
      simpa only [l.basis_sets.prod_self.mem_iff, true_imp_iff, subset_inter_iff,
        prod_self_subset_prod_self, and_assoc] using this
    choose t htl htW hts using this
    have : ∀ n : ℕ, ⋂ k ≤ n, t k ⊆ t n := fun n => by apply iInter₂_subset; rfl
    exact ⟨fun n => ⋂ k ≤ n, t k, fun m n h =>
      biInter_subset_biInter_left fun k (hk : k ≤ m) => hk.trans h, fun n =>
      (biInter_mem (finite_le_nat n)).2 fun k _ => htl k, fun n =>
      (prod_mono (this n) (this n)).trans (htW n), fun n => (this n).trans (hts n)⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : UniformSpace X
    s : Set X
    inst✝ : (uniformity X).IsCountablyGenerated
    hs : IsSeqCompact s
    l : Filter X
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    V : Nat → Set (Prod X X)
    hV : (uniformity X).HasAntitoneBasis V
    W : Nat → Set (Prod X X)
    hW : ∀ (n : Nat), Membership.mem (uniformity X) (W n)
    hWV : ∀ (n : Nat), HasSubset.Subset (compRel (W n) (W n)) (V n)
    hWV' : ∀ (n : Nat), HasSubset.Subset (W n) (V n)
    t : Nat → Set X
    ht_anti : Antitone t
    htl : ∀ (n : Nat), Membership.mem l (t n)
    htW : ∀ (n : Nat), HasSubset.Subset (SProd.sprod (t n) (t n)) (W n)
    hts : ∀ (n : Nat), HasSubset.Subset (t n) s
    ⊢ Exists fun x => And (Membership.mem s x) (LE.le l (nhds x))
  -/
  choose u hu using fun n => Filter.nonempty_of_mem (htl n)
  have huc : CauchySeq u := hV.toHasBasis.cauchySeq_iff.2 fun N _ =>
      ⟨N, fun m hm n hn => hWV' _ <| @htW N (_, _) ⟨ht_anti hm (hu _), ht_anti hn (hu _)⟩⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : UniformSpace X
    s : Set X
    inst✝ : (uniformity X).IsCountablyGenerated
    hs : IsSeqCompact s
    l : Filter X
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    V : Nat → Set (Prod X X)
    hV : (uniformity X).HasAntitoneBasis V
    W : Nat → Set (Prod X X)
    hW : ∀ (n : Nat), Membership.mem (uniformity X) (W n)
    hWV : ∀ (n : Nat), HasSubset.Subset (compRel (W n) (W n)) (V n)
    hWV' : ∀ (n : Nat), HasSubset.Subset (W n) (V n)
    t : Nat → Set X
    ht_anti : Antitone t
    htl : ∀ (n : Nat), Membership.mem l (t n)
    htW : ∀ (n : Nat), HasSubset.Subset (SProd.sprod (t n) (t n)) (W n)
    hts : ∀ (n : Nat), HasSubset.Subset (t n) s
    u : Nat → X
    hu : ∀ (n : Nat), Membership.mem (t n) (u n)
    huc : CauchySeq u
    ⊢ Exists fun x => And (Membership.mem s x) (LE.le l (nhds x))
  -/
  rcases hs.exists_tendsto (fun n => hts n (hu n)) huc with ⟨x, hxs, hx⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : UniformSpace X
    s : Set X
    inst✝ : (uniformity X).IsCountablyGenerated
    hs : IsSeqCompact s
    l : Filter X
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    V : Nat → Set (Prod X X)
    hV : (uniformity X).HasAntitoneBasis V
    W : Nat → Set (Prod X X)
    hW : ∀ (n : Nat), Membership.mem (uniformity X) (W n)
    hWV : ∀ (n : Nat), HasSubset.Subset (compRel (W n) (W n)) (V n)
    hWV' : ∀ (n : Nat), HasSubset.Subset (W n) (V n)
    t : Nat → Set X
    ht_anti : Antitone t
    htl : ∀ (n : Nat), Membership.mem l (t n)
    htW : ∀ (n : Nat), HasSubset.Subset (SProd.sprod (t n) (t n)) (W n)
    hts : ∀ (n : Nat), HasSubset.Subset (t n) s
    u : Nat → X
    hu : ∀ (n : Nat), Membership.mem (t n) (u n)
    huc : CauchySeq u
    x : X
    hxs : Membership.mem s x
    hx : Filter.Tendsto u Filter.atTop (nhds x)
    ⊢ Exists fun x => And (Membership.mem s x) (LE.le l (nhds x))
  -/
  refine ⟨x, hxs, (nhds_basis_uniformity' hV.toHasBasis).ge_iff.2 fun N _ => ?_⟩
  obtain ⟨n, hNn, hn⟩ : ∃ n, N ≤ n ∧ u n ∈ ball x (W N) :=
    ((eventually_ge_atTop N).and (hx <| ball_mem_nhds x (hW N))).exists
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : UniformSpace X
    s : Set X
    inst✝ : (uniformity X).IsCountablyGenerated
    hs : IsSeqCompact s
    l : Filter X
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    V : Nat → Set (Prod X X)
    hV : (uniformity X).HasAntitoneBasis V
    W : Nat → Set (Prod X X)
    hW : ∀ (n : Nat), Membership.mem (uniformity X) (W n)
    hWV : ∀ (n : Nat), HasSubset.Subset (compRel (W n) (W n)) (V n)
    hWV' : ∀ (n : Nat), HasSubset.Subset (W n) (V n)
    t : Nat → Set X
    ht_anti : Antitone t
    htl : ∀ (n : Nat), Membership.mem l (t n)
    htW : ∀ (n : Nat), HasSubset.Subset (SProd.sprod (t n) (t n)) (W n)
    hts : ∀ (n : Nat), HasSubset.Subset (t n) s
    u : Nat → X
    hu : ∀ (n : Nat), Membership.mem (t n) (u n)
    huc : CauchySeq u
    x : X
    hxs : Membership.mem s x
    hx : Filter.Tendsto u Filter.atTop (nhds x)
    N : Nat
    x✝ : True
    n : Nat
    hNn : LE.le N n
    hn : Membership.mem (UniformSpace.ball x (W N)) (u n)
    ⊢ Membership.mem l (UniformSpace.ball x (V N))
  -/
  refine mem_of_superset (htl n) fun y hy => hWV N ⟨u n, hn, htW N ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : UniformSpace X
    s : Set X
    inst✝ : (uniformity X).IsCountablyGenerated
    hs : IsSeqCompact s
    l : Filter X
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    V : Nat → Set (Prod X X)
    hV : (uniformity X).HasAntitoneBasis V
    W : Nat → Set (Prod X X)
    hW : ∀ (n : Nat), Membership.mem (uniformity X) (W n)
    hWV : ∀ (n : Nat), HasSubset.Subset (compRel (W n) (W n)) (V n)
    hWV' : ∀ (n : Nat), HasSubset.Subset (W n) (V n)
    t : Nat → Set X
    ht_anti : Antitone t
    htl : ∀ (n : Nat), Membership.mem l (t n)
    htW : ∀ (n : Nat), HasSubset.Subset (SProd.sprod (t n) (t n)) (W n)
    hts : ∀ (n : Nat), HasSubset.Subset (t n) s
    u : Nat → X
    hu : ∀ (n : Nat), Membership.mem (t n) (u n)
    huc : CauchySeq u
    x : X
    hxs : Membership.mem s x
    hx : Filter.Tendsto u Filter.atTop (nhds x)
    N : Nat
    x✝ : True
    n : Nat
    hNn : LE.le N n
    hn : Membership.mem (UniformSpace.ball x (W N)) (u n)
    y : X
    hy : Membership.mem (t n) y
    ⊢ Membership.mem (SProd.sprod (t N) (t N)) { fst := u n, snd := { fst := x, sn …
  -/
  exact ⟨ht_anti hNn (hu n), ht_anti hNn hy⟩
  /-
    🎉 no goals
  -/


/-- If `𝓤 β` is countably generated, then any sequentially compact set is compact. -/
protected theorem IsSeqCompact.isCompact (hs : IsSeqCompact s) : IsCompact s :=
  isCompact_iff_totallyBounded_isComplete.2 ⟨hs.totallyBounded, hs.isComplete⟩


/-- A version of Bolzano-Weierstrass: in a uniform space with countably generated uniformity filter
(e.g., in a metric space), a set is compact if and only if it is sequentially compact. -/
protected theorem UniformSpace.isCompact_iff_isSeqCompact : IsCompact s ↔ IsSeqCompact s :=
  ⟨fun H => H.isSeqCompact, fun H => H.isCompact⟩


theorem UniformSpace.compactSpace_iff_seqCompactSpace : CompactSpace X ↔ SeqCompactSpace X := by
  /-
    X : Type u_1
    inst✝¹ : UniformSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    ⊢ Iff (CompactSpace X) (SeqCompactSpace X)
  -/
  simp only [← isCompact_univ_iff, seqCompactSpace_iff, UniformSpace.isCompact_iff_isSeqCompact]
  /-
    🎉 no goals
  -/


