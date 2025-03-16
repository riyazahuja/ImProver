/-- The open sets of the least topology containing a collection of basic sets. -/
inductive GenerateOpen (g : Set (Set α)) : Set α → Prop
  | basic : ∀ s ∈ g, GenerateOpen g s
  | univ : GenerateOpen g univ
  | inter : ∀ s t, GenerateOpen g s → GenerateOpen g t → GenerateOpen g (s ∩ t)
  | sUnion : ∀ S : Set (Set α), (∀ s ∈ S, GenerateOpen g s) → GenerateOpen g (⋃₀ S)


/-- The smallest topological space containing the collection `g` of basic sets -/
def generateFrom (g : Set (Set α)) : TopologicalSpace α where
  IsOpen := GenerateOpen g
  isOpen_univ := GenerateOpen.univ
  isOpen_inter := GenerateOpen.inter
  isOpen_sUnion := GenerateOpen.sUnion


theorem isOpen_generateFrom_of_mem {g : Set (Set α)} {s : Set α} (hs : s ∈ g) :
    IsOpen[generateFrom g] s :=
  GenerateOpen.basic s hs


theorem nhds_generateFrom {g : Set (Set α)} {a : α} :
    @nhds α (generateFrom g) a = ⨅ s ∈ { s | a ∈ s ∧ s ∈ g }, 𝓟 s := by
  /-
    α : Type u
    g : Set (Set α)
    a : α
    ⊢ Eq (nhds a) (iInf fun s => iInf fun h => Filter.principal s)
  -/
  letI := generateFrom g
  /-
    α : Type u
    g : Set (Set α)
    a : α
    this : TopologicalSpace α := TopologicalSpace.generateFrom g
    ⊢ Eq (nhds a) (iInf fun s => iInf fun h => Filter.principal s)
  -/
  rw [nhds_def]
  /-
    α : Type u
    g : Set (Set α)
    a : α
    this : TopologicalSpace α := TopologicalSpace.generateFrom g
    ⊢ Eq (iInf fun s => iInf fun h => Filter.principal s) (iInf fun s => iInf fun  …
  -/
  refine le_antisymm (biInf_mono fun s ⟨as, sg⟩ => ⟨as, .basic _ sg⟩) <| le_iInf₂ ?_
  /-
    α : Type u
    g : Set (Set α)
    a : α
    this : TopologicalSpace α := TopologicalSpace.generateFrom g
    ⊢ ∀ (i : Set α), Membership.mem (setOf fun s => And (Membership.mem s a) (IsOp …
  -/
  rintro s ⟨ha, hs⟩
  induction hs with
  | basic _ hs => exact iInf₂_le _ ⟨ha, hs⟩
  | univ => exact le_top.trans_eq principal_univ.symm
  | inter _ _ _ _ hs ht => exact (le_inf (hs ha.1) (ht ha.2)).trans_eq inf_principal
  | sUnion _ _ hS =>
    let ⟨t, htS, hat⟩ := ha
    exact (hS t htS hat).trans (principal_mono.2 <| subset_sUnion_of_mem htS)


lemma tendsto_nhds_generateFrom_iff {β : Type*} {m : α → β} {f : Filter α} {g : Set (Set β)}
    {b : β} : Tendsto m f (@nhds β (generateFrom g) b) ↔ ∀ s ∈ g, b ∈ s → m ⁻¹' s ∈ f := by
  simp only [nhds_generateFrom, @forall_swap (b ∈ _), tendsto_iInf, mem_setOf_eq, and_imp,
                        /-
                          α : Type u
                          β : Type u_1
                          m : α → β
                          f : Filter α
                          g : Set (Set β)
                          b : β
                          ⊢ Iff (∀ (i : Set β), Membership.mem g i → Membership.mem i b → Filter.Eventua …
                        -/
    tendsto_principal]; rfl
                        /-
                          🎉 no goals
                        -/


/-- Construct a topology on α given the filter of neighborhoods of each point of α. -/
protected def mkOfNhds (n : α → Filter α) : TopologicalSpace α where
  IsOpen s := ∀ a ∈ s, s ∈ n a
  isOpen_univ _ _ := univ_mem
  isOpen_inter := fun _s _t hs ht x ⟨hxs, hxt⟩ => inter_mem (hs x hxs) (ht x hxt)
  isOpen_sUnion := fun _s hs _a ⟨x, hx, hxa⟩ =>
    mem_of_superset (hs x hx _ hxa) (subset_sUnion_of_mem hx)


theorem nhds_mkOfNhds_of_hasBasis {n : α → Filter α} {ι : α → Sort*} {p : ∀ a, ι a → Prop}
    {s : ∀ a, ι a → Set α} (hb : ∀ a, (n a).HasBasis (p a) (s a))
    (hpure : ∀ a i, p a i → a ∈ s a i) (hopen : ∀ a i, p a i → ∀ᶠ x in n a, s a i ∈ n x) (a : α) :
    @nhds α (.mkOfNhds n) a = n a := by
  /-
    α : Type u
    n : α → Filter α
    ι : α → Sort u_1
    p : (a : α) → ι a → Prop
    s : (a : α) → ι a → Set α
    hb : ∀ (a : α), (n a).HasBasis (p a) (s a)
    hpure : ∀ (a : α) (i : ι a), p a i → Membership.mem (s a i) a
    hopen : ∀ (a : α) (i : ι a), p a i → Filter.Eventually (fun x => Membership.me …
    a : α
    ⊢ Eq (nhds a) (n a)
  -/
  let t : TopologicalSpace α := .mkOfNhds n
  /-
    α : Type u
    n : α → Filter α
    ι : α → Sort u_1
    p : (a : α) → ι a → Prop
    s : (a : α) → ι a → Set α
    hb : ∀ (a : α), (n a).HasBasis (p a) (s a)
    hpure : ∀ (a : α) (i : ι a), p a i → Membership.mem (s a i) a
    hopen : ∀ (a : α) (i : ι a), p a i → Filter.Eventually (fun x => Membership.me …
    a : α
    t : TopologicalSpace α := TopologicalSpace.mkOfNhds n
    ⊢ Eq (nhds a) (n a)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      n : α → Filter α
      ι : α → Sort u_1
      p : (a : α) → ι a → Prop
      s : (a : α) → ι a → Set α
      hb : ∀ (a : α), (n a).HasBasis (p a) (s a)
      hpure : ∀ (a : α) (i : ι a), p a i → Membership.mem (s a i) a
      hopen : ∀ (a : α) (i : ι a), p a i → Filter.Eventually (fun x => Membership.me …
      a : α
      t : TopologicalSpace α := TopologicalSpace.mkOfNhds n
      ⊢ LE.le (nhds a) (n a)
    -/
  · intro U hU
    /-
      case a
      α : Type u
      n : α → Filter α
      ι : α → Sort u_1
      p : (a : α) → ι a → Prop
      s : (a : α) → ι a → Set α
      hb : ∀ (a : α), (n a).HasBasis (p a) (s a)
      hpure : ∀ (a : α) (i : ι a), p a i → Membership.mem (s a i) a
      hopen : ∀ (a : α) (i : ι a), p a i → Filter.Eventually (fun x => Membership.me …
      a : α
      t : TopologicalSpace α := TopologicalSpace.mkOfNhds n
      U : Set α
      hU : Membership.mem (n a) U
      ⊢ Membership.mem (nhds a) U
    -/
    replace hpure : pure ≤ n := fun x ↦ (hb x).ge_iff.2 (hpure x)
    /-
      case a
      α : Type u
      n : α → Filter α
      ι : α → Sort u_1
      p : (a : α) → ι a → Prop
      s : (a : α) → ι a → Set α
      hb : ∀ (a : α), (n a).HasBasis (p a) (s a)
      hopen : ∀ (a : α) (i : ι a), p a i → Filter.Eventually (fun x => Membership.me …
      a : α
      t : TopologicalSpace α := TopologicalSpace.mkOfNhds n
      U : Set α
      hU : Membership.mem (n a) U
      hpure : LE.le Pure.pure n
      ⊢ Membership.mem (nhds a) U
    -/
    refine mem_nhds_iff.2 ⟨{x | U ∈ n x}, fun x hx ↦ hpure x hx, fun x hx ↦ ?_, hU⟩
    /-
      case a
      α : Type u
      n : α → Filter α
      ι : α → Sort u_1
      p : (a : α) → ι a → Prop
      s : (a : α) → ι a → Set α
      hb : ∀ (a : α), (n a).HasBasis (p a) (s a)
      hopen : ∀ (a : α) (i : ι a), p a i → Filter.Eventually (fun x => Membership.me …
      a : α
      t : TopologicalSpace α := TopologicalSpace.mkOfNhds n
      U : Set α
      hU : Membership.mem (n a) U
      hpure : LE.le Pure.pure n
      x : α
      hx : Membership.mem (setOf fun x => Membership.mem (n x) U) x
      ⊢ Membership.mem (n x) (setOf fun x => Membership.mem (n x) U)
    -/
    rcases (hb x).mem_iff.1 hx with ⟨i, hpi, hi⟩
    /-
      case a.intro.intro
      α : Type u
      n : α → Filter α
      ι : α → Sort u_1
      p : (a : α) → ι a → Prop
      s : (a : α) → ι a → Set α
      hb : ∀ (a : α), (n a).HasBasis (p a) (s a)
      hopen : ∀ (a : α) (i : ι a), p a i → Filter.Eventually (fun x => Membership.me …
      a : α
      t : TopologicalSpace α := TopologicalSpace.mkOfNhds n
      U : Set α
      hU : Membership.mem (n a) U
      hpure : LE.le Pure.pure n
      x : α
      hx : Membership.mem (setOf fun x => Membership.mem (n x) U) x
      i : ι x
      hpi : p x i
      hi : HasSubset.Subset (s x i) U
      ⊢ Membership.mem (n x) (setOf fun x => Membership.mem (n x) U)
    -/
    exact (hopen x i hpi).mono fun y hy ↦ mem_of_superset hy hi
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u
      n : α → Filter α
      ι : α → Sort u_1
      p : (a : α) → ι a → Prop
      s : (a : α) → ι a → Set α
      hb : ∀ (a : α), (n a).HasBasis (p a) (s a)
      hpure : ∀ (a : α) (i : ι a), p a i → Membership.mem (s a i) a
      hopen : ∀ (a : α) (i : ι a), p a i → Filter.Eventually (fun x => Membership.me …
      a : α
      t : TopologicalSpace α := TopologicalSpace.mkOfNhds n
      ⊢ LE.le (n a) (nhds a)
    -/
  · exact (nhds_basis_opens a).ge_iff.2 fun U ⟨haU, hUo⟩ ↦ hUo a haU
    /-
      🎉 no goals
    -/


theorem nhds_mkOfNhds (n : α → Filter α) (a : α) (h₀ : pure ≤ n)
    (h₁ : ∀ a, ∀ s ∈ n a, ∀ᶠ y in n a, s ∈ n y) :
    @nhds α (TopologicalSpace.mkOfNhds n) a = n a :=
  nhds_mkOfNhds_of_hasBasis (fun a ↦ (n a).basis_sets) h₀ h₁ _


theorem nhds_mkOfNhds_single [DecidableEq α] {a₀ : α} {l : Filter α} (h : pure a₀ ≤ l) (b : α) :
    @nhds α (TopologicalSpace.mkOfNhds (update pure a₀ l)) b =
      (update pure a₀ l : α → Filter α) b := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a₀ : α
    l : Filter α
    h : LE.le (Pure.pure a₀) l
    b : α
    ⊢ Eq (nhds b) (Function.update Pure.pure a₀ l b)
  -/
  refine nhds_mkOfNhds _ _ (le_update_iff.mpr ⟨h, fun _ _ => le_rfl⟩) fun a s hs => ?_
  /-
    α : Type u
    inst✝ : DecidableEq α
    a₀ : α
    l : Filter α
    h : LE.le (Pure.pure a₀) l
    b a : α
    s : Set α
    hs : Membership.mem (Function.update Pure.pure a₀ l a) s
    ⊢ Filter.Eventually (fun y => Membership.mem (Function.update Pure.pure a₀ l y …
  -/
  rcases eq_or_ne a a₀ with (rfl | ha)
    /-
      case inl
      α : Type u
      inst✝ : DecidableEq α
      l : Filter α
      b a : α
      s : Set α
      h : LE.le (Pure.pure a) l
      hs : Membership.mem (Function.update Pure.pure a l a) s
      ⊢ Filter.Eventually (fun y => Membership.mem (Function.update Pure.pure a l y) …
    -/
  · filter_upwards [hs] with b hb
    /-
      case h
      α : Type u
      inst✝ : DecidableEq α
      l : Filter α
      b✝ a : α
      s : Set α
      h : LE.le (Pure.pure a) l
      hs : Membership.mem (Function.update Pure.pure a l a) s
      b : α
      hb : Membership.mem s b
      ⊢ Membership.mem (Function.update Pure.pure a l b) s
    -/
    rcases eq_or_ne b a with (rfl | hb)
      /-
        case h.inl
        α : Type u
        inst✝ : DecidableEq α
        l : Filter α
        b✝ : α
        s : Set α
        b : α
        hb : Membership.mem s b
        h : LE.le (Pure.pure b) l
        hs : Membership.mem (Function.update Pure.pure b l b) s
        ⊢ Membership.mem (Function.update Pure.pure b l b) s
      -/
    · exact hs
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        α : Type u
        inst✝ : DecidableEq α
        l : Filter α
        b✝ a : α
        s : Set α
        h : LE.le (Pure.pure a) l
        hs : Membership.mem (Function.update Pure.pure a l a) s
        b : α
        hb✝ : Membership.mem s b
        hb : Ne b a
        ⊢ Membership.mem (Function.update Pure.pure a l b) s
      -/
    · rwa [update_of_ne hb]
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u
      inst✝ : DecidableEq α
      a₀ : α
      l : Filter α
      h : LE.le (Pure.pure a₀) l
      b a : α
      s : Set α
      hs : Membership.mem (Function.update Pure.pure a₀ l a) s
      ha : Ne a a₀
      ⊢ Filter.Eventually (fun y => Membership.mem (Function.update Pure.pure a₀ l y …
    -/
  · simpa only [update_of_ne ha, mem_pure, eventually_pure] using hs
    /-
      🎉 no goals
    -/


theorem nhds_mkOfNhds_filterBasis (B : α → FilterBasis α) (a : α) (h₀ : ∀ x, ∀ n ∈ B x, x ∈ n)
    (h₁ : ∀ x, ∀ n ∈ B x, ∃ n₁ ∈ B x, ∀ x' ∈ n₁, ∃ n₂ ∈ B x', n₂ ⊆ n) :
    @nhds α (TopologicalSpace.mkOfNhds fun x => (B x).filter) a = (B a).filter :=
  nhds_mkOfNhds_of_hasBasis (fun a ↦ (B a).hasBasis) h₀ h₁ a


/-- The ordering on topologies on the type `α`. `t ≤ s` if every set open in `s` is also open in `t`
(`t` is finer than `s`). -/
instance : PartialOrder (TopologicalSpace α) :=
  { PartialOrder.lift (fun t => OrderDual.toDual IsOpen[t]) (fun _ _ => TopologicalSpace.ext) with
    le := fun s t => ∀ U, IsOpen[t] U → IsOpen[s] U }


protected theorem le_def {α} {t s : TopologicalSpace α} : t ≤ s ↔ IsOpen[s] ≤ IsOpen[t] :=
  Iff.rfl


theorem le_generateFrom_iff_subset_isOpen {g : Set (Set α)} {t : TopologicalSpace α} :
    t ≤ generateFrom g ↔ g ⊆ { s | IsOpen[t] s } :=
  ⟨fun ht s hs => ht _ <| .basic s hs, fun hg _s hs =>
    hs.recOn (fun _ h => hg h) isOpen_univ (fun _ _ _ _ => IsOpen.inter) fun _ _ => isOpen_sUnion⟩


/-- If `s` equals the collection of open sets in the topology it generates, then `s` defines a
topology. -/
protected def mkOfClosure (s : Set (Set α)) (hs : { u | GenerateOpen s u } = s) :
    TopologicalSpace α where
  IsOpen u := u ∈ s
  isOpen_univ := hs ▸ TopologicalSpace.GenerateOpen.univ
  isOpen_inter := hs ▸ TopologicalSpace.GenerateOpen.inter
  isOpen_sUnion := hs ▸ TopologicalSpace.GenerateOpen.sUnion


theorem mkOfClosure_sets {s : Set (Set α)} {hs : { u | GenerateOpen s u } = s} :
    TopologicalSpace.mkOfClosure s hs = generateFrom s :=
  TopologicalSpace.ext hs.symm


theorem gc_generateFrom (α) :
    GaloisConnection (fun t : TopologicalSpace α => OrderDual.toDual { s | IsOpen[t] s })
      (generateFrom ∘ OrderDual.ofDual) := fun _ _ =>
  le_generateFrom_iff_subset_isOpen.symm


/-- The Galois coinsertion between `TopologicalSpace α` and `(Set (Set α))ᵒᵈ` whose lower part sends
  a topology to its collection of open subsets, and whose upper part sends a collection of subsets
  of `α` to the topology they generate. -/
def gciGenerateFrom (α : Type*) :
    GaloisCoinsertion (fun t : TopologicalSpace α => OrderDual.toDual { s | IsOpen[t] s })
      (generateFrom ∘ OrderDual.ofDual) where
  gc := gc_generateFrom α
  u_l_le _ s hs := TopologicalSpace.GenerateOpen.basic s hs
  choice g hg := TopologicalSpace.mkOfClosure g
    (Subset.antisymm hg <| le_generateFrom_iff_subset_isOpen.1 <| le_rfl)
  choice_eq _ _ := mkOfClosure_sets


/-- Topologies on `α` form a complete lattice, with `⊥` the discrete topology
  and `⊤` the indiscrete topology. The infimum of a collection of topologies
  is the topology generated by all their open sets, while the supremum is the
  topology whose open sets are those sets open in every member of the collection. -/
instance : CompleteLattice (TopologicalSpace α) := (gciGenerateFrom α).liftCompleteLattice


@[mono, gcongr]
theorem generateFrom_anti {α} {g₁ g₂ : Set (Set α)} (h : g₁ ⊆ g₂) :
    generateFrom g₂ ≤ generateFrom g₁ :=
  (gc_generateFrom _).monotone_u h


theorem generateFrom_setOf_isOpen (t : TopologicalSpace α) :
    generateFrom { s | IsOpen[t] s } = t :=
  (gciGenerateFrom α).u_l_eq t


theorem leftInverse_generateFrom :
    LeftInverse generateFrom fun t : TopologicalSpace α => { s | IsOpen[t] s } :=
  (gciGenerateFrom α).u_l_leftInverse


theorem generateFrom_surjective : Surjective (generateFrom : Set (Set α) → TopologicalSpace α) :=
  (gciGenerateFrom α).u_surjective


theorem setOf_isOpen_injective : Injective fun t : TopologicalSpace α => { s | IsOpen[t] s } :=
  (gciGenerateFrom α).l_injective


theorem IsOpen.mono (hs : IsOpen[t₂] s) (h : t₁ ≤ t₂) : IsOpen[t₁] s := h s hs


theorem IsClosed.mono (hs : IsClosed[t₂] s) (h : t₁ ≤ t₂) : IsClosed[t₁] s :=
  (@isOpen_compl_iff α s t₁).mp <| hs.isOpen_compl.mono h


theorem closure.mono (h : t₁ ≤ t₂) : closure[t₁] s ⊆ closure[t₂] s :=
  @closure_minimal _ s (@closure _ t₂ s) t₁ subset_closure (IsClosed.mono isClosed_closure h)


theorem isOpen_implies_isOpen_iff : (∀ s, IsOpen[t₁] s → IsOpen[t₂] s) ↔ t₂ ≤ t₁ :=
  Iff.rfl


/-- The only open sets in the indiscrete topology are the empty set and the whole space. -/
theorem TopologicalSpace.isOpen_top_iff {α} (U : Set α) : IsOpen[⊤] U ↔ U = ∅ ∨ U = univ :=
  ⟨fun h => by
    induction h with
    | basic _ h => exact False.elim h
    | univ => exact .inr rfl
    | inter _ _ _ _ h₁ h₂ =>
      rcases h₁ with (rfl | rfl) <;> rcases h₂ with (rfl | rfl) <;> simp
    | sUnion _ _ ih => exact sUnion_mem_empty_univ ih, by
      /-
        α : Type u_2
        U : Set α
        ⊢ Or (Eq U EmptyCollection.emptyCollection) (Eq U Set.univ) → IsOpen U
      -/
      rintro (rfl | rfl)
      /-
        case inl
        α : Type u_2
        ⊢ IsOpen EmptyCollection.emptyCollection
      -/
      exacts [@isOpen_empty _ ⊤, @isOpen_univ _ ⊤]⟩
      /-
        🎉 no goals
      -/


/-- A topological space is discrete if every set is open, that is,
  its topology equals the discrete topology `⊥`. -/
class DiscreteTopology (α : Type*) [t : TopologicalSpace α] : Prop where
  /-- The `TopologicalSpace` structure on a type with discrete topology is equal to `⊥`. -/
  eq_bot : t = ⊥


theorem discreteTopology_bot (α : Type*) : @DiscreteTopology α ⊥ :=
  @DiscreteTopology.mk α ⊥ rfl


@[simp]
theorem isOpen_discrete (s : Set α) : IsOpen s := (@DiscreteTopology.eq_bot α _).symm ▸ trivial


@[simp] theorem isClosed_discrete (s : Set α) : IsClosed s := ⟨isOpen_discrete _⟩


@[simp] theorem closure_discrete (s : Set α) : closure s = s := (isClosed_discrete _).closure_eq


                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝¹ : TopologicalSpace α
                                                                        inst✝ : DiscreteTopology α
                                                                        s : Set α
                                                                        ⊢ Iff (Dense s) (Eq s Set.univ)
                                                                      -/
@[simp] theorem dense_discrete {s : Set α} : Dense s ↔ s = univ := by simp [dense_iff_closure_eq]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem denseRange_discrete {ι : Type*} {f : ι → α} : DenseRange f ↔ Surjective f := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : DiscreteTopology α
    ι : Type u_3
    f : ι → α
    ⊢ Iff (DenseRange f) (Function.Surjective f)
  -/
  rw [DenseRange, dense_discrete, range_eq_univ]
  /-
    🎉 no goals
  -/


@[nontriviality, continuity, fun_prop]
theorem continuous_of_discreteTopology [TopologicalSpace β] {f : α → β} : Continuous f :=
  continuous_def.2 fun _ _ => isOpen_discrete _


/-- A function to a discrete topological space is continuous if and only if the preimage of every
singleton is open. -/
theorem continuous_discrete_rng {α} [TopologicalSpace α] [TopologicalSpace β] [DiscreteTopology β]
    {f : α → β} : Continuous f ↔ ∀ b : β, IsOpen (f ⁻¹' {b}) :=
  ⟨fun h _ => (isOpen_discrete _).preimage h, fun h => ⟨fun s _ => by
    /-
      β : Type u_2
      α : Type u_3
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : DiscreteTopology β
      f : α → β
      h : ∀ (b : β), IsOpen (Set.preimage f (Singleton.singleton b))
      s : Set β
      x✝ : IsOpen s
      ⊢ IsOpen (Set.preimage f s)
    -/
    rw [← biUnion_of_singleton s, preimage_iUnion₂]
    /-
      β : Type u_2
      α : Type u_3
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : DiscreteTopology β
      f : α → β
      h : ∀ (b : β), IsOpen (Set.preimage f (Singleton.singleton b))
      s : Set β
      x✝ : IsOpen s
      ⊢ IsOpen (Set.iUnion fun i => Set.iUnion fun j => Set.preimage f (Singleton.si …
    -/
    exact isOpen_biUnion fun _ _ => h _⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem nhds_discrete (α : Type*) [TopologicalSpace α] [DiscreteTopology α] : @nhds α _ = pure :=
  le_antisymm (fun _ s hs => (isOpen_discrete s).mem_nhds hs) pure_le_nhds


theorem mem_nhds_discrete {x : α} {s : Set α} :
                          /-
                            α : Type u_1
                            inst✝¹ : TopologicalSpace α
                            inst✝ : DiscreteTopology α
                            x : α
                            s : Set α
                            ⊢ Iff (Membership.mem (nhds x) s) (Membership.mem s x)
                          -/
    s ∈ 𝓝 x ↔ x ∈ s := by rw [nhds_discrete, mem_pure]
                          /-
                            🎉 no goals
                          -/


theorem le_of_nhds_le_nhds (h : ∀ x, @nhds α t₁ x ≤ @nhds α t₂ x) : t₁ ≤ t₂ := fun s => by
  /-
    α : Type u_1
    t₁ t₂ : TopologicalSpace α
    h : ∀ (x : α), LE.le (nhds x) (nhds x)
    s : Set α
    ⊢ IsOpen s → IsOpen s
  -/
  rw [@isOpen_iff_mem_nhds _ _ t₁, @isOpen_iff_mem_nhds α _ t₂]
  /-
    α : Type u_1
    t₁ t₂ : TopologicalSpace α
    h : ∀ (x : α), LE.le (nhds x) (nhds x)
    s : Set α
    ⊢ (∀ (x : α), Membership.mem s x → Membership.mem (nhds x) s) → ∀ (x : α), Mem …
  -/
  exact fun hs a ha => h _ (hs _ ha)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-01")]
alias eq_of_nhds_eq_nhds := TopologicalSpace.ext_nhds


theorem eq_bot_of_singletons_open {t : TopologicalSpace α} (h : ∀ x, IsOpen[t] {x}) : t = ⊥ :=
  bot_unique fun s _ => biUnion_of_singleton s ▸ isOpen_biUnion fun x _ => h x


theorem forall_open_iff_discrete {X : Type*} [TopologicalSpace X] :
    (∀ s : Set X, IsOpen s) ↔ DiscreteTopology X :=
  ⟨fun h => ⟨eq_bot_of_singletons_open fun _ => h _⟩, @isOpen_discrete _ _⟩


theorem discreteTopology_iff_forall_isClosed [TopologicalSpace α] :
    DiscreteTopology α ↔ ∀ s : Set α, IsClosed s :=
  forall_open_iff_discrete.symm.trans <| compl_surjective.forall.trans <| forall_congr' fun _ ↦
    isOpen_compl_iff


theorem singletons_open_iff_discrete {X : Type*} [TopologicalSpace X] :
    (∀ a : X, IsOpen ({a} : Set X)) ↔ DiscreteTopology X :=
  ⟨fun h => ⟨eq_bot_of_singletons_open h⟩, fun a _ => @isOpen_discrete _ _ a _⟩


theorem DiscreteTopology.of_finite_of_isClosed_singleton [TopologicalSpace α] [Finite α]
    (h : ∀ a : α, IsClosed {a}) : DiscreteTopology α :=
  discreteTopology_iff_forall_isClosed.mpr fun s ↦
    s.iUnion_of_singleton_coe ▸ isClosed_iUnion_of_finite fun _ ↦ h _


theorem discreteTopology_iff_singleton_mem_nhds [TopologicalSpace α] :
    DiscreteTopology α ↔ ∀ x : α, {x} ∈ 𝓝 x := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ Iff (DiscreteTopology α) (∀ (x : α), Membership.mem (nhds x) (Singleton.sing …
  -/
  simp only [← singletons_open_iff_discrete, isOpen_iff_mem_nhds, mem_singleton_iff, forall_eq]
  /-
    🎉 no goals
  -/


/-- This lemma characterizes discrete topological spaces as those whose singletons are
neighbourhoods. -/
theorem discreteTopology_iff_nhds [TopologicalSpace α] :
    DiscreteTopology α ↔ ∀ x : α, 𝓝 x = pure x := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ Iff (DiscreteTopology α) (∀ (x : α), Eq (nhds x) (Pure.pure x))
  -/
  simp [discreteTopology_iff_singleton_mem_nhds, le_pure_iff]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ Iff (∀ (x : α), Membership.mem (nhds x) (Singleton.singleton x)) (∀ (x : α), …
  -/
  apply forall_congr' (fun x ↦ ?_)
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    ⊢ Iff (Membership.mem (nhds x) (Singleton.singleton x)) (Eq (nhds x) (Pure.pur …
  -/
  simp [le_antisymm_iff, pure_le_nhds x]
  /-
    🎉 no goals
  -/


theorem discreteTopology_iff_nhds_ne [TopologicalSpace α] :
    DiscreteTopology α ↔ ∀ x : α, 𝓝[≠] x = ⊥ := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ Iff (DiscreteTopology α) (∀ (x : α), Eq (nhdsWithin x (HasCompl.compl (Singl …
  -/
  simp only [discreteTopology_iff_singleton_mem_nhds, nhdsWithin, inf_principal_eq_bot, compl_compl]
  /-
    🎉 no goals
  -/


/-- If the codomain of a continuous injective function has discrete topology,
then so does the domain.

See also `Embedding.discreteTopology` for an important special case. -/
theorem DiscreteTopology.of_continuous_injective
    {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [DiscreteTopology β] {f : α → β}
    (hc : Continuous f) (hinj : Injective f) : DiscreteTopology α :=
  forall_open_iff_discrete.1 fun s ↦ hinj.preimage_image s ▸ (isOpen_discrete _).preimage hc


theorem isOpen_induced_iff [t : TopologicalSpace β] {s : Set α} {f : α → β} :
    IsOpen[t.induced f] s ↔ ∃ t, IsOpen t ∧ f ⁻¹' t = s :=
  Iff.rfl


theorem isClosed_induced_iff [t : TopologicalSpace β] {s : Set α} {f : α → β} :
    IsClosed[t.induced f] s ↔ ∃ t, IsClosed t ∧ f ⁻¹' t = s := by
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace β
    s : Set α
    f : α → β
    ⊢ Iff (IsClosed s) (Exists fun t_1 => And (IsClosed t_1) (Eq (Set.preimage f t …
  -/
  letI := t.induced f
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace β
    s : Set α
    f : α → β
    this : TopologicalSpace α := TopologicalSpace.induced f t
    ⊢ Iff (IsClosed s) (Exists fun t_1 => And (IsClosed t_1) (Eq (Set.preimage f t …
  -/
  simp only [← isOpen_compl_iff, isOpen_induced_iff]
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace β
    s : Set α
    f : α → β
    this : TopologicalSpace α := TopologicalSpace.induced f t
    ⊢ Iff (Exists fun t_1 => And (IsOpen t_1) (Eq (Set.preimage f t_1) (HasCompl.c …
  -/
  exact compl_surjective.exists.trans (by simp only [preimage_compl, compl_inj_iff])
  /-
    🎉 no goals
  -/


theorem isOpen_coinduced {t : TopologicalSpace α} {s : Set β} {f : α → β} :
    IsOpen[t.coinduced f] s ↔ IsOpen (f ⁻¹' s) :=
  Iff.rfl


theorem isClosed_coinduced {t : TopologicalSpace α} {s : Set β} {f : α → β} :
    IsClosed[t.coinduced f] s ↔ IsClosed (f ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace α
    s : Set β
    f : α → β
    ⊢ Iff (IsClosed s) (IsClosed (Set.preimage f s))
  -/
  simp only [← isOpen_compl_iff, isOpen_coinduced (f := f), preimage_compl]
  /-
    🎉 no goals
  -/


theorem preimage_nhds_coinduced [TopologicalSpace α] {π : α → β} {s : Set β} {a : α}
    (hs : s ∈ @nhds β (TopologicalSpace.coinduced π ‹_›) (π a)) : π ⁻¹' s ∈ 𝓝 a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    π : α → β
    s : Set β
    a : α
    hs : Membership.mem (nhds (π a)) s
    ⊢ Membership.mem (nhds a) (Set.preimage π s)
  -/
  letI := TopologicalSpace.coinduced π ‹_›
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    π : α → β
    s : Set β
    a : α
    hs : Membership.mem (nhds (π a)) s
    this : TopologicalSpace β := TopologicalSpace.coinduced π inst✝
    ⊢ Membership.mem (nhds a) (Set.preimage π s)
  -/
  rcases mem_nhds_iff.mp hs with ⟨V, hVs, V_op, mem_V⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace α
    π : α → β
    s : Set β
    a : α
    hs : Membership.mem (nhds (π a)) s
    this : TopologicalSpace β := TopologicalSpace.coinduced π inst✝
    V : Set β
    hVs : HasSubset.Subset V s
    V_op : IsOpen V
    mem_V : Membership.mem V (π a)
    ⊢ Membership.mem (nhds a) (Set.preimage π s)
  -/
  exact mem_nhds_iff.mpr ⟨π ⁻¹' V, Set.preimage_mono hVs, V_op, mem_V⟩
  /-
    🎉 no goals
  -/


theorem Continuous.coinduced_le (h : Continuous[t, t'] f) : t.coinduced f ≤ t' :=
  (@continuous_def α β t t').1 h


theorem coinduced_le_iff_le_induced {f : α → β} {tα : TopologicalSpace α}
    {tβ : TopologicalSpace β} : tα.coinduced f ≤ tβ ↔ tα ≤ tβ.induced f :=
  ⟨fun h _s ⟨_t, ht, hst⟩ => hst ▸ h _ ht, fun h s hs => h _ ⟨s, hs, rfl⟩⟩


theorem Continuous.le_induced (h : Continuous[t, t'] f) : t ≤ t'.induced f :=
  coinduced_le_iff_le_induced.1 h.coinduced_le


theorem gc_coinduced_induced (f : α → β) :
    GaloisConnection (TopologicalSpace.coinduced f) (TopologicalSpace.induced f) := fun _ _ =>
  coinduced_le_iff_le_induced


theorem induced_mono (h : t₁ ≤ t₂) : t₁.induced g ≤ t₂.induced g :=
  (gc_coinduced_induced g).monotone_u h


theorem coinduced_mono (h : t₁ ≤ t₂) : t₁.coinduced f ≤ t₂.coinduced f :=
  (gc_coinduced_induced f).monotone_l h


@[simp]
theorem induced_top : (⊤ : TopologicalSpace α).induced g = ⊤ :=
  (gc_coinduced_induced g).u_top


@[simp]
theorem induced_inf : (t₁ ⊓ t₂).induced g = t₁.induced g ⊓ t₂.induced g :=
  (gc_coinduced_induced g).u_inf


@[simp]
theorem induced_iInf {ι : Sort w} {t : ι → TopologicalSpace α} :
    (⨅ i, t i).induced g = ⨅ i, (t i).induced g :=
  (gc_coinduced_induced g).u_iInf


@[simp]
theorem induced_sInf {s : Set (TopologicalSpace α)} :
    TopologicalSpace.induced g (sInf s) = sInf (TopologicalSpace.induced g '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    g : β → α
    s : Set (TopologicalSpace α)
    ⊢ Eq (TopologicalSpace.induced g (InfSet.sInf s)) (InfSet.sInf (Set.image (Top …
  -/
  rw [sInf_eq_iInf', sInf_image', induced_iInf]
  /-
    🎉 no goals
  -/


@[simp]
theorem coinduced_bot : (⊥ : TopologicalSpace α).coinduced f = ⊥ :=
  (gc_coinduced_induced f).l_bot


@[simp]
theorem coinduced_sup : (t₁ ⊔ t₂).coinduced f = t₁.coinduced f ⊔ t₂.coinduced f :=
  (gc_coinduced_induced f).l_sup


@[simp]
theorem coinduced_iSup {ι : Sort w} {t : ι → TopologicalSpace α} :
    (⨆ i, t i).coinduced f = ⨆ i, (t i).coinduced f :=
  (gc_coinduced_induced f).l_iSup


@[simp]
theorem coinduced_sSup {s : Set (TopologicalSpace α)} :
    TopologicalSpace.coinduced f (sSup s) = sSup ((TopologicalSpace.coinduced f) '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set (TopologicalSpace α)
    ⊢ Eq (TopologicalSpace.coinduced f (SupSet.sSup s)) (SupSet.sSup (Set.image (T …
  -/
  rw [sSup_eq_iSup', sSup_image', coinduced_iSup]
  /-
    🎉 no goals
  -/


theorem induced_id [t : TopologicalSpace α] : t.induced id = t :=
  TopologicalSpace.ext <|
    funext fun s => propext <| ⟨fun ⟨_, hs, h⟩ => h ▸ hs, fun hs => ⟨s, hs, rfl⟩⟩


theorem induced_compose {tγ : TopologicalSpace γ} {f : α → β} {g : β → γ} :
    (tγ.induced g).induced f = tγ.induced (g ∘ f) :=
  TopologicalSpace.ext <|
    funext fun _ => propext
      ⟨fun ⟨_, ⟨s, hs, h₂⟩, h₁⟩ => h₁ ▸ h₂ ▸ ⟨s, hs, rfl⟩,
        fun ⟨s, hs, h⟩ => ⟨preimage g s, ⟨s, hs, rfl⟩, h ▸ rfl⟩⟩


theorem induced_const [t : TopologicalSpace α] {x : α} : (t.induced fun _ : β => x) = ⊤ :=
  le_antisymm le_top (@continuous_const β α ⊤ t x).le_induced


theorem coinduced_id [t : TopologicalSpace α] : t.coinduced id = t :=
  TopologicalSpace.ext rfl


theorem coinduced_compose [tα : TopologicalSpace α] {f : α → β} {g : β → γ} :
    (tα.coinduced f).coinduced g = tα.coinduced (g ∘ f) :=
  TopologicalSpace.ext rfl


theorem Equiv.induced_symm {α β : Type*} (e : α ≃ β) :
    TopologicalSpace.induced e.symm = TopologicalSpace.coinduced e := by
  /-
    α : Type u_4
    β : Type u_5
    e : Equiv α β
    ⊢ Eq (TopologicalSpace.induced ⇑e.symm) (TopologicalSpace.coinduced ⇑e)
  -/
  ext t U
  /-
    case h.a.h.a
    α : Type u_4
    β : Type u_5
    e : Equiv α β
    t : TopologicalSpace α
    U : Set β
    ⊢ Iff (IsOpen U) (IsOpen U)
  -/
  rw [isOpen_induced_iff, isOpen_coinduced]
  /-
    case h.a.h.a
    α : Type u_4
    β : Type u_5
    e : Equiv α β
    t : TopologicalSpace α
    U : Set β
    ⊢ Iff (Exists fun t_1 => And (IsOpen t_1) (Eq (Set.preimage (⇑e.symm) t_1) U)) …
  -/
  simp only [e.symm.preimage_eq_iff_eq_image, exists_eq_right, ← preimage_equiv_eq_image_symm]
  /-
    🎉 no goals
  -/


theorem Equiv.coinduced_symm {α β : Type*} (e : α ≃ β) :
    TopologicalSpace.coinduced e.symm = TopologicalSpace.induced e :=
  e.symm.induced_symm.symm


instance inhabitedTopologicalSpace {α : Type u} : Inhabited (TopologicalSpace α) :=
  ⟨⊥⟩


instance (priority := 100) Subsingleton.uniqueTopologicalSpace [Subsingleton α] :
    Unique (TopologicalSpace α) where
  default := ⊥
  uniq t :=
    eq_bot_of_singletons_open fun x =>
      Subsingleton.set_cases (@isOpen_empty _ t) (@isOpen_univ _ t) ({x} : Set α)


instance (priority := 100) Subsingleton.discreteTopology [t : TopologicalSpace α] [Subsingleton α] :
    DiscreteTopology α :=
  ⟨Unique.eq_default t⟩


instance : TopologicalSpace Empty := ⊥

instance : DiscreteTopology Empty := ⟨rfl⟩


instance : TopologicalSpace PEmpty := ⊥

instance : DiscreteTopology PEmpty := ⟨rfl⟩


instance : TopologicalSpace PUnit := ⊥

instance : DiscreteTopology PUnit := ⟨rfl⟩


instance : TopologicalSpace Bool := ⊥

instance : DiscreteTopology Bool := ⟨rfl⟩


instance : TopologicalSpace ℕ := ⊥

instance : DiscreteTopology ℕ := ⟨rfl⟩


instance : TopologicalSpace ℤ := ⊥

instance : DiscreteTopology ℤ := ⟨rfl⟩


instance {n} : TopologicalSpace (Fin n) := ⊥

instance {n} : DiscreteTopology (Fin n) := ⟨rfl⟩


instance sierpinskiSpace : TopologicalSpace Prop :=
  generateFrom {{True}}


theorem continuous_empty_function [TopologicalSpace α] [TopologicalSpace β] [IsEmpty β]
    (f : α → β) : Continuous f :=
  letI := Function.isEmpty f
  continuous_of_discreteTopology


theorem le_generateFrom {t : TopologicalSpace α} {g : Set (Set α)} (h : ∀ s ∈ g, IsOpen s) :
    t ≤ generateFrom g :=
  le_generateFrom_iff_subset_isOpen.2 h


theorem induced_generateFrom_eq {α β} {b : Set (Set β)} {f : α → β} :
    (generateFrom b).induced f = generateFrom (preimage f '' b) :=
  le_antisymm (le_generateFrom <| forall_mem_image.2 fun s hs => ⟨s, GenerateOpen.basic _ hs, rfl⟩)
    (coinduced_le_iff_le_induced.1 <| le_generateFrom fun _s hs => .basic _ (mem_image_of_mem _ hs))


theorem le_induced_generateFrom {α β} [t : TopologicalSpace α] {b : Set (Set β)} {f : α → β}
    (h : ∀ a : Set β, a ∈ b → IsOpen (f ⁻¹' a)) : t ≤ induced f (generateFrom b) := by
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace α
    b : Set (Set β)
    f : α → β
    h : ∀ (a : Set β), Membership.mem b a → IsOpen (Set.preimage f a)
    ⊢ LE.le t (TopologicalSpace.induced f (TopologicalSpace.generateFrom b))
  -/
  rw [induced_generateFrom_eq]
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace α
    b : Set (Set β)
    f : α → β
    h : ∀ (a : Set β), Membership.mem b a → IsOpen (Set.preimage f a)
    ⊢ LE.le t (TopologicalSpace.generateFrom (Set.image (Set.preimage f) b))
  -/
  apply le_generateFrom
  /-
    case h
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace α
    b : Set (Set β)
    f : α → β
    h : ∀ (a : Set β), Membership.mem b a → IsOpen (Set.preimage f a)
    ⊢ ∀ (s : Set α), Membership.mem (Set.image (Set.preimage f) b) s → IsOpen s
  -/
  simp only [mem_image, and_imp, forall_apply_eq_imp_iff₂, exists_imp]
  /-
    case h
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace α
    b : Set (Set β)
    f : α → β
    h : ∀ (a : Set β), Membership.mem b a → IsOpen (Set.preimage f a)
    ⊢ ∀ (a : Set β), Membership.mem b a → IsOpen (Set.preimage f a)
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma generateFrom_insert_of_generateOpen {α : Type*} {s : Set (Set α)} {t : Set α}
    (ht : GenerateOpen s t) : generateFrom (insert t s) = generateFrom s := by
  /-
    α : Type u_1
    s : Set (Set α)
    t : Set α
    ht : TopologicalSpace.GenerateOpen s t
    ⊢ Eq (TopologicalSpace.generateFrom (Insert.insert t s)) (TopologicalSpace.gen …
  -/
  refine le_antisymm (generateFrom_anti <| subset_insert t s) (le_generateFrom ?_)
  /-
    α : Type u_1
    s : Set (Set α)
    t : Set α
    ht : TopologicalSpace.GenerateOpen s t
    ⊢ ∀ (s_1 : Set α), Membership.mem (Insert.insert t s) s_1 → IsOpen s_1
  -/
  rintro t (rfl | h)
    /-
      case inl
      α : Type u_1
      s : Set (Set α)
      t : Set α
      ht : TopologicalSpace.GenerateOpen s t
      ⊢ IsOpen t
    -/
  · exact ht
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      s : Set (Set α)
      t✝ : Set α
      ht : TopologicalSpace.GenerateOpen s t✝
      t : Set α
      h : Membership.mem s t
      ⊢ IsOpen t
    -/
  · exact isOpen_generateFrom_of_mem h
    /-
      🎉 no goals
    -/


@[simp]
lemma generateFrom_insert_univ {α : Type*} {s : Set (Set α)} :
    generateFrom (insert univ s) = generateFrom s :=
  generateFrom_insert_of_generateOpen .univ


@[simp]
lemma generateFrom_insert_empty {α : Type*} {s : Set (Set α)} :
    generateFrom (insert ∅ s) = generateFrom s := by
  /-
    α : Type u_1
    s : Set (Set α)
    ⊢ Eq (TopologicalSpace.generateFrom (Insert.insert EmptyCollection.emptyCollec …
  -/
  rw [← sUnion_empty]
  /-
    α : Type u_1
    s : Set (Set α)
    ⊢ Eq (TopologicalSpace.generateFrom (Insert.insert EmptyCollection.emptyCollec …
  -/
  exact generateFrom_insert_of_generateOpen (.sUnion ∅ (fun s_1 a ↦ False.elim a))
  /-
    🎉 no goals
  -/


/-- This construction is left adjoint to the operation sending a topology on `α`
  to its neighborhood filter at a fixed point `a : α`. -/
def nhdsAdjoint (a : α) (f : Filter α) : TopologicalSpace α where
  IsOpen s := a ∈ s → s ∈ f
  isOpen_univ _ := univ_mem
  isOpen_inter := fun _s _t hs ht ⟨has, hat⟩ => inter_mem (hs has) (ht hat)
  isOpen_sUnion := fun _k hk ⟨u, hu, hau⟩ => mem_of_superset (hk u hu hau) (subset_sUnion_of_mem hu)


theorem gc_nhds (a : α) : GaloisConnection (nhdsAdjoint a) fun t => @nhds α t a := fun f t => by
  /-
    α : Type u
    a : α
    f : Filter α
    t : TopologicalSpace α
    ⊢ Iff (LE.le (nhdsAdjoint a f) t) (LE.le f ((fun t => nhds a) t))
  -/
  rw [le_nhds_iff]
  /-
    α : Type u
    a : α
    f : Filter α
    t : TopologicalSpace α
    ⊢ Iff (LE.le (nhdsAdjoint a f) t) (∀ (s : Set α), Membership.mem s a → IsOpen  …
  -/
  exact ⟨fun H s hs has => H _ has hs, fun H s has hs => H _ hs has⟩
  /-
    🎉 no goals
  -/


theorem nhds_mono {t₁ t₂ : TopologicalSpace α} {a : α} (h : t₁ ≤ t₂) :
    @nhds α t₁ a ≤ @nhds α t₂ a :=
  (gc_nhds a).monotone_u h


theorem le_iff_nhds {α : Type*} (t t' : TopologicalSpace α) :
    t ≤ t' ↔ ∀ x, @nhds α t x ≤ @nhds α t' x :=
  ⟨fun h _ => nhds_mono h, le_of_nhds_le_nhds⟩


theorem isOpen_singleton_nhdsAdjoint {α : Type*} {a b : α} (f : Filter α) (hb : b ≠ a) :
    IsOpen[nhdsAdjoint a f] {b} := fun h ↦
  absurd h hb.symm


theorem nhds_nhdsAdjoint_same (a : α) (f : Filter α) :
    @nhds α (nhdsAdjoint a f) a = pure a ⊔ f := by
  /-
    α : Type u
    a : α
    f : Filter α
    ⊢ Eq (nhds a) (Max.max (Pure.pure a) f)
  -/
  let _ := nhdsAdjoint a f
  /-
    α : Type u
    a : α
    f : Filter α
    x✝ : TopologicalSpace α := nhdsAdjoint a f
    ⊢ Eq (nhds a) (Max.max (Pure.pure a) f)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      a : α
      f : Filter α
      x✝ : TopologicalSpace α := nhdsAdjoint a f
      ⊢ LE.le (nhds a) (Max.max (Pure.pure a) f)
    -/
  · rintro t ⟨hat : a ∈ t, htf : t ∈ f⟩
    /-
      case a.intro
      α : Type u
      a : α
      f : Filter α
      x✝ : TopologicalSpace α := nhdsAdjoint a f
      t : Set α
      hat : Membership.mem t a
      htf : Membership.mem f t
      ⊢ Membership.mem (nhds a) t
    -/
    exact IsOpen.mem_nhds (fun _ ↦ htf) hat
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u
      a : α
      f : Filter α
      x✝ : TopologicalSpace α := nhdsAdjoint a f
      ⊢ LE.le (Max.max (Pure.pure a) f) (nhds a)
    -/
  · exact sup_le (pure_le_nhds _) ((gc_nhds a).le_u_l f)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-10")]
alias nhdsAdjoint_nhds := nhds_nhdsAdjoint_same


theorem nhds_nhdsAdjoint_of_ne {a b : α} (f : Filter α) (h : b ≠ a) :
    @nhds α (nhdsAdjoint a f) b = pure b :=
  let _ := nhdsAdjoint a f
  (isOpen_singleton_iff_nhds_eq_pure _).1 <| isOpen_singleton_nhdsAdjoint f h


@[deprecated nhds_nhdsAdjoint_of_ne (since := "2024-02-10")]
theorem nhdsAdjoint_nhds_of_ne (a : α) (f : Filter α) {b : α} (h : b ≠ a) :
    @nhds α (nhdsAdjoint a f) b = pure b :=
  nhds_nhdsAdjoint_of_ne f h


theorem nhds_nhdsAdjoint [DecidableEq α] (a : α) (f : Filter α) :
    @nhds α (nhdsAdjoint a f) = update pure a (pure a ⊔ f) :=
  eq_update_iff.2 ⟨nhds_nhdsAdjoint_same .., fun _ ↦ nhds_nhdsAdjoint_of_ne _⟩


theorem le_nhdsAdjoint_iff' {a : α} {f : Filter α} {t : TopologicalSpace α} :
    t ≤ nhdsAdjoint a f ↔ @nhds α t a ≤ pure a ⊔ f ∧ ∀ b ≠ a, @nhds α t b = pure b := by
  classical
  simp_rw [le_iff_nhds, nhds_nhdsAdjoint, forall_update_iff, (pure_le_nhds _).le_iff_eq]


theorem le_nhdsAdjoint_iff {α : Type*} (a : α) (f : Filter α) (t : TopologicalSpace α) :
    t ≤ nhdsAdjoint a f ↔ @nhds α t a ≤ pure a ⊔ f ∧ ∀ b ≠ a, IsOpen[t] {b} := by
  /-
    α : Type u_1
    a : α
    f : Filter α
    t : TopologicalSpace α
    ⊢ Iff (LE.le t (nhdsAdjoint a f)) (And (LE.le (nhds a) (Max.max (Pure.pure a)  …
  -/
  simp only [le_nhdsAdjoint_iff', @isOpen_singleton_iff_nhds_eq_pure α t]
  /-
    🎉 no goals
  -/


theorem nhds_iInf {ι : Sort*} {t : ι → TopologicalSpace α} {a : α} :
    @nhds α (iInf t) a = ⨅ i, @nhds α (t i) a :=
  (gc_nhds a).u_iInf


theorem nhds_sInf {s : Set (TopologicalSpace α)} {a : α} :
    @nhds α (sInf s) a = ⨅ t ∈ s, @nhds α t a :=
  (gc_nhds a).u_sInf

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: timeouts without `b₁ := t₁`

theorem nhds_inf {t₁ t₂ : TopologicalSpace α} {a : α} :
    @nhds α (t₁ ⊓ t₂) a = @nhds α t₁ a ⊓ @nhds α t₂ a :=
  (gc_nhds a).u_inf (b₁ := t₁)


theorem nhds_top {a : α} : @nhds α ⊤ a = ⊤ :=
  (gc_nhds a).u_top


theorem isOpen_sup {t₁ t₂ : TopologicalSpace α} {s : Set α} :
    IsOpen[t₁ ⊔ t₂] s ↔ IsOpen[t₁] s ∧ IsOpen[t₂] s :=
  Iff.rfl


theorem continuous_iff_coinduced_le {t₁ : TopologicalSpace α} {t₂ : TopologicalSpace β} :
    Continuous[t₁, t₂] f ↔ coinduced f t₁ ≤ t₂ :=
  continuous_def


theorem continuous_iff_le_induced {t₁ : TopologicalSpace α} {t₂ : TopologicalSpace β} :
    Continuous[t₁, t₂] f ↔ t₁ ≤ induced f t₂ :=
  Iff.trans continuous_iff_coinduced_le (gc_coinduced_induced f _ _)


lemma continuous_generateFrom_iff {t : TopologicalSpace α} {b : Set (Set β)} :
    Continuous[t, generateFrom b] f ↔ ∀ s ∈ b, IsOpen (f ⁻¹' s) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    t : TopologicalSpace α
    b : Set (Set β)
    ⊢ Iff (Continuous f) (∀ (s : Set β), Membership.mem b s → IsOpen (Set.preimage …
  -/
  rw [continuous_iff_coinduced_le, le_generateFrom_iff_subset_isOpen]
  /-
    α : Type u
    β : Type v
    f : α → β
    t : TopologicalSpace α
    b : Set (Set β)
    ⊢ Iff (HasSubset.Subset b (setOf fun s => IsOpen s)) (∀ (s : Set β), Membershi …
  -/
  simp only [isOpen_coinduced, preimage_id', subset_def, mem_setOf]
  /-
    🎉 no goals
  -/


@[continuity, fun_prop]
theorem continuous_induced_dom {t : TopologicalSpace β} : Continuous[induced f t, t] f :=
  continuous_iff_le_induced.2 le_rfl


theorem continuous_induced_rng {g : γ → α} {t₂ : TopologicalSpace β} {t₁ : TopologicalSpace γ} :
    Continuous[t₁, induced f t₂] g ↔ Continuous[t₁, t₂] (f ∘ g) := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    f : α → β
    g : γ → α
    t₂ : TopologicalSpace β
    t₁ : TopologicalSpace γ
    ⊢ Iff (Continuous g) (Continuous (Function.comp f g))
  -/
  simp only [continuous_iff_le_induced, induced_compose]
  /-
    🎉 no goals
  -/


theorem continuous_coinduced_rng {t : TopologicalSpace α} :
    Continuous[t, coinduced f t] f :=
  continuous_iff_coinduced_le.2 le_rfl


theorem continuous_coinduced_dom {g : β → γ} {t₁ : TopologicalSpace α} {t₂ : TopologicalSpace γ} :
    Continuous[coinduced f t₁, t₂] g ↔ Continuous[t₁, t₂] (g ∘ f) := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    f : α → β
    g : β → γ
    t₁ : TopologicalSpace α
    t₂ : TopologicalSpace γ
    ⊢ Iff (Continuous g) (Continuous (Function.comp g f))
  -/
  simp only [continuous_iff_coinduced_le, coinduced_compose]
  /-
    🎉 no goals
  -/


theorem continuous_le_dom {t₁ t₂ : TopologicalSpace α} {t₃ : TopologicalSpace β} (h₁ : t₂ ≤ t₁)
    (h₂ : Continuous[t₁, t₃] f) : Continuous[t₂, t₃] f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    t₁ t₂ : TopologicalSpace α
    t₃ : TopologicalSpace β
    h₁ : LE.le t₂ t₁
    h₂ : Continuous f
    ⊢ Continuous f
  -/
  rw [continuous_iff_le_induced] at h₂ ⊢
  /-
    α : Type u
    β : Type v
    f : α → β
    t₁ t₂ : TopologicalSpace α
    t₃ : TopologicalSpace β
    h₁ : LE.le t₂ t₁
    h₂ : LE.le t₁ (TopologicalSpace.induced f t₃)
    ⊢ LE.le t₂ (TopologicalSpace.induced f t₃)
  -/
  exact le_trans h₁ h₂
  /-
    🎉 no goals
  -/


theorem continuous_le_rng {t₁ : TopologicalSpace α} {t₂ t₃ : TopologicalSpace β} (h₁ : t₂ ≤ t₃)
    (h₂ : Continuous[t₁, t₂] f) : Continuous[t₁, t₃] f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    t₁ : TopologicalSpace α
    t₂ t₃ : TopologicalSpace β
    h₁ : LE.le t₂ t₃
    h₂ : Continuous f
    ⊢ Continuous f
  -/
  rw [continuous_iff_coinduced_le] at h₂ ⊢
  /-
    α : Type u
    β : Type v
    f : α → β
    t₁ : TopologicalSpace α
    t₂ t₃ : TopologicalSpace β
    h₁ : LE.le t₂ t₃
    h₂ : LE.le (TopologicalSpace.coinduced f t₁) t₂
    ⊢ LE.le (TopologicalSpace.coinduced f t₁) t₃
  -/
  exact le_trans h₂ h₁
  /-
    🎉 no goals
  -/


theorem continuous_sup_dom {t₁ t₂ : TopologicalSpace α} {t₃ : TopologicalSpace β} :
    Continuous[t₁ ⊔ t₂, t₃] f ↔ Continuous[t₁, t₃] f ∧ Continuous[t₂, t₃] f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    t₁ t₂ : TopologicalSpace α
    t₃ : TopologicalSpace β
    ⊢ Iff (Continuous f) (And (Continuous f) (Continuous f))
  -/
  simp only [continuous_iff_le_induced, sup_le_iff]
  /-
    🎉 no goals
  -/


theorem continuous_sup_rng_left {t₁ : TopologicalSpace α} {t₃ t₂ : TopologicalSpace β} :
    Continuous[t₁, t₂] f → Continuous[t₁, t₂ ⊔ t₃] f :=
  continuous_le_rng le_sup_left


theorem continuous_sup_rng_right {t₁ : TopologicalSpace α} {t₃ t₂ : TopologicalSpace β} :
    Continuous[t₁, t₃] f → Continuous[t₁, t₂ ⊔ t₃] f :=
  continuous_le_rng le_sup_right


theorem continuous_sSup_dom {T : Set (TopologicalSpace α)} {t₂ : TopologicalSpace β} :
    Continuous[sSup T, t₂] f ↔ ∀ t ∈ T, Continuous[t, t₂] f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    T : Set (TopologicalSpace α)
    t₂ : TopologicalSpace β
    ⊢ Iff (Continuous f) (∀ (t : TopologicalSpace α), Membership.mem T t → Continu …
  -/
  simp only [continuous_iff_le_induced, sSup_le_iff]
  /-
    🎉 no goals
  -/


theorem continuous_sSup_rng {t₁ : TopologicalSpace α} {t₂ : Set (TopologicalSpace β)}
    {t : TopologicalSpace β} (h₁ : t ∈ t₂) (hf : Continuous[t₁, t] f) :
    Continuous[t₁, sSup t₂] f :=
  continuous_iff_coinduced_le.2 <| le_sSup_of_le h₁ <| continuous_iff_coinduced_le.1 hf


theorem continuous_iSup_dom {t₁ : ι → TopologicalSpace α} {t₂ : TopologicalSpace β} :
    Continuous[iSup t₁, t₂] f ↔ ∀ i, Continuous[t₁ i, t₂] f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ι : Sort u_2
    t₁ : ι → TopologicalSpace α
    t₂ : TopologicalSpace β
    ⊢ Iff (Continuous f) (∀ (i : ι), Continuous f)
  -/
  simp only [continuous_iff_le_induced, iSup_le_iff]
  /-
    🎉 no goals
  -/


theorem continuous_iSup_rng {t₁ : TopologicalSpace α} {t₂ : ι → TopologicalSpace β} {i : ι}
    (h : Continuous[t₁, t₂ i] f) : Continuous[t₁, iSup t₂] f :=
  continuous_sSup_rng ⟨i, rfl⟩ h


theorem continuous_inf_rng {t₁ : TopologicalSpace α} {t₂ t₃ : TopologicalSpace β} :
    Continuous[t₁, t₂ ⊓ t₃] f ↔ Continuous[t₁, t₂] f ∧ Continuous[t₁, t₃] f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    t₁ : TopologicalSpace α
    t₂ t₃ : TopologicalSpace β
    ⊢ Iff (Continuous f) (And (Continuous f) (Continuous f))
  -/
  simp only [continuous_iff_coinduced_le, le_inf_iff]
  /-
    🎉 no goals
  -/


theorem continuous_inf_dom_left {t₁ t₂ : TopologicalSpace α} {t₃ : TopologicalSpace β} :
    Continuous[t₁, t₃] f → Continuous[t₁ ⊓ t₂, t₃] f :=
  continuous_le_dom inf_le_left


theorem continuous_inf_dom_right {t₁ t₂ : TopologicalSpace α} {t₃ : TopologicalSpace β} :
    Continuous[t₂, t₃] f → Continuous[t₁ ⊓ t₂, t₃] f :=
  continuous_le_dom inf_le_right


theorem continuous_sInf_dom {t₁ : Set (TopologicalSpace α)} {t₂ : TopologicalSpace β}
    {t : TopologicalSpace α} (h₁ : t ∈ t₁) :
    Continuous[t, t₂] f → Continuous[sInf t₁, t₂] f :=
  continuous_le_dom <| sInf_le h₁


theorem continuous_sInf_rng {t₁ : TopologicalSpace α} {T : Set (TopologicalSpace β)} :
    Continuous[t₁, sInf T] f ↔ ∀ t ∈ T, Continuous[t₁, t] f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    t₁ : TopologicalSpace α
    T : Set (TopologicalSpace β)
    ⊢ Iff (Continuous f) (∀ (t : TopologicalSpace β), Membership.mem T t → Continu …
  -/
  simp only [continuous_iff_coinduced_le, le_sInf_iff]
  /-
    🎉 no goals
  -/


theorem continuous_iInf_dom {t₁ : ι → TopologicalSpace α} {t₂ : TopologicalSpace β} {i : ι} :
    Continuous[t₁ i, t₂] f → Continuous[iInf t₁, t₂] f :=
  continuous_le_dom <| iInf_le _ _


theorem continuous_iInf_rng {t₁ : TopologicalSpace α} {t₂ : ι → TopologicalSpace β} :
    Continuous[t₁, iInf t₂] f ↔ ∀ i, Continuous[t₁, t₂ i] f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ι : Sort u_2
    t₁ : TopologicalSpace α
    t₂ : ι → TopologicalSpace β
    ⊢ Iff (Continuous f) (∀ (i : ι), Continuous f)
  -/
  simp only [continuous_iff_coinduced_le, le_iInf_iff]
  /-
    🎉 no goals
  -/


@[continuity, fun_prop]
theorem continuous_bot {t : TopologicalSpace β} : Continuous[⊥, t] f :=
  continuous_iff_le_induced.2 bot_le


@[continuity, fun_prop]
theorem continuous_top {t : TopologicalSpace α} : Continuous[t, ⊤] f :=
  continuous_iff_coinduced_le.2 le_top


theorem continuous_id_iff_le {t t' : TopologicalSpace α} : Continuous[t, t'] id ↔ t ≤ t' :=
  @continuous_def _ _ t t' id


theorem continuous_id_of_le {t t' : TopologicalSpace α} (h : t ≤ t') : Continuous[t, t'] id :=
  continuous_id_iff_le.2 h

-- 𝓝 in the induced topology

theorem mem_nhds_induced [T : TopologicalSpace α] (f : β → α) (a : β) (s : Set β) :
    s ∈ @nhds β (TopologicalSpace.induced f T) a ↔ ∃ u ∈ 𝓝 (f a), f ⁻¹' u ⊆ s := by
  /-
    α : Type u
    β : Type v
    T : TopologicalSpace α
    f : β → α
    a : β
    s : Set β
    ⊢ Iff (Membership.mem (nhds a) s) (Exists fun u => And (Membership.mem (nhds ( …
  -/
  letI := T.induced f
  /-
    α : Type u
    β : Type v
    T : TopologicalSpace α
    f : β → α
    a : β
    s : Set β
    this : TopologicalSpace β := TopologicalSpace.induced f T
    ⊢ Iff (Membership.mem (nhds a) s) (Exists fun u => And (Membership.mem (nhds ( …
  -/
  simp_rw [mem_nhds_iff, isOpen_induced_iff]
  /-
    α : Type u
    β : Type v
    T : TopologicalSpace α
    f : β → α
    a : β
    s : Set β
    this : TopologicalSpace β := TopologicalSpace.induced f T
    ⊢ Iff (Exists fun t => And (HasSubset.Subset t s) (And (Exists fun t_1 => And  …
  -/
  constructor
    /-
      case mp
      α : Type u
      β : Type v
      T : TopologicalSpace α
      f : β → α
      a : β
      s : Set β
      this : TopologicalSpace β := TopologicalSpace.induced f T
      ⊢ (Exists fun t => And (HasSubset.Subset t s) (And (Exists fun t_1 => And (IsO …
    -/
  · rintro ⟨u, usub, ⟨v, openv, rfl⟩, au⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      T : TopologicalSpace α
      f : β → α
      a : β
      s : Set β
      this : TopologicalSpace β := TopologicalSpace.induced f T
      v : Set α
      openv : IsOpen v
      usub : HasSubset.Subset (Set.preimage f v) s
      au : Membership.mem (Set.preimage f v) a
      ⊢ Exists fun u => And (Exists fun t => And (HasSubset.Subset t u) (And (IsOpen …
    -/
    exact ⟨v, ⟨v, Subset.rfl, openv, au⟩, usub⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      β : Type v
      T : TopologicalSpace α
      f : β → α
      a : β
      s : Set β
      this : TopologicalSpace β := TopologicalSpace.induced f T
      ⊢ (Exists fun u => And (Exists fun t => And (HasSubset.Subset t u) (And (IsOpe …
    -/
  · rintro ⟨u, ⟨v, vsubu, openv, amem⟩, finvsub⟩
    /-
      case mpr.intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      T : TopologicalSpace α
      f : β → α
      a : β
      s : Set β
      this : TopologicalSpace β := TopologicalSpace.induced f T
      u : Set α
      finvsub : HasSubset.Subset (Set.preimage f u) s
      v : Set α
      vsubu : HasSubset.Subset v u
      openv : IsOpen v
      amem : Membership.mem v (f a)
      ⊢ Exists fun t => And (HasSubset.Subset t s) (And (Exists fun t_1 => And (IsOp …
    -/
    exact ⟨f ⁻¹' v, (Set.preimage_mono vsubu).trans finvsub, ⟨⟨v, openv, rfl⟩, amem⟩⟩
    /-
      🎉 no goals
    -/


theorem nhds_induced [T : TopologicalSpace α] (f : β → α) (a : β) :
    @nhds β (TopologicalSpace.induced f T) a = comap f (𝓝 (f a)) := by
  /-
    α : Type u
    β : Type v
    T : TopologicalSpace α
    f : β → α
    a : β
    ⊢ Eq (nhds a) (Filter.comap f (nhds (f a)))
  -/
  ext s
  /-
    case h
    α : Type u
    β : Type v
    T : TopologicalSpace α
    f : β → α
    a : β
    s : Set β
    ⊢ Iff (Membership.mem (nhds a) s) (Membership.mem (Filter.comap f (nhds (f a)) …
  -/
  rw [mem_nhds_induced, mem_comap]
  /-
    🎉 no goals
  -/


theorem induced_iff_nhds_eq [tα : TopologicalSpace α] [tβ : TopologicalSpace β] (f : β → α) :
    tβ = tα.induced f ↔ ∀ b, 𝓝 b = comap f (𝓝 <| f b) := by
  /-
    α : Type u
    β : Type v
    tα : TopologicalSpace α
    tβ : TopologicalSpace β
    f : β → α
    ⊢ Iff (Eq tβ (TopologicalSpace.induced f tα)) (∀ (b : β), Eq (nhds b) (Filter. …
  -/
  simp only [ext_iff_nhds, nhds_induced]
  /-
    🎉 no goals
  -/


theorem map_nhds_induced_of_surjective [T : TopologicalSpace α] {f : β → α} (hf : Surjective f)
    (a : β) : map f (@nhds β (TopologicalSpace.induced f T) a) = 𝓝 (f a) := by
  /-
    α : Type u
    β : Type v
    T : TopologicalSpace α
    f : β → α
    hf : Function.Surjective f
    a : β
    ⊢ Eq (Filter.map f (nhds a)) (nhds (f a))
  -/
  rw [nhds_induced, map_comap_of_surjective hf]
  /-
    🎉 no goals
  -/


theorem continuous_nhdsAdjoint_dom [TopologicalSpace β] {f : α → β} {a : α} {l : Filter α} :
    Continuous[nhdsAdjoint a l, _] f ↔ Tendsto f l (𝓝 (f a)) := by
  /-
    α : Type u
    β : Type v
    inst✝ : TopologicalSpace β
    f : α → β
    a : α
    l : Filter α
    ⊢ Iff (Continuous f) (Filter.Tendsto f l (nhds (f a)))
  -/
  simp_rw [continuous_iff_le_induced, gc_nhds _ _, nhds_induced, tendsto_iff_comap]
  /-
    🎉 no goals
  -/


theorem coinduced_nhdsAdjoint (f : α → β) (a : α) (l : Filter α) :
    coinduced f (nhdsAdjoint a l) = nhdsAdjoint (f a) (map f l) :=
  eq_of_forall_ge_iff fun _ ↦ by
    /-
      α : Type u
      β : Type v
      f : α → β
      a : α
      l : Filter α
      x✝ : TopologicalSpace β
      ⊢ Iff (LE.le (TopologicalSpace.coinduced f (nhdsAdjoint a l)) x✝) (LE.le (nhds …
    -/
    rw [gc_nhds, ← continuous_iff_coinduced_le, continuous_nhdsAdjoint_dom, Tendsto]
    /-
      🎉 no goals
    -/


theorem isOpen_induced_eq {s : Set α} :
    IsOpen[induced f t] s ↔ s ∈ preimage f '' { s | IsOpen s } :=
  Iff.rfl


theorem isOpen_induced {s : Set β} (h : IsOpen s) : IsOpen[induced f t] (f ⁻¹' s) :=
  ⟨s, h, rfl⟩


theorem map_nhds_induced_eq (a : α) : map f (@nhds α (induced f t) a) = 𝓝[range f] f a := by
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace β
    f : α → β
    a : α
    ⊢ Eq (Filter.map f (nhds a)) (nhdsWithin (f a) (Set.range f))
  -/
  rw [nhds_induced, Filter.map_comap, nhdsWithin]
  /-
    🎉 no goals
  -/


theorem map_nhds_induced_of_mem {a : α} (h : range f ∈ 𝓝 (f a)) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      t : TopologicalSpace β
                                                      f : α → β
                                                      a : α
                                                      h : Membership.mem (nhds (f a)) (Set.range f)
                                                      ⊢ Eq (Filter.map f (nhds a)) (nhds (f a))
                                                    -/
    map f (@nhds α (induced f t) a) = 𝓝 (f a) := by rw [nhds_induced, Filter.map_comap_of_mem h]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem closure_induced {f : α → β} {a : α} {s : Set α} :
    a ∈ @closure α (t.induced f) s ↔ f a ∈ closure (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace β
    f : α → β
    a : α
    s : Set α
    ⊢ Iff (Membership.mem (closure s) a) (Membership.mem (closure (Set.image f s)) …
  -/
  letI := t.induced f
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace β
    f : α → β
    a : α
    s : Set α
    this : TopologicalSpace α := TopologicalSpace.induced f t
    ⊢ Iff (Membership.mem (closure s) a) (Membership.mem (closure (Set.image f s)) …
  -/
  simp only [mem_closure_iff_frequently, nhds_induced, frequently_comap, mem_image, and_comm]
  /-
    🎉 no goals
  -/


theorem isClosed_induced_iff' {f : α → β} {s : Set α} :
    IsClosed[t.induced f] s ↔ ∀ a, f a ∈ closure (f '' s) → a ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace β
    f : α → β
    s : Set α
    ⊢ Iff (IsClosed s) (∀ (a : α), Membership.mem (closure (Set.image f s)) (f a)  …
  -/
  letI := t.induced f
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace β
    f : α → β
    s : Set α
    this : TopologicalSpace α := TopologicalSpace.induced f t
    ⊢ Iff (IsClosed s) (∀ (a : α), Membership.mem (closure (Set.image f s)) (f a)  …
  -/
  simp only [← closure_subset_iff_isClosed, subset_def, closure_induced]
  /-
    🎉 no goals
  -/


@[simp]
theorem isOpen_singleton_true : IsOpen ({True} : Set Prop) :=
  TopologicalSpace.GenerateOpen.basic _ (mem_singleton _)


@[simp]
theorem nhds_true : 𝓝 True = pure True :=
  le_antisymm (le_pure_iff.2 <| isOpen_singleton_true.mem_nhds <| mem_singleton _) (pure_le_nhds _)


@[simp]
theorem nhds_false : 𝓝 False = ⊤ :=
                                                 /-
                                                   ⊢ Eq (iInf fun s => iInf fun h => Filter.principal s) Top.top
                                                 -/
  TopologicalSpace.nhds_generateFrom.trans <| by simp [@and_comm (_ ∈ _), iInter_and]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem tendsto_nhds_true {l : Filter α} {p : α → Prop} :
                                                /-
                                                  α : Type u_1
                                                  l : Filter α
                                                  p : α → Prop
                                                  ⊢ Iff (Filter.Tendsto p l (nhds True)) (Filter.Eventually (fun x => p x) l)
                                                -/
    Tendsto p l (𝓝 True) ↔ ∀ᶠ x in l, p x := by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem tendsto_nhds_Prop {l : Filter α} {p : α → Prop} {q : Prop} :
    Tendsto p l (𝓝 q) ↔ (q → ∀ᶠ x in l, p x) := by
  /-
    α : Type u_1
    l : Filter α
    p : α → Prop
    q : Prop
    ⊢ Iff (Filter.Tendsto p l (nhds q)) (q → Filter.Eventually (fun x => p x) l)
  -/
                 /-
                   🎉 no goals
                 -/
  by_cases q <;> simp [*]
                 /-
                   🎉 no goals
                 -/


theorem continuous_Prop {p : α → Prop} : Continuous p ↔ IsOpen { x | p x } := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    p : α → Prop
    ⊢ Iff (Continuous p) (IsOpen (setOf fun x => p x))
  -/
  simp only [continuous_iff_continuousAt, ContinuousAt, tendsto_nhds_Prop, isOpen_iff_mem_nhds]; rfl
                                                                                                 /-
                                                                                                   🎉 no goals
                                                                                                 -/


theorem isOpen_iff_continuous_mem {s : Set α} : IsOpen s ↔ Continuous (· ∈ s) :=
  continuous_Prop.symm


theorem generateFrom_union (a₁ a₂ : Set (Set α)) :
    generateFrom (a₁ ∪ a₂) = generateFrom a₁ ⊓ generateFrom a₂ :=
  (gc_generateFrom α).u_inf


theorem setOf_isOpen_sup (t₁ t₂ : TopologicalSpace α) :
    { s | IsOpen[t₁ ⊔ t₂] s } = { s | IsOpen[t₁] s } ∩ { s | IsOpen[t₂] s } :=
  rfl


theorem generateFrom_iUnion {f : ι → Set (Set α)} :
    generateFrom (⋃ i, f i) = ⨅ i, generateFrom (f i) :=
  (gc_generateFrom α).u_iInf


theorem setOf_isOpen_iSup {t : ι → TopologicalSpace α} :
    { s | IsOpen[⨆ i, t i] s } = ⋂ i, { s | IsOpen[t i] s } :=
  (gc_generateFrom α).l_iSup


theorem generateFrom_sUnion {S : Set (Set (Set α))} :
    generateFrom (⋃₀ S) = ⨅ s ∈ S, generateFrom s :=
  (gc_generateFrom α).u_sInf


theorem setOf_isOpen_sSup {T : Set (TopologicalSpace α)} :
    { s | IsOpen[sSup T] s } = ⋂ t ∈ T, { s | IsOpen[t] s } :=
  (gc_generateFrom α).l_sSup


theorem generateFrom_union_isOpen (a b : TopologicalSpace α) :
    generateFrom ({ s | IsOpen[a] s } ∪ { s | IsOpen[b] s }) = a ⊓ b :=
  (gciGenerateFrom α).u_inf_l _ _


theorem generateFrom_iUnion_isOpen (f : ι → TopologicalSpace α) :
    generateFrom (⋃ i, { s | IsOpen[f i] s }) = ⨅ i, f i :=
  (gciGenerateFrom α).u_iInf_l _


theorem generateFrom_inter (a b : TopologicalSpace α) :
    generateFrom ({ s | IsOpen[a] s } ∩ { s | IsOpen[b] s }) = a ⊔ b :=
  (gciGenerateFrom α).u_sup_l _ _


theorem generateFrom_iInter (f : ι → TopologicalSpace α) :
    generateFrom (⋂ i, { s | IsOpen[f i] s }) = ⨆ i, f i :=
  (gciGenerateFrom α).u_iSup_l _


theorem generateFrom_iInter_of_generateFrom_eq_self (f : ι → Set (Set α))
    (hf : ∀ i, { s | IsOpen[generateFrom (f i)] s } = f i) :
    generateFrom (⋂ i, f i) = ⨆ i, generateFrom (f i) :=
  (gciGenerateFrom α).u_iSup_of_lu_eq_self f hf


theorem isOpen_iSup_iff {s : Set α} : IsOpen[⨆ i, t i] s ↔ ∀ i, IsOpen[t i] s :=
  show s ∈ {s | IsOpen[iSup t] s} ↔ s ∈ { x : Set α | ∀ i : ι, IsOpen[t i] x } by
    /-
      α : Type u
      ι : Sort v
      t : ι → TopologicalSpace α
      s : Set α
      ⊢ Iff (Membership.mem (setOf fun s => IsOpen s) s) (Membership.mem (setOf fun  …
    -/
    simp [setOf_isOpen_iSup]
    /-
      🎉 no goals
    -/


theorem isOpen_sSup_iff {s : Set α} {T : Set (TopologicalSpace α)} :
    IsOpen[sSup T] s ↔ ∀ t ∈ T, IsOpen[t] s := by
  /-
    α : Type u
    s : Set α
    T : Set (TopologicalSpace α)
    ⊢ Iff (IsOpen s) (∀ (t : TopologicalSpace α), Membership.mem T t → IsOpen s)
  -/
  simp only [sSup_eq_iSup, isOpen_iSup_iff]
  /-
    🎉 no goals
  -/


set_option tactic.skipAssignedInstances false in
theorem isClosed_iSup_iff {s : Set α} : IsClosed[⨆ i, t i] s ↔ ∀ i, IsClosed[t i] s := by
  /-
    α : Type u
    ι : Sort v
    t : ι → TopologicalSpace α
    s : Set α
    ⊢ Iff (IsClosed s) (∀ (i : ι), IsClosed s)
  -/
  simp [← @isOpen_compl_iff _ _ (⨆ i, t i), ← @isOpen_compl_iff _ _ (t _), isOpen_iSup_iff]
  /-
    🎉 no goals
  -/


theorem isClosed_sSup_iff {s : Set α} {T : Set (TopologicalSpace α)} :
    IsClosed[sSup T] s ↔ ∀ t ∈ T, IsClosed[t] s := by
  /-
    α : Type u
    s : Set α
    T : Set (TopologicalSpace α)
    ⊢ Iff (IsClosed s) (∀ (t : TopologicalSpace α), Membership.mem T t → IsClosed s)
  -/
  simp only [sSup_eq_iSup, isClosed_iSup_iff]
  /-
    🎉 no goals
  -/


