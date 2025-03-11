/-- Heine-Cantor: a continuous function on a compact uniform space is uniformly
continuous. -/
theorem CompactSpace.uniformContinuous_of_continuous [CompactSpace α] {f : α → β}
    (h : Continuous f) : UniformContinuous f :=
  calc map (Prod.map f f) (𝓤 α)
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   inst✝² : UniformSpace α
                                                   inst✝¹ : UniformSpace β
                                                   inst✝ : CompactSpace α
                                                   f : α → β
                                                   h : Continuous f
                                                   ⊢ Eq (Filter.map (Prod.map f f) (uniformity α)) (Filter.map (Prod.map f f) (nh …
                                                 -/
    = map (Prod.map f f) (𝓝ˢ (diagonal α)) := by rw [nhdsSet_diagonal_eq_uniformity]
                                                 /-
                                                   🎉 no goals
                                                 -/
  _ ≤ 𝓝ˢ (diagonal β) := (h.prodMap h).tendsto_nhdsSet mapsTo_prod_map_diagonal
  _ ≤ 𝓤 β := nhdsSet_diagonal_le_uniformity


/-- Heine-Cantor: a continuous function on a compact set of a uniform space is uniformly
continuous. -/
theorem IsCompact.uniformContinuousOn_of_continuous {s : Set α} {f : α → β} (hs : IsCompact s)
    (hf : ContinuousOn f s) : UniformContinuousOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set α
    f : α → β
    hs : IsCompact s
    hf : ContinuousOn f s
    ⊢ UniformContinuousOn f s
  -/
  rw [uniformContinuousOn_iff_restrict]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set α
    f : α → β
    hs : IsCompact s
    hf : ContinuousOn f s
    ⊢ UniformContinuous (s.restrict f)
  -/
  rw [isCompact_iff_compactSpace] at hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set α
    f : α → β
    hs : CompactSpace ↑s
    hf : ContinuousOn f s
    ⊢ UniformContinuous (s.restrict f)
  -/
  rw [continuousOn_iff_continuous_restrict] at hf
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set α
    f : α → β
    hs : CompactSpace ↑s
    hf : Continuous (s.restrict f)
    ⊢ UniformContinuous (s.restrict f)
  -/
  exact CompactSpace.uniformContinuous_of_continuous hf
  /-
    🎉 no goals
  -/


/-- If `s` is compact and `f` is continuous at all points of `s`, then `f` is
"uniformly continuous at the set `s`", i.e. `f x` is close to `f y` whenever `x ∈ s` and `y` is
close to `x` (even if `y` is not itself in `s`, so this is a stronger assertion than
`UniformContinuousOn s`). -/
theorem IsCompact.uniformContinuousAt_of_continuousAt {r : Set (β × β)} {s : Set α}
    (hs : IsCompact s) (f : α → β) (hf : ∀ a ∈ s, ContinuousAt f a) (hr : r ∈ 𝓤 β) :
    { x : α × α | x.1 ∈ s → (f x.1, f x.2) ∈ r } ∈ 𝓤 α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    r : Set (Prod β β)
    s : Set α
    hs : IsCompact s
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → ContinuousAt f a
    hr : Membership.mem (uniformity β) r
    ⊢ Membership.mem (uniformity α) (setOf fun x => Membership.mem s x.1 → Members …
  -/
  obtain ⟨t, ht, htsymm, htr⟩ := comp_symm_mem_uniformity_sets hr
  choose U hU T hT hb using fun a ha =>
    exists_mem_nhds_ball_subset_of_mem_nhds ((hf a ha).preimage_mem_nhds <| mem_nhds_left _ ht)
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    r : Set (Prod β β)
    s : Set α
    hs : IsCompact s
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → ContinuousAt f a
    hr : Membership.mem (uniformity β) r
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : SymmetricRel t
    htr : HasSubset.Subset (compRel t t) r
    U : (a : α) → Membership.mem s a → Set α
    hU : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (nhds a) (U a ha)
    T : (a : α) → Membership.mem s a → Set (Prod α α)
    hT : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (uniformity α) (T a ha)
    hb : ∀ (a : α) (ha : Membership.mem s a) (a' : α), Membership.mem (U a ha) a'  …
    ⊢ Membership.mem (uniformity α) (setOf fun x => Membership.mem s x.1 → Members …
  -/
  obtain ⟨fs, hsU⟩ := hs.elim_nhds_subcover' U hU
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    r : Set (Prod β β)
    s : Set α
    hs : IsCompact s
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → ContinuousAt f a
    hr : Membership.mem (uniformity β) r
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : SymmetricRel t
    htr : HasSubset.Subset (compRel t t) r
    U : (a : α) → Membership.mem s a → Set α
    hU : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (nhds a) (U a ha)
    T : (a : α) → Membership.mem s a → Set (Prod α α)
    hT : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (uniformity α) (T a ha)
    hb : ∀ (a : α) (ha : Membership.mem s a) (a' : α), Membership.mem (U a ha) a'  …
    fs : Finset ↑s
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
    ⊢ Membership.mem (uniformity α) (setOf fun x => Membership.mem s x.1 → Members …
  -/
  apply mem_of_superset ((biInter_finset_mem fs).2 fun a _ => hT a a.2)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    r : Set (Prod β β)
    s : Set α
    hs : IsCompact s
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → ContinuousAt f a
    hr : Membership.mem (uniformity β) r
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : SymmetricRel t
    htr : HasSubset.Subset (compRel t t) r
    U : (a : α) → Membership.mem s a → Set α
    hU : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (nhds a) (U a ha)
    T : (a : α) → Membership.mem s a → Set (Prod α α)
    hT : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (uniformity α) (T a ha)
    hb : ∀ (a : α) (ha : Membership.mem s a) (a' : α), Membership.mem (U a ha) a'  …
    fs : Finset ↑s
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
    ⊢ HasSubset.Subset (Set.iInter fun i => Set.iInter fun h => T ↑i ⋯) (setOf fun …
  -/
  rintro ⟨a₁, a₂⟩ h h₁
  /-
    case intro.intro.intro.intro.mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    r : Set (Prod β β)
    s : Set α
    hs : IsCompact s
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → ContinuousAt f a
    hr : Membership.mem (uniformity β) r
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : SymmetricRel t
    htr : HasSubset.Subset (compRel t t) r
    U : (a : α) → Membership.mem s a → Set α
    hU : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (nhds a) (U a ha)
    T : (a : α) → Membership.mem s a → Set (Prod α α)
    hT : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (uniformity α) (T a ha)
    hb : ∀ (a : α) (ha : Membership.mem s a) (a' : α), Membership.mem (U a ha) a'  …
    fs : Finset ↑s
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
    a₁ a₂ : α
    h : Membership.mem (Set.iInter fun i => Set.iInter fun h => T ↑i ⋯) { fst := a …
    h₁ : Membership.mem s { fst := a₁, snd := a₂ }.1
    ⊢ Membership.mem r { fst := f { fst := a₁, snd := a₂ }.1, snd := f { fst := a₁ …
  -/
  obtain ⟨a, ha, haU⟩ := Set.mem_iUnion₂.1 (hsU h₁)
  /-
    case intro.intro.intro.intro.mk.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    r : Set (Prod β β)
    s : Set α
    hs : IsCompact s
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → ContinuousAt f a
    hr : Membership.mem (uniformity β) r
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : SymmetricRel t
    htr : HasSubset.Subset (compRel t t) r
    U : (a : α) → Membership.mem s a → Set α
    hU : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (nhds a) (U a ha)
    T : (a : α) → Membership.mem s a → Set (Prod α α)
    hT : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (uniformity α) (T a ha)
    hb : ∀ (a : α) (ha : Membership.mem s a) (a' : α), Membership.mem (U a ha) a'  …
    fs : Finset ↑s
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
    a₁ a₂ : α
    h : Membership.mem (Set.iInter fun i => Set.iInter fun h => T ↑i ⋯) { fst := a …
    h₁ : Membership.mem s { fst := a₁, snd := a₂ }.1
    a : ↑s
    ha : Membership.mem fs a
    haU : Membership.mem (U ↑a ⋯) { fst := a₁, snd := a₂ }.1
    ⊢ Membership.mem r { fst := f { fst := a₁, snd := a₂ }.1, snd := f { fst := a₁ …
  -/
  apply htr
  /-
    case intro.intro.intro.intro.mk.intro.intro.a
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    r : Set (Prod β β)
    s : Set α
    hs : IsCompact s
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → ContinuousAt f a
    hr : Membership.mem (uniformity β) r
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : SymmetricRel t
    htr : HasSubset.Subset (compRel t t) r
    U : (a : α) → Membership.mem s a → Set α
    hU : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (nhds a) (U a ha)
    T : (a : α) → Membership.mem s a → Set (Prod α α)
    hT : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (uniformity α) (T a ha)
    hb : ∀ (a : α) (ha : Membership.mem s a) (a' : α), Membership.mem (U a ha) a'  …
    fs : Finset ↑s
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
    a₁ a₂ : α
    h : Membership.mem (Set.iInter fun i => Set.iInter fun h => T ↑i ⋯) { fst := a …
    h₁ : Membership.mem s { fst := a₁, snd := a₂ }.1
    a : ↑s
    ha : Membership.mem fs a
    haU : Membership.mem (U ↑a ⋯) { fst := a₁, snd := a₂ }.1
    ⊢ Membership.mem (compRel t t) { fst := f { fst := a₁, snd := a₂ }.1, snd := f …
  -/
  refine ⟨f a, htsymm.mk_mem_comm.1 (hb _ _ _ haU ?_), hb _ _ _ haU ?_⟩
  /-
    case intro.intro.intro.intro.mk.intro.intro.a.refine_1
    α : Type u_1
    β : Type u_2
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    r : Set (Prod β β)
    s : Set α
    hs : IsCompact s
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → ContinuousAt f a
    hr : Membership.mem (uniformity β) r
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : SymmetricRel t
    htr : HasSubset.Subset (compRel t t) r
    U : (a : α) → Membership.mem s a → Set α
    hU : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (nhds a) (U a ha)
    T : (a : α) → Membership.mem s a → Set (Prod α α)
    hT : ∀ (a : α) (ha : Membership.mem s a), Membership.mem (uniformity α) (T a ha)
    hb : ∀ (a : α) (ha : Membership.mem s a) (a' : α), Membership.mem (U a ha) a'  …
    fs : Finset ↑s
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
    a₁ a₂ : α
    h : Membership.mem (Set.iInter fun i => Set.iInter fun h => T ↑i ⋯) { fst := a …
    h₁ : Membership.mem s { fst := a₁, snd := a₂ }.1
    a : ↑s
    ha : Membership.mem fs a
    haU : Membership.mem (U ↑a ⋯) { fst := a₁, snd := a₂ }.1
    ⊢ Membership.mem (UniformSpace.ball { fst := a₁, snd := a₂ }.1 (T ↑a ⋯)) { fst …
  -/
  exacts [mem_ball_self _ (hT a a.2), mem_iInter₂.1 h a ha]
  /-
    🎉 no goals
  -/


theorem Continuous.uniformContinuous_of_tendsto_cocompact {f : α → β} {x : β}
    (h_cont : Continuous f) (hx : Tendsto f (cocompact α) (𝓝 x)) : UniformContinuous f :=
  uniformContinuous_def.2 fun r hr => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      x : β
      h_cont : Continuous f
      hx : Filter.Tendsto f (Filter.cocompact α) (nhds x)
      r : Set (Prod β β)
      hr : Membership.mem (uniformity β) r
      ⊢ Membership.mem (uniformity α) (setOf fun x => Membership.mem r { fst := f x. …
    -/
    obtain ⟨t, ht, htsymm, htr⟩ := comp_symm_mem_uniformity_sets hr
    /-
      case intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      x : β
      h_cont : Continuous f
      hx : Filter.Tendsto f (Filter.cocompact α) (nhds x)
      r : Set (Prod β β)
      hr : Membership.mem (uniformity β) r
      t : Set (Prod β β)
      ht : Membership.mem (uniformity β) t
      htsymm : SymmetricRel t
      htr : HasSubset.Subset (compRel t t) r
      ⊢ Membership.mem (uniformity α) (setOf fun x => Membership.mem r { fst := f x. …
    -/
    obtain ⟨s, hs, hst⟩ := mem_cocompact.1 (hx <| mem_nhds_left _ ht)
    apply
      mem_of_superset
        (symmetrize_mem_uniformity <|
          (hs.uniformContinuousAt_of_continuousAt f fun _ _ => h_cont.continuousAt) <|
            symmetrize_mem_uniformity hr)
    /-
      case intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      x : β
      h_cont : Continuous f
      hx : Filter.Tendsto f (Filter.cocompact α) (nhds x)
      r : Set (Prod β β)
      hr : Membership.mem (uniformity β) r
      t : Set (Prod β β)
      ht : Membership.mem (uniformity β) t
      htsymm : SymmetricRel t
      htr : HasSubset.Subset (compRel t t) r
      s : Set α
      hs : IsCompact s
      hst : HasSubset.Subset (HasCompl.compl s) (Set.preimage f (setOf fun y => Memb …
      ⊢ HasSubset.Subset (symmetrizeRel (setOf fun x => Membership.mem s x.1 → Membe …
    -/
    rintro ⟨b₁, b₂⟩ h
    /-
      case intro.intro.intro.intro.intro.mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      x : β
      h_cont : Continuous f
      hx : Filter.Tendsto f (Filter.cocompact α) (nhds x)
      r : Set (Prod β β)
      hr : Membership.mem (uniformity β) r
      t : Set (Prod β β)
      ht : Membership.mem (uniformity β) t
      htsymm : SymmetricRel t
      htr : HasSubset.Subset (compRel t t) r
      s : Set α
      hs : IsCompact s
      hst : HasSubset.Subset (HasCompl.compl s) (Set.preimage f (setOf fun y => Memb …
      b₁ b₂ : α
      h : Membership.mem (symmetrizeRel (setOf fun x => Membership.mem s x.1 → Membe …
      ⊢ Membership.mem (setOf fun x => Membership.mem r { fst := f x.1, snd := f x.2 …
    -/
    by_cases h₁ : b₁ ∈ s; · exact (h.1 h₁).1
                            /-
                              🎉 no goals
                            -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      x : β
      h_cont : Continuous f
      hx : Filter.Tendsto f (Filter.cocompact α) (nhds x)
      r : Set (Prod β β)
      hr : Membership.mem (uniformity β) r
      t : Set (Prod β β)
      ht : Membership.mem (uniformity β) t
      htsymm : SymmetricRel t
      htr : HasSubset.Subset (compRel t t) r
      s : Set α
      hs : IsCompact s
      hst : HasSubset.Subset (HasCompl.compl s) (Set.preimage f (setOf fun y => Memb …
      b₁ b₂ : α
      h : Membership.mem (symmetrizeRel (setOf fun x => Membership.mem s x.1 → Membe …
      h₁ : Not (Membership.mem s b₁)
      ⊢ Membership.mem (setOf fun x => Membership.mem r { fst := f x.1, snd := f x.2 …
    -/
    by_cases h₂ : b₂ ∈ s; · exact (h.2 h₂).2
                            /-
                              🎉 no goals
                            -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      x : β
      h_cont : Continuous f
      hx : Filter.Tendsto f (Filter.cocompact α) (nhds x)
      r : Set (Prod β β)
      hr : Membership.mem (uniformity β) r
      t : Set (Prod β β)
      ht : Membership.mem (uniformity β) t
      htsymm : SymmetricRel t
      htr : HasSubset.Subset (compRel t t) r
      s : Set α
      hs : IsCompact s
      hst : HasSubset.Subset (HasCompl.compl s) (Set.preimage f (setOf fun y => Memb …
      b₁ b₂ : α
      h : Membership.mem (symmetrizeRel (setOf fun x => Membership.mem s x.1 → Membe …
      h₁ : Not (Membership.mem s b₁)
      h₂ : Not (Membership.mem s b₂)
      ⊢ Membership.mem (setOf fun x => Membership.mem r { fst := f x.1, snd := f x.2 …
    -/
    apply htr
    /-
      case neg.a
      α : Type u_1
      β : Type u_2
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      x : β
      h_cont : Continuous f
      hx : Filter.Tendsto f (Filter.cocompact α) (nhds x)
      r : Set (Prod β β)
      hr : Membership.mem (uniformity β) r
      t : Set (Prod β β)
      ht : Membership.mem (uniformity β) t
      htsymm : SymmetricRel t
      htr : HasSubset.Subset (compRel t t) r
      s : Set α
      hs : IsCompact s
      hst : HasSubset.Subset (HasCompl.compl s) (Set.preimage f (setOf fun y => Memb …
      b₁ b₂ : α
      h : Membership.mem (symmetrizeRel (setOf fun x => Membership.mem s x.1 → Membe …
      h₁ : Not (Membership.mem s b₁)
      h₂ : Not (Membership.mem s b₂)
      ⊢ Membership.mem (compRel t t) { fst := f { fst := b₁, snd := b₂ }.1, snd := f …
    -/
    exact ⟨x, htsymm.mk_mem_comm.1 (hst h₁), hst h₂⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem HasCompactMulSupport.uniformContinuous_of_continuous {f : α → β} [One β]
    (h1 : HasCompactMulSupport f) (h2 : Continuous f) : UniformContinuous f :=
  h2.uniformContinuous_of_tendsto_cocompact h1.is_one_at_infty


/-- A family of functions `α → β → γ` tends uniformly to its value at `x` if `α` is locally compact,
`β` is compact and `f` is continuous on `U × (univ : Set β)` for some neighborhood `U` of `x`. -/
theorem ContinuousOn.tendstoUniformly [LocallyCompactSpace α] [CompactSpace β] [UniformSpace γ]
    {f : α → β → γ} {x : α} {U : Set α} (hxU : U ∈ 𝓝 x) (h : ContinuousOn (↿f) (U ×ˢ univ)) :
    TendstoUniformly f (f x) (𝓝 x) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : UniformSpace α
    inst✝³ : UniformSpace β
    inst✝² : LocallyCompactSpace α
    inst✝¹ : CompactSpace β
    inst✝ : UniformSpace γ
    f : α → β → γ
    x : α
    U : Set α
    hxU : Membership.mem (nhds x) U
    h : ContinuousOn (Function.HasUncurry.uncurry f) (SProd.sprod U Set.univ)
    ⊢ TendstoUniformly f (f x) (nhds x)
  -/
  rcases LocallyCompactSpace.local_compact_nhds _ _ hxU with ⟨K, hxK, hKU, hK⟩
  have : UniformContinuousOn (↿f) (K ×ˢ univ) :=
    IsCompact.uniformContinuousOn_of_continuous (hK.prod isCompact_univ)
      (h.mono <| prod_mono hKU Subset.rfl)
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁴ : UniformSpace α
    inst✝³ : UniformSpace β
    inst✝² : LocallyCompactSpace α
    inst✝¹ : CompactSpace β
    inst✝ : UniformSpace γ
    f : α → β → γ
    x : α
    U : Set α
    hxU : Membership.mem (nhds x) U
    h : ContinuousOn (Function.HasUncurry.uncurry f) (SProd.sprod U Set.univ)
    K : Set α
    hxK : Membership.mem (nhds x) K
    hKU : HasSubset.Subset K U
    hK : IsCompact K
    this : UniformContinuousOn (Function.HasUncurry.uncurry f) (SProd.sprod K Set. …
    ⊢ TendstoUniformly f (f x) (nhds x)
  -/
  exact this.tendstoUniformly hxK
  /-
    🎉 no goals
  -/


/-- A continuous family of functions `α → β → γ` tends uniformly to its value at `x`
if `α` is weakly locally compact and `β` is compact. -/
theorem Continuous.tendstoUniformly [WeaklyLocallyCompactSpace α] [CompactSpace β] [UniformSpace γ]
    (f : α → β → γ) (h : Continuous ↿f) (x : α) : TendstoUniformly f (f x) (𝓝 x) :=
  let ⟨K, hK, hxK⟩ := exists_compact_mem_nhds x
  have : UniformContinuousOn (↿f) (K ×ˢ univ) :=
    IsCompact.uniformContinuousOn_of_continuous (hK.prod isCompact_univ) h.continuousOn
  this.tendstoUniformly hxK


/-- In a product space `α × β`, assume that a function `f` is continuous on `s × k` where `k` is
compact. Then, along the fiber above any `q ∈ s`, `f` is transversely uniformly continuous, i.e.,
if `p ∈ s` is close enough to `q`, then `f p x` is uniformly close to `f q x` for all `x ∈ k`. -/
lemma IsCompact.mem_uniformity_of_prod
    {α β E : Type*} [TopologicalSpace α] [TopologicalSpace β] [UniformSpace E]
    {f : α → β → E} {s : Set α} {k : Set β} {q : α} {u : Set (E × E)}
    (hk : IsCompact k) (hf : ContinuousOn f.uncurry (s ×ˢ k)) (hq : q ∈ s) (hu : u ∈ 𝓤 E) :
    ∃ v ∈ 𝓝[s] q, ∀ p ∈ v, ∀ x ∈ k, (f p x, f q x) ∈ u := by
  /-
    α : Type u_4
    β : Type u_5
    E : Type u_6
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : UniformSpace E
    f : α → β → E
    s : Set α
    k : Set β
    q : α
    u : Set (Prod E E)
    hk : IsCompact k
    hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
    hq : Membership.mem s q
    hu : Membership.mem (uniformity E) u
    ⊢ Exists fun v => And (Membership.mem (nhdsWithin q s) v) (∀ (p : α), Membersh …
  -/
  apply hk.induction_on (p := fun t ↦ ∃ v ∈ 𝓝[s] q, ∀ p ∈ v, ∀ x ∈ t, (f p x, f q x) ∈ u)
    /-
      case he
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      ⊢ Exists fun v => And (Membership.mem (nhdsWithin q s) v) (∀ (p : α), Membersh …
    -/
  · exact ⟨univ, univ_mem, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case hmono
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      ⊢ ∀ ⦃s_1 t : Set β⦄, HasSubset.Subset s_1 t → (Exists fun v => And (Membership …
    -/
  · intro t' t ht't ⟨v, v_mem, hv⟩
    /-
      case hmono
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      t' t : Set β
      ht't : HasSubset.Subset t' t
      v : Set α
      v_mem : Membership.mem (nhdsWithin q s) v
      hv : ∀ (p : α), Membership.mem v p → ∀ (x : β), Membership.mem t x → Membershi …
      ⊢ Exists fun v => And (Membership.mem (nhdsWithin q s) v) (∀ (p : α), Membersh …
    -/
    exact ⟨v, v_mem, fun p hp x hx ↦ hv p hp x (ht't hx)⟩
    /-
      🎉 no goals
    -/
    /-
      case hunion
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      ⊢ ∀ ⦃s_1 t : Set β⦄, (Exists fun v => And (Membership.mem (nhdsWithin q s) v)  …
    -/
  · intro t t' ⟨v, v_mem, hv⟩ ⟨v', v'_mem, hv'⟩
    /-
      case hunion
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      t t' : Set β
      v : Set α
      v_mem : Membership.mem (nhdsWithin q s) v
      hv : ∀ (p : α), Membership.mem v p → ∀ (x : β), Membership.mem t x → Membershi …
      v' : Set α
      v'_mem : Membership.mem (nhdsWithin q s) v'
      hv' : ∀ (p : α), Membership.mem v' p → ∀ (x : β), Membership.mem t' x → Member …
      ⊢ Exists fun v => And (Membership.mem (nhdsWithin q s) v) (∀ (p : α), Membersh …
    -/
    refine ⟨v ∩ v', inter_mem v_mem v'_mem, fun p hp x hx ↦ ?_⟩
    /-
      case hunion
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      t t' : Set β
      v : Set α
      v_mem : Membership.mem (nhdsWithin q s) v
      hv : ∀ (p : α), Membership.mem v p → ∀ (x : β), Membership.mem t x → Membershi …
      v' : Set α
      v'_mem : Membership.mem (nhdsWithin q s) v'
      hv' : ∀ (p : α), Membership.mem v' p → ∀ (x : β), Membership.mem t' x → Member …
      p : α
      hp : Membership.mem (Inter.inter v v') p
      x : β
      hx : Membership.mem (Union.union t t') x
      ⊢ Membership.mem u { fst := f p x, snd := f q x }
    -/
    rcases hx with h'x|h'x
      /-
        case hunion.inl
        α : Type u_4
        β : Type u_5
        E : Type u_6
        inst✝² : TopologicalSpace α
        inst✝¹ : TopologicalSpace β
        inst✝ : UniformSpace E
        f : α → β → E
        s : Set α
        k : Set β
        q : α
        u : Set (Prod E E)
        hk : IsCompact k
        hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
        hq : Membership.mem s q
        hu : Membership.mem (uniformity E) u
        t t' : Set β
        v : Set α
        v_mem : Membership.mem (nhdsWithin q s) v
        hv : ∀ (p : α), Membership.mem v p → ∀ (x : β), Membership.mem t x → Membershi …
        v' : Set α
        v'_mem : Membership.mem (nhdsWithin q s) v'
        hv' : ∀ (p : α), Membership.mem v' p → ∀ (x : β), Membership.mem t' x → Member …
        p : α
        hp : Membership.mem (Inter.inter v v') p
        x : β
        h'x : Membership.mem t x
        ⊢ Membership.mem u { fst := f p x, snd := f q x }
      -/
    · exact hv p hp.1 x h'x
      /-
        🎉 no goals
      -/
      /-
        case hunion.inr
        α : Type u_4
        β : Type u_5
        E : Type u_6
        inst✝² : TopologicalSpace α
        inst✝¹ : TopologicalSpace β
        inst✝ : UniformSpace E
        f : α → β → E
        s : Set α
        k : Set β
        q : α
        u : Set (Prod E E)
        hk : IsCompact k
        hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
        hq : Membership.mem s q
        hu : Membership.mem (uniformity E) u
        t t' : Set β
        v : Set α
        v_mem : Membership.mem (nhdsWithin q s) v
        hv : ∀ (p : α), Membership.mem v p → ∀ (x : β), Membership.mem t x → Membershi …
        v' : Set α
        v'_mem : Membership.mem (nhdsWithin q s) v'
        hv' : ∀ (p : α), Membership.mem v' p → ∀ (x : β), Membership.mem t' x → Member …
        p : α
        hp : Membership.mem (Inter.inter v v') p
        x : β
        h'x : Membership.mem t' x
        ⊢ Membership.mem u { fst := f p x, snd := f q x }
      -/
    · exact hv' p hp.2 x h'x
      /-
        🎉 no goals
      -/
    /-
      case hnhds
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      ⊢ ∀ (x : β), Membership.mem k x → Exists fun t => And (Membership.mem (nhdsWit …
    -/
  · rcases comp_symm_of_uniformity hu with ⟨u', u'_mem, u'_symm, hu'⟩
    /-
      case hnhds.intro.intro.intro
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      u' : Set (Prod E E)
      u'_mem : Membership.mem (uniformity E) u'
      u'_symm : ∀ {a b : E}, Membership.mem u' { fst := a, snd := b } → Membership.m …
      hu' : HasSubset.Subset (compRel u' u') u
      ⊢ ∀ (x : β), Membership.mem k x → Exists fun t => And (Membership.mem (nhdsWit …
    -/
    intro x hx
    obtain ⟨v, hv, w, hw, hvw⟩ :
      ∃ v ∈ 𝓝[s] q, ∃ w ∈ 𝓝[k] x, v ×ˢ w ⊆ f.uncurry ⁻¹' {z | (f q x, z) ∈ u'} :=
        mem_nhdsWithin_prod_iff.1 (hf (q, x) ⟨hq, hx⟩ (mem_nhds_left (f q x) u'_mem))
    /-
      case hnhds.intro.intro.intro.intro.intro.intro.intro
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      u' : Set (Prod E E)
      u'_mem : Membership.mem (uniformity E) u'
      u'_symm : ∀ {a b : E}, Membership.mem u' { fst := a, snd := b } → Membership.m …
      hu' : HasSubset.Subset (compRel u' u') u
      x : β
      hx : Membership.mem k x
      v : Set α
      hv : Membership.mem (nhdsWithin q s) v
      w : Set β
      hw : Membership.mem (nhdsWithin x k) w
      hvw : HasSubset.Subset (SProd.sprod v w) (Set.preimage (Function.uncurry f) (s …
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x k) t) (Exists fun v => And …
    -/
    refine ⟨w, hw, v, hv, fun p hp y hy ↦ ?_⟩
    /-
      case hnhds.intro.intro.intro.intro.intro.intro.intro
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      u' : Set (Prod E E)
      u'_mem : Membership.mem (uniformity E) u'
      u'_symm : ∀ {a b : E}, Membership.mem u' { fst := a, snd := b } → Membership.m …
      hu' : HasSubset.Subset (compRel u' u') u
      x : β
      hx : Membership.mem k x
      v : Set α
      hv : Membership.mem (nhdsWithin q s) v
      w : Set β
      hw : Membership.mem (nhdsWithin x k) w
      hvw : HasSubset.Subset (SProd.sprod v w) (Set.preimage (Function.uncurry f) (s …
      p : α
      hp : Membership.mem v p
      y : β
      hy : Membership.mem w y
      ⊢ Membership.mem u { fst := f p y, snd := f q y }
    -/
    have A : (f q x, f p y) ∈ u' := hvw (⟨hp, hy⟩ : (p, y) ∈ v ×ˢ w)
    /-
      case hnhds.intro.intro.intro.intro.intro.intro.intro
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      u' : Set (Prod E E)
      u'_mem : Membership.mem (uniformity E) u'
      u'_symm : ∀ {a b : E}, Membership.mem u' { fst := a, snd := b } → Membership.m …
      hu' : HasSubset.Subset (compRel u' u') u
      x : β
      hx : Membership.mem k x
      v : Set α
      hv : Membership.mem (nhdsWithin q s) v
      w : Set β
      hw : Membership.mem (nhdsWithin x k) w
      hvw : HasSubset.Subset (SProd.sprod v w) (Set.preimage (Function.uncurry f) (s …
      p : α
      hp : Membership.mem v p
      y : β
      hy : Membership.mem w y
      A : Membership.mem u' { fst := f q x, snd := f p y }
      ⊢ Membership.mem u { fst := f p y, snd := f q y }
    -/
    have B : (f q x, f q y) ∈ u' := hvw (⟨mem_of_mem_nhdsWithin hq hv, hy⟩ : (q, y) ∈ v ×ˢ w)
    /-
      case hnhds.intro.intro.intro.intro.intro.intro.intro
      α : Type u_4
      β : Type u_5
      E : Type u_6
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : UniformSpace E
      f : α → β → E
      s : Set α
      k : Set β
      q : α
      u : Set (Prod E E)
      hk : IsCompact k
      hf : ContinuousOn (Function.uncurry f) (SProd.sprod s k)
      hq : Membership.mem s q
      hu : Membership.mem (uniformity E) u
      u' : Set (Prod E E)
      u'_mem : Membership.mem (uniformity E) u'
      u'_symm : ∀ {a b : E}, Membership.mem u' { fst := a, snd := b } → Membership.m …
      hu' : HasSubset.Subset (compRel u' u') u
      x : β
      hx : Membership.mem k x
      v : Set α
      hv : Membership.mem (nhdsWithin q s) v
      w : Set β
      hw : Membership.mem (nhdsWithin x k) w
      hvw : HasSubset.Subset (SProd.sprod v w) (Set.preimage (Function.uncurry f) (s …
      p : α
      hp : Membership.mem v p
      y : β
      hy : Membership.mem w y
      A : Membership.mem u' { fst := f q x, snd := f p y }
      B : Membership.mem u' { fst := f q x, snd := f q y }
      ⊢ Membership.mem u { fst := f p y, snd := f q y }
    -/
    exact hu' (prod_mk_mem_compRel (u'_symm A) B)
    /-
      🎉 no goals
    -/


/-- An equicontinuous family of functions defined on a compact uniform space is automatically
uniformly equicontinuous. -/
theorem CompactSpace.uniformEquicontinuous_of_equicontinuous {ι : Type*} {F : ι → β → α}
    [CompactSpace β] (h : Equicontinuous F) : UniformEquicontinuous F := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    ι : Type u_4
    F : ι → β → α
    inst✝ : CompactSpace β
    h : Equicontinuous F
    ⊢ UniformEquicontinuous F
  -/
  rw [equicontinuous_iff_continuous] at h
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    ι : Type u_4
    F : ι → β → α
    inst✝ : CompactSpace β
    h : Continuous (Function.comp (⇑UniformFun.ofFun) (Function.swap F))
    ⊢ UniformEquicontinuous F
  -/
  rw [uniformEquicontinuous_iff_uniformContinuous]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    ι : Type u_4
    F : ι → β → α
    inst✝ : CompactSpace β
    h : Continuous (Function.comp (⇑UniformFun.ofFun) (Function.swap F))
    ⊢ UniformContinuous (Function.comp (⇑UniformFun.ofFun) (Function.swap F))
  -/
  exact CompactSpace.uniformContinuous_of_continuous h
  /-
    🎉 no goals
  -/


