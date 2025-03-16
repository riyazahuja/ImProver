/-- Type class for noetherian spaces. It is defined to be spaces whose open sets satisfies ACC. -/
abbrev NoetherianSpace : Prop := WellFoundedGT (Opens α)


theorem noetherianSpace_iff_opens : NoetherianSpace α ↔ ∀ s : Opens α, IsCompact (s : Set α) := by
  rw [NoetherianSpace, CompleteLattice.wellFoundedGT_iff_isSupFiniteCompact,
    CompleteLattice.isSupFiniteCompact_iff_all_elements_compact]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ Iff (∀ (k : TopologicalSpace.Opens α), CompleteLattice.IsCompactElement k) ( …
  -/
  exact forall_congr' Opens.isCompactElement_iff
  /-
    🎉 no goals
  -/


instance (priority := 100) NoetherianSpace.compactSpace [h : NoetherianSpace α] : CompactSpace α :=
  ⟨(noetherianSpace_iff_opens α).mp h ⊤⟩


/-- In a Noetherian space, all sets are compact. -/
protected theorem NoetherianSpace.isCompact [NoetherianSpace α] (s : Set α) : IsCompact s := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    s : Set α
    ⊢ IsCompact s
  -/
  refine isCompact_iff_finite_subcover.2 fun U hUo hs => ?_
  rcases ((noetherianSpace_iff_opens α).mp ‹_› ⟨⋃ i, U i, isOpen_iUnion hUo⟩).elim_finite_subcover U
    hUo Set.Subset.rfl with ⟨t, ht⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    s : Set α
    ι✝ : Type u_1
    U : ι✝ → Set α
    hUo : ∀ (i : ι✝), IsOpen (U i)
    hs : HasSubset.Subset s (Set.iUnion fun i => U i)
    t : Finset ι✝
    ht : HasSubset.Subset (↑{ carrier := Set.iUnion fun i => U i, is_open' := ⋯ }) …
    ⊢ Exists fun t => HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h =>  …
  -/
  exact ⟨t, hs.trans ht⟩
  /-
    🎉 no goals
  -/


protected theorem _root_.Topology.IsInducing.noetherianSpace [NoetherianSpace α] {i : β → α}
    (hi : IsInducing i) : NoetherianSpace β :=
  (noetherianSpace_iff_opens _).2 fun _ => hi.isCompact_iff.2 (NoetherianSpace.isCompact _)


@[deprecated (since := "2024-10-28")]
alias _root_.Inducing.noetherianSpace := IsInducing.noetherianSpace


/-- [Stacks: Lemma 0052 (1)](https://stacks.math.columbia.edu/tag/0052)-/
instance NoetherianSpace.set [NoetherianSpace α] (s : Set α) : NoetherianSpace s :=
  IsInducing.subtypeVal.noetherianSpace


open List in
theorem noetherianSpace_TFAE :
    TFAE [NoetherianSpace α,
      WellFoundedLT (Closeds α),
      ∀ s : Set α, IsCompact s,
      ∀ s : Opens α, IsCompact (s : Set α)] := by
  tfae_have 1 ↔ 2 := by
    simp_rw [isWellFounded_iff]
    exact Opens.compl_bijective.2.wellFounded_iff (@OrderIso.compl (Set α)).lt_iff_lt.symm
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    tfae_1_iff_2 : Iff (TopologicalSpace.NoetherianSpace α) (WellFoundedLT (Topolo …
    ⊢ (List.cons (TopologicalSpace.NoetherianSpace α) (List.cons (WellFoundedLT (T …
  -/
  tfae_have 1 ↔ 4 := noetherianSpace_iff_opens α
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    tfae_1_iff_2 : Iff (TopologicalSpace.NoetherianSpace α) (WellFoundedLT (Topolo …
    tfae_1_iff_4 : Iff (TopologicalSpace.NoetherianSpace α) (∀ (s : TopologicalSpa …
    ⊢ (List.cons (TopologicalSpace.NoetherianSpace α) (List.cons (WellFoundedLT (T …
  -/
  tfae_have 1 → 3 := @NoetherianSpace.isCompact α _
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    tfae_1_iff_2 : Iff (TopologicalSpace.NoetherianSpace α) (WellFoundedLT (Topolo …
    tfae_1_iff_4 : Iff (TopologicalSpace.NoetherianSpace α) (∀ (s : TopologicalSpa …
    tfae_1_to_3 : TopologicalSpace.NoetherianSpace α → ∀ (s : Set α), IsCompact s
    ⊢ (List.cons (TopologicalSpace.NoetherianSpace α) (List.cons (WellFoundedLT (T …
  -/
  tfae_have 3 → 4 := fun h s => h s
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    tfae_1_iff_2 : Iff (TopologicalSpace.NoetherianSpace α) (WellFoundedLT (Topolo …
    tfae_1_iff_4 : Iff (TopologicalSpace.NoetherianSpace α) (∀ (s : TopologicalSpa …
    tfae_1_to_3 : TopologicalSpace.NoetherianSpace α → ∀ (s : Set α), IsCompact s
    tfae_3_to_4 : (∀ (s : Set α), IsCompact s) → ∀ (s : TopologicalSpace.Opens α), …
    ⊢ (List.cons (TopologicalSpace.NoetherianSpace α) (List.cons (WellFoundedLT (T …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem noetherianSpace_iff_isCompact : NoetherianSpace α ↔ ∀ s : Set α, IsCompact s :=
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ Eq ((List.cons (TopologicalSpace.NoetherianSpace α) (List.cons (WellFoundedL …
  -/
  /-
    🎉 no goals
  -/
  (noetherianSpace_TFAE α).out 0 2
  /-
    🎉 no goals
  -/


instance [NoetherianSpace α] : WellFoundedLT (Closeds α) :=
          /-
            α : Type u_1
            β : Type u_2
            inst✝² : TopologicalSpace α
            inst✝¹ : TopologicalSpace β
            inst✝ : TopologicalSpace.NoetherianSpace α
            ⊢ Eq ((List.cons (TopologicalSpace.NoetherianSpace α) (List.cons (WellFoundedL …
          -/
          /-
            🎉 no goals
          -/
  Iff.mp ((noetherianSpace_TFAE α).out 0 1) ‹_›
          /-
            🎉 no goals
          -/


@[deprecated "No deprecation message was provided." (since := "2024-10-07")]
theorem NoetherianSpace.wellFounded_closeds [NoetherianSpace α] :
    WellFounded fun s t : Closeds α => s < t :=
  wellFounded_lt


instance {α} : NoetherianSpace (CofiniteTopology α) := by
  simp only [noetherianSpace_iff_isCompact, isCompact_iff_ultrafilter_le_nhds,
    CofiniteTopology.nhds_eq, Ultrafilter.le_sup_iff, Filter.le_principal_iff]
  /-
    α✝ : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α✝
    inst✝ : TopologicalSpace β
    α : Type u_3
    ⊢ ∀ (s : Set (CofiniteTopology α)) (f : Ultrafilter (CofiniteTopology α)), Mem …
  -/
  intro s f hs
  /-
    α✝ : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α✝
    inst✝ : TopologicalSpace β
    α : Type u_3
    s : Set (CofiniteTopology α)
    f : Ultrafilter (CofiniteTopology α)
    hs : Membership.mem (↑f) s
    ⊢ Exists fun x => And (Membership.mem s x) (Or (LE.le (↑f) (Pure.pure x)) (LE. …
  -/
  rcases f.le_cofinite_or_eq_pure with (hf | ⟨a, rfl⟩)
    /-
      case inl
      α✝ : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α✝
      inst✝ : TopologicalSpace β
      α : Type u_3
      s : Set (CofiniteTopology α)
      f : Ultrafilter (CofiniteTopology α)
      hs : Membership.mem (↑f) s
      hf : LE.le (↑f) Filter.cofinite
      ⊢ Exists fun x => And (Membership.mem s x) (Or (LE.le (↑f) (Pure.pure x)) (LE. …
    -/
  · rcases Filter.nonempty_of_mem hs with ⟨a, ha⟩
    /-
      case inl.intro
      α✝ : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α✝
      inst✝ : TopologicalSpace β
      α : Type u_3
      s : Set (CofiniteTopology α)
      f : Ultrafilter (CofiniteTopology α)
      hs : Membership.mem (↑f) s
      hf : LE.le (↑f) Filter.cofinite
      a : CofiniteTopology α
      ha : Membership.mem s a
      ⊢ Exists fun x => And (Membership.mem s x) (Or (LE.le (↑f) (Pure.pure x)) (LE. …
    -/
    exact ⟨a, ha, Or.inr hf⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α✝ : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α✝
      inst✝ : TopologicalSpace β
      α : Type u_3
      s : Set (CofiniteTopology α)
      a : CofiniteTopology α
      hs : Membership.mem (↑(Pure.pure a)) s
      ⊢ Exists fun x => And (Membership.mem s x) (Or (LE.le (↑(Pure.pure a)) (Pure.p …
    -/
  · exact ⟨a, hs, Or.inl le_rfl⟩
    /-
      🎉 no goals
    -/


theorem noetherianSpace_of_surjective [NoetherianSpace α] (f : α → β) (hf : Continuous f)
    (hf' : Function.Surjective f) : NoetherianSpace β :=
  noetherianSpace_iff_isCompact.2 <| (Set.image_surjective.mpr hf').forall.2 fun s =>
    (NoetherianSpace.isCompact s).image hf


theorem noetherianSpace_iff_of_homeomorph (f : α ≃ₜ β) : NoetherianSpace α ↔ NoetherianSpace β :=
  ⟨fun _ => noetherianSpace_of_surjective f f.continuous f.surjective,
    fun _ => noetherianSpace_of_surjective f.symm f.symm.continuous f.symm.surjective⟩


theorem NoetherianSpace.range [NoetherianSpace α] (f : α → β) (hf : Continuous f) :
    NoetherianSpace (Set.range f) :=
  noetherianSpace_of_surjective (Set.rangeFactorization f) (hf.subtype_mk _)
    Set.surjective_onto_range


theorem noetherianSpace_set_iff (s : Set α) :
    NoetherianSpace s ↔ ∀ t, t ⊆ s → IsCompact t := by
  simp only [noetherianSpace_iff_isCompact, IsEmbedding.subtypeVal.isCompact_iff,
    Subtype.forall_set_subtype]


@[simp]
theorem noetherian_univ_iff : NoetherianSpace (Set.univ : Set α) ↔ NoetherianSpace α :=
  noetherianSpace_iff_of_homeomorph (Homeomorph.Set.univ α)


theorem NoetherianSpace.iUnion {ι : Type*} (f : ι → Set α) [Finite ι]
    [hf : ∀ i, NoetherianSpace (f i)] : NoetherianSpace (⋃ i, f i) := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    ι : Type u_3
    f : ι → Set α
    inst✝ : Finite ι
    hf : ∀ (i : ι), TopologicalSpace.NoetherianSpace ↑(f i)
    ⊢ TopologicalSpace.NoetherianSpace ↑(Set.iUnion fun i => f i)
  -/
  simp_rw [noetherianSpace_set_iff] at hf ⊢
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    ι : Type u_3
    f : ι → Set α
    inst✝ : Finite ι
    hf : ∀ (i : ι) (t : Set α), HasSubset.Subset t (f i) → IsCompact t
    ⊢ ∀ (t : Set α), HasSubset.Subset t (Set.iUnion fun i => f i) → IsCompact t
  -/
  intro t ht
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    ι : Type u_3
    f : ι → Set α
    inst✝ : Finite ι
    hf : ∀ (i : ι) (t : Set α), HasSubset.Subset t (f i) → IsCompact t
    t : Set α
    ht : HasSubset.Subset t (Set.iUnion fun i => f i)
    ⊢ IsCompact t
  -/
  rw [← Set.inter_eq_left.mpr ht, Set.inter_iUnion]
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    ι : Type u_3
    f : ι → Set α
    inst✝ : Finite ι
    hf : ∀ (i : ι) (t : Set α), HasSubset.Subset t (f i) → IsCompact t
    t : Set α
    ht : HasSubset.Subset t (Set.iUnion fun i => f i)
    ⊢ IsCompact (Set.iUnion fun i => Inter.inter t (f i))
  -/
  exact isCompact_iUnion fun i => hf i _ Set.inter_subset_right
  /-
    🎉 no goals
  -/

-- This is not an instance since it makes a loop with `t2_space_discrete`.

theorem NoetherianSpace.discrete [NoetherianSpace α] [T2Space α] : DiscreteTopology α :=
  ⟨eq_bot_iff.mpr fun _ _ => isClosed_compl_iff.mp (NoetherianSpace.isCompact _).isClosed⟩


/-- Spaces that are both Noetherian and Hausdorff are finite. -/
theorem NoetherianSpace.finite [NoetherianSpace α] [T2Space α] : Finite α :=
  Finite.of_finite_univ (NoetherianSpace.isCompact Set.univ).finite_of_discrete


instance (priority := 100) Finite.to_noetherianSpace [Finite α] : NoetherianSpace α :=
  ⟨Finite.wellFounded_of_trans_of_irrefl _⟩


/-- In a Noetherian space, every closed set is a finite union of irreducible closed sets. -/
theorem NoetherianSpace.exists_finite_set_closeds_irreducible [NoetherianSpace α] (s : Closeds α) :
    ∃ S : Set (Closeds α), S.Finite ∧ (∀ t ∈ S, IsIrreducible (t : Set α)) ∧ s = sSup S := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    s : TopologicalSpace.Closeds α
    ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
  -/
  apply wellFounded_lt.induction s; clear s
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    ⊢ ∀ (x : TopologicalSpace.Closeds α), (∀ (y : TopologicalSpace.Closeds α), LT. …
  -/
  intro s H
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    s : TopologicalSpace.Closeds α
    H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
    ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
  -/
  rcases eq_or_ne s ⊥ with rfl | h₀
    /-
      case inl
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace.NoetherianSpace α
      H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y Bot.bot → Exists fun S => And  …
      ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
    -/
  · use ∅; simp
           /-
             🎉 no goals
           -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace.NoetherianSpace α
      s : TopologicalSpace.Closeds α
      H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
      h₀ : Ne s Bot.bot
      ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
    -/
  · by_cases h₁ : IsPreirreducible (s : Set α)
      /-
        case pos
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        h₁ : IsPreirreducible ↑s
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
    · replace h₁ : IsIrreducible (s : Set α) := ⟨Closeds.coe_nonempty.2 h₀, h₁⟩
      /-
        case pos
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        h₁ : IsIrreducible ↑s
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
      use {s}; simp [h₁]
               /-
                 🎉 no goals
               -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        h₁ : Not (IsPreirreducible ↑s)
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
    · simp only [isPreirreducible_iff_isClosed_union_isClosed, not_forall, not_or] at h₁
      /-
        case neg
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        h₁ : Exists fun x => Exists fun x_1 => Exists fun h => Exists fun h => Exists  …
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
      obtain ⟨z₁, z₂, hz₁, hz₂, h, hz₁', hz₂'⟩ := h₁
      /-
        case neg.intro.intro.intro.intro.intro.intro
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        z₁ z₂ : Set α
        hz₁ : IsClosed z₁
        hz₂ : IsClosed z₂
        h : HasSubset.Subset (↑s) (Union.union z₁ z₂)
        hz₁' : Not (HasSubset.Subset (↑s) z₁)
        hz₂' : Not (HasSubset.Subset (↑s) z₂)
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
      lift z₁ to Closeds α using hz₁
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        z₂ : Set α
        hz₂ : IsClosed z₂
        hz₂' : Not (HasSubset.Subset (↑s) z₂)
        z₁ : TopologicalSpace.Closeds α
        h : HasSubset.Subset (↑s) (Union.union (↑z₁) z₂)
        hz₁' : Not (HasSubset.Subset ↑s ↑z₁)
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
      lift z₂ to Closeds α using hz₂
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        z₁ : TopologicalSpace.Closeds α
        hz₁' : Not (HasSubset.Subset ↑s ↑z₁)
        z₂ : TopologicalSpace.Closeds α
        hz₂' : Not (HasSubset.Subset ↑s ↑z₂)
        h : HasSubset.Subset (↑s) (Union.union ↑z₁ ↑z₂)
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
      rcases H (s ⊓ z₁) (inf_lt_left.2 hz₁') with ⟨S₁, hSf₁, hS₁, h₁⟩
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        z₁ : TopologicalSpace.Closeds α
        hz₁' : Not (HasSubset.Subset ↑s ↑z₁)
        z₂ : TopologicalSpace.Closeds α
        hz₂' : Not (HasSubset.Subset ↑s ↑z₂)
        h : HasSubset.Subset (↑s) (Union.union ↑z₁ ↑z₂)
        S₁ : Set (TopologicalSpace.Closeds α)
        hSf₁ : S₁.Finite
        hS₁ : ∀ (t : TopologicalSpace.Closeds α), Membership.mem S₁ t → IsIrreducible ↑t
        h₁ : Eq (Min.min s z₁) (SupSet.sSup S₁)
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
      rcases H (s ⊓ z₂) (inf_lt_left.2 hz₂') with ⟨S₂, hSf₂, hS₂, h₂⟩
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.int …
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        z₁ : TopologicalSpace.Closeds α
        hz₁' : Not (HasSubset.Subset ↑s ↑z₁)
        z₂ : TopologicalSpace.Closeds α
        hz₂' : Not (HasSubset.Subset ↑s ↑z₂)
        h : HasSubset.Subset (↑s) (Union.union ↑z₁ ↑z₂)
        S₁ : Set (TopologicalSpace.Closeds α)
        hSf₁ : S₁.Finite
        hS₁ : ∀ (t : TopologicalSpace.Closeds α), Membership.mem S₁ t → IsIrreducible ↑t
        h₁ : Eq (Min.min s z₁) (SupSet.sSup S₁)
        S₂ : Set (TopologicalSpace.Closeds α)
        hSf₂ : S₂.Finite
        hS₂ : ∀ (t : TopologicalSpace.Closeds α), Membership.mem S₂ t → IsIrreducible ↑t
        h₂ : Eq (Min.min s z₂) (SupSet.sSup S₂)
        ⊢ Exists fun S => And S.Finite (And (∀ (t : TopologicalSpace.Closeds α), Membe …
      -/
      refine ⟨S₁ ∪ S₂, hSf₁.union hSf₂, Set.union_subset_iff.2 ⟨hS₁, hS₂⟩, ?_⟩
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.int …
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace.NoetherianSpace α
        s : TopologicalSpace.Closeds α
        H : ∀ (y : TopologicalSpace.Closeds α), LT.lt y s → Exists fun S => And S.Fini …
        h₀ : Ne s Bot.bot
        z₁ : TopologicalSpace.Closeds α
        hz₁' : Not (HasSubset.Subset ↑s ↑z₁)
        z₂ : TopologicalSpace.Closeds α
        hz₂' : Not (HasSubset.Subset ↑s ↑z₂)
        h : HasSubset.Subset (↑s) (Union.union ↑z₁ ↑z₂)
        S₁ : Set (TopologicalSpace.Closeds α)
        hSf₁ : S₁.Finite
        hS₁ : ∀ (t : TopologicalSpace.Closeds α), Membership.mem S₁ t → IsIrreducible ↑t
        h₁ : Eq (Min.min s z₁) (SupSet.sSup S₁)
        S₂ : Set (TopologicalSpace.Closeds α)
        hSf₂ : S₂.Finite
        hS₂ : ∀ (t : TopologicalSpace.Closeds α), Membership.mem S₂ t → IsIrreducible ↑t
        h₂ : Eq (Min.min s z₂) (SupSet.sSup S₂)
        ⊢ Eq s (SupSet.sSup (Union.union S₁ S₂))
      -/
      rwa [sSup_union, ← h₁, ← h₂, ← inf_sup_left, left_eq_inf]
      /-
        🎉 no goals
      -/


/-- In a Noetherian space, every closed set is a finite union of irreducible closed sets. -/
theorem NoetherianSpace.exists_finite_set_isClosed_irreducible [NoetherianSpace α]
    {s : Set α} (hs : IsClosed s) : ∃ S : Set (Set α), S.Finite ∧
      (∀ t ∈ S, IsClosed t) ∧ (∀ t ∈ S, IsIrreducible t) ∧ s = ⋃₀ S := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    s : Set α
    hs : IsClosed s
    ⊢ Exists fun S => And S.Finite (And (∀ (t : Set α), Membership.mem S t → IsClo …
  -/
  lift s to Closeds α using hs
  /-
    case intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    s : TopologicalSpace.Closeds α
    ⊢ Exists fun S => And S.Finite (And (∀ (t : Set α), Membership.mem S t → IsClo …
  -/
  rcases NoetherianSpace.exists_finite_set_closeds_irreducible s with ⟨S, hSf, hS, rfl⟩
  refine ⟨(↑) '' S, hSf.image _, Set.forall_mem_image.2 fun S _ ↦ S.2, Set.forall_mem_image.2 hS,
    ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    S : Set (TopologicalSpace.Closeds α)
    hSf : S.Finite
    hS : ∀ (t : TopologicalSpace.Closeds α), Membership.mem S t → IsIrreducible ↑t
    ⊢ Eq (↑(SupSet.sSup S)) (Set.image SetLike.coe S).sUnion
  -/
  lift S to Finset (Closeds α) using hSf
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    S : Finset (TopologicalSpace.Closeds α)
    hS : ∀ (t : TopologicalSpace.Closeds α), Membership.mem (↑S) t → IsIrreducible …
    ⊢ Eq (↑(SupSet.sSup ↑S)) (Set.image SetLike.coe ↑S).sUnion
  -/
  simp [← Finset.sup_id_eq_sSup, Closeds.coe_finset_sup]
  /-
    🎉 no goals
  -/


/-- In a Noetherian space, every closed set is a finite union of irreducible closed sets. -/
theorem NoetherianSpace.exists_finset_irreducible [NoetherianSpace α] (s : Closeds α) :
    ∃ S : Finset (Closeds α), (∀ k : S, IsIrreducible (k : Set α)) ∧ s = S.sup id := by
  simpa [Set.exists_finite_iff_finset, Finset.sup_id_eq_sSup]
    using NoetherianSpace.exists_finite_set_closeds_irreducible s


/-- [Stacks: Lemma 0052 (2)](https://stacks.math.columbia.edu/tag/0052) -/
theorem NoetherianSpace.finite_irreducibleComponents [NoetherianSpace α] :
    (irreducibleComponents α).Finite := by
  obtain ⟨S : Set (Set α), hSf, hSc, hSi, hSU⟩ :=
    NoetherianSpace.exists_finite_set_isClosed_irreducible isClosed_univ (α := α)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    S : Set (Set α)
    hSf : S.Finite
    hSc : ∀ (t : Set α), Membership.mem S t → IsClosed t
    hSi : ∀ (t : Set α), Membership.mem S t → IsIrreducible t
    hSU : Eq Set.univ S.sUnion
    ⊢ (irreducibleComponents α).Finite
  -/
  refine hSf.subset fun s hs => ?_
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    S : Set (Set α)
    hSf : S.Finite
    hSc : ∀ (t : Set α), Membership.mem S t → IsClosed t
    hSi : ∀ (t : Set α), Membership.mem S t → IsIrreducible t
    hSU : Eq Set.univ S.sUnion
    s : Set α
    hs : Membership.mem (irreducibleComponents α) s
    ⊢ Membership.mem S s
  -/
  lift S to Finset (Set α) using hSf
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    s : Set α
    hs : Membership.mem (irreducibleComponents α) s
    S : Finset (Set α)
    hSc : ∀ (t : Set α), Membership.mem (↑S) t → IsClosed t
    hSi : ∀ (t : Set α), Membership.mem (↑S) t → IsIrreducible t
    hSU : Eq Set.univ (↑S).sUnion
    ⊢ Membership.mem (↑S) s
  -/
  rcases isIrreducible_iff_sUnion_isClosed.1 hs.1 S hSc (hSU ▸ Set.subset_univ _) with ⟨t, htS, ht⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.NoetherianSpace α
    s : Set α
    hs : Membership.mem (irreducibleComponents α) s
    S : Finset (Set α)
    hSc : ∀ (t : Set α), Membership.mem (↑S) t → IsClosed t
    hSi : ∀ (t : Set α), Membership.mem (↑S) t → IsIrreducible t
    hSU : Eq Set.univ (↑S).sUnion
    t : Set α
    htS : Membership.mem S t
    ht : HasSubset.Subset s t
    ⊢ Membership.mem (↑S) s
  -/
  rwa [ht.antisymm (hs.2 (hSi _ htS) ht)]
  /-
    🎉 no goals
  -/


/-- [Stacks: Lemma 0052 (3)](https://stacks.math.columbia.edu/tag/0052) -/
theorem NoetherianSpace.exists_open_ne_empty_le_irreducibleComponent [NoetherianSpace α]
    (Z : Set α) (H : Z ∈ irreducibleComponents α) :
    ∃ o : Set α, IsOpen o ∧ o ≠ ∅ ∧ o ≤ Z := by
  classical

  let ι : Set (Set α) := irreducibleComponents α \ {Z}
  have hι : ι.Finite := NoetherianSpace.finite_irreducibleComponents.subset Set.diff_subset
  have hι' : Finite ι := by rwa [Set.finite_coe_iff]

  let U := Z \ ⋃ (x : ι), x
  have hU0 : U ≠ ∅ := fun r ↦ by
    obtain ⟨Z', hZ'⟩ := isIrreducible_iff_sUnion_isClosed.mp H.1 hι.toFinset
      (fun z hz ↦ by
        simp only [Set.Finite.mem_toFinset, Set.mem_diff, Set.mem_singleton_iff] at hz
        exact isClosed_of_mem_irreducibleComponents _ hz.1)
      (by
        rw [Set.Finite.coe_toFinset, Set.sUnion_eq_iUnion]
        rw [Set.diff_eq_empty] at r
        exact r)
    simp only [Set.Finite.mem_toFinset, Set.mem_diff, Set.mem_singleton_iff] at hZ'
    exact hZ'.1.2 <| le_antisymm (H.2 hZ'.1.1.1 hZ'.2) hZ'.2

  have hU1 : U = (⋃ (x : ι), x.1) ᶜ := by
    rw [Set.compl_eq_univ_diff]
    refine le_antisymm (Set.diff_subset_diff le_top <| subset_refl _) ?_
    rw [← Set.compl_eq_univ_diff]
    refine Set.compl_subset_iff_union.mpr (le_antisymm le_top ?_)
    rw [Set.union_comm, ← Set.sUnion_eq_iUnion, ← Set.sUnion_insert]
    rintro a -
    by_cases h : a ∈ U
    · exact ⟨U, Set.mem_insert _ _, h⟩
    · rw [Set.mem_diff, Decidable.not_and_iff_or_not_not, not_not, Set.mem_iUnion] at h
      rcases h with (h|⟨i, hi⟩)
      · refine ⟨irreducibleComponent a, Or.inr ?_, mem_irreducibleComponent⟩
        simp only [ι, Set.mem_diff, Set.mem_singleton_iff]
        refine ⟨irreducibleComponent_mem_irreducibleComponents _, ?_⟩
        rintro rfl
        exact h mem_irreducibleComponent
      · exact ⟨i, Or.inr i.2, hi⟩

  refine ⟨U, hU1 ▸ isOpen_compl_iff.mpr ?_, hU0, sdiff_le⟩
  exact isClosed_iUnion_of_finite fun i ↦ isClosed_of_mem_irreducibleComponents i.1 i.2.1


