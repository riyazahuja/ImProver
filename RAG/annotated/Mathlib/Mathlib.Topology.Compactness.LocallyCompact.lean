instance [WeaklyLocallyCompactSpace X] [WeaklyLocallyCompactSpace Y] :
    WeaklyLocallyCompactSpace (X × Y) where
  exists_compact_mem_nhds x :=
    let ⟨s₁, hc₁, h₁⟩ := exists_compact_mem_nhds x.1
    let ⟨s₂, hc₂, h₂⟩ := exists_compact_mem_nhds x.2
    ⟨s₁ ×ˢ s₂, hc₁.prod hc₂, prod_mem_nhds h₁ h₂⟩


instance {ι : Type*} [Finite ι] {X : ι → Type*} [(i : ι) → TopologicalSpace (X i)]
    [(i : ι) → WeaklyLocallyCompactSpace (X i)] :
    WeaklyLocallyCompactSpace ((i : ι) → X i) where
  exists_compact_mem_nhds f := by
    /-
      X✝ : Type u_1
      Y : Type u_2
      ι✝ : Type u_3
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s t : Set X✝
      ι : Type u_4
      inst✝² : Finite ι
      X : ι → Type u_5
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : ∀ (i : ι), WeaklyLocallyCompactSpace (X i)
      f : (i : ι) → X i
      ⊢ Exists fun s => And (IsCompact s) (Membership.mem (nhds f) s)
    -/
    choose s hsc hs using fun i ↦ exists_compact_mem_nhds (f i)
    /-
      X✝ : Type u_1
      Y : Type u_2
      ι✝ : Type u_3
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s✝ t : Set X✝
      ι : Type u_4
      inst✝² : Finite ι
      X : ι → Type u_5
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : ∀ (i : ι), WeaklyLocallyCompactSpace (X i)
      f : (i : ι) → X i
      s : (i : ι) → Set (X i)
      hsc : ∀ (i : ι), IsCompact (s i)
      hs : ∀ (i : ι), Membership.mem (nhds (f i)) (s i)
      ⊢ Exists fun s => And (IsCompact s) (Membership.mem (nhds f) s)
    -/
    exact ⟨pi univ s, isCompact_univ_pi hsc, set_pi_mem_nhds univ.toFinite fun i _ ↦ hs i⟩
    /-
      🎉 no goals
    -/


instance (priority := 100) [CompactSpace X] : WeaklyLocallyCompactSpace X where
  exists_compact_mem_nhds _ := ⟨univ, isCompact_univ, univ_mem⟩


protected theorem Topology.IsClosedEmbedding.weaklyLocallyCompactSpace [WeaklyLocallyCompactSpace Y]
    {f : X → Y} (hf : IsClosedEmbedding f) : WeaklyLocallyCompactSpace X where
  exists_compact_mem_nhds x :=
    let ⟨K, hK, hKx⟩ := exists_compact_mem_nhds (f x)
    ⟨f ⁻¹' K, hf.isCompact_preimage hK, hf.continuous.continuousAt hKx⟩


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.weaklyLocallyCompactSpace := IsClosedEmbedding.weaklyLocallyCompactSpace


protected theorem IsClosed.weaklyLocallyCompactSpace [WeaklyLocallyCompactSpace X]
    {s : Set X} (hs : IsClosed s) : WeaklyLocallyCompactSpace s :=
  hs.isClosedEmbedding_subtypeVal.weaklyLocallyCompactSpace


theorem IsOpenQuotientMap.weaklyLocallyCompactSpace [WeaklyLocallyCompactSpace X]
    {f : X → Y} (hf : IsOpenQuotientMap f) : WeaklyLocallyCompactSpace Y where
  exists_compact_mem_nhds := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : WeaklyLocallyCompactSpace X
      f : X → Y
      hf : IsOpenQuotientMap f
      ⊢ ∀ (x : Y), Exists fun s => And (IsCompact s) (Membership.mem (nhds x) s)
    -/
    refine hf.surjective.forall.2 fun x ↦ ?_
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : WeaklyLocallyCompactSpace X
      f : X → Y
      hf : IsOpenQuotientMap f
      x : X
      ⊢ Exists fun s => And (IsCompact s) (Membership.mem (nhds (f x)) s)
    -/
    rcases exists_compact_mem_nhds x with ⟨K, hKc, hKx⟩
    /-
      case intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : WeaklyLocallyCompactSpace X
      f : X → Y
      hf : IsOpenQuotientMap f
      x : X
      K : Set X
      hKc : IsCompact K
      hKx : Membership.mem (nhds x) K
      ⊢ Exists fun s => And (IsCompact s) (Membership.mem (nhds (f x)) s)
    -/
    exact ⟨f '' K, hKc.image hf.continuous, hf.isOpenMap.image_mem_nhds hKx⟩
    /-
      🎉 no goals
    -/


/-- In a weakly locally compact space,
every compact set is contained in the interior of a compact set. -/
theorem exists_compact_superset [WeaklyLocallyCompactSpace X] {K : Set X} (hK : IsCompact K) :
    ∃ K', IsCompact K' ∧ K ⊆ interior K' := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : WeaklyLocallyCompactSpace X
    K : Set X
    hK : IsCompact K
    ⊢ Exists fun K' => And (IsCompact K') (HasSubset.Subset K (interior K'))
  -/
  choose s hc hmem using fun x : X ↦ exists_compact_mem_nhds x
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : WeaklyLocallyCompactSpace X
    K : Set X
    hK : IsCompact K
    s : X → Set X
    hc : ∀ (x : X), IsCompact (s x)
    hmem : ∀ (x : X), Membership.mem (nhds x) (s x)
    ⊢ Exists fun K' => And (IsCompact K') (HasSubset.Subset K (interior K'))
  -/
  rcases hK.elim_nhds_subcover _ fun x _ ↦ interior_mem_nhds.2 (hmem x) with ⟨I, -, hIK⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : WeaklyLocallyCompactSpace X
    K : Set X
    hK : IsCompact K
    s : X → Set X
    hc : ∀ (x : X), IsCompact (s x)
    hmem : ∀ (x : X), Membership.mem (nhds x) (s x)
    I : Finset X
    hIK : HasSubset.Subset K (Set.iUnion fun x => Set.iUnion fun h => interior (s  …
    ⊢ Exists fun K' => And (IsCompact K') (HasSubset.Subset K (interior K'))
  -/
  refine ⟨⋃ x ∈ I, s x, I.isCompact_biUnion fun _ _ ↦ hc _, hIK.trans ?_⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : WeaklyLocallyCompactSpace X
    K : Set X
    hK : IsCompact K
    s : X → Set X
    hc : ∀ (x : X), IsCompact (s x)
    hmem : ∀ (x : X), Membership.mem (nhds x) (s x)
    I : Finset X
    hIK : HasSubset.Subset K (Set.iUnion fun x => Set.iUnion fun h => interior (s  …
    ⊢ HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => interior (s x)) (i …
  -/
  exact iUnion₂_subset fun x hx ↦ interior_mono <| subset_iUnion₂ (s := fun x _ ↦ s x) x hx
  /-
    🎉 no goals
  -/


/-- In a weakly locally compact space,
the filters `𝓝 x` and `cocompact X` are disjoint for all `X`. -/
theorem disjoint_nhds_cocompact [WeaklyLocallyCompactSpace X] (x : X) :
    Disjoint (𝓝 x) (cocompact X) :=
  let ⟨_, hc, hx⟩ := exists_compact_mem_nhds x
  disjoint_of_disjoint_of_mem disjoint_compl_right hx hc.compl_mem_cocompact


theorem compact_basis_nhds [LocallyCompactSpace X] (x : X) :
    (𝓝 x).HasBasis (fun s => s ∈ 𝓝 x ∧ IsCompact s) fun s => s :=
                        /-
                          X : Type u_1
                          inst✝¹ : TopologicalSpace X
                          inst✝ : LocallyCompactSpace X
                          x : X
                          ⊢ ∀ (t : Set X), Membership.mem (nhds x) t → Exists fun r => And (Membership.m …
                        -/
  hasBasis_self.2 <| by simpa only [and_comm] using LocallyCompactSpace.local_compact_nhds x
                        /-
                          🎉 no goals
                        -/


theorem local_compact_nhds [LocallyCompactSpace X] {x : X} {n : Set X} (h : n ∈ 𝓝 x) :
    ∃ s ∈ 𝓝 x, s ⊆ n ∧ IsCompact s :=
  LocallyCompactSpace.local_compact_nhds _ _ h


theorem LocallyCompactSpace.of_hasBasis {ι : X → Type*} {p : ∀ x, ι x → Prop}
    {s : ∀ x, ι x → Set X} (h : ∀ x, (𝓝 x).HasBasis (p x) (s x))
    (hc : ∀ x i, p x i → IsCompact (s x i)) : LocallyCompactSpace X :=
  ⟨fun x _t ht =>
    let ⟨i, hp, ht⟩ := (h x).mem_iff.1 ht
    ⟨s x i, (h x).mem_of_mem hp, ht, hc x i hp⟩⟩


instance Prod.locallyCompactSpace (X : Type*) (Y : Type*) [TopologicalSpace X]
    [TopologicalSpace Y] [LocallyCompactSpace X] [LocallyCompactSpace Y] :
    LocallyCompactSpace (X × Y) :=
  have := fun x : X × Y => (compact_basis_nhds x.1).prod_nhds' (compact_basis_nhds x.2)
 .of_hasBasis this fun _ _ ⟨⟨_, h₁⟩, _, h₂⟩ => h₁.prod h₂


/-- In general it suffices that all but finitely many of the spaces are compact,
  but that's not straightforward to state and use. -/
instance Pi.locallyCompactSpace_of_finite [Finite ι] : LocallyCompactSpace (∀ i, X i) :=
  ⟨fun t n hn => by
    /-
      X✝ : Type u_1
      Y : Type u_2
      ι : Type u_3
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s t✝ : Set X✝
      X : ι → Type u_4
      inst✝² : (i : ι) → TopologicalSpace (X i)
      inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
      inst✝ : Finite ι
      t : (i : ι) → X i
      n : Set ((i : ι) → X i)
      hn : Membership.mem (nhds t) n
      ⊢ Exists fun s => And (Membership.mem (nhds t) s) (And (HasSubset.Subset s n)  …
    -/
    rw [nhds_pi, Filter.mem_pi] at hn
    /-
      X✝ : Type u_1
      Y : Type u_2
      ι : Type u_3
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s t✝ : Set X✝
      X : ι → Type u_4
      inst✝² : (i : ι) → TopologicalSpace (X i)
      inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
      inst✝ : Finite ι
      t : (i : ι) → X i
      n : Set ((i : ι) → X i)
      hn : Exists fun I => And I.Finite (Exists fun t_1 => And (∀ (i : ι), Membershi …
      ⊢ Exists fun s => And (Membership.mem (nhds t) s) (And (HasSubset.Subset s n)  …
    -/
    obtain ⟨s, -, n', hn', hsub⟩ := hn
    choose n'' hn'' hsub' hc using fun i =>
      LocallyCompactSpace.local_compact_nhds (t i) (n' i) (hn' i)
    /-
      case intro.intro.intro.intro
      X✝ : Type u_1
      Y : Type u_2
      ι : Type u_3
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s✝ t✝ : Set X✝
      X : ι → Type u_4
      inst✝² : (i : ι) → TopologicalSpace (X i)
      inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
      inst✝ : Finite ι
      t : (i : ι) → X i
      n : Set ((i : ι) → X i)
      s : Set ι
      n' : (i : ι) → Set (X i)
      hn' : ∀ (i : ι), Membership.mem (nhds (t i)) (n' i)
      hsub : HasSubset.Subset (s.pi n') n
      n'' : (i : ι) → Set (X i)
      hn'' : ∀ (i : ι), Membership.mem (nhds (t i)) (n'' i)
      hsub' : ∀ (i : ι), HasSubset.Subset (n'' i) (n' i)
      hc : ∀ (i : ι), IsCompact (n'' i)
      ⊢ Exists fun s => And (Membership.mem (nhds t) s) (And (HasSubset.Subset s n)  …
    -/
    refine ⟨(Set.univ : Set ι).pi n'', ?_, subset_trans (fun _ h => ?_) hsub, isCompact_univ_pi hc⟩
      /-
        case intro.intro.intro.intro.refine_1
        X✝ : Type u_1
        Y : Type u_2
        ι : Type u_3
        inst✝⁴ : TopologicalSpace X✝
        inst✝³ : TopologicalSpace Y
        s✝ t✝ : Set X✝
        X : ι → Type u_4
        inst✝² : (i : ι) → TopologicalSpace (X i)
        inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
        inst✝ : Finite ι
        t : (i : ι) → X i
        n : Set ((i : ι) → X i)
        s : Set ι
        n' : (i : ι) → Set (X i)
        hn' : ∀ (i : ι), Membership.mem (nhds (t i)) (n' i)
        hsub : HasSubset.Subset (s.pi n') n
        n'' : (i : ι) → Set (X i)
        hn'' : ∀ (i : ι), Membership.mem (nhds (t i)) (n'' i)
        hsub' : ∀ (i : ι), HasSubset.Subset (n'' i) (n' i)
        hc : ∀ (i : ι), IsCompact (n'' i)
        ⊢ Membership.mem (nhds t) (Set.univ.pi n'')
      -/
    · exact (set_pi_mem_nhds_iff (@Set.finite_univ ι _) _).mpr fun i _ => hn'' i
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2
        X✝ : Type u_1
        Y : Type u_2
        ι : Type u_3
        inst✝⁴ : TopologicalSpace X✝
        inst✝³ : TopologicalSpace Y
        s✝ t✝ : Set X✝
        X : ι → Type u_4
        inst✝² : (i : ι) → TopologicalSpace (X i)
        inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
        inst✝ : Finite ι
        t : (i : ι) → X i
        n : Set ((i : ι) → X i)
        s : Set ι
        n' : (i : ι) → Set (X i)
        hn' : ∀ (i : ι), Membership.mem (nhds (t i)) (n' i)
        hsub : HasSubset.Subset (s.pi n') n
        n'' : (i : ι) → Set (X i)
        hn'' : ∀ (i : ι), Membership.mem (nhds (t i)) (n'' i)
        hsub' : ∀ (i : ι), HasSubset.Subset (n'' i) (n' i)
        hc : ∀ (i : ι), IsCompact (n'' i)
        x✝ : (i : ι) → X i
        h : Membership.mem (Set.univ.pi n'') x✝
        ⊢ Membership.mem (s.pi n') x✝
      -/
    · exact fun i _ => hsub' i (h i trivial)⟩
      /-
        🎉 no goals
      -/


/-- For spaces that are not Hausdorff. -/
instance Pi.locallyCompactSpace [∀ i, CompactSpace (X i)] : LocallyCompactSpace (∀ i, X i) :=
  ⟨fun t n hn => by
    /-
      X✝ : Type u_1
      Y : Type u_2
      ι : Type u_3
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s t✝ : Set X✝
      X : ι → Type u_4
      inst✝² : (i : ι) → TopologicalSpace (X i)
      inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
      inst✝ : ∀ (i : ι), CompactSpace (X i)
      t : (i : ι) → X i
      n : Set ((i : ι) → X i)
      hn : Membership.mem (nhds t) n
      ⊢ Exists fun s => And (Membership.mem (nhds t) s) (And (HasSubset.Subset s n)  …
    -/
    rw [nhds_pi, Filter.mem_pi] at hn
    /-
      X✝ : Type u_1
      Y : Type u_2
      ι : Type u_3
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s t✝ : Set X✝
      X : ι → Type u_4
      inst✝² : (i : ι) → TopologicalSpace (X i)
      inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
      inst✝ : ∀ (i : ι), CompactSpace (X i)
      t : (i : ι) → X i
      n : Set ((i : ι) → X i)
      hn : Exists fun I => And I.Finite (Exists fun t_1 => And (∀ (i : ι), Membershi …
      ⊢ Exists fun s => And (Membership.mem (nhds t) s) (And (HasSubset.Subset s n)  …
    -/
    obtain ⟨s, hs, n', hn', hsub⟩ := hn
    choose n'' hn'' hsub' hc using fun i =>
      LocallyCompactSpace.local_compact_nhds (t i) (n' i) (hn' i)
    /-
      case intro.intro.intro.intro
      X✝ : Type u_1
      Y : Type u_2
      ι : Type u_3
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s✝ t✝ : Set X✝
      X : ι → Type u_4
      inst✝² : (i : ι) → TopologicalSpace (X i)
      inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
      inst✝ : ∀ (i : ι), CompactSpace (X i)
      t : (i : ι) → X i
      n : Set ((i : ι) → X i)
      s : Set ι
      hs : s.Finite
      n' : (i : ι) → Set (X i)
      hn' : ∀ (i : ι), Membership.mem (nhds (t i)) (n' i)
      hsub : HasSubset.Subset (s.pi n') n
      n'' : (i : ι) → Set (X i)
      hn'' : ∀ (i : ι), Membership.mem (nhds (t i)) (n'' i)
      hsub' : ∀ (i : ι), HasSubset.Subset (n'' i) (n' i)
      hc : ∀ (i : ι), IsCompact (n'' i)
      ⊢ Exists fun s => And (Membership.mem (nhds t) s) (And (HasSubset.Subset s n)  …
    -/
    refine ⟨s.pi n'', ?_, subset_trans (fun _ => ?_) hsub, ?_⟩
      /-
        case intro.intro.intro.intro.refine_1
        X✝ : Type u_1
        Y : Type u_2
        ι : Type u_3
        inst✝⁴ : TopologicalSpace X✝
        inst✝³ : TopologicalSpace Y
        s✝ t✝ : Set X✝
        X : ι → Type u_4
        inst✝² : (i : ι) → TopologicalSpace (X i)
        inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
        inst✝ : ∀ (i : ι), CompactSpace (X i)
        t : (i : ι) → X i
        n : Set ((i : ι) → X i)
        s : Set ι
        hs : s.Finite
        n' : (i : ι) → Set (X i)
        hn' : ∀ (i : ι), Membership.mem (nhds (t i)) (n' i)
        hsub : HasSubset.Subset (s.pi n') n
        n'' : (i : ι) → Set (X i)
        hn'' : ∀ (i : ι), Membership.mem (nhds (t i)) (n'' i)
        hsub' : ∀ (i : ι), HasSubset.Subset (n'' i) (n' i)
        hc : ∀ (i : ι), IsCompact (n'' i)
        ⊢ Membership.mem (nhds t) (s.pi n'')
      -/
    · exact (set_pi_mem_nhds_iff hs _).mpr fun i _ => hn'' i
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2
        X✝ : Type u_1
        Y : Type u_2
        ι : Type u_3
        inst✝⁴ : TopologicalSpace X✝
        inst✝³ : TopologicalSpace Y
        s✝ t✝ : Set X✝
        X : ι → Type u_4
        inst✝² : (i : ι) → TopologicalSpace (X i)
        inst✝¹ : ∀ (i : ι), LocallyCompactSpace (X i)
        inst✝ : ∀ (i : ι), CompactSpace (X i)
        t : (i : ι) → X i
        n : Set ((i : ι) → X i)
        s : Set ι
        hs : s.Finite
        n' : (i : ι) → Set (X i)
        hn' : ∀ (i : ι), Membership.mem (nhds (t i)) (n' i)
        hsub : HasSubset.Subset (s.pi n') n
        n'' : (i : ι) → Set (X i)
        hn'' : ∀ (i : ι), Membership.mem (nhds (t i)) (n'' i)
        hsub' : ∀ (i : ι), HasSubset.Subset (n'' i) (n' i)
        hc : ∀ (i : ι), IsCompact (n'' i)
        x✝ : (i : ι) → X i
        ⊢ Membership.mem (s.pi n'') x✝ → Membership.mem (s.pi n') x✝
      -/
    · exact forall₂_imp fun i _ hi' => hsub' i hi'
      /-
        🎉 no goals
      -/
    · classical
      rw [← Set.univ_pi_ite]
      refine isCompact_univ_pi fun i => ?_
      by_cases h : i ∈ s
      · rw [if_pos h]
        exact hc i
      · rw [if_neg h]
        exact CompactSpace.isCompact_univ⟩


instance Function.locallyCompactSpace_of_finite [Finite ι] [LocallyCompactSpace Y] :
    LocallyCompactSpace (ι → Y) :=
  Pi.locallyCompactSpace_of_finite


instance Function.locallyCompactSpace [LocallyCompactSpace Y] [CompactSpace Y] :
    LocallyCompactSpace (ι → Y) :=
  Pi.locallyCompactSpace


instance (priority := 900) [LocallyCompactSpace X] : LocallyCompactPair X Y where
  exists_mem_nhds_isCompact_mapsTo hf hs :=
    let ⟨K, hKx, hKs, hKc⟩ := local_compact_nhds (hf.continuousAt hs); ⟨K, hKx, hKc, hKs⟩


instance (priority := 100) [LocallyCompactSpace X] : WeaklyLocallyCompactSpace X where
  exists_compact_mem_nhds (x : X) :=
    let ⟨K, hx, _, hKc⟩ := local_compact_nhds (x := x) univ_mem; ⟨K, hKc, hx⟩


/-- A reformulation of the definition of locally compact space: In a locally compact space,
  every open set containing `x` has a compact subset containing `x` in its interior. -/
theorem exists_compact_subset [LocallyCompactSpace X] {x : X} {U : Set X} (hU : IsOpen U)
    (hx : x ∈ U) : ∃ K : Set X, IsCompact K ∧ x ∈ interior K ∧ K ⊆ U := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocallyCompactSpace X
    x : X
    U : Set X
    hU : IsOpen U
    hx : Membership.mem U x
    ⊢ Exists fun K => And (IsCompact K) (And (Membership.mem (interior K) x) (HasS …
  -/
  rcases LocallyCompactSpace.local_compact_nhds x U (hU.mem_nhds hx) with ⟨K, h1K, h2K, h3K⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : LocallyCompactSpace X
    x : X
    U : Set X
    hU : IsOpen U
    hx : Membership.mem U x
    K : Set X
    h1K : Membership.mem (nhds x) K
    h2K : HasSubset.Subset K U
    h3K : IsCompact K
    ⊢ Exists fun K => And (IsCompact K) (And (Membership.mem (interior K) x) (HasS …
  -/
  exact ⟨K, h3K, mem_interior_iff_mem_nhds.2 h1K, h2K⟩
  /-
    🎉 no goals
  -/


/-- If `f : X → Y` is a continuous map in a locally compact pair of topological spaces,
`K : set X` is a compact set, and `U` is an open neighbourhood of `f '' K`,
then there exists a compact neighbourhood `L` of `K` such that `f` maps `L` to `U`.

This is a generalization of `exists_mem_nhds_isCompact_mapsTo`. -/
lemma exists_mem_nhdsSet_isCompact_mapsTo [LocallyCompactPair X Y] {f : X → Y} {K : Set X}
    {U : Set Y} (hf : Continuous f) (hK : IsCompact K) (hU : IsOpen U) (hKU : MapsTo f K U) :
    ∃ L ∈ 𝓝ˢ K, IsCompact L ∧ MapsTo f L U := by
  choose! V hxV hVc hVU using fun x (hx : x ∈ K) ↦
    exists_mem_nhds_isCompact_mapsTo hf (hU.mem_nhds (hKU hx))
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : LocallyCompactPair X Y
    f : X → Y
    K : Set X
    U : Set Y
    hf : Continuous f
    hK : IsCompact K
    hU : IsOpen U
    hKU : Set.MapsTo f K U
    V : X → Set X
    hxV : ∀ (x : X), Membership.mem K x → Membership.mem (nhds x) (V x)
    hVc : ∀ (x : X), Membership.mem K x → IsCompact (V x)
    hVU : ∀ (x : X), Membership.mem K x → Set.MapsTo f (V x) U
    ⊢ Exists fun L => And (Membership.mem (nhdsSet K) L) (And (IsCompact L) (Set.M …
  -/
  rcases hK.elim_nhds_subcover_nhdsSet hxV with ⟨s, hsK, hKs⟩
  exact ⟨_, hKs, s.isCompact_biUnion fun x hx ↦ hVc x (hsK x hx), mapsTo_iUnion₂.2 fun x hx ↦
    hVU x (hsK x hx)⟩


/-- In a locally compact space, for every containment `K ⊆ U` of a compact set `K` in an open
  set `U`, there is a compact neighborhood `L` such that `K ⊆ L ⊆ U`: equivalently, there is a
  compact `L` such that `K ⊆ interior L` and `L ⊆ U`.
  See also `exists_compact_closed_between`, in which one guarantees additionally that `L` is closed
  if the space is regular. -/
theorem exists_compact_between [LocallyCompactSpace X] {K U : Set X} (hK : IsCompact K)
    (hU : IsOpen U) (h_KU : K ⊆ U) : ∃ L, IsCompact L ∧ K ⊆ interior L ∧ L ⊆ U :=
  let ⟨L, hKL, hL, hLU⟩ := exists_mem_nhdsSet_isCompact_mapsTo continuous_id hK hU h_KU
  ⟨L, hL, subset_interior_iff_mem_nhdsSet.2 hKL, hLU⟩


theorem IsOpenQuotientMap.locallyCompactSpace [LocallyCompactSpace X] {f : X → Y}
    (hf : IsOpenQuotientMap f) : LocallyCompactSpace Y where
  local_compact_nhds := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : LocallyCompactSpace X
      f : X → Y
      hf : IsOpenQuotientMap f
      ⊢ ∀ (x : Y) (n : Set Y), Membership.mem (nhds x) n → Exists fun s => And (Memb …
    -/
    refine hf.surjective.forall.2 fun x U hU ↦ ?_
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : LocallyCompactSpace X
      f : X → Y
      hf : IsOpenQuotientMap f
      x : X
      U : Set Y
      hU : Membership.mem (nhds (f x)) U
      ⊢ Exists fun s => And (Membership.mem (nhds (f x)) s) (And (HasSubset.Subset s …
    -/
    rcases local_compact_nhds (hf.continuous.continuousAt hU) with ⟨K, hKx, hKU, hKc⟩
    /-
      case intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : LocallyCompactSpace X
      f : X → Y
      hf : IsOpenQuotientMap f
      x : X
      U : Set Y
      hU : Membership.mem (nhds (f x)) U
      K : Set X
      hKx : Membership.mem (nhds x) K
      hKU : HasSubset.Subset K (Set.preimage f U)
      hKc : IsCompact K
      ⊢ Exists fun s => And (Membership.mem (nhds (f x)) s) (And (HasSubset.Subset s …
    -/
    exact ⟨f '' K, hf.isOpenMap.image_mem_nhds hKx, image_subset_iff.2 hKU, hKc.image hf.continuous⟩
    /-
      🎉 no goals
    -/


/-- If `f` is a topology inducing map with a locally compact codomain and a locally closed range,
then the domain of `f` is a locally compact space. -/
theorem Topology.IsInducing.locallyCompactSpace [LocallyCompactSpace Y] {f : X → Y}
    (hf : IsInducing f) (h : IsLocallyClosed (range f)) : LocallyCompactSpace X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : LocallyCompactSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    h : IsLocallyClosed (Set.range f)
    ⊢ LocallyCompactSpace X
  -/
  rcases h with ⟨U, Z, hU, hZ, hUZ⟩
  have (x : X) : (𝓝 x).HasBasis (fun s ↦ (s ∈ 𝓝 (f x) ∧ IsCompact s) ∧ s ⊆ U)
      (fun s ↦ f ⁻¹' (s ∩ Z)) := by
    have H : U ∈ 𝓝 (f x) := hU.mem_nhds (hUZ.subset <| mem_range_self _).1
    rw [hf.nhds_eq_comap, ← comap_nhdsWithin_range, hUZ,
      nhdsWithin_inter_of_mem (nhdsWithin_le_nhds H)]
    exact (nhdsWithin_hasBasis ((compact_basis_nhds (f x)).restrict_subset H) _).comap _
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : LocallyCompactSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    U Z : Set Y
    hU : IsOpen U
    hZ : IsClosed Z
    hUZ : Eq (Set.range f) (Inter.inter U Z)
    this : ∀ (x : X), (nhds x).HasBasis (fun s => And (And (Membership.mem (nhds ( …
    ⊢ LocallyCompactSpace X
  -/
  refine .of_hasBasis this fun x s ⟨⟨_, hs⟩, hsU⟩ ↦ ?_
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : LocallyCompactSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    U Z : Set Y
    hU : IsOpen U
    hZ : IsClosed Z
    hUZ : Eq (Set.range f) (Inter.inter U Z)
    this : ∀ (x : X), (nhds x).HasBasis (fun s => And (And (Membership.mem (nhds ( …
    x : X
    s : Set Y
    x✝ : And (And (Membership.mem (nhds (f x)) s) (IsCompact s)) (HasSubset.Subset …
    left✝ : Membership.mem (nhds (f x)) s
    hs : IsCompact s
    hsU : HasSubset.Subset s U
    ⊢ IsCompact (Set.preimage f (Inter.inter s Z))
  -/
  rw [hf.isCompact_preimage_iff]
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : LocallyCompactSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    U Z : Set Y
    hU : IsOpen U
    hZ : IsClosed Z
    hUZ : Eq (Set.range f) (Inter.inter U Z)
    this : ∀ (x : X), (nhds x).HasBasis (fun s => And (And (Membership.mem (nhds ( …
    x : X
    s : Set Y
    x✝ : And (And (Membership.mem (nhds (f x)) s) (IsCompact s)) (HasSubset.Subset …
    left✝ : Membership.mem (nhds (f x)) s
    hs : IsCompact s
    hsU : HasSubset.Subset s U
    ⊢ IsCompact (Inter.inter s Z)
  -/
  exacts [hs.inter_right hZ, hUZ ▸ by gcongr]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias Inducing.locallyCompactSpace := IsInducing.locallyCompactSpace


protected theorem Topology.IsClosedEmbedding.locallyCompactSpace [LocallyCompactSpace Y] {f : X → Y}
    (hf : IsClosedEmbedding f) : LocallyCompactSpace X :=
  hf.isInducing.locallyCompactSpace hf.isClosed_range.isLocallyClosed


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.locallyCompactSpace := IsClosedEmbedding.locallyCompactSpace


protected theorem Topology.IsOpenEmbedding.locallyCompactSpace [LocallyCompactSpace Y] {f : X → Y}
    (hf : IsOpenEmbedding f) : LocallyCompactSpace X :=
  hf.isInducing.locallyCompactSpace hf.isOpen_range.isLocallyClosed


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.locallyCompactSpace := IsOpenEmbedding.locallyCompactSpace


protected theorem IsLocallyClosed.locallyCompactSpace [LocallyCompactSpace X] {s : Set X}
    (hs : IsLocallyClosed s) : LocallyCompactSpace s :=
                                                   /-
                                                     X : Type u_1
                                                     inst✝¹ : TopologicalSpace X
                                                     inst✝ : LocallyCompactSpace X
                                                     s : Set X
                                                     hs : IsLocallyClosed s
                                                     ⊢ IsLocallyClosed (Set.range Subtype.val)
                                                   -/
  IsEmbedding.subtypeVal.locallyCompactSpace <| by rwa [Subtype.range_val]
                                                   /-
                                                     🎉 no goals
                                                   -/


protected theorem IsClosed.locallyCompactSpace [LocallyCompactSpace X] {s : Set X}
    (hs : IsClosed s) : LocallyCompactSpace s :=
  hs.isLocallyClosed.locallyCompactSpace


protected theorem IsOpen.locallyCompactSpace [LocallyCompactSpace X] {s : Set X} (hs : IsOpen s) :
    LocallyCompactSpace s :=
  hs.isLocallyClosed.locallyCompactSpace

