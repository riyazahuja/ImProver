/-- Any nonempty compact Hausdorff semigroup where right-multiplication is continuous contains
an idempotent, i.e. an `m` such that `m * m = m`. -/
@[to_additive
      "Any nonempty compact Hausdorff additive semigroup where right-addition is continuous
      contains an idempotent, i.e. an `m` such that `m + m = m`"]
theorem exists_idempotent_of_compact_t2_of_continuous_mul_left {M} [Nonempty M] [Semigroup M]
    [TopologicalSpace M] [CompactSpace M] [T2Space M]
    (continuous_mul_left : ∀ r : M, Continuous (· * r)) : ∃ m : M, m * m = m := by
  /- We apply Zorn's lemma to the poset of nonempty closed subsemigroups of `M`.
     It will turn out that any minimal element is `{m}` for an idempotent `m : M`. -/
  let S : Set (Set M) :=
    { N | IsClosed N ∧ N.Nonempty ∧ ∀ (m) (_ : m ∈ N) (m') (_ : m' ∈ N), m * m' ∈ N }
  /-
    M : Type u_1
    inst✝⁴ : Nonempty M
    inst✝³ : Semigroup M
    inst✝² : TopologicalSpace M
    inst✝¹ : CompactSpace M
    inst✝ : T2Space M
    continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
    S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
    ⊢ Exists fun m => Eq (HMul.hMul m m) m
  -/
  rsuffices ⟨N, hN⟩ : ∃ N', Minimal (· ∈ S) N'
    /-
      case intro
      M : Type u_1
      inst✝⁴ : Nonempty M
      inst✝³ : Semigroup M
      inst✝² : TopologicalSpace M
      inst✝¹ : CompactSpace M
      inst✝ : T2Space M
      continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
      S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
      N : Set M
      hN : Minimal (fun x => Membership.mem S x) N
      ⊢ Exists fun m => Eq (HMul.hMul m m) m
    -/
  · obtain ⟨N_closed, ⟨m, hm⟩, N_mul⟩ := hN.prop
    /-
      case intro.intro.intro.intro
      M : Type u_1
      inst✝⁴ : Nonempty M
      inst✝³ : Semigroup M
      inst✝² : TopologicalSpace M
      inst✝¹ : CompactSpace M
      inst✝ : T2Space M
      continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
      S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
      N : Set M
      hN : Minimal (fun x => Membership.mem S x) N
      N_closed : IsClosed N
      N_mul : ∀ (m : M), Membership.mem N m → ∀ (m' : M), Membership.mem N m' → Memb …
      m : M
      hm : Membership.mem N m
      ⊢ Exists fun m => Eq (HMul.hMul m m) m
    -/
    use m
    /- We now have an element `m : M` of a minimal subsemigroup `N`, and want to show `m + m = m`.
    We first show that every element of `N` is of the form `m' + m`. -/
    have scaling_eq_self : (· * m) '' N = N := by
      apply hN.eq_of_subset
      · refine ⟨(continuous_mul_left m).isClosedMap _ N_closed, ⟨_, ⟨m, hm, rfl⟩⟩, ?_⟩
        rintro _ ⟨m'', hm'', rfl⟩ _ ⟨m', hm', rfl⟩
        exact ⟨m'' * m * m', N_mul _ (N_mul _ hm'' _ hm) _ hm', mul_assoc _ _ _⟩
      · rintro _ ⟨m', hm', rfl⟩
        exact N_mul _ hm' _ hm
    /- In particular, this means that `m' * m = m` for some `m'`. We now use minimality again
       to show that this holds for all `m' ∈ N`. -/
    have absorbing_eq_self : N ∩ { m' | m' * m = m } = N := by
      apply hN.eq_of_subset
      · refine ⟨N_closed.inter ((T1Space.t1 m).preimage (continuous_mul_left m)), ?_, ?_⟩
        · rwa [← scaling_eq_self] at hm
        · rintro m'' ⟨mem'', eq'' : _ = m⟩ m' ⟨mem', eq' : _ = m⟩
          refine ⟨N_mul _ mem'' _ mem', ?_⟩
          rw [Set.mem_setOf_eq, mul_assoc, eq', eq'']
      apply Set.inter_subset_left
    /-
      case h
      M : Type u_1
      inst✝⁴ : Nonempty M
      inst✝³ : Semigroup M
      inst✝² : TopologicalSpace M
      inst✝¹ : CompactSpace M
      inst✝ : T2Space M
      continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
      S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
      N : Set M
      hN : Minimal (fun x => Membership.mem S x) N
      N_closed : IsClosed N
      N_mul : ∀ (m : M), Membership.mem N m → ∀ (m' : M), Membership.mem N m' → Memb …
      m : M
      hm : Membership.mem N m
      scaling_eq_self : Eq (Set.image (fun x => HMul.hMul x m) N) N
      absorbing_eq_self : Eq (Inter.inter N (setOf fun m' => Eq (HMul.hMul m' m) m)) N
      ⊢ Eq (HMul.hMul m m) m
    -/
    rw [← absorbing_eq_self] at hm
    /-
      case h
      M : Type u_1
      inst✝⁴ : Nonempty M
      inst✝³ : Semigroup M
      inst✝² : TopologicalSpace M
      inst✝¹ : CompactSpace M
      inst✝ : T2Space M
      continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
      S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
      N : Set M
      hN : Minimal (fun x => Membership.mem S x) N
      N_closed : IsClosed N
      N_mul : ∀ (m : M), Membership.mem N m → ∀ (m' : M), Membership.mem N m' → Memb …
      m : M
      hm : Membership.mem (Inter.inter N (setOf fun m' => Eq (HMul.hMul m' m) m)) m
      scaling_eq_self : Eq (Set.image (fun x => HMul.hMul x m) N) N
      absorbing_eq_self : Eq (Inter.inter N (setOf fun m' => Eq (HMul.hMul m' m) m)) N
      ⊢ Eq (HMul.hMul m m) m
    -/
    exact hm.2
    /-
      🎉 no goals
    -/
  /-
    M : Type u_1
    inst✝⁴ : Nonempty M
    inst✝³ : Semigroup M
    inst✝² : TopologicalSpace M
    inst✝¹ : CompactSpace M
    inst✝ : T2Space M
    continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
    S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
    ⊢ Exists fun N' => Minimal (fun x => Membership.mem S x) N'
  -/
  refine zorn_superset _ fun c hcs hc => ?_
  refine
    ⟨⋂₀ c, ⟨isClosed_sInter fun t ht => (hcs ht).1, ?_, fun m hm m' hm' => ?_⟩, fun s hs =>
      Set.sInter_subset_of_mem hs⟩
    /-
      case refine_1
      M : Type u_1
      inst✝⁴ : Nonempty M
      inst✝³ : Semigroup M
      inst✝² : TopologicalSpace M
      inst✝¹ : CompactSpace M
      inst✝ : T2Space M
      continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
      S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
      c : Set (Set M)
      hcs : HasSubset.Subset c S
      hc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
      ⊢ c.sInter.Nonempty
    -/
  · obtain rfl | hcnemp := c.eq_empty_or_nonempty
      /-
        case refine_1.inl
        M : Type u_1
        inst✝⁴ : Nonempty M
        inst✝³ : Semigroup M
        inst✝² : TopologicalSpace M
        inst✝¹ : CompactSpace M
        inst✝ : T2Space M
        continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
        S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
        hcs : HasSubset.Subset EmptyCollection.emptyCollection S
        hc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) EmptyCollection.emptyCollec …
        ⊢ EmptyCollection.emptyCollection.sInter.Nonempty
      -/
    · rw [Set.sInter_empty]
      /-
        case refine_1.inl
        M : Type u_1
        inst✝⁴ : Nonempty M
        inst✝³ : Semigroup M
        inst✝² : TopologicalSpace M
        inst✝¹ : CompactSpace M
        inst✝ : T2Space M
        continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
        S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
        hcs : HasSubset.Subset EmptyCollection.emptyCollection S
        hc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) EmptyCollection.emptyCollec …
        ⊢ Set.univ.Nonempty
      -/
      apply Set.univ_nonempty
      /-
        🎉 no goals
      -/
    convert
      @IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed _ _ _ hcnemp.coe_sort
        ((↑) : c → Set M) ?_ ?_ ?_ ?_
      /-
        case h.e'_2
        M : Type u_1
        inst✝⁴ : Nonempty M
        inst✝³ : Semigroup M
        inst✝² : TopologicalSpace M
        inst✝¹ : CompactSpace M
        inst✝ : T2Space M
        continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
        S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
        c : Set (Set M)
        hcs : HasSubset.Subset c S
        hc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
        hcnemp : c.Nonempty
        ⊢ Eq c.sInter (Set.iInter fun i => ↑i)
      -/
    · exact Set.sInter_eq_iInter
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.convert_1
        M : Type u_1
        inst✝⁴ : Nonempty M
        inst✝³ : Semigroup M
        inst✝² : TopologicalSpace M
        inst✝¹ : CompactSpace M
        inst✝ : T2Space M
        continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
        S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
        c : Set (Set M)
        hcs : HasSubset.Subset c S
        hc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
        hcnemp : c.Nonempty
        ⊢ Directed (fun x1 x2 => Superset x1 x2) Subtype.val
      -/
    · refine DirectedOn.directed_val (IsChain.directedOn hc.symm)
      /-
        🎉 no goals
      -/
    /-
      case refine_1.inr.convert_2
      M : Type u_1
      inst✝⁴ : Nonempty M
      inst✝³ : Semigroup M
      inst✝² : TopologicalSpace M
      inst✝¹ : CompactSpace M
      inst✝ : T2Space M
      continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
      S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
      c : Set (Set M)
      hcs : HasSubset.Subset c S
      hc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
      hcnemp : c.Nonempty
      ⊢ ∀ (i : ↑c), (↑i).Nonempty
    -/
    exacts [fun i => (hcs i.prop).2.1, fun i => (hcs i.prop).1.isCompact, fun i => (hcs i.prop).1]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝⁴ : Nonempty M
      inst✝³ : Semigroup M
      inst✝² : TopologicalSpace M
      inst✝¹ : CompactSpace M
      inst✝ : T2Space M
      continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
      S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
      c : Set (Set M)
      hcs : HasSubset.Subset c S
      hc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
      m : M
      hm : Membership.mem c.sInter m
      m' : M
      hm' : Membership.mem c.sInter m'
      ⊢ Membership.mem c.sInter (HMul.hMul m m')
    -/
  · rw [Set.mem_sInter]
    /-
      case refine_2
      M : Type u_1
      inst✝⁴ : Nonempty M
      inst✝³ : Semigroup M
      inst✝² : TopologicalSpace M
      inst✝¹ : CompactSpace M
      inst✝ : T2Space M
      continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
      S : Set (Set M) := setOf fun N => And (IsClosed N) (And N.Nonempty (∀ (m : M), …
      c : Set (Set M)
      hcs : HasSubset.Subset c S
      hc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
      m : M
      hm : Membership.mem c.sInter m
      m' : M
      hm' : Membership.mem c.sInter m'
      ⊢ ∀ (t : Set M), Membership.mem c t → Membership.mem t (HMul.hMul m m')
    -/
    exact fun t ht => (hcs ht).2.2 m (Set.mem_sInter.mp hm t ht) m' (Set.mem_sInter.mp hm' t ht)
    /-
      🎉 no goals
    -/


/-- A version of `exists_idempotent_of_compact_t2_of_continuous_mul_left` where the idempotent lies
in some specified nonempty compact subsemigroup. -/
@[to_additive exists_idempotent_in_compact_add_subsemigroup
      "A version of
      `exists_idempotent_of_compact_t2_of_continuous_add_left` where the idempotent lies in
      some specified nonempty compact additive subsemigroup."]
theorem exists_idempotent_in_compact_subsemigroup {M} [Semigroup M] [TopologicalSpace M] [T2Space M]
    (continuous_mul_left : ∀ r : M, Continuous (· * r)) (s : Set M) (snemp : s.Nonempty)
    (s_compact : IsCompact s) (s_add : ∀ᵉ (x ∈ s) (y ∈ s), x * y ∈ s) :
    ∃ m ∈ s, m * m = m := by
  /-
    M : Type u_1
    inst✝² : Semigroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : T2Space M
    continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
    s : Set M
    snemp : s.Nonempty
    s_compact : IsCompact s
    s_add : ∀ (x : M), Membership.mem s x → ∀ (y : M), Membership.mem s y → Member …
    ⊢ Exists fun m => And (Membership.mem s m) (Eq (HMul.hMul m m) m)
  -/
  let M' := { m // m ∈ s }
  letI : Semigroup M' :=
    { mul := fun p q => ⟨p.1 * q.1, s_add _ p.2 _ q.2⟩
      mul_assoc := fun p q r => Subtype.eq (mul_assoc _ _ _) }
  /-
    M : Type u_1
    inst✝² : Semigroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : T2Space M
    continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
    s : Set M
    snemp : s.Nonempty
    s_compact : IsCompact s
    s_add : ∀ (x : M), Membership.mem s x → ∀ (y : M), Membership.mem s y → Member …
    M' : Type (max 0 u_1) := Subtype fun m => Membership.mem s m
    this : Semigroup M' := Semigroup.mk ⋯
    ⊢ Exists fun m => And (Membership.mem s m) (Eq (HMul.hMul m m) m)
  -/
  haveI : CompactSpace M' := isCompact_iff_compactSpace.mp s_compact
  /-
    M : Type u_1
    inst✝² : Semigroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : T2Space M
    continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
    s : Set M
    snemp : s.Nonempty
    s_compact : IsCompact s
    s_add : ∀ (x : M), Membership.mem s x → ∀ (y : M), Membership.mem s y → Member …
    M' : Type (max 0 u_1) := Subtype fun m => Membership.mem s m
    this✝ : Semigroup M' := Semigroup.mk ⋯
    this : CompactSpace M'
    ⊢ Exists fun m => And (Membership.mem s m) (Eq (HMul.hMul m m) m)
  -/
  haveI : Nonempty M' := nonempty_subtype.mpr snemp
  have : ∀ p : M', Continuous (· * p) := fun p =>
    ((continuous_mul_left p.1).comp continuous_subtype_val).subtype_mk _
  /-
    M : Type u_1
    inst✝² : Semigroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : T2Space M
    continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
    s : Set M
    snemp : s.Nonempty
    s_compact : IsCompact s
    s_add : ∀ (x : M), Membership.mem s x → ∀ (y : M), Membership.mem s y → Member …
    M' : Type (max 0 u_1) := Subtype fun m => Membership.mem s m
    this✝² : Semigroup M' := Semigroup.mk ⋯
    this✝¹ : CompactSpace M'
    this✝ : Nonempty M'
    this : ∀ (p : M'), Continuous fun x => HMul.hMul x p
    ⊢ Exists fun m => And (Membership.mem s m) (Eq (HMul.hMul m m) m)
  -/
  obtain ⟨⟨m, hm⟩, idem⟩ := exists_idempotent_of_compact_t2_of_continuous_mul_left this
  /-
    case intro.mk
    M : Type u_1
    inst✝² : Semigroup M
    inst✝¹ : TopologicalSpace M
    inst✝ : T2Space M
    continuous_mul_left : ∀ (r : M), Continuous fun x => HMul.hMul x r
    s : Set M
    snemp : s.Nonempty
    s_compact : IsCompact s
    s_add : ∀ (x : M), Membership.mem s x → ∀ (y : M), Membership.mem s y → Member …
    M' : Type (max 0 u_1) := Subtype fun m => Membership.mem s m
    this✝² : Semigroup M' := Semigroup.mk ⋯
    this✝¹ : CompactSpace M'
    this✝ : Nonempty M'
    this : ∀ (p : M'), Continuous fun x => HMul.hMul x p
    m : M
    hm : Membership.mem s m
    idem : Eq (HMul.hMul ⟨m, hm⟩ ⟨m, hm⟩) ⟨m, hm⟩
    ⊢ Exists fun m => And (Membership.mem s m) (Eq (HMul.hMul m m) m)
  -/
  exact ⟨m, hm, Subtype.ext_iff.mp idem⟩
  /-
    🎉 no goals
  -/

