/-- Let `c : ι → Set α` be an open cover of a compact set `s`. Then there exists an entourage
`n` such that for each `x ∈ s` its `n`-neighborhood is contained in some `c i`. -/
theorem lebesgue_number_lemma {ι : Sort*} {U : ι → Set α} (hK : IsCompact K)
    (hopen : ∀ i, IsOpen (U i)) (hcover : K ⊆ ⋃ i, U i) :
    ∃ V ∈ 𝓤 α, ∀ x ∈ K, ∃ i, ball x V ⊆ U i := by
  have : ∀ x ∈ K, ∃ i, ∃ V ∈ 𝓤 α, ball x (V ○ V) ⊆ U i := fun x hx ↦ by
    obtain ⟨i, hi⟩ := mem_iUnion.1 (hcover hx)
    rw [← (hopen i).mem_nhds_iff, nhds_eq_comap_uniformity, ← lift'_comp_uniformity] at hi
    exact ⟨i, (((basis_sets _).lift' <| monotone_id.compRel monotone_id).comap _).mem_iff.1 hi⟩
  /-
    α : Type ua
    inst✝ : UniformSpace α
    K : Set α
    ι : Sort u_2
    U : ι → Set α
    hK : IsCompact K
    hopen : ∀ (i : ι), IsOpen (U i)
    hcover : HasSubset.Subset K (Set.iUnion fun i => U i)
    this : ∀ (x : α), Membership.mem K x → Exists fun i => Exists fun V => And (Me …
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (∀ (x : α), Membership …
  -/
  choose ind W hW hWU using this
  rcases hK.elim_nhds_subcover' (fun x hx ↦ ball x (W x hx)) (fun x hx ↦ ball_mem_nhds _ (hW x hx))
    with ⟨t, ht⟩
  /-
    case intro
    α : Type ua
    inst✝ : UniformSpace α
    K : Set α
    ι : Sort u_2
    U : ι → Set α
    hK : IsCompact K
    hopen : ∀ (i : ι), IsOpen (U i)
    hcover : HasSubset.Subset K (Set.iUnion fun i => U i)
    ind : (x : α) → Membership.mem K x → ι
    W : (x : α) → Membership.mem K x → Set (Prod α α)
    hW : ∀ (x : α) (a : Membership.mem K x), Membership.mem (uniformity α) (W x a)
    hWU : ∀ (x : α) (a : Membership.mem K x), HasSubset.Subset (UniformSpace.ball  …
    t : Finset ↑K
    ht : HasSubset.Subset K (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (∀ (x : α), Membership …
  -/
  refine ⟨⋂ x ∈ t, W x x.2, (biInter_finset_mem _).2 fun x _ ↦ hW x x.2, fun x hx ↦ ?_⟩
  /-
    case intro
    α : Type ua
    inst✝ : UniformSpace α
    K : Set α
    ι : Sort u_2
    U : ι → Set α
    hK : IsCompact K
    hopen : ∀ (i : ι), IsOpen (U i)
    hcover : HasSubset.Subset K (Set.iUnion fun i => U i)
    ind : (x : α) → Membership.mem K x → ι
    W : (x : α) → Membership.mem K x → Set (Prod α α)
    hW : ∀ (x : α) (a : Membership.mem K x), Membership.mem (uniformity α) (W x a)
    hWU : ∀ (x : α) (a : Membership.mem K x), HasSubset.Subset (UniformSpace.ball  …
    t : Finset ↑K
    ht : HasSubset.Subset K (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    x : α
    hx : Membership.mem K x
    ⊢ Exists fun i => HasSubset.Subset (UniformSpace.ball x (Set.iInter fun x => S …
  -/
  rcases mem_iUnion₂.1 (ht hx) with ⟨y, hyt, hxy⟩
  /-
    case intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    K : Set α
    ι : Sort u_2
    U : ι → Set α
    hK : IsCompact K
    hopen : ∀ (i : ι), IsOpen (U i)
    hcover : HasSubset.Subset K (Set.iUnion fun i => U i)
    ind : (x : α) → Membership.mem K x → ι
    W : (x : α) → Membership.mem K x → Set (Prod α α)
    hW : ∀ (x : α) (a : Membership.mem K x), Membership.mem (uniformity α) (W x a)
    hWU : ∀ (x : α) (a : Membership.mem K x), HasSubset.Subset (UniformSpace.ball  …
    t : Finset ↑K
    ht : HasSubset.Subset K (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    x : α
    hx : Membership.mem K x
    y : ↑K
    hyt : Membership.mem t y
    hxy : Membership.mem (UniformSpace.ball (↑y) (W ↑y ⋯)) x
    ⊢ Exists fun i => HasSubset.Subset (UniformSpace.ball x (Set.iInter fun x => S …
  -/
  exact ⟨ind y y.2, fun z hz ↦ hWU _ _ ⟨x, hxy, mem_iInter₂.1 hz _ hyt⟩⟩
  /-
    🎉 no goals
  -/


/-- Let `U : ι → Set α` be an open cover of a compact set `K`.
Then there exists an entourage `V`
such that for each `x ∈ K` its `V`-neighborhood is included in some `U i`.

Moreover, one can choose an entourage from a given basis. -/
protected theorem Filter.HasBasis.lebesgue_number_lemma {ι' ι : Sort*} {p : ι' → Prop}
    {V : ι' → Set (α × α)} {U : ι → Set α} (hbasis : (𝓤 α).HasBasis p V) (hK : IsCompact K)
    (hopen : ∀ j, IsOpen (U j)) (hcover : K ⊆ ⋃ j, U j) :
    ∃ i, p i ∧ ∀ x ∈ K, ∃ j, ball x (V i) ⊆ U j := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    K : Set α
    ι' : Sort u_2
    ι : Sort u_3
    p : ι' → Prop
    V : ι' → Set (Prod α α)
    U : ι → Set α
    hbasis : (uniformity α).HasBasis p V
    hK : IsCompact K
    hopen : ∀ (j : ι), IsOpen (U j)
    hcover : HasSubset.Subset K (Set.iUnion fun j => U j)
    ⊢ Exists fun i => And (p i) (∀ (x : α), Membership.mem K x → Exists fun j => H …
  -/
  refine (hbasis.exists_iff ?_).1 (lebesgue_number_lemma hK hopen hcover)
  /-
    α : Type ua
    inst✝ : UniformSpace α
    K : Set α
    ι' : Sort u_2
    ι : Sort u_3
    p : ι' → Prop
    V : ι' → Set (Prod α α)
    U : ι → Set α
    hbasis : (uniformity α).HasBasis p V
    hK : IsCompact K
    hopen : ∀ (j : ι), IsOpen (U j)
    hcover : HasSubset.Subset K (Set.iUnion fun j => U j)
    ⊢ ∀ ⦃s t : Set (Prod α α)⦄, HasSubset.Subset s t → (∀ (x : α), Membership.mem  …
  -/
  exact fun s t hst ht x hx ↦ (ht x hx).imp fun i hi ↦ Subset.trans (ball_mono hst _) hi
  /-
    🎉 no goals
  -/


/-- Let `c : Set (Set α)` be an open cover of a compact set `s`. Then there exists an entourage
`n` such that for each `x ∈ s` its `n`-neighborhood is contained in some `t ∈ c`. -/
theorem lebesgue_number_lemma_sUnion {S : Set (Set α)}
    (hK : IsCompact K) (hopen : ∀ s ∈ S, IsOpen s) (hcover : K ⊆ ⋃₀ S) :
    ∃ V ∈ 𝓤 α, ∀ x ∈ K, ∃ s ∈ S, ball x V ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    K : Set α
    S : Set (Set α)
    hK : IsCompact K
    hopen : ∀ (s : Set α), Membership.mem S s → IsOpen s
    hcover : HasSubset.Subset K S.sUnion
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (∀ (x : α), Membership …
  -/
  rw [sUnion_eq_iUnion] at hcover
  /-
    α : Type ua
    inst✝ : UniformSpace α
    K : Set α
    S : Set (Set α)
    hK : IsCompact K
    hopen : ∀ (s : Set α), Membership.mem S s → IsOpen s
    hcover : HasSubset.Subset K (Set.iUnion fun i => ↑i)
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (∀ (x : α), Membership …
  -/
  simpa using lebesgue_number_lemma hK (by simpa) hcover
  /-
    🎉 no goals
  -/


/-- If `K` is a compact set in a uniform space and `{V i | p i}` is a basis of entourages,
then `{⋃ x ∈ K, UniformSpace.ball x (V i) | p i}` is a basis of `𝓝ˢ K`.

Here "`{s i | p i}` is a basis of a filter `l`" means `Filter.HasBasis l p s`. -/
theorem IsCompact.nhdsSet_basis_uniformity {p : ι → Prop} {V : ι → Set (α × α)}
    (hbasis : (𝓤 α).HasBasis p V) (hK : IsCompact K) :
    (𝓝ˢ K).HasBasis p fun i => ⋃ x ∈ K, ball x (V i) where
  mem_iff' U := by
    /-
      α : Type ua
      ι : Sort u_1
      inst✝ : UniformSpace α
      K : Set α
      p : ι → Prop
      V : ι → Set (Prod α α)
      hbasis : (uniformity α).HasBasis p V
      hK : IsCompact K
      U : Set α
      ⊢ Iff (Membership.mem (nhdsSet K) U) (Exists fun i => And (p i) (HasSubset.Sub …
    -/
    constructor
      /-
        case mp
        α : Type ua
        ι : Sort u_1
        inst✝ : UniformSpace α
        K : Set α
        p : ι → Prop
        V : ι → Set (Prod α α)
        hbasis : (uniformity α).HasBasis p V
        hK : IsCompact K
        U : Set α
        ⊢ Membership.mem (nhdsSet K) U → Exists fun i => And (p i) (HasSubset.Subset ( …
      -/
    · intro H
      have HKU : K ⊆ ⋃ _ : Unit, interior U := by
        simpa only [iUnion_const, subset_interior_iff_mem_nhdsSet] using H
      obtain ⟨i, hpi, hi⟩ : ∃ i, p i ∧ ⋃ x ∈ K, ball x (V i) ⊆ interior U := by
        simpa using hbasis.lebesgue_number_lemma hK (fun _ ↦ isOpen_interior) HKU
      /-
        case mp.intro.intro
        α : Type ua
        ι : Sort u_1
        inst✝ : UniformSpace α
        K : Set α
        p : ι → Prop
        V : ι → Set (Prod α α)
        hbasis : (uniformity α).HasBasis p V
        hK : IsCompact K
        U : Set α
        H : Membership.mem (nhdsSet K) U
        HKU : HasSubset.Subset K (Set.iUnion fun x => interior U)
        i : ι
        hpi : p i
        hi : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ba …
        ⊢ Exists fun i => And (p i) (HasSubset.Subset (Set.iUnion fun x => Set.iUnion  …
      -/
      exact ⟨i, hpi, hi.trans interior_subset⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type ua
        ι : Sort u_1
        inst✝ : UniformSpace α
        K : Set α
        p : ι → Prop
        V : ι → Set (Prod α α)
        hbasis : (uniformity α).HasBasis p V
        hK : IsCompact K
        U : Set α
        ⊢ (Exists fun i => And (p i) (HasSubset.Subset (Set.iUnion fun x => Set.iUnion …
      -/
    · rintro ⟨i, hpi, hi⟩
      /-
        case mpr.intro.intro
        α : Type ua
        ι : Sort u_1
        inst✝ : UniformSpace α
        K : Set α
        p : ι → Prop
        V : ι → Set (Prod α α)
        hbasis : (uniformity α).HasBasis p V
        hK : IsCompact K
        U : Set α
        i : ι
        hpi : p i
        hi : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ba …
        ⊢ Membership.mem (nhdsSet K) U
      -/
      refine mem_of_superset (bUnion_mem_nhdsSet fun x _ ↦ ?_) hi
      /-
        case mpr.intro.intro
        α : Type ua
        ι : Sort u_1
        inst✝ : UniformSpace α
        K : Set α
        p : ι → Prop
        V : ι → Set (Prod α α)
        hbasis : (uniformity α).HasBasis p V
        hK : IsCompact K
        U : Set α
        i : ι
        hpi : p i
        hi : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ba …
        x : α
        x✝ : Membership.mem K x
        ⊢ Membership.mem (nhds x) (UniformSpace.ball x (V i))
      -/
      exact ball_mem_nhds _ <| hbasis.mem_of_mem hpi
      /-
        🎉 no goals
      -/

-- TODO: move to a separate file, golf using the regularity of a uniform space.

theorem Disjoint.exists_uniform_thickening {A B : Set α} (hA : IsCompact A) (hB : IsClosed B)
    (h : Disjoint A B) : ∃ V ∈ 𝓤 α, Disjoint (⋃ x ∈ A, ball x V) (⋃ x ∈ B, ball x V) := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (Disjoint (Set.iUnion  …
  -/
  have : Bᶜ ∈ 𝓝ˢ A := hB.isOpen_compl.mem_nhdsSet.mpr h.le_compl_right
  /-
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    this : Membership.mem (nhdsSet A) (HasCompl.compl B)
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (Disjoint (Set.iUnion  …
  -/
  rw [(hA.nhdsSet_basis_uniformity (Filter.basis_sets _)).mem_iff] at this
  /-
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    this : Exists fun i => And (Membership.mem (uniformity α) i) (HasSubset.Subset …
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (Disjoint (Set.iUnion  …
  -/
  rcases this with ⟨U, hU, hUAB⟩
  /-
    case intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hUAB : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (Disjoint (Set.iUnion  …
  -/
  rcases comp_symm_mem_uniformity_sets hU with ⟨V, hV, hVsymm, hVU⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hUAB : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVsymm : SymmetricRel V
    hVU : HasSubset.Subset (compRel V V) U
    ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (Disjoint (Set.iUnion  …
  -/
  refine ⟨V, hV, Set.disjoint_left.mpr fun x => ?_⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hUAB : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVsymm : SymmetricRel V
    hVU : HasSubset.Subset (compRel V V) U
    x : α
    ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ball x  …
  -/
  simp only [mem_iUnion₂]
  /-
    case intro.intro.intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hUAB : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVsymm : SymmetricRel V
    hVU : HasSubset.Subset (compRel V V) U
    x : α
    ⊢ (Exists fun i => Exists fun j => Membership.mem (UniformSpace.ball i V) x) → …
  -/
  rintro ⟨a, ha, hxa⟩ ⟨b, hb, hxb⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hUAB : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVsymm : SymmetricRel V
    hVU : HasSubset.Subset (compRel V V) U
    x a : α
    ha : Membership.mem A a
    hxa : Membership.mem (UniformSpace.ball a V) x
    b : α
    hb : Membership.mem B b
    hxb : Membership.mem (UniformSpace.ball b V) x
    ⊢ False
  -/
  rw [mem_ball_symmetry hVsymm] at hxa hxb
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hUAB : HasSubset.Subset (Set.iUnion fun x => Set.iUnion fun h => UniformSpace. …
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVsymm : SymmetricRel V
    hVU : HasSubset.Subset (compRel V V) U
    x a : α
    ha : Membership.mem A a
    hxa : Membership.mem (UniformSpace.ball x V) a
    b : α
    hb : Membership.mem B b
    hxb : Membership.mem (UniformSpace.ball x V) b
    ⊢ False
  -/
  exact hUAB (mem_iUnion₂_of_mem ha <| hVU <| mem_comp_of_mem_ball hVsymm hxa hxb) hb
  /-
    🎉 no goals
  -/


theorem Disjoint.exists_uniform_thickening_of_basis {p : ι → Prop} {s : ι → Set (α × α)}
    (hU : (𝓤 α).HasBasis p s) {A B : Set α} (hA : IsCompact A) (hB : IsClosed B)
    (h : Disjoint A B) : ∃ i, p i ∧ Disjoint (⋃ x ∈ A, ball x (s i)) (⋃ x ∈ B, ball x (s i)) := by
  /-
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    s : ι → Set (Prod α α)
    hU : (uniformity α).HasBasis p s
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    ⊢ Exists fun i => And (p i) (Disjoint (Set.iUnion fun x => Set.iUnion fun h => …
  -/
  rcases h.exists_uniform_thickening hA hB with ⟨V, hV, hVAB⟩
  /-
    case intro.intro
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    s : ι → Set (Prod α α)
    hU : (uniformity α).HasBasis p s
    A B : Set α
    hA : IsCompact A
    hB : IsClosed B
    h : Disjoint A B
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVAB : Disjoint (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ball x V …
    ⊢ Exists fun i => And (p i) (Disjoint (Set.iUnion fun x => Set.iUnion fun h => …
  -/
  rcases hU.mem_iff.1 hV with ⟨i, hi, hiV⟩
  exact ⟨i, hi, hVAB.mono (iUnion₂_mono fun a _ => ball_mono hiV a)
    (iUnion₂_mono fun b _ => ball_mono hiV b)⟩


/-- A useful consequence of the Lebesgue number lemma: given any compact set `K` contained in an
open set `U`, we can find an (open) entourage `V` such that the ball of size `V` about any point of
`K` is contained in `U`. -/
theorem lebesgue_number_of_compact_open {K U : Set α} (hK : IsCompact K)
    (hU : IsOpen U) (hKU : K ⊆ U) : ∃ V ∈ 𝓤 α, IsOpen V ∧ ∀ x ∈ K, UniformSpace.ball x V ⊆ U :=
  let ⟨V, ⟨hV, hVo⟩, hVU⟩ :=
    (hK.nhdsSet_basis_uniformity uniformity_hasBasis_open).mem_iff.1 (hU.mem_nhdsSet.2 hKU)
  ⟨V, hV, hVo, iUnion₂_subset_iff.1 hVU⟩



/-- On a compact uniform space, the topology determines the uniform structure, entourages are
exactly the neighborhoods of the diagonal. -/
theorem nhdsSet_diagonal_eq_uniformity [CompactSpace α] : 𝓝ˢ (diagonal α) = 𝓤 α := by
  /-
    α : Type ua
    inst✝¹ : UniformSpace α
    inst✝ : CompactSpace α
    ⊢ Eq (nhdsSet (Set.diagonal α)) (uniformity α)
  -/
  refine nhdsSet_diagonal_le_uniformity.antisymm ?_
  have :
    (𝓤 (α × α)).HasBasis (fun U => U ∈ 𝓤 α) fun U =>
      (fun p : (α × α) × α × α => ((p.1.1, p.2.1), p.1.2, p.2.2)) ⁻¹' U ×ˢ U := by
    rw [uniformity_prod_eq_comap_prod]
    exact (𝓤 α).basis_sets.prod_self.comap _
  /-
    α : Type ua
    inst✝¹ : UniformSpace α
    inst✝ : CompactSpace α
    this : (uniformity (Prod α α)).HasBasis (fun U => Membership.mem (uniformity α …
    ⊢ LE.le (uniformity α) (nhdsSet (Set.diagonal α))
  -/
  refine (isCompact_diagonal.nhdsSet_basis_uniformity this).ge_iff.2 fun U hU => ?_
  exact mem_of_superset hU fun ⟨x, y⟩ hxy => mem_iUnion₂.2
    ⟨(x, x), rfl, refl_mem_uniformity hU, hxy⟩


/-- On a compact uniform space, the topology determines the uniform structure, entourages are
exactly the neighborhoods of the diagonal. -/
theorem compactSpace_uniformity [CompactSpace α] : 𝓤 α = ⨆ x, 𝓝 (x, x) :=
  nhdsSet_diagonal_eq_uniformity.symm.trans (nhdsSet_diagonal _)


theorem unique_uniformity_of_compact [t : TopologicalSpace γ] [CompactSpace γ]
    {u u' : UniformSpace γ} (h : u.toTopologicalSpace = t) (h' : u'.toTopologicalSpace = t) :
    u = u' := by
  /-
    γ : Type uc
    t : TopologicalSpace γ
    inst✝ : CompactSpace γ
    u u' : UniformSpace γ
    h : Eq UniformSpace.toTopologicalSpace t
    h' : Eq UniformSpace.toTopologicalSpace t
    ⊢ Eq u u'
  -/
  refine UniformSpace.ext ?_
  /-
    γ : Type uc
    t : TopologicalSpace γ
    inst✝ : CompactSpace γ
    u u' : UniformSpace γ
    h : Eq UniformSpace.toTopologicalSpace t
    h' : Eq UniformSpace.toTopologicalSpace t
    ⊢ Eq (uniformity γ) (uniformity γ)
  -/
  have : @CompactSpace γ u.toTopologicalSpace := by rwa [h]
  /-
    γ : Type uc
    t : TopologicalSpace γ
    inst✝ : CompactSpace γ
    u u' : UniformSpace γ
    h : Eq UniformSpace.toTopologicalSpace t
    h' : Eq UniformSpace.toTopologicalSpace t
    this : CompactSpace γ
    ⊢ Eq (uniformity γ) (uniformity γ)
  -/
  have : @CompactSpace γ u'.toTopologicalSpace := by rwa [h']
  /-
    γ : Type uc
    t : TopologicalSpace γ
    inst✝ : CompactSpace γ
    u u' : UniformSpace γ
    h : Eq UniformSpace.toTopologicalSpace t
    h' : Eq UniformSpace.toTopologicalSpace t
    this✝ : CompactSpace γ
    this : CompactSpace γ
    ⊢ Eq (uniformity γ) (uniformity γ)
  -/
  rw [@compactSpace_uniformity _ u, compactSpace_uniformity, h, h']
  /-
    🎉 no goals
  -/


