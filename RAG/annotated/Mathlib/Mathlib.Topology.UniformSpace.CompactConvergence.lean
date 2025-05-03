/-- Compact-open topology on `C(α, β)` agrees with the topology of uniform convergence on compacts:
a family of continuous functions `F i` tends to `f` in the compact-open topology
if and only if the `F i` tends to `f` uniformly on all compact sets. -/
theorem tendsto_iff_forall_isCompact_tendstoUniformlyOn
    {ι : Type u₃} {p : Filter ι} {F : ι → C(α, β)} {f} :
    Tendsto F p (𝓝 f) ↔ ∀ K, IsCompact K → TendstoUniformlyOn (fun i a => F i a) f p K := by
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : TopologicalSpace α
    inst✝ : UniformSpace β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    f : ContinuousMap α β
    ⊢ Iff (Filter.Tendsto F p (nhds f)) (∀ (K : Set α), IsCompact K → TendstoUnifo …
  -/
  rw [tendsto_nhds_compactOpen]
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : TopologicalSpace α
    inst✝ : UniformSpace β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    f : ContinuousMap α β
    ⊢ Iff (∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f)  …
  -/
  constructor
  · -- Let us prove that convergence in the compact-open topology
    -- implies uniform convergence on compacts.
    -- Consider a compact set `K`
    /-
      case mp
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      ⊢ (∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U  …
    -/
    intro h K hK
    -- Since `K` is compact, it suffices to prove locally uniform convergence
    /-
      case mp
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      ⊢ TendstoUniformlyOn (fun i a => (F i) a) (⇑f) p K
    -/
    rw [← tendstoLocallyUniformlyOn_iff_tendstoUniformlyOn_of_compact hK]
    -- Now choose an entourage `U` in the codomain and a point `x ∈ K`.
    /-
      case mp
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      ⊢ TendstoLocallyUniformlyOn (fun i a => (F i) a) (⇑f) p K
    -/
    intro U hU x _
    -- Choose an open symmetric entourage `V` such that `V ○ V ⊆ U`.
    /-
      case mp
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Filter.Eventually ( …
    -/
    rcases comp_open_symm_mem_uniformity_sets hU with ⟨V, hV, hVo, hVsymm, hVU⟩
    -- Then choose a closed entourage `W ⊆ V`
    /-
      case mp.intro.intro.intro.intro
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVo : IsOpen V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Filter.Eventually ( …
    -/
    rcases mem_uniformity_isClosed hV with ⟨W, hW, hWc, hWU⟩
    -- Consider `s = {y ∈ K | (f x, f y) ∈ W}`
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVo : IsOpen V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      W : Set (Prod β β)
      hW : Membership.mem (uniformity β) W
      hWc : IsClosed W
      hWU : HasSubset.Subset W V
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Filter.Eventually ( …
    -/
    set s := K ∩ f ⁻¹' ball (f x) W
    -- This is a neighbourhood of `x` within `K`, because `W` is an entourage.
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVo : IsOpen V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      W : Set (Prod β β)
      hW : Membership.mem (uniformity β) W
      hWc : IsClosed W
      hWU : HasSubset.Subset W V
      s : Set α := Inter.inter K (Set.preimage (⇑f) (UniformSpace.ball (f x) W))
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Filter.Eventually ( …
    -/
    have hnhds : s ∈ 𝓝[K] x := inter_mem_nhdsWithin _ <| f.continuousAt _ (ball_mem_nhds _ hW)
    -- This set is compact because it is an intersection of `K`
    -- with a closed set `{y | (f x, f y) ∈ W} = f ⁻¹' UniformSpace.ball (f x) W`
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVo : IsOpen V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      W : Set (Prod β β)
      hW : Membership.mem (uniformity β) W
      hWc : IsClosed W
      hWU : HasSubset.Subset W V
      s : Set α := Inter.inter K (Set.preimage (⇑f) (UniformSpace.ball (f x) W))
      hnhds : Membership.mem (nhdsWithin x K) s
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Filter.Eventually ( …
    -/
    have hcomp : IsCompact s := hK.inter_right <| (isClosed_ball _ hWc).preimage f.continuous
    -- `f` maps `s` to the open set `ball (f x) V = {z | (f x, z) ∈ V}`
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVo : IsOpen V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      W : Set (Prod β β)
      hW : Membership.mem (uniformity β) W
      hWc : IsClosed W
      hWU : HasSubset.Subset W V
      s : Set α := Inter.inter K (Set.preimage (⇑f) (UniformSpace.ball (f x) W))
      hnhds : Membership.mem (nhdsWithin x K) s
      hcomp : IsCompact s
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Filter.Eventually ( …
    -/
    have hmaps : MapsTo f s (ball (f x) V) := fun x hx ↦ hWU hx.2
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVo : IsOpen V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      W : Set (Prod β β)
      hW : Membership.mem (uniformity β) W
      hWc : IsClosed W
      hWU : HasSubset.Subset W V
      s : Set α := Inter.inter K (Set.preimage (⇑f) (UniformSpace.ball (f x) W))
      hnhds : Membership.mem (nhdsWithin x K) s
      hcomp : IsCompact s
      hmaps : Set.MapsTo (⇑f) s (UniformSpace.ball (f x) V)
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Filter.Eventually ( …
    -/
    use s, hnhds
    -- Continuous maps `F i` in a neighbourhood of `f` map `s` to `ball (f x) V` as well.
    /-
      case right
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVo : IsOpen V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      W : Set (Prod β β)
      hW : Membership.mem (uniformity β) W
      hWc : IsClosed W
      hWU : HasSubset.Subset W V
      s : Set α := Inter.inter K (Set.preimage (⇑f) (UniformSpace.ball (f x) W))
      hnhds : Membership.mem (nhdsWithin x K) s
      hcomp : IsCompact s
      hmaps : Set.MapsTo (⇑f) s (UniformSpace.ball (f x) V)
      ⊢ Filter.Eventually (fun n => ∀ (y : α), Membership.mem s y → Membership.mem U …
    -/
    refine (h s hcomp _ (isOpen_ball _ hVo) hmaps).mono fun g hg y hy ↦ ?_
    -- Then for `y ∈ s` we have `(f y, f x) ∈ V` and `(f x, F i y) ∈ V`, thus `(f y, F i y) ∈ U`
    /-
      case right
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → ∀ (U : Set β), IsOpen U → Set.MapsTo (⇑f) K U …
      K : Set α
      hK : IsCompact K
      U : Set (Prod β β)
      hU : Membership.mem (uniformity β) U
      x : α
      a✝ : Membership.mem K x
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVo : IsOpen V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      W : Set (Prod β β)
      hW : Membership.mem (uniformity β) W
      hWc : IsClosed W
      hWU : HasSubset.Subset W V
      s : Set α := Inter.inter K (Set.preimage (⇑f) (UniformSpace.ball (f x) W))
      hnhds : Membership.mem (nhdsWithin x K) s
      hcomp : IsCompact s
      hmaps : Set.MapsTo (⇑f) s (UniformSpace.ball (f x) V)
      g : ι
      hg : Set.MapsTo (⇑(F g)) s (UniformSpace.ball (f x) V)
      y : α
      hy : Membership.mem s y
      ⊢ Membership.mem U { fst := f y, snd := (fun i a => (F i) a) g y }
    -/
    exact hVU ⟨f x, hVsymm.mk_mem_comm.2 <| hmaps hy, hg hy⟩
    /-
      🎉 no goals
    -/
  · -- Now we prove that uniform convergence on compacts
    -- implies convergence in the compact-open topology
    -- Consider a compact set `K`, an open set `U`, and a continuous map `f` that maps `K` to `U`
    /-
      case mpr
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      ⊢ (∀ (K : Set α), IsCompact K → TendstoUniformlyOn (fun i a => (F i) a) (⇑f) p …
    -/
    intro h K hK U hU hf
    -- Due to Lebesgue number lemma, there exists an entourage `V`
    -- such that `U` includes the `V`-thickening of `f '' K`.
    rcases lebesgue_number_of_compact_open (hK.image (map_continuous f)) hU hf.image_subset
        with ⟨V, hV, -, hVf⟩
    -- Then any continuous map that is uniformly `V`-close to `f` on `K`
    -- maps `K` to `U` as well
    /-
      case mpr.intro.intro.intro
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      ι : Type u₃
      p : Filter ι
      F : ι → ContinuousMap α β
      f : ContinuousMap α β
      h : ∀ (K : Set α), IsCompact K → TendstoUniformlyOn (fun i a => (F i) a) (⇑f)  …
      K : Set α
      hK : IsCompact K
      U : Set β
      hU : IsOpen U
      hf : Set.MapsTo (⇑f) K U
      V : Set (Prod β β)
      hV : Membership.mem (uniformity β) V
      hVf : ∀ (x : β), Membership.mem (Set.image (⇑f) K) x → HasSubset.Subset (Unifo …
      ⊢ Filter.Eventually (fun a => Set.MapsTo (⇑(F a)) K U) p
    -/
    filter_upwards [h K hK V hV] with g hg x hx using hVf _ (mem_image_of_mem f hx) (hg x hx)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-19")] alias
tendsto_iff_forall_compact_tendstoUniformlyOn := tendsto_iff_forall_isCompact_tendstoUniformlyOn


/-- Interpret a bundled continuous map as an element of `α →ᵤ[{K | IsCompact K}] β`.

We use this map to induce the `UniformSpace` structure on `C(α, β)`. -/
def toUniformOnFunIsCompact (f : C(α, β)) : α →ᵤ[{K | IsCompact K}] β :=
  UniformOnFun.ofFun {K | IsCompact K} f


@[simp]
theorem toUniformOnFun_toFun (f : C(α, β)) :
    UniformOnFun.toFun _ f.toUniformOnFunIsCompact = f := rfl


theorem range_toUniformOnFunIsCompact :
    range (toUniformOnFunIsCompact) = {f : UniformOnFun α β {K | IsCompact K} | Continuous f} :=
  Set.ext fun f ↦ ⟨fun g ↦ g.choose_spec ▸ g.choose.2, fun hf ↦ ⟨⟨f, hf⟩, rfl⟩⟩


open UniformSpace in
/-- Uniform space structure on `C(α, β)`.

The uniformity comes from `α →ᵤ[{K | IsCompact K}] β` (i.e., `UniformOnFun α β {K | IsCompact K}`)
which defines topology of uniform convergence on compact sets.
We use `ContinuousMap.tendsto_iff_forall_isCompact_tendstoUniformlyOn`
to show that the induced topology agrees with the compact-open topology
and replace the topology with `compactOpen` to avoid non-defeq diamonds,
see Note [forgetful inheritance]. -/
instance compactConvergenceUniformSpace : UniformSpace C(α, β) :=
  .replaceTopology (.comap toUniformOnFunIsCompact inferInstance) <| by
    /-
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      K : Set α
      V : Set (Prod β β)
      f : ContinuousMap α β
      ⊢ Eq ContinuousMap.compactOpen UniformSpace.toTopologicalSpace
    -/
    refine TopologicalSpace.ext_nhds fun f ↦ eq_of_forall_le_iff fun l ↦ ?_
    simp_rw [← tendsto_id', tendsto_iff_forall_isCompact_tendstoUniformlyOn,
      nhds_induced, tendsto_comap_iff, UniformOnFun.tendsto_iff_tendstoUniformlyOn]
    /-
      α : Type u₁
      β : Type u₂
      inst✝¹ : TopologicalSpace α
      inst✝ : UniformSpace β
      K : Set α
      V : Set (Prod β β)
      f✝ f : ContinuousMap α β
      l : Filter (ContinuousMap α β)
      ⊢ Iff (∀ (K : Set α), IsCompact K → TendstoUniformlyOn (fun i a => (id i) a) ( …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem isUniformEmbedding_toUniformOnFunIsCompact :
    IsUniformEmbedding (toUniformOnFunIsCompact : C(α, β) → α →ᵤ[{K | IsCompact K}] β) where
  comap_uniformity := rfl
  injective := DFunLike.coe_injective


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_toUniformOnFunIsCompact := isUniformEmbedding_toUniformOnFunIsCompact

-- The following definitions and theorems
-- used to be a part of the construction of the `UniformSpace C(α, β)` structure
-- before it was migrated to `UniformOnFun`


theorem _root_.Filter.HasBasis.compactConvergenceUniformity {ι : Type*} {pi : ι → Prop}
    {s : ι → Set (β × β)} (h : (𝓤 β).HasBasis pi s) :
    HasBasis (𝓤 C(α, β)) (fun p : Set α × ι => IsCompact p.1 ∧ pi p.2) fun p =>
      { fg : C(α, β) × C(α, β) | ∀ x ∈ p.1, (fg.1 x, fg.2 x) ∈ s p.2 } := by
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : TopologicalSpace α
    inst✝ : UniformSpace β
    ι : Type u_1
    pi : ι → Prop
    s : ι → Set (Prod β β)
    h : (uniformity β).HasBasis pi s
    ⊢ (uniformity (ContinuousMap α β)).HasBasis (fun p => And (IsCompact p.1) (pi  …
  -/
  rw [← isUniformEmbedding_toUniformOnFunIsCompact.comap_uniformity]
  exact .comap _ <| UniformOnFun.hasBasis_uniformity_of_basis _ _ {K | IsCompact K}
    ⟨∅, isCompact_empty⟩ (directedOn_of_sup_mem fun _ _ ↦ IsCompact.union) h


theorem hasBasis_compactConvergenceUniformity :
    HasBasis (𝓤 C(α, β)) (fun p : Set α × Set (β × β) => IsCompact p.1 ∧ p.2 ∈ 𝓤 β) fun p =>
      { fg : C(α, β) × C(α, β) | ∀ x ∈ p.1, (fg.1 x, fg.2 x) ∈ p.2 } :=
  (basis_sets _).compactConvergenceUniformity


theorem mem_compactConvergence_entourage_iff (X : Set (C(α, β) × C(α, β))) :
    X ∈ 𝓤 C(α, β) ↔
      ∃ (K : Set α) (V : Set (β × β)), IsCompact K ∧ V ∈ 𝓤 β ∧
        { fg : C(α, β) × C(α, β) | ∀ x ∈ K, (fg.1 x, fg.2 x) ∈ V } ⊆ X := by
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : TopologicalSpace α
    inst✝ : UniformSpace β
    X : Set (Prod (ContinuousMap α β) (ContinuousMap α β))
    ⊢ Iff (Membership.mem (uniformity (ContinuousMap α β)) X) (Exists fun K => Exi …
  -/
  simp [hasBasis_compactConvergenceUniformity.mem_iff, and_assoc]
  /-
    🎉 no goals
  -/


/-- If `K` is a compact exhaustion of `α`
and `V i` bounded by `p i` is a basis of entourages of `β`,
then `fun (n, i) ↦ {(f, g) | ∀ x ∈ K n, (f x, g x) ∈ V i}` bounded by `p i`
is a basis of entourages of `C(α, β)`. -/
theorem _root_.CompactExhaustion.hasBasis_compactConvergenceUniformity {ι : Type*}
    {p : ι → Prop} {V : ι → Set (β × β)} (K : CompactExhaustion α) (hb : (𝓤 β).HasBasis p V) :
    HasBasis (𝓤 C(α, β)) (fun i : ℕ × ι ↦ p i.2) fun i ↦
      {fg | ∀ x ∈ K i.1, (fg.1 x, fg.2 x) ∈ V i.2} :=
  (UniformOnFun.hasBasis_uniformity_of_covering_of_basis {K | IsCompact K} K.isCompact
    (Monotone.directed_le K.subset) (fun _ ↦ K.exists_superset_of_isCompact) hb).comap _


theorem _root_.CompactExhaustion.hasAntitoneBasis_compactConvergenceUniformity
    {V : ℕ → Set (β × β)} (K : CompactExhaustion α) (hb : (𝓤 β).HasAntitoneBasis V) :
    HasAntitoneBasis (𝓤 C(α, β)) fun n ↦ {fg | ∀ x ∈ K n, (fg.1 x, fg.2 x) ∈ V n} :=
  (UniformOnFun.hasAntitoneBasis_uniformity {K | IsCompact K} K.isCompact
    K.subset (fun _ ↦ K.exists_superset_of_isCompact) hb).comap _


/-- If `α` is a weakly locally compact σ-compact space
(e.g., a proper pseudometric space or a compact spaces)
and the uniformity on `β` is pseudometrizable,
then the uniformity on `C(α, β)` is pseudometrizable too.
-/
instance [WeaklyLocallyCompactSpace α] [SigmaCompactSpace α] [IsCountablyGenerated (𝓤 β)] :
    IsCountablyGenerated (𝓤 (C(α, β))) :=
  let ⟨_V, hV⟩ := exists_antitone_basis (𝓤 β)
  ((CompactExhaustion.choice α).hasAntitoneBasis_compactConvergenceUniformity
    hV).isCountablyGenerated


/-- Locally uniform convergence implies convergence in the compact-open topology. -/
theorem tendsto_of_tendstoLocallyUniformly (h : TendstoLocallyUniformly (fun i a => F i a) f p) :
    Tendsto F p (𝓝 f) := by
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : TopologicalSpace α
    inst✝ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    h : TendstoLocallyUniformly (fun i a => (F i) a) (⇑f) p
    ⊢ Filter.Tendsto F p (nhds f)
  -/
  rw [tendsto_iff_forall_isCompact_tendstoUniformlyOn]
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : TopologicalSpace α
    inst✝ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    h : TendstoLocallyUniformly (fun i a => (F i) a) (⇑f) p
    ⊢ ∀ (K : Set α), IsCompact K → TendstoUniformlyOn (fun i a => (F i) a) (⇑f) p K
  -/
  intro K hK
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : TopologicalSpace α
    inst✝ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    h : TendstoLocallyUniformly (fun i a => (F i) a) (⇑f) p
    K : Set α
    hK : IsCompact K
    ⊢ TendstoUniformlyOn (fun i a => (F i) a) (⇑f) p K
  -/
  rw [← tendstoLocallyUniformlyOn_iff_tendstoUniformlyOn_of_compact hK]
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : TopologicalSpace α
    inst✝ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    h : TendstoLocallyUniformly (fun i a => (F i) a) (⇑f) p
    K : Set α
    hK : IsCompact K
    ⊢ TendstoLocallyUniformlyOn (fun i a => (F i) a) (⇑f) p K
  -/
  exact h.tendstoLocallyUniformlyOn
  /-
    🎉 no goals
  -/


/-- In a weakly locally compact space,
convergence in the compact-open topology is the same as locally uniform convergence.

The right-to-left implication holds in any topological space,
see `ContinuousMap.tendsto_of_tendstoLocallyUniformly`. -/
theorem tendsto_iff_tendstoLocallyUniformly [WeaklyLocallyCompactSpace α] :
    Tendsto F p (𝓝 f) ↔ TendstoLocallyUniformly (fun i a => F i a) f p := by
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    inst✝ : WeaklyLocallyCompactSpace α
    ⊢ Iff (Filter.Tendsto F p (nhds f)) (TendstoLocallyUniformly (fun i a => (F i) …
  -/
  refine ⟨fun h V hV x ↦ ?_, tendsto_of_tendstoLocallyUniformly⟩
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    inst✝ : WeaklyLocallyCompactSpace α
    h : Filter.Tendsto F p (nhds f)
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    x : α
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (Filter.Eventually (fun n => …
  -/
  rw [tendsto_iff_forall_isCompact_tendstoUniformlyOn] at h
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    inst✝ : WeaklyLocallyCompactSpace α
    h : ∀ (K : Set α), IsCompact K → TendstoUniformlyOn (fun i a => (F i) a) (⇑f)  …
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    x : α
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (Filter.Eventually (fun n => …
  -/
  obtain ⟨n, hn₁, hn₂⟩ := exists_compact_mem_nhds x
  /-
    case intro.intro
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    inst✝ : WeaklyLocallyCompactSpace α
    h : ∀ (K : Set α), IsCompact K → TendstoUniformlyOn (fun i a => (F i) a) (⇑f)  …
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    x : α
    n : Set α
    hn₁ : IsCompact n
    hn₂ : Membership.mem (nhds x) n
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (Filter.Eventually (fun n => …
  -/
  exact ⟨n, hn₂, h n hn₁ V hV⟩
  /-
    🎉 no goals
  -/


theorem uniformContinuous_comp (g : C(β, δ)) (hg : UniformContinuous g) :
    UniformContinuous (ContinuousMap.comp g : C(α, β) → C(α, δ)) :=
  isUniformEmbedding_toUniformOnFunIsCompact.uniformContinuous_iff.mpr <|
    UniformOnFun.postcomp_uniformContinuous hg |>.comp
      isUniformEmbedding_toUniformOnFunIsCompact.uniformContinuous


theorem isUniformInducing_comp (g : C(β, δ)) (hg : IsUniformInducing g) :
    IsUniformInducing (ContinuousMap.comp g : C(α, β) → C(α, δ)) :=
  isUniformEmbedding_toUniformOnFunIsCompact.isUniformInducing.of_comp_iff.mp <|
    UniformOnFun.postcomp_isUniformInducing hg |>.comp
      isUniformEmbedding_toUniformOnFunIsCompact.isUniformInducing


@[deprecated (since := "2024-10-05")]
alias uniformInducing_comp := isUniformInducing_comp


theorem isUniformEmbedding_comp (g : C(β, δ)) (hg : IsUniformEmbedding g) :
    IsUniformEmbedding (ContinuousMap.comp g : C(α, β) → C(α, δ)) :=
  isUniformEmbedding_toUniformOnFunIsCompact.of_comp_iff.mp <|
    UniformOnFun.postcomp_isUniformEmbedding hg |>.comp
      isUniformEmbedding_toUniformOnFunIsCompact


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_comp := isUniformEmbedding_comp


theorem uniformContinuous_comp_left (g : C(α, γ)) :
    UniformContinuous (fun f ↦ f.comp g : C(γ, β) → C(α, β)) :=
  isUniformEmbedding_toUniformOnFunIsCompact.uniformContinuous_iff.mpr <|
    UniformOnFun.precomp_uniformContinuous (fun _ hK ↦ hK.image g.continuous) |>.comp
      isUniformEmbedding_toUniformOnFunIsCompact.uniformContinuous


/-- Any pair of a homeomorphism `X ≃ₜ Z` and an isomorphism `Y ≃ᵤ T` of uniform spaces gives rise
to an isomorphism `C(X, Y) ≃ᵤ C(Z, T)`. -/
protected def _root_.UniformEquiv.arrowCongr (φ : α ≃ₜ γ) (ψ : β ≃ᵤ δ) :
    C(α, β) ≃ᵤ C(γ, δ) where
  toFun f := .comp ψ.toHomeomorph <| f.comp φ.symm
  invFun f := .comp ψ.symm.toHomeomorph <| f.comp φ
  left_inv f := ext fun _ ↦ ψ.left_inv (f _) |>.trans <| congrArg f <| φ.left_inv _
  right_inv f := ext fun _ ↦ ψ.right_inv (f _) |>.trans <| congrArg f <| φ.right_inv _
  uniformContinuous_toFun := uniformContinuous_comp _ ψ.uniformContinuous |>.comp <|
    uniformContinuous_comp_left _
  uniformContinuous_invFun := uniformContinuous_comp _ ψ.symm.uniformContinuous |>.comp <|
    uniformContinuous_comp_left _


theorem hasBasis_compactConvergenceUniformity_of_compact :
    HasBasis (𝓤 C(α, β)) (fun V : Set (β × β) => V ∈ 𝓤 β) fun V =>
      { fg : C(α, β) × C(α, β) | ∀ x, (fg.1 x, fg.2 x) ∈ V } :=
  hasBasis_compactConvergenceUniformity.to_hasBasis
    (fun p hp => ⟨p.2, hp.2, fun _fg hfg x _hx => hfg x⟩) fun V hV =>
    ⟨⟨univ, V⟩, ⟨isCompact_univ, hV⟩, fun _fg hfg x => hfg x (mem_univ x)⟩


/-- Convergence in the compact-open topology is the same as uniform convergence for sequences of
continuous functions on a compact space. -/
theorem tendsto_iff_tendstoUniformly :
    Tendsto F p (𝓝 f) ↔ TendstoUniformly (fun i a => F i a) f p := by
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    inst✝ : CompactSpace α
    ⊢ Iff (Filter.Tendsto F p (nhds f)) (TendstoUniformly (fun i a => (F i) a) (⇑f …
  -/
  rw [tendsto_iff_forall_isCompact_tendstoUniformlyOn, ← tendstoUniformlyOn_univ]
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    f : ContinuousMap α β
    ι : Type u₃
    p : Filter ι
    F : ι → ContinuousMap α β
    inst✝ : CompactSpace α
    ⊢ Iff (∀ (K : Set α), IsCompact K → TendstoUniformlyOn (fun i a => (F i) a) (⇑ …
  -/
  exact ⟨fun h => h univ isCompact_univ, fun h K _hK => h.mono (subset_univ K)⟩
  /-
    🎉 no goals
  -/


theorem uniformSpace_eq_inf_precomp_of_cover {δ₁ δ₂ : Type*} [TopologicalSpace δ₁]
    [TopologicalSpace δ₂] (φ₁ : C(δ₁, α)) (φ₂ : C(δ₂, α)) (h_proper₁ : IsProperMap φ₁)
    (h_proper₂ : IsProperMap φ₂) (h_cover : range φ₁ ∪ range φ₂ = univ) :
    (inferInstanceAs <| UniformSpace C(α, β)) =
      .comap (comp · φ₁) inferInstance ⊓
      .comap (comp · φ₂) inferInstance := by
  -- We check the analogous result for `UniformOnFun` using
  -- `UniformOnFun.uniformSpace_eq_inf_precomp_of_cover`...
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  set 𝔖 : Set (Set α) := {K | IsCompact K}
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  set 𝔗₁ : Set (Set δ₁) := {K | IsCompact K}
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗₁ : Set (Set δ₁) := setOf fun K => IsCompact K
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  set 𝔗₂ : Set (Set δ₂) := {K | IsCompact K}
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗₁ : Set (Set δ₁) := setOf fun K => IsCompact K
    𝔗₂ : Set (Set δ₂) := setOf fun K => IsCompact K
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  have h_image₁ : MapsTo (φ₁ '' ·) 𝔗₁ 𝔖 := fun K hK ↦ hK.image φ₁.continuous
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗₁ : Set (Set δ₁) := setOf fun K => IsCompact K
    𝔗₂ : Set (Set δ₂) := setOf fun K => IsCompact K
    h_image₁ : Set.MapsTo (fun x => Set.image (⇑φ₁) x) 𝔗₁ 𝔖
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  have h_image₂ : MapsTo (φ₂ '' ·) 𝔗₂ 𝔖 := fun K hK ↦ hK.image φ₂.continuous
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗₁ : Set (Set δ₁) := setOf fun K => IsCompact K
    𝔗₂ : Set (Set δ₂) := setOf fun K => IsCompact K
    h_image₁ : Set.MapsTo (fun x => Set.image (⇑φ₁) x) 𝔗₁ 𝔖
    h_image₂ : Set.MapsTo (fun x => Set.image (⇑φ₂) x) 𝔗₂ 𝔖
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  have h_preimage₁ : MapsTo (φ₁ ⁻¹' ·) 𝔖 𝔗₁ := fun K ↦ h_proper₁.isCompact_preimage
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗₁ : Set (Set δ₁) := setOf fun K => IsCompact K
    𝔗₂ : Set (Set δ₂) := setOf fun K => IsCompact K
    h_image₁ : Set.MapsTo (fun x => Set.image (⇑φ₁) x) 𝔗₁ 𝔖
    h_image₂ : Set.MapsTo (fun x => Set.image (⇑φ₂) x) 𝔗₂ 𝔖
    h_preimage₁ : Set.MapsTo (fun x => Set.preimage (⇑φ₁) x) 𝔖 𝔗₁
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  have h_preimage₂ : MapsTo (φ₂ ⁻¹' ·) 𝔖 𝔗₂ := fun K ↦ h_proper₂.isCompact_preimage
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗₁ : Set (Set δ₁) := setOf fun K => IsCompact K
    𝔗₂ : Set (Set δ₂) := setOf fun K => IsCompact K
    h_image₁ : Set.MapsTo (fun x => Set.image (⇑φ₁) x) 𝔗₁ 𝔖
    h_image₂ : Set.MapsTo (fun x => Set.image (⇑φ₂) x) 𝔗₂ 𝔖
    h_preimage₁ : Set.MapsTo (fun x => Set.preimage (⇑φ₁) x) 𝔖 𝔗₁
    h_preimage₂ : Set.MapsTo (fun x => Set.preimage (⇑φ₂) x) 𝔖 𝔗₂
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  have h_cover' : ∀ S ∈ 𝔖, S ⊆ range φ₁ ∪ range φ₂ := fun S _ ↦ h_cover ▸ subset_univ _
  -- ... and we just pull it back.
  simp_rw +zetaDelta [compactConvergenceUniformSpace, replaceTopology_eq,
    UniformOnFun.uniformSpace_eq_inf_precomp_of_cover _ _ _ _ _
      h_image₁ h_image₂ h_preimage₁ h_preimage₂ h_cover',
    UniformSpace.comap_inf, ← UniformSpace.comap_comap]
  /-
    α : Type u₁
    β : Type u₂
    inst✝³ : TopologicalSpace α
    inst✝² : UniformSpace β
    δ₁ : Type u_1
    δ₂ : Type u_2
    inst✝¹ : TopologicalSpace δ₁
    inst✝ : TopologicalSpace δ₂
    φ₁ : ContinuousMap δ₁ α
    φ₂ : ContinuousMap δ₂ α
    h_proper₁ : IsProperMap ⇑φ₁
    h_proper₂ : IsProperMap ⇑φ₂
    h_cover : Eq (Union.union (Set.range ⇑φ₁) (Set.range ⇑φ₂)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗₁ : Set (Set δ₁) := setOf fun K => IsCompact K
    𝔗₂ : Set (Set δ₂) := setOf fun K => IsCompact K
    h_image₁ : Set.MapsTo (fun x => Set.image (⇑φ₁) x) 𝔗₁ 𝔖
    h_image₂ : Set.MapsTo (fun x => Set.image (⇑φ₂) x) 𝔗₂ 𝔖
    h_preimage₁ : Set.MapsTo (fun x => Set.preimage (⇑φ₁) x) 𝔖 𝔗₁
    h_preimage₂ : Set.MapsTo (fun x => Set.preimage (⇑φ₂) x) 𝔖 𝔗₂
    h_cover' : ∀ (S : Set α), Membership.mem 𝔖 S → HasSubset.Subset S (Union.union …
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (Min.min (UniformSpa …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem uniformSpace_eq_iInf_precomp_of_cover {δ : ι → Type*} [∀ i, TopologicalSpace (δ i)]
    (φ : Π i, C(δ i, α)) (h_proper : ∀ i, IsProperMap (φ i))
    (h_lf : LocallyFinite fun i ↦ range (φ i)) (h_cover : ⋃ i, range (φ i) = univ) :
    (inferInstanceAs <| UniformSpace C(α, β)) = ⨅ i, .comap (comp · (φ i)) inferInstance := by
  -- We check the analogous result for `UniformOnFun` using
  -- `UniformOnFun.uniformSpace_eq_iInf_precomp_of_cover`...
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    ι : Type u₃
    δ : ι → Type u_1
    inst✝ : (i : ι) → TopologicalSpace (δ i)
    φ : (i : ι) → ContinuousMap (δ i) α
    h_proper : ∀ (i : ι), IsProperMap ⇑(φ i)
    h_lf : LocallyFinite fun i => Set.range ⇑(φ i)
    h_cover : Eq (Set.iUnion fun i => Set.range ⇑(φ i)) Set.univ
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (iInf fun i => Unifo …
  -/
  set 𝔖 : Set (Set α) := {K | IsCompact K}
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    ι : Type u₃
    δ : ι → Type u_1
    inst✝ : (i : ι) → TopologicalSpace (δ i)
    φ : (i : ι) → ContinuousMap (δ i) α
    h_proper : ∀ (i : ι), IsProperMap ⇑(φ i)
    h_lf : LocallyFinite fun i => Set.range ⇑(φ i)
    h_cover : Eq (Set.iUnion fun i => Set.range ⇑(φ i)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (iInf fun i => Unifo …
  -/
  set 𝔗 : Π i, Set (Set (δ i)) := fun i ↦ {K | IsCompact K}
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    ι : Type u₃
    δ : ι → Type u_1
    inst✝ : (i : ι) → TopologicalSpace (δ i)
    φ : (i : ι) → ContinuousMap (δ i) α
    h_proper : ∀ (i : ι), IsProperMap ⇑(φ i)
    h_lf : LocallyFinite fun i => Set.range ⇑(φ i)
    h_cover : Eq (Set.iUnion fun i => Set.range ⇑(φ i)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗 : (i : ι) → Set (Set (δ i)) := fun i => setOf fun K => IsCompact K
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (iInf fun i => Unifo …
  -/
  have h_image : ∀ i, MapsTo (φ i '' ·) (𝔗 i) 𝔖 := fun i K hK ↦ hK.image (φ i).continuous
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    ι : Type u₃
    δ : ι → Type u_1
    inst✝ : (i : ι) → TopologicalSpace (δ i)
    φ : (i : ι) → ContinuousMap (δ i) α
    h_proper : ∀ (i : ι), IsProperMap ⇑(φ i)
    h_lf : LocallyFinite fun i => Set.range ⇑(φ i)
    h_cover : Eq (Set.iUnion fun i => Set.range ⇑(φ i)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗 : (i : ι) → Set (Set (δ i)) := fun i => setOf fun K => IsCompact K
    h_image : ∀ (i : ι), Set.MapsTo (fun x => Set.image (⇑(φ i)) x) (𝔗 i) 𝔖
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (iInf fun i => Unifo …
  -/
  have h_preimage : ∀ i, MapsTo (φ i ⁻¹' ·) 𝔖 (𝔗 i) := fun i K ↦ (h_proper i).isCompact_preimage
  have h_cover' : ∀ S ∈ 𝔖, ∃ I : Set ι, I.Finite ∧ S ⊆ ⋃ i ∈ I, range (φ i) := fun S hS ↦ by
    refine ⟨{i | (range (φ i) ∩ S).Nonempty}, h_lf.finite_nonempty_inter_compact hS,
      inter_eq_right.mp ?_⟩
    simp_rw [iUnion₂_inter, mem_setOf, iUnion_nonempty_self, ← iUnion_inter, h_cover, univ_inter]
  -- ... and we just pull it back.
  simp_rw +zetaDelta [compactConvergenceUniformSpace, replaceTopology_eq,
    UniformOnFun.uniformSpace_eq_iInf_precomp_of_cover _ _ _ h_image h_preimage h_cover',
    UniformSpace.comap_iInf, ← UniformSpace.comap_comap]
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    ι : Type u₃
    δ : ι → Type u_1
    inst✝ : (i : ι) → TopologicalSpace (δ i)
    φ : (i : ι) → ContinuousMap (δ i) α
    h_proper : ∀ (i : ι), IsProperMap ⇑(φ i)
    h_lf : LocallyFinite fun i => Set.range ⇑(φ i)
    h_cover : Eq (Set.iUnion fun i => Set.range ⇑(φ i)) Set.univ
    𝔖 : Set (Set α) := setOf fun K => IsCompact K
    𝔗 : (i : ι) → Set (Set (δ i)) := fun i => setOf fun K => IsCompact K
    h_image : ∀ (i : ι), Set.MapsTo (fun x => Set.image (⇑(φ i)) x) (𝔗 i) 𝔖
    h_preimage : ∀ (i : ι), Set.MapsTo (fun x => Set.preimage (⇑(φ i)) x) 𝔖 (𝔗 i)
    h_cover' : ∀ (S : Set α), Membership.mem 𝔖 S → Exists fun I => And I.Finite (H …
    ⊢ Eq (inferInstanceAs (UniformSpace (ContinuousMap α β))) (iInf fun i => iInf  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If the topology on `α` is generated by its restrictions to compact sets, then the space of
continuous maps `C(α, β)` is complete (wrt the compact convergence uniformity).

Sufficient conditions on `α` to satisfy this condition are (weak) local compactness (see
`ContinuousMap.instCompleteSpaceOfWeaklyLocallyCompactSpace`) and sequential compactness (see
`ContinuousMap.instCompleteSpaceOfSequentialSpace`). -/
lemma completeSpace_of_restrictGenTopology (h : RestrictGenTopology {K : Set α | IsCompact K}) :
    CompleteSpace C(α, β) := by
  rw [completeSpace_iff_isComplete_range
    isUniformEmbedding_toUniformOnFunIsCompact.isUniformInducing,
    range_toUniformOnFunIsCompact, ← completeSpace_coe_iff_isComplete]
  /-
    α : Type u₁
    β : Type u₂
    inst✝² : TopologicalSpace α
    inst✝¹ : UniformSpace β
    inst✝ : CompleteSpace β
    h : Topology.RestrictGenTopology (setOf fun K => IsCompact K)
    ⊢ CompleteSpace ↑(setOf fun f => Continuous f)
  -/
  exact (UniformOnFun.isClosed_setOf_continuous h).completeSpace_coe
  /-
    🎉 no goals
  -/


instance instCompleteSpaceOfWeaklyLocallyCompactSpace [WeaklyLocallyCompactSpace α] :
    CompleteSpace C(α, β) :=
  completeSpace_of_restrictGenTopology RestrictGenTopology.isCompact_of_weaklyLocallyCompact


instance instCompleteSpaceOfSequentialSpace [SequentialSpace α] :
    CompleteSpace C(α, β) :=
  completeSpace_of_restrictGenTopology RestrictGenTopology.isCompact_of_seq


