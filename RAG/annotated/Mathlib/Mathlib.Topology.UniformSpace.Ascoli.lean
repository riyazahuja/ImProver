/-- Let `X` be a compact topological space, `α` a uniform space, and `F : ι → (X → α)` an
equicontinuous family. Then, the uniform structures of uniform convergence and pointwise
convergence induce the same uniform structure on `ι`.

In other words, pointwise convergence and uniform convergence coincide on an equicontinuous
subset of `X → α`.

Consider using `Equicontinuous.isUniformInducing_uniformFun_iff_pi` and
`Equicontinuous.inducing_uniformFun_iff_pi` instead, to avoid rewriting instances. -/
theorem Equicontinuous.comap_uniformFun_eq [CompactSpace X] (F_eqcont : Equicontinuous F) :
    (UniformFun.uniformSpace X α).comap F =
    (Pi.uniformSpace _).comap F := by
  -- The `≤` inequality is trivial
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ⊢ Eq (UniformSpace.comap F (UniformFun.uniformSpace X α)) (UniformSpace.comap  …
  -/
  refine le_antisymm (UniformSpace.comap_mono UniformFun.uniformContinuous_toFun) ?_
  -- A bit of rewriting to get a nice intermediate statement.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ⊢ LE.le (UniformSpace.comap F (Pi.uniformSpace fun i => α)) (UniformSpace.coma …
  -/
  change comap _ _ ≤ comap _ _
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ⊢ LE.le (Filter.comap (fun p => { fst := F p.1, snd := F p.2 }) (uniformity (X …
  -/
  simp_rw [Pi.uniformity, Filter.comap_iInf, comap_comap, Function.comp_def]
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ⊢ LE.le (iInf fun i => Filter.comap (fun x => { fst := F x.1 i, snd := F x.2 i …
  -/
  refine ((UniformFun.hasBasis_uniformity X α).comap (Prod.map F F)).ge_iff.mpr ?_
  -- Core of the proof: we need to show that, for any entourage `U` in `α`,
  -- the set `𝐓(U) := {(i,j) : ι × ι | ∀ x : X, (F i x, F j x) ∈ U}` belongs to the filter
  -- `⨅ x, comap ((i,j) ↦ (F i x, F j x)) (𝓤 α)`.
  -- In other words, we have to show that it contains a finite intersection of
  -- sets of the form `𝐒(V, x) := {(i,j) : ι × ι | (F i x, F j x) ∈ V}` for some
  -- `x : X` and `V ∈ 𝓤 α`.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ⊢ ∀ (i' : Set (Prod α α)), Membership.mem (uniformity α) i' → Membership.mem ( …
  -/
  intro U hU
  -- We will do an `ε/3` argument, so we start by choosing a symmetric entourage `V ∈ 𝓤 α`
  -- such that `V ○ V ○ V ⊆ U`.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Membership.mem (iInf fun i => Filter.comap (fun x => { fst := F x.1 i, snd : …
  -/
  rcases comp_comp_symm_mem_uniformity_sets hU with ⟨V, hV, Vsymm, hVU⟩
  -- Set `Ω x := {y | ∀ i, (F i x, F i y) ∈ V}`. The equicontinuity of `F` guarantees that
  -- each `Ω x` is a neighborhood of `x`.
  /-
    case intro.intro.intro
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    Vsymm : SymmetricRel V
    hVU : HasSubset.Subset (compRel (compRel V V) V) U
    ⊢ Membership.mem (iInf fun i => Filter.comap (fun x => { fst := F x.1 i, snd : …
  -/
  let Ω x : Set X := {y | ∀ i, (F i x, F i y) ∈ V}
  -- Hence, by compactness of `X`, we can find some `A ⊆ X` finite such that the `Ω a`s for `a ∈ A`
  -- still cover `X`.
  /-
    case intro.intro.intro
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    Vsymm : SymmetricRel V
    hVU : HasSubset.Subset (compRel (compRel V V) V) U
    Ω : X → Set X := fun x => setOf fun y => ∀ (i : ι), Membership.mem V { fst :=  …
    ⊢ Membership.mem (iInf fun i => Filter.comap (fun x => { fst := F x.1 i, snd : …
  -/
  rcases CompactSpace.elim_nhds_subcover Ω (fun x ↦ F_eqcont x V hV) with ⟨A, Acover⟩
  -- We now claim that `⋂ a ∈ A, 𝐒(V, a) ⊆ 𝐓(U)`.
  have : (⋂ a ∈ A, {ij : ι × ι | (F ij.1 a, F ij.2 a) ∈ V}) ⊆
      (Prod.map F F) ⁻¹' UniformFun.gen X α U := by
    -- Given `(i, j) ∈ ⋂ a ∈ A, 𝐒(V, a)` and `x : X`, we have to prove that `(F i x, F j x) ∈ U`.
    rintro ⟨i, j⟩ hij x
    rw [mem_iInter₂] at hij
    -- We know that `x ∈ Ω a` for some `a ∈ A`, so that both `(F i x, F i a)` and `(F j a, F j x)`
    -- are in `V`.
    rcases mem_iUnion₂.mp (Acover.symm.subset <| mem_univ x) with ⟨a, ha, hax⟩
    -- Since `(i, j) ∈ 𝐒(V, a)` we also have `(F i a, F j a) ∈ V`, and finally we get
    -- `(F i x, F j x) ∈ V ○ V ○ V ⊆ U`.
    exact hVU (prod_mk_mem_compRel (prod_mk_mem_compRel
      (Vsymm.mk_mem_comm.mp (hax i)) (hij a ha)) (hax j))
  -- This completes the proof.
  exact mem_of_superset
    (A.iInter_mem_sets.mpr fun x _ ↦ mem_iInf_of_mem x <| preimage_mem_comap hV) this


/-- Let `X` be a compact topological space, `α` a uniform space, and `F : ι → (X → α)` an
equicontinuous family. Then, the uniform structures of uniform convergence and pointwise
convergence induce the same uniform structure on `ι`.

In other words, pointwise convergence and uniform convergence coincide on an equicontinuous
subset of `X → α`.

This is a version of `Equicontinuous.comap_uniformFun_eq` stated in terms of `IsUniformInducing`
for convenuence. -/
lemma Equicontinuous.isUniformInducing_uniformFun_iff_pi [UniformSpace ι] [CompactSpace X]
    (F_eqcont : Equicontinuous F) :
    IsUniformInducing (UniformFun.ofFun ∘ F) ↔ IsUniformInducing F := by
  rw [isUniformInducing_iff_uniformSpace, isUniformInducing_iff_uniformSpace,
      ← F_eqcont.comap_uniformFun_eq]
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : UniformSpace α
    F : ι → X → α
    inst✝¹ : UniformSpace ι
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ⊢ Iff (Eq (UniformSpace.comap (Function.comp (⇑UniformFun.ofFun) F) (UniformFu …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias Equicontinuous.uniformInducing_uniformFun_iff_pi :=
  Equicontinuous.isUniformInducing_uniformFun_iff_pi


/-- Let `X` be a compact topological space, `α` a uniform space, and `F : ι → (X → α)` an
equicontinuous family. Then, the topologies of uniform convergence and pointwise convergence induce
the same topology on `ι`.

In other words, pointwise convergence and uniform convergence coincide on an equicontinuous
subset of `X → α`.

This is a consequence of `Equicontinuous.comap_uniformFun_eq`, stated in terms of `IsInducing`
for convenuence. -/
lemma Equicontinuous.inducing_uniformFun_iff_pi [TopologicalSpace ι] [CompactSpace X]
    (F_eqcont : Equicontinuous F) :
    IsInducing (UniformFun.ofFun ∘ F) ↔ IsInducing F := by
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : UniformSpace α
    F : ι → X → α
    inst✝¹ : TopologicalSpace ι
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ⊢ Iff (Topology.IsInducing (Function.comp (⇑UniformFun.ofFun) F)) (Topology.Is …
  -/
  rw [isInducing_iff, isInducing_iff]
  change (_ = (UniformFun.uniformSpace X α |>.comap F |>.toTopologicalSpace)) ↔
         (_ = (Pi.uniformSpace _ |>.comap F |>.toTopologicalSpace))
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : UniformSpace α
    F : ι → X → α
    inst✝¹ : TopologicalSpace ι
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ⊢ Iff (Eq inst✝¹ UniformSpace.toTopologicalSpace) (Eq inst✝¹ UniformSpace.toTo …
  -/
  rw [F_eqcont.comap_uniformFun_eq]
  /-
    🎉 no goals
  -/


/-- Let `X` be a compact topological space, `α` a uniform space, `F : ι → (X → α)` an
equicontinuous family, and `ℱ` a filter on `ι`. Then, `F` tends *uniformly* to `f : X → α` along
`ℱ` iff it tends to `f` *pointwise* along `ℱ`. -/
theorem Equicontinuous.tendsto_uniformFun_iff_pi [CompactSpace X]
    (F_eqcont : Equicontinuous F) (ℱ : Filter ι) (f : X → α) :
    Tendsto (UniformFun.ofFun ∘ F) ℱ (𝓝 <| UniformFun.ofFun f) ↔
    Tendsto F ℱ (𝓝 f) := by
  -- Assume `ℱ` is non trivial.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ℱ : Filter ι
    f : X → α
    ⊢ Iff (Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) F) ℱ (nhds (UniformFu …
  -/
  rcases ℱ.eq_or_neBot with rfl | ℱ_ne
    /-
      case inl
      ι : Type u_1
      X : Type u_2
      α : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : UniformSpace α
      F : ι → X → α
      inst✝ : CompactSpace X
      F_eqcont : Equicontinuous F
      f : X → α
      ⊢ Iff (Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) F) Bot.bot (nhds (Uni …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : CompactSpace X
    F_eqcont : Equicontinuous F
    ℱ : Filter ι
    f : X → α
    ℱ_ne : ℱ.NeBot
    ⊢ Iff (Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) F) ℱ (nhds (UniformFu …
  -/
  constructor <;> intro H
  -- The forward direction is always true, the interesting part is the converse.
    /-
      case inr.mp
      ι : Type u_1
      X : Type u_2
      α : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : UniformSpace α
      F : ι → X → α
      inst✝ : CompactSpace X
      F_eqcont : Equicontinuous F
      ℱ : Filter ι
      f : X → α
      ℱ_ne : ℱ.NeBot
      H : Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) F) ℱ (nhds (UniformFun.o …
      ⊢ Filter.Tendsto F ℱ (nhds f)
    -/
  · exact UniformFun.uniformContinuous_toFun.continuous.tendsto _|>.comp H
    /-
      🎉 no goals
    -/
  -- To prove it, assume that `F` tends to `f` *pointwise* along `ℱ`.
    /-
      case inr.mpr
      ι : Type u_1
      X : Type u_2
      α : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : UniformSpace α
      F : ι → X → α
      inst✝ : CompactSpace X
      F_eqcont : Equicontinuous F
      ℱ : Filter ι
      f : X → α
      ℱ_ne : ℱ.NeBot
      H : Filter.Tendsto F ℱ (nhds f)
      ⊢ Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) F) ℱ (nhds (UniformFun.ofF …
    -/
  · set S : Set (X → α) := closure (range F)
    /-
      case inr.mpr
      ι : Type u_1
      X : Type u_2
      α : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : UniformSpace α
      F : ι → X → α
      inst✝ : CompactSpace X
      F_eqcont : Equicontinuous F
      ℱ : Filter ι
      f : X → α
      ℱ_ne : ℱ.NeBot
      H : Filter.Tendsto F ℱ (nhds f)
      S : Set (X → α) := closure (Set.range F)
      ⊢ Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) F) ℱ (nhds (UniformFun.ofF …
    -/
    set 𝒢 : Filter S := comap (↑) (map F ℱ)
    -- We would like to use `Equicontinuous.comap_uniformFun_eq`, but applying it to `F` is not
    -- enough since `f` has no reason to be in the range of `F`.
    -- Instead, we will apply it to the inclusion `(↑) : S → (X → α)` where `S` is the closure of
    -- the range of `F` *for the product topology*.
    -- We know that `S` is still equicontinuous...
    have hS : S.Equicontinuous := closure' (by rwa [equicontinuous_iff_range] at F_eqcont)
      continuous_id
    -- ... hence, as announced, the product topology and uniform convergence topology
    -- coincide on `S`.
    have ind : IsInducing (UniformFun.ofFun ∘ (↑) : S → X →ᵤ α) :=
      hS.inducing_uniformFun_iff_pi.mpr ⟨rfl⟩
    -- By construction, `f` is in `S`.
    /-
      case inr.mpr
      ι : Type u_1
      X : Type u_2
      α : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : UniformSpace α
      F : ι → X → α
      inst✝ : CompactSpace X
      F_eqcont : Equicontinuous F
      ℱ : Filter ι
      f : X → α
      ℱ_ne : ℱ.NeBot
      H : Filter.Tendsto F ℱ (nhds f)
      S : Set (X → α) := closure (Set.range F)
      𝒢 : Filter ↑S := Filter.comap Subtype.val (Filter.map F ℱ)
      hS : S.Equicontinuous
      ind : Topology.IsInducing (Function.comp (⇑UniformFun.ofFun) Subtype.val)
      ⊢ Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) F) ℱ (nhds (UniformFun.ofF …
    -/
    have f_mem : f ∈ S := mem_closure_of_tendsto H range_mem_map
    -- To conclude, we just have to translate our hypothesis and goal as statements about
    -- `S`, on which we know the two topologies at play coincide.
    -- For this, we define a filter on `S` by `𝒢 := comap (↑) (map F ℱ)`, and note that
    -- it satisfies `map (↑) 𝒢 = map F ℱ`. Thus, both our hypothesis and our goal
    -- can be rewritten as `𝒢 ≤ 𝓝 f`, where the neighborhood filter in the RHS corresponds
    -- to one of the two topologies at play on `S`. Since they coincide, we are done.
    have h𝒢ℱ : map (↑) 𝒢 = map F ℱ := Filter.map_comap_of_mem
      (Subtype.range_coe ▸ mem_of_superset range_mem_map subset_closure)
    have H' : Tendsto id 𝒢 (𝓝 ⟨f, f_mem⟩) := by
      rwa [tendsto_id', nhds_induced, ← map_le_iff_le_comap, h𝒢ℱ]
    /-
      case inr.mpr
      ι : Type u_1
      X : Type u_2
      α : Type u_3
      inst✝² : TopologicalSpace X
      inst✝¹ : UniformSpace α
      F : ι → X → α
      inst✝ : CompactSpace X
      F_eqcont : Equicontinuous F
      ℱ : Filter ι
      f : X → α
      ℱ_ne : ℱ.NeBot
      H : Filter.Tendsto F ℱ (nhds f)
      S : Set (X → α) := closure (Set.range F)
      𝒢 : Filter ↑S := Filter.comap Subtype.val (Filter.map F ℱ)
      hS : S.Equicontinuous
      ind : Topology.IsInducing (Function.comp (⇑UniformFun.ofFun) Subtype.val)
      f_mem : Membership.mem S f
      h𝒢ℱ : Eq (Filter.map Subtype.val 𝒢) (Filter.map F ℱ)
      H' : Filter.Tendsto id 𝒢 (nhds ⟨f, f_mem⟩)
      ⊢ Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) F) ℱ (nhds (UniformFun.ofF …
    -/
    rwa [ind.tendsto_nhds_iff, comp_id, ← tendsto_map'_iff, h𝒢ℱ] at H'
    /-
      🎉 no goals
    -/


/-- Let `X` be a topological space, `𝔖` a family of compact subsets of `X`, `α` a uniform space,
and `F : ι → (X → α)` a family which is equicontinuous on each `K ∈ 𝔖`. Then, the uniform
structures of uniform convergence on `𝔖` and pointwise convergence on `⋃₀ 𝔖` induce the same
uniform structure on `ι`.

In particular, pointwise convergence and compact convergence coincide on an equicontinuous
subset of `X → α`.

Consider using `EquicontinuousOn.isUniformInducing_uniformOnFun_iff_pi'` and
`EquicontinuousOn.inducing_uniformOnFun_iff_pi'` instead to avoid rewriting instances,
as well as their unprimed versions in case `𝔖` covers `X`. -/
theorem EquicontinuousOn.comap_uniformOnFun_eq {𝔖 : Set (Set X)} (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K) :
    (UniformOnFun.uniformSpace X α 𝔖).comap F =
    (Pi.uniformSpace _).comap ((⋃₀ 𝔖).restrict ∘ F) := by
  -- Recall that the uniform structure on `X →ᵤ[𝔖] α` is the one induced by all the maps
  -- `K.restrict : (X →ᵤ[𝔖] α) → (K →ᵤ α)` for `K ∈ 𝔖`. Its pullback along `F`, which is
  -- the LHS of our goal, is thus the uniform structure induced by the maps
  -- `K.restrict ∘ F : ι → (K →ᵤ α)` for `K ∈ 𝔖`.
  have H1 : (UniformOnFun.uniformSpace X α 𝔖).comap F =
      ⨅ (K ∈ 𝔖), (UniformFun.uniformSpace _ _).comap (K.restrict ∘ F) := by
    simp_rw [UniformOnFun.uniformSpace, UniformSpace.comap_iInf, ← UniformSpace.comap_comap,
      UniformFun.ofFun, Equiv.coe_fn_mk, UniformOnFun.toFun, UniformOnFun.ofFun, Function.comp_def,
      UniformFun, Equiv.coe_fn_symm_mk]
  -- Now, note that a similar fact is true for the uniform structure on `X → α` induced by
  -- the map `(⋃₀ 𝔖).restrict : (X → α) → ((⋃₀ 𝔖) → α)`: it is equal to the one induced by
  -- all maps `K.restrict : (X → α) → (K → α)` for `K ∈ 𝔖`, which means that the RHS of our
  -- goal is the uniform structure induced by the maps `K.restrict ∘ F : ι → (K → α)` for `K ∈ 𝔖`.
  have H2 : (Pi.uniformSpace _).comap ((⋃₀ 𝔖).restrict ∘ F) =
      ⨅ (K ∈ 𝔖), (Pi.uniformSpace _).comap (K.restrict ∘ F) := by
    simp_rw [UniformSpace.comap_comap, Pi.uniformSpace_comap_restrict_sUnion (fun _ ↦ α) 𝔖,
      UniformSpace.comap_iInf]
  -- But, for `K ∈ 𝔖` fixed, we know that the uniform structures of `K →ᵤ α` and `K → α`
  -- induce, via the equicontinuous family `K.restrict ∘ F`, the same uniform structure on `ι`.
  have H3 : ∀ K ∈ 𝔖, (UniformFun.uniformSpace K α).comap (K.restrict ∘ F) =
      (Pi.uniformSpace _).comap (K.restrict ∘ F) := fun K hK ↦ by
    have : CompactSpace K := isCompact_iff_compactSpace.mp (𝔖_compact K hK)
    exact (equicontinuous_restrict_iff _ |>.mpr <| F_eqcont K hK).comap_uniformFun_eq
  -- Combining these three facts completes the proof.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    F : ι → X → α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    H1 : Eq (UniformSpace.comap F (UniformOnFun.uniformSpace X α 𝔖)) (iInf fun K = …
    H2 : Eq (UniformSpace.comap (Function.comp 𝔖.sUnion.restrict F) (Pi.uniformSpa …
    H3 : ∀ (K : Set X), Membership.mem 𝔖 K → Eq (UniformSpace.comap (Function.comp …
    ⊢ Eq (UniformSpace.comap F (UniformOnFun.uniformSpace X α 𝔖)) (UniformSpace.co …
  -/
  simp_rw [H1, H2, iInf_congr fun K ↦ iInf_congr fun hK ↦ H3 K hK]
  /-
    🎉 no goals
  -/


/-- Let `X` be a topological space, `𝔖` a family of compact subsets of `X`, `α` a uniform space,
and `F : ι → (X → α)` a family which is equicontinuous on each `K ∈ 𝔖`. Then, the uniform
structures of uniform convergence on `𝔖` and pointwise convergence on `⋃₀ 𝔖` induce the same
uniform structure on `ι`.

In particular, pointwise convergence and compact convergence coincide on an equicontinuous
subset of `X → α`.

This is a version of `EquicontinuousOn.comap_uniformOnFun_eq` stated in terms of `IsUniformInducing`
for convenuence. -/
lemma EquicontinuousOn.isUniformInducing_uniformOnFun_iff_pi' [UniformSpace ι]
    {𝔖 : Set (Set X)} (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K) :
    IsUniformInducing (UniformOnFun.ofFun 𝔖 ∘ F) ↔
    IsUniformInducing ((⋃₀ 𝔖).restrict ∘ F) := by
  rw [isUniformInducing_iff_uniformSpace, isUniformInducing_iff_uniformSpace,
      ← EquicontinuousOn.comap_uniformOnFun_eq 𝔖_compact F_eqcont]
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : UniformSpace ι
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ⊢ Iff (Eq (UniformSpace.comap (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F) (Uni …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias EquicontinuousOn.uniformInducing_uniformOnFun_iff_pi' :=
  EquicontinuousOn.isUniformInducing_uniformOnFun_iff_pi'


/-- Let `X` be a topological space, `𝔖` a covering of `X` by compact subsets, `α` a uniform space,
and `F : ι → (X → α)` a family which is equicontinuous on each `K ∈ 𝔖`. Then, the uniform
structures of uniform convergence on `𝔖` and pointwise convergence induce the same
uniform structure on `ι`.

This is a specialization of `EquicontinuousOn.isUniformInducing_uniformOnFun_iff_pi'` to
the case where `𝔖` covers `X`. -/
lemma EquicontinuousOn.isUniformInducing_uniformOnFun_iff_pi [UniformSpace ι]
    {𝔖 : Set (Set X)} (𝔖_covers : ⋃₀ 𝔖 = univ) (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K) :
    IsUniformInducing (UniformOnFun.ofFun 𝔖 ∘ F) ↔
    IsUniformInducing F := by
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : UniformSpace ι
    𝔖 : Set (Set X)
    𝔖_covers : Eq 𝔖.sUnion Set.univ
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ⊢ Iff (IsUniformInducing (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F)) (IsUnifo …
  -/
  rw [eq_univ_iff_forall] at 𝔖_covers
  -- This obviously follows from the previous lemma, we formalize it by going through the
  -- isomorphism of uniform spaces between `(⋃₀ 𝔖) → α` and `X → α`.
  let φ : ((⋃₀ 𝔖) → α) ≃ᵤ (X → α) := UniformEquiv.piCongrLeft (β := fun _ ↦ α)
    (Equiv.subtypeUnivEquiv 𝔖_covers)
  rw [EquicontinuousOn.isUniformInducing_uniformOnFun_iff_pi' 𝔖_compact F_eqcont,
      show restrict (⋃₀ 𝔖) ∘ F = φ.symm ∘ F by rfl]
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : UniformSpace ι
    𝔖 : Set (Set X)
    𝔖_covers : ∀ (x : X), Membership.mem 𝔖.sUnion x
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    φ : UniformEquiv (↑𝔖.sUnion → α) (X → α) := UniformEquiv.piCongrLeft (Equiv.su …
    ⊢ Iff (IsUniformInducing (Function.comp (⇑φ.symm) F)) (IsUniformInducing F)
  -/
  exact ⟨fun H ↦ φ.isUniformInducing.comp H, fun H ↦ φ.symm.isUniformInducing.comp H⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias EquicontinuousOn.uniformInducing_uniformOnFun_iff_pi :=
  EquicontinuousOn.isUniformInducing_uniformOnFun_iff_pi


/-- Let `X` be a topological space, `𝔖` a family of compact subsets of `X`, `α` a uniform space,
and `F : ι → (X → α)` a family which is equicontinuous on each `K ∈ 𝔖`. Then, the topologies
of uniform convergence on `𝔖` and pointwise convergence on `⋃₀ 𝔖` induce the same topology on  `ι`.

In particular, pointwise convergence and compact convergence coincide on an equicontinuous
subset of `X → α`.

This is a consequence of `EquicontinuousOn.comap_uniformOnFun_eq` stated in terms of `IsInducing`
for convenience. -/
lemma EquicontinuousOn.inducing_uniformOnFun_iff_pi' [TopologicalSpace ι]
    {𝔖 : Set (Set X)} (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K) :
    IsInducing (UniformOnFun.ofFun 𝔖 ∘ F) ↔
    IsInducing ((⋃₀ 𝔖).restrict ∘ F) := by
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : TopologicalSpace ι
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ⊢ Iff (Topology.IsInducing (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F)) (Topol …
  -/
  rw [isInducing_iff, isInducing_iff]
  change (_ = ((UniformOnFun.uniformSpace X α 𝔖).comap F).toTopologicalSpace) ↔
    (_ = ((Pi.uniformSpace _).comap ((⋃₀ 𝔖).restrict ∘ F)).toTopologicalSpace)
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : TopologicalSpace ι
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ⊢ Iff (Eq inst✝ UniformSpace.toTopologicalSpace) (Eq inst✝ UniformSpace.toTopo …
  -/
  rw [← EquicontinuousOn.comap_uniformOnFun_eq 𝔖_compact F_eqcont]
  /-
    🎉 no goals
  -/


/-- Let `X` be a topological space, `𝔖` a covering of `X` by compact subsets, `α` a uniform space,
and `F : ι → (X → α)` a family which is equicontinuous on each `K ∈ 𝔖`. Then, the topologies
of uniform convergence on `𝔖` and pointwise convergence induce the same topology on `ι`.

This is a specialization of `EquicontinuousOn.inducing_uniformOnFun_iff_pi'` to
the case where `𝔖` covers `X`. -/
lemma EquicontinuousOn.isInducing_uniformOnFun_iff_pi [TopologicalSpace ι]
    {𝔖 : Set (Set X)} (𝔖_covers : ⋃₀ 𝔖 = univ) (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K) :
    IsInducing (UniformOnFun.ofFun 𝔖 ∘ F) ↔
    IsInducing F := by
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : TopologicalSpace ι
    𝔖 : Set (Set X)
    𝔖_covers : Eq 𝔖.sUnion Set.univ
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ⊢ Iff (Topology.IsInducing (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F)) (Topol …
  -/
  rw [eq_univ_iff_forall] at 𝔖_covers
  -- This obviously follows from the previous lemma, we formalize it by going through the
  -- homeomorphism between `(⋃₀ 𝔖) → α` and `X → α`.
  let φ : ((⋃₀ 𝔖) → α) ≃ₜ (X → α) := Homeomorph.piCongrLeft (Y := fun _ ↦ α)
    (Equiv.subtypeUnivEquiv 𝔖_covers)
  rw [EquicontinuousOn.inducing_uniformOnFun_iff_pi' 𝔖_compact F_eqcont,
      show restrict (⋃₀ 𝔖) ∘ F = φ.symm ∘ F by rfl]
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : TopologicalSpace ι
    𝔖 : Set (Set X)
    𝔖_covers : ∀ (x : X), Membership.mem 𝔖.sUnion x
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    φ : Homeomorph (↑𝔖.sUnion → α) (X → α) := Homeomorph.piCongrLeft (Equiv.subtyp …
    ⊢ Iff (Topology.IsInducing (Function.comp (⇑φ.symm) F)) (Topology.IsInducing F)
  -/
  exact ⟨fun H ↦ φ.isInducing.comp H, fun H ↦ φ.symm.isInducing.comp H⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias EquicontinuousOn.inducing_uniformOnFun_iff_pi :=
  EquicontinuousOn.isInducing_uniformOnFun_iff_pi

-- TODO: find a way to factor common elements of this proof and the proof of
-- `EquicontinuousOn.comap_uniformOnFun_eq`

/-- Let `X` be a topological space, `𝔖` a family of compact subsets of `X`,
`α` a uniform space, `F : ι → (X → α)` a family equicontinuous on each `K ∈ 𝔖`, and `ℱ` a filter
on `ι`. Then, `F` tends to `f : X → α` along `ℱ` *uniformly on each `K ∈ 𝔖`* iff it tends to `f`
*pointwise on `⋃₀ 𝔖`* along `ℱ`. -/
theorem EquicontinuousOn.tendsto_uniformOnFun_iff_pi'
    {𝔖 : Set (Set X)} (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K) (ℱ : Filter ι) (f : X → α) :
    Tendsto (UniformOnFun.ofFun 𝔖 ∘ F) ℱ (𝓝 <| UniformOnFun.ofFun 𝔖 f) ↔
    Tendsto ((⋃₀ 𝔖).restrict ∘ F) ℱ (𝓝 <| (⋃₀ 𝔖).restrict f) := by
  -- Recall that the uniform structure on `X →ᵤ[𝔖] α` is the one induced by all the maps
  -- `K.restrict : (X →ᵤ[𝔖] α) → (K →ᵤ α)` for `K ∈ 𝔖`.
  -- Similarly, the uniform structure on `X → α` induced by the map
  -- `(⋃₀ 𝔖).restrict : (X → α) → ((⋃₀ 𝔖) → α)` is equal to the one induced by
  -- all maps `K.restrict : (X → α) → (K → α)` for `K ∈ 𝔖`
  -- Thus, we just have to compare the two sides of our goal when restricted to some
  -- `K ∈ 𝔖`, where we can apply `Equicontinuous.tendsto_uniformFun_iff_pi`.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    F : ι → X → α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ℱ : Filter ι
    f : X → α
    ⊢ Iff (Filter.Tendsto (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F) ℱ (nhds ((Un …
  -/
  rw [← Filter.tendsto_comap_iff (g := (⋃₀ 𝔖).restrict), ← nhds_induced]
  simp_rw [UniformOnFun.topologicalSpace_eq, Pi.induced_restrict_sUnion 𝔖 (π := fun _ ↦ α),
    _root_.nhds_iInf, nhds_induced, tendsto_iInf, tendsto_comap_iff]
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    F : ι → X → α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ℱ : Filter ι
    f : X → α
    ⊢ Iff (∀ (i : Set X), Membership.mem 𝔖 i → Filter.Tendsto (Function.comp (Func …
  -/
  congrm ∀ K (hK : K ∈ 𝔖), ?_
  /-
    case a
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    F : ι → X → α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ℱ : Filter ι
    f : X → α
    K : Set X
    hK : Membership.mem 𝔖 K
    ⊢ Iff (Filter.Tendsto (Function.comp (Function.comp (⇑UniformFun.ofFun) (Funct …
  -/
  have : CompactSpace K := isCompact_iff_compactSpace.mp (𝔖_compact K hK)
  /-
    case a
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    F : ι → X → α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ℱ : Filter ι
    f : X → α
    K : Set X
    hK : Membership.mem 𝔖 K
    this : CompactSpace ↑K
    ⊢ Iff (Filter.Tendsto (Function.comp (Function.comp (⇑UniformFun.ofFun) (Funct …
  -/
  rw [← (equicontinuous_restrict_iff _ |>.mpr <| F_eqcont K hK).tendsto_uniformFun_iff_pi]
  /-
    case a
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    F : ι → X → α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ℱ : Filter ι
    f : X → α
    K : Set X
    hK : Membership.mem 𝔖 K
    this : CompactSpace ↑K
    ⊢ Iff (Filter.Tendsto (Function.comp (Function.comp (⇑UniformFun.ofFun) (Funct …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Let `X` be a topological space, `𝔖` a covering of `X` by compact subsets,
`α` a uniform space, `F : ι → (X → α)` a family equicontinuous on each `K ∈ 𝔖`, and `ℱ` a filter
on `ι`. Then, `F` tends to `f : X → α` along `ℱ` *uniformly on each `K ∈ 𝔖`* iff it tends to `f`
*pointwise* along `ℱ`.

This is a specialization of `EquicontinuousOn.tendsto_uniformOnFun_iff_pi'` to the case
where `𝔖` covers `X`. -/
theorem EquicontinuousOn.tendsto_uniformOnFun_iff_pi
    {𝔖 : Set (Set X)} (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K) (𝔖_covers : ⋃₀ 𝔖 = univ)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K) (ℱ : Filter ι) (f : X → α) :
    Tendsto (UniformOnFun.ofFun 𝔖 ∘ F) ℱ (𝓝 <| UniformOnFun.ofFun 𝔖 f) ↔
    Tendsto F ℱ (𝓝 f) := by
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    F : ι → X → α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    𝔖_covers : Eq 𝔖.sUnion Set.univ
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    ℱ : Filter ι
    f : X → α
    ⊢ Iff (Filter.Tendsto (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F) ℱ (nhds ((Un …
  -/
  rw [eq_univ_iff_forall] at 𝔖_covers
  let φ : ((⋃₀ 𝔖) → α) ≃ₜ (X → α) := Homeomorph.piCongrLeft (Y := fun _ ↦ α)
    (Equiv.subtypeUnivEquiv 𝔖_covers)
  rw [EquicontinuousOn.tendsto_uniformOnFun_iff_pi' 𝔖_compact F_eqcont,
      show restrict (⋃₀ 𝔖) ∘ F = φ.symm ∘ F by rfl, show restrict (⋃₀ 𝔖) f = φ.symm f by rfl,
      φ.symm.isInducing.tendsto_nhds_iff]


/-- Let `X` be a topological space, `𝔖` a family of compact subsets of `X` and
`α` a uniform space. An equicontinuous subset of `X → α` is closed in the topology of uniform
convergence on all `K ∈ 𝔖` iff it is closed in the topology of pointwise convergence on `⋃₀ 𝔖`. -/
theorem EquicontinuousOn.isClosed_range_pi_of_uniformOnFun'
    {𝔖 : Set (Set X)} (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K)
    (H : IsClosed (range <| UniformOnFun.ofFun 𝔖 ∘ F)) :
    IsClosed (range <| (⋃₀ 𝔖).restrict ∘ F) := by
  -- Do we have no equivalent of `nontriviality`?
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    F : ι → X → α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    H : IsClosed (Set.range (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F))
    ⊢ IsClosed (Set.range (Function.comp 𝔖.sUnion.restrict F))
  -/
  rcases isEmpty_or_nonempty α with _ | _
    /-
      case inl
      ι : Type u_1
      X : Type u_2
      α : Type u_3
      inst✝¹ : TopologicalSpace X
      inst✝ : UniformSpace α
      F : ι → X → α
      𝔖 : Set (Set X)
      𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
      F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
      H : IsClosed (Set.range (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F))
      h✝ : IsEmpty α
      ⊢ IsClosed (Set.range (Function.comp 𝔖.sUnion.restrict F))
    -/
  · simp [isClosed_discrete]
    /-
      🎉 no goals
    -/
  -- This follows from the previous lemmas and the characterization of the closure using filters.
  simp_rw [isClosed_iff_clusterPt, ← Filter.map_top, ← mapClusterPt_def,
    mapClusterPt_iff_ultrafilter, range_comp, Subtype.coe_injective.surjective_comp_right.forall,
    ← restrict_eq, ← EquicontinuousOn.tendsto_uniformOnFun_iff_pi' 𝔖_compact F_eqcont]
  exact fun f ⟨u, _, hu⟩ ↦ mem_image_of_mem _ <| H.mem_of_tendsto hu <|
    Eventually.of_forall mem_range_self


/-- Let `X` be a topological space, `𝔖` a covering of `X` by compact subsets, and
`α` a uniform space. An equicontinuous subset of `X → α` is closed in the topology of uniform
convergence on all `K ∈ 𝔖` iff it is closed in the topology of pointwise convergence.

This is a specialization of `EquicontinuousOn.isClosed_range_pi_of_uniformOnFun'` to the case where
`𝔖` covers `X`. -/
theorem EquicontinuousOn.isClosed_range_uniformOnFun_iff_pi
    {𝔖 : Set (Set X)} (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K) (𝔖_covers : ⋃₀ 𝔖 = univ)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K) :
    IsClosed (range <| UniformOnFun.ofFun 𝔖 ∘ F) ↔
    IsClosed (range F) := by
  -- This follows from the previous lemmas and the characterization of the closure using filters.
  simp_rw [isClosed_iff_clusterPt, ← Filter.map_top, ← mapClusterPt_def,
    mapClusterPt_iff_ultrafilter, range_comp, (UniformOnFun.ofFun 𝔖).surjective.forall,
    ← EquicontinuousOn.tendsto_uniformOnFun_iff_pi 𝔖_compact 𝔖_covers F_eqcont,
    (UniformOnFun.ofFun 𝔖).injective.mem_set_image]


alias ⟨EquicontinuousOn.isClosed_range_pi_of_uniformOnFun, _⟩ :=
  EquicontinuousOn.isClosed_range_uniformOnFun_iff_pi


/-- A version of the **Arzela-Ascoli theorem**.

Let `X` be a topological space, `𝔖` a family of compact subsets of `X`, `α` a uniform space,
and `F : ι → (X → α)`. Assume that:
* `F`, viewed as a function `ι → (X →ᵤ[𝔖] α)`, is closed and inducing
* `F` is equicontinuous on each `K ∈ 𝔖`
* For all `x ∈ ⋃₀ 𝔖`, the range of `i ↦ F i x` is contained in some fixed compact subset.

Then `ι` is compact. -/
theorem ArzelaAscoli.compactSpace_of_closed_inducing' [TopologicalSpace ι] {𝔖 : Set (Set X)}
    (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K) (F_ind : IsInducing (UniformOnFun.ofFun 𝔖 ∘ F))
    (F_cl : IsClosed <| range <| UniformOnFun.ofFun 𝔖 ∘ F)
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K)
    (F_pointwiseCompact : ∀ K ∈ 𝔖, ∀ x ∈ K, ∃ Q, IsCompact Q ∧ ∀ i, F i x ∈ Q) :
    CompactSpace ι := by
  -- By equicontinuity, we know that the topology on `ι` is also the one induced by
  -- `restrict (⋃₀ 𝔖) ∘ F`.
  have : IsInducing (restrict (⋃₀ 𝔖) ∘ F) := by
    rwa [EquicontinuousOn.inducing_uniformOnFun_iff_pi' 𝔖_compact F_eqcont] at F_ind
  -- Thus, we just have to check that the range of this map is compact.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : TopologicalSpace ι
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_ind : Topology.IsInducing (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F)
    F_cl : IsClosed (Set.range (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F))
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    F_pointwiseCompact : ∀ (K : Set X), Membership.mem 𝔖 K → ∀ (x : X), Membership …
    this : Topology.IsInducing (Function.comp 𝔖.sUnion.restrict F)
    ⊢ CompactSpace ι
  -/
  rw [← isCompact_univ_iff, this.isCompact_iff, image_univ]
  -- But then we are working in a product space, where compactness can easily be proven using
  -- Tykhonov's theorem! More precisely, for each `x ∈ ⋃₀ 𝔖`, choose a compact set `Q x`
  -- containing all `F i x`s.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : TopologicalSpace ι
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_ind : Topology.IsInducing (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F)
    F_cl : IsClosed (Set.range (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F))
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    F_pointwiseCompact : ∀ (K : Set X), Membership.mem 𝔖 K → ∀ (x : X), Membership …
    this : Topology.IsInducing (Function.comp 𝔖.sUnion.restrict F)
    ⊢ IsCompact (Set.range (Function.comp 𝔖.sUnion.restrict F))
  -/
  rw [← forall_sUnion] at F_pointwiseCompact
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace α
    F : ι → X → α
    inst✝ : TopologicalSpace ι
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_ind : Topology.IsInducing (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F)
    F_cl : IsClosed (Set.range (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F))
    F_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn F K
    F_pointwiseCompact : ∀ (x : X), Membership.mem 𝔖.sUnion x → Exists fun Q => An …
    this : Topology.IsInducing (Function.comp 𝔖.sUnion.restrict F)
    ⊢ IsCompact (Set.range (Function.comp 𝔖.sUnion.restrict F))
  -/
  choose! Q Q_compact F_in_Q using F_pointwiseCompact
  -- Notice that, since the range of `F` is closed in `X →ᵤ[𝔖] α`, equicontinuity ensures that
  -- the range of `(⋃₀ 𝔖).restrict ∘ F` is still closed in the product topology.
  -- But it's contained in the product of the `Q x`s, which is compact by Tykhonov, hence
  -- it is compact as well.
  refine IsCompact.of_isClosed_subset (isCompact_univ_pi fun x ↦ Q_compact x x.2)
    (EquicontinuousOn.isClosed_range_pi_of_uniformOnFun' 𝔖_compact F_eqcont F_cl)
    (range_subset_iff.mpr fun i x _ ↦ F_in_Q x x.2 i)


/-- A version of the **Arzela-Ascoli theorem**.

Let `X, ι` be topological spaces, `𝔖` a covering of `X` by compact subsets, `α` a uniform space,
and `F : ι → (X → α)`. Assume that:
* `F`, viewed as a function `ι → (X →ᵤ[𝔖] α)`, is a closed embedding (in other words, `ι`
  identifies to a closed subset of `X →ᵤ[𝔖] α` through `F`)
* `F` is equicontinuous on each `K ∈ 𝔖`
* For all `x`, the range of `i ↦ F i x` is contained in some fixed compact subset.

Then `ι` is compact. -/
theorem ArzelaAscoli.compactSpace_of_isClosedEmbedding [TopologicalSpace ι] {𝔖 : Set (Set X)}
    (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K) (F_clemb : IsClosedEmbedding (UniformOnFun.ofFun 𝔖 ∘ F))
    (F_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn F K)
    (F_pointwiseCompact : ∀ K ∈ 𝔖, ∀ x ∈ K, ∃ Q, IsCompact Q ∧ ∀ i, F i x ∈ Q) :
    CompactSpace ι :=
  compactSpace_of_closed_inducing' 𝔖_compact F_clemb.isInducing F_clemb.isClosed_range
    F_eqcont F_pointwiseCompact


@[deprecated (since := "2024-10-20")]
alias ArzelaAscoli.compactSpace_of_closedEmbedding := ArzelaAscoli.compactSpace_of_isClosedEmbedding


/-- A version of the **Arzela-Ascoli theorem**.

Let `X, ι` be topological spaces, `𝔖` a covering of `X` by compact subsets, `α` a T2 uniform space,
`F : ι → (X → α)`, and `s` a subset of `ι`. Assume that:
* `F`, viewed as a function `ι → (X →ᵤ[𝔖] α)`, is a closed embedding (in other words, `ι`
  identifies to a closed subset of `X →ᵤ[𝔖] α` through `F`)
* `F '' s` is equicontinuous on each `K ∈ 𝔖`
* For all `x ∈ ⋃₀ 𝔖`, the image of `s` under `i ↦ F i x` is contained in some fixed compact subset.

Then `s` has compact closure in `ι`. -/
theorem ArzelaAscoli.isCompact_closure_of_isClosedEmbedding [TopologicalSpace ι] [T2Space α]
    {𝔖 : Set (Set X)} (𝔖_compact : ∀ K ∈ 𝔖, IsCompact K)
    (F_clemb : IsClosedEmbedding (UniformOnFun.ofFun 𝔖 ∘ F))
    {s : Set ι} (s_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn (F ∘ ((↑) : s → ι)) K)
    (s_pointwiseCompact : ∀ K ∈ 𝔖, ∀ x ∈ K, ∃ Q, IsCompact Q ∧ ∀ i ∈ s, F i x ∈ Q) :
    IsCompact (closure s) := by
  -- We apply `ArzelaAscoli.compactSpace_of_isClosedEmbedding` to the map
  -- `F ∘ (↑) : closure s → (X → α)`, for which all the hypotheses are easily verified.
  /-
    ι : Type u_1
    X : Type u_2
    α : Type u_3
    inst✝³ : TopologicalSpace X
    inst✝² : UniformSpace α
    F : ι → X → α
    inst✝¹ : TopologicalSpace ι
    inst✝ : T2Space α
    𝔖 : Set (Set X)
    𝔖_compact : ∀ (K : Set X), Membership.mem 𝔖 K → IsCompact K
    F_clemb : Topology.IsClosedEmbedding (Function.comp (⇑(UniformOnFun.ofFun 𝔖)) F)
    s : Set ι
    s_eqcont : ∀ (K : Set X), Membership.mem 𝔖 K → EquicontinuousOn (Function.comp …
    s_pointwiseCompact : ∀ (K : Set X), Membership.mem 𝔖 K → ∀ (x : X), Membership …
    ⊢ IsCompact (closure s)
  -/
  rw [isCompact_iff_compactSpace]
  have : ∀ K ∈ 𝔖, ∀ x ∈ K, Continuous (eval x ∘ F) := fun K hK x hx ↦
    UniformOnFun.uniformContinuous_eval_of_mem _ _ hx hK |>.continuous.comp F_clemb.continuous
  have cls_eqcont : ∀ K ∈ 𝔖, EquicontinuousOn (F ∘ ((↑) : closure s → ι)) K :=
    fun K hK ↦ (s_eqcont K hK).closure' <| show Continuous (K.restrict ∘ F) from
      continuous_pi fun ⟨x, hx⟩ ↦ this K hK x hx
  have cls_pointwiseCompact : ∀ K ∈ 𝔖, ∀ x ∈ K, ∃ Q, IsCompact Q ∧ ∀ i ∈ closure s, F i x ∈ Q :=
    fun K hK x hx ↦ (s_pointwiseCompact K hK x hx).imp fun Q hQ ↦ ⟨hQ.1, closure_minimal hQ.2 <|
      hQ.1.isClosed.preimage (this K hK x hx)⟩
  exact ArzelaAscoli.compactSpace_of_isClosedEmbedding 𝔖_compact
    (F_clemb.comp isClosed_closure.isClosedEmbedding_subtypeVal) cls_eqcont
    fun K hK x hx ↦ (cls_pointwiseCompact K hK x hx).imp fun Q hQ ↦ ⟨hQ.1, by simpa using hQ.2⟩


@[deprecated (since := "2024-10-20")]
alias ArzelaAscoli.isCompact_closure_of_closedEmbedding :=
  ArzelaAscoli.isCompact_closure_of_isClosedEmbedding


/-- A version of the **Arzela-Ascoli theorem**.

If an equicontinuous family of continuous functions is compact in the pointwise topology, then it
is compact in the compact open topology. -/
theorem ArzelaAscoli.isCompact_of_equicontinuous
    (S : Set C(X, α)) (hS1 : IsCompact (ContinuousMap.toFun '' S))
    (hS2 : Equicontinuous ((↑) : S → X → α)) : IsCompact S := by
  suffices h : IsInducing (Equiv.Set.image _ S DFunLike.coe_injective) by
    rw [isCompact_iff_compactSpace] at hS1 ⊢
    exact (Equiv.toHomeomorphOfIsInducing _ h).symm.compactSpace
  /-
    X : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : UniformSpace α
    S : Set (ContinuousMap X α)
    hS1 : IsCompact (Set.image ContinuousMap.toFun S)
    hS2 : Equicontinuous fun x => ⇑↑x
    ⊢ Topology.IsInducing ⇑(Equiv.Set.image (fun f => ⇑f) S ⋯)
  -/
  rw [← IsInducing.subtypeVal.of_comp_iff, ← EquicontinuousOn.isInducing_uniformOnFun_iff_pi _ _ _]
    /-
      X : Type u_2
      α : Type u_3
      inst✝¹ : TopologicalSpace X
      inst✝ : UniformSpace α
      S : Set (ContinuousMap X α)
      hS1 : IsCompact (Set.image ContinuousMap.toFun S)
      hS2 : Equicontinuous fun x => ⇑↑x
      ⊢ Topology.IsInducing (Function.comp (⇑(UniformOnFun.ofFun ?m.47233)) (Functio …
    -/
  · exact ContinuousMap.isUniformEmbedding_toUniformOnFunIsCompact.isInducing.comp .subtypeVal
    /-
      🎉 no goals
    -/
    /-
      X : Type u_2
      α : Type u_3
      inst✝¹ : TopologicalSpace X
      inst✝ : UniformSpace α
      S : Set (ContinuousMap X α)
      hS1 : IsCompact (Set.image ContinuousMap.toFun S)
      hS2 : Equicontinuous fun x => ⇑↑x
      ⊢ Eq (setOf fun K => IsCompact K).sUnion Set.univ
    -/
  · exact eq_univ_iff_forall.mpr (fun x ↦ mem_sUnion_of_mem (mem_singleton x) isCompact_singleton)
    /-
      🎉 no goals
    -/
    /-
      X : Type u_2
      α : Type u_3
      inst✝¹ : TopologicalSpace X
      inst✝ : UniformSpace α
      S : Set (ContinuousMap X α)
      hS1 : IsCompact (Set.image ContinuousMap.toFun S)
      hS2 : Equicontinuous fun x => ⇑↑x
      ⊢ ∀ (K : Set X), Membership.mem (setOf fun K => IsCompact K) K → IsCompact K
    -/
  · exact fun _ ↦ id
    /-
      🎉 no goals
    -/
    /-
      X : Type u_2
      α : Type u_3
      inst✝¹ : TopologicalSpace X
      inst✝ : UniformSpace α
      S : Set (ContinuousMap X α)
      hS1 : IsCompact (Set.image ContinuousMap.toFun S)
      hS2 : Equicontinuous fun x => ⇑↑x
      ⊢ ∀ (K : Set X), Membership.mem (setOf fun K => IsCompact K) K → Equicontinuou …
    -/
  · exact fun K _ ↦ hS2.equicontinuousOn K
    /-
      🎉 no goals
    -/

