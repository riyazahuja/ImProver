/-- A topological space is **locally connected** if each neighborhood filter admits a basis
of connected *open* sets. Note that it is equivalent to each point having a basis of connected
(non necessarily open) sets but in a non-trivial way, so we choose this definition and prove the
equivalence later in `locallyConnectedSpace_iff_connected_basis`. -/
class LocallyConnectedSpace (α : Type*) [TopologicalSpace α] : Prop where
  /-- Open connected neighborhoods form a basis of the neighborhoods filter. -/
  open_connected_basis : ∀ x, (𝓝 x).HasBasis (fun s : Set α => IsOpen s ∧ x ∈ s ∧ IsConnected s) id


theorem locallyConnectedSpace_iff_hasBasis_isOpen_isConnected :
    LocallyConnectedSpace α ↔
      ∀ x, (𝓝 x).HasBasis (fun s : Set α => IsOpen s ∧ x ∈ s ∧ IsConnected s) id :=
  ⟨@LocallyConnectedSpace.open_connected_basis _ _, LocallyConnectedSpace.mk⟩


@[deprecated (since := "2024-11-18")] alias locallyConnectedSpace_iff_open_connected_basis :=
  locallyConnectedSpace_iff_hasBasis_isOpen_isConnected


theorem locallyConnectedSpace_iff_subsets_isOpen_isConnected :
    LocallyConnectedSpace α ↔
      ∀ x, ∀ U ∈ 𝓝 x, ∃ V : Set α, V ⊆ U ∧ IsOpen V ∧ x ∈ V ∧ IsConnected V := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (LocallyConnectedSpace α) (∀ (x : α) (U : Set α), Membership.mem (nhds x …
  -/
  simp_rw [locallyConnectedSpace_iff_hasBasis_isOpen_isConnected]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (∀ (x : α), (nhds x).HasBasis (fun s => And (IsOpen s) (And (Membership. …
  -/
  refine forall_congr' fun _ => ?_
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x✝ : α
    ⊢ Iff ((nhds x✝).HasBasis (fun s => And (IsOpen s) (And (Membership.mem s x✝)  …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      x✝ : α
      ⊢ (nhds x✝).HasBasis (fun s => And (IsOpen s) (And (Membership.mem s x✝) (IsCo …
    -/
  · intro h U hU
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      x✝ : α
      h : (nhds x✝).HasBasis (fun s => And (IsOpen s) (And (Membership.mem s x✝) (Is …
      U : Set α
      hU : Membership.mem (nhds x✝) U
      ⊢ Exists fun V => And (HasSubset.Subset V U) (And (IsOpen V) (And (Membership. …
    -/
    rcases h.mem_iff.mp hU with ⟨V, hV, hVU⟩
    /-
      case mp.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      x✝ : α
      h : (nhds x✝).HasBasis (fun s => And (IsOpen s) (And (Membership.mem s x✝) (Is …
      U : Set α
      hU : Membership.mem (nhds x✝) U
      V : Set α
      hV : And (IsOpen V) (And (Membership.mem V x✝) (IsConnected V))
      hVU : HasSubset.Subset (id V) U
      ⊢ Exists fun V => And (HasSubset.Subset V U) (And (IsOpen V) (And (Membership. …
    -/
    exact ⟨V, hVU, hV⟩
    /-
      🎉 no goals
    -/
  · exact fun h => ⟨fun U => ⟨fun hU =>
      let ⟨V, hVU, hV⟩ := h U hU
      ⟨V, hV, hVU⟩, fun ⟨V, ⟨hV, hxV, _⟩, hVU⟩ => mem_nhds_iff.mpr ⟨V, hVU, hV, hxV⟩⟩⟩


@[deprecated (since := "2024-11-18")] alias locallyConnectedSpace_iff_open_connected_subsets :=
  locallyConnectedSpace_iff_subsets_isOpen_isConnected


/-- A space with discrete topology is a locally connected space. -/
instance (priority := 100) DiscreteTopology.toLocallyConnectedSpace (α) [TopologicalSpace α]
    [DiscreteTopology α] : LocallyConnectedSpace α :=
  locallyConnectedSpace_iff_subsets_isOpen_isConnected.2 fun x _U hU =>
    ⟨{x}, singleton_subset_iff.2 <| mem_of_mem_nhds hU, isOpen_discrete _, rfl,
      isConnected_singleton⟩


theorem connectedComponentIn_mem_nhds [LocallyConnectedSpace α] {F : Set α} {x : α} (h : F ∈ 𝓝 x) :
    connectedComponentIn F x ∈ 𝓝 x := by
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    F : Set α
    x : α
    h : Membership.mem (nhds x) F
    ⊢ Membership.mem (nhds x) (connectedComponentIn F x)
  -/
  rw [(LocallyConnectedSpace.open_connected_basis x).mem_iff] at h
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    F : Set α
    x : α
    h : Exists fun i => And (And (IsOpen i) (And (Membership.mem i x) (IsConnected …
    ⊢ Membership.mem (nhds x) (connectedComponentIn F x)
  -/
  rcases h with ⟨s, ⟨h1s, hxs, h2s⟩, hsF⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    F : Set α
    x : α
    s : Set α
    hsF : HasSubset.Subset (id s) F
    h1s : IsOpen s
    hxs : Membership.mem s x
    h2s : IsConnected s
    ⊢ Membership.mem (nhds x) (connectedComponentIn F x)
  -/
  exact mem_nhds_iff.mpr ⟨s, h2s.isPreconnected.subset_connectedComponentIn hxs hsF, h1s, hxs⟩
  /-
    🎉 no goals
  -/


protected theorem IsOpen.connectedComponentIn [LocallyConnectedSpace α] {F : Set α} {x : α}
    (hF : IsOpen F) : IsOpen (connectedComponentIn F x) := by
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    F : Set α
    x : α
    hF : IsOpen F
    ⊢ IsOpen (connectedComponentIn F x)
  -/
  rw [isOpen_iff_mem_nhds]
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    F : Set α
    x : α
    hF : IsOpen F
    ⊢ ∀ (x_1 : α), Membership.mem (connectedComponentIn F x) x_1 → Membership.mem  …
  -/
  intro y hy
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    F : Set α
    x : α
    hF : IsOpen F
    y : α
    hy : Membership.mem (connectedComponentIn F x) y
    ⊢ Membership.mem (nhds y) (connectedComponentIn F x)
  -/
  rw [connectedComponentIn_eq hy]
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    F : Set α
    x : α
    hF : IsOpen F
    y : α
    hy : Membership.mem (connectedComponentIn F x) y
    ⊢ Membership.mem (nhds y) (connectedComponentIn F y)
  -/
  exact connectedComponentIn_mem_nhds (hF.mem_nhds <| connectedComponentIn_subset F x hy)
  /-
    🎉 no goals
  -/


theorem isOpen_connectedComponent [LocallyConnectedSpace α] {x : α} :
    IsOpen (connectedComponent x) := by
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    x : α
    ⊢ IsOpen (connectedComponent x)
  -/
  rw [← connectedComponentIn_univ]
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyConnectedSpace α
    x : α
    ⊢ IsOpen (connectedComponentIn Set.univ x)
  -/
  exact isOpen_univ.connectedComponentIn
  /-
    🎉 no goals
  -/


theorem isClopen_connectedComponent [LocallyConnectedSpace α] {x : α} :
    IsClopen (connectedComponent x) :=
  ⟨isClosed_connectedComponent, isOpen_connectedComponent⟩


theorem locallyConnectedSpace_iff_connectedComponentIn_open :
    LocallyConnectedSpace α ↔
      ∀ F : Set α, IsOpen F → ∀ x ∈ F, IsOpen (connectedComponentIn F x) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (LocallyConnectedSpace α) (∀ (F : Set α), IsOpen F → ∀ (x : α), Membersh …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ LocallyConnectedSpace α → ∀ (F : Set α), IsOpen F → ∀ (x : α), Membership.me …
    -/
  · intro h
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      h : LocallyConnectedSpace α
      ⊢ ∀ (F : Set α), IsOpen F → ∀ (x : α), Membership.mem F x → IsOpen (connectedC …
    -/
    exact fun F hF x _ => hF.connectedComponentIn
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ (∀ (F : Set α), IsOpen F → ∀ (x : α), Membership.mem F x → IsOpen (connected …
    -/
  · intro h
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (F : Set α), IsOpen F → ∀ (x : α), Membership.mem F x → IsOpen (connecte …
      ⊢ LocallyConnectedSpace α
    -/
    rw [locallyConnectedSpace_iff_subsets_isOpen_isConnected]
    refine fun x U hU =>
        ⟨connectedComponentIn (interior U) x,
          (connectedComponentIn_subset _ _).trans interior_subset, h _ isOpen_interior x ?_,
          mem_connectedComponentIn ?_, isConnected_connectedComponentIn_iff.mpr ?_⟩ <;>
      /-
        case mpr.refine_1
        α : Type u
        inst✝ : TopologicalSpace α
        h : ∀ (F : Set α), IsOpen F → ∀ (x : α), Membership.mem F x → IsOpen (connecte …
        x : α
        U : Set α
        hU : Membership.mem (nhds x) U
        ⊢ Membership.mem (interior U) x
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      exact mem_interior_iff_mem_nhds.mpr hU
      /-
        🎉 no goals
      -/


theorem locallyConnectedSpace_iff_connected_subsets :
    LocallyConnectedSpace α ↔ ∀ (x : α), ∀ U ∈ 𝓝 x, ∃ V ∈ 𝓝 x, IsPreconnected V ∧ V ⊆ U := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (LocallyConnectedSpace α) (∀ (x : α) (U : Set α), Membership.mem (nhds x …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ LocallyConnectedSpace α → ∀ (x : α) (U : Set α), Membership.mem (nhds x) U → …
    -/
  · rw [locallyConnectedSpace_iff_subsets_isOpen_isConnected]
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ (∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And (Has …
    -/
    intro h x U hxU
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And (Ha …
      x : α
      U : Set α
      hxU : Membership.mem (nhds x) U
      ⊢ Exists fun V => And (Membership.mem (nhds x) V) (And (IsPreconnected V) (Has …
    -/
    rcases h x U hxU with ⟨V, hVU, hV₁, hxV, hV₂⟩
    /-
      case mp.intro.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And (Ha …
      x : α
      U : Set α
      hxU : Membership.mem (nhds x) U
      V : Set α
      hVU : HasSubset.Subset V U
      hV₁ : IsOpen V
      hxV : Membership.mem V x
      hV₂ : IsConnected V
      ⊢ Exists fun V => And (Membership.mem (nhds x) V) (And (IsPreconnected V) (Has …
    -/
    exact ⟨V, hV₁.mem_nhds hxV, hV₂.isPreconnected, hVU⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ (∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And (Mem …
    -/
  · rw [locallyConnectedSpace_iff_connectedComponentIn_open]
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ (∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And (Mem …
    -/
    refine fun h U hU x _ => isOpen_iff_mem_nhds.mpr fun y hy => ?_
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And (Me …
      U : Set α
      hU : IsOpen U
      x : α
      x✝ : Membership.mem U x
      y : α
      hy : Membership.mem (connectedComponentIn U x) y
      ⊢ Membership.mem (nhds y) (connectedComponentIn U x)
    -/
    rw [connectedComponentIn_eq hy]
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And (Me …
      U : Set α
      hU : IsOpen U
      x : α
      x✝ : Membership.mem U x
      y : α
      hy : Membership.mem (connectedComponentIn U x) y
      ⊢ Membership.mem (nhds y) (connectedComponentIn U y)
    -/
    rcases h y U (hU.mem_nhds <| (connectedComponentIn_subset _ _) hy) with ⟨V, hVy, hV, hVU⟩
    /-
      case mpr.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And (Me …
      U : Set α
      hU : IsOpen U
      x : α
      x✝ : Membership.mem U x
      y : α
      hy : Membership.mem (connectedComponentIn U x) y
      V : Set α
      hVy : Membership.mem (nhds y) V
      hV : IsPreconnected V
      hVU : HasSubset.Subset V U
      ⊢ Membership.mem (nhds y) (connectedComponentIn U y)
    -/
    exact Filter.mem_of_superset hVy (hV.subset_connectedComponentIn (mem_of_mem_nhds hVy) hVU)
    /-
      🎉 no goals
    -/


theorem locallyConnectedSpace_iff_connected_basis :
    LocallyConnectedSpace α ↔
      ∀ x, (𝓝 x).HasBasis (fun s : Set α => s ∈ 𝓝 x ∧ IsPreconnected s) id := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (LocallyConnectedSpace α) (∀ (x : α), (nhds x).HasBasis (fun s => And (M …
  -/
  rw [locallyConnectedSpace_iff_connected_subsets]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (∀ (x : α) (U : Set α), Membership.mem (nhds x) U → Exists fun V => And  …
  -/
  exact forall_congr' fun x => Filter.hasBasis_self.symm
  /-
    🎉 no goals
  -/


theorem locallyConnectedSpace_of_connected_bases {ι : Type*} (b : α → ι → Set α) (p : α → ι → Prop)
    (hbasis : ∀ x, (𝓝 x).HasBasis (p x) (b x))
    (hconnected : ∀ x i, p x i → IsPreconnected (b x i)) : LocallyConnectedSpace α := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ι : Type u_3
    b : α → ι → Set α
    p : α → ι → Prop
    hbasis : ∀ (x : α), (nhds x).HasBasis (p x) (b x)
    hconnected : ∀ (x : α) (i : ι), p x i → IsPreconnected (b x i)
    ⊢ LocallyConnectedSpace α
  -/
  rw [locallyConnectedSpace_iff_connected_basis]
  exact fun x =>
    (hbasis x).to_hasBasis
      (fun i hi => ⟨b x i, ⟨(hbasis x).mem_of_mem hi, hconnected x i hi⟩, subset_rfl⟩) fun s hs =>
      ⟨(hbasis x).index s hs.1, ⟨(hbasis x).property_index hs.1, (hbasis x).set_index_subset hs.1⟩⟩


lemma Topology.IsOpenEmbedding.locallyConnectedSpace [LocallyConnectedSpace α] [TopologicalSpace β]
    {f : β → α} (h : IsOpenEmbedding f) : LocallyConnectedSpace β := by
  refine locallyConnectedSpace_of_connected_bases (fun _ s ↦ f ⁻¹' s)
    (fun x s ↦ (IsOpen s ∧ f x ∈ s ∧ IsConnected s) ∧ s ⊆ range f) (fun x ↦ ?_)
    (fun x s hxs ↦ hxs.1.2.2.isPreconnected.preimage_of_isOpenMap h.injective h.isOpenMap hxs.2)
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LocallyConnectedSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    h : Topology.IsOpenEmbedding f
    x : β
    ⊢ (nhds x).HasBasis ((fun x s => And (And (IsOpen s) (And (Membership.mem s (f …
  -/
  rw [h.nhds_eq_comap]
  exact LocallyConnectedSpace.open_connected_basis (f x) |>.restrict_subset
    (h.isOpen_range.mem_nhds <| mem_range_self _) |>.comap _


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.locallyConnectedSpace := IsOpenEmbedding.locallyConnectedSpace


theorem IsOpen.locallyConnectedSpace [LocallyConnectedSpace α] {U : Set α} (hU : IsOpen U) :
    LocallyConnectedSpace U :=
  hU.isOpenEmbedding_subtypeVal.locallyConnectedSpace


