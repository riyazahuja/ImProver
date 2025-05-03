theorem IsClosed.isGδ {X : Type*} [UniformSpace X] [IsCountablyGenerated (𝓤 X)] {s : Set X}
    (hs : IsClosed s) : IsGδ s := by
  /-
    X : Type u_3
    inst✝¹ : UniformSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    s : Set X
    hs : IsClosed s
    ⊢ IsGδ s
  -/
  rcases (@uniformity_hasBasis_open X _).exists_antitone_subbasis with ⟨U, hUo, hU, -⟩
  /-
    case intro.intro.mk
    X : Type u_3
    inst✝¹ : UniformSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    s : Set X
    hs : IsClosed s
    U : Nat → Set (Prod X X)
    hUo : ∀ (i : Nat), And (Membership.mem (uniformity X) (U i)) (IsOpen (U i))
    hU : (uniformity X).HasBasis (fun x => True) fun i => id (U i)
    ⊢ IsGδ s
  -/
  rw [← hs.closure_eq, ← hU.biInter_biUnion_ball]
  /-
    case intro.intro.mk
    X : Type u_3
    inst✝¹ : UniformSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    s : Set X
    hs : IsClosed s
    U : Nat → Set (Prod X X)
    hUo : ∀ (i : Nat), And (Membership.mem (uniformity X) (U i)) (IsOpen (U i))
    hU : (uniformity X).HasBasis (fun x => True) fun i => id (U i)
    ⊢ IsGδ (Set.iInter fun i => Set.iInter fun x => Set.iUnion fun x => Set.iUnion …
  -/
  refine .biInter (to_countable _) fun n _ => IsOpen.isGδ ?_
  /-
    case intro.intro.mk
    X : Type u_3
    inst✝¹ : UniformSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    s : Set X
    hs : IsClosed s
    U : Nat → Set (Prod X X)
    hUo : ∀ (i : Nat), And (Membership.mem (uniformity X) (U i)) (IsOpen (U i))
    hU : (uniformity X).HasBasis (fun x => True) fun i => id (U i)
    n : Nat
    x✝ : Membership.mem (fun i => True) n
    ⊢ IsOpen (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ball x (id (U n …
  -/
  exact isOpen_biUnion fun x _ => UniformSpace.isOpen_ball _ (hUo _).2
  /-
    🎉 no goals
  -/


/-- The set of points where a function is continuous is a Gδ set. -/
theorem IsGδ.setOf_continuousAt [UniformSpace Y] [IsCountablyGenerated (𝓤 Y)] (f : X → Y) :
    IsGδ { x | ContinuousAt f x } := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace Y
    inst✝ : (uniformity Y).IsCountablyGenerated
    f : X → Y
    ⊢ IsGδ (setOf fun x => ContinuousAt f x)
  -/
  obtain ⟨U, _, hU⟩ := (@uniformity_hasBasis_open_symmetric Y _).exists_antitone_subbasis
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace Y
    inst✝ : (uniformity Y).IsCountablyGenerated
    f : X → Y
    U : Nat → Set (Prod Y Y)
    left✝ : ∀ (i : Nat), And (Membership.mem (uniformity Y) (U i)) (And (IsOpen (U …
    hU : (uniformity Y).HasAntitoneBasis fun i => id (U i)
    ⊢ IsGδ (setOf fun x => ContinuousAt f x)
  -/
  simp only [Uniform.continuousAt_iff_prod, nhds_prod_eq]
  simp only [(nhds_basis_opens _).prod_self.tendsto_iff hU.toHasBasis, forall_prop_of_true,
    setOf_forall, id]
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace Y
    inst✝ : (uniformity Y).IsCountablyGenerated
    f : X → Y
    U : Nat → Set (Prod Y Y)
    left✝ : ∀ (i : Nat), And (Membership.mem (uniformity Y) (U i)) (And (IsOpen (U …
    hU : (uniformity Y).HasAntitoneBasis fun i => id (U i)
    ⊢ IsGδ (Set.iInter fun i => setOf fun x => Exists fun ia => And (And (Membersh …
  -/
  refine .iInter fun k ↦ IsOpen.isGδ <| isOpen_iff_mem_nhds.2 fun x ↦ ?_
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace Y
    inst✝ : (uniformity Y).IsCountablyGenerated
    f : X → Y
    U : Nat → Set (Prod Y Y)
    left✝ : ∀ (i : Nat), And (Membership.mem (uniformity Y) (U i)) (And (IsOpen (U …
    hU : (uniformity Y).HasAntitoneBasis fun i => id (U i)
    k : Nat
    x : X
    ⊢ Membership.mem (setOf fun x => Exists fun ia => And (And (Membership.mem ia  …
  -/
  rintro ⟨s, ⟨hsx, hso⟩, hsU⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : UniformSpace Y
    inst✝ : (uniformity Y).IsCountablyGenerated
    f : X → Y
    U : Nat → Set (Prod Y Y)
    left✝ : ∀ (i : Nat), And (Membership.mem (uniformity Y) (U i)) (And (IsOpen (U …
    hU : (uniformity Y).HasAntitoneBasis fun i => id (U i)
    k : Nat
    x : X
    s : Set X
    hsU : ∀ (x : Prod X X), Membership.mem (SProd.sprod s s) x → Membership.mem (U …
    hsx : Membership.mem s x
    hso : IsOpen s
    ⊢ Membership.mem (nhds x) (setOf fun x => Exists fun ia => And (And (Membershi …
  -/
  filter_upwards [IsOpen.mem_nhds hso hsx] with _ hy using ⟨s, ⟨hy, hso⟩, hsU⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-15")] alias isGδ_setOf_continuousAt := IsGδ.setOf_continuousAt


