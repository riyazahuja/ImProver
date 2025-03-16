/-- A topological additive group is nonarchimedean if every neighborhood of 0
  contains an open subgroup. -/
class NonarchimedeanAddGroup (G : Type*) [AddGroup G] [TopologicalSpace G] extends
  TopologicalAddGroup G : Prop where
  is_nonarchimedean : ∀ U ∈ 𝓝 (0 : G), ∃ V : OpenAddSubgroup G, (V : Set G) ⊆ U


/-- A topological group is nonarchimedean if every neighborhood of 1 contains an open subgroup. -/
@[to_additive]
class NonarchimedeanGroup (G : Type*) [Group G] [TopologicalSpace G] extends TopologicalGroup G :
  Prop where
  is_nonarchimedean : ∀ U ∈ 𝓝 (1 : G), ∃ V : OpenSubgroup G, (V : Set G) ⊆ U


/-- A topological ring is nonarchimedean if its underlying topological additive
  group is nonarchimedean. -/
class NonarchimedeanRing (R : Type*) [Ring R] [TopologicalSpace R] extends TopologicalRing R :
  Prop where
  is_nonarchimedean : ∀ U ∈ 𝓝 (0 : R), ∃ V : OpenAddSubgroup R, (V : Set R) ⊆ U

-- see Note [lower instance priority]

/-- Every nonarchimedean ring is naturally a nonarchimedean additive group. -/
instance (priority := 100) NonarchimedeanRing.to_nonarchimedeanAddGroup (R : Type*) [Ring R]
    [TopologicalSpace R] [t : NonarchimedeanRing R] : NonarchimedeanAddGroup R :=
  { t with }


/-- If a topological group embeds into a nonarchimedean group, then it is nonarchimedean. -/
@[to_additive]
theorem nonarchimedean_of_emb (f : G →* H) (emb : IsOpenEmbedding f) : NonarchimedeanGroup H :=
  { is_nonarchimedean := fun U hU =>
      have h₁ : f ⁻¹' U ∈ 𝓝 (1 : G) := by
        /-
          G : Type u_1
          inst✝⁵ : Group G
          inst✝⁴ : TopologicalSpace G
          inst✝³ : NonarchimedeanGroup G
          H : Type u_2
          inst✝² : Group H
          inst✝¹ : TopologicalSpace H
          inst✝ : TopologicalGroup H
          f : MonoidHom G H
          emb : Topology.IsOpenEmbedding ⇑f
          U : Set H
          hU : Membership.mem (nhds 1) U
          ⊢ Membership.mem (nhds 1) (Set.preimage (⇑f) U)
        -/
        apply emb.continuous.tendsto
        /-
          case a
          G : Type u_1
          inst✝⁵ : Group G
          inst✝⁴ : TopologicalSpace G
          inst✝³ : NonarchimedeanGroup G
          H : Type u_2
          inst✝² : Group H
          inst✝¹ : TopologicalSpace H
          inst✝ : TopologicalGroup H
          f : MonoidHom G H
          emb : Topology.IsOpenEmbedding ⇑f
          U : Set H
          hU : Membership.mem (nhds 1) U
          ⊢ Membership.mem (nhds (f 1)) U
        -/
        rwa [f.map_one]
        /-
          🎉 no goals
        -/
      let ⟨V, hV⟩ := is_nonarchimedean (f ⁻¹' U) h₁
      ⟨{ Subgroup.map f V with isOpen' := emb.isOpenMap _ V.isOpen }, Set.image_subset_iff.2 hV⟩ }


/-- An open neighborhood of the identity in the cartesian product of two nonarchimedean groups
contains the cartesian product of an open neighborhood in each group. -/
@[to_additive NonarchimedeanAddGroup.prod_subset "An open neighborhood of the identity in
the cartesian product of two nonarchimedean groups contains the cartesian product of
an open neighborhood in each group."]
theorem prod_subset {U} (hU : U ∈ 𝓝 (1 : G × K)) :
    ∃ (V : OpenSubgroup G) (W : OpenSubgroup K), (V : Set G) ×ˢ (W : Set K) ⊆ U := by
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : NonarchimedeanGroup G
    K : Type u_3
    inst✝² : Group K
    inst✝¹ : TopologicalSpace K
    inst✝ : NonarchimedeanGroup K
    U : Set (Prod G K)
    hU : Membership.mem (nhds 1) U
    ⊢ Exists fun V => Exists fun W => HasSubset.Subset (SProd.sprod ↑V ↑W) U
  -/
  rw [nhds_prod_eq, Filter.mem_prod_iff] at hU
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : NonarchimedeanGroup G
    K : Type u_3
    inst✝² : Group K
    inst✝¹ : TopologicalSpace K
    inst✝ : NonarchimedeanGroup K
    U : Set (Prod G K)
    hU : Exists fun t₁ => And (Membership.mem (nhds 1.1) t₁) (Exists fun t₂ => And …
    ⊢ Exists fun V => Exists fun W => HasSubset.Subset (SProd.sprod ↑V ↑W) U
  -/
  rcases hU with ⟨U₁, hU₁, U₂, hU₂, h⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : NonarchimedeanGroup G
    K : Type u_3
    inst✝² : Group K
    inst✝¹ : TopologicalSpace K
    inst✝ : NonarchimedeanGroup K
    U : Set (Prod G K)
    U₁ : Set G
    hU₁ : Membership.mem (nhds 1.1) U₁
    U₂ : Set K
    hU₂ : Membership.mem (nhds 1.2) U₂
    h : HasSubset.Subset (SProd.sprod U₁ U₂) U
    ⊢ Exists fun V => Exists fun W => HasSubset.Subset (SProd.sprod ↑V ↑W) U
  -/
  cases' is_nonarchimedean _ hU₁ with V hV
  /-
    case intro.intro.intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : NonarchimedeanGroup G
    K : Type u_3
    inst✝² : Group K
    inst✝¹ : TopologicalSpace K
    inst✝ : NonarchimedeanGroup K
    U : Set (Prod G K)
    U₁ : Set G
    hU₁ : Membership.mem (nhds 1.1) U₁
    U₂ : Set K
    hU₂ : Membership.mem (nhds 1.2) U₂
    h : HasSubset.Subset (SProd.sprod U₁ U₂) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U₁
    ⊢ Exists fun V => Exists fun W => HasSubset.Subset (SProd.sprod ↑V ↑W) U
  -/
  cases' is_nonarchimedean _ hU₂ with W hW
  /-
    case intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : NonarchimedeanGroup G
    K : Type u_3
    inst✝² : Group K
    inst✝¹ : TopologicalSpace K
    inst✝ : NonarchimedeanGroup K
    U : Set (Prod G K)
    U₁ : Set G
    hU₁ : Membership.mem (nhds 1.1) U₁
    U₂ : Set K
    hU₂ : Membership.mem (nhds 1.2) U₂
    h : HasSubset.Subset (SProd.sprod U₁ U₂) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U₁
    W : OpenSubgroup K
    hW : HasSubset.Subset (↑W) U₂
    ⊢ Exists fun V => Exists fun W => HasSubset.Subset (SProd.sprod ↑V ↑W) U
  -/
  use V; use W
  /-
    case h
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : NonarchimedeanGroup G
    K : Type u_3
    inst✝² : Group K
    inst✝¹ : TopologicalSpace K
    inst✝ : NonarchimedeanGroup K
    U : Set (Prod G K)
    U₁ : Set G
    hU₁ : Membership.mem (nhds 1.1) U₁
    U₂ : Set K
    hU₂ : Membership.mem (nhds 1.2) U₂
    h : HasSubset.Subset (SProd.sprod U₁ U₂) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U₁
    W : OpenSubgroup K
    hW : HasSubset.Subset (↑W) U₂
    ⊢ HasSubset.Subset (SProd.sprod ↑V ↑W) U
  -/
  rw [Set.prod_subset_iff]
  /-
    case h
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : NonarchimedeanGroup G
    K : Type u_3
    inst✝² : Group K
    inst✝¹ : TopologicalSpace K
    inst✝ : NonarchimedeanGroup K
    U : Set (Prod G K)
    U₁ : Set G
    hU₁ : Membership.mem (nhds 1.1) U₁
    U₂ : Set K
    hU₂ : Membership.mem (nhds 1.2) U₂
    h : HasSubset.Subset (SProd.sprod U₁ U₂) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U₁
    W : OpenSubgroup K
    hW : HasSubset.Subset (↑W) U₂
    ⊢ ∀ (x : G), Membership.mem (↑V) x → ∀ (y : K), Membership.mem (↑W) y → Member …
  -/
  intro x hX y hY
  /-
    case h
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : NonarchimedeanGroup G
    K : Type u_3
    inst✝² : Group K
    inst✝¹ : TopologicalSpace K
    inst✝ : NonarchimedeanGroup K
    U : Set (Prod G K)
    U₁ : Set G
    hU₁ : Membership.mem (nhds 1.1) U₁
    U₂ : Set K
    hU₂ : Membership.mem (nhds 1.2) U₂
    h : HasSubset.Subset (SProd.sprod U₁ U₂) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U₁
    W : OpenSubgroup K
    hW : HasSubset.Subset (↑W) U₂
    x : G
    hX : Membership.mem (↑V) x
    y : K
    hY : Membership.mem (↑W) y
    ⊢ Membership.mem U { fst := x, snd := y }
  -/
  exact Set.Subset.trans (Set.prod_mono hV hW) h (Set.mem_sep hX hY)
  /-
    🎉 no goals
  -/


/-- An open neighborhood of the identity in the cartesian square of a nonarchimedean group
contains the cartesian square of an open neighborhood in the group. -/
@[to_additive NonarchimedeanAddGroup.prod_self_subset "An open neighborhood of the identity in
the cartesian square of a nonarchimedean group contains the cartesian square of
an open neighborhood in the group."]
theorem prod_self_subset {U} (hU : U ∈ 𝓝 (1 : G × G)) :
    ∃ V : OpenSubgroup G, (V : Set G) ×ˢ (V : Set G) ⊆ U :=
  let ⟨V, W, h⟩ := prod_subset hU
             /-
               G : Type u_1
               inst✝² : Group G
               inst✝¹ : TopologicalSpace G
               inst✝ : NonarchimedeanGroup G
               U : Set (Prod G G)
               hU : Membership.mem (nhds 1) U
               V W : OpenSubgroup G
               h : HasSubset.Subset (SProd.sprod ↑V ↑W) U
               ⊢ HasSubset.Subset (SProd.sprod ↑(Min.min V W) ↑(Min.min V W)) U
             -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  ⟨V ⊓ W, by refine Set.Subset.trans (Set.prod_mono ?_ ?_) ‹_› <;> simp⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The cartesian product of two nonarchimedean groups is nonarchimedean. -/
@[to_additive "The cartesian product of two nonarchimedean groups is nonarchimedean."]
instance : NonarchimedeanGroup (G × K) where
  is_nonarchimedean _ hU :=
    let ⟨V, W, h⟩ := prod_subset hU
    ⟨V.prod W, ‹_›⟩


/-- The cartesian product of two nonarchimedean rings is nonarchimedean. -/
instance : NonarchimedeanRing (R × S) where
  is_nonarchimedean := NonarchimedeanAddGroup.is_nonarchimedean


/-- Given an open subgroup `U` and an element `r` of a nonarchimedean ring, there is an open
  subgroup `V` such that `r • V` is contained in `U`. -/
theorem left_mul_subset (U : OpenAddSubgroup R) (r : R) :
    ∃ V : OpenAddSubgroup R, r • (V : Set R) ⊆ U :=
  ⟨U.comap (AddMonoidHom.mulLeft r) (continuous_mul_left r), (U : Set R).image_preimage_subset _⟩


/-- An open subgroup of a nonarchimedean ring contains the square of another one. -/
theorem mul_subset (U : OpenAddSubgroup R) : ∃ V : OpenAddSubgroup R, (V : Set R) * V ⊆ U := by
  let ⟨V, H⟩ := prod_self_subset <| (U.isOpen.preimage continuous_mul).mem_nhds <| by
    simpa only [Set.mem_preimage, Prod.snd_zero, mul_zero] using U.zero_mem
  /-
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : TopologicalSpace R
    inst✝ : NonarchimedeanRing R
    U V : OpenAddSubgroup R
    H : HasSubset.Subset (SProd.sprod ↑V ↑V) (Set.preimage (fun p => HMul.hMul p.1 …
    ⊢ Exists fun V => HasSubset.Subset (HMul.hMul ↑V ↑V) ↑U
  -/
  use V
  /-
    case h
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : TopologicalSpace R
    inst✝ : NonarchimedeanRing R
    U V : OpenAddSubgroup R
    H : HasSubset.Subset (SProd.sprod ↑V ↑V) (Set.preimage (fun p => HMul.hMul p.1 …
    ⊢ HasSubset.Subset (HMul.hMul ↑V ↑V) ↑U
  -/
  rintro v ⟨a, ha, b, hb, hv⟩
  /-
    case h.intro.intro.intro.intro
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : TopologicalSpace R
    inst✝ : NonarchimedeanRing R
    U V : OpenAddSubgroup R
    H : HasSubset.Subset (SProd.sprod ↑V ↑V) (Set.preimage (fun p => HMul.hMul p.1 …
    v a : R
    ha : Membership.mem (↑V) a
    b : R
    hb : Membership.mem (↑V) b
    hv : Eq ((fun x1 x2 => HMul.hMul x1 x2) a b) v
    ⊢ Membership.mem (↑U) v
  -/
  have hy := H (Set.mk_mem_prod ha hb)
  /-
    case h.intro.intro.intro.intro
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : TopologicalSpace R
    inst✝ : NonarchimedeanRing R
    U V : OpenAddSubgroup R
    H : HasSubset.Subset (SProd.sprod ↑V ↑V) (Set.preimage (fun p => HMul.hMul p.1 …
    v a : R
    ha : Membership.mem (↑V) a
    b : R
    hb : Membership.mem (↑V) b
    hv : Eq ((fun x1 x2 => HMul.hMul x1 x2) a b) v
    hy : Membership.mem (Set.preimage (fun p => HMul.hMul p.1 p.2) ↑U) { fst := a, …
    ⊢ Membership.mem (↑U) v
  -/
  simp only [Set.mem_preimage, SetLike.mem_coe, hv] at hy
  /-
    case h.intro.intro.intro.intro
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : TopologicalSpace R
    inst✝ : NonarchimedeanRing R
    U V : OpenAddSubgroup R
    H : HasSubset.Subset (SProd.sprod ↑V ↑V) (Set.preimage (fun p => HMul.hMul p.1 …
    v a : R
    ha : Membership.mem (↑V) a
    b : R
    hb : Membership.mem (↑V) b
    hv : Eq ((fun x1 x2 => HMul.hMul x1 x2) a b) v
    hy : Membership.mem U v
    ⊢ Membership.mem (↑U) v
  -/
  rw [SetLike.mem_coe]
  /-
    case h.intro.intro.intro.intro
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : TopologicalSpace R
    inst✝ : NonarchimedeanRing R
    U V : OpenAddSubgroup R
    H : HasSubset.Subset (SProd.sprod ↑V ↑V) (Set.preimage (fun p => HMul.hMul p.1 …
    v a : R
    ha : Membership.mem (↑V) a
    b : R
    hb : Membership.mem (↑V) b
    hv : Eq ((fun x1 x2 => HMul.hMul x1 x2) a b) v
    hy : Membership.mem U v
    ⊢ Membership.mem U v
  -/
  exact hy
  /-
    🎉 no goals
  -/


