theorem PartitionOfUnity.finsum_smul_mem_convex {s : Set X} (f : PartitionOfUnity ι X s)
    {g : ι → X → E} {t : Set E} {x : X} (hx : x ∈ s) (hg : ∀ i, f i x ≠ 0 → g i x ∈ t)
    (ht : Convex ℝ t) : (∑ᶠ i, f i x • g i x) ∈ t :=
  ht.finsum_mem (fun _ => f.nonneg _ _) (f.sum_eq_one hx) hg


/-- Let `X` be a normal paracompact topological space (e.g., any extended metric space). Let `E` be
a topological real vector space. Let `t : X → Set E` be a family of convex sets. Suppose that for
each point `x : X`, there exists a neighborhood `U ∈ 𝓝 X` and a function `g : X → E` that is
continuous on `U` and sends each `y ∈ U` to a point of `t y`. Then there exists a continuous map
`g : C(X, E)` such that `g x ∈ t x` for all `x`. See also
`exists_continuous_forall_mem_convex_of_local_const`. -/
theorem exists_continuous_forall_mem_convex_of_local (ht : ∀ x, Convex ℝ (t x))
    (H : ∀ x : X, ∃ U ∈ 𝓝 x, ∃ g : X → E, ContinuousOn g U ∧ ∀ y ∈ U, g y ∈ t y) :
    ∃ g : C(X, E), ∀ x, g x ∈ t x := by
  /-
    X : Type u_2
    E : Type u_3
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : NormalSpace X
    inst✝³ : ParacompactSpace X
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    t : X → Set E
    ht : ∀ (x : X), Convex Real (t x)
    H : ∀ (x : X), Exists fun U => And (Membership.mem (nhds x) U) (Exists fun g = …
    ⊢ Exists fun g => ∀ (x : X), Membership.mem (t x) (g x)
  -/
  choose U hU g hgc hgt using H
  obtain ⟨f, hf⟩ := PartitionOfUnity.exists_isSubordinate isClosed_univ (fun x => interior (U x))
    (fun x => isOpen_interior) fun x _ => mem_iUnion.2 ⟨x, mem_interior_iff_mem_nhds.2 (hU x)⟩
  refine ⟨⟨fun x => ∑ᶠ i, f i x • g i x,
    hf.continuous_finsum_smul (fun i => isOpen_interior) fun i => (hgc i).mono interior_subset⟩,
    fun x => f.finsum_smul_mem_convex (mem_univ x) (fun i hi => hgt _ _ ?_) (ht _)⟩
  /-
    case intro
    X : Type u_2
    E : Type u_3
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : NormalSpace X
    inst✝³ : ParacompactSpace X
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    t : X → Set E
    ht : ∀ (x : X), Convex Real (t x)
    U : X → Set X
    hU : ∀ (x : X), Membership.mem (nhds x) (U x)
    g : X → X → E
    hgc : ∀ (x : X), ContinuousOn (g x) (U x)
    hgt : ∀ (x y : X), Membership.mem (U x) y → Membership.mem (t y) (g x y)
    f : PartitionOfUnity X X
    hf : f.IsSubordinate fun x => interior (U x)
    x i : X
    hi : Ne ((f i) x) 0
    ⊢ Membership.mem (U i) x
  -/
  exact interior_subset (hf _ <| subset_closure hi)
  /-
    🎉 no goals
  -/


/-- Let `X` be a normal paracompact topological space (e.g., any extended metric space). Let `E` be
a topological real vector space. Let `t : X → Set E` be a family of convex sets. Suppose that for
each point `x : X`, there exists a vector `c : E` that belongs to `t y` for all `y` in a
neighborhood of `x`. Then there exists a continuous map `g : C(X, E)` such that `g x ∈ t x` for all
`x`. See also `exists_continuous_forall_mem_convex_of_local`. -/
theorem exists_continuous_forall_mem_convex_of_local_const (ht : ∀ x, Convex ℝ (t x))
    (H : ∀ x : X, ∃ c : E, ∀ᶠ y in 𝓝 x, c ∈ t y) : ∃ g : C(X, E), ∀ x, g x ∈ t x :=
  exists_continuous_forall_mem_convex_of_local ht fun x =>
    let ⟨c, hc⟩ := H x
    ⟨_, hc, fun _ => c, continuousOn_const, fun _ => id⟩

