theorem totallyBounded_convexHull (hs : TotallyBounded s) :
    TotallyBounded (convexHull ℝ s) := by
  /-
    E : Type u_1
    s : Set E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    ⊢ TotallyBounded ((convexHull Real) s)
  -/
  rw [totallyBounded_iff_subset_finite_iUnion_nhds_zero]
  /-
    E : Type u_1
    s : Set E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    ⊢ ∀ (U : Set E), Membership.mem (nhds 0) U → Exists fun t => And t.Finite (Has …
  -/
  intro U hU
  /-
    E : Type u_1
    s : Set E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    U : Set E
    hU : Membership.mem (nhds 0) U
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset ((convexHull Real) s) (Set.iU …
  -/
  obtain ⟨W, hW₁, hW₂⟩ := exists_nhds_zero_half hU
  /-
    case intro.intro
    E : Type u_1
    s : Set E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    U : Set E
    hU : Membership.mem (nhds 0) U
    W : Set E
    hW₁ : Membership.mem (nhds 0) W
    hW₂ : ∀ (v : E), Membership.mem W v → ∀ (w : E), Membership.mem W w → Membersh …
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset ((convexHull Real) s) (Set.iU …
  -/
  obtain ⟨V, ⟨hV₁, hV₂, hV₃⟩⟩ := (locallyConvexSpace_iff_exists_convex_subset_zero ℝ E).mp lcs W hW₁
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    s : Set E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    U : Set E
    hU : Membership.mem (nhds 0) U
    W : Set E
    hW₁ : Membership.mem (nhds 0) W
    hW₂ : ∀ (v : E), Membership.mem W v → ∀ (w : E), Membership.mem W w → Membersh …
    V : Set E
    hV₁ : Membership.mem (nhds 0) V
    hV₂ : Convex Real V
    hV₃ : HasSubset.Subset V W
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset ((convexHull Real) s) (Set.iU …
  -/
  obtain ⟨t, ⟨htf, hts⟩⟩ := (totallyBounded_iff_subset_finite_iUnion_nhds_zero.mp hs) _ hV₁
  obtain ⟨t', ⟨htf', hts'⟩⟩ := (totallyBounded_iff_subset_finite_iUnion_nhds_zero.mp
    (IsCompact.totallyBounded (Finite.isCompact_convexHull htf)) _ hV₁)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    s : Set E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    U : Set E
    hU : Membership.mem (nhds 0) U
    W : Set E
    hW₁ : Membership.mem (nhds 0) W
    hW₂ : ∀ (v : E), Membership.mem W v → ∀ (w : E), Membership.mem W w → Membersh …
    V : Set E
    hV₁ : Membership.mem (nhds 0) V
    hV₂ : Convex Real V
    hV₃ : HasSubset.Subset V W
    t : Set E
    htf : t.Finite
    hts : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => HVAdd.hVAdd  …
    t' : Set E
    htf' : t'.Finite
    hts' : HasSubset.Subset ((convexHull Real) t) (Set.iUnion fun y => Set.iUnion  …
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset ((convexHull Real) s) (Set.iU …
  -/
  use t', htf'
  /-
    case right
    E : Type u_1
    s : Set E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    U : Set E
    hU : Membership.mem (nhds 0) U
    W : Set E
    hW₁ : Membership.mem (nhds 0) W
    hW₂ : ∀ (v : E), Membership.mem W v → ∀ (w : E), Membership.mem W w → Membersh …
    V : Set E
    hV₁ : Membership.mem (nhds 0) V
    hV₂ : Convex Real V
    hV₃ : HasSubset.Subset V W
    t : Set E
    htf : t.Finite
    hts : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => HVAdd.hVAdd  …
    t' : Set E
    htf' : t'.Finite
    hts' : HasSubset.Subset ((convexHull Real) t) (Set.iUnion fun y => Set.iUnion  …
    ⊢ HasSubset.Subset ((convexHull Real) s) (Set.iUnion fun y => Set.iUnion fun h …
  -/
  simp only [iUnion_vadd_set, vadd_eq_add] at hts hts' ⊢
  calc convexHull ℝ s
    _ ⊆ convexHull ℝ (t + V) := convexHull_mono hts
    _ ⊆ convexHull ℝ t + convexHull ℝ V := convexHull_add_subset
    _ = convexHull ℝ t + V := by rw [hV₂.convexHull_eq]
    _ ⊆ t' + V + V := add_subset_add_right hts'
    _ = t' + (V + V) := by rw [add_assoc]
    _ ⊆ t' + (W + W) := add_subset_add_left (add_subset_add hV₃ hV₃)
    _ ⊆ t' + U := add_subset_add_left (add_subset_iff.mpr hW₂)

