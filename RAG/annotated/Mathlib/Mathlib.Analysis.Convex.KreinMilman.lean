/-- **Krein-Milman lemma**: In a LCTVS, any nonempty compact set has an extreme point. -/
theorem IsCompact.extremePoints_nonempty (hscomp : IsCompact s) (hsnemp : s.Nonempty) :
    (s.extremePoints ℝ).Nonempty := by
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hsnemp : s.Nonempty
    ⊢ (Set.extremePoints Real s).Nonempty
  -/
  let S : Set (Set E) := { t | t.Nonempty ∧ IsClosed t ∧ IsExtreme ℝ s t }
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hsnemp : s.Nonempty
    S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
    ⊢ (Set.extremePoints Real s).Nonempty
  -/
  rsuffices ⟨t, ht⟩ : ∃ t, Minimal (· ∈ S) t
    /-
      case intro
      E : Type u_1
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : T2Space E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hscomp : IsCompact s
      hsnemp : s.Nonempty
      S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
      t : Set E
      ht : Minimal (fun x => Membership.mem S x) t
      ⊢ (Set.extremePoints Real s).Nonempty
    -/
  · obtain ⟨⟨x,hxt⟩, htclos, hst⟩ := ht.prop
    /-
      case intro.intro.intro.intro
      E : Type u_1
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : T2Space E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hscomp : IsCompact s
      hsnemp : s.Nonempty
      S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
      t : Set E
      ht : Minimal (fun x => Membership.mem S x) t
      x : E
      hxt : Membership.mem t x
      htclos : IsClosed t
      hst : IsExtreme Real s t
      ⊢ (Set.extremePoints Real s).Nonempty
    -/
    refine ⟨x, IsExtreme.mem_extremePoints ?_⟩
    /-
      case intro.intro.intro.intro
      E : Type u_1
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : T2Space E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hscomp : IsCompact s
      hsnemp : s.Nonempty
      S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
      t : Set E
      ht : Minimal (fun x => Membership.mem S x) t
      x : E
      hxt : Membership.mem t x
      htclos : IsClosed t
      hst : IsExtreme Real s t
      ⊢ IsExtreme Real s (Singleton.singleton x)
    -/
    rwa [← eq_singleton_iff_unique_mem.2 ⟨hxt, fun y hyB => ?_⟩]
    /-
      E : Type u_1
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : T2Space E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hscomp : IsCompact s
      hsnemp : s.Nonempty
      S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
      t : Set E
      ht : Minimal (fun x => Membership.mem S x) t
      x : E
      hxt : Membership.mem t x
      htclos : IsClosed t
      hst : IsExtreme Real s t
      y : E
      hyB : Membership.mem t y
      ⊢ Eq y x
    -/
    by_contra hyx
    /-
      E : Type u_1
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : T2Space E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hscomp : IsCompact s
      hsnemp : s.Nonempty
      S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
      t : Set E
      ht : Minimal (fun x => Membership.mem S x) t
      x : E
      hxt : Membership.mem t x
      htclos : IsClosed t
      hst : IsExtreme Real s t
      y : E
      hyB : Membership.mem t y
      hyx : Not (Eq y x)
      ⊢ False
    -/
    obtain ⟨l, hl⟩ := geometric_hahn_banach_point_point hyx
    obtain ⟨z, hzt, hz⟩ :=
      (hscomp.of_isClosed_subset htclos hst.1).exists_isMaxOn ⟨x, hxt⟩
        l.continuous.continuousOn
    /-
      case intro.intro.intro
      E : Type u_1
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : T2Space E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hscomp : IsCompact s
      hsnemp : s.Nonempty
      S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
      t : Set E
      ht : Minimal (fun x => Membership.mem S x) t
      x : E
      hxt : Membership.mem t x
      htclos : IsClosed t
      hst : IsExtreme Real s t
      y : E
      hyB : Membership.mem t y
      hyx : Not (Eq y x)
      l : ContinuousLinearMap (RingHom.id Real) E Real
      hl : LT.lt (l y) (l x)
      z : E
      hzt : Membership.mem t z
      hz : IsMaxOn (⇑l) t z
      ⊢ False
    -/
    have h : IsExposed ℝ t ({ z ∈ t | ∀ w ∈ t, l w ≤ l z }) := fun _ => ⟨l, rfl⟩
    rw [ht.eq_of_ge (y := ({ z ∈ t | ∀ w ∈ t, l w ≤ l z }))
      ⟨⟨z, hzt, hz⟩, h.isClosed htclos, hst.trans h.isExtreme⟩ (t.sep_subset _)] at hyB
    /-
      case intro.intro.intro
      E : Type u_1
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : T2Space E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hscomp : IsCompact s
      hsnemp : s.Nonempty
      S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
      t : Set E
      ht : Minimal (fun x => Membership.mem S x) t
      x : E
      hxt : Membership.mem t x
      htclos : IsClosed t
      hst : IsExtreme Real s t
      y : E
      hyx : Not (Eq y x)
      l : ContinuousLinearMap (RingHom.id Real) E Real
      hyB : Membership.mem (setOf fun z => And (Membership.mem t z) (∀ (w : E), Memb …
      hl : LT.lt (l y) (l x)
      z : E
      hzt : Membership.mem t z
      hz : IsMaxOn (⇑l) t z
      h : IsExposed Real t (setOf fun z => And (Membership.mem t z) (∀ (w : E), Memb …
      ⊢ False
    -/
    exact hl.not_le (hyB.2 x hxt)
    /-
      🎉 no goals
    -/
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hsnemp : s.Nonempty
    S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
    ⊢ Exists fun t => Minimal (fun x => Membership.mem S x) t
  -/
  refine zorn_superset _ fun F hFS hF => ?_
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hsnemp : s.Nonempty
    S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
    F : Set (Set E)
    hFS : HasSubset.Subset F S
    hF : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) F
    ⊢ Exists fun lb => And (Membership.mem S lb) (∀ (s : Set E), Membership.mem F  …
  -/
  obtain rfl | hFnemp := F.eq_empty_or_nonempty
    /-
      case inl
      E : Type u_1
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : T2Space E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hscomp : IsCompact s
      hsnemp : s.Nonempty
      S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
      hFS : HasSubset.Subset EmptyCollection.emptyCollection S
      hF : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) EmptyCollection.emptyCollec …
      ⊢ Exists fun lb => And (Membership.mem S lb) (∀ (s : Set E), Membership.mem Em …
    -/
  · exact ⟨s, ⟨hsnemp, hscomp.isClosed, IsExtreme.rfl⟩, fun _ => False.elim⟩
    /-
      🎉 no goals
    -/
  refine ⟨⋂₀ F, ⟨?_, isClosed_sInter fun t ht => (hFS ht).2.1,
    isExtreme_sInter hFnemp fun t ht => (hFS ht).2.2⟩, fun t ht => sInter_subset_of_mem ht⟩
  /-
    case inr
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hsnemp : s.Nonempty
    S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
    F : Set (Set E)
    hFS : HasSubset.Subset F S
    hF : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) F
    hFnemp : F.Nonempty
    ⊢ F.sInter.Nonempty
  -/
  haveI : Nonempty (↥F) := hFnemp.to_subtype
  /-
    case inr
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hsnemp : s.Nonempty
    S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
    F : Set (Set E)
    hFS : HasSubset.Subset F S
    hF : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) F
    hFnemp : F.Nonempty
    this : Nonempty ↑F
    ⊢ F.sInter.Nonempty
  -/
  rw [sInter_eq_iInter]
  refine IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed _ (fun t u => ?_)
    (fun t => (hFS t.mem).1)
    (fun t => hscomp.of_isClosed_subset (hFS t.mem).2.1 (hFS t.mem).2.2.1) fun t =>
      (hFS t.mem).2.1
  /-
    case inr
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hsnemp : s.Nonempty
    S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
    F : Set (Set E)
    hFS : HasSubset.Subset F S
    hF : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) F
    hFnemp : F.Nonempty
    this : Nonempty ↑F
    t u : ↑F
    ⊢ Exists fun z => And ((fun x1 x2 => Superset x1 x2) ↑t ↑z) ((fun x1 x2 => Sup …
  -/
  obtain htu | hut := hF.total t.mem u.mem
  /-
    case inr.inl
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hsnemp : s.Nonempty
    S : Set (Set E) := setOf fun t => And t.Nonempty (And (IsClosed t) (IsExtreme  …
    F : Set (Set E)
    hFS : HasSubset.Subset F S
    hF : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) F
    hFnemp : F.Nonempty
    this : Nonempty ↑F
    t u : ↑F
    htu : HasSubset.Subset ↑t ↑u
    ⊢ Exists fun z => And ((fun x1 x2 => Superset x1 x2) ↑t ↑z) ((fun x1 x2 => Sup …
  -/
  exacts [⟨t, Subset.rfl, htu⟩, ⟨u, hut, Subset.rfl⟩]
  /-
    🎉 no goals
  -/


/-- **Krein-Milman theorem**: In a LCTVS, any compact convex set is the closure of the convex hull
    of its extreme points. -/
theorem closure_convexHull_extremePoints (hscomp : IsCompact s) (hAconv : Convex ℝ s) :
    closure (convexHull ℝ <| s.extremePoints ℝ) = s := by
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hAconv : Convex Real s
    ⊢ Eq (closure ((convexHull Real) (Set.extremePoints Real s))) s
  -/
  apply (closure_minimal (convexHull_min extremePoints_subset hAconv) hscomp.isClosed).antisymm
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hAconv : Convex Real s
    ⊢ HasSubset.Subset s (closure ((convexHull Real) (Set.extremePoints Real s)))
  -/
  by_contra hs
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hAconv : Convex Real s
    hs : Not (HasSubset.Subset s (closure ((convexHull Real) (Set.extremePoints Re …
    ⊢ False
  -/
  obtain ⟨x, hxA, hxt⟩ := not_subset.1 hs
  obtain ⟨l, r, hlr, hrx⟩ :=
    geometric_hahn_banach_closed_point (convex_convexHull _ _).closure isClosed_closure hxt
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hAconv : Convex Real s
    hs : Not (HasSubset.Subset s (closure ((convexHull Real) (Set.extremePoints Re …
    x : E
    hxA : Membership.mem s x
    hxt : Not (Membership.mem (closure ((convexHull Real) (Set.extremePoints Real  …
    l : ContinuousLinearMap (RingHom.id Real) E Real
    r : Real
    hlr : ∀ (a : E), Membership.mem (closure ((convexHull Real) (Set.extremePoints …
    hrx : LT.lt r (l x)
    ⊢ False
  -/
  have h : IsExposed ℝ s ({ y ∈ s | ∀ z ∈ s, l z ≤ l y }) := fun _ => ⟨l, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hAconv : Convex Real s
    hs : Not (HasSubset.Subset s (closure ((convexHull Real) (Set.extremePoints Re …
    x : E
    hxA : Membership.mem s x
    hxt : Not (Membership.mem (closure ((convexHull Real) (Set.extremePoints Real  …
    l : ContinuousLinearMap (RingHom.id Real) E Real
    r : Real
    hlr : ∀ (a : E), Membership.mem (closure ((convexHull Real) (Set.extremePoints …
    hrx : LT.lt r (l x)
    h : IsExposed Real s (setOf fun y => And (Membership.mem s y) (∀ (z : E), Memb …
    ⊢ False
  -/
  obtain ⟨z, hzA, hz⟩ := hscomp.exists_isMaxOn ⟨x, hxA⟩ l.continuous.continuousOn
  /-
    case intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hscomp : IsCompact s
    hAconv : Convex Real s
    hs : Not (HasSubset.Subset s (closure ((convexHull Real) (Set.extremePoints Re …
    x : E
    hxA : Membership.mem s x
    hxt : Not (Membership.mem (closure ((convexHull Real) (Set.extremePoints Real  …
    l : ContinuousLinearMap (RingHom.id Real) E Real
    r : Real
    hlr : ∀ (a : E), Membership.mem (closure ((convexHull Real) (Set.extremePoints …
    hrx : LT.lt r (l x)
    h : IsExposed Real s (setOf fun y => And (Membership.mem s y) (∀ (z : E), Memb …
    z : E
    hzA : Membership.mem s z
    hz : IsMaxOn (⇑l) s z
    ⊢ False
  -/
  obtain ⟨y, hy⟩ := (h.isCompact hscomp).extremePoints_nonempty ⟨z, hzA, hz⟩
  linarith [hlr _ (subset_closure <| subset_convexHull _ _ <|
    h.isExtreme.extremePoints_subset_extremePoints hy), hy.1.2 x hxA]


/-- A continuous affine map is surjective from the extreme points of a compact set to the extreme
points of the image of that set. This inclusion is in general strict. -/
lemma surjOn_extremePoints_image (f : E →ᴬ[ℝ] F) (hs : IsCompact s) :
    SurjOn f (extremePoints ℝ s) (extremePoints ℝ (f '' s)) := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module Real E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : T2Space E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul Real E
    inst✝⁴ : LocallyConvexSpace Real E
    s : Set E
    inst✝³ : AddCommGroup F
    inst✝² : Module Real F
    inst✝¹ : TopologicalSpace F
    inst✝ : T1Space F
    f : ContinuousAffineMap Real E F
    hs : IsCompact s
    ⊢ Set.SurjOn (⇑f) (Set.extremePoints Real s) (Set.extremePoints Real (Set.imag …
  -/
  rintro w hw
  -- The fiber of `w` is nonempty and compact
  have ht : IsCompact {x ∈ s | f x = w} :=
    hs.inter_right <| isClosed_singleton.preimage f.continuous
  /-
    E : Type u_1
    F : Type u_2
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module Real E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : T2Space E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul Real E
    inst✝⁴ : LocallyConvexSpace Real E
    s : Set E
    inst✝³ : AddCommGroup F
    inst✝² : Module Real F
    inst✝¹ : TopologicalSpace F
    inst✝ : T1Space F
    f : ContinuousAffineMap Real E F
    hs : IsCompact s
    w : F
    hw : Membership.mem (Set.extremePoints Real (Set.image (⇑f) s)) w
    ht : IsCompact (setOf fun x => And (Membership.mem s x) (Eq (f x) w))
    ⊢ Membership.mem (Set.image (⇑f) (Set.extremePoints Real s)) w
  -/
  have ht₀ : {x ∈ s | f x = w}.Nonempty := by simpa using extremePoints_subset hw
  -- Hence by the Krein-Milman lemma it has an extreme point `x`
  /-
    E : Type u_1
    F : Type u_2
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module Real E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : T2Space E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul Real E
    inst✝⁴ : LocallyConvexSpace Real E
    s : Set E
    inst✝³ : AddCommGroup F
    inst✝² : Module Real F
    inst✝¹ : TopologicalSpace F
    inst✝ : T1Space F
    f : ContinuousAffineMap Real E F
    hs : IsCompact s
    w : F
    hw : Membership.mem (Set.extremePoints Real (Set.image (⇑f) s)) w
    ht : IsCompact (setOf fun x => And (Membership.mem s x) (Eq (f x) w))
    ht₀ : (setOf fun x => And (Membership.mem s x) (Eq (f x) w)).Nonempty
    ⊢ Membership.mem (Set.image (⇑f) (Set.extremePoints Real s)) w
  -/
  obtain ⟨x, ⟨hx, rfl⟩, hyt⟩ := ht.extremePoints_nonempty ht₀
  -- `f x = w` and `x` is an extreme point of `s`, so we're done
  /-
    case intro.intro.intro
    E : Type u_1
    F : Type u_2
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module Real E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : T2Space E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul Real E
    inst✝⁴ : LocallyConvexSpace Real E
    s : Set E
    inst✝³ : AddCommGroup F
    inst✝² : Module Real F
    inst✝¹ : TopologicalSpace F
    inst✝ : T1Space F
    f : ContinuousAffineMap Real E F
    hs : IsCompact s
    x : E
    hx : Membership.mem s x
    hw : Membership.mem (Set.extremePoints Real (Set.image (⇑f) s)) (f x)
    ht : IsCompact (setOf fun x_1 => And (Membership.mem s x_1) (Eq (f x_1) (f x)))
    ht₀ : (setOf fun x_1 => And (Membership.mem s x_1) (Eq (f x_1) (f x))).Nonempty
    hyt : ∀ ⦃x₁ : E⦄, Membership.mem (setOf fun x_1 => And (Membership.mem s x_1)  …
    ⊢ Membership.mem (Set.image (⇑f) (Set.extremePoints Real s)) (f x)
  -/
  refine mem_image_of_mem _ ⟨hx, fun y hy z hz hxyz ↦ ?_⟩
  /-
    case intro.intro.intro
    E : Type u_1
    F : Type u_2
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module Real E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : T2Space E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul Real E
    inst✝⁴ : LocallyConvexSpace Real E
    s : Set E
    inst✝³ : AddCommGroup F
    inst✝² : Module Real F
    inst✝¹ : TopologicalSpace F
    inst✝ : T1Space F
    f : ContinuousAffineMap Real E F
    hs : IsCompact s
    x : E
    hx : Membership.mem s x
    hw : Membership.mem (Set.extremePoints Real (Set.image (⇑f) s)) (f x)
    ht : IsCompact (setOf fun x_1 => And (Membership.mem s x_1) (Eq (f x_1) (f x)))
    ht₀ : (setOf fun x_1 => And (Membership.mem s x_1) (Eq (f x_1) (f x))).Nonempty
    hyt : ∀ ⦃x₁ : E⦄, Membership.mem (setOf fun x_1 => And (Membership.mem s x_1)  …
    y : E
    hy : Membership.mem s y
    z : E
    hz : Membership.mem s z
    hxyz : Membership.mem (openSegment Real y z) x
    ⊢ And (Eq y x) (Eq z x)
  -/
  have := by simpa using image_openSegment _ f.toAffineMap y z
  have := hw.2 (mem_image_of_mem _ hy) (mem_image_of_mem _ hz) <| by
    rw [← this]; exact mem_image_of_mem _ hxyz
  /-
    case intro.intro.intro
    E : Type u_1
    F : Type u_2
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module Real E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : T2Space E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul Real E
    inst✝⁴ : LocallyConvexSpace Real E
    s : Set E
    inst✝³ : AddCommGroup F
    inst✝² : Module Real F
    inst✝¹ : TopologicalSpace F
    inst✝ : T1Space F
    f : ContinuousAffineMap Real E F
    hs : IsCompact s
    x : E
    hx : Membership.mem s x
    hw : Membership.mem (Set.extremePoints Real (Set.image (⇑f) s)) (f x)
    ht : IsCompact (setOf fun x_1 => And (Membership.mem s x_1) (Eq (f x_1) (f x)))
    ht₀ : (setOf fun x_1 => And (Membership.mem s x_1) (Eq (f x_1) (f x))).Nonempty
    hyt : ∀ ⦃x₁ : E⦄, Membership.mem (setOf fun x_1 => And (Membership.mem s x_1)  …
    y : E
    hy : Membership.mem s y
    z : E
    hz : Membership.mem s z
    hxyz : Membership.mem (openSegment Real y z) x
    this✝ : Eq (Set.image (fun a => f a) (openSegment Real y z)) (openSegment Real …
    this : And (Eq (f y) (f x)) (Eq (f z) (f x))
    ⊢ And (Eq y x) (Eq z x)
  -/
  exact hyt ⟨hy, this.1⟩ ⟨hz, this.2⟩ hxyz
  /-
    🎉 no goals
  -/

