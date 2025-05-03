/-- A set is absolutely convex if it is balanced and convex. Mathlib's definition of `Convex`
requires the scalars to be an `OrderedSemiring` whereas the definition of `Balanced` requires the
scalars to be a `SeminormedRing`. Mathlib doesn't currently have a concept of a semi-normed ordered
ring, so we define a set as `AbsConvex` if it is balanced over a `SeminormedRing` `𝕜` and convex
over `ℝ`. -/
def AbsConvex (s : Set E) : Prop := Balanced 𝕜 s ∧ Convex ℝ s


theorem AbsConvex.empty : AbsConvex 𝕜 (∅ : Set E) := ⟨balanced_empty, convex_empty⟩


theorem AbsConvex.univ : AbsConvex 𝕜 (univ : Set E) := ⟨balanced_univ, convex_univ⟩


theorem AbsConvex.inter {s t : Set E} (hs : AbsConvex 𝕜 s) (ht : AbsConvex 𝕜 t) :
    AbsConvex 𝕜 (s ∩ t) := ⟨hs.1.inter ht.1, hs.2.inter ht.2⟩


theorem AbsConvex.sInter {S : Set (Set E)} (h : ∀ s ∈ S, AbsConvex 𝕜 s) : AbsConvex 𝕜 (⋂₀ S) :=
  ⟨.sInter fun s hs => (h s hs).1, convex_sInter fun s hs => (h s hs).2⟩


theorem AbsConvex.iInter {ι : Sort*} {s : ι → Set E} (h : ∀ i, AbsConvex 𝕜 (s i)) :
    AbsConvex 𝕜 (⋂ i, s i) :=
  sInter_range s ▸ AbsConvex.sInter <| forall_mem_range.2 h


/-- The absolute convex hull of a set `s` is the minimal absolute convex set that includes `s`. -/
@[simps! isClosed]
def absConvexHull : ClosureOperator (Set E) :=
  .ofCompletePred (AbsConvex 𝕜) fun _ ↦ .sInter


theorem subset_absConvexHull : s ⊆ absConvexHull 𝕜 s :=
  (absConvexHull 𝕜).le_closure s


theorem absConvex_absConvexHull : AbsConvex 𝕜 (absConvexHull 𝕜 s) :=
  (absConvexHull 𝕜).isClosed_closure s


theorem balanced_absConvexHull : Balanced 𝕜 (absConvexHull 𝕜 s) :=
  absConvex_absConvexHull.1


theorem convex_absConvexHull : Convex ℝ (absConvexHull 𝕜 s) :=
  absConvex_absConvexHull.2


variable (𝕜 s) in
theorem absConvexHull_eq_iInter :
    absConvexHull 𝕜 s = ⋂ (t : Set E) (_ : s ⊆ t) (_ : AbsConvex 𝕜 t), t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : SMul Real E
    inst✝ : AddCommMonoid E
    s : Set E
    ⊢ Eq ((absConvexHull 𝕜) s) (Set.iInter fun t => Set.iInter fun x => Set.iInter …
  -/
  simp [absConvexHull, iInter_subtype, iInter_and]
  /-
    🎉 no goals
  -/


theorem mem_absConvexHull_iff : x ∈ absConvexHull 𝕜 s ↔ ∀ t, s ⊆ t → AbsConvex 𝕜 t → x ∈ t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : SMul Real E
    inst✝ : AddCommMonoid E
    s : Set E
    x : E
    ⊢ Iff (Membership.mem ((absConvexHull 𝕜) s) x) (∀ (t : Set E), HasSubset.Subse …
  -/
  simp_rw [absConvexHull_eq_iInter, mem_iInter]
  /-
    🎉 no goals
  -/


theorem absConvexHull_min : s ⊆ t → AbsConvex 𝕜 t → absConvexHull 𝕜 s ⊆ t :=
  (absConvexHull 𝕜).closure_min


theorem AbsConvex.absConvexHull_subset_iff (ht : AbsConvex 𝕜 t) : absConvexHull 𝕜 s ⊆ t ↔ s ⊆ t :=
  (show (absConvexHull 𝕜).IsClosed t from ht).closure_le_iff


@[mono, gcongr]
theorem absConvexHull_mono (hst : s ⊆ t) : absConvexHull 𝕜 s ⊆ absConvexHull 𝕜 t :=
  (absConvexHull 𝕜).monotone hst


lemma absConvexHull_eq_self : absConvexHull 𝕜 s = s ↔ AbsConvex 𝕜 s :=
  (absConvexHull 𝕜).isClosed_iff.symm


alias ⟨_, AbsConvex.absConvexHull_eq⟩ := absConvexHull_eq_self


@[simp]
theorem absConvexHull_univ : absConvexHull 𝕜 (univ : Set E) = univ :=
  ClosureOperator.closure_top (absConvexHull 𝕜)


@[simp]
theorem absConvexHull_empty : absConvexHull 𝕜 (∅ : Set E) = ∅ :=
  AbsConvex.empty.absConvexHull_eq


@[simp]
theorem absConvexHull_eq_empty : absConvexHull 𝕜 s = ∅ ↔ s = ∅ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : SMul Real E
    inst✝ : AddCommMonoid E
    s : Set E
    ⊢ Iff (Eq ((absConvexHull 𝕜) s) EmptyCollection.emptyCollection) (Eq s EmptyCo …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : SeminormedRing 𝕜
      inst✝² : SMul 𝕜 E
      inst✝¹ : SMul Real E
      inst✝ : AddCommMonoid E
      s : Set E
      ⊢ Eq ((absConvexHull 𝕜) s) EmptyCollection.emptyCollection → Eq s EmptyCollect …
    -/
  · intro h
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : SeminormedRing 𝕜
      inst✝² : SMul 𝕜 E
      inst✝¹ : SMul Real E
      inst✝ : AddCommMonoid E
      s : Set E
      h : Eq ((absConvexHull 𝕜) s) EmptyCollection.emptyCollection
      ⊢ Eq s EmptyCollection.emptyCollection
    -/
    rw [← Set.subset_empty_iff, ← h]
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : SeminormedRing 𝕜
      inst✝² : SMul 𝕜 E
      inst✝¹ : SMul Real E
      inst✝ : AddCommMonoid E
      s : Set E
      h : Eq ((absConvexHull 𝕜) s) EmptyCollection.emptyCollection
      ⊢ HasSubset.Subset s ((absConvexHull 𝕜) s)
    -/
    exact subset_absConvexHull
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : SeminormedRing 𝕜
      inst✝² : SMul 𝕜 E
      inst✝¹ : SMul Real E
      inst✝ : AddCommMonoid E
      s : Set E
      ⊢ Eq s EmptyCollection.emptyCollection → Eq ((absConvexHull 𝕜) s) EmptyCollect …
    -/
  · rintro rfl
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : SeminormedRing 𝕜
      inst✝² : SMul 𝕜 E
      inst✝¹ : SMul Real E
      inst✝ : AddCommMonoid E
      ⊢ Eq ((absConvexHull 𝕜) EmptyCollection.emptyCollection) EmptyCollection.empty …
    -/
    exact absConvexHull_empty
    /-
      🎉 no goals
    -/


@[simp]
theorem absConvexHull_nonempty : (absConvexHull 𝕜 s).Nonempty ↔ s.Nonempty := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : SMul Real E
    inst✝ : AddCommMonoid E
    s : Set E
    ⊢ Iff ((absConvexHull 𝕜) s).Nonempty s.Nonempty
  -/
  rw [nonempty_iff_ne_empty, nonempty_iff_ne_empty, Ne, Ne]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : SeminormedRing 𝕜
    inst✝² : SMul 𝕜 E
    inst✝¹ : SMul Real E
    inst✝ : AddCommMonoid E
    s : Set E
    ⊢ Iff (Not (Eq ((absConvexHull 𝕜) s) EmptyCollection.emptyCollection)) (Not (E …
  -/
  exact not_congr absConvexHull_eq_empty
  /-
    🎉 no goals
  -/


protected alias ⟨_, Set.Nonempty.absConvexHull⟩ := absConvexHull_nonempty


theorem absConvex_closed_sInter {S : Set (Set E)} (h : ∀ s ∈ S, AbsConvex 𝕜 s ∧ IsClosed s) :
    AbsConvex 𝕜 (⋂₀ S) ∧ IsClosed (⋂₀ S) :=
  ⟨AbsConvex.sInter (fun s hs => (h s hs).1), isClosed_sInter fun _ hs => (h _ hs).2⟩


/-- The absolutely convex closed hull of a set `s` is the minimal absolutely convex closed set that
includes `s`. -/
@[simps! isClosed]
def closedAbsConvexHull : ClosureOperator (Set E) :=
  .ofCompletePred (fun s => AbsConvex 𝕜 s ∧ IsClosed s) fun _ ↦ absConvex_closed_sInter


theorem absConvex_convexClosedHull {s : Set E} :
    AbsConvex 𝕜 (closedAbsConvexHull 𝕜 s) := ((closedAbsConvexHull 𝕜).isClosed_closure s).1


theorem isClosed_closedAbsConvexHull {s : Set E} :
    IsClosed (closedAbsConvexHull 𝕜 s) := ((closedAbsConvexHull 𝕜).isClosed_closure s).2


theorem subset_closedAbsConvexHull {s : Set E} : s ⊆ closedAbsConvexHull 𝕜 s :=
  (closedAbsConvexHull 𝕜).le_closure s


theorem closure_subset_closedAbsConvexHull {s : Set E} : closure s ⊆ closedAbsConvexHull 𝕜 s :=
  closure_minimal subset_closedAbsConvexHull isClosed_closedAbsConvexHull


theorem closedAbsConvexHull_min {s t : Set E} (hst : s ⊆ t) (h_conv : AbsConvex 𝕜 t)
    (h_closed : IsClosed t) : closedAbsConvexHull 𝕜 s ⊆ t :=
  (closedAbsConvexHull 𝕜).closure_min hst ⟨h_conv, h_closed⟩


theorem absConvexHull_subset_closedAbsConvexHull {s : Set E} :
    (absConvexHull 𝕜) s ⊆ (closedAbsConvexHull 𝕜) s :=
  absConvexHull_min subset_closedAbsConvexHull absConvex_convexClosedHull


@[simp]
theorem closedAbsConvexHull_closure_eq_closedAbsConvexHull {s : Set E} :
    closedAbsConvexHull 𝕜 (closure s) = closedAbsConvexHull 𝕜 s :=
  subset_antisymm (by simpa using ((closedAbsConvexHull 𝕜).monotone
      (closure_subset_closedAbsConvexHull (𝕜 := 𝕜) (E := E))))
    ((closedAbsConvexHull 𝕜).monotone subset_closure)


theorem AbsConvex.closure {s : Set E} (hs : AbsConvex 𝕜 s) : AbsConvex 𝕜 (closure s) :=
  ⟨Balanced.closure hs.1, Convex.closure hs.2⟩


theorem closedAbsConvexHull_eq_closure_absConvexHull {s : Set E} :
    closedAbsConvexHull 𝕜 s = closure (absConvexHull 𝕜 s) := subset_antisymm
  (closedAbsConvexHull_min (subset_trans (subset_absConvexHull) subset_closure)
    (AbsConvex.closure absConvex_absConvexHull) isClosed_closure)
  (closure_minimal absConvexHull_subset_closedAbsConvexHull isClosed_closedAbsConvexHull)


theorem nhds_hasBasis_absConvex :
    (𝓝 (0 : E)).HasBasis (fun s : Set E => s ∈ 𝓝 (0 : E) ∧ AbsConvex 𝕜 s) id := by
  refine
    (LocallyConvexSpace.convex_basis_zero ℝ E).to_hasBasis (fun s hs => ?_) fun s hs =>
      ⟨s, ⟨hs.1, hs.2.2⟩, rfl.subset⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : Module Real E
    inst✝³ : SMulCommClass Real 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : And (Membership.mem (nhds 0) s) (Convex Real s)
    ⊢ Exists fun i' => And (And (Membership.mem (nhds 0) i') (AbsConvex 𝕜 i')) (Ha …
  -/
  refine ⟨convexHull ℝ (balancedCore 𝕜 s), ?_, convexHull_min (balancedCore_subset s) hs.2⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : Module Real E
    inst✝³ : SMulCommClass Real 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : And (Membership.mem (nhds 0) s) (Convex Real s)
    ⊢ And (Membership.mem (nhds 0) ((convexHull Real) (balancedCore 𝕜 s))) (AbsCon …
  -/
  refine ⟨Filter.mem_of_superset (balancedCore_mem_nhds_zero hs.1) (subset_convexHull ℝ _), ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : Module Real E
    inst✝³ : SMulCommClass Real 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : And (Membership.mem (nhds 0) s) (Convex Real s)
    ⊢ AbsConvex 𝕜 ((convexHull Real) (balancedCore 𝕜 s))
  -/
  refine ⟨(balancedCore_balanced s).convexHull, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : Module Real E
    inst✝³ : SMulCommClass Real 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : And (Membership.mem (nhds 0) s) (Convex Real s)
    ⊢ Convex Real ((convexHull Real) (balancedCore 𝕜 s))
  -/
  exact convex_convexHull ℝ (balancedCore 𝕜 s)
  /-
    🎉 no goals
  -/


theorem nhds_hasBasis_absConvex_open :
    (𝓝 (0 : E)).HasBasis (fun s => (0 : E) ∈ s ∧ IsOpen s ∧ AbsConvex 𝕜 s) id := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : SMulCommClass Real 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : LocallyConvexSpace Real E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : TopologicalAddGroup E
    ⊢ (nhds 0).HasBasis (fun s => And (Membership.mem s 0) (And (IsOpen s) (AbsCon …
  -/
  refine (nhds_hasBasis_absConvex 𝕜 E).to_hasBasis ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module Real E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : LocallyConvexSpace Real E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : TopologicalAddGroup E
      ⊢ ∀ (i : Set E), And (Membership.mem (nhds 0) i) (AbsConvex 𝕜 i) → Exists fun  …
    -/
  · rintro s ⟨hs_nhds, hs_balanced, hs_convex⟩
    /-
      case refine_1.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module Real E
      inst✝⁵ : SMulCommClass Real 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : LocallyConvexSpace Real E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : TopologicalAddGroup E
      s : Set E
      hs_nhds : Membership.mem (nhds 0) s
      hs_balanced : Balanced 𝕜 s
      hs_convex : Convex Real s
      ⊢ Exists fun i' => And (And (Membership.mem i' 0) (And (IsOpen i') (AbsConvex  …
    -/
    refine ⟨interior s, ?_, interior_subset⟩
    exact
      ⟨mem_interior_iff_mem_nhds.mpr hs_nhds, isOpen_interior,
        hs_balanced.interior (mem_interior_iff_mem_nhds.mpr hs_nhds), hs_convex.interior⟩
  /-
    case refine_2
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : SMulCommClass Real 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : LocallyConvexSpace Real E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : TopologicalAddGroup E
    ⊢ ∀ (i' : Set E), And (Membership.mem i' 0) (And (IsOpen i') (AbsConvex 𝕜 i')) …
  -/
  rintro s ⟨hs_zero, hs_open, hs_balanced, hs_convex⟩
  /-
    case refine_2.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : SMulCommClass Real 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : LocallyConvexSpace Real E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : TopologicalAddGroup E
    s : Set E
    hs_zero : Membership.mem s 0
    hs_open : IsOpen s
    hs_balanced : Balanced 𝕜 s
    hs_convex : Convex Real s
    ⊢ Exists fun i => And (And (Membership.mem (nhds 0) i) (AbsConvex 𝕜 i)) (HasSu …
  -/
  exact ⟨s, ⟨hs_open.mem_nhds hs_zero, hs_balanced, hs_convex⟩, rfl.subset⟩
  /-
    🎉 no goals
  -/


theorem absConvexHull_add_subset {s t : Set E} :
    absConvexHull 𝕜 (s + t) ⊆ absConvexHull 𝕜 s + absConvexHull 𝕜 t :=
  absConvexHull_min (add_subset_add subset_absConvexHull subset_absConvexHull)
    ⟨Balanced.add balanced_absConvexHull balanced_absConvexHull,
      Convex.add convex_absConvexHull convex_absConvexHull⟩


theorem absConvexHull_eq_convexHull_balancedHull [SMulCommClass ℝ 𝕜 E] {s : Set E} :
    absConvexHull 𝕜 s = convexHull ℝ (balancedHull 𝕜 s) := le_antisymm
  (absConvexHull_min
    ((subset_convexHull ℝ s).trans (convexHull_mono (subset_balancedHull 𝕜)))
      ⟨Balanced.convexHull (balancedHull.balanced s), convex_convexHull ..⟩)
  (convexHull_min (balanced_absConvexHull.balancedHull_subset_of_subset subset_absConvexHull)
      convex_absConvexHull)


/-- In general, equality doesn't hold here - e.g. consider `s := {(-1, 1), (1, 1)}` in `ℝ²`. -/
theorem balancedHull_convexHull_subseteq_absConvexHull {s : Set E} :
    balancedHull 𝕜 (convexHull ℝ s) ⊆ absConvexHull 𝕜 s :=
  balanced_absConvexHull.balancedHull_subset_of_subset
    (convexHull_min subset_absConvexHull convex_absConvexHull)


lemma balancedHull_subset_convexHull_union_neg {s : Set E} :
    balancedHull ℝ s ⊆ convexHull ℝ (s ∪ -s) := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    ⊢ HasSubset.Subset (balancedHull Real s) ((convexHull Real) (Union.union s (Ne …
  -/
  intro a ha
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    a : E
    ha : Membership.mem (balancedHull Real s) a
    ⊢ Membership.mem ((convexHull Real) (Union.union s (Neg.neg s))) a
  -/
  obtain ⟨r, hr, y, hy, rfl⟩ := mem_balancedHull_iff.1 ha
  /-
    case intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    r : Real
    hr : LE.le (Norm.norm r) 1
    y : E
    hy : Membership.mem s y
    ha : Membership.mem (balancedHull Real s) ((fun x => HSMul.hSMul r x) y)
    ⊢ Membership.mem ((convexHull Real) (Union.union s (Neg.neg s))) ((fun x => HS …
  -/
  apply segment_subset_convexHull (mem_union_left (-s) hy) (mem_union_right _ (neg_mem_neg.mpr hy))
  /-
    case intro.intro.intro.intro.a
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    r : Real
    hr : LE.le (Norm.norm r) 1
    y : E
    hy : Membership.mem s y
    ha : Membership.mem (balancedHull Real s) ((fun x => HSMul.hSMul r x) y)
    ⊢ Membership.mem (segment Real y (Neg.neg y)) ((fun x => HSMul.hSMul r x) y)
  -/
  have : 0 ≤ 1 + r := neg_le_iff_add_nonneg'.mp (neg_le_of_abs_le hr)
  /-
    case intro.intro.intro.intro.a
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    r : Real
    hr : LE.le (Norm.norm r) 1
    y : E
    hy : Membership.mem s y
    ha : Membership.mem (balancedHull Real s) ((fun x => HSMul.hSMul r x) y)
    this : LE.le 0 (HAdd.hAdd 1 r)
    ⊢ Membership.mem (segment Real y (Neg.neg y)) ((fun x => HSMul.hSMul r x) y)
  -/
  have : 0 ≤ 1 - r := sub_nonneg.2 (le_of_abs_le hr)
  /-
    case intro.intro.intro.intro.a
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    r : Real
    hr : LE.le (Norm.norm r) 1
    y : E
    hy : Membership.mem s y
    ha : Membership.mem (balancedHull Real s) ((fun x => HSMul.hSMul r x) y)
    this✝ : LE.le 0 (HAdd.hAdd 1 r)
    this : LE.le 0 (HSub.hSub 1 r)
    ⊢ Membership.mem (segment Real y (Neg.neg y)) ((fun x => HSMul.hSMul r x) y)
  -/
  refine ⟨(1 + r)/2, (1 - r)/2, by positivity, by positivity, by ring, ?_⟩
  /-
    case intro.intro.intro.intro.a
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    r : Real
    hr : LE.le (Norm.norm r) 1
    y : E
    hy : Membership.mem s y
    ha : Membership.mem (balancedHull Real s) ((fun x => HSMul.hSMul r x) y)
    this✝ : LE.le 0 (HAdd.hAdd 1 r)
    this : LE.le 0 (HSub.hSub 1 r)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HAdd.hAdd 1 r) 2) y) (HSMul.hSMul (HD …
  -/
  rw [smul_neg, ← sub_eq_add_neg, ← sub_smul]
  /-
    case intro.intro.intro.intro.a
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    r : Real
    hr : LE.le (Norm.norm r) 1
    y : E
    hy : Membership.mem s y
    ha : Membership.mem (balancedHull Real s) ((fun x => HSMul.hSMul r x) y)
    this✝ : LE.le 0 (HAdd.hAdd 1 r)
    this : LE.le 0 (HSub.hSub 1 r)
    ⊢ Eq (HSMul.hSMul (HSub.hSub (HDiv.hDiv (HAdd.hAdd 1 r) 2) (HDiv.hDiv (HSub.hS …
  -/
  apply congrFun (congrArg HSMul.hSMul _) y
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    r : Real
    hr : LE.le (Norm.norm r) 1
    y : E
    hy : Membership.mem s y
    ha : Membership.mem (balancedHull Real s) ((fun x => HSMul.hSMul r x) y)
    this✝ : LE.le 0 (HAdd.hAdd 1 r)
    this : LE.le 0 (HSub.hSub 1 r)
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HAdd.hAdd 1 r) 2) (HDiv.hDiv (HSub.hSub 1 r) 2)) r
  -/
  ring_nf
  /-
    🎉 no goals
  -/


@[simp]
theorem convexHull_union_neg_eq_absConvexHull {s : Set E} :
    convexHull ℝ (s ∪ -s) = absConvexHull ℝ s := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    ⊢ Eq ((convexHull Real) (Union.union s (Neg.neg s))) ((absConvexHull Real) s)
  -/
  rw [absConvexHull_eq_convexHull_balancedHull]
  exact le_antisymm (convexHull_mono (union_subset (subset_balancedHull ℝ)
    (fun _ _ => by rw [mem_balancedHull_iff]; use -1; aesop)))
    (by
      rw [← Convex.convexHull_eq (convex_convexHull ℝ (s ∪ -s))]
      exact convexHull_mono balancedHull_subset_convexHull_union_neg)


theorem totallyBounded_absConvexHull (hs : TotallyBounded s) :
    TotallyBounded (absConvexHull ℝ s) := by
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    ⊢ TotallyBounded ((absConvexHull Real) s)
  -/
  rw [← convexHull_union_neg_eq_absConvexHull]
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    ⊢ TotallyBounded ((convexHull Real) (Union.union s (Neg.neg s)))
  -/
  apply totallyBounded_convexHull
  /-
    case hs
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    ⊢ TotallyBounded (Union.union s (Neg.neg s))
  -/
  rw [totallyBounded_union]
  /-
    case hs
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    lcs : LocallyConvexSpace Real E
    inst✝ : ContinuousSMul Real E
    hs : TotallyBounded s
    ⊢ And (TotallyBounded s) (TotallyBounded (Neg.neg s))
  -/
  exact ⟨hs, totallyBounded_neg hs⟩
  /-
    🎉 no goals
  -/


/-- The type of absolutely convex open sets. -/
def AbsConvexOpenSets :=
  { s : Set E // (0 : E) ∈ s ∧ IsOpen s ∧ AbsConvex 𝕜 s }


noncomputable instance AbsConvexOpenSets.instCoeTC : CoeTC (AbsConvexOpenSets 𝕜 E) (Set E) :=
  ⟨Subtype.val⟩


theorem coe_zero_mem (s : AbsConvexOpenSets 𝕜 E) : (0 : E) ∈ (s : Set E) :=
  s.2.1


theorem coe_isOpen (s : AbsConvexOpenSets 𝕜 E) : IsOpen (s : Set E) :=
  s.2.2.1


theorem coe_nhds (s : AbsConvexOpenSets 𝕜 E) : (s : Set E) ∈ 𝓝 (0 : E) :=
  s.coe_isOpen.mem_nhds s.coe_zero_mem


theorem coe_balanced (s : AbsConvexOpenSets 𝕜 E) : Balanced 𝕜 (s : Set E) :=
  s.2.2.2.1


theorem coe_convex (s : AbsConvexOpenSets 𝕜 E) : Convex ℝ (s : Set E) :=
  s.2.2.2.2


instance AbsConvexOpenSets.instNonempty : Nonempty (AbsConvexOpenSets 𝕜 E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Zero E
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : SMul 𝕜 E
    inst✝ : SMul Real E
    ⊢ Nonempty (AbsConvexOpenSets 𝕜 E)
  -/
  rw [← exists_true_iff_nonempty]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Zero E
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : SMul 𝕜 E
    inst✝ : SMul Real E
    ⊢ Exists fun x => True
  -/
  dsimp only [AbsConvexOpenSets]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Zero E
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : SMul 𝕜 E
    inst✝ : SMul Real E
    ⊢ Exists fun x => True
  -/
  rw [Subtype.exists]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Zero E
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : SMul 𝕜 E
    inst✝ : SMul Real E
    ⊢ Exists fun a => Exists fun b => True
  -/
  exact ⟨Set.univ, ⟨mem_univ 0, isOpen_univ, balanced_univ, convex_univ⟩, trivial⟩
  /-
    🎉 no goals
  -/


/-- The family of seminorms defined by the gauges of absolute convex open sets. -/
noncomputable def gaugeSeminormFamily : SeminormFamily 𝕜 E (AbsConvexOpenSets 𝕜 E) := fun s =>
  gaugeSeminorm s.coe_balanced s.coe_convex (absorbent_nhds_zero s.coe_nhds)


theorem gaugeSeminormFamily_ball (s : AbsConvexOpenSets 𝕜 E) :
    (gaugeSeminormFamily 𝕜 E s).ball 0 1 = (s : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : Module Real E
    inst✝¹ : IsScalarTower Real 𝕜 E
    inst✝ : ContinuousSMul Real E
    s : AbsConvexOpenSets 𝕜 E
    ⊢ Eq ((gaugeSeminormFamily 𝕜 E s).ball 0 1) ↑s
  -/
  dsimp only [gaugeSeminormFamily]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : Module Real E
    inst✝¹ : IsScalarTower Real 𝕜 E
    inst✝ : ContinuousSMul Real E
    s : AbsConvexOpenSets 𝕜 E
    ⊢ Eq ((gaugeSeminorm ⋯ ⋯ ⋯).ball 0 1) ↑s
  -/
  rw [Seminorm.ball_zero_eq]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : Module Real E
    inst✝¹ : IsScalarTower Real 𝕜 E
    inst✝ : ContinuousSMul Real E
    s : AbsConvexOpenSets 𝕜 E
    ⊢ Eq (setOf fun y => LT.lt ((gaugeSeminorm ⋯ ⋯ ⋯) y) 1) ↑s
  -/
  simp_rw [gaugeSeminorm_toFun]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : Module 𝕜 E
    inst✝² : Module Real E
    inst✝¹ : IsScalarTower Real 𝕜 E
    inst✝ : ContinuousSMul Real E
    s : AbsConvexOpenSets 𝕜 E
    ⊢ Eq (setOf fun y => LT.lt (gauge (↑s) y) 1) ↑s
  -/
  exact gauge_lt_one_eq_self_of_isOpen s.coe_convex s.coe_zero_mem s.coe_isOpen
  /-
    🎉 no goals
  -/


/-- The topology of a locally convex space is induced by the gauge seminorm family. -/
theorem with_gaugeSeminormFamily : WithSeminorms (gaugeSeminormFamily 𝕜 E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    ⊢ WithSeminorms (gaugeSeminormFamily 𝕜 E)
  -/
  refine SeminormFamily.withSeminorms_of_hasBasis _ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    ⊢ (nhds 0).HasBasis (fun s => Membership.mem (gaugeSeminormFamily 𝕜 E).basisSe …
  -/
  refine (nhds_hasBasis_absConvex_open 𝕜 E).to_hasBasis (fun s hs => ?_) fun s hs => ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : TopologicalSpace E
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module Real E
      inst✝⁵ : IsScalarTower Real 𝕜 E
      inst✝⁴ : ContinuousSMul Real E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : SMulCommClass Real 𝕜 E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hs : And (Membership.mem s 0) (And (IsOpen s) (AbsConvex 𝕜 s))
      ⊢ Exists fun i' => And (Membership.mem (gaugeSeminormFamily 𝕜 E).basisSets i') …
    -/
  · refine ⟨s, ⟨?_, rfl.subset⟩⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : TopologicalSpace E
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module Real E
      inst✝⁵ : IsScalarTower Real 𝕜 E
      inst✝⁴ : ContinuousSMul Real E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : SMulCommClass Real 𝕜 E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hs : And (Membership.mem s 0) (And (IsOpen s) (AbsConvex 𝕜 s))
      ⊢ Membership.mem (gaugeSeminormFamily 𝕜 E).basisSets s
    -/
    convert (gaugeSeminormFamily _ _).basisSets_singleton_mem ⟨s, hs⟩ one_pos
    /-
      case h.e'_5
      𝕜 : Type u_1
      E : Type u_2
      inst✝¹⁰ : RCLike 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : TopologicalSpace E
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module Real E
      inst✝⁵ : IsScalarTower Real 𝕜 E
      inst✝⁴ : ContinuousSMul Real E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : SMulCommClass Real 𝕜 E
      inst✝ : LocallyConvexSpace Real E
      s : Set E
      hs : And (Membership.mem s 0) (And (IsOpen s) (AbsConvex 𝕜 s))
      ⊢ Eq s ((gaugeSeminormFamily 𝕜 E ⟨s, hs⟩).ball 0 1)
    -/
    rw [gaugeSeminormFamily_ball, Subtype.coe_mk]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hs : Membership.mem (gaugeSeminormFamily 𝕜 E).basisSets s
    ⊢ Exists fun i => And (And (Membership.mem i 0) (And (IsOpen i) (AbsConvex 𝕜 i …
  -/
  refine ⟨s, ⟨?_, rfl.subset⟩⟩
  /-
    case refine_2
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hs : Membership.mem (gaugeSeminormFamily 𝕜 E).basisSets s
    ⊢ And (Membership.mem s 0) (And (IsOpen s) (AbsConvex 𝕜 s))
  -/
  rw [SeminormFamily.basisSets_iff] at hs
  /-
    case refine_2
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hs : Exists fun i => Exists fun r => And (LT.lt 0 r) (Eq s ((i.sup (gaugeSemin …
    ⊢ And (Membership.mem s 0) (And (IsOpen s) (AbsConvex 𝕜 s))
  -/
  rcases hs with ⟨t, r, hr, rfl⟩
  /-
    case refine_2.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    t : Finset (AbsConvexOpenSets 𝕜 E)
    r : Real
    hr : LT.lt 0 r
    ⊢ And (Membership.mem ((t.sup (gaugeSeminormFamily 𝕜 E)).ball 0 r) 0) (And (Is …
  -/
  rw [Seminorm.ball_finset_sup_eq_iInter _ _ _ hr]
  -- We have to show that the intersection contains zero, is open, balanced, and convex
  refine
    ⟨mem_iInter₂.mpr fun _ _ => by simp [Seminorm.mem_ball_zero, hr],
      isOpen_biInter_finset fun S _ => ?_,
      balanced_iInter₂ fun _ _ => Seminorm.balanced_ball_zero _ _,
      convex_iInter₂ fun _ _ => Seminorm.convex_ball ..⟩
  -- The only nontrivial part is to show that the ball is open
  /-
    case refine_2.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    t : Finset (AbsConvexOpenSets 𝕜 E)
    r : Real
    hr : LT.lt 0 r
    S : AbsConvexOpenSets 𝕜 E
    x✝ : Membership.mem t S
    ⊢ IsOpen ((gaugeSeminormFamily 𝕜 E S).ball 0 r)
  -/
  have hr' : r = ‖(r : 𝕜)‖ * 1 := by simp [abs_of_pos hr]
  /-
    case refine_2.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    t : Finset (AbsConvexOpenSets 𝕜 E)
    r : Real
    hr : LT.lt 0 r
    S : AbsConvexOpenSets 𝕜 E
    x✝ : Membership.mem t S
    hr' : Eq r (HMul.hMul (Norm.norm ↑r) 1)
    ⊢ IsOpen ((gaugeSeminormFamily 𝕜 E S).ball 0 r)
  -/
  have hr'' : (r : 𝕜) ≠ 0 := by simp [hr.ne']
  /-
    case refine_2.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    t : Finset (AbsConvexOpenSets 𝕜 E)
    r : Real
    hr : LT.lt 0 r
    S : AbsConvexOpenSets 𝕜 E
    x✝ : Membership.mem t S
    hr' : Eq r (HMul.hMul (Norm.norm ↑r) 1)
    hr'' : Ne (↑r) 0
    ⊢ IsOpen ((gaugeSeminormFamily 𝕜 E S).ball 0 r)
  -/
  rw [hr', ← Seminorm.smul_ball_zero hr'', gaugeSeminormFamily_ball]
  /-
    case refine_2.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹⁰ : RCLike 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module Real E
    inst✝⁵ : IsScalarTower Real 𝕜 E
    inst✝⁴ : ContinuousSMul Real E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : SMulCommClass Real 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    t : Finset (AbsConvexOpenSets 𝕜 E)
    r : Real
    hr : LT.lt 0 r
    S : AbsConvexOpenSets 𝕜 E
    x✝ : Membership.mem t S
    hr' : Eq r (HMul.hMul (Norm.norm ↑r) 1)
    hr'' : Ne (↑r) 0
    ⊢ IsOpen (HSMul.hSMul ↑r ↑S)
  -/
  exact S.coe_isOpen.smul₀ hr''
  /-
    🎉 no goals
  -/

