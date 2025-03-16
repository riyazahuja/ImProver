/-- A subset of a topological real vector space is ample
if the convex hull of each of its connected components is the full space. -/
def AmpleSet (s : Set F) : Prop :=
  ∀ x ∈ s, convexHull ℝ (connectedComponentIn s x) = univ


/-- A whole vector space is ample. -/
@[simp]
theorem ampleSet_univ {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] :
    AmpleSet (univ : Set F) := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    ⊢ AmpleSet Set.univ
  -/
  intro x _
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x : F
    a✝ : Membership.mem Set.univ x
    ⊢ Eq ((convexHull Real) (connectedComponentIn Set.univ x)) Set.univ
  -/
  rw [connectedComponentIn_univ, PreconnectedSpace.connectedComponent_eq_univ, convexHull_univ]
  /-
    🎉 no goals
  -/


/-- The empty set in a vector space is ample. -/
@[simp]
theorem ampleSet_empty : AmpleSet (∅ : Set F) := fun _ ↦ False.elim


/-- The union of two ample sets is ample. -/
theorem union {s t : Set F} (hs : AmpleSet s) (ht : AmpleSet t) : AmpleSet (s ∪ t) := by
  /-
    F : Type u_1
    inst✝² : AddCommGroup F
    inst✝¹ : Module Real F
    inst✝ : TopologicalSpace F
    s t : Set F
    hs : AmpleSet s
    ht : AmpleSet t
    ⊢ AmpleSet (Union.union s t)
  -/
  intro x hx
  rcases hx with (h | h) <;>
  -- The connected component of `x ∈ s` in `s ∪ t` contains the connected component of `x` in `s`,
  -- hence is also full; similarly for `t`.
  [have hx := hs x h; have hx := ht x h] <;>
  rw [← Set.univ_subset_iff, ← hx] <;>
  apply convexHull_mono <;>
  apply connectedComponentIn_mono <;>
  [apply subset_union_left; apply subset_union_right]


/-- Images of ample sets under continuous affine equivalences are ample. -/
theorem image {s : Set E} (h : AmpleSet s) (L : E ≃ᵃL[ℝ] F) :
    AmpleSet (L '' s) := forall_mem_image.mpr fun x hx ↦
  calc (convexHull ℝ) (connectedComponentIn (L '' s) (L x))
    _ = (convexHull ℝ) (L '' (connectedComponentIn s x)) :=
          .symm <| congrArg _ <| L.toHomeomorph.image_connectedComponentIn hx
    _ = L '' (convexHull ℝ (connectedComponentIn s x)) :=
          .symm <| L.toAffineMap.image_convexHull _
                   /-
                     F : Type u_1
                     inst✝⁵ : AddCommGroup F
                     inst✝⁴ : Module Real F
                     inst✝³ : TopologicalSpace F
                     E : Type u_2
                     inst✝² : AddCommGroup E
                     inst✝¹ : Module Real E
                     inst✝ : TopologicalSpace E
                     s : Set E
                     h : AmpleSet s
                     L : ContinuousAffineEquiv Real E F
                     x : E
                     hx : Membership.mem s x
                     ⊢ Eq (Set.image (⇑L) ((convexHull Real) (connectedComponentIn s x))) Set.univ
                   -/
    _ = univ := by rw [h x hx, image_univ, L.surjective.range_eq]
                   /-
                     🎉 no goals
                   -/


/-- A set is ample iff its image under a continuous affine equivalence is. -/
theorem image_iff {s : Set E} (L : E ≃ᵃL[ℝ] F) :
    AmpleSet (L '' s) ↔ AmpleSet s :=
  ⟨fun h ↦ (L.symm_image_image s) ▸ h.image L.symm, fun h ↦ h.image L⟩


/-- Pre-images of ample sets under continuous affine equivalences are ample. -/
theorem preimage {s : Set F} (h : AmpleSet s) (L : E ≃ᵃL[ℝ] F) : AmpleSet (L ⁻¹' s) := by
  /-
    F : Type u_1
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module Real F
    inst✝³ : TopologicalSpace F
    E : Type u_2
    inst✝² : AddCommGroup E
    inst✝¹ : Module Real E
    inst✝ : TopologicalSpace E
    s : Set F
    h : AmpleSet s
    L : ContinuousAffineEquiv Real E F
    ⊢ AmpleSet (Set.preimage (⇑L) s)
  -/
  rw [← L.image_symm_eq_preimage]
  /-
    F : Type u_1
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : Module Real F
    inst✝³ : TopologicalSpace F
    E : Type u_2
    inst✝² : AddCommGroup E
    inst✝¹ : Module Real E
    inst✝ : TopologicalSpace E
    s : Set F
    h : AmpleSet s
    L : ContinuousAffineEquiv Real E F
    ⊢ AmpleSet (Set.image (⇑L.symm) s)
  -/
  exact h.image L.symm
  /-
    🎉 no goals
  -/


/-- A set is ample iff its pre-image under a continuous affine equivalence is. -/
theorem preimage_iff {s : Set F} (L : E ≃ᵃL[ℝ] F) :
    AmpleSet (L ⁻¹' s) ↔ AmpleSet s :=
  ⟨fun h ↦ L.image_preimage s ▸ h.image L, fun h ↦ h.preimage L⟩


/-- Affine translations of ample sets are ample. -/
theorem vadd [ContinuousAdd E] {s : Set E} (h : AmpleSet s) {y : E} :
    AmpleSet (y +ᵥ s) :=
  h.image (ContinuousAffineEquiv.constVAdd ℝ E y)


/-- A set is ample iff its affine translation is. -/
theorem vadd_iff [ContinuousAdd E] {s : Set E} {y : E} :
    AmpleSet (y +ᵥ s) ↔ AmpleSet s :=
  AmpleSet.image_iff (ContinuousAffineEquiv.constVAdd ℝ E y)


/-- Let `E` be a linear subspace in a real vector space.
If `E` has codimension at least two, its complement is ample. -/
theorem of_one_lt_codim [TopologicalAddGroup F] [ContinuousSMul ℝ F] {E : Submodule ℝ F}
    (hcodim : 1 < Module.rank ℝ (F ⧸ E)) :
    AmpleSet (Eᶜ : Set F) := fun x hx ↦ by
  /-
    F : Type u_1
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module Real F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousSMul Real F
    E : Submodule Real F
    hcodim : LT.lt 1 (Module.rank Real (HasQuotient.Quotient F E))
    x : F
    hx : Membership.mem (HasCompl.compl ↑E) x
    ⊢ Eq ((convexHull Real) (connectedComponentIn (HasCompl.compl ↑E) x)) Set.univ
  -/
  rw [E.connectedComponentIn_eq_self_of_one_lt_codim hcodim hx, eq_univ_iff_forall]
  /-
    F : Type u_1
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module Real F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousSMul Real F
    E : Submodule Real F
    hcodim : LT.lt 1 (Module.rank Real (HasQuotient.Quotient F E))
    x : F
    hx : Membership.mem (HasCompl.compl ↑E) x
    ⊢ ∀ (x : F), Membership.mem ((convexHull Real) (HasCompl.compl ↑E)) x
  -/
  intro y
  /-
    F : Type u_1
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module Real F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousSMul Real F
    E : Submodule Real F
    hcodim : LT.lt 1 (Module.rank Real (HasQuotient.Quotient F E))
    x : F
    hx : Membership.mem (HasCompl.compl ↑E) x
    y : F
    ⊢ Membership.mem ((convexHull Real) (HasCompl.compl ↑E)) y
  -/
  by_cases h : y ∈ E
  · obtain ⟨z, hz⟩ : ∃ z, z ∉ E := by
      rw [← not_forall, ← Submodule.eq_top_iff']
      rintro rfl
      simp [rank_zero_iff.2 inferInstance] at hcodim
    /-
      case pos.intro
      F : Type u_1
      inst✝⁴ : AddCommGroup F
      inst✝³ : Module Real F
      inst✝² : TopologicalSpace F
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousSMul Real F
      E : Submodule Real F
      hcodim : LT.lt 1 (Module.rank Real (HasQuotient.Quotient F E))
      x : F
      hx : Membership.mem (HasCompl.compl ↑E) x
      y : F
      h : Membership.mem E y
      z : F
      hz : Not (Membership.mem E z)
      ⊢ Membership.mem ((convexHull Real) (HasCompl.compl ↑E)) y
    -/
    refine segment_subset_convexHull ?_ ?_ (mem_segment_sub_add y z) <;>
      /-
        case pos.intro.refine_1
        F : Type u_1
        inst✝⁴ : AddCommGroup F
        inst✝³ : Module Real F
        inst✝² : TopologicalSpace F
        inst✝¹ : TopologicalAddGroup F
        inst✝ : ContinuousSMul Real F
        E : Submodule Real F
        hcodim : LT.lt 1 (Module.rank Real (HasQuotient.Quotient F E))
        x : F
        hx : Membership.mem (HasCompl.compl ↑E) x
        y : F
        h : Membership.mem E y
        z : F
        hz : Not (Membership.mem E z)
        ⊢ Membership.mem (HasCompl.compl ↑E) (HSub.hSub y z)
      -/
      /-
        🎉 no goals
      -/
      simpa [sub_eq_add_neg, Submodule.add_mem_iff_right _ h]
      /-
        🎉 no goals
      -/
    /-
      case neg
      F : Type u_1
      inst✝⁴ : AddCommGroup F
      inst✝³ : Module Real F
      inst✝² : TopologicalSpace F
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousSMul Real F
      E : Submodule Real F
      hcodim : LT.lt 1 (Module.rank Real (HasQuotient.Quotient F E))
      x : F
      hx : Membership.mem (HasCompl.compl ↑E) x
      y : F
      h : Not (Membership.mem E y)
      ⊢ Membership.mem ((convexHull Real) (HasCompl.compl ↑E)) y
    -/
  · exact subset_convexHull ℝ (Eᶜ : Set F) h
    /-
      🎉 no goals
    -/


