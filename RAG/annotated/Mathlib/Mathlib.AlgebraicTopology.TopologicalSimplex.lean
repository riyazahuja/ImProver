instance (x : SimplexCategory) : Fintype (ConcreteCategory.forget.obj x) :=
  inferInstanceAs (Fintype (Fin _))


/-- The topological simplex associated to `x : SimplexCategory`.
  This is the object part of the functor `SimplexCategory.toTop`. -/
def toTopObj (x : SimplexCategory) := { f : x → ℝ≥0 | ∑ i, f i = 1 }


instance (x : SimplexCategory) : CoeFun x.toTopObj fun _ => x → ℝ≥0 :=
  ⟨fun f => (f : x → ℝ≥0)⟩


@[ext]
theorem toTopObj.ext {x : SimplexCategory} (f g : x.toTopObj) : (f : x → ℝ≥0) = g → f = g :=
  Subtype.ext


open Classical in
/-- A morphism in `SimplexCategory` induces a map on the associated topological spaces. -/
def toTopMap {x y : SimplexCategory} (f : x ⟶ y) (g : x.toTopObj) : y.toTopObj :=
  ⟨fun i => ∑ j ∈ Finset.univ.filter (f · = i), g j, by
    /-
      x y : SimplexCategory
      f : Quiver.Hom x y
      g : ↑x.toTopObj
      ⊢ Membership.mem y.toTopObj fun i => (Finset.filter (fun x_1 => Eq (f x_1) i)  …
    -/
    simp only [toTopObj, Set.mem_setOf]
    /-
      x y : SimplexCategory
      f : Quiver.Hom x y
      g : ↑x.toTopObj
      ⊢ Eq (Finset.univ.sum fun i => (Finset.filter (fun x_1 => Eq (f x_1) i) Finset …
    -/
    rw [← Finset.sum_biUnion]
      /-
        x y : SimplexCategory
        f : Quiver.Hom x y
        g : ↑x.toTopObj
        ⊢ Eq ((Finset.univ.biUnion fun i => Finset.filter (fun x_1 => Eq (f x_1) i) Fi …
      -/
    · have hg : ∑ i : (forget SimplexCategory).obj x, g i = 1 := g.2
      /-
        x y : SimplexCategory
        f : Quiver.Hom x y
        g : ↑x.toTopObj
        hg : Eq (Finset.univ.sum fun i => ↑g i) 1
        ⊢ Eq ((Finset.univ.biUnion fun i => Finset.filter (fun x_1 => Eq (f x_1) i) Fi …
      -/
      convert hg
      /-
        case h.e'_2.h
        x y : SimplexCategory
        f : Quiver.Hom x y
        g : ↑x.toTopObj
        hg : Eq (Finset.univ.sum fun i => ↑g i) 1
        ⊢ Eq (Finset.univ.biUnion fun i => Finset.filter (fun x_1 => Eq (f x_1) i) Fin …
      -/
      simp [Finset.eq_univ_iff_forall]
      /-
        🎉 no goals
      -/
      /-
        x y : SimplexCategory
        f : Quiver.Hom x y
        g : ↑x.toTopObj
        ⊢ (↑Finset.univ).PairwiseDisjoint fun i => Finset.filter (fun x_1 => Eq (f x_1 …
      -/
    · apply Set.pairwiseDisjoint_filter⟩
      /-
        🎉 no goals
      -/


open Classical in
@[simp]
theorem coe_toTopMap {x y : SimplexCategory} (f : x ⟶ y) (g : x.toTopObj) (i : y) :
    toTopMap f g i = ∑ j ∈ Finset.univ.filter (f · = i), g j :=
  rfl


@[continuity]
theorem continuous_toTopMap {x y : SimplexCategory} (f : x ⟶ y) : Continuous (toTopMap f) := by
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    ⊢ Continuous (SimplexCategory.toTopMap f)
  -/
  refine Continuous.subtype_mk (continuous_pi fun i => ?_) _
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    i : (CategoryTheory.forget SimplexCategory).obj y
    ⊢ Continuous fun a => (Finset.filter (fun x_1 => Eq (f x_1) i) Finset.univ).su …
  -/
  dsimp only [coe_toTopMap]
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    i : (CategoryTheory.forget SimplexCategory).obj y
    ⊢ Continuous fun a => (Finset.filter (fun x_1 => Eq (f x_1) i) Finset.univ).su …
  -/
  exact continuous_finset_sum _ (fun j _ => (continuous_apply _).comp continuous_subtype_val)
  /-
    🎉 no goals
  -/


/-- The functor associating the topological `n`-simplex to `[n] : SimplexCategory`. -/
@[simps obj map]
def toTop : SimplexCategory ⥤ TopCat where
  obj x := TopCat.of x.toTopObj
                           /-
                             X✝ Y✝ : SimplexCategory
                             f : Quiver.Hom X✝ Y✝
                             ⊢ Continuous (SimplexCategory.toTopMap f)
                           -/
  map f := ⟨toTopMap f, by continuity⟩
                           /-
                             🎉 no goals
                           -/
  map_id := by
    classical
    intro Δ
    ext f
    apply toTopObj.ext
    funext i
    change (Finset.univ.filter (· = i)).sum _ = _
    simp [Finset.sum_filter, CategoryTheory.id_apply]
  map_comp := fun f g => by
    classical
    ext h
    apply toTopObj.ext
    funext i
    dsimp
    simp only [comp_apply, TopCat.coe_of_of, ContinuousMap.coe_mk, coe_toTopMap]
    rw [← Finset.sum_biUnion]
    · apply Finset.sum_congr
      · exact Finset.ext (fun j => ⟨fun hj => by simpa using hj, fun hj => by simpa using hj⟩)
      · tauto
    · apply Set.pairwiseDisjoint_filter


