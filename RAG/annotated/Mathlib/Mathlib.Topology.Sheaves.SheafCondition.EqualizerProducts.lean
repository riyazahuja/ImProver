/-- The product of the sections of a presheaf over a family of open sets. -/
def piOpens : C :=
  ∏ᶜ fun i : ι => F.obj (op (U i))


/-- The product of the sections of a presheaf over the pairwise intersections of
a family of open sets.
-/
def piInters : C :=
  ∏ᶜ fun p : ι × ι => F.obj (op (U p.1 ⊓ U p.2))


/-- The morphism `Π F.obj (U i) ⟶ Π F.obj (U i) ⊓ (U j)` whose components
are given by the restriction maps from `U i` to `U i ⊓ U j`.
-/
def leftRes : piOpens F U ⟶ piInters.{v'} F U :=
  Pi.lift fun p : ι × ι => Pi.π _ p.1 ≫ F.map (infLELeft (U p.1) (U p.2)).op


/-- The morphism `Π F.obj (U i) ⟶ Π F.obj (U i) ⊓ (U j)` whose components
are given by the restriction maps from `U j` to `U i ⊓ U j`.
-/
def rightRes : piOpens F U ⟶ piInters.{v'} F U :=
  Pi.lift fun p : ι × ι => Pi.π _ p.2 ≫ F.map (infLERight (U p.1) (U p.2)).op


/-- The morphism `F.obj U ⟶ Π F.obj (U i)` whose components
are given by the restriction maps from `U j` to `U i ⊓ U j`.
-/
def res : F.obj (op (iSup U)) ⟶ piOpens.{v'} F U :=
  Pi.lift fun i : ι => F.map (TopologicalSpace.Opens.leSupr U i).op


@[simp, elementwise]
theorem res_π (i : ι) : res F U ≫ limit.π _ ⟨i⟩ = F.map (Opens.leSupr U i).op := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.Presheaf.SheafConditionEquali …
  -/
  rw [res, limit.lift_π, Fan.mk_π_app]
  /-
    🎉 no goals
  -/


@[elementwise]
theorem w : res F U ≫ leftRes F U = res F U ≫ rightRes F U := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.Presheaf.SheafConditionEquali …
  -/
  dsimp [res, leftRes, rightRes]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.lift fun i  …
  -/
  refine limit.hom_ext (fun _ => ?_)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    x✝ : CategoryTheory.Discrete (Prod ι ι)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [limit.lift_π, limit.lift_π_assoc, Fan.mk_π_app, Category.assoc]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    x✝ : CategoryTheory.Discrete (Prod ι ι)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (TopologicalSpace.Opens.leSupr …
  -/
  rw [← F.map_comp]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    x✝ : CategoryTheory.Discrete (Prod ι ι)
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (TopologicalSpace.Opens.leSupr …
  -/
  rw [← F.map_comp]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    x✝ : CategoryTheory.Discrete (Prod ι ι)
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (TopologicalSpace.Opens.leSupr …
  -/
  congr 1
  /-
    🎉 no goals
  -/


/-- The equalizer diagram for the sheaf condition.
-/
abbrev diagram : WalkingParallelPair ⥤ C :=
  parallelPair (leftRes.{v'} F U) (rightRes F U)


/-- The restriction map `F.obj U ⟶ Π F.obj (U i)` gives a cone over the equalizer diagram
for the sheaf condition. The sheaf condition asserts this cone is a limit cone.
-/
def fork : Fork.{v} (leftRes F U) (rightRes F U) :=
  Fork.ofι _ (w F U)


@[simp]
theorem fork_pt : (fork F U).pt = F.obj (op (iSup U)) :=
  rfl


@[simp]
theorem fork_ι : (fork F U).ι = res F U :=
  rfl


@[simp]
theorem fork_π_app_walkingParallelPair_zero : (fork F U).π.app WalkingParallelPair.zero = res F U :=
  rfl

-- Porting note: Shortcut simplifier

@[simp (high)]
theorem fork_π_app_walkingParallelPair_one :
    (fork F U).π.app WalkingParallelPair.one = res F U ≫ leftRes F U :=
  rfl


/-- Isomorphic presheaves have isomorphic `piOpens` for any cover `U`. -/
@[simp]
def piOpens.isoOfIso (α : F ≅ G) : piOpens F U ≅ piOpens.{v'} G U :=
  Pi.mapIso fun _ => α.app _


/-- Isomorphic presheaves have isomorphic `piInters` for any cover `U`. -/
@[simp]
def piInters.isoOfIso (α : F ≅ G) : piInters F U ≅ piInters.{v'} G U :=
  Pi.mapIso fun _ => α.app _


/-- Isomorphic presheaves have isomorphic sheaf condition diagrams. -/
def diagram.isoOfIso (α : F ≅ G) : diagram F U ≅ diagram.{v'} G U :=
  NatIso.ofComponents (by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasProducts C
      X : TopCat
      F : TopCat.Presheaf C X
      ι : Type v'
      U : ι → TopologicalSpace.Opens ↑X
      G : TopCat.Presheaf C X
      α : CategoryTheory.Iso F G
      ⊢ (X_1 : CategoryTheory.Limits.WalkingParallelPair) → CategoryTheory.Iso ((Top …
    -/
    rintro ⟨⟩
      /-
        case zero
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasProducts C
        X : TopCat
        F : TopCat.Presheaf C X
        ι : Type v'
        U : ι → TopologicalSpace.Opens ↑X
        G : TopCat.Presheaf C X
        α : CategoryTheory.Iso F G
        ⊢ CategoryTheory.Iso ((TopCat.Presheaf.SheafConditionEqualizerProducts.diagram …
      -/
    · exact piOpens.isoOfIso U α
      /-
        🎉 no goals
      -/
      /-
        case one
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasProducts C
        X : TopCat
        F : TopCat.Presheaf C X
        ι : Type v'
        U : ι → TopologicalSpace.Opens ↑X
        G : TopCat.Presheaf C X
        α : CategoryTheory.Iso F G
        ⊢ CategoryTheory.Iso ((TopCat.Presheaf.SheafConditionEqualizerProducts.diagram …
      -/
    · exact piInters.isoOfIso U α)
      /-
        🎉 no goals
      -/
    (by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasProducts C
        X : TopCat
        F : TopCat.Presheaf C X
        ι : Type v'
        U : ι → TopologicalSpace.Opens ↑X
        G : TopCat.Presheaf C X
        α : CategoryTheory.Iso F G
        ⊢ ∀ {X_1 Y : CategoryTheory.Limits.WalkingParallelPair} (f : Quiver.Hom X_1 Y) …
      -/
      rintro ⟨⟩ ⟨⟩ ⟨⟩
        /-
          case zero.zero.id
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          G : TopCat.Presheaf C X
          α : CategoryTheory.Iso F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.Presheaf.SheafConditionEqual …
        -/
      · simp
        /-
          🎉 no goals
        -/
      · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
        /-
          case zero.one.left
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          G : TopCat.Presheaf C X
          α : CategoryTheory.Iso F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.Presheaf.SheafConditionEqual …
        -/
        refine limit.hom_ext (fun _ => ?_)
        simp only [leftRes, piOpens.isoOfIso, piInters.isoOfIso, parallelPair_map_left,
          Functor.mapIso_hom, lim_map, limit.lift_map, limit.lift_π, Cones.postcompose_obj_π,
          NatTrans.comp_app, Fan.mk_π_app, Discrete.natIso_hom_app, Iso.app_hom, Category.assoc,
          NatTrans.naturality, limMap_π_assoc]
      · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
        /-
          case zero.one.right
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          G : TopCat.Presheaf C X
          α : CategoryTheory.Iso F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.Presheaf.SheafConditionEqual …
        -/
        refine limit.hom_ext (fun _ => ?_)
        simp only [rightRes, piOpens.isoOfIso, piInters.isoOfIso, parallelPair_map_right,
          Functor.mapIso_hom, lim_map, limit.lift_map, limit.lift_π, Cones.postcompose_obj_π,
          NatTrans.comp_app, Fan.mk_π_app, Discrete.natIso_hom_app, Iso.app_hom, Category.assoc,
          NatTrans.naturality, limMap_π_assoc]
        /-
          case one.one.id
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          G : TopCat.Presheaf C X
          α : CategoryTheory.Iso F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.Presheaf.SheafConditionEqual …
        -/
      · simp)
        /-
          🎉 no goals
        -/


/-- If `F G : Presheaf C X` are isomorphic presheaves,
then the `fork F U`, the canonical cone of the sheaf condition diagram for `F`,
is isomorphic to `fork F G` postcomposed with the corresponding isomorphism between
sheaf condition diagrams.
-/
def fork.isoOfIso (α : F ≅ G) :
    fork F U ≅ (Cones.postcompose (diagram.isoOfIso U α).inv).obj (fork G U) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    G : TopCat.Presheaf C X
    α : CategoryTheory.Iso F G
    ⊢ CategoryTheory.Iso (TopCat.Presheaf.SheafConditionEqualizerProducts.fork F U …
  -/
  fapply Fork.ext
    /-
      case i
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasProducts C
      X : TopCat
      F : TopCat.Presheaf C X
      ι : Type v'
      U : ι → TopologicalSpace.Opens ↑X
      G : TopCat.Presheaf C X
      α : CategoryTheory.Iso F G
      ⊢ CategoryTheory.Iso (TopCat.Presheaf.SheafConditionEqualizerProducts.fork F U …
    -/
  · apply α.app
    /-
      🎉 no goals
    -/
  · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
    /-
      case w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasProducts C
      X : TopCat
      F : TopCat.Presheaf C X
      ι : Type v'
      U : ι → TopologicalSpace.Opens ↑X
      G : TopCat.Presheaf C X
      α : CategoryTheory.Iso F G
      ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp (α.app { unop := iSup U }) …
    -/
    refine limit.hom_ext (fun _ => ?_)
    /-
      case w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasProducts C
      X : TopCat
      F : TopCat.Presheaf C X
      ι : Type v'
      U : ι → TopologicalSpace.Opens ↑X
      G : TopCat.Presheaf C X
      α : CategoryTheory.Iso F G
      x✝ : CategoryTheory.Discrete ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp only [Fork.ι]
    -- Ugh, `simp` can't unfold abbreviations.
    simp only [res, diagram.isoOfIso, Iso.app_hom, piOpens.isoOfIso, Cones.postcompose_obj_π,
      NatTrans.comp_app, fork_π_app_walkingParallelPair_zero, NatIso.ofComponents_inv_app,
      Functor.mapIso_inv, lim_map, limit.lift_map, Category.assoc, limit.lift_π, Fan.mk_π_app,
      Discrete.natIso_inv_app, Iso.app_inv, NatTrans.naturality, Iso.hom_inv_id_app_assoc]


/-- The sheaf condition for a `F : Presheaf C X` requires that the morphism
`F.obj U ⟶ ∏ᶜ F.obj (U i)` (where `U` is some open set which is the union of the `U i`)
is the equalizer of the two morphisms
`∏ᶜ F.obj (U i) ⟶ ∏ᶜ F.obj (U i) ⊓ (U j)`.
-/
def IsSheafEqualizerProducts (F : Presheaf.{v', v, u} C X) : Prop :=
  ∀ ⦃ι : Type v'⦄ (U : ι → Opens X), Nonempty (IsLimit (SheafConditionEqualizerProducts.fork F U))


/-- Implementation of `SheafConditionPairwiseIntersections.coneEquiv`. -/
@[simps]
def coneEquivFunctorObj (c : Cone ((diagram U).op ⋙ F)) :
    Cone (SheafConditionEqualizerProducts.diagram F U) where
  pt := c.pt
  π :=
    { app := fun Z =>
        WalkingParallelPair.casesOn Z (Pi.lift fun i : ι => c.π.app (op (single i)))
          (Pi.lift fun b : ι × ι => c.π.app (op (pair b.1 b.2)))
      naturality := fun Y Z f => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
          Y Z : CategoryTheory.Limits.WalkingParallelPair
          f : Quiver.Hom Y Z
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Categ …
        -/
                                /-
                                  🎉 no goals
                                -/
        cases Y <;> cases Z <;> cases f
        · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
          /-
            case zero.zero.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Categ …
          -/
          refine limit.hom_ext fun i => ?_
          /-
            case zero.zero.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            i : CategoryTheory.Discrete ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          dsimp
          simp only [limit.lift_π, Category.id_comp, Fan.mk_π_app, CategoryTheory.Functor.map_id,
            Category.assoc]
          /-
            case zero.zero.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            i : CategoryTheory.Discrete ι
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.single i.as }) (CategoryTheory …
          -/
          dsimp
          /-
            case zero.zero.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            i : CategoryTheory.Discrete ι
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.single i.as }) (CategoryTheory …
          -/
          simp only [limit.lift_π, Category.id_comp, Fan.mk_π_app]
          /-
            🎉 no goals
          -/
        · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
          /-
            case zero.one.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Categ …
          -/
          refine limit.hom_ext fun ⟨i, j⟩ => ?_
          /-
            case zero.one.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            x✝ : CategoryTheory.Discrete (Prod ι ι)
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          dsimp [SheafConditionEqualizerProducts.leftRes]
          simp only [limit.lift_π, limit.lift_π_assoc, Category.id_comp, Fan.mk_π_app,
            Category.assoc]
          /-
            case zero.one.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            x✝ : CategoryTheory.Discrete (Prod ι ι)
            i j : ι
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.pair i j }) (CategoryTheory.Ca …
          -/
          have h := c.π.naturality (Quiver.Hom.op (Hom.left i j))
          /-
            case zero.one.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            x✝ : CategoryTheory.Discrete (Prod ι ι)
            i j : ι
            h : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Op …
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.pair i j }) (CategoryTheory.Ca …
          -/
          dsimp at h
          /-
            case zero.one.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            x✝ : CategoryTheory.Discrete (Prod ι ι)
            i j : ι
            h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c …
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.pair i j }) (CategoryTheory.Ca …
          -/
          simpa using h
          /-
            🎉 no goals
          -/
        · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
          /-
            case zero.one.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Categ …
          -/
          refine limit.hom_ext fun ⟨i, j⟩ => ?_
          /-
            case zero.one.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            x✝ : CategoryTheory.Discrete (Prod ι ι)
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          dsimp [SheafConditionEqualizerProducts.rightRes]
          simp only [limit.lift_π, limit.lift_π_assoc, Category.id_comp, Fan.mk_π_app,
            Category.assoc]
          /-
            case zero.one.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            x✝ : CategoryTheory.Discrete (Prod ι ι)
            i j : ι
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.pair i j }) (CategoryTheory.Ca …
          -/
          have h := c.π.naturality (Quiver.Hom.op (Hom.right i j))
          /-
            case zero.one.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            x✝ : CategoryTheory.Discrete (Prod ι ι)
            i j : ι
            h : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Op …
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.pair i j }) (CategoryTheory.Ca …
          -/
          dsimp at h
          /-
            case zero.one.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            x✝ : CategoryTheory.Discrete (Prod ι ι)
            i j : ι
            h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c …
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.pair i j }) (CategoryTheory.Ca …
          -/
          simpa using h
          /-
            🎉 no goals
          -/
        · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
          /-
            case one.one.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Categ …
          -/
          refine limit.hom_ext fun i => ?_
          /-
            case one.one.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            i : CategoryTheory.Discrete (Prod ι ι)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          dsimp
          simp only [limit.lift_π, Category.id_comp, Fan.mk_π_app, CategoryTheory.Functor.map_id,
            Category.assoc]
          /-
            case one.one.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            i : CategoryTheory.Discrete (Prod ι ι)
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.pair i.as.1 i.as.2 }) (Categor …
          -/
          dsimp
          /-
            case one.one.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            i : CategoryTheory.Discrete (Prod ι ι)
            ⊢ Eq (c.π.app { unop := CategoryTheory.Pairwise.pair i.as.1 i.as.2 }) (Categor …
          -/
          simp only [limit.lift_π, Category.id_comp, Fan.mk_π_app] }
          /-
            🎉 no goals
          -/


/-- Implementation of `SheafConditionPairwiseIntersections.coneEquiv`. -/
@[simps!]
def coneEquivFunctor :
    Limits.Cone ((diagram U).op ⋙ F) ⥤
      Limits.Cone (SheafConditionEqualizerProducts.diagram F U) where
  obj c := coneEquivFunctorObj F U c
  map {c c'} f :=
    { hom := f.hom
      w := fun j => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c c' : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp …
          f : Quiver.Hom c c'
          j : CategoryTheory.Limits.WalkingParallelPair
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (((fun c => TopCat.Presheaf.She …
        -/
        cases j <;>
          · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
            /-
              case zero
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              c c' : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp …
              f : Quiver.Hom c c'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (((fun c => TopCat.Presheaf.She …
            -/
            refine limit.hom_ext fun i => ?_
            simp only [Limits.Fan.mk_π_app, Limits.ConeMorphism.w, Limits.limit.lift_π,
              Category.assoc, coneEquivFunctorObj_π_app] }


/-- Implementation of `SheafConditionPairwiseIntersections.coneEquiv`. -/
@[simps]
def coneEquivInverseObj (c : Limits.Cone (SheafConditionEqualizerProducts.diagram F U)) :
    Limits.Cone ((diagram U).op ⋙ F) where
  pt := c.pt
  π :=
    { app := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          ⊢ (X_1 : Opposite (CategoryTheory.Pairwise ι)) → Quiver.Hom (((CategoryTheory. …
        -/
        intro x
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          x : Opposite (CategoryTheory.Pairwise ι)
          ⊢ Quiver.Hom (((CategoryTheory.Functor.const (Opposite (CategoryTheory.Pairwis …
        -/
        induction x using Opposite.rec' with | h x => ?_
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          x : CategoryTheory.Pairwise ι
          ⊢ Quiver.Hom (((CategoryTheory.Functor.const (Opposite (CategoryTheory.Pairwis …
        -/
        rcases x with (⟨i⟩ | ⟨i, j⟩)
          /-
            case h.single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i : ι
            ⊢ Quiver.Hom (((CategoryTheory.Functor.const (Opposite (CategoryTheory.Pairwis …
          -/
        · exact c.π.app WalkingParallelPair.zero ≫ Pi.π _ i
          /-
            🎉 no goals
          -/
          /-
            case h.pair
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            ⊢ Quiver.Hom (((CategoryTheory.Functor.const (Opposite (CategoryTheory.Pairwis …
          -/
        · exact c.π.app WalkingParallelPair.one ≫ Pi.π _ (i, j)
          /-
            🎉 no goals
          -/
      naturality := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          ⊢ ∀ ⦃X_1 Y : Opposite (CategoryTheory.Pairwise ι)⦄ (f : Quiver.Hom X_1 Y), Eq  …
        -/
        intro x y f
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          x y : Opposite (CategoryTheory.Pairwise ι)
          f : Quiver.Hom x y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
        -/
        induction x using Opposite.rec' with | h x => ?_
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          y : Opposite (CategoryTheory.Pairwise ι)
          x : CategoryTheory.Pairwise ι
          f : Quiver.Hom { unop := x } y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
        -/
        induction y using Opposite.rec' with | h y => ?_
        /-
          case h.h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          x y : CategoryTheory.Pairwise ι
          f : Quiver.Hom { unop := x } { unop := y }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
        -/
        have ef : f = f.unop.op := rfl
        /-
          case h.h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          x y : CategoryTheory.Pairwise ι
          f : Quiver.Hom { unop := x } { unop := y }
          ef : Eq f f.unop.op
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
        -/
        revert ef
        /-
          case h.h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          x y : CategoryTheory.Pairwise ι
          f : Quiver.Hom { unop := x } { unop := y }
          ⊢ Eq f f.unop.op → Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Fu …
        -/
        generalize f.unop = f'
        /-
          case h.h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          x y : CategoryTheory.Pairwise ι
          f : Quiver.Hom { unop := x } { unop := y }
          f' : Quiver.Hom (Opposite.unop { unop := y }) (Opposite.unop { unop := x })
          ⊢ Eq f f'.op → Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functo …
        -/
        rintro rfl
        /-
          case h.h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
          x y : CategoryTheory.Pairwise ι
          f' : Quiver.Hom (Opposite.unop { unop := y }) (Opposite.unop { unop := x })
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
        -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
        rcases x with (⟨i⟩ | ⟨⟩) <;> rcases y with (⟨⟩ | ⟨j, j⟩) <;> rcases f' with ⟨⟩
          /-
            case h.h.single.single.id_single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
          -/
        · dsimp
          /-
            case h.h.single.single.id_single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          rw [F.map_id]
          /-
            case h.h.single.single.id_single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          simp
          /-
            🎉 no goals
          -/
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
          -/
        · dsimp
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          simp only [Category.id_comp, Category.assoc]
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          have h := c.π.naturality WalkingParallelPairHom.left
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Cat …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          dsimp [SheafConditionEqualizerProducts.leftRes] at h
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          simp only [Category.id_comp] at h
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (c.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          have h' := h =≫ Pi.π _ (i, j)
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (c.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory …
            h' : Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Wal …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          rw [h']
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (c.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory …
            h' : Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Wal …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          simp only [Category.assoc, limit.lift_π, Fan.mk_π_app]
          /-
            case h.h.single.pair.left
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (c.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory …
            h' : Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Wal …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) (Cat …
          -/
          rfl
          /-
            🎉 no goals
          -/
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
          -/
        · dsimp
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          simp only [Category.id_comp, Category.assoc]
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          have h := c.π.naturality WalkingParallelPairHom.right
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Cat …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          dsimp [SheafConditionEqualizerProducts.rightRes] at h
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          simp only [Category.id_comp] at h
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (c.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          have h' := h =≫ Pi.π _ (j, i)
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (c.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory …
            h' : Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Wal …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          rw [h']
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (c.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory …
            h' : Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Wal …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          simp
          /-
            case h.h.single.pair.right
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            i j : ι
            h : Eq (c.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory …
            h' : Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Wal …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) (Cat …
          -/
          rfl
          /-
            🎉 no goals
          -/
          /-
            case h.h.pair.pair.id_pair
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            a✝¹ a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
          -/
        · dsimp
          /-
            case h.h.pair.pair.id_pair
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            a✝¹ a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          rw [F.map_id]
          /-
            case h.h.pair.pair.id_pair
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
            a✝¹ a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          simp }
          /-
            🎉 no goals
          -/


/-- Implementation of `SheafConditionPairwiseIntersections.coneEquiv`. -/
@[simps!]
def coneEquivInverse :
    Limits.Cone (SheafConditionEqualizerProducts.diagram F U) ⥤
      Limits.Cone ((diagram U).op ⋙ F) where
  obj c := coneEquivInverseObj F U c
  map {c c'} f :=
    { hom := f.hom
      w := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c c' : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProd …
          f : Quiver.Hom c c'
          ⊢ ∀ (j : Opposite (CategoryTheory.Pairwise ι)), Eq (CategoryTheory.CategoryStr …
        -/
        intro x
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c c' : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProd …
          f : Quiver.Hom c c'
          x : Opposite (CategoryTheory.Pairwise ι)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (((fun c => TopCat.Presheaf.She …
        -/
        induction x using Opposite.rec' with | h x => ?_
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c c' : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProd …
          f : Quiver.Hom c c'
          x : CategoryTheory.Pairwise ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (((fun c => TopCat.Presheaf.She …
        -/
        rcases x with (⟨i⟩ | ⟨i, j⟩)
          /-
            case h.single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c c' : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProd …
            f : Quiver.Hom c c'
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (((fun c => TopCat.Presheaf.She …
          -/
        · dsimp
          /-
            case h.single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c c' : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProd …
            f : Quiver.Hom c c'
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (CategoryTheory.CategoryStruct. …
          -/
          dsimp only [Fork.ι]
          /-
            case h.single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c c' : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProd …
            f : Quiver.Hom c c'
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (CategoryTheory.CategoryStruct. …
          -/
          rw [← f.w WalkingParallelPair.zero, Category.assoc]
          /-
            🎉 no goals
          -/
          /-
            case h.pair
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c c' : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProd …
            f : Quiver.Hom c c'
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (((fun c => TopCat.Presheaf.She …
          -/
        · dsimp
          /-
            case h.pair
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c c' : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProd …
            f : Quiver.Hom c c'
            i j : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (CategoryTheory.CategoryStruct. …
          -/
          rw [← f.w WalkingParallelPair.one, Category.assoc] }
          /-
            🎉 no goals
          -/


/-- Implementation of `SheafConditionPairwiseIntersections.coneEquiv`. -/
@[simps]
def coneEquivUnitIsoApp (c : Cone ((diagram U).op ⋙ F)) :
    (𝟭 (Cone ((diagram U).op ⋙ F))).obj c ≅
      (coneEquivFunctor F U ⋙ coneEquivInverse F U).obj c where
  hom :=
    { hom := 𝟙 _
      w := fun j => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
          j : Opposite (CategoryTheory.Pairwise ι)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
        -/
        induction j using Opposite.rec' with | h j => ?_
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
          j : CategoryTheory.Pairwise ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
        -/
        rcases j with ⟨⟩ <;>
          /-
            case h.single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
          -/
          /-
            case h.single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          /-
            🎉 no goals
          -/
          /-
            case h.pair
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            a✝¹ a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          simp only [Limits.Fan.mk_π_app, Category.id_comp, Limits.limit.lift_π] }
          /-
            🎉 no goals
          -/
  inv :=
    { hom := 𝟙 _
      w := fun j => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
          j : Opposite (CategoryTheory.Pairwise ι)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((( …
        -/
        induction j using Opposite.rec' with | h j => ?_
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasProducts C
          X : TopCat
          F : TopCat.Presheaf C X
          ι : Type v'
          U : ι → TopologicalSpace.Opens ↑X
          c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
          j : CategoryTheory.Pairwise ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((( …
        -/
        rcases j with ⟨⟩ <;>
          /-
            case h.single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((( …
          -/
          /-
            case h.single
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          /-
            🎉 no goals
          -/
          /-
            case h.pair
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasProducts C
            X : TopCat
            F : TopCat.Presheaf C X
            ι : Type v'
            U : ι → TopologicalSpace.Opens ↑X
            c : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).op.comp F)
            a✝¹ a✝ : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
          simp only [Limits.Fan.mk_π_app, Category.id_comp, Limits.limit.lift_π] }
          /-
            🎉 no goals
          -/


/-- Implementation of `SheafConditionPairwiseIntersections.coneEquiv`. -/
@[simps!]
def coneEquivUnitIso :
    𝟭 (Limits.Cone ((diagram U).op ⋙ F)) ≅ coneEquivFunctor F U ⋙ coneEquivInverse F U :=
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasProducts C
    X : TopCat
    F : TopCat.Presheaf C X
    ι : Type v'
    U : ι → TopologicalSpace.Opens ↑X
    ⊢ ∀ {X_1 Y : CategoryTheory.Limits.Cone ((CategoryTheory.Pairwise.diagram U).o …
  -/
  NatIso.ofComponents (coneEquivUnitIsoApp F U)
  /-
    🎉 no goals
  -/


/-- Implementation of `SheafConditionPairwiseIntersections.coneEquiv`. -/
@[simps!]
def coneEquivCounitIso :
    coneEquivInverse F U ⋙ coneEquivFunctor F U ≅
      𝟭 (Limits.Cone (SheafConditionEqualizerProducts.diagram F U)) :=
  NatIso.ofComponents
    (fun c =>
      { hom :=
          { hom := 𝟙 _
            w := by
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
              -/
              rintro ⟨_ | _⟩
              · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
                /-
                  case zero
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((( …
                -/
                refine limit.hom_ext fun ⟨j⟩ => ?_
                /-
                  case zero
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  x✝ : CategoryTheory.Discrete ι
                  j : ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                dsimp [coneEquivInverse]
                /-
                  case zero
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  x✝ : CategoryTheory.Discrete ι
                  j : ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                simp only [Limits.Fan.mk_π_app, Category.id_comp, Limits.limit.lift_π]
                /-
                  🎉 no goals
                -/
              · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
                /-
                  case one
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((( …
                -/
                refine limit.hom_ext fun ⟨i, j⟩ => ?_
                /-
                  case one
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  x✝ : CategoryTheory.Discrete (Prod ι ι)
                  i j : ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                dsimp [coneEquivInverse]
                /-
                  case one
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  x✝ : CategoryTheory.Discrete (Prod ι ι)
                  i j : ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                simp only [Limits.Fan.mk_π_app, Category.id_comp, Limits.limit.lift_π] }
                /-
                  🎉 no goals
                -/
        inv :=
          { hom := 𝟙 _
            w := by
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
              -/
              rintro ⟨_ | _⟩
              · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
                /-
                  case zero
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
                -/
                refine limit.hom_ext fun ⟨j⟩ => ?_
                /-
                  case zero
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  x✝ : CategoryTheory.Discrete ι
                  j : ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                dsimp [coneEquivInverse]
                /-
                  case zero
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  x✝ : CategoryTheory.Discrete ι
                  j : ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                simp only [Limits.Fan.mk_π_app, Category.id_comp, Limits.limit.lift_π]
                /-
                  🎉 no goals
                -/
              · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
                /-
                  case one
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
                -/
                refine limit.hom_ext fun ⟨i, j⟩ => ?_
                /-
                  case one
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  x✝ : CategoryTheory.Discrete (Prod ι ι)
                  i j : ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                dsimp [coneEquivInverse]
                /-
                  case one
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : CategoryTheory.Limits.HasProducts C
                  X : TopCat
                  F : TopCat.Presheaf C X
                  ι : Type v'
                  U : ι → TopologicalSpace.Opens ↑X
                  c : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProduct …
                  x✝ : CategoryTheory.Discrete (Prod ι ι)
                  i j : ι
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                simp only [Limits.Fan.mk_π_app, Category.id_comp, Limits.limit.lift_π] } })
                /-
                  🎉 no goals
                -/
    fun {c d} f => by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasProducts C
      X : TopCat
      F : TopCat.Presheaf C X
      ι : Type v'
      U : ι → TopologicalSpace.Opens ↑X
      c d : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProdu …
      f : Quiver.Hom c d
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.SheafConditionPair …
    -/
    ext
    /-
      case w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasProducts C
      X : TopCat
      F : TopCat.Presheaf C X
      ι : Type v'
      U : ι → TopologicalSpace.Opens ↑X
      c d : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProdu …
      f : Quiver.Hom c d
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopCat.Presheaf.SheafConditionPair …
    -/
    dsimp
    /-
      case w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasProducts C
      X : TopCat
      F : TopCat.Presheaf C X
      ι : Type v'
      U : ι → TopologicalSpace.Opens ↑X
      c d : CategoryTheory.Limits.Cone (TopCat.Presheaf.SheafConditionEqualizerProdu …
      f : Quiver.Hom c d
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (CategoryTheory.CategoryStruct. …
    -/
    simp only [Category.comp_id, Category.id_comp]
    /-
      🎉 no goals
    -/


/--
Cones over `diagram U ⋙ F` are the same as a cones over the usual sheaf condition equalizer diagram.
-/
@[simps]
def coneEquiv :
    Limits.Cone ((diagram U).op ⋙ F) ≌
      Limits.Cone (SheafConditionEqualizerProducts.diagram F U) where
  functor := coneEquivFunctor F U
  inverse := coneEquivInverse F U
  unitIso := coneEquivUnitIso F U
  counitIso := coneEquivCounitIso F U

-- Porting note: not supported in Lean 4
-- attribute [local reducible]
--   SheafConditionEqualizerProducts.res SheafConditionEqualizerProducts.leftRes


/-- If `SheafConditionEqualizerProducts.fork` is an equalizer,
then `F.mapCone (cone U)` is a limit cone.
-/
def isLimitMapConeOfIsLimitSheafConditionFork
    (P : IsLimit (SheafConditionEqualizerProducts.fork F U)) : IsLimit (F.mapCone (cocone U).op) :=
  IsLimit.ofIsoLimit ((IsLimit.ofConeEquiv (coneEquiv F U).symm).symm P)
    { hom :=
        { hom := 𝟙 _
          w := by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
              ⊢ ∀ (j : Opposite (CategoryTheory.Pairwise ι)), Eq (CategoryTheory.CategoryStr …
            -/
            intro x
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
              x : Opposite (CategoryTheory.Pairwise ι)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((T …
            -/
            induction x with | h x => ?_
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
              x : CategoryTheory.Pairwise ι
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((T …
            -/
            rcases x with ⟨⟩
              /-
                case h.single
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
                a✝ : ι
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((T …
              -/
            · simp
              /-
                case h.single
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
                a✝ : ι
                ⊢ Eq (F.map (CategoryTheory.Pairwise.coconeιApp U (CategoryTheory.Pairwise.sin …
              -/
              rfl
              /-
                🎉 no goals
              -/
            · dsimp [coneEquivInverse, SheafConditionEqualizerProducts.res,
                SheafConditionEqualizerProducts.leftRes]
              simp only [limit.lift_π, limit.lift_π_assoc, Category.id_comp, Fan.mk_π_app,
                Category.assoc]
              /-
                case h.pair
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
                a✝¹ a✝ : ι
                ⊢ Eq (F.map (CategoryTheory.Pairwise.coconeιApp U (CategoryTheory.Pairwise.pai …
              -/
              rw [← F.map_comp]
              /-
                case h.pair
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
                a✝¹ a✝ : ι
                ⊢ Eq (F.map (CategoryTheory.Pairwise.coconeιApp U (CategoryTheory.Pairwise.pai …
              -/
              rfl }
              /-
                🎉 no goals
              -/
      inv :=
        { hom := 𝟙 _
          w := by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
              ⊢ ∀ (j : Opposite (CategoryTheory.Pairwise ι)), Eq (CategoryTheory.CategoryStr …
            -/
            intro x
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
              x : Opposite (CategoryTheory.Pairwise ι)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
            -/
            induction x with | h x => ?_
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
              x : CategoryTheory.Pairwise ι
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
            -/
            rcases x with ⟨⟩
              /-
                case h.single
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
                a✝ : ι
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
              -/
            · simp
              /-
                case h.single
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
                a✝ : ι
                ⊢ Eq (F.map (TopologicalSpace.Opens.leSupr U a✝).op) (F.map (CategoryTheory.Pa …
              -/
              rfl
              /-
                🎉 no goals
              -/
            · dsimp [coneEquivInverse, SheafConditionEqualizerProducts.res,
                SheafConditionEqualizerProducts.leftRes]
              simp only [limit.lift_π, limit.lift_π_assoc, Category.id_comp, Fan.mk_π_app,
                Category.assoc]
              /-
                case h.pair
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
                a✝¹ a✝ : ι
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (TopologicalSpace.Opens.leSupr …
              -/
              rw [← F.map_comp]
              /-
                case h.pair
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                P : CategoryTheory.Limits.IsLimit (TopCat.Presheaf.SheafConditionEqualizerProd …
                a✝¹ a✝ : ι
                ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (TopologicalSpace.Opens.leSupr …
              -/
              rfl } }
              /-
                🎉 no goals
              -/


/-- If `F.mapCone (cone U)` is a limit cone,
then `SheafConditionEqualizerProducts.fork` is an equalizer.
-/
def isLimitSheafConditionForkOfIsLimitMapCone (Q : IsLimit (F.mapCone (cocone U).op)) :
    IsLimit (SheafConditionEqualizerProducts.fork F U) :=
  IsLimit.ofIsoLimit ((IsLimit.ofConeEquiv (coneEquiv F U)).symm Q)
    { hom :=
        { hom := 𝟙 _
          w := by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
              ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
            -/
            rintro ⟨⟩
              /-
                case zero
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((T …
              -/
            · simp
              /-
                case zero
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                ⊢ Eq (TopCat.Presheaf.SheafConditionEqualizerProducts.res F U) (CategoryTheory …
              -/
              rfl
              /-
                🎉 no goals
              -/
            · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
              /-
                case one
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((T …
              -/
              refine limit.hom_ext fun ⟨i, j⟩ => ?_
              dsimp [coneEquivInverse, SheafConditionEqualizerProducts.res,
                SheafConditionEqualizerProducts.leftRes]
              simp only [limit.lift_π, limit.lift_π_assoc, Category.id_comp, Fan.mk_π_app,
                Category.assoc]
              /-
                case one
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                x✝ : CategoryTheory.Discrete (Prod ι ι)
                i j : ι
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (TopologicalSpace.Opens.leSupr …
              -/
              rw [← F.map_comp]
              /-
                case one
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                x✝ : CategoryTheory.Discrete (Prod ι ι)
                i j : ι
                ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (TopologicalSpace.Opens.leSupr …
              -/
              rfl }
              /-
                🎉 no goals
              -/
      inv :=
        { hom := 𝟙 _
          w := by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasProducts C
              X : TopCat
              F : TopCat.Presheaf C X
              ι : Type v'
              U : ι → TopologicalSpace.Opens ↑X
              Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
              ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
            -/
            rintro ⟨⟩
              /-
                case zero
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (To …
              -/
            · simp
              /-
                case zero
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                ⊢ Eq (CategoryTheory.Limits.Pi.lift fun i => F.map (CategoryTheory.Pairwise.co …
              -/
              rfl
              /-
                🎉 no goals
              -/
            · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` can't see `limit.hom_ext` applies here:
              /-
                case one
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (To …
              -/
              refine limit.hom_ext fun ⟨i, j⟩ => ?_
              dsimp [coneEquivInverse, SheafConditionEqualizerProducts.res,
                SheafConditionEqualizerProducts.leftRes]
              simp only [limit.lift_π, limit.lift_π_assoc, Category.id_comp, Fan.mk_π_app,
                Category.assoc]
              /-
                case one
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                x✝ : CategoryTheory.Discrete (Prod ι ι)
                i j : ι
                ⊢ Eq (F.map (CategoryTheory.Pairwise.coconeιApp U (CategoryTheory.Pairwise.pai …
              -/
              rw [← F.map_comp]
              /-
                case one
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Limits.HasProducts C
                X : TopCat
                F : TopCat.Presheaf C X
                ι : Type v'
                U : ι → TopologicalSpace.Opens ↑X
                Q : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (CategoryT …
                x✝ : CategoryTheory.Discrete (Prod ι ι)
                i j : ι
                ⊢ Eq (F.map (CategoryTheory.Pairwise.coconeιApp U (CategoryTheory.Pairwise.pai …
              -/
              rfl } }
              /-
                🎉 no goals
              -/


/-- The sheaf condition in terms of an equalizer diagram is equivalent
to the default sheaf condition.
-/
theorem isSheaf_iff_isSheafEqualizerProducts (F : Presheaf C X) :
    F.IsSheaf ↔ F.IsSheafEqualizerProducts :=
  (isSheaf_iff_isSheafPairwiseIntersections F).trans <|
    Iff.intro (fun h _ U => ⟨isLimitSheafConditionForkOfIsLimitMapCone F U (h U).some⟩) fun h _ U =>
      ⟨isLimitMapConeOfIsLimitSheafConditionFork F U (h U).some⟩


