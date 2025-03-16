/-- An open immersion of PresheafedSpaces is an open embedding `f : X ⟶ U ⊆ Y` of the underlying
spaces, such that the sheaf map `Y(V) ⟶ f _* X(V)` is an iso for each `V ⊆ U`.
-/
class PresheafedSpace.IsOpenImmersion {X Y : PresheafedSpace C} (f : X ⟶ Y) : Prop where
  /-- the underlying continuous map of underlying spaces from the source to an open subset of the
    target. -/
  base_open : IsOpenEmbedding f.base
  /-- the underlying sheaf morphism is an isomorphism on each open subset -/
  c_iso : ∀ U : Opens X, IsIso (f.c.app (op (base_open.isOpenMap.functor.obj U)))


/-- A morphism of SheafedSpaces is an open immersion if it is an open immersion as a morphism
of PresheafedSpaces
-/
abbrev SheafedSpace.IsOpenImmersion {X Y : SheafedSpace C} (f : X ⟶ Y) : Prop :=
  PresheafedSpace.IsOpenImmersion f


/-- A morphism of LocallyRingedSpaces is an open immersion if it is an open immersion as a morphism
of SheafedSpaces
-/
abbrev LocallyRingedSpace.IsOpenImmersion {X Y : LocallyRingedSpace} (f : X ⟶ Y) : Prop :=
  SheafedSpace.IsOpenImmersion f.1


local notation "IsOpenImmersion" => PresheafedSpace.IsOpenImmersion


/-- The functor `Opens X ⥤ Opens Y` associated with an open immersion `f : X ⟶ Y`. -/
abbrev opensFunctor :=
  H.base_open.isOpenMap.functor


/-- An open immersion `f : X ⟶ Y` induces an isomorphism `X ≅ Y|_{f(X)}`. -/
@[simps! hom_c_app]
noncomputable def isoRestrict : X ≅ Y.restrict H.base_open :=
  PresheafedSpace.isoOfComponents (Iso.refl _) <| by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Iso ((TopCat.Presheaf.pushforward C (CategoryTheory.Iso.refl  …
    -/
    symm
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Iso (Y.restrict ⋯).presheaf ((TopCat.Presheaf.pushforward C ( …
    -/
    fapply NatIso.ofComponents
      /-
        case app
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        ⊢ (X_1 : Opposite (TopologicalSpace.Opens ↑↑(Y.restrict ⋯))) → CategoryTheory. …
      -/
    · intro U
      /-
        case app
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        U : Opposite (TopologicalSpace.Opens ↑↑(Y.restrict ⋯))
        ⊢ CategoryTheory.Iso ((Y.restrict ⋯).presheaf.obj U) (((TopCat.Presheaf.pushfo …
      -/
      refine asIso (f.c.app (op (opensFunctor f |>.obj (unop U)))) ≪≫ X.presheaf.mapIso (eqToIso ?_)
      /-
        case app
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        U : Opposite (TopologicalSpace.Opens ↑↑(Y.restrict ⋯))
        ⊢ Eq ((TopologicalSpace.Opens.map f.base).op.obj { unop := (AlgebraicGeometry. …
      -/
      induction U using Opposite.rec' with | h U => ?_
      /-
        case app.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        U : TopologicalSpace.Opens ↑↑(Y.restrict ⋯)
        ⊢ Eq ((TopologicalSpace.Opens.map f.base).op.obj { unop := (AlgebraicGeometry. …
      -/
      cases U
      /-
        case app.h.mk
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        carrier✝ : Set ↑↑(Y.restrict ⋯)
        is_open'✝ : IsOpen carrier✝
        ⊢ Eq ((TopologicalSpace.Opens.map f.base).op.obj { unop := (AlgebraicGeometry. …
      -/
      dsimp only [IsOpenMap.functor, Functor.op, Opens.map]
      /-
        case app.h.mk
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        carrier✝ : Set ↑↑(Y.restrict ⋯)
        is_open'✝ : IsOpen carrier✝
        ⊢ Eq { unop := { carrier := Set.preimage ⇑f.base ↑((AlgebraicGeometry.Presheaf …
      -/
      congr 2
      /-
        case app.h.mk.e_unop.e_carrier
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        carrier✝ : Set ↑↑(Y.restrict ⋯)
        is_open'✝ : IsOpen carrier✝
        ⊢ Eq (Set.preimage ⇑f.base ↑((AlgebraicGeometry.PresheafedSpace.IsOpenImmersio …
      -/
      erw [Set.preimage_image_eq _ H.base_open.injective]
      /-
        case app.h.mk.e_unop.e_carrier
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        carrier✝ : Set ↑↑(Y.restrict ⋯)
        is_open'✝ : IsOpen carrier✝
        ⊢ Eq (↑{ carrier := carrier✝, is_open' := is_open'✝ }) carrier✝
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case naturality
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        ⊢ autoParam (∀ {X_1 Y_1 : Opposite (TopologicalSpace.Opens ↑↑(Y.restrict ⋯))}  …
      -/
    · intro U V i
      /-
        case naturality
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        U V : Opposite (TopologicalSpace.Opens ↑↑(Y.restrict ⋯))
        i : Quiver.Hom U V
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Y.restrict ⋯).presheaf.map i) ((Cat …
      -/
      dsimp
      simp only [NatTrans.naturality_assoc, TopCat.Presheaf.pushforward_obj_obj,
        TopCat.Presheaf.pushforward_obj_map, Quiver.Hom.unop_op, Category.assoc]
      /-
        case naturality
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        U V : Opposite (TopologicalSpace.Opens ↑↑(Y.restrict ⋯))
        i : Quiver.Hom U V
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.c.app { unop := ⋯.functor.obj (Opp …
      -/
      rw [← X.presheaf.map_comp, ← X.presheaf.map_comp]
      /-
        case naturality
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        U V : Opposite (TopologicalSpace.Opens ↑↑(Y.restrict ⋯))
        i : Quiver.Hom U V
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.c.app { unop := ⋯.functor.obj (Opp …
      -/
      congr 1
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
theorem isoRestrict_hom_ofRestrict : (isoRestrict f).hom ≫ Y.ofRestrict _ = f := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `NatTrans.ext`
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  refine PresheafedSpace.Hom.ext _ _ rfl <| NatTrans.ext <| funext fun x => ?_
  simp only [isoRestrict_hom_c_app, NatTrans.comp_app, eqToHom_refl,
    ofRestrict_c_app, Category.assoc, whiskerRight_id']
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    x : Opposite (TopologicalSpace.Opens ↑↑Y)
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
  -/
  erw [Category.comp_id, comp_c_app, f.c.naturality_assoc, ← X.presheaf.map_comp]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    x : Opposite (TopologicalSpace.Opens ↑↑Y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.c.app x) (X.presheaf.map (Category …
  -/
  trans f.c.app x ≫ X.presheaf.map (𝟙 _)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      x : Opposite (TopologicalSpace.Opens ↑↑Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.c.app x) (X.presheaf.map (Category …
    -/
  · congr 1
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      x : Opposite (TopologicalSpace.Opens ↑↑Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.c.app x) (X.presheaf.map (Category …
    -/
  · erw [X.presheaf.map_id, Category.comp_id]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem isoRestrict_inv_ofRestrict : (isoRestrict f).inv ≫ f = Y.ofRestrict _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  rw [Iso.inv_comp_eq, isoRestrict_hom_ofRestrict]
  /-
    🎉 no goals
  -/


instance mono : Mono f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    ⊢ CategoryTheory.Mono f
  -/
  rw [← H.isoRestrict_hom_ofRestrict]; apply mono_comp
                                       /-
                                         🎉 no goals
                                       -/


lemma c_iso' {V : Opens Y} (U : Opens X) (h : V = (opensFunctor f).obj U) :
    IsIso (f.c.app (Opposite.op V)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    V : TopologicalSpace.Opens ↑↑Y
    U : TopologicalSpace.Opens ↑↑X
    h : Eq V ((AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.opensFunctor f).o …
    ⊢ CategoryTheory.IsIso (f.c.app { unop := V })
  -/
  subst h
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U : TopologicalSpace.Opens ↑↑X
    ⊢ CategoryTheory.IsIso (f.c.app { unop := (AlgebraicGeometry.PresheafedSpace.I …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The composition of two open immersions is an open immersion. -/
instance comp {Z : PresheafedSpace C} (g : Y ⟶ Z) [hg : IsOpenImmersion g] :
    IsOpenImmersion (f ≫ g) where
  base_open := hg.base_open.comp H.base_open
  c_iso U := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      Z : AlgebraicGeometry.PresheafedSpace C
      g : Quiver.Hom Y Z
      hg : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
      U : TopologicalSpace.Opens ↑↑X
      ⊢ CategoryTheory.IsIso ((CategoryTheory.CategoryStruct.comp f g).c.app { unop  …
    -/
    generalize_proofs h
    dsimp only [AlgebraicGeometry.PresheafedSpace.comp_c_app, unop_op, Functor.op, comp_base,
      Opens.map_comp_obj]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      Z : AlgebraicGeometry.PresheafedSpace C
      g : Quiver.Hom Y Z
      hg : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
      U : TopologicalSpace.Opens ↑↑X
      h : IsOpenMap ⇑(CategoryTheory.CategoryStruct.comp f g).base
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (g.c.app { unop :=  …
    -/
    apply IsIso.comp_isIso'
      /-
        case x
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        Z : AlgebraicGeometry.PresheafedSpace C
        g : Quiver.Hom Y Z
        hg : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
        U : TopologicalSpace.Opens ↑↑X
        h : IsOpenMap ⇑(CategoryTheory.CategoryStruct.comp f g).base
        ⊢ CategoryTheory.IsIso (g.c.app { unop := h.functor.obj U })
      -/
    · exact c_iso' g ((opensFunctor f).obj U) (by ext; simp)
      /-
        🎉 no goals
      -/
      /-
        case x
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        Z : AlgebraicGeometry.PresheafedSpace C
        g : Quiver.Hom Y Z
        hg : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
        U : TopologicalSpace.Opens ↑↑X
        h : IsOpenMap ⇑(CategoryTheory.CategoryStruct.comp f g).base
        ⊢ CategoryTheory.IsIso (f.c.app { unop := (TopologicalSpace.Opens.map g.base). …
      -/
    · apply c_iso' f U
      /-
        case x
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        Z : AlgebraicGeometry.PresheafedSpace C
        g : Quiver.Hom Y Z
        hg : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
        U : TopologicalSpace.Opens ↑↑X
        h : IsOpenMap ⇑(CategoryTheory.CategoryStruct.comp f g).base
        ⊢ Eq ((TopologicalSpace.Opens.map g.base).obj (h.functor.obj U)) ((AlgebraicGe …
      -/
      ext1
      /-
        case x.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        Z : AlgebraicGeometry.PresheafedSpace C
        g : Quiver.Hom Y Z
        hg : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
        U : TopologicalSpace.Opens ↑↑X
        h : IsOpenMap ⇑(CategoryTheory.CategoryStruct.comp f g).base
        ⊢ Eq ↑((TopologicalSpace.Opens.map g.base).obj (h.functor.obj U)) ↑((Algebraic …
      -/
      dsimp only [Opens.map_coe, IsOpenMap.coe_functor_obj, comp_base, TopCat.coe_comp]
      /-
        case x.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Y
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        Z : AlgebraicGeometry.PresheafedSpace C
        g : Quiver.Hom Y Z
        hg : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
        U : TopologicalSpace.Opens ↑↑X
        h : IsOpenMap ⇑(CategoryTheory.CategoryStruct.comp f g).base
        ⊢ Eq (Set.preimage (⇑g.base) (Set.image (Function.comp ⇑g.base ⇑f.base) ↑U)) ( …
      -/
      rw [Set.image_comp, Set.preimage_image_eq _ hg.base_open.injective]
      /-
        🎉 no goals
      -/


/-- For an open immersion `f : X ⟶ Y` and an open set `U ⊆ X`, we have the map `X(U) ⟶ Y(U)`. -/
noncomputable def invApp (U : Opens X) :
    X.presheaf.obj (op U) ⟶ Y.presheaf.obj (op (opensFunctor f |>.obj U)) :=
                              /-
                                C : Type u
                                inst✝ : CategoryTheory.Category.{v, u} C
                                X Y : AlgebraicGeometry.PresheafedSpace C
                                f : Quiver.Hom X Y
                                H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                                U : TopologicalSpace.Opens ↑↑X
                                ⊢ Eq { unop := U } ((TopologicalSpace.Opens.map f.base).op.obj { unop := (Alge …
                              -/
  X.presheaf.map (eqToHom (by simp [Opens.map, Set.preimage_image_eq _ H.base_open.injective])) ≫
                              /-
                                🎉 no goals
                              -/
    inv (f.c.app (op (opensFunctor f |>.obj U)))


@[simp, reassoc]
theorem inv_naturality {U V : (Opens X)ᵒᵖ} (i : U ⟶ V) :
    X.presheaf.map i ≫ H.invApp _ (unop V) =
      invApp f (unop U) ≫ Y.presheaf.map (opensFunctor f |>.op.map i) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U V : Opposite (TopologicalSpace.Opens ↑↑X)
    i : Quiver.Hom U V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map i) (AlgebraicGeometry …
  -/
  simp only [invApp, ← Category.assoc]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U V : Opposite (TopologicalSpace.Opens ↑↑X)
    i : Quiver.Hom U V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [IsIso.comp_inv_eq]
  simp only [Functor.op_obj, op_unop, ← X.presheaf.map_comp, Functor.op_map, Category.assoc,
    NatTrans.naturality, Quiver.Hom.unop_op, IsIso.inv_hom_id_assoc,
    TopCat.Presheaf.pushforward_obj_map]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U V : Opposite (TopologicalSpace.Opens ↑↑X)
    i : Quiver.Hom U V
    ⊢ Eq (X.presheaf.map (CategoryTheory.CategoryStruct.comp i (CategoryTheory.eqT …
  -/
  congr 1
  /-
    🎉 no goals
  -/


                                                  /-
                                                    C : Type u
                                                    inst✝ : CategoryTheory.Category.{v, u} C
                                                    X Y : AlgebraicGeometry.PresheafedSpace C
                                                    f : Quiver.Hom X Y
                                                    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                                                    U : TopologicalSpace.Opens ↑↑X
                                                    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.invA …
                                                  -/
instance (U : Opens X) : IsIso (invApp f U) := by delta invApp; infer_instance
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem inv_invApp (U : Opens X) :
    inv (H.invApp _ U) =
      f.c.app (op (opensFunctor f |>.obj U)) ≫
        X.presheaf.map
                       /-
                         C : Type u
                         inst✝ : CategoryTheory.Category.{v, u} C
                         X Y : AlgebraicGeometry.PresheafedSpace C
                         f : Quiver.Hom X Y
                         H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                         U : TopologicalSpace.Opens ↑↑X
                         ⊢ Eq ((TopologicalSpace.Opens.map f.base).op.obj { unop := (AlgebraicGeometry. …
                       -/
          (eqToHom (by simp [Opens.map, Set.preimage_image_eq _ H.base_open.injective])) := by
                       /-
                         🎉 no goals
                       -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U : TopologicalSpace.Opens ↑↑X
    ⊢ Eq (CategoryTheory.inv (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.in …
  -/
  rw [← cancel_epi (H.invApp _ U), IsIso.hom_inv_id]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U : TopologicalSpace.Opens ↑↑X
    ⊢ Eq (CategoryTheory.CategoryStruct.id (X.presheaf.obj { unop := U })) (Catego …
  -/
  delta invApp
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U : TopologicalSpace.Opens ↑↑X
    ⊢ Eq (CategoryTheory.CategoryStruct.id (X.presheaf.obj { unop := U })) (Catego …
  -/
  simp [← Functor.map_comp]
  /-
    🎉 no goals
  -/


@[simp, reassoc, elementwise]
theorem invApp_app (U : Opens X) :
    invApp f U ≫ f.c.app (op (opensFunctor f |>.obj U)) = X.presheaf.map
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     X Y : AlgebraicGeometry.PresheafedSpace C
                     f : Quiver.Hom X Y
                     H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                     U : TopologicalSpace.Opens ↑↑X
                     ⊢ Eq { unop := U } ((TopologicalSpace.Opens.map f.base).op.obj { unop := (Alge …
                   -/
      (eqToHom (by simp [Opens.map, Set.preimage_image_eq _ H.base_open.injective])) := by
                   /-
                     🎉 no goals
                   -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U : TopologicalSpace.Opens ↑↑X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  rw [invApp, Category.assoc, IsIso.inv_hom_id, Category.comp_id]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem app_invApp (U : Opens Y) :
    f.c.app (op U) ≫ H.invApp _ ((Opens.map f.base).obj U) =
      Y.presheaf.map
        ((homOfLE (Set.image_preimage_subset f.base U.1)).op :
          op U ⟶ op (opensFunctor f |>.obj ((Opens.map f.base).obj U))) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U : TopologicalSpace.Opens ↑↑Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.c.app { unop := U }) (AlgebraicGeo …
  -/
  erw [← Category.assoc]; rw [IsIso.comp_inv_eq, f.c.naturality]; congr
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A variant of `app_inv_app` that gives an `eqToHom` instead of `homOfLe`. -/
@[reassoc]
theorem app_inv_app' (U : Opens Y) (hU : (U : Set Y) ⊆ Set.range f.base) :
    f.c.app (op U) ≫ invApp f ((Opens.map f.base).obj U) =
      Y.presheaf.map
        (eqToHom
            (le_antisymm (Set.image_preimage_subset f.base U.1) <|
              (Set.image_preimage_eq_inter_range (f := f.base) (t := U.1)).symm ▸
                Set.subset_inter_iff.mpr ⟨fun _ h => h, hU⟩)).op := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U : TopologicalSpace.Opens ↑↑Y
    hU : HasSubset.Subset (↑U) (Set.range ⇑f.base)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.c.app { unop := U }) (AlgebraicGeo …
  -/
  erw [← Category.assoc]; rw [IsIso.comp_inv_eq, f.c.naturality]; congr
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- An isomorphism is an open immersion. -/
instance ofIso {X Y : PresheafedSpace C} (H : X ≅ Y) : IsOpenImmersion H.hom where
  base_open := (TopCat.homeoOfIso ((forget C).mapIso H)).isOpenEmbedding
  -- Porting note: `inferInstance` will fail if Lean is not told that `H.hom.c` is iso
  c_iso _ := letI : IsIso H.hom.c := c_isIso_of_iso H.hom; inferInstance


instance (priority := 100) ofIsIso {X Y : PresheafedSpace C} (f : X ⟶ Y) [IsIso f] :
    IsOpenImmersion f :=
  AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.ofIso (asIso f)


instance ofRestrict {X : TopCat} (Y : PresheafedSpace C) {f : X ⟶ Y.carrier}
    (hf : IsOpenEmbedding f) : IsOpenImmersion (Y.ofRestrict hf) where
  base_open := hf
  c_iso U := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
      f✝ : Quiver.Hom X✝ Y✝
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
      X : TopCat
      Y : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X ↑Y
      hf : Topology.IsOpenEmbedding ⇑f
      U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
      ⊢ CategoryTheory.IsIso ((Y.ofRestrict hf).c.app { unop := ⋯.functor.obj U })
    -/
    dsimp
    have : (Opens.map f).obj (hf.isOpenMap.functor.obj U) = U := by
      ext1
      exact Set.preimage_image_eq _ hf.injective
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
      f✝ : Quiver.Hom X✝ Y✝
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
      X : TopCat
      Y : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X ↑Y
      hf : Topology.IsOpenEmbedding ⇑f
      U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
      this : Eq ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U)) U
      ⊢ CategoryTheory.IsIso (Y.presheaf.map (⋯.adjunction.counit.app (⋯.functor.obj …
    -/
    convert_to IsIso (Y.presheaf.map (𝟙 _))
      /-
        case h.e'_4
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        f✝ : Quiver.Hom X✝ Y✝
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
        X : TopCat
        Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X ↑Y
        hf : Topology.IsOpenEmbedding ⇑f
        U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
        this : Eq ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U)) U
        ⊢ Eq (Y.presheaf.obj { unop := ⋯.functor.obj ((TopologicalSpace.Opens.map f).o …
      -/
    · congr
      /-
        🎉 no goals
      -/
    · -- Porting note: was `apply Subsingleton.helim; rw [this]`
      -- See https://github.com/leanprover/lean4/issues/2273
      /-
        case h.e'_5
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        f✝ : Quiver.Hom X✝ Y✝
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
        X : TopCat
        Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X ↑Y
        hf : Topology.IsOpenEmbedding ⇑f
        U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
        this : Eq ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U)) U
        e_4✝ : Eq (Y.presheaf.obj { unop := ⋯.functor.obj ((TopologicalSpace.Opens.map …
        ⊢ HEq (Y.presheaf.map (⋯.adjunction.counit.app (⋯.functor.obj U)).op) (Y.presh …
      -/
      congr
        /-
          case h.e'_5.e_8.e_3.h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
          f✝ : Quiver.Hom X✝ Y✝
          H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
          X : TopCat
          Y : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X ↑Y
          hf : Topology.IsOpenEmbedding ⇑f
          U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
          this : Eq ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U)) U
          e_4✝ : Eq (Y.presheaf.obj { unop := ⋯.functor.obj ((TopologicalSpace.Opens.map …
          ⊢ Eq (⋯.functor.obj ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U))) (O …
        -/
      · simp only [unop_op]
        /-
          case h.e'_5.e_8.e_3.h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
          f✝ : Quiver.Hom X✝ Y✝
          H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
          X : TopCat
          Y : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X ↑Y
          hf : Topology.IsOpenEmbedding ⇑f
          U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
          this : Eq ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U)) U
          e_4✝ : Eq (Y.presheaf.obj { unop := ⋯.functor.obj ((TopologicalSpace.Opens.map …
          ⊢ Eq (⋯.functor.obj ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U))) (⋯ …
        -/
        congr
        /-
          🎉 no goals
        -/
      /-
        case h.e'_5.e_8.e_5
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        f✝ : Quiver.Hom X✝ Y✝
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
        X : TopCat
        Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X ↑Y
        hf : Topology.IsOpenEmbedding ⇑f
        U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
        this : Eq ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U)) U
        e_4✝ : Eq (Y.presheaf.obj { unop := ⋯.functor.obj ((TopologicalSpace.Opens.map …
        ⊢ HEq (⋯.adjunction.counit.app (⋯.functor.obj U)) (CategoryTheory.CategoryStru …
      -/
      apply Subsingleton.helim
      /-
        case h.e'_5.e_8.e_5.h₂
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        f✝ : Quiver.Hom X✝ Y✝
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
        X : TopCat
        Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X ↑Y
        hf : Topology.IsOpenEmbedding ⇑f
        U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
        this : Eq ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U)) U
        e_4✝ : Eq (Y.presheaf.obj { unop := ⋯.functor.obj ((TopologicalSpace.Opens.map …
        ⊢ Eq (Quiver.Hom (⋯.functor.obj ((TopologicalSpace.Opens.map f).obj (⋯.functor …
      -/
      rw [this]
      /-
        🎉 no goals
      -/
      /-
        case convert_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        f✝ : Quiver.Hom X✝ Y✝
        H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f✝
        X : TopCat
        Y : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X ↑Y
        hf : Topology.IsOpenEmbedding ⇑f
        U : TopologicalSpace.Opens ↑↑(Y.restrict hf)
        this : Eq ((TopologicalSpace.Opens.map f).obj (⋯.functor.obj U)) U
        ⊢ CategoryTheory.IsIso (Y.presheaf.map (CategoryTheory.CategoryStruct.id { uno …
      -/
    · infer_instance
      /-
        🎉 no goals
      -/


@[elementwise, simp]
theorem ofRestrict_invApp {C : Type*} [Category C] (X : PresheafedSpace C) {Y : TopCat}
    {f : Y ⟶ TopCat.of X.carrier} (h : IsOpenEmbedding f) (U : Opens (X.restrict h).carrier) :
    (PresheafedSpace.IsOpenImmersion.ofRestrict X h).invApp _ U = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    Y : TopCat
    f : Quiver.Hom Y (TopCat.of ↑↑X)
    h : Topology.IsOpenEmbedding ⇑f
    U : TopologicalSpace.Opens ↑↑(X.restrict h)
    ⊢ Eq (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.invApp (X.ofRestrict h …
  -/
  delta invApp
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    Y : TopCat
    f : Quiver.Hom Y (TopCat.of ↑↑X)
    h : Topology.IsOpenEmbedding ⇑f
    U : TopologicalSpace.Opens ↑↑(X.restrict h)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((X.restrict h).presheaf.map (Categor …
  -/
  rw [IsIso.comp_inv_eq, Category.id_comp]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    Y : TopCat
    f : Quiver.Hom Y (TopCat.of ↑↑X)
    h : Topology.IsOpenEmbedding ⇑f
    U : TopologicalSpace.Opens ↑↑(X.restrict h)
    ⊢ Eq ((X.restrict h).presheaf.map (CategoryTheory.eqToHom ⋯)) ((X.ofRestrict h …
  -/
  change X.presheaf.map _ = X.presheaf.map _
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    Y : TopCat
    f : Quiver.Hom Y (TopCat.of ↑↑X)
    h : Topology.IsOpenEmbedding ⇑f
    U : TopologicalSpace.Opens ↑↑(X.restrict h)
    ⊢ Eq (X.presheaf.map (⋯.functor.op.map (CategoryTheory.eqToHom ⋯))) (X.preshea …
  -/
  congr 1
  /-
    🎉 no goals
  -/


/-- An open immersion is an iso if the underlying continuous map is epi. -/
theorem to_iso [h' : Epi f.base] : IsIso f := by
  have : ∀ (U : (Opens Y)ᵒᵖ), IsIso (f.c.app U) := by
    intro U
    have : U = op (opensFunctor f |>.obj ((Opens.map f.base).obj (unop U))) := by
      induction U using Opposite.rec' with | h U => ?_
      cases U
      dsimp only [Functor.op, Opens.map]
      congr
      exact (Set.image_preimage_eq _ ((TopCat.epi_iff_surjective _).mp h')).symm
    convert H.c_iso (Opens.map f.base |>.obj <| unop U)

  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    h' : CategoryTheory.Epi f.base
    this : ∀ (U : Opposite (TopologicalSpace.Opens ↑↑Y)), CategoryTheory.IsIso (f. …
    ⊢ CategoryTheory.IsIso f
  -/
  have : IsIso f.c := NatIso.isIso_of_isIso_app _

  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    h' : CategoryTheory.Epi f.base
    this✝ : ∀ (U : Opposite (TopologicalSpace.Opens ↑↑Y)), CategoryTheory.IsIso (f …
    this : CategoryTheory.IsIso f.c
    ⊢ CategoryTheory.IsIso f
  -/
  apply (config := { allowSynthFailures := true }) isIso_of_components
  let t : X ≃ₜ Y := (Homeomorph.ofIsEmbedding _ H.base_open.isEmbedding).trans
    { toFun := Subtype.val
      invFun := fun x =>
        ⟨x, by rw [Set.range_eq_univ.mpr ((TopCat.epi_iff_surjective _).mp h')]; trivial⟩
      left_inv := fun ⟨_, _⟩ => rfl
      right_inv := fun _ => rfl }
  /-
    case inst
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    h' : CategoryTheory.Epi f.base
    this✝ : ∀ (U : Opposite (TopologicalSpace.Opens ↑↑Y)), CategoryTheory.IsIso (f …
    this : CategoryTheory.IsIso f.c
    t : Homeomorph ↑↑X ↑↑Y := (Homeomorph.ofIsEmbedding ⇑f.base ⋯).trans { toFun : …
    ⊢ CategoryTheory.IsIso f.base
  -/
  exact (TopCat.isoOfHomeo t).isIso_hom
  /-
    🎉 no goals
  -/


instance stalk_iso [HasColimits C] (x : X) : IsIso (f.stalkMap x) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    inst✝ : CategoryTheory.Limits.HasColimits C
    x : ↑↑X
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap f x)
  -/
  rw [← H.isoRestrict_hom_ofRestrict, PresheafedSpace.stalkMap.comp]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    inst✝ : CategoryTheory.Limits.HasColimits C
    x : ↑↑X
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- (Implementation.) The projection map when constructing the pullback along an open immersion.
-/
def pullbackConeOfLeftFst :
    Y.restrict (TopCat.snd_isOpenEmbedding_of_left hf.base_open g.base) ⟶ X where
  base := pullback.fst _ _
  c :=
    { app := fun U =>
        hf.invApp _ (unop U) ≫
          g.c.app (op (hf.base_open.isOpenMap.functor.obj (unop U))) ≫
            Y.presheaf.map
              (eqToHom
                (by
                  simp only [IsOpenMap.functor, Subtype.mk_eq_mk, unop_op, op_inj_iff, Opens.map,
                    Subtype.coe_mk, Functor.op_obj]
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    X Y Z : AlgebraicGeometry.PresheafedSpace C
                    f : Quiver.Hom X Z
                    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                    g : Quiver.Hom Y Z
                    U : Opposite (TopologicalSpace.Opens ↑↑X)
                    ⊢ Eq { carrier := Set.preimage ⇑g.base ↑{ carrier := Set.image ⇑f.base ↑(Oppos …
                  -/
                  apply LE.le.antisymm
                    /-
                      case a
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y Z : AlgebraicGeometry.PresheafedSpace C
                      f : Quiver.Hom X Z
                      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                      g : Quiver.Hom Y Z
                      U : Opposite (TopologicalSpace.Opens ↑↑X)
                      ⊢ LE.le { carrier := Set.preimage ⇑g.base ↑{ carrier := Set.image ⇑f.base ↑(Op …
                    -/
                  · rintro _ ⟨_, h₁, h₂⟩
                    /-
                      case a.intro.intro
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y Z : AlgebraicGeometry.PresheafedSpace C
                      f : Quiver.Hom X Z
                      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                      g : Quiver.Hom Y Z
                      U : Opposite (TopologicalSpace.Opens ↑↑X)
                      a✝ : ↑↑Y
                      w✝ : ↑↑X
                      h₁ : Membership.mem (↑(Opposite.unop U)) w✝
                      h₂ : Eq (f.base w✝) (g.base a✝)
                      ⊢ Membership.mem (↑{ carrier := Set.image ⇑(CategoryTheory.Limits.pullback.snd …
                    -/
                    use (TopCat.pullbackIsoProdSubtype _ _).inv ⟨⟨_, _⟩, h₂⟩
                    -- Porting note: need a slight hand holding
                    -- used to be `simpa using h₁` before https://github.com/leanprover-community/mathlib4/pull/13170
                    /-
                      case h
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y Z : AlgebraicGeometry.PresheafedSpace C
                      f : Quiver.Hom X Z
                      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                      g : Quiver.Hom Y Z
                      U : Opposite (TopologicalSpace.Opens ↑↑X)
                      a✝ : ↑↑Y
                      w✝ : ↑↑X
                      h₁ : Membership.mem (↑(Opposite.unop U)) w✝
                      h₂ : Eq (f.base w✝) (g.base a✝)
                      ⊢ And (Membership.mem (↑{ carrier := Set.preimage ⇑(CategoryTheory.Limits.pull …
                    -/
                    change _ ∈ _ ⁻¹' _ ∧ _
                    simp only [TopCat.coe_of, restrict_carrier, Set.preimage_id', Set.mem_preimage,
                      SetLike.mem_coe]
                    /-
                      case h
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y Z : AlgebraicGeometry.PresheafedSpace C
                      f : Quiver.Hom X Z
                      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                      g : Quiver.Hom Y Z
                      U : Opposite (TopologicalSpace.Opens ↑↑X)
                      a✝ : ↑↑Y
                      w✝ : ↑↑X
                      h₁ : Membership.mem (↑(Opposite.unop U)) w✝
                      h₂ : Eq (f.base w✝) (g.base a✝)
                      ⊢ And (Membership.mem (Opposite.unop U) ((CategoryTheory.Limits.pullback.fst f …
                    -/
                    constructor
                      /-
                        case h.left
                        C : Type u
                        inst✝ : CategoryTheory.Category.{v, u} C
                        X Y Z : AlgebraicGeometry.PresheafedSpace C
                        f : Quiver.Hom X Z
                        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                        g : Quiver.Hom Y Z
                        U : Opposite (TopologicalSpace.Opens ↑↑X)
                        a✝ : ↑↑Y
                        w✝ : ↑↑X
                        h₁ : Membership.mem (↑(Opposite.unop U)) w✝
                        h₂ : Eq (f.base w✝) (g.base a✝)
                        ⊢ Membership.mem (Opposite.unop U) ((CategoryTheory.Limits.pullback.fst f.base …
                      -/
                    · change _ ∈ U.unop at h₁
                      /-
                        case h.left
                        C : Type u
                        inst✝ : CategoryTheory.Category.{v, u} C
                        X Y Z : AlgebraicGeometry.PresheafedSpace C
                        f : Quiver.Hom X Z
                        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                        g : Quiver.Hom Y Z
                        U : Opposite (TopologicalSpace.Opens ↑↑X)
                        a✝ : ↑↑Y
                        w✝ : ↑↑X
                        h₂ : Eq (f.base w✝) (g.base a✝)
                        h₁ : Membership.mem (Opposite.unop U) w✝
                        ⊢ Membership.mem (Opposite.unop U) ((CategoryTheory.Limits.pullback.fst f.base …
                      -/
                      convert h₁
                      /-
                        case h.e'_5
                        C : Type u
                        inst✝ : CategoryTheory.Category.{v, u} C
                        X Y Z : AlgebraicGeometry.PresheafedSpace C
                        f : Quiver.Hom X Z
                        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                        g : Quiver.Hom Y Z
                        U : Opposite (TopologicalSpace.Opens ↑↑X)
                        a✝ : ↑↑Y
                        w✝ : ↑↑X
                        h₂ : Eq (f.base w✝) (g.base a✝)
                        h₁ : Membership.mem (Opposite.unop U) w✝
                        ⊢ Eq ((CategoryTheory.Limits.pullback.fst f.base g.base) ((TopCat.pullbackIsoP …
                      -/
                      erw [TopCat.pullbackIsoProdSubtype_inv_fst_apply]
                      /-
                        🎉 no goals
                      -/
                      /-
                        case h.right
                        C : Type u
                        inst✝ : CategoryTheory.Category.{v, u} C
                        X Y Z : AlgebraicGeometry.PresheafedSpace C
                        f : Quiver.Hom X Z
                        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                        g : Quiver.Hom Y Z
                        U : Opposite (TopologicalSpace.Opens ↑↑X)
                        a✝ : ↑↑Y
                        w✝ : ↑↑X
                        h₁ : Membership.mem (↑(Opposite.unop U)) w✝
                        h₂ : Eq (f.base w✝) (g.base a✝)
                        ⊢ Eq ((CategoryTheory.Limits.pullback.snd f.base g.base) ((TopCat.pullbackIsoP …
                      -/
                    · erw [TopCat.pullbackIsoProdSubtype_inv_snd_apply]
                      /-
                        🎉 no goals
                      -/
                    /-
                      case a
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y Z : AlgebraicGeometry.PresheafedSpace C
                      f : Quiver.Hom X Z
                      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                      g : Quiver.Hom Y Z
                      U : Opposite (TopologicalSpace.Opens ↑↑X)
                      ⊢ LE.le { carrier := Set.image ⇑(CategoryTheory.Limits.pullback.snd f.base g.b …
                    -/
                  · rintro _ ⟨x, h₁, rfl⟩
                    -- next line used to be
                    --  `exact ⟨_, h₁, ConcreteCategory.congr_hom pullback.condition x⟩))`
                    -- before https://github.com/leanprover-community/mathlib4/pull/13170
                    /-
                      case a.intro.intro
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y Z : AlgebraicGeometry.PresheafedSpace C
                      f : Quiver.Hom X Z
                      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                      g : Quiver.Hom Y Z
                      U : Opposite (TopologicalSpace.Opens ↑↑X)
                      x : ↑(CategoryTheory.Limits.pullback f.base g.base)
                      h₁ : Membership.mem (↑{ carrier := Set.preimage ⇑(CategoryTheory.Limits.pullba …
                      ⊢ Membership.mem (↑{ carrier := Set.preimage ⇑g.base ↑{ carrier := Set.image ⇑ …
                    -/
                    refine ⟨_, h₁, ?_⟩
                    /-
                      case a.intro.intro
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y Z : AlgebraicGeometry.PresheafedSpace C
                      f : Quiver.Hom X Z
                      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                      g : Quiver.Hom Y Z
                      U : Opposite (TopologicalSpace.Opens ↑↑X)
                      x : ↑(CategoryTheory.Limits.pullback f.base g.base)
                      h₁ : Membership.mem (↑{ carrier := Set.preimage ⇑(CategoryTheory.Limits.pullba …
                      ⊢ Eq (f.base ((CategoryTheory.Limits.pullback.fst f.base g.base) x)) (g.base ( …
                    -/
                    change (_ ≫ f.base) _ = (_ ≫ g.base) _
                    /-
                      case a.intro.intro
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y Z : AlgebraicGeometry.PresheafedSpace C
                      f : Quiver.Hom X Z
                      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                      g : Quiver.Hom Y Z
                      U : Opposite (TopologicalSpace.Opens ↑↑X)
                      x : ↑(CategoryTheory.Limits.pullback f.base g.base)
                      h₁ : Membership.mem (↑{ carrier := Set.preimage ⇑(CategoryTheory.Limits.pullba …
                      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst  …
                    -/
                    rw [pullback.condition]))
                    /-
                      🎉 no goals
                    -/
      naturality := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          ⊢ ∀ ⦃X_1 Y_1 : Opposite (TopologicalSpace.Opens ↑↑X)⦄ (f_1 : Quiver.Hom X_1 Y_ …
        -/
        intro U V i
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          U V : Opposite (TopologicalSpace.Opens ↑↑X)
          i : Quiver.Hom U V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map i) ((fun U => Categor …
        -/
        induction U using Opposite.rec'
        /-
          case h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          V : Opposite (TopologicalSpace.Opens ↑↑X)
          X✝ : TopologicalSpace.Opens ↑↑X
          i : Quiver.Hom { unop := X✝ } V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map i) ((fun U => Categor …
        -/
        induction V using Opposite.rec'
        -- Note: this doesn't fire in `simp` because of reduction of the term via structure eta
        -- before discrimination tree key generation
        /-
          case h.h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          X✝¹ X✝ : TopologicalSpace.Opens ↑↑X
          i : Quiver.Hom { unop := X✝¹ } { unop := X✝ }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map i) ((fun U => Categor …
        -/
        rw [inv_naturality_assoc]
        /-
          case h.h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          X✝¹ X✝ : TopologicalSpace.Opens ↑↑X
          i : Quiver.Hom { unop := X✝¹ } { unop := X✝ }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
        -/
        dsimp
        simp only [NatTrans.naturality_assoc, TopCat.Presheaf.pushforward_obj_map,
          Quiver.Hom.unop_op, ← Functor.map_comp, Category.assoc]
        /-
          case h.h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          X✝¹ X✝ : TopologicalSpace.Opens ↑↑X
          i : Quiver.Hom { unop := X✝¹ } { unop := X✝ }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
        -/
        rfl }
        /-
          🎉 no goals
        -/


theorem pullback_cone_of_left_condition : pullbackConeOfLeftFst f g ≫ f = Y.ofRestrict _ ≫ g := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `NatTrans.ext`
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  refine PresheafedSpace.Hom.ext _ _ ?_ <| NatTrans.ext <| funext fun U => ?_
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
  · simpa using pullback.condition
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      U : Opposite (TopologicalSpace.Opens ↑↑Z)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
  · induction U using Opposite.rec'
    -- Porting note: `NatTrans.comp_app` is not picked up by `dsimp`
    -- Perhaps see : https://github.com/leanprover-community/mathlib4/issues/5026
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      X✝ : TopologicalSpace.Opens ↑↑Z
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    rw [NatTrans.comp_app]
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      X✝ : TopologicalSpace.Opens ↑↑Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
    -/
    dsimp only [comp_c_app, unop_op, whiskerRight_app, pullbackConeOfLeftFst]
    -- simp only [ofRestrict_c_app, NatTrans.comp_app]
    simp only [app_invApp_assoc,
      eqToHom_app, Category.assoc, NatTrans.naturality_assoc]
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      X✝ : TopologicalSpace.Opens ↑↑Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.c.app { unop := X✝ }) (CategoryThe …
    -/
    erw [← Y.presheaf.map_comp, ← Y.presheaf.map_comp]
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      X✝ : TopologicalSpace.Opens ↑↑Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.c.app { unop := X✝ }) (Y.presheaf. …
    -/
    congr 1
    /-
      🎉 no goals
    -/


/-- We construct the pullback along an open immersion via restricting along the pullback of the
maps of underlying spaces (which is also an open embedding).
-/
def pullbackConeOfLeft : PullbackCone f g :=
  PullbackCone.mk (pullbackConeOfLeftFst f g) (Y.ofRestrict _)
    (pullback_cone_of_left_condition f g)


/-- (Implementation.) Any cone over `cospan f g` indeed factors through the constructed cone.
-/
def pullbackConeOfLeftLift : s.pt ⟶ (pullbackConeOfLeft f g).pt where
  base :=
    pullback.lift s.fst.base s.snd.base
      (congr_arg (fun x => PresheafedSpace.Hom.base x) s.condition)
  c :=
    { app := fun U =>
        s.snd.c.app _ ≫
          s.pt.presheaf.map
            (eqToHom
              (by
                /-
                  C : Type u
                  inst✝ : CategoryTheory.Category.{v, u} C
                  X Y Z : AlgebraicGeometry.PresheafedSpace C
                  f : Quiver.Hom X Z
                  hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                  g : Quiver.Hom Y Z
                  s : CategoryTheory.Limits.PullbackCone f g
                  U : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.PresheafedSpace.IsOp …
                  ⊢ Eq ((TopologicalSpace.Opens.map s.snd.base).op.obj (⋯.functor.op.obj U)) ((T …
                -/
                dsimp only [Opens.map, IsOpenMap.functor, Functor.op]
                /-
                  C : Type u
                  inst✝ : CategoryTheory.Category.{v, u} C
                  X Y Z : AlgebraicGeometry.PresheafedSpace C
                  f : Quiver.Hom X Z
                  hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                  g : Quiver.Hom Y Z
                  s : CategoryTheory.Limits.PullbackCone f g
                  U : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.PresheafedSpace.IsOp …
                  ⊢ Eq { unop := { carrier := Set.preimage ⇑s.snd.base ↑{ carrier := Set.image ⇑ …
                -/
                congr 2
                let s' : PullbackCone f.base g.base := PullbackCone.mk s.fst.base s.snd.base
                  -- Porting note: in mathlib3, this is just an underscore
                  (congr_arg Hom.base s.condition)

                /-
                  case e_unop.e_carrier
                  C : Type u
                  inst✝ : CategoryTheory.Category.{v, u} C
                  X Y Z : AlgebraicGeometry.PresheafedSpace C
                  f : Quiver.Hom X Z
                  hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                  g : Quiver.Hom Y Z
                  s : CategoryTheory.Limits.PullbackCone f g
                  U : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.PresheafedSpace.IsOp …
                  s' : CategoryTheory.Limits.PullbackCone f.base g.base := CategoryTheory.Limits …
                  ⊢ Eq (Set.preimage ⇑s.snd.base ↑{ carrier := Set.image ⇑(CategoryTheory.Limits …
                -/
                have : _ = s.snd.base := limit.lift_π s' WalkingCospan.right
                conv_lhs =>
                  rw [← this]
                  dsimp [s']
                  rw [Function.comp_def, ← Set.preimage_preimage]
                rw [Set.preimage_image_eq _
                    (TopCat.snd_isOpenEmbedding_of_left hf.base_open g.base).injective]
                /-
                  case e_unop.e_carrier
                  C : Type u
                  inst✝ : CategoryTheory.Category.{v, u} C
                  X Y Z : AlgebraicGeometry.PresheafedSpace C
                  f : Quiver.Hom X Z
                  hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                  g : Quiver.Hom Y Z
                  s : CategoryTheory.Limits.PullbackCone f g
                  U : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.PresheafedSpace.IsOp …
                  s' : CategoryTheory.Limits.PullbackCone f.base g.base := CategoryTheory.Limits …
                  this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lif …
                  ⊢ Eq (Set.preimage ⇑(CategoryTheory.Limits.limit.lift (CategoryTheory.Limits.c …
                -/
                rfl))
                /-
                  🎉 no goals
                -/
      naturality := fun U V i => by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          U V : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.PresheafedSpace.Is …
          i : Quiver.Hom U V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.PresheafedSpace.I …
        -/
        erw [s.snd.c.naturality_assoc]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          U V : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.PresheafedSpace.Is …
          i : Quiver.Hom U V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.snd.c.app (⋯.functor.op.obj U)) (C …
        -/
        rw [Category.assoc]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          U V : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.PresheafedSpace.Is …
          i : Quiver.Hom U V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.snd.c.app (⋯.functor.op.obj U)) (C …
        -/
        erw [← s.pt.presheaf.map_comp, ← s.pt.presheaf.map_comp]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          U V : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.PresheafedSpace.Is …
          i : Quiver.Hom U V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.snd.c.app (⋯.functor.op.obj U)) (s …
        -/
        congr 1 }
        /-
          🎉 no goals
        -/

-- this lemma is not a `simp` lemma, because it is an implementation detail

theorem pullbackConeOfLeftLift_fst :
    pullbackConeOfLeftLift f g s ≫ (pullbackConeOfLeft f g).fst = s.fst := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `NatTrans.ext`
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  refine PresheafedSpace.Hom.ext _ _ ?_ <| NatTrans.ext <| funext fun x => ?_
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
  · change pullback.lift _ _ _ ≫ pullback.fst _ _ = _
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : Opposite (TopologicalSpace.Opens ↑↑X)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
  · induction x using Opposite.rec' with | h x => ?_
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    change ((_ ≫ _) ≫ _ ≫ _) ≫ _ = _
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp_rw [Category.assoc]
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
    erw [← s.pt.presheaf.map_comp]
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
    erw [s.snd.c.naturality_assoc]
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
    have := congr_app s.condition (op (opensFunctor f |>.obj x))
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      this : Eq ((CategoryTheory.CategoryStruct.comp s.fst f).c.app { unop := (Algeb …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
    dsimp only [comp_c_app, unop_op] at this
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      this : Eq (CategoryTheory.CategoryStruct.comp (f.c.app { unop := (AlgebraicGeo …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
    rw [← IsIso.comp_inv_eq] at this
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
    replace this := reassoc_of% this
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      this : ∀ {Z_1 : C} (h : Quiver.Hom (((TopCat.Presheaf.pushforward C (CategoryT …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
    erw [← this, hf.invApp_app_assoc, s.fst.c.naturality_assoc]
    /-
      case refine_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : TopologicalSpace.Opens ↑↑X
      this : ∀ {Z_1 : C} (h : Quiver.Hom (((TopCat.Presheaf.pushforward C (CategoryT …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.fst.c.app { unop := x }) (Category …
    -/
    simp [eqToHom_map]
    /-
      🎉 no goals
    -/

-- this lemma is not a `simp` lemma, because it is an implementation detail

theorem pullbackConeOfLeftLift_snd :
    pullbackConeOfLeftLift f g s ≫ (pullbackConeOfLeft f g).snd = s.snd := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` did not pick up `NatTrans.ext`
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  refine PresheafedSpace.Hom.ext _ _ ?_ <| NatTrans.ext <| funext fun x => ?_
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
    -/
  · change pullback.lift _ _ _ ≫ pullback.snd _ _ = _
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : Opposite (TopologicalSpace.Opens ↑↑Y)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
  · change (_ ≫ _ ≫ _) ≫ _ = _
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : Opposite (TopologicalSpace.Opens ↑↑Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp_rw [Category.assoc]
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : Opposite (TopologicalSpace.Opens ↑↑Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.PresheafedSpace.I …
    -/
    erw [s.snd.c.naturality_assoc]
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : Opposite (TopologicalSpace.Opens ↑↑Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.snd.c.app x) (CategoryTheory.Categ …
    -/
    erw [← s.pt.presheaf.map_comp, ← s.pt.presheaf.map_comp]
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X Z
      hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      g : Quiver.Hom Y Z
      s : CategoryTheory.Limits.PullbackCone f g
      x : Opposite (TopologicalSpace.Opens ↑↑Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.snd.c.app x) (s.pt.presheaf.map (C …
    -/
    trans s.snd.c.app x ≫ s.pt.presheaf.map (𝟙 _)
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Z
        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.PullbackCone f g
        x : Opposite (TopologicalSpace.Opens ↑↑Y)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.snd.c.app x) (s.pt.presheaf.map (C …
      -/
    · congr 1
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Z
        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.PullbackCone f g
        x : Opposite (TopologicalSpace.Opens ↑↑Y)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.snd.c.app x) (s.pt.presheaf.map (C …
      -/
    · rw [s.pt.presheaf.map_id]; erw [Category.comp_id]
                                 /-
                                   🎉 no goals
                                 -/


instance pullbackConeSndIsOpenImmersion : IsOpenImmersion (pullbackConeOfLeft f g).snd := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (AlgebraicGeometry.Preshea …
  -/
  erw [CategoryTheory.Limits.PullbackCone.mk_snd]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (Y.ofRestrict ⋯)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The constructed pullback cone is indeed the pullback. -/
def pullbackConeOfLeftIsLimit : IsLimit (pullbackConeOfLeft f g) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ CategoryTheory.Limits.IsLimit (AlgebraicGeometry.PresheafedSpace.IsOpenImmer …
  -/
  apply PullbackCone.isLimitAux'
  /-
    case create
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ (s : CategoryTheory.Limits.PullbackCone f g) → Subtype fun l => And (Eq (Cat …
  -/
  intro s
  /-
    case create
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (AlgebraicGeo …
  -/
  use pullbackConeOfLeftLift f g s
  /-
    case property
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpa …
  -/
  use pullbackConeOfLeftLift_fst f g s
  /-
    case right
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpa …
  -/
  use pullbackConeOfLeftLift_snd f g s
  /-
    case right
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    ⊢ ∀ {m : Quiver.Hom s.pt (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.pu …
  -/
  intro m _ h₂
  /-
    case right
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.pullbac …
    a✝ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.PresheafedSpa …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.PresheafedSpa …
    ⊢ Eq m (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftLi …
  -/
  rw [← cancel_mono (pullbackConeOfLeft f g).snd]
  /-
    case right
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.pullbac …
    a✝ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.PresheafedSpa …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.PresheafedSpa …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.PresheafedSpace. …
  -/
  exact h₂.trans (pullbackConeOfLeftLift_snd f g s).symm
  /-
    🎉 no goals
  -/


instance hasPullback_of_left : HasPullback f g :=
  ⟨⟨⟨_, pullbackConeOfLeftIsLimit f g⟩⟩⟩


instance hasPullback_of_right : HasPullback g f :=
  hasPullback_symmetry f g


/-- Open immersions are stable under base-change. -/
instance pullbackSndOfLeft : IsOpenImmersion (pullback.snd f g) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (CategoryTheory.Limits.pul …
  -/
  delta pullback.snd
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (CategoryTheory.Limits.lim …
  -/
  rw [← limit.isoLimitCone_hom_π ⟨_, pullbackConeOfLeftIsLimit f g⟩ WalkingCospan.right]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (CategoryTheory.CategorySt …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Open immersions are stable under base-change. -/
instance pullbackFstOfRight : IsOpenImmersion (pullback.fst g f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (CategoryTheory.Limits.pul …
  -/
  rw [← pullbackSymmetry_hom_comp_snd]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (CategoryTheory.CategorySt …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance pullbackToBaseIsOpenImmersion [IsOpenImmersion g] :
    IsOpenImmersion (limit.π (cospan f g) WalkingCospan.one) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    inst✝ : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (CategoryTheory.Limits.lim …
  -/
  rw [← limit.w (cospan f g) WalkingCospan.Hom.inl, cospan_map_inl]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    s : CategoryTheory.Limits.PullbackCone f g
    inst✝ : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (CategoryTheory.CategorySt …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance forget_preservesLimitsOfLeft : PreservesLimit (cospan f g) (forget C) :=
  preservesLimit_of_preserves_limit_cone (pullbackConeOfLeftIsLimit f g)
    (by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Z
        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ CategoryTheory.Limits.IsLimit ((AlgebraicGeometry.PresheafedSpace.forget C). …
      -/
      apply (IsLimit.postcomposeHomEquiv (diagramIsoCospan _) _).toFun
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Z
        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose (Cat …
      -/
      refine (IsLimit.equivIsoLimit ?_).toFun (limit.isLimit (cospan f.base g.base))
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Z
        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit.cone (CategoryTheory.Limits. …
      -/
      fapply Cones.ext
        /-
          case φ
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit.cone (CategoryTheory.Limits. …
        -/
      · exact Iso.refl _
        /-
          🎉 no goals
        -/
      /-
        case w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Z
        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ autoParam (∀ (j : CategoryTheory.Limits.WalkingCospan), Eq ((CategoryTheory. …
      -/
      change ∀ j, _ = 𝟙 _ ≫ _ ≫ _
      /-
        case w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Z
        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq ((CategoryTheory.Limits.limi …
      -/
      simp_rw [Category.id_comp]
      /-
        case w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X Z
        hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq ((CategoryTheory.Limits.limi …
      -/
      rintro (_ | _ | _) <;> symm
        /-
          case w.none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicGeometry.PresheafedSpace. …
        -/
      · erw [Category.comp_id]
        /-
          case w.none
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          ⊢ Eq (((AlgebraicGeometry.PresheafedSpace.forget C).mapCone (AlgebraicGeometry …
        -/
        exact limit.w (cospan f.base g.base) WalkingCospan.Hom.inl
        /-
          🎉 no goals
        -/
        /-
          case w.some.left
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicGeometry.PresheafedSpace. …
        -/
      · exact Category.comp_id _
        /-
          🎉 no goals
        -/
        /-
          case w.some.right
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.PresheafedSpace C
          f : Quiver.Hom X Z
          hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.PullbackCone f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicGeometry.PresheafedSpace. …
        -/
      · exact Category.comp_id _)
        /-
          🎉 no goals
        -/


instance forget_preservesLimitsOfRight : PreservesLimit (cospan g f) (forget C) :=
  preservesPullback_symmetry (forget C) f g


theorem pullback_snd_isIso_of_range_subset (H : Set.range g.base ⊆ Set.range f.base) :
    IsIso (pullback.snd f g) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    H : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
  -/
  haveI := TopCat.snd_iso_of_left_embedding_range_subset hf.base_open.isEmbedding g.base H
  have : IsIso (pullback.snd f g).base := by
    delta pullback.snd
    rw [← limit.isoLimitCone_hom_π ⟨_, pullbackConeOfLeftIsLimit f g⟩ WalkingCospan.right]
    change IsIso (_ ≫ pullback.snd _ _)
    infer_instance
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    H : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
    this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f.base g.base)
    this : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g).base
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
  -/
  apply to_iso
  /-
    🎉 no goals
  -/


/-- The universal property of open immersions:
For an open immersion `f : X ⟶ Z`, given any morphism of schemes `g : Y ⟶ Z` whose topological
image is contained in the image of `f`, we can lift this morphism to a unique `Y ⟶ X` that
commutes with these maps.
-/
def lift (H : Set.range g.base ⊆ Set.range f.base) : Y ⟶ X :=
  haveI := pullback_snd_isIso_of_range_subset f g H
  inv (pullback.snd f g) ≫ pullback.fst _ _


@[simp, reassoc]
theorem lift_fac (H : Set.range g.base ⊆ Set.range f.base) : lift f g H ≫ f = g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Z
    hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    g : Quiver.Hom Y Z
    H : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  erw [Category.assoc]; rw [IsIso.inv_comp_eq]; exact pullback.condition
                                                /-
                                                  🎉 no goals
                                                -/


theorem lift_uniq (H : Set.range g.base ⊆ Set.range f.base) (l : Y ⟶ X) (hl : l ≫ f = g) :
                         /-
                           C : Type u
                           inst✝ : CategoryTheory.Category.{v, u} C
                           X Y Z : AlgebraicGeometry.PresheafedSpace C
                           f : Quiver.Hom X Z
                           hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                           g : Quiver.Hom Y Z
                           H : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
                           l : Quiver.Hom Y X
                           hl : Eq (CategoryTheory.CategoryStruct.comp l f) g
                           ⊢ Eq l (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.lift f g H)
                         -/
    l = lift f g H := by rw [← cancel_mono f, hl, lift_fac]
                         /-
                           🎉 no goals
                         -/


/-- Two open immersions with equal range is isomorphic. -/
@[simps]
def isoOfRangeEq [IsOpenImmersion g] (e : Set.range f.base = Set.range g.base) : X ≅ Y where
  hom := lift g f (le_of_eq e)
  inv := lift f g (le_of_eq e.symm)
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     X Y Z : AlgebraicGeometry.PresheafedSpace C
                     f : Quiver.Hom X Z
                     hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                     g : Quiver.Hom Y Z
                     s : CategoryTheory.Limits.PullbackCone f g
                     inst✝ : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
                     e : Eq (Set.range ⇑f.base) (Set.range ⇑g.base)
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
                   -/
  hom_inv_id := by rw [← cancel_mono f]; simp
                                         /-
                                           🎉 no goals
                                         -/
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     X Y Z : AlgebraicGeometry.PresheafedSpace C
                     f : Quiver.Hom X Z
                     hf : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                     g : Quiver.Hom Y Z
                     s : CategoryTheory.Limits.PullbackCone f g
                     inst✝ : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion g
                     e : Eq (Set.range ⇑f.base) (Set.range ⇑g.base)
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
                   -/
  inv_hom_id := by rw [← cancel_mono g]; simp
                                         /-
                                           🎉 no goals
                                         -/


/-- If `X ⟶ Y` is an open immersion, and `Y` is a SheafedSpace, then so is `X`. -/
def toSheafedSpace (f : X ⟶ Y.toPresheafedSpace) [H : IsOpenImmersion f] : SheafedSpace C where
  IsSheaf := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.PresheafedSpace C
      Y : AlgebraicGeometry.SheafedSpace C
      f : Quiver.Hom X Y.toPresheafedSpace
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      ⊢ X.presheaf.IsSheaf
    -/
    apply TopCat.Presheaf.isSheaf_of_iso (sheafIsoOfIso (isoRestrict f).symm).symm
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.PresheafedSpace C
      Y : AlgebraicGeometry.SheafedSpace C
      f : Quiver.Hom X Y.toPresheafedSpace
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      ⊢ ((TopCat.Presheaf.pushforward C (AlgebraicGeometry.PresheafedSpace.IsOpenImm …
    -/
    apply TopCat.Sheaf.pushforward_sheaf_of_sheaf
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.PresheafedSpace C
      Y : AlgebraicGeometry.SheafedSpace C
      f : Quiver.Hom X Y.toPresheafedSpace
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      ⊢ (Y.restrict ⋯).presheaf.IsSheaf
    -/
    exact (Y.restrict H.base_open).IsSheaf
    /-
      🎉 no goals
    -/
  toPresheafedSpace := X


@[simp]
theorem toSheafedSpace_toPresheafedSpace : (toSheafedSpace Y f).toPresheafedSpace = X :=
  rfl


/-- If `X ⟶ Y` is an open immersion of PresheafedSpaces, and `Y` is a SheafedSpace, we can
upgrade it into a morphism of SheafedSpaces.
-/
def toSheafedSpaceHom : toSheafedSpace Y f ⟶ Y :=
  f


@[simp]
theorem toSheafedSpaceHom_base : (toSheafedSpaceHom Y f).base = f.base :=
  rfl


@[simp]
theorem toSheafedSpaceHom_c : (toSheafedSpaceHom Y f).c = f.c :=
  rfl


instance toSheafedSpace_isOpenImmersion : SheafedSpace.IsOpenImmersion (toSheafedSpaceHom Y f) :=
  H


@[simp]
theorem sheafedSpace_toSheafedSpace {X Y : SheafedSpace C} (f : X ⟶ Y) [IsOpenImmersion f] :
                                 /-
                                   C : Type u
                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                   X Y : AlgebraicGeometry.SheafedSpace C
                                   f : Quiver.Hom X Y
                                   inst✝ : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
                                   ⊢ Eq (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toSheafedSpace Y f) X
                                 -/
    toSheafedSpace Y f = X := by cases X; rfl
                                          /-
                                            🎉 no goals
                                          -/


/-- If `X ⟶ Y` is an open immersion, and `Y` is a LocallyRingedSpace, then so is `X`. -/
def toLocallyRingedSpace : LocallyRingedSpace where
  toSheafedSpace := toSheafedSpace Y.toSheafedSpace f
  isLocalRing x :=
    haveI : IsLocalRing (Y.presheaf.stalk (f.base x)) := Y.isLocalRing _
    (asIso (f.stalkMap x)).commRingCatIsoToRingEquiv.isLocalRing


@[simp]
theorem toLocallyRingedSpace_toSheafedSpace :
    (toLocallyRingedSpace Y f).toSheafedSpace = toSheafedSpace Y.1 f :=
  rfl


/-- If `X ⟶ Y` is an open immersion of PresheafedSpaces, and `Y` is a LocallyRingedSpace, we can
upgrade it into a morphism of LocallyRingedSpace.
-/
def toLocallyRingedSpaceHom : toLocallyRingedSpace Y f ⟶ Y :=
  ⟨f, fun _ => inferInstance⟩


@[simp]
theorem toLocallyRingedSpaceHom_val : (toLocallyRingedSpaceHom Y f).toShHom = f :=
  rfl


instance toLocallyRingedSpace_isOpenImmersion :
    LocallyRingedSpace.IsOpenImmersion (toLocallyRingedSpaceHom Y f) :=
  H


@[simp]
theorem locallyRingedSpace_toLocallyRingedSpace {X Y : LocallyRingedSpace} (f : X ⟶ Y)
    [LocallyRingedSpace.IsOpenImmersion f] : toLocallyRingedSpace Y f.1 = X := by
    /-
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ Eq (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toLocallyRingedSpace Y …
    -/
    cases X; delta toLocallyRingedSpace; simp
                                         /-
                                           🎉 no goals
                                         -/


theorem isIso_of_subset {X Y : PresheafedSpace C} (f : X ⟶ Y)
    [H : PresheafedSpace.IsOpenImmersion f] (U : Opens Y.carrier)
    (hU : (U : Set Y.carrier) ⊆ Set.range f.base) : IsIso (f.c.app <| op U) := by
  have : U = H.base_open.isOpenMap.functor.obj ((Opens.map f.base).obj U) := by
    ext1
    exact (Set.inter_eq_left.mpr hU).symm.trans Set.image_preimage_eq_inter_range.symm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    U : TopologicalSpace.Opens ↑↑Y
    hU : HasSubset.Subset (↑U) (Set.range ⇑f.base)
    this : Eq U (⋯.functor.obj ((TopologicalSpace.Opens.map f.base).obj U))
    ⊢ CategoryTheory.IsIso (f.c.app { unop := U })
  -/
  convert H.c_iso ((Opens.map f.base).obj U)
  /-
    🎉 no goals
  -/


instance (priority := 100) of_isIso {X Y : SheafedSpace C} (f : X ⟶ Y) [IsIso f] :
    SheafedSpace.IsOpenImmersion f :=
  @PresheafedSpace.IsOpenImmersion.ofIsIso _ _ _ _ f
    (SheafedSpace.forgetToPresheafedSpace.map_isIso _)


instance comp {X Y Z : SheafedSpace C} (f : X ⟶ Y) (g : Y ⟶ Z) [SheafedSpace.IsOpenImmersion f]
    [SheafedSpace.IsOpenImmersion g] : SheafedSpace.IsOpenImmersion (f ≫ g) :=
  PresheafedSpace.IsOpenImmersion.comp f g


local notation "forget" => SheafedSpace.forgetToPresheafedSpace


instance : Mono f :=
                                                                       /-
                                                                         C : Type u
                                                                         inst✝ : CategoryTheory.Category.{v, u} C
                                                                         X Y Z : AlgebraicGeometry.SheafedSpace C
                                                                         f : Quiver.Hom X Z
                                                                         g : Quiver.Hom Y Z
                                                                         H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
                                                                         ⊢ CategoryTheory.Mono f
                                                                       -/
  (forget).mono_of_mono_map (show @Mono (PresheafedSpace C) _ _ _ f by infer_instance)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance forgetMapIsOpenImmersion : PresheafedSpace.IsOpenImmersion ((forget).map f) :=
  ⟨H.base_open, H.c_iso⟩


instance hasLimit_cospan_forget_of_left : HasLimit (cospan f g ⋙ forget) := by
  have : HasLimit (cospan ((cospan f g ⋙ forget).map Hom.inl)
      ((cospan f g ⋙ forget).map Hom.inr)) := by
    change HasLimit (cospan ((forget).map f) ((forget).map g))
    infer_instance
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.cospan (((Categor …
    ⊢ CategoryTheory.Limits.HasLimit ((CategoryTheory.Limits.cospan f g).comp Alge …
  -/
  apply hasLimitOfIso (diagramIsoCospan _).symm
  /-
    🎉 no goals
  -/


instance hasLimit_cospan_forget_of_left' :
    HasLimit (cospan ((cospan f g ⋙ forget).map Hom.inl) ((cospan f g ⋙ forget).map Hom.inr)) :=
  show HasLimit (cospan ((forget).map f) ((forget).map g)) from inferInstance


instance hasLimit_cospan_forget_of_right : HasLimit (cospan g f ⋙ forget) := by
  have : HasLimit (cospan ((cospan g f ⋙ forget).map Hom.inl)
      ((cospan g f ⋙ forget).map Hom.inr)) := by
    change HasLimit (cospan ((forget).map g) ((forget).map f))
    infer_instance
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.cospan (((Categor …
    ⊢ CategoryTheory.Limits.HasLimit ((CategoryTheory.Limits.cospan g f).comp Alge …
  -/
  apply hasLimitOfIso (diagramIsoCospan _).symm
  /-
    🎉 no goals
  -/


instance hasLimit_cospan_forget_of_right' :
    HasLimit (cospan ((cospan g f ⋙ forget).map Hom.inl) ((cospan g f ⋙ forget).map Hom.inr)) :=
  show HasLimit (cospan ((forget).map g) ((forget).map f)) from inferInstance


instance forgetCreatesPullbackOfLeft : CreatesLimit (cospan f g) forget :=
  createsLimitOfFullyFaithfulOfIso
    (PresheafedSpace.IsOpenImmersion.toSheafedSpace Y
      (@pullback.snd (PresheafedSpace C) _ _ _ _ f g _))
                                                  /-
                                                    C : Type u
                                                    inst✝ : CategoryTheory.Category.{v, u} C
                                                    X Y Z : AlgebraicGeometry.SheafedSpace C
                                                    f : Quiver.Hom X Z
                                                    g : Quiver.Hom Y Z
                                                    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
                                                    ⊢ Eq (CategoryTheory.Limits.pullback f g) (CategoryTheory.Limits.pullback (((C …
                                                  -/
    (eqToIso (show pullback _ _ = pullback _ _ by congr) ≪≫
                                                  /-
                                                    🎉 no goals
                                                  -/
      HasLimit.isoOfNatIso (diagramIsoCospan _).symm)


instance forgetCreatesPullbackOfRight : CreatesLimit (cospan g f) forget :=
  createsLimitOfFullyFaithfulOfIso
    (PresheafedSpace.IsOpenImmersion.toSheafedSpace Y
      (@pullback.fst (PresheafedSpace C) _ _ _ _ g f _))
                                                  /-
                                                    C : Type u
                                                    inst✝ : CategoryTheory.Category.{v, u} C
                                                    X Y Z : AlgebraicGeometry.SheafedSpace C
                                                    f : Quiver.Hom X Z
                                                    g : Quiver.Hom Y Z
                                                    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
                                                    ⊢ Eq (CategoryTheory.Limits.pullback g f) (CategoryTheory.Limits.pullback (((C …
                                                  -/
    (eqToIso (show pullback _ _ = pullback _ _ by congr) ≪≫
                                                  /-
                                                    🎉 no goals
                                                  -/
      HasLimit.isoOfNatIso (diagramIsoCospan _).symm)


instance sheafedSpace_forgetPreserves_of_left :
    PreservesLimit (cospan f g) (SheafedSpace.forget C) :=
  @Limits.comp_preservesLimit _ _ _ _ _ _ (cospan f g) _ _ forget (PresheafedSpace.forget C)
    inferInstance <| by
      have : PreservesLimit
        (cospan ((cospan f g ⋙ forget).map Hom.inl)
          ((cospan f g ⋙ forget).map Hom.inr)) (PresheafedSpace.forget C) := by
        dsimp
        infer_instance
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.SheafedSpace C
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
        this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan (((C …
        ⊢ CategoryTheory.Limits.PreservesLimit ((CategoryTheory.Limits.cospan f g).com …
      -/
      apply preservesLimit_of_iso_diagram _ (diagramIsoCospan _).symm
      /-
        🎉 no goals
      -/


instance sheafedSpace_forgetPreserves_of_right :
    PreservesLimit (cospan g f) (SheafedSpace.forget C) :=
  preservesPullback_symmetry _ _ _


instance sheafedSpace_hasPullback_of_left : HasPullback f g :=
  hasLimit_of_created (cospan f g) forget


instance sheafedSpace_hasPullback_of_right : HasPullback g f :=
  hasLimit_of_created (cospan g f) forget


/-- Open immersions are stable under base-change. -/
instance sheafedSpace_pullback_snd_of_left :
    SheafedSpace.IsOpenImmersion (pullback.snd f g) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.Limits.pullba …
  -/
  delta pullback.snd
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.Limits.limit. …
  -/
  have : _ = limit.π (cospan f g) right := preservesLimitIso_hom_π forget (cospan f g) right
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIs …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.Limits.limit. …
  -/
  rw [← this]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIs …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  have := HasLimit.isoOfNatIso_hom_π (diagramIsoCospan (cospan f g ⋙ forget)) right
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitI …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit. …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  erw [Category.comp_id] at this
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitI …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit. …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  rw [← this]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitI …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit. …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  dsimp
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitI …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit. …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance sheafedSpace_pullback_fst_of_right :
    SheafedSpace.IsOpenImmersion (pullback.fst g f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.Limits.pullba …
  -/
  delta pullback.fst
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.Limits.limit. …
  -/
  have : _ = limit.π (cospan g f) left := preservesLimitIso_hom_π forget (cospan g f) left
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIs …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.Limits.limit. …
  -/
  rw [← this]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIs …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  have := HasLimit.isoOfNatIso_hom_π (diagramIsoCospan (cospan g f ⋙ forget)) left
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitI …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit. …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  erw [Category.comp_id] at this
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitI …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit. …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  rw [← this]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitI …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit. …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  dsimp
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitI …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit. …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance sheafedSpace_pullback_to_base_isOpenImmersion [SheafedSpace.IsOpenImmersion g] :
    SheafedSpace.IsOpenImmersion (limit.π (cospan f g) one : pullback f g ⟶ Z) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.SheafedSpace.IsOpenImmersion g
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.Limits.limit. …
  -/
  rw [← limit.w (cospan f g) Hom.inl, cospan_map_inl]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.SheafedSpace.IsOpenImmersion g
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Suppose `X Y : SheafedSpace C`, where `C` is a concrete category,
whose forgetful functor reflects isomorphisms, preserves limits and filtered colimits.
Then a morphism `X ⟶ Y` that is a topological open embedding
is an open immersion iff every stalk map is an iso.
-/
theorem of_stalk_iso {X Y : SheafedSpace C} (f : X ⟶ Y) (hf : IsOpenEmbedding f.base)
    [H : ∀ x : X.1, IsIso (f.stalkMap x)] : SheafedSpace.IsOpenImmersion f :=
  { base_open := hf
    c_iso := fun U => by
      apply (config := {allowSynthFailures := true})
        TopCat.Presheaf.app_isIso_of_stalkFunctor_map_iso
          (show Y.sheaf ⟶ (TopCat.Sheaf.pushforward _ f.base).obj X.sheaf from ⟨f.c⟩)
      /-
        case inst
        C : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} C
        inst✝⁵ : CategoryTheory.Limits.HasLimits C
        inst✝⁴ : CategoryTheory.Limits.HasColimits C
        inst✝³ : CategoryTheory.ConcreteCategory C
        inst✝² : (CategoryTheory.forget C).ReflectsIsomorphisms
        inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
        inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
        X Y : AlgebraicGeometry.SheafedSpace C
        f : Quiver.Hom X Y
        hf : Topology.IsOpenEmbedding ⇑f.base
        H : ∀ (x : ↑↑X.toPresheafedSpace), CategoryTheory.IsIso (AlgebraicGeometry.Pre …
        U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
        ⊢ ∀ (x : Subtype fun x => Membership.mem (⋯.functor.obj U) x), CategoryTheory. …
      -/
      rintro ⟨_, y, hy, rfl⟩
      /-
        case inst.mk.intro.intro
        C : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} C
        inst✝⁵ : CategoryTheory.Limits.HasLimits C
        inst✝⁴ : CategoryTheory.Limits.HasColimits C
        inst✝³ : CategoryTheory.ConcreteCategory C
        inst✝² : (CategoryTheory.forget C).ReflectsIsomorphisms
        inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
        inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
        X Y : AlgebraicGeometry.SheafedSpace C
        f : Quiver.Hom X Y
        hf : Topology.IsOpenEmbedding ⇑f.base
        H : ∀ (x : ↑↑X.toPresheafedSpace), CategoryTheory.IsIso (AlgebraicGeometry.Pre …
        U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
        y : ↑↑X.toPresheafedSpace
        hy : Membership.mem (↑U) y
        ⊢ CategoryTheory.IsIso ((TopCat.Presheaf.stalkFunctor C ↑⟨f.base y, ⋯⟩).map (l …
      -/
      specialize H y
      /-
        case inst.mk.intro.intro
        C : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} C
        inst✝⁵ : CategoryTheory.Limits.HasLimits C
        inst✝⁴ : CategoryTheory.Limits.HasColimits C
        inst✝³ : CategoryTheory.ConcreteCategory C
        inst✝² : (CategoryTheory.forget C).ReflectsIsomorphisms
        inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
        inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
        X Y : AlgebraicGeometry.SheafedSpace C
        f : Quiver.Hom X Y
        hf : Topology.IsOpenEmbedding ⇑f.base
        U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
        y : ↑↑X.toPresheafedSpace
        hy : Membership.mem (↑U) y
        H : CategoryTheory.IsIso (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap f y)
        ⊢ CategoryTheory.IsIso ((TopCat.Presheaf.stalkFunctor C ↑⟨f.base y, ⋯⟩).map (l …
      -/
      delta PresheafedSpace.Hom.stalkMap at H
      haveI H' := TopCat.Presheaf.stalkPushforward.stalkPushforward_iso_of_isInducing C
        hf.toIsInducing X.presheaf y
      /-
        case inst.mk.intro.intro
        C : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} C
        inst✝⁵ : CategoryTheory.Limits.HasLimits C
        inst✝⁴ : CategoryTheory.Limits.HasColimits C
        inst✝³ : CategoryTheory.ConcreteCategory C
        inst✝² : (CategoryTheory.forget C).ReflectsIsomorphisms
        inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
        inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
        X Y : AlgebraicGeometry.SheafedSpace C
        f : Quiver.Hom X Y
        hf : Topology.IsOpenEmbedding ⇑f.base
        U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
        y : ↑↑X.toPresheafedSpace
        hy : Membership.mem (↑U) y
        H : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((TopCat.Presheaf …
        H' : CategoryTheory.IsIso (TopCat.Presheaf.stalkPushforward C f.base X.preshea …
        ⊢ CategoryTheory.IsIso ((TopCat.Presheaf.stalkFunctor C ↑⟨f.base y, ⋯⟩).map (l …
      -/
      have := IsIso.comp_isIso' H (@IsIso.inv_isIso _ _ _ _ _ H')
      /-
        case inst.mk.intro.intro
        C : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} C
        inst✝⁵ : CategoryTheory.Limits.HasLimits C
        inst✝⁴ : CategoryTheory.Limits.HasColimits C
        inst✝³ : CategoryTheory.ConcreteCategory C
        inst✝² : (CategoryTheory.forget C).ReflectsIsomorphisms
        inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget C)
        inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
        X Y : AlgebraicGeometry.SheafedSpace C
        f : Quiver.Hom X Y
        hf : Topology.IsOpenEmbedding ⇑f.base
        U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
        y : ↑↑X.toPresheafedSpace
        hy : Membership.mem (↑U) y
        H : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((TopCat.Presheaf …
        H' : CategoryTheory.IsIso (TopCat.Presheaf.stalkPushforward C f.base X.preshea …
        this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheor …
        ⊢ CategoryTheory.IsIso ((TopCat.Presheaf.stalkFunctor C ↑⟨f.base y, ⋯⟩).map (l …
      -/
      rwa [Category.assoc, IsIso.hom_inv_id, Category.comp_id] at this }
      /-
        🎉 no goals
      -/


/-- The functor `Opens X ⥤ Opens Y` associated with an open immersion `f : X ⟶ Y`. -/
abbrev opensFunctor : Opens X ⥤ Opens Y :=
  H.base_open.isOpenMap.functor


/-- An open immersion `f : X ⟶ Y` induces an isomorphism `X ≅ Y|_{f(X)}`. -/
@[simps! hom_c_app]
noncomputable def isoRestrict : X ≅ Y.restrict H.base_open :=
  SheafedSpace.isoMk <| PresheafedSpace.IsOpenImmersion.isoRestrict f


@[reassoc (attr := simp)]
theorem isoRestrict_hom_ofRestrict : (isoRestrict f).hom ≫ Y.ofRestrict _ = f :=
  PresheafedSpace.IsOpenImmersion.isoRestrict_hom_ofRestrict f


@[reassoc (attr := simp)]
theorem isoRestrict_inv_ofRestrict : (isoRestrict f).inv ≫ f = Y.ofRestrict _ :=
  PresheafedSpace.IsOpenImmersion.isoRestrict_inv_ofRestrict f


/-- For an open immersion `f : X ⟶ Y` and an open set `U ⊆ X`, we have the map `X(U) ⟶ Y(U)`. -/
noncomputable def invApp (U : Opens X) :
    X.presheaf.obj (op U) ⟶ Y.presheaf.obj (op (opensFunctor f |>.obj U)) :=
  PresheafedSpace.IsOpenImmersion.invApp f U


@[reassoc (attr := simp)]
theorem inv_naturality {U V : (Opens X)ᵒᵖ} (i : U ⟶ V) :
    X.presheaf.map i ≫ H.invApp _ (unop V) =
      H.invApp _ (unop U) ≫ Y.presheaf.map (opensFunctor f |>.op.map i) :=
  PresheafedSpace.IsOpenImmersion.inv_naturality f i


                                                    /-
                                                      C : Type u
                                                      inst✝ : CategoryTheory.Category.{v, u} C
                                                      X Y : AlgebraicGeometry.SheafedSpace C
                                                      f : Quiver.Hom X Y
                                                      H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
                                                      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
                                                      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.SheafedSpace.IsOpenImmersion.invApp  …
                                                    -/
instance (U : Opens X) : IsIso (H.invApp _ U) := by delta invApp; infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem inv_invApp (U : Opens X) :
    inv (H.invApp _ U) =
      f.c.app (op (opensFunctor f |>.obj U)) ≫ X.presheaf.map
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       X Y : AlgebraicGeometry.SheafedSpace C
                       f : Quiver.Hom X Y
                       H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
                       U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
                       ⊢ Eq ((TopologicalSpace.Opens.map f.base).op.obj { unop := (AlgebraicGeometry. …
                     -/
        (eqToHom (by simp [Opens.map, Set.preimage_image_eq _ H.base_open.injective])) :=
                     /-
                       🎉 no goals
                     -/
  PresheafedSpace.IsOpenImmersion.inv_invApp f U


@[reassoc (attr := simp)]
theorem invApp_app (U : Opens X) :
    H.invApp _ U ≫ f.c.app (op (opensFunctor f |>.obj U)) = X.presheaf.map
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     X Y : AlgebraicGeometry.SheafedSpace C
                     f : Quiver.Hom X Y
                     H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
                     U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
                     ⊢ Eq { unop := U } ((TopologicalSpace.Opens.map f.base).op.obj { unop := (Alge …
                   -/
      (eqToHom (by simp [Opens.map, Set.preimage_image_eq _ H.base_open.injective])) :=
                   /-
                     🎉 no goals
                   -/
  PresheafedSpace.IsOpenImmersion.invApp_app f U


attribute [elementwise] invApp_app


@[reassoc (attr := simp)]
theorem app_invApp (U : Opens Y) :
    f.c.app (op U) ≫ H.invApp _ ((Opens.map f.base).obj U) =
      Y.presheaf.map
        ((homOfLE (Set.image_preimage_subset f.base U.1)).op :
          op U ⟶ op (opensFunctor f |>.obj ((Opens.map f.base).obj U))) :=
  PresheafedSpace.IsOpenImmersion.app_invApp f U


/-- A variant of `app_inv_app` that gives an `eqToHom` instead of `homOfLe`. -/
@[reassoc]
theorem app_inv_app' (U : Opens Y) (hU : (U : Set Y) ⊆ Set.range f.base) :
    f.c.app (op U) ≫ invApp f ((Opens.map f.base).obj U) =
      Y.presheaf.map
        (eqToHom <|
            le_antisymm (Set.image_preimage_subset f.base U.1) <|
              (Set.image_preimage_eq_inter_range (f := f.base) (t := U.1)).symm ▸
                Set.subset_inter_iff.mpr ⟨fun _ h => h, hU⟩).op :=
  PresheafedSpace.IsOpenImmersion.app_invApp f U


instance ofRestrict {X : TopCat} (Y : SheafedSpace C) {f : X ⟶ Y.carrier}
    (hf : IsOpenEmbedding f) : IsOpenImmersion (Y.ofRestrict hf) :=
  PresheafedSpace.IsOpenImmersion.ofRestrict _ hf


@[elementwise, simp]
theorem ofRestrict_invApp {C : Type*} [Category C] (X : SheafedSpace C) {Y : TopCat}
    {f : Y ⟶ TopCat.of X.carrier} (h : IsOpenEmbedding f) (U : Opens (X.restrict h).carrier) :
    (SheafedSpace.IsOpenImmersion.ofRestrict X h).invApp _ U = 𝟙 _ :=
  PresheafedSpace.IsOpenImmersion.ofRestrict_invApp _ h U


/-- An open immersion is an iso if the underlying continuous map is epi. -/
theorem to_iso [h' : Epi f.base] : IsIso f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    h' : CategoryTheory.Epi f.base
    ⊢ CategoryTheory.IsIso f
  -/
  haveI : IsIso (forgetToPresheafedSpace.map f) := PresheafedSpace.IsOpenImmersion.to_iso f
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : AlgebraicGeometry.SheafedSpace C
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.SheafedSpace.IsOpenImmersion f
    h' : CategoryTheory.Epi f.base
    this : CategoryTheory.IsIso (AlgebraicGeometry.SheafedSpace.forgetToPresheafed …
    ⊢ CategoryTheory.IsIso f
  -/
  apply isIso_of_reflects_iso _ (SheafedSpace.forgetToPresheafedSpace)
  /-
    🎉 no goals
  -/


instance stalk_iso [HasColimits C] (x : X) :
    IsIso (f.stalkMap x) :=
  PresheafedSpace.IsOpenImmersion.stalk_iso f x


theorem sigma_ι_isOpenEmbedding : IsOpenEmbedding (colimit.ι F i).base := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i : CategoryTheory.Discrete ι
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.Limits.colimit.ι F i).base
  -/
  rw [← show _ = (colimit.ι F i).base from ι_preservesColimitIso_inv (SheafedSpace.forget C) F i]
  have : _ = _ ≫ colimit.ι (Discrete.functor ((F ⋙ SheafedSpace.forget C).obj ∘ Discrete.mk)) i :=
    HasColimit.isoOfNatIso_ι_hom Discrete.natIsoFunctor i
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i : CategoryTheory.Discrete ι
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι …
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  rw [← Iso.eq_comp_inv] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i : CategoryTheory.Discrete ι
    this : Eq (CategoryTheory.Limits.colimit.ι (F.comp (AlgebraicGeometry.SheafedS …
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  rw [this]
  have : colimit.ι _ _ ≫ _ = _ :=
    TopCat.sigmaIsoSigma_hom_ι.{v, v} ((F ⋙ SheafedSpace.forget C).obj ∘ Discrete.mk) i.as
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i : CategoryTheory.Discrete ι
    this✝ : Eq (CategoryTheory.Limits.colimit.ι (F.comp (AlgebraicGeometry.Sheafed …
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι …
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  rw [← Iso.eq_comp_inv] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i : CategoryTheory.Discrete ι
    this✝ : Eq (CategoryTheory.Limits.colimit.ι (F.comp (AlgebraicGeometry.Sheafed …
    this : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Discrete.functor (F …
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  cases i
  /-
    case mk
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    as✝ : ι
    this✝ : Eq (CategoryTheory.Limits.colimit.ι (F.comp (AlgebraicGeometry.Sheafed …
    this : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Discrete.functor (F …
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  rw [this, ← Category.assoc]
  -- Porting note: `simp_rw` can't use `TopCat.isOpenEmbedding_iff_comp_isIso` and
  -- `TopCat.isOpenEmbedding_iff_isIso_comp`.
  -- See https://github.com/leanprover-community/mathlib4/issues/5026
  erw [TopCat.isOpenEmbedding_iff_comp_isIso, TopCat.isOpenEmbedding_iff_comp_isIso,
    TopCat.isOpenEmbedding_iff_comp_isIso, TopCat.isOpenEmbedding_iff_isIso_comp]
  /-
    case mk
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    as✝ : ι
    this✝ : Eq (CategoryTheory.Limits.colimit.ι (F.comp (AlgebraicGeometry.Sheafed …
    this : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Discrete.functor (F …
    ⊢ Topology.IsOpenEmbedding ⇑(TopCat.sigmaι (Function.comp (F.comp (AlgebraicGe …
  -/
  exact .sigmaMk
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias sigma_ι_openEmbedding := sigma_ι_isOpenEmbedding


theorem image_preimage_is_empty (j : Discrete ι) (h : i ≠ j) (U : Opens (F.obj i)) :
    (Opens.map (colimit.ι (F ⋙ SheafedSpace.forgetToPresheafedSpace) j).base).obj
        ((Opens.map (preservesColimitIso SheafedSpace.forgetToPresheafedSpace F).inv.base).obj
          ((sigma_ι_isOpenEmbedding F i).isOpenMap.functor.obj U)) =
      ⊥ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i j : CategoryTheory.Discrete ι
    h : Ne i j
    U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
    ⊢ Eq ((TopologicalSpace.Opens.map (CategoryTheory.Limits.colimit.ι (F.comp Alg …
  -/
  ext x
  /-
    case h.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i j : CategoryTheory.Discrete ι
    h : Ne i j
    U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
    x : ↑↑((F.comp AlgebraicGeometry.SheafedSpace.forgetToPresheafedSpace).obj j)
    ⊢ Iff (Membership.mem (↑((TopologicalSpace.Opens.map (CategoryTheory.Limits.co …
  -/
  apply iff_false_intro
  /-
    case h.h.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i j : CategoryTheory.Discrete ι
    h : Ne i j
    U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
    x : ↑↑((F.comp AlgebraicGeometry.SheafedSpace.forgetToPresheafedSpace).obj j)
    ⊢ Not (Membership.mem (↑((TopologicalSpace.Opens.map (CategoryTheory.Limits.co …
  -/
  rintro ⟨y, hy, eq⟩
  replace eq := ConcreteCategory.congr_arg (preservesColimitIso (SheafedSpace.forget C) F ≪≫
    HasColimit.isoOfNatIso Discrete.natIsoFunctor ≪≫ TopCat.sigmaIsoSigma.{v, v} _).hom eq
  /-
    case h.h.h.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i j : CategoryTheory.Discrete ι
    h : Ne i j
    U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
    x : ↑↑((F.comp AlgebraicGeometry.SheafedSpace.forgetToPresheafedSpace).obj j)
    y : ↑↑(F.obj i).toPresheafedSpace
    hy : Membership.mem (↑U) y
    eq : Eq (((CategoryTheory.preservesColimitIso (AlgebraicGeometry.SheafedSpace. …
    ⊢ False
  -/
  simp_rw [CategoryTheory.Iso.trans_hom, ← TopCat.comp_app, ← PresheafedSpace.comp_base] at eq
  /-
    case h.h.h.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i j : CategoryTheory.Discrete ι
    h : Ne i j
    U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
    x : ↑↑((F.comp AlgebraicGeometry.SheafedSpace.forgetToPresheafedSpace).obj j)
    y : ↑↑(F.obj i).toPresheafedSpace
    hy : Membership.mem (↑U) y
    eq : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesColimitI …
    ⊢ False
  -/
  rw [ι_preservesColimitIso_inv] at eq
  change
    ((SheafedSpace.forget C).map (colimit.ι F i) ≫ _) y =
      ((SheafedSpace.forget C).map (colimit.ι F j) ≫ _) x at eq
  /-
    case h.h.h.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    i j : CategoryTheory.Discrete ι
    h : Ne i j
    U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
    x : ↑↑((F.comp AlgebraicGeometry.SheafedSpace.forgetToPresheafedSpace).obj j)
    y : ↑↑(F.obj i).toPresheafedSpace
    hy : Membership.mem (↑U) y
    eq : Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.SheafedSpace. …
    ⊢ False
  -/
  cases i; cases j
  rw [ι_preservesColimitIso_hom_assoc, ι_preservesColimitIso_hom_assoc,
    HasColimit.isoOfNatIso_ι_hom_assoc, HasColimit.isoOfNatIso_ι_hom_assoc,
    TopCat.sigmaIsoSigma_hom_ι, TopCat.sigmaIsoSigma_hom_ι] at eq
  /-
    case h.h.h.intro.intro.mk.mk
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    ι : Type v
    F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
    inst✝ : CategoryTheory.Limits.HasColimit F
    as✝¹ : ι
    U : TopologicalSpace.Opens ↑↑(F.obj { as := as✝¹ }).toPresheafedSpace
    y : ↑↑(F.obj { as := as✝¹ }).toPresheafedSpace
    hy : Membership.mem (↑U) y
    as✝ : ι
    x : ↑↑((F.comp AlgebraicGeometry.SheafedSpace.forgetToPresheafedSpace).obj { a …
    h : Ne { as := as✝¹ } { as := as✝ }
    eq : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Discrete.natIsoFu …
    ⊢ False
  -/
  exact h (congr_arg Discrete.mk (congr_arg Sigma.fst eq))
  /-
    🎉 no goals
  -/


instance sigma_ι_isOpenImmersion [HasStrictTerminalObjects C] :
    SheafedSpace.IsOpenImmersion (colimit.ι F i) where
  base_open := sigma_ι_isOpenEmbedding F i
  c_iso U := by
    have e : colimit.ι F i = _ :=
      (ι_preservesColimitIso_inv SheafedSpace.forgetToPresheafedSpace F i).symm
    have H :
      IsOpenEmbedding
        (colimit.ι (F ⋙ SheafedSpace.forgetToPresheafedSpace) i ≫
            (preservesColimitIso SheafedSpace.forgetToPresheafedSpace F).inv).base :=
      e ▸ sigma_ι_isOpenEmbedding F i
    suffices IsIso <| (colimit.ι (F ⋙ SheafedSpace.forgetToPresheafedSpace) i ≫
        (preservesColimitIso SheafedSpace.forgetToPresheafedSpace F).inv).c.app <|
      op (H.isOpenMap.functor.obj U) by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11083): just `convert` is very slow, so helps it a bit
      convert this using 2 <;> congr
    rw [PresheafedSpace.comp_c_app,
      ← PresheafedSpace.colimitPresheafObjIsoComponentwiseLimit_hom_π]
    -- Porting note: this instance created manually to make the `inferInstance` below work
    have inst1 : IsIso (preservesColimitIso forgetToPresheafedSpace F).inv.c :=
      PresheafedSpace.c_isIso_of_iso _
    rsuffices : IsIso
        (limit.π
          (PresheafedSpace.componentwiseDiagram (F ⋙ SheafedSpace.forgetToPresheafedSpace)
            ((Opens.map
                  (preservesColimitIso SheafedSpace.forgetToPresheafedSpace F).inv.base).obj
              (unop <| op <| H.isOpenMap.functor.obj U)))
          (op i))
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasLimits C
        ι : Type v
        F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
        inst✝¹ : CategoryTheory.Limits.HasColimit F
        i : CategoryTheory.Discrete ι
        inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
        U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
        e : Eq (CategoryTheory.Limits.colimit.ι F i) (CategoryTheory.CategoryStruct.co …
        H : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryThe …
        inst1 : CategoryTheory.IsIso (CategoryTheory.preservesColimitIso AlgebraicGeom …
        this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.limit.π (AlgebraicGeometry …
        ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pr …
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasLimits C
      ι : Type v
      F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      i : CategoryTheory.Discrete ι
      inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
      U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
      e : Eq (CategoryTheory.Limits.colimit.ι F i) (CategoryTheory.CategoryStruct.co …
      H : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryThe …
      inst1 : CategoryTheory.IsIso (CategoryTheory.preservesColimitIso AlgebraicGeom …
      ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.limit.π (AlgebraicGeometry.Presh …
    -/
    apply limit_π_isIso_of_is_strict_terminal
    /-
      case H
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasLimits C
      ι : Type v
      F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      i : CategoryTheory.Discrete ι
      inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
      U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
      e : Eq (CategoryTheory.Limits.colimit.ι F i) (CategoryTheory.CategoryStruct.co …
      H : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryThe …
      inst1 : CategoryTheory.IsIso (CategoryTheory.preservesColimitIso AlgebraicGeom …
      ⊢ (j : Opposite (CategoryTheory.Discrete ι)) → Ne j { unop := i } → CategoryTh …
    -/
    intro j hj
    /-
      case H
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasLimits C
      ι : Type v
      F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      i : CategoryTheory.Discrete ι
      inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
      U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
      e : Eq (CategoryTheory.Limits.colimit.ι F i) (CategoryTheory.CategoryStruct.co …
      H : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryThe …
      inst1 : CategoryTheory.IsIso (CategoryTheory.preservesColimitIso AlgebraicGeom …
      j : Opposite (CategoryTheory.Discrete ι)
      hj : Ne j { unop := i }
      ⊢ CategoryTheory.Limits.IsTerminal ((AlgebraicGeometry.PresheafedSpace.compone …
    -/
    induction j using Opposite.rec' with | h j => ?_
    /-
      case H.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasLimits C
      ι : Type v
      F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      i : CategoryTheory.Discrete ι
      inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
      U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
      e : Eq (CategoryTheory.Limits.colimit.ι F i) (CategoryTheory.CategoryStruct.co …
      H : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryThe …
      inst1 : CategoryTheory.IsIso (CategoryTheory.preservesColimitIso AlgebraicGeom …
      j : CategoryTheory.Discrete ι
      hj : Ne { unop := j } { unop := i }
      ⊢ CategoryTheory.Limits.IsTerminal ((AlgebraicGeometry.PresheafedSpace.compone …
    -/
    dsimp
    /-
      case H.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasLimits C
      ι : Type v
      F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      i : CategoryTheory.Discrete ι
      inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
      U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
      e : Eq (CategoryTheory.Limits.colimit.ι F i) (CategoryTheory.CategoryStruct.co …
      H : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryThe …
      inst1 : CategoryTheory.IsIso (CategoryTheory.preservesColimitIso AlgebraicGeom …
      j : CategoryTheory.Discrete ι
      hj : Ne { unop := j } { unop := i }
      ⊢ CategoryTheory.Limits.IsTerminal ((F.obj j).presheaf.obj { unop := (Topologi …
    -/
    convert (F.obj j).sheaf.isTerminalOfEmpty using 3
    /-
      case h.e'_3.h.e'_6.h.e'_2
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasLimits C
      ι : Type v
      F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      i : CategoryTheory.Discrete ι
      inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
      U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
      e : Eq (CategoryTheory.Limits.colimit.ι F i) (CategoryTheory.CategoryStruct.co …
      H : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryThe …
      inst1 : CategoryTheory.IsIso (CategoryTheory.preservesColimitIso AlgebraicGeom …
      j : CategoryTheory.Discrete ι
      hj : Ne { unop := j } { unop := i }
      ⊢ Eq ((TopologicalSpace.Opens.map (CategoryTheory.Limits.colimit.ι (F.comp Alg …
    -/
    convert image_preimage_is_empty F i j (fun h => hj (congr_arg op h.symm)) U using 6
    /-
      case h.e'_2.h.h.e'_6.h.e'_6.h.h.e'_5.h.e'_5.h.e'_3
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasLimits C
      ι : Type v
      F : CategoryTheory.Functor (CategoryTheory.Discrete ι) (AlgebraicGeometry.Shea …
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      i : CategoryTheory.Discrete ι
      inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
      U : TopologicalSpace.Opens ↑↑(F.obj i).toPresheafedSpace
      e : Eq (CategoryTheory.Limits.colimit.ι F i) (CategoryTheory.CategoryStruct.co …
      H : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp (CategoryThe …
      inst1 : CategoryTheory.IsIso (CategoryTheory.preservesColimitIso AlgebraicGeom …
      j : CategoryTheory.Discrete ι
      hj : Ne { unop := j } { unop := i }
      e_1✝¹ : Eq (TopologicalSpace.Opens ↑↑(F.obj j).toPresheafedSpace) (Topological …
      e_1✝ : Eq (TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F).toPreshe …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    exact (congr_arg PresheafedSpace.Hom.base e).symm
    /-
      🎉 no goals
    -/


instance (X : LocallyRingedSpace) {U : TopCat} (f : U ⟶ X.toTopCat) (hf : IsOpenEmbedding f) :
    LocallyRingedSpace.IsOpenImmersion (X.ofRestrict hf) :=
  PresheafedSpace.IsOpenImmersion.ofRestrict X.toPresheafedSpace hf


instance (priority := 100) of_isIso [IsIso g] : LocallyRingedSpace.IsOpenImmersion g :=
  @PresheafedSpace.IsOpenImmersion.ofIsIso _ _ _ _ g.1
    ⟨⟨(inv g).1, by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.LocallyRingedSpace
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
          inst✝ : CategoryTheory.IsIso g
          ⊢ And (Eq (CategoryTheory.CategoryStruct.comp g.toHom (CategoryTheory.inv g).t …
        -/
        erw [← LocallyRingedSpace.comp_toShHom]; rw [IsIso.hom_inv_id]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.LocallyRingedSpace
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
          inst✝ : CategoryTheory.IsIso g
          ⊢ And (Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (CategoryTheory.Ca …
        -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
        erw [← LocallyRingedSpace.comp_toShHom]; rw [IsIso.inv_hom_id]; constructor <;> rfl⟩⟩
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


instance comp (g : Z ⟶ Y) [LocallyRingedSpace.IsOpenImmersion g] :
    LocallyRingedSpace.IsOpenImmersion (f ≫ g) :=
  PresheafedSpace.IsOpenImmersion.comp f.1 g.1


instance mono : Mono f :=
                                                                                   /-
                                                                                     C : Type u
                                                                                     inst✝ : CategoryTheory.Category.{v, u} C
                                                                                     X Y Z : AlgebraicGeometry.LocallyRingedSpace
                                                                                     f : Quiver.Hom X Z
                                                                                     g : Quiver.Hom Y Z
                                                                                     H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
                                                                                     ⊢ CategoryTheory.Mono (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom f)
                                                                                   -/
  LocallyRingedSpace.forgetToSheafedSpace.mono_of_mono_map (show Mono f.toShHom by infer_instance)
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


instance : SheafedSpace.IsOpenImmersion (LocallyRingedSpace.forgetToSheafedSpace.map f) :=
  H

-- note to reviewers: is there a `count_heartbeats` for this?

set_option synthInstance.maxHeartbeats 40000 in
/-- An explicit pullback cone over `cospan f g` if `f` is an open immersion. -/
def pullbackConeOfLeft : PullbackCone f g := by
  refine PullbackCone.mk ?_
      (Y.ofRestrict (TopCat.snd_isOpenEmbedding_of_left H.base_open g.base)) ?_
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ Quiver.Hom (Y.restrict ⋯) X
    -/
  · use PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftFst f.1 g.1
    /-
      case prop
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ ∀ (x : ↑↑(Y.restrict ⋯).toPresheafedSpace), IsLocalHom (AlgebraicGeometry.Pr …
    -/
    intro x
    have := PresheafedSpace.stalkMap.congr_hom _ _
        (PresheafedSpace.IsOpenImmersion.pullback_cone_of_left_condition f.1 g.1) x
    /-
      case prop
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      x : ↑↑(Y.restrict ⋯).toPresheafedSpace
      this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.Cate …
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometr …
    -/
    rw [PresheafedSpace.stalkMap.comp, PresheafedSpace.stalkMap.comp] at this
    /-
      case prop
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      x : ↑↑(Y.restrict ⋯).toPresheafedSpace
      this : Eq (CategoryTheory.CategoryStruct.comp (f.stalkMap ((AlgebraicGeometry. …
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometr …
    -/
    rw [← IsIso.eq_inv_comp] at this
    /-
      case prop
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      x : ↑↑(Y.restrict ⋯).toPresheafedSpace
      this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometry.P …
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometr …
    -/
    rw [this]
    /-
      case prop
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      x : ↑↑(Y.restrict ⋯).toPresheafedSpace
      this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometry.P …
      ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (f.stalkM …
    -/
    dsimp
    /-
      case prop
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      x : ↑↑(Y.restrict ⋯).toPresheafedSpace
      this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometry.P …
      ⊢ IsLocalHom ((((AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (Y.ofRestrict  …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  · exact LocallyRingedSpace.Hom.ext'
        (PresheafedSpace.IsOpenImmersion.pullback_cone_of_left_condition _ _)


instance : LocallyRingedSpace.IsOpenImmersion (pullbackConeOfLeft f g).snd :=
                                                                             /-
                                                                               C : Type u
                                                                               inst✝ : CategoryTheory.Category.{v, u} C
                                                                               X Y Z : AlgebraicGeometry.LocallyRingedSpace
                                                                               f : Quiver.Hom X Z
                                                                               g : Quiver.Hom Y Z
                                                                               H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
                                                                               ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (Y.ofRestrict ⋯)
                                                                             -/
  show PresheafedSpace.IsOpenImmersion (Y.toPresheafedSpace.ofRestrict _) by infer_instance
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


set_option synthInstance.maxHeartbeats 40000 in
/-- The constructed `pullbackConeOfLeft` is indeed limiting. -/
def pullbackConeOfLeftIsLimit : IsLimit (pullbackConeOfLeft f g) :=
  PullbackCone.isLimitAux' _ fun s => by
    refine ⟨LocallyRingedSpace.Hom.mk (PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftLift
        f.1 g.1 (PullbackCone.mk _ _ (congr_arg LocallyRingedSpace.Hom.toShHom s.condition))) ?_,
      LocallyRingedSpace.Hom.ext'
        (PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftLift_fst f.1 g.1 _),
      LocallyRingedSpace.Hom.ext'
          (PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftLift_snd f.1 g.1 _), ?_⟩
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ ∀ (x : ↑↑s.pt.toPresheafedSpace), IsLocalHom (AlgebraicGeometry.PresheafedSp …
      -/
    · intro x
      have :=
        PresheafedSpace.stalkMap.congr_hom _ _
          (PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftLift_snd f.1 g.1
            (PullbackCone.mk s.fst.1 s.snd.1
              (congr_arg LocallyRingedSpace.Hom.toShHom s.condition)))
          x
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        s : CategoryTheory.Limits.PullbackCone f g
        x : ↑↑s.pt.toPresheafedSpace
        this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.Cate …
        ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometr …
      -/
      change _ = _ ≫ s.snd.1.stalkMap x at this
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        s : CategoryTheory.Limits.PullbackCone f g
        x : ↑↑s.pt.toPresheafedSpace
        this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.Cate …
        ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometr …
      -/
      rw [PresheafedSpace.stalkMap.comp, ← IsIso.eq_inv_comp] at this
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        s : CategoryTheory.Limits.PullbackCone f g
        x : ↑↑s.pt.toPresheafedSpace
        this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometry.P …
        ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometr …
      -/
      rw [this]
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        s : CategoryTheory.Limits.PullbackCone f g
        x : ↑↑s.pt.toPresheafedSpace
        this : Eq (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (AlgebraicGeometry.P …
        ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (Algebrai …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        s : CategoryTheory.Limits.PullbackCone f g
        ⊢ ∀ {m : Quiver.Hom s.pt (AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion …
      -/
    · intro m _ h₂
      /-
        case refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        s : CategoryTheory.Limits.PullbackCone f g
        m : Quiver.Hom s.pt (AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion.pull …
        a✝ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.LocallyRinged …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.LocallyRinged …
        ⊢ Eq m { toHom := AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.pullbackCo …
      -/
      rw [← cancel_mono (pullbackConeOfLeft f g).snd]
      exact h₂.trans <| LocallyRingedSpace.Hom.ext'
        (PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftLift_snd f.1 g.1 <|
          PullbackCone.mk s.fst.1 s.snd.1 <| congr_arg
            LocallyRingedSpace.Hom.toShHom s.condition).symm


/-- Open immersions are stable under base-change. -/
instance pullback_snd_of_left :
    LocallyRingedSpace.IsOpenImmersion (pullback.snd f g) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Limits. …
  -/
  delta pullback.snd
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Limits. …
  -/
  rw [← limit.isoLimitCone_hom_π ⟨_, pullbackConeOfLeftIsLimit f g⟩ WalkingCospan.right]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Categor …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Open immersions are stable under base-change. -/
instance pullback_fst_of_right :
    LocallyRingedSpace.IsOpenImmersion (pullback.fst g f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Limits. …
  -/
  rw [← pullbackSymmetry_hom_comp_snd]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Categor …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance pullback_to_base_isOpenImmersion [LocallyRingedSpace.IsOpenImmersion g] :
    LocallyRingedSpace.IsOpenImmersion (limit.π (cospan f g) WalkingCospan.one) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion g
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Limits. …
  -/
  rw [← limit.w (cospan f g) WalkingCospan.Hom.inl, cospan_map_inl]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion g
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Categor …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance forget_preservesPullbackOfLeft :
    PreservesLimit (cospan f g) LocallyRingedSpace.forgetToSheafedSpace :=
  preservesLimit_of_preserves_limit_cone (pullbackConeOfLeftIsLimit f g) <| by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Limits.IsLimit (AlgebraicGeometry.LocallyRingedSpace.forgetTo …
    -/
    apply (isLimitMapConePullbackConeEquiv _ _).symm.toFun
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (Algebr …
    -/
    apply isLimitOfIsLimitPullbackConeMap SheafedSpace.forgetToPresheafedSpace
    /-
      case l
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (Algebr …
    -/
    exact PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftIsLimit f.1 g.1
    /-
      🎉 no goals
    -/


instance forgetToPresheafedSpace_preservesPullback_of_left :
    PreservesLimit (cospan f g)
      (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace) :=
  preservesLimit_of_preserves_limit_cone (pullbackConeOfLeftIsLimit f g) <| by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Limits.IsLimit ((AlgebraicGeometry.LocallyRingedSpace.forgetT …
    -/
    apply (isLimitMapConePullbackConeEquiv _ _).symm.toFun
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk ((Algeb …
    -/
    exact PresheafedSpace.IsOpenImmersion.pullbackConeOfLeftIsLimit f.1 g.1
    /-
      🎉 no goals
    -/


instance forgetToPresheafedSpacePreservesOpenImmersion :
    PresheafedSpace.IsOpenImmersion
      ((LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace).map f) :=
  H


instance forgetToTop_preservesPullback_of_left :
    PreservesLimit (cospan f g)
      (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forget _) := by
  change PreservesLimit _ <|
    (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace) ⋙
    PresheafedSpace.forget _
  -- Porting note: was `apply (config := { instances := False }) ...`
  -- See https://github.com/leanprover/lean4/issues/2273
  have : PreservesLimit
      (cospan ((cospan f g ⋙ forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace).map
        WalkingCospan.Hom.inl)
      ((cospan f g ⋙ forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace).map
        WalkingCospan.Hom.inr)) (PresheafedSpace.forget CommRingCat) := by
    dsimp; infer_instance
  have : PreservesLimit (cospan f g ⋙ forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace)
      (PresheafedSpace.forget CommRingCat) := by
    apply preservesLimit_of_iso_diagram _ (diagramIsoCospan _).symm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    this✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan ((( …
    this : CategoryTheory.Limits.PreservesLimit ((CategoryTheory.Limits.cospan f g …
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g) ((Al …
  -/
  apply Limits.comp_preservesLimit
  /-
    🎉 no goals
  -/


instance forget_reflectsPullback_of_left :
    ReflectsLimit (cospan f g) LocallyRingedSpace.forgetToSheafedSpace :=
  reflectsLimit_of_reflectsIsomorphisms _ _


instance forget_preservesPullback_of_right :
    PreservesLimit (cospan g f) LocallyRingedSpace.forgetToSheafedSpace :=
  preservesPullback_symmetry _ _ _


instance forgetToPresheafedSpace_preservesPullback_of_right :
    PreservesLimit (cospan g f)
      (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace) :=
  preservesPullback_symmetry _ _ _


instance forget_reflectsPullback_of_right :
    ReflectsLimit (cospan g f) LocallyRingedSpace.forgetToSheafedSpace :=
  reflectsLimit_of_reflectsIsomorphisms _ _


instance forgetToPresheafedSpace_reflectsPullback_of_left :
    ReflectsLimit (cospan f g)
      (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace) :=
  reflectsLimit_of_reflectsIsomorphisms _ _


instance forgetToPresheafedSpace_reflectsPullback_of_right :
    ReflectsLimit (cospan g f)
      (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace) :=
  reflectsLimit_of_reflectsIsomorphisms _ _


theorem pullback_snd_isIso_of_range_subset (H' : Set.range g.base ⊆ Set.range f.base) :
    IsIso (pullback.snd f g) := by
  apply (config := {allowSynthFailures := true}) Functor.ReflectsIsomorphisms.reflects
    (F := LocallyRingedSpace.forgetToSheafedSpace)
  apply (config := {allowSynthFailures := true}) Functor.ReflectsIsomorphisms.reflects
    (F := SheafedSpace.forgetToPresheafedSpace)
  erw [← PreservesPullback.iso_hom_snd
      (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace) f g]
  -- Porting note: was `inferInstance`
  exact IsIso.comp_isIso' inferInstance <|
    PresheafedSpace.IsOpenImmersion.pullback_snd_isIso_of_range_subset _ _ H'


/-- The universal property of open immersions:
For an open immersion `f : X ⟶ Z`, given any morphism of schemes `g : Y ⟶ Z` whose topological
image is contained in the image of `f`, we can lift this morphism to a unique `Y ⟶ X` that
commutes with these maps.
-/
def lift (H' : Set.range g.base ⊆ Set.range f.base) : Y ⟶ X :=
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance manually
  have := pullback_snd_isIso_of_range_subset f g H'
  inv (pullback.snd f g) ≫ pullback.fst _ _


@[simp, reassoc]
theorem lift_fac (H' : Set.range g.base ⊆ Set.range f.base) : lift f g H' ≫ f = g := by
  /-
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  erw [Category.assoc]; rw [IsIso.inv_comp_eq]; exact pullback.condition
                                                /-
                                                  🎉 no goals
                                                -/


theorem lift_uniq (H' : Set.range g.base ⊆ Set.range f.base) (l : Y ⟶ X) (hl : l ≫ f = g) :
                          /-
                            X Y Z : AlgebraicGeometry.LocallyRingedSpace
                            f : Quiver.Hom X Z
                            g : Quiver.Hom Y Z
                            H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
                            H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
                            l : Quiver.Hom Y X
                            hl : Eq (CategoryTheory.CategoryStruct.comp l f) g
                            ⊢ Eq l (AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion.lift f g H')
                          -/
    l = lift f g H' := by rw [← cancel_mono f, hl, lift_fac]
                          /-
                            🎉 no goals
                          -/


theorem lift_range (H' : Set.range g.base ⊆ Set.range f.base) :
    Set.range (lift f g H').base = f.base ⁻¹' Set.range g.base := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance manually
  /-
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
    ⊢ Eq (Set.range ⇑(AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion.lift f  …
  -/
  have := pullback_snd_isIso_of_range_subset f g H'
  /-
    X Y Z : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
    this : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
    ⊢ Eq (Set.range ⇑(AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion.lift f  …
  -/
  dsimp only [lift]
  have : _ = (pullback.fst f g).base :=
    PreservesPullback.iso_hom_fst
      (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forget _) f g
  rw [LocallyRingedSpace.comp_base, ← this, ← Category.assoc, coe_comp, Set.range_comp,
      Set.range_eq_univ.mpr, Set.image_univ]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): change `rw` to `erw` on this lemma
    /-
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
      this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
      ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.pullback.fst ((AlgebraicGeometry.Local …
    -/
  · erw [TopCat.pullback_fst_range]
    /-
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
      this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
      ⊢ Eq (setOf fun x => Exists fun y => Eq (((AlgebraicGeometry.LocallyRingedSpac …
    -/
    ext
    /-
      case h
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
      this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
      x✝ : ↑((AlgebraicGeometry.LocallyRingedSpace.forgetToSheafedSpace.comp (Algebr …
      ⊢ Iff (Membership.mem (setOf fun x => Exists fun y => Eq (((AlgebraicGeometry. …
    -/
    constructor
      /-
        case h.mp
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
        this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
        x✝ : ↑((AlgebraicGeometry.LocallyRingedSpace.forgetToSheafedSpace.comp (Algebr …
        ⊢ Membership.mem (setOf fun x => Exists fun y => Eq (((AlgebraicGeometry.Local …
      -/
    · rintro ⟨y, eq⟩; exact ⟨y, eq.symm⟩
                      /-
                        🎉 no goals
                      -/
      /-
        case h.mpr
        X Y Z : AlgebraicGeometry.LocallyRingedSpace
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
        this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
        x✝ : ↑((AlgebraicGeometry.LocallyRingedSpace.forgetToSheafedSpace.comp (Algebr …
        ⊢ Membership.mem (Set.preimage (⇑f.base) (Set.range ⇑g.base)) x✝ → Membership. …
      -/
    · rintro ⟨y, eq⟩; exact ⟨y, eq.symm⟩
                      /-
                        🎉 no goals
                      -/
  · rw [← TopCat.epi_iff_surjective, show (inv (pullback.snd f g)).base = _ from
        (LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forget _).map_inv _]
    /-
      X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
      this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv ( …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- An open immersion is isomorphic to the induced open subscheme on its image. -/
noncomputable def isoRestrict {X Y : LocallyRingedSpace} (f : X ⟶ Y)
    [H : LocallyRingedSpace.IsOpenImmersion f] :
    X ≅ Y.restrict H.base_open :=
  LocallyRingedSpace.isoOfSheafedSpaceIso <|
    SheafedSpace.forgetToPresheafedSpace.preimageIso <|
      PresheafedSpace.IsOpenImmersion.isoRestrict f.1


/-- The functor `Opens X ⥤ Opens Y` associated with an open immersion `f : X ⟶ Y`. -/
abbrev opensFunctor {X Y : LocallyRingedSpace} (f : X ⟶ Y)
    [H : LocallyRingedSpace.IsOpenImmersion f] : Opens X ⥤ Opens Y :=
  H.base_open.isOpenMap.functor


/-- Suppose `X Y : SheafedSpace C`, where `C` is a concrete category,
whose forgetful functor reflects isomorphisms, preserves limits and filtered colimits.
Then a morphism `X ⟶ Y` that is a topological open embedding
is an open immersion iff every stalk map is an iso.
-/
theorem of_stalk_iso {X Y : LocallyRingedSpace} (f : X ⟶ Y) (hf : IsOpenEmbedding f.base)
    [stalk_iso : ∀ x : X.1, IsIso (f.stalkMap x)] :
    LocallyRingedSpace.IsOpenImmersion f :=
  SheafedSpace.IsOpenImmersion.of_stalk_iso _ hf (H := stalk_iso)


@[reassoc (attr := simp)]
theorem isoRestrict_hom_ofRestrict : (isoRestrict f).hom ≫ Y.ofRestrict _ = f := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  ext1
  /-
    case h
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (CategoryTheory.Categor …
  -/
  dsimp [isoRestrict, isoOfSheafedSpaceIso]
  /-
    case h
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.SheafedSpace.forge …
  -/
  apply SheafedSpace.forgetToPresheafedSpace.map_injective
  /-
    case h.a
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ Eq (AlgebraicGeometry.SheafedSpace.forgetToPresheafedSpace.map (CategoryTheo …
  -/
  rw [Functor.map_comp, SheafedSpace.forgetToPresheafedSpace.map_preimage]
  /-
    case h.a
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  exact SheafedSpace.IsOpenImmersion.isoRestrict_hom_ofRestrict f.1
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem isoRestrict_inv_ofRestrict : (isoRestrict f).inv ≫ f = Y.ofRestrict _ := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  simp only [← isoRestrict_hom_ofRestrict f, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/

/-- For an open immersion `f : X ⟶ Y` and an open set `U ⊆ X`, we have the map `X(U) ⟶ Y(U)`. -/
noncomputable def invApp (U : Opens X) :
    X.presheaf.obj (op U) ⟶ Y.presheaf.obj (op (opensFunctor f |>.obj U)) :=
  PresheafedSpace.IsOpenImmersion.invApp f.1 U


@[reassoc (attr := simp)]
theorem inv_naturality {U V : (Opens X)ᵒᵖ} (i : U ⟶ V) :
    X.presheaf.map i ≫ H.invApp _ (unop V) =
      H.invApp _ (unop U) ≫ Y.presheaf.map (opensFunctor f |>.op.map i) :=
  PresheafedSpace.IsOpenImmersion.inv_naturality f.1 i


                                                    /-
                                                      C : Type u
                                                      inst✝ : CategoryTheory.Category.{v, u} C
                                                      X Y : AlgebraicGeometry.LocallyRingedSpace
                                                      f : Quiver.Hom X Y
                                                      H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
                                                      U : TopologicalSpace.Opens ↑X.toTopCat
                                                      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion.i …
                                                    -/
instance (U : Opens X) : IsIso (H.invApp _ U) := by delta invApp; infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem inv_invApp (U : Opens X) :
    inv (H.invApp _ U) =
      f.c.app (op (opensFunctor f |>.obj U)) ≫ X.presheaf.map
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       X Y : AlgebraicGeometry.LocallyRingedSpace
                       f : Quiver.Hom X Y
                       H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
                       U : TopologicalSpace.Opens ↑X.toTopCat
                       ⊢ Eq ((TopologicalSpace.Opens.map f.base).op.obj { unop := (AlgebraicGeometry. …
                     -/
        (eqToHom (by simp [Opens.map, Set.preimage_image_eq _ H.base_open.injective])) :=
                     /-
                       🎉 no goals
                     -/
  PresheafedSpace.IsOpenImmersion.inv_invApp f.1 U


@[reassoc (attr := simp)]
theorem invApp_app (U : Opens X) :
    H.invApp _ U ≫ f.c.app (op (opensFunctor f |>.obj U)) = X.presheaf.map
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     X Y : AlgebraicGeometry.LocallyRingedSpace
                     f : Quiver.Hom X Y
                     H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
                     U : TopologicalSpace.Opens ↑X.toTopCat
                     ⊢ Eq { unop := U } ((TopologicalSpace.Opens.map f.base).op.obj { unop := (Alge …
                   -/
      (eqToHom (by simp [Opens.map, Set.preimage_image_eq _ H.base_open.injective])) :=
                   /-
                     🎉 no goals
                   -/
  PresheafedSpace.IsOpenImmersion.invApp_app f.1 U


attribute [elementwise nosimp] invApp_app


@[reassoc (attr := simp)]
theorem app_invApp (U : Opens Y) :
    f.c.app (op U) ≫ H.invApp _ ((Opens.map f.base).obj U) =
      Y.presheaf.map
        ((homOfLE (Set.image_preimage_subset f.base U.1)).op :
          op U ⟶ op (opensFunctor f |>.obj ((Opens.map f.base).obj U))) :=
  PresheafedSpace.IsOpenImmersion.app_invApp f.1 U


/-- A variant of `app_inv_app` that gives an `eqToHom` instead of `homOfLe`. -/
@[reassoc]
theorem app_inv_app' (U : Opens Y) (hU : (U : Set Y) ⊆ Set.range f.base) :
    f.c.app (op U) ≫ H.invApp _ ((Opens.map f.base).obj U) =
      Y.presheaf.map
        (eqToHom <|
            le_antisymm (Set.image_preimage_subset f.base U.1) <|
              (Set.image_preimage_eq_inter_range (f := f.base) (t := U.1)).symm ▸
                Set.subset_inter_iff.mpr ⟨fun _ h => h, hU⟩).op :=
  PresheafedSpace.IsOpenImmersion.app_invApp f.1 U


instance ofRestrict {X : TopCat} (Y : LocallyRingedSpace) {f : X ⟶ Y.carrier}
    (hf : IsOpenEmbedding f) : IsOpenImmersion (Y.ofRestrict hf) :=
  PresheafedSpace.IsOpenImmersion.ofRestrict _ hf


@[elementwise, simp]
theorem ofRestrict_invApp (X : LocallyRingedSpace) {Y : TopCat}
    {f : Y ⟶ TopCat.of X.carrier} (h : IsOpenEmbedding f) (U : Opens (X.restrict h).carrier) :
    (LocallyRingedSpace.IsOpenImmersion.ofRestrict X h).invApp _ U = 𝟙 _ :=
  PresheafedSpace.IsOpenImmersion.ofRestrict_invApp _ h U


instance stalk_iso (x : X) : IsIso (f.stalkMap x) :=
  PresheafedSpace.IsOpenImmersion.stalk_iso f.1 x


theorem to_iso [h' : Epi f.base] : IsIso f := by
  suffices IsIso (LocallyRingedSpace.forgetToSheafedSpace.map f) from
    isIso_of_reflects_iso _ LocallyRingedSpace.forgetToSheafedSpace
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
    h' : CategoryTheory.Epi f.base
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.LocallyRingedSpace.forgetToSheafedSp …
  -/
  exact SheafedSpace.IsOpenImmersion.to_iso f.1
  /-
    🎉 no goals
  -/


