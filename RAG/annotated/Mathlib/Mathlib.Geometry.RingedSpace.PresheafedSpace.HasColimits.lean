@[simp]
theorem map_id_c_app (F : J ⥤ PresheafedSpace.{_, _, v} C) (j) (U) :
    (F.map (𝟙 j)).c.app U =
      (Pushforward.id (F.obj j).presheaf).inv.app U ≫
                           /-
                             J : Type u'
                             inst✝¹ : CategoryTheory.Category.{v', u'} J
                             C : Type u
                             inst✝ : CategoryTheory.Category.{v, u} C
                             F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
                             j : J
                             U : Opposite (TopologicalSpace.Opens ↑↑(F.obj j))
                             ⊢ Eq (CategoryTheory.CategoryStruct.id ↑(F.obj j)) (F.map (CategoryTheory.Cate …
                           -/
        (pushforwardEq (by simp) (F.obj j).presheaf).hom.app U := by
                           /-
                             🎉 no goals
                           -/
  /-
    J : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    j : J
    U : Opposite (TopologicalSpace.Opens ↑↑(F.obj j))
    ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.id j)).c.app U) (CategoryTheory.Ca …
  -/
  simp [PresheafedSpace.congr_app (F.map_id j)]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp_c_app (F : J ⥤ PresheafedSpace.{_, _, v} C) {j₁ j₂ j₃}
    (f : j₁ ⟶ j₂) (g : j₂ ⟶ j₃) (U) :
    (F.map (f ≫ g)).c.app U =
      (F.map g).c.app U ≫
        ((pushforward C (F.map g).base).map (F.map f).c).app U ≫
          (pushforwardEq (congr_arg Hom.base (F.map_comp f g).symm) _).hom.app U := by
  /-
    J : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    j₁ j₂ j₃ : J
    f : Quiver.Hom j₁ j₂
    g : Quiver.Hom j₂ j₃
    U : Opposite (TopologicalSpace.Opens ↑↑(F.obj j₃))
    ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.comp f g)).c.app U) (CategoryTheor …
  -/
  simp [PresheafedSpace.congr_app (F.map_comp f g)]
  /-
    🎉 no goals
  -/

-- See note [dsimp, simp]

/-- Given a diagram of `PresheafedSpace C`s, its colimit is computed by pushing the sheaves onto
the colimit of the underlying spaces, and taking componentwise limit.
This is the componentwise diagram for an open set `U` of the colimit of the underlying spaces.
-/
@[simps]
def componentwiseDiagram (F : J ⥤ PresheafedSpace.{_, _, v} C) [HasColimit F]
    (U : Opens (Limits.colimit F).carrier) : Jᵒᵖ ⥤ C where
  obj j := (F.obj (unop j)).presheaf.obj (op ((Opens.map (colimit.ι F (unop j)).base).obj U))
  map {j k} f := (F.map f.unop).c.app _ ≫
                                               /-
                                                 J : Type u'
                                                 inst✝² : CategoryTheory.Category.{v', u'} J
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
                                                 inst✝ : CategoryTheory.Limits.HasColimit F
                                                 U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
                                                 j k : Opposite J
                                                 f : Quiver.Hom j k
                                                 ⊢ Eq ((TopologicalSpace.Opens.map (F.map f.unop).base).op.obj { unop := (Topol …
                                               -/
    (F.obj (unop k)).presheaf.map (eqToHom (by rw [← colimit.w F f.unop, comp_base]; rfl))
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
  map_comp {i j k} f g := by
    /-
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      i j k : Opposite J
      f : Quiver.Hom i j
      g : Quiver.Hom j k
      ⊢ Eq ({ obj := fun j => (F.obj (Opposite.unop j)).presheaf.obj { unop := (Topo …
    -/
    dsimp
    /-
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      i j k : Opposite J
      f : Quiver.Hom i j
      g : Quiver.Hom j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map (CategoryTheory.CategoryStruc …
    -/
    simp only [assoc, CategoryTheory.NatTrans.naturality_assoc]
    /-
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      i j k : Opposite J
      f : Quiver.Hom i j
      g : Quiver.Hom j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map (CategoryTheory.CategoryStruc …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Given a diagram of presheafed spaces,
we can push all the presheaves forward to the colimit `X` of the underlying topological spaces,
obtaining a diagram in `(Presheaf C X)ᵒᵖ`.
-/
@[simps]
def pushforwardDiagramToColimit (F : J ⥤ PresheafedSpace.{_, _, v} C) :
    J ⥤ (Presheaf C (colimit (F ⋙ PresheafedSpace.forget C)))ᵒᵖ where
  obj j := op (colimit.ι (F ⋙ PresheafedSpace.forget C) j _* (F.obj j).presheaf)
  map {j j'} f :=
    ((pushforward C (colimit.ι (F ⋙ PresheafedSpace.forget C) j')).map (F.map f).c ≫
      (Pushforward.comp ((F ⋙ PresheafedSpace.forget C).map f)
        (colimit.ι (F ⋙ PresheafedSpace.forget C) j') (F.obj j).presheaf).inv ≫
      (pushforwardEq (colimit.w (F ⋙ PresheafedSpace.forget C) f) (F.obj j).presheaf).hom).op
  map_id j := by
    /-
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      j : J
      ⊢ Eq ({ obj := fun j => { unop := (TopCat.Presheaf.pushforward C (CategoryTheo …
    -/
    apply (opEquiv _ _).injective
    /-
      case a
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      j : J
      ⊢ Eq ((CategoryTheory.opEquiv ({ obj := fun j => { unop := (TopCat.Presheaf.pu …
    -/
    refine NatTrans.ext (funext fun U => ?_)
    induction U with
    | h U =>
      simp [opEquiv]
      rfl
  map_comp {j₁ j₂ j₃} f g := by
    /-
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      j₁ j₂ j₃ : J
      f : Quiver.Hom j₁ j₂
      g : Quiver.Hom j₂ j₃
      ⊢ Eq ({ obj := fun j => { unop := (TopCat.Presheaf.pushforward C (CategoryTheo …
    -/
    apply (opEquiv _ _).injective
    /-
      case a
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      j₁ j₂ j₃ : J
      f : Quiver.Hom j₁ j₂
      g : Quiver.Hom j₂ j₃
      ⊢ Eq ((CategoryTheory.opEquiv ({ obj := fun j => { unop := (TopCat.Presheaf.pu …
    -/
    refine NatTrans.ext (funext fun U => ?_)
    /-
      case a
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      j₁ j₂ j₃ : J
      f : Quiver.Hom j₁ j₂
      g : Quiver.Hom j₂ j₃
      U : Opposite (TopologicalSpace.Opens ↑(CategoryTheory.Limits.colimit (F.comp ( …
      ⊢ Eq (((CategoryTheory.opEquiv ({ obj := fun j => { unop := (TopCat.Presheaf.p …
    -/
    dsimp [opEquiv]
    have :
      op ((Opens.map (F.map g).base).obj
          ((Opens.map (colimit.ι (F ⋙ forget C) j₃)).obj U.unop)) =
        op ((Opens.map (colimit.ι (F ⋙ PresheafedSpace.forget C) j₂)).obj (unop U)) := by
      apply unop_injective
      rw [← Opens.map_comp_obj]
      congr
      exact colimit.w (F ⋙ PresheafedSpace.forget C) g
    simp only [map_comp_c_app, pushforward_obj_obj, pushforward_map_app, comp_base,
      pushforwardEq_hom_app, op_obj, Opens.map_comp_obj, id_comp, assoc, eqToHom_map_comp,
      NatTrans.naturality_assoc, pushforward_obj_map, eqToHom_unop]
    /-
      case a
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      j₁ j₂ j₃ : J
      f : Quiver.Hom j₁ j₂
      g : Quiver.Hom j₂ j₃
      U : Opposite (TopologicalSpace.Opens ↑(CategoryTheory.Limits.colimit (F.comp ( …
      this : Eq { unop := (TopologicalSpace.Opens.map (F.map g).base).obj ((Topologi …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map g).c.app { unop := (Topologic …
    -/
    simp [NatTrans.congr (α := (F.map f).c) this]
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `AlgebraicGeometry.PresheafedSpace.instHasColimits`.
-/
def colimit (F : J ⥤ PresheafedSpace.{_, _, v} C) : PresheafedSpace C where
  carrier := Limits.colimit (F ⋙ PresheafedSpace.forget C)
  presheaf := limit (pushforwardDiagramToColimit F).leftOp


@[simp]
theorem colimit_carrier (F : J ⥤ PresheafedSpace.{_, _, v} C) :
    (colimit F).carrier = Limits.colimit (F ⋙ PresheafedSpace.forget C) :=
  rfl


@[simp]
theorem colimit_presheaf (F : J ⥤ PresheafedSpace.{_, _, v} C) :
    (colimit F).presheaf = limit (pushforwardDiagramToColimit F).leftOp :=
  rfl


/-- Auxiliary definition for `AlgebraicGeometry.PresheafedSpace.instHasColimits`.
-/
@[simps]
def colimitCocone (F : J ⥤ PresheafedSpace.{_, _, v} C) : Cocone F where
  pt := colimit F
  ι :=
    { app := fun j =>
        { base := colimit.ι (F ⋙ PresheafedSpace.forget C) j
          c := limit.π _ (op j) }
      naturality := fun {j j'} f => by
        /-
          J : Type u'
          inst✝³ : CategoryTheory.Category.{v', u'} J
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
          inst✝ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) (T …
          F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { base := Catego …
        -/
        ext1
          /-
            case w
            J : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
            inst✝ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) (T …
            F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
            j j' : J
            f : Quiver.Hom j j'
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { base := Catego …
          -/
        · ext x
          /-
            case w.w
            J : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
            inst✝ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) (T …
            F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
            j j' : J
            f : Quiver.Hom j j'
            x : (CategoryTheory.forget TopCat).obj ↑(F.obj j)
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { base := Categ …
          -/
          exact colimit.w_apply (F ⋙ PresheafedSpace.forget C) f x
          /-
            🎉 no goals
          -/
          /-
            case h
            J : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
            inst✝ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) (T …
            F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
            j j' : J
            f : Quiver.Hom j j'
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · ext ⟨U, hU⟩
          /-
            case h.w.mk
            J : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
            inst✝ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) (T …
            F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
            j j' : J
            f : Quiver.Hom j j'
            U : Set ↑↑(((CategoryTheory.Functor.const J).obj (AlgebraicGeometry.Presheafed …
            hU : IsOpen U
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
          -/
          dsimp [-Presheaf.comp_app]
          /-
            case h.w.mk
            J : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
            inst✝ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) (T …
            F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
            j j' : J
            f : Quiver.Hom j j'
            U : Set ↑↑(((CategoryTheory.Functor.const J).obj (AlgebraicGeometry.Presheafed …
            hU : IsOpen U
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
          -/
          rw [PresheafedSpace.id_c_app, map_id]
          /-
            case h.w.mk
            J : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
            inst✝ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) (T …
            F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
            j j' : J
            f : Quiver.Hom j j'
            U : Set ↑↑(((CategoryTheory.Functor.const J).obj (AlgebraicGeometry.Presheafed …
            hU : IsOpen U
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
          -/
          erw [id_comp]
          rw [NatTrans.comp_app, PresheafedSpace.comp_c_app, whiskerRight_app, eqToHom_app,
            ← congr_arg NatTrans.app (limit.w (pushforwardDiagramToColimit F).leftOp f.op),
            NatTrans.comp_app, Functor.leftOp_map, pushforwardDiagramToColimit_map]
          /-
            case h.w.mk
            J : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
            inst✝ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) (T …
            F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
            j j' : J
            f : Quiver.Hom j j'
            U : Set ↑↑(((CategoryTheory.Functor.const J).obj (AlgebraicGeometry.Presheafed …
            hU : IsOpen U
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          simp }
          /-
            🎉 no goals
          -/


/-- Auxiliary definition for `AlgebraicGeometry.PresheafedSpace.colimitCoconeIsColimit`.
-/
def descCApp (F : J ⥤ PresheafedSpace.{_, _, v} C) (s : Cocone F) (U : (Opens s.pt.carrier)ᵒᵖ) :
    s.pt.presheaf.obj U ⟶
      (colimit.desc (F ⋙ PresheafedSpace.forget C) ((PresheafedSpace.forget C).mapCocone s) _*
            limit (pushforwardDiagramToColimit F).leftOp).obj
        U := by
  refine
    limit.lift _
        { pt := s.pt.presheaf.obj U
          π :=
            { app := fun j => ?_
              naturality := fun j j' f => ?_ } } ≫
      (limitObjIsoLimitCompEvaluation _ _).inv
  -- We still need to construct the `app` and `naturality'` fields omitted above.
    /-
      case refine_1
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j : Opposite J
      ⊢ Quiver.Hom (((CategoryTheory.Functor.const (Opposite J)).obj (s.pt.presheaf. …
    -/
  · refine (s.ι.app (unop j)).c.app U ≫ (F.obj (unop j)).presheaf.map (eqToHom ?_)
    /-
      case refine_1
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j : Opposite J
      ⊢ Eq ((TopologicalSpace.Opens.map (s.ι.app (Opposite.unop j)).base).op.obj U)  …
    -/
    dsimp
    /-
      case refine_1
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j : Opposite J
      ⊢ Eq { unop := (TopologicalSpace.Opens.map (s.ι.app (Opposite.unop j)).base).o …
    -/
    rw [← Opens.map_comp_obj]
    /-
      case refine_1
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j : Opposite J
      ⊢ Eq { unop := (TopologicalSpace.Opens.map (s.ι.app (Opposite.unop j)).base).o …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j j' : Opposite J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
    -/
  · dsimp
    /-
      case refine_2
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j j' : Opposite J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (s. …
    -/
    rw [PresheafedSpace.congr_app (s.w f.unop).symm U]
    have w :=
      Functor.congr_obj
        (congr_arg Opens.map (colimit.ι_desc ((PresheafedSpace.forget C).mapCocone s) (unop j)))
        (unop U)
    /-
      case refine_2
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j j' : Opposite J
      f : Quiver.Hom j j'
      w : Eq ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStruct.comp (Categ …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (s. …
    -/
    simp only [Opens.map_comp_obj_unop] at w
    /-
      case refine_2
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j j' : Opposite J
      f : Quiver.Hom j j'
      w : Eq ((TopologicalSpace.Opens.map (CategoryTheory.Limits.colimit.ι (F.comp ( …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (s. …
    -/
    replace w := congr_arg op w
    /-
      case refine_2
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j j' : Opposite J
      f : Quiver.Hom j j'
      w : Eq { unop := (TopologicalSpace.Opens.map (CategoryTheory.Limits.colimit.ι  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (s. …
    -/
    have w' := NatTrans.congr (F.map f.unop).c w
    /-
      case refine_2
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j j' : Opposite J
      f : Quiver.Hom j j'
      w : Eq { unop := (TopologicalSpace.Opens.map (CategoryTheory.Limits.colimit.ι  …
      w' : Eq ((F.map f.unop).c.app { unop := (TopologicalSpace.Opens.map (CategoryT …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (s. …
    -/
    rw [w']
    /-
      case refine_2
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      U : Opposite (TopologicalSpace.Opens ↑↑s.pt)
      j j' : Opposite J
      f : Quiver.Hom j j'
      w : Eq { unop := (TopologicalSpace.Opens.map (CategoryTheory.Limits.colimit.ι  …
      w' : Eq ((F.map f.unop).c.app { unop := (TopologicalSpace.Opens.map (CategoryT …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (s. …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem desc_c_naturality (F : J ⥤ PresheafedSpace.{_, _, v} C) (s : Cocone F)
    {U V : (Opens s.pt.carrier)ᵒᵖ} (i : U ⟶ V) :
    s.pt.presheaf.map i ≫ descCApp F s V =
      descCApp F s U ≫
        (colimit.desc (F ⋙ forget C) ((forget C).mapCocone s) _* (colimitCocone F).pt.presheaf).map
          i := by
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    s : CategoryTheory.Limits.Cocone F
    U V : Opposite (TopologicalSpace.Opens ↑↑s.pt)
    i : Quiver.Hom U V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.pt.presheaf.map i) (AlgebraicGeome …
  -/
  dsimp [descCApp]
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    s : CategoryTheory.Limits.Cocone F
    U V : Opposite (TopologicalSpace.Opens ↑↑s.pt)
    i : Quiver.Hom U V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.pt.presheaf.map i) (CategoryTheory …
  -/
  refine limit_obj_ext (fun j => ?_)
  have w := Functor.congr_hom (congr_arg Opens.map
    (colimit.ι_desc ((PresheafedSpace.forget C).mapCocone s) (unop j))) i.unop
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    s : CategoryTheory.Limits.Cocone F
    U V : Opposite (TopologicalSpace.Opens ↑↑s.pt)
    i : Quiver.Hom U V
    j : Opposite J
    w : Eq ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStruct.comp (Categ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Opens.map_comp_map] at w
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    s : CategoryTheory.Limits.Cocone F
    U V : Opposite (TopologicalSpace.Opens ↑↑s.pt)
    i : Quiver.Hom U V
    j : Opposite J
    w : Eq ((TopologicalSpace.Opens.map (CategoryTheory.Limits.colimit.ι (F.comp ( …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [congr_arg Quiver.Hom.op w]
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `AlgebraicGeometry.PresheafedSpace.colimitCoconeIsColimit`.
-/
def desc (F : J ⥤ PresheafedSpace.{_, _, v} C) (s : Cocone F) : colimit F ⟶ s.pt where
  base := colimit.desc (F ⋙ PresheafedSpace.forget C) ((PresheafedSpace.forget C).mapCocone s)
  c :=
    { app := fun U => descCApp F s U
      naturality := fun _ _ i => desc_c_naturality F s i }


theorem desc_fac (F : J ⥤ PresheafedSpace.{_, _, v} C) (s : Cocone F) (j : J) :
    (colimitCocone F).ι.app j ≫ desc F s = s.ι.app j := by
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    s : CategoryTheory.Limits.Cocone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.PresheafedSpace.c …
  -/
  ext U
    /-
      case w.w
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      j : J
      U : (CategoryTheory.forget TopCat).obj ↑(F.obj j)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.PresheafedSpace. …
    -/
  · simp [desc]
    /-
      🎉 no goals
    -/
  · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): the original proof is just `ext; dsimp [desc, descCApp]; simpa`,
    -- but this has to be expanded a bit
    /-
      case h.w
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      j : J
      U : TopologicalSpace.Opens ↑↑s.pt
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    rw [NatTrans.comp_app, PresheafedSpace.comp_c_app, whiskerRight_app]
    /-
      case h.w
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      j : J
      U : TopologicalSpace.Opens ↑↑s.pt
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp [desc, descCApp]
    /-
      case h.w
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      j : J
      U : TopologicalSpace.Opens ↑↑s.pt
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [eqToHom_app, op_obj, Opens.map_comp_obj, eqToHom_map, Functor.leftOp, assoc]
    /-
      case h.w
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      j : J
      U : TopologicalSpace.Opens ↑↑s.pt
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift ({  …
    -/
    rw [limitObjIsoLimitCompEvaluation_inv_π_app_assoc]
    /-
      case h.w
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      j : J
      U : TopologicalSpace.Opens ↑↑s.pt
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift ({  …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `AlgebraicGeometry.PresheafedSpace.instHasColimits`.
-/
def colimitCoconeIsColimit (F : J ⥤ PresheafedSpace.{_, _, v} C) :
    IsColimit (colimitCocone F) where
  desc s := desc F s
  fac s := desc_fac F s
  uniq s m w := by
    -- We need to use the identity on the continuous maps twice, so we prepare that first:
    have t :
      m.base =
        colimit.desc (F ⋙ PresheafedSpace.forget C) ((PresheafedSpace.forget C).mapCocone s) := by
      dsimp
      ext j
      rw [colimit.ι_desc, mapCocone_ι_app, ← w j]
      simp
    /-
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (AlgebraicGeometry.PresheafedSpace.colimitCocone F).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Pres …
      t : Eq m.base (CategoryTheory.Limits.colimit.desc (F.comp (AlgebraicGeometry.P …
      ⊢ Eq m ((fun s => AlgebraicGeometry.PresheafedSpace.ColimitCoconeIsColimit.des …
    -/
    ext : 1
      /-
        case w
        J : Type u'
        inst✝⁴ : CategoryTheory.Category.{v', u'} J
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
        inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
        F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom (AlgebraicGeometry.PresheafedSpace.colimitCocone F).pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Pres …
        t : Eq m.base (CategoryTheory.Limits.colimit.desc (F.comp (AlgebraicGeometry.P …
        ⊢ Eq m.base ((fun s => AlgebraicGeometry.PresheafedSpace.ColimitCoconeIsColimi …
      -/
    · exact t
      /-
        🎉 no goals
      -/
      /-
        case h
        J : Type u'
        inst✝⁴ : CategoryTheory.Category.{v', u'} J
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
        inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
        F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom (AlgebraicGeometry.PresheafedSpace.colimitCocone F).pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Pres …
        t : Eq m.base (CategoryTheory.Limits.colimit.desc (F.comp (AlgebraicGeometry.P …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m.c (CategoryTheory.whiskerRight (Cat …
      -/
    · refine NatTrans.ext (funext fun U => limit_obj_ext fun j => ?_)
      simp [desc, descCApp,
        PresheafedSpace.congr_app (w (unop j)).symm U,
        NatTrans.congr (limit.π (pushforwardDiagramToColimit F).leftOp j)
        (congr_arg op (Functor.congr_obj (congr_arg Opens.map t) (unop U)))]


instance : HasColimitsOfShape J (PresheafedSpace.{_, _, v} C) where
  has_colimit F := ⟨colimitCocone F, colimitCoconeIsColimit F⟩


instance : PreservesColimitsOfShape J (PresheafedSpace.forget.{u, v, v} C) :=
  ⟨fun {F} => preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit F) <| by
    /-
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      ⊢ CategoryTheory.Limits.IsColimit ((AlgebraicGeometry.PresheafedSpace.forget C …
    -/
    apply IsColimit.ofIsoColimit (colimit.isColimit _)
    /-
      J : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit.cocone (F.comp (AlgebraicG …
    -/
    fapply Cocones.ext
      /-
        case φ
        J : Type u'
        inst✝⁴ : CategoryTheory.Category.{v', u'} J
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
        inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
        F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
        ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit.cocone (F.comp (AlgebraicG …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case w
        J : Type u'
        inst✝⁴ : CategoryTheory.Category.{v', u'} J
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
        inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
        F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
        ⊢ autoParam (∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
      -/
    · intro j
      /-
        case w
        J : Type u'
        inst✝⁴ : CategoryTheory.Category.{v', u'} J
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
        inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
        F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
      -/
      simp⟩
      /-
        🎉 no goals
      -/


/-- When `C` has limits, the category of presheaved spaces with values in `C` itself has colimits.
-/
instance instHasColimits [HasLimits C] : HasColimits (PresheafedSpace.{_, _, v} C) :=
  ⟨fun {_ _} => ⟨fun {F} => ⟨colimitCocone F, colimitCoconeIsColimit F⟩⟩⟩


/-- The underlying topological space of a colimit of presheaved spaces is
the colimit of the underlying topological spaces.
-/
instance forget_preservesColimits [HasLimits C] :
    PreservesColimits (PresheafedSpace.forget.{_, _, v} C) where
  preservesColimitsOfShape {J 𝒥} :=
    { preservesColimit := fun {F} => preservesColimit_of_preserves_colimit_cocone
          (colimitCoconeIsColimit F)
                                                         /-
                                                           J✝ : Type u'
                                                           inst✝⁵ : CategoryTheory.Category.{v', u'} J✝
                                                           C : Type u
                                                           inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                           inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J✝ TopCat
                                                           inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J✝)  …
                                                           inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J✝) C
                                                           inst✝ : CategoryTheory.Limits.HasLimits C
                                                           J : Type v
                                                           𝒥 : CategoryTheory.Category.{v, v} J
                                                           F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
                                                           ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.co …
                                                         -/
          (IsColimit.ofIsoColimit (colimit.isColimit _) (Cocones.ext (Iso.refl _))) }
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- The components of the colimit of a diagram of `PresheafedSpace C` is obtained
via taking componentwise limits.
-/
def colimitPresheafObjIsoComponentwiseLimit (F : J ⥤ PresheafedSpace.{_, _, v} C) [HasColimit F]
    (U : Opens (Limits.colimit F).carrier) :
    (Limits.colimit F).presheaf.obj (op U) ≅ limit (componentwiseDiagram F U) := by
  refine
    ((sheafIsoOfIso (colimit.isoColimitCocone ⟨_, colimitCoconeIsColimit F⟩).symm).app
          (op U)).trans
      ?_
  /-
    J : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    inst✝ : CategoryTheory.Limits.HasColimit F
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    ⊢ CategoryTheory.Iso (((TopCat.Presheaf.pushforward C (CategoryTheory.Limits.c …
  -/
  refine (limitObjIsoLimitCompEvaluation _ _).trans (Limits.lim.mapIso ?_)
  /-
    J : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    inst✝ : CategoryTheory.Limits.HasColimit F
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    ⊢ CategoryTheory.Iso ((AlgebraicGeometry.PresheafedSpace.pushforwardDiagramToC …
  -/
  fapply NatIso.ofComponents
    /-
      case app
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      ⊢ (X : Opposite J) → CategoryTheory.Iso (((AlgebraicGeometry.PresheafedSpace.p …
    -/
  · intro X
    /-
      case app
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X : Opposite J
      ⊢ CategoryTheory.Iso (((AlgebraicGeometry.PresheafedSpace.pushforwardDiagramTo …
    -/
    refine (F.obj (unop X)).presheaf.mapIso (eqToIso ?_)
    simp only [Functor.op_obj, unop_op, op_inj_iff, Opens.map_coe, SetLike.ext'_iff,
      Set.preimage_preimage]
    /-
      case app
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X : Opposite J
      ⊢ Eq (Set.preimage (fun x => (CategoryTheory.Limits.colimit.isoColimitCocone { …
    -/
    refine congr_arg (Set.preimage · U.1) (funext fun x => ?_)
    /-
      case app
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X : Opposite J
      x : ↑((F.comp (AlgebraicGeometry.PresheafedSpace.forget C)).obj (Opposite.unop …
      ⊢ Eq ((CategoryTheory.Limits.colimit.isoColimitCocone { cocone := AlgebraicGeo …
    -/
    erw [← TopCat.comp_app]
    /-
      case app
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X : Opposite J
      x : ↑((F.comp (AlgebraicGeometry.PresheafedSpace.forget C)).obj (Opposite.unop …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F. …
    -/
    congr
    /-
      case app.e_a
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X : Opposite J
      x : ↑((F.comp (AlgebraicGeometry.PresheafedSpace.forget C)).obj (Opposite.unop …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
    -/
    exact ι_preservesColimitIso_inv (forget C) F (unop X)
    /-
      🎉 no goals
    -/
    /-
      case naturality
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      ⊢ autoParam (∀ {X Y : Opposite J} (f : Quiver.Hom X Y), Eq (CategoryTheory.Cat …
    -/
  · intro X Y f
    /-
      case naturality
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X Y : Opposite J
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicGeometry.PresheafedSpace. …
    -/
    change ((F.map f.unop).c.app _ ≫ _ ≫ _) ≫ (F.obj (unop Y)).presheaf.map _ = _ ≫ _
    /-
      case naturality
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X Y : Opposite J
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [TopCat.Presheaf.Pushforward.comp_inv_app]
    /-
      case naturality
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X Y : Opposite J
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    erw [Category.id_comp]
    /-
      case naturality
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X Y : Opposite J
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [Category.assoc]
    erw [← (F.obj (unop Y)).presheaf.map_comp, (F.map f.unop).c.naturality_assoc,
      ← (F.obj (unop Y)).presheaf.map_comp]
    /-
      case naturality
      J : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J TopCat
      inst✝² : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
      F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
      inst✝ : CategoryTheory.Limits.HasColimit F
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
      X Y : Opposite J
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f.unop).c.app ((TopologicalSp …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem colimitPresheafObjIsoComponentwiseLimit_inv_ι_app (F : J ⥤ PresheafedSpace.{_, _, v} C)
    (U : Opens (Limits.colimit F).carrier) (j : J) :
    (colimitPresheafObjIsoComponentwiseLimit F U).inv ≫ (colimit.ι F j).c.app (op U) =
      limit.π _ (op j) := by
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.co …
  -/
  delta colimitPresheafObjIsoComponentwiseLimit
  rw [Iso.trans_inv, Iso.trans_inv, Iso.app_inv, sheafIsoOfIso_inv, pushforwardToOfIso_app,
    congr_app (Iso.symm_inv _)]
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  rw [map_id, comp_id, assoc, assoc, assoc, NatTrans.naturality,
      ← comp_c_app_assoc,
      congr_app (colimit.isoColimitCocone_ι_hom _ _), assoc]
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap (Catego …
  -/
  erw [limitObjIsoLimitCompEvaluation_inv_π_app_assoc, limMap_π_assoc]
  -- Porting note: `convert` doesn't work due to meta variable, so change to a `suffices` block
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Algeb …
  -/
  set f := _
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    f : ?m.274238 := ?m.274239
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Algeb …
  -/
  change _ ≫ f = _
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    f : Quiver.Hom ((AlgebraicGeometry.PresheafedSpace.componentwiseDiagram F U).o …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Algeb …
  -/
  suffices f_eq : f = 𝟙 _ by rw [f_eq, comp_id]
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    f : Quiver.Hom ((AlgebraicGeometry.PresheafedSpace.componentwiseDiagram F U).o …
    ⊢ Eq f (CategoryTheory.CategoryStruct.id ((AlgebraicGeometry.PresheafedSpace.c …
  -/
  erw [← (F.obj j).presheaf.map_id]
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    f : Quiver.Hom ((AlgebraicGeometry.PresheafedSpace.componentwiseDiagram F U).o …
    ⊢ Eq f ((F.obj j).presheaf.map (CategoryTheory.CategoryStruct.id { unop := (To …
  -/
  change (F.obj j).presheaf.map _ ≫ _ = _
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    f : Quiver.Hom ((AlgebraicGeometry.PresheafedSpace.componentwiseDiagram F U).o …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).presheaf.map (CategoryTheo …
  -/
  erw [← (F.obj j).presheaf.map_comp, ← (F.obj j).presheaf.map_comp]
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    f : Quiver.Hom ((AlgebraicGeometry.PresheafedSpace.componentwiseDiagram F U).o …
    ⊢ Eq ((F.obj j).presheaf.map (CategoryTheory.CategoryStruct.comp (CategoryTheo …
  -/
  congr 1
  /-
    🎉 no goals
  -/


@[simp]
theorem colimitPresheafObjIsoComponentwiseLimit_hom_π (F : J ⥤ PresheafedSpace.{_, _, v} C)
    (U : Opens (Limits.colimit F).carrier) (j : J) :
    (colimitPresheafObjIsoComponentwiseLimit F U).hom ≫ limit.π _ (op j) =
      (colimit.ι F j).c.app (op U) := by
  /-
    J : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J TopCat
    inst✝¹ : ∀ (X : TopCat), CategoryTheory.Limits.HasLimitsOfShape (Opposite J) ( …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite J) C
    F : CategoryTheory.Functor J (AlgebraicGeometry.PresheafedSpace C)
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.colimit F)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.co …
  -/
  rw [← Iso.eq_inv_comp, colimitPresheafObjIsoComponentwiseLimit_inv_ι_app]
  /-
    🎉 no goals
  -/


