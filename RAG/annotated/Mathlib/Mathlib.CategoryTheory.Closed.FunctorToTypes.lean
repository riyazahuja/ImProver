/-- When `F G H : C ⥤ Type max w v u`, we have `(G ⟶ F.functorHom H) ≃ (F ⊗ G ⟶ H)`. -/
@[simps!]
def functorHomEquiv (G H : C ⥤ Type max w v u) : (G ⟶ F.functorHom H) ≃ (F ⊗ G ⟶ H) :=
  (Functor.functorHomEquiv F H G).trans (homObjEquiv F H G)


/-- Given a morphism `f : G ⟶ H`, an object `c : C`, and an element of `(F.functorHom G).obj c`,
construct an element of `(F.functorHom H).obj c`. -/
@[simps]
def rightAdj_map {F G H : C ⥤ Type max w v u} (f : G ⟶ H) (c : C) (a : (F.functorHom G).obj c) :
    (F.functorHom H).obj c where
  app d b := a.app d b ≫ f.app d
  naturality g h := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ F G H : CategoryTheory.Functor C (Type (max w v u))
      f : Quiver.Hom G H
      c : C
      a : (F.functorHom G).obj c
      c✝ d✝ : C
      g : Quiver.Hom c✝ d✝
      h : (Opposite.unop (CategoryTheory.coyoneda.rightOp.obj c)).obj c✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map g) ((fun d b => CategoryTheory …
    -/
    have := a.naturality g h
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ F G H : CategoryTheory.Functor C (Type (max w v u))
      f : Quiver.Hom G H
      c : C
      a : (F.functorHom G).obj c
      c✝ d✝ : C
      g : Quiver.Hom c✝ d✝
      h : (Opposite.unop (CategoryTheory.coyoneda.rightOp.obj c)).obj c✝
      this : Eq (CategoryTheory.CategoryStruct.comp (F.map g) (a.app d✝ ((Opposite.u …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map g) ((fun d b => CategoryTheory …
    -/
    change (F.map g ≫ a.app _ (h ≫ g)) ≫ _ = _
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ F G H : CategoryTheory.Functor C (Type (max w v u))
      f : Quiver.Hom G H
      c : C
      a : (F.functorHom G).obj c
      c✝ d✝ : C
      g : Quiver.Hom c✝ d✝
      h : (Opposite.unop (CategoryTheory.coyoneda.rightOp.obj c)).obj c✝
      this : Eq (CategoryTheory.CategoryStruct.comp (F.map g) (a.app d✝ ((Opposite.u …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    aesop
    /-
      🎉 no goals
    -/


/-- A right adjoint of `tensorLeft F`. -/
@[simps!]
def rightAdj : (C ⥤ Type max w v u) ⥤ C ⥤ Type max w v u where
  obj G := F.functorHom G
  map f := { app := rightAdj_map f }


/-- The adjunction `tensorLeft F ⊣ rightAdj F`. -/
def adj : tensorLeft F ⊣ rightAdj F where
  unit := {
    app := fun G ↦ (functorHomEquiv F G _).2 (𝟙 _)
    naturality := fun G H f ↦ by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        F G H : CategoryTheory.Functor C (Type (max w v u))
        f : Quiver.Hom G H
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
      -/
      dsimp [rightAdj]
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        F G H : CategoryTheory.Functor C (Type (max w v u))
        f : Quiver.Hom G H
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.FunctorToTypes.fun …
      -/
      ext _
      /-
        case w.h.h.h.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        F G H : CategoryTheory.Functor C (Type (max w v u))
        f : Quiver.Hom G H
        x✝ : C
        a✝¹ : G.obj x✝
        Y✝ : C
        f✝ : Quiver.Hom x✝ Y✝
        a✝ : F.obj Y✝
        ⊢ Eq (((CategoryTheory.CategoryStruct.comp f ((CategoryTheory.FunctorToTypes.f …
      -/
      simp [FunctorToTypes.naturality] }
      /-
        🎉 no goals
      -/
  counit := { app := fun G ↦ functorHomEquiv F _ G (𝟙 _) }


instance closed : Closed F where
  adj := adj F


instance monoidalClosed : MonoidalClosed (C ⥤ Type max w v u) where


