/-- Auxiliary definition for `CategoryTheory.Functor.closed`.
The internal hom functor `F ⟶[C] -` -/
@[simps!]
def closedIhom (F : D ⥤ C) : (D ⥤ C) ⥤ D ⥤ C :=
  ((whiskeringRight₂ D Cᵒᵖ C C).obj internalHom).obj (Groupoid.invFunctor D ⋙ F.op)


/-- Auxiliary definition for `CategoryTheory.Functor.closed`.
The unit for the adjunction `(tensorLeft F) ⊣ (ihom F)`. -/
@[simps]
def closedUnit (F : D ⥤ C) : 𝟭 (D ⥤ C) ⟶ tensorLeft F ⋙ closedIhom F where
  app G :=
  { app := fun X => (ihom.coev (F.obj X)).app (G.obj X)
    naturality := by
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.4081, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        ⊢ ∀ ⦃X Y : D⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
      -/
      intro X Y f
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.4081, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.id (Categor …
      -/
      dsimp
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.4081, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((CategoryTheory.ihom.coev  …
      -/
      simp only [ihom.coev_naturality, closedIhom_obj_map, Monoidal.tensorObj_map]
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.4081, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ihom.coev (F.obj Y)) …
      -/
      dsimp
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.4081, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ihom.coev (F.obj Y)) …
      -/
      rw [coev_app_comp_pre_app_assoc, ← Functor.map_comp, tensorHom_def]
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.4081, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ihom.coev (F.obj Y)) …
      -/
      simp }
      /-
        🎉 no goals
      -/


/-- Auxiliary definition for `CategoryTheory.Functor.closed`.
The counit for the adjunction `(tensorLeft F) ⊣ (ihom F)`. -/
@[simps]
def closedCounit (F : D ⥤ C) : closedIhom F ⋙ tensorLeft F ⟶ 𝟭 (D ⥤ C) where
  app G :=
  { app := fun X => (ihom.ev (F.obj X)).app (G.obj X)
    naturality := by
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.11014, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        ⊢ ∀ ⦃X Y : D⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
      -/
      intro X Y f
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.11014, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.closedIhom.comp (CategoryTheory. …
      -/
      dsimp
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.11014, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp only [closedIhom_obj_map, pre_comm_ihom_map]
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.11014, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      rw [tensorHom_def]
      /-
        D : Type u_1
        C : Type u_2
        inst✝³ : CategoryTheory.Groupoid D
        inst✝² : CategoryTheory.Category.{?u.11014, u_2} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        inst✝ : CategoryTheory.MonoidalClosed C
        F G : CategoryTheory.Functor D C
        X Y : D
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp }
      /-
        🎉 no goals
      -/


/-- If `C` is a monoidal closed category and `D` is a groupoid, then every functor `F : D ⥤ C` is
closed in the functor category `F : D ⥤ C` with the pointwise monoidal structure. -/
-- Porting note: removed `@[simps]`, as some of the generated lemmas were failing the simpNF linter,
-- and none of the generated lemmas was actually used in mathlib3.
instance closed (F : D ⥤ C) : Closed F where
  rightAdj := closedIhom F
  adj :=
    { unit := closedUnit F
      counit := closedCounit F }


/-- If `C` is a monoidal closed category and `D` is groupoid, then the functor category `D ⥤ C`,
with the pointwise monoidal structure, is monoidal closed. -/
@[simps! closed_adj]
instance monoidalClosed : MonoidalClosed (D ⥤ C) where


theorem ihom_map (F : D ⥤ C) {G H : D ⥤ C} (f : G ⟶ H) : (ihom F).map f = (closedIhom F).map f :=
  rfl


theorem ihom_ev_app (F G : D ⥤ C) : (ihom.ev F).app G = (closedCounit F).app G :=
  rfl


theorem ihom_coev_app (F G : D ⥤ C) : (ihom.coev F).app G = (closedUnit F).app G :=
  rfl


