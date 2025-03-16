/-- The Yoneda embedding for preadditive categories sends an object `Y` to the presheaf sending an
object `X` to the `End Y`-module of morphisms `X ⟶ Y`.
-/
@[simps]
def preadditiveYonedaObj (Y : C) : Cᵒᵖ ⥤ ModuleCat.{v} (End Y) where
  obj X := ModuleCat.of _ (X.unop ⟶ Y)
  map f := ModuleCat.ofHom
    { toFun := fun g => f.unop ≫ g
      map_add' := fun _ _ => comp_add _ _ _ _ _ _
      map_smul' := fun _ _ => Eq.symm <| Category.assoc _ _ _ }


/-- The Yoneda embedding for preadditive categories sends an object `Y` to the presheaf sending an
object `X` to the group of morphisms `X ⟶ Y`. At each point, we get an additional `End Y`-module
structure, see `preadditiveYonedaObj`.
-/
@[simps]
def preadditiveYoneda : C ⥤ Cᵒᵖ ⥤ AddCommGrp.{v} where
  obj Y := preadditiveYonedaObj Y ⋙ forget₂ _ _
  map f :=
    { app := fun _ =>
        { toFun := fun g => g ≫ f
          map_zero' := Limits.zero_comp
          map_add' := fun _ _ => add_comp _ _ _ _ _ _ }
      naturality := fun _ _ _ => AddCommGrp.ext fun _ => Category.assoc _ _ _ }
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   inst✝ : CategoryTheory.Preadditive C
                   x✝ : C
                   ⊢ Eq ({ obj := fun Y => (CategoryTheory.preadditiveYonedaObj Y).comp (Category …
                 -/
  map_id _ := by ext; dsimp; simp
                             /-
                               🎉 no goals
                             -/
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       inst✝ : CategoryTheory.Preadditive C
                       X✝ Y✝ Z✝ : C
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun Y => (CategoryTheory.preadditiveYonedaObj Y).comp (Category …
                     -/
  map_comp f g := by ext; dsimp; simp
                                 /-
                                   🎉 no goals
                                 -/


/-- The Yoneda embedding for preadditive categories sends an object `X` to the copresheaf sending an
object `Y` to the `End X`-module of morphisms `X ⟶ Y`.
-/
@[simps]
def preadditiveCoyonedaObj (X : Cᵒᵖ) : C ⥤ ModuleCat.{v} (End X) where
  obj Y := ModuleCat.of _ (unop X ⟶ Y)
  map f := ModuleCat.ofHom
    { toFun := fun g => g ≫ f
      map_add' := fun _ _ => add_comp _ _ _ _ _ _
      map_smul' := fun _ _ => Category.assoc _ _ _ }


/-- The Yoneda embedding for preadditive categories sends an object `X` to the copresheaf sending an
object `Y` to the group of morphisms `X ⟶ Y`. At each point, we get an additional `End X`-module
structure, see `preadditiveCoyonedaObj`.
-/
@[simps]
def preadditiveCoyoneda : Cᵒᵖ ⥤ C ⥤ AddCommGrp.{v} where
  obj X := preadditiveCoyonedaObj X ⋙ forget₂ _ _
  map f :=
    { app := fun _ =>
        { toFun := fun g => f.unop ≫ g
          map_zero' := Limits.comp_zero
          map_add' := fun _ _ => comp_add _ _ _ _ _ _ }
      naturality := fun _ _ _ =>
        AddCommGrp.ext fun _ => Eq.symm <| Category.assoc _ _ _ }
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   inst✝ : CategoryTheory.Preadditive C
                   x✝ : Opposite C
                   ⊢ Eq ({ obj := fun X => (CategoryTheory.preadditiveCoyonedaObj X).comp (Catego …
                 -/
  map_id _ := by ext; dsimp; simp
                             /-
                               🎉 no goals
                             -/
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       inst✝ : CategoryTheory.Preadditive C
                       X✝ Y✝ Z✝ : Opposite C
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun X => (CategoryTheory.preadditiveCoyonedaObj X).comp (Catego …
                     -/
  map_comp f g := by ext; dsimp; simp
                                 /-
                                   🎉 no goals
                                 -/

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

instance additive_yonedaObj (X : C) : Functor.Additive (preadditiveYonedaObj X) where


instance additive_yonedaObj' (X : C) : Functor.Additive (preadditiveYoneda.obj X) where


instance additive_coyonedaObj (X : Cᵒᵖ) : Functor.Additive (preadditiveCoyonedaObj X) where


instance additive_coyonedaObj' (X : Cᵒᵖ) : Functor.Additive (preadditiveCoyoneda.obj X) where


/-- Composing the preadditive yoneda embedding with the forgetful functor yields the regular
Yoneda embedding.
-/
@[simp]
theorem whiskering_preadditiveYoneda :
    preadditiveYoneda ⋙
        (whiskeringRight Cᵒᵖ AddCommGrp (Type v)).obj (forget AddCommGrp) =
      yoneda :=
  rfl


/-- Composing the preadditive yoneda embedding with the forgetful functor yields the regular
Yoneda embedding.
-/
@[simp]
theorem whiskering_preadditiveCoyoneda :
    preadditiveCoyoneda ⋙
        (whiskeringRight C AddCommGrp (Type v)).obj (forget AddCommGrp) =
      coyoneda :=
  rfl


instance full_preadditiveYoneda : (preadditiveYoneda : C ⥤ Cᵒᵖ ⥤ AddCommGrp).Full :=
  let _ : Functor.Full (preadditiveYoneda ⋙
      (whiskeringRight Cᵒᵖ AddCommGrp (Type v)).obj (forget AddCommGrp)) :=
    Yoneda.yoneda_full
  Functor.Full.of_comp_faithful preadditiveYoneda
    ((whiskeringRight Cᵒᵖ AddCommGrp (Type v)).obj (forget AddCommGrp))


instance full_preadditiveCoyoneda : (preadditiveCoyoneda : Cᵒᵖ ⥤ C ⥤ AddCommGrp).Full :=
  let _ : Functor.Full (preadditiveCoyoneda ⋙
      (whiskeringRight C AddCommGrp (Type v)).obj (forget AddCommGrp)) :=
    Coyoneda.coyoneda_full
  Functor.Full.of_comp_faithful preadditiveCoyoneda
    ((whiskeringRight C AddCommGrp (Type v)).obj (forget AddCommGrp))


instance faithful_preadditiveYoneda : (preadditiveYoneda : C ⥤ Cᵒᵖ ⥤ AddCommGrp).Faithful :=
  Functor.Faithful.of_comp_eq whiskering_preadditiveYoneda


instance faithful_preadditiveCoyoneda :
    (preadditiveCoyoneda : Cᵒᵖ ⥤ C ⥤ AddCommGrp).Faithful :=
  Functor.Faithful.of_comp_eq whiskering_preadditiveCoyoneda


