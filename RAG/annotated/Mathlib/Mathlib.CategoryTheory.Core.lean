/-- The core of a category C is the groupoid whose morphisms are all the
isomorphisms of C. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not yet ported
-- @[nolint has_nonempty_instance]

def Core (C : Type u₁) := C


instance coreCategory : Groupoid.{v₁} (Core C) where
  Hom (X Y : C) := X ≅ Y
  id (X : C) := Iso.refl X
  comp f g := Iso.trans f g
  inv {_ _} f := Iso.symm f


@[simp]
/- Porting note: abomination -/
theorem id_hom (X : C) : Iso.hom (coreCategory.id X) = @CategoryStruct.id C _ X := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.id X).hom (CategoryTheory.CategoryStruct.i …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_hom {X Y Z : Core C} (f : X ⟶ Y) (g : Y ⟶ Z) : (f ≫ g).hom = f.hom ≫ g.hom :=
  rfl


/-- The core of a category is naturally included in the category. -/
def inclusion : Core C ⥤ C where
  obj := id
  map f := f.hom

-- Porting note: This worked without proof before.

instance : (inclusion C).Faithful where
  map_injective := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      ⊢ ∀ {X Y : CategoryTheory.Core C}, Function.Injective (CategoryTheory.Core.inc …
    -/
    intro _ _
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y✝ : CategoryTheory.Core C
      ⊢ Function.Injective (CategoryTheory.Core.inclusion C).map
    -/
    apply Iso.ext
    /-
      🎉 no goals
    -/


/-- A functor from a groupoid to a category C factors through the core of C. -/
def functorToCore (F : G ⥤ C) : G ⥤ Core C where
  obj X := F.obj X
  map f := { hom := F.map f, inv := F.map (Groupoid.inv f) }


/-- We can functorially associate to any functor from a groupoid to the core of a category `C`,
a functor from the groupoid to `C`, simply by composing with the embedding `Core C ⥤ C`.
-/
def forgetFunctorToCore : (G ⥤ Core C) ⥤ G ⥤ C :=
  (whiskeringRight _ _ _).obj (inclusion C)


/-- `ofEquivFunctor m` lifts a type-level `EquivFunctor`
to a categorical functor `Core (Type u₁) ⥤ Core (Type u₂)`.
-/
def ofEquivFunctor (m : Type u₁ → Type u₂) [EquivFunctor m] : Core (Type u₁) ⥤ Core (Type u₂) where
  obj := m
  map f := (EquivFunctor.mapEquiv m f.toEquiv).toIso
                 /-
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   m : Type u₁ → Type u₂
                   inst✝ : EquivFunctor m
                   α : CategoryTheory.Core (Type u₁)
                   ⊢ Eq ({ obj := m, map := fun {X Y} f => (EquivFunctor.mapEquiv m (CategoryTheo …
                 -/
  map_id α := by apply Iso.ext; funext x; exact congr_fun (EquivFunctor.map_refl' _) x
                                          /-
                                            🎉 no goals
                                          -/
  map_comp f g := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      m : Type u₁ → Type u₂
      inst✝ : EquivFunctor m
      X✝ Y✝ Z✝ : CategoryTheory.Core (Type u₁)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := m, map := fun {X Y} f => (EquivFunctor.mapEquiv m (CategoryTheo …
    -/
    apply Iso.ext; funext x; dsimp
    /-
      case w.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      m : Type u₁ → Type u₂
      inst✝ : EquivFunctor m
      X✝ Y✝ Z✝ : CategoryTheory.Core (Type u₁)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      x : { obj := m, map := fun {X Y} f => (EquivFunctor.mapEquiv m (CategoryTheory …
      ⊢ Eq (EquivFunctor.map (CategoryTheory.Iso.toEquiv (CategoryTheory.CategoryStr …
    -/
    erw [Iso.toEquiv_comp, EquivFunctor.map_trans']
    /-
      case w.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      m : Type u₁ → Type u₂
      inst✝ : EquivFunctor m
      X✝ Y✝ Z✝ : CategoryTheory.Core (Type u₁)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      x : { obj := m, map := fun {X Y} f => (EquivFunctor.mapEquiv m (CategoryTheory …
      ⊢ Eq (Function.comp (EquivFunctor.map (CategoryTheory.Iso.toEquiv g)) (EquivFu …
    -/
    rw [Function.comp]
    /-
      🎉 no goals
    -/


