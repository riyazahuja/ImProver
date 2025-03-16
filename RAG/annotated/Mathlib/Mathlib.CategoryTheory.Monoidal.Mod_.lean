/-- A module object for a monoid object, all internal to some monoidal category. -/
structure Mod_ (A : Mon_ C) where
  X : C
  act : A.X ⊗ X ⟶ X
  one_act : (A.one ▷ X) ≫ act = (λ_ X).hom := by aesop_cat
  assoc : (A.mul ▷ X) ≫ act = (α_ A.X A.X X).hom ≫ (A.X ◁ act) ≫ act := by aesop_cat


attribute [reassoc (attr := simp)] Mod_.one_act Mod_.assoc


theorem assoc_flip :
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 inst✝ : CategoryTheory.MonoidalCategory C
                                                                                 A : Mon_ C
                                                                                 M : Mod_ A
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                               -/
    (A.X ◁ M.act) ≫ M.act = (α_ A.X A.X M.X).inv ≫ (A.mul ▷ M.X) ≫ M.act := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- A morphism of module objects. -/
@[ext]
structure Hom (M N : Mod_ A) where
  hom : M.X ⟶ N.X
  act_hom : M.act ≫ hom = (A.X ◁ hom) ≫ N.act := by aesop_cat


attribute [reassoc (attr := simp)] Hom.act_hom


/-- The identity morphism on a module object. -/
@[simps]
def id (M : Mod_ A) : Hom M M where hom := 𝟙 M.X


instance homInhabited (M : Mod_ A) : Inhabited (Hom M M) :=
  ⟨id M⟩


/-- Composition of module object morphisms. -/
@[simps]
def comp {M N O : Mod_ A} (f : Hom M N) (g : Hom N O) : Hom M O where hom := f.hom ≫ g.hom


instance : Category (Mod_ A) where
  Hom M N := Hom M N
  id := id
  comp f g := comp f g


@[ext]
lemma hom_ext {M N : Mod_ A} (f₁ f₂ : M ⟶ N) (h : f₁.hom = f₂.hom) : f₁ = f₂ :=
  Hom.ext h


@[simp]
theorem id_hom' (M : Mod_ A) : (𝟙 M : M ⟶ M).hom = 𝟙 M.X := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    A : Mon_ C
    M : Mod_ A
    ⊢ Eq (CategoryTheory.CategoryStruct.id M).hom (CategoryTheory.CategoryStruct.i …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_hom' {M N K : Mod_ A} (f : M ⟶ N) (g : N ⟶ K) :
    (f ≫ g).hom = f.hom ≫ g.hom :=
  rfl


/-- A monoid object as a module over itself. -/
@[simps]
def regular : Mod_ A where
  X := A.X
  act := A.mul


instance : Inhabited (Mod_ A) :=
  ⟨regular A⟩


/-- The forgetful functor from module objects to the ambient category. -/
def forget : Mod_ A ⥤ C where
  obj A := A.X
  map f := f.hom


set_option maxHeartbeats 400000 in
/-- A morphism of monoid objects induces a "restriction" or "comap" functor
between the categories of module objects.
-/
@[simps]
def comap {A B : Mon_ C} (f : A ⟶ B) : Mod_ B ⥤ Mod_ A where
  obj M :=
    { X := M.X
      act := (f.hom ▷ M.X) ≫ M.act
      one_act := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 1 2 => rw [← comp_whiskerRight]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        rw [f.one_hom, one_act]
        /-
          🎉 no goals
        -/
      assoc := by
        -- oh, for homotopy.io in a widget!
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 2 3 => rw [whisker_exchange]
        simp only [whiskerRight_tensor, MonoidalCategory.whiskerLeft_comp, Category.assoc,
          Iso.hom_inv_id_assoc]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 4 5 => rw [Mod_.assoc_flip]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 3 4 => rw [associator_inv_naturality_middle]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 2 4 => rw [Iso.hom_inv_id_assoc]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 1 2 => rw [← MonoidalCategory.comp_whiskerRight, ← whisker_exchange]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 1 2 => rw [← MonoidalCategory.comp_whiskerRight, ← tensorHom_def', ← f.mul_hom]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M✝ : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          M : Mod_ B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        rw [comp_whiskerRight, Category.assoc] }
        /-
          🎉 no goals
        -/
  map g :=
    { hom := g.hom
      act_hom := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          X✝ Y✝ : Mod_ B
          g : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun M => { X := M.X, act := Categor …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          X✝ Y✝ : Mod_ B
          g : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        slice_rhs 1 2 => rw [whisker_exchange]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          X✝ Y✝ : Mod_ B
          g : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        slice_rhs 2 3 => rw [← g.act_hom]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          inst✝ : CategoryTheory.MonoidalCategory C
          A✝ : Mon_ C
          M : Mod_ A✝
          A B : Mon_ C
          f : Quiver.Hom A B
          X✝ Y✝ : Mod_ B
          g : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [Category.assoc] }
        /-
          🎉 no goals
        -/

-- Lots more could be said about `comap`, e.g. how it interacts with
-- identities, compositions, and equalities of monoid object morphisms.

