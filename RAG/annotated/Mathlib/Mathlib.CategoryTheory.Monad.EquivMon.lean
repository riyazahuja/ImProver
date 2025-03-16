/-- To every `Monad C` we associated a monoid object in `C ⥤ C`. -/
@[simps]
def toMon (M : Monad C) : Mon_ (C ⥤ C) where
  X := (M : C ⥤ C)
  one := M.η
  mul := M.μ
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    M : CategoryTheory.Monad C
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                  -/
  mul_assoc := by ext; simp [M.assoc]
                       /-
                         🎉 no goals
                       -/


/-- Passing from `Monad C` to `Mon_ (C ⥤ C)` is functorial. -/
@[simps]
def monadToMon : Monad C ⥤ Mon_ (C ⥤ C) where
  obj := toMon
  map f := { hom := f.toNatTrans }


/-- To every monoid object in `C ⥤ C` we associate a `Monad C`. -/
@[simps η μ]
def ofMon (M : Mon_ (C ⥤ C)) : Monad C where
  toFunctor := M.X
  η := M.one
  μ := M.mul
  left_unit := fun X => by
    -- Porting note: now using `erw`
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      M : Mon_ (CategoryTheory.Functor C C)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.one.app (M.X.obj X)) (M.mul.app X) …
    -/
    erw [← whiskerLeft_app, ← NatTrans.comp_app, M.mul_one]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      M : Mon_ (CategoryTheory.Functor C C)
      X : C
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.rightUnitor M.X).hom.app X) (Cate …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_unit := fun X => by
    -- Porting note: now using `erw`
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      M : Mon_ (CategoryTheory.Functor C C)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.X.map (M.mul.app X)) (M.mul.app X) …
    -/
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      M : Mon_ (CategoryTheory.Functor C C)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.X.map (M.one.app X)) (M.mul.app X) …
    -/
    erw [← whiskerRight_app, ← NatTrans.comp_app, M.one_mul]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      M : Mon_ (CategoryTheory.Functor C C)
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight M.mul M …
    -/
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      M : Mon_ (CategoryTheory.Functor C C)
      X : C
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.leftUnitor M.X).hom.app X) (Categ …
    -/
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      M : Mon_ (CategoryTheory.Functor C C)
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  assoc := fun X => by
    rw [← whiskerLeft_app, ← whiskerRight_app, ← NatTrans.comp_app]
    -- Porting note: had to add this step:
    erw [M.mul_assoc]
    simp

-- Porting note: `@[simps]` fails to generate `ofMon_obj`:

@[simp] lemma ofMon_obj (M : Mon_ (C ⥤ C)) (X : C) : (ofMon M).obj X = M.X.obj X := rfl


/-- Passing from `Mon_ (C ⥤ C)` to `Monad C` is functorial. -/
@[simps]
def monToMonad : Mon_ (C ⥤ C) ⥤ Monad C where
  obj := ofMon
  map {X Y} f :=
    { f.hom with
      app_η := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : Mon_ (CategoryTheory.Functor C C)
          f : Quiver.Hom X Y
          ⊢ ∀ (X_1 : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monad.o …
        -/
        intro X
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y : Mon_ (CategoryTheory.Functor C C)
          f : Quiver.Hom X✝ Y
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monad.ofMon X✝).η.ap …
        -/
        erw [← NatTrans.comp_app, f.one_hom]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y : Mon_ (CategoryTheory.Functor C C)
          f : Quiver.Hom X✝ Y
          X : C
          ⊢ Eq (Y.one.app X) ((CategoryTheory.Monad.ofMon Y).η.app X)
        -/
        simp only [Functor.id_obj, ofMon_obj, ofMon_η]
        /-
          🎉 no goals
        -/
      app_μ := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : Mon_ (CategoryTheory.Functor C C)
          f : Quiver.Hom X Y
          ⊢ ∀ (X_1 : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monad.o …
        -/
        intro Z
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : Mon_ (CategoryTheory.Functor C C)
          f : Quiver.Hom X Y
          Z : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monad.ofMon X).μ.app …
        -/
        erw [← NatTrans.comp_app, f.mul_hom]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : Mon_ (CategoryTheory.Functor C C)
          f : Quiver.Hom X Y
          Z : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
        -/
        dsimp
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : Mon_ (CategoryTheory.Functor C C)
          f : Quiver.Hom X Y
          Z : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [Category.assoc, NatTrans.naturality, ofMon_obj, ofMon] }
        /-
          🎉 no goals
        -/


/-- Oh, monads are just monoids in the category of endofunctors (equivalence of categories). -/
@[simps]
def monadMonEquiv : Monad C ≌ Mon_ (C ⥤ C) where
  functor := monadToMon _
  inverse := monToMonad _
  unitIso :=
  { hom := { app := fun _ => { app := fun _ => 𝟙 _ } }
    inv := { app := fun _ => { app := fun _ => 𝟙 _ } } }
  counitIso :=
  { hom := { app := fun _ => { hom := 𝟙 _ } }
    inv := { app := fun _ => { hom := 𝟙 _ } } }

-- Sanity check

