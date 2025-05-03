/-- A monoid object in a functor category induces a functor to the category of monoid objects. -/
@[simps]
def functorObj (A : Mon_ (C ⥤ D)) : C ⥤ Mon_ D where
  obj X :=
  { X := A.X.obj X
    one := A.one.app X
    mul := A.mul.app X
    one_mul := congr_app A.one_mul X
    mul_one := congr_app A.mul_one X
    mul_assoc := congr_app A.mul_assoc X }
  map f :=
  { hom := A.X.map f
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                    inst✝ : CategoryTheory.MonoidalCategory D
                    A : Mon_ (CategoryTheory.Functor C D)
                    X✝ Y✝ : C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => { X := A.X.obj X, one := A …
                  -/
    one_hom := by rw [← A.one.naturality, tensorUnit_map]; dsimp; rw [Category.id_comp]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                    inst✝ : CategoryTheory.MonoidalCategory D
                    A : Mon_ (CategoryTheory.Functor C D)
                    X✝ Y✝ : C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => { X := A.X.obj X, one := A …
                  -/
    mul_hom := by dsimp; rw [← A.mul.naturality, tensorObj_map] }
                         /-
                           🎉 no goals
                         -/
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   inst✝ : CategoryTheory.MonoidalCategory D
                   A : Mon_ (CategoryTheory.Functor C D)
                   X : C
                   ⊢ Eq ({ obj := fun X => { X := A.X.obj X, one := A.one.app X, mul := A.mul.app …
                 -/
  map_id X := by ext; dsimp; rw [CategoryTheory.Functor.map_id]
                             /-
                               🎉 no goals
                             -/
                     /-
                       C : Type u₁
                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                       inst✝ : CategoryTheory.MonoidalCategory D
                       A : Mon_ (CategoryTheory.Functor C D)
                       X✝ Y✝ Z✝ : C
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun X => { X := A.X.obj X, one := A.one.app X, mul := A.mul.app …
                     -/
  map_comp f g := by ext; dsimp; rw [Functor.map_comp]
                                 /-
                                   🎉 no goals
                                 -/


/-- Functor translating a monoid object in a functor category
to a functor into the category of monoid objects.
-/
@[simps]
def functor : Mon_ (C ⥤ D) ⥤ C ⥤ Mon_ D where
  obj := functorObj
  map f :=
  { app := fun X =>
    { hom := f.hom.app X
      one_hom := congr_app f.one_hom X
      mul_hom := congr_app f.mul_hom X } }


/-- A functor to the category of monoid objects can be translated as a monoid object
in the functor category. -/
@[simps]
def inverseObj (F : C ⥤ Mon_ D) : Mon_ (C ⥤ D) where
  X := F ⋙ Mon_.forget D
  one := { app := fun X => (F.obj X).one }
  mul := { app := fun X => (F.obj X).mul }


/-- Functor translating a functor into the category of monoid objects
to a monoid object in the functor category
-/
@[simps]
def inverse : (C ⥤ Mon_ D) ⥤ Mon_ (C ⥤ D) where
  obj := inverseObj
  map α :=
  { hom :=
    { app := fun X => (α.app X).hom
      naturality := fun _ _ f => congr_arg Mon_.Hom.hom (α.naturality f) } }


/-- The unit for the equivalence `Mon_ (C ⥤ D) ≌ C ⥤ Mon_ D`.
-/
@[simps!]
def unitIso : 𝟭 (Mon_ (C ⥤ D)) ≅ functor ⋙ inverse :=
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.MonoidalCategory D
    ⊢ ∀ {X Y : Mon_ (CategoryTheory.Functor C D)} (f : Quiver.Hom X Y), Eq (Catego …
  -/
  NatIso.ofComponents (fun A =>
  /-
    🎉 no goals
  -/
  { hom := { hom := { app := fun _ => 𝟙 _ } }
    inv := { hom := { app := fun _ => 𝟙 _ } } })


/-- The counit for the equivalence `Mon_ (C ⥤ D) ≌ C ⥤ Mon_ D`.
-/
@[simps!]
def counitIso : inverse ⋙ functor ≅ 𝟭 (C ⥤ Mon_ D) :=
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.MonoidalCategory D
    ⊢ ∀ {X Y : CategoryTheory.Functor C (Mon_ D)} (f : Quiver.Hom X Y), Eq (Catego …
  -/
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.MonoidalCategory D
      A : CategoryTheory.Functor C (Mon_ D)
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
    -/
  NatIso.ofComponents (fun A =>
    /-
      🎉 no goals
    -/
  /-
    🎉 no goals
  -/
    NatIso.ofComponents (fun X => { hom := { hom := 𝟙 _ }, inv := { hom := 𝟙 _ } }))


/-- When `D` is a monoidal category,
monoid objects in `C ⥤ D` are the same thing
as functors from `C` into the monoid objects of `D`.
-/
@[simps]
def monFunctorCategoryEquivalence : Mon_ (C ⥤ D) ≌ C ⥤ Mon_ D where
  functor := functor
  inverse := inverse
  unitIso := unitIso
  counitIso := counitIso


/--
A comonoid object in a functor category induces a functor to the category of comonoid objects.
-/
@[simps]
def functorObj (A : Comon_ (C ⥤ D)) : C ⥤ Comon_ D where
  obj X :=
  { X := A.X.obj X
    counit := A.counit.app X
    comul := A.comul.app X
    counit_comul := congr_app A.counit_comul X
    comul_counit := congr_app A.comul_counit X
    comul_assoc := congr_app A.comul_assoc X }
  map f :=
  { hom := A.X.map f
                     /-
                       C : Type u₁
                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                       inst✝ : CategoryTheory.MonoidalCategory D
                       A : Comon_ (CategoryTheory.Functor C D)
                       X✝ Y✝ : C
                       f : Quiver.Hom X✝ Y✝
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (A.X.map f) ((fun X => { X := A.X.obj …
                     -/
    hom_counit := by dsimp; rw [A.counit.naturality, tensorUnit_map]; dsimp; rw [Category.comp_id]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                    /-
                      C : Type u₁
                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                      inst✝ : CategoryTheory.MonoidalCategory D
                      A : Comon_ (CategoryTheory.Functor C D)
                      X✝ Y✝ : C
                      f : Quiver.Hom X✝ Y✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (A.X.map f) ((fun X => { X := A.X.obj …
                    -/
    hom_comul := by dsimp; rw [A.comul.naturality, tensorObj_map] }
                           /-
                             🎉 no goals
                           -/
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   inst✝ : CategoryTheory.MonoidalCategory D
                   A : Comon_ (CategoryTheory.Functor C D)
                   X : C
                   ⊢ Eq ({ obj := fun X => { X := A.X.obj X, counit := A.counit.app X, comul := A …
                 -/
  map_id X := by ext; dsimp; rw [CategoryTheory.Functor.map_id]
                             /-
                               🎉 no goals
                             -/
                     /-
                       C : Type u₁
                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                       inst✝ : CategoryTheory.MonoidalCategory D
                       A : Comon_ (CategoryTheory.Functor C D)
                       X✝ Y✝ Z✝ : C
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun X => { X := A.X.obj X, counit := A.counit.app X, comul := A …
                     -/
  map_comp f g := by ext; dsimp; rw [Functor.map_comp]
                                 /-
                                   🎉 no goals
                                 -/


/-- Functor translating a comonoid object in a functor category
to a functor into the category of comonoid objects.
-/
@[simps]
def functor : Comon_ (C ⥤ D) ⥤ C ⥤ Comon_ D where
  obj := functorObj
  map f :=
  { app := fun X =>
    { hom := f.hom.app X
      hom_counit := congr_app f.hom_counit X
      hom_comul := congr_app f.hom_comul X } }


/-- A functor to the category of comonoid objects can be translated as a comonoid object
in the functor category. -/
@[simps]
def inverseObj (F : C ⥤ Comon_ D) : Comon_ (C ⥤ D) where
  X := F ⋙ Comon_.forget D
  counit := { app := fun X => (F.obj X).counit }
  comul := { app := fun X => (F.obj X).comul }


/-- Functor translating a functor into the category of comonoid objects
to a comonoid object in the functor category
-/
@[simps]
private def inverse : (C ⥤ Comon_ D) ⥤ Comon_ (C ⥤ D) where
  obj := inverseObj
  map α :=
  { hom :=
    { app := fun X => (α.app X).hom
      naturality := fun _ _ f => congr_arg Comon_.Hom.hom (α.naturality f) }
                     /-
                       C : Type u₁
                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                       inst✝ : CategoryTheory.MonoidalCategory D
                       X✝ Y✝ : CategoryTheory.Functor C (Comon_ D)
                       α : Quiver.Hom X✝ Y✝
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun X => (α.app X).hom, natu …
                     -/
    hom_counit := by ext x; dsimp; rw [(α.app x).hom_counit]
                                   /-
                                     🎉 no goals
                                   -/
                    /-
                      C : Type u₁
                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                      inst✝ : CategoryTheory.MonoidalCategory D
                      X✝ Y✝ : CategoryTheory.Functor C (Comon_ D)
                      α : Quiver.Hom X✝ Y✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun X => (α.app X).hom, natu …
                    -/
    hom_comul := by ext x; dsimp; rw [(α.app x).hom_comul] }
                                  /-
                                    🎉 no goals
                                  -/


/-- The unit for the equivalence `Comon_ (C ⥤ D) ≌ C ⥤ Comon_ D`.
-/
@[simps!]
private def unitIso : 𝟭 (Comon_ (C ⥤ D)) ≅ functor ⋙ inverse :=
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.MonoidalCategory D
    ⊢ ∀ {X Y : Comon_ (CategoryTheory.Functor C D)} (f : Quiver.Hom X Y), Eq (Cate …
  -/
  NatIso.ofComponents (fun A =>
  /-
    🎉 no goals
  -/
  { hom := { hom := { app := fun _ => 𝟙 _ } }
    inv := { hom := { app := fun _ => 𝟙 _ } } })


/-- The counit for the equivalence `Mon_ (C ⥤ D) ≌ C ⥤ Mon_ D`.
-/
@[simps!]
def counitIso : inverse ⋙ functor ≅ 𝟭 (C ⥤ Comon_ D) :=
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.MonoidalCategory D
    ⊢ ∀ {X Y : CategoryTheory.Functor C (Comon_ D)} (f : Quiver.Hom X Y), Eq (Cate …
  -/
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.MonoidalCategory D
      A : CategoryTheory.Functor C (Comon_ D)
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
    -/
  NatIso.ofComponents (fun A =>
    /-
      🎉 no goals
    -/
  /-
    🎉 no goals
  -/
    NatIso.ofComponents (fun X => { hom := { hom := 𝟙 _ }, inv := { hom := 𝟙 _ } }) )


/-- When `D` is a monoidal category,
comonoid objects in `C ⥤ D` are the same thing
as functors from `C` into the comonoid objects of `D`.
-/
@[simps]
def comonFunctorCategoryEquivalence : Comon_ (C ⥤ D) ≌ C ⥤ Comon_ D where
  functor := functor
  inverse := inverse
  unitIso := unitIso
  counitIso := counitIso


/-- Functor translating a commutative monoid object in a functor category
to a functor into the category of commutative monoid objects.
-/
@[simps!]
def functor : CommMon_ (C ⥤ D) ⥤ C ⥤ CommMon_ D where
  obj A :=
    { (monFunctorCategoryEquivalence C D).functor.obj A.toMon_ with
      obj := fun X =>
        { ((monFunctorCategoryEquivalence C D).functor.obj A.toMon_).obj X with
          mul_comm := congr_app A.mul_comm X } }
  map f := { app := fun X => ((monFunctorCategoryEquivalence C D).functor.map f).app X }


/-- Functor translating a functor into the category of commutative monoid objects
to a commutative monoid object in the functor category
-/
@[simps!]
def inverse : (C ⥤ CommMon_ D) ⥤ CommMon_ (C ⥤ D) where
  obj F :=
    { (monFunctorCategoryEquivalence C D).inverse.obj (F ⋙ CommMon_.forget₂Mon_ D) with
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       inst✝¹ : CategoryTheory.MonoidalCategory D
                       inst✝ : CategoryTheory.BraidedCategory D
                       F : CategoryTheory.Functor C (CommMon_ D)
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
                     -/
      mul_comm := by ext X; exact (F.obj X).mul_comm }
                            /-
                              🎉 no goals
                            -/
  map α := (monFunctorCategoryEquivalence C D).inverse.map (whiskerRight α _)


/-- The unit for the equivalence `CommMon_ (C ⥤ D) ≌ C ⥤ CommMon_ D`.
-/
@[simps!]
def unitIso : 𝟭 (CommMon_ (C ⥤ D)) ≅ functor ⋙ inverse :=
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    inst✝ : CategoryTheory.BraidedCategory D
    ⊢ ∀ {X Y : CommMon_ (CategoryTheory.Functor C D)} (f : Quiver.Hom X Y), Eq (Ca …
  -/
  NatIso.ofComponents (fun A =>
  /-
    🎉 no goals
  -/
  { hom := { hom := { app := fun _ => 𝟙 _ }  }
    inv := { hom := { app := fun _ => 𝟙 _ }  } })


/-- The counit for the equivalence `CommMon_ (C ⥤ D) ≌ C ⥤ CommMon_ D`.
-/
@[simps!]
def counitIso : inverse ⋙ functor ≅ 𝟭 (C ⥤ CommMon_ D) :=
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    inst✝ : CategoryTheory.BraidedCategory D
    ⊢ ∀ {X Y : CategoryTheory.Functor C (CommMon_ D)} (f : Quiver.Hom X Y), Eq (Ca …
  -/
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.MonoidalCategory D
      inst✝ : CategoryTheory.BraidedCategory D
      A : CategoryTheory.Functor C (CommMon_ D)
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
    -/
  NatIso.ofComponents (fun A =>
    /-
      🎉 no goals
    -/
  /-
    🎉 no goals
  -/
    NatIso.ofComponents (fun X => { hom := { hom := 𝟙 _ }, inv := { hom := 𝟙 _ } }) )


/-- When `D` is a braided monoidal category,
commutative monoid objects in `C ⥤ D` are the same thing
as functors from `C` into the commutative monoid objects of `D`.
-/
@[simps]
def commMonFunctorCategoryEquivalence : CommMon_ (C ⥤ D) ≌ C ⥤ CommMon_ D where
  functor := functor
  inverse := inverse
  unitIso := unitIso
  counitIso := counitIso


