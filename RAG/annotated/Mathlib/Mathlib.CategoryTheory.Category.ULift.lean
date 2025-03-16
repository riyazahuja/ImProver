/-- The functorial version of `ULift.up`. -/
@[simps]
def ULift.upFunctor : C ⥤ ULift.{u₂} C where
  obj := ULift.up
  map f := f


/-- The functorial version of `ULift.down`. -/
@[simps]
def ULift.downFunctor : ULift.{u₂} C ⥤ C where
  obj := ULift.down
  map f := f


/-- The categorical equivalence between `C` and `ULift C`. -/
@[simps]
def ULift.equivalence : C ≌ ULift.{u₂} C where
  functor := ULift.upFunctor
  inverse := ULift.downFunctor
  unitIso :=
    { hom := 𝟙 _
      inv := 𝟙 _ }
  counitIso :=
    { hom :=
        { app := fun _ => 𝟙 _
          naturality := fun X Y f => by
            /-
              C : Type u₁
              inst✝ : CategoryTheory.Category.{v₁, u₁} C
              X Y : ULift.{u₂, u₁} C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ULift.downFunctor.co …
            -/
            change f ≫ 𝟙 _ = 𝟙 _ ≫ f
            /-
              C : Type u₁
              inst✝ : CategoryTheory.Category.{v₁, u₁} C
              X Y : ULift.{u₂, u₁} C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
            -/
            simp }
            /-
              🎉 no goals
            -/
      inv :=
        { app := fun _ => 𝟙 _
          naturality := fun X Y f => by
            /-
              C : Type u₁
              inst✝ : CategoryTheory.Category.{v₁, u₁} C
              X Y : ULift.{u₂, u₁} C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (ULift.{u …
            -/
            change f ≫ 𝟙 _ = 𝟙 _ ≫ f
            /-
              C : Type u₁
              inst✝ : CategoryTheory.Category.{v₁, u₁} C
              X Y : ULift.{u₂, u₁} C
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
            -/
            simp }
            /-
              🎉 no goals
            -/
      hom_inv_id := by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun x => CategoryTheory.Cate …
        -/
        ext
        /-
          case w.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          x✝ : ULift.{u₂, u₁} C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun x => CategoryTheory.Cat …
        -/
        change 𝟙 _ ≫ 𝟙 _ = 𝟙 _
        /-
          case w.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          x✝ : ULift.{u₂, u₁} C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
        -/
        simp
        /-
          🎉 no goals
        -/
      inv_hom_id := by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun x => CategoryTheory.Cate …
        -/
        ext
        /-
          case w.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          x✝ : ULift.{u₂, u₁} C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun x => CategoryTheory.Cat …
        -/
        change 𝟙 _ ≫ 𝟙 _ = 𝟙 _
        /-
          case w.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          x✝ : ULift.{u₂, u₁} C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
        -/
        simp }
        /-
          🎉 no goals
        -/
  functor_unitIso_comp X := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ULift.upFunctor.map ( …
    -/
    change 𝟙 X ≫ 𝟙 X = 𝟙 X
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- `ULiftHom.{w} C` is an alias for `C`, which is endowed with a category instance
  whose morphisms are obtained by applying `ULift.{w}` to the morphisms from `C`.
-/
def ULiftHom.{w,u} (C : Type u) : Type u :=
  let _ := ULift.{w} C
  C


instance {C} [Inhabited C] : Inhabited (ULiftHom C) :=
  ⟨(default : C)⟩


/-- The obvious function `ULiftHom C → C`. -/
def ULiftHom.objDown {C} (A : ULiftHom C) : C :=
  A


/-- The obvious function `C → ULiftHom C`. -/
def ULiftHom.objUp {C} (A : C) : ULiftHom C :=
  A


@[simp]
theorem objDown_objUp {C} (A : C) : (ULiftHom.objUp A).objDown = A :=
  rfl


@[simp]
theorem objUp_objDown {C} (A : ULiftHom C) : ULiftHom.objUp A.objDown = A :=
  rfl


instance ULiftHom.category : Category.{max v₂ v₁} (ULiftHom.{v₂} C) where
  Hom A B := ULift.{v₂} <| A.objDown ⟶ B.objDown
  id _ := ⟨𝟙 _⟩
  comp f g := ⟨f.down ≫ g.down⟩


/-- One half of the quivalence between `C` and `ULiftHom C`. -/
@[simps]
def ULiftHom.up : C ⥤ ULiftHom C where
  obj := ULiftHom.objUp
  map f := ⟨f⟩


/-- One half of the quivalence between `C` and `ULiftHom C`. -/
@[simps]
def ULiftHom.down : ULiftHom C ⥤ C where
  obj := ULiftHom.objDown
  map f := f.down


/-- The equivalence between `C` and `ULiftHom C`. -/
def ULiftHom.equiv : C ≌ ULiftHom C where
  functor := ULiftHom.up
  inverse := ULiftHom.down
             /-
               C : Type u₁
               inst✝ : CategoryTheory.Category.{v₁, u₁} C
               ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
             -/
  unitIso := NatIso.ofComponents fun _ => eqToIso rfl
             /-
               🎉 no goals
             -/
               /-
                 C : Type u₁
                 inst✝ : CategoryTheory.Category.{v₁, u₁} C
                 ⊢ ∀ {X Y : CategoryTheory.ULiftHom C} (f : Quiver.Hom X Y), Eq (CategoryTheory …
               -/
  counitIso := NatIso.ofComponents fun _ => eqToIso rfl
               /-
                 🎉 no goals
               -/


/-- `AsSmall C` is a small category equivalent to `C`.
  More specifically, if `C : Type u` is endowed with `Category.{v} C`, then
  `AsSmall.{w} C : Type (max w v u)` is endowed with an instance of a small category.

  The objects and morphisms of `AsSmall C` are defined by applying `ULift` to the
  objects and morphisms of `C`.

  Note: We require a category instance for this definition in order to have direct
  access to the universe level `v`.
-/
@[nolint unusedArguments]
def AsSmall.{w, v, u} (D : Type u) [Category.{v} D] := ULift.{max w v} D


instance : SmallCategory (AsSmall.{w₁} C) where
  Hom X Y := ULift.{max w₁ u₁} <| X.down ⟶ Y.down
  id _ := ⟨𝟙 _⟩
  comp f g := ⟨f.down ≫ g.down⟩


/-- One half of the equivalence between `C` and `AsSmall C`. -/
@[simps]
def AsSmall.up : C ⥤ AsSmall C where
  obj X := ⟨X⟩
  map f := ⟨f⟩


/-- One half of the equivalence between `C` and `AsSmall C`. -/
@[simps]
def AsSmall.down : AsSmall C ⥤ C where
  obj X := ULift.down X
  map f := f.down


@[reassoc]
theorem down_comp {X Y Z : AsSmall C} (f : X ⟶ Y) (g : Y ⟶ Z) : (f ≫ g).down = f.down ≫ g.down :=
  rfl


@[simp]
theorem eqToHom_down {X Y : AsSmall C} (h : X = Y) :
    (eqToHom h).down = eqToHom (congrArg ULift.down h) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : CategoryTheory.AsSmall C
    h : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom h).down (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : CategoryTheory.AsSmall C
    ⊢ Eq (CategoryTheory.eqToHom ⋯).down (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The equivalence between `C` and `AsSmall C`. -/
@[simps]
def AsSmall.equiv : C ≌ AsSmall C where
  functor := AsSmall.up
  inverse := AsSmall.down
             /-
               C : Type u₁
               inst✝ : CategoryTheory.Category.{v₁, u₁} C
               ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
             -/
  unitIso := NatIso.ofComponents fun _ => eqToIso rfl
             /-
               🎉 no goals
             -/
               /-
                 C : Type u₁
                 inst✝ : CategoryTheory.Category.{v₁, u₁} C
                 ⊢ ∀ {X Y : CategoryTheory.AsSmall C} (f : Quiver.Hom X Y), Eq (CategoryTheory. …
               -/
  counitIso := NatIso.ofComponents fun _ => eqToIso <| ULift.ext _ _ rfl
               /-
                 🎉 no goals
               -/


instance [Inhabited C] : Inhabited (AsSmall C) :=
  ⟨⟨default⟩⟩


/-- The equivalence between `C` and `ULiftHom (ULift C)`. -/
def ULiftHomULiftCategory.equiv.{v', u', v, u} (C : Type u) [Category.{v} C] :
    C ≌ ULiftHom.{v'} (ULift.{u'} C) :=
  ULift.equivalence.trans ULiftHom.equiv


