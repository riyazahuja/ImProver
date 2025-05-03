/-- Category of categories. -/
@[nolint checkUnivs]
def Cat :=
  Bundled Category.{v, u}


instance : Inhabited Cat :=
  ⟨⟨Type u, CategoryTheory.types⟩⟩

-- Porting note: maybe this coercion should be defined to be `objects.obj`?

instance : CoeSort Cat (Type u) :=
  ⟨Bundled.α⟩


instance str (C : Cat.{v, u}) : Category.{v, u} C :=
  Bundled.str C


/-- Construct a bundled `Cat` from the underlying type and the typeclass. -/
def of (C : Type u) [Category.{v} C] : Cat.{v, u} :=
  Bundled.of C


/-- Bicategory structure on `Cat` -/
instance bicategory : Bicategory.{max v u, max v u} Cat.{v, u} where
  Hom C D := C ⥤ D
  id C := 𝟭 C
  comp F G := F ⋙ G
  homCategory := fun _ _ => Functor.category
  whiskerLeft {_} {_} {_} F _ _ η := whiskerLeft F η
  whiskerRight {_} {_} {_} _ _ η H := whiskerRight η H
  associator {_} {_} {_} _ := Functor.associator
  leftUnitor {_} _ := Functor.leftUnitor
  rightUnitor {_} _ := Functor.rightUnitor
  pentagon := fun {_} {_} {_} {_} {_}=> Functor.pentagon
  triangle {_} {_} {_} := Functor.triangle


/-- `Cat` is a strict bicategory. -/
instance bicategory.strict : Bicategory.Strict Cat.{v, u} where
                          /-
                            C D : CategoryTheory.Cat
                            F : Quiver.Hom C D
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id C)  …
                          -/
  id_comp {C} {D} F := by cases F; rfl
                                   /-
                                     🎉 no goals
                                   -/
                          /-
                            C D : CategoryTheory.Cat
                            F : Quiver.Hom C D
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp F (CategoryTheory.CategoryStruct.id D …
                          -/
  comp_id {C} {D} F := by cases F; rfl
                                   /-
                                     🎉 no goals
                                   -/
              /-
                ⊢ ∀ {a b c d : CategoryTheory.Cat} (f : Quiver.Hom a b) (g : Quiver.Hom b c) ( …
              -/
  assoc := by intros; rfl
                      /-
                        🎉 no goals
                      -/


/-- Category structure on `Cat` -/
instance category : LargeCategory.{max v u} Cat.{v, u} :=
  StrictBicategory.category Cat.{v, u}


@[simp]
theorem id_obj {C : Cat} (X : C) : (𝟙 C : C ⥤ C).obj X = X :=
  rfl


@[simp]
theorem id_map {C : Cat} {X Y : C} (f : X ⟶ Y) : (𝟙 C : C ⥤ C).map f = f :=
  rfl


@[simp]
theorem comp_obj {C D E : Cat} (F : C ⟶ D) (G : D ⟶ E) (X : C) : (F ≫ G).obj X = G.obj (F.obj X) :=
  rfl


@[simp]
theorem comp_map {C D E : Cat} (F : C ⟶ D) (G : D ⟶ E) {X Y : C} (f : X ⟶ Y) :
    (F ≫ G).map f = G.map (F.map f) :=
  rfl


@[simp]
theorem id_app {C D : Cat} (F : C ⟶ D) (X : C) : (𝟙 F : F ⟶ F).app X = 𝟙 (F.obj X) := rfl


@[simp]
theorem comp_app {C D : Cat} {F G H : C ⟶ D} (α : F ⟶ G) (β : G ⟶ H) (X : C) :
    (α ≫ β).app X = α.app X ≫ β.app X := rfl


@[simp]
lemma whiskerLeft_app {C D E : Cat} (F : C ⟶ D) {G H : D ⟶ E} (η : G ⟶ H) (X : C) :
    (F ◁ η).app X = η.app (F.obj X) :=
  rfl


@[simp]
lemma whiskerRight_app {C D E : Cat} {F G : C ⟶ D} (H : D ⟶ E) (η : F ⟶ G) (X : C) :
    (η ▷ H).app X = H.map (η.app X) :=
  rfl


@[simp]
theorem eqToHom_app {C D : Cat} (F G : C ⟶ D) (h : F = G) (X : C) :
    (eqToHom h).app X = eqToHom (Functor.congr_obj h X) :=
  CategoryTheory.eqToHom_app h X


                                                                                          /-
                                                                                            B C : CategoryTheory.Cat
                                                                                            F : Quiver.Hom B C
                                                                                            X : ↑B
                                                                                            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id B) …
                                                                                          -/
lemma leftUnitor_hom_app {B C : Cat} (F : B ⟶ C) (X : B) : (λ_ F).hom.app X = eqToHom (by simp) :=
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
  rfl


                                                                                          /-
                                                                                            B C : CategoryTheory.Cat
                                                                                            F : Quiver.Hom B C
                                                                                            X : ↑B
                                                                                            ⊢ Eq (F.obj X) ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
                                                                                          -/
lemma leftUnitor_inv_app {B C : Cat} (F : B ⟶ C) (X : B) : (λ_ F).inv.app X = eqToHom (by simp) :=
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
  rfl


                                                                                           /-
                                                                                             B C : CategoryTheory.Cat
                                                                                             F : Quiver.Hom B C
                                                                                             X : ↑B
                                                                                             ⊢ Eq ((CategoryTheory.CategoryStruct.comp F (CategoryTheory.CategoryStruct.id  …
                                                                                           -/
lemma rightUnitor_hom_app {B C : Cat} (F : B ⟶ C) (X : B) : (ρ_ F).hom.app X = eqToHom (by simp) :=
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
  rfl


                                                                                           /-
                                                                                             B C : CategoryTheory.Cat
                                                                                             F : Quiver.Hom B C
                                                                                             X : ↑B
                                                                                             ⊢ Eq (F.obj X) ((CategoryTheory.CategoryStruct.comp F (CategoryTheory.Category …
                                                                                           -/
lemma rightUnitor_inv_app {B C : Cat} (F : B ⟶ C) (X : B) : (ρ_ F).inv.app X = eqToHom (by simp) :=
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
  rfl


lemma associator_hom_app {B C D E : Cat} (F : B ⟶ C) (G : C ⟶ D) (H : D ⟶ E) (X : B) :
                                       /-
                                         B C D E : CategoryTheory.Cat
                                         F : Quiver.Hom B C
                                         G : Quiver.Hom C D
                                         H : Quiver.Hom D E
                                         X : ↑B
                                         ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
                                       -/
    (α_ F G H).hom.app X = eqToHom (by simp) :=
                                       /-
                                         🎉 no goals
                                       -/
  rfl


lemma associator_inv_app {B C D E : Cat} (F : B ⟶ C) (G : C ⟶ D) (H : D ⟶ E) (X : B) :
                                       /-
                                         B C D E : CategoryTheory.Cat
                                         F : Quiver.Hom B C
                                         G : Quiver.Hom C D
                                         H : Quiver.Hom D E
                                         X : ↑B
                                         ⊢ Eq ((CategoryTheory.CategoryStruct.comp F (CategoryTheory.CategoryStruct.com …
                                       -/
    (α_ F G H).inv.app X = eqToHom (by simp) :=
                                       /-
                                         🎉 no goals
                                       -/
  rfl


/-- The identity in the category of categories equals the identity functor.-/
theorem id_eq_id (X : Cat) : 𝟙 X = 𝟭 X := rfl


/-- Composition in the category of categories equals functor composition.-/
theorem comp_eq_comp {X Y Z : Cat} (F : X ⟶ Y) (G : Y ⟶ Z) : F ≫ G = F ⋙ G := rfl


@[simp] theorem of_α (C) [Category C] : (of C).α = C := rfl


/-- Functor that gets the set of objects of a category. It is not
called `forget`, because it is not a faithful functor. -/
def objects : Cat.{v, u} ⥤ Type u where
  obj C := C
  map F := F.obj

-- Porting note: this instance was needed for CategoryTheory.Category.Cat.Limit

instance (X : Cat.{v, u}) : Category (objects.obj X) := (inferInstance : Category X)


/-- Any isomorphism in `Cat` induces an equivalence of the underlying categories. -/
def equivOfIso {C D : Cat} (γ : C ≅ D) : C ≌ D where
  functor := γ.hom
  inverse := γ.inv
  unitIso := eqToIso <| Eq.symm γ.hom_inv_id
  counitIso := eqToIso γ.inv_hom_id


/-- Embedding `Type` into `Cat` as discrete categories.

This ought to be modelled as a 2-functor!
-/
@[simps]
def typeToCat : Type u ⥤ Cat where
  obj X := Cat.of (Discrete X)
  map := fun {X} {Y} f => by
    /-
      X Y : Type u
      f : Quiver.Hom X Y
      ⊢ Quiver.Hom ((fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X)) X)  …
    -/
    dsimp
    /-
      X Y : Type u
      f : Quiver.Hom X Y
      ⊢ Quiver.Hom (CategoryTheory.Cat.of (CategoryTheory.Discrete X)) (CategoryTheo …
    -/
    exact Discrete.functor (Discrete.mk ∘ f)
    /-
      🎉 no goals
    -/
  map_id X := by
    /-
      X : Type u
      ⊢ Eq ({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X), map …
    -/
    apply Functor.ext
      /-
        case h_map
        X : Type u
        ⊢ autoParam (∀ (X_1 Y : ↑({ obj := fun X => CategoryTheory.Cat.of (CategoryThe …
      -/
    · intro X Y f
      /-
        case h_map
        X✝ : Type u
        X Y : ↑({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X), m …
        f : Quiver.Hom X Y
        ⊢ Eq (({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X), ma …
      -/
      cases f
      /-
        case h_map.up
        X✝ : Type u
        X Y : ↑({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X), m …
        down✝ : PLift (Eq X.as Y.as)
        ⊢ Eq (({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X), ma …
      -/
      simp only [id_eq, eqToHom_refl, Cat.id_map, Category.comp_id, Category.id_comp]
      /-
        case h_map.up
        X✝ : Type u
        X Y : ↑({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X), m …
        down✝ : PLift (Eq X.as Y.as)
        ⊢ Eq ((CategoryTheory.Discrete.functor (Function.comp CategoryTheory.Discrete. …
      -/
      apply ULift.ext
      /-
        case h_map.up.h
        X✝ : Type u
        X Y : ↑({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X), m …
        down✝ : PLift (Eq X.as Y.as)
        ⊢ Eq ((CategoryTheory.Discrete.functor (Function.comp CategoryTheory.Discrete. …
      -/
      aesop_cat
      /-
        🎉 no goals
      -/
      /-
        case h_obj
        X : Type u
        ⊢ ∀ (X_1 : ↑({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete  …
      -/
    · aesop_cat
      /-
        🎉 no goals
      -/
                     /-
                       X✝ Y✝ Z✝ : Type u
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun X => CategoryTheory.Cat.of (CategoryTheory.Discrete X), map …
                     -/
  map_comp f g := by apply Functor.ext; aesop_cat
                                        /-
                                          🎉 no goals
                                        -/


instance : Functor.Faithful typeToCat.{u} where
  map_injective {_X} {_Y} _f _g h :=
    funext fun x => congr_arg Discrete.as (Functor.congr_obj h ⟨x⟩)


instance : Functor.Full typeToCat.{u} where
  map_surjective F := ⟨Discrete.as ∘ F.obj ∘ Discrete.mk, by
    /-
      X✝ Y✝ : Type u
      F : Quiver.Hom (CategoryTheory.typeToCat.obj X✝) (CategoryTheory.typeToCat.obj …
      ⊢ Eq (CategoryTheory.typeToCat.map (Function.comp CategoryTheory.Discrete.as ( …
    -/
    apply Functor.ext
      /-
        case h_map
        X✝ Y✝ : Type u
        F : Quiver.Hom (CategoryTheory.typeToCat.obj X✝) (CategoryTheory.typeToCat.obj …
        ⊢ autoParam (∀ (X Y : ↑(CategoryTheory.typeToCat.obj X✝)) (f : Quiver.Hom X Y) …
      -/
    · intro x y f
      /-
        case h_map
        X✝ Y✝ : Type u
        F : Quiver.Hom (CategoryTheory.typeToCat.obj X✝) (CategoryTheory.typeToCat.obj …
        x y : ↑(CategoryTheory.typeToCat.obj X✝)
        f : Quiver.Hom x y
        ⊢ Eq ((CategoryTheory.typeToCat.map (Function.comp CategoryTheory.Discrete.as  …
      -/
      dsimp
      /-
        case h_map
        X✝ Y✝ : Type u
        F : Quiver.Hom (CategoryTheory.typeToCat.obj X✝) (CategoryTheory.typeToCat.obj …
        x y : ↑(CategoryTheory.typeToCat.obj X✝)
        f : Quiver.Hom x y
        ⊢ Eq ((CategoryTheory.Discrete.functor (Function.comp CategoryTheory.Discrete. …
      -/
      apply ULift.ext
      /-
        case h_map.h
        X✝ Y✝ : Type u
        F : Quiver.Hom (CategoryTheory.typeToCat.obj X✝) (CategoryTheory.typeToCat.obj …
        x y : ↑(CategoryTheory.typeToCat.obj X✝)
        f : Quiver.Hom x y
        ⊢ Eq ((CategoryTheory.Discrete.functor (Function.comp CategoryTheory.Discrete. …
      -/
      aesop_cat
      /-
        🎉 no goals
      -/
      /-
        case h_obj
        X✝ Y✝ : Type u
        F : Quiver.Hom (CategoryTheory.typeToCat.obj X✝) (CategoryTheory.typeToCat.obj …
        ⊢ ∀ (X : ↑(CategoryTheory.typeToCat.obj X✝)), Eq ((CategoryTheory.typeToCat.ma …
      -/
    · rintro ⟨x⟩
      /-
        case h_obj.mk
        X✝ Y✝ : Type u
        F : Quiver.Hom (CategoryTheory.typeToCat.obj X✝) (CategoryTheory.typeToCat.obj …
        x : X✝
        ⊢ Eq ((CategoryTheory.typeToCat.map (Function.comp CategoryTheory.Discrete.as  …
      -/
      apply Discrete.ext
      /-
        case h_obj.mk.as
        X✝ Y✝ : Type u
        F : Quiver.Hom (CategoryTheory.typeToCat.obj X✝) (CategoryTheory.typeToCat.obj …
        x : X✝
        ⊢ Eq ((CategoryTheory.typeToCat.map (Function.comp CategoryTheory.Discrete.as  …
      -/
      rfl⟩
      /-
        🎉 no goals
      -/


