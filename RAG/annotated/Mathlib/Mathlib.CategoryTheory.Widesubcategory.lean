/-- `InducedWideCategory D F P`, where `F : C → D`, is a typeclass synonym for `C`,
which provides a category structure so that the morphisms `X ⟶ Y` are the morphisms
in `D` from `F X` to `F Y` which satisfy a property `P : MorphismProperty D` that is multiplicative.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
@[nolint unusedArguments]
def InducedWideCategory (_F : C → D) (_P : MorphismProperty D) [IsMultiplicative _P] :=
  C


instance InducedWideCategory.hasCoeToSort {α : Sort*} [CoeSort D α] :
    CoeSort (InducedWideCategory D F P) α :=
  ⟨fun c => F c⟩


@[simps!]
instance InducedWideCategory.category :
    Category (InducedWideCategory D F P) where
  Hom X Y := {f : F X ⟶ F Y | P f}
  id X := ⟨𝟙 (F X), P.id_mem (F X)⟩
  comp {_ _ _} f g := ⟨f.1 ≫ g.1, P.comp_mem _ _ f.2 g.2⟩


/-- The forgetful functor from an induced wide category to the original category. -/
@[simps]
def wideInducedFunctor : InducedWideCategory D F P ⥤ D where
  obj := F
  map {_ _} f := f.1


/-- The induced functor `wideInducedFunctor F P : InducedWideCategory D F P ⥤ D`
is faithful. -/
instance InducedWideCategory.faithful : (wideInducedFunctor F P).Faithful where
  map_injective {X Y} f g eq := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : C → D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.IsMultiplicative
      X Y : CategoryTheory.InducedWideCategory D F P
      f g : Quiver.Hom X Y
      eq : Eq ((CategoryTheory.wideInducedFunctor F P).map f) ((CategoryTheory.wideI …
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : C → D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.IsMultiplicative
      X Y : CategoryTheory.InducedWideCategory D F P
      g : Quiver.Hom X Y
      val✝ : Quiver.Hom (F X) (F Y)
      property✝ : Membership.mem (setOf fun f => P f) val✝
      eq : Eq ((CategoryTheory.wideInducedFunctor F P).map ⟨val✝, property✝⟩) ((Cate …
      ⊢ Eq ⟨val✝, property✝⟩ g
    -/
    cases g
    /-
      case mk.mk
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : C → D
      P : CategoryTheory.MorphismProperty D
      inst✝ : P.IsMultiplicative
      X Y : CategoryTheory.InducedWideCategory D F P
      val✝¹ : Quiver.Hom (F X) (F Y)
      property✝¹ : Membership.mem (setOf fun f => P f) val✝¹
      val✝ : Quiver.Hom (F X) (F Y)
      property✝ : Membership.mem (setOf fun f => P f) val✝
      eq : Eq ((CategoryTheory.wideInducedFunctor F P).map ⟨val✝¹, property✝¹⟩) ((Ca …
      ⊢ Eq ⟨val✝¹, property✝¹⟩ ⟨val✝, property✝⟩
    -/
    aesop
    /-
      🎉 no goals
    -/


/--
Structure for wide subcategories. Objects ignore the morphism property.
-/
@[ext, nolint unusedArguments]
structure WideSubcategory (_P : MorphismProperty C) [IsMultiplicative _P] where
  /-- The category of which this is a wide subcategory -/
  obj : C


instance WideSubcategory.category : Category.{v₁} (WideSubcategory P) :=
  InducedWideCategory.category WideSubcategory.obj P


@[simp]
lemma WideSubcategory.id_def (X : WideSubcategory P) : (CategoryStruct.id X).1 = 𝟙 X.obj := rfl


@[simp]
lemma WideSubcategory.comp_def {X Y Z : WideSubcategory P} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).1 = (f.1 ≫ g.1 : X.obj ⟶ Z.obj) := rfl


/-- The forgetful functor from a wide subcategory into the original category
("forgetting" the condition).
-/
def wideSubcategoryInclusion : WideSubcategory P ⥤ C :=
  wideInducedFunctor WideSubcategory.obj P


@[simp]
theorem wideSubcategoryInclusion.obj (X) : (wideSubcategoryInclusion P).obj X = X.obj :=
  rfl


@[simp]
theorem wideSubcategoryInclusion.map {X Y} {f : X ⟶ Y} :
    (wideSubcategoryInclusion P).map f = f.1 :=
  rfl


/-- The inclusion of a wide subcategory is faithful. -/
instance wideSubcategory.faithful : (wideSubcategoryInclusion P).Faithful :=
  inferInstanceAs (wideInducedFunctor WideSubcategory.obj P).Faithful


