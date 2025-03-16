/--
`OverClass X S` is the typeclass containing the data of a structure morphism `X ↘ S : X ⟶ S`.
-/
class OverClass (X S : C) : Type v where
  ofHom ::
  /-- The structure morphism. Use `X ↘ S` instead. -/
  hom : X ⟶ S


/--
The structure morphism `X ↘ S : X ⟶ S` given `OverClass X S`.
The instance argument is an `optParam` instead so that it appears in the discrimination tree.
-/
def over (X S : C) (_ : OverClass X S := by infer_instance) : X ⟶ S := OverClass.hom


/-- The structure morphism `X ↘ S : X ⟶ S` given `OverClass X S`. -/
notation:90 X:90 " ↘ " S:90 => CategoryTheory.over X S inferInstance


/-- See Note [custom simps projection] -/
def OverClass.Simps.over (X S : C) [OverClass X S] : X ⟶ S := X ↘ S


/--
`X.CanonicallyOverClass S` is the typeclass containing the data of a
structure morphism `X ↘ S : X ⟶ S`,
and that `S` is (uniquely) inferable from the structure of `X`.
-/
class CanonicallyOverClass (X : C) (S : semiOutParam C) extends OverClass X S where


/-- See Note [custom simps projection] -/
def CanonicallyOverClass.Simps.over (X S : C) [CanonicallyOverClass X S] : X ⟶ S := X ↘ S


@[simps]
instance : OverClass X X := ⟨𝟙 _⟩


instance : IsIso (S ↘ S) := inferInstanceAs (IsIso (𝟙 S))

-- This cannot be a simp lemma be cause it loops with `comp_over`.

@[simps (config := .lemmasOnly)]
instance (priority := 900) [CanonicallyOverClass X Y] [OverClass Y S] : OverClass X S :=
  ⟨X ↘ Y ≫ Y ↘ S⟩


/-- Given `OverClass X S` and `OverClass Y S` and `f : X ⟶ Y`,
`HomIsOver f S` is the typeclass asserting `f` commutes with the structure morphisms. -/
class HomIsOver (f : X ⟶ Y) (S : C) [OverClass X S] [OverClass Y S] : Prop where
  comp_over : f ≫ Y ↘ S = X ↘ S := by aesop


@[reassoc (attr := simp)]
lemma comp_over [OverClass X S] [OverClass Y S] [HomIsOver f S] :
    f ≫ Y ↘ S = X ↘ S :=
  HomIsOver.comp_over


instance [OverClass X S] : HomIsOver (𝟙 X) S where


instance [OverClass X S] [OverClass Y S] [OverClass Z S]
    (f : X ⟶ Y) (g : Y ⟶ Z) [HomIsOver f S] [HomIsOver g S] :
    HomIsOver (f ≫ g) S where


/-- `Scheme.IsOverTower X Y S` is the typeclass asserting that the structure morphisms
`X ↘ Y`, `Y ↘ S`, and `X ↘ S` commute. -/
abbrev IsOverTower (X Y S : C) [OverClass X S] [OverClass Y S] [OverClass X Y] :=
  HomIsOver (X ↘ Y) S


instance [OverClass X S] : IsOverTower X X S where

instance [OverClass X S] : IsOverTower X S S where


instance [CanonicallyOverClass X Y] [OverClass Y S] : IsOverTower X Y S :=
  ⟨rfl⟩


lemma homIsOver_of_isOverTower [OverClass X S] [OverClass X S'] [OverClass Y S]
    [OverClass Y S'] [OverClass S S']
    [IsOverTower X S S'] [IsOverTower Y S S'] [HomIsOver f S] : HomIsOver f S' := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    S S' : C
    inst✝⁷ : CategoryTheory.OverClass X S
    inst✝⁶ : CategoryTheory.OverClass X S'
    inst✝⁵ : CategoryTheory.OverClass Y S
    inst✝⁴ : CategoryTheory.OverClass Y S'
    inst✝³ : CategoryTheory.OverClass S S'
    inst✝² : CategoryTheory.IsOverTower X S S'
    inst✝¹ : CategoryTheory.IsOverTower Y S S'
    inst✝ : CategoryTheory.HomIsOver f S
    ⊢ CategoryTheory.HomIsOver f S'
  -/
  constructor
  /-
    case comp_over
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    S S' : C
    inst✝⁷ : CategoryTheory.OverClass X S
    inst✝⁶ : CategoryTheory.OverClass X S'
    inst✝⁵ : CategoryTheory.OverClass Y S
    inst✝⁴ : CategoryTheory.OverClass Y S'
    inst✝³ : CategoryTheory.OverClass S S'
    inst✝² : CategoryTheory.IsOverTower X S S'
    inst✝¹ : CategoryTheory.IsOverTower Y S S'
    inst✝ : CategoryTheory.HomIsOver f S
    ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.over Y S …
  -/
  rw [← comp_over (Y ↘ S), comp_over_assoc f, comp_over]
  /-
    🎉 no goals
  -/


instance [CanonicallyOverClass X S]
    [OverClass X S'] [OverClass Y S] [OverClass Y S'] [OverClass S S']
    [IsOverTower X S S'] [IsOverTower Y S S'] [HomIsOver f S] : HomIsOver f S' :=
  homIsOver_of_isOverTower f S S'


instance [OverClass X S]
    [OverClass X S'] [CanonicallyOverClass Y S] [OverClass Y S'] [OverClass S S']
    [IsOverTower X S S'] [IsOverTower Y S S'] [HomIsOver f S] : HomIsOver f S' :=
  homIsOver_of_isOverTower f S S'


variable (X) in
/-- Bundle `X` with an `OverClass X S` instance into `Over S`. -/
@[simps! hom left]
def OverClass.asOver [OverClass X S] : Over S := Over.mk (X ↘ S)


/-- Bundle a morphism `f : X ⟶ Y` with `HomIsOver f S` into a morphism in `Over S`. -/
@[simps! left]
def OverClass.asOverHom [OverClass X S] [OverClass Y S] (f : X ⟶ Y) [HomIsOver f S] :
    OverClass.asOver X S ⟶ OverClass.asOver Y S :=
  Over.homMk f (comp_over f S)


@[simps]
instance OverClass.fromOver {S : C} (X : Over S) : OverClass X.left S where
  hom := X.hom


instance {S : C} {X Y : Over S} (f : X ⟶ Y) : HomIsOver f.left S where
  comp_over := Over.w f


