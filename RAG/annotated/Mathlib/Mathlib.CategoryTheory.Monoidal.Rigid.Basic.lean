/-- An exact pairing is a pair of objects `X Y : C` which admit
  a coevaluation and evaluation morphism which fulfill two triangle equalities. -/
class ExactPairing (X Y : C) where
  /-- Coevaluation of an exact pairing.

  Do not use directly. Use `ExactPairing.coevaluation` instead. -/
  coevaluation' : 𝟙_ C ⟶ X ⊗ Y
  /-- Evaluation of an exact pairing.

  Do not use directly. Use `ExactPairing.evaluation` instead. -/
  evaluation' : Y ⊗ X ⟶ 𝟙_ C
  coevaluation_evaluation' :
    Y ◁ coevaluation' ≫ (α_ _ _ _).inv ≫ evaluation' ▷ Y = (ρ_ Y).hom ≫ (λ_ Y).inv := by
    aesop_cat
  evaluation_coevaluation' :
    coevaluation' ▷ X ≫ (α_ _ _ _).hom ≫ X ◁ evaluation' = (λ_ X).hom ≫ (ρ_ X).inv := by
    aesop_cat


/-- Coevaluation of an exact pairing. -/
def coevaluation : 𝟙_ C ⟶ X ⊗ Y := @coevaluation' _ _ _ X Y _


/-- Evaluation of an exact pairing. -/
def evaluation : Y ⊗ X ⟶ 𝟙_ C := @evaluation' _ _ _ X Y _


@[inherit_doc] notation "η_" => ExactPairing.coevaluation

@[inherit_doc] notation "ε_" => ExactPairing.evaluation


lemma coevaluation_evaluation :
    Y ◁ η_ _ _ ≫ (α_ _ _ _).inv ≫ ε_ X _ ▷ Y = (ρ_ Y).hom ≫ (λ_ Y).inv :=
  coevaluation_evaluation'


lemma evaluation_coevaluation :
    η_ _ _ ▷ X ≫ (α_ _ _ _).hom ≫ X ◁ ε_ _ Y = (λ_ X).hom ≫ (ρ_ X).inv :=
  evaluation_coevaluation'


lemma coevaluation_evaluation'' :
    Y ◁ η_ X Y ⊗≫ ε_ X Y ▷ Y = ⊗𝟙.hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝ : CategoryTheory.ExactPairing X Y
    ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.MonoidalCategoryStruct.whisk …
  -/
                                          /-
                                            🎉 no goals
                                          -/
  convert coevaluation_evaluation X Y <;> simp [monoidalComp]
                                          /-
                                            🎉 no goals
                                          -/


lemma evaluation_coevaluation'' :
    η_ X Y ▷ X ⊗≫ X ◁ ε_ X Y = ⊗𝟙.hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝ : CategoryTheory.ExactPairing X Y
    ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.MonoidalCategoryStruct.whisk …
  -/
                                          /-
                                            🎉 no goals
                                          -/
  convert evaluation_coevaluation X Y <;> simp [monoidalComp]
                                          /-
                                            🎉 no goals
                                          -/


attribute [reassoc (attr := simp)] ExactPairing.coevaluation_evaluation

attribute [reassoc (attr := simp)] ExactPairing.evaluation_coevaluation


instance exactPairingUnit : ExactPairing (𝟙_ C) (𝟙_ C) where
  coevaluation' := (ρ_ _).inv
  evaluation' := (ρ_ _).hom
                                 /-
                                   C : Type u₁
                                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                   inst✝ : CategoryTheory.MonoidalCategory C
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                 -/
  coevaluation_evaluation' := by monoidal_coherence
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   C : Type u₁
                                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                   inst✝ : CategoryTheory.MonoidalCategory C
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                 -/
  evaluation_coevaluation' := by monoidal_coherence
                                 /-
                                   🎉 no goals
                                 -/


/-- A class of objects which have a right dual. -/
class HasRightDual (X : C) where
  /-- The right dual of the object `X`. -/
  rightDual : C
  [exact : ExactPairing X rightDual]


/-- A class of objects which have a left dual. -/
class HasLeftDual (Y : C) where
  /-- The left dual of the object `X`. -/
  leftDual : C
  [exact : ExactPairing leftDual Y]


@[inherit_doc] prefix:1024 "ᘁ" => leftDual

@[inherit_doc] postfix:1024 "ᘁ" => rightDual


instance hasRightDualUnit : HasRightDual (𝟙_ C) where
  rightDual := 𝟙_ C


instance hasLeftDualUnit : HasLeftDual (𝟙_ C) where
  leftDual := 𝟙_ C


instance hasRightDualLeftDual {X : C} [HasLeftDual X] : HasRightDual ᘁX where
  rightDual := X


instance hasLeftDualRightDual {X : C} [HasRightDual X] : HasLeftDual Xᘁ where
  leftDual := X


@[simp]
theorem leftDual_rightDual {X : C} [HasRightDual X] : ᘁXᘁ = X :=
  rfl


@[simp]
theorem rightDual_leftDual {X : C} [HasLeftDual X] : (ᘁX)ᘁ = X :=
  rfl


/-- The right adjoint mate `fᘁ : Xᘁ ⟶ Yᘁ` of a morphism `f : X ⟶ Y`. -/
def rightAdjointMate {X Y : C} [HasRightDual X] [HasRightDual Y] (f : X ⟶ Y) : Yᘁ ⟶ Xᘁ :=
  (ρ_ _).inv ≫ _ ◁ η_ _ _ ≫ _ ◁ f ▷ _ ≫ (α_ _ _ _).inv ≫ ε_ _ _ ▷ _ ≫ (λ_ _).hom


/-- The left adjoint mate `ᘁf : ᘁY ⟶ ᘁX` of a morphism `f : X ⟶ Y`. -/
def leftAdjointMate {X Y : C} [HasLeftDual X] [HasLeftDual Y] (f : X ⟶ Y) : ᘁY ⟶ ᘁX :=
  (λ_ _).inv ≫ η_ (ᘁX) X ▷ _ ≫ (_ ◁ f) ▷ _ ≫ (α_ _ _ _).hom ≫ _ ◁ ε_ _ _ ≫ (ρ_ _).hom


@[inherit_doc] notation f "ᘁ" => rightAdjointMate f

@[inherit_doc] notation "ᘁ" f => leftAdjointMate f


@[simp]
theorem rightAdjointMate_id {X : C} [HasRightDual X] : (𝟙 X)ᘁ = 𝟙 (Xᘁ) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X : C
    inst✝ : CategoryTheory.HasRightDual X
    ⊢ Eq (CategoryTheory.rightAdjointMate (CategoryTheory.CategoryStruct.id X)) (C …
  -/
  simp [rightAdjointMate]
  /-
    🎉 no goals
  -/


@[simp]
theorem leftAdjointMate_id {X : C} [HasLeftDual X] : (ᘁ(𝟙 X)) = 𝟙 (ᘁX) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X : C
    inst✝ : CategoryTheory.HasLeftDual X
    ⊢ Eq (CategoryTheory.leftAdjointMate (CategoryTheory.CategoryStruct.id X)) (Ca …
  -/
  simp [leftAdjointMate]
  /-
    🎉 no goals
  -/


theorem rightAdjointMate_comp {X Y Z : C} [HasRightDual X] [HasRightDual Y] {f : X ⟶ Y}
    {g : Xᘁ ⟶ Z} :
    fᘁ ≫ g =
      (ρ_ (Yᘁ)).inv ≫
        _ ◁ η_ X (Xᘁ) ≫ _ ◁ (f ⊗ g) ≫ (α_ (Yᘁ) Y Z).inv ≫ ε_ Y (Yᘁ) ▷ _ ≫ (λ_ Z).hom :=
  calc
    _ = 𝟙 _ ⊗≫ (Yᘁ : C) ◁ η_ X Xᘁ ≫ Yᘁ ◁ f ▷ Xᘁ ⊗≫ (ε_ Y Yᘁ ▷ Xᘁ ≫ 𝟙_ C ◁ g) ⊗≫ 𝟙 _ := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.MonoidalCategory C
        X Y Z : C
        inst✝¹ : CategoryTheory.HasRightDual X
        inst✝ : CategoryTheory.HasRightDual Y
        f : Quiver.Hom X Y
        g : Quiver.Hom (CategoryTheory.HasRightDual.rightDual X) Z
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.rightAdjointMate f) g …
      -/
      dsimp only [rightAdjointMate]; monoidal
                                     /-
                                       🎉 no goals
                                     -/
    _ = _ := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.MonoidalCategory C
        X Y Z : C
        inst✝¹ : CategoryTheory.HasRightDual X
        inst✝ : CategoryTheory.HasRightDual Y
        f : Quiver.Hom X Y
        g : Quiver.Hom (CategoryTheory.HasRightDual.rightDual X) Z
        ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
      -/
      rw [← whisker_exchange, tensorHom_def]; monoidal
                                              /-
                                                🎉 no goals
                                              -/


theorem leftAdjointMate_comp {X Y Z : C} [HasLeftDual X] [HasLeftDual Y] {f : X ⟶ Y}
    {g : (ᘁX) ⟶ Z} :
    (ᘁf) ≫ g =
      (λ_ _).inv ≫
        η_ (ᘁX : C) X ▷ _ ≫ (g ⊗ f) ▷ _ ≫ (α_ _ _ _).hom ≫ _ ◁ ε_ _ _ ≫ (ρ_ _).hom :=
  calc
    _ = 𝟙 _ ⊗≫ η_ (ᘁX : C) X ▷ (ᘁY) ⊗≫ (ᘁX) ◁ f ▷ (ᘁY) ⊗≫ ((ᘁX) ◁ ε_ (ᘁY) Y ≫ g ▷ 𝟙_ C) ⊗≫ 𝟙 _ := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.MonoidalCategory C
        X Y Z : C
        inst✝¹ : CategoryTheory.HasLeftDual X
        inst✝ : CategoryTheory.HasLeftDual Y
        f : Quiver.Hom X Y
        g : Quiver.Hom (CategoryTheory.HasLeftDual.leftDual X) Z
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.leftAdjointMate f) g) …
      -/
      dsimp only [leftAdjointMate]; monoidal
                                    /-
                                      🎉 no goals
                                    -/
    _ = _ := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.MonoidalCategory C
        X Y Z : C
        inst✝¹ : CategoryTheory.HasLeftDual X
        inst✝ : CategoryTheory.HasLeftDual Y
        f : Quiver.Hom X Y
        g : Quiver.Hom (CategoryTheory.HasLeftDual.leftDual X) Z
        ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
      -/
      rw [whisker_exchange, tensorHom_def']; monoidal
                                             /-
                                               🎉 no goals
                                             -/


/-- The composition of right adjoint mates is the adjoint mate of the composition. -/
@[reassoc]
theorem comp_rightAdjointMate {X Y Z : C} [HasRightDual X] [HasRightDual Y] [HasRightDual Z]
    {f : X ⟶ Y} {g : Y ⟶ Z} : (f ≫ g)ᘁ = gᘁ ≫ fᘁ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasRightDual X
    inst✝¹ : CategoryTheory.HasRightDual Y
    inst✝ : CategoryTheory.HasRightDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.rightAdjointMate (CategoryTheory.CategoryStruct.comp f g) …
  -/
  rw [rightAdjointMate_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasRightDual X
    inst✝¹ : CategoryTheory.HasRightDual Y
    inst✝ : CategoryTheory.HasRightDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.rightAdjointMate (CategoryTheory.CategoryStruct.comp f g) …
  -/
  simp only [rightAdjointMate, comp_whiskerRight]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasRightDual X
    inst✝¹ : CategoryTheory.HasRightDual Y
    inst✝ : CategoryTheory.HasRightDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [← Category.assoc]; congr 3; simp only [Category.assoc]
  /-
    case e_a.e_a.e_a
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasRightDual X
    inst✝¹ : CategoryTheory.HasRightDual Y
    inst✝ : CategoryTheory.HasRightDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [← MonoidalCategory.whiskerLeft_comp]; congr 2
  /-
    case e_a.e_a.e_a.e_a.e_f
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasRightDual X
    inst✝¹ : CategoryTheory.HasRightDual Y
    inst✝ : CategoryTheory.HasRightDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ExactPairing.coevalua …
  -/
  symm
  calc
    _ = 𝟙 _ ⊗≫ (η_ Y Yᘁ ▷ 𝟙_ C ≫ (Y ⊗ Yᘁ) ◁ η_ X Xᘁ) ⊗≫ Y ◁ Yᘁ ◁ f ▷ Xᘁ ⊗≫
        Y ◁ ε_ Y Yᘁ ▷ Xᘁ ⊗≫ g ▷ Xᘁ ⊗≫ 𝟙 _ := by
      rw [tensorHom_def']; monoidal
    _ = η_ X Xᘁ ⊗≫ (η_ Y Yᘁ ▷ (X ⊗ Xᘁ) ≫ (Y ⊗ Yᘁ) ◁ f ▷ Xᘁ) ⊗≫
        Y ◁ ε_ Y Yᘁ ▷ Xᘁ ⊗≫ g ▷ Xᘁ ⊗≫ 𝟙 _ := by
      rw [← whisker_exchange]; monoidal
    _ = η_ X Xᘁ ⊗≫ f ▷ Xᘁ ⊗≫ (η_ Y Yᘁ ▷ Y ⊗≫ Y ◁ ε_ Y Yᘁ) ▷ Xᘁ ⊗≫ g ▷ Xᘁ ⊗≫ 𝟙 _ := by
      rw [← whisker_exchange]; monoidal
    _ = η_ X Xᘁ ≫ f ▷ Xᘁ ≫ g ▷ Xᘁ := by
      rw [evaluation_coevaluation'']; monoidal


/-- The composition of left adjoint mates is the adjoint mate of the composition. -/
@[reassoc]
theorem comp_leftAdjointMate {X Y Z : C} [HasLeftDual X] [HasLeftDual Y] [HasLeftDual Z] {f : X ⟶ Y}
    {g : Y ⟶ Z} : (ᘁf ≫ g) = (ᘁg) ≫ ᘁf := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasLeftDual X
    inst✝¹ : CategoryTheory.HasLeftDual Y
    inst✝ : CategoryTheory.HasLeftDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.leftAdjointMate (CategoryTheory.CategoryStruct.comp f g)) …
  -/
  rw [leftAdjointMate_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasLeftDual X
    inst✝¹ : CategoryTheory.HasLeftDual Y
    inst✝ : CategoryTheory.HasLeftDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.leftAdjointMate (CategoryTheory.CategoryStruct.comp f g)) …
  -/
  simp only [leftAdjointMate, MonoidalCategory.whiskerLeft_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasLeftDual X
    inst✝¹ : CategoryTheory.HasLeftDual Y
    inst✝ : CategoryTheory.HasLeftDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [← Category.assoc]; congr 3; simp only [Category.assoc]
  /-
    case e_a.e_a.e_a
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasLeftDual X
    inst✝¹ : CategoryTheory.HasLeftDual Y
    inst✝ : CategoryTheory.HasLeftDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [← comp_whiskerRight]; congr 2
  /-
    case e_a.e_a.e_a.e_a.e_f
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    inst✝² : CategoryTheory.HasLeftDual X
    inst✝¹ : CategoryTheory.HasLeftDual Y
    inst✝ : CategoryTheory.HasLeftDual Z
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ExactPairing.coevalua …
  -/
  symm
  calc
    _ = 𝟙 _ ⊗≫ ((𝟙_ C) ◁ η_ (ᘁY) Y ≫ η_ (ᘁX) X ▷ ((ᘁY) ⊗ Y)) ⊗≫ (ᘁX) ◁ f ▷ (ᘁY) ▷ Y ⊗≫
        (ᘁX) ◁ ε_ (ᘁY) Y ▷ Y ⊗≫ (ᘁX) ◁ g := by
      rw [tensorHom_def]; monoidal
    _ = η_ (ᘁX) X ⊗≫ (((ᘁX) ⊗ X) ◁ η_ (ᘁY) Y ≫ ((ᘁX) ◁ f) ▷ ((ᘁY) ⊗ Y)) ⊗≫
        (ᘁX) ◁ ε_ (ᘁY) Y ▷ Y ⊗≫ (ᘁX) ◁ g := by
      rw [whisker_exchange]; monoidal
    _ = η_ (ᘁX) X ⊗≫ ((ᘁX) ◁ f) ⊗≫ (ᘁX) ◁ (Y ◁ η_ (ᘁY) Y ⊗≫ ε_ (ᘁY) Y ▷ Y) ⊗≫ (ᘁX) ◁ g := by
      rw [whisker_exchange]; monoidal
    _ = η_ (ᘁX) X ≫ (ᘁX) ◁ f ≫ (ᘁX) ◁ g := by
      rw [coevaluation_evaluation'']; monoidal


/-- Given an exact pairing on `Y Y'`,
we get a bijection on hom-sets `(Y' ⊗ X ⟶ Z) ≃ (X ⟶ Y ⊗ Z)`
by "pulling the string on the left" up or down.

This gives the adjunction `tensorLeftAdjunction Y Y' : tensorLeft Y' ⊣ tensorLeft Y`.

This adjunction is often referred to as "Frobenius reciprocity" in the
fusion categories / planar algebras / subfactors literature.
-/
def tensorLeftHomEquiv (X Y Y' Z : C) [ExactPairing Y Y'] : (Y' ⊗ X ⟶ Z) ≃ (X ⟶ Y ⊗ Z) where
  toFun f := (λ_ _).inv ≫ η_ _ _ ▷ _ ≫ (α_ _ _ _).hom ≫ _ ◁ f
  invFun f := Y' ◁ f ≫ (α_ _ _ _).inv ≫ ε_ _ _ ▷ _ ≫ (λ_ _).hom
  left_inv f := by
    calc
      _ = 𝟙 _ ⊗≫ Y' ◁ η_ Y Y' ▷ X ⊗≫ ((Y' ⊗ Y) ◁ f ≫ ε_ Y Y' ▷ Z) ⊗≫ 𝟙 _ := by
        monoidal
      _ = 𝟙 _ ⊗≫ (Y' ◁ η_ Y Y' ⊗≫ ε_ Y Y' ▷ Y') ▷ X ⊗≫ f := by
        rw [whisker_exchange]; monoidal
      _ = f := by
        rw [coevaluation_evaluation'']; monoidal
  right_inv f := by
    calc
      _ = 𝟙 _ ⊗≫ (η_ Y Y' ▷ X ≫ (Y ⊗ Y') ◁ f) ⊗≫ Y ◁ ε_ Y Y' ▷ Z ⊗≫ 𝟙 _ := by
        monoidal
      _ = f ⊗≫ (η_ Y Y' ▷ Y ⊗≫ Y ◁ ε_ Y Y') ▷ Z ⊗≫ 𝟙 _ := by
        rw [← whisker_exchange]; monoidal
      _ = f := by
        rw [evaluation_coevaluation'']; monoidal


/-- Given an exact pairing on `Y Y'`,
we get a bijection on hom-sets `(X ⊗ Y ⟶ Z) ≃ (X ⟶ Z ⊗ Y')`
by "pulling the string on the right" up or down.
-/
def tensorRightHomEquiv (X Y Y' Z : C) [ExactPairing Y Y'] : (X ⊗ Y ⟶ Z) ≃ (X ⟶ Z ⊗ Y') where
  toFun f := (ρ_ _).inv ≫ _ ◁ η_ _ _ ≫ (α_ _ _ _).inv ≫ f ▷ _
  invFun f := f ▷ _ ≫ (α_ _ _ _).hom ≫ _ ◁ ε_ _ _ ≫ (ρ_ _).hom
  left_inv f := by
    calc
      _ = 𝟙 _ ⊗≫ X ◁ η_ Y Y' ▷ Y ⊗≫ (f ▷ (Y' ⊗ Y) ≫ Z ◁ ε_ Y Y') ⊗≫ 𝟙 _ := by
        monoidal
      _ = 𝟙 _ ⊗≫ X ◁ (η_ Y Y' ▷ Y ⊗≫ Y ◁ ε_ Y Y') ⊗≫ f := by
        rw [← whisker_exchange]; monoidal
      _ = f := by
        rw [evaluation_coevaluation'']; monoidal
  right_inv f := by
    calc
      _ = 𝟙 _ ⊗≫ (X ◁ η_ Y Y' ≫ f ▷ (Y ⊗ Y')) ⊗≫ Z ◁ ε_ Y Y' ▷ Y' ⊗≫ 𝟙 _ := by
        monoidal
      _ = f ⊗≫ Z ◁ (Y' ◁ η_ Y Y' ⊗≫ ε_ Y Y' ▷ Y') ⊗≫ 𝟙 _ := by
        rw [whisker_exchange]; monoidal
      _ = f := by
        rw [coevaluation_evaluation'']; monoidal


theorem tensorLeftHomEquiv_naturality {X Y Y' Z Z' : C} [ExactPairing Y Y'] (f : Y' ⊗ X ⟶ Z)
    (g : Z ⟶ Z') :
    (tensorLeftHomEquiv X Y Y' Z') (f ≫ g) = (tensorLeftHomEquiv X Y Y' Z) f ≫ Y ◁ g := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y Y' Z Z' : C
    inst✝ : CategoryTheory.ExactPairing Y Y'
    f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj Y' X) Z
    g : Quiver.Hom Z Z'
    ⊢ Eq ((CategoryTheory.tensorLeftHomEquiv X Y Y' Z') (CategoryTheory.CategorySt …
  -/
  simp [tensorLeftHomEquiv]
  /-
    🎉 no goals
  -/


theorem tensorLeftHomEquiv_symm_naturality {X X' Y Y' Z : C} [ExactPairing Y Y'] (f : X ⟶ X')
    (g : X' ⟶ Y ⊗ Z) :
    (tensorLeftHomEquiv X Y Y' Z).symm (f ≫ g) =
      _ ◁ f ≫ (tensorLeftHomEquiv X' Y Y' Z).symm g := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X X' Y Y' Z : C
    inst✝ : CategoryTheory.ExactPairing Y Y'
    f : Quiver.Hom X X'
    g : Quiver.Hom X' (CategoryTheory.MonoidalCategoryStruct.tensorObj Y Z)
    ⊢ Eq ((CategoryTheory.tensorLeftHomEquiv X Y Y' Z).symm (CategoryTheory.Catego …
  -/
  simp [tensorLeftHomEquiv]
  /-
    🎉 no goals
  -/


theorem tensorRightHomEquiv_naturality {X Y Y' Z Z' : C} [ExactPairing Y Y'] (f : X ⊗ Y ⟶ Z)
    (g : Z ⟶ Z') :
    (tensorRightHomEquiv X Y Y' Z') (f ≫ g) = (tensorRightHomEquiv X Y Y' Z) f ≫ g ▷ Y' := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y Y' Z Z' : C
    inst✝ : CategoryTheory.ExactPairing Y Y'
    f : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X Y) Z
    g : Quiver.Hom Z Z'
    ⊢ Eq ((CategoryTheory.tensorRightHomEquiv X Y Y' Z') (CategoryTheory.CategoryS …
  -/
  simp [tensorRightHomEquiv]
  /-
    🎉 no goals
  -/


theorem tensorRightHomEquiv_symm_naturality {X X' Y Y' Z : C} [ExactPairing Y Y'] (f : X ⟶ X')
    (g : X' ⟶ Z ⊗ Y') :
    (tensorRightHomEquiv X Y Y' Z).symm (f ≫ g) =
      f ▷ Y ≫ (tensorRightHomEquiv X' Y Y' Z).symm g := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X X' Y Y' Z : C
    inst✝ : CategoryTheory.ExactPairing Y Y'
    f : Quiver.Hom X X'
    g : Quiver.Hom X' (CategoryTheory.MonoidalCategoryStruct.tensorObj Z Y')
    ⊢ Eq ((CategoryTheory.tensorRightHomEquiv X Y Y' Z).symm (CategoryTheory.Categ …
  -/
  simp [tensorRightHomEquiv]
  /-
    🎉 no goals
  -/


/-- If `Y Y'` have an exact pairing,
then the functor `tensorLeft Y'` is left adjoint to `tensorLeft Y`.
-/
def tensorLeftAdjunction (Y Y' : C) [ExactPairing Y Y'] : tensorLeft Y' ⊣ tensorLeft Y :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Z => tensorLeftHomEquiv X Y Y' Z
      homEquiv_naturality_left_symm := fun f g => tensorLeftHomEquiv_symm_naturality f g
      homEquiv_naturality_right := fun f g => tensorLeftHomEquiv_naturality f g }


/-- If `Y Y'` have an exact pairing,
then the functor `tensor_right Y` is left adjoint to `tensor_right Y'`.
-/
def tensorRightAdjunction (Y Y' : C) [ExactPairing Y Y'] : tensorRight Y ⊣ tensorRight Y' :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Z => tensorRightHomEquiv X Y Y' Z
      homEquiv_naturality_left_symm := fun f g => tensorRightHomEquiv_symm_naturality f g
      homEquiv_naturality_right := fun f g => tensorRightHomEquiv_naturality f g }


/--
If `Y` has a left dual `ᘁY`, then it is a closed object, with the internal hom functor `Y ⟶[C] -`
given by left tensoring by `ᘁY`.
This has to be a definition rather than an instance to avoid diamonds, for example between
`category_theory.monoidal_closed.functor_closed` and
`CategoryTheory.Monoidal.functorHasLeftDual`. Moreover, in concrete applications there is often
a more useful definition of the internal hom object than `ᘁY ⊗ X`, in which case the closed
structure shouldn't come from `has_left_dual` (e.g. in the category `FinVect k`, it is more
convenient to define the internal hom as `Y →ₗ[k] X` rather than `ᘁY ⊗ X` even though these are
naturally isomorphic).
-/
def closedOfHasLeftDual (Y : C) [HasLeftDual Y] : Closed Y where
  adj := tensorLeftAdjunction (ᘁY) Y


/-- `tensorLeftHomEquiv` commutes with tensoring on the right -/
theorem tensorLeftHomEquiv_tensor {X X' Y Y' Z Z' : C} [ExactPairing Y Y'] (f : X ⟶ Y ⊗ Z)
    (g : X' ⟶ Z') :
    (tensorLeftHomEquiv (X ⊗ X') Y Y' (Z ⊗ Z')).symm ((f ⊗ g) ≫ (α_ _ _ _).hom) =
      (α_ _ _ _).inv ≫ ((tensorLeftHomEquiv X Y Y' Z).symm f ⊗ g) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X X' Y Y' Z Z' : C
    inst✝ : CategoryTheory.ExactPairing Y Y'
    f : Quiver.Hom X (CategoryTheory.MonoidalCategoryStruct.tensorObj Y Z)
    g : Quiver.Hom X' Z'
    ⊢ Eq ((CategoryTheory.tensorLeftHomEquiv (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [tensorLeftHomEquiv, tensorHom_def']
  /-
    🎉 no goals
  -/


/-- `tensorRightHomEquiv` commutes with tensoring on the left -/
theorem tensorRightHomEquiv_tensor {X X' Y Y' Z Z' : C} [ExactPairing Y Y'] (f : X ⟶ Z ⊗ Y')
    (g : X' ⟶ Z') :
    (tensorRightHomEquiv (X' ⊗ X) Y Y' (Z' ⊗ Z)).symm ((g ⊗ f) ≫ (α_ _ _ _).inv) =
      (α_ _ _ _).hom ≫ (g ⊗ (tensorRightHomEquiv X Y Y' Z).symm f) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X X' Y Y' Z Z' : C
    inst✝ : CategoryTheory.ExactPairing Y Y'
    f : Quiver.Hom X (CategoryTheory.MonoidalCategoryStruct.tensorObj Z Y')
    g : Quiver.Hom X' Z'
    ⊢ Eq ((CategoryTheory.tensorRightHomEquiv (CategoryTheory.MonoidalCategoryStru …
  -/
  simp [tensorRightHomEquiv, tensorHom_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem tensorLeftHomEquiv_symm_coevaluation_comp_whiskerLeft {Y Y' Z : C} [ExactPairing Y Y']
    (f : Y' ⟶ Z) : (tensorLeftHomEquiv _ _ _ _).symm (η_ _ _ ≫ Y ◁ f) = (ρ_ _).hom ≫ f := by
  calc
    _ = Y' ◁ η_ Y Y' ⊗≫ ((Y' ⊗ Y) ◁ f ≫ ε_ Y Y' ▷ Z) ⊗≫ 𝟙 _ := by
      dsimp [tensorLeftHomEquiv]; monoidal
    _ = (Y' ◁ η_ Y Y' ⊗≫ ε_ Y Y' ▷ Y') ⊗≫ f := by
      rw [whisker_exchange]; monoidal
    _ = _ := by rw [coevaluation_evaluation'']; monoidal


@[simp]
theorem tensorLeftHomEquiv_symm_coevaluation_comp_whiskerRight {X Y : C} [HasRightDual X]
    [HasRightDual Y] (f : X ⟶ Y) :
    (tensorLeftHomEquiv _ _ _ _).symm (η_ _ _ ≫ f ▷ (Xᘁ)) = (ρ_ _).hom ≫ fᘁ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasRightDual X
    inst✝ : CategoryTheory.HasRightDual Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.tensorLeftHomEquiv CategoryTheory.MonoidalCategoryStruct …
  -/
  dsimp [tensorLeftHomEquiv, rightAdjointMate]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasRightDual X
    inst✝ : CategoryTheory.HasRightDual Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem tensorRightHomEquiv_symm_coevaluation_comp_whiskerLeft {X Y : C} [HasLeftDual X]
    [HasLeftDual Y] (f : X ⟶ Y) :
    (tensorRightHomEquiv _ (ᘁY) _ _).symm (η_ (ᘁX : C) X ≫ (ᘁX : C) ◁ f) = (λ_ _).hom ≫ ᘁf := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasLeftDual X
    inst✝ : CategoryTheory.HasLeftDual Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.tensorRightHomEquiv CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [tensorRightHomEquiv, leftAdjointMate]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasLeftDual X
    inst✝ : CategoryTheory.HasLeftDual Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem tensorRightHomEquiv_symm_coevaluation_comp_whiskerRight {Y Y' Z : C} [ExactPairing Y Y']
    (f : Y ⟶ Z) : (tensorRightHomEquiv _ Y _ _).symm (η_ Y Y' ≫ f ▷ Y') = (λ_ _).hom ≫ f :=
  calc
    _ = η_ Y Y' ▷ Y ⊗≫ (f ▷ (Y' ⊗ Y) ≫ Z ◁ ε_ Y Y') ⊗≫ 𝟙 _ := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        Y Y' Z : C
        inst✝ : CategoryTheory.ExactPairing Y Y'
        f : Quiver.Hom Y Z
        ⊢ Eq ((CategoryTheory.tensorRightHomEquiv CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp [tensorRightHomEquiv]; monoidal
                                   /-
                                     🎉 no goals
                                   -/
    _ = (η_ Y Y' ▷ Y ⊗≫ Y ◁ ε_ Y Y') ⊗≫ f := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        Y Y' Z : C
        inst✝ : CategoryTheory.ExactPairing Y Y'
        f : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.MonoidalCategoryStruct.whisk …
      -/
      rw [← whisker_exchange]; monoidal
                               /-
                                 🎉 no goals
                               -/
    _ = _ := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        Y Y' Z : C
        inst✝ : CategoryTheory.ExactPairing Y Y'
        f : Quiver.Hom Y Z
        ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.monoidalComp (CategoryTheory …
      -/
      rw [evaluation_coevaluation'']; monoidal
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem tensorLeftHomEquiv_whiskerLeft_comp_evaluation {Y Z : C} [HasLeftDual Z] (f : Y ⟶ ᘁZ) :
    (tensorLeftHomEquiv _ _ _ _) (Z ◁ f ≫ ε_ _ _) = f ≫ (ρ_ _).inv :=
  calc
    _ = 𝟙 _ ⊗≫ (η_ (ᘁZ : C) Z ▷ Y ≫ ((ᘁZ) ⊗ Z) ◁ f) ⊗≫ (ᘁZ) ◁ ε_ (ᘁZ) Z := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        Y Z : C
        inst✝ : CategoryTheory.HasLeftDual Z
        f : Quiver.Hom Y (CategoryTheory.HasLeftDual.leftDual Z)
        ⊢ Eq ((CategoryTheory.tensorLeftHomEquiv Y (CategoryTheory.HasLeftDual.leftDua …
      -/
      dsimp [tensorLeftHomEquiv]; monoidal
                                  /-
                                    🎉 no goals
                                  -/
    _ = f ⊗≫ (η_ (ᘁZ) Z ▷ (ᘁZ) ⊗≫ (ᘁZ) ◁ ε_ (ᘁZ) Z) := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        Y Z : C
        inst✝ : CategoryTheory.HasLeftDual Z
        f : Quiver.Hom Y (CategoryTheory.HasLeftDual.leftDual Z)
        ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id Y) (Catego …
      -/
      rw [← whisker_exchange]; monoidal
                               /-
                                 🎉 no goals
                               -/
    _ = _ := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        Y Z : C
        inst✝ : CategoryTheory.HasLeftDual Z
        f : Quiver.Hom Y (CategoryTheory.HasLeftDual.leftDual Z)
        ⊢ Eq (CategoryTheory.monoidalComp f (CategoryTheory.monoidalComp (CategoryTheo …
      -/
      rw [evaluation_coevaluation'']; monoidal
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem tensorLeftHomEquiv_whiskerRight_comp_evaluation {X Y : C} [HasLeftDual X] [HasLeftDual Y]
    (f : X ⟶ Y) : (tensorLeftHomEquiv _ _ _ _) (f ▷ _ ≫ ε_ _ _) = (ᘁf) ≫ (ρ_ _).inv := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasLeftDual X
    inst✝ : CategoryTheory.HasLeftDual Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.tensorLeftHomEquiv (CategoryTheory.HasLeftDual.leftDual  …
  -/
  dsimp [tensorLeftHomEquiv, leftAdjointMate]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasLeftDual X
    inst✝ : CategoryTheory.HasLeftDual Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem tensorRightHomEquiv_whiskerLeft_comp_evaluation {X Y : C} [HasRightDual X] [HasRightDual Y]
    (f : X ⟶ Y) : (tensorRightHomEquiv _ _ _ _) ((Yᘁ : C) ◁ f ≫ ε_ _ _) = fᘁ ≫ (λ_ _).inv := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasRightDual X
    inst✝ : CategoryTheory.HasRightDual Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.tensorRightHomEquiv (CategoryTheory.HasRightDual.rightDu …
  -/
  dsimp [tensorRightHomEquiv, rightAdjointMate]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasRightDual X
    inst✝ : CategoryTheory.HasRightDual Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem tensorRightHomEquiv_whiskerRight_comp_evaluation {X Y : C} [HasRightDual X] (f : Y ⟶ Xᘁ) :
    (tensorRightHomEquiv _ _ _ _) (f ▷ X ≫ ε_ X (Xᘁ)) = f ≫ (λ_ _).inv :=
  calc
    _ = 𝟙 _ ⊗≫ (Y ◁ η_ X Xᘁ ≫ f ▷ (X ⊗ Xᘁ)) ⊗≫ ε_ X Xᘁ ▷ Xᘁ := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        X Y : C
        inst✝ : CategoryTheory.HasRightDual X
        f : Quiver.Hom Y (CategoryTheory.HasRightDual.rightDual X)
        ⊢ Eq ((CategoryTheory.tensorRightHomEquiv Y X (CategoryTheory.HasRightDual.rig …
      -/
      dsimp [tensorRightHomEquiv]; monoidal
                                   /-
                                     🎉 no goals
                                   -/
    _ = f ⊗≫ (Xᘁ ◁ η_ X Xᘁ ⊗≫ ε_ X Xᘁ ▷ Xᘁ) := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        X Y : C
        inst✝ : CategoryTheory.HasRightDual X
        f : Quiver.Hom Y (CategoryTheory.HasRightDual.rightDual X)
        ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id Y) (Catego …
      -/
      rw [whisker_exchange]; monoidal
                             /-
                               🎉 no goals
                             -/
    _ = _ := by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.MonoidalCategory C
        X Y : C
        inst✝ : CategoryTheory.HasRightDual X
        f : Quiver.Hom Y (CategoryTheory.HasRightDual.rightDual X)
        ⊢ Eq (CategoryTheory.monoidalComp f (CategoryTheory.monoidalComp (CategoryTheo …
      -/
      rw [coevaluation_evaluation'']; monoidal
                                      /-
                                        🎉 no goals
                                      -/

-- Next four lemmas passing `fᘁ` or `ᘁf` through (co)evaluations.

@[reassoc]
theorem coevaluation_comp_rightAdjointMate {X Y : C} [HasRightDual X] [HasRightDual Y] (f : X ⟶ Y) :
    η_ Y (Yᘁ) ≫ _ ◁ (fᘁ) = η_ _ _ ≫ f ▷ _ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasRightDual X
    inst✝ : CategoryTheory.HasRightDual Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ExactPairing.coevalua …
  -/
  apply_fun (tensorLeftHomEquiv _ Y (Yᘁ) _).symm
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasRightDual X
    inst✝ : CategoryTheory.HasRightDual Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.tensorLeftHomEquiv CategoryTheory.MonoidalCategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem leftAdjointMate_comp_evaluation {X Y : C} [HasLeftDual X] [HasLeftDual Y] (f : X ⟶ Y) :
    X ◁ (ᘁf) ≫ ε_ _ _ = f ▷ _ ≫ ε_ _ _ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasLeftDual X
    inst✝ : CategoryTheory.HasLeftDual Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply_fun tensorLeftHomEquiv _ (ᘁX) X _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasLeftDual X
    inst✝ : CategoryTheory.HasLeftDual Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.tensorLeftHomEquiv (CategoryTheory.HasLeftDual.leftDual  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem coevaluation_comp_leftAdjointMate {X Y : C} [HasLeftDual X] [HasLeftDual Y] (f : X ⟶ Y) :
    η_ (ᘁY) Y ≫ (ᘁf) ▷ Y = η_ (ᘁX) X ≫ (ᘁX) ◁ f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasLeftDual X
    inst✝ : CategoryTheory.HasLeftDual Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ExactPairing.coevalua …
  -/
  apply_fun (tensorRightHomEquiv _ (ᘁY) Y _).symm
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasLeftDual X
    inst✝ : CategoryTheory.HasLeftDual Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.tensorRightHomEquiv CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem rightAdjointMate_comp_evaluation {X Y : C} [HasRightDual X] [HasRightDual Y] (f : X ⟶ Y) :
    (fᘁ ▷ X) ≫ ε_ X (Xᘁ) = ((Yᘁ) ◁ f) ≫ ε_ Y (Yᘁ) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasRightDual X
    inst✝ : CategoryTheory.HasRightDual Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply_fun tensorRightHomEquiv _ X (Xᘁ) _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X Y : C
    inst✝¹ : CategoryTheory.HasRightDual X
    inst✝ : CategoryTheory.HasRightDual Y
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.tensorRightHomEquiv (CategoryTheory.HasRightDual.rightDu …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Transport an exact pairing across an isomorphism in the first argument. -/
def exactPairingCongrLeft {X X' Y : C} [ExactPairing X' Y] (i : X ≅ X') : ExactPairing X Y where
  evaluation' := Y ◁ i.hom ≫ ε_ _ _
  coevaluation' := η_ _ _ ≫ i.inv ▷ Y
  evaluation_coevaluation' :=
    calc
      _ = η_ X' Y ▷ X ⊗≫ (i.inv ▷ (Y ⊗ X) ≫ X ◁ (Y ◁ i.hom)) ⊗≫ X ◁ ε_ X' Y := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X X' Y : C
          inst✝ : CategoryTheory.ExactPairing X' Y
          i : CategoryTheory.Iso X X'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        monoidal
        /-
          🎉 no goals
        -/
      _ = 𝟙 _ ⊗≫ (η_ X' Y ▷ X ≫ (X' ⊗ Y) ◁ i.hom) ⊗≫
          (i.inv ▷ (Y ⊗ X') ≫ X ◁ ε_ X' Y) ⊗≫ 𝟙 _ := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X X' Y : C
          inst✝ : CategoryTheory.ExactPairing X' Y
          i : CategoryTheory.Iso X X'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.MonoidalCategoryStruct.whisk …
        -/
        rw [← whisker_exchange]; monoidal
                                 /-
                                   🎉 no goals
                                 -/
      _ = 𝟙 _ ⊗≫ i.hom ⊗≫ (η_ X' Y ▷ X' ⊗≫ X' ◁ ε_ X' Y) ⊗≫ i.inv ⊗≫ 𝟙 _ := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X X' Y : C
          inst✝ : CategoryTheory.ExactPairing X' Y
          i : CategoryTheory.Iso X X'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
        -/
        rw [← whisker_exchange, ← whisker_exchange]; monoidal
                                                     /-
                                                       🎉 no goals
                                                     -/
      _ = 𝟙 _ ⊗≫ (i.hom ≫ i.inv) ⊗≫ 𝟙 _ := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X X' Y : C
          inst✝ : CategoryTheory.ExactPairing X' Y
          i : CategoryTheory.Iso X X'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
        -/
        rw [evaluation_coevaluation'']; monoidal
                                        /-
                                          🎉 no goals
                                        -/
      _ = (λ_ X).hom ≫ (ρ_ X).inv := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X X' Y : C
          inst✝ : CategoryTheory.ExactPairing X' Y
          i : CategoryTheory.Iso X X'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
        -/
        rw [Iso.hom_inv_id]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X X' Y : C
          inst✝ : CategoryTheory.ExactPairing X' Y
          i : CategoryTheory.Iso X X'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
        -/
        monoidal
        /-
          🎉 no goals
        -/
  coevaluation_evaluation' := by
    calc
      _ = Y ◁ η_ X' Y ≫ Y ◁ (i.inv ≫ i.hom) ▷ Y ⊗≫ ε_ X' Y ▷ Y := by
        monoidal
      _ = Y ◁ η_ X' Y ⊗≫ ε_ X' Y ▷ Y := by
        rw [Iso.inv_hom_id]; monoidal
      _ = _ := by
        rw [coevaluation_evaluation'']
        monoidal


/-- Transport an exact pairing across an isomorphism in the second argument. -/
def exactPairingCongrRight {X Y Y' : C} [ExactPairing X Y'] (i : Y ≅ Y') : ExactPairing X Y where
  evaluation' := i.hom ▷ X ≫ ε_ _ _
  coevaluation' := η_ _ _ ≫ X ◁ i.inv
  evaluation_coevaluation' := by
    calc
      _ = η_ X Y' ▷ X ⊗≫ X ◁ (i.inv ≫ i.hom) ▷ X ≫ X ◁ ε_ X Y' := by
        monoidal
      _ = η_ X Y' ▷ X ⊗≫ X ◁ ε_ X Y' := by
        rw [Iso.inv_hom_id]; monoidal
      _ = _ := by
        rw [evaluation_coevaluation'']
        monoidal
  coevaluation_evaluation' :=
    calc
      _ = Y ◁ η_ X Y' ⊗≫ (Y ◁ (X ◁ i.inv) ≫ i.hom ▷ (X ⊗ Y)) ⊗≫ ε_ X Y' ▷ Y := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X Y Y' : C
          inst✝ : CategoryTheory.ExactPairing X Y'
          i : CategoryTheory.Iso Y Y'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        monoidal
        /-
          🎉 no goals
        -/
      _ = 𝟙 _ ⊗≫ (Y ◁ η_ X Y' ≫ i.hom ▷ (X ⊗ Y')) ⊗≫
          ((Y' ⊗ X) ◁ i.inv ≫ ε_ X Y' ▷ Y) ⊗≫ 𝟙 _ := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X Y Y' : C
          inst✝ : CategoryTheory.ExactPairing X Y'
          i : CategoryTheory.Iso Y Y'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.MonoidalCategoryStruct.whisk …
        -/
        rw [whisker_exchange]; monoidal
                               /-
                                 🎉 no goals
                               -/
      _ = 𝟙 _ ⊗≫ i.hom ⊗≫ (Y' ◁ η_ X Y' ⊗≫ ε_ X Y' ▷ Y') ⊗≫ i.inv ⊗≫ 𝟙 _ := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X Y Y' : C
          inst✝ : CategoryTheory.ExactPairing X Y'
          i : CategoryTheory.Iso Y Y'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
        -/
        rw [whisker_exchange, whisker_exchange]; monoidal
                                                 /-
                                                   🎉 no goals
                                                 -/
      _ = 𝟙 _ ⊗≫ (i.hom ≫ i.inv) ⊗≫ 𝟙 _ := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X Y Y' : C
          inst✝ : CategoryTheory.ExactPairing X Y'
          i : CategoryTheory.Iso Y Y'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
        -/
        rw [coevaluation_evaluation'']; monoidal
                                        /-
                                          🎉 no goals
                                        -/
      _ = (ρ_ Y).hom ≫ (λ_ Y).inv := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X Y Y' : C
          inst✝ : CategoryTheory.ExactPairing X Y'
          i : CategoryTheory.Iso Y Y'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
        -/
        rw [Iso.hom_inv_id]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.MonoidalCategory C
          X Y Y' : C
          inst✝ : CategoryTheory.ExactPairing X Y'
          i : CategoryTheory.Iso Y Y'
          ⊢ Eq (CategoryTheory.monoidalComp (CategoryTheory.CategoryStruct.id (CategoryT …
        -/
        monoidal
        /-
          🎉 no goals
        -/


/-- Transport an exact pairing across isomorphisms. -/
def exactPairingCongr {X X' Y Y' : C} [ExactPairing X' Y'] (i : X ≅ X') (j : Y ≅ Y') :
    ExactPairing X Y :=
  haveI : ExactPairing X' Y := exactPairingCongrRight j
  exactPairingCongrLeft i


/-- Right duals are isomorphic. -/
def rightDualIso {X Y₁ Y₂ : C} (p₁ : ExactPairing X Y₁) (p₂ : ExactPairing X Y₂) : Y₁ ≅ Y₂ where
  hom := @rightAdjointMate C _ _ X X ⟨Y₂⟩ ⟨Y₁⟩ (𝟙 X)
  inv := @rightAdjointMate C _ _ X X ⟨Y₁⟩ ⟨Y₂⟩ (𝟙 X)
  -- Porting note: no implicit arguments were required below:
  hom_inv_id := by
    rw [← @comp_rightAdjointMate C _ _ X X X ⟨Y₁⟩ ⟨Y₂⟩ ⟨Y₁⟩, Category.comp_id,
      @rightAdjointMate_id _ _ _ _ ⟨Y₁⟩]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.MonoidalCategory C
      X Y₁ Y₂ : C
      p₁ : CategoryTheory.ExactPairing X Y₁
      p₂ : CategoryTheory.ExactPairing X Y₂
      ⊢ Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.HasRightDual.rightDual  …
    -/
    rfl
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    rw [← @comp_rightAdjointMate C _ _ X X X ⟨Y₂⟩ ⟨Y₁⟩ ⟨Y₂⟩, Category.comp_id,
      @rightAdjointMate_id _ _ _ _ ⟨Y₂⟩]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.MonoidalCategory C
      X Y₁ Y₂ : C
      p₁ : CategoryTheory.ExactPairing X Y₁
      p₂ : CategoryTheory.ExactPairing X Y₂
      ⊢ Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.HasRightDual.rightDual  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Left duals are isomorphic. -/
def leftDualIso {X₁ X₂ Y : C} (p₁ : ExactPairing X₁ Y) (p₂ : ExactPairing X₂ Y) : X₁ ≅ X₂ where
  hom := @leftAdjointMate C _ _ Y Y ⟨X₂⟩ ⟨X₁⟩ (𝟙 Y)
  inv := @leftAdjointMate C _ _ Y Y ⟨X₁⟩ ⟨X₂⟩ (𝟙 Y)
  -- Porting note: no implicit arguments were required below:
  hom_inv_id := by
    rw [← @comp_leftAdjointMate C _ _ Y Y Y ⟨X₁⟩ ⟨X₂⟩ ⟨X₁⟩, Category.comp_id,
      @leftAdjointMate_id _ _ _ _ ⟨X₁⟩]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.MonoidalCategory C
      X₁ X₂ Y : C
      p₁ : CategoryTheory.ExactPairing X₁ Y
      p₂ : CategoryTheory.ExactPairing X₂ Y
      ⊢ Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.HasLeftDual.leftDual Y) …
    -/
    rfl
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    rw [← @comp_leftAdjointMate C _ _ Y Y Y ⟨X₂⟩ ⟨X₁⟩ ⟨X₂⟩, Category.comp_id,
      @leftAdjointMate_id _ _ _ _ ⟨X₂⟩]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.MonoidalCategory C
      X₁ X₂ Y : C
      p₁ : CategoryTheory.ExactPairing X₁ Y
      p₂ : CategoryTheory.ExactPairing X₂ Y
      ⊢ Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.HasLeftDual.leftDual Y) …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem rightDualIso_id {X Y : C} (p : ExactPairing X Y) : rightDualIso p p = Iso.refl Y := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    p : CategoryTheory.ExactPairing X Y
    ⊢ Eq (CategoryTheory.rightDualIso p p) (CategoryTheory.Iso.refl Y)
  -/
  ext
  /-
    case w
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    p : CategoryTheory.ExactPairing X Y
    ⊢ Eq (CategoryTheory.rightDualIso p p).hom (CategoryTheory.Iso.refl Y).hom
  -/
  simp only [rightDualIso, Iso.refl_hom, @rightAdjointMate_id _ _ _ _ ⟨Y⟩]
  /-
    🎉 no goals
  -/


@[simp]
theorem leftDualIso_id {X Y : C} (p : ExactPairing X Y) : leftDualIso p p = Iso.refl X := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    p : CategoryTheory.ExactPairing X Y
    ⊢ Eq (CategoryTheory.leftDualIso p p) (CategoryTheory.Iso.refl X)
  -/
  ext
  /-
    case w
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    p : CategoryTheory.ExactPairing X Y
    ⊢ Eq (CategoryTheory.leftDualIso p p).hom (CategoryTheory.Iso.refl X).hom
  -/
  simp only [leftDualIso, Iso.refl_hom, @leftAdjointMate_id _ _ _ _ ⟨X⟩]
  /-
    🎉 no goals
  -/


/-- A right rigid monoidal category is one in which every object has a right dual. -/
class RightRigidCategory (C : Type u) [Category.{v} C] [MonoidalCategory.{v} C] where
  [rightDual : ∀ X : C, HasRightDual X]


/-- A left rigid monoidal category is one in which every object has a right dual. -/
class LeftRigidCategory (C : Type u) [Category.{v} C] [MonoidalCategory.{v} C] where
  [leftDual : ∀ X : C, HasLeftDual X]


/-- Any left rigid category is monoidal closed, with the internal hom `X ⟶[C] Y = ᘁX ⊗ Y`.
This has to be a definition rather than an instance to avoid diamonds, for example between
`category_theory.monoidal_closed.functor_category` and
`CategoryTheory.Monoidal.leftRigidFunctorCategory`. Moreover, in concrete applications there is
often a more useful definition of the internal hom object than `ᘁY ⊗ X`, in which case the monoidal
closed structure shouldn't come the rigid structure (e.g. in the category `FinVect k`, it is more
convenient to define the internal hom as `Y →ₗ[k] X` rather than `ᘁY ⊗ X` even though these are
naturally isomorphic). -/
def monoidalClosedOfLeftRigidCategory (C : Type u) [Category.{v} C] [MonoidalCategory.{v} C]
    [LeftRigidCategory C] : MonoidalClosed C where
  closed X := closedOfHasLeftDual X


/-- A rigid monoidal category is a monoidal category which is left rigid and right rigid. -/
class RigidCategory (C : Type u) [Category.{v} C] [MonoidalCategory.{v} C] extends
    RightRigidCategory C, LeftRigidCategory C


