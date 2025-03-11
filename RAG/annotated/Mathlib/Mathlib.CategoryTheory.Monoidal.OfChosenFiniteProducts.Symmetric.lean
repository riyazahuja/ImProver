theorem braiding_naturality {X X' Y Y' : C} (f : X ⟶ Y) (g : X' ⟶ Y') :
    tensorHom ℬ f g ≫ (Limits.BinaryFan.braiding (ℬ Y Y').isLimit (ℬ Y' Y).isLimit).hom =
      (Limits.BinaryFan.braiding (ℬ X X').isLimit (ℬ X' X).isLimit).hom ≫ tensorHom ℬ g f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X X' Y Y' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X' Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalOfChosenFinit …
  -/
  dsimp [tensorHom, Limits.BinaryFan.braiding]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X X' Y Y' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X' Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ Y Y').isLimit.lift (CategoryTheor …
  -/
  apply (ℬ _ _).isLimit.hom_ext
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X X' Y Y' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X' Y'
    ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  rintro ⟨⟨⟩⟩ <;> · dsimp [Limits.IsLimit.conePointUniqueUpToIso]; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem hexagon_forward (X Y Z : C) :
    (BinaryFan.associatorOfLimitCone ℬ X Y Z).hom ≫
        (Limits.BinaryFan.braiding (ℬ X (tensorObj ℬ Y Z)).isLimit
              (ℬ (tensorObj ℬ Y Z) X).isLimit).hom ≫
          (BinaryFan.associatorOfLimitCone ℬ Y Z X).hom =
      tensorHom ℬ (Limits.BinaryFan.braiding (ℬ X Y).isLimit (ℬ Y X).isLimit).hom (𝟙 Z) ≫
        (BinaryFan.associatorOfLimitCone ℬ Y X Z).hom ≫
          tensorHom ℬ (𝟙 Y) (Limits.BinaryFan.braiding (ℬ X Z).isLimit (ℬ Z X).isLimit).hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.asso …
  -/
  dsimp [tensorHom, Limits.BinaryFan.braiding]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.asso …
  -/
  apply (ℬ _ _).isLimit.hom_ext; rintro ⟨⟨⟩⟩
    /-
      case mk.left
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · dsimp [Limits.IsLimit.conePointUniqueUpToIso]; simp
                                                   /-
                                                     🎉 no goals
                                                   -/
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · apply (ℬ _ _).isLimit.hom_ext
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X Y Z : C
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    rintro ⟨⟨⟩⟩ <;> · dsimp [Limits.IsLimit.conePointUniqueUpToIso]; simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem hexagon_reverse (X Y Z : C) :
    (BinaryFan.associatorOfLimitCone ℬ X Y Z).inv ≫
        (Limits.BinaryFan.braiding (ℬ (tensorObj ℬ X Y) Z).isLimit
              (ℬ Z (tensorObj ℬ X Y)).isLimit).hom ≫
          (BinaryFan.associatorOfLimitCone ℬ Z X Y).inv =
      tensorHom ℬ (𝟙 X) (Limits.BinaryFan.braiding (ℬ Y Z).isLimit (ℬ Z Y).isLimit).hom ≫
        (BinaryFan.associatorOfLimitCone ℬ X Z Y).inv ≫
          tensorHom ℬ (Limits.BinaryFan.braiding (ℬ X Z).isLimit (ℬ Z X).isLimit).hom (𝟙 Y) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.asso …
  -/
  dsimp [tensorHom, Limits.BinaryFan.braiding]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.asso …
  -/
  apply (ℬ _ _).isLimit.hom_ext; rintro ⟨⟨⟩⟩
    /-
      case mk.left
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · apply (ℬ _ _).isLimit.hom_ext
    /-
      case mk.left
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X Y Z : C
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨⟨⟩⟩ <;>
      · dsimp [BinaryFan.associatorOfLimitCone, BinaryFan.associator,
          Limits.IsLimit.conePointUniqueUpToIso]
        /-
          case mk.left.mk.left
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
          X Y Z : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        /-
          🎉 no goals
        -/
        simp
        /-
          🎉 no goals
        -/
  · dsimp [BinaryFan.associatorOfLimitCone, BinaryFan.associator,
      Limits.IsLimit.conePointUniqueUpToIso]
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem symmetry (X Y : C) :
    (Limits.BinaryFan.braiding (ℬ X Y).isLimit (ℬ Y X).isLimit).hom ≫
        (Limits.BinaryFan.braiding (ℬ Y X).isLimit (ℬ X Y).isLimit).hom =
      𝟙 (tensorObj ℬ X Y) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.brai …
  -/
  dsimp [tensorHom, Limits.BinaryFan.braiding]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ X Y).isLimit.conePointUniqueUpToI …
  -/
  apply (ℬ _ _).isLimit.hom_ext
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y : C
    ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  rintro ⟨⟨⟩⟩ <;> · dsimp [Limits.IsLimit.conePointUniqueUpToIso]; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The monoidal structure coming from finite products is symmetric.
-/
def symmetricOfChosenFiniteProducts :
    SymmetricCategory (MonoidalOfChosenFiniteProductsSynonym 𝒯 ℬ) where
  braiding _ _ := Limits.BinaryFan.braiding (ℬ _ _).isLimit (ℬ _ _).isLimit
  braiding_naturality_left f X := braiding_naturality ℬ f (𝟙 X)
  braiding_naturality_right X _ _ f := braiding_naturality ℬ (𝟙 X) f
  hexagon_forward X Y Z := hexagon_forward ℬ X Y Z
  hexagon_reverse X Y Z := hexagon_reverse ℬ X Y Z
  symmetry X Y := symmetry ℬ X Y


