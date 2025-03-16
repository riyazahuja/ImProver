@[reassoc]
theorem leftUnitor_tensor'' (X Y : C) :
    (α_ (𝟙_ C) X Y).hom ≫ (λ_ (X ⊗ Y)).hom = (λ_ X).hom ⊗ 𝟙 Y := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


@[reassoc]
theorem leftUnitor_tensor' (X Y : C) :
    (λ_ (X ⊗ Y)).hom = (α_ (𝟙_ C) X Y).inv ≫ ((λ_ X).hom ⊗ 𝟙 Y) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


@[reassoc]
theorem leftUnitor_tensor_inv' (X Y : C) :
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                                                        inst✝ : CategoryTheory.MonoidalCategory C
                                                                        X Y : C
                                                                        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
                                                                      -/
    (λ_ (X ⊗ Y)).inv = ((λ_ X).inv ⊗ 𝟙 Y) ≫ (α_ (𝟙_ C) X Y).hom := by monoidal_coherence
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[reassoc]
theorem id_tensor_rightUnitor_inv (X Y : C) : 𝟙 X ⊗ (ρ_ Y).inv = (ρ_ _).inv ≫ (α_ _ _ _).hom := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


@[reassoc]
theorem leftUnitor_inv_tensor_id (X Y : C) : (λ_ X).inv ⊗ 𝟙 Y = (λ_ _).inv ≫ (α_ _ _ _).inv := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


@[reassoc]
theorem pentagon_inv_inv_hom (W X Y Z : C) :
    (α_ W (X ⊗ Y) Z).inv ≫ ((α_ W X Y).inv ⊗ 𝟙 Z) ≫ (α_ (W ⊗ X) Y Z).hom =
      (𝟙 W ⊗ (α_ X Y Z).hom) ≫ (α_ W X (Y ⊗ Z)).inv := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


theorem unitors_equal : (λ_ (𝟙_ C)).hom = (ρ_ (𝟙_ C)).hom := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor CategoryTheory.Monoidal …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


theorem unitors_inv_equal : (λ_ (𝟙_ C)).inv = (ρ_ (𝟙_ C)).inv := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor CategoryTheory.Monoidal …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


@[reassoc]
theorem pentagon_hom_inv {W X Y Z : C} :
    (α_ W X (Y ⊗ Z)).hom ≫ (𝟙 W ⊗ (α_ X Y Z).inv) =
      (α_ (W ⊗ X) Y Z).inv ≫ ((α_ W X Y).hom ⊗ 𝟙 Z) ≫ (α_ W (X ⊗ Y) Z).hom := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


@[reassoc]
theorem pentagon_inv_hom (W X Y Z : C) :
    (α_ (W ⊗ X) Y Z).inv ≫ ((α_ W X Y).hom ⊗ 𝟙 Z) =
      (α_ W X (Y ⊗ Z)).hom ≫ (𝟙 W ⊗ (α_ X Y Z).inv) ≫ (α_ W (X ⊗ Y) Z).inv := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal_coherence
  /-
    🎉 no goals
  -/


