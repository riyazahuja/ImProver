@[simp]
theorem map_id (F : C × D ⥤ E) (X : C) (Y : D) :
    F.map ((𝟙 X, 𝟙 Y) : (X, Y) ⟶ (X, Y)) = 𝟙 (F.obj (X, Y)) :=
  F.map_id (X, Y)


@[simp]
theorem map_id_comp (F : C × D ⥤ E) (W : C) {X Y Z : D} (f : X ⟶ Y) (g : Y ⟶ Z) :
    F.map ((𝟙 W, f ≫ g) : (W, X) ⟶ (W, Z)) =
      F.map ((𝟙 W, f) : (W, X) ⟶ (W, Y)) ≫ F.map ((𝟙 W, g) : (W, Y) ⟶ (W, Z)) := by
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor (Prod C D) E
    W : C
    X Y Z : D
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (F.map { fst := CategoryTheory.CategoryStruct.id W, snd := CategoryTheory …
  -/
  rw [← Functor.map_comp, prod_comp, Category.comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp_id (F : C × D ⥤ E) (X Y Z : C) (W : D) (f : X ⟶ Y) (g : Y ⟶ Z) :
    F.map ((f ≫ g, 𝟙 W) : (X, W) ⟶ (Z, W)) =
      F.map ((f, 𝟙 W) : (X, W) ⟶ (Y, W)) ≫ F.map ((g, 𝟙 W) : (Y, W) ⟶ (Z, W)) := by
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor (Prod C D) E
    X Y Z : C
    W : D
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (F.map { fst := CategoryTheory.CategoryStruct.comp f g, snd := CategoryTh …
  -/
  rw [← Functor.map_comp, prod_comp, Category.comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem diagonal (F : C × D ⥤ E) (X X' : C) (f : X ⟶ X') (Y Y' : D) (g : Y ⟶ Y') :
    F.map ((𝟙 X, g) : (X, Y) ⟶ (X, Y')) ≫ F.map ((f, 𝟙 Y') : (X, Y') ⟶ (X', Y')) =
      F.map ((f, g) : (X, Y) ⟶ (X', Y')) := by
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor (Prod C D) E
    X X' : C
    f : Quiver.Hom X X'
    Y Y' : D
    g : Quiver.Hom Y Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { fst := CategoryTheory.Catego …
  -/
  rw [← Functor.map_comp, prod_comp, Category.id_comp, Category.comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem diagonal' (F : C × D ⥤ E) (X X' : C) (f : X ⟶ X') (Y Y' : D) (g : Y ⟶ Y') :
    F.map ((f, 𝟙 Y) : (X, Y) ⟶ (X', Y)) ≫ F.map ((𝟙 X', g) : (X', Y) ⟶ (X', Y')) =
      F.map ((f, g) : (X, Y) ⟶ (X', Y')) := by
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor (Prod C D) E
    X X' : C
    f : Quiver.Hom X X'
    Y Y' : D
    g : Quiver.Hom Y Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { fst := f, snd := CategoryThe …
  -/
  rw [← Functor.map_comp, prod_comp, Category.id_comp, Category.comp_id]
  /-
    🎉 no goals
  -/


