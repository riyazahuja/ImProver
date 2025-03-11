/-- A functor `F : C ⥤ D` between monoidal categories is lax monoidal if it is
equipped with morphisms `ε : 𝟙 _D ⟶ F.obj (𝟙_ C)` and `μ X Y : F.obj X ⊗ F.obj Y ⟶ F.obj (X ⊗ Y)`,
satisfying the appropriate coherences. -/
class LaxMonoidal where
  /-- unit morphism -/
  ε' : 𝟙_ D ⟶ F.obj (𝟙_ C)
  /-- tensorator -/
  μ' : ∀ X Y : C, F.obj X ⊗ F.obj Y ⟶ F.obj (X ⊗ Y)
  μ'_natural_left :
    ∀ {X Y : C} (f : X ⟶ Y) (X' : C),
      F.map f ▷ F.obj X' ≫ μ' Y X' = μ' X X' ≫ F.map (f ▷ X') := by
    aesop_cat
  μ'_natural_right :
    ∀ {X Y : C} (X' : C) (f : X ⟶ Y) ,
      F.obj X' ◁ F.map f ≫ μ' X' Y = μ' X' X ≫ F.map (X' ◁ f) := by
    aesop_cat
  /-- associativity of the tensorator -/
  associativity' :
    ∀ X Y Z : C,
      μ' X Y ▷ F.obj Z ≫ μ' (X ⊗ Y) Z ≫ F.map (α_ X Y Z).hom =
        (α_ (F.obj X) (F.obj Y) (F.obj Z)).hom ≫ F.obj X ◁ μ' Y Z ≫ μ' X (Y ⊗ Z) := by
    aesop_cat
  -- unitality
  left_unitality' :
    ∀ X : C, (λ_ (F.obj X)).hom = ε' ▷ F.obj X ≫ μ' (𝟙_ C) X ≫ F.map (λ_ X).hom := by
      aesop_cat
  right_unitality' :
    ∀ X : C, (ρ_ (F.obj X)).hom = F.obj X ◁ ε' ≫ μ' X (𝟙_ C) ≫ F.map (ρ_ X).hom := by
    aesop_cat


/-- the unit morphism of a lax monoidal functor-/
def ε : 𝟙_ D ⟶ F.obj (𝟙_ C) := ε'


/-- the tensorator of a lax monoidal functor -/
def μ (X Y : C) : F.obj X ⊗ F.obj Y ⟶ F.obj (X ⊗ Y) := μ' X Y


@[reassoc (attr := simp)]
lemma μ_natural_left {X Y : C} (f : X ⟶ Y) (X' : C) :
    F.map f ▷ F.obj X' ≫ μ F Y X' = μ F X X' ≫ F.map (f ▷ X') := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.LaxMonoidal
    X Y : C
    f : Quiver.Hom X Y
    X' : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply μ'_natural_left
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma μ_natural_right {X Y : C} (X' : C) (f : X ⟶ Y) :
    F.obj X' ◁ F.map f ≫ μ F X' Y = μ F X' X ≫ F.map (X' ◁ f) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.LaxMonoidal
    X Y X' : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply μ'_natural_right
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma associativity (X Y Z : C) :
    μ F X Y ▷ F.obj Z ≫ μ F (X ⊗ Y) Z ≫ F.map (α_ X Y Z).hom =
        (α_ (F.obj X) (F.obj Y) (F.obj Z)).hom ≫ F.obj X ◁ μ F Y Z ≫ μ F X (Y ⊗ Z) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.LaxMonoidal
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply associativity'
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma left_unitality (X : C) :
    (λ_ (F.obj X)).hom = ε F ▷ F.obj X ≫ μ F (𝟙_ C) X ≫ F.map (λ_ X).hom := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.LaxMonoidal
    X : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (F.obj X)).hom (Categor …
  -/
  apply left_unitality'
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma right_unitality (X : C) :
    (ρ_ (F.obj X)).hom = F.obj X ◁ ε F ≫ μ F X (𝟙_ C) ≫ F.map (ρ_ X).hom := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.LaxMonoidal
    X : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (F.obj X)).hom (Catego …
  -/
  apply right_unitality'
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem μ_natural {X Y X' Y' : C} (f : X ⟶ Y) (g : X' ⟶ Y') :
    (F.map f ⊗ F.map g) ≫ μ F Y Y' = μ F X X' ≫ F.map (f ⊗ g) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.LaxMonoidal
    X Y X' Y' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X' Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem left_unitality_inv (X : C) :
    (λ_ (F.obj X)).inv ≫ ε F ▷ F.obj X ≫ μ F (𝟙_ C) X = F.map (λ_ X).inv := by
  rw [Iso.inv_comp_eq, left_unitality, Category.assoc, Category.assoc, ← F.map_comp,
    Iso.hom_inv_id, F.map_id, comp_id]


@[reassoc (attr := simp)]
theorem right_unitality_inv (X : C) :
    (ρ_ (F.obj X)).inv ≫ F.obj X ◁ ε F ≫ μ F X (𝟙_ C) = F.map (ρ_ X).inv := by
  rw [Iso.inv_comp_eq, right_unitality, Category.assoc, Category.assoc, ← F.map_comp,
    Iso.hom_inv_id, F.map_id, comp_id]


@[reassoc (attr := simp)]
theorem associativity_inv (X Y Z : C) :
    F.obj X ◁ μ F Y Z ≫ μ F X (Y ⊗ Z) ≫ F.map (α_ X Y Z).inv =
      (α_ (F.obj X) (F.obj Y) (F.obj Z)).inv ≫ μ F X Y ▷ F.obj Z ≫ μ F (X ⊗ Y) Z := by
  rw [Iso.eq_inv_comp, ← associativity_assoc, ← F.map_comp, Iso.hom_inv_id,
    F.map_id, comp_id]


variable {F}
    /- unit morphism -/
    (ε' : 𝟙_ D ⟶ F.obj (𝟙_ C))
    /- tensorator -/
    (μ' : ∀ X Y : C, F.obj X ⊗ F.obj Y ⟶ F.obj (X ⊗ Y))
    (μ'_natural :
      ∀ {X Y X' Y' : C} (f : X ⟶ Y) (g : X' ⟶ Y'),
        (F.map f ⊗ F.map g) ≫ μ' Y Y' = μ' X X' ≫ F.map (f ⊗ g) := by
      aesop_cat)
    /- associativity of the tensorator -/
    (associativity' :
      ∀ X Y Z : C,
        (μ' X Y ⊗ 𝟙 (F.obj Z)) ≫ μ' (X ⊗ Y) Z ≫ F.map (α_ X Y Z).hom =
          (α_ (F.obj X) (F.obj Y) (F.obj Z)).hom ≫ (𝟙 (F.obj X) ⊗ μ' Y Z) ≫ μ' X (Y ⊗ Z) := by
      aesop_cat)
    /- unitality -/
    (left_unitality' :
      ∀ X : C, (λ_ (F.obj X)).hom = (ε' ⊗ 𝟙 (F.obj X)) ≫ μ' (𝟙_ C) X ≫ F.map (λ_ X).hom := by
        aesop_cat)
    (right_unitality' :
      ∀ X : C, (ρ_ (F.obj X)).hom = (𝟙 (F.obj X) ⊗ ε') ≫ μ' X (𝟙_ C) ≫ F.map (ρ_ X).hom := by
        aesop_cat)


/--
A constructor for lax monoidal functors whose axioms are described by `tensorHom` instead of
`whiskerLeft` and `whiskerRight`.
-/
def ofTensorHom : F.LaxMonoidal where
  ε' := ε'
  μ' := μ'
  μ'_natural_left := fun f X' => by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      ε' : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.obj Catego …
      μ' : (X Y : C) → Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj ( …
      μ'_natural : autoParam (∀ {X Y X' Y' : C} (f : Quiver.Hom X Y) (g : Quiver.Hom …
      associativity' : autoParam (∀ (X Y Z : C), Eq (CategoryTheory.CategoryStruct.c …
      left_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategoryStr …
      right_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategorySt …
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      X' : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp_rw [← tensorHom_id, ← F.map_id, μ'_natural]
    /-
      🎉 no goals
    -/
  μ'_natural_right := fun X' f => by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      ε' : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.obj Catego …
      μ' : (X Y : C) → Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj ( …
      μ'_natural : autoParam (∀ {X Y X' Y' : C} (f : Quiver.Hom X Y) (g : Quiver.Hom …
      associativity' : autoParam (∀ (X Y Z : C), Eq (CategoryTheory.CategoryStruct.c …
      left_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategoryStr …
      right_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategorySt …
      X✝ Y✝ X' : C
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp_rw [← id_tensorHom, ← F.map_id, μ'_natural]
    /-
      🎉 no goals
    -/
  associativity' := fun X Y Z => by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      ε' : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.obj Catego …
      μ' : (X Y : C) → Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj ( …
      μ'_natural : autoParam (∀ {X Y X' Y' : C} (f : Quiver.Hom X Y) (g : Quiver.Hom …
      associativity' : autoParam (∀ (X Y Z : C), Eq (CategoryTheory.CategoryStruct.c …
      left_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategoryStr …
      right_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategorySt …
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp_rw [← tensorHom_id, ← id_tensorHom, associativity']
    /-
      🎉 no goals
    -/
  left_unitality' := fun X => by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      ε' : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.obj Catego …
      μ' : (X Y : C) → Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj ( …
      μ'_natural : autoParam (∀ {X Y X' Y' : C} (f : Quiver.Hom X Y) (g : Quiver.Hom …
      associativity' : autoParam (∀ (X Y Z : C), Eq (CategoryTheory.CategoryStruct.c …
      left_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategoryStr …
      right_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategorySt …
      X : C
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (F.obj X)).hom (Categor …
    -/
    simp_rw [← tensorHom_id, left_unitality']
    /-
      🎉 no goals
    -/
  right_unitality' := fun X => by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      ε' : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.obj Catego …
      μ' : (X Y : C) → Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj ( …
      μ'_natural : autoParam (∀ {X Y X' Y' : C} (f : Quiver.Hom X Y) (g : Quiver.Hom …
      associativity' : autoParam (∀ (X Y Z : C), Eq (CategoryTheory.CategoryStruct.c …
      left_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategoryStr …
      right_unitality' : autoParam (∀ (X : C), Eq (CategoryTheory.MonoidalCategorySt …
      X : C
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (F.obj X)).hom (Catego …
    -/
    simp_rw [← id_tensorHom, right_unitality']
    /-
      🎉 no goals
    -/


lemma ofTensorHom_ε :
    letI := (ofTensorHom ε' μ' μ'_natural associativity' left_unitality' right_unitality')
    ε F = ε' := rfl


lemma ofTensorHom_μ :
    letI := (ofTensorHom ε' μ' μ'_natural associativity' left_unitality' right_unitality')
    μ F = μ' := rfl


instance id : (𝟭 C).LaxMonoidal where
  ε' := 𝟙 _
  μ' _ _ := 𝟙 _


@[simp]
lemma id_ε : ε (𝟭 C) = 𝟙 _ := rfl


@[simp]
lemma id_μ (X Y : C) : μ (𝟭 C) X Y = 𝟙 _ := rfl


instance comp : (F ⋙ G).LaxMonoidal where
  ε' := ε G ≫ G.map (ε F)
  μ' X Y := μ G _ _ ≫ G.map (μ F X Y)
  μ'_natural_left _ _ := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.LaxMonoidal
      inst✝ : G.LaxMonoidal
      X✝ Y✝ : C
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp_rw [comp_obj, F.comp_map, μ_natural_left_assoc, assoc, ← G.map_comp, μ_natural_left]
    /-
      🎉 no goals
    -/
  μ'_natural_right _ _ := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.LaxMonoidal
      inst✝ : G.LaxMonoidal
      X✝ Y✝ x✝¹ : C
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp_rw [comp_obj, F.comp_map, μ_natural_right_assoc, assoc, ← G.map_comp, μ_natural_right]
    /-
      🎉 no goals
    -/
  associativity' _ _ _ := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.LaxMonoidal
      inst✝ : G.LaxMonoidal
      x✝² x✝¹ x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    dsimp
    simp_rw [comp_whiskerRight, assoc, μ_natural_left_assoc, MonoidalCategory.whiskerLeft_comp,
      assoc, μ_natural_right_assoc, ← associativity_assoc, ← G.map_comp, associativity]


@[simp]
lemma comp_ε : ε (F ⋙ G) = ε G ≫ G.map (ε F) := rfl


@[simp]
lemma comp_μ (X Y : C) : μ (F ⋙ G) X Y = μ G _ _ ≫ G.map (μ F X Y) := rfl


/-- A functor `F : C ⥤ D` between monoidal categories is oplax monoidal if it is
equipped with morphisms `η : F.obj (𝟙_ C) ⟶ 𝟙 _D` and `δ X Y : F.obj (X ⊗ Y) ⟶ F.obj X ⊗ F.obj Y`,
satisfying the appropriate coherences. -/
class OplaxMonoidal where
  /-- counit morphism -/
  η' : F.obj (𝟙_ C) ⟶ 𝟙_ D
  /-- cotensorator -/
  δ' : ∀ X Y : C, F.obj (X ⊗ Y) ⟶ F.obj X ⊗ F.obj Y
  δ'_natural_left :
    ∀ {X Y : C} (f : X ⟶ Y) (X' : C),
      δ' X X' ≫ F.map f ▷ F.obj X' = F.map (f ▷ X') ≫ δ' Y X' := by
    aesop_cat
  δ'_natural_right :
    ∀ {X Y : C} (X' : C) (f : X ⟶ Y) ,
      δ' X' X ≫ F.obj X' ◁ F.map f = F.map (X' ◁ f) ≫ δ' X' Y := by
    aesop_cat
  /-- associativity of the tensorator -/
  oplax_associativity' :
    ∀ X Y Z : C,
      δ' (X ⊗ Y) Z ≫ δ' X Y ▷ F.obj Z ≫ (α_ (F.obj X) (F.obj Y) (F.obj Z)).hom =
        F.map (α_ X Y Z).hom ≫ δ' X (Y ⊗ Z) ≫ F.obj X ◁ δ' Y Z := by
    aesop_cat
  -- unitality
  oplax_left_unitality' :
    ∀ X : C, (λ_ (F.obj X)).inv = F.map (λ_ X).inv ≫ δ' (𝟙_ C) X ≫ η' ▷ F.obj X := by
      aesop_cat
  oplax_right_unitality' :
    ∀ X : C, (ρ_ (F.obj X)).inv = F.map (ρ_ X).inv ≫ δ' X (𝟙_ C) ≫ F.obj X ◁ η' := by
      aesop_cat


/-- the counit morphism of a lax monoidal functor-/
def η : F.obj (𝟙_ C) ⟶ 𝟙_ D := η'


/-- the cotensorator of an oplax monoidal functor -/
def δ (X Y : C) : F.obj (X ⊗ Y) ⟶ F.obj X ⊗ F.obj Y := δ' X Y


@[reassoc (attr := simp)]
lemma δ_natural_left {X Y : C} (f : X ⟶ Y) (X' : C) :
    δ F X X' ≫ F.map f ▷ F.obj X' = F.map (f ▷ X') ≫ δ F Y X' := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.OplaxMonoidal
    X Y : C
    f : Quiver.Hom X Y
    X' : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
  -/
  apply δ'_natural_left
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma δ_natural_right {X Y : C} (X' : C) (f : X ⟶ Y) :
    δ F X' X ≫ F.obj X' ◁ F.map f = F.map (X' ◁ f) ≫ δ F X' Y := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.OplaxMonoidal
    X Y X' : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
  -/
  apply δ'_natural_right
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma associativity (X Y Z : C) :
    δ F (X ⊗ Y) Z ≫ δ F X Y ▷ F.obj Z ≫ (α_ (F.obj X) (F.obj Y) (F.obj Z)).hom =
      F.map (α_ X Y Z).hom ≫ δ F X (Y ⊗ Z) ≫ F.obj X ◁ δ F Y Z := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.OplaxMonoidal
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
  -/
  apply oplax_associativity'
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma left_unitality (X : C) :
    (λ_ (F.obj X)).inv = F.map (λ_ X).inv ≫ δ F (𝟙_ C) X ≫ η F ▷ F.obj X := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.OplaxMonoidal
    X : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (F.obj X)).inv (Categor …
  -/
  apply oplax_left_unitality'
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma right_unitality (X : C) :
    (ρ_ (F.obj X)).inv = F.map (ρ_ X).inv ≫ δ F X (𝟙_ C) ≫ F.obj X ◁ η F := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.OplaxMonoidal
    X : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (F.obj X)).inv (Catego …
  -/
  apply oplax_right_unitality'
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem δ_natural {X Y X' Y' : C} (f : X ⟶ Y) (g : X' ⟶ Y') :
    δ F X X' ≫ (F.map f ⊗ F.map g) = F.map (f ⊗ g) ≫ δ F Y Y' := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.OplaxMonoidal
    X Y X' Y' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X' Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem left_unitality_hom  (X : C) :
    δ F (𝟙_ C) X ≫ η F ▷ F.obj X ≫ (λ_ (F.obj X)).hom = F.map (λ_ X).hom := by
  rw [← Category.assoc, ← Iso.eq_comp_inv, left_unitality, ← Category.assoc,
    ← F.map_comp, Iso.hom_inv_id, F.map_id, id_comp]


@[reassoc (attr := simp)]
theorem right_unitality_hom (X : C) :
    δ F X (𝟙_ C) ≫ F.obj X ◁ η F ≫ (ρ_ (F.obj X)).hom = F.map (ρ_ X).hom := by
  rw [← Category.assoc, ← Iso.eq_comp_inv, right_unitality, ← Category.assoc,
    ← F.map_comp, Iso.hom_inv_id, F.map_id, id_comp]


@[reassoc (attr := simp)]
theorem associativity_inv (X Y Z : C) :
    δ F X (Y ⊗ Z) ≫ F.obj X ◁ δ F Y Z ≫ (α_ (F.obj X) (F.obj Y) (F.obj Z)).inv =
      F.map (α_ X Y Z).inv ≫ δ F (X ⊗ Y) Z ≫ δ F X Y ▷ F.obj Z := by
  rw [← Category.assoc, Iso.comp_inv_eq, Category.assoc, Category.assoc, associativity,
    ← Category.assoc, ← F.map_comp, Iso.inv_hom_id, F.map_id, id_comp]


instance id : (𝟭 C).OplaxMonoidal where
  η' := 𝟙 _
  δ' _ _ := 𝟙 _


@[simp]
lemma id_η : η (𝟭 C) = 𝟙 _ := rfl


@[simp]
lemma id_δ (X Y : C) : δ (𝟭 C) X Y = 𝟙 _ := rfl


instance comp : (F ⋙ G).OplaxMonoidal where
  η' := G.map (η F) ≫ η G
  δ' X Y := G.map (δ F X Y) ≫ δ G _ _
  δ'_natural_left {X Y} f X' := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.OplaxMonoidal
      inst✝ : G.OplaxMonoidal
      X Y : C
      f : Quiver.Hom X Y
      X' : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.CategoryS …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.OplaxMonoidal
      inst✝ : G.OplaxMonoidal
      X Y : C
      f : Quiver.Hom X Y
      X' : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, δ_natural_left, ← G.map_comp_assoc, δ_natural_left, map_comp, assoc]
    /-
      🎉 no goals
    -/
  δ'_natural_right _ _ := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.OplaxMonoidal
      inst✝ : G.OplaxMonoidal
      X✝ Y✝ x✝¹ : C
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.CategoryS …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.OplaxMonoidal
      inst✝ : G.OplaxMonoidal
      X✝ Y✝ x✝¹ : C
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, δ_natural_right, ← G.map_comp_assoc, δ_natural_right, map_comp, assoc]
    /-
      🎉 no goals
    -/
  oplax_associativity' X Y Z := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.OplaxMonoidal
      inst✝ : G.OplaxMonoidal
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.CategoryS …
    -/
    dsimp
    rw [comp_whiskerRight, assoc, assoc, assoc, δ_natural_left_assoc, associativity,
      ← G.map_comp_assoc, ← G.map_comp_assoc, assoc, associativity, map_comp, map_comp,
      assoc, assoc, MonoidalCategory.whiskerLeft_comp, δ_natural_right_assoc]


@[simp]
lemma comp_η : η (F ⋙ G) = G.map (η F) ≫ η G := rfl


@[simp]
lemma comp_δ (X Y : C) : δ (F ⋙ G) X Y = G.map (δ F X Y) ≫ δ G _ _ := rfl


/-- A functor between monoidal categories is monoidal if it is lax and oplax monoidals,
and both data give inverse isomorphisms. -/
class Monoidal extends F.LaxMonoidal, F.OplaxMonoidal where
  ε_η : ε F ≫ η F = 𝟙 _ := by aesop_cat
  η_ε : η F ≫ ε F = 𝟙 _ := by aesop_cat
  μ_δ (X Y : C) : μ F X Y ≫ δ F X Y = 𝟙 _ := by aesop_cat
  δ_μ (X Y : C) : δ F X Y ≫ μ F X Y = 𝟙 _ := by aesop_cat


attribute [reassoc (attr := simp)] ε_η η_ε μ_δ δ_μ


/-- The isomorphism `𝟙_ D ≅ F.obj (𝟙_ C)` when `F` is a monoidal functor. -/
@[simps]
def εIso : 𝟙_ D ≅ F.obj (𝟙_ C) where
  hom := ε F
  inv := η F


/-- The isomorphism `F.obj X ⊗ F.obj Y ≅ F.obj (X ⊗ Y)` when `F` is a monoidal functor. -/
@[simps]
def μIso (X Y : C) : F.obj X ⊗ F.obj Y ≅ F.obj (X ⊗ Y) where
  hom := μ F X Y
  inv := δ F X Y


instance : IsIso (ε F) := (εIso F).isIso_hom

instance : IsIso (η F) := (εIso F).isIso_inv

instance (X Y : C) : IsIso (μ F X Y) := (μIso F X Y).isIso_hom

instance (X Y : C) : IsIso (δ F X Y) := (μIso F X Y).isIso_inv


@[reassoc (attr := simp)]
lemma map_ε_η (G : D ⥤ C') : G.map (ε F) ≫ G.map (η F) = 𝟙 _ :=
  (εIso F).map_hom_inv_id G


@[reassoc (attr := simp)]
lemma map_η_ε (G : D ⥤ C') : G.map (η F) ≫ G.map (ε F) = 𝟙 _ :=
  (εIso F).map_inv_hom_id G


@[reassoc (attr := simp)]
lemma map_μ_δ (G : D ⥤ C') (X Y : C) : G.map (μ F X Y) ≫ G.map (δ F X Y) = 𝟙 _ :=
  (μIso F X Y).map_hom_inv_id G


@[reassoc (attr := simp)]
lemma map_δ_μ (G : D ⥤ C') (X Y : C) : G.map (δ F X Y) ≫ G.map (μ F X Y) = 𝟙 _ :=
  (μIso F X Y).map_inv_hom_id G


@[reassoc (attr := simp)]
lemma whiskerRight_ε_η (T : D) : ε F ▷ T ≫ η F ▷ T = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    T : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← MonoidalCategory.comp_whiskerRight, ε_η, id_whiskerRight]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma whiskerRight_η_ε (T : D) : η F ▷ T ≫ ε F ▷ T = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    T : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← MonoidalCategory.comp_whiskerRight, η_ε, id_whiskerRight]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma whiskerRight_μ_δ (X Y : C) (T : D) : μ F X Y ▷ T ≫ δ F X Y ▷ T = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    X Y : C
    T : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← MonoidalCategory.comp_whiskerRight, μ_δ, id_whiskerRight]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma whiskerRight_δ_μ (X Y : C) (T : D) : δ F X Y ▷ T ≫ μ F X Y ▷ T = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    X Y : C
    T : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← MonoidalCategory.comp_whiskerRight, δ_μ, id_whiskerRight]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma whiskerLeft_ε_η (T : D) : T ◁ ε F ≫ T ◁ η F = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    T : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← MonoidalCategory.whiskerLeft_comp, ε_η, MonoidalCategory.whiskerLeft_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma whiskerLeft_η_ε (T : D) : T ◁ η F ≫ T ◁ ε F = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    T : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← MonoidalCategory.whiskerLeft_comp, η_ε, MonoidalCategory.whiskerLeft_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma whiskerLeft_μ_δ (X Y : C) (T : D) : T ◁ μ F X Y ≫ T ◁ δ F X Y = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    X Y : C
    T : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← MonoidalCategory.whiskerLeft_comp, μ_δ, MonoidalCategory.whiskerLeft_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma whiskerLeft_δ_μ (X Y : C) (T : D) : T ◁ δ F X Y ≫ T ◁ μ F X Y = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    X Y : C
    T : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← MonoidalCategory.whiskerLeft_comp, δ_μ, MonoidalCategory.whiskerLeft_id]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem map_tensor {X Y X' Y' : C} (f : X ⟶ Y) (g : X' ⟶ Y') :
                                                                    /-
                                                                      C : Type u₁
                                                                      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                      inst✝³ : CategoryTheory.MonoidalCategory C
                                                                      D : Type u₂
                                                                      inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                      inst✝¹ : CategoryTheory.MonoidalCategory D
                                                                      F : CategoryTheory.Functor C D
                                                                      inst✝ : F.Monoidal
                                                                      X Y X' Y' : C
                                                                      f : Quiver.Hom X Y
                                                                      g : Quiver.Hom X' Y'
                                                                      ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.tensorHom f g)) (CategoryTh …
                                                                    -/
    F.map (f ⊗ g) = δ F X X' ≫ (F.map f ⊗ F.map g) ≫ μ F Y Y' := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[reassoc]
theorem map_whiskerLeft (X : C) {Y Z : C} (f : Y ⟶ Z) :
                                                                /-
                                                                  C : Type u₁
                                                                  inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                  inst✝³ : CategoryTheory.MonoidalCategory C
                                                                  D : Type u₂
                                                                  inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                  inst✝¹ : CategoryTheory.MonoidalCategory D
                                                                  F : CategoryTheory.Functor C D
                                                                  inst✝ : F.Monoidal
                                                                  X Y Z : C
                                                                  f : Quiver.Hom Y Z
                                                                  ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X f)) (Category …
                                                                -/
    F.map (X ◁ f) = δ F X Y ≫ F.obj X ◁ F.map f ≫ μ F X Z := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[reassoc]
theorem map_whiskerRight {X Y : C} (f : X ⟶ Y) (Z : C) :
                                                                /-
                                                                  C : Type u₁
                                                                  inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                  inst✝³ : CategoryTheory.MonoidalCategory C
                                                                  D : Type u₂
                                                                  inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                  inst✝¹ : CategoryTheory.MonoidalCategory D
                                                                  F : CategoryTheory.Functor C D
                                                                  inst✝ : F.Monoidal
                                                                  X Y : C
                                                                  f : Quiver.Hom X Y
                                                                  Z : C
                                                                  ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.whiskerRight f Z)) (Categor …
                                                                -/
    F.map (f ▷ Z) = δ F X Z ≫ F.map f ▷ F.obj Z ≫ μ F Y Z := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[reassoc]
theorem map_associator (X Y Z : C) :
    F.map (α_ X Y Z).hom =
      δ F (X ⊗ Y) Z ≫ δ F X Y ▷ F.obj Z ≫
        (α_ (F.obj X) (F.obj Y) (F.obj Z)).hom ≫ F.obj X ◁ μ F Y Z ≫ μ F X (Y ⊗ Z) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    inst✝ : F.Monoidal
    X Y Z : C
    ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.associator X Y Z).hom) (Cat …
  -/
  rw [← LaxMonoidal.associativity F, whiskerRight_δ_μ_assoc, δ_μ_assoc]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem map_associator_inv (X Y Z : C) :
    F.map (α_ X Y Z).inv =
      δ F X (Y ⊗ Z) ≫ F.obj X ◁ δ F Y Z ≫
        (α_ (F.obj X) (F.obj Y) (F.obj Z)).inv ≫ μ F X Y ▷ F.obj Z ≫ μ F (X ⊗ Y) Z := by
  rw [← cancel_epi (F.map (α_ X Y Z).hom), Iso.map_hom_inv_id, map_associator,
    assoc, assoc, assoc, assoc, OplaxMonoidal.associativity_inv_assoc,
    whiskerRight_δ_μ_assoc, δ_μ, comp_id, LaxMonoidal.associativity_inv,
    Iso.hom_inv_id_assoc, whiskerRight_δ_μ_assoc, δ_μ]


@[reassoc]
theorem map_leftUnitor (X : C) :
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 inst✝³ : CategoryTheory.MonoidalCategory C
                                                                                 D : Type u₂
                                                                                 inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                                 inst✝¹ : CategoryTheory.MonoidalCategory D
                                                                                 F : CategoryTheory.Functor C D
                                                                                 inst✝ : F.Monoidal
                                                                                 X : C
                                                                                 ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).hom) (Categor …
                                                                               -/
    F.map (λ_ X).hom = δ F (𝟙_ C) X ≫ η F ▷ F.obj X ≫ (λ_ (F.obj X)).hom := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[reassoc]
theorem map_leftUnitor_inv (X : C) :
                                                                                /-
                                                                                  C : Type u₁
                                                                                  inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                                  inst✝³ : CategoryTheory.MonoidalCategory C
                                                                                  D : Type u₂
                                                                                  inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                                  inst✝¹ : CategoryTheory.MonoidalCategory D
                                                                                  F : CategoryTheory.Functor C D
                                                                                  inst✝ : F.Monoidal
                                                                                  X : C
                                                                                  ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).inv) (Categor …
                                                                                -/
    F.map (λ_ X).inv = (λ_ (F.obj X)).inv ≫ ε F ▷ F.obj X ≫ μ F (𝟙_ C) X  := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[reassoc]
theorem map_rightUnitor (X : C) :
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 inst✝³ : CategoryTheory.MonoidalCategory C
                                                                                 D : Type u₂
                                                                                 inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                                 inst✝¹ : CategoryTheory.MonoidalCategory D
                                                                                 F : CategoryTheory.Functor C D
                                                                                 inst✝ : F.Monoidal
                                                                                 X : C
                                                                                 ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).hom) (Catego …
                                                                               -/
    F.map (ρ_ X).hom = δ F X (𝟙_ C) ≫ F.obj X ◁ η F ≫ (ρ_ (F.obj X)).hom := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[reassoc]
theorem map_rightUnitor_inv (X : C) :
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 inst✝³ : CategoryTheory.MonoidalCategory C
                                                                                 D : Type u₂
                                                                                 inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                                 inst✝¹ : CategoryTheory.MonoidalCategory D
                                                                                 F : CategoryTheory.Functor C D
                                                                                 inst✝ : F.Monoidal
                                                                                 X : C
                                                                                 ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).inv) (Catego …
                                                                               -/
    F.map (ρ_ X).inv = (ρ_ (F.obj X)).inv ≫ F.obj X ◁ ε F  ≫ μ F X (𝟙_ C):= by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The tensorator as a natural isomorphism. -/
@[simps!]
noncomputable def μNatIso :
    Functor.prod F F ⋙ tensor D ≅ tensor C ⋙ F :=
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝ : F.Monoidal
    ⊢ ∀ {X Y : Prod C C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.c …
  -/
  NatIso.ofComponents (fun _ ↦ μIso F _ _)
  /-
    🎉 no goals
  -/


/-- Monoidal functors commute with left tensoring up to isomorphism -/
@[simps!]
noncomputable def commTensorLeft (X : C) :
    F ⋙ tensorLeft (F.obj X) ≅ tensorLeft X ⋙ F :=
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝ : F.Monoidal
    X : C
    ⊢ ∀ {X_1 Y : C} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
  -/
  NatIso.ofComponents (fun Y => μIso F X Y)
  /-
    🎉 no goals
  -/


/-- Monoidal functors commute with right tensoring up to isomorphism -/
@[simps!]
noncomputable def commTensorRight (X : C) :
    F ⋙ tensorRight (F.obj X) ≅ tensorRight X ⋙ F :=
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝ : F.Monoidal
    X : C
    ⊢ ∀ {X_1 Y : C} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
  -/
  NatIso.ofComponents (fun Y => μIso F Y X)
  /-
    🎉 no goals
  -/


instance : (𝟭 C).Monoidal where


instance [F.Monoidal] [G.Monoidal] : (F ⋙ G).Monoidal where
            /-
              C : Type u₁
              inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁷ : CategoryTheory.MonoidalCategory C
              D : Type u₂
              inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
              inst✝⁵ : CategoryTheory.MonoidalCategory D
              E : Type u₃
              inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
              inst✝³ : CategoryTheory.MonoidalCategory E
              C' : Type u₁'
              inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
              F : CategoryTheory.Functor C D
              G : CategoryTheory.Functor D E
              inst✝¹ : F.Monoidal
              inst✝ : G.Monoidal
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
            -/
  ε_η := by simp
            /-
              🎉 no goals
            -/
            /-
              C : Type u₁
              inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁷ : CategoryTheory.MonoidalCategory C
              D : Type u₂
              inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
              inst✝⁵ : CategoryTheory.MonoidalCategory D
              E : Type u₃
              inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
              inst✝³ : CategoryTheory.MonoidalCategory E
              C' : Type u₁'
              inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
              F : CategoryTheory.Functor C D
              G : CategoryTheory.Functor D E
              inst✝¹ : F.Monoidal
              inst✝ : G.Monoidal
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
            -/
  η_ε := by simp
            /-
              🎉 no goals
            -/
                /-
                  C : Type u₁
                  inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝⁷ : CategoryTheory.MonoidalCategory C
                  D : Type u₂
                  inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝⁵ : CategoryTheory.MonoidalCategory D
                  E : Type u₃
                  inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
                  inst✝³ : CategoryTheory.MonoidalCategory E
                  C' : Type u₁'
                  inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
                  F : CategoryTheory.Functor C D
                  G : CategoryTheory.Functor D E
                  inst✝¹ : F.Monoidal
                  inst✝ : G.Monoidal
                  x✝¹ x✝ : C
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
                -/
  μ_δ _ _ := by simp
                /-
                  🎉 no goals
                -/
                /-
                  C : Type u₁
                  inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝⁷ : CategoryTheory.MonoidalCategory C
                  D : Type u₂
                  inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝⁵ : CategoryTheory.MonoidalCategory D
                  E : Type u₃
                  inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
                  inst✝³ : CategoryTheory.MonoidalCategory E
                  C' : Type u₁'
                  inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
                  F : CategoryTheory.Functor C D
                  G : CategoryTheory.Functor D E
                  inst✝¹ : F.Monoidal
                  inst✝ : G.Monoidal
                  x✝¹ x✝ : C
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
                -/
  δ_μ _ _ := by simp
                /-
                  🎉 no goals
                -/


/-- Structure which is a helper in order to show that a functor is monoidal. It
consists of isomorphisms `εIso` and `μIso` such that the morphisms `.hom` induced
by these isomorphisms satisfy the axioms of lax monoidal functors. -/
structure CoreMonoidal where
  /-- unit morphism -/
  εIso : 𝟙_ D ≅ F.obj (𝟙_ C)
  /-- tensorator -/
  μIso : ∀ X Y : C, F.obj X ⊗ F.obj Y ≅ F.obj (X ⊗ Y)
  μIso_hom_natural_left :
    ∀ {X Y : C} (f : X ⟶ Y) (X' : C),
      F.map f ▷ F.obj X' ≫ (μIso Y X').hom = (μIso X X').hom ≫ F.map (f ▷ X') := by
    aesop_cat
  μIso_hom_natural_right :
    ∀ {X Y : C} (X' : C) (f : X ⟶ Y) ,
      F.obj X' ◁ F.map f ≫ (μIso X' Y).hom = (μIso X' X).hom ≫ F.map (X' ◁ f) := by
    aesop_cat
  /-- associativity of the tensorator -/
  associativity :
    ∀ X Y Z : C,
      (μIso X Y).hom ▷ F.obj Z ≫ (μIso (X ⊗ Y) Z).hom ≫ F.map (α_ X Y Z).hom =
        (α_ (F.obj X) (F.obj Y) (F.obj Z)).hom ≫ F.obj X ◁ (μIso Y Z).hom ≫
          (μIso X (Y ⊗ Z)).hom := by
    aesop_cat
  -- unitality
  left_unitality :
    ∀ X : C, (λ_ (F.obj X)).hom = εIso.hom ▷ F.obj X ≫ (μIso (𝟙_ C) X).hom ≫ F.map (λ_ X).hom := by
      aesop_cat
  right_unitality :
    ∀ X : C, (ρ_ (F.obj X)).hom = F.obj X ◁ εIso.hom ≫ (μIso X (𝟙_ C)).hom ≫ F.map (ρ_ X).hom := by
    aesop_cat


attribute [reassoc (attr := simp)] μIso_hom_natural_left
  μIso_hom_natural_right associativity


attribute [reassoc] left_unitality right_unitality


/-- The lax monoidal functor structure induced by a `Functor.CoreMonoidal` structure. -/
def toLaxMonoidal : F.LaxMonoidal where
  ε' := h.εIso.hom
  μ' X Y := (h.μIso X Y).hom
  left_unitality' := h.left_unitality
  right_unitality' := h.right_unitality


lemma toLaxMonoidal_ε :
    letI := h.toLaxMonoidal
    LaxMonoidal.ε F = h.εIso.hom := rfl


lemma toLaxMonoidal_μ (X Y : C) :
    letI := h.toLaxMonoidal
    LaxMonoidal.μ F X Y = (h.μIso X Y).hom := rfl


/-- The oplax monoidal functor structure induced by a `Functor.CoreMonoidal` structure. -/
def toOplaxMonoidal : F.OplaxMonoidal where
  η' := h.εIso.inv
  δ' X Y := (h.μIso X Y).inv
  δ'_natural_left _ _ := by
    rw [← cancel_epi (h.μIso _ _).hom, Iso.hom_inv_id_assoc,
      ← h.μIso_hom_natural_left_assoc, Iso.hom_inv_id, comp_id]
  δ'_natural_right _ _ := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      h : F.CoreMonoidal
      X✝ Y✝ x✝¹ : C
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => (h.μIso X Y).inv) x✝¹ X✝ …
    -/
    dsimp
    rw [← cancel_epi (h.μIso _ _).hom, Iso.hom_inv_id_assoc,
      ← h.μIso_hom_natural_right_assoc, Iso.hom_inv_id, comp_id]
  oplax_associativity' X Y Z := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      h : F.CoreMonoidal
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => (h.μIso X Y).inv) (Categ …
    -/
    dsimp
    rw [← cancel_epi (h.μIso (X ⊗ Y) Z).hom, Iso.hom_inv_id_assoc,
      ← cancel_epi ((h.μIso X Y).hom ▷ F.obj Z), hom_inv_whiskerRight_assoc,
      associativity_assoc, Iso.hom_inv_id_assoc, whiskerLeft_hom_inv, comp_id]
  oplax_left_unitality' _ := by
    rw [← cancel_epi (λ_ _).hom, Iso.hom_inv_id, h.left_unitality, assoc, assoc,
      Iso.map_hom_inv_id_assoc, Iso.hom_inv_id_assoc,hom_inv_whiskerRight]
  oplax_right_unitality' _ := by
    rw [← cancel_epi (ρ_ _).hom, Iso.hom_inv_id, h.right_unitality, assoc, assoc,
      Iso.map_hom_inv_id_assoc, Iso.hom_inv_id_assoc, whiskerLeft_hom_inv]


lemma toOplaxMonoidal_η :
    letI := h.toOplaxMonoidal
    OplaxMonoidal.η F = h.εIso.inv := rfl


lemma toOplaxMonoidal_δ  (X Y : C) :
    letI := h.toOplaxMonoidal
    OplaxMonoidal.δ F X Y = (h.μIso X Y).inv := rfl


attribute [local simp] toLaxMonoidal_ε toLaxMonoidal_μ toOplaxMonoidal_η toOplaxMonoidal_δ in
/-- The monoidal functor structure induced by a `Functor.CoreMonoidal` structure. -/
@[simps! toLaxMonoidal toOplaxMonoidal]
def toMonoidal : F.Monoidal where
  toLaxMonoidal := h.toLaxMonoidal
  toOplaxMonoidal := h.toOplaxMonoidal


/-- The `Functor.CoreMonoidal` structure given by a lax monoidal functor such
that `ε` and `μ` are isomorphisms. -/
noncomputable def ofLaxMonoidal [F.LaxMonoidal] [IsIso (ε F)] [∀ X Y, IsIso (μ F X Y)] :
    F.CoreMonoidal where
  εIso := asIso (ε F)
  μIso X Y := asIso (μ F X Y)


/-- The `Functor.CoreMonoidal` structure given by an oplax monoidal functor such
that `η` and `δ` are isomorphisms. -/
noncomputable def ofOplaxMonoidal [F.OplaxMonoidal] [IsIso (η F)] [∀ X Y, IsIso (δ F X Y)] :
    F.CoreMonoidal where
  εIso := (asIso (η F)).symm
  μIso X Y := (asIso (δ F X Y)).symm
  associativity X Y Z := by
    /-
      C : Type u₁
      inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁴ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      h : F.CoreMonoidal
      inst✝² : F.OplaxMonoidal
      inst✝¹ : CategoryTheory.IsIso (CategoryTheory.Functor.OplaxMonoidal.η F)
      inst✝ : ∀ (X Y : C), CategoryTheory.IsIso (CategoryTheory.Functor.OplaxMonoida …
      X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp [← cancel_epi (δ F X Y ▷ F.obj Z), ← cancel_epi (δ F (X ⊗ Y) Z)]
    /-
      🎉 no goals
    -/
                         /-
                           C : Type u₁
                           inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
                           inst✝⁸ : CategoryTheory.MonoidalCategory C
                           D : Type u₂
                           inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
                           inst✝⁶ : CategoryTheory.MonoidalCategory D
                           E : Type u₃
                           inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                           inst✝⁴ : CategoryTheory.MonoidalCategory E
                           C' : Type u₁'
                           inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
                           F : CategoryTheory.Functor C D
                           G : CategoryTheory.Functor D E
                           h : F.CoreMonoidal
                           inst✝² : F.OplaxMonoidal
                           inst✝¹ : CategoryTheory.IsIso (CategoryTheory.Functor.OplaxMonoidal.η F)
                           inst✝ : ∀ (X Y : C), CategoryTheory.IsIso (CategoryTheory.Functor.OplaxMonoida …
                           X : C
                           ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (F.obj X)).hom (Categor …
                         -/
  left_unitality X := by simp [← cancel_epi (λ_ (F.obj X)).inv]
                         /-
                           🎉 no goals
                         -/
                          /-
                            C : Type u₁
                            inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
                            inst✝⁸ : CategoryTheory.MonoidalCategory C
                            D : Type u₂
                            inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
                            inst✝⁶ : CategoryTheory.MonoidalCategory D
                            E : Type u₃
                            inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                            inst✝⁴ : CategoryTheory.MonoidalCategory E
                            C' : Type u₁'
                            inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
                            F : CategoryTheory.Functor C D
                            G : CategoryTheory.Functor D E
                            h : F.CoreMonoidal
                            inst✝² : F.OplaxMonoidal
                            inst✝¹ : CategoryTheory.IsIso (CategoryTheory.Functor.OplaxMonoidal.η F)
                            inst✝ : ∀ (X Y : C), CategoryTheory.IsIso (CategoryTheory.Functor.OplaxMonoida …
                            X : C
                            ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (F.obj X)).hom (Catego …
                          -/
  right_unitality X := by simp [← cancel_epi (ρ_ (F.obj X)).inv]
                          /-
                            🎉 no goals
                          -/


/-- The `Functor.Monoidal` structure given by a lax monoidal functor such
that `ε` and `μ` are isomorphisms. -/
noncomputable def Monoidal.ofLaxMonoidal
    [F.LaxMonoidal] [IsIso (ε F)] [∀ X Y, IsIso (μ F X Y)] :=
  (CoreMonoidal.ofLaxMonoidal F).toMonoidal


/-- The `Functor.Monoidal` structure given by an oplax monoidal functor such
that `η` and `δ` are isomorphisms. -/
noncomputable def Monoidal.ofOplaxMonoidal
    [F.OplaxMonoidal] [IsIso (η F)] [∀ X Y, IsIso (δ F X Y)] :=
  (CoreMonoidal.ofOplaxMonoidal F).toMonoidal


instance : (prod F G).LaxMonoidal where
  ε' := (ε F, ε G)
  μ' X Y := (μ F _ _, μ G _ _)
  μ'_natural_left _ _ := by
    /-
      C : Type u₁
      inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁴ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor E C'
      inst✝² : CategoryTheory.MonoidalCategory C'
      inst✝¹ : F.LaxMonoidal
      inst✝ : G.LaxMonoidal
      X✝ Y✝ : Prod C E
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Prod C E
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    ext
    all_goals
      simp only [prod_obj, prodMonoidal_tensorObj, prod_map,
        prodMonoidal_whiskerRight, prod_comp, μ_natural_left]
  μ'_natural_right _ _ := by
    /-
      C : Type u₁
      inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁴ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor E C'
      inst✝² : CategoryTheory.MonoidalCategory C'
      inst✝¹ : F.LaxMonoidal
      inst✝ : G.LaxMonoidal
      X✝ Y✝ x✝¹ : Prod C E
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    ext
    all_goals
      simp only [prod_obj, prodMonoidal_tensorObj, prod_map, prodMonoidal_whiskerLeft, prod_comp,
        μ_natural_right]
  associativity' _ _ _ := by
    /-
      C : Type u₁
      inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁴ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor E C'
      inst✝² : CategoryTheory.MonoidalCategory C'
      inst✝¹ : F.LaxMonoidal
      inst✝ : G.LaxMonoidal
      x✝² x✝¹ x✝ : Prod C E
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    ext
    all_goals
      simp only [prod_obj, prodMonoidal_tensorObj, prodMonoidal_whiskerRight,
        prodMonoidal_associator, Iso.prod_hom, prod_map, prod_comp,
        LaxMonoidal.associativity, prodMonoidal_whiskerLeft]
  left_unitality' _ := by
    /-
      C : Type u₁
      inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁴ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor E C'
      inst✝² : CategoryTheory.MonoidalCategory C'
      inst✝¹ : F.LaxMonoidal
      inst✝ : G.LaxMonoidal
      x✝ : Prod C E
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor ((F.prod G).obj x✝)).ho …
    -/
    ext
    all_goals
      simp only [prodMonoidal_tensorUnit, prod_obj, prodMonoidal_tensorObj,
        prodMonoidal_leftUnitor_hom_fst, LaxMonoidal.left_unitality, prodMonoidal_whiskerRight,
        prod_map, prodMonoidal_leftUnitor_hom_snd, prod_comp]
  right_unitality' _ := by
    /-
      C : Type u₁
      inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁸ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁶ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁴ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor E C'
      inst✝² : CategoryTheory.MonoidalCategory C'
      inst✝¹ : F.LaxMonoidal
      inst✝ : G.LaxMonoidal
      x✝ : Prod C E
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor ((F.prod G).obj x✝)).h …
    -/
    ext
    all_goals
      simp only [prod_obj, prodMonoidal_tensorUnit, prodMonoidal_tensorObj,
        prodMonoidal_rightUnitor_hom_fst, LaxMonoidal.right_unitality, prodMonoidal_whiskerLeft,
        prod_map, prodMonoidal_rightUnitor_hom_snd, prod_comp]


@[simp] lemma prod_ε_fst : (ε (prod F G)).1 = ε F := rfl

@[simp] lemma prod_ε_snd : (ε (prod F G)).2 = ε G := rfl

@[simp] lemma prod_μ_fst (X Y : C × E) : (μ (prod F G) X Y).1 = μ F _ _ := rfl

@[simp] lemma prod_μ_snd (X Y : C × E) : (μ (prod F G) X Y).2 = μ G _ _ := rfl


instance : (prod F G).OplaxMonoidal where
  η' := (η F, η G)
  δ' X Y := (δ F _ _, δ G _ _)


@[simp] lemma prod_η_fst : (η (prod F G)).1 = η F := rfl

@[simp] lemma prod_η_snd : (η (prod F G)).2 = η G := rfl

@[simp] lemma prod_δ_fst (X Y : C × E) : (δ (prod F G) X Y).1 = δ F _ _ := rfl

@[simp] lemma prod_δ_snd (X Y : C × E) : (δ (prod F G) X Y).2 = δ G _ _ := rfl


instance [F.Monoidal] [G.Monoidal] : (prod F G).Monoidal where
            /-
              C : Type u₁
              inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁸ : CategoryTheory.MonoidalCategory C
              D : Type u₂
              inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
              inst✝⁶ : CategoryTheory.MonoidalCategory D
              E : Type u₃
              inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
              inst✝⁴ : CategoryTheory.MonoidalCategory E
              C' : Type u₁'
              inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
              F✝ : CategoryTheory.Functor C D
              G✝ : CategoryTheory.Functor D E
              F : CategoryTheory.Functor C D
              G : CategoryTheory.Functor E C'
              inst✝² : CategoryTheory.MonoidalCategory C'
              inst✝¹ : F.Monoidal
              inst✝ : G.Monoidal
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
            -/
                    /-
                      🎉 no goals
                    -/
  ε_η := by ext <;> apply Monoidal.ε_η
                    /-
                      🎉 no goals
                    -/
            /-
              C : Type u₁
              inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁸ : CategoryTheory.MonoidalCategory C
              D : Type u₂
              inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
              inst✝⁶ : CategoryTheory.MonoidalCategory D
              E : Type u₃
              inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
              inst✝⁴ : CategoryTheory.MonoidalCategory E
              C' : Type u₁'
              inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
              F✝ : CategoryTheory.Functor C D
              G✝ : CategoryTheory.Functor D E
              F : CategoryTheory.Functor C D
              G : CategoryTheory.Functor E C'
              inst✝² : CategoryTheory.MonoidalCategory C'
              inst✝¹ : F.Monoidal
              inst✝ : G.Monoidal
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
            -/
                    /-
                      🎉 no goals
                    -/
  η_ε := by ext <;> apply Monoidal.η_ε
                    /-
                      🎉 no goals
                    -/
                /-
                  C : Type u₁
                  inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝⁸ : CategoryTheory.MonoidalCategory C
                  D : Type u₂
                  inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝⁶ : CategoryTheory.MonoidalCategory D
                  E : Type u₃
                  inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                  inst✝⁴ : CategoryTheory.MonoidalCategory E
                  C' : Type u₁'
                  inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
                  F✝ : CategoryTheory.Functor C D
                  G✝ : CategoryTheory.Functor D E
                  F : CategoryTheory.Functor C D
                  G : CategoryTheory.Functor E C'
                  inst✝² : CategoryTheory.MonoidalCategory C'
                  inst✝¹ : F.Monoidal
                  inst✝ : G.Monoidal
                  x✝¹ x✝ : Prod C E
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
                -/
                        /-
                          🎉 no goals
                        -/
  μ_δ _ _ := by ext <;> apply Monoidal.μ_δ
                        /-
                          🎉 no goals
                        -/
                /-
                  C : Type u₁
                  inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝⁸ : CategoryTheory.MonoidalCategory C
                  D : Type u₂
                  inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝⁶ : CategoryTheory.MonoidalCategory D
                  E : Type u₃
                  inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                  inst✝⁴ : CategoryTheory.MonoidalCategory E
                  C' : Type u₁'
                  inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
                  F✝ : CategoryTheory.Functor C D
                  G✝ : CategoryTheory.Functor D E
                  F : CategoryTheory.Functor C D
                  G : CategoryTheory.Functor E C'
                  inst✝² : CategoryTheory.MonoidalCategory C'
                  inst✝¹ : F.Monoidal
                  inst✝ : G.Monoidal
                  x✝¹ x✝ : Prod C E
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
                -/
                        /-
                          🎉 no goals
                        -/
  δ_μ _ _ := by ext <;> apply Monoidal.δ_μ
                        /-
                          🎉 no goals
                        -/


instance : (diag C).Monoidal :=
  CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun _ _ ↦ Iso.refl _ }


@[simp] lemma diag_ε : ε (diag C) = 𝟙 _ := rfl

@[simp] lemma diag_η : η (diag C) = 𝟙 _ := rfl

@[simp] lemma diag_μ (X Y : C) : μ (diag C) X Y = 𝟙 _ := rfl

@[simp] lemma diag_δ (X Y : C) : δ (diag C) X Y = 𝟙 _ := rfl


/-- The functor `C ⥤ D × E` obtained from two lax monoidal functors is lax monoidal. -/
instance LaxMonoidal.prod' : (prod' F G).LaxMonoidal :=
  inferInstanceAs (diag C ⋙ prod F G).LaxMonoidal


@[simp] lemma prod'_ε_fst : (ε (prod' F G)).1 = ε F := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε (F.prod' G)).1 (CategoryTheory.Func …
  -/
  change _ ≫ F.map (𝟙 _) = _
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
  -/
  rw [Functor.map_id, Category.comp_id]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε (F.prod G)).1 (CategoryTheory.Funct …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma prod'_ε_snd : (ε (prod' F G)).2 = ε G := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε (F.prod' G)).2 (CategoryTheory.Func …
  -/
  change _ ≫ G.map (𝟙 _) = _
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
  -/
  rw [Functor.map_id, Category.comp_id]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε (F.prod G)).2 (CategoryTheory.Funct …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma prod'_μ_fst (X Y : C) : (μ (prod' F G) X Y).1 = μ F X Y := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ (F.prod' G) X Y).1 (CategoryTheory. …
  -/
  change _ ≫ F.map (𝟙 _) = _
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
  -/
  rw [Functor.map_id, Category.comp_id]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ (F.prod G) ((CategoryTheory.Functor …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma prod'_μ_snd (X Y : C) : (μ (prod' F G) X Y).2 = μ G X Y := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ (F.prod' G) X Y).2 (CategoryTheory. …
  -/
  change _ ≫ G.map (𝟙 _) = _
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
  -/
  rw [Functor.map_id, Category.comp_id]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.LaxMonoidal
    inst✝ : G.LaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ (F.prod G) ((CategoryTheory.Functor …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The functor `C ⥤ D × E` obtained from two oplax monoidal functors is oplax monoidal. -/
instance OplaxMonoidal.prod' : (prod' F G).OplaxMonoidal :=
  inferInstanceAs (diag C ⋙ prod F G).OplaxMonoidal


@[simp] lemma prod'_η_fst : (η (prod' F G)).1 = η F := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    ⊢ Eq (CategoryTheory.Functor.OplaxMonoidal.η (F.prod' G)).1 (CategoryTheory.Fu …
  -/
  change F.map (𝟙 _)  ≫ _ = _
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  rw [Functor.map_id, Category.id_comp]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    ⊢ Eq (CategoryTheory.Functor.OplaxMonoidal.η (F.prod G)).1 (CategoryTheory.Fun …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma prod'_η_snd : (η (prod' F G)).2 = η G := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    ⊢ Eq (CategoryTheory.Functor.OplaxMonoidal.η (F.prod' G)).2 (CategoryTheory.Fu …
  -/
  change G.map (𝟙 _)  ≫ _ = _
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStruct …
  -/
  rw [Functor.map_id, Category.id_comp]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    ⊢ Eq (CategoryTheory.Functor.OplaxMonoidal.η (F.prod G)).2 (CategoryTheory.Fun …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma prod'_δ_fst (X Y : C) : (δ (prod' F G) X Y).1 = δ F X Y := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.Functor.OplaxMonoidal.δ (F.prod' G) X Y).1 (CategoryTheor …
  -/
  change F.map (𝟙 _) ≫ _ = _
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  rw [Functor.map_id, Category.id_comp]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.Functor.OplaxMonoidal.δ (F.prod G) ((CategoryTheory.Funct …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma prod'_δ_snd (X Y : C) : (δ (prod' F G) X Y).2 = δ G X Y := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.Functor.OplaxMonoidal.δ (F.prod' G) X Y).2 (CategoryTheor …
  -/
  change G.map (𝟙 _) ≫ _ = _
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStruct …
  -/
  rw [Functor.map_id, Category.id_comp]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor C E
    inst✝¹ : F.OplaxMonoidal
    inst✝ : G.OplaxMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.Functor.OplaxMonoidal.δ (F.prod G) ((CategoryTheory.Funct …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma prod_comp_fst {C D : Type*} [Category C] [Category D]
    {X Y Z : C × D} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).1 = f.1 ≫ g.1 := rfl


@[simp, reassoc]
lemma prod_comp_snd {C D : Type*} [Category C] [Category D]
    {X Y Z : C × D} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).2 = f.2 ≫ g.2 := rfl


/-- The functor `C ⥤ D × E` obtained from two monoidal functors is monoidal. -/
instance Monoidal.prod' [F.Monoidal] [G.Monoidal] :
    (prod' F G).Monoidal where
  -- automation should work, but it is terribly slow
  ε_η := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor C E
      inst✝¹ : F.Monoidal
      inst✝ : G.Monoidal
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
    -/
    ext
    · simp only [prod_comp_fst, prod'_ε_fst, prod'_η_fst, ε_η,
        prodMonoidal_tensorUnit, prod_id]
    · simp only [prod_comp_snd, prod'_ε_snd, prod'_η_snd, ε_η,
        prodMonoidal_tensorUnit, prod_id]
  η_ε := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor C E
      inst✝¹ : F.Monoidal
      inst✝ : G.Monoidal
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
    -/
    ext
    · simp only [prod_comp_fst, prod'_ε_fst, prod'_η_fst, η_ε,
        prod_id, prod'_obj]
    · simp only [prod_comp_snd, prod'_ε_snd, prod'_η_snd, η_ε,
        prod_id, prod'_obj]
  μ_δ _ _ := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor C E
      inst✝¹ : F.Monoidal
      inst✝ : G.Monoidal
      x✝¹ x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    ext
    · simp only [prod_comp_fst, prod'_μ_fst, prod'_δ_fst, μ_δ,
        prod'_obj, prodMonoidal_tensorObj, prod_id]
    · simp only [prod_comp_snd, prod'_μ_snd, prod'_δ_snd, μ_δ,
        prod'_obj, prodMonoidal_tensorObj, prod_id]
  δ_μ _ _ := by
    /-
      C : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁷ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝² : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor C E
      inst✝¹ : F.Monoidal
      inst✝ : G.Monoidal
      x✝¹ x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
    -/
    ext
    · simp only [prod_comp_fst, prod'_μ_fst, prod'_δ_fst, δ_μ,
        prod'_obj, prod_id]
    · simp only [prod_comp_snd, prod'_μ_snd, prod'_δ_snd, δ_μ,
        prod'_obj, prod_id]


/-- The right adjoint of an oplax monoidal functor is lax monoidal. -/
def rightAdjointLaxMonoidal : G.LaxMonoidal where
  ε' := adj.homEquiv _ _ (η F)
  μ' X Y := adj.homEquiv _ _ (δ F _ _ ≫ (adj.counit.app X ⊗ adj.counit.app Y))
  μ'_natural_left {X Y} f X' := by
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y : D
      f : Quiver.Hom X Y
      X' : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    dsimp [Adjunction.homEquiv_apply]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y : D
      f : Quiver.Hom X Y
      X' : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    erw [adj.unit.naturality_assoc]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y : D
      f : Quiver.Hom X Y
      X' : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y : D
      f : Quiver.Hom X Y
      X' : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
    -/
    simp only [← G.map_comp, assoc, ← δ_natural_left_assoc F]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y : D
      f : Quiver.Hom X Y
      X' : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
    -/
    erw [NatTrans.whiskerRight_app_tensor_app adj.counit adj.counit]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y : D
      f : Quiver.Hom X Y
      X' : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
    -/
    dsimp
    /-
      🎉 no goals
    -/
  μ'_natural_right {X' Y'} X g := by
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X' Y' X : D
      g : Quiver.Hom X' Y'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    dsimp [Adjunction.homEquiv_apply]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X' Y' X : D
      g : Quiver.Hom X' Y'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    erw [adj.unit.naturality_assoc]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X' Y' X : D
      g : Quiver.Hom X' Y'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X' Y' X : D
      g : Quiver.Hom X' Y'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
    -/
    simp only [← G.map_comp, assoc, ← δ_natural_right_assoc F]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X' Y' X : D
      g : Quiver.Hom X' Y'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
    -/
    erw [NatTrans.whiskerLeft_app_tensor_app adj.counit adj.counit]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X' Y' X : D
      g : Quiver.Hom X' Y'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
    -/
    dsimp
    /-
      🎉 no goals
    -/
  associativity' X Y Z := (adj.homEquiv _ _).symm.injective (by
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y Z : D
      ⊢ Eq ((adj.homEquiv (CategoryTheory.MonoidalCategoryStruct.tensorObj (Category …
    -/
    dsimp
    simp only [homEquiv_unit, homEquiv_counit, map_comp, assoc, comp_whiskerRight,
      counit_naturality, counit_naturality_assoc, left_triangle_components_assoc,
      MonoidalCategory.whiskerLeft_comp]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y Z : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y Z : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
    -/
    rw [← δ_natural_left_assoc, ← δ_natural_left_assoc, ← δ_natural_left_assoc]
    erw [NatTrans.whiskerRight_app_tensor_app_assoc adj.counit adj.counit,
      NatTrans.whiskerRight_app_tensor_app_assoc adj.counit adj.counit]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y Z : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
    -/
    rw [tensorHom_def, assoc]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y Z : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
    -/
    dsimp
    rw [← comp_whiskerRight_assoc, left_triangle_components, id_whiskerRight, id_comp,
      whisker_exchange_assoc, whisker_exchange_assoc, ← tensorHom_def_assoc,
      associator_naturality, OplaxMonoidal.associativity_assoc]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y Z : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
    -/
    rw [← δ_natural_right_assoc, ← δ_natural_right_assoc, ← δ_natural_right_assoc]
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X Y Z : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
    -/
    nth_rw 4 [tensorHom_def]
    rw [← whisker_exchange, ← MonoidalCategory.whiskerLeft_comp_assoc,
      ← MonoidalCategory.whiskerLeft_comp_assoc,
      ← MonoidalCategory.whiskerLeft_comp_assoc, assoc, assoc,
      counit_naturality, counit_naturality_assoc, left_triangle_components_assoc,
      MonoidalCategory.whiskerLeft_comp, assoc, tensorHom_def, whisker_exchange])
  left_unitality' X := (adj.homEquiv _ _).symm.injective (by
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X : D
      ⊢ Eq ((adj.homEquiv (CategoryTheory.MonoidalCategoryStruct.tensorObj CategoryT …
    -/
    dsimp
    rw [homEquiv_counit, homEquiv_counit, homEquiv_unit, homEquiv_unit, comp_whiskerRight,
      map_comp, map_comp, map_comp, map_comp, map_comp, map_comp, assoc, assoc, assoc, assoc,
      assoc, counit_naturality, counit_naturality_assoc, counit_naturality_assoc,
      left_triangle_components_assoc, ← δ_natural_left_assoc, ← δ_natural_left_assoc,
      tensorHom_def, assoc, ← MonoidalCategory.comp_whiskerRight_assoc,
      ← MonoidalCategory.comp_whiskerRight_assoc, assoc, counit_naturality,
      left_triangle_components_assoc, id_whiskerLeft, assoc, assoc, Iso.inv_hom_id, comp_id,
      left_unitality_hom_assoc])
  right_unitality' X := (adj.homEquiv _ _).symm.injective (by
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      X : D
      ⊢ Eq ((adj.homEquiv (CategoryTheory.MonoidalCategoryStruct.tensorObj (G.obj X) …
    -/
    dsimp
    rw [homEquiv_counit, homEquiv_unit, MonoidalCategory.whiskerLeft_comp, homEquiv_unit,
      homEquiv_counit, map_comp, map_comp, map_comp, map_comp, map_comp, map_comp,
      assoc, assoc, assoc, assoc, assoc, counit_naturality, counit_naturality_assoc,
      counit_naturality_assoc, left_triangle_components_assoc, ← δ_natural_right_assoc,
      ← δ_natural_right_assoc, tensorHom_def, assoc, ← whisker_exchange_assoc,
      ← MonoidalCategory.whiskerLeft_comp_assoc, ← MonoidalCategory.whiskerLeft_comp_assoc,
      assoc, counit_naturality, left_triangle_components_assoc, MonoidalCategory.whiskerRight_id,
      assoc, assoc, Iso.inv_hom_id, comp_id, right_unitality_hom_assoc])


lemma rightAdjointLaxMonoidal_ε :
    letI := adj.rightAdjointLaxMonoidal
    ε G = adj.homEquiv _ _ (η F) := rfl


lemma rightAdjointLaxMonoidal_μ (X Y : D) :
    letI := adj.rightAdjointLaxMonoidal
    μ G X Y = adj.homEquiv _ _ (δ F _ _ ≫ (adj.counit.app X ⊗ adj.counit.app Y)) := rfl


/-- When `adj : F ⊣ G` is an adjunction, with `F` oplax monoidal and `G` monoidal,
this typeclass expresses compatibilities between the adjunction and the (op)lax
monoidal structures. -/
class IsMonoidal [G.LaxMonoidal] : Prop where
  leftAdjoint_ε : ε G = adj.homEquiv _ _ (η F) := by aesop_cat
  leftAdjoint_μ (X Y : D) :
    μ G X Y = adj.homEquiv _ _ (δ F _ _ ≫ (adj.counit.app X ⊗ adj.counit.app Y)) := by aesop_cat


instance :
    letI := adj.rightAdjointLaxMonoidal
    adj.IsMonoidal := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
    F✝ : CategoryTheory.Functor C D
    G✝ : CategoryTheory.Functor D E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : F.OplaxMonoidal
    ⊢ adj.IsMonoidal
  -/
  letI := adj.rightAdjointLaxMonoidal
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
    F✝ : CategoryTheory.Functor C D
    G✝ : CategoryTheory.Functor D E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : F.OplaxMonoidal
    this : G.LaxMonoidal := adj.rightAdjointLaxMonoidal
    ⊢ adj.IsMonoidal
  -/
  constructor
    /-
      case leftAdjoint_ε
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      this : G.LaxMonoidal := adj.rightAdjointLaxMonoidal
      ⊢ autoParam (Eq (CategoryTheory.Functor.LaxMonoidal.ε G) ((adj.homEquiv Catego …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case leftAdjoint_μ
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      this : G.LaxMonoidal := adj.rightAdjointLaxMonoidal
      ⊢ autoParam (∀ (X Y : D), Eq (CategoryTheory.Functor.LaxMonoidal.μ G X Y) ((ad …
    -/
  · intro _ _
    /-
      case leftAdjoint_μ
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁴ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : F.OplaxMonoidal
      this : G.LaxMonoidal := adj.rightAdjointLaxMonoidal
      X✝ Y✝ : D
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ G X✝ Y✝) ((adj.homEquiv (CategoryTh …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[reassoc]
lemma unit_app_unit_comp_map_η : adj.unit.app (𝟙_ C) ≫ G.map (η F) = ε G :=
  Adjunction.IsMonoidal.leftAdjoint_ε.symm


@[reassoc]
lemma unit_app_tensor_comp_map_δ (X Y : C) :
    adj.unit.app (X ⊗ Y) ≫ G.map (δ F X Y) = (adj.unit.app X ⊗ adj.unit.app Y) ≫ μ G _ _ := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.OplaxMonoidal
    inst✝¹ : G.LaxMonoidal
    inst✝ : adj.IsMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
  -/
  rw [IsMonoidal.leftAdjoint_μ (adj := adj), homEquiv_unit]
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.OplaxMonoidal
    inst✝¹ : G.LaxMonoidal
    inst✝ : adj.IsMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (CategoryTheory.Monoida …
  -/
  dsimp
  simp only [← adj.unit_naturality_assoc, ← Functor.map_comp, ← δ_natural_assoc,
    ← tensor_comp, left_triangle_components, tensorHom_id, id_whiskerRight, comp_id]


@[reassoc]
lemma map_ε_comp_counit_app_unit : F.map (ε G) ≫ adj.counit.app (𝟙_ D) = η F := by
  rw [IsMonoidal.leftAdjoint_ε (adj := adj), homEquiv_unit, map_comp,
    assoc, counit_naturality, left_triangle_components_assoc]


@[reassoc]
lemma map_μ_comp_counit_app_tensor (X Y : D) :
    F.map (μ G X Y) ≫ adj.counit.app (X ⊗ Y) =
      δ F _ _ ≫ (adj.counit.app X ⊗ adj.counit.app Y) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.OplaxMonoidal
    inst✝¹ : G.LaxMonoidal
    inst✝ : adj.IsMonoidal
    X Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Functor.LaxMon …
  -/
  rw [IsMonoidal.leftAdjoint_μ (adj := adj), homEquiv_unit]
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.OplaxMonoidal
    inst✝¹ : G.LaxMonoidal
    inst✝ : adj.IsMonoidal
    X Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : (Adjunction.id (C := C)).IsMonoidal where
                      /-
                        C : Type u₁
                        inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
                        inst✝⁸ : CategoryTheory.MonoidalCategory C
                        D : Type u₂
                        inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
                        inst✝⁶ : CategoryTheory.MonoidalCategory D
                        E : Type u₃
                        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                        inst✝⁴ : CategoryTheory.MonoidalCategory E
                        C' : Type u₁'
                        inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
                        F✝ : CategoryTheory.Functor C D
                        G✝ : CategoryTheory.Functor D E
                        F : CategoryTheory.Functor C D
                        G : CategoryTheory.Functor D C
                        adj : CategoryTheory.Adjunction F G
                        inst✝² : F.OplaxMonoidal
                        inst✝¹ : G.LaxMonoidal
                        inst✝ : adj.IsMonoidal
                        ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε (CategoryTheory.Functor.id C)) ((Ca …
                      -/
  leftAdjoint_ε := by simp [id, homEquiv]
                      /-
                        🎉 no goals
                      -/
                      /-
                        C : Type u₁
                        inst✝⁹ : CategoryTheory.Category.{v₁, u₁} C
                        inst✝⁸ : CategoryTheory.MonoidalCategory C
                        D : Type u₂
                        inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
                        inst✝⁶ : CategoryTheory.MonoidalCategory D
                        E : Type u₃
                        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                        inst✝⁴ : CategoryTheory.MonoidalCategory E
                        C' : Type u₁'
                        inst✝³ : CategoryTheory.Category.{v₁', u₁'} C'
                        F✝ : CategoryTheory.Functor C D
                        G✝ : CategoryTheory.Functor D E
                        F : CategoryTheory.Functor C D
                        G : CategoryTheory.Functor D C
                        adj : CategoryTheory.Adjunction F G
                        inst✝² : F.OplaxMonoidal
                        inst✝¹ : G.LaxMonoidal
                        inst✝ : adj.IsMonoidal
                        ⊢ ∀ (X Y : C), Eq (CategoryTheory.Functor.LaxMonoidal.μ (CategoryTheory.Functo …
                      -/
  leftAdjoint_μ := by simp [id, homEquiv]
                      /-
                        🎉 no goals
                      -/


instance isMonoidal_comp {F' : D ⥤ E} {G' : E ⥤ D} (adj' : F' ⊣ G')
  [F'.OplaxMonoidal] [G'.LaxMonoidal] [adj'.IsMonoidal] : (adj.comp adj').IsMonoidal where
  leftAdjoint_ε := by
    /-
      C : Type u₁
      inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹¹ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁹ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁷ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁵ : F.OplaxMonoidal
      inst✝⁴ : G.LaxMonoidal
      inst✝³ : adj.IsMonoidal
      F' : CategoryTheory.Functor D E
      G' : CategoryTheory.Functor E D
      adj' : CategoryTheory.Adjunction F' G'
      inst✝² : F'.OplaxMonoidal
      inst✝¹ : G'.LaxMonoidal
      inst✝ : adj'.IsMonoidal
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε (G'.comp G)) (((adj.comp adj').homE …
    -/
    dsimp [homEquiv]
    rw [← adj.unit_app_unit_comp_map_η, ← adj'.unit_app_unit_comp_map_η,
      assoc, comp_unit_app, assoc, ← Functor.map_comp,
      ← adj'.unit_naturality_assoc, ← map_comp, ← map_comp]
  leftAdjoint_μ X Y := by
    /-
      C : Type u₁
      inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹¹ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁹ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁷ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁵ : F.OplaxMonoidal
      inst✝⁴ : G.LaxMonoidal
      inst✝³ : adj.IsMonoidal
      F' : CategoryTheory.Functor D E
      G' : CategoryTheory.Functor E D
      adj' : CategoryTheory.Adjunction F' G'
      inst✝² : F'.OplaxMonoidal
      inst✝¹ : G'.LaxMonoidal
      inst✝ : adj'.IsMonoidal
      X Y : E
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ (G'.comp G) X Y) (((adj.comp adj'). …
    -/
    apply ((adj.comp adj').homEquiv _ _).symm.injective
    /-
      case a
      C : Type u₁
      inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹¹ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁹ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁷ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁵ : F.OplaxMonoidal
      inst✝⁴ : G.LaxMonoidal
      inst✝³ : adj.IsMonoidal
      F' : CategoryTheory.Functor D E
      G' : CategoryTheory.Functor E D
      adj' : CategoryTheory.Adjunction F' G'
      inst✝² : F'.OplaxMonoidal
      inst✝¹ : G'.LaxMonoidal
      inst✝ : adj'.IsMonoidal
      X Y : E
      ⊢ Eq (((adj.comp adj').homEquiv (CategoryTheory.MonoidalCategoryStruct.tensorO …
    -/
    erw [Equiv.symm_apply_apply]
    /-
      case a
      C : Type u₁
      inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹¹ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁹ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁷ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁵ : F.OplaxMonoidal
      inst✝⁴ : G.LaxMonoidal
      inst✝³ : adj.IsMonoidal
      F' : CategoryTheory.Functor D E
      G' : CategoryTheory.Functor E D
      adj' : CategoryTheory.Adjunction F' G'
      inst✝² : F'.OplaxMonoidal
      inst✝¹ : G'.LaxMonoidal
      inst✝ : adj'.IsMonoidal
      X Y : E
      ⊢ Eq (((adj.comp adj').homEquiv (CategoryTheory.MonoidalCategoryStruct.tensorO …
    -/
    dsimp [homEquiv]
    /-
      case a
      C : Type u₁
      inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹¹ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁹ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁷ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁵ : F.OplaxMonoidal
      inst✝⁴ : G.LaxMonoidal
      inst✝³ : adj.IsMonoidal
      F' : CategoryTheory.Functor D E
      G' : CategoryTheory.Functor E D
      adj' : CategoryTheory.Adjunction F' G'
      inst✝² : F'.OplaxMonoidal
      inst✝¹ : G'.LaxMonoidal
      inst✝ : adj'.IsMonoidal
      X Y : E
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.map (F.map (CategoryTheory.Catego …
    -/
    rw [comp_counit_app, comp_counit_app, comp_counit_app, assoc, tensor_comp, δ_natural_assoc]
    /-
      case a
      C : Type u₁
      inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹¹ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁹ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁷ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
      F✝ : CategoryTheory.Functor C D
      G✝ : CategoryTheory.Functor D E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝⁵ : F.OplaxMonoidal
      inst✝⁴ : G.LaxMonoidal
      inst✝³ : adj.IsMonoidal
      F' : CategoryTheory.Functor D E
      G' : CategoryTheory.Functor E D
      adj' : CategoryTheory.Adjunction F' G'
      inst✝² : F'.OplaxMonoidal
      inst✝¹ : G'.LaxMonoidal
      inst✝ : adj'.IsMonoidal
      X Y : E
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.map (F.map (CategoryTheory.Catego …
    -/
    dsimp
    rw [← adj'.map_μ_comp_counit_app_tensor, ← map_comp_assoc, ← map_comp_assoc,
      ← map_comp_assoc, ← adj.map_μ_comp_counit_app_tensor, assoc,
      F.map_comp_assoc, counit_naturality]


instance [e.inverse.Monoidal] : e.symm.functor.Monoidal := inferInstanceAs (e.inverse.Monoidal)

instance [e.functor.Monoidal] : e.symm.inverse.Monoidal := inferInstanceAs (e.functor.Monoidal)


/-- If a monoidal functor `F` is an equivalence of categories then its inverse is also monoidal. -/
noncomputable def inverseMonoidal [e.functor.Monoidal] : e.inverse.Monoidal := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    e : CategoryTheory.Equivalence C D
    inst✝ : e.functor.Monoidal
    ⊢ e.inverse.Monoidal
  -/
  letI := e.toAdjunction.rightAdjointLaxMonoidal
  have : IsIso (LaxMonoidal.ε e.inverse) := by
    simp only [this, Adjunction.rightAdjointLaxMonoidal_ε, Adjunction.homEquiv_unit]
    infer_instance
  have : ∀ (X Y : D), IsIso (LaxMonoidal.μ e.inverse X Y) := fun X Y ↦ by
    simp only [Adjunction.rightAdjointLaxMonoidal_μ, Adjunction.homEquiv_unit]
    infer_instance
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    inst✝² : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝¹ : CategoryTheory.Category.{v₁', u₁'} C'
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    e : CategoryTheory.Equivalence C D
    inst✝ : e.functor.Monoidal
    this✝¹ : e.inverse.LaxMonoidal := e.toAdjunction.rightAdjointLaxMonoidal
    this✝ : CategoryTheory.IsIso (CategoryTheory.Functor.LaxMonoidal.ε e.inverse)
    this : ∀ (X Y : D), CategoryTheory.IsIso (CategoryTheory.Functor.LaxMonoidal.μ …
    ⊢ e.inverse.Monoidal
  -/
  apply Monoidal.ofLaxMonoidal
  /-
    🎉 no goals
  -/


/-- An equivalence of categories involving monoidal functors is monoidal if the underlying
adjunction satisfies certain compatibilities with respect to the monoidal functor data. -/
abbrev IsMonoidal [e.functor.Monoidal] [e.inverse.Monoidal] : Prop := e.toAdjunction.IsMonoidal


@[reassoc]
lemma unitIso_hom_app_comp_inverse_map_η_functor :
    e.unitIso.hom.app (𝟙_ C) ≫ e.inverse.map (η e.functor) = ε e.inverse :=
  e.toAdjunction.unit_app_unit_comp_map_η


@[reassoc]
lemma unitIso_hom_app_tensor_comp_inverse_map_δ_functor (X Y : C) :
    e.unitIso.hom.app (X ⊗ Y) ≫ e.inverse.map (δ e.functor X Y) =
      (e.unitIso.hom.app X ⊗ e.unitIso.hom.app Y) ≫ μ e.inverse _ _ :=
  e.toAdjunction.unit_app_tensor_comp_map_δ X Y


@[reassoc]
lemma functor_map_ε_inverse_comp_counitIso_hom_app :
    e.functor.map (ε e.inverse) ≫ e.counitIso.hom.app (𝟙_ D) = η e.functor :=
  e.toAdjunction.map_ε_comp_counit_app_unit


@[reassoc]
lemma functor_map_μ_inverse_comp_counitIso_hom_app_tensor (X Y : D) :
    e.functor.map (μ e.inverse X Y) ≫ e.counitIso.hom.app (X ⊗ Y) =
      δ e.functor _ _ ≫ (e.counitIso.hom.app X ⊗ e.counitIso.hom.app Y) :=
  e.toAdjunction.map_μ_comp_counit_app_tensor X Y


@[reassoc]
lemma counitIso_inv_app_comp_functor_map_η_inverse :
    e.counitIso.inv.app (𝟙_ D) ≫ e.functor.map (η e.inverse) = ε e.functor := by
  rw [← cancel_epi (η e.functor), Monoidal.η_ε, ← functor_map_ε_inverse_comp_counitIso_hom_app,
    Category.assoc, Iso.hom_inv_id_app_assoc, Monoidal.map_ε_η]


@[reassoc]
lemma counitIso_inv_app_tensor_comp_functor_map_δ_inverse (X Y : C) :
    e.counitIso.inv.app (e.functor.obj X ⊗ e.functor.obj Y) ≫
      e.functor.map (δ e.inverse (e.functor.obj X) (e.functor.obj Y)) =
      μ e.functor X Y ≫ e.functor.map (e.unitIso.hom.app X ⊗ e.unitIso.hom.app Y) := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    e : CategoryTheory.Equivalence C D
    inst✝² : e.functor.Monoidal
    inst✝¹ : e.inverse.Monoidal
    inst✝ : e.IsMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.counitIso.inv.app (CategoryTheory. …
  -/
  rw [← cancel_epi (δ e.functor _ _), Monoidal.δ_μ_assoc]
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.MonoidalCategory D
    e : CategoryTheory.Equivalence C D
    inst✝² : e.functor.Monoidal
    inst✝¹ : e.inverse.Monoidal
    inst✝ : e.IsMonoidal
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMonoidal …
  -/
  apply e.inverse.map_injective
  simp [← cancel_epi (e.unitIso.hom.app (X ⊗ Y)), Functor.map_comp,
    unitIso_hom_app_tensor_comp_inverse_map_δ_functor_assoc]


instance : (refl (C := C)).functor.Monoidal := inferInstanceAs (𝟭 C).Monoidal

instance : (refl (C := C)).inverse.Monoidal := inferInstanceAs (𝟭 C).Monoidal


/-- The obvious auto-equivalence of a monoidal category is monoidal. -/
instance isMonoidal_refl : (Equivalence.refl (C := C)).IsMonoidal :=
  inferInstanceAs (Adjunction.id (C := C)).IsMonoidal


/-- The inverse of a monoidal category equivalence is also a monoidal category equivalence. -/
instance isMonoidal_symm [e.inverse.Monoidal] [e.IsMonoidal] :
    e.symm.IsMonoidal where
  leftAdjoint_ε := by
    /-
      C : Type u₁
      inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁰ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁸ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁷ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁶ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁵ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : CategoryTheory.Equivalence C D
      inst✝⁴ : e.functor.Monoidal
      inst✝³ : e.inverse.Monoidal
      inst✝² : e.IsMonoidal
      inst✝¹ : e.inverse.Monoidal
      inst✝ : e.IsMonoidal
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε e.symm.inverse) ((e.symm.toAdjuncti …
    -/
    simp only [toAdjunction, Adjunction.homEquiv_unit]
    /-
      C : Type u₁
      inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁰ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁸ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁷ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁶ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁵ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : CategoryTheory.Equivalence C D
      inst✝⁴ : e.functor.Monoidal
      inst✝³ : e.inverse.Monoidal
      inst✝² : e.IsMonoidal
      inst✝¹ : e.inverse.Monoidal
      inst✝ : e.IsMonoidal
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε e.symm.inverse) (CategoryTheory.Cat …
    -/
    dsimp [symm]
    /-
      C : Type u₁
      inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁰ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁸ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁷ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁶ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁵ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : CategoryTheory.Equivalence C D
      inst✝⁴ : e.functor.Monoidal
      inst✝³ : e.inverse.Monoidal
      inst✝² : e.IsMonoidal
      inst✝¹ : e.inverse.Monoidal
      inst✝ : e.IsMonoidal
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.ε e.functor) (CategoryTheory.Category …
    -/
    rw [counitIso_inv_app_comp_functor_map_η_inverse]
    /-
      🎉 no goals
    -/
  leftAdjoint_μ X Y := by
    /-
      C : Type u₁
      inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁰ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁸ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁷ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁶ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁵ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : CategoryTheory.Equivalence C D
      inst✝⁴ : e.functor.Monoidal
      inst✝³ : e.inverse.Monoidal
      inst✝² : e.IsMonoidal
      inst✝¹ : e.inverse.Monoidal
      inst✝ : e.IsMonoidal
      X Y : C
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ e.symm.inverse X Y) ((e.symm.toAdju …
    -/
    simp only [toAdjunction, Adjunction.homEquiv_unit]
    /-
      C : Type u₁
      inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁰ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁸ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁷ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁶ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁵ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : CategoryTheory.Equivalence C D
      inst✝⁴ : e.functor.Monoidal
      inst✝³ : e.inverse.Monoidal
      inst✝² : e.IsMonoidal
      inst✝¹ : e.inverse.Monoidal
      inst✝ : e.IsMonoidal
      X Y : C
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ e.symm.inverse X Y) (CategoryTheory …
    -/
    dsimp [symm]
    rw [map_comp, counitIso_inv_app_tensor_comp_functor_map_δ_inverse_assoc,
      ← Functor.map_comp, ← tensor_comp, Iso.hom_inv_id_app, Iso.hom_inv_id_app]
    /-
      C : Type u₁
      inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁰ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁸ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁷ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁶ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁵ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : CategoryTheory.Equivalence C D
      inst✝⁴ : e.functor.Monoidal
      inst✝³ : e.inverse.Monoidal
      inst✝² : e.IsMonoidal
      inst✝¹ : e.inverse.Monoidal
      inst✝ : e.IsMonoidal
      X Y : C
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ e.functor X Y) (CategoryTheory.Cate …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁰ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁸ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁷ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁶ : CategoryTheory.MonoidalCategory E
      C' : Type u₁'
      inst✝⁵ : CategoryTheory.Category.{v₁', u₁'} C'
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : CategoryTheory.Equivalence C D
      inst✝⁴ : e.functor.Monoidal
      inst✝³ : e.inverse.Monoidal
      inst✝² : e.IsMonoidal
      inst✝¹ : e.inverse.Monoidal
      inst✝ : e.IsMonoidal
      X Y : C
      ⊢ Eq (CategoryTheory.Functor.LaxMonoidal.μ e.functor X Y) (CategoryTheory.Cate …
    -/
    rw [tensorHom_id, id_whiskerRight, map_id, comp_id]
    /-
      🎉 no goals
    -/


instance [e'.functor.Monoidal] : (e.trans e').functor.Monoidal :=
  inferInstanceAs (e.functor ⋙ e'.functor).Monoidal


instance [e'.inverse.Monoidal] : (e.trans e').inverse.Monoidal :=
  inferInstanceAs (e'.inverse ⋙ e.inverse).Monoidal


/-- The composition of two monoidal category equivalences is monoidal. -/
instance isMonoidal_trans [e'.functor.Monoidal] [e'.inverse.Monoidal] [e'.IsMonoidal] :
    (e.trans e').IsMonoidal := by
  /-
    C : Type u₁
    inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹¹ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁹ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    inst✝⁷ : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    e : CategoryTheory.Equivalence C D
    inst✝⁵ : e.functor.Monoidal
    inst✝⁴ : e.inverse.Monoidal
    inst✝³ : e.IsMonoidal
    e' : CategoryTheory.Equivalence D E
    inst✝² : e'.functor.Monoidal
    inst✝¹ : e'.inverse.Monoidal
    inst✝ : e'.IsMonoidal
    ⊢ (e.trans e').IsMonoidal
  -/
  dsimp [Equivalence.IsMonoidal]
  /-
    C : Type u₁
    inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹¹ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁹ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    inst✝⁷ : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    e : CategoryTheory.Equivalence C D
    inst✝⁵ : e.functor.Monoidal
    inst✝⁴ : e.inverse.Monoidal
    inst✝³ : e.IsMonoidal
    e' : CategoryTheory.Equivalence D E
    inst✝² : e'.functor.Monoidal
    inst✝¹ : e'.inverse.Monoidal
    inst✝ : e'.IsMonoidal
    ⊢ (e.trans e').toAdjunction.IsMonoidal
  -/
  rw [trans_toAdjunction]
  /-
    C : Type u₁
    inst✝¹² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹¹ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁹ : CategoryTheory.MonoidalCategory D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    inst✝⁷ : CategoryTheory.MonoidalCategory E
    C' : Type u₁'
    inst✝⁶ : CategoryTheory.Category.{v₁', u₁'} C'
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    e : CategoryTheory.Equivalence C D
    inst✝⁵ : e.functor.Monoidal
    inst✝⁴ : e.inverse.Monoidal
    inst✝³ : e.IsMonoidal
    e' : CategoryTheory.Equivalence D E
    inst✝² : e'.functor.Monoidal
    inst✝¹ : e'.inverse.Monoidal
    inst✝ : e'.IsMonoidal
    ⊢ (e.toAdjunction.comp e'.toAdjunction).IsMonoidal
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Bundled version of lax monoidal functors. This type is equipped with a category
structure in `CategoryTheory.Monoidal.NaturalTransformation`. -/
structure LaxMonoidalFunctor extends C ⥤ D where
  laxMonoidal : toFunctor.LaxMonoidal := by infer_instance


/-- Constructor for `LaxMonoidalFunctor C D`. -/
@[simps toFunctor]
def of (F : C ⥤ D) [F.LaxMonoidal] : LaxMonoidalFunctor C D where
  toFunctor := F


