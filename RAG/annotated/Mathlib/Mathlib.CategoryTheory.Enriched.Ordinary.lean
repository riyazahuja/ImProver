/-- An enriched ordinary category is a category `C` that is also enriched
over a category `V` in such a way that morphisms `X ⟶ Y` in `C` identify
to morphisms `𝟙_ V ⟶ (X ⟶[V] Y)` in `V`. -/
class EnrichedOrdinaryCategory extends EnrichedCategory V C where
  /-- morphisms `X ⟶ Y` in the category identify morphisms
    `𝟙_ V ⟶ (X ⟶[V] Y)` in `V` -/
  homEquiv {X Y : C} : (X ⟶ Y) ≃ (𝟙_ V ⟶ (X ⟶[V] Y))
  homEquiv_id (X : C) : homEquiv (𝟙 X) = eId V X := by aesop_cat
  homEquiv_comp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    homEquiv (f ≫ g) = (λ_ _).inv ≫ (homEquiv f ⊗ homEquiv g) ≫
      eComp V X Y Z := by aesop_cat


/-- The bijection `(X ⟶ Y) ≃ (𝟙_ V ⟶ (X ⟶[V] Y))` given by a
`EnrichedOrdinaryCategory` instance. -/
def eHomEquiv {X Y : C} : (X ⟶ Y) ≃ (𝟙_ V ⟶ (X ⟶[V] Y)) :=
  EnrichedOrdinaryCategory.homEquiv


@[simp]
lemma eHomEquiv_id (X : C) : eHomEquiv V (𝟙 X) = eId V X :=
  EnrichedOrdinaryCategory.homEquiv_id _


@[reassoc]
lemma eHomEquiv_comp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    eHomEquiv V (f ≫ g) = (λ_ _).inv ≫ (eHomEquiv V f ⊗ eHomEquiv V g) ≫ eComp V X Y Z :=
  EnrichedOrdinaryCategory.homEquiv_comp _ _


/-- The morphism `(X' ⟶[V] Y) ⟶ (X ⟶[V] Y)` induced by a morphism `X ⟶ X'`. -/
def eHomWhiskerRight {X X' : C} (f : X ⟶ X') (Y : C) :
    (X' ⟶[V] Y) ⟶ (X ⟶[V] Y) :=
  (λ_ _).inv ≫ eHomEquiv V f ▷ _ ≫ eComp V X X' Y


@[simp]
lemma eHomWhiskerRight_id (X Y : C) : eHomWhiskerRight V (𝟙 X) Y = 𝟙 _ := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y : C
    ⊢ Eq (CategoryTheory.eHomWhiskerRight V (CategoryTheory.CategoryStruct.id X) Y …
  -/
  simp [eHomWhiskerRight]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma eHomWhiskerRight_comp {X X' X'' : C} (f : X ⟶ X') (f' : X' ⟶ X'') (Y : C) :
    eHomWhiskerRight V (f ≫ f') Y = eHomWhiskerRight V f' Y ≫ eHomWhiskerRight V f Y := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X X' X'' : C
    f : Quiver.Hom X X'
    f' : Quiver.Hom X' X''
    Y : C
    ⊢ Eq (CategoryTheory.eHomWhiskerRight V (CategoryTheory.CategoryStruct.comp f  …
  -/
  dsimp [eHomWhiskerRight]
  rw [assoc, assoc, eHomEquiv_comp, comp_whiskerRight_assoc, comp_whiskerRight_assoc, ← e_assoc',
    tensorHom_def', comp_whiskerRight_assoc, id_whiskerLeft, comp_whiskerRight_assoc,
    ← comp_whiskerRight_assoc, Iso.inv_hom_id, id_whiskerRight_assoc,
    comp_whiskerRight_assoc, leftUnitor_inv_whiskerRight_assoc,
    ← associator_inv_naturality_left_assoc, Iso.inv_hom_id_assoc,
    ← whisker_exchange_assoc, id_whiskerLeft_assoc, Iso.inv_hom_id_assoc]


/-- Whiskering commutes with the enriched composition. -/
@[reassoc]
lemma eComp_eHomWhiskerRight {X X' : C} (f : X ⟶ X') (Y Z : C) :
    eComp V X' Y Z ≫ eHomWhiskerRight V f Z =
      eHomWhiskerRight V f Y ▷ _ ≫ eComp V X Y Z := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X X' : C
    f : Quiver.Hom X X'
    Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eComp V X' Y Z) (Cate …
  -/
  dsimp [eHomWhiskerRight]
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X X' : C
    f : Quiver.Hom X X'
    Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eComp V X' Y Z) (Cate …
  -/
  rw [leftUnitor_inv_naturality_assoc, whisker_exchange_assoc]
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X X' : C
    f : Quiver.Hom X X'
    Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [e_assoc']
  /-
    🎉 no goals
  -/


/-- The morphism `(X ⟶[V] Y) ⟶ (X ⟶[V] Y')` induced by a morphism `Y ⟶ Y'`. -/
def eHomWhiskerLeft (X : C) {Y Y' : C} (g : Y ⟶ Y') :
    (X ⟶[V] Y) ⟶ (X ⟶[V] Y') :=
  (ρ_ _).inv ≫ _ ◁ eHomEquiv V g ≫ eComp V X Y Y'


@[simp]
lemma eHomWhiskerLeft_id (X Y : C) : eHomWhiskerLeft V X (𝟙 Y) = 𝟙 _ := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y : C
    ⊢ Eq (CategoryTheory.eHomWhiskerLeft V X (CategoryTheory.CategoryStruct.id Y)) …
  -/
  simp [eHomWhiskerLeft]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma eHomWhiskerLeft_comp (X : C) {Y Y' Y'' : C} (g : Y ⟶ Y') (g' : Y' ⟶ Y'') :
    eHomWhiskerLeft V X (g ≫ g') = eHomWhiskerLeft V X g ≫ eHomWhiskerLeft V X g' := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y Y' Y'' : C
    g : Quiver.Hom Y Y'
    g' : Quiver.Hom Y' Y''
    ⊢ Eq (CategoryTheory.eHomWhiskerLeft V X (CategoryTheory.CategoryStruct.comp g …
  -/
  dsimp [eHomWhiskerLeft]
  rw [assoc, assoc, eHomEquiv_comp, MonoidalCategory.whiskerLeft_comp_assoc,
    MonoidalCategory.whiskerLeft_comp_assoc, ← e_assoc, tensorHom_def,
    MonoidalCategory.whiskerRight_id_assoc, MonoidalCategory.whiskerLeft_comp_assoc,
    MonoidalCategory.whiskerLeft_comp_assoc, MonoidalCategory.whiskerLeft_comp_assoc,
    whiskerLeft_rightUnitor_assoc, whiskerLeft_rightUnitor_inv_assoc,
    triangle_assoc_comp_left_inv_assoc, MonoidalCategory.whiskerRight_id_assoc,
    Iso.hom_inv_id_assoc, Iso.inv_hom_id_assoc,
    associator_inv_naturality_right_assoc, Iso.hom_inv_id_assoc,
    whisker_exchange_assoc, MonoidalCategory.whiskerRight_id_assoc, Iso.inv_hom_id_assoc]


/-- Whiskering commutes with the enriched composition. -/
@[reassoc]
lemma eComp_eHomWhiskerLeft (X Y : C) {Z Z' : C} (g : Z ⟶ Z') :
    eComp V X Y Z ≫ eHomWhiskerLeft V X g =
      _ ◁ eHomWhiskerLeft V Y g ≫ eComp V X Y Z' := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y Z Z' : C
    g : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eComp V X Y Z) (Categ …
  -/
  dsimp [eHomWhiskerLeft]
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y Z Z' : C
    g : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eComp V X Y Z) (Categ …
  -/
  rw [rightUnitor_inv_naturality_assoc, ← whisker_exchange_assoc]
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y Z Z' : C
    g : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [e_assoc']
  /-
    🎉 no goals
  -/


/-- Given an isomorphism `α : Y ≅ Y₁` in C, the enriched composition map
`eComp V X Y Z : (X ⟶[V] Y) ⊗ (Y ⟶[V] Z) ⟶ (X ⟶[V] Z)` factors through the `V`
object `(X ⟶[V] Y₁) ⊗ (Y₁ ⟶[V] Z)` via the map defined by whiskering in the
middle with `α.hom` and `α.inv`. -/
@[reassoc]
lemma eHom_whisker_cancel {X Y Y₁ Z : C} (α : Y  ≅ Y₁) :
    eHomWhiskerLeft V X α.hom ▷ _ ≫ _ ◁ eHomWhiskerRight V α.inv Z ≫
      eComp V X Y₁ Z = eComp V X Y Z := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y Y₁ Z : C
    α : CategoryTheory.Iso Y Y₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [eHomWhiskerLeft, eHomWhiskerRight]
  simp only [MonoidalCategory.whiskerLeft_comp_assoc, whisker_assoc_symm,
    triangle_assoc_comp_left_inv_assoc, e_assoc', assoc]
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y Y₁ Z : C
    α : CategoryTheory.Iso Y Y₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [← comp_whiskerRight_assoc]
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y Y₁ Z : C
    α : CategoryTheory.Iso Y Y₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  change (eHomWhiskerLeft V X α.hom ≫ eHomWhiskerLeft V X α.inv) ▷ _ ≫ _ = _
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X Y Y₁ Z : C
    α : CategoryTheory.Iso Y Y₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [← eHomWhiskerLeft_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma eHom_whisker_cancel_inv {X Y Y₁ Z : C} (α : Y  ≅ Y₁) :
    eHomWhiskerLeft V X α.inv ▷ _ ≫ _ ◁ eHomWhiskerRight V α.hom Z ≫
      eComp V X Y Z = eComp V X Y₁ Z := eHom_whisker_cancel V α.symm


@[reassoc]
lemma eHom_whisker_exchange {X X' Y Y' : C} (f : X ⟶ X') (g : Y ⟶ Y') :
    eHomWhiskerLeft V X' g ≫ eHomWhiskerRight V f Y' =
      eHomWhiskerRight V f Y ≫ eHomWhiskerLeft V X g := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X X' Y Y' : C
    f : Quiver.Hom X X'
    g : Quiver.Hom Y Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eHomWhiskerLeft V X'  …
  -/
  dsimp [eHomWhiskerLeft, eHomWhiskerRight]
  rw [assoc, assoc, assoc, assoc, leftUnitor_inv_naturality_assoc,
    whisker_exchange_assoc, ← e_assoc, leftUnitor_tensor_inv_assoc,
    associator_inv_naturality_left_assoc, Iso.hom_inv_id_assoc,
    ← comp_whiskerRight_assoc, whisker_exchange_assoc,
    MonoidalCategory.whiskerRight_id_assoc, assoc, Iso.inv_hom_id_assoc,
    whisker_exchange_assoc, MonoidalCategory.whiskerRight_id_assoc, Iso.inv_hom_id_assoc]


variable (C) in
/-- The bifunctor `Cᵒᵖ ⥤ C ⥤ V` which sends `X : Cᵒᵖ` and `Y : C` to `X ⟶[V] Y`. -/
@[simps]
def eHomFunctor : Cᵒᵖ ⥤ C ⥤ V where
  obj X :=
    { obj := fun Y => X.unop ⟶[V] Y
      map := fun φ => eHomWhiskerLeft V X.unop φ }
  map φ :=
    { app := fun Y => eHomWhiskerRight V φ.unop Y }


instance ForgetEnrichment.EnrichedOrdinaryCategory {D : Type*} [EnrichedCategory V D] :
    EnrichedOrdinaryCategory V (ForgetEnrichment V D) where
  toEnrichedCategory := inferInstanceAs (EnrichedCategory V D)
  homEquiv := Equiv.refl _
  homEquiv_id _ := Category.id_comp _
  homEquiv_comp _ _ := Category.assoc _ _ _


