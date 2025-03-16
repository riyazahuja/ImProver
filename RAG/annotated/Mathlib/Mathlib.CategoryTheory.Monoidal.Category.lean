/-- Auxiliary structure to carry only the data fields of (and provide notation for)
`MonoidalCategory`. -/
class MonoidalCategoryStruct (C : Type u) [𝒞 : Category.{v} C] where
  /-- curried tensor product of objects -/
  tensorObj : C → C → C
  /-- left whiskering for morphisms -/
  whiskerLeft (X : C) {Y₁ Y₂ : C} (f : Y₁ ⟶ Y₂) : tensorObj X Y₁ ⟶ tensorObj X Y₂
  /-- right whiskering for morphisms -/
  whiskerRight {X₁ X₂ : C} (f : X₁ ⟶ X₂) (Y : C) : tensorObj X₁ Y ⟶ tensorObj X₂ Y
  /-- Tensor product of identity maps is the identity: `(𝟙 X₁ ⊗ 𝟙 X₂) = 𝟙 (X₁ ⊗ X₂)` -/
  -- By default, it is defined in terms of whiskerings.
  tensorHom {X₁ Y₁ X₂ Y₂ : C} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) : (tensorObj X₁ X₂ ⟶ tensorObj Y₁ Y₂) :=
    whiskerRight f X₂ ≫ whiskerLeft Y₁ g
  /-- The tensor unity in the monoidal structure `𝟙_ C` -/
  tensorUnit : C
  /-- The associator isomorphism `(X ⊗ Y) ⊗ Z ≃ X ⊗ (Y ⊗ Z)` -/
  associator : ∀ X Y Z : C, tensorObj (tensorObj X Y) Z ≅ tensorObj X (tensorObj Y Z)
  /-- The left unitor: `𝟙_ C ⊗ X ≃ X` -/
  leftUnitor : ∀ X : C, tensorObj tensorUnit X ≅ X
  /-- The right unitor: `X ⊗ 𝟙_ C ≃ X` -/
  rightUnitor : ∀ X : C, tensorObj X tensorUnit ≅ X


/-- Notation for `tensorObj`, the tensor product of objects in a monoidal category -/
scoped infixr:70 " ⊗ " => MonoidalCategoryStruct.tensorObj


/-- Notation for the `whiskerLeft` operator of monoidal categories -/
scoped infixr:81 " ◁ " => MonoidalCategoryStruct.whiskerLeft


/-- Notation for the `whiskerRight` operator of monoidal categories -/
scoped infixl:81 " ▷ " => MonoidalCategoryStruct.whiskerRight


/-- Notation for `tensorHom`, the tensor product of morphisms in a monoidal category -/
scoped infixr:70 " ⊗ " => MonoidalCategoryStruct.tensorHom


/-- Notation for `tensorUnit`, the two-sided identity of `⊗` -/
scoped notation "𝟙_ " C:max => (MonoidalCategoryStruct.tensorUnit : C)


open Lean PrettyPrinter.Delaborator SubExpr in
/-- Used to ensure that `𝟙_` notation is used, as the ascription makes this not automatic. -/
@[app_delab CategoryTheory.MonoidalCategoryStruct.tensorUnit]
def delabTensorUnit : Delab := whenPPOption getPPNotation <| withOverApp 3 do
  let e ← getExpr
  guard <| e.isAppOfArity ``MonoidalCategoryStruct.tensorUnit 3
  let C ← withNaryArg 0 delab
  `(𝟙_ $C)


/-- Notation for the monoidal `associator`: `(X ⊗ Y) ⊗ Z ≃ X ⊗ (Y ⊗ Z)` -/
scoped notation "α_" => MonoidalCategoryStruct.associator


/-- Notation for the `leftUnitor`: `𝟙_C ⊗ X ≃ X` -/
scoped notation "λ_" => MonoidalCategoryStruct.leftUnitor


/-- Notation for the `rightUnitor`: `X ⊗ 𝟙_C ≃ X` -/
scoped notation "ρ_" => MonoidalCategoryStruct.rightUnitor


/--
In a monoidal category, we can take the tensor product of objects, `X ⊗ Y` and of morphisms `f ⊗ g`.
Tensor product does not need to be strictly associative on objects, but there is a
specified associator, `α_ X Y Z : (X ⊗ Y) ⊗ Z ≅ X ⊗ (Y ⊗ Z)`. There is a tensor unit `𝟙_ C`,
with specified left and right unitor isomorphisms `λ_ X : 𝟙_ C ⊗ X ≅ X` and `ρ_ X : X ⊗ 𝟙_ C ≅ X`.
These associators and unitors satisfy the pentagon and triangle equations.

See <https://stacks.math.columbia.edu/tag/0FFK>.
-/
-- Porting note: The Mathport did not translate the temporary notation
class MonoidalCategory (C : Type u) [𝒞 : Category.{v} C] extends MonoidalCategoryStruct C where
  tensorHom_def {X₁ Y₁ X₂ Y₂ : C} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
    f ⊗ g = (f ▷ X₂) ≫ (Y₁ ◁ g) := by
      aesop_cat
  /-- Tensor product of identity maps is the identity: `(𝟙 X₁ ⊗ 𝟙 X₂) = 𝟙 (X₁ ⊗ X₂)` -/
  tensor_id : ∀ X₁ X₂ : C, 𝟙 X₁ ⊗ 𝟙 X₂ = 𝟙 (X₁ ⊗ X₂) := by aesop_cat
  /--
  Composition of tensor products is tensor product of compositions:
  `(f₁ ⊗ g₁) ∘ (f₂ ⊗ g₂) = (f₁ ∘ f₂) ⊗ (g₁ ⊗ g₂)`
  -/
  tensor_comp :
    ∀ {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (g₁ : Y₁ ⟶ Z₁) (g₂ : Y₂ ⟶ Z₂),
      (f₁ ≫ g₁) ⊗ (f₂ ≫ g₂) = (f₁ ⊗ f₂) ≫ (g₁ ⊗ g₂) := by
    aesop_cat
  whiskerLeft_id : ∀ (X Y : C), X ◁ 𝟙 Y = 𝟙 (X ⊗ Y) := by
    aesop_cat
  id_whiskerRight : ∀ (X Y : C), 𝟙 X ▷ Y = 𝟙 (X ⊗ Y) := by
    aesop_cat
  /-- Naturality of the associator isomorphism: `(f₁ ⊗ f₂) ⊗ f₃ ≃ f₁ ⊗ (f₂ ⊗ f₃)` -/
  associator_naturality :
    ∀ {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (f₃ : X₃ ⟶ Y₃),
      ((f₁ ⊗ f₂) ⊗ f₃) ≫ (α_ Y₁ Y₂ Y₃).hom = (α_ X₁ X₂ X₃).hom ≫ (f₁ ⊗ (f₂ ⊗ f₃)) := by
    aesop_cat
  /--
  Naturality of the left unitor, commutativity of `𝟙_ C ⊗ X ⟶ 𝟙_ C ⊗ Y ⟶ Y` and `𝟙_ C ⊗ X ⟶ X ⟶ Y`
  -/
  leftUnitor_naturality :
    ∀ {X Y : C} (f : X ⟶ Y), 𝟙_ _ ◁ f ≫ (λ_ Y).hom = (λ_ X).hom ≫ f := by
    aesop_cat
  /--
  Naturality of the right unitor: commutativity of `X ⊗ 𝟙_ C ⟶ Y ⊗ 𝟙_ C ⟶ Y` and `X ⊗ 𝟙_ C ⟶ X ⟶ Y`
  -/
  rightUnitor_naturality :
    ∀ {X Y : C} (f : X ⟶ Y), f ▷ 𝟙_ _ ≫ (ρ_ Y).hom = (ρ_ X).hom ≫ f := by
    aesop_cat
  /--
  The pentagon identity relating the isomorphism between `X ⊗ (Y ⊗ (Z ⊗ W))` and `((X ⊗ Y) ⊗ Z) ⊗ W`
  -/
  pentagon :
    ∀ W X Y Z : C,
      (α_ W X Y).hom ▷ Z ≫ (α_ W (X ⊗ Y) Z).hom ≫ W ◁ (α_ X Y Z).hom =
        (α_ (W ⊗ X) Y Z).hom ≫ (α_ W X (Y ⊗ Z)).hom := by
    aesop_cat
  /--
  The identity relating the isomorphisms between `X ⊗ (𝟙_ C ⊗ Y)`, `(X ⊗ 𝟙_ C) ⊗ Y` and `X ⊗ Y`
  -/
  triangle :
    ∀ X Y : C, (α_ X (𝟙_ _) Y).hom ≫ X ◁ (λ_ Y).hom = (ρ_ X).hom ▷ Y := by
    aesop_cat


attribute [reassoc] MonoidalCategory.tensorHom_def

attribute [reassoc, simp] MonoidalCategory.whiskerLeft_id

attribute [reassoc, simp] MonoidalCategory.id_whiskerRight

attribute [reassoc] MonoidalCategory.tensor_comp

attribute [reassoc] MonoidalCategory.associator_naturality

attribute [reassoc] MonoidalCategory.leftUnitor_naturality

attribute [reassoc] MonoidalCategory.rightUnitor_naturality

attribute [reassoc (attr := simp)] MonoidalCategory.pentagon

attribute [reassoc (attr := simp)] MonoidalCategory.triangle


@[simp]
theorem id_tensorHom (X : C) {Y₁ Y₂ : C} (f : Y₁ ⟶ Y₂) :
    𝟙 X ⊗ f = X ◁ f := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y₁ Y₂ : C
    f : Quiver.Hom Y₁ Y₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem tensorHom_id {X₁ X₂ : C} (f : X₁ ⟶ X₂) (Y : C) :
    f ⊗ 𝟙 Y = f ▷ Y := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X₁ X₂ : C
    f : Quiver.Hom X₁ X₂
    Y : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f (CategoryTheory.Catego …
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem whiskerLeft_comp (W : C) {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    W ◁ (f ≫ g) = W ◁ f ≫ W ◁ g := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft W (CategoryTheory.Cate …
  -/
  simp only [← id_tensorHom, ← tensor_comp, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem id_whiskerLeft {X Y : C} (f : X ⟶ Y) :
    𝟙_ C ◁ f = (λ_ X).hom ≫ f ≫ (λ_ Y).inv := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft CategoryTheory.Monoida …
  -/
  rw [← assoc, ← leftUnitor_naturality]; simp [id_tensorHom]
                                         /-
                                           🎉 no goals
                                         -/


@[reassoc, simp]
theorem tensor_whiskerLeft (X Y : C) {Z Z' : C} (f : Z ⟶ Z') :
    (X ⊗ Y) ◁ f = (α_ X Y Z).hom ≫ X ◁ Y ◁ f ≫ (α_ X Y Z').inv := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z Z' : C
    f : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft (CategoryTheory.Monoid …
  -/
  simp only [← id_tensorHom, ← tensorHom_id]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z Z' : C
    f : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  rw [← assoc, ← associator_naturality]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z Z' : C
    f : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem comp_whiskerRight {W X Y : C} (f : W ⟶ X) (g : X ⟶ Y) (Z : C) :
    (f ≫ g) ▷ Z = f ▷ Z ≫ g ▷ Z := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom X Y
    Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Categ …
  -/
  simp only [← tensorHom_id, ← tensor_comp, id_comp]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem whiskerRight_id {X Y : C} (f : X ⟶ Y) :
    f ▷ 𝟙_ C = (ρ_ X).hom ≫ f ≫ (ρ_ Y).inv := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight f CategoryTheory.Mono …
  -/
  rw [← assoc, ← rightUnitor_naturality]; simp [tensorHom_id]
                                          /-
                                            🎉 no goals
                                          -/


@[reassoc, simp]
theorem whiskerRight_tensor {X X' : C} (f : X ⟶ X') (Y Z : C) :
    f ▷ (Y ⊗ Z) = (α_ X Y Z).inv ≫ f ▷ Y ▷ Z ≫ (α_ X' Y Z).hom := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X X' : C
    f : Quiver.Hom X X'
    Y Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight f (CategoryTheory.Mon …
  -/
  simp only [← id_tensorHom, ← tensorHom_id]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X X' : C
    f : Quiver.Hom X X'
    Y Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f (CategoryTheory.Catego …
  -/
  rw [associator_naturality]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X X' : C
    f : Quiver.Hom X X'
    Y Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f (CategoryTheory.Catego …
  -/
  simp [tensor_id]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem whisker_assoc (X : C) {Y Y' : C} (f : Y ⟶ Y') (Z : C) :
    (X ◁ f) ▷ Z = (α_ X Y Z).hom ≫ X ◁ f ▷ Z ≫ (α_ X Y' Z).inv := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Y' : C
    f : Quiver.Hom Y Y'
    Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Monoi …
  -/
  simp only [← id_tensorHom, ← tensorHom_id]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Y' : C
    f : Quiver.Hom Y Y'
    Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  rw [← assoc, ← associator_naturality]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Y' : C
    f : Quiver.Hom Y Y'
    Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem whisker_exchange {W X Y Z : C} (f : W ⟶ X) (g : Y ⟶ Z) :
    W ◁ g ≫ f ▷ Z = f ▷ Y ≫ X ◁ g := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [← id_tensorHom, ← tensorHom_id, ← tensor_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem tensorHom_def' {X₁ Y₁ X₂ Y₂ : C} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
    f ⊗ g = X₁ ◁ g ≫ f ▷ Y₂ :=
  whisker_exchange f g ▸ tensorHom_def f g


@[reassoc (attr := simp)]
theorem whiskerLeft_hom_inv (X : C) {Y Z : C} (f : Y ≅ Z) :
    X ◁ f.hom ≫ X ◁ f.inv = 𝟙 (X ⊗ Y) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    f : CategoryTheory.Iso Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← whiskerLeft_comp, hom_inv_id, whiskerLeft_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem hom_inv_whiskerRight {X Y : C} (f : X ≅ Y) (Z : C) :
    f.hom ▷ Z ≫ f.inv ▷ Z = 𝟙 (X ⊗ Z) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : CategoryTheory.Iso X Y
    Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← comp_whiskerRight, hom_inv_id, id_whiskerRight]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerLeft_inv_hom (X : C) {Y Z : C} (f : Y ≅ Z) :
    X ◁ f.inv ≫ X ◁ f.hom = 𝟙 (X ⊗ Z) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    f : CategoryTheory.Iso Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← whiskerLeft_comp, inv_hom_id, whiskerLeft_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inv_hom_whiskerRight {X Y : C} (f : X ≅ Y) (Z : C) :
    f.inv ▷ Z ≫ f.hom ▷ Z = 𝟙 (Y ⊗ Z) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : CategoryTheory.Iso X Y
    Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← comp_whiskerRight, inv_hom_id, id_whiskerRight]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerLeft_hom_inv' (X : C) {Y Z : C} (f : Y ⟶ Z) [IsIso f] :
    X ◁ f ≫ X ◁ inv f = 𝟙 (X ⊗ Y) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    f : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← whiskerLeft_comp, IsIso.hom_inv_id, whiskerLeft_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem hom_inv_whiskerRight' {X Y : C} (f : X ⟶ Y) [IsIso f] (Z : C) :
    f ▷ Z ≫ inv f ▷ Z = 𝟙 (X ⊗ Z) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← comp_whiskerRight, IsIso.hom_inv_id, id_whiskerRight]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerLeft_inv_hom' (X : C) {Y Z : C} (f : Y ⟶ Z) [IsIso f] :
    X ◁ inv f ≫ X ◁ f = 𝟙 (X ⊗ Z) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    f : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← whiskerLeft_comp, IsIso.inv_hom_id, whiskerLeft_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inv_hom_whiskerRight' {X Y : C} (f : X ⟶ Y) [IsIso f] (Z : C) :
    inv f ▷ Z ≫ f ▷ Z = 𝟙 (Y ⊗ Z) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← comp_whiskerRight, IsIso.inv_hom_id, id_whiskerRight]
  /-
    🎉 no goals
  -/


/-- The left whiskering of an isomorphism is an isomorphism. -/
@[simps]
def whiskerLeftIso (X : C) {Y Z : C} (f : Y ≅ Z) : X ⊗ Y ≅ X ⊗ Z where
  hom := X ◁ f.hom
  inv := X ◁ f.inv


instance whiskerLeft_isIso (X : C) {Y Z : C} (f : Y ⟶ Z) [IsIso f] : IsIso (X ◁ f) :=
  (whiskerLeftIso X (asIso f)).isIso_hom


@[simp]
theorem inv_whiskerLeft (X : C) {Y Z : C} (f : Y ⟶ Z) [IsIso f] :
    inv (X ◁ f) = X ◁ inv f := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    f : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X  …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
lemma whiskerLeftIso_refl (W X : C) :
    whiskerLeftIso W (Iso.refl X) = Iso.refl (W ⊗ X) :=
  Iso.ext (whiskerLeft_id W X)


@[simp]
lemma whiskerLeftIso_trans (W : C) {X Y Z : C} (f : X ≅ Y) (g : Y ≅ Z) :
    whiskerLeftIso W (f ≪≫ g) = whiskerLeftIso W f ≪≫ whiskerLeftIso W g :=
  Iso.ext (whiskerLeft_comp W f.hom g.hom)


@[simp]
lemma whiskerLeftIso_symm (W : C) {X Y : C} (f : X ≅ Y) :
    (whiskerLeftIso W f).symm = whiskerLeftIso W f.symm := rfl


/-- The right whiskering of an isomorphism is an isomorphism. -/
@[simps!]
def whiskerRightIso {X Y : C} (f : X ≅ Y) (Z : C) : X ⊗ Z ≅ Y ⊗ Z where
  hom := f.hom ▷ Z
  inv := f.inv ▷ Z


instance whiskerRight_isIso {X Y : C} (f : X ⟶ Y) (Z : C) [IsIso f] : IsIso (f ▷ Z) :=
  (whiskerRightIso (asIso f) Z).isIso_hom


@[simp]
theorem inv_whiskerRight {X Y : C} (f : X ⟶ Y) (Z : C) [IsIso f] :
    inv (f ▷ Z) = inv f ▷ Z := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.MonoidalCategoryStruct.whiskerRight f …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
lemma whiskerRightIso_refl (X W : C) :
    whiskerRightIso (Iso.refl X) W = Iso.refl (X ⊗ W) :=
  Iso.ext (id_whiskerRight X W)


@[simp]
lemma whiskerRightIso_trans {X Y Z : C} (f : X ≅ Y) (g : Y ≅ Z) (W : C) :
    whiskerRightIso (f ≪≫ g) W = whiskerRightIso f W ≪≫ whiskerRightIso g W :=
  Iso.ext (comp_whiskerRight f.hom g.hom W)


@[simp]
lemma whiskerRightIso_symm {X Y : C} (f : X ≅ Y) (W : C) :
    (whiskerRightIso f W).symm = whiskerRightIso f.symm W := rfl


/-- The tensor product of two isomorphisms is an isomorphism. -/
@[simps]
def tensorIso {X Y X' Y' : C} (f : X ≅ Y)
    (g : X' ≅ Y') : X ⊗ X' ≅ Y ⊗ Y' where
  hom := f.hom ⊗ g.hom
  inv := f.inv ⊗ g.inv
                   /-
                     C : Type u
                     𝒞 : CategoryTheory.Category.{v, u} C
                     inst✝ : CategoryTheory.MonoidalCategory C
                     X Y X' Y' : C
                     f : CategoryTheory.Iso X Y
                     g : CategoryTheory.Iso X' Y'
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                   -/
  hom_inv_id := by rw [← tensor_comp, Iso.hom_inv_id, Iso.hom_inv_id, ← tensor_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u
                     𝒞 : CategoryTheory.Category.{v, u} C
                     inst✝ : CategoryTheory.MonoidalCategory C
                     X Y X' Y' : C
                     f : CategoryTheory.Iso X Y
                     g : CategoryTheory.Iso X' Y'
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                   -/
  inv_hom_id := by rw [← tensor_comp, Iso.inv_hom_id, Iso.inv_hom_id, ← tensor_id]
                   /-
                     🎉 no goals
                   -/


/-- Notation for `tensorIso`, the tensor product of isomorphisms -/
scoped infixr:70 " ⊗ " => tensorIso


theorem tensorIso_def {X Y X' Y' : C} (f : X ≅ Y) (g : X' ≅ Y') :
    f ⊗ g = whiskerRightIso f X' ≪≫ whiskerLeftIso Y g :=
  Iso.ext (tensorHom_def f.hom g.hom)


theorem tensorIso_def' {X Y X' Y' : C} (f : X ≅ Y) (g : X' ≅ Y') :
    f ⊗ g = whiskerLeftIso X g ≪≫ whiskerRightIso f Y' :=
  Iso.ext (tensorHom_def' f.hom g.hom)


instance tensor_isIso {W X Y Z : C} (f : W ⟶ X) [IsIso f] (g : Y ⟶ Z) [IsIso g] : IsIso (f ⊗ g) :=
  (asIso f ⊗ asIso g).isIso_hom


@[simp]
theorem inv_tensor {W X Y Z : C} (f : W ⟶ X) [IsIso f] (g : Y ⟶ Z) [IsIso g] :
    inv (f ⊗ g) = inv f ⊗ inv g := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom W X
    inst✝¹ : CategoryTheory.IsIso f
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.MonoidalCategoryStruct.tensorHom f g) …
  -/
  simp [tensorHom_def ,whisker_exchange]
  /-
    🎉 no goals
  -/


theorem whiskerLeft_dite {P : Prop} [Decidable P]
    (X : C) {Y Z : C} (f : P → (Y ⟶ Z)) (f' : ¬P → (Y ⟶ Z)) :
      X ◁ (if h : P then f h else f' h) = if h : P then X ◁ f h else X ◁ f' h := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    P : Prop
    inst✝ : Decidable P
    X Y Z : C
    f : P → Quiver.Hom Y Z
    f' : Not P → Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (dite P (fun h => f  …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


theorem dite_whiskerRight {P : Prop} [Decidable P]
    {X Y : C} (f : P → (X ⟶ Y)) (f' : ¬P → (X ⟶ Y)) (Z : C) :
      (if h : P then f h else f' h) ▷ Z = if h : P then f h ▷ Z else f' h ▷ Z := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    P : Prop
    inst✝ : Decidable P
    X Y : C
    f : P → Quiver.Hom X Y
    f' : Not P → Quiver.Hom X Y
    Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (dite P (fun h => f h …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


theorem tensor_dite {P : Prop} [Decidable P] {W X Y Z : C} (f : W ⟶ X) (g : P → (Y ⟶ Z))
    (g' : ¬P → (Y ⟶ Z)) : (f ⊗ if h : P then g h else g' h) =
                                              /-
                                                C : Type u
                                                𝒞 : CategoryTheory.Category.{v, u} C
                                                inst✝¹ : CategoryTheory.MonoidalCategory C
                                                P : Prop
                                                inst✝ : Decidable P
                                                W X Y Z : C
                                                f : Quiver.Hom W X
                                                g : P → Quiver.Hom Y Z
                                                g' : Not P → Quiver.Hom Y Z
                                                ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f (dite P (fun h => g h) …
                                              -/
                                                            /-
                                                              🎉 no goals
                                                            -/
    if h : P then f ⊗ g h else f ⊗ g' h := by split_ifs <;> rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem dite_tensor {P : Prop} [Decidable P] {W X Y Z : C} (f : W ⟶ X) (g : P → (Y ⟶ Z))
    (g' : ¬P → (Y ⟶ Z)) : (if h : P then g h else g' h) ⊗ f =
                                              /-
                                                C : Type u
                                                𝒞 : CategoryTheory.Category.{v, u} C
                                                inst✝¹ : CategoryTheory.MonoidalCategory C
                                                P : Prop
                                                inst✝ : Decidable P
                                                W X Y Z : C
                                                f : Quiver.Hom W X
                                                g : P → Quiver.Hom Y Z
                                                g' : Not P → Quiver.Hom Y Z
                                                ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (dite P (fun h => g h) f …
                                              -/
                                                            /-
                                                              🎉 no goals
                                                            -/
    if h : P then g h ⊗ f else g' h ⊗ f := by split_ifs <;> rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem whiskerLeft_eqToHom (X : C) {Y Z : C} (f : Y = Z) :
    X ◁ eqToHom f = eqToHom (congr_arg₂ tensorObj rfl f) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z : C
    f : Eq Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (CategoryTheory.eqTo …
  -/
  cases f
  /-
    case refl
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (CategoryTheory.eqTo …
  -/
  simp only [whiskerLeft_id, eqToHom_refl]
  /-
    🎉 no goals
  -/


@[simp]
theorem eqToHom_whiskerRight {X Y : C} (f : X = Y) (Z : C) :
    eqToHom f ▷ Z = eqToHom (congr_arg₂ tensorObj f rfl) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : Eq X Y
    Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.eqToH …
  -/
  cases f
  /-
    case refl
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Z : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.eqToH …
  -/
  simp only [id_whiskerRight, eqToHom_refl]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem associator_naturality_left {X X' : C} (f : X ⟶ X') (Y Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X X' : C
                                                                       f : Quiver.Hom X X'
                                                                       Y Z : C
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                     -/
    f ▷ Y ▷ Z ≫ (α_ X' Y Z).hom = (α_ X Y Z).hom ≫ f ▷ (Y ⊗ Z) := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_inv_naturality_left {X X' : C} (f : X ⟶ X') (Y Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X X' : C
                                                                       f : Quiver.Hom X X'
                                                                       Y Z : C
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                     -/
    f ▷ (Y ⊗ Z) ≫ (α_ X' Y Z).inv = (α_ X Y Z).inv ≫ f ▷ Y ▷ Z := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem whiskerRight_tensor_symm {X X' : C} (f : X ⟶ X') (Y Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X X' : C
                                                                       f : Quiver.Hom X X'
                                                                       Y Z : C
                                                                       ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Monoi …
                                                                     -/
    f ▷ Y ▷ Z = (α_ X Y Z).hom ≫ f ▷ (Y ⊗ Z) ≫ (α_ X' Y Z).inv := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_naturality_middle (X : C) {Y Y' : C} (f : Y ⟶ Y') (Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X Y Y' : C
                                                                       f : Quiver.Hom Y Y'
                                                                       Z : C
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                     -/
    (X ◁ f) ▷ Z ≫ (α_ X Y' Z).hom = (α_ X Y Z).hom ≫ X ◁ f ▷ Z := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_inv_naturality_middle (X : C) {Y Y' : C} (f : Y ⟶ Y') (Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X Y Y' : C
                                                                       f : Quiver.Hom Y Y'
                                                                       Z : C
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                     -/
    X ◁ f ▷ Z ≫ (α_ X Y' Z).inv = (α_ X Y Z).inv ≫ (X ◁ f) ▷ Z := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem whisker_assoc_symm (X : C) {Y Y' : C} (f : Y ⟶ Y') (Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X Y Y' : C
                                                                       f : Quiver.Hom Y Y'
                                                                       Z : C
                                                                       ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (CategoryTheory.Mono …
                                                                     -/
    X ◁ f ▷ Z = (α_ X Y Z).inv ≫ (X ◁ f) ▷ Z ≫ (α_ X Y' Z).hom := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_naturality_right (X Y : C) {Z Z' : C} (f : Z ⟶ Z') :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X Y Z Z' : C
                                                                       f : Quiver.Hom Z Z'
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                     -/
    (X ⊗ Y) ◁ f ≫ (α_ X Y Z').hom = (α_ X Y Z).hom ≫ X ◁ Y ◁ f := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_inv_naturality_right (X Y : C) {Z Z' : C} (f : Z ⟶ Z') :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X Y Z Z' : C
                                                                       f : Quiver.Hom Z Z'
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                     -/
    X ◁ Y ◁ f ≫ (α_ X Y Z').inv = (α_ X Y Z).inv ≫ (X ⊗ Y) ◁ f := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem tensor_whiskerLeft_symm (X Y : C) {Z Z' : C} (f : Z ⟶ Z') :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X Y Z Z' : C
                                                                       f : Quiver.Hom Z Z'
                                                                       ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (CategoryTheory.Mono …
                                                                     -/
    X ◁ Y ◁ f = (α_ X Y Z).inv ≫ (X ⊗ Y) ◁ f ≫ (α_ X Y Z').hom := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem leftUnitor_inv_naturality {X Y : C} (f : X ⟶ Y) :
                                              /-
                                                C : Type u
                                                𝒞 : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.MonoidalCategory C
                                                X Y : C
                                                f : Quiver.Hom X Y
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.MonoidalCategoryStr …
                                              -/
    f ≫ (λ_ Y).inv = (λ_ X).inv ≫ _ ◁ f := by simp
                                              /-
                                                🎉 no goals
                                              -/


@[reassoc]
theorem id_whiskerLeft_symm {X X' : C} (f : X ⟶ X') :
    f = (λ_ X).inv ≫ 𝟙_ C ◁ f ≫ (λ_ X').hom := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X X' : C
    f : Quiver.Hom X X'
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStr …
  -/
  simp only [id_whiskerLeft, assoc, inv_hom_id, comp_id, inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem rightUnitor_inv_naturality {X X' : C} (f : X ⟶ X') :
                                               /-
                                                 C : Type u
                                                 𝒞 : CategoryTheory.Category.{v, u} C
                                                 inst✝ : CategoryTheory.MonoidalCategory C
                                                 X X' : C
                                                 f : Quiver.Hom X X'
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.MonoidalCategoryStr …
                                               -/
    f ≫ (ρ_ X').inv = (ρ_ X).inv ≫ f ▷ _ := by simp
                                               /-
                                                 🎉 no goals
                                               -/


@[reassoc]
theorem whiskerRight_id_symm {X Y : C} (f : X ⟶ Y) :
    f = (ρ_ X).inv ≫ f ▷ 𝟙_ C ≫ (ρ_ Y).hom := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStr …
  -/
  simp
  /-
    🎉 no goals
  -/


                                                                                    /-
                                                                                      C : Type u
                                                                                      𝒞 : CategoryTheory.Category.{v, u} C
                                                                                      inst✝ : CategoryTheory.MonoidalCategory C
                                                                                      X Y : C
                                                                                      f g : Quiver.Hom X Y
                                                                                      ⊢ Iff (Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft CategoryTheory.Mo …
                                                                                    -/
theorem whiskerLeft_iff {X Y : C} (f g : X ⟶ Y) : 𝟙_ C ◁ f = 𝟙_ C ◁ g ↔ f = g := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


                                                                                     /-
                                                                                       C : Type u
                                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                                       X Y : C
                                                                                       f g : Quiver.Hom X Y
                                                                                       ⊢ Iff (Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight f CategoryTheory …
                                                                                     -/
theorem whiskerRight_iff {X Y : C} (f g : X ⟶ Y) : f ▷ 𝟙_ C = g ▷ 𝟙_ C ↔ f = g := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[reassoc (attr := simp)]
theorem pentagon_inv :
    W ◁ (α_ X Y Z).inv ≫ (α_ W (X ⊗ Y) Z).inv ≫ (α_ W X Y).inv ▷ Z =
      (α_ W X (Y ⊗ Z)).inv ≫ (α_ (W ⊗ X) Y Z).inv :=
                       /-
                         C : Type u
                         𝒞 : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.MonoidalCategory C
                         W X Y Z : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.M …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem pentagon_inv_inv_hom_hom_inv :
    (α_ W (X ⊗ Y) Z).inv ≫ (α_ W X Y).inv ▷ Z ≫ (α_ (W ⊗ X) Y Z).hom =
      W ◁ (α_ X Y Z).hom ≫ (α_ W X (Y ⊗ Z)).inv := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← cancel_epi (W ◁ (α_ X Y Z).inv), ← cancel_mono (α_ (W ⊗ X) Y Z).inv]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_inv_hom_hom_hom_inv :
    (α_ (W ⊗ X) Y Z).inv ≫ (α_ W X Y).hom ▷ Z ≫ (α_ W (X ⊗ Y) Z).hom =
      (α_ W X (Y ⊗ Z)).hom ≫ W ◁ (α_ X Y Z).inv :=
                       /-
                         C : Type u
                         𝒞 : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.MonoidalCategory C
                         W X Y Z : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.M …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem pentagon_hom_inv_inv_inv_inv :
    W ◁ (α_ X Y Z).hom ≫ (α_ W X (Y ⊗ Z)).inv ≫ (α_ (W ⊗ X) Y Z).inv =
      (α_ W (X ⊗ Y) Z).inv ≫ (α_ W X Y).inv ▷ Z := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [← cancel_epi (W ◁ (α_ X Y Z).inv)]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_hom_hom_inv_hom_hom :
    (α_ (W ⊗ X) Y Z).hom ≫ (α_ W X (Y ⊗ Z)).hom ≫ W ◁ (α_ X Y Z).inv =
      (α_ W X Y).hom ▷ Z ≫ (α_ W (X ⊗ Y) Z).hom :=
                       /-
                         C : Type u
                         𝒞 : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.MonoidalCategory C
                         W X Y Z : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.M …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem pentagon_hom_inv_inv_inv_hom :
    (α_ W X (Y ⊗ Z)).hom ≫ W ◁ (α_ X Y Z).inv ≫ (α_ W (X ⊗ Y) Z).inv =
      (α_ (W ⊗ X) Y Z).inv ≫ (α_ W X Y).hom ▷ Z := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← cancel_epi (α_ W X (Y ⊗ Z)).inv, ← cancel_mono ((α_ W X Y).inv ▷ Z)]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_hom_hom_inv_inv_hom :
    (α_ W (X ⊗ Y) Z).hom ≫ W ◁ (α_ X Y Z).hom ≫ (α_ W X (Y ⊗ Z)).inv =
      (α_ W X Y).inv ▷ Z ≫ (α_ (W ⊗ X) Y Z).hom :=
                       /-
                         C : Type u
                         𝒞 : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.MonoidalCategory C
                         W X Y Z : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.M …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem pentagon_inv_hom_hom_hom_hom :
    (α_ W X Y).inv ▷ Z ≫ (α_ (W ⊗ X) Y Z).hom ≫ (α_ W X (Y ⊗ Z)).hom =
      (α_ W (X ⊗ Y) Z).hom ≫ W ◁ (α_ X Y Z).hom := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [← cancel_epi ((α_ W X Y).hom ▷ Z)]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_inv_inv_hom_inv_inv :
    (α_ W X (Y ⊗ Z)).inv ≫ (α_ (W ⊗ X) Y Z).inv ≫ (α_ W X Y).hom ▷ Z =
      W ◁ (α_ X Y Z).inv ≫ (α_ W (X ⊗ Y) Z).inv :=
                       /-
                         C : Type u
                         𝒞 : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.MonoidalCategory C
                         W X Y Z : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.M …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem triangle_assoc_comp_right (X Y : C) :
    (α_ X (𝟙_ C) Y).inv ≫ ((ρ_ X).hom ▷ Y) = X ◁ (λ_ Y).hom := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← triangle, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem triangle_assoc_comp_right_inv (X Y : C) :
    (ρ_ X).inv ▷ Y ≫ (α_ X (𝟙_ C) Y).hom = X ◁ (λ_ Y).inv := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [← cancel_mono (X ◁ (λ_ Y).hom)]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem triangle_assoc_comp_left_inv (X Y : C) :
    (X ◁ (λ_ Y).inv) ≫ (α_ X (𝟙_ C) Y).inv = (ρ_ X).inv ▷ Y := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [← cancel_mono ((ρ_ X).hom ▷ Y)]
  /-
    🎉 no goals
  -/


/-- We state it as a simp lemma, which is regarded as an involved version of
`id_whiskerRight X Y : 𝟙 X ▷ Y = 𝟙 (X ⊗ Y)`.
-/
@[reassoc, simp]
theorem leftUnitor_whiskerRight (X Y : C) :
    (λ_ X).hom ▷ Y = (α_ (𝟙_ C) X Y).hom ≫ (λ_ (X ⊗ Y)).hom := by
  rw [← whiskerLeft_iff, whiskerLeft_comp, ← cancel_epi (α_ _ _ _).hom, ←
      cancel_epi ((α_ _ _ _).hom ▷ _), pentagon_assoc, triangle, ← associator_naturality_middle, ←
      comp_whiskerRight_assoc, triangle, associator_naturality_left]


@[reassoc, simp]
theorem leftUnitor_inv_whiskerRight (X Y : C) :
    (λ_ X).inv ▷ Y = (λ_ (X ⊗ Y)).inv ≫ (α_ (𝟙_ C) X Y).inv :=
                       /-
                         C : Type u
                         𝒞 : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.MonoidalCategory C
                         X Y : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.MonoidalCategoryStruct.whiskerRight ( …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc, simp]
theorem whiskerLeft_rightUnitor (X Y : C) :
    X ◁ (ρ_ Y).hom = (α_ X Y (𝟙_ C)).inv ≫ (ρ_ (X ⊗ Y)).hom := by
  rw [← whiskerRight_iff, comp_whiskerRight, ← cancel_epi (α_ _ _ _).inv, ←
      cancel_epi (X ◁ (α_ _ _ _).inv), pentagon_inv_assoc, triangle_assoc_comp_right, ←
      associator_inv_naturality_middle, ← whiskerLeft_comp_assoc, triangle_assoc_comp_right,
      associator_inv_naturality_right]


@[reassoc, simp]
theorem whiskerLeft_rightUnitor_inv (X Y : C) :
    X ◁ (ρ_ Y).inv = (ρ_ (X ⊗ Y)).inv ≫ (α_ X Y (𝟙_ C)).hom :=
                       /-
                         C : Type u
                         𝒞 : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.MonoidalCategory C
                         X Y : C
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X  …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc]
theorem leftUnitor_tensor (X Y : C) :
                                                                  /-
                                                                    C : Type u
                                                                    𝒞 : CategoryTheory.Category.{v, u} C
                                                                    inst✝ : CategoryTheory.MonoidalCategory C
                                                                    X Y : C
                                                                    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
                                                                  -/
    (λ_ (X ⊗ Y)).hom = (α_ (𝟙_ C) X Y).inv ≫ (λ_ X).hom ▷ Y := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[reassoc]
theorem leftUnitor_tensor_inv (X Y : C) :
                                                                  /-
                                                                    C : Type u
                                                                    𝒞 : CategoryTheory.Category.{v, u} C
                                                                    inst✝ : CategoryTheory.MonoidalCategory C
                                                                    X Y : C
                                                                    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
                                                                  -/
    (λ_ (X ⊗ Y)).inv = (λ_ X).inv ▷ Y ≫ (α_ (𝟙_ C) X Y).hom := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[reassoc]
theorem rightUnitor_tensor (X Y : C) :
                                                                  /-
                                                                    C : Type u
                                                                    𝒞 : CategoryTheory.Category.{v, u} C
                                                                    inst✝ : CategoryTheory.MonoidalCategory C
                                                                    X Y : C
                                                                    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (CategoryTheory.Monoid …
                                                                  -/
    (ρ_ (X ⊗ Y)).hom = (α_ X Y (𝟙_ C)).hom ≫ X ◁ (ρ_ Y).hom := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[reassoc]
theorem rightUnitor_tensor_inv (X Y : C) :
                                                                  /-
                                                                    C : Type u
                                                                    𝒞 : CategoryTheory.Category.{v, u} C
                                                                    inst✝ : CategoryTheory.MonoidalCategory C
                                                                    X Y : C
                                                                    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (CategoryTheory.Monoid …
                                                                  -/
    (ρ_ (X ⊗ Y)).inv = X ◁ (ρ_ Y).inv ≫ (α_ X Y (𝟙_ C)).inv := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[reassoc]
theorem associator_inv_naturality {X Y Z X' Y' Z' : C} (f : X ⟶ X') (g : Y ⟶ Y') (h : Z ⟶ Z') :
    (f ⊗ g ⊗ h) ≫ (α_ X' Y' Z').inv = (α_ X Y Z).inv ≫ ((f ⊗ g) ⊗ h) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z X' Y' Z' : C
    f : Quiver.Hom X X'
    g : Quiver.Hom Y Y'
    h : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem associator_conjugation {X X' Y Y' Z Z' : C} (f : X ⟶ X') (g : Y ⟶ Y') (h : Z ⟶ Z') :
    (f ⊗ g) ⊗ h = (α_ X Y Z).hom ≫ (f ⊗ g ⊗ h) ≫ (α_ X' Y' Z').inv := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X X' Y Y' Z Z' : C
    f : Quiver.Hom X X'
    g : Quiver.Hom Y Y'
    h : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  rw [associator_inv_naturality, hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem associator_inv_conjugation {X X' Y Y' Z Z' : C} (f : X ⟶ X') (g : Y ⟶ Y') (h : Z ⟶ Z') :
    f ⊗ g ⊗ h = (α_ X Y Z).inv ≫ ((f ⊗ g) ⊗ h) ≫ (α_ X' Y' Z').hom := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X X' Y Y' Z Z' : C
    f : Quiver.Hom X X'
    g : Quiver.Hom Y Y'
    h : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f (CategoryTheory.Monoid …
  -/
  rw [associator_naturality, inv_hom_id_assoc]
  /-
    🎉 no goals
  -/

-- TODO these next two lemmas aren't so fundamental, and perhaps could be removed
-- (replacing their usages by their proofs).

@[reassoc]
theorem id_tensor_associator_naturality {X Y Z Z' : C} (h : Z ⟶ Z') :
    (𝟙 (X ⊗ Y) ⊗ h) ≫ (α_ X Y Z').hom = (α_ X Y Z).hom ≫ (𝟙 X ⊗ 𝟙 Y ⊗ h) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z Z' : C
    h : Quiver.Hom Z Z'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_id, associator_naturality]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem id_tensor_associator_inv_naturality {X Y Z X' : C} (f : X ⟶ X') :
    (f ⊗ 𝟙 (Y ⊗ Z)) ≫ (α_ X' Y Z).inv = (α_ X Y Z).inv ≫ ((f ⊗ 𝟙 Y) ⊗ 𝟙 Z) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    X Y Z X' : C
    f : Quiver.Hom X X'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_id, associator_inv_naturality]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem hom_inv_id_tensor {V W X Y Z : C} (f : V ≅ W) (g : X ⟶ Y) (h : Y ⟶ Z) :
    (f.hom ⊗ g) ≫ (f.inv ⊗ h) = (𝟙 V ⊗ g) ≫ (𝟙 V ⊗ h) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    V W X Y Z : C
    f : CategoryTheory.Iso V W
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp, f.hom_inv_id]; simp [id_tensorHom]
                                    /-
                                      🎉 no goals
                                    -/


@[reassoc (attr := simp)]
theorem inv_hom_id_tensor {V W X Y Z : C} (f : V ≅ W) (g : X ⟶ Y) (h : Y ⟶ Z) :
    (f.inv ⊗ g) ≫ (f.hom ⊗ h) = (𝟙 W ⊗ g) ≫ (𝟙 W ⊗ h) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    V W X Y Z : C
    f : CategoryTheory.Iso V W
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp, f.inv_hom_id]; simp [id_tensorHom]
                                    /-
                                      🎉 no goals
                                    -/


@[reassoc (attr := simp)]
theorem tensor_hom_inv_id {V W X Y Z : C} (f : V ≅ W) (g : X ⟶ Y) (h : Y ⟶ Z) :
    (g ⊗ f.hom) ≫ (h ⊗ f.inv) = (g ⊗ 𝟙 V) ≫ (h ⊗ 𝟙 V) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    V W X Y Z : C
    f : CategoryTheory.Iso V W
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp, f.hom_inv_id]; simp [tensorHom_id]
                                    /-
                                      🎉 no goals
                                    -/


@[reassoc (attr := simp)]
theorem tensor_inv_hom_id {V W X Y Z : C} (f : V ≅ W) (g : X ⟶ Y) (h : Y ⟶ Z) :
    (g ⊗ f.inv) ≫ (h ⊗ f.hom) = (g ⊗ 𝟙 W) ≫ (h ⊗ 𝟙 W) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    V W X Y Z : C
    f : CategoryTheory.Iso V W
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp, f.inv_hom_id]; simp [tensorHom_id]
                                    /-
                                      🎉 no goals
                                    -/


@[reassoc (attr := simp)]
theorem hom_inv_id_tensor' {V W X Y Z : C} (f : V ⟶ W) [IsIso f] (g : X ⟶ Y) (h : Y ⟶ Z) :
    (f ⊗ g) ≫ (inv f ⊗ h) = (𝟙 V ⊗ g) ≫ (𝟙 V ⊗ h) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    V W X Y Z : C
    f : Quiver.Hom V W
    inst✝ : CategoryTheory.IsIso f
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp, IsIso.hom_inv_id]; simp [id_tensorHom]
                                        /-
                                          🎉 no goals
                                        -/


@[reassoc (attr := simp)]
theorem inv_hom_id_tensor' {V W X Y Z : C} (f : V ⟶ W) [IsIso f] (g : X ⟶ Y) (h : Y ⟶ Z) :
    (inv f ⊗ g) ≫ (f ⊗ h) = (𝟙 W ⊗ g) ≫ (𝟙 W ⊗ h) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    V W X Y Z : C
    f : Quiver.Hom V W
    inst✝ : CategoryTheory.IsIso f
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp, IsIso.inv_hom_id]; simp [id_tensorHom]
                                        /-
                                          🎉 no goals
                                        -/


@[reassoc (attr := simp)]
theorem tensor_hom_inv_id' {V W X Y Z : C} (f : V ⟶ W) [IsIso f] (g : X ⟶ Y) (h : Y ⟶ Z) :
    (g ⊗ f) ≫ (h ⊗ inv f) = (g ⊗ 𝟙 V) ≫ (h ⊗ 𝟙 V) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    V W X Y Z : C
    f : Quiver.Hom V W
    inst✝ : CategoryTheory.IsIso f
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp, IsIso.hom_inv_id]; simp [tensorHom_id]
                                        /-
                                          🎉 no goals
                                        -/


@[reassoc (attr := simp)]
theorem tensor_inv_hom_id' {V W X Y Z : C} (f : V ⟶ W) [IsIso f] (g : X ⟶ Y) (h : Y ⟶ Z) :
    (g ⊗ inv f) ≫ (h ⊗ f) = (g ⊗ 𝟙 W) ≫ (h ⊗ 𝟙 W) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    V W X Y Z : C
    f : Quiver.Hom V W
    inst✝ : CategoryTheory.IsIso f
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp, IsIso.inv_hom_id]; simp [tensorHom_id]
                                        /-
                                          🎉 no goals
                                        -/


/--
A constructor for monoidal categories that requires `tensorHom` instead of `whiskerLeft` and
`whiskerRight`.
-/
abbrev ofTensorHom [MonoidalCategoryStruct C]
    (tensor_id : ∀ X₁ X₂ : C, tensorHom (𝟙 X₁) (𝟙 X₂) = 𝟙 (tensorObj X₁ X₂) := by
      aesop_cat)
    (id_tensorHom : ∀ (X : C) {Y₁ Y₂ : C} (f : Y₁ ⟶ Y₂), tensorHom (𝟙 X) f = whiskerLeft X f := by
      aesop_cat)
    (tensorHom_id : ∀ {X₁ X₂ : C} (f : X₁ ⟶ X₂) (Y : C), tensorHom f (𝟙 Y) = whiskerRight f Y := by
      aesop_cat)
    (tensor_comp :
      ∀ {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (g₁ : Y₁ ⟶ Z₁) (g₂ : Y₂ ⟶ Z₂),
        tensorHom (f₁ ≫ g₁) (f₂ ≫ g₂) = tensorHom f₁ f₂ ≫ tensorHom g₁ g₂ := by
          aesop_cat)
    (associator_naturality :
      ∀ {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (f₃ : X₃ ⟶ Y₃),
        tensorHom (tensorHom f₁ f₂) f₃ ≫ (associator Y₁ Y₂ Y₃).hom =
          (associator X₁ X₂ X₃).hom ≫ tensorHom f₁ (tensorHom f₂ f₃) := by
            aesop_cat)
    (leftUnitor_naturality :
      ∀ {X Y : C} (f : X ⟶ Y),
        tensorHom (𝟙 tensorUnit) f ≫ (leftUnitor Y).hom = (leftUnitor X).hom ≫ f := by
          aesop_cat)
    (rightUnitor_naturality :
      ∀ {X Y : C} (f : X ⟶ Y),
        tensorHom f (𝟙 tensorUnit) ≫ (rightUnitor Y).hom = (rightUnitor X).hom ≫ f := by
          aesop_cat)
    (pentagon :
      ∀ W X Y Z : C,
        tensorHom (associator W X Y).hom (𝟙 Z) ≫
            (associator W (tensorObj X Y) Z).hom ≫ tensorHom (𝟙 W) (associator X Y Z).hom =
          (associator (tensorObj W X) Y Z).hom ≫ (associator W X (tensorObj Y Z)).hom := by
            aesop_cat)
    (triangle :
      ∀ X Y : C,
        (associator X tensorUnit Y).hom ≫ tensorHom (𝟙 X) (leftUnitor Y).hom =
          tensorHom (rightUnitor X).hom (𝟙 Y) := by
            aesop_cat) :
      MonoidalCategory C where
                      /-
                        C : Type u
                        𝒞 : CategoryTheory.Category.{v, u} C
                        inst✝¹ : CategoryTheory.MonoidalCategory C
                        W X Y Z : C
                        inst✝ : CategoryTheory.MonoidalCategoryStruct C
                        tensor_id : autoParam (∀ (X₁ X₂ : C), Eq (CategoryTheory.MonoidalCategoryStruc …
                        id_tensorHom : autoParam (∀ (X : C) {Y₁ Y₂ : C} (f : Quiver.Hom Y₁ Y₂), Eq (Ca …
                        tensorHom_id : autoParam (∀ {X₁ X₂ : C} (f : Quiver.Hom X₁ X₂) (Y : C), Eq (Ca …
                        tensor_comp : autoParam (∀ {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C} (f₁ : Quiver.Hom X₁ Y₁) (f₂ …
                        associator_naturality : autoParam (∀ {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : Quiver.Hom  …
                        leftUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Categ …
                        rightUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Cate …
                        pentagon : autoParam (∀ (W X Y Z : C), Eq (CategoryTheory.CategoryStruct.comp  …
                        triangle : autoParam (∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (Cat …
                        ⊢ ∀ {X₁ Y₁ X₂ Y₂ : C} (f : Quiver.Hom X₁ Y₁) (g : Quiver.Hom X₂ Y₂), Eq (Categ …
                      -/
  tensorHom_def := by intros; simp [← id_tensorHom, ← tensorHom_id, ← tensor_comp]
                              /-
                                🎉 no goals
                              -/
                       /-
                         C : Type u
                         𝒞 : CategoryTheory.Category.{v, u} C
                         inst✝¹ : CategoryTheory.MonoidalCategory C
                         W X Y Z : C
                         inst✝ : CategoryTheory.MonoidalCategoryStruct C
                         tensor_id : autoParam (∀ (X₁ X₂ : C), Eq (CategoryTheory.MonoidalCategoryStruc …
                         id_tensorHom : autoParam (∀ (X : C) {Y₁ Y₂ : C} (f : Quiver.Hom Y₁ Y₂), Eq (Ca …
                         tensorHom_id : autoParam (∀ {X₁ X₂ : C} (f : Quiver.Hom X₁ X₂) (Y : C), Eq (Ca …
                         tensor_comp : autoParam (∀ {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C} (f₁ : Quiver.Hom X₁ Y₁) (f₂ …
                         associator_naturality : autoParam (∀ {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : Quiver.Hom  …
                         leftUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Categ …
                         rightUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Cate …
                         pentagon : autoParam (∀ (W X Y Z : C), Eq (CategoryTheory.CategoryStruct.comp  …
                         triangle : autoParam (∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (Cat …
                         ⊢ ∀ (X Y : C), Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (Catego …
                       -/
  whiskerLeft_id := by intros; simp [← id_tensorHom, ← tensor_id]
                               /-
                                 🎉 no goals
                               -/
                        /-
                          C : Type u
                          𝒞 : CategoryTheory.Category.{v, u} C
                          inst✝¹ : CategoryTheory.MonoidalCategory C
                          W X Y Z : C
                          inst✝ : CategoryTheory.MonoidalCategoryStruct C
                          tensor_id : autoParam (∀ (X₁ X₂ : C), Eq (CategoryTheory.MonoidalCategoryStruc …
                          id_tensorHom : autoParam (∀ (X : C) {Y₁ Y₂ : C} (f : Quiver.Hom Y₁ Y₂), Eq (Ca …
                          tensorHom_id : autoParam (∀ {X₁ X₂ : C} (f : Quiver.Hom X₁ X₂) (Y : C), Eq (Ca …
                          tensor_comp : autoParam (∀ {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C} (f₁ : Quiver.Hom X₁ Y₁) (f₂ …
                          associator_naturality : autoParam (∀ {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : Quiver.Hom  …
                          leftUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Categ …
                          rightUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Cate …
                          pentagon : autoParam (∀ (W X Y Z : C), Eq (CategoryTheory.CategoryStruct.comp  …
                          triangle : autoParam (∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (Cat …
                          ⊢ ∀ (X Y : C), Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (Categor …
                        -/
  id_whiskerRight := by intros; simp [← tensorHom_id, tensor_id]
                                /-
                                  🎉 no goals
                                -/
                 /-
                   C : Type u
                   𝒞 : CategoryTheory.Category.{v, u} C
                   inst✝¹ : CategoryTheory.MonoidalCategory C
                   W X Y Z : C
                   inst✝ : CategoryTheory.MonoidalCategoryStruct C
                   tensor_id : autoParam (∀ (X₁ X₂ : C), Eq (CategoryTheory.MonoidalCategoryStruc …
                   id_tensorHom : autoParam (∀ (X : C) {Y₁ Y₂ : C} (f : Quiver.Hom Y₁ Y₂), Eq (Ca …
                   tensorHom_id : autoParam (∀ {X₁ X₂ : C} (f : Quiver.Hom X₁ X₂) (Y : C), Eq (Ca …
                   tensor_comp : autoParam (∀ {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C} (f₁ : Quiver.Hom X₁ Y₁) (f₂ …
                   associator_naturality : autoParam (∀ {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : Quiver.Hom  …
                   leftUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Categ …
                   rightUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Cate …
                   pentagon : autoParam (∀ (W X Y Z : C), Eq (CategoryTheory.CategoryStruct.comp  …
                   triangle : autoParam (∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (Cat …
                   ⊢ ∀ (W X Y Z : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Mono …
                 -/
  pentagon := by intros; simp [← id_tensorHom, ← tensorHom_id, pentagon]
                         /-
                           🎉 no goals
                         -/
                 /-
                   C : Type u
                   𝒞 : CategoryTheory.Category.{v, u} C
                   inst✝¹ : CategoryTheory.MonoidalCategory C
                   W X Y Z : C
                   inst✝ : CategoryTheory.MonoidalCategoryStruct C
                   tensor_id : autoParam (∀ (X₁ X₂ : C), Eq (CategoryTheory.MonoidalCategoryStruc …
                   id_tensorHom : autoParam (∀ (X : C) {Y₁ Y₂ : C} (f : Quiver.Hom Y₁ Y₂), Eq (Ca …
                   tensorHom_id : autoParam (∀ {X₁ X₂ : C} (f : Quiver.Hom X₁ X₂) (Y : C), Eq (Ca …
                   tensor_comp : autoParam (∀ {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C} (f₁ : Quiver.Hom X₁ Y₁) (f₂ …
                   associator_naturality : autoParam (∀ {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : Quiver.Hom  …
                   leftUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Categ …
                   rightUnitor_naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (Cate …
                   pentagon : autoParam (∀ (W X Y Z : C), Eq (CategoryTheory.CategoryStruct.comp  …
                   triangle : autoParam (∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (Cat …
                   ⊢ ∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoidal …
                 -/
  triangle := by intros; simp [← id_tensorHom, ← tensorHom_id, triangle]
                         /-
                           🎉 no goals
                         -/


@[reassoc]
theorem comp_tensor_id (f : W ⟶ X) (g : X ⟶ Y) : f ≫ g ⊗ 𝟙 Z = (f ⊗ 𝟙 Z) ≫ (g ⊗ 𝟙 Z) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem id_tensor_comp (f : W ⟶ X) (g : X ⟶ Y) : 𝟙 Z ⊗ f ≫ g = (𝟙 Z ⊗ f) ≫ (𝟙 Z ⊗ g) := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem id_tensor_comp_tensor_id (f : W ⟶ X) (g : Y ⟶ Z) : (𝟙 Y ⊗ f) ≫ (g ⊗ 𝟙 X) = g ⊗ f := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem tensor_id_comp_id_tensor (f : W ⟶ X) (g : Y ⟶ Z) : (g ⊗ 𝟙 W) ≫ (𝟙 Z ⊗ f) = g ⊗ f := by
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← tensor_comp]
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  simp
  /-
    🎉 no goals
  -/


                                                                                            /-
                                                                                              C : Type u
                                                                                              𝒞 : CategoryTheory.Category.{v, u} C
                                                                                              inst✝ : CategoryTheory.MonoidalCategory C
                                                                                              X Y : C
                                                                                              f g : Quiver.Hom X Y
                                                                                              ⊢ Iff (Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Cat …
                                                                                            -/
theorem tensor_left_iff {X Y : C} (f g : X ⟶ Y) : 𝟙 (𝟙_ C) ⊗ f = 𝟙 (𝟙_ C) ⊗ g ↔ f = g := by simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


                                                                                             /-
                                                                                               C : Type u
                                                                                               𝒞 : CategoryTheory.Category.{v, u} C
                                                                                               inst✝ : CategoryTheory.MonoidalCategory C
                                                                                               X Y : C
                                                                                               f g : Quiver.Hom X Y
                                                                                               ⊢ Iff (Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f (CategoryTheory.C …
                                                                                             -/
theorem tensor_right_iff {X Y : C} (f g : X ⟶ Y) : f ⊗ 𝟙 (𝟙_ C) = g ⊗ 𝟙 (𝟙_ C) ↔ f = g := by simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


/-- The tensor product expressed as a functor. -/
@[simps]
def tensor : C × C ⥤ C where
  obj X := X.1 ⊗ X.2
  map {X Y : C × C} (f : X ⟶ Y) := f.1 ⊗ f.2


/-- The left-associated triple tensor product as a functor. -/
def leftAssocTensor : C × C × C ⥤ C where
  obj X := (X.1 ⊗ X.2.1) ⊗ X.2.2
  map {X Y : C × C × C} (f : X ⟶ Y) := (f.1 ⊗ f.2.1) ⊗ f.2.2


@[simp]
theorem leftAssocTensor_obj (X) : (leftAssocTensor C).obj X = (X.1 ⊗ X.2.1) ⊗ X.2.2 :=
  rfl


@[simp]
theorem leftAssocTensor_map {X Y} (f : X ⟶ Y) : (leftAssocTensor C).map f = (f.1 ⊗ f.2.1) ⊗ f.2.2 :=
  rfl


/-- The right-associated triple tensor product as a functor. -/
def rightAssocTensor : C × C × C ⥤ C where
  obj X := X.1 ⊗ X.2.1 ⊗ X.2.2
  map {X Y : C × C × C} (f : X ⟶ Y) := f.1 ⊗ f.2.1 ⊗ f.2.2


@[simp]
theorem rightAssocTensor_obj (X) : (rightAssocTensor C).obj X = X.1 ⊗ X.2.1 ⊗ X.2.2 :=
  rfl


@[simp]
theorem rightAssocTensor_map {X Y} (f : X ⟶ Y) : (rightAssocTensor C).map f = f.1 ⊗ f.2.1 ⊗ f.2.2 :=
  rfl


/-- The tensor product bifunctor `C ⥤ C ⥤ C` of a monoidal category. -/
@[simps]
def curriedTensor : C ⥤ C ⥤ C where
  obj X :=
    { obj := fun Y => X ⊗ Y
      map := fun g => X ◁ g }
  map f :=
    { app := fun Y => f ▷ Y }


/-- Tensoring on the left with a fixed object, as a functor. -/
@[simps!]
def tensorLeft (X : C) : C ⥤ C := (curriedTensor C).obj X


/-- Tensoring on the right with a fixed object, as a functor. -/
@[simps!]
def tensorRight (X : C) : C ⥤ C := (curriedTensor C).flip.obj X


/-- The functor `fun X ↦ 𝟙_ C ⊗ X`. -/
abbrev tensorUnitLeft : C ⥤ C := tensorLeft (𝟙_ C)


/-- The functor `fun X ↦ X ⊗ 𝟙_ C`. -/
abbrev tensorUnitRight : C ⥤ C := tensorRight (𝟙_ C)

-- We can express the associator and the unitors, given componentwise above,
-- as natural isomorphisms.
-- Porting Note: Had to add a `simps!` because Lean was complaining this wasn't a constructor app.

/-- The associator as a natural isomorphism. -/
@[simps!]
def associatorNatIso : leftAssocTensor C ≅ rightAssocTensor C :=
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ ∀ {X Y : Prod C (Prod C C)} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categor …
  -/
  NatIso.ofComponents (fun _ => MonoidalCategory.associator _ _ _)
  /-
    🎉 no goals
  -/

-- Porting Note: same as above

/-- The left unitor as a natural isomorphism. -/
@[simps!]
def leftUnitorNatIso : tensorUnitLeft C ≅ 𝟭 C :=
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
  -/
  NatIso.ofComponents MonoidalCategory.leftUnitor
  /-
    🎉 no goals
  -/

-- Porting Note: same as above

/-- The right unitor as a natural isomorphism. -/
@[simps!]
def rightUnitorNatIso : tensorUnitRight C ≅ 𝟭 C :=
  /-
    C : Type u
    𝒞 : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    W X Y Z : C
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
  -/
  NatIso.ofComponents MonoidalCategory.rightUnitor
  /-
    🎉 no goals
  -/


/-- The associator as a natural isomorphism between trifunctors `C ⥤ C ⥤ C ⥤ C`. -/
@[simps!]
def curriedAssociatorNatIso :
    bifunctorComp₁₂ (curriedTensor C) (curriedTensor C) ≅
      bifunctorComp₂₃ (curriedTensor C) (curriedTensor C) :=
                                                                /-
                                                                  C : Type u
                                                                  𝒞 : CategoryTheory.Category.{v, u} C
                                                                  inst✝ : CategoryTheory.MonoidalCategory C
                                                                  W X Y Z X₁ X₂ : C
                                                                  ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                 /-
                                   🎉 no goals
                                 -/
  NatIso.ofComponents (fun X₁ => NatIso.ofComponents (fun X₂ => NatIso.ofComponents
  /-
    🎉 no goals
  -/
    (fun X₃ => α_ X₁ X₂ X₃)))


/-- Tensoring on the left with `X ⊗ Y` is naturally isomorphic to
tensoring on the left with `Y`, and then again with `X`.
-/
def tensorLeftTensor (X Y : C) : tensorLeft (X ⊗ Y) ≅ tensorLeft Y ⋙ tensorLeft X :=
                                                            /-
                                                              C : Type u
                                                              𝒞 : CategoryTheory.Category.{v, u} C
                                                              inst✝ : CategoryTheory.MonoidalCategory C
                                                              W X✝ Y✝ Z✝ X Y Z Z' : C
                                                              f : Quiver.Hom Z Z'
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
                                                            -/
  NatIso.ofComponents (associator _ _) fun {Z} {Z'} f => by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem tensorLeftTensor_hom_app (X Y Z : C) :
    (tensorLeftTensor X Y).hom.app Z = (associator X Y Z).hom :=
  rfl


@[simp]
theorem tensorLeftTensor_inv_app (X Y Z : C) :
                                                                    /-
                                                                      C : Type u
                                                                      𝒞 : CategoryTheory.Category.{v, u} C
                                                                      inst✝ : CategoryTheory.MonoidalCategory C
                                                                      X Y Z : C
                                                                      ⊢ Eq ((CategoryTheory.MonoidalCategory.tensorLeftTensor X Y).inv.app Z) (Categ …
                                                                    -/
    (tensorLeftTensor X Y).inv.app Z = (associator X Y Z).inv := by simp [tensorLeftTensor]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- Tensoring on the left, as a functor from `C` into endofunctors of `C`.

TODO: show this is an op-monoidal functor.
-/
abbrev tensoringLeft : C ⥤ C ⥤ C := curriedTensor C


instance : (tensoringLeft C).Faithful where
  map_injective {X} {Y} f g h := by
    /-
      C : Type u
      𝒞 : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.MonoidalCategory C
      W X✝ Y✝ Z X Y : C
      f g : Quiver.Hom X Y
      h : Eq ((CategoryTheory.MonoidalCategory.tensoringLeft C).map f) ((CategoryThe …
      ⊢ Eq f g
    -/
    injections h
    /-
      C : Type u
      𝒞 : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.MonoidalCategory C
      W X✝ Y✝ Z X Y : C
      f g : Quiver.Hom X Y
      h : Eq (fun Y_1 => CategoryTheory.MonoidalCategoryStruct.whiskerRight f Y_1) f …
      ⊢ Eq f g
    -/
    replace h := congr_fun h (𝟙_ C)
    /-
      C : Type u
      𝒞 : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.MonoidalCategory C
      W X✝ Y✝ Z X Y : C
      f g : Quiver.Hom X Y
      h : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight f CategoryTheory.Mo …
      ⊢ Eq f g
    -/
    simpa using h
    /-
      🎉 no goals
    -/


/-- Tensoring on the right, as a functor from `C` into endofunctors of `C`.

We later show this is a monoidal functor.
-/
abbrev tensoringRight : C ⥤ C ⥤ C := (curriedTensor C).flip


instance : (tensoringRight C).Faithful where
  map_injective {X} {Y} f g h := by
    /-
      C : Type u
      𝒞 : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.MonoidalCategory C
      W X✝ Y✝ Z X Y : C
      f g : Quiver.Hom X Y
      h : Eq ((CategoryTheory.MonoidalCategory.tensoringRight C).map f) ((CategoryTh …
      ⊢ Eq f g
    -/
    injections h
    /-
      C : Type u
      𝒞 : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.MonoidalCategory C
      W X✝ Y✝ Z X Y : C
      f g : Quiver.Hom X Y
      h : Eq (fun j => ((CategoryTheory.MonoidalCategory.curriedTensor C).obj j).map …
      ⊢ Eq f g
    -/
    replace h := congr_fun h (𝟙_ C)
    /-
      C : Type u
      𝒞 : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.MonoidalCategory C
      W X✝ Y✝ Z X Y : C
      f g : Quiver.Hom X Y
      h : Eq (((CategoryTheory.MonoidalCategory.curriedTensor C).obj CategoryTheory. …
      ⊢ Eq f g
    -/
    simpa using h
    /-
      🎉 no goals
    -/


/-- Tensoring on the right with `X ⊗ Y` is naturally isomorphic to
tensoring on the right with `X`, and then again with `Y`.
-/
def tensorRightTensor (X Y : C) : tensorRight (X ⊗ Y) ≅ tensorRight X ⋙ tensorRight Y :=
                                                                              /-
                                                                                C : Type u
                                                                                𝒞 : CategoryTheory.Category.{v, u} C
                                                                                inst✝ : CategoryTheory.MonoidalCategory C
                                                                                W X✝ Y✝ Z✝ X Y Z Z' : C
                                                                                f : Quiver.Hom Z Z'
                                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
                                                                              -/
  NatIso.ofComponents (fun Z => (associator Z X Y).symm) fun {Z} {Z'} f => by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem tensorRightTensor_hom_app (X Y Z : C) :
    (tensorRightTensor X Y).hom.app Z = (associator Z X Y).inv :=
  rfl


@[simp]
theorem tensorRightTensor_inv_app (X Y Z : C) :
                                                                     /-
                                                                       C : Type u
                                                                       𝒞 : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.MonoidalCategory C
                                                                       X Y Z : C
                                                                       ⊢ Eq ((CategoryTheory.MonoidalCategory.tensorRightTensor X Y).inv.app Z) (Cate …
                                                                     -/
    (tensorRightTensor X Y).inv.app Z = (associator Z X Y).hom := by simp [tensorRightTensor]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simps! tensorObj tensorHom tensorUnit whiskerLeft whiskerRight associator]
instance prodMonoidal : MonoidalCategory (C₁ × C₂) where
  tensorObj X Y := (X.1 ⊗ Y.1, X.2 ⊗ Y.2)
  tensorHom f g := (f.1 ⊗ g.1, f.2 ⊗ g.2)
  whiskerLeft X _ _ f := (whiskerLeft X.1 f.1, whiskerLeft X.2 f.2)
  whiskerRight f X := (whiskerRight f.1 X.1, whiskerRight f.2 X.2)
                      /-
                        C : Type u
                        𝒞 : CategoryTheory.Category.{v, u} C
                        inst✝⁴ : CategoryTheory.MonoidalCategory C
                        W X Y Z : C
                        C₁ : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
                        inst✝² : CategoryTheory.MonoidalCategory C₁
                        C₂ : Type u₂
                        inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
                        inst✝ : CategoryTheory.MonoidalCategory C₂
                        ⊢ ∀ {X₁ Y₁ X₂ Y₂ : Prod C₁ C₂} (f : Quiver.Hom X₁ Y₁) (g : Quiver.Hom X₂ Y₂),  …
                      -/
  tensorHom_def := by simp [tensorHom_def]
                      /-
                        🎉 no goals
                      -/
  tensorUnit := (𝟙_ C₁, 𝟙_ C₂)
  associator X Y Z := (α_ X.1 Y.1 Z.1).prod (α_ X.2 Y.2 Z.2)
  leftUnitor := fun ⟨X₁, X₂⟩ => (λ_ X₁).prod (λ_ X₂)
  rightUnitor := fun ⟨X₁, X₂⟩ => (ρ_ X₁).prod (ρ_ X₂)


@[simp]
theorem prodMonoidal_leftUnitor_hom_fst (X : C₁ × C₂) :
    ((λ_ X).hom : 𝟙_ _ ⊗ X ⟶ X).1 = (λ_ X.1).hom := by
  /-
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    X : Prod C₁ C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).hom.1 (CategoryTheor …
  -/
  cases X
  /-
    case mk
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    fst✝ : C₁
    snd✝ : C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor { fst := fst✝, snd := s …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem prodMonoidal_leftUnitor_hom_snd (X : C₁ × C₂) :
    ((λ_ X).hom : 𝟙_ _ ⊗ X ⟶ X).2 = (λ_ X.2).hom := by
  /-
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    X : Prod C₁ C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).hom.2 (CategoryTheor …
  -/
  cases X
  /-
    case mk
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    fst✝ : C₁
    snd✝ : C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor { fst := fst✝, snd := s …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem prodMonoidal_leftUnitor_inv_fst (X : C₁ × C₂) :
    ((λ_ X).inv : X ⟶ 𝟙_ _ ⊗ X).1 = (λ_ X.1).inv := by
  /-
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    X : Prod C₁ C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).inv.1 (CategoryTheor …
  -/
  cases X
  /-
    case mk
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    fst✝ : C₁
    snd✝ : C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor { fst := fst✝, snd := s …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem prodMonoidal_leftUnitor_inv_snd (X : C₁ × C₂) :
    ((λ_ X).inv : X ⟶ 𝟙_ _ ⊗ X).2 = (λ_ X.2).inv := by
  /-
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    X : Prod C₁ C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor X).inv.2 (CategoryTheor …
  -/
  cases X
  /-
    case mk
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    fst✝ : C₁
    snd✝ : C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor { fst := fst✝, snd := s …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem prodMonoidal_rightUnitor_hom_fst (X : C₁ × C₂) :
    ((ρ_ X).hom : X ⊗ 𝟙_ _ ⟶ X).1 = (ρ_ X.1).hom := by
  /-
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    X : Prod C₁ C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).hom.1 (CategoryTheo …
  -/
  cases X
  /-
    case mk
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    fst✝ : C₁
    snd✝ : C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor { fst := fst✝, snd :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem prodMonoidal_rightUnitor_hom_snd (X : C₁ × C₂) :
    ((ρ_ X).hom : X ⊗ 𝟙_ _ ⟶ X).2 = (ρ_ X.2).hom := by
  /-
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    X : Prod C₁ C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).hom.2 (CategoryTheo …
  -/
  cases X
  /-
    case mk
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    fst✝ : C₁
    snd✝ : C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor { fst := fst✝, snd :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem prodMonoidal_rightUnitor_inv_fst (X : C₁ × C₂) :
    ((ρ_ X).inv : X ⟶ X ⊗ 𝟙_ _).1 = (ρ_ X.1).inv := by
  /-
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    X : Prod C₁ C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).inv.1 (CategoryTheo …
  -/
  cases X
  /-
    case mk
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    fst✝ : C₁
    snd✝ : C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor { fst := fst✝, snd :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem prodMonoidal_rightUnitor_inv_snd (X : C₁ × C₂) :
    ((ρ_ X).inv : X ⟶ X ⊗ 𝟙_ _).2 = (ρ_ X.2).inv := by
  /-
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    X : Prod C₁ C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor X).inv.2 (CategoryTheo …
  -/
  cases X
  /-
    case mk
    C₁ : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.MonoidalCategory C₁
    C₂ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝ : CategoryTheory.MonoidalCategory C₂
    fst✝ : C₁
    snd✝ : C₂
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor { fst := fst✝, snd :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma tensor_naturality {X Y X' Y' : J} (f : X ⟶ Y) (g : X' ⟶ Y') :
    (F.map f ⊗ G.map g) ≫ (α.app Y ⊗ β.app Y') =
      (α.app X ⊗ β.app X') ≫ (F'.map f ⊗ G'.map g) := by
  /-
    J : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.MonoidalCategory C
    F G F' G' : CategoryTheory.Functor J C
    α : Quiver.Hom F F'
    β : Quiver.Hom G G'
    X Y X' Y' : J
    f : Quiver.Hom X Y
    g : Quiver.Hom X' Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [← tensor_comp, naturality]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma whiskerRight_app_tensor_app {X Y : J} (f : X ⟶ Y) (X' : J) :
    F.map f ▷ G.obj X' ≫ (α.app Y ⊗ β.app X') =
      (α.app X ⊗ β.app X') ≫ F'.map f ▷ (G'.obj X') := by
  /-
    J : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.MonoidalCategory C
    F G F' G' : CategoryTheory.Functor J C
    α : Quiver.Hom F F'
    β : Quiver.Hom G G'
    X Y : J
    f : Quiver.Hom X Y
    X' : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simpa using tensor_naturality α β f (𝟙 X')
  /-
    🎉 no goals
  -/


@[reassoc]
lemma whiskerLeft_app_tensor_app {X' Y' : J} (f : X' ⟶ Y') (X : J) :
    F.obj X ◁ G.map f ≫ (α.app X ⊗ β.app Y') =
      (α.app X ⊗ β.app X') ≫ F'.obj X ◁ G'.map f := by
  /-
    J : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.MonoidalCategory C
    F G F' G' : CategoryTheory.Functor J C
    α : Quiver.Hom F F'
    β : Quiver.Hom G G'
    X' Y' : J
    f : Quiver.Hom X' Y'
    X : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simpa using tensor_naturality α β (𝟙 X) f
  /-
    🎉 no goals
  -/


