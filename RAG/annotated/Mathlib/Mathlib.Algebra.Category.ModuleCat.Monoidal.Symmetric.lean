/-- (implementation) the braiding for R-modules -/
def braiding (M N : ModuleCat.{u} R) : M ⊗ N ≅ N ⊗ M :=
  LinearEquiv.toModuleIso (TensorProduct.comm R M N)


@[simp]
theorem braiding_naturality {X₁ X₂ Y₁ Y₂ : ModuleCat.{u} R} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
    (f ⊗ g) ≫ (Y₁.braiding Y₂).hom = (X₁.braiding X₂).hom ≫ (g ⊗ f) := by
  /-
    R : Type u
    inst✝ : CommRing R
    X₁ X₂ Y₁ Y₂ : ModuleCat R
    f : Quiver.Hom X₁ Y₁
    g : Quiver.Hom X₂ Y₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext : 1
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    X₁ X₂ Y₁ Y₂ : ModuleCat R
    f : Quiver.Hom X₁ Y₁
    g : Quiver.Hom X₂ Y₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply TensorProduct.ext'
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X₁ X₂ Y₁ Y₂ : ModuleCat R
    f : Quiver.Hom X₁ Y₁
    g : Quiver.Hom X₂ Y₂
    ⊢ ∀ (x : ↑X₁) (y : ↑X₂), Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheo …
  -/
  intro x y
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X₁ X₂ Y₁ Y₂ : ModuleCat R
    f : Quiver.Hom X₁ Y₁
    g : Quiver.Hom X₂ Y₂
    x : ↑X₁
    y : ↑X₂
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem braiding_naturality_left {X Y : ModuleCat R} (f : X ⟶ Y) (Z : ModuleCat R) :
    f ▷ Z ≫ (braiding Y Z).hom = (braiding X Z).hom ≫ Z ◁ f := by
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : ModuleCat R
    f : Quiver.Hom X Y
    Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp_rw [← id_tensorHom]
  /-
    R : Type u
    inst✝ : CommRing R
    X Y : ModuleCat R
    f : Quiver.Hom X Y
    Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply braiding_naturality
  /-
    🎉 no goals
  -/


@[simp]
theorem braiding_naturality_right (X : ModuleCat R) {Y Z : ModuleCat R} (f : Y ⟶ Z) :
    X ◁ f ≫ (braiding X Z).hom = (braiding X Y).hom ≫ f ▷ X := by
  /-
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    f : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp_rw [← id_tensorHom]
  /-
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    f : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply braiding_naturality
  /-
    🎉 no goals
  -/


@[simp]
theorem hexagon_forward (X Y Z : ModuleCat.{u} R) :
    (α_ X Y Z).hom ≫ (braiding X _).hom ≫ (α_ Y Z X).hom =
      (braiding X Y).hom ▷ Z ≫ (α_ Y X Z).hom ≫ Y ◁ (braiding X Z).hom := by
  /-
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext : 1
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply TensorProduct.ext_threefold
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    ⊢ ∀ (x : ↑X) (y : ↑Y) (z : ↑Z), Eq ((CategoryTheory.CategoryStruct.comp (Categ …
  -/
  intro x y z
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    x : ↑X
    y : ↑Y
    z : ↑Z
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem hexagon_reverse (X Y Z : ModuleCat.{u} R) :
    (α_ X Y Z).inv ≫ (braiding _ Z).hom ≫ (α_ Z X Y).inv =
      X ◁ (Y.braiding Z).hom ≫ (α_ X Z Y).inv ≫ (X.braiding Z).hom ▷ Y := by
  /-
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply (cancel_epi (α_ X Y Z).hom).1
  /-
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext : 1
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply TensorProduct.ext_threefold
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    ⊢ ∀ (x : ↑X) (y : ↑Y) (z : ↑Z), Eq ((CategoryTheory.CategoryStruct.comp (Categ …
  -/
  intro x y z
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X Y Z : ModuleCat R
    x : ↑X
    y : ↑Y
    z : ↑Z
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The symmetric monoidal structure on `Module R`. -/
instance symmetricCategory : SymmetricCategory (ModuleCat.{u} R) where
  braiding := braiding
  braiding_naturality_left := braiding_naturality_left
  braiding_naturality_right := braiding_naturality_right
  hexagon_forward := hexagon_forward
  hexagon_reverse := hexagon_reverse
  -- Porting note: this proof was automatic in Lean3
  -- now `aesop` is applying `ModuleCat.ext` in favour of `TensorProduct.ext`.
  symmetry _ _ := by
    /-
      R : Type u
      inst✝ : CommRing R
      x✝¹ x✝ : ModuleCat R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
    -/
    ext : 1
    /-
      case hf
      R : Type u
      inst✝ : CommRing R
      x✝¹ x✝ : ModuleCat R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
    -/
    apply TensorProduct.ext'
    /-
      case hf.H
      R : Type u
      inst✝ : CommRing R
      x✝¹ x✝ : ModuleCat R
      ⊢ ∀ (x : ↑x✝¹) (y : ↑x✝), Eq ((CategoryTheory.CategoryStruct.comp (CategoryThe …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


@[simp]
theorem braiding_hom_apply {M N : ModuleCat.{u} R} (m : M) (n : N) :
    ((β_ M N).hom : M ⊗ N ⟶ N ⊗ M) (m ⊗ₜ n) = n ⊗ₜ m :=
  rfl


@[simp]
theorem braiding_inv_apply {M N : ModuleCat.{u} R} (m : M) (n : N) :
    ((β_ M N).inv : N ⊗ M ⟶ M ⊗ N) (n ⊗ₜ m) = m ⊗ₜ n :=
  rfl


theorem tensorμ_eq_tensorTensorTensorComm {A B C D : ModuleCat R} :
    tensorμ A B C D = ofHom (TensorProduct.tensorTensorTensorComm R A B C D).toLinearMap :=
  ModuleCat.hom_ext <| TensorProduct.ext <| TensorProduct.ext <| LinearMap.ext₂ fun _ _ =>
    TensorProduct.ext <| LinearMap.ext₂ fun _ _ => rfl


@[simp]
theorem tensorμ_apply
    {A B C D : ModuleCat R} (x : A) (y : B) (z : C) (w : D) :
    tensorμ A B C D ((x ⊗ₜ y) ⊗ₜ (z ⊗ₜ w)) = (x ⊗ₜ z) ⊗ₜ (y ⊗ₜ w) := rfl


