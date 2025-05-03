/-- A category is `MonoidalPreadditive` if tensoring is additive in both factors.

Note we don't `extend Preadditive C` here, as `Abelian C` already extends it,
and we'll need to have both typeclasses sometimes.
-/
class MonoidalPreadditive : Prop where
  whiskerLeft_zero : ∀ {X Y Z : C}, X ◁ (0 : Y ⟶ Z) = 0 := by aesop_cat
  zero_whiskerRight : ∀ {X Y Z : C}, (0 : Y ⟶ Z) ▷ X = 0 := by aesop_cat
  whiskerLeft_add : ∀ {X Y Z : C} (f g : Y ⟶ Z), X ◁ (f + g) = X ◁ f + X ◁ g := by aesop_cat
  add_whiskerRight : ∀ {X Y Z : C} (f g : Y ⟶ Z), (f + g) ▷ X = f ▷ X + g ▷ X := by aesop_cat


@[simp (low)]
theorem tensor_zero {W X Y Z : C} (f : W ⟶ X) : f ⊗ (0 : Y ⟶ Z) = 0 := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.MonoidalPreadditive C
    W X Y Z : C
    f : Quiver.Hom W X
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f 0) 0
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/

-- The priority setting will not be needed when we replace `f ⊗ 𝟙 X` by `f ▷ X`.

@[simp (low)]
theorem zero_tensor {W X Y Z : C} (f : Y ⟶ Z) : (0 : W ⟶ X) ⊗ f = 0 := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.MonoidalPreadditive C
    W X Y Z : C
    f : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom 0 f) 0
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/


theorem tensor_add {W X Y Z : C} (f : W ⟶ X) (g h : Y ⟶ Z) : f ⊗ (g + h) = f ⊗ g + f ⊗ h := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.MonoidalPreadditive C
    W X Y Z : C
    f : Quiver.Hom W X
    g h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f (HAdd.hAdd g h)) (HAdd …
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/


theorem add_tensor {W X Y Z : C} (f g : W ⟶ X) (h : Y ⟶ Z) : (f + g) ⊗ h = f ⊗ h + g ⊗ h := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.MonoidalPreadditive C
    W X Y Z : C
    f g : Quiver.Hom W X
    h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (HAdd.hAdd f g) h) (HAdd …
  -/
  simp [tensorHom_def]
  /-
    🎉 no goals
  -/


instance tensorLeft_additive (X : C) : (tensorLeft X).Additive where


instance tensorRight_additive (X : C) : (tensorRight X).Additive where


instance tensoringLeft_additive (X : C) : ((tensoringLeft C).obj X).Additive where


instance tensoringRight_additive (X : C) : ((tensoringRight C).obj X).Additive where


/-- A faithful additive monoidal functor to a monoidal preadditive category
ensures that the domain is monoidal preadditive. -/
theorem monoidalPreadditive_of_faithful {D} [Category D] [Preadditive D] [MonoidalCategory D]
    (F : D ⥤ C) [F.Monoidal] [F.Faithful] [F.Additive] :
    MonoidalPreadditive D :=
  { whiskerLeft_zero := by
      /-
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        ⊢ ∀ {X Y Z : D}, Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X 0) 0
      -/
      intros
      /-
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        X✝ Y✝ Z✝ : D
        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ 0) 0
      -/
      apply F.map_injective
      /-
        case a
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        X✝ Y✝ Z✝ : D
        ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ 0)) (F.map 0)
      -/
      simp [Functor.Monoidal.map_whiskerLeft]
      /-
        🎉 no goals
      -/
    zero_whiskerRight := by
      /-
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        ⊢ ∀ {X Y Z : D}, Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight 0 X) 0
      -/
      intros
      /-
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        X✝ Y✝ Z✝ : D
        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight 0 X✝) 0
      -/
      apply F.map_injective
      /-
        case a
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        X✝ Y✝ Z✝ : D
        ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.whiskerRight 0 X✝)) (F.map 0)
      -/
      simp [Functor.Monoidal.map_whiskerRight]
      /-
        🎉 no goals
      -/
    whiskerLeft_add := by
      /-
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        ⊢ ∀ {X Y Z : D} (f g : Quiver.Hom Y Z), Eq (CategoryTheory.MonoidalCategoryStr …
      -/
      intros
      /-
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        X✝ Y✝ Z✝ : D
        f✝ g✝ : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ (HAdd.hAdd f✝ g✝))  …
      -/
      apply F.map_injective
      simp only [Functor.Monoidal.map_whiskerLeft, Functor.map_add, Preadditive.comp_add,
        Preadditive.add_comp, MonoidalPreadditive.whiskerLeft_add]
    add_whiskerRight := by
      /-
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        ⊢ ∀ {X Y Z : D} (f g : Quiver.Hom Y Z), Eq (CategoryTheory.MonoidalCategoryStr …
      -/
      intros
      /-
        C : Type u_1
        inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁸ : CategoryTheory.Preadditive C
        inst✝⁷ : CategoryTheory.MonoidalCategory C
        inst✝⁶ : CategoryTheory.MonoidalPreadditive C
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : CategoryTheory.MonoidalCategory D
        F : CategoryTheory.Functor D C
        inst✝² : F.Monoidal
        inst✝¹ : F.Faithful
        inst✝ : F.Additive
        X✝ Y✝ Z✝ : D
        f✝ g✝ : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (HAdd.hAdd f✝ g✝) X✝) …
      -/
      apply F.map_injective
      simp only [Functor.Monoidal.map_whiskerRight, Functor.map_add, Preadditive.comp_add,
        Preadditive.add_comp, MonoidalPreadditive.add_whiskerRight] }


theorem whiskerLeft_sum (P : C) {Q R : C} {J : Type*} (s : Finset J) (g : J → (Q ⟶ R)) :
    P ◁ ∑ j ∈ s, g j = ∑ j ∈ s, P ◁ g j :=
  map_sum ((tensoringLeft C).obj P).mapAddHom g s


theorem sum_whiskerRight {Q R : C} {J : Type*} (s : Finset J) (g : J → (Q ⟶ R)) (P : C) :
    (∑ j ∈ s, g j) ▷ P = ∑ j ∈ s, g j ▷ P :=
  map_sum ((tensoringRight C).obj P).mapAddHom g s


theorem tensor_sum {P Q R S : C} {J : Type*} (s : Finset J) (f : P ⟶ Q) (g : J → (R ⟶ S)) :
    (f ⊗ ∑ j ∈ s, g j) = ∑ j ∈ s, f ⊗ g j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.MonoidalPreadditive C
    P Q R S : C
    J : Type u_2
    s : Finset J
    f : Quiver.Hom P Q
    g : J → Quiver.Hom R S
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom f (s.sum fun j => g j))  …
  -/
  simp only [tensorHom_def, whiskerLeft_sum, Preadditive.comp_sum]
  /-
    🎉 no goals
  -/


theorem sum_tensor {P Q R S : C} {J : Type*} (s : Finset J) (f : P ⟶ Q) (g : J → (R ⟶ S)) :
    (∑ j ∈ s, g j) ⊗ f = ∑ j ∈ s, g j ⊗ f := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.MonoidalPreadditive C
    P Q R S : C
    J : Type u_2
    s : Finset J
    f : Quiver.Hom P Q
    g : J → Quiver.Hom R S
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (s.sum fun j => g j) f)  …
  -/
  simp only [tensorHom_def, sum_whiskerRight, Preadditive.sum_comp]
  /-
    🎉 no goals
  -/

-- In a closed monoidal category, this would hold because
-- `tensorLeft X` is a left adjoint and hence preserves all colimits.
-- In any case it is true in any preadditive category.

instance (X : C) : PreservesFiniteBiproducts (tensorLeft X) where
  preserves {J} :=
    { preserves := fun {f} =>
        { preserves := fun {b} i => ⟨isBilimitOfTotal _ (by
            /-
              C : Type u_1
              inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
              inst✝³ : CategoryTheory.Preadditive C
              inst✝² : CategoryTheory.MonoidalCategory C
              inst✝¹ : CategoryTheory.MonoidalPreadditive C
              X : C
              J : Type
              inst✝ : Fintype J
              f : J → C
              b : CategoryTheory.Limits.Bicone f
              i : b.IsBilimit
              ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (((CategoryT …
            -/
            dsimp
            /-
              C : Type u_1
              inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
              inst✝³ : CategoryTheory.Preadditive C
              inst✝² : CategoryTheory.MonoidalCategory C
              inst✝¹ : CategoryTheory.MonoidalPreadditive C
              X : C
              J : Type
              inst✝ : Fintype J
              f : J → C
              b : CategoryTheory.Limits.Bicone f
              i : b.IsBilimit
              ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (CategoryThe …
            -/
            simp_rw [← id_tensorHom]
            simp only [← tensor_comp, Category.comp_id, ← tensor_sum, ← tensor_id,
              IsBilimit.total i])⟩ } }


instance (X : C) : PreservesFiniteBiproducts (tensorRight X) where
  preserves {J} :=
    { preserves := fun {f} =>
        { preserves := fun {b} i => ⟨isBilimitOfTotal _ (by
            /-
              C : Type u_1
              inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
              inst✝³ : CategoryTheory.Preadditive C
              inst✝² : CategoryTheory.MonoidalCategory C
              inst✝¹ : CategoryTheory.MonoidalPreadditive C
              X : C
              J : Type
              inst✝ : Fintype J
              f : J → C
              b : CategoryTheory.Limits.Bicone f
              i : b.IsBilimit
              ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (((CategoryT …
            -/
            dsimp
            /-
              C : Type u_1
              inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
              inst✝³ : CategoryTheory.Preadditive C
              inst✝² : CategoryTheory.MonoidalCategory C
              inst✝¹ : CategoryTheory.MonoidalPreadditive C
              X : C
              J : Type
              inst✝ : Fintype J
              f : J → C
              b : CategoryTheory.Limits.Bicone f
              i : b.IsBilimit
              ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (CategoryThe …
            -/
            simp_rw [← tensorHom_id]
            simp only [← tensor_comp, Category.comp_id, ← sum_tensor, ← tensor_id,
               IsBilimit.total i])⟩ } }


/-- The isomorphism showing how tensor product on the left distributes over direct sums. -/
def leftDistributor {J : Type} [Fintype J] (X : C) (f : J → C) : X ⊗ ⨁ f ≅ ⨁ fun j => X ⊗ f j :=
  (tensorLeft X).mapBiproduct f


theorem leftDistributor_hom {J : Type} [Fintype J] (X : C) (f : J → C) :
    (leftDistributor X f).hom =
      ∑ j : J, (X ◁ biproduct.π f j) ≫ biproduct.ι (fun j => X ⊗ f j) j := by
  classical
  ext
  dsimp [leftDistributor, Functor.mapBiproduct, Functor.mapBicone]
  erw [biproduct.lift_π]
  simp only [Preadditive.sum_comp, Category.assoc, biproduct.ι_π, comp_dite, comp_zero,
    Finset.sum_dite_eq', Finset.mem_univ, ite_true, eqToHom_refl, Category.comp_id]


theorem leftDistributor_inv {J : Type} [Fintype J] (X : C) (f : J → C) :
    (leftDistributor X f).inv = ∑ j : J, biproduct.π _ j ≫ (X ◁ biproduct.ι f j) := by
  classical
  ext
  dsimp [leftDistributor, Functor.mapBiproduct, Functor.mapBicone]
  simp only [Preadditive.comp_sum, biproduct.ι_π_assoc, dite_comp, zero_comp,
    Finset.sum_dite_eq, Finset.mem_univ, ite_true, eqToHom_refl, Category.id_comp,
    biproduct.ι_desc]


@[reassoc (attr := simp)]
theorem leftDistributor_hom_comp_biproduct_π {J : Type} [Fintype J] (X : C) (f : J → C) (j : J) :
    (leftDistributor X f).hom ≫ biproduct.π _ j = X ◁ biproduct.π _ j := by
  classical
  simp [leftDistributor_hom, Preadditive.sum_comp, biproduct.ι_π, comp_dite]


@[reassoc (attr := simp)]
theorem biproduct_ι_comp_leftDistributor_hom {J : Type} [Fintype J] (X : C) (f : J → C) (j : J) :
    (X ◁ biproduct.ι _ j) ≫ (leftDistributor X f).hom = biproduct.ι (fun j => X ⊗ f j) j := by
  classical
  simp [leftDistributor_hom, Preadditive.comp_sum, ← MonoidalCategory.whiskerLeft_comp_assoc,
    biproduct.ι_π, whiskerLeft_dite, dite_comp]


@[reassoc (attr := simp)]
theorem leftDistributor_inv_comp_biproduct_π {J : Type} [Fintype J] (X : C) (f : J → C) (j : J) :
    (leftDistributor X f).inv ≫ (X ◁ biproduct.π _ j) = biproduct.π _ j := by
  classical
  simp [leftDistributor_inv, Preadditive.sum_comp, ← MonoidalCategory.whiskerLeft_comp,
    biproduct.ι_π, whiskerLeft_dite, comp_dite]


@[reassoc (attr := simp)]
theorem biproduct_ι_comp_leftDistributor_inv {J : Type} [Fintype J] (X : C) (f : J → C) (j : J) :
    biproduct.ι _ j ≫ (leftDistributor X f).inv = X ◁ biproduct.ι _ j := by
  classical
  simp [leftDistributor_inv, Preadditive.comp_sum, ← id_tensor_comp, biproduct.ι_π_assoc, dite_comp]


theorem leftDistributor_assoc {J : Type} [Fintype J] (X Y : C) (f : J → C) :
    (asIso (𝟙 X) ⊗ leftDistributor Y f) ≪≫ leftDistributor X _ =
      (α_ X Y (⨁ f)).symm ≪≫ leftDistributor (X ⊗ Y) f ≪≫ biproduct.mapIso fun _ => α_ X Y _ := by
  classical
  ext
  simp only [Category.comp_id, Category.assoc, eqToHom_refl, Iso.trans_hom, Iso.symm_hom,
    asIso_hom, comp_zero, comp_dite, Preadditive.sum_comp, Preadditive.comp_sum, tensor_sum,
    id_tensor_comp, tensorIso_hom, leftDistributor_hom, biproduct.mapIso_hom, biproduct.ι_map,
    biproduct.ι_π, Finset.sum_dite_irrel, Finset.sum_dite_eq', Finset.sum_const_zero]
  simp_rw [← id_tensorHom]
  simp only [← id_tensor_comp, biproduct.ι_π]
  simp only [id_tensor_comp, tensor_dite, comp_dite]
  simp


/-- The isomorphism showing how tensor product on the right distributes over direct sums. -/
def rightDistributor {J : Type} [Fintype J] (f : J → C) (X : C) : (⨁ f) ⊗ X ≅ ⨁ fun j => f j ⊗ X :=
  (tensorRight X).mapBiproduct f


theorem rightDistributor_hom {J : Type} [Fintype J] (f : J → C) (X : C) :
    (rightDistributor f X).hom =
      ∑ j : J, (biproduct.π f j ▷ X) ≫ biproduct.ι (fun j => f j ⊗ X) j := by
  classical
  ext
  dsimp [rightDistributor, Functor.mapBiproduct, Functor.mapBicone]
  erw [biproduct.lift_π]
  simp only [Preadditive.sum_comp, Category.assoc, biproduct.ι_π, comp_dite, comp_zero,
    Finset.sum_dite_eq', Finset.mem_univ, eqToHom_refl, Category.comp_id, ite_true]


theorem rightDistributor_inv {J : Type} [Fintype J] (f : J → C) (X : C) :
    (rightDistributor f X).inv = ∑ j : J, biproduct.π _ j ≫ (biproduct.ι f j ▷ X) := by
  classical
  ext
  dsimp [rightDistributor, Functor.mapBiproduct, Functor.mapBicone]
  simp only [biproduct.ι_desc, Preadditive.comp_sum, ne_eq, biproduct.ι_π_assoc, dite_comp,
    zero_comp, Finset.sum_dite_eq, Finset.mem_univ, eqToHom_refl, Category.id_comp, ite_true]


@[reassoc (attr := simp)]
theorem rightDistributor_hom_comp_biproduct_π {J : Type} [Fintype J] (f : J → C) (X : C) (j : J) :
    (rightDistributor f X).hom ≫ biproduct.π _ j = biproduct.π _ j ▷ X := by
  classical
  simp [rightDistributor_hom, Preadditive.sum_comp, biproduct.ι_π, comp_dite]


@[reassoc (attr := simp)]
theorem biproduct_ι_comp_rightDistributor_hom {J : Type} [Fintype J] (f : J → C) (X : C) (j : J) :
    (biproduct.ι _ j ▷ X) ≫ (rightDistributor f X).hom = biproduct.ι (fun j => f j ⊗ X) j := by
  classical
  simp [rightDistributor_hom, Preadditive.comp_sum, ← comp_whiskerRight_assoc, biproduct.ι_π,
    dite_whiskerRight, dite_comp]


@[reassoc (attr := simp)]
theorem rightDistributor_inv_comp_biproduct_π {J : Type} [Fintype J] (f : J → C) (X : C) (j : J) :
    (rightDistributor f X).inv ≫ (biproduct.π _ j ▷ X) = biproduct.π _ j := by
  classical
  simp [rightDistributor_inv, Preadditive.sum_comp, ← MonoidalCategory.comp_whiskerRight,
    biproduct.ι_π, dite_whiskerRight, comp_dite]


@[reassoc (attr := simp)]
theorem biproduct_ι_comp_rightDistributor_inv {J : Type} [Fintype J] (f : J → C) (X : C) (j : J) :
    biproduct.ι _ j ≫ (rightDistributor f X).inv = biproduct.ι _ j ▷ X := by
  classical
  simp [rightDistributor_inv, Preadditive.comp_sum, ← id_tensor_comp, biproduct.ι_π_assoc,
    dite_comp]


theorem rightDistributor_assoc {J : Type} [Fintype J] (f : J → C) (X Y : C) :
    (rightDistributor f X ⊗ asIso (𝟙 Y)) ≪≫ rightDistributor _ Y =
      α_ (⨁ f) X Y ≪≫ rightDistributor f (X ⊗ Y) ≪≫ biproduct.mapIso fun _ => (α_ _ X Y).symm := by
  classical
  ext
  simp only [Category.comp_id, Category.assoc, eqToHom_refl, Iso.symm_hom, Iso.trans_hom,
    asIso_hom, comp_zero, comp_dite, Preadditive.sum_comp, Preadditive.comp_sum, sum_tensor,
    comp_tensor_id, tensorIso_hom, rightDistributor_hom, biproduct.mapIso_hom, biproduct.ι_map,
    biproduct.ι_π, Finset.sum_dite_irrel, Finset.sum_dite_eq', Finset.sum_const_zero,
    Finset.mem_univ, if_true]
  simp_rw [← tensorHom_id]
  simp only [← comp_tensor_id, biproduct.ι_π, dite_tensor, comp_dite]
  simp


theorem leftDistributor_rightDistributor_assoc {J : Type _} [Fintype J]
    (X : C) (f : J → C) (Y : C) :
    (leftDistributor X f ⊗ asIso (𝟙 Y)) ≪≫ rightDistributor _ Y =
      α_ X (⨁ f) Y ≪≫
        (asIso (𝟙 X) ⊗ rightDistributor _ Y) ≪≫
          leftDistributor X _ ≪≫ biproduct.mapIso fun _ => (α_ _ _ _).symm := by
  classical
  ext
  simp only [Category.comp_id, Category.assoc, eqToHom_refl, Iso.symm_hom, Iso.trans_hom,
    asIso_hom, comp_zero, comp_dite, Preadditive.sum_comp, Preadditive.comp_sum, sum_tensor,
    tensor_sum, comp_tensor_id, tensorIso_hom, leftDistributor_hom, rightDistributor_hom,
    biproduct.mapIso_hom, biproduct.ι_map, biproduct.ι_π, Finset.sum_dite_irrel,
    Finset.sum_dite_eq', Finset.sum_const_zero, Finset.mem_univ, if_true]
  simp_rw [← tensorHom_id, ← id_tensorHom]
  simp only [← comp_tensor_id, ← id_tensor_comp_assoc, Category.assoc, biproduct.ι_π, comp_dite,
    dite_comp, tensor_dite, dite_tensor]
  simp


@[ext]
theorem leftDistributor_ext_left {J : Type} [Fintype J] {X Y : C} {f : J → C} {g h : X ⊗ ⨁ f ⟶ Y}
    (w : ∀ j, (X ◁ biproduct.ι f j) ≫ g = (X ◁ biproduct.ι f j) ≫ h) : g = h := by
  classical
  apply (cancel_epi (leftDistributor X f).inv).mp
  ext
  simp? [leftDistributor_inv, Preadditive.comp_sum_assoc, biproduct.ι_π_assoc, dite_comp] says
    simp only [leftDistributor_inv, Preadditive.comp_sum_assoc, biproduct.ι_π_assoc, dite_comp,
      zero_comp, Finset.sum_dite_eq, Finset.mem_univ, ↓reduceIte, eqToHom_refl, Category.id_comp]
  apply w


@[ext]
theorem leftDistributor_ext_right {J : Type} [Fintype J] {X Y : C} {f : J → C} {g h : X ⟶ Y ⊗ ⨁ f}
    (w : ∀ j, g ≫ (Y ◁ biproduct.π f j) = h ≫ (Y ◁ biproduct.π f j)) : g = h := by
  classical
  apply (cancel_mono (leftDistributor Y f).hom).mp
  ext
  simp? [leftDistributor_hom, Preadditive.sum_comp, Preadditive.comp_sum_assoc, biproduct.ι_π,
      comp_dite] says
    simp only [leftDistributor_hom, Category.assoc, Preadditive.sum_comp, biproduct.ι_π, comp_dite,
      comp_zero, Finset.sum_dite_eq', Finset.mem_univ, ↓reduceIte, eqToHom_refl, Category.comp_id]

  apply w

-- One might wonder how many iterated tensor products we need simp lemmas for.
-- The answer is two: this lemma is needed to verify the pentagon identity.

@[ext]
theorem leftDistributor_ext₂_left {J : Type} [Fintype J]
    {X Y Z : C} {f : J → C} {g h : X ⊗ (Y ⊗ ⨁ f) ⟶ Z}
    (w : ∀ j, (X ◁ (Y ◁ biproduct.ι f j)) ≫ g = (X ◁ (Y ◁ biproduct.ι f j)) ≫ h) :
    g = h := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    X Y Z : C
    f : J → C
    g h : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X (CategoryT …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoidal …
    ⊢ Eq g h
  -/
  apply (cancel_epi (α_ _ _ _).hom).mp
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    X Y Z : C
    f : J → C
    g h : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X (CategoryT …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoidal …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext
  /-
    case w
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    X Y Z : C
    f : J → C
    g h : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj X (CategoryT …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoidal …
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [w]
  /-
    🎉 no goals
  -/


@[ext]
theorem leftDistributor_ext₂_right {J : Type} [Fintype J]
    {X Y Z : C} {f : J → C} {g h : X ⟶ Y ⊗ (Z ⊗ ⨁ f)}
    (w : ∀ j, g ≫ (Y ◁ (Z ◁ biproduct.π f j)) = h ≫ (Y ◁ (Z ◁ biproduct.π f j))) :
    g = h := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    X Y Z : C
    f : J → C
    g h : Quiver.Hom X (CategoryTheory.MonoidalCategoryStruct.tensorObj Y (Categor …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Monoid …
    ⊢ Eq g h
  -/
  apply (cancel_mono (α_ _ _ _).inv).mp
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    X Y Z : C
    f : J → C
    g h : Quiver.Hom X (CategoryTheory.MonoidalCategoryStruct.tensorObj Y (Categor …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Monoid …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.MonoidalCategoryStr …
  -/
  ext
  /-
    case w
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    X Y Z : C
    f : J → C
    g h : Quiver.Hom X (CategoryTheory.MonoidalCategoryStruct.tensorObj Y (Categor …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Monoid …
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
  -/
  simp [w]
  /-
    🎉 no goals
  -/


@[ext]
theorem rightDistributor_ext_left {J : Type} [Fintype J]
    {f : J → C} {X Y : C} {g h : (⨁ f) ⊗ X ⟶ Y}
    (w : ∀ j, (biproduct.ι f j ▷ X) ≫ g = (biproduct.ι f j ▷ X) ≫ h) : g = h := by
  classical
  apply (cancel_epi (rightDistributor f X).inv).mp
  ext
  simp? [rightDistributor_inv, Preadditive.comp_sum_assoc, biproduct.ι_π_assoc, dite_comp] says
    simp only [rightDistributor_inv, Preadditive.comp_sum_assoc, biproduct.ι_π_assoc, dite_comp,
      zero_comp, Finset.sum_dite_eq, Finset.mem_univ, ↓reduceIte, eqToHom_refl, Category.id_comp]
  apply w


@[ext]
theorem rightDistributor_ext_right {J : Type} [Fintype J]
    {f : J → C} {X Y : C} {g h : X ⟶ (⨁ f) ⊗ Y}
    (w : ∀ j, g ≫ (biproduct.π f j ▷ Y) = h ≫ (biproduct.π f j ▷ Y)) : g = h := by
  classical
  apply (cancel_mono (rightDistributor f Y).hom).mp
  ext
  simp? [rightDistributor_hom, Preadditive.sum_comp, Preadditive.comp_sum_assoc, biproduct.ι_π,
      comp_dite] says
    simp only [rightDistributor_hom, Category.assoc, Preadditive.sum_comp, biproduct.ι_π, comp_dite,
      comp_zero, Finset.sum_dite_eq', Finset.mem_univ, ↓reduceIte, eqToHom_refl, Category.comp_id]
  apply w


@[ext]
theorem rightDistributor_ext₂_left {J : Type} [Fintype J]
    {f : J → C} {X Y Z : C} {g h : ((⨁ f) ⊗ X) ⊗ Y ⟶ Z}
    (w : ∀ j, ((biproduct.ι f j ▷ X) ▷ Y) ≫ g = ((biproduct.ι f j ▷ X) ▷ Y) ≫ h) :
    g = h := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    f : J → C
    X Y Z : C
    g h : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryThe …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoidal …
    ⊢ Eq g h
  -/
  apply (cancel_epi (α_ _ _ _).inv).mp
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    f : J → C
    X Y Z : C
    g h : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryThe …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoidal …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext
  /-
    case w
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    f : J → C
    X Y Z : C
    g h : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryThe …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monoidal …
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp [w]
  /-
    🎉 no goals
  -/


@[ext]
theorem rightDistributor_ext₂_right {J : Type} [Fintype J]
    {f : J → C} {X Y Z : C} {g h : X ⟶ ((⨁ f) ⊗ Y) ⊗ Z}
    (w : ∀ j, g ≫ ((biproduct.π f j ▷ Y) ▷ Z) = h ≫ ((biproduct.π f j ▷ Y) ▷ Z)) :
    g = h := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    f : J → C
    X Y Z : C
    g h : Quiver.Hom X (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryT …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Monoid …
    ⊢ Eq g h
  -/
  apply (cancel_mono (α_ _ _ _).hom).mp
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    f : J → C
    X Y Z : C
    g h : Quiver.Hom X (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryT …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Monoid …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.MonoidalCategoryStr …
  -/
  ext
  /-
    case w
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    J : Type
    inst✝ : Fintype J
    f : J → C
    X Y Z : C
    g h : Quiver.Hom X (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryT …
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Monoid …
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
  -/
  simp [w]
  /-
    🎉 no goals
  -/


