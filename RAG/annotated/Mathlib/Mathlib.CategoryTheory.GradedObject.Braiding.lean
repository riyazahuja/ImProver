/-- The braiding `tensorObj X Y ≅ tensorObj Y X` when `X` and `Y` are graded objects
indexed by a commutative additive monoid. -/
noncomputable def braiding [HasTensor X Y] [HasTensor Y X] : tensorObj X Y ≅ tensorObj Y X where
  hom k := tensorObjDesc (fun i j hij => (β_ _ _).hom ≫
                             /-
                               I : Type u_1
                               inst✝⁵ : AddCommMonoid I
                               C : Type u_2
                               inst✝⁴ : CategoryTheory.Category.{?u.147, u_2} C
                               inst✝³ : CategoryTheory.MonoidalCategory C
                               X Y Z : CategoryTheory.GradedObject I C
                               inst✝² : CategoryTheory.BraidedCategory C
                               inst✝¹ : X.HasTensor Y
                               inst✝ : Y.HasTensor X
                               k i j : I
                               hij : Eq (HAdd.hAdd i j) k
                               ⊢ Eq (HAdd.hAdd j i) k
                             -/
    ιTensorObj Y X j i k (by simpa only [add_comm j i] using hij))
                             /-
                               🎉 no goals
                             -/
  inv k := tensorObjDesc (fun i j hij => (β_ _ _).inv ≫
                             /-
                               I : Type u_1
                               inst✝⁵ : AddCommMonoid I
                               C : Type u_2
                               inst✝⁴ : CategoryTheory.Category.{?u.147, u_2} C
                               inst✝³ : CategoryTheory.MonoidalCategory C
                               X Y Z : CategoryTheory.GradedObject I C
                               inst✝² : CategoryTheory.BraidedCategory C
                               inst✝¹ : X.HasTensor Y
                               inst✝ : Y.HasTensor X
                               k i j : I
                               hij : Eq (HAdd.hAdd i j) k
                               ⊢ Eq (HAdd.hAdd j i) k
                             -/
    ιTensorObj X Y j i k (by simpa only [add_comm j i] using hij))
                             /-
                               🎉 no goals
                             -/


variable {Y Z} in
lemma braiding_naturality_right [HasTensor X Y] [HasTensor Y X] [HasTensor X Z] [HasTensor Z X]
    (f : Y ⟶ Z) :
    whiskerLeft X f ≫ (braiding X Z).hom = (braiding X Y).hom ≫ whiskerRight f X  := by
  /-
    I : Type u_1
    inst✝⁷ : AddCommMonoid I
    C : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.GradedObject I C
    inst✝⁴ : CategoryTheory.BraidedCategory C
    inst✝³ : X.HasTensor Y
    inst✝² : Y.HasTensor X
    inst✝¹ : X.HasTensor Z
    inst✝ : Z.HasTensor X
    f : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  dsimp [braiding]
  /-
    I : Type u_1
    inst✝⁷ : AddCommMonoid I
    C : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.GradedObject I C
    inst✝⁴ : CategoryTheory.BraidedCategory C
    inst✝³ : X.HasTensor Y
    inst✝² : Y.HasTensor X
    inst✝¹ : X.HasTensor Z
    inst✝ : Z.HasTensor X
    f : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


variable {X Y} in
lemma braiding_naturality_left [HasTensor Y Z] [HasTensor Z Y] [HasTensor X Z] [HasTensor Z X]
    (f : X ⟶ Y) :
    whiskerRight f Z ≫ (braiding Y Z).hom = (braiding X Z).hom ≫ whiskerLeft Z f  := by
  /-
    I : Type u_1
    inst✝⁷ : AddCommMonoid I
    C : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.GradedObject I C
    inst✝⁴ : CategoryTheory.BraidedCategory C
    inst✝³ : Y.HasTensor Z
    inst✝² : Z.HasTensor Y
    inst✝¹ : X.HasTensor Z
    inst✝ : Z.HasTensor X
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  dsimp [braiding]
  /-
    I : Type u_1
    inst✝⁷ : AddCommMonoid I
    C : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} C
    inst✝⁵ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.GradedObject I C
    inst✝⁴ : CategoryTheory.BraidedCategory C
    inst✝³ : Y.HasTensor Z
    inst✝² : Z.HasTensor Y
    inst✝¹ : X.HasTensor Z
    inst✝ : Z.HasTensor X
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


lemma hexagon_forward [HasTensor X Y] [HasTensor Y X] [HasTensor Y Z]
    [HasTensor Z X] [HasTensor X Z]
    [HasTensor (tensorObj X Y) Z] [HasTensor X (tensorObj Y Z)]
    [HasTensor (tensorObj Y Z) X] [HasTensor Y (tensorObj Z X)]
    [HasTensor (tensorObj Y X) Z] [HasTensor Y (tensorObj X Z)]
    [HasGoodTensor₁₂Tensor X Y Z] [HasGoodTensorTensor₂₃ X Y Z]
    [HasGoodTensor₁₂Tensor Y Z X] [HasGoodTensorTensor₂₃ Y Z X]
    [HasGoodTensor₁₂Tensor Y X Z] [HasGoodTensorTensor₂₃ Y X Z] :
    (associator X Y Z).hom ≫ (braiding X (tensorObj Y Z)).hom ≫ (associator Y Z X).hom =
      whiskerRight (braiding X Y).hom Z ≫ (associator Y X Z).hom ≫
        whiskerLeft Y (braiding X Z).hom := by
  /-
    I : Type u_1
    inst✝²⁰ : AddCommMonoid I
    C : Type u_2
    inst✝¹⁹ : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹⁸ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.GradedObject I C
    inst✝¹⁷ : CategoryTheory.BraidedCategory C
    inst✝¹⁶ : X.HasTensor Y
    inst✝¹⁵ : Y.HasTensor X
    inst✝¹⁴ : Y.HasTensor Z
    inst✝¹³ : Z.HasTensor X
    inst✝¹² : X.HasTensor Z
    inst✝¹¹ : (CategoryTheory.GradedObject.Monoidal.tensorObj X Y).HasTensor Z
    inst✝¹⁰ : X.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Y Z)
    inst✝⁹ : (CategoryTheory.GradedObject.Monoidal.tensorObj Y Z).HasTensor X
    inst✝⁸ : Y.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Z X)
    inst✝⁷ : (CategoryTheory.GradedObject.Monoidal.tensorObj Y X).HasTensor Z
    inst✝⁶ : Y.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X Z)
    inst✝⁵ : X.HasGoodTensor₁₂Tensor Y Z
    inst✝⁴ : X.HasGoodTensorTensor₂₃ Y Z
    inst✝³ : Y.HasGoodTensor₁₂Tensor Z X
    inst✝² : Y.HasGoodTensorTensor₂₃ Z X
    inst✝¹ : Y.HasGoodTensor₁₂Tensor X Z
    inst✝ : Y.HasGoodTensorTensor₂₃ X Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  ext k i₁ i₂ i₃ h
  /-
    case h.h
    I : Type u_1
    inst✝²⁰ : AddCommMonoid I
    C : Type u_2
    inst✝¹⁹ : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹⁸ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.GradedObject I C
    inst✝¹⁷ : CategoryTheory.BraidedCategory C
    inst✝¹⁶ : X.HasTensor Y
    inst✝¹⁵ : Y.HasTensor X
    inst✝¹⁴ : Y.HasTensor Z
    inst✝¹³ : Z.HasTensor X
    inst✝¹² : X.HasTensor Z
    inst✝¹¹ : (CategoryTheory.GradedObject.Monoidal.tensorObj X Y).HasTensor Z
    inst✝¹⁰ : X.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Y Z)
    inst✝⁹ : (CategoryTheory.GradedObject.Monoidal.tensorObj Y Z).HasTensor X
    inst✝⁸ : Y.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Z X)
    inst✝⁷ : (CategoryTheory.GradedObject.Monoidal.tensorObj Y X).HasTensor Z
    inst✝⁶ : Y.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X Z)
    inst✝⁵ : X.HasGoodTensor₁₂Tensor Y Z
    inst✝⁴ : X.HasGoodTensorTensor₂₃ Y Z
    inst✝³ : Y.HasGoodTensor₁₂Tensor Z X
    inst✝² : Y.HasGoodTensorTensor₂₃ Z X
    inst✝¹ : Y.HasGoodTensor₁₂Tensor X Z
    inst✝ : Y.HasGoodTensorTensor₂₃ X Z
    k i₁ i₂ i₃ : I
    h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  dsimp [braiding]
  conv_lhs => rw [ιTensorObj₃'_associator_hom_assoc, ιTensorObj₃_eq X Y Z i₁ i₂ i₃ k h _ rfl,
    assoc, ι_tensorObjDesc_assoc, assoc, ← MonoidalCategory.id_tensorHom,
    BraidedCategory.braiding_naturality_assoc,
    BraidedCategory.braiding_tensor_right, assoc, assoc, assoc, assoc, Iso.hom_inv_id_assoc,
    MonoidalCategory.tensorHom_id,
    ← ιTensorObj₃'_eq_assoc Y Z X i₂ i₃ i₁ k (by rw [add_comm _ i₁, ← add_assoc, h]) _ rfl,
    ιTensorObj₃'_associator_hom, Iso.inv_hom_id_assoc]
  conv_rhs => rw [ιTensorObj₃'_eq X Y Z i₁ i₂ i₃ k h _ rfl, assoc, ι_tensorHom_assoc,
    ← MonoidalCategory.tensorHom_id,
    ← MonoidalCategory.tensor_comp_assoc, id_comp, ι_tensorObjDesc,
    categoryOfGradedObjects_id, MonoidalCategory.comp_tensor_id, assoc,
    MonoidalCategory.tensorHom_id, MonoidalCategory.tensorHom_id,
    ← ιTensorObj₃'_eq_assoc Y X Z i₂ i₁ i₃ k
      (by rw [add_comm i₂ i₁, h]) (i₁ + i₂) (add_comm i₂ i₁),
    ιTensorObj₃'_associator_hom_assoc,
    ιTensorObj₃_eq Y X Z i₂ i₁ i₃ k (by rw [add_comm i₂ i₁, h]) _ rfl, assoc,
    ι_tensorHom, categoryOfGradedObjects_id, ← MonoidalCategory.tensorHom_id,
    ← MonoidalCategory.id_tensorHom,
    ← MonoidalCategory.id_tensor_comp_assoc,
    ι_tensorObjDesc, MonoidalCategory.id_tensor_comp, assoc,
    ← MonoidalCategory.id_tensor_comp_assoc, MonoidalCategory.tensorHom_id,
    MonoidalCategory.id_tensorHom, MonoidalCategory.whiskerLeft_comp, assoc,
    ← ιTensorObj₃_eq Y Z X i₂ i₃ i₁ k (by rw [add_comm _ i₁, ← add_assoc, h])
      (i₁ + i₃) (add_comm _ _ )]


lemma hexagon_reverse [HasTensor X Y] [HasTensor Y Z] [HasTensor Z X]
    [HasTensor Z Y] [HasTensor X Z]
    [HasTensor (tensorObj X Y) Z] [HasTensor X (tensorObj Y Z)]
    [HasTensor Z (tensorObj X Y)] [HasTensor (tensorObj Z X) Y]
    [HasTensor X (tensorObj Z Y)] [HasTensor (tensorObj X Z) Y]
    [HasGoodTensor₁₂Tensor X Y Z] [HasGoodTensorTensor₂₃ X Y Z]
    [HasGoodTensor₁₂Tensor Z X Y] [HasGoodTensorTensor₂₃ Z X Y]
    [HasGoodTensor₁₂Tensor X Z Y] [HasGoodTensorTensor₂₃ X Z Y]:
    (associator X Y Z).inv ≫ (braiding (tensorObj X Y) Z).hom ≫ (associator Z X Y).inv =
      whiskerLeft X (braiding Y Z).hom ≫ (associator X Z Y).inv ≫
        whiskerRight (braiding X Z).hom Y := by
  /-
    I : Type u_1
    inst✝²⁰ : AddCommMonoid I
    C : Type u_2
    inst✝¹⁹ : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹⁸ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.GradedObject I C
    inst✝¹⁷ : CategoryTheory.BraidedCategory C
    inst✝¹⁶ : X.HasTensor Y
    inst✝¹⁵ : Y.HasTensor Z
    inst✝¹⁴ : Z.HasTensor X
    inst✝¹³ : Z.HasTensor Y
    inst✝¹² : X.HasTensor Z
    inst✝¹¹ : (CategoryTheory.GradedObject.Monoidal.tensorObj X Y).HasTensor Z
    inst✝¹⁰ : X.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Y Z)
    inst✝⁹ : Z.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X Y)
    inst✝⁸ : (CategoryTheory.GradedObject.Monoidal.tensorObj Z X).HasTensor Y
    inst✝⁷ : X.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Z Y)
    inst✝⁶ : (CategoryTheory.GradedObject.Monoidal.tensorObj X Z).HasTensor Y
    inst✝⁵ : X.HasGoodTensor₁₂Tensor Y Z
    inst✝⁴ : X.HasGoodTensorTensor₂₃ Y Z
    inst✝³ : Z.HasGoodTensor₁₂Tensor X Y
    inst✝² : Z.HasGoodTensorTensor₂₃ X Y
    inst✝¹ : X.HasGoodTensor₁₂Tensor Z Y
    inst✝ : X.HasGoodTensorTensor₂₃ Z Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  ext k i₁ i₂ i₃ h
  /-
    case h.h
    I : Type u_1
    inst✝²⁰ : AddCommMonoid I
    C : Type u_2
    inst✝¹⁹ : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹⁸ : CategoryTheory.MonoidalCategory C
    X Y Z : CategoryTheory.GradedObject I C
    inst✝¹⁷ : CategoryTheory.BraidedCategory C
    inst✝¹⁶ : X.HasTensor Y
    inst✝¹⁵ : Y.HasTensor Z
    inst✝¹⁴ : Z.HasTensor X
    inst✝¹³ : Z.HasTensor Y
    inst✝¹² : X.HasTensor Z
    inst✝¹¹ : (CategoryTheory.GradedObject.Monoidal.tensorObj X Y).HasTensor Z
    inst✝¹⁰ : X.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Y Z)
    inst✝⁹ : Z.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X Y)
    inst✝⁸ : (CategoryTheory.GradedObject.Monoidal.tensorObj Z X).HasTensor Y
    inst✝⁷ : X.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Z Y)
    inst✝⁶ : (CategoryTheory.GradedObject.Monoidal.tensorObj X Z).HasTensor Y
    inst✝⁵ : X.HasGoodTensor₁₂Tensor Y Z
    inst✝⁴ : X.HasGoodTensorTensor₂₃ Y Z
    inst✝³ : Z.HasGoodTensor₁₂Tensor X Y
    inst✝² : Z.HasGoodTensorTensor₂₃ X Y
    inst✝¹ : X.HasGoodTensor₁₂Tensor Z Y
    inst✝ : X.HasGoodTensorTensor₂₃ Z Y
    k i₁ i₂ i₃ : I
    h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  dsimp [braiding]
  conv_lhs => rw [ιTensorObj₃_associator_inv_assoc, ιTensorObj₃'_eq X Y Z i₁ i₂ i₃ k h _ rfl, assoc,
    ι_tensorObjDesc_assoc, assoc, ← MonoidalCategory.tensorHom_id,
    BraidedCategory.braiding_naturality_assoc,
    BraidedCategory.braiding_tensor_left, assoc, assoc, assoc, assoc, Iso.inv_hom_id_assoc,
    MonoidalCategory.id_tensorHom,
    ← ιTensorObj₃_eq_assoc Z X Y i₃ i₁ i₂ k (by rw [add_assoc, add_comm i₃, h]) _ rfl,
    ιTensorObj₃_associator_inv, Iso.hom_inv_id_assoc]
  conv_rhs => rw [ιTensorObj₃_eq X Y Z i₁ i₂ i₃ k h _ rfl, assoc, ι_tensorHom_assoc,
    ← MonoidalCategory.id_tensorHom,
    ← MonoidalCategory.tensor_comp_assoc, id_comp, ι_tensorObjDesc,
    categoryOfGradedObjects_id, MonoidalCategory.id_tensor_comp, assoc,
    MonoidalCategory.id_tensorHom, MonoidalCategory.id_tensorHom,
    ← ιTensorObj₃_eq_assoc X Z Y i₁ i₃ i₂ k
      (by rw [add_assoc, add_comm i₃, ← add_assoc, h]) (i₂ + i₃) (add_comm _ _),
    ιTensorObj₃_associator_inv_assoc,
    ιTensorObj₃'_eq X Z Y i₁ i₃ i₂ k (by rw [add_assoc, add_comm i₃, ← add_assoc, h]) _ rfl,
    assoc, ι_tensorHom, categoryOfGradedObjects_id, ← MonoidalCategory.tensorHom_id,
    ← MonoidalCategory.comp_tensor_id_assoc,
    ι_tensorObjDesc, MonoidalCategory.comp_tensor_id, assoc,
    MonoidalCategory.tensorHom_id, MonoidalCategory.tensorHom_id,
    ← ιTensorObj₃'_eq Z X Y i₃ i₁ i₂ k (by rw [add_assoc, add_comm i₃, h])
      (i₁ + i₃) (add_comm _ _)]


@[reassoc (attr := simp)]
lemma symmetry [SymmetricCategory C] [HasTensor X Y] [HasTensor Y X] :
    (braiding X Y).hom ≫ (braiding Y X).hom = 𝟙 _ := by
  /-
    I : Type u_1
    inst✝⁵ : AddCommMonoid I
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y : CategoryTheory.GradedObject I C
    inst✝² : CategoryTheory.SymmetricCategory C
    inst✝¹ : X.HasTensor Y
    inst✝ : Y.HasTensor X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  dsimp [braiding]
  /-
    I : Type u_1
    inst✝⁵ : AddCommMonoid I
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X Y : CategoryTheory.GradedObject I C
    inst✝² : CategoryTheory.SymmetricCategory C
    inst✝¹ : X.HasTensor Y
    inst✝ : Y.HasTensor X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun k => CategoryTheory.GradedObject …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


noncomputable instance braidedCategory [BraidedCategory C] :
    BraidedCategory (GradedObject I C) where
  braiding X Y := Monoidal.braiding X Y
  braiding_naturality_left _ _:= Monoidal.braiding_naturality_left _ _
  braiding_naturality_right _ _ _ _  := Monoidal.braiding_naturality_right _ _
  hexagon_forward _ _ _ := Monoidal.hexagon_forward _ _ _
  hexagon_reverse _ _ _ := Monoidal.hexagon_reverse _ _ _


noncomputable instance symmetricCategory [SymmetricCategory C] :
    SymmetricCategory (GradedObject I C) where
  symmetry _ _ := Monoidal.symmetry _ _


