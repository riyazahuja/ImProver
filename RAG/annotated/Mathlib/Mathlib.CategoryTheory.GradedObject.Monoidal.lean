/-- The tensor product of two graded objects `X₁` and `X₂` exists if for any `n`,
the coproduct of the objects `X₁ i ⊗ X₂ j` for `i + j = n` exists. -/
abbrev HasTensor (X₁ X₂ : GradedObject I C) : Prop :=
  HasMap (((mapBifunctor (curriedTensor C) I I).obj X₁).obj X₂) (fun ⟨i, j⟩ => i + j)


lemma hasTensor_of_iso {X₁ X₂ Y₁ Y₂ : GradedObject I C}
    (e₁ : X₁ ≅ Y₁) (e₂ : X₂ ≅ Y₂) [HasTensor X₁ X₂] :
    HasTensor Y₁ Y₂ := by
  let e : ((mapBifunctor (curriedTensor C) I I).obj X₁).obj X₂ ≅
    ((mapBifunctor (curriedTensor C) I I).obj Y₁).obj Y₂ := isoMk _ _
      (fun ⟨i, j⟩ ↦ (eval i).mapIso e₁ ⊗ (eval j).mapIso e₂)
  /-
    I : Type u
    inst✝³ : AddMonoid I
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X₁ X₂ Y₁ Y₂ : CategoryTheory.GradedObject I C
    e₁ : CategoryTheory.Iso X₁ Y₁
    e₂ : CategoryTheory.Iso X₂ Y₂
    inst✝ : X₁.HasTensor X₂
    e : CategoryTheory.Iso (((CategoryTheory.GradedObject.mapBifunctor (CategoryTh …
    ⊢ Y₁.HasTensor Y₂
  -/
  exact hasMap_of_iso e _
  /-
    🎉 no goals
  -/


/-- The tensor product of two graded objects. -/
noncomputable abbrev tensorObj (X₁ X₂ : GradedObject I C) [HasTensor X₁ X₂] :
    GradedObject I C :=
  mapBifunctorMapObj (curriedTensor C) (fun ⟨i, j⟩ => i + j) X₁ X₂


/-- The inclusion of a summand in a tensor product of two graded objects. -/
noncomputable def ιTensorObj (i₁ i₂ i₁₂ : I) (h : i₁ + i₂ = i₁₂) :
  X₁ i₁ ⊗ X₂ i₂ ⟶ tensorObj X₁ X₂ i₁₂ :=
    ιMapBifunctorMapObj (curriedTensor C) _ _ _ _ _ _ h


@[ext]
lemma tensorObj_ext {A : C} {j : I} (f g : tensorObj X₁ X₂ j ⟶ A)
    (h : ∀ (i₁ i₂ : I) (hi : i₁ + i₂ = j),
      ιTensorObj X₁ X₂ i₁ i₂ j hi ≫ f = ιTensorObj X₁ X₂ i₁ i₂ j hi ≫ g) : f = g := by
  /-
    I : Type u
    inst✝³ : AddMonoid I
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X₁ X₂ : CategoryTheory.GradedObject I C
    inst✝ : X₁.HasTensor X₂
    A : C
    j : I
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂ j) A
    h : ∀ (i₁ i₂ : I) (hi : Eq (HAdd.hAdd i₁ i₂) j), Eq (CategoryTheory.CategorySt …
    ⊢ Eq f g
  -/
  apply mapObj_ext
  /-
    case hfg
    I : Type u
    inst✝³ : AddMonoid I
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X₁ X₂ : CategoryTheory.GradedObject I C
    inst✝ : X₁.HasTensor X₂
    A : C
    j : I
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂ j) A
    h : ∀ (i₁ i₂ : I) (hi : Eq (HAdd.hAdd i₁ i₂) j), Eq (CategoryTheory.CategorySt …
    ⊢ ∀ (i : Prod I I) (hij : Eq (CategoryTheory.GradedObject.HasTensor.match_1 (f …
  -/
  rintro ⟨i₁, i₂⟩ hi
  /-
    case hfg.mk
    I : Type u
    inst✝³ : AddMonoid I
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X₁ X₂ : CategoryTheory.GradedObject I C
    inst✝ : X₁.HasTensor X₂
    A : C
    j : I
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂ j) A
    h : ∀ (i₁ i₂ : I) (hi : Eq (HAdd.hAdd i₁ i₂) j), Eq (CategoryTheory.CategorySt …
    i₁ i₂ : I
    hi : Eq (CategoryTheory.GradedObject.HasTensor.match_1 (fun x => I) { fst := i …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.GradedObject.mapBi …
  -/
  exact h i₁ i₂ hi
  /-
    🎉 no goals
  -/


/-- Constructor for morphisms from a tensor product of two graded objects. -/
noncomputable def tensorObjDesc {A : C} {k : I}
    (f : ∀ (i₁ i₂ : I) (_ : i₁ + i₂ = k), X₁ i₁ ⊗ X₂ i₂ ⟶ A) : tensorObj X₁ X₂ k ⟶ A :=
  mapBifunctorMapObjDesc f


@[reassoc (attr := simp)]
lemma ι_tensorObjDesc {A : C} {k : I}
    (f : ∀ (i₁ i₂ : I) (_ : i₁ + i₂ = k), X₁ i₁ ⊗ X₂ i₂ ⟶ A) (i₁ i₂ : I) (hi : i₁ + i₂ = k) :
    ιTensorObj X₁ X₂ i₁ i₂ k hi ≫ tensorObjDesc f = f i₁ i₂ hi := by
  /-
    I : Type u
    inst✝³ : AddMonoid I
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X₁ X₂ : CategoryTheory.GradedObject I C
    inst✝ : X₁.HasTensor X₂
    A : C
    k : I
    f : (i₁ i₂ : I) → Eq (HAdd.hAdd i₁ i₂) k → Quiver.Hom (CategoryTheory.Monoidal …
    i₁ i₂ : I
    hi : Eq (HAdd.hAdd i₁ i₂) k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  apply ι_mapBifunctorMapObjDesc
  /-
    🎉 no goals
  -/


/-- The morphism `tensorObj X₁ Y₁ ⟶ tensorObj X₂ Y₂` induced by morphisms of graded
objects `f : X₁ ⟶ X₂` and `g : Y₁ ⟶ Y₂`. -/
noncomputable def tensorHom {X₁ X₂ Y₁ Y₂ : GradedObject I C} (f : X₁ ⟶ X₂) (g : Y₁ ⟶ Y₂)
    [HasTensor X₁ Y₁] [HasTensor X₂ Y₂] :
    tensorObj X₁ Y₁ ⟶ tensorObj X₂ Y₂ :=
  mapBifunctorMapMap _ _ f g


@[reassoc (attr := simp)]
lemma ι_tensorHom {X₁ X₂ Y₁ Y₂ : GradedObject I C} (f : X₁ ⟶ X₂) (g : Y₁ ⟶ Y₂)
    [HasTensor X₁ Y₁] [HasTensor X₂ Y₂] (i₁ i₂ i₁₂ : I) (h : i₁ + i₂ = i₁₂) :
    ιTensorObj X₁ Y₁ i₁ i₂ i₁₂ h ≫ tensorHom f g i₁₂ =
      (f i₁ ⊗ g i₂) ≫ ιTensorObj X₂ Y₂ i₁ i₂ i₁₂ h := by
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ Y₁ Y₂ : CategoryTheory.GradedObject I C
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom Y₁ Y₂
    inst✝¹ : X₁.HasTensor Y₁
    inst✝ : X₂.HasTensor Y₂
    i₁ i₂ i₁₂ : I
    h : Eq (HAdd.hAdd i₁ i₂) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  rw [MonoidalCategory.tensorHom_def, assoc]
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ Y₁ Y₂ : CategoryTheory.GradedObject I C
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom Y₁ Y₂
    inst✝¹ : X₁.HasTensor Y₁
    inst✝ : X₂.HasTensor Y₂
    i₁ i₂ i₁₂ : I
    h : Eq (HAdd.hAdd i₁ i₂) i₁₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  apply ι_mapBifunctorMapMap
  /-
    🎉 no goals
  -/


/-- The morphism `tensorObj X Y₁ ⟶ tensorObj X Y₂` induced by a morphism of graded objects
`φ : Y₁ ⟶ Y₂`. -/
noncomputable abbrev whiskerLeft (X : GradedObject I C) {Y₁ Y₂ : GradedObject I C} (φ : Y₁ ⟶ Y₂)
    [HasTensor X Y₁] [HasTensor X Y₂] : tensorObj X Y₁ ⟶ tensorObj X Y₂ :=
  tensorHom (𝟙 X) φ


/-- The morphism `tensorObj X₁ Y ⟶ tensorObj X₂ Y` induced by a morphism of graded objects
`φ : X₁ ⟶ X₂`. -/
noncomputable abbrev whiskerRight {X₁ X₂ : GradedObject I C} (φ : X₁ ⟶ X₂) (Y : GradedObject I C)
    [HasTensor X₁ Y] [HasTensor X₂ Y] : tensorObj X₁ Y ⟶ tensorObj X₂ Y :=
  tensorHom φ (𝟙 Y)


@[simp]
lemma tensor_id (X Y : GradedObject I C) [HasTensor X Y] :
    tensorHom (𝟙 X) (𝟙 Y) = 𝟙 _ := by
  /-
    I : Type u
    inst✝³ : AddMonoid I
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : CategoryTheory.GradedObject I C
    inst✝ : X.HasTensor Y
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.tensorHom (CategoryTheory.CategoryS …
  -/
  dsimp [tensorHom, mapBifunctorMapMap]
  /-
    I : Type u
    inst✝³ : AddMonoid I
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : CategoryTheory.GradedObject I C
    inst✝ : X.HasTensor Y
    ⊢ Eq (CategoryTheory.GradedObject.mapMap (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Functor.map_id, NatTrans.id_app, comp_id, mapMap_id]
  /-
    I : Type u
    inst✝³ : AddMonoid I
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    X Y : CategoryTheory.GradedObject I C
    inst✝ : X.HasTensor Y
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((((CategoryTheory.GradedObject.mapBifu …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma tensor_comp {X₁ X₂ X₃ Y₁ Y₂ Y₃ : GradedObject I C} (f₁ : X₁ ⟶ X₂) (f₂ : X₂ ⟶ X₃)
    (g₁ : Y₁ ⟶ Y₂) (g₂ : Y₂ ⟶ Y₃) [HasTensor X₁ Y₁] [HasTensor X₂ Y₂] [HasTensor X₃ Y₃] :
    tensorHom (f₁ ≫ f₂) (g₁ ≫ g₂) = tensorHom f₁ g₁ ≫ tensorHom f₂ g₂ := by
  /-
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.GradedObject I C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    inst✝² : X₁.HasTensor Y₁
    inst✝¹ : X₂.HasTensor Y₂
    inst✝ : X₃.HasTensor Y₃
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.tensorHom (CategoryTheory.CategoryS …
  -/
  dsimp only [tensorHom, mapBifunctorMapMap]
  /-
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.GradedObject I C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    inst✝² : X₁.HasTensor Y₁
    inst✝¹ : X₂.HasTensor Y₂
    inst✝ : X₃.HasTensor Y₃
    ⊢ Eq (CategoryTheory.GradedObject.mapMap (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [← mapMap_comp]
  /-
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.GradedObject I C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    inst✝² : X₁.HasTensor Y₁
    inst✝¹ : X₂.HasTensor Y₂
    inst✝ : X₃.HasTensor Y₃
    ⊢ Eq (CategoryTheory.GradedObject.mapMap (CategoryTheory.CategoryStruct.comp ( …
  -/
  apply congr_mapMap
  /-
    case h
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.GradedObject I C
    f₁ : Quiver.Hom X₁ X₂
    f₂ : Quiver.Hom X₂ X₃
    g₁ : Quiver.Hom Y₁ Y₂
    g₂ : Quiver.Hom Y₂ Y₃
    inst✝² : X₁.HasTensor Y₁
    inst✝¹ : X₂.HasTensor Y₂
    inst✝ : X₃.HasTensor Y₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.GradedObject.mapBif …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The isomorphism `tensorObj X₁ Y₁ ≅ tensorObj X₂ Y₂` induced by isomorphisms of graded
objects `e : X₁ ≅ X₂` and `e' : Y₁ ≅ Y₂`. -/
@[simps]
noncomputable def tensorIso {X₁ X₂ Y₁ Y₂ : GradedObject I C} (e : X₁ ≅ X₂) (e' : Y₁ ≅ Y₂)
    [HasTensor X₁ Y₁] [HasTensor X₂ Y₂] :
    tensorObj X₁ Y₁ ≅ tensorObj X₂ Y₂ where
  hom := tensorHom e.hom e'.hom
  inv := tensorHom e.inv e'.inv
                   /-
                     I : Type u
                     inst✝⁴ : AddMonoid I
                     C : Type u_1
                     inst✝³ : CategoryTheory.Category.{?u.42851, u_1} C
                     inst✝² : CategoryTheory.MonoidalCategory C
                     X₁ X₂ Y₁ Y₂ : CategoryTheory.GradedObject I C
                     e : CategoryTheory.Iso X₁ X₂
                     e' : CategoryTheory.Iso Y₁ Y₂
                     inst✝¹ : X₁.HasTensor Y₁
                     inst✝ : X₂.HasTensor Y₂
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
                   -/
  hom_inv_id := by simp only [← tensor_comp, Iso.hom_inv_id, tensor_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     I : Type u
                     inst✝⁴ : AddMonoid I
                     C : Type u_1
                     inst✝³ : CategoryTheory.Category.{?u.42851, u_1} C
                     inst✝² : CategoryTheory.MonoidalCategory C
                     X₁ X₂ Y₁ Y₂ : CategoryTheory.GradedObject I C
                     e : CategoryTheory.Iso X₁ X₂
                     e' : CategoryTheory.Iso Y₁ Y₂
                     inst✝¹ : X₁.HasTensor Y₁
                     inst✝ : X₂.HasTensor Y₂
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
                   -/
  inv_hom_id := by simp only [← tensor_comp, Iso.inv_hom_id, tensor_id]
                   /-
                     🎉 no goals
                   -/


lemma tensorHom_def {X₁ X₂ Y₁ Y₂ : GradedObject I C} (f : X₁ ⟶ X₂) (g : Y₁ ⟶ Y₂)
    [HasTensor X₁ Y₁] [HasTensor X₂ Y₂] [HasTensor X₂ Y₁] :
    tensorHom f g = whiskerRight f Y₁ ≫ whiskerLeft X₂ g := by
  /-
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ Y₁ Y₂ : CategoryTheory.GradedObject I C
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom Y₁ Y₂
    inst✝² : X₁.HasTensor Y₁
    inst✝¹ : X₂.HasTensor Y₂
    inst✝ : X₂.HasTensor Y₁
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.tensorHom f g) (CategoryTheory.Cate …
  -/
  rw [← tensor_comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


/-- This is the addition map `I × I × I → I` for an additive monoid `I`. -/
def r₁₂₃ : I × I × I → I := fun ⟨i, j, k⟩ => i + j + k


/-- Auxiliary definition for `associator`. -/
@[reducible] def ρ₁₂ : BifunctorComp₁₂IndexData (r₁₂₃ : _ → I) where
  I₁₂ := I
  p := fun ⟨i₁, i₂⟩ => i₁ + i₂
  q := fun ⟨i₁₂, i₃⟩ => i₁₂ + i₃
  hpq := fun _ => rfl


/-- Auxiliary definition for `associator`. -/
@[reducible] def ρ₂₃ : BifunctorComp₂₃IndexData (r₁₂₃ : _ → I) where
  I₂₃ := I
  p := fun ⟨i₂, i₃⟩ => i₂ + i₃
  q := fun ⟨i₁₂, i₃⟩ => i₁₂ + i₃
  hpq _ := (add_assoc _ _ _).symm


variable (I) in
/-- Auxiliary definition for `associator`. -/
@[reducible]
def triangleIndexData : TriangleIndexData (r₁₂₃ : _ → I) (fun ⟨i₁, i₃⟩ => i₁ + i₃) where
  p₁₂ := fun ⟨i₁, i₂⟩ => i₁ + i₂
  p₂₃ := fun ⟨i₂, i₃⟩ => i₂ + i₃
  hp₁₂ := fun _ => rfl
  hp₂₃ := fun _ => (add_assoc _ _ _).symm
  h₁ := add_zero
  h₃ := zero_add


/-- Given three graded objects `X₁`, `X₂`, `X₃` in `GradedObject I C`, this is the
assumption that for all `i₁₂ : I` and `i₃ : I`, the tensor product functor `- ⊗ X₃ i₃`
commutes with the coproduct of the objects `X₁ i₁ ⊗ X₂ i₂` such that `i₁ + i₂ = i₁₂`. -/
abbrev _root_.CategoryTheory.GradedObject.HasGoodTensor₁₂Tensor (X₁ X₂ X₃ : GradedObject I C) :=
  HasGoodTrifunctor₁₂Obj (curriedTensor C) (curriedTensor C) ρ₁₂ X₁ X₂ X₃


/-- Given three graded objects `X₁`, `X₂`, `X₃` in `GradedObject I C`, this is the
assumption that for all `i₁ : I` and `i₂₃ : I`, the tensor product functor `X₁ i₁ ⊗ -`
commutes with the coproduct of the objects `X₂ i₂ ⊗ X₃ i₃` such that `i₂ + i₃ = i₂₃`. -/
abbrev _root_.CategoryTheory.GradedObject.HasGoodTensorTensor₂₃ (X₁ X₂ X₃ : GradedObject I C) :=
  HasGoodTrifunctor₂₃Obj (curriedTensor C) (curriedTensor C) ρ₂₃ X₁ X₂ X₃


/-- The inclusion `X₁ i₁ ⊗ X₂ i₂ ⊗ X₃ i₃ ⟶ tensorObj X₁ (tensorObj X₂ X₃) j`
when `i₁ + i₂ + i₃ = j`. -/
noncomputable def ιTensorObj₃ (i₁ i₂ i₃ j : I) (h : i₁ + i₂ + i₃ = j) :
    X₁ i₁ ⊗ X₂ i₂ ⊗ X₃ i₃ ⟶ tensorObj X₁ (tensorObj X₂ X₃) j :=
  X₁ i₁ ◁ ιTensorObj X₂ X₃ i₂ i₃ _ rfl ≫ ιTensorObj X₁ (tensorObj X₂ X₃) i₁ (i₂ + i₃) j
        /-
          I : Type u
          inst✝⁶ : AddMonoid I
          C : Type u_1
          inst✝⁵ : CategoryTheory.Category.{?u.52785, u_1} C
          inst✝⁴ : CategoryTheory.MonoidalCategory C
          Z : C
          X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.GradedObject I C
          inst✝³ : X₂.HasTensor X₃
          inst✝² : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
          inst✝¹ : Y₂.HasTensor Y₃
          inst✝ : Y₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Y₂ Y₃)
          i₁ i₂ i₃ j : I
          h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j
          ⊢ Eq (HAdd.hAdd i₁ (HAdd.hAdd i₂ i₃)) j
        -/
    (by rw [← add_assoc, h])
        /-
          🎉 no goals
        -/


@[reassoc]
lemma ιTensorObj₃_eq (i₁ i₂ i₃ j : I) (h : i₁ + i₂ + i₃ = j) (i₂₃ : I) (h' : i₂ + i₃ = i₂₃) :
    ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j h =
      (X₁ i₁ ◁ ιTensorObj X₂ X₃ i₂ i₃ i₂₃ h') ≫
                                                     /-
                                                       I : Type u
                                                       inst✝⁶ : AddMonoid I
                                                       C : Type u_1
                                                       inst✝⁵ : CategoryTheory.Category.{?u.68386, u_1} C
                                                       inst✝⁴ : CategoryTheory.MonoidalCategory C
                                                       Z : C
                                                       X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.GradedObject I C
                                                       inst✝³ : X₂.HasTensor X₃
                                                       inst✝² : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
                                                       inst✝¹ : Y₂.HasTensor Y₃
                                                       inst✝ : Y₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Y₂ Y₃)
                                                       i₁ i₂ i₃ j : I
                                                       h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j
                                                       i₂₃ : I
                                                       h' : Eq (HAdd.hAdd i₂ i₃) i₂₃
                                                       ⊢ Eq (HAdd.hAdd i₁ i₂₃) j
                                                     -/
        ιTensorObj X₁ (tensorObj X₂ X₃) i₁ i₂₃ j (by rw [← h', ← add_assoc, h]) := by
                                                     /-
                                                       🎉 no goals
                                                     -/
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₂.HasTensor X₃
    inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
    i₁ i₂ i₃ j : I
    h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j
    i₂₃ : I
    h' : Eq (HAdd.hAdd i₂ i₃) i₂₃
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j h)  …
  -/
  subst h'
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₂.HasTensor X₃
    inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
    i₁ i₂ i₃ j : I
    h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j h)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιTensorObj₃_tensorHom (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (f₃ : X₃ ⟶ Y₃)
    (i₁ i₂ i₃ j : I) (h : i₁ + i₂ + i₃ = j) :
    ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ tensorHom f₁ (tensorHom f₂ f₃) j =
      (f₁ i₁ ⊗ f₂ i₂ ⊗ f₃ i₃) ≫ ιTensorObj₃ Y₁ Y₂ Y₃ i₁ i₂ i₃ j h := by
  rw [ιTensorObj₃_eq _ _ _ i₁ i₂ i₃ j h _  rfl,
    ιTensorObj₃_eq _ _ _ i₁ i₂ i₃ j h _  rfl, assoc, ι_tensorHom,
    ← id_tensorHom, ← id_tensorHom, ← MonoidalCategory.tensor_comp_assoc, ι_tensorHom,
    ← MonoidalCategory.tensor_comp_assoc, id_comp, comp_id]


@[ext (iff := false)]
lemma tensorObj₃_ext {j : I} {A : C} (f g : tensorObj X₁ (tensorObj X₂ X₃) j ⟶ A)
    [H : HasGoodTensorTensor₂₃ X₁ X₂ X₃]
    (h : ∀ (i₁ i₂ i₃ : I) (hi : i₁ + i₂ + i₃ = j),
      ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j hi ≫ f = ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j hi ≫ g) :
      f = g := by
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₂.HasTensor X₃
    inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    H : X₁.HasGoodTensorTensor₂₃ X₂ X₃
    h : ∀ (i₁ i₂ i₃ : I) (hi : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j), Eq (Categor …
    ⊢ Eq f g
  -/
  apply mapBifunctorBifunctor₂₃MapObj_ext (H := H)
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₂.HasTensor X₃
    inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    H : X₁.HasGoodTensorTensor₂₃ X₂ X₃
    h : ∀ (i₁ i₂ i₃ : I) (hi : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j), Eq (Categor …
    ⊢ ∀ (i₁ i₂ i₃ : I) (h : Eq (CategoryTheory.GradedObject.Monoidal.r₁₂₃ { fst := …
  -/
  intro i₁ i₂ i₃ hi
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₂.HasTensor X₃
    inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    H : X₁.HasGoodTensorTensor₂₃ X₂ X₃
    h : ∀ (i₁ i₂ i₃ : I) (hi : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j), Eq (Categor …
    i₁ i₂ i₃ : I
    hi : Eq (CategoryTheory.GradedObject.Monoidal.r₁₂₃ { fst := i₁, snd := { fst : …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  exact h i₁ i₂ i₃ hi
  /-
    🎉 no goals
  -/


/-- The inclusion `X₁ i₁ ⊗ X₂ i₂ ⊗ X₃ i₃ ⟶ tensorObj (tensorObj X₁ X₂) X₃ j`
when `i₁ + i₂ + i₃ = j`. -/
noncomputable def ιTensorObj₃' (i₁ i₂ i₃ j : I) (h : i₁ + i₂ + i₃ = j) :
    (X₁ i₁ ⊗ X₂ i₂) ⊗ X₃ i₃ ⟶ tensorObj (tensorObj X₁ X₂) X₃ j :=
  (ιTensorObj X₁ X₂ i₁ i₂ (i₁ + i₂) rfl ▷ X₃ i₃) ≫
    ιTensorObj (tensorObj X₁ X₂) X₃ (i₁ + i₂) i₃ j h


@[reassoc]
lemma ιTensorObj₃'_eq (i₁ i₂ i₃ j : I) (h : i₁ + i₂ + i₃ = j) (i₁₂ : I)
    (h' : i₁ + i₂ = i₁₂) :
    ιTensorObj₃' X₁ X₂ X₃ i₁ i₂ i₃ j h =
      (ιTensorObj X₁ X₂ i₁ i₂ i₁₂ h' ▷ X₃ i₃) ≫
                                                     /-
                                                       I : Type u
                                                       inst✝⁶ : AddMonoid I
                                                       C : Type u_1
                                                       inst✝⁵ : CategoryTheory.Category.{?u.104632, u_1} C
                                                       inst✝⁴ : CategoryTheory.MonoidalCategory C
                                                       Z : C
                                                       X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.GradedObject I C
                                                       inst✝³ : X₁.HasTensor X₂
                                                       inst✝² : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
                                                       inst✝¹ : Y₁.HasTensor Y₂
                                                       inst✝ : (CategoryTheory.GradedObject.Monoidal.tensorObj Y₁ Y₂).HasTensor Y₃
                                                       i₁ i₂ i₃ j : I
                                                       h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j
                                                       i₁₂ : I
                                                       h' : Eq (HAdd.hAdd i₁ i₂) i₁₂
                                                       ⊢ Eq (HAdd.hAdd i₁₂ i₃) j
                                                     -/
        ιTensorObj (tensorObj X₁ X₂) X₃ i₁₂ i₃ j (by rw [← h', h]) := by
                                                     /-
                                                       🎉 no goals
                                                     -/
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₁.HasTensor X₂
    inst✝ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
    i₁ i₂ i₃ j : I
    h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j
    i₁₂ : I
    h' : Eq (HAdd.hAdd i₁ i₂) i₁₂
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.ιTensorObj₃' X₁ X₂ X₃ i₁ i₂ i₃ j h) …
  -/
  subst h'
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₁.HasTensor X₂
    inst✝ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
    i₁ i₂ i₃ j : I
    h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.ιTensorObj₃' X₁ X₂ X₃ i₁ i₂ i₃ j h) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιTensorObj₃'_tensorHom (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (f₃ : X₃ ⟶ Y₃)
    (i₁ i₂ i₃ j : I) (h : i₁ + i₂ + i₃ = j) :
    ιTensorObj₃' X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ tensorHom (tensorHom f₁ f₂) f₃ j =
      ((f₁ i₁ ⊗ f₂ i₂) ⊗ f₃ i₃) ≫ ιTensorObj₃' Y₁ Y₂ Y₃ i₁ i₂ i₃ j h := by
  rw [ιTensorObj₃'_eq _ _ _ i₁ i₂ i₃ j h _  rfl,
    ιTensorObj₃'_eq _ _ _ i₁ i₂ i₃ j h _  rfl, assoc, ι_tensorHom,
    ← tensorHom_id, ← tensorHom_id, ← MonoidalCategory.tensor_comp_assoc, id_comp,
    ι_tensorHom, ← MonoidalCategory.tensor_comp_assoc, comp_id]


@[ext (iff := false)]
lemma tensorObj₃'_ext {j : I} {A : C} (f g : tensorObj (tensorObj X₁ X₂) X₃ j ⟶ A)
    [H : HasGoodTensor₁₂Tensor X₁ X₂ X₃]
    (h : ∀ (i₁ i₂ i₃ : I) (h : i₁ + i₂ + i₃ = j),
      ιTensorObj₃' X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ f = ιTensorObj₃' X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ g) :
      f = g := by
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₁.HasTensor X₂
    inst✝ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj (CategoryTheo …
    H : X₁.HasGoodTensor₁₂Tensor X₂ X₃
    h : ∀ (i₁ i₂ i₃ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j), Eq (Category …
    ⊢ Eq f g
  -/
  apply mapBifunctor₁₂BifunctorMapObj_ext (H := H)
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₁.HasTensor X₂
    inst✝ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj (CategoryTheo …
    H : X₁.HasGoodTensor₁₂Tensor X₂ X₃
    h : ∀ (i₁ i₂ i₃ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j), Eq (Category …
    ⊢ ∀ (i₁ i₂ i₃ : I) (h : Eq (CategoryTheory.GradedObject.Monoidal.r₁₂₃ { fst := …
  -/
  intro i₁ i₂ i₃ hi
  /-
    I : Type u
    inst✝⁴ : AddMonoid I
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ : CategoryTheory.GradedObject I C
    inst✝¹ : X₁.HasTensor X₂
    inst✝ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj (CategoryTheo …
    H : X₁.HasGoodTensor₁₂Tensor X₂ X₃
    h : ∀ (i₁ i₂ i₃ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j), Eq (Category …
    i₁ i₂ i₃ : I
    hi : Eq (CategoryTheory.GradedObject.Monoidal.r₁₂₃ { fst := i₁, snd := { fst : …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  exact h i₁ i₂ i₃ hi
  /-
    🎉 no goals
  -/


/-- The associator isomorphism for graded objects. -/
noncomputable def associator [HasGoodTensor₁₂Tensor X₁ X₂ X₃] [HasGoodTensorTensor₂₃ X₁ X₂ X₃] :
  tensorObj (tensorObj X₁ X₂) X₃ ≅ tensorObj X₁ (tensorObj X₂ X₃) :=
    mapBifunctorAssociator (MonoidalCategory.curriedAssociatorNatIso C) ρ₁₂ ρ₂₃ X₁ X₂ X₃


@[reassoc (attr := simp)]
lemma ιTensorObj₃'_associator_hom
    [HasGoodTensor₁₂Tensor X₁ X₂ X₃] [HasGoodTensorTensor₂₃ X₁ X₂ X₃]
    (i₁ i₂ i₃ j : I) (h : i₁ + i₂ + i₃ = j) :
    ιTensorObj₃' X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ (associator X₁ X₂ X₃).hom j =
      (α_ _ _ _).hom ≫ ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j h :=
  ι_mapBifunctorAssociator_hom (MonoidalCategory.curriedAssociatorNatIso C)
    ρ₁₂ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h


@[reassoc (attr := simp)]
lemma ιTensorObj₃_associator_inv
    [HasGoodTensor₁₂Tensor X₁ X₂ X₃] [HasGoodTensorTensor₂₃ X₁ X₂ X₃]
    (i₁ i₂ i₃ j : I) (h : i₁ + i₂ + i₃ = j) :
    ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫ (associator X₁ X₂ X₃).inv j =
      (α_ _ _ _).inv ≫ ιTensorObj₃' X₁ X₂ X₃ i₁ i₂ i₃ j h :=
  ι_mapBifunctorAssociator_inv (MonoidalCategory.curriedAssociatorNatIso C)
    ρ₁₂ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h


variable [HasTensor Y₁ Y₂] [HasTensor (tensorObj Y₁ Y₂) Y₃] [HasTensor Y₂ Y₃]
  [HasTensor Y₁ (tensorObj Y₂ Y₃)] in
lemma associator_naturality (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (f₃ : X₃ ⟶ Y₃)
    [HasGoodTensor₁₂Tensor X₁ X₂ X₃] [HasGoodTensorTensor₂₃ X₁ X₂ X₃]
    [HasGoodTensor₁₂Tensor Y₁ Y₂ Y₃] [HasGoodTensorTensor₂₃ Y₁ Y₂ Y₃] :
    tensorHom (tensorHom f₁ f₂) f₃ ≫ (associator Y₁ Y₂ Y₃).hom =
      (associator X₁ X₂ X₃).hom ≫ tensorHom f₁ (tensorHom f₂ f₃) := by
        #adaptation_note
        /-- this used to be aesop_cat, but that broke with
        https://github.com/leanprover/lean4/pull/4154 -/
        /-
          I : Type u
          inst✝¹⁴ : AddMonoid I
          C : Type u_1
          inst✝¹³ : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹² : CategoryTheory.MonoidalCategory C
          X₁ X₂ X₃ Y₁ Y₂ Y₃ : CategoryTheory.GradedObject I C
          inst✝¹¹ : X₁.HasTensor X₂
          inst✝¹⁰ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
          inst✝⁹ : X₂.HasTensor X₃
          inst✝⁸ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
          inst✝⁷ : Y₁.HasTensor Y₂
          inst✝⁶ : (CategoryTheory.GradedObject.Monoidal.tensorObj Y₁ Y₂).HasTensor Y₃
          inst✝⁵ : Y₂.HasTensor Y₃
          inst✝⁴ : Y₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Y₂ Y₃)
          f₁ : Quiver.Hom X₁ Y₁
          f₂ : Quiver.Hom X₂ Y₂
          f₃ : Quiver.Hom X₃ Y₃
          inst✝³ : X₁.HasGoodTensor₁₂Tensor X₂ X₃
          inst✝² : X₁.HasGoodTensorTensor₂₃ X₂ X₃
          inst✝¹ : Y₁.HasGoodTensor₁₂Tensor Y₂ Y₃
          inst✝ : Y₁.HasGoodTensorTensor₂₃ Y₂ Y₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
        -/
        ext x i₁ i₂ i₃ h : 2
        simp only [categoryOfGradedObjects_comp, ιTensorObj₃'_tensorHom_assoc,
          associator_conjugation, ιTensorObj₃'_associator_hom, assoc, Iso.inv_hom_id_assoc,
          ιTensorObj₃'_associator_hom_assoc, ιTensorObj₃_tensorHom]


/-- Given `Z : C` and three graded objects `X₁`, `X₂` and `X₃` in `GradedObject I C`,
this typeclass expresses that functor `Z ⊗ _` commutes with the coproduct of
the objects `X₁ i₁ ⊗ (X₂ i₂ ⊗ X₃ i₃)` such that `i₁ + i₂ + i₃ = j` for a certain `j`.
See lemma `left_tensor_tensorObj₃_ext`. -/
abbrev _root_.CategoryTheory.GradedObject.HasLeftTensor₃ObjExt (j : I) := PreservesColimit
  (Discrete.functor fun (i : { i : (I × I × I) | i.1 + i.2.1 + i.2.2 = j }) ↦
    (((mapTrifunctor (bifunctorComp₂₃ (curriedTensor C)
      (curriedTensor C)) I I I).obj X₁).obj X₂).obj X₃ i)
   ((curriedTensor C).obj Z)


@[ext (iff := false)]
lemma left_tensor_tensorObj₃_ext {j : I} {A : C} (Z : C)
    (f g : Z ⊗ tensorObj X₁ (tensorObj X₂ X₃) j ⟶ A)
    [H : HasGoodTensorTensor₂₃ X₁ X₂ X₃]
    [hZ : HasLeftTensor₃ObjExt Z X₁ X₂ X₃ j]
    (h : ∀ (i₁ i₂ i₃ : I) (h : i₁ + i₂ + i₃ = j),
      (_ ◁ ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j h) ≫ f =
        (_ ◁ ιTensorObj₃ X₁ X₂ X₃ i₁ i₂ i₃ j h) ≫ g) : f = g := by
    refine (@isColimitOfPreserves C _ C _ _ _ _ ((curriedTensor C).obj Z) _
      (isColimitCofan₃MapBifunctorBifunctor₂₃MapObj (H := H) (j := j)) hZ).hom_ext ?_
    /-
      I : Type u
      inst✝⁴ : AddMonoid I
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.MonoidalCategory C
      X₁ X₂ X₃ : CategoryTheory.GradedObject I C
      inst✝¹ : X₂.HasTensor X₃
      inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
      j : I
      A Z : C
      f g : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj Z (CategoryT …
      H : X₁.HasGoodTensorTensor₂₃ X₂ X₃
      hZ : CategoryTheory.GradedObject.HasLeftTensor₃ObjExt Z X₁ X₂ X₃ j
      h : ∀ (i₁ i₂ i₃ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j), Eq (Category …
      ⊢ ∀ (j_1 : CategoryTheory.Discrete ↑(Set.preimage CategoryTheory.GradedObject. …
    -/
    intro ⟨⟨i₁, i₂, i₃⟩, hi⟩
    /-
      I : Type u
      inst✝⁴ : AddMonoid I
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.MonoidalCategory C
      X₁ X₂ X₃ : CategoryTheory.GradedObject I C
      inst✝¹ : X₂.HasTensor X₃
      inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
      j : I
      A Z : C
      f g : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj Z (CategoryT …
      H : X₁.HasGoodTensorTensor₂₃ X₂ X₃
      hZ : CategoryTheory.GradedObject.HasLeftTensor₃ObjExt Z X₁ X₂ X₃ j
      h : ∀ (i₁ i₂ i₃ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) j), Eq (Category …
      i₁ i₂ i₃ : I
      hi : Membership.mem (Set.preimage CategoryTheory.GradedObject.Monoidal.r₁₂₃ (S …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.MonoidalCategory.c …
    -/
    exact h _ _ _ hi
    /-
      🎉 no goals
    -/


/-- The inclusion
`X₁ i₁ ⊗ X₂ i₂ ⊗ X₃ i₃ ⊗ X₄ i₄ ⟶ tensorObj X₁ (tensorObj X₂ (tensorObj X₃ X₄)) j`
when `i₁ + i₂ + i₃ + i₄ = j`. -/
noncomputable def ιTensorObj₄ (i₁ i₂ i₃ i₄ j : I) (h : i₁ + i₂ + i₃ + i₄ = j) :
    X₁ i₁ ⊗ X₂ i₂ ⊗ X₃ i₃ ⊗ X₄ i₄ ⟶ tensorObj X₁ (tensorObj X₂ (tensorObj X₃ X₄)) j :=
  (_ ◁ ιTensorObj₃ X₂ X₃ X₄ i₂ i₃ i₄ _ rfl) ≫
    ιTensorObj X₁ (tensorObj X₂ (tensorObj X₃ X₄)) i₁ (i₂ + i₃ + i₄) j
          /-
            I : Type u
            inst✝⁵ : AddMonoid I
            C : Type u_1
            inst✝⁴ : CategoryTheory.Category.{?u.163123, u_1} C
            inst✝³ : CategoryTheory.MonoidalCategory C
            X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
            inst✝² : X₃.HasTensor X₄
            inst✝¹ : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
            inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Categ …
            i₁ i₂ i₃ i₄ j : I
            h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄) j
            ⊢ Eq (HAdd.hAdd i₁ (HAdd.hAdd (HAdd.hAdd i₂ i₃) i₄)) j
          -/
      (by rw [← h, ← add_assoc, ← add_assoc])
          /-
            🎉 no goals
          -/


lemma ιTensorObj₄_eq (i₁ i₂ i₃ i₄ j : I) (h : i₁ + i₂ + i₃ + i₄ = j) (i₂₃₄ : I)
    (hi : i₂ + i₃ + i₄ = i₂₃₄) :
    ιTensorObj₄ X₁ X₂ X₃ X₄ i₁ i₂ i₃ i₄ j h =
      (_ ◁ ιTensorObj₃ X₂ X₃ X₄ i₂ i₃ i₄ _ hi) ≫
        ιTensorObj X₁ (tensorObj X₂ (tensorObj X₃ X₄)) i₁ i₂₃₄ j
              /-
                I : Type u
                inst✝⁵ : AddMonoid I
                C : Type u_1
                inst✝⁴ : CategoryTheory.Category.{?u.205340, u_1} C
                inst✝³ : CategoryTheory.MonoidalCategory C
                X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
                inst✝² : X₃.HasTensor X₄
                inst✝¹ : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
                inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Categ …
                i₁ i₂ i₃ i₄ j : I
                h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄) j
                i₂₃₄ : I
                hi : Eq (HAdd.hAdd (HAdd.hAdd i₂ i₃) i₄) i₂₃₄
                ⊢ Eq (HAdd.hAdd i₁ i₂₃₄) j
              -/
          (by rw [← hi, ← add_assoc, ← add_assoc, h]) := by
              /-
                🎉 no goals
              -/
  /-
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝² : X₃.HasTensor X₄
    inst✝¹ : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Categ …
    i₁ i₂ i₃ i₄ j : I
    h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄) j
    i₂₃₄ : I
    hi : Eq (HAdd.hAdd (HAdd.hAdd i₂ i₃) i₄) i₂₃₄
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.ιTensorObj₄ X₁ X₂ X₃ X₄ i₁ i₂ i₃ i₄ …
  -/
  subst hi
  /-
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝² : X₃.HasTensor X₄
    inst✝¹ : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Categ …
    i₁ i₂ i₃ i₄ j : I
    h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄) j
    ⊢ Eq (CategoryTheory.GradedObject.Monoidal.ιTensorObj₄ X₁ X₂ X₃ X₄ i₁ i₂ i₃ i₄ …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given four graded objects, this is the condition
`HasLeftTensor₃ObjExt (X₁ i₁) X₂ X₃ X₄ i₂₃₄` for all indices `i₁` and `i₂₃₄`,
see the lemma `tensorObj₄_ext`. -/
abbrev _root_.CategoryTheory.GradedObject.HasTensor₄ObjExt :=
  ∀ (i₁ i₂₃₄ : I), HasLeftTensor₃ObjExt (X₁ i₁) X₂ X₃ X₄ i₂₃₄


@[ext (iff := false)]
lemma tensorObj₄_ext {j : I} {A : C} (f g : tensorObj X₁ (tensorObj X₂ (tensorObj X₃ X₄)) j ⟶ A)
    [HasGoodTensorTensor₂₃ X₂ X₃ X₄]
    [H : HasTensor₄ObjExt X₁ X₂ X₃ X₄]
    (h : ∀ (i₁ i₂ i₃ i₄ : I) (h : i₁ + i₂ + i₃ + i₄ = j),
      ιTensorObj₄ X₁ X₂ X₃ X₄ i₁ i₂ i₃ i₄ j h ≫ f =
        ιTensorObj₄ X₁ X₂ X₃ X₄ i₁ i₂ i₃ i₄ j h ≫ g) : f = g := by
  /-
    I : Type u
    inst✝⁶ : AddMonoid I
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝³ : X₃.HasTensor X₄
    inst✝² : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝¹ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Cate …
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    inst✝ : X₂.HasGoodTensorTensor₂₃ X₃ X₄
    H : X₁.HasTensor₄ObjExt X₂ X₃ X₄
    h : ∀ (i₁ i₂ i₃ i₄ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄ …
    ⊢ Eq f g
  -/
  apply tensorObj_ext
  /-
    case h
    I : Type u
    inst✝⁶ : AddMonoid I
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝³ : X₃.HasTensor X₄
    inst✝² : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝¹ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Cate …
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    inst✝ : X₂.HasGoodTensorTensor₂₃ X₃ X₄
    H : X₁.HasTensor₄ObjExt X₂ X₃ X₄
    h : ∀ (i₁ i₂ i₃ i₄ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄ …
    ⊢ ∀ (i₁ i₂ : I) (hi : Eq (HAdd.hAdd i₁ i₂) j), Eq (CategoryTheory.CategoryStru …
  -/
  intro i₁ i₂₃₄ h'
  /-
    case h
    I : Type u
    inst✝⁶ : AddMonoid I
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝³ : X₃.HasTensor X₄
    inst✝² : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝¹ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Cate …
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    inst✝ : X₂.HasGoodTensorTensor₂₃ X₃ X₄
    H : X₁.HasTensor₄ObjExt X₂ X₃ X₄
    h : ∀ (i₁ i₂ i₃ i₄ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄ …
    i₁ i₂₃₄ : I
    h' : Eq (HAdd.hAdd i₁ i₂₃₄) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  apply left_tensor_tensorObj₃_ext
  /-
    case h.h
    I : Type u
    inst✝⁶ : AddMonoid I
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝³ : X₃.HasTensor X₄
    inst✝² : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝¹ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Cate …
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    inst✝ : X₂.HasGoodTensorTensor₂₃ X₃ X₄
    H : X₁.HasTensor₄ObjExt X₂ X₃ X₄
    h : ∀ (i₁ i₂ i₃ i₄ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄ …
    i₁ i₂₃₄ : I
    h' : Eq (HAdd.hAdd i₁ i₂₃₄) j
    ⊢ ∀ (i₁_1 i₂ i₃ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd i₁_1 i₂) i₃) i₂₃₄), Eq (Cat …
  -/
  intro i₂ i₃ i₄ h''
  /-
    case h.h
    I : Type u
    inst✝⁶ : AddMonoid I
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝³ : X₃.HasTensor X₄
    inst✝² : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝¹ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Cate …
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    inst✝ : X₂.HasGoodTensorTensor₂₃ X₃ X₄
    H : X₁.HasTensor₄ObjExt X₂ X₃ X₄
    h : ∀ (i₁ i₂ i₃ i₄ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄ …
    i₁ i₂₃₄ : I
    h' : Eq (HAdd.hAdd i₁ i₂₃₄) j
    i₂ i₃ i₄ : I
    h'' : Eq (HAdd.hAdd (HAdd.hAdd i₂ i₃) i₄) i₂₃₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  have hj : i₁ + i₂ + i₃ + i₄ = j := by simp only [← h', ← h'', add_assoc]
  /-
    case h.h
    I : Type u
    inst✝⁶ : AddMonoid I
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝³ : X₃.HasTensor X₄
    inst✝² : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝¹ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Cate …
    j : I
    A : C
    f g : Quiver.Hom (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryT …
    inst✝ : X₂.HasGoodTensorTensor₂₃ X₃ X₄
    H : X₁.HasTensor₄ObjExt X₂ X₃ X₄
    h : ∀ (i₁ i₂ i₃ i₄ : I) (h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄ …
    i₁ i₂₃₄ : I
    h' : Eq (HAdd.hAdd i₁ i₂₃₄) j
    i₂ i₃ i₄ : I
    h'' : Eq (HAdd.hAdd (HAdd.hAdd i₂ i₃) i₄) i₂₃₄
    hj : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simpa only [assoc, ιTensorObj₄_eq X₁ X₂ X₃ X₄ i₁ i₂ i₃ i₄ j hj i₂₃₄ h''] using h i₁ i₂ i₃ i₄ hj
  /-
    🎉 no goals
  -/


@[reassoc]
lemma pentagon_inv :
    tensorHom (𝟙 X₁) (associator X₂ X₃ X₄).inv ≫ (associator X₁ (tensorObj X₂ X₃) X₄).inv ≫
        tensorHom (associator X₁ X₂ X₃).inv (𝟙 X₄) =
    (associator X₁ X₂ (tensorObj X₃ X₄)).inv ≫ (associator (tensorObj X₁ X₂) X₃ X₄).inv := by
  /-
    I : Type u
    inst✝²⁵ : AddMonoid I
    C : Type u_1
    inst✝²⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝²³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝²² : X₁.HasTensor X₂
    inst✝²¹ : X₂.HasTensor X₃
    inst✝²⁰ : X₃.HasTensor X₄
    inst✝¹⁹ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
    inst✝¹⁸ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
    inst✝¹⁷ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃).HasTensor X₄
    inst✝¹⁶ : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝¹⁵ : (CategoryTheory.GradedObject.Monoidal.tensorObj (CategoryTheory.Grad …
    inst✝¹⁴ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryTheory.G …
    inst✝¹³ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj (Catego …
    inst✝¹² : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Cat …
    inst✝¹¹ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor (Ca …
    inst✝¹⁰ : X₁.HasGoodTensor₁₂Tensor X₂ X₃
    inst✝⁹ : X₁.HasGoodTensorTensor₂₃ X₂ X₃
    inst✝⁸ : X₁.HasGoodTensor₁₂Tensor (CategoryTheory.GradedObject.Monoidal.tensor …
    inst✝⁷ : X₁.HasGoodTensorTensor₂₃ (CategoryTheory.GradedObject.Monoidal.tensor …
    inst✝⁶ : X₂.HasGoodTensor₁₂Tensor X₃ X₄
    inst✝⁵ : X₂.HasGoodTensorTensor₂₃ X₃ X₄
    inst✝⁴ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasGoodTensor₁ …
    inst✝³ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasGoodTensorT …
    inst✝² : X₁.HasGoodTensor₁₂Tensor X₂ (CategoryTheory.GradedObject.Monoidal.ten …
    inst✝¹ : X₁.HasGoodTensorTensor₂₃ X₂ (CategoryTheory.GradedObject.Monoidal.ten …
    inst✝ : X₁.HasTensor₄ObjExt X₂ X₃ X₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  ext j i₁ i₂ i₃ i₄ h
  /-
    case h.h
    I : Type u
    inst✝²⁵ : AddMonoid I
    C : Type u_1
    inst✝²⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝²³ : CategoryTheory.MonoidalCategory C
    X₁ X₂ X₃ X₄ : CategoryTheory.GradedObject I C
    inst✝²² : X₁.HasTensor X₂
    inst✝²¹ : X₂.HasTensor X₃
    inst✝²⁰ : X₃.HasTensor X₄
    inst✝¹⁹ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor X₃
    inst✝¹⁸ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃)
    inst✝¹⁷ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ X₃).HasTensor X₄
    inst✝¹⁶ : X₂.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₃ X₄)
    inst✝¹⁵ : (CategoryTheory.GradedObject.Monoidal.tensorObj (CategoryTheory.Grad …
    inst✝¹⁴ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ (CategoryTheory.G …
    inst✝¹³ : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj (Catego …
    inst✝¹² : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj X₂ (Cat …
    inst✝¹¹ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasTensor (Ca …
    inst✝¹⁰ : X₁.HasGoodTensor₁₂Tensor X₂ X₃
    inst✝⁹ : X₁.HasGoodTensorTensor₂₃ X₂ X₃
    inst✝⁸ : X₁.HasGoodTensor₁₂Tensor (CategoryTheory.GradedObject.Monoidal.tensor …
    inst✝⁷ : X₁.HasGoodTensorTensor₂₃ (CategoryTheory.GradedObject.Monoidal.tensor …
    inst✝⁶ : X₂.HasGoodTensor₁₂Tensor X₃ X₄
    inst✝⁵ : X₂.HasGoodTensorTensor₂₃ X₃ X₄
    inst✝⁴ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasGoodTensor₁ …
    inst✝³ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ X₂).HasGoodTensorT …
    inst✝² : X₁.HasGoodTensor₁₂Tensor X₂ (CategoryTheory.GradedObject.Monoidal.ten …
    inst✝¹ : X₁.HasGoodTensorTensor₂₃ X₂ (CategoryTheory.GradedObject.Monoidal.ten …
    inst✝ : X₁.HasTensor₄ObjExt X₂ X₃ X₄
    j i₁ i₂ i₃ i₄ : I
    h : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd i₁ i₂) i₃) i₄) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  dsimp only [categoryOfGradedObjects_comp]
  conv_lhs =>
    rw [ιTensorObj₄_eq X₁ X₂ X₃ X₄ i₁ i₂ i₃ i₄ j h _ rfl, assoc, ι_tensorHom_assoc]
    dsimp only [categoryOfGradedObjects_id, id_eq, eq_mpr_eq_cast, cast_eq]
    rw [id_tensorHom, ← MonoidalCategory.whiskerLeft_comp_assoc, ιTensorObj₃_associator_inv,
      ιTensorObj₃'_eq X₂ X₃ X₄ i₂ i₃ i₄ _ rfl _ rfl, MonoidalCategory.whiskerLeft_comp_assoc,
      MonoidalCategory.whiskerLeft_comp_assoc,
      ← ιTensorObj₃_eq_assoc X₁ (tensorObj X₂ X₃) X₄ i₁ (i₂ + i₃) i₄ j
        (by simp only [← add_assoc, h]) _ rfl, ιTensorObj₃_associator_inv_assoc,
      ιTensorObj₃'_eq_assoc X₁ (tensorObj X₂ X₃) X₄ i₁ (i₂ + i₃) i₄ j
        (by simp only [← add_assoc, h]) (i₁ + i₂ + i₃) (by rw [add_assoc]), ι_tensorHom]
    dsimp only [id_eq, eq_mpr_eq_cast, categoryOfGradedObjects_id]
    rw [tensorHom_id, whisker_assoc_symm_assoc, Iso.hom_inv_id_assoc,
      ← MonoidalCategory.comp_whiskerRight_assoc, ← MonoidalCategory.comp_whiskerRight_assoc,
      ← ιTensorObj₃_eq X₁ X₂ X₃ i₁ i₂ i₃ _ rfl _ rfl, ιTensorObj₃_associator_inv,
      MonoidalCategory.comp_whiskerRight_assoc, MonoidalCategory.pentagon_inv_assoc]
  conv_rhs =>
    rw [ιTensorObj₄_eq X₁ X₂ X₃ X₄ i₁ i₂ i₃ i₄ _ _ _ rfl,
      ιTensorObj₃_eq X₂ X₃ X₄ i₂ i₃ i₄ _ rfl _ rfl, assoc,
      MonoidalCategory.whiskerLeft_comp_assoc,
      ← ιTensorObj₃_eq_assoc X₁ X₂ (tensorObj X₃ X₄) i₁ i₂ (i₃ + i₄) j
        (by rw [← add_assoc, h]) (i₂ + i₃ + i₄) (by rw [add_assoc]),
      ιTensorObj₃_associator_inv_assoc, associator_inv_naturality_right_assoc,
      ιTensorObj₃'_eq_assoc X₁ X₂ (tensorObj X₃ X₄) i₁ i₂ (i₃ + i₄) j
        (by rw [← add_assoc, h]) _ rfl, whisker_exchange_assoc,
      ← ιTensorObj₃_eq_assoc (tensorObj X₁ X₂) X₃ X₄ (i₁ + i₂) i₃ i₄ j h _ rfl,
      ιTensorObj₃_associator_inv, whiskerRight_tensor_assoc, Iso.hom_inv_id_assoc,
      ιTensorObj₃'_eq (tensorObj X₁ X₂) X₃ X₄ (i₁ + i₂) i₃ i₄ j h _ rfl,
      ← MonoidalCategory.comp_whiskerRight_assoc,
      ← ιTensorObj₃'_eq X₁ X₂ X₃ i₁ i₂ i₃ _ rfl _ rfl]


lemma pentagon : tensorHom (associator X₁ X₂ X₃).hom (𝟙 X₄) ≫
    (associator X₁ (tensorObj X₂ X₃) X₄).hom ≫ tensorHom (𝟙 X₁) (associator X₂ X₃ X₄).hom =
    (associator (tensorObj X₁ X₂) X₃ X₄).hom ≫ (associator X₁ X₂ (tensorObj X₃ X₄)).hom := by
  rw [← cancel_epi (associator (tensorObj X₁ X₂) X₃ X₄).inv,
    ← cancel_epi (associator X₁ X₂ (tensorObj X₃ X₄)).inv, Iso.inv_hom_id_assoc,
    Iso.inv_hom_id, ← pentagon_inv_assoc, ← tensor_comp_assoc, id_comp, Iso.inv_hom_id,
    tensor_id, id_comp, Iso.inv_hom_id_assoc, ← tensor_comp, id_comp, Iso.inv_hom_id,
    tensor_id]


/-- The unit of the tensor product on graded objects is `(single₀ I).obj (𝟙_ C)`. -/
noncomputable def tensorUnit : GradedObject I C := (single₀ I).obj (𝟙_ C)


/-- The canonical isomorphism `tensorUnit 0 ≅ 𝟙_ C` -/
noncomputable def tensorUnit₀ : (tensorUnit : GradedObject I C) 0 ≅ 𝟙_ C :=
  singleObjApplyIso (0 : I) (𝟙_ C)


/-- `tensorUnit i` is an initial object when `i ≠ 0`. -/
noncomputable def isInitialTensorUnitApply (i : I) (hi : i ≠ 0) :
    IsInitial ((tensorUnit : GradedObject I C) i) :=
  isInitialSingleObjApply _ _ _ hi


instance : HasTensor tensorUnit X :=
  mapBifunctorLeftUnitor_hasMap _ _ (leftUnitorNatIso C) _ zero_add _


instance : HasMap (((mapBifunctor (curriedTensor C) I I).obj
    ((single₀ I).obj (𝟙_ C))).obj X) (fun ⟨i₁, i₂⟩ => i₁ + i₂) :=
  (inferInstance : HasTensor tensorUnit X)


/-- The left unitor isomorphism for graded objects. -/
noncomputable def leftUnitor : tensorObj tensorUnit X ≅ X :=
    mapBifunctorLeftUnitor (curriedTensor C) (𝟙_ C)
      (leftUnitorNatIso C) (fun (⟨i₁, i₂⟩ : I × I) => i₁ + i₂) zero_add X


lemma leftUnitor_inv_apply (i : I) :
    (leftUnitor X).inv i = (λ_ (X i)).inv ≫ tensorUnit₀.inv ▷ (X i) ≫
      ιTensorObj tensorUnit X 0 i i (zero_add i) := rfl


@[reassoc (attr := simp)]
lemma leftUnitor_naturality (φ : X ⟶ X') :
    tensorHom (𝟙 (tensorUnit)) φ ≫ (leftUnitor X').hom =
      (leftUnitor X).hom ≫ φ := by
  /-
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : DecidableEq I
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    X X' : CategoryTheory.GradedObject I C
    φ : Quiver.Hom X X'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  apply mapBifunctorLeftUnitor_naturality
  /-
    🎉 no goals
  -/


instance : HasTensor X tensorUnit :=
  mapBifunctorRightUnitor_hasMap (curriedTensor C) _
    (rightUnitorNatIso C) _ add_zero _


instance : HasMap (((mapBifunctor (curriedTensor C) I I).obj X).obj
    ((single₀ I).obj (𝟙_ C))) (fun ⟨i₁, i₂⟩ => i₁ + i₂) :=
  (inferInstance : HasTensor X tensorUnit)


/-- The right unitor isomorphism for graded objects. -/
noncomputable def rightUnitor : tensorObj X tensorUnit ≅ X :=
    mapBifunctorRightUnitor (curriedTensor C) (𝟙_ C)
      (rightUnitorNatIso C) (fun (⟨i₁, i₂⟩ : I × I) => i₁ + i₂) add_zero X


lemma rightUnitor_inv_apply (i : I) :
    (rightUnitor X).inv i = (ρ_ (X i)).inv ≫ (X i) ◁ tensorUnit₀.inv ≫
      ιTensorObj X tensorUnit i 0 i (add_zero i) := rfl


@[reassoc (attr := simp)]
lemma rightUnitor_naturality (φ : X ⟶ X') :
    tensorHom φ (𝟙 (tensorUnit)) ≫ (rightUnitor X').hom =
      (rightUnitor X).hom ≫ φ := by
  /-
    I : Type u
    inst✝⁵ : AddMonoid I
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : DecidableEq I
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fun …
    X X' : CategoryTheory.GradedObject I C
    φ : Quiver.Hom X X'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.Monoidal …
  -/
  apply mapBifunctorRightUnitor_naturality
  /-
    🎉 no goals
  -/


lemma triangle :
    (associator X₁ tensorUnit X₃).hom ≫ tensorHom (𝟙 X₁) (leftUnitor X₃).hom =
      tensorHom (rightUnitor X₁).hom (𝟙 X₃) := by
  convert mapBifunctor_triangle (curriedAssociatorNatIso C) (𝟙_ C)
    (rightUnitorNatIso C) (leftUnitorNatIso C) (triangleIndexData I) X₁ X₃ (by simp)
  /-
    case convert_3
    I : Type u
    inst✝¹¹ : AddMonoid I
    C : Type u_1
    inst✝¹⁰ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁹ : CategoryTheory.MonoidalCategory C
    inst✝⁸ : DecidableEq I
    inst✝⁷ : CategoryTheory.Limits.HasInitial C
    inst✝⁶ : ∀ (X₁ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
    inst✝⁵ : ∀ (X₂ : C), CategoryTheory.Limits.PreservesColimit (CategoryTheory.Fu …
    X₁ X₃ : CategoryTheory.GradedObject I C
    inst✝⁴ : X₁.HasTensor X₃
    inst✝³ : (CategoryTheory.GradedObject.Monoidal.tensorObj X₁ CategoryTheory.Gra …
    inst✝² : X₁.HasTensor (CategoryTheory.GradedObject.Monoidal.tensorObj Category …
    inst✝¹ : X₁.HasGoodTensor₁₂Tensor CategoryTheory.GradedObject.Monoidal.tensorU …
    inst✝ : X₁.HasGoodTensorTensor₂₃ CategoryTheory.GradedObject.Monoidal.tensorUn …
    ⊢ CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj (CategoryTheory.MonoidalC …
  -/
  all_goals assumption
  /-
    🎉 no goals
  -/


noncomputable instance monoidalCategory : MonoidalCategory (GradedObject I C) where
  tensorObj X Y := Monoidal.tensorObj X Y
  tensorHom f g := Monoidal.tensorHom f g
  tensorHom_def f g := Monoidal.tensorHom_def f g
  whiskerLeft X _ _ φ := Monoidal.whiskerLeft X φ
  whiskerRight {_ _ φ Y} := Monoidal.whiskerRight φ Y
  tensorUnit := Monoidal.tensorUnit
  associator X₁ X₂ X₃ := Monoidal.associator X₁ X₂ X₃
  associator_naturality f₁ f₂ f₃ := Monoidal.associator_naturality f₁ f₂ f₃
  leftUnitor X := Monoidal.leftUnitor X
  leftUnitor_naturality := Monoidal.leftUnitor_naturality
  rightUnitor X := Monoidal.rightUnitor X
  rightUnitor_naturality := Monoidal.rightUnitor_naturality
  tensor_comp f₁ f₂ g₁ g₂ := Monoidal.tensor_comp f₁ g₁ f₂ g₂
  pentagon X₁ X₂ X₃ X₄ := Monoidal.pentagon X₁ X₂ X₃ X₄
  triangle X₁ X₂ := Monoidal.triangle X₁ X₂


instance (n : ℕ) : Finite ((fun (i : ℕ × ℕ) => i.1 + i.2) ⁻¹' {n}) := by
  refine Finite.of_injective (fun ⟨⟨i₁, i₂⟩, (hi : i₁ + i₂ = n)⟩ =>
    ((⟨i₁, by omega⟩, ⟨i₂, by omega⟩) : Fin (n + 1) × Fin (n + 1) )) ?_
  /-
    I : Type u
    inst✝² : AddMonoid I
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.410906, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    n : Nat
    ⊢ Function.Injective fun x => CategoryTheory.GradedObject.instFiniteElemProdNa …
  -/
  rintro ⟨⟨_, _⟩, _⟩ ⟨⟨_, _⟩, _⟩ h
  /-
    case mk.mk.mk.mk
    I : Type u
    inst✝² : AddMonoid I
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.410906, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    n fst✝¹ snd✝¹ : Nat
    property✝¹ : Membership.mem (Set.preimage (fun i => HAdd.hAdd i.1 i.2) (Single …
    fst✝ snd✝ : Nat
    property✝ : Membership.mem (Set.preimage (fun i => HAdd.hAdd i.1 i.2) (Singlet …
    h : Eq ((fun x => CategoryTheory.GradedObject.instFiniteElemProdNatPreimageHAd …
    ⊢ Eq ⟨{ fst := fst✝¹, snd := snd✝¹ }, property✝¹⟩ ⟨{ fst := fst✝, snd := snd✝  …
  -/
  simpa using h
  /-
    🎉 no goals
  -/


instance (n : ℕ) : Finite ({ i : (ℕ × ℕ × ℕ) | i.1 + i.2.1 + i.2.2 = n }) := by
  refine Finite.of_injective (fun ⟨⟨i₁, i₂, i₃⟩, (hi : i₁ + i₂ + i₃ = n)⟩ =>
    (⟨⟨i₁, by omega⟩, ⟨i₂, by omega⟩, ⟨i₃, by omega⟩⟩ :
      Fin (n + 1) × Fin (n + 1) × Fin (n + 1))) ?_
  /-
    I : Type u
    inst✝² : AddMonoid I
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.412092, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    n : Nat
    ⊢ Function.Injective fun x => CategoryTheory.GradedObject.instFiniteElemProdNa …
  -/
  rintro ⟨⟨_, _, _⟩, _⟩ ⟨⟨_, _, _⟩, _⟩ h
  /-
    case mk.mk.mk.mk.mk.mk
    I : Type u
    inst✝² : AddMonoid I
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.412092, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    n fst✝³ fst✝² snd✝¹ : Nat
    property✝¹ : Membership.mem (setOf fun i => Eq (HAdd.hAdd (HAdd.hAdd i.1 i.2.1 …
    fst✝¹ fst✝ snd✝ : Nat
    property✝ : Membership.mem (setOf fun i => Eq (HAdd.hAdd (HAdd.hAdd i.1 i.2.1) …
    h : Eq ((fun x => CategoryTheory.GradedObject.instFiniteElemProdNatSetOfEqHAdd …
    ⊢ Eq ⟨{ fst := fst✝³, snd := { fst := fst✝², snd := snd✝¹ } }, property✝¹⟩ ⟨{  …
  -/
  simpa using h
  /-
    🎉 no goals
  -/


