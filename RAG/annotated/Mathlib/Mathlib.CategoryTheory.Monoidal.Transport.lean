/-- The data needed to induce a `MonoidalCategory` via the functor `F`; namely, pre-existing
definitions of `⊗`, `𝟙_`, `▷`, `◁` that are preserved by `F`.
-/
structure InducingFunctorData [MonoidalCategoryStruct D] (F : D ⥤ C) where
  /-- Analogous to `CategoryTheory.LaxMonoidalFunctor.μIso` -/
  μIso : ∀ X Y,
    F.obj X ⊗ F.obj Y ≅ F.obj (X ⊗ Y)
  whiskerLeft_eq : ∀ (X : D) {Y₁ Y₂ : D} (f : Y₁ ⟶ Y₂),
    F.map (X ◁ f) = (μIso _ _).inv ≫ (F.obj X ◁ F.map f) ≫ (μIso _ _).hom := by
    aesop_cat
  whiskerRight_eq : ∀ {X₁ X₂ : D} (f : X₁ ⟶ X₂) (Y : D),
    F.map (f ▷ Y) = (μIso _ _).inv ≫ (F.map f ▷ F.obj Y) ≫ (μIso _ _).hom := by
    aesop_cat
  tensorHom_eq : ∀ {X₁ Y₁ X₂ Y₂ : D} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂),
    F.map (f ⊗ g) = (μIso _ _).inv ≫ (F.map f ⊗ F.map g) ≫ (μIso _ _).hom := by
    aesop_cat
  /-- Analogous to `CategoryTheory.LaxMonoidalFunctor.εIso` -/
  εIso : 𝟙_ _ ≅ F.obj (𝟙_ _)
  associator_eq : ∀ X Y Z : D,
    F.map (α_ X Y Z).hom =
      (((μIso _ _).symm ≪≫ ((μIso _ _).symm ⊗ .refl _))
        ≪≫ α_ (F.obj X) (F.obj Y) (F.obj Z)
        ≪≫ ((.refl _ ⊗ μIso _ _) ≪≫ μIso _ _)).hom := by
    aesop_cat
  leftUnitor_eq : ∀ X : D,
    F.map (λ_ X).hom =
      (((μIso _ _).symm ≪≫ (εIso.symm ⊗ .refl _)) ≪≫ λ_ (F.obj X)).hom := by
    aesop_cat
  rightUnitor_eq : ∀ X : D,
    F.map (ρ_ X).hom =
      (((μIso _ _).symm ≪≫ (.refl _ ⊗ εIso.symm)) ≪≫ ρ_ (F.obj X)).hom := by
    aesop_cat


/--
Induce the lawfulness of the monoidal structure along an faithful functor of (plain) categories,
where the operations are already defined on the destination type `D`.

The functor `F` must preserve all the data parts of the monoidal structure between the two
categories.

-/
def induced [MonoidalCategoryStruct D] (F : D ⥤ C) [F.Faithful]
    (fData : InducingFunctorData F) :
    MonoidalCategory.{v₂} D where
  tensorHom_def {X₁ Y₁ X₂ Y₂} f g := F.map_injective <| by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
      F : CategoryTheory.Functor D C
      inst✝ : F.Faithful
      fData : CategoryTheory.Monoidal.InducingFunctorData F
      X₁ Y₁ X₂ Y₂ : D
      f : Quiver.Hom X₁ Y₁
      g : Quiver.Hom X₂ Y₂
      ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.tensorHom f g)) (F.map (Cat …
    -/
    rw [fData.tensorHom_eq, Functor.map_comp, fData.whiskerRight_eq, fData.whiskerLeft_eq]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
      F : CategoryTheory.Functor D C
      inst✝ : F.Faithful
      fData : CategoryTheory.Monoidal.InducingFunctorData F
      X₁ Y₁ X₂ Y₂ : D
      f : Quiver.Hom X₁ Y₁
      g : Quiver.Hom X₂ Y₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fData.μIso X₁ X₂).inv (CategoryTheor …
    -/
    simp only [tensorHom_def, assoc, Iso.hom_inv_id_assoc]
    /-
      🎉 no goals
    -/
                                           /-
                                             C : Type u₁
                                             inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                             inst✝³ : CategoryTheory.MonoidalCategory C
                                             D : Type u₂
                                             inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                             inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
                                             F : CategoryTheory.Functor D C
                                             inst✝ : F.Faithful
                                             fData : CategoryTheory.Monoidal.InducingFunctorData F
                                             X₁ X₂ : D
                                             ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.C …
                                           -/
  tensor_id X₁ X₂ := F.map_injective <| by cases fData; aesop_cat
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                                       /-
                                                                         C : Type u₁
                                                                         inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                         inst✝³ : CategoryTheory.MonoidalCategory C
                                                                         D : Type u₂
                                                                         inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                         inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
                                                                         F : CategoryTheory.Functor D C
                                                                         inst✝ : F.Faithful
                                                                         fData : CategoryTheory.Monoidal.InducingFunctorData F
                                                                         X₁ Y₁ Z₁ X₂ Y₂ Z₂ : D
                                                                         f₁ : Quiver.Hom X₁ Y₁
                                                                         f₂ : Quiver.Hom X₂ Y₂
                                                                         g₁ : Quiver.Hom Y₁ Z₁
                                                                         g₂ : Quiver.Hom Y₂ Z₂
                                                                         ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.C …
                                                                       -/
  tensor_comp {X₁ Y₁ Z₁ X₂ Y₂ Z₂} f₁ f₂ g₁ g₂ := F.map_injective <| by cases fData; aesop_cat
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                                              /-
                                                C : Type u₁
                                                inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                inst✝³ : CategoryTheory.MonoidalCategory C
                                                D : Type u₂
                                                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
                                                F : CategoryTheory.Functor D C
                                                inst✝ : F.Faithful
                                                fData : CategoryTheory.Monoidal.InducingFunctorData F
                                                X Y : D
                                                ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X (CategoryTheo …
                                              -/
  whiskerLeft_id X Y := F.map_injective <| by simp [fData.whiskerLeft_eq]
                                              /-
                                                🎉 no goals
                                              -/
                                               /-
                                                 C : Type u₁
                                                 inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                 inst✝³ : CategoryTheory.MonoidalCategory C
                                                 D : Type u₂
                                                 inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                 inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
                                                 F : CategoryTheory.Functor D C
                                                 inst✝ : F.Faithful
                                                 fData : CategoryTheory.Monoidal.InducingFunctorData F
                                                 X Y : D
                                                 ⊢ Eq (F.map (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheor …
                                               -/
  id_whiskerRight X Y := F.map_injective <| by simp [fData.whiskerRight_eq]
                                               /-
                                                 🎉 no goals
                                               -/
                                        /-
                                          C : Type u₁
                                          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                          inst✝³ : CategoryTheory.MonoidalCategory C
                                          D : Type u₂
                                          inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                          inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
                                          F : CategoryTheory.Functor D C
                                          inst✝ : F.Faithful
                                          fData : CategoryTheory.Monoidal.InducingFunctorData F
                                          X Y : D
                                          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
                                        -/
  triangle X Y := F.map_injective <| by cases fData; aesop_cat
                                                     /-
                                                       🎉 no goals
                                                     -/
  pentagon W X Y Z := F.map_injective <| by
    simp only [Functor.map_comp, fData.whiskerRight_eq, fData.associator_eq, Iso.trans_assoc,
      Iso.trans_hom, Iso.symm_hom, tensorIso_hom, Iso.refl_hom, tensorHom_id, id_tensorHom,
      comp_whiskerRight, whisker_assoc, assoc, fData.whiskerLeft_eq,
      MonoidalCategory.whiskerLeft_comp, Iso.hom_inv_id_assoc, whiskerLeft_hom_inv_assoc,
      hom_inv_whiskerRight_assoc, Iso.inv_hom_id_assoc, Iso.cancel_iso_inv_left]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
      F : CategoryTheory.Functor D C
      inst✝ : F.Faithful
      fData : CategoryTheory.Monoidal.InducingFunctorData F
      W X Y Z : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_lhs 5 6 =>
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
      F : CategoryTheory.Functor D C
      inst✝ : F.Faithful
      fData : CategoryTheory.Monoidal.InducingFunctorData F
      W X Y Z : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
      F : CategoryTheory.Functor D C
      inst✝ : F.Faithful
      fData : CategoryTheory.Monoidal.InducingFunctorData F
      X Y : D
      f : Quiver.Hom X Y
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
      rw [← MonoidalCategory.whiskerLeft_comp, hom_inv_whiskerRight]
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
      F : CategoryTheory.Functor D C
      inst✝ : F.Faithful
      fData : CategoryTheory.Monoidal.InducingFunctorData F
      X₁ X₂ X₃ Y₁ Y₂ Y₃ : D
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₂
      f₃ : Quiver.Hom X₃ Y₃
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
      F : CategoryTheory.Functor D C
      inst✝ : F.Faithful
      fData : CategoryTheory.Monoidal.InducingFunctorData F
      X Y : D
      f : Quiver.Hom X Y
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCatego …
    -/
    /-
      🎉 no goals
    -/
    rw [whisker_exchange_assoc]
    /-
      🎉 no goals
    -/
    simp
  leftUnitor_naturality {X Y : D} f := F.map_injective <| by
    simp [fData.leftUnitor_eq, fData.whiskerLeft_eq, whisker_exchange_assoc]
  rightUnitor_naturality {X Y : D} f := F.map_injective <| by
    simp [fData.rightUnitor_eq, fData.whiskerRight_eq, ← whisker_exchange_assoc]
  associator_naturality {X₁ X₂ X₃ Y₁ Y₂ Y₃} f₁ f₂ f₃ := F.map_injective <| by
    simp [fData.tensorHom_eq, fData.associator_eq, tensorHom_def, whisker_exchange_assoc]


/-- A faithful functor equipped with a `InducingFunctorData` structure is monoidal. -/
def fromInducedCoreMonoidal [MonoidalCategoryStruct D] (F : D ⥤ C) [F.Faithful]
    (fData : InducingFunctorData F) :
    letI := induced F fData
    F.CoreMonoidal := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.MonoidalCategoryStruct D
    F : CategoryTheory.Functor D C
    inst✝ : F.Faithful
    fData : CategoryTheory.Monoidal.InducingFunctorData F
    ⊢ F.CoreMonoidal
  -/
  letI := induced F fData
  exact
    { εIso := fData.εIso
      μIso := fData.μIso
      μIso_hom_natural_left := fun _ ↦ by simp [fData.whiskerRight_eq]
      μIso_hom_natural_right := fun _ ↦ by simp [fData.whiskerLeft_eq]
      associativity := fun _ _ _ ↦ by simp [fData.associator_eq]
      left_unitality := fun _ ↦ by simp [fData.leftUnitor_eq]
      right_unitality := fun _ ↦ by simp [fData.rightUnitor_eq] }


instance fromInducedMonoidal [MonoidalCategoryStruct D] (F : D ⥤ C) [F.Faithful]
    (fData : InducingFunctorData F) :
    letI := induced F fData
    F.Monoidal :=
  letI := induced F fData
  (fromInducedCoreMonoidal F fData).toMonoidal


/-- Transport a monoidal structure along an equivalence of (plain) categories.
-/
@[simps (config := .lemmasOnly)]
def transportStruct (e : C ≌ D) : MonoidalCategoryStruct.{v₂} D where
  tensorObj X Y := e.functor.obj (e.inverse.obj X ⊗ e.inverse.obj Y)
  whiskerLeft X _ _ f := e.functor.map (e.inverse.obj X ◁ e.inverse.map f)
  whiskerRight f X := e.functor.map (e.inverse.map f ▷ e.inverse.obj X)
  tensorHom f g := e.functor.map (e.inverse.map f ⊗ e.inverse.map g)
  tensorUnit := e.functor.obj (𝟙_ C)
  associator X Y Z :=
    e.functor.mapIso
      (whiskerRightIso (e.unitIso.app _).symm _ ≪≫
        α_ (e.inverse.obj X) (e.inverse.obj Y) (e.inverse.obj Z) ≪≫
        whiskerLeftIso _ (e.unitIso.app _))
  leftUnitor X :=
    e.functor.mapIso ((whiskerRightIso (e.unitIso.app _).symm _) ≪≫ λ_ (e.inverse.obj X)) ≪≫
      e.counitIso.app _
  rightUnitor X :=
    e.functor.mapIso ((whiskerLeftIso _ (e.unitIso.app _).symm) ≪≫ ρ_ (e.inverse.obj X)) ≪≫
      e.counitIso.app _


attribute [local simp] transportStruct in
/-- Transport a monoidal structure along an equivalence of (plain) categories.
-/
def transport (e : C ≌ D) : MonoidalCategory.{v₂} D :=
  letI : MonoidalCategoryStruct.{v₂} D := transportStruct e
  induced e.inverse
    { μIso := fun _ _ => e.unitIso.app _
      εIso := e.unitIso.app _ }


/-- A type synonym for `D`, which will carry the transported monoidal structure. -/
@[nolint unusedArguments]
def Transported (_ : C ≌ D) := D


instance (e : C ≌ D) : Category (Transported e) := (inferInstance : Category D)


instance Transported.instMonoidalCategoryStruct (e : C ≌ D) :
    MonoidalCategoryStruct (Transported e) :=
  transportStruct e


instance Transported.instMonoidalCategory (e : C ≌ D) : MonoidalCategory (Transported e) :=
  transport e


instance (e : C ≌ D) : Inhabited (Transported e) :=
  ⟨𝟙_ _⟩


/-- We upgrade the equivalence of categories `e : C ≌ D` to a monoidal category
equivalence `C ≌ Transported e`. -/
abbrev equivalenceTransported : C ≌ Transported e := e


instance : (equivalenceTransported e).inverse.Monoidal := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    ⊢ (CategoryTheory.Monoidal.equivalenceTransported e).inverse.Monoidal
  -/
  dsimp only [Transported.instMonoidalCategory]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    ⊢ (CategoryTheory.Monoidal.equivalenceTransported e).inverse.Monoidal
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : (equivalenceTransported e).symm.functor.Monoidal :=
  inferInstanceAs (equivalenceTransported e).inverse.Monoidal


instance : (equivalenceTransported e).functor.Monoidal :=
  (equivalenceTransported e).symm.inverseMonoidal


instance : (equivalenceTransported e).symm.inverse.Monoidal :=
  inferInstanceAs (equivalenceTransported e).functor.Monoidal


instance : (equivalenceTransported e).symm.IsMonoidal := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    ⊢ (CategoryTheory.Monoidal.equivalenceTransported e).symm.IsMonoidal
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The unit isomorphism upgrades to a monoidal isomorphism. -/
instance : NatTrans.IsMonoidal (equivalenceTransported e).unit :=
  inferInstanceAs (NatTrans.IsMonoidal (equivalenceTransported e).symm.counitIso.inv)


/-- The counit isomorphism upgrades to a monoidal isomorphism. -/
instance : NatTrans.IsMonoidal (equivalenceTransported e).counit :=
  inferInstanceAs (NatTrans.IsMonoidal (equivalenceTransported e).symm.unitIso.inv)


