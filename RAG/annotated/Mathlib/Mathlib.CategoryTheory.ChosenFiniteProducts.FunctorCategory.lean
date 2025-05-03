/-- The chosen terminal object in `J ⥤ C`. -/
abbrev chosenTerminal : J ⥤ C := (Functor.const J).obj (𝟙_ C)


/-- The chosen terminal object in `J ⥤ C` is terminal. -/
def chosenTerminalIsTerminal : IsTerminal (chosenTerminal J C) :=
  evaluationJointlyReflectsLimits _
    (fun _ => isLimitChangeEmptyCone _ ChosenFiniteProducts.terminal.2 _ (Iso.refl _))


/-- The chosen binary product on `J ⥤ C`. -/
@[simps]
def chosenProd : J ⥤ C where
  obj j := F₁.obj j ⊗ F₂.obj j
  map φ := F₁.map φ ⊗ F₂.map φ


/-- The first projection `chosenProd F₁ F₂ ⟶ F₁`. -/
@[simps]
def fst : chosenProd F₁ F₂ ⟶ F₁ where
  app _ := ChosenFiniteProducts.fst _ _


/-- The second projection `chosenProd F₁ F₂ ⟶ F₂`. -/
@[simps]
def snd : chosenProd F₁ F₂ ⟶ F₂ where
  app _ := ChosenFiniteProducts.snd _ _


/-- `Functor.chosenProd F₁ F₂` is a binary product of `F₁` and `F₂`. -/
noncomputable def isLimit : IsLimit (BinaryFan.mk (fst F₁ F₂) (snd F₁ F₂)) :=
  evaluationJointlyReflectsLimits _ (fun j =>
                                                 /-
                                                   J : Type u_1
                                                   C : Type u_2
                                                   inst✝² : CategoryTheory.Category.{?u.9457, u_1} J
                                                   inst✝¹ : CategoryTheory.Category.{?u.9461, u_2} C
                                                   inst✝ : CategoryTheory.ChosenFiniteProducts C
                                                   F₁ F₂ : CategoryTheory.Functor J C
                                                   j : J
                                                   ⊢ CategoryTheory.Iso (((CategoryTheory.Limits.pair F₁ F₂).comp ((CategoryTheor …
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    (IsLimit.postcomposeHomEquiv (mapPairIso (by exact Iso.refl _) (by exact Iso.refl _)) _).1
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
      (IsLimit.ofIsoLimit (ChosenFiniteProducts.product (X := F₁.obj j) (Y := F₂.obj j)).2
                                    /-
                                      J : Type u_1
                                      C : Type u_2
                                      inst✝² : CategoryTheory.Category.{?u.9457, u_1} J
                                      inst✝¹ : CategoryTheory.Category.{?u.9461, u_2} C
                                      inst✝ : CategoryTheory.ChosenFiniteProducts C
                                      F₁ F₂ : CategoryTheory.Functor J C
                                      j : J
                                      ⊢ ∀ (j_1 : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq ((Ca …
                                    -/
        (Cones.ext (Iso.refl _) (by rintro ⟨_|_⟩; all_goals aesop_cat))))
                                                  /-
                                                    🎉 no goals
                                                  -/


noncomputable instance chosenFiniteProducts :
    ChosenFiniteProducts (J ⥤ C) where
  terminal := ⟨_, chosenTerminalIsTerminal J C⟩
  product F₁ F₂ := ⟨_, chosenProd.isLimit F₁ F₂⟩


@[simp]
lemma leftUnitor_hom_app (F : J ⥤ C) (j : J) :
    (λ_ F).hom.app j = (λ_ (F.obj j)).hom := rfl


@[simp]
lemma leftUnitor_inv_app (F : J ⥤ C) (j : J) :
    (λ_ F).inv.app j = (λ_ (F.obj j)).inv := by
  rw [← cancel_mono ((λ_ (F.obj j)).hom), Iso.inv_hom_id, ← leftUnitor_hom_app,
    Iso.inv_hom_id_app]


@[simp]
lemma rightUnitor_hom_app (F : J ⥤ C) (j : J) :
    (ρ_ F).hom.app j = (ρ_ (F.obj j)).hom := rfl


@[simp]
lemma rightUnitor_inv_app (F : J ⥤ C) (j : J) :
    (ρ_ F).inv.app j = (ρ_ (F.obj j)).inv := by
  rw [← cancel_mono ((ρ_ (F.obj j)).hom), Iso.inv_hom_id, ← rightUnitor_hom_app,
    Iso.inv_hom_id_app]


@[reassoc (attr := simp)]
lemma tensorHom_app_fst {F₁ F₁' F₂ F₂' : J ⥤ C} (f : F₁ ⟶ F₁') (g : F₂ ⟶ F₂') (j : J) :
    (f ⊗ g).app j ≫ fst _ _ = fst _ _ ≫ f.app j := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    F₁ F₁' F₂ F₂' : CategoryTheory.Functor J C
    f : Quiver.Hom F₁ F₁'
    g : Quiver.Hom F₂ F₂'
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
  -/
  change (f ⊗ g).app j ≫ (fst F₁' F₂').app j = _
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    F₁ F₁' F₂ F₂' : CategoryTheory.Functor J C
    f : Quiver.Hom F₁ F₁'
    g : Quiver.Hom F₂ F₂'
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
  -/
  rw [← NatTrans.comp_app, tensorHom_fst, NatTrans.comp_app]
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    F₁ F₁' F₂ F₂' : CategoryTheory.Functor J C
    f : Quiver.Hom F₁ F₁'
    g : Quiver.Hom F₂ F₂'
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ChosenFiniteProducts …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma tensorHom_app_snd {F₁ F₁' F₂ F₂' : J ⥤ C} (f : F₁ ⟶ F₁') (g : F₂ ⟶ F₂') (j : J) :
    (f ⊗ g).app j ≫ snd _ _ = snd _ _ ≫ g.app j := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    F₁ F₁' F₂ F₂' : CategoryTheory.Functor J C
    f : Quiver.Hom F₁ F₁'
    g : Quiver.Hom F₂ F₂'
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
  -/
  change (f ⊗ g).app j ≫ (snd F₁' F₂').app j = _
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    F₁ F₁' F₂ F₂' : CategoryTheory.Functor J C
    f : Quiver.Hom F₁ F₁'
    g : Quiver.Hom F₂ F₂'
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
  -/
  rw [← NatTrans.comp_app, tensorHom_snd, NatTrans.comp_app]
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    F₁ F₁' F₂ F₂' : CategoryTheory.Functor J C
    f : Quiver.Hom F₁ F₁'
    g : Quiver.Hom F₂ F₂'
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ChosenFiniteProducts …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma whiskerLeft_app_fst (F₁ : J ⥤ C) {F₂ F₂' : J ⥤ C} (g : F₂ ⟶ F₂') (j : J) :
    (F₁ ◁ g).app j ≫ fst _ _ = fst _ _ :=
                                           /-
                                             J : Type u_1
                                             C : Type u_2
                                             inst✝² : CategoryTheory.Category.{u_3, u_1} J
                                             inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
                                             inst✝ : CategoryTheory.ChosenFiniteProducts C
                                             F₁ F₂ F₂' : CategoryTheory.Functor J C
                                             g : Quiver.Hom F₂ F₂'
                                             j : J
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
                                           -/
  (tensorHom_app_fst (𝟙 F₁) g j).trans (by simp)
                                           /-
                                             🎉 no goals
                                           -/


@[reassoc (attr := simp)]
lemma whiskerLeft_app_snd (F₁ : J ⥤ C) {F₂ F₂' : J ⥤ C} (g : F₂ ⟶ F₂') (j : J) :
    (F₁ ◁ g).app j ≫ snd _ _ = snd _ _ ≫ g.app j :=
  (tensorHom_app_snd (𝟙 F₁) g j)


@[reassoc (attr := simp)]
lemma whiskerRight_app_fst {F₁ F₁' : J ⥤ C} (f : F₁ ⟶ F₁') (F₂ : J ⥤ C) (j : J) :
    (f ▷ F₂).app j ≫ fst _ _ = fst _ _ ≫ f.app j :=
  (tensorHom_app_fst f (𝟙 F₂) j)


@[reassoc (attr := simp)]
lemma whiskerRight_app_snd {F₁ F₁' : J ⥤ C} (f : F₁ ⟶ F₁') (F₂ : J ⥤ C) (j : J) :
    (f ▷ F₂).app j ≫ snd _ _ = snd _ _ :=
                                           /-
                                             J : Type u_1
                                             C : Type u_2
                                             inst✝² : CategoryTheory.Category.{u_3, u_1} J
                                             inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
                                             inst✝ : CategoryTheory.ChosenFiniteProducts C
                                             F₁ F₁' : CategoryTheory.Functor J C
                                             f : Quiver.Hom F₁ F₁'
                                             F₂ : CategoryTheory.Functor J C
                                             j : J
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
                                           -/
  (tensorHom_app_snd f (𝟙 F₂) j).trans (by simp)
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
lemma associator_hom_app (F₁ F₂ F₃ : J ⥤ C) (j : J) :
    (α_ F₁ F₂ F₃).hom.app j = (α_ _ _ _).hom := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    F₁ F₂ F₃ : CategoryTheory.Functor J C
    j : J
    ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.associator F₁ F₂ F₃).hom.app j) ( …
  -/
  apply hom_ext
    /-
      case h_fst
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
      inst✝ : CategoryTheory.ChosenFiniteProducts C
      F₁ F₂ F₃ : CategoryTheory.Functor J C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
    -/
  · change _ ≫ (fst F₁ (F₂ ⊗ F₃)).app j = _
    /-
      case h_fst
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
      inst✝ : CategoryTheory.ChosenFiniteProducts C
      F₁ F₂ F₃ : CategoryTheory.Functor J C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
    -/
    rw [← NatTrans.comp_app, associator_hom_fst]
    /-
      case h_fst
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
      inst✝ : CategoryTheory.ChosenFiniteProducts C
      F₁ F₂ F₃ : CategoryTheory.Functor J C
      j : J
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts …
    -/
    erw [associator_hom_fst]
    /-
      case h_fst
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
      inst✝ : CategoryTheory.ChosenFiniteProducts C
      F₁ F₂ F₃ : CategoryTheory.Functor J C
      j : J
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h_snd
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} J
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
      inst✝ : CategoryTheory.ChosenFiniteProducts C
      F₁ F₂ F₃ : CategoryTheory.Functor J C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
    -/
  · apply hom_ext
      /-
        case h_snd.h_fst
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_1} J
        inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
        inst✝ : CategoryTheory.ChosenFiniteProducts C
        F₁ F₂ F₃ : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · change (_ ≫ (snd F₁ (F₂ ⊗ F₃)).app j) ≫ (fst F₂ F₃).app j = _
      /-
        case h_snd.h_fst
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_1} J
        inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
        inst✝ : CategoryTheory.ChosenFiniteProducts C
        F₁ F₂ F₃ : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      rw [← NatTrans.comp_app, ← NatTrans.comp_app, assoc, associator_hom_snd_fst, assoc]
      /-
        case h_snd.h_fst
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_1} J
        inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
        inst✝ : CategoryTheory.ChosenFiniteProducts C
        F₁ F₂ F₃ : CategoryTheory.Functor J C
        j : J
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts …
      -/
      erw [associator_hom_snd_fst]
      /-
        case h_snd.h_fst
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_1} J
        inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
        inst✝ : CategoryTheory.ChosenFiniteProducts C
        F₁ F₂ F₃ : CategoryTheory.Functor J C
        j : J
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case h_snd.h_snd
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_1} J
        inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
        inst✝ : CategoryTheory.ChosenFiniteProducts C
        F₁ F₂ F₃ : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · change (_ ≫ (snd F₁ (F₂ ⊗ F₃)).app j) ≫ (snd F₂ F₃).app j = _
      /-
        case h_snd.h_snd
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_1} J
        inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
        inst✝ : CategoryTheory.ChosenFiniteProducts C
        F₁ F₂ F₃ : CategoryTheory.Functor J C
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      rw [← NatTrans.comp_app, ← NatTrans.comp_app, assoc, associator_hom_snd_snd, assoc]
      /-
        case h_snd.h_snd
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_1} J
        inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
        inst✝ : CategoryTheory.ChosenFiniteProducts C
        F₁ F₂ F₃ : CategoryTheory.Functor J C
        j : J
        ⊢ Eq ((CategoryTheory.ChosenFiniteProducts.snd (CategoryTheory.MonoidalCategor …
      -/
      erw [associator_hom_snd_snd]
      /-
        case h_snd.h_snd
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_1} J
        inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
        inst✝ : CategoryTheory.ChosenFiniteProducts C
        F₁ F₂ F₃ : CategoryTheory.Functor J C
        j : J
        ⊢ Eq ((CategoryTheory.ChosenFiniteProducts.snd (CategoryTheory.MonoidalCategor …
      -/
      rfl
      /-
        🎉 no goals
      -/


@[simp]
lemma associator_inv_app (F₁ F₂ F₃ : J ⥤ C) (j : J) :
    (α_ F₁ F₂ F₃).inv.app j = (α_ _ _ _).inv := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    F₁ F₂ F₃ : CategoryTheory.Functor J C
    j : J
    ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.associator F₁ F₂ F₃).inv.app j) ( …
  -/
  rw [← cancel_mono ((α_ _ _ _).hom), Iso.inv_hom_id, ← associator_hom_app, Iso.inv_hom_id_app]
  /-
    🎉 no goals
  -/


noncomputable instance {K : Type*} [Category K] [HasColimitsOfShape K C]
    [∀ X : C, PreservesColimitsOfShape K (tensorLeft X)] {F : J ⥤ C} :
    PreservesColimitsOfShape K (tensorLeft F) := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} J
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    K : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} K
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape K (CategoryT …
    F : CategoryTheory.Functor J C
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape K (CategoryTheory.MonoidalCat …
  -/
  apply preservesColimitsOfShape_of_evaluation
  /-
    case x
    J : Type u_1
    C : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} J
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    K : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} K
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape K (CategoryT …
    F : CategoryTheory.Functor J C
    ⊢ ∀ (k : J), CategoryTheory.Limits.PreservesColimitsOfShape K ((CategoryTheory …
  -/
  intro k
  haveI : tensorLeft F ⋙ (evaluation J C).obj k ≅ (evaluation J C).obj k ⋙ tensorLeft (F.obj k) :=
    NatIso.ofComponents (fun _ ↦ Iso.refl _)
  /-
    case x
    J : Type u_1
    C : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} J
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    K : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} K
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape K (CategoryT …
    F : CategoryTheory.Functor J C
    k : J
    this : CategoryTheory.Iso ((CategoryTheory.MonoidalCategory.tensorLeft F).comp …
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape K ((CategoryTheory.MonoidalCa …
  -/
  exact preservesColimitsOfShape_of_natIso this.symm
  /-
    🎉 no goals
  -/


