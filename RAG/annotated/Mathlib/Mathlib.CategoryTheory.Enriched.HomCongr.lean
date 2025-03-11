/-- Given isomorphisms `α : X ≅ X₁` and `β : Y ≅ Y₁` in `C`, we can construct
an isomorphism between `V` objects `X ⟶[V] Y` and `X₁ ⟶[V] Y₁`. -/
@[simps]
def eHomCongr {X Y X₁ Y₁ : C} (α : X ≅ X₁) (β : Y ≅ Y₁) :
    (X ⟶[V] Y) ≅ (X₁ ⟶[V] Y₁) where
  hom := eHomWhiskerRight V α.inv Y ≫ eHomWhiskerLeft V X₁ β.hom
  inv := eHomWhiskerRight V α.hom Y₁ ≫ eHomWhiskerLeft V X β.inv
  hom_inv_id := by
    /-
      V : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} V
      inst✝² : CategoryTheory.MonoidalCategory V
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
      X Y X₁ Y₁ : C
      α : CategoryTheory.Iso X X₁
      β : CategoryTheory.Iso Y Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [← eHom_whisker_exchange]
    /-
      V : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} V
      inst✝² : CategoryTheory.MonoidalCategory V
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
      X Y X₁ Y₁ : C
      α : CategoryTheory.Iso X X₁
      β : CategoryTheory.Iso Y Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    slice_lhs 2 3 => rw [← eHomWhiskerRight_comp]
    /-
      V : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} V
      inst✝² : CategoryTheory.MonoidalCategory V
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
      X Y X₁ Y₁ : C
      α : CategoryTheory.Iso X X₁
      β : CategoryTheory.Iso Y Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eHomWhiskerLeft V X β …
    -/
    simp [← eHomWhiskerLeft_comp]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      V : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} V
      inst✝² : CategoryTheory.MonoidalCategory V
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
      X Y X₁ Y₁ : C
      α : CategoryTheory.Iso X X₁
      β : CategoryTheory.Iso Y Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [← eHom_whisker_exchange]
    /-
      V : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} V
      inst✝² : CategoryTheory.MonoidalCategory V
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
      X Y X₁ Y₁ : C
      α : CategoryTheory.Iso X X₁
      β : CategoryTheory.Iso Y Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    slice_lhs 2 3 => rw [← eHomWhiskerRight_comp]
    /-
      V : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} V
      inst✝² : CategoryTheory.MonoidalCategory V
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
      X Y X₁ Y₁ : C
      α : CategoryTheory.Iso X X₁
      β : CategoryTheory.Iso Y Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eHomWhiskerLeft V X₁  …
    -/
    simp [← eHomWhiskerLeft_comp]
    /-
      🎉 no goals
    -/


lemma eHomCongr_refl (X Y : C) :
                                                                      /-
                                                                        V : Type u'
                                                                        inst✝³ : CategoryTheory.Category.{v', u'} V
                                                                        inst✝² : CategoryTheory.MonoidalCategory V
                                                                        C : Type u
                                                                        inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                        inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
                                                                        X Y : C
                                                                        ⊢ Eq (CategoryTheory.Iso.eHomCongr V (CategoryTheory.Iso.refl X) (CategoryTheo …
                                                                      -/
    eHomCongr V (Iso.refl X) (Iso.refl Y) = Iso.refl (X ⟶[V] Y) := by aesop
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma eHomCongr_trans {X₁ Y₁ X₂ Y₂ X₃ Y₃ : C} (α₁ : X₁ ≅ X₂) (β₁ : Y₁ ≅ Y₂)
    (α₂ : X₂ ≅ X₃) (β₂ : Y₂ ≅ Y₃) :
    eHomCongr V (α₁ ≪≫ α₂) (β₁ ≪≫ β₂) =
      eHomCongr V α₁ β₁ ≪≫ eHomCongr V α₂ β₂ := by
  /-
    V : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} V
    inst✝² : CategoryTheory.MonoidalCategory V
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.EnrichedOrdinaryCategory V C
    X₁ Y₁ X₂ Y₂ X₃ Y₃ : C
    α₁ : CategoryTheory.Iso X₁ X₂
    β₁ : CategoryTheory.Iso Y₁ Y₂
    α₂ : CategoryTheory.Iso X₂ X₃
    β₂ : CategoryTheory.Iso Y₂ Y₃
    ⊢ Eq (CategoryTheory.Iso.eHomCongr V (α₁.trans α₂) (β₁.trans β₂)) ((CategoryTh …
  -/
  ext; simp [eHom_whisker_exchange_assoc]
       /-
         🎉 no goals
       -/


lemma eHomCongr_symm {X Y X₁ Y₁ : C} (α : X ≅ X₁) (β : Y ≅ Y₁) :
    (eHomCongr V α β).symm = eHomCongr V α.symm β.symm := rfl


/-- `eHomCongr` respects composition of morphisms. Recall that for any
composable pair of arrows `f : X ⟶ Y` and `g : Y ⟶ Z` in `C`, the composite
`f ≫ g` in `C` defines a morphism `𝟙_ V ⟶ (X ⟶[V] Z)` in `V`. Composing with
the isomorphism `eHomCongr V α γ` yields a morphism in `V` that can be factored
through the enriched composition map as shown:
`𝟙_ V ⟶ 𝟙_ V ⊗ 𝟙_ V ⟶ (X₁ ⟶[V] Y₁) ⊗ (Y₁ ⟶[V] Z₁) ⟶ (X₁ ⟶[V] Z₁)`. -/
@[reassoc]
lemma eHomCongr_comp {X Y Z X₁ Y₁ Z₁ : C} (α : X ≅ X₁) (β : Y ≅ Y₁) (γ : Z ≅ Z₁)
    (f : X ⟶ Y) (g : Y ⟶ Z) :
    eHomEquiv V (f ≫ g) ≫ (eHomCongr V α γ).hom =
      (λ_ _).inv ≫ (eHomEquiv V f ≫ (eHomCongr V α β).hom) ▷ _ ≫
        _ ◁ (eHomEquiv V g ≫ (eHomCongr V β γ).hom) ≫ eComp V X₁ Y₁ Z₁ := by
  simp only [eHomCongr, MonoidalCategory.whiskerRight_id, assoc,
    MonoidalCategory.whiskerLeft_comp]
  rw [rightUnitor_inv_naturality_assoc, rightUnitor_inv_naturality_assoc,
    rightUnitor_inv_naturality_assoc, hom_inv_id_assoc, ← whisker_exchange_assoc,
    ← whisker_exchange_assoc, ← eComp_eHomWhiskerLeft, eHom_whisker_cancel_assoc,
    ← eComp_eHomWhiskerRight_assoc, ← tensorHom_def_assoc,
    ← eHomEquiv_comp_assoc]


/-- The inverse map defined by `eHomCongr` respects composition of morphisms. -/
@[reassoc]
lemma eHomCongr_inv_comp {X Y Z X₁ Y₁ Z₁ : C} (α : X ≅ X₁) (β : Y ≅ Y₁)
    (γ : Z ≅ Z₁) (f : X₁ ⟶ Y₁) (g : Y₁ ⟶ Z₁) :
    eHomEquiv V (f ≫ g) ≫ (eHomCongr V α γ).inv =
      (λ_ _).inv ≫ (eHomEquiv V f ≫ (eHomCongr V α β).inv) ▷ _ ≫
        _ ◁ (eHomEquiv V g ≫ (eHomCongr V β γ).inv) ≫ eComp V X Y Z :=
  eHomCongr_comp V α.symm β.symm γ.symm f g


