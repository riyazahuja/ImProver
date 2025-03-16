/-- Given two functors `F₁` and `F₂` from a category `J` to a `V`-enriched
ordinary category `C`, this is the diagram `Jᵒᵖ ⥤ J ⥤ V` whose end shall be
the `V`-morphisms in `J ⥤ V` from `F₁` to `F₂`. -/
@[simps!]
def diagram : Jᵒᵖ ⥤ J ⥤ V := F₁.op ⋙ eHomFunctor V C ⋙ (whiskeringLeft J C V).obj F₂


/-- The condition that the end `diagram V F₁ F₂` exists, see `enrichedHom`. -/
abbrev HasEnrichedHom := HasEnd (diagram V F₁ F₂)


/-- The `V`-enriched hom from `F₁` to `F₂` when `F₁` and `F₂` are functors `J ⥤ C`
and `C` is a `V`-enriched category. -/
noncomputable abbrev enrichedHom : V := end_ (diagram V F₁ F₂)


/-- The projection `enrichedHom V F₁ F₂ ⟶ F₁.obj j ⟶[V] F₂.obj j` in the category `V`
for any `j : J` when `F₁` and `F₂` are functors `J ⥤ C` and `C` is a `V`-enriched category. -/
noncomputable abbrev enrichedHomπ (j : J) : enrichedHom V F₁ F₂ ⟶ F₁.obj j ⟶[V] F₂.obj j :=
  end_.π _ j


@[reassoc]
lemma enrichedHom_condition {i j : J} (f : i ⟶ j) :
    enrichedHomπ V F₁ F₂ i ≫ (ρ_ _).inv ≫
      _ ◁ (eHomEquiv V) (F₂.map f) ≫ eComp V _ _ _  =
    enrichedHomπ V F₁ F₂ j ≫ (λ_ _).inv ≫
      (eHomEquiv V) (F₁.map f) ▷ _ ≫ eComp V _ _ _ :=
  end_.condition (diagram V F₁ F₂) f


/-- Given functors `F₁` and `F₂` in `J ⥤ C`, where `C` is a `V`-enriched ordinary category,
this is the isomorphism `(F₁ ⟶ F₂) ≃ (𝟙_ V ⟶ enrichedHom V F₁ F₂)` in the category `V`. -/
noncomputable def homEquiv : (F₁ ⟶ F₂) ≃ (𝟙_ V ⟶ enrichedHom V F₁ F₂) where
  toFun τ := end_.lift (fun j ↦ eHomEquiv V (τ.app j)) (fun i j f ↦ by
    /-
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      τ : Quiver.Hom F₁ F₂
      i j : J
      f : Quiver.Hom i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => (CategoryTheory.eHomEquiv  …
    -/
    trans eHomEquiv V (τ.app i ≫ F₂.map f)
      /-
        V : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
        inst✝⁵ : CategoryTheory.MonoidalCategory V
        C : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
        J : Type u₃
        inst✝³ : CategoryTheory.Category.{v₃, u₃} J
        K : Type u₄
        inst✝² : CategoryTheory.Category.{v₄, u₄} K
        inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
        F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
        inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
        τ : Quiver.Hom F₁ F₂
        i j : J
        f : Quiver.Hom i j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => (CategoryTheory.eHomEquiv  …
      -/
    · dsimp
      simp only [eHomEquiv_comp, tensorHom_def_assoc, MonoidalCategory.whiskerRight_id,
        ← unitors_equal, assoc, Iso.inv_hom_id_assoc, eHomWhiskerLeft]
      /-
        V : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
        inst✝⁵ : CategoryTheory.MonoidalCategory V
        C : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
        J : Type u₃
        inst✝³ : CategoryTheory.Category.{v₃, u₃} J
        K : Type u₄
        inst✝² : CategoryTheory.Category.{v₄, u₄} K
        inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
        F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
        inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
        τ : Quiver.Hom F₁ F₂
        i j : J
        f : Quiver.Hom i j
        ⊢ Eq ((CategoryTheory.eHomEquiv V) (CategoryTheory.CategoryStruct.comp (τ.app  …
      -/
    · dsimp
      simp only [← NatTrans.naturality, eHomEquiv_comp, tensorHom_def', id_whiskerLeft,
        assoc, Iso.inv_hom_id_assoc, eHomWhiskerRight])
  invFun g :=
    { app := fun j ↦ (eHomEquiv V).symm (g ≫ end_.π _ j)
      naturality := fun i j f ↦ (eHomEquiv V).injective (by
        /-
          V : Type u₁
          inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁵ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝³ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝² : CategoryTheory.Category.{v₄, u₄} K
          inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          g : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (CategoryTheor …
          i j : J
          f : Quiver.Hom i j
          ⊢ Eq ((CategoryTheory.eHomEquiv V) (CategoryTheory.CategoryStruct.comp (F₁.map …
        -/
        dsimp
        /-
          V : Type u₁
          inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁵ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝³ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝² : CategoryTheory.Category.{v₄, u₄} K
          inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          g : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (CategoryTheor …
          i j : J
          f : Quiver.Hom i j
          ⊢ Eq ((CategoryTheory.eHomEquiv V) (CategoryTheory.CategoryStruct.comp (F₁.map …
        -/
        simp only [eHomEquiv_comp, Equiv.apply_symm_apply, Iso.cancel_iso_inv_left]
        conv_rhs =>
          rw [tensorHom_def_assoc, MonoidalCategory.whiskerRight_id_assoc, assoc,
            enrichedHom_condition V F₁ F₂ f]
        conv_lhs =>
          rw [tensorHom_def'_assoc, MonoidalCategory.whiskerLeft_comp_assoc,
            id_whiskerLeft_assoc, id_whiskerLeft_assoc, Iso.inv_hom_id_assoc, unitors_equal]) }
                   /-
                     V : Type u₁
                     inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
                     inst✝⁵ : CategoryTheory.MonoidalCategory V
                     C : Type u₂
                     inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
                     J : Type u₃
                     inst✝³ : CategoryTheory.Category.{v₃, u₃} J
                     K : Type u₄
                     inst✝² : CategoryTheory.Category.{v₄, u₄} K
                     inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
                     F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
                     inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
                     τ : Quiver.Hom F₁ F₂
                     ⊢ Eq ((fun g => { app := fun j => (CategoryTheory.eHomEquiv V).symm (CategoryT …
                   -/
  left_inv τ := by aesop
                   /-
                     🎉 no goals
                   -/
                    /-
                      V : Type u₁
                      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
                      inst✝⁵ : CategoryTheory.MonoidalCategory V
                      C : Type u₂
                      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
                      J : Type u₃
                      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
                      K : Type u₄
                      inst✝² : CategoryTheory.Category.{v₄, u₄} K
                      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
                      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
                      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
                      g : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (CategoryTheor …
                      ⊢ Eq ((fun τ => CategoryTheory.Limits.end_.lift (fun j => (CategoryTheory.eHom …
                    -/
  right_inv g := by aesop
                    /-
                      🎉 no goals
                    -/


@[reassoc (attr := simp)]
lemma homEquiv_apply_π (τ : F₁ ⟶ F₂) (j : J) :
    homEquiv V τ ≫ enrichedHomπ V _ _ j = eHomEquiv V (τ.app j) := by
  /-
    V : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁴ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} J
    inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    τ : Quiver.Hom F₁ F₂
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Enriched.FunctorCate …
  -/
  simp [homEquiv]
  /-
    🎉 no goals
  -/


/-- The identity for the `V`-enrichment of the category `J ⥤ C` over `V`. -/
noncomputable def enrichedId : 𝟙_ V ⟶ enrichedHom V F₁ F₁ := homEquiv _ (𝟙 F₁)


@[reassoc (attr := simp)]
lemma enrichedId_π (j : J) : enrichedId V F₁ ≫ end_.π _ j = eId V (F₁.obj j) := by
  /-
    V : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁴ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} J
    inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₁
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
  -/
  simp [enrichedId]
  /-
    🎉 no goals
  -/


@[simp]
lemma homEquiv_id : homEquiv V (𝟙 F₁) = enrichedId V F₁ := rfl


/-- The composition for the `V`-enrichment of the category `J ⥤ C` over `V`. -/
noncomputable def enrichedComp : enrichedHom V F₁ F₂ ⊗ enrichedHom V F₂ F₃ ⟶ enrichedHom V F₁ F₃ :=
  end_.lift (fun j ↦ (end_.π _ j ⊗ end_.π _ j) ≫ eComp V _ _ _) (fun i j f ↦ by
    /-
      V : Type u₁
      inst✝⁸ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁷ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝⁴ : CategoryTheory.Category.{v₄, u₄} K
      inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
      i j : J
      f : Quiver.Hom i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => CategoryTheory.CategoryStr …
    -/
    dsimp
    trans (end_.π (diagram V F₁ F₂) i ⊗ end_.π (diagram V F₂ F₃) j) ≫
      (ρ_ _).inv ▷ _ ≫ (_ ◁ (eHomEquiv V (F₂.map f))) ▷ _ ≫ eComp V _ (F₂.obj i) _ ▷ _ ≫
        eComp V _ (F₂.obj j) _
      /-
        V : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} V
        inst✝⁷ : CategoryTheory.MonoidalCategory V
        C : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C
        J : Type u₃
        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} J
        K : Type u₄
        inst✝⁴ : CategoryTheory.Category.{v₄, u₄} K
        inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
        F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
        inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
        inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
        inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
        i j : J
        f : Quiver.Hom i j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · have := end_.condition (diagram V F₂ F₃) f
      /-
        V : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} V
        inst✝⁷ : CategoryTheory.MonoidalCategory V
        C : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C
        J : Type u₃
        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} J
        K : Type u₄
        inst✝⁴ : CategoryTheory.Category.{v₄, u₄} K
        inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
        F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
        inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
        inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
        inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
        i j : J
        f : Quiver.Hom i j
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π (C …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp [eHomWhiskerLeft, eHomWhiskerRight] at this ⊢
      /-
        V : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} V
        inst✝⁷ : CategoryTheory.MonoidalCategory V
        C : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C
        J : Type u₃
        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} J
        K : Type u₄
        inst✝⁴ : CategoryTheory.Category.{v₄, u₄} K
        inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
        F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
        inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
        inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
        inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
        i j : J
        f : Quiver.Hom i j
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π (C …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      conv_lhs => rw [assoc, tensorHom_def_assoc]
      conv_rhs =>
        rw [tensorHom_def_assoc, whisker_assoc_assoc, e_assoc,
          triangle_assoc_comp_right_inv_assoc, ← MonoidalCategory.whiskerLeft_comp_assoc,
          ← MonoidalCategory.whiskerLeft_comp_assoc, ← MonoidalCategory.whiskerLeft_comp_assoc,
          assoc, assoc, ← this, MonoidalCategory.whiskerLeft_comp_assoc,
          MonoidalCategory.whiskerLeft_comp_assoc, MonoidalCategory.whiskerLeft_comp_assoc,
          ← e_assoc, whiskerLeft_rightUnitor_inv_assoc, associator_inv_naturality_right_assoc,
          Iso.hom_inv_id_assoc, whisker_exchange_assoc, MonoidalCategory.whiskerRight_id_assoc,
          Iso.inv_hom_id_assoc]
      /-
        V : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} V
        inst✝⁷ : CategoryTheory.MonoidalCategory V
        C : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C
        J : Type u₃
        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} J
        K : Type u₄
        inst✝⁴ : CategoryTheory.Category.{v₄, u₄} K
        inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
        F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
        inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
        inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
        inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
        i j : J
        f : Quiver.Hom i j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
    · have := end_.condition (diagram V F₁ F₂) f
      /-
        V : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} V
        inst✝⁷ : CategoryTheory.MonoidalCategory V
        C : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C
        J : Type u₃
        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} J
        K : Type u₄
        inst✝⁴ : CategoryTheory.Category.{v₄, u₄} K
        inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
        F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
        inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
        inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
        inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
        i j : J
        f : Quiver.Hom i j
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π (C …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp [eHomWhiskerLeft, eHomWhiskerRight] at this ⊢
      conv_lhs =>
        rw [tensorHom_def'_assoc, ← comp_whiskerRight_assoc,
          ← comp_whiskerRight_assoc, ← comp_whiskerRight_assoc,
          assoc, assoc, this, comp_whiskerRight_assoc, comp_whiskerRight_assoc,
          comp_whiskerRight_assoc, leftUnitor_inv_whiskerRight_assoc,
          ← associator_inv_naturality_left_assoc, ← e_assoc',
          Iso.inv_hom_id_assoc, ← whisker_exchange_assoc, id_whiskerLeft_assoc,
          Iso.inv_hom_id_assoc]
      /-
        V : Type u₁
        inst✝⁸ : CategoryTheory.Category.{v₁, u₁} V
        inst✝⁷ : CategoryTheory.MonoidalCategory V
        C : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C
        J : Type u₃
        inst✝⁵ : CategoryTheory.Category.{v₃, u₃} J
        K : Type u₄
        inst✝⁴ : CategoryTheory.Category.{v₄, u₄} K
        inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
        F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
        inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
        inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
        inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
        i j : J
        f : Quiver.Hom i j
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π (C …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      conv_rhs => rw [assoc, tensorHom_def'_assoc])
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
lemma enrichedComp_π (j : J) :
    enrichedComp V F₁ F₂ F₃ ≫ end_.π _ j =
      (end_.π (diagram V F₁ F₂) j ⊗ end_.π (diagram V F₂ F₃) j) ≫ eComp V _ _ _ := by
  /-
    V : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁶ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
    inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ F₃ : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
  -/
  simp [enrichedComp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homEquiv_comp (f : F₁ ⟶ F₂) (g : F₂ ⟶ F₃) :
    (homEquiv V) (f ≫ g) = (λ_ (𝟙_ V)).inv ≫ ((homEquiv V) f ⊗ (homEquiv V) g) ≫
    enrichedComp V F₁ F₂ F₃ := by
  /-
    V : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁶ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
    inst✝³ : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ F₃ : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
    f : Quiver.Hom F₁ F₂
    g : Quiver.Hom F₂ F₃
    ⊢ Eq ((CategoryTheory.Enriched.FunctorCategory.homEquiv V) (CategoryTheory.Cat …
  -/
  ext j
  simp only [homEquiv_apply_π, NatTrans.comp_app, eHomEquiv_comp, assoc,
    enrichedComp_π, Functor.op_obj, ← tensor_comp_assoc]


@[reassoc (attr := simp)]
lemma enriched_id_comp [HasEnrichedHom V F₁ F₁] [HasEnrichedHom V F₁ F₂] :
    (λ_ (enrichedHom V F₁ F₂)).inv ≫ enrichedId V F₁ ▷ enrichedHom V F₁ F₂ ≫
      enrichedComp V F₁ F₁ F₂ = 𝟙 _ := by
  /-
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₁
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext j
  rw [assoc, assoc, enrichedComp_π, id_comp, tensorHom_def, assoc,
    ← comp_whiskerRight_assoc, enrichedId_π, ← whisker_exchange_assoc,
    id_whiskerLeft, assoc, assoc, Iso.inv_hom_id_assoc]
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₁
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π (Catego …
  -/
  dsimp
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₁
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π (Catego …
  -/
  rw [e_id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma enriched_comp_id [HasEnrichedHom V F₁ F₂] [HasEnrichedHom V F₂ F₂] :
    (ρ_ (enrichedHom V F₁ F₂)).inv ≫ enrichedHom V F₁ F₂ ◁ enrichedId V F₂ ≫
      enrichedComp V F₁ F₂ F₂ = 𝟙 _ := by
  /-
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext j
  rw [assoc, assoc, enrichedComp_π, id_comp, tensorHom_def', assoc,
    ← MonoidalCategory.whiskerLeft_comp_assoc, enrichedId_π,
    whisker_exchange_assoc, MonoidalCategory.whiskerRight_id, assoc, assoc,
    Iso.inv_hom_id_assoc]
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₂
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π (Catego …
  -/
  dsimp
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₂
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π (Catego …
  -/
  rw [e_comp_id, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma enriched_assoc [HasEnrichedHom V F₁ F₂] [HasEnrichedHom V F₁ F₃] [HasEnrichedHom V F₁ F₄]
    [HasEnrichedHom V F₂ F₃] [HasEnrichedHom V F₂ F₄] [HasEnrichedHom V F₃ F₄] :
    (α_ (enrichedHom V F₁ F₂) (enrichedHom V F₂ F₃) (enrichedHom V F₃ F₄)).inv ≫
      enrichedComp V F₁ F₂ F₃ ▷ enrichedHom V F₃ F₄ ≫ enrichedComp V F₁ F₃ F₄ =
      enrichedHom V F₁ F₂ ◁ enrichedComp V F₂ F₃ F₄ ≫ enrichedComp V F₁ F₂ F₄ := by
  /-
    V : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁹ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝⁷ : CategoryTheory.Category.{v₃, u₃} J
    inst✝⁶ : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
    inst✝⁵ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    inst✝⁴ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
    inst✝³ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₄
    inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₄
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₃ F₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext j
  conv_lhs =>
    rw [assoc, assoc, enrichedComp_π,
      tensorHom_def_assoc, ← comp_whiskerRight_assoc, enrichedComp_π,
      comp_whiskerRight_assoc, ← whisker_exchange_assoc,
      ← whisker_exchange_assoc, ← tensorHom_def'_assoc, ← associator_inv_naturality_assoc]
  conv_rhs =>
    rw [assoc, enrichedComp_π, tensorHom_def'_assoc, ← MonoidalCategory.whiskerLeft_comp_assoc,
      enrichedComp_π, MonoidalCategory.whiskerLeft_comp_assoc, whisker_exchange_assoc,
      whisker_exchange_assoc, ← tensorHom_def_assoc]
  /-
    case h
    V : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁹ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝⁷ : CategoryTheory.Category.{v₃, u₃} J
    inst✝⁶ : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
    inst✝⁵ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    inst✝⁴ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
    inst✝³ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₄
    inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₄
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₃ F₄
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp
  /-
    case h
    V : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁹ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝⁷ : CategoryTheory.Category.{v₃, u₃} J
    inst✝⁶ : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
    inst✝⁵ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    inst✝⁴ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₃
    inst✝³ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₄
    inst✝² : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₃
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₂ F₄
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₃ F₄
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [e_assoc]
  /-
    🎉 no goals
  -/


/-- If `C` is a `V`-enriched ordinary category, and `C` has suitable limits,
then `J ⥤ C` is also a `V`-enriched ordinary category. -/
noncomputable def enrichedOrdinaryCategory [∀ (F₁ F₂ : J ⥤ C), HasEnrichedHom V F₁ F₂] :
    EnrichedOrdinaryCategory V (J ⥤ C) where
  Hom F₁ F₂ := enrichedHom V F₁ F₂
  id F := enrichedId V F
  comp F₁ F₂ F₃ := enrichedComp V F₁ F₂ F₃
  assoc _ _ _ _ := enriched_assoc _ _ _ _ _
  homEquiv := homEquiv V
  homEquiv_id _ := homEquiv_id V _
  homEquiv_comp f g := homEquiv_comp V f g


/-- If `F₁` and `F₂` are functors `J ⥤ C`, and `G : K ⥤ J`,
then this is the induced morphism
`enrichedHom V F₁ F₂ ⟶ enrichedHom V (G ⋙ F₁) (G ⋙ F₂)` in `V`
when `C` is a category enriched in `V`. -/
noncomputable abbrev precompEnrichedHom :
    enrichedHom V F₁ F₂ ⟶ enrichedHom V (G ⋙ F₁) (G ⋙ F₂) :=
  end_.lift (fun x ↦ enrichedHomπ V F₁ F₂ (G.obj x))
    (fun _ _ f ↦ enrichedHom_condition V F₁ F₂ (G.map f))


/-- Given functors `F₁` and `F₂` in `J ⥤ C`, where `C` is a category enriched in `V`,
this condition allows the definition of `functorEnrichedHom V F₁ F₂ : J ⥤ V`. -/
abbrev HasFunctorEnrichedHom :=
  ∀ (j : J), HasEnrichedHom V (Under.forget j ⋙ F₁) (Under.forget j ⋙ F₂)


instance {j j' : J} (f : j ⟶ j') :
    HasEnrichedHom V (Under.map f ⋙ Under.forget j ⋙ F₁)
      (Under.map f ⋙ Under.forget j ⋙ F₂) :=
  inferInstanceAs (HasEnrichedHom V (Under.forget j' ⋙ F₁) (Under.forget j' ⋙ F₂))


/-- Given functors `F₁` and `F₂` in `J ⥤ C`, where `C` is a category enriched in `V`,
this is the enriched hom functor from `F₁` to `F₂` in `J ⥤ V`. -/
@[simps!]
noncomputable def functorEnrichedHom : J ⥤ V where
  obj j := enrichedHom V (Under.forget j ⋙ F₁) (Under.forget j ⋙ F₂)
  map f := precompEnrichedHom V (Under.forget _ ⋙ F₁) (Under.forget _ ⋙ F₂) (Under.map f)
  map_id X := by
    /-
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X : J
      ⊢ Eq ({ obj := fun j => CategoryTheory.Enriched.FunctorCategory.enrichedHom V  …
    -/
    dsimp
    /-
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X : J
      ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.precompEnrichedHom V ((CategoryT …
    -/
    ext j
    /-
      case h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X : J
      j : CategoryTheory.Under X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
    -/
    dsimp
    /-
      case h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X : J
      j : CategoryTheory.Under X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
    -/
    simp only [end_.lift_π, id_comp]
    /-
      case h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X : J
      j : CategoryTheory.Under X
      ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.enrichedHomπ V ((CategoryTheory. …
    -/
    congr 1
    /-
      case h.h.e_12.h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X : J
      j : CategoryTheory.Under X
      ⊢ Eq ((CategoryTheory.Under.map (CategoryTheory.CategoryStruct.id X)).obj j) j
    -/
    simp [Under.map, Comma.mapLeft]
    /-
      case h.h.e_12.h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X : J
      j : CategoryTheory.Under X
      ⊢ Eq { left := j.left, right := j.right, hom := j.hom } j
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X✝ Y✝ Z✝ : J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun j => CategoryTheory.Enriched.FunctorCategory.enrichedHom V  …
    -/
    dsimp
    /-
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X✝ Y✝ Z✝ : J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.precompEnrichedHom V ((CategoryT …
    -/
    ext j
    /-
      case h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X✝ Y✝ Z✝ : J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      j : CategoryTheory.Under Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
    -/
    rw [end_.lift_π, assoc]
    /-
      case h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X✝ Y✝ Z✝ : J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      j : CategoryTheory.Under Z✝
      ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.enrichedHomπ V ((CategoryTheory. …
    -/
    erw [end_.lift_π, end_.lift_π]
    /-
      case h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X✝ Y✝ Z✝ : J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      j : CategoryTheory.Under Z✝
      ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.enrichedHomπ V ((CategoryTheory. …
    -/
    congr 1
    /-
      case h.h.e_12.h
      V : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁵ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝² : CategoryTheory.Category.{v₄, u₄} K
      inst✝¹ : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      X✝ Y✝ Z✝ : J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      j : CategoryTheory.Under Z✝
      ⊢ Eq ((CategoryTheory.Under.map (CategoryTheory.CategoryStruct.comp f g)).obj  …
    -/
    simp [Under.map, Comma.mapLeft]
    /-
      🎉 no goals
    -/


/-- The (limit) cone expressing that the limit of `functorEnrichedHom V F₁ F₂`
is `enrichedHom V F₁ F₂`. -/
@[simps pt]
noncomputable def coneFunctorEnrichedHom : Cone (functorEnrichedHom V F₁ F₂) where
  pt := enrichedHom V F₁ F₂
  π :=
    { app := fun j ↦ precompEnrichedHom V F₁ F₂ (Under.forget j)
      naturality := fun j j' f ↦ by
        /-
          V : Type u₁
          inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁶ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝³ : CategoryTheory.Category.{v₄, u₄} K
          inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          V : Type u₁
          inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁶ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝³ : CategoryTheory.Category.{v₄, u₄} K
          inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
        -/
        rw [id_comp]
        /-
          V : Type u₁
          inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁶ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝³ : CategoryTheory.Category.{v₄, u₄} K
          inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.precompEnrichedHom V F₁ F₂ (Cate …
        -/
        ext k
        /-
          case h
          V : Type u₁
          inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁶ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝³ : CategoryTheory.Category.{v₄, u₄} K
          inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          j j' : J
          f : Quiver.Hom j j'
          k : CategoryTheory.Under j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
        -/
        rw [assoc, end_.lift_π]
        /-
          case h
          V : Type u₁
          inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁶ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝³ : CategoryTheory.Category.{v₄, u₄} K
          inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          j j' : J
          f : Quiver.Hom j j'
          k : CategoryTheory.Under j'
          ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.enrichedHomπ V F₁ F₂ ((CategoryT …
        -/
        erw [end_.lift_π]
        /-
          case h
          V : Type u₁
          inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁶ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝³ : CategoryTheory.Category.{v₄, u₄} K
          inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          j j' : J
          f : Quiver.Hom j j'
          k : CategoryTheory.Under j'
          ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.enrichedHomπ V F₁ F₂ ((CategoryT …
        -/
        rw [end_.lift_π]
        /-
          case h
          V : Type u₁
          inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
          inst✝⁶ : CategoryTheory.MonoidalCategory V
          C : Type u₂
          inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
          J : Type u₃
          inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
          K : Type u₄
          inst✝³ : CategoryTheory.Category.{v₄, u₄} K
          inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
          F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
          inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
          inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
          j j' : J
          f : Quiver.Hom j j'
          k : CategoryTheory.Under j'
          ⊢ Eq (CategoryTheory.Enriched.FunctorCategory.enrichedHomπ V F₁ F₂ ((CategoryT …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- Auxiliary definition for `Enriched.FunctorCategory.isLimitConeFunctorEnrichedHom`. -/
noncomputable def lift : s.pt ⟶ enrichedHom V F₁ F₂ :=
  end_.lift (fun j ↦ s.π.app j ≫ enrichedHomπ V _ _ (Under.mk (𝟙 j))) (fun j j' f ↦ by
    /-
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => CategoryTheory.CategoryStr …
    -/
    dsimp
    /-
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [← s.w f, assoc, assoc, assoc]
    /-
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j) (CategoryTheory.CategoryS …
    -/
    dsimp [functorEnrichedHom]
    erw [end_.lift_π_assoc,
      enrichedHom_condition V (Under.forget j ⋙ F₁) (Under.forget j ⋙ F₂)
      (Under.homMk f : Under.mk (𝟙 j) ⟶ Under.mk f)]
    /-
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j) (CategoryTheory.CategoryS …
    -/
    congr 3
    /-
      case e_a.e_a.h.e_12.h
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.Under.mk f) ((CategoryTheory.Under.map f).obj (CategoryTh …
    -/
    simp [Under.map, Comma.mapLeft]
    /-
      case e_a.e_a.h.e_12.h
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.Under.mk f) { left := (CategoryTheory.Under.mk (CategoryT …
    -/
    rfl)
    /-
      🎉 no goals
    -/


lemma fac (j : J) : lift s ≫ (coneFunctorEnrichedHom V F₁ F₂).π.app j = s.π.app j := by
  /-
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
  -/
  dsimp [coneFunctorEnrichedHom]
  /-
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
  -/
  ext k
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    k : CategoryTheory.Under j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc]
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    k : CategoryTheory.Under j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Enriched.FunctorCateg …
  -/
  erw [end_.lift_π, end_.lift_π, ← s.w k.hom]
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    k : CategoryTheory.Under j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc]
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    k : CategoryTheory.Under j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app ((CategoryTheory.Functor.fro …
  -/
  dsimp
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    k : CategoryTheory.Under j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j) (CategoryTheory.CategoryS …
  -/
  erw [end_.lift_π]
  /-
    case h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    k : CategoryTheory.Under j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j) (CategoryTheory.Enriched. …
  -/
  congr
  /-
    case h.e_a.h.e_12.h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    k : CategoryTheory.Under j
    ⊢ Eq ((CategoryTheory.Under.map k.hom).obj (CategoryTheory.Under.mk (CategoryT …
  -/
  simp [Under.map, Comma.mapLeft]
  /-
    case h.e_a.h.e_12.h
    V : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} V
    inst✝⁵ : CategoryTheory.MonoidalCategory V
    C : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C
    J : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} J
    inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
    F₁ F₂ : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
    inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
    s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
    j : J
    k : CategoryTheory.Under j
    ⊢ Eq { left := (CategoryTheory.Under.mk (CategoryTheory.CategoryStruct.id k.ri …
  -/
  rfl
  /-
    🎉 no goals
  -/


open isLimitConeFunctorEnrichedHom in
/-- The limit of `functorEnrichedHom V F₁ F₂` is `enrichedHom V F₁ F₂`. -/
noncomputable def isLimitConeFunctorEnrichedHom :
    IsLimit (coneFunctorEnrichedHom V F₁ F₂) where
  lift := lift
  fac := fac
  uniq s m hm := by
    /-
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      m : Quiver.Hom s.pt (CategoryTheory.Enriched.FunctorCategory.coneFunctorEnrich …
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Enri …
      ⊢ Eq m (CategoryTheory.Enriched.FunctorCategory.isLimitConeFunctorEnrichedHom. …
    -/
    dsimp
    /-
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      m : Quiver.Hom s.pt (CategoryTheory.Enriched.FunctorCategory.coneFunctorEnrich …
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Enri …
      ⊢ Eq m (CategoryTheory.Enriched.FunctorCategory.isLimitConeFunctorEnrichedHom. …
    -/
    ext j
    /-
      case h
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      m : Quiver.Hom s.pt (CategoryTheory.Enriched.FunctorCategory.coneFunctorEnrich …
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Enri …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.end_.π (Cate …
    -/
    have := ((hm j).trans (fac s j).symm) =≫ enrichedHomπ V _ _ (Under.mk (𝟙 j))
    /-
      case h
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      m : Quiver.Hom s.pt (CategoryTheory.Enriched.FunctorCategory.coneFunctorEnrich …
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Enri …
      j : J
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.end_.π (Cate …
    -/
    dsimp [coneFunctorEnrichedHom] at this
    /-
      case h
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      m : Quiver.Hom s.pt (CategoryTheory.Enriched.FunctorCategory.coneFunctorEnrich …
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Enri …
      j : J
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.end_.π (Cate …
    -/
    rw [assoc, assoc, end_.lift_π] at this
    /-
      case h
      V : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} V
      inst✝⁶ : CategoryTheory.MonoidalCategory V
      C : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C
      J : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} J
      K : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} K
      inst✝² : CategoryTheory.EnrichedOrdinaryCategory V C
      F₁ F₂ F₃ F₄ : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Enriched.FunctorCategory.HasFunctorEnrichedHom V F₁ F₂
      inst✝ : CategoryTheory.Enriched.FunctorCategory.HasEnrichedHom V F₁ F₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Enriched.FunctorCategory.functo …
      m : Quiver.Hom s.pt (CategoryTheory.Enriched.FunctorCategory.coneFunctorEnrich …
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Enri …
      j : J
      this : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Enriched.Funct …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.end_.π (Cate …
    -/
    exact this
    /-
      🎉 no goals
    -/


