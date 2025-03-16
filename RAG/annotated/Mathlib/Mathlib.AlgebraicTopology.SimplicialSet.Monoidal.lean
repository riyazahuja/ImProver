noncomputable instance : ChosenFiniteProducts SSet.{u} :=
  (inferInstance : ChosenFiniteProducts (SimplexCategoryᵒᵖ ⥤ Type u))


@[simp]
lemma leftUnitor_hom_app_apply (K : SSet.{u}) {Δ : SimplexCategoryᵒᵖ} (x : (𝟙_ _ ⊗ K).obj Δ) :
    (λ_ K).hom.app Δ x = x.2 := rfl


@[simp]
lemma leftUnitor_inv_app_apply (K : SSet.{u}) {Δ : SimplexCategoryᵒᵖ} (x : K.obj Δ) :
    (λ_ K).inv.app Δ x = ⟨PUnit.unit, x⟩ := rfl


@[simp]
lemma rightUnitor_hom_app_apply (K : SSet.{u}) {Δ : SimplexCategoryᵒᵖ} (x : (K ⊗ 𝟙_ _).obj Δ) :
    (ρ_ K).hom.app Δ x = x.1 := rfl


@[simp]
lemma rightUnitor_inv_app_apply (K : SSet.{u}) {Δ : SimplexCategoryᵒᵖ} (x : K.obj Δ) :
    (ρ_ K).inv.app Δ x = ⟨x, PUnit.unit⟩ := rfl


@[simp]
lemma tensorHom_app_apply {K K' L L' : SSet.{u}} (f : K ⟶ K') (g : L ⟶ L')
    {Δ : SimplexCategoryᵒᵖ} (x : (K ⊗ L).obj Δ) :
    (f ⊗ g).app Δ x = ⟨f.app Δ x.1, g.app Δ x.2⟩ := rfl


@[simp]
lemma whiskerLeft_app_apply (K : SSet.{u}) {L L' : SSet.{u}} (g : L ⟶ L')
    {Δ : SimplexCategoryᵒᵖ} (x : (K ⊗ L).obj Δ) :
    (K ◁ g).app Δ x = ⟨x.1, g.app Δ x.2⟩ := rfl


@[simp]
lemma whiskerRight_app_apply {K K' : SSet.{u}} (f : K ⟶ K') (L : SSet.{u})
    {Δ : SimplexCategoryᵒᵖ} (x : (K ⊗ L).obj Δ) :
    (f ▷ L).app Δ x = ⟨f.app Δ x.1, x.2⟩ := rfl


@[simp]
lemma associator_hom_app_apply (K L M : SSet.{u}) {Δ : SimplexCategoryᵒᵖ}
    (x : ((K ⊗ L) ⊗ M).obj Δ) :
    (α_ K L M).hom.app Δ x = ⟨x.1.1, x.1.2, x.2⟩ := rfl


@[simp]
lemma associator_inv_app_apply (K L M : SSet.{u}) {Δ : SimplexCategoryᵒᵖ}
    (x : (K ⊗ L ⊗ M).obj Δ) :
    (α_ K L M).inv.app Δ x = ⟨⟨x.1, x.2.1⟩, x.2.2⟩ := rfl


/-- The bijection `(𝟙_ SSet ⟶ K) ≃ K _[0]`. -/
def unitHomEquiv (K : SSet.{u}) : (𝟙_ _ ⟶ K) ≃ K _[0] where
  toFun φ := φ.app _ PUnit.unit
  invFun x :=
    { app := fun Δ _ => K.map (SimplexCategory.const Δ.unop [0] 0).op x
      naturality := fun Δ Δ' f => by
        /-
          K : SSet
          x : K.obj { unop := SimplexCategory.mk 0 }
          Δ Δ' : Opposite SimplexCategory
          f : Quiver.Hom Δ Δ'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        ext ⟨⟩
        /-
          case h.unit
          K : SSet
          x : K.obj { unop := SimplexCategory.mk 0 }
          Δ Δ' : Opposite SimplexCategory
          f : Quiver.Hom Δ Δ'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        dsimp
        /-
          case h.unit
          K : SSet
          x : K.obj { unop := SimplexCategory.mk 0 }
          Δ Δ' : Opposite SimplexCategory
          f : Quiver.Hom Δ Δ'
          ⊢ Eq (K.map ((Opposite.unop Δ').const (SimplexCategory.mk 0) 0).op x) (K.map f …
        -/
        rw [← FunctorToTypes.map_comp_apply]
        /-
          case h.unit
          K : SSet
          x : K.obj { unop := SimplexCategory.mk 0 }
          Δ Δ' : Opposite SimplexCategory
          f : Quiver.Hom Δ Δ'
          ⊢ Eq (K.map ((Opposite.unop Δ').const (SimplexCategory.mk 0) 0).op x) (K.map ( …
        -/
        rfl }
        /-
          🎉 no goals
        -/
  left_inv φ := by
    /-
      K : SSet
      φ : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit K
      ⊢ Eq ((fun x => { app := fun Δ x_1 => K.map ((Opposite.unop Δ).const (SimplexC …
    -/
    ext Δ ⟨⟩
    /-
      case w.h.unit
      K : SSet
      φ : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit K
      Δ : Opposite SimplexCategory
      ⊢ Eq (((fun x => { app := fun Δ x_1 => K.map ((Opposite.unop Δ).const (Simplex …
    -/
    dsimp
    /-
      case w.h.unit
      K : SSet
      φ : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit K
      Δ : Opposite SimplexCategory
      ⊢ Eq (K.map ((Opposite.unop Δ).const (SimplexCategory.mk 0) 0).op (φ.app { uno …
    -/
    rw [← FunctorToTypes.naturality]
    /-
      case w.h.unit
      K : SSet
      φ : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit K
      Δ : Opposite SimplexCategory
      ⊢ Eq (φ.app Δ (CategoryTheory.MonoidalCategoryStruct.tensorUnit.map ((Opposite …
    -/
    rfl
    /-
      🎉 no goals
    -/
                    /-
                      K : SSet
                      x : K.obj { unop := SimplexCategory.mk 0 }
                      ⊢ Eq ((fun φ => φ.app { unop := SimplexCategory.mk 0 } PUnit.unit) ((fun x =>  …
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


