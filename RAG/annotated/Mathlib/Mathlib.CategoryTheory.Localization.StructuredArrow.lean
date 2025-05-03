/-- The bijection `StructuredArrow (L.obj X) L ≃ StructuredArrow (L'.obj X) L'`
when `L` and `L'` are two localization functors for the same class of morphisms. -/
@[simps]
noncomputable def structuredArrowEquiv :
    StructuredArrow (L.obj X) L ≃ StructuredArrow (L'.obj X) L' where
  toFun f := StructuredArrow.mk (homEquiv W L L' f.hom)
  invFun f := StructuredArrow.mk (homEquiv W L' L f.hom)
  left_inv f := by
    /-
      C : Type u_1
      D : Type u_2
      D' : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.208, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.212, u_2} D
      inst✝² : CategoryTheory.Category.{?u.216, u_3} D'
      W : CategoryTheory.MorphismProperty C
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      inst✝¹ : L.IsLocalization W
      inst✝ : L'.IsLocalization W
      X : C
      f : CategoryTheory.StructuredArrow (L.obj X) L
      ⊢ Eq ((fun f => CategoryTheory.StructuredArrow.mk ((CategoryTheory.Localizatio …
    -/
    obtain ⟨Y, f, rfl⟩ := f.mk_surjective
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      D' : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.208, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.212, u_2} D
      inst✝² : CategoryTheory.Category.{?u.216, u_3} D'
      W : CategoryTheory.MorphismProperty C
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      inst✝¹ : L.IsLocalization W
      inst✝ : L'.IsLocalization W
      X Y : C
      f : Quiver.Hom (L.obj X) (L.obj Y)
      ⊢ Eq ((fun f => CategoryTheory.StructuredArrow.mk ((CategoryTheory.Localizatio …
    -/
    dsimp
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      D' : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.208, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.212, u_2} D
      inst✝² : CategoryTheory.Category.{?u.216, u_3} D'
      W : CategoryTheory.MorphismProperty C
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      inst✝¹ : L.IsLocalization W
      inst✝ : L'.IsLocalization W
      X Y : C
      f : Quiver.Hom (L.obj X) (L.obj Y)
      ⊢ Eq (CategoryTheory.StructuredArrow.mk ((CategoryTheory.Localization.homEquiv …
    -/
    rw [← homEquiv_symm_apply, Equiv.symm_apply_apply]
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      C : Type u_1
      D : Type u_2
      D' : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.208, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.212, u_2} D
      inst✝² : CategoryTheory.Category.{?u.216, u_3} D'
      W : CategoryTheory.MorphismProperty C
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      inst✝¹ : L.IsLocalization W
      inst✝ : L'.IsLocalization W
      X : C
      f : CategoryTheory.StructuredArrow (L'.obj X) L'
      ⊢ Eq ((fun f => CategoryTheory.StructuredArrow.mk ((CategoryTheory.Localizatio …
    -/
    obtain ⟨Y, f, rfl⟩ := f.mk_surjective
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      D' : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.208, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.212, u_2} D
      inst✝² : CategoryTheory.Category.{?u.216, u_3} D'
      W : CategoryTheory.MorphismProperty C
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      inst✝¹ : L.IsLocalization W
      inst✝ : L'.IsLocalization W
      X Y : C
      f : Quiver.Hom (L'.obj X) (L'.obj Y)
      ⊢ Eq ((fun f => CategoryTheory.StructuredArrow.mk ((CategoryTheory.Localizatio …
    -/
    dsimp
    /-
      case intro.intro
      C : Type u_1
      D : Type u_2
      D' : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.208, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.212, u_2} D
      inst✝² : CategoryTheory.Category.{?u.216, u_3} D'
      W : CategoryTheory.MorphismProperty C
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      inst✝¹ : L.IsLocalization W
      inst✝ : L'.IsLocalization W
      X Y : C
      f : Quiver.Hom (L'.obj X) (L'.obj Y)
      ⊢ Eq (CategoryTheory.StructuredArrow.mk ((CategoryTheory.Localization.homEquiv …
    -/
    rw [← homEquiv_symm_apply, Equiv.symm_apply_apply]
    /-
      🎉 no goals
    -/


open Construction in
private lemma induction_structuredArrow'
    (hP₀ : P (StructuredArrow.mk (𝟙 (W.Q.obj X))))
    (hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Y₁ ⟶ Y₂) (φ : W.Q.obj X ⟶ W.Q.obj Y₁),
      P (StructuredArrow.mk φ) → P (StructuredArrow.mk (φ ≫ W.Q.map f)))
    (hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Y₁ ⟶ Y₂) (hw : W w) (φ : W.Q.obj X ⟶ W.Q.obj Y₂),
      P (StructuredArrow.mk φ) → P (StructuredArrow.mk (φ ≫ (isoOfHom W.Q W w hw).inv)))
    (g : StructuredArrow (W.Q.obj X) W.Q) : P g := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_4, u_1} C
    W : CategoryTheory.MorphismProperty C
    X : C
    P : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop
    hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
    hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (W.Q.obj X) (W.Q.ob …
    hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (W.Q.obj …
    g : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q
    ⊢ P g
  -/
  let X₀ : Paths (LocQuiver W) := ⟨X⟩
  suffices ∀ ⦃Y₀ : Paths (LocQuiver W)⦄ (f : X₀ ⟶ Y₀),
      P (StructuredArrow.mk ((Quotient.functor (relations W)).map f)) by
    obtain ⟨Y, g, rfl⟩ := g.mk_surjective
    obtain ⟨g, rfl⟩ := (Quotient.functor (relations W)).map_surjective g
    exact this g
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_4, u_1} C
    W : CategoryTheory.MorphismProperty C
    X : C
    P : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop
    hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
    hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (W.Q.obj X) (W.Q.ob …
    hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (W.Q.obj …
    g : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q
    X₀ : CategoryTheory.Paths (CategoryTheory.Localization.Construction.LocQuiver  …
    ⊢ ∀ ⦃Y₀ : CategoryTheory.Paths (CategoryTheory.Localization.Construction.LocQu …
  -/
  intro Y₀ f
  induction f with
  | nil => exact hP₀
  | cons f g hf =>
      obtain (g|⟨w, hw⟩) := g
      · exact hP₁ g _ hf
      · simpa only [← Construction.wInv_eq_isoOfHom_inv w hw] using hP₂ w hw _ hf


@[elab_as_elim]
lemma induction_structuredArrow
    (hP₀ : P (StructuredArrow.mk (𝟙 (L.obj X))))
    (hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Y₁ ⟶ Y₂) (φ : L.obj X ⟶ L.obj Y₁),
      P (StructuredArrow.mk φ) → P (StructuredArrow.mk (φ ≫ L.map f)))
    (hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Y₁ ⟶ Y₂) (hw : W w) (φ : L.obj X ⟶ L.obj Y₂),
      P (StructuredArrow.mk φ) → P (StructuredArrow.mk (φ ≫ (isoOfHom L W w hw).inv)))
    (g : StructuredArrow (L.obj X) L) : P g := by
  let P' : StructuredArrow (W.Q.obj X) W.Q → Prop :=
    fun g ↦ P (structuredArrowEquiv W W.Q L g)
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    X : C
    P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
    hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
    hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
    hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
    g : CategoryTheory.StructuredArrow (L.obj X) L
    P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
    ⊢ P g
  -/
  rw [← (structuredArrowEquiv W W.Q L).apply_symm_apply g]
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    X : C
    P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
    hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
    hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
    hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
    g : CategoryTheory.StructuredArrow (L.obj X) L
    P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
    ⊢ P ((CategoryTheory.Localization.structuredArrowEquiv W W.Q L) ((CategoryTheo …
  -/
  apply induction_structuredArrow' W P'
    /-
      case hP₀
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      X : C
      P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
      hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
      hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
      hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
      g : CategoryTheory.StructuredArrow (L.obj X) L
      P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
      ⊢ P' (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id (W.Q …
    -/
  · convert hP₀
    /-
      case h.e'_1
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      X : C
      P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
      hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
      hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
      hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
      g : CategoryTheory.StructuredArrow (L.obj X) L
      P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
      ⊢ Eq ((CategoryTheory.Localization.structuredArrowEquiv W W.Q L) (CategoryTheo …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case hP₁
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      X : C
      P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
      hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
      hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
      hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
      g : CategoryTheory.StructuredArrow (L.obj X) L
      P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
      ⊢ ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (W.Q.obj X) (W.Q.obj Y₁ …
    -/
  · intros Y₁ Y₂ f φ hφ
    /-
      case hP₁
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      X : C
      P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
      hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
      hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
      hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
      g : CategoryTheory.StructuredArrow (L.obj X) L
      P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
      Y₁ Y₂ : C
      f : Quiver.Hom Y₁ Y₂
      φ : Quiver.Hom (W.Q.obj X) (W.Q.obj Y₁)
      hφ : P' (CategoryTheory.StructuredArrow.mk φ)
      ⊢ P' (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp φ  …
    -/
    convert hP₁ f (homEquiv W W.Q L φ) hφ
    /-
      case h.e'_1
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      X : C
      P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
      hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
      hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
      hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
      g : CategoryTheory.StructuredArrow (L.obj X) L
      P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
      Y₁ Y₂ : C
      f : Quiver.Hom Y₁ Y₂
      φ : Quiver.Hom (W.Q.obj X) (W.Q.obj Y₁)
      hφ : P' (CategoryTheory.StructuredArrow.mk φ)
      ⊢ Eq ((CategoryTheory.Localization.structuredArrowEquiv W W.Q L) (CategoryTheo …
    -/
    simp [homEquiv_comp]
    /-
      🎉 no goals
    -/
    /-
      case hP₂
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      X : C
      P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
      hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
      hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
      hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
      g : CategoryTheory.StructuredArrow (L.obj X) L
      P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
      ⊢ ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (W.Q.obj X)  …
    -/
  · intros Y₁ Y₂ w hw φ hφ
    /-
      case hP₂
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      X : C
      P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
      hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
      hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
      hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
      g : CategoryTheory.StructuredArrow (L.obj X) L
      P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
      Y₁ Y₂ : C
      w : Quiver.Hom Y₁ Y₂
      hw : W w
      φ : Quiver.Hom (W.Q.obj X) (W.Q.obj Y₂)
      hφ : P' (CategoryTheory.StructuredArrow.mk φ)
      ⊢ P' (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp φ  …
    -/
    convert hP₂ w hw (homEquiv W W.Q L φ) hφ
    /-
      case h.e'_1
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      X : C
      P : CategoryTheory.StructuredArrow (L.obj X) L → Prop
      hP₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.id ( …
      hP₁ : ∀ ⦃Y₁ Y₂ : C⦄ (f : Quiver.Hom Y₁ Y₂) (φ : Quiver.Hom (L.obj X) (L.obj Y₁ …
      hP₂ : ∀ ⦃Y₁ Y₂ : C⦄ (w : Quiver.Hom Y₁ Y₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
      g : CategoryTheory.StructuredArrow (L.obj X) L
      P' : CategoryTheory.StructuredArrow (W.Q.obj X) W.Q → Prop := fun g => P ((Cat …
      Y₁ Y₂ : C
      w : Quiver.Hom Y₁ Y₂
      hw : W w
      φ : Quiver.Hom (W.Q.obj X) (W.Q.obj Y₂)
      hφ : P' (CategoryTheory.StructuredArrow.mk φ)
      ⊢ Eq ((CategoryTheory.Localization.structuredArrowEquiv W W.Q L) (CategoryTheo …
    -/
    simp [homEquiv_comp, homEquiv_isoOfHom_inv]
    /-
      🎉 no goals
    -/


@[elab_as_elim]
lemma induction_costructuredArrow
    (hP₀ : P (CostructuredArrow.mk (𝟙 (L.obj Y))))
    (hP₁ : ∀ ⦃X₁ X₂ : C⦄ (f : X₁ ⟶ X₂) (φ : L.obj X₂ ⟶ L.obj Y),
      P (CostructuredArrow.mk φ) → P (CostructuredArrow.mk (L.map f ≫ φ)))
    (hP₂ : ∀ ⦃X₁ X₂ : C⦄ (w : X₁ ⟶ X₂) (hw : W w) (φ : L.obj X₁ ⟶ L.obj Y),
      P (CostructuredArrow.mk φ) → P (CostructuredArrow.mk ((isoOfHom L W w hw).inv ≫ φ)))
    (g : CostructuredArrow L (L.obj Y)) : P g := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    Y : C
    P : CategoryTheory.CostructuredArrow L (L.obj Y) → Prop
    hP₀ : P (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.id …
    hP₁ : ∀ ⦃X₁ X₂ : C⦄ (f : Quiver.Hom X₁ X₂) (φ : Quiver.Hom (L.obj X₂) (L.obj Y …
    hP₂ : ∀ ⦃X₁ X₂ : C⦄ (w : Quiver.Hom X₁ X₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
    g : CategoryTheory.CostructuredArrow L (L.obj Y)
    ⊢ P g
  -/
  let g' := StructuredArrow.mk (T := L.op) (Y := op g.left) g.hom.op
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    Y : C
    P : CategoryTheory.CostructuredArrow L (L.obj Y) → Prop
    hP₀ : P (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.id …
    hP₁ : ∀ ⦃X₁ X₂ : C⦄ (f : Quiver.Hom X₁ X₂) (φ : Quiver.Hom (L.obj X₂) (L.obj Y …
    hP₂ : ∀ ⦃X₁ X₂ : C⦄ (w : Quiver.Hom X₁ X₂) (hw : W w) (φ : Quiver.Hom (L.obj X …
    g : CategoryTheory.CostructuredArrow L (L.obj Y)
    g' : CategoryTheory.StructuredArrow { unop := (CategoryTheory.Functor.fromPUni …
    ⊢ P g
  -/
  show P (CostructuredArrow.mk g'.hom.unop)
  induction g' using induction_structuredArrow L.op W.op with
  | hP₀ => exact hP₀
  | hP₁ f φ hφ => exact hP₁ f.unop φ.unop hφ
  | hP₂ w hw φ hφ => simpa [isoOfHom_op_inv L W w hw] using hP₂ w.unop hw φ.unop hφ


