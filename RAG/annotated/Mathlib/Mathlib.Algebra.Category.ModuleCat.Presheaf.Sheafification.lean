/-- Given a locally bijective morphism `α : R₀ ⟶ R.val` where `R₀` is a presheaf of rings
and `R` a sheaf of rings (i.e. `R` identifies to the sheafification of `R₀`), this is
the associated sheaf of modules functor `PresheafOfModules.{v} R₀ ⥤ SheafOfModules.{v} R`. -/
@[simps! (config := .lemmasOnly) map]
noncomputable def sheafification : PresheafOfModules.{v} R₀ ⥤ SheafOfModules.{v} R where
  obj M₀ := sheafify α (CategoryTheory.toSheafify J M₀.presheaf)
  map f := sheafifyMap _ _ _ f
    ((toPresheaf R₀ ⋙ presheafToSheaf J AddCommGrp).map f)
          /-
            C : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} C
            J : CategoryTheory.GrothendieckTopology C
            R₀ : CategoryTheory.Functor (Opposite C) RingCat
            R : CategoryTheory.Sheaf J RingCat
            α : Quiver.Hom R₀ R.val
            inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
            inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
            inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
            inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
            X✝ Y✝ : PresheafOfModules R₀
            f : Quiver.Hom X✝ Y✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.toPresheaf R₀).ma …
          -/
      (by apply toSheafify_naturality)
          /-
            🎉 no goals
          -/
  map_id M₀ := by
    /-
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
      inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
      M₀ : PresheafOfModules R₀
      ⊢ Eq ({ obj := fun M₀ => PresheafOfModules.sheafify α (CategoryTheory.toSheafi …
    -/
    ext1
    /-
      case h
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
      inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
      M₀ : PresheafOfModules R₀
      ⊢ Eq ({ obj := fun M₀ => PresheafOfModules.sheafify α (CategoryTheory.toSheafi …
    -/
    apply (toPresheaf _).map_injective
    /-
      case h.a
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
      inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
      M₀ : PresheafOfModules R₀
      ⊢ Eq ((PresheafOfModules.toPresheaf R.val).map ({ obj := fun M₀ => PresheafOfM …
    -/
    simp
    /-
      case h.a
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
      inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
      M₀ : PresheafOfModules R₀
      ⊢ Eq ((PresheafOfModules.toPresheaf R.val).map (PresheafOfModules.homMk (Categ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp _ _ := by
    /-
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
      inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
      X✝ Y✝ Z✝ : PresheafOfModules R₀
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun M₀ => PresheafOfModules.sheafify α (CategoryTheory.toSheafi …
    -/
    ext1
    /-
      case h
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
      inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
      X✝ Y✝ Z✝ : PresheafOfModules R₀
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun M₀ => PresheafOfModules.sheafify α (CategoryTheory.toSheafi …
    -/
    apply (toPresheaf _).map_injective
    /-
      case h.a
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
      inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
      X✝ Y✝ Z✝ : PresheafOfModules R₀
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ((PresheafOfModules.toPresheaf R.val).map ({ obj := fun M₀ => PresheafOfM …
    -/
    simp
    /-
      case h.a
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
      inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
      X✝ Y✝ Z✝ : PresheafOfModules R₀
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ((PresheafOfModules.toPresheaf R.val).map (PresheafOfModules.homMk (Categ …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The sheafification of presheaves of modules commutes with the functor which
forgets the module structures. -/
noncomputable def sheafificationCompToSheaf :
    sheafification.{v} α ⋙ SheafOfModules.toSheaf _ ≅
      toPresheaf _ ⋙ presheafToSheaf J AddCommGrp :=
  Iso.refl _


/-- The sheafification of presheaves of modules commutes with the functor which
forgets the module structures. -/
noncomputable def sheafificationCompForgetCompToPresheaf :
    sheafification.{v} α ⋙ SheafOfModules.forget _ ⋙ toPresheaf _ ≅
      toPresheaf _ ⋙ presheafToSheaf J AddCommGrp ⋙ sheafToPresheaf J AddCommGrp :=
  Iso.refl _


/-- The bijection between types of morphisms which is part of the adjunction
`sheafificationAdjunction`. -/
noncomputable def sheafificationHomEquiv
    {P : PresheafOfModules.{v} R₀} {F : SheafOfModules.{v} R} :
    ((sheafification α).obj P ⟶ F) ≃
      (P ⟶ (restrictScalars α).obj ((SheafOfModules.forget _).obj F)) := by
  /-
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
    inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
    P : PresheafOfModules R₀
    F : SheafOfModules R
    ⊢ Equiv (Quiver.Hom ((PresheafOfModules.sheafification α).obj P) F) (Quiver.Ho …
  -/
  apply sheafifyHomEquiv
  /-
    🎉 no goals
  -/


lemma toPresheaf_map_sheafificationHomEquiv_def
    {P : PresheafOfModules.{v} R₀} {F : SheafOfModules.{v} R}
    (f : (sheafification α).obj P ⟶ F) :
    (toPresheaf R₀).map (sheafificationHomEquiv α f) =
      CategoryTheory.toSheafify J P.presheaf ≫ (toPresheaf R.val).map f.val := rfl


lemma toPresheaf_map_sheafificationHomEquiv
    {P : PresheafOfModules.{v} R₀} {F : SheafOfModules.{v} R}
    (f : (sheafification α).obj P ⟶ F) :
    (toPresheaf R₀).map (sheafificationHomEquiv α f) =
      (sheafificationAdjunction J AddCommGrp).homEquiv P.presheaf
        ((SheafOfModules.toSheaf _).obj F) ((SheafOfModules.toSheaf _).map f) := by
  /-
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
    inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
    P : PresheafOfModules R₀
    F : SheafOfModules R
    f : Quiver.Hom ((PresheafOfModules.sheafification α).obj P) F
    ⊢ Eq ((PresheafOfModules.toPresheaf R₀).map ((PresheafOfModules.sheafification …
  -/
  rw [toPresheaf_map_sheafificationHomEquiv_def, Adjunction.homEquiv_unit]
  /-
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
    inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
    P : PresheafOfModules R₀
    F : SheafOfModules R
    f : Quiver.Hom ((PresheafOfModules.sheafification α).obj P) F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.toSheafify J P.preshe …
  -/
  dsimp
  /-
    🎉 no goals
  -/


lemma toSheaf_map_sheafificationHomEquiv_symm
    {P : PresheafOfModules.{v} R₀} {F : SheafOfModules.{v} R}
    (g : P ⟶ (restrictScalars α).obj ((SheafOfModules.forget _).obj F)) :
    (SheafOfModules.toSheaf _).map ((sheafificationHomEquiv α).symm g) =
      (((sheafificationAdjunction J AddCommGrp).homEquiv
        P.presheaf ((SheafOfModules.toSheaf R).obj F)).symm ((toPresheaf R₀).map g)) := by
  /-
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
    inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
    P : PresheafOfModules R₀
    F : SheafOfModules R
    g : Quiver.Hom P ((PresheafOfModules.restrictScalars α).obj ((SheafOfModules.f …
    ⊢ Eq ((SheafOfModules.toSheaf R).map ((PresheafOfModules.sheafificationHomEqui …
  -/
  obtain ⟨f, rfl⟩ := (sheafificationHomEquiv α).surjective g
  /-
    case intro
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
    inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
    P : PresheafOfModules R₀
    F : SheafOfModules R
    f : Quiver.Hom ((PresheafOfModules.sheafification α).obj P) F
    ⊢ Eq ((SheafOfModules.toSheaf R).map ((PresheafOfModules.sheafificationHomEqui …
  -/
  apply ((sheafificationAdjunction J AddCommGrp).homEquiv _ _).injective
  /-
    case intro.a
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
    inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
    P : PresheafOfModules R₀
    F : SheafOfModules R
    f : Quiver.Hom ((PresheafOfModules.sheafification α).obj P) F
    ⊢ Eq (((CategoryTheory.sheafificationAdjunction J AddCommGrp).homEquiv P.presh …
  -/
  rw [Equiv.apply_symm_apply, Adjunction.homEquiv_unit, Equiv.symm_apply_apply]
  /-
    case intro.a
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
    inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
    P : PresheafOfModules R₀
    F : SheafOfModules R
    f : Quiver.Hom ((PresheafOfModules.sheafification α).obj P) F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.sheafificationAdjunc …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a locally bijective morphism `α : R₀ ⟶ R.val` where `R₀` is a presheaf of rings
and `R` a sheaf of rings, this is the adjunction
`sheafification.{v} α ⊣ SheafOfModules.forget R ⋙ restrictScalars α`. -/
noncomputable def sheafificationAdjunction :
    sheafification.{v} α ⊣ SheafOfModules.forget R ⋙ restrictScalars α :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ ↦ sheafificationHomEquiv α
      homEquiv_naturality_left_symm := fun {P₀ Q₀ N} f g ↦ by
        /-
          C : Type u'
          inst✝⁴ : CategoryTheory.Category.{v', u'} C
          J : CategoryTheory.GrothendieckTopology C
          R₀ : CategoryTheory.Functor (Opposite C) RingCat
          R : CategoryTheory.Sheaf J RingCat
          α : Quiver.Hom R₀ R.val
          inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
          inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
          inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
          inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
          P₀ Q₀ : PresheafOfModules R₀
          N : SheafOfModules R
          f : Quiver.Hom P₀ Q₀
          g : Quiver.Hom Q₀ (((SheafOfModules.forget R).comp (PresheafOfModules.restrict …
          ⊢ Eq (((fun x x_1 => PresheafOfModules.sheafificationHomEquiv α) P₀ N).symm (C …
        -/
        apply (SheafOfModules.toSheaf _).map_injective
        /-
          case a
          C : Type u'
          inst✝⁴ : CategoryTheory.Category.{v', u'} C
          J : CategoryTheory.GrothendieckTopology C
          R₀ : CategoryTheory.Functor (Opposite C) RingCat
          R : CategoryTheory.Sheaf J RingCat
          α : Quiver.Hom R₀ R.val
          inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
          inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
          inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
          inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
          P₀ Q₀ : PresheafOfModules R₀
          N : SheafOfModules R
          f : Quiver.Hom P₀ Q₀
          g : Quiver.Hom Q₀ (((SheafOfModules.forget R).comp (PresheafOfModules.restrict …
          ⊢ Eq ((SheafOfModules.toSheaf R).map (((fun x x_1 => PresheafOfModules.sheafif …
        -/
        rw [Functor.map_comp]
        erw [toSheaf_map_sheafificationHomEquiv_symm,
          toSheaf_map_sheafificationHomEquiv_symm α g]
        /-
          case a
          C : Type u'
          inst✝⁴ : CategoryTheory.Category.{v', u'} C
          J : CategoryTheory.GrothendieckTopology C
          R₀ : CategoryTheory.Functor (Opposite C) RingCat
          R : CategoryTheory.Sheaf J RingCat
          α : Quiver.Hom R₀ R.val
          inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
          inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
          inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
          inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
          P₀ Q₀ : PresheafOfModules R₀
          N : SheafOfModules R
          f : Quiver.Hom P₀ Q₀
          g : Quiver.Hom Q₀ (((SheafOfModules.forget R).comp (PresheafOfModules.restrict …
          ⊢ Eq (((CategoryTheory.sheafificationAdjunction J AddCommGrp).homEquiv P₀.pres …
        -/
        rw [Functor.map_comp]
        apply (CategoryTheory.sheafificationAdjunction J
          AddCommGrp.{v}).homEquiv_naturality_left_symm
      homEquiv_naturality_right := fun {P₀ M N} f g ↦ by
        /-
          C : Type u'
          inst✝⁴ : CategoryTheory.Category.{v', u'} C
          J : CategoryTheory.GrothendieckTopology C
          R₀ : CategoryTheory.Functor (Opposite C) RingCat
          R : CategoryTheory.Sheaf J RingCat
          α : Quiver.Hom R₀ R.val
          inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
          inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
          inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
          inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
          P₀ : PresheafOfModules R₀
          M N : SheafOfModules R
          f : Quiver.Hom ((PresheafOfModules.sheafification α).obj P₀) M
          g : Quiver.Hom M N
          ⊢ Eq (((fun x x_1 => PresheafOfModules.sheafificationHomEquiv α) P₀ N) (Catego …
        -/
        apply (toPresheaf _).map_injective
        /-
          case a
          C : Type u'
          inst✝⁴ : CategoryTheory.Category.{v', u'} C
          J : CategoryTheory.GrothendieckTopology C
          R₀ : CategoryTheory.Functor (Opposite C) RingCat
          R : CategoryTheory.Sheaf J RingCat
          α : Quiver.Hom R₀ R.val
          inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
          inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
          inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
          inst✝ : CategoryTheory.HasWeakSheafify J AddCommGrp
          P₀ : PresheafOfModules R₀
          M N : SheafOfModules R
          f : Quiver.Hom ((PresheafOfModules.sheafification α).obj P₀) M
          g : Quiver.Hom M N
          ⊢ Eq ((PresheafOfModules.toPresheaf R₀).map (((fun x x_1 => PresheafOfModules. …
        -/
        erw [toPresheaf_map_sheafificationHomEquiv] }
        /-
          🎉 no goals
        -/


lemma sheafificationAdjunction_homEquiv_apply {P : PresheafOfModules.{v} R₀}
    {F : SheafOfModules.{v} R} (f : (sheafification α).obj P ⟶ F) :
    (sheafificationAdjunction α).homEquiv P F f = sheafificationHomEquiv α f := rfl


@[simp]
lemma toPresheaf_map_sheafificationAdjunction_unit_app (M₀ : PresheafOfModules.{v} R₀) :
    (toPresheaf _).map ((sheafificationAdjunction α).unit.app M₀) =
      CategoryTheory.toSheafify J M₀.presheaf := rfl


instance : (sheafification.{v} α).IsLeftAdjoint :=
  (sheafificationAdjunction α).isLeftAdjoint


noncomputable instance :
    PreservesFiniteLimits (sheafification.{v} α ⋙ SheafOfModules.toSheaf.{v} R) :=
  comp_preservesFiniteLimits (toPresheaf.{v} R₀) (presheafToSheaf J AddCommGrp)


instance : (SheafOfModules.toSheaf.{v} R ⋙ sheafToPresheaf _ _).ReflectsIsomorphisms :=
  inferInstanceAs (SheafOfModules.forget.{v} R ⋙ toPresheaf _).ReflectsIsomorphisms


instance : (SheafOfModules.toSheaf.{v} R).ReflectsIsomorphisms :=
  reflectsIsomorphisms_of_comp (SheafOfModules.toSheaf.{v} R) (sheafToPresheaf J _)


noncomputable instance : ReflectsFiniteLimits (SheafOfModules.toSheaf.{v} R) where
  reflects _ _ _ := inferInstance


noncomputable instance : PreservesFiniteLimits (sheafification.{v} α) :=
  preservesFiniteLimits_of_reflects_of_preserves
    (sheafification.{v} α) (SheafOfModules.toSheaf.{v} R)


