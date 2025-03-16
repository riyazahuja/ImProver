/-- A morphism property `W` on a category `C` is compatible with the shift by a
monoid `A` when for all `a : A`, a morphism `f` belongs to `W`
if and only if `f⟦a⟧'` does. -/
class IsCompatibleWithShift : Prop where
  /-- the condition that for all `a : A`, the morphism property `W` is not changed when
  we take its inverse image by the shift functor by `a` -/
  condition : ∀ (a : A), W.inverseImage (shiftFunctor C a) = W


lemma iff {X Y : C} (f : X ⟶ Y) (a : A) : W (f⟦a⟧') ↔ W f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    A : Type w
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : W.IsCompatibleWithShift A
    X Y : C
    f : Quiver.Hom X Y
    a : A
    ⊢ Iff (W ((CategoryTheory.shiftFunctor C a).map f)) (W f)
  -/
  conv_rhs => rw [← @IsCompatibleWithShift.condition _ _ W A _ _ _ a]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    A : Type w
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : W.IsCompatibleWithShift A
    X Y : C
    f : Quiver.Hom X Y
    a : A
    ⊢ Iff (W ((CategoryTheory.shiftFunctor C a).map f)) (W.inverseImage (CategoryT …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma shiftFunctor_comp_inverts (a : A) :
    W.IsInvertedBy (shiftFunctor C a ⋙ L) := fun _ _ f hf =>
                                 /-
                                   C : Type u₁
                                   D : Type u₂
                                   inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
                                   inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                                   L : CategoryTheory.Functor C D
                                   W : CategoryTheory.MorphismProperty C
                                   inst✝³ : L.IsLocalization W
                                   A : Type w
                                   inst✝² : AddMonoid A
                                   inst✝¹ : CategoryTheory.HasShift C A
                                   inst✝ : W.IsCompatibleWithShift A
                                   a : A
                                   x✝¹ x✝ : C
                                   f : Quiver.Hom x✝¹ x✝
                                   hf : W f
                                   ⊢ W ((CategoryTheory.shiftFunctor C a).map f)
                                 -/
  Localization.inverts L W _ (by simpa only [iff] using hf)
                                 /-
                                   🎉 no goals
                                 -/


variable {A} in
lemma shift {X Y : C} {f : X ⟶ Y} (hf : W f) (a : A) : W (f⟦a⟧') := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    W : CategoryTheory.MorphismProperty C
    A : Type w
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : W.IsCompatibleWithShift A
    X Y : C
    f : Quiver.Hom X Y
    hf : W f
    a : A
    ⊢ W ((CategoryTheory.shiftFunctor C a).map f)
  -/
  simpa only [IsCompatibleWithShift.iff W f a] using hf
  /-
    🎉 no goals
  -/


variable {A} in
/-- The morphism of localizer from `W` to `W` given by the functor `shiftFunctor C a`
when `a : A` and `W` is compatible with the shift by `A`. -/
abbrev shiftLocalizerMorphism (a : A) : LocalizerMorphism W W where
  functor := shiftFunctor C a
            /-
              C : Type u₁
              D : Type u₂
              inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
              E : Type u₃
              inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
              L : CategoryTheory.Functor C D
              W : CategoryTheory.MorphismProperty C
              inst✝³ : L.IsLocalization W
              A : Type w
              inst✝² : AddMonoid A
              inst✝¹ : CategoryTheory.HasShift C A
              inst✝ : W.IsCompatibleWithShift A
              a : A
              ⊢ LE.le W (W.inverseImage (CategoryTheory.shiftFunctor C a))
            -/
  map := by rw [MorphismProperty.IsCompatibleWithShift.condition]
            /-
              🎉 no goals
            -/


/-- When `L : C ⥤ D` is a localization functor with respect to a morphism property `W`
that is compatible with the shift by a monoid `A` on `C`, this is the induced
shift on the category `D`. -/
noncomputable def HasShift.localized : HasShift D A :=
  have := Localization.full_whiskeringLeft L W D
  have := Localization.faithful_whiskeringLeft L W D
  HasShift.induced L A
    (fun a => Localization.lift (shiftFunctor C a ⋙ L)
      (MorphismProperty.IsCompatibleWithShift.shiftFunctor_comp_inverts L W a) L)
    (fun _ => Localization.fac _ _ _)


/-- The localization functor `L : C ⥤ D` is compatible with the shift. -/
@[nolint unusedHavesSuffices]
noncomputable def Functor.CommShift.localized :
    @Functor.CommShift _ _ _ _ L A _ _ (HasShift.localized L W A) :=
  have := Localization.full_whiskeringLeft L W D
  have := Localization.faithful_whiskeringLeft L W D
  Functor.CommShift.ofInduced _ _ _ _


/-- The localized category `W.Localization` is endowed with the induced shift. -/
noncomputable instance HasShift.localization :
    HasShift W.Localization A :=
  HasShift.localized W.Q W A


/-- The localization functor `W.Q : C ⥤ W.Localization` is compatible with the shift. -/
noncomputable instance MorphismProperty.commShift_Q :
    W.Q.CommShift A :=
  Functor.CommShift.localized W.Q W A


/-- The localized category `W.Localization'` is endowed with the induced shift. -/
noncomputable instance HasShift.localization' :
    HasShift W.Localization' A :=
  HasShift.localized W.Q' W A


/-- The localization functor `W.Q' : C ⥤ W.Localization'` is compatible with the shift. -/
noncomputable instance MorphismProperty.commShift_Q' :
    W.Q'.CommShift A :=
  Functor.CommShift.localized W.Q' W A


/-- Auxiliary definition for `Functor.commShiftOfLocalization`. -/
noncomputable def iso (a : A) :
    shiftFunctor D a ⋙ F' ≅ F' ⋙ shiftFunctor E a :=
  Localization.liftNatIso L W (L ⋙ shiftFunctor D a ⋙ F')
    (L ⋙ F' ⋙ shiftFunctor E a) _ _
      ((Functor.associator _ _ _).symm ≪≫
        isoWhiskerRight (L.commShiftIso a).symm F' ≪≫
        Functor.associator _ _ _ ≪≫
        isoWhiskerLeft _ (Lifting.iso L W F F') ≪≫
        F.commShiftIso a ≪≫
        isoWhiskerRight (Lifting.iso L W F F').symm _ ≪≫ Functor.associator _ _ _)


@[simp, reassoc]
lemma iso_hom_app (a : A) (X : C) :
    (commShiftOfLocalization.iso L W F F' a).hom.app (L.obj X) =
      F'.map ((L.commShiftIso a).inv.app X) ≫
      (Lifting.iso L W F F').hom.app (X⟦a⟧) ≫
        (F.commShiftIso a).hom.app X ≫
          (shiftFunctor E a).map ((Lifting.iso L W F F').inv.app X) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    A : Type w
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    F : CategoryTheory.Functor C E
    F' : CategoryTheory.Functor D E
    inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : L.CommShift A
    inst✝ : F.CommShift A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.Functor.commShiftOfLocalization.iso L W F F' a).hom.app  …
  -/
  simp [commShiftOfLocalization.iso]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma iso_inv_app (a : A) (X : C) :
    (commShiftOfLocalization.iso L W F F' a).inv.app (L.obj X) =
        (shiftFunctor E a).map ((Lifting.iso L W F F').hom.app X) ≫
        (F.commShiftIso a).inv.app X ≫
      (Lifting.iso L W F F').inv.app (X⟦a⟧) ≫
      F'.map ((L.commShiftIso a).hom.app X) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    A : Type w
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    F : CategoryTheory.Functor C E
    F' : CategoryTheory.Functor D E
    inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : L.CommShift A
    inst✝ : F.CommShift A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.Functor.commShiftOfLocalization.iso L W F F' a).inv.app  …
  -/
  simp [commShiftOfLocalization.iso]
  /-
    🎉 no goals
  -/


/-- In the context of localization of categories, if a functor
is induced by a functor which commutes with the shift, then
this functor commutes with the shift. -/
noncomputable def commShiftOfLocalization : F'.CommShift A where
  iso := commShiftOfLocalization.iso L W F F'
  zero := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      ⊢ Eq (CategoryTheory.Functor.commShiftOfLocalization.iso L W F F' 0) (Category …
    -/
    ext1
    /-
      case w
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      ⊢ Eq (CategoryTheory.Functor.commShiftOfLocalization.iso L W F F' 0).hom (Cate …
    -/
    apply natTrans_ext L W
    /-
      case w
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      ⊢ ∀ (X : C), Eq ((CategoryTheory.Functor.commShiftOfLocalization.iso L W F F'  …
    -/
    intro X
    /-
      case w
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      X : C
      ⊢ Eq ((CategoryTheory.Functor.commShiftOfLocalization.iso L W F F' 0).hom.app  …
    -/
    dsimp
    simp only [commShiftOfLocalization.iso_hom_app, comp_obj, commShiftIso_zero,
      CommShift.isoZero_inv_app, map_comp, CommShift.isoZero_hom_app, Category.assoc,
      ← NatTrans.naturality_assoc, ← NatTrans.naturality]
    /-
      case w
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.map ((CategoryTheory.shiftFunctor …
    -/
    dsimp
    simp only [← Functor.map_comp_assoc, ← Functor.map_comp,
      Iso.inv_hom_id_app, id_obj, map_id, Category.id_comp, Iso.hom_inv_id_app_assoc]
  add a b := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      a b : A
      ⊢ Eq (CategoryTheory.Functor.commShiftOfLocalization.iso L W F F' (HAdd.hAdd a …
    -/
    ext1
    /-
      case w
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      a b : A
      ⊢ Eq (CategoryTheory.Functor.commShiftOfLocalization.iso L W F F' (HAdd.hAdd a …
    -/
    apply natTrans_ext L W
    /-
      case w
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      a b : A
      ⊢ ∀ (X : C), Eq ((CategoryTheory.Functor.commShiftOfLocalization.iso L W F F'  …
    -/
    intro X
    /-
      case w
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      a b : A
      X : C
      ⊢ Eq ((CategoryTheory.Functor.commShiftOfLocalization.iso L W F F' (HAdd.hAdd  …
    -/
    dsimp
    simp only [commShiftOfLocalization.iso_hom_app, comp_obj, commShiftIso_add,
      CommShift.isoAdd_inv_app, map_comp, CommShift.isoAdd_hom_app, Category.assoc]
    /-
      case w
      C : Type u₁
      D : Type u₂
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝⁷ : L.IsLocalization W
      A : Type w
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift C A
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : L.CommShift A
      inst✝ : F.CommShift A
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.map ((CategoryTheory.shiftFunctor …
    -/
    congr 1
    rw [← cancel_epi (F'.map ((shiftFunctor D b).map ((L.commShiftIso a).hom.app X))),
      ← F'.map_comp_assoc, ← map_comp, Iso.hom_inv_id_app, map_id, map_id, Category.id_comp]
    conv_lhs =>
      erw [← NatTrans.naturality_assoc]
      dsimp
      rw [← Functor.map_comp_assoc, ← map_comp_assoc, Category.assoc,
        ← map_comp, Iso.inv_hom_id_app]
      dsimp
      rw [map_id, Category.comp_id, ← NatTrans.naturality]
      dsimp
    conv_rhs =>
      erw [← NatTrans.naturality_assoc]
      dsimp
      rw [← Functor.map_comp_assoc, ← map_comp, Iso.hom_inv_id_app]
      dsimp
      rw [map_id, map_id, Category.id_comp, commShiftOfLocalization.iso_hom_app,
        Category.assoc, Category.assoc, Category.assoc, ← map_comp_assoc,
        Iso.inv_hom_id_app, map_id, Category.id_comp]


lemma commShiftOfLocalization_iso_hom_app (a : A) (X : C) :
    letI := Functor.commShiftOfLocalization L W A F F'
    (F'.commShiftIso a).hom.app (L.obj X) =
      F'.map ((L.commShiftIso a).inv.app X) ≫ (Lifting.iso L W F F').hom.app (X⟦a⟧) ≫
        (F.commShiftIso a).hom.app X ≫
          (shiftFunctor E a).map ((Lifting.iso L W F F').inv.app X) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    A : Type w
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    F : CategoryTheory.Functor C E
    F' : CategoryTheory.Functor D E
    inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : L.CommShift A
    inst✝ : F.CommShift A
    a : A
    X : C
    ⊢ Eq ((F'.commShiftIso a).hom.app (L.obj X)) (CategoryTheory.CategoryStruct.co …
  -/
  apply commShiftOfLocalization.iso_hom_app
  /-
    🎉 no goals
  -/


lemma commShiftOfLocalization_iso_inv_app (a : A) (X : C) :
    letI := Functor.commShiftOfLocalization L W A F F'
    (F'.commShiftIso a).inv.app (L.obj X) =
      (shiftFunctor E a).map ((Lifting.iso L W F F').hom.app X) ≫
      (F.commShiftIso a).inv.app X ≫ (Lifting.iso L W F F').inv.app (X⟦a⟧) ≫
     F'.map ((L.commShiftIso a).hom.app X) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    A : Type w
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    F : CategoryTheory.Functor C E
    F' : CategoryTheory.Functor D E
    inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : L.CommShift A
    inst✝ : F.CommShift A
    a : A
    X : C
    ⊢ Eq ((F'.commShiftIso a).inv.app (L.obj X)) (CategoryTheory.CategoryStruct.co …
  -/
  apply commShiftOfLocalization.iso_inv_app
  /-
    🎉 no goals
  -/


instance NatTrans.commShift_iso_hom_of_localization :
    letI := Functor.commShiftOfLocalization L W A F F'
    NatTrans.CommShift (Lifting.iso L W F F').hom A := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    A : Type w
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    F : CategoryTheory.Functor C E
    F' : CategoryTheory.Functor D E
    inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : L.CommShift A
    inst✝ : F.CommShift A
    ⊢ CategoryTheory.NatTrans.CommShift (CategoryTheory.Localization.Lifting.iso L …
  -/
  letI := Functor.commShiftOfLocalization L W A F F'
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    A : Type w
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    F : CategoryTheory.Functor C E
    F' : CategoryTheory.Functor D E
    inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : L.CommShift A
    inst✝ : F.CommShift A
    this : F'.CommShift A := L.commShiftOfLocalization W A F F'
    ⊢ CategoryTheory.NatTrans.CommShift (CategoryTheory.Localization.Lifting.iso L …
  -/
  constructor
  /-
    case shift_comm
    C : Type u₁
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    A : Type w
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    F : CategoryTheory.Functor C E
    F' : CategoryTheory.Functor D E
    inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : L.CommShift A
    inst✝ : F.CommShift A
    this : F'.CommShift A := L.commShiftOfLocalization W A F F'
    ⊢ autoParam (∀ (a : A), Eq (CategoryTheory.CategoryStruct.comp ((L.comp F').co …
  -/
  intro a
  /-
    case shift_comm
    C : Type u₁
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁸ : CategoryTheory.Category.{v₃, u₃} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    A : Type w
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    F : CategoryTheory.Functor C E
    F' : CategoryTheory.Functor D E
    inst✝⁴ : CategoryTheory.Localization.Lifting L W F F'
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : L.CommShift A
    inst✝ : F.CommShift A
    this : F'.CommShift A := L.commShiftOfLocalization W A F F'
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.comp F').commShiftIso a).hom (Cat …
  -/
  ext X
  simp only [comp_app, whiskerRight_app, whiskerLeft_app,
    Functor.commShiftIso_comp_hom_app,
    Functor.commShiftOfLocalization_iso_hom_app,
    Category.assoc, ← Functor.map_comp, ← Functor.map_comp_assoc,
    Iso.hom_inv_id_app, Functor.map_id, Iso.inv_hom_id_app,
    Category.comp_id, Category.id_comp, Functor.comp_obj]


