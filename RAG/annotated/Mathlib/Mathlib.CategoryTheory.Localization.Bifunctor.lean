/-- Classes of morphisms `W₁ : MorphismProperty C₁` and `W₂ : MorphismProperty C₂` are said
to be inverted by `F : C₁ ⥤ C₂ ⥤ E` if `W₁.prod W₂` is inverted by
the functor `uncurry.obj F : C₁ × C₂ ⥤ E`. -/
def IsInvertedBy₂ (W₁ : MorphismProperty C₁) (W₂ : MorphismProperty C₂)
    (F : C₁ ⥤ C₂ ⥤ E) : Prop :=
  (W₁.prod W₂).IsInvertedBy (uncurry.obj F)


/-- Given functors `L₁ : C₁ ⥤ D₁`, `L₂ : C₂ ⥤ D₂`, morphisms properties `W₁` on `C₁`
and `W₂` on `C₂`, and functors `F : C₁ ⥤ C₂ ⥤ E` and `F' : D₁ ⥤ D₂ ⥤ E`, we say
`Lifting₂ L₁ L₂ W₁ W₂ F F'` holds if `F` is induced by `F'`, up to an isomorphism. -/
class Lifting₂ (W₁ : MorphismProperty C₁) (W₂ : MorphismProperty C₂)
    (F : C₁ ⥤ C₂ ⥤ E) (F' : D₁ ⥤ D₂ ⥤ E) where
  /-- the isomorphism `(((whiskeringLeft₂ E).obj L₁).obj L₂).obj F' ≅ F` expressing
  that `F` is induced by `F'` up to an isomorphism -/
  iso' : (((whiskeringLeft₂ E).obj L₁).obj L₂).obj F' ≅ F


/-- The isomorphism `(((whiskeringLeft₂ E).obj L₁).obj L₂).obj F' ≅ F` when
`Lifting₂ L₁ L₂ W₁ W₂ F F'` holds. -/
noncomputable def Lifting₂.iso : (((whiskeringLeft₂ E).obj L₁).obj L₂).obj F' ≅ F :=
  Lifting₂.iso' W₁ W₂


/-- If `Lifting₂ L₁ L₂ W₁ W₂ F F'` holds, then `Lifting L₂ W₂ (F.obj X₁) (F'.obj (L₁.obj X₁))`
holds for any `X₁ : C₁`. -/
noncomputable def Lifting₂.fst (X₁ : C₁) :
    Lifting L₂ W₂ (F.obj X₁) (F'.obj (L₁.obj X₁)) where
  iso' := ((evaluation _ _).obj X₁).mapIso (Lifting₂.iso L₁ L₂ W₁ W₂ F F')


noncomputable instance Lifting₂.flip : Lifting₂ L₂ L₁ W₂ W₁ F.flip F'.flip where
  iso' := (flipFunctor _ _ _).mapIso (Lifting₂.iso L₁ L₂ W₁ W₂ F F')


/-- If `Lifting₂ L₁ L₂ W₁ W₂ F F'` holds, then
`Lifting L₁ W₁ (F.flip.obj X₂) (F'.flip.obj (L₂.obj X₂))` holds for any `X₂ : C₂`. -/
noncomputable def Lifting₂.snd (X₂ : C₂) :
    Lifting L₁ W₁ (F.flip.obj X₂) (F'.flip.obj (L₂.obj X₂)) :=
  Lifting₂.fst L₂ L₁ W₂ W₁ F.flip F'.flip X₂


noncomputable instance Lifting₂.uncurry [Lifting₂ L₁ L₂ W₁ W₂ F F'] :
    Lifting (L₁.prod L₂) (W₁.prod W₂) (uncurry.obj F) (uncurry.obj F') where
  iso' := CategoryTheory.uncurry.mapIso (Lifting₂.iso L₁ L₂ W₁ W₂ F F')


/-- Given localization functor `L₁ : C₁ ⥤ D₁` and `L₂ : C₂ ⥤ D₂` with respect
to `W₁ : MorphismProperty C₁` and `W₂ : MorphismProperty C₂` respectively,
and a bifunctor `F : C₁ ⥤ C₂ ⥤ E` which inverts `W₁` and `W₂`, this is
the induced localized bifunctor `D₁ ⥤ D₂ ⥤ E`. -/
noncomputable def lift₂ : D₁ ⥤ D₂ ⥤ E :=
  curry.obj (lift (uncurry.obj F) hF (L₁.prod L₂))


noncomputable instance : Lifting₂ L₁ L₂ W₁ W₂ F (lift₂ F hF L₁ L₂) where
  iso' := (curryObjProdComp _ _ _).symm ≪≫
    curry.mapIso (fac (uncurry.obj F) hF (L₁.prod L₂)) ≪≫
    currying.unitIso.symm.app F


noncomputable instance Lifting₂.liftingLift₂ (X₁ : C₁) :
    Lifting L₂ W₂ (F.obj X₁) ((lift₂ F hF L₁ L₂).obj (L₁.obj X₁)) :=
  Lifting₂.fst _ _ W₁ _ _ _ _


noncomputable instance Lifting₂.liftingLift₂Flip (X₂ : C₂) :
    Lifting L₁ W₁ (F.flip.obj X₂) ((lift₂ F hF L₁ L₂).flip.obj (L₂.obj X₂)) :=
  Lifting₂.snd _ _ _ W₂ _ _ _


lemma lift₂_iso_hom_app_app₁ (X₁ : C₁) (X₂ : C₂) :
    ((Lifting₂.iso L₁ L₂ W₁ W₂ F (lift₂ F hF L₁ L₂)).hom.app X₁).app X₂ =
      (Lifting.iso L₂ W₂ (F.obj X₁) ((lift₂ F hF L₁ L₂).obj (L₁.obj X₁))).hom.app X₂ :=
  rfl


lemma lift₂_iso_hom_app_app₂ (X₁ : C₁) (X₂ : C₂) :
    ((Lifting₂.iso L₁ L₂ W₁ W₂ F (lift₂ F hF L₁ L₂)).hom.app X₁).app X₂ =
      (Lifting.iso L₁ W₁ (F.flip.obj X₂) ((lift₂ F hF L₁ L₂).flip.obj (L₂.obj X₂))).hom.app X₁ :=
  rfl


/-- The natural transformation `F₁' ⟶ F₂'` of bifunctors induced by a
natural transformation `τ : F₁ ⟶ F₂` when `Lifting₂ L₁ L₂ W₁ W₂ F₁ F₁'`
and `Lifting₂ L₁ L₂ W₁ W₂ F₂ F₂'` hold. -/
noncomputable def lift₂NatTrans (τ : F₁ ⟶ F₂) : F₁' ⟶ F₂' :=
  fullyFaithfulUncurry.preimage
    (liftNatTrans (L₁.prod L₂) (W₁.prod W₂) (uncurry.obj F₁)
      (uncurry.obj F₂) (uncurry.obj F₁') (uncurry.obj F₂') (uncurry.map τ))


@[simp]
theorem lift₂NatTrans_app_app (τ : F₁ ⟶ F₂) (X₁ : C₁) (X₂ : C₂) :
    ((lift₂NatTrans L₁ L₂ W₁ W₂ F₁ F₂ F₁' F₂' τ).app (L₁.obj X₁)).app (L₂.obj X₂) =
      ((Lifting₂.iso L₁ L₂ W₁ W₂ F₁ F₁').hom.app X₁).app X₂ ≫ (τ.app X₁).app X₂ ≫
        ((Lifting₂.iso L₁ L₂ W₁ W₂ F₂ F₂').inv.app X₁).app X₂ := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_3
    D₂ : Type u_4
    E : Type u_5
    inst✝¹⁰ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁹ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁸ : CategoryTheory.Category.{u_11, u_3} D₁
    inst✝⁷ : CategoryTheory.Category.{u_10, u_4} D₂
    inst✝⁶ : CategoryTheory.Category.{u_7, u_5} E
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝⁵ : L₁.IsLocalization W₁
    inst✝⁴ : L₂.IsLocalization W₂
    inst✝³ : W₁.ContainsIdentities
    inst✝² : W₂.ContainsIdentities
    F₁ F₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ E)
    F₁' F₂' : CategoryTheory.Functor D₁ (CategoryTheory.Functor D₂ E)
    inst✝¹ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₁ F₁'
    inst✝ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₂ F₂'
    τ : Quiver.Hom F₁ F₂
    X₁ : C₁
    X₂ : C₂
    ⊢ Eq (((CategoryTheory.Localization.lift₂NatTrans L₁ L₂ W₁ W₂ F₁ F₂ F₁' F₂' τ) …
  -/
  dsimp [lift₂NatTrans, fullyFaithfulUncurry, Equivalence.fullyFaithfulFunctor]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_3
    D₂ : Type u_4
    E : Type u_5
    inst✝¹⁰ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁹ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁸ : CategoryTheory.Category.{u_11, u_3} D₁
    inst✝⁷ : CategoryTheory.Category.{u_10, u_4} D₂
    inst✝⁶ : CategoryTheory.Category.{u_7, u_5} E
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝⁵ : L₁.IsLocalization W₁
    inst✝⁴ : L₂.IsLocalization W₂
    inst✝³ : W₁.ContainsIdentities
    inst✝² : W₂.ContainsIdentities
    F₁ F₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ E)
    F₁' F₂' : CategoryTheory.Functor D₁ (CategoryTheory.Functor D₂ E)
    inst✝¹ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₁ F₁'
    inst✝ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₂ F₂'
    τ : Quiver.Hom F₁ F₂
    X₁ : C₁
    X₂ : C₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((F …
  -/
  simp only [currying_unitIso_hom_app_app_app, currying_unitIso_inv_app_app_app, comp_id, id_comp]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_3
    D₂ : Type u_4
    E : Type u_5
    inst✝¹⁰ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁹ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁸ : CategoryTheory.Category.{u_11, u_3} D₁
    inst✝⁷ : CategoryTheory.Category.{u_10, u_4} D₂
    inst✝⁶ : CategoryTheory.Category.{u_7, u_5} E
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝⁵ : L₁.IsLocalization W₁
    inst✝⁴ : L₂.IsLocalization W₂
    inst✝³ : W₁.ContainsIdentities
    inst✝² : W₂.ContainsIdentities
    F₁ F₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ E)
    F₁' F₂' : CategoryTheory.Functor D₁ (CategoryTheory.Functor D₂ E)
    inst✝¹ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₁ F₁'
    inst✝ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₂ F₂'
    τ : Quiver.Hom F₁ F₂
    X₁ : C₁
    X₂ : C₂
    ⊢ Eq ((CategoryTheory.Localization.liftNatTrans (L₁.prod L₂) (W₁.prod W₂) (Cat …
  -/
  exact liftNatTrans_app _ _ _ _ (uncurry.obj F₁') (uncurry.obj F₂') (uncurry.map τ) ⟨X₁, X₂⟩
  /-
    🎉 no goals
  -/


variable {F₁' F₂'} in
include W₁ W₂ in
theorem natTrans₂_ext {τ τ' : F₁' ⟶ F₂'}
    (h : ∀ (X₁ : C₁) (X₂ : C₂), (τ.app (L₁.obj X₁)).app (L₂.obj X₂) =
      (τ'.app (L₁.obj X₁)).app (L₂.obj X₂)) : τ = τ' :=
  uncurry.map_injective (natTrans_ext (L₁.prod L₂) (W₁.prod W₂) (fun _ ↦ h _ _))


/-- The natural isomorphism `F₁' ≅ F₂'` of bifunctors induced by a
natural isomorphism `e : F₁ ≅ F₂` when `Lifting₂ L₁ L₂ W₁ W₂ F₁ F₁'`
and `Lifting₂ L₁ L₂ W₁ W₂ F₂ F₂'` hold. -/
noncomputable def lift₂NatIso (e : F₁ ≅ F₂) : F₁' ≅ F₂' where
  hom := lift₂NatTrans L₁ L₂ W₁ W₂ F₁ F₂ F₁' F₂' e.hom
  inv := lift₂NatTrans L₁ L₂ W₁ W₂ F₂ F₁ F₂' F₁' e.inv
                                              /-
                                                C₁ : Type u_1
                                                C₂ : Type u_2
                                                D₁ : Type u_3
                                                D₂ : Type u_4
                                                E : Type u_5
                                                E' : Type u_6
                                                inst✝¹¹ : CategoryTheory.Category.{?u.57566, u_1} C₁
                                                inst✝¹⁰ : CategoryTheory.Category.{?u.57570, u_2} C₂
                                                inst✝⁹ : CategoryTheory.Category.{?u.57574, u_3} D₁
                                                inst✝⁸ : CategoryTheory.Category.{?u.57578, u_4} D₂
                                                inst✝⁷ : CategoryTheory.Category.{?u.57582, u_5} E
                                                inst✝⁶ : CategoryTheory.Category.{?u.57586, u_6} E'
                                                L₁ : CategoryTheory.Functor C₁ D₁
                                                L₂ : CategoryTheory.Functor C₂ D₂
                                                W₁ : CategoryTheory.MorphismProperty C₁
                                                W₂ : CategoryTheory.MorphismProperty C₂
                                                inst✝⁵ : L₁.IsLocalization W₁
                                                inst✝⁴ : L₂.IsLocalization W₂
                                                inst✝³ : W₁.ContainsIdentities
                                                inst✝² : W₂.ContainsIdentities
                                                F₁ F₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ E)
                                                F₁' F₂' : CategoryTheory.Functor D₁ (CategoryTheory.Functor D₂ E)
                                                inst✝¹ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₁ F₁'
                                                inst✝ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₂ F₂'
                                                e : CategoryTheory.Iso F₁ F₂
                                                ⊢ ∀ (X₁ : C₁) (X₂ : C₂), Eq (((CategoryTheory.CategoryStruct.comp (CategoryThe …
                                              -/
  hom_inv_id := natTrans₂_ext L₁ L₂ W₁ W₂ (by aesop_cat)
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                C₁ : Type u_1
                                                C₂ : Type u_2
                                                D₁ : Type u_3
                                                D₂ : Type u_4
                                                E : Type u_5
                                                E' : Type u_6
                                                inst✝¹¹ : CategoryTheory.Category.{?u.57566, u_1} C₁
                                                inst✝¹⁰ : CategoryTheory.Category.{?u.57570, u_2} C₂
                                                inst✝⁹ : CategoryTheory.Category.{?u.57574, u_3} D₁
                                                inst✝⁸ : CategoryTheory.Category.{?u.57578, u_4} D₂
                                                inst✝⁷ : CategoryTheory.Category.{?u.57582, u_5} E
                                                inst✝⁶ : CategoryTheory.Category.{?u.57586, u_6} E'
                                                L₁ : CategoryTheory.Functor C₁ D₁
                                                L₂ : CategoryTheory.Functor C₂ D₂
                                                W₁ : CategoryTheory.MorphismProperty C₁
                                                W₂ : CategoryTheory.MorphismProperty C₂
                                                inst✝⁵ : L₁.IsLocalization W₁
                                                inst✝⁴ : L₂.IsLocalization W₂
                                                inst✝³ : W₁.ContainsIdentities
                                                inst✝² : W₂.ContainsIdentities
                                                F₁ F₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ E)
                                                F₁' F₂' : CategoryTheory.Functor D₁ (CategoryTheory.Functor D₂ E)
                                                inst✝¹ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₁ F₁'
                                                inst✝ : CategoryTheory.Localization.Lifting₂ L₁ L₂ W₁ W₂ F₂ F₂'
                                                e : CategoryTheory.Iso F₁ F₂
                                                ⊢ ∀ (X₁ : C₁) (X₂ : C₂), Eq (((CategoryTheory.CategoryStruct.comp (CategoryThe …
                                              -/
  inv_hom_id := natTrans₂_ext L₁ L₂ W₁ W₂ (by aesop_cat)
                                              /-
                                                🎉 no goals
                                              -/


