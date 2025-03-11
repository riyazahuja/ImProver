/-- Basic constructor of an equivalence between localized categories -/
noncomputable def equivalence : D₁ ≌ D₂ :=
  Equivalence.mk G' F' (liftNatIso L₁ W₁ L₁ (G ⋙ F') (𝟭 D₁) (G' ⋙ F') α.symm)
    (liftNatIso L₂ W₂ (F ⋙ G') L₂ (F' ⋙ G') (𝟭 D₂) β)


@[simp]
lemma equivalence_counitIso_app (X : C₂) :
    (equivalence L₁ W₁ L₂ W₂ G G' F F' α β).counitIso.app (L₂.obj X) =
      (Lifting.iso L₂ W₂ (F ⋙ G') (F' ⋙ G')).app X ≪≫ β.app X := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁷ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_7, u_4} D₁
    inst✝⁴ : CategoryTheory.Category.{u_6, u_5} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝² : L₂.IsLocalization W₂
    G : CategoryTheory.Functor C₁ D₂
    G' : CategoryTheory.Functor D₁ D₂
    inst✝¹ : CategoryTheory.Localization.Lifting L₁ W₁ G G'
    F : CategoryTheory.Functor C₂ D₁
    F' : CategoryTheory.Functor D₂ D₁
    inst✝ : CategoryTheory.Localization.Lifting L₂ W₂ F F'
    α : CategoryTheory.Iso (G.comp F') L₁
    β : CategoryTheory.Iso (F.comp G') L₂
    X : C₂
    ⊢ Eq ((CategoryTheory.Localization.equivalence L₁ W₁ L₂ W₂ G G' F F' α β).coun …
  -/
  ext
  /-
    case w
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁷ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_7, u_4} D₁
    inst✝⁴ : CategoryTheory.Category.{u_6, u_5} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝² : L₂.IsLocalization W₂
    G : CategoryTheory.Functor C₁ D₂
    G' : CategoryTheory.Functor D₁ D₂
    inst✝¹ : CategoryTheory.Localization.Lifting L₁ W₁ G G'
    F : CategoryTheory.Functor C₂ D₁
    F' : CategoryTheory.Functor D₂ D₁
    inst✝ : CategoryTheory.Localization.Lifting L₂ W₂ F F'
    α : CategoryTheory.Iso (G.comp F') L₁
    β : CategoryTheory.Iso (F.comp G') L₂
    X : C₂
    ⊢ Eq ((CategoryTheory.Localization.equivalence L₁ W₁ L₂ W₂ G G' F F' α β).coun …
  -/
  dsimp [equivalence, Equivalence.mk]
  /-
    case w
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁷ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_7, u_4} D₁
    inst✝⁴ : CategoryTheory.Category.{u_6, u_5} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝² : L₂.IsLocalization W₂
    G : CategoryTheory.Functor C₁ D₂
    G' : CategoryTheory.Functor D₁ D₂
    inst✝¹ : CategoryTheory.Localization.Lifting L₁ W₁ G G'
    F : CategoryTheory.Functor C₂ D₁
    F' : CategoryTheory.Functor D₂ D₁
    inst✝ : CategoryTheory.Localization.Lifting L₂ W₂ F F'
    α : CategoryTheory.Iso (G.comp F') L₁
    β : CategoryTheory.Iso (F.comp G') L₂
    X : C₂
    ⊢ Eq ((CategoryTheory.Localization.liftNatTrans L₂ W₂ (F.comp G') L₂ (F'.comp  …
  -/
  rw [liftNatTrans_app]
  /-
    case w
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁷ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_7, u_4} D₁
    inst✝⁴ : CategoryTheory.Category.{u_6, u_5} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝² : L₂.IsLocalization W₂
    G : CategoryTheory.Functor C₁ D₂
    G' : CategoryTheory.Functor D₁ D₂
    inst✝¹ : CategoryTheory.Localization.Lifting L₁ W₁ G G'
    F : CategoryTheory.Functor C₂ D₁
    F' : CategoryTheory.Functor D₂ D₁
    inst✝ : CategoryTheory.Localization.Lifting L₂ W₂ F F'
    α : CategoryTheory.Iso (G.comp F') L₁
    β : CategoryTheory.Iso (F.comp G') L₂
    X : C₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Localization.Lifting …
  -/
  dsimp [Lifting.iso]
  /-
    case w
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁷ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_7, u_4} D₁
    inst✝⁴ : CategoryTheory.Category.{u_6, u_5} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝² : L₂.IsLocalization W₂
    G : CategoryTheory.Functor C₁ D₂
    G' : CategoryTheory.Functor D₁ D₂
    inst✝¹ : CategoryTheory.Localization.Lifting L₁ W₁ G G'
    F : CategoryTheory.Functor C₂ D₁
    F' : CategoryTheory.Functor D₂ D₁
    inst✝ : CategoryTheory.Localization.Lifting L₂ W₂ F F'
    α : CategoryTheory.Iso (G.comp F') L₁
    β : CategoryTheory.Iso (F.comp G') L₂
    X : C₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G'.map ((CategoryTheory.Localization …
  -/
  rw [comp_id]
  /-
    🎉 no goals
  -/


include L₁ W₁ L₂ W₂ G F F' α β in
/-- Basic constructor of an equivalence between localized categories -/
lemma isEquivalence : G'.IsEquivalence :=
  (equivalence L₁ W₁ L₂ W₂ G G' F F' α β).isEquivalence_functor


/-- If `L₁ : C₁ ⥤ D` is a localization functor for `W₁ : MorphismProperty C₁`, then it is also
the case of a functor `L₂ : C₂ ⥤ D` for a suitable `W₂ : MorphismProperty C₂` when
we have an equivalence of category `E : C₁ ≌ C₂` and an isomorphism `E.functor ⋙ L₂ ≅ L₁`. -/
lemma of_equivalence_source (L₁ : C₁ ⥤ D) (W₁ : MorphismProperty C₁)
    (L₂ : C₂ ⥤ D) (W₂ : MorphismProperty C₂)
    (E : C₁ ≌ C₂) (hW₁ : W₁ ≤ W₂.isoClosure.inverseImage E.functor) (hW₂ : W₂.IsInvertedBy L₂)
    [L₁.IsLocalization W₁] (iso : E.functor ⋙ L₂ ≅ L₁) : L₂.IsLocalization W₂ := by
  have h : W₁.IsInvertedBy (E.functor ⋙ W₂.Q) := fun _ _ f hf => by
    obtain ⟨_, _, f', hf', ⟨e⟩⟩ := hW₁ f hf
    exact ((MorphismProperty.isomorphisms _).arrow_mk_iso_iff
      (W₂.Q.mapArrow.mapIso e)).1 (Localization.inverts W₂.Q W₂ _ hf')
  exact
    { inverts := hW₂
      isEquivalence :=
        Localization.isEquivalence W₂.Q W₂ L₁ W₁ L₂ (Construction.lift L₂ hW₂)
          (E.functor ⋙ W₂.Q) (Localization.lift (E.functor ⋙ W₂.Q) h L₁) (by
            calc
              L₂ ⋙ lift (E.functor ⋙ W₂.Q) h L₁ ≅ _ := (leftUnitor _).symm
              _ ≅ _ := isoWhiskerRight E.counitIso.symm _
              _ ≅ E.inverse ⋙ E.functor ⋙ L₂ ⋙ lift (E.functor ⋙ W₂.Q) h L₁ :=
                    Functor.associator _ _ _
              _ ≅ E.inverse ⋙ L₁ ⋙ lift (E.functor ⋙ W₂.Q) h L₁ :=
                    isoWhiskerLeft E.inverse ((Functor.associator _ _ _).symm ≪≫
                      isoWhiskerRight iso _)
              _ ≅ E.inverse ⋙ E.functor ⋙ W₂.Q :=
                    isoWhiskerLeft _ (Localization.fac (E.functor ⋙ W₂.Q) h L₁)
              _ ≅ (E.inverse ⋙ E.functor) ⋙ W₂.Q := (Functor.associator _ _ _).symm
              _ ≅ 𝟭 C₂ ⋙ W₂.Q := isoWhiskerRight E.counitIso _
              _ ≅ W₂.Q := leftUnitor _)
          (Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ (Lifting.iso W₂.Q W₂ _ _)  ≪≫ iso) }


/-- If `L₁ : C₁ ⥤ D₁` is a localization functor for `W₁ : MorphismProperty C₁`, then if we
transport this functor `L₁` via equivalences `C₁ ≌ C₂` and `D₁ ≌ D₂` to get a functor
`L₂ : C₂ ⥤ D₂`, then `L₂` is also a localization functor for
a suitable `W₂ : MorphismProperty C₂`. -/
lemma of_equivalences (L₁ : C₁ ⥤ D₁) (W₁ : MorphismProperty C₁) [L₁.IsLocalization W₁]
    (L₂ : C₂ ⥤ D₂) (W₂ : MorphismProperty C₂)
    (E : C₁ ≌ C₂) (E' : D₁ ≌ D₂) [CatCommSq E.functor L₁ L₂ E'.functor]
    (hW₁ : W₁ ≤ W₂.isoClosure.inverseImage E.functor) (hW₂ : W₂.IsInvertedBy L₂) :
    L₂.IsLocalization W₂ := by
  haveI : (E.functor ⋙ L₂).IsLocalization W₁ :=
    of_equivalence_target L₁ W₁ _ E' ((CatCommSq.iso _ _ _ _).symm)
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_7, u_4} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_5} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    E : CategoryTheory.Equivalence C₁ C₂
    E' : CategoryTheory.Equivalence D₁ D₂
    inst✝ : CategoryTheory.CatCommSq E.functor L₁ L₂ E'.functor
    hW₁ : LE.le W₁ (W₂.isoClosure.inverseImage E.functor)
    hW₂ : W₂.IsInvertedBy L₂
    this : (E.functor.comp L₂).IsLocalization W₁
    ⊢ L₂.IsLocalization W₂
  -/
  exact of_equivalence_source (E.functor ⋙ L₂) W₁ L₂ W₂ E hW₁ hW₂ (Iso.refl _)
  /-
    🎉 no goals
  -/


