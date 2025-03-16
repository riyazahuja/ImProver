/-- Under some conditions on the `MorphismProperty`, functors satisfying the strict
universal property of the localization are stable under composition -/
def StrictUniversalPropertyFixedTarget.comp
    (h₁ : StrictUniversalPropertyFixedTarget L₁ W₁ E)
    (h₂ : StrictUniversalPropertyFixedTarget L₂ W₂ E)
    (W₃ : MorphismProperty C₁) (hW₃ : W₃.IsInvertedBy (L₁ ⋙ L₂))
    (hW₁₃ : W₁ ≤ W₃) (hW₂₃ : W₂ ≤ W₃.map L₁) :
    StrictUniversalPropertyFixedTarget (L₁ ⋙ L₂) W₃ E where
  inverts := hW₃
  lift F hF := h₂.lift (h₁.lift F (MorphismProperty.IsInvertedBy.of_le _ _  F hF hW₁₃)) (by
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      E : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} E
      L₁ : CategoryTheory.Functor C₁ C₂
      L₂ : CategoryTheory.Functor C₂ C₃
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      h₁ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L₁ W₁ E
      h₂ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L₂ W₂ E
      W₃ : CategoryTheory.MorphismProperty C₁
      hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
      hW₁₃ : LE.le W₁ W₃
      hW₂₃ : LE.le W₂ (W₃.map L₁)
      F : CategoryTheory.Functor C₁ E
      hF : W₃.IsInvertedBy F
      ⊢ W₂.IsInvertedBy (h₁.lift F ⋯)
    -/
    refine MorphismProperty.IsInvertedBy.of_le _ _ _ ?_ hW₂₃
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      E : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} E
      L₁ : CategoryTheory.Functor C₁ C₂
      L₂ : CategoryTheory.Functor C₂ C₃
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      h₁ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L₁ W₁ E
      h₂ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L₂ W₂ E
      W₃ : CategoryTheory.MorphismProperty C₁
      hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
      hW₁₃ : LE.le W₁ W₃
      hW₂₃ : LE.le W₂ (W₃.map L₁)
      F : CategoryTheory.Functor C₁ E
      hF : W₃.IsInvertedBy F
      ⊢ (W₃.map L₁).IsInvertedBy (h₁.lift F ⋯)
    -/
    simpa only [MorphismProperty.IsInvertedBy.map_iff, h₁.fac F] using hF)
    /-
      🎉 no goals
    -/
                 /-
                   C₁ : Type u₁
                   C₂ : Type u₂
                   C₃ : Type u₃
                   E : Type u₄
                   inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
                   inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
                   inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
                   inst✝ : CategoryTheory.Category.{v₄, u₄} E
                   L₁ : CategoryTheory.Functor C₁ C₂
                   L₂ : CategoryTheory.Functor C₂ C₃
                   W₁ : CategoryTheory.MorphismProperty C₁
                   W₂ : CategoryTheory.MorphismProperty C₂
                   h₁ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L₁ W₁ E
                   h₂ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L₂ W₂ E
                   W₃ : CategoryTheory.MorphismProperty C₁
                   hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
                   hW₁₃ : LE.le W₁ W₃
                   hW₂₃ : LE.le W₂ (W₃.map L₁)
                   F : CategoryTheory.Functor C₁ E
                   hF : W₃.IsInvertedBy F
                   ⊢ Eq ((L₁.comp L₂).comp ((fun F hF => h₂.lift (h₁.lift F ⋯) ⋯) F hF)) F
                 -/
  fac F hF := by rw [Functor.assoc, h₂.fac, h₁.fac]
                 /-
                   🎉 no goals
                 -/
                                             /-
                                               C₁ : Type u₁
                                               C₂ : Type u₂
                                               C₃ : Type u₃
                                               E : Type u₄
                                               inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
                                               inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
                                               inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
                                               inst✝ : CategoryTheory.Category.{v₄, u₄} E
                                               L₁ : CategoryTheory.Functor C₁ C₂
                                               L₂ : CategoryTheory.Functor C₂ C₃
                                               W₁ : CategoryTheory.MorphismProperty C₁
                                               W₂ : CategoryTheory.MorphismProperty C₂
                                               h₁ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L₁ W₁ E
                                               h₂ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L₂ W₂ E
                                               W₃ : CategoryTheory.MorphismProperty C₁
                                               hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
                                               hW₁₃ : LE.le W₁ W₃
                                               hW₂₃ : LE.le W₂ (W₃.map L₁)
                                               x✝¹ x✝ : CategoryTheory.Functor C₃ E
                                               h : Eq ((L₁.comp L₂).comp x✝¹) ((L₁.comp L₂).comp x✝)
                                               ⊢ Eq (L₁.comp (L₂.comp x✝¹)) (L₁.comp (L₂.comp x✝))
                                             -/
  uniq _ _ h := h₂.uniq _ _ (h₁.uniq _ _ (by simpa only [Functor.assoc] using h))
                                             /-
                                               🎉 no goals
                                             -/


lemma comp [L₁.IsLocalization W₁] [L₂.IsLocalization W₂]
    (W₃ : MorphismProperty C₁) (hW₃ : W₃.IsInvertedBy (L₁ ⋙ L₂))
    (hW₁₃ : W₁ ≤ W₃) (hW₂₃ : W₂ ≤ W₃.map L₁) :
    (L₁ ⋙ L₂).IsLocalization W₃ := by
  -- The proof proceeds by reducing to the case of the constructed
  -- localized categories, which satisfy the strict universal property
  -- of the localization. In order to do this, we introduce
  -- an equivalence of categories `E₂ : C₂ ≅ W₁.Localization`. Via
  -- this equivalence, we introduce `W₂' : MorphismProperty W₁.Localization`
  -- which corresponds to `W₂` via the equivalence `E₂`.
  -- Then, we have a localizer morphism `Φ : LocalizerMorphism W₂ W₂'` which
  -- is a localized equivalence (because `E₂` is an equivalence).
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝² : CategoryTheory.Category.{v₃, u₃} C₃
    L₁ : CategoryTheory.Functor C₁ C₂
    L₂ : CategoryTheory.Functor C₂ C₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝¹ : L₁.IsLocalization W₁
    inst✝ : L₂.IsLocalization W₂
    W₃ : CategoryTheory.MorphismProperty C₁
    hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
    hW₁₃ : LE.le W₁ W₃
    hW₂₃ : LE.le W₂ (W₃.map L₁)
    ⊢ (L₁.comp L₂).IsLocalization W₃
  -/
  let E₂ := Localization.uniq L₁ W₁.Q W₁
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝² : CategoryTheory.Category.{v₃, u₃} C₃
    L₁ : CategoryTheory.Functor C₁ C₂
    L₂ : CategoryTheory.Functor C₂ C₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝¹ : L₁.IsLocalization W₁
    inst✝ : L₂.IsLocalization W₂
    W₃ : CategoryTheory.MorphismProperty C₁
    hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
    hW₁₃ : LE.le W₁ W₃
    hW₂₃ : LE.le W₂ (W₃.map L₁)
    E₂ : CategoryTheory.Equivalence C₂ W₁.Localization := CategoryTheory.Localizat …
    ⊢ (L₁.comp L₂).IsLocalization W₃
  -/
  let W₂' := W₂.map E₂.functor
  let Φ : LocalizerMorphism W₂ W₂' :=
    { functor := E₂.functor
      map := by
        have eq := W₂.isoClosure.inverseImage_map_eq_of_isEquivalence E₂.functor
        rw [MorphismProperty.map_isoClosure] at eq
        rw [eq]
        apply W₂.le_isoClosure }
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝² : CategoryTheory.Category.{v₃, u₃} C₃
    L₁ : CategoryTheory.Functor C₁ C₂
    L₂ : CategoryTheory.Functor C₂ C₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝¹ : L₁.IsLocalization W₁
    inst✝ : L₂.IsLocalization W₂
    W₃ : CategoryTheory.MorphismProperty C₁
    hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
    hW₁₃ : LE.le W₁ W₃
    hW₂₃ : LE.le W₂ (W₃.map L₁)
    E₂ : CategoryTheory.Equivalence C₂ W₁.Localization := CategoryTheory.Localizat …
    W₂' : CategoryTheory.MorphismProperty W₁.Localization := W₂.map E₂.functor
    Φ : CategoryTheory.LocalizerMorphism W₂ W₂' := { functor := E₂.functor, map := …
    ⊢ (L₁.comp L₂).IsLocalization W₃
  -/
  have := LocalizerMorphism.IsLocalizedEquivalence.of_equivalence Φ (by rfl)
  -- The fact that `Φ` is a localized equivalence allows to consider
  -- the induced equivalence of categories `E₃ : C₃ ≅ W₂'.Localization`, and
  -- the isomorphism `iso : (W₁.Q ⋙ W₂'.Q) ⋙ E₃.inverse ≅ L₁ ⋙ L₂`
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝² : CategoryTheory.Category.{v₃, u₃} C₃
    L₁ : CategoryTheory.Functor C₁ C₂
    L₂ : CategoryTheory.Functor C₂ C₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝¹ : L₁.IsLocalization W₁
    inst✝ : L₂.IsLocalization W₂
    W₃ : CategoryTheory.MorphismProperty C₁
    hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
    hW₁₃ : LE.le W₁ W₃
    hW₂₃ : LE.le W₂ (W₃.map L₁)
    E₂ : CategoryTheory.Equivalence C₂ W₁.Localization := CategoryTheory.Localizat …
    W₂' : CategoryTheory.MorphismProperty W₁.Localization := W₂.map E₂.functor
    Φ : CategoryTheory.LocalizerMorphism W₂ W₂' := { functor := E₂.functor, map := …
    this : Φ.IsLocalizedEquivalence
    ⊢ (L₁.comp L₂).IsLocalization W₃
  -/
  let E₃ := (Φ.localizedFunctor L₂ W₂'.Q).asEquivalence
  let iso : (W₁.Q ⋙ W₂'.Q) ⋙ E₃.inverse ≅ L₁ ⋙ L₂ := by
    calc
      _ ≅ L₁ ⋙ E₂.functor ⋙ W₂'.Q ⋙ E₃.inverse :=
          Functor.associator _ _ _ ≪≫ isoWhiskerRight (compUniqFunctor L₁ W₁.Q W₁).symm _ ≪≫
            Functor.associator _ _ _
      _ ≅ L₁ ⋙ L₂ ⋙ E₃.functor ⋙ E₃.inverse :=
          isoWhiskerLeft _ ((Functor.associator _ _ _).symm ≪≫
            isoWhiskerRight (Φ.catCommSq L₂ W₂'.Q).iso E₃.inverse ≪≫ Functor.associator _ _ _)
      _ ≅ L₁ ⋙ L₂ := isoWhiskerLeft _ (isoWhiskerLeft _ E₃.unitIso.symm ≪≫ L₂.rightUnitor)
  -- In order to show `(W₁.Q ⋙ W₂'.Q).IsLocalization W₃`, we need
  -- to check the assumptions of `StrictUniversalPropertyFixedTarget.comp`
  have hW₃' : W₃.IsInvertedBy (W₁.Q ⋙ W₂'.Q) := by
    simpa only [← MorphismProperty.IsInvertedBy.iff_comp _ _ E₃.inverse,
      MorphismProperty.IsInvertedBy.iff_of_iso W₃ iso] using hW₃
  have hW₂₃' : W₂' ≤ W₃.map W₁.Q := (MorphismProperty.monotone_map E₂.functor hW₂₃).trans
    (by simpa only [W₃.map_map]
      using le_of_eq (W₃.map_eq_of_iso (compUniqFunctor L₁ W₁.Q W₁)))
  have : (W₁.Q ⋙ W₂'.Q).IsLocalization W₃ := by
    refine IsLocalization.mk' _ _ ?_ ?_
    all_goals
      exact (StrictUniversalPropertyFixedTarget.comp
        (strictUniversalPropertyFixedTargetQ W₁ _)
        (strictUniversalPropertyFixedTargetQ W₂' _) W₃ hW₃' hW₁₃ hW₂₃')
  -- Finally, the previous result can be transported via the equivalence `E₃`
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝² : CategoryTheory.Category.{v₃, u₃} C₃
    L₁ : CategoryTheory.Functor C₁ C₂
    L₂ : CategoryTheory.Functor C₂ C₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝¹ : L₁.IsLocalization W₁
    inst✝ : L₂.IsLocalization W₂
    W₃ : CategoryTheory.MorphismProperty C₁
    hW₃ : W₃.IsInvertedBy (L₁.comp L₂)
    hW₁₃ : LE.le W₁ W₃
    hW₂₃ : LE.le W₂ (W₃.map L₁)
    E₂ : CategoryTheory.Equivalence C₂ W₁.Localization := CategoryTheory.Localizat …
    W₂' : CategoryTheory.MorphismProperty W₁.Localization := W₂.map E₂.functor
    Φ : CategoryTheory.LocalizerMorphism W₂ W₂' := { functor := E₂.functor, map := …
    this✝ : Φ.IsLocalizedEquivalence
    E₃ : CategoryTheory.Equivalence C₃ W₂'.Localization := (Φ.localizedFunctor L₂  …
    iso : CategoryTheory.Iso ((W₁.Q.comp W₂'.Q).comp E₃.inverse) (L₁.comp L₂) := T …
    hW₃' : W₃.IsInvertedBy (W₁.Q.comp W₂'.Q)
    hW₂₃' : LE.le W₂' (W₃.map W₁.Q)
    this : (W₁.Q.comp W₂'.Q).IsLocalization W₃
    ⊢ (L₁.comp L₂).IsLocalization W₃
  -/
  exact IsLocalization.of_equivalence_target _ W₃ _ E₃.symm iso
  /-
    🎉 no goals
  -/


lemma of_comp (W₃ : MorphismProperty C₁)
    [L₁.IsLocalization W₁] [(L₁ ⋙ L₂).IsLocalization W₃]
    (hW₁₃ : W₁ ≤ W₃) (hW₂₃ : W₂ = W₃.map L₁) :
    L₂.IsLocalization W₂ := by
    have : (L₁ ⋙ W₂.Q).IsLocalization W₃ :=
      comp L₁ W₂.Q W₁ W₂ W₃ (fun X Y f hf => Localization.inverts W₂.Q W₂ _
        (by simpa only [hW₂₃] using W₃.map_mem_map _ _ hf)) hW₁₃
        (by rw [hW₂₃])
    exact IsLocalization.of_equivalence_target W₂.Q W₂ L₂
      (Localization.uniq (L₁ ⋙ W₂.Q) (L₁ ⋙ L₂) W₃)
      (liftNatIso L₁ W₁ _ _ _ _
        ((Functor.associator _ _ _).symm ≪≫
          Localization.compUniqFunctor (L₁ ⋙ W₂.Q) (L₁ ⋙ L₂) W₃))


