/-- A localizer morphism `Φ : LocalizerMorphism W₁ W₂` is a right derivability
structure if it has right resolutions and the 2-square where the left and right functors
are localizations functors for `W₁` and `W₂` are Guitart exact. -/
class IsRightDerivabilityStructure : Prop where
  hasRightResolutions : Φ.HasRightResolutions := by infer_instance
  guitartExact' : TwoSquare.GuitartExact ((Φ.catCommSq W₁.Q W₂.Q).iso).hom


lemma isRightDerivabilityStructure_iff [Φ.HasRightResolutions] (e : Φ.functor ⋙ L₂ ≅ L₁ ⋙ F) :
    Φ.IsRightDerivabilityStructure ↔ TwoSquare.GuitartExact e.hom := by
  have : Φ.IsRightDerivabilityStructure ↔
      TwoSquare.GuitartExact ((Φ.catCommSq W₁.Q W₂.Q).iso).hom :=
    ⟨fun h => h.guitartExact', fun h => ⟨inferInstance, h⟩⟩
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D₁
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : Φ.HasRightResolutions
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp F)
    this : Iff Φ.IsRightDerivabilityStructure (CategoryTheory.TwoSquare.GuitartExa …
    ⊢ Iff Φ.IsRightDerivabilityStructure (CategoryTheory.TwoSquare.GuitartExact e. …
  -/
  rw [this]
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D₁
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : Φ.HasRightResolutions
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp F)
    this : Iff Φ.IsRightDerivabilityStructure (CategoryTheory.TwoSquare.GuitartExa …
    ⊢ Iff (CategoryTheory.TwoSquare.GuitartExact (CategoryTheory.CatCommSq.iso Φ.f …
  -/
  let e' := (Φ.catCommSq W₁.Q W₂.Q).iso
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D₁
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : Φ.HasRightResolutions
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp F)
    this : Iff Φ.IsRightDerivabilityStructure (CategoryTheory.TwoSquare.GuitartExa …
    e' : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp (Φ.localizedFunctor W …
    ⊢ Iff (CategoryTheory.TwoSquare.GuitartExact (CategoryTheory.CatCommSq.iso Φ.f …
  -/
  let E₁ := Localization.uniq W₁.Q L₁ W₁
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D₁
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : Φ.HasRightResolutions
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp F)
    this : Iff Φ.IsRightDerivabilityStructure (CategoryTheory.TwoSquare.GuitartExa …
    e' : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp (Φ.localizedFunctor W …
    E₁ : CategoryTheory.Equivalence W₁.Localization D₁ := CategoryTheory.Localizat …
    ⊢ Iff (CategoryTheory.TwoSquare.GuitartExact (CategoryTheory.CatCommSq.iso Φ.f …
  -/
  let E₂ := Localization.uniq W₂.Q L₂ W₂
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D₁
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : Φ.HasRightResolutions
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp F)
    this : Iff Φ.IsRightDerivabilityStructure (CategoryTheory.TwoSquare.GuitartExa …
    e' : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp (Φ.localizedFunctor W …
    E₁ : CategoryTheory.Equivalence W₁.Localization D₁ := CategoryTheory.Localizat …
    E₂ : CategoryTheory.Equivalence W₂.Localization D₂ := CategoryTheory.Localizat …
    ⊢ Iff (CategoryTheory.TwoSquare.GuitartExact (CategoryTheory.CatCommSq.iso Φ.f …
  -/
  let e₁ : W₁.Q ⋙ E₁.functor ≅ L₁ := compUniqFunctor W₁.Q L₁ W₁
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D₁
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : Φ.HasRightResolutions
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp F)
    this : Iff Φ.IsRightDerivabilityStructure (CategoryTheory.TwoSquare.GuitartExa …
    e' : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp (Φ.localizedFunctor W …
    E₁ : CategoryTheory.Equivalence W₁.Localization D₁ := CategoryTheory.Localizat …
    E₂ : CategoryTheory.Equivalence W₂.Localization D₂ := CategoryTheory.Localizat …
    e₁ : CategoryTheory.Iso (W₁.Q.comp E₁.functor) L₁ := CategoryTheory.Localizati …
    ⊢ Iff (CategoryTheory.TwoSquare.GuitartExact (CategoryTheory.CatCommSq.iso Φ.f …
  -/
  let e₂ : W₂.Q ⋙ E₂.functor ≅ L₂ := compUniqFunctor W₂.Q L₂ W₂
  let e'' : (Φ.functor ⋙ W₂.Q) ⋙ E₂.functor ≅ (W₁.Q ⋙ E₁.functor) ⋙ F :=
    Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ e₂ ≪≫ e ≪≫ isoWhiskerRight e₁.symm F
  let e''' : Φ.localizedFunctor W₁.Q W₂.Q ⋙ E₂.functor ≅ E₁.functor ⋙ F :=
    liftNatIso W₁.Q W₁ _ _ _ _ e''
  have : TwoSquare.vComp' e'.hom e'''.hom e₁ e₂ = e.hom := by
    ext X₁
    rw [TwoSquare.vComp'_app, liftNatIso_hom, liftNatTrans_app]
    simp only [Functor.comp_obj, Iso.trans_hom, isoWhiskerLeft_hom, isoWhiskerRight_hom,
      Iso.symm_hom, NatTrans.comp_app, Functor.associator_hom_app, whiskerLeft_app,
      whiskerRight_app, id_comp, assoc, e'']
    dsimp [Lifting.iso]
    rw [F.map_id, id_comp, ← F.map_comp, Iso.inv_hom_id_app, F.map_id, comp_id,
      ← Functor.map_comp_assoc]
    erw [show (CatCommSq.iso Φ.functor W₁.Q W₂.Q (localizedFunctor Φ W₁.Q W₂.Q)).hom =
      (Lifting.iso W₁.Q W₁ _ _).inv by rfl, Iso.inv_hom_id_app]
    simp
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} D₁
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : Φ.HasRightResolutions
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp F)
    this✝ : Iff Φ.IsRightDerivabilityStructure (CategoryTheory.TwoSquare.GuitartEx …
    e' : CategoryTheory.Iso (Φ.functor.comp W₂.Q) (W₁.Q.comp (Φ.localizedFunctor W …
    E₁ : CategoryTheory.Equivalence W₁.Localization D₁ := CategoryTheory.Localizat …
    E₂ : CategoryTheory.Equivalence W₂.Localization D₂ := CategoryTheory.Localizat …
    e₁ : CategoryTheory.Iso (W₁.Q.comp E₁.functor) L₁ := CategoryTheory.Localizati …
    e₂ : CategoryTheory.Iso (W₂.Q.comp E₂.functor) L₂ := CategoryTheory.Localizati …
    e'' : CategoryTheory.Iso ((Φ.functor.comp W₂.Q).comp E₂.functor) ((W₁.Q.comp E …
    e''' : CategoryTheory.Iso ((Φ.localizedFunctor W₁.Q W₂.Q).comp E₂.functor) (E₁ …
    this : Eq (CategoryTheory.TwoSquare.vComp' e'.hom e'''.hom e₁ e₂) e.hom
    ⊢ Iff (CategoryTheory.TwoSquare.GuitartExact (CategoryTheory.CatCommSq.iso Φ.f …
  -/
  rw [← TwoSquare.GuitartExact.vComp'_iff_of_equivalences e'.hom E₁ E₂ e''' e₁ e₂, this]
  /-
    🎉 no goals
  -/


lemma guitartExact_of_isRightDerivabilityStructure' [h : Φ.IsRightDerivabilityStructure]
    (e : Φ.functor ⋙ L₂ ≅ L₁ ⋙ F) : TwoSquare.GuitartExact e.hom := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} D₁
    inst✝² : CategoryTheory.Category.{u_3, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₁.IsLocalization W₁
    inst✝ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    h : Φ.IsRightDerivabilityStructure
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp F)
    ⊢ CategoryTheory.TwoSquare.GuitartExact e.hom
  -/
  simpa only [Φ.isRightDerivabilityStructure_iff L₁ L₂ F e] using h
  /-
    🎉 no goals
  -/


lemma guitartExact_of_isRightDerivabilityStructure [Φ.IsRightDerivabilityStructure] :
    TwoSquare.GuitartExact ((Φ.catCommSq L₁ L₂).iso).hom :=
  guitartExact_of_isRightDerivabilityStructure' _ _ _ _ _


instance [W₁.ContainsIdentities] : (LocalizerMorphism.id W₁).HasRightResolutions :=
  fun X₂ => ⟨RightResolution.mk (𝟙 X₂) (W₁.id_mem X₂)⟩


instance [W₁.ContainsIdentities] : (LocalizerMorphism.id W₁).IsRightDerivabilityStructure := by
  rw [(LocalizerMorphism.id W₁).isRightDerivabilityStructure_iff W₁.Q W₁.Q (𝟭 W₁.Localization)
    (Iso.refl _)]
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.38472, u_1} D₁
    inst✝³ : CategoryTheory.Category.{?u.38476, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : W₁.ContainsIdentities
    ⊢ CategoryTheory.TwoSquare.GuitartExact (CategoryTheory.Iso.refl ((CategoryThe …
  -/
  dsimp
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    D₁ : Type u_1
    D₂ : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.38472, u_1} D₁
    inst✝³ : CategoryTheory.Category.{?u.38476, u_2} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝² : L₁.IsLocalization W₁
    inst✝¹ : L₂.IsLocalization W₂
    F : CategoryTheory.Functor D₁ D₂
    inst✝ : W₁.ContainsIdentities
    ⊢ CategoryTheory.TwoSquare.GuitartExact (CategoryTheory.CategoryStruct.id ((Ca …
  -/
  exact TwoSquare.guitartExact_id W₁.Q
  /-
    🎉 no goals
  -/


