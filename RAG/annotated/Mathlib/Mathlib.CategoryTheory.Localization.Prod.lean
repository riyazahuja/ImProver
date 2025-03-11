lemma prod_uniq (F₁ F₂ : (W₁.Localization × W₂.Localization ⥤ E))
    (h : (W₁.Q.prod W₂.Q) ⋙ F₁ = (W₁.Q.prod W₂.Q) ⋙ F₂) :
      F₁ = F₂ := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝ : CategoryTheory.Category.{v₅, u₅} E
    F₁ F₂ : CategoryTheory.Functor (Prod W₁.Localization W₂.Localization) E
    h : Eq ((W₁.Q.prod W₂.Q).comp F₁) ((W₁.Q.prod W₂.Q).comp F₂)
    ⊢ Eq F₁ F₂
  -/
  apply Functor.curry_obj_injective
  /-
    case h
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝ : CategoryTheory.Category.{v₅, u₅} E
    F₁ F₂ : CategoryTheory.Functor (Prod W₁.Localization W₂.Localization) E
    h : Eq ((W₁.Q.prod W₂.Q).comp F₁) ((W₁.Q.prod W₂.Q).comp F₂)
    ⊢ Eq (CategoryTheory.curry.obj F₁) (CategoryTheory.curry.obj F₂)
  -/
  apply Construction.uniq
  /-
    case h.h
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝ : CategoryTheory.Category.{v₅, u₅} E
    F₁ F₂ : CategoryTheory.Functor (Prod W₁.Localization W₂.Localization) E
    h : Eq ((W₁.Q.prod W₂.Q).comp F₁) ((W₁.Q.prod W₂.Q).comp F₂)
    ⊢ Eq (W₁.Q.comp (CategoryTheory.curry.obj F₁)) (W₁.Q.comp (CategoryTheory.curr …
  -/
  apply Functor.flip_injective
  /-
    case h.h.h
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝ : CategoryTheory.Category.{v₅, u₅} E
    F₁ F₂ : CategoryTheory.Functor (Prod W₁.Localization W₂.Localization) E
    h : Eq ((W₁.Q.prod W₂.Q).comp F₁) ((W₁.Q.prod W₂.Q).comp F₂)
    ⊢ Eq (W₁.Q.comp (CategoryTheory.curry.obj F₁)).flip (W₁.Q.comp (CategoryTheory …
  -/
  apply Construction.uniq
  /-
    case h.h.h.h
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝ : CategoryTheory.Category.{v₅, u₅} E
    F₁ F₂ : CategoryTheory.Functor (Prod W₁.Localization W₂.Localization) E
    h : Eq ((W₁.Q.prod W₂.Q).comp F₁) ((W₁.Q.prod W₂.Q).comp F₂)
    ⊢ Eq (W₂.Q.comp (W₁.Q.comp (CategoryTheory.curry.obj F₁)).flip) (W₂.Q.comp (W₁ …
  -/
  apply Functor.flip_injective
  /-
    case h.h.h.h.h
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝ : CategoryTheory.Category.{v₅, u₅} E
    F₁ F₂ : CategoryTheory.Functor (Prod W₁.Localization W₂.Localization) E
    h : Eq ((W₁.Q.prod W₂.Q).comp F₁) ((W₁.Q.prod W₂.Q).comp F₂)
    ⊢ Eq (W₂.Q.comp (W₁.Q.comp (CategoryTheory.curry.obj F₁)).flip).flip (W₂.Q.com …
  -/
  apply Functor.uncurry_obj_injective
  /-
    case h.h.h.h.h.h
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝ : CategoryTheory.Category.{v₅, u₅} E
    F₁ F₂ : CategoryTheory.Functor (Prod W₁.Localization W₂.Localization) E
    h : Eq ((W₁.Q.prod W₂.Q).comp F₁) ((W₁.Q.prod W₂.Q).comp F₂)
    ⊢ Eq (CategoryTheory.uncurry.obj (W₂.Q.comp (W₁.Q.comp (CategoryTheory.curry.o …
  -/
  simpa only [Functor.uncurry_obj_curry_obj_flip_flip] using h
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `prodLift`. -/
noncomputable def prodLift₁ [W₂.ContainsIdentities]
    (hF : (W₁.prod W₂).IsInvertedBy F) :
    W₁.Localization ⥤ C₂ ⥤ E :=
  Construction.lift (curry.obj F) (fun _ _ f₁ hf₁ => by
    haveI : ∀ (X₂ : C₂), IsIso (((curry.obj F).map f₁).app X₂) :=
      fun X₂ => hF _ ⟨hf₁, MorphismProperty.id_mem _ _⟩
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      D₁ : Type u₃
      D₂ : Type u₄
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝³ : CategoryTheory.Category.{v₃, u₃} D₁
      inst✝² : CategoryTheory.Category.{v₄, u₄} D₂
      L₁ : CategoryTheory.Functor C₁ D₁
      W₁ : CategoryTheory.MorphismProperty C₁
      L₂ : CategoryTheory.Functor C₂ D₂
      W₂ : CategoryTheory.MorphismProperty C₂
      E : Type u₅
      inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
      F : CategoryTheory.Functor (Prod C₁ C₂) E
      inst✝ : W₂.ContainsIdentities
      hF : (W₁.prod W₂).IsInvertedBy F
      x✝¹ x✝ : C₁
      f₁ : Quiver.Hom x✝¹ x✝
      hf₁ : W₁ f₁
      this : ∀ (X₂ : C₂), CategoryTheory.IsIso (((CategoryTheory.curry.obj F).map f₁ …
      ⊢ CategoryTheory.IsIso ((CategoryTheory.curry.obj F).map f₁)
    -/
    apply NatIso.isIso_of_isIso_app)
    /-
      🎉 no goals
    -/


lemma prod_fac₁ [W₂.ContainsIdentities] :
    W₁.Q ⋙ prodLift₁ F hF = curry.obj F :=
  Construction.fac _ _


/-- The lifting of a functor `F : C₁ × C₂ ⥤ E` inverting `W₁.prod W₂` to a functor
`W₁.Localization × W₂.Localization ⥤ E` -/
noncomputable def prodLift :
    W₁.Localization × W₂.Localization ⥤ E := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    D₁ : Type u₃
    D₂ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} D₁
    inst✝³ : CategoryTheory.Category.{v₄, u₄} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝² : CategoryTheory.Category.{v₅, u₅} E
    F : CategoryTheory.Functor (Prod C₁ C₂) E
    hF : (W₁.prod W₂).IsInvertedBy F
    inst✝¹ : W₁.ContainsIdentities
    inst✝ : W₂.ContainsIdentities
    ⊢ CategoryTheory.Functor (Prod W₁.Localization W₂.Localization) E
  -/
  refine uncurry.obj (Construction.lift (prodLift₁ F hF).flip ?_).flip
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    D₁ : Type u₃
    D₂ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} D₁
    inst✝³ : CategoryTheory.Category.{v₄, u₄} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝² : CategoryTheory.Category.{v₅, u₅} E
    F : CategoryTheory.Functor (Prod C₁ C₂) E
    hF : (W₁.prod W₂).IsInvertedBy F
    inst✝¹ : W₁.ContainsIdentities
    inst✝ : W₂.ContainsIdentities
    ⊢ W₂.IsInvertedBy (CategoryTheory.Localization.StrictUniversalPropertyFixedTar …
  -/
  intro _ _ f₂ hf₂
  haveI : ∀ (X₁ : W₁.Localization),
      IsIso (((Functor.flip (prodLift₁ F hF)).map f₂).app X₁) := fun X₁ => by
    obtain ⟨X₁, rfl⟩ := (Construction.objEquiv W₁).surjective X₁
    exact ((MorphismProperty.isomorphisms E).arrow_mk_iso_iff
      (((Functor.mapArrowFunctor _ _).mapIso
        (eqToIso (Functor.congr_obj (prod_fac₁ F hF) X₁))).app (Arrow.mk f₂))).2
          (hF _ ⟨MorphismProperty.id_mem _ _, hf₂⟩)
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    D₁ : Type u₃
    D₂ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} D₁
    inst✝³ : CategoryTheory.Category.{v₄, u₄} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝² : CategoryTheory.Category.{v₅, u₅} E
    F : CategoryTheory.Functor (Prod C₁ C₂) E
    hF : (W₁.prod W₂).IsInvertedBy F
    inst✝¹ : W₁.ContainsIdentities
    inst✝ : W₂.ContainsIdentities
    X✝ Y✝ : C₂
    f₂ : Quiver.Hom X✝ Y✝
    hf₂ : W₂ f₂
    this : ∀ (X₁ : W₁.Localization), CategoryTheory.IsIso (((CategoryTheory.Locali …
    ⊢ CategoryTheory.IsIso ((CategoryTheory.Localization.StrictUniversalPropertyFi …
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


lemma prod_fac₂ :
    W₂.Q ⋙ (curry.obj (prodLift F hF)).flip = (prodLift₁ F hF).flip := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝² : CategoryTheory.Category.{v₅, u₅} E
    F : CategoryTheory.Functor (Prod C₁ C₂) E
    hF : (W₁.prod W₂).IsInvertedBy F
    inst✝¹ : W₁.ContainsIdentities
    inst✝ : W₂.ContainsIdentities
    ⊢ Eq (W₂.Q.comp (CategoryTheory.curry.obj (CategoryTheory.Localization.StrictU …
  -/
  simp only [prodLift, Functor.curry_obj_uncurry_obj, Functor.flip_flip]
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    E : Type u₅
    inst✝² : CategoryTheory.Category.{v₅, u₅} E
    F : CategoryTheory.Functor (Prod C₁ C₂) E
    hF : (W₁.prod W₂).IsInvertedBy F
    inst✝¹ : W₁.ContainsIdentities
    inst✝ : W₂.ContainsIdentities
    ⊢ Eq (W₂.Q.comp (CategoryTheory.Localization.Construction.lift (CategoryTheory …
  -/
  apply Construction.fac
  /-
    🎉 no goals
  -/


lemma prod_fac :
    (W₁.Q.prod W₂.Q) ⋙ prodLift F hF = F := by
  rw [← Functor.uncurry_obj_curry_obj_flip_flip', prod_fac₂, Functor.flip_flip, prod_fac₁,
    Functor.uncurry_obj_curry_obj]


/-- The product of two (constructed) localized categories satisfies the universal
property of the localized category of the product. -/
noncomputable def prod :
    StrictUniversalPropertyFixedTarget (W₁.Q.prod W₂.Q) (W₁.prod W₂) E where
  inverts := (Localization.inverts W₁.Q W₁).prod (Localization.inverts W₂.Q W₂)
  lift := fun F hF => prodLift F hF
  fac := fun F hF => prod_fac F hF
  uniq := prod_uniq


lemma Construction.prodIsLocalization :
    (W₁.Q.prod W₂.Q).IsLocalization (W₁.prod W₂) :=
  Functor.IsLocalization.mk' _ _
    (StrictUniversalPropertyFixedTarget.prod W₁ W₂)
    (StrictUniversalPropertyFixedTarget.prod W₁ W₂)


/-- If `L₁ : C₁ ⥤ D₁` and `L₂ : C₂ ⥤ D₂` are localization functors
for `W₁ : MorphismProperty C₁` and `W₂ : MorphismProperty C₂` respectively,
and if both `W₁` and `W₂` contain identities, then the product
functor `L₁.prod L₂ : C₁ × C₂ ⥤ D₁ × D₂` is a localization functor for `W₁.prod W₂`. -/
instance prod [L₁.IsLocalization W₁] [L₂.IsLocalization W₂] :
    (L₁.prod L₂).IsLocalization (W₁.prod W₂) := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    D₁ : Type u₃
    D₂ : Type u₄
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} D₁
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₂
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝³ : W₁.ContainsIdentities
    inst✝² : W₂.ContainsIdentities
    inst✝¹ : L₁.IsLocalization W₁
    inst✝ : L₂.IsLocalization W₂
    ⊢ (L₁.prod L₂).IsLocalization (W₁.prod W₂)
  -/
  haveI := Construction.prodIsLocalization W₁ W₂
  exact of_equivalence_target (W₁.Q.prod W₂.Q) (W₁.prod W₂) (L₁.prod L₂)
    ((uniq W₁.Q L₁ W₁).prod (uniq W₂.Q L₂ W₂))
    (NatIso.prod (compUniqFunctor W₁.Q L₁ W₁) (compUniqFunctor W₂.Q L₂ W₂))


