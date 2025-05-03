/-- If `W₁ : MorphismProperty C₁` and `W₂ : MorphismProperty C₂`, a `LocalizerMorphism W₁ W₂`
is the datum of a functor `C₁ ⥤ C₂` which sends morphisms in `W₁` to morphisms in `W₂` -/
structure LocalizerMorphism where
  /-- a functor between the two categories -/
  functor : C₁ ⥤ C₂
  /-- the functor is compatible with the `MorphismProperty` -/
  map : W₁ ≤ W₂.inverseImage functor


/-- The identity functor as a morphism of localizers. -/
@[simps]
def id : LocalizerMorphism W₁ W₁ where
  functor := 𝟭 C₁
  map _ _ _ hf := hf


/-- The composition of two localizers morphisms. -/
@[simps]
def comp (Φ : LocalizerMorphism W₁ W₂) (Ψ : LocalizerMorphism W₂ W₃) :
    LocalizerMorphism W₁ W₃ where
  functor := Φ.functor ⋙ Ψ.functor
  map _ _ _ hf := Ψ.map _ (Φ.map _ hf)


/-- The opposite localizer morphism `LocalizerMorphism W₁.op W₂.op` deduced
from `Φ : LocalizerMorphism W₁ W₂`. -/
@[simps]
def op : LocalizerMorphism W₁.op W₂.op where
  functor := Φ.functor.op
  map _ _ _ hf := Φ.map _ hf


lemma inverts : W₁.IsInvertedBy (Φ.functor ⋙ L₂) :=
  fun _ _ _ hf => Localization.inverts L₂ W₂ _ (Φ.map _ hf)


/-- When `Φ : LocalizerMorphism W₁ W₂` and that `L₁` and `L₂` are localization functors
for `W₁` and `W₂`, then `Φ.localizedFunctor L₁ L₂` is the induced functor on the
localized categories. --/
noncomputable def localizedFunctor : D₁ ⥤ D₂ :=
  lift (Φ.functor ⋙ L₂) (Φ.inverts _) L₁


noncomputable instance liftingLocalizedFunctor :
    Lifting L₁ W₁ (Φ.functor ⋙ L₂) (Φ.localizedFunctor L₁ L₂) := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    D₁ : Type u₄
    D₂ : Type u₅
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝³ : CategoryTheory.Category.{v₄, u₄} D₁
    inst✝² : CategoryTheory.Category.{v₅, u₅} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    W₃ : CategoryTheory.MorphismProperty C₃
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    ⊢ CategoryTheory.Localization.Lifting L₁ W₁ (Φ.functor.comp L₂) (Φ.localizedFu …
  -/
  dsimp [localizedFunctor]
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    D₁ : Type u₄
    D₂ : Type u₅
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝³ : CategoryTheory.Category.{v₄, u₄} D₁
    inst✝² : CategoryTheory.Category.{v₅, u₅} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    W₃ : CategoryTheory.MorphismProperty C₃
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    ⊢ CategoryTheory.Localization.Lifting L₁ W₁ (Φ.functor.comp L₂) (CategoryTheor …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The 2-commutative square expressing that `Φ.localizedFunctor L₁ L₂` lifts the
functor `Φ.functor`  -/
noncomputable instance catCommSq : CatCommSq Φ.functor L₁ L₂ (Φ.localizedFunctor L₁ L₂) :=
  CatCommSq.mk (Lifting.iso _ W₁ _ _).symm


/-- If a localizer morphism induces an equivalence on some choice of localized categories,
it will be so for any choice of localized categoriees. -/
lemma isEquivalence_imp [G.IsEquivalence] : G'.IsEquivalence :=
  let E₁ := Localization.uniq L₁ L₁' W₁
  let E₂ := Localization.uniq L₂ L₂' W₂
  let e : L₁ ⋙ G ⋙ E₂.functor ≅ L₁ ⋙ E₁.functor ⋙ G' :=
    calc
      L₁ ⋙ G ⋙ E₂.functor ≅ Φ.functor ⋙ L₂ ⋙ E₂.functor :=
          (Functor.associator _ _ _).symm ≪≫
            isoWhiskerRight (CatCommSq.iso Φ.functor L₁ L₂ G).symm E₂.functor ≪≫
            Functor.associator _ _ _
      _ ≅ Φ.functor ⋙ L₂' := isoWhiskerLeft Φ.functor (compUniqFunctor L₂ L₂' W₂)
      _ ≅ L₁' ⋙ G' := CatCommSq.iso Φ.functor L₁' L₂' G'
      _ ≅ L₁ ⋙ E₁.functor ⋙ G' :=
            isoWhiskerRight (compUniqFunctor L₁ L₁' W₁).symm G' ≪≫ Functor.associator _ _ _
  have := Functor.isEquivalence_of_iso
    (liftNatIso L₁ W₁ _ _ (G ⋙ E₂.functor) (E₁.functor ⋙ G') e)
  Functor.isEquivalence_of_comp_left E₁.functor G'


lemma isEquivalence_iff : G.IsEquivalence ↔ G'.IsEquivalence :=
  ⟨fun _ => Φ.isEquivalence_imp L₁ L₂ G L₁' L₂' G',
    fun _ => Φ.isEquivalence_imp L₁' L₂' G' L₁ L₂ G⟩


/-- Condition that a `LocalizerMorphism` induces an equivalence on the localized categories -/
class IsLocalizedEquivalence : Prop where
  /-- the induced functor on the constructed localized categories is an equivalence -/
  isEquivalence : (Φ.localizedFunctor W₁.Q W₂.Q).IsEquivalence


lemma IsLocalizedEquivalence.mk' [CatCommSq Φ.functor L₁ L₂ G] [G.IsEquivalence] :
    Φ.IsLocalizedEquivalence where
  isEquivalence := by
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      D₁ : Type u₄
      D₂ : Type u₅
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D₁
      inst✝⁴ : CategoryTheory.Category.{v₅, u₅} D₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      L₁ : CategoryTheory.Functor C₁ D₁
      inst✝³ : L₁.IsLocalization W₁
      L₂ : CategoryTheory.Functor C₂ D₂
      inst✝² : L₂.IsLocalization W₂
      G : CategoryTheory.Functor D₁ D₂
      inst✝¹ : CategoryTheory.CatCommSq Φ.functor L₁ L₂ G
      inst✝ : G.IsEquivalence
      ⊢ (Φ.localizedFunctor W₁.Q W₂.Q).IsEquivalence
    -/
    rw [Φ.isEquivalence_iff W₁.Q W₂.Q (Φ.localizedFunctor W₁.Q W₂.Q) L₁ L₂ G]
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      D₁ : Type u₄
      D₂ : Type u₅
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D₁
      inst✝⁴ : CategoryTheory.Category.{v₅, u₅} D₂
      W₁ : CategoryTheory.MorphismProperty C₁
      W₂ : CategoryTheory.MorphismProperty C₂
      Φ : CategoryTheory.LocalizerMorphism W₁ W₂
      L₁ : CategoryTheory.Functor C₁ D₁
      inst✝³ : L₁.IsLocalization W₁
      L₂ : CategoryTheory.Functor C₂ D₂
      inst✝² : L₂.IsLocalization W₂
      G : CategoryTheory.Functor D₁ D₂
      inst✝¹ : CategoryTheory.CatCommSq Φ.functor L₁ L₂ G
      inst✝ : G.IsEquivalence
      ⊢ G.IsEquivalence
    -/
    exact inferInstance
    /-
      🎉 no goals
    -/


/-- If a `LocalizerMorphism` is a localized equivalence, then any compatible functor
between the localized categories is an equivalence. -/
lemma isEquivalence [h : Φ.IsLocalizedEquivalence] [CatCommSq Φ.functor L₁ L₂ G] :
    G.IsEquivalence := (by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    D₁ : Type u₄
    D₂ : Type u₅
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₁
    inst✝³ : CategoryTheory.Category.{v₅, u₅} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝² : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    G : CategoryTheory.Functor D₁ D₂
    h : Φ.IsLocalizedEquivalence
    inst✝ : CategoryTheory.CatCommSq Φ.functor L₁ L₂ G
    ⊢ G.IsEquivalence
  -/
  rw [Φ.isEquivalence_iff L₁ L₂ G W₁.Q W₂.Q (Φ.localizedFunctor W₁.Q W₂.Q)]
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    D₁ : Type u₄
    D₂ : Type u₅
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} D₁
    inst✝³ : CategoryTheory.Category.{v₅, u₅} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝² : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    G : CategoryTheory.Functor D₁ D₂
    h : Φ.IsLocalizedEquivalence
    inst✝ : CategoryTheory.CatCommSq Φ.functor L₁ L₂ G
    ⊢ (Φ.localizedFunctor W₁.Q W₂.Q).IsEquivalence
  -/
  exact h.isEquivalence)
  /-
    🎉 no goals
  -/


/-- If a `LocalizerMorphism` is a localized equivalence, then the induced functor on
the localized categories is an equivalence -/
instance localizedFunctor_isEquivalence [Φ.IsLocalizedEquivalence] :
    (Φ.localizedFunctor L₁ L₂).IsEquivalence :=
  Φ.isEquivalence L₁ L₂ _


/-- When `Φ : LocalizerMorphism W₁ W₂`, if the composition `Φ.functor ⋙ L₂` is a
localization functor for `W₁`, then `Φ` is a localized equivalence. -/
lemma IsLocalizedEquivalence.of_isLocalization_of_isLocalization
    [(Φ.functor ⋙ L₂).IsLocalization W₁] :
    IsLocalizedEquivalence Φ := by
  have : CatCommSq Φ.functor (Φ.functor ⋙ L₂) L₂ (𝟭 D₂) :=
    CatCommSq.mk (Functor.rightUnitor _).symm
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    D₂ : Type u₅
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝³ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝² : CategoryTheory.Category.{v₅, u₅} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    inst✝ : (Φ.functor.comp L₂).IsLocalization W₁
    this : CategoryTheory.CatCommSq Φ.functor (Φ.functor.comp L₂) L₂ (CategoryTheo …
    ⊢ Φ.IsLocalizedEquivalence
  -/
  exact IsLocalizedEquivalence.mk' Φ (Φ.functor ⋙ L₂) L₂ (𝟭 D₂)
  /-
    🎉 no goals
  -/


/-- When the underlying functor `Φ.functor` of `Φ : LocalizerMorphism W₁ W₂` is
an equivalence of categories and that `W₁` and `W₂` essentially correspond to each
other via this equivalence, then `Φ` is a localized equivalence. -/
lemma IsLocalizedEquivalence.of_equivalence [Φ.functor.IsEquivalence]
    (h : W₂ ≤ W₁.map Φ.functor) : IsLocalizedEquivalence Φ := by
  haveI : Functor.IsLocalization (Φ.functor ⋙ MorphismProperty.Q W₂) W₁ := by
    refine Functor.IsLocalization.of_equivalence_source W₂.Q W₂ (Φ.functor ⋙ W₂.Q) W₁
      (Functor.asEquivalence Φ.functor).symm ?_ (Φ.inverts W₂.Q)
      ((Functor.associator _ _ _).symm ≪≫ isoWhiskerRight ((Equivalence.unitIso _).symm) _ ≪≫
        Functor.leftUnitor _)
    erw [W₁.isoClosure.inverseImage_equivalence_functor_eq_map_inverse]
    rw [MorphismProperty.map_isoClosure]
    exact h
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    inst✝ : Φ.functor.IsEquivalence
    h : LE.le W₂ (W₁.map Φ.functor)
    this : (Φ.functor.comp W₂.Q).IsLocalization W₁
    ⊢ Φ.IsLocalizedEquivalence
  -/
  exact IsLocalizedEquivalence.of_isLocalization_of_isLocalization Φ W₂.Q
  /-
    🎉 no goals
  -/


instance IsLocalizedEquivalence.isLocalization [Φ.IsLocalizedEquivalence] :
    (Φ.functor ⋙ L₂).IsLocalization W₁ :=
  Functor.IsLocalization.of_iso _ ((Φ.catCommSq W₁.Q L₂).iso).symm


/-- The localizer morphism from `W₁.arrow` to `W₂.arrow` that is induced by
`Φ : LocalizerMorphism W₁ W₂`. -/
@[simps]
def arrow : LocalizerMorphism W₁.arrow W₂.arrow where
  functor := Φ.functor.mapArrow
  map _ _ _ hf := ⟨Φ.map _ hf.1, Φ.map _ hf.2⟩


