attribute [local instance] ConcreteCategory.instFunLike in
/-- In any concrete category, injective morphisms are monomorphisms. -/
theorem mono_of_injective {X Y : C} (f : X ⟶ Y) (i : Function.Injective f) :
    Mono f :=
  (forget C).mono_of_mono_map ((mono_iff_injective ((forget C).map f)).2 i)


instance forget₂_preservesMonomorphisms (C : Type u) (D : Type u')
    [Category.{v} C] [ConcreteCategory.{w} C] [Category.{v'} D] [ConcreteCategory.{w} D]
    [HasForget₂ C D] [(forget C).PreservesMonomorphisms] :
    (forget₂ C D).PreservesMonomorphisms :=
  have : (forget₂ C D ⋙ forget D).PreservesMonomorphisms := by
    /-
      C✝ : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C✝
      inst✝⁶ : CategoryTheory.ConcreteCategory C✝
      C : Type u
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Category.{v', u'} D
      inst✝² : CategoryTheory.ConcreteCategory D
      inst✝¹ : CategoryTheory.HasForget₂ C D
      inst✝ : (CategoryTheory.forget C).PreservesMonomorphisms
      ⊢ ((CategoryTheory.forget₂ C D).comp (CategoryTheory.forget D)).PreservesMonom …
    -/
    simp only [HasForget₂.forget_comp]
    /-
      C✝ : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C✝
      inst✝⁶ : CategoryTheory.ConcreteCategory C✝
      C : Type u
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Category.{v', u'} D
      inst✝² : CategoryTheory.ConcreteCategory D
      inst✝¹ : CategoryTheory.HasForget₂ C D
      inst✝ : (CategoryTheory.forget C).PreservesMonomorphisms
      ⊢ (CategoryTheory.forget C).PreservesMonomorphisms
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  Functor.preservesMonomorphisms_of_preserves_of_reflects _ (forget D)


instance forget₂_preservesEpimorphisms (C : Type u) (D : Type u')
    [Category.{v} C] [ConcreteCategory.{w} C] [Category.{v'} D] [ConcreteCategory.{w} D]
    [HasForget₂ C D] [(forget C).PreservesEpimorphisms] :
    (forget₂ C D).PreservesEpimorphisms :=
  have : (forget₂ C D ⋙ forget D).PreservesEpimorphisms := by
    /-
      C✝ : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C✝
      inst✝⁶ : CategoryTheory.ConcreteCategory C✝
      C : Type u
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Category.{v', u'} D
      inst✝² : CategoryTheory.ConcreteCategory D
      inst✝¹ : CategoryTheory.HasForget₂ C D
      inst✝ : (CategoryTheory.forget C).PreservesEpimorphisms
      ⊢ ((CategoryTheory.forget₂ C D).comp (CategoryTheory.forget D)).PreservesEpimo …
    -/
    simp only [HasForget₂.forget_comp]
    /-
      C✝ : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C✝
      inst✝⁶ : CategoryTheory.ConcreteCategory C✝
      C : Type u
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.Category.{v', u'} D
      inst✝² : CategoryTheory.ConcreteCategory D
      inst✝¹ : CategoryTheory.HasForget₂ C D
      inst✝ : (CategoryTheory.forget C).PreservesEpimorphisms
      ⊢ (CategoryTheory.forget C).PreservesEpimorphisms
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  Functor.preservesEpimorphisms_of_preserves_of_reflects _ (forget D)


lemma surjective_le_epimorphisms :
    MorphismProperty.surjective C ≤ epimorphisms C :=
  fun _ _ _ hf => (forget C).epi_of_epi_map ((epi_iff_surjective _).2 hf)


lemma injective_le_monomorphisms :
    MorphismProperty.injective C ≤ monomorphisms C :=
  fun _ _ _ hf => (forget C).mono_of_mono_map ((mono_iff_injective _).2 hf)


lemma surjective_eq_epimorphisms_iff :
    MorphismProperty.surjective C = epimorphisms C ↔ (forget C).PreservesEpimorphisms := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    ⊢ Iff (Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.Morph …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      ⊢ Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.MorphismPr …
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.Morphism …
      ⊢ (CategoryTheory.forget C).PreservesEpimorphisms
    -/
    constructor
    /-
      case mp.preserves
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.Morphism …
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y) [inst : CategoryTheory.Epi f], CategoryTheo …
    -/
    rintro _ _ f (hf : epimorphisms C f)
    /-
      case mp.preserves
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.Morphism …
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      ⊢ CategoryTheory.Epi ((CategoryTheory.forget C).map f)
    -/
    rw [epi_iff_surjective]
    /-
      case mp.preserves
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.Morphism …
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      ⊢ Function.Surjective ((CategoryTheory.forget C).map f)
    -/
    rw [← h] at hf
    /-
      case mp.preserves
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.Morphism …
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.surjective C f
      ⊢ Function.Surjective ((CategoryTheory.forget C).map f)
    -/
    exact hf
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      ⊢ (CategoryTheory.forget C).PreservesEpimorphisms → Eq (CategoryTheory.Morphis …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesEpimorphisms
      ⊢ Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.MorphismPr …
    -/
    apply le_antisymm (surjective_le_epimorphisms C)
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesEpimorphisms
      ⊢ LE.le (CategoryTheory.MorphismProperty.epimorphisms C) (CategoryTheory.Morph …
    -/
    intro _ _ f hf
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesEpimorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      ⊢ CategoryTheory.MorphismProperty.surjective C f
    -/
    have : Epi f := hf
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesEpimorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      this : CategoryTheory.Epi f
      ⊢ CategoryTheory.MorphismProperty.surjective C f
    -/
    change Function.Surjective ((forget C).map f)
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesEpimorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      this : CategoryTheory.Epi f
      ⊢ Function.Surjective ((CategoryTheory.forget C).map f)
    -/
    rw [← epi_iff_surjective]
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesEpimorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      this : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi ((CategoryTheory.forget C).map f)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma injective_eq_monomorphisms_iff :
    MorphismProperty.injective C = monomorphisms C ↔ (forget C).PreservesMonomorphisms := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    ⊢ Iff (Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.Morphi …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      ⊢ Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.MorphismPro …
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.MorphismP …
      ⊢ (CategoryTheory.forget C).PreservesMonomorphisms
    -/
    constructor
    /-
      case mp.preserves
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.MorphismP …
      ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y) [inst : CategoryTheory.Mono f], CategoryThe …
    -/
    rintro _ _ f (hf : monomorphisms C f)
    /-
      case mp.preserves
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.MorphismP …
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.monomorphisms C f
      ⊢ CategoryTheory.Mono ((CategoryTheory.forget C).map f)
    -/
    rw [mono_iff_injective]
    /-
      case mp.preserves
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.MorphismP …
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.monomorphisms C f
      ⊢ Function.Injective ((CategoryTheory.forget C).map f)
    -/
    rw [← h] at hf
    /-
      case mp.preserves
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      h : Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.MorphismP …
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.injective C f
      ⊢ Function.Injective ((CategoryTheory.forget C).map f)
    -/
    exact hf
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      ⊢ (CategoryTheory.forget C).PreservesMonomorphisms → Eq (CategoryTheory.Morphi …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesMonomorphisms
      ⊢ Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.MorphismPro …
    -/
    apply le_antisymm (injective_le_monomorphisms C)
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesMonomorphisms
      ⊢ LE.le (CategoryTheory.MorphismProperty.monomorphisms C) (CategoryTheory.Morp …
    -/
    intro _ _ f hf
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesMonomorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.monomorphisms C f
      ⊢ CategoryTheory.MorphismProperty.injective C f
    -/
    have : Mono f := hf
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesMonomorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.monomorphisms C f
      this : CategoryTheory.Mono f
      ⊢ CategoryTheory.MorphismProperty.injective C f
    -/
    change Function.Injective ((forget C).map f)
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesMonomorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.monomorphisms C f
      this : CategoryTheory.Mono f
      ⊢ Function.Injective ((CategoryTheory.forget C).map f)
    -/
    rw [← mono_iff_injective]
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      a✝ : (CategoryTheory.forget C).PreservesMonomorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.monomorphisms C f
      this : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono ((CategoryTheory.forget C).map f)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma injective_eq_monomorphisms [(forget C).PreservesMonomorphisms] :
    MorphismProperty.injective C = monomorphisms C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : (CategoryTheory.forget C).PreservesMonomorphisms
    ⊢ Eq (CategoryTheory.MorphismProperty.injective C) (CategoryTheory.MorphismPro …
  -/
  rw [injective_eq_monomorphisms_iff]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : (CategoryTheory.forget C).PreservesMonomorphisms
    ⊢ (CategoryTheory.forget C).PreservesMonomorphisms
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma surjective_eq_epimorphisms [(forget C).PreservesEpimorphisms] :
    MorphismProperty.surjective C = epimorphisms C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : (CategoryTheory.forget C).PreservesEpimorphisms
    ⊢ Eq (CategoryTheory.MorphismProperty.surjective C) (CategoryTheory.MorphismPr …
  -/
  rw [surjective_eq_epimorphisms_iff]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : (CategoryTheory.forget C).PreservesEpimorphisms
    ⊢ (CategoryTheory.forget C).PreservesEpimorphisms
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A concrete category with strong epi mono factorizations and such that
the forget functor preserves mono and epi admits functorial surjective/injective
factorizations. -/
noncomputable def functorialSurjectiveInjectiveFactorizationData :
    FunctorialSurjectiveInjectiveFactorizationData C :=
  (functorialEpiMonoFactorizationData C).ofLE
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.ConcreteCategory C
          inst✝² : CategoryTheory.Limits.HasStrongEpiMonoFactorisations C
          inst✝¹ : (CategoryTheory.forget C).PreservesMonomorphisms
          inst✝ : (CategoryTheory.forget C).PreservesEpimorphisms
          ⊢ LE.le (CategoryTheory.MorphismProperty.epimorphisms C) (CategoryTheory.Morph …
        -/
    (by rw [surjective_eq_epimorphisms])
        /-
          🎉 no goals
        -/
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.ConcreteCategory C
          inst✝² : CategoryTheory.Limits.HasStrongEpiMonoFactorisations C
          inst✝¹ : (CategoryTheory.forget C).PreservesMonomorphisms
          inst✝ : (CategoryTheory.forget C).PreservesEpimorphisms
          ⊢ LE.le (CategoryTheory.MorphismProperty.monomorphisms C) (CategoryTheory.Morp …
        -/
    (by rw [injective_eq_monomorphisms])
        /-
          🎉 no goals
        -/


instance (priority := 100) : HasFunctorialSurjectiveInjectiveFactorization C where
  nonempty_functorialFactorizationData :=
    ⟨functorialSurjectiveInjectiveFactorizationData C⟩


theorem injective_of_mono_of_preservesPullback {X Y : C} (f : X ⟶ Y) [Mono f]
    [PreservesLimitsOfShape WalkingCospan (forget C)] : Function.Injective f :=
  (mono_iff_injective ((forget C).map f)).mp inferInstance


theorem mono_iff_injective_of_preservesPullback {X Y : C} (f : X ⟶ Y)
    [PreservesLimitsOfShape WalkingCospan (forget C)] : Mono f ↔ Function.Injective f :=
  ((forget C).mono_map_iff_mono _).symm.trans (mono_iff_injective _)


/-- In any concrete category, surjective morphisms are epimorphisms. -/
theorem epi_of_surjective {X Y : C} (f : X ⟶ Y) (s : Function.Surjective f) :
    Epi f :=
  (forget C).epi_of_epi_map ((epi_iff_surjective ((forget C).map f)).2 s)


theorem surjective_of_epi_of_preservesPushout {X Y : C} (f : X ⟶ Y) [Epi f]
    [PreservesColimitsOfShape WalkingSpan (forget C)] : Function.Surjective f :=
  (epi_iff_surjective ((forget C).map f)).mp inferInstance


theorem epi_iff_surjective_of_preservesPushout {X Y : C} (f : X ⟶ Y)
    [PreservesColimitsOfShape WalkingSpan (forget C)] : Epi f ↔ Function.Surjective f :=
  ((forget C).epi_map_iff_epi _).symm.trans (epi_iff_surjective _)


theorem bijective_of_isIso {X Y : C} (f : X ⟶ Y) [IsIso f] :
    Function.Bijective ((forget C).map f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Function.Bijective ((CategoryTheory.forget C).map f)
  -/
  rw [← isIso_iff_bijective]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ CategoryTheory.IsIso ((CategoryTheory.forget C).map f)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If the forgetful functor of a concrete category reflects isomorphisms, being an isomorphism
is equivalent to being bijective. -/
theorem isIso_iff_bijective [(forget C).ReflectsIsomorphisms]
    {X Y : C} (f : X ⟶ Y) : IsIso f ↔ Function.Bijective ((forget C).map f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f) (Function.Bijective ((CategoryTheory.forget C). …
  -/
  rw [← CategoryTheory.isIso_iff_bijective]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f) (CategoryTheory.IsIso ((CategoryTheory.forget C …
  -/
  exact ⟨fun _ ↦ inferInstance, fun _ ↦ isIso_of_reflects_iso f (forget C)⟩
  /-
    🎉 no goals
  -/


