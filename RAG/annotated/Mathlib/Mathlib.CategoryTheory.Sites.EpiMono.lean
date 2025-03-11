/-- The class of locally injective morphisms of sheaves, see `Sheaf.IsLocallyInjective`. -/
def locallyInjective : MorphismProperty (Sheaf J A) :=
  fun _ _  f => IsLocallyInjective f


/-- The class of locally surjective morphisms of sheaves, see `Sheaf.IsLocallySurjective`. -/
def locallySurjective : MorphismProperty (Sheaf J A) :=
  fun _ _  f => IsLocallySurjective f


/-- Given a functorial surjective/injective factorizations of morphisms in a concrete
category `A`, this is the induced functorial locally surjective/locally injective
factorization of morphisms in the category `Sheaf J A`. -/
noncomputable def functorialLocallySurjectiveInjectiveFactorization :
    (locallySurjective J A).FunctorialFactorizationData (locallyInjective J A) where
  Z := (sheafToPresheaf J A).mapArrow ⋙ (data.functorCategory Cᵒᵖ).Z ⋙ presheafToSheaf J A
  i := whiskerLeft Arrow.leftFunc (inv (sheafificationAdjunction J A).counit) ≫
        whiskerLeft (sheafToPresheaf J A).mapArrow
          (whiskerRight (data.functorCategory Cᵒᵖ).i (presheafToSheaf J A))
  p := whiskerLeft (sheafToPresheaf J A).mapArrow
        (whiskerRight (data.functorCategory Cᵒᵖ).p (presheafToSheaf J A)) ≫
          whiskerLeft Arrow.rightFunc (sheafificationAdjunction J A).counit
  fac := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    ext f : 2
    /-
      case w.h
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      f : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    dsimp
    simp only [assoc, ← Functor.map_comp_assoc,
      MorphismProperty.FunctorialFactorizationData.fac_app,
      NatIso.isIso_inv_app, IsIso.inv_comp_eq]
    /-
      case w.h
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      f : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.presheafToSheaf J A) …
    -/
    exact (sheafificationAdjunction J A).counit.naturality f.hom
    /-
      🎉 no goals
    -/
  hi _ := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      x✝ : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
      ⊢ CategoryTheory.Sheaf.locallySurjective J A ((CategoryTheory.CategoryStruct.c …
    -/
    dsimp [locallySurjective]
    rw [← isLocallySurjective_sheafToPresheaf_map_iff, Functor.map_comp,
      Presheaf.comp_isLocallySurjective_iff, isLocallySurjective_sheafToPresheaf_map_iff,
      Presheaf.isLocallySurjective_presheafToSheaf_map_iff]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      x✝ : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective J ((CategoryTheory.MorphismPrope …
    -/
    apply Presheaf.isLocallySurjective_of_surjective
    /-
      case H
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      x✝ : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
      ⊢ ∀ (U : Opposite C), Function.Surjective ⇑(((CategoryTheory.MorphismProperty. …
    -/
    apply (data.functorCategory Cᵒᵖ).hi
    /-
      🎉 no goals
    -/
  hp _ := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      x✝ : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
      ⊢ CategoryTheory.Sheaf.locallyInjective J A ((CategoryTheory.CategoryStruct.co …
    -/
    dsimp [locallyInjective]
    rw [← isLocallyInjective_sheafToPresheaf_map_iff, Functor.map_comp,
      Presheaf.isLocallyInjective_comp_iff, isLocallyInjective_sheafToPresheaf_map_iff,
      Presheaf.isLocallyInjective_presheafToSheaf_map_iff]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      x✝ : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
      ⊢ CategoryTheory.Presheaf.IsLocallyInjective J ((CategoryTheory.MorphismProper …
    -/
    apply Presheaf.isLocallyInjective_of_injective
    /-
      case hφ
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} A
      inst✝³ : CategoryTheory.ConcreteCategory A
      inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝¹ : J.WEqualsLocallyBijective A
      data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
      inst✝ : CategoryTheory.HasWeakSheafify J A
      x✝ : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
      ⊢ ∀ (X : Opposite C), Function.Injective ⇑(((CategoryTheory.MorphismProperty.F …
    -/
    apply (data.functorCategory Cᵒᵖ).hp
    /-
      🎉 no goals
    -/


instance : IsLocallySurjective
            ((functorialLocallySurjectiveInjectiveFactorization J data).i.app f) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} A
    inst✝³ : CategoryTheory.ConcreteCategory A
    inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
    inst✝¹ : J.WEqualsLocallyBijective A
    data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
    inst✝ : CategoryTheory.HasWeakSheafify J A
    f : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
    ⊢ CategoryTheory.Sheaf.IsLocallySurjective ((CategoryTheory.Sheaf.functorialLo …
  -/
  apply (functorialLocallySurjectiveInjectiveFactorization J data).hi
  /-
    🎉 no goals
  -/


instance : IsLocallyInjective
            ((functorialLocallySurjectiveInjectiveFactorization J data).p.app f) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} A
    inst✝³ : CategoryTheory.ConcreteCategory A
    inst✝² : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
    inst✝¹ : J.WEqualsLocallyBijective A
    data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
    inst✝ : CategoryTheory.HasWeakSheafify J A
    f : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
    ⊢ CategoryTheory.Sheaf.IsLocallyInjective ((CategoryTheory.Sheaf.functorialLoc …
  -/
  apply (functorialLocallySurjectiveInjectiveFactorization J data).hp
  /-
    🎉 no goals
  -/


instance : Epi ((functorialLocallySurjectiveInjectiveFactorization J data).i.app f) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} A
    inst✝⁴ : CategoryTheory.ConcreteCategory A
    inst✝³ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
    inst✝² : J.WEqualsLocallyBijective A
    data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    f : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
    inst✝ : J.HasSheafCompose (CategoryTheory.forget A)
    ⊢ CategoryTheory.Epi ((CategoryTheory.Sheaf.functorialLocallySurjectiveInjecti …
  -/
  apply epi_of_isLocallySurjective
  /-
    🎉 no goals
  -/


instance : Mono ((functorialLocallySurjectiveInjectiveFactorization J data).p.app f) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} A
    inst✝⁴ : CategoryTheory.ConcreteCategory A
    inst✝³ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
    inst✝² : J.WEqualsLocallyBijective A
    data : CategoryTheory.ConcreteCategory.FunctorialSurjectiveInjectiveFactorizat …
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    f : CategoryTheory.Arrow (CategoryTheory.Sheaf J A)
    inst✝ : J.HasSheafCompose (CategoryTheory.forget A)
    ⊢ CategoryTheory.Mono ((CategoryTheory.Sheaf.functorialLocallySurjectiveInject …
  -/
  apply mono_of_isLocallyInjective
  /-
    🎉 no goals
  -/


instance : (locallySurjective J A).HasFunctorialFactorization (locallyInjective J A) where
  nonempty_functorialFactorizationData :=
    ⟨functorialLocallySurjectiveInjectiveFactorization J
      (MorphismProperty.functorialFactorizationData _ _)⟩


lemma isLocallySurjective_iff_epi' :
    IsLocallySurjective φ ↔ Epi φ := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁶ : CategoryTheory.Category.{v', u'} A
    inst✝⁵ : CategoryTheory.ConcreteCategory A
    inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
    inst✝³ : J.WEqualsLocallyBijective A
    inst✝² : CategoryTheory.HasSheafify J A
    inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
    inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
    F G : CategoryTheory.Sheaf J A
    φ : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.Sheaf.IsLocallySurjective φ) (CategoryTheory.Epi φ)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ → CategoryTheory.Epi φ
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Sheaf.IsLocallySurjective φ
      ⊢ CategoryTheory.Epi φ
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      ⊢ CategoryTheory.Epi φ → CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Epi φ
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    let data := (locallySurjective J A).factorizationData (locallyInjective J A) φ
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Epi φ
      data : (CategoryTheory.Sheaf.locallySurjective J A).MapFactorizationData (Cate …
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    have : IsLocallySurjective data.i := data.hi
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Epi φ
      data : (CategoryTheory.Sheaf.locallySurjective J A).MapFactorizationData (Cate …
      this : CategoryTheory.Sheaf.IsLocallySurjective data.i
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    have : IsLocallyInjective data.p := data.hp
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Epi φ
      data : (CategoryTheory.Sheaf.locallySurjective J A).MapFactorizationData (Cate …
      this✝ : CategoryTheory.Sheaf.IsLocallySurjective data.i
      this : CategoryTheory.Sheaf.IsLocallyInjective data.p
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    have : Epi data.p := epi_of_epi_fac data.fac
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Epi φ
      data : (CategoryTheory.Sheaf.locallySurjective J A).MapFactorizationData (Cate …
      this✝¹ : CategoryTheory.Sheaf.IsLocallySurjective data.i
      this✝ : CategoryTheory.Sheaf.IsLocallyInjective data.p
      this : CategoryTheory.Epi data.p
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    have := mono_of_isLocallyInjective data.p
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Epi φ
      data : (CategoryTheory.Sheaf.locallySurjective J A).MapFactorizationData (Cate …
      this✝² : CategoryTheory.Sheaf.IsLocallySurjective data.i
      this✝¹ : CategoryTheory.Sheaf.IsLocallyInjective data.p
      this✝ : CategoryTheory.Epi data.p
      this : CategoryTheory.Mono data.p
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    have := isIso_of_mono_of_epi data.p
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Epi φ
      data : (CategoryTheory.Sheaf.locallySurjective J A).MapFactorizationData (Cate …
      this✝³ : CategoryTheory.Sheaf.IsLocallySurjective data.i
      this✝² : CategoryTheory.Sheaf.IsLocallyInjective data.p
      this✝¹ : CategoryTheory.Epi data.p
      this✝ : CategoryTheory.Mono data.p
      this : CategoryTheory.IsIso data.p
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective φ
    -/
    rw [← data.fac]
    /-
      case mpr
      C : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝⁶ : CategoryTheory.Category.{v', u'} A
      inst✝⁵ : CategoryTheory.ConcreteCategory A
      inst✝⁴ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
      inst✝³ : J.WEqualsLocallyBijective A
      inst✝² : CategoryTheory.HasSheafify J A
      inst✝¹ : J.HasSheafCompose (CategoryTheory.forget A)
      inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf J A)
      F G : CategoryTheory.Sheaf J A
      φ : Quiver.Hom F G
      a✝ : CategoryTheory.Epi φ
      data : (CategoryTheory.Sheaf.locallySurjective J A).MapFactorizationData (Cate …
      this✝³ : CategoryTheory.Sheaf.IsLocallySurjective data.i
      this✝² : CategoryTheory.Sheaf.IsLocallyInjective data.p
      this✝¹ : CategoryTheory.Epi data.p
      this✝ : CategoryTheory.Mono data.p
      this : CategoryTheory.IsIso data.p
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective (CategoryTheory.CategoryStruct.comp …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


