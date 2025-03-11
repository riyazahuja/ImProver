variable
  [(coherentTopology CompHaus).WEqualsLocallyBijective A]
  [HasSheafify (coherentTopology CompHaus) A]
  [(coherentTopology CompHaus.{u}).HasSheafCompose (CategoryTheory.forget A)]
  [Balanced (Sheaf (coherentTopology CompHaus) A)]
  [PreservesFiniteProducts (CategoryTheory.forget A)] in
lemma epi_iff_locallySurjective_on_compHaus : Epi f ↔
    ∀ (S : CompHaus) (y : Y.val.obj ⟨S⟩),
      (∃ (S' : CompHaus) (φ : S' ⟶ S) (_ : Function.Surjective φ) (x : X.val.obj ⟨S'⟩),
        f.val.app ⟨S'⟩ x = Y.val.map ⟨φ⟩ y) := by
  rw [← isLocallySurjective_iff_epi', coherentTopology.isLocallySurjective_iff,
    regularTopology.isLocallySurjective_iff]
  /-
    A : Type u'
    inst✝⁷ : CategoryTheory.Category.{v', u'} A
    inst✝⁶ : CategoryTheory.ConcreteCategory A
    inst✝⁵ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
    X Y : Condensed A
    f : Quiver.Hom X Y
    inst✝⁴ : (CategoryTheory.coherentTopology CompHaus).WEqualsLocallyBijective A
    inst✝³ : CategoryTheory.HasSheafify (CategoryTheory.coherentTopology CompHaus) A
    inst✝² : (CategoryTheory.coherentTopology CompHaus).HasSheafCompose (CategoryT …
    inst✝¹ : CategoryTheory.Balanced (CategoryTheory.Sheaf (CategoryTheory.coheren …
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget A)
    ⊢ Iff (∀ (X_1 : CompHaus) (y : (CategoryTheory.forget A).obj (Y.val.obj { unop …
  -/
  simp_rw [((CompHaus.effectiveEpi_tfae _).out 0 2 :)]
  /-
    🎉 no goals
  -/


variable
  [PreservesFiniteProducts (CategoryTheory.forget A)]
  [∀ (X : CompHausᵒᵖ), HasLimitsOfShape (StructuredArrow X Stonean.toCompHaus.op) A]
  [(extensiveTopology Stonean).WEqualsLocallyBijective A]
  [HasSheafify (extensiveTopology Stonean) A]
  [(extensiveTopology Stonean.{u}).HasSheafCompose (CategoryTheory.forget A)]
  [Balanced (Sheaf (extensiveTopology Stonean) A)] in
lemma epi_iff_surjective_on_stonean : Epi f ↔
    ∀ (S : Stonean), Function.Surjective (f.val.app (op S.compHaus)) := by
  rw [← (StoneanCompHaus.equivalence A).inverse.epi_map_iff_epi,
    ← Presheaf.coherentExtensiveEquivalence.functor.epi_map_iff_epi,
    ← isLocallySurjective_iff_epi']
  /-
    A : Type u'
    inst✝⁸ : CategoryTheory.Category.{v', u'} A
    inst✝⁷ : CategoryTheory.ConcreteCategory A
    inst✝⁶ : CategoryTheory.ConcreteCategory.HasFunctorialSurjectiveInjectiveFacto …
    X Y : Condensed A
    f : Quiver.Hom X Y
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget A)
    inst✝⁴ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
    inst✝³ : (CategoryTheory.extensiveTopology Stonean).WEqualsLocallyBijective A
    inst✝² : CategoryTheory.HasSheafify (CategoryTheory.extensiveTopology Stonean) A
    inst✝¹ : (CategoryTheory.extensiveTopology Stonean).HasSheafCompose (CategoryT …
    inst✝ : CategoryTheory.Balanced (CategoryTheory.Sheaf (CategoryTheory.extensiv …
    ⊢ Iff (CategoryTheory.Sheaf.IsLocallySurjective (CategoryTheory.Presheaf.coher …
  -/
  exact extensiveTopology.isLocallySurjective_iff (D := A) _
  /-
    🎉 no goals
  -/


lemma epi_iff_locallySurjective_on_compHaus : Epi f ↔
    ∀ (S : CompHaus) (y : Y.val.obj ⟨S⟩),
      (∃ (S' : CompHaus) (φ : S' ⟶ S) (_ : Function.Surjective φ) (x : X.val.obj ⟨S'⟩),
        f.val.app ⟨S'⟩ x = Y.val.map ⟨φ⟩ y) :=
  Condensed.epi_iff_locallySurjective_on_compHaus _ f


lemma epi_iff_surjective_on_stonean : Epi f ↔
    ∀ (S : Stonean), Function.Surjective (f.val.app (op S.compHaus)) :=
  Condensed.epi_iff_surjective_on_stonean _ f


lemma epi_iff_surjective_on_stonean : Epi f ↔
    ∀ (S : Stonean), Function.Surjective (f.val.app (op S.compHaus)) :=
  have : HasLimitsOfSize.{u, u+1} (ModuleCat R) := hasLimitsOfSizeShrink.{u, u+1, u+1, u+1} _
  Condensed.epi_iff_surjective_on_stonean _ f


