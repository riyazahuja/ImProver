lemma hasExactColimitsOfShape [HasColimitsOfShape J A] [HasExactColimitsOfShape J A]
    [HasFiniteLimits A] : HasExactColimitsOfShape J (Condensed.{u} A) := by
  let e : Condensed.{u} A ≌ Sheaf (extensiveTopology Stonean.{u}) A :=
    (StoneanCompHaus.equivalence A).symm.trans Presheaf.coherentExtensiveEquivalence
  have : HasColimitsOfShape J (Sheaf (extensiveTopology Stonean.{u}) A) :=
    hasColimitsOfShape_of_hasColimitsOfShape_createsColimitsOfShape e.inverse
  /-
    A : Type u_1
    J : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} A
    inst✝⁷ : CategoryTheory.Category.{u_3, u_2} J
    inst✝⁶ : CategoryTheory.Preadditive A
    inst✝⁵ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
    inst✝⁴ : CategoryTheory.HasWeakSheafify (CategoryTheory.coherentTopology CompH …
    inst✝³ : CategoryTheory.HasWeakSheafify (CategoryTheory.extensiveTopology Ston …
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J A
    inst✝¹ : CategoryTheory.HasExactColimitsOfShape J A
    inst✝ : CategoryTheory.Limits.HasFiniteLimits A
    e : CategoryTheory.Equivalence (Condensed A) (CategoryTheory.Sheaf (CategoryTh …
    this : CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.Sheaf (Categ …
    ⊢ CategoryTheory.HasExactColimitsOfShape J (Condensed A)
  -/
  exact HasExactColimitsOfShape.domain_of_functor _ e.functor
  /-
    🎉 no goals
  -/


lemma hasExactLimitsOfShape [HasLimitsOfShape J A] [HasExactLimitsOfShape J A]
    [HasFiniteColimits A] : HasExactLimitsOfShape J (Condensed.{u} A) := by
  let e : Condensed.{u} A ≌ Sheaf (extensiveTopology Stonean.{u}) A :=
    (StoneanCompHaus.equivalence A).symm.trans Presheaf.coherentExtensiveEquivalence
  have : HasLimitsOfShape J (Sheaf (extensiveTopology Stonean.{u}) A) :=
    hasLimitsOfShape_of_hasLimitsOfShape_createsLimitsOfShape e.inverse
  /-
    A : Type u_1
    J : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} A
    inst✝⁷ : CategoryTheory.Category.{u_3, u_2} J
    inst✝⁶ : CategoryTheory.Preadditive A
    inst✝⁵ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
    inst✝⁴ : CategoryTheory.HasWeakSheafify (CategoryTheory.coherentTopology CompH …
    inst✝³ : CategoryTheory.HasWeakSheafify (CategoryTheory.extensiveTopology Ston …
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape J A
    inst✝¹ : CategoryTheory.HasExactLimitsOfShape J A
    inst✝ : CategoryTheory.Limits.HasFiniteColimits A
    e : CategoryTheory.Equivalence (Condensed A) (CategoryTheory.Sheaf (CategoryTh …
    this : CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.Sheaf (Categor …
    ⊢ CategoryTheory.HasExactLimitsOfShape J (Condensed A)
  -/
  exact HasExactLimitsOfShape.domain_of_functor _ e.functor
  /-
    🎉 no goals
  -/


local instance : HasLimitsOfSize.{u, u+1} (ModuleCat.{u+1} R) :=
  hasLimitsOfSizeShrink.{u, u+1, u+1, u} _


instance : AB5 (CondensedMod.{u} R) where
  ofShape J _ _ := hasExactColimitsOfShape (ModuleCat R) J


instance : AB4 (CondensedMod.{u} R) := AB4.of_AB5 _


instance : AB4Star (CondensedMod.{u} R) where
  ofShape J := hasExactLimitsOfShape (ModuleCat R) (Discrete J)


