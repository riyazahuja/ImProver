/-- The functor `N` for the equivalence is `normalizedMooreComplex A` -/
def N : SimplicialObject A ⥤ ChainComplex A ℕ :=
  AlgebraicTopology.normalizedMooreComplex A


/-- The functor `Γ` for the equivalence is the same as in the pseudoabelian case. -/
def Γ : ChainComplex A ℕ ⥤ SimplicialObject A :=
  Idempotents.DoldKan.Γ


/-- The comparison isomorphism between `normalizedMooreComplex A` and
the functor `Idempotents.DoldKan.N` from the pseudoabelian case -/
@[simps!]
def comparisonN : (N : SimplicialObject A ⥤ _) ≅ Idempotents.DoldKan.N :=
  calc
    N ≅ N ⋙ 𝟭 _ := Functor.leftUnitor N
    _ ≅ N ⋙ (toKaroubiEquivalence _).functor ⋙ (toKaroubiEquivalence _).inverse :=
          isoWhiskerLeft _ (toKaroubiEquivalence _).unitIso
    _ ≅ (N ⋙ (toKaroubiEquivalence _).functor) ⋙ (toKaroubiEquivalence _).inverse :=
          Iso.refl _
    _ ≅ N₁ ⋙ (toKaroubiEquivalence _).inverse :=
          isoWhiskerRight (N₁_iso_normalizedMooreComplex_comp_toKaroubi A).symm _
    _ ≅ Idempotents.DoldKan.N := Iso.refl _


/-- The Dold-Kan equivalence for abelian categories -/
@[simps! functor]
def equivalence : SimplicialObject A ≌ ChainComplex A ℕ :=
  (Idempotents.DoldKan.equivalence (C := A)).changeFunctor comparisonN.symm


theorem equivalence_inverse : (equivalence : SimplicialObject A ≌ _).inverse = Γ :=
  rfl


