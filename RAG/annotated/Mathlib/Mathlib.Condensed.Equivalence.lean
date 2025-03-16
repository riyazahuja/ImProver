/-- The equivalence from coherent sheaves on `Stonean` to coherent sheaves on `CompHaus`
    (i.e. condensed sets). -/
noncomputable
def equivalence (A : Type*) [Category A]
    [∀ X, HasLimitsOfShape (StructuredArrow X Stonean.toCompHaus.op) A] :
    Sheaf (coherentTopology Stonean) A ≌ Condensed.{u} A :=
  coherentTopology.equivalence' Stonean.toCompHaus A


instance : Stonean.toProfinite.PreservesEffectiveEpis where
  preserves f h :=
     /-
       X✝ Y✝ : Stonean
       f : Quiver.Hom X✝ Y✝
       h : CategoryTheory.EffectiveEpi f
       ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi (?m.1580 f h)) (List.cons (Categ …
     -/
     /-
       🎉 no goals
     -/
     /-
       🎉 no goals
     -/
                                                    /-
                                                      🎉 no goals
                                                    -/
    ((Profinite.effectiveEpi_tfae _).out 0 2).mpr (((Stonean.effectiveEpi_tfae _).out 0 2).mp h)
                                                    /-
                                                      🎉 no goals
                                                    -/


instance : Stonean.toProfinite.ReflectsEffectiveEpis where
  reflects f h :=
     /-
       X✝ Y✝ : Stonean
       f : Quiver.Hom X✝ Y✝
       h : CategoryTheory.EffectiveEpi (Stonean.toProfinite.map f)
       ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi f) (List.cons (CategoryTheory.Ep …
     -/
     /-
       🎉 no goals
     -/
     /-
       🎉 no goals
     -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    ((Stonean.effectiveEpi_tfae f).out 0 2).mpr (((Profinite.effectiveEpi_tfae _).out 0 2).mp h)
                                                  /-
                                                    🎉 no goals
                                                  -/


/--
An effective presentation of an `X : Profinite` with respect to the inclusion functor from `Stonean`
-/
noncomputable def stoneanToProfiniteEffectivePresentation (X : Profinite) :
    Stonean.toProfinite.EffectivePresentation X where
  p := X.presentation
  f := Profinite.presentation.π X
                   /-
                     X : Profinite
                     ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi ?m.2190) (List.cons (CategoryThe …
                   -/
                   /-
                     🎉 no goals
                   -/
  effectiveEpi := ((Profinite.effectiveEpi_tfae _).out 0 1).mpr (inferInstance : Epi _)
                   /-
                     🎉 no goals
                   -/


instance : Stonean.toProfinite.EffectivelyEnough where
  presentation X := ⟨stoneanToProfiniteEffectivePresentation X⟩


/-- The equivalence from coherent sheaves on `Stonean` to coherent sheaves on `Profinite`. -/
noncomputable
def equivalence (A : Type*) [Category A]
    [∀ X, HasLimitsOfShape (StructuredArrow X Stonean.toProfinite.op) A] :
    Sheaf (coherentTopology Stonean) A ≌ Sheaf (coherentTopology Profinite) A :=
  coherentTopology.equivalence' Stonean.toProfinite A


/-- The equivalence from coherent sheaves on `Profinite` to coherent sheaves on `CompHaus`
    (i.e. condensed sets). -/
noncomputable
def equivalence (A : Type*) [Category A]
    [∀ X, HasLimitsOfShape (StructuredArrow X profiniteToCompHaus.op) A] :
    Sheaf (coherentTopology Profinite) A ≌ Condensed.{u} A :=
  coherentTopology.equivalence' profiniteToCompHaus A


lemma isSheafProfinite
    [∀ Y, HasLimitsOfShape (StructuredArrow Y profiniteToCompHaus.{u}.op) A] :
    Presheaf.IsSheaf (coherentTopology Profinite)
    (profiniteToCompHaus.op ⋙ X.val) :=
  ((ProfiniteCompHaus.equivalence A).inverse.obj X).cond


lemma isSheafStonean
    [∀ Y, HasLimitsOfShape (StructuredArrow Y Stonean.toCompHaus.{u}.op) A] :
    Presheaf.IsSheaf (coherentTopology Stonean)
    (Stonean.toCompHaus.op ⋙ X.val) :=
  ((StoneanCompHaus.equivalence A).inverse.obj X).cond


