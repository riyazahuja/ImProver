instance [HasColimitsOfShape K (PresheafOfModules.{v} R.val)] :
    HasColimitsOfShape K (SheafOfModules.{v} R) where
  has_colimit F := by
    let e : F ≅ (F ⋙ forget R) ⋙ PresheafOfModules.sheafification (𝟙 R.val) :=
      isoWhiskerLeft F (asIso (PresheafOfModules.sheafificationAdjunction (𝟙 R.val)).counit).symm
    /-
      C : Type u'
      inst✝⁴ : CategoryTheory.Category.{v', u'} C
      J : CategoryTheory.GrothendieckTopology C
      R : CategoryTheory.Sheaf J RingCat
      inst✝³ : CategoryTheory.HasWeakSheafify J AddCommGrp
      inst✝² : J.WEqualsLocallyBijective AddCommGrp
      K : Type w
      inst✝¹ : CategoryTheory.Category.{w', w} K
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape K (PresheafOfModules R.val)
      F : CategoryTheory.Functor K (SheafOfModules R)
      e : CategoryTheory.Iso F ((F.comp (SheafOfModules.forget R)).comp (PresheafOfM …
      ⊢ CategoryTheory.Limits.HasColimit F
    -/
    exact hasColimitOfIso e
    /-
      🎉 no goals
    -/


