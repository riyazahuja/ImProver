noncomputable instance : Abelian (SheafOfModules.{v} R) := by
  /-
    C : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R : CategoryTheory.Sheaf J RingCat
    inst✝¹ : CategoryTheory.HasSheafify J AddCommGrp
    inst✝ : J.WEqualsLocallyBijective AddCommGrp
    ⊢ CategoryTheory.Abelian (SheafOfModules R)
  -/
  let adj := PresheafOfModules.sheafificationAdjunction (𝟙 R.val)
  /-
    C : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} C
    J : CategoryTheory.GrothendieckTopology C
    R : CategoryTheory.Sheaf J RingCat
    inst✝¹ : CategoryTheory.HasSheafify J AddCommGrp
    inst✝ : J.WEqualsLocallyBijective AddCommGrp
    adj : CategoryTheory.Adjunction (PresheafOfModules.sheafification (CategoryThe …
    ⊢ CategoryTheory.Abelian (SheafOfModules R)
  -/
  exact abelianOfAdjunction _ _ (asIso (adj.counit)) adj
  /-
    🎉 no goals
  -/


