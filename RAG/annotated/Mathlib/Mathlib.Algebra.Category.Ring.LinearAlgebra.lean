lemma nontrivial_of_isPushout_of_isField {A B C D : CommRingCat.{u}}
    (hA : IsField A) {f : A ⟶ B} {g : A ⟶ C} {inl : B ⟶ D} {inr : C ⟶ D}
    [Nontrivial B] [Nontrivial C]
    (h : IsPushout f g inl inr) : Nontrivial D := by
  /-
    A B C D : CommRingCat
    hA : IsField ↑A
    f : Quiver.Hom A B
    g : Quiver.Hom A C
    inl : Quiver.Hom B D
    inr : Quiver.Hom C D
    inst✝¹ : Nontrivial ↑B
    inst✝ : Nontrivial ↑C
    h : CategoryTheory.IsPushout f g inl inr
    ⊢ Nontrivial ↑D
  -/
  letI : Field A := hA.toField
  /-
    A B C D : CommRingCat
    hA : IsField ↑A
    f : Quiver.Hom A B
    g : Quiver.Hom A C
    inl : Quiver.Hom B D
    inr : Quiver.Hom C D
    inst✝¹ : Nontrivial ↑B
    inst✝ : Nontrivial ↑C
    h : CategoryTheory.IsPushout f g inl inr
    this : Field ↑A := hA.toField
    ⊢ Nontrivial ↑D
  -/
  algebraize [f.hom, g.hom]
  let e : D ≅ .of (B ⊗[A] C) :=
    IsColimit.coconePointUniqueUpToIso h.isColimit (CommRingCat.pushoutCoconeIsColimit A B C)
  /-
    A B C D : CommRingCat
    hA : IsField ↑A
    f : Quiver.Hom A B
    g : Quiver.Hom A C
    inl : Quiver.Hom B D
    inr : Quiver.Hom C D
    inst✝¹ : Nontrivial ↑B
    inst✝ : Nontrivial ↑C
    h : CategoryTheory.IsPushout f g inl inr
    this : Field ↑A := hA.toField
    algInst✝¹ : Algebra ↑A ↑B := f.hom.toAlgebra
    algInst✝ : Algebra ↑A ↑C := g.hom.toAlgebra
    e : CategoryTheory.Iso D (CommRingCat.of (TensorProduct ↑A ↑B ↑C)) := h.isColi …
    ⊢ Nontrivial ↑D
  -/
  let e' : D ≃ B ⊗[A] C := e.commRingCatIsoToRingEquiv.toEquiv
  /-
    A B C D : CommRingCat
    hA : IsField ↑A
    f : Quiver.Hom A B
    g : Quiver.Hom A C
    inl : Quiver.Hom B D
    inr : Quiver.Hom C D
    inst✝¹ : Nontrivial ↑B
    inst✝ : Nontrivial ↑C
    h : CategoryTheory.IsPushout f g inl inr
    this : Field ↑A := hA.toField
    algInst✝¹ : Algebra ↑A ↑B := f.hom.toAlgebra
    algInst✝ : Algebra ↑A ↑C := g.hom.toAlgebra
    e : CategoryTheory.Iso D (CommRingCat.of (TensorProduct ↑A ↑B ↑C)) := h.isColi …
    e' : Equiv (↑D) (TensorProduct ↑A ↑B ↑C) := e.commRingCatIsoToRingEquiv.toEquiv
    ⊢ Nontrivial ↑D
  -/
  exact e'.nontrivial
  /-
    🎉 no goals
  -/


