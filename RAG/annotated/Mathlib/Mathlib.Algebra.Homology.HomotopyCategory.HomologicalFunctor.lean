instance (n : ℤ) : (homologyFunctor C (ComplexShape.up ℤ) n).IsHomological :=
  Functor.IsHomological.mk' _ (fun T hT => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      n : Int
      T : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape. …
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ Exists fun T' => Exists fun e => ((CategoryTheory.Pretriangulated.shortCompl …
    -/
    rw [distinguished_iff_iso_trianglehOfDegreewiseSplit] at hT
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      n : Int
      T : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape. …
      hT✝ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      hT : Exists fun S => Exists fun σ => Nonempty (CategoryTheory.Iso T (CochainCo …
      ⊢ Exists fun T' => Exists fun e => ((CategoryTheory.Pretriangulated.shortCompl …
    -/
    obtain ⟨S, σ, ⟨e⟩⟩ := hT
    have hS := HomologicalComplex.shortExact_of_degreewise_shortExact S
      (fun n => (σ n).shortExact)
    exact ⟨_, e, (ShortComplex.exact_iff_of_iso
      (S.mapNatIso (homologyFunctorFactors C (ComplexShape.up ℤ) n))).2 (hS.homology_exact₂ n)⟩)


