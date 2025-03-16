/-- The connecting homomorphism `Q.obj (S.X₃) ⟶ (Q.obj S.X₁)⟦(1 : ℤ)⟧`
in the derived category when `S` is a short exact short complex of
cochain complexes in an abelian category. -/
noncomputable def triangleOfSESδ :
  Q.obj (S.X₃) ⟶ (Q.obj S.X₁)⟦(1 : ℤ)⟧ :=
    have := CochainComplex.mappingCone.quasiIso_descShortComplex hS
    inv (Q.map (CochainComplex.mappingCone.descShortComplex S)) ≫
      Q.map (CochainComplex.mappingCone.triangle S.f).mor₃ ≫
      (Q.commShiftIso (1 : ℤ)).hom.app S.X₁


/-- The distinguished triangle in the derived category associated to a short
exact sequence of cochain complexes. -/
@[simps!]
noncomputable def triangleOfSES : Triangle (DerivedCategory C) :=
  Triangle.mk (Q.map S.f) (Q.map S.g) (triangleOfSESδ hS)


/-- The triangle `triangleOfSES` attached to a short exact sequence `S` of cochain
complexes is isomorphism to the standard distinguished triangle associated to
the morphism `S.f`. -/
noncomputable def triangleOfSESIso :
    triangleOfSES hS ≅ Q.mapTriangle.obj (CochainComplex.mappingCone.triangle S.f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    ⊢ CategoryTheory.Iso (DerivedCategory.triangleOfSES hS) (DerivedCategory.Q.map …
  -/
  have := CochainComplex.mappingCone.quasiIso_descShortComplex hS
  refine Iso.symm (Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _)
    (asIso (Q.map (CochainComplex.mappingCone.descShortComplex S))) ?_ ?_ ?_)
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      hS : S.ShortExact
      this : QuasiIso (CochainComplex.mappingCone.descShortComplex S)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (DerivedCategory.Q.mapTriangle.obj (C …
    -/
  · dsimp [triangleOfSES]
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      hS : S.ShortExact
      this : QuasiIso (CochainComplex.mappingCone.descShortComplex S)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (DerivedCategory.Q.map S.f) (Category …
    -/
    simp only [comp_id, id_comp]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      hS : S.ShortExact
      this : QuasiIso (CochainComplex.mappingCone.descShortComplex S)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (DerivedCategory.Q.mapTriangle.obj (C …
    -/
  · dsimp
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      hS : S.ShortExact
      this : QuasiIso (CochainComplex.mappingCone.descShortComplex S)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (DerivedCategory.Q.map (CochainComple …
    -/
    simp only [← Q.map_comp, CochainComplex.mappingCone.inr_descShortComplex, id_comp]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      hS : S.ShortExact
      this : QuasiIso (CochainComplex.mappingCone.descShortComplex S)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (DerivedCategory.Q.mapTriangle.obj (C …
    -/
  · dsimp [triangleOfSESδ]
    /-
      case refine_3
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      hS : S.ShortExact
      this : QuasiIso (CochainComplex.mappingCone.descShortComplex S)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [CategoryTheory.Functor.map_id, comp_id, IsIso.hom_inv_id_assoc]
    /-
      🎉 no goals
    -/


lemma triangleOfSES_distinguished :
    triangleOfSES hS ∈ distTriang (DerivedCategory C) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Derive …
  -/
  rw [mem_distTriang_iff]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    ⊢ Exists fun X => Exists fun Y => Exists fun f => Nonempty (CategoryTheory.Iso …
  -/
  exact ⟨_, _, S.f, ⟨triangleOfSESIso hS⟩⟩
  /-
    🎉 no goals
  -/


