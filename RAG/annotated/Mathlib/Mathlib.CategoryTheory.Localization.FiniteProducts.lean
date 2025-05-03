lemma inverts :
    (W.functorCategory (Discrete J)).IsInvertedBy (lim ⋙ L) :=
  fun _ _ f hf => Localization.inverts L W _ (hW.lim_map f hf)


/-- The (candidate) limit functor for the localized category.
It is induced by `lim ⋙ L : (Discrete J ⥤ C) ⥤ D`. -/
noncomputable abbrev limitFunctor :
    (Discrete J ⥤ D) ⥤ D :=
  Localization.lift _ (inverts L hW)
    ((whiskeringRight (Discrete J) C D).obj L)


/-- The functor `limitFunctor L hW` is induced by `lim ⋙ L`. -/
noncomputable def compLimitFunctorIso :
    ((whiskeringRight (Discrete J) C D).obj L) ⋙ limitFunctor L hW ≅
      lim ⋙ L := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝³ : L.IsLocalization W
    J : Type
    inst✝² : CategoryTheory.Limits.HasProductsOfShape J C
    hW : W.IsStableUnderProductsOfShape J
    inst✝¹ : W.ContainsIdentities
    inst✝ : Finite J
    ⊢ CategoryTheory.Iso (((CategoryTheory.whiskeringRight (CategoryTheory.Discret …
  -/
  apply Localization.fac
  /-
    🎉 no goals
  -/


instance :
    CatCommSq (Functor.const (Discrete J)) L
      ((whiskeringRight (Discrete J) C D).obj L) (Functor.const (Discrete J)) where
  iso' := (Functor.compConstIso _ _).symm


noncomputable instance :
    CatCommSq lim ((whiskeringRight (Discrete J) C D).obj L) L (limitFunctor L hW) where
  iso' := (compLimitFunctorIso L hW).symm


/-- The adjunction between the constant functor `D ⥤ (Discrete J ⥤ D)`
and `limitFunctor L hW`. -/
noncomputable def adj :
    Functor.const _ ⊣ limitFunctor L hW :=
  constLimAdj.localization L W ((whiskeringRight (Discrete J) C D).obj L)
    (W.functorCategory (Discrete J)) (Functor.const _) (limitFunctor L hW)


lemma adj_counit_app (F : Discrete J ⥤ C) :
    (adj L hW).counit.app (F ⋙ L) =
      (Functor.const (Discrete J)).map ((compLimitFunctorIso L hW).hom.app F) ≫
        (Functor.compConstIso (Discrete J) L).hom.app (lim.obj F) ≫
        whiskerRight (constLimAdj.counit.app F) L := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝³ : L.IsLocalization W
    J : Type
    inst✝² : CategoryTheory.Limits.HasProductsOfShape J C
    hW : W.IsStableUnderProductsOfShape J
    inst✝¹ : W.ContainsIdentities
    inst✝ : Finite J
    F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    ⊢ Eq ((CategoryTheory.Localization.HasProductsOfShapeAux.adj L hW).counit.app  …
  -/
  apply constLimAdj.localization_counit_app
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `Localization.preservesProductsOfShape`. -/
noncomputable def isLimitMapCone (F : Discrete J ⥤ C) :
    IsLimit (L.mapCone (limit.cone F)) :=
  IsLimit.ofIsoLimit (isLimitConeOfAdj (adj L hW) (F ⋙ L))
                                                      /-
                                                        C : Type u₁
                                                        D : Type u₂
                                                        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
                                                        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                                                        L : CategoryTheory.Functor C D
                                                        W : CategoryTheory.MorphismProperty C
                                                        inst✝³ : L.IsLocalization W
                                                        J : Type
                                                        inst✝² : CategoryTheory.Limits.HasProductsOfShape J C
                                                        hW : W.IsStableUnderProductsOfShape J
                                                        inst✝¹ : W.ContainsIdentities
                                                        inst✝ : Finite J
                                                        F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
                                                        ⊢ ∀ (j : CategoryTheory.Discrete J), Eq ((CategoryTheory.Limits.coneOfAdj (Cat …
                                                      -/
    (Cones.ext ((compLimitFunctorIso L hW).app F) (by simp [adj_counit_app, constLimAdj]))
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma hasProductsOfShape (J : Type) [Finite J] [HasProductsOfShape J C]
    (hW : W.IsStableUnderProductsOfShape J) :
    HasProductsOfShape J D :=
  hasLimitsOfShape_iff_isLeftAdjoint_const.2
    (HasProductsOfShapeAux.adj L hW).isLeftAdjoint


/-- When `C` has finite products indexed by `J`, `W : MorphismProperty C` contains
identities and is stable by products indexed by `J`,
then any localization functor for `W` preserves finite products indexed by `J`. -/
lemma preservesProductsOfShape (J : Type) [Finite J]
    [HasProductsOfShape J C] (hW : W.IsStableUnderProductsOfShape J) :
    PreservesLimitsOfShape (Discrete J) L where
  preservesLimit {F} := preservesLimit_of_preserves_limit_cone (limit.isLimit F)
    (HasProductsOfShapeAux.isLimitMapCone L hW F)


include W in
lemma hasFiniteProducts : HasFiniteProducts D :=
  ⟨fun _ => hasProductsOfShape L W _
    (W.isStableUnderProductsOfShape_of_isStableUnderFiniteProducts _)⟩


include W in
/-- When `C` has finite products and `W : MorphismProperty C` contains
identities and is stable by finite products,
then any localization functor for `W` preserves finite products. -/
lemma preservesFiniteProducts :
    PreservesFiniteProducts L where
  preserves J _ := preservesProductsOfShape L W J
      (W.isStableUnderProductsOfShape_of_isStableUnderFiniteProducts _)


instance : HasFiniteProducts (W.Localization) := hasFiniteProducts W.Q W


noncomputable instance : PreservesFiniteProducts W.Q := preservesFiniteProducts W.Q W


instance [W.HasLocalization] :
    HasFiniteProducts (W.Localization') :=
  hasFiniteProducts W.Q' W


noncomputable instance [W.HasLocalization] :
    PreservesFiniteProducts W.Q' :=
  preservesFiniteProducts W.Q' W


