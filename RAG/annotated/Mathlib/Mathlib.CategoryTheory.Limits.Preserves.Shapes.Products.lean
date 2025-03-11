/-- The map of a fan is a limit iff the fan consisting of the mapped morphisms is a limit. This
essentially lets us commute `Fan.mk` with `Functor.mapCone`.
-/
def isLimitMapConeFanMkEquiv {P : C} (g : ∀ j, P ⟶ f j) :
    IsLimit (Functor.mapCone G (Fan.mk P g)) ≃
      IsLimit (Fan.mk _ fun j => G.map (g j) : Fan fun j => G.obj (f j)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    P : C
    g : (j : J) → Quiver.Hom P (f j)
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.Fan.m …
  -/
  refine (IsLimit.postcomposeHomEquiv ?_ _).symm.trans (IsLimit.equivIsoLimit ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      J : Type w
      f : J → C
      P : C
      g : (j : J) → Quiver.Hom P (f j)
      ⊢ CategoryTheory.Iso ((CategoryTheory.Discrete.functor f).comp G) (CategoryThe …
    -/
  · refine Discrete.natIso fun j => Iso.refl (G.obj (f j.as))
    /-
      🎉 no goals
    -/
  refine Cones.ext (Iso.refl _) fun j =>
      by dsimp; cases j; simp


/-- The property of preserving products expressed in terms of fans. -/
def isLimitFanMkObjOfIsLimit [PreservesLimit (Discrete.functor f) G] {P : C} (g : ∀ j, P ⟶ f j)
    (t : IsLimit (Fan.mk _ g)) :
    IsLimit (Fan.mk (G.obj P) fun j => G.map (g j) : Fan fun j => G.obj (f j)) :=
  isLimitMapConeFanMkEquiv _ _ _ (isLimitOfPreserves G t)


/-- The property of reflecting products expressed in terms of fans. -/
def isLimitOfIsLimitFanMkObj [ReflectsLimit (Discrete.functor f) G] {P : C} (g : ∀ j, P ⟶ f j)
    (t : IsLimit (Fan.mk _ fun j => G.map (g j) : Fan fun j => G.obj (f j))) :
    IsLimit (Fan.mk P g) :=
  isLimitOfReflects G ((isLimitMapConeFanMkEquiv _ _ _).symm t)


/--
If `G` preserves products and `C` has them, then the fan constructed of the mapped projection of a
product is a limit.
-/
def isLimitOfHasProductOfPreservesLimit [PreservesLimit (Discrete.functor f) G] :
    IsLimit (Fan.mk _ fun j : J => G.map (Pi.π f j) : Fan fun j => G.obj (f j)) :=
  isLimitFanMkObjOfIsLimit G f _ (productIsProduct _)


/-- If `pi_comparison G f` is an isomorphism, then `G` preserves the limit of `f`. -/
lemma PreservesProduct.of_iso_comparison [i : IsIso (piComparison G f)] :
    PreservesLimit (Discrete.functor f) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasProduct f
    inst✝ : CategoryTheory.Limits.HasProduct fun j => G.obj (f j)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.piComparison G f)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor f) G
  -/
  apply preservesLimit_of_preserves_limit_cone (productIsProduct f)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasProduct f
    inst✝ : CategoryTheory.Limits.HasProduct fun j => G.obj (f j)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.piComparison G f)
    ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.Fan.mk (Cate …
  -/
  apply (isLimitMapConeFanMkEquiv _ _ _).symm _
  exact @IsLimit.ofPointIso _ _ _ _ _ _ _
    (limit.isLimit (Discrete.functor fun j : J => G.obj (f j))) i


/--
If `G` preserves limits, we have an isomorphism from the image of a product to the product of the
images.
-/
def PreservesProduct.iso : G.obj (∏ᶜ f) ≅ ∏ᶜ fun j => G.obj (f j) :=
  IsLimit.conePointUniqueUpToIso (isLimitOfHasProductOfPreservesLimit G f) (limit.isLimit _)


@[simp]
theorem PreservesProduct.iso_hom : (PreservesProduct.iso G f).hom = piComparison G f :=
  rfl


instance : IsIso (piComparison G f) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    inst✝² : CategoryTheory.Limits.HasProduct f
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => G.obj (f j)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.piComparison G f)
  -/
  rw [← PreservesProduct.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    inst✝² : CategoryTheory.Limits.HasProduct f
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => G.obj (f j)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesProduct.iso G f).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The map of a cofan is a colimit iff the cofan consisting of the mapped morphisms is a colimit.
This essentially lets us commute `Cofan.mk` with `Functor.mapCocone`.
-/
def isColimitMapCoconeCofanMkEquiv {P : C} (g : ∀ j, f j ⟶ P) :
    IsColimit (Functor.mapCocone G (Cofan.mk P g)) ≃
      IsColimit (Cofan.mk _ fun j => G.map (g j) : Cofan fun j => G.obj (f j)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    P : C
    g : (j : J) → Quiver.Hom (f j) P
    ⊢ Equiv (CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.C …
  -/
  refine (IsColimit.precomposeHomEquiv ?_ _).symm.trans (IsColimit.equivIsoColimit ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      J : Type w
      f : J → C
      P : C
      g : (j : J) → Quiver.Hom (f j) P
      ⊢ CategoryTheory.Iso (CategoryTheory.Discrete.functor fun j => G.obj (f j)) (( …
    -/
  · refine Discrete.natIso fun j => Iso.refl (G.obj (f j.as))
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    P : C
    g : (j : J) → Quiver.Hom (f j) P
    ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cocones.precompose (CategoryTheor …
  -/
  refine Cocones.ext (Iso.refl _) fun j => by dsimp; cases j; simp
  /-
    🎉 no goals
  -/


/-- The property of preserving coproducts expressed in terms of cofans. -/
def isColimitCofanMkObjOfIsColimit [PreservesColimit (Discrete.functor f) G] {P : C}
    (g : ∀ j, f j ⟶ P) (t : IsColimit (Cofan.mk _ g)) :
    IsColimit (Cofan.mk (G.obj P) fun j => G.map (g j) : Cofan fun j => G.obj (f j)) :=
  isColimitMapCoconeCofanMkEquiv _ _ _ (isColimitOfPreserves G t)


/-- The property of reflecting coproducts expressed in terms of cofans. -/
def isColimitOfIsColimitCofanMkObj [ReflectsColimit (Discrete.functor f) G] {P : C}
    (g : ∀ j, f j ⟶ P)
    (t : IsColimit (Cofan.mk _ fun j => G.map (g j) : Cofan fun j => G.obj (f j))) :
    IsColimit (Cofan.mk P g) :=
  isColimitOfReflects G ((isColimitMapCoconeCofanMkEquiv _ _ _).symm t)


/-- If `G` preserves coproducts and `C` has them,
then the cofan constructed of the mapped inclusion of a coproduct is a colimit.
-/
def isColimitOfHasCoproductOfPreservesColimit [PreservesColimit (Discrete.functor f) G] :
    IsColimit (Cofan.mk _ fun j : J => G.map (Sigma.ι f j) : Cofan fun j => G.obj (f j)) :=
  isColimitCofanMkObjOfIsColimit G f _ (coproductIsCoproduct _)


/-- If `sigma_comparison G f` is an isomorphism, then `G` preserves the colimit of `f`. -/
lemma PreservesCoproduct.of_iso_comparison [i : IsIso (sigmaComparison G f)] :
    PreservesColimit (Discrete.functor f) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasCoproduct f
    inst✝ : CategoryTheory.Limits.HasCoproduct fun j => G.obj (f j)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.sigmaComparison G f)
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Discrete.functor f) G
  -/
  apply preservesColimit_of_preserves_colimit_cocone (coproductIsCoproduct f)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    inst✝¹ : CategoryTheory.Limits.HasCoproduct f
    inst✝ : CategoryTheory.Limits.HasCoproduct fun j => G.obj (f j)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.sigmaComparison G f)
    ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.Cofan.mk …
  -/
  apply (isColimitMapCoconeCofanMkEquiv _ _ _).symm _
  exact @IsColimit.ofPointIso _ _ _ _ _ _ _
    (colimit.isColimit (Discrete.functor fun j : J => G.obj (f j))) i


/-- If `G` preserves colimits,
we have an isomorphism from the image of a coproduct to the coproduct of the images.
-/
def PreservesCoproduct.iso : G.obj (∐ f) ≅ ∐ fun j => G.obj (f j) :=
  IsColimit.coconePointUniqueUpToIso (isColimitOfHasCoproductOfPreservesColimit G f)
    (colimit.isColimit _)


@[simp]
theorem PreservesCoproduct.inv_hom : (PreservesCoproduct.iso G f).inv = sigmaComparison G f := rfl


instance : IsIso (sigmaComparison G f) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    inst✝² : CategoryTheory.Limits.HasCoproduct f
    inst✝¹ : CategoryTheory.Limits.HasCoproduct fun j => G.obj (f j)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Discrete.functo …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.sigmaComparison G f)
  -/
  rw [← PreservesCoproduct.inv_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    J : Type w
    f : J → C
    inst✝² : CategoryTheory.Limits.HasCoproduct f
    inst✝¹ : CategoryTheory.Limits.HasCoproduct fun j => G.obj (f j)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Discrete.functo …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesCoproduct.iso G f).inv
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `F` preserves the limit of every `Discrete.functor f`, it preserves all limits of shape
`Discrete J`. -/
lemma preservesLimitsOfShape_of_discrete (F : C ⥤ D)
    [∀ (f : J → C), PreservesLimit (Discrete.functor f) F] :
    PreservesLimitsOfShape (Discrete J) F where
  preservesLimit := preservesLimit_of_iso_diagram F (Discrete.natIsoFunctor).symm


/-- If `F` preserves the colimit of every `Discrete.functor f`, it preserves all colimits of shape
`Discrete J`. -/
lemma preservesColimitsOfShape_of_discrete (F : C ⥤ D)
    [∀ (f : J → C), PreservesColimit (Discrete.functor f) F] :
    PreservesColimitsOfShape (Discrete J) F where
  preservesColimit := preservesColimit_of_iso_diagram F (Discrete.natIsoFunctor).symm


