/--
The map of a binary fan is a limit iff the fork consisting of the mapped morphisms is a limit. This
essentially lets us commute `BinaryFan.mk` with `Functor.mapCone`.
-/
def isLimitMapConeBinaryFanEquiv :
    IsLimit (G.mapCone (BinaryFan.mk f g)) ≃ IsLimit (BinaryFan.mk (G.map f) (G.map g)) :=
  (IsLimit.postcomposeHomEquiv (diagramIsoPair _) _).symm.trans
    (IsLimit.equivIsoLimit
      (Cones.ext (Iso.refl _)
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝ : CategoryTheory.Category.{v₂, u₂} D
              G : CategoryTheory.Functor C D
              P X Y Z : C
              f : Quiver.Hom P X
              g : Quiver.Hom P Y
              ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (((Cat …
            -/
                               /-
                                 🎉 no goals
                               -/
        (by rintro (_ | _) <;> simp)))
                               /-
                                 🎉 no goals
                               -/


/-- The property of preserving products expressed in terms of binary fans. -/
def mapIsLimitOfPreservesOfIsLimit [PreservesLimit (pair X Y) G] (l : IsLimit (BinaryFan.mk f g)) :
    IsLimit (BinaryFan.mk (G.map f) (G.map g)) :=
  isLimitMapConeBinaryFanEquiv G f g (isLimitOfPreserves G l)


/-- The property of reflecting products expressed in terms of binary fans. -/
def isLimitOfReflectsOfMapIsLimit [ReflectsLimit (pair X Y) G]
    (l : IsLimit (BinaryFan.mk (G.map f) (G.map g))) : IsLimit (BinaryFan.mk f g) :=
  isLimitOfReflects G ((isLimitMapConeBinaryFanEquiv G f g).symm l)


/-- If `G` preserves binary products and `C` has them, then the binary fan constructed of the mapped
morphisms of the binary product cone is a limit.
-/
def isLimitOfHasBinaryProductOfPreservesLimit [PreservesLimit (pair X Y) G] :
    IsLimit (BinaryFan.mk (G.map (Limits.prod.fst : X ⨯ Y ⟶ X)) (G.map Limits.prod.snd)) :=
  mapIsLimitOfPreservesOfIsLimit G _ _ (prodIsProd X Y)


/-- If the product comparison map for `G` at `(X,Y)` is an isomorphism, then `G` preserves the
pair of `(X,Y)`.
-/
lemma PreservesLimitPair.of_iso_prod_comparison [i : IsIso (prodComparison G X Y)] :
    PreservesLimit (pair X Y) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison G X Y)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) G
  -/
  apply preservesLimit_of_preserves_limit_cone (prodIsProd X Y)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison G X Y)
    ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.BinaryFan.mk …
  -/
  apply (isLimitMapConeBinaryFanEquiv _ _ _).symm _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison G X Y)
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk (G.map Cat …
  -/
  refine @IsLimit.ofPointIso _ _ _ _ _ _ _ (limit.isLimit (pair (G.obj X) (G.obj Y))) ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison G X Y)
    ⊢ CategoryTheory.IsIso ((CategoryTheory.Limits.limit.isLimit (CategoryTheory.L …
  -/
  apply i
  /-
    🎉 no goals
  -/


/-- If `G` preserves the product of `(X,Y)`, then the product comparison map for `G` at `(X,Y)` is
an isomorphism.
-/
def PreservesLimitPair.iso : G.obj (X ⨯ Y) ≅ G.obj X ⨯ G.obj Y :=
  IsLimit.conePointUniqueUpToIso (isLimitOfHasBinaryProductOfPreservesLimit G X Y) (limit.isLimit _)


@[simp]
theorem PreservesLimitPair.iso_hom : (PreservesLimitPair.iso G X Y).hom = prodComparison G X Y :=
  rfl


@[simp, reassoc]
theorem PreservesLimitPair.iso_inv_fst :
    (PreservesLimitPair.iso G X Y).inv ≫ G.map prod.fst = prod.fst := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesLimit …
  -/
  rw [← Iso.cancel_iso_hom_left (PreservesLimitPair.iso G X Y), ← Category.assoc, Iso.hom_inv_id]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (G. …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem PreservesLimitPair.iso_inv_snd :
    (PreservesLimitPair.iso G X Y).inv ≫ G.map prod.snd = prod.snd := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesLimit …
  -/
  rw [← Iso.cancel_iso_hom_left (PreservesLimitPair.iso G X Y), ← Category.assoc, Iso.hom_inv_id]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (G. …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : IsIso (prodComparison G X Y) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    P X Y Z : C
    f : Quiver.Hom P X
    g : Quiver.Hom P Y
    inst✝² : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) G
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison G X Y)
  -/
  rw [← PreservesLimitPair.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    P X Y Z : C
    f : Quiver.Hom P X
    g : Quiver.Hom P Y
    inst✝² : CategoryTheory.Limits.HasBinaryProduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (G.obj X) (G.obj Y)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) G
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesLimitPair.iso G X Y).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The map of a binary cofan is a colimit iff
the cofork consisting of the mapped morphisms is a colimit.
This essentially lets us commute `BinaryCofan.mk` with `Functor.mapCocone`.
-/
def isColimitMapCoconeBinaryCofanEquiv :
    IsColimit (Functor.mapCocone G (BinaryCofan.mk f g))
    ≃ IsColimit (BinaryCofan.mk (G.map f) (G.map g)) :=
  (IsColimit.precomposeHomEquiv (diagramIsoPair _).symm _).symm.trans
    (IsColimit.equivIsoColimit
      (Cocones.ext (Iso.refl _)
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝ : CategoryTheory.Category.{v₂, u₂} D
              G : CategoryTheory.Functor C D
              P X Y Z : C
              f : Quiver.Hom X P
              g : Quiver.Hom Y P
              ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
            -/
                               /-
                                 🎉 no goals
                               -/
        (by rintro (_ | _) <;> simp)))
                               /-
                                 🎉 no goals
                               -/


/-- The property of preserving coproducts expressed in terms of binary cofans. -/
def mapIsColimitOfPreservesOfIsColimit [PreservesColimit (pair X Y) G]
    (l : IsColimit (BinaryCofan.mk f g)) : IsColimit (BinaryCofan.mk (G.map f) (G.map g)) :=
  isColimitMapCoconeBinaryCofanEquiv G f g (isColimitOfPreserves G l)


/-- The property of reflecting coproducts expressed in terms of binary cofans. -/
def isColimitOfReflectsOfMapIsColimit [ReflectsColimit (pair X Y) G]
    (l : IsColimit (BinaryCofan.mk (G.map f) (G.map g))) : IsColimit (BinaryCofan.mk f g) :=
  isColimitOfReflects G ((isColimitMapCoconeBinaryCofanEquiv G f g).symm l)


/--
If `G` preserves binary coproducts and `C` has them, then the binary cofan constructed of the mapped
morphisms of the binary product cocone is a colimit.
-/
def isColimitOfHasBinaryCoproductOfPreservesColimit [PreservesColimit (pair X Y) G] :
    IsColimit (BinaryCofan.mk (G.map (Limits.coprod.inl : X ⟶ X ⨿ Y)) (G.map Limits.coprod.inr)) :=
  mapIsColimitOfPreservesOfIsColimit G _ _ (coprodIsCoprod X Y)


/-- If the coproduct comparison map for `G` at `(X,Y)` is an isomorphism, then `G` preserves the
pair of `(X,Y)`.
-/
lemma PreservesColimitPair.of_iso_coprod_comparison [i : IsIso (coprodComparison G X Y)] :
    PreservesColimit (pair X Y) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct (G.obj X) (G.obj Y)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison G X Y)
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.pair X Y) G
  -/
  apply preservesColimit_of_preserves_colimit_cocone (coprodIsCoprod X Y)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct (G.obj X) (G.obj Y)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison G X Y)
    ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.BinaryCo …
  -/
  apply (isColimitMapCoconeBinaryCofanEquiv _ _ _).symm _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct (G.obj X) (G.obj Y)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison G X Y)
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (G.map …
  -/
  refine @IsColimit.ofPointIso _ _ _ _ _ _ _ (colimit.isColimit (pair (G.obj X) (G.obj Y))) ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct (G.obj X) (G.obj Y)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison G X Y)
    ⊢ CategoryTheory.IsIso ((CategoryTheory.Limits.colimit.isColimit (CategoryTheo …
  -/
  apply i
  /-
    🎉 no goals
  -/


/--
If `G` preserves the coproduct of `(X,Y)`, then the coproduct comparison map for `G` at `(X,Y)` is
an isomorphism.
-/
def PreservesColimitPair.iso : G.obj X ⨿ G.obj Y ≅ G.obj (X ⨿ Y) :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit _)
    (isColimitOfHasBinaryCoproductOfPreservesColimit G X Y)


@[simp]
theorem PreservesColimitPair.iso_hom :
    (PreservesColimitPair.iso G X Y).hom = coprodComparison G X Y := rfl


instance : IsIso (coprodComparison G X Y) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    P X Y Z : C
    f : Quiver.Hom X P
    g : Quiver.Hom Y P
    inst✝² : CategoryTheory.Limits.HasBinaryCoproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct (G.obj X) (G.obj Y)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.pair X Y …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison G X Y)
  -/
  rw [← PreservesColimitPair.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    P X Y Z : C
    f : Quiver.Hom X P
    g : Quiver.Hom Y P
    inst✝² : CategoryTheory.Limits.HasBinaryCoproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct (G.obj X) (G.obj Y)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.pair X Y …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesColimitPair.iso G X Y). …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


