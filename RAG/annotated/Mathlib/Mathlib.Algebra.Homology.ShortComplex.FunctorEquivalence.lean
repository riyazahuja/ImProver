/-- The obvious functor `ShortComplex (J ⥤ C) ⥤ J ⥤ ShortComplex C`. -/
@[simps]
def functor : ShortComplex (J ⥤ C) ⥤ J ⥤ ShortComplex C where
  obj S :=
    { obj := fun j => S.map ((evaluation J C).obj j)
      map := fun f => S.mapNatTrans ((evaluation J C).map f) }
  map φ :=
    { app := fun j => ((evaluation J C).obj j).mapShortComplex.map φ }


/-- The obvious functor `(J ⥤ ShortComplex C) ⥤ ShortComplex (J ⥤ C)`. -/
@[simps]
def inverse : (J ⥤ ShortComplex C) ⥤ ShortComplex (J ⥤ C) where
  obj F :=
    { f := whiskerLeft F π₁Toπ₂
      g := whiskerLeft F π₂Toπ₃
                 /-
                   J : Type u_1
                   C : Type u_2
                   inst✝² : CategoryTheory.Category.{?u.17620, u_1} J
                   inst✝¹ : CategoryTheory.Category.{?u.17624, u_2} C
                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                   F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft F Categor …
                 -/
      zero := by aesop_cat }
                 /-
                   🎉 no goals
                 -/
  map φ := Hom.mk (whiskerRight φ π₁) (whiskerRight φ π₂) (whiskerRight φ π₃)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.17620, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.17624, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X✝ Y✝ : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          φ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight φ Catego …
        -/
        /-
          🎉 no goals
        -/
    (by aesop_cat) (by aesop_cat)
                       /-
                         🎉 no goals
                       -/


/-- The unit isomorphism of the equivalence
`ShortComplex.functorEquivalence : ShortComplex (J ⥤ C) ≌ J ⥤ ShortComplex C`. -/
@[simps!]
def unitIso : 𝟭 _ ≅ functor J C ⋙ inverse J C :=
  NatIso.ofComponents (fun _ => isoMk
                                                   /-
                                                     J : Type u_1
                                                     C : Type u_2
                                                     inst✝² : CategoryTheory.Category.{?u.31931, u_1} J
                                                     inst✝¹ : CategoryTheory.Category.{?u.31935, u_2} C
                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                     x✝ : CategoryTheory.ShortComplex (CategoryTheory.Functor J C)
                                                     ⊢ ∀ {X Y : J} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                                                   -/
    (NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat))
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     J : Type u_1
                                                     C : Type u_2
                                                     inst✝² : CategoryTheory.Category.{?u.31931, u_1} J
                                                     inst✝¹ : CategoryTheory.Category.{?u.31935, u_2} C
                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                     x✝ : CategoryTheory.ShortComplex (CategoryTheory.Functor J C)
                                                     ⊢ ∀ {X Y : J} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                                                   -/
    (NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat))
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     J : Type u_1
                                                     C : Type u_2
                                                     inst✝² : CategoryTheory.Category.{?u.31931, u_1} J
                                                     inst✝¹ : CategoryTheory.Category.{?u.31935, u_2} C
                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                     x✝ : CategoryTheory.ShortComplex (CategoryTheory.Functor J C)
                                                     ⊢ ∀ {X Y : J} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                                                   -/
    (NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat))
                                                   /-
                                                     🎉 no goals
                                                   -/
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.31931, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.31935, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          x✝ : CategoryTheory.ShortComplex (CategoryTheory.Functor J C)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NatIso.ofComponents ( …
        -/
        /-
          🎉 no goals
        -/
                       /-
                         🎉 no goals
                       -/
    (by aesop_cat) (by aesop_cat)) (by aesop_cat)
                                       /-
                                         🎉 no goals
                                       -/


/-- The counit isomorphism of the equivalence
`ShortComplex.functorEquivalence : ShortComplex (J ⥤ C) ≌ J ⥤ ShortComplex C`. -/
@[simps!]
def counitIso : inverse J C ⋙ functor J C ≅ 𝟭 _ :=
  NatIso.ofComponents (fun _ => NatIso.ofComponents
    (fun _ => isoMk (Iso.refl _) (Iso.refl _) (Iso.refl _)
          /-
            J : Type u_1
            C : Type u_2
            inst✝² : CategoryTheory.Category.{?u.68136, u_1} J
            inst✝¹ : CategoryTheory.Category.{?u.68140, u_2} C
            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
            x✝¹ : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
            x✝ : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((((Category …
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
      (by aesop_cat) (by aesop_cat)) (by aesop_cat)) (by aesop_cat)
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- The obvious equivalence `ShortComplex (J ⥤ C) ≌ J ⥤ ShortComplex C`. -/
@[simps]
def functorEquivalence : ShortComplex (J ⥤ C) ≌ J ⥤ ShortComplex C where
  functor := FunctorEquivalence.functor J C
  inverse := FunctorEquivalence.inverse J C
  unitIso := FunctorEquivalence.unitIso J C
  counitIso := FunctorEquivalence.counitIso J C


