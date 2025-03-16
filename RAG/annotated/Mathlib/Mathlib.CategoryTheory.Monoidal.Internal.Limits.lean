/-- We construct the (candidate) limit of a functor `F : J ⥤ Mon_ C`
by interpreting it as a functor `Mon_ (J ⥤ C)`,
and noting that taking limits is a lax monoidal functor,
and hence sends monoid objects to monoid objects.
-/
@[simps!]
def limit (F : J ⥤ Mon_ C) : Mon_ C :=
  lim.mapMon.obj ((monFunctorCategoryEquivalence J C).inverse.obj F)


/-- Implementation of `Mon_.hasLimits`: a limiting cone over a functor `F : J ⥤ Mon_ C`.
-/
@[simps]
def limitCone (F : J ⥤ Mon_ C) : Cone F where
  pt := limit F
  π :=
    { app := fun j => { hom := limit.π (F ⋙ Mon_.forget C) j }
                                     /-
                                       J : Type w
                                       inst✝³ : CategoryTheory.SmallCategory J
                                       C : Type u
                                       inst✝² : CategoryTheory.Category.{v, u} C
                                       inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
                                       inst✝ : CategoryTheory.MonoidalCategory C
                                       F : CategoryTheory.Functor J (Mon_ C)
                                       j j' : J
                                       f : Quiver.Hom j j'
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                                     -/
      naturality := fun j j' f => by ext; exact (limit.cone (F ⋙ Mon_.forget C)).π.naturality f }
                                          /-
                                            🎉 no goals
                                          -/


/-- The image of the proposed limit cone for `F : J ⥤ Mon_ C` under the forgetful functor
`forget C : Mon_ C ⥤ C` is isomorphic to the limit cone of `F ⋙ forget C`.
-/
def forgetMapConeLimitConeIso (F : J ⥤ Mon_ C) :
    (forget C).mapCone (limitCone F) ≅ limit.cone (F ⋙ forget C) :=
                             /-
                               J : Type w
                               inst✝³ : CategoryTheory.SmallCategory J
                               C : Type u
                               inst✝² : CategoryTheory.Category.{v, u} C
                               inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
                               inst✝ : CategoryTheory.MonoidalCategory C
                               F : CategoryTheory.Functor J (Mon_ C)
                               ⊢ ∀ (j : J), Eq (((Mon_.forget C).mapCone (Mon_.limitCone F)).π.app j) (Catego …
                             -/
  Cones.ext (Iso.refl _) (by aesop_cat)
                             /-
                               🎉 no goals
                             -/


/-- Implementation of `Mon_.hasLimitsOfShape`:
the proposed cone over a functor `F : J ⥤ Mon_ C` is a limit cone.
-/
@[simps]
def limitConeIsLimit (F : J ⥤ Mon_ C) : IsLimit (limitCone F) where
  lift s :=
    { hom := limit.lift (F ⋙ Mon_.forget C) ((Mon_.forget C).mapCone s)
      mul_hom := limit.hom_ext (fun j ↦ by
        /-
          J : Type w
          inst✝³ : CategoryTheory.SmallCategory J
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
          inst✝ : CategoryTheory.MonoidalCategory C
          F : CategoryTheory.Functor J (Mon_ C)
          s : CategoryTheory.Limits.Cone F
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
        -/
        dsimp
        simp only [Category.assoc, limit.lift_π, Functor.mapCone_pt, forget_obj,
          Functor.mapCone_π_app, forget_map, Hom.mul_hom, limMap_π, tensorObj_obj, Functor.comp_obj,
          MonFunctorCategoryEquivalence.inverseObj_mul_app, lim_μ_π_assoc, lim_obj,
          ← MonoidalCategory.tensor_comp_assoc]) }
                /-
                  J : Type w
                  inst✝³ : CategoryTheory.SmallCategory J
                  C : Type u
                  inst✝² : CategoryTheory.Category.{v, u} C
                  inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
                  inst✝ : CategoryTheory.MonoidalCategory C
                  F : CategoryTheory.Functor J (Mon_ C)
                  s : CategoryTheory.Limits.Cone F
                  h : J
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => { hom := CategoryTheory.Li …
                -/
  fac s h := by ext; simp
                     /-
                       🎉 no goals
                     -/
  uniq s m w := by
    /-
      J : Type w
      inst✝³ : CategoryTheory.SmallCategory J
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝ : CategoryTheory.MonoidalCategory C
      F : CategoryTheory.Functor J (Mon_ C)
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (Mon_.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((Mon_.limitCone F).π. …
      ⊢ Eq m ((fun s => { hom := CategoryTheory.Limits.limit.lift (F.comp (Mon_.forg …
    -/
    ext1
    /-
      case w
      J : Type w
      inst✝³ : CategoryTheory.SmallCategory J
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝ : CategoryTheory.MonoidalCategory C
      F : CategoryTheory.Functor J (Mon_ C)
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (Mon_.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((Mon_.limitCone F).π. …
      ⊢ Eq m.hom ((fun s => { hom := CategoryTheory.Limits.limit.lift (F.comp (Mon_. …
    -/
    refine limit.hom_ext (fun j => ?_)
    /-
      case w
      J : Type w
      inst✝³ : CategoryTheory.SmallCategory J
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝ : CategoryTheory.MonoidalCategory C
      F : CategoryTheory.Functor J (Mon_ C)
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (Mon_.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((Mon_.limitCone F).π. …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m.hom (CategoryTheory.Limits.limit.π  …
    -/
    dsimp; simp only [Mon_.forget_map, limit.lift_π, Functor.mapCone_π_app]
    /-
      case w
      J : Type w
      inst✝³ : CategoryTheory.SmallCategory J
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
      inst✝ : CategoryTheory.MonoidalCategory C
      F : CategoryTheory.Functor J (Mon_ C)
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (Mon_.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((Mon_.limitCone F).π. …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m.hom (CategoryTheory.Limits.limit.π  …
    -/
    exact congr_arg Mon_.Hom.hom (w j)
    /-
      🎉 no goals
    -/


instance hasLimitsOfShape [HasLimitsOfShape J C] : HasLimitsOfShape J (Mon_ C) where
  has_limit := fun F => HasLimit.mk
    { cone := limitCone F
      isLimit := limitConeIsLimit F }


instance forget_freservesLimitsOfShape : PreservesLimitsOfShape J (Mon_.forget C) where
  preservesLimit := fun {F} =>
    preservesLimit_of_preserves_limit_cone (limitConeIsLimit F)
      (IsLimit.ofIsoLimit (limit.isLimit (F ⋙ Mon_.forget C)) (forgetMapConeLimitConeIso F).symm)


