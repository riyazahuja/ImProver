instance {J : Type} [Finite J] (Z : J → ModuleCat.{v} k) [∀ j, FiniteDimensional k (Z j)] :
    FiniteDimensional k (∏ᶜ fun j => Z j : ModuleCat.{v} k) :=
                                                                /-
                                                                  J✝ : Type
                                                                  inst✝⁴ : CategoryTheory.SmallCategory J✝
                                                                  inst✝³ : CategoryTheory.FinCategory J✝
                                                                  k : Type v
                                                                  inst✝² : Field k
                                                                  J : Type
                                                                  inst✝¹ : Finite J
                                                                  Z : J → ModuleCat k
                                                                  inst✝ : ∀ (j : J), FiniteDimensional k ↑(Z j)
                                                                  ⊢ FiniteDimensional k ↑(ModuleCat.of k ((j : J) → ↑(Z j)))
                                                                -/
  haveI : FiniteDimensional k (ModuleCat.of k (∀ j, Z j)) := by unfold ModuleCat.of; infer_instance
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
  FiniteDimensional.of_injective (ModuleCat.piIsoPi _).hom.hom
                                            /-
                                              J✝ : Type
                                              inst✝⁴ : CategoryTheory.SmallCategory J✝
                                              inst✝³ : CategoryTheory.FinCategory J✝
                                              k : Type v
                                              inst✝² : Field k
                                              J : Type
                                              inst✝¹ : Finite J
                                              Z : J → ModuleCat k
                                              inst✝ : ∀ (j : J), FiniteDimensional k ↑(Z j)
                                              this : FiniteDimensional k ↑(ModuleCat.of k ((j : J) → ↑(Z j)))
                                              ⊢ CategoryTheory.Mono (ModuleCat.piIsoPi fun j => Z j).hom
                                            -/
    ((ModuleCat.mono_iff_injective _).1 (by infer_instance))
                                            /-
                                              🎉 no goals
                                            -/


/-- Finite limits of finite dimensional vectors spaces are finite dimensional,
because we can realise them as subobjects of a finite product. -/
instance (F : J ⥤ FGModuleCat k) :
    FiniteDimensional k (limit (F ⋙ forget₂ (FGModuleCat k) (ModuleCat.{v} k)) : ModuleCat.{v} k) :=
  haveI : ∀ j, FiniteDimensional k ((F ⋙ forget₂ (FGModuleCat k) (ModuleCat.{v} k)).obj j) := by
    /-
      J : Type
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.FinCategory J
      k : Type v
      inst✝ : Field k
      F : CategoryTheory.Functor J (FGModuleCat k)
      ⊢ ∀ (j : J), FiniteDimensional k ↑((F.comp (CategoryTheory.forget₂ (FGModuleCa …
    -/
    intro j; change FiniteDimensional k (F.obj j); infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/
  FiniteDimensional.of_injective
    (limitSubobjectProduct (F ⋙ forget₂ (FGModuleCat k) (ModuleCat.{v} k))).hom
    ((ModuleCat.mono_iff_injective _).1 inferInstance)


/-- The forgetful functor from `FGModuleCat k` to `ModuleCat k` creates all finite limits. -/
def forget₂CreatesLimit (F : J ⥤ FGModuleCat k) :
    CreatesLimit F (forget₂ (FGModuleCat k) (ModuleCat.{v} k)) :=
  createsLimitOfFullyFaithfulOfIso
    ⟨(limit (F ⋙ forget₂ (FGModuleCat k) (ModuleCat.{v} k)) : ModuleCat.{v} k), inferInstance⟩
    (Iso.refl _)


instance : CreatesLimitsOfShape J (forget₂ (FGModuleCat k) (ModuleCat.{v} k)) where
  CreatesLimit {F} := forget₂CreatesLimit F


instance (J : Type) [Category J] [FinCategory J] :
    HasLimitsOfShape J (FGModuleCat.{v} k) :=
  hasLimitsOfShape_of_hasLimitsOfShape_createsLimitsOfShape
    (forget₂ (FGModuleCat k) (ModuleCat.{v} k))


instance : HasFiniteLimits (FGModuleCat k) where
  out _ _ _ := inferInstance


instance : PreservesFiniteLimits (forget₂ (FGModuleCat k) (ModuleCat.{v} k)) where
  preservesFiniteLimits _ _ _ := inferInstance


