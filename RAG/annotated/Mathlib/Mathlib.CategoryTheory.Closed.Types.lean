/-- The adjunction `tensorLeft.obj X ⊣ coyoneda.obj (Opposite.op X)`
for any `X : Type v₁`. -/
def Types.tensorProductAdjunction (X : Type v₁) :
    tensorLeft X ⊣ coyoneda.obj (Opposite.op X) where
  unit := { app := fun Z (z : Z) x => ⟨x, z⟩ }
  counit := { app := fun _ xf => xf.2 xf.1 }


instance (X : Type v₁) : (tensorLeft X).IsLeftAdjoint :=
  ⟨_, ⟨Types.tensorProductAdjunction X⟩⟩


instance : CartesianClosed (Type v₁) := CartesianClosed.mk _
  (fun X => Exponentiable.mk _ _ (Types.tensorProductAdjunction X))


instance {C : Type v₁} [SmallCategory C] : CartesianClosed (C ⥤ Type v₁) :=
  CartesianClosed.mk _
    (fun F => by
      /-
        C✝ : Type v₂
        inst✝¹ : CategoryTheory.Category.{v₁, v₂} C✝
        C : Type v₁
        inst✝ : CategoryTheory.SmallCategory C
        F : CategoryTheory.Functor C (Type v₁)
        ⊢ CategoryTheory.Exponentiable F
      -/
      haveI : ∀ X : Type v₁, PreservesColimits (tensorLeft X) := by infer_instance
      /-
        C✝ : Type v₂
        inst✝¹ : CategoryTheory.Category.{v₁, v₂} C✝
        C : Type v₁
        inst✝ : CategoryTheory.SmallCategory C
        F : CategoryTheory.Functor C (Type v₁)
        this : ∀ (X : Type v₁), CategoryTheory.Limits.PreservesColimits (CategoryTheor …
        ⊢ CategoryTheory.Exponentiable F
      -/
      letI : PreservesColimits (tensorLeft F) := ⟨by infer_instance⟩
      /-
        C✝ : Type v₂
        inst✝¹ : CategoryTheory.Category.{v₁, v₂} C✝
        C : Type v₁
        inst✝ : CategoryTheory.SmallCategory C
        F : CategoryTheory.Functor C (Type v₁)
        this✝ : ∀ (X : Type v₁), CategoryTheory.Limits.PreservesColimits (CategoryTheo …
        this : CategoryTheory.Limits.PreservesColimits (CategoryTheory.MonoidalCategor …
        ⊢ CategoryTheory.Exponentiable F
      -/
      have := Presheaf.isLeftAdjoint_of_preservesColimits (tensorLeft F)
      /-
        C✝ : Type v₂
        inst✝¹ : CategoryTheory.Category.{v₁, v₂} C✝
        C : Type v₁
        inst✝ : CategoryTheory.SmallCategory C
        F : CategoryTheory.Functor C (Type v₁)
        this✝¹ : ∀ (X : Type v₁), CategoryTheory.Limits.PreservesColimits (CategoryThe …
        this✝ : CategoryTheory.Limits.PreservesColimits (CategoryTheory.MonoidalCatego …
        this : (CategoryTheory.MonoidalCategory.tensorLeft F).IsLeftAdjoint
        ⊢ CategoryTheory.Exponentiable F
      -/
      exact Exponentiable.mk _ _ (Adjunction.ofIsLeftAdjoint (tensorLeft F)))
      /-
        🎉 no goals
      -/

-- TODO: once we have `MonoidalClosed` instances for functor categories into general monoidal
-- closed categories, replace this with that, as it will be a more explicit construction.

/-- This is not a good instance because of the universe levels. Below is the instance where the
target category is `Type (max u₁ v₁)`. -/
def cartesianClosedFunctorToTypes {C : Type u₁} [Category.{v₁} C] :
    CartesianClosed (C ⥤ Type (max u₁ v₁ u₂)) :=
  let e : (ULiftHom.{max u₁ v₁ u₂} (ULift.{max u₁ v₁ u₂} C)) ⥤ Type (max u₁ v₁ u₂) ≌
      C ⥤ Type (max u₁ v₁ u₂) :=
      Functor.asEquivalence ((whiskeringLeft _ _ _).obj
        (ULift.equivalence.trans ULiftHom.equiv).functor)
  cartesianClosedOfEquiv e

-- TODO: once we have `MonoidalClosed` instances for functor categories into general monoidal
-- closed categories, replace this with that, as it will be a more explicit construction.

instance {C : Type u₁} [Category.{v₁} C] : CartesianClosed (C ⥤ Type (max u₁ v₁)) :=
  cartesianClosedFunctorToTypes

-- TODO: once we have `MonoidalClosed` instances for functor categories into general monoidal
-- closed categories, replace this with that, as it will be a more explicit construction.

instance {C : Type u₁} [Category.{v₁} C] [EssentiallySmall.{v₁} C] :
    CartesianClosed (C ⥤ Type v₁) :=
  let e : (SmallModel C) ⥤ Type v₁ ≌ C ⥤ Type v₁ :=
    Functor.asEquivalence ((whiskeringLeft _ _ _).obj (equivSmallModel _).functor)
  cartesianClosedOfEquiv e


