/-- Promote a monoidal category to a bicategory with a single object.
(The objects of the monoidal category become the 1-morphisms,
with composition given by tensor product,
and the morphisms of the monoidal category become the 2-morphisms.)
-/
@[nolint unusedArguments]
def MonoidalSingleObj (C : Type*) [Category C] [MonoidalCategory C] :=
  PUnit --deriving Inhabited

-- Porting note: `deriving` didn't work. Create this instance manually.

instance : Inhabited (MonoidalSingleObj C) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.82, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    ⊢ Inhabited (CategoryTheory.MonoidalSingleObj C)
  -/
  unfold MonoidalSingleObj
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.82, u_1} C
    inst✝ : CategoryTheory.MonoidalCategory C
    ⊢ Inhabited PUnit.{?u.98}
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : Bicategory (MonoidalSingleObj C) where
  Hom _ _ := C
  id _ := 𝟙_ C
  comp X Y := tensorObj X Y
  whiskerLeft X _ _ f := X ◁ f
  whiskerRight f Z := f ▷ Z
  associator X Y Z := α_ X Y Z
  leftUnitor X := λ_ X
  rightUnitor X := ρ_ X
  whisker_exchange := whisker_exchange


/-- The unique object in the bicategory obtained by "promoting" a monoidal category. -/
@[nolint unusedArguments]
protected def star : MonoidalSingleObj C :=
  PUnit.unit


/-- The monoidal functor from the endomorphisms of the single object
when we promote a monoidal category to a single object bicategory,
to the original monoidal category.

We subsequently show this is an equivalence.
-/
@[simps]
def endMonoidalStarFunctor : (EndMonoidal (MonoidalSingleObj.star C)) ⥤ C where
  obj X := X
  map f := f


instance : (endMonoidalStarFunctor C).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun _ _ ↦ Iso.refl _ }


/-- The equivalence between the endomorphisms of the single object
when we promote a monoidal category to a single object bicategory,
and the original monoidal category.
-/
@[simps]
noncomputable def endMonoidalStarFunctorEquivalence :
    EndMonoidal (MonoidalSingleObj.star C) ≌ C where
  functor := endMonoidalStarFunctor C
  inverse :=
    { obj := fun X => X
      map := fun f => f }
  unitIso := Iso.refl _
  counitIso := Iso.refl _


instance endMonoidalStarFunctor_isEquivalence : (endMonoidalStarFunctor C).IsEquivalence :=
  (endMonoidalStarFunctorEquivalence C).isEquivalence_functor


