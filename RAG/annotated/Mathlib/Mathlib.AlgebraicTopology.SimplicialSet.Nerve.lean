/-- The nerve of a category -/
@[simps]
def nerve (C : Type u) [Category.{v} C] : SSet.{max u v} where
  obj Δ := ComposableArrows C (Δ.unop.len)
  map f x := x.whiskerLeft (SimplexCategory.toCat.map f.unop)


instance {C : Type*} [Category C] {Δ : SimplexCategoryᵒᵖ} : Category ((nerve C).obj Δ) :=
  (inferInstance : Category (ComposableArrows C (Δ.unop.len)))


/-- Given a functor `C ⥤ D`, we obtain a morphism `nerve C ⟶ nerve D` of simplicial sets. -/
@[simps]
def nerveMap {C D : Type u} [Category.{v} C] [Category.{v} D] (F : C ⥤ D) : nerve C ⟶ nerve D :=
  { app := fun _ => (F.mapComposableArrows _).obj }


/-- The nerve of a category, as a functor `Cat ⥤ SSet` -/
@[simps]
def nerveFunctor : Cat.{v, u} ⥤ SSet where
  obj C := nerve C
  map F := nerveMap F


/-- The 0-simplices of the nerve of a category are equivalent to the objects of the category. -/
def nerveEquiv (C : Type u) [Category.{v} C] : nerve C _[0] ≃ C where
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            f : (CategoryTheory.nerve C).obj { unop := SimplexCategory.mk 0 }
                            ⊢ LT.lt 0 (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk 0 }).len 1)
                          -/
  toFun f := f.obj ⟨0, by omega⟩
                          /-
                            🎉 no goals
                          -/
  invFun f := (Functor.const _).obj f
  left_inv f := ComposableArrows.ext₀ rfl
  right_inv f := rfl


lemma δ₀_eq {x : nerve C _[n + 1]} : (nerve C).δ (0 : Fin (n + 2)) x = x.δ₀ := rfl


