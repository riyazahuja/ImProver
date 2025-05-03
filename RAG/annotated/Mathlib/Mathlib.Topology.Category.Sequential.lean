/-- The type sequential topological spaces. -/
structure Sequential where
  /-- The underlying topological space of an object of `Sequential`. -/
  toTop : TopCat.{u}
  /-- The underlying topological space is sequential. -/
  [is_sequential : SequentialSpace toTop]


instance : Inhabited Sequential.{u} :=
  ⟨{ toTop := { α := ULift (Fin 37) } }⟩


instance : CoeSort Sequential Type* :=
  ⟨fun X => X.toTop⟩


instance : Category.{u, u+1} Sequential.{u} :=
  InducedCategory.category toTop


instance : ConcreteCategory.{u} Sequential.{u} :=
  InducedCategory.concreteCategory _


/-- Constructor for objects of the category `Sequential`. -/
def of : Sequential.{u} where
  toTop := TopCat.of X
  is_sequential := ‹_›


/-- The fully faithful embedding of `Sequential` in `TopCat`. -/
@[simps!]
def sequentialToTop : Sequential.{u} ⥤ TopCat.{u} :=
  inducedFunctor _


/-- The functor to `TopCat` is indeed fully faithful.-/
def fullyFaithfulSequentialToTop : sequentialToTop.FullyFaithful :=
  fullyFaithfulInducedFunctor _


instance : sequentialToTop.{u}.Full  :=
  inferInstanceAs (inducedFunctor _).Full


instance : sequentialToTop.{u}.Faithful :=
  inferInstanceAs (inducedFunctor _).Faithful


/-- Construct an isomorphism from a homeomorphism. -/
@[simps hom inv]
def isoOfHomeo {X Y : Sequential.{u}} (f : X ≃ₜ Y) : X ≅ Y where
  hom := ⟨f, f.continuous⟩
  inv := ⟨f.symm, f.symm.continuous⟩
  hom_inv_id := by
    /-
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : SequentialSpace X✝
      X Y : Sequential
      f : Homeomorph ↑X.toTop ↑Y.toTop
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑f, continuous_toFun := ⋯  …
    -/
    ext x
    /-
      case w
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : SequentialSpace X✝
      X Y : Sequential
      f : Homeomorph ↑X.toTop ↑Y.toTop
      x : (CategoryTheory.forget Sequential).obj X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑f, continuous_toFun := ⋯ …
    -/
    exact f.symm_apply_apply x
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : SequentialSpace X✝
      X Y : Sequential
      f : Homeomorph ↑X.toTop ↑Y.toTop
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑f.symm, continuous_toFun  …
    -/
    ext x
    /-
      case w
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : SequentialSpace X✝
      X Y : Sequential
      f : Homeomorph ↑X.toTop ↑Y.toTop
      x : (CategoryTheory.forget Sequential).obj Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑f.symm, continuous_toFun …
    -/
    exact f.apply_symm_apply x
    /-
      🎉 no goals
    -/


/-- Construct a homeomorphism from an isomorphism. -/
@[simps]
def homeoOfIso {X Y : Sequential.{u}} (f : X ≅ Y) : X ≃ₜ Y where
  toFun := f.hom
  invFun := f.inv
                   /-
                     X✝ : Type u
                     inst✝¹ : TopologicalSpace X✝
                     inst✝ : SequentialSpace X✝
                     X Y : Sequential
                     f : CategoryTheory.Iso X Y
                     x : ↑X.toTop
                     ⊢ Eq (f.inv (f.hom x)) x
                   -/
  left_inv x := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      X✝ : Type u
                      inst✝¹ : TopologicalSpace X✝
                      inst✝ : SequentialSpace X✝
                      X Y : Sequential
                      f : CategoryTheory.Iso X Y
                      x : ↑Y.toTop
                      ⊢ Eq (f.hom (f.inv x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/
  continuous_toFun := f.hom.continuous
  continuous_invFun := f.inv.continuous


/-- The equivalence between isomorphisms in `Sequential` and homeomorphisms
of topological spaces. -/
@[simps]
def isoEquivHomeo {X Y : Sequential.{u}} : (X ≅ Y) ≃ (X ≃ₜ Y) where
  toFun := homeoOfIso
  invFun := isoOfHomeo
  left_inv f := by
    /-
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : SequentialSpace X✝
      X Y : Sequential
      f : CategoryTheory.Iso X Y
      ⊢ Eq (Sequential.isoOfHomeo (Sequential.homeoOfIso f)) f
    -/
    ext
    /-
      case w.w
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : SequentialSpace X✝
      X Y : Sequential
      f : CategoryTheory.Iso X Y
      x✝ : (CategoryTheory.forget Sequential).obj X
      ⊢ Eq ((Sequential.isoOfHomeo (Sequential.homeoOfIso f)).hom x✝) (f.hom x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : SequentialSpace X✝
      X Y : Sequential
      f : Homeomorph ↑X.toTop ↑Y.toTop
      ⊢ Eq (Sequential.homeoOfIso (Sequential.isoOfHomeo f)) f
    -/
    ext
    /-
      case H
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : SequentialSpace X✝
      X Y : Sequential
      f : Homeomorph ↑X.toTop ↑Y.toTop
      x✝ : ↑X.toTop
      ⊢ Eq ((Sequential.homeoOfIso (Sequential.isoOfHomeo f)) x✝) (f x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/


