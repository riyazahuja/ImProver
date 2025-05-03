/-- `CompactlyGenerated.{u, w}` is the type of `u`-compactly generated `w`-small topological spaces.
This should always be used with explicit universe parameters. -/
structure CompactlyGenerated where
  /-- The underlying topological space of an object of `CompactlyGenerated`. -/
  toTop : TopCat.{w}
  /-- The underlying topological space is compactly generated. -/
  [is_compactly_generated : UCompactlyGeneratedSpace.{u} toTop]


instance : Inhabited CompactlyGenerated.{u, w} :=
  ⟨{ toTop := { α := ULift (Fin 37) } }⟩


instance : CoeSort CompactlyGenerated Type* :=
  ⟨fun X => X.toTop⟩


instance : Category.{w, w+1} CompactlyGenerated.{u, w} :=
  InducedCategory.category toTop


instance : ConcreteCategory.{w} CompactlyGenerated.{u, w} :=
  InducedCategory.concreteCategory _


/-- Constructor for objects of the category `CompactlyGenerated`. -/
def of : CompactlyGenerated.{u, w} where
  toTop := TopCat.of X
  is_compactly_generated := ‹_›


/-- The fully faithful embedding of `CompactlyGenerated` in `TopCat`. -/
@[simps!]
def compactlyGeneratedToTop : CompactlyGenerated.{u, w} ⥤ TopCat.{w} :=
  inducedFunctor _


/-- `compactlyGeneratedToTop` is fully faithful. -/
def fullyFaithfulCompactlyGeneratedToTop : compactlyGeneratedToTop.{u, w}.FullyFaithful :=
  fullyFaithfulInducedFunctor _


instance : compactlyGeneratedToTop.{u, w}.Full := fullyFaithfulCompactlyGeneratedToTop.full


instance : compactlyGeneratedToTop.{u, w}.Faithful := fullyFaithfulCompactlyGeneratedToTop.faithful


/-- Construct an isomorphism from a homeomorphism. -/
@[simps hom inv]
def isoOfHomeo {X Y : CompactlyGenerated.{u, w}} (f : X ≃ₜ Y) : X ≅ Y where
  hom := ⟨f, f.continuous⟩
  inv := ⟨f.symm, f.symm.continuous⟩
  hom_inv_id := by
    /-
      X✝ : Type w
      inst✝¹ : TopologicalSpace X✝
      inst✝ : UCompactlyGeneratedSpace X✝
      X Y : CompactlyGenerated
      f : Homeomorph ↑X.toTop ↑Y.toTop
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑f, continuous_toFun := ⋯  …
    -/
    ext x
    /-
      case w
      X✝ : Type w
      inst✝¹ : TopologicalSpace X✝
      inst✝ : UCompactlyGeneratedSpace X✝
      X Y : CompactlyGenerated
      f : Homeomorph ↑X.toTop ↑Y.toTop
      x : (CategoryTheory.forget CompactlyGenerated).obj X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑f, continuous_toFun := ⋯ …
    -/
    exact f.symm_apply_apply x
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      X✝ : Type w
      inst✝¹ : TopologicalSpace X✝
      inst✝ : UCompactlyGeneratedSpace X✝
      X Y : CompactlyGenerated
      f : Homeomorph ↑X.toTop ↑Y.toTop
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑f.symm, continuous_toFun  …
    -/
    ext x
    /-
      case w
      X✝ : Type w
      inst✝¹ : TopologicalSpace X✝
      inst✝ : UCompactlyGeneratedSpace X✝
      X Y : CompactlyGenerated
      f : Homeomorph ↑X.toTop ↑Y.toTop
      x : (CategoryTheory.forget CompactlyGenerated).obj Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑f.symm, continuous_toFun …
    -/
    exact f.apply_symm_apply x
    /-
      🎉 no goals
    -/


/-- Construct a homeomorphism from an isomorphism. -/
@[simps]
def homeoOfIso {X Y : CompactlyGenerated.{u, w}} (f : X ≅ Y) : X ≃ₜ Y where
  toFun := f.hom
  invFun := f.inv
                   /-
                     X✝ : Type w
                     inst✝¹ : TopologicalSpace X✝
                     inst✝ : UCompactlyGeneratedSpace X✝
                     X Y : CompactlyGenerated
                     f : CategoryTheory.Iso X Y
                     x : ↑X.toTop
                     ⊢ Eq (f.inv (f.hom x)) x
                   -/
  left_inv x := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      X✝ : Type w
                      inst✝¹ : TopologicalSpace X✝
                      inst✝ : UCompactlyGeneratedSpace X✝
                      X Y : CompactlyGenerated
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


/-- The equivalence between isomorphisms in `CompactlyGenerated` and homeomorphisms
of topological spaces. -/
@[simps]
def isoEquivHomeo {X Y : CompactlyGenerated.{u, w}} : (X ≅ Y) ≃ (X ≃ₜ Y) where
  toFun := homeoOfIso
  invFun := isoOfHomeo
  left_inv f := by
    /-
      X✝ : Type w
      inst✝¹ : TopologicalSpace X✝
      inst✝ : UCompactlyGeneratedSpace X✝
      X Y : CompactlyGenerated
      f : CategoryTheory.Iso X Y
      ⊢ Eq (CompactlyGenerated.isoOfHomeo (CompactlyGenerated.homeoOfIso f)) f
    -/
    ext
    /-
      case w.w
      X✝ : Type w
      inst✝¹ : TopologicalSpace X✝
      inst✝ : UCompactlyGeneratedSpace X✝
      X Y : CompactlyGenerated
      f : CategoryTheory.Iso X Y
      x✝ : (CategoryTheory.forget CompactlyGenerated).obj X
      ⊢ Eq ((CompactlyGenerated.isoOfHomeo (CompactlyGenerated.homeoOfIso f)).hom x✝ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      X✝ : Type w
      inst✝¹ : TopologicalSpace X✝
      inst✝ : UCompactlyGeneratedSpace X✝
      X Y : CompactlyGenerated
      f : Homeomorph ↑X.toTop ↑Y.toTop
      ⊢ Eq (CompactlyGenerated.homeoOfIso (CompactlyGenerated.isoOfHomeo f)) f
    -/
    ext
    /-
      case H
      X✝ : Type w
      inst✝¹ : TopologicalSpace X✝
      inst✝ : UCompactlyGeneratedSpace X✝
      X Y : CompactlyGenerated
      f : Homeomorph ↑X.toTop ↑Y.toTop
      x✝ : ↑X.toTop
      ⊢ Eq ((CompactlyGenerated.homeoOfIso (CompactlyGenerated.isoOfHomeo f)) x✝) (f …
    -/
    rfl
    /-
      🎉 no goals
    -/


