/-- If you rotate a triangle, you get another triangle.
Given a triangle of the form:
```
      f       g       h
  X  ───> Y  ───> Z  ───> X⟦1⟧
```
applying `rotate` gives a triangle of the form:
```
      g       h        -f⟦1⟧'
  Y  ───> Z  ───>  X⟦1⟧ ───> Y⟦1⟧
```
-/
@[simps!]
def Triangle.rotate (T : Triangle C) : Triangle C :=
  Triangle.mk T.mor₂ T.mor₃ (-T.mor₁⟦1⟧')


/-- Given a triangle of the form:
```
      f       g       h
  X  ───> Y  ───> Z  ───> X⟦1⟧
```
applying `invRotate` gives a triangle that can be thought of as:
```
        -h⟦-1⟧'     f       g
  Z⟦-1⟧  ───>  X  ───> Y  ───> Z
```
(note that this diagram doesn't technically fit the definition of triangle, as `Z⟦-1⟧⟦1⟧` is
not necessarily equal to `Z`, but it is isomorphic, by the `counitIso` of `shiftEquiv C 1`)
-/
@[simps!]
def Triangle.invRotate (T : Triangle C) : Triangle C :=
  Triangle.mk (-T.mor₃⟦(-1 : ℤ)⟧' ≫ (shiftEquiv C (1 : ℤ)).unitIso.inv.app _) (T.mor₁)
    (T.mor₂ ≫ (shiftEquiv C (1 : ℤ)).counitIso.inv.app _ )


/-- Rotating triangles gives an endofunctor on the category of triangles in `C`.
-/
@[simps]
def rotate : Triangle C ⥤ Triangle C where
  obj := Triangle.rotate
  map f :=
  { hom₁ := f.hom₂
    hom₂ := f.hom₃
    hom₃ := f.hom₁⟦1⟧'
    comm₃ := by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp X✝.rotate.mor₃ ((CategoryTheory.shift …
      -/
      dsimp
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (Neg.neg ((CategoryTheory.shiftFuncto …
      -/
      simp only [comp_neg, neg_comp, ← Functor.map_comp, f.comm₁] }
      /-
        🎉 no goals
      -/


/-- The inverse rotation of triangles gives an endofunctor on the category of triangles in `C`.
-/
@[simps]
def invRotate : Triangle C ⥤ Triangle C where
  obj := Triangle.invRotate
  map f :=
  { hom₁ := f.hom₃⟦-1⟧'
    hom₂ := f.hom₁
    hom₃ := f.hom₂
    comm₁ := by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp X✝.invRotate.mor₁ f.hom₁) (CategoryTh …
      -/
      dsimp
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (Neg.neg (CategoryTheory.CategoryStru …
      -/
      simp only [neg_comp, assoc, comp_neg, neg_inj, ← Functor.map_comp_assoc, ← f.comm₃]
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C (-1)) …
      -/
      rw [Functor.map_comp, assoc]
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C (-1)) …
      -/
      erw [← NatTrans.naturality]
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C (-1)) …
      -/
      rfl
      /-
        🎉 no goals
      -/
    comm₃ := by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp X✝.invRotate.mor₃ ((CategoryTheory.sh …
      -/
      erw [← reassoc_of% f.comm₂, Category.assoc, ← NatTrans.naturality]
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.HasShift C Int
        X : C
        X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp X✝.mor₂ (CategoryTheory.CategoryStruc …
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- The unit isomorphism of the auto-equivalence of categories `triangleRotation C` of
`Triangle C` given by the rotation of triangles. -/
@[simps!]
def rotCompInvRot : 𝟭 (Triangle C) ≅ rotate C ⋙ invRotate C :=
                               /-
                                 C : Type u
                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                 inst✝² : CategoryTheory.Preadditive C
                                 inst✝¹ : CategoryTheory.HasShift C Int
                                 X : C
                                 inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                 T : CategoryTheory.Pretriangulated.Triangle C
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
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
  NatIso.ofComponents fun T => Triangle.isoMk _ _
  /-
    🎉 no goals
  -/
    ((shiftEquiv C (1 : ℤ)).unitIso.app T.obj₁) (Iso.refl _) (Iso.refl _)


/-- The counit isomorphism of the auto-equivalence of categories `triangleRotation C` of
`Triangle C` given by the rotation of triangles. -/
@[simps!]
def invRotCompRot : invRotate C ⋙ rotate C ≅ 𝟭 (Triangle C) :=
                               /-
                                 C : Type u
                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                 inst✝² : CategoryTheory.Preadditive C
                                 inst✝¹ : CategoryTheory.HasShift C Int
                                 X : C
                                 inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                 T : CategoryTheory.Pretriangulated.Triangle C
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.inv …
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
  NatIso.ofComponents fun T => Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _)
  /-
    🎉 no goals
  -/
    ((shiftEquiv C (1 : ℤ)).counitIso.app T.obj₃)


/-- Rotating triangles gives an auto-equivalence on the category of triangles in `C`.
-/
@[simps]
def triangleRotation : Equivalence (Triangle C) (Triangle C) where
  functor := rotate C
  inverse := invRotate C
  unitIso := rotCompInvRot
  counitIso := invRotCompRot


instance : (rotate C).IsEquivalence := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    X : C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    ⊢ (CategoryTheory.Pretriangulated.rotate C).IsEquivalence
  -/
  change (triangleRotation C).functor.IsEquivalence
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    X : C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    ⊢ (CategoryTheory.Pretriangulated.triangleRotation C).functor.IsEquivalence
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : (invRotate C).IsEquivalence := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    X : C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    ⊢ (CategoryTheory.Pretriangulated.invRotate C).IsEquivalence
  -/
  change (triangleRotation C).inverse.IsEquivalence
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    X : C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    ⊢ (CategoryTheory.Pretriangulated.triangleRotation C).inverse.IsEquivalence
  -/
  infer_instance
  /-
    🎉 no goals
  -/


