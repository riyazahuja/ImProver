/-- An object `X` is a retract of `Y` if there are morphisms `i : X ⟶ Y` and `r : Y ⟶ X` such
that `i ≫ r = 𝟙 X`. -/
structure Retract (X Y : C) where
  /-- the split monomorphism -/
  i : X ⟶ Y
  /-- the split epimorphism -/
  r : Y ⟶ X
  retract : i ≫ r = 𝟙 X := by aesop_cat


attribute [reassoc (attr := simp)] retract


/-- If `X` is a retract of `Y`, then `F.obj X` is a retract of `F.obj Y`. -/
@[simps]
def map (F : C ⥤ D) : Retract (F.obj X) (F.obj Y) where
  i := F.map h.i
  r := F.map h.r
                /-
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  D : Type u'
                  inst✝ : CategoryTheory.Category.{v', u'} D
                  X Y : C
                  h : CategoryTheory.Retract X Y
                  F : CategoryTheory.Functor C D
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map h.i) (F.map h.r)) (CategoryThe …
                -/
  retract := by rw [← F.map_comp h.i h.r, h.retract, F.map_id]
                /-
                  🎉 no goals
                -/


/-- a retract determines a split epimorphism. -/
@[simps] def splitEpi : SplitEpi h.r where
  section_ := h.i


/-- a retract determines a split monomorphism. -/
@[simps] def splitMono : SplitMono h.i where
  retraction := h.r


instance : IsSplitEpi h.r := ⟨⟨h.splitEpi⟩⟩


instance : IsSplitMono h.i := ⟨⟨h.splitMono⟩⟩


/--
```
  X -------> Z -------> X
  |          |          |
  f          g          f
  |          |          |
  v          v          v
  Y -------> W -------> Y

```
A morphism `f : X ⟶ Y` is a retract of `g : Z ⟶ W` if there are morphisms `i : f ⟶ g`
and `r : g ⟶ f` in the arrow category such that `i ≫ r = 𝟙 f`. -/
abbrev RetractArrow {X Y Z W : C} (f : X ⟶ Y) (g : Z ⟶ W) := Retract (Arrow.mk f) (Arrow.mk g)


@[reassoc]
lemma i_w : h.i.left ≫ g = f ≫ h.i.right := h.i.w


@[reassoc]
lemma r_w : h.r.left ≫ f = g ≫ h.r.right := h.r.w


/-- The top of a retract diagram of morphisms determines a retract of objects. -/
@[simps!]
def left : Retract X Z := h.map Arrow.leftFunc


/-- The bottom of a retract diagram of morphisms determines a retract of objects. -/
@[simps!]
def right : Retract Y W := h.map Arrow.rightFunc


@[reassoc (attr := simp)]
lemma retract_left : h.i.left ≫ h.r.left = 𝟙 X := h.left.retract


@[reassoc (attr := simp)]
lemma retract_right : h.i.right ≫ h.r.right = 𝟙 Y := h.right.retract


instance : IsSplitEpi h.r.left := ⟨⟨h.left.splitEpi⟩⟩


instance : IsSplitEpi h.r.right := ⟨⟨h.right.splitEpi⟩⟩


instance : IsSplitMono h.i.left := ⟨⟨h.left.splitMono⟩⟩


instance : IsSplitMono h.i.right := ⟨⟨h.right.splitMono⟩⟩


