/-- The proposition that a square
```
  W ---f---> X
  |          |
  g          h
  |          |
  v          v
  Y ---i---> Z

```
is a commuting square.
-/
structure CommSq {W X Y Z : C} (f : W ⟶ X) (g : W ⟶ Y) (h : X ⟶ Z) (i : Y ⟶ Z) : Prop where
  /-- The square commutes. -/
  w : f ≫ h = g ≫ i


attribute [reassoc] CommSq.w


theorem flip (p : CommSq f g h i) : CommSq g f i h :=
  ⟨p.w.symm⟩


theorem of_arrow {f g : Arrow C} (h : f ⟶ g) : CommSq f.hom h.left h.right g.hom :=
  ⟨h.w.symm⟩


/-- The commutative square in the opposite category associated to a commutative square. -/
theorem op (p : CommSq f g h i) : CommSq i.op h.op g.op f.op :=
      /-
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        p : CategoryTheory.CommSq f g h i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp i.op g.op) (CategoryTheory.CategorySt …
      -/
  ⟨by simp only [← op_comp, p.w]⟩
      /-
        🎉 no goals
      -/


/-- The commutative square associated to a commutative square in the opposite category. -/
theorem unop {W X Y Z : Cᵒᵖ} {f : W ⟶ X} {g : W ⟶ Y} {h : X ⟶ Z} {i : Y ⟶ Z} (p : CommSq f g h i) :
    CommSq i.unop h.unop g.unop f.unop :=
      /-
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : Opposite C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        p : CategoryTheory.CommSq f g h i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp i.unop g.unop) (CategoryTheory.Catego …
      -/
  ⟨by simp only [← unop_comp, p.w]⟩
      /-
        🎉 no goals
      -/


theorem vert_inv {g : W ≅ Y} {h : X ≅ Z} (p : CommSq f g.hom h.hom i) :
    CommSq i g.inv h.inv f :=
      /-
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W X
        i : Quiver.Hom Y Z
        g : CategoryTheory.Iso W Y
        h : CategoryTheory.Iso X Z
        p : CategoryTheory.CommSq f g.hom h.hom i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp i h.inv) (CategoryTheory.CategoryStru …
      -/
  ⟨by rw [Iso.comp_inv_eq, Category.assoc, Iso.eq_inv_comp, p.w]⟩
      /-
        🎉 no goals
      -/


theorem horiz_inv {f : W ≅ X} {i : Y ≅ Z} (p : CommSq f.hom g h i.hom) :
    CommSq f.inv h g i.inv :=
  flip (vert_inv (flip p))


/-- The horizontal composition of two commutative squares as below is a commutative square.
```
  W ---f---> X ---f'--> X'
  |          |          |
  g          h          h'
  |          |          |
  v          v          v
  Y ---i---> Z ---i'--> Z'

```
-/
lemma horiz_comp {W X X' Y Z Z' : C} {f : W ⟶ X} {f' : X ⟶ X'} {g : W ⟶ Y} {h : X ⟶ Z}
    {h' : X' ⟶ Z'} {i : Y ⟶ Z} {i' : Z ⟶ Z'} (hsq₁ : CommSq f g h i) (hsq₂ : CommSq f' h h' i') :
    CommSq (f ≫ f') g h' (i ≫ i') :=
      /-
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        W X X' Y Z Z' : C
        f : Quiver.Hom W X
        f' : Quiver.Hom X X'
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        h' : Quiver.Hom X' Z'
        i : Quiver.Hom Y Z
        i' : Quiver.Hom Z Z'
        hsq₁ : CategoryTheory.CommSq f g h i
        hsq₂ : CategoryTheory.CommSq f' h h' i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
  ⟨by rw [← Category.assoc, Category.assoc, ← hsq₁.w, hsq₂.w, Category.assoc]⟩
      /-
        🎉 no goals
      -/


/-- The vertical composition of two commutative squares as below is a commutative square.
```
  W ---f---> X
  |          |
  g          h
  |          |
  v          v
  Y ---i---> Z
  |          |
  g'         h'
  |          |
  v          v
  Y'---i'--> Z'

```
-/
lemma vert_comp {W X Y Y' Z Z' : C} {f : W ⟶ X} {g : W ⟶ Y} {g' : Y ⟶ Y'} {h : X ⟶ Z}
    {h' : Z ⟶ Z'} {i : Y ⟶ Z} {i' : Y' ⟶ Z'} (hsq₁ : CommSq f g h i) (hsq₂ : CommSq i g' h' i') :
    CommSq f (g ≫ g') (h ≫ h') i' :=
  flip (horiz_comp (flip hsq₁) (flip hsq₂))



theorem eq_of_mono {f : W ⟶ X} {g : W ⟶ X} {i : X ⟶ Y} [Mono i] (sq : CommSq f g i i) : f = g :=
  (cancel_mono i).1 sq.w


theorem eq_of_epi {f : W ⟶ X} {h : X ⟶ Y} {i : X ⟶ Y} [Epi f] (sq : CommSq f f h i) : h = i :=
  (cancel_epi f).1 sq.w


theorem map_commSq (s : CommSq f g h i) : CommSq (F.map f) (F.map g) (F.map h) (F.map i) :=
      /-
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        D : Type u_2
        inst✝ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        W X Y Z : C
        f : Quiver.Hom W X
        g : Quiver.Hom W Y
        h : Quiver.Hom X Z
        i : Quiver.Hom Y Z
        s : CategoryTheory.CommSq f g h i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (F.map h)) (CategoryTheory. …
      -/
  ⟨by simpa using congr_arg (fun k : W ⟶ Z => F.map k) s.w⟩
      /-
        🎉 no goals
      -/


alias CommSq.map := Functor.map_commSq


/-- Now we consider a square:
```
  A ---f---> X
  |          |
  i          p
  |          |
  v          v
  B ---g---> Y
```

The datum of a lift in a commutative square, i.e. an up-right-diagonal
morphism which makes both triangles commute. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
@[ext]
structure LiftStruct (sq : CommSq f i p g) where
  /-- The lift. -/
  l : B ⟶ X
  /-- The upper left triangle commutes. -/
  fac_left : i ≫ l = f := by aesop_cat
  /-- The lower right triangle commutes. -/
  fac_right : l ≫ p = g := by aesop_cat


/-- A `LiftStruct` for a commutative square gives a `LiftStruct` for the
corresponding square in the opposite category. -/
@[simps]
def op {sq : CommSq f i p g} (l : LiftStruct sq) : LiftStruct sq.op where
  l := l.l.op
                 /-
                   C : Type u_1
                   inst✝ : CategoryTheory.Category.{?u.12244, u_1} C
                   A B X Y : C
                   f : Quiver.Hom A X
                   i : Quiver.Hom A B
                   p : Quiver.Hom X Y
                   g : Quiver.Hom B Y
                   sq : CategoryTheory.CommSq f i p g
                   l : sq.LiftStruct
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp p.op l.l.op) g.op
                 -/
  fac_left := by rw [← op_comp, l.fac_right]
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u_1
                    inst✝ : CategoryTheory.Category.{?u.12244, u_1} C
                    A B X Y : C
                    f : Quiver.Hom A X
                    i : Quiver.Hom A B
                    p : Quiver.Hom X Y
                    g : Quiver.Hom B Y
                    sq : CategoryTheory.CommSq f i p g
                    l : sq.LiftStruct
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp l.l.op i.op) f.op
                  -/
  fac_right := by rw [← op_comp, l.fac_left]
                  /-
                    🎉 no goals
                  -/


/-- A `LiftStruct` for a commutative square in the opposite category
gives a `LiftStruct` for the corresponding square in the original category. -/
@[simps]
def unop {A B X Y : Cᵒᵖ} {f : A ⟶ X} {i : A ⟶ B} {p : X ⟶ Y} {g : B ⟶ Y} {sq : CommSq f i p g}
    (l : LiftStruct sq) : LiftStruct sq.unop where
  l := l.l.unop
                 /-
                   C : Type u_1
                   inst✝ : CategoryTheory.Category.{?u.13162, u_1} C
                   A✝ B✝ X✝ Y✝ : C
                   f✝ : Quiver.Hom A✝ X✝
                   i✝ : Quiver.Hom A✝ B✝
                   p✝ : Quiver.Hom X✝ Y✝
                   g✝ : Quiver.Hom B✝ Y✝
                   A B X Y : Opposite C
                   f : Quiver.Hom A X
                   i : Quiver.Hom A B
                   p : Quiver.Hom X Y
                   g : Quiver.Hom B Y
                   sq : CategoryTheory.CommSq f i p g
                   l : sq.LiftStruct
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp p.unop l.l.unop) g.unop
                 -/
  fac_left := by rw [← unop_comp, l.fac_right]
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u_1
                    inst✝ : CategoryTheory.Category.{?u.13162, u_1} C
                    A✝ B✝ X✝ Y✝ : C
                    f✝ : Quiver.Hom A✝ X✝
                    i✝ : Quiver.Hom A✝ B✝
                    p✝ : Quiver.Hom X✝ Y✝
                    g✝ : Quiver.Hom B✝ Y✝
                    A B X Y : Opposite C
                    f : Quiver.Hom A X
                    i : Quiver.Hom A B
                    p : Quiver.Hom X Y
                    g : Quiver.Hom B Y
                    sq : CategoryTheory.CommSq f i p g
                    l : sq.LiftStruct
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp l.l.unop i.unop) f.unop
                  -/
  fac_right := by rw [← unop_comp, l.fac_left]
                  /-
                    🎉 no goals
                  -/


/-- Equivalences of `LiftStruct` for a square and the corresponding square
in the opposite category. -/
@[simps]
def opEquiv (sq : CommSq f i p g) : LiftStruct sq ≃ LiftStruct sq.op where
  toFun := op
  invFun := unop
                 /-
                   C : Type u_1
                   inst✝ : CategoryTheory.Category.{?u.14239, u_1} C
                   A B X Y : C
                   f : Quiver.Hom A X
                   i : Quiver.Hom A B
                   p : Quiver.Hom X Y
                   g : Quiver.Hom B Y
                   sq : CategoryTheory.CommSq f i p g
                   ⊢ Function.LeftInverse CategoryTheory.CommSq.LiftStruct.unop CategoryTheory.Co …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u_1
                    inst✝ : CategoryTheory.Category.{?u.14239, u_1} C
                    A B X Y : C
                    f : Quiver.Hom A X
                    i : Quiver.Hom A B
                    p : Quiver.Hom X Y
                    g : Quiver.Hom B Y
                    sq : CategoryTheory.CommSq f i p g
                    ⊢ Function.RightInverse CategoryTheory.CommSq.LiftStruct.unop CategoryTheory.C …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- Equivalences of `LiftStruct` for a square in the oppositive category and
the corresponding square in the original category. -/
def unopEquiv {A B X Y : Cᵒᵖ} {f : A ⟶ X} {i : A ⟶ B} {p : X ⟶ Y} {g : B ⟶ Y}
    (sq : CommSq f i p g) : LiftStruct sq ≃ LiftStruct sq.unop where
  toFun := unop
  invFun := op
                 /-
                   C : Type u_1
                   inst✝ : CategoryTheory.Category.{?u.15448, u_1} C
                   A✝ B✝ X✝ Y✝ : C
                   f✝ : Quiver.Hom A✝ X✝
                   i✝ : Quiver.Hom A✝ B✝
                   p✝ : Quiver.Hom X✝ Y✝
                   g✝ : Quiver.Hom B✝ Y✝
                   A B X Y : Opposite C
                   f : Quiver.Hom A X
                   i : Quiver.Hom A B
                   p : Quiver.Hom X Y
                   g : Quiver.Hom B Y
                   sq : CategoryTheory.CommSq f i p g
                   ⊢ Function.LeftInverse CategoryTheory.CommSq.LiftStruct.op CategoryTheory.Comm …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u_1
                    inst✝ : CategoryTheory.Category.{?u.15448, u_1} C
                    A✝ B✝ X✝ Y✝ : C
                    f✝ : Quiver.Hom A✝ X✝
                    i✝ : Quiver.Hom A✝ B✝
                    p✝ : Quiver.Hom X✝ Y✝
                    g✝ : Quiver.Hom B✝ Y✝
                    A B X Y : Opposite C
                    f : Quiver.Hom A X
                    i : Quiver.Hom A B
                    p : Quiver.Hom X Y
                    g : Quiver.Hom B Y
                    sq : CategoryTheory.CommSq f i p g
                    ⊢ Function.RightInverse CategoryTheory.CommSq.LiftStruct.op CategoryTheory.Com …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


instance subsingleton_liftStruct_of_epi (sq : CommSq f i p g) [Epi i] :
    Subsingleton (LiftStruct sq) :=
  ⟨fun l₁ l₂ => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : C
      f : Quiver.Hom A X
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      g : Quiver.Hom B Y
      sq : CategoryTheory.CommSq f i p g
      inst✝ : CategoryTheory.Epi i
      l₁ l₂ : sq.LiftStruct
      ⊢ Eq l₁ l₂
    -/
    ext
    /-
      case l
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : C
      f : Quiver.Hom A X
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      g : Quiver.Hom B Y
      sq : CategoryTheory.CommSq f i p g
      inst✝ : CategoryTheory.Epi i
      l₁ l₂ : sq.LiftStruct
      ⊢ Eq l₁.l l₂.l
    -/
    rw [← cancel_epi i]
    /-
      case l
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : C
      f : Quiver.Hom A X
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      g : Quiver.Hom B Y
      sq : CategoryTheory.CommSq f i p g
      inst✝ : CategoryTheory.Epi i
      l₁ l₂ : sq.LiftStruct
      ⊢ Eq (CategoryTheory.CategoryStruct.comp i l₁.l) (CategoryTheory.CategoryStruc …
    -/
    simp only [LiftStruct.fac_left]⟩
    /-
      🎉 no goals
    -/


instance subsingleton_liftStruct_of_mono (sq : CommSq f i p g) [Mono p] :
    Subsingleton (LiftStruct sq) :=
  ⟨fun l₁ l₂ => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : C
      f : Quiver.Hom A X
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      g : Quiver.Hom B Y
      sq : CategoryTheory.CommSq f i p g
      inst✝ : CategoryTheory.Mono p
      l₁ l₂ : sq.LiftStruct
      ⊢ Eq l₁ l₂
    -/
    ext
    /-
      case l
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : C
      f : Quiver.Hom A X
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      g : Quiver.Hom B Y
      sq : CategoryTheory.CommSq f i p g
      inst✝ : CategoryTheory.Mono p
      l₁ l₂ : sq.LiftStruct
      ⊢ Eq l₁.l l₂.l
    -/
    rw [← cancel_mono p]
    /-
      case l
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : C
      f : Quiver.Hom A X
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      g : Quiver.Hom B Y
      sq : CategoryTheory.CommSq f i p g
      inst✝ : CategoryTheory.Mono p
      l₁ l₂ : sq.LiftStruct
      ⊢ Eq (CategoryTheory.CategoryStruct.comp l₁.l p) (CategoryTheory.CategoryStruc …
    -/
    simp only [LiftStruct.fac_right]⟩
    /-
      🎉 no goals
    -/


/-- The assertion that a square has a `LiftStruct`. -/
class HasLift : Prop where
  /-- Square has a `LiftStruct`. -/
  exists_lift : Nonempty sq.LiftStruct


theorem mk' (l : sq.LiftStruct) : HasLift sq :=
  ⟨Nonempty.intro l⟩


theorem iff : HasLift sq ↔ Nonempty sq.LiftStruct := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y : C
    f : Quiver.Hom A X
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    g : Quiver.Hom B Y
    sq : CategoryTheory.CommSq f i p g
    ⊢ Iff sq.HasLift (Nonempty sq.LiftStruct)
  -/
  constructor
  /-
    case mp
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y : C
    f : Quiver.Hom A X
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    g : Quiver.Hom B Y
    sq : CategoryTheory.CommSq f i p g
    ⊢ sq.HasLift → Nonempty sq.LiftStruct
  -/
  exacts [fun h => h.exists_lift, fun h => mk h]
  /-
    🎉 no goals
  -/


theorem iff_op : HasLift sq ↔ HasLift sq.op := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y : C
    f : Quiver.Hom A X
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    g : Quiver.Hom B Y
    sq : CategoryTheory.CommSq f i p g
    ⊢ Iff sq.HasLift ⋯.HasLift
  -/
  rw [iff, iff]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y : C
    f : Quiver.Hom A X
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    g : Quiver.Hom B Y
    sq : CategoryTheory.CommSq f i p g
    ⊢ Iff (Nonempty sq.LiftStruct) (Nonempty ⋯.LiftStruct)
  -/
  exact Nonempty.congr (LiftStruct.opEquiv sq).toFun (LiftStruct.opEquiv sq).invFun
  /-
    🎉 no goals
  -/


theorem iff_unop {A B X Y : Cᵒᵖ} {f : A ⟶ X} {i : A ⟶ B} {p : X ⟶ Y} {g : B ⟶ Y}
    (sq : CommSq f i p g) : HasLift sq ↔ HasLift sq.unop := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y : Opposite C
    f : Quiver.Hom A X
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    g : Quiver.Hom B Y
    sq : CategoryTheory.CommSq f i p g
    ⊢ Iff sq.HasLift ⋯.HasLift
  -/
  rw [iff, iff]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y : Opposite C
    f : Quiver.Hom A X
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    g : Quiver.Hom B Y
    sq : CategoryTheory.CommSq f i p g
    ⊢ Iff (Nonempty sq.LiftStruct) (Nonempty ⋯.LiftStruct)
  -/
  exact Nonempty.congr (LiftStruct.unopEquiv sq).toFun (LiftStruct.unopEquiv sq).invFun
  /-
    🎉 no goals
  -/


/-- A choice of a diagonal morphism that is part of a `LiftStruct` when
the square has a lift. -/
noncomputable def lift [hsq : HasLift sq] : B ⟶ X :=
  hsq.exists_lift.some.l


@[reassoc (attr := simp)]
theorem fac_left [hsq : HasLift sq] : i ≫ sq.lift = f :=
  hsq.exists_lift.some.fac_left


@[reassoc (attr := simp)]
theorem fac_right [hsq : HasLift sq] : sq.lift ≫ p = g :=
  hsq.exists_lift.some.fac_right


