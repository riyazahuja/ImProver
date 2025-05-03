/-- `HasLiftingProperty i p` means that `i` has the left lifting
property with respect to `p`, or equivalently that `p` has
the right lifting property with respect to `i`. -/
class HasLiftingProperty : Prop where
  /-- Unique field expressing that any commutative square built from `f` and `g` has a lift -/
  sq_hasLift : ∀ {f : A ⟶ X} {g : B ⟶ Y} (sq : CommSq f i p g), sq.HasLift


instance (priority := 100) sq_hasLift_of_hasLiftingProperty {f : A ⟶ X} {g : B ⟶ Y}
    (sq : CommSq f i p g) [hip : HasLiftingProperty i p] : sq.HasLift := hip.sq_hasLift _


theorem op (h : HasLiftingProperty i p) : HasLiftingProperty p.op i.op :=
  ⟨fun {f} {g} sq => by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : C
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      h : CategoryTheory.HasLiftingProperty i p
      f : Quiver.Hom { unop := Y } { unop := B }
      g : Quiver.Hom { unop := X } { unop := A }
      sq : CategoryTheory.CommSq f p.op i.op g
      ⊢ sq.HasLift
    -/
    simp only [CommSq.HasLift.iff_unop, Quiver.Hom.unop_op]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : C
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      h : CategoryTheory.HasLiftingProperty i p
      f : Quiver.Hom { unop := Y } { unop := B }
      g : Quiver.Hom { unop := X } { unop := A }
      sq : CategoryTheory.CommSq f p.op i.op g
      ⊢ ⋯.HasLift
    -/
    infer_instance⟩
    /-
      🎉 no goals
    -/


theorem unop {A B X Y : Cᵒᵖ} {i : A ⟶ B} {p : X ⟶ Y} (h : HasLiftingProperty i p) :
    HasLiftingProperty p.unop i.unop :=
  ⟨fun {f} {g} sq => by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : Opposite C
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      h : CategoryTheory.HasLiftingProperty i p
      f : Quiver.Hom (Opposite.unop Y) (Opposite.unop B)
      g : Quiver.Hom (Opposite.unop X) (Opposite.unop A)
      sq : CategoryTheory.CommSq f p.unop i.unop g
      ⊢ sq.HasLift
    -/
    rw [CommSq.HasLift.iff_op]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : Opposite C
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      h : CategoryTheory.HasLiftingProperty i p
      f : Quiver.Hom (Opposite.unop Y) (Opposite.unop B)
      g : Quiver.Hom (Opposite.unop X) (Opposite.unop A)
      sq : CategoryTheory.CommSq f p.unop i.unop g
      ⊢ ⋯.HasLift
    -/
    simp only [Quiver.Hom.op_unop]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      A B X Y : Opposite C
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      h : CategoryTheory.HasLiftingProperty i p
      f : Quiver.Hom (Opposite.unop Y) (Opposite.unop B)
      g : Quiver.Hom (Opposite.unop X) (Opposite.unop A)
      sq : CategoryTheory.CommSq f p.unop i.unop g
      ⊢ ⋯.HasLift
    -/
    infer_instance⟩
    /-
      🎉 no goals
    -/


theorem iff_op : HasLiftingProperty i p ↔ HasLiftingProperty p.op i.op :=
  ⟨op, unop⟩


theorem iff_unop {A B X Y : Cᵒᵖ} (i : A ⟶ B) (p : X ⟶ Y) :
    HasLiftingProperty i p ↔ HasLiftingProperty p.unop i.unop :=
  ⟨unop, op⟩


instance (priority := 100) of_left_iso [IsIso i] : HasLiftingProperty i p :=
  ⟨fun {f} {g} sq =>
    CommSq.HasLift.mk'
      { l := inv i ≫ f
                       /-
                         C : Type u_1
                         inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                         A B B' X Y Y' : C
                         i : Quiver.Hom A B
                         i' : Quiver.Hom B B'
                         p : Quiver.Hom X Y
                         p' : Quiver.Hom Y Y'
                         inst✝ : CategoryTheory.IsIso i
                         f : Quiver.Hom A X
                         g : Quiver.Hom B Y
                         sq : CategoryTheory.CommSq f i p g
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp i (CategoryTheory.CategoryStruct.comp …
                       -/
        fac_left := by simp only [IsIso.hom_inv_id_assoc]
                       /-
                         🎉 no goals
                       -/
                        /-
                          C : Type u_1
                          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                          A B B' X Y Y' : C
                          i : Quiver.Hom A B
                          i' : Quiver.Hom B B'
                          p : Quiver.Hom X Y
                          p' : Quiver.Hom Y Y'
                          inst✝ : CategoryTheory.IsIso i
                          f : Quiver.Hom A X
                          g : Quiver.Hom B Y
                          sq : CategoryTheory.CommSq f i p g
                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                        -/
        fac_right := by simp only [sq.w, assoc, IsIso.inv_hom_id_assoc] }⟩
                        /-
                          🎉 no goals
                        -/


instance (priority := 100) of_right_iso [IsIso p] : HasLiftingProperty i p :=
  ⟨fun {f} {g} sq =>
    CommSq.HasLift.mk'
      { l := g ≫ inv p
                       /-
                         C : Type u_1
                         inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                         A B B' X Y Y' : C
                         i : Quiver.Hom A B
                         i' : Quiver.Hom B B'
                         p : Quiver.Hom X Y
                         p' : Quiver.Hom Y Y'
                         inst✝ : CategoryTheory.IsIso p
                         f : Quiver.Hom A X
                         g : Quiver.Hom B Y
                         sq : CategoryTheory.CommSq f i p g
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp i (CategoryTheory.CategoryStruct.comp …
                       -/
        fac_left := by simp only [← sq.w_assoc, IsIso.hom_inv_id, comp_id]
                       /-
                         🎉 no goals
                       -/
                        /-
                          C : Type u_1
                          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                          A B B' X Y Y' : C
                          i : Quiver.Hom A B
                          i' : Quiver.Hom B B'
                          p : Quiver.Hom X Y
                          p' : Quiver.Hom Y Y'
                          inst✝ : CategoryTheory.IsIso p
                          f : Quiver.Hom A X
                          g : Quiver.Hom B Y
                          sq : CategoryTheory.CommSq f i p g
                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
                        -/
        fac_right := by simp only [assoc, IsIso.inv_hom_id, comp_id] }⟩
                        /-
                          🎉 no goals
                        -/


instance of_comp_left [HasLiftingProperty i p] [HasLiftingProperty i' p] :
    HasLiftingProperty (i ≫ i') p :=
  ⟨fun {f} {g} sq => by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      A B B' X Y Y' : C
      i : Quiver.Hom A B
      i' : Quiver.Hom B B'
      p : Quiver.Hom X Y
      p' : Quiver.Hom Y Y'
      inst✝¹ : CategoryTheory.HasLiftingProperty i p
      inst✝ : CategoryTheory.HasLiftingProperty i' p
      f : Quiver.Hom A X
      g : Quiver.Hom B' Y
      sq : CategoryTheory.CommSq f (CategoryTheory.CategoryStruct.comp i i') p g
      ⊢ sq.HasLift
    -/
    have fac := sq.w
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      A B B' X Y Y' : C
      i : Quiver.Hom A B
      i' : Quiver.Hom B B'
      p : Quiver.Hom X Y
      p' : Quiver.Hom Y Y'
      inst✝¹ : CategoryTheory.HasLiftingProperty i p
      inst✝ : CategoryTheory.HasLiftingProperty i' p
      f : Quiver.Hom A X
      g : Quiver.Hom B' Y
      sq : CategoryTheory.CommSq f (CategoryTheory.CategoryStruct.comp i i') p g
      fac : Eq (CategoryTheory.CategoryStruct.comp f p) (CategoryTheory.CategoryStru …
      ⊢ sq.HasLift
    -/
    rw [assoc] at fac
    exact
      CommSq.HasLift.mk'
        { l := (CommSq.mk (CommSq.mk fac).fac_right).lift
          fac_left := by simp only [assoc, CommSq.fac_left]
          fac_right := by simp only [CommSq.fac_right] }⟩


instance of_comp_right [HasLiftingProperty i p] [HasLiftingProperty i p'] :
    HasLiftingProperty i (p ≫ p') :=
  ⟨fun {f} {g} sq => by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      A B B' X Y Y' : C
      i : Quiver.Hom A B
      i' : Quiver.Hom B B'
      p : Quiver.Hom X Y
      p' : Quiver.Hom Y Y'
      inst✝¹ : CategoryTheory.HasLiftingProperty i p
      inst✝ : CategoryTheory.HasLiftingProperty i p'
      f : Quiver.Hom A X
      g : Quiver.Hom B Y'
      sq : CategoryTheory.CommSq f i (CategoryTheory.CategoryStruct.comp p p') g
      ⊢ sq.HasLift
    -/
    have fac := sq.w
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      A B B' X Y Y' : C
      i : Quiver.Hom A B
      i' : Quiver.Hom B B'
      p : Quiver.Hom X Y
      p' : Quiver.Hom Y Y'
      inst✝¹ : CategoryTheory.HasLiftingProperty i p
      inst✝ : CategoryTheory.HasLiftingProperty i p'
      f : Quiver.Hom A X
      g : Quiver.Hom B Y'
      sq : CategoryTheory.CommSq f i (CategoryTheory.CategoryStruct.comp p p') g
      fac : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct. …
      ⊢ sq.HasLift
    -/
    rw [← assoc] at fac
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      A B B' X Y Y' : C
      i : Quiver.Hom A B
      i' : Quiver.Hom B B'
      p : Quiver.Hom X Y
      p' : Quiver.Hom Y Y'
      inst✝¹ : CategoryTheory.HasLiftingProperty i p
      inst✝ : CategoryTheory.HasLiftingProperty i p'
      f : Quiver.Hom A X
      g : Quiver.Hom B Y'
      sq : CategoryTheory.CommSq f i (CategoryTheory.CategoryStruct.comp p p') g
      fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      ⊢ sq.HasLift
    -/
    let _ := (CommSq.mk (CommSq.mk fac).fac_left.symm).lift
    exact
      CommSq.HasLift.mk'
        { l := (CommSq.mk (CommSq.mk fac).fac_left.symm).lift
          fac_left := by simp only [CommSq.fac_left]
          fac_right := by simp only [CommSq.fac_right_assoc, CommSq.fac_right] }⟩


theorem of_arrow_iso_left {A B A' B' X Y : C} {i : A ⟶ B} {i' : A' ⟶ B'}
    (e : Arrow.mk i ≅ Arrow.mk i') (p : X ⟶ Y) [hip : HasLiftingProperty i p] :
    HasLiftingProperty i' p := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B A' B' X Y : C
    i : Quiver.Hom A B
    i' : Quiver.Hom A' B'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk i) (CategoryTheory.Arrow.mk i')
    p : Quiver.Hom X Y
    hip : CategoryTheory.HasLiftingProperty i p
    ⊢ CategoryTheory.HasLiftingProperty i' p
  -/
  rw [Arrow.iso_w' e]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B A' B' X Y : C
    i : Quiver.Hom A B
    i' : Quiver.Hom A' B'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk i) (CategoryTheory.Arrow.mk i')
    p : Quiver.Hom X Y
    hip : CategoryTheory.HasLiftingProperty i p
    ⊢ CategoryTheory.HasLiftingProperty (CategoryTheory.CategoryStruct.comp e.inv. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem of_arrow_iso_right {A B X Y X' Y' : C} (i : A ⟶ B) {p : X ⟶ Y} {p' : X' ⟶ Y'}
    (e : Arrow.mk p ≅ Arrow.mk p') [hip : HasLiftingProperty i p] : HasLiftingProperty i p' := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y X' Y' : C
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    p' : Quiver.Hom X' Y'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk p) (CategoryTheory.Arrow.mk p')
    hip : CategoryTheory.HasLiftingProperty i p
    ⊢ CategoryTheory.HasLiftingProperty i p'
  -/
  rw [Arrow.iso_w' e]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y X' Y' : C
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    p' : Quiver.Hom X' Y'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk p) (CategoryTheory.Arrow.mk p')
    hip : CategoryTheory.HasLiftingProperty i p
    ⊢ CategoryTheory.HasLiftingProperty i (CategoryTheory.CategoryStruct.comp e.in …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem iff_of_arrow_iso_left {A B A' B' X Y : C} {i : A ⟶ B} {i' : A' ⟶ B'}
    (e : Arrow.mk i ≅ Arrow.mk i') (p : X ⟶ Y) :
    HasLiftingProperty i p ↔ HasLiftingProperty i' p := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B A' B' X Y : C
    i : Quiver.Hom A B
    i' : Quiver.Hom A' B'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk i) (CategoryTheory.Arrow.mk i')
    p : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.HasLiftingProperty i p) (CategoryTheory.HasLiftingProper …
  -/
  constructor <;> intro
  /-
    case mp
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B A' B' X Y : C
    i : Quiver.Hom A B
    i' : Quiver.Hom A' B'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk i) (CategoryTheory.Arrow.mk i')
    p : Quiver.Hom X Y
    a✝ : CategoryTheory.HasLiftingProperty i p
    ⊢ CategoryTheory.HasLiftingProperty i' p
  -/
  exacts [of_arrow_iso_left e p, of_arrow_iso_left e.symm p]
  /-
    🎉 no goals
  -/


theorem iff_of_arrow_iso_right {A B X Y X' Y' : C} (i : A ⟶ B) {p : X ⟶ Y} {p' : X' ⟶ Y'}
    (e : Arrow.mk p ≅ Arrow.mk p') : HasLiftingProperty i p ↔ HasLiftingProperty i p' := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y X' Y' : C
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    p' : Quiver.Hom X' Y'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk p) (CategoryTheory.Arrow.mk p')
    ⊢ Iff (CategoryTheory.HasLiftingProperty i p) (CategoryTheory.HasLiftingProper …
  -/
  constructor <;> intro
  /-
    case mp
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B X Y X' Y' : C
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    p' : Quiver.Hom X' Y'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk p) (CategoryTheory.Arrow.mk p')
    a✝ : CategoryTheory.HasLiftingProperty i p
    ⊢ CategoryTheory.HasLiftingProperty i p'
  -/
  exacts [of_arrow_iso_right i e, of_arrow_iso_right i e.symm]
  /-
    🎉 no goals
  -/


lemma RetractArrow.leftLiftingProperty
    {X Y Z W Z' W' : C} {g : Z ⟶ W} {g' : Z' ⟶ W'}
    (h : RetractArrow g' g) (f : X ⟶ Y) [HasLiftingProperty g f] : HasLiftingProperty g' f where
  sq_hasLift := fun {u v} sq ↦ by
    have sq' : CommSq (h.r.left ≫ u) g f (h.r.right ≫ v) := by simp only [Arrow.mk_left,
      Arrow.mk_right, Category.assoc, sq.w, Arrow.w_mk_right_assoc, Arrow.mk_hom, CommSq.mk]
    exact
      ⟨⟨{ l := h.i.right ≫ sq'.lift
          fac_left := by
            simp only [← h.i_w_assoc, sq'.fac_left, h.retract_left_assoc,
              Arrow.mk_left, Category.id_comp]}⟩⟩


lemma RetractArrow.rightLiftingProperty
    {X Y Z W X' Y' : C} {f : X ⟶ Y} {f' : X' ⟶ Y'}
    (h : RetractArrow f' f) (g : Z ⟶ W) [HasLiftingProperty g f] : HasLiftingProperty g f' where
  sq_hasLift := fun {u v} sq ↦
    have sq' : CommSq (u ≫ h.i.left) g f (v ≫ h.i.right) :=
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
            X Y Z W X' Y' : C
            f : Quiver.Hom X Y
            f' : Quiver.Hom X' Y'
            h : CategoryTheory.RetractArrow f' f
            g : Quiver.Hom Z W
            inst✝ : CategoryTheory.HasLiftingProperty g f
            u : Quiver.Hom Z X'
            v : Quiver.Hom W Y'
            sq : CategoryTheory.CommSq u g f' v
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
          -/
      ⟨by rw [← Category.assoc, ← sq.w, Category.assoc, RetractArrow.i_w, Category.assoc]⟩
          /-
            🎉 no goals
          -/
    ⟨⟨{ l := sq'.lift ≫ h.r.left}⟩⟩


