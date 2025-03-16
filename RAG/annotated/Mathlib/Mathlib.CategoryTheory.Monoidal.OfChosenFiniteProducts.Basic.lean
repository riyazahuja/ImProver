/-- Swap the two sides of a `BinaryFan`. -/
def BinaryFan.swap {P Q : C} (t : BinaryFan P Q) : BinaryFan Q P :=
  BinaryFan.mk t.snd t.fst


@[simp]
theorem BinaryFan.swap_fst {P Q : C} (t : BinaryFan P Q) : t.swap.fst = t.snd :=
  rfl


@[simp]
theorem BinaryFan.swap_snd {P Q : C} (t : BinaryFan P Q) : t.swap.snd = t.fst :=
  rfl


/-- If a binary fan `t` over `P Q` is a limit cone, then `t.swap` is a limit cone over `Q P`.
-/
@[simps]
def IsLimit.swapBinaryFan {P Q : C} {t : BinaryFan P Q} (I : IsLimit t) : IsLimit t.swap where
  lift s := I.lift (BinaryFan.swap s)
              /-
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                X Y P Q : C
                t : CategoryTheory.Limits.BinaryFan P Q
                I : CategoryTheory.Limits.IsLimit t
                s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair Q P)
                ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
              -/
                              /-
                                🎉 no goals
                              -/
  fac s := by rintro ⟨⟨⟩⟩ <;> simp
                              /-
                                🎉 no goals
                              -/
  uniq s m w := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y P Q : C
      t : CategoryTheory.Limits.BinaryFan P Q
      I : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair Q P)
      m : Quiver.Hom s.pt t.swap.pt
      w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      ⊢ Eq m ((fun s => I.lift (CategoryTheory.Limits.BinaryFan.swap s)) s)
    -/
    have h := I.uniq (BinaryFan.swap s) m
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y P Q : C
      t : CategoryTheory.Limits.BinaryFan P Q
      I : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair Q P)
      m : Quiver.Hom s.pt t.swap.pt
      w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ Eq m ((fun s => I.lift (CategoryTheory.Limits.BinaryFan.swap s)) s)
    -/
    rw [h]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y P Q : C
      t : CategoryTheory.Limits.BinaryFan P Q
      I : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair Q P)
      m : Quiver.Hom s.pt t.swap.pt
      w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨j⟩
    /-
      case mk
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y P Q : C
      t : CategoryTheory.Limits.BinaryFan P Q
      I : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair Q P)
      m : Quiver.Hom s.pt t.swap.pt
      w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      j : CategoryTheory.Limits.WalkingPair
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (t.π.app { as := j })) ((CategoryTh …
    -/
    specialize w ⟨WalkingPair.swap j⟩
    /-
      case mk
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y P Q : C
      t : CategoryTheory.Limits.BinaryFan P Q
      I : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair Q P)
      m : Quiver.Hom s.pt t.swap.pt
      h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      j : CategoryTheory.Limits.WalkingPair
      w : Eq (CategoryTheory.CategoryStruct.comp m (t.swap.π.app { as := CategoryThe …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (t.π.app { as := j })) ((CategoryTh …
    -/
                /-
                  🎉 no goals
                -/
    cases j <;> exact w
                /-
                  🎉 no goals
                -/


/-- Construct `HasBinaryProduct Q P` from `HasBinaryProduct P Q`.
This can't be an instance, as it would cause a loop in typeclass search.
-/
theorem HasBinaryProduct.swap (P Q : C) [HasBinaryProduct P Q] : HasBinaryProduct Q P :=
  HasLimit.mk ⟨BinaryFan.swap (limit.cone (pair P Q)), (limit.isLimit (pair P Q)).swapBinaryFan⟩


/-- Given a limit cone over `X` and `Y`, and another limit cone over `Y` and `X`, we can construct
an isomorphism between the cone points. Relative to some fixed choice of limits cones for every
pair, these isomorphisms constitute a braiding.
-/
def BinaryFan.braiding {X Y : C} {s : BinaryFan X Y} (P : IsLimit s) {t : BinaryFan Y X}
    (Q : IsLimit t) : s.pt ≅ t.pt :=
  IsLimit.conePointUniqueUpToIso P Q.swapBinaryFan


/-- Given binary fans `sXY` over `X Y`, and `sYZ` over `Y Z`, and `s` over `sXY.X Z`,
if `sYZ` is a limit cone we can construct a binary fan over `X sYZ.X`.

This is an ingredient of building the associator for a cartesian category.
-/
def BinaryFan.assoc {X Y Z : C} {sXY : BinaryFan X Y} {sYZ : BinaryFan Y Z} (Q : IsLimit sYZ)
    (s : BinaryFan sXY.pt Z) : BinaryFan X sYZ.pt :=
  BinaryFan.mk (s.fst ≫ sXY.fst) (Q.lift (BinaryFan.mk (s.fst ≫ sXY.snd) s.snd))


@[simp]
theorem BinaryFan.assoc_fst {X Y Z : C} {sXY : BinaryFan X Y} {sYZ : BinaryFan Y Z}
    (Q : IsLimit sYZ) (s : BinaryFan sXY.pt Z) : (BinaryFan.assoc Q s).fst = s.fst ≫ sXY.fst :=
  rfl


@[simp]
theorem BinaryFan.assoc_snd {X Y Z : C} {sXY : BinaryFan X Y} {sYZ : BinaryFan Y Z}
    (Q : IsLimit sYZ) (s : BinaryFan sXY.pt Z) :
    (BinaryFan.assoc Q s).snd = Q.lift (BinaryFan.mk (s.fst ≫ sXY.snd) s.snd) :=
  rfl


/-- Given binary fans `sXY` over `X Y`, and `sYZ` over `Y Z`, and `s` over `X sYZ.X`,
if `sYZ` is a limit cone we can construct a binary fan over `sXY.X Z`.

This is an ingredient of building the associator for a cartesian category.
-/
def BinaryFan.assocInv {X Y Z : C} {sXY : BinaryFan X Y} (P : IsLimit sXY) {sYZ : BinaryFan Y Z}
    (s : BinaryFan X sYZ.pt) : BinaryFan sXY.pt Z :=
  BinaryFan.mk (P.lift (BinaryFan.mk s.fst (s.snd ≫ sYZ.fst))) (s.snd ≫ sYZ.snd)


@[simp]
theorem BinaryFan.assocInv_fst {X Y Z : C} {sXY : BinaryFan X Y} (P : IsLimit sXY)
    {sYZ : BinaryFan Y Z} (s : BinaryFan X sYZ.pt) :
    (BinaryFan.assocInv P s).fst = P.lift (BinaryFan.mk s.fst (s.snd ≫ sYZ.fst)) :=
  rfl


@[simp]
theorem BinaryFan.assocInv_snd {X Y Z : C} {sXY : BinaryFan X Y} (P : IsLimit sXY)
    {sYZ : BinaryFan Y Z} (s : BinaryFan X sYZ.pt) :
    (BinaryFan.assocInv P s).snd = s.snd ≫ sYZ.snd :=
  rfl


/-- If all the binary fans involved a limit cones, `BinaryFan.assoc` produces another limit cone.
-/
@[simps]
def IsLimit.assoc {X Y Z : C} {sXY : BinaryFan X Y} (P : IsLimit sXY) {sYZ : BinaryFan Y Z}
    (Q : IsLimit sYZ) {s : BinaryFan sXY.pt Z} (R : IsLimit s) : IsLimit (BinaryFan.assoc Q s) where
  lift t := R.lift (BinaryFan.assocInv P t)
  fac t := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y Z : C
      sXY : CategoryTheory.Limits.BinaryFan X Y
      P : CategoryTheory.Limits.IsLimit sXY
      sYZ : CategoryTheory.Limits.BinaryFan Y Z
      Q : CategoryTheory.Limits.IsLimit sYZ
      s : CategoryTheory.Limits.BinaryFan sXY.pt Z
      R : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
                    /-
                      🎉 no goals
                    -/
    rintro ⟨⟨⟩⟩ <;> simp
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y Z : C
      sXY : CategoryTheory.Limits.BinaryFan X Y
      P : CategoryTheory.Limits.IsLimit sXY
      sYZ : CategoryTheory.Limits.BinaryFan Y Z
      Q : CategoryTheory.Limits.IsLimit sYZ
      s : CategoryTheory.Limits.BinaryFan sXY.pt Z
      R : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.lift (CategoryTheory.Limits.Binary …
    -/
    apply Q.hom_ext
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y Z : C
      sXY : CategoryTheory.Limits.BinaryFan X Y
      P : CategoryTheory.Limits.IsLimit sXY
      sYZ : CategoryTheory.Limits.BinaryFan Y Z
      Q : CategoryTheory.Limits.IsLimit sYZ
      s : CategoryTheory.Limits.BinaryFan sXY.pt Z
      R : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
                    /-
                      🎉 no goals
                    -/
    rintro ⟨⟨⟩⟩ <;> simp
                    /-
                      🎉 no goals
                    -/
  uniq t m w := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y Z : C
      sXY : CategoryTheory.Limits.BinaryFan X Y
      P : CategoryTheory.Limits.IsLimit sXY
      sYZ : CategoryTheory.Limits.BinaryFan Y Z
      Q : CategoryTheory.Limits.IsLimit sYZ
      s : CategoryTheory.Limits.BinaryFan sXY.pt Z
      R : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
      m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
      w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      ⊢ Eq m ((fun t => R.lift (CategoryTheory.Limits.BinaryFan.assocInv P t)) t)
    -/
    have h := R.uniq (BinaryFan.assocInv P t) m
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y Z : C
      sXY : CategoryTheory.Limits.BinaryFan X Y
      P : CategoryTheory.Limits.IsLimit sXY
      sYZ : CategoryTheory.Limits.BinaryFan Y Z
      Q : CategoryTheory.Limits.IsLimit sYZ
      s : CategoryTheory.Limits.BinaryFan sXY.pt Z
      R : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
      m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
      w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ Eq m ((fun t => R.lift (CategoryTheory.Limits.BinaryFan.assocInv P t)) t)
    -/
    rw [h]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y Z : C
      sXY : CategoryTheory.Limits.BinaryFan X Y
      P : CategoryTheory.Limits.IsLimit sXY
      sYZ : CategoryTheory.Limits.BinaryFan Y Z
      Q : CategoryTheory.Limits.IsLimit sYZ
      s : CategoryTheory.Limits.BinaryFan sXY.pt Z
      R : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
      m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
      w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨⟨⟩⟩ <;> simp
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y Z : C
        sXY : CategoryTheory.Limits.BinaryFan X Y
        P : CategoryTheory.Limits.IsLimit sXY
        sYZ : CategoryTheory.Limits.BinaryFan Y Z
        Q : CategoryTheory.Limits.IsLimit sYZ
        s : CategoryTheory.Limits.BinaryFan sXY.pt Z
        R : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
        m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
        w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
        h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m s.fst) (P.lift (CategoryTheory.Limi …
      -/
    · apply P.hom_ext
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y Z : C
        sXY : CategoryTheory.Limits.BinaryFan X Y
        P : CategoryTheory.Limits.IsLimit sXY
        sYZ : CategoryTheory.Limits.BinaryFan Y Z
        Q : CategoryTheory.Limits.IsLimit sYZ
        s : CategoryTheory.Limits.BinaryFan sXY.pt Z
        R : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
        m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
        w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
        h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
      -/
      rintro ⟨⟨⟩⟩ <;> simp
        /-
          case mk.left.mk.left
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ X Y Z : C
          sXY : CategoryTheory.Limits.BinaryFan X Y
          P : CategoryTheory.Limits.IsLimit sXY
          sYZ : CategoryTheory.Limits.BinaryFan Y Z
          Q : CategoryTheory.Limits.IsLimit sYZ
          s : CategoryTheory.Limits.BinaryFan sXY.pt Z
          R : CategoryTheory.Limits.IsLimit s
          t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
          m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
          w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
          h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
        -/
      · exact w ⟨WalkingPair.left⟩
        /-
          🎉 no goals
        -/
        /-
          case mk.left.mk.right
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ X Y Z : C
          sXY : CategoryTheory.Limits.BinaryFan X Y
          P : CategoryTheory.Limits.IsLimit sXY
          sYZ : CategoryTheory.Limits.BinaryFan Y Z
          Q : CategoryTheory.Limits.IsLimit sYZ
          s : CategoryTheory.Limits.BinaryFan sXY.pt Z
          R : CategoryTheory.Limits.IsLimit s
          t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
          m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
          w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
          h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
        -/
      · specialize w ⟨WalkingPair.right⟩
        simp? at w says
          simp only [pair_obj_right, BinaryFan.π_app_right, BinaryFan.assoc_snd,
            Functor.const_obj_obj, pair_obj_left] at w
        /-
          case mk.left.mk.right
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ X Y Z : C
          sXY : CategoryTheory.Limits.BinaryFan X Y
          P : CategoryTheory.Limits.IsLimit sXY
          sYZ : CategoryTheory.Limits.BinaryFan Y Z
          Q : CategoryTheory.Limits.IsLimit sYZ
          s : CategoryTheory.Limits.BinaryFan sXY.pt Z
          R : CategoryTheory.Limits.IsLimit s
          t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
          m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
          h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
          w : Eq (CategoryTheory.CategoryStruct.comp m (Q.lift (CategoryTheory.Limits.Bi …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
        -/
        rw [← w]
        /-
          case mk.left.mk.right
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ X Y Z : C
          sXY : CategoryTheory.Limits.BinaryFan X Y
          P : CategoryTheory.Limits.IsLimit sXY
          sYZ : CategoryTheory.Limits.BinaryFan Y Z
          Q : CategoryTheory.Limits.IsLimit sYZ
          s : CategoryTheory.Limits.BinaryFan sXY.pt Z
          R : CategoryTheory.Limits.IsLimit s
          t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
          m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
          h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
          w : Eq (CategoryTheory.CategoryStruct.comp m (Q.lift (CategoryTheory.Limits.Bi …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
        -/
        simp
        /-
          🎉 no goals
        -/
      /-
        case mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y Z : C
        sXY : CategoryTheory.Limits.BinaryFan X Y
        P : CategoryTheory.Limits.IsLimit sXY
        sYZ : CategoryTheory.Limits.BinaryFan Y Z
        Q : CategoryTheory.Limits.IsLimit sYZ
        s : CategoryTheory.Limits.BinaryFan sXY.pt Z
        R : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
        m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
        w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
        h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m s.snd) (CategoryTheory.CategoryStru …
      -/
    · specialize w ⟨WalkingPair.right⟩
      simp? at w says
        simp only [pair_obj_right, BinaryFan.π_app_right, BinaryFan.assoc_snd,
          Functor.const_obj_obj, pair_obj_left] at w
      /-
        case mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y Z : C
        sXY : CategoryTheory.Limits.BinaryFan X Y
        P : CategoryTheory.Limits.IsLimit sXY
        sYZ : CategoryTheory.Limits.BinaryFan Y Z
        Q : CategoryTheory.Limits.IsLimit sYZ
        s : CategoryTheory.Limits.BinaryFan sXY.pt Z
        R : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
        m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
        h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        w : Eq (CategoryTheory.CategoryStruct.comp m (Q.lift (CategoryTheory.Limits.Bi …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m s.snd) (CategoryTheory.CategoryStru …
      -/
      rw [← w]
      /-
        case mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y Z : C
        sXY : CategoryTheory.Limits.BinaryFan X Y
        P : CategoryTheory.Limits.IsLimit sXY
        sYZ : CategoryTheory.Limits.BinaryFan Y Z
        Q : CategoryTheory.Limits.IsLimit sYZ
        s : CategoryTheory.Limits.BinaryFan sXY.pt Z
        R : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X sYZ.pt)
        m : Quiver.Hom t.pt (CategoryTheory.Limits.BinaryFan.assoc Q s).pt
        h : (∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Ca …
        w : Eq (CategoryTheory.CategoryStruct.comp m (Q.lift (CategoryTheory.Limits.Bi …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m s.snd) (CategoryTheory.CategoryStru …
      -/
      simp
      /-
        🎉 no goals
      -/


/-- Given two pairs of limit cones corresponding to the parenthesisations of `X × Y × Z`,
we obtain an isomorphism between the cone points.
-/
abbrev BinaryFan.associator {X Y Z : C} {sXY : BinaryFan X Y} (P : IsLimit sXY)
    {sYZ : BinaryFan Y Z} (Q : IsLimit sYZ) {s : BinaryFan sXY.pt Z} (R : IsLimit s)
    {t : BinaryFan X sYZ.pt} (S : IsLimit t) : s.pt ≅ t.pt :=
  IsLimit.conePointUniqueUpToIso (IsLimit.assoc P Q R) S


/-- Given a fixed family of limit data for every pair `X Y`, we obtain an associator.
-/
abbrev BinaryFan.associatorOfLimitCone (L : ∀ X Y : C, LimitCone (pair X Y)) (X Y Z : C) :
    (L (L X Y).cone.pt Z).cone.pt ≅ (L X (L Y Z).cone.pt).cone.pt :=
  BinaryFan.associator (L X Y).isLimit (L Y Z).isLimit (L (L X Y).cone.pt Z).isLimit
    (L X (L Y Z).cone.pt).isLimit


/-- Construct a left unitor from specified limit cones.
-/
@[simps]
def BinaryFan.leftUnitor {X : C} {s : Cone (Functor.empty.{0} C)} (P : IsLimit s)
    {t : BinaryFan s.pt X} (Q : IsLimit t) : t.pt ≅ X where
  hom := t.snd
  inv := Q.lift <| BinaryFan.mk (P.lift ⟨_, fun x => x.as.elim, fun {x} => x.as.elim⟩) (𝟙 _)
  hom_inv_id := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y X : C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
      P : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.BinaryFan s.pt X
      Q : CategoryTheory.Limits.IsLimit t
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t.snd (Q.lift (CategoryTheory.Limits. …
    -/
    apply Q.hom_ext
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y X : C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
      P : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.BinaryFan s.pt X
      Q : CategoryTheory.Limits.IsLimit t
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨⟨⟩⟩
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y X : C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
        P : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.BinaryFan s.pt X
        Q : CategoryTheory.Limits.IsLimit t
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
      -/
    · apply P.hom_ext
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y X : C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
        P : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.BinaryFan s.pt X
        Q : CategoryTheory.Limits.IsLimit t
        ⊢ ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategoryStruc …
      -/
      rintro ⟨⟨⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y X : C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
        P : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.BinaryFan s.pt X
        Q : CategoryTheory.Limits.IsLimit t
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
      -/
    · simp
      /-
        🎉 no goals
      -/


/-- Construct a right unitor from specified limit cones.
-/
@[simps]
def BinaryFan.rightUnitor {X : C} {s : Cone (Functor.empty.{0} C)} (P : IsLimit s)
    {t : BinaryFan X s.pt} (Q : IsLimit t) : t.pt ≅ X where
  hom := t.fst
  inv := Q.lift <| BinaryFan.mk (𝟙 _) <| P.lift ⟨_, fun x => x.as.elim, fun {x} => x.as.elim⟩
  hom_inv_id := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y X : C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
      P : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.BinaryFan X s.pt
      Q : CategoryTheory.Limits.IsLimit t
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t.fst (Q.lift (CategoryTheory.Limits. …
    -/
    apply Q.hom_ext
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y X : C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
      P : CategoryTheory.Limits.IsLimit s
      t : CategoryTheory.Limits.BinaryFan X s.pt
      Q : CategoryTheory.Limits.IsLimit t
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨⟨⟩⟩
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y X : C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
        P : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.BinaryFan X s.pt
        Q : CategoryTheory.Limits.IsLimit t
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y X : C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
        P : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.BinaryFan X s.pt
        Q : CategoryTheory.Limits.IsLimit t
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
      -/
    · apply P.hom_ext
      /-
        case mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y X : C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty C)
        P : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.BinaryFan X s.pt
        Q : CategoryTheory.Limits.IsLimit t
        ⊢ ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategoryStruc …
      -/
      rintro ⟨⟨⟩⟩
      /-
        🎉 no goals
      -/


/-- Implementation of the tensor product for `MonoidalOfChosenFiniteProducts`. -/
abbrev tensorObj (X Y : C) : C :=
  (ℬ X Y).cone.pt


/-- Implementation of the tensor product of morphisms for `MonoidalOfChosenFiniteProducts`. -/
abbrev tensorHom {W X Y Z : C} (f : W ⟶ X) (g : Y ⟶ Z) : tensorObj ℬ W Y ⟶ tensorObj ℬ X Z :=
  (BinaryFan.IsLimit.lift' (ℬ X Z).isLimit ((ℬ W Y).cone.π.app ⟨WalkingPair.left⟩ ≫ f)
      (((ℬ W Y).cone.π.app ⟨WalkingPair.right⟩ : (ℬ W Y).cone.pt ⟶ Y) ≫ g)).val


theorem tensor_id (X₁ X₂ : C) : tensorHom ℬ (𝟙 X₁) (𝟙 X₂) = 𝟙 (tensorObj ℬ X₁ X₂) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ X₂ : C
    ⊢ Eq (CategoryTheory.MonoidalOfChosenFiniteProducts.tensorHom ℬ (CategoryTheor …
  -/
  apply IsLimit.hom_ext (ℬ _ _).isLimit
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ X₂ : C
    ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
  -/
  rintro ⟨⟨⟩⟩ <;>
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        X₁ X₂ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalOfChosenFinit …
      -/
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        X₁ X₂ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ X₁ X₂).isLimit.lift (CategoryTheo …
      -/
      /-
        🎉 no goals
      -/
      /-
        case mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        X₁ X₂ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ X₁ X₂).isLimit.lift (CategoryTheo …
      -/
      simp
      /-
        🎉 no goals
      -/


theorem tensor_comp {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (g₁ : Y₁ ⟶ Z₁)
    (g₂ : Y₂ ⟶ Z₂) : tensorHom ℬ (f₁ ≫ g₁) (f₂ ≫ g₂) = tensorHom ℬ f₁ f₂ ≫ tensorHom ℬ g₁ g₂ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom Y₁ Z₁
    g₂ : Quiver.Hom Y₂ Z₂
    ⊢ Eq (CategoryTheory.MonoidalOfChosenFiniteProducts.tensorHom ℬ (CategoryTheor …
  -/
  apply IsLimit.hom_ext (ℬ _ _).isLimit
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom Y₁ Z₁
    g₂ : Quiver.Hom Y₂ Z₂
    ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
  -/
  rintro ⟨⟨⟩⟩ <;>
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C
        f₁ : Quiver.Hom X₁ Y₁
        f₂ : Quiver.Hom X₂ Y₂
        g₁ : Quiver.Hom Y₁ Z₁
        g₂ : Quiver.Hom Y₂ Z₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalOfChosenFinit …
      -/
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C
        f₁ : Quiver.Hom X₁ Y₁
        f₂ : Quiver.Hom X₂ Y₂
        g₁ : Quiver.Hom Y₁ Z₁
        g₂ : Quiver.Hom Y₂ Z₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ Z₁ Z₂).isLimit.lift (CategoryTheo …
      -/
      /-
        🎉 no goals
      -/
      /-
        case mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        X₁ Y₁ Z₁ X₂ Y₂ Z₂ : C
        f₁ : Quiver.Hom X₁ Y₁
        f₂ : Quiver.Hom X₂ Y₂
        g₁ : Quiver.Hom Y₁ Z₁
        g₂ : Quiver.Hom Y₂ Z₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ Z₁ Z₂).isLimit.lift (CategoryTheo …
      -/
      simp
      /-
        🎉 no goals
      -/


theorem pentagon (W X Y Z : C) :
    tensorHom ℬ (BinaryFan.associatorOfLimitCone ℬ W X Y).hom (𝟙 Z) ≫
        (BinaryFan.associatorOfLimitCone ℬ W (tensorObj ℬ X Y) Z).hom ≫
          tensorHom ℬ (𝟙 W) (BinaryFan.associatorOfLimitCone ℬ X Y Z).hom =
      (BinaryFan.associatorOfLimitCone ℬ (tensorObj ℬ W X) Y Z).hom ≫
        (BinaryFan.associatorOfLimitCone ℬ W X (tensorObj ℬ Y Z)).hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalOfChosenFinit …
  -/
  dsimp [tensorHom]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ (ℬ W (ℬ X Y).cone.pt).cone.pt Z). …
  -/
  apply IsLimit.hom_ext (ℬ _ _).isLimit; rintro ⟨⟨⟩⟩
    /-
      case mk.left
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      W X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      W X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · apply IsLimit.hom_ext (ℬ _ _).isLimit
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      W X Y Z : C
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨⟨⟩⟩
      /-
        case mk.right.mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        W X Y Z : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case mk.right.mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      W X Y Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    apply IsLimit.hom_ext (ℬ _ _).isLimit
    /-
      case mk.right.mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      W X Y Z : C
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨⟨⟩⟩
      /-
        case mk.right.mk.right.mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        W X Y Z : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mk.right.mk.right.mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        W X Y Z : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem triangle (X Y : C) :
    (BinaryFan.associatorOfLimitCone ℬ X 𝒯.cone.pt Y).hom ≫
        tensorHom ℬ (𝟙 X) (BinaryFan.leftUnitor 𝒯.isLimit (ℬ 𝒯.cone.pt Y).isLimit).hom =
      tensorHom ℬ (BinaryFan.rightUnitor 𝒯.isLimit (ℬ X 𝒯.cone.pt).isLimit).hom (𝟙 Y) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.asso …
  -/
  dsimp [tensorHom]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.asso …
  -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  apply IsLimit.hom_ext (ℬ _ _).isLimit; rintro ⟨⟨⟩⟩ <;> simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem leftUnitor_naturality {X₁ X₂ : C} (f : X₁ ⟶ X₂) :
    tensorHom ℬ (𝟙 𝒯.cone.pt) f ≫ (BinaryFan.leftUnitor 𝒯.isLimit (ℬ 𝒯.cone.pt X₂).isLimit).hom =
      (BinaryFan.leftUnitor 𝒯.isLimit (ℬ 𝒯.cone.pt X₁).isLimit).hom ≫ f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ X₂ : C
    f : Quiver.Hom X₁ X₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalOfChosenFinit …
  -/
  dsimp [tensorHom]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ X₂ : C
    f : Quiver.Hom X₁ X₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ 𝒯.cone.pt X₂).isLimit.lift (Categ …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem rightUnitor_naturality {X₁ X₂ : C} (f : X₁ ⟶ X₂) :
    tensorHom ℬ f (𝟙 𝒯.cone.pt) ≫ (BinaryFan.rightUnitor 𝒯.isLimit (ℬ X₂ 𝒯.cone.pt).isLimit).hom =
      (BinaryFan.rightUnitor 𝒯.isLimit (ℬ X₁ 𝒯.cone.pt).isLimit).hom ≫ f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ X₂ : C
    f : Quiver.Hom X₁ X₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalOfChosenFinit …
  -/
  dsimp [tensorHom]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ X₂ : C
    f : Quiver.Hom X₁ X₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ X₂ 𝒯.cone.pt).isLimit.lift (Categ …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem associator_naturality {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (f₃ : X₃ ⟶ Y₃) :
    tensorHom ℬ (tensorHom ℬ f₁ f₂) f₃ ≫ (BinaryFan.associatorOfLimitCone ℬ Y₁ Y₂ Y₃).hom =
      (BinaryFan.associatorOfLimitCone ℬ X₁ X₂ X₃).hom ≫ tensorHom ℬ f₁ (tensorHom ℬ f₂ f₃) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    f₃ : Quiver.Hom X₃ Y₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalOfChosenFinit …
  -/
  dsimp [tensorHom]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    f₃ : Quiver.Hom X₃ Y₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ℬ (CategoryTheory.MonoidalOfChosenF …
  -/
  apply IsLimit.hom_ext (ℬ _ _).isLimit; rintro ⟨⟨⟩⟩
    /-
      case mk.left
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₂
      f₃ : Quiver.Hom X₃ Y₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₂
      f₃ : Quiver.Hom X₃ Y₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · apply IsLimit.hom_ext (ℬ _ _).isLimit
    /-
      case mk.right
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
      X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₂
      f₃ : Quiver.Hom X₃ Y₃
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨⟨⟩⟩
      /-
        case mk.right.mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
        f₁ : Quiver.Hom X₁ Y₁
        f₂ : Quiver.Hom X₂ Y₂
        f₃ : Quiver.Hom X₃ Y₃
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mk.right.mk.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
        X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
        f₁ : Quiver.Hom X₁ Y₁
        f₂ : Quiver.Hom X₂ Y₂
        f₃ : Quiver.Hom X₃ Y₃
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        🎉 no goals
      -/


/-- A category with a terminal object and binary products has a natural monoidal structure. -/
def monoidalOfChosenFiniteProducts : MonoidalCategory C :=
  letI : MonoidalCategoryStruct C :=
    { tensorUnit := 𝒯.cone.pt
      tensorObj := tensorObj ℬ
      tensorHom := tensorHom ℬ
      whiskerLeft := @fun X {_ _} g ↦ tensorHom ℬ (𝟙 X) g
      whiskerRight := @fun{_ _} f Y ↦ tensorHom ℬ f (𝟙 Y)
      associator := BinaryFan.associatorOfLimitCone ℬ
      leftUnitor := fun X ↦ BinaryFan.leftUnitor 𝒯.isLimit (ℬ 𝒯.cone.pt X).isLimit
      rightUnitor := fun X ↦ BinaryFan.rightUnitor 𝒯.isLimit (ℬ X 𝒯.cone.pt).isLimit}
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    this : CategoryTheory.MonoidalCategoryStruct C := { tensorObj := CategoryTheor …
    ⊢ ∀ (X : C) {Y₁ Y₂ : C} (f : Quiver.Hom Y₁ Y₂), Eq (CategoryTheory.MonoidalCat …
  -/
  /-
    🎉 no goals
  -/
  .ofTensorHom
  /-
    🎉 no goals
  -/
    (tensor_id := tensor_id ℬ)
    (tensor_comp := tensor_comp ℬ)
    (pentagon := pentagon ℬ)
    (triangle := triangle 𝒯 ℬ)
    (leftUnitor_naturality := leftUnitor_naturality 𝒯 ℬ)
    (rightUnitor_naturality := rightUnitor_naturality 𝒯 ℬ)
    (associator_naturality := associator_naturality ℬ)


/-- A type synonym for `C` carrying a monoidal category structure corresponding to
a fixed choice of limit data for the empty functor, and for `pair X Y` for every `X Y : C`.

This is an implementation detail for `SymmetricOfChosenFiniteProducts`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter `has_nonempty_instance` not ported yet
-- @[nolint has_nonempty_instance]
@[nolint unusedArguments]
def MonoidalOfChosenFiniteProductsSynonym (_𝒯 : LimitCone (Functor.empty.{0} C))
    (_ℬ : ∀ X Y : C, LimitCone (pair X Y)) :=
  C


instance : Category (MonoidalOfChosenFiniteProductsSynonym 𝒯 ℬ) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    ⊢ CategoryTheory.Category.{?u.82428, u} (CategoryTheory.MonoidalOfChosenFinite …
  -/
  dsimp [MonoidalOfChosenFiniteProductsSynonym]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    𝒯 : CategoryTheory.Limits.LimitCone (CategoryTheory.Functor.empty C)
    ℬ : (X Y : C) → CategoryTheory.Limits.LimitCone (CategoryTheory.Limits.pair X Y)
    ⊢ CategoryTheory.Category.{?u.82428, u} C
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : MonoidalCategory (MonoidalOfChosenFiniteProductsSynonym 𝒯 ℬ) :=
  monoidalOfChosenFiniteProducts 𝒯 ℬ


