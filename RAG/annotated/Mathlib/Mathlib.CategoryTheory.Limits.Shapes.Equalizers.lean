/-- The type of objects for the diagram indexing a (co)equalizer. -/
inductive WalkingParallelPair : Type
  | zero
  | one
  deriving DecidableEq, Inhabited


/-- The type family of morphisms for the diagram indexing a (co)equalizer. -/
inductive WalkingParallelPairHom : WalkingParallelPair → WalkingParallelPair → Type
  | left : WalkingParallelPairHom zero one
  | right : WalkingParallelPairHom zero one
  | id (X : WalkingParallelPair) : WalkingParallelPairHom X X
  deriving DecidableEq

/- Porting note: this simplifies using walkingParallelPairHom_id; replacement is below;
simpNF still complains of striking this from the simp list -/

/-- Satisfying the inhabited linter -/
instance : Inhabited (WalkingParallelPairHom zero one) where default := WalkingParallelPairHom.left


/-- Composition of morphisms in the indexing diagram for (co)equalizers. -/
def WalkingParallelPairHom.comp :
    -- Porting note: changed X Y Z to implicit to match comp fields in precategory
    ∀ {X Y Z : WalkingParallelPair} (_ : WalkingParallelPairHom X Y)
      (_ : WalkingParallelPairHom Y Z), WalkingParallelPairHom X Z
  | _, _, _, id _, h => h
  | _, _, _, left, id one => left
  | _, _, _, right, id one => right

-- Porting note: adding these since they are simple and aesop couldn't directly prove them

theorem WalkingParallelPairHom.id_comp
    {X Y : WalkingParallelPair} (g : WalkingParallelPairHom X Y) : comp (id X) g = g :=
  rfl


theorem WalkingParallelPairHom.comp_id
    {X Y : WalkingParallelPair} (f : WalkingParallelPairHom X Y) : comp f (id Y) = f := by
  /-
    X Y : CategoryTheory.Limits.WalkingParallelPair
    f : CategoryTheory.Limits.WalkingParallelPairHom X Y
    ⊢ Eq (f.comp (CategoryTheory.Limits.WalkingParallelPairHom.id Y)) f
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases f <;> rfl
              /-
                🎉 no goals
              -/


theorem WalkingParallelPairHom.assoc {X Y Z W : WalkingParallelPair}
    (f : WalkingParallelPairHom X Y) (g : WalkingParallelPairHom Y Z)
    (h : WalkingParallelPairHom Z W) : comp (comp f g) h = comp f (comp g h) := by
  /-
    X Y Z W : CategoryTheory.Limits.WalkingParallelPair
    f : CategoryTheory.Limits.WalkingParallelPairHom X Y
    g : CategoryTheory.Limits.WalkingParallelPairHom Y Z
    h : CategoryTheory.Limits.WalkingParallelPairHom Z W
    ⊢ Eq ((f.comp g).comp h) (f.comp (g.comp h))
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
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  cases f <;> cases g <;> cases h <;> rfl
                                      /-
                                        🎉 no goals
                                      -/


instance walkingParallelPairHomCategory : SmallCategory WalkingParallelPair where
  Hom := WalkingParallelPairHom
  id := id
  comp := comp
  comp_id := comp_id
  id_comp := id_comp
  assoc := assoc


@[simp]
theorem walkingParallelPairHom_id (X : WalkingParallelPair) : WalkingParallelPairHom.id X = 𝟙 X :=
  rfl

-- Porting note: simpNF asked me to do this because the LHS of the non-primed version reduced

@[simp]
theorem WalkingParallelPairHom.id.sizeOf_spec' (X : WalkingParallelPair) :
                                                                                /-
                                                                                  X : CategoryTheory.Limits.WalkingParallelPair
                                                                                  ⊢ Eq (SizeOf.sizeOf (CategoryTheory.CategoryStruct.id X)) (HAdd.hAdd 1 (SizeOf …
                                                                                -/
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
    (WalkingParallelPairHom._sizeOf_inst X X).sizeOf (𝟙 X) = 1 + sizeOf X := by cases X <;> rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


/-- The functor `WalkingParallelPair ⥤ WalkingParallelPairᵒᵖ` sending left to left and right to
right.
-/
def walkingParallelPairOp : WalkingParallelPair ⥤ WalkingParallelPairᵒᵖ where
                    /-
                      x : CategoryTheory.Limits.WalkingParallelPair
                      ⊢ CategoryTheory.Limits.WalkingParallelPair
                    -/
  obj x := op <| by cases x; exacts [one, zero]
                             /-
                               🎉 no goals
                             -/
  map f := by
    /-
      X✝ Y✝ : CategoryTheory.Limits.WalkingParallelPair
      f : Quiver.Hom X✝ Y✝
      ⊢ Quiver.Hom ((fun x => { unop := CategoryTheory.Limits.WalkingParallelPair.ca …
    -/
    cases f <;> apply Quiver.Hom.op
    /-
      case left.f
      ⊢ Quiver.Hom (CategoryTheory.Limits.WalkingParallelPair.casesOn (motive := fun …
    -/
    exacts [left, right, WalkingParallelPairHom.id _]
    /-
      🎉 no goals
    -/
                 /-
                   ⊢ ∀ {X Y Z : CategoryTheory.Limits.WalkingParallelPair} (f : Quiver.Hom X Y) ( …
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
                                                        /-
                                                          🎉 no goals
                                                        -/
  map_comp := by rintro _ _ _ (_|_|_) g <;> cases g <;> rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem walkingParallelPairOp_zero : walkingParallelPairOp.obj zero = op one := rfl


@[simp]
theorem walkingParallelPairOp_one : walkingParallelPairOp.obj one = op zero := rfl


@[simp]
theorem walkingParallelPairOp_left :
    walkingParallelPairOp.map left = @Quiver.Hom.op _ _ zero one left := rfl


@[simp]
theorem walkingParallelPairOp_right :
    walkingParallelPairOp.map right = @Quiver.Hom.op _ _ zero one right := rfl


/--
The equivalence `WalkingParallelPair ⥤ WalkingParallelPairᵒᵖ` sending left to left and right to
right.
-/
@[simps functor inverse]
def walkingParallelPairOpEquiv : WalkingParallelPair ≌ WalkingParallelPairᵒᵖ where
  functor := walkingParallelPairOp
  inverse := walkingParallelPairOp.leftOp
  unitIso :=
                                              /-
                                                j : CategoryTheory.Limits.WalkingParallelPair
                                                ⊢ Eq ((CategoryTheory.Functor.id CategoryTheory.Limits.WalkingParallelPair).ob …
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    NatIso.ofComponents (fun j => eqToIso (by cases j <;> rfl))
                                                          /-
                                                            🎉 no goals
                                                          -/
          /-
            ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingParallelPair} (f : Quiver.Hom X Y), Eq …
          -/
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
      (by rintro _ _ (_ | _ | _) <;> simp)
                                     /-
                                       🎉 no goals
                                     -/
  counitIso :=
    NatIso.ofComponents (fun j => eqToIso (by
            /-
              j : Opposite CategoryTheory.Limits.WalkingParallelPair
              ⊢ Eq ((CategoryTheory.Limits.walkingParallelPairOp.leftOp.comp CategoryTheory. …
            -/
            induction' j with X
            /-
              case h
              X : CategoryTheory.Limits.WalkingParallelPair
              ⊢ Eq ((CategoryTheory.Limits.walkingParallelPairOp.leftOp.comp CategoryTheory. …
            -/
                        /-
                          🎉 no goals
                        -/
            cases X <;> rfl))
                        /-
                          🎉 no goals
                        -/
      (fun {i} {j} f => by
      /-
        i j : Opposite CategoryTheory.Limits.WalkingParallelPair
        f : Quiver.Hom i j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.walkingParall …
      -/
      induction' i with i
      /-
        case h
        j : Opposite CategoryTheory.Limits.WalkingParallelPair
        i : CategoryTheory.Limits.WalkingParallelPair
        f : Quiver.Hom { unop := i } j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.walkingParall …
      -/
      induction' j with j
      /-
        case h.h
        i j : CategoryTheory.Limits.WalkingParallelPair
        f : Quiver.Hom { unop := i } { unop := j }
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.walkingParall …
      -/
      let g := f.unop
      /-
        case h.h
        i j : CategoryTheory.Limits.WalkingParallelPair
        f : Quiver.Hom { unop := i } { unop := j }
        g : Quiver.Hom (Opposite.unop { unop := j }) (Opposite.unop { unop := i }) :=  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.walkingParall …
      -/
      have : f = g.op := rfl
      /-
        case h.h
        i j : CategoryTheory.Limits.WalkingParallelPair
        f : Quiver.Hom { unop := i } { unop := j }
        g : Quiver.Hom (Opposite.unop { unop := j }) (Opposite.unop { unop := i }) :=  …
        this : Eq f g.op
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.walkingParall …
      -/
      rw [this]
      /-
        case h.h
        i j : CategoryTheory.Limits.WalkingParallelPair
        f : Quiver.Hom { unop := i } { unop := j }
        g : Quiver.Hom (Opposite.unop { unop := j }) (Opposite.unop { unop := i }) :=  …
        this : Eq f g.op
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.walkingParall …
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
                                          /-
                                            🎉 no goals
                                          -/
      cases i <;> cases j <;> cases g <;> rfl)
                                          /-
                                            🎉 no goals
                                          -/
                                      /-
                                        j : CategoryTheory.Limits.WalkingParallelPair
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.walkingParalle …
                                      -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  functor_unitIso_comp := fun j => by cases j <;> rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem walkingParallelPairOpEquiv_unitIso_zero :
    walkingParallelPairOpEquiv.unitIso.app zero = Iso.refl zero := rfl


@[simp]
theorem walkingParallelPairOpEquiv_unitIso_one :
    walkingParallelPairOpEquiv.unitIso.app one = Iso.refl one := rfl


@[simp]
theorem walkingParallelPairOpEquiv_counitIso_zero :
    walkingParallelPairOpEquiv.counitIso.app (op zero) = Iso.refl (op zero) := rfl


@[simp]
theorem walkingParallelPairOpEquiv_counitIso_one :
    walkingParallelPairOpEquiv.counitIso.app (op one) = Iso.refl (op one) :=
  rfl


/-- `parallelPair f g` is the diagram in `C` consisting of the two morphisms `f` and `g` with
    common domain and codomain. -/
def parallelPair (f g : X ⟶ Y) : WalkingParallelPair ⥤ C where
  obj x :=
    match x with
    | zero => X
    | one => Y
  map h :=
    match h with
    | WalkingParallelPairHom.id _ => 𝟙 _
    | left => f
    | right => g
  -- `sorry` can cope with this, but it's too slow:
  map_comp := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      ⊢ ∀ {X_1 Y_1 Z : CategoryTheory.Limits.WalkingParallelPair} (f_1 : Quiver.Hom  …
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
                                      /-
                                        🎉 no goals
                                      -/
    rintro _ _ _ ⟨⟩ g <;> cases g <;> {dsimp; simp}
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem parallelPair_obj_zero (f g : X ⟶ Y) : (parallelPair f g).obj zero = X := rfl


@[simp]
theorem parallelPair_obj_one (f g : X ⟶ Y) : (parallelPair f g).obj one = Y := rfl


@[simp]
theorem parallelPair_map_left (f g : X ⟶ Y) : (parallelPair f g).map left = f := rfl


@[simp]
theorem parallelPair_map_right (f g : X ⟶ Y) : (parallelPair f g).map right = g := rfl


@[simp]
theorem parallelPair_functor_obj {F : WalkingParallelPair ⥤ C} (j : WalkingParallelPair) :
                                                                    /-
                                                                      C : Type u
                                                                      inst✝ : CategoryTheory.Category.{v, u} C
                                                                      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                                      j : CategoryTheory.Limits.WalkingParallelPair
                                                                      ⊢ Eq ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limits.Walking …
                                                                    -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    (parallelPair (F.map left) (F.map right)).obj j = F.obj j := by cases j <;> rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- Every functor indexing a (co)equalizer is naturally isomorphic (actually, equal) to a
    `parallelPair` -/
@[simps!]
def diagramIsoParallelPair (F : WalkingParallelPair ⥤ C) :
    F ≅ parallelPair (F.map left) (F.map right) :=
                                              /-
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                X Y : C
                                                F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                j : CategoryTheory.Limits.WalkingParallelPair
                                                ⊢ Eq (F.obj j) ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limi …
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
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
  NatIso.ofComponents (fun j => eqToIso <| by cases j <;> rfl) (by rintro _ _ (_|_|_) <;> simp)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- Construct a morphism between parallel pairs. -/
def parallelPairHom {X' Y' : C} (f g : X ⟶ Y) (f' g' : X' ⟶ Y') (p : X ⟶ X') (q : Y ⟶ Y')
    (wf : f ≫ q = p ≫ f') (wg : g ≫ q = p ≫ g') : parallelPair f g ⟶ parallelPair f' g' where
  app j :=
    match j with
    | zero => p
    | one => q
  naturality := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y X' Y' : C
      f g : Quiver.Hom X Y
      f' g' : Quiver.Hom X' Y'
      p : Quiver.Hom X X'
      q : Quiver.Hom Y Y'
      wf : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStruc …
      wg : Eq (CategoryTheory.CategoryStruct.comp g q) (CategoryTheory.CategoryStruc …
      ⊢ ∀ ⦃X_1 Y_1 : CategoryTheory.Limits.WalkingParallelPair⦄ (f_1 : Quiver.Hom X_ …
    -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
    rintro _ _ ⟨⟩ <;> {dsimp; simp [wf,wg]}
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem parallelPairHom_app_zero {X' Y' : C} (f g : X ⟶ Y) (f' g' : X' ⟶ Y') (p : X ⟶ X')
    (q : Y ⟶ Y') (wf : f ≫ q = p ≫ f') (wg : g ≫ q = p ≫ g') :
    (parallelPairHom f g f' g' p q wf wg).app zero = p :=
  rfl


@[simp]
theorem parallelPairHom_app_one {X' Y' : C} (f g : X ⟶ Y) (f' g' : X' ⟶ Y') (p : X ⟶ X')
    (q : Y ⟶ Y') (wf : f ≫ q = p ≫ f') (wg : g ≫ q = p ≫ g') :
    (parallelPairHom f g f' g' p q wf wg).app one = q :=
  rfl


/-- Construct a natural isomorphism between functors out of the walking parallel pair from
its components. -/
@[simps!]
def parallelPair.ext {F G : WalkingParallelPair ⥤ C} (zero : F.obj zero ≅ G.obj zero)
    (one : F.obj one ≅ G.obj one) (left : F.map left ≫ one.hom = zero.hom ≫ G.map left)
    (right : F.map right ≫ one.hom = zero.hom ≫ G.map right) : F ≅ G :=
  NatIso.ofComponents
    (by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
        zero : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingParallelPair.zer …
        one : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingParallelPair.one) …
        left : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Wal …
        right : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Wa …
        ⊢ (X : CategoryTheory.Limits.WalkingParallelPair) → CategoryTheory.Iso (F.obj  …
      -/
      rintro ⟨j⟩
      /-
        case zero
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
        zero : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingParallelPair.zer …
        one : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingParallelPair.one) …
        left : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Wal …
        right : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Wa …
        ⊢ CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingParallelPair.zero) (G …
      -/
      exacts [zero, one])
      /-
        🎉 no goals
      -/
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
          zero : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingParallelPair.zer …
          one : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingParallelPair.one) …
          left : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Wal …
          right : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Wa …
          ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingParallelPair} (f : Quiver.Hom X Y), Eq …
        -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
    (by rintro _ _ ⟨_⟩ <;> simp [left, right])
                           /-
                             🎉 no goals
                           -/


/-- Construct a natural isomorphism between `parallelPair f g` and `parallelPair f' g'` given
equalities `f = f'` and `g = g'`. -/
@[simps!]
def parallelPair.eqOfHomEq {f g f' g' : X ⟶ Y} (hf : f = f') (hg : g = g') :
    parallelPair f g ≅ parallelPair f' g' :=
                                                 /-
                                                   C : Type u
                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                   X Y : C
                                                   f g f' g' : Quiver.Hom X Y
                                                   hf : Eq f f'
                                                   hg : Eq g g'
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.parallelPair  …
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  parallelPair.ext (Iso.refl _) (Iso.refl _) (by simp [hf]) (by simp [hg])
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- A fork on `f` and `g` is just a `Cone (parallelPair f g)`. -/
abbrev Fork (f g : X ⟶ Y) :=
  Cone (parallelPair f g)


/-- A cofork on `f` and `g` is just a `Cocone (parallelPair f g)`. -/
abbrev Cofork (f g : X ⟶ Y) :=
  Cocone (parallelPair f g)


/-- A fork `t` on the parallel pair `f g : X ⟶ Y` consists of two morphisms
    `t.π.app zero : t.pt ⟶ X`
    and `t.π.app one : t.pt ⟶ Y`. Of these, only the first one is interesting, and we give it the
    shorter name `Fork.ι t`. -/
def Fork.ι (t : Fork f g) :=
  t.π.app zero


@[simp]
theorem Fork.app_zero_eq_ι (t : Fork f g) : t.π.app zero = t.ι :=
  rfl


/-- A cofork `t` on the parallelPair `f g : X ⟶ Y` consists of two morphisms
    `t.ι.app zero : X ⟶ t.pt` and `t.ι.app one : Y ⟶ t.pt`. Of these, only the second one is
    interesting, and we give it the shorter name `Cofork.π t`. -/
def Cofork.π (t : Cofork f g) :=
  t.ι.app one


@[simp]
theorem Cofork.app_one_eq_π (t : Cofork f g) : t.ι.app one = t.π :=
  rfl


@[simp]
theorem Fork.app_one_eq_ι_comp_left (s : Fork f g) : s.π.app one = s.ι ≫ f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Fork f g
    ⊢ Eq (s.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory.C …
  -/
  rw [← s.app_zero_eq_ι, ← s.w left, parallelPair_map_left]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem Fork.app_one_eq_ι_comp_right (s : Fork f g) : s.π.app one = s.ι ≫ g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Fork f g
    ⊢ Eq (s.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory.C …
  -/
  rw [← s.app_zero_eq_ι, ← s.w right, parallelPair_map_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem Cofork.app_zero_eq_comp_π_left (s : Cofork f g) : s.ι.app zero = f ≫ s.π := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    ⊢ Eq (s.ι.app CategoryTheory.Limits.WalkingParallelPair.zero) (CategoryTheory. …
  -/
  rw [← s.app_one_eq_π, ← s.w left, parallelPair_map_left]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem Cofork.app_zero_eq_comp_π_right (s : Cofork f g) : s.ι.app zero = g ≫ s.π := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    s : CategoryTheory.Limits.Cofork f g
    ⊢ Eq (s.ι.app CategoryTheory.Limits.WalkingParallelPair.zero) (CategoryTheory. …
  -/
  rw [← s.app_one_eq_π, ← s.w right, parallelPair_map_right]
  /-
    🎉 no goals
  -/


/-- A fork on `f g : X ⟶ Y` is determined by the morphism `ι : P ⟶ X` satisfying `ι ≫ f = ι ≫ g`.
-/
@[simps]
def Fork.ofι {P : C} (ι : P ⟶ X) (w : ι ≫ f = ι ≫ g) : Fork f g where
  pt := P
  π :=
    { app := fun X => by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y : C
          f g : Quiver.Hom X✝ Y
          P : C
          ι : Quiver.Hom P X✝
          w : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruct …
          X : CategoryTheory.Limits.WalkingParallelPair
          ⊢ Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPara …
        -/
        cases X
          /-
            case zero
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X Y : C
            f g : Quiver.Hom X Y
            P : C
            ι : Quiver.Hom P X
            w : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruct …
            ⊢ Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPara …
          -/
        · exact ι
          /-
            🎉 no goals
          -/
          /-
            case one
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X Y : C
            f g : Quiver.Hom X Y
            P : C
            ι : Quiver.Hom P X
            w : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruct …
            ⊢ Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPara …
          -/
        · exact ι ≫ f
          /-
            🎉 no goals
          -/
      naturality := fun {X} {Y} f =>
           /-
             C : Type u
             inst✝ : CategoryTheory.Category.{v, u} C
             X✝ Y✝ : C
             f✝ g : Quiver.Hom X✝ Y✝
             P : C
             ι : Quiver.Hom P X✝
             w : Eq (CategoryTheory.CategoryStruct.comp ι f✝) (CategoryTheory.CategoryStruc …
             X Y : CategoryTheory.Limits.WalkingParallelPair
             f : Quiver.Hom X Y
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Categ …
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
                                                         /-
                                                           🎉 no goals
                                                         -/
        by cases X <;> cases Y <;> cases f <;> dsimp <;> simp; assumption }
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- A cofork on `f g : X ⟶ Y` is determined by the morphism `π : Y ⟶ P` satisfying
    `f ≫ π = g ≫ π`. -/
@[simps]
def Cofork.ofπ {P : C} (π : Y ⟶ P) (w : f ≫ π = g ≫ π) : Cofork f g where
  pt := P
  ι :=
    { app := fun X => WalkingParallelPair.casesOn X (f ≫ π) π
                                    /-
                                      C : Type u
                                      inst✝ : CategoryTheory.Category.{v, u} C
                                      X Y : C
                                      f✝ g : Quiver.Hom X Y
                                      P : C
                                      π : Quiver.Hom Y P
                                      w : Eq (CategoryTheory.CategoryStruct.comp f✝ π) (CategoryTheory.CategoryStruc …
                                      i j : CategoryTheory.Limits.WalkingParallelPair
                                      f : Quiver.Hom i j
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.parallelPair  …
                                    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
      naturality := fun i j f => by cases f <;> dsimp <;> simp [w] }
                                                          /-
                                                            🎉 no goals
                                                          -/

-- See note [dsimp, simp]

@[simp]
theorem Fork.ι_ofι {P : C} (ι : P ⟶ X) (w : ι ≫ f = ι ≫ g) : (Fork.ofι ι w).ι = ι :=
  rfl


@[simp]
theorem Cofork.π_ofπ {P : C} (π : Y ⟶ P) (w : f ≫ π = g ≫ π) : (Cofork.ofπ π w).π = π :=
  rfl


@[reassoc (attr := simp)]
theorem Fork.condition (t : Fork f g) : t.ι ≫ f = t.ι ≫ g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    t : CategoryTheory.Limits.Fork f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp t.ι f) (CategoryTheory.CategoryStruct …
  -/
  rw [← t.app_one_eq_ι_comp_left, ← t.app_one_eq_ι_comp_right]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem Cofork.condition (t : Cofork f g) : f ≫ t.π = g ≫ t.π := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    t : CategoryTheory.Limits.Cofork f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f t.π) (CategoryTheory.CategoryStruct …
  -/
  rw [← t.app_zero_eq_comp_π_left, ← t.app_zero_eq_comp_π_right]
  /-
    🎉 no goals
  -/


/-- To check whether two maps are equalized by both maps of a fork, it suffices to check it for the
    first map -/
theorem Fork.equalizer_ext (s : Fork f g) {W : C} {k l : W ⟶ s.pt} (h : k ≫ s.ι = l ≫ s.ι) :
    ∀ j : WalkingParallelPair, k ≫ s.π.app j = l ≫ s.π.app j
  | zero => h
  | one => by
    have : k ≫ ι s ≫ f = l ≫ ι s ≫ f := by
      simp only [← Category.assoc]; exact congrArg (· ≫ f) h
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      s : CategoryTheory.Limits.Fork f g
      W : C
      k l : Quiver.Hom W s.pt
      h : Eq (CategoryTheory.CategoryStruct.comp k s.ι) (CategoryTheory.CategoryStru …
      this : Eq (CategoryTheory.CategoryStruct.comp k (CategoryTheory.CategoryStruct …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp k (s.π.app CategoryTheory.Limits.Walk …
    -/
    rw [s.app_one_eq_ι_comp_left, this]
    /-
      🎉 no goals
    -/


/-- To check whether two maps are coequalized by both maps of a cofork, it suffices to check it for
    the second map -/
theorem Cofork.coequalizer_ext (s : Cofork f g) {W : C} {k l : s.pt ⟶ W}
    (h : Cofork.π s ≫ k = Cofork.π s ≫ l) : ∀ j : WalkingParallelPair, s.ι.app j ≫ k = s.ι.app j ≫ l
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 X Y : C
                 f g : Quiver.Hom X Y
                 s : CategoryTheory.Limits.Cofork f g
                 W : C
                 k l : Quiver.Hom s.pt W
                 h : Eq (CategoryTheory.CategoryStruct.comp s.π k) (CategoryTheory.CategoryStru …
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Walkin …
               -/
  | zero => by simp only [s.app_zero_eq_comp_π_left, Category.assoc, h]
               /-
                 🎉 no goals
               -/
  | one => h


theorem Fork.IsLimit.hom_ext {s : Fork f g} (hs : IsLimit s) {W : C} {k l : W ⟶ s.pt}
    (h : k ≫ Fork.ι s = l ≫ Fork.ι s) : k = l :=
  hs.hom_ext <| Fork.equalizer_ext _ h


theorem Cofork.IsColimit.hom_ext {s : Cofork f g} (hs : IsColimit s) {W : C} {k l : s.pt ⟶ W}
    (h : Cofork.π s ≫ k = Cofork.π s ≫ l) : k = l :=
  hs.hom_ext <| Cofork.coequalizer_ext _ h


@[reassoc (attr := simp)]
theorem Fork.IsLimit.lift_ι {s t : Fork f g} (hs : IsLimit s) : hs.lift t ≫ s.ι = t.ι :=
  hs.fac _ _


@[reassoc (attr := simp)]
theorem Cofork.IsColimit.π_desc {s t : Cofork f g} (hs : IsColimit s) : s.π ≫ hs.desc t = t.π :=
  hs.fac _ _

-- Porting note: `Fork.IsLimit.lift` was added in order to ease the port

/-- If `s` is a limit fork over `f` and `g`, then a morphism `k : W ⟶ X` satisfying
    `k ≫ f = k ≫ g` induces a morphism `l : W ⟶ s.pt` such that `l ≫ fork.ι s = k`. -/
def Fork.IsLimit.lift {s : Fork f g} (hs : IsLimit s) {W : C} (k : W ⟶ X) (h : k ≫ f = k ≫ g) :
    W ⟶ s.pt :=
  hs.lift (Fork.ofι _ h)


@[reassoc (attr := simp)]
lemma Fork.IsLimit.lift_ι' {s : Fork f g} (hs : IsLimit s) {W : C} (k : W ⟶ X) (h : k ≫ f = k ≫ g) :
    Fork.IsLimit.lift hs k h ≫ Fork.ι s = k :=
    hs.fac _ _


/-- If `s` is a limit fork over `f` and `g`, then a morphism `k : W ⟶ X` satisfying
    `k ≫ f = k ≫ g` induces a morphism `l : W ⟶ s.pt` such that `l ≫ fork.ι s = k`. -/
def Fork.IsLimit.lift' {s : Fork f g} (hs : IsLimit s) {W : C} (k : W ⟶ X) (h : k ≫ f = k ≫ g) :
    { l : W ⟶ s.pt // l ≫ Fork.ι s = k } :=
                                /-
                                  C : Type u
                                  inst✝ : CategoryTheory.Category.{v, u} C
                                  X Y : C
                                  f g : Quiver.Hom X Y
                                  s : CategoryTheory.Limits.Fork f g
                                  hs : CategoryTheory.Limits.IsLimit s
                                  W : C
                                  k : Quiver.Hom W X
                                  h : Eq (CategoryTheory.CategoryStruct.comp k f) (CategoryTheory.CategoryStruct …
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.IsLimit.l …
                                -/
  ⟨Fork.IsLimit.lift hs k h, by simp⟩
                                /-
                                  🎉 no goals
                                -/

-- Porting note: `Cofork.IsColimit.desc` was added in order to ease the port

/-- If `s` is a colimit cofork over `f` and `g`, then a morphism `k : Y ⟶ W` satisfying
    `f ≫ k = g ≫ k` induces a morphism `l : s.pt ⟶ W` such that `cofork.π s ≫ l = k`. -/
def Cofork.IsColimit.desc {s : Cofork f g} (hs : IsColimit s) {W : C} (k : Y ⟶ W)
    (h : f ≫ k = g ≫ k) : s.pt ⟶ W :=
  hs.desc (Cofork.ofπ _ h)


@[reassoc (attr := simp)]
lemma Cofork.IsColimit.π_desc' {s : Cofork f g} (hs : IsColimit s) {W : C} (k : Y ⟶ W)
    (h : f ≫ k = g ≫ k) : Cofork.π s ≫ Cofork.IsColimit.desc hs k h = k :=
  hs.fac _ _


/-- If `s` is a colimit cofork over `f` and `g`, then a morphism `k : Y ⟶ W` satisfying
    `f ≫ k = g ≫ k` induces a morphism `l : s.pt ⟶ W` such that `cofork.π s ≫ l = k`. -/
def Cofork.IsColimit.desc' {s : Cofork f g} (hs : IsColimit s) {W : C} (k : Y ⟶ W)
    (h : f ≫ k = g ≫ k) : { l : s.pt ⟶ W // Cofork.π s ≫ l = k } :=
                                    /-
                                      C : Type u
                                      inst✝ : CategoryTheory.Category.{v, u} C
                                      X Y : C
                                      f g : Quiver.Hom X Y
                                      s : CategoryTheory.Limits.Cofork f g
                                      hs : CategoryTheory.Limits.IsColimit s
                                      W : C
                                      k : Quiver.Hom Y W
                                      h : Eq (CategoryTheory.CategoryStruct.comp f k) (CategoryTheory.CategoryStruct …
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp s.π (CategoryTheory.Limits.Cofork.IsC …
                                    -/
  ⟨Cofork.IsColimit.desc hs k h, by simp⟩
                                    /-
                                      🎉 no goals
                                    -/


theorem Fork.IsLimit.existsUnique {s : Fork f g} (hs : IsLimit s) {W : C} (k : W ⟶ X)
    (h : k ≫ f = k ≫ g) : ∃! l : W ⟶ s.pt, l ≫ Fork.ι s = k :=
  ⟨hs.lift <| Fork.ofι _ h, hs.fac _ _, fun _ hm =>
    Fork.IsLimit.hom_ext hs <| hm.symm ▸ (hs.fac (Fork.ofι _ h) WalkingParallelPair.zero).symm⟩


theorem Cofork.IsColimit.existsUnique {s : Cofork f g} (hs : IsColimit s) {W : C} (k : Y ⟶ W)
    (h : f ≫ k = g ≫ k) : ∃! d : s.pt ⟶ W, Cofork.π s ≫ d = k :=
  ⟨hs.desc <| Cofork.ofπ _ h, hs.fac _ _, fun _ hm =>
    Cofork.IsColimit.hom_ext hs <| hm.symm ▸ (hs.fac (Cofork.ofπ _ h) WalkingParallelPair.one).symm⟩


/-- This is a slightly more convenient method to verify that a fork is a limit cone. It
    only asks for a proof of facts that carry any mathematical content -/
@[simps]
def Fork.IsLimit.mk (t : Fork f g) (lift : ∀ s : Fork f g, s.pt ⟶ t.pt)
    (fac : ∀ s : Fork f g, lift s ≫ Fork.ι t = Fork.ι s)
    (uniq : ∀ (s : Fork f g) (m : s.pt ⟶ t.pt) (_ : m ≫ t.ι = s.ι), m = lift s) : IsLimit t :=
  { lift
    fac := fun s j =>
      WalkingParallelPair.casesOn j (fac s) <| by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          f g : Quiver.Hom X Y
          t : CategoryTheory.Limits.Fork f g
          lift : (s : CategoryTheory.Limits.Fork f g) → Quiver.Hom s.pt t.pt
          fac : ∀ (s : CategoryTheory.Limits.Fork f g), Eq (CategoryTheory.CategoryStruc …
          uniq : ∀ (s : CategoryTheory.Limits.Fork f g) (m : Quiver.Hom s.pt t.pt), Eq ( …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f g)
          j : CategoryTheory.Limits.WalkingParallelPair
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) (t.π.app CategoryTheory.Limi …
        -/
        erw [← s.w left, ← t.w left, ← Category.assoc, fac]; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/
                            /-
                              C : Type u
                              inst✝ : CategoryTheory.Category.{v, u} C
                              X Y : C
                              f g : Quiver.Hom X Y
                              t : CategoryTheory.Limits.Fork f g
                              lift : (s : CategoryTheory.Limits.Fork f g) → Quiver.Hom s.pt t.pt
                              fac : ∀ (s : CategoryTheory.Limits.Fork f g), Eq (CategoryTheory.CategoryStruc …
                              uniq : ∀ (s : CategoryTheory.Limits.Fork f g) (m : Quiver.Hom s.pt t.pt), Eq ( …
                              s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f g)
                              m : Quiver.Hom s.pt t.pt
                              j : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
                              ⊢ Eq m (lift s)
                            -/
    uniq := fun s m j => by aesop}
                            /-
                              🎉 no goals
                            -/


/-- This is another convenient method to verify that a fork is a limit cone. It
    only asks for a proof of facts that carry any mathematical content, and allows access to the
    same `s` for all parts. -/
def Fork.IsLimit.mk' {X Y : C} {f g : X ⟶ Y} (t : Fork f g)
    (create : ∀ s : Fork f g, { l // l ≫ t.ι = s.ι ∧ ∀ {m}, m ≫ t.ι = s.ι → m = l }) : IsLimit t :=
  Fork.IsLimit.mk t (fun s => (create s).1) (fun s => (create s).2.1) fun s _ w => (create s).2.2 w


/-- This is a slightly more convenient method to verify that a cofork is a colimit cocone. It
    only asks for a proof of facts that carry any mathematical content -/
def Cofork.IsColimit.mk (t : Cofork f g) (desc : ∀ s : Cofork f g, t.pt ⟶ s.pt)
    (fac : ∀ s : Cofork f g, Cofork.π t ≫ desc s = Cofork.π s)
    (uniq : ∀ (s : Cofork f g) (m : t.pt ⟶ s.pt) (_ : t.π ≫ m = s.π), m = desc s) : IsColimit t :=
  { desc
    fac := fun s j =>
                                        /-
                                          C : Type u
                                          inst✝ : CategoryTheory.Category.{v, u} C
                                          X Y : C
                                          f g : Quiver.Hom X Y
                                          t : CategoryTheory.Limits.Cofork f g
                                          desc : (s : CategoryTheory.Limits.Cofork f g) → Quiver.Hom t.pt s.pt
                                          fac : ∀ (s : CategoryTheory.Limits.Cofork f g), Eq (CategoryTheory.CategoryStr …
                                          uniq : ∀ (s : CategoryTheory.Limits.Cofork f g) (m : Quiver.Hom t.pt s.pt), Eq …
                                          s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f g)
                                          j : CategoryTheory.Limits.WalkingParallelPair
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app CategoryTheory.Limits.Walkin …
                                        -/
      WalkingParallelPair.casesOn j (by erw [← s.w left, ← t.w left, Category.assoc, fac]; rfl)
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
        (fac s)
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 X Y : C
                 f g : Quiver.Hom X Y
                 t : CategoryTheory.Limits.Cofork f g
                 desc : (s : CategoryTheory.Limits.Cofork f g) → Quiver.Hom t.pt s.pt
                 fac : ∀ (s : CategoryTheory.Limits.Cofork f g), Eq (CategoryTheory.CategoryStr …
                 uniq : ∀ (s : CategoryTheory.Limits.Cofork f g) (m : Quiver.Hom t.pt s.pt), Eq …
                 ⊢ ∀ (s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f g) …
               -/
    uniq := by aesop }
               /-
                 🎉 no goals
               -/


/-- This is another convenient method to verify that a fork is a limit cone. It
    only asks for a proof of facts that carry any mathematical content, and allows access to the
    same `s` for all parts. -/
def Cofork.IsColimit.mk' {X Y : C} {f g : X ⟶ Y} (t : Cofork f g)
    (create : ∀ s : Cofork f g, { l : t.pt ⟶ s.pt // t.π ≫ l = s.π
                                    ∧ ∀ {m}, t.π ≫ m = s.π → m = l }) : IsColimit t :=
  Cofork.IsColimit.mk t (fun s => (create s).1) (fun s => (create s).2.1) fun s _ w =>
    (create s).2.2 w


/-- Noncomputably make a limit cone from the existence of unique factorizations. -/
noncomputable def Fork.IsLimit.ofExistsUnique {t : Fork f g}
    (hs : ∀ s : Fork f g, ∃! l : s.pt ⟶ t.pt, l ≫ Fork.ι t = Fork.ι s) : IsLimit t := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    t : CategoryTheory.Limits.Fork f g
    hs : ∀ (s : CategoryTheory.Limits.Fork f g), ExistsUnique fun l => Eq (Categor …
    ⊢ CategoryTheory.Limits.IsLimit t
  -/
  choose d hd hd' using hs
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    t : CategoryTheory.Limits.Fork f g
    d : (s : CategoryTheory.Limits.Fork f g) → Quiver.Hom s.pt t.pt
    hd : ∀ (s : CategoryTheory.Limits.Fork f g), (fun l => Eq (CategoryTheory.Cate …
    hd' : ∀ (s : CategoryTheory.Limits.Fork f g) (y : Quiver.Hom s.pt t.pt), (fun  …
    ⊢ CategoryTheory.Limits.IsLimit t
  -/
  exact Fork.IsLimit.mk _ d hd fun s m hm => hd' _ _ hm
  /-
    🎉 no goals
  -/


/-- Noncomputably make a colimit cocone from the existence of unique factorizations. -/
noncomputable def Cofork.IsColimit.ofExistsUnique {t : Cofork f g}
    (hs : ∀ s : Cofork f g, ∃! d : t.pt ⟶ s.pt, Cofork.π t ≫ d = Cofork.π s) : IsColimit t := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    t : CategoryTheory.Limits.Cofork f g
    hs : ∀ (s : CategoryTheory.Limits.Cofork f g), ExistsUnique fun d => Eq (Categ …
    ⊢ CategoryTheory.Limits.IsColimit t
  -/
  choose d hd hd' using hs
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    t : CategoryTheory.Limits.Cofork f g
    d : (s : CategoryTheory.Limits.Cofork f g) → Quiver.Hom t.pt s.pt
    hd : ∀ (s : CategoryTheory.Limits.Cofork f g), (fun d => Eq (CategoryTheory.Ca …
    hd' : ∀ (s : CategoryTheory.Limits.Cofork f g) (y : Quiver.Hom t.pt s.pt), (fu …
    ⊢ CategoryTheory.Limits.IsColimit t
  -/
  exact Cofork.IsColimit.mk _ d hd fun s m hm => hd' _ _ hm
  /-
    🎉 no goals
  -/


/--
Given a limit cone for the pair `f g : X ⟶ Y`, for any `Z`, morphisms from `Z` to its point are in
bijection with morphisms `h : Z ⟶ X` such that `h ≫ f = h ≫ g`.
Further, this bijection is natural in `Z`: see `Fork.IsLimit.homIso_natural`.
This is a special case of `IsLimit.homIso'`, often useful to construct adjunctions.
-/
@[simps]
def Fork.IsLimit.homIso {X Y : C} {f g : X ⟶ Y} {t : Fork f g} (ht : IsLimit t) (Z : C) :
    (Z ⟶ t.pt) ≃ { h : Z ⟶ X // h ≫ f = h ≫ g } where
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            X✝ Y✝ : C
                            f✝ g✝ : Quiver.Hom X✝ Y✝
                            X Y : C
                            f g : Quiver.Hom X Y
                            t : CategoryTheory.Limits.Fork f g
                            ht : CategoryTheory.Limits.IsLimit t
                            Z : C
                            k : Quiver.Hom Z t.pt
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp k …
                          -/
  toFun k := ⟨k ≫ t.ι, by simp only [Category.assoc, t.condition]⟩
                          /-
                            🎉 no goals
                          -/
  invFun h := (Fork.IsLimit.lift' ht _ h.prop).1
  left_inv _ := Fork.IsLimit.hom_ext ht (Fork.IsLimit.lift' _ _ _).prop
  right_inv _ := Subtype.ext (Fork.IsLimit.lift' ht _ _).prop


/-- The bijection of `Fork.IsLimit.homIso` is natural in `Z`. -/
theorem Fork.IsLimit.homIso_natural {X Y : C} {f g : X ⟶ Y} {t : Fork f g} (ht : IsLimit t)
    {Z Z' : C} (q : Z' ⟶ Z) (k : Z ⟶ t.pt) :
    (Fork.IsLimit.homIso ht _ (q ≫ k) : Z' ⟶ X) = q ≫ (Fork.IsLimit.homIso ht _ k : Z ⟶ X) :=
  Category.assoc _ _ _


/-- Given a colimit cocone for the pair `f g : X ⟶ Y`, for any `Z`, morphisms from the cocone point
to `Z` are in bijection with morphisms `h : Y ⟶ Z` such that `f ≫ h = g ≫ h`.
Further, this bijection is natural in `Z`: see `Cofork.IsColimit.homIso_natural`.
This is a special case of `IsColimit.homIso'`, often useful to construct adjunctions.
-/
@[simps]
def Cofork.IsColimit.homIso {X Y : C} {f g : X ⟶ Y} {t : Cofork f g} (ht : IsColimit t) (Z : C) :
    (t.pt ⟶ Z) ≃ { h : Y ⟶ Z // f ≫ h = g ≫ h } where
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            X✝ Y✝ : C
                            f✝ g✝ : Quiver.Hom X✝ Y✝
                            X Y : C
                            f g : Quiver.Hom X Y
                            t : CategoryTheory.Limits.Cofork f g
                            ht : CategoryTheory.Limits.IsColimit t
                            Z : C
                            k : Quiver.Hom t.pt Z
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                          -/
  toFun k := ⟨t.π ≫ k, by simp only [← Category.assoc, t.condition]⟩
                          /-
                            🎉 no goals
                          -/
  invFun h := (Cofork.IsColimit.desc' ht _ h.prop).1
  left_inv _ := Cofork.IsColimit.hom_ext ht (Cofork.IsColimit.desc' _ _ _).prop
  right_inv _ := Subtype.ext (Cofork.IsColimit.desc' ht _ _).prop


/-- The bijection of `Cofork.IsColimit.homIso` is natural in `Z`. -/
theorem Cofork.IsColimit.homIso_natural {X Y : C} {f g : X ⟶ Y} {t : Cofork f g} {Z Z' : C}
    (q : Z ⟶ Z') (ht : IsColimit t) (k : t.pt ⟶ Z) :
    (Cofork.IsColimit.homIso ht _ (k ≫ q) : Y ⟶ Z') =
      (Cofork.IsColimit.homIso ht _ k : Y ⟶ Z) ≫ q :=
  (Category.assoc _ _ _).symm


/-- This is a helper construction that can be useful when verifying that a category has all
    equalizers. Given `F : WalkingParallelPair ⥤ C`, which is really the same as
    `parallelPair (F.map left) (F.map right)`, and a fork on `F.map left` and `F.map right`,
    we get a cone on `F`.

    If you're thinking about using this, have a look at `hasEqualizers_of_hasLimit_parallelPair`,
    which you may find to be an easier way of achieving your goal. -/
def Cone.ofFork {F : WalkingParallelPair ⥤ C} (t : Fork (F.map left) (F.map right)) : Cone F where
  pt := t.pt
  π :=
                                              /-
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                X✝ Y : C
                                                f g : Quiver.Hom X✝ Y
                                                F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                t : CategoryTheory.Limits.Fork (F.map CategoryTheory.Limits.WalkingParallelPai …
                                                X : CategoryTheory.Limits.WalkingParallelPair
                                                ⊢ Eq ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limits.Walking …
                                              -/
    { app := fun X => t.π.app X ≫ eqToHom (by aesop)
                                              /-
                                                🎉 no goals
                                              -/
                       /-
                         C : Type u
                         inst✝ : CategoryTheory.Category.{v, u} C
                         X Y : C
                         f g : Quiver.Hom X Y
                         F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                         t : CategoryTheory.Limits.Fork (F.map CategoryTheory.Limits.WalkingParallelPai …
                         ⊢ ∀ ⦃X Y : CategoryTheory.Limits.WalkingParallelPair⦄ (f : Quiver.Hom X Y), Eq …
                       -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
      naturality := by rintro _ _ (_|_|_) <;> {dsimp; simp [t.condition]}}
                                              /-
                                                🎉 no goals
                                              -/


/-- This is a helper construction that can be useful when verifying that a category has all
    coequalizers. Given `F : WalkingParallelPair ⥤ C`, which is really the same as
    `parallelPair (F.map left) (F.map right)`, and a cofork on `F.map left` and `F.map right`,
    we get a cocone on `F`.

    If you're thinking about using this, have a look at
    `hasCoequalizers_of_hasColimit_parallelPair`, which you may find to be an easier way of
    achieving your goal. -/
def Cocone.ofCofork {F : WalkingParallelPair ⥤ C} (t : Cofork (F.map left) (F.map right)) :
    Cocone F where
  pt := t.pt
  ι :=
                                  /-
                                    C : Type u
                                    inst✝ : CategoryTheory.Category.{v, u} C
                                    X✝ Y : C
                                    f g : Quiver.Hom X✝ Y
                                    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                    t : CategoryTheory.Limits.Cofork (F.map CategoryTheory.Limits.WalkingParallelP …
                                    X : CategoryTheory.Limits.WalkingParallelPair
                                    ⊢ Eq (F.obj X) ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limi …
                                  -/
    { app := fun X => eqToHom (by aesop) ≫ t.ι.app X
                                  /-
                                    🎉 no goals
                                  -/
                       /-
                         C : Type u
                         inst✝ : CategoryTheory.Category.{v, u} C
                         X Y : C
                         f g : Quiver.Hom X Y
                         F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                         t : CategoryTheory.Limits.Cofork (F.map CategoryTheory.Limits.WalkingParallelP …
                         ⊢ ∀ ⦃X Y : CategoryTheory.Limits.WalkingParallelPair⦄ (f : Quiver.Hom X Y), Eq …
                       -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
      naturality := by rintro _ _ (_|_|_) <;> {dsimp; simp [t.condition]}}
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem Cone.ofFork_π {F : WalkingParallelPair ⥤ C} (t : Fork (F.map left) (F.map right)) (j) :
                                                      /-
                                                        C : Type u
                                                        inst✝ : CategoryTheory.Category.{v, u} C
                                                        X Y : C
                                                        f g : Quiver.Hom X Y
                                                        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                        t : CategoryTheory.Limits.Fork (F.map CategoryTheory.Limits.WalkingParallelPai …
                                                        j : CategoryTheory.Limits.WalkingParallelPair
                                                        ⊢ Eq ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limits.Walking …
                                                      -/
    (Cone.ofFork t).π.app j = t.π.app j ≫ eqToHom (by aesop) := rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem Cocone.ofCofork_ι {F : WalkingParallelPair ⥤ C} (t : Cofork (F.map left) (F.map right))
                                                    /-
                                                      C : Type u
                                                      inst✝ : CategoryTheory.Category.{v, u} C
                                                      X Y : C
                                                      f g : Quiver.Hom X Y
                                                      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                      t : CategoryTheory.Limits.Cofork (F.map CategoryTheory.Limits.WalkingParallelP …
                                                      j : CategoryTheory.Limits.WalkingParallelPair
                                                      ⊢ Eq (F.obj j) ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limi …
                                                    -/
    (j) : (Cocone.ofCofork t).ι.app j = eqToHom (by aesop) ≫ t.ι.app j := rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Given `F : WalkingParallelPair ⥤ C`, which is really the same as
    `parallelPair (F.map left) (F.map right)` and a cone on `F`, we get a fork on
    `F.map left` and `F.map right`. -/
def Fork.ofCone {F : WalkingParallelPair ⥤ C} (t : Cone F) : Fork (F.map left) (F.map right) where
  pt := t.pt
                                                 /-
                                                   C : Type u
                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                   X✝ Y : C
                                                   f g : Quiver.Hom X✝ Y
                                                   F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                   t : CategoryTheory.Limits.Cone F
                                                   X : CategoryTheory.Limits.WalkingParallelPair
                                                   ⊢ Eq (F.obj X) ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limi …
                                                 -/
  π := { app := fun X => t.π.app X ≫ eqToHom (by aesop)
                                                 /-
                                                   🎉 no goals
                                                 -/
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            X Y : C
                            f g : Quiver.Hom X Y
                            F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                            t : CategoryTheory.Limits.Cone F
                            ⊢ ∀ ⦃X Y : CategoryTheory.Limits.WalkingParallelPair⦄ (f : Quiver.Hom X Y), Eq …
                          -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
         naturality := by rintro _ _ (_|_|_) <;> {dsimp; simp}}
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- Given `F : WalkingParallelPair ⥤ C`, which is really the same as
    `parallelPair (F.map left) (F.map right)` and a cocone on `F`, we get a cofork on
    `F.map left` and `F.map right`. -/
def Cofork.ofCocone {F : WalkingParallelPair ⥤ C} (t : Cocone F) :
    Cofork (F.map left) (F.map right) where
  pt := t.pt
                                     /-
                                       C : Type u
                                       inst✝ : CategoryTheory.Category.{v, u} C
                                       X✝ Y : C
                                       f g : Quiver.Hom X✝ Y
                                       F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                       t : CategoryTheory.Limits.Cocone F
                                       X : CategoryTheory.Limits.WalkingParallelPair
                                       ⊢ Eq ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limits.Walking …
                                     -/
  ι := { app := fun X => eqToHom (by aesop) ≫ t.ι.app X
                                     /-
                                       🎉 no goals
                                     -/
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            X Y : C
                            f g : Quiver.Hom X Y
                            F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                            t : CategoryTheory.Limits.Cocone F
                            ⊢ ∀ ⦃X Y : CategoryTheory.Limits.WalkingParallelPair⦄ (f : Quiver.Hom X Y), Eq …
                          -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
         naturality := by rintro _ _ (_|_|_) <;> {dsimp; simp}}
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem Fork.ofCone_π {F : WalkingParallelPair ⥤ C} (t : Cone F) (j) :
                                                      /-
                                                        C : Type u
                                                        inst✝ : CategoryTheory.Category.{v, u} C
                                                        X Y : C
                                                        f g : Quiver.Hom X Y
                                                        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                        t : CategoryTheory.Limits.Cone F
                                                        j : CategoryTheory.Limits.WalkingParallelPair
                                                        ⊢ Eq (F.obj j) ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limi …
                                                      -/
    (Fork.ofCone t).π.app j = t.π.app j ≫ eqToHom (by aesop) := rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem Cofork.ofCocone_ι {F : WalkingParallelPair ⥤ C} (t : Cocone F) (j) :
                                              /-
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                X Y : C
                                                f g : Quiver.Hom X Y
                                                F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                t : CategoryTheory.Limits.Cocone F
                                                j : CategoryTheory.Limits.WalkingParallelPair
                                                ⊢ Eq ((CategoryTheory.Limits.parallelPair (F.map CategoryTheory.Limits.Walking …
                                              -/
    (Cofork.ofCocone t).ι.app j = eqToHom (by aesop) ≫ t.ι.app j := rfl
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem Fork.ι_postcompose {f' g' : X ⟶ Y} {α : parallelPair f g ⟶ parallelPair f' g'}
    {c : Fork f g} : Fork.ι ((Cones.postcompose α).obj c) = c.ι ≫ α.app _ :=
  rfl


@[simp]
theorem Cofork.π_precompose {f' g' : X ⟶ Y} {α : parallelPair f g ⟶ parallelPair f' g'}
    {c : Cofork f' g'} : Cofork.π ((Cocones.precompose α).obj c) = α.app _ ≫ c.π :=
  rfl


/-- Helper function for constructing morphisms between equalizer forks.
-/
@[simps]
def Fork.mkHom {s t : Fork f g} (k : s.pt ⟶ t.pt) (w : k ≫ t.ι = s.ι) : s ⟶ t where
  hom := k
  w := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      s t : CategoryTheory.Limits.Fork f g
      k : Quiver.Hom s.pt t.pt
      w : Eq (CategoryTheory.CategoryStruct.comp k t.ι) s.ι
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
    -/
    rintro ⟨_ | _⟩
      /-
        case zero
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        s t : CategoryTheory.Limits.Fork f g
        k : Quiver.Hom s.pt t.pt
        w : Eq (CategoryTheory.CategoryStruct.comp k t.ι) s.ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp k (t.π.app CategoryTheory.Limits.Walk …
      -/
    · exact w
      /-
        🎉 no goals
      -/
      /-
        case one
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        s t : CategoryTheory.Limits.Fork f g
        k : Quiver.Hom s.pt t.pt
        w : Eq (CategoryTheory.CategoryStruct.comp k t.ι) s.ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp k (t.π.app CategoryTheory.Limits.Walk …
      -/
    · simp only [Fork.app_one_eq_ι_comp_left,← Category.assoc]
      /-
        case one
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        s t : CategoryTheory.Limits.Fork f g
        k : Quiver.Hom s.pt t.pt
        w : Eq (CategoryTheory.CategoryStruct.comp k t.ι) s.ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp k …
      -/
      congr
      /-
        🎉 no goals
      -/


/-- To construct an isomorphism between forks,
it suffices to give an isomorphism between the cone points
and check that it commutes with the `ι` morphisms.
-/
@[simps]
def Fork.ext {s t : Fork f g} (i : s.pt ≅ t.pt) (w : i.hom ≫ t.ι = s.ι := by aesop_cat) :
    s ≅ t where
  hom := Fork.mkHom i.hom w
                              /-
                                C : Type u
                                inst✝ : CategoryTheory.Category.{v, u} C
                                X Y : C
                                f g : Quiver.Hom X Y
                                s t : CategoryTheory.Limits.Fork f g
                                i : CategoryTheory.Iso s.pt t.pt
                                w : autoParam (Eq (CategoryTheory.CategoryStruct.comp i.hom t.ι) s.ι) _auto✝
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp i.inv s.ι) t.ι
                              -/
  inv := Fork.mkHom i.inv (by rw [← w, Iso.inv_hom_id_assoc])
                              /-
                                🎉 no goals
                              -/


/-- Two forks of the form `ofι` are isomorphic whenever their `ι`'s are equal. -/
def ForkOfι.ext {P : C} {ι ι' : P ⟶ X} (w : ι ≫ f = ι ≫ g) (w' : ι' ≫ f = ι' ≫ g) (h : ι = ι') :
    Fork.ofι ι w ≅ Fork.ofι ι' w' :=
                            /-
                              C : Type u
                              inst✝ : CategoryTheory.Category.{v, u} C
                              X Y : C
                              f g : Quiver.Hom X Y
                              P : C
                              ι ι' : Quiver.Hom P X
                              w : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruct …
                              w' : Eq (CategoryTheory.CategoryStruct.comp ι' f) (CategoryTheory.CategoryStru …
                              h : Eq ι ι'
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
                            -/
  Fork.ext (Iso.refl _) (by simp [h])
                            /-
                              🎉 no goals
                            -/


/-- Every fork is isomorphic to one of the form `Fork.of_ι _ _`. -/
def Fork.isoForkOfι (c : Fork f g) : c ≅ Fork.ofι c.ι c.condition :=
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 X Y : C
                 f g : Quiver.Hom X Y
                 c : CategoryTheory.Limits.Fork f g
                 ⊢ CategoryTheory.Iso c.pt (CategoryTheory.Limits.Fork.ofι c.ι ⋯).pt
               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  Fork.ext (by simp only [Fork.ofι_pt, Functor.const_obj_obj]; rfl) (by simp)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/--
Given two forks with isomorphic components in such a way that the natural diagrams commute, then if
one is a limit, then the other one is as well.
-/
def Fork.isLimitOfIsos {X' Y' : C} (c : Fork f g) (hc : IsLimit c)
    {f' g' : X' ⟶ Y'} (c' : Fork f' g')
    (e₀ : X ≅ X') (e₁ : Y ≅ Y') (e : c.pt ≅ c'.pt)
    (comm₁ : e₀.hom ≫ f' = f ≫ e₁.hom := by aesop_cat)
    (comm₂ : e₀.hom ≫ g' = g ≫ e₁.hom := by aesop_cat)
    (comm₃ : e.hom ≫ c'.ι = c.ι ≫ e₀.hom := by aesop_cat) : IsLimit c' :=
  let i : parallelPair f g ≅ parallelPair f' g' := parallelPair.ext e₀ e₁ comm₁.symm comm₂.symm
  (IsLimit.equivOfNatIsoOfIso i c c' (Fork.ext e comm₃)) hc


/-- Helper function for constructing morphisms between coequalizer coforks.
-/
@[simps]
def Cofork.mkHom {s t : Cofork f g} (k : s.pt ⟶ t.pt) (w : s.π ≫ k = t.π) : s ⟶ t where
  hom := k
  w := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      s t : CategoryTheory.Limits.Cofork f g
      k : Quiver.Hom s.pt t.pt
      w : Eq (CategoryTheory.CategoryStruct.comp s.π k) t.π
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
    -/
    rintro ⟨_ | _⟩
      /-
        case zero
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        s t : CategoryTheory.Limits.Cofork f g
        k : Quiver.Hom s.pt t.pt
        w : Eq (CategoryTheory.CategoryStruct.comp s.π k) t.π
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Walkin …
      -/
    · simp [Cofork.app_zero_eq_comp_π_left, w]
      /-
        🎉 no goals
      -/
      /-
        case one
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        s t : CategoryTheory.Limits.Cofork f g
        k : Quiver.Hom s.pt t.pt
        w : Eq (CategoryTheory.CategoryStruct.comp s.π k) t.π
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Walkin …
      -/
    · exact w
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
theorem Fork.hom_comp_ι {s t : Fork f g} (f : s ⟶ t) : f.hom ≫ t.ι = s.ι := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f✝ g : Quiver.Hom X Y
    s t : CategoryTheory.Limits.Fork f✝ g
    f : Quiver.Hom s t
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom t.ι) s.ι
  -/
  cases s; cases t; cases f; aesop
                             /-
                               🎉 no goals
                             -/


@[reassoc (attr := simp)]
theorem Fork.π_comp_hom {s t : Cofork f g} (f : s ⟶ t) : s.π ≫ f.hom = t.π := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f✝ g : Quiver.Hom X Y
    s t : CategoryTheory.Limits.Cofork f✝ g
    f : Quiver.Hom s t
    ⊢ Eq (CategoryTheory.CategoryStruct.comp s.π f.hom) t.π
  -/
  cases s; cases t; cases f; aesop
                             /-
                               🎉 no goals
                             -/


/-- To construct an isomorphism between coforks,
it suffices to give an isomorphism between the cocone points
and check that it commutes with the `π` morphisms.
-/
@[simps]
def Cofork.ext {s t : Cofork f g} (i : s.pt ≅ t.pt) (w : s.π ≫ i.hom = t.π := by aesop_cat) :
    s ≅ t where
  hom := Cofork.mkHom i.hom w
                                /-
                                  C : Type u
                                  inst✝ : CategoryTheory.Category.{v, u} C
                                  X Y : C
                                  f g : Quiver.Hom X Y
                                  s t : CategoryTheory.Limits.Cofork f g
                                  i : CategoryTheory.Iso s.pt t.pt
                                  w : autoParam (Eq (CategoryTheory.CategoryStruct.comp s.π i.hom) t.π) _auto✝
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp t.π i.inv) s.π
                                -/
  inv := Cofork.mkHom i.inv (by rw [Iso.comp_inv_eq, w])
                                /-
                                  🎉 no goals
                                -/


/-- Every cofork is isomorphic to one of the form `Cofork.ofπ _ _`. -/
def Cofork.isoCoforkOfπ (c : Cofork f g) : c ≅ Cofork.ofπ c.π c.condition :=
                 /-
                   C : Type u
                   inst✝ : CategoryTheory.Category.{v, u} C
                   X Y : C
                   f g : Quiver.Hom X Y
                   c : CategoryTheory.Limits.Cofork f g
                   ⊢ CategoryTheory.Iso c.pt (CategoryTheory.Limits.Cofork.ofπ c.π ⋯).pt
                 -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  Cofork.ext (by simp only [Cofork.ofπ_pt, Functor.const_obj_obj]; rfl) (by dsimp; simp)
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- `HasEqualizer f g` represents a particular choice of limiting cone
for the parallel pair of morphisms `f` and `g`.
-/
abbrev HasEqualizer :=
  HasLimit (parallelPair f g)


/-- If an equalizer of `f` and `g` exists, we can access an arbitrary choice of such by
    saying `equalizer f g`. -/
noncomputable abbrev equalizer : C :=
  limit (parallelPair f g)


/-- If an equalizer of `f` and `g` exists, we can access the inclusion
    `equalizer f g ⟶ X` by saying `equalizer.ι f g`. -/
noncomputable abbrev equalizer.ι : equalizer f g ⟶ X :=
  limit.π (parallelPair f g) zero


/-- An equalizer cone for a parallel pair `f` and `g` -/
noncomputable abbrev equalizer.fork : Fork f g :=
  limit.cone (parallelPair f g)


@[simp]
theorem equalizer.fork_ι : (equalizer.fork f g).ι = equalizer.ι f g :=
  rfl


@[simp]
theorem equalizer.fork_π_app_zero : (equalizer.fork f g).π.app zero = equalizer.ι f g :=
  rfl


@[reassoc]
theorem equalizer.condition : equalizer.ι f g ≫ f = equalizer.ι f g ≫ g :=
  Fork.condition <| limit.cone <| parallelPair f g


/-- The equalizer built from `equalizer.ι f g` is limiting. -/
noncomputable def equalizerIsEqualizer : IsLimit (Fork.ofι (equalizer.ι f g)
    (equalizer.condition f g)) :=
                                                                  /-
                                                                    C : Type u
                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                    X Y : C
                                                                    f g : Quiver.Hom X Y
                                                                    inst✝ : CategoryTheory.Limits.HasEqualizer f g
                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
                                                                  -/
  IsLimit.ofIsoLimit (limit.isLimit _) (Fork.ext (Iso.refl _) (by aesop))
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A morphism `k : W ⟶ X` satisfying `k ≫ f = k ≫ g` factors through the equalizer of `f` and `g`
    via `equalizer.lift : W ⟶ equalizer f g`. -/
noncomputable abbrev equalizer.lift {W : C} (k : W ⟶ X) (h : k ≫ f = k ≫ g) : W ⟶ equalizer f g :=
  limit.lift (parallelPair f g) (Fork.ofι k h)


@[reassoc]
theorem equalizer.lift_ι {W : C} (k : W ⟶ X) (h : k ≫ f = k ≫ g) :
    equalizer.lift k h ≫ equalizer.ι f g = k :=
  limit.lift_π _ _


/-- A morphism `k : W ⟶ X` satisfying `k ≫ f = k ≫ g` induces a morphism `l : W ⟶ equalizer f g`
    satisfying `l ≫ equalizer.ι f g = k`. -/
noncomputable def equalizer.lift' {W : C} (k : W ⟶ X) (h : k ≫ f = k ≫ g) :
    { l : W ⟶ equalizer f g // l ≫ equalizer.ι f g = k } :=
  ⟨equalizer.lift k h, equalizer.lift_ι _ _⟩


/-- Two maps into an equalizer are equal if they are equal when composed with the equalizer map. -/
@[ext]
theorem equalizer.hom_ext {W : C} {k l : W ⟶ equalizer f g}
    (h : k ≫ equalizer.ι f g = l ≫ equalizer.ι f g) : k = l :=
  Fork.IsLimit.hom_ext (limit.isLimit _) h


theorem equalizer.existsUnique {W : C} (k : W ⟶ X) (h : k ≫ f = k ≫ g) :
    ∃! l : W ⟶ equalizer f g, l ≫ equalizer.ι f g = k :=
  Fork.IsLimit.existsUnique (limit.isLimit _) _ h


/-- An equalizer morphism is a monomorphism -/
instance equalizer.ι_mono : Mono (equalizer.ι f g) where
  right_cancellation _ _ w := equalizer.hom_ext w


/-- The equalizer morphism in any limit cone is a monomorphism. -/
theorem mono_of_isLimit_fork {c : Fork f g} (i : IsLimit c) : Mono (Fork.ι c) :=
  { right_cancellation := fun _ _ w => Fork.IsLimit.hom_ext i w }


/-- The identity determines a cone on the equalizer diagram of `f` and `g` if `f = g`. -/
def idFork (h : f = g) : Fork f g :=
  Fork.ofι (𝟙 X) <| h ▸ rfl


/-- The identity on `X` is an equalizer of `(f, g)`, if `f = g`. -/
def isLimitIdFork (h : f = g) : IsLimit (idFork h) :=
  Fork.IsLimit.mk _ (fun s => Fork.ι s) (fun _ => Category.comp_id _) fun s m h => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      h✝ : Eq f g
      s : CategoryTheory.Limits.Fork f g
      m : Quiver.Hom s.pt (CategoryTheory.Limits.idFork h✝).pt
      h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.idFork h✝) …
      ⊢ Eq m ((fun s => s.ι) s)
    -/
    convert h
    /-
      case h.e'_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      h✝ : Eq f g
      s : CategoryTheory.Limits.Fork f g
      m : Quiver.Hom s.pt (CategoryTheory.Limits.idFork h✝).pt
      h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.idFork h✝) …
      e_1✝ : Eq (Quiver.Hom s.pt (CategoryTheory.Limits.idFork h✝).pt) (Quiver.Hom s …
      ⊢ Eq m (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.idFork h✝) …
    -/
    exact (Category.comp_id _).symm
    /-
      🎉 no goals
    -/


/-- Every equalizer of `(f, g)`, where `f = g`, is an isomorphism. -/
theorem isIso_limit_cone_parallelPair_of_eq (h₀ : f = g) {c : Fork f g} (h : IsLimit c) :
    IsIso c.ι :=
  Iso.isIso_hom <| IsLimit.conePointUniqueUpToIso h <| isLimitIdFork h₀


/-- The equalizer of `(f, g)`, where `f = g`, is an isomorphism. -/
theorem equalizer.ι_of_eq [HasEqualizer f g] (h : f = g) : IsIso (equalizer.ι f g) :=
  isIso_limit_cone_parallelPair_of_eq h <| limit.isLimit _


/-- Every equalizer of `(f, f)` is an isomorphism. -/
theorem isIso_limit_cone_parallelPair_of_self {c : Fork f f} (h : IsLimit c) : IsIso c.ι :=
  isIso_limit_cone_parallelPair_of_eq rfl h


/-- An equalizer that is an epimorphism is an isomorphism. -/
theorem isIso_limit_cone_parallelPair_of_epi {c : Fork f g} (h : IsLimit c) [Epi c.ι] : IsIso c.ι :=
  isIso_limit_cone_parallelPair_of_eq ((cancel_epi _).1 (Fork.condition c)) h


/-- Two morphisms are equal if there is a fork whose inclusion is epi. -/
theorem eq_of_epi_fork_ι (t : Fork f g) [Epi (Fork.ι t)] : f = g :=
  (cancel_epi (Fork.ι t)).1 <| Fork.condition t


/-- If the equalizer of two morphisms is an epimorphism, then the two morphisms are equal. -/
theorem eq_of_epi_equalizer [HasEqualizer f g] [Epi (equalizer.ι f g)] : f = g :=
  (cancel_epi (equalizer.ι f g)).1 <| equalizer.condition _ _


instance hasEqualizer_of_self : HasEqualizer f f :=
  HasLimit.mk
    { cone := idFork rfl
      isLimit := isLimitIdFork rfl }


/-- The equalizer inclusion for `(f, f)` is an isomorphism. -/
instance equalizer.ι_of_self : IsIso (equalizer.ι f f) :=
  equalizer.ι_of_eq rfl


/-- The equalizer of a morphism with itself is isomorphic to the source. -/
noncomputable def equalizer.isoSourceOfSelf : equalizer f f ≅ X :=
  asIso (equalizer.ι f f)


@[simp]
theorem equalizer.isoSourceOfSelf_hom : (equalizer.isoSourceOfSelf f).hom = equalizer.ι f f :=
  rfl


@[simp]
theorem equalizer.isoSourceOfSelf_inv :
                                                                 /-
                                                                   C : Type u
                                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                                   X Y : C
                                                                   f g : Quiver.Hom X Y
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                                                                 -/
    (equalizer.isoSourceOfSelf f).inv = equalizer.lift (𝟙 X) (by simp) := by
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.Limits.equalizer.isoSourceOfSelf f).inv (CategoryTheory.L …
  -/
  ext
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.isoS …
  -/
  simp [equalizer.isoSourceOfSelf]
  /-
    🎉 no goals
  -/


/-- `HasCoequalizer f g` represents a particular choice of colimiting cocone
for the parallel pair of morphisms `f` and `g`.
-/
abbrev HasCoequalizer :=
  HasColimit (parallelPair f g)


/-- If a coequalizer of `f` and `g` exists, we can access an arbitrary choice of such by
    saying `coequalizer f g`. -/
noncomputable abbrev coequalizer : C :=
  colimit (parallelPair f g)


/-- If a coequalizer of `f` and `g` exists, we can access the corresponding projection by
    saying `coequalizer.π f g`. -/
noncomputable abbrev coequalizer.π : Y ⟶ coequalizer f g :=
  colimit.ι (parallelPair f g) one


/-- An arbitrary choice of coequalizer cocone for a parallel pair `f` and `g`.
-/
noncomputable abbrev coequalizer.cofork : Cofork f g :=
  colimit.cocone (parallelPair f g)


@[simp]
theorem coequalizer.cofork_π : (coequalizer.cofork f g).π = coequalizer.π f g :=
  rfl


theorem coequalizer.cofork_ι_app_one : (coequalizer.cofork f g).ι.app one = coequalizer.π f g :=
  rfl


@[reassoc]
theorem coequalizer.condition : f ≫ coequalizer.π f g = g ≫ coequalizer.π f g :=
  Cofork.condition <| colimit.cocone <| parallelPair f g


/-- The cofork built from `coequalizer.π f g` is colimiting. -/
noncomputable def coequalizerIsCoequalizer :
    IsColimit (Cofork.ofπ (coequalizer.π f g) (coequalizer.condition f g)) :=
                                                                            /-
                                                                              C : Type u
                                                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                              X Y : C
                                                                              f g : Quiver.Hom X Y
                                                                              inst✝ : CategoryTheory.Limits.HasCoequalizer f g
                                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
                                                                            -/
  IsColimit.ofIsoColimit (colimit.isColimit _) (Cofork.ext (Iso.refl _) (by aesop))
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- Any morphism `k : Y ⟶ W` satisfying `f ≫ k = g ≫ k` factors through the coequalizer of `f`
    and `g` via `coequalizer.desc : coequalizer f g ⟶ W`. -/
noncomputable abbrev coequalizer.desc {W : C} (k : Y ⟶ W) (h : f ≫ k = g ≫ k) :
    coequalizer f g ⟶ W :=
  colimit.desc (parallelPair f g) (Cofork.ofπ k h)


@[reassoc]
theorem coequalizer.π_desc {W : C} (k : Y ⟶ W) (h : f ≫ k = g ≫ k) :
    coequalizer.π f g ≫ coequalizer.desc k h = k :=
  colimit.ι_desc _ _


theorem coequalizer.π_colimMap_desc {X' Y' Z : C} (f' g' : X' ⟶ Y') [HasCoequalizer f' g']
    (p : X ⟶ X') (q : Y ⟶ Y') (wf : f ≫ q = p ≫ f') (wg : g ≫ q = p ≫ g') (h : Y' ⟶ Z)
    (wh : f' ≫ h = g' ≫ h) :
    coequalizer.π f g ≫ colimMap (parallelPairHom f g f' g' p q wf wg) ≫ coequalizer.desc h wh =
      q ≫ h := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
    X' Y' Z : C
    f' g' : Quiver.Hom X' Y'
    inst✝ : CategoryTheory.Limits.HasCoequalizer f' g'
    p : Quiver.Hom X X'
    q : Quiver.Hom Y Y'
    wf : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStruc …
    wg : Eq (CategoryTheory.CategoryStruct.comp g q) (CategoryTheory.CategoryStruc …
    h : Quiver.Hom Y' Z
    wh : Eq (CategoryTheory.CategoryStruct.comp f' h) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  rw [ι_colimMap_assoc, parallelPairHom_app_one, coequalizer.π_desc]
  /-
    🎉 no goals
  -/


/-- Any morphism `k : Y ⟶ W` satisfying `f ≫ k = g ≫ k` induces a morphism
    `l : coequalizer f g ⟶ W` satisfying `coequalizer.π ≫ g = l`. -/
noncomputable def coequalizer.desc' {W : C} (k : Y ⟶ W) (h : f ≫ k = g ≫ k) :
    { l : coequalizer f g ⟶ W // coequalizer.π f g ≫ l = k } :=
  ⟨coequalizer.desc k h, coequalizer.π_desc _ _⟩


/-- Two maps from a coequalizer are equal if they are equal when composed with the coequalizer
    map -/
@[ext]
theorem coequalizer.hom_ext {W : C} {k l : coequalizer f g ⟶ W}
    (h : coequalizer.π f g ≫ k = coequalizer.π f g ≫ l) : k = l :=
  Cofork.IsColimit.hom_ext (colimit.isColimit _) h


theorem coequalizer.existsUnique {W : C} (k : Y ⟶ W) (h : f ≫ k = g ≫ k) :
    ∃! d : coequalizer f g ⟶ W, coequalizer.π f g ≫ d = k :=
  Cofork.IsColimit.existsUnique (colimit.isColimit _) _ h


/-- A coequalizer morphism is an epimorphism -/
instance coequalizer.π_epi : Epi (coequalizer.π f g) where
  left_cancellation _ _ w := coequalizer.hom_ext w


/-- The coequalizer morphism in any colimit cocone is an epimorphism. -/
theorem epi_of_isColimit_cofork {c : Cofork f g} (i : IsColimit c) : Epi c.π :=
  { left_cancellation := fun _ _ w => Cofork.IsColimit.hom_ext i w }


/-- The identity determines a cocone on the coequalizer diagram of `f` and `g`, if `f = g`. -/
def idCofork (h : f = g) : Cofork f g :=
  Cofork.ofπ (𝟙 Y) <| h ▸ rfl


/-- The identity on `Y` is a coequalizer of `(f, g)`, where `f = g`. -/
def isColimitIdCofork (h : f = g) : IsColimit (idCofork h) :=
  Cofork.IsColimit.mk _ (fun s => Cofork.π s) (fun _ => Category.id_comp _) fun s m h => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      h✝ : Eq f g
      s : CategoryTheory.Limits.Cofork f g
      m : Quiver.Hom (CategoryTheory.Limits.idCofork h✝).pt s.pt
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.idCofork h✝) …
      ⊢ Eq m ((fun s => s.π) s)
    -/
    convert h
    /-
      case h.e'_2.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      h✝ : Eq f g
      s : CategoryTheory.Limits.Cofork f g
      m : Quiver.Hom (CategoryTheory.Limits.idCofork h✝).pt s.pt
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.idCofork h✝) …
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.idCofork h✝).pt s.pt) (Quiver.Hom …
      ⊢ Eq m (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.idCofork h✝) …
    -/
    exact (Category.id_comp _).symm
    /-
      🎉 no goals
    -/


/-- Every coequalizer of `(f, g)`, where `f = g`, is an isomorphism. -/
theorem isIso_colimit_cocone_parallelPair_of_eq (h₀ : f = g) {c : Cofork f g} (h : IsColimit c) :
    IsIso c.π :=
  Iso.isIso_hom <| IsColimit.coconePointUniqueUpToIso (isColimitIdCofork h₀) h


/-- The coequalizer of `(f, g)`, where `f = g`, is an isomorphism. -/
theorem coequalizer.π_of_eq [HasCoequalizer f g] (h : f = g) : IsIso (coequalizer.π f g) :=
  isIso_colimit_cocone_parallelPair_of_eq h <| colimit.isColimit _


/-- Every coequalizer of `(f, f)` is an isomorphism. -/
theorem isIso_colimit_cocone_parallelPair_of_self {c : Cofork f f} (h : IsColimit c) : IsIso c.π :=
  isIso_colimit_cocone_parallelPair_of_eq rfl h


/-- A coequalizer that is a monomorphism is an isomorphism. -/
theorem isIso_limit_cocone_parallelPair_of_epi {c : Cofork f g} (h : IsColimit c) [Mono c.π] :
    IsIso c.π :=
  isIso_colimit_cocone_parallelPair_of_eq ((cancel_mono _).1 (Cofork.condition c)) h


/-- Two morphisms are equal if there is a cofork whose projection is mono. -/
theorem eq_of_mono_cofork_π (t : Cofork f g) [Mono (Cofork.π t)] : f = g :=
  (cancel_mono (Cofork.π t)).1 <| Cofork.condition t


/-- If the coequalizer of two morphisms is a monomorphism, then the two morphisms are equal. -/
theorem eq_of_mono_coequalizer [HasCoequalizer f g] [Mono (coequalizer.π f g)] : f = g :=
  (cancel_mono (coequalizer.π f g)).1 <| coequalizer.condition _ _


instance hasCoequalizer_of_self : HasCoequalizer f f :=
  HasColimit.mk
    { cocone := idCofork rfl
      isColimit := isColimitIdCofork rfl }


/-- The coequalizer projection for `(f, f)` is an isomorphism. -/
instance coequalizer.π_of_self : IsIso (coequalizer.π f f) :=
  coequalizer.π_of_eq rfl


/-- The coequalizer of a morphism with itself is isomorphic to the target. -/
noncomputable def coequalizer.isoTargetOfSelf : coequalizer f f ≅ Y :=
  (asIso (coequalizer.π f f)).symm


@[simp]
theorem coequalizer.isoTargetOfSelf_hom :
                                                                     /-
                                                                       C : Type u
                                                                       inst✝ : CategoryTheory.Category.{v, u} C
                                                                       X Y : C
                                                                       f g : Quiver.Hom X Y
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                                                                     -/
    (coequalizer.isoTargetOfSelf f).hom = coequalizer.desc (𝟙 Y) (by simp) := by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.Limits.coequalizer.isoTargetOfSelf f).hom (CategoryTheory …
  -/
  ext
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  simp [coequalizer.isoTargetOfSelf]
  /-
    🎉 no goals
  -/


@[simp]
theorem coequalizer.isoTargetOfSelf_inv : (coequalizer.isoTargetOfSelf f).inv = coequalizer.π f f :=
  rfl


/-- The comparison morphism for the equalizer of `f,g`.
This is an isomorphism iff `G` preserves the equalizer of `f,g`; see
`CategoryTheory/Limits/Preserves/Shapes/Equalizers.lean`
-/
noncomputable def equalizerComparison [HasEqualizer f g] [HasEqualizer (G.map f) (G.map g)] :
    G.obj (equalizer f g) ⟶ equalizer (G.map f) (G.map g) :=
  equalizer.lift (G.map (equalizer.ι _ _))
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X Y : C
          f g : Quiver.Hom X Y
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
          inst✝ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.equaliz …
        -/
    (by simp only [← G.map_comp]; rw [equalizer.condition])
                                  /-
                                    🎉 no goals
                                  -/


@[reassoc (attr := simp)]
theorem equalizerComparison_comp_π [HasEqualizer f g] [HasEqualizer (G.map f) (G.map g)] :
    equalizerComparison f g G ≫ equalizer.ι (G.map f) (G.map g) = G.map (equalizer.ι f g) :=
  equalizer.lift_ι _ _


@[reassoc (attr := simp)]
theorem map_lift_equalizerComparison [HasEqualizer f g] [HasEqualizer (G.map f) (G.map g)] {Z : C}
    {h : Z ⟶ X} (w : h ≫ f = h ≫ g) :
    G.map (equalizer.lift h w) ≫ equalizerComparison f g G =
                                   /-
                                     C : Type u
                                     inst✝³ : CategoryTheory.Category.{v, u} C
                                     X Y : C
                                     f g : Quiver.Hom X Y
                                     D : Type u₂
                                     inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                     G : CategoryTheory.Functor C D
                                     inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
                                     inst✝ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
                                     Z : C
                                     h : Quiver.Hom Z X
                                     w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) (CategoryTheory. …
                                   -/
      equalizer.lift (G.map h) (by simp only [← G.map_comp, w]) := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
    inst✝ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    Z : C
    h : Quiver.Hom Z X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.equaliz …
  -/
  apply equalizer.hom_ext
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
    inst✝ : CategoryTheory.Limits.HasEqualizer (G.map f) (G.map g)
    Z : C
    h : Quiver.Hom Z X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [← G.map_comp]
  /-
    🎉 no goals
  -/


/-- The comparison morphism for the coequalizer of `f,g`. -/
noncomputable def coequalizerComparison [HasCoequalizer f g] [HasCoequalizer (G.map f) (G.map g)] :
    coequalizer (G.map f) (G.map g) ⟶ G.obj (coequalizer f g) :=
  coequalizer.desc (G.map (coequalizer.π _ _))
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X Y : C
          f g : Quiver.Hom X Y
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
          inst✝ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map (CategoryTheory.Limi …
        -/
    (by simp only [← G.map_comp]; rw [coequalizer.condition])
                                  /-
                                    🎉 no goals
                                  -/


@[reassoc (attr := simp)]
theorem ι_comp_coequalizerComparison [HasCoequalizer f g] [HasCoequalizer (G.map f) (G.map g)] :
    coequalizer.π _ _ ≫ coequalizerComparison f g G = G.map (coequalizer.π _ _) :=
  coequalizer.π_desc _ _


@[reassoc (attr := simp)]
theorem coequalizerComparison_map_desc [HasCoequalizer f g] [HasCoequalizer (G.map f) (G.map g)]
    {Z : C} {h : Y ⟶ Z} (w : f ≫ h = g ≫ h) :
    coequalizerComparison f g G ≫ G.map (coequalizer.desc h w) =
                                     /-
                                       C : Type u
                                       inst✝³ : CategoryTheory.Category.{v, u} C
                                       X Y : C
                                       f g : Quiver.Hom X Y
                                       D : Type u₂
                                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                       G : CategoryTheory.Functor C D
                                       inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
                                       inst✝ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
                                       Z : C
                                       h : Quiver.Hom Y Z
                                       w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) (CategoryTheory. …
                                     -/
      coequalizer.desc (G.map h) (by simp only [← G.map_comp, w]) := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
    inst✝ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    Z : C
    h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizerCom …
  -/
  apply coequalizer.hom_ext
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
    inst✝ : CategoryTheory.Limits.HasCoequalizer (G.map f) (G.map g)
    Z : C
    h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  simp [← G.map_comp]
  /-
    🎉 no goals
  -/


/-- `HasEqualizers` represents a choice of equalizer for every pair of morphisms -/
abbrev HasEqualizers :=
  HasLimitsOfShape WalkingParallelPair C


/-- `HasCoequalizers` represents a choice of coequalizer for every pair of morphisms -/
abbrev HasCoequalizers :=
  HasColimitsOfShape WalkingParallelPair C


/-- If `C` has all limits of diagrams `parallelPair f g`, then it has all equalizers -/
theorem hasEqualizers_of_hasLimit_parallelPair
    [∀ {X Y : C} {f g : X ⟶ Y}, HasLimit (parallelPair f g)] : HasEqualizers C :=
  { has_limit := fun F => hasLimitOfIso (diagramIsoParallelPair F).symm }


/-- If `C` has all colimits of diagrams `parallelPair f g`, then it has all coequalizers -/
theorem hasCoequalizers_of_hasColimit_parallelPair
    [∀ {X Y : C} {f g : X ⟶ Y}, HasColimit (parallelPair f g)] : HasCoequalizers C :=
  { has_colimit := fun F => hasColimitOfIso (diagramIsoParallelPair F) }


/-- A split mono `f` equalizes `(retraction f ≫ f)` and `(𝟙 Y)`.
Here we build the cone, and show in `isSplitMonoEqualizes` that it is a limit cone.
-/
-- @[simps (config := { rhsMd := semireducible })] Porting note: no semireducible
@[simps!]
noncomputable def coneOfIsSplitMono : Fork (𝟙 Y) (retraction f ≫ f) :=
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   X Y : C
                   f g : Quiver.Hom X Y
                   inst✝ : CategoryTheory.IsSplitMono f
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                 -/
  Fork.ofι f (by simp)
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem coneOfIsSplitMono_ι : (coneOfIsSplitMono f).ι = f :=
  rfl


/-- A split mono `f` equalizes `(retraction f ≫ f)` and `(𝟙 Y)`.
-/
noncomputable def isSplitMonoEqualizes {X Y : C} (f : X ⟶ Y) [IsSplitMono f] :
    IsLimit (coneOfIsSplitMono f) :=
  Fork.IsLimit.mk' _ fun s =>
    ⟨s.ι ≫ retraction f, by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : C
        f✝ g : Quiver.Hom X✝ Y✝
        inst✝¹ : CategoryTheory.IsSplitMono f✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitMono f
        s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.id Y) (CategoryT …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
      -/
      dsimp
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : C
        f✝ g : Quiver.Hom X✝ Y✝
        inst✝¹ : CategoryTheory.IsSplitMono f✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitMono f
        s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.id Y) (CategoryT …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
      -/
      rw [Category.assoc, ← s.condition]
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : C
        f✝ g : Quiver.Hom X✝ Y✝
        inst✝¹ : CategoryTheory.IsSplitMono f✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitMono f
        s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.id Y) (CategoryT …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι (CategoryTheory.CategoryStruct.id …
      -/
      /-
        🎉 no goals
      -/
      apply Category.comp_id, fun hm => by simp [← hm]⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- We show that the converse to `isSplitMonoEqualizes` is true:
Whenever `f` equalizes `(r ≫ f)` and `(𝟙 Y)`, then `r` is a retraction of `f`. -/
def splitMonoOfEqualizer {X Y : C} {f : X ⟶ Y} {r : Y ⟶ X} (hr : f ≫ r ≫ f = f)
    (h : IsLimit (Fork.ofι f (hr.trans (Category.comp_id _).symm : f ≫ r ≫ f = f ≫ 𝟙 Y))) :
    SplitMono f where
  retraction := r
  id := Fork.IsLimit.hom_ext h ((Category.assoc _ _ _).trans <| hr.trans (Category.id_comp _).symm)


/-- The fork obtained by postcomposing an equalizer fork with a monomorphism is an equalizer. -/
def isEqualizerCompMono {c : Fork f g} (i : IsLimit c) {Z : C} (h : Y ⟶ Z) [hm : Mono h] :
    have : Fork.ι c ≫ f ≫ h = Fork.ι c ≫ g ≫ h := by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        c : CategoryTheory.Limits.Fork f g
        i : CategoryTheory.Limits.IsLimit c
        Z : C
        h : Quiver.Hom Y Z
        hm : CategoryTheory.Mono h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c.ι (CategoryTheory.CategoryStruct.co …
      -/
      simp only [← Category.assoc]
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        c : CategoryTheory.Limits.Fork f g
        i : CategoryTheory.Limits.IsLimit c
        Z : C
        h : Quiver.Hom Y Z
        hm : CategoryTheory.Mono h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp c …
      -/
      exact congrArg (· ≫ h) c.condition
      /-
        🎉 no goals
      -/
                              /-
                                C : Type u
                                inst✝ : CategoryTheory.Category.{v, u} C
                                X Y : C
                                f g : Quiver.Hom X Y
                                c : CategoryTheory.Limits.Fork f g
                                i : CategoryTheory.Limits.IsLimit c
                                Z : C
                                h : Quiver.Hom Y Z
                                hm : CategoryTheory.Mono h
                                this : Eq (CategoryTheory.CategoryStruct.comp c.ι (CategoryTheory.CategoryStru …
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp c.ι (CategoryTheory.CategoryStruct.co …
                              -/
    IsLimit (Fork.ofι c.ι (by simp [this]) : Fork (f ≫ h) (g ≫ h)) :=
                              /-
                                🎉 no goals
                              -/
  Fork.IsLimit.mk' _ fun s =>
                                          /-
                                            C : Type u
                                            inst✝ : CategoryTheory.Category.{v, u} C
                                            X Y : C
                                            f g : Quiver.Hom X Y
                                            c : CategoryTheory.Limits.Fork f g
                                            i : CategoryTheory.Limits.IsLimit c
                                            Z : C
                                            h : Quiver.Hom Y Z
                                            hm : CategoryTheory.Mono h
                                            s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp f h) (Categ …
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι f) (CategoryTheory.CategoryStruct …
                                          -/
    let s' : Fork f g := Fork.ofι s.ι (by apply hm.right_cancellation; simp [s.condition])
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
    let l := Fork.IsLimit.lift' i s'.ι s'.condition
    ⟨l.1, l.2, fun hm => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        c : CategoryTheory.Limits.Fork f g
        i : CategoryTheory.Limits.IsLimit c
        Z : C
        h : Quiver.Hom Y Z
        hm✝ : CategoryTheory.Mono h
        s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp f h) (Categ …
        s' : CategoryTheory.Limits.Fork f g := CategoryTheory.Limits.Fork.ofι s.ι ⋯
        l : Subtype fun l => Eq (CategoryTheory.CategoryStruct.comp l c.ι) s'.ι := Cat …
        m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
        hm : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Fork.ofι …
        ⊢ Eq m✝ ↑l
      -/
      apply Fork.IsLimit.hom_ext i; rw [Fork.ι_ofι] at hm; rw [hm]; exact l.2.symm⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[instance]
theorem hasEqualizer_comp_mono [HasEqualizer f g] {Z : C} (h : Y ⟶ Z) [Mono h] :
    HasEqualizer (f ≫ h) (g ≫ h) :=
  ⟨⟨{   cone := _
        isLimit := isEqualizerCompMono (limit.isLimit _) h }⟩⟩


/-- An equalizer of an idempotent morphism and the identity is split mono. -/
@[simps]
def splitMonoOfIdempotentOfIsLimitFork {X : C} {f : X ⟶ X} (hf : f ≫ f = f) {c : Fork (𝟙 X) f}
    (i : IsLimit c) : SplitMono c.ι where
                                       /-
                                         C : Type u
                                         inst✝ : CategoryTheory.Category.{v, u} C
                                         X✝ Y : C
                                         f✝ g : Quiver.Hom X✝ Y
                                         X : C
                                         f : Quiver.Hom X X
                                         hf : Eq (CategoryTheory.CategoryStruct.comp f f) f
                                         c : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.id X) f
                                         i : CategoryTheory.Limits.IsLimit c
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id X …
                                       -/
  retraction := i.lift (Fork.ofι f (by simp [hf]))
                                       /-
                                         🎉 no goals
                                       -/
  id := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y : C
      f✝ g : Quiver.Hom X✝ Y
      X : C
      f : Quiver.Hom X X
      hf : Eq (CategoryTheory.CategoryStruct.comp f f) f
      c : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.id X) f
      i : CategoryTheory.Limits.IsLimit c
      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.ι (i.lift (CategoryTheory.Limits.Fo …
    -/
    letI := mono_of_isLimit_fork i
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y : C
      f✝ g : Quiver.Hom X✝ Y
      X : C
      f : Quiver.Hom X X
      hf : Eq (CategoryTheory.CategoryStruct.comp f f) f
      c : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.id X) f
      i : CategoryTheory.Limits.IsLimit c
      this : CategoryTheory.Mono c.ι := CategoryTheory.Limits.mono_of_isLimit_fork i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.ι (i.lift (CategoryTheory.Limits.Fo …
    -/
    rw [← cancel_mono_id c.ι, Category.assoc, Fork.IsLimit.lift_ι, Fork.ι_ofι, ← c.condition]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y : C
      f✝ g : Quiver.Hom X✝ Y
      X : C
      f : Quiver.Hom X X
      hf : Eq (CategoryTheory.CategoryStruct.comp f f) f
      c : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.id X) f
      i : CategoryTheory.Limits.IsLimit c
      this : CategoryTheory.Mono c.ι := CategoryTheory.Limits.mono_of_isLimit_fork i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.ι (CategoryTheory.CategoryStruct.id …
    -/
    exact Category.comp_id c.ι
    /-
      🎉 no goals
    -/


/-- The equalizer of an idempotent morphism and the identity is split mono. -/
noncomputable def splitMonoOfIdempotentEqualizer {X : C} {f : X ⟶ X} (hf : f ≫ f = f)
    [HasEqualizer (𝟙 X) f] : SplitMono (equalizer.ι (𝟙 X) f) :=
  splitMonoOfIdempotentOfIsLimitFork _ hf (limit.isLimit _)


/-- A split epi `f` coequalizes `(f ≫ section_ f)` and `(𝟙 X)`.
Here we build the cocone, and show in `isSplitEpiCoequalizes` that it is a colimit cocone.
-/
-- @[simps (config := { rhsMd := semireducible })] Porting note: no semireducible
@[simps!]
noncomputable def coconeOfIsSplitEpi : Cofork (𝟙 X) (f ≫ section_ f) :=
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     X Y : C
                     f g : Quiver.Hom X Y
                     inst✝ : CategoryTheory.IsSplitEpi f
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                   -/
  Cofork.ofπ f (by simp)
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem coconeOfIsSplitEpi_π : (coconeOfIsSplitEpi f).π = f :=
  rfl


/-- A split epi `f` coequalizes `(f ≫ section_ f)` and `(𝟙 X)`.
-/
noncomputable def isSplitEpiCoequalizes {X Y : C} (f : X ⟶ Y) [IsSplitEpi f] :
    IsColimit (coconeOfIsSplitEpi f) :=
  Cofork.IsColimit.mk' _ fun s =>
    ⟨section_ f ≫ s.π, by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : C
        f✝ g : Quiver.Hom X✝ Y✝
        inst✝¹ : CategoryTheory.IsSplitEpi f✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitEpi f
        s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.id X) (Categor …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coconeOfIsSpli …
      -/
      dsimp
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : C
        f✝ g : Quiver.Hom X✝ Y✝
        inst✝¹ : CategoryTheory.IsSplitEpi f✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitEpi f
        s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.id X) (Categor …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      /-
        🎉 no goals
      -/
      rw [← Category.assoc, ← s.condition, Category.id_comp], fun hm => by simp [← hm]⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- We show that the converse to `isSplitEpiEqualizes` is true:
Whenever `f` coequalizes `(f ≫ s)` and `(𝟙 X)`, then `s` is a section of `f`. -/
def splitEpiOfCoequalizer {X Y : C} {f : X ⟶ Y} {s : Y ⟶ X} (hs : f ≫ s ≫ f = f)
    (h :
      IsColimit
        (Cofork.ofπ f
          ((Category.assoc _ _ _).trans <| hs.trans (Category.id_comp f).symm :
            (f ≫ s) ≫ f = 𝟙 X ≫ f))) :
    SplitEpi f where
  section_ := s
  id := Cofork.IsColimit.hom_ext h (hs.trans (Category.comp_id _).symm)


/-- The cofork obtained by precomposing a coequalizer cofork with an epimorphism is
a coequalizer. -/
def isCoequalizerEpiComp {c : Cofork f g} (i : IsColimit c) {W : C} (h : W ⟶ X) [hm : Epi h] :
    have : (h ≫ f) ≫ Cofork.π c = (h ≫ g) ≫ Cofork.π c := by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        c : CategoryTheory.Limits.Cofork f g
        i : CategoryTheory.Limits.IsColimit c
        W : C
        h : Quiver.Hom W X
        hm : CategoryTheory.Epi h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
      -/
      simp only [Category.assoc]
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        c : CategoryTheory.Limits.Cofork f g
        i : CategoryTheory.Limits.IsColimit c
        W : C
        h : Quiver.Hom W X
        hm : CategoryTheory.Epi h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.comp …
      -/
      exact congrArg (h ≫ ·) c.condition
      /-
        🎉 no goals
      -/
    IsColimit (Cofork.ofπ c.π (this) : Cofork (h ≫ f) (h ≫ g)) :=
  Cofork.IsColimit.mk' _ fun s =>
    let s' : Cofork f g :=
                         /-
                           C : Type u
                           inst✝ : CategoryTheory.Category.{v, u} C
                           X Y : C
                           f g : Quiver.Hom X Y
                           c : CategoryTheory.Limits.Cofork f g
                           i : CategoryTheory.Limits.IsColimit c
                           W : C
                           h : Quiver.Hom W X
                           hm : CategoryTheory.Epi h
                           s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp h f) (Cat …
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp f s.π) (CategoryTheory.CategoryStruct …
                         -/
      Cofork.ofπ s.π (by apply hm.left_cancellation; simp_rw [← Category.assoc, s.condition])
                                                     /-
                                                       🎉 no goals
                                                     -/
    let l := Cofork.IsColimit.desc' i s'.π s'.condition
    ⟨l.1, l.2, fun hm => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        f g : Quiver.Hom X Y
        c : CategoryTheory.Limits.Cofork f g
        i : CategoryTheory.Limits.IsColimit c
        W : C
        h : Quiver.Hom W X
        hm✝ : CategoryTheory.Epi h
        s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp h f) (Cat …
        s' : CategoryTheory.Limits.Cofork f g := CategoryTheory.Limits.Cofork.ofπ s.π ⋯
        l : Subtype fun l => Eq (CategoryTheory.CategoryStruct.comp c.π l) s'.π := Cat …
        m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
        hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
        ⊢ Eq m✝ ↑l
      -/
      apply Cofork.IsColimit.hom_ext i; rw [Cofork.π_ofπ] at hm; rw [hm]; exact l.2.symm⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem hasCoequalizer_epi_comp [HasCoequalizer f g] {W : C} (h : W ⟶ X) [Epi h] :
    HasCoequalizer (h ≫ f) (h ≫ g) :=
  ⟨⟨{   cocone := _
        isColimit := isCoequalizerEpiComp (colimit.isColimit _) h }⟩⟩


/-- A coequalizer of an idempotent morphism and the identity is split epi. -/
@[simps]
def splitEpiOfIdempotentOfIsColimitCofork {X : C} {f : X ⟶ X} (hf : f ≫ f = f) {c : Cofork (𝟙 X) f}
    (i : IsColimit c) : SplitEpi c.π where
                                       /-
                                         C : Type u
                                         inst✝ : CategoryTheory.Category.{v, u} C
                                         X✝ Y : C
                                         f✝ g : Quiver.Hom X✝ Y
                                         X : C
                                         f : Quiver.Hom X X
                                         hf : Eq (CategoryTheory.CategoryStruct.comp f f) f
                                         c : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.id X) f
                                         i : CategoryTheory.Limits.IsColimit c
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                                       -/
  section_ := i.desc (Cofork.ofπ f (by simp [hf]))
                                       /-
                                         🎉 no goals
                                       -/
  id := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y : C
      f✝ g : Quiver.Hom X✝ Y
      X : C
      f : Quiver.Hom X X
      hf : Eq (CategoryTheory.CategoryStruct.comp f f) f
      c : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.id X) f
      i : CategoryTheory.Limits.IsColimit c
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.desc (CategoryTheory.Limits.Cofork …
    -/
    letI := epi_of_isColimit_cofork i
    rw [← cancel_epi_id c.π, ← Category.assoc, Cofork.IsColimit.π_desc, Cofork.π_ofπ, ←
      c.condition]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X✝ Y : C
      f✝ g : Quiver.Hom X✝ Y
      X : C
      f : Quiver.Hom X X
      hf : Eq (CategoryTheory.CategoryStruct.comp f f) f
      c : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.id X) f
      i : CategoryTheory.Limits.IsColimit c
      this : CategoryTheory.Epi c.π := CategoryTheory.Limits.epi_of_isColimit_cofork i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
    -/
    exact Category.id_comp _
    /-
      🎉 no goals
    -/


/-- The coequalizer of an idempotent morphism and the identity is split epi. -/
noncomputable def splitEpiOfIdempotentCoequalizer {X : C} {f : X ⟶ X} (hf : f ≫ f = f)
    [HasCoequalizer (𝟙 X) f] : SplitEpi (coequalizer.π (𝟙 X) f) :=
  splitEpiOfIdempotentOfIsColimitCofork _ hf (colimit.isColimit _)


