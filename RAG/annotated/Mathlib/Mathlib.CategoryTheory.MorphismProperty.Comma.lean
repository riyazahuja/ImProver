lemma costructuredArrow_iso_iff (P : MorphismProperty T) [P.RespectsIso]
    {L : A ⥤ T} {X : T} {f g : CostructuredArrow L X} (e : f ≅ g) :
    P f.hom ↔ P g.hom :=
  P.comma_iso_iff e


lemma over_iso_iff (P : MorphismProperty T) [P.RespectsIso] {X : T} {f g : Over X} (e : f ≅ g) :
    P f.hom ↔ P g.hom :=
  P.comma_iso_iff e


/-- `P.Comma L R Q W` is the subcategory of `Comma L R` consisting of
objects `X : Comma L R` where `X.hom` satisfies `P`. The morphisms are given by
morphisms in `Comma L R` where the left one satisfies `Q` and the right one satisfies `W`. -/
@[ext]
protected structure Comma (Q : MorphismProperty A) (W : MorphismProperty B) extends Comma L R where
  prop : P toComma.hom


/-- A morphism in `P.Comma L R Q W` is a morphism in `Comma L R` where the left
hom satisfies `Q` and the right one satisfies `W`. -/
@[ext]
structure Hom (X Y : P.Comma L R Q W) extends CommaMorphism X.toComma Y.toComma where
  prop_hom_left : Q toCommaMorphism.left
  prop_hom_right : W toCommaMorphism.right


/-- The underlying morphism of objects in `Comma L R`. -/
abbrev Hom.hom {X Y : P.Comma L R Q W} (f : Comma.Hom X Y) : X.toComma ⟶ Y.toComma :=
  f.toCommaMorphism


@[simp, nolint simpVarHead]
lemma Hom.hom_mk {X Y : P.Comma L R Q W}
    (f : CommaMorphism X.toComma Y.toComma) (hf) (hg) :
  Comma.Hom.hom ⟨f, hf, hg⟩ = f := rfl


lemma Hom.hom_left {X Y : P.Comma L R Q W} (f : Comma.Hom X Y) : f.hom.left = f.left := rfl


lemma Hom.hom_right {X Y : P.Comma L R Q W} (f : Comma.Hom X Y) : f.hom.right = f.right := rfl


/-- See Note [custom simps projection] -/
def Hom.Simps.hom {X Y : P.Comma L R Q W} (f : X.Hom Y) :
    X.toComma ⟶ Y.toComma :=
  f.hom


/-- The identity morphism of an object in `P.Comma L R Q W`. -/
@[simps]
def id [Q.ContainsIdentities] [W.ContainsIdentities] (X : P.Comma L R Q W) : Comma.Hom X X where
  left := 𝟙 X.left
  prop_hom_left := Q.id_mem X.toComma.left
  prop_hom_right := W.id_mem X.toComma.right


/-- Composition of morphisms in `P.Comma L R Q W`. -/
@[simps]
def Hom.comp [Q.IsStableUnderComposition] [W.IsStableUnderComposition] {X Y Z : P.Comma L R Q W}
    (f : Comma.Hom X Y) (g : Comma.Hom Y Z) :
    Comma.Hom X Z where
  left := f.left ≫ g.left
  right := f.right ≫ g.right
  prop_hom_left := Q.comp_mem _ _ f.prop_hom_left g.prop_hom_left
  prop_hom_right := W.comp_mem _ _ f.prop_hom_right g.prop_hom_right


variable (L R P Q W) in
instance : Category (P.Comma L R Q W) where
  Hom X Y := X.Hom Y
  id X := X.id
  comp f g := f.comp g


lemma toCommaMorphism_eq_hom {X Y : P.Comma L R Q W} (f : X ⟶ Y) : f.toCommaMorphism = f.hom := rfl


/-- Alternative `ext` lemma for `Comma.Hom`. -/
@[ext]
lemma Hom.ext' {X Y : P.Comma L R Q W} {f g : X ⟶ Y} (h : f.hom = g.hom) :
    f = g := Comma.Hom.ext
  (congrArg CommaMorphism.left h)
  (congrArg CommaMorphism.right h)


@[simp]
lemma id_hom (X : P.Comma L R Q W) : (𝟙 X : X ⟶ X).hom = 𝟙 X.toComma := rfl


@[simp]
lemma comp_hom {X Y Z : P.Comma L R Q W} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).hom = f.hom ≫ g.hom := rfl


@[reassoc]
lemma comp_left {X Y Z : P.Comma L R Q W} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).left = f.left ≫ g.left := rfl


@[reassoc]
lemma comp_right {X Y Z : P.Comma L R Q W} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).right = f.right ≫ g.right := rfl


/-- If `i` is an isomorphism in `Comma L R`, it is also a morphism in `P.Comma L R Q W`. -/
@[simps hom]
def homFromCommaOfIsIso [Q.RespectsIso] [W.RespectsIso] {X Y : P.Comma L R Q W}
    (i : X.toComma ⟶ Y.toComma) [IsIso i] :
    X ⟶ Y where
  __ := i
  prop_hom_left := Q.of_isIso i.left
  prop_hom_right := W.of_isIso i.right


instance [Q.RespectsIso] [W.RespectsIso] {X Y : P.Comma L R Q W} (i : X.toComma ⟶ Y.toComma)
    [IsIso i] : IsIso (homFromCommaOfIsIso i) := by
  /-
    A : Type u_1
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝⁴ : Q.IsMultiplicative
    inst✝³ : W.IsMultiplicative
    inst✝² : Q.RespectsIso
    inst✝¹ : W.RespectsIso
    X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
    i : Quiver.Hom X.toComma Y.toComma
    inst✝ : CategoryTheory.IsIso i
    ⊢ CategoryTheory.IsIso (CategoryTheory.MorphismProperty.Comma.homFromCommaOfIs …
  -/
  constructor
  /-
    case out
    A : Type u_1
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝⁴ : Q.IsMultiplicative
    inst✝³ : W.IsMultiplicative
    inst✝² : Q.RespectsIso
    inst✝¹ : W.RespectsIso
    X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
    i : Quiver.Hom X.toComma Y.toComma
    inst✝ : CategoryTheory.IsIso i
    ⊢ Exists fun inv => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  use homFromCommaOfIsIso (inv i)
  /-
    case h
    A : Type u_1
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝⁴ : Q.IsMultiplicative
    inst✝³ : W.IsMultiplicative
    inst✝² : Q.RespectsIso
    inst✝¹ : W.RespectsIso
    X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
    i : Quiver.Hom X.toComma Y.toComma
    inst✝ : CategoryTheory.IsIso i
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty …
  -/
                              /-
                                🎉 no goals
                              -/
  constructor <;> ext : 1 <;> simp
                              /-
                                🎉 no goals
                              -/


/-- Any isomorphism between objects of `P.Comma L R Q W` in `Comma L R` is also an isomorphism
in `P.Comma L R Q W`.  -/
@[simps]
def isoFromComma [Q.RespectsIso] [W.RespectsIso] {X Y : P.Comma L R Q W}
    (i : X.toComma ≅ Y.toComma) : X ≅ Y where
  hom := homFromCommaOfIsIso i.hom
  inv := homFromCommaOfIsIso i.inv


/-- Constructor for isomorphisms in `P.Comma L R Q W` from isomorphisms of the left and right
components and naturality in the forward direction. -/
@[simps!]
def isoMk [Q.RespectsIso] [W.RespectsIso] {X Y : P.Comma L R Q W} (l : X.left ≅ Y.left)
    (r : X.right ≅ Y.right) (h : L.map l.hom ≫ Y.hom = X.hom ≫ R.map r.hom := by aesop_cat) :
    X ≅ Y :=
  isoFromComma (CategoryTheory.Comma.isoMk l r h)


/-- The forgetful functor. -/
@[simps]
def forget : P.Comma L R Q W ⥤ Comma L R where
  obj X := X.toComma
  map f := f.hom


instance : (forget L R P Q W).Faithful where
  map_injective := Comma.Hom.ext'


instance {X Y : P.Comma L R Q W} (f : X ⟶ Y) [IsIso f] : IsIso f.hom :=
  (forget L R P Q W).map_isIso f


lemma hom_homFromCommaOfIsIso [Q.RespectsIso] [W.RespectsIso] {X Y : P.Comma L R Q W}
    (i : X ⟶ Y) [IsIso i.hom] :
    homFromCommaOfIsIso i.hom = i :=
  rfl


lemma inv_hom {X Y : P.Comma L R Q W} (f : X ⟶ Y) [IsIso f] : (inv f).hom = inv f.hom := by
  /-
    A : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝³ : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝² : Q.IsMultiplicative
    inst✝¹ : W.IsMultiplicative
    X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.MorphismProperty.Comma.Hom.hom (CategoryTheory.inv f)) (C …
  -/
  apply IsIso.eq_inv_of_hom_inv_id
  /-
    case hom_inv_id
    A : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝³ : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝² : Q.IsMultiplicative
    inst✝¹ : W.IsMultiplicative
    X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.Comm …
  -/
  rw [← comp_hom, IsIso.hom_inv_id, id_hom]
  /-
    🎉 no goals
  -/


instance [Q.RespectsIso] [W.RespectsIso] : (forget L R P Q W).ReflectsIsomorphisms where
  reflects f hf := by
    /-
      A : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_4, u_1} A
      B : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_5, u_2} B
      T : Type u_3
      inst✝⁴ : CategoryTheory.Category.{u_6, u_3} T
      L : CategoryTheory.Functor A T
      R : CategoryTheory.Functor B T
      P : CategoryTheory.MorphismProperty T
      Q : CategoryTheory.MorphismProperty A
      W : CategoryTheory.MorphismProperty B
      inst✝³ : Q.IsMultiplicative
      inst✝² : W.IsMultiplicative
      inst✝¹ : Q.RespectsIso
      inst✝ : W.RespectsIso
      A✝ B✝ : CategoryTheory.MorphismProperty.Comma L R P Q W
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso ((CategoryTheory.MorphismProperty.Comma.forget L R P …
      ⊢ CategoryTheory.IsIso f
    -/
    simp only [forget_obj, forget_map] at hf
    /-
      A : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_4, u_1} A
      B : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_5, u_2} B
      T : Type u_3
      inst✝⁴ : CategoryTheory.Category.{u_6, u_3} T
      L : CategoryTheory.Functor A T
      R : CategoryTheory.Functor B T
      P : CategoryTheory.MorphismProperty T
      Q : CategoryTheory.MorphismProperty A
      W : CategoryTheory.MorphismProperty B
      inst✝³ : Q.IsMultiplicative
      inst✝² : W.IsMultiplicative
      inst✝¹ : Q.RespectsIso
      inst✝ : W.RespectsIso
      A✝ B✝ : CategoryTheory.MorphismProperty.Comma L R P Q W
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso (CategoryTheory.MorphismProperty.Comma.Hom.hom f)
      ⊢ CategoryTheory.IsIso f
    -/
    rw [← hom_homFromCommaOfIsIso f]
    /-
      A : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_4, u_1} A
      B : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_5, u_2} B
      T : Type u_3
      inst✝⁴ : CategoryTheory.Category.{u_6, u_3} T
      L : CategoryTheory.Functor A T
      R : CategoryTheory.Functor B T
      P : CategoryTheory.MorphismProperty T
      Q : CategoryTheory.MorphismProperty A
      W : CategoryTheory.MorphismProperty B
      inst✝³ : Q.IsMultiplicative
      inst✝² : W.IsMultiplicative
      inst✝¹ : Q.RespectsIso
      inst✝ : W.RespectsIso
      A✝ B✝ : CategoryTheory.MorphismProperty.Comma L R P Q W
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso (CategoryTheory.MorphismProperty.Comma.Hom.hom f)
      ⊢ CategoryTheory.IsIso (CategoryTheory.MorphismProperty.Comma.homFromCommaOfIs …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- The forgetful functor from the full subcategory of `Comma L R` defined by `P` is
fully faithful. -/
def forgetFullyFaithful : (forget L R P ⊤ ⊤).FullyFaithful where
  preimage {X Y} f := ⟨f, trivial, trivial⟩


instance : (forget L R P ⊤ ⊤).Full :=
  Functor.FullyFaithful.full (forgetFullyFaithful L R P)


@[simp]
lemma eqToHom_left {X Y : P.Comma L R Q W} (h : X = Y) :
                                   /-
                                     A : Type u_1
                                     inst✝⁴ : CategoryTheory.Category.{?u.83063, u_1} A
                                     B : Type u_2
                                     inst✝³ : CategoryTheory.Category.{?u.83070, u_2} B
                                     T : Type u_3
                                     inst✝² : CategoryTheory.Category.{?u.83077, u_3} T
                                     L : CategoryTheory.Functor A T
                                     R : CategoryTheory.Functor B T
                                     P : CategoryTheory.MorphismProperty T
                                     Q : CategoryTheory.MorphismProperty A
                                     W : CategoryTheory.MorphismProperty B
                                     inst✝¹ : Q.IsMultiplicative
                                     inst✝ : W.IsMultiplicative
                                     X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
                                     h : Eq X Y
                                     ⊢ Eq X.left Y.left
                                   -/
    (eqToHom h).left = eqToHom (by rw [h]) := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    A : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝² : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝¹ : Q.IsMultiplicative
    inst✝ : W.IsMultiplicative
    X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
    h : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom h).left (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    A : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝² : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝¹ : Q.IsMultiplicative
    inst✝ : W.IsMultiplicative
    X : CategoryTheory.MorphismProperty.Comma L R P Q W
    ⊢ Eq (CategoryTheory.eqToHom ⋯).left (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma eqToHom_right {X Y : P.Comma L R Q W} (h : X = Y) :
                                    /-
                                      A : Type u_1
                                      inst✝⁴ : CategoryTheory.Category.{?u.84453, u_1} A
                                      B : Type u_2
                                      inst✝³ : CategoryTheory.Category.{?u.84460, u_2} B
                                      T : Type u_3
                                      inst✝² : CategoryTheory.Category.{?u.84467, u_3} T
                                      L : CategoryTheory.Functor A T
                                      R : CategoryTheory.Functor B T
                                      P : CategoryTheory.MorphismProperty T
                                      Q : CategoryTheory.MorphismProperty A
                                      W : CategoryTheory.MorphismProperty B
                                      inst✝¹ : Q.IsMultiplicative
                                      inst✝ : W.IsMultiplicative
                                      X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
                                      h : Eq X Y
                                      ⊢ Eq X.right Y.right
                                    -/
    (eqToHom h).right = eqToHom (by rw [h]) := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    A : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝² : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝¹ : Q.IsMultiplicative
    inst✝ : W.IsMultiplicative
    X Y : CategoryTheory.MorphismProperty.Comma L R P Q W
    h : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom h).right (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    A : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} A
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_2} B
    T : Type u_3
    inst✝² : CategoryTheory.Category.{u_6, u_3} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    P : CategoryTheory.MorphismProperty T
    Q : CategoryTheory.MorphismProperty A
    W : CategoryTheory.MorphismProperty B
    inst✝¹ : Q.IsMultiplicative
    inst✝ : W.IsMultiplicative
    X : CategoryTheory.MorphismProperty.Comma L R P Q W
    ⊢ Eq (CategoryTheory.eqToHom ⋯).right (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Lift a functor `F : C ⥤ Comma L R` to the subcategory `P.Comma L R Q W` under
suitable assumptions on `F`. -/
@[simps obj_toComma map_hom]
def lift {C : Type*} [Category C] (F : C ⥤ Comma L R)
    (hP : ∀ X, P (F.obj X).hom)
    (hQ : ∀ {X Y} (f : X ⟶ Y), Q (F.map f).left)
    (hW : ∀ {X Y} (f : X ⟶ Y), W (F.map f).right) :
    C ⥤ P.Comma L R Q W where
  obj X :=
    { __ := F.obj X
      prop := hP X }
  map {X Y} f :=
    { __ := F.map f
      prop_hom_left := hQ f
      prop_hom_right := hW f }


variable (R) in
/-- A natural transformation `L₁ ⟶ L₂` induces a functor `P.Comma L₂ R Q W ⥤ P.Comma L₁ R Q W`. -/
@[simps!]
def mapLeft (l : L₁ ⟶ L₂) (hl : ∀ X : P.Comma L₂ R Q W, P (l.app X.left ≫ X.hom)) :
    P.Comma L₂ R Q W ⥤ P.Comma L₁ R Q W :=
  lift (forget _ _ _ _ _ ⋙ CategoryTheory.Comma.mapLeft R l) hl
    (fun f ↦ f.prop_hom_left) (fun f ↦ f.prop_hom_right)


variable (L) in
/-- A natural transformation `R₁ ⟶ R₂` induces a functor `P.Comma L R₁ Q W ⥤ P.Comma L R₂ Q W`. -/
@[simps!]
def mapRight (r : R₁ ⟶ R₂) (hr : ∀ X : P.Comma L R₁ Q W, P (X.hom ≫ r.app X.right)) :
    P.Comma L R₁ Q W ⥤ P.Comma L R₂ Q W :=
  lift (forget _ _ _ _ _ ⋙ CategoryTheory.Comma.mapRight L r) hr
    (fun f ↦ f.prop_hom_left) (fun f ↦ f.prop_hom_right)


/-- Given a morphism property `P` on a category `C` and an object `X : C`, this is the
subcategory of `Over X` defined by `P` where morphisms satisfy `Q`. -/
protected abbrev Over : Type _ :=
  P.Comma (Functor.id T) (Functor.fromPUnit.{0} X) Q ⊤


/-- The forgetful functor from the full subcategory of `Over X` defined by `P` to `Over X`. -/
protected abbrev Over.forget : P.Over Q X ⥤ Over X :=
  Comma.forget (Functor.id T) (Functor.fromPUnit.{0} X) P Q ⊤


instance : (Over.forget P ⊤ X).Faithful := inferInstanceAs <| (Comma.forget _ _ _ _ _).Faithful

instance : (Over.forget P ⊤ X).Full := inferInstanceAs <| (Comma.forget _ _ _ _ _).Full


/-- Construct a morphism in `P.Over Q X` from a morphism in `Over.X`. -/
@[simps hom]
def Over.Hom.mk {A B : P.Over Q X} (f : A.toComma ⟶ B.toComma) (hf : Q f.left) : A ⟶ B where
  __ := f
  prop_hom_left := hf
  prop_hom_right := trivial


variable (Q) in
/-- Make an object of `P.Over Q X` from a morphism `f : A ⟶ X` and a proof of `P f`. -/
@[simps hom left]
protected def Over.mk {A : T} (f : A ⟶ X) (hf : P f) : P.Over Q X where
  left := A
  right := ⟨⟨⟩⟩
  hom := f
  prop := hf


/-- Make a morphism in `P.Over Q X` from a morphism in `T` with compatibilities. -/
@[simps hom]
protected def Over.homMk {A B : P.Over Q X} (f : A.left ⟶ B.left)
    (w : f ≫ B.hom = A.hom := by aesop_cat) (hf : Q f := by trivial) : A ⟶ B where
  __ := CategoryTheory.Over.homMk f w
  prop_hom_left := hf
  prop_hom_right := trivial


/-- Make an isomorphism in `P.Over Q X` from an isomorphism in `T` with compatibilities. -/
@[simps! hom_left inv_left]
protected def Over.isoMk [Q.RespectsIso] {A B : P.Over Q X} (f : A.left ≅ B.left)
    (w : f.hom ≫ B.hom = A.hom := by aesop_cat) : A ≅ B :=
  /-
    T : Type u_1
    inst✝² : CategoryTheory.Category.{?u.119354, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    X : T
    inst✝¹ : Q.IsMultiplicative
    inst✝ : Q.RespectsIso
    A B : P.Over Q X
    f : CategoryTheory.Iso A.left B.left
    w : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom B.hom) A.hom) _auto✝
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id T).map f. …
  -/
  Comma.isoMk f (Discrete.eqToIso' rfl)
  /-
    🎉 no goals
  -/


@[ext]
lemma Over.Hom.ext {A B : P.Over Q X} {f g : A ⟶ B} (h : f.left = g.left) : f = g := by
  /-
    T : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    X : T
    inst✝ : Q.IsMultiplicative
    A B : P.Over Q X
    f g : Quiver.Hom A B
    h : Eq f.left g.left
    ⊢ Eq f g
  -/
  ext
    /-
      case h.h₁
      T : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} T
      P Q : CategoryTheory.MorphismProperty T
      X : T
      inst✝ : Q.IsMultiplicative
      A B : P.Over Q X
      f g : Quiver.Hom A B
      h : Eq f.left g.left
      ⊢ Eq (CategoryTheory.MorphismProperty.Comma.Hom.hom f).left (CategoryTheory.Mo …
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case h.h₂
      T : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} T
      P Q : CategoryTheory.MorphismProperty T
      X : T
      inst✝ : Q.IsMultiplicative
      A B : P.Over Q X
      f g : Quiver.Hom A B
      h : Eq f.left g.left
      ⊢ Eq (CategoryTheory.MorphismProperty.Comma.Hom.hom f).right (CategoryTheory.M …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[reassoc]
lemma Over.w {A B : P.Over Q X} (f : A ⟶ B) :
    f.left ≫ B.hom = A.hom := by
  /-
    T : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    X : T
    inst✝ : Q.IsMultiplicative
    A B : P.Over Q X
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.left B.hom) A.hom
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a morphism property `P` on a category `C` and an object `X : C`, this is the
subcategory of `Under X` defined by `P` where morphisms satisfy `Q`. -/
protected abbrev Under : Type _ :=
  P.Comma (Functor.fromPUnit.{0} X) (Functor.id T) ⊤ Q


/-- The forgetful functor from the full subcategory of `Under X` defined by `P` to `Under X`. -/
protected abbrev Under.forget : P.Under Q X ⥤ Under X :=
  Comma.forget (Functor.fromPUnit.{0} X) (Functor.id T) P ⊤ Q


instance : (Under.forget P ⊤ X).Faithful := inferInstanceAs <| (Comma.forget _ _ _ _ _).Faithful

instance : (Under.forget P ⊤ X).Full := inferInstanceAs <| (Comma.forget _ _ _ _ _).Full


/-- Construct a morphism in `P.Under Q X` from a morphism in `Under.X`. -/
@[simps hom]
def Under.Hom.mk {A B : P.Under Q X} (f : A.toComma ⟶ B.toComma) (hf : Q f.right) : A ⟶ B where
  __ := f
  prop_hom_left := trivial
  prop_hom_right := hf


variable (Q) in
/-- Make an object of `P.Under Q X` from a morphism `f : A ⟶ X` and a proof of `P f`. -/
@[simps hom left]
protected def Under.mk {A : T} (f : X ⟶ A) (hf : P f) : P.Under Q X where
  left := ⟨⟨⟩⟩
  right := A
  hom := f
  prop := hf


/-- Make a morphism in `P.Under Q X` from a morphism in `T` with compatibilities. -/
@[simps hom]
protected def Under.homMk {A B : P.Under Q X} (f : A.right ⟶ B.right)
    (w : A.hom ≫ f = B.hom := by aesop_cat) (hf : Q f := by trivial) : A ⟶ B where
  __ := CategoryTheory.Under.homMk f w
  prop_hom_left := trivial
  prop_hom_right := hf


/-- Make an isomorphism in `P.Under Q X` from an isomorphism in `T` with compatibilities. -/
@[simps! hom_right inv_right]
protected def Under.isoMk [Q.RespectsIso] {A B : P.Under Q X} (f : A.right ≅ B.right)
    (w : A.hom ≫ f.hom = B.hom := by aesop_cat) : A ≅ B :=
  /-
    T : Type u_1
    inst✝² : CategoryTheory.Category.{?u.141892, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    X : T
    inst✝¹ : Q.IsMultiplicative
    inst✝ : Q.RespectsIso
    A B : P.Under Q X
    f : CategoryTheory.Iso A.right B.right
    w : autoParam (Eq (CategoryTheory.CategoryStruct.comp A.hom f.hom) B.hom) _auto✝
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.fromPUnit X) …
  -/
  Comma.isoMk (Discrete.eqToIso' rfl) f
  /-
    🎉 no goals
  -/


@[ext]
lemma Under.Hom.ext {A B : P.Under Q X} {f g : A ⟶ B} (h : f.right = g.right) : f = g := by
  /-
    T : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    X : T
    inst✝ : Q.IsMultiplicative
    A B : P.Under Q X
    f g : Quiver.Hom A B
    h : Eq f.right g.right
    ⊢ Eq f g
  -/
  ext
    /-
      case h.h₁
      T : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} T
      P Q : CategoryTheory.MorphismProperty T
      X : T
      inst✝ : Q.IsMultiplicative
      A B : P.Under Q X
      f g : Quiver.Hom A B
      h : Eq f.right g.right
      ⊢ Eq (CategoryTheory.MorphismProperty.Comma.Hom.hom f).left (CategoryTheory.Mo …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.h₂
      T : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} T
      P Q : CategoryTheory.MorphismProperty T
      X : T
      inst✝ : Q.IsMultiplicative
      A B : P.Under Q X
      f g : Quiver.Hom A B
      h : Eq f.right g.right
      ⊢ Eq (CategoryTheory.MorphismProperty.Comma.Hom.hom f).right (CategoryTheory.M …
    -/
  · exact h
    /-
      🎉 no goals
    -/


@[reassoc]
lemma Under.w {A B : P.Under Q X} (f : A ⟶ B) :
    A.hom ≫ f.right = B.hom := by
  /-
    T : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} T
    P Q : CategoryTheory.MorphismProperty T
    X : T
    inst✝ : Q.IsMultiplicative
    A B : P.Under Q X
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp A.hom f.right) B.hom
  -/
  simp
  /-
    🎉 no goals
  -/


