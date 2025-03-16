/-- A `PresheafedSpace C` is a topological space equipped with a presheaf of `C`s. -/
structure PresheafedSpace where
  carrier : TopCat
  protected presheaf : carrier.Presheaf C


instance coeCarrier : CoeOut (PresheafedSpace C) TopCat where coe X := X.carrier


instance : CoeSort (PresheafedSpace C) Type* where coe X := X.carrier

-- Porting note: the following lemma is removed because it is a syntactic tauto
/-@[simp]
theorem as_coe (X : PresheafedSpace.{w, v, u} C) : X.carrier = (X : TopCat.{w}) :=
  rfl-/

-- Porting note: removed @[simp] as the `simpVarHead` linter complains
-- @[simp]

theorem mk_coe (carrier) (presheaf) :
    (({ carrier
        presheaf } : PresheafedSpace C) : TopCat) = carrier :=
  rfl


instance (X : PresheafedSpace C) : TopologicalSpace X :=
  X.carrier.str


/-- The constant presheaf on `X` with value `Z`. -/
def const (X : TopCat) (Z : C) : PresheafedSpace C where
  carrier := X
  presheaf := (Functor.const _).obj Z


instance [Inhabited C] : Inhabited (PresheafedSpace C) :=
  ⟨const (TopCat.of PEmpty) default⟩


/-- A morphism between presheafed spaces `X` and `Y` consists of a continuous map
    `f` between the underlying topological spaces, and a (notice contravariant!) map
    from the presheaf on `Y` to the pushforward of the presheaf on `X` via `f`. -/
structure Hom (X Y : PresheafedSpace C) where
  base : (X : TopCat) ⟶ (Y : TopCat)
  c : Y.presheaf ⟶ base _* X.presheaf

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): eventually, the `ext` lemma shall be applied to terms in `X ⟶ Y`
-- rather than `Hom X Y`, this one was renamed `Hom.ext` instead of `ext`,
-- and the more practical lemma `ext` is defined just after the definition
-- of the `Category` instance

@[ext (iff := false)]
theorem Hom.ext {X Y : PresheafedSpace C} (α β : Hom X Y) (w : α.base = β.base)
                                         /-
                                           C : Type u_1
                                           inst✝ : CategoryTheory.Category.{?u.4092, u_1} C
                                           X Y : AlgebraicGeometry.PresheafedSpace C
                                           α β : X.Hom Y
                                           w : Eq α.base β.base
                                           ⊢ Eq (TopologicalSpace.Opens.map α.base).op (TopologicalSpace.Opens.map β.base …
                                         -/
    (h : α.c ≫ whiskerRight (eqToHom (by rw [w])) _ = β.c) : α = β := by
                                         /-
                                           🎉 no goals
                                         -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    α β : X.Hom Y
    w : Eq α.base β.base
    h : Eq (CategoryTheory.CategoryStruct.comp α.c (CategoryTheory.whiskerRight (C …
    ⊢ Eq α β
  -/
  rcases α with ⟨base, c⟩
  /-
    case mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    β : X.Hom Y
    base : Quiver.Hom ↑X ↑Y
    c : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base).obj X.presheaf)
    w : Eq { base := base, c := c }.base β.base
    h : Eq (CategoryTheory.CategoryStruct.comp { base := base, c := c }.c (Categor …
    ⊢ Eq { base := base, c := c } β
  -/
  rcases β with ⟨base', c'⟩
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    base : Quiver.Hom ↑X ↑Y
    c : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base).obj X.presheaf)
    base' : Quiver.Hom ↑X ↑Y
    c' : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base').obj X.preshe …
    w : Eq { base := base, c := c }.base { base := base', c := c' }.base
    h : Eq (CategoryTheory.CategoryStruct.comp { base := base, c := c }.c (Categor …
    ⊢ Eq { base := base, c := c } { base := base', c := c' }
  -/
  dsimp at w
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    base : Quiver.Hom ↑X ↑Y
    c : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base).obj X.presheaf)
    base' : Quiver.Hom ↑X ↑Y
    c' : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base').obj X.preshe …
    w : Eq base base'
    h : Eq (CategoryTheory.CategoryStruct.comp { base := base, c := c }.c (Categor …
    ⊢ Eq { base := base, c := c } { base := base', c := c' }
  -/
  subst w
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    base : Quiver.Hom ↑X ↑Y
    c c' : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base).obj X.presh …
    h : Eq (CategoryTheory.CategoryStruct.comp { base := base, c := c }.c (Categor …
    ⊢ Eq { base := base, c := c } { base := base, c := c' }
  -/
  dsimp at h
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    base : Quiver.Hom ↑X ↑Y
    c c' : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base).obj X.presh …
    h : Eq (CategoryTheory.CategoryStruct.comp c (CategoryTheory.whiskerRight (Cat …
    ⊢ Eq { base := base, c := c } { base := base, c := c' }
  -/
  erw [whiskerRight_id', comp_id] at h
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    base : Quiver.Hom ↑X ↑Y
    c c' : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base).obj X.presh …
    h : Eq c c'
    ⊢ Eq { base := base, c := c } { base := base, c := c' }
  -/
  subst h
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    base : Quiver.Hom ↑X ↑Y
    c : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base).obj X.presheaf)
    ⊢ Eq { base := base, c := c } { base := base, c := c }
  -/
  rfl
  /-
    🎉 no goals
  -/

-- TODO including `injections` would make tidy work earlier.

theorem hext {X Y : PresheafedSpace C} (α β : Hom X Y) (w : α.base = β.base) (h : HEq α.c β.c) :
    α = β := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    α β : X.Hom Y
    w : Eq α.base β.base
    h : HEq α.c β.c
    ⊢ Eq α β
  -/
  cases α
  /-
    case mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    β : X.Hom Y
    base✝ : Quiver.Hom ↑X ↑Y
    c✝ : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base✝).obj X.preshe …
    w : Eq { base := base✝, c := c✝ }.base β.base
    h : HEq { base := base✝, c := c✝ }.c β.c
    ⊢ Eq { base := base✝, c := c✝ } β
  -/
  cases β
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    base✝¹ : Quiver.Hom ↑X ↑Y
    c✝¹ : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base✝¹).obj X.pres …
    base✝ : Quiver.Hom ↑X ↑Y
    c✝ : Quiver.Hom Y.presheaf ((TopCat.Presheaf.pushforward C base✝).obj X.preshe …
    w : Eq { base := base✝¹, c := c✝¹ }.base { base := base✝, c := c✝ }.base
    h : HEq { base := base✝¹, c := c✝¹ }.c { base := base✝, c := c✝ }.c
    ⊢ Eq { base := base✝¹, c := c✝¹ } { base := base✝, c := c✝ }
  -/
  congr
  /-
    🎉 no goals
  -/

-- Porting note: `eqToHom` is no longer necessary in the definition of `c`

/-- The identity morphism of a `PresheafedSpace`. -/
def id (X : PresheafedSpace C) : Hom X X where
  base := 𝟙 (X : TopCat)
  c := 𝟙 _


instance homInhabited (X : PresheafedSpace C) : Inhabited (Hom X X) :=
  ⟨id X⟩


/-- Composition of morphisms of `PresheafedSpace`s. -/
def comp {X Y Z : PresheafedSpace C} (α : Hom X Y) (β : Hom Y Z) : Hom X Z where
  base := α.base ≫ β.base
  c := β.c ≫ (Presheaf.pushforward _ β.base).map α.c


theorem comp_c {X Y Z : PresheafedSpace C} (α : Hom X Y) (β : Hom Y Z) :
    (comp α β).c = β.c ≫ (Presheaf.pushforward _ β.base).map α.c :=
  rfl


attribute [local simp] id comp

-- Porting note: in mathlib3, `tidy` could (almost) prove the category axioms, but proofs
-- were included because `tidy` was slow. Here, `aesop_cat` succeeds reasonably quickly
-- for `comp_id` and `assoc`

/-- The category of PresheafedSpaces. Morphisms are pairs, a continuous map and a presheaf map
    from the presheaf on the target to the pushforward of the presheaf on the source. -/
instance categoryOfPresheafedSpaces : Category (PresheafedSpace C) where
  Hom := Hom
  id := id
  comp := comp
  id_comp _ := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.11869, u_1} C
      X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.11869, u_1} C
      X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq { base := CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStru …
    -/
    ext
      /-
        case w.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.11869, u_1} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        x✝¹ : Quiver.Hom X✝ Y✝
        x✝ : (CategoryTheory.forget TopCat).obj ↑X✝
        ⊢ Eq ({ base := CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStr …
      -/
    · dsimp
      /-
        case w.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.11869, u_1} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        x✝¹ : Quiver.Hom X✝ Y✝
        x✝ : (CategoryTheory.forget TopCat).obj ↑X✝
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ↑X …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case h.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.11869, u_1} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        x✝ : Quiver.Hom X✝ Y✝
        U✝ : TopologicalSpace.Opens ↑↑Y✝
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp { base := CategoryTheory.CategoryStr …
      -/
    · dsimp
      /-
        case h.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.11869, u_1} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        x✝ : Quiver.Hom X✝ Y✝
        U✝ : TopologicalSpace.Opens ↑↑Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [map_id, whiskerRight_id', assoc]
      /-
        case h.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.11869, u_1} C
        X✝ Y✝ : AlgebraicGeometry.PresheafedSpace C
        x✝ : Quiver.Hom X✝ Y✝
        U✝ : TopologicalSpace.Opens ↑↑Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (x✝.c.app { unop := U✝ }) (CategoryTh …
      -/
      erw [comp_id, comp_id]
      /-
        🎉 no goals
      -/


/-- Cast `Hom X Y` as an arrow `X ⟶ Y` of presheaves. -/
abbrev Hom.toPshHom {X Y : PresheafedSpace C} (f : Hom X Y) : X ⟶ Y := f


@[ext (iff := false)]
theorem ext {X Y : PresheafedSpace C} (α β : X ⟶ Y) (w : α.base = β.base)
                                         /-
                                           C : Type u_1
                                           inst✝ : CategoryTheory.Category.{?u.17553, u_1} C
                                           X Y : AlgebraicGeometry.PresheafedSpace C
                                           α β : Quiver.Hom X Y
                                           w : Eq α.base β.base
                                           ⊢ Eq (TopologicalSpace.Opens.map α.base).op (TopologicalSpace.Opens.map β.base …
                                         -/
    (h : α.c ≫ whiskerRight (eqToHom (by rw [w])) _ = β.c) : α = β :=
                                         /-
                                           🎉 no goals
                                         -/
  Hom.ext α β w h


@[simp]
theorem id_base (X : PresheafedSpace C) : (𝟙 X : X ⟶ X).base = 𝟙 (X : TopCat) :=
  rfl

-- Porting note: `eqToHom` is no longer needed in the statements of `id_c` and `id_c_app`

theorem id_c (X : PresheafedSpace C) :
    (𝟙 X : X ⟶ X).c = 𝟙 X.presheaf :=
  rfl


@[simp]
theorem id_c_app (X : PresheafedSpace C) (U) :
    (𝟙 X : X ⟶ X).c.app U = X.presheaf.map (𝟙 U) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    U : Opposite (TopologicalSpace.Opens ↑↑X)
    ⊢ Eq ((CategoryTheory.CategoryStruct.id X).c.app U) (X.presheaf.map (CategoryT …
  -/
  rw [id_c, map_id]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    U : Opposite (TopologicalSpace.Opens ↑↑X)
    ⊢ Eq ((CategoryTheory.CategoryStruct.id X.presheaf).app U) (CategoryTheory.Cat …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_base {X Y Z : PresheafedSpace C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).base = f.base ≫ g.base :=
  rfl


instance (X Y : PresheafedSpace C) : CoeFun (X ⟶ Y) fun _ => (↑X → ↑Y) :=
  ⟨fun f => f.base⟩

-- Porting note: removed as this is a syntactic tauto
--theorem coe_to_fun_eq {X Y : PresheafedSpace.{v, v, u} C} (f : X ⟶ Y) : (f : ↑X → ↑Y) = f.base :=
--  rfl

-- The `reassoc` attribute was added despite the LHS not being a composition of two homs,
-- for the reasons explained in the docstring.
-- Porting note: as there is no composition in the LHS it is purposely `@[reassoc, simp]` rather
-- than `@[reassoc (attr := simp)]`

/-- Sometimes rewriting with `comp_c_app` doesn't work because of dependent type issues.
In that case, `erw comp_c_app_assoc` might make progress.
The lemma `comp_c_app_assoc` is also better suited for rewrites in the opposite direction. -/
@[reassoc, simp]
theorem comp_c_app {X Y Z : PresheafedSpace C} (α : X ⟶ Y) (β : Y ⟶ Z) (U) :
    (α ≫ β).c.app U = β.c.app U ≫ α.c.app (op ((Opens.map β.base).obj (unop U))) :=
  rfl


theorem congr_app {X Y : PresheafedSpace C} {α β : X ⟶ Y} (h : α = β) (U) :
                                                        /-
                                                          C : Type u_1
                                                          inst✝ : CategoryTheory.Category.{?u.24382, u_1} C
                                                          X Y : AlgebraicGeometry.PresheafedSpace C
                                                          α β : Quiver.Hom X Y
                                                          h : Eq α β
                                                          U : Opposite (TopologicalSpace.Opens ↑↑Y)
                                                          ⊢ Eq ((TopologicalSpace.Opens.map β.base).op.obj U) ((TopologicalSpace.Opens.m …
                                                        -/
    α.c.app U = β.c.app U ≫ X.presheaf.map (eqToHom (by subst h; rfl)) := by
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    α β : Quiver.Hom X Y
    h : Eq α β
    U : Opposite (TopologicalSpace.Opens ↑↑Y)
    ⊢ Eq (α.c.app U) (CategoryTheory.CategoryStruct.comp (β.c.app U) (X.presheaf.m …
  -/
  subst h
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : AlgebraicGeometry.PresheafedSpace C
    α : Quiver.Hom X Y
    U : Opposite (TopologicalSpace.Opens ↑↑Y)
    ⊢ Eq (α.c.app U) (CategoryTheory.CategoryStruct.comp (α.c.app U) (X.presheaf.m …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The forgetful functor from `PresheafedSpace` to `TopCat`. -/
@[simps]
def forget : PresheafedSpace C ⥤ TopCat where
  obj X := (X : TopCat)
  map f := f.base


/-- An isomorphism of `PresheafedSpace`s is a homeomorphism of the underlying space, and a
natural transformation between the sheaves.
-/
@[simps hom inv]
def isoOfComponents (H : X.1 ≅ Y.1) (α : H.hom _* X.2 ≅ Y.2) : X ≅ Y where
  hom :=
    { base := H.hom
      c := α.inv }
  inv :=
    { base := H.inv
      c := Presheaf.toPushforwardOfIso H α.hom }
                   /-
                     C : Type u_1
                     inst✝ : CategoryTheory.Category.{?u.27259, u_1} C
                     X Y : AlgebraicGeometry.PresheafedSpace C
                     H : CategoryTheory.Iso ↑X ↑Y
                     α : CategoryTheory.Iso ((TopCat.Presheaf.pushforward C H.hom).obj X.presheaf)  …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { base := H.hom, c := α.inv } { base  …
                   -/
                           /-
                             🎉 no goals
                           -/
  hom_inv_id := by ext <;> simp
                           /-
                             🎉 no goals
                           -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.27259, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso ↑X ↑Y
      α : CategoryTheory.Iso ((TopCat.Presheaf.pushforward C H.hom).obj X.presheaf)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { base := H.inv, c := TopCat.Presheaf …
    -/
    ext
      /-
        case w.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.27259, u_1} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        H : CategoryTheory.Iso ↑X ↑Y
        α : CategoryTheory.Iso ((TopCat.Presheaf.pushforward C H.hom).obj X.presheaf)  …
        x✝ : (CategoryTheory.forget TopCat).obj ↑Y
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp { base := H.inv, c := TopCat.Preshea …
      -/
    · dsimp
      /-
        case w.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.27259, u_1} C
        X Y : AlgebraicGeometry.PresheafedSpace C
        H : CategoryTheory.Iso ↑X ↑Y
        α : CategoryTheory.Iso ((TopCat.Presheaf.pushforward C H.hom).obj X.presheaf)  …
        x✝ : (CategoryTheory.forget TopCat).obj ↑Y
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp H.inv H.hom) x✝) ((CategoryTheory.Ca …
      -/
      rw [H.inv_hom_id]
      /-
        🎉 no goals
      -/
    /-
      case h.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.27259, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso ↑X ↑Y
      α : CategoryTheory.Iso ((TopCat.Presheaf.pushforward C H.hom).obj X.presheaf)  …
      U✝ : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    dsimp
    /-
      case h.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.27259, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso ↑X ↑Y
      α : CategoryTheory.Iso ((TopCat.Presheaf.pushforward C H.hom).obj X.presheaf)  …
      U✝ : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Presheaf.toPushforwardOfIso_app, assoc, ← α.hom.naturality]
    /-
      case h.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.27259, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso ↑X ↑Y
      α : CategoryTheory.Iso ((TopCat.Presheaf.pushforward C H.hom).obj X.presheaf)  …
      U✝ : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.inv.app { unop := U✝ }) (CategoryT …
    -/
    simp only [eqToHom_map, eqToHom_app, eqToHom_trans_assoc, eqToHom_refl, id_comp]
    /-
      case h.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.27259, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso ↑X ↑Y
      α : CategoryTheory.Iso ((TopCat.Presheaf.pushforward C H.hom).obj X.presheaf)  …
      U✝ : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.inv.app { unop := U✝ }) (α.hom.app …
    -/
    apply Iso.inv_hom_id_app
    /-
      🎉 no goals
    -/


/-- Isomorphic `PresheafedSpace`s have naturally isomorphic presheaves. -/
@[simps]
def sheafIsoOfIso (H : X ≅ Y) : Y.2 ≅ H.hom.base _* X.2 where
  hom := H.hom.c
  inv := Presheaf.pushforwardToOfIso ((forget _).mapIso H).symm H.inv.c
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp H.hom.c (TopCat.Presheaf.pushforwardT …
    -/
    ext U
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp H.hom.c (TopCat.Presheaf.pushforward …
    -/
    rw [NatTrans.comp_app]
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.hom.c.app { unop := U }) ((TopCat. …
    -/
    simpa using congr_arg (fun f => f ≫ eqToHom _) (congr_app H.inv_hom_id (op U))
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.Presheaf.pushforwardToOfIso ( …
    -/
    ext U
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (TopCat.Presheaf.pushforwardToOfIso  …
    -/
    dsimp
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.Presheaf.pushforwardToOfIso  …
    -/
    rw [NatTrans.id_app]
    simp only [Presheaf.pushforwardToOfIso_app, Iso.symm_inv, mapIso_hom, forget_map,
      Iso.symm_hom, mapIso_inv, eqToHom_map, assoc]
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.inv.c.app { unop := (TopologicalSp …
    -/
    have eq₁ := congr_app H.hom_inv_id (op ((Opens.map H.hom.base).obj U))
    have eq₂ := H.hom.c.naturality (eqToHom (congr_obj (congr_arg Opens.map
      ((forget C).congr_map H.inv_hom_id.symm)) U)).op
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      eq₁ : Eq ((CategoryTheory.CategoryStruct.comp H.hom H.inv).c.app { unop := (To …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory.e …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.inv.c.app { unop := (TopologicalSp …
    -/
    rw [id_c, NatTrans.id_app, id_comp, eqToHom_map, comp_c_app] at eq₁
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp (H.inv.c.app { unop := (Topologic …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory.e …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.inv.c.app { unop := (TopologicalSp …
    -/
    rw [eqToHom_op, eqToHom_map] at eq₂
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp (H.inv.c.app { unop := (Topologic …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (H.hom …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.inv.c.app { unop := (TopologicalSp …
    -/
    erw [eq₂, reassoc_of% eq₁]
    /-
      case w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.33734, u_1} C
      X Y : AlgebraicGeometry.PresheafedSpace C
      H : CategoryTheory.Iso X Y
      U : TopologicalSpace.Opens ↑↑Y
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp (H.inv.c.app { unop := (Topologic …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (H.hom …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (((TopCat. …
    -/
    simp
    /-
      🎉 no goals
    -/


instance base_isIso_of_iso (f : X ⟶ Y) [IsIso f] : IsIso f.base :=
  ((forget _).mapIso (asIso f)).isIso_hom


instance c_isIso_of_iso (f : X ⟶ Y) [IsIso f] : IsIso f.c :=
  (sheafIsoOfIso (asIso f)).isIso_hom


/-- This could be used in conjunction with `CategoryTheory.NatIso.isIso_of_isIso_app`. -/
theorem isIso_of_components (f : X ⟶ Y) [IsIso f.base] [IsIso f.c] : IsIso f :=
  (isoOfComponents (asIso f.base) (asIso f.c).symm).isIso_hom


/-- The restriction of a presheafed space along an open embedding into the space.
-/
@[simps]
def restrict {U : TopCat} (X : PresheafedSpace C) {f : U ⟶ (X : TopCat)}
    (h : IsOpenEmbedding f) : PresheafedSpace C where
  carrier := U
  presheaf := h.isOpenMap.functor.op ⋙ X.presheaf


/-- The map from the restriction of a presheafed space.
-/
@[simps]
def ofRestrict {U : TopCat} (X : PresheafedSpace C) {f : U ⟶ (X : TopCat)}
    (h : IsOpenEmbedding f) : X.restrict h ⟶ X where
  base := f
  c :=
    { app := fun V => X.presheaf.map (h.isOpenMap.adjunction.counit.app V.unop).op
      naturality := fun U V f =>
        show _ = _ ≫ X.presheaf.map _ by
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.54286, u_1} C
            U✝ : TopCat
            X : AlgebraicGeometry.PresheafedSpace C
            f✝ : Quiver.Hom U✝ ↑X
            h : Topology.IsOpenEmbedding ⇑f✝
            U V : Opposite (TopologicalSpace.Opens ↑↑X)
            f : Quiver.Hom U V
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map f) ((fun V => X.presh …
          -/
          rw [← map_comp, ← map_comp]
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.54286, u_1} C
            U✝ : TopCat
            X : AlgebraicGeometry.PresheafedSpace C
            f✝ : Quiver.Hom U✝ ↑X
            h : Topology.IsOpenEmbedding ⇑f✝
            U V : Opposite (TopologicalSpace.Opens ↑↑X)
            f : Quiver.Hom U V
            ⊢ Eq (X.presheaf.map (CategoryTheory.CategoryStruct.comp f (⋯.adjunction.couni …
          -/
          rfl }
          /-
            🎉 no goals
          -/


instance ofRestrict_mono {U : TopCat} (X : PresheafedSpace C) (f : U ⟶ X.1)
    (hf : IsOpenEmbedding f) : Mono (X.ofRestrict hf) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    U : TopCat
    X : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom U ↑X
    hf : Topology.IsOpenEmbedding ⇑f
    ⊢ CategoryTheory.Mono (X.ofRestrict hf)
  -/
  haveI : Mono f := (TopCat.mono_iff_injective _).mpr hf.injective
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    U : TopCat
    X : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom U ↑X
    hf : Topology.IsOpenEmbedding ⇑f
    this : CategoryTheory.Mono f
    ⊢ CategoryTheory.Mono (X.ofRestrict hf)
  -/
  constructor
  /-
    case right_cancellation
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    U : TopCat
    X : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom U ↑X
    hf : Topology.IsOpenEmbedding ⇑f
    this : CategoryTheory.Mono f
    ⊢ ∀ {Z : AlgebraicGeometry.PresheafedSpace C} (g h : Quiver.Hom Z (X.restrict  …
  -/
  intro Z g₁ g₂ eq
  /-
    case right_cancellation
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    U : TopCat
    X : AlgebraicGeometry.PresheafedSpace C
    f : Quiver.Hom U ↑X
    hf : Topology.IsOpenEmbedding ⇑f
    this : CategoryTheory.Mono f
    Z : AlgebraicGeometry.PresheafedSpace C
    g₁ g₂ : Quiver.Hom Z (X.restrict hf)
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
    ⊢ Eq g₁ g₂
  -/
  ext1
    /-
      case right_cancellation.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      U : TopCat
      X : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom U ↑X
      hf : Topology.IsOpenEmbedding ⇑f
      this : CategoryTheory.Mono f
      Z : AlgebraicGeometry.PresheafedSpace C
      g₁ g₂ : Quiver.Hom Z (X.restrict hf)
      eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
      ⊢ Eq g₁.base g₂.base
    -/
  · have := congr_arg PresheafedSpace.Hom.base eq
    /-
      case right_cancellation.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      U : TopCat
      X : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom U ↑X
      hf : Topology.IsOpenEmbedding ⇑f
      this✝ : CategoryTheory.Mono f
      Z : AlgebraicGeometry.PresheafedSpace C
      g₁ g₂ : Quiver.Hom Z (X.restrict hf)
      eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
      this : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)).base (Cate …
      ⊢ Eq g₁.base g₂.base
    -/
    simp only [PresheafedSpace.comp_base, PresheafedSpace.ofRestrict_base] at this
    /-
      case right_cancellation.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      U : TopCat
      X : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom U ↑X
      hf : Topology.IsOpenEmbedding ⇑f
      this✝ : CategoryTheory.Mono f
      Z : AlgebraicGeometry.PresheafedSpace C
      g₁ g₂ : Quiver.Hom Z (X.restrict hf)
      eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
      this : Eq (CategoryTheory.CategoryStruct.comp g₁.base f) (CategoryTheory.Categ …
      ⊢ Eq g₁.base g₂.base
    -/
    rw [cancel_mono] at this
    /-
      case right_cancellation.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      U : TopCat
      X : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom U ↑X
      hf : Topology.IsOpenEmbedding ⇑f
      this✝ : CategoryTheory.Mono f
      Z : AlgebraicGeometry.PresheafedSpace C
      g₁ g₂ : Quiver.Hom Z (X.restrict hf)
      eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
      this : Eq g₁.base g₂.base
      ⊢ Eq g₁.base g₂.base
    -/
    exact this
    /-
      🎉 no goals
    -/
    /-
      case right_cancellation.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      U : TopCat
      X : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom U ↑X
      hf : Topology.IsOpenEmbedding ⇑f
      this : CategoryTheory.Mono f
      Z : AlgebraicGeometry.PresheafedSpace C
      g₁ g₂ : Quiver.Hom Z (X.restrict hf)
      eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁.c (CategoryTheory.whiskerRight (Ca …
    -/
  · ext V
    have hV : (Opens.map (X.ofRestrict hf).base).obj (hf.isOpenMap.functor.obj V) = V := by
      ext1
      exact Set.preimage_image_eq _ hf.injective
    haveI :
      IsIso (hf.isOpenMap.adjunction.counit.app (unop (op (hf.isOpenMap.functor.obj V)))) :=
        NatIso.isIso_app_of_isIso
          (whiskerLeft hf.isOpenMap.functor hf.isOpenMap.adjunction.counit) V
    /-
      case right_cancellation.h.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      U : TopCat
      X : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom U ↑X
      hf : Topology.IsOpenEmbedding ⇑f
      this✝ : CategoryTheory.Mono f
      Z : AlgebraicGeometry.PresheafedSpace C
      g₁ g₂ : Quiver.Hom Z (X.restrict hf)
      eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
      V : TopologicalSpace.Opens ↑↑(X.restrict hf)
      hV : Eq ((TopologicalSpace.Opens.map (X.ofRestrict hf).base).obj (⋯.functor.ob …
      this : CategoryTheory.IsIso (⋯.adjunction.counit.app (Opposite.unop { unop :=  …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp g₁.c (CategoryTheory.whiskerRight (C …
    -/
    have := PresheafedSpace.congr_app eq (op (hf.isOpenMap.functor.obj V))
    rw [PresheafedSpace.comp_c_app, PresheafedSpace.comp_c_app,
      PresheafedSpace.ofRestrict_c_app, Category.assoc, cancel_epi] at this
    have h : _ ≫ _ = _ ≫ _ ≫ _ :=
      congr_arg (fun f => (X.restrict hf).presheaf.map (eqToHom hV).op ≫ f) this
    /-
      case right_cancellation.h.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      U : TopCat
      X : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom U ↑X
      hf : Topology.IsOpenEmbedding ⇑f
      this✝¹ : CategoryTheory.Mono f
      Z : AlgebraicGeometry.PresheafedSpace C
      g₁ g₂ : Quiver.Hom Z (X.restrict hf)
      eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
      V : TopologicalSpace.Opens ↑↑(X.restrict hf)
      hV : Eq ((TopologicalSpace.Opens.map (X.ofRestrict hf).base).obj (⋯.functor.ob …
      this✝ : CategoryTheory.IsIso (⋯.adjunction.counit.app (Opposite.unop { unop := …
      this : Eq (g₁.c.app { unop := (TopologicalSpace.Opens.map (X.ofRestrict hf).ba …
      h : Eq (CategoryTheory.CategoryStruct.comp ((X.restrict hf).presheaf.map (Cate …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp g₁.c (CategoryTheory.whiskerRight (C …
    -/
    simp only [g₁.c.naturality, g₂.c.naturality_assoc] at h
    simp only [eqToHom_op, eqToHom_unop, eqToHom_map, eqToHom_trans,
      ← IsIso.comp_inv_eq, inv_eqToHom, Category.assoc] at h
    /-
      case right_cancellation.h.w
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_3, u_1} C
      U : TopCat
      X : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom U ↑X
      hf : Topology.IsOpenEmbedding ⇑f
      this✝¹ : CategoryTheory.Mono f
      Z : AlgebraicGeometry.PresheafedSpace C
      g₁ g₂ : Quiver.Hom Z (X.restrict hf)
      eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (X.ofRestrict hf)) (CategoryThe …
      V : TopologicalSpace.Opens ↑↑(X.restrict hf)
      hV : Eq ((TopologicalSpace.Opens.map (X.ofRestrict hf).base).obj (⋯.functor.ob …
      this✝ : CategoryTheory.IsIso (⋯.adjunction.counit.app (Opposite.unop { unop := …
      this : Eq (g₁.c.app { unop := (TopologicalSpace.Opens.map (X.ofRestrict hf).ba …
      h : Eq (CategoryTheory.CategoryStruct.comp (g₁.c.app { unop := V }) (CategoryT …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp g₁.c (CategoryTheory.whiskerRight (C …
    -/
    simpa using h
    /-
      🎉 no goals
    -/


theorem restrict_top_presheaf (X : PresheafedSpace C) :
    (X.restrict (Opens.isOpenEmbedding ⊤)).presheaf =
      (Opens.inclusionTopIso X.carrier).inv _* X.presheaf := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    ⊢ Eq (X.restrict ⋯).presheaf ((TopCat.Presheaf.pushforward C (TopologicalSpace …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    ⊢ Eq (⋯.functor.op.comp X.presheaf) ((TopCat.Presheaf.pushforward C (Topologic …
  -/
  rw [Opens.inclusion'_top_functor X.carrier]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    ⊢ Eq ((TopologicalSpace.Opens.map (TopologicalSpace.Opens.inclusionTopIso ↑X). …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ofRestrict_top_c (X : PresheafedSpace C) :
    (X.ofRestrict (Opens.isOpenEmbedding ⊤)).c =
      eqToHom
        (by
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.69380, u_1} C
            X : AlgebraicGeometry.PresheafedSpace C
            ⊢ Eq X.presheaf ((TopCat.Presheaf.pushforward C (X.ofRestrict ⋯).base).obj (X. …
          -/
          rw [restrict_top_presheaf, ← Presheaf.Pushforward.comp_eq]
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.69380, u_1} C
            X : AlgebraicGeometry.PresheafedSpace C
            ⊢ Eq X.presheaf ((TopCat.Presheaf.pushforward C (CategoryTheory.CategoryStruct …
          -/
          erw [Iso.inv_hom_id]
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.69380, u_1} C
            X : AlgebraicGeometry.PresheafedSpace C
            ⊢ Eq X.presheaf ((TopCat.Presheaf.pushforward C (CategoryTheory.CategoryStruct …
          -/
          rw [Presheaf.id_pushforward]
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.69380, u_1} C
            X : AlgebraicGeometry.PresheafedSpace C
            ⊢ Eq X.presheaf ((CategoryTheory.Functor.id (TopCat.Presheaf C ↑X)).obj X.pres …
          -/
          dsimp) := by
          /-
            🎉 no goals
          -/
  /- another approach would be to prove the left hand side
       is a natural isomorphism, but I encountered a universe
       issue when `apply NatIso.isIso_of_isIso_app`. -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    ⊢ Eq (X.ofRestrict ⋯).c (CategoryTheory.eqToHom ⋯)
  -/
  ext
  /-
    case w
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    U✝ : TopologicalSpace.Opens ↑↑X
    ⊢ Eq ((X.ofRestrict ⋯).c.app { unop := U✝ }) ((CategoryTheory.eqToHom ⋯).app { …
  -/
  dsimp [ofRestrict]
  /-
    case w
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    U✝ : TopologicalSpace.Opens ↑↑X
    ⊢ Eq (X.presheaf.map (⋯.adjunction.counit.app U✝).op) ((CategoryTheory.eqToHom …
  -/
  erw [eqToHom_map, eqToHom_app]
  /-
    case w.p
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : AlgebraicGeometry.PresheafedSpace C
    U✝ : TopologicalSpace.Opens ↑↑X
    ⊢ Eq { unop := U✝ } { unop := ⋯.functor.obj ((TopologicalSpace.Opens.map Top.t …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The map to the restriction of a presheafed space along the canonical inclusion from the top
subspace.
-/
@[simps]
def toRestrictTop (X : PresheafedSpace C) : X ⟶ X.restrict (Opens.isOpenEmbedding ⊤) where
  base := (Opens.inclusionTopIso X.carrier).inv
  c := eqToHom (restrict_top_presheaf X)


/-- The isomorphism from the restriction to the top subspace.
-/
@[simps]
def restrictTopIso (X : PresheafedSpace C) : X.restrict (Opens.isOpenEmbedding ⊤) ≅ X where
  hom := X.ofRestrict _
  inv := X.toRestrictTop
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.74253, u_1} C
      X : AlgebraicGeometry.PresheafedSpace C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ofRestrict ⋯) X.toRestrictTop) (Ca …
    -/
    ext
      /-
        case w.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.74253, u_1} C
        X : AlgebraicGeometry.PresheafedSpace C
        x✝ : (CategoryTheory.forget TopCat).obj ↑(X.restrict ⋯)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (X.ofRestrict ⋯) X.toRestrictTop).ba …
      -/
    · rfl
      /-
        🎉 no goals
      -/
    · erw [comp_c, toRestrictTop_c, whiskerRight_id',
        comp_id, ofRestrict_top_c, eqToHom_map, eqToHom_trans, eqToHom_refl]
      /-
        case h.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.74253, u_1} C
        X : AlgebraicGeometry.PresheafedSpace C
        U✝ : TopologicalSpace.Opens ↑↑(X.restrict ⋯)
        ⊢ Eq ((CategoryTheory.CategoryStruct.id (X.restrict ⋯).presheaf).app { unop := …
      -/
      rfl
      /-
        🎉 no goals
      -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.74253, u_1} C
      X : AlgebraicGeometry.PresheafedSpace C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp X.toRestrictTop (X.ofRestrict ⋯)) (Ca …
    -/
    ext
      /-
        case w.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.74253, u_1} C
        X : AlgebraicGeometry.PresheafedSpace C
        x✝ : (CategoryTheory.forget TopCat).obj ↑X
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp X.toRestrictTop (X.ofRestrict ⋯)).ba …
      -/
    · rfl
      /-
        🎉 no goals
      -/
    · erw [comp_c, ofRestrict_top_c, toRestrictTop_c, eqToHom_map, whiskerRight_id', comp_id,
        eqToHom_trans, eqToHom_refl]
      /-
        case h.w
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.74253, u_1} C
        X : AlgebraicGeometry.PresheafedSpace C
        U✝ : TopologicalSpace.Opens ↑↑X
        ⊢ Eq ((CategoryTheory.CategoryStruct.id X.presheaf).app { unop := U✝ }) ((Cate …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- The global sections, notated Gamma.
-/
@[simps]
def Γ : (PresheafedSpace C)ᵒᵖ ⥤ C where
  obj X := (unop X).presheaf.obj (op ⊤)
  map f := f.unop.c.app (op ⊤)


theorem Γ_obj_op (X : PresheafedSpace C) : Γ.obj (op X) = X.presheaf.obj (op ⊤) :=
  rfl


theorem Γ_map_op {X Y : PresheafedSpace C} (f : X ⟶ Y) : Γ.map f.op = f.c.app (op ⊤) :=
  rfl


/-- We can apply a functor `F : C ⥤ D` to the values of the presheaf in any `PresheafedSpace C`,
    giving a functor `PresheafedSpace C ⥤ PresheafedSpace D` -/
def mapPresheaf (F : C ⥤ D) : PresheafedSpace C ⥤ PresheafedSpace D where
  obj X :=
    { carrier := X.carrier
      presheaf := X.presheaf ⋙ F }
  map f :=
    { base := f.base
      c := whiskerRight f.c F }
  -- Porting note: these proofs were automatic in mathlib3
  map_id X := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.86791, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.86798, u_2} D
      F : CategoryTheory.Functor C D
      X : AlgebraicGeometry.PresheafedSpace C
      ⊢ Eq ({ obj := fun X => { carrier := ↑X, presheaf := CategoryTheory.Functor.co …
    -/
    ext U
      /-
        case w.w
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.86791, u_1} C
        D : Type u_2
        inst✝ : CategoryTheory.Category.{?u.86798, u_2} D
        F : CategoryTheory.Functor C D
        X : AlgebraicGeometry.PresheafedSpace C
        U : (CategoryTheory.forget TopCat).obj ↑({ obj := fun X => { carrier := ↑X, pr …
        ⊢ Eq (({ obj := fun X => { carrier := ↑X, presheaf := CategoryTheory.Functor.c …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h.w
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.86791, u_1} C
        D : Type u_2
        inst✝ : CategoryTheory.Category.{?u.86798, u_2} D
        F : CategoryTheory.Functor C D
        X : AlgebraicGeometry.PresheafedSpace C
        U : TopologicalSpace.Opens ↑↑({ obj := fun X => { carrier := ↑X, presheaf := C …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ({ obj := fun X => { carrier := ↑X,  …
      -/
    · simp
      /-
        🎉 no goals
      -/
  map_comp f g := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.86791, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.86798, u_2} D
      F : CategoryTheory.Functor C D
      X✝ Y✝ Z✝ : AlgebraicGeometry.PresheafedSpace C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => { carrier := ↑X, presheaf := CategoryTheory.Functor.co …
    -/
    ext U
      /-
        case w.w
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.86791, u_1} C
        D : Type u_2
        inst✝ : CategoryTheory.Category.{?u.86798, u_2} D
        F : CategoryTheory.Functor C D
        X✝ Y✝ Z✝ : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        U : (CategoryTheory.forget TopCat).obj ↑({ obj := fun X => { carrier := ↑X, pr …
        ⊢ Eq (({ obj := fun X => { carrier := ↑X, presheaf := CategoryTheory.Functor.c …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h.w
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.86791, u_1} C
        D : Type u_2
        inst✝ : CategoryTheory.Category.{?u.86798, u_2} D
        F : CategoryTheory.Functor C D
        X✝ Y✝ Z✝ : AlgebraicGeometry.PresheafedSpace C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        U : TopologicalSpace.Opens ↑↑({ obj := fun X => { carrier := ↑X, presheaf := C …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ({ obj := fun X => { carrier := ↑X,  …
      -/
    · simp
      /-
        🎉 no goals
      -/


@[simp]
theorem mapPresheaf_obj_X (F : C ⥤ D) (X : PresheafedSpace C) :
    (F.mapPresheaf.obj X : TopCat) = (X : TopCat) :=
  rfl


@[simp]
theorem mapPresheaf_obj_presheaf (F : C ⥤ D) (X : PresheafedSpace C) :
    (F.mapPresheaf.obj X).presheaf = X.presheaf ⋙ F :=
  rfl


@[simp]
theorem mapPresheaf_map_f (F : C ⥤ D) {X Y : PresheafedSpace C} (f : X ⟶ Y) :
    (F.mapPresheaf.map f).base = f.base :=
  rfl


@[simp]
theorem mapPresheaf_map_c (F : C ⥤ D) {X Y : PresheafedSpace C} (f : X ⟶ Y) :
    (F.mapPresheaf.map f).c = whiskerRight f.c F :=
  rfl


/-- A natural transformation induces a natural transformation between the `map_presheaf` functors.
-/
def onPresheaf {F G : C ⥤ D} (α : F ⟶ G) : G.mapPresheaf ⟶ F.mapPresheaf where
  app X :=
    { base := 𝟙 _
      c := whiskerLeft X.presheaf α ≫ eqToHom (Presheaf.Pushforward.id_eq _).symm }

-- TODO Assemble the last two constructions into a functor
--   `(C ⥤ D) ⥤ (PresheafedSpace C ⥤ PresheafedSpace D)`

