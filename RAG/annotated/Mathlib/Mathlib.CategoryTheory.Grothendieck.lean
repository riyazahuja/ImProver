/--
The Grothendieck construction (often written as `∫ F` in mathematics) for a functor `F : C ⥤ Cat`
gives a category whose
* objects `X` consist of `X.base : C` and `X.fiber : F.obj base`
* morphisms `f : X ⟶ Y` consist of
  `base : X.base ⟶ Y.base` and
  `f.fiber : (F.map base).obj X.fiber ⟶ Y.fiber`
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): no such linter yet
-- @[nolint has_nonempty_instance]
structure Grothendieck where
  /-- The underlying object in `C` -/
  base : C
  /-- The object in the fiber of the base object. -/
  fiber : F.obj base


/-- A morphism in the Grothendieck category `F : C ⥤ Cat` consists of
`base : X.base ⟶ Y.base` and `f.fiber : (F.map base).obj X.fiber ⟶ Y.fiber`.
-/
structure Hom (X Y : Grothendieck F) where
  /-- The morphism between base objects. -/
  base : X.base ⟶ Y.base
  /-- The morphism from the pushforward to the source fiber object to the target fiber object. -/
  fiber : (F.map base).obj X.fiber ⟶ Y.fiber


@[ext (iff := false)]
theorem ext {X Y : Grothendieck F} (f g : Hom X Y) (w_base : f.base = g.base)
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             D : Type u₁
                             inst✝ : CategoryTheory.Category.{v₁, u₁} D
                             F : CategoryTheory.Functor C CategoryTheory.Cat
                             X Y : CategoryTheory.Grothendieck F
                             f g : X.Hom Y
                             w_base : Eq f.base g.base
                             ⊢ Eq ((F.map g.base).obj X.fiber) ((F.map f.base).obj X.fiber)
                           -/
    (w_fiber : eqToHom (by rw [w_base]) ≫ f.fiber = g.fiber) : f = g := by
                           /-
                             🎉 no goals
                           -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X Y : CategoryTheory.Grothendieck F
    f g : X.Hom Y
    w_base : Eq f.base g.base
    w_fiber : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) f. …
    ⊢ Eq f g
  -/
  cases f; cases g
  /-
    case mk.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X Y : CategoryTheory.Grothendieck F
    base✝¹ : Quiver.Hom X.base Y.base
    fiber✝¹ : Quiver.Hom ((F.map base✝¹).obj X.fiber) Y.fiber
    base✝ : Quiver.Hom X.base Y.base
    fiber✝ : Quiver.Hom ((F.map base✝).obj X.fiber) Y.fiber
    w_base : Eq { base := base✝¹, fiber := fiber✝¹ }.base { base := base✝, fiber : …
    w_fiber : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) {  …
    ⊢ Eq { base := base✝¹, fiber := fiber✝¹ } { base := base✝, fiber := fiber✝ }
  -/
  congr
  /-
    case mk.mk.h.e_7
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X Y : CategoryTheory.Grothendieck F
    base✝¹ : Quiver.Hom X.base Y.base
    fiber✝¹ : Quiver.Hom ((F.map base✝¹).obj X.fiber) Y.fiber
    base✝ : Quiver.Hom X.base Y.base
    fiber✝ : Quiver.Hom ((F.map base✝).obj X.fiber) Y.fiber
    w_base : Eq { base := base✝¹, fiber := fiber✝¹ }.base { base := base✝, fiber : …
    w_fiber : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) {  …
    ⊢ HEq fiber✝¹ fiber✝
  -/
  dsimp at w_base
  /-
    case mk.mk.h.e_7
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X Y : CategoryTheory.Grothendieck F
    base✝¹ : Quiver.Hom X.base Y.base
    fiber✝¹ : Quiver.Hom ((F.map base✝¹).obj X.fiber) Y.fiber
    base✝ : Quiver.Hom X.base Y.base
    fiber✝ : Quiver.Hom ((F.map base✝).obj X.fiber) Y.fiber
    w_base : Eq base✝¹ base✝
    w_fiber : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) {  …
    ⊢ HEq fiber✝¹ fiber✝
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- The identity morphism in the Grothendieck category.
-/
def id (X : Grothendieck F) : Hom X X where
  base := 𝟙 X.base
                       /-
                         C : Type u
                         inst✝¹ : CategoryTheory.Category.{v, u} C
                         D : Type u₁
                         inst✝ : CategoryTheory.Category.{v₁, u₁} D
                         F : CategoryTheory.Functor C CategoryTheory.Cat
                         X : CategoryTheory.Grothendieck F
                         ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.id X.base)).obj X.fiber) X.fiber
                       -/
  fiber := eqToHom (by erw [CategoryTheory.Functor.map_id, Functor.id_obj X.fiber])
                       /-
                         🎉 no goals
                       -/


instance (X : Grothendieck F) : Inhabited (Hom X X) :=
  ⟨id X⟩


/-- Composition of morphisms in the Grothendieck category.
-/
def comp {X Y Z : Grothendieck F} (f : Hom X Y) (g : Hom Y Z) : Hom X Z where
  base := f.base ≫ g.base
  fiber :=
                /-
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  D : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} D
                  F : CategoryTheory.Functor C CategoryTheory.Cat
                  X Y Z : CategoryTheory.Grothendieck F
                  f : X.Hom Y
                  g : Y.Hom Z
                  ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.comp f.base g.base)).obj X.fiber)  …
                -/
    eqToHom (by erw [Functor.map_comp, Functor.comp_obj]) ≫ (F.map g.base).map f.fiber ≫ g.fiber
                /-
                  🎉 no goals
                -/


instance : Category (Grothendieck F) where
  Hom X Y := Grothendieck.Hom X Y
  id X := Grothendieck.id X
  comp f g := Grothendieck.comp f g
  comp_id {X Y} f := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      X Y : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
    -/
    dsimp; ext
      /-
        case w_base
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F : CategoryTheory.Functor C CategoryTheory.Cat
        X Y : CategoryTheory.Grothendieck F
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.Grothendieck.comp f Y.id).base f.base
      -/
    · simp [comp, id]
      /-
        🎉 no goals
      -/
                  /-
                    C : Type u
                    inst✝¹ : CategoryTheory.Category.{v, u} C
                    D : Type u₁
                    inst✝ : CategoryTheory.Category.{v₁, u₁} D
                    F : CategoryTheory.Functor C CategoryTheory.Cat
                    X✝ Y✝ : CategoryTheory.Grothendieck F
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
                  -/
                                 /-
                                   🎉 no goals
                                 -/
      /-
        case w_fiber
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F : CategoryTheory.Functor C CategoryTheory.Cat
        X Y : CategoryTheory.Grothendieck F
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
                                 /-
                                   🎉 no goals
                                 -/
    · dsimp [comp, id]
      /-
        case w_fiber
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F : CategoryTheory.Functor C CategoryTheory.Cat
        X Y : CategoryTheory.Grothendieck F
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      rw [← NatIso.naturality_2 (eqToIso (F.map_id Y.base)) f.fiber]
      /-
        case w_fiber
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F : CategoryTheory.Functor C CategoryTheory.Cat
        X Y : CategoryTheory.Grothendieck F
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      simp
      /-
        🎉 no goals
      -/
  id_comp f := by dsimp; ext <;> simp [comp, id]
  assoc f g h := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Grothendieck F
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    dsimp; ext
      /-
        case w_base
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F : CategoryTheory.Functor C CategoryTheory.Cat
        W✝ X✝ Y✝ Z✝ : CategoryTheory.Grothendieck F
        f : Quiver.Hom W✝ X✝
        g : Quiver.Hom X✝ Y✝
        h : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.Grothendieck.comp (CategoryTheory.Grothendieck.comp f g)  …
      -/
    · simp [comp, id]
      /-
        🎉 no goals
      -/
      /-
        case w_fiber
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F : CategoryTheory.Functor C CategoryTheory.Cat
        W✝ X✝ Y✝ Z✝ : CategoryTheory.Grothendieck F
        f : Quiver.Hom W✝ X✝
        g : Quiver.Hom X✝ Y✝
        h : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
    · dsimp [comp, id]
      /-
        case w_fiber
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F : CategoryTheory.Functor C CategoryTheory.Cat
        W✝ X✝ Y✝ Z✝ : CategoryTheory.Grothendieck F
        f : Quiver.Hom W✝ X✝
        g : Quiver.Hom X✝ Y✝
        h : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      rw [← NatIso.naturality_2 (eqToIso (F.map_comp _ _)) f.fiber]
      /-
        case w_fiber
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F : CategoryTheory.Functor C CategoryTheory.Cat
        W✝ X✝ Y✝ Z✝ : CategoryTheory.Grothendieck F
        f : Quiver.Hom W✝ X✝
        g : Quiver.Hom X✝ Y✝
        h : Quiver.Hom Y✝ Z✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      simp
      /-
        🎉 no goals
      -/


@[simp]
theorem id_base (X : Grothendieck F) :
    Hom.base (𝟙 X) = 𝟙 X.base := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X : CategoryTheory.Grothendieck F
    ⊢ Eq (CategoryTheory.CategoryStruct.id X).base (CategoryTheory.CategoryStruct. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem id_fiber (X : Grothendieck F) :
                                  /-
                                    C : Type u
                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                    D : Type u₁
                                    inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                    F : CategoryTheory.Functor C CategoryTheory.Cat
                                    X : CategoryTheory.Grothendieck F
                                    ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.id X).base).obj X.fiber) X.fiber
                                  -/
    Hom.fiber (𝟙 X) = eqToHom (by erw [CategoryTheory.Functor.map_id, Functor.id_obj X.fiber]) :=
                                  /-
                                    🎉 no goals
                                  -/
  rfl


@[simp]
theorem comp_base {X Y Z : Grothendieck F} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).base = f.base ≫ g.base :=
  rfl


@[simp]
theorem comp_fiber {X Y Z : Grothendieck F} (f : X ⟶ Y) (g : Y ⟶ Z) :
    Hom.fiber (f ≫ g) =
                /-
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  D : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} D
                  F : CategoryTheory.Functor C CategoryTheory.Cat
                  X Y Z : CategoryTheory.Grothendieck F
                  f : Quiver.Hom X Y
                  g : Quiver.Hom Y Z
                  ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.comp f g).base).obj X.fiber) ((F.m …
                -/
    eqToHom (by erw [Functor.map_comp, Functor.comp_obj]) ≫
                /-
                  🎉 no goals
                -/
    (F.map g.base).map f.fiber ≫ g.fiber :=
  rfl



theorem congr {X Y : Grothendieck F} {f g : X ⟶ Y} (h : f = g) :
                          /-
                            C : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} C
                            D : Type u₁
                            inst✝ : CategoryTheory.Category.{v₁, u₁} D
                            F : CategoryTheory.Functor C CategoryTheory.Cat
                            X Y : CategoryTheory.Grothendieck F
                            f g : Quiver.Hom X Y
                            h : Eq f g
                            ⊢ Eq ((F.map f.base).obj X.fiber) ((F.map g.base).obj X.fiber)
                          -/
    f.fiber = eqToHom (by subst h; rfl) ≫ g.fiber := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X Y : CategoryTheory.Grothendieck F
    f g : Quiver.Hom X Y
    h : Eq f g
    ⊢ Eq f.fiber (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) g. …
  -/
  subst h
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X Y : CategoryTheory.Grothendieck F
    f : Quiver.Hom X Y
    ⊢ Eq f.fiber (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) f. …
  -/
  dsimp
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X Y : CategoryTheory.Grothendieck F
    f : Quiver.Hom X Y
    ⊢ Eq f.fiber (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma eqToHom_eq {X Y : Grothendieck F} (hF : X = Y) :
                                       /-
                                         C : Type u
                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                         D : Type u₁
                                         inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                         F : CategoryTheory.Functor C CategoryTheory.Cat
                                         X Y : CategoryTheory.Grothendieck F
                                         hF : Eq X Y
                                         ⊢ Eq X.base Y.base
                                       -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    eqToHom hF = { base := eqToHom (by subst hF; rfl), fiber := eqToHom (by subst hF; simp) } := by
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X Y : CategoryTheory.Grothendieck F
    hF : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom hF) { base := CategoryTheory.eqToHom ⋯, fiber :=  …
  -/
  subst hF
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    X : CategoryTheory.Grothendieck F
    ⊢ Eq (CategoryTheory.eqToHom ⋯) { base := CategoryTheory.eqToHom ⋯, fiber := C …
  -/
  rfl
  /-
    🎉 no goals
  -/

/-- The forgetful functor from `Grothendieck F` to the source category. -/
@[simps!]
def forget : Grothendieck F ⥤ C where
  obj X := X.1
  map f := f.1


/-- The Grothendieck construction is functorial: a natural transformation `α : F ⟶ G` induces
a functor `Grothendieck.map : Grothendieck F ⥤ Grothendieck G`.
-/
@[simps!]
def map (α : F ⟶ G) : Grothendieck F ⥤ Grothendieck G where
  obj X :=
  { base := X.base
    fiber := (α.app X.base).obj X.fiber }
  map {X Y} f :=
  { base := f.base
    fiber := (eqToHom (α.naturality f.base).symm).app X.fiber ≫ (α.app Y.base).map f.fiber }
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   D : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} D
                   F G : CategoryTheory.Functor C CategoryTheory.Cat
                   α : Quiver.Hom F G
                   X : CategoryTheory.Grothendieck F
                   ⊢ Eq ({ obj := fun X => { base := X.base, fiber := (α.app X.base).obj X.fiber  …
                 -/
  map_id X := by simp only [Cat.eqToHom_app, id_fiber, eqToHom_map, eqToHom_trans]; rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  map_comp {X Y Z} f g := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F G : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      X Y Z : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => { base := X.base, fiber := (α.app X.base).obj X.fiber  …
    -/
    dsimp
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F G : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      X Y Z : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq { base := CategoryTheory.CategoryStruct.comp f.base g.base, fiber := Cate …
    -/
    congr 1
    /-
      case e_fiber
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F G : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      X Y Z : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.eqToHom ⋯).app X.fib …
    -/
    simp only [comp_fiber f g, ← Category.assoc, Functor.map_comp, eqToHom_map]
    /-
      case e_fiber
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F G : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      X Y Z : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case e_fiber.e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F G : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      X Y Z : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Cat.eqToHom_app, Cat.comp_obj, eqToHom_trans, eqToHom_map, Category.assoc]
    /-
      case e_fiber.e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F G : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      X Y Z : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ((α.app Z. …
    -/
    erw [Functor.congr_hom (α.naturality g.base).symm f.fiber]
    /-
      case e_fiber.e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F G : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      X Y Z : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ((α.app Z. …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem map_obj {α : F ⟶ G} (X : Grothendieck F) :
    (Grothendieck.map α).obj X = ⟨X.base, (α.app X.base).obj X.fiber⟩ := rfl


theorem map_map {α : F ⟶ G} {X Y : Grothendieck F} {f : X ⟶ Y} :
    (Grothendieck.map α).map f =
    ⟨f.base, (eqToHom (α.naturality f.base).symm).app X.fiber ≫ (α.app Y.base).map f.fiber⟩ := rfl


/-- The functor `Grothendieck.map α : Grothendieck F ⥤ Grothendieck G` lies over `C`.-/
theorem functor_comp_forget {α : F ⟶ G} :
    Grothendieck.map α ⋙ Grothendieck.forget G = Grothendieck.forget F := rfl


theorem map_id_eq : map (𝟙 F) = 𝟙 (Cat.of <| Grothendieck <| F) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    ⊢ Eq (CategoryTheory.Grothendieck.map (CategoryTheory.CategoryStruct.id F)) (C …
  -/
  fapply Functor.ext
    /-
      case h_obj
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      ⊢ ∀ (X : CategoryTheory.Grothendieck F), Eq ((CategoryTheory.Grothendieck.map  …
    -/
  · intro X
    /-
      case h_obj
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      X : CategoryTheory.Grothendieck F
      ⊢ Eq ((CategoryTheory.Grothendieck.map (CategoryTheory.CategoryStruct.id F)).o …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      ⊢ autoParam (∀ (X Y : CategoryTheory.Grothendieck F) (f : Quiver.Hom X Y), Eq  …
    -/
  · intro X Y f
    /-
      case h_map
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      X Y : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      ⊢ Eq ((CategoryTheory.Grothendieck.map (CategoryTheory.CategoryStruct.id F)).m …
    -/
    simp [map_map]
    /-
      case h_map
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      X Y : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      ⊢ Eq { base := f.base, fiber := f.fiber } f
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Making the equality of functors into an isomorphism. Note: we should avoid equality of functors
if possible, and we should prefer `map_id_iso` to `map_id_eq` whenever we can. -/
def mapIdIso : map (𝟙 F) ≅ 𝟙 (Cat.of <| Grothendieck <| F) := eqToIso map_id_eq


theorem map_comp_eq (α : F ⟶ G) (β : G ⟶ H) :
    map (α ≫ β) = map α ⋙ map β := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G H : CategoryTheory.Functor C CategoryTheory.Cat
    α : Quiver.Hom F G
    β : Quiver.Hom G H
    ⊢ Eq (CategoryTheory.Grothendieck.map (CategoryTheory.CategoryStruct.comp α β) …
  -/
  fapply Functor.ext
    /-
      case h_obj
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G H : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      β : Quiver.Hom G H
      ⊢ ∀ (X : CategoryTheory.Grothendieck F), Eq ((CategoryTheory.Grothendieck.map  …
    -/
  · intro X
    /-
      case h_obj
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G H : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      β : Quiver.Hom G H
      X : CategoryTheory.Grothendieck F
      ⊢ Eq ((CategoryTheory.Grothendieck.map (CategoryTheory.CategoryStruct.comp α β …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G H : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      β : Quiver.Hom G H
      ⊢ autoParam (∀ (X Y : CategoryTheory.Grothendieck F) (f : Quiver.Hom X Y), Eq  …
    -/
  · intro X Y f
    simp only [map_map, map_obj_base, NatTrans.comp_app, Cat.comp_obj, Cat.comp_map,
      eqToHom_refl, Functor.comp_map, Functor.map_comp, Category.comp_id, Category.id_comp]
    /-
      case h_map
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G H : CategoryTheory.Functor C CategoryTheory.Cat
      α : Quiver.Hom F G
      β : Quiver.Hom G H
      X Y : CategoryTheory.Grothendieck F
      f : Quiver.Hom X Y
      ⊢ Eq { base := f.base, fiber := CategoryTheory.CategoryStruct.comp ((CategoryT …
    -/
    fapply Grothendieck.ext
      /-
        case h_map.w_base
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F G H : CategoryTheory.Functor C CategoryTheory.Cat
        α : Quiver.Hom F G
        β : Quiver.Hom G H
        X Y : CategoryTheory.Grothendieck F
        f : Quiver.Hom X Y
        ⊢ Eq { base := f.base, fiber := CategoryTheory.CategoryStruct.comp ((CategoryT …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h_map.w_fiber
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F G H : CategoryTheory.Functor C CategoryTheory.Cat
        α : Quiver.Hom F G
        β : Quiver.Hom G H
        X Y : CategoryTheory.Grothendieck F
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) { base :=  …
      -/
    · simp
      /-
        🎉 no goals
      -/


/-- Making the equality of functors into an isomorphism. Note: we should avoid equality of functors
if possible, and we should prefer `map_comp_iso` to `map_comp_eq` whenever we can. -/
def mapCompIso (α : F ⟶ G) (β : G ⟶ H) : map (α ≫ β) ≅ map α ⋙ map β := eqToIso (map_comp_eq α β)


/-- The inverse functor to build the equivalence `compAsSmallFunctorEquivalence`. -/
@[simps]
def compAsSmallFunctorEquivalenceInverse :
    Grothendieck F ⥤ Grothendieck (F ⋙ Cat.asSmallFunctor.{w}) where
  obj X := ⟨X.base, AsSmall.up.obj X.fiber⟩
  map f := ⟨f.base, AsSmall.up.map f.fiber⟩


/-- The functor to build the equivalence `compAsSmallFunctorEquivalence`. -/
@[simps]
def compAsSmallFunctorEquivalenceFunctor :
    Grothendieck (F ⋙ Cat.asSmallFunctor.{w}) ⥤ Grothendieck F where
  obj X := ⟨X.base, AsSmall.down.obj X.fiber⟩
  map f := ⟨f.base, AsSmall.down.map f.fiber⟩
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   D : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} D
                   F : CategoryTheory.Functor C CategoryTheory.Cat
                   G : CategoryTheory.Functor C CategoryTheory.Cat
                   H : CategoryTheory.Functor C CategoryTheory.Cat
                   x✝ : CategoryTheory.Grothendieck (F.comp CategoryTheory.Cat.asSmallFunctor)
                   ⊢ Eq ({ obj := fun X => { base := X.base, fiber := CategoryTheory.AsSmall.down …
                 -/
                                            /-
                                              🎉 no goals
                                            -/
  map_id _ := by apply Grothendieck.ext <;> simp
                                            /-
                                              🎉 no goals
                                            -/
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       D : Type u₁
                       inst✝ : CategoryTheory.Category.{v₁, u₁} D
                       F : CategoryTheory.Functor C CategoryTheory.Cat
                       G : CategoryTheory.Functor C CategoryTheory.Cat
                       H : CategoryTheory.Functor C CategoryTheory.Cat
                       X✝ Y✝ Z✝ : CategoryTheory.Grothendieck (F.comp CategoryTheory.Cat.asSmallFunct …
                       x✝¹ : Quiver.Hom X✝ Y✝
                       x✝ : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun X => { base := X.base, fiber := CategoryTheory.AsSmall.down …
                     -/
                                                /-
                                                  🎉 no goals
                                                -/
  map_comp _ _ := by apply Grothendieck.ext <;> simp [down_comp]
                                                /-
                                                  🎉 no goals
                                                -/


/-- Taking the Grothendieck construction on `F ⋙ asSmallFunctor`, where
`asSmallFunctor : Cat ⥤ Cat` is the functor which turns each category into a small category of a
(potentiall) larger universe, is equivalent to the Grothendieck construction on `F` itself. -/
@[simps]
def compAsSmallFunctorEquivalence :
    Grothendieck (F ⋙ Cat.asSmallFunctor.{w}) ≌ Grothendieck F where
  functor := compAsSmallFunctorEquivalenceFunctor F
  inverse := compAsSmallFunctorEquivalenceInverse F
  counitIso := Iso.refl _
  unitIso := Iso.refl _


/-- Mapping a Grothendieck construction along the whiskering of any natural transformation
`α : F ⟶ G` with the functor `asSmallFunctor : Cat ⥤ Cat` is naturally isomorphic to conjugating
`map α` with the equivalence between `Grothendieck (F ⋙ asSmallFunctor)` and `Grothendieck F`. -/
def mapWhiskerRightAsSmallFunctor (α : F ⟶ G) :
    map (whiskerRight α Cat.asSmallFunctor.{w}) ≅
    (compAsSmallFunctorEquivalence F).functor ⋙ map α ⋙
      (compAsSmallFunctorEquivalence G).inverse :=
  NatIso.ofComponents
    (fun X => Iso.refl _)
    (fun f => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        F G : CategoryTheory.Functor C CategoryTheory.Cat
        H : CategoryTheory.Functor C CategoryTheory.Cat
        α : Quiver.Hom F G
        X✝ Y✝ : CategoryTheory.Grothendieck (F.comp CategoryTheory.Cat.asSmallFunctor)
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Grothendieck.map (Ca …
      -/
      fapply Grothendieck.ext
        /-
          case w_base
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F G : CategoryTheory.Functor C CategoryTheory.Cat
          H : CategoryTheory.Functor C CategoryTheory.Cat
          α : Quiver.Hom F G
          X✝ Y✝ : CategoryTheory.Grothendieck (F.comp CategoryTheory.Cat.asSmallFunctor)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Grothendieck.map (Ca …
        -/
      · simp [compAsSmallFunctorEquivalenceInverse]
        /-
          🎉 no goals
        -/
      · simp only [compAsSmallFunctorEquivalence_functor, compAsSmallFunctorEquivalence_inverse,
          Functor.comp_obj, compAsSmallFunctorEquivalenceInverse_obj_base, map_obj_base,
          compAsSmallFunctorEquivalenceFunctor_obj_base, Cat.asSmallFunctor_obj, Cat.of_α,
          Iso.refl_hom, Functor.comp_map, comp_base, id_base,
          compAsSmallFunctorEquivalenceInverse_map_base, map_map_base,
          compAsSmallFunctorEquivalenceFunctor_map_base, Cat.asSmallFunctor_map, map_obj_fiber,
          whiskerRight_app, AsSmall.down_obj, AsSmall.up_obj_down,
          compAsSmallFunctorEquivalenceInverse_obj_fiber,
          compAsSmallFunctorEquivalenceFunctor_obj_fiber, comp_fiber, map_map_fiber,
          AsSmall.down_map, down_comp, eqToHom_down, AsSmall.up_map_down, Functor.map_comp,
          eqToHom_map, id_fiber, Category.assoc, eqToHom_trans_assoc,
          compAsSmallFunctorEquivalenceInverse_map_fiber,
          compAsSmallFunctorEquivalenceFunctor_map_fiber, eqToHom_comp_iff, comp_eqToHom_iff]
        /-
          case w_fiber
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F G : CategoryTheory.Functor C CategoryTheory.Cat
          H : CategoryTheory.Functor C CategoryTheory.Cat
          α : Quiver.Hom F G
          X✝ Y✝ : CategoryTheory.Grothendieck (F.comp CategoryTheory.Cat.asSmallFunctor)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.AsSmall.up.map ((G.map (CategoryTheory.CategoryStruct.id  …
        -/
        simp only [eqToHom_trans_assoc, Category.assoc, conj_eqToHom_iff_heq']
        /-
          case w_fiber
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F G : CategoryTheory.Functor C CategoryTheory.Cat
          H : CategoryTheory.Functor C CategoryTheory.Cat
          α : Quiver.Hom F G
          X✝ Y✝ : CategoryTheory.Grothendieck (F.comp CategoryTheory.Cat.asSmallFunctor)
          f : Quiver.Hom X✝ Y✝
          ⊢ HEq (CategoryTheory.AsSmall.up.map ((G.map (CategoryTheory.CategoryStruct.id …
        -/
        rw [G.map_id]
        /-
          case w_fiber
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F G : CategoryTheory.Functor C CategoryTheory.Cat
          H : CategoryTheory.Functor C CategoryTheory.Cat
          α : Quiver.Hom F G
          X✝ Y✝ : CategoryTheory.Grothendieck (F.comp CategoryTheory.Cat.asSmallFunctor)
          f : Quiver.Hom X✝ Y✝
          ⊢ HEq (CategoryTheory.AsSmall.up.map ((CategoryTheory.CategoryStruct.id (G.obj …
        -/
        simp )
        /-
          🎉 no goals
        -/


/-- The Grothendieck construction as a functor from the functor category `E ⥤ Cat` to the
over category `Over E`. -/
def functor {E : Cat.{v,u}} : (E ⥤ Cat.{v,u}) ⥤ Over (T := Cat.{v,u}) E where
  obj F := Over.mk (X := E) (Y := Cat.of (Grothendieck F)) (Grothendieck.forget F)
  map {_ _} α := Over.homMk (X:= E) (Grothendieck.map α) Grothendieck.functor_comp_forget
  map_id F := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F✝ : CategoryTheory.Functor C CategoryTheory.Cat
      E : CategoryTheory.Cat
      F : CategoryTheory.Functor (↑E) CategoryTheory.Cat
      ⊢ Eq ({ obj := fun F => CategoryTheory.Over.mk (CategoryTheory.Grothendieck.fo …
    -/
    ext
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F✝ : CategoryTheory.Functor C CategoryTheory.Cat
      E : CategoryTheory.Cat
      F : CategoryTheory.Functor (↑E) CategoryTheory.Cat
      ⊢ Eq ({ obj := fun F => CategoryTheory.Over.mk (CategoryTheory.Grothendieck.fo …
    -/
    exact Grothendieck.map_id_eq (F := F)
    /-
      🎉 no goals
    -/
  map_comp α β := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      E : CategoryTheory.Cat
      X✝ Y✝ Z✝ : CategoryTheory.Functor (↑E) CategoryTheory.Cat
      α : Quiver.Hom X✝ Y✝
      β : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.Over.mk (CategoryTheory.Grothendieck.fo …
    -/
    simp [Grothendieck.map_comp_eq α β]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      E : CategoryTheory.Cat
      X✝ Y✝ Z✝ : CategoryTheory.Functor (↑E) CategoryTheory.Cat
      α : Quiver.Hom X✝ Y✝
      β : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.Over.homMk ((CategoryTheory.Grothendieck.map α).comp (Cat …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `grothendieckTypeToCat`, to speed up elaboration. -/
@[simps!]
def grothendieckTypeToCatFunctor : Grothendieck (G ⋙ typeToCat) ⥤ G.Elements where
  obj X := ⟨X.1, X.2.as⟩
  map f := ⟨f.1, f.2.1.1⟩


/-- Auxiliary definition for `grothendieckTypeToCat`, to speed up elaboration. -/
-- Porting note:
-- `simps` is incorrectly producing Prop-valued projections here,
-- so we manually specify which ones to produce.
-- See https://leanprover.zulipchat.com/#narrow/stream/144837-PR-reviews/topic/!4.233204.20simps.20bug.20.28Grothendieck.20construction.29
@[simps! obj_base obj_fiber_as map_base]
def grothendieckTypeToCatInverse : G.Elements ⥤ Grothendieck (G ⋙ typeToCat) where
  obj X := ⟨X.1, ⟨X.2⟩⟩
  map f := ⟨f.1, ⟨⟨f.2⟩⟩⟩


/-- The Grothendieck construction applied to a functor to `Type`
(thought of as a functor to `Cat` by realising a type as a discrete category)
is the same as the 'category of elements' construction.
-/
-- See porting note on grothendieckTypeToCatInverse.
-- We just want to turn off grothendieckTypeToCat_inverse_map_fiber_down_down,
-- so have to list the complement here for `@[simps]`.
@[simps! functor_obj_fst functor_obj_snd functor_map_coe inverse_obj_base inverse_obj_fiber_as
  inverse_map_base unitIso_hom_app_base unitIso_hom_app_fiber unitIso_inv_app_base
  unitIso_inv_app_fiber counitIso_hom_app_coe counitIso_inv_app_coe]
def grothendieckTypeToCat : Grothendieck (G ⋙ typeToCat) ≌ G.Elements where
  functor := grothendieckTypeToCatFunctor G
  inverse := grothendieckTypeToCatInverse G
  unitIso :=
    NatIso.ofComponents
      (fun X => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          X : CategoryTheory.Grothendieck (G.comp CategoryTheory.typeToCat)
          ⊢ CategoryTheory.Iso ((CategoryTheory.Functor.id (CategoryTheory.Grothendieck  …
        -/
        rcases X with ⟨_, ⟨⟩⟩
        /-
          case mk.mk
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          base✝ : C
          as✝ : G.obj base✝
          ⊢ CategoryTheory.Iso ((CategoryTheory.Functor.id (CategoryTheory.Grothendieck  …
        -/
        exact Iso.refl _)
        /-
          🎉 no goals
        -/
      (by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          ⊢ ∀ {X Y : CategoryTheory.Grothendieck (G.comp CategoryTheory.typeToCat)} (f : …
        -/
        rintro ⟨_, ⟨⟩⟩ ⟨_, ⟨⟩⟩ ⟨base, ⟨⟨f⟩⟩⟩
        /-
          case mk.mk.mk.mk.mk.up.up
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          base✝¹ : C
          as✝¹ : G.obj base✝¹
          base✝ : C
          as✝ : G.obj base✝
          base : Quiver.Hom { base := base✝¹, fiber := { as := as✝¹ } }.base { base := b …
          f : Eq (((G.comp CategoryTheory.typeToCat).map base).obj { base := base✝¹, fib …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
        -/
        dsimp at *
        /-
          case mk.mk.mk.mk.mk.up.up
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          base✝¹ : C
          as✝¹ : G.obj base✝¹
          base✝ : C
          as✝ : G.obj base✝
          base : Quiver.Hom { base := base✝¹, fiber := { as := as✝¹ } }.base { base := b …
          f : Eq (((G.comp CategoryTheory.typeToCat).map base).obj { base := base✝¹, fib …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { base := base, fiber := { down := {  …
        -/
        simp
        /-
          case mk.mk.mk.mk.mk.up.up
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          base✝¹ : C
          as✝¹ : G.obj base✝¹
          base✝ : C
          as✝ : G.obj base✝
          base : Quiver.Hom { base := base✝¹, fiber := { as := as✝¹ } }.base { base := b …
          f : Eq (((G.comp CategoryTheory.typeToCat).map base).obj { base := base✝¹, fib …
          ⊢ Eq { base := base, fiber := { down := { down := f } } } ((CategoryTheory.Gro …
        -/
        rfl)
        /-
          🎉 no goals
        -/
  counitIso :=
    NatIso.ofComponents
      (fun X => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          X : G.Elements
          ⊢ CategoryTheory.Iso (((CategoryTheory.Grothendieck.grothendieckTypeToCatInver …
        -/
        cases X
        /-
          case mk
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          fst✝ : C
          snd✝ : G.obj fst✝
          ⊢ CategoryTheory.Iso (((CategoryTheory.Grothendieck.grothendieckTypeToCatInver …
        -/
        exact Iso.refl _)
        /-
          🎉 no goals
        -/
      (by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          ⊢ ∀ {X Y : G.Elements} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
        -/
        rintro ⟨⟩ ⟨⟩ ⟨f, e⟩
        /-
          case mk.mk.mk
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          fst✝¹ : C
          snd✝¹ : G.obj fst✝¹
          fst✝ : C
          snd✝ : G.obj fst✝
          f : Quiver.Hom ⟨fst✝¹, snd✝¹⟩.fst ⟨fst✝, snd✝⟩.fst
          e : Eq (G.map f ⟨fst✝¹, snd✝¹⟩.snd) ⟨fst✝, snd✝⟩.snd
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Grothendieck.grothe …
        -/
        dsimp at *
        /-
          case mk.mk.mk
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          fst✝¹ : C
          snd✝¹ : G.obj fst✝¹
          fst✝ : C
          snd✝ : G.obj fst✝
          f : Quiver.Hom ⟨fst✝¹, snd✝¹⟩.fst ⟨fst✝, snd✝⟩.fst
          e : Eq (G.map f ⟨fst✝¹, snd✝¹⟩.snd) ⟨fst✝, snd✝⟩.snd
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Grothendieck.grothen …
        -/
        simp
        /-
          case mk.mk.mk
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} D
          F : CategoryTheory.Functor C CategoryTheory.Cat
          G : CategoryTheory.Functor C (Type w)
          fst✝¹ : C
          snd✝¹ : G.obj fst✝¹
          fst✝ : C
          snd✝ : G.obj fst✝
          f : Quiver.Hom ⟨fst✝¹, snd✝¹⟩.fst ⟨fst✝, snd✝⟩.fst
          e : Eq (G.map f ⟨fst✝¹, snd✝¹⟩.snd) ⟨fst✝, snd✝⟩.snd
          ⊢ Eq ((CategoryTheory.Grothendieck.grothendieckTypeToCatFunctor G).map ((Categ …
        -/
        rfl)
        /-
          🎉 no goals
        -/
  functor_unitIso_comp := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      ⊢ ∀ (X : CategoryTheory.Grothendieck (G.comp CategoryTheory.typeToCat)), Eq (C …
    -/
    rintro ⟨_, ⟨⟩⟩
    /-
      case mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      base✝ : C
      as✝ : G.obj base✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Grothendieck.grothen …
    -/
    dsimp
    /-
      case mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      base✝ : C
      as✝ : G.obj base✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Grothendieck.grothen …
    -/
    simp
    /-
      case mk.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      base✝ : C
      as✝ : G.obj base✝
      ⊢ Eq (Sigma.rec (motive := fun t => Eq ((CategoryTheory.Grothendieck.grothendi …
    -/
    rfl
    /-
      🎉 no goals
    -/


variable (F) in
/-- Applying a functor `G : D ⥤ C` to the base of the Grothendieck construction induces a functor
`Grothendieck (G ⋙ F) ⥤ Grothendieck F`. -/
@[simps]
def pre (G : D ⥤ C) : Grothendieck (G ⋙ F) ⥤ Grothendieck F where
  obj X := ⟨G.obj X.base, X.fiber⟩
  map f := ⟨G.map f.base, f.fiber⟩
                                                    /-
                                                      C : Type u
                                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                                      D : Type u₁
                                                      inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                                      F : CategoryTheory.Functor C CategoryTheory.Cat
                                                      G✝ : CategoryTheory.Functor C (Type w)
                                                      G : CategoryTheory.Functor D C
                                                      X : CategoryTheory.Grothendieck (G.comp F)
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ({ obj :=  …
                                                    -/
  map_id X := Grothendieck.ext _ _ (G.map_id _) (by simp)
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                            /-
                                                              C : Type u
                                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                                              D : Type u₁
                                                              inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                                              F : CategoryTheory.Functor C CategoryTheory.Cat
                                                              G✝ : CategoryTheory.Functor C (Type w)
                                                              G : CategoryTheory.Functor D C
                                                              X✝ Y✝ Z✝ : CategoryTheory.Grothendieck (G.comp F)
                                                              f : Quiver.Hom X✝ Y✝
                                                              g : Quiver.Hom Y✝ Z✝
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ({ obj :=  …
                                                            -/
  map_comp f g := Grothendieck.ext _ _ (G.map_comp _ _) (by simp)
                                                            /-
                                                              🎉 no goals
                                                            -/


variable (F) in
/-- The inclusion of a fiber `F.obj c` of a functor `F : C ⥤ Cat` into its Grothendieck
construction.-/
@[simps obj map]
def ι (c : C) : F.obj c ⥤ Grothendieck F where
  obj d := ⟨c, d⟩
                             /-
                               C : Type u
                               inst✝² : CategoryTheory.Category.{v, u} C
                               D : Type u₁
                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                               F : CategoryTheory.Functor C CategoryTheory.Cat
                               G : CategoryTheory.Functor C (Type w)
                               E : Type u_1
                               inst✝ : CategoryTheory.Category.{?u.151647, u_1} E
                               c : C
                               X✝ Y✝ : ↑(F.obj c)
                               f : Quiver.Hom X✝ Y✝
                               ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.id ((fun d => { base := c, fiber : …
                             -/
  map f := ⟨𝟙 _, eqToHom (by simp) ≫ f⟩
                             /-
                               🎉 no goals
                             -/
  map_id d := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.151647, u_1} E
      c : C
      d : ↑(F.obj c)
      ⊢ Eq ({ obj := fun d => { base := c, fiber := d }, map := fun {X Y} f => { bas …
    -/
    dsimp
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.151647, u_1} E
      c : C
      d : ↑(F.obj c)
      ⊢ Eq { base := CategoryTheory.CategoryStruct.id c, fiber := CategoryTheory.Cat …
    -/
    congr
    /-
      case e_fiber
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.151647, u_1} E
      c : C
      d : ↑(F.obj c)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    simp only [Category.comp_id]
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.151647, u_1} E
      c : C
      X✝ Y✝ Z✝ : ↑(F.obj c)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun d => { base := c, fiber := d }, map := fun {X Y} f => { bas …
    -/
    apply Grothendieck.ext _ _ (by simp)
    simp only [comp_base, ← Category.assoc, eqToHom_trans, comp_fiber, Functor.map_comp,
      eqToHom_map]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.151647, u_1} E
      c : C
      X✝ Y✝ Z✝ : ↑(F.obj c)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case e_a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.151647, u_1} E
      c : C
      X✝ Y✝ Z✝ : ↑(F.obj c)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) f) (Catego …
    -/
    simp only [eqToHom_comp_iff, Category.assoc, eqToHom_trans_assoc]
    /-
      case e_a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.151647, u_1} E
      c : C
      X✝ Y✝ Z✝ : ↑(F.obj c)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Categor …
    -/
    apply Functor.congr_hom (F.map_id _).symm
    /-
      🎉 no goals
    -/


instance faithful_ι (c : C) : (ι F c).Faithful where
  map_injective f := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.155721, u_1} E
      c : C
      X✝ Y✝ : ↑(F.obj c)
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      f : Eq ((CategoryTheory.Grothendieck.ι F c).map a₁✝) ((CategoryTheory.Grothend …
      ⊢ Eq a₁✝ a₂✝
    -/
    injection f with _ f
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.155721, u_1} E
      c : C
      X✝ Y✝ : ↑(F.obj c)
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      base_eq✝ : Eq (CategoryTheory.CategoryStruct.id ((fun d => { base := c, fiber  …
      f : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) a₁✝) (Ca …
      ⊢ Eq a₁✝ a₂✝
    -/
    rwa [cancel_epi] at f
    /-
      🎉 no goals
    -/


/-- Every morphism `f : X ⟶ Y` in the base category induces a natural transformation from the fiber
inclusion `ι F X` to the composition `F.map f ⋙ ι F Y`. -/
@[simps]
def ιNatTrans {X Y : C} (f : X ⟶ Y) : ι F X ⟶ F.map f ⋙ ι F Y where
  app d := ⟨f, 𝟙 _⟩
  naturality _ _ _ := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.156515, u_1} E
      X Y : C
      f : Quiver.Hom X Y
      x✝² x✝¹ : ↑(F.obj X)
      x✝ : Quiver.Hom x✝² x✝¹
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Grothendieck.ι F X). …
    -/
    simp only [ι, Functor.comp_obj, Functor.comp_map]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      F : CategoryTheory.Functor C CategoryTheory.Cat
      G : CategoryTheory.Functor C (Type w)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.156515, u_1} E
      X Y : C
      f : Quiver.Hom X Y
      x✝² x✝¹ : ↑(F.obj X)
      x✝ : Quiver.Hom x✝² x✝¹
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { base := CategoryTheory.CategoryStru …
    -/
    exact Grothendieck.ext _ _ (by simp) (by simp [eqToHom_map])
    /-
      🎉 no goals
    -/


                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                                                  F : CategoryTheory.Functor C CategoryTheory.Cat
                                                  G : CategoryTheory.Functor C (Type w)
                                                  E : Type u_1
                                                  inst✝ : CategoryTheory.Category.{?u.165780, u_1} E
                                                  fib : (c : C) → CategoryTheory.Functor (↑(F.obj c)) E
                                                  hom : {c c' : C} → (f : Quiver.Hom c c') → Quiver.Hom (fib c) (CategoryTheory. …
                                                  c : C
                                                  ⊢ Eq (fib c) (CategoryTheory.Functor.comp (F.map (CategoryTheory.CategoryStruc …
                                                -/
variable (hom_id : ∀ c, hom (𝟙 c) = eqToHom (by simp only [Functor.map_id]; rfl))
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
variable (hom_comp : ∀ c₁ c₂ c₃ (f : c₁ ⟶ c₂) (g : c₂ ⟶ c₃), hom (f ≫ g) =
                                                      /-
                                                        C : Type u
                                                        inst✝² : CategoryTheory.Category.{v, u} C
                                                        D : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                                                        F : CategoryTheory.Functor C CategoryTheory.Cat
                                                        G : CategoryTheory.Functor C (Type w)
                                                        E : Type u_1
                                                        inst✝ : CategoryTheory.Category.{?u.165780, u_1} E
                                                        fib : (c : C) → CategoryTheory.Functor (↑(F.obj c)) E
                                                        hom : {c c' : C} → (f : Quiver.Hom c c') → Quiver.Hom (fib c) (CategoryTheory. …
                                                        hom_id : ∀ (c : C), Eq (hom (CategoryTheory.CategoryStruct.id c)) (CategoryThe …
                                                        c₁ c₂ c₃ : C
                                                        f : Quiver.Hom c₁ c₂
                                                        g : Quiver.Hom c₂ c₃
                                                        ⊢ Eq (CategoryTheory.Functor.comp (F.map f) (CategoryTheory.Functor.comp (F.ma …
                                                      -/
  hom f ≫ whiskerLeft (F.map f) (hom g) ≫ eqToHom (by simp only [Functor.map_comp]; rfl))
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/

/-- Construct a functor from `Grothendieck F` to another category `E` by providing a family of
functors on the fibers of `Grothendieck F`, a family of natural transformations on morphisms in the
base of `Grothendieck F` and coherence data for this family of natural transformations. -/
@[simps]
def functorFrom : Grothendieck F ⥤ E where
  obj X := (fib X.base).obj X.fiber
  map {X Y} f := (hom f.base).app X.fiber ≫ (fib Y.base).map f.fiber
                 /-
                   C : Type u
                   inst✝² : CategoryTheory.Category.{v, u} C
                   D : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                   F : CategoryTheory.Functor C CategoryTheory.Cat
                   G : CategoryTheory.Functor C (Type w)
                   E : Type u_1
                   inst✝ : CategoryTheory.Category.{?u.165780, u_1} E
                   fib : (c : C) → CategoryTheory.Functor (↑(F.obj c)) E
                   hom : {c c' : C} → (f : Quiver.Hom c c') → Quiver.Hom (fib c) (CategoryTheory. …
                   hom_id : ∀ (c : C), Eq (hom (CategoryTheory.CategoryStruct.id c)) (CategoryThe …
                   hom_comp : ∀ (c₁ c₂ c₃ : C) (f : Quiver.Hom c₁ c₂) (g : Quiver.Hom c₂ c₃), Eq  …
                   X : CategoryTheory.Grothendieck F
                   ⊢ Eq ({ obj := fun X => (fib X.base).obj X.fiber, map := fun {X Y} f => Catego …
                 -/
  map_id X := by simp [hom_id]
                 /-
                   🎉 no goals
                 -/
                     /-
                       C : Type u
                       inst✝² : CategoryTheory.Category.{v, u} C
                       D : Type u₁
                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                       F : CategoryTheory.Functor C CategoryTheory.Cat
                       G : CategoryTheory.Functor C (Type w)
                       E : Type u_1
                       inst✝ : CategoryTheory.Category.{?u.165780, u_1} E
                       fib : (c : C) → CategoryTheory.Functor (↑(F.obj c)) E
                       hom : {c c' : C} → (f : Quiver.Hom c c') → Quiver.Hom (fib c) (CategoryTheory. …
                       hom_id : ∀ (c : C), Eq (hom (CategoryTheory.CategoryStruct.id c)) (CategoryThe …
                       hom_comp : ∀ (c₁ c₂ c₃ : C) (f : Quiver.Hom c₁ c₂) (g : Quiver.Hom c₂ c₃), Eq  …
                       X✝ Y✝ Z✝ : CategoryTheory.Grothendieck F
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun X => (fib X.base).obj X.fiber, map := fun {X Y} f => Catego …
                     -/
  map_comp f g := by simp [hom_comp]
                     /-
                       🎉 no goals
                     -/


egoryTheory.Functor (↑(F.obj c)) E
                                                  hom : {c c' : C} → (f : Quiver.Hom c c') → Quiver.Hom (fib c) (CategoryTheory. …
                                                  c : C
                                                  ⊢ Eq (fib c) (CategoryTheory.Functor.comp (F.map (CategoryTheory.CategoryStruc …
                                                -/
variable (hom_id : ∀ c, hom (𝟙 c) = eqToHom (by simp only [Functor.map_id]; rfl))
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
variable (hom_comp : ∀ c₁ c₂ c₃ (f : c₁ ⟶ c₂) (g : c₂ ⟶ c₃), hom (f ≫ g) =
                                                      /-
                                                        C : Type u
                                                        inst✝² : CategoryTheory.Category.{v, u} C
                                                        D : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                                                        F : CategoryTheory.Functor C CategoryTheory.Cat
                                                        G : CategoryTheory.Functor C (Type w)
                                                        E : Type u_1
                                                        inst✝ : CategoryTheory.Category.{?u.173344, u_1} E
                                                        fib : (c : C) → CategoryTheory.Functor (↑(F.obj c)) E
                                                        hom : {c c' : C} → (f : Quiver.Hom c c') → Quiver.Hom (fib c) (CategoryTheory. …
                                                        hom_id : ∀ (c : C), Eq (hom (CategoryTheory.CategoryStruct.id c)) (CategoryThe …
                                                        c₁ c₂ c₃ : C
                                                        f : Quiver.Hom c₁ c₂
                                                        g : Quiver.Hom c₂ c₃
                                                        ⊢ Eq (CategoryTheory.Functor.comp (F.map f) (CategoryTheory.Functor.comp (F.ma …
                                                      -/
  hom f ≫ whiskerLeft (F.map f) (hom g) ≫ eqToHom (by simp only [Functor.map_comp]; rfl))
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/

/-- Construct a functor from `Grothendieck F` to another category `E` by providing a family of
functors on the fibers of `Grothendieck F`, a family of natural transformations on morphisms in the
base of `Grothendieck F` and coherence data for this family of natural transformations. -/
@[simps]
def functorFrom : Grothendieck F ⥤ E where
  obj X := (fib X.base).obj X.fiber
  map {X Y} f := (hom f.base).app X.fiber ≫ (fib Y.base).map f.fiber
  map_id X := by simp [hom_id]
  map_comp f g := by simp [hom_comp]

/-- `Grothendieck.ι F c` composed with `Grothendieck.functorFrom` is isomorphic a functor on a fiber
on `F` supplied as the first argument to `Grothendieck.functorFrom`. -/
def ιCompFunctorFrom (c : C) : ι F c ⋙ (functorFrom fib hom hom_id hom_comp) ≅ fib c :=
                                                         /-
                                                           C : Type u
                                                           inst✝² : CategoryTheory.Category.{v, u} C
                                                           D : Type u₁
                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                                                           F : CategoryTheory.Functor C CategoryTheory.Cat
                                                           G : CategoryTheory.Functor C (Type w)
                                                           E : Type u_1
                                                           inst✝ : CategoryTheory.Category.{?u.173344, u_1} E
                                                           fib : (c : C) → CategoryTheory.Functor (↑(F.obj c)) E
                                                           hom : {c c' : C} → (f : Quiver.Hom c c') → Quiver.Hom (fib c) (CategoryTheory. …
                                                           hom_id : ∀ (c : C), Eq (hom (CategoryTheory.CategoryStruct.id c)) (CategoryThe …
                                                           hom_comp : ∀ (c₁ c₂ c₃ : C) (f : Quiver.Hom c₁ c₂) (g : Quiver.Hom c₂ c₃), Eq  …
                                                           c : C
                                                           X✝ Y✝ : ↑(F.obj c)
                                                           f : Quiver.Hom X✝ Y✝
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Grothendieck.ι F c) …
                                                         -/
  NatIso.ofComponents (fun _ => Iso.refl _) (fun f => by simp [hom_id])
                                                         /-
                                                           🎉 no goals
                                                         -/


