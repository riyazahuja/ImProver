/-- Formally adjoin a terminal object to a category. -/
inductive WithTerminal : Type u
  | of : C → WithTerminal
  | star : WithTerminal
  deriving Inhabited


/-- Formally adjoin an initial object to a category. -/
inductive WithInitial : Type u
  | of : C → WithInitial
  | star : WithInitial
  deriving Inhabited


/-- Morphisms for `WithTerminal C`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `nolint has_nonempty_instance`
@[simp]
def Hom : WithTerminal C → WithTerminal C → Type v
  | of X, of Y => X ⟶ Y
  | star, of _ => PEmpty
  | _, star => PUnit

/-- Identity morphisms for `WithTerminal C`. -/
@[simp]
def id : ∀ X : WithTerminal C, Hom X X
  | of _ => 𝟙 _
  | star => PUnit.unit


/-- Composition of morphisms for `WithTerminal C`. -/
@[simp]
def comp : ∀ {X Y Z : WithTerminal C}, Hom X Y → Hom Y Z → Hom X Z
  | of _X, of _Y, of _Z => fun f g => f ≫ g
  | of _X, _, star => fun _f _g => PUnit.unit
  | star, of _X, _ => fun f _g => PEmpty.elim f
  | _, star, of _Y => fun _f g => PEmpty.elim g
  | star, star, star => fun _ _ => PUnit.unit

instance : Category.{v} (WithTerminal C) where
  Hom X Y := Hom X Y
  id _ := id _
  comp := comp
  assoc {a b c d} f g h := by
    -- Porting note: it would be nice to automate this away as well.
    -- I tried splitting this into separate `Quiver` and `Category` instances,
    -- so the `false_of_from_star` destruct rule below can be used here.
    -- That works, but causes mysterious failures of `aesop_cat` in `map`.
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a b c d : CategoryTheory.WithTerminal C
      f : Quiver.Hom a b
      g : Quiver.Hom b c
      h : Quiver.Hom c d
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
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
    cases a <;> cases b <;> cases c <;> cases d <;> try aesop_cat
                                                    /-
                                                      🎉 no goals
                                                    -/
      /-
        case of.of.star.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝² a✝¹ : C
        f : Quiver.Hom (CategoryTheory.WithTerminal.of a✝²) (CategoryTheory.WithTermin …
        g : Quiver.Hom (CategoryTheory.WithTerminal.of a✝¹) CategoryTheory.WithTermina …
        a✝ : C
        h : Quiver.Hom CategoryTheory.WithTerminal.star (CategoryTheory.WithTerminal.o …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · exact (h : PEmpty).elim
      /-
        🎉 no goals
      -/
      /-
        case of.star.of.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝² : C
        f : Quiver.Hom (CategoryTheory.WithTerminal.of a✝²) CategoryTheory.WithTermina …
        a✝¹ : C
        g : Quiver.Hom CategoryTheory.WithTerminal.star (CategoryTheory.WithTerminal.o …
        a✝ : C
        h : Quiver.Hom (CategoryTheory.WithTerminal.of a✝¹) (CategoryTheory.WithTermin …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · exact (g : PEmpty).elim
      /-
        🎉 no goals
      -/
      /-
        case of.star.star.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ : C
        f : Quiver.Hom (CategoryTheory.WithTerminal.of a✝¹) CategoryTheory.WithTermina …
        g : Quiver.Hom CategoryTheory.WithTerminal.star CategoryTheory.WithTerminal.star
        a✝ : C
        h : Quiver.Hom CategoryTheory.WithTerminal.star (CategoryTheory.WithTerminal.o …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · exact (h : PEmpty).elim
      /-
        🎉 no goals
      -/


/-- Helper function for typechecking. -/
def down {X Y : C} (f : of X ⟶ of Y) : X ⟶ Y := f


@[simp] lemma down_id {X : C} : down (𝟙 (of X)) = 𝟙 X := rfl

@[simp] lemma down_comp {X Y Z : C} (f : of X ⟶ of Y) (g : of Y ⟶ of Z) :
    down (f ≫ g) = down f ≫ down g :=
  rfl


@[aesop safe destruct (rule_sets := [CategoryTheory])]
lemma false_of_from_star {X : C} (f : star ⟶ of X) : False := (f : PEmpty).elim


/-- The inclusion from `C` into `WithTerminal C`. -/
def incl : C ⥤ WithTerminal C where
  obj := of
  map f := f


instance : (incl : C ⥤ _).Full where
  map_surjective f := ⟨f, rfl⟩


instance : (incl : C ⥤ _).Faithful where


/-- Map `WithTerminal` with respect to a functor `F : C ⥤ D`. -/
@[simps]
def map {D : Type*} [Category D] (F : C ⥤ D) : WithTerminal C ⥤ WithTerminal D where
  obj X :=
    match X with
    | of x => of <| F.obj x
    | star => star
  map {X Y} f :=
    match X, Y, f with
    | of _, of _, f => F.map (down f)
    | of _, star, _ => PUnit.unit
    | star, star, _ => PUnit.unit


/-- A natural isomorphism between the functor `map (𝟭 C)` and `𝟭 (WithTerminal C)`. -/
@[simps!]
def mapId (C : Type*) [Category C] : map (𝟭 C) ≅ 𝟭 (WithTerminal C) :=
  NatIso.ofComponents (fun X => match X with
    | of _ => Iso.refl _
                              /-
                                C✝ : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C✝
                                C : Type u_1
                                inst✝ : CategoryTheory.Category.{?u.89123, u_1} C
                                ⊢ ∀ {X Y : CategoryTheory.WithTerminal C} (f : Quiver.Hom X Y), Eq (CategoryTh …
                              -/
    | star => Iso.refl _) (by aesop_cat)
                              /-
                                🎉 no goals
                              -/


/-- A natural isomorphism between the functor `map (F ⋙ G) ` and `map F ⋙ map G `. -/
@[simps!]
def mapComp {D E : Type*} [Category D] [Category E] (F : C ⥤ D) (G : D ⥤ E) :
    map (F ⋙ G) ≅ map F ⋙ map G :=
  NatIso.ofComponents (fun X => match X with
    | of _ => Iso.refl _
                              /-
                                C : Type u
                                inst✝² : CategoryTheory.Category.{v, u} C
                                D : Type u_1
                                E : Type u_2
                                inst✝¹ : CategoryTheory.Category.{?u.105335, u_1} D
                                inst✝ : CategoryTheory.Category.{?u.105339, u_2} E
                                F : CategoryTheory.Functor C D
                                G : CategoryTheory.Functor D E
                                ⊢ ∀ {X Y : CategoryTheory.WithTerminal C} (f : Quiver.Hom X Y), Eq (CategoryTh …
                              -/
    | star => Iso.refl _) (by aesop_cat)
                              /-
                                🎉 no goals
                              -/


/-- From a natural transformation of functors `C ⥤ D`, the induced natural transformation
of functors `WithTerminal C ⥤ WithTerminal D`. -/
@[simps]
def map₂ {D : Type*} [Category D] {F G : C ⥤ D} (η : F ⟶ G) : map F ⟶ map G where
  app := fun X => match X with
    | of x => η.app x
    | star => 𝟙 star
  naturality := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝ : CategoryTheory.Category.{?u.146250, u_1} D
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      ⊢ ∀ ⦃X Y : CategoryTheory.WithTerminal C⦄ (f : Quiver.Hom X Y), Eq (CategoryTh …
    -/
    intro X Y f
    match X, Y, f with
    | of x, of y, f => exact η.naturality f
    | of x, star, _ => rfl
    | star, star, _ => rfl

-- Note: ...

/-- The prelax functor from `Cat` to `Cat` defined with `WithTerminal`. -/
@[simps]
def prelaxfunctor : PrelaxFunctor Cat Cat where
  obj C := Cat.of (WithTerminal C)
  map := map
  map₂ := map₂
  map₂_id := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b : CategoryTheory.Cat} (f : Quiver.Hom a b), Eq ({ obj := fun C => Cat …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq ({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithTerminal ↑C) …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq ({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithTerminal ↑C) …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      X : ↑({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithTerminal ↑C) …
      ⊢ Eq (({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithTerminal ↑C …
    -/
                /-
                  🎉 no goals
                -/
    cases X <;> rfl
                /-
                  🎉 no goals
                -/
  map₂_comp := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b : CategoryTheory.Cat} {f g h : Quiver.Hom a b} (η : Quiver.Hom f g) ( …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ g✝ h✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      θ✝ : Quiver.Hom g✝ h✝
      ⊢ Eq ({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithTerminal ↑C) …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ g✝ h✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      θ✝ : Quiver.Hom g✝ h✝
      ⊢ Eq ({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithTerminal ↑C) …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ g✝ h✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      θ✝ : Quiver.Hom g✝ h✝
      X : ↑({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithTerminal ↑C) …
      ⊢ Eq (({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithTerminal ↑C …
    -/
                /-
                  🎉 no goals
                -/
    cases X <;> rfl
                /-
                  🎉 no goals
                -/


/-- The pseudofunctor from `Cat` to `Cat` defined with `WithTerminal`. -/
@[simps]
def pseudofunctor : Pseudofunctor Cat Cat where
  toPrelaxFunctor := prelaxfunctor
  mapId C := mapId C
  mapComp := mapComp
  map₂_whisker_left := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b c : CategoryTheory.Cat} (f : Quiver.Hom a b) {g h : Quiver.Hom b c} ( …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ h✝ : Quiver.Hom b✝ c✝
      η✝ : Quiver.Hom g✝ h✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ h✝ : Quiver.Hom b✝ c✝
      η✝ : Quiver.Hom g✝ h✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ h✝ : Quiver.Hom b✝ c✝
      η✝ : Quiver.Hom g✝ h✝
      X : ↑(CategoryTheory.WithTerminal.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        g✝ h✝ : Quiver.Hom b✝ c✝
        η✝ : Quiver.Hom g✝ h✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
      -/
    · rw [NatTrans.comp_app, NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, Cat.whiskerLeft_app, mapComp_hom_app,
        Iso.refl_hom, mapComp_inv_app, Iso.refl_inv, Category.comp_id, Category.id_comp]
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ c✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝ b✝
        g✝ h✝ : Quiver.Hom b✝ c✝
        η✝ : Quiver.Hom g✝ h✝
        ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  map₂_whisker_right := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b c : CategoryTheory.Cat} {f g : Quiver.Hom a b} (η : Quiver.Hom f g) ( …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ g✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      h✝ : Quiver.Hom b✝ c✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ g✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      h✝ : Quiver.Hom b✝ c✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ g✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      h✝ : Quiver.Hom b✝ c✝
      X : ↑(CategoryTheory.WithTerminal.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ : CategoryTheory.Cat
        f✝ g✝ : Quiver.Hom a✝¹ b✝
        η✝ : Quiver.Hom f✝ g✝
        h✝ : Quiver.Hom b✝ c✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
      -/
    · rw [NatTrans.comp_app, NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, Cat.whiskerRight_app, mapComp_hom_app,
        Iso.refl_hom, map_map, mapComp_inv_app, Iso.refl_inv, Category.comp_id, Category.id_comp]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ : CategoryTheory.Cat
        f✝ g✝ : Quiver.Hom a✝¹ b✝
        η✝ : Quiver.Hom f✝ g✝
        h✝ : Quiver.Hom b✝ c✝
        a✝ : ↑a✝¹
        ⊢ Eq (h✝.map (η✝.app a✝)) (h✝.map (CategoryTheory.WithTerminal.down (η✝.app a✝ …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ c✝ : CategoryTheory.Cat
        f✝ g✝ : Quiver.Hom a✝ b✝
        η✝ : Quiver.Hom f✝ g✝
        h✝ : Quiver.Hom b✝ c✝
        ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  map₂_associator := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b c d : CategoryTheory.Cat} (f : Quiver.Hom a b) (g : Quiver.Hom b c) ( …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ d✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ : Quiver.Hom b✝ c✝
      h✝ : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    dsimp
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ d✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ : Quiver.Hom b✝ c✝
      h✝ : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.WithTerminal.map₂ (CategoryTheory.Bicategory.associator f …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ d✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ : Quiver.Hom b✝ c✝
      h✝ : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.WithTerminal.map₂ (CategoryTheory.Bicategory.associator f …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ d✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ : Quiver.Hom b✝ c✝
      h✝ : Quiver.Hom c✝ d✝
      X : CategoryTheory.WithTerminal ↑a✝
      ⊢ Eq ((CategoryTheory.WithTerminal.map₂ (CategoryTheory.Bicategory.associator  …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ d✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        g✝ : Quiver.Hom b✝ c✝
        h✝ : Quiver.Hom c✝ d✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithTerminal.map₂ (CategoryTheory.Bicategory.associator  …
      -/
    · rw [NatTrans.comp_app,NatTrans.comp_app,NatTrans.comp_app,NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        Bicategory.Strict.associator_eqToIso, eqToIso_refl, Iso.refl_hom,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, mapComp_hom_app, Cat.whiskerRight_app,
        map_map, down_id, Functor.map_id, Cat.whiskerLeft_app, mapComp_inv_app, Iso.refl_inv,
        Category.comp_id, Category.id_comp]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ d✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        g✝ : Quiver.Hom b✝ c✝
        h✝ : Quiver.Hom c✝ d✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.CategoryStruct.id (CategoryTheory.CategoryStruct.comp (C …
      -/
      rw [NatTrans.id_app, NatTrans.id_app]
      simp only [Cat.comp_obj, Bicategory.whiskerRight, whiskerRight_app, map_obj, mapComp_hom_app,
        Iso.refl_hom, map_map, down_id, Functor.map_id, Bicategory.whiskerLeft, whiskerLeft_app,
        mapComp_inv_app, Iso.refl_inv, Category.comp_id]
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ c✝ d✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝ b✝
        g✝ : Quiver.Hom b✝ c✝
        h✝ : Quiver.Hom c✝ d✝
        ⊢ Eq ((CategoryTheory.WithTerminal.map₂ (CategoryTheory.Bicategory.associator  …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  map₂_left_unitor := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b : CategoryTheory.Cat} (f : Quiver.Hom a b), Eq (CategoryTheory.WithTe …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      X : ↑(CategoryTheory.WithTerminal.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
      -/
    · rw [NatTrans.comp_app, NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        Bicategory.Strict.leftUnitor_eqToIso, eqToIso_refl, Iso.refl_hom,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, mapComp_hom_app, Cat.whiskerRight_app,
        mapId_hom_app, map_map, Category.id_comp]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.CategoryStruct.id (CategoryTheory.CategoryStruct.comp (C …
      -/
      rw [NatTrans.id_app, NatTrans.id_app]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct.comp (C …
      -/
      simp only [Cat.comp_obj, map_obj, Category.comp_id]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id (f✝.obj ((CategoryTheory.CategoryStruct …
      -/
      rw [← Functor.map_id]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (f✝.map (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝ b✝
        ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  map₂_right_unitor := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b : CategoryTheory.Cat} (f : Quiver.Hom a b), Eq (CategoryTheory.WithTe …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      X : ↑(CategoryTheory.WithTerminal.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
      -/
    · rw [NatTrans.comp_app, NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        Bicategory.Strict.rightUnitor_eqToIso, eqToIso_refl, Iso.refl_hom,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, mapComp_hom_app, Cat.whiskerLeft_app,
        mapId_hom_app, Category.id_comp]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.CategoryStruct.id (CategoryTheory.CategoryStruct.comp f✝ …
      -/
      rw [NatTrans.id_app, NatTrans.id_app]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct.comp f✝ …
      -/
      simp only [Cat.comp_obj, map_obj, Category.comp_id]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct.id b✝). …
      -/
      rw [← Functor.map_id]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.CategoryStruct.id b✝).map (CategoryTheory.CategoryStruct …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝ b✝
        ⊢ Eq ((CategoryTheory.WithTerminal.prelaxfunctor.map₂ (CategoryTheory.Bicatego …
      -/
    · rfl
      /-
        🎉 no goals
      -/


instance {X : WithTerminal C} : Unique (X ⟶ star) where
  default :=
    match X with
    | of _ => PUnit.unit
    | star => PUnit.unit
             /-
               C : Type u
               inst✝ : CategoryTheory.Category.{v, u} C
               X : CategoryTheory.WithTerminal C
               ⊢ ∀ (a : Quiver.Hom X CategoryTheory.WithTerminal.star), Eq a Inhabited.default
             -/
  uniq := by aesop_cat
             /-
               🎉 no goals
             -/


/-- `WithTerminal.star` is terminal. -/
def starTerminal : Limits.IsTerminal (star : WithTerminal C) :=
  Limits.IsTerminal.ofUnique _


/-- Lift a functor `F : C ⥤ D` to `WithTerminal C ⥤ D`. -/
@[simps]
def lift {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, F.obj x ⟶ Z)
    (hM : ∀ (x y : C) (f : x ⟶ y), F.map f ≫ M y = M x) : WithTerminal C ⥤ D where
  obj X :=
    match X with
    | of x => F.obj x
    | star => Z
  map {X Y} f :=
    match X, Y, f with
    | of _, of _, f => F.map (down f)
    | of x, star, _ => M x
    | star, star, _ => 𝟙 Z


/-- The isomorphism between `incl ⋙ lift F _ _` with `F`. -/
@[simps!]
def inclLift {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, F.obj x ⟶ Z)
    (hM : ∀ (x y : C) (f : x ⟶ y), F.map f ≫ M y = M x) : incl ⋙ lift F M hM ≅ F where
  hom := { app := fun _ => 𝟙 _ }
  inv := { app := fun _ => 𝟙 _ }


/-- The isomorphism between `(lift F _ _).obj WithTerminal.star` with `Z`. -/
@[simps!]
def liftStar {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, F.obj x ⟶ Z)
    (hM : ∀ (x y : C) (f : x ⟶ y), F.map f ≫ M y = M x) : (lift F M hM).obj star ≅ Z :=
  eqToIso rfl


theorem lift_map_liftStar {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, F.obj x ⟶ Z)
    (hM : ∀ (x y : C) (f : x ⟶ y), F.map f ≫ M y = M x) (x : C) :
    (lift F M hM).map (starTerminal.from (incl.obj x)) ≫ (liftStar F M hM).hom =
      (inclLift F M hM).hom.app x ≫ M x := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    Z : D
    F : CategoryTheory.Functor C D
    M : (x : C) → Quiver.Hom (F.obj x) Z
    hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
    x : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.WithTerminal.lift F  …
  -/
  erw [Category.id_comp, Category.comp_id]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    Z : D
    F : CategoryTheory.Functor C D
    M : (x : C) → Quiver.Hom (F.obj x) Z
    hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
    x : C
    ⊢ Eq ((CategoryTheory.WithTerminal.lift F M hM).map (CategoryTheory.WithTermin …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The uniqueness of `lift`. -/
@[simp]
def liftUnique {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, F.obj x ⟶ Z)
    (hM : ∀ (x y : C) (f : x ⟶ y), F.map f ≫ M y = M x)
    (G : WithTerminal C ⥤ D) (h : incl ⋙ G ≅ F)
    (hG : G.obj star ≅ Z)
    (hh : ∀ x : C, G.map (starTerminal.from (incl.obj x)) ≫ hG.hom = h.hom.app x ≫ M x) :
    G ≅ lift F M hM :=
  NatIso.ofComponents
    (fun X =>
      match X with
      | of x => h.app x
      | star => hG)
    (by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.263728, u_1} D
        Z : D
        F : CategoryTheory.Functor C D
        M : (x : C) → Quiver.Hom (F.obj x) Z
        hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
        G : CategoryTheory.Functor (CategoryTheory.WithTerminal C) D
        h : CategoryTheory.Iso (CategoryTheory.WithTerminal.incl.comp G) F
        hG : CategoryTheory.Iso (G.obj CategoryTheory.WithTerminal.star) Z
        hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory. …
        ⊢ ∀ {X Y : CategoryTheory.WithTerminal C} (f : Quiver.Hom X Y), Eq (CategoryTh …
      -/
      rintro (X | X) (Y | Y) f
        /-
          case of.of
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.263728, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom (F.obj x) Z
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithTerminal C) D
          h : CategoryTheory.Iso (CategoryTheory.WithTerminal.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithTerminal.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory. …
          X Y : C
          f : Quiver.Hom (CategoryTheory.WithTerminal.of X) (CategoryTheory.WithTerminal …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.W …
        -/
      · apply h.hom.naturality
        /-
          🎉 no goals
        -/
        /-
          case of.star
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.263728, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom (F.obj x) Z
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithTerminal C) D
          h : CategoryTheory.Iso (CategoryTheory.WithTerminal.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithTerminal.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory. …
          X : C
          f : Quiver.Hom (CategoryTheory.WithTerminal.of X) CategoryTheory.WithTerminal. …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.W …
        -/
      · cases f
        /-
          case of.star.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.263728, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom (F.obj x) Z
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithTerminal C) D
          h : CategoryTheory.Iso (CategoryTheory.WithTerminal.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithTerminal.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory. …
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map PUnit.unit) ((fun X => Categor …
        -/
        exact hh _
        /-
          🎉 no goals
        -/
        /-
          case star.of
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.263728, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom (F.obj x) Z
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithTerminal C) D
          h : CategoryTheory.Iso (CategoryTheory.WithTerminal.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithTerminal.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory. …
          Y : C
          f : Quiver.Hom CategoryTheory.WithTerminal.star (CategoryTheory.WithTerminal.o …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.W …
        -/
      · cases f
        /-
          🎉 no goals
        -/
        /-
          case star.star
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.263728, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom (F.obj x) Z
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithTerminal C) D
          h : CategoryTheory.Iso (CategoryTheory.WithTerminal.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithTerminal.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory. …
          f : Quiver.Hom CategoryTheory.WithTerminal.star CategoryTheory.WithTerminal.star
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.W …
        -/
      · cases f
        /-
          case star.star.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.263728, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom (F.obj x) Z
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithTerminal C) D
          h : CategoryTheory.Iso (CategoryTheory.WithTerminal.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithTerminal.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory. …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map PUnit.unit) ((fun X => Categor …
        -/
        change G.map (𝟙 _) ≫ hG.hom = hG.hom ≫ 𝟙 _
        /-
          case star.star.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.263728, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom (F.obj x) Z
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithTerminal C) D
          h : CategoryTheory.Iso (CategoryTheory.WithTerminal.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithTerminal.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory. …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStruct …
        -/
        simp)
        /-
          🎉 no goals
        -/


/-- A variant of `lift` with `Z` a terminal object. -/
@[simps!]
def liftToTerminal {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (hZ : Limits.IsTerminal Z) :
    WithTerminal C ⥤ D :=
  lift F (fun _x => hZ.from _) fun _x _y _f => hZ.hom_ext _ _


/-- A variant of `incl_lift` with `Z` a terminal object. -/
@[simps!]
def inclLiftToTerminal {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (hZ : Limits.IsTerminal Z) :
    incl ⋙ liftToTerminal F hZ ≅ F :=
  inclLift _ _ _


/-- A variant of `lift_unique` with `Z` a terminal object. -/
@[simps!]
def liftToTerminalUnique {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (hZ : Limits.IsTerminal Z)
    (G : WithTerminal C ⥤ D) (h : incl ⋙ G ≅ F) (hG : G.obj star ≅ Z) : G ≅ liftToTerminal F hZ :=
  liftUnique F (fun _z => hZ.from _) (fun _x _y _f => hZ.hom_ext _ _) G h hG fun _x =>
    hZ.hom_ext _ _


/-- Constructs a morphism to `star` from `of X`. -/
@[simp]
def homFrom (X : C) : incl.obj X ⟶ star :=
  starTerminal.from _


instance isIso_of_from_star {X : WithTerminal C} (f : star ⟶ X) : IsIso f :=
  match X with
  | of _X => f.elim
  | star => ⟨f, rfl, rfl⟩


/-- Morphisms for `WithInitial C`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `nolint has_nonempty_instance`
@[simp]
def Hom : WithInitial C → WithInitial C → Type v
  | of X, of Y => X ⟶ Y
  | of _, _ => PEmpty
  | star, _ => PUnit

/-- Identity morphisms for `WithInitial C`. -/
@[simp]
def id : ∀ X : WithInitial C, Hom X X
  | of _ => 𝟙 _
  | star => PUnit.unit


/-- Composition of morphisms for `WithInitial C`. -/
@[simp]
def comp : ∀ {X Y Z : WithInitial C}, Hom X Y → Hom Y Z → Hom X Z
  | of _X, of _Y, of _Z => fun f g => f ≫ g
  | star, _, of _X => fun _f _g => PUnit.unit
  | _, of _X, star => fun _f g => PEmpty.elim g
  | of _Y, star, _ => fun f _g => PEmpty.elim f
  | star, star, star => fun _ _ => PUnit.unit

instance : Category.{v} (WithInitial C) where
  Hom X Y := Hom X Y
  id X := id X
  comp f g := comp f g
  assoc {a b c d} f g h := by
    -- Porting note: it would be nice to automate this away as well.
    -- See the note on `Category (WithTerminal C)`
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a b c d : CategoryTheory.WithInitial C
      f : Quiver.Hom a b
      g : Quiver.Hom b c
      h : Quiver.Hom c d
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
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
    cases a <;> cases b <;> cases c <;> cases d <;> try aesop_cat
                                                    /-
                                                      🎉 no goals
                                                    -/
      /-
        case of.of.star.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝² a✝¹ : C
        f : Quiver.Hom (CategoryTheory.WithInitial.of a✝²) (CategoryTheory.WithInitial …
        g : Quiver.Hom (CategoryTheory.WithInitial.of a✝¹) CategoryTheory.WithInitial. …
        a✝ : C
        h : Quiver.Hom CategoryTheory.WithInitial.star (CategoryTheory.WithInitial.of  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · exact (g : PEmpty).elim
      /-
        🎉 no goals
      -/
      /-
        case of.star.of.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝² : C
        f : Quiver.Hom (CategoryTheory.WithInitial.of a✝²) CategoryTheory.WithInitial. …
        a✝¹ : C
        g : Quiver.Hom CategoryTheory.WithInitial.star (CategoryTheory.WithInitial.of  …
        a✝ : C
        h : Quiver.Hom (CategoryTheory.WithInitial.of a✝¹) (CategoryTheory.WithInitial …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · exact (f : PEmpty).elim
      /-
        🎉 no goals
      -/
      /-
        case of.star.star.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ : C
        f : Quiver.Hom (CategoryTheory.WithInitial.of a✝¹) CategoryTheory.WithInitial. …
        g : Quiver.Hom CategoryTheory.WithInitial.star CategoryTheory.WithInitial.star
        a✝ : C
        h : Quiver.Hom CategoryTheory.WithInitial.star (CategoryTheory.WithInitial.of  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · exact (f : PEmpty).elim
      /-
        🎉 no goals
      -/


@[aesop safe destruct (rule_sets := [CategoryTheory])]
lemma false_of_to_star {X : C} (f : of X ⟶ star) : False := (f : PEmpty).elim


/-- The inclusion of `C` into `WithInitial C`. -/
def incl : C ⥤ WithInitial C where
  obj := of
  map f := f


/-- Map `WithInitial` with respect to a functor `F : C ⥤ D`. -/
@[simps]
def map {D : Type*} [Category D] (F : C ⥤ D) : WithInitial C ⥤ WithInitial D where
  obj X :=
    match X with
    | of x => of <| F.obj x
    | star => star
  map {X Y} f :=
    match X, Y, f with
    | of _, of _, f => F.map (down f)
    | star, of _, _ => PUnit.unit
    | star, star, _ => PUnit.unit


/-- A natural isomorphism between the functor `map (𝟭 C)` and `𝟭 (WithInitial C)`. -/
@[simps!]
def mapId (C : Type*) [Category C] : map (𝟭 C) ≅ 𝟭 (WithInitial C) :=
  NatIso.ofComponents (fun X => match X with
    | of _ => Iso.refl _
                              /-
                                C✝ : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C✝
                                C : Type u_1
                                inst✝ : CategoryTheory.Category.{?u.361858, u_1} C
                                ⊢ ∀ {X Y : CategoryTheory.WithInitial C} (f : Quiver.Hom X Y), Eq (CategoryThe …
                              -/
    | star => Iso.refl _) (by aesop_cat)
                              /-
                                🎉 no goals
                              -/


/-- A natural isomorphism between the functor `map (F ⋙ G) ` and `map F ⋙ map G `. -/
@[simps!]
def mapComp {D E : Type*} [Category D] [Category E] (F : C ⥤ D) (G : D ⥤ E) :
    map (F ⋙ G) ≅ map F ⋙ map G :=
  NatIso.ofComponents (fun X => match X with
    | of _ => Iso.refl _
                              /-
                                C : Type u
                                inst✝² : CategoryTheory.Category.{v, u} C
                                D : Type u_1
                                E : Type u_2
                                inst✝¹ : CategoryTheory.Category.{?u.378103, u_1} D
                                inst✝ : CategoryTheory.Category.{?u.378107, u_2} E
                                F : CategoryTheory.Functor C D
                                G : CategoryTheory.Functor D E
                                ⊢ ∀ {X Y : CategoryTheory.WithInitial C} (f : Quiver.Hom X Y), Eq (CategoryThe …
                              -/
    | star => Iso.refl _) (by aesop_cat)
                              /-
                                🎉 no goals
                              -/


/-- From a natural transformation of functors `C ⥤ D`, the induced natural transformation
of functors `WithInitial C ⥤ WithInitial D`. -/
@[simps]
def map₂ {D : Type*} [Category D] {F G : C ⥤ D} (η : F ⟶ G) : map F ⟶ map G where
  app := fun X => match X with
    | of x => η.app x
    | star => 𝟙 star
  naturality := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u_1
      inst✝ : CategoryTheory.Category.{?u.419084, u_1} D
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      ⊢ ∀ ⦃X Y : CategoryTheory.WithInitial C⦄ (f : Quiver.Hom X Y), Eq (CategoryThe …
    -/
    intro X Y f
    match X, Y, f with
    | of x, of y, f => exact η.naturality f
    | star, of x, _ => rfl
    | star, star, _ => rfl


/-- The prelax functor from `Cat` to `Cat` defined with `WithInitial`. -/
@[simps]
def prelaxfunctor : PrelaxFunctor Cat Cat where
  obj C := Cat.of (WithInitial C)
  map := map
  map₂ := map₂
  map₂_id := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b : CategoryTheory.Cat} (f : Quiver.Hom a b), Eq ({ obj := fun C => Cat …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq ({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithInitial ↑C), …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq ({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithInitial ↑C), …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      X : ↑({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithInitial ↑C), …
      ⊢ Eq (({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithInitial ↑C) …
    -/
                /-
                  🎉 no goals
                -/
    cases X <;> rfl
                /-
                  🎉 no goals
                -/
  map₂_comp := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b : CategoryTheory.Cat} {f g h : Quiver.Hom a b} (η : Quiver.Hom f g) ( …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ g✝ h✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      θ✝ : Quiver.Hom g✝ h✝
      ⊢ Eq ({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithInitial ↑C), …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ g✝ h✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      θ✝ : Quiver.Hom g✝ h✝
      ⊢ Eq ({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithInitial ↑C), …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ g✝ h✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      θ✝ : Quiver.Hom g✝ h✝
      X : ↑({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithInitial ↑C), …
      ⊢ Eq (({ obj := fun C => CategoryTheory.Cat.of (CategoryTheory.WithInitial ↑C) …
    -/
                /-
                  🎉 no goals
                -/
    cases X <;> rfl
                /-
                  🎉 no goals
                -/


/-- The pseudofunctor from `Cat` to `Cat` defined with `WithInitial`. -/
@[simps]
def pseudofunctor : Pseudofunctor Cat Cat where
  toPrelaxFunctor := prelaxfunctor
  mapId C := mapId C
  mapComp := mapComp
  map₂_whisker_left := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b c : CategoryTheory.Cat} (f : Quiver.Hom a b) {g h : Quiver.Hom b c} ( …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ h✝ : Quiver.Hom b✝ c✝
      η✝ : Quiver.Hom g✝ h✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ h✝ : Quiver.Hom b✝ c✝
      η✝ : Quiver.Hom g✝ h✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ h✝ : Quiver.Hom b✝ c✝
      η✝ : Quiver.Hom g✝ h✝
      X : ↑(CategoryTheory.WithInitial.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        g✝ h✝ : Quiver.Hom b✝ c✝
        η✝ : Quiver.Hom g✝ h✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rw [NatTrans.comp_app, NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, Cat.whiskerLeft_app, mapComp_hom_app,
        Iso.refl_hom, mapComp_inv_app, Iso.refl_inv, Category.comp_id, Category.id_comp]
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ c✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝ b✝
        g✝ h✝ : Quiver.Hom b✝ c✝
        η✝ : Quiver.Hom g✝ h✝
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  map₂_whisker_right := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b c : CategoryTheory.Cat} {f g : Quiver.Hom a b} (η : Quiver.Hom f g) ( …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ g✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      h✝ : Quiver.Hom b✝ c✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ g✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      h✝ : Quiver.Hom b✝ c✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ : CategoryTheory.Cat
      f✝ g✝ : Quiver.Hom a✝ b✝
      η✝ : Quiver.Hom f✝ g✝
      h✝ : Quiver.Hom b✝ c✝
      X : ↑(CategoryTheory.WithInitial.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ : CategoryTheory.Cat
        f✝ g✝ : Quiver.Hom a✝¹ b✝
        η✝ : Quiver.Hom f✝ g✝
        h✝ : Quiver.Hom b✝ c✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rw [NatTrans.comp_app, NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, Cat.whiskerRight_app, mapComp_hom_app,
        Iso.refl_hom, map_map, mapComp_inv_app, Iso.refl_inv, Category.comp_id, Category.id_comp]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ : CategoryTheory.Cat
        f✝ g✝ : Quiver.Hom a✝¹ b✝
        η✝ : Quiver.Hom f✝ g✝
        h✝ : Quiver.Hom b✝ c✝
        a✝ : ↑a✝¹
        ⊢ Eq (h✝.map (η✝.app a✝)) (h✝.map (CategoryTheory.WithInitial.down (η✝.app a✝)))
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ c✝ : CategoryTheory.Cat
        f✝ g✝ : Quiver.Hom a✝ b✝
        η✝ : Quiver.Hom f✝ g✝
        h✝ : Quiver.Hom b✝ c✝
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  map₂_associator := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b c d : CategoryTheory.Cat} (f : Quiver.Hom a b) (g : Quiver.Hom b c) ( …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ d✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ : Quiver.Hom b✝ c✝
      h✝ : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ d✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ : Quiver.Hom b✝ c✝
      h✝ : Quiver.Hom c✝ d✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ c✝ d✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      g✝ : Quiver.Hom b✝ c✝
      h✝ : Quiver.Hom c✝ d✝
      X : ↑(CategoryTheory.WithInitial.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ d✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        g✝ : Quiver.Hom b✝ c✝
        h✝ : Quiver.Hom c✝ d✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rw [NatTrans.comp_app,NatTrans.comp_app,NatTrans.comp_app,NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        Bicategory.Strict.associator_eqToIso, eqToIso_refl, Iso.refl_hom,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, mapComp_hom_app, Cat.whiskerRight_app,
        map_map, down_id, Functor.map_id, Cat.whiskerLeft_app, mapComp_inv_app, Iso.refl_inv,
        Category.comp_id, Category.id_comp]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ d✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        g✝ : Quiver.Hom b✝ c✝
        h✝ : Quiver.Hom c✝ d✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.CategoryStruct.id (CategoryTheory.CategoryStruct.comp (C …
      -/
      rw [NatTrans.id_app, NatTrans.id_app]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ c✝ d✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        g✝ : Quiver.Hom b✝ c✝
        h✝ : Quiver.Hom c✝ d✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct.comp (C …
      -/
      simp only [Cat.comp_obj, map_obj, Category.comp_id]
      /-
        🎉 no goals
      -/
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ c✝ d✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝ b✝
        g✝ : Quiver.Hom b✝ c✝
        h✝ : Quiver.Hom c✝ d✝
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  map₂_left_unitor := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b : CategoryTheory.Cat} (f : Quiver.Hom a b), Eq (CategoryTheory.WithIn …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      X : ↑(CategoryTheory.WithInitial.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rw [NatTrans.comp_app, NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        Bicategory.Strict.leftUnitor_eqToIso, eqToIso_refl, Iso.refl_hom,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, mapComp_hom_app, Cat.whiskerRight_app,
        mapId_hom_app, map_map, Category.id_comp]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.CategoryStruct.id (CategoryTheory.CategoryStruct.comp (C …
      -/
      rw [NatTrans.id_app, NatTrans.id_app]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct.comp (C …
      -/
      simp only [Cat.comp_obj, map_obj, Category.comp_id]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id (f✝.obj ((CategoryTheory.CategoryStruct …
      -/
      rw [← Functor.map_id]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (f✝.map (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝ b✝
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  map₂_right_unitor := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {a b : CategoryTheory.Cat} (f : Quiver.Hom a b), Eq (CategoryTheory.WithIn …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    apply NatTrans.ext
    /-
      case app
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategory …
    -/
    funext X
    /-
      case app.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      a✝ b✝ : CategoryTheory.Cat
      f✝ : Quiver.Hom a✝ b✝
      X : ↑(CategoryTheory.WithInitial.prelaxfunctor.obj a✝)
      ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
    -/
    cases X
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rw [NatTrans.comp_app, NatTrans.comp_app]
      simp only [prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_obj,
        prelaxfunctor_toPrelaxFunctorStruct_toPrefunctor_map, map_obj, Cat.comp_obj,
        Bicategory.Strict.rightUnitor_eqToIso, eqToIso_refl, Iso.refl_hom,
        prelaxfunctor_toPrelaxFunctorStruct_map₂, map₂_app, mapComp_hom_app, Cat.whiskerLeft_app,
        mapId_hom_app, Category.id_comp]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq ((CategoryTheory.CategoryStruct.id (CategoryTheory.CategoryStruct.comp f✝ …
      -/
      rw [NatTrans.id_app, NatTrans.id_app]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct.comp f✝ …
      -/
      simp only [Cat.comp_obj, map_obj, Category.comp_id]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.CategoryStruct.id b✝). …
      -/
      rw [← Functor.map_id, Cat.id_map]
      /-
        case app.h.of
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝¹ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝¹ b✝
        a✝ : ↑a✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.id (f✝.obj a✝)) (CategoryTheory.CategorySt …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case app.h.star
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        a✝ b✝ : CategoryTheory.Cat
        f✝ : Quiver.Hom a✝ b✝
        ⊢ Eq ((CategoryTheory.WithInitial.prelaxfunctor.map₂ (CategoryTheory.Bicategor …
      -/
    · rfl
      /-
        🎉 no goals
      -/


instance {X : WithInitial C} : Unique (star ⟶ X) where
  default :=
    match X with
    | of _x => PUnit.unit
    | star => PUnit.unit
             /-
               C : Type u
               inst✝ : CategoryTheory.Category.{v, u} C
               X : CategoryTheory.WithInitial C
               ⊢ ∀ (a : Quiver.Hom CategoryTheory.WithInitial.star X), Eq a Inhabited.default
             -/
  uniq := by aesop_cat
             /-
               🎉 no goals
             -/


/-- `WithInitial.star` is initial. -/
def starInitial : Limits.IsInitial (star : WithInitial C) :=
  Limits.IsInitial.ofUnique _


/-- Lift a functor `F : C ⥤ D` to `WithInitial C ⥤ D`. -/
@[simps]
def lift {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, Z ⟶ F.obj x)
    (hM : ∀ (x y : C) (f : x ⟶ y), M x ≫ F.map f = M y) : WithInitial C ⥤ D where
  obj X :=
    match X with
    | of x => F.obj x
    | star => Z
  map {X Y} f :=
    match X, Y, f with
    | of _, of _, f => F.map (down f)
    | star, of _, _ => M _
    | star, star, _ => 𝟙 _


/-- The isomorphism between `incl ⋙ lift F _ _` with `F`. -/
@[simps!]
def inclLift {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, Z ⟶ F.obj x)
    (hM : ∀ (x y : C) (f : x ⟶ y), M x ≫ F.map f = M y) : incl ⋙ lift F M hM ≅ F where
  hom := { app := fun _ => 𝟙 _ }
  inv := { app := fun _ => 𝟙 _ }


/-- The isomorphism between `(lift F _ _).obj WithInitial.star` with `Z`. -/
@[simps!]
def liftStar {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, Z ⟶ F.obj x)
    (hM : ∀ (x y : C) (f : x ⟶ y), M x ≫ F.map f = M y) : (lift F M hM).obj star ≅ Z :=
  eqToIso rfl


theorem liftStar_lift_map {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, Z ⟶ F.obj x)
    (hM : ∀ (x y : C) (f : x ⟶ y), M x ≫ F.map f = M y) (x : C) :
    (liftStar F M hM).hom ≫ (lift F M hM).map (starInitial.to (incl.obj x)) =
      M x ≫ (inclLift F M hM).hom.app x := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    Z : D
    F : CategoryTheory.Functor C D
    M : (x : C) → Quiver.Hom Z (F.obj x)
    hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
    x : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.WithInitial.liftStar  …
  -/
  erw [Category.id_comp, Category.comp_id]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    Z : D
    F : CategoryTheory.Functor C D
    M : (x : C) → Quiver.Hom Z (F.obj x)
    hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
    x : C
    ⊢ Eq ((CategoryTheory.WithInitial.lift F M hM).map (CategoryTheory.WithInitial …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The uniqueness of `lift`. -/
@[simp]
def liftUnique {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (M : ∀ x : C, Z ⟶ F.obj x)
    (hM : ∀ (x y : C) (f : x ⟶ y), M x ≫ F.map f = M y)
    (G : WithInitial C ⥤ D) (h : incl ⋙ G ≅ F)
    (hG : G.obj star ≅ Z)
    (hh : ∀ x : C, hG.symm.hom ≫ G.map (starInitial.to (incl.obj x)) = M x ≫ h.symm.hom.app x) :
    G ≅ lift F M hM :=
  NatIso.ofComponents
    (fun X =>
      match X with
      | of x => h.app x
      | star => hG)
    (by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u_1
        inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
        Z : D
        F : CategoryTheory.Functor C D
        M : (x : C) → Quiver.Hom Z (F.obj x)
        hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
        G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
        h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
        hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
        hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
        ⊢ ∀ {X Y : CategoryTheory.WithInitial C} (f : Quiver.Hom X Y), Eq (CategoryThe …
      -/
      rintro (X | X) (Y | Y) f
        /-
          case of.of
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          X Y : C
          f : Quiver.Hom (CategoryTheory.WithInitial.of X) (CategoryTheory.WithInitial.o …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.W …
        -/
      · apply h.hom.naturality
        /-
          🎉 no goals
        -/
        /-
          case of.star
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          X : C
          f : Quiver.Hom (CategoryTheory.WithInitial.of X) CategoryTheory.WithInitial.star
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.W …
        -/
      · cases f
        /-
          🎉 no goals
        -/
        /-
          case star.of
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          Y : C
          f : Quiver.Hom CategoryTheory.WithInitial.star (CategoryTheory.WithInitial.of Y)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.W …
        -/
      · cases f
        /-
          case star.of.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          Y : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map PUnit.unit) ((fun X => Categor …
        -/
        change G.map _ ≫ h.hom.app _ = hG.hom ≫ _
        /-
          case star.of.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          Y : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map PUnit.unit) (h.hom.app Y)) (Ca …
        -/
        symm
        /-
          case star.of.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          Y : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp hG.hom ((CategoryTheory.WithInitial.l …
        -/
        erw [← Iso.eq_inv_comp, ← Category.assoc, hh]
        /-
          case star.of.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          Y : C
          ⊢ Eq ((CategoryTheory.WithInitial.lift F M hM).map PUnit.unit) (CategoryTheory …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case star.star
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          f : Quiver.Hom CategoryTheory.WithInitial.star CategoryTheory.WithInitial.star
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.W …
        -/
      · cases f
        /-
          case star.star.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map PUnit.unit) ((fun X => Categor …
        -/
        change G.map (𝟙 _) ≫ hG.hom = hG.hom ≫ 𝟙 _
        /-
          case star.star.unit
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.540010, u_1} D
          Z : D
          F : CategoryTheory.Functor C D
          M : (x : C) → Quiver.Hom Z (F.obj x)
          hM : ∀ (x y : C) (f : Quiver.Hom x y), Eq (CategoryTheory.CategoryStruct.comp  …
          G : CategoryTheory.Functor (CategoryTheory.WithInitial C) D
          h : CategoryTheory.Iso (CategoryTheory.WithInitial.incl.comp G) F
          hG : CategoryTheory.Iso (G.obj CategoryTheory.WithInitial.star) Z
          hh : ∀ (x : C), Eq (CategoryTheory.CategoryStruct.comp hG.symm.hom (G.map (Cat …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStruct …
        -/
        simp)
        /-
          🎉 no goals
        -/


/-- A variant of `lift` with `Z` an initial object. -/
@[simps!]
def liftToInitial {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (hZ : Limits.IsInitial Z) :
    WithInitial C ⥤ D :=
  lift F (fun _x => hZ.to _) fun _x _y _f => hZ.hom_ext _ _


/-- A variant of `incl_lift` with `Z` an initial object. -/
@[simps!]
def inclLiftToInitial {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (hZ : Limits.IsInitial Z) :
    incl ⋙ liftToInitial F hZ ≅ F :=
  inclLift _ _ _


/-- A variant of `lift_unique` with `Z` an initial object. -/
@[simps!]
def liftToInitialUnique {D : Type*} [Category D] {Z : D} (F : C ⥤ D) (hZ : Limits.IsInitial Z)
    (G : WithInitial C ⥤ D) (h : incl ⋙ G ≅ F) (hG : G.obj star ≅ Z) : G ≅ liftToInitial F hZ :=
  liftUnique F (fun _z => hZ.to _) (fun _x _y _f => hZ.hom_ext _ _) G h hG fun _x => hZ.hom_ext _ _


/-- Constructs a morphism from `star` to `of X`. -/
@[simp]
def homTo (X : C) : star ⟶ incl.obj X :=
  starInitial.to _

-- Porting note: need to do cases analysis

instance isIso_of_to_star {X : WithInitial C} (f : X ⟶ star) : IsIso f :=
  match X with
  | of _X => f.elim
  | star => ⟨f, rfl, rfl⟩


