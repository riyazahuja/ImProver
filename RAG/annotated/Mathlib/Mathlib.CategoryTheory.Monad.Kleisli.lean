/-- The objects for the Kleisli category of the monad `T : Monad C`, which are the same
thing as objects of the base category `C`.
-/
@[nolint unusedArguments]
def Kleisli (_T : Monad C) :=
  C


instance [Inhabited C] (T : Monad C) : Inhabited (Kleisli T) :=
  ⟨(default : C)⟩


/-- The Kleisli category on a monad `T`.
    cf Definition 5.2.9 in [Riehl][riehl2017]. -/
instance category : Category (Kleisli T) where
  Hom := fun X Y : C => X ⟶ (T : C ⥤ C).obj Y
  id X := T.η.app X
  comp {_} {_} {Z} f g := f ≫ (T : C ⥤ C).map g ≫ T.μ.app Z
  id_comp {X} {Y} f := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y : CategoryTheory.Kleisli T
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
    -/
    dsimp -- Porting note: unfold comp
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y : CategoryTheory.Kleisli T
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.η.app X) (CategoryTheory.CategoryS …
    -/
    rw [← T.η.naturality_assoc f, T.left_unit]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y : CategoryTheory.Kleisli T
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map f) …
    -/
    apply Category.comp_id
    /-
      🎉 no goals
    -/
  assoc f g h := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Kleisli T
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp only [Functor.map_comp, Category.assoc, Monad.assoc]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Kleisli T
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
    -/
    erw [T.μ.naturality_assoc]
    /-
      🎉 no goals
    -/


/-- The left adjoint of the adjunction which induces the monad `(T, η_ T, μ_ T)`. -/
@[simps]
def toKleisli : C ⥤ Kleisli T where
  obj X := (X : Kleisli T)
  map {X} {Y} f := (f ≫ T.η.app Y : X ⟶ T.obj Y)
  map_comp {X} {Y} {Z} f g := by
    -- Porting note: hack for missing unfold_projs tactic
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => CategoryTheory.CategoryStruct …
    -/
    change _ = (f ≫ (Monad.η T).app Y) ≫ T.map (g ≫ (Monad.η T).app Z) ≫ T.μ.app Z
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => CategoryTheory.CategoryStruct …
    -/
    simp [← T.η.naturality g]
    /-
      🎉 no goals
    -/


/-- The right adjoint of the adjunction which induces the monad `(T, η_ T, μ_ T)`. -/
@[simps]
def fromKleisli : Kleisli T ⥤ C where
  obj X := T.obj X
  map {_} {Y} f := T.map f ≫ T.μ.app Y
  map_id _ := T.right_unit _
  map_comp {X} {Y} {Z} f g := by
    -- Porting note: hack for missing unfold_projs tactic
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y Z : CategoryTheory.Kleisli T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => T.obj X, map := fun {x Y} f => CategoryTheory.Category …
    -/
    change T.map (f ≫ T.map g ≫ T.μ.app Z) ≫ T.μ.app Z = _
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y Z : CategoryTheory.Kleisli T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (CategoryTheory.CategoryStruct …
    -/
    simp only [Functor.map_comp, Category.assoc]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y Z : CategoryTheory.Kleisli T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map f) (CategoryTheory.CategoryStr …
    -/
    rw [← T.μ.naturality_assoc g, T.assoc]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.Monad C
      X Y Z : CategoryTheory.Kleisli T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map f) (CategoryTheory.CategoryStr …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The Kleisli adjunction which gives rise to the monad `(T, η_ T, μ_ T)`.
    cf Lemma 5.2.11 of [Riehl][riehl2017]. -/
def adj : toKleisli T ⊣ fromKleisli T :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y => Equiv.refl (X ⟶ T.obj Y)
      homEquiv_naturality_left_symm := fun {X} {Y} {Z} f g => by
        -- Porting note: used to be unfold_projs; dsimp
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          T : CategoryTheory.Monad C
          X Y : C
          Z : CategoryTheory.Kleisli T
          f : Quiver.Hom X Y
          g : Quiver.Hom Y ((CategoryTheory.Kleisli.Adjunction.fromKleisli T).obj Z)
          ⊢ Eq (((fun X Y => Equiv.refl (Quiver.Hom X (T.obj Y))) X Z).symm (CategoryThe …
        -/
        change f ≫ g = (f ≫ T.η.app Y) ≫ T.map g ≫ T.μ.app Z
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          T : CategoryTheory.Monad C
          X Y : C
          Z : CategoryTheory.Kleisli T
          f : Quiver.Hom X Y
          g : Quiver.Hom Y ((CategoryTheory.Kleisli.Adjunction.fromKleisli T).obj Z)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
        -/
        rw [Category.assoc, ← T.η.naturality_assoc g, Functor.id_map]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          T : CategoryTheory.Monad C
          X Y : C
          Z : CategoryTheory.Kleisli T
          f : Quiver.Hom X Y
          g : Quiver.Hom Y ((CategoryTheory.Kleisli.Adjunction.fromKleisli T).obj Z)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
        -/
        dsimp
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          T : CategoryTheory.Monad C
          X Y : C
          Z : CategoryTheory.Kleisli T
          f : Quiver.Hom X Y
          g : Quiver.Hom Y ((CategoryTheory.Kleisli.Adjunction.fromKleisli T).obj Z)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
        -/
        simp [Monad.left_unit] }
        /-
          🎉 no goals
        -/


/-- The composition of the adjunction gives the original functor. -/
def toKleisliCompFromKleisliIsoSelf : toKleisli T ⋙ fromKleisli T ≅ T :=
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.Monad C
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The objects for the co-Kleisli category of the comonad `U : Comonad C`, which are the same
thing as objects of the base category `C`.
-/
@[nolint unusedArguments]
def Cokleisli (_U : Comonad C) :=
  C


instance [Inhabited C] (U : Comonad C) : Inhabited (Cokleisli U) :=
  ⟨(default : C)⟩


/-- The co-Kleisli category on a comonad `U`. -/
instance category : Category (Cokleisli U) where
  Hom := fun X Y : C => (U : C ⥤ C).obj X ⟶ Y
  id X := U.ε.app X
  comp {X} {_} {_} f g := U.δ.app X ≫ (U : C ⥤ C).map f ≫ g
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    U : CategoryTheory.Comonad C
                    X✝ Y✝ : CategoryTheory.Cokleisli U
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
                  -/
  id_comp f := by dsimp; rw [U.right_counit_assoc]
                         /-
                           🎉 no goals
                         -/
  assoc {X} {Y} {Z} {W} f g h := by
    -- Porting note: working around lack of unfold_projs
    change U.δ.app X ≫ U.map (U.δ.app X ≫ U.map f ≫ g) ≫ h =
      U.δ.app X ≫ U.map f ≫ (U.δ.app Y ≫ U.map g ≫ h)
    -- Porting note: something was broken here and was easier just to redo from scratch
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      U : CategoryTheory.Comonad C
      X Y Z W : CategoryTheory.Cokleisli U
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z W
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (U.δ.app X) (CategoryTheory.CategoryS …
    -/
    simp only [Functor.map_comp, ← Category.assoc, eq_whisker]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      U : CategoryTheory.Comonad C
      X Y Z W : CategoryTheory.Cokleisli U
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z W
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.assoc, U.δ.naturality, Functor.comp_map, U.coassoc_assoc]
    /-
      🎉 no goals
    -/


/-- The right adjoint of the adjunction which induces the comonad `(U, ε_ U, δ_ U)`. -/
@[simps]
def toCokleisli : C ⥤ Cokleisli U where
  obj X := (X : Cokleisli U)
  map {X} {_} f := (U.ε.app X ≫ f : _)
  map_comp {X} {Y} {_} f g := by
    -- Porting note: working around lack of unfold_projs
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      U : CategoryTheory.Comonad C
      X Y x✝ : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y x✝
      ⊢ Eq ({ obj := fun X => X, map := fun {X x} f => CategoryTheory.CategoryStruct …
    -/
    change U.ε.app X ≫ f ≫ g = U.δ.app X ≫ U.map (U.ε.app X ≫ f) ≫ U.ε.app Y ≫ g
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      U : CategoryTheory.Comonad C
      X Y x✝ : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (U.ε.app X) (CategoryTheory.CategoryS …
    -/
    simp [← U.ε.naturality g]
    /-
      🎉 no goals
    -/


/-- The left adjoint of the adjunction which induces the comonad `(U, ε_ U, δ_ U)`. -/
@[simps]
def fromCokleisli : Cokleisli U ⥤ C where
  obj X := U.obj X
  map {X} {_} f := U.δ.app X ≫ U.map f
  map_id _ := U.right_counit _
  map_comp {X} {Y} {_} f g := by
    -- Porting note: working around lack of unfold_projs
    change U.δ.app X ≫ U.map (U.δ.app X ≫ U.map f ≫ g) =
      (U.δ.app X ≫ U.map f) ≫ (U.δ.app Y ≫ U.map g)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      U : CategoryTheory.Comonad C
      X Y x✝ : CategoryTheory.Cokleisli U
      f : Quiver.Hom X Y
      g : Quiver.Hom Y x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (U.δ.app X) (U.map (CategoryTheory.Ca …
    -/
    simp only [Functor.map_comp, ← Category.assoc]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      U : CategoryTheory.Comonad C
      X Y x✝ : CategoryTheory.Cokleisli U
      f : Quiver.Hom X Y
      g : Quiver.Hom Y x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [Comonad.coassoc]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      U : CategoryTheory.Comonad C
      X Y x✝ : CategoryTheory.Cokleisli U
      f : Quiver.Hom X Y
      g : Quiver.Hom Y x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.assoc, NatTrans.naturality, Functor.comp_map]
    /-
      🎉 no goals
    -/


/-- The co-Kleisli adjunction which gives rise to the monad `(U, ε_ U, δ_ U)`. -/
def adj : fromCokleisli U ⊣ toCokleisli U :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y => Equiv.refl (U.obj X ⟶ Y)
      homEquiv_naturality_right := fun {X} {Y} {_} f g => by
        -- Porting note: working around lack of unfold_projs
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          U : CategoryTheory.Comonad C
          X : CategoryTheory.Cokleisli U
          Y x✝ : C
          f : Quiver.Hom ((CategoryTheory.Cokleisli.Adjunction.fromCokleisli U).obj X) Y
          g : Quiver.Hom Y x✝
          ⊢ Eq (((fun X Y => Equiv.refl (Quiver.Hom (U.obj X) Y)) X x✝) (CategoryTheory. …
        -/
        change f ≫ g = U.δ.app X ≫ U.map f ≫ U.ε.app Y ≫ g
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          U : CategoryTheory.Comonad C
          X : CategoryTheory.Cokleisli U
          Y x✝ : C
          f : Quiver.Hom ((CategoryTheory.Cokleisli.Adjunction.fromCokleisli U).obj X) Y
          g : Quiver.Hom Y x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
        -/
        rw [← Category.assoc (U.map f), U.ε.naturality]; dsimp
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          U : CategoryTheory.Comonad C
          X : CategoryTheory.Cokleisli U
          Y x✝ : C
          f : Quiver.Hom ((CategoryTheory.Cokleisli.Adjunction.fromCokleisli U).obj X) Y
          g : Quiver.Hom Y x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
        -/
        simp only [← Category.assoc, Comonad.left_counit, Category.id_comp] }
        /-
          🎉 no goals
        -/


/-- The composition of the adjunction gives the original functor. -/
def toCokleisliCompFromCokleisliIsoSelf : toCokleisli U ⋙ fromCokleisli U ≅ U :=
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    U : CategoryTheory.Comonad C
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


