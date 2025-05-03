/-- The category of R-algebras and their morphisms. -/
structure AlgebraCat where
  private mk ::
  /-- The underlying type. -/
  carrier : Type v
  [isRing : Ring carrier]
  [isAlgebra : Algebra R carrier]

-- Porting note: typemax hack to fix universe complaints

/-- An alias for `AlgebraCat.{max u₁ u₂}`, to deal around unification issues.
Since the universe the ring lives in can be inferred, we put that last. -/
@[nolint checkUnivs]
abbrev AlgebraCatMax.{v₁, v₂, u₁} (R : Type u₁) [CommRing R] := AlgebraCat.{max v₁ v₂} R


instance : CoeSort (AlgebraCat R) (Type v) :=
  ⟨AlgebraCat.carrier⟩


/-- The object in the category of R-algebras associated to a type equipped with the appropriate
typeclasses. This is the preferred way to construct a term of `AlgebraCat R`. -/
abbrev of (X : Type v) [Ring X] [Algebra R X] : AlgebraCat.{v} R :=
  ⟨X⟩


lemma coe_of (X : Type v) [Ring X] [Algebra R X] : (of R X : Type v) = X :=
  rfl


variable {R} in
/-- The type of morphisms in `AlgebraCat R`. -/
@[ext]
structure Hom (A B : AlgebraCat.{v} R) where
  private mk ::
  /-- The underlying algebra map. -/
  hom : A →ₐ[R] B


instance : Category (AlgebraCat.{v} R) where
  Hom A B := Hom A B
  id A := ⟨AlgHom.id R A⟩
  comp f g := ⟨g.hom.comp f.hom⟩


instance {M N : AlgebraCat.{v} R} : CoeFun (M ⟶ N) (fun _ ↦ M → N) where
  coe f := f.hom


@[simp]
lemma hom_id {A : AlgebraCat.{v} R} : (𝟙 A : A ⟶ A).hom = AlgHom.id R A := rfl

/- Provided for rewriting. -/

lemma id_apply (A : AlgebraCat.{v} R) (a : A) :
                              /-
                                R : Type u
                                inst✝ : CommRing R
                                A : AlgebraCat R
                                a : ↑A
                                ⊢ Eq ((CategoryTheory.CategoryStruct.id A).hom a) a
                              -/
    (𝟙 A : A ⟶ A) a = a := by simp
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma hom_comp {A B C : AlgebraCat.{v} R} (f : A ⟶ B) (g : B ⟶ C) :
    (f ≫ g).hom = g.hom.comp f.hom := rfl

/- Provided for rewriting. -/

lemma comp_apply {A B C : AlgebraCat.{v} R} (f : A ⟶ B) (g : B ⟶ C) (a : A) :
                              /-
                                R : Type u
                                inst✝ : CommRing R
                                A B C : AlgebraCat R
                                f : Quiver.Hom A B
                                g : Quiver.Hom B C
                                a : ↑A
                                ⊢ Eq ((CategoryTheory.CategoryStruct.comp f g).hom a) (g.hom (f.hom a))
                              -/
    (f ≫ g) a = g (f a) := by simp
                              /-
                                🎉 no goals
                              -/


@[ext]
lemma hom_ext {A B : AlgebraCat.{v} R} {f g : A ⟶ B} (hf : f.hom = g.hom) : f = g :=
  Hom.ext hf


/-- Typecheck an `AlgHom` as a morphism in `AlgebraCat R`. -/
abbrev ofHom {R : Type u} [CommRing R] {X Y : Type v} [Ring X] [Algebra R X] [Ring Y] [Algebra R Y]
    (f : X →ₐ[R] Y) : of R X ⟶ of R Y :=
  ⟨f⟩


lemma hom_ofHom {R : Type u} [CommRing R] {X Y : Type v} [Ring X] [Algebra R X] [Ring Y]
    [Algebra R Y] (f : X →ₐ[R] Y) : (ofHom f).hom = f := rfl


@[simp]
lemma ofHom_hom {A B : AlgebraCat.{v} R} (f : A ⟶ B) :
    ofHom (Hom.hom f) = f := rfl


@[simp]
lemma ofHom_id {X : Type v} [Ring X] [Algebra R X] : ofHom (AlgHom.id R X) = 𝟙 (of R X) := rfl


@[simp]
lemma ofHom_comp {X Y Z : Type v} [Ring X] [Ring Y] [Ring Z] [Algebra R X] [Algebra R Y]
    [Algebra R Z] (f : X →ₐ[R] Y) (g : Y →ₐ[R] Z) :
    ofHom (g.comp f) = ofHom f ≫ ofHom g :=
  rfl


lemma ofHom_apply {R : Type u} [CommRing R] {X Y : Type v} [Ring X] [Algebra R X] [Ring Y]
    [Algebra R Y] (f : X →ₐ[R] Y) (x : X) : ofHom f x = f x := rfl


@[simp]
lemma inv_hom_apply {A B : AlgebraCat.{v} R} (e : A ≅ B) (x : A) : e.inv (e.hom x) = x := by
  /-
    R : Type u
    inst✝ : CommRing R
    A B : AlgebraCat R
    e : CategoryTheory.Iso A B
    x : ↑A
    ⊢ Eq (e.inv.hom (e.hom.hom x)) x
  -/
  rw [← comp_apply]
  /-
    R : Type u
    inst✝ : CommRing R
    A B : AlgebraCat R
    e : CategoryTheory.Iso A B
    x : ↑A
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.hom e.inv).hom x) x
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma hom_inv_apply {A B : AlgebraCat.{v} R} (e : A ≅ B) (x : B) : e.hom (e.inv x) = x := by
  /-
    R : Type u
    inst✝ : CommRing R
    A B : AlgebraCat R
    e : CategoryTheory.Iso A B
    x : ↑B
    ⊢ Eq (e.hom.hom (e.inv.hom x)) x
  -/
  rw [← comp_apply]
  /-
    R : Type u
    inst✝ : CommRing R
    A B : AlgebraCat R
    e : CategoryTheory.Iso A B
    x : ↑B
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.inv e.hom).hom x) x
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Inhabited (AlgebraCat R) :=
  ⟨of R R⟩


instance : ConcreteCategory.{v} (AlgebraCat.{v} R) where
  forget :=
    { obj := fun R => R
      map := fun f => f.hom }
                                  /-
                                    R : Type u
                                    inst✝ : CommRing R
                                    X✝ Y✝ : AlgebraCat R
                                    a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                                    h : Eq ({ obj := fun R_1 => ↑R_1, map := fun {X Y} f => ⇑↑f.hom, map_id := ⋯,  …
                                    ⊢ Eq a₁✝ a₂✝
                                  -/
  forget_faithful := ⟨fun h => by ext x; simpa using congrFun h x⟩
                                         /-
                                           🎉 no goals
                                         -/


lemma forget_obj {A : AlgebraCat.{v} R} : (forget (AlgebraCat.{v} R)).obj A = A := rfl


lemma forget_map {A B : AlgebraCat.{v} R} (f : A ⟶ B) :
    (forget (AlgebraCat.{v} R)).map f = f :=
  rfl


instance {S : AlgebraCat.{v} R} : Ring ((forget (AlgebraCat R)).obj S) :=
  (inferInstance : Ring S.carrier)


instance {S : AlgebraCat.{v} R} : Algebra R ((forget (AlgebraCat R)).obj S) :=
  (inferInstance : Algebra R S.carrier)


instance hasForgetToRing : HasForget₂ (AlgebraCat.{v} R) RingCat.{v} where
  forget₂ :=
    { obj := fun A => RingCat.of A
      map := fun f => RingCat.ofHom f.hom.toRingHom }


instance hasForgetToModule : HasForget₂ (AlgebraCat.{v} R) (ModuleCat.{v} R) where
  forget₂ :=
    { obj := fun M => ModuleCat.of R M
      map := fun f => ModuleCat.ofHom f.hom.toLinearMap }


@[simp]
lemma forget₂_module_obj (X : AlgebraCat.{v} R) :
    (forget₂ (AlgebraCat.{v} R) (ModuleCat.{v} R)).obj X = ModuleCat.of R X :=
  rfl


@[simp]
lemma forget₂_module_map {X Y : AlgebraCat.{v} R} (f : X ⟶ Y) :
    (forget₂ (AlgebraCat.{v} R) (ModuleCat.{v} R)).map f = ModuleCat.ofHom f.hom.toLinearMap :=
  rfl


variable {R} in
/-- Forgetting to the underlying type and then building the bundled object returns the original
algebra. -/
@[simps]
def ofSelfIso (M : AlgebraCat.{v} R) : AlgebraCat.of R M ≅ M where
  hom := 𝟙 M
  inv := 𝟙 M


/-- The "free algebra" functor, sending a type `S` to the free algebra on `S`. -/
@[simps! obj map]
def free : Type u ⥤ AlgebraCat.{u} R where
  obj S := of R (FreeAlgebra R S)
  map f := ofHom <| FreeAlgebra.lift _ <| FreeAlgebra.ι _ ∘ f


/-- The free/forget adjunction for `R`-algebras. -/
def adj : free.{u} R ⊣ forget (AlgebraCat.{u} R) :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ =>
        { toFun := fun f ↦ (FreeAlgebra.lift _).symm f.hom
          invFun := fun f ↦ ofHom <| (FreeAlgebra.lift _) f
                                 /-
                                   R : Type u
                                   inst✝ : CommRing R
                                   x✝¹ : Type u
                                   x✝ : AlgebraCat R
                                   f : Quiver.Hom ((AlgebraCat.free R).obj x✝¹) x✝
                                   ⊢ Eq ((fun f => AlgebraCat.ofHom ((FreeAlgebra.lift R) f)) ((fun f => (FreeAlg …
                                 -/
          left_inv := fun f ↦ by aesop
                                 /-
                                   🎉 no goals
                                 -/
                                  /-
                                    R : Type u
                                    inst✝ : CommRing R
                                    x✝¹ : Type u
                                    x✝ : AlgebraCat R
                                    f : Quiver.Hom x✝¹ ((CategoryTheory.forget (AlgebraCat R)).obj x✝)
                                    ⊢ Eq ((fun f => (FreeAlgebra.lift R).symm f.hom) ((fun f => AlgebraCat.ofHom ( …
                                  -/
          right_inv := fun f ↦ by simp [forget_obj, forget_map] } }
                                  /-
                                    🎉 no goals
                                  -/


instance : (forget (AlgebraCat.{u} R)).IsRightAdjoint := (adj R).isRightAdjoint


/-- Build an isomorphism in the category `AlgebraCat R` from a `AlgEquiv` between `Algebra`s. -/
@[simps]
def AlgEquiv.toAlgebraIso {g₁ : Ring X₁} {g₂ : Ring X₂} {m₁ : Algebra R X₁} {m₂ : Algebra R X₂}
    (e : X₁ ≃ₐ[R] X₂) : AlgebraCat.of R X₁ ≅ AlgebraCat.of R X₂ where
  hom := AlgebraCat.ofHom (e : X₁ →ₐ[R] X₂)
  inv := AlgebraCat.ofHom (e.symm : X₂ →ₐ[R] X₁)


/-- Build a `AlgEquiv` from an isomorphism in the category `AlgebraCat R`. -/
@[simps]
def toAlgEquiv {X Y : AlgebraCat R} (i : X ≅ Y) : X ≃ₐ[R] Y :=
  { i.hom.hom with
    toFun := i.hom
    invFun := i.inv
                           /-
                             R : Type u
                             inst✝ : CommRing R
                             X₁ X₂ : Type u
                             X Y : AlgebraCat R
                             i : CategoryTheory.Iso X Y
                             x : ↑X
                             ⊢ Eq (i.inv.hom (i.hom.hom x)) x
                           -/
    left_inv := fun x ↦ by simp
                           /-
                             🎉 no goals
                           -/
                            /-
                              R : Type u
                              inst✝ : CommRing R
                              X₁ X₂ : Type u
                              X Y : AlgebraCat R
                              i : CategoryTheory.Iso X Y
                              x : ↑Y
                              ⊢ Eq (i.hom.hom (i.inv.hom x)) x
                            -/
    right_inv := fun x ↦ by simp }
                            /-
                              🎉 no goals
                            -/


/-- Algebra equivalences between `Algebra`s are the same as (isomorphic to) isomorphisms in
`AlgebraCat`. -/
@[simps]
def algEquivIsoAlgebraIso {X Y : Type u} [Ring X] [Ring Y] [Algebra R X] [Algebra R Y] :
    (X ≃ₐ[R] Y) ≅ AlgebraCat.of R X ≅ AlgebraCat.of R Y where
  hom e := e.toAlgebraIso
  inv i := i.toAlgEquiv


instance AlgebraCat.forget_reflects_isos : (forget (AlgebraCat.{u} R)).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      R : Type u
      inst✝ : CommRing R
      X₁ X₂ : Type u
      X Y : AlgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (AlgebraCat R)).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget (AlgebraCat.{u} R)).map f)
    /-
      R : Type u
      inst✝ : CommRing R
      X₁ X₂ : Type u
      X Y : AlgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (AlgebraCat R)).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget (AlgebraCat R)).obj X) ((Catego …
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃ₐ[R] Y := { f.hom, i.toEquiv with }
    /-
      R : Type u
      inst✝ : CommRing R
      X₁ X₂ : Type u
      X Y : AlgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (AlgebraCat R)).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget (AlgebraCat R)).obj X) ((Catego …
      e : AlgEquiv R ↑X ↑Y :=
        let __src := f.hom;
        let __src_1 := i.toEquiv;
        { toFun := (↑↑__src.toRingHom).toFun, invFun := __src_1.invFun, left_inv :=  …
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toAlgebraIso.isIso_hom
    /-
      🎉 no goals
    -/

