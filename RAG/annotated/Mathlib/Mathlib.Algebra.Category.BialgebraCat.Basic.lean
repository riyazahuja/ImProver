/-- The category of `R`-bialgebras. -/
structure BialgebraCat where
  /-- The underlying type. -/
  carrier : Type v
  [instRing : Ring carrier]
  [instBialgebra : Bialgebra R carrier]


instance : CoeSort (BialgebraCat.{v} R) (Type v) :=
  ⟨(·.carrier)⟩


/-- The object in the category of `R`-bialgebras associated to an `R`-bialgebra. -/
@[simps]
def of (X : Type v) [Ring X] [Bialgebra R X] :
    BialgebraCat R where
  carrier := X


@[simp]
lemma of_comul {X : Type v} [Ring X] [Bialgebra R X] :
    Coalgebra.comul (A := of R X) = Coalgebra.comul (R := R) (A := X) := rfl


@[simp]
lemma of_counit {X : Type v} [Ring X] [Bialgebra R X] :
    Coalgebra.counit (A := of R X) = Coalgebra.counit (R := R) (A := X) := rfl


/-- A type alias for `BialgHom` to avoid confusion between the categorical and
algebraic spellings of composition. -/
@[ext]
structure Hom (V W : BialgebraCat.{v} R) where
  /-- The underlying `BialgHom` -/
  toBialgHom : V →ₐc[R] W


lemma Hom.toBialgHom_injective (V W : BialgebraCat.{v} R) :
    Function.Injective (Hom.toBialgHom : Hom V W → _) :=
                      /-
                        R : Type u
                        inst✝ : CommRing R
                        V W : BialgebraCat R
                        x✝² x✝¹ : V.Hom W
                        f g : BialgHom R V.carrier W.carrier
                        x✝ : Eq { toBialgHom := f }.toBialgHom { toBialgHom := g }.toBialgHom
                        ⊢ Eq { toBialgHom := f } { toBialgHom := g }
                      -/
  fun ⟨f⟩ ⟨g⟩ _ => by congr
                      /-
                        🎉 no goals
                      -/


instance category : Category (BialgebraCat.{v} R) where
  Hom X Y := Hom X Y
  id X := ⟨BialgHom.id R X⟩
  comp f g := ⟨BialgHom.comp g.toBialgHom f.toBialgHom⟩

-- TODO: if `Quiver.Hom` and the instance above were `reducible`, this wouldn't be needed.

@[ext]
lemma hom_ext {X Y : BialgebraCat.{v} R} (f g : X ⟶ Y) (h : f.toBialgHom = g.toBialgHom) :
    f = g :=
  Hom.ext h


/-- Typecheck a `BialgHom` as a morphism in `BialgebraCat R`. -/
abbrev ofHom {X Y : Type v} [Ring X] [Ring Y]
    [Bialgebra R X] [Bialgebra R Y] (f : X →ₐc[R] Y) :
    of R X ⟶ of R Y :=
  ⟨f⟩


@[simp] theorem toBialgHom_comp {X Y Z : BialgebraCat.{v} R} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).toBialgHom = g.toBialgHom.comp f.toBialgHom :=
  rfl


@[simp] theorem toBialgHom_id {M : BialgebraCat.{v} R} :
    Hom.toBialgHom (𝟙 M) = BialgHom.id _ _ :=
  rfl


instance concreteCategory : ConcreteCategory.{v} (BialgebraCat.{v} R) where
  forget :=
    { obj := fun M => M
      map := fun f => f.toBialgHom }
  forget_faithful :=
    { map_injective := fun {_ _} => DFunLike.coe_injective.comp <| Hom.toBialgHom_injective _ _ }


instance hasForgetToAlgebra : HasForget₂ (BialgebraCat R) (AlgebraCat R) where
  forget₂ :=
    { obj := fun X => AlgebraCat.of R X
      map := fun {X Y} f => AlgebraCat.ofHom f.toBialgHom }


@[simp]
theorem forget₂_algebra_obj (X : BialgebraCat R) :
    (forget₂ (BialgebraCat R) (AlgebraCat R)).obj X = AlgebraCat.of R X :=
  rfl


@[simp]
theorem forget₂_algebra_map (X Y : BialgebraCat R) (f : X ⟶ Y) :
    (forget₂ (BialgebraCat R) (AlgebraCat R)).map f = AlgebraCat.ofHom f.toBialgHom :=
  rfl


instance hasForgetToCoalgebra : HasForget₂ (BialgebraCat R) (CoalgebraCat R) where
  forget₂ :=
    { obj := fun X => CoalgebraCat.of R X
      map := fun {_ _} f => CoalgebraCat.ofHom f.toBialgHom }


@[simp]
theorem forget₂_coalgebra_obj (X : BialgebraCat R) :
    (forget₂ (BialgebraCat R) (CoalgebraCat R)).obj X = CoalgebraCat.of R X :=
  rfl


@[simp]
theorem forget₂_coalgebra_map (X Y : BialgebraCat R) (f : X ⟶ Y) :
    (forget₂ (BialgebraCat R) (CoalgebraCat R)).map f = CoalgebraCat.ofHom f.toBialgHom :=
  rfl


/-- Build an isomorphism in the category `BialgebraCat R` from a
`BialgEquiv`. -/
@[simps]
def toBialgebraCatIso (e : X ≃ₐc[R] Y) : BialgebraCat.of R X ≅ BialgebraCat.of R Y where
  hom := BialgebraCat.ofHom e
  inv := BialgebraCat.ofHom e.symm
  hom_inv_id := Hom.ext <| DFunLike.ext _ _ e.left_inv
  inv_hom_id := Hom.ext <| DFunLike.ext _ _ e.right_inv


@[simp] theorem toBialgebraCatIso_refl : toBialgebraCatIso (BialgEquiv.refl R X) = .refl _ :=
  rfl


@[simp] theorem toBialgebraCatIso_symm (e : X ≃ₐc[R] Y) :
    toBialgebraCatIso e.symm = (toBialgebraCatIso e).symm :=
  rfl


@[simp] theorem toBialgebraCatIso_trans (e : X ≃ₐc[R] Y) (f : Y ≃ₐc[R] Z) :
    toBialgebraCatIso (e.trans f) = toBialgebraCatIso e ≪≫ toBialgebraCatIso f :=
  rfl


/-- Build a `BialgEquiv` from an isomorphism in the category
`BialgebraCat R`. -/
def toBialgEquiv (i : X ≅ Y) : X ≃ₐc[R] Y :=
  { i.hom.toBialgHom with
    invFun := i.inv.toBialgHom
    left_inv := fun x => BialgHom.congr_fun (congr_arg BialgebraCat.Hom.toBialgHom i.3) x
    right_inv := fun x => BialgHom.congr_fun (congr_arg BialgebraCat.Hom.toBialgHom i.4) x }


@[simp] theorem toBialgEquiv_toBialgHom (i : X ≅ Y) :
    (i.toBialgEquiv : X →ₐc[R] Y) = i.hom.1 := rfl


@[simp] theorem toBialgEquiv_refl : toBialgEquiv (.refl X) = .refl _ _ :=
  rfl


@[simp] theorem toBialgEquiv_symm (e : X ≅ Y) :
    toBialgEquiv e.symm = (toBialgEquiv e).symm :=
  rfl


@[simp] theorem toBialgEquiv_trans (e : X ≅ Y) (f : Y ≅ Z) :
    toBialgEquiv (e ≪≫ f) = e.toBialgEquiv.trans f.toBialgEquiv :=
  rfl


instance BialgebraCat.forget_reflects_isos :
    (forget (BialgebraCat.{v} R)).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : BialgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (BialgebraCat R)).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget (BialgebraCat.{v} R)).map f)
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : BialgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (BialgebraCat R)).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget (BialgebraCat R)).obj X) ((Cate …
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃ₐc[R] Y := { f.toBialgHom, i.toEquiv with }
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : BialgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (BialgebraCat R)).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget (BialgebraCat R)).obj X) ((Cate …
      e : BialgEquiv R X.carrier Y.carrier :=
        let __src := f.toBialgHom;
        let __src_1 := i.toEquiv;
        { toCoalgHom := __src.toCoalgHom, invFun := __src_1.invFun, left_inv := ⋯, r …
      ⊢ CategoryTheory.IsIso f
    -/
    exact ⟨e.toBialgebraCatIso.isIso_hom.1⟩
    /-
      🎉 no goals
    -/

