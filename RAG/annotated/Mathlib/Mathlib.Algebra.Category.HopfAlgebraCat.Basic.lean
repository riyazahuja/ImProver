/-- The category of `R`-Hopf algebras. -/
structure HopfAlgebraCat where
  /-- The underlying type. -/
  carrier : Type v
  [instRing : Ring carrier]
  [instHopfAlgebra : HopfAlgebra R carrier]


instance : CoeSort (HopfAlgebraCat.{v} R) (Type v) :=
  ⟨(·.carrier)⟩


/-- The object in the category of `R`-Hopf algebras associated to an `R`-Hopf algebra. -/
@[simps]
def of (X : Type v) [Ring X] [HopfAlgebra R X] :
    HopfAlgebraCat R where
  carrier := X


@[simp]
lemma of_comul {X : Type v} [Ring X] [HopfAlgebra R X] :
    Coalgebra.comul (A := of R X) = Coalgebra.comul (R := R) (A := X) := rfl


@[simp]
lemma of_counit {X : Type v} [Ring X] [HopfAlgebra R X] :
    Coalgebra.counit (A := of R X) = Coalgebra.counit (R := R) (A := X) := rfl


/-- A type alias for `BialgHom` to avoid confusion between the categorical and
algebraic spellings of composition. -/
@[ext]
structure Hom (V W : HopfAlgebraCat.{v} R) where
  /-- The underlying `BialgHom`. -/
  toBialgHom : V →ₐc[R] W


lemma Hom.toBialgHom_injective (V W : HopfAlgebraCat.{v} R) :
    Function.Injective (Hom.toBialgHom : Hom V W → _) :=
                      /-
                        R : Type u
                        inst✝ : CommRing R
                        V W : HopfAlgebraCat R
                        x✝² x✝¹ : V.Hom W
                        f g : BialgHom R V.carrier W.carrier
                        x✝ : Eq { toBialgHom := f }.toBialgHom { toBialgHom := g }.toBialgHom
                        ⊢ Eq { toBialgHom := f } { toBialgHom := g }
                      -/
  fun ⟨f⟩ ⟨g⟩ _ => by congr
                      /-
                        🎉 no goals
                      -/


instance category : Category (HopfAlgebraCat.{v} R) where
  Hom X Y := Hom X Y
  id X := ⟨BialgHom.id R X⟩
  comp f g := ⟨BialgHom.comp g.toBialgHom f.toBialgHom⟩

-- TODO: if `Quiver.Hom` and the instance above were `reducible`, this wouldn't be needed.

@[ext]
lemma hom_ext {X Y : HopfAlgebraCat.{v} R} (f g : X ⟶ Y) (h : f.toBialgHom = g.toBialgHom) :
    f = g :=
  Hom.ext h


/-- Typecheck a `BialgHom` as a morphism in `HopfAlgebraCat R`. -/
abbrev ofHom {X Y : Type v} [Ring X] [Ring Y]
    [HopfAlgebra R X] [HopfAlgebra R Y] (f : X →ₐc[R] Y) :
    of R X ⟶ of R Y :=
  ⟨f⟩


@[simp] theorem toBialgHom_comp {X Y Z : HopfAlgebraCat.{v} R} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).toBialgHom = g.toBialgHom.comp f.toBialgHom :=
  rfl


@[simp] theorem toBialgHom_id {M : HopfAlgebraCat.{v} R} :
    Hom.toBialgHom (𝟙 M) = BialgHom.id _ _ :=
  rfl


instance concreteCategory : ConcreteCategory.{v} (HopfAlgebraCat.{v} R) where
  forget :=
    { obj := fun M => M
      map := fun f => f.toBialgHom }
  forget_faithful :=
    { map_injective := fun {_ _} => DFunLike.coe_injective.comp <| Hom.toBialgHom_injective _ _ }


instance hasForgetToBialgebra : HasForget₂ (HopfAlgebraCat R) (BialgebraCat R) where
  forget₂ :=
    { obj := fun X => BialgebraCat.of R X
      map := fun {_ _} f => BialgebraCat.ofHom f.toBialgHom }


@[simp]
theorem forget₂_bialgebra_obj (X : HopfAlgebraCat R) :
    (forget₂ (HopfAlgebraCat R) (BialgebraCat R)).obj X = BialgebraCat.of R X :=
  rfl


@[simp]
theorem forget₂_bialgebra_map (X Y : HopfAlgebraCat R) (f : X ⟶ Y) :
    (forget₂ (HopfAlgebraCat R) (BialgebraCat R)).map f = BialgebraCat.ofHom f.toBialgHom :=
  rfl


/-- Build an isomorphism in the category `HopfAlgebraCat R` from a
`BialgEquiv`. -/
@[simps]
def toHopfAlgebraCatIso (e : X ≃ₐc[R] Y) : HopfAlgebraCat.of R X ≅ HopfAlgebraCat.of R Y where
  hom := HopfAlgebraCat.ofHom e
  inv := HopfAlgebraCat.ofHom e.symm
  hom_inv_id := Hom.ext <| DFunLike.ext _ _ e.left_inv
  inv_hom_id := Hom.ext <| DFunLike.ext _ _ e.right_inv


@[simp] theorem toHopfAlgebraCatIso_refl :
    toHopfAlgebraCatIso (BialgEquiv.refl R X) = .refl _ :=
  rfl


@[simp] theorem toHopfAlgebraCatIso_symm (e : X ≃ₐc[R] Y) :
    toHopfAlgebraCatIso e.symm = (toHopfAlgebraCatIso e).symm :=
  rfl


@[simp] theorem toHopfAlgebraCatIso_trans (e : X ≃ₐc[R] Y) (f : Y ≃ₐc[R] Z) :
    toHopfAlgebraCatIso (e.trans f) = toHopfAlgebraCatIso e ≪≫ toHopfAlgebraCatIso f :=
  rfl


/-- Build a `BialgEquiv` from an isomorphism in the category
`HopfAlgebraCat R`. -/
def toHopfAlgEquiv (i : X ≅ Y) : X ≃ₐc[R] Y :=
  { i.hom.toBialgHom with
    invFun := i.inv.toBialgHom
    left_inv := fun x => BialgHom.congr_fun (congr_arg HopfAlgebraCat.Hom.toBialgHom i.3) x
    right_inv := fun x => BialgHom.congr_fun (congr_arg HopfAlgebraCat.Hom.toBialgHom i.4) x }


@[simp] theorem toHopfAlgEquiv_toBialgHom (i : X ≅ Y) :
    (i.toHopfAlgEquiv : X →ₐc[R] Y) = i.hom.1 := rfl


@[simp] theorem toHopfAlgEquiv_refl : toHopfAlgEquiv (.refl X) = .refl _ _ :=
  rfl


@[simp] theorem toHopfAlgEquiv_symm (e : X ≅ Y) :
    toHopfAlgEquiv e.symm = (toHopfAlgEquiv e).symm :=
  rfl


@[simp] theorem toHopfAlgEquiv_trans (e : X ≅ Y) (f : Y ≅ Z) :
    toHopfAlgEquiv (e ≪≫ f) = e.toHopfAlgEquiv.trans f.toHopfAlgEquiv :=
  rfl


instance HopfAlgebraCat.forget_reflects_isos :
    (forget (HopfAlgebraCat.{v} R)).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : HopfAlgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (HopfAlgebraCat R)).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget (HopfAlgebraCat.{v} R)).map f)
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : HopfAlgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (HopfAlgebraCat R)).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget (HopfAlgebraCat R)).obj X) ((Ca …
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃ₐc[R] Y := { f.toBialgHom, i.toEquiv with }
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : HopfAlgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (HopfAlgebraCat R)).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget (HopfAlgebraCat R)).obj X) ((Ca …
      e : BialgEquiv R X.carrier Y.carrier :=
        let __src := f.toBialgHom;
        let __src_1 := i.toEquiv;
        { toCoalgHom := __src.toCoalgHom, invFun := __src_1.invFun, left_inv := ⋯, r …
      ⊢ CategoryTheory.IsIso f
    -/
    exact ⟨e.toHopfAlgebraCatIso.isIso_hom.1⟩
    /-
      🎉 no goals
    -/

