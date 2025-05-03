/-- The category of `R`-coalgebras. -/
structure CoalgebraCat extends ModuleCat.{v} R where
  instCoalgebra : Coalgebra R carrier


instance : CoeSort (CoalgebraCat.{v} R) (Type v) :=
  ⟨(·.carrier)⟩


@[simp] theorem moduleCat_of_toModuleCat (X : CoalgebraCat.{v} R) :
    ModuleCat.of R X.toModuleCat = X.toModuleCat :=
  rfl


/-- The object in the category of `R`-coalgebras associated to an `R`-coalgebra. -/
@[simps]
def of (X : Type v) [AddCommGroup X] [Module R X] [Coalgebra R X] :
    CoalgebraCat R :=
  { ModuleCat.of R X with
    instCoalgebra := (inferInstance : Coalgebra R X) }


@[simp]
lemma of_comul {X : Type v} [AddCommGroup X] [Module R X] [Coalgebra R X] :
    Coalgebra.comul (A := of R X) = Coalgebra.comul (R := R) (A := X) := rfl


@[simp]
lemma of_counit {X : Type v} [AddCommGroup X] [Module R X] [Coalgebra R X] :
    Coalgebra.counit (A := of R X) = Coalgebra.counit (R := R) (A := X) := rfl


/-- A type alias for `CoalgHom` to avoid confusion between the categorical and
algebraic spellings of composition. -/
@[ext]
structure Hom (V W : CoalgebraCat.{v} R) where
  /-- The underlying `CoalgHom` -/
  toCoalgHom : V →ₗc[R] W


lemma Hom.toCoalgHom_injective (V W : CoalgebraCat.{v} R) :
    Function.Injective (Hom.toCoalgHom : Hom V W → _) :=
                      /-
                        R : Type u
                        inst✝ : CommRing R
                        V W : CoalgebraCat R
                        x✝² x✝¹ : V.Hom W
                        f g : CoalgHom R ↑V.toModuleCat ↑W.toModuleCat
                        x✝ : Eq { toCoalgHom := f }.toCoalgHom { toCoalgHom := g }.toCoalgHom
                        ⊢ Eq { toCoalgHom := f } { toCoalgHom := g }
                      -/
  fun ⟨f⟩ ⟨g⟩ _ => by congr
                      /-
                        🎉 no goals
                      -/


instance category : Category (CoalgebraCat.{v} R) where
  Hom M N := Hom M N
  id M := ⟨CoalgHom.id R M⟩
  comp f g := ⟨CoalgHom.comp g.toCoalgHom f.toCoalgHom⟩

-- TODO: if `Quiver.Hom` and the instance above were `reducible`, this wouldn't be needed.

@[ext]
lemma hom_ext {M N : CoalgebraCat.{v} R} (f g : M ⟶ N) (h : f.toCoalgHom = g.toCoalgHom) :
    f = g :=
  Hom.ext h


/-- Typecheck a `CoalgHom` as a morphism in `CoalgebraCat R`. -/
abbrev ofHom {X Y : Type v} [AddCommGroup X] [Module R X] [AddCommGroup Y] [Module R Y]
    [Coalgebra R X] [Coalgebra R Y] (f : X →ₗc[R] Y) :
    of R X ⟶ of R Y :=
  ⟨f⟩


@[simp] theorem toCoalgHom_comp {M N U : CoalgebraCat.{v} R} (f : M ⟶ N) (g : N ⟶ U) :
    (f ≫ g).toCoalgHom = g.toCoalgHom.comp f.toCoalgHom :=
  rfl


@[simp] theorem toCoalgHom_id {M : CoalgebraCat.{v} R} :
    Hom.toCoalgHom (𝟙 M) = CoalgHom.id _ _ :=
  rfl


instance concreteCategory : ConcreteCategory.{v} (CoalgebraCat.{v} R) where
  forget :=
    { obj := fun M => M
      map := fun f => f.toCoalgHom }
  forget_faithful :=
    { map_injective := fun {_ _} => DFunLike.coe_injective.comp <| Hom.toCoalgHom_injective _ _ }


instance hasForgetToModule : HasForget₂ (CoalgebraCat R) (ModuleCat R) where
  forget₂ :=
    { obj := fun M => ModuleCat.of R M
      map := fun f => ModuleCat.ofHom f.toCoalgHom.toLinearMap }


@[simp]
theorem forget₂_obj (X : CoalgebraCat R) :
    (forget₂ (CoalgebraCat R) (ModuleCat R)).obj X = ModuleCat.of R X :=
  rfl


@[simp]
theorem forget₂_map (X Y : CoalgebraCat R) (f : X ⟶ Y) :
    (forget₂ (CoalgebraCat R) (ModuleCat R)).map f = ModuleCat.ofHom (f.toCoalgHom : X →ₗ[R] Y) :=
  rfl


/-- Build an isomorphism in the category `CoalgebraCat R` from a
`CoalgEquiv`. -/
@[simps]
def toCoalgebraCatIso (e : X ≃ₗc[R] Y) : CoalgebraCat.of R X ≅ CoalgebraCat.of R Y where
  hom := CoalgebraCat.ofHom e
  inv := CoalgebraCat.ofHom e.symm
  hom_inv_id := Hom.ext <| DFunLike.ext _ _ e.left_inv
  inv_hom_id := Hom.ext <| DFunLike.ext _ _ e.right_inv


@[simp] theorem toCoalgebraCatIso_refl :
    toCoalgebraCatIso (CoalgEquiv.refl R X) = .refl _ :=
  rfl


@[simp] theorem toCoalgebraCatIso_symm (e : X ≃ₗc[R] Y) :
    toCoalgebraCatIso e.symm = (toCoalgebraCatIso e).symm :=
  rfl


@[simp] theorem toCoalgebraCatIso_trans (e : X ≃ₗc[R] Y) (f : Y ≃ₗc[R] Z) :
    toCoalgebraCatIso (e.trans f) = toCoalgebraCatIso e ≪≫ toCoalgebraCatIso f :=
  rfl


/-- Build a `CoalgEquiv` from an isomorphism in the category
`CoalgebraCat R`. -/
def toCoalgEquiv (i : X ≅ Y) : X ≃ₗc[R] Y :=
  { i.hom.toCoalgHom with
    invFun := i.inv.toCoalgHom
    left_inv := fun x => CoalgHom.congr_fun (congr_arg CoalgebraCat.Hom.toCoalgHom i.3) x
    right_inv := fun x => CoalgHom.congr_fun (congr_arg CoalgebraCat.Hom.toCoalgHom i.4) x }


@[simp] theorem toCoalgEquiv_toCoalgHom (i : X ≅ Y) :
    i.toCoalgEquiv = i.hom.toCoalgHom := rfl


@[simp] theorem toCoalgEquiv_refl : toCoalgEquiv (.refl X) = .refl _ _ :=
  rfl


@[simp] theorem toCoalgEquiv_symm (e : X ≅ Y) :
    toCoalgEquiv e.symm = (toCoalgEquiv e).symm :=
  rfl


@[simp] theorem toCoalgEquiv_trans (e : X ≅ Y) (f : Y ≅ Z) :
    toCoalgEquiv (e ≪≫ f) = e.toCoalgEquiv.trans f.toCoalgEquiv :=
  rfl


instance CoalgebraCat.forget_reflects_isos :
    (forget (CoalgebraCat.{v} R)).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : CoalgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (CoalgebraCat R)).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget (CoalgebraCat.{v} R)).map f)
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : CoalgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (CoalgebraCat R)).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget (CoalgebraCat R)).obj X) ((Cate …
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃ₗc[R] Y := { f.toCoalgHom, i.toEquiv with }
    /-
      R : Type u
      inst✝ : CommRing R
      X Y : CoalgebraCat R
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget (CoalgebraCat R)).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget (CoalgebraCat R)).obj X) ((Cate …
      e : CoalgEquiv R ↑X.toModuleCat ↑Y.toModuleCat :=
        let __src := f.toCoalgHom;
        let __src_1 := i.toEquiv;
        { toCoalgHom := __src, invFun := __src_1.invFun, left_inv := ⋯, right_inv := …
      ⊢ CategoryTheory.IsIso f
    -/
    exact ⟨e.toCoalgebraCatIso.isIso_hom.1⟩
    /-
      🎉 no goals
    -/

