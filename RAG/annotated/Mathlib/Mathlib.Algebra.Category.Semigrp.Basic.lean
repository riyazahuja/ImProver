/-- The category of magmas and magma morphisms. -/
@[to_additive]
def MagmaCat : Type (u + 1) :=
  Bundled Mul


@[to_additive]
instance bundledHom : BundledHom @MulHom :=
  ⟨@MulHom.toFun, @MulHom.id, @MulHom.comp,
       /-
         ⊢ ∀ {α β : Type ?u.90} (Iα : Mul α) (Iβ : Mul β), Function.Injective MulHom.to …
       -/
               /-
                 🎉 no goals
               -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    by intros; apply @DFunLike.coe_injective, by aesop_cat, by aesop_cat⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/

-- Porting note: deriving failed for `ConcreteCategory`,
-- "default handlers have not been implemented yet"
-- https://github.com/leanprover-community/mathlib4/issues/5020

deriving instance LargeCategory for MagmaCat

instance instConcreteCategory : ConcreteCategory MagmaCat := BundledHom.concreteCategory MulHom


attribute [to_additive] instMagmaCatLargeCategory instConcreteCategory


@[to_additive]
instance : CoeSort MagmaCat Type* where
  coe X := X.α

-- Porting note: Hinting to Lean that `forget R` and `R` are the same

unif_hint forget_obj_eq_coe (R : MagmaCat) where ⊢
  (forget MagmaCat).obj R ≟ R

unif_hint _root_.AddMagmaCat.forget_obj_eq_coe (R : AddMagmaCat) where ⊢
  (forget AddMagmaCat).obj R ≟ R


@[to_additive]
instance (X : MagmaCat) : Mul X := X.str


@[to_additive]
instance instFunLike (X Y : MagmaCat) : FunLike (X ⟶ Y) X Y :=
  inferInstanceAs <| FunLike (X →ₙ* Y) X Y


@[to_additive]
instance instMulHomClass (X Y : MagmaCat) : MulHomClass (X ⟶ Y) X Y :=
  inferInstanceAs <| MulHomClass (X →ₙ* Y) X Y


/-- Construct a bundled `MagmaCat` from the underlying type and typeclass. -/
@[to_additive]
def of (M : Type u) [Mul M] : MagmaCat :=
  Bundled.of M


@[to_additive (attr := simp)]
theorem coe_of (R : Type u) [Mul R] : (MagmaCat.of R : Type u) = R :=
  rfl


@[to_additive (attr := simp)]
lemma mulEquiv_coe_eq {X Y : Type _} [Mul X] [Mul Y] (e : X ≃* Y) :
    (@DFunLike.coe (MagmaCat.of X ⟶ MagmaCat.of Y) _ (fun _ => (forget MagmaCat).obj _)
      ConcreteCategory.instFunLike (e : X →ₙ* Y) : X → Y) = ↑e :=
  rfl


/-- Typecheck a `MulHom` as a morphism in `MagmaCat`. -/
@[to_additive]
def ofHom {X Y : Type u} [Mul X] [Mul Y] (f : X →ₙ* Y) : of X ⟶ of Y := f


@[to_additive] -- Porting note: simp removed, simpNF says LHS simplifies to itself
theorem ofHom_apply {X Y : Type u} [Mul X] [Mul Y] (f : X →ₙ* Y) (x : X) : ofHom f x = f x :=
  rfl


@[to_additive]
instance : Inhabited MagmaCat :=
  ⟨MagmaCat.of PEmpty⟩


/-- The category of semigroups and semigroup morphisms. -/
@[to_additive]
def Semigrp : Type (u + 1) :=
  Bundled Semigroup


@[to_additive]
instance : BundledHom.ParentProjection @Semigroup.toMul := ⟨⟩


deriving instance LargeCategory for Semigrp

-- Porting note: deriving failed for `ConcreteCategory`,
-- "default handlers have not been implemented yet"
-- https://github.com/leanprover-community/mathlib4/issues/5020

instance instConcreteCategory : ConcreteCategory Semigrp :=
  BundledHom.concreteCategory (fun _ _ => _)


attribute [to_additive] instSemigrpLargeCategory Semigrp.instConcreteCategory


@[to_additive]
instance : CoeSort Semigrp Type* where
  coe X := X.α

-- Porting note: Hinting to Lean that `forget R` and `R` are the same

unif_hint forget_obj_eq_coe (R : Semigrp) where ⊢
  (forget Semigrp).obj R ≟ R

unif_hint _root_.AddSemigrp.forget_obj_eq_coe (R : AddSemigrp) where ⊢
  (forget AddSemigrp).obj R ≟ R


@[to_additive]
instance (X : Semigrp) : Semigroup X := X.str


@[to_additive]
instance instFunLike (X Y : Semigrp) : FunLike (X ⟶ Y) X Y :=
  inferInstanceAs <| FunLike (X →ₙ* Y) X Y


@[to_additive]
instance instMulHomClass (X Y : Semigrp) : MulHomClass (X ⟶ Y) X Y :=
  inferInstanceAs <| MulHomClass (X →ₙ* Y) X Y


/-- Construct a bundled `Semigrp` from the underlying type and typeclass. -/
@[to_additive]
def of (M : Type u) [Semigroup M] : Semigrp :=
  Bundled.of M


@[to_additive (attr := simp)]
theorem coe_of (R : Type u) [Semigroup R] : (Semigrp.of R : Type u) = R :=
  rfl


@[to_additive (attr := simp)]
lemma mulEquiv_coe_eq {X Y : Type _} [Semigroup X] [Semigroup Y] (e : X ≃* Y) :
    (@DFunLike.coe (Semigrp.of X ⟶ Semigrp.of Y) _ (fun _ => (forget Semigrp).obj _)
      ConcreteCategory.instFunLike (e : X →ₙ* Y) : X → Y) = ↑e :=
  rfl


/-- Typecheck a `MulHom` as a morphism in `Semigrp`. -/
@[to_additive]
def ofHom {X Y : Type u} [Semigroup X] [Semigroup Y] (f : X →ₙ* Y) : of X ⟶ of Y :=
  f


@[to_additive] -- Porting note: simp removed, simpNF says LHS simplifies to itself
theorem ofHom_apply {X Y : Type u} [Semigroup X] [Semigroup Y] (f : X →ₙ* Y) (x : X) :
    ofHom f x = f x :=
  rfl


@[to_additive]
instance : Inhabited Semigrp :=
  ⟨Semigrp.of PEmpty⟩


@[to_additive]
instance hasForgetToMagmaCat : HasForget₂ Semigrp MagmaCat :=
  BundledHom.forget₂ _ _


/-- Build an isomorphism in the category `MagmaCat` from a `MulEquiv` between `Mul`s. -/
@[to_additive (attr := simps)
      "Build an isomorphism in the category `AddMagmaCat` from an `AddEquiv` between `Add`s."]
def MulEquiv.toMagmaCatIso (e : X ≃* Y) : MagmaCat.of X ≅ MagmaCat.of Y where
  hom := e.toMulHom
  inv := e.symm.toMulHom
  hom_inv_id := by
    /-
      X Y : Type u
      inst✝¹ : Mul X
      inst✝ : Mul Y
      e : MulEquiv X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp e.toMulHom e.symm.toMulHom) (Category …
    -/
    ext
    /-
      case w
      X Y : Type u
      inst✝¹ : Mul X
      inst✝ : Mul Y
      e : MulEquiv X Y
      x✝ : (CategoryTheory.forget MagmaCat).obj (MagmaCat.of X)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.toMulHom e.symm.toMulHom) x✝) ((Ca …
    -/
    simp_rw [comp_apply, toMulHom_eq_coe, MagmaCat.mulEquiv_coe_eq, symm_apply_apply, id_apply]
    /-
      🎉 no goals
    -/


/-- Build an isomorphism in the category `Semigroup` from a `MulEquiv` between `Semigroup`s. -/
@[to_additive (attr := simps)
  "Build an isomorphism in the category
  `AddSemigroup` from an `AddEquiv` between `AddSemigroup`s."]
def MulEquiv.toSemigrpIso (e : X ≃* Y) : Semigrp.of X ≅ Semigrp.of Y where
  hom := e.toMulHom
  inv := e.symm.toMulHom


/-- Build a `MulEquiv` from an isomorphism in the category `MagmaCat`. -/
@[to_additive
      "Build an `AddEquiv` from an isomorphism in the category `AddMagmaCat`."]
def magmaCatIsoToMulEquiv {X Y : MagmaCat} (i : X ≅ Y) : X ≃* Y :=
  MulHom.toMulEquiv i.hom i.inv i.hom_inv_id i.inv_hom_id


/-- Build a `MulEquiv` from an isomorphism in the category `Semigroup`. -/
@[to_additive
  "Build an `AddEquiv` from an isomorphism in the category `AddSemigroup`."]
def semigrpIsoToMulEquiv {X Y : Semigrp} (i : X ≅ Y) : X ≃* Y :=
  MulHom.toMulEquiv i.hom i.inv i.hom_inv_id i.inv_hom_id


/-- multiplicative equivalences between `Mul`s are the same as (isomorphic to) isomorphisms
in `MagmaCat` -/
@[to_additive
    "additive equivalences between `Add`s are the same
    as (isomorphic to) isomorphisms in `AddMagmaCat`"]
def mulEquivIsoMagmaIso {X Y : Type u} [Mul X] [Mul Y] :
    X ≃* Y ≅ MagmaCat.of X ≅ MagmaCat.of Y where
  hom e := e.toMagmaCatIso
  inv i := i.magmaCatIsoToMulEquiv


/-- multiplicative equivalences between `Semigroup`s are the same as (isomorphic to) isomorphisms
in `Semigroup` -/
@[to_additive
  "additive equivalences between `AddSemigroup`s are
  the same as (isomorphic to) isomorphisms in `AddSemigroup`"]
def mulEquivIsoSemigrpIso {X Y : Type u} [Semigroup X] [Semigroup Y] :
    X ≃* Y ≅ Semigrp.of X ≅ Semigrp.of Y where
  hom e := e.toSemigrpIso
  inv i := i.semigrpIsoToMulEquiv


@[to_additive]
instance MagmaCat.forgetReflectsIsos : (forget MagmaCat.{u}).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      X✝ Y✝ : Type u
      X Y : MagmaCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget MagmaCat).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget MagmaCat).map f)
    /-
      X✝ Y✝ : Type u
      X Y : MagmaCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget MagmaCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget MagmaCat).obj X) ((CategoryTheo …
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃* Y := { f, i.toEquiv with }
    /-
      X✝ Y✝ : Type u
      X Y : MagmaCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget MagmaCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget MagmaCat).obj X) ((CategoryTheo …
      e : MulEquiv ↑X ↑Y :=
        let __src := i.toEquiv;
        { toFun := f.toFun, invFun := __src.invFun, left_inv := ⋯, right_inv := ⋯, m …
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toMagmaCatIso.isIso_hom
    /-
      🎉 no goals
    -/


@[to_additive]
instance Semigrp.forgetReflectsIsos : (forget Semigrp.{u}).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      X✝ Y✝ : Type u
      X Y : Semigrp
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget Semigrp).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget Semigrp).map f)
    /-
      X✝ Y✝ : Type u
      X Y : Semigrp
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget Semigrp).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget Semigrp).obj X) ((CategoryTheor …
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃* Y := { f, i.toEquiv with }
    /-
      X✝ Y✝ : Type u
      X Y : Semigrp
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget Semigrp).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget Semigrp).obj X) ((CategoryTheor …
      e : MulEquiv ↑X ↑Y :=
        let __src := i.toEquiv;
        { toFun := f.toFun, invFun := __src.invFun, left_inv := ⋯, right_inv := ⋯, m …
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toSemigrpIso.isIso_hom
    /-
      🎉 no goals
    -/

-- Porting note: this was added in order to ensure that `forget₂ CommMonCat MonCat`
-- automatically reflects isomorphisms
-- we could have used `CategoryTheory.ConcreteCategory.ReflectsIso` alternatively

@[to_additive]
instance Semigrp.forget₂_full : (forget₂ Semigrp MagmaCat).Full where
  map_surjective f := ⟨f, rfl⟩


