/-- The category of monoids and monoid morphisms. -/
@[to_additive AddMonCat]
def MonCat : Type (u + 1) :=
  Bundled Monoid


/-- `MonoidHom` doesn't actually assume associativity. This alias is needed to make the category
theory machinery work. -/
@[to_additive]
abbrev AssocMonoidHom (M N : Type*) [Monoid M] [Monoid N] :=
  MonoidHom M N


@[to_additive]
instance bundledHom : BundledHom AssocMonoidHom where
  toFun {_ _} _ _ f := ⇑f
  id _ := MonoidHom.id _
  comp _ _ _ := MonoidHom.comp


deriving instance LargeCategory for MonCat

attribute [to_additive instAddMonCatLargeCategory] instMonCatLargeCategory

-- Porting note: https://github.com/leanprover-community/mathlib4/issues/5020

@[to_additive]
instance concreteCategory : ConcreteCategory MonCat :=
  BundledHom.concreteCategory _


@[to_additive]
instance : CoeSort MonCat Type* where
  coe X := X.α


@[to_additive]
instance (X : MonCat) : Monoid X := X.str

-- Porting note (https://github.com/leanprover-community/mathlib4/pull/10670): this instance was not necessary in mathlib

@[to_additive]
instance {X Y : MonCat} : CoeFun (X ⟶ Y) fun _ => X → Y where
  coe (f : X →* Y) := f


@[to_additive]
instance instFunLike (X Y : MonCat) : FunLike (X ⟶ Y) X Y :=
  inferInstanceAs <| FunLike (X →* Y) X Y


@[to_additive]
instance instMonoidHomClass (X Y : MonCat) : MonoidHomClass (X ⟶ Y) X Y :=
  inferInstanceAs <| MonoidHomClass (X →* Y) X Y


@[to_additive (attr := simp)]
lemma coe_id {X : MonCat} : (𝟙 X : X → X) = id := rfl


@[to_additive (attr := simp)]
lemma coe_comp {X Y Z : MonCat} {f : X ⟶ Y} {g : Y ⟶ Z} : (f ≫ g : X → Z) = g ∘ f := rfl


@[to_additive (attr := simp)] lemma forget_map {X Y : MonCat} (f : X ⟶ Y) :
    (forget MonCat).map f = f := rfl


@[to_additive (attr := ext)]
lemma ext {X Y : MonCat} {f g : X ⟶ Y} (w : ∀ x : X, f x = g x) : f = g :=
  MonoidHom.ext w


/-- Construct a bundled `MonCat` from the underlying type and typeclass. -/
@[to_additive]
def of (M : Type u) [Monoid M] : MonCat :=
  Bundled.of M


@[to_additive]
theorem coe_of (R : Type u) [Monoid R] : (MonCat.of R : Type u) = R := rfl


@[to_additive]
instance : Inhabited MonCat :=
  -- The default instance for `Monoid PUnit` is derived via `CommRing` which breaks to_additive
  ⟨@of PUnit (@DivInvMonoid.toMonoid _ (@Group.toDivInvMonoid _
    (@CommGroup.toGroup _ PUnit.commGroup)))⟩


/-- Typecheck a `MonoidHom` as a morphism in `MonCat`. -/
@[to_additive]
def ofHom {X Y : Type u} [Monoid X] [Monoid Y] (f : X →* Y) : of X ⟶ of Y := f


@[to_additive (attr := simp)]
lemma ofHom_apply {X Y : Type u} [Monoid X] [Monoid Y] (f : X →* Y) (x : X) :
    (ofHom f) x = f x := rfl

---- Porting note: added to ease the port of `CategoryTheory.Action.Basic`

@[to_additive]
instance (X Y : MonCat.{u}) : One (X ⟶ Y) := ⟨ofHom 1⟩


@[to_additive (attr := simp)]
lemma oneHom_apply (X Y : MonCat.{u}) (x : X) : (1 : X ⟶ Y) x = 1 := rfl

---- Porting note: added to ease the port of `CategoryTheory.Action.Basic`

@[to_additive (attr := simp)]
lemma one_of {A : Type*} [Monoid A] : (1 : MonCat.of A) = (1 : A) := rfl


@[to_additive (attr := simp)]
lemma mul_of {A : Type*} [Monoid A] (a b : A) :
    @HMul.hMul (MonCat.of A) (MonCat.of A) (MonCat.of A) _ a b = a * b := rfl


@[to_additive]
                                                           /-
                                                             G : Type u_1
                                                             inst✝ : Group G
                                                             ⊢ Group ↑(MonCat.of G)
                                                           -/
instance {G : Type*} [Group G] : Group (MonCat.of G) := by assumption
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Universe lift functor for monoids. -/
@[to_additive (attr := simps)
  "Universe lift functor for additive monoids."]
def uliftFunctor : MonCat.{v} ⥤ MonCat.{max v u} where
  obj X := MonCat.of (ULift.{u, v} X)
  map {_ _} f := MonCat.ofHom <|
    MulEquiv.ulift.symm.toMonoidHom.comp <| f.comp MulEquiv.ulift.toMonoidHom
                 /-
                   X : MonCat
                   ⊢ Eq ({ obj := fun X => MonCat.of (ULift.{u, v} ↑X), map := fun {x x_1} f => M …
                 -/
  map_id X := by rfl
                 /-
                   🎉 no goals
                 -/
                             /-
                               X Y Z : MonCat
                               f : Quiver.Hom X Y
                               g : Quiver.Hom Y Z
                               ⊢ Eq ({ obj := fun X => MonCat.of (ULift.{u, v} ↑X), map := fun {x x_1} f => M …
                             -/
  map_comp {X Y Z} f g := by rfl
                             /-
                               🎉 no goals
                             -/


/-- The category of commutative monoids and monoid morphisms. -/
@[to_additive AddCommMonCat]
def CommMonCat : Type (u + 1) :=
  Bundled CommMonoid


@[to_additive]
instance : BundledHom.ParentProjection @CommMonoid.toMonoid := ⟨⟩


deriving instance LargeCategory for CommMonCat

attribute [to_additive instAddCommMonCatLargeCategory] instCommMonCatLargeCategory

-- Porting note: https://github.com/leanprover-community/mathlib4/issues/5020

@[to_additive]
instance concreteCategory : ConcreteCategory CommMonCat := by
  /-
    ⊢ CategoryTheory.ConcreteCategory CommMonCat
  -/
  dsimp only [CommMonCat]
  /-
    ⊢ CategoryTheory.ConcreteCategory (CategoryTheory.Bundled CommMonoid)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[to_additive]
instance : CoeSort CommMonCat Type* where
  coe X := X.α


@[to_additive]
instance (X : CommMonCat) : CommMonoid X := X.str

-- Porting note (https://github.com/leanprover-community/mathlib4/pull/10670): this instance was not necessary in mathlib

@[to_additive]
instance {X Y : CommMonCat} : CoeFun (X ⟶ Y) fun _ => X → Y where
  coe (f : X →* Y) := f


@[to_additive]
instance instFunLike (X Y : CommMonCat) : FunLike (X ⟶ Y) X Y :=
                               /-
                                 X Y : CommMonCat
                                 ⊢ FunLike (MonoidHom ↑X ↑Y) ↑X ↑Y
                               -/
  show FunLike (X →* Y) X Y by infer_instance
                               /-
                                 🎉 no goals
                               -/


@[to_additive (attr := simp)]
lemma coe_id {X : CommMonCat} : (𝟙 X : X → X) = id := rfl


@[to_additive (attr := simp)]
lemma coe_comp {X Y Z : CommMonCat} {f : X ⟶ Y} {g : Y ⟶ Z} : (f ≫ g : X → Z) = g ∘ f := rfl


@[to_additive (attr := simp)]
lemma forget_map {X Y : CommMonCat} (f : X ⟶ Y) :
    (forget CommMonCat).map f = (f : X → Y) :=
  rfl


@[to_additive (attr := ext)]
lemma ext {X Y : CommMonCat} {f g : X ⟶ Y} (w : ∀ x : X, f x = g x) : f = g :=
  MonoidHom.ext w


/-- Construct a bundled `CommMonCat` from the underlying type and typeclass. -/
@[to_additive]
def of (M : Type u) [CommMonoid M] : CommMonCat :=
  Bundled.of M


@[to_additive]
instance : Inhabited CommMonCat :=
  -- The default instance for `CommMonoid PUnit` is derived via `CommRing` which breaks to_additive
  ⟨@of PUnit (@CommGroup.toCommMonoid _ PUnit.commGroup)⟩

-- Porting note: removed `@[simp]` here, as it makes it harder to tell when to apply
-- bundled or unbundled lemmas.
-- (This change seems dangerous!)

@[to_additive]
theorem coe_of (R : Type u) [CommMonoid R] : (CommMonCat.of R : Type u) = R :=
  rfl


@[to_additive hasForgetToAddMonCat]
instance hasForgetToMonCat : HasForget₂ CommMonCat MonCat :=
  BundledHom.forget₂ _ _


@[to_additive]
instance : Coe CommMonCat.{u} MonCat.{u} where coe := (forget₂ CommMonCat MonCat).obj

-- Porting note: this was added to make automation work (it already exists for MonCat)

/-- Typecheck a `MonoidHom` as a morphism in `CommMonCat`. -/
@[to_additive]
def ofHom {X Y : Type u} [CommMonoid X] [CommMonoid Y] (f : X →* Y) : of X ⟶ of Y := f


@[to_additive (attr := simp)]
lemma ofHom_apply {X Y : Type u} [CommMonoid X] [CommMonoid Y] (f : X →* Y) (x : X) :
    (ofHom f) x = f x := rfl


/-- Universe lift functor for commutative monoids. -/
@[to_additive (attr := simps)
  "Universe lift functor for additive commutative monoids."]
def uliftFunctor : CommMonCat.{v} ⥤ CommMonCat.{max v u} where
  obj X := CommMonCat.of (ULift.{u, v} X)
  map {_ _} f := CommMonCat.ofHom <|
    MulEquiv.ulift.symm.toMonoidHom.comp <| f.comp MulEquiv.ulift.toMonoidHom
                 /-
                   X : CommMonCat
                   ⊢ Eq ({ obj := fun X => CommMonCat.of (ULift.{u, v} ↑X), map := fun {x x_1} f  …
                 -/
  map_id X := by rfl
                 /-
                   🎉 no goals
                 -/
                             /-
                               X Y Z : CommMonCat
                               f : Quiver.Hom X Y
                               g : Quiver.Hom Y Z
                               ⊢ Eq ({ obj := fun X => CommMonCat.of (ULift.{u, v} ↑X), map := fun {x x_1} f  …
                             -/
  map_comp {X Y Z} f g := by rfl
                             /-
                               🎉 no goals
                             -/


/-- Build an isomorphism in the category `MonCat` from a `MulEquiv` between `Monoid`s. -/
@[to_additive (attr := simps) AddEquiv.toAddMonCatIso
      "Build an isomorphism in the category `AddMonCat` from\nan `AddEquiv` between `AddMonoid`s."]
def MulEquiv.toMonCatIso (e : X ≃* Y) : MonCat.of X ≅ MonCat.of Y where
  hom := MonCat.ofHom e.toMonoidHom
  inv := MonCat.ofHom e.symm.toMonoidHom


/-- Build an isomorphism in the category `CommMonCat` from a `MulEquiv` between `CommMonoid`s. -/
@[to_additive (attr := simps) AddEquiv.toAddCommMonCatIso]
def MulEquiv.toCommMonCatIso (e : X ≃* Y) : CommMonCat.of X ≅ CommMonCat.of Y where
  hom := CommMonCat.ofHom e.toMonoidHom
  inv := CommMonCat.ofHom e.symm.toMonoidHom


/-- Build a `MulEquiv` from an isomorphism in the category `MonCat`. -/
@[to_additive addMonCatIsoToAddEquiv
      "Build an `AddEquiv` from an isomorphism in the category\n`AddMonCat`."]
def monCatIsoToMulEquiv {X Y : MonCat} (i : X ≅ Y) : X ≃* Y :=
  MonoidHom.toMulEquiv i.hom i.inv i.hom_inv_id i.inv_hom_id


/-- Build a `MulEquiv` from an isomorphism in the category `CommMonCat`. -/
@[to_additive "Build an `AddEquiv` from an isomorphism in the category\n`AddCommMonCat`."]
def commMonCatIsoToMulEquiv {X Y : CommMonCat} (i : X ≅ Y) : X ≃* Y :=
  MonoidHom.toMulEquiv i.hom i.inv i.hom_inv_id i.inv_hom_id


/-- multiplicative equivalences between `Monoid`s are the same as (isomorphic to) isomorphisms
in `MonCat` -/
@[to_additive addEquivIsoAddMonCatIso]
def mulEquivIsoMonCatIso {X Y : Type u} [Monoid X] [Monoid Y] :
    X ≃* Y ≅ MonCat.of X ≅ MonCat.of Y where
  hom e := e.toMonCatIso
  inv i := i.monCatIsoToMulEquiv


/-- multiplicative equivalences between `CommMonoid`s are the same as (isomorphic to) isomorphisms
in `CommMonCat` -/
@[to_additive addEquivIsoAddCommMonCatIso]
def mulEquivIsoCommMonCatIso {X Y : Type u} [CommMonoid X] [CommMonoid Y] :
    X ≃* Y ≅ CommMonCat.of X ≅ CommMonCat.of Y where
  hom e := e.toCommMonCatIso
  inv i := i.commMonCatIsoToMulEquiv


@[to_additive]
instance MonCat.forget_reflects_isos : (forget MonCat.{u}).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      X✝ Y✝ : Type u
      X Y : MonCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget MonCat).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget MonCat).map f)
    -- Again a problem that exists already creeps into other things https://github.com/leanprover/lean4/pull/2644
    -- this used to be `by aesop`; see next declaration
    /-
      X✝ Y✝ : Type u
      X Y : MonCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget MonCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget MonCat).obj X) ((CategoryTheory …
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃* Y := MulEquiv.mk i.toEquiv (MonoidHom.map_mul (show MonoidHom X Y from f))
    /-
      X✝ Y✝ : Type u
      X Y : MonCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget MonCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget MonCat).obj X) ((CategoryTheory …
      e : MulEquiv ↑X ↑Y := { toEquiv := i.toEquiv, map_mul' := ⋯ }
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toMonCatIso.isIso_hom
    /-
      🎉 no goals
    -/


@[to_additive]
instance CommMonCat.forget_reflects_isos : (forget CommMonCat.{u}).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      X✝ Y✝ : Type u
      X Y : CommMonCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommMonCat).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget CommMonCat).map f)
    let e : X ≃* Y := MulEquiv.mk i.toEquiv
      -- Porting FIXME: this would ideally be `by aesop`, as in `MonCat.forget_reflects_isos`
      (MonoidHom.map_mul (show MonoidHom X Y from f))
    /-
      X✝ Y✝ : Type u
      X Y : CommMonCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommMonCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget CommMonCat).obj X) ((CategoryTh …
      e : MulEquiv ↑X ↑Y := { toEquiv := i.toEquiv, map_mul' := ⋯ }
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toCommMonCatIso.isIso_hom
    /-
      🎉 no goals
    -/

-- Porting note: this was added in order to ensure that `forget₂ CommMonCat MonCat`
-- automatically reflects isomorphisms
-- we could have used `CategoryTheory.ConcreteCategory.ReflectsIso` alternatively

@[to_additive]
instance CommMonCat.forget₂_full : (forget₂ CommMonCat MonCat).Full where
  map_surjective f := ⟨f, rfl⟩


@[to_additive (attr := simp)] theorem MonoidHom.comp_id_monCat
    {G : MonCat.{u}} {H : Type u} [Monoid H] (f : G →* H) : f.comp (𝟙 G) = f :=
  Category.id_comp (MonCat.ofHom f)

@[to_additive (attr := simp)] theorem MonoidHom.id_monCat_comp
    {G : Type u} [Monoid G] {H : MonCat.{u}} (f : G →* H) : MonoidHom.comp (𝟙 H) f = f :=
  Category.comp_id (MonCat.ofHom f)


@[to_additive (attr := simp)] theorem MonoidHom.comp_id_commMonCat
    {G : CommMonCat.{u}} {H : Type u} [CommMonoid H] (f : G →* H) : f.comp (𝟙 G) = f :=
  Category.id_comp (CommMonCat.ofHom f)

@[to_additive (attr := simp)] theorem MonoidHom.id_commMonCat_comp
    {G : Type u} [CommMonoid G] {H : CommMonCat.{u}} (f : G →* H) : MonoidHom.comp (𝟙 H) f = f :=
  Category.comp_id (CommMonCat.ofHom f)

