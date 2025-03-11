/-- A `LocallyRingedSpace` is a topological space equipped with a sheaf of commutative rings
such that all the stalks are local rings.

A morphism of locally ringed spaces is a morphism of ringed spaces
such that the morphisms induced on stalks are local ring homomorphisms. -/
structure LocallyRingedSpace extends SheafedSpace CommRingCat.{u} where
  /-- Stalks of a locally ringed space are local rings. -/
  isLocalRing : ∀ x, IsLocalRing (presheaf.stalk x)


/-- An alias for `toSheafedSpace`, where the result type is a `RingedSpace`.
This allows us to use dot-notation for the `RingedSpace` namespace.
 -/
def toRingedSpace : RingedSpace :=
  X.toSheafedSpace


/-- The underlying topological space of a locally ringed space. -/
def toTopCat : TopCat :=
  X.1.carrier


instance : CoeSort LocallyRingedSpace (Type u) :=
  ⟨fun X : LocallyRingedSpace => (X.toTopCat : Type _)⟩


instance (x : X) : IsLocalRing (X.presheaf.stalk x) :=
  X.isLocalRing x

-- PROJECT: how about a typeclass "HasStructureSheaf" to mediate the 𝒪 notation, rather
-- than defining it over and over for `PresheafedSpace`, `LocallyRingedSpace`, `Scheme`, etc.

/-- The structure sheaf of a locally ringed space. -/
def 𝒪 : Sheaf CommRingCat X.toTopCat :=
  X.sheaf


/-- A morphism of locally ringed spaces is a morphism of ringed spaces
 such that the morphisms induced on stalks are local ring homomorphisms. -/
@[ext]
structure Hom (X Y : LocallyRingedSpace.{u})
    extends X.toPresheafedSpace.Hom Y.toPresheafedSpace : Type _ where
  /-- the underlying morphism induces a local ring homomorphism on stalks -/
  prop : ∀ x, IsLocalHom (toHom.stalkMap x).hom


/-- A morphism of locally ringed spaces as a morphism of sheafed spaces. -/
abbrev Hom.toShHom {X Y : LocallyRingedSpace.{u}} (f : X.Hom Y) :
  X.toSheafedSpace ⟶ Y.toSheafedSpace := f.1


@[simp, nolint simpVarHead]
lemma Hom.toShHom_mk {X Y : LocallyRingedSpace.{u}}
    (f : X.toPresheafedSpace.Hom Y.toPresheafedSpace) (hf) :
  Hom.toShHom ⟨f, hf⟩ = f := rfl


instance : Quiver LocallyRingedSpace :=
  ⟨Hom⟩


@[ext] lemma Hom.ext' {X Y : LocallyRingedSpace.{u}} {f g : X ⟶ Y} (h : f.toShHom = g.toShHom) :
                /-
                  X Y : AlgebraicGeometry.LocallyRingedSpace
                  f g : Quiver.Hom X Y
                  h : Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom f) (AlgebraicGeometry …
                  ⊢ Eq f g
                -/
    f = g := by cases f; cases g; congr
                                  /-
                                    🎉 no goals
                                  -/


/-- See Note [custom simps projection] -/
def Hom.Simps.toShHom {X Y : LocallyRingedSpace.{u}} (f : X.Hom Y) :
    X.toSheafedSpace ⟶ Y.toSheafedSpace :=
  f.toShHom


/-- A morphism of locally ringed spaces `f : X ⟶ Y` induces
a local ring homomorphism from `Y.stalk (f x)` to `X.stalk x` for any `x : X`.
-/
noncomputable def Hom.stalkMap {X Y : LocallyRingedSpace.{u}} (f : Hom X Y) (x : X) :
    Y.presheaf.stalk (f.1.1 x) ⟶ X.presheaf.stalk x :=
  f.toShHom.stalkMap x


@[instance]
theorem isLocalHomStalkMap {X Y : LocallyRingedSpace.{u}} (f : X ⟶ Y) (x : X) :
    IsLocalHom (f.stalkMap x).hom :=
  f.2 x


@[instance]
theorem isLocalHomValStalkMap {X Y : LocallyRingedSpace.{u}} (f : X ⟶ Y) (x : X) :
    IsLocalHom (f.toShHom.stalkMap x).hom :=
  f.2 x


@[deprecated (since := "2024-10-10")]
alias isLocalRingHomValStalkMap := isLocalHomValStalkMap


/-- The identity morphism on a locally ringed space. -/
@[simps! toShHom]
def id (X : LocallyRingedSpace.{u}) : Hom X X :=
                                   /-
                                     X✝ X : AlgebraicGeometry.LocallyRingedSpace
                                     x : ↑↑X.toPresheafedSpace
                                     ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.C …
                                   -/
  ⟨𝟙 X.toSheafedSpace, fun x => by dsimp; erw [PresheafedSpace.stalkMap.id]; infer_instance⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance (X : LocallyRingedSpace.{u}) : Inhabited (Hom X X) :=
  ⟨id X⟩


/-- Composition of morphisms of locally ringed spaces. -/
def comp {X Y Z : LocallyRingedSpace.{u}} (f : Hom X Y) (g : Hom Y Z) : Hom X Z :=
  ⟨f.toShHom ≫ g.toShHom, fun x => by
    /-
      X✝ X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : X.Hom Y
      g : Y.Hom Z
      x : ↑↑X.toPresheafedSpace
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.C …
    -/
    dsimp
    /-
      X✝ X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : X.Hom Y
      g : Y.Hom Z
      x : ↑↑X.toPresheafedSpace
      ⊢ IsLocalHom (AlgebraicGeometry.PresheafedSpace.Hom.stalkMap (CategoryTheory.C …
    -/
    erw [PresheafedSpace.stalkMap.comp]
    /-
      X✝ X Y Z : AlgebraicGeometry.LocallyRingedSpace
      f : X.Hom Y
      g : Y.Hom Z
      x : ↑↑X.toPresheafedSpace
      ⊢ IsLocalHom (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Presheafed …
    -/
    infer_instance⟩
    /-
      🎉 no goals
    -/


/-- The category of locally ringed spaces. -/
instance : Category LocallyRingedSpace.{u} where
  Hom := Hom
  id := id
  comp {_ _ _} f g := comp f g
                                    /-
                                      X✝ X Y : AlgebraicGeometry.LocallyRingedSpace
                                      f : Quiver.Hom X Y
                                      ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (CategoryTheory.Categor …
                                    -/
                                    /-
                                      X✝ X Y : AlgebraicGeometry.LocallyRingedSpace
                                      f : Quiver.Hom X Y
                                      ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (CategoryTheory.Categor …
                                    -/
  comp_id {X Y} f := Hom.ext' <| by simp [comp]
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  id_comp {X Y} f := Hom.ext' <| by simp [comp]
                                          /-
                                            X x✝³ x✝² x✝¹ x✝ : AlgebraicGeometry.LocallyRingedSpace
                                            f : Quiver.Hom x✝³ x✝²
                                            g : Quiver.Hom x✝² x✝¹
                                            h : Quiver.Hom x✝¹ x✝
                                            ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (CategoryTheory.Categor …
                                          -/
  assoc {_ _ _ _} f g h := Hom.ext' <| by simp [comp]
                                          /-
                                            🎉 no goals
                                          -/


/-- The forgetful functor from `LocallyRingedSpace` to `SheafedSpace CommRing`. -/
@[simps]
def forgetToSheafedSpace : LocallyRingedSpace.{u} ⥤ SheafedSpace CommRingCat.{u} where
  obj X := X.toSheafedSpace
  map {_ _} f := f.1


/-- The canonical map `X ⟶ Spec Γ(X, ⊤)`. This is the unit of the `Γ-Spec` adjunction. -/
instance : forgetToSheafedSpace.Faithful where
  map_injective {_ _} _ _ h := Hom.ext' h


/-- The forgetful functor from `LocallyRingedSpace` to `Top`. -/
@[simps!]
def forgetToTop : LocallyRingedSpace.{u} ⥤ TopCat.{u} :=
  forgetToSheafedSpace ⋙ SheafedSpace.forget _


@[simp]
theorem comp_toShHom {X Y Z : LocallyRingedSpace.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).toShHom = f.toShHom ≫ g.toShHom :=
  rfl


/-- A variant of `id_toShHom'` that works with `𝟙 X` instead of `id X`. -/
@[simp] theorem id_toShHom' (X : LocallyRingedSpace.{u}) :
    Hom.toShHom (𝟙 X) = 𝟙 X.toSheafedSpace :=
  rfl


theorem comp_base {X Y Z : LocallyRingedSpace.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).base = f.base ≫ g.base :=
  rfl


theorem comp_c {X Y Z : LocallyRingedSpace.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).c = g.c ≫ (Presheaf.pushforward _ g.base).map f.c :=
  rfl


theorem comp_c_app {X Y Z : LocallyRingedSpace.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) (U : (Opens Z)ᵒᵖ) :
    (f ≫ g).c.app U = g.c.app U ≫ f.c.app (op <| (Opens.map g.base).obj U.unop) :=
  rfl


/-- Given two locally ringed spaces `X` and `Y`, an isomorphism between `X` and `Y` as _sheafed_
spaces can be lifted to a morphism `X ⟶ Y` as locally ringed spaces.

See also `isoOfSheafedSpaceIso`.
-/
@[simps! toShHom]
def homOfSheafedSpaceHomOfIsIso {X Y : LocallyRingedSpace.{u}}
    (f : X.toSheafedSpace ⟶ Y.toSheafedSpace) [IsIso f] : X ⟶ Y :=
  Hom.mk f fun _ =>
    -- Here we need to see that the stalk maps are really local ring homomorphisms.
    -- This can be solved by type class inference, because stalk maps of isomorphisms
    -- are isomorphisms and isomorphisms are local ring homomorphisms.
    inferInstance


/-- Given two locally ringed spaces `X` and `Y`, an isomorphism between `X` and `Y` as _sheafed_
spaces can be lifted to an isomorphism `X ⟶ Y` as locally ringed spaces.

This is related to the property that the functor `forgetToSheafedSpace` reflects isomorphisms.
In fact, it is slightly stronger as we do not require `f` to come from a morphism between
_locally_ ringed spaces.
-/
def isoOfSheafedSpaceIso {X Y : LocallyRingedSpace.{u}} (f : X.toSheafedSpace ≅ Y.toSheafedSpace) :
    X ≅ Y where
  hom := homOfSheafedSpaceHomOfIsIso f.hom
  inv := homOfSheafedSpaceHomOfIsIso f.inv
  hom_inv_id := Hom.ext' f.hom_inv_id
  inv_hom_id := Hom.ext' f.inv_hom_id


instance : forgetToSheafedSpace.ReflectsIsomorphisms where reflects {_ _} f i :=
  { out :=
      ⟨homOfSheafedSpaceHomOfIsIso (CategoryTheory.inv (forgetToSheafedSpace.map f)),
        Hom.ext' (IsIso.hom_inv_id (I := i)), Hom.ext' (IsIso.inv_hom_id (I := i))⟩ }


instance is_sheafedSpace_iso {X Y : LocallyRingedSpace.{u}} (f : X ⟶ Y) [IsIso f] :
    IsIso f.toShHom :=
  LocallyRingedSpace.forgetToSheafedSpace.map_isIso f


/-- The restriction of a locally ringed space along an open embedding.
-/
@[simps!]
def restrict {U : TopCat} (X : LocallyRingedSpace.{u}) {f : U ⟶ X.toTopCat}
    (h : IsOpenEmbedding f) : LocallyRingedSpace where
  isLocalRing := by
    /-
      X✝ : AlgebraicGeometry.LocallyRingedSpace
      U : TopCat
      X : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom U X.toTopCat
      h : Topology.IsOpenEmbedding ⇑f
      ⊢ ∀ (x : ↑↑(X.restrict h).toPresheafedSpace), IsLocalRing ↑((X.restrict h).pre …
    -/
    intro x
    -- We show that the stalk of the restriction is isomorphic to the original stalk,
    /-
      X✝ : AlgebraicGeometry.LocallyRingedSpace
      U : TopCat
      X : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom U X.toTopCat
      h : Topology.IsOpenEmbedding ⇑f
      x : ↑↑(X.restrict h).toPresheafedSpace
      ⊢ IsLocalRing ↑((X.restrict h).presheaf.stalk x)
    -/
    apply @RingEquiv.isLocalRing _ _ _ (X.isLocalRing (f x))
    /-
      case e
      X✝ : AlgebraicGeometry.LocallyRingedSpace
      U : TopCat
      X : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom U X.toTopCat
      h : Topology.IsOpenEmbedding ⇑f
      x : ↑↑(X.restrict h).toPresheafedSpace
      ⊢ RingEquiv ↑(X.presheaf.stalk (f x)) ↑((X.restrict h).presheaf.stalk x)
    -/
    exact (X.restrictStalkIso h x).symm.commRingCatIsoToRingEquiv
    /-
      🎉 no goals
    -/
  toSheafedSpace := X.toSheafedSpace.restrict h


/-- The canonical map from the restriction to the subspace. -/
def ofRestrict {U : TopCat} (X : LocallyRingedSpace.{u})
    {f : U ⟶ X.toTopCat} (h : IsOpenEmbedding f) : X.restrict h ⟶ X :=
  ⟨X.toPresheafedSpace.ofRestrict h, fun _ => inferInstance⟩


/-- The restriction of a locally ringed space `X` to the top subspace is isomorphic to `X` itself.
-/
def restrictTopIso (X : LocallyRingedSpace.{u}) :
    X.restrict (Opens.isOpenEmbedding ⊤) ≅ X :=
  isoOfSheafedSpaceIso X.toSheafedSpace.restrictTopIso


/-- The global sections, notated Gamma.
-/
def Γ : LocallyRingedSpace.{u}ᵒᵖ ⥤ CommRingCat.{u} :=
  forgetToSheafedSpace.op ⋙ SheafedSpace.Γ


theorem Γ_def : Γ = forgetToSheafedSpace.op ⋙ SheafedSpace.Γ :=
  rfl


@[simp]
theorem Γ_obj (X : LocallyRingedSpace.{u}ᵒᵖ) : Γ.obj X = X.unop.presheaf.obj (op ⊤) :=
  rfl


theorem Γ_obj_op (X : LocallyRingedSpace.{u}) : Γ.obj (op X) = X.presheaf.obj (op ⊤) :=
  rfl


@[simp]
theorem Γ_map {X Y : LocallyRingedSpace.{u}ᵒᵖ} (f : X ⟶ Y) : Γ.map f = f.unop.c.app (op ⊤) :=
  rfl


theorem Γ_map_op {X Y : LocallyRingedSpace.{u}} (f : X ⟶ Y) : Γ.map f.op = f.c.app (op ⊤) :=
  rfl


/-- The empty locally ringed space. -/
def empty : LocallyRingedSpace.{u} where
  carrier := TopCat.of PEmpty
  presheaf := (CategoryTheory.Functor.const _).obj (CommRingCat.of PUnit)
  IsSheaf := Presheaf.isSheaf_of_isTerminal _ CommRingCat.punitIsTerminal
  isLocalRing x := PEmpty.elim x


instance : EmptyCollection LocallyRingedSpace.{u} := ⟨LocallyRingedSpace.empty⟩


/-- The canonical map from the empty locally ringed space. -/
def emptyTo (X : LocallyRingedSpace.{u}) : ∅ ⟶ X :=
                                /-
                                  X✝ X : AlgebraicGeometry.LocallyRingedSpace
                                  ⊢ Continuous fun x => PEmpty.elim x
                                -/
  ⟨⟨⟨fun x => PEmpty.elim x, by fun_prop⟩,
                                /-
                                  🎉 no goals
                                -/
                                              /-
                                                X✝ X : AlgebraicGeometry.LocallyRingedSpace
                                                U : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
                                                ⊢ RingHom (↑(X.presheaf.toPrefunctor.1 U)) PUnit.{u + 1}
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
    { app := fun U => CommRingCat.ofHom <| by refine ⟨⟨⟨0, ?_⟩, ?_⟩, ?_, ?_⟩ <;> intros <;> rfl }⟩,
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
    fun x => PEmpty.elim x⟩


noncomputable
instance {X : LocallyRingedSpace.{u}} : Unique (∅ ⟶ X) where
  default := LocallyRingedSpace.emptyTo X
               /-
                 X✝ X : AlgebraicGeometry.LocallyRingedSpace
                 f : Quiver.Hom EmptyCollection.emptyCollection X
                 ⊢ Eq f Inhabited.default
               -/
  uniq f := by ext ⟨⟩ x; aesop_cat
                         /-
                           🎉 no goals
                         -/


/-- The empty space is initial in `LocallyRingedSpace`. -/
noncomputable
def emptyIsInitial : Limits.IsInitial (∅ : LocallyRingedSpace.{u}) := Limits.IsInitial.ofUnique _

-- This actually holds for all ringed spaces with nontrivial stalks.

theorem basicOpen_zero (X : LocallyRingedSpace.{u}) (U : Opens X.carrier) :
    X.toRingedSpace.basicOpen (0 : X.presheaf.obj <| op U) = ⊥ := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    ⊢ Eq (X.toRingedSpace.basicOpen 0) Bot.bot
  -/
  ext x
  simp only [RingedSpace.basicOpen, Opens.coe_mk, Set.mem_image, Set.mem_setOf_eq, Subtype.exists,
    exists_and_right, exists_eq_right, Opens.coe_bot, Set.mem_empty_iff_false,
    iff_false, not_exists]
  /-
    case h.h
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : ↑↑X.toRingedSpace.toPresheafedSpace
    ⊢ ∀ (x_1 : Membership.mem U x), Not (IsUnit ((X.toRingedSpace.presheaf.germ U  …
  -/
  intros hx
  /-
    case h.h
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : ↑↑X.toRingedSpace.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Not (IsUnit ((X.toRingedSpace.presheaf.germ U x hx).hom 0))
  -/
  rw [map_zero, isUnit_zero_iff]
  /-
    case h.h
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : ↑↑X.toRingedSpace.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Not (Eq 0 1)
  -/
  change (0 : X.presheaf.stalk x) ≠ (1 : X.presheaf.stalk x)
  /-
    case h.h
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : ↑↑X.toRingedSpace.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Ne 0 1
  -/
  exact zero_ne_one
  /-
    🎉 no goals
  -/


@[simp]
lemma basicOpen_eq_bot_of_isNilpotent (X : LocallyRingedSpace.{u}) (U : Opens X.carrier)
    (f : (X.presheaf.obj <| op U)) (hf : IsNilpotent f) :
    X.toRingedSpace.basicOpen f = ⊥ := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    hf : IsNilpotent f
    ⊢ Eq (X.toRingedSpace.basicOpen f) Bot.bot
  -/
  obtain ⟨n, hn⟩ := hf
  cases n.eq_zero_or_pos with
  | inr h =>
    rw [←  X.toRingedSpace.basicOpen_pow f n h, hn]
    simp [basicOpen_zero]
  | inl h =>
    rw [h, pow_zero] at hn
    simp [eq_zero_of_zero_eq_one hn.symm f, basicOpen_zero]


instance component_nontrivial (X : LocallyRingedSpace.{u}) (U : Opens X.carrier) [hU : Nonempty U] :
    Nontrivial (X.presheaf.obj <| op U) :=
  (X.presheaf.germ _ _ hU.some.2).hom.domain_nontrivial


@[simp]
lemma iso_hom_base_inv_base {X Y : LocallyRingedSpace.{u}} (e : X ≅ Y) :
    e.hom.base ≫ e.inv.base = 𝟙 _ := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom.base e.inv.base) (CategoryTheor …
  -/
  rw [← SheafedSpace.comp_base, ← LocallyRingedSpace.comp_toShHom]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (CategoryTheory.Categor …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma iso_hom_base_inv_base_apply {X Y : LocallyRingedSpace.{u}} (e : X ≅ Y) (x : X) :
    (e.inv.base (e.hom.base x)) = x := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    x : ↑X.toTopCat
    ⊢ Eq (e.inv.base (e.hom.base x)) x
  -/
  show (e.hom.base ≫ e.inv.base) x = 𝟙 X.toPresheafedSpace x
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    x : ↑X.toTopCat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.hom.base e.inv.base) x) ((Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma iso_inv_base_hom_base {X Y : LocallyRingedSpace.{u}} (e : X ≅ Y) :
    e.inv.base ≫ e.hom.base = 𝟙 _ := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv.base e.hom.base) (CategoryTheor …
  -/
  rw [← SheafedSpace.comp_base, ← LocallyRingedSpace.comp_toShHom]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (CategoryTheory.Categor …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma iso_inv_base_hom_base_apply {X Y : LocallyRingedSpace.{u}} (e : X ≅ Y) (y : Y) :
    (e.hom.base (e.inv.base y)) = y := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    y : ↑Y.toTopCat
    ⊢ Eq (e.hom.base (e.inv.base y)) y
  -/
  show (e.inv.base ≫ e.hom.base) y = 𝟙 Y.toPresheafedSpace y
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    y : ↑Y.toTopCat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.inv.base e.hom.base) y) ((Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma stalkMap_id (X : LocallyRingedSpace.{u}) (x : X) :
    (𝟙 X : X ⟶ X).stalkMap x = 𝟙 (X.presheaf.stalk x) :=
  PresheafedSpace.stalkMap.id _ x


lemma stalkMap_comp (x : X) :
    (f ≫ g : X ⟶ Z).stalkMap x = g.stalkMap (f.base x) ≫ f.stalkMap x :=
  PresheafedSpace.stalkMap.comp f.toShHom g.toShHom x


@[reassoc]
lemma stalkSpecializes_stalkMap (x x' : X) (h : x ⤳ x') :
    Y.presheaf.stalkSpecializes (f.base.map_specializes h) ≫ f.stalkMap x =
      f.stalkMap x' ≫ X.presheaf.stalkSpecializes h :=
  PresheafedSpace.stalkMap.stalkSpecializes_stalkMap f.toShHom h


lemma stalkSpecializes_stalkMap_apply (x x' : X) (h : x ⤳ x') (y) :
    f.stalkMap x (Y.presheaf.stalkSpecializes (f.base.map_specializes h) y) =
      (X.presheaf.stalkSpecializes h (f.stalkMap x' y)) :=
  DFunLike.congr_fun (CommRingCat.hom_ext_iff.mp (stalkSpecializes_stalkMap f x x' h)) y


@[reassoc]
lemma stalkMap_congr (f g : X ⟶ Y) (hfg : f = g) (x x' : X) (hxx' : x = x') :
    f.stalkMap x ≫ X.presheaf.stalkSpecializes (specializes_of_eq hxx'.symm) =
      Y.presheaf.stalkSpecializes (specializes_of_eq <| hfg ▸ hxx' ▸ rfl) ≫ g.stalkMap x' := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    hfg : Eq f g
    x x' : ↑X.toTopCat
    hxx' : Eq x x'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  subst hfg
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    x x' : ↑X.toTopCat
    hxx' : Eq x x'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  subst hxx'
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
lemma stalkMap_congr_hom (f g : X ⟶ Y) (hfg : f = g) (x : X) :
    f.stalkMap x = Y.presheaf.stalkSpecializes (specializes_of_eq <| hfg ▸ rfl) ≫
      g.stalkMap x := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f g : Quiver.Hom X Y
    hfg : Eq f g
    x : ↑X.toTopCat
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.stalkMap f x) (CategoryTheory.C …
  -/
  subst hfg
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    x : ↑X.toTopCat
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.Hom.stalkMap f x) (CategoryTheory.C …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
lemma stalkMap_congr_point {X Y : LocallyRingedSpace.{u}} (f : X ⟶ Y) (x x' : X) (hxx' : x = x') :
    f.stalkMap x ≫ X.presheaf.stalkSpecializes (specializes_of_eq hxx'.symm) =
      Y.presheaf.stalkSpecializes (specializes_of_eq <| hxx' ▸ rfl) ≫ f.stalkMap x' := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    x x' : ↑X.toTopCat
    hxx' : Eq x x'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  subst hxx'
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma stalkMap_hom_inv (e : X ≅ Y) (y : Y) :
    e.hom.stalkMap (e.inv.base y) ≫ e.inv.stalkMap y =
                                                           /-
                                                             X✝ X Y Z : AlgebraicGeometry.LocallyRingedSpace
                                                             f : Quiver.Hom X Y
                                                             g : Quiver.Hom Y Z
                                                             e : CategoryTheory.Iso X Y
                                                             y : ↑Y.toTopCat
                                                             ⊢ Eq y (e.hom.base (e.inv.base y))
                                                           -/
      Y.presheaf.stalkSpecializes (specializes_of_eq <| by simp) := by
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    y : ↑Y.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  rw [← stalkMap_comp, LocallyRingedSpace.stalkMap_congr_hom (e.inv ≫ e.hom) (𝟙 _) (by simp)]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    y : ↑Y.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.stalkSpecializes ⋯) (Alge …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma stalkMap_hom_inv_apply (e : X ≅ Y) (y : Y) (z) :
    e.inv.stalkMap y (e.hom.stalkMap (e.inv.base y) z) =
                                                           /-
                                                             X✝ X Y Z : AlgebraicGeometry.LocallyRingedSpace
                                                             f : Quiver.Hom X Y
                                                             g : Quiver.Hom Y Z
                                                             e : CategoryTheory.Iso X Y
                                                             y : ↑Y.toTopCat
                                                             z : ↑(Y.presheaf.stalk (e.hom.base (e.inv.base y)))
                                                             ⊢ Eq y (e.hom.base (e.inv.base y))
                                                           -/
      Y.presheaf.stalkSpecializes (specializes_of_eq <| by simp) z :=
                                                           /-
                                                             🎉 no goals
                                                           -/
  DFunLike.congr_fun (CommRingCat.hom_ext_iff.mp (stalkMap_hom_inv e y)) z


@[reassoc (attr := simp)]
lemma stalkMap_inv_hom (e : X ≅ Y) (x : X) :
    e.inv.stalkMap (e.hom.base x) ≫ e.hom.stalkMap x =
                                                           /-
                                                             X✝ X Y Z : AlgebraicGeometry.LocallyRingedSpace
                                                             f : Quiver.Hom X Y
                                                             g : Quiver.Hom Y Z
                                                             e : CategoryTheory.Iso X Y
                                                             x : ↑X.toTopCat
                                                             ⊢ Eq x (e.inv.base (e.hom.base x))
                                                           -/
      X.presheaf.stalkSpecializes (specializes_of_eq <| by simp) := by
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.LocallyRingedSpace …
  -/
  rw [← stalkMap_comp, LocallyRingedSpace.stalkMap_congr_hom (e.hom ≫ e.inv) (𝟙 _) (by simp)]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    e : CategoryTheory.Iso X Y
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.stalkSpecializes ⋯) (Alge …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma stalkMap_inv_hom_apply (e : X ≅ Y) (x : X) (y) :
    e.hom.stalkMap x (e.inv.stalkMap (e.hom.base x) y) =
                                                           /-
                                                             X✝ X Y Z : AlgebraicGeometry.LocallyRingedSpace
                                                             f : Quiver.Hom X Y
                                                             g : Quiver.Hom Y Z
                                                             e : CategoryTheory.Iso X Y
                                                             x : ↑X.toTopCat
                                                             y : ↑(X.presheaf.stalk (e.inv.base (e.hom.base x)))
                                                             ⊢ Eq x (e.inv.base (e.hom.base x))
                                                           -/
      X.presheaf.stalkSpecializes (specializes_of_eq <| by simp) y :=
                                                           /-
                                                             🎉 no goals
                                                           -/
  DFunLike.congr_fun (CommRingCat.hom_ext_iff.mp (stalkMap_inv_hom e x)) y


@[reassoc (attr := simp)]
lemma stalkMap_germ (U : Opens Y) (x : X) (hx : f.base x ∈ U) :
    Y.presheaf.germ U (f.base x) hx ≫ f.stalkMap x =
      f.c.app (op U) ≫ X.presheaf.germ ((Opens.map f.base).obj U) x hx :=
  PresheafedSpace.stalkMap_germ f.toShHom U x hx


lemma stalkMap_germ_apply (U : Opens Y) (x : X) (hx : f.base x ∈ U) (y) :
    f.stalkMap x (Y.presheaf.germ U (f.base x) hx y) =
      X.presheaf.germ ((Opens.map f.base).obj U) x hx (f.c.app (op U) y) :=
  PresheafedSpace.stalkMap_germ_apply f.toShHom U x hx y


theorem preimage_basicOpen {X Y : LocallyRingedSpace.{u}} (f : X ⟶ Y) {U : Opens Y}
    (s : Y.presheaf.obj (op U)) :
    (Opens.map f.base).obj (Y.toRingedSpace.basicOpen s) =
      @RingedSpace.basicOpen X.toRingedSpace ((Opens.map f.base).obj U) (f.c.app _ s) := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑Y.toTopCat
    s : ↑(Y.presheaf.obj { unop := U })
    ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj (Y.toRingedSpace.basicOpen s)) ( …
  -/
  ext x
  /-
    case h.h
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑Y.toTopCat
    s : ↑(Y.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    ⊢ Iff (Membership.mem (↑((TopologicalSpace.Opens.map f.base).obj (Y.toRingedSp …
  -/
  constructor
    /-
      case h.h.mp
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (↑((TopologicalSpace.Opens.map f.base).obj (Y.toRingedSpace.b …
    -/
  · rintro ⟨hxU, hx⟩
    /-
      case h.h.mp.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U (f.base x)
      hx : IsUnit ((Y.toRingedSpace.presheaf.germ U (f.base x) hxU).hom s)
      ⊢ Membership.mem (↑(X.toRingedSpace.basicOpen ((f.c.app { unop := U }).hom s)) …
    -/
    rw [SetLike.mem_coe, X.toRingedSpace.mem_basicOpen _ _ hxU]
    /-
      case h.h.mp.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U (f.base x)
      hx : IsUnit ((Y.toRingedSpace.presheaf.germ U (f.base x) hxU).hom s)
      ⊢ IsUnit ((X.toRingedSpace.presheaf.germ ((TopologicalSpace.Opens.map f.base). …
    -/
    delta toRingedSpace
    /-
      case h.h.mp.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U (f.base x)
      hx : IsUnit ((Y.toRingedSpace.presheaf.germ U (f.base x) hxU).hom s)
      ⊢ IsUnit ((X.presheaf.germ ((TopologicalSpace.Opens.map f.base).obj U) x hxU). …
    -/
    rw [← stalkMap_germ_apply]
    /-
      case h.h.mp.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U (f.base x)
      hx : IsUnit ((Y.toRingedSpace.presheaf.germ U (f.base x) hxU).hom s)
      ⊢ IsUnit ((AlgebraicGeometry.LocallyRingedSpace.Hom.stalkMap f x).hom ((Y.pres …
    -/
    exact (f.stalkMap _).hom.isUnit_map hx
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (↑(X.toRingedSpace.basicOpen ((f.c.app { unop := U }).hom s)) …
    -/
  · rintro ⟨hxU, hx⟩
    /-
      case h.h.mpr.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx : IsUnit ((X.toRingedSpace.presheaf.germ ((TopologicalSpace.Opens.map f.bas …
      ⊢ Membership.mem (↑((TopologicalSpace.Opens.map f.base).obj (Y.toRingedSpace.b …
    -/
    simp only [Opens.map_coe, Set.mem_preimage, SetLike.mem_coe, toRingedSpace] at hx ⊢
    /-
      case h.h.mpr.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx : IsUnit ((X.presheaf.germ ((TopologicalSpace.Opens.map f.base).obj U) x hx …
      ⊢ Membership.mem (AlgebraicGeometry.RingedSpace.basicOpen Y.toSheafedSpace s)  …
    -/
    rw [RingedSpace.mem_basicOpen _ s (f.base x) hxU]
    /-
      case h.h.mpr.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx : IsUnit ((X.presheaf.germ ((TopologicalSpace.Opens.map f.base).obj U) x hx …
      ⊢ IsUnit ((Y.presheaf.germ U (f.base x) hxU).hom s)
    -/
    rw [← stalkMap_germ_apply] at hx
    /-
      case h.h.mpr.intro
      X Y : AlgebraicGeometry.LocallyRingedSpace
      f : Quiver.Hom X Y
      U : TopologicalSpace.Opens ↑Y.toTopCat
      s : ↑(Y.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx : IsUnit ((AlgebraicGeometry.LocallyRingedSpace.Hom.stalkMap f x).hom ((Y.p …
      ⊢ IsUnit ((Y.presheaf.germ U (f.base x) hxU).hom s)
    -/
    exact (isUnit_map_iff (f.toShHom.stalkMap _).hom _).mp hx
    /-
      🎉 no goals
    -/


/-- For an open embedding `f : U ⟶ X` and a point `x : U`, we get an isomorphism between the stalk
of `X` at `f x` and the stalk of the restriction of `X` along `f` at t `x`. -/
noncomputable
def restrictStalkIso : (X.restrict h).presheaf.stalk x ≅ X.presheaf.stalk (f x) :=
  X.toPresheafedSpace.restrictStalkIso h x


@[reassoc (attr := simp)]
lemma restrictStalkIso_hom_eq_germ :
    (X.restrict h).presheaf.germ _ x hx ≫ (X.restrictStalkIso h x).hom =
      X.presheaf.germ (h.isOpenMap.functor.obj V) (f x) ⟨x, hx, rfl⟩ :=
  PresheafedSpace.restrictStalkIso_hom_eq_germ X.toPresheafedSpace h V x hx


lemma restrictStalkIso_hom_eq_germ_apply (y) :
    (X.restrictStalkIso h x).hom ((X.restrict h).presheaf.germ _ x hx y) =
      X.presheaf.germ (h.isOpenMap.functor.obj V) (f x) ⟨x, hx, rfl⟩ y :=
  PresheafedSpace.restrictStalkIso_hom_eq_germ_apply X.toPresheafedSpace h V x hx y


@[reassoc (attr := simp)]
lemma restrictStalkIso_inv_eq_germ :
    X.presheaf.germ (h.isOpenMap.functor.obj V) (f x) ⟨x, hx, rfl⟩ ≫
      (X.restrictStalkIso h x).inv = (X.restrict h).presheaf.germ _ x hx :=
  PresheafedSpace.restrictStalkIso_inv_eq_germ X.toPresheafedSpace h V x hx


lemma restrictStalkIso_inv_eq_germ_apply (y) :
    (X.restrictStalkIso h x).inv
      (X.presheaf.germ (h.isOpenMap.functor.obj V) (f x) ⟨x, hx, rfl⟩ y) =
        (X.restrict h).presheaf.germ _ x hx y :=
  PresheafedSpace.restrictStalkIso_inv_eq_germ_apply X.toPresheafedSpace h V x hx y


lemma restrictStalkIso_inv_eq_ofRestrict :
    (X.restrictStalkIso h x).inv = (X.ofRestrict h).stalkMap x :=
  PresheafedSpace.restrictStalkIso_inv_eq_ofRestrict X.toPresheafedSpace h x


instance ofRestrict_stalkMap_isIso : IsIso ((X.ofRestrict h).stalkMap x) :=
  PresheafedSpace.ofRestrict_stalkMap_isIso X.toPresheafedSpace h x


