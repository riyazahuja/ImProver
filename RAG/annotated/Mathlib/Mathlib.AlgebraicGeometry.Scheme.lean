/-- We define `Scheme` as an `X : LocallyRingedSpace`,
along with a proof that every point has an open neighbourhood `U`
so that the restriction of `X` to `U` is isomorphic,
as a locally ringed space, to `Spec.toLocallyRingedSpace.obj (op R)`
for some `R : CommRingCat`.
-/
structure Scheme extends LocallyRingedSpace where
  local_affine :
    ∀ x : toLocallyRingedSpace,
      ∃ (U : OpenNhds x) (R : CommRingCat),
        Nonempty
          (toLocallyRingedSpace.restrict U.isOpenEmbedding ≅ Spec.toLocallyRingedSpace.obj (op R))


instance : CoeSort Scheme Type* where
  coe X := X.carrier


/-- The type of open sets of a scheme. -/
abbrev Opens (X : Scheme) : Type* := TopologicalSpace.Opens X


/-- A morphism between schemes is a morphism between the underlying locally ringed spaces. -/
structure Hom (X Y : Scheme) extends X.toLocallyRingedSpace.Hom Y.toLocallyRingedSpace where


/-- Cast a morphism of schemes into morphisms of local ringed spaces. -/
abbrev Hom.toLRSHom {X Y : Scheme.{u}} (f : X.Hom Y) :
    X.toLocallyRingedSpace ⟶ Y.toLocallyRingedSpace :=
  f.toHom_1


/-- See Note [custom simps projection] -/
def Hom.Simps.toLRSHom {X Y : Scheme.{u}} (f : X.Hom Y) :
    X.toLocallyRingedSpace ⟶ Y.toLocallyRingedSpace :=
  f.toLRSHom


/-- Schemes are a full subcategory of locally ringed spaces.
-/
instance : Category Scheme where
  id X := Hom.mk (𝟙 X.toLocallyRingedSpace)
  comp f g := Hom.mk (f.toLRSHom ≫ g.toLRSHom)


/-- `f ⁻¹ᵁ U` is notation for `(Opens.map f.base).obj U`,
  the preimage of an open set `U` under `f`. -/
scoped[AlgebraicGeometry] notation3:90 f:91 " ⁻¹ᵁ " U:90 =>
  @Prefunctor.obj (Scheme.Opens _) _ (Scheme.Opens _) _
    (Opens.map (f : Scheme.Hom _ _).base).toPrefunctor U


/-- `Γ(X, U)` is notation for `X.presheaf.obj (op U)`. -/
scoped[AlgebraicGeometry] notation3 "Γ(" X ", " U ")" =>
  (PresheafedSpace.presheaf (SheafedSpace.toPresheafedSpace
    (LocallyRingedSpace.toSheafedSpace (Scheme.toLocallyRingedSpace X)))).obj
    (op (α := Scheme.Opens _) U)


instance {X : Scheme.{u}} : Subsingleton Γ(X, ⊥) :=
  CommRingCat.subsingleton_of_isTerminal X.sheaf.isTerminalOfEmpty


@[continuity, fun_prop]
lemma Hom.continuous {X Y : Scheme} (f : X.Hom Y) : Continuous f.base := f.base.2


/-- The structure sheaf of a scheme. -/
protected abbrev sheaf (X : Scheme) :=
  X.toSheafedSpace.sheaf


/-- Given a morphism of schemes `f : X ⟶ Y`, and open `U ⊆ Y`,
this is the induced map `Γ(Y, U) ⟶ Γ(X, f ⁻¹ᵁ U)`. -/
abbrev app (U : Y.Opens) : Γ(Y, U) ⟶ Γ(X, f ⁻¹ᵁ U) :=
  f.c.app (op U)


/-- Given a morphism of schemes `f : X ⟶ Y`,
this is the induced map `Γ(Y, ⊤) ⟶ Γ(X, ⊤)`. -/
abbrev appTop : Γ(Y, ⊤) ⟶ Γ(X, ⊤) :=
  f.app ⊤


@[reassoc]
lemma naturality (i : op U' ⟶ op U) :
    Y.presheaf.map i ≫ f.app U = f.app U' ≫ X.presheaf.map ((Opens.map f.base).map i.unop).op :=
  f.c.naturality i


/-- Given a morphism of schemes `f : X ⟶ Y`, and open sets `U ⊆ Y`, `V ⊆ f ⁻¹' U`,
this is the induced map `Γ(Y, U) ⟶ Γ(X, V)`. -/
def appLE (U : Y.Opens) (V : X.Opens) (e : V ≤ f ⁻¹ᵁ U) : Γ(Y, U) ⟶ Γ(X, V) :=
  f.app U ≫ X.presheaf.map (homOfLE e).op


@[reassoc (attr := simp)]
lemma appLE_map (e : V ≤ f ⁻¹ᵁ U) (i : op V ⟶ op V') :
    f.appLE U V e ≫ X.presheaf.map i = f.appLE U V' (i.unop.le.trans e) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    U : Y.Opens
    V V' : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    i : Quiver.Hom { unop := V } { unop := V' }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.appLE U V e) (X.presheaf.map i)) ( …
  -/
  rw [Hom.appLE, Category.assoc, ← Functor.map_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    U : Y.Opens
    V V' : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    i : Quiver.Hom { unop := V } { unop := V' }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app U) (X.presheaf.map (CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma appLE_map' (e : V ≤ f ⁻¹ᵁ U) (i : V = V') :
    f.appLE U V' (i ▸ e) ≫ X.presheaf.map (eqToHom i).op = f.appLE U V e :=
  appLE_map _ _ _


@[reassoc (attr := simp)]
lemma map_appLE (e : V ≤ f ⁻¹ᵁ U) (i : op U' ⟶ op U) :
    Y.presheaf.map i ≫ f.appLE U V e =
      f.appLE U' V (e.trans ((Opens.map f.base).map i.unop).le) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    U U' : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    i : Quiver.Hom { unop := U' } { unop := U }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map i) (f.appLE U V e)) ( …
  -/
  rw [Hom.appLE, f.naturality_assoc, ← Functor.map_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    U U' : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    i : Quiver.Hom { unop := U' } { unop := U }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app U') (X.presheaf.map (CategoryT …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma map_appLE' (e : V ≤ f ⁻¹ᵁ U) (i : U' = U) :
    Y.presheaf.map (eqToHom i).op ≫ f.appLE U' V (i ▸ e) = f.appLE U V e :=
  map_appLE _ _ _


lemma app_eq_appLE {U : Y.Opens} :
    f.app U = f.appLE U _ le_rfl := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    U : Y.Opens
    ⊢ Eq (f.app U) (f.appLE U ((TopologicalSpace.Opens.map f.base).obj U) ⋯)
  -/
  simp [Hom.appLE]
  /-
    🎉 no goals
  -/


lemma appLE_eq_app {U : Y.Opens} :
    f.appLE U (f ⁻¹ᵁ U) le_rfl = f.app U :=
  (app_eq_appLE f).symm


lemma appLE_congr (e : V ≤ f ⁻¹ᵁ U) (e₁ : U = U') (e₂ : V = V')
    (P : ∀ {R S : CommRingCat.{u}} (_ : R ⟶ S), Prop) :
    P (f.appLE U V e) ↔ P (f.appLE U' V' (e₁ ▸ e₂ ▸ e)) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    U U' : Y.Opens
    V V' : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    e₁ : Eq U U'
    e₂ : Eq V V'
    P : {R S : CommRingCat} → Quiver.Hom R S → Prop
    ⊢ Iff (P (f.appLE U V e)) (P (f.appLE U' V' ⋯))
  -/
  subst e₁; subst e₂; rfl
                      /-
                        🎉 no goals
                      -/


/-- A morphism of schemes `f : X ⟶ Y` induces a local ring homomorphism from
`Y.presheaf.stalk (f x)` to `X.presheaf.stalk x` for any `x : X`. -/
def stalkMap (x : X) : Y.presheaf.stalk (f.base x) ⟶ X.presheaf.stalk x :=
  f.toLRSHom.stalkMap x


@[ext (iff := false)]
protected lemma ext {f g : X ⟶ Y} (h_base : f.base = g.base)
    (h_app : ∀ U, f.app U ≫ X.presheaf.map
      (eqToHom congr((Opens.map $h_base.symm).obj U)).op = g.app U) : f = g := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X Y
    h_base : Eq f.base g.base
    h_app : ∀ (U : Y.Opens), Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeom …
    ⊢ Eq f g
  -/
  cases f; cases g; congr 1
  exact LocallyRingedSpace.Hom.ext' <| SheafedSpace.ext _ _ h_base
    (TopCat.Presheaf.ext fun U ↦ by simpa using h_app U)


/-- An alternative ext lemma for scheme morphisms. -/
protected lemma ext' {f g : X ⟶ Y} (h : f.toLRSHom = g.toLRSHom) : f = g := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X Y
    h : Eq (AlgebraicGeometry.Scheme.Hom.toLRSHom f) (AlgebraicGeometry.Scheme.Hom …
    ⊢ Eq f g
  -/
  cases f; cases g; congr 1
                    /-
                      🎉 no goals
                    -/


lemma preimage_iSup {ι} (U : ι → Opens Y) : f ⁻¹ᵁ iSup U = ⨆ i, f ⁻¹ᵁ U i :=
                /-
                  X Y : AlgebraicGeometry.Scheme
                  f : X.Hom Y
                  ι : Sort u_1
                  U : ι → Y.Opens
                  ⊢ Eq ↑((TopologicalSpace.Opens.map f.base).obj (iSup U)) ↑(iSup fun i => (Topo …
                -/
  Opens.ext (by simp)
                /-
                  🎉 no goals
                -/


lemma preimage_iSup_eq_top {ι} {U : ι → Opens Y} (hU : iSup U = ⊤) :
    ⨆ i, f ⁻¹ᵁ U i = ⊤ := f.preimage_iSup U ▸ hU ▸ rfl


lemma preimage_le_preimage_of_le {U U' : Y.Opens} (hUU' : U ≤ U') :
    f ⁻¹ᵁ U ≤ f ⁻¹ᵁ U' :=
  fun _ ha ↦ hUU' ha


@[simp]
lemma preimage_comp {X Y Z : Scheme.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) (U) :
    (f ≫ g) ⁻¹ᵁ U = f ⁻¹ᵁ g ⁻¹ᵁ U := rfl


/-- The forgetful functor from `Scheme` to `LocallyRingedSpace`. -/
@[simps!]
def forgetToLocallyRingedSpace : Scheme ⥤ LocallyRingedSpace where
  obj := toLocallyRingedSpace
  map := Hom.toLRSHom


/-- The forget functor `Scheme ⥤ LocallyRingedSpace` is fully faithful. -/
@[simps preimage_toLRSHom]
def fullyFaithfulForgetToLocallyRingedSpace :
    forgetToLocallyRingedSpace.FullyFaithful where
  preimage := Hom.mk


instance : forgetToLocallyRingedSpace.Full :=
  fullyFaithfulForgetToLocallyRingedSpace.full


instance : forgetToLocallyRingedSpace.Faithful :=
  fullyFaithfulForgetToLocallyRingedSpace.faithful


/-- The forgetful functor from `Scheme` to `TopCat`. -/
@[simps!]
def forgetToTop : Scheme ⥤ TopCat :=
  Scheme.forgetToLocallyRingedSpace ⋙ LocallyRingedSpace.forgetToTop


/-- An isomorphism of schemes induces a homeomorphism of the underlying topological spaces. -/
noncomputable def homeoOfIso {X Y : Scheme.{u}} (e : X ≅ Y) : X ≃ₜ Y :=
  TopCat.homeoOfIso (forgetToTop.mapIso e)


@[simp]
lemma homeoOfIso_symm {X Y : Scheme} (e : X ≅ Y) :
    (homeoOfIso e).symm = homeoOfIso e.symm := rfl


@[simp]
lemma homeoOfIso_apply {X Y : Scheme} (e : X ≅ Y) (x : X) :
    homeoOfIso e x = e.hom.base x := rfl


alias _root_.CategoryTheory.Iso.schemeIsoToHomeo := homeoOfIso


/-- An isomorphism of schemes induces a homeomorphism of the underlying topological spaces. -/
noncomputable def Hom.homeomorph {X Y : Scheme.{u}} (f : X.Hom Y) [IsIso (C := Scheme) f] :
    X ≃ₜ Y :=
  (asIso f).schemeIsoToHomeo


@[simp]
lemma Hom.homeomorph_apply {X Y : Scheme.{u}} (f : X.Hom Y) [IsIso (C := Scheme) f] (x) :
    f.homeomorph x = f.base x := rfl

-- Porting note: Lean seems not able to find this coercion any more

instance hasCoeToTopCat : CoeOut Scheme TopCat where
  coe X := X.carrier

-- Porting note: added this unification hint just in case

/-- forgetful functor to `TopCat` is the same as coercion -/
unif_hint forgetToTop_obj_eq_coe (X : Scheme) where ⊢
  forgetToTop.obj X ≟ (X : TopCat)


@[simp]
theorem id.base (X : Scheme) : (𝟙 X : _).base = 𝟙 _ :=
  rfl


@[simp]
theorem id_app {X : Scheme} (U : X.Opens) :
    (𝟙 X : _).app U = 𝟙 _ := rfl


@[simp]
theorem id_appTop {X : Scheme} :
    (𝟙 X : _).appTop = 𝟙 _ :=
  rfl


@[reassoc]
theorem comp_toLRSHom {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).toLRSHom = f.toLRSHom ≫ g.toLRSHom :=
  rfl


@[simp, reassoc] -- reassoc lemma does not need `simp`
theorem comp_coeBase {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).base = f.base ≫ g.base :=
  rfl

-- Porting note: removed elementwise attribute, as generated lemmas were trivial.

@[reassoc]
theorem comp_base {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).base = f.base ≫ g.base :=
  rfl


theorem comp_base_apply {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) :
    (f ≫ g).base x = g.base (f.base x) := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp f g).base x) (g.base (f.base x))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, reassoc] -- reassoc lemma does not need `simp`
theorem comp_app {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) (U) :
    (f ≫ g).app U = g.app U ≫ f.app _ :=
  rfl


@[simp, reassoc] -- reassoc lemma does not need `simp`
theorem comp_appTop {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).appTop = g.appTop ≫ f.appTop :=
  rfl


@[deprecated (since := "2024-06-23")] alias comp_val_c_app := comp_app

@[deprecated (since := "2024-06-23")] alias comp_val_c_app_assoc := comp_app_assoc


theorem appLE_comp_appLE {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) (U V W e₁ e₂) :
    g.appLE U V e₁ ≫ f.appLE V W e₂ =
      (f ≫ g).appLE U W (e₂.trans ((Opens.map f.base).map (homOfLE e₁)).le) := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : Z.Opens
    V : Y.Opens
    W : X.Opens
    e₁ : LE.le V ((TopologicalSpace.Opens.map g.base).obj U)
    e₂ : LE.le W ((TopologicalSpace.Opens.map f.base).obj V)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appLE g …
  -/
  dsimp [Hom.appLE]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : Z.Opens
    V : Y.Opens
    W : X.Opens
    e₁ : LE.le V ((TopologicalSpace.Opens.map g.base).obj U)
    e₂ : LE.le W ((TopologicalSpace.Opens.map f.base).obj V)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, f.naturality_assoc, ← Functor.map_comp]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : Z.Opens
    V : Y.Opens
    W : X.Opens
    e₁ : LE.le V ((TopologicalSpace.Opens.map g.base).obj U)
    e₂ : LE.le W ((TopologicalSpace.Opens.map f.base).obj V)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app g U …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp, reassoc] -- reassoc lemma does not need `simp`
theorem comp_appLE {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) (U V e) :
    (f ≫ g).appLE U V e = g.app U ≫ f.appLE _ V e := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : Z.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStruct.comp f …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE (CategoryTheory.CategoryStruct.comp f …
  -/
  rw [g.app_eq_appLE, appLE_comp_appLE]
  /-
    🎉 no goals
  -/


theorem congr_app {X Y : Scheme} {f g : X ⟶ Y} (e : f = g) (U) :
                                                    /-
                                                      X Y : AlgebraicGeometry.Scheme
                                                      f g : Quiver.Hom X Y
                                                      e : Eq f g
                                                      U : Y.Opens
                                                      ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj U) ((TopologicalSpace.Opens.map  …
                                                    -/
    f.app U = g.app U ≫ X.presheaf.map (eqToHom (by subst e; rfl)).op := by
                                                             /-
                                                               🎉 no goals
                                                             -/
  /-
    X Y : AlgebraicGeometry.Scheme
    f g : Quiver.Hom X Y
    e : Eq f g
    U : Y.Opens
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app f U) (CategoryTheory.CategoryStruct.com …
  -/
  subst e; dsimp; simp
                  /-
                    🎉 no goals
                  -/


theorem app_eq {X Y : Scheme} (f : X ⟶ Y) {U V : Y.Opens} (e : U = V) :
    f.app U =
      Y.presheaf.map (eqToHom e.symm).op ≫
        f.app V ≫
          X.presheaf.map (eqToHom (congr_arg (Opens.map f.base).obj e)).op := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U V : Y.Opens
    e : Eq U V
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app f U) (CategoryTheory.CategoryStruct.com …
  -/
  rw [← IsIso.inv_comp_eq, ← Functor.map_inv, f.naturality]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U V : Y.Opens
    e : Eq U V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app f V …
  -/
  cases e
  /-
    case refl
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app f U …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem eqToHom_c_app {X Y : Scheme} (e : X = Y) (U) :
                                    /-
                                      X Y : AlgebraicGeometry.Scheme
                                      e : Eq X Y
                                      U : Y.Opens
                                      ⊢ Eq (Y.presheaf.obj { unop := U }) (X.presheaf.obj { unop := (TopologicalSpac …
                                    -/
                                             /-
                                               🎉 no goals
                                             -/
    (eqToHom e).app U = eqToHom (by subst e; rfl) := by subst e; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/

-- Porting note: in `AffineScheme.lean` file, `eqToHom_op` can't be used in `(e)rw` or `simp(_rw)`
-- when terms get very complicated. See `AlgebraicGeometry.IsAffineOpen.isLocalization_stalk_aux`.

lemma presheaf_map_eqToHom_op (X : Scheme) (U V : X.Opens) (i : U = V) :
    X.presheaf.map (eqToHom i).op = eqToHom (i ▸ rfl) := by
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    i : Eq U V
    ⊢ Eq (X.presheaf.map (CategoryTheory.eqToHom i).op) (CategoryTheory.eqToHom ⋯)
  -/
  rw [eqToHom_op, eqToHom_map]
  /-
    🎉 no goals
  -/


instance is_locallyRingedSpace_iso {X Y : Scheme} (f : X ⟶ Y) [IsIso f] : IsIso f.toLRSHom :=
  forgetToLocallyRingedSpace.map_isIso f


instance base_isIso {X Y : Scheme.{u}} (f : X ⟶ Y) [IsIso f] : IsIso f.base :=
  Scheme.forgetToTop.map_isIso f

-- Porting note: need an extra instance here.

instance {X Y : Scheme} (f : X ⟶ Y) [IsIso f] (U) : IsIso (f.c.app U) :=
  haveI := PresheafedSpace.c_isIso_of_iso f.toPshHom
  NatIso.isIso_app_of_isIso f.c _


instance {X Y : Scheme} (f : X ⟶ Y) [IsIso f] (U) : IsIso (f.app U) :=
  haveI := PresheafedSpace.c_isIso_of_iso f.toPshHom
  NatIso.isIso_app_of_isIso f.c _


@[simp]
theorem inv_app {X Y : Scheme} (f : X ⟶ Y) [IsIso f] (U : X.Opens) :
    (inv f).app U =
                                                             /-
                                                               X Y : AlgebraicGeometry.Scheme
                                                               f : Quiver.Hom X Y
                                                               inst✝ : CategoryTheory.IsIso f
                                                               U : X.Opens
                                                               ⊢ Eq ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStruct.comp f (Categ …
                                                             -/
      X.presheaf.map (eqToHom (show (f ≫ inv f) ⁻¹ᵁ U = U by rw [IsIso.hom_inv_id]; rfl)).op ≫
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
        inv (f.app ((inv f) ⁻¹ᵁ U)) := by
  rw [IsIso.eq_comp_inv, ← Scheme.comp_app, Scheme.congr_app (IsIso.hom_inv_id f),
    Scheme.id_app, Category.id_comp]


theorem inv_appTop {X Y : Scheme} (f : X ⟶ Y) [IsIso f] :
                                          /-
                                            X Y : AlgebraicGeometry.Scheme
                                            f : Quiver.Hom X Y
                                            inst✝ : CategoryTheory.IsIso f
                                            ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.inv f)) (CategoryThe …
                                          -/
    (inv f).appTop = inv (f.appTop) := by simp
                                          /-
                                            🎉 no goals
                                          -/


@[deprecated (since := "2024-11-23")] alias inv_app_top := inv_appTop


/-- The spectrum of a commutative ring, as a scheme.
-/
def Spec (R : CommRingCat) : Scheme where
  local_affine _ := ⟨⟨⊤, trivial⟩, R, ⟨(Spec.toLocallyRingedSpace.obj (op R)).restrictTopIso⟩⟩
  toLocallyRingedSpace := Spec.locallyRingedSpaceObj R


theorem Spec_toLocallyRingedSpace (R : CommRingCat) :
    (Spec R).toLocallyRingedSpace = Spec.locallyRingedSpaceObj R :=
  rfl


/-- The induced map of a ring homomorphism on the ring spectra, as a morphism of schemes.
-/
def Spec.map {R S : CommRingCat} (f : R ⟶ S) : Spec S ⟶ Spec R :=
  ⟨Spec.locallyRingedSpaceMap f⟩


@[simp]
theorem Spec.map_id (R : CommRingCat) : Spec.map (𝟙 R) = 𝟙 (Spec R) :=
  Scheme.Hom.ext' <| Spec.locallyRingedSpaceMap_id R


@[reassoc, simp]
theorem Spec.map_comp {R S T : CommRingCat} (f : R ⟶ S) (g : S ⟶ T) :
    Spec.map (f ≫ g) = Spec.map g ≫ Spec.map f :=
  Scheme.Hom.ext' <| Spec.locallyRingedSpaceMap_comp f g


/-- The spectrum, as a contravariant functor from commutative rings to schemes. -/
@[simps]
protected def Scheme.Spec : CommRingCatᵒᵖ ⥤ Scheme where
  obj R := Spec (unop R)
  map f := Spec.map f.unop
                 /-
                   R : Opposite CommRingCat
                   ⊢ Eq ({ obj := fun R => AlgebraicGeometry.Spec (Opposite.unop R), map := fun { …
                 -/
  map_id R := by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       X✝ Y✝ Z✝ : Opposite CommRingCat
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun R => AlgebraicGeometry.Spec (Opposite.unop R), map := fun { …
                     -/
  map_comp f g := by simp
                     /-
                       🎉 no goals
                     -/


lemma Spec.map_eqToHom {R S : CommRingCat} (e : R = S) :
    Spec.map (eqToHom e) = eqToHom (e ▸ rfl) := by
  /-
    R S : CommRingCat
    e : Eq R S
    ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.eqToHom e)) (CategoryTheory.e …
  -/
  subst e; exact Spec.map_id _
           /-
             🎉 no goals
           -/


instance {R S : CommRingCat} (f : R ⟶ S) [IsIso f] : IsIso (Spec.map f) :=
  inferInstanceAs (IsIso <| Scheme.Spec.map f.op)


@[simp]
lemma Spec.map_inv {R S : CommRingCat} (f : R ⟶ S) [IsIso f] :
    Spec.map (inv f) = inv (Spec.map f) := by
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.inv f)) (CategoryTheory.inv ( …
  -/
  show Scheme.Spec.map (inv f).op = inv (Scheme.Spec.map f.op)
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (AlgebraicGeometry.Scheme.Spec.map (CategoryTheory.inv f).op) (CategoryTh …
  -/
  rw [op_inv, ← Scheme.Spec.map_inv]
  /-
    🎉 no goals
  -/


lemma Spec_carrier (R : CommRingCat.{u}) : (Spec R).carrier = PrimeSpectrum R := rfl

lemma Spec_sheaf (R : CommRingCat.{u}) : (Spec R).sheaf = Spec.structureSheaf R := rfl

lemma Spec_presheaf (R : CommRingCat.{u}) : (Spec R).presheaf = (Spec.structureSheaf R).1 := rfl

lemma Spec.map_base : (Spec.map f).base = PrimeSpectrum.comap f.hom := rfl

lemma Spec.map_base_apply (x : Spec S) : (Spec.map f).base x = PrimeSpectrum.comap f.hom x := rfl


lemma Spec.map_app (U) :
    (Spec.map f).app U =
      CommRingCat.ofHom (StructureSheaf.comap f.hom U (Spec.map f ⁻¹ᵁ U) le_rfl) := rfl


lemma Spec.map_appLE {U V} (e : U ≤ Spec.map f ⁻¹ᵁ V) :
    (Spec.map f).appLE V U e = CommRingCat.ofHom (StructureSheaf.comap f.hom V U e) := rfl


instance {A : CommRingCat} [Nontrivial A] : Nonempty (Spec A) :=
  inferInstanceAs <| Nonempty (PrimeSpectrum A)


/-- The empty scheme. -/
@[simps]
def empty : Scheme where
  carrier := TopCat.of PEmpty
  presheaf := (CategoryTheory.Functor.const _).obj (CommRingCat.of PUnit)
  IsSheaf := Presheaf.isSheaf_of_isTerminal _ CommRingCat.punitIsTerminal
  isLocalRing x := PEmpty.elim x
  local_affine x := PEmpty.elim x


instance : EmptyCollection Scheme :=
  ⟨empty⟩


instance : Inhabited Scheme :=
  ⟨∅⟩


/-- The global sections, notated Gamma.
-/
def Γ : Schemeᵒᵖ ⥤ CommRingCat :=
  Scheme.forgetToLocallyRingedSpace.op ⋙ LocallyRingedSpace.Γ


theorem Γ_def : Γ = Scheme.forgetToLocallyRingedSpace.op ⋙ LocallyRingedSpace.Γ :=
  rfl


@[simp]
theorem Γ_obj (X : Schemeᵒᵖ) : Γ.obj X = Γ(unop X, ⊤) :=
  rfl


theorem Γ_obj_op (X : Scheme) : Γ.obj (op X) = Γ(X, ⊤) :=
  rfl


@[simp]
theorem Γ_map {X Y : Schemeᵒᵖ} (f : X ⟶ Y) : Γ.map f = f.unop.appTop :=
  rfl


theorem Γ_map_op {X Y : Scheme} (f : X ⟶ Y) : Γ.map f.op = f.appTop :=
  rfl


/--
The counit (`SpecΓIdentity.inv.op`) of the adjunction `Γ ⊣ Spec` as an isomorphism.
This is almost never needed in practical use cases. Use `ΓSpecIso` instead.
-/
def SpecΓIdentity : Scheme.Spec.rightOp ⋙ Scheme.Γ ≅ 𝟭 _ :=
  Iso.symm <| NatIso.ofComponents.{u,u,u+1,u+1}
    (fun R => asIso (StructureSheaf.toOpen R ⊤))
                       /-
                         X Y : CommRingCat
                         f : Quiver.Hom X Y
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id CommRingC …
                       -/
    (fun {X Y} f => by convert Spec_Γ_naturality (R := X) (S := Y) f)
                       /-
                         🎉 no goals
                       -/


/-- The global sections of `Spec R` is isomorphic to `R`. -/
def ΓSpecIso : Γ(Spec R, ⊤) ≅ R := SpecΓIdentity.app R


@[simp] lemma SpecΓIdentity_app : SpecΓIdentity.app R = ΓSpecIso R := rfl

@[simp] lemma SpecΓIdentity_hom_app : SpecΓIdentity.hom.app R = (ΓSpecIso R).hom := rfl

@[simp] lemma SpecΓIdentity_inv_app : SpecΓIdentity.inv.app R = (ΓSpecIso R).inv := rfl


@[reassoc (attr := simp)]
lemma ΓSpecIso_naturality {R S : CommRingCat.{u}} (f : R ⟶ S) :
    (Spec.map f).appTop ≫ (ΓSpecIso S).hom = (ΓSpecIso R).hom ≫ f := SpecΓIdentity.hom.naturality f

-- The RHS is not necessarily simpler than the LHS, but this direction coincides with the simp
-- direction of `NatTrans.naturality`.

@[reassoc (attr := simp)]
lemma ΓSpecIso_inv_naturality {R S : CommRingCat.{u}} (f : R ⟶ S) :
    f ≫ (ΓSpecIso S).inv = (ΓSpecIso R).inv ≫ (Spec.map f).appTop := SpecΓIdentity.inv.naturality f

-- This is not marked simp to respect the abstraction

lemma ΓSpecIso_inv : (ΓSpecIso R).inv = StructureSheaf.toOpen R ⊤ := rfl


lemma toOpen_eq (U) :
        /-
          R : CommRingCat
          U : ?m.163440
          ⊢ Quiver.Hom R ((AlgebraicGeometry.Spec R).presheaf.obj { unop := ?m.164572 })
        -/
    (by exact StructureSheaf.toOpen R U) =
        /-
          🎉 no goals
        -/
    (ΓSpecIso R).inv ≫ (Spec R).presheaf.map (homOfLE le_top).op := rfl


instance {K} [Field K] : Unique (Spec (.of K)) :=
  inferInstanceAs <| Unique (PrimeSpectrum K)


@[simp]
lemma default_asIdeal {K} [Field K] : (default : Spec (.of K)).asIdeal = ⊥ := rfl


/-- The subset of the underlying space where the given section does not vanish. -/
def basicOpen : X.Opens :=
  X.toLocallyRingedSpace.toRingedSpace.basicOpen f


theorem mem_basicOpen (x : X) (hx : x ∈ U) :
    x ∈ X.basicOpen f ↔ IsUnit (X.presheaf.germ U x hx f) :=
  RingedSpace.mem_basicOpen _ _ _ _


/-- A variant of `mem_basicOpen` for bundled `x : U`. -/
@[simp]
theorem mem_basicOpen' (x : U) : ↑x ∈ X.basicOpen f ↔ IsUnit (X.presheaf.germ U x x.2 f) :=
  RingedSpace.mem_basicOpen _ _ _ _


/-- A variant of `mem_basicOpen` without the `x ∈ U` assumption. -/
theorem mem_basicOpen'' {U : X.Opens} (f : Γ(X, U)) (x : X) :
    x ∈ X.basicOpen f ↔ ∃ (m : x ∈ U), IsUnit (X.presheaf.germ U x m f) :=
  Iff.rfl


@[simp]
theorem mem_basicOpen_top (f : Γ(X, ⊤)) (x : X) :
    x ∈ X.basicOpen f ↔ IsUnit (X.presheaf.germ ⊤ x trivial f) :=
  RingedSpace.mem_top_basicOpen _ f x


@[simp]
theorem basicOpen_res (i : op U ⟶ op V) : X.basicOpen (X.presheaf.map i f) = V ⊓ X.basicOpen f :=
  RingedSpace.basicOpen_res _ i f

-- This should fire before `basicOpen_res`.

@[simp 1100]
theorem basicOpen_res_eq (i : op U ⟶ op V) [IsIso i] :
    X.basicOpen (X.presheaf.map i f) = X.basicOpen f :=
  RingedSpace.basicOpen_res_eq _ i f


@[sheaf_restrict]
theorem basicOpen_le : X.basicOpen f ≤ U :=
  RingedSpace.basicOpen_le _ _


@[sheaf_restrict]
lemma basicOpen_restrict (i : V ⟶ U) (f : Γ(X, U)) :
    -- Help `restrict` to infer which forgetful functor we're taking
    X.basicOpen (TopCat.Presheaf.restrict (C := CommRingCat) f i) ≤ X.basicOpen f :=
  (Scheme.basicOpen_res _ _ _).trans_le inf_le_right


@[simp]
theorem preimage_basicOpen {X Y : Scheme.{u}} (f : X ⟶ Y) {U : Y.Opens} (r : Γ(Y, U)) :
    f ⁻¹ᵁ (Y.basicOpen r) = X.basicOpen (f.app U r) :=
  LocallyRingedSpace.preimage_basicOpen f.toLRSHom r


theorem preimage_basicOpen_top {X Y : Scheme.{u}} (f : X ⟶ Y) (r : Γ(Y, ⊤)) :
    f ⁻¹ᵁ (Y.basicOpen r) = X.basicOpen (f.appTop r) :=
  preimage_basicOpen ..


lemma basicOpen_appLE {X Y : Scheme.{u}} (f : X ⟶ Y) (U : X.Opens) (V : Y.Opens) (e : U ≤ f ⁻¹ᵁ V)
    (s : Γ(Y, V)) : X.basicOpen (f.appLE V U e s) = U ⊓ f ⁻¹ᵁ (Y.basicOpen s) := by
  simp only [preimage_basicOpen, Hom.appLE, CommRingCat.comp_apply, RingHom.coe_comp,
    Function.comp_apply]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : X.Opens
    V : Y.Opens
    e : LE.le U ((TopologicalSpace.Opens.map f.base).obj V)
    s : ↑(Y.presheaf.obj { unop := V })
    ⊢ Eq (X.basicOpen ((X.presheaf.map (CategoryTheory.homOfLE e).op).hom ((Algebr …
  -/
  rw [basicOpen_res]
  /-
    🎉 no goals
  -/


@[simp]
theorem basicOpen_zero (U : X.Opens) : X.basicOpen (0 : Γ(X, U)) = ⊥ :=
  LocallyRingedSpace.basicOpen_zero _ U


@[simp]
theorem basicOpen_mul : X.basicOpen (f * g) = X.basicOpen f ⊓ X.basicOpen g :=
  RingedSpace.basicOpen_mul _ _ _


lemma basicOpen_pow {n : ℕ} (h : 0 < n) : X.basicOpen (f ^ n) = X.basicOpen f :=
  RingedSpace.basicOpen_pow _ _ _ h


theorem basicOpen_of_isUnit {f : Γ(X, U)} (hf : IsUnit f) : X.basicOpen f = U :=
  RingedSpace.basicOpen_of_isUnit _ hf


instance algebra_section_section_basicOpen {X : Scheme} {U : X.Opens} (f : Γ(X, U)) :
    Algebra Γ(X, U) Γ(X, X.basicOpen f) :=
  (X.presheaf.map (homOfLE <| X.basicOpen_le f : _ ⟶ U).op).hom.toAlgebra


/--
The zero locus of a set of sections `s` over an open set `U` is the closed set consisting of
the complement of `U` and of all points of `U`, where all elements of `f` vanish.
-/
def zeroLocus {U : X.Opens} (s : Set Γ(X, U)) : Set X := X.toRingedSpace.zeroLocus s


lemma zeroLocus_def {U : X.Opens} (s : Set Γ(X, U)) :
    X.zeroLocus s = ⋂ f ∈ s, (X.basicOpen f).carrierᶜ :=
  rfl


lemma zeroLocus_isClosed {U : X.Opens} (s : Set Γ(X, U)) :
    IsClosed (X.zeroLocus s) :=
  X.toRingedSpace.zeroLocus_isClosed s


lemma zeroLocus_singleton {U : X.Opens} (f : Γ(X, U)) :
    X.zeroLocus {f} = (X.basicOpen f).carrierᶜ :=
  X.toRingedSpace.zeroLocus_singleton f


@[simp]
lemma zeroLocus_empty_eq_univ {U : X.Opens} :
    X.zeroLocus (∅ : Set Γ(X, U)) = Set.univ :=
  X.toRingedSpace.zeroLocus_empty_eq_univ


@[simp]
lemma mem_zeroLocus_iff {U : X.Opens} (s : Set Γ(X, U)) (x : X) :
    x ∈ X.zeroLocus s ↔ ∀ f ∈ s, x ∉ X.basicOpen f :=
  X.toRingedSpace.mem_zeroLocus_iff s x


theorem basicOpen_eq_of_affine {R : CommRingCat} (f : R) :
    (Spec R).basicOpen ((Scheme.ΓSpecIso R).inv f) = PrimeSpectrum.basicOpen f := by
  /-
    R : CommRingCat
    f : ↑R
    ⊢ Eq ((AlgebraicGeometry.Spec R).basicOpen ((AlgebraicGeometry.Scheme.ΓSpecIso …
  -/
  ext x
  /-
    case h.h
    R : CommRingCat
    f : ↑R
    x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    ⊢ Iff (Membership.mem (↑((AlgebraicGeometry.Spec R).basicOpen ((AlgebraicGeome …
  -/
  simp only [SetLike.mem_coe, Scheme.mem_basicOpen_top, Opens.coe_top]
  /-
    case h.h
    R : CommRingCat
    f : ↑R
    x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    ⊢ Iff (IsUnit (((AlgebraicGeometry.Spec R).presheaf.germ Top.top x trivial).ho …
  -/
  suffices IsUnit (StructureSheaf.toStalk R x f) ↔ f ∉ PrimeSpectrum.asIdeal x by exact this
  rw [← isUnit_map_iff (StructureSheaf.stalkToFiberRingHom R x).hom,
    StructureSheaf.stalkToFiberRingHom_toStalk]
  exact
    (IsLocalization.AtPrime.isUnit_to_map_iff (Localization.AtPrime (PrimeSpectrum.asIdeal x))
        (PrimeSpectrum.asIdeal x) f :
      _)


@[simp]
theorem basicOpen_eq_of_affine' {R : CommRingCat} (f : Γ(Spec R, ⊤)) :
    (Spec R).basicOpen f = PrimeSpectrum.basicOpen ((Scheme.ΓSpecIso R).hom f) := by
  /-
    R : CommRingCat
    f : ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := Top.top })
    ⊢ Eq ((AlgebraicGeometry.Spec R).basicOpen f) (PrimeSpectrum.basicOpen ((Algeb …
  -/
  convert basicOpen_eq_of_affine ((Scheme.ΓSpecIso R).hom f)
  /-
    case h.e'_2.h.e'_3
    R : CommRingCat
    f : ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := Top.top })
    ⊢ Eq f ((AlgebraicGeometry.Scheme.ΓSpecIso R).inv.hom ((AlgebraicGeometry.Sche …
  -/
  exact (Iso.hom_inv_id_apply (Scheme.ΓSpecIso R) f).symm
  /-
    🎉 no goals
  -/


theorem Scheme.Spec_map_presheaf_map_eqToHom {X : Scheme} {U V : X.Opens} (h : U = V) (W) :
                                                                   /-
                                                                     X : AlgebraicGeometry.Scheme
                                                                     U V : X.Opens
                                                                     h : Eq U V
                                                                     W : (AlgebraicGeometry.Spec (X.presheaf.obj { unop := V })).Opens
                                                                     ⊢ Eq ((AlgebraicGeometry.Spec (X.presheaf.obj { unop := V })).presheaf.obj { u …
                                                                   -/
    (Spec.map (X.presheaf.map (eqToHom h).op)).app W = eqToHom (by cases h; dsimp; simp) := by
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  have : Scheme.Spec.map (X.presheaf.map (𝟙 (op U))).op = 𝟙 _ := by
    rw [X.presheaf.map_id, op_id, Scheme.Spec.map_id]
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    h : Eq U V
    W : (AlgebraicGeometry.Spec (X.presheaf.obj { unop := V })).Opens
    this : Eq (AlgebraicGeometry.Scheme.Spec.map (X.presheaf.map (CategoryTheory.C …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.Spec.map (X.presheaf …
  -/
  cases h
  /-
    case refl
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    this : Eq (AlgebraicGeometry.Scheme.Spec.map (X.presheaf.map (CategoryTheory.C …
    W : (AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).Opens
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.Spec.map (X.presheaf …
  -/
  refine (Scheme.congr_app this _).trans ?_
  /-
    case refl
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    this : Eq (AlgebraicGeometry.Scheme.Spec.map (X.presheaf.map (CategoryTheory.C …
    W : (AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app (Ca …
  -/
  simp [eqToHom_map]
  /-
    🎉 no goals
  -/


lemma germ_eq_zero_of_pow_mul_eq_zero {X : Scheme.{u}} {U : Opens X} (x : U) {f s : Γ(X, U)}
    (hx : x.val ∈ X.basicOpen s) {n : ℕ} (hf : s ^ n * f = 0) : X.presheaf.germ U x x.2 f = 0 := by
  /-
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : Subtype fun x => Membership.mem U x
    f s : ↑(X.presheaf.obj { unop := U })
    hx : Membership.mem (X.basicOpen s) ↑x
    n : Nat
    hf : Eq (HMul.hMul (HPow.hPow s n) f) 0
    ⊢ Eq ((X.presheaf.germ U ↑x ⋯).hom f) 0
  -/
  rw [Scheme.mem_basicOpen] at hx
  have hu : IsUnit (X.presheaf.germ _ x x.2 (s ^ n)) := by
    rw [map_pow]
    exact IsUnit.pow n hx
  /-
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : Subtype fun x => Membership.mem U x
    f s : ↑(X.presheaf.obj { unop := U })
    hx✝ : Membership.mem (X.basicOpen s) ↑x
    hx : IsUnit ((X.presheaf.germ U ↑x ⋯).hom s)
    n : Nat
    hf : Eq (HMul.hMul (HPow.hPow s n) f) 0
    hu : IsUnit ((X.presheaf.germ U ↑x ⋯).hom (HPow.hPow s n))
    ⊢ Eq ((X.presheaf.germ U ↑x ⋯).hom f) 0
  -/
  rw [← hu.mul_right_eq_zero, ← map_mul, hf, map_zero]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma Scheme.iso_hom_base_inv_base {X Y : Scheme.{u}} (e : X ≅ Y) :
    e.hom.base ≫ e.inv.base = 𝟙 _ :=
  LocallyRingedSpace.iso_hom_base_inv_base (Scheme.forgetToLocallyRingedSpace.mapIso e)


@[simp]
lemma Scheme.iso_hom_base_inv_base_apply {X Y : Scheme.{u}} (e : X ≅ Y) (x : X) :
    (e.inv.base (e.hom.base x)) = x := by
  /-
    X Y : AlgebraicGeometry.Scheme
    e : CategoryTheory.Iso X Y
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq (e.inv.base (e.hom.base x)) x
  -/
  show (e.hom.base ≫ e.inv.base) x = 𝟙 X.toPresheafedSpace x
  /-
    X Y : AlgebraicGeometry.Scheme
    e : CategoryTheory.Iso X Y
    x : ↑↑X.toPresheafedSpace
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.hom.base e.inv.base) x) ((Category …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma Scheme.iso_inv_base_hom_base {X Y : Scheme.{u}} (e : X ≅ Y) :
    e.inv.base ≫ e.hom.base = 𝟙 _ :=
  LocallyRingedSpace.iso_inv_base_hom_base (Scheme.forgetToLocallyRingedSpace.mapIso e)


@[simp]
lemma Scheme.iso_inv_base_hom_base_apply {X Y : Scheme.{u}} (e : X ≅ Y) (y : Y) :
    (e.hom.base (e.inv.base y)) = y := by
  /-
    X Y : AlgebraicGeometry.Scheme
    e : CategoryTheory.Iso X Y
    y : ↑↑Y.toPresheafedSpace
    ⊢ Eq (e.hom.base (e.inv.base y)) y
  -/
  show (e.inv.base ≫ e.hom.base) y = 𝟙 Y.toPresheafedSpace y
  /-
    X Y : AlgebraicGeometry.Scheme
    e : CategoryTheory.Iso X Y
    y : ↑↑Y.toPresheafedSpace
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.inv.base e.hom.base) y) ((Category …
  -/
  simp
  /-
    🎉 no goals
  -/


instance (x) : IsLocalHom (f.stalkMap x).hom :=
  f.prop x


@[simp]
lemma stalkMap_id (X : Scheme.{u}) (x : X) :
    (𝟙 X : X ⟶ X).stalkMap x = 𝟙 (X.presheaf.stalk x) :=
  PresheafedSpace.stalkMap.id _ x


lemma stalkMap_comp {X Y Z : Scheme.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) :
    (f ≫ g : X ⟶ Z).stalkMap x = g.stalkMap (f.base x) ≫ f.stalkMap x :=
  PresheafedSpace.stalkMap.comp f.toPshHom g.toPshHom x


@[reassoc]
lemma stalkSpecializes_stalkMap (x x' : X)
    (h : x ⤳ x') : Y.presheaf.stalkSpecializes (f.base.map_specializes h) ≫ f.stalkMap x =
      f.stalkMap x' ≫ X.presheaf.stalkSpecializes h :=
  PresheafedSpace.stalkMap.stalkSpecializes_stalkMap f.toPshHom h


lemma stalkSpecializes_stalkMap_apply (x x' : X) (h : x ⤳ x') (y) :
    f.stalkMap x (Y.presheaf.stalkSpecializes (f.base.map_specializes h) y) =
      (X.presheaf.stalkSpecializes h (f.stalkMap x' y)) :=
  DFunLike.congr_fun (CommRingCat.hom_ext_iff.mp (stalkSpecializes_stalkMap f x x' h)) y


@[reassoc]
lemma stalkMap_congr (f g : X ⟶ Y) (hfg : f = g) (x x' : X)
    (hxx' : x = x') : f.stalkMap x ≫ (X.presheaf.stalkCongr (.of_eq hxx')).hom =
      (Y.presheaf.stalkCongr (.of_eq <| hfg ▸ hxx' ▸ rfl)).hom ≫ g.stalkMap x' :=
  LocallyRingedSpace.stalkMap_congr f.toLRSHom g.toLRSHom congr(($hfg).toLRSHom) x x' hxx'


@[reassoc]
lemma stalkMap_congr_hom (f g : X ⟶ Y) (hfg : f = g) (x : X) :
    f.stalkMap x = (Y.presheaf.stalkCongr (.of_eq <| hfg ▸ rfl)).hom ≫ g.stalkMap x :=
  LocallyRingedSpace.stalkMap_congr_hom f.toLRSHom g.toLRSHom congr(($hfg).toLRSHom) x


@[reassoc]
lemma stalkMap_congr_point (x x' : X) (hxx' : x = x') :
    f.stalkMap x ≫ (X.presheaf.stalkCongr (.of_eq hxx')).hom =
      (Y.presheaf.stalkCongr (.of_eq <| hxx' ▸ rfl)).hom ≫ f.stalkMap x' :=
  LocallyRingedSpace.stalkMap_congr_point f.toLRSHom x x' hxx'


@[reassoc (attr := simp)]
lemma stalkMap_hom_inv (e : X ≅ Y) (y : Y) :
    e.hom.stalkMap (e.inv.base y) ≫ e.inv.stalkMap y =
                                         /-
                                           X Y : AlgebraicGeometry.Scheme
                                           f : Quiver.Hom X Y
                                           e : CategoryTheory.Iso X Y
                                           y : ↑↑Y.toPresheafedSpace
                                           ⊢ Eq (e.hom.base (e.inv.base y)) y
                                         -/
      (Y.presheaf.stalkCongr (.of_eq (by simp))).hom :=
                                         /-
                                           🎉 no goals
                                         -/
  LocallyRingedSpace.stalkMap_hom_inv (forgetToLocallyRingedSpace.mapIso e) y


@[simp]
lemma stalkMap_hom_inv_apply (e : X ≅ Y) (y : Y) (z) :
    e.inv.stalkMap y (e.hom.stalkMap (e.inv.base y) z) =
                                         /-
                                           X Y : AlgebraicGeometry.Scheme
                                           f : Quiver.Hom X Y
                                           e : CategoryTheory.Iso X Y
                                           y : ↑↑Y.toPresheafedSpace
                                           z : ↑(Y.presheaf.stalk (e.hom.base (e.inv.base y)))
                                           ⊢ Eq (e.hom.base (e.inv.base y)) y
                                         -/
      (Y.presheaf.stalkCongr (.of_eq (by simp))).hom z :=
                                         /-
                                           🎉 no goals
                                         -/
  DFunLike.congr_fun (CommRingCat.hom_ext_iff.mp (stalkMap_hom_inv e y)) z


@[reassoc (attr := simp)]
lemma stalkMap_inv_hom (e : X ≅ Y) (x : X) :
    e.inv.stalkMap (e.hom.base x) ≫ e.hom.stalkMap x =
                                         /-
                                           X Y : AlgebraicGeometry.Scheme
                                           f : Quiver.Hom X Y
                                           e : CategoryTheory.Iso X Y
                                           x : ↑↑X.toPresheafedSpace
                                           ⊢ Eq (e.inv.base (e.hom.base x)) x
                                         -/
      (X.presheaf.stalkCongr (.of_eq (by simp))).hom :=
                                         /-
                                           🎉 no goals
                                         -/
  LocallyRingedSpace.stalkMap_inv_hom (forgetToLocallyRingedSpace.mapIso e) x


@[simp]
lemma stalkMap_inv_hom_apply (e : X ≅ Y) (x : X) (y) :
    e.hom.stalkMap x (e.inv.stalkMap (e.hom.base x) y) =
                                         /-
                                           X Y : AlgebraicGeometry.Scheme
                                           f : Quiver.Hom X Y
                                           e : CategoryTheory.Iso X Y
                                           x : ↑↑X.toPresheafedSpace
                                           y : ↑(X.presheaf.stalk (e.inv.base (e.hom.base x)))
                                           ⊢ Eq (e.inv.base (e.hom.base x)) x
                                         -/
      (X.presheaf.stalkCongr (.of_eq (by simp))).hom y :=
                                         /-
                                           🎉 no goals
                                         -/
  DFunLike.congr_fun (CommRingCat.hom_ext_iff.mp (stalkMap_inv_hom e x)) y


@[reassoc (attr := simp)]
lemma stalkMap_germ (U : Y.Opens) (x : X) (hx : f.base x ∈ U) :
    Y.presheaf.germ U (f.base x) hx ≫ f.stalkMap x =
      f.app U ≫ X.presheaf.germ (f ⁻¹ᵁ U) x hx :=
  PresheafedSpace.stalkMap_germ f.toPshHom U x hx


@[simp]
lemma stalkMap_germ_apply (U : Y.Opens) (x : X) (hx : f.base x ∈ U) (y) :
    f.stalkMap x (Y.presheaf.germ _ (f.base x) hx y) =
      X.presheaf.germ (f ⁻¹ᵁ U) x hx (f.app U y) :=
  PresheafedSpace.stalkMap_germ_apply f.toPshHom U x hx y


@[simp]
lemma Spec_closedPoint {R S : CommRingCat} [IsLocalRing R] [IsLocalRing S]
    {f : R ⟶ S} [IsLocalHom f.hom] : (Spec.map f).base (closedPoint S) = closedPoint R :=
  IsLocalRing.comap_closedPoint f.hom


