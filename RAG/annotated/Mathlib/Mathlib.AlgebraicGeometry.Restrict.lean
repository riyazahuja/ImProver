/-- Open subset of a scheme as a scheme. -/
@[coe]
def toScheme {X : Scheme.{u}} (U : X.Opens) : Scheme.{u} :=
  X.restrict U.isOpenEmbedding


instance : CoeOut X.Opens Scheme := ⟨toScheme⟩


/-- The restriction of a scheme to an open subset. -/
@[simps! base_apply]
def ι : ↑U ⟶ X := X.ofRestrict _


instance : IsOpenImmersion U.ι := inferInstanceAs (IsOpenImmersion (X.ofRestrict _))


@[simps! over] instance : U.toScheme.CanonicallyOver X where
  hom := U.ι


instance (U : X.Opens) : U.ι.IsOver X where


lemma toScheme_carrier : (U : Type u) = (U : Set X) := rfl


lemma toScheme_presheaf_obj (V) : Γ(U, V) = Γ(X, U.ι ''ᵁ V) := rfl


@[simp]
lemma toScheme_presheaf_map {V W} (i : V ⟶ W) :
    U.toScheme.presheaf.map i = X.presheaf.map (U.ι.opensFunctor.map i.unop).op := rfl


@[simp]
lemma ι_app (V) : U.ι.app V = X.presheaf.map
    (homOfLE (x := U.ι ''ᵁ U.ι ⁻¹ᵁ V) (Set.image_preimage_subset _ _)).op :=
  rfl


@[simp]
lemma ι_appTop :
    U.ι.appTop = X.presheaf.map (homOfLE (x := U.ι ''ᵁ ⊤) le_top).op :=
  rfl


@[simp]
lemma ι_appLE (V W e) :
    U.ι.appLE V W e =
      X.presheaf.map (homOfLE (x := U.ι ''ᵁ W) (Set.image_subset_iff.mpr ‹_›)).op := by
  simp only [Hom.appLE, ι_app, Functor.op_obj, Opens.carrier_eq_coe, toScheme_presheaf_map,
    Quiver.Hom.unop_op, Hom.opensFunctor_map_homOfLE, Opens.coe_inclusion', ← Functor.map_comp]
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    W : (↑U).Opens
    e : LE.le W ((TopologicalSpace.Opens.map U.ι.base).obj V)
    ⊢ Eq (X.presheaf.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOf …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma ι_appIso (V) : U.ι.appIso V = Iso.refl _ :=
  X.ofRestrict_appIso _ _


@[simp]
lemma opensRange_ι : U.ι.opensRange = U :=
  Opens.ext Subtype.range_val


@[simp]
lemma range_ι : Set.range U.ι.base = U :=
  Subtype.range_val


lemma ι_image_top : U.ι ''ᵁ ⊤ = U :=
  U.isOpenEmbedding_obj_top


lemma ι_image_le (W : U.toScheme.Opens) : U.ι ''ᵁ W ≤ U := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    W : (↑U).Opens
    ⊢ LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj W) U
  -/
  simp_rw [← U.ι_image_top]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    W : (↑U).Opens
    ⊢ LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj W) ((AlgebraicGeo …
  -/
  exact U.ι.image_le_image_of_le le_top
  /-
    🎉 no goals
  -/


@[simp]
lemma ι_preimage_self : U.ι ⁻¹ᵁ U = ⊤ :=
  Opens.inclusion'_map_eq_top _


instance ι_appLE_isIso :
    IsIso (U.ι.appLE U ⊤ U.ι_preimage_self.ge) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.appLE U.ι U Top.top ⋯)
  -/
  simp only [ι, ofRestrict_appLE]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ CategoryTheory.IsIso (X.presheaf.map (CategoryTheory.homOfLE ⋯).op)
  -/
  show IsIso (X.presheaf.map (eqToIso U.ι_image_top).hom.op)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ CategoryTheory.IsIso (X.presheaf.map (CategoryTheory.eqToIso ⋯).hom.op)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


                                                                            /-
                                                                              C : Type u₁
                                                                              inst✝ : CategoryTheory.Category.{v, u₁} C
                                                                              X : AlgebraicGeometry.Scheme
                                                                              U : X.Opens
                                                                              ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj (Opposite.unop { uno …
                                                                            -/
lemma ι_app_self : U.ι.app U = X.presheaf.map (eqToHom (X := U.ι ''ᵁ _) (by simp)).op := rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


lemma eq_presheaf_map_eqToHom {V W : Opens U} (e : U.ι ''ᵁ V = U.ι ''ᵁ W) :
    X.presheaf.map (eqToHom e).op =
      U.toScheme.presheaf.map (eqToHom <| U.isOpenEmbedding.functor_obj_injective e).op := rfl


@[simp]
lemma nonempty_iff : Nonempty U.toScheme ↔ (U : Set X).Nonempty := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Iff (Nonempty ↑↑(↑U).toPresheafedSpace) (↑U).Nonempty
  -/
  simp only [toScheme_carrier, SetLike.coe_sort_coe, nonempty_subtype]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Iff (Exists fun x => Membership.mem U x) (↑U).Nonempty
  -/
  rfl
  /-
    🎉 no goals
  -/


attribute [-simp] eqToHom_op in
/-- The global sections of the restriction is isomorphic to the sections on the open set. -/
@[simps!]
def topIso : Γ(U, ⊤) ≅ Γ(X, U) :=
  X.presheaf.mapIso (eqToIso U.ι_image_top.symm).op


/-- The stalks of an open subscheme are isomorphic to the stalks of the original scheme. -/
def stalkIso {X : Scheme.{u}} (U : X.Opens) (x : U) :
    U.toScheme.presheaf.stalk x ≅ X.presheaf.stalk x.1 :=
  X.restrictStalkIso (Opens.isOpenEmbedding _) _


@[reassoc (attr := simp)]
lemma germ_stalkIso_hom {X : Scheme.{u}} (U : X.Opens)
    {V : U.toScheme.Opens} (x : U) (hx : x ∈ V) :
      U.toScheme.presheaf.germ V x hx ≫ (U.stalkIso x).hom =
        X.presheaf.germ (U.ι ''ᵁ V) x.1 ⟨x, hx, rfl⟩ :=
    PresheafedSpace.restrictStalkIso_hom_eq_germ _ U.isOpenEmbedding _ _ _


@[reassoc]
lemma germ_stalkIso_inv {X : Scheme.{u}} (U : X.Opens) (V : U.toScheme.Opens) (x : U)
    (hx : x ∈ V) : X.presheaf.germ (U.ι ''ᵁ V) x ⟨x, hx, rfl⟩ ≫
      (U.stalkIso x).inv = U.toScheme.presheaf.germ V x hx :=
  PresheafedSpace.restrictStalkIso_inv_eq_germ X.toPresheafedSpace U.isOpenEmbedding V x hx


/-- If `U` is a family of open sets that covers `X`, then `X.restrict U` forms an `X.open_cover`. -/
@[simps! J obj map]
def Scheme.openCoverOfISupEqTop {s : Type*} (X : Scheme.{u}) (U : s → X.Opens)
    (hU : ⨆ i, U i = ⊤) : X.OpenCover where
  J := s
  obj i := U i
  map i := (U i).ι
  f x :=
                                                                /-
                                                                  C : Type u₁
                                                                  inst✝ : CategoryTheory.Category.{v, u₁} C
                                                                  X✝ : AlgebraicGeometry.Scheme
                                                                  U✝ : X✝.Opens
                                                                  s : Type u_1
                                                                  X : AlgebraicGeometry.Scheme
                                                                  U : s → X.Opens
                                                                  hU : Eq (iSup fun i => U i) Top.top
                                                                  x : ↑↑X.toPresheafedSpace
                                                                  ⊢ Membership.mem Top.top x
                                                                -/
    haveI : x ∈ ⨆ i, U i := hU.symm ▸ show x ∈ (⊤ : X.Opens) by trivial
                                                                /-
                                                                  🎉 no goals
                                                                -/
    (Opens.mem_iSup.mp this).choose
  covers x := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X✝ : AlgebraicGeometry.Scheme
      U✝ : X✝.Opens
      s : Type u_1
      X : AlgebraicGeometry.Scheme
      U : s → X.Opens
      hU : Eq (iSup fun i => U i) Top.top
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun i => (U i).ι) ((fun x => ⋯.choose) x)).base …
    -/
    erw [Subtype.range_coe]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X✝ : AlgebraicGeometry.Scheme
      U✝ : X✝.Opens
      s : Type u_1
      X : AlgebraicGeometry.Scheme
      U : s → X.Opens
      hU : Eq (iSup fun i => U i) Top.top
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (↑(U ((fun x => ⋯.choose) x))) x
    -/
    have : x ∈ ⨆ i, U i := hU.symm ▸ show x ∈ (⊤ : X.Opens) by trivial
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X✝ : AlgebraicGeometry.Scheme
      U✝ : X✝.Opens
      s : Type u_1
      X : AlgebraicGeometry.Scheme
      U : s → X.Opens
      hU : Eq (iSup fun i => U i) Top.top
      x : ↑↑X.toPresheafedSpace
      this : Membership.mem (iSup fun i => U i) x
      ⊢ Membership.mem (↑(U ((fun x => ⋯.choose) x))) x
    -/
    exact (Opens.mem_iSup.mp this).choose_spec
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-24")]
noncomputable alias Scheme.openCoverOfSuprEqTop := Scheme.openCoverOfISupEqTop


/-- The open sets of an open subscheme corresponds to the open sets containing in the subset. -/
@[simps!]
def opensRestrict :
    Scheme.Opens U ≃ { V : X.Opens // V ≤ U } :=
                                                                       /-
                                                                         C : Type u₁
                                                                         inst✝ : CategoryTheory.Category.{v, u₁} C
                                                                         X : AlgebraicGeometry.Scheme
                                                                         U : X.Opens
                                                                         ⊢ Eq (fun U_1 => LE.le U_1 (AlgebraicGeometry.Scheme.Hom.opensRange U.ι)) fun  …
                                                                       -/
  (IsOpenImmersion.opensEquiv (U.ι)).trans (Equiv.subtypeEquivProp (by simp))
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance ΓRestrictAlgebra {X : Scheme.{u}} (U : X.Opens) :
    Algebra (Γ(X, ⊤)) Γ(U, ⊤) :=
  U.ι.appTop.hom.toAlgebra


lemma Scheme.map_basicOpen (r : Γ(U, ⊤)) :
    U.ι ''ᵁ U.toScheme.basicOpen r = X.basicOpen
      (X.presheaf.map (eqToHom U.isOpenEmbedding_obj_top.symm).op r) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    r : ↑((↑U).presheaf.obj { unop := Top.top })
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen r))  …
  -/
  refine (Scheme.image_basicOpen (X.ofRestrict U.isOpenEmbedding) r).trans ?_
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    r : ↑((↑U).presheaf.obj { unop := Top.top })
    ⊢ Eq (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appIso (X.ofRestrict ⋯) Top.t …
  -/
  rw [← Scheme.basicOpen_res_eq _ _ (eqToHom U.isOpenEmbedding_obj_top).op]
  rw [← CommRingCat.comp_apply, ← CategoryTheory.Functor.map_comp, ← op_comp, eqToHom_trans,
    eqToHom_refl, op_id, CategoryTheory.Functor.map_id]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    r : ↑((↑U).presheaf.obj { unop := Top.top })
    ⊢ Eq (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appIso (X.ofRestrict ⋯) Top.t …
  -/
  congr
  /-
    case e_f.e_a.e_self
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    r : ↑((↑U).presheaf.obj { unop := Top.top })
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appIso (X.ofRestrict ⋯) Top.top).inv (Categ …
  -/
  exact PresheafedSpace.IsOpenImmersion.ofRestrict_invApp _ _ _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-23")] alias Scheme.map_basicOpen' := Scheme.map_basicOpen


lemma Scheme.Opens.ι_image_basicOpen (r : Γ(U, ⊤)) :
    U.ι ''ᵁ U.toScheme.basicOpen r = X.basicOpen r := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    r : ↑((↑U).presheaf.obj { unop := Top.top })
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen r))  …
  -/
  rw [Scheme.map_basicOpen, Scheme.basicOpen_res_eq]
  /-
    🎉 no goals
  -/


lemma Scheme.map_basicOpen_map (r : Γ(X, U)) :
    U.ι ''ᵁ (U.toScheme.basicOpen <| U.topIso.inv r) = X.basicOpen r := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    r : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen (U.t …
  -/
  simp only [Scheme.Opens.toScheme_presheaf_obj]
  rw [Scheme.map_basicOpen, Scheme.basicOpen_res_eq, Scheme.Opens.topIso_inv,
    Scheme.basicOpen_res_eq X]


/-- If `U ≤ V`, then `U` is also a subscheme of `V`. -/
protected noncomputable
def Scheme.homOfLE (X : Scheme.{u}) {U V : X.Opens} (e : U ≤ V) : (U : Scheme.{u}) ⟶ V :=
                                   /-
                                     C : Type u₁
                                     inst✝ : CategoryTheory.Category.{v, u₁} C
                                     X✝ : AlgebraicGeometry.Scheme
                                     U✝ : X✝.Opens
                                     X : AlgebraicGeometry.Scheme
                                     U V : X.Opens
                                     e : LE.le U V
                                     ⊢ HasSubset.Subset (Set.range ⇑U.ι.base) (Set.range ⇑V.ι.base)
                                   -/
  IsOpenImmersion.lift V.ι U.ι (by simpa using e)
                                   /-
                                     🎉 no goals
                                   -/


@[reassoc (attr := simp)]
lemma Scheme.homOfLE_ι (X : Scheme.{u}) {U V : X.Opens} (e : U ≤ V) :
    X.homOfLE e ≫ V.ι = U.ι :=
  IsOpenImmersion.lift_fac _ _ _


instance {U V : X.Opens} (h : U ≤ V) : (X.homOfLE h).IsOver X where


@[simp]
lemma Scheme.homOfLE_rfl (X : Scheme.{u}) (U : X.Opens) : X.homOfLE (refl U) = 𝟙 _ := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Eq (X.homOfLE ⋯) (CategoryTheory.CategoryStruct.id ↑U)
  -/
  rw [← cancel_mono U.ι, Scheme.homOfLE_ι, Category.id_comp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma Scheme.homOfLE_homOfLE (X : Scheme.{u}) {U V W : X.Opens} (e₁ : U ≤ V) (e₂ : V ≤ W) :
    X.homOfLE e₁ ≫ X.homOfLE e₂ = X.homOfLE (e₁.trans e₂) := by
  /-
    X : AlgebraicGeometry.Scheme
    U V W : X.Opens
    e₁ : LE.le U V
    e₂ : LE.le V W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE e₁) (X.homOfLE e₂)) (X.hom …
  -/
  rw [← cancel_mono W.ι, Category.assoc, Scheme.homOfLE_ι, Scheme.homOfLE_ι, Scheme.homOfLE_ι]
  /-
    🎉 no goals
  -/


theorem Scheme.homOfLE_base {U V : X.Opens} (e : U ≤ V) :
    (X.homOfLE e).base = (Opens.toTopCat _).map (homOfLE e) := by
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    ⊢ Eq (X.homOfLE e).base ((TopologicalSpace.Opens.toTopCat ↑X.toPresheafedSpace …
  -/
  ext a; refine Subtype.ext ?_ -- Porting note: `ext` did not pick up `Subtype.ext`
  /-
    case w
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    a : (CategoryTheory.forget TopCat).obj ↑(↑U).toPresheafedSpace
    ⊢ Eq ↑((X.homOfLE e).base a) ↑(((TopologicalSpace.Opens.toTopCat ↑X.toPresheaf …
  -/
  exact congr($(X.homOfLE_ι e).base a)
  /-
    🎉 no goals
  -/


@[simp]
theorem Scheme.homOfLE_apply {U V : X.Opens} (e : U ≤ V) (x : U) :
    ((X.homOfLE e).base x).1 = x := by
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq ↑((X.homOfLE e).base x) ↑x
  -/
  rw [homOfLE_base]
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq ↑(((TopologicalSpace.Opens.toTopCat ↑X.toPresheafedSpace).map (CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Scheme.ι_image_homOfLE_le_ι_image {U V : X.Opens} (e : U ≤ V) (W : Opens V) :
    U.ι ''ᵁ (X.homOfLE e ⁻¹ᵁ W) ≤ V.ι ''ᵁ W := by
  simp only [← SetLike.coe_subset_coe, IsOpenMap.coe_functor_obj, Set.image_subset_iff,
    Scheme.homOfLE_base, Opens.map_coe, Opens.inclusion'_apply]
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    ⊢ HasSubset.Subset (Set.preimage ⇑((TopologicalSpace.Opens.toTopCat ↑X.toPresh …
  -/
  rintro _ h
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    a✝ : ↑↑(↑U).toPresheafedSpace
    h : Membership.mem (Set.preimage ⇑((TopologicalSpace.Opens.toTopCat ↑X.toPresh …
    ⊢ Membership.mem (Set.preimage (⇑(AlgebraicGeometry.Scheme.Hom.toLRSHom U.ι).b …
  -/
  exact ⟨_, h, rfl⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem Scheme.homOfLE_app {U V : X.Opens} (e : U ≤ V) (W : Opens V) :
    (X.homOfLE e).app W =
      X.presheaf.map (homOfLE <| X.ι_image_homOfLE_le_ι_image e W).op := by
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (X.presheaf.map (Categ …
  -/
  have e₁ := Scheme.congr_app (X.homOfLE_ι e) (V.ι ''ᵁ W)
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    e₁ : Eq (AlgebraicGeometry.Scheme.Hom.app (CategoryTheory.CategoryStruct.comp  …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (X.presheaf.map (Categ …
  -/
  have : V.ι ⁻¹ᵁ V.ι ''ᵁ W = W := W.map_functor_eq (U := V)
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    e₁ : Eq (AlgebraicGeometry.Scheme.Hom.app (CategoryTheory.CategoryStruct.comp  …
    this : Eq ((TopologicalSpace.Opens.map V.ι.base).obj ((AlgebraicGeometry.Schem …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (X.presheaf.map (Categ …
  -/
  have e₂ := (X.homOfLE e).naturality (eqToIso this).hom.op
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    e₁ : Eq (AlgebraicGeometry.Scheme.Hom.app (CategoryTheory.CategoryStruct.comp  …
    this : Eq ((TopologicalSpace.Opens.map V.ι.base).obj ((AlgebraicGeometry.Schem …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp ((↑V).presheaf.map (CategoryTheory …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (X.presheaf.map (Categ …
  -/
  have e₃ := e₂.symm.trans e₁
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    e₁ : Eq (AlgebraicGeometry.Scheme.Hom.app (CategoryTheory.CategoryStruct.comp  …
    this : Eq ((TopologicalSpace.Opens.map V.ι.base).obj ((AlgebraicGeometry.Schem …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp ((↑V).presheaf.map (CategoryTheory …
    e₃ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app  …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (X.presheaf.map (Categ …
  -/
  dsimp at e₃ ⊢
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    e₁ : Eq (AlgebraicGeometry.Scheme.Hom.app (CategoryTheory.CategoryStruct.comp  …
    this : Eq ((TopologicalSpace.Opens.map V.ι.base).obj ((AlgebraicGeometry.Schem …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp ((↑V).presheaf.map (CategoryTheory …
    e₃ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app  …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (X.presheaf.map (Categ …
  -/
  rw [← IsIso.eq_comp_inv, ← Functor.map_inv, ← Functor.map_comp] at e₃
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    e₁ : Eq (AlgebraicGeometry.Scheme.Hom.app (CategoryTheory.CategoryStruct.comp  …
    this : Eq ((TopologicalSpace.Opens.map V.ι.base).obj ((AlgebraicGeometry.Schem …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp ((↑V).presheaf.map (CategoryTheory …
    e₃ : Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (CategoryTheory.Cat …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (X.presheaf.map (Categ …
  -/
  rw [e₃, ← Functor.map_comp]
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    W : (↑V).Opens
    e₁ : Eq (AlgebraicGeometry.Scheme.Hom.app (CategoryTheory.CategoryStruct.comp  …
    this : Eq ((TopologicalSpace.Opens.map V.ι.base).obj ((AlgebraicGeometry.Schem …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp ((↑V).presheaf.map (CategoryTheory …
    e₃ : Eq (AlgebraicGeometry.Scheme.Hom.app (X.homOfLE e) W) (CategoryTheory.Cat …
    ⊢ Eq (X.presheaf.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.Categ …
  -/
  congr 1
  /-
    🎉 no goals
  -/


theorem Scheme.homOfLE_appTop {U V : X.Opens} (e : U ≤ V) :
    (X.homOfLE e).appTop =
      X.presheaf.map (homOfLE <| X.ι_image_homOfLE_le_ι_image e ⊤).op :=
  homOfLE_app ..


instance (X : Scheme.{u}) {U V : X.Opens} (e : U ≤ V) : IsOpenImmersion (X.homOfLE e) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X✝ : AlgebraicGeometry.Scheme
    U✝ : X✝.Opens
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    ⊢ AlgebraicGeometry.IsOpenImmersion (X.homOfLE e)
  -/
  delta Scheme.homOfLE
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X✝ : AlgebraicGeometry.Scheme
    U✝ : X✝.Opens
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    e : LE.le U V
    ⊢ AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.IsOpenImmersion.lift V. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/

-- Porting note: `simps` can't synthesize `obj_left, obj_hom, mapLeft`

variable (X) in
/-- The functor taking open subsets of `X` to open subschemes of `X`. -/
-- @[simps obj_left obj_hom mapLeft]
def Scheme.restrictFunctor : X.Opens ⥤ Over X where
  obj U := Over.mk U.ι
                                                 /-
                                                   C : Type u₁
                                                   inst✝ : CategoryTheory.Category.{v, u₁} C
                                                   X : AlgebraicGeometry.Scheme
                                                   U✝ U V : X.Opens
                                                   i : Quiver.Hom U V
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE ⋯) ((fun U => CategoryTheo …
                                                 -/
  map {U V} i := Over.homMk (X.homOfLE i.le) (by simp)
                                                 /-
                                                   🎉 no goals
                                                 -/
  map_id U := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      U✝ U : X.Opens
      ⊢ Eq ({ obj := fun U => CategoryTheory.Over.mk U.ι, map := fun {U V} i => Cate …
    -/
    ext1
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      U✝ U : X.Opens
      ⊢ Eq ({ obj := fun U => CategoryTheory.Over.mk U.ι, map := fun {U V} i => Cate …
    -/
    exact Scheme.homOfLE_rfl _ _
    /-
      🎉 no goals
    -/
  map_comp {U V W} i j := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      U✝ U V W : X.Opens
      i : Quiver.Hom U V
      j : Quiver.Hom V W
      ⊢ Eq ({ obj := fun U => CategoryTheory.Over.mk U.ι, map := fun {U V} i => Cate …
    -/
    ext1
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      U✝ U V W : X.Opens
      i : Quiver.Hom U V
      j : Quiver.Hom V W
      ⊢ Eq ({ obj := fun U => CategoryTheory.Over.mk U.ι, map := fun {U V} i => Cate …
    -/
    exact (X.homOfLE_homOfLE i.le j.le).symm
    /-
      🎉 no goals
    -/


@[simp] lemma Scheme.restrictFunctor_obj_left (U : X.Opens) :
  (X.restrictFunctor.obj U).left = U := rfl


@[simp] lemma Scheme.restrictFunctor_obj_hom (U : X.Opens) :
  (X.restrictFunctor.obj U).hom = U.ι := rfl


@[simp]
lemma Scheme.restrictFunctor_map_left {U V : X.Opens} (i : U ⟶ V) :
    (X.restrictFunctor.map i).left = (X.homOfLE i.le) := rfl


@[deprecated (since := "2024-10-20")]
alias Scheme.restrictFunctor_map_ofRestrict := Scheme.homOfLE_ι

@[deprecated (since := "2024-10-20")]
alias Scheme.restrictFunctor_map_ofRestrict_assoc := Scheme.homOfLE_ι_assoc


@[deprecated (since := "2024-10-20")]
alias Scheme.restrictFunctor_map_base := Scheme.homOfLE_base

@[deprecated (since := "2024-10-20")]
alias Scheme.restrictFunctor_map_app_aux := Scheme.ι_image_homOfLE_le_ι_image

@[deprecated (since := "2024-10-20")]
alias Scheme.restrictFunctor_map_app := Scheme.homOfLE_app


/-- The functor that restricts to open subschemes and then takes global section is
isomorphic to the structure sheaf. -/
@[simps!]
def Scheme.restrictFunctorΓ : X.restrictFunctor.op ⋙ (Over.forget X).op ⋙ Scheme.Γ ≅ X.presheaf :=
  NatIso.ofComponents
    (fun U => X.presheaf.mapIso ((eqToIso (unop U).isOpenEmbedding_obj_top).symm.op : _))
    (by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v, u₁} C
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        ⊢ ∀ {X_1 Y : Opposite X.Opens} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.Cate …
      -/
      intro U V i
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v, u₁} C
        X : AlgebraicGeometry.Scheme
        U✝ : X.Opens
        U V : Opposite X.Opens
        i : Quiver.Hom U V
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((X.restrictFunctor.op.comp ((Categor …
      -/
      dsimp
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v, u₁} C
        X : AlgebraicGeometry.Scheme
        U✝ : X.Opens
        U V : Opposite X.Opens
        i : Quiver.Hom U V
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appTop  …
      -/
      rw [X.homOfLE_appTop, ← Functor.map_comp, ← Functor.map_comp]
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v, u₁} C
        X : AlgebraicGeometry.Scheme
        U✝ : X.Opens
        U V : Opposite X.Opens
        i : Quiver.Hom U V
        ⊢ Eq (X.presheaf.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOf …
      -/
      congr 1)
      /-
        🎉 no goals
      -/


/-- `X ∣_ U ∣_ V` is isomorphic to `X ∣_ V ∣_ U` -/
noncomputable
def Scheme.restrictRestrictComm (X : Scheme.{u}) (U V : X.Opens) :
    (U.ι ⁻¹ᵁ V).toScheme ≅ V.ι ⁻¹ᵁ U :=
  IsOpenImmersion.isoOfRangeEq (Opens.ι _ ≫ U.ι) (Opens.ι _ ≫ V.ι) <| by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X✝ : AlgebraicGeometry.Scheme
      U✝ : X✝.Opens
      X : AlgebraicGeometry.Scheme
      U V : X.Opens
      ⊢ Eq (Set.range ⇑(CategoryTheory.CategoryStruct.comp ((TopologicalSpace.Opens. …
    -/
    simp [Set.image_preimage_eq_inter_range, Set.inter_comm (U : Set X), Set.range_comp]
    /-
      🎉 no goals
    -/


/-- If `f : X ⟶ Y` is an open immersion, then for any `U : X.Opens`,
we have the isomorphism `U ≅ f ''ᵁ U`. -/
noncomputable
def Scheme.Hom.isoImage
    {X Y : Scheme.{u}} (f : X.Hom Y) [IsOpenImmersion f] (U : X.Opens) :
    U.toScheme ≅ f ''ᵁ U :=
                                                               /-
                                                                 C : Type u₁
                                                                 inst✝¹ : CategoryTheory.Category.{v, u₁} C
                                                                 X✝ : AlgebraicGeometry.Scheme
                                                                 U✝ : X✝.Opens
                                                                 X Y : AlgebraicGeometry.Scheme
                                                                 f : X.Hom Y
                                                                 inst✝ : AlgebraicGeometry.IsOpenImmersion f
                                                                 U : X.Opens
                                                                 ⊢ Eq (Set.range ⇑(CategoryTheory.CategoryStruct.comp U.ι f).base) (Set.range ⇑ …
                                                               -/
  IsOpenImmersion.isoOfRangeEq (Opens.ι _ ≫ f) (Opens.ι _) (by simp [Set.range_comp])
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[reassoc (attr := simp)]
lemma Scheme.Hom.isoImage_hom_ι
    {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] (U : X.Opens) :
    (f.isoImage U).hom ≫ (f ''ᵁ U).ι = U.ι ≫ f :=
  IsOpenImmersion.isoOfRangeEq_hom_fac _ _ _


@[reassoc (attr := simp)]
lemma Scheme.Hom.isoImage_inv_ι
    {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] (U : X.Opens) :
    (f.isoImage U).inv ≫ U.ι ≫ f = (f ''ᵁ U).ι :=
  IsOpenImmersion.isoOfRangeEq_inv_fac _ _ _


@[deprecated (since := "2024-10-20")]
alias Scheme.restrictRestrict := Scheme.Hom.isoImage

@[deprecated (since := "2024-10-20")]
alias Scheme.restrictRestrict_hom_restrict := Scheme.Hom.isoImage_hom_ι

@[deprecated (since := "2024-10-20")]
alias Scheme.restrictRestrict_inv_restrict_restrict := Scheme.Hom.isoImage_inv_ι

@[deprecated (since := "2024-10-20")]
alias Scheme.restrictRestrict_hom_restrict_assoc := Scheme.Hom.isoImage_hom_ι_assoc

@[deprecated (since := "2024-10-20")]
alias Scheme.restrictRestrict_inv_restrict_restrict_assoc := Scheme.Hom.isoImage_inv_ι_assoc


/-- `(⊤ : X.Opens)` as a scheme is isomorphic to `X`. -/
@[simps hom]
def Scheme.topIso (X : Scheme) : ↑(⊤ : X.Opens) ≅ X where
  hom := Scheme.Opens.ι _
  inv := ⟨X.restrictTopIso.inv⟩
  hom_inv_id := Hom.ext' X.restrictTopIso.hom_inv_id
  inv_hom_id := Hom.ext' X.restrictTopIso.inv_hom_id


@[reassoc (attr := simp)]
lemma Scheme.toIso_inv_ι (X : Scheme.{u}) : X.topIso.inv ≫ Opens.ι _ = 𝟙 _ :=
  X.topIso.inv_hom_id


@[reassoc (attr := simp)]
lemma Scheme.ι_toIso_inv (X : Scheme.{u}) : Opens.ι _ ≫ X.topIso.inv = 𝟙 _ :=
  X.topIso.hom_inv_id


/-- If `U = V`, then `X ∣_ U` is isomorphic to `X ∣_ V`. -/
noncomputable
def Scheme.isoOfEq (X : Scheme.{u}) {U V : X.Opens} (e : U = V) :
    (U : Scheme.{u}) ≅ V :=
                                           /-
                                             C : Type u₁
                                             inst✝ : CategoryTheory.Category.{v, u₁} C
                                             X✝ : AlgebraicGeometry.Scheme
                                             U✝ : X✝.Opens
                                             X : AlgebraicGeometry.Scheme
                                             U V : X.Opens
                                             e : Eq U V
                                             ⊢ Eq (Set.range ⇑U.ι.base) (Set.range ⇑V.ι.base)
                                           -/
  IsOpenImmersion.isoOfRangeEq U.ι V.ι (by rw [e])
                                           /-
                                             🎉 no goals
                                           -/


@[reassoc (attr := simp)]
lemma Scheme.isoOfEq_hom_ι (X : Scheme.{u}) {U V : X.Opens} (e : U = V) :
    (X.isoOfEq e).hom ≫ V.ι = U.ι :=
  IsOpenImmersion.isoOfRangeEq_hom_fac _ _ _


@[reassoc (attr := simp)]
lemma Scheme.isoOfEq_inv_ι (X : Scheme.{u}) {U V : X.Opens} (e : U = V) :
    (X.isoOfEq e).inv ≫ U.ι = V.ι :=
  IsOpenImmersion.isoOfRangeEq_inv_fac _ _ _


@[simp]
lemma Scheme.isoOfEq_rfl (X : Scheme.{u}) (U : X.Opens) : X.isoOfEq (refl U) = Iso.refl _ := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Eq (X.isoOfEq ⋯) (CategoryTheory.Iso.refl ↑U)
  -/
  ext1
  /-
    case w
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Eq (X.isoOfEq ⋯).hom (CategoryTheory.Iso.refl ↑U).hom
  -/
  rw [← cancel_mono U.ι, Scheme.isoOfEq_hom_ι, Iso.refl_hom, Category.id_comp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")] alias Scheme.restrictIsoOfEq := Scheme.isoOfEq


/-- The restriction of an isomorphism onto an open set. -/
noncomputable def Scheme.Hom.preimageIso {X Y : Scheme.{u}} (f : X.Hom Y) [IsIso (C := Scheme) f]
    (U : Y.Opens) : (f ⁻¹ᵁ U).toScheme ≅ U := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    inst✝ : CategoryTheory.IsIso f
    U : Y.Opens
    ⊢ CategoryTheory.Iso ↑((TopologicalSpace.Opens.map f.base).obj U) ↑U
  -/
  apply IsOpenImmersion.isoOfRangeEq (f := (f ⁻¹ᵁ U).ι ≫ f) U.ι _
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    inst✝ : CategoryTheory.IsIso f
    U : Y.Opens
    ⊢ Eq (Set.range ⇑(CategoryTheory.CategoryStruct.comp ((TopologicalSpace.Opens. …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    inst✝ : CategoryTheory.IsIso f
    U : Y.Opens
    ⊢ Eq (Set.range (Function.comp ⇑f.base ⇑((TopologicalSpace.Opens.map f.base).o …
  -/
  rw [Set.range_comp, Opens.range_ι, Opens.range_ι]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    inst✝ : CategoryTheory.IsIso f
    U : Y.Opens
    ⊢ Eq (Set.image ⇑f.base ↑((TopologicalSpace.Opens.map f.base).obj U)) ↑U
  -/
  refine @Set.image_preimage_eq _ _ f.base U.1 f.homeomorph.surjective
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma Scheme.Hom.preimageIso_hom_ι {X Y : Scheme.{u}} (f : X.Hom Y) [IsIso (C := Scheme) f]
    (U : Y.Opens) : (f.preimageIso U).hom ≫ U.ι = (f ⁻¹ᵁ U).ι ≫ f :=
  IsOpenImmersion.isoOfRangeEq_hom_fac _ _ _


@[reassoc (attr := simp)]
lemma Scheme.Hom.preimageIso_inv_ι {X Y : Scheme.{u}} (f : X.Hom Y) [IsIso (C := Scheme) f]
    (U : Y.Opens) : (f.preimageIso U).inv ≫ (f ⁻¹ᵁ U).ι ≫ f = U.ι :=
  IsOpenImmersion.isoOfRangeEq_inv_fac _ _ _


@[deprecated (since := "2024-10-20")] alias Scheme.restrictMapIso := Scheme.Hom.preimageIso


/-- Given a morphism `f : X ⟶ Y` and an open set `U ⊆ Y`, we have `X ×[Y] U ≅ X |_{f ⁻¹ U}` -/
def pullbackRestrictIsoRestrict {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) :
    pullback f (U.ι) ≅ f ⁻¹ᵁ U := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.pullback f U.ι) ↑((TopologicalSpac …
  -/
  refine IsOpenImmersion.isoOfRangeEq (pullback.fst f _) (Scheme.Opens.ι _) ?_
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.pullback.fst f U.ι).base) (Set.range ⇑ …
  -/
  simp [IsOpenImmersion.range_pullback_fst_of_right]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem pullbackRestrictIsoRestrict_inv_fst {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) :
    (pullbackRestrictIsoRestrict f U).inv ≫ pullback.fst f _ = (f ⁻¹ᵁ U).ι := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.pullbackRestrictIs …
  -/
  delta pullbackRestrictIsoRestrict; simp
                                     /-
                                       🎉 no goals
                                     -/


@[reassoc (attr := simp)]
theorem pullbackRestrictIsoRestrict_hom_ι {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) :
    (pullbackRestrictIsoRestrict f U).hom ≫ (f ⁻¹ᵁ U).ι = pullback.fst f _ := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.pullbackRestrictIs …
  -/
  delta pullbackRestrictIsoRestrict; simp
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-10-20")]
alias pullbackRestrictIsoRestrict_hom_restrict := pullbackRestrictIsoRestrict_hom_ι

@[deprecated (since := "2024-10-20")]
alias pullbackRestrictIsoRestrict_hom_restrict_assoc := pullbackRestrictIsoRestrict_hom_ι_assoc


/-- The restriction of a morphism `X ⟶ Y` onto `X |_{f ⁻¹ U} ⟶ Y |_ U`. -/
def morphismRestrict {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) : (f ⁻¹ᵁ U).toScheme ⟶ U :=
  (pullbackRestrictIsoRestrict f U).inv ≫ pullback.snd _ _


/-- the notation for restricting a morphism of scheme to an open subset of the target scheme -/
infixl:85 " ∣_ " => morphismRestrict


@[reassoc (attr := simp)]
theorem pullbackRestrictIsoRestrict_hom_morphismRestrict {X Y : Scheme.{u}} (f : X ⟶ Y)
    (U : Y.Opens) : (pullbackRestrictIsoRestrict f U).hom ≫ f ∣_ U = pullback.snd _ _ :=
  Iso.hom_inv_id_assoc _ _


@[reassoc (attr := simp)]
theorem morphismRestrict_ι {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) :
    (f ∣_ U) ≫ U.ι = (f ⁻¹ᵁ U).ι ≫ f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict f …
  -/
  delta morphismRestrict
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, pullback.condition.symm, pullbackRestrictIsoRestrict_inv_fst_assoc]
  /-
    🎉 no goals
  -/


theorem isPullback_morphismRestrict {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) :
    IsPullback (f ∣_ U) (f ⁻¹ᵁ U).ι U.ι f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ CategoryTheory.IsPullback (AlgebraicGeometry.morphismRestrict f U) ((Topolog …
  -/
  delta morphismRestrict
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp (AlgebraicGeom …
  -/
  rw [← Category.id_comp f]
  refine
    (IsPullback.of_horiz_isIso ⟨?_⟩).paste_horiz
      (IsPullback.of_hasPullback f (Y.ofRestrict U.isOpenEmbedding)).flip
  -- Porting note: changed `rw` to `erw`
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.pullbackRestrictIs …
  -/
  erw [pullbackRestrictIsoRestrict_inv_fst]; rw [Category.comp_id]
                                             /-
                                               🎉 no goals
                                             -/


lemma isPullback_opens_inf_le {X : Scheme} {U V W : X.Opens} (hU : U ≤ W) (hV : V ≤ W) :
    IsPullback (X.homOfLE inf_le_left) (X.homOfLE inf_le_right) (X.homOfLE hU) (X.homOfLE hV) := by
  refine (isPullback_morphismRestrict (X.homOfLE hV) (W.ι ⁻¹ᵁ U)).of_iso (V.ι.isoImage _ ≪≫
    X.isoOfEq ?_) (W.ι.isoImage _ ≪≫ X.isoOfEq ?_) (Iso.refl _) (Iso.refl _) ?_ ?_ ?_ ?_
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      U V W : X.Opens
      hU : LE.le U W
      hV : LE.le V W
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor V.ι).obj ((TopologicalSpace.O …
    -/
  · rw [← TopologicalSpace.Opens.map_comp_obj, ← Scheme.comp_base, Scheme.homOfLE_ι]
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      U V W : X.Opens
      hU : LE.le U W
      hV : LE.le V W
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor V.ι).obj ((TopologicalSpace.O …
    -/
    exact V.functor_map_eq_inf U
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : AlgebraicGeometry.Scheme
      U V W : X.Opens
      hU : LE.le U W
      hV : LE.le V W
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor W.ι).obj ((TopologicalSpace.O …
    -/
  · exact (W.functor_map_eq_inf U).trans (by simpa)
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    X : AlgebraicGeometry.Scheme
    U V W : X.Opens
    hU : LE.le U W
    hV : LE.le V W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict ( …
  -/
  all_goals { simp [← cancel_mono (Scheme.Opens.ι _)] }
  /-
    🎉 no goals
  -/


lemma isPullback_opens_inf {X : Scheme} (U V : X.Opens) :
    IsPullback (X.homOfLE inf_le_left) (X.homOfLE inf_le_right) U.ι V.ι :=
  (isPullback_morphismRestrict V.ι U).of_iso (V.ι.isoImage _ ≪≫ X.isoOfEq
                                                                         /-
                                                                           X : AlgebraicGeometry.Scheme
                                                                           U V : X.Opens
                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict V …
                                                                         -/
    (V.functor_map_eq_inf U)) (Iso.refl _) (Iso.refl _) (Iso.refl _) (by simp [← cancel_mono U.ι])
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
        /-
          X : AlgebraicGeometry.Scheme
          U V : X.Opens
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopologicalSpace.Opens.map V.ι.base …
        -/
        /-
          🎉 no goals
        -/
                                      /-
                                        🎉 no goals
                                      -/
    (by simp [← cancel_mono V.ι]) (by simp) (by simp)
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
lemma morphismRestrict_id {X : Scheme.{u}} (U : X.Opens) : 𝟙 X ∣_ U = 𝟙 _ := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Eq (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.id X)  …
  -/
  rw [← cancel_mono U.ι, morphismRestrict_ι, Category.comp_id, Category.id_comp]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Eq ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStruct.id X).base).o …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem morphismRestrict_comp {X Y Z : Scheme.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) (U : Opens Z) :
    (f ≫ g) ∣_ U = f ∣_ g ⁻¹ᵁ U ≫ g ∣_ U := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : TopologicalSpace.Opens ↑↑Z.toPresheafedSpace
    ⊢ Eq (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.comp f …
  -/
  delta morphismRestrict
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : TopologicalSpace.Opens ↑↑Z.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.pullbackRestrictIs …
  -/
  rw [← pullbackRightPullbackFstIso_inv_snd_snd]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : TopologicalSpace.Opens ↑↑Z.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.pullbackRestrictIs …
  -/
  simp_rw [← Category.assoc]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : TopologicalSpace.Opens ↑↑Z.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  /-
    case e_a
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : TopologicalSpace.Opens ↑↑Z.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [← cancel_mono (pullback.fst _ _)]
  /-
    case e_a
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : TopologicalSpace.Opens ↑↑Z.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp_rw [Category.assoc]
  rw [pullbackRestrictIsoRestrict_inv_fst, pullbackRightPullbackFstIso_inv_snd_fst, ←
    pullback.condition, pullbackRestrictIsoRestrict_inv_fst_assoc,
    pullbackRestrictIsoRestrict_inv_fst_assoc]
  /-
    case e_a
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    U : TopologicalSpace.Opens ↑↑Z.toPresheafedSpace
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopologicalSpace.Opens.map (Categor …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance {X Y : Scheme.{u}} (f : X ⟶ Y) [IsIso f] (U : Y.Opens) : IsIso (f ∣_ U) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    U : Y.Opens
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.morphismRestrict f U)
  -/
  delta morphismRestrict; infer_instance
                          /-
                            🎉 no goals
                          -/


theorem morphismRestrict_base_coe {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) (x) :
    @Coe.coe U Y (⟨fun x => x.1⟩) ((f ∣_ U).base x) = f.base x.1 :=
  congr_arg (fun f => (Scheme.Hom.toLRSHom f).base x)
    (morphismRestrict_ι f U)


theorem morphismRestrict_base {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) :
    ⇑(f ∣_ U).base = U.1.restrictPreimage f.base :=
  funext fun x => Subtype.ext (morphismRestrict_base_coe f U x)


theorem image_morphismRestrict_preimage {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) (V : Opens U) :
    (f ⁻¹ᵁ U).ι ''ᵁ ((f ∣_ U) ⁻¹ᵁ V) = f ⁻¹ᵁ (U.ι ''ᵁ V) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor ((TopologicalSpace.Opens.map  …
  -/
  ext1
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
    ⊢ Eq ↑((AlgebraicGeometry.Scheme.Hom.opensFunctor ((TopologicalSpace.Opens.map …
  -/
  ext x
  /-
    case h.h
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
    x : ↑↑X.toPresheafedSpace
    ⊢ Iff (Membership.mem (↑((AlgebraicGeometry.Scheme.Hom.opensFunctor ((Topologi …
  -/
  constructor
    /-
      case h.h.mp
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (↑((AlgebraicGeometry.Scheme.Hom.opensFunctor ((TopologicalSp …
    -/
  · rintro ⟨⟨x, hx⟩, hx' : (f ∣_ U).base _ ∈ V, rfl⟩
    /-
      case h.h.mp.intro.mk.intro
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx' : Membership.mem V ((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩)
      ⊢ Membership.mem (↑((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometr …
    -/
    refine ⟨⟨_, hx⟩, ?_, rfl⟩
    -- Porting note: this rewrite was not necessary
    /-
      case h.h.mp.intro.mk.intro
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx' : Membership.mem V ((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩)
      ⊢ Membership.mem ↑V ⟨f.base x, hx⟩
    -/
    rw [SetLike.mem_coe]
    /-
      case h.h.mp.intro.mk.intro
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx' : Membership.mem V ((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩)
      ⊢ Membership.mem V ⟨f.base x, hx⟩
    -/
    convert hx'
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext1` is not compiling
    /-
      case h.e'_5
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx' : Membership.mem V ((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩)
      ⊢ Eq ⟨f.base x, hx⟩ ((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩)
    -/
    refine Subtype.ext ?_
    /-
      case h.e'_5
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) x
      hx' : Membership.mem V ((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩)
      ⊢ Eq ↑⟨f.base x, hx⟩ ↑((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩)
    -/
    exact (morphismRestrict_base_coe f U ⟨x, hx⟩).symm
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (↑((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometr …
    -/
  · rintro ⟨⟨x, hx⟩, hx' : _ ∈ V.1, rfl : x = _⟩
    /-
      case h.h.mpr.intro.mk.intro
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U (f.base x)
      hx' : Membership.mem V.carrier ⟨f.base x, hx⟩
      ⊢ Membership.mem (↑((AlgebraicGeometry.Scheme.Hom.opensFunctor ((TopologicalSp …
    -/
    refine ⟨⟨_, hx⟩, (?_ : (f ∣_ U).base ⟨x, hx⟩ ∈ V.1), rfl⟩
    /-
      case h.h.mpr.intro.mk.intro
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U (f.base x)
      hx' : Membership.mem V.carrier ⟨f.base x, hx⟩
      ⊢ Membership.mem V.carrier ((AlgebraicGeometry.morphismRestrict f U).base ⟨x,  …
    -/
    convert hx'
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext1` is compiling
    /-
      case h.e'_5
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U (f.base x)
      hx' : Membership.mem V.carrier ⟨f.base x, hx⟩
      ⊢ Eq ((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩) ⟨f.base x, hx⟩
    -/
    refine Subtype.ext ?_
    /-
      case h.e'_5
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U (f.base x)
      hx' : Membership.mem V.carrier ⟨f.base x, hx⟩
      ⊢ Eq ↑((AlgebraicGeometry.morphismRestrict f U).base ⟨x, hx⟩) ↑⟨f.base x, hx⟩
    -/
    exact morphismRestrict_base_coe f U ⟨x, hx⟩
    /-
      🎉 no goals
    -/


lemma eqToHom_eq_homOfLE {C} [Preorder C] {X Y : C} (e : X = Y) : eqToHom e = homOfLE e.le := rfl


open Scheme in
theorem morphismRestrict_app {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) (V : U.toScheme.Opens) :
    (f ∣_ U).app V = f.app (U.ι ''ᵁ V) ≫
        X.presheaf.map (eqToHom (image_morphismRestrict_preimage f U V)).op := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : (↑U).Opens
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.morphismRestrict f U …
  -/
  have := Scheme.congr_app (morphismRestrict_ι f U) (U.ι ''ᵁ V)
  simp only [Scheme.preimage_comp, Opens.toScheme_presheaf_obj, Hom.app_eq_appLE, comp_appLE,
    Opens.ι_appLE, eqToHom_op, Opens.toScheme_presheaf_map, eqToHom_unop] at this
  have e : U.ι ⁻¹ᵁ (U.ι ''ᵁ V) = V :=
    Opens.ext (Set.preimage_image_eq _ Subtype.coe_injective)
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : (↑U).Opens
    this : Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory. …
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj ((AlgebraicGeometry.Scheme.H …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.morphismRestrict f U …
  -/
  have e' : (f ∣_ U) ⁻¹ᵁ V = (f ∣_ U) ⁻¹ᵁ U.ι ⁻¹ᵁ U.ι ''ᵁ V := by rw [e]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : (↑U).Opens
    this : Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory. …
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj ((AlgebraicGeometry.Scheme.H …
    e' : Eq ((TopologicalSpace.Opens.map (AlgebraicGeometry.morphismRestrict f U). …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.morphismRestrict f U …
  -/
  simp only [Opens.toScheme_presheaf_obj, Hom.app_eq_appLE, eqToHom_op, Hom.appLE_map]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : (↑U).Opens
    this : Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory. …
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj ((AlgebraicGeometry.Scheme.H …
    e' : Eq ((TopologicalSpace.Opens.map (AlgebraicGeometry.morphismRestrict f U). …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE (AlgebraicGeometry.morphismRestrict f …
  -/
  rw [← (f ∣_ U).appLE_map' _ e', ← (f ∣_ U).map_appLE' _ e]
  simp only [Opens.toScheme_presheaf_obj, eqToHom_eq_homOfLE, Opens.toScheme_presheaf_map,
    Quiver.Hom.unop_op, Hom.opensFunctor_map_homOfLE]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : (↑U).Opens
    this : Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory. …
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj ((AlgebraicGeometry.Scheme.H …
    e' : Eq ((TopologicalSpace.Opens.map (AlgebraicGeometry.morphismRestrict f U). …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [this, Hom.appLE_map, Hom.appLE_map, Hom.appLE_map]
  /-
    🎉 no goals
  -/


theorem morphismRestrict_appTop {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) :
    (f ∣_ U).appTop = f.app (U.ι ''ᵁ ⊤) ≫
        X.presheaf.map (eqToHom (image_morphismRestrict_preimage f U ⊤)).op :=
  morphismRestrict_app ..


@[simp]
theorem morphismRestrict_app' {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) (V : Opens U) :
    (f ∣_ U).app V = f.appLE _ _ (image_morphismRestrict_preimage f U V).le :=
  morphismRestrict_app f U V


@[simp]
theorem morphismRestrict_appLE {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) (V W e) :
    (f ∣_ U).appLE V W e = f.appLE (U.ι ''ᵁ V) ((f ⁻¹ᵁ U).ι ''ᵁ W)
      ((Set.image_subset _ e).trans (image_morphismRestrict_preimage f U V).le) := by
  rw [Scheme.Hom.appLE, morphismRestrict_app', Scheme.Opens.toScheme_presheaf_map,
    Scheme.Hom.appLE_map]


theorem Γ_map_morphismRestrict {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) :
    Scheme.Γ.map (f ∣_ U).op =
      Y.presheaf.map (eqToHom U.isOpenEmbedding_obj_top.symm).op ≫
        f.app U ≫ X.presheaf.map (eqToHom (f ⁻¹ᵁ U).isOpenEmbedding_obj_top).op := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (AlgebraicGeometry.Scheme.Γ.map (AlgebraicGeometry.morphismRestrict f U). …
  -/
  rw [Scheme.Γ_map_op, morphismRestrict_appTop f U, f.naturality_assoc, ← X.presheaf.map_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app f ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Restricting a morphism onto the image of an open immersion is isomorphic to the base change
along the immersion. -/
def morphismRestrictOpensRange
    {X Y U : Scheme.{u}} (f : X ⟶ Y) (g : U ⟶ Y) [hg : IsOpenImmersion g] :
    Arrow.mk (f ∣_ g.opensRange) ≅ Arrow.mk (pullback.snd f g) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom U Y
    hg : AlgebraicGeometry.IsOpenImmersion g
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (AlgebraicGeometry.morphismRestr …
  -/
  let V : Y.Opens := g.opensRange
  let e :=
    IsOpenImmersion.isoOfRangeEq g V.ι Subtype.range_coe.symm
  let t : pullback f g ⟶ pullback f V.ι :=
    pullback.map _ _ _ _ (𝟙 _) e.hom (𝟙 _) (by rw [Category.comp_id, Category.id_comp])
      (by rw [Category.comp_id, IsOpenImmersion.isoOfRangeEq_hom_fac])
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom U Y
    hg : AlgebraicGeometry.IsOpenImmersion g
    V : Y.Opens := AlgebraicGeometry.Scheme.Hom.opensRange g
    e : CategoryTheory.Iso U ↑V := AlgebraicGeometry.IsOpenImmersion.isoOfRangeEq  …
    t : Quiver.Hom (CategoryTheory.Limits.pullback f g) (CategoryTheory.Limits.pul …
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (AlgebraicGeometry.morphismRestr …
  -/
  symm
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom U Y
    hg : AlgebraicGeometry.IsOpenImmersion g
    V : Y.Opens := AlgebraicGeometry.Scheme.Hom.opensRange g
    e : CategoryTheory.Iso U ↑V := AlgebraicGeometry.IsOpenImmersion.isoOfRangeEq  …
    t : Quiver.Hom (CategoryTheory.Limits.pullback f g) (CategoryTheory.Limits.pul …
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (CategoryTheory.Limits.pullback. …
  -/
  refine Arrow.isoMk (asIso t ≪≫ pullbackRestrictIsoRestrict f V) e ?_
  rw [Iso.trans_hom, asIso_hom, ← Iso.comp_inv_eq, ← cancel_mono g, Arrow.mk_hom, Arrow.mk_hom,
    Category.assoc, Category.assoc, Category.assoc, IsOpenImmersion.isoOfRangeEq_inv_fac,
    ← pullback.condition, morphismRestrict_ι,
    pullbackRestrictIsoRestrict_hom_ι_assoc, pullback.lift_fst_assoc, Category.comp_id]


/-- The restrictions onto two equal open sets are isomorphic. This currently has bad defeqs when
unfolded, but it should not matter for now. Replace this definition if better defeqs are needed. -/
def morphismRestrictEq {X Y : Scheme.{u}} (f : X ⟶ Y) {U V : Y.Opens} (e : U = V) :
    Arrow.mk (f ∣_ U) ≅ Arrow.mk (f ∣_ V) :=
              /-
                C : Type u₁
                inst✝ : CategoryTheory.Category.{v, u₁} C
                X Y : AlgebraicGeometry.Scheme
                f : Quiver.Hom X Y
                U V : Y.Opens
                e : Eq U V
                ⊢ Eq (CategoryTheory.Arrow.mk (AlgebraicGeometry.morphismRestrict f U)) (Categ …
              -/
  eqToIso (by subst e; rfl)
                       /-
                         🎉 no goals
                       -/


/-- Restricting a morphism twice is isomorphic to one restriction. -/
def morphismRestrictRestrict {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) (V : U.toScheme.Opens) :
    Arrow.mk (f ∣_ U ∣_ V) ≅ Arrow.mk (f ∣_ U.ι ''ᵁ V) := by
  refine Arrow.isoMk' _ _ ((Scheme.Opens.ι _).isoImage _ ≪≫ Scheme.isoOfEq _ ?_)
    ((Scheme.Opens.ι _).isoImage _) ?_
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : (↑U).Opens
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor ((TopologicalSpace.Opens.map  …
    -/
  · ext x
    simp only [IsOpenMap.coe_functor_obj, Opens.coe_inclusion',
      Opens.map_coe, Set.mem_image, Set.mem_preimage, SetLike.mem_coe, morphismRestrict_base]
    /-
      case refine_1.h.h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : (↑U).Opens
      x : ↑↑X.toPresheafedSpace
      ⊢ Iff (Exists fun x_1 => And (Membership.mem V (U.carrier.restrictPreimage (⇑f …
    -/
    constructor
      /-
        case refine_1.h.h.mp
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v, u₁} C
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        U : Y.Opens
        V : (↑U).Opens
        x : ↑↑X.toPresheafedSpace
        ⊢ (Exists fun x_1 => And (Membership.mem V (U.carrier.restrictPreimage (⇑f.bas …
      -/
    · rintro ⟨⟨a, h₁⟩, h₂, rfl⟩
      /-
        case refine_1.h.h.mp.intro.mk.intro
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v, u₁} C
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        U : Y.Opens
        V : (↑U).Opens
        a : ↑↑X.toPresheafedSpace
        h₁ : Membership.mem ((TopologicalSpace.Opens.map f.base).obj U) a
        h₂ : Membership.mem V (U.carrier.restrictPreimage ⇑f.base ⟨a, h₁⟩)
        ⊢ Exists fun x => And (Membership.mem V x) (Eq ((AlgebraicGeometry.Scheme.Hom. …
      -/
      exact ⟨_, h₂, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.h.h.mpr
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v, u₁} C
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        U : Y.Opens
        V : (↑U).Opens
        x : ↑↑X.toPresheafedSpace
        ⊢ (Exists fun x_1 => And (Membership.mem V x_1) (Eq ((AlgebraicGeometry.Scheme …
      -/
    · rintro ⟨⟨a, h₁⟩, h₂, rfl : a = _⟩
      /-
        case refine_1.h.h.mpr.intro.mk.intro
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v, u₁} C
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        U : Y.Opens
        V : (↑U).Opens
        x : ↑↑X.toPresheafedSpace
        h₁ : Membership.mem U (f.base x)
        h₂ : Membership.mem V ⟨f.base x, h₁⟩
        ⊢ Exists fun x_1 => And (Membership.mem V (U.carrier.restrictPreimage (⇑f.base …
      -/
      exact ⟨⟨x, h₁⟩, h₂, rfl⟩
      /-
        🎉 no goals
      -/
  · rw [← cancel_mono (Scheme.Opens.ι _), Iso.trans_hom, Category.assoc, Category.assoc,
      Category.assoc, morphismRestrict_ι, Scheme.isoOfEq_hom_ι_assoc,
      Scheme.Hom.isoImage_hom_ι_assoc,
      Scheme.Hom.isoImage_hom_ι,
      morphismRestrict_ι_assoc, morphismRestrict_ι]


/-- Restricting a morphism twice onto a basic open set is isomorphic to one restriction. -/
def morphismRestrictRestrictBasicOpen {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) (r : Γ(Y, U)) :
    Arrow.mk (f ∣_ U ∣_
          U.toScheme.basicOpen (Y.presheaf.map (eqToHom U.isOpenEmbedding_obj_top).op r)) ≅
      Arrow.mk (f ∣_ Y.basicOpen r) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (AlgebraicGeometry.morphismRestr …
  -/
  refine morphismRestrictRestrict _ _ _ ≪≫ morphismRestrictEq _ ?_
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen ((Y. …
  -/
  have e := Scheme.preimage_basicOpen U.ι r
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen ((Y. …
  -/
  rw [Scheme.Opens.ι_app] at e
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen ((Y. …
  -/
  rw [← U.toScheme.basicOpen_res_eq _ (eqToHom U.inclusion'_map_eq_top).op]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen (((↑ …
  -/
  erw [← CommRingCat.comp_apply]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen ((Ca …
  -/
  erw [← Y.presheaf.map_comp]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen ((Y. …
  -/
  rw [eqToHom_op, eqToHom_op, eqToHom_map, eqToHom_trans]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen ((Y. …
  -/
  erw [← e]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((TopologicalSpace.O …
  -/
  ext1
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq ↑((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((TopologicalSpace. …
  -/
  dsimp [Opens.map_coe]
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ Eq (Set.image (⇑(AlgebraicGeometry.Scheme.Hom.toLRSHom U.ι).base) (Set.preim …
  -/
  rw [Set.image_preimage_eq_inter_range, Set.inter_eq_left, Scheme.Opens.range_ι]
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    r : ↑(Y.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map U.ι.base).obj (Y.basicOpen r)) ((↑U).basic …
    ⊢ HasSubset.Subset ↑(Y.basicOpen r) ↑U
  -/
  exact Y.basicOpen_le r
  /-
    🎉 no goals
  -/


/-- The stalk map of a restriction of a morphism is isomorphic to the stalk map of the original map.
-/
def morphismRestrictStalkMap {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) (x) :
    Arrow.mk ((f ∣_ U).stalkMap x) ≅ Arrow.mk (f.stalkMap x.1) := Arrow.isoMk' _ _
  (U.stalkIso ((f ∣_ U).base x) ≪≫
    (TopCat.Presheaf.stalkCongr _ <| Inseparable.of_eq <| morphismRestrict_base_coe f U x))
  ((f ⁻¹ᵁ U).stalkIso x) <| by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      x : ↑↑(↑((TopologicalSpace.Opens.map f.base).obj U)).toPresheafedSpace
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((U.stalkIso ((AlgebraicGeometry.morp …
    -/
    apply TopCat.Presheaf.stalk_hom_ext
    /-
      case ih
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      x : ↑↑(↑((TopologicalSpace.Opens.map f.base).obj U)).toPresheafedSpace
      ⊢ ∀ (U_1 : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace) (hxU : Membership. …
    -/
    intro V hxV
    /-
      case ih
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      x : ↑↑(↑((TopologicalSpace.Opens.map f.base).obj U)).toPresheafedSpace
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      hxV : Membership.mem V ((AlgebraicGeometry.morphismRestrict f U).base x)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((↑U).presheaf.germ V ((AlgebraicGeom …
    -/
    change ↑(f ⁻¹ᵁ U) at x
    /-
      case ih
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      V : TopologicalSpace.Opens ↑↑(↑U).toPresheafedSpace
      x : ↑↑(↑((TopologicalSpace.Opens.map f.base).obj U)).toPresheafedSpace
      hxV : Membership.mem V ((AlgebraicGeometry.morphismRestrict f U).base x)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((↑U).presheaf.germ V ((AlgebraicGeom …
    -/
    simp [Scheme.stalkMap_germ_assoc, Scheme.Hom.appLE]
    /-
      🎉 no goals
    -/


instance {X Y : Scheme.{u}} (f : X ⟶ Y) (U : Y.Opens) [IsOpenImmersion f] :
    IsOpenImmersion (f ∣_ U) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.morphismRestrict f U)
  -/
  delta morphismRestrict
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp (Algeb …
  -/
  exact PresheafedSpace.IsOpenImmersion.comp _ _
  /-
    🎉 no goals
  -/


/-- The restriction of a morphism `f : X ⟶ Y` to open sets on the source and target. -/
def resLE (f : Hom X Y) (U : Y.Opens) (V : X.Opens) (e : V ≤ f ⁻¹ᵁ U) : V.toScheme ⟶ U.toScheme :=
  X.homOfLE e ≫ f ∣_ U


lemma resLE_eq_morphismRestrict : f.resLE U (f ⁻¹ᵁ U) le_rfl = f ∣_ U := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.resLE f U ((TopologicalSpace.Opens.map f.ba …
  -/
  simp [resLE]
  /-
    🎉 no goals
  -/


lemma resLE_id (i : V ≤ V') : resLE (𝟙 X) V' V i = X.homOfLE i := by
  /-
    X : AlgebraicGeometry.Scheme
    V V' : X.Opens
    i : LE.le V V'
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.resLE (CategoryTheory.CategoryStruct.id X)  …
  -/
  simp only [resLE, morphismRestrict_id]
  /-
    X : AlgebraicGeometry.Scheme
    V V' : X.Opens
    i : LE.le V V'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE i) (CategoryTheory.Categor …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma resLE_comp_ι : f.resLE U V e ≫ U.ι = V.ι ≫ f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.resLE f …
  -/
  simp [resLE]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma resLE_comp_resLE {Z : Scheme.{u}} (g : Y ⟶ Z) {W : Z.Opens} (e') :
    f.resLE U V e ≫ g.resLE W U e' = (f ≫ g).resLE W V
      (e.trans ((Opens.map f.base).map (homOfLE e')).le) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    Z : AlgebraicGeometry.Scheme
    g : Quiver.Hom Y Z
    W : Z.Opens
    e' : LE.le U ((TopologicalSpace.Opens.map g.base).obj W)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.resLE f …
  -/
  simp [← cancel_mono W.ι]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma map_resLE (i : V' ≤ V) :
    X.homOfLE i ≫ f.resLE U V e = f.resLE U V' (i.trans e) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V V' : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    i : LE.le V' V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.homOfLE i) (AlgebraicGeometry.Sche …
  -/
  simp_rw [← resLE_id, resLE_comp_resLE, Category.id_comp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma resLE_map (i : U ≤ U') :
    f.resLE U V e ≫ Y.homOfLE i =
      f.resLE U' V (e.trans ((Opens.map f.base).map i.hom).le) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U U' : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    i : LE.le U U'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.resLE f …
  -/
  simp_rw [← resLE_id, resLE_comp_resLE, Category.comp_id]
  /-
    🎉 no goals
  -/


lemma resLE_congr (e₁ : U = U') (e₂ : V = V') (P : MorphismProperty Scheme.{u}) :
    P (f.resLE U V e) ↔ P (f.resLE U' V' (e₁ ▸ e₂ ▸ e)) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U U' : Y.Opens
    V V' : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    e₁ : Eq U U'
    e₂ : Eq V V'
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    ⊢ Iff (P (AlgebraicGeometry.Scheme.Hom.resLE f U V e)) (P (AlgebraicGeometry.S …
  -/
  subst e₁; subst e₂; rfl
                      /-
                        🎉 no goals
                      -/


lemma resLE_preimage (f : X ⟶ Y) {U : Y.Opens} {V : X.Opens} (e : V ≤ f ⁻¹ᵁ U)
    (O : U.toScheme.Opens) :
    f.resLE U V e ⁻¹ᵁ O = V.ι ⁻¹ᵁ (f ⁻¹ᵁ U.ι ''ᵁ O) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    O : (↑U).Opens
    ⊢ Eq ((TopologicalSpace.Opens.map (AlgebraicGeometry.Scheme.Hom.resLE f U V e) …
  -/
  rw [← preimage_comp, ← resLE_comp_ι f e, preimage_comp, preimage_image_eq]
  /-
    🎉 no goals
  -/


lemma le_preimage_resLE_iff {U : Y.Opens} {V : X.Opens} (e : V ≤ f ⁻¹ᵁ U)
    (O : U.toScheme.Opens) (W : V.toScheme.Opens) :
    W ≤ (f.resLE U V e) ⁻¹ᵁ O ↔ V.ι ''ᵁ W ≤ f ⁻¹ᵁ U.ι ''ᵁ O := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    O : (↑U).Opens
    W : (↑V).Opens
    ⊢ Iff (LE.le W ((TopologicalSpace.Opens.map (AlgebraicGeometry.Scheme.Hom.resL …
  -/
  simp [resLE_preimage, ← image_le_image_iff V.ι, image_preimage_eq_opensRange_inter, V.ι_image_le]
  /-
    🎉 no goals
  -/


lemma resLE_appLE {U : Y.Opens} {V : X.Opens} (e : V ≤ f ⁻¹ᵁ U)
    (O : U.toScheme.Opens) (W : V.toScheme.Opens) (e' : W ≤ resLE f U V e ⁻¹ᵁ O) :
    (f.resLE U V e).appLE O W e' =
      f.appLE (U.ι ''ᵁ O) (V.ι ''ᵁ W) ((le_preimage_resLE_iff f e O W).mp e') := by
  simp only [appLE, resLE, comp_coeBase, Opens.map_comp_obj, comp_app, morphismRestrict_app',
    homOfLE_leOfHom, homOfLE_app, Category.assoc, Opens.toScheme_presheaf_map, Quiver.Hom.unop_op,
    opensFunctor_map_homOfLE]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    O : (↑U).Opens
    W : (↑V).Opens
    e' : LE.le W ((TopologicalSpace.Opens.map (AlgebraicGeometry.Scheme.Hom.resLE  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app f ( …
  -/
  rw [← X.presheaf.map_comp, ← X.presheaf.map_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    O : (↑U).Opens
    W : (↑V).Opens
    e' : LE.le W ((TopologicalSpace.Opens.map (AlgebraicGeometry.Scheme.Hom.resLE  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app f ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `f.resLE U V` induces `f.appLE U V` on global sections. -/
noncomputable def arrowResLEAppIso (f : X ⟶ Y) (U : Y.Opens) (V : X.Opens) (e : V ≤ f ⁻¹ᵁ U) :
    Arrow.mk ((f.resLE U V e).appTop) ≅ Arrow.mk (f.appLE U V e) :=
  Arrow.isoMk U.topIso V.topIso <| by
  simp only [Opens.map_top, Arrow.mk_left, Arrow.mk_right, Functor.id_obj, Scheme.Opens.topIso_hom,
    eqToHom_op, Arrow.mk_hom, Scheme.Hom.map_appLE]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} C
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE f ((AlgebraicGeometry.Scheme.Hom.open …
  -/
  rw [Scheme.Hom.appTop, ← Scheme.Hom.appLE_eq_app, Scheme.Hom.resLE_appLE, Scheme.Hom.appLE_map]
  /-
    🎉 no goals
  -/


/-- The restriction of an open cover to an open subset. -/
@[simps! J obj map]
noncomputable
def Scheme.OpenCover.restrict {X : Scheme.{u}} (𝒰 : X.OpenCover) (U : Opens X) :
    U.toScheme.OpenCover := by
  refine Cover.copy (𝒰.pullbackCover U.ι) 𝒰.J _ (𝒰.map · ∣_ U) (Equiv.refl _)
    (fun i ↦ IsOpenImmersion.isoOfRangeEq (Opens.ι _) (pullback.snd _ _) ?_) ?_
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : X.Opens
      i : 𝒰.J
      ⊢ Eq (Set.range ⇑((TopologicalSpace.Opens.map (𝒰.map i).base).obj U).ι.base) ( …
    -/
  · erw [IsOpenImmersion.range_pullback_snd_of_left U.ι (𝒰.map i)]
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : X.Opens
      i : 𝒰.J
      ⊢ Eq (Set.range ⇑((TopologicalSpace.Opens.map (𝒰.map i).base).obj U).ι.base) ( …
    -/
    rw [Opens.opensRange_ι]
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : X.Opens
      i : 𝒰.J
      ⊢ Eq (Set.range ⇑((TopologicalSpace.Opens.map (𝒰.map i).base).obj U).ι.base) ( …
    -/
    exact Subtype.range_val
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : X.Opens
      ⊢ ∀ (i : 𝒰.J), Eq ((fun x => AlgebraicGeometry.morphismRestrict (𝒰.map x) U) i …
    -/
  · intro i
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : X.Opens
      i : 𝒰.J
      ⊢ Eq ((fun x => AlgebraicGeometry.morphismRestrict (𝒰.map x) U) i) (CategoryTh …
    -/
    rw [← cancel_mono U.ι]
    simp only [morphismRestrict_ι, Cover.pullbackCover_J, Equiv.refl_apply, Cover.pullbackCover_obj,
      Cover.pullbackCover_map, Category.assoc, pullback.condition]
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} C
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : X.Opens
      i : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopologicalSpace.Opens.map (𝒰.map i …
    -/
    rw [IsOpenImmersion.isoOfRangeEq_hom_fac_assoc]
    /-
      🎉 no goals
    -/


