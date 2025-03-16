/-- A morphism of Schemes is an open immersion if it is an open immersion as a morphism
of LocallyRingedSpaces
-/
abbrev IsOpenImmersion {X Y : Scheme.{u}} (f : X ⟶ Y) : Prop :=
  LocallyRingedSpace.IsOpenImmersion f.toLRSHom


instance IsOpenImmersion.comp {X Y Z : Scheme.{u}} (f : X ⟶ Y) (g : Y ⟶ Z)
  [IsOpenImmersion f] [IsOpenImmersion g] : IsOpenImmersion (f ≫ g) :=
LocallyRingedSpace.IsOpenImmersion.comp f.toLRSHom g.toLRSHom


/-- To show that a locally ringed space is a scheme, it suffices to show that it has a jointly
surjective family of open immersions from affine schemes. -/
protected def scheme (X : LocallyRingedSpace.{u})
    (h :
      ∀ x : X,
        ∃ (R : CommRingCat) (f : Spec.toLocallyRingedSpace.obj (op R) ⟶ X),
          (x ∈ Set.range f.base : _) ∧ LocallyRingedSpace.IsOpenImmersion f) :
    Scheme where
  toLocallyRingedSpace := X
  local_affine := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.LocallyRingedSpace
      h : ∀ (x : ↑X.toTopCat), Exists fun R => Exists fun f => And (Membership.mem ( …
      ⊢ ∀ (x : ↑X.toTopCat), Exists fun U => Exists fun R => Nonempty (CategoryTheor …
    -/
    intro x
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.LocallyRingedSpace
      h : ∀ (x : ↑X.toTopCat), Exists fun R => Exists fun f => And (Membership.mem ( …
      x : ↑X.toTopCat
      ⊢ Exists fun U => Exists fun R => Nonempty (CategoryTheory.Iso (X.restrict ⋯)  …
    -/
    obtain ⟨R, f, h₁, h₂⟩ := h x
    /-
      case intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.LocallyRingedSpace
      h : ∀ (x : ↑X.toTopCat), Exists fun R => Exists fun f => And (Membership.mem ( …
      x : ↑X.toTopCat
      R : CommRingCat
      f : Quiver.Hom (AlgebraicGeometry.Spec.toLocallyRingedSpace.obj { unop := R }) X
      h₁ : Membership.mem (Set.range ⇑f.base) x
      h₂ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ Exists fun U => Exists fun R => Nonempty (CategoryTheory.Iso (X.restrict ⋯)  …
    -/
    refine ⟨⟨⟨_, h₂.base_open.isOpen_range⟩, h₁⟩, R, ⟨?_⟩⟩
    /-
      case intro.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.LocallyRingedSpace
      h : ∀ (x : ↑X.toTopCat), Exists fun R => Exists fun f => And (Membership.mem ( …
      x : ↑X.toTopCat
      R : CommRingCat
      f : Quiver.Hom (AlgebraicGeometry.Spec.toLocallyRingedSpace.obj { unop := R }) X
      h₁ : Membership.mem (Set.range ⇑f.base) x
      h₂ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Iso (X.restrict ⋯) (AlgebraicGeometry.Spec.toLocallyRingedSpa …
    -/
    apply LocallyRingedSpace.isoOfSheafedSpaceIso
    /-
      case intro.intro.intro.f
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.LocallyRingedSpace
      h : ∀ (x : ↑X.toTopCat), Exists fun R => Exists fun f => And (Membership.mem ( …
      x : ↑X.toTopCat
      R : CommRingCat
      f : Quiver.Hom (AlgebraicGeometry.Spec.toLocallyRingedSpace.obj { unop := R }) X
      h₁ : Membership.mem (Set.range ⇑f.base) x
      h₂ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Iso (X.restrict ⋯).toSheafedSpace (AlgebraicGeometry.Spec.toL …
    -/
    refine SheafedSpace.forgetToPresheafedSpace.preimageIso ?_
    /-
      case intro.intro.intro.f
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.LocallyRingedSpace
      h : ∀ (x : ↑X.toTopCat), Exists fun R => Exists fun f => And (Membership.mem ( …
      x : ↑X.toTopCat
      R : CommRingCat
      f : Quiver.Hom (AlgebraicGeometry.Spec.toLocallyRingedSpace.obj { unop := R }) X
      h₁ : Membership.mem (Set.range ⇑f.base) x
      h₂ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
      ⊢ CategoryTheory.Iso (AlgebraicGeometry.SheafedSpace.forgetToPresheafedSpace.o …
    -/
    apply PresheafedSpace.IsOpenImmersion.isoOfRangeEq (PresheafedSpace.ofRestrict _ _) f.1
      /-
        case intro.intro.intro.f
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : AlgebraicGeometry.LocallyRingedSpace
        h : ∀ (x : ↑X.toTopCat), Exists fun R => Exists fun f => And (Membership.mem ( …
        x : ↑X.toTopCat
        R : CommRingCat
        f : Quiver.Hom (AlgebraicGeometry.Spec.toLocallyRingedSpace.obj { unop := R }) X
        h₁ : Membership.mem (Set.range ⇑f.base) x
        h₂ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        ⊢ Eq (Set.range ⇑(X.ofRestrict ?m.4297).base) (Set.range ⇑f.base)
      -/
    · exact Subtype.range_coe_subtype
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : AlgebraicGeometry.LocallyRingedSpace
        h : ∀ (x : ↑X.toTopCat), Exists fun R => Exists fun f => And (Membership.mem ( …
        x : ↑X.toTopCat
        R : CommRingCat
        f : Quiver.Hom (AlgebraicGeometry.Spec.toLocallyRingedSpace.obj { unop := R }) X
        h₁ : Membership.mem (Set.range ⇑f.base) x
        h₂ : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion f
        ⊢ Topology.IsOpenEmbedding ⇑{ obj := { carrier := Set.range ⇑f.base, is_open'  …
      -/
    · exact Opens.isOpenEmbedding _ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11187): was `infer_instance`
      /-
        🎉 no goals
      -/


theorem IsOpenImmersion.isOpen_range {X Y : Scheme.{u}} (f : X ⟶ Y) [H : IsOpenImmersion f] :
    IsOpen (Set.range f.base) :=
  H.base_open.isOpen_range


@[deprecated (since := "2024-03-17")]
alias IsOpenImmersion.open_range := IsOpenImmersion.isOpen_range


theorem isOpenEmbedding : IsOpenEmbedding f.base :=
  H.base_open


@[deprecated (since := "2024-10-18")]
alias openEmbedding := isOpenEmbedding


/-- The image of an open immersion as an open set. -/
@[simps]
def opensRange : Y.Opens :=
  ⟨_, f.isOpenEmbedding.isOpen_range⟩


/-- The functor `opens X ⥤ opens Y` associated with an open immersion `f : X ⟶ Y`. -/
abbrev opensFunctor : X.Opens ⥤ Y.Opens :=
  LocallyRingedSpace.IsOpenImmersion.opensFunctor f.toLRSHom


/-- `f ''ᵁ U` is notation for the image (as an open set) of `U` under an open immersion `f`. -/
scoped[AlgebraicGeometry] notation3:90 f:91 " ''ᵁ " U:90 => (Scheme.Hom.opensFunctor f).obj U


lemma image_le_image_of_le {U V : X.Opens} (e : U ≤ V) : f ''ᵁ U ≤ f ''ᵁ V := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U V : X.Opens
    e : LE.le U V
    ⊢ LE.le (f.opensFunctor.obj U) (f.opensFunctor.obj V)
  -/
  rintro a ⟨u, hu, rfl⟩
  /-
    case intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U V : X.Opens
    e : LE.le U V
    u : ↑↑X.toPresheafedSpace
    hu : Membership.mem (↑U) u
    ⊢ Membership.mem (↑(f.opensFunctor.obj V)) (f.toLRSHom.base u)
  -/
  exact Set.mem_image_of_mem (⇑f.base) (e hu)
  /-
    🎉 no goals
  -/


@[simp]
lemma opensFunctor_map_homOfLE {U V : X.Opens} (e : U ≤ V) :
    (Scheme.Hom.opensFunctor f).map (homOfLE e) = homOfLE (f.image_le_image_of_le e) :=
  rfl


@[simp]
lemma image_top_eq_opensRange : f ''ᵁ ⊤ = f.opensRange := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Eq (f.opensFunctor.obj Top.top) f.opensRange
  -/
  apply Opens.ext
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Eq ↑(f.opensFunctor.obj Top.top) ↑f.opensRange
  -/
  simp
  /-
    🎉 no goals
  -/


lemma opensRange_comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z)
    [IsOpenImmersion f] [IsOpenImmersion g] : (f ≫ g).opensRange = g ''ᵁ f.opensRange :=
  TopologicalSpace.Opens.ext (Set.range_comp g.base f.base)


lemma opensRange_of_isIso {X Y : Scheme} (f : X ⟶ Y) [IsIso f] :
    f.opensRange = ⊤ :=
  TopologicalSpace.Opens.ext (Set.range_eq_univ.mpr f.homeomorph.surjective)


lemma opensRange_comp_of_isIso {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z)
    [IsIso f] [IsOpenImmersion g] : (f ≫ g).opensRange = g.opensRange := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.IsIso f
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.CategoryStruct.c …
  -/
  rw [opensRange_comp, opensRange_of_isIso, image_top_eq_opensRange]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_image_eq (U : X.Opens) : f ⁻¹ᵁ f ''ᵁ U = U := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U : X.Opens
    ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj (f.opensFunctor.obj U)) U
  -/
  apply Opens.ext
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U : X.Opens
    ⊢ Eq ↑((TopologicalSpace.Opens.map f.base).obj (f.opensFunctor.obj U)) ↑U
  -/
  simp [Set.preimage_image_eq _ f.isOpenEmbedding.injective]
  /-
    🎉 no goals
  -/


lemma image_le_image_iff (f : X ⟶ Y) [IsOpenImmersion f] (U U' : X.Opens) :
    f ''ᵁ U ≤ f ''ᵁ U' ↔ U ≤ U' := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U U' : X.Opens
    ⊢ Iff (LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj U) ((Algebraic …
  -/
  refine ⟨fun h ↦ ?_, image_le_image_of_le f⟩
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U U' : X.Opens
    h : LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj U) ((AlgebraicGeo …
    ⊢ LE.le U U'
  -/
  rw [← preimage_image_eq f U, ← preimage_image_eq f U']
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U U' : X.Opens
    h : LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj U) ((AlgebraicGeo …
    ⊢ LE.le ((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometry.Scheme.Ho …
  -/
  apply preimage_le_preimage_of_le f h
  /-
    🎉 no goals
  -/


lemma image_preimage_eq_opensRange_inter (U : Y.Opens) : f ''ᵁ f ⁻¹ᵁ U = f.opensRange ⊓ U := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U : Y.Opens
    ⊢ Eq (f.opensFunctor.obj ((TopologicalSpace.Opens.map f.base).obj U)) (Min.min …
  -/
  apply Opens.ext
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U : Y.Opens
    ⊢ Eq ↑(f.opensFunctor.obj ((TopologicalSpace.Opens.map f.base).obj U)) ↑(Min.m …
  -/
  simp [Set.image_preimage_eq_range_inter]
  /-
    🎉 no goals
  -/


lemma image_injective : Function.Injective (f ''ᵁ ·) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Function.Injective fun x => f.opensFunctor.obj x
  -/
  intro U V hUV
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U V : X.Opens
    hUV : Eq ((fun x => f.opensFunctor.obj x) U) ((fun x => f.opensFunctor.obj x) V)
    ⊢ Eq U V
  -/
  simpa using congrArg (f ⁻¹ᵁ ·) hUV
  /-
    🎉 no goals
  -/


lemma image_iSup {ι : Sort*} (s : ι → X.Opens) :
    (f ''ᵁ ⨆ (i : ι), s i) = ⨆ (i : ι), f ''ᵁ s i := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ι : Sort u_1
    s : ι → X.Opens
    ⊢ Eq (f.opensFunctor.obj (iSup fun i => s i)) (iSup fun i => f.opensFunctor.ob …
  -/
  ext : 1
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ι : Sort u_1
    s : ι → X.Opens
    ⊢ Eq ↑(f.opensFunctor.obj (iSup fun i => s i)) ↑(iSup fun i => f.opensFunctor. …
  -/
  simp [Set.image_iUnion]
  /-
    🎉 no goals
  -/


lemma image_iSup₂ {ι : Sort*} {κ : ι → Sort*} (s : (i : ι) → κ i → X.Opens) :
    (f ''ᵁ ⨆ (i : ι), ⨆ (j : κ i), s i j) = ⨆ (i : ι), ⨆ (j : κ i), f ''ᵁ s i j := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ι : Sort u_1
    κ : ι → Sort u_2
    s : (i : ι) → κ i → X.Opens
    ⊢ Eq (f.opensFunctor.obj (iSup fun i => iSup fun j => s i j)) (iSup fun i => i …
  -/
  ext : 1
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ι : Sort u_1
    κ : ι → Sort u_2
    s : (i : ι) → κ i → X.Opens
    ⊢ Eq ↑(f.opensFunctor.obj (iSup fun i => iSup fun j => s i j)) ↑(iSup fun i => …
  -/
  simp [Set.image_iUnion₂]
  /-
    🎉 no goals
  -/


/-- The isomorphism `Γ(Y, f(U)) ≅ Γ(X, U)` induced by an open immersion `f : X ⟶ Y`. -/
def appIso (U) : Γ(Y, f ''ᵁ U) ≅ Γ(X, U) :=
  (asIso <| LocallyRingedSpace.IsOpenImmersion.invApp f.toLRSHom U).symm


@[reassoc (attr := simp)]
theorem appIso_inv_naturality {U V : X.Opens} (i : op U ⟶ op V) :
    X.presheaf.map i ≫ (f.appIso V).inv =
      (f.appIso U).inv ≫ Y.presheaf.map (f.opensFunctor.op.map i) :=
  PresheafedSpace.IsOpenImmersion.inv_naturality _ _


theorem appIso_hom (U) :
    (f.appIso U).hom = f.app (f ''ᵁ U) ≫ X.presheaf.map
      (eqToHom (preimage_image_eq f U).symm).op :=
                                                                      /-
                                                                        X Y : AlgebraicGeometry.Scheme
                                                                        f : X.Hom Y
                                                                        H : AlgebraicGeometry.IsOpenImmersion f
                                                                        U : X.Opens
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.toPshHom.c.app { unop := (Algebrai …
                                                                      -/
  (PresheafedSpace.IsOpenImmersion.inv_invApp f.toPshHom U).trans (by rw [eqToHom_op])
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem appIso_hom' (U) :
    (f.appIso U).hom = f.appLE (f ''ᵁ U) U (preimage_image_eq f U).ge :=
  f.appIso_hom U


@[reassoc (attr := simp)]
theorem app_appIso_inv (U) :
    f.app U ≫ (f.appIso (f ⁻¹ᵁ U)).inv =
      Y.presheaf.map (homOfLE (Set.image_preimage_subset f.base U.1)).op :=
  PresheafedSpace.IsOpenImmersion.app_invApp _ _


/-- A variant of `app_invApp` that gives an `eqToHom` instead of `homOfLE`. -/
@[reassoc]
theorem app_invApp' (U) (hU : U ≤ f.opensRange) :
    f.app U ≫ (f.appIso (f ⁻¹ᵁ U)).inv =
                                               /-
                                                 C : Type u
                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                 X Y : AlgebraicGeometry.Scheme
                                                 f : X.Hom Y
                                                 H : AlgebraicGeometry.IsOpenImmersion f
                                                 U : Y.Opens
                                                 hU : LE.le U f.opensRange
                                                 ⊢ Eq ↑(f.opensFunctor.obj ((TopologicalSpace.Opens.map f.base).obj U)) ↑U
                                               -/
      Y.presheaf.map (eqToHom (Opens.ext <| by simpa [Set.image_preimage_eq_inter_range])).op :=
                                               /-
                                                 🎉 no goals
                                               -/
  PresheafedSpace.IsOpenImmersion.app_invApp _ _


@[reassoc (attr := simp), elementwise nosimp]
theorem appIso_inv_app (U) :
    (f.appIso U).inv ≫ f.app (f ''ᵁ U) = X.presheaf.map (eqToHom (preimage_image_eq f U)).op :=
                                                             /-
                                                               X Y : AlgebraicGeometry.Scheme
                                                               f : X.Hom Y
                                                               H : AlgebraicGeometry.IsOpenImmersion f
                                                               U : X.Opens
                                                               ⊢ Eq (X.presheaf.map (CategoryTheory.eqToHom ⋯)) (X.presheaf.map (CategoryTheo …
                                                             -/
  (PresheafedSpace.IsOpenImmersion.invApp_app _ _).trans (by rw [eqToHom_op])
                                                             /-
                                                               🎉 no goals
                                                             -/


/--
`elementwise` generates the `ConcreteCategory.instFunLike` lemma, we want `CommRingCat.Hom.hom`.
-/
theorem appIso_inv_app_apply' (U) (x) :
    f.app (f ''ᵁ U) ((f.appIso U).inv x) = X.presheaf.map (eqToHom (preimage_image_eq f U)).op x :=
  appIso_inv_app_apply f U x


@[reassoc (attr := simp), elementwise nosimp]
lemma appLE_appIso_inv {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] {U : Y.Opens}
    {V : X.Opens} (e : V ≤ f ⁻¹ᵁ U) :
    f.appLE U V e ≫ (f.appIso V).inv =
        Y.presheaf.map (homOfLE <| (f.image_le_image_of_le e).trans
          (f.image_preimage_eq_opensRange_inter U ▸ inf_le_right)).op := by
  simp only [appLE, Category.assoc, appIso_inv_naturality, Functor.op_obj, Functor.op_map,
    Quiver.Hom.unop_op, opensFunctor_map_homOfLE, app_appIso_inv_assoc, Opens.carrier_eq_coe]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory.homOf …
  -/
  rw [← Functor.map_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : Y.Opens
    V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    ⊢ Eq (Y.presheaf.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOf …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma appIso_inv_appLE {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] {U V : X.Opens}
    (e : V ≤ f ⁻¹ᵁ f ''ᵁ U) :
    (f.appIso U).inv ≫ f.appLE (f ''ᵁ U) V e =
                                    /-
                                      C : Type u
                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                      X✝ Y✝ : AlgebraicGeometry.Scheme
                                      f✝ : X✝.Hom Y✝
                                      H : AlgebraicGeometry.IsOpenImmersion f✝
                                      X Y : AlgebraicGeometry.Scheme
                                      f : Quiver.Hom X Y
                                      inst✝ : AlgebraicGeometry.IsOpenImmersion f
                                      U V : X.Opens
                                      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometry.Schem …
                                      ⊢ LE.le V U
                                    -/
        X.presheaf.map (homOfLE (by rwa [preimage_image_eq] at e)).op := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometry.Schem …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appIso  …
  -/
  simp only [appLE, appIso_inv_app_assoc, eqToHom_op]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometry.Schem …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map (CategoryTheory.eqToH …
  -/
  rw [← Functor.map_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U V : X.Opens
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometry.Schem …
    ⊢ Eq (X.presheaf.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToH …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The open sets of an open subscheme corresponds to the open sets containing in the image. -/
@[simps]
def IsOpenImmersion.opensEquiv {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] :
    X.Opens ≃ { U : Y.Opens // U ≤ f.opensRange } where
  toFun U := ⟨f ''ᵁ U, Set.image_subset_range _ _⟩
  invFun U := f ⁻¹ᵁ U
  left_inv _ := Opens.ext (Set.preimage_image_eq _ f.isOpenEmbedding.injective)
  right_inv U := Subtype.ext (Opens.ext (Set.image_preimage_eq_of_subset U.2))


instance basic_open_isOpenImmersion {R : CommRingCat.{u}} (f : R) :
    IsOpenImmersion (Spec.map (CommRingCat.ofHom (algebraMap R (Localization.Away f)))) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CommRingCat
    f : ↑R
    ⊢ AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRingCat.o …
  -/
  apply SheafedSpace.IsOpenImmersion.of_stalk_iso (H := ?_)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R : CommRingCat
      f : ↑R
      ⊢ Topology.IsOpenEmbedding ⇑(AlgebraicGeometry.Scheme.Hom.toLRSHom (AlgebraicG …
    -/
  · exact (PrimeSpectrum.localization_away_isOpenEmbedding (Localization.Away f) f : _)
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R : CommRingCat
      f : ↑R
      hf : Topology.IsOpenEmbedding ⇑(AlgebraicGeometry.Scheme.Hom.toLRSHom (Algebra …
      ⊢ ∀ (x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of (Localization.Away f))).toP …
    -/
  · intro x
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R : CommRingCat
      f : ↑R
      hf : Topology.IsOpenEmbedding ⇑(AlgebraicGeometry.Scheme.Hom.toLRSHom (Algebra …
      x : ↑↑(AlgebraicGeometry.Spec (CommRingCat.of (Localization.Away f))).toPreshe …
      ⊢ CategoryTheory.IsIso ((AlgebraicGeometry.Scheme.Hom.toLRSHom (AlgebraicGeome …
    -/
    exact Spec_map_localization_isIso R (Submonoid.powers f) x
    /-
      🎉 no goals
    -/


instance {R} [CommRing R] (f : R) :
    IsOpenImmersion (Spec.map (CommRingCat.ofHom (algebraMap R (Localization.Away f)))) :=
  basic_open_isOpenImmersion (R := .of R) f


lemma _root_.AlgebraicGeometry.IsOpenImmersion.of_isLocalization {R S} [CommRing R] [CommRing S]
    [Algebra R S] (f : R) [IsLocalization.Away f S] :
    IsOpenImmersion (Spec.map (CommRingCat.ofHom (algebraMap R S))) := by
  have e := (IsLocalization.algEquiv (.powers f) S
    (Localization.Away f)).symm.toAlgHom.comp_algebraMap
  /-
    R S : Type u_1
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    f : R
    inst✝ : IsLocalization.Away f S
    e : Eq ((↑↑(IsLocalization.algEquiv (Submonoid.powers f) S (Localization.Away  …
    ⊢ AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRingCat.o …
  -/
  rw [← e, CommRingCat.ofHom_comp, Spec.map_comp]
  have H : IsIso (CommRingCat.ofHom (IsLocalization.algEquiv
    (Submonoid.powers f) S (Localization.Away f)).symm.toAlgHom.toRingHom) := by
    exact inferInstanceAs (IsIso <| (IsLocalization.algEquiv
      (Submonoid.powers f) S (Localization.Away f)).toRingEquiv.toCommRingCatIso.inv)
  /-
    R S : Type u_1
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    f : R
    inst✝ : IsLocalization.Away f S
    e : Eq ((↑↑(IsLocalization.algEquiv (Submonoid.powers f) S (Localization.Away  …
    H : CategoryTheory.IsIso (CommRingCat.ofHom (↑(IsLocalization.algEquiv (Submon …
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp (Algeb …
  -/
  simp only [AlgEquiv.toAlgHom_eq_coe, AlgHom.toRingHom_eq_coe, AlgEquiv.toAlgHom_toRingHom] at H ⊢
  /-
    R S : Type u_1
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    f : R
    inst✝ : IsLocalization.Away f S
    e : Eq ((↑↑(IsLocalization.algEquiv (Submonoid.powers f) S (Localization.Away  …
    H : CategoryTheory.IsIso (CommRingCat.ofHom ↑(IsLocalization.algEquiv (Submono …
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp (Algeb …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem exists_affine_mem_range_and_range_subset
    {X : Scheme.{u}} {x : X} {U : X.Opens} (hxU : x ∈ U) :
    ∃ (R : CommRingCat) (f : Spec R ⟶ X),
      IsOpenImmersion f ∧ x ∈ Set.range f.base ∧ Set.range f.base ⊆ U := by
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    U : X.Opens
    hxU : Membership.mem U x
    ⊢ Exists fun R => Exists fun f => And (AlgebraicGeometry.IsOpenImmersion f) (A …
  -/
  obtain ⟨⟨V, hxV⟩, R, ⟨e⟩⟩ := X.2 x
  have : e.hom.base ⟨x, hxV⟩ ∈ (Opens.map (e.inv.base ≫ V.inclusion')).obj U :=
    show ((e.hom ≫ e.inv).base ⟨x, hxV⟩).1 ∈ U from e.hom_inv_id ▸ hxU
  obtain ⟨_, ⟨_, ⟨r : R, rfl⟩, rfl⟩, hr, hr'⟩ :=
    PrimeSpectrum.isBasis_basic_opens.exists_subset_of_mem_open this (Opens.is_open' _)
  let f : Spec (CommRingCat.of (Localization.Away r)) ⟶ X :=
    Spec.map (CommRingCat.ofHom (algebraMap R (Localization.Away r))) ≫ ⟨e.inv ≫ X.ofRestrict _⟩
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    U : X.Opens
    hxU : Membership.mem U x
    V : TopologicalSpace.Opens ↑X.toTopCat
    hxV : Membership.mem V x
    R : CommRingCat
    e : CategoryTheory.Iso (X.restrict ⋯) (AlgebraicGeometry.Spec.toLocallyRingedS …
    this : Membership.mem ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStr …
    r : ↑R
    hr : Membership.mem (↑(PrimeSpectrum.basicOpen r)) (e.hom.base ⟨x, hxV⟩)
    hr' : HasSubset.Subset ↑(PrimeSpectrum.basicOpen r) ↑((TopologicalSpace.Opens. …
    f : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (Localization.Away r))) …
    ⊢ Exists fun R => Exists fun f => And (AlgebraicGeometry.IsOpenImmersion f) (A …
  -/
  refine ⟨.of (Localization.Away r), f, inferInstance, ?_⟩
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    U : X.Opens
    hxU : Membership.mem U x
    V : TopologicalSpace.Opens ↑X.toTopCat
    hxV : Membership.mem V x
    R : CommRingCat
    e : CategoryTheory.Iso (X.restrict ⋯) (AlgebraicGeometry.Spec.toLocallyRingedS …
    this : Membership.mem ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStr …
    r : ↑R
    hr : Membership.mem (↑(PrimeSpectrum.basicOpen r)) (e.hom.base ⟨x, hxV⟩)
    hr' : HasSubset.Subset ↑(PrimeSpectrum.basicOpen r) ↑((TopologicalSpace.Opens. …
    f : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (Localization.Away r))) …
    ⊢ And (Membership.mem (Set.range ⇑f.base) x) (HasSubset.Subset (Set.range ⇑f.b …
  -/
  rw [Scheme.comp_base, TopCat.coe_comp, Set.range_comp]
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    U : X.Opens
    hxU : Membership.mem U x
    V : TopologicalSpace.Opens ↑X.toTopCat
    hxV : Membership.mem V x
    R : CommRingCat
    e : CategoryTheory.Iso (X.restrict ⋯) (AlgebraicGeometry.Spec.toLocallyRingedS …
    this : Membership.mem ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStr …
    r : ↑R
    hr : Membership.mem (↑(PrimeSpectrum.basicOpen r)) (e.hom.base ⟨x, hxV⟩)
    hr' : HasSubset.Subset ↑(PrimeSpectrum.basicOpen r) ↑((TopologicalSpace.Opens. …
    f : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (Localization.Away r))) …
    ⊢ And (Membership.mem (Set.image (⇑{ toHom_1 := CategoryTheory.CategoryStruct. …
  -/
  erw [PrimeSpectrum.localization_away_comap_range (Localization.Away r) r]
  /-
    case intro.mk.intro.intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    U : X.Opens
    hxU : Membership.mem U x
    V : TopologicalSpace.Opens ↑X.toTopCat
    hxV : Membership.mem V x
    R : CommRingCat
    e : CategoryTheory.Iso (X.restrict ⋯) (AlgebraicGeometry.Spec.toLocallyRingedS …
    this : Membership.mem ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStr …
    r : ↑R
    hr : Membership.mem (↑(PrimeSpectrum.basicOpen r)) (e.hom.base ⟨x, hxV⟩)
    hr' : HasSubset.Subset ↑(PrimeSpectrum.basicOpen r) ↑((TopologicalSpace.Opens. …
    f : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (Localization.Away r))) …
    ⊢ And (Membership.mem (Set.image ⇑{ toHom_1 := CategoryTheory.CategoryStruct.c …
  -/
  exact ⟨⟨_, hr, congr(($(e.hom_inv_id).base ⟨x, hxV⟩).1)⟩, Set.image_subset_iff.mpr hr'⟩
  /-
    🎉 no goals
  -/


/-- If `X ⟶ Y` is an open immersion, and `Y` is a scheme, then so is `X`. -/
def toScheme : Scheme := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : AlgebraicGeometry.PresheafedSpace CommRingCat
    Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y.toPresheafedSpace
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    ⊢ AlgebraicGeometry.Scheme
  -/
  apply LocallyRingedSpace.IsOpenImmersion.scheme (toLocallyRingedSpace _ f)
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : AlgebraicGeometry.PresheafedSpace CommRingCat
    Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y.toPresheafedSpace
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    ⊢ ∀ (x : ↑(AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toLocallyRingedSp …
  -/
  intro x
  obtain ⟨R, i, _, h₁, h₂⟩ :=
    Scheme.exists_affine_mem_range_and_range_subset (U := ⟨_, H.base_open.isOpen_range⟩) ⟨x, rfl⟩
  /-
    case intro.intro.intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : AlgebraicGeometry.PresheafedSpace CommRingCat
    Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y.toPresheafedSpace
    H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
    x : ↑(AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toLocallyRingedSpace Y …
    R : CommRingCat
    i : Quiver.Hom (AlgebraicGeometry.Spec R) Y
    left✝ : AlgebraicGeometry.IsOpenImmersion i
    h₁ : Membership.mem (Set.range ⇑i.base) (f.base x)
    h₂ : HasSubset.Subset (Set.range ⇑i.base) ↑{ carrier := Set.range ⇑f.base, is_ …
    ⊢ Exists fun R => Exists fun f_1 => And (Membership.mem (Set.range ⇑f_1.base)  …
  -/
  refine ⟨R, LocallyRingedSpace.IsOpenImmersion.lift (toLocallyRingedSpaceHom _ f) _ h₂, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.PresheafedSpace CommRingCat
      Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y.toPresheafedSpace
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      x : ↑(AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toLocallyRingedSpace Y …
      R : CommRingCat
      i : Quiver.Hom (AlgebraicGeometry.Spec R) Y
      left✝ : AlgebraicGeometry.IsOpenImmersion i
      h₁ : Membership.mem (Set.range ⇑i.base) (f.base x)
      h₂ : HasSubset.Subset (Set.range ⇑i.base) ↑{ carrier := Set.range ⇑f.base, is_ …
      ⊢ Membership.mem (Set.range ⇑(AlgebraicGeometry.LocallyRingedSpace.IsOpenImmer …
    -/
  · rw [LocallyRingedSpace.IsOpenImmersion.lift_range]; exact h₁
                                                        /-
                                                          🎉 no goals
                                                        -/
    /-
      case intro.intro.intro.intro.refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : AlgebraicGeometry.PresheafedSpace CommRingCat
      Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y.toPresheafedSpace
      H : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion f
      x : ↑(AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toLocallyRingedSpace Y …
      R : CommRingCat
      i : Quiver.Hom (AlgebraicGeometry.Spec R) Y
      left✝ : AlgebraicGeometry.IsOpenImmersion i
      h₁ : Membership.mem (Set.range ⇑i.base) (f.base x)
      h₂ : HasSubset.Subset (Set.range ⇑i.base) ↑{ carrier := Set.range ⇑f.base, is_ …
      ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (AlgebraicGeometry.Loca …
    -/
  · delta LocallyRingedSpace.IsOpenImmersion.lift; infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem toScheme_toLocallyRingedSpace :
    (toScheme Y f).toLocallyRingedSpace = toLocallyRingedSpace Y.1 f :=
  rfl


/-- If `X ⟶ Y` is an open immersion of PresheafedSpaces, and `Y` is a Scheme, we can
upgrade it into a morphism of Schemes.
-/
def toSchemeHom : toScheme Y f ⟶ Y :=
  ⟨toLocallyRingedSpaceHom _ f⟩


@[simp]
theorem toSchemeHom_toPshHom : (toSchemeHom Y f).toPshHom = f :=
  rfl


instance toSchemeHom_isOpenImmersion : AlgebraicGeometry.IsOpenImmersion (toSchemeHom Y f) :=
  H


theorem scheme_eq_of_locallyRingedSpace_eq {X Y : Scheme.{u}}
    (H : X.toLocallyRingedSpace = Y.toLocallyRingedSpace) : X = Y := by
  /-
    X Y : AlgebraicGeometry.Scheme
    H : Eq X.toLocallyRingedSpace Y.toLocallyRingedSpace
    ⊢ Eq X Y
  -/
  cases X; cases Y; congr
                    /-
                      🎉 no goals
                    -/


theorem scheme_toScheme {X Y : Scheme.{u}} (f : X ⟶ Y) [AlgebraicGeometry.IsOpenImmersion f] :
    toScheme Y f.toPshHom = X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Eq (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toScheme Y f.toPshHom) X
  -/
  apply scheme_eq_of_locallyRingedSpace_eq
  /-
    case H
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Eq (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toScheme Y f.toPshHom) …
  -/
  exact locallyRingedSpace_toLocallyRingedSpace f.toLRSHom
  /-
    🎉 no goals
  -/


/-- The restriction of a Scheme along an open embedding. -/
@[simps! (config := .lemmasOnly) carrier, simps! presheaf_obj]
def Scheme.restrict : Scheme :=
  { PresheafedSpace.IsOpenImmersion.toScheme X (X.toPresheafedSpace.ofRestrict h) with
    toPresheafedSpace := X.toPresheafedSpace.restrict h }


lemma Scheme.restrict_toPresheafedSpace :
    (X.restrict h).toPresheafedSpace = X.toPresheafedSpace.restrict h := rfl


/-- The canonical map from the restriction to the subspace. -/
@[simps! toLRSHom_base, simps! (config := .lemmasOnly) toLRSHom_c_app]
def Scheme.ofRestrict : X.restrict h ⟶ X :=
  ⟨X.toLocallyRingedSpace.ofRestrict h⟩


@[simp]
lemma Scheme.ofRestrict_app (V) :
    (X.ofRestrict h).app V = X.presheaf.map (h.isOpenMap.adjunction.counit.app V).op  :=
  Scheme.ofRestrict_toLRSHom_c_app X h (op V)


instance IsOpenImmersion.ofRestrict : IsOpenImmersion (X.ofRestrict h) :=
                                                                             /-
                                                                               C : Type u
                                                                               inst✝ : CategoryTheory.Category.{v, u} C
                                                                               U : TopCat
                                                                               X : AlgebraicGeometry.Scheme
                                                                               f : Quiver.Hom U (TopCat.of ↑↑X.toPresheafedSpace)
                                                                               h : Topology.IsOpenEmbedding ⇑f
                                                                               ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (X.ofRestrict h)
                                                                             -/
  show PresheafedSpace.IsOpenImmersion (X.toPresheafedSpace.ofRestrict h) by infer_instance
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
lemma Scheme.ofRestrict_appLE (V W e) :
    (X.ofRestrict h).appLE V W e = X.presheaf.map
                                                 /-
                                                   C : Type u
                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                   U : TopCat
                                                   X : AlgebraicGeometry.Scheme
                                                   f : Quiver.Hom U (TopCat.of ↑↑X.toPresheafedSpace)
                                                   h : Topology.IsOpenEmbedding ⇑f
                                                   V : X.Opens
                                                   W : (X.restrict h).Opens
                                                   e : LE.le W ((TopologicalSpace.Opens.map (X.ofRestrict h).base).obj V)
                                                   ⊢ LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor (X.ofRestrict h)).obj (Opp …
                                                 -/
      (homOfLE (show X.ofRestrict h ''ᵁ _ ≤ _ by exact Set.image_subset_iff.mpr e)).op := by
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    U : TopCat
    X : AlgebraicGeometry.Scheme
    f : Quiver.Hom U (TopCat.of ↑↑X.toPresheafedSpace)
    h : Topology.IsOpenEmbedding ⇑f
    V : X.Opens
    W : (X.restrict h).Opens
    e : LE.le W ((TopologicalSpace.Opens.map (X.ofRestrict h).base).obj V)
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE (X.ofRestrict h) V W e) (X.presheaf.m …
  -/
  dsimp [Hom.appLE]
  /-
    U : TopCat
    X : AlgebraicGeometry.Scheme
    f : Quiver.Hom U (TopCat.of ↑↑X.toPresheafedSpace)
    h : Topology.IsOpenEmbedding ⇑f
    V : X.Opens
    W : (X.restrict h).Opens
    e : LE.le W ((TopologicalSpace.Opens.map (X.ofRestrict h).base).obj V)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map (⋯.adjunction.counit. …
  -/
  exact (X.presheaf.map_comp _ _).symm
  /-
    🎉 no goals
  -/


@[simp]
lemma Scheme.ofRestrict_appIso (U) :
    (X.ofRestrict h).appIso U = Iso.refl _ := by
  /-
    U✝ : TopCat
    X : AlgebraicGeometry.Scheme
    f : Quiver.Hom U✝ (TopCat.of ↑↑X.toPresheafedSpace)
    h : Topology.IsOpenEmbedding ⇑f
    U : (X.restrict h).Opens
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appIso (X.ofRestrict h) U) (CategoryTheory. …
  -/
  ext1
  simp only [restrict_presheaf_obj, Hom.appIso_hom', ofRestrict_appLE, homOfLE_refl, op_id,
    CategoryTheory.Functor.map_id, Iso.refl_hom]


@[simp]
lemma Scheme.restrict_presheaf_map (V W) (i : V ⟶ W) :
    (X.restrict h).presheaf.map i = X.presheaf.map (homOfLE (show X.ofRestrict h ''ᵁ W.unop ≤
      X.ofRestrict h ''ᵁ V.unop from Set.image_subset _ i.unop.le)).op := rfl


instance (priority := 100) of_isIso [IsIso g] : IsOpenImmersion g :=
  LocallyRingedSpace.IsOpenImmersion.of_isIso _


theorem to_iso {X Y : Scheme.{u}} (f : X ⟶ Y) [h : IsOpenImmersion f] [Epi f.base] : IsIso f :=
  @isIso_of_reflects_iso _ _ _ _ _ _ f
    (Scheme.forgetToLocallyRingedSpace ⋙
      LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forgetToPresheafedSpace)
    (@PresheafedSpace.IsOpenImmersion.to_iso _ _ _ _ f.toPshHom h _) _


theorem of_stalk_iso {X Y : Scheme.{u}} (f : X ⟶ Y) (hf : IsOpenEmbedding f.base)
    [∀ x, IsIso (f.stalkMap x)] : IsOpenImmersion f :=
  haveI (x : X) : IsIso (f.toShHom.stalkMap x) := inferInstanceAs <| IsIso (f.stalkMap x)
  SheafedSpace.IsOpenImmersion.of_stalk_iso f.toShHom hf


instance stalk_iso {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] (x : X) :
    IsIso (f.stalkMap x) :=
  inferInstanceAs <| IsIso (f.toLRSHom.stalkMap x)


lemma of_comp {X Y Z : Scheme.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) [IsOpenImmersion g]
    [IsOpenImmersion (f ≫ g)] : IsOpenImmersion f :=
  haveI (x : X) : IsIso (f.stalkMap x) :=
    haveI : IsIso (g.stalkMap (f.base x) ≫ f.stalkMap x) := by
      /-
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        inst✝¹ : AlgebraicGeometry.IsOpenImmersion g
        inst✝ : AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp  …
        x : ↑↑X.toPresheafedSpace
        ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
      -/
      rw [← Scheme.stalkMap_comp]
      /-
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        inst✝¹ : AlgebraicGeometry.IsOpenImmersion g
        inst✝ : AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp  …
        x : ↑↑X.toPresheafedSpace
        ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory. …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    IsIso.of_isIso_comp_left (f := g.stalkMap (f.base x)) _
  IsOpenImmersion.of_stalk_iso _ <|
    IsOpenEmbedding.of_comp _ (Scheme.Hom.isOpenEmbedding g) (Scheme.Hom.isOpenEmbedding (f ≫ g))


theorem iff_stalk_iso {X Y : Scheme.{u}} (f : X ⟶ Y) :
    IsOpenImmersion f ↔ IsOpenEmbedding f.base ∧ ∀ x, IsIso (f.stalkMap x) :=
  ⟨fun H => ⟨H.1, fun x ↦ inferInstanceAs <| IsIso (f.toPshHom.stalkMap x)⟩,
    fun ⟨h₁, h₂⟩ => @IsOpenImmersion.of_stalk_iso _ _ f h₁ h₂⟩


theorem _root_.AlgebraicGeometry.isIso_iff_isOpenImmersion {X Y : Scheme.{u}} (f : X ⟶ Y) :
    IsIso f ↔ IsOpenImmersion f ∧ Epi f.base :=
  ⟨fun _ => ⟨inferInstance, inferInstance⟩, fun ⟨h₁, h₂⟩ => @IsOpenImmersion.to_iso _ _ f h₁ h₂⟩


theorem _root_.AlgebraicGeometry.isIso_iff_stalk_iso {X Y : Scheme.{u}} (f : X ⟶ Y) :
    IsIso f ↔ IsIso f.base ∧ ∀ x, IsIso (f.stalkMap x) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f) (And (CategoryTheory.IsIso f.base) (∀ (x : ↑↑X. …
  -/
  rw [isIso_iff_isOpenImmersion, IsOpenImmersion.iff_stalk_iso, and_comm, ← and_assoc]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (And (And (CategoryTheory.Epi f.base) (Topology.IsOpenEmbedding ⇑f.base) …
  -/
  refine and_congr ⟨?_, ?_⟩ Iff.rfl
    /-
      case refine_1
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ And (CategoryTheory.Epi f.base) (Topology.IsOpenEmbedding ⇑f.base) → Categor …
    -/
  · rintro ⟨h₁, h₂⟩
    convert_to
      IsIso
        (TopCat.isoOfHomeo
            (Homeomorph.homeomorphOfContinuousOpen
              (.ofBijective _ ⟨h₂.injective, (TopCat.epi_iff_surjective _).mp h₁⟩) h₂.continuous
              h₂.isOpenMap)).hom
    /-
      case refine_1.intro
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h₁ : CategoryTheory.Epi f.base
      h₂ : Topology.IsOpenEmbedding ⇑f.base
      ⊢ CategoryTheory.IsIso (TopCat.isoOfHomeo (Homeomorph.homeomorphOfContinuousOp …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.IsIso f.base → And (CategoryTheory.Epi f.base) (Topology.IsOp …
    -/
  · intro H; exact ⟨inferInstance, (TopCat.homeoOfIso (asIso f.base)).isOpenEmbedding⟩
             /-
               🎉 no goals
             -/


/-- An open immersion induces an isomorphism from the domain onto the image -/
def isoRestrict : X ≅ (Z.restrict H.base_open : _) :=
  Scheme.fullyFaithfulForgetToLocallyRingedSpace.preimageIso
    (LocallyRingedSpace.IsOpenImmersion.isoRestrict f.toLRSHom)


local notation "forget" => Scheme.forgetToLocallyRingedSpace


instance mono : Mono f :=
  Scheme.forgetToLocallyRingedSpace.mono_of_mono_map
                             /-
                               C : Type u
                               inst✝ : CategoryTheory.Category.{v, u} C
                               X Y Z : AlgebraicGeometry.Scheme
                               f : Quiver.Hom X Z
                               g : Quiver.Hom Y Z
                               H : AlgebraicGeometry.IsOpenImmersion f
                               ⊢ CategoryTheory.Mono (AlgebraicGeometry.Scheme.Hom.toLRSHom f)
                             -/
    (show Mono f.toLRSHom by infer_instance)
                             /-
                               🎉 no goals
                             -/


instance forget_map_isOpenImmersion : LocallyRingedSpace.IsOpenImmersion ((forget).map f) :=
  ⟨H.base_open, H.c_iso⟩


instance hasLimit_cospan_forget_of_left :
    HasLimit (cospan f g ⋙ Scheme.forgetToLocallyRingedSpace) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.HasLimit ((CategoryTheory.Limits.cospan f g).comp Alge …
  -/
  apply @hasLimitOfIso _ _ _ _ _ _ ?_ (diagramIsoCospan.{u} _).symm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.cospan (((CategoryTheo …
  -/
  change HasLimit (cospan ((forget).map f) ((forget).map g))
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.cospan (AlgebraicGeome …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance hasLimit_cospan_forget_of_left' :
    HasLimit (cospan ((cospan f g ⋙ forget).map Hom.inl) ((cospan f g ⋙ forget).map Hom.inr)) :=
  show HasLimit (cospan ((forget).map f) ((forget).map g)) from inferInstance


instance hasLimit_cospan_forget_of_right : HasLimit (cospan g f ⋙ forget) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.HasLimit ((CategoryTheory.Limits.cospan g f).comp Alge …
  -/
  apply @hasLimitOfIso _ _ _ _ _ _ ?_ (diagramIsoCospan.{u} _).symm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.cospan (((CategoryTheo …
  -/
  change HasLimit (cospan ((forget).map g) ((forget).map f))
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.cospan (AlgebraicGeome …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance hasLimit_cospan_forget_of_right' :
    HasLimit (cospan ((cospan g f ⋙ forget).map Hom.inl) ((cospan g f ⋙ forget).map Hom.inr)) :=
  show HasLimit (cospan ((forget).map g) ((forget).map f)) from inferInstance


instance forgetCreatesPullbackOfLeft : CreatesLimit (cospan f g) forget :=
  createsLimitOfFullyFaithfulOfIso
    (PresheafedSpace.IsOpenImmersion.toScheme Y (pullback.snd f.toLRSHom g.toLRSHom).toShHom)
                 /-
                   C : Type u
                   inst✝ : CategoryTheory.Category.{v, u} C
                   X Y Z : AlgebraicGeometry.Scheme
                   f : Quiver.Hom X Z
                   g : Quiver.Hom Y Z
                   H : AlgebraicGeometry.IsOpenImmersion f
                   ⊢ Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj (AlgebraicGeomet …
                 -/
    (eqToIso (by simp) ≪≫ HasLimit.isoOfNatIso (diagramIsoCospan _).symm)
                 /-
                   🎉 no goals
                 -/


instance forgetCreatesPullbackOfRight : CreatesLimit (cospan g f) forget :=
  createsLimitOfFullyFaithfulOfIso
    (PresheafedSpace.IsOpenImmersion.toScheme Y (pullback.fst g.toLRSHom f.toLRSHom).1)
                 /-
                   C : Type u
                   inst✝ : CategoryTheory.Category.{v, u} C
                   X Y Z : AlgebraicGeometry.Scheme
                   f : Quiver.Hom X Z
                   g : Quiver.Hom Y Z
                   H : AlgebraicGeometry.IsOpenImmersion f
                   ⊢ Eq (AlgebraicGeometry.Scheme.forgetToLocallyRingedSpace.obj (AlgebraicGeomet …
                 -/
    (eqToIso (by simp) ≪≫ HasLimit.isoOfNatIso (diagramIsoCospan _).symm)
                 /-
                   🎉 no goals
                 -/


instance forget_preservesOfLeft : PreservesLimit (cospan f g) forget :=
  CategoryTheory.preservesLimit_of_createsLimit_and_hasLimit _ _


instance forget_preservesOfRight : PreservesLimit (cospan g f) forget :=
  preservesPullback_symmetry _ _ _


instance hasPullback_of_left : HasPullback f g :=
  hasLimit_of_created (cospan f g) forget


instance hasPullback_of_right : HasPullback g f :=
  hasLimit_of_created (cospan g f) forget


instance pullback_snd_of_left : IsOpenImmersion (pullback.snd f g) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.Limits.pullback.snd f g)
  -/
  have := PreservesPullback.iso_hom_snd forget f g
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.Limits.pullback.snd f g)
  -/
  dsimp only [Scheme.forgetToLocallyRingedSpace, inducedFunctor_map] at this
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.Limits.pullback.snd f g)
  -/
  change LocallyRingedSpace.IsOpenImmersion _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (AlgebraicGeometry.Sche …
  -/
  rw [← this]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Preserves …
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Categor …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance pullback_fst_of_right : IsOpenImmersion (pullback.fst g f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.Limits.pullback.fst g f)
  -/
  rw [← pullbackSymmetry_hom_comp_snd]
  -- Porting note: was just `infer_instance`, it is a bit weird that no explicit class instance is
  -- provided but still class inference fail to find this
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp (Categ …
  -/
  exact LocallyRingedSpace.IsOpenImmersion.comp (H := inferInstance) _ _
  /-
    🎉 no goals
  -/


instance pullback_to_base [IsOpenImmersion g] :
    IsOpenImmersion (limit.π (cospan f g) WalkingCospan.one) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.Limits.limit.π (CategoryTh …
  -/
  rw [← limit.w (cospan f g) WalkingCospan.Hom.inl]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp (Categ …
  -/
  change IsOpenImmersion (_ ≫ f)
  -- Porting note: was just `infer_instance`, it is a bit weird that no explicit class instance is
  -- provided but still class inference fail to find this
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp (Categ …
  -/
  exact LocallyRingedSpace.IsOpenImmersion.comp (H := inferInstance) _ _
  /-
    🎉 no goals
  -/


instance forgetToTop_preserves_of_left : PreservesLimit (cospan f g) Scheme.forgetToTop := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g) Alge …
  -/
  delta Scheme.forgetToTop
  refine @Limits.comp_preservesLimit _ _ _ _ _ _ (K := cospan f g) _ _ (F := forget)
    (G := LocallyRingedSpace.forgetToTop) ?_ ?_
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.IsOpenImmersion f
      ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g) Alge …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.PreservesLimit ((CategoryTheory.Limits.cospan f g).com …
  -/
  refine @preservesLimit_of_iso_diagram _ _ _ _ _ _ _ _ _ (diagramIsoCospan.{u} _).symm ?_
  /-
    case refine_2
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan (((Catego …
  -/
  dsimp [LocallyRingedSpace.forgetToTop]
  /-
    case refine_2
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan (Algebrai …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance forgetToTop_preserves_of_right : PreservesLimit (cospan g f) Scheme.forgetToTop :=
  preservesPullback_symmetry _ _ _


theorem range_pullback_snd_of_left :
    Set.range (pullback.snd f g).base = (g ⁻¹ᵁ f.opensRange).1 := by
  rw [← show _ = (pullback.snd f g).base from
    PreservesPullback.iso_hom_snd Scheme.forgetToTop f g, TopCat.coe_comp, Set.range_comp,
    Set.range_eq_univ.mpr, ← @Set.preimage_univ _ _ (pullback.fst f.base g.base)]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): was `rw`
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.IsOpenImmersion f
      ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.snd (AlgebraicGeometry.Schem …
    -/
  · erw [TopCat.pullback_snd_image_fst_preimage]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.IsOpenImmersion f
      ⊢ Eq (Set.preimage (⇑(AlgebraicGeometry.Scheme.forgetToTop.map g)) (Set.image  …
    -/
    rw [Set.image_univ]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.IsOpenImmersion f
      ⊢ Eq (Set.preimage (⇑(AlgebraicGeometry.Scheme.forgetToTop.map g)) (Set.range  …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Function.Surjective ⇑(CategoryTheory.Limits.PreservesPullback.iso AlgebraicG …
  -/
  rw [← TopCat.epi_iff_surjective]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.PreservesPullback.iso AlgebraicGeo …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem opensRange_pullback_snd_of_left :
    (pullback.snd f g).opensRange = g ⁻¹ᵁ f.opensRange :=
  Opens.ext (range_pullback_snd_of_left f g)


theorem range_pullback_fst_of_right :
    Set.range (pullback.fst g f).base =
      ((Opens.map g.base).obj ⟨Set.range f.base, H.base_open.isOpen_range⟩).1 := by
  rw [← show _ = (pullback.fst g f).base from
    PreservesPullback.iso_hom_fst Scheme.forgetToTop g f, TopCat.coe_comp, Set.range_comp,
    Set.range_eq_univ.mpr, ← @Set.preimage_univ _ _ (pullback.snd g.base f.base)]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): was `rw`
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.IsOpenImmersion f
      ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.fst (AlgebraicGeometry.Schem …
    -/
  · erw [TopCat.pullback_fst_image_snd_preimage]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.IsOpenImmersion f
      ⊢ Eq (Set.preimage (⇑(AlgebraicGeometry.Scheme.forgetToTop.map g)) (Set.image  …
    -/
    rw [Set.image_univ]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      H : AlgebraicGeometry.IsOpenImmersion f
      ⊢ Eq (Set.preimage (⇑(AlgebraicGeometry.Scheme.forgetToTop.map g)) (Set.range  …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Function.Surjective ⇑(CategoryTheory.Limits.PreservesPullback.iso AlgebraicG …
  -/
  rw [← TopCat.epi_iff_surjective]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.PreservesPullback.iso AlgebraicGeo …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem opensRange_pullback_fst_of_right :
    (pullback.fst g f).opensRange = g ⁻¹ᵁ f.opensRange :=
  Opens.ext (range_pullback_fst_of_right f g)


theorem range_pullback_to_base_of_left :
    Set.range (pullback.fst f g ≫ f).base =
      Set.range f.base ∩ Set.range g.base := by
  rw [pullback.condition, Scheme.comp_base, TopCat.coe_comp, Set.range_comp,
    range_pullback_snd_of_left, Opens.carrier_eq_coe, Opens.map_obj, Opens.coe_mk,
    Set.image_preimage_eq_inter_range, Opens.carrier_eq_coe, Scheme.Hom.coe_opensRange]


theorem range_pullback_to_base_of_right :
    Set.range (pullback.fst g f ≫ g).base =
      Set.range g.base ∩ Set.range f.base := by
  rw [Scheme.comp_base, TopCat.coe_comp, Set.range_comp, range_pullback_fst_of_right,
    Opens.map_obj, Opens.carrier_eq_coe, Opens.coe_mk, Set.image_preimage_eq_inter_range,
    Set.inter_comm]


/-- The universal property of open immersions:
For an open immersion `f : X ⟶ Z`, given any morphism of schemes `g : Y ⟶ Z` whose topological
image is contained in the image of `f`, we can lift this morphism to a unique `Y ⟶ X` that
commutes with these maps.
-/
def lift (H' : Set.range g.base ⊆ Set.range f.base) : Y ⟶ X :=
  ⟨LocallyRingedSpace.IsOpenImmersion.lift f.toLRSHom g.toLRSHom H'⟩


@[simp, reassoc]
theorem lift_fac (H' : Set.range g.base ⊆ Set.range f.base) : lift f g H' ≫ f = g :=
  Scheme.Hom.ext' <| LocallyRingedSpace.IsOpenImmersion.lift_fac f.toLRSHom g.toLRSHom H'


theorem lift_uniq (H' : Set.range g.base ⊆ Set.range f.base) (l : Y ⟶ X) (hl : l ≫ f = g) :
    l = lift f g H' :=
  Scheme.Hom.ext' <| LocallyRingedSpace.IsOpenImmersion.lift_uniq
    f.toLRSHom g.toLRSHom H' l.toLRSHom congr(($hl).toLRSHom)


/-- Two open immersions with equal range are isomorphic. -/
def isoOfRangeEq [IsOpenImmersion g] (e : Set.range f.base = Set.range g.base) : X ≅ Y where
  hom := lift g f (le_of_eq e)
  inv := lift f g (le_of_eq e.symm)
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     X Y Z : AlgebraicGeometry.Scheme
                     f : Quiver.Hom X Z
                     g : Quiver.Hom Y Z
                     H : AlgebraicGeometry.IsOpenImmersion f
                     inst✝ : AlgebraicGeometry.IsOpenImmersion g
                     e : Eq (Set.range ⇑f.base) (Set.range ⇑g.base)
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.IsOpenImmersion.li …
                   -/
  hom_inv_id := by rw [← cancel_mono f]; simp
                                         /-
                                           🎉 no goals
                                         -/
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     X Y Z : AlgebraicGeometry.Scheme
                     f : Quiver.Hom X Z
                     g : Quiver.Hom Y Z
                     H : AlgebraicGeometry.IsOpenImmersion f
                     inst✝ : AlgebraicGeometry.IsOpenImmersion g
                     e : Eq (Set.range ⇑f.base) (Set.range ⇑g.base)
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.IsOpenImmersion.li …
                   -/
  inv_hom_id := by rw [← cancel_mono g]; simp
                                         /-
                                           🎉 no goals
                                         -/


@[simp, reassoc]
lemma isoOfRangeEq_hom_fac {X Y Z : Scheme.{u}} (f : X ⟶ Z) (g : Y ⟶ Z)
    [IsOpenImmersion f] [IsOpenImmersion g] (e : Set.range f.base = Set.range g.base) :
    (isoOfRangeEq f g e).hom ≫ g = f :=
  lift_fac _ _ (le_of_eq e)


@[simp, reassoc]
lemma isoOfRangeEq_inv_fac {X Y Z : Scheme.{u}} (f : X ⟶ Z) (g : Y ⟶ Z)
    [IsOpenImmersion f] [IsOpenImmersion g] (e : Set.range f.base = Set.range g.base) :
    (isoOfRangeEq f g e).inv ≫ f = g :=
  lift_fac _ _ (le_of_eq e.symm)


theorem app_eq_invApp_app_of_comp_eq_aux {X Y U : Scheme.{u}} (f : Y ⟶ U) (g : U ⟶ X) (fg : Y ⟶ X)
    (H : fg = f ≫ g) [h : IsOpenImmersion g] (V : U.Opens) :
    f ⁻¹ᵁ V = fg ⁻¹ᵁ (g ''ᵁ V) := by
  /-
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom Y U
    g : Quiver.Hom U X
    fg : Quiver.Hom Y X
    H : Eq fg (CategoryTheory.CategoryStruct.comp f g)
    h : AlgebraicGeometry.IsOpenImmersion g
    V : U.Opens
    ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj V) ((TopologicalSpace.Opens.map  …
  -/
  subst H
  /-
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom Y U
    g : Quiver.Hom U X
    h : AlgebraicGeometry.IsOpenImmersion g
    V : U.Opens
    ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj V) ((TopologicalSpace.Opens.map  …
  -/
  rw [Scheme.comp_base, Opens.map_comp_obj]
  /-
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom Y U
    g : Quiver.Hom U X
    h : AlgebraicGeometry.IsOpenImmersion g
    V : U.Opens
    ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj V) ((TopologicalSpace.Opens.map  …
  -/
  congr 1
  /-
    case e_a
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom Y U
    g : Quiver.Hom U X
    h : AlgebraicGeometry.IsOpenImmersion g
    V : U.Opens
    ⊢ Eq V ((TopologicalSpace.Opens.map g.base).obj ((AlgebraicGeometry.Scheme.Hom …
  -/
  ext1
  /-
    case e_a.h
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom Y U
    g : Quiver.Hom U X
    h : AlgebraicGeometry.IsOpenImmersion g
    V : U.Opens
    ⊢ Eq ↑V ↑((TopologicalSpace.Opens.map g.base).obj ((AlgebraicGeometry.Scheme.H …
  -/
  exact (Set.preimage_image_eq _ h.base_open.injective).symm
  /-
    🎉 no goals
  -/


/-- The `fg` argument is to avoid nasty stuff about dependent types. -/
theorem app_eq_appIso_inv_app_of_comp_eq {X Y U : Scheme.{u}} (f : Y ⟶ U) (g : U ⟶ X) (fg : Y ⟶ X)
    (H : fg = f ≫ g) [h : IsOpenImmersion g] (V : U.Opens) :
    f.app V = (g.appIso V).inv ≫ fg.app (g ''ᵁ V) ≫ Y.presheaf.map
      (eqToHom <| IsOpenImmersion.app_eq_invApp_app_of_comp_eq_aux f g fg H V).op := by
  /-
    X Y U : AlgebraicGeometry.Scheme
    f : Quiver.Hom Y U
    g : Quiver.Hom U X
    fg : Quiver.Hom Y X
    H : Eq fg (CategoryTheory.CategoryStruct.comp f g)
    h : AlgebraicGeometry.IsOpenImmersion g
    V : U.Opens
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app f V) (CategoryTheory.CategoryStruct.com …
  -/
  subst H
  rw [Scheme.comp_app, Category.assoc, Scheme.Hom.appIso_inv_app_assoc, f.naturality_assoc,
    ← Functor.map_comp, ← op_comp, Quiver.Hom.unop_op, eqToHom_map, eqToHom_trans,
    eqToHom_op, eqToHom_refl, CategoryTheory.Functor.map_id, Category.comp_id]


theorem lift_app {X Y U : Scheme.{u}} (f : U ⟶ Y) (g : X ⟶ Y) [IsOpenImmersion f] (H)
    (V : U.Opens) :
    (IsOpenImmersion.lift f g H).app V = (f.appIso V).inv ≫ g.app (f ''ᵁ V) ≫
      X.presheaf.map (eqToHom <| IsOpenImmersion.app_eq_invApp_app_of_comp_eq_aux _ _ _
        (IsOpenImmersion.lift_fac f g H).symm V).op :=
  IsOpenImmersion.app_eq_appIso_inv_app_of_comp_eq _ _ _ (lift_fac _ _ _).symm _


/-- If `f` is an open immersion `X ⟶ Y`, the global sections of `X`
are naturally isomorphic to the sections of `Y` over the image of `f`. -/
noncomputable
def ΓIso {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] (U : Y.Opens) :
    Γ(X, f⁻¹ᵁ U) ≅ Γ(Y, f.opensRange ⊓ U) :=
  (f.appIso (f⁻¹ᵁ U)).symm ≪≫
    Y.presheaf.mapIso (eqToIso <| (f.image_preimage_eq_opensRange_inter U).symm).op


@[simp]
lemma ΓIso_inv {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] (U : Y.Opens) :
    (ΓIso f U).inv = f.appLE (f.opensRange ⊓ U) (f⁻¹ᵁ U)
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X✝ Y✝ Z : AlgebraicGeometry.Scheme
            f✝ : Quiver.Hom X✝ Z
            g : Quiver.Hom Y✝ Z
            H : AlgebraicGeometry.IsOpenImmersion f✝
            X Y : AlgebraicGeometry.Scheme
            f : Quiver.Hom X Y
            inst✝ : AlgebraicGeometry.IsOpenImmersion f
            U : Y.Opens
            ⊢ LE.le ((TopologicalSpace.Opens.map f.base).obj U) ((TopologicalSpace.Opens.m …
          -/
      (by rw [← f.image_preimage_eq_opensRange_inter, f.preimage_image_eq]) := by
          /-
            🎉 no goals
          -/
  simp only [ΓIso, Iso.trans_inv, Functor.mapIso_inv, Iso.op_inv, eqToIso.inv, eqToHom_op,
    asIso_inv, IsIso.comp_inv_eq, Iso.symm_inv, Scheme.Hom.appIso_hom', Scheme.Hom.map_appLE]


@[reassoc, elementwise]
lemma map_ΓIso_inv {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] (U : Y.Opens) :
    Y.presheaf.map (homOfLE inf_le_right).op ≫ (ΓIso f U).inv = f.app U := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory.homOf …
  -/
  simp [Scheme.Hom.appLE_eq_app]
  /-
    🎉 no goals
  -/


@[reassoc, elementwise]
lemma ΓIso_hom_map {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] (U : Y.Opens) :
    f.app U ≫ (ΓIso f U).hom = Y.presheaf.map (homOfLE inf_le_right).op := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.app f U …
  -/
  rw [← map_ΓIso_inv]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : Y.Opens
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [-ΓIso_inv]
  /-
    🎉 no goals
  -/


/-- Given an open immersion `f : U ⟶ X`, the isomorphism between global sections
  of `U` and the sections of `X` at the image of `f`. -/
noncomputable
def ΓIsoTop {X Y : Scheme.{u}} (f : X ⟶ Y) [IsOpenImmersion f] :
    Γ(X, ⊤) ≅ Γ(Y, f.opensRange) :=
  (f.appIso ⊤).symm ≪≫ Y.presheaf.mapIso (eqToIso f.image_top_eq_opensRange.symm).op


instance {Z : Scheme.{u}} (f : X ⟶ Z) (g : Y ⟶ Z) [IsOpenImmersion f]
    (H' : Set.range g.base ⊆ Set.range f.base) [IsOpenImmersion g] :
    IsOpenImmersion (IsOpenImmersion.lift f g H') :=
                                                                  /-
                                                                    C : Type u
                                                                    inst✝² : CategoryTheory.Category.{v, u} C
                                                                    X Y Z✝ : AlgebraicGeometry.Scheme
                                                                    f✝ : Quiver.Hom X Z✝
                                                                    g✝ : Quiver.Hom Y Z✝
                                                                    H : AlgebraicGeometry.IsOpenImmersion f✝
                                                                    Z : AlgebraicGeometry.Scheme
                                                                    f : Quiver.Hom X Z
                                                                    g : Quiver.Hom Y Z
                                                                    inst✝¹ : AlgebraicGeometry.IsOpenImmersion f
                                                                    H' : HasSubset.Subset (Set.range ⇑g.base) (Set.range ⇑f.base)
                                                                    inst✝ : AlgebraicGeometry.IsOpenImmersion g
                                                                    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp (Algeb …
                                                                  -/
  haveI : IsOpenImmersion (IsOpenImmersion.lift f g H' ≫ f) := by simpa
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  IsOpenImmersion.of_comp _ f


instance isOpenImmersion_isStableUnderComposition :
    MorphismProperty.IsStableUnderComposition @IsOpenImmersion where
  comp_mem f g _ _ := LocallyRingedSpace.IsOpenImmersion.comp f.toLRSHom g.toLRSHom


instance isOpenImmersion_respectsIso : MorphismProperty.RespectsIso @IsOpenImmersion := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ CategoryTheory.MorphismProperty.RespectsIso @AlgebraicGeometry.IsOpenImmersion
  -/
  apply MorphismProperty.respectsIso_of_isStableUnderComposition
  /-
    case hP
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ LE.le (CategoryTheory.MorphismProperty.isomorphisms AlgebraicGeometry.Scheme …
  -/
  intro _ _ f (hf : IsIso f)
  /-
    case hP
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y✝
    hf : CategoryTheory.IsIso f
    ⊢ AlgebraicGeometry.IsOpenImmersion f
  -/
  have : IsIso f := hf
  /-
    case hP
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y✝
    hf this : CategoryTheory.IsIso f
    ⊢ AlgebraicGeometry.IsOpenImmersion f
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isOpenImmersion_isMultiplicative :
    MorphismProperty.IsMultiplicative @IsOpenImmersion where
  id_mem _ := inferInstance


instance isOpenImmersion_stableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange @IsOpenImmersion :=
  MorphismProperty.IsStableUnderBaseChange.mk' <| by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ (X Y S : AlgebraicGeometry.Scheme) (f : Quiver.Hom X S) (g : Quiver.Hom Y  …
    -/
    intro X Y Z f g _ H; infer_instance
                         /-
                           🎉 no goals
                         -/


theorem image_basicOpen {X Y : Scheme.{u}} (f : X ⟶ Y) [H : IsOpenImmersion f] {U : X.Opens}
    (r : Γ(X, U)) :
    f ''ᵁ X.basicOpen r = Y.basicOpen ((f.appIso U).inv r) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U : X.Opens
    r : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj (X.basicOpen r)) (Y.ba …
  -/
  have e := Scheme.preimage_basicOpen f ((f.appIso U).inv r)
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    U : X.Opens
    r : ↑(X.presheaf.obj { unop := U })
    e : Eq ((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen ((AlgebraicGeomet …
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj (X.basicOpen r)) (Y.ba …
  -/
  rw [Scheme.Hom.appIso_inv_app_apply', Scheme.basicOpen_res, inf_eq_right.mpr _] at e
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      r : ↑(X.presheaf.obj { unop := U })
      e : Eq ((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen ((AlgebraicGeomet …
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj (X.basicOpen r)) (Y.ba …
    -/
  · rw [← e, f.image_preimage_eq_opensRange_inter, inf_eq_right]
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      r : ↑(X.presheaf.obj { unop := U })
      e : Eq ((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen ((AlgebraicGeomet …
      ⊢ LE.le (Y.basicOpen ((AlgebraicGeometry.Scheme.Hom.appIso f U).inv.hom r)) (A …
    -/
    refine Set.Subset.trans (Scheme.basicOpen_le _ _) (Set.image_subset_range _ _)
    /-
      🎉 no goals
    -/
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      r : ↑(X.presheaf.obj { unop := U })
      e : Eq ((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen ((AlgebraicGeomet …
      ⊢ LE.le (X.basicOpen r) ((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGe …
    -/
  · exact (X.basicOpen_le r).trans (f.preimage_image_eq _).ge
    /-
      🎉 no goals
    -/


