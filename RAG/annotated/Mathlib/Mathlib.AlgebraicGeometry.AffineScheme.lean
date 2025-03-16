/-- The category of affine schemes -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet
-- @[nolint has_nonempty_instance]
def AffineScheme :=
  Scheme.Spec.EssImageSubcategory
deriving Category


/-- A Scheme is affine if the canonical map `X ⟶ Spec Γ(X)` is an isomorphism. -/
class IsAffine (X : Scheme) : Prop where
  affine : IsIso X.toSpecΓ


instance (X : Scheme.{u}) [IsAffine X] : IsIso (ΓSpec.adjunction.unit.app X) := @IsAffine.affine X _


/-- The canonical isomorphism `X ≅ Spec Γ(X)` for an affine scheme. -/
@[simps! (config := .lemmasOnly) hom]
def Scheme.isoSpec (X : Scheme) [IsAffine X] : X ≅ Spec Γ(X, ⊤) :=
  asIso X.toSpecΓ


@[reassoc]
theorem Scheme.isoSpec_hom_naturality {X Y : Scheme} [IsAffine X] [IsAffine Y] (f : X ⟶ Y) :
    X.isoSpec.hom ≫ Spec.map (f.appTop) = f ≫ Y.isoSpec.hom := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp X.isoSpec.hom (AlgebraicGeometry.Spec …
  -/
  simp only [isoSpec, asIso_hom, Scheme.toSpecΓ_naturality]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem Scheme.isoSpec_inv_naturality {X Y : Scheme} [IsAffine X] [IsAffine Y] (f : X ⟶ Y) :
    Spec.map (f.appTop) ≫ Y.isoSpec.inv = X.isoSpec.inv ≫ f := by
  rw [Iso.eq_inv_comp, isoSpec, asIso_hom, ← Scheme.toSpecΓ_naturality_assoc, isoSpec,
    asIso_inv, IsIso.hom_inv_id, Category.comp_id]


/-- Construct an affine scheme from a scheme and the information that it is affine.
Also see `AffineScheme.of` for a typeclass version. -/
@[simps]
def AffineScheme.mk (X : Scheme) (_ : IsAffine X) : AffineScheme :=
  ⟨X, ΓSpec.adjunction.mem_essImage_of_unit_isIso _⟩


/-- Construct an affine scheme from a scheme. Also see `AffineScheme.mk` for a non-typeclass
version. -/
def AffineScheme.of (X : Scheme) [h : IsAffine X] : AffineScheme :=
  AffineScheme.mk X h


/-- Type check a morphism of schemes as a morphism in `AffineScheme`. -/
def AffineScheme.ofHom {X Y : Scheme} [IsAffine X] [IsAffine Y] (f : X ⟶ Y) :
    AffineScheme.of X ⟶ AffineScheme.of Y :=
  f


theorem mem_Spec_essImage (X : Scheme) : X ∈ Scheme.Spec.essImage ↔ IsAffine X :=
  ⟨fun h => ⟨Functor.essImage.unit_isIso h⟩,
    fun _ => ΓSpec.adjunction.mem_essImage_of_unit_isIso _⟩


instance isAffine_affineScheme (X : AffineScheme.{u}) : IsAffine X.obj :=
  ⟨Functor.essImage.unit_isIso X.property⟩


instance (R : CommRingCatᵒᵖ) : IsAffine (Scheme.Spec.obj R) :=
  AlgebraicGeometry.isAffine_affineScheme ⟨_, Scheme.Spec.obj_mem_essImage R⟩


instance isAffine_Spec (R : CommRingCat) : IsAffine (Spec R) :=
  AlgebraicGeometry.isAffine_affineScheme ⟨_, Scheme.Spec.obj_mem_essImage (op R)⟩


theorem isAffine_of_isIso {X Y : Scheme} (f : X ⟶ Y) [IsIso f] [h : IsAffine Y] : IsAffine X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    h : AlgebraicGeometry.IsAffine Y
    ⊢ AlgebraicGeometry.IsAffine X
  -/
  rw [← mem_Spec_essImage] at h ⊢; exact Functor.essImage.ofIso (asIso f).symm h
                                   /-
                                     🎉 no goals
                                   -/


/-- If `f : X ⟶ Y` is a morphism between affine schemes, the corresponding arrow is isomorphic
to the arrow of the morphism on prime spectra induced by the map on global sections. -/
noncomputable
def arrowIsoSpecΓOfIsAffine {X Y : Scheme} [IsAffine X] [IsAffine Y] (f : X ⟶ Y) :
    Arrow.mk f ≅ Arrow.mk (Spec.map f.appTop) :=
  Arrow.isoMk X.isoSpec Y.isoSpec (ΓSpec.adjunction.unit_naturality _)


/-- If `f : A ⟶ B` is a ring homomorphism, the corresponding arrow is isomorphic
to the arrow of the morphism induced on global sections by the map on prime spectra. -/
def arrowIsoΓSpecOfIsAffine {A B : CommRingCat} (f : A ⟶ B) :
    Arrow.mk f ≅ Arrow.mk ((Spec.map f).appTop) :=
  Arrow.isoMk (Scheme.ΓSpecIso _).symm (Scheme.ΓSpecIso _).symm
    (Scheme.ΓSpecIso_inv_naturality f).symm


theorem Scheme.isoSpec_Spec (R : CommRingCat.{u}) :
    (Spec R).isoSpec = Scheme.Spec.mapIso (Scheme.ΓSpecIso R).op :=
  Iso.ext (SpecMap_ΓSpecIso_hom R).symm


@[simp] theorem Scheme.isoSpec_Spec_hom (R : CommRingCat.{u}) :
    (Spec R).isoSpec.hom = Spec.map (Scheme.ΓSpecIso R).hom :=
  (SpecMap_ΓSpecIso_hom R).symm


@[simp] theorem Scheme.isoSpec_Spec_inv (R : CommRingCat.{u}) :
    (Spec R).isoSpec.inv = Spec.map (Scheme.ΓSpecIso R).inv :=
  congr($(isoSpec_Spec R).inv)


lemma ext_of_isAffine {X Y : Scheme} [IsAffine Y] {f g : X ⟶ Y} (e : f.appTop = g.appTop) :
    f = g := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    f g : Quiver.Hom X Y
    e : Eq (AlgebraicGeometry.Scheme.Hom.appTop f) (AlgebraicGeometry.Scheme.Hom.a …
    ⊢ Eq f g
  -/
  rw [← cancel_mono Y.toSpecΓ, Scheme.toSpecΓ_naturality, Scheme.toSpecΓ_naturality, e]
  /-
    🎉 no goals
  -/


/-- The `Spec` functor into the category of affine schemes. -/
def Spec : CommRingCatᵒᵖ ⥤ AffineScheme :=
  Scheme.Spec.toEssImage

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11081): cannot automatically derive

instance Spec_full : Spec.Full := Functor.Full.toEssImage _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11081): cannot automatically derive

instance Spec_faithful : Spec.Faithful := Functor.Faithful.toEssImage _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11081): cannot automatically derive

instance Spec_essSurj : Spec.EssSurj := Functor.EssSurj.toEssImage (F := _)


/-- The forgetful functor `AffineScheme ⥤ Scheme`. -/
@[simps!]
def forgetToScheme : AffineScheme ⥤ Scheme :=
  Scheme.Spec.essImageInclusion

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11081): cannot automatically derive

instance forgetToScheme_full : forgetToScheme.Full :=
show (Scheme.Spec.essImageInclusion).Full from inferInstance

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11081): cannot automatically derive

instance forgetToScheme_faithful : forgetToScheme.Faithful :=
show (Scheme.Spec.essImageInclusion).Faithful from inferInstance


/-- The global section functor of an affine scheme. -/
def Γ : AffineSchemeᵒᵖ ⥤ CommRingCat :=
  forgetToScheme.op ⋙ Scheme.Γ


/-- The category of affine schemes is equivalent to the category of commutative rings. -/
def equivCommRingCat : AffineScheme ≌ CommRingCatᵒᵖ :=
  equivEssImageOfReflective.symm


instance : Γ.{u}.rightOp.IsEquivalence := equivCommRingCat.isEquivalence_functor


instance : Γ.{u}.rightOp.op.IsEquivalence := equivCommRingCat.op.isEquivalence_functor


instance ΓIsEquiv : Γ.{u}.IsEquivalence :=
  inferInstanceAs (Γ.{u}.rightOp.op ⋙ (opOpEquivalence _).functor).IsEquivalence


instance hasColimits : HasColimits AffineScheme.{u} :=
  haveI := Adjunction.has_limits_of_equivalence.{u} Γ.{u}
  Adjunction.has_colimits_of_equivalence.{u} (opOpEquivalence AffineScheme.{u}).inverse


instance hasLimits : HasLimits AffineScheme.{u} := by
  /-
    ⊢ CategoryTheory.Limits.HasLimits AlgebraicGeometry.AffineScheme
  -/
  haveI := Adjunction.has_colimits_of_equivalence Γ.{u}
  /-
    this : CategoryTheory.Limits.HasColimitsOfSize.{u, u, u, u + 1} (Opposite Alge …
    ⊢ CategoryTheory.Limits.HasLimits AlgebraicGeometry.AffineScheme
  -/
  haveI : HasLimits AffineScheme.{u}ᵒᵖᵒᵖ := Limits.hasLimits_op_of_hasColimits
  /-
    this✝ : CategoryTheory.Limits.HasColimitsOfSize.{u, u, u, u + 1} (Opposite Alg …
    this : CategoryTheory.Limits.HasLimits (Opposite (Opposite AlgebraicGeometry.A …
    ⊢ CategoryTheory.Limits.HasLimits AlgebraicGeometry.AffineScheme
  -/
  exact Adjunction.has_limits_of_equivalence (opOpEquivalence AffineScheme.{u}).inverse
  /-
    🎉 no goals
  -/


noncomputable instance Γ_preservesLimits : PreservesLimits Γ.{u}.rightOp := inferInstance


noncomputable instance forgetToScheme_preservesLimits : PreservesLimits forgetToScheme := by
  apply (config := { allowSynthFailures := true })
    @preservesLimits_of_natIso _ _ _ _ _ _
      (isoWhiskerRight equivCommRingCat.unitIso forgetToScheme).symm
  /-
    ⊢ CategoryTheory.Limits.PreservesLimitsOfSize.{u_1, u_1, u_1, u_1, u_1 + 1, u_ …
  -/
  change PreservesLimits (equivCommRingCat.functor ⋙ Scheme.Spec)
  /-
    ⊢ CategoryTheory.Limits.PreservesLimits (AlgebraicGeometry.AffineScheme.equivC …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- An open subset of a scheme is affine if the open subscheme is affine. -/
def IsAffineOpen {X : Scheme} (U : X.Opens) : Prop :=
  IsAffine U


/-- The set of affine opens as a subset of `opens X`. -/
def Scheme.affineOpens (X : Scheme) : Set X.Opens :=
  {U : X.Opens | IsAffineOpen U}


instance {Y : Scheme.{u}} (U : Y.affineOpens) : IsAffine U :=
  U.property


theorem isAffineOpen_opensRange {X Y : Scheme} [IsAffine X] (f : X ⟶ Y)
    [H : IsOpenImmersion f] : IsAffineOpen (Scheme.Hom.opensRange f) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.IsAffineOpen (AlgebraicGeometry.Scheme.Hom.opensRange f)
  -/
  refine isAffine_of_isIso (IsOpenImmersion.isoOfRangeEq f (Y.ofRestrict _) ?_).inv
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ Eq (Set.range ⇑f.base) (Set.range ⇑(Y.ofRestrict ⋯).base)
  -/
  exact Subtype.range_val.symm
  /-
    🎉 no goals
  -/


theorem isAffineOpen_top (X : Scheme) [IsAffine X] : IsAffineOpen (⊤ : X.Opens) := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    ⊢ AlgebraicGeometry.IsAffineOpen Top.top
  -/
  convert isAffineOpen_opensRange (𝟙 X)
  /-
    case h.e'_2
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    ⊢ Eq Top.top (AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Category …
  -/
  ext1
  /-
    case h.e'_2.h
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    ⊢ Eq ↑Top.top ↑(AlgebraicGeometry.Scheme.Hom.opensRange (CategoryTheory.Catego …
  -/
  exact Set.range_id.symm
  /-
    🎉 no goals
  -/


instance Scheme.isAffine_affineCover (X : Scheme) (i : X.affineCover.J) :
    IsAffine (X.affineCover.obj i) :=
  isAffine_Spec _


instance Scheme.isAffine_affineBasisCover (X : Scheme) (i : X.affineBasisCover.J) :
    IsAffine (X.affineBasisCover.obj i) :=
  isAffine_Spec _


instance Scheme.isAffine_affineOpenCover (X : Scheme) (𝒰 : X.AffineOpenCover) (i : 𝒰.J) :
    IsAffine (𝒰.openCover.obj i) :=
  inferInstanceAs (IsAffine (Spec (𝒰.obj i)))


instance {X} [IsAffine X] (i) :
    IsAffine ((Scheme.coverOfIsIso (P := @IsOpenImmersion) (𝟙 X)).obj i) := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    i : (AlgebraicGeometry.Scheme.coverOfIsIso (CategoryTheory.CategoryStruct.id X …
    ⊢ AlgebraicGeometry.IsAffine ((AlgebraicGeometry.Scheme.coverOfIsIso (Category …
  -/
  dsimp; infer_instance
         /-
           🎉 no goals
         -/


theorem isBasis_affine_open (X : Scheme) : Opens.IsBasis X.affineOpens := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ TopologicalSpace.Opens.IsBasis X.affineOpens
  -/
  rw [Opens.isBasis_iff_nbhd]
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ ∀ {U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace} {x : ↑↑X.toPresheafedSp …
  -/
  rintro U x (hU : x ∈ (U : Set X))
  /-
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : ↑↑X.toPresheafedSpace
    hU : Membership.mem (↑U) x
    ⊢ Exists fun U' => And (Membership.mem X.affineOpens U') (And (Membership.mem  …
  -/
  obtain ⟨S, hS, hxS, hSU⟩ := X.affineBasisCover_is_basis.exists_subset_of_mem_open hU U.isOpen
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : ↑↑X.toPresheafedSpace
    hU : Membership.mem (↑U) x
    S : Set ↑↑X.toPresheafedSpace
    hS : Membership.mem (setOf fun x => Exists fun a => Eq x (Set.range ⇑(X.affine …
    hxS : Membership.mem S x
    hSU : HasSubset.Subset S ↑U
    ⊢ Exists fun U' => And (Membership.mem X.affineOpens U') (And (Membership.mem  …
  -/
  refine ⟨⟨S, X.affineBasisCover_is_basis.isOpen hS⟩, ?_, hxS, hSU⟩
  /-
    case intro.intro.intro
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : ↑↑X.toPresheafedSpace
    hU : Membership.mem (↑U) x
    S : Set ↑↑X.toPresheafedSpace
    hS : Membership.mem (setOf fun x => Exists fun a => Eq x (Set.range ⇑(X.affine …
    hxS : Membership.mem S x
    hSU : HasSubset.Subset S ↑U
    ⊢ Membership.mem X.affineOpens { carrier := S, is_open' := ⋯ }
  -/
  rcases hS with ⟨i, rfl⟩
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    x : ↑↑X.toPresheafedSpace
    hU : Membership.mem (↑U) x
    i : X.affineBasisCover.J
    hxS : Membership.mem (Set.range ⇑(X.affineBasisCover.map i).base) x
    hSU : HasSubset.Subset (Set.range ⇑(X.affineBasisCover.map i).base) ↑U
    ⊢ Membership.mem X.affineOpens { carrier := Set.range ⇑(X.affineBasisCover.map …
  -/
  exact isAffineOpen_opensRange _
  /-
    🎉 no goals
  -/


theorem iSup_affineOpens_eq_top (X : Scheme) : ⨆ i : X.affineOpens, (i : X.Opens) = ⊤ := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Eq (iSup fun i => ↑i) Top.top
  -/
  apply Opens.ext
  /-
    case h
    X : AlgebraicGeometry.Scheme
    ⊢ Eq ↑(iSup fun i => ↑i) ↑Top.top
  -/
  rw [Opens.coe_iSup]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    ⊢ Eq (Set.iUnion fun i => ↑↑i) ↑Top.top
  -/
  apply IsTopologicalBasis.sUnion_eq
  /-
    case h.self
    X : AlgebraicGeometry.Scheme
    ⊢ TopologicalSpace.IsTopologicalBasis (Set.range fun i => ↑↑i)
  -/
  rw [← Set.image_eq_range]
  /-
    case h.self
    X : AlgebraicGeometry.Scheme
    ⊢ TopologicalSpace.IsTopologicalBasis (Set.image SetLike.coe X.affineOpens)
  -/
  exact isBasis_affine_open X
  /-
    🎉 no goals
  -/


theorem Scheme.map_PrimeSpectrum_basicOpen_of_affine
    (X : Scheme) [IsAffine X] (f : Γ(X, ⊤)) :
    X.isoSpec.hom ⁻¹ᵁ PrimeSpectrum.basicOpen f = X.basicOpen f :=
  Scheme.toSpecΓ_preimage_basicOpen _ _


theorem isBasis_basicOpen (X : Scheme) [IsAffine X] :
    Opens.IsBasis (Set.range (X.basicOpen : Γ(X, ⊤) → X.Opens)) := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    ⊢ TopologicalSpace.Opens.IsBasis (Set.range X.basicOpen)
  -/
  delta Opens.IsBasis
  convert PrimeSpectrum.isBasis_basic_opens.isInducing
    (TopCat.homeoOfIso (Scheme.forgetToTop.mapIso X.isoSpec)).isInducing using 1
  /-
    case h.e'_3
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    ⊢ Eq (Set.image SetLike.coe (Set.range X.basicOpen)) (Set.image (Set.preimage  …
  -/
  ext
  /-
    case h.e'_3.h
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    x✝ : Set ↑↑X.toPresheafedSpace
    ⊢ Iff (Membership.mem (Set.image SetLike.coe (Set.range X.basicOpen)) x✝) (Mem …
  -/
  simp only [Set.mem_image, exists_exists_eq_and]
  /-
    case h.e'_3.h
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    x✝ : Set ↑↑X.toPresheafedSpace
    ⊢ Iff (Exists fun x => And (Membership.mem (Set.range X.basicOpen) x) (Eq (↑x) …
  -/
  constructor
    /-
      case h.e'_3.h.mp
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      x✝ : Set ↑↑X.toPresheafedSpace
      ⊢ (Exists fun x => And (Membership.mem (Set.range X.basicOpen) x) (Eq (↑x) x✝) …
    -/
  · rintro ⟨_, ⟨x, rfl⟩, rfl⟩
    /-
      case h.e'_3.h.mp.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      x : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ Exists fun x_1 => And (Exists fun x => And (Membership.mem (Set.range PrimeS …
    -/
    refine ⟨_, ⟨_, ⟨x, rfl⟩, rfl⟩, ?_⟩
    /-
      case h.e'_3.h.mp.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      x : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ Eq (Set.preimage ⇑(TopCat.homeoOfIso (AlgebraicGeometry.Scheme.forgetToTop.m …
    -/
    exact congr_arg Opens.carrier (Scheme.toSpecΓ_preimage_basicOpen _ _)
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.mpr
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      x✝ : Set ↑↑X.toPresheafedSpace
      ⊢ (Exists fun x => And (Exists fun x_1 => And (Membership.mem (Set.range Prime …
    -/
  · rintro ⟨_, ⟨_, ⟨x, rfl⟩, rfl⟩, rfl⟩
    /-
      case h.e'_3.h.mpr.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      x : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ Exists fun x_1 => And (Membership.mem (Set.range X.basicOpen) x_1) (Eq (↑x_1 …
    -/
    refine ⟨_, ⟨x, rfl⟩, ?_⟩
    /-
      case h.e'_3.h.mpr.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      x : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ Eq (↑(X.basicOpen x)) (Set.preimage ⇑(TopCat.homeoOfIso (AlgebraicGeometry.S …
    -/
    exact congr_arg Opens.carrier (Scheme.toSpecΓ_preimage_basicOpen _ _).symm
    /-
      🎉 no goals
    -/


/-- The canonical map `U ⟶ Spec Γ(X, U)` for an open `U ⊆ X`. -/
noncomputable
def Scheme.Opens.toSpecΓ {X : Scheme.{u}} (U : X.Opens) :
    U.toScheme ⟶ Spec Γ(X, U) :=
  U.toScheme.toSpecΓ ≫ Spec.map U.topIso.inv


@[reassoc (attr := simp)]
lemma Scheme.Opens.toSpecΓ_SpecMap_map {X : Scheme} (U V : X.Opens) (h : U ≤ V) :
    U.toSpecΓ ≫ Spec.map (X.presheaf.map (homOfLE h).op) = X.homOfLE h ≫ V.toSpecΓ := by
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    h : LE.le U V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp U.toSpecΓ (AlgebraicGeometry.Spec.map …
  -/
  delta Scheme.Opens.toSpecΓ
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.Opens
    h : LE.le U V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [← Spec.map_comp, ← X.presheaf.map_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma Scheme.Opens.toSpecΓ_top {X : Scheme} :
    (⊤ : X.Opens).toSpecΓ = (⊤ : X.Opens).ι ≫ X.toSpecΓ := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Eq Top.top.toSpecΓ (CategoryTheory.CategoryStruct.comp Top.top.ι X.toSpecΓ)
  -/
  simp [Scheme.Opens.toSpecΓ]; rfl
                               /-
                                 🎉 no goals
                               -/


attribute [-simp] eqToHom_op in
/-- The isomorphism `U ≅ Spec Γ(X, U)` for an affine `U`. -/
@[simps! (config := .lemmasOnly) inv]
def isoSpec :
    ↑U ≅ Spec Γ(X, U) :=
  haveI : IsAffine U := hU
  U.toScheme.isoSpec ≪≫ Scheme.Spec.mapIso U.topIso.symm.op


lemma isoSpec_hom {X : Scheme.{u}} {U : X.Opens} (hU : IsAffineOpen U) :
    hU.isoSpec.hom = U.toSpecΓ := rfl


open IsLocalRing in
lemma isoSpec_hom_base_apply (x : U) :
    hU.isoSpec.hom.base x = (Spec.map (X.presheaf.germ U x x.2)).base (closedPoint _) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq (hU.isoSpec.hom.base x) ((AlgebraicGeometry.Spec.map (X.presheaf.germ U ↑ …
  -/
  dsimp [IsAffineOpen.isoSpec_hom, Scheme.isoSpec_hom, Scheme.toSpecΓ_base, Scheme.Opens.toSpecΓ]
  rw [← Scheme.comp_base_apply, ← Spec.map_comp,
    (Iso.eq_comp_inv _).mpr (Scheme.Opens.germ_stalkIso_hom U (V := ⊤) x trivial),
    X.presheaf.germ_res_assoc, Spec.map_comp, Scheme.comp_base_apply]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq ((AlgebraicGeometry.Spec.map (X.presheaf.germ U ↑x ⋯)).base ((AlgebraicGe …
  -/
  congr 1
  /-
    case h.e_6.h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq ((AlgebraicGeometry.Spec.map (U.stalkIso x).inv).base (IsLocalRing.closed …
  -/
  exact IsLocalRing.comap_closedPoint (U.stalkIso x).inv.hom
  /-
    🎉 no goals
  -/


lemma isoSpec_inv_appTop :
    hU.isoSpec.inv.appTop = U.topIso.hom ≫ (Scheme.ΓSpecIso Γ(X, U)).inv := by
  simp only [Scheme.Opens.toScheme_presheaf_obj, isoSpec_inv, Scheme.isoSpec, asIso_inv,
    Scheme.comp_coeBase, Opens.map_comp_obj, Opens.map_top, Scheme.comp_app, Scheme.inv_appTop,
    Scheme.Opens.topIso_hom, Scheme.ΓSpecIso_inv_naturality, IsIso.inv_comp_eq]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.Spec.map (X.presheaf …
  -/
  rw [Scheme.toSpecΓ_appTop]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app (AlgebraicGeometry.Spec.map (X.presheaf …
  -/
  erw [Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


lemma isoSpec_hom_appTop :
    hU.isoSpec.hom.appTop = (Scheme.ΓSpecIso Γ(X, U)).hom ≫ U.topIso.inv := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop hU.isoSpec.hom) (CategoryTheory.Cate …
  -/
  have := congr(inv $hU.isoSpec_inv_appTop)
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    this : Eq (CategoryTheory.inv (AlgebraicGeometry.Scheme.Hom.appTop hU.isoSpec. …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop hU.isoSpec.hom) (CategoryTheory.Cate …
  -/
  rw [IsIso.inv_comp, IsIso.Iso.inv_inv, IsIso.Iso.inv_hom] at this
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    this : Eq (CategoryTheory.inv (AlgebraicGeometry.Scheme.Hom.appTop hU.isoSpec. …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop hU.isoSpec.hom) (CategoryTheory.Cate …
  -/
  have := (Scheme.Γ.map_inv hU.isoSpec.inv.op).trans this
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    this✝ : Eq (CategoryTheory.inv (AlgebraicGeometry.Scheme.Hom.appTop hU.isoSpec …
    this : Eq (AlgebraicGeometry.Scheme.Γ.map (CategoryTheory.inv hU.isoSpec.inv.o …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appTop hU.isoSpec.hom) (CategoryTheory.Cate …
  -/
  rwa [← op_inv, IsIso.Iso.inv_inv] at this
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-16")] alias isoSpec_inv_app_top := isoSpec_inv_appTop

@[deprecated (since := "2024-11-16")] alias isoSpec_hom_app_top := isoSpec_hom_appTop


/-- The open immersion `Spec Γ(X, U) ⟶ X` for an affine `U`. -/
def fromSpec :
    Spec Γ(X, U) ⟶ X :=
  haveI : IsAffine U := hU
  hU.isoSpec.inv ≫ U.ι


instance isOpenImmersion_fromSpec :
    IsOpenImmersion hU.fromSpec := by
  /-
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ AlgebraicGeometry.IsOpenImmersion hU.fromSpec
  -/
  delta fromSpec
  /-
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp hU.iso …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma isoSpec_inv_ι : hU.isoSpec.inv ≫ U.ι = hU.fromSpec := rfl


@[simp]
theorem range_fromSpec :
    Set.range hU.fromSpec.base = (U : Set X) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (Set.range ⇑hU.fromSpec.base) ↑U
  -/
  delta IsAffineOpen.fromSpec; dsimp [IsAffineOpen.isoSpec_inv]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (Set.range (Function.comp (⇑U.ι.base) (Function.comp ⇑(↑U).isoSpec.inv.ba …
  -/
  rw [Set.range_comp, Set.range_eq_univ.mpr, Set.image_univ]
    /-
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ Eq (Set.range ⇑U.ι.base) ↑U
    -/
  · exact Subtype.range_coe
    /-
      🎉 no goals
    -/
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Function.Surjective (Function.comp ⇑(↑U).isoSpec.inv.base ⇑(AlgebraicGeometr …
  -/
  rw [← coe_comp, ← TopCat.epi_iff_surjective]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Sp …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem opensRange_fromSpec : hU.fromSpec.opensRange = U := Opens.ext (range_fromSpec hU)


@[reassoc (attr := simp)]
theorem map_fromSpec {V : X.Opens} (hV : IsAffineOpen V) (f : op U ⟶ op V) :
    Spec.map (X.presheaf.map f) ≫ hU.fromSpec = hV.fromSpec := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    f : Quiver.Hom { unop := U } { unop := V }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
  -/
  have : IsAffine U := hU
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    f : Quiver.Hom { unop := U } { unop := V }
    this : AlgebraicGeometry.IsAffine ↑U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
  -/
  haveI : IsAffine _ := hV
  conv_rhs =>
    rw [fromSpec, ← X.homOfLE_ι (V := U) f.unop.le, isoSpec_inv, Category.assoc,
      ← Scheme.isoSpec_inv_naturality_assoc,
      ← Spec.map_comp_assoc, Scheme.homOfLE_appTop, ← Functor.map_comp]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    f : Quiver.Hom { unop := U } { unop := V }
    this✝ : AlgebraicGeometry.IsAffine ↑U
    this : AlgebraicGeometry.IsAffine ↑V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
  -/
  rw [fromSpec, isoSpec_inv, Category.assoc, ← Spec.map_comp_assoc, ← Functor.map_comp]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    f : Quiver.Hom { unop := U } { unop := V }
    this✝ : AlgebraicGeometry.IsAffine ↑U
    this : AlgebraicGeometry.IsAffine ↑V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (X.preshe …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma Spec_map_appLE_fromSpec (f : X ⟶ Y) {V : X.Opens} {U : Y.Opens}
    (hU : IsAffineOpen U) (hV : IsAffineOpen V) (i : V ≤ f ⁻¹ᵁ U) :
    Spec.map (f.appLE U V i) ≫ hU.fromSpec = hV.fromSpec ≫ f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    V : X.Opens
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    hV : AlgebraicGeometry.IsAffineOpen V
    i : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  have : IsAffine U := hU
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    V : X.Opens
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    hV : AlgebraicGeometry.IsAffineOpen V
    i : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    this : AlgebraicGeometry.IsAffine ↑U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  simp only [IsAffineOpen.fromSpec, Category.assoc, isoSpec_inv]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    V : X.Opens
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    hV : AlgebraicGeometry.IsAffineOpen V
    i : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    this : AlgebraicGeometry.IsAffine ↑U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
  -/
  simp_rw [← Scheme.homOfLE_ι _ i]
  rw [Category.assoc, ← morphismRestrict_ι,
    ← Category.assoc _ (f ∣_ U) U.ι, ← @Scheme.isoSpec_inv_naturality_assoc,
    ← Spec.map_comp_assoc, ← Spec.map_comp_assoc, Scheme.comp_appTop, morphismRestrict_appTop,
    Scheme.homOfLE_appTop, Scheme.Hom.app_eq_appLE, Scheme.Hom.appLE_map,
    Scheme.Hom.appLE_map, Scheme.Hom.appLE_map, Scheme.Hom.map_appLE]


lemma fromSpec_top [IsAffine X] : (isAffineOpen_top X).fromSpec = X.isoSpec.inv := by
  rw [fromSpec, isoSpec_inv, Category.assoc, ← @Scheme.isoSpec_inv_naturality,
    ← Spec.map_comp_assoc, Scheme.Opens.ι_appTop, ← X.presheaf.map_comp, ← op_comp,
    eqToHom_comp_homOfLE, ← eqToHom_eq_homOfLE rfl, eqToHom_refl, op_id, X.presheaf.map_id,
    Spec.map_id, Category.id_comp]


lemma fromSpec_app_of_le (V : X.Opens) (h : U ≤ V) :
    hU.fromSpec.app V = X.presheaf.map (homOfLE h).op ≫
      (Scheme.ΓSpecIso Γ(X, U)).inv ≫ (Spec _).presheaf.map (homOfLE le_top).op := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    h : LE.le U V
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app hU.fromSpec V) (CategoryTheory.Category …
  -/
  have : U.ι ⁻¹ᵁ V = ⊤ := eq_top_iff.mpr fun x _ ↦ h x.2
  rw [IsAffineOpen.fromSpec, Scheme.comp_app, Scheme.Opens.ι_app, Scheme.app_eq _ this,
    ← Scheme.Hom.appTop, IsAffineOpen.isoSpec_inv_appTop]
  simp only [Scheme.Opens.toScheme_presheaf_map, Scheme.Opens.topIso_hom,
    Category.assoc, ← X.presheaf.map_comp_assoc]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    h : LE.le U V
    this : Eq ((TopologicalSpace.Opens.map U.ι.base).obj V) Top.top
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map (CategoryTheory.Categ …
  -/
  rfl
  /-
    🎉 no goals
  -/


include hU in
protected theorem isCompact :
    IsCompact (U : Set X) := by
  convert @IsCompact.image _ _ _ _ Set.univ hU.fromSpec.base PrimeSpectrum.compactSpace.1
    (by fun_prop)
  /-
    case h.e'_3
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (↑U) (Set.image (⇑hU.fromSpec.base) Set.univ)
  -/
  convert hU.range_fromSpec.symm
  /-
    case h.e'_3
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (Set.image (⇑hU.fromSpec.base) Set.univ) (Set.range ⇑hU.fromSpec.base)
  -/
  exact Set.image_univ
  /-
    🎉 no goals
  -/


include hU in
theorem image_of_isOpenImmersion (f : X ⟶ Y) [H : IsOpenImmersion f] :
    IsAffineOpen (f ''ᵁ U) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFunctor f …
  -/
  have : IsAffine _ := hU
  /-
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    this : AlgebraicGeometry.IsAffine ↑U
    ⊢ AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFunctor f …
  -/
  convert isAffineOpen_opensRange (U.ι ≫ f)
  /-
    case h.e'_2
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    this : AlgebraicGeometry.IsAffine ↑U
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj U) (AlgebraicGeometry. …
  -/
  ext1
  /-
    case h.e'_2.h
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    this : AlgebraicGeometry.IsAffine ↑U
    ⊢ Eq ↑((AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj U) ↑(AlgebraicGeometr …
  -/
  exact Set.image_eq_range _ _
  /-
    🎉 no goals
  -/


theorem preimage_of_isIso {U : Y.Opens} (hU : IsAffineOpen U) (f : X ⟶ Y) [IsIso f] :
    IsAffineOpen (f ⁻¹ᵁ U) :=
  haveI : IsAffine _ := hU
  isAffine_of_isIso (f ∣_ U)


theorem _root_.AlgebraicGeometry.Scheme.Hom.isAffineOpen_iff_of_isOpenImmersion
    (f : AlgebraicGeometry.Scheme.Hom X Y) [H : IsOpenImmersion f] {U : X.Opens} :
    IsAffineOpen (f ''ᵁ U) ↔ IsAffineOpen U := by
  refine ⟨fun hU => @isAffine_of_isIso _ _
    (IsOpenImmersion.isoOfRangeEq (X.ofRestrict U.isOpenEmbedding ≫ f) (Y.ofRestrict _) ?_).hom
      ?_ hU, fun hU => hU.image_of_isOpenImmersion f⟩
    /-
      case refine_1
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen (f.opensFunctor.obj U)
      ⊢ Eq (Set.range ⇑(CategoryTheory.CategoryStruct.comp (X.ofRestrict ⋯) f).base) …
    -/
  · rw [Scheme.comp_base, coe_comp, Set.range_comp]
    /-
      case refine_1
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen (f.opensFunctor.obj U)
      ⊢ Eq (Set.image (⇑f.base) (Set.range ⇑(X.ofRestrict ⋯).base)) (Set.range ⇑(Y.o …
    -/
    dsimp [Opens.coe_inclusion', Scheme.restrict]
    /-
      case refine_1
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen (f.opensFunctor.obj U)
      ⊢ Eq (Set.image (⇑f.base) (Set.range ⇑(TopologicalSpace.Opens.inclusion' U)))  …
    -/
    erw [Subtype.range_coe, Subtype.range_coe] -- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case refine_1
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen (f.opensFunctor.obj U)
      ⊢ Eq (Set.image ⇑f.base ↑U) ↑(f.opensFunctor.obj U)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen (f.opensFunctor.obj U)
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.IsOpenImmersion.isoOfRangeEq (Catego …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


/-- The affine open sets of an open subscheme corresponds to
the affine open sets containing in the image. -/
@[simps]
def _root_.AlgebraicGeometry.IsOpenImmersion.affineOpensEquiv (f : X ⟶ Y) [H : IsOpenImmersion f] :
    X.affineOpens ≃ { U : Y.affineOpens // U ≤ f.opensRange } where
  toFun U := ⟨⟨f ''ᵁ U, U.2.image_of_isOpenImmersion f⟩, Set.image_subset_range _ _⟩
  invFun U := ⟨f ⁻¹ᵁ U, f.isAffineOpen_iff_of_isOpenImmersion.mp (by
    /-
      X Y : AlgebraicGeometry.Scheme
      U✝ : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U✝
      f✝ : ↑(X.presheaf.obj { unop := U✝ })
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.IsOpenImmersion f
      U : Subtype fun U => LE.le (↑U) (AlgebraicGeometry.Scheme.Hom.opensRange f)
      ⊢ AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFunctor f …
    -/
    rw [show f ''ᵁ f ⁻¹ᵁ U = U from Opens.ext (Set.image_preimage_eq_of_subset U.2)]; exact U.1.2)⟩
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
  left_inv _ := Subtype.ext (Opens.ext (Set.preimage_image_eq _ H.base_open.injective))
  right_inv U := Subtype.ext (Subtype.ext (Opens.ext (Set.image_preimage_eq_of_subset U.2)))


/-- The affine open sets of an open subscheme
corresponds to the affine open sets containing in the subset. -/
@[simps! apply_coe_coe]
def _root_.AlgebraicGeometry.affineOpensRestrict {X : Scheme.{u}} (U : X.Opens) :
    U.toScheme.affineOpens ≃ { V : X.affineOpens // V ≤ U } :=
                                                                           /-
                                                                             X✝ Y : AlgebraicGeometry.Scheme
                                                                             U✝ : X✝.Opens
                                                                             hU : AlgebraicGeometry.IsAffineOpen U✝
                                                                             f : ↑(X✝.presheaf.obj { unop := U✝ })
                                                                             X : AlgebraicGeometry.Scheme
                                                                             U : X.Opens
                                                                             ⊢ Eq (fun U_1 => LE.le (↑U_1) (AlgebraicGeometry.Scheme.Hom.opensRange U.ι)) f …
                                                                           -/
  (IsOpenImmersion.affineOpensEquiv U.ι).trans (Equiv.subtypeEquivProp (by simp))
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
lemma _root_.AlgebraicGeometry.affineOpensRestrict_symm_apply_coe
    {X : Scheme.{u}} (U : X.Opens) (V) :
    ((affineOpensRestrict U).symm V).1 = U.ι ⁻¹ᵁ V := rfl


instance (priority := 100) _root_.AlgebraicGeometry.Scheme.compactSpace_of_isAffine
    (X : Scheme) [IsAffine X] :
    CompactSpace X :=
  ⟨(isAffineOpen_top X).isCompact⟩


@[simp]
theorem fromSpec_preimage_self :
    hU.fromSpec ⁻¹ᵁ U = ⊤ := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq ((TopologicalSpace.Opens.map hU.fromSpec.base).obj U) Top.top
  -/
  ext1
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq ↑((TopologicalSpace.Opens.map hU.fromSpec.base).obj U) ↑Top.top
  -/
  rw [Opens.map_coe, Opens.coe_top, ← hU.range_fromSpec, ← Set.image_univ]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (Set.preimage (⇑hU.fromSpec.base) (Set.image (⇑hU.fromSpec.base) Set.univ …
  -/
  exact Set.preimage_image_eq _ PresheafedSpace.IsOpenImmersion.base_open.injective
  /-
    🎉 no goals
  -/


theorem ΓSpecIso_hom_fromSpec_app :
    (Scheme.ΓSpecIso Γ(X, U)).hom ≫ hU.fromSpec.app U =
      (Spec Γ(X, U)).presheaf.map (eqToHom hU.fromSpec_preimage_self).op := by
  simp only [fromSpec, Scheme.comp_coeBase, Opens.map_comp_obj, Scheme.comp_app,
    Scheme.Opens.ι_app_self, eqToHom_op, Scheme.app_eq _ U.ι_preimage_self,
    Scheme.Opens.toScheme_presheaf_map, eqToHom_unop, eqToHom_map U.ι.opensFunctor, Opens.map_top,
    isoSpec_inv_appTop, Scheme.Opens.topIso_hom, Category.assoc, ← Functor.map_comp_assoc,
    eqToHom_trans, eqToHom_refl, X.presheaf.map_id, Category.id_comp, Iso.hom_inv_id_assoc]


@[elementwise]
theorem fromSpec_app_self :
    hU.fromSpec.app U = (Scheme.ΓSpecIso Γ(X, U)).inv ≫
      (Spec Γ(X, U)).presheaf.map (eqToHom hU.fromSpec_preimage_self).op := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app hU.fromSpec U) (CategoryTheory.Category …
  -/
  rw [← hU.ΓSpecIso_hom_fromSpec_app, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


theorem fromSpec_preimage_basicOpen' :
    hU.fromSpec ⁻¹ᵁ X.basicOpen f = (Spec Γ(X, U)).basicOpen ((Scheme.ΓSpecIso Γ(X, U)).inv f) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ((TopologicalSpace.Opens.map hU.fromSpec.base).obj (X.basicOpen f)) ((Alg …
  -/
  rw [Scheme.preimage_basicOpen, hU.fromSpec_app_self]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).basicOpen ((Cate …
  -/
  exact Scheme.basicOpen_res_eq _ _ (eqToHom hU.fromSpec_preimage_self).op
  /-
    🎉 no goals
  -/


theorem fromSpec_preimage_basicOpen :
    hU.fromSpec ⁻¹ᵁ X.basicOpen f = PrimeSpectrum.basicOpen f := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ((TopologicalSpace.Opens.map hU.fromSpec.base).obj (X.basicOpen f)) (Prim …
  -/
  rw [fromSpec_preimage_basicOpen', ← basicOpen_eq_of_affine]
  /-
    🎉 no goals
  -/


theorem fromSpec_image_basicOpen :
    hU.fromSpec ''ᵁ (PrimeSpectrum.basicOpen f) = X.basicOpen f := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor hU.fromSpec).obj (PrimeSpectr …
  -/
  rw [← hU.fromSpec_preimage_basicOpen]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor hU.fromSpec).obj ((Topologica …
  -/
  ext1
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ↑((AlgebraicGeometry.Scheme.Hom.opensFunctor hU.fromSpec).obj ((Topologic …
  -/
  change hU.fromSpec.base '' (hU.fromSpec.base ⁻¹' (X.basicOpen f : Set X)) = _
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq (Set.image (⇑hU.fromSpec.base) (Set.preimage ⇑hU.fromSpec.base ↑(X.basicO …
  -/
  rw [Set.image_preimage_eq_inter_range, Set.inter_eq_left, hU.range_fromSpec]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ HasSubset.Subset ↑(X.basicOpen f) ↑U
  -/
  exact Scheme.basicOpen_le _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem basicOpen_fromSpec_app :
    (Spec Γ(X, U)).basicOpen (hU.fromSpec.app U f) = PrimeSpectrum.basicOpen f := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq ((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).basicOpen ((Alge …
  -/
  rw [← hU.fromSpec_preimage_basicOpen, Scheme.preimage_basicOpen]
  /-
    🎉 no goals
  -/


include hU in
theorem basicOpen :
    IsAffineOpen (X.basicOpen f) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ AlgebraicGeometry.IsAffineOpen (X.basicOpen f)
  -/
  rw [← hU.fromSpec_image_basicOpen, Scheme.Hom.isAffineOpen_iff_of_isOpenImmersion]
  convert isAffineOpen_opensRange
    (Spec.map (CommRingCat.ofHom <| algebraMap Γ(X, U) (Localization.Away f)))
  /-
    case h.e'_2
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq (PrimeSpectrum.basicOpen f) (AlgebraicGeometry.Scheme.Hom.opensRange (Alg …
  -/
  exact Opens.ext (PrimeSpectrum.localization_away_comap_range (Localization.Away f) f).symm
  /-
    🎉 no goals
  -/


lemma Spec_basicOpen {R : CommRingCat} (f : R) :
    IsAffineOpen (X := Spec R) (PrimeSpectrum.basicOpen f) :=
  basicOpen_eq_of_affine f ▸ (isAffineOpen_top (Spec (.of R))).basicOpen _


instance [IsAffine X] (r : Γ(X, ⊤)) : IsAffine (X.basicOpen r) :=
  (isAffineOpen_top X).basicOpen _


include hU in
theorem ι_basicOpen_preimage (r : Γ(X, ⊤)) :
    IsAffineOpen ((X.basicOpen r).ι ⁻¹ᵁ U) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map (X.basicOpen r). …
  -/
  apply (X.basicOpen r).ι.isAffineOpen_iff_of_isOpenImmersion.mp
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFunctor ( …
  -/
  dsimp [Scheme.Hom.opensFunctor, LocallyRingedSpace.IsOpenImmersion.opensFunctor]
  rw [Opens.functor_obj_map_obj, Opens.isOpenEmbedding_obj_top, inf_comm,
    ← Scheme.basicOpen_res _ _ (homOfLE le_top).op]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ AlgebraicGeometry.IsAffineOpen (X.basicOpen ((X.presheaf.map (CategoryTheory …
  -/
  exact hU.basicOpen _
  /-
    🎉 no goals
  -/


include hU in
theorem exists_basicOpen_le {V : X.Opens} (x : V) (h : ↑x ∈ U) :
    ∃ f : Γ(X, U), X.basicOpen f ≤ V ∧ ↑x ∈ X.basicOpen f := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    x : Subtype fun x => Membership.mem V x
    h : Membership.mem U ↑x
    ⊢ Exists fun f => And (LE.le (X.basicOpen f) V) (Membership.mem (X.basicOpen f …
  -/
  have : IsAffine _ := hU
  obtain ⟨_, ⟨_, ⟨r, rfl⟩, rfl⟩, h₁, h₂⟩ :=
    (isBasis_basicOpen U).exists_subset_of_mem_open (x.2 : (⟨x, h⟩ : U) ∈ _)
      ((Opens.map U.inclusion').obj V).isOpen
  have :
    U.ι ''ᵁ (U.toScheme.basicOpen r) =
      X.basicOpen (X.presheaf.map (eqToHom U.isOpenEmbedding_obj_top.symm).op r) := by
    refine (Scheme.image_basicOpen U.ι r).trans ?_
    rw [Scheme.basicOpen_res_eq]
    simp only [Scheme.Opens.toScheme_presheaf_obj, Scheme.Opens.ι_appIso, Iso.refl_inv,
      CommRingCat.id_apply]
  /-
    case intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    x : Subtype fun x => Membership.mem V x
    h : Membership.mem U ↑x
    this✝ : AlgebraicGeometry.IsAffine ↑U
    r : ↑((↑U).presheaf.obj { unop := Top.top })
    h₁ : Membership.mem ↑((↑U).basicOpen r) ⟨↑x, h⟩
    h₂ : HasSubset.Subset ↑((↑U).basicOpen r) ↑((TopologicalSpace.Opens.map (Topol …
    this : Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen …
    ⊢ Exists fun f => And (LE.le (X.basicOpen f) V) (Membership.mem (X.basicOpen f …
  -/
  use X.presheaf.map (eqToHom U.isOpenEmbedding_obj_top.symm).op r
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    x : Subtype fun x => Membership.mem V x
    h : Membership.mem U ↑x
    this✝ : AlgebraicGeometry.IsAffine ↑U
    r : ↑((↑U).presheaf.obj { unop := Top.top })
    h₁ : Membership.mem ↑((↑U).basicOpen r) ⟨↑x, h⟩
    h₂ : HasSubset.Subset ↑((↑U).basicOpen r) ↑((TopologicalSpace.Opens.map (Topol …
    this : Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen …
    ⊢ And (LE.le (X.basicOpen ((X.presheaf.map (CategoryTheory.eqToHom ⋯).op).hom  …
  -/
  rw [← this]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    x : Subtype fun x => Membership.mem V x
    h : Membership.mem U ↑x
    this✝ : AlgebraicGeometry.IsAffine ↑U
    r : ↑((↑U).presheaf.obj { unop := Top.top })
    h₁ : Membership.mem ↑((↑U).basicOpen r) ⟨↑x, h⟩
    h₂ : HasSubset.Subset ↑((↑U).basicOpen r) ↑((TopologicalSpace.Opens.map (Topol …
    this : Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicOpen …
    ⊢ And (LE.le ((AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj ((↑U).basicO …
  -/
  exact ⟨Set.image_subset_iff.mpr h₂, ⟨_, h⟩, h₁, rfl⟩
  /-
    🎉 no goals
  -/


noncomputable
instance {R : CommRingCat} {U} : Algebra R Γ(Spec R, U) :=
  ((Scheme.ΓSpecIso R).inv ≫ (Spec R).presheaf.map (homOfLE le_top).op).hom.toAlgebra


@[simp]
lemma algebraMap_Spec_obj {R : CommRingCat} {U} : algebraMap R Γ(Spec R, U) =
    ((Scheme.ΓSpecIso R).inv ≫ (Spec R).presheaf.map (homOfLE le_top).op).hom := rfl


instance {R : CommRingCat} {f : R} :
    IsLocalization.Away f Γ(Spec R, PrimeSpectrum.basicOpen f) :=
  inferInstanceAs (IsLocalization.Away f
    ((Spec.structureSheaf R).val.obj (op <| PrimeSpectrum.basicOpen f)))


/-- Given an affine open U and some `f : U`,
this is the canonical map `Γ(𝒪ₓ, D(f)) ⟶ Γ(Spec 𝒪ₓ(U), D(f))`
This is an isomorphism, as witnessed by an `IsIso` instance. -/
def basicOpenSectionsToAffine :
    Γ(X, X.basicOpen f) ⟶ Γ(Spec Γ(X, U), PrimeSpectrum.basicOpen f) :=
  hU.fromSpec.c.app (op <| X.basicOpen f) ≫
    (Spec Γ(X, U)).presheaf.map (eqToHom <| (hU.fromSpec_preimage_basicOpen f).symm).op


instance basicOpenSectionsToAffine_isIso :
    IsIso (basicOpenSectionsToAffine hU f) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ CategoryTheory.IsIso (hU.basicOpenSectionsToAffine f)
  -/
  delta basicOpenSectionsToAffine
  /-
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (hU.fromSpec.c.app  …
  -/
  refine IsIso.comp_isIso' ?_ inferInstance
  /-
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ CategoryTheory.IsIso (hU.fromSpec.c.app { unop := X.basicOpen f })
  -/
  apply PresheafedSpace.IsOpenImmersion.isIso_of_subset
  /-
    case hU
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ HasSubset.Subset (↑(X.basicOpen f)) (Set.range ⇑hU.fromSpec.base)
  -/
  rw [hU.range_fromSpec]
  /-
    case hU
    X Y : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ HasSubset.Subset ↑(X.basicOpen f) ↑U
  -/
  exact RingedSpace.basicOpen_le _ _
  /-
    🎉 no goals
  -/


include hU in
theorem isLocalization_basicOpen :
    IsLocalization.Away f Γ(X, X.basicOpen f) := by
  apply
    (IsLocalization.isLocalization_iff_of_ringEquiv (Submonoid.powers f)
      (asIso <| basicOpenSectionsToAffine hU f).commRingCatIsoToRingEquiv).mpr
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ IsLocalization (Submonoid.powers f) ↑((AlgebraicGeometry.Spec (X.presheaf.ob …
  -/
  convert StructureSheaf.IsLocalization.to_basicOpen _ f using 1
  -- Porting note: more hand holding is required here, the next 4 lines were not necessary
  /-
    case h.e'_3.h.h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq ((CategoryTheory.asIso (hU.basicOpenSectionsToAffine f)).commRingCatIsoTo …
  -/
  delta StructureSheaf.openAlgebra
  /-
    case h.e'_3.h.h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq ((CategoryTheory.asIso (hU.basicOpenSectionsToAffine f)).commRingCatIsoTo …
  -/
  congr 1
  /-
    case h.e'_3.h.h.e_i
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq ((CategoryTheory.asIso (hU.basicOpenSectionsToAffine f)).commRingCatIsoTo …
  -/
  rw [RingEquiv.toRingHom_eq_coe, CategoryTheory.Iso.commRingCatIsoToRingEquiv_toRingHom, asIso_hom]
  /-
    case h.e'_3.h.h.e_i
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq ((hU.basicOpenSectionsToAffine f).hom.comp (algebraMap ↑(X.presheaf.obj { …
  -/
  dsimp [CommRingCat.ofHom, RingHom.algebraMap_toAlgebra]
  /-
    case h.e'_3.h.h.e_i
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq ((hU.basicOpenSectionsToAffine f).hom.comp (X.presheaf.map (CategoryTheor …
  -/
  change (X.presheaf.map _ ≫ basicOpenSectionsToAffine hU f).hom = _
  /-
    case h.e'_3.h.h.e_i
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map (CategoryTheory.homOf …
  -/
  delta basicOpenSectionsToAffine
  /-
    case h.e'_3.h.h.e_i
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.presheaf.map (CategoryTheory.homOf …
  -/
  rw [hU.fromSpec.naturality_assoc, hU.fromSpec_app_self]
  /-
    case h.e'_3.h.h.e_i
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, ← Functor.map_comp, ← op_comp]
  /-
    case h.e'_3.h.h.e_i
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    e_1✝ : Eq ↑((AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).presheaf.o …
    he✝ : Eq CommRing.toCommSemiring CommRing.toCommSemiring
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.ΓSpecIso (X …
  -/
  exact CommRingCat.hom_ext_iff.mp (StructureSheaf.toOpen_res _ _ _ _)
  /-
    🎉 no goals
  -/


instance _root_.AlgebraicGeometry.isLocalization_away_of_isAffine
    [IsAffine X] (r : Γ(X, ⊤)) :
    IsLocalization.Away r Γ(X, X.basicOpen r) :=
  isLocalization_basicOpen (isAffineOpen_top X) r


lemma appLE_eq_away_map {X Y : Scheme.{u}} (f : X ⟶ Y) {U : Y.Opens} (hU : IsAffineOpen U)
    {V : X.Opens} (hV : IsAffineOpen V) (e) (r : Γ(Y, U)) :
    letI := hU.isLocalization_basicOpen r
    letI := hV.isLocalization_basicOpen (f.appLE U V e r)
                                                                /-
                                                                  X✝ Y✝ : AlgebraicGeometry.Scheme
                                                                  U✝ : X✝.Opens
                                                                  hU✝ : AlgebraicGeometry.IsAffineOpen U✝
                                                                  f✝ : ↑(X✝.presheaf.obj { unop := U✝ })
                                                                  X Y : AlgebraicGeometry.Scheme
                                                                  f : Quiver.Hom X Y
                                                                  U : Y.Opens
                                                                  hU : AlgebraicGeometry.IsAffineOpen U
                                                                  V : X.Opens
                                                                  hV : AlgebraicGeometry.IsAffineOpen V
                                                                  e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
                                                                  r : ↑(Y.presheaf.obj { unop := U })
                                                                  this✝ : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r }) := A …
                                                                  this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f U V e).hom r …
                                                                  ⊢ LE.le (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f U V e).hom r)) ((T …
                                                                -/
    f.appLE (Y.basicOpen r) (X.basicOpen (f.appLE U V e r)) (by simp [Scheme.Hom.appLE]) =
                                                                /-
                                                                  🎉 no goals
                                                                -/
        CommRingCat.ofHom (IsLocalization.Away.map _ _ (f.appLE U V e).hom r) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r) (X.basicOpen ((Alge …
  -/
  letI := hU.isLocalization_basicOpen r
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    this : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r }) := Al …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r) (X.basicOpen ((Alge …
  -/
  letI := hV.isLocalization_basicOpen (f.appLE U V e r)
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    this✝ : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r }) := A …
    this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f U V e).hom r …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r) (X.basicOpen ((Alge …
  -/
  ext : 1
  /-
    case hf
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    this✝ : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r }) := A …
    this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f U V e).hom r …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r) (X.basicOpen ((Alge …
  -/
  apply IsLocalization.ringHom_ext (.powers r)
  rw [IsLocalization.Away.map, IsLocalization.map_comp,
    RingHom.algebraMap_toAlgebra, RingHom.algebraMap_toAlgebra, ← CommRingCat.hom_comp,
    ← CommRingCat.hom_comp, Scheme.Hom.appLE_map, Scheme.Hom.map_appLE]


lemma app_basicOpen_eq_away_map {X Y : Scheme.{u}} (f : X ⟶ Y) {U : Y.Opens}
    (hU : IsAffineOpen U) (h : IsAffineOpen (f ⁻¹ᵁ U)) (r : Γ(Y, U)) :
    haveI := hU.isLocalization_basicOpen r
    haveI := h.isLocalization_basicOpen (f.app U r)
    f.app (Y.basicOpen r) =
      (CommRingCat.ofHom
        (IsLocalization.Away.map Γ(Y, Y.basicOpen r) Γ(X, X.basicOpen (f.app U r)) (f.app U).hom r)
                                      /-
                                        X✝ Y✝ : AlgebraicGeometry.Scheme
                                        U✝ : X✝.Opens
                                        hU✝ : AlgebraicGeometry.IsAffineOpen U✝
                                        f✝ : ↑(X✝.presheaf.obj { unop := U✝ })
                                        X Y : AlgebraicGeometry.Scheme
                                        f : Quiver.Hom X Y
                                        U : Y.Opens
                                        hU : AlgebraicGeometry.IsAffineOpen U
                                        h : AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj U)
                                        r : ↑(Y.presheaf.obj { unop := U })
                                        this✝ : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r })
                                        this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.app f U).hom r) ↑(X. …
                                        ⊢ Eq ((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen r)) (X.basicOpen (( …
                                      -/
        ≫ X.presheaf.map (eqToHom (by simp)).op) := by
                                      /-
                                        🎉 no goals
                                      -/
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app f (Y.basicOpen r)) (CategoryTheory.Cate …
  -/
  haveI := hU.isLocalization_basicOpen r
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    this : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r })
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app f (Y.basicOpen r)) (CategoryTheory.Cate …
  -/
  haveI := h.isLocalization_basicOpen (f.app U r)
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    this✝ : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r })
    this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.app f U).hom r) ↑(X. …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app f (Y.basicOpen r)) (CategoryTheory.Cate …
  -/
  ext : 1
  /-
    case hf
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    this✝ : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r })
    this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.app f U).hom r) ↑(X. …
    ⊢ Eq (AlgebraicGeometry.Scheme.Hom.app f (Y.basicOpen r)).hom (CategoryTheory. …
  -/
  apply IsLocalization.ringHom_ext (.powers r)
  rw [IsLocalization.Away.map, CommRingCat.hom_comp, RingHom.comp_assoc,
    IsLocalization.map_comp, RingHom.algebraMap_toAlgebra,
    RingHom.algebraMap_toAlgebra, ← RingHom.comp_assoc, ← CommRingCat.hom_comp,
    ← CommRingCat.hom_comp, ← X.presheaf.map_comp]
  /-
    case hf.h
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    h : AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj U)
    r : ↑(Y.presheaf.obj { unop := U })
    this✝ : IsLocalization.Away r ↑(Y.presheaf.obj { unop := Y.basicOpen r })
    this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.app f U).hom r) ↑(X. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.map (CategoryTheory.homOf …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `f.app (Y.basicOpen r)` is isomorphic to map induced on localizations
`Γ(Y, Y.basicOpen r) ⟶ Γ(X, X.basicOpen (f.app U r))` -/
def appBasicOpenIsoAwayMap {X Y : Scheme.{u}} (f : X ⟶ Y) {U : Y.Opens}
    (hU : IsAffineOpen U) (h : IsAffineOpen (f ⁻¹ᵁ U)) (r : Γ(Y, U)) :
    haveI := hU.isLocalization_basicOpen r
    haveI := h.isLocalization_basicOpen (f.app U r)
    Arrow.mk (f.app (Y.basicOpen r)) ≅
      Arrow.mk (CommRingCat.ofHom (IsLocalization.Away.map Γ(Y, Y.basicOpen r)
        Γ(X, X.basicOpen (f.app U r)) (f.app U).hom r)) :=
                                                           /-
                                                             X✝ Y✝ : AlgebraicGeometry.Scheme
                                                             U✝ : X✝.Opens
                                                             hU✝ : AlgebraicGeometry.IsAffineOpen U✝
                                                             f✝ : ↑(X✝.presheaf.obj { unop := U✝ })
                                                             X Y : AlgebraicGeometry.Scheme
                                                             f : Quiver.Hom X Y
                                                             U : Y.Opens
                                                             hU : AlgebraicGeometry.IsAffineOpen U
                                                             h : AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj U)
                                                             r : ↑(Y.presheaf.obj { unop := U })
                                                             ⊢ Eq (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app f U).hom r)) ((Topologica …
                                                           -/
  Arrow.isoMk (Iso.refl _) (X.presheaf.mapIso (eqToIso (by simp)).op) <| by
                                                           /-
                                                             🎉 no goals
                                                           -/
    /-
      X✝ Y✝ : AlgebraicGeometry.Scheme
      U✝ : X✝.Opens
      hU✝ : AlgebraicGeometry.IsAffineOpen U✝
      f✝ : ↑(X✝.presheaf.obj { unop := U✝ })
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      h : AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj U)
      r : ↑(Y.presheaf.obj { unop := U })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
    -/
    simp [hU.app_basicOpen_eq_away_map f h]
    /-
      🎉 no goals
    -/


include hU in
theorem isLocalization_of_eq_basicOpen {V : X.Opens} (i : V ⟶ U) (e : V = X.basicOpen f) :
    @IsLocalization.Away _ _ f Γ(X, V) _ (X.presheaf.map i.op).hom.toAlgebra := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    V : X.Opens
    i : Quiver.Hom V U
    e : Eq V (X.basicOpen f)
    ⊢ IsLocalization.Away f ↑(X.presheaf.obj { unop := V })
  -/
  subst e; convert isLocalization_basicOpen hU f using 3
           /-
             🎉 no goals
           -/


instance _root_.AlgebraicGeometry.Γ_restrict_isLocalization
    (X : Scheme.{u}) [IsAffine X] (r : Γ(X, ⊤)) :
    IsLocalization.Away r Γ(X.basicOpen r, ⊤) :=
  (isAffineOpen_top X).isLocalization_of_eq_basicOpen r _ (Opens.isOpenEmbedding_obj_top _)


include hU in
theorem basicOpen_basicOpen_is_basicOpen (g : Γ(X, X.basicOpen f)) :
    ∃ f' : Γ(X, U), X.basicOpen f' = X.basicOpen g := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    g : ↑(X.presheaf.obj { unop := X.basicOpen f })
    ⊢ Exists fun f' => Eq (X.basicOpen f') (X.basicOpen g)
  -/
  have := isLocalization_basicOpen hU f
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    g : ↑(X.presheaf.obj { unop := X.basicOpen f })
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    ⊢ Exists fun f' => Eq (X.basicOpen f') (X.basicOpen g)
  -/
  obtain ⟨x, ⟨_, n, rfl⟩, rfl⟩ := IsLocalization.surj'' (Submonoid.powers f) g
  /-
    case intro.intro.mk.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    x : ↑(X.presheaf.obj { unop := U })
    n : Nat
    ⊢ Exists fun f' => Eq (X.basicOpen f') (X.basicOpen (HSMul.hSMul x ↑((IsLocali …
  -/
  use f * x
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    x : ↑(X.presheaf.obj { unop := U })
    n : Nat
    ⊢ Eq (X.basicOpen (HMul.hMul f x)) (X.basicOpen (HSMul.hSMul x ↑((IsLocalizati …
  -/
  rw [Algebra.smul_def, Scheme.basicOpen_mul, Scheme.basicOpen_mul, RingHom.algebraMap_toAlgebra]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    x : ↑(X.presheaf.obj { unop := U })
    n : Nat
    ⊢ Eq (Min.min (X.basicOpen f) (X.basicOpen x)) (Min.min (X.basicOpen ((X.presh …
  -/
  rw [Scheme.basicOpen_res]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    x : ↑(X.presheaf.obj { unop := U })
    n : Nat
    ⊢ Eq (Min.min (X.basicOpen f) (X.basicOpen x)) (Min.min (Min.min (X.basicOpen  …
  -/
  refine (inf_eq_left.mpr ?_).symm
  -- Porting note: a little help is needed here
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    x : ↑(X.presheaf.obj { unop := U })
    n : Nat
    ⊢ LE.le (Min.min (X.basicOpen f) (X.basicOpen x)) (X.basicOpen ↑((IsLocalizati …
  -/
  convert inf_le_left (α := X.Opens) using 1
  /-
    case h.e'_4
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    this : IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
    x : ↑(X.presheaf.obj { unop := U })
    n : Nat
    ⊢ Eq (X.basicOpen ↑((IsLocalization.toInvSubmonoid (Submonoid.powers f) ↑(X.pr …
  -/
  apply Scheme.basicOpen_of_isUnit
  apply
    Submonoid.leftInv_le_isUnit _
      (IsLocalization.toInvSubmonoid (Submonoid.powers f) (Γ(X, X.basicOpen f))
        _).prop


include hU in
theorem _root_.AlgebraicGeometry.exists_basicOpen_le_affine_inter
    {V : X.Opens} (hV : IsAffineOpen V) (x : X) (hx : x ∈ U ⊓ V) :
    ∃ (f : Γ(X, U)) (g : Γ(X, V)), X.basicOpen f = X.basicOpen g ∧ x ∈ X.basicOpen f := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (Min.min U V) x
    ⊢ Exists fun f => Exists fun g => And (Eq (X.basicOpen f) (X.basicOpen g)) (Me …
  -/
  obtain ⟨f, hf₁, hf₂⟩ := hU.exists_basicOpen_le ⟨x, hx.2⟩ hx.1
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (Min.min U V) x
    f : ↑(X.presheaf.obj { unop := U })
    hf₁ : LE.le (X.basicOpen f) V
    hf₂ : Membership.mem (X.basicOpen f) ↑⟨x, ⋯⟩
    ⊢ Exists fun f => Exists fun g => And (Eq (X.basicOpen f) (X.basicOpen g)) (Me …
  -/
  obtain ⟨g, hg₁, hg₂⟩ := hV.exists_basicOpen_le ⟨x, hf₂⟩ hx.2
  obtain ⟨f', hf'⟩ :=
    basicOpen_basicOpen_is_basicOpen hU f (X.presheaf.map (homOfLE hf₁ : _ ⟶ V).op g)
  /-
    case intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (Min.min U V) x
    f : ↑(X.presheaf.obj { unop := U })
    hf₁ : LE.le (X.basicOpen f) V
    hf₂ : Membership.mem (X.basicOpen f) ↑⟨x, ⋯⟩
    g : ↑(X.presheaf.obj { unop := V })
    hg₁ : LE.le (X.basicOpen g) (X.basicOpen f)
    hg₂ : Membership.mem (X.basicOpen g) ↑⟨x, hf₂⟩
    f' : ↑(X.presheaf.obj { unop := U })
    hf' : Eq (X.basicOpen f') (X.basicOpen ((X.presheaf.map (CategoryTheory.homOfL …
    ⊢ Exists fun f => Exists fun g => And (Eq (X.basicOpen f) (X.basicOpen g)) (Me …
  -/
  replace hf' := (hf'.trans (RingedSpace.basicOpen_res _ _ _)).trans (inf_eq_right.mpr hg₁)
  /-
    case intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (Min.min U V) x
    f : ↑(X.presheaf.obj { unop := U })
    hf₁ : LE.le (X.basicOpen f) V
    hf₂ : Membership.mem (X.basicOpen f) ↑⟨x, ⋯⟩
    g : ↑(X.presheaf.obj { unop := V })
    hg₁ : LE.le (X.basicOpen g) (X.basicOpen f)
    hg₂ : Membership.mem (X.basicOpen g) ↑⟨x, hf₂⟩
    f' : ↑(X.presheaf.obj { unop := U })
    hf' : Eq (X.basicOpen f') (X.toRingedSpace.basicOpen g)
    ⊢ Exists fun f => Exists fun g => And (Eq (X.basicOpen f) (X.basicOpen g)) (Me …
  -/
  exact ⟨f', g, hf', hf'.symm ▸ hg₂⟩
  /-
    🎉 no goals
  -/


/-- The prime ideal of `𝒪ₓ(U)` corresponding to a point `x : U`. -/
noncomputable def primeIdealOf (x : U) :
    PrimeSpectrum Γ(X, U) :=
  hU.isoSpec.hom.base x


theorem fromSpec_primeIdealOf (x : U) :
    hU.fromSpec.base (hU.primeIdealOf x) = x.1 := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq (hU.fromSpec.base (hU.primeIdealOf x)) ↑x
  -/
  dsimp only [IsAffineOpen.fromSpec, Subtype.coe_mk, IsAffineOpen.primeIdealOf]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp hU.isoSpec.inv U.ι).base (hU.isoSpec …
  -/
  rw [← Scheme.comp_base_apply, Iso.hom_inv_id_assoc]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq (U.ι.base x) ↑x
  -/
  rfl
  /-
    🎉 no goals
  -/


open IsLocalRing in
theorem primeIdealOf_eq_map_closedPoint (x : U) :
    hU.primeIdealOf x = (Spec.map (X.presheaf.germ _ x x.2)).base (closedPoint _) :=
  hU.isoSpec_hom_base_apply _


theorem isLocalization_stalk' (y : PrimeSpectrum Γ(X, U)) (hy : hU.fromSpec.base y ∈ U) :
    @IsLocalization.AtPrime
      (R := Γ(X, U))
      (S := X.presheaf.stalk <| hU.fromSpec.base y) _ _
      ((TopCat.Presheaf.algebra_section_stalk X.presheaf _)) y.asIdeal _ := by
  apply
    (@IsLocalization.isLocalization_iff_of_ringEquiv (R := Γ(X, U))
      (S := X.presheaf.stalk (hU.fromSpec.base y)) _ y.asIdeal.primeCompl _
      (TopCat.Presheaf.algebra_section_stalk X.presheaf ⟨hU.fromSpec.base y, hy⟩) _ _
      (asIso <| hU.fromSpec.stalkMap y).commRingCatIsoToRingEquiv).mpr
  -- Porting note: need to know what the ring is and after convert, instead of equality
  -- we get an `iff`.
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U })
    hy : Membership.mem U (hU.fromSpec.base y)
    ⊢ IsLocalization y.asIdeal.primeCompl ↑((AlgebraicGeometry.Spec (X.presheaf.ob …
  -/
  convert StructureSheaf.IsLocalization.to_stalk Γ(X, U) y using 1
  /-
    case a
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U })
    hy : Membership.mem U (hU.fromSpec.base y)
    ⊢ Iff (IsLocalization y.asIdeal.primeCompl ↑((AlgebraicGeometry.Spec (X.preshe …
  -/
  delta IsLocalization.AtPrime StructureSheaf.stalkAlgebra
  /-
    case a
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U })
    hy : Membership.mem U (hU.fromSpec.base y)
    ⊢ Iff (IsLocalization y.asIdeal.primeCompl ↑((AlgebraicGeometry.Spec (X.preshe …
  -/
  rw [iff_iff_eq]
  /-
    case a
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U })
    hy : Membership.mem U (hU.fromSpec.base y)
    ⊢ Eq (IsLocalization y.asIdeal.primeCompl ↑((AlgebraicGeometry.Spec (X.preshea …
  -/
  congr 2
  rw [RingHom.algebraMap_toAlgebra, RingEquiv.toRingHom_eq_coe,
    CategoryTheory.Iso.commRingCatIsoToRingEquiv_toRingHom, asIso_hom, ← CommRingCat.hom_comp,
    Scheme.stalkMap_germ, IsAffineOpen.fromSpec_app_self, Category.assoc, TopCat.Presheaf.germ_res]
  /-
    case a.h.e_6.h.e_i
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U })
    hy : Membership.mem U (hU.fromSpec.base y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.ΓSpecIso (X …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: I have split this into two lemmas

theorem isLocalization_stalk (x : U) :
    IsLocalization.AtPrime (X.presheaf.stalk x) (hU.primeIdealOf x).asIdeal := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : Subtype fun x => Membership.mem U x
    ⊢ IsLocalization.AtPrime (↑(X.presheaf.stalk ↑x)) (hU.primeIdealOf x).asIdeal
  -/
  rcases x with ⟨x, hx⟩
  /-
    case mk
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ IsLocalization.AtPrime (↑(X.presheaf.stalk ↑⟨x, hx⟩)) (hU.primeIdealOf ⟨x, h …
  -/
  set y := hU.primeIdealOf ⟨x, hx⟩ with hy
  /-
    case mk
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U }) := hU.primeIdealOf ⟨x, hx⟩
    hy : Eq y (hU.primeIdealOf ⟨x, hx⟩)
    ⊢ IsLocalization.AtPrime (↑(X.presheaf.stalk ↑⟨x, hx⟩)) y.asIdeal
  -/
  have : hU.fromSpec.base y = x := hy ▸ hU.fromSpec_primeIdealOf ⟨x, hx⟩
  /-
    case mk
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U }) := hU.primeIdealOf ⟨x, hx⟩
    hy : Eq y (hU.primeIdealOf ⟨x, hx⟩)
    this : Eq (hU.fromSpec.base y) x
    ⊢ IsLocalization.AtPrime (↑(X.presheaf.stalk ↑⟨x, hx⟩)) y.asIdeal
  -/
  clear_value y
  /-
    case mk
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U })
    hy : Eq y (hU.primeIdealOf ⟨x, hx⟩)
    this : Eq (hU.fromSpec.base y) x
    ⊢ IsLocalization.AtPrime (↑(X.presheaf.stalk ↑⟨x, hx⟩)) y.asIdeal
  -/
  subst this
  /-
    case mk
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    y : PrimeSpectrum ↑(X.presheaf.obj { unop := U })
    hx : Membership.mem U (hU.fromSpec.base y)
    hy : Eq y (hU.primeIdealOf ⟨hU.fromSpec.base y, hx⟩)
    ⊢ IsLocalization.AtPrime (↑(X.presheaf.stalk ↑⟨hU.fromSpec.base y, hx⟩)) y.asI …
  -/
  exact hU.isLocalization_stalk' y hx
  /-
    🎉 no goals
  -/


lemma stalkMap_injective (f : X ⟶ Y) {U : Opens Y} (hU : IsAffineOpen U) (x : X)
    (hx : f.base x ∈ U)
    (h : ∀ g, f.stalkMap x (Y.presheaf.germ U (f.base x) hx g) = 0 →
      Y.presheaf.germ U (f.base x) hx g = 0) :
    Function.Injective (f.stalkMap x) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U (f.base x)
    h : ∀ (g : ↑(Y.presheaf.obj { unop := U })), Eq ((AlgebraicGeometry.Scheme.Hom …
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  letI := Y.presheaf.algebra_section_stalk ⟨f.base x, hx⟩
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U (f.base x)
    h : ∀ (g : ↑(Y.presheaf.obj { unop := U })), Eq ((AlgebraicGeometry.Scheme.Hom …
    this : Algebra ↑(Y.presheaf.obj { unop := U }) ↑(Y.presheaf.stalk ↑⟨f.base x,  …
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  apply (hU.isLocalization_stalk ⟨f.base x, hx⟩).injective_of_map_algebraMap_zero
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
    hU : AlgebraicGeometry.IsAffineOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U (f.base x)
    h : ∀ (g : ↑(Y.presheaf.obj { unop := U })), Eq ((AlgebraicGeometry.Scheme.Hom …
    this : Algebra ↑(Y.presheaf.obj { unop := U }) ↑(Y.presheaf.stalk ↑⟨f.base x,  …
    ⊢ ∀ (x_1 : ↑(Y.presheaf.obj { unop := U })), Eq ((AlgebraicGeometry.Scheme.Hom …
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- The basic open set of a section `f` on an affine open as an `X.affineOpens`. -/
@[simps]
def _root_.AlgebraicGeometry.Scheme.affineBasicOpen
    (X : Scheme) {U : X.affineOpens} (f : Γ(X, U)) : X.affineOpens :=
  ⟨X.basicOpen f, U.prop.basicOpen f⟩


include hU in
/--
In an affine open set `U`, a family of basic open covers `U` iff the sections span `Γ(X, U)`.
See `iSup_basicOpen_of_span_eq_top` for the inverse direction without the affine-ness assumption.
-/
theorem basicOpen_union_eq_self_iff (s : Set Γ(X, U)) :
    ⨆ f : s, X.basicOpen (f : Γ(X, U)) = U ↔ Ideal.span s = ⊤ := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    s : Set ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (Eq (iSup fun f => X.basicOpen ↑f) U) (Eq (Ideal.span s) Top.top)
  -/
  trans ⋃ i : s, (PrimeSpectrum.basicOpen i.1).1 = Set.univ
  · trans
      hU.fromSpec.base ⁻¹' (⨆ f : s, X.basicOpen (f : Γ(X, U))).1 =
        hU.fromSpec.base ⁻¹' U.1
      /-
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        ⊢ Iff (Eq (iSup fun f => X.basicOpen ↑f) U) (Eq (Set.preimage (⇑hU.fromSpec.ba …
      -/
    · refine ⟨fun h => by rw [h], ?_⟩
      /-
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        ⊢ Eq (Set.preimage (⇑hU.fromSpec.base) (iSup fun f => X.basicOpen ↑f).carrier) …
      -/
      intro h
      /-
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        h : Eq (Set.preimage (⇑hU.fromSpec.base) (iSup fun f => X.basicOpen ↑f).carrie …
        ⊢ Eq (iSup fun f => X.basicOpen ↑f) U
      -/
      apply_fun Set.image hU.fromSpec.base at h
      rw [Set.image_preimage_eq_inter_range, Set.image_preimage_eq_inter_range, hU.range_fromSpec]
        at h
      /-
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        h : Eq (Inter.inter (iSup fun f => X.basicOpen ↑f).carrier ↑U) (Inter.inter U. …
        ⊢ Eq (iSup fun f => X.basicOpen ↑f) U
      -/
      simp only [Set.inter_self, Opens.carrier_eq_coe, Set.inter_eq_right] at h
      /-
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        h : HasSubset.Subset ↑U ↑(iSup fun f => X.basicOpen ↑f)
        ⊢ Eq (iSup fun f => X.basicOpen ↑f) U
      -/
      ext1
      /-
        case h
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        h : HasSubset.Subset ↑U ↑(iSup fun f => X.basicOpen ↑f)
        ⊢ Eq ↑(iSup fun f => X.basicOpen ↑f) ↑U
      -/
      refine Set.Subset.antisymm ?_ h
      /-
        case h
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        h : HasSubset.Subset ↑U ↑(iSup fun f => X.basicOpen ↑f)
        ⊢ HasSubset.Subset ↑(iSup fun f => X.basicOpen ↑f) ↑U
      -/
      simp only [Set.iUnion_subset_iff, SetCoe.forall, Opens.coe_iSup]
      /-
        case h
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        h : HasSubset.Subset ↑U ↑(iSup fun f => X.basicOpen ↑f)
        ⊢ ∀ (x : ↑(X.presheaf.obj { unop := U })), Membership.mem s x → HasSubset.Subs …
      -/
      intro x _
      /-
        case h
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        h : HasSubset.Subset ↑U ↑(iSup fun f => X.basicOpen ↑f)
        x : ↑(X.presheaf.obj { unop := U })
        h✝ : Membership.mem s x
        ⊢ HasSubset.Subset ↑(X.basicOpen x) ↑U
      -/
      exact X.basicOpen_le x
      /-
        🎉 no goals
      -/
      /-
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        ⊢ Iff (Eq (Set.preimage (⇑hU.fromSpec.base) (iSup fun f => X.basicOpen ↑f).car …
      -/
    · simp only [Opens.iSup_def, Subtype.coe_mk, Set.preimage_iUnion]
      /-
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        s : Set ↑(X.presheaf.obj { unop := U })
        ⊢ Iff (Eq (Set.iUnion fun i => Set.preimage ⇑hU.fromSpec.base ↑(X.basicOpen ↑i …
      -/
      congr! 1
        /-
          case a.h.e'_2.h
          X : AlgebraicGeometry.Scheme
          U : X.Opens
          hU : AlgebraicGeometry.IsAffineOpen U
          s : Set ↑(X.presheaf.obj { unop := U })
          e_1✝ : Eq (Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).toPre …
          ⊢ Eq (Set.iUnion fun i => Set.preimage ⇑hU.fromSpec.base ↑(X.basicOpen ↑i)) (S …
        -/
      · refine congr_arg (Set.iUnion ·) ?_
        /-
          case a.h.e'_2.h
          X : AlgebraicGeometry.Scheme
          U : X.Opens
          hU : AlgebraicGeometry.IsAffineOpen U
          s : Set ↑(X.presheaf.obj { unop := U })
          e_1✝ : Eq (Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).toPre …
          ⊢ Eq (fun i => Set.preimage ⇑hU.fromSpec.base ↑(X.basicOpen ↑i)) fun i => (Pri …
        -/
        ext1 x
        /-
          case a.h.e'_2.h.h
          X : AlgebraicGeometry.Scheme
          U : X.Opens
          hU : AlgebraicGeometry.IsAffineOpen U
          s : Set ↑(X.presheaf.obj { unop := U })
          e_1✝ : Eq (Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).toPre …
          x : ↑s
          ⊢ Eq (Set.preimage ⇑hU.fromSpec.base ↑(X.basicOpen ↑x)) (PrimeSpectrum.basicOp …
        -/
        exact congr_arg Opens.carrier (hU.fromSpec_preimage_basicOpen _)
        /-
          🎉 no goals
        -/
        /-
          case a.h.e'_3.h
          X : AlgebraicGeometry.Scheme
          U : X.Opens
          hU : AlgebraicGeometry.IsAffineOpen U
          s : Set ↑(X.presheaf.obj { unop := U })
          e_1✝ : Eq (Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := U })).toPre …
          ⊢ Eq (Set.preimage (⇑hU.fromSpec.base) U.carrier) Set.univ
        -/
      · exact congr_arg Opens.carrier hU.fromSpec_preimage_self
        /-
          🎉 no goals
        -/
    /-
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      s : Set ↑(X.presheaf.obj { unop := U })
      ⊢ Iff (Eq (Set.iUnion fun i => (PrimeSpectrum.basicOpen ↑i).carrier) Set.univ) …
    -/
  · simp only [Opens.carrier_eq_coe, PrimeSpectrum.basicOpen_eq_zeroLocus_compl]
    rw [← Set.compl_iInter, Set.compl_univ_iff, ← PrimeSpectrum.zeroLocus_iUnion, ←
      PrimeSpectrum.zeroLocus_empty_iff_eq_top, PrimeSpectrum.zeroLocus_span]
    /-
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      s : Set ↑(X.presheaf.obj { unop := U })
      ⊢ Iff (Eq (PrimeSpectrum.zeroLocus (Set.iUnion fun i => Singleton.singleton ↑i …
    -/
    simp only [Set.iUnion_singleton_eq_range, Subtype.range_val_subtype, Set.setOf_mem_eq]
    /-
      🎉 no goals
    -/


include hU in
theorem self_le_basicOpen_union_iff (s : Set Γ(X, U)) :
    (U ≤ ⨆ f : s, X.basicOpen f.1) ↔ Ideal.span s = ⊤ := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    s : Set ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (LE.le U (iSup fun f => X.basicOpen ↑f)) (Eq (Ideal.span s) Top.top)
  -/
  rw [← hU.basicOpen_union_eq_self_iff, @comm _ Eq]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    s : Set ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (LE.le U (iSup fun f => X.basicOpen ↑f)) (Eq U (iSup fun f => X.basicOpe …
  -/
  refine ⟨fun h => le_antisymm h ?_, le_of_eq⟩
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    s : Set ↑(X.presheaf.obj { unop := U })
    h : LE.le U (iSup fun f => X.basicOpen ↑f)
    ⊢ LE.le (iSup fun f => X.basicOpen ↑f) U
  -/
  simp only [iSup_le_iff, SetCoe.forall]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    s : Set ↑(X.presheaf.obj { unop := U })
    h : LE.le U (iSup fun f => X.basicOpen ↑f)
    ⊢ ∀ (x : ↑(X.presheaf.obj { unop := U })), Membership.mem s x → LE.le (X.basic …
  -/
  intro x _
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    s : Set ↑(X.presheaf.obj { unop := U })
    h : LE.le U (iSup fun f => X.basicOpen ↑f)
    x : ↑(X.presheaf.obj { unop := U })
    h✝ : Membership.mem s x
    ⊢ LE.le (X.basicOpen x) U
  -/
  exact X.basicOpen_le x
  /-
    🎉 no goals
  -/


open _root_.PrimeSpectrum in
/-- The restriction of `Spec.map f` to a basic open `D(r)` is isomorphic to `Spec.map` of the
localization of `f` away from `r`. -/
noncomputable
def SpecMapRestrictBasicOpenIso {R S : CommRingCat} (f : R ⟶ S) (r : R) :
    Arrow.mk (Spec.map f ∣_ (PrimeSpectrum.basicOpen r)) ≅
      Arrow.mk (Spec.map <| CommRingCat.ofHom (Localization.awayMap f.hom r)) := by
  letI e₁ : Localization.Away r ≃ₐ[R] Γ(Spec R, basicOpen r) :=
    IsLocalization.algEquiv (Submonoid.powers r) _ _
  letI e₂ : Localization.Away (f r) ≃ₐ[S] Γ(Spec S, basicOpen (f r)) :=
    IsLocalization.algEquiv (Submonoid.powers (f r)) _ _
  /-
    R S : CommRingCat
    f : Quiver.Hom R S
    r : ↑R
    e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
    e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (AlgebraicGeometry.morphismRestr …
  -/
  refine Arrow.isoMk ?_ ?_ ?_
  · exact (Spec (.of S)).isoOfEq (comap_basicOpen _ _) ≪≫
      (IsAffineOpen.Spec_basicOpen (f r)).isoSpec ≪≫ Scheme.Spec.mapIso e₂.toCommRingCatIso.op
    /-
      case refine_2
      R S : CommRingCat
      f : Quiver.Hom R S
      r : ↑R
      e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
      e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
      ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (AlgebraicGeometry.morphismRestr …
    -/
  · exact (IsAffineOpen.Spec_basicOpen r).isoSpec ≪≫ Scheme.Spec.mapIso e₁.toCommRingCatIso.op
    /-
      🎉 no goals
    -/
  · have := AlgebraicGeometry.IsOpenImmersion.of_isLocalization
      (S := (Localization.Away r)) r
    /-
      case refine_3
      R S : CommRingCat
      f : Quiver.Hom R S
      r : ↑R
      e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
      e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicGeometry.Spec (CommRingCa …
    -/
    rw [← cancel_mono (Spec.map (CommRingCat.ofHom (algebraMap R (Localization.Away r))))]
    simp only [Arrow.mk_left, Arrow.mk_right, Functor.id_obj, Scheme.isoOfEq_rfl, Iso.refl_trans,
      Iso.trans_hom, Functor.mapIso_hom, Iso.op_hom, Iso.symm_hom,
      Scheme.Spec_map, Quiver.Hom.unop_op, Arrow.mk_hom, Category.assoc,
      ← Spec.map_comp]
    show _ ≫ Spec.map (CommRingCat.ofHom
        ((e₂.toRingHom.comp (Localization.awayMap f.hom r)).comp (algebraMap R _)))
      = _ ≫ _ ≫ Spec.map (CommRingCat.ofHom (e₁.toRingHom.comp (algebraMap R _)))
    /-
      case refine_3
      R S : CommRingCat
      f : Quiver.Hom R S
      r : ↑R
      e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
      e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ⋯.isoSpec.hom (AlgebraicGeometry.Spec …
    -/
    rw [RingHom.comp_assoc]
    /-
      case refine_3
      R S : CommRingCat
      f : Quiver.Hom R S
      r : ↑R
      e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
      e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ⋯.isoSpec.hom (AlgebraicGeometry.Spec …
    -/
    conv => enter [1,2,1,1,2]; tactic => exact IsLocalization.map_comp _
    /-
      case refine_3
      R S : CommRingCat
      f : Quiver.Hom R S
      r : ↑R
      e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
      e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ⋯.isoSpec.hom (AlgebraicGeometry.Spec …
    -/
    rw [← RingHom.comp_assoc]
    /-
      case refine_3
      R S : CommRingCat
      f : Quiver.Hom R S
      r : ↑R
      e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
      e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ⋯.isoSpec.hom (AlgebraicGeometry.Spec …
    -/
    conv => enter [1,2,1,1,1]; tactic => exact e₂.toAlgHom.comp_algebraMap
    /-
      case refine_3
      R S : CommRingCat
      f : Quiver.Hom R S
      r : ↑R
      e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
      e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ⋯.isoSpec.hom (AlgebraicGeometry.Spec …
    -/
    conv => enter [2,2,2,1,1]; tactic => exact e₁.toAlgHom.comp_algebraMap
    show _ ≫ Spec.map (f ≫ (Scheme.ΓSpecIso S).inv ≫
      (Spec S).presheaf.map (homOfLE le_top).op) =
      Spec.map f ∣_ basicOpen r ≫ _ ≫ Spec.map ((Scheme.ΓSpecIso R).inv ≫
      (Spec R).presheaf.map (homOfLE le_top).op)
    simp only [IsAffineOpen.isoSpec_hom, homOfLE_leOfHom, Spec.map_comp, Category.assoc,
      Scheme.Opens.toSpecΓ_SpecMap_map_assoc, Scheme.Opens.toSpecΓ_top, Scheme.homOfLE_ι_assoc,
      morphismRestrict_ι_assoc]
    simp only [← SpecMap_ΓSpecIso_hom, ← Spec.map_comp, Category.assoc, Iso.inv_hom_id,
      Category.comp_id, Category.id_comp]
    /-
      case refine_3
      R S : CommRingCat
      f : Quiver.Hom R S
      r : ↑R
      e₁ : AlgEquiv (↑R) (Localization.Away r) ↑((AlgebraicGeometry.Spec R).presheaf …
      e₂ : AlgEquiv (↑S) (Localization.Away (f.hom r)) ↑((AlgebraicGeometry.Spec S). …
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Opens.ι (Pr …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma stalkMap_injective_of_isAffine {X Y : Scheme} (f : X ⟶ Y) [IsAffine Y] (x : X)
    (h : ∀ g, f.stalkMap x (Y.presheaf.Γgerm (f.base x) g) = 0 →
      Y.presheaf.Γgerm (f.base x) g = 0) :
    Function.Injective (f.stalkMap x) :=
  (isAffineOpen_top Y).stalkMap_injective f x trivial h


/--
Given a spanning set of `Γ(X, U)`, the corresponding basic open sets cover `U`.
See `IsAffineOpen.basicOpen_union_eq_self_iff` for the inverse direction for affine open sets.
-/
lemma iSup_basicOpen_of_span_eq_top {X : Scheme} (U) (s : Set Γ(X, U))
    (hs : Ideal.span s = ⊤) : (⨆ i ∈ s, X.basicOpen i) = U := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : Set ↑(X.presheaf.obj { unop := U })
    hs : Eq (Ideal.span s) Top.top
    ⊢ Eq (iSup fun i => iSup fun h => X.basicOpen i) U
  -/
  apply le_antisymm
    /-
      case a
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      ⊢ LE.le (iSup fun i => iSup fun h => X.basicOpen i) U
    -/
  · rw [iSup₂_le_iff]
    /-
      case a
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      ⊢ ∀ (i : ↑(X.presheaf.obj { unop := U })), Membership.mem s i → LE.le (X.basic …
    -/
    exact fun i _ ↦ X.basicOpen_le i
    /-
      🎉 no goals
    -/
    /-
      case a
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      ⊢ LE.le U (iSup fun i => iSup fun h => X.basicOpen i)
    -/
  · intro x hx
    /-
      case a
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (↑U) x
      ⊢ Membership.mem (↑(iSup fun i => iSup fun h => X.basicOpen i)) x
    -/
    obtain ⟨_, ⟨V, hV, rfl⟩, hxV, hVU⟩ := (isBasis_affine_open X).exists_subset_of_mem_open hx U.2
    /-
      case a.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (↑U) x
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hxV : Membership.mem (↑V) x
      hVU : HasSubset.Subset ↑V ↑U
      ⊢ Membership.mem (↑(iSup fun i => iSup fun h => X.basicOpen i)) x
    -/
    refine SetLike.mem_of_subset ?_ hxV
    rw [← (hV.basicOpen_union_eq_self_iff (X.presheaf.map (homOfLE hVU).op '' s)).mpr
      (by rw [← Ideal.map_span, hs, Ideal.map_top])]
    simp only [Opens.iSup_mk, Opens.carrier_eq_coe, Set.iUnion_coe_set, Set.mem_image,
      Set.iUnion_exists, Set.biUnion_and', Set.iUnion_iUnion_eq_right, Scheme.basicOpen_res,
      Opens.coe_inf, Opens.coe_mk, Set.iUnion_subset_iff]
    exact fun i hi ↦ (Set.inter_subset_right.trans
      (Set.subset_iUnion₂ (s := fun x _ ↦ (X.basicOpen x : Set X)) i hi))


/-- Let `P` be a predicate on the affine open sets of `X` satisfying
1. If `P` holds on `U`, then `P` holds on the basic open set of every section on `U`.
2. If `P` holds for a family of basic open sets covering `U`, then `P` holds for `U`.
3. There exists an affine open cover of `X` each satisfying `P`.

Then `P` holds for every affine open of `X`.

This is also known as the **Affine communication lemma** in [*The rising sea*][RisingSea]. -/
@[elab_as_elim]
theorem of_affine_open_cover {X : Scheme} {P : X.affineOpens → Prop}
    {ι} (U : ι → X.affineOpens) (iSup_U : (⨆ i, U i : X.Opens) = ⊤)
    (V : X.affineOpens)
    (basicOpen : ∀ (U : X.affineOpens) (f : Γ(X, U)), P U → P (X.affineBasicOpen f))
    (openCover :
      ∀ (U : X.affineOpens) (s : Finset (Γ(X, U)))
        (_ : Ideal.span (s : Set (Γ(X, U))) = ⊤),
        (∀ f : s, P (X.affineBasicOpen f.1)) → P U)
    (hU : ∀ i, P (U i)) : P V := by
  classical
  have : ∀ (x : V.1), ∃ f : Γ(X, V), ↑x ∈ X.basicOpen f ∧ P (X.affineBasicOpen f) := by
    intro x
    obtain ⟨i, hi⟩ := Opens.mem_iSup.mp (show x.1 ∈ (⨆ i, U i : X.Opens) from iSup_U ▸ trivial)
    obtain ⟨f, g, e, hf⟩ := exists_basicOpen_le_affine_inter V.prop (U i).prop x ⟨x.prop, hi⟩
    refine ⟨f, hf, ?_⟩
    convert basicOpen _ g (hU i) using 1
    ext1
    exact e
  choose f hf₁ hf₂ using this
  suffices Ideal.span (Set.range f) = ⊤ by
    obtain ⟨t, ht₁, ht₂⟩ := (Ideal.span_eq_top_iff_finite _).mp this
    apply openCover V t ht₂
    rintro ⟨i, hi⟩
    obtain ⟨x, rfl⟩ := ht₁ hi
    exact hf₂ x
  rw [← V.prop.self_le_basicOpen_union_iff]
  intro x hx
  rw [iSup_range', SetLike.mem_coe, Opens.mem_iSup]
  exact ⟨_, hf₁ ⟨x, hx⟩⟩


/-- On a locally ringed space `X`, the preimage of the zero locus of the prime spectrum
of `Γ(X, ⊤)` under `toΓSpecFun` agrees with the associated zero locus on `X`. -/
lemma Scheme.toΓSpec_preimage_zeroLocus_eq {X : Scheme.{u}} (s : Set Γ(X, ⊤)) :
    X.toSpecΓ.base ⁻¹' PrimeSpectrum.zeroLocus s = X.zeroLocus s :=
  LocallyRingedSpace.toΓSpec_preimage_zeroLocus_eq s


/-- If `X` is affine, the image of the zero locus of global sections of `X` under `toΓSpecFun`
is the zero locus in terms of the prime spectrum of `Γ(X, ⊤)`. -/
lemma Scheme.toΓSpec_image_zeroLocus_eq_of_isAffine {X : Scheme.{u}} [IsAffine X]
    (s : Set Γ(X, ⊤)) :
    X.isoSpec.hom.base '' X.zeroLocus s = PrimeSpectrum.zeroLocus s := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    ⊢ Eq (Set.image (⇑X.isoSpec.hom.base) (X.zeroLocus s)) (PrimeSpectrum.zeroLocu …
  -/
  erw [← X.toΓSpec_preimage_zeroLocus_eq, Set.image_preimage_eq]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    ⊢ Function.Surjective ⇑X.isoSpec.hom.base
  -/
  exact (bijective_of_isIso X.isoSpec.hom.base).surjective
  /-
    🎉 no goals
  -/


/-- If `X` is an affine scheme, every closed set of `X` is the zero locus
of a set of global sections. -/
lemma Scheme.eq_zeroLocus_of_isClosed_of_isAffine (X : Scheme.{u}) [IsAffine X] (s : Set X) :
    IsClosed s ↔ ∃ I : Ideal (Γ(X, ⊤)), s = X.zeroLocus (I : Set Γ(X, ⊤)) := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    s : Set ↑↑X.toPresheafedSpace
    ⊢ Iff (IsClosed s) (Exists fun I => Eq s (X.zeroLocus ↑I))
  -/
  refine ⟨fun hs ↦ ?_, ?_⟩
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      s : Set ↑↑X.toPresheafedSpace
      hs : IsClosed s
      ⊢ Exists fun I => Eq s (X.zeroLocus ↑I)
    -/
  · let Z : Set (Spec <| Γ(X, ⊤)) := X.toΓSpecFun '' s
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      s : Set ↑↑X.toPresheafedSpace
      hs : IsClosed s
      Z : Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).toPres …
      ⊢ Exists fun I => Eq s (X.zeroLocus ↑I)
    -/
    have hZ : IsClosed Z := (X.isoSpec.hom.homeomorph).isClosedMap _ hs
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      s : Set ↑↑X.toPresheafedSpace
      hs : IsClosed s
      Z : Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).toPres …
      hZ : IsClosed Z
      ⊢ Exists fun I => Eq s (X.zeroLocus ↑I)
    -/
    obtain ⟨I, (hI : Z = _)⟩ := (PrimeSpectrum.isClosed_iff_zeroLocus_ideal _).mp hZ
    /-
      case refine_1.intro
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      s : Set ↑↑X.toPresheafedSpace
      hs : IsClosed s
      Z : Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).toPres …
      hZ : IsClosed Z
      I : Ideal ↑(X.presheaf.obj { unop := Top.top })
      hI : Eq Z (PrimeSpectrum.zeroLocus ↑I)
      ⊢ Exists fun I => Eq s (X.zeroLocus ↑I)
    -/
    use I
    /-
      case h
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      s : Set ↑↑X.toPresheafedSpace
      hs : IsClosed s
      Z : Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).toPres …
      hZ : IsClosed Z
      I : Ideal ↑(X.presheaf.obj { unop := Top.top })
      hI : Eq Z (PrimeSpectrum.zeroLocus ↑I)
      ⊢ Eq s (X.zeroLocus ↑I)
    -/
    simp only [← Scheme.toΓSpec_preimage_zeroLocus_eq, ← hI, Z]
    /-
      case h
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      s : Set ↑↑X.toPresheafedSpace
      hs : IsClosed s
      Z : Set ↑↑(AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })).toPres …
      hZ : IsClosed Z
      I : Ideal ↑(X.presheaf.obj { unop := Top.top })
      hI : Eq Z (PrimeSpectrum.zeroLocus ↑I)
      ⊢ Eq s (Set.preimage (⇑X.toSpecΓ.base) (Set.image X.toΓSpecFun s))
    -/
    erw [Set.preimage_image_eq _ (bijective_of_isIso X.isoSpec.hom.base).injective]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      s : Set ↑↑X.toPresheafedSpace
      ⊢ (Exists fun I => Eq s (X.zeroLocus ↑I)) → IsClosed s
    -/
  · rintro ⟨I, rfl⟩
    /-
      case refine_2.intro
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine X
      I : Ideal ↑(X.presheaf.obj { unop := Top.top })
      ⊢ IsClosed (X.zeroLocus ↑I)
    -/
    exact zeroLocus_isClosed X I.carrier
    /-
      🎉 no goals
    -/


/-- If `X ⟶ Spec A` is a morphism of schemes, then `Spec` of `A ⧸ specTargetImage f`
is the scheme-theoretic image of `f`. For this quotient as an object of `CommRingCat` see
`specTargetImage` below. -/
def specTargetImageIdeal (f : X ⟶ Spec A) : Ideal A :=
  (RingHom.ker <| (((ΓSpec.adjunction).homEquiv X (op A)).symm f).unop.hom)


/-- If `X ⟶ Spec A` is a morphism of schemes, then `Spec` of `specTargetImage f` is the
scheme-theoretic image of `f` and `f` factors as
`specTargetImageFactorization f ≫ Spec.map (specTargetImageRingHom f)`
(see `specTargetImageFactorization_comp`). -/
def specTargetImage (f : X ⟶ Spec A) : CommRingCat :=
  CommRingCat.of (A ⧸ specTargetImageIdeal f)


/-- If `f : X ⟶ Spec A` is a morphism of schemes, then `f` factors via
the inclusion of `Spec (specTargetImage f)` into `X`. -/
def specTargetImageFactorization (f : X ⟶ Spec A) : X ⟶ Spec (specTargetImage f) :=
  (ΓSpec.adjunction).homEquiv X (op <| specTargetImage f) (Opposite.op
    (CommRingCat.ofHom (RingHom.kerLift _)))


/-- If `f : X ⟶ Spec A` is a morphism of schemes, the induced morphism on spectra of
`specTargetImageRingHom f` is the inclusion of the scheme-theoretic image of `f` into `Spec A`. -/
def specTargetImageRingHom (f : X ⟶ Spec A) : A ⟶ specTargetImage f :=
  CommRingCat.ofHom (Ideal.Quotient.mk (specTargetImageIdeal f))


lemma specTargetImageRingHom_surjective : Function.Surjective (specTargetImageRingHom f) :=
  Ideal.Quotient.mk_surjective


lemma specTargetImageFactorization_app_injective :
    Function.Injective <| (specTargetImageFactorization f).appTop := by
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop (AlgebraicGeometry. …
  -/
  let φ : A ⟶ Γ(X, ⊤) := (((ΓSpec.adjunction).homEquiv X (op A)).symm f).unop
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop (AlgebraicGeometry. …
  -/
  let φ' : specTargetImage f ⟶ Scheme.Γ.obj (op X) := CommRingCat.ofHom (RingHom.kerLift φ.hom)
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop (AlgebraicGeometry. …
  -/
  show Function.Injective <| ((ΓSpec.adjunction.homEquiv X _) φ'.op).appTop
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop ((AlgebraicGeometry …
  -/
  rw [ΓSpec_adjunction_homEquiv_eq]
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Function.Injective ⇑(CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.S …
  -/
  apply (RingHom.kerLift_injective φ.hom).comp
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.ΓSpecIso (AlgebraicGeometry.sp …
  -/
  exact ((ConcreteCategory.isIso_iff_bijective (Scheme.ΓSpecIso _).hom).mp inferInstance).injective
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma specTargetImageFactorization_comp :
    specTargetImageFactorization f ≫ Spec.map (specTargetImageRingHom f) = f := by
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.specTargetImageFac …
  -/
  let φ : A ⟶ Γ(X, ⊤) := (((ΓSpec.adjunction).homEquiv X (op A)).symm f).unop
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.specTargetImageFac …
  -/
  let φ' : specTargetImage f ⟶ Scheme.Γ.obj (op X) := CommRingCat.ofHom (RingHom.kerLift φ.hom)
  /-
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.specTargetImageFac …
  -/
  apply ((ΓSpec.adjunction).homEquiv X (op A)).symm.injective
  /-
    case a
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Eq ((AlgebraicGeometry.ΓSpec.adjunction.homEquiv X { unop := A }).symm (Cate …
  -/
  apply Opposite.unop_injective
  /-
    case a.a
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Eq (Opposite.unop ((AlgebraicGeometry.ΓSpec.adjunction.homEquiv X { unop :=  …
  -/
  rw [Adjunction.homEquiv_naturality_left_symm, Adjunction.homEquiv_counit]
  /-
    case a.a
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Eq (Opposite.unop (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Sch …
  -/
  change (_ ≫ _) ≫ _ = φ
  /-
    case a.a
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [← Spec_Γ_naturality]
  /-
    case a.a
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp { …
  -/
  rw [Category.assoc]
  /-
    case a.a
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := Ideal.Quotient.mk (Algebraic …
  -/
  erw [ΓSpecIso_inv_ΓSpec_adjunction_homEquiv φ']
  /-
    case a.a
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := Ideal.Quotient.mk (Algebraic …
  -/
  ext a
  /-
    case a.a.hf.a
    X : AlgebraicGeometry.Scheme
    A : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec A)
    φ : Quiver.Hom A (X.presheaf.obj { unop := Top.top }) := ((AlgebraicGeometry.Γ …
    φ' : Quiver.Hom (AlgebraicGeometry.specTargetImage f) (AlgebraicGeometry.Schem …
    a : ↑A
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp { hom := Ideal.Quotient.mk (Algebrai …
  -/
  apply RingHom.kerLift_mk
  /-
    🎉 no goals
  -/


/-- The scheme-theoretic image of a morphism `f : X ⟶ Y` with affine target.
`f` factors as `affineTargetImageFactorization f ≫ affineTargetImageInclusion f`
(see `affineTargetImageFactorization_comp`). -/
def affineTargetImage (f : X ⟶ Y) : Scheme.{u} :=
  Spec <| specTargetImage (f ≫ Y.isoSpec.hom)


instance : IsAffine (affineTargetImage f) := inferInstanceAs <| IsAffine <| Spec _


/-- The inclusion of the scheme-theoretic image of a morphism with affine target. -/
def affineTargetImageInclusion (f : X ⟶ Y) : affineTargetImage f ⟶ Y :=
  Spec.map (specTargetImageRingHom (f ≫ Y.isoSpec.hom)) ≫ Y.isoSpec.inv


lemma affineTargetImageInclusion_app_surjective :
    Function.Surjective <| (affineTargetImageInclusion f).appTop := by
  simp only [Scheme.comp_coeBase, Opens.map_comp_obj, Opens.map_top, Scheme.comp_app,
    CommRingCat.hom_comp, affineTargetImageInclusion, RingHom.coe_comp]
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    ⊢ Function.Surjective (Function.comp ⇑(AlgebraicGeometry.Scheme.Hom.app (Algeb …
  -/
  apply Function.Surjective.comp
  · haveI : (toMorphismProperty (fun f ↦ Function.Surjective f)).RespectsIso := by
      rw [← toMorphismProperty_respectsIso_iff]
      exact surjective_respectsIso
    exact (MorphismProperty.arrow_mk_iso_iff
      (toMorphismProperty (fun f ↦ Function.Surjective f))
      (arrowIsoΓSpecOfIsAffine (specTargetImageRingHom (f ≫ Y.isoSpec.hom))).symm).mpr <|
        specTargetImageRingHom_surjective (f ≫ Y.isoSpec.hom)
    /-
      case hf
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.app Y.isoSpec.inv Top.top …
    -/
  · apply Function.Bijective.surjective
    /-
      case hf.hf
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      ⊢ Function.Bijective ⇑(AlgebraicGeometry.Scheme.Hom.app Y.isoSpec.inv Top.top) …
    -/
    exact ConcreteCategory.bijective_of_isIso (Scheme.Hom.app Y.isoSpec.inv ⊤)
    /-
      🎉 no goals
    -/


/-- The induced morphism from `X` to the scheme-theoretic image
of a morphism `f : X ⟶ Y` with affine target. -/
def affineTargetImageFactorization (f : X ⟶ Y) : X ⟶ affineTargetImage f :=
  specTargetImageFactorization (f ≫ Y.isoSpec.hom)


lemma affineTargetImageFactorization_app_injective :
    Function.Injective <| (affineTargetImageFactorization f).appTop :=
  specTargetImageFactorization_app_injective (f ≫ Y.isoSpec.hom)


@[reassoc (attr := simp)]
lemma affineTargetImageFactorization_comp :
    affineTargetImageFactorization f ≫ affineTargetImageInclusion f = f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.affineTargetImageF …
  -/
  simp [affineTargetImageFactorization, affineTargetImageInclusion]
  /-
    🎉 no goals
  -/


/-- Variant of `AlgebraicGeometry.localRingHom_comp_stalkIso` for `Spec.map`. -/
@[elementwise]
lemma Scheme.localRingHom_comp_stalkIso {R S : CommRingCat.{u}} (f : R ⟶ S) (p : PrimeSpectrum S) :
    (StructureSheaf.stalkIso R (PrimeSpectrum.comap f.hom p)).hom ≫
      (CommRingCat.ofHom <| Localization.localRingHom
        (PrimeSpectrum.comap f.hom p).asIdeal p.asIdeal f.hom rfl) ≫
      (StructureSheaf.stalkIso S p).inv = (Spec.map f).stalkMap p :=
  AlgebraicGeometry.localRingHom_comp_stalkIso f p


/-- Given a morphism of rings `f : R ⟶ S`, the stalk map of `Spec S ⟶ Spec R` at
a prime of `S` is isomorphic to the localized ring homomorphism. -/
def Scheme.arrowStalkMapSpecIso {R S : CommRingCat.{u}} (f : R ⟶ S) (p : PrimeSpectrum S) :
    Arrow.mk ((Spec.map f).stalkMap p) ≅ Arrow.mk (CommRingCat.ofHom <| Localization.localRingHom
      (PrimeSpectrum.comap f.hom p).asIdeal p.asIdeal f.hom rfl) := Arrow.isoMk
  (StructureSheaf.stalkIso R (PrimeSpectrum.comap f.hom p))
  (StructureSheaf.stalkIso S p) <| by
    /-
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑S
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.sta …
    -/
    rw [← Scheme.localRingHom_comp_stalkIso]
    /-
      R S : CommRingCat
      f : Quiver.Hom R S
      p : PrimeSpectrum ↑S
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.sta …
    -/
    simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-21"), nolint defLemma]
alias isAffineAffineScheme := isAffine_affineScheme

@[deprecated (since := "2024-06-21"), nolint defLemma]
alias SpecIsAffine := isAffine_Spec

@[deprecated (since := "2024-06-21")]
alias isAffineOfIso := isAffine_of_isIso

@[deprecated (since := "2024-06-21")]
alias rangeIsAffineOpenOfOpenImmersion := isAffineOpen_opensRange

@[deprecated (since := "2024-06-21")]
alias topIsAffineOpen := isAffineOpen_top

@[deprecated (since := "2024-06-21"), nolint defLemma]
alias Scheme.affineCoverIsAffine := Scheme.isAffine_affineCover

@[deprecated (since := "2024-06-21"), nolint defLemma]
alias Scheme.affineBasisCoverIsAffine := Scheme.isAffine_affineBasisCover

@[deprecated (since := "2024-06-21")]
alias IsAffineOpen.fromSpec_range := IsAffineOpen.range_fromSpec

@[deprecated (since := "2024-06-21")]
alias IsAffineOpen.imageIsOpenImmersion := IsAffineOpen.image_of_isOpenImmersion

@[deprecated (since := "2024-06-21"), nolint defLemma]
alias Scheme.quasi_compact_of_affine := Scheme.compactSpace_of_isAffine

@[deprecated (since := "2024-06-21")]
alias IsAffineOpen.fromSpec_base_preimage := IsAffineOpen.fromSpec_preimage_self

@[deprecated (since := "2024-06-21")]
alias IsAffineOpen.fromSpec_map_basicOpen' := IsAffineOpen.fromSpec_preimage_basicOpen'

@[deprecated (since := "2024-06-21")]
alias IsAffineOpen.fromSpec_map_basicOpen := IsAffineOpen.fromSpec_preimage_basicOpen

@[deprecated (since := "2024-06-21")]
alias IsAffineOpen.opensFunctor_map_basicOpen := IsAffineOpen.fromSpec_image_basicOpen

@[deprecated (since := "2024-06-21")]
alias IsAffineOpen.basicOpenIsAffine := IsAffineOpen.basicOpen

@[deprecated (since := "2024-06-21")]
alias IsAffineOpen.mapRestrictBasicOpen := IsAffineOpen.ι_basicOpen_preimage


