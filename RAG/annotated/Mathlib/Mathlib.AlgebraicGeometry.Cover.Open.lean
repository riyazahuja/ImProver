/-- An open cover of a scheme `X` is a cover where all component maps are open immersions. -/
abbrev OpenCover (X : Scheme.{u}) : Type _ := Cover.{v} @IsOpenImmersion X


@[deprecated (since := "2024-06-23")] alias OpenCover.Covers := Cover.covers

@[deprecated (since := "2024-11-06")] alias OpenCover.IsOpen := Cover.map_prop


instance (i : 𝒰.J) : IsOpenImmersion (𝒰.map i) := 𝒰.map_prop i


/-- The affine cover of a scheme. -/
def affineCover (X : Scheme.{u}) : OpenCover X where
  J := X
  obj x := Spec (X.local_affine x).choose_spec.choose
  map x :=
    ⟨(X.local_affine x).choose_spec.choose_spec.some.inv ≫ X.toLocallyRingedSpace.ofRestrict _⟩
  f x := x
  covers := by
    /-
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : X✝.OpenCover
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (Set.range ⇑((fun x => { toHom …
    -/
    intro x
    /-
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : X✝.OpenCover
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun x => { toHom_1 := CategoryTheory.CategorySt …
    -/
    erw [TopCat.coe_comp] -- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : X✝.OpenCover
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (Set.range (Function.comp ⇑(AlgebraicGeometry.LocallyRingedSp …
    -/
    rw [Set.range_comp, Set.range_eq_univ.mpr, Set.image_univ]
      /-
        X✝ Y Z : AlgebraicGeometry.Scheme
        𝒰 : X✝.OpenCover
        f : Quiver.Hom X✝ Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        X : AlgebraicGeometry.Scheme
        x : ↑↑X.toPresheafedSpace
        ⊢ Membership.mem (Set.range ⇑(AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom …
      -/
    · erw [Subtype.range_coe_subtype]
      /-
        X✝ Y Z : AlgebraicGeometry.Scheme
        𝒰 : X✝.OpenCover
        f : Quiver.Hom X✝ Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        X : AlgebraicGeometry.Scheme
        x : ↑↑X.toPresheafedSpace
        ⊢ Membership.mem (setOf fun x_1 => Membership.mem ⋯.choose.obj x_1) x
      -/
      exact (X.local_affine x).choose.2
      /-
        🎉 no goals
      -/
    /-
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : X✝.OpenCover
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom ⋯.som …
    -/
    rw [← TopCat.epi_iff_surjective]
    /-
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : X✝.OpenCover
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      ⊢ CategoryTheory.Epi (AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom ⋯.some. …
    -/
    change Epi ((SheafedSpace.forget _).map (LocallyRingedSpace.forgetToSheafedSpace.map _))
    /-
      X✝ Y Z : AlgebraicGeometry.Scheme
      𝒰 : X✝.OpenCover
      f : Quiver.Hom X✝ Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      X : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      ⊢ CategoryTheory.Epi ((AlgebraicGeometry.SheafedSpace.forget CommRingCat).map  …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance : Inhabited X.OpenCover :=
  ⟨X.affineCover⟩


theorem OpenCover.iSup_opensRange {X : Scheme.{u}} (𝒰 : X.OpenCover) :
    ⨆ i, (𝒰.map i).opensRange = ⊤ :=
                  /-
                    X : AlgebraicGeometry.Scheme
                    𝒰 : X.OpenCover
                    ⊢ Eq ↑(iSup fun i => AlgebraicGeometry.Scheme.Hom.opensRange (𝒰.map i)) ↑Top.top
                  -/
  Opens.ext <| by rw [Opens.coe_iSup]; exact 𝒰.iUnion_range
                                       /-
                                         🎉 no goals
                                       -/


/-- Every open cover of a quasi-compact scheme can be refined into a finite subcover.
-/
@[simps! obj map]
def OpenCover.finiteSubcover {X : Scheme.{u}} (𝒰 : OpenCover X) [H : CompactSpace X] :
    OpenCover X := by
  have :=
    @CompactSpace.elim_nhds_subcover _ _ H (fun x : X => Set.range (𝒰.map (𝒰.f x)).base)
      fun x => (IsOpenImmersion.isOpen_range (𝒰.map (𝒰.f x))).mem_nhds (𝒰.covers x)
  /-
    X✝ Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X✝.OpenCover
    f : Quiver.Hom X✝ Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    H : CompactSpace ↑↑X.toPresheafedSpace
    this : Exists fun t => Eq (Set.iUnion fun x => Set.iUnion fun h => (fun x => S …
    ⊢ X.OpenCover
  -/
  let t := this.choose
  have h : ∀ x : X, ∃ y : t, x ∈ Set.range (𝒰.map (𝒰.f y)).base := by
    intro x
    have h' : x ∈ (⊤ : Set X) := trivial
    rw [← Classical.choose_spec this, Set.mem_iUnion] at h'
    rcases h' with ⟨y, _, ⟨hy, rfl⟩, hy'⟩
    exact ⟨⟨y, hy⟩, hy'⟩
  exact
    { J := t
      obj := fun x => 𝒰.obj (𝒰.f x.1)
      map := fun x => 𝒰.map (𝒰.f x.1)
      f := fun x => (h x).choose
      covers := fun x => (h x).choose_spec }


instance [H : CompactSpace X] : Fintype 𝒰.finiteSubcover.J := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    H : CompactSpace ↑↑X.toPresheafedSpace
    ⊢ Fintype 𝒰.finiteSubcover.J
  -/
  delta OpenCover.finiteSubcover; infer_instance
                                  /-
                                    🎉 no goals
                                  -/


theorem OpenCover.compactSpace {X : Scheme.{u}} (𝒰 : X.OpenCover) [Finite 𝒰.J]
    [H : ∀ i, CompactSpace (𝒰.obj i)] : CompactSpace X := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    H : ∀ (i : 𝒰.J), CompactSpace ↑↑(𝒰.obj i).toPresheafedSpace
    ⊢ CompactSpace ↑↑X.toPresheafedSpace
  -/
  cases nonempty_fintype 𝒰.J
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    H : ∀ (i : 𝒰.J), CompactSpace ↑↑(𝒰.obj i).toPresheafedSpace
    val✝ : Fintype 𝒰.J
    ⊢ CompactSpace ↑↑X.toPresheafedSpace
  -/
  rw [← isCompact_univ_iff, ← 𝒰.iUnion_range]
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    H : ∀ (i : 𝒰.J), CompactSpace ↑↑(𝒰.obj i).toPresheafedSpace
    val✝ : Fintype 𝒰.J
    ⊢ IsCompact (Set.iUnion fun i => Set.range ⇑(𝒰.map i).base)
  -/
  apply isCompact_iUnion
  /-
    case intro.h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    H : ∀ (i : 𝒰.J), CompactSpace ↑↑(𝒰.obj i).toPresheafedSpace
    val✝ : Fintype 𝒰.J
    ⊢ ∀ (i : 𝒰.J), IsCompact (Set.range ⇑(𝒰.map i).base)
  -/
  intro i
  /-
    case intro.h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    H : ∀ (i : 𝒰.J), CompactSpace ↑↑(𝒰.obj i).toPresheafedSpace
    val✝ : Fintype 𝒰.J
    i : 𝒰.J
    ⊢ IsCompact (Set.range ⇑(𝒰.map i).base)
  -/
  rw [isCompact_iff_compactSpace]
  exact
    @Homeomorph.compactSpace _ _ _ _ (H i)
      (TopCat.homeoOfIso
        (asIso
          (IsOpenImmersion.isoOfRangeEq (𝒰.map i)
            (X.ofRestrict (Opens.isOpenEmbedding ⟨_, (𝒰.map_prop i).base_open.isOpen_range⟩))
            Subtype.range_coe.symm).hom.base))

/--
An affine open cover of `X` consists of a family of open immersions into `X` from
spectra of rings.
-/
abbrev AffineOpenCover (X : Scheme.{u}) : Type _ :=
  AffineCover.{v} @IsOpenImmersion X


instance {X : Scheme.{u}} (𝒰 : X.AffineOpenCover) (j : 𝒰.J) : IsOpenImmersion (𝒰.map j) :=
  𝒰.map_prop j


/-- The open cover associated to an affine open cover. -/
@[simps! J obj map f covers]
def openCover {X : Scheme.{u}} (𝒰 : X.AffineOpenCover) : X.OpenCover :=
  AffineCover.cover 𝒰


/-- A choice of an affine open cover of a scheme. -/
@[simps]
def affineOpenCover (X : Scheme.{u}) : X.AffineOpenCover where
  J := X.affineCover.J
  map := X.affineCover.map
  f := X.affineCover.f
  covers := X.affineCover.covers


@[simp]
lemma openCover_affineOpenCover (X : Scheme.{u}) : X.affineOpenCover.openCover = X.affineCover :=
  rfl


/-- Given any open cover `𝓤`, this is an affine open cover which refines it.
The morphism in the category of open covers which proves that this is indeed a refinement, see
`AlgebraicGeometry.Scheme.OpenCover.fromAffineRefinement`.
-/
def OpenCover.affineRefinement {X : Scheme.{u}} (𝓤 : X.OpenCover) : X.AffineOpenCover where
  J := (𝓤.bind fun j => (𝓤.obj j).affineCover).J
  map := (𝓤.bind fun j => (𝓤.obj j).affineCover).map
  f := (𝓤.bind fun j => (𝓤.obj j).affineCover).f
  covers := (𝓤.bind fun j => (𝓤.obj j).affineCover).covers


/-- The pullback of the affine refinement is the pullback of the affine cover. -/
def OpenCover.pullbackCoverAffineRefinementObjIso (f : X ⟶ Y) (𝒰 : Y.OpenCover) (i) :
    (𝒰.affineRefinement.openCover.pullbackCover f).obj i ≅
      ((𝒰.obj i.1).affineCover.pullbackCover (𝒰.pullbackHom f i.1)).obj i.2 :=
  pullbackSymmetry _ _ ≪≫ (pullbackRightPullbackFstIso _ _ _).symm ≪≫
    pullbackSymmetry _ _ ≪≫ asIso (pullback.map _ _ _ _ (pullbackSymmetry _ _).hom (𝟙 _) (𝟙 _)
          /-
            X Y Z : AlgebraicGeometry.Scheme
            𝒰✝ : X.OpenCover
            f✝ : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            inst✝ : ∀ (x : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
            f : Quiver.Hom X Y
            𝒰 : Y.OpenCover
            i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰.affineRefinement.openCover …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
          -/
          /-
            🎉 no goals
          -/
      (by simp [Cover.pullbackHom]) (by simp))
                                        /-
                                          🎉 no goals
                                        -/


@[reassoc]
lemma OpenCover.pullbackCoverAffineRefinementObjIso_inv_map (f : X ⟶ Y) (𝒰 : Y.OpenCover) (i) :
    (𝒰.pullbackCoverAffineRefinementObjIso f i).inv ≫
      (𝒰.affineRefinement.openCover.pullbackCover f).map i =
      ((𝒰.obj i.1).affineCover.pullbackCover (𝒰.pullbackHom f i.1)).map i.2 ≫
        (𝒰.pullbackCover f).map i.1 := by
  simp only [Cover.pullbackCover_obj, AffineCover.cover_obj, AffineCover.cover_map,
    pullbackCoverAffineRefinementObjIso, Iso.trans_inv, asIso_inv, Iso.symm_inv, Category.assoc,
    Cover.pullbackCover_map, pullbackSymmetry_inv_comp_fst, IsIso.inv_comp_eq, limit.lift_π_assoc,
    id_eq, PullbackCone.mk_pt, cospan_left, PullbackCone.mk_π_app, pullbackSymmetry_hom_comp_fst]
  convert pullbackSymmetry_inv_comp_snd_assoc
    ((𝒰.obj i.1).affineCover.map i.2) (pullback.fst _ _) _ using 2
  /-
    case h.e'_2.h.h.e'_7
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰.affineRefinement.openCover …
    e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.Limits.p …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackRightP …
  -/
  exact pullbackRightPullbackFstIso_hom_snd _ _ _
  /-
    🎉 no goals
  -/


@[reassoc]
lemma OpenCover.pullbackCoverAffineRefinementObjIso_inv_pullbackHom
    (f : X ⟶ Y) (𝒰 : Y.OpenCover) (i) :
    (𝒰.pullbackCoverAffineRefinementObjIso f i).inv ≫
      𝒰.affineRefinement.openCover.pullbackHom f i =
      (𝒰.obj i.1).affineCover.pullbackHom (𝒰.pullbackHom f i.1) i.2 := by
  simp only [Cover.pullbackCover_obj, Cover.pullbackHom, AffineCover.cover_obj,
    AffineOpenCover.openCover_map, pullbackCoverAffineRefinementObjIso, Iso.trans_inv, asIso_inv,
    Iso.symm_inv, Category.assoc, pullbackSymmetry_inv_comp_snd, IsIso.inv_comp_eq, limit.lift_π,
    id_eq, PullbackCone.mk_pt, PullbackCone.mk_π_app, Category.comp_id]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰.affineRefinement.openCover …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackSymmet …
  -/
  convert pullbackSymmetry_inv_comp_fst ((𝒰.obj i.1).affineCover.map i.2) (pullback.fst _ _)
  /-
    case h.e'_2.h.h.e'_7.h
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰.affineRefinement.openCover …
    e_1✝ : Eq (Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.Limits.p …
    e_5✝ : Eq (𝒰.affineRefinement.openCover.obj i) ((𝒰.obj i.fst).affineCover.obj  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackRightP …
  -/
  exact pullbackRightPullbackFstIso_hom_fst _ _ _
  /-
    🎉 no goals
  -/


/-- A family of elements spanning the unit ideal of `R` gives a affine open cover of `Spec R`. -/
@[simps]
noncomputable
def affineOpenCoverOfSpanRangeEqTop {R : CommRingCat} {ι : Type*} (s : ι → R)
    (hs : Ideal.span (Set.range s) = ⊤) : (Spec R).AffineOpenCover where
  J := ι
  obj i := .of (Localization.Away (s i))
  map i := Spec.map (CommRingCat.ofHom (algebraMap R (Localization.Away (s i))))
  f x := by
    have : ∃ i, s i ∉ x.asIdeal := by
      by_contra! h; apply x.2.ne_top; rwa [← top_le_iff, ← hs, Ideal.span_le, Set.range_subset_iff]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      R : CommRingCat
      ι : Type u_1
      s : ι → ↑R
      hs : Eq (Ideal.span (Set.range s)) Top.top
      x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
      this : Exists fun i => Not (Membership.mem x.asIdeal (s i))
      ⊢ ι
    -/
    exact this.choose
    /-
      🎉 no goals
    -/
  covers x := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      R : CommRingCat
      ι : Type u_1
      s : ι → ↑R
      hs : Eq (Ideal.span (Set.range s)) Top.top
      x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun i => AlgebraicGeometry.Spec.map (CommRingCa …
    -/
    generalize_proofs H
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      R : CommRingCat
      ι : Type u_1
      s : ι → ↑R
      hs : Eq (Ideal.span (Set.range s)) Top.top
      x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
      H : ∀ (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), Exists fun i => No …
      ⊢ Membership.mem (Set.range ⇑((fun i => AlgebraicGeometry.Spec.map (CommRingCa …
    -/
    let i := (H x).choose
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      R : CommRingCat
      ι : Type u_1
      s : ι → ↑R
      hs : Eq (Ideal.span (Set.range s)) Top.top
      x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
      H : ∀ (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), Exists fun i => No …
      i : ι := ⋯.choose
      ⊢ Membership.mem (Set.range ⇑((fun i => AlgebraicGeometry.Spec.map (CommRingCa …
    -/
    have := PrimeSpectrum.localization_away_comap_range (Localization.Away (s i)) (s i)
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      R : CommRingCat
      ι : Type u_1
      s : ι → ↑R
      hs : Eq (Ideal.span (Set.range s)) Top.top
      x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
      H : ∀ (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), Exists fun i => No …
      i : ι := ⋯.choose
      this : Eq (Set.range ⇑(PrimeSpectrum.comap (algebraMap (↑R) (Localization.Away …
      ⊢ Membership.mem (Set.range ⇑((fun i => AlgebraicGeometry.Spec.map (CommRingCa …
    -/
    exact (eq_iff_iff.mp congr(x ∈ $this)).mpr (H x).choose_spec
    /-
      🎉 no goals
    -/


/-- Given any open cover `𝓤`, this is an affine open cover which refines it. -/
def OpenCover.fromAffineRefinement {X : Scheme.{u}} (𝓤 : X.OpenCover) :
    𝓤.affineRefinement.openCover ⟶ 𝓤 where
  idx j := j.fst
  app j := (𝓤.obj j.fst).affineCover.map _


/-- If two global sections agree after restriction to each member of an open cover, then
they agree globally. -/
lemma OpenCover.ext_elem {X : Scheme.{u}} {U : X.Opens} (f g : Γ(X, U)) (𝒰 : X.OpenCover)
    (h : ∀ i : 𝒰.J, (𝒰.map i).app U f = (𝒰.map i).app U g) : f = g := by
  fapply TopCat.Sheaf.eq_of_locally_eq' X.sheaf
    (fun i ↦ (𝒰.map (𝒰.f i)).opensRange ⊓ U) _ (fun _ ↦ homOfLE inf_le_right)
    /-
      case hcover
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      f g : ↑(X.presheaf.obj { unop := U })
      𝒰 : X.OpenCover
      h : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom f) ((A …
      ⊢ LE.le U (iSup fun i => Min.min (AlgebraicGeometry.Scheme.Hom.opensRange (𝒰.m …
    -/
  · intro x hx
    simp only [Opens.iSup_mk, Opens.carrier_eq_coe, Opens.coe_inf, Hom.coe_opensRange, Opens.coe_mk,
      Set.mem_iUnion, Set.mem_inter_iff, Set.mem_range, SetLike.mem_coe, exists_and_right]
    /-
      case hcover
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      f g : ↑(X.presheaf.obj { unop := U })
      𝒰 : X.OpenCover
      h : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom f) ((A …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (↑U) x
      ⊢ And (Exists fun x_1 => Exists fun y => Eq ((𝒰.map (𝒰.f x_1)).base y) x) (Mem …
    -/
    refine ⟨?_, hx⟩
    /-
      case hcover
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      f g : ↑(X.presheaf.obj { unop := U })
      𝒰 : X.OpenCover
      h : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom f) ((A …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (↑U) x
      ⊢ Exists fun x_1 => Exists fun y => Eq ((𝒰.map (𝒰.f x_1)).base y) x
    -/
    simpa using ⟨_, 𝒰.covers x⟩
    /-
      🎉 no goals
    -/
    /-
      case h
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      f g : ↑(X.presheaf.obj { unop := U })
      𝒰 : X.OpenCover
      h : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom f) ((A …
      ⊢ ∀ (i : ↑↑X.toPresheafedSpace), Eq ((X.sheaf.val.map (CategoryTheory.homOfLE  …
    -/
  · intro x
    /-
      case h
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      f g : ↑(X.presheaf.obj { unop := U })
      𝒰 : X.OpenCover
      h : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom f) ((A …
      x : ↑↑X.toPresheafedSpace
      ⊢ Eq ((X.sheaf.val.map (CategoryTheory.homOfLE ⋯).op) f) ((X.sheaf.val.map (Ca …
    -/
    replace h := h (𝒰.f x)
    /-
      case h
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      f g : ↑(X.presheaf.obj { unop := U })
      𝒰 : X.OpenCover
      x : ↑↑X.toPresheafedSpace
      h : Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map (𝒰.f x)) U).hom f) ((Algebrai …
      ⊢ Eq ((X.sheaf.val.map (CategoryTheory.homOfLE ⋯).op) f) ((X.sheaf.val.map (Ca …
    -/
    rw [← IsOpenImmersion.map_ΓIso_inv] at h
    /-
      case h
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      f g : ↑(X.presheaf.obj { unop := U })
      𝒰 : X.OpenCover
      x : ↑↑X.toPresheafedSpace
      h : Eq ((CategoryTheory.CategoryStruct.comp (X.presheaf.map (CategoryTheory.ho …
      ⊢ Eq ((X.sheaf.val.map (CategoryTheory.homOfLE ⋯).op) f) ((X.sheaf.val.map (Ca …
    -/
    exact (IsOpenImmersion.ΓIso (𝒰.map (𝒰.f x)) U).commRingCatIsoToRingEquiv.symm.injective h
    /-
      🎉 no goals
    -/


/-- If the restriction of a global section to each member of an open cover is zero, then it is
globally zero. -/
lemma zero_of_zero_cover {X : Scheme.{u}} {U : X.Opens} (s : Γ(X, U)) (𝒰 : X.OpenCover)
    (h : ∀ i : 𝒰.J, (𝒰.map i).app U s = 0) : s = 0 :=
                             /-
                               X : AlgebraicGeometry.Scheme
                               U : X.Opens
                               s : ↑(X.presheaf.obj { unop := U })
                               𝒰 : X.OpenCover
                               h : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom s) 0
                               i : 𝒰.J
                               ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom s) ((AlgebraicGeometr …
                             -/
  𝒰.ext_elem s 0 (fun i ↦ by rw [map_zero]; exact h i)
                                            /-
                                              🎉 no goals
                                            -/


/-- If a global section is nilpotent on each member of a finite open cover, then `f` is
nilpotent. -/
lemma isNilpotent_of_isNilpotent_cover {X : Scheme.{u}} {U : X.Opens} (s : Γ(X, U))
    (𝒰 : X.OpenCover) [Finite 𝒰.J] (h : ∀ i : 𝒰.J, IsNilpotent ((𝒰.map i).app U s)) :
    IsNilpotent s := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    h : ∀ (i : 𝒰.J), IsNilpotent ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).h …
    ⊢ IsNilpotent s
  -/
  choose fn hfn using h
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    fn : 𝒰.J → Nat
    hfn : ∀ (i : 𝒰.J), Eq (HPow.hPow ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i)  …
    ⊢ IsNilpotent s
  -/
  have : Fintype 𝒰.J := Fintype.ofFinite 𝒰.J
  /- the maximum of all `fn i` (exists, because `𝒰.J` is finite) -/
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    fn : 𝒰.J → Nat
    hfn : ∀ (i : 𝒰.J), Eq (HPow.hPow ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i)  …
    this : Fintype 𝒰.J
    ⊢ IsNilpotent s
  -/
  let N : ℕ := Finset.sup Finset.univ fn
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    fn : 𝒰.J → Nat
    hfn : ∀ (i : 𝒰.J), Eq (HPow.hPow ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i)  …
    this : Fintype 𝒰.J
    N : Nat := Finset.univ.sup fn
    ⊢ IsNilpotent s
  -/
  have hfnleN (i : 𝒰.J) : fn i ≤ N := Finset.le_sup (Finset.mem_univ i)
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    fn : 𝒰.J → Nat
    hfn : ∀ (i : 𝒰.J), Eq (HPow.hPow ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i)  …
    this : Fintype 𝒰.J
    N : Nat := Finset.univ.sup fn
    hfnleN : ∀ (i : 𝒰.J), LE.le (fn i) N
    ⊢ IsNilpotent s
  -/
  use N
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    fn : 𝒰.J → Nat
    hfn : ∀ (i : 𝒰.J), Eq (HPow.hPow ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i)  …
    this : Fintype 𝒰.J
    N : Nat := Finset.univ.sup fn
    hfnleN : ∀ (i : 𝒰.J), LE.le (fn i) N
    ⊢ Eq (HPow.hPow s N) 0
  -/
  apply zero_of_zero_cover (𝒰 := 𝒰)
  /-
    case h.h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    fn : 𝒰.J → Nat
    hfn : ∀ (i : 𝒰.J), Eq (HPow.hPow ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i)  …
    this : Fintype 𝒰.J
    N : Nat := Finset.univ.sup fn
    hfnleN : ∀ (i : 𝒰.J), LE.le (fn i) N
    ⊢ ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom (HPow.hP …
  -/
  on_goal 1 => intro i; simp only [map_pow]
  -- This closes both remaining goals at once.
  /-
    case h.h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    𝒰 : X.OpenCover
    inst✝ : Finite 𝒰.J
    fn : 𝒰.J → Nat
    hfn : ∀ (i : 𝒰.J), Eq (HPow.hPow ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i)  …
    this : Fintype 𝒰.J
    N : Nat := Finset.univ.sup fn
    hfnleN : ∀ (i : 𝒰.J), LE.le (fn i) N
    i : 𝒰.J
    ⊢ Eq (HPow.hPow ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) U).hom s) N) 0
  -/
  exact pow_eq_zero_of_le (hfnleN i) (hfn i)
  /-
    🎉 no goals
  -/


/-- The basic open sets form an affine open cover of `Spec R`. -/
def affineBasisCoverOfAffine (R : CommRingCat.{u}) : OpenCover (Spec R) where
  J := R
  obj r := Spec (CommRingCat.of <| Localization.Away r)
  map r := Spec.map (CommRingCat.ofHom (algebraMap R (Localization.Away r)))
  f _ := 1
  covers r := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      R : CommRingCat
      r : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
      ⊢ Membership.mem (Set.range ⇑((fun r => AlgebraicGeometry.Spec.map (CommRingCa …
    -/
    rw [Set.range_eq_univ.mpr ((TopCat.epi_iff_surjective _).mp _)]
      /-
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        R : CommRingCat
        r : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
        ⊢ Membership.mem Set.univ r
      -/
    · exact trivial
      /-
        🎉 no goals
      -/
    · -- Porting note: need more hand holding here because Lean knows that
      -- `CommRing.ofHom ...` is iso, but without `ofHom` Lean does not know what to do
      /-
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        R : CommRingCat
        r : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
        ⊢ CategoryTheory.Epi ((fun r => AlgebraicGeometry.Spec.map (CommRingCat.ofHom  …
      -/
      change Epi (Spec.map (CommRingCat.ofHom (algebraMap _ _))).base
      /-
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (x : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        R : CommRingCat
        r : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
        ⊢ CategoryTheory.Epi (AlgebraicGeometry.Spec.map (CommRingCat.ofHom (algebraMa …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
  map_prop x := AlgebraicGeometry.Scheme.basic_open_isOpenImmersion x


/-- We may bind the basic open sets of an open affine cover to form an affine cover that is also
a basis. -/
def affineBasisCover (X : Scheme.{u}) : OpenCover X :=
  X.affineCover.bind fun _ => affineBasisCoverOfAffine _


/-- The coordinate ring of a component in the `affine_basis_cover`. -/
def affineBasisCoverRing (X : Scheme.{u}) (i : X.affineBasisCover.J) : CommRingCat :=
  CommRingCat.of <| @Localization.Away (X.local_affine i.1).choose_spec.choose _ i.2


theorem affineBasisCover_obj (X : Scheme.{u}) (i : X.affineBasisCover.J) :
    X.affineBasisCover.obj i = Spec (X.affineBasisCoverRing i) :=
  rfl


theorem affineBasisCover_map_range (X : Scheme.{u}) (x : X)
    (r : (X.local_affine x).choose_spec.choose) :
    Set.range (X.affineBasisCover.map ⟨x, r⟩).base =
      (X.affineCover.map x).base '' (PrimeSpectrum.basicOpen r).1 := by
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    r : ↑⋯.choose
    ⊢ Eq (Set.range ⇑(X.affineBasisCover.map ⟨x, r⟩).base) (Set.image (⇑(X.affineC …
  -/
  erw [coe_comp, Set.range_comp]
  -- Porting note: `congr` fails to see the goal is comparing image of the same function
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    r : ↑⋯.choose
    ⊢ Eq (Set.image (⇑(AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (Algebraic …
  -/
  refine congr_arg (_ '' ·) ?_
  /-
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    r : ↑⋯.choose
    ⊢ Eq (Set.range ⇑(AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (AlgebraicG …
  -/
  exact (PrimeSpectrum.localization_away_comap_range (Localization.Away r) r : _)
  /-
    🎉 no goals
  -/


theorem affineBasisCover_is_basis (X : Scheme.{u}) :
    TopologicalSpace.IsTopologicalBasis
      {x : Set X |
        ∃ a : X.affineBasisCover.J, x = Set.range (X.affineBasisCover.map a).base} := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ TopologicalSpace.IsTopologicalBasis (setOf fun x => Exists fun a => Eq x (Se …
  -/
  apply TopologicalSpace.isTopologicalBasis_of_isOpen_of_nhds
    /-
      case h_open
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (u : Set ↑↑X.toPresheafedSpace), Membership.mem (setOf fun x => Exists fun …
    -/
  · rintro _ ⟨a, rfl⟩
    /-
      case h_open.intro
      X : AlgebraicGeometry.Scheme
      a : X.affineBasisCover.J
      ⊢ IsOpen (Set.range ⇑(X.affineBasisCover.map a).base)
    -/
    exact IsOpenImmersion.isOpen_range (X.affineBasisCover.map a)
    /-
      🎉 no goals
    -/
    /-
      case h_nhds
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (a : ↑↑X.toPresheafedSpace) (u : Set ↑↑X.toPresheafedSpace), Membership.me …
    -/
  · rintro a U haU hU
    /-
      case h_nhds
      X : AlgebraicGeometry.Scheme
      a : ↑↑X.toPresheafedSpace
      U : Set ↑↑X.toPresheafedSpace
      haU : Membership.mem U a
      hU : IsOpen U
      ⊢ Exists fun v => And (Membership.mem (setOf fun x => Exists fun a => Eq x (Se …
    -/
    rcases X.affineCover.covers a with ⟨x, e⟩
    /-
      case h_nhds.intro
      X : AlgebraicGeometry.Scheme
      a : ↑↑X.toPresheafedSpace
      U : Set ↑↑X.toPresheafedSpace
      haU : Membership.mem U a
      hU : IsOpen U
      x : ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace
      e : Eq ((X.affineCover.map (X.affineCover.f a)).base x) a
      ⊢ Exists fun v => And (Membership.mem (setOf fun x => Exists fun a => Eq x (Se …
    -/
    let U' := (X.affineCover.map (X.affineCover.f a)).base ⁻¹' U
    /-
      case h_nhds.intro
      X : AlgebraicGeometry.Scheme
      a : ↑↑X.toPresheafedSpace
      U : Set ↑↑X.toPresheafedSpace
      haU : Membership.mem U a
      hU : IsOpen U
      x : ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace
      e : Eq ((X.affineCover.map (X.affineCover.f a)).base x) a
      U' : Set ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace := Set.pr …
      ⊢ Exists fun v => And (Membership.mem (setOf fun x => Exists fun a => Eq x (Se …
    -/
    have hxU' : x ∈ U' := by rw [← e] at haU; exact haU
    rcases PrimeSpectrum.isBasis_basic_opens.exists_subset_of_mem_open hxU'
        ((X.affineCover.map (X.affineCover.f a)).base.continuous_toFun.isOpen_preimage _
          hU) with
      ⟨_, ⟨_, ⟨s, rfl⟩, rfl⟩, hxV, hVU⟩
    /-
      case h_nhds.intro.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      a : ↑↑X.toPresheafedSpace
      U : Set ↑↑X.toPresheafedSpace
      haU : Membership.mem U a
      hU : IsOpen U
      x : ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace
      e : Eq ((X.affineCover.map (X.affineCover.f a)).base x) a
      U' : Set ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace := Set.pr …
      hxU' : Membership.mem U' x
      s : ↑⋯.choose
      hxV : Membership.mem (↑(PrimeSpectrum.basicOpen s)) x
      hVU : HasSubset.Subset (↑(PrimeSpectrum.basicOpen s)) U'
      ⊢ Exists fun v => And (Membership.mem (setOf fun x => Exists fun a => Eq x (Se …
    -/
    refine ⟨_, ⟨⟨_, s⟩, rfl⟩, ?_, ?_⟩ <;> rw [affineBasisCover_map_range]
      /-
        case h_nhds.intro.intro.intro.intro.intro.intro.intro.refine_1
        X : AlgebraicGeometry.Scheme
        a : ↑↑X.toPresheafedSpace
        U : Set ↑↑X.toPresheafedSpace
        haU : Membership.mem U a
        hU : IsOpen U
        x : ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace
        e : Eq ((X.affineCover.map (X.affineCover.f a)).base x) a
        U' : Set ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace := Set.pr …
        hxU' : Membership.mem U' x
        s : ↑⋯.choose
        hxV : Membership.mem (↑(PrimeSpectrum.basicOpen s)) x
        hVU : HasSubset.Subset (↑(PrimeSpectrum.basicOpen s)) U'
        ⊢ Membership.mem (Set.image (⇑(X.affineCover.map (X.affineCover.f a)).base) (P …
      -/
    · exact ⟨x, hxV, e⟩
      /-
        🎉 no goals
      -/
      /-
        case h_nhds.intro.intro.intro.intro.intro.intro.intro.refine_2
        X : AlgebraicGeometry.Scheme
        a : ↑↑X.toPresheafedSpace
        U : Set ↑↑X.toPresheafedSpace
        haU : Membership.mem U a
        hU : IsOpen U
        x : ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace
        e : Eq ((X.affineCover.map (X.affineCover.f a)).base x) a
        U' : Set ↑↑(X.affineCover.obj (X.affineCover.f a)).toPresheafedSpace := Set.pr …
        hxU' : Membership.mem U' x
        s : ↑⋯.choose
        hxV : Membership.mem (↑(PrimeSpectrum.basicOpen s)) x
        hVU : HasSubset.Subset (↑(PrimeSpectrum.basicOpen s)) U'
        ⊢ HasSubset.Subset (Set.image (⇑(X.affineCover.map (X.affineCover.f a)).base)  …
      -/
    · rw [Set.image_subset_iff]; exact hVU
                                 /-
                                   🎉 no goals
                                 -/


