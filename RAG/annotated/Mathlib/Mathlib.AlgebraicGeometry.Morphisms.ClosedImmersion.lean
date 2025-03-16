/-- A morphism of schemes `X ⟶ Y` is a closed immersion if the underlying
topological map is a closed embedding and the induced stalk maps are surjective. -/
@[mk_iff]
class IsClosedImmersion {X Y : Scheme} (f : X ⟶ Y) : Prop where
  base_closed : IsClosedEmbedding f.base
  surj_on_stalks : ∀ x, Function.Surjective (f.stalkMap x)


lemma Scheme.Hom.isClosedEmbedding {X Y : Scheme} (f : X.Hom Y)
    [IsClosedImmersion f] : IsClosedEmbedding f.base :=
  IsClosedImmersion.base_closed


@[deprecated (since := "2024-10-24")]
alias isClosedEmbedding := Scheme.Hom.isClosedEmbedding

@[deprecated (since := "2024-10-20")]
alias closedEmbedding := isClosedEmbedding


lemma eq_inf : @IsClosedImmersion = (topologically IsClosedEmbedding) ⊓
    stalkwise (fun f ↦ Function.Surjective f) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsClosedImmersion) (Min.min (AlgebraicGeometry.topolo …
  -/
  ext X Y f
  /-
    case h.h.h.a
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.IsClosedImmersion f) (Min.min (AlgebraicGeometry.topo …
  -/
  rw [isClosedImmersion_iff]
  /-
    case h.h.h.a
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (And (Topology.IsClosedEmbedding ⇑f.base) (∀ (x : ↑↑X.toPresheafedSpace) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma iff_isPreimmersion {X Y : Scheme} {f : X ⟶ Y} :
    IsClosedImmersion f ↔ IsPreimmersion f ∧ IsClosed (Set.range f.base) := by
  rw [isClosedImmersion_iff, isPreimmersion_iff, ← surjectiveOnStalks_iff, and_comm, and_assoc,
    isClosedEmbedding_iff]


lemma of_isPreimmersion {X Y : Scheme} (f : X ⟶ Y) [IsPreimmersion f]
    (hf : IsClosed (Set.range f.base)) : IsClosedImmersion f :=
  iff_isPreimmersion.mpr ⟨‹_›, hf⟩


instance (priority := 900) {X Y : Scheme} (f : X ⟶ Y) [IsClosedImmersion f] : IsPreimmersion f :=
  (iff_isPreimmersion.mp ‹_›).1


/-- Isomorphisms are closed immersions. -/
instance {X Y : Scheme} (f : X ⟶ Y) [IsIso f] : IsClosedImmersion f where
  base_closed := Homeomorph.isClosedEmbedding <| TopCat.homeoOfIso (asIso f.base)
  surj_on_stalks := fun _ ↦ (ConcreteCategory.bijective_of_isIso (C := CommRingCat) _).2


instance : MorphismProperty.IsMultiplicative @IsClosedImmersion where
  id_mem _ := inferInstance
  comp_mem {X Y Z} f g hf hg := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsClosedImmersion f
      hg : AlgebraicGeometry.IsClosedImmersion g
      ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.comp f g)
    -/
    refine ⟨hg.base_closed.comp hf.base_closed, fun x ↦ ?_⟩
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsClosedImmersion f
      hg : AlgebraicGeometry.IsClosedImmersion g
      x : ↑↑X.toPresheafedSpace
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory. …
    -/
    rw [Scheme.stalkMap_comp]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.IsClosedImmersion f
      hg : AlgebraicGeometry.IsClosedImmersion g
      x : ↑↑X.toPresheafedSpace
      ⊢ Function.Surjective ⇑(CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
    -/
    exact (hf.surj_on_stalks x).comp (hg.surj_on_stalks (f.base x))
    /-
      🎉 no goals
    -/


/-- Composition of closed immersions is a closed immersion. -/
instance comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsClosedImmersion f]
    [IsClosedImmersion g] : IsClosedImmersion (f ≫ g) :=
  MorphismProperty.IsStableUnderComposition.comp_mem f g inferInstance inferInstance


/-- Composition with an isomorphism preserves closed immersions. -/
instance respectsIso : MorphismProperty.RespectsIso @IsClosedImmersion := by
  /-
    ⊢ CategoryTheory.MorphismProperty.RespectsIso @AlgebraicGeometry.IsClosedImmer …
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  apply MorphismProperty.RespectsIso.mk <;> intro X Y Z e f hf <;> infer_instance
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- Given two commutative rings `R S : CommRingCat` and a surjective morphism
`f : R ⟶ S`, the induced scheme morphism `specObj S ⟶ specObj R` is a
closed immersion. -/
theorem spec_of_surjective {R S : CommRingCat} (f : R ⟶ S) (h : Function.Surjective f) :
    IsClosedImmersion (Spec.map f) where
  base_closed := PrimeSpectrum.isClosedEmbedding_comap_of_surjective _ _ h
  surj_on_stalks x := by
    haveI : (RingHom.toMorphismProperty (fun f ↦ Function.Surjective f)).RespectsIso := by
      rw [← RingHom.toMorphismProperty_respectsIso_iff]
      exact RingHom.surjective_respectsIso
    apply (MorphismProperty.arrow_mk_iso_iff
      (RingHom.toMorphismProperty (fun f ↦ Function.Surjective f))
      (Scheme.arrowStalkMapSpecIso f x)).mpr
    /-
      R S : CommRingCat
      f : Quiver.Hom R S
      h : Function.Surjective ⇑f.hom
      x : ↑↑(AlgebraicGeometry.Spec S).toPresheafedSpace
      this : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] f => Fu …
      ⊢ RingHom.toMorphismProperty (fun {R S} [CommRing R] [CommRing S] f => Functio …
    -/
    exact RingHom.surjective_localRingHom_of_surjective f.hom h x.asIdeal
    /-
      🎉 no goals
    -/


/-- For any ideal `I` in a commutative ring `R`, the quotient map `specObj R ⟶ specObj (R ⧸ I)`
is a closed immersion. -/
instance spec_of_quotient_mk {R : CommRingCat.{u}} (I : Ideal R) :
    IsClosedImmersion (Spec.map (CommRingCat.ofHom (Ideal.Quotient.mk I))) :=
  spec_of_surjective _ Ideal.Quotient.mk_surjective


/-- Any morphism between affine schemes that is surjective on global sections is a
closed immersion. -/
lemma of_surjective_of_isAffine {X Y : Scheme} [IsAffine X] [IsAffine Y] (f : X ⟶ Y)
    (h : Function.Surjective (f.appTop)) : IsClosedImmersion f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    h : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    ⊢ AlgebraicGeometry.IsClosedImmersion f
  -/
  rw [MorphismProperty.arrow_mk_iso_iff @IsClosedImmersion (arrowIsoSpecΓOfIsAffine f)]
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    h : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    ⊢ AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.Spec.map (AlgebraicGe …
  -/
  apply spec_of_surjective
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    h : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
  -/
  exact h
  /-
    🎉 no goals
  -/


/--
If `f ≫ g` and `g` are closed immersions, then `f` is a closed immersion.
Also see `IsClosedImmersion.of_comp` for the general version
where `g` is only required to be separated.
-/
theorem of_comp_isClosedImmersion {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsClosedImmersion g]
    [IsClosedImmersion (f ≫ g)] : IsClosedImmersion f where
  base_closed := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsClosedImmersion g
      inst✝ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.com …
      ⊢ Topology.IsClosedEmbedding ⇑f.base
    -/
    have h := (f ≫ g).isClosedEmbedding
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsClosedImmersion g
      inst✝ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.com …
      h : Topology.IsClosedEmbedding ⇑(CategoryTheory.CategoryStruct.comp f g).base
      ⊢ Topology.IsClosedEmbedding ⇑f.base
    -/
    simp only [Scheme.comp_coeBase, TopCat.coe_comp] at h
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsClosedImmersion g
      inst✝ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.com …
      h : Topology.IsClosedEmbedding (Function.comp ⇑g.base ⇑f.base)
      ⊢ Topology.IsClosedEmbedding ⇑f.base
    -/
    refine .of_continuous_injective_isClosedMap (Scheme.Hom.continuous f) h.injective.of_comp ?_
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsClosedImmersion g
      inst✝ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.com …
      h : Topology.IsClosedEmbedding (Function.comp ⇑g.base ⇑f.base)
      ⊢ IsClosedMap ⇑f.base
    -/
    intro Z hZ
    rw [IsClosedEmbedding.isClosed_iff_image_isClosed g.isClosedEmbedding,
      ← Set.image_comp]
    /-
      X Y Z✝ : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z✝
      inst✝¹ : AlgebraicGeometry.IsClosedImmersion g
      inst✝ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.com …
      h : Topology.IsClosedEmbedding (Function.comp ⇑g.base ⇑f.base)
      Z : Set ↑↑X.toPresheafedSpace
      hZ : IsClosed Z
      ⊢ IsClosed (Set.image (Function.comp ⇑g.base ⇑f.base) Z)
    -/
    exact h.isClosedMap _ hZ
    /-
      🎉 no goals
    -/
  surj_on_stalks x := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsClosedImmersion g
      inst✝ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.com …
      x : ↑↑X.toPresheafedSpace
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    have h := (f ≫ g).stalkMap_surjective x
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsClosedImmersion g
      inst✝ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.com …
      x : ↑↑X.toPresheafedSpace
      h : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheor …
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    simp_rw [Scheme.stalkMap_comp] at h
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.IsClosedImmersion g
      inst✝ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.com …
      x : ↑↑X.toPresheafedSpace
      h : Function.Surjective ⇑(CategoryTheory.CategoryStruct.comp (AlgebraicGeometr …
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    exact Function.Surjective.of_comp h
    /-
      🎉 no goals
    -/


instance Spec_map_residue {X : Scheme.{u}} (x) : IsClosedImmersion (Spec.map (X.residue x)) :=
  IsClosedImmersion.spec_of_surjective (X.residue x)
    Ideal.Quotient.mk_surjective


instance {X Y : Scheme} (f : X ⟶ Y) [IsClosedImmersion f] : QuasiCompact f where
  isCompact_preimage _ _ hU' := base_closed.isCompact_preimage hU'


/-- If `f : X ⟶ Y` is a morphism of schemes with quasi-compact source and affine target, `f`
has a closed image and `f` induces an injection on global sections, then
`f` is surjective. -/
lemma surjective_of_isClosed_range_of_injective [CompactSpace X]
    (hfcl : IsClosed (Set.range f.base)) (hfinj : Function.Injective (f.appTop)) :
    Function.Surjective f.base := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    ⊢ Function.Surjective ⇑f.base
  -/
  obtain ⟨I, hI⟩ := (Scheme.eq_zeroLocus_of_isClosed_of_isAffine Y (Set.range f.base)).mp hfcl
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    ⊢ Function.Surjective ⇑f.base
  -/
  let 𝒰 : X.OpenCover := X.affineCover.finiteSubcover
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    ⊢ Function.Surjective ⇑f.base
  -/
  haveI (i : 𝒰.J) : IsAffine (𝒰.obj i) := Scheme.isAffine_affineCover X _
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ Function.Surjective ⇑f.base
  -/
  apply Set.range_eq_univ.mp
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ Eq (Set.range ⇑f.base) Set.univ
  -/
  apply hI ▸ (Scheme.zeroLocus_eq_top_iff_subset_nilradical _).mpr
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ HasSubset.Subset ↑I ↑(nilradical ↑(Y.presheaf.obj { unop := Top.top }))
  -/
  intro s hs
  simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
    Submodule.mem_toAddSubmonoid, SetLike.mem_coe, mem_nilradical, ← IsNilpotent.map_iff hfinj]
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hs : Membership.mem (↑I) s
    ⊢ IsNilpotent ((AlgebraicGeometry.Scheme.Hom.appTop f).hom s)
  -/
  refine Scheme.isNilpotent_of_isNilpotent_cover _ 𝒰 (fun i ↦ ?_)
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hs : Membership.mem (↑I) s
    i : 𝒰.J
    ⊢ IsNilpotent ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) Top.top).hom ((Alge …
  -/
  rw [Scheme.isNilpotent_iff_basicOpen_eq_bot]
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hs : Membership.mem (↑I) s
    i : 𝒰.J
    ⊢ Eq ((𝒰.obj i).basicOpen ((AlgebraicGeometry.Scheme.Hom.app (𝒰.map i) Top.top …
  -/
  rw [Scheme.basicOpen_eq_bot_iff_forall_evaluation_eq_zero]
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hs : Membership.mem (↑I) s
    i : 𝒰.J
    ⊢ ∀ (x : Subtype fun x => Membership.mem Top.top x), Eq (((𝒰.obj i).evaluation …
  -/
  intro x
  suffices h : f.base ((𝒰.map i).base x.val) ∉ Y.basicOpen s by
    erw [← Scheme.Γevaluation_naturality_apply (𝒰.map i ≫ f)]
    simpa only [Scheme.comp_base, TopCat.coe_comp, Function.comp_apply,
      Scheme.residueFieldMap_comp, CommRingCat.comp_apply, map_eq_zero,
      Scheme.evaluation_eq_zero_iff_not_mem_basicOpen]
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfcl : IsClosed (Set.range ⇑f.base)
    hfinj : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    I : Ideal ↑(Y.presheaf.obj { unop := Top.top })
    hI : Eq (Set.range ⇑f.base) (Y.zeroLocus ↑I)
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hs : Membership.mem (↑I) s
    i : 𝒰.J
    x : Subtype fun x => Membership.mem Top.top x
    ⊢ Not (Membership.mem (Y.basicOpen s) (f.base ((𝒰.map i).base ↑x)))
  -/
  exact (Y.mem_zeroLocus_iff I _).mp (hI ▸ Set.mem_range_self ((𝒰.map i).base x.val)) s hs
  /-
    🎉 no goals
  -/


/-- If `f : X ⟶ Y` is open, injective, `X` is quasi-compact and `Y` is affine, then `f` is stalkwise
injective if it is injective on global sections. -/
lemma stalkMap_injective_of_isOpenMap_of_injective [CompactSpace X]
    (hfopen : IsOpenMap f.base) (hfinj₁ : Function.Injective f.base)
    (hfinj₂ : Function.Injective (f.appTop)) (x : X) :
    Function.Injective (f.stalkMap x) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  let φ : Γ(Y, ⊤) ⟶ Γ(X, ⊤) := f.appTop
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  let 𝒰 : X.OpenCover := X.affineCover.finiteSubcover
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  have (i : 𝒰.J) : IsAffine (𝒰.obj i) := Scheme.isAffine_affineCover X _
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  let res (i : 𝒰.J) : Γ(X, ⊤) ⟶ Γ(𝒰.obj i, ⊤) := (𝒰.map i).appTop
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    res : (i : 𝒰.J) → Quiver.Hom (X.presheaf.obj { unop := Top.top }) ((𝒰.obj i).p …
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  refine stalkMap_injective_of_isAffine _ _ (fun (g : Γ(Y, ⊤)) h ↦ ?_)
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    res : (i : 𝒰.J) → Quiver.Hom (X.presheaf.obj { unop := Top.top }) ((𝒰.obj i).p …
    g : ↑(Y.presheaf.obj { unop := Top.top })
    h : Eq ((AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom ((Y.presheaf.Γgerm (f. …
    ⊢ Eq ((Y.presheaf.Γgerm (f.base x)).hom g) 0
  -/
  rw [TopCat.Presheaf.Γgerm, Scheme.stalkMap_germ_apply] at h
  obtain ⟨U, w, (hx : x ∈ U), hg⟩ :=
    X.toRingedSpace.exists_res_eq_zero_of_germ_eq_zero ⊤ (φ g) ⟨x, trivial⟩ h
  obtain ⟨_, ⟨s, rfl⟩, hyv, bsle⟩ := Opens.isBasis_iff_nbhd.mp (isBasis_basicOpen Y)
    (show f.base x ∈ ⟨f.base '' U.carrier, hfopen U.carrier U.is_open'⟩ from ⟨x, by simpa⟩)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    res : (i : 𝒰.J) → Quiver.Hom (X.presheaf.obj { unop := Top.top }) ((𝒰.obj i).p …
    g : ↑(Y.presheaf.obj { unop := Top.top })
    h : Eq ((X.presheaf.germ ((TopologicalSpace.Opens.map f.base).obj Top.top) x T …
    U : TopologicalSpace.Opens ↑↑X.toRingedSpace.toPresheafedSpace
    w : Quiver.Hom U Top.top
    hx : Membership.mem U x
    hg : Eq ((X.toRingedSpace.presheaf.map w.op).hom (φ.hom g)) 0
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hyv : Membership.mem (Y.basicOpen s) (f.base x)
    bsle : LE.le (Y.basicOpen s) { carrier := Set.image (⇑f.base) U.carrier, is_op …
    ⊢ Eq ((Y.presheaf.Γgerm (f.base x)).hom g) 0
  -/
  let W (i : 𝒰.J) : TopologicalSpace.Opens (𝒰.obj i) := (𝒰.obj i).basicOpen ((res i) (φ s))
  have hwle (i : 𝒰.J) : W i ≤ (𝒰.map i)⁻¹ᵁ U := by
    show (𝒰.obj i).basicOpen ((𝒰.map i ≫ f).appTop s) ≤ _
    rw [← Scheme.preimage_basicOpen_top, Scheme.comp_coeBase, Opens.map_comp_obj]
    refine Scheme.Hom.preimage_le_preimage_of_le _
      (le_trans (f.preimage_le_preimage_of_le bsle) (le_of_eq ?_))
    simp [Set.preimage_image_eq _ hfinj₁]
  have h0 (i : 𝒰.J) : (𝒰.map i).appLE _ (W i) (by simp) (φ g) = 0 := by
    rw [← Scheme.Hom.appLE_map _ _ (homOfLE <| hwle i).op, ← Scheme.Hom.map_appLE _ le_rfl w.op]
    simp only [CommRingCat.comp_apply]
    erw [hg]
    simp only [map_zero]
  have h1 (i : 𝒰.J) : ∃ n, (res i) (φ (s ^ n * g)) = 0 := by
    obtain ⟨n, hn⟩ := exists_of_res_zero_of_qcqs_of_top (s := ((res i) (φ s))) (h0 i)
    exact ⟨n, by rwa [map_mul, map_mul, map_pow, map_pow]⟩
  have h2 : ∃ n, ∀ i, (res i) (φ (s ^ n * g)) = 0 := by
    choose fn hfn using h1
    refine ⟨Finset.sup Finset.univ fn, fun i ↦ ?_⟩
    rw [map_mul, map_pow, map_mul, map_pow]
    simp only [map_mul, map_pow, map_mul, map_pow] at hfn
    apply pow_mul_eq_zero_of_le (Finset.le_sup (Finset.mem_univ i)) (hfn i)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    res : (i : 𝒰.J) → Quiver.Hom (X.presheaf.obj { unop := Top.top }) ((𝒰.obj i).p …
    g : ↑(Y.presheaf.obj { unop := Top.top })
    h : Eq ((X.presheaf.germ ((TopologicalSpace.Opens.map f.base).obj Top.top) x T …
    U : TopologicalSpace.Opens ↑↑X.toRingedSpace.toPresheafedSpace
    w : Quiver.Hom U Top.top
    hx : Membership.mem U x
    hg : Eq ((X.toRingedSpace.presheaf.map w.op).hom (φ.hom g)) 0
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hyv : Membership.mem (Y.basicOpen s) (f.base x)
    bsle : LE.le (Y.basicOpen s) { carrier := Set.image (⇑f.base) U.carrier, is_op …
    W : (i : 𝒰.J) → TopologicalSpace.Opens ↑↑(𝒰.obj i).toPresheafedSpace := fun i  …
    hwle : ∀ (i : 𝒰.J), LE.le (W i) ((TopologicalSpace.Opens.map (𝒰.map i).base).o …
    h0 : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.appLE (𝒰.map i) Top.top (W …
    h1 : ∀ (i : 𝒰.J), Exists fun n => Eq ((res i).hom (φ.hom (HMul.hMul (HPow.hPow …
    h2 : Exists fun n => ∀ (i : 𝒰.J), Eq ((res i).hom (φ.hom (HMul.hMul (HPow.hPow …
    ⊢ Eq ((Y.presheaf.Γgerm (f.base x)).hom g) 0
  -/
  obtain ⟨n, hn⟩ := h2
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    res : (i : 𝒰.J) → Quiver.Hom (X.presheaf.obj { unop := Top.top }) ((𝒰.obj i).p …
    g : ↑(Y.presheaf.obj { unop := Top.top })
    h : Eq ((X.presheaf.germ ((TopologicalSpace.Opens.map f.base).obj Top.top) x T …
    U : TopologicalSpace.Opens ↑↑X.toRingedSpace.toPresheafedSpace
    w : Quiver.Hom U Top.top
    hx : Membership.mem U x
    hg : Eq ((X.toRingedSpace.presheaf.map w.op).hom (φ.hom g)) 0
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hyv : Membership.mem (Y.basicOpen s) (f.base x)
    bsle : LE.le (Y.basicOpen s) { carrier := Set.image (⇑f.base) U.carrier, is_op …
    W : (i : 𝒰.J) → TopologicalSpace.Opens ↑↑(𝒰.obj i).toPresheafedSpace := fun i  …
    hwle : ∀ (i : 𝒰.J), LE.le (W i) ((TopologicalSpace.Opens.map (𝒰.map i).base).o …
    h0 : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.appLE (𝒰.map i) Top.top (W …
    h1 : ∀ (i : 𝒰.J), Exists fun n => Eq ((res i).hom (φ.hom (HMul.hMul (HPow.hPow …
    n : Nat
    hn : ∀ (i : 𝒰.J), Eq ((res i).hom (φ.hom (HMul.hMul (HPow.hPow s n) g))) 0
    ⊢ Eq ((Y.presheaf.Γgerm (f.base x)).hom g) 0
  -/
  apply germ_eq_zero_of_pow_mul_eq_zero (U := ⊤) ⟨f.base x, trivial⟩ hyv
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    res : (i : 𝒰.J) → Quiver.Hom (X.presheaf.obj { unop := Top.top }) ((𝒰.obj i).p …
    g : ↑(Y.presheaf.obj { unop := Top.top })
    h : Eq ((X.presheaf.germ ((TopologicalSpace.Opens.map f.base).obj Top.top) x T …
    U : TopologicalSpace.Opens ↑↑X.toRingedSpace.toPresheafedSpace
    w : Quiver.Hom U Top.top
    hx : Membership.mem U x
    hg : Eq ((X.toRingedSpace.presheaf.map w.op).hom (φ.hom g)) 0
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hyv : Membership.mem (Y.basicOpen s) (f.base x)
    bsle : LE.le (Y.basicOpen s) { carrier := Set.image (⇑f.base) U.carrier, is_op …
    W : (i : 𝒰.J) → TopologicalSpace.Opens ↑↑(𝒰.obj i).toPresheafedSpace := fun i  …
    hwle : ∀ (i : 𝒰.J), LE.le (W i) ((TopologicalSpace.Opens.map (𝒰.map i).base).o …
    h0 : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.appLE (𝒰.map i) Top.top (W …
    h1 : ∀ (i : 𝒰.J), Exists fun n => Eq ((res i).hom (φ.hom (HMul.hMul (HPow.hPow …
    n : Nat
    hn : ∀ (i : 𝒰.J), Eq ((res i).hom (φ.hom (HMul.hMul (HPow.hPow s n) g))) 0
    ⊢ Eq (HMul.hMul (HPow.hPow s ?m.53976) g) 0
  -/
  rw [RingHom.injective_iff_ker_eq_bot, RingHom.ker_eq_bot_iff_eq_zero] at hfinj₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : CompactSpace ↑↑X.toPresheafedSpace
    hfopen : IsOpenMap ⇑f.base
    hfinj₁ : Function.Injective ⇑f.base
    hfinj₂✝ : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    hfinj₂ : ∀ (x : ↑(Y.presheaf.obj { unop := Top.top })), Eq ((AlgebraicGeometry …
    x : ↑↑X.toPresheafedSpace
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    res : (i : 𝒰.J) → Quiver.Hom (X.presheaf.obj { unop := Top.top }) ((𝒰.obj i).p …
    g : ↑(Y.presheaf.obj { unop := Top.top })
    h : Eq ((X.presheaf.germ ((TopologicalSpace.Opens.map f.base).obj Top.top) x T …
    U : TopologicalSpace.Opens ↑↑X.toRingedSpace.toPresheafedSpace
    w : Quiver.Hom U Top.top
    hx : Membership.mem U x
    hg : Eq ((X.toRingedSpace.presheaf.map w.op).hom (φ.hom g)) 0
    s : ↑(Y.presheaf.obj { unop := Top.top })
    hyv : Membership.mem (Y.basicOpen s) (f.base x)
    bsle : LE.le (Y.basicOpen s) { carrier := Set.image (⇑f.base) U.carrier, is_op …
    W : (i : 𝒰.J) → TopologicalSpace.Opens ↑↑(𝒰.obj i).toPresheafedSpace := fun i  …
    hwle : ∀ (i : 𝒰.J), LE.le (W i) ((TopologicalSpace.Opens.map (𝒰.map i).base).o …
    h0 : ∀ (i : 𝒰.J), Eq ((AlgebraicGeometry.Scheme.Hom.appLE (𝒰.map i) Top.top (W …
    h1 : ∀ (i : 𝒰.J), Exists fun n => Eq ((res i).hom (φ.hom (HMul.hMul (HPow.hPow …
    n : Nat
    hn : ∀ (i : 𝒰.J), Eq ((res i).hom (φ.hom (HMul.hMul (HPow.hPow s n) g))) 0
    ⊢ Eq (HMul.hMul (HPow.hPow s (?m.54966 x this g h U w hx hg s hyv bsle hwle h0 …
  -/
  exact hfinj₂ _ (Scheme.zero_of_zero_cover _ _ hn)
  /-
    🎉 no goals
  -/


/-- If `f` is a closed immersion with affine target such that the induced map on global
sections is injective, `f` is an isomorphism. -/
theorem isIso_of_injective_of_isAffine [IsClosedImmersion f]
    (hf : Function.Injective (f.appTop)) : IsIso f := (isIso_iff_stalk_iso f).mpr <|
  have : CompactSpace X := f.isClosedEmbedding.compactSpace
  have hiso : IsIso f.base := TopCat.isIso_of_bijective_of_isClosedMap _
    ⟨f.isClosedEmbedding.injective,
     surjective_of_isClosed_range_of_injective f.isClosedEmbedding.isClosed_range hf⟩
    (f.isClosedEmbedding.isClosedMap)
  ⟨hiso, fun x ↦ (ConcreteCategory.isIso_iff_bijective _).mpr
    ⟨stalkMap_injective_of_isOpenMap_of_injective ((TopCat.homeoOfIso (asIso f.base)).isOpenMap)
    f.isClosedEmbedding.injective hf _, f.stalkMap_surjective x⟩⟩


/-- If `f` is a closed immersion with affine target, the source is affine and
the induced map on global sections is surjective. -/
theorem isAffine_surjective_of_isAffine [IsClosedImmersion f] :
    IsAffine X ∧ Function.Surjective (f.appTop) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsClosedImmersion f
    ⊢ And (AlgebraicGeometry.IsAffine X) (Function.Surjective ⇑(AlgebraicGeometry. …
  -/
  haveI i : IsClosedImmersion f := inferInstance
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ i : AlgebraicGeometry.IsClosedImmersion f
    ⊢ And (AlgebraicGeometry.IsAffine X) (Function.Surjective ⇑(AlgebraicGeometry. …
  -/
  rw [← affineTargetImageFactorization_comp f] at i ⊢
  haveI := of_surjective_of_isAffine (affineTargetImageInclusion f)
    (affineTargetImageInclusion_app_surjective f)
  haveI := IsClosedImmersion.of_comp_isClosedImmersion (affineTargetImageFactorization f)
    (affineTargetImageInclusion f)
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsClosedImmersion f
    i : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.comp (A …
    this✝ : AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.affineTargetIma …
    this : AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.affineTargetImag …
    ⊢ And (AlgebraicGeometry.IsAffine X) (Function.Surjective ⇑(AlgebraicGeometry. …
  -/
  haveI := isIso_of_injective_of_isAffine (affineTargetImageFactorization_app_injective f)
  exact ⟨isAffine_of_isIso (affineTargetImageFactorization f),
    (ConcreteCategory.bijective_of_isIso
      ((affineTargetImageFactorization f).appTop)).surjective.comp <|
      affineTargetImageInclusion_app_surjective f⟩


/-- Being a closed immersion is local at the target. -/
instance IsClosedImmersion.isLocalAtTarget : IsLocalAtTarget @IsClosedImmersion :=
  eq_inf ▸ inferInstance


/-- On morphisms with affine target, being a closed immersion is precisely having affine source
and being surjective on global sections. -/
instance IsClosedImmersion.hasAffineProperty : HasAffineProperty @IsClosedImmersion
    (fun X _ f ↦ IsAffine X ∧ Function.Surjective (f.appTop)) := by
  /-
    ⊢ AlgebraicGeometry.HasAffineProperty @AlgebraicGeometry.IsClosedImmersion fun …
  -/
  convert HasAffineProperty.of_isLocalAtTarget @IsClosedImmersion
  /-
    case h.e'_2.h.h.h.h.a
    x✝³ x✝² : AlgebraicGeometry.Scheme
    x✝¹ : Quiver.Hom x✝³ x✝²
    x✝ : AlgebraicGeometry.IsAffine x✝²
    ⊢ Iff (And (AlgebraicGeometry.IsAffine x✝³) (Function.Surjective ⇑(AlgebraicGe …
  -/
  refine ⟨fun ⟨h₁, h₂⟩ ↦ of_surjective_of_isAffine _ h₂, by apply isAffine_surjective_of_isAffine⟩
  /-
    🎉 no goals
  -/


instance (priority := 900) {X Y : Scheme.{u}} (f : X ⟶ Y) [h : IsClosedImmersion f] :
    IsAffineHom f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsClosedImmersion f
    ⊢ AlgebraicGeometry.IsAffineHom f
  -/
  wlog hY : IsAffine Y
  · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := @IsAffineHom) _
      (iSup_affineOpens_eq_top Y)]
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsClosedImmersion f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ ∀ (i : ↑Y.affineOpens), AlgebraicGeometry.IsAffineHom (AlgebraicGeometry.mor …
    -/
    intro U
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsClosedImmersion f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      ⊢ AlgebraicGeometry.IsAffineHom (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    have H : IsClosedImmersion (f ∣_ U) := IsLocalAtTarget.restrict h U
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsClosedImmersion f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      H : AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.morphismRestrict f  …
      ⊢ AlgebraicGeometry.IsAffineHom (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    exact this _ U.2
    /-
      🎉 no goals
    -/
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsClosedImmersion f
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ AlgebraicGeometry.IsAffineHom f
  -/
  rw [HasAffineProperty.iff_of_isAffine (P := @IsAffineHom)]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsClosedImmersion f
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ AlgebraicGeometry.IsAffine X
  -/
  exact (IsClosedImmersion.isAffine_surjective_of_isAffine f).1
  /-
    🎉 no goals
  -/


/-- Being a closed immersion is stable under base change. -/
instance IsClosedImmersion.isStableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange @IsClosedImmersion := by
  /-
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.I …
  -/
  apply HasAffineProperty.isStableUnderBaseChange
  /-
    case hP'
    ⊢ AlgebraicGeometry.AffineTargetMorphismProperty.IsStableUnderBaseChange fun X …
  -/
  haveI := HasAffineProperty.isLocal_affineProperty @IsClosedImmersion
  /-
    case hP'
    this : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x f [Algeb …
    ⊢ AlgebraicGeometry.AffineTargetMorphismProperty.IsStableUnderBaseChange fun X …
  -/
  apply AffineTargetMorphismProperty.IsStableUnderBaseChange.mk
  /-
    case hP'.H
    this : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x f [Algeb …
    ⊢ ∀ ⦃X Y S : AlgebraicGeometry.Scheme⦄ [inst : AlgebraicGeometry.IsAffine S] [ …
  -/
  intro X Y S _ _ f g ⟨ha, hsurj⟩
  exact ⟨inferInstance, RingHom.surjective_isStableUnderBaseChange.pullback_fst_appTop _
    RingHom.surjective_respectsIso f _ hsurj⟩


/-- Closed immersions are locally of finite type. -/
instance (priority := 900) {X Y : Scheme.{u}} (f : X ⟶ Y) [h : IsClosedImmersion f] :
    LocallyOfFiniteType f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsClosedImmersion f
    ⊢ AlgebraicGeometry.LocallyOfFiniteType f
  -/
  wlog hY : IsAffine Y
  · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := @LocallyOfFiniteType) _
      (iSup_affineOpens_eq_top Y)]
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsClosedImmersion f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ ∀ (i : ↑Y.affineOpens), AlgebraicGeometry.LocallyOfFiniteType (AlgebraicGeom …
    -/
    intro U
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsClosedImmersion f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      ⊢ AlgebraicGeometry.LocallyOfFiniteType (AlgebraicGeometry.morphismRestrict f  …
    -/
    have H : IsClosedImmersion (f ∣_ U) := IsLocalAtTarget.restrict h U
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsClosedImmersion f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      H : AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.morphismRestrict f  …
      ⊢ AlgebraicGeometry.LocallyOfFiniteType (AlgebraicGeometry.morphismRestrict f  …
    -/
    exact this _ U.2
    /-
      🎉 no goals
    -/
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsClosedImmersion f
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ AlgebraicGeometry.LocallyOfFiniteType f
  -/
  obtain ⟨_, hf⟩ := h.isAffine_surjective_of_isAffine
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsClosedImmersion f
    hY : AlgebraicGeometry.IsAffine Y
    left✝ : AlgebraicGeometry.IsAffine X
    hf : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    ⊢ AlgebraicGeometry.LocallyOfFiniteType f
  -/
  rw [HasRingHomProperty.iff_of_isAffine (P := @LocallyOfFiniteType)]
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsClosedImmersion f
    hY : AlgebraicGeometry.IsAffine Y
    left✝ : AlgebraicGeometry.IsAffine X
    hf : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    ⊢ (AlgebraicGeometry.Scheme.Hom.appTop f).hom.FiniteType
  -/
  exact RingHom.FiniteType.of_surjective (Scheme.Hom.app f ⊤).hom hf
  /-
    🎉 no goals
  -/


/-- A surjective closed immersion is an isomorphism when the target is reduced. -/
lemma isIso_of_isClosedImmersion_of_surjective {X Y : Scheme.{u}} (f : X ⟶ Y)
    [IsClosedImmersion f] [Surjective f] [IsReduced Y] :
    IsIso f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.IsClosedImmersion f
    inst✝¹ : AlgebraicGeometry.Surjective f
    inst✝ : AlgebraicGeometry.IsReduced Y
    ⊢ CategoryTheory.IsIso f
  -/
  wlog hY : IsAffine Y
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝² : AlgebraicGeometry.IsClosedImmersion f
      inst✝¹ : AlgebraicGeometry.Surjective f
      inst✝ : AlgebraicGeometry.IsReduced Y
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ CategoryTheory.IsIso f
    -/
  · refine (IsLocalAtTarget.iff_of_openCover (P := .isomorphisms Scheme) Y.affineCover).mpr ?_
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝² : AlgebraicGeometry.IsClosedImmersion f
      inst✝¹ : AlgebraicGeometry.Surjective f
      inst✝ : AlgebraicGeometry.IsReduced Y
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ ∀ (i : Y.affineCover.1), CategoryTheory.MorphismProperty.isomorphisms Algebr …
    -/
    intro i
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝² : AlgebraicGeometry.IsClosedImmersion f
      inst✝¹ : AlgebraicGeometry.Surjective f
      inst✝ : AlgebraicGeometry.IsReduced Y
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      i : Y.affineCover.1
      ⊢ CategoryTheory.MorphismProperty.isomorphisms AlgebraicGeometry.Scheme (Algeb …
    -/
    apply (config := { allowSynthFailures := true }) this
      /-
        case inst
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        inst✝² : AlgebraicGeometry.IsClosedImmersion f
        inst✝¹ : AlgebraicGeometry.Surjective f
        inst✝ : AlgebraicGeometry.IsReduced Y
        this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        i : Y.affineCover.1
        ⊢ AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.Scheme.Cover.pullback …
      -/
    · exact MorphismProperty.pullback_snd _ _ inferInstance
      /-
        🎉 no goals
      -/
      /-
        case inst
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        inst✝² : AlgebraicGeometry.IsClosedImmersion f
        inst✝¹ : AlgebraicGeometry.Surjective f
        inst✝ : AlgebraicGeometry.IsReduced Y
        this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        i : Y.affineCover.1
        ⊢ AlgebraicGeometry.Surjective (AlgebraicGeometry.Scheme.Cover.pullbackHom Y.a …
      -/
    · exact IsLocalAtTarget.of_isPullback (.of_hasPullback f (Y.affineCover.map i)) ‹_›
      /-
        🎉 no goals
      -/
      /-
        case inst
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        inst✝² : AlgebraicGeometry.IsClosedImmersion f
        inst✝¹ : AlgebraicGeometry.Surjective f
        inst✝ : AlgebraicGeometry.IsReduced Y
        this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        i : Y.affineCover.1
        ⊢ AlgebraicGeometry.IsReduced (Y.affineCover.obj i)
      -/
    · exact isReduced_of_isOpenImmersion (Y.affineCover.map i)
      /-
        🎉 no goals
      -/
      /-
        case inr.hY
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        inst✝² : AlgebraicGeometry.IsClosedImmersion f
        inst✝¹ : AlgebraicGeometry.Surjective f
        inst✝ : AlgebraicGeometry.IsReduced Y
        this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        i : Y.affineCover.1
        ⊢ AlgebraicGeometry.IsAffine (Y.affineCover.obj i)
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.IsClosedImmersion f
    inst✝¹ : AlgebraicGeometry.Surjective f
    inst✝ : AlgebraicGeometry.IsReduced Y
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ CategoryTheory.IsIso f
  -/
  apply IsClosedImmersion.isIso_of_injective_of_isAffine
  /-
    case hf
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.IsClosedImmersion f
    inst✝¹ : AlgebraicGeometry.Surjective f
    inst✝ : AlgebraicGeometry.IsReduced Y
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
  -/
  obtain ⟨hX, hf⟩ := HasAffineProperty.iff_of_isAffine.mp ‹IsClosedImmersion f›
  /-
    case hf.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.IsClosedImmersion f
    inst✝¹ : AlgebraicGeometry.Surjective f
    inst✝ : AlgebraicGeometry.IsReduced Y
    hY : AlgebraicGeometry.IsAffine Y
    hX : AlgebraicGeometry.IsAffine X
    hf : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
  -/
  let φ := f.appTop
  suffices RingHom.ker φ.hom ≤ nilradical _ by
    rwa [nilradical_eq_zero, Submodule.zero_eq_bot, le_bot_iff,
      ← RingHom.injective_iff_ker_eq_bot] at this
  /-
    case hf.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.IsClosedImmersion f
    inst✝¹ : AlgebraicGeometry.Surjective f
    inst✝ : AlgebraicGeometry.IsReduced Y
    hY : AlgebraicGeometry.IsAffine Y
    hX : AlgebraicGeometry.IsAffine X
    hf : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    ⊢ LE.le (RingHom.ker φ.hom) (nilradical ↑(Y.presheaf.obj { unop := Top.top }))
  -/
  refine (PrimeSpectrum.zeroLocus_eq_top_iff _).mp ?_
  /-
    case hf.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.IsClosedImmersion f
    inst✝¹ : AlgebraicGeometry.Surjective f
    inst✝ : AlgebraicGeometry.IsReduced Y
    hY : AlgebraicGeometry.IsAffine Y
    hX : AlgebraicGeometry.IsAffine X
    hf : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    ⊢ Eq (PrimeSpectrum.zeroLocus ↑(RingHom.ker φ.hom)) Top.top
  -/
  rw [← range_specComap_of_surjective _ _ hf, Set.top_eq_univ, Set.range_eq_univ]
  have : Surjective (Spec.map (f.appTop)) :=
    (MorphismProperty.arrow_mk_iso_iff @Surjective (arrowIsoSpecΓOfIsAffine f)).mp
    (inferInstanceAs (Surjective f))
  /-
    case hf.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.IsClosedImmersion f
    inst✝¹ : AlgebraicGeometry.Surjective f
    inst✝ : AlgebraicGeometry.IsReduced Y
    hY : AlgebraicGeometry.IsAffine Y
    hX : AlgebraicGeometry.IsAffine X
    hf : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom
    φ : Quiver.Hom (Y.presheaf.obj { unop := Top.top }) (X.presheaf.obj { unop :=  …
    this : AlgebraicGeometry.Surjective (AlgebraicGeometry.Spec.map (AlgebraicGeom …
    ⊢ Function.Surjective (AlgebraicGeometry.Scheme.Hom.appTop f).hom.specComap
  -/
  exact this.1
  /-
    🎉 no goals
  -/


