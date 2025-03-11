/-- A morphism of schemes `X ⟶ Y` is affine if
the preimage of any affine open subset of `Y` is affine. -/
@[mk_iff]
class IsAffineHom {X Y : Scheme} (f : X ⟶ Y) : Prop where
  isAffine_preimage : ∀ U : Y.Opens, IsAffineOpen U → IsAffineOpen (f ⁻¹ᵁ U)


lemma IsAffineOpen.preimage {X Y : Scheme} {U : Y.Opens} (hU : IsAffineOpen U)
    (f : X ⟶ Y) [IsAffineHom f] :
    IsAffineOpen (f ⁻¹ᵁ U) :=
  IsAffineHom.isAffine_preimage _ hU


/-- The preimage of an affine open as an `Scheme.affine_opens`. -/
@[simps]
def affinePreimage {X Y : Scheme} (f : X ⟶ Y) [IsAffineHom f] (U : Y.affineOpens) :
    X.affineOpens :=
  ⟨f ⁻¹ᵁ U.1, IsAffineHom.isAffine_preimage _ U.prop⟩


instance (priority := 900) [IsIso f] : IsAffineHom f :=
  ⟨fun _ hU ↦ hU.preimage_of_isIso f⟩


instance (priority := 900) [IsAffineHom f] : QuasiCompact f :=
  (quasiCompact_iff_forall_affine f).mpr
    (fun U hU ↦ (IsAffineHom.isAffine_preimage U hU).isCompact)


instance [IsAffineHom f] [IsAffineHom g] : IsAffineHom (f ≫ g) := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsAffineHom f
    inst✝ : AlgebraicGeometry.IsAffineHom g
    ⊢ AlgebraicGeometry.IsAffineHom (CategoryTheory.CategoryStruct.comp f g)
  -/
  constructor
  /-
    case isAffine_preimage
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsAffineHom f
    inst✝ : AlgebraicGeometry.IsAffineHom g
    ⊢ ∀ (U : Z.Opens), AlgebraicGeometry.IsAffineOpen U → AlgebraicGeometry.IsAffi …
  -/
  intros U hU
  /-
    case isAffine_preimage
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsAffineHom f
    inst✝ : AlgebraicGeometry.IsAffineHom g
    U : Z.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map (CategoryTheory. …
  -/
  rw [Scheme.comp_base, Opens.map_comp_obj]
  /-
    case isAffine_preimage
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsAffineHom f
    inst✝ : AlgebraicGeometry.IsAffineHom g
    U : Z.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj ((To …
  -/
  apply IsAffineHom.isAffine_preimage
  /-
    case isAffine_preimage.a
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsAffineHom f
    inst✝ : AlgebraicGeometry.IsAffineHom g
    U : Z.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map g.base).obj U)
  -/
  apply IsAffineHom.isAffine_preimage
  /-
    case isAffine_preimage.a.a
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsAffineHom f
    inst✝ : AlgebraicGeometry.IsAffineHom g
    U : Z.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ AlgebraicGeometry.IsAffineOpen U
  -/
  exact hU
  /-
    🎉 no goals
  -/


instance : MorphismProperty.IsMultiplicative @IsAffineHom where
  id_mem := inferInstance
  comp_mem _ _ _ _ := inferInstance


instance {X : Scheme} (r : Γ(X, ⊤)) :
    IsAffineHom (X.basicOpen r).ι := by
  /-
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ AlgebraicGeometry.IsAffineHom (X.basicOpen r).ι
  -/
  constructor
  /-
    case isAffine_preimage
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ ∀ (U : X.Opens), AlgebraicGeometry.IsAffineOpen U → AlgebraicGeometry.IsAffi …
  -/
  intros U hU
  /-
    case isAffine_preimage
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map (X.basicOpen r). …
  -/
  fapply (Scheme.Hom.isAffineOpen_iff_of_isOpenImmersion (X.basicOpen r).ι).mp
  /-
    case isAffine_preimage
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFunctor ( …
  -/
  convert hU.basicOpen (X.presheaf.map (homOfLE le_top).op r)
  /-
    case h.e'_2
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor (X.basicOpen r).ι).obj ((Topo …
  -/
  rw [X.basicOpen_res]
  /-
    case h.e'_2
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor (X.basicOpen r).ι).obj ((Topo …
  -/
  ext1
  /-
    case h.e'_2.h
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq ↑((AlgebraicGeometry.Scheme.Hom.opensFunctor (X.basicOpen r).ι).obj ((Top …
  -/
  refine Set.image_preimage_eq_inter_range.trans ?_
  /-
    case h.e'_2.h
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (Inter.inter U.1 (Set.range ⇑(X.basicOpen r).ι.base)) ↑(Min.min U (X.basi …
  -/
  erw [Subtype.range_coe]
  /-
    case h.e'_2.h
    X✝ Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    g : Quiver.Hom Y Z
    X : AlgebraicGeometry.Scheme
    r : ↑(X.presheaf.obj { unop := Top.top })
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Eq (Inter.inter U.1 ↑(X.basicOpen r)) ↑(Min.min U (X.basicOpen r))
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma isAffineOpen_of_isAffineOpen_basicOpen_aux (s : Set Γ(X, ⊤))
    (hs : Ideal.span s = ⊤) (hs₂ : ∀ i ∈ s, IsAffineOpen (X.basicOpen i)) :
    QuasiSeparatedSpace X := by
  /-
    X : AlgebraicGeometry.Scheme
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
    ⊢ QuasiSeparatedSpace ↑↑X.toPresheafedSpace
  -/
  rw [quasiSeparatedSpace_iff_affine]
  /-
    X : AlgebraicGeometry.Scheme
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
    ⊢ ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
  -/
  intros U V
  /-
    X : AlgebraicGeometry.Scheme
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
    U V : ↑X.affineOpens
    ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
  -/
  obtain ⟨s', hs', e⟩ := (Ideal.span_eq_top_iff_finite _).mp hs
  rw [← Set.inter_univ (_ ∩ _), ← Opens.coe_top, ← iSup_basicOpen_of_span_eq_top _ _ e,
    ← iSup_subtype'', Opens.coe_iSup, Set.inter_iUnion]
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
    U V : ↑X.affineOpens
    s' : Finset ↑(X.presheaf.obj { unop := Top.top })
    hs' : HasSubset.Subset (↑s') s
    e : Eq (Ideal.span ↑s') Top.top
    ⊢ IsCompact (Set.iUnion fun i => Inter.inter (Inter.inter ↑↑U ↑↑V) ↑(X.basicOp …
  -/
  apply isCompact_iUnion
  /-
    case intro.intro.h
    X : AlgebraicGeometry.Scheme
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
    U V : ↑X.affineOpens
    s' : Finset ↑(X.presheaf.obj { unop := Top.top })
    hs' : HasSubset.Subset (↑s') s
    e : Eq (Ideal.span ↑s') Top.top
    ⊢ ∀ (i : ↑↑s'), IsCompact (Inter.inter (Inter.inter ↑↑U ↑↑V) ↑(X.basicOpen ↑i))
  -/
  intro i
  /-
    case intro.intro.h
    X : AlgebraicGeometry.Scheme
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
    U V : ↑X.affineOpens
    s' : Finset ↑(X.presheaf.obj { unop := Top.top })
    hs' : HasSubset.Subset (↑s') s
    e : Eq (Ideal.span ↑s') Top.top
    i : ↑↑s'
    ⊢ IsCompact (Inter.inter (Inter.inter ↑↑U ↑↑V) ↑(X.basicOpen ↑i))
  -/
  rw [Set.inter_inter_distrib_right]
  refine (hs₂ i (hs' i.2)).isQuasiSeparated _ _ Set.inter_subset_right
    (U.1.2.inter (X.basicOpen _).2) ?_ Set.inter_subset_right (V.1.2.inter (X.basicOpen _).2) ?_
    /-
      case intro.intro.h.refine_1
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      U V : ↑X.affineOpens
      s' : Finset ↑(X.presheaf.obj { unop := Top.top })
      hs' : HasSubset.Subset (↑s') s
      e : Eq (Ideal.span ↑s') Top.top
      i : ↑↑s'
      ⊢ IsCompact (Inter.inter ↑↑U ↑(X.basicOpen ↑i))
    -/
  · rw [← Opens.coe_inf, ← X.basicOpen_res _ (homOfLE le_top).op]
    /-
      case intro.intro.h.refine_1
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      U V : ↑X.affineOpens
      s' : Finset ↑(X.presheaf.obj { unop := Top.top })
      hs' : HasSubset.Subset (↑s') s
      e : Eq (Ideal.span ↑s') Top.top
      i : ↑↑s'
      ⊢ IsCompact ↑(X.basicOpen ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom  …
    -/
    exact (U.2.basicOpen _).isCompact
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.h.refine_2
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      U V : ↑X.affineOpens
      s' : Finset ↑(X.presheaf.obj { unop := Top.top })
      hs' : HasSubset.Subset (↑s') s
      e : Eq (Ideal.span ↑s') Top.top
      i : ↑↑s'
      ⊢ IsCompact (Inter.inter ↑↑V ↑(X.basicOpen ↑i))
    -/
  · rw [← Opens.coe_inf, ← X.basicOpen_res _ (homOfLE le_top).op]
    /-
      case intro.intro.h.refine_2
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      U V : ↑X.affineOpens
      s' : Finset ↑(X.presheaf.obj { unop := Top.top })
      hs' : HasSubset.Subset (↑s') s
      e : Eq (Ideal.span ↑s') Top.top
      i : ↑↑s'
      ⊢ IsCompact ↑(X.basicOpen ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom  …
    -/
    exact (V.2.basicOpen _).isCompact
    /-
      🎉 no goals
    -/


lemma isAffine_of_isAffineOpen_basicOpen (s : Set Γ(X, ⊤))
    (hs : Ideal.span s = ⊤) (hs₂ : ∀ i ∈ s, IsAffineOpen (X.basicOpen i)) :
    IsAffine X := by
  /-
    X : AlgebraicGeometry.Scheme
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
    ⊢ AlgebraicGeometry.IsAffine X
  -/
  have : QuasiSeparatedSpace X := isAffineOpen_of_isAffineOpen_basicOpen_aux s hs hs₂
  have : CompactSpace X := by
    obtain ⟨s', hs', e⟩ := (Ideal.span_eq_top_iff_finite _).mp hs
    rw [← isCompact_univ_iff, ← Opens.coe_top, ← iSup_basicOpen_of_span_eq_top _ _ e]
    simp only [Finset.mem_coe, Opens.iSup_mk, Opens.carrier_eq_coe, Opens.coe_mk]
    apply s'.isCompact_biUnion
    exact fun i hi ↦ (hs₂ _ (hs' hi)).isCompact
  /-
    X : AlgebraicGeometry.Scheme
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
    this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
    this : CompactSpace ↑↑X.toPresheafedSpace
    ⊢ AlgebraicGeometry.IsAffine X
  -/
  constructor
  refine HasAffineProperty.of_iSup_eq_top (P := MorphismProperty.isomorphisms Scheme)
    (fun i : s ↦ ⟨PrimeSpectrum.basicOpen i.1, ?_⟩) ?_ (fun i ↦ ⟨?_, ?_⟩)
    /-
      case affine.refine_1
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ Membership.mem (AlgebraicGeometry.Spec (X.presheaf.obj { unop := Top.top })) …
    -/
  · show IsAffineOpen _
    /-
      case affine.refine_1
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ AlgebraicGeometry.IsAffineOpen (PrimeSpectrum.basicOpen ↑i)
    -/
    simp only [← basicOpen_eq_of_affine]
    /-
      case affine.refine_1
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Spec (X.presheaf.obj { un …
    -/
    exact (isAffineOpen_top (Scheme.Spec.obj (op _))).basicOpen _
    /-
      🎉 no goals
    -/
    /-
      case affine.refine_2
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      ⊢ Eq (iSup fun i => ↑((fun i => ⟨PrimeSpectrum.basicOpen ↑i, ⋯⟩) i)) Top.top
    -/
  · rw [PrimeSpectrum.iSup_basicOpen_eq_top_iff, Subtype.range_coe_subtype, Set.setOf_mem_eq, hs]
    /-
      🎉 no goals
    -/
    /-
      case affine.refine_3
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map X.toSpecΓ.base).obj …
    -/
  · rw [Scheme.toSpecΓ_preimage_basicOpen]
    /-
      case affine.refine_3
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ AlgebraicGeometry.IsAffine ↑(X.basicOpen ↑i)
    -/
    exact hs₂ _ i.2
    /-
      🎉 no goals
    -/
  · simp only [Functor.comp_obj, Functor.rightOp_obj, Scheme.Γ_obj, Scheme.Spec_obj, id_eq,
      eq_mpr_eq_cast, Functor.id_obj, Opens.map_top, morphismRestrict_app]
    /-
      case affine.refine_4
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
    -/
    refine IsIso.comp_isIso' ?_ inferInstance
    /-
      case affine.refine_4
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.app X.toSpecΓ ((Algebraic …
    -/
    convert isIso_ΓSpec_adjunction_unit_app_basicOpen i.1 using 0
    /-
      case a
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ Iff (CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.app X.toSpecΓ ((Alge …
    -/
    refine congr(IsIso ((ΓSpec.adjunction.unit.app X).app $(?_)))
    /-
      case a
      X : AlgebraicGeometry.Scheme
      s : Set ↑(X.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → Alge …
      this✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      this : CompactSpace ↑↑X.toPresheafedSpace
      i : ↑s
      ⊢ Eq ((AlgebraicGeometry.Scheme.Hom.opensFunctor (AlgebraicGeometry.Scheme.Ope …
    -/
    rw [Opens.isOpenEmbedding_obj_top]
    /-
      🎉 no goals
    -/


/--
If `s` is a spanning set of `Γ(X, U)`, such that each `X.basicOpen i` is affine, then `U` is also
affine.
-/
lemma isAffineOpen_of_isAffineOpen_basicOpen (U) (s : Set Γ(X, U))
    (hs : Ideal.span s = ⊤) (hs₂ : ∀ i ∈ s, IsAffineOpen (X.basicOpen i)) :
    IsAffineOpen U := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    s : Set ↑(X.presheaf.obj { unop := U })
    hs : Eq (Ideal.span s) Top.top
    hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := U })), Membership.mem s i → AlgebraicG …
    ⊢ AlgebraicGeometry.IsAffineOpen U
  -/
  apply isAffine_of_isAffineOpen_basicOpen (U.topIso.inv '' s)
    /-
      case hs
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := U })), Membership.mem s i → AlgebraicG …
      ⊢ Eq (Ideal.span (Set.image (⇑U.topIso.inv.hom) s)) Top.top
    -/
  · rw [← Ideal.map_span U.topIso.inv.hom, hs, Ideal.map_top]
    /-
      🎉 no goals
    -/
    /-
      case hs₂
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := U })), Membership.mem s i → AlgebraicG …
      ⊢ ∀ (i : ↑((↑U).presheaf.obj { unop := Top.top })), Membership.mem (Set.image  …
    -/
  · rintro _ ⟨j, hj, rfl⟩
    /-
      case hs₂.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := U })), Membership.mem s i → AlgebraicG …
      j : ↑(X.presheaf.obj { unop := U })
      hj : Membership.mem s j
      ⊢ AlgebraicGeometry.IsAffineOpen ((↑U).basicOpen (U.topIso.inv.hom j))
    -/
    rw [← (Scheme.Opens.ι _).isAffineOpen_iff_of_isOpenImmersion, Scheme.image_basicOpen]
    /-
      case hs₂.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      s : Set ↑(X.presheaf.obj { unop := U })
      hs : Eq (Ideal.span s) Top.top
      hs₂ : ∀ (i : ↑(X.presheaf.obj { unop := U })), Membership.mem s i → AlgebraicG …
      j : ↑(X.presheaf.obj { unop := U })
      hj : Membership.mem s j
      ⊢ AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.a …
    -/
    simpa [Scheme.Opens.toScheme_presheaf_obj] using hs₂ j hj
    /-
      🎉 no goals
    -/


instance : HasAffineProperty @IsAffineHom fun X _ _ _ ↦ IsAffine X where
  isLocal_affineProperty := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x => Alge …
    -/
    constructor
      /-
        case respectsIso
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ (AlgebraicGeometry.AffineTargetMorphismProperty.toProperty fun X x x_1 x =>  …
      -/
    · apply AffineTargetMorphismProperty.respectsIso_mk
        /-
          case respectsIso.h₁
          X Y Z : AlgebraicGeometry.Scheme
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme}, CategoryTheory.Iso X Y → Quiver.Hom Y  …
        -/
      · rintro X Y Z e _ _ H
        /-
          case respectsIso.h₁
          X✝ Y✝ Z✝ : AlgebraicGeometry.Scheme
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          X Y Z : AlgebraicGeometry.Scheme
          e : CategoryTheory.Iso X Y
          f✝ : Quiver.Hom Y Z
          inst✝ : AlgebraicGeometry.IsAffine Z
          H : AlgebraicGeometry.IsAffine Y
          ⊢ AlgebraicGeometry.IsAffine X
        -/
        have : IsAffine _ := H
        /-
          case respectsIso.h₁
          X✝ Y✝ Z✝ : AlgebraicGeometry.Scheme
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          X Y Z : AlgebraicGeometry.Scheme
          e : CategoryTheory.Iso X Y
          f✝ : Quiver.Hom Y Z
          inst✝ : AlgebraicGeometry.IsAffine Z
          H this : AlgebraicGeometry.IsAffine Y
          ⊢ AlgebraicGeometry.IsAffine X
        -/
        exact isAffine_of_isIso e.hom
        /-
          🎉 no goals
        -/
        /-
          case respectsIso.h₂
          X Y Z : AlgebraicGeometry.Scheme
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme}, CategoryTheory.Iso Y Z → Quiver.Hom X  …
        -/
      · exact fun _ _ _ ↦ id
        /-
          🎉 no goals
        -/
      /-
        case to_basicOpen
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y] (f  …
      -/
    · intro X Y _ f r H
      /-
        case to_basicOpen
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H : AlgebraicGeometry.IsAffine X
        ⊢ AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj (Y.basi …
      -/
      have : IsAffine X := H
      /-
        case to_basicOpen
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H this : AlgebraicGeometry.IsAffine X
        ⊢ AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj (Y.basi …
      -/
      show IsAffineOpen _
      /-
        case to_basicOpen
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H this : AlgebraicGeometry.IsAffine X
        ⊢ AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj (Y.b …
      -/
      rw [Scheme.preimage_basicOpen]
      /-
        case to_basicOpen
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H this : AlgebraicGeometry.IsAffine X
        ⊢ AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.a …
      -/
      exact (isAffineOpen_top X).basicOpen _
      /-
        🎉 no goals
      -/
      /-
        case of_basicOpenCover
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y] (f  …
      -/
    · intro X Y _ f S hS hS'
      /-
        case of_basicOpenCover
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        S : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hS : Eq (Ideal.span ↑S) Top.top
        hS' : ∀ (r : Subtype fun x => Membership.mem S x), AlgebraicGeometry.IsAffine  …
        ⊢ AlgebraicGeometry.IsAffine X
      -/
      apply_fun Ideal.map (f.appTop).hom at hS
      /-
        case of_basicOpenCover
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        S : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hS' : ∀ (r : Subtype fun x => Membership.mem S x), AlgebraicGeometry.IsAffine  …
        hS : Eq (Ideal.map (AlgebraicGeometry.Scheme.Hom.appTop f).hom (Ideal.span ↑S) …
        ⊢ AlgebraicGeometry.IsAffine X
      -/
      rw [Ideal.map_span, Ideal.map_top] at hS
      /-
        case of_basicOpenCover
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        S : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hS' : ∀ (r : Subtype fun x => Membership.mem S x), AlgebraicGeometry.IsAffine  …
        hS : Eq (Ideal.span (Set.image ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom ↑S …
        ⊢ AlgebraicGeometry.IsAffine X
      -/
      apply isAffine_of_isAffineOpen_basicOpen _ hS
      /-
        case of_basicOpenCover
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        S : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hS' : ∀ (r : Subtype fun x => Membership.mem S x), AlgebraicGeometry.IsAffine  …
        hS : Eq (Ideal.span (Set.image ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom ↑S …
        ⊢ ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem (Set.image ⇑(A …
      -/
      have : ∀ i : S, IsAffineOpen (f⁻¹ᵁ Y.basicOpen i.1) := hS'
      /-
        case of_basicOpenCover
        X✝ Y✝ Z : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        S : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hS' : ∀ (r : Subtype fun x => Membership.mem S x), AlgebraicGeometry.IsAffine  …
        hS : Eq (Ideal.span (Set.image ⇑(AlgebraicGeometry.Scheme.Hom.appTop f).hom ↑S …
        this : ∀ (i : Subtype fun x => Membership.mem S x), AlgebraicGeometry.IsAffine …
        ⊢ ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem (Set.image ⇑(A …
      -/
      simpa [Scheme.preimage_basicOpen] using this
      /-
        🎉 no goals
      -/
  eq_targetAffineLocally' := by
    /-
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (@AlgebraicGeometry.IsAffineHom) (AlgebraicGeometry.targetAffineLocally f …
    -/
    ext X Y f
    simp only [targetAffineLocally, Scheme.affineOpens, Set.coe_setOf, Set.mem_setOf_eq,
      Subtype.forall, isAffineHom_iff]
    /-
      case h
      X✝ Y✝ Z : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → AlgebraicGeometry.I …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma isAffineHom_isStableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange @IsAffineHom := by
  /-
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.I …
  -/
  apply HasAffineProperty.isStableUnderBaseChange
  /-
    case hP'
    ⊢ AlgebraicGeometry.AffineTargetMorphismProperty.IsStableUnderBaseChange fun X …
  -/
  letI := HasAffineProperty.isLocal_affineProperty
  /-
    case hP'
    this : ∀ (P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme) {Q : o …
    ⊢ AlgebraicGeometry.AffineTargetMorphismProperty.IsStableUnderBaseChange fun X …
  -/
  apply AffineTargetMorphismProperty.IsStableUnderBaseChange.mk
  /-
    case hP'.H
    this : ∀ (P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme) {Q : o …
    ⊢ ∀ ⦃X Y S : AlgebraicGeometry.Scheme⦄ [inst : AlgebraicGeometry.IsAffine S] [ …
  -/
  introv X hX H
  /-
    case hP'.H
    this : ∀ (P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme) {Q : o …
    X✝ Y S : AlgebraicGeometry.Scheme
    X : AlgebraicGeometry.IsAffine S
    hX : AlgebraicGeometry.IsAffine X✝
    f : Quiver.Hom X✝ S
    g : Quiver.Hom Y S
    H : AlgebraicGeometry.IsAffine Y
    ⊢ AlgebraicGeometry.IsAffine (CategoryTheory.Limits.pullback f g)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (priority := 100) isAffineHom_of_isAffine [IsAffine X] [IsAffine Y] : IsAffineHom f :=
  (HasAffineProperty.iff_of_isAffine (P := @IsAffineHom)).mpr inferInstance


lemma isAffine_of_isAffineHom [IsAffineHom f] [IsAffine Y] : IsAffine X :=
  (HasAffineProperty.iff_of_isAffine (P := @IsAffineHom) (f := f)).mp inferInstance


