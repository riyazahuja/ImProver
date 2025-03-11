/--
A morphism is "quasi-compact" if the underlying map of topological spaces is, i.e. if the preimages
of quasi-compact open sets are quasi-compact.
-/
@[mk_iff]
class QuasiCompact (f : X ⟶ Y) : Prop where
  /-- Preimage of compact open set under a quasi-compact morphism between schemes is compact. -/
  isCompact_preimage : ∀ U : Set Y, IsOpen U → IsCompact U → IsCompact (f.base ⁻¹' U)


theorem quasiCompact_iff_spectral : QuasiCompact f ↔ IsSpectralMap f.base :=
                  /-
                    X Y : AlgebraicGeometry.Scheme
                    f : Quiver.Hom X Y
                    x✝ : AlgebraicGeometry.QuasiCompact f
                    h : ∀ (U : Set ↑↑Y.toPresheafedSpace), IsOpen U → IsCompact U → IsCompact (Set …
                    ⊢ Continuous ⇑f.base
                  -/
  ⟨fun ⟨h⟩ => ⟨by fun_prop, h⟩, fun h => ⟨h.2⟩⟩
                  /-
                    🎉 no goals
                  -/


instance (priority := 900) quasiCompact_of_isIso {X Y : Scheme} (f : X ⟶ Y) [IsIso f] :
    QuasiCompact f := by
  /-
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ AlgebraicGeometry.QuasiCompact f
  -/
  constructor
  /-
    case isCompact_preimage
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ ∀ (U : Set ↑↑Y.toPresheafedSpace), IsOpen U → IsCompact U → IsCompact (Set.p …
  -/
  intro U _ hU'
  /-
    case isCompact_preimage
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    U : Set ↑↑Y.toPresheafedSpace
    a✝ : IsOpen U
    hU' : IsCompact U
    ⊢ IsCompact (Set.preimage (⇑f.base) U)
  -/
  convert hU'.image (inv f.base).continuous_toFun using 1
  /-
    case h.e'_3
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    U : Set ↑↑Y.toPresheafedSpace
    a✝ : IsOpen U
    hU' : IsCompact U
    ⊢ Eq (Set.preimage (⇑f.base) U) (Set.image (CategoryTheory.inv f.base).toFun U)
  -/
  rw [Set.image_eq_preimage_of_inverse]
    /-
      case h.e'_3.h₁
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      U : Set ↑↑Y.toPresheafedSpace
      a✝ : IsOpen U
      hU' : IsCompact U
      ⊢ Function.LeftInverse (⇑f.base) (CategoryTheory.inv f.base).toFun
    -/
  · delta Function.LeftInverse
    /-
      case h.e'_3.h₁
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      U : Set ↑↑Y.toPresheafedSpace
      a✝ : IsOpen U
      hU' : IsCompact U
      ⊢ ∀ (x : ↑↑Y.toPresheafedSpace), Eq (f.base ((CategoryTheory.inv f.base).toFun …
    -/
    exact IsIso.inv_hom_id_apply f.base
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h₂
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso f
      U : Set ↑↑Y.toPresheafedSpace
      a✝ : IsOpen U
      hU' : IsCompact U
      ⊢ Function.RightInverse (⇑f.base) (CategoryTheory.inv f.base).toFun
    -/
  · exact IsIso.hom_inv_id_apply f.base
    /-
      🎉 no goals
    -/


instance quasiCompact_comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [QuasiCompact f]
    [QuasiCompact g] : QuasiCompact (f ≫ g) := by
  /-
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.QuasiCompact f
    inst✝ : AlgebraicGeometry.QuasiCompact g
    ⊢ AlgebraicGeometry.QuasiCompact (CategoryTheory.CategoryStruct.comp f g)
  -/
  constructor
  /-
    case isCompact_preimage
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.QuasiCompact f
    inst✝ : AlgebraicGeometry.QuasiCompact g
    ⊢ ∀ (U : Set ↑↑Z.toPresheafedSpace), IsOpen U → IsCompact U → IsCompact (Set.p …
  -/
  intro U hU hU'
  /-
    case isCompact_preimage
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.QuasiCompact f
    inst✝ : AlgebraicGeometry.QuasiCompact g
    U : Set ↑↑Z.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    ⊢ IsCompact (Set.preimage (⇑(CategoryTheory.CategoryStruct.comp f g).base) U)
  -/
  rw [Scheme.comp_base, TopCat.coe_comp, Set.preimage_comp]
  /-
    case isCompact_preimage
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.QuasiCompact f
    inst✝ : AlgebraicGeometry.QuasiCompact g
    U : Set ↑↑Z.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    ⊢ IsCompact (Set.preimage (⇑f.base) (Set.preimage (⇑g.base) U))
  -/
  apply QuasiCompact.isCompact_preimage
    /-
      case isCompact_preimage.a
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝¹ : AlgebraicGeometry.QuasiCompact f
      inst✝ : AlgebraicGeometry.QuasiCompact g
      U : Set ↑↑Z.toPresheafedSpace
      hU : IsOpen U
      hU' : IsCompact U
      ⊢ IsOpen (Set.preimage (⇑g.base) U)
    -/
  · exact Continuous.isOpen_preimage (by fun_prop) _ hU
    /-
      🎉 no goals
    -/
  /-
    case isCompact_preimage.a
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.QuasiCompact f
    inst✝ : AlgebraicGeometry.QuasiCompact g
    U : Set ↑↑Z.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    ⊢ IsCompact (Set.preimage (⇑g.base) U)
  -/
                                            /-
                                              🎉 no goals
                                            -/
  apply QuasiCompact.isCompact_preimage <;> assumption
                                            /-
                                              🎉 no goals
                                            -/


theorem isCompactOpen_iff_eq_finset_affine_union {X : Scheme} (U : Set X) :
    IsCompact U ∧ IsOpen U ↔ ∃ s : Set X.affineOpens, s.Finite ∧ U = ⋃ i ∈ s, i := by
  apply Opens.IsBasis.isCompact_open_iff_eq_finite_iUnion
    (fun (U : X.affineOpens) => (U : X.Opens))
    /-
      case hb
      X : AlgebraicGeometry.Scheme
      U : Set ↑↑X.toPresheafedSpace
      ⊢ TopologicalSpace.Opens.IsBasis (Set.range fun U => ↑U)
    -/
  · rw [Subtype.range_coe]; exact isBasis_affine_open X
                            /-
                              🎉 no goals
                            -/
    /-
      case hb'
      X : AlgebraicGeometry.Scheme
      U : Set ↑↑X.toPresheafedSpace
      ⊢ ∀ (i : ↑X.affineOpens), IsCompact ↑↑i
    -/
  · exact fun i => i.2.isCompact
    /-
      🎉 no goals
    -/


theorem isCompactOpen_iff_eq_basicOpen_union {X : Scheme} [IsAffine X] (U : Set X) :
    IsCompact U ∧ IsOpen U ↔
      ∃ s : Set Γ(X, ⊤), s.Finite ∧ U = ⋃ i ∈ s, X.basicOpen i :=
  (isBasis_basicOpen X).isCompact_open_iff_eq_finite_iUnion _
    (fun _ => ((isAffineOpen_top _).basicOpen _).isCompact) _


theorem quasiCompact_iff_forall_affine :
    QuasiCompact f ↔
      ∀ U : Y.Opens, IsAffineOpen U → IsCompact (f ⁻¹ᵁ U : Set X) := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.QuasiCompact f) (∀ (U : Y.Opens), AlgebraicGeometry.I …
  -/
  rw [quasiCompact_iff]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (∀ (U : Set ↑↑Y.toPresheafedSpace), IsOpen U → IsCompact U → IsCompact ( …
  -/
  refine ⟨fun H U hU => H U U.isOpen hU.isCompact, ?_⟩
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → IsCompact ↑((Topologica …
  -/
  intro H U hU hU'
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → IsCompact ↑((Topologic …
    U : Set ↑↑Y.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    ⊢ IsCompact (Set.preimage (⇑f.base) U)
  -/
  obtain ⟨S, hS, rfl⟩ := (isCompactOpen_iff_eq_finset_affine_union U).mp ⟨hU', hU⟩
  /-
    case intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → IsCompact ↑((Topologic …
    S : Set ↑Y.affineOpens
    hS : S.Finite
    hU : IsOpen (Set.iUnion fun i => Set.iUnion fun h => ↑↑i)
    hU' : IsCompact (Set.iUnion fun i => Set.iUnion fun h => ↑↑i)
    ⊢ IsCompact (Set.preimage (⇑f.base) (Set.iUnion fun i => Set.iUnion fun h => ↑ …
  -/
  simp only [Set.preimage_iUnion]
  /-
    case intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → IsCompact ↑((Topologic …
    S : Set ↑Y.affineOpens
    hS : S.Finite
    hU : IsOpen (Set.iUnion fun i => Set.iUnion fun h => ↑↑i)
    hU' : IsCompact (Set.iUnion fun i => Set.iUnion fun h => ↑↑i)
    ⊢ IsCompact (Set.iUnion fun i => Set.iUnion fun i_1 => Set.preimage ⇑f.base ↑↑i)
  -/
  exact Set.Finite.isCompact_biUnion hS (fun i _ => H i i.prop)
  /-
    🎉 no goals
  -/


theorem isCompact_basicOpen (X : Scheme) {U : X.Opens} (hU : IsCompact (U : Set X))
    (f : Γ(X, U)) : IsCompact (X.basicOpen f : Set X) := by
  classical
  refine ((isCompactOpen_iff_eq_finset_affine_union _).mpr ?_).1
  obtain ⟨s, hs, e⟩ := (isCompactOpen_iff_eq_finset_affine_union _).mp ⟨hU, U.isOpen⟩
  let g : s → X.affineOpens := by
    intro V
    use V.1 ⊓ X.basicOpen f
    have : V.1.1 ⟶ U := by
      apply homOfLE; change _ ⊆ (U : Set X); rw [e]
      convert Set.subset_iUnion₂ (s := fun (U : X.affineOpens) (_ : U ∈ s) => (U : Set X))
        V V.prop using 1
    erw [← X.toLocallyRingedSpace.toRingedSpace.basicOpen_res this.op]
    exact IsAffineOpen.basicOpen V.1.prop _
  haveI : Finite s := hs.to_subtype
  refine ⟨Set.range g, Set.finite_range g, ?_⟩
  refine (Set.inter_eq_right.mpr
            (SetLike.coe_subset_coe.2 <| RingedSpace.basicOpen_le _ _)).symm.trans ?_
  rw [e, Set.iUnion₂_inter]
  apply le_antisymm <;> apply Set.iUnion₂_subset
  · intro i hi
    -- Porting note: had to make explicit the first given parameter to `Set.subset_iUnion₂`
    exact Set.Subset.trans (Set.Subset.rfl : _ ≤ g ⟨i, hi⟩)
      (@Set.subset_iUnion₂ _ _ _
        (fun (i : X.affineOpens) (_ : i ∈ Set.range g) => (i : Set X.toPresheafedSpace)) _
        (Set.mem_range_self ⟨i, hi⟩))
  · rintro ⟨i, hi⟩ ⟨⟨j, hj⟩, hj'⟩
    rw [← hj']
    refine Set.Subset.trans ?_ (Set.subset_iUnion₂ j hj)
    exact Set.Subset.rfl


instance : HasAffineProperty @QuasiCompact (fun X _ _ _ ↦ CompactSpace X) where
  eq_targetAffineLocally' := by
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Eq (@AlgebraicGeometry.QuasiCompact) (AlgebraicGeometry.targetAffineLocally  …
    -/
    ext X Y f
    simp only [quasiCompact_iff_forall_affine, isCompact_iff_compactSpace, targetAffineLocally,
      Subtype.forall]
    /-
      case h
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → CompactSpace ↑↑((To …
    -/
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x => Comp …
    -/
    rfl
      /-
        case respectsIso
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ⊢ (AlgebraicGeometry.AffineTargetMorphismProperty.toProperty fun X x x_1 x =>  …
      -/
    /-
      🎉 no goals
    -/
      /-
        case respectsIso.h₁
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f : Quiver.Hom X✝ Y✝
        X Y Z : AlgebraicGeometry.Scheme
        e : CategoryTheory.Iso X Y
        f✝ : Quiver.Hom Y Z
        inst✝ : AlgebraicGeometry.IsAffine Z
        H : CompactSpace ↑↑Y.toPresheafedSpace
        ⊢ CompactSpace ↑↑X.toPresheafedSpace
      -/
  isLocal_affineProperty := by
      /-
        🎉 no goals
      -/
      /-
        case to_basicOpen
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y] (f  …
      -/
    constructor
      /-
        case to_basicOpen
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H : CompactSpace ↑↑X.toPresheafedSpace
        ⊢ CompactSpace ↑↑(↑((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen r))). …
      -/
    · apply AffineTargetMorphismProperty.respectsIso_mk <;> rintro X Y Z e _ _ H
      /-
        case to_basicOpen
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H : CompactSpace ↑↑X.toPresheafedSpace
        ⊢ CompactSpace ↑↑(↑((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen r))). …
      -/
      exacts [@Homeomorph.compactSpace _ _ _ _ H (TopCat.homeoOfIso (asIso e.inv.base)), H]
      /-
        case to_basicOpen
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H : CompactSpace ↑↑X.toPresheafedSpace
        ⊢ CompactSpace ↑↑(↑(X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app f Top.top). …
      -/
    · introv _ H
      /-
        case to_basicOpen
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H : CompactSpace ↑↑X.toPresheafedSpace
        ⊢ IsCompact ↑(X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app f Top.top).hom r))
      -/
      change CompactSpace ((Opens.map f.base).obj (Y.basicOpen r))
      /-
        case to_basicOpen
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H : IsCompact Set.univ
        ⊢ IsCompact ↑(X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app f Top.top).hom r))
      -/
      rw [Scheme.preimage_basicOpen f r]
      /-
        case to_basicOpen.hU
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        H : IsCompact Set.univ
        ⊢ IsCompact ↑((TopologicalSpace.Opens.map f.base).obj Top.top)
      -/
      erw [← isCompact_iff_compactSpace]
      /-
        🎉 no goals
      -/
      /-
        case of_basicOpenCover
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y] (f  …
      -/
      rw [← isCompact_univ_iff] at H
      /-
        case of_basicOpenCover
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        X Y : AlgebraicGeometry.Scheme
        H : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        S : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hS : Eq (Ideal.span ↑S) Top.top
        hS' : ∀ (r : Subtype fun x => Membership.mem S x), CompactSpace ↑↑(↑((Topologi …
        ⊢ CompactSpace ↑↑X.toPresheafedSpace
      -/
      apply isCompact_basicOpen
        /-
          case of_basicOpenCover
          X✝ Y✝ : AlgebraicGeometry.Scheme
          f✝ : Quiver.Hom X✝ Y✝
          X Y : AlgebraicGeometry.Scheme
          H : AlgebraicGeometry.IsAffine Y
          f : Quiver.Hom X Y
          S : Finset ↑(Y.presheaf.obj { unop := Top.top })
          hS : Eq (iSup fun f => Y.basicOpen ↑f) Top.top
          hS' : ∀ (r : Subtype fun x => Membership.mem S x), CompactSpace ↑↑(↑((Topologi …
          ⊢ CompactSpace ↑↑X.toPresheafedSpace
        -/
      exact H
        /-
          case of_basicOpenCover
          X✝ Y✝ : AlgebraicGeometry.Scheme
          f✝ : Quiver.Hom X✝ Y✝
          X Y : AlgebraicGeometry.Scheme
          H : AlgebraicGeometry.IsAffine Y
          f : Quiver.Hom X Y
          S : Finset ↑(Y.presheaf.obj { unop := Top.top })
          hS : Eq (iSup fun f => Y.basicOpen ↑f) Top.top
          hS' : ∀ (r : Subtype fun x => Membership.mem S x), CompactSpace ↑↑(↑((Topologi …
          ⊢ IsCompact Set.univ
        -/
    · rintro X Y H f S hS hS'
        /-
          case of_basicOpenCover
          X✝ Y✝ : AlgebraicGeometry.Scheme
          f✝ : Quiver.Hom X✝ Y✝
          X Y : AlgebraicGeometry.Scheme
          H : AlgebraicGeometry.IsAffine Y
          f : Quiver.Hom X Y
          S : Finset ↑(Y.presheaf.obj { unop := Top.top })
          hS : Eq (iSup fun f => Y.basicOpen ↑f) Top.top
          hS' : ∀ (r : Subtype fun x => Membership.mem S x), CompactSpace ↑↑(↑((Topologi …
          ⊢ IsCompact ((TopologicalSpace.Opens.map f.base).obj Top.top).carrier
        -/
      rw [← IsAffineOpen.basicOpen_union_eq_self_iff] at hS
        /-
          case of_basicOpenCover
          X✝ Y✝ : AlgebraicGeometry.Scheme
          f✝ : Quiver.Hom X✝ Y✝
          X Y : AlgebraicGeometry.Scheme
          H : AlgebraicGeometry.IsAffine Y
          f : Quiver.Hom X Y
          S : Finset ↑(Y.presheaf.obj { unop := Top.top })
          hS : Eq (iSup fun f => Y.basicOpen ↑f) Top.top
          hS' : ∀ (r : Subtype fun x => Membership.mem S x), CompactSpace ↑↑(↑((Topologi …
          ⊢ IsCompact ((TopologicalSpace.Opens.map f.base).obj (iSup fun f => Y.basicOpe …
        -/
      · rw [← isCompact_univ_iff]
        /-
          case of_basicOpenCover
          X✝ Y✝ : AlgebraicGeometry.Scheme
          f✝ : Quiver.Hom X✝ Y✝
          X Y : AlgebraicGeometry.Scheme
          H : AlgebraicGeometry.IsAffine Y
          f : Quiver.Hom X Y
          S : Finset ↑(Y.presheaf.obj { unop := Top.top })
          hS : Eq (iSup fun f => Y.basicOpen ↑f) Top.top
          hS' : ∀ (r : Subtype fun x => Membership.mem S x), CompactSpace ↑↑(↑((Topologi …
          ⊢ IsCompact (Set.preimage ⇑f.base ↑(iSup fun f => Y.basicOpen ↑f))
        -/
        change IsCompact ((Opens.map f.base).obj ⊤).1
        /-
          case of_basicOpenCover
          X✝ Y✝ : AlgebraicGeometry.Scheme
          f✝ : Quiver.Hom X✝ Y✝
          X Y : AlgebraicGeometry.Scheme
          H : AlgebraicGeometry.IsAffine Y
          f : Quiver.Hom X Y
          S : Finset ↑(Y.presheaf.obj { unop := Top.top })
          hS : Eq (iSup fun f => Y.basicOpen ↑f) Top.top
          hS' : ∀ (r : Subtype fun x => Membership.mem S x), CompactSpace ↑↑(↑((Topologi …
          ⊢ IsCompact (Set.iUnion fun i => Set.preimage (⇑f.base) (Y.basicOpen ↑i).carri …
        -/
        rw [← hS]
        /-
          🎉 no goals
        -/
        /-
          case of_basicOpenCover.hU
          X✝ Y✝ : AlgebraicGeometry.Scheme
          f✝ : Quiver.Hom X✝ Y✝
          X Y : AlgebraicGeometry.Scheme
          H : AlgebraicGeometry.IsAffine Y
          f : Quiver.Hom X Y
          S : Finset ↑(Y.presheaf.obj { unop := Top.top })
          hS : Eq (Ideal.span ↑S) Top.top
          hS' : ∀ (r : Subtype fun x => Membership.mem S x), CompactSpace ↑↑(↑((Topologi …
          ⊢ AlgebraicGeometry.IsAffineOpen Top.top
        -/
        dsimp [Opens.map]
        /-
          🎉 no goals
        -/
        simp only [Opens.iSup_mk, Opens.coe_mk, Set.preimage_iUnion]
        exact isCompact_iUnion fun i => isCompact_iff_compactSpace.mpr (hS' i)
      · exact isAffineOpen_top _


theorem quasiCompact_over_affine_iff {X Y : Scheme} (f : X ⟶ Y) [IsAffine Y] :
    QuasiCompact f ↔ CompactSpace X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (AlgebraicGeometry.QuasiCompact f) (CompactSpace ↑↑X.toPresheafedSpace)
  -/
  rw [HasAffineProperty.iff_of_isAffine (P := @QuasiCompact)]
  /-
    🎉 no goals
  -/


theorem compactSpace_iff_quasiCompact (X : Scheme) :
    CompactSpace X ↔ QuasiCompact (terminal.from X) := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Iff (CompactSpace ↑↑X.toPresheafedSpace) (AlgebraicGeometry.QuasiCompact (Ca …
  -/
  rw [HasAffineProperty.iff_of_isAffine (P := @QuasiCompact)]
  /-
    🎉 no goals
  -/


instance quasiCompact_isStableUnderComposition :
    MorphismProperty.IsStableUnderComposition @QuasiCompact where
  comp_mem _ _ _ _ := inferInstance


instance quasiCompact_isStableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange @QuasiCompact := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.Q …
  -/
  letI := HasAffineProperty.isLocal_affineProperty @QuasiCompact
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x => …
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.Q …
  -/
  apply HasAffineProperty.isStableUnderBaseChange
  /-
    case hP'
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x => …
    ⊢ AlgebraicGeometry.AffineTargetMorphismProperty.IsStableUnderBaseChange fun X …
  -/
  apply AffineTargetMorphismProperty.IsStableUnderBaseChange.mk
  /-
    case hP'.H
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x => …
    ⊢ ∀ ⦃X Y S : AlgebraicGeometry.Scheme⦄ [inst : AlgebraicGeometry.IsAffine S] [ …
  -/
  intro X Y S _ _ f g h
  /-
    case hP'.H
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    this : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x => …
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    h : CompactSpace ↑↑Y.toPresheafedSpace
    ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
  -/
  let 𝒰 := Scheme.Pullback.openCoverOfRight Y.affineCover.finiteSubcover f g
  /-
    case hP'.H
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    this : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x => …
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    h : CompactSpace ↑↑Y.toPresheafedSpace
    𝒰 : (CategoryTheory.Limits.pullback f g).OpenCover := AlgebraicGeometry.Scheme …
    ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
  -/
  have : Finite 𝒰.J := by dsimp [𝒰]; infer_instance
  /-
    case hP'.H
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    this✝ : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x = …
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    h : CompactSpace ↑↑Y.toPresheafedSpace
    𝒰 : (CategoryTheory.Limits.pullback f g).OpenCover := AlgebraicGeometry.Scheme …
    this : Finite 𝒰.J
    ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
  -/
  have : ∀ i, CompactSpace (𝒰.obj i) := by intro i; dsimp [𝒰]; infer_instance
  /-
    case hP'.H
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    this✝¹ : AlgebraicGeometry.AffineTargetMorphismProperty.IsLocal fun X x x_1 x  …
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    h : CompactSpace ↑↑Y.toPresheafedSpace
    𝒰 : (CategoryTheory.Limits.pullback f g).OpenCover := AlgebraicGeometry.Scheme …
    this✝ : Finite 𝒰.J
    this : ∀ (i : 𝒰.J), CompactSpace ↑↑(𝒰.obj i).toPresheafedSpace
    ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
  -/
  exact 𝒰.compactSpace
  /-
    🎉 no goals
  -/


instance (f : X ⟶ Z) (g : Y ⟶ Z) [QuasiCompact g] : QuasiCompact (pullback.fst f g) :=
  MorphismProperty.pullback_fst f g inferInstance


instance (f : X ⟶ Z) (g : Y ⟶ Z) [QuasiCompact f] : QuasiCompact (pullback.snd f g) :=
  MorphismProperty.pullback_snd f g inferInstance


lemma compactSpace_iff_exists :
    CompactSpace X ↔ ∃ R, ∃ f : Spec R ⟶ X, Function.Surjective f.base := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Iff (CompactSpace ↑↑X.toPresheafedSpace) (Exists fun R => Exists fun f => Fu …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨R, f, hf⟩ ↦ ⟨hf.range_eq ▸ isCompact_range f.continuous⟩⟩
  /-
    X : AlgebraicGeometry.Scheme
    h : CompactSpace ↑↑X.toPresheafedSpace
    ⊢ Exists fun R => Exists fun f => Function.Surjective ⇑f.base
  -/
  let 𝒰 : X.OpenCover := X.affineCover.finiteSubcover
  /-
    X : AlgebraicGeometry.Scheme
    h : CompactSpace ↑↑X.toPresheafedSpace
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    ⊢ Exists fun R => Exists fun f => Function.Surjective ⇑f.base
  -/
  have (x : 𝒰.J) : IsAffine (𝒰.obj x) := X.isAffine_affineCover _
  /-
    X : AlgebraicGeometry.Scheme
    h : CompactSpace ↑↑X.toPresheafedSpace
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (x : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj x)
    ⊢ Exists fun R => Exists fun f => Function.Surjective ⇑f.base
  -/
  refine ⟨Γ(∐ 𝒰.obj, ⊤), (∐ 𝒰.obj).isoSpec.inv ≫ Sigma.desc 𝒰.map, ?_⟩
  refine Function.Surjective.comp (g := (Sigma.desc 𝒰.map).base)
    (fun x ↦ ?_) (∐ 𝒰.obj).isoSpec.inv.surjective
  /-
    X : AlgebraicGeometry.Scheme
    h : CompactSpace ↑↑X.toPresheafedSpace
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (x : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj x)
    x : ↑↑X.toPresheafedSpace
    ⊢ Exists fun a => Eq ((CategoryTheory.Limits.Sigma.desc 𝒰.map).base a) x
  -/
  obtain ⟨y, hy⟩ := 𝒰.covers x
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    h : CompactSpace ↑↑X.toPresheafedSpace
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    this : ∀ (x : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj x)
    x : ↑↑X.toPresheafedSpace
    y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace
    hy : Eq ((𝒰.map (𝒰.f x)).base y) x
    ⊢ Exists fun a => Eq ((CategoryTheory.Limits.Sigma.desc 𝒰.map).base a) x
  -/
  exact ⟨(Sigma.ι 𝒰.obj (𝒰.f x)).base y, by rw [← Scheme.comp_base_apply, Sigma.ι_desc, hy]⟩
  /-
    🎉 no goals
  -/


lemma isCompact_iff_exists {U : X.Opens} :
    IsCompact (U : Set X) ↔ ∃ R, ∃ f : Spec R ⟶ X, Set.range f.base = U := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Iff (IsCompact ↑U) (Exists fun R => Exists fun f => Eq (Set.range ⇑f.base) ↑U)
  -/
  refine isCompact_iff_compactSpace.trans ((compactSpace_iff_exists (X := U)).trans ?_)
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    ⊢ Iff (Exists fun R => Exists fun f => Function.Surjective ⇑f.base) (Exists fu …
  -/
  refine ⟨fun ⟨R, f, hf⟩ ↦ ⟨R, f ≫ U.ι, by simp [hf.range_comp]⟩, fun ⟨R, f, hf⟩ ↦ ?_⟩
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x✝ : Exists fun R => Exists fun f => Eq (Set.range ⇑f.base) ↑U
    R : CommRingCat
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    hf : Eq (Set.range ⇑f.base) ↑U
    ⊢ Exists fun R => Exists fun f => Function.Surjective ⇑f.base
  -/
  refine ⟨R, IsOpenImmersion.lift U.ι f (by simp [hf]), ?_⟩
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x✝ : Exists fun R => Exists fun f => Eq (Set.range ⇑f.base) ↑U
    R : CommRingCat
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    hf : Eq (Set.range ⇑f.base) ↑U
    ⊢ Function.Surjective ⇑(AlgebraicGeometry.IsOpenImmersion.lift U.ι f ⋯).base
  -/
  rw [← Set.range_eq_univ]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x✝ : Exists fun R => Exists fun f => Eq (Set.range ⇑f.base) ↑U
    R : CommRingCat
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    hf : Eq (Set.range ⇑f.base) ↑U
    ⊢ Eq (Set.range ⇑(AlgebraicGeometry.IsOpenImmersion.lift U.ι f ⋯).base) Set.univ
  -/
  apply show Function.Injective (U.ι.base '' ·) from Set.image_val_injective
  /-
    case a
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x✝ : Exists fun R => Exists fun f => Eq (Set.range ⇑f.base) ↑U
    R : CommRingCat
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    hf : Eq (Set.range ⇑f.base) ↑U
    ⊢ Eq ((fun x => Set.image (⇑U.ι.base) x) (Set.range ⇑(AlgebraicGeometry.IsOpen …
  -/
  simp only [Set.image_univ, Scheme.Opens.range_ι]
  /-
    case a
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    x✝ : Exists fun R => Exists fun f => Eq (Set.range ⇑f.base) ↑U
    R : CommRingCat
    f : Quiver.Hom (AlgebraicGeometry.Spec R) X
    hf : Eq (Set.range ⇑f.base) ↑U
    ⊢ Eq (Set.image (⇑U.ι.base) (Set.range ⇑(AlgebraicGeometry.IsOpenImmersion.lif …
  -/
  rwa [← Set.range_comp, ← TopCat.coe_comp, ← Scheme.comp_base, IsOpenImmersion.lift_fac]
  /-
    🎉 no goals
  -/


@[stacks 01K9]
lemma isClosedMap_iff_specializingMap (f : X ⟶ Y) [QuasiCompact f] :
    IsClosedMap f.base ↔ SpecializingMap f.base := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.QuasiCompact f
    ⊢ Iff (IsClosedMap ⇑f.base) (SpecializingMap ⇑f.base)
  -/
  refine ⟨fun h ↦ h.specializingMap, fun H ↦ ?_⟩
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.QuasiCompact f
    H : SpecializingMap ⇑f.base
    ⊢ IsClosedMap ⇑f.base
  -/
  wlog hY : ∃ R, Y = Spec R
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiCompact f
      H : SpecializingMap ⇑f.base
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
      hY : Not (Exists fun R => Eq Y (AlgebraicGeometry.Spec R))
      ⊢ IsClosedMap ⇑f.base
    -/
  · show topologically @IsClosedMap f
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiCompact f
      H : SpecializingMap ⇑f.base
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
      hY : Not (Exists fun R => Eq Y (AlgebraicGeometry.Spec R))
      ⊢ AlgebraicGeometry.topologically (@IsClosedMap) f
    -/
    rw [IsLocalAtTarget.iff_of_openCover (P := topologically @IsClosedMap) Y.affineCover]
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiCompact f
      H : SpecializingMap ⇑f.base
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
      hY : Not (Exists fun R => Eq Y (AlgebraicGeometry.Spec R))
      ⊢ ∀ (i : Y.affineCover.1), AlgebraicGeometry.topologically (@IsClosedMap) (Alg …
    -/
    intro i
    haveI hqc : QuasiCompact (Y.affineCover.pullbackHom f i) :=
        IsLocalAtTarget.of_isPullback (.of_hasPullback _ _) inferInstance
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiCompact f
      H : SpecializingMap ⇑f.base
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebra …
      hY : Not (Exists fun R => Eq Y (AlgebraicGeometry.Spec R))
      i : Y.affineCover.1
      hqc : AlgebraicGeometry.QuasiCompact (AlgebraicGeometry.Scheme.Cover.pullbackH …
      ⊢ AlgebraicGeometry.topologically (@IsClosedMap) (AlgebraicGeometry.Scheme.Cov …
    -/
    refine this (Y.affineCover.pullbackHom f i) ?_ ⟨_, rfl⟩
    exact IsLocalAtTarget.of_isPullback
      (P := topologically @SpecializingMap) (.of_hasPullback _ _) H
  /-
    X✝ Y✝ X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.QuasiCompact f
    H : SpecializingMap ⇑f.base
    hY : Exists fun R => Eq Y (AlgebraicGeometry.Spec R)
    ⊢ IsClosedMap ⇑f.base
  -/
  obtain ⟨S, rfl⟩ := hY
  /-
    case intro
    X✝ Y X : AlgebraicGeometry.Scheme
    S : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec S)
    inst✝ : AlgebraicGeometry.QuasiCompact f
    H : SpecializingMap ⇑f.base
    ⊢ IsClosedMap ⇑f.base
  -/
  clear * - H
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    S : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec S)
    inst✝ : AlgebraicGeometry.QuasiCompact f
    H : SpecializingMap ⇑f.base
    ⊢ IsClosedMap ⇑f.base
  -/
  intros Z hZ
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    S : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec S)
    inst✝ : AlgebraicGeometry.QuasiCompact f
    H : SpecializingMap ⇑f.base
    Z : Set ↑↑X.toPresheafedSpace
    hZ : IsClosed Z
    ⊢ IsClosed (Set.image (⇑f.base) Z)
  -/
  replace H := hZ.stableUnderSpecialization.image H
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    S : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec S)
    inst✝ : AlgebraicGeometry.QuasiCompact f
    Z : Set ↑↑X.toPresheafedSpace
    hZ : IsClosed Z
    H : StableUnderSpecialization (Set.image (⇑f.base) Z)
    ⊢ IsClosed (Set.image (⇑f.base) Z)
  -/
  wlog hX : ∃ R, X = Spec R
  · obtain ⟨R, g, hg⟩ :=
      compactSpace_iff_exists.mp ((quasiCompact_over_affine_iff f).mp inferInstance)
    /-
      case intro.inr.intro.intro
      X : AlgebraicGeometry.Scheme
      S : CommRingCat
      f : Quiver.Hom X (AlgebraicGeometry.Spec S)
      inst✝ : AlgebraicGeometry.QuasiCompact f
      Z : Set ↑↑X.toPresheafedSpace
      hZ : IsClosed Z
      H : StableUnderSpecialization (Set.image (⇑f.base) Z)
      this : ∀ {X : AlgebraicGeometry.Scheme} (S : CommRingCat) (f : Quiver.Hom X (A …
      hX : Not (Exists fun R => Eq X (AlgebraicGeometry.Spec R))
      R : CommRingCat
      g : Quiver.Hom (AlgebraicGeometry.Spec R) X
      hg : Function.Surjective ⇑g.base
      ⊢ IsClosed (Set.image (⇑f.base) Z)
    -/
    have inst : QuasiCompact (g ≫ f) := HasAffineProperty.iff_of_isAffine.mpr (by infer_instance)
    /-
      case intro.inr.intro.intro
      X : AlgebraicGeometry.Scheme
      S : CommRingCat
      f : Quiver.Hom X (AlgebraicGeometry.Spec S)
      inst✝ : AlgebraicGeometry.QuasiCompact f
      Z : Set ↑↑X.toPresheafedSpace
      hZ : IsClosed Z
      H : StableUnderSpecialization (Set.image (⇑f.base) Z)
      this : ∀ {X : AlgebraicGeometry.Scheme} (S : CommRingCat) (f : Quiver.Hom X (A …
      hX : Not (Exists fun R => Eq X (AlgebraicGeometry.Spec R))
      R : CommRingCat
      g : Quiver.Hom (AlgebraicGeometry.Spec R) X
      hg : Function.Surjective ⇑g.base
      inst : AlgebraicGeometry.QuasiCompact (CategoryTheory.CategoryStruct.comp g f)
      ⊢ IsClosed (Set.image (⇑f.base) Z)
    -/
    have := this _ (g ≫ f) (g.base ⁻¹' Z) (hZ.preimage g.continuous)
    simp_rw [Scheme.comp_base, TopCat.comp_app, ← Set.image_image,
      Set.image_preimage_eq _ hg] at this
    /-
      case intro.inr.intro.intro
      X : AlgebraicGeometry.Scheme
      S : CommRingCat
      f : Quiver.Hom X (AlgebraicGeometry.Spec S)
      inst✝ : AlgebraicGeometry.QuasiCompact f
      Z : Set ↑↑X.toPresheafedSpace
      hZ : IsClosed Z
      H : StableUnderSpecialization (Set.image (⇑f.base) Z)
      this✝ : ∀ {X : AlgebraicGeometry.Scheme} (S : CommRingCat) (f : Quiver.Hom X ( …
      hX : Not (Exists fun R => Eq X (AlgebraicGeometry.Spec R))
      R : CommRingCat
      g : Quiver.Hom (AlgebraicGeometry.Spec R) X
      hg : Function.Surjective ⇑g.base
      inst : AlgebraicGeometry.QuasiCompact (CategoryTheory.CategoryStruct.comp g f)
      this : StableUnderSpecialization (Set.image (⇑f.base) Z) → (Exists fun R_1 =>  …
      ⊢ IsClosed (Set.image (⇑f.base) Z)
    -/
    exact this H ⟨_, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    X : AlgebraicGeometry.Scheme
    S : CommRingCat
    f : Quiver.Hom X (AlgebraicGeometry.Spec S)
    inst✝ : AlgebraicGeometry.QuasiCompact f
    Z : Set ↑↑X.toPresheafedSpace
    hZ : IsClosed Z
    H : StableUnderSpecialization (Set.image (⇑f.base) Z)
    hX : Exists fun R => Eq X (AlgebraicGeometry.Spec R)
    ⊢ IsClosed (Set.image (⇑f.base) Z)
  -/
  obtain ⟨R, rfl⟩ := hX
  /-
    case intro
    S R : CommRingCat
    f : Quiver.Hom (AlgebraicGeometry.Spec R) (AlgebraicGeometry.Spec S)
    inst✝ : AlgebraicGeometry.QuasiCompact f
    Z : Set ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    hZ : IsClosed Z
    H : StableUnderSpecialization (Set.image (⇑f.base) Z)
    ⊢ IsClosed (Set.image (⇑f.base) Z)
  -/
  obtain ⟨φ, rfl⟩ := Spec.homEquiv.symm.surjective f
  /-
    case intro.intro
    S R : CommRingCat
    Z : Set ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
    hZ : IsClosed Z
    φ : Quiver.Hom S R
    inst✝ : AlgebraicGeometry.QuasiCompact (AlgebraicGeometry.Spec.homEquiv.symm φ)
    H : StableUnderSpecialization (Set.image (⇑(AlgebraicGeometry.Spec.homEquiv.sy …
    ⊢ IsClosed (Set.image (⇑(AlgebraicGeometry.Spec.homEquiv.symm φ).base) Z)
  -/
  exact PrimeSpectrum.isClosed_image_of_stableUnderSpecialization φ.hom Z hZ H
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem compact_open_induction_on {P : X.Opens → Prop} (S : X.Opens)
    (hS : IsCompact S.1) (h₁ : P ⊥)
    (h₂ : ∀ (S : X.Opens) (_ : IsCompact S.1) (U : X.affineOpens), P S → P (S ⊔ U)) :
    P S := by
  classical
  obtain ⟨s, hs, hs'⟩ := (isCompactOpen_iff_eq_finset_affine_union S.1).mp ⟨hS, S.2⟩
  replace hs' : S = iSup fun i : s => (i : X.Opens) := by ext1; simpa using hs'
  subst hs'
  apply @Set.Finite.induction_on _ _ _ hs
  · convert h₁; rw [iSup_eq_bot]; rintro ⟨_, h⟩; exact h.elim
  · intro x s _ hs h₄
    have : IsCompact (⨆ i : s, (i : X.Opens)).1 := by
      refine ((isCompactOpen_iff_eq_finset_affine_union _).mpr ?_).1; exact ⟨s, hs, by simp⟩
    convert h₂ _ this x h₄
    rw [iSup_subtype, sup_comm]
    conv_rhs => rw [iSup_subtype]
    exact iSup_insert


theorem exists_pow_mul_eq_zero_of_res_basicOpen_eq_zero_of_isAffineOpen (X : Scheme)
    {U : X.Opens} (hU : IsAffineOpen U) (x f : Γ(X, U))
    (H : x |_ᵣ (X.basicOpen f) = 0) :
    ∃ n : ℕ, f ^ n * x = 0 := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow f n) x) 0
  -/
  rw [← map_zero (X.presheaf.map (homOfLE <| X.basicOpen_le f : X.basicOpen f ⟶ U).op).hom] at H
  #adaptation_note
  /--
  Prior to nightly-2024-09-29, we could use dot notation here:
  `(hU.isLocalization_basicOpen f).exists_of_eq H`
  This is no longer possible;
  likely changing the signature of `IsLocalization.Away.exists_of_eq` is in order.
  -/
  obtain ⟨n, e⟩ :=
    @IsLocalization.Away.exists_of_eq _ _ _ _ _ _ (hU.isLocalization_basicOpen f) _ _ H
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) ((X.presh …
    n : Nat
    e : Eq (HMul.hMul (HPow.hPow f n) x) (HMul.hMul (HPow.hPow f n) 0)
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow f n) x) 0
  -/
  exact ⟨n, by simpa [mul_comm x] using e⟩
  /-
    🎉 no goals
  -/


/-- If `x : Γ(X, U)` is zero on `D(f)` for some `f : Γ(X, U)`, and `U` is quasi-compact, then
`f ^ n * x = 0` for some `n`. -/
theorem exists_pow_mul_eq_zero_of_res_basicOpen_eq_zero_of_isCompact (X : Scheme.{u})
    {U : X.Opens} (hU : IsCompact U.1) (x f : Γ(X, U))
    (H : x |_ᵣ (X.basicOpen f) = 0) :
    ∃ n : ℕ, f ^ n * x = 0 := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow f n) x) 0
  -/
  obtain ⟨s, hs, e⟩ := (isCompactOpen_iff_eq_finset_affine_union U.1).mp ⟨hU, U.2⟩
  replace e : U = iSup fun i : s => (i : X.Opens) := by
    ext1; simpa using e
  have h₁ : ∀ i : s, i.1.1 ≤ U := by
    intro i
    change (i : X.Opens) ≤ U
    rw [e]
    -- Porting note: `exact le_iSup _ _` no longer works
    exact le_iSup (fun (i : s) => (i : Opens (X.toPresheafedSpace))) _
  have H' := fun i : s =>
    exists_pow_mul_eq_zero_of_res_basicOpen_eq_zero_of_isAffineOpen X i.1.2
      (X.presheaf.map (homOfLE (h₁ i)).op x) (X.presheaf.map (homOfLE (h₁ i)).op f) ?_
  /-
    case intro.intro.refine_2
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    H' : ∀ (i : ↑s), Exists fun n => Eq (HMul.hMul (HPow.hPow ((X.presheaf.map (Ca …
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow f n) x) 0
  -/
  swap
    /-
      case intro.intro.refine_1
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      x f : ↑(X.presheaf.obj { unop := U })
      H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
      s : Set ↑X.affineOpens
      hs : s.Finite
      e : Eq U (iSup fun i => ↑↑i)
      h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
      i : ↑s
      ⊢ Eq (TopCat.Presheaf.restrictOpenCommRingCat ((X.presheaf.map (CategoryTheory …
    -/
  · show (X.presheaf.map (homOfLE _).op) ((X.presheaf.map (homOfLE _).op).hom x) = 0
    /-
      case intro.intro.refine_1
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      x f : ↑(X.presheaf.obj { unop := U })
      H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
      s : Set ↑X.affineOpens
      hs : s.Finite
      e : Eq U (iSup fun i => ↑↑i)
      h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
      i : ↑s
      ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom ((X.presheaf.map (Cat …
    -/
    have H : (X.presheaf.map (homOfLE _).op) x = 0 := H
    /-
      case intro.intro.refine_1
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      x f : ↑(X.presheaf.obj { unop := U })
      H✝ : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
      s : Set ↑X.affineOpens
      hs : s.Finite
      e : Eq U (iSup fun i => ↑↑i)
      h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
      i : ↑s
      H : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom x) 0
      ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom ((X.presheaf.map (Cat …
    -/
    convert congr_arg (X.presheaf.map (homOfLE _).op).hom H
      /-
        case h.e'_2
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : IsCompact U.carrier
        x f : ↑(X.presheaf.obj { unop := U })
        H✝ : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
        s : Set ↑X.affineOpens
        hs : s.Finite
        e : Eq U (iSup fun i => ↑↑i)
        h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
        i : ↑s
        H : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom x) 0
        ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom ((X.presheaf.map (Cat …
      -/
    · simp only [← CommRingCat.comp_apply, ← Functor.map_comp]
        /-
          case h.e'_2
          X : AlgebraicGeometry.Scheme
          U : X.Opens
          hU : IsCompact U.carrier
          x f : ↑(X.presheaf.obj { unop := U })
          H✝ : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
          s : Set ↑X.affineOpens
          hs : s.Finite
          e : Eq U (iSup fun i => ↑↑i)
          h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
          i : ↑s
          H : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom x) 0
          ⊢ Eq ((X.presheaf.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homO …
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case h.e'_3
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : IsCompact U.carrier
        x f : ↑(X.presheaf.obj { unop := U })
        H✝ : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
        s : Set ↑X.affineOpens
        hs : s.Finite
        e : Eq U (iSup fun i => ↑↑i)
        h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
        i : ↑s
        H : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom x) 0
        ⊢ Eq 0 ((X.presheaf.map (CategoryTheory.homOfLE ?intro.intro.refine_1.convert_ …
      -/
    · rw [map_zero]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_1.convert_1
        X : AlgebraicGeometry.Scheme
        U : X.Opens
        hU : IsCompact U.carrier
        x f : ↑(X.presheaf.obj { unop := U })
        H✝ : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
        s : Set ↑X.affineOpens
        hs : s.Finite
        e : Eq U (iSup fun i => ↑↑i)
        h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
        i : ↑s
        H : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom x) 0
        ⊢ LE.le (X.basicOpen ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom f)) ( …
      -/
    · simp only [Scheme.basicOpen_res, inf_le_right]
      /-
        🎉 no goals
      -/
  /-
    case intro.intro.refine_2
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    H' : ∀ (i : ↑s), Exists fun n => Eq (HMul.hMul (HPow.hPow ((X.presheaf.map (Ca …
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow f n) x) 0
  -/
  choose n hn using H'
  /-
    case intro.intro.refine_2
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    n : ↑s → Nat
    hn : ∀ (i : ↑s), Eq (HMul.hMul (HPow.hPow ((X.presheaf.map (CategoryTheory.hom …
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow f n) x) 0
  -/
  haveI := hs.to_subtype
  /-
    case intro.intro.refine_2
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    n : ↑s → Nat
    hn : ∀ (i : ↑s), Eq (HMul.hMul (HPow.hPow ((X.presheaf.map (CategoryTheory.hom …
    this : Finite ↑s
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow f n) x) 0
  -/
  cases nonempty_fintype s
  /-
    case intro.intro.refine_2.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    n : ↑s → Nat
    hn : ∀ (i : ↑s), Eq (HMul.hMul (HPow.hPow ((X.presheaf.map (CategoryTheory.hom …
    this : Finite ↑s
    val✝ : Fintype ↑s
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow f n) x) 0
  -/
  use Finset.univ.sup n
  suffices ∀ i : s, X.presheaf.map (homOfLE (h₁ i)).op (f ^ Finset.univ.sup n * x) = 0 by
    subst e
    apply TopCat.Sheaf.eq_of_locally_eq X.sheaf fun i : s => (i : X.Opens)
    intro i
    show _ = (X.sheaf.val.map _) 0
    rw [map_zero]
    apply this
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    n : ↑s → Nat
    hn : ∀ (i : ↑s), Eq (HMul.hMul (HPow.hPow ((X.presheaf.map (CategoryTheory.hom …
    this : Finite ↑s
    val✝ : Fintype ↑s
    ⊢ ∀ (i : ↑s), Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMu …
  -/
  intro i
  replace hn :=
    congr_arg (fun x => X.presheaf.map (homOfLE (h₁ i)).op (f ^ (Finset.univ.sup n - n i)) * x)
      (hn i)
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    n : ↑s → Nat
    this : Finite ↑s
    val✝ : Fintype ↑s
    i : ↑s
    hn : Eq ((fun x => HMul.hMul ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).h …
    ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow.hPow …
  -/
  dsimp at hn
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    n : ↑s → Nat
    this : Finite ↑s
    val✝ : Fintype ↑s
    i : ↑s
    hn : Eq (HMul.hMul ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HPow.h …
    ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow.hPow …
  -/
  simp only [← map_mul, ← map_pow] at hn
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    n : ↑s → Nat
    this : Finite ↑s
    val✝ : Fintype ↑s
    i : ↑s
    hn : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow.h …
    ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow.hPow …
  -/
  rwa [mul_zero, ← mul_assoc, ← pow_add, tsub_add_cancel_of_le] at hn
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    x f : ↑(X.presheaf.obj { unop := U })
    H : Eq (TopCat.Presheaf.restrictOpenCommRingCat x (X.basicOpen f) ⋯) 0
    s : Set ↑X.affineOpens
    hs : s.Finite
    e : Eq U (iSup fun i => ↑↑i)
    h₁ : ∀ (i : ↑s), LE.le (↑↑i) U
    n : ↑s → Nat
    this : Finite ↑s
    val✝ : Fintype ↑s
    i : ↑s
    hn : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow.h …
    ⊢ LE.le (n i) (Finset.univ.sup n)
  -/
  apply Finset.le_sup (Finset.mem_univ i)
  /-
    🎉 no goals
  -/


/-- A section over a compact open of a scheme is nilpotent if and only if its associated
basic open is empty. -/
lemma Scheme.isNilpotent_iff_basicOpen_eq_bot_of_isCompact {X : Scheme.{u}}
    {U : X.Opens} (hU : IsCompact (U : Set X)) (f : Γ(X, U)) :
    IsNilpotent f ↔ X.basicOpen f = ⊥ := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact ↑U
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (IsNilpotent f) (Eq (X.basicOpen f) Bot.bot)
  -/
  refine ⟨X.basicOpen_eq_bot_of_isNilpotent U f, fun hf ↦ ?_⟩
  have h : (1 : Γ(X, U)) |_ᵣ (X.basicOpen f) = 0 := by
    have e : X.basicOpen f ≤ ⊥ := by rw [hf]
    rw [← CommRingCat.presheaf_restrict_restrict X e bot_le]
    have : Subsingleton Γ(X, ⊥) :=
      CommRingCat.subsingleton_of_isTerminal X.sheaf.isTerminalOfEmpty
    rw [Subsingleton.eq_zero (1 |_ᵣ ⊥)]
    show X.presheaf.map _ 0 = 0
    rw [map_zero]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact ↑U
    f : ↑(X.presheaf.obj { unop := U })
    hf : Eq (X.basicOpen f) Bot.bot
    h : Eq (TopCat.Presheaf.restrictOpenCommRingCat 1 (X.basicOpen f) ⋯) 0
    ⊢ IsNilpotent f
  -/
  obtain ⟨n, hn⟩ := exists_pow_mul_eq_zero_of_res_basicOpen_eq_zero_of_isCompact X hU 1 f h
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact ↑U
    f : ↑(X.presheaf.obj { unop := U })
    hf : Eq (X.basicOpen f) Bot.bot
    h : Eq (TopCat.Presheaf.restrictOpenCommRingCat 1 (X.basicOpen f) ⋯) 0
    n : Nat
    hn : Eq (HMul.hMul (HPow.hPow f n) 1) 0
    ⊢ IsNilpotent f
  -/
  rw [mul_one] at hn
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact ↑U
    f : ↑(X.presheaf.obj { unop := U })
    hf : Eq (X.basicOpen f) Bot.bot
    h : Eq (TopCat.Presheaf.restrictOpenCommRingCat 1 (X.basicOpen f) ⋯) 0
    n : Nat
    hn : Eq (HPow.hPow f n) 0
    ⊢ IsNilpotent f
  -/
  use n
  /-
    🎉 no goals
  -/


/-- A global section of a quasi-compact scheme is nilpotent if and only if its associated
basic open is empty. -/
lemma Scheme.isNilpotent_iff_basicOpen_eq_bot {X : Scheme.{u}}
    [CompactSpace X] (f : Γ(X, ⊤)) :
    IsNilpotent f ↔ X.basicOpen f = ⊥ :=
  isNilpotent_iff_basicOpen_eq_bot_of_isCompact (U := ⊤) (CompactSpace.isCompact_univ) f


/-- The zero locus of a set of sections over a compact open of a scheme is `X` if and only if
`s` is contained in the nilradical of `Γ(X, U)`. -/
lemma Scheme.zeroLocus_eq_top_iff_subset_nilradical_of_isCompact {X : Scheme.{u}} {U : X.Opens}
    (hU : IsCompact (U : Set X)) (s : Set Γ(X, U)) :
    X.zeroLocus s = ⊤ ↔ s ⊆ nilradical Γ(X, U) := by
  simp [Scheme.zeroLocus_def, ← Scheme.isNilpotent_iff_basicOpen_eq_bot_of_isCompact hU,
    ← mem_nilradical, Set.subset_def]


/-- The zero locus of a set of sections over a compact open of a scheme is `X` if and only if
`s` is contained in the nilradical of `Γ(X, U)`. -/
lemma Scheme.zeroLocus_eq_top_iff_subset_nilradical {X : Scheme.{u}}
    [CompactSpace X] (s : Set Γ(X, ⊤)) :
    X.zeroLocus s = ⊤ ↔ s ⊆ nilradical Γ(X, ⊤) :=
  zeroLocus_eq_top_iff_subset_nilradical_of_isCompact (U := ⊤) (CompactSpace.isCompact_univ) s


