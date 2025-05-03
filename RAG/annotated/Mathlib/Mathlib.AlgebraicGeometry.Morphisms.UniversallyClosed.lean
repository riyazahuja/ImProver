/-- A morphism of schemes `f : X ⟶ Y` is universally closed if the base change `X ×[Y] Y' ⟶ Y'`
along any morphism `Y' ⟶ Y` is (topologically) a closed map.
-/
@[mk_iff]
class UniversallyClosed (f : X ⟶ Y) : Prop where
  out : universally (topologically @IsClosedMap) f


lemma Scheme.Hom.isClosedMap {X Y : Scheme} (f : X.Hom Y) [UniversallyClosed f] :
    IsClosedMap f.base := UniversallyClosed.out _ _ _ IsPullback.of_id_snd


theorem universallyClosed_eq : @UniversallyClosed = universally (topologically @IsClosedMap) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.UniversallyClosed) (AlgebraicGeometry.topologically @ …
  -/
  ext X Y f; rw [universallyClosed_iff]
             /-
               🎉 no goals
             -/


instance (priority := 900) [IsClosedImmersion f] : UniversallyClosed f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsClosedImmersion f
    ⊢ AlgebraicGeometry.UniversallyClosed f
  -/
  rw [universallyClosed_eq]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsClosedImmersion f
    ⊢ (AlgebraicGeometry.topologically @IsClosedMap).universally f
  -/
  intro X' Y' i₁ i₂ f' hf
  have hf' : IsClosedImmersion f' :=
    MorphismProperty.of_isPullback hf.flip inferInstance
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsClosedImmersion f
    X' Y' : AlgebraicGeometry.Scheme
    i₁ : Quiver.Hom X' X
    i₂ : Quiver.Hom Y' Y
    f' : Quiver.Hom X' Y'
    hf : CategoryTheory.IsPullback f' i₁ i₂ f
    hf' : AlgebraicGeometry.IsClosedImmersion f'
    ⊢ AlgebraicGeometry.topologically (@IsClosedMap) f'
  -/
  exact hf'.base_closed.isClosedMap
  /-
    🎉 no goals
  -/


theorem universallyClosed_respectsIso : RespectsIso @UniversallyClosed :=
  universallyClosed_eq.symm ▸ universally_respectsIso (topologically @IsClosedMap)


instance universallyClosed_isStableUnderBaseChange : IsStableUnderBaseChange @UniversallyClosed :=
  universallyClosed_eq.symm ▸ universally_isStableUnderBaseChange (topologically @IsClosedMap)


instance isClosedMap_isStableUnderComposition :
    IsStableUnderComposition (topologically @IsClosedMap) where
  comp_mem f g hf hg := IsClosedMap.comp (f := f.base) (g := g.base) hg hf


instance universallyClosed_isStableUnderComposition :
    IsStableUnderComposition @UniversallyClosed := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderComposition @AlgebraicGeometry. …
  -/
  rw [universallyClosed_eq]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ (AlgebraicGeometry.topologically @IsClosedMap).universally.IsStableUnderComp …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma UniversallyClosed.of_comp_surjective {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z)
    [UniversallyClosed (f ≫ g)] [Surjective f] : UniversallyClosed g := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.UniversallyClosed (CategoryTheory.CategoryStruct.co …
    inst✝ : AlgebraicGeometry.Surjective f
    ⊢ AlgebraicGeometry.UniversallyClosed g
  -/
  constructor
  /-
    case out
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.UniversallyClosed (CategoryTheory.CategoryStruct.co …
    inst✝ : AlgebraicGeometry.Surjective f
    ⊢ (AlgebraicGeometry.topologically @IsClosedMap).universally g
  -/
  intro X' Y' i₁ i₂ f' H
  /-
    case out
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.UniversallyClosed (CategoryTheory.CategoryStruct.co …
    inst✝ : AlgebraicGeometry.Surjective f
    X' Y' : AlgebraicGeometry.Scheme
    i₁ : Quiver.Hom X' Y
    i₂ : Quiver.Hom Y' Z
    f' : Quiver.Hom X' Y'
    H : CategoryTheory.IsPullback f' i₁ i₂ g
    ⊢ AlgebraicGeometry.topologically (@IsClosedMap) f'
  -/
  have := UniversallyClosed.out _ _ _ ((IsPullback.of_hasPullback i₁ f).paste_horiz H)
  exact IsClosedMap.of_comp_surjective (MorphismProperty.pullback_fst (P := @Surjective) _ _ ‹_›).1
    (Scheme.Hom.continuous _) this


instance universallyClosedTypeComp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z)
    [hf : UniversallyClosed f] [hg : UniversallyClosed g] : UniversallyClosed (f ≫ g) :=
  comp_mem _ _ _ hf hg


instance : MorphismProperty.IsMultiplicative @UniversallyClosed where
  id_mem _ := inferInstance


instance universallyClosed_fst {X Y Z : Scheme} (f : X ⟶ Z) (g : Y ⟶ Z) [hg : UniversallyClosed g] :
    UniversallyClosed (pullback.fst f g) :=
  MorphismProperty.pullback_fst f g hg


instance universallyClosed_snd {X Y Z : Scheme} (f : X ⟶ Z) (g : Y ⟶ Z) [hf : UniversallyClosed f] :
    UniversallyClosed (pullback.snd f g) :=
  MorphismProperty.pullback_snd f g hf


instance universallyClosed_isLocalAtTarget : IsLocalAtTarget @UniversallyClosed := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.IsLocalAtTarget @AlgebraicGeometry.UniversallyClosed
  -/
  rw [universallyClosed_eq]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.topologically @IsClosed …
  -/
  apply universally_isLocalAtTarget
  /-
    case hP₂
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u_1} (U :  …
  -/
  intro X Y f ι U hU H
  /-
    case hP₂
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Type u_1
    U : ι → Y.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), AlgebraicGeometry.topologically (@IsClosedMap) (AlgebraicGeomet …
    ⊢ AlgebraicGeometry.topologically (@IsClosedMap) f
  -/
  simp_rw [topologically, morphismRestrict_base] at H
  /-
    case hP₂
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Type u_1
    U : ι → Y.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsClosedMap ((U i).carrier.restrictPreimage ⇑f.base)
    ⊢ AlgebraicGeometry.topologically (@IsClosedMap) f
  -/
  exact (isClosedMap_iff_isClosedMap_of_iSup_eq_top hU).mpr H
  /-
    🎉 no goals
  -/


open Scheme.Pullback _root_.PrimeSpectrum MvPolynomial in
/-- If `X` is universally closed over a field, then `X` is quasi-compact. -/
lemma compactSpace_of_universallyClosed
    {K} [Field K] (f : X ⟶ Spec (.of K)) [UniversallyClosed f] : CompactSpace X := by
  classical
  let 𝒰 : X.OpenCover := X.affineCover
  let U (i : 𝒰.J) : X.Opens := (𝒰.map i).opensRange
  let T : Scheme := Spec (.of <| MvPolynomial 𝒰.J K)
  let q : T ⟶ Spec (.of K) := Spec.map (CommRingCat.ofHom MvPolynomial.C)
  let Ti (i : 𝒰.J) : T.Opens := basicOpen (MvPolynomial.X i)
  let fT : pullback f q ⟶ T := pullback.snd f q
  let p : pullback f q ⟶ X := pullback.fst f q
  let Z : Set (pullback f q : _) := (⨆ i, fT ⁻¹ᵁ (Ti i) ⊓ p ⁻¹ᵁ (U i) : (pullback f q).Opens)ᶜ
  have hZ : IsClosed Z := by
    simp only [Z, isClosed_compl_iff, Opens.coe_iSup, Opens.coe_inf, Opens.map_coe]
    exact isOpen_iUnion fun i ↦ (fT.continuous.1 _ (Ti i).2).inter (p.continuous.1 _ (U i).2)
  let Zc : T.Opens := ⟨(fT.base '' Z)ᶜ, (fT.isClosedMap _ hZ).isOpen_compl⟩
  let ψ : MvPolynomial 𝒰.J K →ₐ[K] K := MvPolynomial.aeval (fun _ ↦ 1)
  let t : T := (Spec.map <| CommRingCat.ofHom ψ.toRingHom).base default
  have ht (i : 𝒰.J) : t ∈ Ti i := show ψ (.X i) ≠ 0 by simp [ψ]
  have htZc : t ∈ Zc := by
    intro ⟨z, hz, hzt⟩
    suffices ∃ i, fT.base z ∈ Ti i ∧ p.base z ∈ U i from hz (by simpa)
    exact ⟨𝒰.f (p.base z), hzt ▸ ht _, by simpa [U] using 𝒰.covers (p.base z)⟩
  obtain ⟨U', ⟨g, rfl⟩, htU', hU'le⟩ := Opens.isBasis_iff_nbhd.mp isBasis_basic_opens htZc
  let σ : Finset 𝒰.J := MvPolynomial.vars g
  let φ : MvPolynomial 𝒰.J K →+* MvPolynomial 𝒰.J K :=
    (MvPolynomial.aeval fun i : 𝒰.J ↦ if i ∈ σ then MvPolynomial.X i else 0).toRingHom
  let t' : T := (Spec.map (CommRingCat.ofHom φ)).base t
  have ht'g : t' ∈ PrimeSpectrum.basicOpen g :=
    show φ g ∉ t.asIdeal from (show φ g = g from aeval_ite_mem_eq_self g subset_rfl).symm ▸ htU'
  have h : t' ∉ fT.base '' Z := hU'le ht'g
  suffices ⋃ i ∈ σ, (U i).1 = Set.univ from
    ⟨this ▸ Finset.isCompact_biUnion _ fun i _ ↦ isCompact_range (𝒰.map i).continuous⟩
  rw [Set.iUnion₂_eq_univ_iff]
  contrapose! h
  obtain ⟨x, hx⟩ := h
  obtain ⟨z, rfl, hzr⟩ := exists_preimage_pullback x t' (Subsingleton.elim (f.base x) (q.base t'))
  suffices ∀ i, t ∈ (Ti i).comap (comap φ) → p.base z ∉ U i from ⟨z, by simpa [Z, p, fT, hzr], hzr⟩
  intro i hi₁ hi₂
  rw [comap_basicOpen, show φ (.X i) = 0 by simpa [φ] using (hx i · hi₂), basicOpen_zero] at hi₁
  cases hi₁


@[stacks 04XU]
lemma Scheme.Hom.isProperMap (f : X.Hom Y) [UniversallyClosed f] : IsProperMap f.base := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    inst✝ : AlgebraicGeometry.UniversallyClosed f
    ⊢ IsProperMap ⇑f.base
  -/
  rw [isProperMap_iff_isClosedMap_and_compact_fibers]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : X.Hom Y
    inst✝ : AlgebraicGeometry.UniversallyClosed f
    ⊢ And (Continuous ⇑f.base) (And (IsClosedMap ⇑f.base) (∀ (y : ↑↑Y.toPresheafed …
  -/
  refine ⟨Scheme.Hom.continuous f, ?_, ?_⟩
    /-
      case refine_1
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      inst✝ : AlgebraicGeometry.UniversallyClosed f
      ⊢ IsClosedMap ⇑f.base
    -/
  · exact MorphismProperty.universally_le (P := topologically @IsClosedMap) _ UniversallyClosed.out
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      inst✝ : AlgebraicGeometry.UniversallyClosed f
      ⊢ ∀ (y : ↑↑Y.toPresheafedSpace), IsCompact (Set.preimage (⇑f.base) (Singleton. …
    -/
  · intro y
    /-
      case refine_2
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      inst✝ : AlgebraicGeometry.UniversallyClosed f
      y : ↑↑Y.toPresheafedSpace
      ⊢ IsCompact (Set.preimage (⇑f.base) (Singleton.singleton y))
    -/
    have := compactSpace_of_universallyClosed (pullback.snd f (Y.fromSpecResidueField y))
    /-
      case refine_2
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      inst✝ : AlgebraicGeometry.UniversallyClosed f
      y : ↑↑Y.toPresheafedSpace
      this : CompactSpace ↑↑(CategoryTheory.Limits.pullback f (Y.fromSpecResidueFiel …
      ⊢ IsCompact (Set.preimage (⇑f.base) (Singleton.singleton y))
    -/
    rw [← Scheme.range_fromSpecResidueField, ← Scheme.Pullback.range_fst]
    /-
      case refine_2
      X Y : AlgebraicGeometry.Scheme
      f : X.Hom Y
      inst✝ : AlgebraicGeometry.UniversallyClosed f
      y : ↑↑Y.toPresheafedSpace
      this : CompactSpace ↑↑(CategoryTheory.Limits.pullback f (Y.fromSpecResidueFiel …
      ⊢ IsCompact (Set.range ⇑(CategoryTheory.Limits.pullback.fst f (Y.fromSpecResid …
    -/
    exact isCompact_range (Scheme.Hom.continuous _)
    /-
      🎉 no goals
    -/


instance (priority := 900) [UniversallyClosed f] : QuasiCompact f where
  isCompact_preimage _ _ := f.isProperMap.isCompact_preimage


lemma universallyClosed_eq_universallySpecializing :
    @UniversallyClosed = (topologically @SpecializingMap).universally ⊓ @QuasiCompact := by
  /-
    ⊢ Eq (@AlgebraicGeometry.UniversallyClosed) (Min.min (AlgebraicGeometry.topolo …
  -/
  rw [← universally_eq_iff (P := @QuasiCompact).mpr inferInstance, ← universally_inf]
  /-
    ⊢ Eq (@AlgebraicGeometry.UniversallyClosed) (Min.min (AlgebraicGeometry.topolo …
  -/
  apply le_antisymm
    /-
      case a
      ⊢ LE.le (@AlgebraicGeometry.UniversallyClosed) (Min.min (AlgebraicGeometry.top …
    -/
  · rw [← universally_eq_iff (P := @UniversallyClosed).mpr inferInstance]
    /-
      case a
      ⊢ LE.le (CategoryTheory.MorphismProperty.universally @AlgebraicGeometry.Univer …
    -/
    exact universally_mono fun X Y f H ↦ ⟨f.isClosedMap.specializingMap, inferInstance⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      ⊢ LE.le (Min.min (AlgebraicGeometry.topologically @SpecializingMap) @Algebraic …
    -/
  · rw [universallyClosed_eq]
    /-
      case a
      ⊢ LE.le (Min.min (AlgebraicGeometry.topologically @SpecializingMap) @Algebraic …
    -/
    exact universally_mono fun X Y f ⟨h₁, h₂⟩ ↦ (isClosedMap_iff_specializingMap _).mpr h₁
    /-
      🎉 no goals
    -/


