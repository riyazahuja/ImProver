/-- The class of morphisms `f : X ⟶ Y` between schemes such that
`𝒪_{Y, f x} ⟶ 𝒪_{X, x}` is surjective for all `x : X`. -/
@[mk_iff]
class SurjectiveOnStalks : Prop where
  surj_on_stalks : ∀ x, Function.Surjective (f.stalkMap x)


theorem Scheme.Hom.stalkMap_surjective (f : X.Hom Y) [SurjectiveOnStalks f] (x) :
    Function.Surjective (f.stalkMap x) :=
  SurjectiveOnStalks.surj_on_stalks x


instance (priority := 900) [IsOpenImmersion f] : SurjectiveOnStalks f :=
  ⟨fun _ ↦ (ConcreteCategory.bijective_of_isIso (C := CommRingCat) _).2⟩


instance : MorphismProperty.IsMultiplicative @SurjectiveOnStalks where
  id_mem _ := inferInstance
  comp_mem {X Y Z} f g hf hg := by
    /-
      X✝ Y✝ Z✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.SurjectiveOnStalks f
      hg : AlgebraicGeometry.SurjectiveOnStalks g
      ⊢ AlgebraicGeometry.SurjectiveOnStalks (CategoryTheory.CategoryStruct.comp f g)
    -/
    refine ⟨fun x ↦ ?_⟩
    /-
      X✝ Y✝ Z✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.SurjectiveOnStalks f
      hg : AlgebraicGeometry.SurjectiveOnStalks g
      x : ↑↑X.toPresheafedSpace
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory. …
    -/
    rw [Scheme.stalkMap_comp]
    /-
      X✝ Y✝ Z✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.SurjectiveOnStalks f
      hg : AlgebraicGeometry.SurjectiveOnStalks g
      x : ↑↑X.toPresheafedSpace
      ⊢ Function.Surjective ⇑(CategoryTheory.CategoryStruct.comp (AlgebraicGeometry. …
    -/
    exact (hf.surj_on_stalks x).comp (hg.surj_on_stalks (f.base x))
    /-
      🎉 no goals
    -/


instance comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [SurjectiveOnStalks f]
    [SurjectiveOnStalks g] : SurjectiveOnStalks (f ≫ g) :=
  MorphismProperty.IsStableUnderComposition.comp_mem f g inferInstance inferInstance


lemma eq_stalkwise :
    @SurjectiveOnStalks = stalkwise (Function.Surjective ·) := by
  /-
    ⊢ Eq (@AlgebraicGeometry.SurjectiveOnStalks) (AlgebraicGeometry.stalkwise fun  …
  -/
  ext; exact surjectiveOnStalks_iff _
       /-
         🎉 no goals
       -/


instance : IsLocalAtTarget @SurjectiveOnStalks :=
  eq_stalkwise ▸ stalkwiseIsLocalAtTarget_of_respectsIso RingHom.surjective_respectsIso


instance : IsLocalAtSource @SurjectiveOnStalks :=
  eq_stalkwise ▸ stalkwise_isLocalAtSource_of_respectsIso RingHom.surjective_respectsIso


lemma Spec_iff {R S : CommRingCat.{u}} {φ : R ⟶ S} :
    SurjectiveOnStalks (Spec.map φ) ↔ RingHom.SurjectiveOnStalks φ.hom := by
  rw [eq_stalkwise, stalkwise_Spec_map_iff RingHom.surjective_respectsIso,
    RingHom.SurjectiveOnStalks]


instance : HasRingHomProperty @SurjectiveOnStalks RingHom.SurjectiveOnStalks :=
  eq_stalkwise ▸ .stalkwise RingHom.surjective_respectsIso


variable {f} in
lemma iff_of_isAffine [IsAffine X] [IsAffine Y] :
    SurjectiveOnStalks f ↔ RingHom.SurjectiveOnStalks (f.app ⊤).hom := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (AlgebraicGeometry.SurjectiveOnStalks f) (AlgebraicGeometry.Scheme.Hom.a …
  -/
  rw [← Spec_iff, MorphismProperty.arrow_mk_iso_iff @SurjectiveOnStalks (arrowIsoSpecΓOfIsAffine f)]
  /-
    🎉 no goals
  -/


theorem of_comp [SurjectiveOnStalks (f ≫ g)] : SurjectiveOnStalks f := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.SurjectiveOnStalks (CategoryTheory.CategoryStruct.co …
    ⊢ AlgebraicGeometry.SurjectiveOnStalks f
  -/
  refine ⟨fun x ↦ ?_⟩
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.SurjectiveOnStalks (CategoryTheory.CategoryStruct.co …
    x : ↑↑X.toPresheafedSpace
    ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  have := (f ≫ g).stalkMap_surjective x
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.SurjectiveOnStalks (CategoryTheory.CategoryStruct.co …
    x : ↑↑X.toPresheafedSpace
    this : Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTh …
    ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  rw [Scheme.stalkMap_comp] at this
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.SurjectiveOnStalks (CategoryTheory.CategoryStruct.co …
    x : ↑↑X.toPresheafedSpace
    this : Function.Surjective ⇑(CategoryTheory.CategoryStruct.comp (AlgebraicGeom …
    ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
  -/
  exact Function.Surjective.of_comp this
  /-
    🎉 no goals
  -/


instance stableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange @SurjectiveOnStalks := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.S …
  -/
  apply HasRingHomProperty.isStableUnderBaseChange
  /-
    case hP
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => RingH …
  -/
  apply RingHom.IsStableUnderBaseChange.mk
    /-
      case hP.h₁
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => RingHom.Surjectiv …
    -/
  · exact (HasRingHomProperty.isLocal_ringHomProperty @SurjectiveOnStalks).respectsIso
    /-
      🎉 no goals
    -/
  /-
    case hP.h₂
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ ∀ ⦃R S T : Type u_1⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
  -/
  intros R S T _ _ _ _ _ H
  /-
    case hP.h₂
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    R S T : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    H : (algebraMap R T).SurjectiveOnStalks
    ⊢ Algebra.TensorProduct.includeLeftRingHom.SurjectiveOnStalks
  -/
  exact H.baseChange
  /-
    🎉 no goals
  -/


/-- If `Y ⟶ S` is surjective on stalks, then for every `X ⟶ S`, `X ×ₛ Y` is a subset of
`X × Y` (cartesian product as topological spaces) with the induced topology. -/
lemma isEmbedding_pullback {X Y S : Scheme.{u}} (f : X ⟶ S) (g : Y ⟶ S) [SurjectiveOnStalks g] :
    IsEmbedding (fun x ↦ ((pullback.fst f g).base x, (pullback.snd f g).base x)) := by
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
    ⊢ Topology.IsEmbedding fun x => { fst := (CategoryTheory.Limits.pullback.fst f …
  -/
  let L := (fun x ↦ ((pullback.fst f g).base x, (pullback.snd f g).base x))
  have H : ∀ R A B (f' : Spec A ⟶ Spec R) (g' : Spec B ⟶ Spec R) (iX : Spec A ⟶ X)
      (iY : Spec B ⟶ Y) (iS : Spec R ⟶ S) (e₁ e₂), IsOpenImmersion iX → IsOpenImmersion iY →
      IsOpenImmersion iS → IsEmbedding (L ∘ (pullback.map f' g' f g iX iY iS e₁ e₂).base) := by
    intro R A B f' g' iX iY iS e₁ e₂ _ _ _
    have H : SurjectiveOnStalks g' :=
      have : SurjectiveOnStalks (g' ≫ iS) := e₂ ▸ inferInstance
      .of_comp _ iS
    obtain ⟨φ, rfl⟩ : ∃ φ, Spec.map φ = f' := ⟨_, Spec.map_preimage _⟩
    obtain ⟨ψ, rfl⟩ : ∃ ψ, Spec.map ψ = g' := ⟨_, Spec.map_preimage _⟩
    algebraize [φ.hom, ψ.hom]
    rw [HasRingHomProperty.Spec_iff (P := @SurjectiveOnStalks)] at H
    convert ((iX.isOpenEmbedding.prodMap iY.isOpenEmbedding).isEmbedding.comp
      (PrimeSpectrum.isEmbedding_tensorProductTo_of_surjectiveOnStalks R A B H)).comp
      (Scheme.homeoOfIso (pullbackSpecIso R A B)).isEmbedding
    ext1 x
    obtain ⟨x, rfl⟩ := (Scheme.homeoOfIso (pullbackSpecIso R A B).symm).surjective x
    simp only [Scheme.homeoOfIso_apply, Function.comp_apply]
    ext
    · simp only [L, ← Scheme.comp_base_apply, pullback.lift_fst, Iso.symm_hom,
        Iso.inv_hom_id]
      erw [← Scheme.comp_base_apply, pullbackSpecIso_inv_fst_assoc]
      rfl
    · simp only [L, ← Scheme.comp_base_apply, pullback.lift_snd, Iso.symm_hom,
        Iso.inv_hom_id]
      erw [← Scheme.comp_base_apply, pullbackSpecIso_inv_snd_assoc]
      rfl
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
    L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
    H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
    ⊢ Topology.IsEmbedding fun x => { fst := (CategoryTheory.Limits.pullback.fst f …
  -/
  let 𝒰 := S.affineOpenCover.openCover
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
    L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
    H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
    𝒰 : S.OpenCover := S.affineOpenCover.openCover
    ⊢ Topology.IsEmbedding fun x => { fst := (CategoryTheory.Limits.pullback.fst f …
  -/
  let 𝒱 (i) := ((𝒰.pullbackCover f).obj i).affineOpenCover.openCover
  /-
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
    L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
    H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
    𝒰 : S.OpenCover := S.affineOpenCover.openCover
    𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
    ⊢ Topology.IsEmbedding fun x => { fst := (CategoryTheory.Limits.pullback.fst f …
  -/
  let 𝒲 (i) := ((𝒰.pullbackCover g).obj i).affineOpenCover.openCover
  let U (ijk : Σ i, (𝒱 i).J × (𝒲 i).J) : TopologicalSpace.Opens (X.carrier × Y) :=
    ⟨{ P | P.1 ∈ ((𝒱 ijk.1).map ijk.2.1 ≫ (𝒰.pullbackCover f).map ijk.1).opensRange ∧
          P.2 ∈ ((𝒲 ijk.1).map ijk.2.2 ≫ (𝒰.pullbackCover g).map ijk.1).opensRange },
      (continuous_fst.1 _ ((𝒱 ijk.1).map ijk.2.1 ≫
      (𝒰.pullbackCover f).map ijk.1).opensRange.2).inter (continuous_snd.1 _
      ((𝒲 ijk.1).map ijk.2.2 ≫ (𝒰.pullbackCover g).map ijk.1).opensRange.2)⟩
  have : Set.range L ⊆ (iSup U : _) := by
    simp only [Scheme.Cover.pullbackCover_J, Scheme.Cover.pullbackCover_obj, Set.range_subset_iff]
    intro z
    simp only [SetLike.mem_coe, TopologicalSpace.Opens.mem_iSup, Sigma.exists, Prod.exists]
    obtain ⟨is, s, hsx⟩ := 𝒰.exists_eq (f.base ((pullback.fst f g).base z))
    have hsy : (𝒰.map is).base s = g.base ((pullback.snd f g).base z) := by
      rwa [← Scheme.comp_base_apply, ← pullback.condition, Scheme.comp_base_apply]
    obtain ⟨x : (𝒰.pullbackCover f).obj is, hx⟩ :=
      Scheme.IsJointlySurjectivePreserving.exists_preimage_fst_triplet_of_prop
        (P := @IsOpenImmersion) inferInstance _ _ hsx.symm
    obtain ⟨y : (𝒰.pullbackCover g).obj is, hy⟩ :=
      Scheme.IsJointlySurjectivePreserving.exists_preimage_fst_triplet_of_prop
        (P := @IsOpenImmersion) inferInstance _ _ hsy.symm
    obtain ⟨ix, x, rfl⟩ := (𝒱 is).exists_eq x
    obtain ⟨iy, y, rfl⟩ := (𝒲 is).exists_eq y
    refine ⟨is, ix, iy, ⟨x, hx⟩, ⟨y, hy⟩⟩
  let 𝓤 := (Scheme.Pullback.openCoverOfBase 𝒰 f g).bind
    (fun i ↦ Scheme.Pullback.openCoverOfLeftRight (𝒱 i) (𝒲 i) _ _)
  refine isEmbedding_of_iSup_eq_top_of_preimage_subset_range _ ?_ U this _ (fun i ↦ (𝓤.map i).base)
    (fun i ↦ (𝓤.map i).continuous) ?_ ?_
    /-
      case refine_1
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
      L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
      H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
      𝒰 : S.OpenCover := S.affineOpenCover.openCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
      U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
      this : HasSubset.Subset (Set.range L) ↑(iSup U)
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      ⊢ Continuous fun x => { fst := (CategoryTheory.Limits.pullback.fst f g).base x …
    -/
  · fun_prop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
      L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
      H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
      𝒰 : S.OpenCover := S.affineOpenCover.openCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
      U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
      this : HasSubset.Subset (Set.range L) ↑(iSup U)
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      ⊢ ∀ (i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J), HasSubset.Subset (Set.preimage  …
    -/
  · rintro i x ⟨⟨x₁, hx₁⟩, ⟨x₂, hx₂⟩⟩
    obtain ⟨x₁', hx₁'⟩ :=
      Scheme.IsJointlySurjectivePreserving.exists_preimage_fst_triplet_of_prop
        (P := @IsOpenImmersion) inferInstance _ _ hx₁.symm
    obtain ⟨x₂', hx₂'⟩ :=
      Scheme.IsJointlySurjectivePreserving.exists_preimage_fst_triplet_of_prop
        (P := @IsOpenImmersion) inferInstance _ _ hx₂.symm
    obtain ⟨z, hz⟩ :=
      Scheme.IsJointlySurjectivePreserving.exists_preimage_fst_triplet_of_prop
        (P := @IsOpenImmersion) inferInstance _ _ (hx₁'.trans hx₂'.symm)
    /-
      case refine_2.intro.intro.intro.intro.intro.intro
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
      L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
      H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
      𝒰 : S.OpenCover := S.affineOpenCover.openCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
      U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
      this : HasSubset.Subset (Set.range L) ↑(iSup U)
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
      x : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
      x₁ : ↑↑((𝒱 i.fst).obj i.snd.1).toPresheafedSpace
      hx₁ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒱 i.fst).map i.snd.1) ((Algebr …
      x₂ : ↑↑((𝒲 i.fst).obj i.snd.2).toPresheafedSpace
      hx₂ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒲 i.fst).map i.snd.2) ((Algebr …
      x₁' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f  …
      hx₁' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
      x₂' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f  …
      hx₂' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
      z : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
      hz : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.f …
      ⊢ Membership.mem (Set.range ((fun i => ⇑(𝓤.map i).base) i)) x
    -/
    refine ⟨(pullbackFstFstIso _ _ _ _ _ _ (𝒰.map i.1) ?_ ?_).hom.base z, ?_⟩
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.refine_1
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
        L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
        H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
        𝒰 : S.OpenCover := S.affineOpenCover.openCover
        𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
        𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
        U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
        this : HasSubset.Subset (Set.range L) ↑(iSup U)
        𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
        i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
        x : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        x₁ : ↑↑((𝒱 i.fst).obj i.snd.1).toPresheafedSpace
        hx₁ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒱 i.fst).map i.snd.1) ((Algebr …
        x₂ : ↑↑((𝒲 i.fst).obj i.snd.2).toPresheafedSpace
        hx₂ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒲 i.fst).map i.snd.2) ((Algebr …
        x₁' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f  …
        hx₁' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        x₂' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f  …
        hx₂' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        z : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
        hz : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.f …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp [pullback.condition]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.refine_2
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
        L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
        H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
        𝒰 : S.OpenCover := S.affineOpenCover.openCover
        𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
        𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
        U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
        this : HasSubset.Subset (Set.range L) ↑(iSup U)
        𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
        i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
        x : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        x₁ : ↑↑((𝒱 i.fst).obj i.snd.1).toPresheafedSpace
        hx₁ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒱 i.fst).map i.snd.1) ((Algebr …
        x₂ : ↑↑((𝒲 i.fst).obj i.snd.2).toPresheafedSpace
        hx₂ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒲 i.fst).map i.snd.2) ((Algebr …
        x₁' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f  …
        hx₁' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        x₂' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f  …
        hx₂' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        z : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
        hz : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.f …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp [pullback.condition]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.refine_3
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
        L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
        H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
        𝒰 : S.OpenCover := S.affineOpenCover.openCover
        𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
        𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
        U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
        this : HasSubset.Subset (Set.range L) ↑(iSup U)
        𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
        i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
        x : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        x₁ : ↑↑((𝒱 i.fst).obj i.snd.1).toPresheafedSpace
        hx₁ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒱 i.fst).map i.snd.1) ((Algebr …
        x₂ : ↑↑((𝒲 i.fst).obj i.snd.2).toPresheafedSpace
        hx₂ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒲 i.fst).map i.snd.2) ((Algebr …
        x₁' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f  …
        hx₁' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        x₂' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f  …
        hx₂' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        z : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
        hz : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.f …
        ⊢ Eq ((fun i => ⇑(𝓤.map i).base) i ((CategoryTheory.Limits.pullbackFstFstIso ( …
      -/
    · dsimp only
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.refine_3
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
        L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
        H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
        𝒰 : S.OpenCover := S.affineOpenCover.openCover
        𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
        𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
        U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
        this : HasSubset.Subset (Set.range L) ↑(iSup U)
        𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
        i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
        x : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        x₁ : ↑↑((𝒱 i.fst).obj i.snd.1).toPresheafedSpace
        hx₁ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒱 i.fst).map i.snd.1) ((Algebr …
        x₂ : ↑↑((𝒲 i.fst).obj i.snd.2).toPresheafedSpace
        hx₂ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒲 i.fst).map i.snd.2) ((Algebr …
        x₁' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f  …
        hx₁' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        x₂' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f  …
        hx₂' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        z : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
        hz : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.f …
        ⊢ Eq ((𝓤.map i).base ((CategoryTheory.Limits.pullbackFstFstIso (CategoryTheory …
      -/
      rw [← hx₁', ← hz, ← Scheme.comp_base_apply]
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.refine_3
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
        L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
        H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
        𝒰 : S.OpenCover := S.affineOpenCover.openCover
        𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
        𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
        U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
        this : HasSubset.Subset (Set.range L) ↑(iSup U)
        𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
        i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
        x : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        x₁ : ↑↑((𝒱 i.fst).obj i.snd.1).toPresheafedSpace
        hx₁ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒱 i.fst).map i.snd.1) ((Algebr …
        x₂ : ↑↑((𝒲 i.fst).obj i.snd.2).toPresheafedSpace
        hx₂ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒲 i.fst).map i.snd.2) ((Algebr …
        x₁' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f  …
        hx₁' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        x₂' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f  …
        hx₂' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        z : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
        hz : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.f …
        ⊢ Eq ((𝓤.map i).base ((CategoryTheory.Limits.pullbackFstFstIso (CategoryTheory …
      -/
      erw [← Scheme.comp_base_apply]
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.refine_3
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
        L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
        H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
        𝒰 : S.OpenCover := S.affineOpenCover.openCover
        𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
        𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
        U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
        this : HasSubset.Subset (Set.range L) ↑(iSup U)
        𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
        i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
        x : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        x₁ : ↑↑((𝒱 i.fst).obj i.snd.1).toPresheafedSpace
        hx₁ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒱 i.fst).map i.snd.1) ((Algebr …
        x₂ : ↑↑((𝒲 i.fst).obj i.snd.2).toPresheafedSpace
        hx₂ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒲 i.fst).map i.snd.2) ((Algebr …
        x₁' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f  …
        hx₁' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        x₂' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f  …
        hx₂' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        z : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
        hz : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.f …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackFstFs …
      -/
      congr 4
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.refine_3.e_a.e_self.e_self.e …
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
        L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
        H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
        𝒰 : S.OpenCover := S.affineOpenCover.openCover
        𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
        𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
        U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
        this : HasSubset.Subset (Set.range L) ↑(iSup U)
        𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
        i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
        x : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace
        x₁ : ↑↑((𝒱 i.fst).obj i.snd.1).toPresheafedSpace
        hx₁ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒱 i.fst).map i.snd.1) ((Algebr …
        x₂ : ↑↑((𝒲 i.fst).obj i.snd.2).toPresheafedSpace
        hx₂ : Eq ((CategoryTheory.CategoryStruct.comp ((𝒲 i.fst).map i.snd.2) ((Algebr …
        x₁' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst f  …
        hx₁' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        x₂' : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.snd f  …
        hx₂' : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback …
        z : ↑↑(CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.fst (Cat …
        hz : Eq ((CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.f …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackFstFst …
      -/
                                 /-
                                   🎉 no goals
                                 -/
      apply pullback.hom_ext <;> simp [𝓤, ← pullback.condition, ← pullback.condition_assoc]
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case refine_3
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
      L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
      H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
      𝒰 : S.OpenCover := S.affineOpenCover.openCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
      U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
      this : HasSubset.Subset (Set.range L) ↑(iSup U)
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      ⊢ ∀ (i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J), Topology.IsEmbedding (Function. …
    -/
  · intro i
    have := H (S.affineOpenCover.obj i.1) (((𝒰.pullbackCover f).obj i.1).affineOpenCover.obj i.2.1)
        (((𝒰.pullbackCover g).obj i.1).affineOpenCover.obj i.2.2)
        ((𝒱 i.1).map i.2.1 ≫ 𝒰.pullbackHom f i.1)
        ((𝒲 i.1).map i.2.2 ≫ 𝒰.pullbackHom g i.1)
        ((𝒱 i.1).map i.2.1 ≫ (𝒰.pullbackCover f).map i.1)
        ((𝒲 i.1).map i.2.2 ≫ (𝒰.pullbackCover g).map i.1)
        (𝒰.map i.1) (by simp [pullback.condition]) (by simp [pullback.condition])
        inferInstance inferInstance inferInstance
    /-
      case refine_3
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
      L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
      H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
      𝒰 : S.OpenCover := S.affineOpenCover.openCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
      U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
      this✝ : HasSubset.Subset (Set.range L) ↑(iSup U)
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
      this : Topology.IsEmbedding (Function.comp L ⇑(CategoryTheory.Limits.pullback. …
      ⊢ Topology.IsEmbedding (Function.comp (fun x => { fst := (CategoryTheory.Limit …
    -/
    convert this using 6
    /-
      case h.e'_5.h.h.e'_5.h.h.e'_5.h.e'_5.h.h.e'_3.h.h.e'_3.h
      X Y S : AlgebraicGeometry.Scheme
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      inst✝ : AlgebraicGeometry.SurjectiveOnStalks g
      L : ↑↑(CategoryTheory.Limits.pullback f g).toPresheafedSpace → Prod ↑↑X.toPres …
      H : ∀ (R A B : CommRingCat) (f' : Quiver.Hom (AlgebraicGeometry.Spec A) (Algeb …
      𝒰 : S.OpenCover := S.affineOpenCover.openCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      𝒲 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 g).J) → ((AlgebraicGe …
      U : (Sigma fun i => Prod (𝒱 i).J (𝒲 i).J) → TopologicalSpace.Opens (Prod ↑↑X.t …
      this✝ : HasSubset.Subset (Set.range L) ↑(iSup U)
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      i : Sigma fun i => Prod (𝒱 i).J (𝒲 i).J
      this : Topology.IsEmbedding (Function.comp L ⇑(CategoryTheory.Limits.pullback. …
      e_1✝² : Eq ↑↑(𝓤.obj i).toPresheafedSpace ↑↑(CategoryTheory.Limits.pullback (Ca …
      e_3✝ : Eq (𝓤.obj i).toPresheafedSpace (CategoryTheory.Limits.pullback (Categor …
      e_1✝¹ : Eq (𝓤.obj i).toLocallyRingedSpace (CategoryTheory.Limits.pullback (Cat …
      e_1✝ : Eq (𝓤.obj i) (CategoryTheory.Limits.pullback (CategoryTheory.CategorySt …
      ⊢ Eq (𝓤.map i) (CategoryTheory.Limits.pullback.map (CategoryTheory.CategoryStr …
    -/
    apply pullback.hom_ext <;>
      simp [𝓤, ← pullback.condition, ← pullback.condition_assoc,
        Scheme.Cover.pullbackHom]


