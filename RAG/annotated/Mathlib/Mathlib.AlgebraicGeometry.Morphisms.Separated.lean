/-- A morphism is separated if the diagonal map is a closed immersion. -/
@[mk_iff]
class IsSeparated : Prop where
  /-- A morphism is separated if the diagonal map is a closed immersion. -/
  diagonal_isClosedImmersion : IsClosedImmersion (pullback.diagonal f) := by infer_instance


theorem isSeparated_eq_diagonal_isClosedImmersion :
    @IsSeparated = MorphismProperty.diagonal @IsClosedImmersion := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsSeparated) (CategoryTheory.MorphismProperty.diagona …
  -/
  ext
  /-
    case h.h.h.a
    x✝² x✝¹ : AlgebraicGeometry.Scheme
    x✝ : Quiver.Hom x✝² x✝¹
    ⊢ Iff (AlgebraicGeometry.IsSeparated x✝) (CategoryTheory.MorphismProperty.diag …
  -/
  exact isSeparated_iff _
  /-
    🎉 no goals
  -/


/-- Monomorphisms are separated. -/
instance (priority := 900) isSeparated_of_mono [Mono f] : IsSeparated f where


instance : MorphismProperty.RespectsIso @IsSeparated := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ CategoryTheory.MorphismProperty.RespectsIso @AlgebraicGeometry.IsSeparated
  -/
  rw [isSeparated_eq_diagonal_isClosedImmersion]
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ (CategoryTheory.MorphismProperty.diagonal @AlgebraicGeometry.IsClosedImmersi …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (priority := 900) [IsSeparated f] : QuasiSeparated f where


instance stableUnderComposition : MorphismProperty.IsStableUnderComposition @IsSeparated := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderComposition @AlgebraicGeometry. …
  -/
  rw [isSeparated_eq_diagonal_isClosedImmersion]
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ (CategoryTheory.MorphismProperty.diagonal @AlgebraicGeometry.IsClosedImmersi …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [IsSeparated f] [IsSeparated g] : IsSeparated (f ≫ g) :=
  stableUnderComposition.comp_mem f g inferInstance inferInstance


instance : MorphismProperty.IsMultiplicative @IsSeparated where
  id_mem _ := inferInstance


instance isStableUnderBaseChange : MorphismProperty.IsStableUnderBaseChange @IsSeparated := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ CategoryTheory.MorphismProperty.IsStableUnderBaseChange @AlgebraicGeometry.I …
  -/
  rw [isSeparated_eq_diagonal_isClosedImmersion]
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ (CategoryTheory.MorphismProperty.diagonal @AlgebraicGeometry.IsClosedImmersi …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : IsLocalAtTarget @IsSeparated := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ AlgebraicGeometry.IsLocalAtTarget @AlgebraicGeometry.IsSeparated
  -/
  rw [isSeparated_eq_diagonal_isClosedImmersion]
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ AlgebraicGeometry.IsLocalAtTarget (CategoryTheory.MorphismProperty.diagonal  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (R S : CommRingCat.{u}) (f : R ⟶ S) : IsSeparated (Spec.map f) := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    R S : CommRingCat
    f : Quiver.Hom R S
    ⊢ AlgebraicGeometry.IsSeparated (AlgebraicGeometry.Spec.map f)
  -/
  constructor
  /-
    case diagonal_isClosedImmersion
    W X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    R S : CommRingCat
    f : Quiver.Hom R S
    ⊢ autoParam (AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullba …
  -/
  letI := f.hom.toAlgebra
  /-
    case diagonal_isClosedImmersion
    W X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    R S : CommRingCat
    f : Quiver.Hom R S
    this : Algebra ↑R ↑S := f.hom.toAlgebra
    ⊢ autoParam (AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullba …
  -/
  show IsClosedImmersion (Limits.pullback.diagonal (Spec.map (CommRingCat.ofHom (algebraMap R S))))
  /-
    case diagonal_isClosedImmersion
    W X Y Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    R S : CommRingCat
    f : Quiver.Hom R S
    this : Algebra ↑R ↑S := f.hom.toAlgebra
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullback.diagonal …
  -/
  rw [diagonal_Spec_map, MorphismProperty.cancel_right_of_respectsIso @IsClosedImmersion]
  exact .spec_of_surjective _ fun x ↦ ⟨.tmul R 1 x,
    (Algebra.TensorProduct.lmul'_apply_tmul (R := R) (S := S) 1 x).trans (one_mul x)⟩


@[instance 100]
lemma of_isAffineHom [h : IsAffineHom f] : IsSeparated f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsAffineHom f
    ⊢ AlgebraicGeometry.IsSeparated f
  -/
  wlog hY : IsAffine Y
  · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := @IsSeparated) _
      (iSup_affineOpens_eq_top Y)]
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsAffineHom f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ ∀ (i : ↑Y.affineOpens), AlgebraicGeometry.IsSeparated (AlgebraicGeometry.mor …
    -/
    intro U
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsAffineHom f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      ⊢ AlgebraicGeometry.IsSeparated (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    have H : IsAffineHom (f ∣_ U) := IsLocalAtTarget.restrict h U
    /-
      case inr
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : AlgebraicGeometry.IsAffineHom f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [h : AlgebraicG …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      H : AlgebraicGeometry.IsAffineHom (AlgebraicGeometry.morphismRestrict f ↑U)
      ⊢ AlgebraicGeometry.IsSeparated (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    exact this _ U.2
    /-
      🎉 no goals
    -/
  /-
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsAffineHom f
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ AlgebraicGeometry.IsSeparated f
  -/
  have : IsAffine X := HasAffineProperty.iff_of_isAffine.mp h
  /-
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsAffineHom f
    hY : AlgebraicGeometry.IsAffine Y
    this : AlgebraicGeometry.IsAffine X
    ⊢ AlgebraicGeometry.IsSeparated f
  -/
  rw [MorphismProperty.arrow_mk_iso_iff @IsSeparated (arrowIsoSpecΓOfIsAffine f)]
  /-
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.IsAffineHom f
    hY : AlgebraicGeometry.IsAffine Y
    this : AlgebraicGeometry.IsAffine X
    ⊢ AlgebraicGeometry.IsSeparated (AlgebraicGeometry.Spec.map (AlgebraicGeometry …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {S T : Scheme.{u}} (f : X ⟶ S) (g : Y ⟶ S) (i : S ⟶ T) [IsSeparated i] :
    IsClosedImmersion (pullback.mapDesc f g i) :=
  MorphismProperty.of_isPullback (pullback_map_diagonal_isPullback f g i)
    inferInstance


/-- Given `f : X ⟶ Y` and `g : Y ⟶ Z` such that `g` is separated, the induced map
`X ⟶ X ×[Z] Y` is a closed immersion. -/
instance [IsSeparated g] :
    IsClosedImmersion (pullback.lift (𝟙 _) f (Category.id_comp (f ≫ g))) := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsSeparated g
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullback.lift (Ca …
  -/
  rw [← MorphismProperty.cancel_left_of_respectsIso @IsClosedImmersion (pullback.fst f (𝟙 Y))]
  rw [← MorphismProperty.cancel_right_of_respectsIso @IsClosedImmersion _
    (pullback.congrHom rfl (Category.id_comp g)).inv]
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsSeparated g
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.comp (Cat …
  -/
  convert (inferInstanceAs <| IsClosedImmersion (pullback.mapDesc f (𝟙 _) g)) using 1
  /-
    case h.e'_3
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsSeparated g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
              /-
                🎉 no goals
              -/
  ext : 1 <;> simp [pullback.condition]
              /-
                🎉 no goals
              -/


lemma Scheme.Pullback.diagonalCoverDiagonalRange_eq_top_of_injective
    (hf : Function.Injective f.base) :
    diagonalCoverDiagonalRange f 𝒰 𝒱 = ⊤ := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    ⊢ Eq (AlgebraicGeometry.Scheme.Pullback.diagonalCoverDiagonalRange f 𝒰 𝒱) Top. …
  -/
  rw [← top_le_iff]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    ⊢ LE.le Top.top (AlgebraicGeometry.Scheme.Pullback.diagonalCoverDiagonalRange  …
  -/
  rintro x -
  simp only [diagonalCoverDiagonalRange, openCoverOfBase_J, openCoverOfBase_obj,
    openCoverOfLeftRight_J, Opens.iSup_mk, Opens.carrier_eq_coe, Hom.coe_opensRange, Opens.coe_mk,
    Set.mem_iUnion, Set.mem_range, Sigma.exists]
  have H : (pullback.fst f f).base x = (pullback.snd f f).base x :=
    hf (by rw [← Scheme.comp_base_apply, ← Scheme.comp_base_apply, pullback.condition])
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  let i := 𝒰.f (f.base ((pullback.fst f f).base x))
  obtain ⟨y : 𝒰.obj i, hy : (𝒰.map i).base y = f.base _⟩ :=
    𝒰.covers (f.base ((pullback.fst f f).base x))
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f).b …
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  obtain ⟨z, hz₁, hz₂⟩ := exists_preimage_pullback _ _ hy.symm
  /-
    case intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f).b …
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) ((CategoryT …
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  let j := (𝒱 i).f z
  /-
    case intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f).b …
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) ((CategoryT …
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  obtain ⟨w : (𝒱 i).obj j, hy : ((𝒱 i).map j).base w = z⟩ := (𝒱 i).covers z
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f). …
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) ((CategoryT …
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  refine ⟨i, j, ?_⟩
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f). …
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) ((CategoryT …
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Exists fun y => Eq (((AlgebraicGeometry.Scheme.Pullback.diagonalCover f 𝒰 𝒱) …
  -/
  simp_rw [diagonalCover_map]
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f). …
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) ((CategoryT …
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Exists fun y => Eq ((CategoryTheory.Limits.pullback.map (CategoryTheory.Cate …
  -/
  show x ∈ Set.range _
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f). …
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) ((CategoryT …
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map (CategoryTheo …
  -/
  dsimp only [diagonalCover, Cover.bind_obj, openCoverOfLeftRight_obj]
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f). …
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) ((CategoryT …
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.map (CategoryTheo …
  -/
  rw [range_map]
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    hf : Function.Injective ⇑f.base
    x : ↑↑(CategoryTheory.Limits.pullback.diagonalObj f).toPresheafedSpace
    H : Eq ((CategoryTheory.Limits.pullback.fst f f).base x) ((CategoryTheory.Limi …
    i : 𝒰.J := 𝒰.f (f.base ((CategoryTheory.Limits.pullback.fst f f).base x))
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base ((CategoryTheory.Limits.pullback.fst f f). …
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) ((CategoryT …
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Membership.mem (Inter.inter (Set.preimage (⇑(CategoryTheory.Limits.pullback. …
  -/
  simp [← H, ← hz₁, ← hy]
  /-
    🎉 no goals
  -/


lemma Scheme.Pullback.range_diagonal_subset_diagonalCoverDiagonalRange :
    Set.range (pullback.diagonal f).base ⊆ diagonalCoverDiagonalRange f 𝒰 𝒱 := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    ⊢ HasSubset.Subset (Set.range ⇑(CategoryTheory.Limits.pullback.diagonal f).bas …
  -/
  rintro _ ⟨x, rfl⟩
  simp only [diagonalCoverDiagonalRange, openCoverOfBase_J, openCoverOfBase_obj,
    openCoverOfLeftRight_J, Opens.iSup_mk, Opens.carrier_eq_coe, Hom.coe_opensRange, Opens.coe_mk,
    Set.mem_iUnion, Set.mem_range, Sigma.exists]
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  let i := 𝒰.f (f.base x)
  /-
    case intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  obtain ⟨y : 𝒰.obj i, hy : (𝒰.map i).base y = f.base x⟩ := 𝒰.covers (f.base x)
  /-
    case intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy : Eq ((𝒰.map i).base y) (f.base x)
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  obtain ⟨z, hz₁, hz₂⟩ := exists_preimage_pullback _ _ hy.symm
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy : Eq ((𝒰.map i).base y) (f.base x)
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) x
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  let j := (𝒱 i).f z
  /-
    case intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy : Eq ((𝒰.map i).base y) (f.base x)
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) x
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  obtain ⟨w : (𝒱 i).obj j, hy : ((𝒱 i).map j).base w = z⟩ := (𝒱 i).covers z
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base x)
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) x
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Exists fun a => Exists fun b => Exists fun y => Eq (((AlgebraicGeometry.Sche …
  -/
  refine ⟨i, j, (pullback.diagonal ((𝒱 i).map j ≫ pullback.snd f (𝒰.map i))).base w, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base x)
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) x
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Eq (((AlgebraicGeometry.Scheme.Pullback.diagonalCover f 𝒰 𝒱).map ⟨i, { fst : …
  -/
  rw [← hz₁, ← hy, ← Scheme.comp_base_apply, ← Scheme.comp_base_apply]
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base x)
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) x
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Eq (((AlgebraicGeometry.Scheme.Pullback.diagonalCover f 𝒰 𝒱).map ⟨i, { fst : …
  -/
  dsimp only [diagonalCover, Cover.pullbackHom, Cover.bind_obj, openCoverOfLeftRight_obj]
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base x)
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) x
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Eq (((AlgebraicGeometry.Scheme.Cover.bind (AlgebraicGeometry.Scheme.Pullback …
  -/
  rw [← Scheme.comp_base_apply]
  /-
    case intro.intro.intro.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base x)
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) x
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.diag …
  -/
  congr 4
  /-
    case intro.intro.intro.intro.intro.e_a.e_self.e_self.e_self
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    x : ↑↑X.toPresheafedSpace
    i : 𝒰.J := 𝒰.f (f.base x)
    y : ↑↑(𝒰.obj i).toPresheafedSpace
    hy✝ : Eq ((𝒰.map i).base y) (f.base x)
    z : ↑↑(CategoryTheory.Limits.pullback f (𝒰.map i)).toPresheafedSpace
    hz₁ : Eq ((CategoryTheory.Limits.pullback.fst f (𝒰.map i)).base z) x
    hz₂ : Eq ((CategoryTheory.Limits.pullback.snd f (𝒰.map i)).base z) y
    j : (𝒱 i).J := (𝒱 i).f z
    w : ↑↑((𝒱 i).obj j).toPresheafedSpace
    hy : Eq (((𝒱 i).map j).base w) z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.diago …
  -/
                             /-
                               🎉 no goals
                             -/
  apply pullback.hom_ext <;> simp
                             /-
                               🎉 no goals
                             -/


lemma isClosedImmersion_diagonal_restrict_diagonalCoverDiagonalRange
    [∀ i, IsAffine (𝒰.obj i)] [∀ i j, IsAffine ((𝒱 i).obj j)] :
    IsClosedImmersion (pullback.diagonal f ∣_ diagonalCoverDiagonalRange f 𝒰 𝒱) := by
  let U : (Σ i, (𝒱 i).J) → (diagonalCoverDiagonalRange f 𝒰 𝒱).toScheme.Opens := fun i ↦
    (diagonalCoverDiagonalRange f 𝒰 𝒱).ι ⁻¹ᵁ ((diagonalCover f 𝒰 𝒱).map ⟨i.1, i.2, i.2⟩).opensRange
  have hU (i) : (diagonalCoverDiagonalRange f 𝒰 𝒱).ι ''ᵁ U i =
      ((diagonalCover f 𝒰 𝒱).map ⟨i.1, i.2, i.2⟩).opensRange := by
    rw [TopologicalSpace.Opens.functor_obj_map_obj, inf_eq_right, Hom.image_top_eq_opensRange,
      Opens.opensRange_ι]
    exact le_iSup (fun i : Σ i, (𝒱 i).J ↦ ((diagonalCover f 𝒰 𝒱).map ⟨i.1, i.2, i.2⟩).opensRange) i
  have hf : iSup U = ⊤ := (TopologicalSpace.Opens.map_iSup _ _).symm.trans
    (diagonalCoverDiagonalRange f 𝒰 𝒱).ι_preimage_self
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    inst✝ : ∀ (i : 𝒰.J) (j : (𝒱 i).J), AlgebraicGeometry.IsAffine ((𝒱 i).obj j)
    U : (Sigma fun i => (𝒱 i).J) → (↑(AlgebraicGeometry.Scheme.Pullback.diagonalCo …
    hU : ∀ (i : Sigma fun i => (𝒱 i).J), Eq ((AlgebraicGeometry.Scheme.Hom.opensFu …
    hf : Eq (iSup U) Top.top
    ⊢ AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.morphismRestrict (Cat …
  -/
  rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := @IsClosedImmersion) _ hf]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    inst✝ : ∀ (i : 𝒰.J) (j : (𝒱 i).J), AlgebraicGeometry.IsAffine ((𝒱 i).obj j)
    U : (Sigma fun i => (𝒱 i).J) → (↑(AlgebraicGeometry.Scheme.Pullback.diagonalCo …
    hU : ∀ (i : Sigma fun i => (𝒱 i).J), Eq ((AlgebraicGeometry.Scheme.Hom.opensFu …
    hf : Eq (iSup U) Top.top
    ⊢ ∀ (i : Sigma fun i => (𝒱 i).J), AlgebraicGeometry.IsClosedImmersion (Algebra …
  -/
  intro i
  rw [MorphismProperty.arrow_mk_iso_iff (P := @IsClosedImmersion) (morphismRestrictRestrict _ _ _),
    MorphismProperty.arrow_mk_iso_iff (P := @IsClosedImmersion) (morphismRestrictEq _ (hU i)),
    MorphismProperty.arrow_mk_iso_iff (P := @IsClosedImmersion) (diagonalRestrictIsoDiagonal ..)]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    inst✝ : ∀ (i : 𝒰.J) (j : (𝒱 i).J), AlgebraicGeometry.IsAffine ((𝒱 i).obj j)
    U : (Sigma fun i => (𝒱 i).J) → (↑(AlgebraicGeometry.Scheme.Pullback.diagonalCo …
    hU : ∀ (i : Sigma fun i => (𝒱 i).J), Eq ((AlgebraicGeometry.Scheme.Hom.opensFu …
    hf : Eq (iSup U) Top.top
    i : Sigma fun i => (𝒱 i).J
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullback.diagonal …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[stacks 0DVA]
lemma isSeparated_of_injective (hf : Function.Injective f.base) :
    IsSeparated f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : Function.Injective ⇑f.base
    ⊢ AlgebraicGeometry.IsSeparated f
  -/
  constructor
  /-
    case diagonal_isClosedImmersion
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : Function.Injective ⇑f.base
    ⊢ autoParam (AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullba …
  -/
  let 𝒰 := Y.affineCover
  /-
    case diagonal_isClosedImmersion
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : Function.Injective ⇑f.base
    𝒰 : Y.OpenCover := Y.affineCover
    ⊢ autoParam (AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullba …
  -/
  let 𝒱 (i) := (pullback f (𝒰.map i)).affineCover
  /-
    case diagonal_isClosedImmersion
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : Function.Injective ⇑f.base
    𝒰 : Y.OpenCover := Y.affineCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover := fun  …
    ⊢ autoParam (AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullba …
  -/
  refine IsLocalAtTarget.of_iSup_eq_top (fun i : PUnit.{0} ↦ ⊤) (by simp) fun _ ↦ ?_
  /-
    case diagonal_isClosedImmersion
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : Function.Injective ⇑f.base
    𝒰 : Y.OpenCover := Y.affineCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover := fun  …
    x✝ : PUnit.{0}
    ⊢ AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.morphismRestrict (Cat …
  -/
  rw [← diagonalCoverDiagonalRange_eq_top_of_injective f 𝒰 𝒱 hf]
  /-
    case diagonal_isClosedImmersion
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : Function.Injective ⇑f.base
    𝒰 : Y.OpenCover := Y.affineCover
    𝒱 : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover := fun  …
    x✝ : PUnit.{0}
    ⊢ AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.morphismRestrict (Cat …
  -/
  exact isClosedImmersion_diagonal_restrict_diagonalCoverDiagonalRange f 𝒰 𝒱
  /-
    🎉 no goals
  -/


lemma IsClosedImmersion.of_comp [IsClosedImmersion (f ≫ g)] [IsSeparated g] :
    IsClosedImmersion f := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.co …
    inst✝ : AlgebraicGeometry.IsSeparated g
    ⊢ AlgebraicGeometry.IsClosedImmersion f
  -/
  rw [← pullback.lift_snd (𝟙 _) f (Category.id_comp (f ≫ g))]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.co …
    inst✝ : AlgebraicGeometry.IsSeparated g
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.comp (Cat …
  -/
  have := MorphismProperty.pullback_snd (P := @IsClosedImmersion) (f ≫ g) g inferInstance
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.co …
    inst✝ : AlgebraicGeometry.IsSeparated g
    this : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullback.snd …
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.comp (Cat …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma IsSeparated.of_comp [IsSeparated (f ≫ g)] : IsSeparated f := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsSeparated (CategoryTheory.CategoryStruct.comp f g)
    ⊢ AlgebraicGeometry.IsSeparated f
  -/
  have := IsSeparated.diagonal_isClosedImmersion (f := f ≫ g)
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsSeparated (CategoryTheory.CategoryStruct.comp f g)
    this : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pullback.dia …
    ⊢ AlgebraicGeometry.IsSeparated f
  -/
  rw [pullback.diagonal_comp] at this
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsSeparated (CategoryTheory.CategoryStruct.comp f g)
    this : AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.comp …
    ⊢ AlgebraicGeometry.IsSeparated f
  -/
  exact ⟨@IsClosedImmersion.of_comp _ _ _ _ _ this inferInstance⟩
  /-
    🎉 no goals
  -/


lemma IsSeparated.comp_iff [IsSeparated g] : IsSeparated (f ≫ g) ↔ IsSeparated f :=
  ⟨fun _ ↦ .of_comp f g, fun _ ↦ inferInstance⟩


@[stacks 01KM]
instance isClosedImmersion_equalizer_ι_left {S : Scheme} {X Y : Over S} [IsSeparated Y.hom]
    (f g : X ⟶ Y) : IsClosedImmersion (equalizer.ι f g).left := by
  refine MorphismProperty.of_isPullback
    ((Limits.isPullback_equalizer_prod f g).map (Over.forget _)).flip ?_
  rw [← MorphismProperty.cancel_right_of_respectsIso @IsClosedImmersion _
    (Over.prodLeftIsoPullback Y Y).hom]
  /-
    W X✝ Y✝ Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    g✝ : Quiver.Hom Y✝ Z
    S : AlgebraicGeometry.Scheme
    X Y : CategoryTheory.Over S
    inst✝ : AlgebraicGeometry.IsSeparated Y.hom
    f g : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.comp ((Ca …
  -/
  convert (inferInstanceAs (IsClosedImmersion (pullback.diagonal Y.hom)))
  /-
    case h.e'_3.h
    W X✝ Y✝ Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    g✝ : Quiver.Hom Y✝ Z
    S : AlgebraicGeometry.Scheme
    X Y : CategoryTheory.Over S
    inst✝ : AlgebraicGeometry.IsSeparated Y.hom
    f g : Quiver.Hom X Y
    e_1✝ : Eq ((CategoryTheory.Over.forget S).obj Y) ((CategoryTheory.Functor.id A …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Over.forget S).map ( …
  -/
           /-
             🎉 no goals
           -/
  ext1 <;> simp [← Over.comp_left]
           /-
             🎉 no goals
           -/


/--
Suppose `X` is a reduced scheme and that `f g : X ⟶ Y` agree over some separated `Y ⟶ Z`.
Then `f = g` if `ι ≫ f = ι ≫ g` for some dominant `ι`.
-/
lemma ext_of_isDominant_of_isSeparated [IsReduced X] {f g : X ⟶ Y}
    (s : Y ⟶ Z) [IsSeparated s] (h : f ≫ s = g ≫ s)
    (ι : W ⟶ X) [IsDominant ι] (hU : ι ≫ f = ι ≫ g) : f = g := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    ⊢ Eq f g
  -/
  let X' : Over Z := Over.mk (f ≫ s)
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    ⊢ Eq f g
  -/
  let Y' : Over Z := Over.mk s
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    Y' : CategoryTheory.Over Z := CategoryTheory.Over.mk s
    ⊢ Eq f g
  -/
  let U' : Over Z := Over.mk (ι ≫ f ≫ s)
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    Y' : CategoryTheory.Over Z := CategoryTheory.Over.mk s
    U' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    ⊢ Eq f g
  -/
  let f' : X' ⟶ Y' := Over.homMk f
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    Y' : CategoryTheory.Over Z := CategoryTheory.Over.mk s
    U' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    f' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk f ⋯
    ⊢ Eq f g
  -/
  let g' : X' ⟶ Y' := Over.homMk g
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    Y' : CategoryTheory.Over Z := CategoryTheory.Over.mk s
    U' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    f' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk f ⋯
    g' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk g ⋯
    ⊢ Eq f g
  -/
  let ι' : U' ⟶ X' := Over.homMk ι
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    Y' : CategoryTheory.Over Z := CategoryTheory.Over.mk s
    U' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    f' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk f ⋯
    g' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk g ⋯
    ι' : Quiver.Hom U' X' := CategoryTheory.Over.homMk ι ⋯
    ⊢ Eq f g
  -/
  have : IsSeparated Y'.hom := ‹_›
  have : IsDominant (equalizer.ι f' g').left := by
    apply (config := { allowSynthFailures := true }) IsDominant.of_comp (equalizer.lift ι' ?_).left
    · rwa [← Over.comp_left, equalizer.lift_ι]
    · ext1; exact hU
  have : Surjective (equalizer.ι f' g').left :=
    surjective_of_isDominant_of_isClosed_range _ IsClosedImmersion.base_closed.2
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    Y' : CategoryTheory.Over Z := CategoryTheory.Over.mk s
    U' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    f' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk f ⋯
    g' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk g ⋯
    ι' : Quiver.Hom U' X' := CategoryTheory.Over.homMk ι ⋯
    this✝¹ : AlgebraicGeometry.IsSeparated Y'.hom
    this✝ : AlgebraicGeometry.IsDominant (CategoryTheory.Limits.equalizer.ι f' g') …
    this : AlgebraicGeometry.Surjective (CategoryTheory.Limits.equalizer.ι f' g'). …
    ⊢ Eq f g
  -/
  have := isIso_of_isClosedImmersion_of_surjective (Y := X) (equalizer.ι f' g').left
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    Y' : CategoryTheory.Over Z := CategoryTheory.Over.mk s
    U' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    f' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk f ⋯
    g' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk g ⋯
    ι' : Quiver.Hom U' X' := CategoryTheory.Over.homMk ι ⋯
    this✝² : AlgebraicGeometry.IsSeparated Y'.hom
    this✝¹ : AlgebraicGeometry.IsDominant (CategoryTheory.Limits.equalizer.ι f' g' …
    this✝ : AlgebraicGeometry.Surjective (CategoryTheory.Limits.equalizer.ι f' g') …
    this : CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.ι f' g').left
    ⊢ Eq f g
  -/
  rw [← cancel_epi (equalizer.ι f' g').left]
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsReduced X
    f g : Quiver.Hom X Y
    s : Quiver.Hom Y Z
    inst✝¹ : AlgebraicGeometry.IsSeparated s
    h : Eq (CategoryTheory.CategoryStruct.comp f s) (CategoryTheory.CategoryStruct …
    ι : Quiver.Hom W X
    inst✝ : AlgebraicGeometry.IsDominant ι
    hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
    X' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    Y' : CategoryTheory.Over Z := CategoryTheory.Over.mk s
    U' : CategoryTheory.Over Z := CategoryTheory.Over.mk (CategoryTheory.CategoryS …
    f' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk f ⋯
    g' : Quiver.Hom X' Y' := CategoryTheory.Over.homMk g ⋯
    ι' : Quiver.Hom U' X' := CategoryTheory.Over.homMk ι ⋯
    this✝² : AlgebraicGeometry.IsSeparated Y'.hom
    this✝¹ : AlgebraicGeometry.IsDominant (CategoryTheory.Limits.equalizer.ι f' g' …
    this✝ : AlgebraicGeometry.Surjective (CategoryTheory.Limits.equalizer.ι f' g') …
    this : CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.ι f' g').left
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι f' …
  -/
  exact congr($(equalizer.condition f' g').left)
  /-
    🎉 no goals
  -/


variable (S) in
/--
Suppose `X` is a reduced `S`-scheme and `Y` is a separated `S`-scheme.
For any `S`-morphisms `f g : X ⟶ Y`, `f = g` if `ι ≫ f = ι ≫ g` for some dominant `ι`.
-/
lemma ext_of_isDominant_of_isSeparated' [X.Over S] [Y.Over S] [IsReduced X] [IsSeparated (Y ↘ S)]
    {f g : X ⟶ Y} [f.IsOver S] [g.IsOver S] {W} (ι : W ⟶ X) [IsDominant ι]
    (hU : ι ≫ f = ι ≫ g) : f = g :=
                                               /-
                                                 X Y S : AlgebraicGeometry.Scheme
                                                 inst✝⁶ : X.Over S
                                                 inst✝⁵ : Y.Over S
                                                 inst✝⁴ : AlgebraicGeometry.IsReduced X
                                                 inst✝³ : AlgebraicGeometry.IsSeparated (CategoryTheory.over Y S inferInstance)
                                                 f g : Quiver.Hom X Y
                                                 inst✝² : AlgebraicGeometry.Scheme.Hom.IsOver f S
                                                 inst✝¹ : AlgebraicGeometry.Scheme.Hom.IsOver g S
                                                 W : AlgebraicGeometry.Scheme
                                                 ι : Quiver.Hom W X
                                                 inst✝ : AlgebraicGeometry.IsDominant ι
                                                 hU : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruc …
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.over Y S inferInsta …
                                               -/
  ext_of_isDominant_of_isSeparated (Y ↘ S) (by simp) ι hU
                                               /-
                                                 🎉 no goals
                                               -/


/-- A scheme `X` is separated if it is separated over `⊤_ Scheme`. -/
@[mk_iff]
protected class IsSeparated (X : Scheme.{u}) : Prop where
  isSeparated_terminal_from : IsSeparated (terminal.from X)


lemma isSeparated_iff_isClosedImmersion_prod_lift {X : Scheme.{u}} :
    X.IsSeparated ↔ IsClosedImmersion (prod.lift (𝟙 X) (𝟙 X)) := by
  rw [isSeparated_iff, AlgebraicGeometry.isSeparated_iff, iff_iff_eq,
    ← MorphismProperty.cancel_right_of_respectsIso @IsClosedImmersion _ (prodIsoPullback X X).hom]
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Eq (autoParam (AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.pu …
  -/
  congr
  /-
    case e_f
    X : AlgebraicGeometry.Scheme
    ⊢ Eq (CategoryTheory.Limits.pullback.diagonal (CategoryTheory.Limits.terminal. …
  -/
              /-
                🎉 no goals
              -/
  ext : 1 <;> simp
              /-
                🎉 no goals
              -/


instance [X.IsSeparated] : IsClosedImmersion (prod.lift (𝟙 X) (𝟙 X)) := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : X.IsSeparated
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.Limits.prod.lift (Catego …
  -/
  rwa [← isSeparated_iff_isClosedImmersion_prod_lift]
  /-
    🎉 no goals
  -/


instance (priority := 900) {X : Scheme.{u}} [IsAffine X] : X.IsSeparated := ⟨inferInstance⟩


instance (priority := 900) [X.IsSeparated] : IsSeparated f := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : X.IsSeparated
    ⊢ AlgebraicGeometry.IsSeparated f
  -/
  apply (config := { allowSynthFailures := true }) @IsSeparated.of_comp (g := terminal.from Y)
  /-
    case inst
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : X.IsSeparated
    ⊢ AlgebraicGeometry.IsSeparated (CategoryTheory.CategoryStruct.comp f (Categor …
  -/
  rw [terminal.comp_from]
  /-
    case inst
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : X.IsSeparated
    ⊢ AlgebraicGeometry.IsSeparated (CategoryTheory.Limits.terminal.from X)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (f g : X ⟶ Y) [Y.IsSeparated] : IsClosedImmersion (Limits.equalizer.ι f g) :=
  MorphismProperty.of_isPullback (isPullback_equalizer_prod f g).flip inferInstance


instance IsSeparated.hasAffineProperty :
    HasAffineProperty @IsSeparated fun X _ _ _ ↦ X.IsSeparated := by
  /-
    W X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ AlgebraicGeometry.HasAffineProperty @AlgebraicGeometry.IsSeparated fun X x x …
  -/
  convert HasAffineProperty.of_isLocalAtTarget @IsSeparated with X Y f hY
  /-
    case h.e'_2.h.h.h.h.a
    W X✝ Y✝ Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    g : Quiver.Hom Y✝ Z
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ Iff X.IsSeparated (AlgebraicGeometry.AffineTargetMorphismProperty.of (@Algeb …
  -/
  rw [Scheme.isSeparated_iff, ← terminal.comp_from f, IsSeparated.comp_iff]
  /-
    case h.e'_2.h.h.h.h.a
    W X✝ Y✝ Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    g : Quiver.Hom Y✝ Z
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (AlgebraicGeometry.IsSeparated f) (AlgebraicGeometry.AffineTargetMorphis …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
Suppose `f g : X ⟶ Y` where `X` is a reduced scheme and `Y` is a separated scheme.
Then `f = g` if `ι ≫ f = ι ≫ g` for some dominant `ι`.

Also see `ext_of_isDominant_of_isSeparated` for the general version over arbitrary bases.
-/
lemma ext_of_isDominant [IsReduced X] {f g : X ⟶ Y} [Y.IsSeparated]
    (ι : W ⟶ X) [IsDominant ι] (hU : ι ≫ f = ι ≫ g) : f = g :=
  ext_of_isDominant_of_isSeparated (Limits.terminal.from _) (Limits.terminal.hom_ext _ _) ι hU


