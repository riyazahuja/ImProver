/-- A family of gluing data consists of
1. An index type `J`
2. A scheme `U i` for each `i : J`.
3. A scheme `V i j` for each `i j : J`.
  (Note that this is `J × J → Scheme` rather than `J → J → Scheme` to connect to the
  limits library easier.)
4. An open immersion `f i j : V i j ⟶ U i` for each `i j : ι`.
5. A transition map `t i j : V i j ⟶ V j i` for each `i j : ι`.
such that
6. `f i i` is an isomorphism.
7. `t i i` is the identity.
8. `V i j ×[U i] V i k ⟶ V i j ⟶ V j i` factors through `V j k ×[U j] V j i ⟶ V j i` via some
    `t' : V i j ×[U i] V i k ⟶ V j k ×[U j] V j i`.
9. `t' i j k ≫ t' j k i ≫ t' k i j = 𝟙 _`.

We can then glue the schemes `U i` together by identifying `V i j` with `V j i`, such
that the `U i`'s are open subschemes of the glued space.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): @[nolint has_nonempty_instance]; linter not ported yet
structure GlueData extends CategoryTheory.GlueData Scheme where
  f_open : ∀ i j, IsOpenImmersion (f i j)


local notation "𝖣" => D.toGlueData


/-- The glue data of locally ringed spaces associated to a family of glue data of schemes. -/
abbrev toLocallyRingedSpaceGlueData : LocallyRingedSpace.GlueData :=
  { f_open := D.f_open
    toGlueData := 𝖣.mapGlueData forgetToLocallyRingedSpace }


instance (i j : 𝖣.J) :
    LocallyRingedSpace.IsOpenImmersion ((D.toLocallyRingedSpaceGlueData).toGlueData.f i j) := by
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i j : D.J
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (D.toLocallyRingedSpace …
  -/
  apply GlueData.f_open
  /-
    🎉 no goals
  -/


instance (i j : 𝖣.J) :
    SheafedSpace.IsOpenImmersion
      (D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.toGlueData.f i j) := by
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i j : D.J
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (D.toLocallyRingedSpaceGlueDa …
  -/
  apply GlueData.f_open
  /-
    🎉 no goals
  -/


instance (i j : 𝖣.J) :
    PresheafedSpace.IsOpenImmersion
      (D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.toPresheafedSpaceGlueData.toGlueData.f
        i j) := by
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i j : D.J
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (D.toLocallyRingedSpaceGlu …
  -/
  apply GlueData.f_open
  /-
    🎉 no goals
  -/

-- Porting note: this was not needed.

instance (i : 𝖣.J) :
    LocallyRingedSpace.IsOpenImmersion ((D.toLocallyRingedSpaceGlueData).toGlueData.ι i) := by
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i : D.J
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (D.toLocallyRingedSpace …
  -/
  apply LocallyRingedSpace.GlueData.ι_isOpenImmersion
  /-
    🎉 no goals
  -/


/-- (Implementation). The glued scheme of a glue data.
This should not be used outside this file. Use `AlgebraicGeometry.Scheme.GlueData.glued` instead. -/
def gluedScheme : Scheme := by
  apply LocallyRingedSpace.IsOpenImmersion.scheme
    D.toLocallyRingedSpaceGlueData.toGlueData.glued
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    ⊢ ∀ (x : ↑D.toLocallyRingedSpaceGlueData.glued.toTopCat), Exists fun R => Exis …
  -/
  intro x
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    x : ↑D.toLocallyRingedSpaceGlueData.glued.toTopCat
    ⊢ Exists fun R => Exists fun f => And (Membership.mem (Set.range ⇑f.base) x) ( …
  -/
  obtain ⟨i, y, rfl⟩ := D.toLocallyRingedSpaceGlueData.ι_jointly_surjective x
  refine ⟨_, ((D.U i).affineCover.map y).toLRSHom ≫
    D.toLocallyRingedSpaceGlueData.toGlueData.ι i, ?_⟩
  /-
    case intro.intro
    D : AlgebraicGeometry.Scheme.GlueData
    i : D.toLocallyRingedSpaceGlueData.J
    y : ↑(D.toLocallyRingedSpaceGlueData.U i).toTopCat
    ⊢ And (Membership.mem (Set.range ⇑(CategoryTheory.CategoryStruct.comp (Algebra …
  -/
  constructor
    /-
      case intro.intro.left
      D : AlgebraicGeometry.Scheme.GlueData
      i : D.toLocallyRingedSpaceGlueData.J
      y : ↑(D.toLocallyRingedSpaceGlueData.U i).toTopCat
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.CategoryStruct.comp (AlgebraicGeo …
    -/
  · erw [TopCat.coe_comp, Set.range_comp] -- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case intro.intro.left
      D : AlgebraicGeometry.Scheme.GlueData
      i : D.toLocallyRingedSpaceGlueData.J
      y : ↑(D.toLocallyRingedSpaceGlueData.U i).toTopCat
      ⊢ Membership.mem (Set.image (⇑(AlgebraicGeometry.LocallyRingedSpace.Hom.toShHo …
    -/
    refine Set.mem_image_of_mem _ ?_
    /-
      case intro.intro.left
      D : AlgebraicGeometry.Scheme.GlueData
      i : D.toLocallyRingedSpaceGlueData.J
      y : ↑(D.toLocallyRingedSpaceGlueData.U i).toTopCat
      ⊢ Membership.mem (Set.range ⇑(AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom …
    -/
    exact (D.U i).affineCover.covers y
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.right
      D : AlgebraicGeometry.Scheme.GlueData
      i : D.toLocallyRingedSpaceGlueData.J
      y : ↑(D.toLocallyRingedSpaceGlueData.U i).toTopCat
      ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (CategoryTheory.Categor …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


instance : CreatesColimit 𝖣.diagram.multispan forgetToLocallyRingedSpace :=
  createsColimitOfFullyFaithfulOfIso D.gluedScheme
    (HasColimit.isoOfNatIso (𝖣.diagramIso forgetToLocallyRingedSpace).symm)


instance : PreservesColimit (𝖣.diagram.multispan) forgetToTop :=
  inferInstanceAs (PreservesColimit (𝖣.diagram).multispan (forgetToLocallyRingedSpace ⋙
      LocallyRingedSpace.forgetToSheafedSpace ⋙ SheafedSpace.forget CommRingCat))


instance : HasMulticoequalizer 𝖣.diagram :=
  hasColimit_of_created _ forgetToLocallyRingedSpace


/-- The glued scheme of a glued space. -/
abbrev glued : Scheme :=
  𝖣.glued


/-- The immersion from `D.U i` into the glued space. -/
abbrev ι (i : D.J) : D.U i ⟶ D.glued :=
  𝖣.ι i


/-- The gluing as sheafed spaces is isomorphic to the gluing as presheafed spaces. -/
abbrev isoLocallyRingedSpace :
    D.glued.toLocallyRingedSpace ≅ D.toLocallyRingedSpaceGlueData.toGlueData.glued :=
  𝖣.gluedIso forgetToLocallyRingedSpace


theorem ι_isoLocallyRingedSpace_inv (i : D.J) :
    D.toLocallyRingedSpaceGlueData.toGlueData.ι i ≫
      D.isoLocallyRingedSpace.inv = (𝖣.ι i).toLRSHom :=
  𝖣.ι_gluedIso_inv forgetToLocallyRingedSpace i


instance ι_isOpenImmersion (i : D.J) : IsOpenImmersion (𝖣.ι i) := by
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i : D.J
    ⊢ AlgebraicGeometry.IsOpenImmersion (D.ι i)
  -/
  rw [IsOpenImmersion, ← D.ι_isoLocallyRingedSpace_inv]; infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem ι_jointly_surjective (x : 𝖣.glued.carrier) :
    ∃ (i : D.J) (y : (D.U i).carrier), (D.ι i).base y = x :=
  𝖣.ι_jointly_surjective (forgetToTop ⋙ forget TopCat) x

-- Porting note: promote to higher priority to short circuit simplifier

@[simp (high), reassoc]
theorem glue_condition (i j : D.J) : D.t i j ≫ D.f j i ≫ D.ι j = D.f i j ≫ D.ι i :=
  𝖣.glue_condition i j


/-- The pullback cone spanned by `V i j ⟶ U i` and `V i j ⟶ U j`.
This is a pullback diagram (`vPullbackConeIsLimit`). -/
def vPullbackCone (i j : D.J) : PullbackCone (D.ι i) (D.ι j) :=
                                                    /-
                                                      D : AlgebraicGeometry.Scheme.GlueData
                                                      i j : D.J
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.f i j) (D.ι i)) (CategoryTheory.Ca …
                                                    -/
  PullbackCone.mk (D.f i j) (D.t i j ≫ D.f j i) (by simp)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The following diagram is a pullback, i.e. `Vᵢⱼ` is the intersection of `Uᵢ` and `Uⱼ` in `X`.

Vᵢⱼ ⟶ Uᵢ
 |      |
 ↓      ↓
 Uⱼ ⟶ X
-/
def vPullbackConeIsLimit (i j : D.J) : IsLimit (D.vPullbackCone i j) :=
  𝖣.vPullbackConeIsLimitOfMap forgetToLocallyRingedSpace i j
    (D.toLocallyRingedSpaceGlueData.vPullbackConeIsLimit _ _)

-- Porting note: new notation

local notation "D_" => TopCat.GlueData.toGlueData <|
  D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.toPresheafedSpaceGlueData.toTopGlueData


/-- The underlying topological space of the glued scheme is isomorphic to the gluing of the
underlying spaces -/
def isoCarrier :
    D.glued.carrier ≅ (D_).glued := by
  refine (PresheafedSpace.forget _).mapIso ?_ ≪≫
    GlueData.gluedIso _ (PresheafedSpace.forget.{_, _, u} _)
  refine SheafedSpace.forgetToPresheafedSpace.mapIso ?_ ≪≫
    SheafedSpace.GlueData.isoPresheafedSpace _
  refine LocallyRingedSpace.forgetToSheafedSpace.mapIso ?_ ≪≫
    LocallyRingedSpace.GlueData.isoSheafedSpace _
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    ⊢ CategoryTheory.Iso D.glued.toLocallyRingedSpace D.toLocallyRingedSpaceGlueDa …
  -/
  exact Scheme.GlueData.isoLocallyRingedSpace _
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_isoCarrier_inv (i : D.J) :
    (D_).ι i ≫ D.isoCarrier.inv = (D.ι i).base := by
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.toLocallyRingedSpaceGlueData.toShe …
  -/
  delta isoCarrier
  rw [Iso.trans_inv, GlueData.ι_gluedIso_inv_assoc, Functor.mapIso_inv, Iso.trans_inv,
    Functor.mapIso_inv, Iso.trans_inv, SheafedSpace.forgetToPresheafedSpace_map, forget_map,
    forget_map, ← PresheafedSpace.comp_base, ← Category.assoc,
    D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.ι_isoPresheafedSpace_inv i]
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.toLocallyRingedSpaceGlueData.toShe …
  -/
  erw [← Category.assoc, D.toLocallyRingedSpaceGlueData.ι_isoSheafedSpace_inv i]
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.toLocallyRingedSpaceGlueData.ι i). …
  -/
  change (_ ≫ D.isoLocallyRingedSpace.inv).base = _
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i : D.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.toLocallyRingedSpaceGlueData.ι i)  …
  -/
  rw [D.ι_isoLocallyRingedSpace_inv i]
  /-
    🎉 no goals
  -/


/-- An equivalence relation on `Σ i, D.U i` that holds iff `𝖣 .ι i x = 𝖣 .ι j y`.
See `AlgebraicGeometry.Scheme.GlueData.ι_eq_iff`. -/
def Rel (a b : Σ i, ((D.U i).carrier : Type _)) : Prop :=
  a = b ∨
    ∃ x : (D.V (a.1, b.1)).carrier, (D.f _ _).base x = a.2 ∧ (D.t _ _ ≫ D.f _ _).base x = b.2


theorem ι_eq_iff (i j : D.J) (x : (D.U i).carrier) (y : (D.U j).carrier) :
    (𝖣.ι i).base x = (𝖣.ι j).base y ↔ D.Rel ⟨i, x⟩ ⟨j, y⟩ := by
  refine Iff.trans ?_
    (TopCat.GlueData.ι_eq_iff_rel
      D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.toPresheafedSpaceGlueData.toTopGlueData
      i j x y)
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    i j : D.J
    x : ↑↑(D.U i).toPresheafedSpace
    y : ↑↑(D.U j).toPresheafedSpace
    ⊢ Iff (Eq ((D.ι i).base x) ((D.ι j).base y)) (Eq ((D.toLocallyRingedSpaceGlueD …
  -/
  rw [← ((TopCat.mono_iff_injective D.isoCarrier.inv).mp _).eq_iff, ← comp_apply]
    /-
      D : AlgebraicGeometry.Scheme.GlueData
      i j : D.J
      x : ↑↑(D.U i).toPresheafedSpace
      y : ↑↑(D.U j).toPresheafedSpace
      ⊢ Iff (Eq ((D.ι i).base x) ((D.ι j).base y)) (Eq ((CategoryTheory.CategoryStru …
    -/
  · simp_rw [← D.ι_isoCarrier_inv]
    /-
      D : AlgebraicGeometry.Scheme.GlueData
      i j : D.J
      x : ↑↑(D.U i).toPresheafedSpace
      y : ↑↑(D.U j).toPresheafedSpace
      ⊢ Iff (Eq ((CategoryTheory.CategoryStruct.comp (((((D.mapGlueData AlgebraicGeo …
    -/
    rfl -- `rfl` was not needed before https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      🎉 no goals
    -/
    /-
      D : AlgebraicGeometry.Scheme.GlueData
      i j : D.J
      x : ↑↑(D.U i).toPresheafedSpace
      y : ↑↑(D.U j).toPresheafedSpace
      ⊢ CategoryTheory.Mono D.isoCarrier.inv
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


theorem isOpen_iff (U : Set D.glued.carrier) : IsOpen U ↔ ∀ i, IsOpen ((D.ι i).base ⁻¹' U) := by
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    U : Set ↑↑D.glued.toPresheafedSpace
    ⊢ Iff (IsOpen U) (∀ (i : D.J), IsOpen (Set.preimage (⇑(D.ι i).base) U))
  -/
  rw [← (TopCat.homeoOfIso D.isoCarrier.symm).isOpen_preimage, TopCat.GlueData.isOpen_iff]
  /-
    D : AlgebraicGeometry.Scheme.GlueData
    U : Set ↑↑D.glued.toPresheafedSpace
    ⊢ Iff (∀ (i : D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.toPresheaf …
  -/
  apply forall_congr'
  /-
    case h
    D : AlgebraicGeometry.Scheme.GlueData
    U : Set ↑↑D.glued.toPresheafedSpace
    ⊢ ∀ (a : D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.toPresheafedSpa …
  -/
  intro i
  /-
    case h
    D : AlgebraicGeometry.Scheme.GlueData
    U : Set ↑↑D.glued.toPresheafedSpace
    i : D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.toPresheafedSpaceGlu …
    ⊢ Iff (IsOpen (Set.preimage (⇑(D.toLocallyRingedSpaceGlueData.toSheafedSpaceGl …
  -/
  rw [← Set.preimage_comp, ← ι_isoCarrier_inv]
  /-
    case h
    D : AlgebraicGeometry.Scheme.GlueData
    U : Set ↑↑D.glued.toPresheafedSpace
    i : D.toLocallyRingedSpaceGlueData.toSheafedSpaceGlueData.toPresheafedSpaceGlu …
    ⊢ Iff (IsOpen (Set.preimage (Function.comp ⇑(TopCat.homeoOfIso D.isoCarrier.sy …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The open cover of the glued space given by the glue data. -/
@[simps (config := .lemmasOnly)]
def openCover (D : Scheme.GlueData) : OpenCover D.glued where
  J := D.J
  obj := D.U
  map := D.ι
  f x := (D.ι_jointly_surjective x).choose
  covers x := ⟨_, (D.ι_jointly_surjective x).choose_spec.choose_spec⟩


/-- (Implementation) the transition maps in the glue data associated with an open cover. -/
def gluedCoverT' (x y z : 𝒰.J) :
    pullback (pullback.fst (𝒰.map x) (𝒰.map y)) (pullback.fst (𝒰.map x) (𝒰.map z)) ⟶
      pullback (pullback.fst (𝒰.map y) (𝒰.map z)) (pullback.fst (𝒰.map y) (𝒰.map x)) := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.f …
  -/
  refine (pullbackRightPullbackFstIso _ _ _).hom ≫ ?_
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine ?_ ≫ (pullbackSymmetry _ _).hom
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine ?_ ≫ (pullbackRightPullbackFstIso _ _ _).inv
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine pullback.map _ _ _ _ (pullbackSymmetry _ _).hom (𝟙 _) (𝟙 _) ?_ ?_
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      x y z : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp [pullback.condition]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      x y z : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝒰.map z) (CategoryTheory.CategoryStr …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp, reassoc]
theorem gluedCoverT'_fst_fst (x y z : 𝒰.J) :
    𝒰.gluedCoverT' x y z ≫ pullback.fst _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
  -/
  delta gluedCoverT'; simp
                      /-
                        🎉 no goals
                      -/


@[simp, reassoc]
theorem gluedCoverT'_fst_snd (x y z : 𝒰.J) :
    gluedCoverT' 𝒰 x y z ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
      pullback.snd _ _ ≫ pullback.snd _ _ := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
  -/
  delta gluedCoverT'; simp
                      /-
                        🎉 no goals
                      -/


@[simp, reassoc]
theorem gluedCoverT'_snd_fst (x y z : 𝒰.J) :
    gluedCoverT' 𝒰 x y z ≫ pullback.snd _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
  -/
  delta gluedCoverT'; simp
                      /-
                        🎉 no goals
                      -/


@[simp, reassoc]
theorem gluedCoverT'_snd_snd (x y z : 𝒰.J) :
    gluedCoverT' 𝒰 x y z ≫ pullback.snd _ _ ≫ pullback.snd _ _ =
      pullback.fst _ _ ≫ pullback.fst _ _ := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
  -/
  delta gluedCoverT'; simp
                      /-
                        🎉 no goals
                      -/


theorem glued_cover_cocycle_fst (x y z : 𝒰.J) :
    gluedCoverT' 𝒰 x y z ≫ gluedCoverT' 𝒰 y z x ≫ gluedCoverT' 𝒰 z x y ≫ pullback.fst _ _ =
      pullback.fst _ _ := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
  -/
                             /-
                               🎉 no goals
                             -/
  apply pullback.hom_ext <;> simp
                             /-
                               🎉 no goals
                             -/


theorem glued_cover_cocycle_snd (x y z : 𝒰.J) :
    gluedCoverT' 𝒰 x y z ≫ gluedCoverT' 𝒰 y z x ≫ gluedCoverT' 𝒰 z x y ≫ pullback.snd _ _ =
      pullback.snd _ _ := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
  -/
                             /-
                               🎉 no goals
                             -/
  apply pullback.hom_ext <;> simp [pullback.condition]
                             /-
                               🎉 no goals
                             -/


theorem glued_cover_cocycle (x y z : 𝒰.J) :
    gluedCoverT' 𝒰 x y z ≫ gluedCoverT' 𝒰 y z x ≫ gluedCoverT' 𝒰 z x y = 𝟙 _ := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y z : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
  -/
  apply pullback.hom_ext <;> simp_rw [Category.id_comp, Category.assoc]
    /-
      case h₀
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      x y z : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
    -/
  · apply glued_cover_cocycle_fst
    /-
      🎉 no goals
    -/
    /-
      case h₁
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      x y z : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.glued …
    -/
  · apply glued_cover_cocycle_snd
    /-
      🎉 no goals
    -/


/-- The glue data associated with an open cover.
The canonical isomorphism `𝒰.gluedCover.glued ⟶ X` is provided by `𝒰.fromGlued`. -/
@[simps]
def gluedCover : Scheme.GlueData.{u} where
  J := 𝒰.J
  U := 𝒰.obj
  V := fun ⟨x, y⟩ => pullback (𝒰.map x) (𝒰.map y)
  f _ _ := pullback.fst _ _
  f_id _ := inferInstance
  t _ _ := (pullbackSymmetry _ _).hom
               /-
                 X : AlgebraicGeometry.Scheme
                 𝒰 : X.OpenCover
                 x : 𝒰.J
                 ⊢ Eq ((fun x x_1 => (CategoryTheory.Limits.pullbackSymmetry (𝒰.map x) (𝒰.map x …
               -/
  t_id x := by simp
               /-
                 🎉 no goals
               -/
  t' x y z := gluedCoverT' 𝒰 x y z
                    /-
                      X : AlgebraicGeometry.Scheme
                      𝒰 : X.OpenCover
                      x y z : 𝒰.J
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x y z => AlgebraicGeometry.Sche …
                    -/
                                               /-
                                                 🎉 no goals
                                               -/
  t_fac x y z := by apply pullback.hom_ext <;> simp
                                               /-
                                                 🎉 no goals
                                               -/
  -- The `cocycle` field could have been `by tidy` but lean timeouts.
  cocycle x y z := glued_cover_cocycle 𝒰 x y z
  f_open _ := inferInstance


/-- The canonical morphism from the gluing of an open cover of `X` into `X`.
This is an isomorphism, as witnessed by an `IsIso` instance. -/
def fromGlued : 𝒰.gluedCover.glued ⟶ X := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    ⊢ Quiver.Hom (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued X
  -/
  fapply Multicoequalizer.desc
    /-
      case k
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      ⊢ (b : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).diagram.R) → Quiver.Hom ( …
    -/
  · exact fun x => 𝒰.map x
    /-
      🎉 no goals
    -/
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    ⊢ ∀ (a : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).diagram.L), Eq (Categor …
  -/
  rintro ⟨x, y⟩
  /-
    case h.mk
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.glue …
  -/
  change pullback.fst _ _ ≫ _ = ((pullbackSymmetry _ _).hom ≫ pullback.fst _ _) ≫ _
  /-
    case h.mk
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  simpa using pullback.condition
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem ι_fromGlued (x : 𝒰.J) : 𝒰.gluedCover.ι x ≫ 𝒰.fromGlued = 𝒰.map x :=
  Multicoequalizer.π_desc _ _ _ _ _


theorem fromGlued_injective : Function.Injective 𝒰.fromGlued.base := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base
  -/
  intro x y h
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x y : ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    h : Eq ((AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base x) ((AlgebraicGeomet …
    ⊢ Eq x y
  -/
  obtain ⟨i, x, rfl⟩ := 𝒰.gluedCover.ι_jointly_surjective x
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    y : ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    h : Eq ((AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base (((AlgebraicGeometry …
    ⊢ Eq (((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).ι i).base x) y
  -/
  obtain ⟨j, y, rfl⟩ := 𝒰.gluedCover.ι_jointly_surjective y
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
    h : Eq ((AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base (((AlgebraicGeometry …
    ⊢ Eq (((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).ι i).base x) (((Algebraic …
  -/
  rw [← comp_apply, ← comp_apply] at h
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
    h : Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.g …
    ⊢ Eq (((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).ι i).base x) (((Algebraic …
  -/
  simp_rw [← Scheme.comp_base] at h
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
    h : Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.g …
    ⊢ Eq (((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).ι i).base x) (((Algebraic …
  -/
  rw [ι_fromGlued, ι_fromGlued] at h
  let e :=
    (TopCat.pullbackConeIsLimit _ _).conePointUniqueUpToIso
      (isLimitOfHasPullbackOfPreservesLimit Scheme.forgetToTop (𝒰.map i) (𝒰.map j))
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
    h : Eq ((𝒰.map i).base x) ((𝒰.map j).base y)
    e : CategoryTheory.Iso (TopCat.pullbackCone (AlgebraicGeometry.Scheme.forgetTo …
    ⊢ Eq (((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).ι i).base x) (((Algebraic …
  -/
  rw [𝒰.gluedCover.ι_eq_iff]
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
    h : Eq ((𝒰.map i).base x) ((𝒰.map j).base y)
    e : CategoryTheory.Iso (TopCat.pullbackCone (AlgebraicGeometry.Scheme.forgetTo …
    ⊢ (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).Rel ⟨i, x⟩ ⟨j, y⟩
  -/
  right
  /-
    case intro.intro.intro.intro.h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
    h : Eq ((𝒰.map i).base x) ((𝒰.map j).base y)
    e : CategoryTheory.Iso (TopCat.pullbackCone (AlgebraicGeometry.Scheme.forgetTo …
    ⊢ Exists fun x_1 => And (Eq (((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).f  …
  -/
  use e.hom ⟨⟨x, y⟩, h⟩
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
    h : Eq ((𝒰.map i).base x) ((𝒰.map j).base y)
    e : CategoryTheory.Iso (TopCat.pullbackCone (AlgebraicGeometry.Scheme.forgetTo …
    ⊢ And (Eq (((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).f ⟨i, x⟩.fst ⟨j, y⟩. …
  -/
  constructor
    /-
      case h.left
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
      x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
      j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
      y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
      h : Eq ((𝒰.map i).base x) ((𝒰.map j).base y)
      e : CategoryTheory.Iso (TopCat.pullbackCone (AlgebraicGeometry.Scheme.forgetTo …
      ⊢ Eq (((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).f ⟨i, x⟩.fst ⟨j, y⟩.fst). …
    -/
  · erw [← comp_apply e.hom, IsLimit.conePointUniqueUpToIso_hom_comp _ _ WalkingCospan.left]; rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
  · erw [← comp_apply e.hom, pullbackSymmetry_hom_comp_fst,
      IsLimit.conePointUniqueUpToIso_hom_comp _ _ WalkingCospan.right]
    /-
      case h.right
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
      x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
      j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
      y : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U j).toPresheafedSpace
      h : Eq ((𝒰.map i).base x) ((𝒰.map j).base y)
      e : CategoryTheory.Iso (TopCat.pullbackCone (AlgebraicGeometry.Scheme.forgetTo …
      ⊢ Eq (((TopCat.pullbackCone (AlgebraicGeometry.Scheme.forgetToTop.map (𝒰.map i …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance fromGlued_stalk_iso (x : 𝒰.gluedCover.glued.carrier) :
    IsIso (𝒰.fromGlued.stalkMap x) := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x : ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
  -/
  obtain ⟨i, x, rfl⟩ := 𝒰.gluedCover.ι_jointly_surjective x
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
  -/
  have := stalkMap_congr_hom _ _ (𝒰.ι_fromGlued i) x
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    this : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
  -/
  rw [stalkMap_comp, ← IsIso.eq_comp_inv] at this
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    this : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeometry.Scheme.Cov …
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeomet …
  -/
  rw [this]
  /-
    case intro.intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    x : ↑↑((AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).U i).toPresheafedSpace
    this : Eq (AlgebraicGeometry.Scheme.Hom.stalkMap (AlgebraicGeometry.Scheme.Cov …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem fromGlued_open_map : IsOpenMap 𝒰.fromGlued.base := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    ⊢ IsOpenMap ⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base
  -/
  intro U hU
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    hU : IsOpen U
    ⊢ IsOpen (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base) U)
  -/
  rw [isOpen_iff_forall_mem_open]
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    hU : IsOpen U
    ⊢ ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (Set.image (⇑(AlgebraicGeometr …
  -/
  intro x hx
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    hU : IsOpen U
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
    ⊢ Exists fun t => And (HasSubset.Subset t (Set.image (⇑(AlgebraicGeometry.Sche …
  -/
  rw [𝒰.gluedCover.isOpen_iff] at hU
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
    ⊢ Exists fun t => And (HasSubset.Subset t (Set.image (⇑(AlgebraicGeometry.Sche …
  -/
  use 𝒰.fromGlued.base '' U ∩ Set.range (𝒰.map (𝒰.f x)).base
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
    ⊢ And (HasSubset.Subset (Inter.inter (Set.image (⇑(AlgebraicGeometry.Scheme.Co …
  -/
  use Set.inter_subset_left
  /-
    case right
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
    hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
    ⊢ And (IsOpen (Inter.inter (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGl …
  -/
  constructor
    /-
      case right.left
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
      hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
      ⊢ IsOpen (Inter.inter (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰 …
    -/
  · rw [← Set.image_preimage_eq_inter_range]
    /-
      case right.left
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
      hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
      ⊢ IsOpen (Set.image (⇑(𝒰.map (𝒰.f x)).base) (Set.preimage (⇑(𝒰.map (𝒰.f x)).ba …
    -/
    apply (show IsOpenImmersion (𝒰.map (𝒰.f x)) from inferInstance).base_open.isOpenMap
    /-
      case right.left.a
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
      hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
      ⊢ IsOpen (Set.preimage (⇑(𝒰.map (𝒰.f x)).base) (Set.image (⇑(AlgebraicGeometry …
    -/
    convert hU (𝒰.f x) using 1
    /-
      case h.e'_3.h
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
      hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
      e_1✝ : Eq ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace ↑↑((AlgebraicGeometry.Scheme.Cov …
      ⊢ Eq (Set.preimage (⇑(𝒰.map (𝒰.f x)).base) (Set.image (⇑(AlgebraicGeometry.Sch …
    -/
    rw [← ι_fromGlued]; erw [coe_comp]; rw [Set.preimage_comp]
    /-
      case h.e'_3.h
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
      hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
      e_1✝ : Eq ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace ↑↑((AlgebraicGeometry.Scheme.Cov …
      ⊢ Eq (Set.preimage (⇑(AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (Algebr …
    -/
    congr! 1
    /-
      case h.e'_3.h.h.e'_4
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
      hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
      e_1✝ : Eq ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace ↑↑((AlgebraicGeometry.Scheme.Cov …
      ⊢ Eq (Set.preimage (⇑(AlgebraicGeometry.LocallyRingedSpace.Hom.toShHom (Algebr …
    -/
    exact Set.preimage_image_eq _ 𝒰.fromGlued_injective
    /-
      🎉 no goals
    -/
    /-
      case right.right
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      U : Set ↑↑(AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued.toPresheafedSpace
      hU : ∀ (i : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J), IsOpen (Set.prei …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰). …
      ⊢ Membership.mem (Inter.inter (Set.image (⇑(AlgebraicGeometry.Scheme.Cover.fro …
    -/
  · exact ⟨hx, 𝒰.covers x⟩
    /-
      🎉 no goals
    -/


theorem fromGlued_isOpenEmbedding : IsOpenEmbedding 𝒰.fromGlued.base :=
                                         /-
                                           X : AlgebraicGeometry.Scheme
                                           𝒰 : X.OpenCover
                                           ⊢ Continuous ⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base
                                         -/
  .of_continuous_injective_isOpenMap (by fun_prop) 𝒰.fromGlued_injective 𝒰.fromGlued_open_map
                                         /-
                                           🎉 no goals
                                         -/


@[deprecated (since := "2024-10-18")]
alias fromGlued_openEmbedding := fromGlued_isOpenEmbedding


instance : Epi 𝒰.fromGlued.base := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    ⊢ CategoryTheory.Epi (AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base
  -/
  rw [TopCat.epi_iff_surjective]
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base
  -/
  intro x
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x : ↑↑X.toPresheafedSpace
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base a) x
  -/
  obtain ⟨y, h⟩ := 𝒰.covers x
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x : ↑↑X.toPresheafedSpace
    y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace
    h : Eq ((𝒰.map (𝒰.f x)).base y) x
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base a) x
  -/
  use (𝒰.gluedCover.ι (𝒰.f x)).base y
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x : ↑↑X.toPresheafedSpace
    y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace
    h : Eq ((𝒰.map (𝒰.f x)).base y) x
    ⊢ Eq ((AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).base (((AlgebraicGeometry.S …
  -/
  rw [← comp_apply]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x : ↑↑X.toPresheafedSpace
    y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace
    h : Eq ((𝒰.map (𝒰.f x)).base y) x
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.glu …
  -/
  rw [← 𝒰.ι_fromGlued (𝒰.f x)] at h
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    x : ↑↑X.toPresheafedSpace
    y : ↑↑(𝒰.obj (𝒰.f x)).toPresheafedSpace
    h : Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.g …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.glu …
  -/
  exact h
  /-
    🎉 no goals
  -/


instance fromGlued_open_immersion : IsOpenImmersion 𝒰.fromGlued :=
  IsOpenImmersion.of_stalk_iso _ 𝒰.fromGlued_isOpenEmbedding


instance : IsIso 𝒰.fromGlued :=
  let F := Scheme.forgetToLocallyRingedSpace ⋙ LocallyRingedSpace.forgetToSheafedSpace ⋙
    SheafedSpace.forgetToPresheafedSpace
  have : IsIso (F.map (fromGlued 𝒰)) := by
    /-
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      F : CategoryTheory.Functor AlgebraicGeometry.Scheme (AlgebraicGeometry.Preshea …
      ⊢ CategoryTheory.IsIso (F.map (AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰))
    -/
    change IsIso 𝒰.fromGlued.toPshHom
    /-
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      F : CategoryTheory.Functor AlgebraicGeometry.Scheme (AlgebraicGeometry.Preshea …
      ⊢ CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Cover.fromGlued 𝒰).toPshHom
    -/
    apply PresheafedSpace.IsOpenImmersion.to_iso
    /-
      🎉 no goals
    -/
  isIso_of_reflects_iso _ F


/-- Given an open cover of `X`, and a morphism `𝒰.obj x ⟶ Y` for each open subscheme in the cover,
such that these morphisms are compatible in the intersection (pullback), we may glue the morphisms
together into a morphism `X ⟶ Y`.

Note:
If `X` is exactly (defeq to) the gluing of `U i`, then using `Multicoequalizer.desc` suffices.
-/
def glueMorphisms {Y : Scheme} (f : ∀ x, 𝒰.obj x ⟶ Y)
    (hf : ∀ x y, pullback.fst (𝒰.map x) (𝒰.map y) ≫ f x = pullback.snd _ _ ≫ f y) :
    X ⟶ Y := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
    hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    ⊢ Quiver.Hom X Y
  -/
  refine inv 𝒰.fromGlued ≫ ?_
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
    hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    ⊢ Quiver.Hom (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).glued Y
  -/
  fapply Multicoequalizer.desc
    /-
      case k
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      Y : AlgebraicGeometry.Scheme
      f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
      hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
      ⊢ (b : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).diagram.R) → Quiver.Hom ( …
    -/
  · exact f
    /-
      🎉 no goals
    -/
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
    hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    ⊢ ∀ (a : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).diagram.L), Eq (Categor …
  -/
  rintro ⟨i, j⟩
  /-
    case h.mk
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
    hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    i j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.glue …
  -/
  change pullback.fst _ _ ≫ f i = (_ ≫ _) ≫ f j
  /-
    case h.mk
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
    hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    i j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  erw [pullbackSymmetry_hom_comp_fst]
  /-
    case h.mk
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
    hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    i j : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  exact hf i j
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem ι_glueMorphisms {Y : Scheme} (f : ∀ x, 𝒰.obj x ⟶ Y)
    (hf : ∀ x y, pullback.fst (𝒰.map x) (𝒰.map y) ≫ f x = pullback.snd _ _ ≫ f y)
    (x : 𝒰.J) : 𝒰.map x ≫ 𝒰.glueMorphisms f hf = f x := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
    hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    x : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) (AlgebraicGeometry.Scheme.C …
  -/
  rw [← ι_fromGlued, Category.assoc]
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f : (x : 𝒰.J) → Quiver.Hom (𝒰.obj x) Y
    hf : ∀ (x y : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    x : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.glue …
  -/
  erw [IsIso.hom_inv_id_assoc, Multicoequalizer.π_desc]
  /-
    🎉 no goals
  -/


theorem hom_ext {Y : Scheme} (f₁ f₂ : X ⟶ Y) (h : ∀ x, 𝒰.map x ≫ f₁ = 𝒰.map x ≫ f₂) : f₁ = f₂ := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f₁ f₂ : Quiver.Hom X Y
    h : ∀ (x : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) f₁) (Categor …
    ⊢ Eq f₁ f₂
  -/
  rw [← cancel_epi 𝒰.fromGlued]
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f₁ f₂ : Quiver.Hom X Y
    h : ∀ (x : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) f₁) (Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.fromG …
  -/
  apply Multicoequalizer.hom_ext
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f₁ f₂ : Quiver.Hom X Y
    h : ∀ (x : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) f₁) (Categor …
    ⊢ ∀ (b : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).diagram.R), Eq (Categor …
  -/
  intro x
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f₁ f₂ : Quiver.Hom X Y
    h : ∀ (x : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) f₁) (Categor …
    x : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multicoequaliz …
  -/
  erw [Multicoequalizer.π_desc_assoc]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f₁ f₂ : Quiver.Hom X Y
    h : ∀ (x : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) f₁) (Categor …
    x : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) f₁) (CategoryTheory.Categor …
  -/
  erw [Multicoequalizer.π_desc_assoc]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    Y : AlgebraicGeometry.Scheme
    f₁ f₂ : Quiver.Hom X Y
    h : ∀ (x : 𝒰.J), Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) f₁) (Categor …
    x : (AlgebraicGeometry.Scheme.Cover.gluedCover 𝒰).diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝒰.map x) f₁) (CategoryTheory.Categor …
  -/
  exact h x
  /-
    🎉 no goals
  -/


