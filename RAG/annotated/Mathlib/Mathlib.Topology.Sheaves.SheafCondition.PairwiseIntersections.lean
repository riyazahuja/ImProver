/-- An alternative formulation of the sheaf condition
(which we prove equivalent to the usual one below as
`isSheaf_iff_isSheafPairwiseIntersections`).

A presheaf is a sheaf if `F` sends the cone `(Pairwise.cocone U).op` to a limit cone.
(Recall `Pairwise.cocone U` has cone point `iSup U`, mapping down to the `U i` and the `U i ⊓ U j`.)
-/
def IsSheafPairwiseIntersections (F : Presheaf C X) : Prop :=
  ∀ ⦃ι : Type w⦄ (U : ι → Opens X), Nonempty (IsLimit (F.mapCone (Pairwise.cocone U).op))


/-- An alternative formulation of the sheaf condition
(which we prove equivalent to the usual one below as
`isSheaf_iff_isSheafPreservesLimitPairwiseIntersections`).

A presheaf is a sheaf if `F` preserves the limit of `Pairwise.diagram U`.
(Recall `Pairwise.diagram U` is the diagram consisting of the pairwise intersections
`U i ⊓ U j` mapping into the open sets `U i`. This diagram has limit `iSup U`.)
-/
def IsSheafPreservesLimitPairwiseIntersections (F : Presheaf C X) : Prop :=
  ∀ ⦃ι : Type w⦄ (U : ι → Opens X), Nonempty (PreservesLimit (Pairwise.diagram U).op F)


/-- Implementation detail:
the object level of `pairwiseToOpensLeCover : Pairwise ι ⥤ OpensLeCover U`
-/
@[simp]
def pairwiseToOpensLeCoverObj : Pairwise ι → OpensLeCover U
  | single i => ⟨U i, ⟨i, le_rfl⟩⟩
  | Pairwise.pair i j => ⟨U i ⊓ U j, ⟨i, inf_le_left⟩⟩


/-- Implementation detail:
the morphism level of `pairwiseToOpensLeCover : Pairwise ι ⥤ OpensLeCover U`
-/
def pairwiseToOpensLeCoverMap :
    ∀ {V W : Pairwise ι}, (V ⟶ W) → (pairwiseToOpensLeCoverObj U V ⟶ pairwiseToOpensLeCoverObj U W)
  | _, _, id_single _ => 𝟙 _
  | _, _, id_pair _ _ => 𝟙 _
  | _, _, left _ _ => homOfLE inf_le_left
  | _, _, right _ _ => homOfLE inf_le_right


/-- The category of single and double intersections of the `U i` maps into the category
of open sets below some `U i`.
-/
@[simps]
def pairwiseToOpensLeCover : Pairwise ι ⥤ OpensLeCover U where
  obj := pairwiseToOpensLeCoverObj U
  map {_ _} i := pairwiseToOpensLeCoverMap U i


instance (V : OpensLeCover U) : Nonempty (StructuredArrow V (pairwiseToOpensLeCover U)) :=
  ⟨@StructuredArrow.mk _ _ _ _ _ (single V.index) _ V.homToIndex⟩

-- This is a case bash: for each pair of types of objects in `Pairwise ι`,
-- we have to explicitly construct a zigzag.

/-- The diagram consisting of the `U i` and `U i ⊓ U j` is cofinal in the diagram
of all opens contained in some `U i`.
-/
instance : Functor.Final (pairwiseToOpensLeCover U) :=
  ⟨fun V =>
    isConnected_of_zigzag fun A B => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        ι : Type w
        U : ι → TopologicalSpace.Opens ↑X
        V : TopCat.Presheaf.SheafCondition.OpensLeCover U
        A B : CategoryTheory.StructuredArrow V (TopCat.Presheaf.SheafCondition.pairwis …
        ⊢ Exists fun l => And (List.Chain CategoryTheory.Zag A l) (Eq ((List.cons A l) …
      -/
      rcases A with ⟨⟨⟨⟩⟩, ⟨i⟩ | ⟨i, j⟩, a⟩ <;> rcases B with ⟨⟨⟨⟩⟩, ⟨i'⟩ | ⟨i', j'⟩, b⟩
      · refine
          ⟨[{   left := ⟨⟨⟩⟩
                right := pair i i'
                hom := (le_inf a.le b.le).hom }, _], ?_, rfl⟩
        exact
          List.Chain.cons
            (Or.inr
              ⟨{  left := 𝟙 _
                  right := left i i' }⟩)
            (List.Chain.cons
              (Or.inl
                ⟨{  left := 𝟙 _
                    right := right i i' }⟩)
              List.Chain.nil)
      · refine
          ⟨[{   left := ⟨⟨⟩⟩
                right := pair i' i
                hom := (le_inf (b.le.trans inf_le_left) a.le).hom },
              { left := ⟨⟨⟩⟩
                right := single i'
                hom := (b.le.trans inf_le_left).hom }, _], ?_, rfl⟩
        exact
          List.Chain.cons
            (Or.inr
              ⟨{  left := 𝟙 _
                  right := right i' i }⟩)
            (List.Chain.cons
              (Or.inl
                ⟨{  left := 𝟙 _
                    right := left i' i }⟩)
              (List.Chain.cons
                (Or.inr
                  ⟨{  left := 𝟙 _
                      right := left i' j' }⟩)
                List.Chain.nil))
      · refine
          ⟨[{   left := ⟨⟨⟩⟩
                right := single i
                hom := (a.le.trans inf_le_left).hom },
              { left := ⟨⟨⟩⟩
                right := pair i i'
                hom := (le_inf (a.le.trans inf_le_left) b.le).hom }, _], ?_, rfl⟩
        exact
          List.Chain.cons
            (Or.inl
              ⟨{  left := 𝟙 _
                  right := left i j }⟩)
            (List.Chain.cons
              (Or.inr
                ⟨{  left := 𝟙 _
                    right := left i i' }⟩)
              (List.Chain.cons
                (Or.inl
                  ⟨{  left := 𝟙 _
                      right := right i i' }⟩)
                List.Chain.nil))
      · refine
          ⟨[{   left := ⟨⟨⟩⟩
                right := single i
                hom := (a.le.trans inf_le_left).hom },
              { left := ⟨⟨⟩⟩
                right := pair i i'
                hom := (le_inf (a.le.trans inf_le_left) (b.le.trans inf_le_left)).hom },
              { left := ⟨⟨⟩⟩
                right := single i'
                hom := (b.le.trans inf_le_left).hom }, _], ?_, rfl⟩
        exact
          List.Chain.cons
            (Or.inl
              ⟨{  left := 𝟙 _
                  right := left i j }⟩)
            (List.Chain.cons
              (Or.inr
                ⟨{  left := 𝟙 _
                    right := left i i' }⟩)
              (List.Chain.cons
                (Or.inl
                  ⟨{  left := 𝟙 _
                      right := right i i' }⟩)
                (List.Chain.cons
                  (Or.inr
                    ⟨{  left := 𝟙 _
                        right := left i' j' }⟩)
                  List.Chain.nil)))⟩


/-- The diagram in `Opens X` indexed by pairwise intersections from `U` is isomorphic
(in fact, equal) to the diagram factored through `OpensLeCover U`.
-/
def pairwiseDiagramIso :
    Pairwise.diagram U ≅ pairwiseToOpensLeCover U ⋙ fullSubcategoryInclusion _ where
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       X : TopCat
                       ι : Type w
                       U : ι → TopologicalSpace.Opens ↑X
                       ⊢ (X_1 : CategoryTheory.Pairwise ι) → Quiver.Hom ((CategoryTheory.Pairwise.dia …
                     -/
                                             /-
                                               🎉 no goals
                                             -/
  hom := { app := by rintro (i | ⟨i, j⟩) <;> exact 𝟙 _ }
                                             /-
                                               🎉 no goals
                                             -/
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       X : TopCat
                       ι : Type w
                       U : ι → TopologicalSpace.Opens ↑X
                       ⊢ (X_1 : CategoryTheory.Pairwise ι) → Quiver.Hom (((TopCat.Presheaf.SheafCondi …
                     -/
                                             /-
                                               🎉 no goals
                                             -/
  inv := { app := by rintro (i | ⟨i, j⟩) <;> exact 𝟙 _ }
                                             /-
                                               🎉 no goals
                                             -/


/--
The cocone `Pairwise.cocone U` with cocone point `iSup U` over `Pairwise.diagram U` is isomorphic
to the cocone `opensLeCoverCocone U` (with the same cocone point)
after appropriate whiskering and postcomposition.
-/
def pairwiseCoconeIso :
    (Pairwise.cocone U).op ≅
      (Cones.postcomposeEquivalence (NatIso.op (pairwiseDiagramIso U : _) : _)).functor.obj
        ((opensLeCoverCocone U).op.whisker (pairwiseToOpensLeCover U).op) :=
                             /-
                               C : Type u
                               inst✝ : CategoryTheory.Category.{v, u} C
                               X : TopCat
                               ι : Type w
                               U : ι → TopologicalSpace.Opens ↑X
                               ⊢ ∀ (j : Opposite (CategoryTheory.Pairwise ι)), Eq ((CategoryTheory.Pairwise.c …
                             -/
  Cones.ext (Iso.refl _) (by aesop_cat)
                             /-
                               🎉 no goals
                             -/


/-- The sheaf condition
in terms of a limit diagram over all `{ V : Opens X // ∃ i, V ≤ U i }`
is equivalent to the reformulation
in terms of a limit diagram over `U i` and `U i ⊓ U j`.
-/
theorem isSheafOpensLeCover_iff_isSheafPairwiseIntersections :
    F.IsSheafOpensLeCover ↔ F.IsSheafPairwiseIntersections :=
  forall₂_congr fun _ U =>
    Equiv.nonempty_congr <|
      calc
        IsLimit (F.mapCone (opensLeCoverCocone U).op) ≃
            IsLimit ((F.mapCone (opensLeCoverCocone U).op).whisker (pairwiseToOpensLeCover U).op) :=
          (Functor.Initial.isLimitWhiskerEquiv (pairwiseToOpensLeCover U).op _).symm
        _ ≃ IsLimit (F.mapCone ((opensLeCoverCocone U).op.whisker (pairwiseToOpensLeCover U).op)) :=
          (IsLimit.equivIsoLimit F.mapConeWhisker.symm)
        _ ≃
            IsLimit
              ((Cones.postcomposeEquivalence _).functor.obj
                (F.mapCone ((opensLeCoverCocone U).op.whisker (pairwiseToOpensLeCover U).op))) :=
          (IsLimit.postcomposeHomEquiv _ _).symm
        _ ≃
            IsLimit
              (F.mapCone
                ((Cones.postcomposeEquivalence _).functor.obj
                  ((opensLeCoverCocone U).op.whisker (pairwiseToOpensLeCover U).op))) :=
          (IsLimit.equivIsoLimit (Functor.mapConePostcomposeEquivalenceFunctor _).symm)
        _ ≃ IsLimit (F.mapCone (Pairwise.cocone U).op) :=
          IsLimit.equivIsoLimit ((Cones.functoriality _ _).mapIso (pairwiseCoconeIso U : _).symm)


/-- The sheaf condition in terms of an equalizer diagram is equivalent
to the reformulation in terms of a limit diagram over `U i` and `U i ⊓ U j`.
-/
theorem isSheaf_iff_isSheafPairwiseIntersections : F.IsSheaf ↔ F.IsSheafPairwiseIntersections := by
  rw [isSheaf_iff_isSheafOpensLeCover,
    isSheafOpensLeCover_iff_isSheafPairwiseIntersections]


/-- The sheaf condition in terms of an equalizer diagram is equivalent
to the reformulation in terms of the presheaf preserving the limit of the diagram
consisting of the `U i` and `U i ⊓ U j`.
-/
theorem isSheaf_iff_isSheafPreservesLimitPairwiseIntersections :
    F.IsSheaf ↔ F.IsSheafPreservesLimitPairwiseIntersections := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Presheaf C X
    ⊢ Iff F.IsSheaf F.IsSheafPreservesLimitPairwiseIntersections
  -/
  rw [isSheaf_iff_isSheafPairwiseIntersections]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Presheaf C X
    ⊢ Iff F.IsSheafPairwiseIntersections F.IsSheafPreservesLimitPairwiseIntersecti …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Presheaf C X
      ⊢ F.IsSheafPairwiseIntersections → F.IsSheafPreservesLimitPairwiseIntersections
    -/
  · intro h ι U
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Presheaf C X
      h : F.IsSheafPairwiseIntersections
      ι : Type w
      U : ι → TopologicalSpace.Opens ↑X
      ⊢ Nonempty (CategoryTheory.Limits.PreservesLimit (CategoryTheory.Pairwise.diag …
    -/
    exact ⟨preservesLimit_of_preserves_limit_cone (Pairwise.coconeIsColimit U).op (h U).some⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Presheaf C X
      ⊢ F.IsSheafPreservesLimitPairwiseIntersections → F.IsSheafPairwiseIntersections
    -/
  · intro h ι U
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Presheaf C X
      h : F.IsSheafPreservesLimitPairwiseIntersections
      ι : Type w
      U : ι → TopologicalSpace.Opens ↑X
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (C …
    -/
    haveI := (h U).some
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Presheaf C X
      h : F.IsSheafPreservesLimitPairwiseIntersections
      ι : Type w
      U : ι → TopologicalSpace.Opens ↑X
      this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Pairwise.diagram U …
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.mapCone F (C …
    -/
    exact ⟨isLimitOfPreserves _ (Pairwise.coconeIsColimit U).op⟩
    /-
      🎉 no goals
    -/


/-- For a sheaf `F`, `F(U ⊔ V)` is the pullback of `F(U) ⟶ F(U ⊓ V)` and `F(V) ⟶ F(U ⊓ V)`.
This is the pullback cone. -/
def interUnionPullbackCone :
    PullbackCone (F.1.map (homOfLE inf_le_left : U ⊓ V ⟶ _).op)
      (F.1.map (homOfLE inf_le_right).op) :=
  PullbackCone.mk (F.1.map (homOfLE le_sup_left).op) (F.1.map (homOfLE le_sup_right).op) <| by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.val.map (CategoryTheory.homOfLE ⋯) …
    -/
    rw [← F.1.map_comp, ← F.1.map_comp]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      ⊢ Eq (F.val.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOfLE ⋯) …
    -/
    congr 1
    /-
      🎉 no goals
    -/


@[simp]
theorem interUnionPullbackCone_pt : (interUnionPullbackCone F U V).pt = F.1.obj (op <| U ⊔ V) :=
  rfl


@[simp]
theorem interUnionPullbackCone_fst :
    (interUnionPullbackCone F U V).fst = F.1.map (homOfLE le_sup_left).op :=
  rfl


@[simp]
theorem interUnionPullbackCone_snd :
    (interUnionPullbackCone F U V).snd = F.1.map (homOfLE le_sup_right).op :=
  rfl


/-- (Implementation).
Every cone over `F(U) ⟶ F(U ⊓ V)` and `F(V) ⟶ F(U ⊓ V)` factors through `F(U ⊔ V)`.
-/
def interUnionPullbackConeLift : s.pt ⟶ F.1.obj (op (U ⊔ V)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ⊢ Quiver.Hom s.pt (F.val.obj { unop := Max.max U V })
  -/
  let ι : ULift.{w} WalkingPair → Opens X := fun j => WalkingPair.casesOn j.down U V
  have hι : U ⊔ V = iSup ι := by
    ext
    rw [Opens.coe_iSup, Set.mem_iUnion]
    constructor
    · rintro (h | h)
      exacts [⟨⟨WalkingPair.left⟩, h⟩, ⟨⟨WalkingPair.right⟩, h⟩]
    · rintro ⟨⟨_ | _⟩, h⟩
      exacts [Or.inl h, Or.inr h]
  refine
    (F.presheaf.isSheaf_iff_isSheafPairwiseIntersections.mp F.2 ι).some.lift
        ⟨s.pt,
          { app := ?_
            naturality := ?_ }⟩ ≫
      F.1.map (eqToHom hι).op
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
      ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
      hι : Eq (Max.max U V) (iSup ι)
      ⊢ (X_1 : Opposite (CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Limits …
    -/
  · rintro ((_ | _) | (_ | _))
    exacts [s.fst, s.snd, s.fst ≫ F.1.map (homOfLE inf_le_left).op,
      s.snd ≫ F.1.map (homOfLE inf_le_left).op]
  /-
    case refine_2
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    ⊢ ∀ ⦃X_1 Y : Opposite (CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Li …
  -/
  rintro ⟨i⟩ ⟨j⟩ f
  /-
    case refine_2.op.op
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    i j : CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Limits.WalkingPair)
    f : Quiver.Hom { unop := i } { unop := j }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
  -/
  let g : j ⟶ i := f.unop
  /-
    case refine_2.op.op
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    i j : CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Limits.WalkingPair)
    f : Quiver.Hom { unop := i } { unop := j }
    g : Quiver.Hom j i := f.unop
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
  -/
  have : f = g.op := rfl
  /-
    case refine_2.op.op
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    i j : CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Limits.WalkingPair)
    f : Quiver.Hom { unop := i } { unop := j }
    g : Quiver.Hom j i := f.unop
    this : Eq f g.op
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
  -/
  clear_value g
  /-
    case refine_2.op.op
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    i j : CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Limits.WalkingPair)
    f : Quiver.Hom { unop := i } { unop := j }
    g : Quiver.Hom j i
    this : Eq f g.op
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
  -/
  subst this
  /-
    case refine_2.op.op
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    i j : CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Limits.WalkingPair)
    g : Quiver.Hom j i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
  -/
  rcases i with (⟨⟨_ | _⟩⟩ | ⟨⟨_ | _⟩, ⟨_⟩⟩) <;>
  /-
    case refine_2.op.op.single.up.left
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    j : CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Limits.WalkingPair)
    g : Quiver.Hom j (CategoryTheory.Pairwise.single { down := CategoryTheory.Limi …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
  -/
  rcases j with (⟨⟨_ | _⟩⟩ | ⟨⟨_ | _⟩, ⟨_⟩⟩) <;>
  /-
    case refine_2.op.op.single.up.left.single.up.left
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    g : Quiver.Hom (CategoryTheory.Pairwise.single { down := CategoryTheory.Limits …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  rcases g with ⟨⟩ <;>
  /-
    case refine_2.op.op.single.up.left.single.up.left.id_single
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
  -/
  dsimp [Pairwise.diagram] <;>
  /-
    case refine_2.op.op.single.up.left.single.up.left.id_single
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id s.p …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp only [ι, Category.id_comp, s.condition, CategoryTheory.Functor.map_id, Category.comp_id]
  /-
    🎉 no goals
  -/
  rw [← cancel_mono (F.1.map (eqToHom <| inf_comm U V : U ⊓ V ⟶ _).op), Category.assoc,
    Category.assoc, ← F.1.map_comp, ← F.1.map_comp]
  /-
    case refine_2.op.op.single.up.left.pair.up.right.up.right
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp s.snd (F.val.map (CategoryTheory.Cate …
  -/
  exact s.condition.symm
  /-
    🎉 no goals
  -/


theorem interUnionPullbackConeLift_left :
    interUnionPullbackConeLift F U V s ≫ F.1.map (homOfLE le_sup_left).op = s.fst := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U V s)  …
  -/
  erw [Category.assoc]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (⋯.some.lift { pt := s.pt, π := { app …
  -/
  simp_rw [← F.1.map_comp]
  exact
    (F.presheaf.isSheaf_iff_isSheafPairwiseIntersections.mp F.2 _).some.fac _ <|
      op <| Pairwise.single <| ULift.up WalkingPair.left


theorem interUnionPullbackConeLift_right :
    interUnionPullbackConeLift F U V s ≫ F.1.map (homOfLE le_sup_right).op = s.snd := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U V s)  …
  -/
  erw [Category.assoc]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (⋯.some.lift { pt := s.pt, π := { app …
  -/
  simp_rw [← F.1.map_comp]
  exact
    (F.presheaf.isSheaf_iff_isSheafPairwiseIntersections.mp F.2 _).some.fac _ <|
      op <| Pairwise.single <| ULift.up WalkingPair.right


/-- For a sheaf `F`, `F(U ⊔ V)` is the pullback of `F(U) ⟶ F(U ⊓ V)` and `F(V) ⟶ F(U ⊓ V)`. -/
def isLimitPullbackCone : IsLimit (interUnionPullbackCone F U V) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ⊢ CategoryTheory.Limits.IsLimit (F.interUnionPullbackCone U V)
  -/
  let ι : ULift.{w} WalkingPair → Opens X := fun ⟨j⟩ => WalkingPair.casesOn j U V
  have hι : U ⊔ V = iSup ι := by
    ext
    rw [Opens.coe_iSup, Set.mem_iUnion]
    constructor
    · rintro (h | h)
      exacts [⟨⟨WalkingPair.left⟩, h⟩, ⟨⟨WalkingPair.right⟩, h⟩]
    · rintro ⟨⟨_ | _⟩, h⟩
      exacts [Or.inl h, Or.inr h]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    ⊢ CategoryTheory.Limits.IsLimit (F.interUnionPullbackCone U V)
  -/
  apply PullbackCone.isLimitAux'
  /-
    case create
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    ⊢ (s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯ …
  -/
  intro s
  /-
    case create
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (F.interUnion …
  -/
  use interUnionPullbackConeLift F U V s
  /-
    case property
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    F : TopCat.Sheaf C X
    U V : TopologicalSpace.Opens ↑X
    s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
    ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
    hι : Eq (Max.max U V) (iSup ι)
    s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U  …
  -/
  refine ⟨?_, ?_, ?_⟩
    /-
      case property.refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
      ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
      hι : Eq (Max.max U V) (iSup ι)
      s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U V s)  …
    -/
  · apply interUnionPullbackConeLift_left
    /-
      🎉 no goals
    -/
    /-
      case property.refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
      ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
      hι : Eq (Max.max U V) (iSup ι)
      s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U V s)  …
    -/
  · apply interUnionPullbackConeLift_right
    /-
      🎉 no goals
    -/
    /-
      case property.refine_3
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
      ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
      hι : Eq (Max.max U V) (iSup ι)
      s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
      ⊢ ∀ {m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt}, Eq (CategoryTheor …
    -/
  · intro m h₁ h₂
    /-
      case property.refine_3
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
      ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
      hι : Eq (Max.max U V) (iSup ι)
      s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
      m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
      ⊢ Eq m (F.interUnionPullbackConeLift U V s)
    -/
    rw [← cancel_mono (F.1.map (eqToHom hι.symm).op)]
    /-
      case property.refine_3
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
      ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
      hι : Eq (Max.max U V) (iSup ι)
      s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
      m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.eqToHom  …
    -/
    apply (F.presheaf.isSheaf_iff_isSheafPairwiseIntersections.mp F.2 ι).some.hom_ext
    /-
      case property.refine_3
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
      ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
      hι : Eq (Max.max U V) (iSup ι)
      s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
      m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
      ⊢ ∀ (j : Opposite (CategoryTheory.Pairwise (ULift.{w, 0} CategoryTheory.Limits …
    -/
    rintro ((_ | _) | (_ | _)) <;>
    /-
      case property.refine_3.op.single.up.left
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : TopCat
      F : TopCat.Sheaf C X
      U V : TopologicalSpace.Opens ↑X
      s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
      ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
      hι : Eq (Max.max U V) (iSup ι)
      s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
      m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
    -/
    rw [Category.assoc, Category.assoc]
      /-
        case property.refine_3.op.single.up.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
      -/
    · erw [← F.1.map_comp]
      /-
        case property.refine_3.op.single.up.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.Category …
      -/
      convert h₁
      /-
        case h.e'_3.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        e_1✝ : Eq (Quiver.Hom s.pt (((CategoryTheory.Pairwise.diagram ι).op.comp F.pre …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U V s)  …
      -/
      apply interUnionPullbackConeLift_left
      /-
        🎉 no goals
      -/
      /-
        case property.refine_3.op.single.up.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
      -/
    · erw [← F.1.map_comp]
      /-
        case property.refine_3.op.single.up.right
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.Category …
      -/
      convert h₂
      /-
        case h.e'_3.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        e_1✝ : Eq (Quiver.Hom s.pt (((CategoryTheory.Pairwise.diagram ι).op.comp F.pre …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U V s)  …
      -/
      apply interUnionPullbackConeLift_right
      /-
        🎉 no goals
      -/
    all_goals
      dsimp only [Functor.op, Pairwise.cocone_ι_app, Functor.mapCone_π_app, Cocone.op,
        Pairwise.coconeιApp, unop_op, op_comp, NatTrans.op]
      simp_rw [F.1.map_comp, ← Category.assoc]
      congr 1
      simp_rw [Category.assoc, ← F.1.map_comp]
      /-
        case property.refine_3.op.pair.up.left.e_a
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        a✝ : ULift.{w, 0} CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.Category …
      -/
    · convert h₁
      /-
        case h.e'_3
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        a✝ : ULift.{w, 0} CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U V s)  …
      -/
      apply interUnionPullbackConeLift_left
      /-
        🎉 no goals
      -/
      /-
        case property.refine_3.op.pair.up.right.e_a
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        a✝ : ULift.{w, 0} CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (F.val.map (CategoryTheory.Category …
      -/
    · convert h₂
      /-
        case h.e'_3
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : TopCat
        F : TopCat.Sheaf C X
        U V : TopologicalSpace.Opens ↑X
        s✝ : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯). …
        ι : ULift.{w, 0} CategoryTheory.Limits.WalkingPair → TopologicalSpace.Opens ↑X …
        hι : Eq (Max.max U V) (iSup ι)
        s : CategoryTheory.Limits.PullbackCone (F.val.map (CategoryTheory.homOfLE ⋯).o …
        m : Quiver.Hom s.pt (F.interUnionPullbackCone U V).pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).f …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (F.interUnionPullbackCone U V).s …
        a✝ : ULift.{w, 0} CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.interUnionPullbackConeLift U V s)  …
      -/
      apply interUnionPullbackConeLift_right
      /-
        🎉 no goals
      -/


/-- If `U, V` are disjoint, then `F(U ⊔ V) = F(U) × F(V)`. -/
def isProductOfDisjoint (h : U ⊓ V = ⊥) :
    IsLimit
      (BinaryFan.mk (F.1.map (homOfLE le_sup_left : _ ⟶ U ⊔ V).op)
        (F.1.map (homOfLE le_sup_right : _ ⟶ U ⊔ V).op)) :=
  isProductOfIsTerminalIsPullback _ _ _ _ (F.isTerminalOfEqEmpty h) (isLimitPullbackCone F U V)


