/-- Auxiliary definition to define the topology on `X(*)` for a light condensed set `X`. -/
private def coinducingCoprod :
    (Σ (i : (S : LightProfinite.{u}) × X.val.obj ⟨S⟩), i.fst) →
      X.val.obj ⟨LightProfinite.of PUnit⟩ :=
  fun ⟨⟨_, i⟩, s⟩ ↦ X.val.map ((of PUnit.{u+1}).const s).op i


/-- Let `X` be a light condensed set. We define a topology on `X(*)` as the quotient topology of
all the maps from light profinite sets `S` to `X(*)`, corresponding to elements of `X(S)`.
In other words, the topology coinduced by the map `LightCondSet.coinducingCoprod` above. -/
local instance underlyingTopologicalSpace :
    TopologicalSpace (X.val.obj ⟨LightProfinite.of PUnit⟩) :=
  TopologicalSpace.coinduced (coinducingCoprod X) inferInstance


/-- The object part of the functor `LightCondSet ⥤ TopCat` -/
def toTopCat : TopCat.{u} := TopCat.of (X.val.obj ⟨LightProfinite.of PUnit⟩)


lemma continuous_coinducingCoprod {S : LightProfinite.{u}} (x : X.val.obj ⟨S⟩) :
    Continuous fun a ↦ (X.coinducingCoprod ⟨⟨S, x⟩, a⟩) := by
  suffices ∀ (i : (T : LightProfinite.{u}) × X.val.obj ⟨T⟩),
      Continuous (fun (a : i.fst) ↦ X.coinducingCoprod ⟨i, a⟩) from this ⟨_, _⟩
  /-
    X : LightCondSet
    S : LightProfinite
    x : X.val.obj { unop := S }
    ⊢ ∀ (i : Sigma fun T => X.val.obj { unop := T }), Continuous fun a => LightCon …
  -/
  rw [← continuous_sigma_iff]
  /-
    X : LightCondSet
    S : LightProfinite
    x : X.val.obj { unop := S }
    ⊢ Continuous (LightCondSet.coinducingCoprod X)
  -/
  apply continuous_coinduced_rng
  /-
    🎉 no goals
  -/


/-- The map part of the functor `LightCondSet ⥤ TopCat` -/
@[simps]
def toTopCatMap : X.toTopCat ⟶ Y.toTopCat where
  toFun := f.val.app ⟨LightProfinite.of PUnit⟩
  continuous_toFun := by
    /-
      X Y : LightCondSet
      f : Quiver.Hom X Y
      ⊢ Continuous (f.val.app { unop := LightProfinite.of PUnit.{u + 1} })
    -/
    rw [continuous_coinduced_dom]
    /-
      X Y : LightCondSet
      f : Quiver.Hom X Y
      ⊢ Continuous (Function.comp (f.val.app { unop := LightProfinite.of PUnit.{u +  …
    -/
    apply continuous_sigma
    /-
      case hf
      X Y : LightCondSet
      f : Quiver.Hom X Y
      ⊢ ∀ (i : Sigma fun S => X.val.obj { unop := S }), Continuous fun a => Function …
    -/
    intro ⟨S, x⟩
    /-
      case hf
      X Y : LightCondSet
      f : Quiver.Hom X Y
      S : LightProfinite
      x : X.val.obj { unop := S }
      ⊢ Continuous fun a => Function.comp (f.val.app { unop := LightProfinite.of PUn …
    -/
    simp only [Function.comp_apply, coinducingCoprod]
    rw [show (fun (a : S) ↦ f.val.app ⟨of PUnit⟩ (X.val.map ((of PUnit.{u+1}).const a).op x)) = _
      from funext fun a ↦ NatTrans.naturality_apply f.val ((of PUnit.{u+1}).const a).op x]
    /-
      case hf
      X Y : LightCondSet
      f : Quiver.Hom X Y
      S : LightProfinite
      x : X.val.obj { unop := S }
      ⊢ Continuous fun a => (Y.val.map (CompHausLike.const (LightProfinite.of PUnit. …
    -/
    exact continuous_coinducingCoprod _ _
    /-
      🎉 no goals
    -/


/-- The functor `LightCondSet ⥤ TopCat` -/
@[simps]
def _root_.lightCondSetToTopCat : LightCondSet.{u} ⥤ TopCat.{u} where
  obj X := X.toTopCat
  map f := toTopCatMap f


/-- The counit of the adjunction `lightCondSetToTopCat ⊣ topCatToLightCondSet` -/
def topCatAdjunctionCounit (X : TopCat.{u}) : X.toLightCondSet.toTopCat ⟶ X where
  toFun x := x.1 PUnit.unit
  continuous_toFun := by
    /-
      X✝ Y : LightCondSet
      f : Quiver.Hom X✝ Y
      X : TopCat
      ⊢ Continuous fun x => x.toFun PUnit.unit
    -/
    rw [continuous_coinduced_dom]
    /-
      X✝ Y : LightCondSet
      f : Quiver.Hom X✝ Y
      X : TopCat
      ⊢ Continuous (Function.comp (fun x => x.toFun PUnit.unit) (LightCondSet.coindu …
    -/
    continuity
    /-
      🎉 no goals
    -/


/-- The counit of the adjunction `lightCondSetToTopCat ⊣ topCatToLightCondSet` is always bijective,
but not an isomorphism in general (the inverse isn't continuous unless `X` is sequential).
-/
def topCatAdjunctionCounitEquiv (X : TopCat.{u}) : X.toLightCondSet.toTopCat ≃ X where
  toFun := topCatAdjunctionCounit X
  invFun x := ContinuousMap.const _ x
  left_inv _ := rfl
  right_inv _ := rfl


lemma topCatAdjunctionCounit_bijective (X : TopCat.{u}) :
    Function.Bijective (topCatAdjunctionCounit X) :=
  (topCatAdjunctionCounitEquiv X).bijective


/-- The unit of the adjunction `lightCondSetToTopCat ⊣ topCatToLightCondSet` -/
@[simps val_app val_app_apply]
def topCatAdjunctionUnit (X : LightCondSet.{u}) : X ⟶ X.toTopCat.toLightCondSet where
  val := {
    app := fun S x ↦ {
      toFun := fun s ↦ X.val.map ((of PUnit.{u+1}).const s).op x
      continuous_toFun := by
        suffices ∀ (i : (T : LightProfinite.{u}) × X.val.obj ⟨T⟩),
          Continuous (fun (a : i.fst) ↦ X.coinducingCoprod ⟨i, a⟩) from this ⟨_, _⟩
        /-
          X✝ Y : LightCondSet
          f : Quiver.Hom X✝ Y
          X : LightCondSet
          S : Opposite LightProfinite
          x : X.val.obj S
          ⊢ ∀ (i : Sigma fun T => X.val.obj { unop := T }), Continuous fun a => LightCon …
        -/
        rw [← continuous_sigma_iff]
        /-
          X✝ Y : LightCondSet
          f : Quiver.Hom X✝ Y
          X : LightCondSet
          S : Opposite LightProfinite
          x : X.val.obj S
          ⊢ Continuous (LightCondSet.coinducingCoprod X)
        -/
        apply continuous_coinduced_rng }
        /-
          🎉 no goals
        -/
    naturality := fun _ _ _ ↦ by
      /-
        X✝ Y : LightCondSet
        f : Quiver.Hom X✝ Y
        X : LightCondSet
        x✝² x✝¹ : Opposite LightProfinite
        x✝ : Quiver.Hom x✝² x✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.val.map x✝) ((fun S x => { toFun : …
      -/
      ext
      simp only [TopCat.toSheafCompHausLike_val_obj, CompHausLike.compHausLikeToTop_obj,
        Opposite.op_unop, types_comp_apply, TopCat.toSheafCompHausLike_val_map,
        ← FunctorToTypes.map_comp_apply]
      /-
        case h
        X✝ Y : LightCondSet
        f : Quiver.Hom X✝ Y
        X : LightCondSet
        x✝² x✝¹ : Opposite LightProfinite
        x✝ : Quiver.Hom x✝² x✝¹
        a✝ : X.val.obj x✝²
        ⊢ Eq { toFun := fun s => X.val.map (CategoryTheory.CategoryStruct.comp x✝ (Com …
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- The adjunction `lightCondSetToTopCat ⊣ topCatToLightCondSet` -/
noncomputable def topCatAdjunction : lightCondSetToTopCat.{u} ⊣ topCatToLightCondSet where
  unit := { app := topCatAdjunctionUnit }
  counit := { app := topCatAdjunctionCounit }
  left_triangle_components Y := by
    /-
      X Y✝ : LightCondSet
      f : Quiver.Hom X Y✝
      Y : LightCondSet
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (lightCondSetToTopCat.map ({ app := L …
    -/
    ext
    /-
      case w
      X Y✝ : LightCondSet
      f : Quiver.Hom X Y✝
      Y : LightCondSet
      x✝ : (CategoryTheory.forget TopCat).obj (lightCondSetToTopCat.obj ((CategoryTh …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (lightCondSetToTopCat.map ({ app :=  …
    -/
    change Y.val.map (𝟙 _) _ = _
    /-
      case w
      X Y✝ : LightCondSet
      f : Quiver.Hom X Y✝
      Y : LightCondSet
      x✝ : (CategoryTheory.forget TopCat).obj (lightCondSetToTopCat.obj ((CategoryTh …
      ⊢ Eq (Y.val.map (CategoryTheory.CategoryStruct.id { unop := Opposite.unop { un …
    -/
    simp
    /-
      🎉 no goals
    -/


instance (X : TopCat) : Epi (topCatAdjunction.counit.app X) := by
  /-
    X✝ Y : LightCondSet
    f : Quiver.Hom X✝ Y
    X : TopCat
    ⊢ CategoryTheory.Epi (LightCondSet.topCatAdjunction.counit.app X)
  -/
  rw [TopCat.epi_iff_surjective]
  /-
    X✝ Y : LightCondSet
    f : Quiver.Hom X✝ Y
    X : TopCat
    ⊢ Function.Surjective ⇑(LightCondSet.topCatAdjunction.counit.app X)
  -/
  exact (topCatAdjunctionCounit_bijective _).2
  /-
    🎉 no goals
  -/


instance : topCatToLightCondSet.Faithful := topCatAdjunction.faithful_R_of_epi_counit_app


instance (X : LightCondSet.{u}) : SequentialSpace X.toTopCat := by
  /-
    X✝ Y : LightCondSet
    f : Quiver.Hom X✝ Y
    X : LightCondSet
    ⊢ SequentialSpace ↑X.toTopCat
  -/
  apply SequentialSpace.coinduced
  /-
    🎉 no goals
  -/


instance (X : LightCondSet.{u}) : SequentialSpace (lightCondSetToTopCat.obj X) :=
  inferInstanceAs (SequentialSpace X.toTopCat)


/-- The functor from light condensed sets to topological spaces lands in sequential spaces. -/
def lightCondSetToSequential : LightCondSet.{u} ⥤ Sequential.{u} where
  obj X := Sequential.of (lightCondSetToTopCat.obj X)
  map f := toTopCatMap f


/--
The functor from topological spaces to light condensed sets restricted to sequential spaces.
-/
noncomputable def sequentialToLightCondSet :
    Sequential.{u} ⥤ LightCondSet.{u} :=
  sequentialToTop ⋙ topCatToLightCondSet


/--
The adjunction `lightCondSetToTopCat ⊣ topCatToLightCondSet` restricted to sequential
spaces.
-/
noncomputable def sequentialAdjunction :
    lightCondSetToSequential ⊣ sequentialToLightCondSet :=
  topCatAdjunction.restrictFullyFaithful (iC := 𝟭 _) (iD := sequentialToTop)
    (Functor.FullyFaithful.id _) fullyFaithfulSequentialToTop
    (Iso.refl _) (Iso.refl _)


/--
The counit of the adjunction `lightCondSetToSequential ⊣ sequentialToLightCondSet`
is a homeomorphism.

Note: for now, we only have `ℕ∪{∞}` as a light profinite set at universe level 0, which is why we
can only prove this for `X : TopCat.{0}`.
-/
def sequentialAdjunctionHomeo (X : TopCat.{0}) [SequentialSpace X] :
    X.toLightCondSet.toTopCat ≃ₜ X where
  toEquiv := topCatAdjunctionCounitEquiv X
  continuous_toFun := (topCatAdjunctionCounit X).continuous
  continuous_invFun := by
    /-
      X✝ Y : LightCondSet
      f : Quiver.Hom X✝ Y
      X : TopCat
      inst✝ : SequentialSpace ↑X
      ⊢ Continuous (LightCondSet.topCatAdjunctionCounitEquiv X).invFun
    -/
    apply SeqContinuous.continuous
    /-
      case hf
      X✝ Y : LightCondSet
      f : Quiver.Hom X✝ Y
      X : TopCat
      inst✝ : SequentialSpace ↑X
      ⊢ SeqContinuous (LightCondSet.topCatAdjunctionCounitEquiv X).invFun
    -/
    unfold SeqContinuous
    /-
      case hf
      X✝ Y : LightCondSet
      f : Quiver.Hom X✝ Y
      X : TopCat
      inst✝ : SequentialSpace ↑X
      ⊢ ∀ ⦃x : Nat → ↑X⦄ ⦃p : ↑X⦄, Filter.Tendsto x Filter.atTop (nhds p) → Filter.T …
    -/
    intro f p h
    /-
      case hf
      X✝ Y : LightCondSet
      f✝ : Quiver.Hom X✝ Y
      X : TopCat
      inst✝ : SequentialSpace ↑X
      f : Nat → ↑X
      p : ↑X
      h : Filter.Tendsto f Filter.atTop (nhds p)
      ⊢ Filter.Tendsto (Function.comp (LightCondSet.topCatAdjunctionCounitEquiv X).i …
    -/
    let g := (topCatAdjunctionCounitEquiv X).invFun ∘ (OnePoint.continuousMapMkNat f p h)
    /-
      case hf
      X✝ Y : LightCondSet
      f✝ : Quiver.Hom X✝ Y
      X : TopCat
      inst✝ : SequentialSpace ↑X
      f : Nat → ↑X
      p : ↑X
      h : Filter.Tendsto f Filter.atTop (nhds p)
      g : OnePoint Nat → ↑X.toLightCondSet.toTopCat := Function.comp (LightCondSet.t …
      ⊢ Filter.Tendsto (Function.comp (LightCondSet.topCatAdjunctionCounitEquiv X).i …
    -/
    change Filter.Tendsto (fun n : ℕ ↦ g n) _ _
    /-
      case hf
      X✝ Y : LightCondSet
      f✝ : Quiver.Hom X✝ Y
      X : TopCat
      inst✝ : SequentialSpace ↑X
      f : Nat → ↑X
      p : ↑X
      h : Filter.Tendsto f Filter.atTop (nhds p)
      g : OnePoint Nat → ↑X.toLightCondSet.toTopCat := Function.comp (LightCondSet.t …
      ⊢ Filter.Tendsto (fun n => g ↑n) Filter.atTop (nhds ((LightCondSet.topCatAdjun …
    -/
    erw [← OnePoint.continuous_iff_from_nat]
    /-
      case hf
      X✝ Y : LightCondSet
      f✝ : Quiver.Hom X✝ Y
      X : TopCat
      inst✝ : SequentialSpace ↑X
      f : Nat → ↑X
      p : ↑X
      h : Filter.Tendsto f Filter.atTop (nhds p)
      g : OnePoint Nat → ↑X.toLightCondSet.toTopCat := Function.comp (LightCondSet.t …
      ⊢ Continuous g
    -/
    let x : X.toLightCondSet.val.obj ⟨(ℕ∪{∞})⟩ := OnePoint.continuousMapMkNat f p h
    /-
      case hf
      X✝ Y : LightCondSet
      f✝ : Quiver.Hom X✝ Y
      X : TopCat
      inst✝ : SequentialSpace ↑X
      f : Nat → ↑X
      p : ↑X
      h : Filter.Tendsto f Filter.atTop (nhds p)
      g : OnePoint Nat → ↑X.toLightCondSet.toTopCat := Function.comp (LightCondSet.t …
      x : X.toLightCondSet.val.obj { unop := LightProfinite.NatUnionInfty } := OnePo …
      ⊢ Continuous g
    -/
    exact continuous_coinducingCoprod X.toLightCondSet x
    /-
      🎉 no goals
    -/


/--
The counit of the adjunction `lightCondSetToSequential ⊣ sequentialToLightCondSet`
is an isomorphism.

Note: for now, we only have `ℕ∪{∞}` as a light profinite set at universe level 0, which is why we
can only prove this for `X : Sequential.{0}`.
-/
noncomputable def sequentialAdjunctionCounitIso (X : Sequential.{0}) :
    lightCondSetToSequential.obj (sequentialToLightCondSet.obj X) ≅ X :=
  isoOfHomeo (sequentialAdjunctionHomeo X.toTop)


instance : IsIso sequentialAdjunction.{0}.counit := by
  /-
    X Y : LightCondSet
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.IsIso LightCondSet.sequentialAdjunction.counit
  -/
  rw [NatTrans.isIso_iff_isIso_app]
  /-
    X Y : LightCondSet
    f : Quiver.Hom X Y
    ⊢ ∀ (X : Sequential), CategoryTheory.IsIso (LightCondSet.sequentialAdjunction. …
  -/
  intro X
  /-
    X✝ Y : LightCondSet
    f : Quiver.Hom X✝ Y
    X : Sequential
    ⊢ CategoryTheory.IsIso (LightCondSet.sequentialAdjunction.counit.app X)
  -/
  exact inferInstanceAs (IsIso (sequentialAdjunctionCounitIso X).hom)
  /-
    🎉 no goals
  -/


/--
The functor from topological spaces to light condensed sets restricted to sequential spaces
is fully faithful.

Note: for now, we only have `ℕ∪{∞}` as a light profinite set at universe level 0, which is why we
can only prove this for the functor `Sequential.{0} ⥤ LightCondSet.{0}`.
-/
noncomputable def fullyFaithfulSequentialToLightCondSet :
    sequentialToLightCondSet.{0}.FullyFaithful :=
  sequentialAdjunction.fullyFaithfulROfIsIsoCounit


