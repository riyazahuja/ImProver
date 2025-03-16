/-- Auxiliary definition to define the topology on `X(*)` for a condensed set `X`. -/
private def CondensedSet.coinducingCoprod :
    (Σ (i : (S : CompHaus.{u}) × X.val.obj ⟨S⟩), i.fst) → X.val.obj ⟨of PUnit⟩ :=
  fun ⟨⟨_, i⟩, s⟩ ↦ X.val.map ((of PUnit.{u+1}).const s).op i


/-- Let `X` be a condensed set. We define a topology on `X(*)` as the quotient topology of
all the maps from compact Hausdorff `S` spaces to `X(*)`, corresponding to elements of `X(S)`.
In other words, the topology coinduced by the map `CondensedSet.coinducingCoprod` above. -/
local instance : TopologicalSpace (X.val.obj ⟨CompHaus.of PUnit⟩) :=
  TopologicalSpace.coinduced (coinducingCoprod X) inferInstance


/-- The object part of the functor `CondensedSet ⥤ TopCat`  -/
def CondensedSet.toTopCat : TopCat.{u+1} := TopCat.of (X.val.obj ⟨of PUnit⟩)


lemma continuous_coinducingCoprod {S : CompHaus.{u}} (x : X.val.obj ⟨S⟩) :
    Continuous fun a ↦ (X.coinducingCoprod ⟨⟨S, x⟩, a⟩) := by
  suffices ∀ (i : (T : CompHaus.{u}) × X.val.obj ⟨T⟩),
      Continuous (fun (a : i.fst) ↦ X.coinducingCoprod ⟨i, a⟩) from this ⟨_, _⟩
  /-
    X : CondensedSet
    S : CompHaus
    x : X.val.obj { unop := S }
    ⊢ ∀ (i : Sigma fun T => X.val.obj { unop := T }), Continuous fun a => Condense …
  -/
  rw [← continuous_sigma_iff]
  /-
    X : CondensedSet
    S : CompHaus
    x : X.val.obj { unop := S }
    ⊢ Continuous (CondensedSet.coinducingCoprod X)
  -/
  apply continuous_coinduced_rng
  /-
    🎉 no goals
  -/


/-- The map part of the functor `CondensedSet ⥤ TopCat`  -/
@[simps]
def toTopCatMap : X.toTopCat ⟶ Y.toTopCat where
  toFun := f.val.app ⟨of PUnit⟩
  continuous_toFun := by
    /-
      X Y : CondensedSet
      f : Quiver.Hom X Y
      ⊢ Continuous (f.val.app { unop := CompHaus.of PUnit.{u + 1} })
    -/
    rw [continuous_coinduced_dom]
    /-
      X Y : CondensedSet
      f : Quiver.Hom X Y
      ⊢ Continuous (Function.comp (f.val.app { unop := CompHaus.of PUnit.{u + 1} })  …
    -/
    apply continuous_sigma
    /-
      case hf
      X Y : CondensedSet
      f : Quiver.Hom X Y
      ⊢ ∀ (i : Sigma fun S => X.val.obj { unop := S }), Continuous fun a => Function …
    -/
    intro ⟨S, x⟩
    /-
      case hf
      X Y : CondensedSet
      f : Quiver.Hom X Y
      S : CompHaus
      x : X.val.obj { unop := S }
      ⊢ Continuous fun a => Function.comp (f.val.app { unop := CompHaus.of PUnit.{u  …
    -/
    simp only [Function.comp_apply, coinducingCoprod]
    rw [show (fun (a : S) ↦ f.val.app ⟨of PUnit⟩ (X.val.map ((of PUnit.{u+1}).const a).op x)) = _
      from funext fun a ↦ NatTrans.naturality_apply f.val ((of PUnit.{u+1}).const a).op x]
    /-
      case hf
      X Y : CondensedSet
      f : Quiver.Hom X Y
      S : CompHaus
      x : X.val.obj { unop := S }
      ⊢ Continuous fun a => (Y.val.map (CompHausLike.const (CompHaus.of PUnit.{u + 1 …
    -/
    exact continuous_coinducingCoprod Y _
    /-
      🎉 no goals
    -/


/-- The functor `CondensedSet ⥤ TopCat`  -/
@[simps]
def condensedSetToTopCat : CondensedSet.{u} ⥤ TopCat.{u+1} where
  obj X := X.toTopCat
  map f := toTopCatMap f


/-- The counit of the adjunction `condensedSetToTopCat ⊣ topCatToCondensedSet` -/
@[simps]
def topCatAdjunctionCounit (X : TopCat.{u+1}) : X.toCondensedSet.toTopCat ⟶ X where
  toFun x := x.1 PUnit.unit
  continuous_toFun := by
    /-
      X✝ : CondensedSet
      X : TopCat
      ⊢ Continuous fun x => x.toFun PUnit.unit
    -/
    rw [continuous_coinduced_dom]
    /-
      X✝ : CondensedSet
      X : TopCat
      ⊢ Continuous (Function.comp (fun x => x.toFun PUnit.unit) (CondensedSet.coindu …
    -/
    continuity
    /-
      🎉 no goals
    -/


/-- The counit of the adjunction `condensedSetToTopCat ⊣ topCatToCondensedSet` is always bijective,
but not an isomorphism in general (the inverse isn't continuous unless `X` is compactly generated).
-/
def topCatAdjunctionCounitEquiv (X : TopCat.{u+1}) : X.toCondensedSet.toTopCat ≃ X where
  toFun := topCatAdjunctionCounit X
  invFun x := ContinuousMap.const _ x
  left_inv _ := rfl
  right_inv _ := rfl


lemma topCatAdjunctionCounit_bijective (X : TopCat.{u+1}) :
    Function.Bijective (topCatAdjunctionCounit X) :=
  (topCatAdjunctionCounitEquiv X).bijective


/-- The unit of the adjunction `condensedSetToTopCat ⊣ topCatToCondensedSet` -/
@[simps val_app val_app_apply]
def topCatAdjunctionUnit (X : CondensedSet.{u}) : X ⟶ X.toTopCat.toCondensedSet where
  val := {
    app := fun S x ↦ {
      toFun := fun s ↦ X.val.map ((of PUnit.{u+1}).const s).op x
      continuous_toFun := by
        suffices ∀ (i : (T : CompHaus.{u}) × X.val.obj ⟨T⟩),
          Continuous (fun (a : i.fst) ↦ X.coinducingCoprod ⟨i, a⟩) from this ⟨_, _⟩
        /-
          X✝ X : CondensedSet
          S : Opposite CompHaus
          x : X.val.obj S
          ⊢ ∀ (i : Sigma fun T => X.val.obj { unop := T }), Continuous fun a => Condense …
        -/
        rw [← continuous_sigma_iff]
        /-
          X✝ X : CondensedSet
          S : Opposite CompHaus
          x : X.val.obj S
          ⊢ Continuous (CondensedSet.coinducingCoprod X)
        -/
        apply continuous_coinduced_rng }
        /-
          🎉 no goals
        -/
    naturality := fun _ _ _ ↦ by
      /-
        X✝ X : CondensedSet
        x✝² x✝¹ : Opposite CompHaus
        x✝ : Quiver.Hom x✝² x✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.val.map x✝) ((fun S x => { toFun : …
      -/
      ext
      simp only [TopCat.toSheafCompHausLike_val_obj, CompHausLike.compHausLikeToTop_obj,
        Opposite.op_unop, types_comp_apply, TopCat.toSheafCompHausLike_val_map,
        ← FunctorToTypes.map_comp_apply]
      /-
        case h
        X✝ X : CondensedSet
        x✝² x✝¹ : Opposite CompHaus
        x✝ : Quiver.Hom x✝² x✝¹
        a✝ : X.val.obj x✝²
        ⊢ Eq { toFun := fun s => X.val.map (CategoryTheory.CategoryStruct.comp x✝ (Com …
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- The adjunction `condensedSetToTopCat ⊣ topCatToCondensedSet` -/
noncomputable def topCatAdjunction : condensedSetToTopCat.{u} ⊣ topCatToCondensedSet where
  unit := { app := topCatAdjunctionUnit }
  counit := { app := topCatAdjunctionCounit }
  left_triangle_components Y := by
    /-
      X Y : CondensedSet
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (condensedSetToTopCat.map ({ app := C …
    -/
    ext
    /-
      case w
      X Y : CondensedSet
      x✝ : (CategoryTheory.forget TopCat).obj (condensedSetToTopCat.obj ((CategoryTh …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (condensedSetToTopCat.map ({ app :=  …
    -/
    change Y.val.map (𝟙 _) _ = _
    /-
      case w
      X Y : CondensedSet
      x✝ : (CategoryTheory.forget TopCat).obj (condensedSetToTopCat.obj ((CategoryTh …
      ⊢ Eq (Y.val.map (CategoryTheory.CategoryStruct.id { unop := Opposite.unop { un …
    -/
    simp
    /-
      🎉 no goals
    -/


instance (X : TopCat) : Epi (topCatAdjunction.counit.app X) := by
  /-
    X✝ : CondensedSet
    X : TopCat
    ⊢ CategoryTheory.Epi (CondensedSet.topCatAdjunction.counit.app X)
  -/
  rw [TopCat.epi_iff_surjective]
  /-
    X✝ : CondensedSet
    X : TopCat
    ⊢ Function.Surjective ⇑(CondensedSet.topCatAdjunction.counit.app X)
  -/
  exact (topCatAdjunctionCounit_bijective _).2
  /-
    🎉 no goals
  -/


instance : topCatToCondensedSet.Faithful := topCatAdjunction.faithful_R_of_epi_counit_app


instance (X : CondensedSet.{u}) : UCompactlyGeneratedSpace.{u, u+1} X.toTopCat := by
  /-
    X✝ X : CondensedSet
    ⊢ UCompactlyGeneratedSpace ↑X.toTopCat
  -/
  apply uCompactlyGeneratedSpace_of_continuous_maps
  /-
    case h
    X✝ X : CondensedSet
    ⊢ ∀ {Y : Type (u + 1)} [tY : TopologicalSpace Y] (f : ↑X.toTopCat → Y), (∀ (S  …
  -/
  intro Y _ f h
  /-
    case h
    X✝ X : CondensedSet
    Y : Type (u + 1)
    tY✝ : TopologicalSpace Y
    f : ↑X.toTopCat → Y
    h : ∀ (S : CompHaus) (g : ContinuousMap ↑S.toTop ↑X.toTopCat), Continuous (Fun …
    ⊢ Continuous f
  -/
  rw [continuous_coinduced_dom, continuous_sigma_iff]
  /-
    case h
    X✝ X : CondensedSet
    Y : Type (u + 1)
    tY✝ : TopologicalSpace Y
    f : ↑X.toTopCat → Y
    h : ∀ (S : CompHaus) (g : ContinuousMap ↑S.toTop ↑X.toTopCat), Continuous (Fun …
    ⊢ ∀ (i : Sigma fun S => X.val.obj { unop := S }), Continuous fun a => Function …
  -/
  exact fun ⟨S, s⟩ ↦ h S ⟨_, continuous_coinducingCoprod X _⟩
  /-
    🎉 no goals
  -/


instance (X : CondensedSet.{u}) : UCompactlyGeneratedSpace.{u, u+1} (condensedSetToTopCat.obj X) :=
  inferInstanceAs (UCompactlyGeneratedSpace.{u, u+1} X.toTopCat)


/-- The functor from condensed sets to topological spaces lands in compactly generated spaces. -/
def condensedSetToCompactlyGenerated : CondensedSet.{u} ⥤ CompactlyGenerated.{u, u+1} where
  obj X := CompactlyGenerated.of (condensedSetToTopCat.obj X)
  map f := toTopCatMap f


/--
The functor from topological spaces to condensed sets restricted to compactly generated spaces.
-/
noncomputable def compactlyGeneratedToCondensedSet :
    CompactlyGenerated.{u, u+1} ⥤ CondensedSet.{u} :=
  compactlyGeneratedToTop ⋙ topCatToCondensedSet



/--
The adjunction `condensedSetToTopCat ⊣ topCatToCondensedSet` restricted to compactly generated
spaces.
-/
noncomputable def compactlyGeneratedAdjunction :
    condensedSetToCompactlyGenerated ⊣ compactlyGeneratedToCondensedSet :=
  topCatAdjunction.restrictFullyFaithful (iC := 𝟭 _) (iD := compactlyGeneratedToTop)
    (Functor.FullyFaithful.id _) fullyFaithfulCompactlyGeneratedToTop
    (Iso.refl _) (Iso.refl _)


/--
The counit of the adjunction `condensedSetToCompactlyGenerated ⊣ compactlyGeneratedToCondensedSet`
is a homeomorphism.
-/
def compactlyGeneratedAdjunctionCounitHomeo (X : TopCat.{u+1}) [UCompactlyGeneratedSpace.{u} X] :
    X.toCondensedSet.toTopCat ≃ₜ X where
  toEquiv := topCatAdjunctionCounitEquiv X
  continuous_toFun := (topCatAdjunctionCounit X).continuous
  continuous_invFun := by
    /-
      X✝ : CondensedSet
      X : TopCat
      inst✝ : UCompactlyGeneratedSpace ↑X
      ⊢ Continuous (CondensedSet.topCatAdjunctionCounitEquiv X).invFun
    -/
    apply continuous_from_uCompactlyGeneratedSpace
    /-
      case h
      X✝ : CondensedSet
      X : TopCat
      inst✝ : UCompactlyGeneratedSpace ↑X
      ⊢ ∀ (S : CompHaus) (g : ContinuousMap ↑S.toTop ↑X), Continuous (Function.comp  …
    -/
    exact fun _ _ ↦ continuous_coinducingCoprod X.toCondensedSet _
    /-
      🎉 no goals
    -/


/--
The counit of the adjunction `condensedSetToCompactlyGenerated ⊣ compactlyGeneratedToCondensedSet`
is an isomorphism.
-/
noncomputable def compactlyGeneratedAdjunctionCounitIso (X : CompactlyGenerated.{u, u+1}) :
    condensedSetToCompactlyGenerated.obj (compactlyGeneratedToCondensedSet.obj X) ≅ X :=
  isoOfHomeo (compactlyGeneratedAdjunctionCounitHomeo X.toTop)


instance : IsIso compactlyGeneratedAdjunction.counit := by
  /-
    X : CondensedSet
    ⊢ CategoryTheory.IsIso CondensedSet.compactlyGeneratedAdjunction.counit
  -/
  rw [NatTrans.isIso_iff_isIso_app]
  /-
    X : CondensedSet
    ⊢ ∀ (X : CompactlyGenerated), CategoryTheory.IsIso (CondensedSet.compactlyGene …
  -/
  intro X
  /-
    X✝ : CondensedSet
    X : CompactlyGenerated
    ⊢ CategoryTheory.IsIso (CondensedSet.compactlyGeneratedAdjunction.counit.app X)
  -/
  exact inferInstanceAs (IsIso (compactlyGeneratedAdjunctionCounitIso X).hom)
  /-
    🎉 no goals
  -/


/--
The functor from topological spaces to condensed sets restricted to compactly generated spaces
is fully faithful.
-/
noncomputable def fullyFaithfulCompactlyGeneratedToCondensedSet :
    compactlyGeneratedToCondensedSet.FullyFaithful :=
  compactlyGeneratedAdjunction.fullyFaithfulROfIsIsoCounit


