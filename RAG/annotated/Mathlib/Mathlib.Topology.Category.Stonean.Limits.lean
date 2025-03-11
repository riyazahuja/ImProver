instance : HasExplicitFiniteCoproducts.{w, u} (fun Y ↦ ExtremallyDisconnected Y) where
  hasProp _ := { hasProp := show ExtremallyDisconnected (Σ (_a : _), _) from inferInstance}


lemma extremallyDisconnected_preimage : ExtremallyDisconnected (i ⁻¹' (Set.range f)) where
  open_closure U hU := by
    have h : IsClopen (i ⁻¹' (Set.range f)) :=
      ⟨IsClosed.preimage i.continuous (isCompact_range f.continuous).isClosed,
        IsOpen.preimage i.continuous hi.isOpen_range⟩
    rw [← (closure U).preimage_image_eq Subtype.coe_injective,
      ← h.1.isClosedEmbedding_subtypeVal.closure_image_eq U]
    exact isOpen_induced (ExtremallyDisconnected.open_closure _
      (h.2.isOpenEmbedding_subtypeVal.isOpenMap U hU))


lemma extremallyDisconnected_pullback : ExtremallyDisconnected {xy : X × Y | f xy.1 = i xy.2} :=
  have := extremallyDisconnected_preimage i hi
  let e := (TopCat.pullbackHomeoPreimage i i.2 f hi.isEmbedding).symm
  let e' : {xy : X × Y | f xy.1 = i xy.2} ≃ₜ {xy : Y × X | i xy.1 = f xy.2} := by
    exact TopCat.homeoOfIso
      ((TopCat.pullbackIsoProdSubtype f i).symm ≪≫ pullbackSymmetry _ _ ≪≫
        (TopCat.pullbackIsoProdSubtype i f))
  extremallyDisconnected_of_homeo (e.trans e'.symm)


instance : HasExplicitPullbacksOfInclusions (fun (Y : TopCat.{u}) ↦ ExtremallyDisconnected Y) := by
  /-
    X Y Z : Stonean
    f : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hi : Topology.IsOpenEmbedding ⇑f
    ⊢ CompHausLike.HasExplicitPullbacksOfInclusions fun Y => ExtremallyDisconnecte …
  -/
  apply CompHausLike.hasPullbacksOfInclusions
  /-
    case hP'
    X Y Z : Stonean
    f : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hi : Topology.IsOpenEmbedding ⇑f
    ⊢ ∀ ⦃X Y B : CompHausLike fun Y => ExtremallyDisconnected ↑Y⦄ (f : Quiver.Hom  …
  -/
  intro _ _ _ _ _ hi
  /-
    case hP'
    X Y Z : Stonean
    f : Quiver.Hom X Z
    i : Quiver.Hom Y Z
    hi✝ : Topology.IsOpenEmbedding ⇑f
    X✝ Y✝ B✝ : CompHausLike fun Y => ExtremallyDisconnected ↑Y
    f✝ : Quiver.Hom X✝ B✝
    g✝ : Quiver.Hom Y✝ B✝
    hi : Topology.IsOpenEmbedding ⇑f✝
    ⊢ CompHausLike.HasExplicitPullback f✝ g✝
  -/
  exact ⟨extremallyDisconnected_pullback _ hi⟩
  /-
    🎉 no goals
  -/


