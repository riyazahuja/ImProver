theorem IsStableUnderBaseChange.pullback_fst_appTop
    (hP : IsStableUnderBaseChange P) (hP' : RespectsIso P)
    {X Y S : Scheme} [IsAffine X] [IsAffine Y] [IsAffine S] (f : X ⟶ S) (g : Y ⟶ S)
    (H : P g.appTop.hom) : P (pullback.fst f g).appTop.hom := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): change `rw` to `erw`
  erw [← PreservesPullback.iso_inv_fst AffineScheme.forgetToScheme (AffineScheme.ofHom f)
      (AffineScheme.ofHom g)]
  rw [Scheme.comp_appTop, CommRingCat.hom_comp, hP'.cancel_right_isIso,
    AffineScheme.forgetToScheme_map]
  have := congr_arg Quiver.Hom.unop
      (PreservesPullback.iso_hom_fst AffineScheme.Γ.rightOp (AffineScheme.ofHom f)
        (AffineScheme.ofHom g))
  simp only [AffineScheme.Γ, Functor.rightOp_obj, Functor.comp_obj, Functor.op_obj, unop_comp,
    AffineScheme.forgetToScheme_obj, Scheme.Γ_obj, Functor.rightOp_map, Functor.comp_map,
    Functor.op_map, Quiver.Hom.unop_op, AffineScheme.forgetToScheme_map, Scheme.Γ_map] at this
  rw [← this, CommRingCat.hom_comp, hP'.cancel_right_isIso, ← pushoutIsoUnopPullback_inl_hom,
    CommRingCat.hom_comp, hP'.cancel_right_isIso]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => P
    hP' : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    X Y S : AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsAffine X
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    inst✝ : AlgebraicGeometry.IsAffine S
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H : P (AlgebraicGeometry.Scheme.Hom.appTop g).hom
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback. …
    ⊢ P (CategoryTheory.Limits.pushout.inl (AlgebraicGeometry.Scheme.Hom.appTop (A …
  -/
  exact hP.pushout_inl _ hP' _ _ H
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-23")]
alias IsStableUnderBaseChange.pullback_fst_app_top :=
IsStableUnderBaseChange.pullback_fst_appTop


/-- For `P` a property of ring homomorphisms, `sourceAffineLocally P` holds for `f : X ⟶ Y`
whenever `P` holds for the restriction of `f` on every affine open subset of `X`. -/
def sourceAffineLocally : AffineTargetMorphismProperty := fun X _ f _ =>
  ∀ U : X.affineOpens, P (f.appLE ⊤ U le_top).hom


/-- For `P` a property of ring homomorphisms, `affineLocally P` holds for `f : X ⟶ Y` if for each
affine open `U = Spec A ⊆ Y` and `V = Spec B ⊆ f ⁻¹' U`, the ring hom `A ⟶ B` satisfies `P`.
Also see `affineLocally_iff_affineOpens_le`. -/
abbrev affineLocally : MorphismProperty Scheme.{u} :=
  targetAffineLocally (sourceAffineLocally P)


theorem sourceAffineLocally_respectsIso (h₁ : RingHom.RespectsIso P) :
    (sourceAffineLocally P).toProperty.RespectsIso := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    ⊢ (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing S] = …
  -/
  apply AffineTargetMorphismProperty.respectsIso_mk
    /-
      case h₁
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Quive …
    -/
  · introv H U
    have : IsIso (e.hom.appLE (e.hom ''ᵁ U) U.1 (e.hom.preimage_image_eq _).ge) :=
      inferInstanceAs (IsIso (e.hom.app _ ≫
        X.presheaf.map (eqToHom (e.hom.preimage_image_eq _).symm).op))
    rw [← Scheme.appLE_comp_appLE _ _ ⊤ (e.hom ''ᵁ U) U.1 le_top (e.hom.preimage_image_eq _).ge,
      CommRingCat.hom_comp, h₁.cancel_right_isIso]
    /-
      case h₁
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso X Y
      f : Quiver.Hom Y Z
      inst✝ : AlgebraicGeometry.IsAffine Z
      H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
      U : ↑X.affineOpens
      this : CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.appLE e.hom ((Algebr …
      ⊢ P (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ((AlgebraicGeometry.Scheme.H …
    -/
    exact H ⟨_, U.prop.image_of_isOpenImmersion e.hom⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Quive …
    -/
  · introv H U
    /-
      case h₂
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsAffine Y
      H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
      U : ↑X.affineOpens
      ⊢ P (AlgebraicGeometry.Scheme.Hom.appLE (CategoryTheory.CategoryStruct.comp f  …
    -/
    rw [Scheme.comp_appLE, CommRingCat.hom_comp, h₁.cancel_left_isIso]
    /-
      case h₂
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsAffine Y
      H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
      U : ↑X.affineOpens
      ⊢ P (AlgebraicGeometry.Scheme.Hom.appLE f ((TopologicalSpace.Opens.map e.hom.b …
    -/
    exact H U
    /-
      🎉 no goals
    -/


theorem affineLocally_respectsIso (h : RingHom.RespectsIso P) : (affineLocally P).RespectsIso :=
  letI := sourceAffineLocally_respectsIso P h
  inferInstance


open Scheme in
theorem sourceAffineLocally_morphismRestrict {X Y : Scheme.{u}} (f : X ⟶ Y)
    (U : Y.Opens) (hU : IsAffineOpen U) :
    @sourceAffineLocally P _ _ (f ∣_ U) hU ↔
      ∀ (V : X.affineOpens) (e : V.1 ≤ f ⁻¹ᵁ U), P (f.appLE U V e).hom := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Iff (AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing …
  -/
  dsimp only [sourceAffineLocally]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Iff (∀ (U_1 : ↑(↑((TopologicalSpace.Opens.map f.base).obj U)).affineOpens),  …
  -/
  simp only [morphismRestrict_appLE]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Iff (∀ (U_1 : ↑(↑((TopologicalSpace.Opens.map f.base).obj U)).affineOpens),  …
  -/
  rw [(affineOpensRestrict (f ⁻¹ᵁ U)).forall_congr_left, Subtype.forall]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ Iff (∀ (a : ↑X.affineOpens) (b : LE.le (↑a) ((TopologicalSpace.Opens.map f.b …
  -/
  refine forall₂_congr fun V h ↦ ?_
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : ↑X.affineOpens
    h : LE.le (↑V) ((TopologicalSpace.Opens.map f.base).obj U)
    ⊢ Iff (P (AlgebraicGeometry.Scheme.Hom.appLE f ((AlgebraicGeometry.Scheme.Hom. …
  -/
  have := (affineOpensRestrict (f ⁻¹ᵁ U)).apply_symm_apply ⟨V, h⟩
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    U : Y.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    V : ↑X.affineOpens
    h : LE.le (↑V) ((TopologicalSpace.Opens.map f.base).obj U)
    this : Eq ((AlgebraicGeometry.affineOpensRestrict ((TopologicalSpace.Opens.map …
    ⊢ Iff (P (AlgebraicGeometry.Scheme.Hom.appLE f ((AlgebraicGeometry.Scheme.Hom. …
  -/
  exact f.appLE_congr _ (Opens.ι_image_top _) congr($(this).1.1) (fun f => P f.hom)
  /-
    🎉 no goals
  -/


theorem affineLocally_iff_affineOpens_le {X Y : Scheme.{u}} (f : X ⟶ Y) :
    affineLocally.{u} P f ↔
      ∀ (U : Y.affineOpens) (V : X.affineOpens) (e : V.1 ≤ f ⁻¹ᵁ U.1), P (f.appLE U V e).hom :=
  forall_congr' fun U ↦ sourceAffineLocally_morphismRestrict P f U U.2


theorem sourceAffineLocally_isLocal (h₁ : RingHom.RespectsIso P)
    (h₂ : RingHom.LocalizationAwayPreserves P) (h₃ : RingHom.OfLocalizationSpan P) :
    (sourceAffineLocally P).IsLocal := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
    ⊢ (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing S] = …
  -/
  constructor
    /-
      case respectsIso
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      ⊢ (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing S] = …
    -/
  · exact sourceAffineLocally_respectsIso P h₁
    /-
      🎉 no goals
    -/
    /-
      case to_basicOpen
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y] (f  …
    -/
  · intro X Y _ f r H
    /-
      case to_basicOpen
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      r : ↑(Y.presheaf.obj { unop := Top.top })
      H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
      ⊢ AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] = …
    -/
    rw [sourceAffineLocally_morphismRestrict]
    /-
      case to_basicOpen
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      r : ↑(Y.presheaf.obj { unop := Top.top })
      H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
      ⊢ ∀ (V : ↑X.affineOpens) (e : LE.le (↑V) ((TopologicalSpace.Opens.map f.base). …
    -/
    intro U hU
    have : X.basicOpen (f.appLE ⊤ U (by simp) r) = U := by
      simp only [Scheme.Hom.appLE, Opens.map_top, CommRingCat.comp_apply, RingHom.coe_comp,
        Function.comp_apply]
      rw [Scheme.basicOpen_res]
      simpa using hU
    rw [← f.appLE_congr _ rfl this (fun f => P f.hom),
      IsAffineOpen.appLE_eq_away_map f (isAffineOpen_top Y) U.2 _ r]
    /-
      case to_basicOpen
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      r : ↑(Y.presheaf.obj { unop := Top.top })
      H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
      U : ↑X.affineOpens
      hU : LE.le (↑U) ((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen r))
      this : Eq (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑U ⋯).ho …
      ⊢ P (CommRingCat.ofHom (IsLocalization.Away.map (↑(Y.presheaf.toPrefunctor.1 { …
    -/
    simp only
    /-
      case to_basicOpen
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      r : ↑(Y.presheaf.obj { unop := Top.top })
      H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
      U : ↑X.affineOpens
      hU : LE.le (↑U) ((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen r))
      this : Eq (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑U ⋯).ho …
      ⊢ P (IsLocalization.Away.map (↑(Y.presheaf.toPrefunctor.1 { unop := Y.basicOpe …
    -/
    apply (config := { allowSynthFailures := true }) h₂
    /-
      case to_basicOpen.a
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      r : ↑(Y.presheaf.obj { unop := Top.top })
      H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
      U : ↑X.affineOpens
      hU : LE.le (↑U) ((TopologicalSpace.Opens.map f.base).obj (Y.basicOpen r))
      this : Eq (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑U ⋯).ho …
      ⊢ P (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑U ⋯).hom
    -/
    exact H U
    /-
      🎉 no goals
    -/
    /-
      case of_basicOpenCover
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y] (f  …
    -/
  · introv hs hs' U
    /-
      case of_basicOpenCover
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hs' : ∀ (r : Subtype fun x => Membership.mem s x), AlgebraicGeometry.sourceAff …
      U : ↑X.affineOpens
      ⊢ P (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑U ⋯).hom
    -/
    apply h₃ _ _ hs
    /-
      case of_basicOpenCover
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hs' : ∀ (r : Subtype fun x => Membership.mem s x), AlgebraicGeometry.sourceAff …
      U : ↑X.affineOpens
      ⊢ ∀ (r : ↑↑s), (fun {R S} [CommRing R] [CommRing S] => P) (Localization.awayMa …
    -/
    intro r
    /-
      case of_basicOpenCover
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      h₂ : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      h₃ : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => P
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hs' : ∀ (r : Subtype fun x => Membership.mem s x), AlgebraicGeometry.sourceAff …
      U : ↑X.affineOpens
      r : ↑↑s
      ⊢ P (Localization.awayMap (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑U ⋯). …
    -/
    simp_rw [sourceAffineLocally_morphismRestrict] at hs'
    have := hs' r ⟨X.basicOpen (f.appLE ⊤ U le_top r.1), U.2.basicOpen (f.appLE ⊤ U le_top r.1)⟩
      (by simp [Scheme.Hom.appLE])
    rwa [IsAffineOpen.appLE_eq_away_map f (isAffineOpen_top Y) U.2,
      ← h₁.is_localization_away_iff] at this


lemma affineLocally_le {Q : ∀ {R S : Type u} [CommRing R] [CommRing S], (R →+* S) → Prop}
    (hPQ : ∀ {R S : Type u} [CommRing R] [CommRing S] {f : R →+* S}, P f → Q f) :
    affineLocally P ≤ affineLocally Q :=
  fun _ _ _ hf U V ↦ hPQ (hf U V)


/-- If `P` holds for `f` over affine opens `U₂` of `Y` and `V₂` of `X` and `U₁` (resp. `V₁`) are
open affine neighborhoods of `x` (resp. `f.base x`), then `P` also holds for `f`
over some basic open of `U₁` (resp. `V₁`). -/
lemma exists_basicOpen_le_appLE_of_appLE_of_isAffine
    (hPa : StableUnderCompositionWithLocalizationAwayTarget P) (hPl : LocalizationAwayPreserves P)
    (x : X) (U₁ : Y.affineOpens) (U₂ : Y.affineOpens) (V₁ : X.affineOpens) (V₂ : X.affineOpens)
    (hx₁ : x ∈ V₁.1) (hx₂ : x ∈ V₂.1) (e₂ : V₂.1 ≤ f ⁻¹ᵁ U₂.1) (h₂ : P (f.appLE U₂ V₂ e₂).hom)
    (hfx₁ : f.base x ∈ U₁.1) :
    ∃ (r : Γ(Y, U₁)) (s : Γ(X, V₁)) (_ : x ∈ X.basicOpen s)
      (e : X.basicOpen s ≤ f ⁻¹ᵁ Y.basicOpen r),
        P (f.appLE (Y.basicOpen r) (X.basicOpen s) e).hom := by
  obtain ⟨r, r', hBrr', hBfx⟩ := exists_basicOpen_le_affine_inter U₁.2 U₂.2 (f.base x)
    ⟨hfx₁, e₂ hx₂⟩
  /-
    case intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    x : ↑↑X.toPresheafedSpace
    U₁ U₂ : ↑Y.affineOpens
    V₁ V₂ : ↑X.affineOpens
    hx₁ : Membership.mem (↑V₁) x
    hx₂ : Membership.mem (↑V₂) x
    e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
    h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
    hfx₁ : Membership.mem (↑U₁) (f.base x)
    r : ↑(Y.presheaf.obj { unop := ↑U₁ })
    r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
    hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
    hBfx : Membership.mem (Y.basicOpen r) (f.base x)
    ⊢ Exists fun r => Exists fun s => Exists fun x => Exists fun e => P (Algebraic …
  -/
  have ha : IsAffineOpen (X.basicOpen (f.appLE U₂ V₂ e₂ r')) := V₂.2.basicOpen _
  have hxa : x ∈ X.basicOpen (f.appLE U₂ V₂ e₂ r') := by
    simpa [Scheme.Hom.appLE, ← Scheme.preimage_basicOpen] using And.intro hx₂ (hBrr' ▸ hBfx)
  /-
    case intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    x : ↑↑X.toPresheafedSpace
    U₁ U₂ : ↑Y.affineOpens
    V₁ V₂ : ↑X.affineOpens
    hx₁ : Membership.mem (↑V₁) x
    hx₂ : Membership.mem (↑V₂) x
    e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
    h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
    hfx₁ : Membership.mem (↑U₁) (f.base x)
    r : ↑(Y.presheaf.obj { unop := ↑U₁ })
    r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
    hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
    hBfx : Membership.mem (Y.basicOpen r) (f.base x)
    ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
    hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
    ⊢ Exists fun r => Exists fun s => Exists fun x => Exists fun e => P (Algebraic …
  -/
  obtain ⟨s, s', hBss', hBx⟩ := exists_basicOpen_le_affine_inter V₁.2 ha x ⟨hx₁, hxa⟩
  /-
    case intro.intro.intro.intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    x : ↑↑X.toPresheafedSpace
    U₁ U₂ : ↑Y.affineOpens
    V₁ V₂ : ↑X.affineOpens
    hx₁ : Membership.mem (↑V₁) x
    hx₂ : Membership.mem (↑V₂) x
    e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
    h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
    hfx₁ : Membership.mem (↑U₁) (f.base x)
    r : ↑(Y.presheaf.obj { unop := ↑U₁ })
    r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
    hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
    hBfx : Membership.mem (Y.basicOpen r) (f.base x)
    ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
    hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
    s : ↑(X.presheaf.obj { unop := ↑V₁ })
    s' : ↑(X.presheaf.obj { unop := X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app …
    hBss' : Eq (X.basicOpen s) (X.basicOpen s')
    hBx : Membership.mem (X.basicOpen s) x
    ⊢ Exists fun r => Exists fun s => Exists fun x => Exists fun e => P (Algebraic …
  -/
  haveI := V₂.2.isLocalization_basicOpen (f.appLE U₂ V₂ e₂ r')
  /-
    case intro.intro.intro.intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    x : ↑↑X.toPresheafedSpace
    U₁ U₂ : ↑Y.affineOpens
    V₁ V₂ : ↑X.affineOpens
    hx₁ : Membership.mem (↑V₁) x
    hx₂ : Membership.mem (↑V₂) x
    e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
    h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
    hfx₁ : Membership.mem (↑U₁) (f.base x)
    r : ↑(Y.presheaf.obj { unop := ↑U₁ })
    r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
    hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
    hBfx : Membership.mem (Y.basicOpen r) (f.base x)
    ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
    hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
    s : ↑(X.presheaf.obj { unop := ↑V₁ })
    s' : ↑(X.presheaf.obj { unop := X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app …
    hBss' : Eq (X.basicOpen s) (X.basicOpen s')
    hBx : Membership.mem (X.basicOpen s) x
    this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂)  …
    ⊢ Exists fun r => Exists fun s => Exists fun x => Exists fun e => P (Algebraic …
  -/
  haveI := U₂.2.isLocalization_basicOpen r'
  /-
    case intro.intro.intro.intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    x : ↑↑X.toPresheafedSpace
    U₁ U₂ : ↑Y.affineOpens
    V₁ V₂ : ↑X.affineOpens
    hx₁ : Membership.mem (↑V₁) x
    hx₂ : Membership.mem (↑V₂) x
    e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
    h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
    hfx₁ : Membership.mem (↑U₁) (f.base x)
    r : ↑(Y.presheaf.obj { unop := ↑U₁ })
    r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
    hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
    hBfx : Membership.mem (Y.basicOpen r) (f.base x)
    ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
    hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
    s : ↑(X.presheaf.obj { unop := ↑V₁ })
    s' : ↑(X.presheaf.obj { unop := X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app …
    hBss' : Eq (X.basicOpen s) (X.basicOpen s')
    hBx : Membership.mem (X.basicOpen s) x
    this✝ : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) …
    this : IsLocalization.Away r' ↑(Y.presheaf.obj { unop := Y.basicOpen r' })
    ⊢ Exists fun r => Exists fun s => Exists fun x => Exists fun e => P (Algebraic …
  -/
  haveI := ha.isLocalization_basicOpen s'
  have ers : X.basicOpen s ≤ f ⁻¹ᵁ Y.basicOpen r := by
    rw [hBss', hBrr']
    apply le_trans (X.basicOpen_le _)
    simp [Scheme.Hom.appLE]
  have heq : f.appLE (Y.basicOpen r') (X.basicOpen s') (hBrr' ▸ hBss' ▸ ers) =
      f.appLE (Y.basicOpen r') (X.basicOpen (f.appLE U₂ V₂ e₂ r')) (by simp [Scheme.Hom.appLE]) ≫
        CommRingCat.ofHom (algebraMap _ _) := by
    simp only [Scheme.Hom.appLE, homOfLE_leOfHom, CommRingCat.comp_apply, Category.assoc]
    congr
    apply X.presheaf.map_comp
  /-
    case intro.intro.intro.intro.intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    x : ↑↑X.toPresheafedSpace
    U₁ U₂ : ↑Y.affineOpens
    V₁ V₂ : ↑X.affineOpens
    hx₁ : Membership.mem (↑V₁) x
    hx₂ : Membership.mem (↑V₂) x
    e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
    h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
    hfx₁ : Membership.mem (↑U₁) (f.base x)
    r : ↑(Y.presheaf.obj { unop := ↑U₁ })
    r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
    hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
    hBfx : Membership.mem (Y.basicOpen r) (f.base x)
    ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
    hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
    s : ↑(X.presheaf.obj { unop := ↑V₁ })
    s' : ↑(X.presheaf.obj { unop := X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app …
    hBss' : Eq (X.basicOpen s) (X.basicOpen s')
    hBx : Membership.mem (X.basicOpen s) x
    this✝¹ : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂ …
    this✝ : IsLocalization.Away r' ↑(Y.presheaf.obj { unop := Y.basicOpen r' })
    this : IsLocalization.Away s' ↑(X.presheaf.obj { unop := X.basicOpen s' })
    ers : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map f.base).obj (Y.basicO …
    heq : Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r') (X.basicOpen s …
    ⊢ Exists fun r => Exists fun s => Exists fun x => Exists fun e => P (Algebraic …
  -/
  refine ⟨r, s, hBx, ers, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      x : ↑↑X.toPresheafedSpace
      U₁ U₂ : ↑Y.affineOpens
      V₁ V₂ : ↑X.affineOpens
      hx₁ : Membership.mem (↑V₁) x
      hx₂ : Membership.mem (↑V₂) x
      e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
      h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
      hfx₁ : Membership.mem (↑U₁) (f.base x)
      r : ↑(Y.presheaf.obj { unop := ↑U₁ })
      r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
      hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
      hBfx : Membership.mem (Y.basicOpen r) (f.base x)
      ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
      hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
      s : ↑(X.presheaf.obj { unop := ↑V₁ })
      s' : ↑(X.presheaf.obj { unop := X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app …
      hBss' : Eq (X.basicOpen s) (X.basicOpen s')
      hBx : Membership.mem (X.basicOpen s) x
      this✝¹ : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂ …
      this✝ : IsLocalization.Away r' ↑(Y.presheaf.obj { unop := Y.basicOpen r' })
      this : IsLocalization.Away s' ↑(X.presheaf.obj { unop := X.basicOpen s' })
      ers : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map f.base).obj (Y.basicO …
      heq : Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r') (X.basicOpen s …
      ⊢ P (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r) (X.basicOpen s) ers) …
    -/
  · rw [f.appLE_congr _ hBrr' hBss' (fun f => P f.hom), heq]
    /-
      case intro.intro.intro.intro.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      x : ↑↑X.toPresheafedSpace
      U₁ U₂ : ↑Y.affineOpens
      V₁ V₂ : ↑X.affineOpens
      hx₁ : Membership.mem (↑V₁) x
      hx₂ : Membership.mem (↑V₂) x
      e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
      h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
      hfx₁ : Membership.mem (↑U₁) (f.base x)
      r : ↑(Y.presheaf.obj { unop := ↑U₁ })
      r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
      hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
      hBfx : Membership.mem (Y.basicOpen r) (f.base x)
      ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
      hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
      s : ↑(X.presheaf.obj { unop := ↑V₁ })
      s' : ↑(X.presheaf.obj { unop := X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app …
      hBss' : Eq (X.basicOpen s) (X.basicOpen s')
      hBx : Membership.mem (X.basicOpen s) x
      this✝¹ : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂ …
      this✝ : IsLocalization.Away r' ↑(Y.presheaf.obj { unop := Y.basicOpen r' })
      this : IsLocalization.Away s' ↑(X.presheaf.obj { unop := X.basicOpen s' })
      ers : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map f.base).obj (Y.basicO …
      heq : Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r') (X.basicOpen s …
      ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appLE f  …
    -/
    apply hPa _ s' _
    /-
      case intro.intro.intro.intro.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      x : ↑↑X.toPresheafedSpace
      U₁ U₂ : ↑Y.affineOpens
      V₁ V₂ : ↑X.affineOpens
      hx₁ : Membership.mem (↑V₁) x
      hx₂ : Membership.mem (↑V₂) x
      e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
      h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
      hfx₁ : Membership.mem (↑U₁) (f.base x)
      r : ↑(Y.presheaf.obj { unop := ↑U₁ })
      r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
      hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
      hBfx : Membership.mem (Y.basicOpen r) (f.base x)
      ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
      hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
      s : ↑(X.presheaf.obj { unop := ↑V₁ })
      s' : ↑(X.presheaf.obj { unop := X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app …
      hBss' : Eq (X.basicOpen s) (X.basicOpen s')
      hBx : Membership.mem (X.basicOpen s) x
      this✝¹ : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂ …
      this✝ : IsLocalization.Away r' ↑(Y.presheaf.obj { unop := Y.basicOpen r' })
      this : IsLocalization.Away s' ↑(X.presheaf.obj { unop := X.basicOpen s' })
      ers : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map f.base).obj (Y.basicO …
      heq : Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r') (X.basicOpen s …
      ⊢ P (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r') (X.basicOpen ((Alge …
    -/
    rw [U₂.2.appLE_eq_away_map f V₂.2]
    /-
      case intro.intro.intro.intro.intro.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
      hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
      x : ↑↑X.toPresheafedSpace
      U₁ U₂ : ↑Y.affineOpens
      V₁ V₂ : ↑X.affineOpens
      hx₁ : Membership.mem (↑V₁) x
      hx₂ : Membership.mem (↑V₂) x
      e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
      h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
      hfx₁ : Membership.mem (↑U₁) (f.base x)
      r : ↑(Y.presheaf.obj { unop := ↑U₁ })
      r' : ↑(Y.presheaf.obj { unop := ↑U₂ })
      hBrr' : Eq (Y.basicOpen r) (Y.basicOpen r')
      hBfx : Membership.mem (Y.basicOpen r) (f.base x)
      ha : AlgebraicGeometry.IsAffineOpen (X.basicOpen ((AlgebraicGeometry.Scheme.Ho …
      hxa : Membership.mem (X.basicOpen ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) …
      s : ↑(X.presheaf.obj { unop := ↑V₁ })
      s' : ↑(X.presheaf.obj { unop := X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app …
      hBss' : Eq (X.basicOpen s) (X.basicOpen s')
      hBx : Membership.mem (X.basicOpen s) x
      this✝¹ : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂ …
      this✝ : IsLocalization.Away r' ↑(Y.presheaf.obj { unop := Y.basicOpen r' })
      this : IsLocalization.Away s' ↑(X.presheaf.obj { unop := X.basicOpen s' })
      ers : LE.le (X.basicOpen s) ((TopologicalSpace.Opens.map f.base).obj (Y.basicO …
      heq : Eq (AlgebraicGeometry.Scheme.Hom.appLE f (Y.basicOpen r') (X.basicOpen s …
      ⊢ P (CommRingCat.ofHom (IsLocalization.Away.map (↑(Y.presheaf.toPrefunctor.1 { …
    -/
    exact hPl _ _ _ _ h₂
    /-
      🎉 no goals
    -/


/-- If `P` holds for `f` over affine opens `U₂` of `Y` and `V₂` of `X` and `U₁` (resp. `V₁`) are
open neighborhoods of `x` (resp. `f.base x`), then `P` also holds for `f` over some affine open
`U'` of `Y` (resp. `V'` of `X`) that is contained in `U₁` (resp. `V₁`). -/
lemma exists_affineOpens_le_appLE_of_appLE
    (hPa : StableUnderCompositionWithLocalizationAwayTarget P) (hPl : LocalizationAwayPreserves P)
    (x : X) (U₁ : Y.Opens) (U₂ : Y.affineOpens) (V₁ : X.Opens) (V₂ : X.affineOpens)
    (hx₁ : x ∈ V₁) (hx₂ : x ∈ V₂.1) (e₂ : V₂.1 ≤ f ⁻¹ᵁ U₂.1) (h₂ : P (f.appLE U₂ V₂ e₂).hom)
    (hfx₁ : f.base x ∈ U₁.1) :
    ∃ (U' : Y.affineOpens) (V' : X.affineOpens) (_ : U'.1 ≤ U₁) (_ : V'.1 ≤ V₁) (_ : x ∈ V'.1)
      (e : V'.1 ≤ f⁻¹ᵁ U'.1), P (f.appLE U' V' e).hom := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    x : ↑↑X.toPresheafedSpace
    U₁ : Y.Opens
    U₂ : ↑Y.affineOpens
    V₁ : X.Opens
    V₂ : ↑X.affineOpens
    hx₁ : Membership.mem V₁ x
    hx₂ : Membership.mem (↑V₂) x
    e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
    h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
    hfx₁ : Membership.mem U₁.carrier (f.base x)
    ⊢ Exists fun U' => Exists fun V' => Exists fun x_1 => Exists fun x_2 => Exists …
  -/
  obtain ⟨r, hBr, hBfx⟩ := U₂.2.exists_basicOpen_le ⟨f.base x, hfx₁⟩ (e₂ hx₂)
  /-
    case intro.intro
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hPa : RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [Comm …
    hPl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => P
    x : ↑↑X.toPresheafedSpace
    U₁ : Y.Opens
    U₂ : ↑Y.affineOpens
    V₁ : X.Opens
    V₂ : ↑X.affineOpens
    hx₁ : Membership.mem V₁ x
    hx₂ : Membership.mem (↑V₂) x
    e₂ : LE.le (↑V₂) ((TopologicalSpace.Opens.map f.base).obj ↑U₂)
    h₂ : P (AlgebraicGeometry.Scheme.Hom.appLE f (↑U₂) (↑V₂) e₂).hom
    hfx₁ : Membership.mem U₁.carrier (f.base x)
    r : ↑(Y.presheaf.obj { unop := ↑U₂ })
    hBr : LE.le (Y.basicOpen r) U₁
    hBfx : Membership.mem (Y.basicOpen r) ↑⟨f.base x, hfx₁⟩
    ⊢ Exists fun U' => Exists fun V' => Exists fun x_1 => Exists fun x_2 => Exists …
  -/
  obtain ⟨s, hBs, hBx⟩ := V₂.2.exists_basicOpen_le ⟨x, hx₁⟩ hx₂
  obtain ⟨r', s', hBx', e', hf'⟩ := exists_basicOpen_le_appLE_of_appLE_of_isAffine hPa hPl x
    ⟨Y.basicOpen r, U₂.2.basicOpen _⟩ U₂ ⟨X.basicOpen s, V₂.2.basicOpen _⟩ V₂ hBx hx₂ e₂ h₂ hBfx
  exact ⟨⟨Y.basicOpen r', (U₂.2.basicOpen _).basicOpen _⟩,
    ⟨X.basicOpen s', (V₂.2.basicOpen _).basicOpen _⟩, le_trans (Y.basicOpen_le _) hBr,
    le_trans (X.basicOpen_le _) hBs, hBx', e', hf'⟩


/--
`HasRingHomProperty P Q` is a type class asserting that `P` is local at the target and the source,
and for `f : Spec B ⟶ Spec A`, it is equivalent to the ring hom property `Q`.
To make the proofs easier, we state it instead as
1. `Q` is local (See `RingHom.PropertyIsLocal`)
2. `P f` if and only if `Q` holds for every `Γ(Y, U) ⟶ Γ(X, V)` for all affine `U`, `V`.
See `HasRingHomProperty.iff_appLE`.
-/
class HasRingHomProperty (P : MorphismProperty Scheme.{u})
    (Q : outParam (∀ {R S : Type u} [CommRing R] [CommRing S], (R →+* S) → Prop)) : Prop where
  isLocal_ringHomProperty : RingHom.PropertyIsLocal Q
  eq_affineLocally' : P = affineLocally Q


lemma copy {P' : MorphismProperty Scheme.{u}}
    {Q' : ∀ {R S : Type u} [CommRing R] [CommRing S], (R →+* S) → Prop}
    (e : P = P') (e' : ∀ {R S : Type u} [CommRing R] [CommRing S] (f : R →+* S), Q f ↔ Q' f) :
    HasRingHomProperty P' Q' := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    P' : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q' : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R  …
    e : Eq P P'
    e' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (f : RingHom R …
    ⊢ AlgebraicGeometry.HasRingHomProperty P' fun {R S} [CommRing R] [CommRing S]  …
  -/
  subst e
  have heq : @Q = @Q' := by
    ext R S _ _ f
    exact (e' f)
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    Q' : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R  …
    e' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (f : RingHom R …
    heq : Eq Q Q'
    ⊢ AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRing S] = …
  -/
  rw [← heq]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    Q' : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R  …
    e' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (f : RingHom R …
    heq : Eq Q Q'
    ⊢ AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRing S] = …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma eq_affineLocally : P = affineLocally Q := eq_affineLocally'


@[local instance]
lemma HasAffineProperty : HasAffineProperty P (sourceAffineLocally Q) where
  isLocal_affineProperty := sourceAffineLocally_isLocal _
    (isLocal_ringHomProperty P).respectsIso
    (isLocal_ringHomProperty P).localizationAwayPreserves
    (isLocal_ringHomProperty P).ofLocalizationSpan
  eq_targetAffineLocally' := eq_affineLocally P

/- This is only `inferInstance` because of the `@[local instance]` on `HasAffineProperty` above. -/

instance (priority := 900) : IsLocalAtTarget P := inferInstance


theorem appLE (H : P f) (U : Y.affineOpens) (V : X.affineOpens) (e) : Q (f.appLE U V e).hom := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : P f
    U : ↑Y.affineOpens
    V : ↑X.affineOpens
    e : LE.le (↑V) ((TopologicalSpace.Opens.map f.base).obj ↑U)
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appLE f (↑U) (↑V) e).hom
  -/
  rw [eq_affineLocally P, affineLocally_iff_affineOpens_le] at H
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((TopologicalS …
    U : ↑Y.affineOpens
    V : ↑X.affineOpens
    e : LE.le (↑V) ((TopologicalSpace.Opens.map f.base).obj ↑U)
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appLE f (↑U) (↑V) e).hom
  -/
  exact H _ _ _
  /-
    🎉 no goals
  -/


theorem appTop (H : P f) [IsAffine X] [IsAffine Y] : Q f.appTop.hom := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : P f
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
  -/
  rw [Scheme.Hom.appTop, Scheme.Hom.app_eq_appLE]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : P f
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ((TopologicalSpace.Opens.map …
  -/
  exact appLE P f H ⟨_, isAffineOpen_top _⟩ ⟨_, isAffineOpen_top _⟩ _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-23")] alias app_top := appTop


include Q in
theorem comp_of_isOpenImmersion [IsOpenImmersion f] (H : P g) :
    P (f ≫ g) := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    H : P g
    ⊢ P (CategoryTheory.CategoryStruct.comp f g)
  -/
  rw [eq_affineLocally P, affineLocally_iff_affineOpens_le] at H ⊢
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    H : ∀ (U : ↑Z.affineOpens) (V : ↑Y.affineOpens) (e : LE.le (↑V) ((TopologicalS …
    ⊢ ∀ (U : ↑Z.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((TopologicalSpa …
  -/
  intro U V e
  have : IsIso (f.appLE (f ''ᵁ V) V.1 (f.preimage_image_eq _).ge) :=
    inferInstanceAs (IsIso (f.app _ ≫
      X.presheaf.map (eqToHom (f.preimage_image_eq _).symm).op))
  rw [← Scheme.appLE_comp_appLE _ _ _ (f ''ᵁ V) V.1
    (Set.image_subset_iff.mpr e) (f.preimage_image_eq _).ge,
    CommRingCat.hom_comp,
    (isLocal_ringHomProperty P).respectsIso.cancel_right_isIso]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    H : ∀ (U : ↑Z.affineOpens) (V : ↑Y.affineOpens) (e : LE.le (↑V) ((TopologicalS …
    U : ↑Z.affineOpens
    V : ↑X.affineOpens
    e : LE.le (↑V) ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStruct.com …
    this : CategoryTheory.IsIso (AlgebraicGeometry.Scheme.Hom.appLE f ((AlgebraicG …
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appLE g (↑U) ((AlgebraicGeometry.Scheme.Hom. …
  -/
  exact H _ ⟨_, V.2.image_of_isOpenImmersion _⟩ _
  /-
    🎉 no goals
  -/


lemma iff_appLE : P f ↔ ∀ (U : Y.affineOpens) (V : X.affineOpens) (e), Q (f.appLE U V e).hom := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (P f) (∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Top …
  -/
  rw [eq_affineLocally P, affineLocally_iff_affineOpens_le]
  /-
    🎉 no goals
  -/


theorem of_source_openCover [IsAffine Y]
    (𝒰 : X.OpenCover) [∀ i, IsAffine (𝒰.obj i)] (H : ∀ i, Q ((𝒰.map i ≫ f).appTop.hom)) :
    P f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    𝒰 : X.OpenCover
    inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    H : ∀ (i : 𝒰.J), Q (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.Catego …
    ⊢ P f
  -/
  rw [HasAffineProperty.iff_of_isAffine (P := P)]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    𝒰 : X.OpenCover
    inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    H : ∀ (i : 𝒰.J), Q (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.Catego …
    ⊢ AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] = …
  -/
  intro U
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsAffine Y
    𝒰 : X.OpenCover
    inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    H : ∀ (i : 𝒰.J), Q (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.Catego …
    U : ↑X.affineOpens
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑U ⋯).hom
  -/
  let S i : X.affineOpens := ⟨_, isAffineOpen_opensRange (𝒰.map i)⟩
  induction U using of_affine_open_cover S 𝒰.iSup_opensRange with
  | basicOpen U r H =>
    simp_rw [Scheme.affineBasicOpen_coe,
      ← f.appLE_map (U := ⊤) le_top (homOfLE (X.basicOpen_le r)).op]
    have := U.2.isLocalization_basicOpen r
    exact (isLocal_ringHomProperty P).StableUnderCompositionWithLocalizationAwayTarget _ r _ H
  | openCover U s hs H =>
    apply (isLocal_ringHomProperty P).ofLocalizationSpanTarget.ofIsLocalization
      (isLocal_ringHomProperty P).respectsIso _ _ hs
    rintro r
    refine ⟨_, _, _, IsAffineOpen.isLocalization_basicOpen U.2 r, ?_⟩
    rw [RingHom.algebraMap_toAlgebra, ← CommRingCat.hom_comp, Scheme.Hom.appLE_map]
    exact H r
  | hU i =>
    specialize H i
    rw [← (isLocal_ringHomProperty P).respectsIso.cancel_right_isIso _
      ((IsOpenImmersion.isoOfRangeEq (𝒰.map i) (S i).1.ι
      Subtype.range_coe.symm).inv.app _), ← CommRingCat.hom_comp, ← Scheme.comp_appTop,
      IsOpenImmersion.isoOfRangeEq_inv_fac_assoc, Scheme.comp_appTop,
      Scheme.Opens.ι_appTop, Scheme.Hom.appTop, Scheme.Hom.app_eq_appLE, Scheme.Hom.appLE_map] at H
    exact (f.appLE_congr _ rfl (by simp) (fun f => Q f.hom)).mp H


theorem iff_of_source_openCover [IsAffine Y] (𝒰 : X.OpenCover) [∀ i, IsAffine (𝒰.obj i)] :
    P f ↔ ∀ i, Q ((𝒰.map i ≫ f).appTop).hom :=
  ⟨fun H i ↦ appTop P _ (comp_of_isOpenImmersion P (𝒰.map i) f H), of_source_openCover 𝒰⟩


theorem iff_of_isAffine [IsAffine X] [IsAffine Y] :
    P f ↔ Q (f.appTop).hom := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (P f) (Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom)
  -/
  rw [iff_of_source_openCover (P := P) (Scheme.coverOfIsIso.{u} (𝟙 _))]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsAffine X
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (∀ (i : (AlgebraicGeometry.Scheme.coverOfIsIso (CategoryTheory.CategoryS …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Spec_iff {R S : CommRingCat.{u}} {φ : R ⟶ S} :
    P (Spec.map φ) ↔ Q φ.hom := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    R S : CommRingCat
    φ : Quiver.Hom R S
    ⊢ Iff (P (AlgebraicGeometry.Spec.map φ)) (Q φ.hom)
  -/
  have H := (isLocal_ringHomProperty P).respectsIso
  rw [iff_of_isAffine (P := P), ← H.cancel_right_isIso _ (Scheme.ΓSpecIso _).hom,
    ← CommRingCat.hom_comp, Scheme.ΓSpecIso_naturality, CommRingCat.hom_comp, H.cancel_left_isIso]


theorem of_iSup_eq_top [IsAffine Y] {ι : Type*}
    (U : ι → X.affineOpens) (hU : ⨆ i, (U i : Opens X) = ⊤)
    (H : ∀ i, Q (f.appLE ⊤ (U i).1 le_top).hom) :
    P f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    ι : Type u_1
    U : ι → ↑X.affineOpens
    hU : Eq (iSup fun i => ↑(U i)) Top.top
    H : ∀ (i : ι), Q (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑(U i) ⋯).hom
    ⊢ P f
  -/
  have (i) : IsAffine ((X.openCoverOfISupEqTop _ hU).obj i) := (U i).2
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    ι : Type u_1
    U : ι → ↑X.affineOpens
    hU : Eq (iSup fun i => ↑(U i)) Top.top
    H : ∀ (i : ι), Q (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑(U i) ⋯).hom
    this : ∀ (i : (X.openCoverOfISupEqTop (fun i => ↑(U i)) hU).J), AlgebraicGeome …
    ⊢ P f
  -/
  refine of_source_openCover (X.openCoverOfISupEqTop _ hU) fun i ↦ ?_
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    ι : Type u_1
    U : ι → ↑X.affineOpens
    hU : Eq (iSup fun i => ↑(U i)) Top.top
    H : ∀ (i : ι), Q (AlgebraicGeometry.Scheme.Hom.appLE f Top.top ↑(U i) ⋯).hom
    this : ∀ (i : (X.openCoverOfISupEqTop (fun i => ↑(U i)) hU).J), AlgebraicGeome …
    i : (X.openCoverOfISupEqTop (fun i => ↑(U i)) hU).J
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.CategoryStruct.comp ( …
  -/
  simpa [Scheme.Hom.app_eq_appLE] using (f.appLE_congr _ rfl (by simp) (fun f => Q f.hom)).mp (H i)
  /-
    🎉 no goals
  -/


theorem iff_of_iSup_eq_top [IsAffine Y] {ι : Type*}
    (U : ι → X.affineOpens) (hU : ⨆ i, (U i : Opens X) = ⊤) :
    P f ↔ ∀ i, Q (f.appLE ⊤ (U i).1 le_top).hom :=
  ⟨fun H _ ↦ appLE P f H ⟨_, isAffineOpen_top _⟩ _ le_top, of_iSup_eq_top U hU⟩


instance : IsLocalAtSource P := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ AlgebraicGeometry.IsLocalAtSource P
  -/
  apply HasAffineProperty.isLocalAtSource
  /-
    case H
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : AlgebraicGeo …
  -/
  intros X Y f _ 𝒰
  simp_rw [← HasAffineProperty.iff_of_isAffine (P := P),
    iff_of_source_openCover 𝒰.affineRefinement.openCover,
    fun i ↦ iff_of_source_openCover (P := P) (f := 𝒰.map i ≫ f) (𝒰.obj i).affineCover]
  /-
    case H
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    X✝ Y✝ Z : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    g : Quiver.Hom Y✝ Z
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    𝒰 : X.OpenCover
    ⊢ Iff (∀ (i : 𝒰.affineRefinement.openCover.J), Q (AlgebraicGeometry.Scheme.Hom …
  -/
  simp [Scheme.OpenCover.affineRefinement, Sigma.forall]
  /-
    🎉 no goals
  -/


lemma containsIdentities (hP : RingHom.ContainsIdentities Q) : P.ContainsIdentities where
  id_mem X := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => Q
      X : AlgebraicGeometry.Scheme
      ⊢ P (CategoryTheory.CategoryStruct.id X)
    -/
    rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := P) _ (iSup_affineOpens_eq_top _)]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => Q
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (i : ↑X.affineOpens), P (AlgebraicGeometry.morphismRestrict (CategoryTheor …
    -/
    intro U
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => Q
      X : AlgebraicGeometry.Scheme
      U : ↑X.affineOpens
      ⊢ P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.id X) ↑U)
    -/
    have : IsAffine (𝟙 X ⁻¹ᵁ U.1) := U.2
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => Q
      X : AlgebraicGeometry.Scheme
      U : ↑X.affineOpens
      this : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map (CategoryTheor …
      ⊢ P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.id X) ↑U)
    -/
    rw [morphismRestrict_id, iff_of_isAffine (P := P), Scheme.id_appTop]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => Q
      X : AlgebraicGeometry.Scheme
      U : ↑X.affineOpens
      this : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map (CategoryTheor …
      ⊢ Q (CategoryTheory.CategoryStruct.id ((↑((TopologicalSpace.Opens.map (Categor …
    -/
    apply hP
    /-
      🎉 no goals
    -/


variable (P) in
open _root_.PrimeSpectrum in
lemma isLocal_ringHomProperty_of_isLocalAtSource_of_isLocalAtTarget
    [IsLocalAtTarget P] [IsLocalAtSource P] :
    RingHom.PropertyIsLocal fun f ↦ P (Spec.map (CommRingCat.ofHom f)) := by
  have hP : RingHom.RespectsIso (fun f ↦ P (Spec.map (CommRingCat.ofHom f))) :=
    RingHom.toMorphismProperty_respectsIso_iff.mpr
      (inferInstanceAs (P.inverseImage Scheme.Spec).unop.RespectsIso)
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
    inst✝ : AlgebraicGeometry.IsLocalAtSource P
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
    ⊢ RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] f => P (Algebrai …
  -/
  constructor
    /-
      case localizationAwayPreserves
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      ⊢ RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] f => P …
    -/
  · intro R S _ _ f r R' S' _ _ _ _ _ _ H
    /-
      case localizationAwayPreserves
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝⁸ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      R S : Type u
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      r : R
      R' S' : Type u
      inst✝⁵ : CommRing R'
      inst✝⁴ : CommRing S'
      inst✝³ : Algebra R R'
      inst✝² : Algebra S S'
      inst✝¹ : IsLocalization.Away r R'
      inst✝ : IsLocalization.Away (f r) S'
      H : P (AlgebraicGeometry.Spec.map (CommRingCat.ofHom f))
      ⊢ P (AlgebraicGeometry.Spec.map (CommRingCat.ofHom (IsLocalization.Away.map R' …
    -/
    refine (RingHom.RespectsIso.is_localization_away_iff hP ..).mp ?_
    exact (MorphismProperty.arrow_mk_iso_iff P (SpecMapRestrictBasicOpenIso
      (CommRingCat.ofHom f) r)).mp (IsLocalAtTarget.restrict H (basicOpen r))
    /-
      case ofLocalizationSpanTarget
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      ⊢ RingHom.OfLocalizationSpanTarget fun {R S} [CommRing R] [CommRing S] f => P  …
    -/
  · intros R S _ _ f s hs H
    apply IsLocalAtSource.of_openCover (Scheme.affineOpenCoverOfSpanRangeEqTop
      (R := CommRingCat.of S) (ι := s) (fun i : s ↦ (i : S)) (by simpa)).openCover
    /-
      case ofLocalizationSpanTarget
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝³ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set S
      hs : Eq (Ideal.span s) Top.top
      H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] f => P (AlgebraicGeometry …
      ⊢ ∀ (i : (AlgebraicGeometry.Scheme.affineOpenCoverOfSpanRangeEqTop (fun i => ↑ …
    -/
    intro i
    simp only [CommRingCat.coe_of, Set.setOf_mem_eq, id_eq, eq_mpr_eq_cast,
      Scheme.AffineOpenCover.openCover_obj, Scheme.affineOpenCoverOfSpanRangeEqTop_obj_carrier,
      Scheme.AffineOpenCover.openCover_map, Scheme.affineOpenCoverOfSpanRangeEqTop_map,
      ← Spec.map_comp]
    /-
      case ofLocalizationSpanTarget
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝³ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set S
      hs : Eq (Ideal.span s) Top.top
      H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] f => P (AlgebraicGeometry …
      i : (AlgebraicGeometry.Scheme.affineOpenCoverOfSpanRangeEqTop (fun i => ↑i) ⋯) …
      ⊢ P (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.comp (CommRingC …
    -/
    exact H i
    /-
      🎉 no goals
    -/
    /-
      case ofLocalizationSpan
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      ⊢ RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] f => P (Algeb …
    -/
  · intro R S _ _  f s hs H
    apply IsLocalAtTarget.of_iSup_eq_top _ (PrimeSpectrum.iSup_basicOpen_eq_top_iff
      (f := fun i : s ↦ (i : R)).mpr (by simpa))
    /-
      case ofLocalizationSpan
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝³ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      R S : Type u
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set R
      hs : Eq (Ideal.span s) Top.top
      H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] f => P (AlgebraicGeometry …
      ⊢ ∀ (i : ↑s), P (AlgebraicGeometry.morphismRestrict (AlgebraicGeometry.Spec.ma …
    -/
    intro i
    exact (MorphismProperty.arrow_mk_iso_iff P (SpecMapRestrictBasicOpenIso
      (CommRingCat.ofHom f) i.1)).mpr (H i)
    /-
      case StableUnderCompositionWithLocalizationAwayTarget
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      ⊢ RingHom.StableUnderCompositionWithLocalizationAwayTarget fun {R S} [CommRing …
    -/
  · intro R S T _ _ _ _ r _ f hf
    /-
      case StableUnderCompositionWithLocalizationAwayTarget
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁶ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝⁵ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      r : S
      inst✝ : IsLocalization.Away r T
      f : RingHom R S
      hf : P (AlgebraicGeometry.Spec.map (CommRingCat.ofHom f))
      ⊢ P (AlgebraicGeometry.Spec.map (CommRingCat.ofHom ((algebraMap S T).comp f)))
    -/
    have := AlgebraicGeometry.IsOpenImmersion.of_isLocalization (S := T) r
    /-
      case StableUnderCompositionWithLocalizationAwayTarget
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁶ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝⁵ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      r : S
      inst✝ : IsLocalization.Away r T
      f : RingHom R S
      hf : P (AlgebraicGeometry.Spec.map (CommRingCat.ofHom f))
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ P (AlgebraicGeometry.Spec.map (CommRingCat.ofHom ((algebraMap S T).comp f)))
    -/
    show P (Spec.map (CommRingCat.ofHom f ≫ CommRingCat.ofHom (algebraMap _ _)))
    /-
      case StableUnderCompositionWithLocalizationAwayTarget
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁶ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝⁵ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      r : S
      inst✝ : IsLocalization.Away r T
      f : RingHom R S
      hf : P (AlgebraicGeometry.Spec.map (CommRingCat.ofHom f))
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ P (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.comp (CommRingC …
    -/
    rw [Spec.map_comp]
    /-
      case StableUnderCompositionWithLocalizationAwayTarget
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝⁶ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝⁵ : AlgebraicGeometry.IsLocalAtSource P
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => P (Algebraic …
      R S T : Type u
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra S T
      r : S
      inst✝ : IsLocalization.Away r T
      f : RingHom R S
      hf : P (AlgebraicGeometry.Spec.map (CommRingCat.ofHom f))
      this : AlgebraicGeometry.IsOpenImmersion (AlgebraicGeometry.Spec.map (CommRing …
      ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (CommRingC …
    -/
    exact IsLocalAtSource.comp hf ..
    /-
      🎉 no goals
    -/


open _root_.PrimeSpectrum in
variable (P) in
lemma of_isLocalAtSource_of_isLocalAtTarget [IsLocalAtTarget P] [IsLocalAtSource P] :
    HasRingHomProperty P (fun f ↦ P (Spec.map (CommRingCat.ofHom f))) where
  isLocal_ringHomProperty :=
    isLocal_ringHomProperty_of_isLocalAtSource_of_isLocalAtTarget P
  eq_affineLocally' := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      ⊢ Eq P (AlgebraicGeometry.affineLocally fun {R S} [CommRing R] [CommRing S] f  …
    -/
    let Q := affineLocally (fun f ↦ P (Spec.map (CommRingCat.ofHom f)))
    have : HasRingHomProperty Q (fun f ↦ P (Spec.map (CommRingCat.ofHom f))) :=
      ⟨isLocal_ringHomProperty_of_isLocalAtSource_of_isLocalAtTarget P, rfl⟩
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      ⊢ Eq P (AlgebraicGeometry.affineLocally fun {R S} [CommRing R] [CommRing S] f  …
    -/
    show P = Q
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      ⊢ Eq P Q
    -/
    ext X Y f
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (P f) (Q f)
    -/
    wlog hY : ∃ R, Y = Spec R generalizing X Y
    · rw [IsLocalAtTarget.iff_of_openCover (P := P) Y.affineCover,
        IsLocalAtTarget.iff_of_openCover (P := Q) Y.affineCover]
      /-
        case h.inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
        inst✝ : AlgebraicGeometry.IsLocalAtSource P
        Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
        this✝ : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRin …
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        this : ∀ ⦃X Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom X Y), (Exists fun R  …
        hY : Not (Exists fun R => Eq Y (AlgebraicGeometry.Spec R))
        ⊢ Iff (∀ (i : Y.affineCover.1), P (AlgebraicGeometry.Scheme.Cover.pullbackHom  …
      -/
      refine forall_congr' fun _ ↦ this _ ⟨_, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hY : Exists fun R => Eq Y (AlgebraicGeometry.Spec R)
      ⊢ Iff (P f) (Q f)
    -/
    obtain ⟨S, rfl⟩ := hY
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      X : AlgebraicGeometry.Scheme
      S : CommRingCat
      f : Quiver.Hom X (AlgebraicGeometry.Spec S)
      ⊢ Iff (P f) (Q f)
    -/
    wlog hX : ∃ R, X = Spec R generalizing X
    · rw [IsLocalAtSource.iff_of_openCover (P := P) X.affineCover,
        IsLocalAtSource.iff_of_openCover (P := Q) X.affineCover]
      /-
        case intro.inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
        inst✝ : AlgebraicGeometry.IsLocalAtSource P
        Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
        this✝ : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRin …
        X : AlgebraicGeometry.Scheme
        S : CommRingCat
        f : Quiver.Hom X (AlgebraicGeometry.Spec S)
        this : ∀ ⦃X : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom X (AlgebraicGeometry.S …
        hX : Not (Exists fun R => Eq X (AlgebraicGeometry.Spec R))
        ⊢ Iff (∀ (i : X.affineCover.J), P (CategoryTheory.CategoryStruct.comp (X.affin …
      -/
      refine forall_congr' fun _ ↦ this _ ⟨_, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      S : CommRingCat
      X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X (AlgebraicGeometry.Spec S)
      hX : Exists fun R => Eq X (AlgebraicGeometry.Spec R)
      ⊢ Iff (P f) (Q f)
    -/
    obtain ⟨R, rfl⟩ := hX
    /-
      case intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      S R : CommRingCat
      f : Quiver.Hom (AlgebraicGeometry.Spec R) (AlgebraicGeometry.Spec S)
      ⊢ Iff (P f) (Q f)
    -/
    obtain ⟨φ, rfl⟩ : ∃ φ, Spec.map φ = f := ⟨_, Spec.map_preimage _⟩
    /-
      case intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      S R : CommRingCat
      φ : Quiver.Hom S R
      ⊢ Iff (P (AlgebraicGeometry.Spec.map φ)) (Q (AlgebraicGeometry.Spec.map φ))
    -/
    rw [HasRingHomProperty.Spec_iff (P := Q)]
    /-
      case intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : AlgebraicGeometry.IsLocalAtSource P
      Q : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme := AlgebraicGeome …
      this : AlgebraicGeometry.HasRingHomProperty Q fun {R S} [CommRing R] [CommRing …
      S R : CommRingCat
      φ : Quiver.Hom S R
      ⊢ Iff (P (AlgebraicGeometry.Spec.map φ)) (P (AlgebraicGeometry.Spec.map (CommR …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma stalkwise {P} (hP : RingHom.RespectsIso P) :
    HasRingHomProperty (stalkwise P) fun {_ S _ _} φ ↦
      ∀ (p : Ideal S) (_ : p.IsPrime), P (Localization.localRingHom _ p φ rfl) := by
  /-
    P : {R S : Type u_1} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R …
    hP : RingHom.RespectsIso P
    ⊢ AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.stalkwise fun {R S}  …
  -/
  have := stalkwiseIsLocalAtTarget_of_respectsIso hP
  /-
    P : {R S : Type u_1} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R …
    hP : RingHom.RespectsIso P
    this : AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.stalkwise fun {R S …
    ⊢ AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.stalkwise fun {R S}  …
  -/
  have := stalkwise_isLocalAtSource_of_respectsIso hP
  /-
    P : {R S : Type u_1} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R …
    hP : RingHom.RespectsIso P
    this✝ : AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.stalkwise fun {R  …
    this : AlgebraicGeometry.IsLocalAtSource (AlgebraicGeometry.stalkwise fun {R S …
    ⊢ AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.stalkwise fun {R S}  …
  -/
  convert of_isLocalAtSource_of_isLocalAtTarget (P := AlgebraicGeometry.stalkwise P) with R S _ _ φ
  /-
    case h.e'_2.h.h.h.h.h.a
    P : {R S : Type u_1} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R …
    hP : RingHom.RespectsIso P
    this✝ : AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.stalkwise fun {R  …
    this : AlgebraicGeometry.IsLocalAtSource (AlgebraicGeometry.stalkwise fun {R S …
    R S : Type u_1
    x✝¹ : CommRing R
    x✝ : CommRing S
    φ : RingHom R S
    ⊢ Iff (∀ (p : Ideal S) (x : p.IsPrime), P (Localization.localRingHom (Ideal.co …
  -/
  exact (stalkwise_Spec_map_iff hP (CommRingCat.ofHom φ)).symm
  /-
    🎉 no goals
  -/


lemma stableUnderComposition (hP : RingHom.StableUnderComposition Q) :
    P.IsStableUnderComposition where
  comp_mem {X Y Z} f g hf hg := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P f
      hg : P g
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    wlog hZ : IsAffine Z generalizing X Y Z
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        this : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.H …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        ⊢ P (CategoryTheory.CategoryStruct.comp f g)
      -/
    · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := P) _ (iSup_affineOpens_eq_top _)]
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        this : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.H …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        ⊢ ∀ (i : ↑Z.affineOpens), P (AlgebraicGeometry.morphismRestrict (CategoryTheor …
      -/
      intro U
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        this : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.H …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        U : ↑Z.affineOpens
        ⊢ P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.comp f  …
      -/
      rw [morphismRestrict_comp]
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        this : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.H …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        U : ↑Z.affineOpens
        ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict f  …
      -/
      exact this _ _ (IsLocalAtTarget.restrict hf _) (IsLocalAtTarget.restrict hg _) U.2
      /-
        🎉 no goals
      -/
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P f
      hg : P g
      hZ : AlgebraicGeometry.IsAffine Z
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    wlog hY : IsAffine Y generalizing X Y
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        hZ : AlgebraicGeometry.IsAffine Z
        this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.Hom …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        ⊢ P (CategoryTheory.CategoryStruct.comp f g)
      -/
    · rw [IsLocalAtSource.iff_of_openCover (P := P) (Y.affineCover.pullbackCover f)]
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        hZ : AlgebraicGeometry.IsAffine Z
        this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.Hom …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        ⊢ ∀ (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover Y.affineCover f).J), P  …
      -/
      intro i
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        hZ : AlgebraicGeometry.IsAffine Z
        this : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.Hom …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        i : (AlgebraicGeometry.Scheme.Cover.pullbackCover Y.affineCover f).J
        ⊢ P (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.pullb …
      -/
      rw [← Scheme.Cover.pullbackHom_map_assoc]
      exact this _ _ (IsLocalAtTarget.of_isPullback (.of_hasPullback _ _) hf)
        (comp_of_isOpenImmersion _ _ _ hg) inferInstance
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      Z : AlgebraicGeometry.Scheme
      hZ : AlgebraicGeometry.IsAffine Z
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P f
      hg : P g
      hY : AlgebraicGeometry.IsAffine Y
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    wlog hX : IsAffine X generalizing X
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        Z : AlgebraicGeometry.Scheme
        hZ : AlgebraicGeometry.IsAffine Z
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        hY : AlgebraicGeometry.IsAffine Y
        this : ∀ {X : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), P f → AlgebraicG …
        hX : Not (AlgebraicGeometry.IsAffine X)
        ⊢ P (CategoryTheory.CategoryStruct.comp f g)
      -/
    · rw [IsLocalAtSource.iff_of_openCover (P := P) X.affineCover]
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        Z : AlgebraicGeometry.Scheme
        hZ : AlgebraicGeometry.IsAffine Z
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        hY : AlgebraicGeometry.IsAffine Y
        this : ∀ {X : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), P f → AlgebraicG …
        hX : Not (AlgebraicGeometry.IsAffine X)
        ⊢ ∀ (i : X.affineCover.J), P (CategoryTheory.CategoryStruct.comp (X.affineCove …
      -/
      intro i
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        Z : AlgebraicGeometry.Scheme
        hZ : AlgebraicGeometry.IsAffine Z
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        hY : AlgebraicGeometry.IsAffine Y
        this : ∀ {X : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), P f → AlgebraicG …
        hX : Not (AlgebraicGeometry.IsAffine X)
        i : X.affineCover.J
        ⊢ P (CategoryTheory.CategoryStruct.comp (X.affineCover.map i) (CategoryTheory. …
      -/
      rw [← Category.assoc]
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        Z : AlgebraicGeometry.Scheme
        hZ : AlgebraicGeometry.IsAffine Z
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        hY : AlgebraicGeometry.IsAffine Y
        this : ∀ {X : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), P f → AlgebraicG …
        hX : Not (AlgebraicGeometry.IsAffine X)
        i : X.affineCover.J
        ⊢ P (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp (X …
      -/
      exact this _ (comp_of_isOpenImmersion _ _ _ hf) inferInstance
      /-
        🎉 no goals
      -/
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      Z : AlgebraicGeometry.Scheme
      hZ : AlgebraicGeometry.IsAffine Z
      Y : AlgebraicGeometry.Scheme
      g : Quiver.Hom Y Z
      hg : P g
      hY : AlgebraicGeometry.IsAffine Y
      X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : P f
      hX : AlgebraicGeometry.IsAffine X
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    rw [iff_of_isAffine (P := P)] at hf hg ⊢
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      Z : AlgebraicGeometry.Scheme
      hZ : AlgebraicGeometry.IsAffine Z
      Y : AlgebraicGeometry.Scheme
      g : Quiver.Hom Y Z
      hg : Q (AlgebraicGeometry.Scheme.Hom.appTop g).hom
      hY : AlgebraicGeometry.IsAffine Y
      X : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
      hX : AlgebraicGeometry.IsAffine X
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.CategoryStruct.comp f …
    -/
    exact hP _ _ hg hf
    /-
      🎉 no goals
    -/


theorem of_comp
    (H : ∀ {R S T : Type u} [CommRing R] [CommRing S] [CommRing T],
      ∀ (f : R →+* S) (g : S →+* T), Q (g.comp f) → Q g)
    {X Y Z : Scheme.{u}} {f : X ⟶ Y} {g : Y ⟶ Z} (h : P (f ≫ g)) : P f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : P (CategoryTheory.CategoryStruct.comp f g)
    ⊢ P f
  -/
  wlog hZ : IsAffine Z generalizing X Y Z
  · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := P) _
      (g.preimage_iSup_eq_top (iSup_affineOpens_eq_top Z))]
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      this : ∀ {X Y Z : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.H …
      hZ : Not (AlgebraicGeometry.IsAffine Z)
      ⊢ ∀ (i : ↑Z.affineOpens), P (AlgebraicGeometry.morphismRestrict f ((Topologica …
    -/
    intro U
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      this : ∀ {X Y Z : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.H …
      hZ : Not (AlgebraicGeometry.IsAffine Z)
      U : ↑Z.affineOpens
      ⊢ P (AlgebraicGeometry.morphismRestrict f ((TopologicalSpace.Opens.map g.base) …
    -/
    have H := IsLocalAtTarget.restrict h U.1
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H✝ : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      this : ∀ {X Y Z : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.H …
      hZ : Not (AlgebraicGeometry.IsAffine Z)
      U : ↑Z.affineOpens
      H : P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.comp  …
      ⊢ P (AlgebraicGeometry.morphismRestrict f ((TopologicalSpace.Opens.map g.base) …
    -/
    rw [morphismRestrict_comp] at H
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H✝ : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      this : ∀ {X Y Z : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.H …
      hZ : Not (AlgebraicGeometry.IsAffine Z)
      U : ↑Z.affineOpens
      H : P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict  …
      ⊢ P (AlgebraicGeometry.morphismRestrict f ((TopologicalSpace.Opens.map g.base) …
    -/
    exact this H inferInstance
    /-
      🎉 no goals
    -/
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : P (CategoryTheory.CategoryStruct.comp f g)
    hZ : AlgebraicGeometry.IsAffine Z
    ⊢ P f
  -/
  wlog hY : IsAffine Y generalizing X Y
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hZ : AlgebraicGeometry.IsAffine Z
      this : ∀ {X Y : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.Hom …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ P f
    -/
  · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := P) _ (iSup_affineOpens_eq_top Y)]
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hZ : AlgebraicGeometry.IsAffine Z
      this : ∀ {X Y : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.Hom …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ ∀ (i : ↑Y.affineOpens), P (AlgebraicGeometry.morphismRestrict f ↑i)
    -/
    intro U
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hZ : AlgebraicGeometry.IsAffine Z
      this : ∀ {X Y : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.Hom …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      ⊢ P (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    have H := comp_of_isOpenImmersion P (f ⁻¹ᵁ U.1).ι (f ≫ g) h
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H✝ : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hZ : AlgebraicGeometry.IsAffine Z
      this : ∀ {X Y : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.Hom …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      H : P (CategoryTheory.CategoryStruct.comp ((TopologicalSpace.Opens.map f.base) …
      ⊢ P (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    rw [← morphismRestrict_ι_assoc] at H
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H✝ : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hZ : AlgebraicGeometry.IsAffine Z
      this : ∀ {X Y : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y} {g : Quiver.Hom …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      H : P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict  …
      ⊢ P (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    exact this H inferInstance
    /-
      🎉 no goals
    -/
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
    Z : AlgebraicGeometry.Scheme
    hZ : AlgebraicGeometry.IsAffine Z
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : P (CategoryTheory.CategoryStruct.comp f g)
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ P f
  -/
  wlog hY : IsAffine X generalizing X
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
      Z : AlgebraicGeometry.Scheme
      hZ : AlgebraicGeometry.IsAffine Z
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hY✝ : AlgebraicGeometry.IsAffine Y
      this : ∀ {X : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y}, P (CategoryTheor …
      hY : Not (AlgebraicGeometry.IsAffine X)
      ⊢ P f
    -/
  · rw [IsLocalAtSource.iff_of_iSup_eq_top (P := P) _ (iSup_affineOpens_eq_top X)]
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
      Z : AlgebraicGeometry.Scheme
      hZ : AlgebraicGeometry.IsAffine Z
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hY✝ : AlgebraicGeometry.IsAffine Y
      this : ∀ {X : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y}, P (CategoryTheor …
      hY : Not (AlgebraicGeometry.IsAffine X)
      ⊢ ∀ (i : ↑X.affineOpens), P (CategoryTheory.CategoryStruct.comp (↑i).ι f)
    -/
    intro U
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
      Z : AlgebraicGeometry.Scheme
      hZ : AlgebraicGeometry.IsAffine Z
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hY✝ : AlgebraicGeometry.IsAffine Y
      this : ∀ {X : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y}, P (CategoryTheor …
      hY : Not (AlgebraicGeometry.IsAffine X)
      U : ↑X.affineOpens
      ⊢ P (CategoryTheory.CategoryStruct.comp (↑U).ι f)
    -/
    have H := comp_of_isOpenImmersion P U.1.ι (f ≫ g) h
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H✝ : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      Z : AlgebraicGeometry.Scheme
      hZ : AlgebraicGeometry.IsAffine Z
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hY✝ : AlgebraicGeometry.IsAffine Y
      this : ∀ {X : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y}, P (CategoryTheor …
      hY : Not (AlgebraicGeometry.IsAffine X)
      U : ↑X.affineOpens
      H : P (CategoryTheory.CategoryStruct.comp (↑U).ι (CategoryTheory.CategoryStruc …
      ⊢ P (CategoryTheory.CategoryStruct.comp (↑U).ι f)
    -/
    rw [← Category.assoc] at H
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      H✝ : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      Z : AlgebraicGeometry.Scheme
      hZ : AlgebraicGeometry.IsAffine Z
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      h : P (CategoryTheory.CategoryStruct.comp f g)
      hY✝ : AlgebraicGeometry.IsAffine Y
      this : ∀ {X : AlgebraicGeometry.Scheme} {f : Quiver.Hom X Y}, P (CategoryTheor …
      hY : Not (AlgebraicGeometry.IsAffine X)
      U : ↑X.affineOpens
      H : P (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
      ⊢ P (CategoryTheory.CategoryStruct.comp (↑U).ι f)
    -/
    exact this H inferInstance
    /-
      🎉 no goals
    -/
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
    Z : AlgebraicGeometry.Scheme
    hZ : AlgebraicGeometry.IsAffine Z
    Y : AlgebraicGeometry.Scheme
    g : Quiver.Hom Y Z
    hY✝ : AlgebraicGeometry.IsAffine Y
    X : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : P (CategoryTheory.CategoryStruct.comp f g)
    hY : AlgebraicGeometry.IsAffine X
    ⊢ P f
  -/
  rw [iff_of_isAffine (P := P)] at h ⊢
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    H : ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
    Z : AlgebraicGeometry.Scheme
    hZ : AlgebraicGeometry.IsAffine Z
    Y : AlgebraicGeometry.Scheme
    g : Quiver.Hom Y Z
    hY✝ : AlgebraicGeometry.IsAffine Y
    X : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : Q (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.CategoryStruct.comp …
    hY : AlgebraicGeometry.IsAffine X
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
  -/
  exact H _ _ h
  /-
    🎉 no goals
  -/


lemma isMultiplicative (hPc : RingHom.StableUnderComposition Q)
    (hPi : RingHom.ContainsIdentities Q) :
    P.IsMultiplicative where
  comp_mem := (stableUnderComposition hPc).comp_mem
  id_mem := (containsIdentities hPi).id_mem


include Q in
lemma of_isOpenImmersion (hP : RingHom.ContainsIdentities Q) [IsOpenImmersion f] : P f :=
  haveI : P.ContainsIdentities := containsIdentities hP
  IsLocalAtSource.of_isOpenImmersion f


lemma isStableUnderBaseChange (hP : RingHom.IsStableUnderBaseChange Q) :
    P.IsStableUnderBaseChange := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    ⊢ P.IsStableUnderBaseChange
  -/
  apply HasAffineProperty.isStableUnderBaseChange
  /-
    case hP'
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    ⊢ (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing S] = …
  -/
  letI := HasAffineProperty.isLocal_affineProperty P
  /-
    case hP'
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing …
    ⊢ (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing S] = …
  -/
  apply AffineTargetMorphismProperty.IsStableUnderBaseChange.mk
  /-
    case hP'.H
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing …
    ⊢ ∀ ⦃X Y S : AlgebraicGeometry.Scheme⦄ [inst : AlgebraicGeometry.IsAffine S] [ …
  -/
  intros X Y S _ _ f g H
  /-
    case hP'.H
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing …
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H : AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] …
    ⊢ AlgebraicGeometry.sourceAffineLocally (fun {R S} [CommRing R] [CommRing S] = …
  -/
  rw [← HasAffineProperty.iff_of_isAffine (P := P)] at H ⊢
  /-
    case hP'.H
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing …
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    H : P g
    ⊢ P (CategoryTheory.Limits.pullback.fst f g)
  -/
  wlog hX : IsAffine Y generalizing Y
  · rw [IsLocalAtSource.iff_of_openCover (P := P)
      (Scheme.Pullback.openCoverOfRight Y.affineCover f g)]
    /-
      case hP'.H.inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
      this✝ : (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRin …
      X Y S : AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsAffine S
      inst✝ : AlgebraicGeometry.IsAffine X
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      H : P g
      this : ∀ ⦃Y : AlgebraicGeometry.Scheme⦄ (g : Quiver.Hom Y S), P g → AlgebraicG …
      hX : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ ∀ (i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfRight Y.affineCover f g …
    -/
    intro i
    simp only [Scheme.Pullback.openCoverOfRight_obj, Scheme.Pullback.openCoverOfRight_map,
      limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app, Category.comp_id]
    /-
      case hP'.H.inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
      hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
      this✝ : (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRin …
      X Y S : AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsAffine S
      inst✝ : AlgebraicGeometry.IsAffine X
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      H : P g
      this : ∀ ⦃Y : AlgebraicGeometry.Scheme⦄ (g : Quiver.Hom Y S), P g → AlgebraicG …
      hX : Not (AlgebraicGeometry.IsAffine Y)
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfRight Y.affineCover f g).J
      ⊢ P (CategoryTheory.Limits.pullback.fst f (CategoryTheory.CategoryStruct.comp  …
    -/
    apply this _ (comp_of_isOpenImmersion _ _ _ H) inferInstance
    /-
      🎉 no goals
    -/
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing …
    X S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    Y : AlgebraicGeometry.Scheme
    g : Quiver.Hom Y S
    H : P g
    hX : AlgebraicGeometry.IsAffine Y
    ⊢ P (CategoryTheory.Limits.pullback.fst f g)
  -/
  rw [iff_of_isAffine (P := P)] at H ⊢
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝² : AlgebraicGeometry.HasRingHomProperty P Q
    hP : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.sourceAffineLocally fun {R S} [CommRing R] [CommRing …
    X S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    Y : AlgebraicGeometry.Scheme
    g : Quiver.Hom Y S
    H : Q (AlgebraicGeometry.Scheme.Hom.appTop g).hom
    hX : AlgebraicGeometry.IsAffine Y
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.Limits.pullback.fst f …
  -/
  exact hP.pullback_fst_appTop _ (isLocal_ringHomProperty P).respectsIso _ _ H
  /-
    🎉 no goals
  -/


include Q in
private lemma respects_isOpenImmersion_aux
    (hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource Q)
    {X Y : Scheme.{u}} [IsAffine Y] {U : Y.Opens}
    (f : X ⟶ U.toScheme) (hf : P f) : P (f ≫ U.ι) := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    U : Y.Opens
    f : Quiver.Hom X ↑U
    hf : P f
    ⊢ P (CategoryTheory.CategoryStruct.comp f U.ι)
  -/
  wlog hYa : ∃ (a : Γ(Y, ⊤)), U = Y.basicOpen a generalizing X Y
    /-
      case inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      U : Y.Opens
      f : Quiver.Hom X ↑U
      hf : P f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
      hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
      ⊢ P (CategoryTheory.CategoryStruct.comp f U.ι)
    -/
  · obtain ⟨(Us : Set Y.Opens), hUs, heq⟩ := Opens.isBasis_iff_cover.mp (isBasis_basicOpen Y) U
    /-
      case inr.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      U : Y.Opens
      f : Quiver.Hom X ↑U
      hf : P f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
      hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
      Us : Set Y.Opens
      hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
      heq : Eq U (SupSet.sSup Us)
      ⊢ P (CategoryTheory.CategoryStruct.comp f U.ι)
    -/
    let V (s : Us) : X.Opens := f ⁻¹ᵁ U.ι ⁻¹ᵁ s
    /-
      case inr.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      U : Y.Opens
      f : Quiver.Hom X ↑U
      hf : P f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
      hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
      Us : Set Y.Opens
      hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
      heq : Eq U (SupSet.sSup Us)
      V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
      ⊢ P (CategoryTheory.CategoryStruct.comp f U.ι)
    -/
    rw [IsLocalAtSource.iff_of_iSup_eq_top (P := P) V]
      /-
        case inr.intro.intro
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        ⊢ ∀ (i : ↑Us), P (CategoryTheory.CategoryStruct.comp (V i).ι (CategoryTheory.C …
      -/
    · intro s
      /-
        case inr.intro.intro
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        s : ↑Us
        ⊢ P (CategoryTheory.CategoryStruct.comp (V s).ι (CategoryTheory.CategoryStruct …
      -/
      let f' : (V s).toScheme ⟶ U.ι ⁻¹ᵁ s := f ∣_ U.ι ⁻¹ᵁ s
      /-
        case inr.intro.intro
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        s : ↑Us
        f' : Quiver.Hom ↑(V s) ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) := Alge …
        ⊢ P (CategoryTheory.CategoryStruct.comp (V s).ι (CategoryTheory.CategoryStruct …
      -/
      have hf' : P f' := IsLocalAtTarget.restrict hf _
      let e : (U.ι ⁻¹ᵁ s).toScheme ≅ s := IsOpenImmersion.isoOfRangeEq ((U.ι ⁻¹ᵁ s).ι ≫ U.ι) s.1.ι
        (by simpa [Set.range_comp, Set.image_preimage_eq_iff, heq] using le_sSup s.2)
      have heq : (V s).ι ≫ f ≫ U.ι = f' ≫ e.hom ≫ s.1.ι := by
        simp only [V, IsOpenImmersion.isoOfRangeEq_hom_fac, f', e, morphismRestrict_ι_assoc]
      /-
        case inr.intro.intro
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq✝ : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        s : ↑Us
        f' : Quiver.Hom ↑(V s) ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) := Alge …
        hf' : P f'
        e : CategoryTheory.Iso ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) ↑↑s :=  …
        heq : Eq (CategoryTheory.CategoryStruct.comp (V s).ι (CategoryTheory.CategoryS …
        ⊢ P (CategoryTheory.CategoryStruct.comp (V s).ι (CategoryTheory.CategoryStruct …
      -/
      rw [heq, ← Category.assoc]
      /-
        case inr.intro.intro
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq✝ : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        s : ↑Us
        f' : Quiver.Hom ↑(V s) ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) := Alge …
        hf' : P f'
        e : CategoryTheory.Iso ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) ↑↑s :=  …
        heq : Eq (CategoryTheory.CategoryStruct.comp (V s).ι (CategoryTheory.CategoryS …
        ⊢ P (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f' …
      -/
      refine this _ ?_ ?_
        /-
          case inr.intro.intro.refine_1
          P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
          Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
          inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
          hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
          X Y : AlgebraicGeometry.Scheme
          inst✝ : AlgebraicGeometry.IsAffine Y
          U : Y.Opens
          f : Quiver.Hom X ↑U
          hf : P f
          this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
          hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
          Us : Set Y.Opens
          hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
          heq✝ : Eq U (SupSet.sSup Us)
          V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
          s : ↑Us
          f' : Quiver.Hom ↑(V s) ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) := Alge …
          hf' : P f'
          e : CategoryTheory.Iso ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) ↑↑s :=  …
          heq : Eq (CategoryTheory.CategoryStruct.comp (V s).ι (CategoryTheory.CategoryS …
          ⊢ P (CategoryTheory.CategoryStruct.comp f' e.hom)
        -/
      · rwa [P.cancel_right_of_respectsIso]
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.intro.refine_2
          P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
          Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
          inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
          hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
          X Y : AlgebraicGeometry.Scheme
          inst✝ : AlgebraicGeometry.IsAffine Y
          U : Y.Opens
          f : Quiver.Hom X ↑U
          hf : P f
          this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
          hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
          Us : Set Y.Opens
          hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
          heq✝ : Eq U (SupSet.sSup Us)
          V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
          s : ↑Us
          f' : Quiver.Hom ↑(V s) ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) := Alge …
          hf' : P f'
          e : CategoryTheory.Iso ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) ↑↑s :=  …
          heq : Eq (CategoryTheory.CategoryStruct.comp (V s).ι (CategoryTheory.CategoryS …
          ⊢ Exists fun a => Eq (↑s) (Y.basicOpen a)
        -/
      · obtain ⟨a, ha⟩ := hUs s.2
        /-
          case inr.intro.intro.refine_2.intro
          P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
          Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
          inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
          hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
          X Y : AlgebraicGeometry.Scheme
          inst✝ : AlgebraicGeometry.IsAffine Y
          U : Y.Opens
          f : Quiver.Hom X ↑U
          hf : P f
          this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
          hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
          Us : Set Y.Opens
          hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
          heq✝ : Eq U (SupSet.sSup Us)
          V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
          s : ↑Us
          f' : Quiver.Hom ↑(V s) ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) := Alge …
          hf' : P f'
          e : CategoryTheory.Iso ↑((TopologicalSpace.Opens.map U.ι.base).obj ↑s) ↑↑s :=  …
          heq : Eq (CategoryTheory.CategoryStruct.comp (V s).ι (CategoryTheory.CategoryS …
          a : ↑(Y.presheaf.obj { unop := Top.top })
          ha : Eq (Y.basicOpen a) ↑s
          ⊢ Exists fun a => Eq (↑s) (Y.basicOpen a)
        -/
        use a, ha.symm
        /-
          🎉 no goals
        -/
      /-
        case inr.intro.intro
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        ⊢ Eq (iSup V) Top.top
      -/
    · apply f.preimage_iSup_eq_top
      /-
        case inr.intro.intro
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        ⊢ Eq (iSup fun i => (TopologicalSpace.Opens.map U.ι.base).obj ↑i) Top.top
      -/
      apply U.ι.image_injective
      /-
        case inr.intro.intro.a
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        ⊢ Eq ((fun x => (AlgebraicGeometry.Scheme.Hom.opensFunctor U.ι).obj x) (iSup f …
      -/
      simp only [U.ι.image_iSup, U.ι.image_preimage_eq_opensRange_inter, Scheme.Opens.opensRange_ι]
      /-
        case inr.intro.intro.a
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        ⊢ Eq (iSup fun i => Min.min U ↑i) ((AlgebraicGeometry.Scheme.Hom.opensFunctor  …
      -/
      conv_rhs => rw [Scheme.Hom.image_top_eq_opensRange, Scheme.Opens.opensRange_ι, heq]
      /-
        case inr.intro.intro.a
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        ⊢ Eq (iSup fun i => Min.min U ↑i) (SupSet.sSup Us)
      -/
      ext : 1
      /-
        case inr.intro.intro.a.h
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        ⊢ Eq ↑(iSup fun i => Min.min U ↑i) ↑(SupSet.sSup Us)
      -/
      have (i : Us) : U ⊓ i.1 = i.1 := by simp [heq, le_sSup i.property]
      /-
        case inr.intro.intro.a.h
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsAffine Y
        U : Y.Opens
        f : Quiver.Hom X ↑U
        hf : P f
        this✝ : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine  …
        hYa : Not (Exists fun a => Eq U (Y.basicOpen a))
        Us : Set Y.Opens
        hUs : HasSubset.Subset Us (Set.range Y.basicOpen)
        heq : Eq U (SupSet.sSup Us)
        V : ↑Us → X.Opens := fun s => (TopologicalSpace.Opens.map f.base).obj ((Topolo …
        this : ∀ (i : ↑Us), Eq (Min.min U ↑i) ↑i
        ⊢ Eq ↑(iSup fun i => Min.min U ↑i) ↑(SupSet.sSup Us)
      -/
      simp [this]
      /-
        🎉 no goals
      -/
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    U : Y.Opens
    f : Quiver.Hom X ↑U
    hf : P f
    hYa : Exists fun a => Eq U (Y.basicOpen a)
    ⊢ P (CategoryTheory.CategoryStruct.comp f U.ι)
  -/
  obtain ⟨a, rfl⟩ := hYa
  /-
    case intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    a : ↑(Y.presheaf.obj { unop := Top.top })
    f : Quiver.Hom X ↑(Y.basicOpen a)
    hf : P f
    ⊢ P (CategoryTheory.CategoryStruct.comp f (Y.basicOpen a).ι)
  -/
  wlog hX : IsAffine X generalizing X Y
    /-
      case intro.inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      a : ↑(Y.presheaf.obj { unop := Top.top })
      f : Quiver.Hom X ↑(Y.basicOpen a)
      hf : P f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
      hX : Not (AlgebraicGeometry.IsAffine X)
      ⊢ P (CategoryTheory.CategoryStruct.comp f (Y.basicOpen a).ι)
    -/
  · rw [IsLocalAtSource.iff_of_iSup_eq_top (P := P) _ (iSup_affineOpens_eq_top _)]
    /-
      case intro.inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      a : ↑(Y.presheaf.obj { unop := Top.top })
      f : Quiver.Hom X ↑(Y.basicOpen a)
      hf : P f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
      hX : Not (AlgebraicGeometry.IsAffine X)
      ⊢ ∀ (i : ↑X.affineOpens), P (CategoryTheory.CategoryStruct.comp (↑i).ι (Catego …
    -/
    intro V
    /-
      case intro.inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      a : ↑(Y.presheaf.obj { unop := Top.top })
      f : Quiver.Hom X ↑(Y.basicOpen a)
      hf : P f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
      hX : Not (AlgebraicGeometry.IsAffine X)
      V : ↑X.affineOpens
      ⊢ P (CategoryTheory.CategoryStruct.comp (↑V).ι (CategoryTheory.CategoryStruct. …
    -/
    rw [← Category.assoc]
    /-
      case intro.inr
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      a : ↑(Y.presheaf.obj { unop := Top.top })
      f : Quiver.Hom X ↑(Y.basicOpen a)
      hf : P f
      this : ∀ {X Y : AlgebraicGeometry.Scheme} [inst : AlgebraicGeometry.IsAffine Y …
      hX : Not (AlgebraicGeometry.IsAffine X)
      V : ↑X.affineOpens
      ⊢ P (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp (↑ …
    -/
    exact this _ _ (IsLocalAtSource.comp hf _) V.2
    /-
      🎉 no goals
    -/
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    a : ↑(Y.presheaf.obj { unop := Top.top })
    f : Quiver.Hom X ↑(Y.basicOpen a)
    hf : P f
    hX : AlgebraicGeometry.IsAffine X
    ⊢ P (CategoryTheory.CategoryStruct.comp f (Y.basicOpen a).ι)
  -/
  rw [HasRingHomProperty.iff_of_isAffine (P := P)] at hf ⊢
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝¹ : AlgebraicGeometry.HasRingHomProperty P Q
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    a : ↑(Y.presheaf.obj { unop := Top.top })
    f : Quiver.Hom X ↑(Y.basicOpen a)
    hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
    hX : AlgebraicGeometry.IsAffine X
    ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheory.CategoryStruct.comp f …
  -/
  exact hQ _ a _ hf
  /-
    🎉 no goals
  -/


/-- Any property of scheme morphisms induced by a property of ring homomorphisms is stable
under composition with open immersions. -/
lemma respects_isOpenImmersion (hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource Q) :
    P.Respects @IsOpenImmersion where
  postcomp {X Y Z} i hi f hf := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y Z : AlgebraicGeometry.Scheme
      i : Quiver.Hom Y Z
      hi : AlgebraicGeometry.IsOpenImmersion i
      f : Quiver.Hom X Y
      hf : P f
      ⊢ P (CategoryTheory.CategoryStruct.comp f i)
    -/
    wlog hZ : IsAffine Z generalizing X Y Z
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        hi : AlgebraicGeometry.IsOpenImmersion i
        f : Quiver.Hom X Y
        hf : P f
        this : ∀ {X Y Z : AlgebraicGeometry.Scheme} (i : Quiver.Hom Y Z), AlgebraicGeo …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        ⊢ P (CategoryTheory.CategoryStruct.comp f i)
      -/
    · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := P) _ (iSup_affineOpens_eq_top _)]
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        hi : AlgebraicGeometry.IsOpenImmersion i
        f : Quiver.Hom X Y
        hf : P f
        this : ∀ {X Y Z : AlgebraicGeometry.Scheme} (i : Quiver.Hom Y Z), AlgebraicGeo …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        ⊢ ∀ (i_1 : ↑Z.affineOpens), P (AlgebraicGeometry.morphismRestrict (CategoryThe …
      -/
      intro U
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        hi : AlgebraicGeometry.IsOpenImmersion i
        f : Quiver.Hom X Y
        hf : P f
        this : ∀ {X Y Z : AlgebraicGeometry.Scheme} (i : Quiver.Hom Y Z), AlgebraicGeo …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        U : ↑Z.affineOpens
        ⊢ P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.comp f  …
      -/
      rw [morphismRestrict_comp]
      /-
        case inr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
        hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
        X Y Z : AlgebraicGeometry.Scheme
        i : Quiver.Hom Y Z
        hi : AlgebraicGeometry.IsOpenImmersion i
        f : Quiver.Hom X Y
        hf : P f
        this : ∀ {X Y Z : AlgebraicGeometry.Scheme} (i : Quiver.Hom Y Z), AlgebraicGeo …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        U : ↑Z.affineOpens
        ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict f  …
      -/
      exact this _ inferInstance _ (IsLocalAtTarget.restrict hf _) U.2
      /-
        🎉 no goals
      -/
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y Z : AlgebraicGeometry.Scheme
      i : Quiver.Hom Y Z
      hi : AlgebraicGeometry.IsOpenImmersion i
      f : Quiver.Hom X Y
      hf : P f
      hZ : AlgebraicGeometry.IsAffine Z
      ⊢ P (CategoryTheory.CategoryStruct.comp f i)
    -/
    let e : Y ≅ i.opensRange.toScheme := IsOpenImmersion.isoOfRangeEq i i.opensRange.ι (by simp)
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y Z : AlgebraicGeometry.Scheme
      i : Quiver.Hom Y Z
      hi : AlgebraicGeometry.IsOpenImmersion i
      f : Quiver.Hom X Y
      hf : P f
      hZ : AlgebraicGeometry.IsAffine Z
      e : CategoryTheory.Iso Y ↑(AlgebraicGeometry.Scheme.Hom.opensRange i) := Algeb …
      ⊢ P (CategoryTheory.CategoryStruct.comp f i)
    -/
    rw [show f ≫ i = f ≫ e.hom ≫ i.opensRange.ι by simp [e], ← Category.assoc]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      X Y Z : AlgebraicGeometry.Scheme
      i : Quiver.Hom Y Z
      hi : AlgebraicGeometry.IsOpenImmersion i
      f : Quiver.Hom X Y
      hf : P f
      hZ : AlgebraicGeometry.IsAffine Z
      e : CategoryTheory.Iso Y ↑(AlgebraicGeometry.Scheme.Hom.opensRange i) := Algeb …
      ⊢ P (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f  …
    -/
    exact respects_isOpenImmersion_aux hQ _ (by rwa [P.cancel_right_of_respectsIso])
    /-
      🎉 no goals
    -/


omit [HasRingHomProperty P Q] in
/-- If `P` is induced by `Locally Q`, it suffices to check `Q` on affine open sets locally around
points of the source. -/
lemma iff_exists_appLE_locally
    (hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource Q)
    (hQi : RespectsIso Q) [HasRingHomProperty P (Locally Q)] :
    P f ↔ ∀ (x : X), ∃ (U : Y.affineOpens) (V : X.affineOpens) (_ : x ∈ V.1) (e : V.1 ≤ f ⁻¹ᵁ U.1),
      Q (f.appLE U V e).hom := by
  have := respects_isOpenImmersion (P := P)
    (RingHom.locally_StableUnderCompositionWithLocalizationAwaySource hQ)
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
    this : P.Respects @AlgebraicGeometry.IsOpenImmersion
    ⊢ Iff (P f) (∀ (x : ↑↑X.toPresheafedSpace), Exists fun U => Exists fun V => Ex …
  -/
  refine ⟨fun hf x ↦ ?_, fun hf ↦ (IsLocalAtSource.iff_exists_resLE (P := P)).mpr <| fun x ↦ ?_⟩
  · obtain ⟨U, hU, hfx, _⟩ := Opens.isBasis_iff_nbhd.mp (isBasis_affine_open Y)
      (Opens.mem_top <| f.base x)
    obtain ⟨V, hV, hx, e⟩ := Opens.isBasis_iff_nbhd.mp (isBasis_affine_open X)
      (show x ∈ f ⁻¹ᵁ U from hfx)
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this : P.Respects @AlgebraicGeometry.IsOpenImmersion
      hf : P f
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => Q (Algebraic …
    -/
    simp_rw [HasRingHomProperty.iff_appLE (P := P), locally_iff_isLocalization hQi] at hf
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => Q (Algebraic …
    -/
    obtain ⟨s, hs, hfs⟩ := hf ⟨U, hU⟩ ⟨V, hV⟩ e
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      s : Finset ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hs : Eq (Ideal.span ↑s) Top.top
      hfs : ∀ (t : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })), Membership.mem s t → ∀ ( …
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => Q (Algebraic …
    -/
    apply iSup_basicOpen_of_span_eq_top at hs
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      s : Finset ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hfs : ∀ (t : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })), Membership.mem s t → ∀ ( …
      hs : Eq (iSup fun i => iSup fun h => X.basicOpen i) ↑⟨V, hV⟩
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => Q (Algebraic …
    -/
    have : x ∈ (⨆ i ∈ s, X.basicOpen i) := hs.symm ▸ hx
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this✝ : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      s : Finset ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hfs : ∀ (t : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })), Membership.mem s t → ∀ ( …
      hs : Eq (iSup fun i => iSup fun h => X.basicOpen i) ↑⟨V, hV⟩
      this : Membership.mem (iSup fun i => iSup fun h => X.basicOpen i) x
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => Q (Algebraic …
    -/
    have : ∃ r ∈ s, x ∈ X.basicOpen r := by simpa using this
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this✝¹ : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      s : Finset ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hfs : ∀ (t : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })), Membership.mem s t → ∀ ( …
      hs : Eq (iSup fun i => iSup fun h => X.basicOpen i) ↑⟨V, hV⟩
      this✝ : Membership.mem (iSup fun i => iSup fun h => X.basicOpen i) x
      this : Exists fun r => And (Membership.mem s r) (Membership.mem (X.basicOpen r …
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => Q (Algebraic …
    -/
    obtain ⟨r, hr, hrs⟩ := this
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this✝ : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      s : Finset ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hfs : ∀ (t : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })), Membership.mem s t → ∀ ( …
      hs : Eq (iSup fun i => iSup fun h => X.basicOpen i) ↑⟨V, hV⟩
      this : Membership.mem (iSup fun i => iSup fun h => X.basicOpen i) x
      r : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hr : Membership.mem s r
      hrs : Membership.mem (X.basicOpen r) x
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => Q (Algebraic …
    -/
    refine ⟨⟨U, hU⟩, ⟨X.basicOpen r, hV.basicOpen r⟩, hrs, (X.basicOpen_le r).trans e, ?_⟩
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this✝ : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      s : Finset ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hfs : ∀ (t : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })), Membership.mem s t → ∀ ( …
      hs : Eq (iSup fun i => iSup fun h => X.basicOpen i) ↑⟨V, hV⟩
      this : Membership.mem (iSup fun i => iSup fun h => X.basicOpen i) x
      r : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hr : Membership.mem s r
      hrs : Membership.mem (X.basicOpen r) x
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.appLE f ↑⟨U, hU⟩ ↑⟨X.basicOpen r, ⋯⟩ ⋯).hom
    -/
    rw [← f.appLE_map e (homOfLE (X.basicOpen_le r)).op]
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this✝ : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      s : Finset ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hfs : ∀ (t : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })), Membership.mem s t → ∀ ( …
      hs : Eq (iSup fun i => iSup fun h => X.basicOpen i) ↑⟨V, hV⟩
      this : Membership.mem (iSup fun i => iSup fun h => X.basicOpen i) x
      r : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hr : Membership.mem s r
      hrs : Membership.mem (X.basicOpen r) x
      ⊢ Q (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appLE f  …
    -/
    haveI : IsLocalization.Away r Γ(X, X.basicOpen r) := hV.isLocalization_basicOpen r
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this✝¹ : P.Respects @AlgebraicGeometry.IsOpenImmersion
      x : ↑↑X.toPresheafedSpace
      U : TopologicalSpace.Opens ↑↑Y.toPresheafedSpace
      hU : Membership.mem Y.affineOpens U
      hfx : Membership.mem U (f.base x)
      right✝ : LE.le U Top.top
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hx : Membership.mem V x
      e : LE.le V ((TopologicalSpace.Opens.map f.base).obj U)
      hf : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((Topological …
      s : Finset ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hfs : ∀ (t : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })), Membership.mem s t → ∀ ( …
      hs : Eq (iSup fun i => iSup fun h => X.basicOpen i) ↑⟨V, hV⟩
      this✝ : Membership.mem (iSup fun i => iSup fun h => X.basicOpen i) x
      r : ↑(X.presheaf.obj { unop := ↑⟨V, hV⟩ })
      hr : Membership.mem s r
      hrs : Membership.mem (X.basicOpen r) x
      this : IsLocalization.Away r ↑(X.presheaf.obj { unop := X.basicOpen r })
      ⊢ Q (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.appLE f  …
    -/
    exact hfs r hr _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this : P.Respects @AlgebraicGeometry.IsOpenImmersion
      hf : ∀ (x : ↑↑X.toPresheafedSpace), Exists fun U => Exists fun V => Exists fun …
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => P (Algebraic …
    -/
  · obtain ⟨U, V, hxV, e, hf⟩ := hf x
    /-
      case refine_2.intro.intro.intro.intro
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this : P.Respects @AlgebraicGeometry.IsOpenImmersion
      hf✝ : ∀ (x : ↑↑X.toPresheafedSpace), Exists fun U => Exists fun V => Exists fu …
      x : ↑↑X.toPresheafedSpace
      U : ↑Y.affineOpens
      V : ↑X.affineOpens
      hxV : Membership.mem (↑V) x
      e : LE.le (↑V) ((TopologicalSpace.Opens.map f.base).obj ↑U)
      hf : Q (AlgebraicGeometry.Scheme.Hom.appLE f (↑U) (↑V) e).hom
      ⊢ Exists fun U => Exists fun V => Exists fun x => Exists fun e => P (Algebraic …
    -/
    use U, V, hxV, e
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      inst✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
      this : P.Respects @AlgebraicGeometry.IsOpenImmersion
      hf✝ : ∀ (x : ↑↑X.toPresheafedSpace), Exists fun U => Exists fun V => Exists fu …
      x : ↑↑X.toPresheafedSpace
      U : ↑Y.affineOpens
      V : ↑X.affineOpens
      hxV : Membership.mem (↑V) x
      e : LE.le (↑V) ((TopologicalSpace.Opens.map f.base).obj ↑U)
      hf : Q (AlgebraicGeometry.Scheme.Hom.appLE f (↑U) (↑V) e).hom
      ⊢ P (AlgebraicGeometry.Scheme.Hom.resLE f (↑U) (↑V) e)
    -/
    simp only [iff_of_isAffine (P := P), Scheme.Hom.appLE, homOfLE_leOfHom] at hf ⊢
    haveI : (toMorphismProperty (Locally Q)).RespectsIso := toMorphismProperty_respectsIso_iff.mp <|
      (isLocal_ringHomProperty P).respectsIso
    exact (MorphismProperty.arrow_mk_iso_iff (toMorphismProperty (Locally Q))
      (arrowResLEAppIso f U V e)).mpr (locally_of hQi _ hf)


/-- `P` can be checked locally around points of the source. -/
lemma iff_exists_appLE
    (hQ : StableUnderCompositionWithLocalizationAwaySource Q) : P f ↔
    ∀ (x : X), ∃ (U : Y.affineOpens) (V : X.affineOpens) (_ : x ∈ V.1) (e : V.1 ≤ f ⁻¹ᵁ U.1),
      Q (f.appLE U V e).hom := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    ⊢ Iff (P f) (∀ (x : ↑↑X.toPresheafedSpace), Exists fun U => Exists fun V => Ex …
  -/
  haveI inst : HasRingHomProperty P Q := inferInstance
  haveI : HasRingHomProperty P (Locally Q) := by
    apply @copy (P' := P) (Q := Q) (Q' := Locally Q)
    · infer_instance
    · rfl
    · intro R S _ _ f
      exact (locally_iff_of_localizationSpanTarget (isLocal_ringHomProperty P).respectsIso
        (isLocal_ringHomProperty P).ofLocalizationSpanTarget _).symm
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    inst : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRing …
    this : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRing …
    ⊢ Iff (P f) (∀ (x : ↑↑X.toPresheafedSpace), Exists fun U => Exists fun V => Ex …
  -/
  rw [iff_exists_appLE_locally (P := P) hQ]
  /-
    case hQi
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    inst : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRing …
    this : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRing …
    ⊢ RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
  -/
  haveI : HasRingHomProperty P Q := inst
  /-
    case hQi
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hQ : RingHom.StableUnderCompositionWithLocalizationAwaySource fun {R S} [CommR …
    inst : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRing …
    this✝ : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRin …
    this : AlgebraicGeometry.HasRingHomProperty P fun {R S} [CommRing R] [CommRing …
    ⊢ RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
  -/
  apply (isLocal_ringHomProperty P (Q := Q)).respectsIso
  /-
    🎉 no goals
  -/


omit [HasRingHomProperty P Q] in
lemma locally_of_iff (hQl : LocalizationAwayPreserves Q)
    (hQa : StableUnderCompositionWithLocalizationAway Q)
    (h : ∀ {X Y : Scheme.{u}} (f : X ⟶ Y), P f ↔
      ∀ (x : X), ∃ (U : Y.affineOpens) (V : X.affineOpens) (_ : x ∈ V.1) (e : V.1 ≤ f ⁻¹ᵁ U.1),
      Q (f.appLE U V e).hom) : HasRingHomProperty P (Locally Q) where
  isLocal_ringHomProperty := locally_propertyIsLocal hQl hQa
  eq_affineLocally' := by
    haveI : HasRingHomProperty (affineLocally (Locally Q)) (Locally Q) :=
      ⟨locally_propertyIsLocal hQl hQa, rfl⟩
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQa : RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R …
      h : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (P f) (∀ (x : …
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      ⊢ Eq P (AlgebraicGeometry.affineLocally fun {R S} [CommRing R] [CommRing S] => …
    -/
    ext X Y f
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQl : RingHom.LocalizationAwayPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQa : RingHom.StableUnderCompositionWithLocalizationAway fun {R S} [CommRing R …
      h : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (P f) (∀ (x : …
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (P f) (AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing …
    -/
    rw [h, iff_exists_appLE_locally (P := affineLocally (Locally Q)) hQa.left hQa.respectsIso]
    /-
      🎉 no goals
    -/


