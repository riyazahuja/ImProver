/-- This is the affine target morphism property where the source is affine and
the induced map of rings on global sections satisfies `P`. -/
def affineAnd : AffineTargetMorphismProperty :=
  fun X _ f ↦ IsAffine X ∧ Q (f.appTop).hom


@[simp]
lemma affineAnd_apply {X Y : Scheme.{u}} (f : X ⟶ Y) [IsAffine Y] :
    affineAnd Q f ↔ IsAffine X ∧ Q (f.appTop).hom :=
  Iff.rfl


attribute [local simp] AffineTargetMorphismProperty.toProperty_apply


/-- If `P` respects isos, also `affineAnd P` respects isomorphisms. -/
lemma affineAnd_respectsIso (hP : RingHom.RespectsIso Q) :
    (affineAnd Q).toProperty.RespectsIso := by
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    ⊢ (AlgebraicGeometry.affineAnd fun {R S} [CommRing R] [CommRing S] => Q).toPro …
  -/
  refine RespectsIso.mk _ ?_ ?_
    /-
      case refine_1
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Quive …
    -/
  · intro X Y Z e f ⟨hZ, ⟨hY, hf⟩⟩
    /-
      case refine_1
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso X Y
      f : Quiver.Hom Y Z
      hZ : AlgebraicGeometry.IsAffine Z
      hY : AlgebraicGeometry.IsAffine Y
      hf : (fun {R S} [CommRing R] [CommRing S] => Q) (AlgebraicGeometry.Scheme.Hom. …
      ⊢ (AlgebraicGeometry.affineAnd fun {R S} [CommRing R] [CommRing S] => Q).toPro …
    -/
    simpa [hP.cancel_right_isIso, isAffine_of_isIso e.hom]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Quive …
    -/
  · intro X Y Z e f ⟨hZ, hf⟩
    /-
      case refine_2
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      hZ : AlgebraicGeometry.IsAffine Y
      hf : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
      ⊢ (AlgebraicGeometry.affineAnd fun {R S} [CommRing R] [CommRing S] => Q).toPro …
    -/
    simpa [AffineTargetMorphismProperty.toProperty, isAffine_of_isIso e.inv, hP.cancel_left_isIso]
    /-
      🎉 no goals
    -/


/-- `affineAnd P` is local if `P` is local on the (algebraic) source. -/
lemma affineAnd_isLocal (hPi : RingHom.RespectsIso Q) (hQl : RingHom.LocalizationPreserves Q)
    (hQs : RingHom.OfLocalizationSpan Q) : (affineAnd Q).IsLocal where
  respectsIso := affineAnd_respectsIso hPi
  to_basicOpen {X Y _} f r := fun ⟨hX, hf⟩ ↦ by
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      x✝¹ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      r : ↑(Y.presheaf.obj { unop := Top.top })
      x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
      hX : AlgebraicGeometry.IsAffine X
      hf : (fun {R S} [CommRing R] [CommRing S] => Q) (AlgebraicGeometry.Scheme.Hom. …
      ⊢ AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) (Alge …
    -/
    simp only [Opens.map_top] at hf
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      x✝¹ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      r : ↑(Y.presheaf.obj { unop := Top.top })
      x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
      hX : AlgebraicGeometry.IsAffine X
      hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
      ⊢ AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) (Alge …
    -/
    constructor
      /-
        case left
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        ⊢ AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj (Y.basi …
      -/
    · simp only [Scheme.preimage_basicOpen, Opens.map_top]
      /-
        case left
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        ⊢ AlgebraicGeometry.IsAffine ↑(X.basicOpen ((AlgebraicGeometry.Scheme.Hom.app  …
      -/
      exact (isAffineOpen_top X).basicOpen _
      /-
        🎉 no goals
      -/
      /-
        case right
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        ⊢ (fun {R S} [CommRing R] [CommRing S] => Q) (AlgebraicGeometry.Scheme.Hom.app …
      -/
    · dsimp only
      /-
        case right
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop (AlgebraicGeometry.morphismRestrict f …
      -/
      rw [morphismRestrict_appTop, CommRingCat.hom_comp, hPi.cancel_right_isIso]
      -- Not sure why the `show` fixes the following `rw` complaining about "motive is incorrect"
      /-
        case right
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.opensFu …
      -/
      show Q (Scheme.Hom.app f ((Y.basicOpen r).ι ''ᵁ ⊤)).hom
      /-
        case right
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.opensFu …
      -/
      rw [Scheme.Opens.ι_image_top]
      rw [(isAffineOpen_top Y).app_basicOpen_eq_away_map f (isAffineOpen_top X),
        CommRingCat.hom_comp, hPi.cancel_right_isIso, ← Scheme.Hom.appTop]
      /-
        case right
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        ⊢ Q (CommRingCat.ofHom (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y. …
      -/
      dsimp only [Opens.map_top]
      /-
        case right
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        ⊢ Q (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y.basicOpen r })) (↑( …
      -/
      haveI := (isAffineOpen_top X).isLocalization_basicOpen (f.appTop r)
      /-
        case right
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appTop f).hom r) ↑(X …
        ⊢ Q (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y.basicOpen r })) (↑( …
      -/
      apply hQl
      /-
        case right.a
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝¹ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        r : ↑(Y.presheaf.obj { unop := Top.top })
        x✝ : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
        hX : AlgebraicGeometry.IsAffine X
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
        this : IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appTop f).hom r) ↑(X …
        ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
      -/
      exact hf
      /-
        🎉 no goals
      -/
  of_basicOpenCover {X Y _} f s hs hf := by
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      x✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hf : ∀ (r : Subtype fun x => Membership.mem s x), AlgebraicGeometry.affineAnd  …
      ⊢ AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
    -/
    dsimp [affineAnd] at hf
    haveI : IsAffine X := by
      apply isAffine_of_isAffineOpen_basicOpen (f.appTop '' s)
      · apply_fun Ideal.map (f.appTop).hom at hs
        rwa [Ideal.map_span, Ideal.map_top] at hs
      · rintro - ⟨r, hr, rfl⟩
        simp_rw [Scheme.preimage_basicOpen] at hf
        exact (hf ⟨r, hr⟩).left
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      x✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hf : ∀ (r : Subtype fun x => Membership.mem s x), And (AlgebraicGeometry.IsAff …
      this : AlgebraicGeometry.IsAffine X
      ⊢ AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
    -/
    refine ⟨inferInstance, hQs.ofIsLocalization' hPi (f.appTop).hom s hs fun a ↦ ?_⟩
    refine ⟨Γ(Y, Y.basicOpen a.val), Γ(X, X.basicOpen (f.appTop a.val)), inferInstance,
      inferInstance, inferInstance, inferInstance, inferInstance, ?_, ?_⟩
      /-
        case refine_1
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        s : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hs : Eq (Ideal.span ↑s) Top.top
        hf : ∀ (r : Subtype fun x => Membership.mem s x), And (AlgebraicGeometry.IsAff …
        this : AlgebraicGeometry.IsAffine X
        a : ↑↑s
        ⊢ IsLocalization.Away ((AlgebraicGeometry.Scheme.Hom.appTop f).hom ↑a) ↑(X.pre …
      -/
    · exact (isAffineOpen_top X).isLocalization_basicOpen (f.appTop a.val)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        s : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hs : Eq (Ideal.span ↑s) Top.top
        hf : ∀ (r : Subtype fun x => Membership.mem s x), And (AlgebraicGeometry.IsAff …
        this : AlgebraicGeometry.IsAffine X
        a : ↑↑s
        ⊢ Q (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y.basicOpen ↑a })) (↑ …
      -/
    · obtain ⟨_, hf⟩ := hf a
      /-
        case refine_2.intro
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        s : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hs : Eq (Ideal.span ↑s) Top.top
        hf✝ : ∀ (r : Subtype fun x => Membership.mem s x), And (AlgebraicGeometry.IsAf …
        this : AlgebraicGeometry.IsAffine X
        a : ↑↑s
        left✝ : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ( …
        hf : Q (AlgebraicGeometry.Scheme.Hom.appTop (AlgebraicGeometry.morphismRestric …
        ⊢ Q (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y.basicOpen ↑a })) (↑ …
      -/
      rw [morphismRestrict_appTop, CommRingCat.hom_comp, hPi.cancel_right_isIso] at hf
      -- Not sure why the `show` fixes the following `rw` complaining about "motive is incorrect"
      /-
        case refine_2.intro
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        s : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hs : Eq (Ideal.span ↑s) Top.top
        hf✝ : ∀ (r : Subtype fun x => Membership.mem s x), And (AlgebraicGeometry.IsAf …
        this : AlgebraicGeometry.IsAffine X
        a : ↑↑s
        left✝ : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ( …
        hf : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.open …
        ⊢ Q (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y.basicOpen ↑a })) (↑ …
      -/
      have hf : Q (Scheme.Hom.app f ((Y.basicOpen a.1).ι ''ᵁ ⊤)).hom := hf
      /-
        case refine_2.intro
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        s : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hs : Eq (Ideal.span ↑s) Top.top
        hf✝¹ : ∀ (r : Subtype fun x => Membership.mem s x), And (AlgebraicGeometry.IsA …
        this : AlgebraicGeometry.IsAffine X
        a : ↑↑s
        left✝ : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ( …
        hf✝ : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.ope …
        hf : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.open …
        ⊢ Q (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y.basicOpen ↑a })) (↑ …
      -/
      rw [Scheme.Opens.ι_image_top] at hf
      /-
        case refine_2.intro
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        s : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hs : Eq (Ideal.span ↑s) Top.top
        hf✝¹ : ∀ (r : Subtype fun x => Membership.mem s x), And (AlgebraicGeometry.IsA …
        this : AlgebraicGeometry.IsAffine X
        a : ↑↑s
        left✝ : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ( …
        hf✝ : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.ope …
        hf : Q (AlgebraicGeometry.Scheme.Hom.app f (Y.basicOpen ↑a)).hom
        ⊢ Q (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y.basicOpen ↑a })) (↑ …
      -/
      rw [(isAffineOpen_top Y).app_basicOpen_eq_away_map _ (isAffineOpen_top X)] at hf
      /-
        case refine_2.intro
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hPi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
        hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
        hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        x✝ : AlgebraicGeometry.IsAffine Y
        f : Quiver.Hom X Y
        s : Finset ↑(Y.presheaf.obj { unop := Top.top })
        hs : Eq (Ideal.span ↑s) Top.top
        hf✝¹ : ∀ (r : Subtype fun x => Membership.mem s x), And (AlgebraicGeometry.IsA …
        this : AlgebraicGeometry.IsAffine X
        a : ↑↑s
        left✝ : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ( …
        hf✝ : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.ope …
        hf : Q (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (IsLocalization. …
        ⊢ Q (IsLocalization.Away.map (↑(Y.presheaf.obj { unop := Y.basicOpen ↑a })) (↑ …
      -/
      rwa [CommRingCat.hom_comp, hPi.cancel_right_isIso] at hf
      /-
        🎉 no goals
      -/


/-- If `P` is stable under base change, so is `affineAnd P`. -/
lemma affineAnd_isStableUnderBaseChange (hQi : RingHom.RespectsIso Q)
    (hQb : RingHom.IsStableUnderBaseChange Q) :
    (affineAnd Q).IsStableUnderBaseChange := by
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    hQb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    ⊢ (AlgebraicGeometry.affineAnd fun {R S} [CommRing R] [CommRing S] => Q).IsSta …
  -/
  haveI : (affineAnd Q).toProperty.RespectsIso := affineAnd_respectsIso hQi
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    hQb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.affineAnd fun {R S} [CommRing R] [CommRing S] => Q). …
    ⊢ (AlgebraicGeometry.affineAnd fun {R S} [CommRing R] [CommRing S] => Q).IsSta …
  -/
  apply AffineTargetMorphismProperty.IsStableUnderBaseChange.mk
  /-
    case H
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    hQb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.affineAnd fun {R S} [CommRing R] [CommRing S] => Q). …
    ⊢ ∀ ⦃X Y S : AlgebraicGeometry.Scheme⦄ [inst : AlgebraicGeometry.IsAffine S] [ …
  -/
  intro X Y S _ _ f g ⟨hY, hg⟩
  /-
    case H
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    hQb : RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] => Q
    this : (AlgebraicGeometry.affineAnd fun {R S} [CommRing R] [CommRing S] => Q). …
    X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    hY : AlgebraicGeometry.IsAffine Y
    hg : (fun {R S} [CommRing R] [CommRing S] => Q) (AlgebraicGeometry.Scheme.Hom. …
    ⊢ AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) (Cate …
  -/
  exact ⟨inferInstance, hQb.pullback_fst_appTop _ hQi f _ hg⟩
  /-
    🎉 no goals
  -/


lemma targetAffineLocally_affineAnd_iff (hQi : RingHom.RespectsIso Q)
    {X Y : Scheme.{u}} (f : X ⟶ Y) :
    targetAffineLocally (affineAnd Q) f ↔ ∀ U : Y.Opens, IsAffineOpen U →
      IsAffineOpen (f ⁻¹ᵁ U) ∧ Q (f.app U).hom := by
  simp only [targetAffineLocally, affineAnd_apply, morphismRestrict_app, CommRingCat.hom_comp,
    hQi.cancel_right_isIso]
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (∀ (U : ↑Y.affineOpens), And (AlgebraicGeometry.IsAffine ↑((TopologicalS …
  -/
  refine ⟨fun hf U hU ↦ ?_, fun h U ↦ ?_⟩
    /-
      case refine_1
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : ∀ (U : ↑Y.affineOpens), And (AlgebraicGeometry.IsAffine ↑((TopologicalSpa …
      U : Y.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ And (AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj …
    -/
  · obtain ⟨hfU, hf⟩ := hf ⟨U, hU⟩
    /-
      case refine_1.intro
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf✝ : ∀ (U : ↑Y.affineOpens), And (AlgebraicGeometry.IsAffine ↑((TopologicalSp …
      U : Y.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hfU : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ↑⟨U …
      hf : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.open …
      ⊢ And (AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map f.base).obj …
    -/
    use hfU
    /-
      case right
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf✝ : ∀ (U : ↑Y.affineOpens), And (AlgebraicGeometry.IsAffine ↑((TopologicalSp …
      U : Y.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hfU : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ↑⟨U …
      hf : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.open …
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f U).hom
    -/
    have hf : Q (Scheme.Hom.app f (((⟨U, hU⟩ : Y.affineOpens) : Y.Opens).ι ''ᵁ ⊤)).hom := hf
    /-
      case right
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf✝¹ : ∀ (U : ↑Y.affineOpens), And (AlgebraicGeometry.IsAffine ↑((TopologicalS …
      U : Y.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      hfU : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ↑⟨U …
      hf✝ : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.ope …
      hf : Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.open …
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f U).hom
    -/
    rwa [Scheme.Opens.ι_image_top] at hf
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → And (AlgebraicGeometry …
      U : ↑Y.affineOpens
      ⊢ And (AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ↑U …
    -/
  · refine ⟨(h U U.2).1, ?_⟩
    /-
      case refine_2
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → And (AlgebraicGeometry …
      U : ↑Y.affineOpens
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.opensFu …
    -/
    show Q (Scheme.Hom.app f ((U : Y.Opens).ι ''ᵁ ⊤)).hom
    /-
      case refine_2
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → And (AlgebraicGeometry …
      U : ↑Y.affineOpens
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.opensFu …
    -/
    rw [Scheme.Opens.ι_image_top]
    /-
      case refine_2
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → And (AlgebraicGeometry …
      U : ↑Y.affineOpens
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f ↑U).hom
    -/
    exact (h U U.2).2
    /-
      🎉 no goals
    -/


/-- Variant of `targetAffineLocally_affineAnd_iff` where `IsAffineHom` is bundled. -/
lemma targetAffineLocally_affineAnd_iff' (hQi : RingHom.RespectsIso Q)
    {X Y : Scheme.{u}} (f : X ⟶ Y) :
    targetAffineLocally (affineAnd Q) f ↔
      IsAffineHom f ∧ ∀ U : Y.Opens, IsAffineOpen U → Q (f.app U).hom := by
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fun  …
  -/
  rw [targetAffineLocally_affineAnd_iff hQi, isAffineHom_iff]
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → And (AlgebraicGeome …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma targetAffineLocally_affineAnd_iff_affineLocally (hQ : RingHom.PropertyIsLocal Q)
    {X Y : Scheme.{u}} (f : X ⟶ Y) :
    targetAffineLocally (affineAnd Q) f ↔ IsAffineHom f ∧ affineLocally Q f := by
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fun  …
  -/
  haveI : HasRingHomProperty (affineLocally Q) Q := ⟨hQ, rfl⟩
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
    ⊢ Iff (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fun  …
  -/
  rw [targetAffineLocally_affineAnd_iff' hQ.respectsIso]
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
    ⊢ Iff (And (AlgebraicGeometry.IsAffineHom f) (∀ (U : Y.Opens), AlgebraicGeomet …
  -/
  simp only [and_congr_right_iff]
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
    ⊢ AlgebraicGeometry.IsAffineHom f → Iff (∀ (U : Y.Opens), AlgebraicGeometry.Is …
  -/
  intro hf
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
    hf : AlgebraicGeometry.IsAffineHom f
    ⊢ Iff (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometr …
  -/
  constructor
    /-
      case mp
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      hf : AlgebraicGeometry.IsAffineHom f
      ⊢ (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.Sc …
    -/
  · wlog hY : IsAffine Y
      /-
        case mp.inr
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        this✝ : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally  …
        hf : AlgebraicGeometry.IsAffineHom f
        this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        ⊢ (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.Sc …
      -/
    · intro h
      rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := affineLocally Q)
        _ (iSup_affineOpens_eq_top _)]
      /-
        case mp.inr
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        this✝ : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally  …
        hf : AlgebraicGeometry.IsAffineHom f
        this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.S …
        ⊢ ∀ (i : ↑Y.affineOpens), AlgebraicGeometry.affineLocally (fun {R S} [CommRing …
      -/
      intro U
      /-
        case mp.inr
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        this✝ : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally  …
        hf : AlgebraicGeometry.IsAffineHom f
        this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.S …
        U : ↑Y.affineOpens
        ⊢ AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => Q) ( …
      -/
      have : IsAffine (f ⁻¹ᵁ U) := hf.isAffine_preimage U U.2
      rw [HasRingHomProperty.iff_of_isAffine (P := affineLocally Q),
        morphismRestrict_appTop, CommRingCat.hom_comp, hQ.respectsIso.cancel_right_isIso]
      /-
        case mp.inr
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        this✝¹ : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally …
        hf : AlgebraicGeometry.IsAffineHom f
        this✝ : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] →  …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.S …
        U : ↑Y.affineOpens
        this : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ↑U)
        ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f ((AlgebraicGeometry.Scheme.Hom.opensFu …
      -/
      apply h
      /-
        case mp.inr.a
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        this✝¹ : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally …
        hf : AlgebraicGeometry.IsAffineHom f
        this✝ : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] →  …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.S …
        U : ↑Y.affineOpens
        this : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ↑U)
        ⊢ AlgebraicGeometry.IsAffineOpen ((AlgebraicGeometry.Scheme.Hom.opensFunctor ( …
      -/
      rw [Scheme.Opens.ι_image_top]
      /-
        case mp.inr.a
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        this✝¹ : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally …
        hf : AlgebraicGeometry.IsAffineHom f
        this✝ : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] →  …
        hY : Not (AlgebraicGeometry.IsAffine Y)
        h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.S …
        U : ↑Y.affineOpens
        this : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map f.base).obj ↑U)
        ⊢ AlgebraicGeometry.IsAffineOpen ↑U
      -/
      exact U.2
      /-
        🎉 no goals
      -/
    /-
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      hf : AlgebraicGeometry.IsAffineHom f
      hY : AlgebraicGeometry.IsAffine Y
      ⊢ (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.Sc …
    -/
    intro h
    /-
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      hf : AlgebraicGeometry.IsAffineHom f
      hY : AlgebraicGeometry.IsAffine Y
      h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.S …
      ⊢ AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => Q) f
    -/
    have : IsAffine X := isAffine_of_isAffineHom f
    /-
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this✝ : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally  …
      hf : AlgebraicGeometry.IsAffineHom f
      hY : AlgebraicGeometry.IsAffine Y
      h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.S …
      this : AlgebraicGeometry.IsAffine X
      ⊢ AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => Q) f
    -/
    rw [HasRingHomProperty.iff_of_isAffine (P := affineLocally Q)]
    /-
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this✝ : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally  …
      hf : AlgebraicGeometry.IsAffineHom f
      hY : AlgebraicGeometry.IsAffine Y
      h : ∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Q (AlgebraicGeometry.S …
      this : AlgebraicGeometry.IsAffine X
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
    -/
    exact h ⊤ (isAffineOpen_top Y)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      hf : AlgebraicGeometry.IsAffineHom f
      ⊢ AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => Q) f …
    -/
  · intro h U hU
    /-
      case mpr
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      hf : AlgebraicGeometry.IsAffineHom f
      h : AlgebraicGeometry.affineLocally (fun {R S} [CommRing R] [CommRing S] => Q) f
      U : Y.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f U).hom
    -/
    rw [affineLocally_iff_affineOpens_le] at h
    /-
      case mpr
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      hf : AlgebraicGeometry.IsAffineHom f
      h : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((TopologicalS …
      U : Y.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.app f U).hom
    -/
    rw [f.app_eq_appLE]
    /-
      case mpr
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : AlgebraicGeometry.HasRingHomProperty (AlgebraicGeometry.affineLocally f …
      hf : AlgebraicGeometry.IsAffineHom f
      h : ∀ (U : ↑Y.affineOpens) (V : ↑X.affineOpens) (e : LE.le (↑V) ((TopologicalS …
      U : Y.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ Q (AlgebraicGeometry.Scheme.Hom.appLE f U ((TopologicalSpace.Opens.map f.bas …
    -/
    exact h ⟨U, hU⟩ ⟨f ⁻¹ᵁ U, hf.isAffine_preimage U hU⟩ (by simp)
    /-
      🎉 no goals
    -/


lemma targetAffineLocally_affineAnd_eq_affineLocally (hQ : RingHom.PropertyIsLocal Q) :
    targetAffineLocally (affineAnd Q) =
      (@IsAffineHom ⊓ @affineLocally Q : MorphismProperty Scheme.{u}) := by
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
    ⊢ Eq (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fun { …
  -/
  ext X Y f
  /-
    case h
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hQ : RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fun  …
  -/
  exact targetAffineLocally_affineAnd_iff_affineLocally hQ f
  /-
    🎉 no goals
  -/


lemma targetAffineLocally_affineAnd_le
    (hQW : ∀ {R S : Type u} [CommRing R] [CommRing S] {f : R →+* S}, Q f → W f) :
    targetAffineLocally (affineAnd Q) ≤ targetAffineLocally (affineAnd W) := by
  /-
    Q W : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R …
    hQW : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] {f : RingHom  …
    ⊢ LE.le (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fu …
  -/
  intro X Y f h U
  /-
    Q W : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R …
    hQW : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] {f : RingHom  …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    h : AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fun {R  …
    U : ↑Y.affineOpens
    ⊢ AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => W) (Alge …
  -/
  exact ⟨(h U).1, hQW (h U).2⟩
  /-
    🎉 no goals
  -/


/-- If `P` is a morphism property affine locally defined by `affineAnd Q`, `P` is stable under
composition if `Q` is. -/
lemma HasAffineProperty.affineAnd_isStableUnderComposition {P : MorphismProperty Scheme.{u}}
    (hA : HasAffineProperty P (affineAnd Q)) (hQ : RingHom.StableUnderComposition Q) :
    P.IsStableUnderComposition where
  comp_mem {X Y Z} f g hf hg := by
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P f
      hg : P g
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    haveI := hA
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P f
      hg : P g
      this : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    wlog hZ : IsAffine Z
      /-
        case inr
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
        hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        this✝ : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun …
        this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        ⊢ P (CategoryTheory.CategoryStruct.comp f g)
      -/
    · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := P) _ (iSup_affineOpens_eq_top _)]
      /-
        case inr
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
        hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        this✝ : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun …
        this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        ⊢ ∀ (i : ↑Z.affineOpens), P (AlgebraicGeometry.morphismRestrict (CategoryTheor …
      -/
      intro U
      /-
        case inr
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
        hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        this✝ : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun …
        this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        U : ↑Z.affineOpens
        ⊢ P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.comp f  …
      -/
      rw [morphismRestrict_comp]
      /-
        case inr
        Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
        hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
        X Y Z : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        hf : P f
        hg : P g
        this✝ : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun …
        this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
        hZ : Not (AlgebraicGeometry.IsAffine Z)
        U : ↑Z.affineOpens
        ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict f  …
      -/
      exact this hA hQ _ _ (IsLocalAtTarget.restrict hf _) (IsLocalAtTarget.restrict hg _) hA U.2
      /-
        🎉 no goals
      -/
    /-
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P f
      hg : P g
      this : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
      hZ : AlgebraicGeometry.IsAffine Z
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    rw [HasAffineProperty.iff_of_isAffine (P := P) (Q := (affineAnd Q))] at hg
    /-
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P f
      this : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
      hZ : AlgebraicGeometry.IsAffine Z
      hg : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) g
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    obtain ⟨hY, hg⟩ := hg
    /-
      case intro
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : P f
      this : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
      hZ : AlgebraicGeometry.IsAffine Z
      hY : AlgebraicGeometry.IsAffine Y
      hg : Q (AlgebraicGeometry.Scheme.Hom.appTop g).hom
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    rw [HasAffineProperty.iff_of_isAffine (P := P) (Q := (affineAnd Q))] at hf
    /-
      case intro
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      this : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
      hZ : AlgebraicGeometry.IsAffine Z
      hY : AlgebraicGeometry.IsAffine Y
      hf : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
      hg : Q (AlgebraicGeometry.Scheme.Hom.appTop g).hom
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    obtain ⟨hX, hf⟩ := hf
    /-
      case intro.intro
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      this : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
      hZ : AlgebraicGeometry.IsAffine Z
      hY : AlgebraicGeometry.IsAffine Y
      hg : Q (AlgebraicGeometry.Scheme.Hom.appTop g).hom
      hX : AlgebraicGeometry.IsAffine X
      hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
      ⊢ P (CategoryTheory.CategoryStruct.comp f g)
    -/
    rw [HasAffineProperty.iff_of_isAffine (P := P) (Q := (affineAnd Q))]
    /-
      case intro.intro
      Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQ : RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] => Q
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      this : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
      hZ : AlgebraicGeometry.IsAffine Z
      hY : AlgebraicGeometry.IsAffine Y
      hg : Q (AlgebraicGeometry.Scheme.Hom.appTop g).hom
      hX : AlgebraicGeometry.IsAffine X
      hf : Q (AlgebraicGeometry.Scheme.Hom.appTop f).hom
      ⊢ AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) (Cate …
    -/
    exact ⟨hX, hQ _ _ hg hf⟩
    /-
      🎉 no goals
    -/


/-- If `P` is a morphism property affine locally defined by `affineAnd Q`, `P` is stable under
base change if `Q` is. -/
lemma HasAffineProperty.affineAnd_isStableUnderBaseChange {P : MorphismProperty Scheme.{u}}
    (_ : HasAffineProperty P (affineAnd Q)) (hQi : RingHom.RespectsIso Q)
    (hQb : RingHom.IsStableUnderBaseChange Q) :
    P.IsStableUnderBaseChange :=
  HasAffineProperty.isStableUnderBaseChange
    (AlgebraicGeometry.affineAnd_isStableUnderBaseChange hQi hQb)


/-- If `Q` contains identities and respects isomorphisms (i.e. is satisfied by isomorphisms),
and `P` is affine locally defined by `affineAnd Q`, then `P` contains identities. -/
lemma HasAffineProperty.affineAnd_containsIdentities {P : MorphismProperty Scheme.{u}}
    (hA : HasAffineProperty P (affineAnd Q)) (hQi : RingHom.RespectsIso Q)
    (hQ : RingHom.ContainsIdentities Q) :
    P.ContainsIdentities where
  id_mem X := by
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQ : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => Q
      X : AlgebraicGeometry.Scheme
      ⊢ P (CategoryTheory.CategoryStruct.id X)
    -/
    rw [eq_targetAffineLocally P, targetAffineLocally_affineAnd_iff hQi]
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQ : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => Q
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (U : X.Opens), AlgebraicGeometry.IsAffineOpen U → And (AlgebraicGeometry.I …
    -/
    intro U hU
    /-
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQ : RingHom.ContainsIdentities fun {R S} [CommRing R] [CommRing S] => Q
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ And (AlgebraicGeometry.IsAffineOpen ((TopologicalSpace.Opens.map (CategoryTh …
    -/
    exact ⟨hU, hQ _⟩
    /-
      🎉 no goals
    -/


/-- A convenience constructor for `HasAffineProperty P (affineAnd Q)`. The `IsAffineHom` is bundled,
since this goes well with defining morphism properties via `extends IsAffineHom`. -/
lemma HasAffineProperty.affineAnd_iff (P : MorphismProperty Scheme.{u})
    (hQi : RingHom.RespectsIso Q) (hQl : RingHom.LocalizationPreserves Q)
    (hQs : RingHom.OfLocalizationSpan Q) :
    HasAffineProperty P (affineAnd Q) ↔
      ∀ {X Y : Scheme.{u}} (f : X ⟶ Y), P f ↔
        (IsAffineHom f ∧ ∀ U : Y.Opens, IsAffineOpen U → Q (f.app U).hom) := by
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
    hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
    ⊢ Iff (AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
  -/
  simp_rw [isAffineHom_iff]
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
    hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
    hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
    ⊢ Iff (AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun  …
  -/
  refine ⟨fun h X Y f ↦ ?_, fun h ↦ ⟨affineAnd_isLocal hQi hQl hQs, ?_⟩⟩
    /-
      case refine_1
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      h : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (P f) (And (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → Algebrai …
    -/
  · rw [eq_targetAffineLocally P, targetAffineLocally_affineAnd_iff hQi]
    /-
      case refine_1
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      h : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → And (AlgebraicGeome …
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      h : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (P f) (And (∀ …
      ⊢ Eq P (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fun …
    -/
  · ext X Y f
    /-
      case refine_2.h
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      h : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (P f) (And (∀ …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (P f) (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAn …
    -/
    rw [targetAffineLocally_affineAnd_iff hQi, h f]
    /-
      case refine_2.h
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hQi : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => Q
      hQl : RingHom.LocalizationPreserves fun {R S} [CommRing R] [CommRing S] => Q
      hQs : RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] => Q
      h : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y), Iff (P f) (And (∀ …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (And (∀ (U : Y.Opens), AlgebraicGeometry.IsAffineOpen U → AlgebraicGeome …
    -/
    aesop
    /-
      🎉 no goals
    -/


lemma HasAffineProperty.affineAnd_le_isAffineHom (P : MorphismProperty Scheme.{u})
    (hA : HasAffineProperty P (affineAnd Q)) : P ≤ @IsAffineHom := by
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
    ⊢ LE.le P @AlgebraicGeometry.IsAffineHom
  -/
  intro X Y f hf
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : P f
    ⊢ AlgebraicGeometry.IsAffineHom f
  -/
  wlog hY : IsAffine Y
    /-
      case inr
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : P f
      this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ AlgebraicGeometry.IsAffineHom f
    -/
  · rw [IsLocalAtTarget.iff_of_iSup_eq_top (P := @IsAffineHom) _ (iSup_affineOpens_eq_top _)]
    /-
      case inr
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : P f
      this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      ⊢ ∀ (i : ↑Y.affineOpens), AlgebraicGeometry.IsAffineHom (AlgebraicGeometry.mor …
    -/
    intro U
    /-
      case inr
      Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      hf : P f
      this : ∀ {Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → R …
      hY : Not (AlgebraicGeometry.IsAffine Y)
      U : ↑Y.affineOpens
      ⊢ AlgebraicGeometry.IsAffineHom (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    exact this P hA _ (IsLocalAtTarget.restrict hf _) U.2
    /-
      🎉 no goals
    -/
  /-
    Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : P f
    hY : AlgebraicGeometry.IsAffine Y
    ⊢ AlgebraicGeometry.IsAffineHom f
  -/
  rw [HasAffineProperty.iff_of_isAffine (P := P) (Q := (affineAnd Q))] at hf
  /-
    Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : AlgebraicGeometry.IsAffine Y
    hf : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
    ⊢ AlgebraicGeometry.IsAffineHom f
  -/
  rw [HasAffineProperty.iff_of_isAffine (P := @IsAffineHom)]
  /-
    Q✝ Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hA : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : AlgebraicGeometry.IsAffine Y
    hf : AlgebraicGeometry.affineAnd (fun {R S} [CommRing R] [CommRing S] => Q) f
    ⊢ AlgebraicGeometry.IsAffine X
  -/
  exact hf.1
  /-
    🎉 no goals
  -/


lemma HasAffineProperty.affineAnd_eq_of_propertyIsLocal {P P' : MorphismProperty Scheme.{u}}
    (hP : HasAffineProperty P (affineAnd Q)) [HasRingHomProperty P' Q] :
    P = (@IsAffineHom ⊓ P' : MorphismProperty Scheme.{u}) := by
  rw [HasAffineProperty.eq_targetAffineLocally (P := P),
    targetAffineLocally_affineAnd_eq_affineLocally,
    HasRingHomProperty.eq_affineLocally (P := P')]
  /-
    Q : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    P P' : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
    inst✝ : AlgebraicGeometry.HasRingHomProperty P' fun {R S} [CommRing R] [CommRi …
    ⊢ RingHom.PropertyIsLocal fun {R S} [CommRing R] [CommRing S] => Q
  -/
  exact HasRingHomProperty.isLocal_ringHomProperty P'
  /-
    🎉 no goals
  -/


lemma HasAffineProperty.affineAnd_le_affineAnd {P P' : MorphismProperty Scheme.{u}}
    (hP : HasAffineProperty P (affineAnd Q)) (hP' : HasAffineProperty P' (affineAnd Q'))
    (hQQ' : ∀ {R S : Type u} [CommRing R] [CommRing S] {f : R →+* S}, Q f → Q' f) :
    P ≤ P' := by
  rw [HasAffineProperty.eq_targetAffineLocally (P := P),
    HasAffineProperty.eq_targetAffineLocally (P := P')]
  /-
    Q Q' : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom  …
    P P' : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.HasAffineProperty P (AlgebraicGeometry.affineAnd fun {R …
    hP' : AlgebraicGeometry.HasAffineProperty P' (AlgebraicGeometry.affineAnd fun  …
    hQQ' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] {f : RingHom …
    ⊢ LE.le (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.affineAnd fu …
  -/
  exact targetAffineLocally_affineAnd_le hQQ'
  /-
    🎉 no goals
  -/


