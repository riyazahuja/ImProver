/-- A morphism is `QuasiSeparated` if diagonal map is quasi-compact. -/
@[mk_iff]
class QuasiSeparated (f : X ⟶ Y) : Prop where
  /-- A morphism is `QuasiSeparated` if diagonal map is quasi-compact. -/
  diagonalQuasiCompact : QuasiCompact (pullback.diagonal f) := by infer_instance


theorem quasiSeparatedSpace_iff_affine (X : Scheme) :
    QuasiSeparatedSpace X ↔ ∀ U V : X.affineOpens, IsCompact (U ∩ V : Set X) := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Iff (QuasiSeparatedSpace ↑↑X.toPresheafedSpace) (∀ (U V : ↑X.affineOpens), I …
  -/
  rw [quasiSeparatedSpace_iff]
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Iff (∀ (U V : Set ↑↑X.toPresheafedSpace), IsOpen U → IsCompact U → IsOpen V  …
  -/
  constructor
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      ⊢ (∀ (U V : Set ↑↑X.toPresheafedSpace), IsOpen U → IsCompact U → IsOpen V → Is …
    -/
  · intro H U V; exact H U V U.1.2 U.2.isCompact V.1.2 V.2.isCompact
                 /-
                   🎉 no goals
                 -/
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      ⊢ (∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)) → ∀ (U V : Set ↑ …
    -/
  · intro H
    suffices
      ∀ (U : X.Opens) (_ : IsCompact U.1) (V : X.Opens) (_ : IsCompact V.1),
        IsCompact (U ⊓ V).1
      by intro U V hU hU' hV hV'; exact this ⟨U, hU⟩ hU' ⟨V, hV⟩ hV'
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
      ⊢ ∀ (U : X.Opens), IsCompact U.carrier → ∀ (V : X.Opens), IsCompact V.carrier  …
    -/
    intro U hU V hV
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
      U : X.Opens
      hU : IsCompact U.carrier
      V : X.Opens
      hV : IsCompact V.carrier
      ⊢ IsCompact (Min.min U V).carrier
    -/
    refine compact_open_induction_on V hV ?_ ?_
      /-
        case mpr.refine_1
        X : AlgebraicGeometry.Scheme
        H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
        U : X.Opens
        hU : IsCompact U.carrier
        V : X.Opens
        hV : IsCompact V.carrier
        ⊢ IsCompact (Min.min U Bot.bot).carrier
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        X : AlgebraicGeometry.Scheme
        H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
        U : X.Opens
        hU : IsCompact U.carrier
        V : X.Opens
        hV : IsCompact V.carrier
        ⊢ ∀ (S : X.Opens), IsCompact S.carrier → ∀ (U_1 : ↑X.affineOpens), IsCompact ( …
      -/
    · intro S _ V hV
      /-
        case mpr.refine_2
        X : AlgebraicGeometry.Scheme
        H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
        U : X.Opens
        hU : IsCompact U.carrier
        V✝ : X.Opens
        hV✝ : IsCompact V✝.carrier
        S : X.Opens
        x✝ : IsCompact S.carrier
        V : ↑X.affineOpens
        hV : IsCompact (Min.min U S).carrier
        ⊢ IsCompact (Min.min U (Max.max S ↑V)).carrier
      -/
      change IsCompact (U.1 ∩ (S.1 ∪ V.1))
      /-
        case mpr.refine_2
        X : AlgebraicGeometry.Scheme
        H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
        U : X.Opens
        hU : IsCompact U.carrier
        V✝ : X.Opens
        hV✝ : IsCompact V✝.carrier
        S : X.Opens
        x✝ : IsCompact S.carrier
        V : ↑X.affineOpens
        hV : IsCompact (Min.min U S).carrier
        ⊢ IsCompact (Inter.inter U.carrier (Union.union S.carrier ↑↑V))
      -/
      rw [Set.inter_union_distrib_left]
      /-
        case mpr.refine_2
        X : AlgebraicGeometry.Scheme
        H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
        U : X.Opens
        hU : IsCompact U.carrier
        V✝ : X.Opens
        hV✝ : IsCompact V✝.carrier
        S : X.Opens
        x✝ : IsCompact S.carrier
        V : ↑X.affineOpens
        hV : IsCompact (Min.min U S).carrier
        ⊢ IsCompact (Union.union (Inter.inter U.carrier S.carrier) (Inter.inter U.carr …
      -/
      apply hV.union
      /-
        case mpr.refine_2
        X : AlgebraicGeometry.Scheme
        H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
        U : X.Opens
        hU : IsCompact U.carrier
        V✝ : X.Opens
        hV✝ : IsCompact V✝.carrier
        S : X.Opens
        x✝ : IsCompact S.carrier
        V : ↑X.affineOpens
        hV : IsCompact (Min.min U S).carrier
        ⊢ IsCompact (Inter.inter U.carrier ↑↑V)
      -/
      clear hV
      /-
        case mpr.refine_2
        X : AlgebraicGeometry.Scheme
        H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
        U : X.Opens
        hU : IsCompact U.carrier
        V✝ : X.Opens
        hV : IsCompact V✝.carrier
        S : X.Opens
        x✝ : IsCompact S.carrier
        V : ↑X.affineOpens
        ⊢ IsCompact (Inter.inter U.carrier ↑↑V)
      -/
      refine compact_open_induction_on U hU ?_ ?_
        /-
          case mpr.refine_2.refine_1
          X : AlgebraicGeometry.Scheme
          H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
          U : X.Opens
          hU : IsCompact U.carrier
          V✝ : X.Opens
          hV : IsCompact V✝.carrier
          S : X.Opens
          x✝ : IsCompact S.carrier
          V : ↑X.affineOpens
          ⊢ IsCompact (Inter.inter Bot.bot.carrier ↑↑V)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case mpr.refine_2.refine_2
          X : AlgebraicGeometry.Scheme
          H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
          U : X.Opens
          hU : IsCompact U.carrier
          V✝ : X.Opens
          hV : IsCompact V✝.carrier
          S : X.Opens
          x✝ : IsCompact S.carrier
          V : ↑X.affineOpens
          ⊢ ∀ (S : X.Opens), IsCompact S.carrier → ∀ (U : ↑X.affineOpens), IsCompact (In …
        -/
      · intro S _ W hW
        /-
          case mpr.refine_2.refine_2
          X : AlgebraicGeometry.Scheme
          H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
          U : X.Opens
          hU : IsCompact U.carrier
          V✝ : X.Opens
          hV : IsCompact V✝.carrier
          S✝ : X.Opens
          x✝¹ : IsCompact S✝.carrier
          V : ↑X.affineOpens
          S : X.Opens
          x✝ : IsCompact S.carrier
          W : ↑X.affineOpens
          hW : IsCompact (Inter.inter S.carrier ↑↑V)
          ⊢ IsCompact (Inter.inter (Max.max S ↑W).carrier ↑↑V)
        -/
        change IsCompact ((S.1 ∪ W.1) ∩ V.1)
        /-
          case mpr.refine_2.refine_2
          X : AlgebraicGeometry.Scheme
          H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
          U : X.Opens
          hU : IsCompact U.carrier
          V✝ : X.Opens
          hV : IsCompact V✝.carrier
          S✝ : X.Opens
          x✝¹ : IsCompact S✝.carrier
          V : ↑X.affineOpens
          S : X.Opens
          x✝ : IsCompact S.carrier
          W : ↑X.affineOpens
          hW : IsCompact (Inter.inter S.carrier ↑↑V)
          ⊢ IsCompact (Inter.inter (Union.union S.carrier ↑↑W) ↑↑V)
        -/
        rw [Set.union_inter_distrib_right]
        /-
          case mpr.refine_2.refine_2
          X : AlgebraicGeometry.Scheme
          H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
          U : X.Opens
          hU : IsCompact U.carrier
          V✝ : X.Opens
          hV : IsCompact V✝.carrier
          S✝ : X.Opens
          x✝¹ : IsCompact S✝.carrier
          V : ↑X.affineOpens
          S : X.Opens
          x✝ : IsCompact S.carrier
          W : ↑X.affineOpens
          hW : IsCompact (Inter.inter S.carrier ↑↑V)
          ⊢ IsCompact (Union.union (Inter.inter S.carrier ↑↑V) (Inter.inter ↑↑W ↑↑V))
        -/
        apply hW.union
        /-
          case mpr.refine_2.refine_2
          X : AlgebraicGeometry.Scheme
          H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
          U : X.Opens
          hU : IsCompact U.carrier
          V✝ : X.Opens
          hV : IsCompact V✝.carrier
          S✝ : X.Opens
          x✝¹ : IsCompact S✝.carrier
          V : ↑X.affineOpens
          S : X.Opens
          x✝ : IsCompact S.carrier
          W : ↑X.affineOpens
          hW : IsCompact (Inter.inter S.carrier ↑↑V)
          ⊢ IsCompact (Inter.inter ↑↑W ↑↑V)
        -/
        apply H
        /-
          🎉 no goals
        -/


theorem quasiCompact_affineProperty_iff_quasiSeparatedSpace {X Y : Scheme} [IsAffine Y]
    (f : X ⟶ Y) :
    AffineTargetMorphismProperty.diagonal (fun X _ _ _ ↦ CompactSpace X) f ↔
      QuasiSeparatedSpace X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.AffineTargetMorphismProperty.diagonal (fun X x x_1 x  …
  -/
  delta AffineTargetMorphismProperty.diagonal
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    ⊢ Iff (∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quive …
  -/
  rw [quasiSeparatedSpace_iff_affine]
  /-
    X Y : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine Y
    f : Quiver.Hom X Y
    ⊢ Iff (∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quive …
  -/
  constructor
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      ⊢ (∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.Ho …
    -/
  · intro H U V
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
    -/
    haveI : IsAffine _ := U.2
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      this : AlgebraicGeometry.IsAffine ↑↑U
      ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
    -/
    haveI : IsAffine _ := V.2
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      this✝ : AlgebraicGeometry.IsAffine ↑↑U
      this : AlgebraicGeometry.IsAffine ↑↑V
      ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
    -/
    let g : pullback U.1.ι V.1.ι ⟶ X := pullback.fst _ _ ≫ U.1.ι
    -- Porting note: `inferInstance` does not work here
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      this✝ : AlgebraicGeometry.IsAffine ↑↑U
      this : AlgebraicGeometry.IsAffine ↑↑V
      g : Quiver.Hom (CategoryTheory.Limits.pullback (↑U).ι (↑V).ι) X := CategoryThe …
      ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
    -/
    have : IsOpenImmersion g := PresheafedSpace.IsOpenImmersion.comp _ _
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      this✝¹ : AlgebraicGeometry.IsAffine ↑↑U
      this✝ : AlgebraicGeometry.IsAffine ↑↑V
      g : Quiver.Hom (CategoryTheory.Limits.pullback (↑U).ι (↑V).ι) X := CategoryThe …
      this : AlgebraicGeometry.IsOpenImmersion g
      ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
    -/
    have e := Homeomorph.ofIsEmbedding _ this.base_open.isEmbedding
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      this✝¹ : AlgebraicGeometry.IsAffine ↑↑U
      this✝ : AlgebraicGeometry.IsAffine ↑↑V
      g : Quiver.Hom (CategoryTheory.Limits.pullback (↑U).ι (↑V).ι) X := CategoryThe …
      this : AlgebraicGeometry.IsOpenImmersion g
      e : Homeomorph ↑↑(CategoryTheory.Limits.pullback (↑U).ι (↑V).ι).toPresheafedSp …
      ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
    -/
    rw [IsOpenImmersion.range_pullback_to_base_of_left] at e
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      this✝¹ : AlgebraicGeometry.IsAffine ↑↑U
      this✝ : AlgebraicGeometry.IsAffine ↑↑V
      g : Quiver.Hom (CategoryTheory.Limits.pullback (↑U).ι (↑V).ι) X := CategoryThe …
      this : AlgebraicGeometry.IsOpenImmersion g
      e : Homeomorph ↑↑(CategoryTheory.Limits.pullback (↑U).ι (↑V).ι).toPresheafedSp …
      ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
    -/
    erw [Subtype.range_coe, Subtype.range_coe] at e
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      this✝¹ : AlgebraicGeometry.IsAffine ↑↑U
      this✝ : AlgebraicGeometry.IsAffine ↑↑V
      g : Quiver.Hom (CategoryTheory.Limits.pullback (↑U).ι (↑V).ι) X := CategoryThe …
      this : AlgebraicGeometry.IsOpenImmersion g
      e : Homeomorph ↑↑(CategoryTheory.Limits.pullback (↑U).ι (↑V).ι).toPresheafedSp …
      ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
    -/
    rw [isCompact_iff_compactSpace]
    /-
      case mp
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U V : ↑X.affineOpens
      this✝¹ : AlgebraicGeometry.IsAffine ↑↑U
      this✝ : AlgebraicGeometry.IsAffine ↑↑V
      g : Quiver.Hom (CategoryTheory.Limits.pullback (↑U).ι (↑V).ι) X := CategoryThe …
      this : AlgebraicGeometry.IsOpenImmersion g
      e : Homeomorph ↑↑(CategoryTheory.Limits.pullback (↑U).ι (↑V).ι).toPresheafedSp …
      ⊢ CompactSpace ↑(Inter.inter ↑↑U ↑↑V)
    -/
    exact @Homeomorph.compactSpace _ _ _ _ (H _ _) e
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      ⊢ (∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)) → ∀ ⦃U₁ U₂ : Alg …
    -/
  · introv H h₁ h₂
    /-
      case mpr
      X Y : AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝¹ : AlgebraicGeometry.IsAffine U₁
      inst✝ : AlgebraicGeometry.IsAffine U₂
      h₁ : AlgebraicGeometry.IsOpenImmersion f₁
      h₂ : AlgebraicGeometry.IsOpenImmersion f₂
      ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f₁ f₂).toPresheafedSpace
    -/
    let g : pullback f₁ f₂ ⟶ X := pullback.fst _ _ ≫ f₁
    -- Porting note: `inferInstance` does not work here
    /-
      case mpr
      X Y : AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝¹ : AlgebraicGeometry.IsAffine U₁
      inst✝ : AlgebraicGeometry.IsAffine U₂
      h₁ : AlgebraicGeometry.IsOpenImmersion f₁
      h₂ : AlgebraicGeometry.IsOpenImmersion f₂
      g : Quiver.Hom (CategoryTheory.Limits.pullback f₁ f₂) X := CategoryTheory.Cate …
      ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f₁ f₂).toPresheafedSpace
    -/
    have : IsOpenImmersion g := PresheafedSpace.IsOpenImmersion.comp _ _
    /-
      case mpr
      X Y : AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝¹ : AlgebraicGeometry.IsAffine U₁
      inst✝ : AlgebraicGeometry.IsAffine U₂
      h₁ : AlgebraicGeometry.IsOpenImmersion f₁
      h₂ : AlgebraicGeometry.IsOpenImmersion f₂
      g : Quiver.Hom (CategoryTheory.Limits.pullback f₁ f₂) X := CategoryTheory.Cate …
      this : AlgebraicGeometry.IsOpenImmersion g
      ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f₁ f₂).toPresheafedSpace
    -/
    have e := Homeomorph.ofIsEmbedding _ this.base_open.isEmbedding
    /-
      case mpr
      X Y : AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝¹ : AlgebraicGeometry.IsAffine U₁
      inst✝ : AlgebraicGeometry.IsAffine U₂
      h₁ : AlgebraicGeometry.IsOpenImmersion f₁
      h₂ : AlgebraicGeometry.IsOpenImmersion f₂
      g : Quiver.Hom (CategoryTheory.Limits.pullback f₁ f₂) X := CategoryTheory.Cate …
      this : AlgebraicGeometry.IsOpenImmersion g
      e : Homeomorph ↑↑(CategoryTheory.Limits.pullback f₁ f₂).toPresheafedSpace ↑(Se …
      ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f₁ f₂).toPresheafedSpace
    -/
    rw [IsOpenImmersion.range_pullback_to_base_of_left] at e
    /-
      case mpr
      X Y : AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      H : ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝¹ : AlgebraicGeometry.IsAffine U₁
      inst✝ : AlgebraicGeometry.IsAffine U₂
      h₁ : AlgebraicGeometry.IsOpenImmersion f₁
      h₂ : AlgebraicGeometry.IsOpenImmersion f₂
      g : Quiver.Hom (CategoryTheory.Limits.pullback f₁ f₂) X := CategoryTheory.Cate …
      this : AlgebraicGeometry.IsOpenImmersion g
      e : Homeomorph ↑↑(CategoryTheory.Limits.pullback f₁ f₂).toPresheafedSpace ↑(In …
      ⊢ CompactSpace ↑↑(CategoryTheory.Limits.pullback f₁ f₂).toPresheafedSpace
    -/
    simp_rw [isCompact_iff_compactSpace] at H
    exact
      @Homeomorph.compactSpace _ _ _ _
        (H ⟨⟨_, h₁.base_open.isOpen_range⟩, isAffineOpen_opensRange _⟩
          ⟨⟨_, h₂.base_open.isOpen_range⟩, isAffineOpen_opensRange _⟩)
        e.symm


theorem quasiSeparated_eq_diagonal_is_quasiCompact :
                                                                    /-
                                                                      ⊢ Eq (@AlgebraicGeometry.QuasiSeparated) (CategoryTheory.MorphismProperty.diag …
                                                                    -/
    @QuasiSeparated = MorphismProperty.diagonal @QuasiCompact := by ext; exact quasiSeparated_iff _
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance : HasAffineProperty @QuasiSeparated (fun X _ _ _ ↦ QuasiSeparatedSpace X) where
  __ := HasAffineProperty.copy
    quasiSeparated_eq_diagonal_is_quasiCompact.symm
        /-
          X Y : AlgebraicGeometry.Scheme
          f : Quiver.Hom X Y
          ⊢ Eq (AlgebraicGeometry.AffineTargetMorphismProperty.diagonal fun X x x_1 x => …
        -/
    (by ext; exact quasiCompact_affineProperty_iff_quasiSeparatedSpace _)
             /-
               🎉 no goals
             -/


instance (priority := 900) quasiSeparatedOfMono {X Y : Scheme} (f : X ⟶ Y) [Mono f] :
    QuasiSeparated f where


instance quasiSeparated_isStableUnderComposition :
    MorphismProperty.IsStableUnderComposition @QuasiSeparated :=
  quasiSeparated_eq_diagonal_is_quasiCompact.symm ▸ inferInstance


instance quasiSeparated_isStableUnderBaseChange :
    MorphismProperty.IsStableUnderBaseChange @QuasiSeparated :=
  quasiSeparated_eq_diagonal_is_quasiCompact.symm ▸ inferInstance


instance quasiSeparatedComp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [QuasiSeparated f]
    [QuasiSeparated g] : QuasiSeparated (f ≫ g) :=
  MorphismProperty.comp_mem _ f g inferInstance inferInstance


theorem quasiSeparated_over_affine_iff {X Y : Scheme} (f : X ⟶ Y) [IsAffine Y] :
    QuasiSeparated f ↔ QuasiSeparatedSpace X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (AlgebraicGeometry.QuasiSeparated f) (QuasiSeparatedSpace ↑↑X.toPresheaf …
  -/
  rw [HasAffineProperty.iff_of_isAffine (P := @QuasiSeparated)]
  /-
    🎉 no goals
  -/


theorem quasiSeparatedSpace_iff_quasiSeparated (X : Scheme) :
    QuasiSeparatedSpace X ↔ QuasiSeparated (terminal.from X) :=
  (quasiSeparated_over_affine_iff _).symm


instance {X Y S : Scheme} (f : X ⟶ S) (g : Y ⟶ S) [QuasiSeparated g] :
    QuasiSeparated (pullback.fst f g) :=
  MorphismProperty.pullback_fst f g inferInstance


instance {X Y S : Scheme} (f : X ⟶ S) (g : Y ⟶ S) [QuasiSeparated f] :
    QuasiSeparated (pullback.snd f g) :=
  MorphismProperty.pullback_snd f g inferInstance


theorem quasiSeparatedSpace_of_quasiSeparated {X Y : Scheme} (f : X ⟶ Y)
    [hY : QuasiSeparatedSpace Y] [QuasiSeparated f] : QuasiSeparatedSpace X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : QuasiSeparatedSpace ↑↑Y.toPresheafedSpace
    inst✝ : AlgebraicGeometry.QuasiSeparated f
    ⊢ QuasiSeparatedSpace ↑↑X.toPresheafedSpace
  -/
  rw [quasiSeparatedSpace_iff_quasiSeparated] at hY ⊢
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : AlgebraicGeometry.QuasiSeparated (CategoryTheory.Limits.terminal.from Y)
    inst✝ : AlgebraicGeometry.QuasiSeparated f
    ⊢ AlgebraicGeometry.QuasiSeparated (CategoryTheory.Limits.terminal.from X)
  -/
  rw [← terminalIsTerminal.hom_ext (f ≫ terminal.from Y) (terminal.from X)]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hY : AlgebraicGeometry.QuasiSeparated (CategoryTheory.Limits.terminal.from Y)
    inst✝ : AlgebraicGeometry.QuasiSeparated f
    ⊢ AlgebraicGeometry.QuasiSeparated (CategoryTheory.CategoryStruct.comp f (Cate …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance quasiSeparatedSpace_of_isAffine (X : Scheme) [IsAffine X] :
    QuasiSeparatedSpace X := by
  /-
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    ⊢ QuasiSeparatedSpace ↑↑X.toPresheafedSpace
  -/
  constructor
  /-
    case inter_isCompact
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    ⊢ ∀ (U V : Set ↑↑X.toPresheafedSpace), IsOpen U → IsCompact U → IsOpen V → IsC …
  -/
  intro U V hU hU' hV hV'
  /-
    case inter_isCompact
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    ⊢ IsCompact (Inter.inter U V)
  -/
  obtain ⟨s, hs, e⟩ := (isCompactOpen_iff_eq_basicOpen_union _).mp ⟨hU', hU⟩
  /-
    case inter_isCompact.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    ⊢ IsCompact (Inter.inter U V)
  -/
  obtain ⟨s', hs', e'⟩ := (isCompactOpen_iff_eq_basicOpen_union _).mp ⟨hV', hV⟩
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    ⊢ IsCompact (Inter.inter U V)
  -/
  rw [e, e', Set.iUnion₂_inter]
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    ⊢ IsCompact (Set.iUnion fun i => Set.iUnion fun j => Inter.inter (↑(X.basicOpe …
  -/
  simp_rw [Set.inter_iUnion₂]
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    ⊢ IsCompact (Set.iUnion fun i => Set.iUnion fun x => Set.iUnion fun i_1 => Set …
  -/
  apply hs.isCompact_biUnion
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    ⊢ ∀ (i : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s i → IsCompac …
  -/
  intro i _
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    i : ↑(X.presheaf.obj { unop := Top.top })
    a✝ : Membership.mem s i
    ⊢ IsCompact (Set.iUnion fun i_1 => Set.iUnion fun j => Inter.inter ↑(X.basicOp …
  -/
  apply hs'.isCompact_biUnion
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    i : ↑(X.presheaf.obj { unop := Top.top })
    a✝ : Membership.mem s i
    ⊢ ∀ (i_1 : ↑(X.presheaf.obj { unop := Top.top })), Membership.mem s' i_1 → IsC …
  -/
  intro i' _
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    i : ↑(X.presheaf.obj { unop := Top.top })
    a✝¹ : Membership.mem s i
    i' : ↑(X.presheaf.obj { unop := Top.top })
    a✝ : Membership.mem s' i'
    ⊢ IsCompact (Inter.inter ↑(X.basicOpen i) ↑(X.basicOpen i'))
  -/
  change IsCompact (X.basicOpen i ⊓ X.basicOpen i').1
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    i : ↑(X.presheaf.obj { unop := Top.top })
    a✝¹ : Membership.mem s i
    i' : ↑(X.presheaf.obj { unop := Top.top })
    a✝ : Membership.mem s' i'
    ⊢ IsCompact (Min.min (X.basicOpen i) (X.basicOpen i')).carrier
  -/
  rw [← Scheme.basicOpen_mul]
  /-
    case inter_isCompact.intro.intro.intro.intro
    X✝ Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X✝ Y
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsAffine X
    U V : Set ↑↑X.toPresheafedSpace
    hU : IsOpen U
    hU' : IsCompact U
    hV : IsOpen V
    hV' : IsCompact V
    s : Set ↑(X.presheaf.obj { unop := Top.top })
    hs : s.Finite
    e : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    s' : Set ↑(X.presheaf.obj { unop := Top.top })
    hs' : s'.Finite
    e' : Eq V (Set.iUnion fun i => Set.iUnion fun h => ↑(X.basicOpen i))
    i : ↑(X.presheaf.obj { unop := Top.top })
    a✝¹ : Membership.mem s i
    i' : ↑(X.presheaf.obj { unop := Top.top })
    a✝ : Membership.mem s' i'
    ⊢ IsCompact (X.basicOpen (HMul.hMul i i')).carrier
  -/
  exact ((isAffineOpen_top _).basicOpen _).isCompact
  /-
    🎉 no goals
  -/


theorem IsAffineOpen.isQuasiSeparated {X : Scheme} {U : X.Opens} (hU : IsAffineOpen U) :
    IsQuasiSeparated (U : Set X) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ IsQuasiSeparated ↑U
  -/
  rw [isQuasiSeparated_iff_quasiSeparatedSpace]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ QuasiSeparatedSpace ↑↑U
  -/
  exacts [@AlgebraicGeometry.quasiSeparatedSpace_of_isAffine _ hU, U.isOpen]
  /-
    🎉 no goals
  -/


theorem QuasiSeparated.of_comp {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [QuasiSeparated (f ≫ g)] :
    QuasiSeparated f := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.QuasiSeparated (CategoryTheory.CategoryStruct.comp f …
    ⊢ AlgebraicGeometry.QuasiSeparated f
  -/
  let 𝒰 := (Z.affineCover.pullbackCover g).bind fun x => Scheme.affineCover _
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.QuasiSeparated (CategoryTheory.CategoryStruct.comp f …
    𝒰 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y := ( …
    ⊢ AlgebraicGeometry.QuasiSeparated f
  -/
  have (i) : IsAffine (𝒰.obj i) := by dsimp [𝒰]; infer_instance
  apply HasAffineProperty.of_openCover
    ((Z.affineCover.pullbackCover g).bind fun x => Scheme.affineCover _)
  /-
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : AlgebraicGeometry.QuasiSeparated (CategoryTheory.CategoryStruct.comp f …
    𝒰 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y := ( …
    this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ ∀ (i : ((AlgebraicGeometry.Scheme.Cover.pullbackCover Z.affineCover g).bind  …
  -/
  rintro ⟨i, j⟩; dsimp at i j
  refine @quasiSeparatedSpace_of_quasiSeparated _ _ ?_
    (HasAffineProperty.of_isPullback (.of_hasPullback _ (Z.affineCover.map i)) ‹_›) ?_
  · exact pullback.map _ _ _ _ (𝟙 _) _ _ (by simp) (Category.comp_id _) ≫
      (pullbackRightPullbackFstIso g (Z.affineCover.map i) f).hom
    /-
      case mk.refine_2
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝ : AlgebraicGeometry.QuasiSeparated (CategoryTheory.CategoryStruct.comp f …
      𝒰 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y := ( …
      this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      i : Z.affineCover.J
      j : (CategoryTheory.Limits.pullback g (Z.affineCover.map i)).affineCover.J
      ⊢ AlgebraicGeometry.QuasiSeparated (CategoryTheory.CategoryStruct.comp (Catego …
    -/
  · exact inferInstance
    /-
      🎉 no goals
    -/


theorem exists_eq_pow_mul_of_isAffineOpen (X : Scheme) (U : X.Opens) (hU : IsAffineOpen U)
    (f : Γ(X, U)) (x : Γ(X, X.basicOpen f)) :
    ∃ (n : ℕ) (y : Γ(X, U)), y |_ᵣ X.basicOpen f = (f |_ᵣ X.basicOpen f) ^ n * x := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    ⊢ Exists fun n => Exists fun y => Eq (TopCat.Presheaf.restrictOpenCommRingCat  …
  -/
  have := (hU.isLocalization_basicOpen f).2
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    this : ∀ (z : ↑(X.presheaf.obj { unop := X.basicOpen f })), Exists fun x => Eq …
    ⊢ Exists fun n => Exists fun y => Eq (TopCat.Presheaf.restrictOpenCommRingCat  …
  -/
  obtain ⟨⟨y, _, n, rfl⟩, d⟩ := this x
  /-
    case intro.mk.mk.intro
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    this : ∀ (z : ↑(X.presheaf.obj { unop := X.basicOpen f })), Exists fun x => Eq …
    y : ↑(X.presheaf.obj { unop := U })
    n : Nat
    d : Eq (HMul.hMul x ((algebraMap ↑(X.presheaf.obj { unop := U }) ↑(X.presheaf. …
    ⊢ Exists fun n => Exists fun y => Eq (TopCat.Presheaf.restrictOpenCommRingCat  …
  -/
  use n, y
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    this : ∀ (z : ↑(X.presheaf.obj { unop := X.basicOpen f })), Exists fun x => Eq …
    y : ↑(X.presheaf.obj { unop := U })
    n : Nat
    d : Eq (HMul.hMul x ((algebraMap ↑(X.presheaf.obj { unop := U }) ↑(X.presheaf. …
    ⊢ Eq (TopCat.Presheaf.restrictOpenCommRingCat y (X.basicOpen f) ⋯) (HMul.hMul  …
  -/
  dsimp only [TopCat.Presheaf.restrictOpenCommRingCat_apply]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    this : ∀ (z : ↑(X.presheaf.obj { unop := X.basicOpen f })), Exists fun x => Eq …
    y : ↑(X.presheaf.obj { unop := U })
    n : Nat
    d : Eq (HMul.hMul x ((algebraMap ↑(X.presheaf.obj { unop := U }) ↑(X.presheaf. …
    ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y) (HMul.hMul (HPow.h …
  -/
  simpa [mul_comm x] using d.symm
  /-
    🎉 no goals
  -/


theorem exists_eq_pow_mul_of_is_compact_of_quasi_separated_space_aux_aux {X : TopCat}
    (F : X.Presheaf CommRingCat) {U₁ U₂ U₃ U₄ U₅ U₆ U₇ : Opens X} {n₁ n₂ : ℕ}
    {y₁ : F.obj (op U₁)} {y₂ : F.obj (op U₂)} {f : F.obj (op <| U₁ ⊔ U₂)}
    {x : F.obj (op U₃)} (h₄₁ : U₄ ≤ U₁) (h₄₂ : U₄ ≤ U₂) (h₅₁ : U₅ ≤ U₁) (h₅₃ : U₅ ≤ U₃)
    (h₆₂ : U₆ ≤ U₂) (h₆₃ : U₆ ≤ U₃) (h₇₄ : U₇ ≤ U₄) (h₇₅ : U₇ ≤ U₅) (h₇₆ : U₇ ≤ U₆)
    (e₁ : y₁ |_ᵣ U₅ = (f |_ᵣ U₁ |_ᵣ U₅) ^ n₁ * x |_ᵣ U₅)
    (e₂ : y₂ |_ᵣ U₆ = (f |_ᵣ U₂ |_ᵣ U₆) ^ n₂ * x |_ᵣ U₆) :
    (((f |_ᵣ U₁) ^ n₂ * y₁) |_ᵣ U₄) |_ᵣ U₇ = (((f |_ᵣ U₂) ^ n₁ * y₂) |_ᵣ U₄) |_ᵣ U₇ := by
  /-
    X : TopCat
    F : TopCat.Presheaf CommRingCat X
    U₁ U₂ U₃ U₄ U₅ U₆ U₇ : TopologicalSpace.Opens ↑X
    n₁ n₂ : Nat
    y₁ : ↑(F.obj { unop := U₁ })
    y₂ : ↑(F.obj { unop := U₂ })
    f : ↑(F.obj { unop := Max.max U₁ U₂ })
    x : ↑(F.obj { unop := U₃ })
    h₄₁ : LE.le U₄ U₁
    h₄₂ : LE.le U₄ U₂
    h₅₁ : LE.le U₅ U₁
    h₅₃ : LE.le U₅ U₃
    h₆₂ : LE.le U₆ U₂
    h₆₃ : LE.le U₆ U₃
    h₇₄ : LE.le U₇ U₄
    h₇₅ : LE.le U₇ U₅
    h₇₆ : LE.le U₇ U₆
    e₁ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₁ U₅ h₅₁) (HMul.hMul (HPow.h …
    e₂ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₂ U₆ h₆₂) (HMul.hMul (HPow.h …
    ⊢ Eq (TopCat.Presheaf.restrictOpenCommRingCat (TopCat.Presheaf.restrictOpenCom …
  -/
  apply_fun (fun x : F.obj (op U₅) ↦ x |_ᵣ U₇) at e₁
  /-
    X : TopCat
    F : TopCat.Presheaf CommRingCat X
    U₁ U₂ U₃ U₄ U₅ U₆ U₇ : TopologicalSpace.Opens ↑X
    n₁ n₂ : Nat
    y₁ : ↑(F.obj { unop := U₁ })
    y₂ : ↑(F.obj { unop := U₂ })
    f : ↑(F.obj { unop := Max.max U₁ U₂ })
    x : ↑(F.obj { unop := U₃ })
    h₄₁ : LE.le U₄ U₁
    h₄₂ : LE.le U₄ U₂
    h₅₁ : LE.le U₅ U₁
    h₅₃ : LE.le U₅ U₃
    h₆₂ : LE.le U₆ U₂
    h₆₃ : LE.le U₆ U₃
    h₇₄ : LE.le U₇ U₄
    h₇₅ : LE.le U₇ U₅
    h₇₆ : LE.le U₇ U₆
    e₂ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₂ U₆ h₆₂) (HMul.hMul (HPow.h …
    e₁ : Eq (TopCat.Presheaf.restrictOpenCommRingCat (TopCat.Presheaf.restrictOpen …
    ⊢ Eq (TopCat.Presheaf.restrictOpenCommRingCat (TopCat.Presheaf.restrictOpenCom …
  -/
  apply_fun (fun x : F.obj (op U₆) ↦ x |_ᵣ U₇) at e₂
  /-
    X : TopCat
    F : TopCat.Presheaf CommRingCat X
    U₁ U₂ U₃ U₄ U₅ U₆ U₇ : TopologicalSpace.Opens ↑X
    n₁ n₂ : Nat
    y₁ : ↑(F.obj { unop := U₁ })
    y₂ : ↑(F.obj { unop := U₂ })
    f : ↑(F.obj { unop := Max.max U₁ U₂ })
    x : ↑(F.obj { unop := U₃ })
    h₄₁ : LE.le U₄ U₁
    h₄₂ : LE.le U₄ U₂
    h₅₁ : LE.le U₅ U₁
    h₅₃ : LE.le U₅ U₃
    h₆₂ : LE.le U₆ U₂
    h₆₃ : LE.le U₆ U₃
    h₇₄ : LE.le U₇ U₄
    h₇₅ : LE.le U₇ U₅
    h₇₆ : LE.le U₇ U₆
    e₁ : Eq (TopCat.Presheaf.restrictOpenCommRingCat (TopCat.Presheaf.restrictOpen …
    e₂ : Eq (TopCat.Presheaf.restrictOpenCommRingCat (TopCat.Presheaf.restrictOpen …
    ⊢ Eq (TopCat.Presheaf.restrictOpenCommRingCat (TopCat.Presheaf.restrictOpenCom …
  -/
  dsimp only [TopCat.Presheaf.restrictOpenCommRingCat_apply] at e₁ e₂ ⊢
  simp only [map_mul, map_pow, ← op_comp, ← F.map_comp, homOfLE_comp, ← CommRingCat.comp_apply]
    at e₁ e₂ ⊢
  /-
    X : TopCat
    F : TopCat.Presheaf CommRingCat X
    U₁ U₂ U₃ U₄ U₅ U₆ U₇ : TopologicalSpace.Opens ↑X
    n₁ n₂ : Nat
    y₁ : ↑(F.obj { unop := U₁ })
    y₂ : ↑(F.obj { unop := U₂ })
    f : ↑(F.obj { unop := Max.max U₁ U₂ })
    x : ↑(F.obj { unop := U₃ })
    h₄₁ : LE.le U₄ U₁
    h₄₂ : LE.le U₄ U₂
    h₅₁ : LE.le U₅ U₁
    h₅₃ : LE.le U₅ U₃
    h₆₂ : LE.le U₆ U₂
    h₆₃ : LE.le U₆ U₃
    h₇₄ : LE.le U₇ U₄
    h₇₅ : LE.le U₇ U₅
    h₇₆ : LE.le U₇ U₆
    e₁ : Eq ((F.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (HPow.hPow ( …
    e₂ : Eq ((F.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (HPow.hPow ( …
    ⊢ Eq (HMul.hMul (HPow.hPow ((F.map (CategoryTheory.homOfLE ⋯).op).hom f) n₂) ( …
  -/
  rw [e₁, e₂, mul_left_comm]
  /-
    🎉 no goals
  -/


theorem exists_eq_pow_mul_of_is_compact_of_quasi_separated_space_aux (X : Scheme)
    (S : X.affineOpens) (U₁ U₂ : X.Opens) {n₁ n₂ : ℕ} {y₁ : Γ(X, U₁)}
    {y₂ : Γ(X, U₂)} {f : Γ(X, U₁ ⊔ U₂)}
    {x : Γ(X, X.basicOpen f)} (h₁ : S.1 ≤ U₁) (h₂ : S.1 ≤ U₂)
    (e₁ : y₁ |_ᵣ X.basicOpen (f |_ᵣ U₁) =
      ((f |_ᵣ U₁ |_ᵣ X.basicOpen _) ^ n₁) * x |_ᵣ X.basicOpen _)
    (e₂ : y₂ |_ᵣ X.basicOpen (f |_ᵣ U₂) =
      ((f |_ᵣ U₂ |_ᵣ X.basicOpen _) ^ n₂) * x |_ᵣ X.basicOpen _) :
    ∃ n : ℕ, ∀ m, n ≤ m →
      ((f |_ᵣ U₁) ^ (m + n₂) * y₁) |_ᵣ S.1 = ((f |_ᵣ U₂) ^ (m + n₁) * y₂) |_ᵣ S.1 := by
  obtain ⟨⟨_, n, rfl⟩, e⟩ :=
    (@IsLocalization.eq_iff_exists _ _ _ _ _ _
      (S.2.isLocalization_basicOpen (f |_ᵣ S.1))
        (((f |_ᵣ U₁) ^ n₂ * y₁) |_ᵣ S.1)
        (((f |_ᵣ U₂) ^ n₁ * y₂) |_ᵣ S.1)).mp <| by
    apply exists_eq_pow_mul_of_is_compact_of_quasi_separated_space_aux_aux (e₁ := e₁) (e₂ := e₂)
    · show X.basicOpen _ ≤ _
      simp only [TopCat.Presheaf.restrictOpenCommRingCat_apply, Scheme.basicOpen_res]
      exact inf_le_inf h₁ le_rfl
    · show X.basicOpen _ ≤ _
      simp only [TopCat.Presheaf.restrictOpenCommRingCat_apply, Scheme.basicOpen_res]
      exact inf_le_inf h₂ le_rfl
  /-
    case intro.mk.intro
    X : AlgebraicGeometry.Scheme
    S : ↑X.affineOpens
    U₁ U₂ : X.Opens
    n₁ n₂ : Nat
    y₁ : ↑(X.presheaf.obj { unop := U₁ })
    y₂ : ↑(X.presheaf.obj { unop := U₂ })
    f : ↑(X.presheaf.obj { unop := Max.max U₁ U₂ })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    h₁ : LE.le (↑S) U₁
    h₂ : LE.le (↑S) U₂
    e₁ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₁ (X.basicOpen (TopCat.Presh …
    e₂ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₂ (X.basicOpen (TopCat.Presh …
    n : Nat
    e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow (TopCat.Presheaf.restrictOpenCommRing …
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (TopCat.Presheaf.restrictOpenCom …
  -/
  use n
  /-
    case h
    X : AlgebraicGeometry.Scheme
    S : ↑X.affineOpens
    U₁ U₂ : X.Opens
    n₁ n₂ : Nat
    y₁ : ↑(X.presheaf.obj { unop := U₁ })
    y₂ : ↑(X.presheaf.obj { unop := U₂ })
    f : ↑(X.presheaf.obj { unop := Max.max U₁ U₂ })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    h₁ : LE.le (↑S) U₁
    h₂ : LE.le (↑S) U₂
    e₁ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₁ (X.basicOpen (TopCat.Presh …
    e₂ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₂ (X.basicOpen (TopCat.Presh …
    n : Nat
    e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow (TopCat.Presheaf.restrictOpenCommRing …
    ⊢ ∀ (m : Nat), LE.le n m → Eq (TopCat.Presheaf.restrictOpenCommRingCat (HMul.h …
  -/
  intros m hm
  /-
    case h
    X : AlgebraicGeometry.Scheme
    S : ↑X.affineOpens
    U₁ U₂ : X.Opens
    n₁ n₂ : Nat
    y₁ : ↑(X.presheaf.obj { unop := U₁ })
    y₂ : ↑(X.presheaf.obj { unop := U₂ })
    f : ↑(X.presheaf.obj { unop := Max.max U₁ U₂ })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    h₁ : LE.le (↑S) U₁
    h₂ : LE.le (↑S) U₂
    e₁ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₁ (X.basicOpen (TopCat.Presh …
    e₂ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₂ (X.basicOpen (TopCat.Presh …
    n : Nat
    e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow (TopCat.Presheaf.restrictOpenCommRing …
    m : Nat
    hm : LE.le n m
    ⊢ Eq (TopCat.Presheaf.restrictOpenCommRingCat (HMul.hMul (HPow.hPow (TopCat.Pr …
  -/
  rw [← tsub_add_cancel_of_le hm]
  simp only [TopCat.Presheaf.restrictOpenCommRingCat_apply,
    pow_add, map_pow, map_mul, mul_assoc, ← Functor.map_comp, ← op_comp, homOfLE_comp,
    Subtype.coe_mk, ← CommRingCat.comp_apply] at e ⊢
  /-
    case h
    X : AlgebraicGeometry.Scheme
    S : ↑X.affineOpens
    U₁ U₂ : X.Opens
    n₁ n₂ : Nat
    y₁ : ↑(X.presheaf.obj { unop := U₁ })
    y₂ : ↑(X.presheaf.obj { unop := U₂ })
    f : ↑(X.presheaf.obj { unop := Max.max U₁ U₂ })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    h₁ : LE.le (↑S) U₁
    h₂ : LE.le (↑S) U₂
    e₁ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₁ (X.basicOpen (TopCat.Presh …
    e₂ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₂ (X.basicOpen (TopCat.Presh …
    n m : Nat
    hm : LE.le n m
    e : Eq (HMul.hMul (HPow.hPow ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).h …
    ⊢ Eq (HMul.hMul (HPow.hPow ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom …
  -/
  rw [e]
  /-
    🎉 no goals
  -/


theorem exists_eq_pow_mul_of_isCompact_of_isQuasiSeparated (X : Scheme.{u}) (U : X.Opens)
    (hU : IsCompact U.1) (hU' : IsQuasiSeparated U.1) (f : Γ(X, U)) (x : Γ(X, X.basicOpen f)) :
    ∃ (n : ℕ) (y : Γ(X, U)), y |_ᵣ X.basicOpen f = (f |_ᵣ X.basicOpen f) ^ n * x := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    hU' : IsQuasiSeparated U.carrier
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    ⊢ Exists fun n => Exists fun y => Eq (TopCat.Presheaf.restrictOpenCommRingCat  …
  -/
  dsimp only [TopCat.Presheaf.restrictOpenCommRingCat_apply]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    hU' : IsQuasiSeparated U.carrier
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑(X.presheaf.obj { unop := X.basicOpen f })
    ⊢ Exists fun n => Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE  …
  -/
  revert hU' f x
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    ⊢ IsQuasiSeparated U.carrier → ∀ (f : ↑(X.presheaf.obj { unop := U })) (x : ↑( …
  -/
  refine compact_open_induction_on U hU ?_ ?_
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      ⊢ IsQuasiSeparated Bot.bot.carrier → ∀ (f : ↑(X.presheaf.obj { unop := Bot.bot …
    -/
  · intro _ f x
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU'✝ : IsQuasiSeparated Bot.bot.carrier
      f : ↑(X.presheaf.obj { unop := Bot.bot })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      ⊢ Exists fun n => Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE  …
    -/
    use 0, f
    refine @Subsingleton.elim _
      (CommRingCat.subsingleton_of_isTerminal (X.sheaf.isTerminalOfEqEmpty ?_)) _ _
    /-
      case h
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU'✝ : IsQuasiSeparated Bot.bot.carrier
      f : ↑(X.presheaf.obj { unop := Bot.bot })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      ⊢ Eq (X.basicOpen f) Bot.bot
    -/
    rw [eq_bot_iff]
    /-
      case h
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU'✝ : IsQuasiSeparated Bot.bot.carrier
      f : ↑(X.presheaf.obj { unop := Bot.bot })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      ⊢ LE.le (X.basicOpen f) Bot.bot
    -/
    exact X.basicOpen_le f
    /-
      🎉 no goals
    -/
  · -- Given `f : 𝒪(S ∪ U), x : 𝒪(X_f)`, we need to show that `f ^ n * x` is the restriction of
    -- some `y : 𝒪(S ∪ U)` for some `n : ℕ`.
    /-
      case refine_2
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      ⊢ ∀ (S : X.Opens), IsCompact S.carrier → ∀ (U : ↑X.affineOpens), (IsQuasiSepar …
    -/
    intro S hS U hU hSU f x
    -- We know that such `y₁, n₁` exists on `S` by the induction hypothesis.
    obtain ⟨n₁, y₁, hy₁⟩ :=
      hU (hSU.of_subset Set.subset_union_left) (X.presheaf.map (homOfLE le_sup_left).op f)
        (X.presheaf.map (homOfLE _).op x)
    -- · rw [X.basicOpen_res]; exact inf_le_right
    -- We know that such `y₂, n₂` exists on `U` since `U` is affine.
    obtain ⟨n₂, y₂, hy₂⟩ :=
      exists_eq_pow_mul_of_isAffineOpen X _ U.2 (X.presheaf.map (homOfLE le_sup_right).op f)
        (X.presheaf.map (homOfLE _).op x)
    /-
      case refine_2.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U✝ : X.Opens
      hU✝ : IsCompact U✝.carrier
      S : X.Opens
      hS : IsCompact S.carrier
      U : ↑X.affineOpens
      hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
      hSU : IsQuasiSeparated (Max.max S ↑U).carrier
      f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n₁ : Nat
      y₁ : ↑(X.presheaf.obj { unop := S })
      hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
      n₂ : Nat
      y₂ : ↑(X.presheaf.obj { unop := ↑U })
      hy₂ : Eq (TopCat.Presheaf.restrictOpenCommRingCat y₂ (X.basicOpen ((X.presheaf …
      ⊢ Exists fun n => Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE  …
    -/
    dsimp only [TopCat.Presheaf.restrictOpenCommRingCat_apply] at hy₂
    -- swap; · rw [X.basicOpen_res]; exact inf_le_right
    -- Since `S ∪ U` is quasi-separated, `S ∩ U` can be covered by finite affine opens.
    obtain ⟨s, hs', hs⟩ :=
      (isCompactOpen_iff_eq_finset_affine_union _).mp
        ⟨hSU _ _ Set.subset_union_left S.2 hS Set.subset_union_right U.1.2
            U.2.isCompact,
          (S ⊓ U.1).2⟩
    /-
      case refine_2.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U✝ : X.Opens
      hU✝ : IsCompact U✝.carrier
      S : X.Opens
      hS : IsCompact S.carrier
      U : ↑X.affineOpens
      hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
      hSU : IsQuasiSeparated (Max.max S ↑U).carrier
      f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n₁ : Nat
      y₁ : ↑(X.presheaf.obj { unop := S })
      hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
      n₂ : Nat
      y₂ : ↑(X.presheaf.obj { unop := ↑U })
      hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
      s : Set ↑X.affineOpens
      hs' : s.Finite
      hs : Eq (Inter.inter ↑S ↑↑U) (Set.iUnion fun i => Set.iUnion fun h => ↑↑i)
      ⊢ Exists fun n => Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE  …
    -/
    haveI := hs'.to_subtype
    /-
      case refine_2.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U✝ : X.Opens
      hU✝ : IsCompact U✝.carrier
      S : X.Opens
      hS : IsCompact S.carrier
      U : ↑X.affineOpens
      hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
      hSU : IsQuasiSeparated (Max.max S ↑U).carrier
      f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n₁ : Nat
      y₁ : ↑(X.presheaf.obj { unop := S })
      hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
      n₂ : Nat
      y₂ : ↑(X.presheaf.obj { unop := ↑U })
      hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
      s : Set ↑X.affineOpens
      hs' : s.Finite
      hs : Eq (Inter.inter ↑S ↑↑U) (Set.iUnion fun i => Set.iUnion fun h => ↑↑i)
      this : Finite ↑s
      ⊢ Exists fun n => Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE  …
    -/
    cases nonempty_fintype s
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U✝ : X.Opens
      hU✝ : IsCompact U✝.carrier
      S : X.Opens
      hS : IsCompact S.carrier
      U : ↑X.affineOpens
      hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
      hSU : IsQuasiSeparated (Max.max S ↑U).carrier
      f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n₁ : Nat
      y₁ : ↑(X.presheaf.obj { unop := S })
      hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
      n₂ : Nat
      y₂ : ↑(X.presheaf.obj { unop := ↑U })
      hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
      s : Set ↑X.affineOpens
      hs' : s.Finite
      hs : Eq (Inter.inter ↑S ↑↑U) (Set.iUnion fun i => Set.iUnion fun h => ↑↑i)
      this : Finite ↑s
      val✝ : Fintype ↑s
      ⊢ Exists fun n => Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE  …
    -/
    replace hs : S ⊓ U.1 = iSup fun i : s => (i : X.Opens) := by ext1; simpa using hs
    have hs₁ : ∀ i : s, i.1.1 ≤ S := by
      intro i; change (i : X.Opens) ≤ S
      refine le_trans ?_ (inf_le_left (b := U.1))
      rw [hs]
      -- Porting note: have to add argument explicitly
      exact @le_iSup X.Opens s _ (fun (i : s) => (i : X.Opens)) i
    have hs₂ : ∀ i : s, i.1.1 ≤ U.1 := by
      intro i; change (i : X.Opens) ≤ U
      refine le_trans ?_ (inf_le_right (a := S))
      rw [hs]
      -- Porting note: have to add argument explicitly
      exact @le_iSup X.Opens s _ (fun (i : s) => (i : X.Opens)) i
    -- On each affine open in the intersection, we have `f ^ (n + n₂) * y₁ = f ^ (n + n₁) * y₂`
    -- for some `n` since `f ^ n₂ * y₁ = f ^ (n₁ + n₂) * x = f ^ n₁ * y₂` on `X_f`.
    have := fun i ↦ exists_eq_pow_mul_of_is_compact_of_quasi_separated_space_aux
      X i.1 S U (hs₁ i) (hs₂ i) hy₁ hy₂
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U✝ : X.Opens
      hU✝ : IsCompact U✝.carrier
      S : X.Opens
      hS : IsCompact S.carrier
      U : ↑X.affineOpens
      hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
      hSU : IsQuasiSeparated (Max.max S ↑U).carrier
      f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n₁ : Nat
      y₁ : ↑(X.presheaf.obj { unop := S })
      hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
      n₂ : Nat
      y₂ : ↑(X.presheaf.obj { unop := ↑U })
      hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
      s : Set ↑X.affineOpens
      hs' : s.Finite
      this✝ : Finite ↑s
      val✝ : Fintype ↑s
      hs : Eq (Min.min S ↑U) (iSup fun i => ↑↑i)
      hs₁ : ∀ (i : ↑s), LE.le (↑↑i) S
      hs₂ : ∀ (i : ↑s), LE.le ↑↑i ↑U
      this : ∀ (i : ↑s), Exists fun n => ∀ (m : Nat), LE.le n m → Eq (TopCat.Preshea …
      ⊢ Exists fun n => Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE  …
    -/
    choose n hn using this
    -- We can thus choose a big enough `n` such that `f ^ (n + n₂) * y₁ = f ^ (n + n₁) * y₂`
    -- on `S ∩ U`.
    have :
      X.presheaf.map (homOfLE <| inf_le_left).op
          (X.presheaf.map (homOfLE le_sup_left).op f ^ (Finset.univ.sup n + n₂) * y₁) =
        X.presheaf.map (homOfLE <| inf_le_right).op
          (X.presheaf.map (homOfLE le_sup_right).op f ^ (Finset.univ.sup n + n₁) * y₂) := by
      fapply X.sheaf.eq_of_locally_eq' fun i : s => i.1.1
      · refine fun i => homOfLE ?_; rw [hs]
        -- Porting note: have to add argument explicitly
        exact @le_iSup X.Opens s _ (fun (i : s) => (i : X.Opens)) i
      · exact le_of_eq hs
      · intro i
        -- This unfolds `X.sheaf` and ensures we use `CommRingCat.hom` to apply the morphism
        show (X.presheaf.map _) _ = (X.presheaf.map _) _
        simp only [← CommRingCat.comp_apply, ← Functor.map_comp, ← op_comp]
        apply hn
        exact Finset.le_sup (Finset.mem_univ _)
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U✝ : X.Opens
      hU✝ : IsCompact U✝.carrier
      S : X.Opens
      hS : IsCompact S.carrier
      U : ↑X.affineOpens
      hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
      hSU : IsQuasiSeparated (Max.max S ↑U).carrier
      f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n₁ : Nat
      y₁ : ↑(X.presheaf.obj { unop := S })
      hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
      n₂ : Nat
      y₂ : ↑(X.presheaf.obj { unop := ↑U })
      hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
      s : Set ↑X.affineOpens
      hs' : s.Finite
      this✝ : Finite ↑s
      val✝ : Fintype ↑s
      hs : Eq (Min.min S ↑U) (iSup fun i => ↑↑i)
      hs₁ : ∀ (i : ↑s), LE.le (↑↑i) S
      hs₂ : ∀ (i : ↑s), LE.le ↑↑i ↑U
      n : ↑s → Nat
      hn : ∀ (i : ↑s) (m : Nat), LE.le (n i) m → Eq (TopCat.Presheaf.restrictOpenCom …
      this : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow …
      ⊢ Exists fun n => Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE  …
    -/
    use Finset.univ.sup n + n₁ + n₂
    -- By the sheaf condition, since `f ^ (n + n₂) * y₁ = f ^ (n + n₁) * y₂`, it can be glued into
    -- the desired section on `S ∪ U`.
    /-
      case h
      X : AlgebraicGeometry.Scheme
      U✝ : X.Opens
      hU✝ : IsCompact U✝.carrier
      S : X.Opens
      hS : IsCompact S.carrier
      U : ↑X.affineOpens
      hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
      hSU : IsQuasiSeparated (Max.max S ↑U).carrier
      f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
      x : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n₁ : Nat
      y₁ : ↑(X.presheaf.obj { unop := S })
      hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
      n₂ : Nat
      y₂ : ↑(X.presheaf.obj { unop := ↑U })
      hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
      s : Set ↑X.affineOpens
      hs' : s.Finite
      this✝ : Finite ↑s
      val✝ : Fintype ↑s
      hs : Eq (Min.min S ↑U) (iSup fun i => ↑↑i)
      hs₁ : ∀ (i : ↑s), LE.le (↑↑i) S
      hs₂ : ∀ (i : ↑s), LE.le ↑↑i ↑U
      n : ↑s → Nat
      hn : ∀ (i : ↑s) (m : Nat), LE.le (n i) m → Eq (TopCat.Presheaf.restrictOpenCom …
      this : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow …
      ⊢ Exists fun y => Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y) (H …
    -/
    use (X.sheaf.objSupIsoProdEqLocus S U.1).inv ⟨⟨_ * _, _ * _⟩, this⟩
    refine (X.sheaf.objSupIsoProdEqLocus_inv_eq_iff _ _ _ (X.basicOpen_res _
      (homOfLE le_sup_left).op) (X.basicOpen_res _ (homOfLE le_sup_right).op)).mpr ⟨?_, ?_⟩
    · -- This unfolds `X.sheaf` and ensures we use `CommRingCat.hom` to apply the morphism
      /-
        case h.refine_1
        X : AlgebraicGeometry.Scheme
        U✝ : X.Opens
        hU✝ : IsCompact U✝.carrier
        S : X.Opens
        hS : IsCompact S.carrier
        U : ↑X.affineOpens
        hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
        hSU : IsQuasiSeparated (Max.max S ↑U).carrier
        f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
        x : ↑(X.presheaf.obj { unop := X.basicOpen f })
        n₁ : Nat
        y₁ : ↑(X.presheaf.obj { unop := S })
        hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
        n₂ : Nat
        y₂ : ↑(X.presheaf.obj { unop := ↑U })
        hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
        s : Set ↑X.affineOpens
        hs' : s.Finite
        this✝ : Finite ↑s
        val✝ : Fintype ↑s
        hs : Eq (Min.min S ↑U) (iSup fun i => ↑↑i)
        hs₁ : ∀ (i : ↑s), LE.le (↑↑i) S
        hs₂ : ∀ (i : ↑s), LE.le ↑↑i ↑U
        n : ↑s → Nat
        hn : ∀ (i : ↑s) (m : Nat), LE.le (n i) m → Eq (TopCat.Presheaf.restrictOpenCom …
        this : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow …
        ⊢ Eq ((X.sheaf.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑⟨{ fst := HMul.hMu …
      -/
      show (X.presheaf.map _) _ = (X.presheaf.map _) _
      /-
        case h.refine_1
        X : AlgebraicGeometry.Scheme
        U✝ : X.Opens
        hU✝ : IsCompact U✝.carrier
        S : X.Opens
        hS : IsCompact S.carrier
        U : ↑X.affineOpens
        hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
        hSU : IsQuasiSeparated (Max.max S ↑U).carrier
        f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
        x : ↑(X.presheaf.obj { unop := X.basicOpen f })
        n₁ : Nat
        y₁ : ↑(X.presheaf.obj { unop := S })
        hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
        n₂ : Nat
        y₂ : ↑(X.presheaf.obj { unop := ↑U })
        hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
        s : Set ↑X.affineOpens
        hs' : s.Finite
        this✝ : Finite ↑s
        val✝ : Fintype ↑s
        hs : Eq (Min.min S ↑U) (iSup fun i => ↑↑i)
        hs₁ : ∀ (i : ↑s), LE.le (↑↑i) S
        hs₂ : ∀ (i : ↑s), LE.le ↑↑i ↑U
        n : ↑s → Nat
        hn : ∀ (i : ↑s) (m : Nat), LE.le (n i) m → Eq (TopCat.Presheaf.restrictOpenCom …
        this : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow …
        ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (↑⟨{ fst := HMul.hMul …
      -/
      rw [add_assoc, add_comm n₁]
      simp only [pow_add, map_pow, map_mul, hy₁, ← CommRingCat.comp_apply, ← mul_assoc,
        ← Functor.map_comp, ← op_comp, homOfLE_comp]
    · -- This unfolds `X.sheaf` and ensures we use `CommRingCat.hom` to apply the morphism
      /-
        case h.refine_2
        X : AlgebraicGeometry.Scheme
        U✝ : X.Opens
        hU✝ : IsCompact U✝.carrier
        S : X.Opens
        hS : IsCompact S.carrier
        U : ↑X.affineOpens
        hU : IsQuasiSeparated S.carrier → ∀ (f : ↑(X.presheaf.obj { unop := S })) (x : …
        hSU : IsQuasiSeparated (Max.max S ↑U).carrier
        f : ↑(X.presheaf.obj { unop := Max.max S ↑U })
        x : ↑(X.presheaf.obj { unop := X.basicOpen f })
        n₁ : Nat
        y₁ : ↑(X.presheaf.obj { unop := S })
        hy₁ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₁) (HMul.hMul (H …
        n₂ : Nat
        y₂ : ↑(X.presheaf.obj { unop := ↑U })
        hy₂ : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom y₂) (HMul.hMul (H …
        s : Set ↑X.affineOpens
        hs' : s.Finite
        this✝ : Finite ↑s
        val✝ : Fintype ↑s
        hs : Eq (Min.min S ↑U) (iSup fun i => ↑↑i)
        hs₁ : ∀ (i : ↑s), LE.le (↑↑i) S
        hs₂ : ∀ (i : ↑s), LE.le ↑↑i ↑U
        n : ↑s → Nat
        hn : ∀ (i : ↑s) (m : Nat), LE.le (n i) m → Eq (TopCat.Presheaf.restrictOpenCom …
        this : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HMul.hMul (HPow …
        ⊢ Eq ((X.sheaf.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑⟨{ fst := HMul.hMu …
      -/
      show (X.presheaf.map _) _ = (X.presheaf.map _) _
      simp only [pow_add, map_pow, map_mul, hy₂, ← CommRingCat.comp_apply, ← mul_assoc,
        ← Functor.map_comp, ← op_comp, homOfLE_comp]


/-- If `U` is qcqs, then `Γ(X, D(f)) ≃ Γ(X, U)_f` for every `f : Γ(X, U)`.
This is known as the **Qcqs lemma** in [R. Vakil, *The rising sea*][RisingSea]. -/
theorem is_localization_basicOpen_of_qcqs {X : Scheme} {U : X.Opens} (hU : IsCompact U.1)
    (hU' : IsQuasiSeparated U.1) (f : Γ(X, U)) :
    IsLocalization.Away f (Γ(X, X.basicOpen f)) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : IsCompact U.carrier
    hU' : IsQuasiSeparated U.carrier
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ IsLocalization.Away f ↑(X.presheaf.obj { unop := X.basicOpen f })
  -/
  constructor
    /-
      case map_units'
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f : ↑(X.presheaf.obj { unop := U })
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Submonoid.powers f) x), IsUnit ((alg …
    -/
  · rintro ⟨_, n, rfl⟩
    /-
      case map_units'.mk.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f : ↑(X.presheaf.obj { unop := U })
      n : Nat
      ⊢ IsUnit ((algebraMap ↑(X.presheaf.obj { unop := U }) ↑(X.presheaf.obj { unop  …
    -/
    simp only [map_pow, Subtype.coe_mk, RingHom.algebraMap_toAlgebra]
    /-
      case map_units'.mk.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f : ↑(X.presheaf.obj { unop := U })
      n : Nat
      ⊢ IsUnit (HPow.hPow ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom f) n)
    -/
    exact IsUnit.pow _ (RingedSpace.isUnit_res_basicOpen _ f)
    /-
      🎉 no goals
    -/
    /-
      case surj'
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f : ↑(X.presheaf.obj { unop := U })
      ⊢ ∀ (z : ↑(X.presheaf.obj { unop := X.basicOpen f })), Exists fun x => Eq (HMu …
    -/
  · intro z
    /-
      case surj'
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f : ↑(X.presheaf.obj { unop := U })
      z : ↑(X.presheaf.obj { unop := X.basicOpen f })
      ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap ↑(X.presheaf.obj { unop := U }) …
    -/
    obtain ⟨n, y, e⟩ := exists_eq_pow_mul_of_isCompact_of_isQuasiSeparated X U hU hU' f z
    /-
      case surj'.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f : ↑(X.presheaf.obj { unop := U })
      z : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n : Nat
      y : ↑(X.presheaf.obj { unop := U })
      e : Eq (TopCat.Presheaf.restrictOpenCommRingCat y (X.basicOpen f) ⋯) (HMul.hMu …
      ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap ↑(X.presheaf.obj { unop := U }) …
    -/
    refine ⟨⟨y, _, n, rfl⟩, ?_⟩
    /-
      case surj'.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f : ↑(X.presheaf.obj { unop := U })
      z : ↑(X.presheaf.obj { unop := X.basicOpen f })
      n : Nat
      y : ↑(X.presheaf.obj { unop := U })
      e : Eq (TopCat.Presheaf.restrictOpenCommRingCat y (X.basicOpen f) ⋯) (HMul.hMu …
      ⊢ Eq (HMul.hMul z ((algebraMap ↑(X.presheaf.obj { unop := U }) ↑(X.presheaf.ob …
    -/
    simpa only [map_pow, Subtype.coe_mk, RingHom.algebraMap_toAlgebra, mul_comm z] using e.symm
    /-
      🎉 no goals
    -/
    /-
      case exists_of_eq
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f : ↑(X.presheaf.obj { unop := U })
      ⊢ ∀ {x y : ↑(X.presheaf.obj { unop := U })}, Eq ((algebraMap ↑(X.presheaf.obj  …
    -/
  · intro x y
    /-
      case exists_of_eq
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f x y : ↑(X.presheaf.obj { unop := U })
      ⊢ Eq ((algebraMap ↑(X.presheaf.obj { unop := U }) ↑(X.presheaf.obj { unop := X …
    -/
    rw [← sub_eq_zero, ← map_sub, RingHom.algebraMap_toAlgebra]
    /-
      case exists_of_eq
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f x y : ↑(X.presheaf.obj { unop := U })
      ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HSub.hSub x y)) 0 →  …
    -/
    simp_rw [← @sub_eq_zero _ _ (_ * x) (_ * y), ← mul_sub]
    /-
      case exists_of_eq
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f x y : ↑(X.presheaf.obj { unop := U })
      ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom (HSub.hSub x y)) 0 →  …
    -/
    generalize x - y = z
    /-
      case exists_of_eq
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f x y z : ↑(X.presheaf.obj { unop := U })
      ⊢ Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom z) 0 → Exists fun c = …
    -/
    intro H
    /-
      case exists_of_eq
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f x y z : ↑(X.presheaf.obj { unop := U })
      H : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom z) 0
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) z) 0
    -/
    obtain ⟨n, e⟩ := exists_pow_mul_eq_zero_of_res_basicOpen_eq_zero_of_isCompact X hU _ _ H
    /-
      case exists_of_eq.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f x y z : ↑(X.presheaf.obj { unop := U })
      H : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom z) 0
      n : Nat
      e : Eq (HMul.hMul (HPow.hPow f n) z) 0
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) z) 0
    -/
    refine ⟨⟨_, n, rfl⟩, ?_⟩
    /-
      case exists_of_eq.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      hU : IsCompact U.carrier
      hU' : IsQuasiSeparated U.carrier
      f x y z : ↑(X.presheaf.obj { unop := U })
      H : Eq ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom z) 0
      n : Nat
      e : Eq (HMul.hMul (HPow.hPow f n) z) 0
      ⊢ Eq (HMul.hMul (↑⟨(fun x => HPow.hPow f x) n, ⋯⟩) z) 0
    -/
    simpa [mul_comm z] using e
    /-
      🎉 no goals
    -/


lemma exists_of_res_eq_of_qcqs {X : Scheme.{u}} {U : TopologicalSpace.Opens X}
    (hU : IsCompact U.carrier) (hU' : IsQuasiSeparated U.carrier)
    {f g s : Γ(X, U)} (hfg : f |_ᵣ X.basicOpen s = g |_ᵣ X.basicOpen s) :
    ∃ n, s ^ n * f = s ^ n * g := by
  /-
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : IsCompact U.carrier
    hU' : IsQuasiSeparated U.carrier
    f g s : ↑(X.presheaf.obj { unop := U })
    hfg : Eq (TopCat.Presheaf.restrictOpenCommRingCat f (X.basicOpen s) ⋯) (TopCat …
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow s n) f) (HMul.hMul (HPow.hPow s n) g)
  -/
  obtain ⟨n, hc⟩ := (is_localization_basicOpen_of_qcqs hU hU' s).exists_of_eq s hfg
  /-
    case intro
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : IsCompact U.carrier
    hU' : IsQuasiSeparated U.carrier
    f g s : ↑(X.presheaf.obj { unop := U })
    hfg : Eq (TopCat.Presheaf.restrictOpenCommRingCat f (X.basicOpen s) ⋯) (TopCat …
    n : Nat
    hc : Eq (HMul.hMul (HPow.hPow s n) f) (HMul.hMul (HPow.hPow s n) g)
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow s n) f) (HMul.hMul (HPow.hPow s n) g)
  -/
  use n
  /-
    🎉 no goals
  -/


lemma exists_of_res_eq_of_qcqs_of_top {X : Scheme.{u}} [CompactSpace X] [QuasiSeparatedSpace X]
    {f g s : Γ(X, ⊤)} (hfg : f |_ᵣ X.basicOpen s = g |_ᵣ X.basicOpen s) :
    ∃ n, s ^ n * f = s ^ n * g :=
  exists_of_res_eq_of_qcqs (U := ⊤) CompactSpace.isCompact_univ isQuasiSeparated_univ hfg


lemma exists_of_res_zero_of_qcqs {X : Scheme.{u}} {U : TopologicalSpace.Opens X}
    (hU : IsCompact U.carrier) (hU' : IsQuasiSeparated U.carrier)
    {f s : Γ(X, U)} (hf : f |_ᵣ X.basicOpen s = 0) :
    ∃ n, s ^ n * f = 0 := by
  suffices h : ∃ n, s ^ n * f = s ^ n * 0 by
    simpa using h
  /-
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : IsCompact U.carrier
    hU' : IsQuasiSeparated U.carrier
    f s : ↑(X.presheaf.obj { unop := U })
    hf : Eq (TopCat.Presheaf.restrictOpenCommRingCat f (X.basicOpen s) ⋯) 0
    ⊢ Exists fun n => Eq (HMul.hMul (HPow.hPow s n) f) (HMul.hMul (HPow.hPow s n) 0)
  -/
  apply exists_of_res_eq_of_qcqs hU hU'
  /-
    X : AlgebraicGeometry.Scheme
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hU : IsCompact U.carrier
    hU' : IsQuasiSeparated U.carrier
    f s : ↑(X.presheaf.obj { unop := U })
    hf : Eq (TopCat.Presheaf.restrictOpenCommRingCat f (X.basicOpen s) ⋯) 0
    ⊢ Eq (TopCat.Presheaf.restrictOpenCommRingCat f (X.basicOpen s) ⋯) (TopCat.Pre …
  -/
  simpa
  /-
    🎉 no goals
  -/


lemma exists_of_res_zero_of_qcqs_of_top {X : Scheme} [CompactSpace X] [QuasiSeparatedSpace X]
    {f s : Γ(X, ⊤)} (hf : f |_ᵣ X.basicOpen s = 0) :
    ∃ n, s ^ n * f = 0 :=
  exists_of_res_zero_of_qcqs (U := ⊤) CompactSpace.isCompact_univ isQuasiSeparated_univ hf


/-- If `U` is qcqs, then `Γ(X, D(f)) ≃ Γ(X, U)_f` for every `f : Γ(X, U)`.
This is known as the **Qcqs lemma** in [R. Vakil, *The rising sea*][RisingSea]. -/
theorem isIso_ΓSpec_adjunction_unit_app_basicOpen {X : Scheme} [CompactSpace X]
    [QuasiSeparatedSpace X] (f : X.presheaf.obj (op ⊤)) :
    IsIso ((ΓSpec.adjunction.unit.app X).c.app (op (PrimeSpectrum.basicOpen f))) := by
  refine @IsIso.of_isIso_comp_right _ _ _ _ _ _ (X.presheaf.map
    (eqToHom (Scheme.toSpecΓ_preimage_basicOpen _ _).symm).op) _ ?_
  /-
    X : AlgebraicGeometry.Scheme
    inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
    inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry …
  -/
  rw [ConcreteCategory.isIso_iff_bijective, CommRingCat.forget_map]
  /-
    X : AlgebraicGeometry.Scheme
    inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
    inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := Top.top })
    ⊢ Function.Bijective ⇑(CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry. …
  -/
  apply (config := { allowSynthFailures := true }) IsLocalization.bijective
    /-
      case inst
      X : AlgebraicGeometry.Scheme
      inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
      inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ IsLocalization ?M ((CategoryTheory.forget CommRingCat).obj (((AlgebraicGeome …
    -/
  · exact StructureSheaf.IsLocalization.to_basicOpen _ _
    /-
      🎉 no goals
    -/
    /-
      case inst
      X : AlgebraicGeometry.Scheme
      inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
      inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ IsLocalization (Submonoid.powers f) ((CategoryTheory.forget CommRingCat).obj …
    -/
  · refine is_localization_basicOpen_of_qcqs ?_ ?_ _
      /-
        case inst.refine_1
        X : AlgebraicGeometry.Scheme
        inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
        inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
        f : ↑(X.presheaf.obj { unop := Top.top })
        ⊢ IsCompact Top.top.carrier
      -/
    · exact isCompact_univ
      /-
        🎉 no goals
      -/
      /-
        case inst.refine_2
        X : AlgebraicGeometry.Scheme
        inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
        inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
        f : ↑(X.presheaf.obj { unop := Top.top })
        ⊢ IsQuasiSeparated Top.top.carrier
      -/
    · exact isQuasiSeparated_univ
      /-
        🎉 no goals
      -/
    /-
      case hf
      X : AlgebraicGeometry.Scheme
      inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
      inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.ΓSpec.adjunction …
    -/
  · simp only [RingHom.algebraMap_toAlgebra]
    -- This `rw` doesn't fire as a `simp` (`only`).
    /-
      case hf
      X : AlgebraicGeometry.Scheme
      inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
      inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.ΓSpec.adjunction …
    -/
    rw [← CommRingCat.hom_comp]
    /-
      case hf
      X : AlgebraicGeometry.Scheme
      inst✝¹ : CompactSpace ↑↑X.toPresheafedSpace
      inst✝ : QuasiSeparatedSpace ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := Top.top })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
    -/
    simp [RingHom.algebraMap_toAlgebra, ← Functor.map_comp]
    /-
      🎉 no goals
    -/


