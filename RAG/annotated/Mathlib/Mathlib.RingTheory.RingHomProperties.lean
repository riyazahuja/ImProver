/-- A property `RespectsIso` if it still holds when composed with an isomorphism -/
def RespectsIso : Prop :=
  (∀ {R S T : Type u} [CommRing R] [CommRing S] [CommRing T],
      ∀ (f : R →+* S) (e : S ≃+* T) (_ : P f), P (e.toRingHom.comp f)) ∧
    ∀ {R S T : Type u} [CommRing R] [CommRing S] [CommRing T],
      ∀ (f : S →+* T) (e : R ≃+* S) (_ : P f), P (f.comp e.toRingHom)


theorem RespectsIso.cancel_left_isIso (hP : RespectsIso @P) {R S T : CommRingCat} (f : R ⟶ S)
    (g : S ⟶ T) [IsIso f] : P (g.hom.comp f.hom) ↔ P g.hom :=
  ⟨fun H => by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso P
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      inst✝ : CategoryTheory.IsIso f
      H : P (g.hom.comp f.hom)
      ⊢ P g.hom
    -/
    convert hP.2 (f ≫ g).hom (asIso f).symm.commRingCatIsoToRingEquiv H
    /-
      case h.e'_5.h.e'_3.h
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso P
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      inst✝ : CategoryTheory.IsIso f
      H : P (g.hom.comp f.hom)
      ⊢ Eq g { hom := (CategoryTheory.CategoryStruct.comp f g).hom.comp (CategoryThe …
    -/
    exact (IsIso.inv_hom_id_assoc _ _).symm, hP.2 g.hom (asIso f).commRingCatIsoToRingEquiv⟩
    /-
      🎉 no goals
    -/


theorem RespectsIso.cancel_right_isIso (hP : RespectsIso @P) {R S T : CommRingCat} (f : R ⟶ S)
    (g : S ⟶ T) [IsIso g] : P (g.hom.comp f.hom) ↔ P f.hom :=
  ⟨fun H => by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso P
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      inst✝ : CategoryTheory.IsIso g
      H : P (g.hom.comp f.hom)
      ⊢ P f.hom
    -/
    convert hP.1 (f ≫ g).hom (asIso g).symm.commRingCatIsoToRingEquiv H
    /-
      case h.e'_5.h.e'_3.h
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso P
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      inst✝ : CategoryTheory.IsIso g
      H : P (g.hom.comp f.hom)
      ⊢ Eq f { hom := (CategoryTheory.asIso g).symm.commRingCatIsoToRingEquiv.toRing …
    -/
    change f = f ≫ g ≫ inv g
    /-
      case h.e'_5.h.e'_3.h
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso P
      R S T : CommRingCat
      f : Quiver.Hom R S
      g : Quiver.Hom S T
      inst✝ : CategoryTheory.IsIso g
      H : P (g.hom.comp f.hom)
      ⊢ Eq f (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.co …
    -/
    simp, hP.1 f.hom (asIso g).commRingCatIsoToRingEquiv⟩
    /-
      🎉 no goals
    -/


theorem RespectsIso.is_localization_away_iff (hP : RingHom.RespectsIso @P) {R S : Type u}
    (R' S' : Type u) [CommRing R] [CommRing S] [CommRing R'] [CommRing S'] [Algebra R R']
    [Algebra S S'] (f : R →+* S) (r : R) [IsLocalization.Away r R'] [IsLocalization.Away (f r) S'] :
    P (Localization.awayMap f r) ↔ P (IsLocalization.Away.map R' S' f r) := by
  let e₁ : R' ≃+* Localization.Away r :=
    (IsLocalization.algEquiv (Submonoid.powers r) _ _).toRingEquiv
  let e₂ : Localization.Away (f r) ≃+* S' :=
    (IsLocalization.algEquiv (Submonoid.powers (f r)) _ _).toRingEquiv
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso P
    R S R' S' : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    f : RingHom R S
    r : R
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    e₁ : RingEquiv R' (Localization.Away r) := (IsLocalization.algEquiv (Submonoid …
    e₂ : RingEquiv (Localization.Away (f r)) S' := (IsLocalization.algEquiv (Submo …
    ⊢ Iff (P (Localization.awayMap f r)) (P (IsLocalization.Away.map R' S' f r))
  -/
  refine (hP.cancel_left_isIso e₁.toCommRingCatIso.hom (CommRingCat.ofHom _)).symm.trans ?_
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso P
    R S R' S' : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    f : RingHom R S
    r : R
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    e₁ : RingEquiv R' (Localization.Away r) := (IsLocalization.algEquiv (Submonoid …
    e₂ : RingEquiv (Localization.Away (f r)) S' := (IsLocalization.algEquiv (Submo …
    ⊢ Iff (P ((CommRingCat.ofHom { toFun := (↑((IsLocalization.toLocalizationWithZ …
  -/
  refine (hP.cancel_right_isIso (CommRingCat.ofHom _) e₂.toCommRingCatIso.hom).symm.trans ?_
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso P
    R S R' S' : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    f : RingHom R S
    r : R
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    e₁ : RingEquiv R' (Localization.Away r) := (IsLocalization.algEquiv (Submonoid …
    e₂ : RingEquiv (Localization.Away (f r)) S' := (IsLocalization.algEquiv (Submo …
    ⊢ Iff (P (e₂.toCommRingCatIso.hom.hom.comp (CommRingCat.ofHom { toFun := Funct …
  -/
  rw [← eq_iff_iff]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso P
    R S R' S' : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    f : RingHom R S
    r : R
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    e₁ : RingEquiv R' (Localization.Away r) := (IsLocalization.algEquiv (Submonoid …
    e₂ : RingEquiv (Localization.Away (f r)) S' := (IsLocalization.algEquiv (Submo …
    ⊢ Eq (P (e₂.toCommRingCatIso.hom.hom.comp (CommRingCat.ofHom { toFun := Functi …
  -/
  congr 1
  -- Porting note: Here, the proof used to have a huge `simp` involving `[anonymous]`, which didn't
  -- work out anymore. The issue seemed to be that it couldn't handle a term in which Ring
  -- homomorphisms were repeatedly casted to the bundled category and back. Here we resolve the
  -- problem by converting the goal to a more straightforward form.
  let e := (e₂ : Localization.Away (f r) →+* S').comp
      (((IsLocalization.map (Localization.Away (f r)) f
            (by rintro x ⟨n, rfl⟩; use n; simp : Submonoid.powers r ≤ Submonoid.comap f
                (Submonoid.powers (f r)))) : Localization.Away r →+* Localization.Away (f r)).comp
                (e₁ : R' →+* Localization.Away r))
  suffices e = IsLocalization.Away.map R' S' f r by
    convert this
  /-
    case e_x
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso P
    R S R' S' : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    f : RingHom R S
    r : R
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    e₁ : RingEquiv R' (Localization.Away r) := (IsLocalization.algEquiv (Submonoid …
    e₂ : RingEquiv (Localization.Away (f r)) S' := (IsLocalization.algEquiv (Submo …
    e : RingHom R' S' := (↑e₂).comp ((IsLocalization.map (Localization.Away (f r)) …
    ⊢ Eq e (IsLocalization.Away.map R' S' f r)
  -/
  apply IsLocalization.ringHom_ext (Submonoid.powers r) _
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso P
    R S R' S' : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    f : RingHom R S
    r : R
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    e₁ : RingEquiv R' (Localization.Away r) := (IsLocalization.algEquiv (Submonoid …
    e₂ : RingEquiv (Localization.Away (f r)) S' := (IsLocalization.algEquiv (Submo …
    e : RingHom R' S' := (↑e₂).comp ((IsLocalization.map (Localization.Away (f r)) …
    ⊢ Eq (e.comp (algebraMap R R')) ((IsLocalization.Away.map R' S' f r).comp (alg …
  -/
  ext1 x
  /-
    case a
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso P
    R S R' S' : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    f : RingHom R S
    r : R
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    e₁ : RingEquiv R' (Localization.Away r) := (IsLocalization.algEquiv (Submonoid …
    e₂ : RingEquiv (Localization.Away (f r)) S' := (IsLocalization.algEquiv (Submo …
    e : RingHom R' S' := (↑e₂).comp ((IsLocalization.map (Localization.Away (f r)) …
    x : R
    ⊢ Eq ((e.comp (algebraMap R R')) x) (((IsLocalization.Away.map R' S' f r).comp …
  -/
  dsimp [e, e₁, e₂, IsLocalization.Away.map]
  /-
    case a
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso P
    R S R' S' : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    f : RingHom R S
    r : R
    inst✝¹ : IsLocalization.Away r R'
    inst✝ : IsLocalization.Away (f r) S'
    e₁ : RingEquiv R' (Localization.Away r) := (IsLocalization.algEquiv (Submonoid …
    e₂ : RingEquiv (Localization.Away (f r)) S' := (IsLocalization.algEquiv (Submo …
    e : RingHom R' S' := (↑e₂).comp ((IsLocalization.map (Localization.Away (f r)) …
    x : R
    ⊢ Eq ((IsLocalization.map S' (RingHom.id S) ⋯) ((IsLocalization.map (Localizat …
  -/
  simp only [IsLocalization.map_eq, id_apply, RingHomCompTriple.comp_apply]
  /-
    🎉 no goals
  -/


/-- A property is `StableUnderComposition` if the composition of two such morphisms
still falls in the class. -/
def StableUnderComposition : Prop :=
  ∀ ⦃R S T⦄ [CommRing R] [CommRing S] [CommRing T],
    ∀ (f : R →+* S) (g : S →+* T) (_ : P f) (_ : P g), P (g.comp f)


theorem StableUnderComposition.respectsIso (hP : RingHom.StableUnderComposition @P)
    (hP' : ∀ {R S : Type u} [CommRing R] [CommRing S] (e : R ≃+* S), P e.toRingHom) :
    RingHom.RespectsIso @P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.StableUnderComposition P
    hP' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEqui …
    ⊢ RingHom.RespectsIso P
  -/
  constructor
    /-
      case left
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderComposition P
      hP' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEqui …
      ⊢ ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : CommR …
    -/
  · introv H
    /-
      case left
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderComposition P
      hP' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEqui …
      R S T : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      H : P f
      ⊢ P (e.toRingHom.comp f)
    -/
    apply hP
    /-
      case left.x
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderComposition P
      hP' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEqui …
      R S T : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      H : P f
      ⊢ P f
    -/
    exacts [H, hP' e]
    /-
      🎉 no goals
    -/
    /-
      case right
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderComposition P
      hP' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEqui …
      ⊢ ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : CommR …
    -/
  · introv H
    /-
      case right
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderComposition P
      hP' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEqui …
      R S T : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      H : P f
      ⊢ P (f.comp e.toRingHom)
    -/
    apply hP
    /-
      case right.x
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.StableUnderComposition P
      hP' : ∀ {R S : Type u} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEqui …
      R S T : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      H : P f
      ⊢ P e.toRingHom
    -/
    exacts [hP' e, H]
    /-
      🎉 no goals
    -/


/-- A morphism property `P` is `IsStableUnderBaseChange` if `P(S →+* A)` implies
`P(B →+* A ⊗[S] B)`. -/
def IsStableUnderBaseChange : Prop :=
  ∀ (R S R' S') [CommRing R] [CommRing S] [CommRing R'] [CommRing S'],
    ∀ [Algebra R S] [Algebra R R'] [Algebra R S'] [Algebra S S'] [Algebra R' S'],
      ∀ [IsScalarTower R S S'] [IsScalarTower R R' S'],
        ∀ [Algebra.IsPushout R S R' S'], P (algebraMap R S) → P (algebraMap R' S')


theorem IsStableUnderBaseChange.mk (h₁ : RespectsIso @P)
    (h₂ :
      ∀ ⦃R S T⦄ [CommRing R] [CommRing S] [CommRing T],
        ∀ [Algebra R S] [Algebra R T],
          P (algebraMap R T) →
            P (Algebra.TensorProduct.includeLeftRingHom : S →+* TensorProduct R S T)) :
    IsStableUnderBaseChange @P := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    h₁ : RingHom.RespectsIso P
    h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
    ⊢ RingHom.IsStableUnderBaseChange P
  -/
  introv R h H
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    h₁ : RingHom.RespectsIso P
    h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
    R S R' S' : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing R'
    inst✝⁷ : CommRing S'
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R R'
    inst✝⁴ : Algebra R S'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R' S'
    inst✝¹ : IsScalarTower R S S'
    inst✝ : IsScalarTower R R' S'
    h : Algebra.IsPushout R S R' S'
    H : P (algebraMap R S)
    ⊢ P (algebraMap R' S')
  -/
  let e := h.symm.1.equiv
  let f' :=
    Algebra.TensorProduct.productMap (IsScalarTower.toAlgHom R R' S')
      (IsScalarTower.toAlgHom R S S')
  have : ∀ x, e x = f' x := by
    intro x
    change e.toLinearMap.restrictScalars R x = f'.toLinearMap x
    congr 1
    apply TensorProduct.ext'
    intro x y
    simp [e, f', IsBaseChange.equiv_tmul, Algebra.smul_def]
  -- Porting Note: This had a lot of implicit inferences which didn't resolve anymore.
  -- Added those in
  convert h₁.1 (_ : R' →+* TensorProduct R R' S) (_ : TensorProduct R R' S ≃+* S')
      (h₂ H : P (_ : R' →+* TensorProduct R R' S))
  /-
    case h.e'_5
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    h₁ : RingHom.RespectsIso P
    h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
    R S R' S' : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing R'
    inst✝⁷ : CommRing S'
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R R'
    inst✝⁴ : Algebra R S'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R' S'
    inst✝¹ : IsScalarTower R S S'
    inst✝ : IsScalarTower R R' S'
    h : Algebra.IsPushout R S R' S'
    H : P (algebraMap R S)
    e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' := ⋯.equiv
    f' : AlgHom R (TensorProduct R R' S) S' := Algebra.TensorProduct.productMap (I …
    this : ∀ (x : TensorProduct R R' S), Eq (e x) (f' x)
    ⊢ Eq (algebraMap R' S') ((RingEquiv.toRingHom ?m.54562).comp Algebra.TensorPro …
  -/
  swap
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso P
      h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      R S R' S' : Type u
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : CommRing R'
      inst✝⁷ : CommRing S'
      inst✝⁶ : Algebra R S
      inst✝⁵ : Algebra R R'
      inst✝⁴ : Algebra R S'
      inst✝³ : Algebra S S'
      inst✝² : Algebra R' S'
      inst✝¹ : IsScalarTower R S S'
      inst✝ : IsScalarTower R R' S'
      h : Algebra.IsPushout R S R' S'
      H : P (algebraMap R S)
      e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' := ⋯.equiv
      f' : AlgHom R (TensorProduct R R' S) S' := Algebra.TensorProduct.productMap (I …
      this : ∀ (x : TensorProduct R R' S), Eq (e x) (f' x)
      ⊢ RingEquiv (TensorProduct R R' S) S'
    -/
  · refine { e with map_mul' := fun x y => ?_ }
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso P
      h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      R S R' S' : Type u
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : CommRing R'
      inst✝⁷ : CommRing S'
      inst✝⁶ : Algebra R S
      inst✝⁵ : Algebra R R'
      inst✝⁴ : Algebra R S'
      inst✝³ : Algebra S S'
      inst✝² : Algebra R' S'
      inst✝¹ : IsScalarTower R S S'
      inst✝ : IsScalarTower R R' S'
      h : Algebra.IsPushout R S R' S'
      H : P (algebraMap R S)
      e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' := ⋯.equiv
      f' : AlgHom R (TensorProduct R R' S) S' := Algebra.TensorProduct.productMap (I …
      this : ∀ (x : TensorProduct R R' S), Eq (e x) (f' x)
      x y : TensorProduct R R' S
      ⊢ Eq ({ toFun := (↑e).toFun, invFun := e.invFun, left_inv := ⋯, right_inv := ⋯ …
    -/
    change e (x * y) = e x * e y
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso P
      h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      R S R' S' : Type u
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : CommRing R'
      inst✝⁷ : CommRing S'
      inst✝⁶ : Algebra R S
      inst✝⁵ : Algebra R R'
      inst✝⁴ : Algebra R S'
      inst✝³ : Algebra S S'
      inst✝² : Algebra R' S'
      inst✝¹ : IsScalarTower R S S'
      inst✝ : IsScalarTower R R' S'
      h : Algebra.IsPushout R S R' S'
      H : P (algebraMap R S)
      e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' := ⋯.equiv
      f' : AlgHom R (TensorProduct R R' S) S' := Algebra.TensorProduct.productMap (I …
      this : ∀ (x : TensorProduct R R' S), Eq (e x) (f' x)
      x y : TensorProduct R R' S
      ⊢ Eq (e (HMul.hMul x y)) (HMul.hMul (e x) (e y))
    -/
    simp_rw [this]
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso P
      h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      R S R' S' : Type u
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : CommRing R'
      inst✝⁷ : CommRing S'
      inst✝⁶ : Algebra R S
      inst✝⁵ : Algebra R R'
      inst✝⁴ : Algebra R S'
      inst✝³ : Algebra S S'
      inst✝² : Algebra R' S'
      inst✝¹ : IsScalarTower R S S'
      inst✝ : IsScalarTower R R' S'
      h : Algebra.IsPushout R S R' S'
      H : P (algebraMap R S)
      e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' := ⋯.equiv
      f' : AlgHom R (TensorProduct R R' S) S' := Algebra.TensorProduct.productMap (I …
      this : ∀ (x : TensorProduct R R' S), Eq (e x) (f' x)
      x y : TensorProduct R R' S
      ⊢ Eq (f' (HMul.hMul x y)) (HMul.hMul (f' x) (f' y))
    -/
    exact map_mul f' _ _
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso P
      h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      R S R' S' : Type u
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : CommRing R'
      inst✝⁷ : CommRing S'
      inst✝⁶ : Algebra R S
      inst✝⁵ : Algebra R R'
      inst✝⁴ : Algebra R S'
      inst✝³ : Algebra S S'
      inst✝² : Algebra R' S'
      inst✝¹ : IsScalarTower R S S'
      inst✝ : IsScalarTower R R' S'
      h : Algebra.IsPushout R S R' S'
      H : P (algebraMap R S)
      e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' := ⋯.equiv
      f' : AlgHom R (TensorProduct R R' S) S' := Algebra.TensorProduct.productMap (I …
      this : ∀ (x : TensorProduct R R' S), Eq (e x) (f' x)
      ⊢ Eq (algebraMap R' S') ({ toFun := (↑e).toFun, invFun := e.invFun, left_inv : …
    -/
  · ext x
    /-
      case h.e'_5.a
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso P
      h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      R S R' S' : Type u
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : CommRing R'
      inst✝⁷ : CommRing S'
      inst✝⁶ : Algebra R S
      inst✝⁵ : Algebra R R'
      inst✝⁴ : Algebra R S'
      inst✝³ : Algebra S S'
      inst✝² : Algebra R' S'
      inst✝¹ : IsScalarTower R S S'
      inst✝ : IsScalarTower R R' S'
      h : Algebra.IsPushout R S R' S'
      H : P (algebraMap R S)
      e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' := ⋯.equiv
      f' : AlgHom R (TensorProduct R R' S) S' := Algebra.TensorProduct.productMap (I …
      this : ∀ (x : TensorProduct R R' S), Eq (e x) (f' x)
      x : R'
      ⊢ Eq ((algebraMap R' S') x) (({ toFun := (↑e).toFun, invFun := e.invFun, left_ …
    -/
    change _ = e (x ⊗ₜ[R] 1)
    -- Porting note: Had `dsimp only [e]` here, which didn't work anymore
    /-
      case h.e'_5.a
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h₁ : RingHom.RespectsIso P
      h₂ : ∀ ⦃R S T : Type u⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Co …
      R S R' S' : Type u
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : CommRing R'
      inst✝⁷ : CommRing S'
      inst✝⁶ : Algebra R S
      inst✝⁵ : Algebra R R'
      inst✝⁴ : Algebra R S'
      inst✝³ : Algebra S S'
      inst✝² : Algebra R' S'
      inst✝¹ : IsScalarTower R S S'
      inst✝ : IsScalarTower R R' S'
      h : Algebra.IsPushout R S R' S'
      H : P (algebraMap R S)
      e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' := ⋯.equiv
      f' : AlgHom R (TensorProduct R R' S) S' := Algebra.TensorProduct.productMap (I …
      this : ∀ (x : TensorProduct R R' S), Eq (e x) (f' x)
      x : R'
      ⊢ Eq ((algebraMap R' S') x) (e (TensorProduct.tmul R x 1))
    -/
    rw [h.symm.1.equiv_tmul, Algebra.smul_def, AlgHom.toLinearMap_apply, map_one, mul_one]
    /-
      🎉 no goals
    -/


theorem IsStableUnderBaseChange.pushout_inl (hP : RingHom.IsStableUnderBaseChange @P)
    (hP' : RingHom.RespectsIso @P) {R S T : CommRingCat} (f : R ⟶ S) (g : R ⟶ T) (H : P g.hom) :
    P (pushout.inl _ _ : S ⟶ pushout f g).hom := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange P
    hP' : RingHom.RespectsIso P
    R S T : CommRingCat
    f : Quiver.Hom R S
    g : Quiver.Hom R T
    H : P g.hom
    ⊢ P (CategoryTheory.Limits.pushout.inl f g).hom
  -/
  letI := f.hom.toAlgebra
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange P
    hP' : RingHom.RespectsIso P
    R S T : CommRingCat
    f : Quiver.Hom R S
    g : Quiver.Hom R T
    H : P g.hom
    this : Algebra ↑R ↑S := f.hom.toAlgebra
    ⊢ P (CategoryTheory.Limits.pushout.inl f g).hom
  -/
  letI := g.hom.toAlgebra
  rw [← show _ = pushout.inl f g from
      colimit.isoColimitCocone_ι_inv ⟨_, CommRingCat.pushoutCoconeIsColimit R S T⟩ WalkingSpan.left,
    CommRingCat.hom_comp, hP'.cancel_right_isIso]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange P
    hP' : RingHom.RespectsIso P
    R S T : CommRingCat
    f : Quiver.Hom R S
    g : Quiver.Hom R T
    H : P g.hom
    this✝ : Algebra ↑R ↑S := f.hom.toAlgebra
    this : Algebra ↑R ↑T := g.hom.toAlgebra
    ⊢ P ({ cocone := CommRingCat.pushoutCocone ↑R ↑S ↑T, isColimit := CommRingCat. …
  -/
  dsimp only [CommRingCat.pushoutCocone_inl, PushoutCocone.ι_app_left]
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange P
    hP' : RingHom.RespectsIso P
    R S T : CommRingCat
    f : Quiver.Hom R S
    g : Quiver.Hom R T
    H : P g.hom
    this✝ : Algebra ↑R ↑S := f.hom.toAlgebra
    this : Algebra ↑R ↑T := g.hom.toAlgebra
    ⊢ P Algebra.TensorProduct.includeLeftRingHom
  -/
  apply hP R T S (TensorProduct R S T)
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.IsStableUnderBaseChange P
    hP' : RingHom.RespectsIso P
    R S T : CommRingCat
    f : Quiver.Hom R S
    g : Quiver.Hom R T
    H : P g.hom
    this✝ : Algebra ↑R ↑S := f.hom.toAlgebra
    this : Algebra ↑R ↑T := g.hom.toAlgebra
    ⊢ P (algebraMap ↑R ↑T)
  -/
  exact H
  /-
    🎉 no goals
  -/


/-- The categorical `MorphismProperty` associated to a property of ring homs expressed
non-categorical terms. -/
def toMorphismProperty : MorphismProperty CommRingCat := fun _ _ f ↦ P f.hom


lemma toMorphismProperty_respectsIso_iff :
    RespectsIso P ↔ (toMorphismProperty P).RespectsIso := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    ⊢ Iff (RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P) (RingHom. …
  -/
  refine ⟨fun h ↦ MorphismProperty.RespectsIso.mk _ ?_ ?_, fun h ↦ ⟨?_, ?_⟩⟩
    /-
      case refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      ⊢ ∀ {X Y Z : CommRingCat} (e : CategoryTheory.Iso X Y) (f : Quiver.Hom Y Z), R …
    -/
  · intro X Y Z e f hf
    /-
      case refine_1
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : CommRingCat
      e : CategoryTheory.Iso X Y
      f : Quiver.Hom Y Z
      hf : RingHom.toMorphismProperty (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ RingHom.toMorphismProperty (fun {R S} [CommRing R] [CommRing S] => P) (Categ …
    -/
    exact h.right f.hom e.commRingCatIsoToRingEquiv hf
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      ⊢ ∀ {X Y Z : CommRingCat} (e : CategoryTheory.Iso Y Z) (f : Quiver.Hom X Y), R …
    -/
  · intro X Y Z e f hf
    /-
      case refine_2
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : CommRingCat
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      hf : RingHom.toMorphismProperty (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ RingHom.toMorphismProperty (fun {R S} [CommRing R] [CommRing S] => P) (Categ …
    -/
    exact h.left f.hom e.commRingCatIsoToRingEquiv hf
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Resp …
      ⊢ ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : CommR …
    -/
  · intro X Y Z _ _ _ f e hf
    exact MorphismProperty.RespectsIso.postcomp (toMorphismProperty P)
      e.toCommRingCatIso.hom (CommRingCat.ofHom f) hf
    /-
      case refine_4
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      h : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Resp …
      ⊢ ∀ {R S T : Type u} [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : CommR …
    -/
  · intro X Y Z _ _ _ f e
    exact MorphismProperty.RespectsIso.precomp (toMorphismProperty P)
      e.toCommRingCatIso.hom (CommRingCat.ofHom f)


