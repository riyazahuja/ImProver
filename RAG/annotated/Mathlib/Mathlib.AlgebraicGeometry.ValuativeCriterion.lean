/--
A valuative commutative square over a morphism `f : X ⟶ Y` is a square
```
Spec K ⟶ Y
  |       |
  ↓       ↓
Spec R ⟶ X
```
where `R` is a valuation ring, and `K` is its ring of fractions.

We are interested in finding lifts `Spec R ⟶ Y` of this diagram.
-/
structure ValuativeCommSq {X Y : Scheme.{u}} (f : X ⟶ Y) where
  /-- The valuation ring of a valuative commutative square. -/
  R : Type u
  [commRing : CommRing R]
  [domain : IsDomain R]
  [valuationRing : ValuationRing R]
  /-- The field of fractions of a valuative commutative square. -/
  K : Type u
  [field : Field K]
  [algebra : Algebra R K]
  [isFractionRing : IsFractionRing R K]
  /-- The top map in a valuative commutative map. -/
  (i₁ : Spec (.of K) ⟶ X)
  /-- The bottom map in a valuative commutative map. -/
  (i₂ : Spec (.of R) ⟶ Y)
  (commSq : CommSq i₁ (Spec.map (CommRingCat.ofHom (algebraMap R K))) f i₂)


/-- A morphism `f : X ⟶ Y` satisfies the existence part of the valuative criterion if
every valuative commutative square over `f` has (at least) a lift. -/
def ValuativeCriterion.Existence : MorphismProperty Scheme :=
  fun _ _ f ↦ ∀ S : ValuativeCommSq f, S.commSq.HasLift


/-- A morphism `f : X ⟶ Y` satisfies the uniqueness part of the valuative criterion if
every valuative commutative square over `f` has at most one lift. -/
def ValuativeCriterion.Uniqueness : MorphismProperty Scheme :=
  fun _ _ f ↦ ∀ S : ValuativeCommSq f, Subsingleton S.commSq.LiftStruct


/-- A morphism `f : X ⟶ Y` satisfies the valuative criterion if
every valuative commutative square over `f` has a unique lift. -/
def ValuativeCriterion : MorphismProperty Scheme :=
  fun _ _ f ↦ ∀ S : ValuativeCommSq f, Nonempty (Unique (S.commSq.LiftStruct))


lemma ValuativeCriterion.iff {f : X ⟶ Y} :
    ValuativeCriterion f ↔ Existence f ∧ Uniqueness f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.ValuativeCriterion f) (And (AlgebraicGeometry.Valuati …
  -/
  show (∀ _, _) ↔ (∀ _, _) ∧ (∀ _, _)
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (∀ (x : AlgebraicGeometry.ValuativeCommSq f), Nonempty (Unique ⋯.LiftStr …
  -/
  simp_rw [← forall_and, unique_iff_subsingleton_and_nonempty, and_comm, CommSq.HasLift.iff]
  /-
    🎉 no goals
  -/


lemma ValuativeCriterion.eq :
    ValuativeCriterion = Existence ⊓ Uniqueness := by
  /-
    ⊢ Eq AlgebraicGeometry.ValuativeCriterion (Min.min AlgebraicGeometry.Valuative …
  -/
  ext X Y f
  /-
    case h
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (AlgebraicGeometry.ValuativeCriterion f) (Min.min AlgebraicGeometry.Valu …
  -/
  exact iff
  /-
    🎉 no goals
  -/


lemma ValuativeCriterion.existence {f : X ⟶ Y} (h : ValuativeCriterion f) :
    ValuativeCriterion.Existence f := (iff.mp h).1


lemma ValuativeCriterion.uniqueness {f : X ⟶ Y} (h : ValuativeCriterion f) :
    ValuativeCriterion.Uniqueness f := (iff.mp h).2


@[stacks 01KE]
lemma specializingMap (H : ValuativeCriterion.Existence f) :
    SpecializingMap f.base := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.ValuativeCriterion.Existence f
    ⊢ SpecializingMap ⇑f.base
  -/
  intro x' y h
  let stalk_y_to_residue_x' : Y.presheaf.stalk y ⟶ X.residueField x' :=
    Y.presheaf.stalkSpecializes h ≫ f.stalkMap x' ≫ X.residue x'
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.ValuativeCriterion.Existence f
    x' : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
    stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
    ⊢ Exists fun a' => And (flip (fun x1 x2 => Specializes x1 x2) a' x') (Eq (f.ba …
  -/
  obtain ⟨A, hA, hA_local⟩ := exists_factor_valuationRing stalk_y_to_residue_x'.hom
  let stalk_y_to_A : Y.presheaf.stalk y ⟶ .of A :=
    CommRingCat.ofHom (stalk_y_to_residue_x'.hom.codRestrict _ hA)
  have w : X.fromSpecResidueField x' ≫ f =
      Spec.map (CommRingCat.ofHom (algebraMap A (X.residueField x'))) ≫
        Spec.map stalk_y_to_A ≫ Y.fromSpecStalk y := by
    rw [Scheme.fromSpecResidueField, Category.assoc, ← Scheme.Spec_map_stalkMap_fromSpecStalk,
      ← Scheme.Spec_map_stalkSpecializes_fromSpecStalk h]
    simp_rw [← Spec.map_comp_assoc]
    rfl
  /-
    case intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.ValuativeCriterion.Existence f
    x' : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
    stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
    A : ValuationSubring ↑(X.residueField x')
    hA : ∀ (x : ↑(Y.presheaf.stalk y)), Membership.mem A.toSubring (stalk_y_to_res …
    hA_local : IsLocalHom (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
    stalk_y_to_A : Quiver.Hom (Y.presheaf.stalk y) (CommRingCat.of (Subtype fun x  …
    w : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecResidueField x') f) (Cat …
    ⊢ Exists fun a' => And (flip (fun x1 x2 => Specializes x1 x2) a' x') (Eq (f.ba …
  -/
  obtain ⟨l, hl₁, hl₂⟩ := (H { R := A, K := X.residueField x', commSq := ⟨w⟩ }).exists_lift
  /-
    case intro.intro.intro.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.ValuativeCriterion.Existence f
    x' : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
    stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
    A : ValuationSubring ↑(X.residueField x')
    hA : ∀ (x : ↑(Y.presheaf.stalk y)), Membership.mem A.toSubring (stalk_y_to_res …
    hA_local : IsLocalHom (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
    stalk_y_to_A : Quiver.Hom (Y.presheaf.stalk y) (CommRingCat.of (Subtype fun x  …
    w : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecResidueField x') f) (Cat …
    l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (AlgebraicGeometry.Valu …
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l f) (AlgebraicGeometry.Valuative …
    ⊢ Exists fun a' => And (flip (fun x1 x2 => Specializes x1 x2) a' x') (Eq (f.ba …
  -/
  dsimp only at hl₁ hl₂
  /-
    case intro.intro.intro.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.ValuativeCriterion.Existence f
    x' : ↑↑X.toPresheafedSpace
    y : ↑↑Y.toPresheafedSpace
    h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
    stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
    A : ValuationSubring ↑(X.residueField x')
    hA : ∀ (x : ↑(Y.presheaf.stalk y)), Membership.mem A.toSubring (stalk_y_to_res …
    hA_local : IsLocalHom (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
    stalk_y_to_A : Quiver.Hom (Y.presheaf.stalk y) (CommRingCat.of (Subtype fun x  …
    w : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecResidueField x') f) (Cat …
    l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (AlgebraicGeometry.Valu …
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l f) (CategoryTheory.CategoryStru …
    ⊢ Exists fun a' => And (flip (fun x1 x2 => Specializes x1 x2) a' x') (Eq (f.ba …
  -/
  refine ⟨l.base (closedPoint A), ?_, ?_⟩
    /-
      case intro.intro.intro.mk.refine_1
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.ValuativeCriterion.Existence f
      x' : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
      stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
      A : ValuationSubring ↑(X.residueField x')
      hA : ∀ (x : ↑(Y.presheaf.stalk y)), Membership.mem A.toSubring (stalk_y_to_res …
      hA_local : IsLocalHom (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
      stalk_y_to_A : Quiver.Hom (Y.presheaf.stalk y) (CommRingCat.of (Subtype fun x  …
      w : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecResidueField x') f) (Cat …
      l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (AlgebraicGeometry.Valu …
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l f) (CategoryTheory.CategoryStru …
      ⊢ flip (fun x1 x2 => Specializes x1 x2) (l.base (IsLocalRing.closedPoint (Subt …
    -/
  · simp_rw [← Scheme.fromSpecResidueField_apply x' (closedPoint (X.residueField x')), ← hl₁]
    /-
      case intro.intro.intro.mk.refine_1
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.ValuativeCriterion.Existence f
      x' : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
      stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
      A : ValuationSubring ↑(X.residueField x')
      hA : ∀ (x : ↑(Y.presheaf.stalk y)), Membership.mem A.toSubring (stalk_y_to_res …
      hA_local : IsLocalHom (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
      stalk_y_to_A : Quiver.Hom (Y.presheaf.stalk y) (CommRingCat.of (Subtype fun x  …
      w : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecResidueField x') f) (Cat …
      l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (AlgebraicGeometry.Valu …
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l f) (CategoryTheory.CategoryStru …
      ⊢ flip (fun x1 x2 => Specializes x1 x2) (l.base (IsLocalRing.closedPoint (Subt …
    -/
    exact (specializes_closedPoint _).map l.base.2
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.mk.refine_2
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.ValuativeCriterion.Existence f
      x' : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
      stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
      A : ValuationSubring ↑(X.residueField x')
      hA : ∀ (x : ↑(Y.presheaf.stalk y)), Membership.mem A.toSubring (stalk_y_to_res …
      hA_local : IsLocalHom (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
      stalk_y_to_A : Quiver.Hom (Y.presheaf.stalk y) (CommRingCat.of (Subtype fun x  …
      w : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecResidueField x') f) (Cat …
      l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (AlgebraicGeometry.Valu …
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l f) (CategoryTheory.CategoryStru …
      ⊢ Eq (f.base (l.base (IsLocalRing.closedPoint (Subtype fun x => Membership.mem …
    -/
  · rw [← Scheme.comp_base_apply, hl₂]
    /-
      case intro.intro.intro.mk.refine_2
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.ValuativeCriterion.Existence f
      x' : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
      stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
      A : ValuationSubring ↑(X.residueField x')
      hA : ∀ (x : ↑(Y.presheaf.stalk y)), Membership.mem A.toSubring (stalk_y_to_res …
      hA_local : IsLocalHom (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
      stalk_y_to_A : Quiver.Hom (Y.presheaf.stalk y) (CommRingCat.of (Subtype fun x  …
      w : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecResidueField x') f) (Cat …
      l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (AlgebraicGeometry.Valu …
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l f) (CategoryTheory.CategoryStru …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map stalk_y_ …
    -/
    simp only [Scheme.comp_coeBase, TopCat.coe_comp, Function.comp_apply]
    have : (Spec.map stalk_y_to_A).base (closedPoint A) = closedPoint (Y.presheaf.stalk y) :=
      comap_closedPoint (S := A) (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
    /-
      case intro.intro.intro.mk.refine_2
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.ValuativeCriterion.Existence f
      x' : ↑↑X.toPresheafedSpace
      y : ↑↑Y.toPresheafedSpace
      h : flip (fun x1 x2 => Specializes x1 x2) y (f.base x')
      stalk_y_to_residue_x' : Quiver.Hom (Y.presheaf.stalk y) (X.residueField x') := …
      A : ValuationSubring ↑(X.residueField x')
      hA : ∀ (x : ↑(Y.presheaf.stalk y)), Membership.mem A.toSubring (stalk_y_to_res …
      hA_local : IsLocalHom (stalk_y_to_residue_x'.hom.codRestrict A.toSubring hA)
      stalk_y_to_A : Quiver.Hom (Y.presheaf.stalk y) (CommRingCat.of (Subtype fun x  …
      w : Eq (CategoryTheory.CategoryStruct.comp (X.fromSpecResidueField x') f) (Cat …
      l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of (AlgebraicGeometry.Valu …
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l f) (CategoryTheory.CategoryStru …
      this : Eq ((AlgebraicGeometry.Spec.map stalk_y_to_A).base (IsLocalRing.closedP …
      ⊢ Eq ((Y.fromSpecStalk y).base ((AlgebraicGeometry.Spec.map stalk_y_to_A).base …
    -/
    rw [this, Y.fromSpecStalk_closedPoint]
    /-
      🎉 no goals
    -/


instance {R S : CommRingCat} (e : R ≅ S) : IsLocalHom e.hom.hom :=
  isLocalHom_of_isIso _


lemma of_specializingMap (H : (topologically @SpecializingMap).universally f) :
    ValuativeCriterion.Existence f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : (AlgebraicGeometry.topologically @SpecializingMap).universally f
    ⊢ AlgebraicGeometry.ValuativeCriterion.Existence f
  -/
  rintro ⟨R, K, i₁, i₂, ⟨w⟩⟩
  /-
    case mk.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : (AlgebraicGeometry.topologically @SpecializingMap).universally f
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    ⊢ ⋯.HasLift
  -/
  haveI : IsDomain (CommRingCat.of R) := ‹_›
  /-
    case mk.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : (AlgebraicGeometry.topologically @SpecializingMap).universally f
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this : IsDomain ↑(CommRingCat.of R)
    ⊢ ⋯.HasLift
  -/
  haveI : ValuationRing (CommRingCat.of R) := ‹_›
  /-
    case mk.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : (AlgebraicGeometry.topologically @SpecializingMap).universally f
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝ : IsDomain ↑(CommRingCat.of R)
    this : ValuationRing ↑(CommRingCat.of R)
    ⊢ ⋯.HasLift
  -/
  letI : Field (CommRingCat.of K) := ‹_›
  /-
    case mk.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : (AlgebraicGeometry.topologically @SpecializingMap).universally f
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝¹ : IsDomain ↑(CommRingCat.of R)
    this✝ : ValuationRing ↑(CommRingCat.of R)
    this : Field ↑(CommRingCat.of K) := field✝
    ⊢ ⋯.HasLift
  -/
  replace H := H (pullback.snd i₂ f) i₂ (pullback.fst i₂ f) (.of_hasPullback i₂ f)
  /-
    case mk.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝¹ : IsDomain ↑(CommRingCat.of R)
    this✝ : ValuationRing ↑(CommRingCat.of R)
    this : Field ↑(CommRingCat.of K) := field✝
    H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
    ⊢ ⋯.HasLift
  -/
  let lft := pullback.lift (Spec.map (CommRingCat.ofHom (algebraMap R K))) i₁ w.symm
  /-
    case mk.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝¹ : IsDomain ↑(CommRingCat.of R)
    this✝ : ValuationRing ↑(CommRingCat.of R)
    this : Field ↑(CommRingCat.of K) := field✝
    H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
    lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
    ⊢ ⋯.HasLift
  -/
  obtain ⟨x, h₁, h₂⟩ := @H (lft.base (closedPoint _)) _ (specializes_closedPoint (R := R) _)
  let e : CommRingCat.of R ≅ (Spec (.of R)).presheaf.stalk ((pullback.fst i₂ f).base x) :=
    (stalkClosedPointIso (.of R)).symm ≪≫
      (Spec (.of R)).presheaf.stalkCongr (.of_eq h₂.symm)
  /-
    case mk.mk.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝¹ : IsDomain ↑(CommRingCat.of R)
    this✝ : ValuationRing ↑(CommRingCat.of R)
    this : Field ↑(CommRingCat.of K) := field✝
    H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
    lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
    x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
    h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
    h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
    e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
    ⊢ ⋯.HasLift
  -/
  let α := e.hom ≫ (pullback.fst i₂ f).stalkMap x
  /-
    case mk.mk.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝¹ : IsDomain ↑(CommRingCat.of R)
    this✝ : ValuationRing ↑(CommRingCat.of R)
    this : Field ↑(CommRingCat.of K) := field✝
    H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
    lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
    x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
    h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
    h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
    e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
    α : Quiver.Hom (CommRingCat.of R) ((CategoryTheory.Limits.pullback i₂ f).presh …
    ⊢ ⋯.HasLift
  -/
  have : IsLocalHom e.hom.hom := isLocalHom_of_isIso e.hom
  have : IsLocalHom α.hom := inferInstanceAs
    (IsLocalHom (((pullback.fst i₂ f).stalkMap x).hom.comp e.hom.hom))
  /-
    case mk.mk.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝³ : IsDomain ↑(CommRingCat.of R)
    this✝² : ValuationRing ↑(CommRingCat.of R)
    this✝¹ : Field ↑(CommRingCat.of K) := field✝
    H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
    lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
    x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
    h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
    h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
    e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
    α : Quiver.Hom (CommRingCat.of R) ((CategoryTheory.Limits.pullback i₂ f).presh …
    this✝ : IsLocalHom e.hom.hom
    this : IsLocalHom α.hom
    ⊢ ⋯.HasLift
  -/
  let β := (pullback i₂ f).presheaf.stalkSpecializes h₁ ≫ Scheme.stalkClosedPointTo lft
  have hαβ : α ≫ β = CommRingCat.ofHom (algebraMap R K) := by
    simp only [CommRingCat.coe_of, Iso.trans_hom, Iso.symm_hom, TopCat.Presheaf.stalkCongr_hom,
      Category.assoc, α, e, β, stalkClosedPointIso_inv, StructureSheaf.toStalk]
    show (Scheme.ΓSpecIso (.of R)).inv ≫ (Spec (.of R)).presheaf.germ _ _ _ ≫ _ = _
    simp only [TopCat.Presheaf.germ_stalkSpecializes_assoc, Scheme.stalkMap_germ_assoc,
      TopologicalSpace.Opens.map_top]
    erw [Scheme.germ_stalkClosedPointTo lft ⊤ trivial,
      ← Scheme.comp_app_assoc lft (pullback.fst i₂ f)]
    rw [pullback.lift_fst]
    simp
  have hbij := (bijective_rangeRestrict_comp_of_valuationRing (R := R) (K := K) α.hom β.hom
    (CommRingCat.hom_ext_iff.mp hαβ))
  let φ : (pullback i₂ f).presheaf.stalk x ⟶ CommRingCat.of R := CommRingCat.ofHom <|
    (RingEquiv.ofBijective _ hbij).symm.toRingHom.comp β.hom.rangeRestrict
  /-
    case mk.mk.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝³ : IsDomain ↑(CommRingCat.of R)
    this✝² : ValuationRing ↑(CommRingCat.of R)
    this✝¹ : Field ↑(CommRingCat.of K) := field✝
    H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
    lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
    x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
    h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
    h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
    e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
    α : Quiver.Hom (CommRingCat.of R) ((CategoryTheory.Limits.pullback i₂ f).presh …
    this✝ : IsLocalHom e.hom.hom
    this : IsLocalHom α.hom
    β : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
    hαβ : Eq (CategoryTheory.CategoryStruct.comp α β) (CommRingCat.ofHom (algebraM …
    hbij : Function.Bijective ⇑(β.hom.rangeRestrict.comp α.hom)
    φ : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
    ⊢ ⋯.HasLift
  -/
  have hαφ : α ≫ φ = 𝟙 _ := by ext x; exact (RingEquiv.ofBijective _ hbij).symm_apply_apply x
  have hαφ' : (pullback.fst i₂ f).stalkMap x ≫ φ = e.inv := by
    rw [← cancel_epi e.hom, ← Category.assoc, hαφ, e.hom_inv_id]
  have hφβ : φ ≫ CommRingCat.ofHom (algebraMap R K) = β :=
    hαβ ▸ CommRingCat.hom_ext (RingHom.ext fun x ↦ congr_arg Subtype.val
      ((RingEquiv.ofBijective _ hbij).apply_symm_apply (β.hom.rangeRestrict x)))
  /-
    case mk.mk.intro.intro
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    R : Type u
    commRing✝ : CommRing R
    domain✝ : IsDomain R
    valuationRing✝ : ValuationRing R
    K : Type u
    field✝ : Field K
    algebra✝ : Algebra R K
    isFractionRing✝ : IsFractionRing R K
    i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
    i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
    w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
    this✝³ : IsDomain ↑(CommRingCat.of R)
    this✝² : ValuationRing ↑(CommRingCat.of R)
    this✝¹ : Field ↑(CommRingCat.of K) := field✝
    H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
    lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
    x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
    h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
    h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
    e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
    α : Quiver.Hom (CommRingCat.of R) ((CategoryTheory.Limits.pullback i₂ f).presh …
    this✝ : IsLocalHom e.hom.hom
    this : IsLocalHom α.hom
    β : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
    hαβ : Eq (CategoryTheory.CategoryStruct.comp α β) (CommRingCat.ofHom (algebraM …
    hbij : Function.Bijective ⇑(β.hom.rangeRestrict.comp α.hom)
    φ : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
    hαφ : Eq (CategoryTheory.CategoryStruct.comp α φ) (CategoryTheory.CategoryStru …
    hαφ' : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.st …
    hφβ : Eq (CategoryTheory.CategoryStruct.comp φ (CommRingCat.ofHom (algebraMap  …
    ⊢ ⋯.HasLift
  -/
  refine ⟨⟨⟨Spec.map ((pullback.snd i₂ f).stalkMap x ≫ φ) ≫ X.fromSpecStalk _, ?_, ?_⟩⟩⟩
    /-
      case mk.mk.intro.intro.refine_1
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      R : Type u
      commRing✝ : CommRing R
      domain✝ : IsDomain R
      valuationRing✝ : ValuationRing R
      K : Type u
      field✝ : Field K
      algebra✝ : Algebra R K
      isFractionRing✝ : IsFractionRing R K
      i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
      i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
      w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
      this✝³ : IsDomain ↑(CommRingCat.of R)
      this✝² : ValuationRing ↑(CommRingCat.of R)
      this✝¹ : Field ↑(CommRingCat.of K) := field✝
      H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
      lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
      x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
      h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
      h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
      e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
      α : Quiver.Hom (CommRingCat.of R) ((CategoryTheory.Limits.pullback i₂ f).presh …
      this✝ : IsLocalHom e.hom.hom
      this : IsLocalHom α.hom
      β : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
      hαβ : Eq (CategoryTheory.CategoryStruct.comp α β) (CommRingCat.ofHom (algebraM …
      hbij : Function.Bijective ⇑(β.hom.rangeRestrict.comp α.hom)
      φ : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
      hαφ : Eq (CategoryTheory.CategoryStruct.comp α φ) (CategoryTheory.CategoryStru …
      hαφ' : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.st …
      hφβ : Eq (CategoryTheory.CategoryStruct.comp φ (CommRingCat.ofHom (algebraMap  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (CommRing …
    -/
  · simp only [← Spec.map_comp_assoc, Category.assoc, hφβ]
    simp only [Spec.map_comp, Category.assoc, Scheme.Spec_map_stalkMap_fromSpecStalk,
      Scheme.Spec_map_stalkSpecializes_fromSpecStalk_assoc, β]
    -- This next line only fires as `rw`, not `simp`:
    /-
      case mk.mk.intro.intro.refine_1
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      R : Type u
      commRing✝ : CommRing R
      domain✝ : IsDomain R
      valuationRing✝ : ValuationRing R
      K : Type u
      field✝ : Field K
      algebra✝ : Algebra R K
      isFractionRing✝ : IsFractionRing R K
      i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
      i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
      w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
      this✝³ : IsDomain ↑(CommRingCat.of R)
      this✝² : ValuationRing ↑(CommRingCat.of R)
      this✝¹ : Field ↑(CommRingCat.of K) := field✝
      H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
      lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
      x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
      h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
      h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
      e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
      α : Quiver.Hom (CommRingCat.of R) ((CategoryTheory.Limits.pullback i₂ f).presh …
      this✝ : IsLocalHom e.hom.hom
      this : IsLocalHom α.hom
      β : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
      hαβ : Eq (CategoryTheory.CategoryStruct.comp α β) (CommRingCat.ofHom (algebraM …
      hbij : Function.Bijective ⇑(β.hom.rangeRestrict.comp α.hom)
      φ : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
      hαφ : Eq (CategoryTheory.CategoryStruct.comp α φ) (CategoryTheory.CategoryStru …
      hαφ' : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.st …
      hφβ : Eq (CategoryTheory.CategoryStruct.comp φ (CommRingCat.ofHom (algebraMap  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Algebrai …
    -/
    rw [Scheme.Spec_stalkClosedPointTo_fromSpecStalk_assoc]
    /-
      case mk.mk.intro.intro.refine_1
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      R : Type u
      commRing✝ : CommRing R
      domain✝ : IsDomain R
      valuationRing✝ : ValuationRing R
      K : Type u
      field✝ : Field K
      algebra✝ : Algebra R K
      isFractionRing✝ : IsFractionRing R K
      i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
      i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
      w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
      this✝³ : IsDomain ↑(CommRingCat.of R)
      this✝² : ValuationRing ↑(CommRingCat.of R)
      this✝¹ : Field ↑(CommRingCat.of K) := field✝
      H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
      lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
      x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
      h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
      h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
      e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
      α : Quiver.Hom (CommRingCat.of R) ((CategoryTheory.Limits.pullback i₂ f).presh …
      this✝ : IsLocalHom e.hom.hom
      this : IsLocalHom α.hom
      β : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
      hαβ : Eq (CategoryTheory.CategoryStruct.comp α β) (CommRingCat.ofHom (algebraM …
      hbij : Function.Bijective ⇑(β.hom.rangeRestrict.comp α.hom)
      φ : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
      hαφ : Eq (CategoryTheory.CategoryStruct.comp α φ) (CategoryTheory.CategoryStru …
      hαφ' : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.st …
      hφβ : Eq (CategoryTheory.CategoryStruct.comp φ (CommRingCat.ofHom (algebraMap  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp lft (CategoryTheory.Limits.pullback.s …
    -/
    simp [lft]
    /-
      🎉 no goals
    -/
  · simp only [Spec.map_comp, Category.assoc, Scheme.Spec_map_stalkMap_fromSpecStalk,
      ← pullback.condition]
    /-
      case mk.mk.intro.intro.refine_2
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      R : Type u
      commRing✝ : CommRing R
      domain✝ : IsDomain R
      valuationRing✝ : ValuationRing R
      K : Type u
      field✝ : Field K
      algebra✝ : Algebra R K
      isFractionRing✝ : IsFractionRing R K
      i₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) X
      i₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of R)) Y
      w : Eq (CategoryTheory.CategoryStruct.comp i₁ f) (CategoryTheory.CategoryStruc …
      this✝³ : IsDomain ↑(CommRingCat.of R)
      this✝² : ValuationRing ↑(CommRingCat.of R)
      this✝¹ : Field ↑(CommRingCat.of K) := field✝
      H : AlgebraicGeometry.topologically (@SpecializingMap) (CategoryTheory.Limits. …
      lft : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of K)) (CategoryTheory.L …
      x : ↑↑(CategoryTheory.Limits.pullback i₂ f).toPresheafedSpace
      h₁ : flip (fun x1 x2 => Specializes x1 x2) x (lft.base (IsLocalRing.closedPoin …
      h₂ : Eq ((CategoryTheory.Limits.pullback.fst i₂ f).base x) (IsLocalRing.closed …
      e : CategoryTheory.Iso (CommRingCat.of R) ((AlgebraicGeometry.Spec (CommRingCa …
      α : Quiver.Hom (CommRingCat.of R) ((CategoryTheory.Limits.pullback i₂ f).presh …
      this✝ : IsLocalHom e.hom.hom
      this : IsLocalHom α.hom
      β : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
      hαβ : Eq (CategoryTheory.CategoryStruct.comp α β) (CommRingCat.ofHom (algebraM …
      hbij : Function.Bijective ⇑(β.hom.rangeRestrict.comp α.hom)
      φ : Quiver.Hom ((CategoryTheory.Limits.pullback i₂ f).presheaf.stalk x) (CommR …
      hαφ : Eq (CategoryTheory.CategoryStruct.comp α φ) (CategoryTheory.CategoryStru …
      hαφ' : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.st …
      hφβ : Eq (CategoryTheory.CategoryStruct.comp φ (CommRingCat.ofHom (algebraMap  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map φ) (Categ …
    -/
    rw [← Scheme.Spec_map_stalkMap_fromSpecStalk_assoc, ← Spec.map_comp_assoc, hαφ']
    simp only [Iso.trans_inv, TopCat.Presheaf.stalkCongr_inv, Iso.symm_inv, Spec.map_comp,
      Category.assoc, Scheme.Spec_map_stalkSpecializes_fromSpecStalk_assoc, e]
    rw [← Spec_stalkClosedPointIso, ← Spec.map_comp_assoc,
      Iso.inv_hom_id, Spec.map_id, Category.id_comp]


instance stableUnderBaseChange : ValuativeCriterion.Existence.IsStableUnderBaseChange := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.ValuativeCriterion.Existence.IsStableUnderBaseChange
  -/
  constructor
  /-
    case of_isPullback
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ ∀ {X Y Y' S : AlgebraicGeometry.Scheme} {f : Quiver.Hom X S} {g : Quiver.Hom …
  -/
  intros Y' X X' Y  Y'_to_Y f X'_to_X f' hP hf commSq
  let commSq' : ValuativeCommSq f :=
  { R := commSq.R
    K := commSq.K
    i₁ := commSq.i₁ ≫ X'_to_X
    i₂ := commSq.i₂ ≫ Y'_to_Y
    commSq := ⟨by simp only [Category.assoc, hP.w, reassoc_of% commSq.commSq.w]⟩ }
  /-
    case of_isPullback
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    Y' X X' Y : AlgebraicGeometry.Scheme
    Y'_to_Y : Quiver.Hom Y' Y
    f : Quiver.Hom X Y
    X'_to_X : Quiver.Hom X' X
    f' : Quiver.Hom X' Y'
    hP : CategoryTheory.IsPullback X'_to_X f' f Y'_to_Y
    hf : AlgebraicGeometry.ValuativeCriterion.Existence f
    commSq : AlgebraicGeometry.ValuativeCommSq f'
    commSq' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCo …
    ⊢ ⋯.HasLift
  -/
  obtain ⟨l₀, hl₁, hl₂⟩ := (hf commSq').exists_lift
  /-
    case of_isPullback.intro.mk
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    Y' X X' Y : AlgebraicGeometry.Scheme
    Y'_to_Y : Quiver.Hom Y' Y
    f : Quiver.Hom X Y
    X'_to_X : Quiver.Hom X' X
    f' : Quiver.Hom X' Y'
    hP : CategoryTheory.IsPullback X'_to_X f' f Y'_to_Y
    hf : AlgebraicGeometry.ValuativeCriterion.Existence f
    commSq : AlgebraicGeometry.ValuativeCommSq f'
    commSq' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCo …
    l₀ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of commSq'.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₀ f) commSq'.i₂
    ⊢ ⋯.HasLift
  -/
  refine ⟨⟨⟨hP.lift l₀ commSq.i₂ (by simp_all only [commSq']), ?_, hP.lift_snd _ _ _⟩⟩⟩
  /-
    case of_isPullback.intro.mk
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    Y' X X' Y : AlgebraicGeometry.Scheme
    Y'_to_Y : Quiver.Hom Y' Y
    f : Quiver.Hom X Y
    X'_to_X : Quiver.Hom X' X
    f' : Quiver.Hom X' Y'
    hP : CategoryTheory.IsPullback X'_to_X f' f Y'_to_Y
    hf : AlgebraicGeometry.ValuativeCriterion.Existence f
    commSq : AlgebraicGeometry.ValuativeCommSq f'
    commSq' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCo …
    l₀ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of commSq'.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₀ f) commSq'.i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (CommRing …
  -/
  apply hP.hom_ext
    /-
      case of_isPullback.intro.mk.h₀
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      Y' X X' Y : AlgebraicGeometry.Scheme
      Y'_to_Y : Quiver.Hom Y' Y
      f : Quiver.Hom X Y
      X'_to_X : Quiver.Hom X' X
      f' : Quiver.Hom X' Y'
      hP : CategoryTheory.IsPullback X'_to_X f' f Y'_to_Y
      hf : AlgebraicGeometry.ValuativeCriterion.Existence f
      commSq : AlgebraicGeometry.ValuativeCommSq f'
      commSq' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCo …
      l₀ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of commSq'.R)) X
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₀ f) commSq'.i₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case of_isPullback.intro.mk.h₁
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      Y' X X' Y : AlgebraicGeometry.Scheme
      Y'_to_Y : Quiver.Hom Y' Y
      f : Quiver.Hom X Y
      X'_to_X : Quiver.Hom X' X
      f' : Quiver.Hom X' Y'
      hP : CategoryTheory.IsPullback X'_to_X f' f Y'_to_Y
      hf : AlgebraicGeometry.ValuativeCriterion.Existence f
      commSq : AlgebraicGeometry.ValuativeCommSq f'
      commSq' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCo …
      l₀ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of commSq'.R)) X
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₀ f) commSq'.i₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp only [Category.assoc]
    /-
      case of_isPullback.intro.mk.h₁
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      Y' X X' Y : AlgebraicGeometry.Scheme
      Y'_to_Y : Quiver.Hom Y' Y
      f : Quiver.Hom X Y
      X'_to_X : Quiver.Hom X' X
      f' : Quiver.Hom X' Y'
      hP : CategoryTheory.IsPullback X'_to_X f' f Y'_to_Y
      hf : AlgebraicGeometry.ValuativeCriterion.Existence f
      commSq : AlgebraicGeometry.ValuativeCommSq f'
      commSq' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCo …
      l₀ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of commSq'.R)) X
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₀ f) commSq'.i₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (CommRing …
    -/
    rw [hP.lift_snd]
    /-
      case of_isPullback.intro.mk.h₁
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      Y' X X' Y : AlgebraicGeometry.Scheme
      Y'_to_Y : Quiver.Hom Y' Y
      f : Quiver.Hom X Y
      X'_to_X : Quiver.Hom X' X
      f' : Quiver.Hom X' Y'
      hP : CategoryTheory.IsPullback X'_to_X f' f Y'_to_Y
      hf : AlgebraicGeometry.ValuativeCriterion.Existence f
      commSq : AlgebraicGeometry.ValuativeCommSq f'
      commSq' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCo …
      l₀ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of commSq'.R)) X
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp l₀ f) commSq'.i₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (CommRing …
    -/
    rw [commSq.commSq.w]
    /-
      🎉 no goals
    -/


@[stacks 01KE]
protected lemma eq :
    ValuativeCriterion.Existence = (topologically @SpecializingMap).universally := by
  /-
    ⊢ Eq AlgebraicGeometry.ValuativeCriterion.Existence (AlgebraicGeometry.topolog …
  -/
  ext
  /-
    case h
    X✝ Y✝ : AlgebraicGeometry.Scheme
    f✝ : Quiver.Hom X✝ Y✝
    ⊢ Iff (AlgebraicGeometry.ValuativeCriterion.Existence f✝) ((AlgebraicGeometry. …
  -/
  constructor
    /-
      case h.mp
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ AlgebraicGeometry.ValuativeCriterion.Existence f✝ → (AlgebraicGeometry.topol …
    -/
  · intro _
    /-
      case h.mp
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      a✝ : AlgebraicGeometry.ValuativeCriterion.Existence f✝
      ⊢ (AlgebraicGeometry.topologically @SpecializingMap).universally f✝
    -/
    apply MorphismProperty.universally_mono
      /-
        case h.mp.a
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        a✝ : AlgebraicGeometry.ValuativeCriterion.Existence f✝
        ⊢ LE.le ?h.mp.a (AlgebraicGeometry.topologically @SpecializingMap)
      -/
    · apply specializingMap
      /-
        🎉 no goals
      -/
      /-
        case h.mp.a
        X✝ Y✝ : AlgebraicGeometry.Scheme
        f✝ : Quiver.Hom X✝ Y✝
        a✝ : AlgebraicGeometry.ValuativeCriterion.Existence f✝
        ⊢ AlgebraicGeometry.ValuativeCriterion.Existence.universally f✝
      -/
    · rwa [MorphismProperty.IsStableUnderBaseChange.universally_eq]
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ (AlgebraicGeometry.topologically @SpecializingMap).universally f✝ → Algebrai …
    -/
  · apply of_specializingMap
    /-
      🎉 no goals
    -/


/-- The **valuative criterion** for universally closed morphisms. -/
@[stacks 01KF]
lemma UniversallyClosed.eq_valuativeCriterion :
    @UniversallyClosed = ValuativeCriterion.Existence ⊓ @QuasiCompact := by
  /-
    ⊢ Eq (@AlgebraicGeometry.UniversallyClosed) (Min.min AlgebraicGeometry.Valuati …
  -/
  rw [universallyClosed_eq_universallySpecializing, ValuativeCriterion.Existence.eq]
  /-
    🎉 no goals
  -/


/-- The **valuative criterion** for universally closed morphisms. -/
@[stacks 01KF]
lemma UniversallyClosed.of_valuativeCriterion [QuasiCompact f]
    (hf : ValuativeCriterion.Existence f) : UniversallyClosed f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.QuasiCompact f
    hf : AlgebraicGeometry.ValuativeCriterion.Existence f
    ⊢ AlgebraicGeometry.UniversallyClosed f
  -/
  rw [eq_valuativeCriterion]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.QuasiCompact f
    hf : AlgebraicGeometry.ValuativeCriterion.Existence f
    ⊢ Min.min AlgebraicGeometry.ValuativeCriterion.Existence (@AlgebraicGeometry.Q …
  -/
  exact ⟨hf, ‹_›⟩
  /-
    🎉 no goals
  -/


/-- The **valuative criterion** for separated morphisms. -/
@[stacks 01L0]
lemma IsSeparated.of_valuativeCriterion [QuasiSeparated f]
    (hf : ValuativeCriterion.Uniqueness f) : IsSeparated f where
  diagonal_isClosedImmersion := by
    suffices h : ValuativeCriterion.Existence (pullback.diagonal f) by
      have : QuasiCompact (pullback.diagonal f) :=
        AlgebraicGeometry.QuasiSeparated.diagonalQuasiCompact
      apply IsClosedImmersion.of_isPreimmersion
      apply IsClosedMap.isClosed_range
      apply (topologically @IsClosedMap).universally_le
      exact (UniversallyClosed.of_valuativeCriterion (pullback.diagonal f) h).out
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiSeparated f
      hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
      ⊢ AlgebraicGeometry.ValuativeCriterion.Existence (CategoryTheory.Limits.pullba …
    -/
    intro S
    have hc : CommSq S.i₁ (Spec.map (CommRingCat.ofHom (algebraMap S.R S.K)))
        f (S.i₂ ≫ pullback.fst f f ≫ f) := ⟨by simp [← S.commSq.w_assoc]⟩
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiSeparated f
      hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
      S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
      hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
      ⊢ ⋯.HasLift
    -/
    let S' : ValuativeCommSq f := ⟨S.R, S.K, S.i₁, S.i₂ ≫ pullback.fst f f ≫ f, hc⟩
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiSeparated f
      hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
      S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
      hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
      S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
      ⊢ ⋯.HasLift
    -/
    have : Subsingleton S'.commSq.LiftStruct := hf S'
    let S'l₁ : S'.commSq.LiftStruct := ⟨S.i₂ ≫ pullback.fst f f,
      by simp [S', ← S.commSq.w_assoc], by simp [S']⟩
    let S'l₂ : S'.commSq.LiftStruct := ⟨S.i₂ ≫ pullback.snd f f,
      by simp [S', ← S.commSq.w_assoc], by simp [S', pullback.condition]⟩
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiSeparated f
      hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
      S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
      hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
      S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
      this : Subsingleton ⋯.LiftStruct
      S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
      S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
      ⊢ ⋯.HasLift
    -/
    have h₁₂ : S'l₁ = S'l₂ := Subsingleton.elim _ _
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiSeparated f
      hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
      S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
      hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
      S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
      this : Subsingleton ⋯.LiftStruct
      S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
      S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
      h₁₂ : Eq S'l₁ S'l₂
      ⊢ ⋯.HasLift
    -/
    constructor
    /-
      case exists_lift
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiSeparated f
      hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
      S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
      hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
      S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
      this : Subsingleton ⋯.LiftStruct
      S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
      S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
      h₁₂ : Eq S'l₁ S'l₂
      ⊢ Nonempty ⋯.LiftStruct
    -/
    constructor
    /-
      case exists_lift.val
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.QuasiSeparated f
      hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
      S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
      hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
      S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
      this : Subsingleton ⋯.LiftStruct
      S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
      S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
      h₁₂ : Eq S'l₁ S'l₂
      ⊢ ⋯.LiftStruct
    -/
    refine ⟨S.i₂ ≫ pullback.fst _ _, ?_, ?_⟩
      /-
        case exists_lift.val.refine_1
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        inst✝ : AlgebraicGeometry.QuasiSeparated f
        hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
        S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
        hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
        S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
        this : Subsingleton ⋯.LiftStruct
        S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
        S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
        h₁₂ : Eq S'l₁ S'l₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (CommRing …
      -/
    · simp [← S.commSq.w_assoc]
      /-
        🎉 no goals
      -/
      /-
        case exists_lift.val.refine_2
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        inst✝ : AlgebraicGeometry.QuasiSeparated f
        hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
        S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
        hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
        S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
        this : Subsingleton ⋯.LiftStruct
        S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
        S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
        h₁₂ : Eq S'l₁ S'l₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
      -/
    · simp
      /-
        case exists_lift.val.refine_2
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        inst✝ : AlgebraicGeometry.QuasiSeparated f
        hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
        S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
        hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
        S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
        this : Subsingleton ⋯.LiftStruct
        S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
        S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
        h₁₂ : Eq S'l₁ S'l₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp S.i₂ (CategoryTheory.CategoryStruct.c …
      -/
      apply IsPullback.hom_ext (IsPullback.of_hasPullback _ _)
        /-
          case exists_lift.val.refine_2.h₀
          X Y : AlgebraicGeometry.Scheme
          f : Quiver.Hom X Y
          inst✝ : AlgebraicGeometry.QuasiSeparated f
          hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
          S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
          hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
          S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
          this : Subsingleton ⋯.LiftStruct
          S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
          S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
          h₁₂ : Eq S'l₁ S'l₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case exists_lift.val.refine_2.h₁
          X Y : AlgebraicGeometry.Scheme
          f : Quiver.Hom X Y
          inst✝ : AlgebraicGeometry.QuasiSeparated f
          hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
          S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
          hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
          S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
          this : Subsingleton ⋯.LiftStruct
          S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
          S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
          h₁₂ : Eq S'l₁ S'l₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
        -/
      · simp only [Category.assoc, pullback.diagonal_snd, Category.comp_id]
        /-
          case exists_lift.val.refine_2.h₁
          X Y : AlgebraicGeometry.Scheme
          f : Quiver.Hom X Y
          inst✝ : AlgebraicGeometry.QuasiSeparated f
          hf : AlgebraicGeometry.ValuativeCriterion.Uniqueness f
          S : AlgebraicGeometry.ValuativeCommSq (CategoryTheory.Limits.pullback.diagonal …
          hc : CategoryTheory.CommSq S.i₁ (AlgebraicGeometry.Spec.map (CommRingCat.ofHom …
          S' : AlgebraicGeometry.ValuativeCommSq f := AlgebraicGeometry.ValuativeCommSq. …
          this : Subsingleton ⋯.LiftStruct
          S'l₁ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
          S'l₂ : ⋯.LiftStruct := { l := CategoryTheory.CategoryStruct.comp S.i₂ (Categor …
          h₁₂ : Eq S'l₁ S'l₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp S.i₂ (CategoryTheory.Limits.pullback. …
        -/
        exact congrArg CommSq.LiftStruct.l h₁₂
        /-
          🎉 no goals
        -/


@[stacks 01KZ]
lemma IsSeparated.valuativeCriterion [IsSeparated f] : ValuativeCriterion.Uniqueness f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    ⊢ AlgebraicGeometry.ValuativeCriterion.Uniqueness f
  -/
  intros S
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    ⊢ Subsingleton ⋯.LiftStruct
  -/
  constructor
  /-
    case allEq
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    ⊢ ∀ (a b : ⋯.LiftStruct), Eq a b
  -/
  rintro ⟨l₁, hl₁, hl₁'⟩ ⟨l₂, hl₂, hl₂'⟩
  /-
    case allEq.mk.mk
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
    l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
    ⊢ Eq { l := l₁, fac_left := hl₁, fac_right := hl₁' } { l := l₂, fac_left := hl …
  -/
  ext : 1
  /-
    case allEq.mk.mk.l
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
    l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
    ⊢ Eq { l := l₁, fac_left := hl₁, fac_right := hl₁' }.l { l := l₂, fac_left :=  …
  -/
  dsimp at *
  /-
    case allEq.mk.mk.l
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
    l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
    ⊢ Eq l₁ l₂
  -/
  have h := hl₁'.trans hl₂'.symm
  /-
    case allEq.mk.mk.l
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
    l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
    h : Eq (CategoryTheory.CategoryStruct.comp l₁ f) (CategoryTheory.CategoryStruc …
    ⊢ Eq l₁ l₂
  -/
  let Z := pullback (pullback.diagonal f) (pullback.lift l₁ l₂ h)
  /-
    case allEq.mk.mk.l
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
    l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
    h : Eq (CategoryTheory.CategoryStruct.comp l₁ f) (CategoryTheory.CategoryStruc …
    Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback (CategoryTheory …
    ⊢ Eq l₁ l₂
  -/
  let g : Z ⟶ Spec (.of S.R) := pullback.snd _ _
  /-
    case allEq.mk.mk.l
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
    l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
    h : Eq (CategoryTheory.CategoryStruct.comp l₁ f) (CategoryTheory.CategoryStruc …
    Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback (CategoryTheory …
    g : Quiver.Hom Z (AlgebraicGeometry.Spec (CommRingCat.of S.R)) := CategoryTheo …
    ⊢ Eq l₁ l₂
  -/
  have : IsClosedImmersion g := MorphismProperty.pullback_snd _ _ inferInstance
  have hZ : IsAffine Z := by
    rw [@HasAffineProperty.iff_of_isAffine @IsClosedImmersion] at this
    exact this.left
  suffices IsIso g by
    rw [← cancel_epi g]
    conv_lhs => rw [← pullback.lift_fst l₁ l₂ h, ← pullback.condition_assoc]
    conv_rhs => rw [← pullback.lift_snd l₁ l₂ h, ← pullback.condition_assoc]
    simp
  suffices h : Function.Bijective (g.appTop) by
    refine (HasAffineProperty.iff_of_isAffine (P := MorphismProperty.isomorphisms Scheme)).mpr ?_
    exact ⟨hZ, (ConcreteCategory.isIso_iff_bijective _).mpr h⟩
  /-
    case allEq.mk.mk.l
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsSeparated f
    S : AlgebraicGeometry.ValuativeCommSq f
    l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
    l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
    hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
    hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
    h : Eq (CategoryTheory.CategoryStruct.comp l₁ f) (CategoryTheory.CategoryStruc …
    Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback (CategoryTheory …
    g : Quiver.Hom Z (AlgebraicGeometry.Spec (CommRingCat.of S.R)) := CategoryTheo …
    this : AlgebraicGeometry.IsClosedImmersion g
    hZ : AlgebraicGeometry.IsAffine Z
    ⊢ Function.Bijective ⇑(AlgebraicGeometry.Scheme.Hom.appTop g).hom
  -/
  constructor
  · let l : Spec (.of S.K) ⟶ Z := by
      apply pullback.lift S.i₁ (Spec.map (CommRingCat.ofHom (algebraMap S.R S.K)))
      apply IsPullback.hom_ext (IsPullback.of_hasPullback _ _)
      simpa using hl₁.symm
      simpa using hl₂.symm
    have hg : l ≫ g = Spec.map (CommRingCat.ofHom (algebraMap S.R S.K)) :=
      pullback.lift_snd _ _ _
    have : Function.Injective ((l ≫ g).appTop) := by
      rw [hg]
      let e := arrowIsoΓSpecOfIsAffine (CommRingCat.ofHom <| algebraMap S.R S.K)
      let P : MorphismProperty CommRingCat :=
        RingHom.toMorphismProperty <| fun f ↦ Function.Injective f
      have : (RingHom.toMorphismProperty <| fun f ↦ Function.Injective f).RespectsIso :=
        RingHom.toMorphismProperty_respectsIso_iff.mp RingHom.injective_respectsIso
      show P _
      rw [← MorphismProperty.arrow_mk_iso_iff (P := P) e]
      exact NoZeroSMulDivisors.algebraMap_injective S.R S.K
    /-
      case allEq.mk.mk.l.left
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsSeparated f
      S : AlgebraicGeometry.ValuativeCommSq f
      l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
      l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
      h : Eq (CategoryTheory.CategoryStruct.comp l₁ f) (CategoryTheory.CategoryStruc …
      Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback (CategoryTheory …
      g : Quiver.Hom Z (AlgebraicGeometry.Spec (CommRingCat.of S.R)) := CategoryTheo …
      this✝ : AlgebraicGeometry.IsClosedImmersion g
      hZ : AlgebraicGeometry.IsAffine Z
      l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.K)) Z := CategoryTheo …
      hg : Eq (CategoryTheory.CategoryStruct.comp l g) (AlgebraicGeometry.Spec.map ( …
      this : Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop (CategoryTheor …
      ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop g).hom
    -/
    rw [Scheme.comp_appTop] at this
    /-
      case allEq.mk.mk.l.left
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsSeparated f
      S : AlgebraicGeometry.ValuativeCommSq f
      l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
      l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
      h : Eq (CategoryTheory.CategoryStruct.comp l₁ f) (CategoryTheory.CategoryStruc …
      Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback (CategoryTheory …
      g : Quiver.Hom Z (AlgebraicGeometry.Spec (CommRingCat.of S.R)) := CategoryTheo …
      this✝ : AlgebraicGeometry.IsClosedImmersion g
      hZ : AlgebraicGeometry.IsAffine Z
      l : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.K)) Z := CategoryTheo …
      hg : Eq (CategoryTheory.CategoryStruct.comp l g) (AlgebraicGeometry.Spec.map ( …
      this : Function.Injective ⇑(CategoryTheory.CategoryStruct.comp (AlgebraicGeome …
      ⊢ Function.Injective ⇑(AlgebraicGeometry.Scheme.Hom.appTop g).hom
    -/
    exact Function.Injective.of_comp this
    /-
      🎉 no goals
    -/
    /-
      case allEq.mk.mk.l.right
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsSeparated f
      S : AlgebraicGeometry.ValuativeCommSq f
      l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
      l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
      h : Eq (CategoryTheory.CategoryStruct.comp l₁ f) (CategoryTheory.CategoryStruc …
      Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback (CategoryTheory …
      g : Quiver.Hom Z (AlgebraicGeometry.Spec (CommRingCat.of S.R)) := CategoryTheo …
      this : AlgebraicGeometry.IsClosedImmersion g
      hZ : AlgebraicGeometry.IsAffine Z
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop g).hom
    -/
  · rw [@HasAffineProperty.iff_of_isAffine @IsClosedImmersion] at this
    /-
      case allEq.mk.mk.l.right
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsSeparated f
      S : AlgebraicGeometry.ValuativeCommSq f
      l₁ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
      hl₁ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₁' : Eq (CategoryTheory.CategoryStruct.comp l₁ f) S.i₂
      l₂ : Quiver.Hom (AlgebraicGeometry.Spec (CommRingCat.of S.R)) X
      hl₂ : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Spec.map (Comm …
      hl₂' : Eq (CategoryTheory.CategoryStruct.comp l₂ f) S.i₂
      h : Eq (CategoryTheory.CategoryStruct.comp l₁ f) (CategoryTheory.CategoryStruc …
      Z : AlgebraicGeometry.Scheme := CategoryTheory.Limits.pullback (CategoryTheory …
      g : Quiver.Hom Z (AlgebraicGeometry.Spec (CommRingCat.of S.R)) := CategoryTheo …
      this : And (AlgebraicGeometry.IsAffine Z) (Function.Surjective ⇑(AlgebraicGeom …
      hZ : AlgebraicGeometry.IsAffine Z
      ⊢ Function.Surjective ⇑(AlgebraicGeometry.Scheme.Hom.appTop g).hom
    -/
    exact this.right
    /-
      🎉 no goals
    -/


/-- The **valuative criterion** for separated morphisms. -/
lemma IsSeparated.eq_valuativeCriterion :
    @IsSeparated = ValuativeCriterion.Uniqueness ⊓ @QuasiSeparated := by
  /-
    ⊢ Eq (@AlgebraicGeometry.IsSeparated) (Min.min AlgebraicGeometry.ValuativeCrit …
  -/
  ext X Y f
  exact ⟨fun _ ↦ ⟨IsSeparated.valuativeCriterion f, inferInstance⟩,
    fun ⟨H, _⟩ ↦ .of_valuativeCriterion f H⟩


/-- The **valuative criterion** for proper morphisms. -/
@[stacks 0BX5]
lemma IsProper.eq_valuativeCriterion :
    @IsProper = ValuativeCriterion ⊓ @QuasiCompact ⊓ @QuasiSeparated ⊓ @LocallyOfFiniteType := by
  rw [isProper_eq, IsSeparated.eq_valuativeCriterion, ValuativeCriterion.eq,
    UniversallyClosed.eq_valuativeCriterion]
  /-
    ⊢ Eq (Min.min (Min.min (Min.min AlgebraicGeometry.ValuativeCriterion.Uniquenes …
  -/
  simp_rw [inf_assoc]
  /-
    ⊢ Eq (Min.min AlgebraicGeometry.ValuativeCriterion.Uniqueness (Min.min (@Algeb …
  -/
  ext X Y f
  /-
    case h.h.h.a
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (Min.min AlgebraicGeometry.ValuativeCriterion.Uniqueness (Min.min (@Alge …
  -/
  show _ ∧ _ ∧ _ ∧ _ ∧ _ ↔ _ ∧ _ ∧ _ ∧ _ ∧ _
  /-
    case h.h.h.a
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ Iff (And (AlgebraicGeometry.ValuativeCriterion.Uniqueness f) (And (Algebraic …
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- The **valuative criterion** for proper morphisms. -/
@[stacks 0BX5]
lemma IsProper.of_valuativeCriterion [QuasiCompact f] [QuasiSeparated f] [LocallyOfFiniteType f]
    (H : ValuativeCriterion f) : IsProper f := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.QuasiCompact f
    inst✝¹ : AlgebraicGeometry.QuasiSeparated f
    inst✝ : AlgebraicGeometry.LocallyOfFiniteType f
    H : AlgebraicGeometry.ValuativeCriterion f
    ⊢ AlgebraicGeometry.IsProper f
  -/
  rw [eq_valuativeCriterion]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝² : AlgebraicGeometry.QuasiCompact f
    inst✝¹ : AlgebraicGeometry.QuasiSeparated f
    inst✝ : AlgebraicGeometry.LocallyOfFiniteType f
    H : AlgebraicGeometry.ValuativeCriterion f
    ⊢ Min.min (Min.min (Min.min AlgebraicGeometry.ValuativeCriterion @AlgebraicGeo …
  -/
  exact ⟨⟨⟨‹_›, ‹_›⟩, ‹_›⟩, ‹_›⟩
  /-
    🎉 no goals
  -/


