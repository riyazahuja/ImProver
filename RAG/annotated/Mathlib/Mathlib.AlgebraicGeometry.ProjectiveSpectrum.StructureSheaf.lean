local notation3 "at " x =>
  HomogeneousLocalization.AtPrime 𝒜
    (HomogeneousIdeal.toIdeal (ProjectiveSpectrum.asHomogeneousIdeal x))


/-- The predicate saying that a dependent function on an open `U` is realised as a fixed fraction
`r / s` of *same grading* in each of the stalks (which are localizations at various prime ideals).
-/
def IsFraction {U : Opens (ProjectiveSpectrum.top 𝒜)} (f : ∀ x : U, at x.1) : Prop :=
  ∃ (i : ℕ) (r s : 𝒜 i) (s_nin : ∀ x : U, s.1 ∉ x.1.asHomogeneousIdeal),
    ∀ x : U, f x = .mk ⟨i, r, s, s_nin x⟩

/--
The predicate `IsFraction` is "prelocal", in the sense that if it holds on `U` it holds on any open
subset `V` of `U`.
-/
def isFractionPrelocal : PrelocalPredicate fun x : ProjectiveSpectrum.top 𝒜 => at x where
  pred f := IsFraction f
            /-
              R : Type u_1
              A : Type u_2
              inst✝³ : CommRing R
              inst✝² : CommRing A
              inst✝¹ : Algebra R A
              𝒜 : Nat → Submodule R A
              inst✝ : GradedAlgebra 𝒜
              ⊢ ∀ {U V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)} (i : Quiver.Hom …
            -/
  res := by rintro V U i f ⟨j, r, s, h, w⟩; exact ⟨j, r, s, (h <| i ·), (w <| i ·)⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- We will define the structure sheaf as the subsheaf of all dependent functions in
`Π x : U, HomogeneousLocalization 𝒜 x` consisting of those functions which can locally be expressed
as a ratio of `A` of same grading. -/
def isLocallyFraction : LocalPredicate fun x : ProjectiveSpectrum.top 𝒜 => at x :=
  (isFractionPrelocal 𝒜).sheafify


theorem zero_mem' (U : (Opens (ProjectiveSpectrum.top 𝒜))ᵒᵖ) :
    (isLocallyFraction 𝒜).pred (0 : ∀ x : U.unop, at x.1) := fun x =>
  ⟨unop U, x.2, 𝟙 (unop U), ⟨0, ⟨0, zero_mem _⟩, ⟨1, one_mem_graded _⟩, _, fun _ => rfl⟩⟩


theorem one_mem' (U : (Opens (ProjectiveSpectrum.top 𝒜))ᵒᵖ) :
    (isLocallyFraction 𝒜).pred (1 : ∀ x : U.unop, at x.1) := fun x =>
  ⟨unop U, x.2, 𝟙 (unop U), ⟨0, ⟨1, one_mem_graded _⟩, ⟨1, one_mem_graded _⟩, _, fun _ => rfl⟩⟩


theorem add_mem' (U : (Opens (ProjectiveSpectrum.top 𝒜))ᵒᵖ) (a b : ∀ x : U.unop, at x.1)
    (ha : (isLocallyFraction 𝒜).pred a) (hb : (isLocallyFraction 𝒜).pred b) :
    (isLocallyFraction 𝒜).pred (a + b) := fun x => by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousL …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    hb : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Projectiv …
  -/
  rcases ha x with ⟨Va, ma, ia, ja, ⟨ra, ra_mem⟩, ⟨sa, sa_mem⟩, hwa, wa⟩
  /-
    case intro.intro.intro.intro.intro.mk.intro.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousL …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    hb : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    Va : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    ma : Membership.mem Va ↑x
    ia : Quiver.Hom Va (Opposite.unop U)
    ja : Nat
    ra : A
    ra_mem : Membership.mem (𝒜 ja) ra
    sa : A
    sa_mem : Membership.mem (𝒜 ja) sa
    hwa : ∀ (x : Subtype fun x => Membership.mem Va x), Not (Membership.mem (↑x).a …
    wa : ∀ (x : Subtype fun x => Membership.mem Va x), Eq ((fun x => a ((fun x =>  …
    ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Projectiv …
  -/
  rcases hb x with ⟨Vb, mb, ib, jb, ⟨rb, rb_mem⟩, ⟨sb, sb_mem⟩, hwb, wb⟩
  refine
    ⟨Va ⊓ Vb, ⟨ma, mb⟩, Opens.infLELeft _ _ ≫ ia, ja + jb,
      ⟨sb * ra + sa * rb,
        add_mem (add_comm jb ja ▸ mul_mem_graded sb_mem ra_mem : sb * ra ∈ 𝒜 (ja + jb))
          (mul_mem_graded sa_mem rb_mem)⟩,
      ⟨sa * sb, mul_mem_graded sa_mem sb_mem⟩, fun y ↦
        y.1.asHomogeneousIdeal.toIdeal.primeCompl.mul_mem (hwa ⟨y.1, y.2.1⟩) (hwb ⟨y.1, y.2.2⟩),
      fun y => ?_⟩
  /-
    case intro.intro.intro.intro.intro.mk.intro.mk.intro.intro.intro.intro.intro.i …
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousL …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    hb : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    Va : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    ma : Membership.mem Va ↑x
    ia : Quiver.Hom Va (Opposite.unop U)
    ja : Nat
    ra : A
    ra_mem : Membership.mem (𝒜 ja) ra
    sa : A
    sa_mem : Membership.mem (𝒜 ja) sa
    hwa : ∀ (x : Subtype fun x => Membership.mem Va x), Not (Membership.mem (↑x).a …
    wa : ∀ (x : Subtype fun x => Membership.mem Va x), Eq ((fun x => a ((fun x =>  …
    Vb : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    mb : Membership.mem Vb ↑x
    ib : Quiver.Hom Vb (Opposite.unop U)
    jb : Nat
    rb : A
    rb_mem : Membership.mem (𝒜 jb) rb
    sb : A
    sb_mem : Membership.mem (𝒜 jb) sb
    hwb : ∀ (x : Subtype fun x => Membership.mem Vb x), Not (Membership.mem (↑x).a …
    wb : ∀ (x : Subtype fun x => Membership.mem Vb x), Eq ((fun x => b ((fun x =>  …
    y : Subtype fun x => Membership.mem (Min.min Va Vb) x
    ⊢ Eq ((fun x => HAdd.hAdd a b ((fun x => ⟨↑x, ⋯⟩) x)) y) (HomogeneousLocalizat …
  -/
  simp only at wa wb
  simp only [Pi.add_apply, wa ⟨y.1, y.2.1⟩, wb ⟨y.1, y.2.2⟩, ext_iff_val,
    val_add, val_mk, add_mk, add_comm (sa * rb)]
  /-
    case intro.intro.intro.intro.intro.mk.intro.mk.intro.intro.intro.intro.intro.i …
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousL …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    hb : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    Va : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    ma : Membership.mem Va ↑x
    ia : Quiver.Hom Va (Opposite.unop U)
    ja : Nat
    ra : A
    ra_mem : Membership.mem (𝒜 ja) ra
    sa : A
    sa_mem : Membership.mem (𝒜 ja) sa
    hwa : ∀ (x : Subtype fun x => Membership.mem Va x), Not (Membership.mem (↑x).a …
    wa : ∀ (x : Subtype fun x => Membership.mem Va x), Eq (a ⟨↑x, ⋯⟩) (Homogeneous …
    Vb : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    mb : Membership.mem Vb ↑x
    ib : Quiver.Hom Vb (Opposite.unop U)
    jb : Nat
    rb : A
    rb_mem : Membership.mem (𝒜 jb) rb
    sb : A
    sb_mem : Membership.mem (𝒜 jb) sb
    hwb : ∀ (x : Subtype fun x => Membership.mem Vb x), Not (Membership.mem (↑x).a …
    wb : ∀ (x : Subtype fun x => Membership.mem Vb x), Eq (b ⟨↑x, ⋯⟩) (Homogeneous …
    y : Subtype fun x => Membership.mem (Min.min Va Vb) x
    ⊢ Eq (Localization.mk (HAdd.hAdd (HMul.hMul sb ra) (HMul.hMul sa rb)) (HMul.hM …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem neg_mem' (U : (Opens (ProjectiveSpectrum.top 𝒜))ᵒᵖ) (a : ∀ x : U.unop, at x.1)
    (ha : (isLocallyFraction 𝒜).pred a) : (isLocallyFraction 𝒜).pred (-a) := fun x => by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousLoc …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Projectiv …
  -/
  rcases ha x with ⟨V, m, i, j, ⟨r, r_mem⟩, ⟨s, s_mem⟩, nin, hy⟩
  /-
    case intro.intro.intro.intro.intro.mk.intro.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousLoc …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    m : Membership.mem V ↑x
    i : Quiver.Hom V (Opposite.unop U)
    j : Nat
    r : A
    r_mem : Membership.mem (𝒜 j) r
    s : A
    s_mem : Membership.mem (𝒜 j) s
    nin : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).as …
    hy : ∀ (x : Subtype fun x => Membership.mem V x), Eq ((fun x => a ((fun x => ⟨ …
    ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Projectiv …
  -/
  refine ⟨V, m, i, j, ⟨-r, Submodule.neg_mem _ r_mem⟩, ⟨s, s_mem⟩, nin, fun y => ?_⟩
  /-
    case intro.intro.intro.intro.intro.mk.intro.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousLoc …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    m : Membership.mem V ↑x
    i : Quiver.Hom V (Opposite.unop U)
    j : Nat
    r : A
    r_mem : Membership.mem (𝒜 j) r
    s : A
    s_mem : Membership.mem (𝒜 j) s
    nin : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).as …
    hy : ∀ (x : Subtype fun x => Membership.mem V x), Eq ((fun x => a ((fun x => ⟨ …
    y : Subtype fun x => Membership.mem V x
    ⊢ Eq ((fun x => Neg.neg a ((fun x => ⟨↑x, ⋯⟩) x)) y) (HomogeneousLocalization. …
  -/
  simp only [ext_iff_val, val_mk] at hy
  /-
    case intro.intro.intro.intro.intro.mk.intro.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousLoc …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    m : Membership.mem V ↑x
    i : Quiver.Hom V (Opposite.unop U)
    j : Nat
    r : A
    r_mem : Membership.mem (𝒜 j) r
    s : A
    s_mem : Membership.mem (𝒜 j) s
    nin : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).as …
    y : Subtype fun x => Membership.mem V x
    hy : ∀ (x : Subtype fun x => Membership.mem V x), Eq (HomogeneousLocalization. …
    ⊢ Eq ((fun x => Neg.neg a ((fun x => ⟨↑x, ⋯⟩) x)) y) (HomogeneousLocalization. …
  -/
  simp only [Pi.neg_apply, ext_iff_val, val_neg, hy, val_mk, neg_mk]
  /-
    🎉 no goals
  -/


theorem mul_mem' (U : (Opens (ProjectiveSpectrum.top 𝒜))ᵒᵖ) (a b : ∀ x : U.unop, at x.1)
    (ha : (isLocallyFraction 𝒜).pred a) (hb : (isLocallyFraction 𝒜).pred b) :
    (isLocallyFraction 𝒜).pred (a * b) := fun x => by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousL …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    hb : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Projectiv …
  -/
  rcases ha x with ⟨Va, ma, ia, ja, ⟨ra, ra_mem⟩, ⟨sa, sa_mem⟩, hwa, wa⟩
  /-
    case intro.intro.intro.intro.intro.mk.intro.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousL …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    hb : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    Va : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    ma : Membership.mem Va ↑x
    ia : Quiver.Hom Va (Opposite.unop U)
    ja : Nat
    ra : A
    ra_mem : Membership.mem (𝒜 ja) ra
    sa : A
    sa_mem : Membership.mem (𝒜 ja) sa
    hwa : ∀ (x : Subtype fun x => Membership.mem Va x), Not (Membership.mem (↑x).a …
    wa : ∀ (x : Subtype fun x => Membership.mem Va x), Eq ((fun x => a ((fun x =>  …
    ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Projectiv …
  -/
  rcases hb x with ⟨Vb, mb, ib, jb, ⟨rb, rb_mem⟩, ⟨sb, sb_mem⟩, hwb, wb⟩
  refine
    ⟨Va ⊓ Vb, ⟨ma, mb⟩, Opens.infLELeft _ _ ≫ ia, ja + jb,
      ⟨ra * rb, SetLike.mul_mem_graded ra_mem rb_mem⟩,
      ⟨sa * sb, SetLike.mul_mem_graded sa_mem sb_mem⟩, fun y =>
      y.1.asHomogeneousIdeal.toIdeal.primeCompl.mul_mem (hwa ⟨y.1, y.2.1⟩) (hwb ⟨y.1, y.2.2⟩),
      fun y ↦ ?_⟩
  simp only [Pi.mul_apply, wa ⟨y.1, y.2.1⟩, wb ⟨y.1, y.2.2⟩, ext_iff_val, val_mul, val_mk,
    Localization.mk_mul]
  /-
    case intro.intro.intro.intro.intro.mk.intro.mk.intro.intro.intro.intro.intro.i …
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))
    a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → HomogeneousL …
    ha : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    hb : (AlgebraicGeometry.ProjectiveSpectrum.StructureSheaf.isLocallyFraction 𝒜) …
    x : Subtype fun x => Membership.mem (Opposite.unop U) x
    Va : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    ma : Membership.mem Va ↑x
    ia : Quiver.Hom Va (Opposite.unop U)
    ja : Nat
    ra : A
    ra_mem : Membership.mem (𝒜 ja) ra
    sa : A
    sa_mem : Membership.mem (𝒜 ja) sa
    hwa : ∀ (x : Subtype fun x => Membership.mem Va x), Not (Membership.mem (↑x).a …
    wa : ∀ (x : Subtype fun x => Membership.mem Va x), Eq ((fun x => a ((fun x =>  …
    Vb : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    mb : Membership.mem Vb ↑x
    ib : Quiver.Hom Vb (Opposite.unop U)
    jb : Nat
    rb : A
    rb_mem : Membership.mem (𝒜 jb) rb
    sb : A
    sb_mem : Membership.mem (𝒜 jb) sb
    hwb : ∀ (x : Subtype fun x => Membership.mem Vb x), Not (Membership.mem (↑x).a …
    wb : ∀ (x : Subtype fun x => Membership.mem Vb x), Eq ((fun x => b ((fun x =>  …
    y : Subtype fun x => Membership.mem (Min.min Va Vb) x
    ⊢ Eq (Localization.mk (HMul.hMul ra rb) (HMul.hMul ⟨sa, ⋯⟩ ⟨sb, ⋯⟩)) (Localiza …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The functions satisfying `isLocallyFraction` form a subring of all dependent functions
`Π x : U, HomogeneousLocalization 𝒜 x`. -/
def sectionsSubring (U : (Opens (ProjectiveSpectrum.top 𝒜))ᵒᵖ) :
    Subring (∀ x : U.unop, at x.1) where
  carrier := {f | (isLocallyFraction 𝒜).pred f}
  zero_mem' := zero_mem' U
  one_mem' := one_mem' U
  add_mem' := add_mem' U _ _
  neg_mem' := neg_mem' U _
  mul_mem' := mul_mem' U _ _


/-- The structure sheaf (valued in `Type`, not yet `CommRing`) is the subsheaf consisting of
functions satisfying `isLocallyFraction`. -/
def structureSheafInType : Sheaf (Type _) (ProjectiveSpectrum.top 𝒜) :=
  subsheafToTypes (isLocallyFraction 𝒜)


instance commRingStructureSheafInTypeObj (U : (Opens (ProjectiveSpectrum.top 𝒜))ᵒᵖ) :
    CommRing ((structureSheafInType 𝒜).1.obj U) :=
  (sectionsSubring U).toCommRing


/-- The structure presheaf, valued in `CommRing`, constructed by dressing up the `Type` valued
structure presheaf. -/
@[simps]
def structurePresheafInCommRing : Presheaf CommRingCat (ProjectiveSpectrum.top 𝒜) where
  obj U := CommRingCat.of ((structureSheafInType 𝒜).1.obj U)
  map i := CommRingCat.ofHom
    { toFun := (structureSheafInType 𝒜).1.map i
      map_zero' := rfl
      map_add' := fun _ _ => rfl
      map_one' := rfl
      map_mul' := fun _ _ => rfl }

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

/-- Some glue, verifying that the structure presheaf valued in `CommRing` agrees with the `Type`
valued structure presheaf. -/
def structurePresheafCompForget :
    structurePresheafInCommRing 𝒜 ⋙ forget CommRingCat ≅ (structureSheafInType 𝒜).1 :=
                                                /-
                                                  R : Type u_1
                                                  A : Type u_2
                                                  inst✝³ : CommRing R
                                                  inst✝² : CommRing A
                                                  inst✝¹ : Algebra R A
                                                  𝒜 : Nat → Submodule R A
                                                  inst✝ : GradedAlgebra 𝒜
                                                  ⊢ ∀ {X Y : Opposite (TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜))} (f : …
                                                -/
  NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                /-
                                                  🎉 no goals
                                                -/


/-- The structure sheaf on `Proj` 𝒜, valued in `CommRing`. -/
def Proj.structureSheaf : Sheaf CommRingCat (ProjectiveSpectrum.top 𝒜) :=
  ⟨structurePresheafInCommRing 𝒜,
    (-- We check the sheaf condition under `forget CommRing`.
          isSheaf_iff_isSheaf_comp
          _ _).mpr
      (isSheaf_of_iso (structurePresheafCompForget 𝒜).symm (structureSheafInType 𝒜).cond)⟩


@[simp]
theorem Proj.res_apply (x) : ((Proj.structureSheaf 𝒜).1.map i s).1 x = s.1 (i.unop x) := rfl


@[ext] theorem Proj.ext (h : s.1 = t.1) : s = t := Subtype.ext h

@[simp] theorem Proj.add_apply : (s + t).1 x = s.1 x + t.1 x := rfl

@[simp] theorem Proj.mul_apply : (s * t).1 x = s.1 x * t.1 x := rfl

@[simp] theorem Proj.sub_apply : (s - t).1 x = s.1 x - t.1 x := rfl

@[simp] theorem Proj.pow_apply (n : ℕ) : (s ^ n).1 x = s.1 x ^ n := rfl

@[simp] theorem Proj.zero_apply : (0 : (Proj.structureSheaf 𝒜).1.obj V).1 x = 0 := rfl

@[simp] theorem Proj.one_apply : (1 : (Proj.structureSheaf 𝒜).1.obj V).1 x = 1 := rfl


/-- `Proj` of a graded ring as a `SheafedSpace`-/
def Proj.toSheafedSpace : SheafedSpace CommRingCat where
  carrier := TopCat.of (ProjectiveSpectrum 𝒜)
  presheaf := (Proj.structureSheaf 𝒜).1
  IsSheaf := (Proj.structureSheaf 𝒜).2


/-- The ring homomorphism that takes a section of the structure sheaf of `Proj` on the open set `U`,
implemented as a subtype of dependent functions to localizations at homogeneous prime ideals, and
evaluates the section on the point corresponding to a given homogeneous prime ideal. -/
def openToLocalization (U : Opens (ProjectiveSpectrum.top 𝒜)) (x : ProjectiveSpectrum.top 𝒜)
    (hx : x ∈ U) : (Proj.structureSheaf 𝒜).1.obj (op U) ⟶ CommRingCat.of (at x) :=
  CommRingCat.ofHom
  { toFun s := (s.1 ⟨x, hx⟩ : _)
    map_one' := rfl
    map_mul' _ _ := rfl
    map_zero' := rfl
    map_add' _ _ := rfl }


/-- The ring homomorphism from the stalk of the structure sheaf of `Proj` at a point corresponding
to a homogeneous prime ideal `x` to the *homogeneous localization* at `x`,
formed by gluing the `openToLocalization` maps. -/
def stalkToFiberRingHom (x : ProjectiveSpectrum.top 𝒜) :
    (Proj.structureSheaf 𝒜).presheaf.stalk x ⟶ CommRingCat.of (at x) :=
  Limits.colimit.desc ((OpenNhds.inclusion x).op ⋙ (Proj.structureSheaf 𝒜).1)
    { pt := _
      ι :=
        { app := fun U =>
            openToLocalization 𝒜 ((OpenNhds.inclusion _).obj U.unop) x U.unop.2 } }


@[simp]
theorem germ_comp_stalkToFiberRingHom
    (U : Opens (ProjectiveSpectrum.top 𝒜)) (x : ProjectiveSpectrum.top 𝒜) (hx : x ∈ U) :
    (Proj.structureSheaf 𝒜).presheaf.germ U x hx ≫ stalkToFiberRingHom 𝒜 x =
      openToLocalization 𝒜 U x hx :=
  Limits.colimit.ι_desc _ _


@[simp]
theorem stalkToFiberRingHom_germ (U : Opens (ProjectiveSpectrum.top 𝒜))
    (x : ProjectiveSpectrum.top 𝒜) (hx : x ∈ U) (s : (Proj.structureSheaf 𝒜).1.obj (op U)) :
    stalkToFiberRingHom 𝒜 x ((Proj.structureSheaf 𝒜).presheaf.germ _ x hx s) = s.1 ⟨x, hx⟩ :=
  RingHom.ext_iff.1 (CommRingCat.hom_ext_iff.mp (germ_comp_stalkToFiberRingHom 𝒜 U x hx)) s


@[deprecated (since := "2024-07-30")] alias stalkToFiberRingHom_germ' := stalkToFiberRingHom_germ


theorem mem_basicOpen_den (x : ProjectiveSpectrum.top 𝒜)
    (f : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.primeCompl) :
    x ∈ ProjectiveSpectrum.basicOpen 𝒜 f.den := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.prime …
    ⊢ Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 ↑f.den) x
  -/
  rw [ProjectiveSpectrum.mem_basicOpen]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    f : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.prime …
    ⊢ Not (Membership.mem x.asHomogeneousIdeal ↑f.den)
  -/
  exact f.den_mem
  /-
    🎉 no goals
  -/


/-- Given a point `x` corresponding to a homogeneous prime ideal, there is a (dependent) function
such that, for any `f` in the homogeneous localization at `x`, it returns the obvious section in the
basic open set `D(f.den)`-/
def sectionInBasicOpen (x : ProjectiveSpectrum.top 𝒜) :
    ∀ f : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.primeCompl,
    (Proj.structureSheaf 𝒜).1.obj (op (ProjectiveSpectrum.basicOpen 𝒜 f.den)) :=
  fun f =>
  ⟨fun y => HomogeneousLocalization.mk ⟨f.deg, f.num, f.den, y.2⟩, fun y =>
    ⟨ProjectiveSpectrum.basicOpen 𝒜 f.den, y.2,
      ⟨𝟙 _, ⟨f.deg, ⟨f.num, f.den, _, fun _ => rfl⟩⟩⟩⟩⟩


open HomogeneousLocalization in
/-- Given any point `x` and `f` in the homogeneous localization at `x`, there is an element in the
stalk at `x` obtained by `sectionInBasicOpen`. This is the inverse of `stalkToFiberRingHom`.
-/
def homogeneousLocalizationToStalk (x : ProjectiveSpectrum.top 𝒜) (y : at x) :
    (Proj.structureSheaf 𝒜).presheaf.stalk x := Quotient.liftOn' y (fun f =>
  (Proj.structureSheaf 𝒜).presheaf.germ _ x (mem_basicOpen_den _ x f) (sectionInBasicOpen _ x f))
  fun f g (e : f.embedding = g.embedding) ↦ by
    simp only [HomogeneousLocalization.NumDenSameDeg.embedding, Localization.mk_eq_mk',
      IsLocalization.mk'_eq_iff_eq,
      IsLocalization.eq_iff_exists x.asHomogeneousIdeal.toIdeal.primeCompl] at e
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      x : ↑(ProjectiveSpectrum.top 𝒜)
      y : HomogeneousLocalization.AtPrime 𝒜 x.asHomogeneousIdeal.toIdeal
      f g : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.pri …
      e : Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul ↑g.den ↑f.num)) (HMul.hMul ( …
      ⊢ Eq ((fun f => ((AlgebraicGeometry.ProjectiveSpectrum.Proj.structureSheaf 𝒜). …
    -/
    obtain ⟨⟨c, hc⟩, hc'⟩ := e
    apply (Proj.structureSheaf 𝒜).presheaf.germ_ext
      (ProjectiveSpectrum.basicOpen 𝒜 f.den.1 ⊓
        ProjectiveSpectrum.basicOpen 𝒜 g.den.1 ⊓ ProjectiveSpectrum.basicOpen 𝒜 c)
      ⟨⟨mem_basicOpen_den _ x f, mem_basicOpen_den _ x g⟩, hc⟩
      (homOfLE inf_le_left ≫ homOfLE inf_le_left) (homOfLE inf_le_left ≫ homOfLE inf_le_right)
    -- Go from `ConcreteCategory.instFunLike` to `CommRingCat.Hom.hom`
    show (Proj.structureSheaf 𝒜).presheaf.map (homOfLE inf_le_left ≫ homOfLE inf_le_left).op
        (sectionInBasicOpen 𝒜 x f) =
      (Proj.structureSheaf 𝒜).presheaf.map (homOfLE inf_le_left ≫ homOfLE inf_le_right).op
        (sectionInBasicOpen 𝒜 x g)
    /-
      case intro.mk
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      x : ↑(ProjectiveSpectrum.top 𝒜)
      y : HomogeneousLocalization.AtPrime 𝒜 x.asHomogeneousIdeal.toIdeal
      f g : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.pri …
      c : A
      hc : Membership.mem x.asHomogeneousIdeal.toIdeal.primeCompl c
      hc' : Eq (HMul.hMul (↑⟨c, hc⟩) (HMul.hMul ↑g.den ↑f.num)) (HMul.hMul (↑⟨c, hc⟩ …
      ⊢ Eq (((AlgebraicGeometry.ProjectiveSpectrum.Proj.structureSheaf 𝒜).presheaf.m …
    -/
    apply Subtype.ext
    /-
      case intro.mk.a
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      x : ↑(ProjectiveSpectrum.top 𝒜)
      y : HomogeneousLocalization.AtPrime 𝒜 x.asHomogeneousIdeal.toIdeal
      f g : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.pri …
      c : A
      hc : Membership.mem x.asHomogeneousIdeal.toIdeal.primeCompl c
      hc' : Eq (HMul.hMul (↑⟨c, hc⟩) (HMul.hMul ↑g.den ↑f.num)) (HMul.hMul (↑⟨c, hc⟩ …
      ⊢ Eq ↑(((AlgebraicGeometry.ProjectiveSpectrum.Proj.structureSheaf 𝒜).presheaf. …
    -/
    ext ⟨t, ⟨htf, htg⟩, ht'⟩
    /-
      case intro.mk.a.h.mk.intro.intro.a
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      x : ↑(ProjectiveSpectrum.top 𝒜)
      y : HomogeneousLocalization.AtPrime 𝒜 x.asHomogeneousIdeal.toIdeal
      f g : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.pri …
      c : A
      hc : Membership.mem x.asHomogeneousIdeal.toIdeal.primeCompl c
      hc' : Eq (HMul.hMul (↑⟨c, hc⟩) (HMul.hMul ↑g.den ↑f.num)) (HMul.hMul (↑⟨c, hc⟩ …
      t : ↑(ProjectiveSpectrum.top 𝒜)
      ht' : Membership.mem (↑(ProjectiveSpectrum.basicOpen 𝒜 c)) t
      htf : Membership.mem (↑(ProjectiveSpectrum.basicOpen 𝒜 ↑f.den)) t
      htg : Membership.mem (↑(ProjectiveSpectrum.basicOpen 𝒜 ↑g.den)) t
      ⊢ Eq (HomogeneousLocalization.val (↑(((AlgebraicGeometry.ProjectiveSpectrum.Pr …
    -/
    rw [Proj.res_apply, Proj.res_apply]
    simp only [sectionInBasicOpen, HomogeneousLocalization.val_mk, Localization.mk_eq_mk',
      IsLocalization.mk'_eq_iff_eq]
    apply (IsLocalization.map_units (M := t.asHomogeneousIdeal.toIdeal.primeCompl)
      (Localization t.asHomogeneousIdeal.toIdeal.primeCompl) ⟨c, ht'⟩).mul_left_cancel
    /-
      case intro.mk.a.h.mk.intro.intro.a
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      x : ↑(ProjectiveSpectrum.top 𝒜)
      y : HomogeneousLocalization.AtPrime 𝒜 x.asHomogeneousIdeal.toIdeal
      f g : HomogeneousLocalization.NumDenSameDeg 𝒜 x.asHomogeneousIdeal.toIdeal.pri …
      c : A
      hc : Membership.mem x.asHomogeneousIdeal.toIdeal.primeCompl c
      hc' : Eq (HMul.hMul (↑⟨c, hc⟩) (HMul.hMul ↑g.den ↑f.num)) (HMul.hMul (↑⟨c, hc⟩ …
      t : ↑(ProjectiveSpectrum.top 𝒜)
      ht' : Membership.mem (↑(ProjectiveSpectrum.basicOpen 𝒜 c)) t
      htf : Membership.mem (↑(ProjectiveSpectrum.basicOpen 𝒜 ↑f.den)) t
      htg : Membership.mem (↑(ProjectiveSpectrum.basicOpen 𝒜 ↑g.den)) t
      ⊢ Eq (HMul.hMul ((algebraMap A (Localization t.asHomogeneousIdeal.toIdeal.prim …
    -/
    rw [← map_mul, ← map_mul, hc']
    /-
      🎉 no goals
    -/


lemma homogeneousLocalizationToStalk_stalkToFiberRingHom (x z) :
    homogeneousLocalizationToStalk 𝒜 x (stalkToFiberRingHom 𝒜 x z) = z := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    z : ↑((AlgebraicGeometry.ProjectiveSpectrum.Proj.structureSheaf 𝒜).presheaf.st …
    ⊢ Eq (AlgebraicGeometry.homogeneousLocalizationToStalk 𝒜 x ((AlgebraicGeometry …
  -/
  obtain ⟨U, hxU, s, rfl⟩ := (Proj.structureSheaf 𝒜).presheaf.germ_exist x z
  show homogeneousLocalizationToStalk 𝒜 x ((stalkToFiberRingHom 𝒜 x).hom
      (((Proj.structureSheaf 𝒜).presheaf.germ U x hxU) s)) =
    ((Proj.structureSheaf 𝒜).presheaf.germ U x hxU) s
  /-
    case intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    U : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj ((AlgebraicGeometry.ProjectiveSpec …
    ⊢ Eq (AlgebraicGeometry.homogeneousLocalizationToStalk 𝒜 x ((AlgebraicGeometry …
  -/
  obtain ⟨V, hxV, i, n, a, b, h, e⟩ := s.2 ⟨x, hxU⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    U : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj ((AlgebraicGeometry.ProjectiveSpec …
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxV : Membership.mem V ↑⟨x, hxU⟩
    i : Quiver.Hom V (Opposite.unop { unop := U })
    n : Nat
    a b : Subtype fun x => Membership.mem (𝒜 n) x
    h : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).asHo …
    e : ∀ (x : Subtype fun x => Membership.mem V x), Eq ((fun x => ↑s ((fun x => ⟨ …
    ⊢ Eq (AlgebraicGeometry.homogeneousLocalizationToStalk 𝒜 x ((AlgebraicGeometry …
  -/
  simp only at e
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    U : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj ((AlgebraicGeometry.ProjectiveSpec …
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxV : Membership.mem V ↑⟨x, hxU⟩
    i : Quiver.Hom V (Opposite.unop { unop := U })
    n : Nat
    a b : Subtype fun x => Membership.mem (𝒜 n) x
    h : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).asHo …
    e : ∀ (x : Subtype fun x => Membership.mem V x), Eq (↑s ⟨↑x, ⋯⟩) (HomogeneousL …
    ⊢ Eq (AlgebraicGeometry.homogeneousLocalizationToStalk 𝒜 x ((AlgebraicGeometry …
  -/
  rw [stalkToFiberRingHom_germ, homogeneousLocalizationToStalk, e ⟨x, hxV⟩, Quotient.liftOn'_mk'']
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    U : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj ((AlgebraicGeometry.ProjectiveSpec …
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxV : Membership.mem V ↑⟨x, hxU⟩
    i : Quiver.Hom V (Opposite.unop { unop := U })
    n : Nat
    a b : Subtype fun x => Membership.mem (𝒜 n) x
    h : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).asHo …
    e : ∀ (x : Subtype fun x => Membership.mem V x), Eq (↑s ⟨↑x, ⋯⟩) (HomogeneousL …
    ⊢ Eq (((AlgebraicGeometry.ProjectiveSpectrum.Proj.structureSheaf 𝒜).presheaf.g …
  -/
  refine Presheaf.germ_ext (C := CommRingCat) _ V hxV (homOfLE <| fun _ h' ↦ h ⟨_, h'⟩) i ?_
  change ((Proj.structureSheaf 𝒜).presheaf.map (homOfLE <| fun _ h' ↦ h ⟨_, h'⟩).op) _ =
    ((Proj.structureSheaf 𝒜).presheaf.map i.op) s
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    U : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj ((AlgebraicGeometry.ProjectiveSpec …
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxV : Membership.mem V ↑⟨x, hxU⟩
    i : Quiver.Hom V (Opposite.unop { unop := U })
    n : Nat
    a b : Subtype fun x => Membership.mem (𝒜 n) x
    h : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).asHo …
    e : ∀ (x : Subtype fun x => Membership.mem V x), Eq (↑s ⟨↑x, ⋯⟩) (HomogeneousL …
    ⊢ Eq (((AlgebraicGeometry.ProjectiveSpectrum.Proj.structureSheaf 𝒜).presheaf.m …
  -/
  apply Subtype.ext
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    U : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj ((AlgebraicGeometry.ProjectiveSpec …
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxV : Membership.mem V ↑⟨x, hxU⟩
    i : Quiver.Hom V (Opposite.unop { unop := U })
    n : Nat
    a b : Subtype fun x => Membership.mem (𝒜 n) x
    h : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).asHo …
    e : ∀ (x : Subtype fun x => Membership.mem V x), Eq (↑s ⟨↑x, ⋯⟩) (HomogeneousL …
    ⊢ Eq ↑(((AlgebraicGeometry.ProjectiveSpectrum.Proj.structureSheaf 𝒜).presheaf. …
  -/
  ext ⟨t, ht⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.a.h.mk.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    U : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj ((AlgebraicGeometry.ProjectiveSpec …
    V : TopologicalSpace.Opens ↑(ProjectiveSpectrum.top 𝒜)
    hxV : Membership.mem V ↑⟨x, hxU⟩
    i : Quiver.Hom V (Opposite.unop { unop := U })
    n : Nat
    a b : Subtype fun x => Membership.mem (𝒜 n) x
    h : ∀ (x : Subtype fun x => Membership.mem V x), Not (Membership.mem (↑x).asHo …
    e : ∀ (x : Subtype fun x => Membership.mem V x), Eq (↑s ⟨↑x, ⋯⟩) (HomogeneousL …
    t : ↑(ProjectiveSpectrum.top 𝒜)
    ht : Membership.mem (Opposite.unop { unop := V }) t
    ⊢ Eq (HomogeneousLocalization.val (↑(((AlgebraicGeometry.ProjectiveSpectrum.Pr …
  -/
  rw [Proj.res_apply, Proj.res_apply]
  simp only [sectionInBasicOpen, HomogeneousLocalization.val_mk, Localization.mk_eq_mk',
    IsLocalization.mk'_eq_iff_eq, e ⟨t, ht⟩]


lemma stalkToFiberRingHom_homogeneousLocalizationToStalk (x z) :
    stalkToFiberRingHom 𝒜 x (homogeneousLocalizationToStalk 𝒜 x z) = z := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ↑(ProjectiveSpectrum.top 𝒜)
    z : HomogeneousLocalization.AtPrime 𝒜 x.asHomogeneousIdeal.toIdeal
    ⊢ Eq ((AlgebraicGeometry.stalkToFiberRingHom 𝒜 x).hom (AlgebraicGeometry.homog …
  -/
  obtain ⟨z, rfl⟩ := Quotient.mk''_surjective z
  rw [homogeneousLocalizationToStalk, Quotient.liftOn'_mk'',
    stalkToFiberRingHom_germ, sectionInBasicOpen]


/-- Using `homogeneousLocalizationToStalk`, we construct a ring isomorphism between stalk at `x`
and homogeneous localization at `x` for any point `x` in `Proj`. -/
def Proj.stalkIso' (x : ProjectiveSpectrum.top 𝒜) :
    (Proj.structureSheaf 𝒜).presheaf.stalk x ≃+* at x where
  __ := (stalkToFiberRingHom _ x).hom
  invFun := homogeneousLocalizationToStalk 𝒜 x
  left_inv := homogeneousLocalizationToStalk_stalkToFiberRingHom 𝒜 x
  right_inv := stalkToFiberRingHom_homogeneousLocalizationToStalk 𝒜 x


@[simp]
theorem Proj.stalkIso'_germ (U : Opens (ProjectiveSpectrum.top 𝒜))
    (x : ProjectiveSpectrum.top 𝒜) (hx : x ∈ U) (s : (Proj.structureSheaf 𝒜).1.obj (op U)) :
    Proj.stalkIso' 𝒜 x ((Proj.structureSheaf 𝒜).presheaf.germ _ x hx s) = s.1 ⟨x, hx⟩ :=
  stalkToFiberRingHom_germ 𝒜 U x hx s


@[deprecated (since := "2024-07-30")] alias Proj.stalkIso'_germ' := Proj.stalkIso'_germ


@[simp]
theorem Proj.stalkIso'_symm_mk (x) (f) :
    (Proj.stalkIso' 𝒜 x).symm (.mk f) = (Proj.structureSheaf 𝒜).presheaf.germ _
      x (mem_basicOpen_den _ x f) (sectionInBasicOpen _ x f) := rfl


/-- `Proj` of a graded ring as a `LocallyRingedSpace`-/
def Proj.toLocallyRingedSpace : LocallyRingedSpace :=
  { Proj.toSheafedSpace 𝒜 with
    isLocalRing := fun x =>
      @RingEquiv.isLocalRing _ _ _ (show IsLocalRing (at x) from inferInstance) _
        (Proj.stalkIso' 𝒜 x).symm }


