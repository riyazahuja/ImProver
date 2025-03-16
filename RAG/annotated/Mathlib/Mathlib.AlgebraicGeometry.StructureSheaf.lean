/-- The prime spectrum, just as a topological space.
-/
def PrimeSpectrum.Top : TopCat :=
  TopCat.of (PrimeSpectrum R)


/-- The type family over `PrimeSpectrum R` consisting of the localization over each point.
-/
def Localizations (P : PrimeSpectrum.Top R) : Type u :=
  Localization.AtPrime P.asIdeal

-- Porting note: can't derive `CommRingCat`

instance commRingLocalizations (P : PrimeSpectrum.Top R) : CommRing <| Localizations R P :=
  inferInstanceAs <| CommRing <| Localization.AtPrime P.asIdeal

-- Porting note: can't derive `IsLocalRing`

instance localRingLocalizations (P : PrimeSpectrum.Top R) : IsLocalRing <| Localizations R P :=
  inferInstanceAs <| IsLocalRing <| Localization.AtPrime P.asIdeal


instance (P : PrimeSpectrum.Top R) : Inhabited (Localizations R P) :=
  ⟨1⟩


instance (U : Opens (PrimeSpectrum.Top R)) (x : U) : Algebra R (Localizations R x) :=
  inferInstanceAs <| Algebra R (Localization.AtPrime x.1.asIdeal)


instance (U : Opens (PrimeSpectrum.Top R)) (x : U) :
    IsLocalization.AtPrime (Localizations R x) (x : PrimeSpectrum.Top R).asIdeal :=
  Localization.isLocalization


/-- The predicate saying that a dependent function on an open `U` is realised as a fixed fraction
`r / s` in each of the stalks (which are localizations at various prime ideals).
-/
def IsFraction {U : Opens (PrimeSpectrum.Top R)} (f : ∀ x : U, Localizations R x) : Prop :=
  ∃ r s : R, ∀ x : U, ¬s ∈ x.1.asIdeal ∧ f x * algebraMap _ _ s = algebraMap _ _ r


theorem IsFraction.eq_mk' {U : Opens (PrimeSpectrum.Top R)} {f : ∀ x : U, Localizations R x}
    (hf : IsFraction f) :
    ∃ r s : R,
      ∀ x : U,
        ∃ hs : s ∉ x.1.asIdeal,
          f x =
            IsLocalization.mk' (Localization.AtPrime _) r
              (⟨s, hs⟩ : (x : PrimeSpectrum.Top R).asIdeal.primeCompl) := by
  /-
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hf : AlgebraicGeometry.StructureSheaf.IsFraction f
    ⊢ Exists fun r => Exists fun s => ∀ (x : Subtype fun x => Membership.mem U x), …
  -/
  rcases hf with ⟨r, s, h⟩
  /-
    case intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    r s : R
    h : ∀ (x : Subtype fun x => Membership.mem U x), And (Not (Membership.mem (↑x) …
    ⊢ Exists fun r => Exists fun s => ∀ (x : Subtype fun x => Membership.mem U x), …
  -/
  refine ⟨r, s, fun x => ⟨(h x).1, (IsLocalization.mk'_eq_iff_eq_mul.mpr ?_).symm⟩⟩
  /-
    case intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    r s : R
    h : ∀ (x : Subtype fun x => Membership.mem U x), And (Not (Membership.mem (↑x) …
    x : Subtype fun x => Membership.mem U x
    ⊢ Eq ((algebraMap R (AlgebraicGeometry.StructureSheaf.Localizations R ↑x)) r)  …
  -/
  exact (h x).2.symm
  /-
    🎉 no goals
  -/


/-- The predicate `IsFraction` is "prelocal",
in the sense that if it holds on `U` it holds on any open subset `V` of `U`.
-/
def isFractionPrelocal : PrelocalPredicate (Localizations R) where
  pred {_} f := IsFraction f
            /-
              R : Type u
              inst✝ : CommRing R
              ⊢ ∀ {U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)} (i …
            -/
  res := by rintro V U i f ⟨r, s, w⟩; exact ⟨r, s, fun x => w (i x)⟩
                                      /-
                                        🎉 no goals
                                      -/


/-- We will define the structure sheaf as
the subsheaf of all dependent functions in `Π x : U, Localizations R x`
consisting of those functions which can locally be expressed as a ratio of
(the images in the localization of) elements of `R`.

Quoting Hartshorne:

For an open set $U ⊆ Spec A$, we define $𝒪(U)$ to be the set of functions
$s : U → ⨆_{𝔭 ∈ U} A_𝔭$, such that $s(𝔭) ∈ A_𝔭$ for each $𝔭$,
and such that $s$ is locally a quotient of elements of $A$:
to be precise, we require that for each $𝔭 ∈ U$, there is a neighborhood $V$ of $𝔭$,
contained in $U$, and elements $a, f ∈ A$, such that for each $𝔮 ∈ V, f ∉ 𝔮$,
and $s(𝔮) = a/f$ in $A_𝔮$.

Now Hartshorne had the disadvantage of not knowing about dependent functions,
so we replace his circumlocution about functions into a disjoint union with
`Π x : U, Localizations x`.
-/
def isLocallyFraction : LocalPredicate (Localizations R) :=
  (isFractionPrelocal R).sheafify


@[simp]
theorem isLocallyFraction_pred {U : Opens (PrimeSpectrum.Top R)} (f : ∀ x : U, Localizations R x) :
    (isLocallyFraction R).pred f =
      ∀ x : U,
        ∃ (V : _) (_ : x.1 ∈ V) (i : V ⟶ U),
          ∃ r s : R,
            ∀ y : V, ¬s ∈ y.1.asIdeal ∧ f (i y : U) * algebraMap _ _ s = algebraMap _ _ r :=
  rfl


/-- The functions satisfying `isLocallyFraction` form a subring.
-/
def sectionsSubring (U : (Opens (PrimeSpectrum.Top R))ᵒᵖ) :
    Subring (∀ x : U.unop, Localizations R x) where
  carrier := { f | (isLocallyFraction R).pred f }
  zero_mem' := by
    /-
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      ⊢ Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSheaf …
    -/
    refine fun x => ⟨unop U, x.2, 𝟙 _, 0, 1, fun y => ⟨?_, ?_⟩⟩
      /-
        case refine_1
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        x y : Subtype fun x => Membership.mem (Opposite.unop U) x
        ⊢ Not (Membership.mem (↑y).asIdeal 1)
      -/
    · rw [← Ideal.ne_top_iff_one]; exact y.1.isPrime.1
    /-
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      ⊢ Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSheaf …
    -/
                                   /-
                                     🎉 no goals
                                   -/
      /-
        case refine_1
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        x y : Subtype fun x => Membership.mem (Opposite.unop U) x
        ⊢ Not (Membership.mem (↑y).asIdeal 1)
      -/
      /-
        case refine_2
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        x y : Subtype fun x => Membership.mem (Opposite.unop U) x
        ⊢ Eq (HMul.hMul ((fun x => 0 ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraMap R (Algebr …
      -/
                                   /-
                                     🎉 no goals
                                   -/
      /-
        case refine_2
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        x y : Subtype fun x => Membership.mem (Opposite.unop U) x
        ⊢ Eq (HMul.hMul ((fun x => 1 ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraMap R (Algebr …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    /-
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      ⊢ ∀ {a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → Algebra …
    -/
  one_mem' := by
    /-
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Structure …
    -/
    refine fun x => ⟨unop U, x.2, 𝟙 _, 1, 1, fun y => ⟨?_, ?_⟩⟩
    /-
      case intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Structure …
    -/
    · rw [← Ideal.ne_top_iff_one]; exact y.1.isPrime.1
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Structure …
    -/
    · simp
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      ⊢ ∀ (x : Subtype fun x => Membership.mem (Min.min Va Vb) x), And (Not (Members …
    -/
  add_mem' := by
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HMul.hMul ((f …
    -/
    intro a b ha hb x
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
      wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HMul.hMul ((f …
    -/
    rcases ha x with ⟨Va, ma, ia, ra, sa, wa⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
      wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
      nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
      wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HMul.hMul ((f …
    -/
    rcases hb x with ⟨Vb, mb, ib, rb, sb, wb⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.l …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))
      -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    refine ⟨Va ⊓ Vb, ⟨ma, mb⟩, Opens.infLELeft _ _ ≫ ia, ra * sb + rb * sa, sa * sb, ?_⟩
                                                  /-
                                                    🎉 no goals
                                                  -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Eq (HMul.hMul ((fun x => HAdd.hAdd a b ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraM …
      -/
    /-
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      ⊢ ∀ {a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → Algebra …
    -/
    intro y
    /-
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Structure …
    -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Eq (HAdd.hAdd (HMul.hMul (a ⟨↑y, ⋯⟩) (HMul.hMul ((algebraMap R (AlgebraicGeo …
      -/
    /-
      case intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Structure …
    -/
    rcases wa (Opens.infLELeft _ _ y) with ⟨nma, wa⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Structure …
    -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Eq (HAdd.hAdd (HMul.hMul (a ⟨↑y, ⋯⟩) (HMul.hMul ((algebraMap R (AlgebraicGeo …
      -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      ⊢ ∀ (x : Subtype fun x => Membership.mem (Min.min Va Vb) x), And (Not (Members …
    -/
    rcases wb (Opens.infLERight _ _ y) with ⟨nmb, wb⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HMul.hMul ((f …
    -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Eq (HAdd.hAdd (HMul.hMul (a ⟨↑y, ⋯⟩) (HMul.hMul ((algebraMap R (AlgebraicGeo …
      -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
      wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HMul.hMul ((f …
    -/
    fconstructor
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
      ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra sa : R
      wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb sb : R
      wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
      wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
      nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
      wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HMul.hMul ((f …
    -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        hb : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Eq (HMul.hMul ((algebraMap R (AlgebraicGeometry.StructureSheaf.Localizations …
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.l …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
        hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))
      -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    · intro H; cases y.1.isPrime.mem_or_mem H <;> contradiction
                                                  /-
                                                    🎉 no goals
                                                  -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
        hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Eq (HMul.hMul ((fun x => HMul.hMul a b ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraM …
      -/
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
        hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Eq (HMul.hMul (HMul.hMul (a ⟨↑y, ⋯⟩) (b ⟨↑y, ⋯⟩)) (HMul.hMul ((algebraMap R  …
      -/
    · simp only [add_mul, RingHom.map_add, Pi.add_apply, RingHom.map_mul]
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeo …
        ha : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
        hb : Membership.mem (setOf fun f => (AlgebraicGeometry.StructureSheaf.isLocall …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HMul.hMul ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y) …
        ⊢ Eq (HMul.hMul (HMul.hMul (a ⟨↑y, ⋯⟩) (b ⟨↑y, ⋯⟩)) (HMul.hMul ((algebraMap R  …
      -/
      rw [← wa, ← wb]
      /-
        🎉 no goals
      -/
      simp only [mul_assoc]
      congr 2
      rw [mul_comm]
  neg_mem' := by
    /-
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      ⊢ ∀ {x : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → Algebraic …
    -/
    intro a ha x
    /-
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Structure …
    -/
    rcases ha x with ⟨V, m, i, r, s, w⟩
    /-
      case intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      m : Membership.mem V ↑x
      i : Quiver.Hom V (Opposite.unop U)
      r s : R
      w : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x) …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Structure …
    -/
    refine ⟨V, m, i, -r, s, ?_⟩
    /-
      case intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      m : Membership.mem V ↑x
      i : Quiver.Hom V (Opposite.unop U)
      r s : R
      w : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x) …
      ⊢ ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x).a …
    -/
    intro y
    /-
      case intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      m : Membership.mem V ↑x
      i : Quiver.Hom V (Opposite.unop U)
      r s : R
      w : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x) …
      y : Subtype fun x => Membership.mem V x
      ⊢ And (Not (Membership.mem (↑y).asIdeal s)) (Eq (HMul.hMul ((fun x => Neg.neg  …
    -/
    rcases w y with ⟨nm, w⟩
    /-
      case intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
      ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      m : Membership.mem V ↑x
      i : Quiver.Hom V (Opposite.unop U)
      r s : R
      w✝ : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
      y : Subtype fun x => Membership.mem V x
      nm : Not (Membership.mem (↑y).asIdeal s)
      w : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraMap R (Alge …
      ⊢ And (Not (Membership.mem (↑y).asIdeal s)) (Eq (HMul.hMul ((fun x => Neg.neg  …
    -/
    fconstructor
      /-
        case intro.intro.intro.intro.intro.intro.left
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        m : Membership.mem V ↑x
        i : Quiver.Hom V (Opposite.unop U)
        r s : R
        w✝ : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
        y : Subtype fun x => Membership.mem V x
        nm : Not (Membership.mem (↑y).asIdeal s)
        w : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraMap R (Alge …
        ⊢ Not (Membership.mem (↑y).asIdeal s)
      -/
    · exact nm
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.right
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        m : Membership.mem V ↑x
        i : Quiver.Hom V (Opposite.unop U)
        r s : R
        w✝ : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
        y : Subtype fun x => Membership.mem V x
        nm : Not (Membership.mem (↑y).asIdeal s)
        w : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraMap R (Alge …
        ⊢ Eq (HMul.hMul ((fun x => Neg.neg a ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraMap R …
      -/
    · simp only [RingHom.map_neg, Pi.neg_apply]
      /-
        case intro.intro.intro.intro.intro.intro.right
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        m : Membership.mem V ↑x
        i : Quiver.Hom V (Opposite.unop U)
        r s : R
        w✝ : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
        y : Subtype fun x => Membership.mem V x
        nm : Not (Membership.mem (↑y).asIdeal s)
        w : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraMap R (Alge …
        ⊢ Eq (HMul.hMul (Neg.neg (a ⟨↑y, ⋯⟩)) ((algebraMap R (AlgebraicGeometry.Struct …
      -/
      rw [← w]
      /-
        case intro.intro.intro.intro.intro.intro.right
        R : Type u
        inst✝ : CommRing R
        U : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R))
        a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → AlgebraicGeome …
        ha : Membership.mem { carrier := setOf fun f => (AlgebraicGeometry.StructureSh …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        m : Membership.mem V ↑x
        i : Quiver.Hom V (Opposite.unop U)
        r s : R
        w✝ : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
        y : Subtype fun x => Membership.mem V x
        nm : Not (Membership.mem (↑y).asIdeal s)
        w : Eq (HMul.hMul ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) y) ((algebraMap R (Alge …
        ⊢ Eq (HMul.hMul (Neg.neg (a ⟨↑y, ⋯⟩)) ((algebraMap R (AlgebraicGeometry.Struct …
      -/
      simp only [neg_mul]
      /-
        🎉 no goals
      -/
  mul_mem' := by
    intro a b ha hb x
    rcases ha x with ⟨Va, ma, ia, ra, sa, wa⟩
    rcases hb x with ⟨Vb, mb, ib, rb, sb, wb⟩
    refine ⟨Va ⊓ Vb, ⟨ma, mb⟩, Opens.infLELeft _ _ ≫ ia, ra * rb, sa * sb, ?_⟩
    intro y
    rcases wa (Opens.infLELeft _ _ y) with ⟨nma, wa⟩
    rcases wb (Opens.infLERight _ _ y) with ⟨nmb, wb⟩
    fconstructor
    · intro H; cases y.1.isPrime.mem_or_mem H <;> contradiction
    · simp only [Pi.mul_apply, RingHom.map_mul]
      rw [← wa, ← wb]
      simp only [mul_left_comm, mul_assoc, mul_comm]


/-- The structure sheaf (valued in `Type`, not yet `CommRingCat`) is the subsheaf consisting of
functions satisfying `isLocallyFraction`.
-/
def structureSheafInType : Sheaf (Type u) (PrimeSpectrum.Top R) :=
  subsheafToTypes (isLocallyFraction R)


instance commRingStructureSheafInTypeObj (U : (Opens (PrimeSpectrum.Top R))ᵒᵖ) :
    CommRing ((structureSheafInType R).1.obj U) :=
  (sectionsSubring R U).toCommRing


/-- The structure presheaf, valued in `CommRingCat`, constructed by dressing up the `Type` valued
structure presheaf.
-/
@[simps]
def structurePresheafInCommRing : Presheaf CommRingCat (PrimeSpectrum.Top R) where
  obj U := CommRingCat.of ((structureSheafInType R).1.obj U)
  map {_ _} i := CommRingCat.ofHom
    { toFun := (structureSheafInType R).1.map i
      map_zero' := rfl
      map_add' := fun _ _ => rfl
      map_one' := rfl
      map_mul' := fun _ _ => rfl }

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

/-- Some glue, verifying that the structure presheaf valued in `CommRingCat` agrees
with the `Type` valued structure presheaf.
-/
def structurePresheafCompForget :
    structurePresheafInCommRing R ⋙ forget CommRingCat ≅ (structureSheafInType R).1 :=
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ ∀ {X Y : Opposite (TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum. …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The structure sheaf on $Spec R$, valued in `CommRingCat`.

This is provided as a bundled `SheafedSpace` as `Spec.SheafedSpace R` later.
-/
def Spec.structureSheaf : Sheaf CommRingCat (PrimeSpectrum.Top R) :=
  ⟨structurePresheafInCommRing R,
    (-- We check the sheaf condition under `forget CommRingCat`.
          isSheaf_iff_isSheaf_comp
          _ _).mpr
      (isSheaf_of_iso (structurePresheafCompForget R).symm (structureSheafInType R).cond)⟩


@[simp]
theorem res_apply (U V : Opens (PrimeSpectrum.Top R)) (i : V ⟶ U)
    (s : (structureSheaf R).1.obj (op U)) (x : V) :
    ((structureSheaf R).1.map i.op s).1 x = (s.1 (i x) : _) :=
  rfl

/-

Notation in this comment

X = Spec R
OX = structure sheaf

In the following we construct an isomorphism between OX_p and R_p given any point p corresponding
to a prime ideal in R.

We do this via 8 steps:

1. def const (f g : R) (V) (hv : V ≤ D_g) : OX(V) [for api]
2. def toOpen (U) : R ⟶ OX(U)
3. [2] def toStalk (p : Spec R) : R ⟶ OX_p
4. [2] def toBasicOpen (f : R) : R_f ⟶ OX(D_f)
5. [3] def localizationToStalk (p : Spec R) : R_p ⟶ OX_p
6. def openToLocalization (U) (p) (hp : p ∈ U) : OX(U) ⟶ R_p
7. [6] def stalkToFiberRingHom (p : Spec R) : OX_p ⟶ R_p
8. [5,7] def stalkIso (p : Spec R) : OX_p ≅ R_p

In the square brackets we list the dependencies of a construction on the previous steps.

-/

/-- The section of `structureSheaf R` on an open `U` sending each `x ∈ U` to the element
`f/g` in the localization of `R` at `x`. -/
def const (f g : R) (U : Opens (PrimeSpectrum.Top R))
    (hu : ∀ x ∈ U, g ∈ (x : PrimeSpectrum.Top R).asIdeal.primeCompl) :
    (structureSheaf R).1.obj (op U) :=
  ⟨fun x => IsLocalization.mk' _ f ⟨g, hu x x.2⟩, fun x =>
    ⟨U, x.2, 𝟙 _, f, g, fun y => ⟨hu y y.2, IsLocalization.mk'_spec _ _ _⟩⟩⟩


@[simp]
theorem const_apply (f g : R) (U : Opens (PrimeSpectrum.Top R))
    (hu : ∀ x ∈ U, g ∈ (x : PrimeSpectrum.Top R).asIdeal.primeCompl) (x : U) :
    (const R f g U hu).1 x =
      IsLocalization.mk' (Localization.AtPrime x.1.asIdeal) f ⟨g, hu x x.2⟩ :=
  rfl


theorem const_apply' (f g : R) (U : Opens (PrimeSpectrum.Top R))
    (hu : ∀ x ∈ U, g ∈ (x : PrimeSpectrum.Top R).asIdeal.primeCompl) (x : U)
    (hx : g ∈ (x : PrimeSpectrum.Top R).asIdeal.primeCompl) :
    (const R f g U hu).1 x = IsLocalization.mk' _ f ⟨g, hx⟩ :=
  rfl


theorem exists_const (U) (s : (structureSheaf R).1.obj (op U)) (x : PrimeSpectrum.Top R)
    (hx : x ∈ U) :
    ∃ (V : Opens (PrimeSpectrum.Top R)) (_ : x ∈ V) (i : V ⟶ U) (f g : R) (hg : _),
      const R f g V hg = (structureSheaf R).1.map i.op s :=
  let ⟨V, hxV, iVU, f, g, hfg⟩ := s.2 ⟨x, hx⟩
  ⟨V, hxV, iVU, f, g, fun y hyV => (hfg ⟨y, hyV⟩).1,
    Subtype.eq <| funext fun y => IsLocalization.mk'_eq_iff_eq_mul.2 <| Eq.symm <| (hfg y).2⟩


@[simp]
theorem res_const (f g : R) (U hu V hv i) :
    (structureSheaf R).1.map i (const R f g U hu) = const R f g V hv :=
  rfl


theorem res_const' (f g : R) (V hv) :
    (structureSheaf R).1.map (homOfLE hv).op (const R f g (PrimeSpectrum.basicOpen g) fun _ => id) =
      const R f g V hv :=
  rfl


theorem const_zero (f : R) (U hu) : const R 0 f U hu = 0 :=
  Subtype.eq <| funext fun x => IsLocalization.mk'_eq_iff_eq_mul.2 <| by
    /-
      R : Type u
      inst✝ : CommRing R
      f : R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hu : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → Me …
      x : Subtype fun x => Membership.mem (Opposite.unop { unop := U }) x
      ⊢ Eq ((algebraMap R (AlgebraicGeometry.StructureSheaf.Localizations R ↑x)) 0)  …
    -/
    rw [RingHom.map_zero]
    /-
      R : Type u
      inst✝ : CommRing R
      f : R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hu : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → Me …
      x : Subtype fun x => Membership.mem (Opposite.unop { unop := U }) x
      ⊢ Eq 0 (HMul.hMul (↑0 x) ((algebraMap R (AlgebraicGeometry.StructureSheaf.Loca …
    -/
    exact (mul_eq_zero_of_left rfl ((algebraMap R (Localizations R x)) _)).symm
    /-
      🎉 no goals
    -/


theorem const_self (f : R) (U hu) : const R f f U hu = 1 :=
  Subtype.eq <| funext fun _ => IsLocalization.mk'_self _ _


theorem const_one (U) : (const R 1 1 U fun _ _ => Submonoid.one_mem _) = 1 :=
  const_self R 1 U _


theorem const_add (f₁ f₂ g₁ g₂ : R) (U hu₁ hu₂) :
    const R f₁ g₁ U hu₁ + const R f₂ g₂ U hu₂ =
      const R (f₁ * g₂ + f₂ * g₁) (g₁ * g₂) U fun x hx =>
        Submonoid.mul_mem _ (hu₁ x hx) (hu₂ x hx) :=
  Subtype.eq <| funext fun x => Eq.symm <| IsLocalization.mk'_add _ _
    ⟨g₁, hu₁ x x.2⟩ ⟨g₂, hu₂ x x.2⟩


theorem const_mul (f₁ f₂ g₁ g₂ : R) (U hu₁ hu₂) :
    const R f₁ g₁ U hu₁ * const R f₂ g₂ U hu₂ =
      const R (f₁ * f₂) (g₁ * g₂) U fun x hx => Submonoid.mul_mem _ (hu₁ x hx) (hu₂ x hx) :=
  Subtype.eq <|
    funext fun x =>
      Eq.symm <| IsLocalization.mk'_mul _ f₁ f₂ ⟨g₁, hu₁ x x.2⟩ ⟨g₂, hu₂ x x.2⟩


theorem const_ext {f₁ f₂ g₁ g₂ : R} {U hu₁ hu₂} (h : f₁ * g₂ = f₂ * g₁) :
    const R f₁ g₁ U hu₁ = const R f₂ g₂ U hu₂ :=
  Subtype.eq <|
    funext fun x =>
                                      /-
                                        R : Type u
                                        inst✝ : CommRing R
                                        f₁ f₂ g₁ g₂ : R
                                        U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                        hu₁ : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → M …
                                        hu₂ : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → M …
                                        h : Eq (HMul.hMul f₁ g₂) (HMul.hMul f₂ g₁)
                                        x : Subtype fun x => Membership.mem (Opposite.unop { unop := U }) x
                                        ⊢ Eq (HMul.hMul (↑⟨g₁, ⋯⟩) f₂) (HMul.hMul (↑⟨g₂, ⋯⟩) f₁)
                                      -/
      IsLocalization.mk'_eq_of_eq (by rw [mul_comm, Subtype.coe_mk, ← h, mul_comm, Subtype.coe_mk])
                                      /-
                                        🎉 no goals
                                      -/


theorem const_congr {f₁ f₂ g₁ g₂ : R} {U hu} (hf : f₁ = f₂) (hg : g₁ = g₂) :
                                                         /-
                                                           R : Type u
                                                           inst✝ : CommRing R
                                                           f₁ f₂ g₁ g₂ : R
                                                           U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                           hu : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → Me …
                                                           hf : Eq f₁ f₂
                                                           hg : Eq g₁ g₂
                                                           ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R f₁ g₁ U hu) (AlgebraicGeometry. …
                                                         -/
    const R f₁ g₁ U hu = const R f₂ g₂ U (hg ▸ hu) := by substs hf hg; rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem const_mul_rev (f g : R) (U hu₁ hu₂) : const R f g U hu₁ * const R g f U hu₂ = 1 := by
  /-
    R : Type u
    inst✝ : CommRing R
    f g : R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hu₁ : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → M …
    hu₂ : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → M …
    ⊢ Eq (HMul.hMul (AlgebraicGeometry.StructureSheaf.const R f g U hu₁) (Algebrai …
  -/
  rw [const_mul, const_congr R rfl (mul_comm g f), const_self]
  /-
    🎉 no goals
  -/


theorem const_mul_cancel (f g₁ g₂ : R) (U hu₁ hu₂) :
    const R f g₁ U hu₁ * const R g₁ g₂ U hu₂ = const R f g₂ U hu₂ := by
  /-
    R : Type u
    inst✝ : CommRing R
    f g₁ g₂ : R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hu₁ : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → M …
    hu₂ : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → M …
    ⊢ Eq (HMul.hMul (AlgebraicGeometry.StructureSheaf.const R f g₁ U hu₁) (Algebra …
  -/
  rw [const_mul, const_ext]; rw [mul_assoc]
                             /-
                               🎉 no goals
                             -/


theorem const_mul_cancel' (f g₁ g₂ : R) (U hu₁ hu₂) :
    const R g₁ g₂ U hu₂ * const R f g₁ U hu₁ = const R f g₂ U hu₂ := by
  /-
    R : Type u
    inst✝ : CommRing R
    f g₁ g₂ : R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hu₁ : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → M …
    hu₂ : ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem U x → M …
    ⊢ Eq (HMul.hMul (AlgebraicGeometry.StructureSheaf.const R g₁ g₂ U hu₂) (Algebr …
  -/
  rw [mul_comm, const_mul_cancel]
  /-
    🎉 no goals
  -/


/-- The canonical ring homomorphism interpreting an element of `R` as
a section of the structure sheaf. -/
def toOpen (U : Opens (PrimeSpectrum.Top R)) :
    CommRingCat.of R ⟶ (structureSheaf R).1.obj (op U) := CommRingCat.ofHom
  { toFun f :=
      ⟨fun _ => algebraMap R _ f, fun x =>
        ⟨U, x.2, 𝟙 _, f, 1, fun y =>
                                                  /-
                                                    R : Type u
                                                    inst✝ : CommRing R
                                                    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                    f : R
                                                    x : Subtype fun x => Membership.mem (Opposite.unop { unop := U }) x
                                                    y : Subtype fun x => Membership.mem U x
                                                    ⊢ Eq (HMul.hMul ((fun x => (fun x => (algebraMap R (AlgebraicGeometry.Structur …
                                                  -/
          ⟨(Ideal.ne_top_iff_one _).1 y.1.2.1, by rw [RingHom.map_one, mul_one]⟩⟩⟩
                                                  /-
                                                    🎉 no goals
                                                  -/
    map_one' := Subtype.eq <| funext fun _ => RingHom.map_one _
    map_mul' _ _ := Subtype.eq <| funext fun _ => RingHom.map_mul _ _ _
    map_zero' := Subtype.eq <| funext fun _ => RingHom.map_zero _
    map_add' _ _ := Subtype.eq <| funext fun _ => RingHom.map_add _ _ _ }


@[simp]
theorem toOpen_res (U V : Opens (PrimeSpectrum.Top R)) (i : V ⟶ U) :
    toOpen R U ≫ (structureSheaf R).1.map i.op = toOpen R V :=
  rfl


@[simp]
theorem toOpen_apply (U : Opens (PrimeSpectrum.Top R)) (f : R) (x : U) :
    (toOpen R U f).1 x = algebraMap _ _ f :=
  rfl


theorem toOpen_eq_const (U : Opens (PrimeSpectrum.Top R)) (f : R) :
    toOpen R U f = const R f 1 U fun x _ => (Ideal.ne_top_iff_one _).1 x.2.1 :=
  Subtype.eq <| funext fun _ => Eq.symm <| IsLocalization.mk'_one _ f


/-- The canonical ring homomorphism interpreting an element of `R` as an element of
the stalk of `structureSheaf R` at `x`. -/
def toStalk (x : PrimeSpectrum.Top R) : CommRingCat.of R ⟶ (structureSheaf R).presheaf.stalk x :=
                                                         /-
                                                           R : Type u
                                                           inst✝ : CommRing R
                                                           x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                           ⊢ Membership.mem Top.top x
                                                         -/
  (toOpen R ⊤ ≫ (structureSheaf R).presheaf.germ _ x (by trivial))
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem toOpen_germ (U : Opens (PrimeSpectrum.Top R)) (x : PrimeSpectrum.Top R) (hx : x ∈ U) :
    toOpen R U ≫ (structureSheaf R).presheaf.germ U x hx = toStalk R x := by
  /-
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hx : Membership.mem U x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
  -/
  rw [← toOpen_res R ⊤ U (homOfLE le_top : U ⟶ ⊤), Category.assoc, Presheaf.germ_res]; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp]
theorem germ_toOpen
    (U : Opens (PrimeSpectrum.Top R)) (x : PrimeSpectrum.Top R) (hx : x ∈ U) (f : R) :
    (structureSheaf R).presheaf.germ U x hx (toOpen R U f) = toStalk R x f := by
  /-
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hx : Membership.mem U x
    f : R
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).presheaf.germ U x hx).hom ((A …
  -/
  rw [← toOpen_germ]; rfl
                      /-
                        🎉 no goals
                      -/


theorem toOpen_Γgerm_apply (x : PrimeSpectrum.Top R) (f : R) :
    (structureSheaf R).presheaf.Γgerm x (toOpen R ⊤ f) = toStalk R x f :=
  rfl


@[deprecated (since := "2024-07-30")] alias germ_to_top := toOpen_Γgerm_apply


theorem isUnit_to_basicOpen_self (f : R) : IsUnit (toOpen R (PrimeSpectrum.basicOpen f) f) :=
  isUnit_of_mul_eq_one _ (const R 1 f (PrimeSpectrum.basicOpen f) fun _ => id) <| by
    /-
      R : Type u
      inst✝ : CommRing R
      f : R
      ⊢ Eq (HMul.hMul ((AlgebraicGeometry.StructureSheaf.toOpen R (PrimeSpectrum.bas …
    -/
    rw [toOpen_eq_const, const_mul_rev]
    /-
      🎉 no goals
    -/


theorem isUnit_toStalk (x : PrimeSpectrum.Top R) (f : x.asIdeal.primeCompl) :
    IsUnit (toStalk R x (f : R)) := by
  /-
    R : Type u
    inst✝ : CommRing R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    ⊢ IsUnit ((AlgebraicGeometry.StructureSheaf.toStalk R x).hom ↑f)
  -/
  rw [← germ_toOpen R (PrimeSpectrum.basicOpen (f : R)) x f.2 (f : R)]
  /-
    R : Type u
    inst✝ : CommRing R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    ⊢ IsUnit (((AlgebraicGeometry.Spec.structureSheaf R).presheaf.germ (PrimeSpect …
  -/
  exact RingHom.isUnit_map _ (isUnit_to_basicOpen_self R f)
  /-
    🎉 no goals
  -/


/-- The canonical ring homomorphism from the localization of `R` at `p` to the stalk
of the structure sheaf at the point `p`. -/
def localizationToStalk (x : PrimeSpectrum.Top R) :
    CommRingCat.of (Localization.AtPrime x.asIdeal) ⟶ (structureSheaf R).presheaf.stalk x :=
  CommRingCat.ofHom <|
    show Localization.AtPrime x.asIdeal →+* _ from IsLocalization.lift (isUnit_toStalk R x)


@[simp]
theorem localizationToStalk_of (x : PrimeSpectrum.Top R) (f : R) :
    localizationToStalk R x (algebraMap _ (Localization _) f) = toStalk R x f :=
  IsLocalization.lift_eq (S := Localization x.asIdeal.primeCompl) _ f


@[simp]
theorem localizationToStalk_mk' (x : PrimeSpectrum.Top R) (f : R) (s : x.asIdeal.primeCompl) :
    localizationToStalk R x (IsLocalization.mk' (Localization.AtPrime x.asIdeal) f s) =
      (structureSheaf R).presheaf.germ (PrimeSpectrum.basicOpen (s : R)) x s.2
        (const R f s (PrimeSpectrum.basicOpen s) fun _ => id) :=
  (IsLocalization.lift_mk'_spec (S := Localization.AtPrime x.asIdeal) _ _ _ _).2 <| by
    rw [← germ_toOpen R (PrimeSpectrum.basicOpen s) x s.2,
      ← germ_toOpen R (PrimeSpectrum.basicOpen s) x s.2, ← RingHom.map_mul, toOpen_eq_const,
      toOpen_eq_const, const_mul_cancel']


/-- The ring homomorphism that takes a section of the structure sheaf of `R` on the open set `U`,
implemented as a subtype of dependent functions to localizations at prime ideals, and evaluates
the section on the point corresponding to a given prime ideal. -/
def openToLocalization (U : Opens (PrimeSpectrum.Top R)) (x : PrimeSpectrum.Top R) (hx : x ∈ U) :
    (structureSheaf R).1.obj (op U) ⟶ CommRingCat.of (Localization.AtPrime x.asIdeal) :=
  CommRingCat.ofHom
  { toFun s := (s.1 ⟨x, hx⟩ : _)
    map_one' := rfl
    map_mul' _ _ := rfl
    map_zero' := rfl
    map_add' _ _ := rfl }


@[simp]
theorem coe_openToLocalization (U : Opens (PrimeSpectrum.Top R)) (x : PrimeSpectrum.Top R)
    (hx : x ∈ U) :
    (openToLocalization R U x hx :
        (structureSheaf R).1.obj (op U) → Localization.AtPrime x.asIdeal) =
      fun s => (s.1 ⟨x, hx⟩ : _) :=
  rfl


theorem openToLocalization_apply (U : Opens (PrimeSpectrum.Top R)) (x : PrimeSpectrum.Top R)
    (hx : x ∈ U) (s : (structureSheaf R).1.obj (op U)) :
    openToLocalization R U x hx s = (s.1 ⟨x, hx⟩ : _) :=
  rfl


/-- The ring homomorphism from the stalk of the structure sheaf of `R` at a point corresponding to
a prime ideal `p` to the localization of `R` at `p`,
formed by gluing the `openToLocalization` maps. -/
def stalkToFiberRingHom (x : PrimeSpectrum.Top R) :
    (structureSheaf R).presheaf.stalk x ⟶ CommRingCat.of (Localization.AtPrime x.asIdeal) :=
  Limits.colimit.desc ((OpenNhds.inclusion x).op ⋙ (structureSheaf R).1)
    { pt := _
      ι := { app := fun U =>
        openToLocalization R ((OpenNhds.inclusion _).obj (unop U)) x (unop U).2 } }


@[simp]
theorem germ_comp_stalkToFiberRingHom
    (U : Opens (PrimeSpectrum.Top R)) (x : PrimeSpectrum.Top R) (hx : x ∈ U) :
    (structureSheaf R).presheaf.germ U x hx ≫ stalkToFiberRingHom R x =
      openToLocalization R U x hx :=
  Limits.colimit.ι_desc _ _


@[simp]
theorem stalkToFiberRingHom_germ (U : Opens (PrimeSpectrum.Top R))
    (x : PrimeSpectrum.Top R) (hx : x ∈ U) (s : (structureSheaf R).1.obj (op U)) :
    stalkToFiberRingHom R x ((structureSheaf R).presheaf.germ U x hx s) = s.1 ⟨x, hx⟩ :=
  RingHom.ext_iff.mp (CommRingCat.hom_ext_iff.mp (germ_comp_stalkToFiberRingHom R U x hx)) s


@[deprecated (since := "2024-07-30")] alias stalkToFiberRingHom_germ' := stalkToFiberRingHom_germ


@[simp]
theorem toStalk_comp_stalkToFiberRingHom (x : PrimeSpectrum.Top R) :
    toStalk R x ≫ stalkToFiberRingHom R x = CommRingCat.ofHom (algebraMap _ _) := by
  /-
    R : Type u
    inst✝ : CommRing R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toS …
  -/
  rw [toStalk, Category.assoc, germ_comp_stalkToFiberRingHom]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem stalkToFiberRingHom_toStalk (x : PrimeSpectrum.Top R) (f : R) :
    stalkToFiberRingHom R x (toStalk R x f) = algebraMap _ _ f :=
  RingHom.ext_iff.1 (CommRingCat.hom_ext_iff.mp (toStalk_comp_stalkToFiberRingHom R x)) _


/-- The ring isomorphism between the stalk of the structure sheaf of `R` at a point `p`
corresponding to a prime ideal in `R` and the localization of `R` at `p`. -/
@[simps]
def stalkIso (x : PrimeSpectrum.Top R) :
    (structureSheaf R).presheaf.stalk x ≅ CommRingCat.of (Localization.AtPrime x.asIdeal) where
  hom := stalkToFiberRingHom R x
  inv := localizationToStalk R x
  hom_inv_id := by
    /-
      R : Type u
      inst✝ : CommRing R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.sta …
    -/
    apply stalk_hom_ext
    /-
      case ih
      R : Type u
      inst✝ : CommRing R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ⊢ ∀ (U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)) (hxU …
    -/
    intro U hxU
    /-
      case ih
      R : Type u
      inst✝ : CommRing R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxU : Membership.mem U x
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Spec.structureShe …
    -/
    ext s
    dsimp only [CommRingCat.hom_comp, RingHom.coe_comp, Function.comp_apply, CommRingCat.hom_id,
      RingHom.coe_id, id_eq]
    /-
      case ih.hf.a
      R : Type u
      inst✝ : CommRing R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxU : Membership.mem U x
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).presheaf.obj { unop := U })
      ⊢ Eq ((AlgebraicGeometry.StructureSheaf.localizationToStalk R x).hom ((Algebra …
    -/
    rw [stalkToFiberRingHom_germ]
    obtain ⟨V, hxV, iVU, f, g, (hg : V ≤ PrimeSpectrum.basicOpen _), hs⟩ :=
      exists_const _ _ s x hxU
    /-
      case ih.hf.a.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxU : Membership.mem U x
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).presheaf.obj { unop := U })
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V x
      iVU : Quiver.Hom V U
      f g : R
      hg : LE.le V (PrimeSpectrum.basicOpen g)
      hs : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hg) (((AlgebraicGeomet …
      ⊢ Eq ((AlgebraicGeometry.StructureSheaf.localizationToStalk R x).hom (↑s ⟨x, h …
    -/
    rw [← res_apply R U V iVU s ⟨x, hxV⟩, ← hs, const_apply, localizationToStalk_mk']
    /-
      case ih.hf.a.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxU : Membership.mem U x
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).presheaf.obj { unop := U })
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V x
      iVU : Quiver.Hom V U
      f g : R
      hg : LE.le V (PrimeSpectrum.basicOpen g)
      hs : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hg) (((AlgebraicGeomet …
      ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).presheaf.germ (PrimeSpectrum. …
    -/
    refine (structureSheaf R).presheaf.germ_ext V hxV (homOfLE hg) iVU ?_
    -- Replace the `ConcreteCategory.instFunLike` instance with `CommRingCat.hom`:
    show (structureSheaf R).presheaf.map (homOfLE hg).op _ =
      (structureSheaf R).presheaf.map iVU.op s
    /-
      case ih.hf.a.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxU : Membership.mem U x
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).presheaf.obj { unop := U })
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V x
      iVU : Quiver.Hom V U
      f g : R
      hg : LE.le V (PrimeSpectrum.basicOpen g)
      hs : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hg) (((AlgebraicGeomet …
      ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).presheaf.map (CategoryTheory. …
    -/
    rw [← hs, res_const']
    /-
      🎉 no goals
    -/
  inv_hom_id := CommRingCat.hom_ext <|
    @IsLocalization.ringHom_ext R _ x.asIdeal.primeCompl (Localization.AtPrime x.asIdeal) _ _
      (Localization.AtPrime x.asIdeal) _ _
      (RingHom.comp (stalkToFiberRingHom R x).hom (localizationToStalk R x).hom)
      (RingHom.id (Localization.AtPrime _)) <| by
        /-
          R : Type u
          inst✝ : CommRing R
          x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          ⊢ Eq (((AlgebraicGeometry.StructureSheaf.stalkToFiberRingHom R x).hom.comp (Al …
        -/
        ext f
        rw [RingHom.comp_apply, RingHom.comp_apply, localizationToStalk_of,
          stalkToFiberRingHom_toStalk, RingHom.comp_apply, RingHom.id_apply]


instance (x : PrimeSpectrum R) : IsIso (stalkToFiberRingHom R x) :=
  (stalkIso R x).isIso_hom


instance (x : PrimeSpectrum R) : IsLocalHom (stalkToFiberRingHom R x).hom :=
  isLocalHom_of_isIso _


instance (x : PrimeSpectrum R) : IsIso (localizationToStalk R x) :=
  (stalkIso R x).isIso_inv


instance (x : PrimeSpectrum R) : IsLocalHom (localizationToStalk R x).hom :=
  isLocalHom_of_isIso _


@[simp, reassoc]
theorem stalkToFiberRingHom_localizationToStalk (x : PrimeSpectrum.Top R) :
    stalkToFiberRingHom R x ≫ localizationToStalk R x = 𝟙 _ :=
  (stalkIso R x).hom_inv_id


@[simp, reassoc]
theorem localizationToStalk_stalkToFiberRingHom (x : PrimeSpectrum.Top R) :
    localizationToStalk R x ≫ stalkToFiberRingHom R x = 𝟙 _ :=
  (stalkIso R x).inv_hom_id


/-- The canonical ring homomorphism interpreting `s ∈ R_f` as a section of the structure sheaf
on the basic open defined by `f ∈ R`. -/
def toBasicOpen (f : R) :
    Localization.Away f →+* (structureSheaf R).1.obj (op <| PrimeSpectrum.basicOpen f) :=
  IsLocalization.Away.lift f (isUnit_to_basicOpen_self R f)


@[simp]
theorem toBasicOpen_mk' (s f : R) (g : Submonoid.powers s) :
    toBasicOpen R s (IsLocalization.mk' (Localization.Away s) f g) =
      const R f g (PrimeSpectrum.basicOpen s) fun _ hx => Submonoid.powers_le.2 hx g.2 :=
  (IsLocalization.lift_mk'_spec _ _ _ _).2 <| by
    /-
      R : Type u
      inst✝ : CommRing R
      s f : R
      g : Subtype fun x => Membership.mem (Submonoid.powers s) x
      ⊢ Eq ((AlgebraicGeometry.StructureSheaf.toOpen R (PrimeSpectrum.basicOpen s)). …
    -/
    rw [toOpen_eq_const, toOpen_eq_const, const_mul_cancel']
    /-
      🎉 no goals
    -/


@[simp]
theorem localization_toBasicOpen (f : R) :
    RingHom.comp (toBasicOpen R f) (algebraMap R (Localization.Away f)) =
    (toOpen R (PrimeSpectrum.basicOpen f)).hom :=
  RingHom.ext fun g => by
    /-
      R : Type u
      inst✝ : CommRing R
      f g : R
      ⊢ Eq (((AlgebraicGeometry.StructureSheaf.toBasicOpen R f).comp (algebraMap R ( …
    -/
    rw [toBasicOpen, IsLocalization.Away.lift, RingHom.comp_apply, IsLocalization.lift_eq]
    /-
      🎉 no goals
    -/


@[simp]
theorem toBasicOpen_to_map (s f : R) :
    toBasicOpen R s (algebraMap R (Localization.Away s) f) =
      const R f 1 (PrimeSpectrum.basicOpen s) fun _ _ => Submonoid.one_mem _ :=
  (IsLocalization.lift_eq _ _).trans <| toOpen_eq_const _ _ _

-- The proof here follows the argument in Hartshorne's Algebraic Geometry, Proposition II.2.2.

theorem toBasicOpen_injective (f : R) : Function.Injective (toBasicOpen R f) := by
  /-
    R : Type u
    inst✝ : CommRing R
    f : R
    ⊢ Function.Injective ⇑(AlgebraicGeometry.StructureSheaf.toBasicOpen R f)
  -/
  intro s t h_eq
  /-
    R : Type u
    inst✝ : CommRing R
    f : R
    s t : Localization.Away f
    h_eq : Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) s) ((AlgebraicGe …
    ⊢ Eq s t
  -/
  obtain ⟨a, ⟨b, hb⟩, rfl⟩ := IsLocalization.mk'_surjective (Submonoid.powers f) s
  /-
    case intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f : R
    t : Localization.Away f
    a b : R
    hb : Membership.mem (Submonoid.powers f) b
    h_eq : Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) (IsLocalization. …
    ⊢ Eq (IsLocalization.mk' (Localization.Away f) a ⟨b, hb⟩) t
  -/
  obtain ⟨c, ⟨d, hd⟩, rfl⟩ := IsLocalization.mk'_surjective (Submonoid.powers f) t
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) (IsLocalization. …
    ⊢ Eq (IsLocalization.mk' (Localization.Away f) a ⟨b, hb⟩) (IsLocalization.mk'  …
  -/
  simp only [toBasicOpen_mk'] at h_eq
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    ⊢ Eq (IsLocalization.mk' (Localization.Away f) a ⟨b, hb⟩) (IsLocalization.mk'  …
  -/
  rw [IsLocalization.eq]
  -- We know that the fractions `a/b` and `c/d` are equal as sections of the structure sheaf on
  -- `basicOpen f`. We need to show that they agree as elements in the localization of `R` at `f`.
  -- This amounts showing that `r * (d * a) = r * (b * c)`, for some power `r = f ^ n` of `f`.
  -- We define `I` as the ideal of *all* elements `r` satisfying the above equation.
  let I : Ideal R :=
    { carrier := { r : R | r * (d * a) = r * (b * c) }
      zero_mem' := by simp only [Set.mem_setOf_eq, zero_mul]
      add_mem' := fun {r₁ r₂} hr₁ hr₂ => by dsimp at hr₁ hr₂ ⊢; simp only [add_mul, hr₁, hr₂]
      smul_mem' := fun {r₁ r₂} hr₂ => by dsimp at hr₂ ⊢; simp only [mul_assoc, hr₂] }
  -- Our claim now reduces to showing that `f` is contained in the radical of `I`
  suffices f ∈ I.radical by
    cases' this with n hn
    exact ⟨⟨f ^ n, n, rfl⟩, hn⟩
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    ⊢ Membership.mem I.radical f
  -/
  rw [← PrimeSpectrum.vanishingIdeal_zeroLocus_eq_radical, PrimeSpectrum.mem_vanishingIdeal]
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    ⊢ ∀ (x : PrimeSpectrum R), Membership.mem (PrimeSpectrum.zeroLocus ↑I) x → Mem …
  -/
  intro p hfp
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    p : PrimeSpectrum R
    hfp : Membership.mem (PrimeSpectrum.zeroLocus ↑I) p
    ⊢ Membership.mem p.asIdeal f
  -/
  contrapose hfp
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    p : PrimeSpectrum R
    hfp : Not (Membership.mem p.asIdeal f)
    ⊢ Not (Membership.mem (PrimeSpectrum.zeroLocus ↑I) p)
  -/
  rw [PrimeSpectrum.mem_zeroLocus, Set.not_subset]
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    p : PrimeSpectrum R
    hfp : Not (Membership.mem p.asIdeal f)
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (Not (Membership.mem (↑p.asIdeal …
  -/
  have := congr_fun (congr_arg Subtype.val h_eq) ⟨p, hfp⟩
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    p : PrimeSpectrum R
    hfp : Not (Membership.mem p.asIdeal f)
    this : Eq (↑(AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basic …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (Not (Membership.mem (↑p.asIdeal …
  -/
  dsimp at this
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    p : PrimeSpectrum R
    hfp : Not (Membership.mem p.asIdeal f)
    this : Eq (IsLocalization.mk' (Localization.AtPrime p.asIdeal) a ⟨b, ⋯⟩) (IsLo …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (Not (Membership.mem (↑p.asIdeal …
  -/
  rw [IsLocalization.eq (S := Localization.AtPrime p.asIdeal)] at this
  /-
    case intro.intro.mk.intro.intro.mk
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    p : PrimeSpectrum R
    hfp : Not (Membership.mem p.asIdeal f)
    this : Exists fun c_1 => Eq (HMul.hMul (↑c_1) (HMul.hMul (↑⟨d, ⋯⟩) a)) (HMul.h …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (Not (Membership.mem (↑p.asIdeal …
  -/
  cases' this with r hr
  /-
    case intro.intro.mk.intro.intro.mk.intro
    R : Type u
    inst✝ : CommRing R
    f a b : R
    hb : Membership.mem (Submonoid.powers f) b
    c d : R
    hd : Membership.mem (Submonoid.powers f) d
    h_eq : Eq (AlgebraicGeometry.StructureSheaf.const R a b (PrimeSpectrum.basicOp …
    I : Ideal R := { carrier := setOf fun r => Eq (HMul.hMul r (HMul.hMul d a)) (H …
    p : PrimeSpectrum R
    hfp : Not (Membership.mem p.asIdeal f)
    r : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
    hr : Eq (HMul.hMul (↑r) (HMul.hMul (↑⟨d, ⋯⟩) a)) (HMul.hMul (↑r) (HMul.hMul (↑ …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (Not (Membership.mem (↑p.asIdeal …
  -/
  exact ⟨r.1, hr, r.2⟩
  /-
    🎉 no goals
  -/

/-
Auxiliary lemma for surjectivity of `toBasicOpen`.
Every section can locally be represented on basic opens `basicOpen g` as a fraction `f/g`
-/

theorem locally_const_basicOpen (U : Opens (PrimeSpectrum.Top R))
    (s : (structureSheaf R).1.obj (op U)) (x : U) :
    ∃ (f g : R) (i : PrimeSpectrum.basicOpen g ⟶ U), x.1 ∈ PrimeSpectrum.basicOpen g ∧
      (const R f g (PrimeSpectrum.basicOpen g) fun _ hy => hy) =
      (structureSheaf R).1.map i.op s := by
  -- First, any section `s` can be represented as a fraction `f/g` on some open neighborhood of `x`
  -- and we may pass to a `basicOpen h`, since these form a basis
  obtain ⟨V, hxV : x.1 ∈ V.1, iVU, f, g, hVDg : V ≤ PrimeSpectrum.basicOpen g, s_eq⟩ :=
    exists_const R U s x.1 x.2
  obtain ⟨_, ⟨h, rfl⟩, hxDh, hDhV : PrimeSpectrum.basicOpen h ≤ V⟩ :=
    PrimeSpectrum.isTopologicalBasis_basic_opens.exists_subset_of_mem_open hxV V.2
  -- The problem is of course, that `g` and `h` don't need to coincide.
  -- But, since `basicOpen h ≤ basicOpen g`, some power of `h` must be a multiple of `g`
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    ⊢ Exists fun f => Exists fun g => Exists fun i => And (Membership.mem (PrimeSp …
  -/
  cases' (PrimeSpectrum.basicOpen_le_basicOpen_iff h g).mp (Set.Subset.trans hDhV hVDg) with n hn
  -- Actually, we will need a *nonzero* power of `h`.
  -- This is because we will need the equality `basicOpen (h ^ n) = basicOpen h`, which only
  -- holds for a nonzero power `n`. We therefore artificially increase `n` by one.
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    hn : Membership.mem (Ideal.span (Singleton.singleton g)) (HPow.hPow h n)
    ⊢ Exists fun f => Exists fun g => Exists fun i => And (Membership.mem (PrimeSp …
  -/
  replace hn := Ideal.mul_mem_right h (Ideal.span {g}) hn
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    hn : Membership.mem (Ideal.span (Singleton.singleton g)) (HMul.hMul (HPow.hPow …
    ⊢ Exists fun f => Exists fun g => Exists fun i => And (Membership.mem (PrimeSp …
  -/
  rw [← pow_succ, Ideal.mem_span_singleton'] at hn
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    hn : Exists fun a => Eq (HMul.hMul a g) (HPow.hPow h (HAdd.hAdd n 1))
    ⊢ Exists fun f => Exists fun g => Exists fun i => And (Membership.mem (PrimeSp …
  -/
  cases' hn with c hc
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    c : R
    hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
    ⊢ Exists fun f => Exists fun g => Exists fun i => And (Membership.mem (PrimeSp …
  -/
  have basic_opens_eq := PrimeSpectrum.basicOpen_pow h (n + 1) (by omega)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    c : R
    hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
    basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
    ⊢ Exists fun f => Exists fun g => Exists fun i => And (Membership.mem (PrimeSp …
  -/
  have i_basic_open := eqToHom basic_opens_eq ≫ homOfLE hDhV
  -- We claim that `(f * c) / h ^ (n+1)` is our desired representation
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    c : R
    hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
    basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
    i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
    ⊢ Exists fun f => Exists fun g => Exists fun i => And (Membership.mem (PrimeSp …
  -/
  use f * c, h ^ (n + 1), i_basic_open ≫ iVU, (basic_opens_eq.symm.le : _) hxDh
  /-
    case right
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    c : R
    hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
    basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
    i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (HMul.hMul f c) (HPow.hPow h (H …
  -/
  rw [op_comp, Functor.map_comp] --, comp_apply, ← s_eq, res_const]
  -- Porting note: `comp_apply` can't be rewritten, so use a change
  change const R _ _ _ _ = (structureSheaf R).1.map i_basic_open.op
    ((structureSheaf R).1.map iVU.op s)
  /-
    case right
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    c : R
    hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
    basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
    i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (HMul.hMul f c) (HPow.hPow h (H …
  -/
  rw [← s_eq, res_const]
  -- Note that the last rewrite here generated an additional goal, which was a parameter
  -- of `res_const`. We prove this goal first
  /-
    case right
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    c : R
    hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
    basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
    i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (HMul.hMul f c) (HPow.hPow h (H …
  -/
  swap
    /-
      case right.hv
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      x : Subtype fun x => Membership.mem U x
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V.carrier ↑x
      iVU : Quiver.Hom V U
      f g : R
      hVDg : LE.le V (PrimeSpectrum.basicOpen g)
      s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
      h : R
      hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
      hDhV : LE.le (PrimeSpectrum.basicOpen h) V
      n : Nat
      c : R
      hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
      basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
      i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
      ⊢ ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem (PrimeSpect …
    -/
  · intro y hy
    /-
      case right.hv
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      x : Subtype fun x => Membership.mem U x
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V.carrier ↑x
      iVU : Quiver.Hom V U
      f g : R
      hVDg : LE.le V (PrimeSpectrum.basicOpen g)
      s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
      h : R
      hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
      hDhV : LE.le (PrimeSpectrum.basicOpen h) V
      n : Nat
      c : R
      hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
      basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
      i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
      y : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hy : Membership.mem (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) y
      ⊢ Membership.mem y.asIdeal.primeCompl g
    -/
    rw [basic_opens_eq] at hy
    /-
      case right.hv
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      x : Subtype fun x => Membership.mem U x
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V.carrier ↑x
      iVU : Quiver.Hom V U
      f g : R
      hVDg : LE.le V (PrimeSpectrum.basicOpen g)
      s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
      h : R
      hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
      hDhV : LE.le (PrimeSpectrum.basicOpen h) V
      n : Nat
      c : R
      hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
      basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
      i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
      y : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hy : Membership.mem (PrimeSpectrum.basicOpen h) y
      ⊢ Membership.mem y.asIdeal.primeCompl g
    -/
    exact (Set.Subset.trans hDhV hVDg : _) hy
    /-
      🎉 no goals
    -/
  -- All that is left is a simple calculation
  /-
    case right
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    c : R
    hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
    basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
    i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (HMul.hMul f c) (HPow.hPow h (H …
  -/
  apply const_ext
  /-
    case right.h
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hxV : Membership.mem V.carrier ↑x
    iVU : Quiver.Hom V U
    f g : R
    hVDg : LE.le V (PrimeSpectrum.basicOpen g)
    s_eq : Eq (AlgebraicGeometry.StructureSheaf.const R f g V hVDg) (((AlgebraicGe …
    h : R
    hxDh : Membership.mem ((fun r => ↑(PrimeSpectrum.basicOpen r)) h) ↑x
    hDhV : LE.le (PrimeSpectrum.basicOpen h) V
    n : Nat
    c : R
    hc : Eq (HMul.hMul c g) (HPow.hPow h (HAdd.hAdd n 1))
    basic_opens_eq : Eq (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1))) (P …
    i_basic_open : Quiver.Hom (PrimeSpectrum.basicOpen (HPow.hPow h (HAdd.hAdd n 1 …
    ⊢ Eq (HMul.hMul (HMul.hMul f c) g) (HMul.hMul f (HPow.hPow h (HAdd.hAdd n 1)))
  -/
  rw [mul_assoc f c g, hc]
  /-
    🎉 no goals
  -/

/-
Auxiliary lemma for surjectivity of `toBasicOpen`.
A local representation of a section `s` as fractions `a i / h i` on finitely many basic opens
`basicOpen (h i)` can be "normalized" in such a way that `a i * h j = h i * a j` for all `i, j`
-/

theorem normalize_finite_fraction_representation (U : Opens (PrimeSpectrum.Top R))
    (s : (structureSheaf R).1.obj (op U)) {ι : Type*} (t : Finset ι) (a h : ι → R)
    (iDh : ∀ i : ι, PrimeSpectrum.basicOpen (h i) ⟶ U)
    (h_cover : U ≤ ⨆ i ∈ t, PrimeSpectrum.basicOpen (h i))
    (hs :
      ∀ i : ι,
        (const R (a i) (h i) (PrimeSpectrum.basicOpen (h i)) fun _ hy => hy) =
          (structureSheaf R).1.map (iDh i).op s) :
    ∃ (a' h' : ι → R) (iDh' : ∀ i : ι, PrimeSpectrum.basicOpen (h' i) ⟶ U),
      (U ≤ ⨆ i ∈ t, PrimeSpectrum.basicOpen (h' i)) ∧
        (∀ (i) (_ : i ∈ t) (j) (_ : j ∈ t), a' i * h' j = h' i * a' j) ∧
          ∀ i ∈ t,
            (structureSheaf R).1.map (iDh' i).op s =
              const R (a' i) (h' i) (PrimeSpectrum.basicOpen (h' i)) fun _ hy => hy := by
  -- First we show that the fractions `(a i * h j) / (h i * h j)` and `(h i * a j) / (h i * h j)`
  -- coincide in the localization of `R` at `h i * h j`
  have fractions_eq :
    ∀ i j : ι,
      IsLocalization.mk' (Localization.Away (h i * h j))
        (a i * h j) ⟨h i * h j, Submonoid.mem_powers _⟩ =
      IsLocalization.mk' _ (h i * a j) ⟨h i * h j, Submonoid.mem_powers _⟩ := by
    intro i j
    let D := PrimeSpectrum.basicOpen (h i * h j)
    let iDi : D ⟶ PrimeSpectrum.basicOpen (h i) := homOfLE (PrimeSpectrum.basicOpen_mul_le_left _ _)
    let iDj : D ⟶ PrimeSpectrum.basicOpen (h j) :=
      homOfLE (PrimeSpectrum.basicOpen_mul_le_right _ _)
    -- Crucially, we need injectivity of `toBasicOpen`
    apply toBasicOpen_injective R (h i * h j)
    rw [toBasicOpen_mk', toBasicOpen_mk']
    simp only []
    -- Here, both sides of the equation are equal to a restriction of `s`
    trans
    on_goal 1 =>
      convert congr_arg ((structureSheaf R).1.map iDj.op) (hs j).symm using 1
      convert congr_arg ((structureSheaf R).1.map iDi.op) (hs i) using 1
    all_goals rw [res_const]; apply const_ext; ring
    -- The remaining two goals were generated during the rewrite of `res_const`
    -- These can be solved immediately
    exacts [PrimeSpectrum.basicOpen_mul_le_left _ _, PrimeSpectrum.basicOpen_mul_le_right _ _]
  -- From the equality in the localization, we obtain for each `(i,j)` some power `(h i * h j) ^ n`
  -- which equalizes `a i * h j` and `h i * a j`
  have exists_power :
    ∀ i j : ι, ∃ n : ℕ, a i * h j * (h i * h j) ^ n = h i * a j * (h i * h j) ^ n := by
    intro i j
    obtain ⟨⟨c, n, rfl⟩, hc⟩ := IsLocalization.eq.mp (fractions_eq i j)
    use n + 1
    rw [pow_succ]
    dsimp at hc
    convert hc using 1 <;> ring
  /-
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    ⊢ Exists fun a' => Exists fun h' => Exists fun iDh' => And (LE.le U (iSup fun  …
  -/
  let n := fun p : ι × ι => (exists_power p.1 p.2).choose
  /-
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    ⊢ Exists fun a' => Exists fun h' => Exists fun iDh' => And (LE.le U (iSup fun  …
  -/
  have n_spec := fun p : ι × ι => (exists_power p.fst p.snd).choose_spec
  -- We need one power `(h i * h j) ^ N` that works for *all* pairs `(i,j)`
  -- Since there are only finitely many indices involved, we can pick the supremum.
  /-
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    ⊢ Exists fun a' => Exists fun h' => Exists fun iDh' => And (LE.le U (iSup fun  …
  -/
  let N := (t ×ˢ t).sup n
  have basic_opens_eq : ∀ i : ι, PrimeSpectrum.basicOpen (h i ^ (N + 1)) =
    PrimeSpectrum.basicOpen (h i) := fun i => PrimeSpectrum.basicOpen_pow _ _ (by omega)
  -- Expanding the fraction `a i / h i` by the power `(h i) ^ n` gives the desired normalization
  refine
    ⟨fun i => a i * h i ^ N, fun i => h i ^ (N + 1), fun i => eqToHom (basic_opens_eq i) ≫ iDh i,
      ?_, ?_, ?_⟩
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      ι : Type u_1
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
      h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
      hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
      fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
      exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
      n : Prod ι ι → Nat := fun p => ⋯.choose
      n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
      N : Nat := (SProd.sprod t t).sup n
      basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
      ⊢ LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen ((fun i => HP …
    -/
  · simpa only [basic_opens_eq] using h_cover
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      ι : Type u_1
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
      h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
      hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
      fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
      exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
      n : Prod ι ι → Nat := fun p => ⋯.choose
      n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
      N : Nat := (SProd.sprod t t).sup n
      basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
      ⊢ ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HMul.hMu …
    -/
  · intro i hi j hj
    -- Here we need to show that our new fractions `a i / h i` satisfy the normalization condition
    -- Of course, the power `N` we used to expand the fractions might be bigger than the power
    -- `n (i, j)` which was originally chosen. We denote their difference by `k`
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      ι : Type u_1
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
      h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
      hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
      fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
      exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
      n : Prod ι ι → Nat := fun p => ⋯.choose
      n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
      N : Nat := (SProd.sprod t t).sup n
      basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
      i : ι
      hi : Membership.mem t i
      j : ι
      hj : Membership.mem t j
      ⊢ Eq (HMul.hMul ((fun i => HMul.hMul (a i) (HPow.hPow (h i) N)) i) ((fun i =>  …
    -/
    have n_le_N : n (i, j) ≤ N := Finset.le_sup (Finset.mem_product.mpr ⟨hi, hj⟩)
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      ι : Type u_1
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
      h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
      hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
      fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
      exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
      n : Prod ι ι → Nat := fun p => ⋯.choose
      n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
      N : Nat := (SProd.sprod t t).sup n
      basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
      i : ι
      hi : Membership.mem t i
      j : ι
      hj : Membership.mem t j
      n_le_N : LE.le (n { fst := i, snd := j }) N
      ⊢ Eq (HMul.hMul ((fun i => HMul.hMul (a i) (HPow.hPow (h i) N)) i) ((fun i =>  …
    -/
    cases' Nat.le.dest n_le_N with k hk
    /-
      case refine_2.intro
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      ι : Type u_1
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
      h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
      hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
      fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
      exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
      n : Prod ι ι → Nat := fun p => ⋯.choose
      n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
      N : Nat := (SProd.sprod t t).sup n
      basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
      i : ι
      hi : Membership.mem t i
      j : ι
      hj : Membership.mem t j
      n_le_N : LE.le (n { fst := i, snd := j }) N
      k : Nat
      hk : Eq (HAdd.hAdd (n { fst := i, snd := j }) k) N
      ⊢ Eq (HMul.hMul ((fun i => HMul.hMul (a i) (HPow.hPow (h i) N)) i) ((fun i =>  …
    -/
    simp only [← hk, pow_add, pow_one]
    -- To accommodate for the difference `k`, we multiply both sides of the equation `n_spec (i, j)`
    -- by `(h i * h j) ^ k`
    /-
      case refine_2.intro
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      ι : Type u_1
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
      h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
      hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
      fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
      exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
      n : Prod ι ι → Nat := fun p => ⋯.choose
      n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
      N : Nat := (SProd.sprod t t).sup n
      basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
      i : ι
      hi : Membership.mem t i
      j : ι
      hj : Membership.mem t j
      n_le_N : LE.le (n { fst := i, snd := j }) N
      k : Nat
      hk : Eq (HAdd.hAdd (n { fst := i, snd := j }) k) N
      ⊢ Eq (HMul.hMul (HMul.hMul (a i) (HMul.hMul (HPow.hPow (h i) (n { fst := i, sn …
    -/
    convert congr_arg (fun z => z * (h i * h j) ^ k) (n_spec (i, j)) using 1 <;>
        /-
          case h.e'_2
          R : Type u
          inst✝ : CommRing R
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          ι : Type u_1
          t : Finset ι
          a h : ι → R
          iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
          h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
          hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
          fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
          exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
          n : Prod ι ι → Nat := fun p => ⋯.choose
          n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
          N : Nat := (SProd.sprod t t).sup n
          basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
          i : ι
          hi : Membership.mem t i
          j : ι
          hj : Membership.mem t j
          n_le_N : LE.le (n { fst := i, snd := j }) N
          k : Nat
          hk : Eq (HAdd.hAdd (n { fst := i, snd := j }) k) N
          ⊢ Eq (HMul.hMul (HMul.hMul (a i) (HMul.hMul (HPow.hPow (h i) (n { fst := i, sn …
        -/
                                /-
                                  🎉 no goals
                                -/
      · simp only [n, mul_pow]; ring
                                /-
                                  🎉 no goals
                                -/
  -- Lastly, we need to show that the new fractions still represent our original `s`
  /-
    case refine_3
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    N : Nat := (SProd.sprod t t).sup n
    basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
    ⊢ ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureSheaf  …
  -/
  intro i _
  /-
    case refine_3
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    N : Nat := (SProd.sprod t t).sup n
    basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
    i : ι
    a✝ : Membership.mem t i
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).val.map ((fun i => CategoryTh …
  -/
  rw [op_comp, Functor.map_comp]
  -- Porting note: `comp_apply` can't be rewritten, so use a change
  change (structureSheaf R).1.map (eqToHom (basic_opens_eq _)).op
    ((structureSheaf R).1.map (iDh i).op s) = _
  /-
    case refine_3
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    N : Nat := (SProd.sprod t t).sup n
    basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
    i : ι
    a✝ : Membership.mem t i
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).val.map (CategoryTheory.eqToH …
  -/
  rw [← hs, res_const]
  -- additional goal spit out by `res_const`
  /-
    case refine_3
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    N : Nat := (SProd.sprod t t).sup n
    basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
    i : ι
    a✝ : Membership.mem t i
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (PrimeSpectrum.basi …
  -/
  swap
    /-
      case refine_3.hv
      R : Type u
      inst✝ : CommRing R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      ι : Type u_1
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
      h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
      hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
      fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
      exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
      n : Prod ι ι → Nat := fun p => ⋯.choose
      n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
      N : Nat := (SProd.sprod t t).sup n
      basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
      i : ι
      a✝ : Membership.mem t i
      ⊢ ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem (PrimeSpect …
    -/
  · exact (basic_opens_eq i).le
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    N : Nat := (SProd.sprod t t).sup n
    basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
    i : ι
    a✝ : Membership.mem t i
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (PrimeSpectrum.basi …
  -/
  apply const_ext
  /-
    case refine_3.h
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    N : Nat := (SProd.sprod t t).sup n
    basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
    i : ι
    a✝ : Membership.mem t i
    ⊢ Eq (HMul.hMul (a i) ((fun i => HPow.hPow (h i) (HAdd.hAdd N 1)) i)) (HMul.hM …
  -/
  dsimp
  /-
    case refine_3.h
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    N : Nat := (SProd.sprod t t).sup n
    basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
    i : ι
    a✝ : Membership.mem t i
    ⊢ Eq (HMul.hMul (a i) (HPow.hPow (h i) (HAdd.hAdd N 1))) (HMul.hMul (HMul.hMul …
  -/
  rw [pow_succ]
  /-
    case refine_3.h
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
    ι : Type u_1
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) U
    h_cover : LE.le U (iSup fun i => iSup fun h_1 => PrimeSpectrum.basicOpen (h i))
    hs : ∀ (i : ι), Eq (AlgebraicGeometry.StructureSheaf.const R (a i) (h i) (Prim …
    fractions_eq : ∀ (i j : ι), Eq (IsLocalization.mk' (Localization.Away (HMul.hM …
    exists_power : ∀ (i j : ι), Exists fun n => Eq (HMul.hMul (HMul.hMul (a i) (h  …
    n : Prod ι ι → Nat := fun p => ⋯.choose
    n_spec : ∀ (p : Prod ι ι), Eq (HMul.hMul (HMul.hMul (a p.1) (h p.2)) (HPow.hPo …
    N : Nat := (SProd.sprod t t).sup n
    basic_opens_eq : ∀ (i : ι), Eq (PrimeSpectrum.basicOpen (HPow.hPow (h i) (HAdd …
    i : ι
    a✝ : Membership.mem t i
    ⊢ Eq (HMul.hMul (a i) (HMul.hMul (HPow.hPow (h i) N) (h i))) (HMul.hMul (HMul. …
  -/
  ring
  /-
    🎉 no goals
  -/

-- Porting note: in the following proof there are two places where `⋃ i, ⋃ (hx : i ∈ _), ... `
-- though `hx` is not used in `...` part, it is still required to maintain the structure of
-- the original proof in mathlib3.

set_option linter.unusedVariables false in
-- The proof here follows the argument in Hartshorne's Algebraic Geometry, Proposition II.2.2.
theorem toBasicOpen_surjective (f : R) : Function.Surjective (toBasicOpen R f) := by
  /-
    R : Type u
    inst✝ : CommRing R
    f : R
    ⊢ Function.Surjective ⇑(AlgebraicGeometry.StructureSheaf.toBasicOpen R f)
  -/
  intro s
  -- In this proof, `basicOpen f` will play two distinct roles: Firstly, it is an open set in the
  -- prime spectrum. Secondly, it is used as an indexing type for various families of objects
  -- (open sets, ring elements, ...). In order to make the distinction clear, we introduce a type
  -- alias `ι` that is used whenever we want think of it as an indexing type.
  /-
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  let ι : Type u := PrimeSpectrum.basicOpen f
  -- First, we pick some cover of basic opens, on which we can represent `s` as a fraction
  /-
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  choose a' h' iDh' hxDh' s_eq' using locally_const_basicOpen R (PrimeSpectrum.basicOpen f) s
  -- Since basic opens are compact, we can pass to a finite subcover
  obtain ⟨t, ht_cover'⟩ :=
    (PrimeSpectrum.isCompact_basicOpen f).elim_finite_subcover
      (fun i : ι => PrimeSpectrum.basicOpen (h' i)) (fun i => PrimeSpectrum.isOpen_basicOpen)
      -- Here, we need to show that our basic opens actually form a cover of `basicOpen f`
      fun x hx => by rw [Set.mem_iUnion]; exact ⟨⟨x, hx⟩, hxDh' ⟨x, hx⟩⟩
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    a' h' : (Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x) → R
    iDh' : (x : Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x) → Q …
    hxDh' : ∀ (x : Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x), …
    s_eq' : ∀ (x : Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x), …
    t : Finset ι
    ht_cover' : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i  …
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  simp only [← Opens.coe_iSup, SetLike.coe_subset_coe] at ht_cover'
  -- We use the normalization lemma from above to obtain the relation `a i * h j = h i * a j`
  obtain ⟨a, h, iDh, ht_cover, ah_ha, s_eq⟩ :=
    normalize_finite_fraction_representation R (PrimeSpectrum.basicOpen f)
      s t a' h' iDh' ht_cover' s_eq'
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    a' h' : (Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x) → R
    iDh' : (x : Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x) → Q …
    hxDh' : ∀ (x : Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x), …
    s_eq' : ∀ (x : Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x), …
    t : Finset ι
    ht_cover' : LE.le (PrimeSpectrum.basicOpen f) (iSup fun i => iSup fun i_1 => P …
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ht_cover : LE.le (PrimeSpectrum.basicOpen f) (iSup fun i => iSup fun h_1 => Pr …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  clear s_eq' iDh' hxDh' ht_cover' a' h'
  -- Porting note: simp with `[← SetLike.coe_subset_coe, Opens.coe_iSup]` does not result in
  -- desired form
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ht_cover : LE.le (PrimeSpectrum.basicOpen f) (iSup fun i => iSup fun h_1 => Pr …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  rw [← SetLike.coe_subset_coe, Opens.coe_iSup] at ht_cover
  replace ht_cover : (PrimeSpectrum.basicOpen f : Set <| PrimeSpectrum R) ⊆
      ⋃ (i : ι) (x : i ∈ t), (PrimeSpectrum.basicOpen (h i) : Set _) := by
    convert ht_cover using 2
    exact funext fun j => by rw [Opens.coe_iSup]
  -- Next we show that some power of `f` is a linear combination of the `h i`
  obtain ⟨n, hn⟩ : f ∈ (Ideal.span (h '' ↑t)).radical := by
    rw [← PrimeSpectrum.vanishingIdeal_zeroLocus_eq_radical, PrimeSpectrum.zeroLocus_span]
    -- Porting note: simp with `PrimeSpectrum.basicOpen_eq_zeroLocus_compl` does not work
    replace ht_cover : (PrimeSpectrum.zeroLocus {f})ᶜ ⊆
        ⋃ (i : ι) (x : i ∈ t), (PrimeSpectrum.zeroLocus {h i})ᶜ := by
      convert ht_cover
      · rw [PrimeSpectrum.basicOpen_eq_zeroLocus_compl]
      · simp only [Opens.iSup_mk, Opens.carrier_eq_coe, PrimeSpectrum.basicOpen_eq_zeroLocus_compl]
    rw [Set.compl_subset_comm] at ht_cover
    -- Why doesn't `simp_rw` do this?
    simp_rw [Set.compl_iUnion, compl_compl, ← PrimeSpectrum.zeroLocus_iUnion,
      ← Finset.set_biUnion_coe, ← Set.image_eq_iUnion] at ht_cover
    apply PrimeSpectrum.vanishingIdeal_anti_mono ht_cover
    exact PrimeSpectrum.subset_vanishingIdeal_zeroLocus {f} (Set.mem_singleton f)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    hn : Membership.mem (Ideal.span (Set.image h ↑t)) (HPow.hPow f n)
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  replace hn := Ideal.mul_mem_right f _ hn
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    hn : Membership.mem (Ideal.span (Set.image h ↑t)) (HMul.hMul (HPow.hPow f n) f)
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  erw [← pow_succ, Finsupp.mem_span_image_iff_linearCombination] at hn
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    hn : Exists fun l => And (Membership.mem (Finsupp.supported R R ↑t) l) (Eq ((F …
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  rcases hn with ⟨b, b_supp, hb⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq ((Finsupp.linearCombination R h) b) (HPow.hPow f (HAdd.hAdd n 1))
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  rw [Finsupp.linearCombination_apply_of_mem_supported R b_supp] at hb
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HSMul.hSMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    ⊢ Exists fun a => Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) a) s
  -/
  dsimp at hb
  -- Finally, we have all the ingredients.
  -- We claim that our preimage is given by `(∑ (i : ι) ∈ t, b i * a i) / f ^ (n+1)`
  use
    IsLocalization.mk' (Localization.Away f) (∑ i ∈ t, b i * a i)
      (⟨f ^ (n + 1), n + 1, rfl⟩ : Submonoid.powers _)
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    ⊢ Eq ((AlgebraicGeometry.StructureSheaf.toBasicOpen R f) (IsLocalization.mk' ( …
  -/
  rw [toBasicOpen_mk']
  -- Since the structure sheaf is a sheaf, we can show the desired equality locally.
  -- Annoyingly, `Sheaf.eq_of_locally_eq'` requires an open cover indexed by a *type*, so we need to
  -- coerce our finset `t` to a type first.
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (t.sum fun i => HMul.hMul (b i) …
  -/
  let tt := ((t : Set (PrimeSpectrum.basicOpen f)) : Type u)
  apply
    (structureSheaf R).eq_of_locally_eq' (fun i : tt => PrimeSpectrum.basicOpen (h i))
      (PrimeSpectrum.basicOpen f) fun i : tt => iDh i
  · -- This feels a little redundant, since already have `ht_cover` as a hypothesis
    -- Unfortunately, `ht_cover` uses a bounded union over the set `t`, while here we have the
    -- Union indexed by the type `tt`, so we need some boilerplate to translate one to the other
    /-
      case h.hcover
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      ⊢ LE.le (PrimeSpectrum.basicOpen f) (iSup fun i => PrimeSpectrum.basicOpen (h  …
    -/
    intro x hx
    /-
      case h.hcover
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hx : Membership.mem (↑(PrimeSpectrum.basicOpen f)) x
      ⊢ Membership.mem (↑(iSup fun i => PrimeSpectrum.basicOpen (h ↑i))) x
    -/
    erw [TopologicalSpace.Opens.mem_iSup]
    /-
      case h.hcover
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hx : Membership.mem (↑(PrimeSpectrum.basicOpen f)) x
      ⊢ Exists fun i => Membership.mem (PrimeSpectrum.basicOpen (h ↑i)) x
    -/
    have := ht_cover hx
    /-
      case h.hcover
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hx : Membership.mem (↑(PrimeSpectrum.basicOpen f)) x
      this : Membership.mem (Set.iUnion fun i => Set.iUnion fun x => ↑(PrimeSpectrum …
      ⊢ Exists fun i => Membership.mem (PrimeSpectrum.basicOpen (h ↑i)) x
    -/
    rw [← Finset.set_biUnion_coe, Set.mem_iUnion₂] at this
    /-
      case h.hcover
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hx : Membership.mem (↑(PrimeSpectrum.basicOpen f)) x
      this : Exists fun i => Exists fun j => Membership.mem (↑(PrimeSpectrum.basicOp …
      ⊢ Exists fun i => Membership.mem (PrimeSpectrum.basicOpen (h ↑i)) x
    -/
    rcases this with ⟨i, i_mem, x_mem⟩
    /-
      case h.hcover.intro.intro
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hx : Membership.mem (↑(PrimeSpectrum.basicOpen f)) x
      i : ι
      i_mem : Membership.mem (↑t) i
      x_mem : Membership.mem (↑(PrimeSpectrum.basicOpen (h i))) x
      ⊢ Exists fun i => Membership.mem (PrimeSpectrum.basicOpen (h ↑i)) x
    -/
    exact ⟨⟨i, i_mem⟩, x_mem⟩
    /-
      🎉 no goals
    -/
  /-
    case h.h
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    ⊢ ∀ (i : tt), Eq (((AlgebraicGeometry.Spec.structureSheaf R).val.map (iDh ↑i). …
  -/
  rintro ⟨i, hi⟩
  /-
    case h.h.mk
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).val.map (iDh ↑⟨i, hi⟩).op) (A …
  -/
  dsimp
  /-
    case h.h.mk
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).val.map (iDh i).op) (Algebrai …
  -/
  change (structureSheaf R).1.map (iDh i).op _ = (structureSheaf R).1.map (iDh i).op _
  /-
    case h.h.mk
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).val.map (iDh i).op).hom (Alge …
  -/
  rw [s_eq i hi, res_const]
  -- Again, `res_const` spits out an additional goal
  /-
    case h.h.mk
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (t.sum fun i => HMul.hMul (b i) …
  -/
  swap
    /-
      case h.h.mk.hv
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      i : ι
      hi : Membership.mem (↑t) i
      ⊢ ∀ (x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)), Membership.mem (PrimeSpect …
    -/
  · intro y hy
    /-
      case h.h.mk.hv
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      i : ι
      hi : Membership.mem (↑t) i
      y : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hy : Membership.mem (PrimeSpectrum.basicOpen (h i)) y
      ⊢ Membership.mem y.asIdeal.primeCompl (HPow.hPow f (HAdd.hAdd n 1))
    -/
    change y ∈ PrimeSpectrum.basicOpen (f ^ (n + 1))
    /-
      case h.h.mk.hv
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      i : ι
      hi : Membership.mem (↑t) i
      y : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hy : Membership.mem (PrimeSpectrum.basicOpen (h i)) y
      ⊢ Membership.mem (PrimeSpectrum.basicOpen (HPow.hPow f (HAdd.hAdd n 1))) y
    -/
    rw [PrimeSpectrum.basicOpen_pow f (n + 1) (by omega)]
    /-
      case h.h.mk.hv
      R : Type u
      inst✝ : CommRing R
      f : R
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
      ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
      t : Finset ι
      a h : ι → R
      iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
      ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
      s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
      ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
      n : Nat
      b : Finsupp ι R
      b_supp : Membership.mem (Finsupp.supported R R ↑t) b
      hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
      tt : Type u := ↑↑t
      i : ι
      hi : Membership.mem (↑t) i
      y : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hy : Membership.mem (PrimeSpectrum.basicOpen (h i)) y
      ⊢ Membership.mem (PrimeSpectrum.basicOpen f) y
    -/
    exact (leOfHom (iDh i) : _) hy
    /-
      🎉 no goals
    -/
  -- The rest of the proof is just computation
  /-
    case h.h.mk
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.const R (t.sum fun i => HMul.hMul (b i) …
  -/
  apply const_ext
  /-
    case h.h.mk.h
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    ⊢ Eq (HMul.hMul (t.sum fun i => HMul.hMul (b i) (a i)) (h i)) (HMul.hMul (a i) …
  -/
  rw [← hb, Finset.sum_mul, Finset.mul_sum]
  /-
    case h.h.mk.h
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    ⊢ Eq (t.sum fun i_1 => HMul.hMul (HMul.hMul (b i_1) (a i_1)) (h i)) (t.sum fun …
  -/
  apply Finset.sum_congr rfl
  /-
    case h.h.mk.h
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    ⊢ ∀ (x : ι), Membership.mem t x → Eq (HMul.hMul (HMul.hMul (b x) (a x)) (h i)) …
  -/
  intro j hj
  /-
    case h.h.mk.h
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    j : ι
    hj : Membership.mem t j
    ⊢ Eq (HMul.hMul (HMul.hMul (b j) (a j)) (h i)) (HMul.hMul (a i) (HMul.hMul (b  …
  -/
  rw [mul_assoc, ah_ha j hj i hi]
  /-
    case h.h.mk.h
    R : Type u
    inst✝ : CommRing R
    f : R
    s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := PrimeSpectru …
    ι : Type u := Subtype fun x => Membership.mem (PrimeSpectrum.basicOpen f) x
    t : Finset ι
    a h : ι → R
    iDh : (i : ι) → Quiver.Hom (PrimeSpectrum.basicOpen (h i)) (PrimeSpectrum.basi …
    ah_ha : ∀ (i : ι), Membership.mem t i → ∀ (j : ι), Membership.mem t j → Eq (HM …
    s_eq : ∀ (i : ι), Membership.mem t i → Eq (((AlgebraicGeometry.Spec.structureS …
    ht_cover : HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) (Set.iUnion fun i = …
    n : Nat
    b : Finsupp ι R
    b_supp : Membership.mem (Finsupp.supported R R ↑t) b
    hb : Eq (t.sum fun i => HMul.hMul (b i) (h i)) (HPow.hPow f (HAdd.hAdd n 1))
    tt : Type u := ↑↑t
    i : ι
    hi : Membership.mem (↑t) i
    j : ι
    hj : Membership.mem t j
    ⊢ Eq (HMul.hMul (b j) (HMul.hMul (h j) (a i))) (HMul.hMul (a i) (HMul.hMul (b  …
  -/
  ring
  /-
    🎉 no goals
  -/


instance isIso_toBasicOpen (f : R) :
    IsIso (CommRingCat.ofHom (toBasicOpen R f)) :=
  haveI : IsIso ((forget CommRingCat).map (CommRingCat.ofHom (toBasicOpen R f))) :=
    (isIso_iff_bijective _).mpr ⟨toBasicOpen_injective R f, toBasicOpen_surjective R f⟩
  isIso_of_reflects_iso _ (forget CommRingCat)


/-- The ring isomorphism between the structure sheaf on `basicOpen f` and the localization of `R`
at the submonoid of powers of `f`. -/
def basicOpenIso (f : R) :
    (structureSheaf R).1.obj (op (PrimeSpectrum.basicOpen f)) ≅
    CommRingCat.of (Localization.Away f) :=
  (asIso (CommRingCat.ofHom (toBasicOpen R f))).symm


instance stalkAlgebra (p : PrimeSpectrum R) : Algebra R ((structureSheaf R).presheaf.stalk p) :=
  (toStalk R p).hom.toAlgebra


@[simp]
theorem stalkAlgebra_map (p : PrimeSpectrum R) (r : R) :
    algebraMap R ((structureSheaf R).presheaf.stalk p) r = toStalk R p r :=
  rfl


/-- Stalk of the structure sheaf at a prime p as localization of R -/
instance IsLocalization.to_stalk (p : PrimeSpectrum R) :
    IsLocalization.AtPrime ((structureSheaf R).presheaf.stalk p) p.asIdeal := by
  convert (IsLocalization.isLocalization_iff_of_ringEquiv (S := Localization.AtPrime p.asIdeal) _
          (stalkIso R p).symm.commRingCatIsoToRingEquiv).mp
      Localization.isLocalization
  /-
    case h.e'_5.h
    R : Type u
    inst✝ : CommRing R
    p : PrimeSpectrum R
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.stalkAlgebra R p) ((AlgebraicGeometry.S …
  -/
  apply Algebra.algebra_ext
  /-
    case h.e'_5.h.h
    R : Type u
    inst✝ : CommRing R
    p : PrimeSpectrum R
    ⊢ ∀ (r : R), Eq ((algebraMap R ↑((AlgebraicGeometry.Spec.structureSheaf R).pre …
  -/
  intro
  /-
    case h.e'_5.h.h
    R : Type u
    inst✝ : CommRing R
    p : PrimeSpectrum R
    r✝ : R
    ⊢ Eq ((algebraMap R ↑((AlgebraicGeometry.Spec.structureSheaf R).presheaf.stalk …
  -/
  rw [stalkAlgebra_map]
  /-
    case h.e'_5.h.h
    R : Type u
    inst✝ : CommRing R
    p : PrimeSpectrum R
    r✝ : R
    ⊢ Eq ((AlgebraicGeometry.StructureSheaf.toStalk R p).hom r✝) ((algebraMap R ↑( …
  -/
  congr 2
  /-
    case h.e'_5.h.h.e_a.e_self
    R : Type u
    inst✝ : CommRing R
    p : PrimeSpectrum R
    r✝ : R
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.toStalk R p) { hom := Algebra.toRingHom }
  -/
  change toStalk R p = _ ≫ (stalkIso R p).inv
  /-
    case h.e'_5.h.h.e_a.e_self
    R : Type u
    inst✝ : CommRing R
    p : PrimeSpectrum R
    r✝ : R
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.toStalk R p) (CategoryTheory.CategorySt …
  -/
  rw [Iso.eq_comp_inv]
  /-
    case h.e'_5.h.h.e_a.e_self
    R : Type u
    inst✝ : CommRing R
    p : PrimeSpectrum R
    r✝ : R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toS …
  -/
  exact toStalk_comp_stalkToFiberRingHom R p
  /-
    🎉 no goals
  -/


instance openAlgebra (U : (Opens (PrimeSpectrum R))ᵒᵖ) : Algebra R ((structureSheaf R).val.obj U) :=
  (toOpen R (unop U)).hom.toAlgebra


@[simp]
theorem openAlgebra_map (U : (Opens (PrimeSpectrum R))ᵒᵖ) (r : R) :
    algebraMap R ((structureSheaf R).val.obj U) r = toOpen R (unop U) r :=
  rfl


/-- Sections of the structure sheaf of Spec R on a basic open as localization of R -/
instance IsLocalization.to_basicOpen (r : R) :
    IsLocalization.Away r ((structureSheaf R).val.obj (op <| PrimeSpectrum.basicOpen r)) := by
  convert (IsLocalization.isLocalization_iff_of_ringEquiv (S := Localization.Away r) _
      (basicOpenIso R r).symm.commRingCatIsoToRingEquiv).mp
      Localization.isLocalization
  /-
    case h.e'_3
    R : Type u
    inst✝ : CommRing R
    r : R
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.openAlgebra R { unop := PrimeSpectrum.b …
  -/
  apply Algebra.algebra_ext
  /-
    case h.e'_3.h
    R : Type u
    inst✝ : CommRing R
    r : R
    ⊢ ∀ (r_1 : R), Eq ((algebraMap R ↑((AlgebraicGeometry.Spec.structureSheaf R).v …
  -/
  intro x
  /-
    case h.e'_3.h
    R : Type u
    inst✝ : CommRing R
    r x : R
    ⊢ Eq ((algebraMap R ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop …
  -/
  congr 1
  /-
    case h.e'_3.h.e_a
    R : Type u
    inst✝ : CommRing R
    r x : R
    ⊢ Eq (algebraMap R ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop  …
  -/
  exact (localization_toBasicOpen R r).symm
  /-
    🎉 no goals
  -/


instance to_basicOpen_epi (r : R) : Epi (toOpen R (PrimeSpectrum.basicOpen r)) :=
  ⟨fun _ _ h => CommRingCat.hom_ext (IsLocalization.ringHom_ext (Submonoid.powers r)
    (CommRingCat.hom_ext_iff.mp h))⟩


@[elementwise]
theorem to_global_factors :
    toOpen R ⊤ =
      CommRingCat.ofHom (algebraMap R (Localization.Away (1 : R))) ≫
        CommRingCat.ofHom (toBasicOpen R (1 : R)) ≫
        (structureSheaf R).1.map (eqToHom PrimeSpectrum.basicOpen_one.symm).op := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.toOpen R Top.top) (CategoryTheory.Categ …
  -/
  rw [← Category.assoc]
  change toOpen R ⊤ =
    (CommRingCat.ofHom <| (toBasicOpen R 1).comp (algebraMap R (Localization.Away 1))) ≫
    (structureSheaf R).1.map (eqToHom _).op
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.toOpen R Top.top) (CategoryTheory.Categ …
  -/
  rw [localization_toBasicOpen R, CommRingCat.ofHom_hom, toOpen_res]
  /-
    🎉 no goals
  -/


instance isIso_to_global : IsIso (toOpen R ⊤) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.StructureSheaf.toOpen R Top.top)
  -/
  let hom := CommRingCat.ofHom (algebraMap R (Localization.Away (1 : R)))
  haveI : IsIso hom :=
    (IsLocalization.atOne R (Localization.Away (1 : R))).toRingEquiv.toCommRingCatIso.isIso_hom
  /-
    R : Type u
    inst✝ : CommRing R
    hom : Quiver.Hom (CommRingCat.of R) (CommRingCat.of (Localization.Away 1)) :=  …
    this : CategoryTheory.IsIso hom
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.StructureSheaf.toOpen R Top.top)
  -/
  rw [to_global_factors R]
  /-
    R : Type u
    inst✝ : CommRing R
    hom : Quiver.Hom (CommRingCat.of R) (CommRingCat.of (Localization.Away 1)) :=  …
    this : CategoryTheory.IsIso hom
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The ring isomorphism between the ring `R` and the global sections `Γ(X, 𝒪ₓ)`. -/
-- Porting note: was @[simps (config := { rhsMd := Tactic.Transparency.semireducible })]
@[simps!]
def globalSectionsIso : CommRingCat.of R ≅ (structureSheaf R).1.obj (op ⊤) :=
  asIso (toOpen R ⊤)

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

@[simp]
theorem globalSectionsIso_hom (R : CommRingCat) : (globalSectionsIso R).hom = toOpen R ⊤ :=
  rfl


@[simp, reassoc, elementwise nosimp]
theorem toStalk_stalkSpecializes {R : Type*} [CommRing R] {x y : PrimeSpectrum R} (h : x ⤳ y) :
    toStalk R y ≫ (structureSheaf R).presheaf.stalkSpecializes h = toStalk R x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toS …
  -/
  dsimp [toStalk]; simp [-toOpen_germ]
                   /-
                     🎉 no goals
                   -/


@[simp, reassoc, elementwise nosimp]
theorem localizationToStalk_stalkSpecializes {R : Type*} [CommRing R] {x y : PrimeSpectrum R}
    (h : x ⤳ y) :
    StructureSheaf.localizationToStalk R y ≫ (structureSheaf R).presheaf.stalkSpecializes h =
      CommRingCat.ofHom (PrimeSpectrum.localizationMapOfSpecializes h) ≫
        StructureSheaf.localizationToStalk R x := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.loc …
  -/
  ext : 1
  /-
    case hf
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.loc …
  -/
  apply IsLocalization.ringHom_ext (S := Localization.AtPrime y.asIdeal) y.asIdeal.primeCompl
  /-
    case hf.h
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.lo …
  -/
  erw [RingHom.comp_assoc]
  /-
    case hf.h
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).presheaf.stalkSpecializes h). …
  -/
  conv_rhs => erw [RingHom.comp_assoc]
  /-
    case hf.h
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).presheaf.stalkSpecializes h). …
  -/
  dsimp [CommRingCat.ofHom, localizationToStalk, PrimeSpectrum.localizationMapOfSpecializes]
  /-
    case hf.h
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).presheaf.stalkSpecializes h). …
  -/
  rw [IsLocalization.lift_comp, IsLocalization.lift_comp, IsLocalization.lift_comp]
  /-
    case hf.h
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (((AlgebraicGeometry.Spec.structureSheaf R).presheaf.stalkSpecializes h). …
  -/
  exact CommRingCat.hom_ext_iff.mp (toStalk_stalkSpecializes h)
  /-
    🎉 no goals
  -/


@[simp, reassoc, elementwise nosimp]
theorem stalkSpecializes_stalk_to_fiber {R : Type*} [CommRing R] {x y : PrimeSpectrum R}
    (h : x ⤳ y) :
    (structureSheaf R).presheaf.stalkSpecializes h ≫ StructureSheaf.stalkToFiberRingHom R x =
      StructureSheaf.stalkToFiberRingHom R y ≫
      (CommRingCat.ofHom (PrimeSpectrum.localizationMapOfSpecializes h)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Spec.structureShe …
  -/
  change _ ≫ (StructureSheaf.stalkIso R x).hom = (StructureSheaf.stalkIso R y).hom ≫ _
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Spec.structureShe …
  -/
  rw [← Iso.eq_comp_inv, Category.assoc, ← Iso.inv_comp_eq]
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : PrimeSpectrum R
    h : Specializes x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.sta …
  -/
  exact localizationToStalk_stalkSpecializes h
  /-
    🎉 no goals
  -/


/--
Given a ring homomorphism `f : R →+* S`, an open set `U` of the prime spectrum of `R` and an open
set `V` of the prime spectrum of `S`, such that `V ⊆ (comap f) ⁻¹' U`, we can push a section `s`
on `U` to a section on `V`, by composing with `Localization.localRingHom _ _ f` from the left and
`comap f` from the right. Explicitly, if `s` evaluates on `comap f p` to `a / b`, its image on `V`
evaluates on `p` to `f(a) / f(b)`.

At the moment, we work with arbitrary dependent functions `s : Π x : U, Localizations R x`. Below,
we prove the predicate `isLocallyFraction` is preserved by this map, hence it can be extended to
a morphism between the structure sheaves of `R` and `S`.
-/
def comapFun (f : R →+* S) (U : Opens (PrimeSpectrum.Top R)) (V : Opens (PrimeSpectrum.Top S))
    (hUV : V.1 ⊆ PrimeSpectrum.comap f ⁻¹' U.1) (s : ∀ x : U, Localizations R x) (y : V) :
    Localizations S y :=
  Localization.localRingHom (PrimeSpectrum.comap f y.1).asIdeal _ f rfl
    (s ⟨PrimeSpectrum.comap f y.1, hUV y.2⟩ : _)


theorem comapFunIsLocallyFraction (f : R →+* S) (U : Opens (PrimeSpectrum.Top R))
    (V : Opens (PrimeSpectrum.Top S)) (hUV : V.1 ⊆ PrimeSpectrum.comap f ⁻¹' U.1)
    (s : ∀ x : U, Localizations R x) (hs : (isLocallyFraction R).toPrelocalPredicate.pred s) :
    (isLocallyFraction S).toPrelocalPredicate.pred (comapFun f U V hUV s) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type u
    inst✝ : CommRing S
    f : RingHom R S
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
    s : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hs : (AlgebraicGeometry.StructureSheaf.isLocallyFraction R).pred s
    ⊢ (AlgebraicGeometry.StructureSheaf.isLocallyFraction S).pred (AlgebraicGeomet …
  -/
  rintro ⟨p, hpV⟩
  -- Since `s` is locally fraction, we can find a neighborhood `W` of `PrimeSpectrum.comap f p`
  -- in `U`, such that `s = a / b` on `W`, for some ring elements `a, b : R`.
  /-
    case mk
    R : Type u
    inst✝¹ : CommRing R
    S : Type u
    inst✝ : CommRing S
    f : RingHom R S
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
    s : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hs : (AlgebraicGeometry.StructureSheaf.isLocallyFraction R).pred s
    p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hpV : Membership.mem V p
    ⊢ Exists fun V_1 => Exists fun x => Exists fun i => (AlgebraicGeometry.Structu …
  -/
  rcases hs ⟨PrimeSpectrum.comap f p, hUV hpV⟩ with ⟨W, m, iWU, a, b, h_frac⟩
  -- We claim that we can write our new section as the fraction `f a / f b` on the neighborhood
  -- `(comap f) ⁻¹ W ⊓ V` of `p`.
  /-
    case mk.intro.intro.intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type u
    inst✝ : CommRing S
    f : RingHom R S
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
    s : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hs : (AlgebraicGeometry.StructureSheaf.isLocallyFraction R).pred s
    p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hpV : Membership.mem V p
    W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    m : Membership.mem W ↑⟨(PrimeSpectrum.comap f) p, ⋯⟩
    iWU : Quiver.Hom W U
    a b : R
    h_frac : ∀ (x : Subtype fun x => Membership.mem W x), And (Not (Membership.mem …
    ⊢ Exists fun V_1 => Exists fun x => Exists fun i => (AlgebraicGeometry.Structu …
  -/
  refine ⟨Opens.comap (PrimeSpectrum.comap f) W ⊓ V, ⟨m, hpV⟩, Opens.infLERight _ _, f a, f b, ?_⟩
  /-
    case mk.intro.intro.intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type u
    inst✝ : CommRing S
    f : RingHom R S
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
    s : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hs : (AlgebraicGeometry.StructureSheaf.isLocallyFraction R).pred s
    p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hpV : Membership.mem V p
    W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    m : Membership.mem W ↑⟨(PrimeSpectrum.comap f) p, ⋯⟩
    iWU : Quiver.Hom W U
    a b : R
    h_frac : ∀ (x : Subtype fun x => Membership.mem W x), And (Not (Membership.mem …
    ⊢ ∀ (x : Subtype fun x => Membership.mem (Min.min ((TopologicalSpace.Opens.com …
  -/
  rintro ⟨q, ⟨hqW, hqV⟩⟩
  /-
    case mk.intro.intro.intro.intro.intro.mk.intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type u
    inst✝ : CommRing S
    f : RingHom R S
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
    s : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hs : (AlgebraicGeometry.StructureSheaf.isLocallyFraction R).pred s
    p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hpV : Membership.mem V p
    W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    m : Membership.mem W ↑⟨(PrimeSpectrum.comap f) p, ⋯⟩
    iWU : Quiver.Hom W U
    a b : R
    h_frac : ∀ (x : Subtype fun x => Membership.mem W x), And (Not (Membership.mem …
    q : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hqW : Membership.mem (↑((TopologicalSpace.Opens.comap (PrimeSpectrum.comap f)) …
    hqV : Membership.mem (↑V) q
    ⊢ And (Not (Membership.mem (↑⟨q, ⋯⟩).asIdeal (f b))) (Eq (HMul.hMul ((fun x => …
  -/
  specialize h_frac ⟨PrimeSpectrum.comap f q, hqW⟩
  /-
    case mk.intro.intro.intro.intro.intro.mk.intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type u
    inst✝ : CommRing S
    f : RingHom R S
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
    s : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hs : (AlgebraicGeometry.StructureSheaf.isLocallyFraction R).pred s
    p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hpV : Membership.mem V p
    W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    m : Membership.mem W ↑⟨(PrimeSpectrum.comap f) p, ⋯⟩
    iWU : Quiver.Hom W U
    a b : R
    q : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hqW : Membership.mem (↑((TopologicalSpace.Opens.comap (PrimeSpectrum.comap f)) …
    hqV : Membership.mem (↑V) q
    h_frac : And (Not (Membership.mem (↑⟨(PrimeSpectrum.comap f) q, hqW⟩).asIdeal  …
    ⊢ And (Not (Membership.mem (↑⟨q, ⋯⟩).asIdeal (f b))) (Eq (HMul.hMul ((fun x => …
  -/
  refine ⟨h_frac.1, ?_⟩
  /-
    case mk.intro.intro.intro.intro.intro.mk.intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type u
    inst✝ : CommRing S
    f : RingHom R S
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
    s : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hs : (AlgebraicGeometry.StructureSheaf.isLocallyFraction R).pred s
    p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hpV : Membership.mem V p
    W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    m : Membership.mem W ↑⟨(PrimeSpectrum.comap f) p, ⋯⟩
    iWU : Quiver.Hom W U
    a b : R
    q : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hqW : Membership.mem (↑((TopologicalSpace.Opens.comap (PrimeSpectrum.comap f)) …
    hqV : Membership.mem (↑V) q
    h_frac : And (Not (Membership.mem (↑⟨(PrimeSpectrum.comap f) q, hqW⟩).asIdeal  …
    ⊢ Eq (HMul.hMul ((fun x => AlgebraicGeometry.StructureSheaf.comapFun f U V hUV …
  -/
  dsimp only [comapFun]
  erw [← Localization.localRingHom_to_map (PrimeSpectrum.comap f q).asIdeal, ← RingHom.map_mul,
    h_frac.2, Localization.localRingHom_to_map]
  /-
    case mk.intro.intro.intro.intro.intro.mk.intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type u
    inst✝ : CommRing S
    f : RingHom R S
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
    s : (x : Subtype fun x => Membership.mem U x) → AlgebraicGeometry.StructureShe …
    hs : (AlgebraicGeometry.StructureSheaf.isLocallyFraction R).pred s
    p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hpV : Membership.mem V p
    W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    m : Membership.mem W ↑⟨(PrimeSpectrum.comap f) p, ⋯⟩
    iWU : Quiver.Hom W U
    a b : R
    q : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
    hqW : Membership.mem (↑((TopologicalSpace.Opens.comap (PrimeSpectrum.comap f)) …
    hqV : Membership.mem (↑V) q
    h_frac : And (Not (Membership.mem (↑⟨(PrimeSpectrum.comap f) q, hqW⟩).asIdeal  …
    ⊢ Eq ((algebraMap S (Localization.AtPrime q.asIdeal)) (f a)) ((algebraMap S (A …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- For a ring homomorphism `f : R →+* S` and open sets `U` and `V` of the prime spectra of `R` and
`S` such that `V ⊆ (comap f) ⁻¹ U`, the induced ring homomorphism from the structure sheaf of `R`
at `U` to the structure sheaf of `S` at `V`.

Explicitly, this map is given as follows: For a point `p : V`, if the section `s` evaluates on `p`
to the fraction `a / b`, its image on `V` evaluates on `p` to the fraction `f(a) / f(b)`.
-/
def comap (f : R →+* S) (U : Opens (PrimeSpectrum.Top R)) (V : Opens (PrimeSpectrum.Top S))
    (hUV : V.1 ⊆ PrimeSpectrum.comap f ⁻¹' U.1) :
    (structureSheaf R).1.obj (op U) →+* (structureSheaf S).1.obj (op V) where
  toFun s := ⟨comapFun f U V hUV s.1, comapFunIsLocallyFraction f U V hUV s.1 s.2⟩
  map_one' :=
    Subtype.ext <|
      funext fun p => by
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (↑((fun s => ⟨AlgebraicGeometry.StructureSheaf.comapFun f U V hUV ↑s, ⋯⟩) …
        -/
        dsimp
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (AlgebraicGeometry.StructureSheaf.comapFun f U V hUV (↑1) p) (↑1 p)
        -/
        rw [comapFun, (sectionsSubring R (op U)).coe_one, Pi.one_apply, RingHom.map_one]
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq 1 (↑1 p)
        -/
        rfl
        /-
          🎉 no goals
        -/
  map_zero' :=
    Subtype.ext <|
      funext fun p => by
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (↑((↑{ toFun := fun s => ⟨AlgebraicGeometry.StructureSheaf.comapFun f U V …
        -/
        dsimp
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (AlgebraicGeometry.StructureSheaf.comapFun f U V hUV (↑0) p) (↑0 p)
        -/
        rw [comapFun, (sectionsSubring R (op U)).coe_zero, Pi.zero_apply, RingHom.map_zero]
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq 0 (↑0 p)
        -/
        rfl
        /-
          🎉 no goals
        -/
  map_add' s t :=
    Subtype.ext <|
      funext fun p => by
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          s t : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (↑((↑{ toFun := fun s => ⟨AlgebraicGeometry.StructureSheaf.comapFun f U V …
        -/
        dsimp
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          s t : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (↑({ toFun := fun s => ⟨AlgebraicGeometry.StructureSheaf.comapFun f U V h …
        -/
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          s t : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (AlgebraicGeometry.StructureSheaf.comapFun f U V hUV (↑(HAdd.hAdd s t)) p …
        -/
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          s t : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (AlgebraicGeometry.StructureSheaf.comapFun f U V hUV (↑(HMul.hMul s t)) p …
        -/
        rw [comapFun, (sectionsSubring R (op U)).coe_add, Pi.add_apply, RingHom.map_add]
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          s t : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (HMul.hMul ((Localization.localRingHom ((PrimeSpectrum.comap f) ↑p).asIde …
        -/
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
          s t : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (HAdd.hAdd ((Localization.localRingHom ((PrimeSpectrum.comap f) ↑p).asIde …
        -/
        /-
          🎉 no goals
        -/
        rfl
        /-
          🎉 no goals
        -/
  map_mul' s t :=
    Subtype.ext <|
      funext fun p => by
        dsimp
        rw [comapFun, (sectionsSubring R (op U)).coe_mul, Pi.mul_apply, RingHom.map_mul]
        rfl


@[simp]
theorem comap_apply (f : R →+* S) (U : Opens (PrimeSpectrum.Top R))
    (V : Opens (PrimeSpectrum.Top S)) (hUV : V.1 ⊆ PrimeSpectrum.comap f ⁻¹' U.1)
    (s : (structureSheaf R).1.obj (op U)) (p : V) :
    (comap f U V hUV s).1 p =
      Localization.localRingHom (PrimeSpectrum.comap f p.1).asIdeal _ f rfl
        (s.1 ⟨PrimeSpectrum.comap f p.1, hUV p.2⟩ : _) :=
  rfl


theorem comap_const (f : R →+* S) (U : Opens (PrimeSpectrum.Top R))
    (V : Opens (PrimeSpectrum.Top S)) (hUV : V.1 ⊆ PrimeSpectrum.comap f ⁻¹' U.1) (a b : R)
    (hb : ∀ x : PrimeSpectrum R, x ∈ U → b ∈ x.asIdeal.primeCompl) :
    comap f U V hUV (const R a b U hb) =
      const S (f a) (f b) V fun p hpV => hb (PrimeSpectrum.comap f p) (hUV hpV) :=
  Subtype.eq <|
    funext fun p => by
      /-
        R : Type u
        inst✝¹ : CommRing R
        S : Type u
        inst✝ : CommRing S
        f : RingHom R S
        U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
        hUV : HasSubset.Subset V.carrier (Set.preimage (⇑(PrimeSpectrum.comap f)) U.ca …
        a b : R
        hb : ∀ (x : PrimeSpectrum R), Membership.mem U x → Membership.mem x.asIdeal.pr …
        p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
        ⊢ Eq (↑((AlgebraicGeometry.StructureSheaf.comap f U V hUV) (AlgebraicGeometry. …
      -/
      rw [comap_apply, const_apply, const_apply, Localization.localRingHom_mk']
      /-
        🎉 no goals
      -/


/-- For an inclusion `i : V ⟶ U` between open sets of the prime spectrum of `R`, the comap of the
identity from OO_X(U) to OO_X(V) equals as the restriction map of the structure sheaf.

This is a generalization of the fact that, for fixed `U`, the comap of the identity from OO_X(U)
to OO_X(U) is the identity.
-/
theorem comap_id_eq_map (U V : Opens (PrimeSpectrum.Top R)) (iVU : V ⟶ U) :
    (comap (RingHom.id R) U V fun _ hpV => leOfHom iVU <| hpV) =
      ((structureSheaf R).1.map iVU.op).hom :=
  RingHom.ext fun s => Subtype.eq <| funext fun p => by
    /-
      R : Type u
      inst✝ : CommRing R
      U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      iVU : Quiver.Hom V U
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
      ⊢ Eq (↑((AlgebraicGeometry.StructureSheaf.comap (RingHom.id R) U V ⋯) s) p) (↑ …
    -/
    rw [comap_apply]
    -- Unfortunately, we cannot use `Localization.localRingHom_id` here, because
    -- `PrimeSpectrum.comap (RingHom.id R) p` is not *definitionally* equal to `p`. Instead, we use
    -- that we can write `s` as a fraction `a/b` in a small neighborhood around `p`. Since
    -- `PrimeSpectrum.comap (RingHom.id R) p` equals `p`, it is also contained in the same
    -- neighborhood, hence `s` equals `a/b` there too.
    /-
      R : Type u
      inst✝ : CommRing R
      U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      iVU : Quiver.Hom V U
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
      ⊢ Eq ((Localization.localRingHom ((PrimeSpectrum.comap (RingHom.id R)) ↑p).asI …
    -/
    obtain ⟨W, hpW, iWU, h⟩ := s.2 (iVU p)
    /-
      case intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      iVU : Quiver.Hom V U
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
      W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hpW : Membership.mem W ↑((fun x => ⟨↑x, ⋯⟩) p)
      iWU : Quiver.Hom W (Opposite.unop { unop := U })
      h : (AlgebraicGeometry.StructureSheaf.isFractionPrelocal R).pred fun x => ↑s ( …
      ⊢ Eq ((Localization.localRingHom ((PrimeSpectrum.comap (RingHom.id R)) ↑p).asI …
    -/
    obtain ⟨a, b, h'⟩ := h.eq_mk'
    /-
      case intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      iVU : Quiver.Hom V U
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
      W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hpW : Membership.mem W ↑((fun x => ⟨↑x, ⋯⟩) p)
      iWU : Quiver.Hom W (Opposite.unop { unop := U })
      h : (AlgebraicGeometry.StructureSheaf.isFractionPrelocal R).pred fun x => ↑s ( …
      a b : R
      h' : ∀ (x : Subtype fun x => Membership.mem W x), Exists fun hs => Eq (↑s ((fu …
      ⊢ Eq ((Localization.localRingHom ((PrimeSpectrum.comap (RingHom.id R)) ↑p).asI …
    -/
    obtain ⟨hb₁, s_eq₁⟩ := h' ⟨p, hpW⟩
    obtain ⟨hb₂, s_eq₂⟩ :=
      h' ⟨PrimeSpectrum.comap (RingHom.id _) p.1, hpW⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      iVU : Quiver.Hom V U
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
      W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hpW : Membership.mem W ↑((fun x => ⟨↑x, ⋯⟩) p)
      iWU : Quiver.Hom W (Opposite.unop { unop := U })
      h : (AlgebraicGeometry.StructureSheaf.isFractionPrelocal R).pred fun x => ↑s ( …
      a b : R
      h' : ∀ (x : Subtype fun x => Membership.mem W x), Exists fun hs => Eq (↑s ((fu …
      hb₁ : Not (Membership.mem (↑⟨↑p, hpW⟩).asIdeal b)
      s_eq₁ : Eq (↑s ((fun x => ⟨↑x, ⋯⟩) ⟨↑p, hpW⟩)) (IsLocalization.mk' (Localizati …
      hb₂ : Not (Membership.mem (↑⟨(PrimeSpectrum.comap (RingHom.id R)) ↑p, hpW⟩).as …
      s_eq₂ : Eq (↑s ((fun x => ⟨↑x, ⋯⟩) ⟨(PrimeSpectrum.comap (RingHom.id R)) ↑p, h …
      ⊢ Eq ((Localization.localRingHom ((PrimeSpectrum.comap (RingHom.id R)) ↑p).asI …
    -/
    dsimp only at s_eq₁ s_eq₂
    /-
      case intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      iVU : Quiver.Hom V U
      s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
      p : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
      W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hpW : Membership.mem W ↑((fun x => ⟨↑x, ⋯⟩) p)
      iWU : Quiver.Hom W (Opposite.unop { unop := U })
      h : (AlgebraicGeometry.StructureSheaf.isFractionPrelocal R).pred fun x => ↑s ( …
      a b : R
      h' : ∀ (x : Subtype fun x => Membership.mem W x), Exists fun hs => Eq (↑s ((fu …
      hb₁ : Not (Membership.mem (↑⟨↑p, hpW⟩).asIdeal b)
      s_eq₁ : Eq (↑s ⟨↑p, ⋯⟩) (IsLocalization.mk' (Localization.AtPrime (↑p).asIdeal …
      hb₂ : Not (Membership.mem (↑⟨(PrimeSpectrum.comap (RingHom.id R)) ↑p, hpW⟩).as …
      s_eq₂ : Eq (↑s ⟨(PrimeSpectrum.comap (RingHom.id R)) ↑p, ⋯⟩) (IsLocalization.m …
      ⊢ Eq ((Localization.localRingHom ((PrimeSpectrum.comap (RingHom.id R)) ↑p).asI …
    -/
    erw [s_eq₂, Localization.localRingHom_mk', ← s_eq₁, ← res_apply _ _ _ iVU]
    /-
      🎉 no goals
    -/


/--
The comap of the identity is the identity. In this variant of the lemma, two open subsets `U` and
`V` are given as arguments, together with a proof that `U = V`. This is useful when `U` and `V`
are not definitionally equal.
-/
theorem comap_id {U V : Opens (PrimeSpectrum.Top R)} (hUV : U = V) :
                                              /-
                                                R : Type u
                                                inst✝² : CommRing R
                                                S : Type u
                                                inst✝¹ : CommRing S
                                                P : Type u
                                                inst✝ : CommRing P
                                                U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                hUV : Eq U V
                                                p : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                hpV : Membership.mem V.carrier p
                                                ⊢ Membership.mem (Set.preimage (⇑(PrimeSpectrum.comap (RingHom.id R))) U.carri …
                                              -/
    (comap (RingHom.id R) U V fun p hpV => by rwa [hUV, PrimeSpectrum.comap_id]) =
                                              /-
                                                🎉 no goals
                                              -/
                                                            /-
                                                              R : Type u
                                                              inst✝² : CommRing R
                                                              S : Type u
                                                              inst✝¹ : CommRing S
                                                              P : Type u
                                                              inst✝ : CommRing P
                                                              U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                              hUV : Eq U V
                                                              ⊢ Eq ((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U }) ((Algeb …
                                                            -/
      (eqToHom (show (structureSheaf R).1.obj (op U) = _ by rw [hUV])).hom := by
                                                            /-
                                                              🎉 no goals
                                                            -/
  /-
    R : Type u
    inst✝ : CommRing R
    U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hUV : Eq U V
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.comap (RingHom.id R) U V ⋯) (CategoryTh …
  -/
  rw [comap_id_eq_map U V (eqToHom hUV.symm), eqToHom_op, eqToHom_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_id' (U : Opens (PrimeSpectrum.Top R)) :
                                              /-
                                                R : Type u
                                                inst✝² : CommRing R
                                                S : Type u
                                                inst✝¹ : CommRing S
                                                P : Type u
                                                inst✝ : CommRing P
                                                U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                p : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                hpU : Membership.mem U.carrier p
                                                ⊢ Membership.mem (Set.preimage (⇑(PrimeSpectrum.comap (RingHom.id R))) U.carri …
                                              -/
    (comap (RingHom.id R) U U fun p hpU => by rwa [PrimeSpectrum.comap_id]) = RingHom.id _ := by
                                              /-
                                                🎉 no goals
                                              -/
  /-
    R : Type u
    inst✝ : CommRing R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    ⊢ Eq (AlgebraicGeometry.StructureSheaf.comap (RingHom.id R) U U ⋯) (RingHom.id …
  -/
  rw [comap_id rfl]; rfl
                     /-
                       🎉 no goals
                     -/


theorem comap_comp (f : R →+* S) (g : S →+* P) (U : Opens (PrimeSpectrum.Top R))
    (V : Opens (PrimeSpectrum.Top S)) (W : Opens (PrimeSpectrum.Top P))
    (hUV : ∀ p ∈ V, PrimeSpectrum.comap f p ∈ U) (hVW : ∀ p ∈ W, PrimeSpectrum.comap g p ∈ V) :
    (comap (g.comp f) U W fun p hpW => hUV (PrimeSpectrum.comap g p) (hVW p hpW)) =
      (comap g V W hVW).comp (comap f U V hUV) :=
  RingHom.ext fun s =>
    Subtype.eq <|
      funext fun p => by
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          g : RingHom S P
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top P)
          hUV : ∀ (p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)), Membership.mem V p → M …
          hVW : ∀ (p : ↑(AlgebraicGeometry.PrimeSpectrum.Top P)), Membership.mem W p → M …
          s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := W }) x
          ⊢ Eq (↑((AlgebraicGeometry.StructureSheaf.comap (g.comp f) U W ⋯) s) p) (↑(((A …
        -/
        rw [comap_apply]
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          g : RingHom S P
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top P)
          hUV : ∀ (p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)), Membership.mem V p → M …
          hVW : ∀ (p : ↑(AlgebraicGeometry.PrimeSpectrum.Top P)), Membership.mem W p → M …
          s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := W }) x
          ⊢ Eq ((Localization.localRingHom ((PrimeSpectrum.comap (g.comp f)) ↑p).asIdeal …
        -/
        rw [Localization.localRingHom_comp _ (PrimeSpectrum.comap g p.1).asIdeal] <;>
        -- refl works here, because `PrimeSpectrum.comap (g.comp f) p` is defeq to
        -- `PrimeSpectrum.comap f (PrimeSpectrum.comap g p)`
        /-
          R : Type u
          inst✝² : CommRing R
          S : Type u
          inst✝¹ : CommRing S
          P : Type u
          inst✝ : CommRing P
          f : RingHom R S
          g : RingHom S P
          U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
          V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top S)
          W : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top P)
          hUV : ∀ (p : ↑(AlgebraicGeometry.PrimeSpectrum.Top S)), Membership.mem V p → M …
          hVW : ∀ (p : ↑(AlgebraicGeometry.PrimeSpectrum.Top P)), Membership.mem W p → M …
          s : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj { unop := U })
          p : Subtype fun x => Membership.mem (Opposite.unop { unop := W }) x
          ⊢ Eq (((Localization.localRingHom ((PrimeSpectrum.comap g) ↑p).asIdeal (↑p).as …
        -/
        /-
          🎉 no goals
        -/
        rfl
        /-
          🎉 no goals
        -/


@[elementwise, reassoc]
theorem toOpen_comp_comap (f : R →+* S) (U : Opens (PrimeSpectrum.Top R)) :
    (toOpen R U ≫ CommRingCat.ofHom (comap f U (Opens.comap (PrimeSpectrum.comap f) U)
        fun _ => id)) =
      CommRingCat.ofHom f ≫ toOpen S _ :=
  CommRingCat.hom_ext <| RingHom.ext fun _ => Subtype.eq <| funext fun _ =>
    Localization.localRingHom_to_map _ _ _ _ _


lemma comap_basicOpen (f : R →+* S) (x : R) :
    comap f (PrimeSpectrum.basicOpen x) (PrimeSpectrum.basicOpen (f x))
        (PrimeSpectrum.comap_basicOpen f x).le =
      IsLocalization.map (M := .powers x) (T := .powers (f x)) _ f
        (Submonoid.powers_le.mpr (Submonoid.mem_powers _)) :=
  IsLocalization.ringHom_ext (.powers x) <| by
    /-
      R : Type u
      inst✝¹ : CommRing R
      S : Type u
      inst✝ : CommRing S
      f : RingHom R S
      x : R
      ⊢ Eq ((AlgebraicGeometry.StructureSheaf.comap f (PrimeSpectrum.basicOpen x) (P …
    -/
    simpa [CommRingCat.hom_ext_iff] using toOpen_comp_comap f _
    /-
      🎉 no goals
    -/


