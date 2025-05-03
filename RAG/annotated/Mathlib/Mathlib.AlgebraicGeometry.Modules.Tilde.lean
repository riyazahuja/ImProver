/-- For an `R`-module `M` and a point `P` in `Spec R`, `Localizations P` is the localized module
`M` at the prime ideal `P`. -/
abbrev Localizations (P : PrimeSpectrum.Top R) :=
LocalizedModule P.asIdeal.primeCompl M


/-- For any open subset `U ⊆ Spec R`, `IsFraction` is the predicate expressing that a function
`f : ∏_{x ∈ U}, Mₓ` is such that for any `𝔭 ∈ U`, `f 𝔭 = m / s` for some `m : M` and `s ∉ 𝔭`.
In short `f` is a fraction on `U`. -/
def isFraction {U : Opens (PrimeSpectrum R)} (f : ∀ 𝔭 : U, Localizations M 𝔭.1) : Prop :=
  ∃ (m : M) (s : R),
    ∀ x : U, ¬s ∈ x.1.asIdeal ∧ s • f x = LocalizedModule.mkLinearMap x.1.asIdeal.primeCompl M m


/--
The property of a function `f : ∏_{x ∈ U}, Mₓ` being a fraction is stable under restriction.
-/
def isFractionPrelocal : PrelocalPredicate (Localizations M) where
  pred {_} f := isFraction M f
            /-
              R : Type u
              inst✝ : CommRing R
              M : ModuleCat R
              ⊢ ∀ {U V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)} (i …
            -/
  res := by rintro V U i f ⟨m, s, w⟩; exact ⟨m, s, fun x => w (i x)⟩
                                      /-
                                        🎉 no goals
                                      -/


/--
For any open subset `U ⊆ Spec R`, `IsLocallyFraction` is the predicate expressing that a function
`f : ∏_{x ∈ U}, Mₓ` is such that for any `𝔭 ∈ U`, there exists an open neighbourhood `V ∋ 𝔭`, such
that for any `𝔮 ∈ V`, `f 𝔮 = m / s` for some `m : M` and `s ∉ 𝔮`.
In short `f` is locally a fraction on `U`.
-/
def isLocallyFraction : LocalPredicate (Localizations M) := (isFractionPrelocal M).sheafify


@[simp]
theorem isLocallyFraction_pred {U : Opens (PrimeSpectrum.Top R)}
    (f : ∀ x : U, Localizations M x) :
    (isLocallyFraction M).pred f =
      ∀ y : U,
        ∃ (V : _) (_ : y.1 ∈ V) (i : V ⟶ U),
          ∃ (m : M) (s: R), ∀ x : V, ¬s ∈ x.1.asIdeal ∧ s • f (i x) =
            LocalizedModule.mkLinearMap x.1.asIdeal.primeCompl M m :=
  rfl

/- M_x is an O_SpecR(U)-module when x is in U -/

noncomputable instance (U : (Opens (PrimeSpectrum.Top R))ᵒᵖ) (x : U.unop):
    Module ((Spec.structureSheaf R).val.obj U) (Localizations M x):=
  Module.compHom (R := (Localization.AtPrime x.1.asIdeal)) _
    ((StructureSheaf.openToLocalization R U.unop x x.2).hom)


@[simp]
lemma sections_smul_localizations_def
    {U : (Opens (PrimeSpectrum.Top R))ᵒᵖ} (x : U.unop)
    (r : (Spec.structureSheaf R).val.obj U)
    (m : Localizations M ↑x) :
  r • m = r.1 x • m := rfl


/--
For any `R`-module `M` and any open subset `U ⊆ Spec R`, `M^~(U)` is an `𝒪_{Spec R}(U)`-submodule
of `∏_{𝔭 ∈ U} M_𝔭`. -/
def sectionsSubmodule (U : (Opens (PrimeSpectrum R))ᵒᵖ) :
    Submodule ((Spec.structureSheaf R).1.obj U) (∀ x : U.unop, Localizations M x.1) where
  carrier := { f | (isLocallyFraction M).pred f }
  zero_mem' x := ⟨unop U, x.2, 𝟙 _, 0, 1, fun y =>
                                                   /-
                                                     R : Type u
                                                     inst✝ : CommRing R
                                                     M : ModuleCat R
                                                     U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
                                                     x : Subtype fun x => Membership.mem (Opposite.unop U) x
                                                     y : Subtype fun x => Membership.mem (Opposite.unop U) x
                                                     ⊢ Eq (HSMul.hSMul 1 ((fun x => 0 ((fun x => ⟨↑x, ⋯⟩) x)) y)) ((LocalizedModule …
                                                   -/
    ⟨Ideal.ne_top_iff_one _ |>.1 y.1.isPrime.1, by simp⟩⟩
    /-
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      ⊢ ∀ {a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleC …
    -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    /-
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
      ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      ⊢ Exists fun V => Exists fun x => Exists fun i => (ModuleCat.Tilde.isFractionP …
    -/
  add_mem' := by
    /-
      case intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
      ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (ModuleCat.Tilde.isFractionP …
    -/
    intro a b ha hb x
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
      ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb : ↑M
      sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (ModuleCat.Tilde.isFractionP …
    -/
    rcases ha x with ⟨Va, ma, ia, ra, sa, wa⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
      ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb : ↑M
      sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      ⊢ ∀ (x : Subtype fun x => Membership.mem (Min.min Va Vb) x), And (Not (Members …
    -/
    rcases hb x with ⟨Vb, mb, ib, rb, sb, wb⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
      ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb : ↑M
      sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HSMul.hSMul ( …
    -/
    refine ⟨Va ⊓ Vb, ⟨ma, mb⟩, Opens.infLELeft _ _ ≫ ia,  sb• ra+ sa•rb , sa * sb, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
      ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb : ↑M
      sb : R
      wb : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
      wa : Eq (HSMul.hSMul sa ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HSMul.hSMul ( …
    -/
    intro y
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
      ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
      Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mb : Membership.mem Vb ↑x
      ib : Quiver.Hom Vb (Opposite.unop U)
      rb : ↑M
      sb : R
      wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
      y : Subtype fun x => Membership.mem (Min.min Va Vb) x
      nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
      wa : Eq (HSMul.hSMul sa ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
      nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
      wb : Eq (HSMul.hSMul sb ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))) (Eq (HSMul.hSMul ( …
    -/
    rcases wa (Opens.infLELeft _ _ y : Va) with ⟨nma, wa⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.l …
        R : Type u
        inst✝ : CommRing R
        M : ModuleCat R
        U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
        ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
        hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra : ↑M
        sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb : ↑M
        sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HSMul.hSMul sa ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HSMul.hSMul sb ((fun x => b ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
        ⊢ Not (Membership.mem (↑y).asIdeal (HMul.hMul sa sb))
      -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    rcases wb (Opens.infLERight _ _ y : Vb) with ⟨nmb, wb⟩
                                                  /-
                                                    🎉 no goals
                                                  -/
    fconstructor
    · intro H; cases y.1.isPrime.mem_or_mem H <;> contradiction
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        M : ModuleCat R
        U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
        ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
        hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra : ↑M
        sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb : ↑M
        sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HSMul.hSMul sa (a ⟨↑y, ⋯⟩)) ((LocalizedModule.mkLinearMap (↑y).asIdea …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HSMul.hSMul sb (b ⟨↑y, ⋯⟩)) ((LocalizedModule.mkLinearMap (↑y).asIdea …
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul sa sb) (a ⟨↑y, ⋯⟩)) (HSMul.hSMul (HMul …
      -/
    · simp only [Opens.coe_inf, Pi.add_apply, smul_add, map_add,
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        M : ModuleCat R
        U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
        ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
        hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra : ↑M
        sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb : ↑M
        sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HSMul.hSMul sa (a ⟨↑y, ⋯⟩)) ((LocalizedModule.mkLinearMap (↑y).asIdea …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HSMul.hSMul sb (b ⟨↑y, ⋯⟩)) ((LocalizedModule.mkLinearMap (↑y).asIdea …
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul sa sb) (a ⟨↑y, ⋯⟩)) (HSMul.hSMul (HMul …
      -/
        LinearMapClass.map_smul] at wa wb ⊢
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        M : ModuleCat R
        U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
        a b : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Ti …
        ha : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
        hb : Membership.mem (setOf fun f => (ModuleCat.Tilde.isLocallyFraction M).pred …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra : ↑M
        sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vb : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mb : Membership.mem Vb ↑x
        ib : Quiver.Hom Vb (Opposite.unop U)
        rb : ↑M
        sb : R
        wb✝ : ∀ (x : Subtype fun x => Membership.mem Vb x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vb) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HSMul.hSMul sa (a ⟨↑y, ⋯⟩)) ((LocalizedModule.mkLinearMap (↑y).asIdea …
        nmb : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sb)
        wb : Eq (HSMul.hSMul sb (b ⟨↑y, ⋯⟩)) ((LocalizedModule.mkLinearMap (↑y).asIdea …
        ⊢ Eq (HMul.hMul sa sb) (HMul.hMul sb sa)
      -/
      rw [← wa, ← wb, ← mul_smul, ← mul_smul]
      /-
        🎉 no goals
      -/
      congr 2
      simp [mul_comm]
  smul_mem' := by
    /-
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      ⊢ ∀ (c : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)) {x : (x : Sub …
    -/
    intro r a ha x
    /-
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
      ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      ⊢ Exists fun V => Exists fun x => Exists fun i => (ModuleCat.Tilde.isFractionP …
    -/
    rcases ha x with ⟨Va, ma, ia, ra, sa, wa⟩
    /-
      case intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
      ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (ModuleCat.Tilde.isFractionP …
    -/
    rcases r.2 x with ⟨Vr, mr, ir, rr, sr, wr⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
      ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mr : Membership.mem Vr ↑x
      ir : Quiver.Hom Vr (Opposite.unop U)
      rr sr : R
      wr : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem (↑ …
      ⊢ Exists fun V => Exists fun x => Exists fun i => (ModuleCat.Tilde.isFractionP …
    -/
    refine ⟨Va ⊓ Vr, ⟨ma, mr⟩, Opens.infLELeft _ _ ≫ ia, rr•ra, sr*sa, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
      ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mr : Membership.mem Vr ↑x
      ir : Quiver.Hom Vr (Opposite.unop U)
      rr sr : R
      wr : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem (↑ …
      ⊢ ∀ (x : Subtype fun x => Membership.mem (Min.min Va Vr) x), And (Not (Members …
    -/
    intro y
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
      ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem (↑ …
      Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mr : Membership.mem Vr ↑x
      ir : Quiver.Hom Vr (Opposite.unop U)
      rr sr : R
      wr : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Min.min Va Vr) x
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sr sa))) (Eq (HSMul.hSMul ( …
    -/
    rcases wa (Opens.infLELeft _ _ y : Va) with ⟨nma, wa⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
      ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
      Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mr : Membership.mem Vr ↑x
      ir : Quiver.Hom Vr (Opposite.unop U)
      rr sr : R
      wr : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Min.min Va Vr) x
      nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
      wa : Eq (HSMul.hSMul sa ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sr sa))) (Eq (HSMul.hSMul ( …
    -/
    rcases wr (Opens.infLERight _ _ y) with ⟨nmr, wr⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
      r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
      a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
      ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
      x : Subtype fun x => Membership.mem (Opposite.unop U) x
      Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      ma : Membership.mem Va ↑x
      ia : Quiver.Hom Va (Opposite.unop U)
      ra : ↑M
      sa : R
      wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
      Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      mr : Membership.mem Vr ↑x
      ir : Quiver.Hom Vr (Opposite.unop U)
      rr sr : R
      wr✝ : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem ( …
      y : Subtype fun x => Membership.mem (Min.min Va Vr) x
      nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
      wa : Eq (HSMul.hSMul sa ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
      nmr : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sr)
      wr : Eq (HMul.hMul ((fun x => ↑r ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y …
      ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul sr sa))) (Eq (HSMul.hSMul ( …
    -/
    fconstructor
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.l …
        R : Type u
        inst✝ : CommRing R
        M : ModuleCat R
        U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
        r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
        a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
        ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra : ↑M
        sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mr : Membership.mem Vr ↑x
        ir : Quiver.Hom Vr (Opposite.unop U)
        rr sr : R
        wr✝ : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vr) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HSMul.hSMul sa ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
        nmr : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sr)
        wr : Eq (HMul.hMul ((fun x => ↑r ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y …
        ⊢ Not (Membership.mem (↑y).asIdeal (HMul.hMul sr sa))
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
        M : ModuleCat R
        U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
        r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
        a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
        ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra : ↑M
        sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mr : Membership.mem Vr ↑x
        ir : Quiver.Hom Vr (Opposite.unop U)
        rr sr : R
        wr✝ : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vr) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HSMul.hSMul sa ((fun x => a ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯ …
        nmr : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sr)
        wr : Eq (HMul.hMul ((fun x => ↑r ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) y …
        ⊢ Eq (HSMul.hSMul (HMul.hMul sr sa) ((fun x => HSMul.hSMul r a ((fun x => ⟨↑x, …
      -/
    · simp only [Opens.coe_inf, Pi.smul_apply, LinearMapClass.map_smul] at wa wr ⊢
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        M : ModuleCat R
        U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
        r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
        a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
        ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra : ↑M
        sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mr : Membership.mem Vr ↑x
        ir : Quiver.Hom Vr (Opposite.unop U)
        rr sr : R
        wr✝ : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vr) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HSMul.hSMul sa (a ⟨↑y, ⋯⟩)) ((LocalizedModule.mkLinearMap (↑y).asIdea …
        nmr : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sr)
        wr : Eq (HMul.hMul (↑r ⟨↑y, ⋯⟩) ((algebraMap R (AlgebraicGeometry.StructureShe …
        ⊢ Eq (HSMul.hSMul (HMul.hMul sr sa) (HSMul.hSMul r (a ⟨↑y, ⋯⟩))) (HSMul.hSMul  …
      -/
      rw [mul_comm, ← Algebra.smul_def] at wr
      rw [sections_smul_localizations_def, ← wa, ← mul_smul, ← smul_assoc, mul_comm sr, mul_smul,
        wr, mul_comm rr, Algebra.smul_def, ← map_mul]
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
        R : Type u
        inst✝ : CommRing R
        M : ModuleCat R
        U : Opposite (TopologicalSpace.Opens (PrimeSpectrum R))
        r : ↑((AlgebraicGeometry.Spec.structureSheaf R).val.obj U)
        a : (x : Subtype fun x => Membership.mem (Opposite.unop U) x) → ModuleCat.Tild …
        ha : Membership.mem { carrier := setOf fun f => (ModuleCat.Tilde.isLocallyFrac …
        x : Subtype fun x => Membership.mem (Opposite.unop U) x
        Va : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        ma : Membership.mem Va ↑x
        ia : Quiver.Hom Va (Opposite.unop U)
        ra : ↑M
        sa : R
        wa✝ : ∀ (x : Subtype fun x => Membership.mem Va x), And (Not (Membership.mem ( …
        Vr : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        mr : Membership.mem Vr ↑x
        ir : Quiver.Hom Vr (Opposite.unop U)
        rr sr : R
        wr✝ : ∀ (x : Subtype fun x => Membership.mem Vr x), And (Not (Membership.mem ( …
        y : Subtype fun x => Membership.mem (Min.min Va Vr) x
        nma : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sa)
        wa : Eq (HSMul.hSMul sa (a ⟨↑y, ⋯⟩)) ((LocalizedModule.mkLinearMap (↑y).asIdea …
        nmr : Not (Membership.mem (↑((fun x => ⟨↑x, ⋯⟩) y)).asIdeal sr)
        wr : Eq (HSMul.hSMul sr (↑r ⟨↑y, ⋯⟩)) ((algebraMap R (AlgebraicGeometry.Struct …
        ⊢ Eq (HSMul.hSMul ((algebraMap R (AlgebraicGeometry.StructureSheaf.Localizatio …
      -/
      rfl
      /-
        🎉 no goals
      -/


/--
For any `R`-module `M`, `TildeInType R M` is the sheaf of set on `Spec R` whose sections on `U` are
the dependent functions that are locally fractions. This is often denoted by `M^~`.

See also `Tilde.isLocallyFraction`.
-/
def tildeInType : Sheaf (Type u) (PrimeSpectrum.Top R) :=
  subsheafToTypes (Tilde.isLocallyFraction M)


instance (U : (Opens (PrimeSpectrum.Top R))ᵒᵖ) :
    AddCommGroup (M.tildeInType.1.obj U) :=
  inferInstanceAs <| AddCommGroup (Tilde.sectionsSubmodule M U)


noncomputable instance (U : (Opens (PrimeSpectrum.Top R))ᵒᵖ) :
    Module ((Spec (.of R)).ringCatSheaf.1.obj U) (M.tildeInType.1.obj U) :=
  inferInstanceAs <| Module _ (Tilde.sectionsSubmodule M U)


/--
`M^~` as a sheaf of `𝒪_{Spec R}`-modules
-/
noncomputable def tilde : (Spec (CommRingCat.of R)).Modules where
  val :=
    { obj := fun U ↦ ModuleCat.of _ (M.tildeInType.val.obj U)
      map := fun {U V} i ↦ ofHom
        -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(Y := ...)`
        -- This suggests `restrictScalars` needs to be redesigned.
        (Y := (restrictScalars ((Spec (CommRingCat.of R)).ringCatSheaf.val.map i).hom).obj
          (of ((Spec (CommRingCat.of R)).ringCatSheaf.val.obj V) (M.tildeInType.val.obj V)))
        { toFun := M.tildeInType.val.map i
                          /-
                            R : Type u
                            inst✝ : CommRing R
                            M : ModuleCat R
                            U V : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec (CommRingCat. …
                            i : Quiver.Hom U V
                            ⊢ ∀ (m : ↑((AlgebraicGeometry.Spec (CommRingCat.of R)).ringCatSheaf.val.obj U) …
                          -/
                         /-
                           R : Type u
                           inst✝ : CommRing R
                           M : ModuleCat R
                           U V : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec (CommRingCat. …
                           i : Quiver.Hom U V
                           ⊢ ∀ (x y : M.tildeInType.val.obj U), Eq (M.tildeInType.val.map i (HAdd.hAdd x  …
                         -/
          map_smul' := by intros; rfl
                                 /-
                                   🎉 no goals
                                 -/
                                  /-
                                    🎉 no goals
                                  -/
          map_add' := by intros; rfl } }
  isSheaf := (TopCat.Presheaf.isSheaf_iff_isSheaf_comp (forget AddCommGrp) _ ).2
    M.tildeInType.2


/--
This is `M^~` as a sheaf of `R`-modules.
-/
noncomputable def tildeInModuleCat :
    TopCat.Presheaf (ModuleCat R) (PrimeSpectrum.Top R) :=
  (PresheafOfModules.forgetToPresheafModuleCat (op ⊤) <|
    Limits.initialOpOfTerminal Limits.isTerminalTop).obj (tilde M).1 ⋙
  ModuleCat.restrictScalars (StructureSheaf.globalSectionsIso R).hom.hom


@[simp]
theorem res_apply (U V : Opens (PrimeSpectrum.Top R)) (i : V ⟶ U)
    (s : (tildeInModuleCat M).obj (op U)) (x : V) :
    ((tildeInModuleCat M).map i.op s).1 x = (s.1 (i x) : _) :=
  rfl


lemma smul_section_apply (r : R) (U : Opens (PrimeSpectrum.Top R))
    (s : (tildeInModuleCat M).1.obj (op U)) (x : U) :
    (r • s).1 x = r • (s.1 x) := rfl


lemma smul_stalk_no_nonzero_divisor {x : PrimeSpectrum R}
    (r : x.asIdeal.primeCompl) (st : (tildeInModuleCat M).stalk x) (hst : r.1 • st = 0) :
    st = 0 := by
  refine Limits.Concrete.colimit_no_zero_smul_divisor
    _ _ _ ⟨op ⟨PrimeSpectrum.basicOpen r.1, r.2⟩, fun U i s hs ↦ Subtype.eq <| funext fun pt ↦ ?_⟩
    _ hst
  apply LocalizedModule.eq_zero_of_smul_eq_zero _ (i.unop pt).2 _
    (congr_fun (Subtype.ext_iff.1 hs) pt)


/--
If `U` is an open subset of `Spec R`, this is the morphism of `R`-modules from `M` to
`M^~(U)`.
-/
def toOpen (U : Opens (PrimeSpectrum.Top R)) :
    ModuleCat.of R M ⟶ (tildeInModuleCat M).1.obj (op U) :=
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(Y := ...)`
  -- This suggests `restrictScalars` needs to be redesigned.
  ModuleCat.ofHom (Y := (tildeInModuleCat M).1.obj (op U))
  { toFun := fun f =>
    ⟨fun x ↦ LocalizedModule.mkLinearMap _ _ f, fun x ↦
                                                                          /-
                                                                            R : Type u
                                                                            inst✝ : CommRing R
                                                                            M : ModuleCat R
                                                                            U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                                            f : ↑M
                                                                            x : Subtype fun x => Membership.mem (Opposite.unop { unop := U }) x
                                                                            y : Subtype fun x => Membership.mem U x
                                                                            ⊢ Eq (HSMul.hSMul 1 ((fun x => (fun x => (LocalizedModule.mkLinearMap (↑x).asI …
                                                                          -/
      ⟨U, x.2, 𝟙 _, f, 1, fun y ↦ ⟨(Ideal.ne_top_iff_one _).1 y.1.2.1, by simp⟩⟩⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    map_add' := fun f g => Subtype.eq <| funext fun x ↦ LinearMap.map_add _ _ _
    map_smul' := fun r m => by
      simp only [isLocallyFraction_pred, LocalizedModule.mkLinearMap_apply, LinearMapClass.map_smul,
        RingHom.id_apply]
      /-
        R : Type u
        inst✝ : CommRing R
        M : ModuleCat R
        U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
        r : R
        m : ↑M
        ⊢ Eq ⟨fun x => HSMul.hSMul r (LocalizedModule.mk m 1), ⋯⟩ (HSMul.hSMul r ⟨fun  …
      -/
      rfl }
      /-
        🎉 no goals
      -/


@[simp]
theorem toOpen_res (U V : Opens (PrimeSpectrum.Top R)) (i : V ⟶ U) :
    toOpen M U ≫ (tildeInModuleCat M).map i.op = toOpen M V :=
  rfl


/--
If `x` is a point of `Spec R`, this is the morphism of `R`-modules from `M` to the stalk of
`M^~` at `x`.
-/
noncomputable def toStalk (x : PrimeSpectrum.Top R) :
    ModuleCat.of R M ⟶ TopCat.Presheaf.stalk (tildeInModuleCat M) x :=
                                                                  /-
                                                                    R : Type u
                                                                    inst✝ : CommRing R
                                                                    M : ModuleCat R
                                                                    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                                                                    ⊢ Membership.mem Top.top x
                                                                  -/
  (toOpen M ⊤ ≫ TopCat.Presheaf.germ (tildeInModuleCat M) ⊤ x (by trivial))
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


open LocalizedModule TopCat.Presheaf in
lemma isUnit_toStalk (x : PrimeSpectrum.Top R) (r : x.asIdeal.primeCompl) :
    IsUnit ((algebraMap R (Module.End R ((tildeInModuleCat M).stalk x))) r) := by
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    r : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    ⊢ IsUnit ((algebraMap R (Module.End R ↑(M.tildeInModuleCat.stalk x))) ↑r)
  -/
  rw [Module.End_isUnit_iff]
  refine ⟨LinearMap.ker_eq_bot.1 <| eq_bot_iff.2 fun st (h : r.1 • st = 0) ↦
    smul_stalk_no_nonzero_divisor M r st h, fun st ↦ ?_⟩
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    r : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    st : ↑(M.tildeInModuleCat.stalk x)
    ⊢ Exists fun a => Eq (((algebraMap R (Module.End R ↑(M.tildeInModuleCat.stalk  …
  -/
  obtain ⟨U, mem, s, rfl⟩ := germ_exist (F := M.tildeInModuleCat) x st
  /-
    case intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    r : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem : Membership.mem U x
    s : (CategoryTheory.forget (ModuleCat R)).obj (M.tildeInModuleCat.obj { unop : …
    ⊢ Exists fun a => Eq (((algebraMap R (Module.End R ↑(M.tildeInModuleCat.stalk  …
  -/
  let O := U ⊓ (PrimeSpectrum.basicOpen r)
  refine ⟨germ M.tildeInModuleCat O x ⟨mem, r.2⟩
    ⟨fun q ↦ (Localization.mk 1 ⟨r, q.2.2⟩ : Localization.AtPrime q.1.asIdeal) • s.1
      ⟨q.1, q.2.1⟩, fun q ↦ ?_⟩, by
        simpa only [Module.algebraMap_end_apply, ← map_smul] using
          germ_ext (C := ModuleCat R) (W := O) (hxW := ⟨mem, r.2⟩) (iWU := 𝟙 _)
            (iWV := homOfLE inf_le_left) _ <|
          Subtype.eq <| funext fun y ↦ smul_eq_iff_of_mem (S := y.1.1.primeCompl) r _ _ _ |>.2 rfl⟩
  /-
    case intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    r : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem : Membership.mem U x
    s : (CategoryTheory.forget (ModuleCat R)).obj (M.tildeInModuleCat.obj { unop : …
    O : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R) := Min.min …
    q : Subtype fun x => Membership.mem (Opposite.unop { unop := O }) x
    ⊢ Exists fun V => Exists fun x_1 => Exists fun i => (ModuleCat.Tilde.isFractio …
  -/
  obtain ⟨V, mem_V, iV, num, den, hV⟩ := s.2 ⟨q.1, q.2.1⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    r : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem : Membership.mem U x
    s : (CategoryTheory.forget (ModuleCat R)).obj (M.tildeInModuleCat.obj { unop : …
    O : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R) := Min.min …
    q : Subtype fun x => Membership.mem (Opposite.unop { unop := O }) x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem_V : Membership.mem V ↑⟨↑q, ⋯⟩
    iV : Quiver.Hom V (Opposite.unop { unop := U })
    num : ↑M
    den : R
    hV : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
    ⊢ Exists fun V => Exists fun x_1 => Exists fun i => (ModuleCat.Tilde.isFractio …
  -/
  refine ⟨V ⊓ O, ⟨mem_V, q.2⟩, homOfLE inf_le_right, num, r * den, fun y ↦ ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    r : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem : Membership.mem U x
    s : (CategoryTheory.forget (ModuleCat R)).obj (M.tildeInModuleCat.obj { unop : …
    O : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R) := Min.min …
    q : Subtype fun x => Membership.mem (Opposite.unop { unop := O }) x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem_V : Membership.mem V ↑⟨↑q, ⋯⟩
    iV : Quiver.Hom V (Opposite.unop { unop := U })
    num : ↑M
    den : R
    hV : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
    y : Subtype fun x => Membership.mem (Min.min V O) x
    ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul (↑r) den))) (Eq (HSMul.hSMu …
  -/
  obtain ⟨h1, h2⟩ := hV ⟨y, y.2.1⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    r : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem : Membership.mem U x
    s : (CategoryTheory.forget (ModuleCat R)).obj (M.tildeInModuleCat.obj { unop : …
    O : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R) := Min.min …
    q : Subtype fun x => Membership.mem (Opposite.unop { unop := O }) x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem_V : Membership.mem V ↑⟨↑q, ⋯⟩
    iV : Quiver.Hom V (Opposite.unop { unop := U })
    num : ↑M
    den : R
    hV : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
    y : Subtype fun x => Membership.mem (Min.min V O) x
    h1 : Not (Membership.mem (↑⟨↑y, ⋯⟩).asIdeal den)
    h2 : Eq (HSMul.hSMul den ((fun x => ↑s ((fun x => ⟨↑x, ⋯⟩) x)) ⟨↑y, ⋯⟩)) ((Loc …
    ⊢ And (Not (Membership.mem (↑y).asIdeal (HMul.hMul (↑r) den))) (Eq (HSMul.hSMu …
  -/
  refine ⟨y.1.asIdeal.primeCompl.mul_mem y.2.2.2 h1, ?_⟩
  simp only [Opens.coe_inf, isLocallyFraction_pred, mkLinearMap_apply,
    smul_eq_iff_of_mem (S := y.1.1.primeCompl) (hr := h1), mk_smul_mk, one_smul, mul_one] at h2 ⊢
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    r : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem : Membership.mem U x
    s : (CategoryTheory.forget (ModuleCat R)).obj (M.tildeInModuleCat.obj { unop : …
    O : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R) := Min.min …
    q : Subtype fun x => Membership.mem (Opposite.unop { unop := O }) x
    V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    mem_V : Membership.mem V ↑⟨↑q, ⋯⟩
    iV : Quiver.Hom V (Opposite.unop { unop := U })
    num : ↑M
    den : R
    hV : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑x …
    y : Subtype fun x => Membership.mem (Min.min V O) x
    h1 : Not (Membership.mem (↑⟨↑y, ⋯⟩).asIdeal den)
    h2 : Eq (↑s ⟨↑y, ⋯⟩) (LocalizedModule.mk num ⟨den, h1⟩)
    ⊢ Eq (HSMul.hSMul (HMul.hMul (↑r) den) (HSMul.hSMul (Localization.mk 1 ⟨↑r, ⋯⟩ …
  -/
  simpa only [h2, mk_smul_mk, one_smul, smul'_mk, mk_eq] using ⟨1, by simp only [one_smul]; rfl⟩
  /-
    🎉 no goals
  -/


/--
The morphism of `R`-modules from the localization of `M` at the prime ideal corresponding to `x`
to the stalk of `M^~` at `x`.
-/
noncomputable def localizationToStalk (x : PrimeSpectrum.Top R) :
    ModuleCat.of R (LocalizedModule x.asIdeal.primeCompl M) ⟶
    (TopCat.Presheaf.stalk (tildeInModuleCat M) x) :=
  ModuleCat.ofHom <| LocalizedModule.lift _ (toStalk M x).hom <| isUnit_toStalk M x



/-- The ring homomorphism that takes a section of the structure sheaf of `R` on the open set `U`,
implemented as a subtype of dependent functions to localizations at prime ideals, and evaluates
the section on the point corresponding to a given prime ideal. -/
def openToLocalization (U : Opens (PrimeSpectrum R)) (x : PrimeSpectrum R) (hx : x ∈ U) :
    (tildeInModuleCat M).obj (op U) ⟶
    ModuleCat.of R (LocalizedModule x.asIdeal.primeCompl M) :=
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(X := ...)` and `(Y := ...)`
  -- This suggests `restrictScalars` needs to be redesigned.
  ModuleCat.ofHom
    (X := (tildeInModuleCat M).obj (op U))
    (Y := ModuleCat.of R (LocalizedModule x.asIdeal.primeCompl M))
  { toFun := fun s => (s.1 ⟨x, hx⟩ : _)
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


/--
The morphism of `R`-modules from the stalk of `M^~` at `x` to the localization of `M` at the
prime ideal of `R` corresponding to `x`.
-/
noncomputable def stalkToFiberLinearMap (x : PrimeSpectrum.Top R) :
    (tildeInModuleCat M).stalk  x ⟶
    ModuleCat.of R (LocalizedModule x.asIdeal.primeCompl M) :=
  Limits.colimit.desc ((OpenNhds.inclusion x).op ⋙ (tildeInModuleCat M))
    { pt := _
      ι :=
      { app := fun U => openToLocalization M ((OpenNhds.inclusion _).obj U.unop) x U.unop.2 } }


@[simp]
theorem germ_comp_stalkToFiberLinearMap (U : Opens (PrimeSpectrum.Top R)) (x) (hx : x ∈ U) :
    (tildeInModuleCat M).germ U x hx ≫ stalkToFiberLinearMap M x =
    openToLocalization M U x hx :=
  Limits.colimit.ι_desc _ _


@[simp]
theorem stalkToFiberLinearMap_germ (U : Opens (PrimeSpectrum.Top R)) (x : PrimeSpectrum.Top R)
    (hx : x ∈ U) (s : (tildeInModuleCat M).1.obj (op U)) :
    (stalkToFiberLinearMap M x).hom
      (TopCat.Presheaf.germ (tildeInModuleCat M) U x hx s) = (s.1 ⟨x, hx⟩ : _) :=
  DFunLike.ext_iff.1 (ModuleCat.hom_ext_iff.mp (germ_comp_stalkToFiberLinearMap M U x hx)) s


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem toOpen_germ (U : Opens (PrimeSpectrum.Top R)) (x) (hx : x ∈ U) :
    toOpen M U ≫ M.tildeInModuleCat.germ U x hx = toStalk M x := by
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    hx : Membership.mem U x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.Tilde.toOpen M U) (M.tilde …
  -/
  rw [← toOpen_res M ⊤ U (homOfLE le_top : U ⟶ ⊤), Category.assoc, Presheaf.germ_res]; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[reassoc (attr := simp)]
theorem toStalk_comp_stalkToFiberLinearMap (x : PrimeSpectrum.Top R) :
    toStalk M x ≫ stalkToFiberLinearMap M x =
    ofHom (LocalizedModule.mkLinearMap x.asIdeal.primeCompl M) := by
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.Tilde.toStalk M x) (Module …
  -/
  rw [toStalk, Category.assoc, germ_comp_stalkToFiberLinearMap]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem stalkToFiberLinearMap_toStalk (x : PrimeSpectrum.Top R) (m : M) :
    (stalkToFiberLinearMap M x).hom (toStalk M x m) =
    LocalizedModule.mk m 1 :=
  LinearMap.ext_iff.1 (ModuleCat.hom_ext_iff.mp (toStalk_comp_stalkToFiberLinearMap M x)) _


/--
If `U` is an open subset of `Spec R`, `m` is an element of `M` and `r` is an element of `R`
that is invertible on `U` (i.e. does not belong to any prime ideal corresponding to a point
in `U`), this is `m / r` seen as a section of `M^~` over `U`.
-/
def const (m : M) (r : R) (U : Opens (PrimeSpectrum.Top R))
    (hu : ∀ x ∈ U, r ∈ (x : PrimeSpectrum.Top R).asIdeal.primeCompl) :
    (tildeInModuleCat M).obj (op U) :=
  ⟨fun x => LocalizedModule.mk m ⟨r, hu x x.2⟩, fun x =>
    ⟨U, x.2, 𝟙 _, m, r, fun y => ⟨hu _ y.2, by
      simpa only [LocalizedModule.mkLinearMap_apply, LocalizedModule.smul'_mk,
        LocalizedModule.mk_eq] using ⟨1, by simp⟩⟩⟩⟩


@[simp]
theorem const_apply (m : M) (r : R) (U : Opens (PrimeSpectrum.Top R))
    (hu : ∀ x ∈ U, r ∈ (x : PrimeSpectrum.Top R).asIdeal.primeCompl) (x : U) :
    (const M m r U hu).1 x = LocalizedModule.mk m ⟨r, hu x x.2⟩ :=
  rfl


theorem exists_const (U) (s : (tildeInModuleCat M).obj (op U)) (x : PrimeSpectrum.Top R)
    (hx : x ∈ U) :
    ∃ (V : Opens (PrimeSpectrum.Top R)) (_ : x ∈ V) (i : V ⟶ U) (f : M) (g : R) (hg : _),
      const M f g V hg = (tildeInModuleCat M).map i.op s :=
  let ⟨V, hxV, iVU, f, g, hfg⟩ := s.2 ⟨x, hx⟩
  ⟨V, hxV, iVU, f, g, fun y hyV => (hfg ⟨y, hyV⟩).1, Subtype.eq <| funext fun y => by
    /-
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      s : ↑(M.tildeInModuleCat.obj { unop := U })
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hx : Membership.mem U x
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V ↑⟨x, hx⟩
      iVU : Quiver.Hom V (Opposite.unop { unop := U })
      f : ↑M
      g : R
      hfg : ∀ (x : Subtype fun x => Membership.mem V x), And (Not (Membership.mem (↑ …
      y : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
      ⊢ Eq (↑(ModuleCat.Tilde.const M f g V ⋯) y) (↑((M.tildeInModuleCat.map iVU.op) …
    -/
    obtain ⟨h1, (h2 : g • s.1 ⟨y, _⟩ = LocalizedModule.mk f 1)⟩ := hfg y
    exact show LocalizedModule.mk f ⟨g, by exact h1⟩ = s.1 (iVU y) by
      set x := s.1 (iVU y); change g • x = _ at h2; clear_value x
      induction x using LocalizedModule.induction_on with
      | h a b =>
        rw [LocalizedModule.smul'_mk, LocalizedModule.mk_eq] at h2
        obtain ⟨c, hc⟩ := h2
        exact LocalizedModule.mk_eq.mpr ⟨c, by simpa using hc.symm⟩⟩


@[simp]
theorem res_const (f : M) (g : R) (U hu V hv i) :
    (tildeInModuleCat M).map i (const M f g U hu) = const M f g V hv :=
  rfl


@[simp]
theorem localizationToStalk_mk (x : PrimeSpectrum.Top R) (f : M) (s : x.asIdeal.primeCompl) :
    (localizationToStalk M x).hom (LocalizedModule.mk f s) =
      (tildeInModuleCat M).germ (PrimeSpectrum.basicOpen (s : R)) x s.2
        (const M f s (PrimeSpectrum.basicOpen s) fun _ => id) :=
  (Module.End_isUnit_iff _ |>.1 (isUnit_toStalk M x s)).injective <| by
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    ⊢ Eq (((algebraMap R (Module.End R ↑(M.tildeInModuleCat.stalk x))) ↑s) ((Modul …
  -/
  erw [← LinearMap.mul_apply]
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    ⊢ Eq ((HMul.hMul ((algebraMap R (Module.End R ↑(M.tildeInModuleCat.stalk x)))  …
  -/
  simp only [IsUnit.mul_val_inv, LinearMap.one_apply, Module.algebraMap_end_apply]
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    ⊢ Eq ((ModuleCat.Tilde.toStalk M x).hom f) (HSMul.hSMul (↑s) ((M.tildeInModule …
  -/
  show (M.tildeInModuleCat.germ ⊤ x ⟨⟩) ((toOpen M ⊤) f) = _
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    ⊢ Eq ((M.tildeInModuleCat.germ Top.top x True.intro).hom ((ModuleCat.Tilde.toO …
  -/
  rw [← map_smul]
  fapply TopCat.Presheaf.germ_ext (W := PrimeSpectrum.basicOpen s.1) (hxW := s.2)
    (F := M.tildeInModuleCat)
    /-
      case iWU
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      f : ↑M
      s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
      ⊢ Quiver.Hom (PrimeSpectrum.basicOpen ↑s) Top.top
    -/
  · exact homOfLE le_top
    /-
      🎉 no goals
    -/
    /-
      case iWV
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      f : ↑M
      s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
      ⊢ Quiver.Hom (PrimeSpectrum.basicOpen ↑s) (PrimeSpectrum.basicOpen ↑s)
    -/
  · exact 𝟙 _
    /-
      🎉 no goals
    -/
  /-
    case ih
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    ⊢ Eq ((M.tildeInModuleCat.map (CategoryTheory.homOfLE ⋯).op) ((ModuleCat.Tilde …
  -/
  refine Subtype.eq <| funext fun y => show LocalizedModule.mk f 1 = _ from ?_
  #adaptation_note /-- https://github.com/leanprover/lean4/pull/6024
    added this refine hack to be able to add type hint in `change` -/
  /-
    case ih
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    y : Subtype fun x_1 => Membership.mem (Opposite.unop { unop := PrimeSpectrum.b …
    ⊢ Eq (LocalizedModule.mk f 1) (↑((M.tildeInModuleCat.map (CategoryTheory.Categ …
  -/
  refine (?_ : @Eq ?ty _ _)
  /-
    case ih
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    y : Subtype fun x_1 => Membership.mem (Opposite.unop { unop := PrimeSpectrum.b …
    ⊢ Eq (LocalizedModule.mk f 1) (↑((M.tildeInModuleCat.map (CategoryTheory.Categ …
  -/
  change LocalizedModule.mk f 1 = (s.1 • LocalizedModule.mk f _ : ?ty)
  /-
    case ih
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    y : Subtype fun x_1 => Membership.mem (Opposite.unop { unop := PrimeSpectrum.b …
    ⊢ Eq (LocalizedModule.mk f 1) (HSMul.hSMul (↑s) (LocalizedModule.mk f ⟨↑s, ⋯⟩))
  -/
  rw [LocalizedModule.smul'_mk, LocalizedModule.mk_eq]
  /-
    case ih
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    f : ↑M
    s : Subtype fun x_1 => Membership.mem x.asIdeal.primeCompl x_1
    y : Subtype fun x_1 => Membership.mem (Opposite.unop { unop := PrimeSpectrum.b …
    ⊢ Exists fun u => Eq (HSMul.hSMul u (HSMul.hSMul ⟨↑s, ⋯⟩ f)) (HSMul.hSMul u (H …
  -/
  exact ⟨1, by simp⟩
  /-
    🎉 no goals
  -/


/--
The isomorphism of `R`-modules from the stalk of `M^~` at `x` to the localization of `M` at the
prime ideal corresponding to `x`.
-/
@[simps]
noncomputable def stalkIso (x : PrimeSpectrum.Top R) :
    TopCat.Presheaf.stalk (tildeInModuleCat M) x ≅
    ModuleCat.of R (LocalizedModule x.asIdeal.primeCompl M) where
  hom := stalkToFiberLinearMap M x
  inv := localizationToStalk M x
  hom_inv_id := TopCat.Presheaf.stalk_hom_ext _ fun U hxU ↦ ModuleCat.hom_ext <|
      LinearMap.ext fun s ↦ by
    show localizationToStalk M x (stalkToFiberLinearMap M x (M.tildeInModuleCat.germ U x hxU s)) =
      M.tildeInModuleCat.germ U x hxU s
    /-
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxU : Membership.mem U x
      s : ↑(M.tildeInModuleCat.obj { unop := U })
      ⊢ Eq ((ModuleCat.Tilde.localizationToStalk M x).hom ((ModuleCat.Tilde.stalkToF …
    -/
    rw [stalkToFiberLinearMap_germ]
    obtain ⟨V, hxV, iVU, f, g, (hg : V ≤ PrimeSpectrum.basicOpen _), hs⟩ :=
      exists_const _ _ s x hxU
    /-
      case intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxU : Membership.mem U x
      s : ↑(M.tildeInModuleCat.obj { unop := U })
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V x
      iVU : Quiver.Hom V U
      f : ↑M
      g : R
      hg : LE.le V (PrimeSpectrum.basicOpen g)
      hs : Eq (ModuleCat.Tilde.const M f g V hg) ((M.tildeInModuleCat.map iVU.op).ho …
      ⊢ Eq ((ModuleCat.Tilde.localizationToStalk M x).hom (↑s ⟨x, hxU⟩)) ((M.tildeIn …
    -/
    rw [← res_apply M U V iVU s ⟨x, hxV⟩, ← hs, const_apply, localizationToStalk_mk]
    /-
      case intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommRing R
      M : ModuleCat R
      x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      U : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxU : Membership.mem U x
      s : ↑(M.tildeInModuleCat.obj { unop := U })
      V : TopologicalSpace.Opens ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
      hxV : Membership.mem V x
      iVU : Quiver.Hom V U
      f : ↑M
      g : R
      hg : LE.le V (PrimeSpectrum.basicOpen g)
      hs : Eq (ModuleCat.Tilde.const M f g V hg) ((M.tildeInModuleCat.map iVU.op).ho …
      ⊢ Eq ((M.tildeInModuleCat.germ (PrimeSpectrum.basicOpen ↑⟨g, ⋯⟩) x ⋯).hom (Mod …
    -/
    exact (tildeInModuleCat M).germ_ext V hxV (homOfLE hg) iVU <| hs ▸ rfl
    /-
      🎉 no goals
    -/
                   /-
                     R : Type u
                     inst✝ : CommRing R
                     M : ModuleCat R
                     x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.Tilde.localizationToStalk  …
                   -/
  inv_hom_id := by ext x; exact x.induction_on (fun _ _ => by
    simp only [hom_comp, LinearMap.coe_comp, Function.comp_apply, hom_id, LinearMap.id_coe, id_eq]
    rw [localizationToStalk_mk, stalkToFiberLinearMap_germ]
    simp)


instance (x : PrimeSpectrum.Top R) :
    IsLocalizedModule x.asIdeal.primeCompl (toStalk M x).hom := by
  convert IsLocalizedModule.of_linearEquiv
    (hf := localizedModuleIsLocalizedModule (M := M) x.asIdeal.primeCompl)
    (e := (stalkIso M x).symm.toLinearEquiv)
  /-
    case h.e'_10.h.e'_5.h.h.h
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    ⊢ Eq (ModuleCat.Tilde.toStalk M x) { hom := (↑(ModuleCat.Tilde.stalkIso M x).s …
  -/
  ext
  simp only [of_coe,
    show (stalkIso M x).symm.toLinearEquiv.toLinearMap = (stalkIso M x).inv.hom by rfl]
  /-
    case h.e'_10.h.e'_5.h.h.h.hom.h
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    x : ↑(AlgebraicGeometry.PrimeSpectrum.Top R)
    x✝ : ↑(ModuleCat.of R ↑M)
    ⊢ Eq ((ModuleCat.Tilde.toStalk M x).hom x✝) (((ModuleCat.Tilde.stalkIso M x).i …
  -/
  erw [LocalizedModule.lift_comp]
  /-
    🎉 no goals
  -/


