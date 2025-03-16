/-- An algebra over a commutative semiring is of `FiniteType` if it is finitely generated
over the base ring as algebra. -/
class Algebra.FiniteType [CommSemiring R] [Semiring A] [Algebra R A] : Prop where
  out : (⊤ : Subalgebra R A).FG


instance (priority := 100) finiteType {R : Type*} (A : Type*) [CommSemiring R] [Semiring A]
    [Algebra R A] [hRA : Module.Finite R A] : Algebra.FiniteType R A :=
  ⟨Subalgebra.fg_of_submodule_fg hRA.1⟩


theorem self : FiniteType R R :=
  ⟨⟨{1}, Subsingleton.elim _ _⟩⟩


protected theorem polynomial : FiniteType R R[X] :=
  ⟨⟨{Polynomial.X}, by
      /-
        R : Type uR
        inst✝ : CommSemiring R
        ⊢ Eq (Algebra.adjoin R ↑(Singleton.singleton Polynomial.X)) Top.top
      -/
      rw [Finset.coe_singleton]
      /-
        R : Type uR
        inst✝ : CommSemiring R
        ⊢ Eq (Algebra.adjoin R (Singleton.singleton Polynomial.X)) Top.top
      -/
      exact Polynomial.adjoin_X⟩⟩
      /-
        🎉 no goals
      -/


protected theorem freeAlgebra (ι : Type*) [Finite ι] : FiniteType R (FreeAlgebra R ι) := by
  /-
    R : Type uR
    inst✝¹ : CommSemiring R
    ι : Type u_1
    inst✝ : Finite ι
    ⊢ Algebra.FiniteType R (FreeAlgebra R ι)
  -/
  cases nonempty_fintype ι
  exact
    ⟨⟨Finset.univ.image (FreeAlgebra.ι R), by
        rw [Finset.coe_image, Finset.coe_univ, Set.image_univ]
        exact FreeAlgebra.adjoin_range_ι R ι⟩⟩


protected theorem mvPolynomial (ι : Type*) [Finite ι] : FiniteType R (MvPolynomial ι R) := by
  /-
    R : Type uR
    inst✝¹ : CommSemiring R
    ι : Type u_1
    inst✝ : Finite ι
    ⊢ Algebra.FiniteType R (MvPolynomial ι R)
  -/
  cases nonempty_fintype ι
  exact
    ⟨⟨Finset.univ.image MvPolynomial.X, by
        rw [Finset.coe_image, Finset.coe_univ, Set.image_univ]
        exact MvPolynomial.adjoin_range_X⟩⟩


theorem of_restrictScalars_finiteType [Algebra S A] [IsScalarTower R S A] [hA : FiniteType R A] :
    FiniteType S A := by
  /-
    R : Type uR
    S : Type uS
    A : Type uA
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    inst✝ : IsScalarTower R S A
    hA : Algebra.FiniteType R A
    ⊢ Algebra.FiniteType S A
  -/
  obtain ⟨s, hS⟩ := hA.out
  /-
    case intro
    R : Type uR
    S : Type uS
    A : Type uA
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    inst✝ : IsScalarTower R S A
    hA : Algebra.FiniteType R A
    s : Finset A
    hS : Eq (Algebra.adjoin R ↑s) Top.top
    ⊢ Algebra.FiniteType S A
  -/
  refine ⟨⟨s, eq_top_iff.2 fun b => ?_⟩⟩
  have le : adjoin R (s : Set A) ≤ Subalgebra.restrictScalars R (adjoin S s) := by
    apply (Algebra.adjoin_le _ : adjoin R (s : Set A) ≤ Subalgebra.restrictScalars R (adjoin S ↑s))
    simp only [Subalgebra.coe_restrictScalars]
    exact Algebra.subset_adjoin
  /-
    case intro
    R : Type uR
    S : Type uS
    A : Type uA
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    inst✝ : IsScalarTower R S A
    hA : Algebra.FiniteType R A
    s : Finset A
    hS : Eq (Algebra.adjoin R ↑s) Top.top
    b : A
    le : LE.le (Algebra.adjoin R ↑s) (Subalgebra.restrictScalars R (Algebra.adjoin …
    ⊢ Membership.mem (Algebra.adjoin S ↑s) b
  -/
  exact le (eq_top_iff.1 hS b)
  /-
    🎉 no goals
  -/


theorem of_surjective (hRA : FiniteType R A) (f : A →ₐ[R] B) (hf : Surjective f) : FiniteType R B :=
  ⟨by
    /-
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      hRA : Algebra.FiniteType R A
      f : AlgHom R A B
      hf : Function.Surjective ⇑f
      ⊢ Top.top.FG
    -/
    convert hRA.1.map f
    /-
      case h.e'_6
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      hRA : Algebra.FiniteType R A
      f : AlgHom R A B
      hf : Function.Surjective ⇑f
      ⊢ Eq Top.top (Subalgebra.map f Top.top)
    -/
    simpa only [map_top f, @eq_comm _ ⊤, eq_top_iff, AlgHom.mem_range] using hf⟩
    /-
      🎉 no goals
    -/


theorem equiv (hRA : FiniteType R A) (e : A ≃ₐ[R] B) : FiniteType R B :=
  hRA.of_surjective e e.surjective


theorem trans [Algebra S A] [IsScalarTower R S A] (hRS : FiniteType R S) (hSA : FiniteType S A) :
    FiniteType R A :=
  ⟨fg_trans' hRS.1 hSA.1⟩


instance quotient (R : Type*) {S : Type*} [CommSemiring R] [CommRing S] [Algebra R S] (I : Ideal S)
    [h : Algebra.FiniteType R S] : Algebra.FiniteType R (S ⧸ I) :=
  Algebra.FiniteType.trans h inferInstance


/-- An algebra is finitely generated if and only if it is a quotient
of a free algebra whose variables are indexed by a finset. -/
theorem iff_quotient_freeAlgebra :
    FiniteType R A ↔
      ∃ (s : Finset A) (f : FreeAlgebra R s →ₐ[R] A), Surjective f := by
  /-
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.FiniteType R A) (Exists fun s => Exists fun f => Function.Surje …
  -/
  constructor
    /-
      case mp
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ⊢ Algebra.FiniteType R A → Exists fun s => Exists fun f => Function.Surjective …
    -/
  · rintro ⟨s, hs⟩
    /-
      case mp.mk.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      ⊢ Exists fun s => Exists fun f => Function.Surjective ⇑f
    -/
    refine ⟨s, FreeAlgebra.lift _ (↑), ?_⟩
    /-
      case mp.mk.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      ⊢ Function.Surjective ⇑((FreeAlgebra.lift R) Subtype.val)
    -/
    intro x
    /-
      case mp.mk.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : A
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) Subtype.val) a) x
    -/
    have hrw : (↑s : Set A) = fun x : A => x ∈ s.val := rfl
    /-
      case mp.mk.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : A
      hrw : Eq ↑s fun x => Membership.mem s.val x
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) Subtype.val) a) x
    -/
    rw [← Set.mem_range, ← AlgHom.coe_range]
    /-
      case mp.mk.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : A
      hrw : Eq ↑s fun x => Membership.mem s.val x
      ⊢ Membership.mem (↑((FreeAlgebra.lift R) Subtype.val).range) x
    -/
    erw [← adjoin_eq_range_freeAlgebra_lift]
    /-
      case mp.mk.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : A
      hrw : Eq ↑s fun x => Membership.mem s.val x
      ⊢ Membership.mem (↑(Algebra.adjoin R (Membership.mem s.val))) x
    -/
    simp_rw [← hrw, hs]
    /-
      case mp.mk.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : A
      hrw : Eq ↑s fun x => Membership.mem s.val x
      ⊢ Membership.mem (↑Top.top) x
    -/
    exact Set.mem_univ x
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ⊢ (Exists fun s => Exists fun f => Function.Surjective ⇑f) → Algebra.FiniteTyp …
    -/
  · rintro ⟨s, ⟨f, hsur⟩⟩
    /-
      case mpr.intro.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      f : AlgHom R (FreeAlgebra R (Subtype fun x => Membership.mem s x)) A
      hsur : Function.Surjective ⇑f
      ⊢ Algebra.FiniteType R A
    -/
    exact FiniteType.of_surjective (FiniteType.freeAlgebra R s) f hsur
    /-
      🎉 no goals
    -/


/-- A commutative algebra is finitely generated if and only if it is a quotient
of a polynomial ring whose variables are indexed by a finset. -/
theorem iff_quotient_mvPolynomial :
    FiniteType R S ↔
      ∃ (s : Finset S) (f : MvPolynomial { x // x ∈ s } R →ₐ[R] S), Surjective f := by
  /-
    R : Type uR
    S : Type uS
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    ⊢ Iff (Algebra.FiniteType R S) (Exists fun s => Exists fun f => Function.Surje …
  -/
  constructor
    /-
      case mp
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ⊢ Algebra.FiniteType R S → Exists fun s => Exists fun f => Function.Surjective …
    -/
  · rintro ⟨s, hs⟩
    /-
      case mp.mk.intro
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      s : Finset S
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      ⊢ Exists fun s => Exists fun f => Function.Surjective ⇑f
    -/
    use s, MvPolynomial.aeval (↑)
    /-
      case h
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      s : Finset S
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      ⊢ Function.Surjective ⇑(MvPolynomial.aeval Subtype.val)
    -/
    intro x
    /-
      case h
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      s : Finset S
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : S
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval Subtype.val) a) x
    -/
    have hrw : (↑s : Set S) = fun x : S => x ∈ s.val := rfl
    /-
      case h
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      s : Finset S
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : S
      hrw : Eq ↑s fun x => Membership.mem s.val x
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval Subtype.val) a) x
    -/
    rw [← Set.mem_range, ← AlgHom.coe_range, ← adjoin_eq_range]
    /-
      case h
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      s : Finset S
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : S
      hrw : Eq ↑s fun x => Membership.mem s.val x
      ⊢ Membership.mem (↑(Algebra.adjoin R (Membership.mem s.val))) x
    -/
    simp_rw [← hrw, hs]
    /-
      case h
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      s : Finset S
      hs : Eq (Algebra.adjoin R ↑s) Top.top
      x : S
      hrw : Eq ↑s fun x => Membership.mem s.val x
      ⊢ Membership.mem (↑Top.top) x
    -/
    exact Set.mem_univ x
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ⊢ (Exists fun s => Exists fun f => Function.Surjective ⇑f) → Algebra.FiniteTyp …
    -/
  · rintro ⟨s, ⟨f, hsur⟩⟩
    /-
      case mpr.intro.intro
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      s : Finset S
      f : AlgHom R (MvPolynomial (Subtype fun x => Membership.mem s x) R) S
      hsur : Function.Surjective ⇑f
      ⊢ Algebra.FiniteType R S
    -/
    exact FiniteType.of_surjective (FiniteType.mvPolynomial R { x // x ∈ s }) f hsur
    /-
      🎉 no goals
    -/


/-- An algebra is finitely generated if and only if it is a quotient
of a polynomial ring whose variables are indexed by a fintype. -/
theorem iff_quotient_freeAlgebra' : FiniteType R A ↔
    ∃ (ι : Type uA) (_ : Fintype ι) (f : FreeAlgebra R ι →ₐ[R] A), Surjective f := by
  /-
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.FiniteType R A) (Exists fun ι => Exists fun x => Exists fun f = …
  -/
  constructor
    /-
      case mp
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ⊢ Algebra.FiniteType R A → Exists fun ι => Exists fun x => Exists fun f => Fun …
    -/
  · rw [iff_quotient_freeAlgebra]
    /-
      case mp
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ⊢ (Exists fun s => Exists fun f => Function.Surjective ⇑f) → Exists fun ι => E …
    -/
    rintro ⟨s, ⟨f, hsur⟩⟩
    /-
      case mp.intro.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Finset A
      f : AlgHom R (FreeAlgebra R (Subtype fun x => Membership.mem s x)) A
      hsur : Function.Surjective ⇑f
      ⊢ Exists fun ι => Exists fun x => Exists fun f => Function.Surjective ⇑f
    -/
    use { x : A // x ∈ s }, inferInstance, f
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ⊢ (Exists fun ι => Exists fun x => Exists fun f => Function.Surjective ⇑f) → A …
    -/
  · rintro ⟨ι, ⟨hfintype, ⟨f, hsur⟩⟩⟩
    /-
      case mpr.intro.intro.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ι : Type uA
      hfintype : Fintype ι
      f : AlgHom R (FreeAlgebra R ι) A
      hsur : Function.Surjective ⇑f
      ⊢ Algebra.FiniteType R A
    -/
    letI : Fintype ι := hfintype
    /-
      case mpr.intro.intro.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ι : Type uA
      hfintype : Fintype ι
      f : AlgHom R (FreeAlgebra R ι) A
      hsur : Function.Surjective ⇑f
      this : Fintype ι := hfintype
      ⊢ Algebra.FiniteType R A
    -/
    exact FiniteType.of_surjective (FiniteType.freeAlgebra R ι) f hsur
    /-
      🎉 no goals
    -/


/-- A commutative algebra is finitely generated if and only if it is a quotient
of a polynomial ring whose variables are indexed by a fintype. -/
theorem iff_quotient_mvPolynomial' : FiniteType R S ↔
    ∃ (ι : Type uS) (_ : Fintype ι) (f : MvPolynomial ι R →ₐ[R] S), Surjective f := by
  /-
    R : Type uR
    S : Type uS
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    ⊢ Iff (Algebra.FiniteType R S) (Exists fun ι => Exists fun x => Exists fun f = …
  -/
  constructor
    /-
      case mp
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ⊢ Algebra.FiniteType R S → Exists fun ι => Exists fun x => Exists fun f => Fun …
    -/
  · rw [iff_quotient_mvPolynomial]
    /-
      case mp
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ⊢ (Exists fun s => Exists fun f => Function.Surjective ⇑f) → Exists fun ι => E …
    -/
    rintro ⟨s, ⟨f, hsur⟩⟩
    /-
      case mp.intro.intro
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      s : Finset S
      f : AlgHom R (MvPolynomial (Subtype fun x => Membership.mem s x) R) S
      hsur : Function.Surjective ⇑f
      ⊢ Exists fun ι => Exists fun x => Exists fun f => Function.Surjective ⇑f
    -/
    use { x : S // x ∈ s }, inferInstance, f
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ⊢ (Exists fun ι => Exists fun x => Exists fun f => Function.Surjective ⇑f) → A …
    -/
  · rintro ⟨ι, ⟨hfintype, ⟨f, hsur⟩⟩⟩
    /-
      case mpr.intro.intro.intro
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ι : Type uS
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) S
      hsur : Function.Surjective ⇑f
      ⊢ Algebra.FiniteType R S
    -/
    letI : Fintype ι := hfintype
    /-
      case mpr.intro.intro.intro
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ι : Type uS
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) S
      hsur : Function.Surjective ⇑f
      this : Fintype ι := hfintype
      ⊢ Algebra.FiniteType R S
    -/
    exact FiniteType.of_surjective (FiniteType.mvPolynomial R ι) f hsur
    /-
      🎉 no goals
    -/


/-- A commutative algebra is finitely generated if and only if it is a quotient of a polynomial ring
in `n` variables. -/
theorem iff_quotient_mvPolynomial'' :
    FiniteType R S ↔ ∃ (n : ℕ) (f : MvPolynomial (Fin n) R →ₐ[R] S), Surjective f := by
  /-
    R : Type uR
    S : Type uS
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    ⊢ Iff (Algebra.FiniteType R S) (Exists fun n => Exists fun f => Function.Surje …
  -/
  constructor
    /-
      case mp
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ⊢ Algebra.FiniteType R S → Exists fun n => Exists fun f => Function.Surjective …
    -/
  · rw [iff_quotient_mvPolynomial']
    /-
      case mp
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ⊢ (Exists fun ι => Exists fun x => Exists fun f => Function.Surjective ⇑f) → E …
    -/
    rintro ⟨ι, hfintype, ⟨f, hsur⟩⟩
    /-
      case mp.intro.intro.intro
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ι : Type uS
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) S
      hsur : Function.Surjective ⇑f
      ⊢ Exists fun n => Exists fun f => Function.Surjective ⇑f
    -/
    have equiv := MvPolynomial.renameEquiv R (Fintype.equivFin ι)
    /-
      case mp.intro.intro.intro
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ι : Type uS
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) S
      hsur : Function.Surjective ⇑f
      equiv : AlgEquiv R (MvPolynomial ι R) (MvPolynomial (Fin (Fintype.card ι)) R)
      ⊢ Exists fun n => Exists fun f => Function.Surjective ⇑f
    -/
    exact ⟨Fintype.card ι, AlgHom.comp f equiv.symm.toAlgHom, by simpa using hsur⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      ⊢ (Exists fun n => Exists fun f => Function.Surjective ⇑f) → Algebra.FiniteTyp …
    -/
  · rintro ⟨n, ⟨f, hsur⟩⟩
    /-
      case mpr.intro.intro
      R : Type uR
      S : Type uS
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) S
      hsur : Function.Surjective ⇑f
      ⊢ Algebra.FiniteType R S
    -/
    exact FiniteType.of_surjective (FiniteType.mvPolynomial R (Fin n)) f hsur
    /-
      🎉 no goals
    -/


instance prod [hA : FiniteType R A] [hB : FiniteType R B] : FiniteType R (A × B) :=
      /-
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        M : Type uM
        N : Type uN
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Algebra R S
        inst✝⁵ : Algebra R A
        inst✝⁴ : Algebra R B
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : AddCommMonoid N
        inst✝ : Module R N
        hA : Algebra.FiniteType R A
        hB : Algebra.FiniteType R B
        ⊢ Top.top.FG
      -/
  ⟨by rw [← Subalgebra.prod_top]; exact hA.1.prod hB.1⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem isNoetherianRing (R S : Type*) [CommRing R] [CommRing S] [Algebra R S]
    [h : Algebra.FiniteType R S] [IsNoetherianRing R] : IsNoetherianRing S := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    h : Algebra.FiniteType R S
    inst✝ : IsNoetherianRing R
    ⊢ IsNoetherianRing S
  -/
  obtain ⟨s, hs⟩ := h.1
  apply
    isNoetherianRing_of_surjective (MvPolynomial s R) S
      (MvPolynomial.aeval (↑) : MvPolynomial s R →ₐ[R] S).toRingHom
  erw [← Set.range_eq_univ, ← AlgHom.coe_range, ←
    Algebra.adjoin_range_eq_range_aeval, Subtype.range_coe_subtype, Finset.setOf_mem, hs]
  /-
    case intro.hf
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    h : Algebra.FiniteType R S
    inst✝ : IsNoetherianRing R
    s : Finset S
    hs : Eq (Algebra.adjoin R ↑s) Top.top
    ⊢ Eq (↑Top.top) Set.univ
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem _root_.Subalgebra.fg_iff_finiteType (S : Subalgebra R A) : S.FG ↔ Algebra.FiniteType R S :=
  S.fg_top.symm.trans ⟨fun h => ⟨h⟩, fun h => h.out⟩


/-- A ring morphism `A →+* B` is of `FiniteType` if `B` is finitely generated as `A`-algebra. -/
@[algebraize]
def FiniteType (f : A →+* B) : Prop :=
  @Algebra.FiniteType A B _ _ f.toAlgebra


theorem finiteType {f : A →+* B} (hf : f.Finite) : FiniteType f :=
  @Module.Finite.finiteType _ _ _ _ f.toAlgebra hf


theorem id : FiniteType (RingHom.id A) :=
  Algebra.FiniteType.self A


theorem comp_surjective {f : A →+* B} {g : B →+* C} (hf : f.FiniteType) (hg : Surjective g) :
    (g.comp f).FiniteType := by
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    f : RingHom A B
    g : RingHom B C
    hf : f.FiniteType
    hg : Function.Surjective ⇑g
    ⊢ (g.comp f).FiniteType
  -/
  algebraize_only [f, g.comp f]
  exact Algebra.FiniteType.of_surjective hf
    { g with
      toFun := g
      commutes' := fun a => rfl }
    hg


theorem of_surjective (f : A →+* B) (hf : Surjective f) : f.FiniteType := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    f : RingHom A B
    hf : Function.Surjective ⇑f
    ⊢ f.FiniteType
  -/
  rw [← f.comp_id]
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    f : RingHom A B
    hf : Function.Surjective ⇑f
    ⊢ (f.comp (RingHom.id A)).FiniteType
  -/
  exact (id A).comp_surjective hf
  /-
    🎉 no goals
  -/


theorem comp {g : B →+* C} {f : A →+* B} (hg : g.FiniteType) (hf : f.FiniteType) :
    (g.comp f).FiniteType := by
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    g : RingHom B C
    f : RingHom A B
    hg : g.FiniteType
    hf : f.FiniteType
    ⊢ (g.comp f).FiniteType
  -/
  algebraize_only [f, g, g.comp f]
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    g : RingHom B C
    f : RingHom A B
    hg : g.FiniteType
    hf : f.FiniteType
    algInst✝² : Algebra A B := f.toAlgebra
    algInst✝¹ : Algebra B C := g.toAlgebra
    algInst✝ : Algebra A C := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' (Eq. …
    ⊢ (g.comp f).FiniteType
  -/
  exact Algebra.FiniteType.trans hf hg
  /-
    🎉 no goals
  -/


theorem of_finite {f : A →+* B} (hf : f.Finite) : f.FiniteType :=
  @Module.Finite.finiteType _ _ _ _ f.toAlgebra hf


alias _root_.RingHom.Finite.to_finiteType := of_finite


theorem of_comp_finiteType {f : A →+* B} {g : B →+* C} (h : (g.comp f).FiniteType) :
    g.FiniteType := by
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    f : RingHom A B
    g : RingHom B C
    h : (g.comp f).FiniteType
    ⊢ g.FiniteType
  -/
  algebraize [f, g, g.comp f]
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    f : RingHom A B
    g : RingHom B C
    h : (g.comp f).FiniteType
    algInst✝² : Algebra A B := f.toAlgebra
    algInst✝¹ : Algebra B C := g.toAlgebra
    algInst✝ : Algebra A C := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' (Eq. …
    algebraizeInst✝ : Algebra.FiniteType A C
    ⊢ g.FiniteType
  -/
  exact Algebra.FiniteType.of_restrictScalars_finiteType A B C
  /-
    🎉 no goals
  -/


/-- An algebra morphism `A →ₐ[R] B` is of `FiniteType` if it is of finite type as ring morphism.
In other words, if `B` is finitely generated as `A`-algebra. -/
def FiniteType (f : A →ₐ[R] B) : Prop :=
  f.toRingHom.FiniteType


theorem finiteType {f : A →ₐ[R] B} (hf : f.Finite) : FiniteType f :=
  RingHom.Finite.finiteType hf


theorem id : FiniteType (AlgHom.id R A) :=
  RingHom.FiniteType.id A


theorem comp {g : B →ₐ[R] C} {f : A →ₐ[R] B} (hg : g.FiniteType) (hf : f.FiniteType) :
    (g.comp f).FiniteType :=
  RingHom.FiniteType.comp hg hf


theorem comp_surjective {f : A →ₐ[R] B} {g : B →ₐ[R] C} (hf : f.FiniteType) (hg : Surjective g) :
    (g.comp f).FiniteType :=
  RingHom.FiniteType.comp_surjective hf hg


theorem of_surjective (f : A →ₐ[R] B) (hf : Surjective f) : f.FiniteType :=
  RingHom.FiniteType.of_surjective f.toRingHom hf


theorem of_comp_finiteType {f : A →ₐ[R] B} {g : B →ₐ[R] C} (h : (g.comp f).FiniteType) :
    g.FiniteType :=
  RingHom.FiniteType.of_comp_finiteType h


/-- An element of `R[M]` is in the subalgebra generated by its support. -/
theorem mem_adjoin_support (f : R[M]) : f ∈ adjoin R (of' R M '' f.support) := by
  suffices span R (of' R M '' f.support) ≤
      Subalgebra.toSubmodule (adjoin R (of' R M '' f.support)) by
    exact this (mem_span_support f)
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : AddMonoid M
    f : AddMonoidAlgebra R M
    ⊢ LE.le (Submodule.span R (Set.image (AddMonoidAlgebra.of' R M) ↑f.support)) ( …
  -/
  rw [Submodule.span_le]
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : AddMonoid M
    f : AddMonoidAlgebra R M
    ⊢ HasSubset.Subset (Set.image (AddMonoidAlgebra.of' R M) ↑f.support) ↑(Subalge …
  -/
  exact subset_adjoin
  /-
    🎉 no goals
  -/


/-- If a set `S` generates, as algebra, `R[M]`, then the set of supports of
elements of `S` generates `R[M]`. -/
theorem support_gen_of_gen {S : Set R[M]} (hS : Algebra.adjoin R S = ⊤) :
    Algebra.adjoin R (⋃ f ∈ S, of' R M '' (f.support : Set M)) = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : AddMonoid M
    S : Set (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    ⊢ Eq (Algebra.adjoin R (Set.iUnion fun f => Set.iUnion fun h => Set.image (Add …
  -/
  refine le_antisymm le_top ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : AddMonoid M
    S : Set (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    ⊢ LE.le Top.top (Algebra.adjoin R (Set.iUnion fun f => Set.iUnion fun h => Set …
  -/
  rw [← hS, adjoin_le_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : AddMonoid M
    S : Set (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    ⊢ HasSubset.Subset S ↑(Algebra.adjoin R (Set.iUnion fun f => Set.iUnion fun h  …
  -/
  intro f hf
  have hincl :
    of' R M '' f.support ⊆ ⋃ (g : R[M]) (_ : g ∈ S), of' R M '' g.support := by
    intro s hs
    exact Set.mem_iUnion₂.2 ⟨f, ⟨hf, hs⟩⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : AddMonoid M
    S : Set (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    f : AddMonoidAlgebra R M
    hf : Membership.mem S f
    hincl : HasSubset.Subset (Set.image (AddMonoidAlgebra.of' R M) ↑f.support) (Se …
    ⊢ Membership.mem (↑(Algebra.adjoin R (Set.iUnion fun f => Set.iUnion fun h =>  …
  -/
  exact adjoin_mono hincl (mem_adjoin_support f)
  /-
    🎉 no goals
  -/


/-- If a set `S` generates, as algebra, `R[M]`, then the image of the union of
the supports of elements of `S` generates `R[M]`. -/
theorem support_gen_of_gen' {S : Set R[M]} (hS : Algebra.adjoin R S = ⊤) :
    Algebra.adjoin R (of' R M '' ⋃ f ∈ S, (f.support : Set M)) = ⊤ := by
  suffices (of' R M '' ⋃ f ∈ S, (f.support : Set M)) = ⋃ f ∈ S, of' R M '' (f.support : Set M) by
    rw [this]
    exact support_gen_of_gen hS
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : AddMonoid M
    S : Set (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    ⊢ Eq (Set.image (AddMonoidAlgebra.of' R M) (Set.iUnion fun f => Set.iUnion fun …
  -/
  simp only [Set.image_iUnion]
  /-
    🎉 no goals
  -/


/-- If `R[M]` is of finite type, then there is a `G : Finset M` such that its
image generates, as algebra, `R[M]`. -/
theorem exists_finset_adjoin_eq_top [h : FiniteType R R[M]] :
    ∃ G : Finset M, Algebra.adjoin R (of' R M '' G) = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : AddMonoid M
    h : Algebra.FiniteType R (AddMonoidAlgebra R M)
    ⊢ Exists fun G => Eq (Algebra.adjoin R (Set.image (AddMonoidAlgebra.of' R M) ↑ …
  -/
  obtain ⟨S, hS⟩ := h
  /-
    case mk.intro
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : AddMonoid M
    S : Finset (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R ↑S) Top.top
    ⊢ Exists fun G => Eq (Algebra.adjoin R (Set.image (AddMonoidAlgebra.of' R M) ↑ …
  -/
  letI : DecidableEq M := Classical.decEq M
  /-
    case mk.intro
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : AddMonoid M
    S : Finset (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R ↑S) Top.top
    this : DecidableEq M := Classical.decEq M
    ⊢ Exists fun G => Eq (Algebra.adjoin R (Set.image (AddMonoidAlgebra.of' R M) ↑ …
  -/
  use Finset.biUnion S fun f => f.support
  have : (Finset.biUnion S fun f => f.support : Set M) = ⋃ f ∈ S, (f.support : Set M) := by
    simp only [Finset.set_biUnion_coe, Finset.coe_biUnion]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : AddMonoid M
    S : Finset (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R ↑S) Top.top
    this✝ : DecidableEq M := Classical.decEq M
    this : Eq (↑(S.biUnion fun f => f.support)) (Set.iUnion fun f => Set.iUnion fu …
    ⊢ Eq (Algebra.adjoin R (Set.image (AddMonoidAlgebra.of' R M) ↑(S.biUnion fun f …
  -/
  rw [this]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : AddMonoid M
    S : Finset (AddMonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R ↑S) Top.top
    this✝ : DecidableEq M := Classical.decEq M
    this : Eq (↑(S.biUnion fun f => f.support)) (Set.iUnion fun f => Set.iUnion fu …
    ⊢ Eq (Algebra.adjoin R (Set.image (AddMonoidAlgebra.of' R M) (Set.iUnion fun f …
  -/
  exact support_gen_of_gen' hS
  /-
    🎉 no goals
  -/


/-- The image of an element `m : M` in `R[M]` belongs the submodule generated by
`S : Set M` if and only if `m ∈ S`. -/
theorem of'_mem_span [Nontrivial R] {m : M} {S : Set M} :
    of' R M m ∈ span R (of' R M '' S) ↔ m ∈ S := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddMonoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    ⊢ Iff (Membership.mem (Submodule.span R (Set.image (AddMonoidAlgebra.of' R M)  …
  -/
  refine ⟨fun h => ?_, fun h => Submodule.subset_span <| Set.mem_image_of_mem (of R M) h⟩
  erw [of', ← Finsupp.supported_eq_span_single, Finsupp.mem_supported,
    Finsupp.support_single_ne_zero _ (one_ne_zero' R)] at h
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddMonoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    h : HasSubset.Subset (↑(Singleton.singleton m)) S
    ⊢ Membership.mem S m
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/--
If the image of an element `m : M` in `R[M]` belongs the submodule generated by
the closure of some `S : Set M` then `m ∈ closure S`. -/
theorem mem_closure_of_mem_span_closure [Nontrivial R] {m : M} {S : Set M}
    (h : of' R M m ∈ span R (Submonoid.closure (of' R M '' S) : Set R[M])) :
    m ∈ closure S := by
  suffices Multiplicative.ofAdd m ∈ Submonoid.closure (Multiplicative.toAdd ⁻¹' S) by
    simpa [← toSubmonoid_closure]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddMonoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    h : Membership.mem (Submodule.span R ↑(Submonoid.closure (Set.image (AddMonoid …
    ⊢ Membership.mem (Submonoid.closure (Set.preimage (⇑Multiplicative.toAdd) S))  …
  -/
  let S' := @Submonoid.closure (Multiplicative M) Multiplicative.mulOneClass S
  have h' : Submonoid.map (of R M) S' = Submonoid.closure ((fun x : M => (of R M) x) '' S) :=
    MonoidHom.map_mclosure _ _
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddMonoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    h : Membership.mem (Submodule.span R ↑(Submonoid.closure (Set.image (AddMonoid …
    S' : Submonoid (Multiplicative M) := Submonoid.closure S
    h' : Eq (Submonoid.map (AddMonoidAlgebra.of R M) S') (Submonoid.closure (Set.i …
    ⊢ Membership.mem (Submonoid.closure (Set.preimage (⇑Multiplicative.toAdd) S))  …
  -/
  rw [Set.image_congr' (show ∀ x, of' R M x = of R M x from fun x => of'_eq_of x), ← h'] at h
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddMonoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    S' : Submonoid (Multiplicative M) := Submonoid.closure S
    h : Membership.mem (Submodule.span R ↑(Submonoid.map (AddMonoidAlgebra.of R M) …
    h' : Eq (Submonoid.map (AddMonoidAlgebra.of R M) S') (Submonoid.closure (Set.i …
    ⊢ Membership.mem (Submonoid.closure (Set.preimage (⇑Multiplicative.toAdd) S))  …
  -/
  simpa using of'_mem_span.1 h
  /-
    🎉 no goals
  -/


/-- If a set `S` generates an additive monoid `M`, then the image of `M` generates, as algebra,
`R[M]`. -/
theorem mvPolynomial_aeval_of_surjective_of_closure [AddCommMonoid M] [CommSemiring R] {S : Set M}
    (hS : closure S = ⊤) :
    Function.Surjective
      (MvPolynomial.aeval fun s : S => of' R M ↑s : MvPolynomial S R → R[M]) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : CommSemiring R
    S : Set M
    hS : Eq (AddSubmonoid.closure S) Top.top
    ⊢ Function.Surjective ⇑(MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s)
  -/
  intro f
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : CommSemiring R
    S : Set M
    hS : Eq (AddSubmonoid.closure S) Top.top
    f : AddMonoidAlgebra R M
    ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
  -/
  induction' f using induction_on with m f g ihf ihg r f ih
    /-
      case hM
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      m : M
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
    -/
  · have : m ∈ closure S := hS.symm ▸ mem_top _
    /-
      case hM
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      m : M
      this : Membership.mem (AddSubmonoid.closure S) m
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
    -/
    refine AddSubmonoid.closure_induction (fun m hm => ?_) ?_ ?_ this
      /-
        case hM.refine_1
        R : Type u_1
        M : Type u_2
        inst✝¹ : AddCommMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (AddSubmonoid.closure S) Top.top
        m✝ : M
        this : Membership.mem (AddSubmonoid.closure S) m✝
        m : M
        hm : Membership.mem S m
        ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
      -/
    · exact ⟨MvPolynomial.X ⟨m, hm⟩, MvPolynomial.aeval_X _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case hM.refine_2
        R : Type u_1
        M : Type u_2
        inst✝¹ : AddCommMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (AddSubmonoid.closure S) Top.top
        m : M
        this : Membership.mem (AddSubmonoid.closure S) m
        ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
      -/
    · exact ⟨1, map_one _⟩
      /-
        🎉 no goals
      -/
      /-
        case hM.refine_3
        R : Type u_1
        M : Type u_2
        inst✝¹ : AddCommMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (AddSubmonoid.closure S) Top.top
        m : M
        this : Membership.mem (AddSubmonoid.closure S) m
        ⊢ ∀ (x y : M), Membership.mem (AddSubmonoid.closure S) x → Membership.mem (Add …
      -/
    · rintro m₁ m₂ _ _ ⟨P₁, hP₁⟩ ⟨P₂, hP₂⟩
      exact
        ⟨P₁ * P₂, by
          rw [map_mul, hP₁, hP₂, of_apply, of_apply, of_apply, single_mul_single,
            one_mul]; rfl⟩
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      f g : AddMonoidAlgebra R M
      ihf : Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R  …
      ihg : Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R  …
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
    -/
  · rcases ihf with ⟨P, rfl⟩
    /-
      case hadd.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      g : AddMonoidAlgebra R M
      ihg : Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R  …
      P : MvPolynomial (↑S) R
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
    -/
    rcases ihg with ⟨Q, rfl⟩
    /-
      case hadd.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      P Q : MvPolynomial (↑S) R
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
    -/
    exact ⟨P + Q, map_add _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case hsmul
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      r : R
      f : AddMonoidAlgebra R M
      ih : Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M …
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
    -/
  · rcases ih with ⟨P, rfl⟩
    /-
      case hsmul.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      r : R
      P : MvPolynomial (↑S) R
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => AddMonoidAlgebra.of' R M ↑s …
    -/
    exact ⟨r • P, map_smul _ _ _⟩
    /-
      🎉 no goals
    -/


/-- If a set `S` generates an additive monoid `M`, then the image of `M` generates, as algebra,
`R[M]`. -/
theorem freeAlgebra_lift_of_surjective_of_closure [CommSemiring R] {S : Set M}
    (hS : closure S = ⊤) :
    Function.Surjective
      (FreeAlgebra.lift R fun s : S => of' R M ↑s : FreeAlgebra R S → R[M]) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : AddMonoid M
    inst✝ : CommSemiring R
    S : Set M
    hS : Eq (AddSubmonoid.closure S) Top.top
    ⊢ Function.Surjective ⇑((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M …
  -/
  intro f
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : AddMonoid M
    inst✝ : CommSemiring R
    S : Set M
    hS : Eq (AddSubmonoid.closure S) Top.top
    f : AddMonoidAlgebra R M
    ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
  -/
  induction' f using induction_on with m f g ihf ihg r f ih
    /-
      case hM
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      m : M
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
    -/
  · have : m ∈ closure S := hS.symm ▸ mem_top _
    /-
      case hM
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      m : M
      this : Membership.mem (AddSubmonoid.closure S) m
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
    -/
    refine AddSubmonoid.closure_induction (fun m hm => ?_) ?_ ?_ this
      /-
        case hM.refine_1
        R : Type u_1
        M : Type u_2
        inst✝¹ : AddMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (AddSubmonoid.closure S) Top.top
        m✝ : M
        this : Membership.mem (AddSubmonoid.closure S) m✝
        m : M
        hm : Membership.mem S m
        ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
      -/
    · exact ⟨FreeAlgebra.ι R ⟨m, hm⟩, FreeAlgebra.lift_ι_apply _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case hM.refine_2
        R : Type u_1
        M : Type u_2
        inst✝¹ : AddMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (AddSubmonoid.closure S) Top.top
        m : M
        this : Membership.mem (AddSubmonoid.closure S) m
        ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
      -/
    · exact ⟨1, map_one _⟩
      /-
        🎉 no goals
      -/
      /-
        case hM.refine_3
        R : Type u_1
        M : Type u_2
        inst✝¹ : AddMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (AddSubmonoid.closure S) Top.top
        m : M
        this : Membership.mem (AddSubmonoid.closure S) m
        ⊢ ∀ (x y : M), Membership.mem (AddSubmonoid.closure S) x → Membership.mem (Add …
      -/
    · rintro m₁ m₂ _ _ ⟨P₁, hP₁⟩ ⟨P₂, hP₂⟩
      exact
        ⟨P₁ * P₂, by
          rw [map_mul, hP₁, hP₂, of_apply, of_apply, of_apply, single_mul_single,
            one_mul]; rfl⟩
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      f g : AddMonoidAlgebra R M
      ihf : Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of'  …
      ihg : Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of'  …
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
    -/
  · rcases ihf with ⟨P, rfl⟩
    /-
      case hadd.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      g : AddMonoidAlgebra R M
      ihg : Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of'  …
      P : FreeAlgebra R ↑S
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
    -/
    rcases ihg with ⟨Q, rfl⟩
    /-
      case hadd.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      P Q : FreeAlgebra R ↑S
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
    -/
    exact ⟨P + Q, map_add _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case hsmul
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      r : R
      f : AddMonoidAlgebra R M
      ih : Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R …
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
    -/
  · rcases ih with ⟨P, rfl⟩
    /-
      case hsmul.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (AddSubmonoid.closure S) Top.top
      r : R
      P : FreeAlgebra R ↑S
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => AddMonoidAlgebra.of' R M  …
    -/
    exact ⟨r • P, map_smul _ _ _⟩
    /-
      🎉 no goals
    -/


/-- If an additive monoid `M` is finitely generated then `R[M]` is of finite
type. -/
instance finiteType_of_fg [CommRing R] [h : AddMonoid.FG M] :
    FiniteType R R[M] := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : AddMonoid M
    inst✝ : CommRing R
    h : AddMonoid.FG M
    ⊢ Algebra.FiniteType R (AddMonoidAlgebra R M)
  -/
  obtain ⟨S, hS⟩ := h.out
  exact (FiniteType.freeAlgebra R (S : Set M)).of_surjective
      (FreeAlgebra.lift R fun s : (S : Set M) => of' R M ↑s)
      (freeAlgebra_lift_of_surjective_of_closure hS)


/-- An additive monoid `M` is finitely generated if and only if `R[M]` is of
finite type. -/
theorem finiteType_iff_fg [CommRing R] [Nontrivial R] :
    FiniteType R R[M] ↔ AddMonoid.FG M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : AddMonoid M
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ Iff (Algebra.FiniteType R (AddMonoidAlgebra R M)) (AddMonoid.FG M)
  -/
  refine ⟨fun h => ?_, fun h => @AddMonoidAlgebra.finiteType_of_fg _ _ _ _ h⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : AddMonoid M
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    h : Algebra.FiniteType R (AddMonoidAlgebra R M)
    ⊢ AddMonoid.FG M
  -/
  obtain ⟨S, hS⟩ := @exists_finset_adjoin_eq_top R M _ _ h
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : AddMonoid M
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    h : Algebra.FiniteType R (AddMonoidAlgebra R M)
    S : Finset M
    hS : Eq (Algebra.adjoin R (Set.image (AddMonoidAlgebra.of' R M) ↑S)) Top.top
    ⊢ AddMonoid.FG M
  -/
  refine AddMonoid.fg_def.2 ⟨S, (eq_top_iff' _).2 fun m => ?_⟩
  have hm : of' R M m ∈ Subalgebra.toSubmodule (adjoin R (of' R M '' ↑S)) := by
    simp only [hS, top_toSubmodule, Submodule.mem_top]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : AddMonoid M
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    h : Algebra.FiniteType R (AddMonoidAlgebra R M)
    S : Finset M
    hS : Eq (Algebra.adjoin R (Set.image (AddMonoidAlgebra.of' R M) ↑S)) Top.top
    m : M
    hm : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin R (Set.image (AddM …
    ⊢ Membership.mem (AddSubmonoid.closure ↑S) m
  -/
  rw [adjoin_eq_span] at hm
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : AddMonoid M
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    h : Algebra.FiniteType R (AddMonoidAlgebra R M)
    S : Finset M
    hS : Eq (Algebra.adjoin R (Set.image (AddMonoidAlgebra.of' R M) ↑S)) Top.top
    m : M
    hm : Membership.mem (Submodule.span R ↑(Submonoid.closure (Set.image (AddMonoi …
    ⊢ Membership.mem (AddSubmonoid.closure ↑S) m
  -/
  exact mem_closure_of_mem_span_closure hm
  /-
    🎉 no goals
  -/


/-- If `R[M]` is of finite type then `M` is finitely generated. -/
theorem fg_of_finiteType [CommRing R] [Nontrivial R] [h : FiniteType R R[M]] :
    AddMonoid.FG M :=
  finiteType_iff_fg.1 h


/-- An additive group `G` is finitely generated if and only if `R[G]` is of
finite type. -/
theorem finiteType_iff_group_fg {G : Type*} [AddCommGroup G] [CommRing R] [Nontrivial R] :
    FiniteType R R[G] ↔ AddGroup.FG G := by
  /-
    R : Type u_1
    G : Type u_3
    inst✝² : AddCommGroup G
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ Iff (Algebra.FiniteType R (AddMonoidAlgebra R G)) (AddGroup.FG G)
  -/
  simpa [AddGroup.fg_iff_addMonoid_fg] using finiteType_iff_fg
  /-
    🎉 no goals
  -/


/-- An element of `MonoidAlgebra R M` is in the subalgebra generated by its support. -/
theorem mem_adjoin_support (f : MonoidAlgebra R M) : f ∈ adjoin R (of R M '' f.support) := by
  suffices span R (of R M '' f.support) ≤ Subalgebra.toSubmodule (adjoin R (of R M '' f.support)) by
    exact this (mem_span_support f)
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Monoid M
    f : MonoidAlgebra R M
    ⊢ LE.le (Submodule.span R (Set.image ⇑(MonoidAlgebra.of R M) ↑f.support)) (Sub …
  -/
  rw [Submodule.span_le]
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Monoid M
    f : MonoidAlgebra R M
    ⊢ HasSubset.Subset (Set.image ⇑(MonoidAlgebra.of R M) ↑f.support) ↑(Subalgebra …
  -/
  exact subset_adjoin
  /-
    🎉 no goals
  -/


/-- If a set `S` generates, as algebra, `MonoidAlgebra R M`, then the set of supports of elements
of `S` generates `MonoidAlgebra R M`. -/
theorem support_gen_of_gen {S : Set (MonoidAlgebra R M)} (hS : Algebra.adjoin R S = ⊤) :
    Algebra.adjoin R (⋃ f ∈ S, of R M '' (f.support : Set M)) = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Monoid M
    S : Set (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    ⊢ Eq (Algebra.adjoin R (Set.iUnion fun f => Set.iUnion fun h => Set.image ⇑(Mo …
  -/
  refine le_antisymm le_top ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Monoid M
    S : Set (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    ⊢ LE.le Top.top (Algebra.adjoin R (Set.iUnion fun f => Set.iUnion fun h => Set …
  -/
  rw [← hS, adjoin_le_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Monoid M
    S : Set (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    ⊢ HasSubset.Subset S ↑(Algebra.adjoin R (Set.iUnion fun f => Set.iUnion fun h  …
  -/
  intro f hf
  -- Porting note: ⋃ notation did not work here. Was
  -- ⋃ (g : MonoidAlgebra R M) (H : g ∈ S), (of R M '' g.support)
  have hincl : (of R M '' f.support) ⊆
      Set.iUnion fun (g : MonoidAlgebra R M)
        => Set.iUnion fun (_ : g ∈ S) => (of R M '' g.support) := by
    intro s hs
    exact Set.mem_iUnion₂.2 ⟨f, ⟨hf, hs⟩⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Monoid M
    S : Set (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    f : MonoidAlgebra R M
    hf : Membership.mem S f
    hincl : HasSubset.Subset (Set.image ⇑(MonoidAlgebra.of R M) ↑f.support) (Set.i …
    ⊢ Membership.mem (↑(Algebra.adjoin R (Set.iUnion fun f => Set.iUnion fun h =>  …
  -/
  exact adjoin_mono hincl (mem_adjoin_support f)
  /-
    🎉 no goals
  -/


/-- If a set `S` generates, as algebra, `MonoidAlgebra R M`, then the image of the union of the
supports of elements of `S` generates `MonoidAlgebra R M`. -/
theorem support_gen_of_gen' {S : Set (MonoidAlgebra R M)} (hS : Algebra.adjoin R S = ⊤) :
    Algebra.adjoin R (of R M '' ⋃ f ∈ S, (f.support : Set M)) = ⊤ := by
  suffices (of R M '' ⋃ f ∈ S, (f.support : Set M)) = ⋃ f ∈ S, of R M '' (f.support : Set M) by
    rw [this]
    exact support_gen_of_gen hS
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : Monoid M
    S : Set (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R S) Top.top
    ⊢ Eq (Set.image (⇑(MonoidAlgebra.of R M)) (Set.iUnion fun f => Set.iUnion fun  …
  -/
  simp only [Set.image_iUnion]
  /-
    🎉 no goals
  -/


/-- If `MonoidAlgebra R M` is of finite type, then there is a `G : Finset M` such that its image
generates, as algebra, `MonoidAlgebra R M`. -/
theorem exists_finset_adjoin_eq_top [h : FiniteType R (MonoidAlgebra R M)] :
    ∃ G : Finset M, Algebra.adjoin R (of R M '' G) = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Monoid M
    h : Algebra.FiniteType R (MonoidAlgebra R M)
    ⊢ Exists fun G => Eq (Algebra.adjoin R (Set.image ⇑(MonoidAlgebra.of R M) ↑G)) …
  -/
  obtain ⟨S, hS⟩ := h
  /-
    case mk.intro
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Monoid M
    S : Finset (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R ↑S) Top.top
    ⊢ Exists fun G => Eq (Algebra.adjoin R (Set.image ⇑(MonoidAlgebra.of R M) ↑G)) …
  -/
  letI : DecidableEq M := Classical.decEq M
  /-
    case mk.intro
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Monoid M
    S : Finset (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R ↑S) Top.top
    this : DecidableEq M := Classical.decEq M
    ⊢ Exists fun G => Eq (Algebra.adjoin R (Set.image ⇑(MonoidAlgebra.of R M) ↑G)) …
  -/
  use Finset.biUnion S fun f => f.support
  have : (Finset.biUnion S fun f => f.support : Set M) = ⋃ f ∈ S, (f.support : Set M) := by
    simp only [Finset.set_biUnion_coe, Finset.coe_biUnion]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Monoid M
    S : Finset (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R ↑S) Top.top
    this✝ : DecidableEq M := Classical.decEq M
    this : Eq (↑(S.biUnion fun f => f.support)) (Set.iUnion fun f => Set.iUnion fu …
    ⊢ Eq (Algebra.adjoin R (Set.image ⇑(MonoidAlgebra.of R M) ↑(S.biUnion fun f => …
  -/
  rw [this]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Monoid M
    S : Finset (MonoidAlgebra R M)
    hS : Eq (Algebra.adjoin R ↑S) Top.top
    this✝ : DecidableEq M := Classical.decEq M
    this : Eq (↑(S.biUnion fun f => f.support)) (Set.iUnion fun f => Set.iUnion fu …
    ⊢ Eq (Algebra.adjoin R (Set.image (⇑(MonoidAlgebra.of R M)) (Set.iUnion fun f  …
  -/
  exact support_gen_of_gen' hS
  /-
    🎉 no goals
  -/


/-- The image of an element `m : M` in `MonoidAlgebra R M` belongs the submodule generated by
`S : Set M` if and only if `m ∈ S`. -/
theorem of_mem_span_of_iff [Nontrivial R] {m : M} {S : Set M} :
    of R M m ∈ span R (of R M '' S) ↔ m ∈ S := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Monoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    ⊢ Iff (Membership.mem (Submodule.span R (Set.image (⇑(MonoidAlgebra.of R M)) S …
  -/
  refine ⟨fun h => ?_, fun h => Submodule.subset_span <| Set.mem_image_of_mem (of R M) h⟩
  erw [of, MonoidHom.coe_mk, ← Finsupp.supported_eq_span_single, Finsupp.mem_supported,
    Finsupp.support_single_ne_zero _ (one_ne_zero' R)] at h
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Monoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    h : HasSubset.Subset (↑(Singleton.singleton m)) S
    ⊢ Membership.mem S m
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/--
If the image of an element `m : M` in `MonoidAlgebra R M` belongs the submodule generated by the
closure of some `S : Set M` then `m ∈ closure S`. -/
theorem mem_closure_of_mem_span_closure [Nontrivial R] {m : M} {S : Set M}
    (h : of R M m ∈ span R (Submonoid.closure (of R M '' S) : Set (MonoidAlgebra R M))) :
    m ∈ closure S := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Monoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    h : Membership.mem (Submodule.span R ↑(Submonoid.closure (Set.image (⇑(MonoidA …
    ⊢ Membership.mem (Submonoid.closure S) m
  -/
  rw [← MonoidHom.map_mclosure] at h
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Monoid M
    inst✝ : Nontrivial R
    m : M
    S : Set M
    h : Membership.mem (Submodule.span R ↑(Submonoid.map (MonoidAlgebra.of R M) (S …
    ⊢ Membership.mem (Submonoid.closure S) m
  -/
  simpa using of_mem_span_of_iff.1 h
  /-
    🎉 no goals
  -/


/-- If a set `S` generates a monoid `M`, then the image of `M` generates, as algebra,
`MonoidAlgebra R M`. -/
theorem mvPolynomial_aeval_of_surjective_of_closure [CommMonoid M] [CommSemiring R] {S : Set M}
    (hS : closure S = ⊤) :
    Function.Surjective
      (MvPolynomial.aeval fun s : S => of R M ↑s : MvPolynomial S R → MonoidAlgebra R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommMonoid M
    inst✝ : CommSemiring R
    S : Set M
    hS : Eq (Submonoid.closure S) Top.top
    ⊢ Function.Surjective ⇑(MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)
  -/
  intro f
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : CommMonoid M
    inst✝ : CommSemiring R
    S : Set M
    hS : Eq (Submonoid.closure S) Top.top
    f : MonoidAlgebra R M
    ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
  -/
  induction' f using induction_on with m f g ihf ihg r f ih
    /-
      case hM
      R : Type u_1
      M : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      m : M
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
    -/
  · have : m ∈ closure S := hS.symm ▸ mem_top _
    /-
      case hM
      R : Type u_1
      M : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      m : M
      this : Membership.mem (Submonoid.closure S) m
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
    -/
    refine Submonoid.closure_induction (fun m hm => ?_) ?_ ?_ this
      /-
        case hM.refine_1
        R : Type u_1
        M : Type u_2
        inst✝¹ : CommMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (Submonoid.closure S) Top.top
        m✝ : M
        this : Membership.mem (Submonoid.closure S) m✝
        m : M
        hm : Membership.mem S m
        ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
      -/
    · exact ⟨MvPolynomial.X ⟨m, hm⟩, MvPolynomial.aeval_X _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case hM.refine_2
        R : Type u_1
        M : Type u_2
        inst✝¹ : CommMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (Submonoid.closure S) Top.top
        m : M
        this : Membership.mem (Submonoid.closure S) m
        ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
      -/
    · exact ⟨1, map_one _⟩
      /-
        🎉 no goals
      -/
      /-
        case hM.refine_3
        R : Type u_1
        M : Type u_2
        inst✝¹ : CommMonoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (Submonoid.closure S) Top.top
        m : M
        this : Membership.mem (Submonoid.closure S) m
        ⊢ ∀ (x y : M), Membership.mem (Submonoid.closure S) x → Membership.mem (Submon …
      -/
    · rintro m₁ m₂ _ _ ⟨P₁, hP₁⟩ ⟨P₂, hP₂⟩
      exact
        ⟨P₁ * P₂, by
          rw [map_mul, hP₁, hP₂, of_apply, of_apply, of_apply, single_mul_single, one_mul]⟩
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      f g : MonoidAlgebra R M
      ihf : Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M)  …
      ihg : Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M)  …
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
    -/
  · rcases ihf with ⟨P, rfl⟩; rcases ihg with ⟨Q, rfl⟩
    /-
      case hadd.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      P Q : MvPolynomial (↑S) R
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
    -/
    exact ⟨P + Q, map_add _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case hsmul
      R : Type u_1
      M : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      r : R
      f : MonoidAlgebra R M
      ih : Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑ …
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
    -/
  · rcases ih with ⟨P, rfl⟩
    /-
      case hsmul.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      r : R
      P : MvPolynomial (↑S) R
      ⊢ Exists fun a => Eq ((MvPolynomial.aeval fun s => (MonoidAlgebra.of R M) ↑s)  …
    -/
    exact ⟨r • P, map_smul _ _ _⟩
    /-
      🎉 no goals
    -/



/-- If a set `S` generates an additive monoid `M`, then the image of `M` generates, as algebra,
`R[M]`. -/
theorem freeAlgebra_lift_of_surjective_of_closure [CommSemiring R] {S : Set M}
    (hS : closure S = ⊤) :
    Function.Surjective
      (FreeAlgebra.lift R fun s : S => of R M ↑s : FreeAlgebra R S → MonoidAlgebra R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Monoid M
    inst✝ : CommSemiring R
    S : Set M
    hS : Eq (Submonoid.closure S) Top.top
    ⊢ Function.Surjective ⇑((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s)
  -/
  intro f
  /-
    R : Type u_1
    M : Type u_2
    inst✝¹ : Monoid M
    inst✝ : CommSemiring R
    S : Set M
    hS : Eq (Submonoid.closure S) Top.top
    f : MonoidAlgebra R M
    ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
  -/
  induction' f using induction_on with m f g ihf ihg r f ih
    /-
      case hM
      R : Type u_1
      M : Type u_2
      inst✝¹ : Monoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      m : M
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
    -/
  · have : m ∈ closure S := hS.symm ▸ mem_top _
    /-
      case hM
      R : Type u_1
      M : Type u_2
      inst✝¹ : Monoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      m : M
      this : Membership.mem (Submonoid.closure S) m
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
    -/
    refine Submonoid.closure_induction (fun m hm => ?_) ?_ ?_ this
      /-
        case hM.refine_1
        R : Type u_1
        M : Type u_2
        inst✝¹ : Monoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (Submonoid.closure S) Top.top
        m✝ : M
        this : Membership.mem (Submonoid.closure S) m✝
        m : M
        hm : Membership.mem S m
        ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
      -/
    · exact ⟨FreeAlgebra.ι R ⟨m, hm⟩, FreeAlgebra.lift_ι_apply _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case hM.refine_2
        R : Type u_1
        M : Type u_2
        inst✝¹ : Monoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (Submonoid.closure S) Top.top
        m : M
        this : Membership.mem (Submonoid.closure S) m
        ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
      -/
    · exact ⟨1, map_one _⟩
      /-
        🎉 no goals
      -/
      /-
        case hM.refine_3
        R : Type u_1
        M : Type u_2
        inst✝¹ : Monoid M
        inst✝ : CommSemiring R
        S : Set M
        hS : Eq (Submonoid.closure S) Top.top
        m : M
        this : Membership.mem (Submonoid.closure S) m
        ⊢ ∀ (x y : M), Membership.mem (Submonoid.closure S) x → Membership.mem (Submon …
      -/
    · rintro m₁ m₂ _ _ ⟨P₁, hP₁⟩ ⟨P₂, hP₂⟩
      exact
        ⟨P₁ * P₂, by
          rw [map_mul, hP₁, hP₂, of_apply, of_apply, of_apply, single_mul_single, one_mul]⟩
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝¹ : Monoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      f g : MonoidAlgebra R M
      ihf : Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M …
      ihg : Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M …
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
    -/
  · rcases ihf with ⟨P, rfl⟩
    /-
      case hadd.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : Monoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      g : MonoidAlgebra R M
      ihg : Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M …
      P : FreeAlgebra R ↑S
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
    -/
    rcases ihg with ⟨Q, rfl⟩
    /-
      case hadd.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : Monoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      P Q : FreeAlgebra R ↑S
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
    -/
    exact ⟨P + Q, map_add _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case hsmul
      R : Type u_1
      M : Type u_2
      inst✝¹ : Monoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      r : R
      f : MonoidAlgebra R M
      ih : Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) …
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
    -/
  · rcases ih with ⟨P, rfl⟩
    /-
      case hsmul.intro
      R : Type u_1
      M : Type u_2
      inst✝¹ : Monoid M
      inst✝ : CommSemiring R
      S : Set M
      hS : Eq (Submonoid.closure S) Top.top
      r : R
      P : FreeAlgebra R ↑S
      ⊢ Exists fun a => Eq (((FreeAlgebra.lift R) fun s => (MonoidAlgebra.of R M) ↑s …
    -/
    exact ⟨r • P, map_smul _ _ _⟩
    /-
      🎉 no goals
    -/


/-- If a monoid `M` is finitely generated then `MonoidAlgebra R M` is of finite type. -/
instance finiteType_of_fg [CommRing R] [Monoid.FG M] : FiniteType R (MonoidAlgebra R M) :=
  (AddMonoidAlgebra.finiteType_of_fg R (Additive M)).equiv (toAdditiveAlgEquiv R M).symm


/-- A monoid `M` is finitely generated if and only if `MonoidAlgebra R M` is of finite type. -/
theorem finiteType_iff_fg [CommRing R] [Nontrivial R] :
    FiniteType R (MonoidAlgebra R M) ↔ Monoid.FG M :=
  ⟨fun h =>
    Monoid.fg_iff_add_fg.2 <|
      AddMonoidAlgebra.finiteType_iff_fg.1 <| h.equiv <| toAdditiveAlgEquiv R M,
    fun h => @MonoidAlgebra.finiteType_of_fg _ _ _ _ h⟩


/-- If `MonoidAlgebra R M` is of finite type then `M` is finitely generated. -/
theorem fg_of_finiteType [CommRing R] [Nontrivial R] [h : FiniteType R (MonoidAlgebra R M)] :
    Monoid.FG M :=
  finiteType_iff_fg.1 h


/-- A group `G` is finitely generated if and only if `R[G]` is of finite type. -/
theorem finiteType_iff_group_fg {G : Type*} [Group G] [CommRing R] [Nontrivial R] :
    FiniteType R (MonoidAlgebra R G) ↔ Group.FG G := by
  /-
    R : Type u_1
    G : Type u_3
    inst✝² : Group G
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ Iff (Algebra.FiniteType R (MonoidAlgebra R G)) (Group.FG G)
  -/
  simpa [Group.fg_iff_monoid_fg] using finiteType_iff_fg
  /-
    🎉 no goals
  -/


open Submodule Module Module.Finite in
/-- Any commutative ring `R` satisfies the `OrzechProperty`, that is, for any finitely generated
`R`-module `M`, any surjective homomorphism `f : N →ₗ[R] M` from a submodule `N` of `M` to `M`
is injective.

This is a consequence of Noetherian case
(`IsNoetherian.injective_of_surjective_of_injective`), which requires that `M` is a
Noetherian module, but allows `R` to be non-commutative. The reduction of this result to
Noetherian case is adapted from <https://math.stackexchange.com/a/1066110>:
suppose `{ m_j }` is a finite set of generator of `M`, for any `n : N` one can write
`i n = ∑ j, b_j * m_j` for `{ b_j }` in `R`, here `i : N →ₗ[R] M` is the standard inclusion.
We can choose `{ n_j }` which are preimages of `{ m_j }` under `f`, and can choose
`{ c_jl }` in `R` such that `i n_j = ∑ l, c_jl * m_l` for each `j`.
Now let `A` be the subring of `R` generated by `{ b_j }` and `{ c_jl }`, then it is
Noetherian. Let `N'` be the `A`-submodule of `N` generated by `n` and `{ n_j }`,
`M'` be the `A`-submodule of `M` generated by `{ m_j }`,
then it's easy to see that `i` and `f` restrict to `N' →ₗ[A] M'`,
and the restricted version of `f` is surjective, hence by Noetherian case,
it is also injective, in particular, if `f n = 0`, then `n = 0`.

See also Orzech's original paper: *Onto endomorphisms are isomorphisms* [orzech1971]. -/
instance (priority := 100) CommRing.orzechProperty
    (R : Type*) [CommRing R] : OrzechProperty R := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ OrzechProperty R
  -/
  refine ⟨fun {M} _ _ _ {N} f hf ↦ ?_⟩
  /-
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    ⊢ Function.Injective ⇑f
  -/
  letI := addCommMonoidToAddCommGroup R (M := M)
  /-
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    ⊢ Function.Injective ⇑f
  -/
  letI := addCommMonoidToAddCommGroup R (M := N)
  /-
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    ⊢ Function.Injective ⇑f
  -/
  let i := N.subtype
  /-
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    ⊢ Function.Injective ⇑f
  -/
  let hi : Function.Injective i := N.injective_subtype
  /-
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    ⊢ Function.Injective ⇑f
  -/
  refine LinearMap.ker_eq_bot.1 <| LinearMap.ker_eq_bot'.2 fun n hn ↦ ?_
  /-
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    ⊢ Eq n 0
  -/
  obtain ⟨k, mj, hmj⟩ := exists_fin (R := R) (M := M)
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    k : Nat
    mj : Fin k → M
    hmj : Eq (Submodule.span R (Set.range mj)) Top.top
    ⊢ Eq n 0
  -/
  rw [← surjective_piEquiv_apply_iff] at hmj
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    k : Nat
    mj : Fin k → M
    hmj : Function.Surjective ⇑((Module.piEquiv (Fin k) R M) mj)
    ⊢ Eq n 0
  -/
  obtain ⟨b, hb⟩ := hmj (i n)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    k : Nat
    mj : Fin k → M
    hmj : Function.Surjective ⇑((Module.piEquiv (Fin k) R M) mj)
    b : Fin k → R
    hb : Eq (((Module.piEquiv (Fin k) R M) mj) b) (i n)
    ⊢ Eq n 0
  -/
  choose nj hnj using fun j ↦ hf (mj j)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    k : Nat
    mj : Fin k → M
    hmj : Function.Surjective ⇑((Module.piEquiv (Fin k) R M) mj)
    b : Fin k → R
    hb : Eq (((Module.piEquiv (Fin k) R M) mj) b) (i n)
    nj : Fin k → Subtype fun x => Membership.mem N x
    hnj : ∀ (j : Fin k), Eq (f (nj j)) (mj j)
    ⊢ Eq n 0
  -/
  choose c hc using fun j ↦ hmj (i (nj j))
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    k : Nat
    mj : Fin k → M
    hmj : Function.Surjective ⇑((Module.piEquiv (Fin k) R M) mj)
    b : Fin k → R
    hb : Eq (((Module.piEquiv (Fin k) R M) mj) b) (i n)
    nj : Fin k → Subtype fun x => Membership.mem N x
    hnj : ∀ (j : Fin k), Eq (f (nj j)) (mj j)
    c : Fin k → Fin k → R
    hc : ∀ (j : Fin k), Eq (((Module.piEquiv (Fin k) R M) mj) (c j)) (i (nj j))
    ⊢ Eq n 0
  -/
  let A := Subring.closure (Set.range b ∪ Set.range c.uncurry)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    k : Nat
    mj : Fin k → M
    hmj : Function.Surjective ⇑((Module.piEquiv (Fin k) R M) mj)
    b : Fin k → R
    hb : Eq (((Module.piEquiv (Fin k) R M) mj) b) (i n)
    nj : Fin k → Subtype fun x => Membership.mem N x
    hnj : ∀ (j : Fin k), Eq (f (nj j)) (mj j)
    c : Fin k → Fin k → R
    hc : ∀ (j : Fin k), Eq (((Module.piEquiv (Fin k) R M) mj) (c j)) (i (nj j))
    A : Subring R := Subring.closure (Union.union (Set.range b) (Set.range (Functi …
    ⊢ Eq n 0
  -/
  let N' := span A ({n} ∪ Set.range nj)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMon …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    k : Nat
    mj : Fin k → M
    hmj : Function.Surjective ⇑((Module.piEquiv (Fin k) R M) mj)
    b : Fin k → R
    hb : Eq (((Module.piEquiv (Fin k) R M) mj) b) (i n)
    nj : Fin k → Subtype fun x => Membership.mem N x
    hnj : ∀ (j : Fin k), Eq (f (nj j)) (mj j)
    c : Fin k → Fin k → R
    hc : ∀ (j : Fin k), Eq (((Module.piEquiv (Fin k) R M) mj) (c j)) (i (nj j))
    A : Subring R := Subring.closure (Union.union (Set.range b) (Set.range (Functi …
    N' : Submodule (Subtype fun x => Membership.mem A x) (Subtype fun x => Members …
    ⊢ Eq n 0
  -/
  let M' := span A (Set.range mj)
  haveI : IsNoetherianRing A := is_noetherian_subring_closure _
    (.union (Set.finite_range _) (Set.finite_range _))
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    M : Type u_1
    x✝² : AddCommMonoid M
    x✝¹ : Module R M
    x✝ : Module.Finite R M
    N : Submodule R M
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M
    hf : Function.Surjective ⇑f
    this✝¹ : AddCommGroup M := Module.addCommMonoidToAddCommGroup R
    this✝ : AddCommGroup (Subtype fun x => Membership.mem N x) := Module.addCommMo …
    i : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem N x) M := N.subt …
    hi : Function.Injective ⇑i := Submodule.injective_subtype N
    n : Subtype fun x => Membership.mem N x
    hn : Eq (f n) 0
    k : Nat
    mj : Fin k → M
    hmj : Function.Surjective ⇑((Module.piEquiv (Fin k) R M) mj)
    b : Fin k → R
    hb : Eq (((Module.piEquiv (Fin k) R M) mj) b) (i n)
    nj : Fin k → Subtype fun x => Membership.mem N x
    hnj : ∀ (j : Fin k), Eq (f (nj j)) (mj j)
    c : Fin k → Fin k → R
    hc : ∀ (j : Fin k), Eq (((Module.piEquiv (Fin k) R M) mj) (c j)) (i (nj j))
    A : Subring R := Subring.closure (Union.union (Set.range b) (Set.range (Functi …
    N' : Submodule (Subtype fun x => Membership.mem A x) (Subtype fun x => Members …
    M' : Submodule (Subtype fun x => Membership.mem A x) M := Submodule.span (Subt …
    this : IsNoetherianRing (Subtype fun x => Membership.mem A x)
    ⊢ Eq n 0
  -/
  haveI : Module.Finite A M' := span_of_finite A (Set.finite_range _)
  refine congr($((LinearMap.ker_eq_bot'.1 <| LinearMap.ker_eq_bot.2 <|
    IsNoetherian.injective_of_surjective_of_injective
      ((i.restrictScalars A).restrict fun x hx ↦ ?_ : N' →ₗ[A] M')
      ((f.restrictScalars A).restrict fun x hx ↦ ?_ : N' →ₗ[A] M')
      (fun _ _ h ↦ injective_subtype _ (hi congr(($h).1)))
      fun ⟨x, hx⟩ ↦ ?_) ⟨n, (subset_span (by simp))⟩ (Subtype.val_injective hn)).1)
  · induction hx using span_induction with
    | mem x hx =>
      change i x ∈ M'
      simp only [Set.singleton_union, Set.mem_insert_iff, Set.mem_range] at hx
      rcases hx with hx | ⟨j, rfl⟩
      · rw [hx, ← hb, piEquiv_apply_apply]
        refine Submodule.sum_mem _ fun j _ ↦ ?_
        let b' : A := ⟨b j, Subring.subset_closure (by simp)⟩
        rw [show b j • mj j = b' • mj j from rfl]
        exact smul_mem _ _ (subset_span (by simp))
      · rw [← hc, piEquiv_apply_apply]
        refine Submodule.sum_mem _ fun j' _ ↦ ?_
        let c' : A := ⟨c j j', Subring.subset_closure
          (by simp [show ∃ a b, c a b = c j j' from ⟨j, j', rfl⟩])⟩
        rw [show c j j' • mj j' = c' • mj j' from rfl]
        exact smul_mem _ _ (subset_span (by simp))
    | zero => simp
    | add x _ y _ hx hy => rw [map_add]; exact add_mem hx hy
    | smul a x _ hx => rw [map_smul]; exact smul_mem _ _ hx
  · induction hx using span_induction with
    | mem x hx =>
      change f x ∈ M'
      simp only [Set.singleton_union, Set.mem_insert_iff, Set.mem_range] at hx
      rcases hx with hx | ⟨j, rfl⟩
      · rw [hx, hn]; exact zero_mem _
      · exact subset_span (by simp [hnj])
    | zero => simp
    | add x _ y _ hx hy => rw [map_add]; exact add_mem hx hy
    | smul a x _ hx => rw [map_smul]; exact smul_mem _ _ hx
  suffices x ∈ LinearMap.range ((f.restrictScalars A).domRestrict N') by
    obtain ⟨a, ha⟩ := this
    exact ⟨a, Subtype.val_injective ha⟩
  induction hx using span_induction with
  | mem x hx =>
    obtain ⟨j, rfl⟩ := hx
    exact ⟨⟨nj j, subset_span (by simp)⟩, hnj j⟩
  | zero => exact zero_mem _
  | add x y _ _ hx hy => exact add_mem hx hy
  | smul a x _ hx => exact smul_mem _ a hx


/-- A theorem by Vasconcelos, given a finite module `M` over a commutative ring, any
surjective endomorphism of `M` is also injective.
It is a consequence of the fact `CommRing.orzechProperty`
that any commutative ring `R` satisfies the `OrzechProperty`;
please use `OrzechProperty.injective_of_surjective_endomorphism` instead.
This is similar to `IsNoetherian.injective_of_surjective_endomorphism` but only applies in the
commutative case, but does not use a Noetherian hypothesis. -/
@[deprecated OrzechProperty.injective_of_surjective_endomorphism (since := "2024-05-30")]
theorem Module.Finite.injective_of_surjective_endomorphism {R : Type*} [CommRing R] {M : Type*}
    [AddCommGroup M] [Module R M] [Module.Finite R M] (f : M →ₗ[R] M)
    (f_surj : Function.Surjective f) : Function.Injective f :=
  OrzechProperty.injective_of_surjective_endomorphism f f_surj


