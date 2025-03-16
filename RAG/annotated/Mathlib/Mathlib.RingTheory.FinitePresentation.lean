/-- An algebra over a commutative semiring is `Algebra.FinitePresentation` if it is the quotient of
a polynomial ring in `n` variables by a finitely generated ideal. -/
class Algebra.FinitePresentation [CommSemiring R] [Semiring A] [Algebra R A] : Prop where
  out : ∃ (n : ℕ) (f : MvPolynomial (Fin n) R →ₐ[R] A), Surjective f ∧ f.toRingHom.ker.FG


/-- A finitely presented algebra is of finite type. -/
instance of_finitePresentation [FinitePresentation R A] : FiniteType R A := by
  /-
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    ⊢ Algebra.FiniteType R A
  -/
  obtain ⟨n, f, hf⟩ := FinitePresentation.out (R := R) (A := A)
  /-
    case intro.intro
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
    ⊢ Algebra.FiniteType R A
  -/
  apply FiniteType.iff_quotient_mvPolynomial''.2
  /-
    case intro.intro
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
    ⊢ Exists fun n => Exists fun f => Function.Surjective ⇑f
  -/
  exact ⟨n, f, hf.1⟩
  /-
    🎉 no goals
  -/


/-- An algebra over a Noetherian ring is finitely generated if and only if it is finitely
presented. -/
theorem of_finiteType [IsNoetherianRing R] : FiniteType R A ↔ FinitePresentation R A := by
  /-
    R : Type w₁
    A : Type w₂
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : IsNoetherianRing R
    ⊢ Iff (Algebra.FiniteType R A) (Algebra.FinitePresentation R A)
  -/
  refine ⟨fun h => ?_, fun hfp => Algebra.FiniteType.of_finitePresentation⟩
  /-
    R : Type w₁
    A : Type w₂
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : IsNoetherianRing R
    h : Algebra.FiniteType R A
    ⊢ Algebra.FinitePresentation R A
  -/
  obtain ⟨n, f, hf⟩ := Algebra.FiniteType.iff_quotient_mvPolynomial''.1 h
  /-
    case intro.intro
    R : Type w₁
    A : Type w₂
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : IsNoetherianRing R
    h : Algebra.FiniteType R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : Function.Surjective ⇑f
    ⊢ Algebra.FinitePresentation R A
  -/
  refine ⟨n, f, hf, ?_⟩
  /-
    case intro.intro
    R : Type w₁
    A : Type w₂
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : IsNoetherianRing R
    h : Algebra.FiniteType R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : Function.Surjective ⇑f
    ⊢ (RingHom.ker f.toRingHom).FG
  -/
  have hnoet : IsNoetherianRing (MvPolynomial (Fin n) R) := by infer_instance
  -- Porting note: rewrote code to help typeclass inference
  /-
    case intro.intro
    R : Type w₁
    A : Type w₂
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : IsNoetherianRing R
    h : Algebra.FiniteType R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : Function.Surjective ⇑f
    hnoet : IsNoetherianRing (MvPolynomial (Fin n) R)
    ⊢ (RingHom.ker f.toRingHom).FG
  -/
  rw [isNoetherianRing_iff] at hnoet
  /-
    case intro.intro
    R : Type w₁
    A : Type w₂
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : IsNoetherianRing R
    h : Algebra.FiniteType R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : Function.Surjective ⇑f
    hnoet : IsNoetherian (MvPolynomial (Fin n) R) (MvPolynomial (Fin n) R)
    ⊢ (RingHom.ker f.toRingHom).FG
  -/
  letI : Module (MvPolynomial (Fin n) R) (MvPolynomial (Fin n) R) := Semiring.toModule
  /-
    case intro.intro
    R : Type w₁
    A : Type w₂
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : IsNoetherianRing R
    h : Algebra.FiniteType R A
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : Function.Surjective ⇑f
    hnoet : IsNoetherian (MvPolynomial (Fin n) R) (MvPolynomial (Fin n) R)
    this : Module (MvPolynomial (Fin n) R) (MvPolynomial (Fin n) R) := Semiring.to …
    ⊢ (RingHom.ker f.toRingHom).FG
  -/
  convert hnoet.noetherian (RingHom.ker f.toRingHom)
  /-
    🎉 no goals
  -/


/-- If `e : A ≃ₐ[R] B` and `A` is finitely presented, then so is `B`. -/
theorem equiv [FinitePresentation R A] (e : A ≃ₐ[R] B) : FinitePresentation R B := by
  /-
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    e : AlgEquiv R A B
    ⊢ Algebra.FinitePresentation R B
  -/
  obtain ⟨n, f, hf⟩ := FinitePresentation.out (R := R) (A := A)
  /-
    case intro.intro
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    e : AlgEquiv R A B
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
    ⊢ Algebra.FinitePresentation R B
  -/
  use n, AlgHom.comp (↑e) f
  /-
    case h
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FinitePresentation R A
    e : AlgEquiv R A B
    n : Nat
    f : AlgHom R (MvPolynomial (Fin n) R) A
    hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
    ⊢ And (Function.Surjective ⇑((↑e).comp f)) (RingHom.ker ((↑e).comp f).toRingHo …
  -/
  constructor
    /-
      case h.left
      R : Type w₁
      A : Type w₂
      B : Type w₃
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      inst✝ : Algebra.FinitePresentation R A
      e : AlgEquiv R A B
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      ⊢ Function.Surjective ⇑((↑e).comp f)
    -/
  · rw [AlgHom.coe_comp]
    /-
      case h.left
      R : Type w₁
      A : Type w₂
      B : Type w₃
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      inst✝ : Algebra.FinitePresentation R A
      e : AlgEquiv R A B
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      ⊢ Function.Surjective (Function.comp ⇑↑e ⇑f)
    -/
    exact Function.Surjective.comp e.surjective hf.1
    /-
      🎉 no goals
    -/
  suffices (RingHom.ker (AlgHom.comp (e : A →ₐ[R] B) f).toRingHom) = RingHom.ker f.toRingHom by
    rw [this]
    exact hf.2
  have hco : (AlgHom.comp (e : A →ₐ[R] B) f).toRingHom = RingHom.comp (e.toRingEquiv : A ≃+* B)
    f.toRingHom := by
    have h : (AlgHom.comp (e : A →ₐ[R] B) f).toRingHom =
      e.toAlgHom.toRingHom.comp f.toRingHom := rfl
    have h1 : ↑e.toRingEquiv = e.toAlgHom.toRingHom := rfl
    rw [h, h1]
  rw [RingHom.ker_eq_comap_bot, hco, ← Ideal.comap_comap, ← RingHom.ker_eq_comap_bot,
    RingHom.ker_coe_equiv (AlgEquiv.toRingEquiv e), RingHom.ker_eq_comap_bot]


/-- The ring of polynomials in finitely many variables is finitely presented. -/
protected instance mvPolynomial (ι : Type*) [Finite ι] :
    FinitePresentation R (MvPolynomial ι R) where
  out := by
    /-
      R : Type w₁
      A : Type w₂
      B : Type w₃
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      ι : Type u_1
      inst✝ : Finite ι
      ⊢ Exists fun n => Exists fun f => And (Function.Surjective ⇑f) (RingHom.ker f. …
    -/
    cases nonempty_fintype ι
    /-
      case intro
      R : Type w₁
      A : Type w₂
      B : Type w₃
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing B
      inst✝¹ : Algebra R B
      ι : Type u_1
      inst✝ : Finite ι
      val✝ : Fintype ι
      ⊢ Exists fun n => Exists fun f => And (Function.Surjective ⇑f) (RingHom.ker f. …
    -/
    let eqv := (MvPolynomial.renameEquiv R <| Fintype.equivFin ι).symm
    exact
      ⟨Fintype.card ι, eqv, eqv.surjective,
        ((RingHom.injective_iff_ker_eq_bot _).1 eqv.injective).symm ▸ Submodule.fg_bot⟩


/-- `R` is finitely presented as `R`-algebra. -/
instance self : FinitePresentation R R :=
  -- Porting note: replaced `PEmpty` with `Empty`
  equiv (MvPolynomial.isEmptyAlgEquiv R Empty)


/-- `R[X]` is finitely presented as `R`-algebra. -/
instance polynomial : FinitePresentation R R[X] :=
  -- Porting note: replaced `PUnit` with `Unit`
  letI := FinitePresentation.mvPolynomial R Unit
  equiv (MvPolynomial.pUnitAlgEquiv R)


/-- The quotient of a finitely presented algebra by a finitely generated ideal is finitely
presented. -/
protected theorem quotient {I : Ideal A} (h : I.FG) [FinitePresentation R A] :
    FinitePresentation R (A ⧸ I) where
  out := by
    /-
      R : Type w₁
      A : Type w₂
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal A
      h : I.FG
      inst✝ : Algebra.FinitePresentation R A
      ⊢ Exists fun n => Exists fun f => And (Function.Surjective ⇑f) (RingHom.ker f. …
    -/
    obtain ⟨n, f, hf⟩ := FinitePresentation.out (R := R) (A := A)
    /-
      case intro.intro
      R : Type w₁
      A : Type w₂
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      I : Ideal A
      h : I.FG
      inst✝ : Algebra.FinitePresentation R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      ⊢ Exists fun n => Exists fun f => And (Function.Surjective ⇑f) (RingHom.ker f. …
    -/
    refine ⟨n, (Ideal.Quotient.mkₐ R I).comp f, ?_, ?_⟩
      /-
        case intro.intro.refine_1
        R : Type w₁
        A : Type w₂
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        I : Ideal A
        h : I.FG
        inst✝ : Algebra.FinitePresentation R A
        n : Nat
        f : AlgHom R (MvPolynomial (Fin n) R) A
        hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
        ⊢ Function.Surjective ⇑((Ideal.Quotient.mkₐ R I).comp f)
      -/
    · exact (Ideal.Quotient.mkₐ_surjective R I).comp hf.1
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_2
        R : Type w₁
        A : Type w₂
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        I : Ideal A
        h : I.FG
        inst✝ : Algebra.FinitePresentation R A
        n : Nat
        f : AlgHom R (MvPolynomial (Fin n) R) A
        hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
        ⊢ (RingHom.ker ((Ideal.Quotient.mkₐ R I).comp f).toRingHom).FG
      -/
    · refine Ideal.fg_ker_comp _ _ hf.2 ?_ hf.1
      /-
        case intro.intro.refine_2
        R : Type w₁
        A : Type w₂
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        I : Ideal A
        h : I.FG
        inst✝ : Algebra.FinitePresentation R A
        n : Nat
        f : AlgHom R (MvPolynomial (Fin n) R) A
        hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
        ⊢ (RingHom.ker (Ideal.Quotient.mkₐ R I).toRingHom).FG
      -/
      simp [h]
      /-
        🎉 no goals
      -/


/-- If `f : A →ₐ[R] B` is surjective with finitely generated kernel and `A` is finitely presented,
then so is `B`. -/
theorem of_surjective {f : A →ₐ[R] B} (hf : Function.Surjective f) (hker : f.toRingHom.ker.FG)
    [FinitePresentation R A] : FinitePresentation R B :=
  letI : FinitePresentation R (A ⧸ RingHom.ker f) := FinitePresentation.quotient hker
  equiv (Ideal.quotientKerAlgEquivOfSurjective hf)


theorem iff :
    FinitePresentation R A ↔
      ∃ (n : _) (I : Ideal (MvPolynomial (Fin n) R)) (_ : (_ ⧸ I) ≃ₐ[R] A), I.FG := by
  /-
    R : Type w₁
    A : Type w₂
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.FinitePresentation R A) (Exists fun n => Exists fun I => Exists …
  -/
  constructor
    /-
      case mp
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ Algebra.FinitePresentation R A → Exists fun n => Exists fun I => Exists fun  …
    -/
  · rintro ⟨n, f, hf⟩
    /-
      case mp.mk.intro.intro
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      ⊢ Exists fun n => Exists fun I => Exists fun x => I.FG
    -/
    exact ⟨n, RingHom.ker f.toRingHom, Ideal.quotientKerAlgEquivOfSurjective hf.1, hf.2⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ (Exists fun n => Exists fun I => Exists fun x => I.FG) → Algebra.FinitePrese …
    -/
  · rintro ⟨n, I, e, hfg⟩
    /-
      case mpr.intro.intro.intro
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      n : Nat
      I : Ideal (MvPolynomial (Fin n) R)
      e : AlgEquiv R (HasQuotient.Quotient (MvPolynomial (Fin n) R) I) A
      hfg : I.FG
      ⊢ Algebra.FinitePresentation R A
    -/
    letI := (FinitePresentation.mvPolynomial R _).quotient hfg
    /-
      case mpr.intro.intro.intro
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      n : Nat
      I : Ideal (MvPolynomial (Fin n) R)
      e : AlgEquiv R (HasQuotient.Quotient (MvPolynomial (Fin n) R) I) A
      hfg : I.FG
      this : Algebra.FinitePresentation R (HasQuotient.Quotient (MvPolynomial (Fin n …
      ⊢ Algebra.FinitePresentation R A
    -/
    exact equiv e
    /-
      🎉 no goals
    -/


/-- An algebra is finitely presented if and only if it is a quotient of a polynomial ring whose
variables are indexed by a fintype by a finitely generated ideal. -/
theorem iff_quotient_mvPolynomial' :
    FinitePresentation R A ↔
      ∃ (ι : Type*) (_ : Fintype ι) (f : MvPolynomial ι R →ₐ[R] A),
        Surjective f ∧ f.toRingHom.ker.FG := by
  /-
    R : Type w₁
    A : Type w₂
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.FinitePresentation R A) (Exists fun ι => Exists fun x => Exists …
  -/
  constructor
    /-
      case mp
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ Algebra.FinitePresentation R A → Exists fun ι => Exists fun x => Exists fun  …
    -/
  · rintro ⟨n, f, hfs, hfk⟩
    /-
      case mp.mk.intro.intro.intro
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hfs : Function.Surjective ⇑f
      hfk : (RingHom.ker f.toRingHom).FG
      ⊢ Exists fun ι => Exists fun x => Exists fun f => And (Function.Surjective ⇑f) …
    -/
    set ulift_var := MvPolynomial.renameEquiv R Equiv.ulift
    refine
      ⟨ULift (Fin n), inferInstance, f.comp ulift_var.toAlgHom, hfs.comp ulift_var.surjective,
        Ideal.fg_ker_comp _ _ ?_ hfk ulift_var.surjective⟩
    /-
      case mp.mk.intro.intro.intro
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hfs : Function.Surjective ⇑f
      hfk : (RingHom.ker f.toRingHom).FG
      ulift_var : AlgEquiv R (MvPolynomial (ULift.{u_1, 0} (Fin n)) R) (MvPolynomial …
      ⊢ (RingHom.ker ↑↑ulift_var).FG
    -/
    erw [RingHom.ker_coe_equiv ulift_var.toRingEquiv]
    /-
      case mp.mk.intro.intro.intro
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      n : Nat
      f : AlgHom R (MvPolynomial (Fin n) R) A
      hfs : Function.Surjective ⇑f
      hfk : (RingHom.ker f.toRingHom).FG
      ulift_var : AlgEquiv R (MvPolynomial (ULift.{u_1, 0} (Fin n)) R) (MvPolynomial …
      ⊢ Bot.bot.FG
    -/
    exact Submodule.fg_bot
    /-
      🎉 no goals
    -/
    -- Porting note: was
    -- convert Submodule.fg_bot
    -- exact RingHom.ker_coe_equiv ulift_var.toRingEquiv
    /-
      case mpr
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ⊢ (Exists fun ι => Exists fun x => Exists fun f => And (Function.Surjective ⇑f …
    -/
  · rintro ⟨ι, hfintype, f, hf⟩
    /-
      case mpr.intro.intro.intro
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ι : Type u_1
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      ⊢ Algebra.FinitePresentation R A
    -/
    have equiv := MvPolynomial.renameEquiv R (Fintype.equivFin ι)
    /-
      case mpr.intro.intro.intro
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ι : Type u_1
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      equiv : AlgEquiv R (MvPolynomial ι R) (MvPolynomial (Fin (Fintype.card ι)) R)
      ⊢ Algebra.FinitePresentation R A
    -/
    use Fintype.card ι, f.comp equiv.symm, hf.1.comp (AlgEquiv.symm equiv).surjective
    /-
      case right
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ι : Type u_1
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      equiv : AlgEquiv R (MvPolynomial ι R) (MvPolynomial (Fin (Fintype.card ι)) R)
      ⊢ (RingHom.ker (f.comp ↑equiv.symm).toRingHom).FG
    -/
    refine Ideal.fg_ker_comp (S := MvPolynomial ι R) (A := A) _ f ?_ hf.2 equiv.symm.surjective
    /-
      case right
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ι : Type u_1
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      equiv : AlgEquiv R (MvPolynomial ι R) (MvPolynomial (Fin (Fintype.card ι)) R)
      ⊢ (RingHom.ker ↑↑equiv.symm).FG
    -/
    erw [RingHom.ker_coe_equiv equiv.symm.toRingEquiv]
    /-
      case right
      R : Type w₁
      A : Type w₂
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      ι : Type u_1
      hfintype : Fintype ι
      f : AlgHom R (MvPolynomial ι R) A
      hf : And (Function.Surjective ⇑f) (RingHom.ker f.toRingHom).FG
      equiv : AlgEquiv R (MvPolynomial ι R) (MvPolynomial (Fin (Fintype.card ι)) R)
      ⊢ Bot.bot.FG
    -/
    exact Submodule.fg_bot
    /-
      🎉 no goals
    -/
    -- Porting note: was
    -- convert Submodule.fg_bot
    -- exact RingHom.ker_coe_equiv equiv.symm.toRingEquiv


universe v in
-- Porting note: make universe level explicit to ensure `ι, ι'` has the same universe level
/-- If `A` is a finitely presented `R`-algebra, then `MvPolynomial (Fin n) A` is finitely presented
as `R`-algebra. -/
theorem mvPolynomial_of_finitePresentation [FinitePresentation.{w₁, w₂} R A]
    (ι : Type v) [Finite ι] :
    FinitePresentation.{w₁, max v w₂} R (MvPolynomial ι A) := by
  /-
    R : Type w₁
    A : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Algebra.FinitePresentation R A
    ι : Type v
    inst✝ : Finite ι
    ⊢ Algebra.FinitePresentation R (MvPolynomial ι A)
  -/
  have hfp : FinitePresentation.{w₁, w₂} R A := inferInstance
  /-
    R : Type w₁
    A : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Algebra.FinitePresentation R A
    ι : Type v
    inst✝ : Finite ι
    hfp : Algebra.FinitePresentation R A
    ⊢ Algebra.FinitePresentation R (MvPolynomial ι A)
  -/
  rw [iff_quotient_mvPolynomial'] at hfp ⊢
  classical
  -- Porting note: use the same universe level
  obtain ⟨(ι' : Type v), _, f, hf_surj, hf_ker⟩ := hfp
  let g := (MvPolynomial.mapAlgHom f).comp (MvPolynomial.sumAlgEquiv R ι ι').toAlgHom
  cases nonempty_fintype (ι ⊕ ι')
  refine
    ⟨ι ⊕ ι', by infer_instance, g,
      (MvPolynomial.map_surjective f.toRingHom hf_surj).comp (AlgEquiv.surjective _),
      Ideal.fg_ker_comp _ _ ?_ ?_ (AlgEquiv.surjective _)⟩
  · erw [RingHom.ker_coe_equiv (MvPolynomial.sumAlgEquiv R ι ι').toRingEquiv]
    exact Submodule.fg_bot
    -- Porting note: was
    -- convert Submodule.fg_bot
    -- exact RingHom.ker_coe_equiv (MvPolynomial.sumAlgEquiv R ι ι').toRingEquiv
  · rw [AlgHom.toRingHom_eq_coe, MvPolynomial.mapAlgHom_coe_ringHom, MvPolynomial.ker_map]
    exact hf_ker.map MvPolynomial.C


/-- If `A` is an `R`-algebra and `S` is an `A`-algebra, both finitely presented, then `S` is
  finitely presented as `R`-algebra. -/
theorem trans [Algebra A B] [IsScalarTower R A B] [FinitePresentation R A]
    [FinitePresentation A B] : FinitePresentation R B := by
  /-
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : Algebra.FinitePresentation R A
    inst✝ : Algebra.FinitePresentation A B
    ⊢ Algebra.FinitePresentation R B
  -/
  have hfpB : FinitePresentation A B := inferInstance
  /-
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : Algebra.FinitePresentation R A
    inst✝ hfpB : Algebra.FinitePresentation A B
    ⊢ Algebra.FinitePresentation R B
  -/
  obtain ⟨n, I, e, hfg⟩ := iff.1 hfpB
  letI : FinitePresentation R (MvPolynomial (Fin n) A ⧸ I) :=
    (mvPolynomial_of_finitePresentation _).quotient hfg
  /-
    case intro.intro.intro
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing A
    inst✝⁶ : Algebra R A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : Algebra.FinitePresentation R A
    inst✝ hfpB : Algebra.FinitePresentation A B
    n : Nat
    I : Ideal (MvPolynomial (Fin n) A)
    e : AlgEquiv A (HasQuotient.Quotient (MvPolynomial (Fin n) A) I) B
    hfg : I.FG
    this : Algebra.FinitePresentation R (HasQuotient.Quotient (MvPolynomial (Fin n …
    ⊢ Algebra.FinitePresentation R B
  -/
  exact equiv (e.restrictScalars R)
  /-
    🎉 no goals
  -/


theorem of_restrict_scalars_finitePresentation [Algebra A B] [IsScalarTower R A B]
    [FinitePresentation.{w₁, w₃} R B] [FiniteType R A] :
    FinitePresentation.{w₂, w₃} A B := by
  classical
  obtain ⟨n, f, hf, s, hs⟩ := FinitePresentation.out (R := R) (A := B)
  letI RX := MvPolynomial (Fin n) R
  letI AX := MvPolynomial (Fin n) A
  refine ⟨n, MvPolynomial.aeval (f ∘ X), ?_, ?_⟩
  · rw [← AlgHom.range_eq_top, ← Algebra.adjoin_range_eq_range_aeval,
      Set.range_comp f MvPolynomial.X, eq_top_iff, ← @adjoin_adjoin_of_tower R A B,
      adjoin_image, adjoin_range_X, Algebra.map_top, (AlgHom.range_eq_top _).mpr hf]
    exact fun {x} => subset_adjoin ⟨⟩
  · obtain ⟨t, ht⟩ := FiniteType.out (R := R) (A := A)
    have := fun i : t => hf (algebraMap A B i)
    choose t' ht' using this
    have ht'' : Algebra.adjoin R (algebraMap A AX '' t ∪ Set.range (X : _ → AX)) = ⊤ := by
      rw [adjoin_union_eq_adjoin_adjoin, ← Subalgebra.restrictScalars_top R (A := AX)
        (S := { x // x ∈ adjoin R ((algebraMap A AX) '' t) })]
      refine congrArg (Subalgebra.restrictScalars R) ?_
      rw [adjoin_algebraMap, ht]
      apply Subalgebra.restrictScalars_injective R
      rw [← adjoin_restrictScalars, adjoin_range_X, Subalgebra.restrictScalars_top,
        Subalgebra.restrictScalars_top]
    letI g : t → AX := fun x => MvPolynomial.C (x : A) - map (algebraMap R A) (t' x)
    refine ⟨s.image (map (algebraMap R A)) ∪ t.attach.image g, ?_⟩
    rw [Finset.coe_union, Finset.coe_image, Finset.coe_image, Finset.attach_eq_univ,
      Finset.coe_univ, Set.image_univ]
    let s₀ := (MvPolynomial.map (algebraMap R A)) '' s ∪ Set.range g
    let I := RingHom.ker (MvPolynomial.aeval (R := A) (f ∘ MvPolynomial.X))
    change Ideal.span s₀ = I
    have leI : Ideal.span ((MvPolynomial.map (algebraMap R A)) '' s ∪ Set.range g) ≤
      RingHom.ker (MvPolynomial.aeval (R := A) (f ∘ MvPolynomial.X)) := by
      rw [Ideal.span_le]
      rintro _ (⟨x, hx, rfl⟩ | ⟨⟨x, hx⟩, rfl⟩) <;>
      erw [RingHom.mem_ker]
      · rw [MvPolynomial.aeval_map_algebraMap (R := R) (A := A), ← aeval_unique]
        have := Ideal.subset_span hx
        rwa [hs] at this
      · rw [map_sub, MvPolynomial.aeval_map_algebraMap, ← aeval_unique,
          MvPolynomial.aeval_C, ht', Subtype.coe_mk, sub_self]
    apply leI.antisymm
    intro x hx
    rw [RingHom.mem_ker] at hx
    let s₀ := (MvPolynomial.map (algebraMap R A)) '' ↑s ∪ Set.range g
    change x ∈ Ideal.span s₀
    have : x ∈ (MvPolynomial.map (algebraMap R A) : _ →+* AX).range.toAddSubmonoid ⊔
      (Ideal.span s₀).toAddSubmonoid := by
      have : x ∈ (⊤ : Subalgebra R AX) := trivial
      rw [← ht''] at this
      refine adjoin_induction ?_ ?_ ?_ ?_ this
      · rintro _ (⟨x, hx, rfl⟩ | ⟨i, rfl⟩)
        · rw [MvPolynomial.algebraMap_eq, ← sub_add_cancel (MvPolynomial.C x)
            (map (algebraMap R A) (t' ⟨x, hx⟩)), add_comm]
          apply AddSubmonoid.add_mem_sup
          · exact Set.mem_range_self _
          · apply Ideal.subset_span
            apply Set.mem_union_right
            exact Set.mem_range_self _
        · apply AddSubmonoid.mem_sup_left
          exact ⟨X i, map_X _ _⟩
      · intro r
        apply AddSubmonoid.mem_sup_left
        exact ⟨C r, map_C _ _⟩
      · intro _ _ _ _ h₁ h₂
        exact add_mem h₁ h₂
      · intro x₁ x₂ _ _ h₁ h₂
        obtain ⟨_, ⟨p₁, rfl⟩, q₁, hq₁, rfl⟩ := AddSubmonoid.mem_sup.mp h₁
        obtain ⟨_, ⟨p₂, rfl⟩, q₂, hq₂, rfl⟩ := AddSubmonoid.mem_sup.mp h₂
        rw [add_mul, mul_add, add_assoc, ← map_mul]
        apply AddSubmonoid.add_mem_sup
        · exact Set.mem_range_self _
        · refine add_mem (Ideal.mul_mem_left _ _ hq₂) (Ideal.mul_mem_right _ _ hq₁)
    obtain ⟨_, ⟨p, rfl⟩, q, hq, rfl⟩ := AddSubmonoid.mem_sup.mp this
    rw [map_add, aeval_map_algebraMap, ← aeval_unique, show MvPolynomial.aeval (f ∘ X) q = 0
      from leI hq, add_zero] at hx
    suffices Ideal.span (s : Set RX) ≤ (Ideal.span s₀).comap (MvPolynomial.map <| algebraMap R A) by
      refine add_mem ?_ hq
      rw [hs] at this
      exact this hx
    rw [Ideal.span_le]
    intro x hx
    apply Ideal.subset_span
    apply Set.mem_union_left
    exact Set.mem_image_of_mem _ hx


/-- This is used to prove the strictly stronger `ker_fg_of_surjective`. Use it instead. -/
theorem ker_fg_of_mvPolynomial {n : ℕ} (f : MvPolynomial (Fin n) R →ₐ[R] A)
    (hf : Function.Surjective f) [FinitePresentation R A] : f.toRingHom.ker.FG := by
  classical
    obtain ⟨m, f', hf', s, hs⟩ := FinitePresentation.out (R := R) (A := A)
    let RXn := MvPolynomial (Fin n) R
    let RXm := MvPolynomial (Fin m) R
    have := fun i : Fin n => hf' (f <| X i)
    choose g hg using this
    have := fun i : Fin m => hf (f' <| X i)
    choose h hh using this
    let aeval_h : RXm →ₐ[R] RXn := aeval h
    let g' : Fin n → RXn := fun i => X i - aeval_h (g i)
    refine ⟨Finset.univ.image g' ∪ s.image aeval_h, ?_⟩
    simp only [Finset.coe_image, Finset.coe_union, Finset.coe_univ, Set.image_univ]
    have hh' : ∀ x, f (aeval_h x) = f' x := by
      intro x
      rw [← f.coe_toRingHom, map_aeval]
      simp_rw [AlgHom.coe_toRingHom, hh]
      rw [AlgHom.comp_algebraMap, ← aeval_eq_eval₂Hom,
        -- Porting note: added line below
        ← funext fun i => Function.comp_apply (f := ↑f') (g := MvPolynomial.X),
        ← aeval_unique]
    let s' := Set.range g' ∪ aeval_h '' s
    have leI : Ideal.span s' ≤ RingHom.ker f.toRingHom := by
      rw [Ideal.span_le]
      rintro _ (⟨i, rfl⟩ | ⟨x, hx, rfl⟩)
      · change f (g' i) = 0
        rw [map_sub, ← hg, hh', sub_self]
      · change f (aeval_h x) = 0
        rw [hh']
        change x ∈ RingHom.ker f'.toRingHom
        rw [← hs]
        exact Ideal.subset_span hx
    apply leI.antisymm
    intro x hx
    have : x ∈ aeval_h.range.toAddSubmonoid ⊔ (Ideal.span s').toAddSubmonoid := by
      have : x ∈ adjoin R (Set.range X : Set RXn) := by
        rw [adjoin_range_X]
        trivial
      refine adjoin_induction ?_ ?_ ?_ ?_ this
      · rintro _ ⟨i, rfl⟩
        rw [← sub_add_cancel (X i) (aeval h (g i)), add_comm]
        apply AddSubmonoid.add_mem_sup
        · exact Set.mem_range_self _
        · apply Submodule.subset_span
          apply Set.mem_union_left
          exact Set.mem_range_self _
      · intro r
        apply AddSubmonoid.mem_sup_left
        exact ⟨C r, aeval_C _ _⟩
      · intro _ _ _ _ h₁ h₂
        exact add_mem h₁ h₂
      · intro p₁ p₂ _ _ h₁ h₂
        obtain ⟨_, ⟨x₁, rfl⟩, y₁, hy₁, rfl⟩ := AddSubmonoid.mem_sup.mp h₁
        obtain ⟨_, ⟨x₂, rfl⟩, y₂, hy₂, rfl⟩ := AddSubmonoid.mem_sup.mp h₂
        rw [mul_add, add_mul, add_assoc, ← map_mul]
        apply AddSubmonoid.add_mem_sup
        · exact Set.mem_range_self _
        · exact add_mem (Ideal.mul_mem_right _ _ hy₁) (Ideal.mul_mem_left _ _ hy₂)
    obtain ⟨_, ⟨x, rfl⟩, y, hy, rfl⟩ := AddSubmonoid.mem_sup.mp this
    refine add_mem ?_ hy
    simp only [RXn, RingHom.mem_ker, AlgHom.toRingHom_eq_coe, AlgHom.coe_toRingHom, map_add,
      show f y = 0 from leI hy, add_zero, hh'] at hx
    suffices Ideal.span (s : Set RXm) ≤ (Ideal.span s').comap aeval_h by
      apply this
      rwa [hs]
    rw [Ideal.span_le]
    intro x hx
    apply Submodule.subset_span
    apply Set.mem_union_right
    exact Set.mem_image_of_mem _ hx


/-- If `f : A →ₐ[R] B` is a surjection between finitely-presented `R`-algebras, then the kernel of
`f` is finitely generated. -/
theorem ker_fG_of_surjective (f : A →ₐ[R] B) (hf : Function.Surjective f)
    [FinitePresentation R A] [FinitePresentation R B] : f.toRingHom.ker.FG := by
  /-
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    inst✝¹ : Algebra.FinitePresentation R A
    inst✝ : Algebra.FinitePresentation R B
    ⊢ (RingHom.ker f.toRingHom).FG
  -/
  obtain ⟨n, g, hg, _⟩ := FinitePresentation.out (R := R) (A := A)
  /-
    case intro.intro.intro
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    inst✝¹ : Algebra.FinitePresentation R A
    inst✝ : Algebra.FinitePresentation R B
    n : Nat
    g : AlgHom R (MvPolynomial (Fin n) R) A
    hg : Function.Surjective ⇑g
    right✝ : (RingHom.ker g.toRingHom).FG
    ⊢ (RingHom.ker f.toRingHom).FG
  -/
  convert (ker_fg_of_mvPolynomial (f.comp g) (hf.comp hg)).map g.toRingHom
  /-
    case h.e'_3
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    inst✝¹ : Algebra.FinitePresentation R A
    inst✝ : Algebra.FinitePresentation R B
    n : Nat
    g : AlgHom R (MvPolynomial (Fin n) R) A
    hg : Function.Surjective ⇑g
    right✝ : (RingHom.ker g.toRingHom).FG
    ⊢ Eq (RingHom.ker f.toRingHom) (Ideal.map g.toRingHom (RingHom.ker (f.comp g). …
  -/
  simp_rw [RingHom.ker_eq_comap_bot, AlgHom.toRingHom_eq_coe, AlgHom.comp_toRingHom]
  /-
    case h.e'_3
    R : Type w₁
    A : Type w₂
    B : Type w₃
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    inst✝¹ : Algebra.FinitePresentation R A
    inst✝ : Algebra.FinitePresentation R B
    n : Nat
    g : AlgHom R (MvPolynomial (Fin n) R) A
    hg : Function.Surjective ⇑g
    right✝ : (RingHom.ker g.toRingHom).FG
    ⊢ Eq (Ideal.comap (↑f) Bot.bot) (Ideal.map (↑g) (Ideal.comap ((↑f).comp ↑g) Bo …
  -/
  rw [← Ideal.comap_comap, Ideal.map_comap_of_surjective (g : MvPolynomial (Fin n) R →+* A) hg]
  /-
    🎉 no goals
  -/


/-- A ring morphism `A →+* B` is of `RingHom.FinitePresentation` if `B` is finitely presented as
`A`-algebra. -/
@[algebraize]
def FinitePresentation (f : A →+* B) : Prop :=
  @Algebra.FinitePresentation A B _ _ f.toAlgebra


theorem of_finitePresentation {f : A →+* B} (hf : f.FinitePresentation) : f.FiniteType :=
  @Algebra.FiniteType.of_finitePresentation A B _ _ f.toAlgebra hf


theorem id : FinitePresentation (RingHom.id A) :=
  Algebra.FinitePresentation.self A


theorem comp_surjective {f : A →+* B} {g : B →+* C} (hf : f.FinitePresentation) (hg : Surjective g)
    (hker : g.ker.FG) : (g.comp f).FinitePresentation := by
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    f : RingHom A B
    g : RingHom B C
    hf : f.FinitePresentation
    hg : Function.Surjective ⇑g
    hker : (RingHom.ker g).FG
    ⊢ (g.comp f).FinitePresentation
  -/
  algebraize [f, g.comp f]
  exact Algebra.FinitePresentation.of_surjective
    (f :=
      { g with
        toFun := g
        commutes' := fun _ => rfl })
    hg hker


theorem of_surjective (f : A →+* B) (hf : Surjective f) (hker : f.ker.FG) :
    f.FinitePresentation := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    f : RingHom A B
    hf : Function.Surjective ⇑f
    hker : (RingHom.ker f).FG
    ⊢ f.FinitePresentation
  -/
  rw [← f.comp_id]
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommRing A
    inst✝ : CommRing B
    f : RingHom A B
    hf : Function.Surjective ⇑f
    hker : (RingHom.ker f).FG
    ⊢ (f.comp (RingHom.id A)).FinitePresentation
  -/
  exact (id A).comp_surjective hf hker
  /-
    🎉 no goals
  -/


theorem of_finiteType [IsNoetherianRing A] {f : A →+* B} : f.FiniteType ↔ f.FinitePresentation :=
  @Algebra.FinitePresentation.of_finiteType A B _ _ f.toAlgebra _


theorem comp {g : B →+* C} {f : A →+* B} (hg : g.FinitePresentation) (hf : f.FinitePresentation) :
    (g.comp f).FinitePresentation := by
  -- Porting note: specify `Algebra` instances to get `SMul`
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    g : RingHom B C
    f : RingHom A B
    hg : g.FinitePresentation
    hf : f.FinitePresentation
    ⊢ (g.comp f).FinitePresentation
  -/
  algebraize [f, g, g.comp f]
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    g : RingHom B C
    f : RingHom A B
    hg : g.FinitePresentation
    hf : f.FinitePresentation
    algInst✝² : Algebra A B := f.toAlgebra
    algInst✝¹ : Algebra B C := g.toAlgebra
    algInst✝ : Algebra A C := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' (Eq. …
    algebraizeInst✝¹ : Algebra.FinitePresentation B C
    algebraizeInst✝ : Algebra.FinitePresentation A B
    ⊢ (g.comp f).FinitePresentation
  -/
  exact Algebra.FinitePresentation.trans A B C
  /-
    🎉 no goals
  -/


theorem of_comp_finiteType (f : A →+* B) {g : B →+* C} (hg : (g.comp f).FinitePresentation)
    (hf : f.FiniteType) : g.FinitePresentation := by
  -- Porting note: need to specify some instances
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    f : RingHom A B
    g : RingHom B C
    hg : (g.comp f).FinitePresentation
    hf : f.FiniteType
    ⊢ g.FinitePresentation
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
    hg : (g.comp f).FinitePresentation
    hf : f.FiniteType
    algInst✝² : Algebra A B := f.toAlgebra
    algInst✝¹ : Algebra B C := g.toAlgebra
    algInst✝ : Algebra A C := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' (Eq. …
    algebraizeInst✝¹ : Algebra.FinitePresentation A C
    algebraizeInst✝ : Algebra.FiniteType A B
    ⊢ g.FinitePresentation
  -/
  exact Algebra.FinitePresentation.of_restrict_scalars_finitePresentation A B C
  /-
    🎉 no goals
  -/


/-- An algebra morphism `A →ₐ[R] B` is of `AlgHom.FinitePresentation` if it is of finite
presentation as ring morphism. In other words, if `B` is finitely presented as `A`-algebra. -/
def FinitePresentation (f : A →ₐ[R] B) : Prop :=
  f.toRingHom.FinitePresentation


theorem of_finitePresentation {f : A →ₐ[R] B} (hf : f.FinitePresentation) : f.FiniteType :=
  RingHom.FiniteType.of_finitePresentation hf


theorem id : FinitePresentation (AlgHom.id R A) :=
  RingHom.FinitePresentation.id A


theorem comp {g : B →ₐ[R] C} {f : A →ₐ[R] B} (hg : g.FinitePresentation)
    (hf : f.FinitePresentation) : (g.comp f).FinitePresentation :=
  RingHom.FinitePresentation.comp hg hf


theorem comp_surjective {f : A →ₐ[R] B} {g : B →ₐ[R] C} (hf : f.FinitePresentation)
    (hg : Surjective g) (hker : g.toRingHom.ker.FG) : (g.comp f).FinitePresentation :=
  RingHom.FinitePresentation.comp_surjective hf hg hker


theorem of_surjective (f : A →ₐ[R] B) (hf : Surjective f) (hker : f.toRingHom.ker.FG) :
    f.FinitePresentation := by
  -- Porting note: added `convert`
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    hker : (RingHom.ker f.toRingHom).FG
    ⊢ f.FinitePresentation
  -/
  convert RingHom.FinitePresentation.of_surjective f hf hker
  /-
    🎉 no goals
  -/


theorem of_finiteType [IsNoetherianRing A] {f : A →ₐ[R] B} : f.FiniteType ↔ f.FinitePresentation :=
  RingHom.FinitePresentation.of_finiteType


nonrec theorem of_comp_finiteType (f : A →ₐ[R] B) {g : B →ₐ[R] C}
    (h : (g.comp f).FinitePresentation) (h' : f.FiniteType) : g.FinitePresentation :=
  h.of_comp_finiteType _ h'


