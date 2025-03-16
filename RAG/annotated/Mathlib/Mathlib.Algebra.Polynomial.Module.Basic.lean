/-- The `R[X]`-module `M[X]` for an `R`-module `M`.
This is isomorphic (as an `R`-module) to `M[X]` when `M` is a ring.

We require all the module instances `Module S (PolynomialModule R M)` to factor through `R` except
`Module R[X] (PolynomialModule R M)`.
In this constraint, we have the following instances for example :
- `R` acts on `PolynomialModule R R[X]`
- `R[X]` acts on `PolynomialModule R R[X]` as `R[Y]` acting on `R[X][Y]`
- `R` acts on `PolynomialModule R[X] R[X]`
- `R[X]` acts on `PolynomialModule R[X] R[X]` as `R[X]` acting on `R[X][Y]`
- `R[X][X]` acts on `PolynomialModule R[X] R[X]` as `R[X][Y]` acting on itself

This is also the reason why `R` is included in the alias, or else there will be two different
instances of `Module R[X] (PolynomialModule R[X])`.

See https://leanprover.zulipchat.com/#narrow/stream/144837-PR-reviews/topic/.2315065.20polynomial.20modules
for the full discussion.
-/
@[nolint unusedArguments]
def PolynomialModule (R M : Type*) [CommRing R] [AddCommGroup M] [Module R M] := ℕ →₀ M


noncomputable instance : Inhabited (PolynomialModule R M) := Finsupp.instInhabited

noncomputable instance : AddCommGroup (PolynomialModule R M) := Finsupp.instAddCommGroup


/-- This is required to have the `IsScalarTower S R M` instance to avoid diamonds. -/
@[nolint unusedArguments]
noncomputable instance : Module S (PolynomialModule R M) :=
  Finsupp.module ℕ M


instance instFunLike : FunLike (PolynomialModule R M) ℕ M :=
  Finsupp.instFunLike


instance : CoeFun (PolynomialModule R M) fun _ => ℕ → M :=
  inferInstanceAs <| CoeFun (_ →₀ _) _


theorem zero_apply (i : ℕ) : (0 : PolynomialModule R M) i = 0 :=
  Finsupp.zero_apply


theorem add_apply (g₁ g₂ : PolynomialModule R M) (a : ℕ) : (g₁ + g₂) a = g₁ a + g₂ a :=
  Finsupp.add_apply g₁ g₂ a


/-- The monomial `m * x ^ i`. This is defeq to `Finsupp.singleAddHom`, and is redefined here
so that it has the desired type signature. -/
noncomputable def single (i : ℕ) : M →+ PolynomialModule R M :=
  Finsupp.singleAddHom i


theorem single_apply (i : ℕ) (m : M) (n : ℕ) : single R i m n = ite (i = n) m 0 :=
  Finsupp.single_apply


/-- `PolynomialModule.single` as a linear map. -/
noncomputable def lsingle (i : ℕ) : M →ₗ[R] PolynomialModule R M :=
  Finsupp.lsingle i


theorem lsingle_apply (i : ℕ) (m : M) (n : ℕ) : lsingle R i m n = ite (i = n) m 0 :=
  Finsupp.single_apply


theorem single_smul (i : ℕ) (r : R) (m : M) : single R i (r • m) = r • single R i m :=
  (lsingle R i).map_smul r m


theorem induction_linear {P : PolynomialModule R M → Prop} (f : PolynomialModule R M) (h0 : P 0)
    (hadd : ∀ f g, P f → P g → P (f + g)) (hsingle : ∀ a b, P (single R a b)) : P f :=
  Finsupp.induction_linear f h0 hadd hsingle


noncomputable instance polynomialModule : Module R[X] (PolynomialModule R M) :=
  inferInstanceAs (Module R[X] (Module.AEval' (Finsupp.lmapDomain M R Nat.succ)))


lemma smul_def (f : R[X]) (m : PolynomialModule R M) :
    f • m = aeval (Finsupp.lmapDomain M R Nat.succ) f m := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Polynomial R
    m : PolynomialModule R M
    ⊢ Eq (HSMul.hSMul f m) (((Polynomial.aeval (Finsupp.lmapDomain M R Nat.succ))  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance (M : Type u) [AddCommGroup M] [Module R M] [Module S M] [IsScalarTower S R M] :
    IsScalarTower S R (PolynomialModule R M) :=
  Finsupp.isScalarTower _ _


instance isScalarTower' (M : Type u) [AddCommGroup M] [Module R M] [Module S M]
    [IsScalarTower S R M] : IsScalarTower S R[X] (PolynomialModule R M) := by
  haveI : IsScalarTower R R[X] (PolynomialModule R M) :=
    inferInstanceAs <| IsScalarTower R R[X] <| Module.AEval' <| Finsupp.lmapDomain M R Nat.succ
  /-
    R : Type u_1
    M✝ : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M✝
    inst✝⁸ : Module R M✝
    I : Ideal R
    S : Type u_3
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra S R
    inst✝⁵ : Module S M✝
    inst✝⁴ : IsScalarTower S R M✝
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower S R M
    this : IsScalarTower R (Polynomial R) (PolynomialModule R M)
    ⊢ IsScalarTower S (Polynomial R) (PolynomialModule R M)
  -/
  constructor
  /-
    case smul_assoc
    R : Type u_1
    M✝ : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M✝
    inst✝⁸ : Module R M✝
    I : Ideal R
    S : Type u_3
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra S R
    inst✝⁵ : Module S M✝
    inst✝⁴ : IsScalarTower S R M✝
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower S R M
    this : IsScalarTower R (Polynomial R) (PolynomialModule R M)
    ⊢ ∀ (x : S) (y : Polynomial R) (z : PolynomialModule R M), Eq (HSMul.hSMul (HS …
  -/
  intro x y z
  /-
    case smul_assoc
    R : Type u_1
    M✝ : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M✝
    inst✝⁸ : Module R M✝
    I : Ideal R
    S : Type u_3
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra S R
    inst✝⁵ : Module S M✝
    inst✝⁴ : IsScalarTower S R M✝
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower S R M
    this : IsScalarTower R (Polynomial R) (PolynomialModule R M)
    x : S
    y : Polynomial R
    z : PolynomialModule R M
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
  -/
  rw [← @IsScalarTower.algebraMap_smul S R, ← @IsScalarTower.algebraMap_smul S R, smul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem monomial_smul_single (i : ℕ) (r : R) (j : ℕ) (m : M) :
    monomial i r • single R j m = single R (i + j) (r • m) := by
  simp only [LinearMap.mul_apply, Polynomial.aeval_monomial, LinearMap.pow_apply,
    Module.algebraMap_end_apply, smul_def]
  induction i generalizing r j m with
  | zero =>
    rw [Function.iterate_zero, zero_add]
    exact Finsupp.smul_single r j m
  | succ n hn =>
    rw [Function.iterate_succ, Function.comp_apply, add_assoc, ← hn]
    congr 2
    rw [Nat.one_add]
    exact Finsupp.mapDomain_single


@[simp]
theorem monomial_smul_apply (i : ℕ) (r : R) (g : PolynomialModule R M) (n : ℕ) :
    (monomial i r • g) n = ite (i ≤ n) (r • g (n - i)) 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    i : Nat
    r : R
    g : PolynomialModule R M
    n : Nat
    ⊢ Eq ((HSMul.hSMul ((Polynomial.monomial i) r) g) n) (ite (LE.le i n) (HSMul.h …
  -/
  induction' g using PolynomialModule.induction_linear with p q hp hq
    /-
      case h0
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      r : R
      n : Nat
      ⊢ Eq ((HSMul.hSMul ((Polynomial.monomial i) r) 0) n) (ite (LE.le i n) (HSMul.h …
    -/
  · simp only [smul_zero, zero_apply, ite_self]
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      r : R
      n : Nat
      p q : PolynomialModule R M
      hp : Eq ((HSMul.hSMul ((Polynomial.monomial i) r) p) n) (ite (LE.le i n) (HSMu …
      hq : Eq ((HSMul.hSMul ((Polynomial.monomial i) r) q) n) (ite (LE.le i n) (HSMu …
      ⊢ Eq ((HSMul.hSMul ((Polynomial.monomial i) r) (HAdd.hAdd p q)) n) (ite (LE.le …
    -/
  · simp only [smul_add, add_apply, hp, hq]
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      r : R
      n : Nat
      p q : PolynomialModule R M
      hp : Eq ((HSMul.hSMul ((Polynomial.monomial i) r) p) n) (ite (LE.le i n) (HSMu …
      hq : Eq ((HSMul.hSMul ((Polynomial.monomial i) r) q) n) (ite (LE.le i n) (HSMu …
      ⊢ Eq (HAdd.hAdd (ite (LE.le i n) (HSMul.hSMul r (p (HSub.hSub n i))) 0) (ite ( …
    -/
    split_ifs
    /-
      case pos
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      r : R
      n : Nat
      p q : PolynomialModule R M
      hp : Eq ((HSMul.hSMul ((Polynomial.monomial i) r) p) n) (ite (LE.le i n) (HSMu …
      hq : Eq ((HSMul.hSMul ((Polynomial.monomial i) r) q) n) (ite (LE.le i n) (HSMu …
      h✝ : LE.le i n
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul r (p (HSub.hSub n i))) (HSMul.hSMul r (q (HSub.hS …
    -/
    exacts [rfl, zero_add 0]
    /-
      🎉 no goals
    -/
    /-
      case hsingle
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      r : R
      n a✝ : Nat
      b✝ : M
      ⊢ Eq ((HSMul.hSMul ((Polynomial.monomial i) r) ((PolynomialModule.single R a✝) …
    -/
  · rw [monomial_smul_single, single_apply, single_apply, smul_ite, smul_zero, ← ite_and]
    /-
      case hsingle
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      r : R
      n a✝ : Nat
      b✝ : M
      ⊢ Eq (ite (Eq (HAdd.hAdd i a✝) n) (HSMul.hSMul r b✝) 0) (ite (And (LE.le i n)  …
    -/
    congr
    /-
      case hsingle.e_c
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      r : R
      n a✝ : Nat
      b✝ : M
      ⊢ Eq (Eq (HAdd.hAdd i a✝) n) (And (LE.le i n) (Eq a✝ (HSub.hSub n i)))
    -/
    rw [eq_iff_iff]
    /-
      case hsingle.e_c
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      r : R
      n a✝ : Nat
      b✝ : M
      ⊢ Iff (Eq (HAdd.hAdd i a✝) n) (And (LE.le i n) (Eq a✝ (HSub.hSub n i)))
    -/
    constructor
      /-
        case hsingle.e_c.mp
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        i : Nat
        r : R
        n a✝ : Nat
        b✝ : M
        ⊢ Eq (HAdd.hAdd i a✝) n → And (LE.le i n) (Eq a✝ (HSub.hSub n i))
      -/
    · rintro rfl
      /-
        case hsingle.e_c.mp
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        i : Nat
        r : R
        a✝ : Nat
        b✝ : M
        ⊢ And (LE.le i (HAdd.hAdd i a✝)) (Eq a✝ (HSub.hSub (HAdd.hAdd i a✝) i))
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case hsingle.e_c.mpr
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        i : Nat
        r : R
        n a✝ : Nat
        b✝ : M
        ⊢ And (LE.le i n) (Eq a✝ (HSub.hSub n i)) → Eq (HAdd.hAdd i a✝) n
      -/
    · rintro ⟨e, rfl⟩
      /-
        case hsingle.e_c.mpr.intro
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        i : Nat
        r : R
        n : Nat
        b✝ : M
        e : LE.le i n
        ⊢ Eq (HAdd.hAdd i (HSub.hSub n i)) n
      -/
      rw [add_comm, tsub_add_cancel_of_le e]
      /-
        🎉 no goals
      -/


@[simp]
theorem smul_single_apply (i : ℕ) (f : R[X]) (m : M) (n : ℕ) :
    (f • single R i m) n = ite (i ≤ n) (f.coeff (n - i) • m) 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    i : Nat
    f : Polynomial R
    m : M
    n : Nat
    ⊢ Eq ((HSMul.hSMul f ((PolynomialModule.single R i) m)) n) (ite (LE.le i n) (H …
  -/
  induction' f using Polynomial.induction_on' with p q hp hq
    /-
      case h_add
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      m : M
      n : Nat
      p q : Polynomial R
      hp : Eq ((HSMul.hSMul p ((PolynomialModule.single R i) m)) n) (ite (LE.le i n) …
      hq : Eq ((HSMul.hSMul q ((PolynomialModule.single R i) m)) n) (ite (LE.le i n) …
      ⊢ Eq ((HSMul.hSMul (HAdd.hAdd p q) ((PolynomialModule.single R i) m)) n) (ite  …
    -/
  · rw [add_smul, Finsupp.add_apply, hp, hq, coeff_add, add_smul]
    /-
      case h_add
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      m : M
      n : Nat
      p q : Polynomial R
      hp : Eq ((HSMul.hSMul p ((PolynomialModule.single R i) m)) n) (ite (LE.le i n) …
      hq : Eq ((HSMul.hSMul q ((PolynomialModule.single R i) m)) n) (ite (LE.le i n) …
      ⊢ Eq (HAdd.hAdd (ite (LE.le i n) (HSMul.hSMul (p.coeff (HSub.hSub n i)) m) 0)  …
    -/
    split_ifs
    /-
      case pos
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      m : M
      n : Nat
      p q : Polynomial R
      hp : Eq ((HSMul.hSMul p ((PolynomialModule.single R i) m)) n) (ite (LE.le i n) …
      hq : Eq ((HSMul.hSMul q ((PolynomialModule.single R i) m)) n) (ite (LE.le i n) …
      h✝ : LE.le i n
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (p.coeff (HSub.hSub n i)) m) (HSMul.hSMul (q.coef …
    -/
    exacts [rfl, zero_add 0]
    /-
      🎉 no goals
    -/
    /-
      case h_monomial
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      m : M
      n n✝ : Nat
      a✝ : R
      ⊢ Eq ((HSMul.hSMul ((Polynomial.monomial n✝) a✝) ((PolynomialModule.single R i …
    -/
  · rw [monomial_smul_single, single_apply, coeff_monomial, ite_smul, zero_smul]
    /-
      case h_monomial
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      i : Nat
      m : M
      n n✝ : Nat
      a✝ : R
      ⊢ Eq (ite (Eq (HAdd.hAdd n✝ i) n) (HSMul.hSMul a✝ m) 0) (ite (LE.le i n) (ite  …
    -/
    by_cases h : i ≤ n
      /-
        case pos
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        i : Nat
        m : M
        n n✝ : Nat
        a✝ : R
        h : LE.le i n
        ⊢ Eq (ite (Eq (HAdd.hAdd n✝ i) n) (HSMul.hSMul a✝ m) 0) (ite (LE.le i n) (ite  …
      -/
    · simp_rw [eq_tsub_iff_add_eq_of_le h, if_pos h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        i : Nat
        m : M
        n n✝ : Nat
        a✝ : R
        h : Not (LE.le i n)
        ⊢ Eq (ite (Eq (HAdd.hAdd n✝ i) n) (HSMul.hSMul a✝ m) 0) (ite (LE.le i n) (ite  …
      -/
    · rw [if_neg h, if_neg]
      /-
        case neg.hnc
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        i : Nat
        m : M
        n n✝ : Nat
        a✝ : R
        h : Not (LE.le i n)
        ⊢ Not (Eq (HAdd.hAdd n✝ i) n)
      -/
      omega
      /-
        🎉 no goals
      -/


theorem smul_apply (f : R[X]) (g : PolynomialModule R M) (n : ℕ) :
    (f • g) n = ∑ x ∈ Finset.antidiagonal n, f.coeff x.1 • g x.2 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Polynomial R
    g : PolynomialModule R M
    n : Nat
    ⊢ Eq ((HSMul.hSMul f g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun x  …
  -/
  induction' f using Polynomial.induction_on' with p q hp hq f_n f_a
    /-
      case h_add
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n : Nat
      p q : Polynomial R
      hp : Eq ((HSMul.hSMul p g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
      hq : Eq ((HSMul.hSMul q g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
      ⊢ Eq ((HSMul.hSMul (HAdd.hAdd p q) g) n) ((Finset.HasAntidiagonal.antidiagonal …
    -/
  · rw [add_smul, Finsupp.add_apply, hp, hq, ← Finset.sum_add_distrib]
    /-
      case h_add
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n : Nat
      p q : Polynomial R
      hp : Eq ((HSMul.hSMul p g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
      hq : Eq ((HSMul.hSMul q g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
      ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => HAdd.hAdd (HSMul.hS …
    -/
    congr
    /-
      case h_add.e_f
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n : Nat
      p q : Polynomial R
      hp : Eq ((HSMul.hSMul p g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
      hq : Eq ((HSMul.hSMul q g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
      ⊢ Eq (fun x => HAdd.hAdd (HSMul.hSMul (p.coeff x.1) (g x.2)) (HSMul.hSMul (q.c …
    -/
    ext
    /-
      case h_add.e_f.h
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n : Nat
      p q : Polynomial R
      hp : Eq ((HSMul.hSMul p g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
      hq : Eq ((HSMul.hSMul q g) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
      x✝ : Prod Nat Nat
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (p.coeff x✝.1) (g x✝.2)) (HSMul.hSMul (q.coeff x✝ …
    -/
    rw [coeff_add, add_smul]
    /-
      🎉 no goals
    -/
  · rw [Finset.Nat.sum_antidiagonal_eq_sum_range_succ fun i j => (monomial f_n f_a).coeff i • g j,
      monomial_smul_apply]
    /-
      case h_monomial
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n f_n : Nat
      f_a : R
      ⊢ Eq (ite (LE.le f_n n) (HSMul.hSMul f_a (g (HSub.hSub n f_n))) 0) ((Finset.ra …
    -/
    simp_rw [Polynomial.coeff_monomial, ← Finset.mem_range_succ_iff]
    /-
      case h_monomial
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n f_n : Nat
      f_a : R
      ⊢ Eq (ite (Membership.mem (Finset.range n.succ) f_n) (HSMul.hSMul f_a (g (HSub …
    -/
    rw [← Finset.sum_ite_eq (Finset.range (Nat.succ n)) f_n (fun x => f_a • g (n - x))]
    /-
      case h_monomial
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n f_n : Nat
      f_a : R
      ⊢ Eq ((Finset.range n.succ).sum fun x => ite (Eq f_n x) (HSMul.hSMul f_a (g (H …
    -/
    congr
    /-
      case h_monomial.e_f
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n f_n : Nat
      f_a : R
      ⊢ Eq (fun x => ite (Eq f_n x) (HSMul.hSMul f_a (g (HSub.hSub n x))) 0) fun x = …
    -/
    ext x
    /-
      case h_monomial.e_f.h
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n f_n : Nat
      f_a : R
      x : Nat
      ⊢ Eq (ite (Eq f_n x) (HSMul.hSMul f_a (g (HSub.hSub n x))) 0) (HSMul.hSMul (it …
    -/
    split_ifs
    /-
      case pos
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      g : PolynomialModule R M
      n f_n : Nat
      f_a : R
      x : Nat
      h✝ : Eq f_n x
      ⊢ Eq (HSMul.hSMul f_a (g (HSub.hSub n x))) (HSMul.hSMul f_a (g (HSub.hSub n x)))
    -/
    exacts [rfl, (zero_smul R _).symm]
    /-
      🎉 no goals
    -/


/-- `PolynomialModule R R` is isomorphic to `R[X]` as an `R[X]` module. -/
noncomputable def equivPolynomialSelf : PolynomialModule R R ≃ₗ[R[X]] R[X] :=
  { (Polynomial.toFinsuppIso R).symm with
    map_smul' := fun r x => by
      /-
        R : Type u_1
        M : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        I : Ideal R
        S : Type u_3
        inst✝³ : CommSemiring S
        inst✝² : Algebra S R
        inst✝¹ : Module S M
        inst✝ : IsScalarTower S R M
        r : Polynomial R
        x : PolynomialModule R R
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) (HSMul …
      -/
      dsimp
      /-
        R : Type u_1
        M : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        I : Ideal R
        S : Type u_3
        inst✝³ : CommSemiring S
        inst✝² : Algebra S R
        inst✝¹ : Module S M
        inst✝ : IsScalarTower S R M
        r : Polynomial R
        x : PolynomialModule R R
        ⊢ Eq ((↑(Polynomial.toFinsuppIso R)).symm (HSMul.hSMul r x)) (HMul.hMul r ((↑( …
      -/
      rw [← RingEquiv.coe_toEquiv_symm, RingEquiv.coe_toEquiv]
      induction x using induction_linear with
      | h0 => rw [smul_zero, map_zero, mul_zero]
      | hadd _ _ hp hq => rw [smul_add, map_add, map_add, mul_add, hp, hq]
      | hsingle n a =>
        ext i
        simp only [coeff_ofFinsupp, smul_single_apply, toFinsuppIso_symm_apply, coeff_ofFinsupp,
        single_apply, smul_eq_mul, Polynomial.coeff_mul, mul_ite, mul_zero]
        split_ifs with hn
        · rw [Finset.sum_eq_single (i - n, n)]
          · simp only [ite_true]
          · rintro ⟨p, q⟩ hpq1 hpq2
            rw [Finset.mem_antidiagonal] at hpq1
            split_ifs with H
            · dsimp at H
              exfalso
              apply hpq2
              rw [← hpq1, H]
              simp only [add_le_iff_nonpos_left, nonpos_iff_eq_zero, add_tsub_cancel_right]
            · rfl
          · intro H
            exfalso
            apply H
            rw [Finset.mem_antidiagonal, tsub_add_cancel_of_le hn]
        · symm
          rw [Finset.sum_ite_of_false, Finset.sum_const_zero]
          simp_rw [Finset.mem_antidiagonal]
          intro x hx
          contrapose! hn
          rw [add_comm, ← hn] at hx
          exact Nat.le.intro hx }


/-- `PolynomialModule R S` is isomorphic to `S[X]` as an `R` module. -/
noncomputable def equivPolynomial {S : Type*} [CommRing S] [Algebra R S] :
    PolynomialModule R S ≃ₗ[R] S[X] :=
  { (Polynomial.toFinsuppIso S).symm with map_smul' := fun _ _ => rfl }


@[simp]
lemma equivPolynomialSelf_apply_eq (p : PolynomialModule R R) :
    equivPolynomialSelf p = equivPolynomial p := rfl


@[simp]
lemma equivPolynomial_single {S : Type*} [CommRing S] [Algebra R S] (n : ℕ) (x : S) :
    equivPolynomial (single R n x) = monomial n x := rfl


/-- The image of a polynomial under a linear map. -/
noncomputable def map (f : M →ₗ[R] M') : PolynomialModule R M →ₗ[R] PolynomialModule R' M' :=
  Finsupp.mapRange.linearMap f


@[simp]
theorem map_single (f : M →ₗ[R] M') (i : ℕ) (m : M) : map R' f (single R i m) = single R' i (f m) :=
  Finsupp.mapRange_single (hf := f.map_zero)


theorem map_smul (f : M →ₗ[R] M') (p : R[X]) (q : PolynomialModule R M) :
    map R' f (p • q) = p.map (algebraMap R R') • map R' f q := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    R' : Type u_4
    M' : Type u_5
    inst✝⁵ : CommRing R'
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R' M'
    inst✝² : Module R M'
    inst✝¹ : Algebra R R'
    inst✝ : IsScalarTower R R' M'
    f : LinearMap (RingHom.id R) M M'
    p : Polynomial R
    q : PolynomialModule R M
    ⊢ Eq ((PolynomialModule.map R' f) (HSMul.hSMul p q)) (HSMul.hSMul (Polynomial. …
  -/
  apply induction_linear q
    /-
      case h0
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f : LinearMap (RingHom.id R) M M'
      p : Polynomial R
      q : PolynomialModule R M
      ⊢ Eq ((PolynomialModule.map R' f) (HSMul.hSMul p 0)) (HSMul.hSMul (Polynomial. …
    -/
  · rw [smul_zero, map_zero, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f : LinearMap (RingHom.id R) M M'
      p : Polynomial R
      q : PolynomialModule R M
      ⊢ ∀ (f_1 g : PolynomialModule R M), Eq ((PolynomialModule.map R' f) (HSMul.hSM …
    -/
  · intro f g e₁ e₂
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f✝ : LinearMap (RingHom.id R) M M'
      p : Polynomial R
      q f g : PolynomialModule R M
      e₁ : Eq ((PolynomialModule.map R' f✝) (HSMul.hSMul p f)) (HSMul.hSMul (Polynom …
      e₂ : Eq ((PolynomialModule.map R' f✝) (HSMul.hSMul p g)) (HSMul.hSMul (Polynom …
      ⊢ Eq ((PolynomialModule.map R' f✝) (HSMul.hSMul p (HAdd.hAdd f g))) (HSMul.hSM …
    -/
    rw [smul_add, map_add, e₁, e₂, map_add, smul_add]
    /-
      🎉 no goals
    -/
  /-
    case hsingle
    R : Type u_1
    M : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    R' : Type u_4
    M' : Type u_5
    inst✝⁵ : CommRing R'
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R' M'
    inst✝² : Module R M'
    inst✝¹ : Algebra R R'
    inst✝ : IsScalarTower R R' M'
    f : LinearMap (RingHom.id R) M M'
    p : Polynomial R
    q : PolynomialModule R M
    ⊢ ∀ (a : Nat) (b : M), Eq ((PolynomialModule.map R' f) (HSMul.hSMul p ((Polyno …
  -/
  intro i m
  induction p using Polynomial.induction_on' with
  | h_add _ _ e₁ e₂ => rw [add_smul, map_add, e₁, e₂, Polynomial.map_add, add_smul]
  | h_monomial => rw [monomial_smul_single, map_single, Polynomial.map_monomial, map_single,
      monomial_smul_single, f.map_smul, algebraMap_smul]


/-- Evaluate a polynomial `p : PolynomialModule R M` at `r : R`. -/
@[simps! (config := .lemmasOnly)]
def eval (r : R) : PolynomialModule R M →ₗ[R] M where
  toFun p := p.sum fun i m => r ^ i • m
  map_add' _ _ := Finsupp.sum_add_index' (fun _ => smul_zero _) fun _ _ _ => smul_add _ _ _
  map_smul' s m := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      I : Ideal R
      S : Type u_3
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : IsScalarTower S R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      r s : R
      m : PolynomialModule R M
      ⊢ Eq ({ toFun := fun p => Finsupp.sum p fun i m => HSMul.hSMul (HPow.hPow r i) …
    -/
    refine (Finsupp.sum_smul_index' ?_).trans ?_
      /-
        case refine_1
        R : Type u_1
        M : Type u_2
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : Module R M
        I : Ideal R
        S : Type u_3
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Algebra S R
        inst✝⁷ : Module S M
        inst✝⁶ : IsScalarTower S R M
        R' : Type u_4
        M' : Type u_5
        inst✝⁵ : CommRing R'
        inst✝⁴ : AddCommGroup M'
        inst✝³ : Module R' M'
        inst✝² : Module R M'
        inst✝¹ : Algebra R R'
        inst✝ : IsScalarTower R R' M'
        r s : R
        m : PolynomialModule R M
        ⊢ ∀ (i : Nat), Eq (HSMul.hSMul (HPow.hPow r i) 0) 0
      -/
    · exact fun i => smul_zero _
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type u_1
        M : Type u_2
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : Module R M
        I : Ideal R
        S : Type u_3
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Algebra S R
        inst✝⁷ : Module S M
        inst✝⁶ : IsScalarTower S R M
        R' : Type u_4
        M' : Type u_5
        inst✝⁵ : CommRing R'
        inst✝⁴ : AddCommGroup M'
        inst✝³ : Module R' M'
        inst✝² : Module R M'
        inst✝¹ : Algebra R R'
        inst✝ : IsScalarTower R R' M'
        r s : R
        m : PolynomialModule R M
        ⊢ Eq (Finsupp.sum m fun i c => HSMul.hSMul (HPow.hPow r i) (HSMul.hSMul s c))  …
      -/
    · simp_rw [RingHom.id_apply, Finsupp.smul_sum]
      /-
        case refine_2
        R : Type u_1
        M : Type u_2
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : Module R M
        I : Ideal R
        S : Type u_3
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Algebra S R
        inst✝⁷ : Module S M
        inst✝⁶ : IsScalarTower S R M
        R' : Type u_4
        M' : Type u_5
        inst✝⁵ : CommRing R'
        inst✝⁴ : AddCommGroup M'
        inst✝³ : Module R' M'
        inst✝² : Module R M'
        inst✝¹ : Algebra R R'
        inst✝ : IsScalarTower R R' M'
        r s : R
        m : PolynomialModule R M
        ⊢ Eq (Finsupp.sum m fun i c => HSMul.hSMul (HPow.hPow r i) (HSMul.hSMul s c))  …
      -/
      congr
      /-
        case refine_2.e_g
        R : Type u_1
        M : Type u_2
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : Module R M
        I : Ideal R
        S : Type u_3
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Algebra S R
        inst✝⁷ : Module S M
        inst✝⁶ : IsScalarTower S R M
        R' : Type u_4
        M' : Type u_5
        inst✝⁵ : CommRing R'
        inst✝⁴ : AddCommGroup M'
        inst✝³ : Module R' M'
        inst✝² : Module R M'
        inst✝¹ : Algebra R R'
        inst✝ : IsScalarTower R R' M'
        r s : R
        m : PolynomialModule R M
        ⊢ Eq (fun i c => HSMul.hSMul (HPow.hPow r i) (HSMul.hSMul s c)) fun a b => HSM …
      -/
      ext i c
      /-
        case refine_2.e_g.h.h
        R : Type u_1
        M : Type u_2
        inst✝¹² : CommRing R
        inst✝¹¹ : AddCommGroup M
        inst✝¹⁰ : Module R M
        I : Ideal R
        S : Type u_3
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Algebra S R
        inst✝⁷ : Module S M
        inst✝⁶ : IsScalarTower S R M
        R' : Type u_4
        M' : Type u_5
        inst✝⁵ : CommRing R'
        inst✝⁴ : AddCommGroup M'
        inst✝³ : Module R' M'
        inst✝² : Module R M'
        inst✝¹ : Algebra R R'
        inst✝ : IsScalarTower R R' M'
        r s : R
        m : PolynomialModule R M
        i : Nat
        c : M
        ⊢ Eq (HSMul.hSMul (HPow.hPow r i) (HSMul.hSMul s c)) (HSMul.hSMul s (HSMul.hSM …
      -/
      rw [smul_comm]
      /-
        🎉 no goals
      -/


@[simp]
theorem eval_single (r : R) (i : ℕ) (m : M) : eval r (single R i m) = r ^ i • m :=
  Finsupp.sum_single_index (smul_zero _)


@[simp]
theorem eval_lsingle (r : R) (i : ℕ) (m : M) : eval r (lsingle R i m) = r ^ i • m :=
  eval_single r i m


theorem eval_smul (p : R[X]) (q : PolynomialModule R M) (r : R) :
    eval r (p • q) = p.eval r • eval r q := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Polynomial R
    q : PolynomialModule R M
    r : R
    ⊢ Eq ((PolynomialModule.eval r) (HSMul.hSMul p q)) (HSMul.hSMul (Polynomial.ev …
  -/
  apply induction_linear q
    /-
      case h0
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      ⊢ Eq ((PolynomialModule.eval r) (HSMul.hSMul p 0)) (HSMul.hSMul (Polynomial.ev …
    -/
  · rw [smul_zero, map_zero, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      ⊢ ∀ (f g : PolynomialModule R M), Eq ((PolynomialModule.eval r) (HSMul.hSMul p …
    -/
  · intro f g e₁ e₂
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      f g : PolynomialModule R M
      e₁ : Eq ((PolynomialModule.eval r) (HSMul.hSMul p f)) (HSMul.hSMul (Polynomial …
      e₂ : Eq ((PolynomialModule.eval r) (HSMul.hSMul p g)) (HSMul.hSMul (Polynomial …
      ⊢ Eq ((PolynomialModule.eval r) (HSMul.hSMul p (HAdd.hAdd f g))) (HSMul.hSMul  …
    -/
    rw [smul_add, map_add, e₁, e₂, map_add, smul_add]
    /-
      🎉 no goals
    -/
  /-
    case hsingle
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Polynomial R
    q : PolynomialModule R M
    r : R
    ⊢ ∀ (a : Nat) (b : M), Eq ((PolynomialModule.eval r) (HSMul.hSMul p ((Polynomi …
  -/
  intro i m
  induction p using Polynomial.induction_on' with
  | h_add _ _ e₁ e₂ => rw [add_smul, map_add, Polynomial.eval_add, e₁, e₂, add_smul]
  | h_monomial => simp only [monomial_smul_single, Polynomial.eval_monomial, eval_single]; module


@[simp]
theorem eval_map (f : M →ₗ[R] M') (q : PolynomialModule R M) (r : R) :
    eval (algebraMap R R' r) (map R' f q) = f (eval r q) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    R' : Type u_4
    M' : Type u_5
    inst✝⁵ : CommRing R'
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R' M'
    inst✝² : Module R M'
    inst✝¹ : Algebra R R'
    inst✝ : IsScalarTower R R' M'
    f : LinearMap (RingHom.id R) M M'
    q : PolynomialModule R M
    r : R
    ⊢ Eq ((PolynomialModule.eval ((algebraMap R R') r)) ((PolynomialModule.map R'  …
  -/
  apply induction_linear q
    /-
      case h0
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f : LinearMap (RingHom.id R) M M'
      q : PolynomialModule R M
      r : R
      ⊢ Eq ((PolynomialModule.eval ((algebraMap R R') r)) ((PolynomialModule.map R'  …
    -/
  · simp_rw [map_zero]
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f : LinearMap (RingHom.id R) M M'
      q : PolynomialModule R M
      r : R
      ⊢ ∀ (f_1 g : PolynomialModule R M), Eq ((PolynomialModule.eval ((algebraMap R  …
    -/
  · intro f g e₁ e₂
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f✝ : LinearMap (RingHom.id R) M M'
      q : PolynomialModule R M
      r : R
      f g : PolynomialModule R M
      e₁ : Eq ((PolynomialModule.eval ((algebraMap R R') r)) ((PolynomialModule.map  …
      e₂ : Eq ((PolynomialModule.eval ((algebraMap R R') r)) ((PolynomialModule.map  …
      ⊢ Eq ((PolynomialModule.eval ((algebraMap R R') r)) ((PolynomialModule.map R'  …
    -/
    simp_rw [map_add, e₁, e₂]
    /-
      🎉 no goals
    -/
    /-
      case hsingle
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f : LinearMap (RingHom.id R) M M'
      q : PolynomialModule R M
      r : R
      ⊢ ∀ (a : Nat) (b : M), Eq ((PolynomialModule.eval ((algebraMap R R') r)) ((Pol …
    -/
  · intro i m
    /-
      case hsingle
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f : LinearMap (RingHom.id R) M M'
      q : PolynomialModule R M
      r : R
      i : Nat
      m : M
      ⊢ Eq ((PolynomialModule.eval ((algebraMap R R') r)) ((PolynomialModule.map R'  …
    -/
    simp only [map_single, eval_single, f.map_smul]
    /-
      case hsingle
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      R' : Type u_4
      M' : Type u_5
      inst✝⁵ : CommRing R'
      inst✝⁴ : AddCommGroup M'
      inst✝³ : Module R' M'
      inst✝² : Module R M'
      inst✝¹ : Algebra R R'
      inst✝ : IsScalarTower R R' M'
      f : LinearMap (RingHom.id R) M M'
      q : PolynomialModule R M
      r : R
      i : Nat
      m : M
      ⊢ Eq (HSMul.hSMul (HPow.hPow ((algebraMap R R') r) i) (f m)) (HSMul.hSMul (HPo …
    -/
    module
    /-
      🎉 no goals
    -/


@[simp]
theorem eval_map' (f : M →ₗ[R] M) (q : PolynomialModule R M) (r : R) :
    eval r (map R f q) = f (eval r q) :=
  eval_map R f q r


@[simp]
lemma aeval_equivPolynomial {S : Type*} [CommRing S] [Algebra S R]
    (f : PolynomialModule S S) (x : R) :
    aeval x (equivPolynomial f) = eval x (map R (Algebra.linearMap S R) f) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_6
    inst✝¹ : CommRing S
    inst✝ : Algebra S R
    f : PolynomialModule S S
    x : R
    ⊢ Eq ((Polynomial.aeval x) (PolynomialModule.equivPolynomial f)) ((PolynomialM …
  -/
  apply induction_linear f
    /-
      case h0
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_6
      inst✝¹ : CommRing S
      inst✝ : Algebra S R
      f : PolynomialModule S S
      x : R
      ⊢ Eq ((Polynomial.aeval x) (PolynomialModule.equivPolynomial 0)) ((PolynomialM …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_6
      inst✝¹ : CommRing S
      inst✝ : Algebra S R
      f : PolynomialModule S S
      x : R
      ⊢ ∀ (f g : PolynomialModule S S), Eq ((Polynomial.aeval x) (PolynomialModule.e …
    -/
  · intro f g e₁ e₂
    /-
      case hadd
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_6
      inst✝¹ : CommRing S
      inst✝ : Algebra S R
      f✝ : PolynomialModule S S
      x : R
      f g : PolynomialModule S S
      e₁ : Eq ((Polynomial.aeval x) (PolynomialModule.equivPolynomial f)) ((Polynomi …
      e₂ : Eq ((Polynomial.aeval x) (PolynomialModule.equivPolynomial g)) ((Polynomi …
      ⊢ Eq ((Polynomial.aeval x) (PolynomialModule.equivPolynomial (HAdd.hAdd f g))) …
    -/
    simp_rw [map_add, e₁, e₂]
    /-
      🎉 no goals
    -/
    /-
      case hsingle
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_6
      inst✝¹ : CommRing S
      inst✝ : Algebra S R
      f : PolynomialModule S S
      x : R
      ⊢ ∀ (a : Nat) (b : S), Eq ((Polynomial.aeval x) (PolynomialModule.equivPolynom …
    -/
  · intro i m
    rw [equivPolynomial_single, aeval_monomial, mul_comm, map_single,
      Algebra.linearMap_apply, eval_single, smul_eq_mul]


/-- `comp p q` is the composition of `p : R[X]` and `q : M[X]` as `q(p(x))`. -/
@[simps!]
noncomputable def comp (p : R[X]) : PolynomialModule R M →ₗ[R] PolynomialModule R M :=
  LinearMap.comp ((eval p).restrictScalars R) (map R[X] (lsingle R 0))


theorem comp_single (p : R[X]) (i : ℕ) (m : M) : comp p (single R i m) = p ^ i • single R 0 m := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Polynomial R
    i : Nat
    m : M
    ⊢ Eq ((PolynomialModule.comp p) ((PolynomialModule.single R i) m)) (HSMul.hSMu …
  -/
  rw [comp_apply, map_single, eval_single]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Polynomial R
    i : Nat
    m : M
    ⊢ Eq (HSMul.hSMul (HPow.hPow p i) ((PolynomialModule.lsingle R 0) m)) (HSMul.h …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem comp_eval (p : R[X]) (q : PolynomialModule R M) (r : R) :
    eval r (comp p q) = eval (p.eval r) q := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Polynomial R
    q : PolynomialModule R M
    r : R
    ⊢ Eq ((PolynomialModule.eval r) ((PolynomialModule.comp p) q)) ((PolynomialMod …
  -/
  rw [← LinearMap.comp_apply]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Polynomial R
    q : PolynomialModule R M
    r : R
    ⊢ Eq (((PolynomialModule.eval r).comp (PolynomialModule.comp p)) q) ((Polynomi …
  -/
  apply induction_linear q
    /-
      case h0
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      ⊢ Eq (((PolynomialModule.eval r).comp (PolynomialModule.comp p)) 0) ((Polynomi …
    -/
  · simp_rw [map_zero]
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      ⊢ ∀ (f g : PolynomialModule R M), Eq (((PolynomialModule.eval r).comp (Polynom …
    -/
  · intro _ _ e₁ e₂
    /-
      case hadd
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      f✝ g✝ : PolynomialModule R M
      e₁ : Eq (((PolynomialModule.eval r).comp (PolynomialModule.comp p)) f✝) ((Poly …
      e₂ : Eq (((PolynomialModule.eval r).comp (PolynomialModule.comp p)) g✝) ((Poly …
      ⊢ Eq (((PolynomialModule.eval r).comp (PolynomialModule.comp p)) (HAdd.hAdd f✝ …
    -/
    simp_rw [map_add, e₁, e₂]
    /-
      🎉 no goals
    -/
    /-
      case hsingle
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      ⊢ ∀ (a : Nat) (b : M), Eq (((PolynomialModule.eval r).comp (PolynomialModule.c …
    -/
  · intro i m
    /-
      case hsingle
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      i : Nat
      m : M
      ⊢ Eq (((PolynomialModule.eval r).comp (PolynomialModule.comp p)) ((PolynomialM …
    -/
    rw [LinearMap.comp_apply, comp_single, eval_single, eval_smul, eval_single, eval_pow]
    /-
      case hsingle
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Polynomial R
      q : PolynomialModule R M
      r : R
      i : Nat
      m : M
      ⊢ Eq (HSMul.hSMul (HPow.hPow (Polynomial.eval r p) i) (HSMul.hSMul (HPow.hPow  …
    -/
    module
    /-
      🎉 no goals
    -/


theorem comp_smul (p p' : R[X]) (q : PolynomialModule R M) :
    comp p (p' • q) = p'.comp p • comp p q := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Polynomial R
    q : PolynomialModule R M
    ⊢ Eq ((PolynomialModule.comp p) (HSMul.hSMul p' q)) (HSMul.hSMul (p'.comp p) ( …
  -/
  rw [comp_apply, map_smul, eval_smul, Polynomial.comp, Polynomial.eval_map, comp_apply]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Polynomial R
    q : PolynomialModule R M
    ⊢ Eq (HSMul.hSMul (Polynomial.eval₂ (algebraMap R (Polynomial R)) p p') ((Poly …
  -/
  rfl
  /-
    🎉 no goals
  -/


