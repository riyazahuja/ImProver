/--
The `R`-derivation from `A[X]` to `M[X]` which applies the derivative to each
of the coefficients.
-/
def mapCoeffs : Derivation R A[X] (PolynomialModule A M) where
  __ := (PolynomialModule.map A d.toLinearMap).comp
    PolynomialModule.equivPolynomial.symm.toLinearMap
                                                                                       /-
                                                                                         R : Type u_1
                                                                                         A : Type u_2
                                                                                         M : Type u_3
                                                                                         inst✝⁵ : CommRing R
                                                                                         inst✝⁴ : CommRing A
                                                                                         inst✝³ : Algebra R A
                                                                                         inst✝² : AddCommGroup M
                                                                                         inst✝¹ : Module A M
                                                                                         inst✝ : Module R M
                                                                                         d : Derivation R A M
                                                                                         ⊢ Eq (Finsupp.mapRange ⇑d ⋯ (Finsupp.single 0 1)) 0
                                                                                       -/
  map_one_eq_zero' := show (Finsupp.single 0 1).mapRange (d : A → M) d.map_zero = 0 by simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  leibniz' p q := by
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : AddCommGroup M
      inst✝¹ : Module A M
      inst✝ : Module R M
      d : Derivation R A M
      p q : Polynomial A
      ⊢ Eq (__spread✝⁻⁰ (HMul.hMul p q)) (HAdd.hAdd (HSMul.hSMul p (__spread✝⁻⁰ q))  …
    -/
    dsimp
    induction p using Polynomial.induction_on' with
    | h_add => simp only [add_mul, map_add, add_smul, smul_add, add_add_add_comm, *]
    | h_monomial n a =>
      induction q using Polynomial.induction_on' with
      | h_add => simp only [mul_add, map_add, add_smul, smul_add, add_add_add_comm, *]
      | h_monomial m b =>
        refine Finsupp.ext fun i ↦ ?_
        dsimp [PolynomialModule.equivPolynomial, PolynomialModule.map]
        simp only [toFinsupp_mul, toFinsupp_monomial, AddMonoidAlgebra.single_mul_single]
        show d _ = _ + _
        erw [Finsupp.mapRange.linearMap_apply, Finsupp.mapRange.linearMap_apply]
        rw [Finsupp.mapRange_single, Finsupp.mapRange_single]
        erw [PolynomialModule.monomial_smul_single, PolynomialModule.monomial_smul_single]
        simp only [AddMonoidAlgebra.single_apply, apply_ite d, leibniz, map_zero, coeFn_coe,
          PolynomialModule.single_apply, ite_add_zero, add_comm m n]


@[simp]
lemma mapCoeffs_apply (p : A[X]) (i) :
    d.mapCoeffs p i = d (coeff p i) := rfl


@[simp]
lemma mapCoeffs_monomial (n : ℕ) (x : A) :
    d.mapCoeffs (monomial n x) = .single A n (d x) := Finsupp.ext fun _ ↦ by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : AddCommGroup M
    inst✝¹ : Module A M
    inst✝ : Module R M
    d : Derivation R A M
    n : Nat
    x : A
    x✝ : Nat
    ⊢ Eq ((d.mapCoeffs ((Polynomial.monomial n) x)) x✝) (((PolynomialModule.single …
  -/
  simp [coeff_monomial, apply_ite d, PolynomialModule.single_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma mapCoeffs_X :
                                     /-
                                       R : Type u_1
                                       A : Type u_2
                                       M : Type u_3
                                       inst✝⁵ : CommRing R
                                       inst✝⁴ : CommRing A
                                       inst✝³ : Algebra R A
                                       inst✝² : AddCommGroup M
                                       inst✝¹ : Module A M
                                       inst✝ : Module R M
                                       d : Derivation R A M
                                       ⊢ Eq (d.mapCoeffs Polynomial.X) 0
                                     -/
    d.mapCoeffs (X : A[X]) = 0 := by simp [← monomial_one_one_eq_X]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
lemma mapCoeffs_C (x : A) :
                                                /-
                                                  R : Type u_1
                                                  A : Type u_2
                                                  M : Type u_3
                                                  inst✝⁵ : CommRing R
                                                  inst✝⁴ : CommRing A
                                                  inst✝³ : Algebra R A
                                                  inst✝² : AddCommGroup M
                                                  inst✝¹ : Module A M
                                                  inst✝ : Module R M
                                                  d : Derivation R A M
                                                  x : A
                                                  ⊢ Eq (d.mapCoeffs (Polynomial.C x)) ((PolynomialModule.single A 0) (d x))
                                                -/
    d.mapCoeffs (C x) = .single A 0 (d x) := by simp [← monomial_zero_left]
                                                /-
                                                  🎉 no goals
                                                -/


theorem apply_aeval_eq' (d' : Derivation R B M') (f : M →ₗ[A] M')
    (h : ∀ a, f (d a) = d' (algebraMap A B a)) (x : B) (p : A[X]) :
    d' (aeval x p) = PolynomialModule.eval x (PolynomialModule.map B f (d.mapCoeffs p)) +
      aeval x (derivative p) • d' x := by
  induction p using Polynomial.induction_on' with
  | h_add => simp_all only [eval_add, map_add, add_smul]; abel
  | h_monomial =>
    simp only [aeval_monomial, leibniz, leibniz_pow, mapCoeffs_monomial,
      PolynomialModule.map_single, PolynomialModule.eval_single, derivative_monomial, map_mul,
      _root_.map_natCast, h]
    rw [add_comm, ← smul_smul, ← smul_smul, Nat.cast_smul_eq_nsmul]



theorem apply_aeval_eq [IsScalarTower R A B] [IsScalarTower A B M'] (d : Derivation R B M')
    (x : B) (p : A[X]) :
    d (aeval x p) = PolynomialModule.eval x ((d.compAlgebraMap A).mapCoeffs p) +
      aeval x (derivative p) • d x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing A
    inst✝⁹ : Algebra R A
    B : Type u_4
    M' : Type u_5
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra R B
    inst✝⁶ : Algebra A B
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : Module B M'
    inst✝³ : Module R M'
    inst✝² : Module A M'
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower A B M'
    d : Derivation R B M'
    x : B
    p : Polynomial A
    ⊢ Eq (d ((Polynomial.aeval x) p)) (HAdd.hAdd ((PolynomialModule.eval x) ((Deri …
  -/
  convert apply_aeval_eq' (d.compAlgebraMap A) d LinearMap.id _ x p
    /-
      case h.e'_3.h.e'_5.h.e'_6
      R : Type u_1
      A : Type u_2
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing A
      inst✝⁹ : Algebra R A
      B : Type u_4
      M' : Type u_5
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra R B
      inst✝⁶ : Algebra A B
      inst✝⁵ : AddCommGroup M'
      inst✝⁴ : Module B M'
      inst✝³ : Module R M'
      inst✝² : Module A M'
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower A B M'
      d : Derivation R B M'
      x : B
      p : Polynomial A
      ⊢ Eq ((Derivation.compAlgebraMap A d).mapCoeffs p) ((PolynomialModule.map B Li …
    -/
  · apply Finsupp.ext
    /-
      case h.e'_3.h.e'_5.h.e'_6.h
      R : Type u_1
      A : Type u_2
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing A
      inst✝⁹ : Algebra R A
      B : Type u_4
      M' : Type u_5
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra R B
      inst✝⁶ : Algebra A B
      inst✝⁵ : AddCommGroup M'
      inst✝⁴ : Module B M'
      inst✝³ : Module R M'
      inst✝² : Module A M'
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower A B M'
      d : Derivation R B M'
      x : B
      p : Polynomial A
      ⊢ ∀ (a : Nat), Eq (((Derivation.compAlgebraMap A d).mapCoeffs p) a) (((Polynom …
    -/
    intro x
    /-
      case h.e'_3.h.e'_5.h.e'_6.h
      R : Type u_1
      A : Type u_2
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing A
      inst✝⁹ : Algebra R A
      B : Type u_4
      M' : Type u_5
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra R B
      inst✝⁶ : Algebra A B
      inst✝⁵ : AddCommGroup M'
      inst✝⁴ : Module B M'
      inst✝³ : Module R M'
      inst✝² : Module A M'
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower A B M'
      d : Derivation R B M'
      x✝ : B
      p : Polynomial A
      x : Nat
      ⊢ Eq (((Derivation.compAlgebraMap A d).mapCoeffs p) x) (((PolynomialModule.map …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      A : Type u_2
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing A
      inst✝⁹ : Algebra R A
      B : Type u_4
      M' : Type u_5
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra R B
      inst✝⁶ : Algebra A B
      inst✝⁵ : AddCommGroup M'
      inst✝⁴ : Module B M'
      inst✝³ : Module R M'
      inst✝² : Module A M'
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower A B M'
      d : Derivation R B M'
      x : B
      p : Polynomial A
      ⊢ ∀ (a : A), Eq (LinearMap.id ((Derivation.compAlgebraMap A d) a)) (d ((algebr …
    -/
  · intro a
    /-
      R : Type u_1
      A : Type u_2
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing A
      inst✝⁹ : Algebra R A
      B : Type u_4
      M' : Type u_5
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra R B
      inst✝⁶ : Algebra A B
      inst✝⁵ : AddCommGroup M'
      inst✝⁴ : Module B M'
      inst✝³ : Module R M'
      inst✝² : Module A M'
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower A B M'
      d : Derivation R B M'
      x : B
      p : Polynomial A
      a : A
      ⊢ Eq (LinearMap.id ((Derivation.compAlgebraMap A d) a)) (d ((algebraMap A B) a))
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem apply_eval_eq (x : A) (p : A[X]) :
    d (eval x p) = PolynomialModule.eval x (d.mapCoeffs p) + eval x (derivative p) • d x :=
  apply_aeval_eq d x p


/--
A specialization of `Derivation.mapCoeffs` for the case of a differential ring.
-/
def mapCoeffs : Derivation ℤ A[X] A[X] :=
  PolynomialModule.equivPolynomialSelf.compDer Differential.deriv.mapCoeffs


@[simp]
lemma coeff_mapCoeffs (p : A[X]) (i) :
    coeff (mapCoeffs p) i = (coeff p i)′ := rfl


@[simp]
lemma mapCoeffs_monomial (n : ℕ) (x : A) :
    mapCoeffs (monomial n x) = monomial n x′ := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Differential A
    n : Nat
    x : A
    ⊢ Eq (Differential.mapCoeffs ((Polynomial.monomial n) x)) ((Polynomial.monomia …
  -/
  simp [mapCoeffs]
  /-
    🎉 no goals
  -/


@[simp]
lemma mapCoeffs_X :
                                   /-
                                     A : Type u_1
                                     inst✝¹ : CommRing A
                                     inst✝ : Differential A
                                     ⊢ Eq (Differential.mapCoeffs Polynomial.X) 0
                                   -/
    mapCoeffs (X : A[X]) = 0 := by simp [← monomial_one_one_eq_X]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
lemma mapCoeffs_C (x : A) :
                                 /-
                                   A : Type u_1
                                   inst✝¹ : CommRing A
                                   inst✝ : Differential A
                                   x : A
                                   ⊢ Eq (Differential.mapCoeffs (Polynomial.C x)) (Polynomial.C x′)
                                 -/
    mapCoeffs (C x) = C x′ := by simp [← monomial_zero_left]
                                 /-
                                   🎉 no goals
                                 -/


theorem deriv_aeval_eq (x : R) (p : A[X]) :
    (aeval x p)′ = aeval x (mapCoeffs p) + aeval x (derivative p) * x′ := by
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : Differential A
    R : Type u_2
    inst✝³ : CommRing R
    inst✝² : Differential R
    inst✝¹ : Algebra A R
    inst✝ : DifferentialAlgebra A R
    x : R
    p : Polynomial A
    ⊢ Eq ((Polynomial.aeval x) p)′ (HAdd.hAdd ((Polynomial.aeval x) (Differential. …
  -/
  convert Derivation.apply_aeval_eq' Differential.deriv _ (Algebra.linearMap A R) ..
    /-
      case h.e'_3.h.e'_5
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : Differential A
      R : Type u_2
      inst✝³ : CommRing R
      inst✝² : Differential R
      inst✝¹ : Algebra A R
      inst✝ : DifferentialAlgebra A R
      x : R
      p : Polynomial A
      ⊢ Eq ((Polynomial.aeval x) (Differential.mapCoeffs p)) ((PolynomialModule.eval …
    -/
  · simp [mapCoeffs]
    /-
      🎉 no goals
    -/
    /-
      case convert_7
      A : Type u_1
      inst✝⁵ : CommRing A
      inst✝⁴ : Differential A
      R : Type u_2
      inst✝³ : CommRing R
      inst✝² : Differential R
      inst✝¹ : Algebra A R
      inst✝ : DifferentialAlgebra A R
      x : R
      p : Polynomial A
      ⊢ ∀ (a : A), Eq ((Algebra.linearMap A R) a′) ((algebraMap A R) a)′
    -/
  · simp [deriv_algebraMap]
    /-
      🎉 no goals
    -/


/--
The unique derivation which can be made to a `DifferentialAlgebra` on `A[X]` with
`X′ = v`.
-/
def implicitDeriv (v : A[X]) :
    Derivation ℤ A[X] A[X] :=
  mapCoeffs + v • derivative'.restrictScalars ℤ


@[simp]
lemma implicitDeriv_C (v : A[X]) (b : A) :
    implicitDeriv v (C b) = C b′ := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Differential A
    v : Polynomial A
    b : A
    ⊢ Eq ((Differential.implicitDeriv v) (Polynomial.C b)) (Polynomial.C b′)
  -/
  simp [implicitDeriv]
  /-
    🎉 no goals
  -/


@[simp]
lemma implicitDeriv_X (v : A[X]) :
    implicitDeriv v X = v := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Differential A
    v : Polynomial A
    ⊢ Eq ((Differential.implicitDeriv v) Polynomial.X) v
  -/
  simp [implicitDeriv]
  /-
    🎉 no goals
  -/


lemma deriv_aeval_eq_implicitDeriv (x : R) (v : A[X]) (h : x′ = aeval x v) (p : A[X]) :
    (aeval x p)′ = aeval x (implicitDeriv v p) := by
  /-
    A : Type u_1
    inst✝⁵ : CommRing A
    inst✝⁴ : Differential A
    R : Type u_2
    inst✝³ : CommRing R
    inst✝² : Differential R
    inst✝¹ : Algebra A R
    inst✝ : DifferentialAlgebra A R
    x : R
    v : Polynomial A
    h : Eq x′ ((Polynomial.aeval x) v)
    p : Polynomial A
    ⊢ Eq ((Polynomial.aeval x) p)′ ((Polynomial.aeval x) ((Differential.implicitDe …
  -/
  simp [deriv_aeval_eq, implicitDeriv, h, mul_comm]
  /-
    🎉 no goals
  -/


lemma algHom_deriv (f : R →ₐ[A] R') (hf : Function.Injective f) (x : R) (h : IsSeparable A x) :
    f (x′) = (f x)′ := by
  /-
    A : Type u_1
    inst✝¹¹ : CommRing A
    inst✝¹⁰ : Differential A
    R : Type u_2
    inst✝⁹ : CommRing R
    inst✝⁸ : Differential R
    inst✝⁷ : Algebra A R
    inst✝⁶ : DifferentialAlgebra A R
    R' : Type u_3
    inst✝⁵ : CommRing R'
    inst✝⁴ : Differential R'
    inst✝³ : Algebra A R'
    inst✝² : DifferentialAlgebra A R'
    inst✝¹ : IsDomain R'
    inst✝ : Nontrivial R
    f : AlgHom A R R'
    hf : Function.Injective ⇑f
    x : R
    h : IsSeparable A x
    ⊢ Eq (f x′) (f x)′
  -/
  let p := minpoly A x
  /-
    A : Type u_1
    inst✝¹¹ : CommRing A
    inst✝¹⁰ : Differential A
    R : Type u_2
    inst✝⁹ : CommRing R
    inst✝⁸ : Differential R
    inst✝⁷ : Algebra A R
    inst✝⁶ : DifferentialAlgebra A R
    R' : Type u_3
    inst✝⁵ : CommRing R'
    inst✝⁴ : Differential R'
    inst✝³ : Algebra A R'
    inst✝² : DifferentialAlgebra A R'
    inst✝¹ : IsDomain R'
    inst✝ : Nontrivial R
    f : AlgHom A R R'
    hf : Function.Injective ⇑f
    x : R
    h : IsSeparable A x
    p : Polynomial A := minpoly A x
    ⊢ Eq (f x′) (f x)′
  -/
  apply mul_left_cancel₀ (a := aeval (f x) (derivative p))
    /-
      case ha
      A : Type u_1
      inst✝¹¹ : CommRing A
      inst✝¹⁰ : Differential A
      R : Type u_2
      inst✝⁹ : CommRing R
      inst✝⁸ : Differential R
      inst✝⁷ : Algebra A R
      inst✝⁶ : DifferentialAlgebra A R
      R' : Type u_3
      inst✝⁵ : CommRing R'
      inst✝⁴ : Differential R'
      inst✝³ : Algebra A R'
      inst✝² : DifferentialAlgebra A R'
      inst✝¹ : IsDomain R'
      inst✝ : Nontrivial R
      f : AlgHom A R R'
      hf : Function.Injective ⇑f
      x : R
      h : IsSeparable A x
      p : Polynomial A := minpoly A x
      ⊢ Ne ((Polynomial.aeval (f x)) (Polynomial.derivative p)) 0
    -/
  · rw [Polynomial.aeval_algHom]
    /-
      case ha
      A : Type u_1
      inst✝¹¹ : CommRing A
      inst✝¹⁰ : Differential A
      R : Type u_2
      inst✝⁹ : CommRing R
      inst✝⁸ : Differential R
      inst✝⁷ : Algebra A R
      inst✝⁶ : DifferentialAlgebra A R
      R' : Type u_3
      inst✝⁵ : CommRing R'
      inst✝⁴ : Differential R'
      inst✝³ : Algebra A R'
      inst✝² : DifferentialAlgebra A R'
      inst✝¹ : IsDomain R'
      inst✝ : Nontrivial R
      f : AlgHom A R R'
      hf : Function.Injective ⇑f
      x : R
      h : IsSeparable A x
      p : Polynomial A := minpoly A x
      ⊢ Ne ((f.comp (Polynomial.aeval x)) (Polynomial.derivative p)) 0
    -/
    simp only [AlgHom.coe_comp, Function.comp_apply, ne_eq, map_eq_zero_iff f hf]
    /-
      case ha
      A : Type u_1
      inst✝¹¹ : CommRing A
      inst✝¹⁰ : Differential A
      R : Type u_2
      inst✝⁹ : CommRing R
      inst✝⁸ : Differential R
      inst✝⁷ : Algebra A R
      inst✝⁶ : DifferentialAlgebra A R
      R' : Type u_3
      inst✝⁵ : CommRing R'
      inst✝⁴ : Differential R'
      inst✝³ : Algebra A R'
      inst✝² : DifferentialAlgebra A R'
      inst✝¹ : IsDomain R'
      inst✝ : Nontrivial R
      f : AlgHom A R R'
      hf : Function.Injective ⇑f
      x : R
      h : IsSeparable A x
      p : Polynomial A := minpoly A x
      ⊢ Not (Eq ((Polynomial.aeval x) (Polynomial.derivative p)) 0)
    -/
    apply Separable.aeval_derivative_ne_zero h (minpoly.aeval A x)
    /-
      🎉 no goals
    -/
  /-
    case h
    A : Type u_1
    inst✝¹¹ : CommRing A
    inst✝¹⁰ : Differential A
    R : Type u_2
    inst✝⁹ : CommRing R
    inst✝⁸ : Differential R
    inst✝⁷ : Algebra A R
    inst✝⁶ : DifferentialAlgebra A R
    R' : Type u_3
    inst✝⁵ : CommRing R'
    inst✝⁴ : Differential R'
    inst✝³ : Algebra A R'
    inst✝² : DifferentialAlgebra A R'
    inst✝¹ : IsDomain R'
    inst✝ : Nontrivial R
    f : AlgHom A R R'
    hf : Function.Injective ⇑f
    x : R
    h : IsSeparable A x
    p : Polynomial A := minpoly A x
    ⊢ Eq (HMul.hMul ((Polynomial.aeval (f x)) (Polynomial.derivative p)) (f x′)) ( …
  -/
  conv => lhs; rw [Polynomial.aeval_algHom]
  /-
    case h
    A : Type u_1
    inst✝¹¹ : CommRing A
    inst✝¹⁰ : Differential A
    R : Type u_2
    inst✝⁹ : CommRing R
    inst✝⁸ : Differential R
    inst✝⁷ : Algebra A R
    inst✝⁶ : DifferentialAlgebra A R
    R' : Type u_3
    inst✝⁵ : CommRing R'
    inst✝⁴ : Differential R'
    inst✝³ : Algebra A R'
    inst✝² : DifferentialAlgebra A R'
    inst✝¹ : IsDomain R'
    inst✝ : Nontrivial R
    f : AlgHom A R R'
    hf : Function.Injective ⇑f
    x : R
    h : IsSeparable A x
    p : Polynomial A := minpoly A x
    ⊢ Eq (HMul.hMul ((f.comp (Polynomial.aeval x)) (Polynomial.derivative p)) (f x …
  -/
  simp [← map_mul]
  /-
    case h
    A : Type u_1
    inst✝¹¹ : CommRing A
    inst✝¹⁰ : Differential A
    R : Type u_2
    inst✝⁹ : CommRing R
    inst✝⁸ : Differential R
    inst✝⁷ : Algebra A R
    inst✝⁶ : DifferentialAlgebra A R
    R' : Type u_3
    inst✝⁵ : CommRing R'
    inst✝⁴ : Differential R'
    inst✝³ : Algebra A R'
    inst✝² : DifferentialAlgebra A R'
    inst✝¹ : IsDomain R'
    inst✝ : Nontrivial R
    f : AlgHom A R R'
    hf : Function.Injective ⇑f
    x : R
    h : IsSeparable A x
    p : Polynomial A := minpoly A x
    ⊢ Eq (f (HMul.hMul ((Polynomial.aeval x) (Polynomial.derivative p)) x′)) (HMul …
  -/
  apply add_left_cancel (a := aeval (f x) (mapCoeffs p))
  /-
    case h
    A : Type u_1
    inst✝¹¹ : CommRing A
    inst✝¹⁰ : Differential A
    R : Type u_2
    inst✝⁹ : CommRing R
    inst✝⁸ : Differential R
    inst✝⁷ : Algebra A R
    inst✝⁶ : DifferentialAlgebra A R
    R' : Type u_3
    inst✝⁵ : CommRing R'
    inst✝⁴ : Differential R'
    inst✝³ : Algebra A R'
    inst✝² : DifferentialAlgebra A R'
    inst✝¹ : IsDomain R'
    inst✝ : Nontrivial R
    f : AlgHom A R R'
    hf : Function.Injective ⇑f
    x : R
    h : IsSeparable A x
    p : Polynomial A := minpoly A x
    ⊢ Eq (HAdd.hAdd ((Polynomial.aeval (f x)) (Differential.mapCoeffs p)) (f (HMul …
  -/
  rw [← deriv_aeval_eq]
  simp only [aeval_algHom, AlgHom.coe_comp, Function.comp_apply, ← map_add, ← deriv_aeval_eq,
    minpoly.aeval, map_zero, p]


omit [Nontrivial R] in
lemma algEquiv_deriv (f : R ≃ₐ[A] R') (x : R) (h : IsSeparable A x) :
    f (x′) = (f x)′ :=
  haveI := f.nontrivial
  algHom_deriv f.toAlgHom f.injective x h


/--
`algHom_deriv` in a separable algebra
-/
lemma algHom_deriv' (f : R →ₐ[A] R') (hf : Function.Injective f) (x : R) :
    f (x′) = (f x)′ := algHom_deriv f hf x (Algebra.IsSeparable.isSeparable' x)


omit [Nontrivial R] in
/--
`algEquiv_deriv` in a separable algebra
-/
lemma algEquiv_deriv' (f : R ≃ₐ[A] R') (x : R) :
    f (x′) = (f x)′ :=
  haveI := f.nontrivial
  algHom_deriv' f.toAlgHom f.injective x


