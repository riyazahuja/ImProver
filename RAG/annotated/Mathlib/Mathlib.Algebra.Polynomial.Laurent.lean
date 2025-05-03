/-- The semiring of Laurent polynomials with coefficients in the semiring `R`.
We denote it by `R[T;T⁻¹]`.
The ring homomorphism `C : R →+* R[T;T⁻¹]` includes `R` as the constant polynomials. -/
abbrev LaurentPolynomial (R : Type*) [Semiring R] :=
  AddMonoidAlgebra R ℤ


@[nolint docBlame]
scoped[LaurentPolynomial] notation:9000 R "[T;T⁻¹]" => LaurentPolynomial R


@[ext]
theorem LaurentPolynomial.ext [Semiring R] {p q : R[T;T⁻¹]} (h : ∀ a, p a = q a) : p = q :=
  Finsupp.ext h


/-- The ring homomorphism, taking a polynomial with coefficients in `R` to a Laurent polynomial
with coefficients in `R`. -/
def Polynomial.toLaurent [Semiring R] : R[X] →+* R[T;T⁻¹] :=
  (mapDomainRingHom R Int.ofNatHom).comp (toFinsuppIso R)


/-- This is not a simp lemma, as it is usually preferable to use the lemmas about `C` and `X`
instead. -/
theorem Polynomial.toLaurent_apply [Semiring R] (p : R[X]) :
    toLaurent p = p.toFinsupp.mapDomain (↑) :=
  rfl


/-- The `R`-algebra map, taking a polynomial with coefficients in `R` to a Laurent polynomial
with coefficients in `R`. -/
def Polynomial.toLaurentAlg [CommSemiring R] : R[X] →ₐ[R] R[T;T⁻¹] :=
  (mapDomainAlgHom R R Int.ofNatHom).comp (toFinsuppIsoAlg R).toAlgHom


@[simp] lemma Polynomial.coe_toLaurentAlg [CommSemiring R] :
    (toLaurentAlg : R[X] → R[T;T⁻¹]) = toLaurent :=
  rfl


theorem Polynomial.toLaurentAlg_apply [CommSemiring R] (f : R[X]) : toLaurentAlg f = toLaurent f :=
  rfl


theorem single_zero_one_eq_one : (Finsupp.single 0 1 : R[T;T⁻¹]) = (1 : R[T;T⁻¹]) :=
  rfl


/-- The ring homomorphism `C`, including `R` into the ring of Laurent polynomials over `R` as
the constant Laurent polynomials. -/
def C : R →+* R[T;T⁻¹] :=
  singleZeroRingHom


theorem algebraMap_apply {R A : Type*} [CommSemiring R] [Semiring A] [Algebra R A] (r : R) :
    algebraMap R (LaurentPolynomial A) r = C (algebraMap R A r) :=
  rfl


/-- When we have `[CommSemiring R]`, the function `C` is the same as `algebraMap R R[T;T⁻¹]`.
(But note that `C` is defined when `R` is not necessarily commutative, in which case
`algebraMap` is not available.)
-/
theorem C_eq_algebraMap {R : Type*} [CommSemiring R] (r : R) : C r = algebraMap R R[T;T⁻¹] r :=
  rfl


theorem single_eq_C (r : R) : Finsupp.single 0 r = C r := rfl


@[simp] lemma C_apply (t : R) (n : ℤ) : C t n = if n = 0 then t else 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    t : R
    n : Int
    ⊢ Eq ((LaurentPolynomial.C t) n) (ite (Eq n 0) t 0)
  -/
  rw [← single_eq_C, Finsupp.single_apply]; aesop
                                            /-
                                              🎉 no goals
                                            -/


/-- The function `n ↦ T ^ n`, implemented as a sequence `ℤ → R[T;T⁻¹]`.

Using directly `T ^ n` does not work, since we want the exponents to be of Type `ℤ` and there
is no `ℤ`-power defined on `R[T;T⁻¹]`.  Using that `T` is a unit introduces extra coercions.
For these reasons, the definition of `T` is as a sequence. -/
def T (n : ℤ) : R[T;T⁻¹] :=
  Finsupp.single n 1


@[simp] lemma T_apply (m n : ℤ) : (T n : R[T;T⁻¹]) m = if n = m then 1 else 0 :=
  Finsupp.single_apply


@[simp]
theorem T_zero : (T 0 : R[T;T⁻¹]) = 1 :=
  rfl


theorem T_add (m n : ℤ) : (T (m + n) : R[T;T⁻¹]) = T m * T n := by
  -- Porting note: was `convert single_mul_single.symm`
  /-
    R : Type u_1
    inst✝ : Semiring R
    m n : Int
    ⊢ Eq (LaurentPolynomial.T (HAdd.hAdd m n)) (HMul.hMul (LaurentPolynomial.T m)  …
  -/
  simp [T, single_mul_single]
  /-
    🎉 no goals
  -/


                                                                      /-
                                                                        R : Type u_1
                                                                        inst✝ : Semiring R
                                                                        m n : Int
                                                                        ⊢ Eq (LaurentPolynomial.T (HSub.hSub m n)) (HMul.hMul (LaurentPolynomial.T m)  …
                                                                      -/
theorem T_sub (m n : ℤ) : (T (m - n) : R[T;T⁻¹]) = T m * T (-n) := by rw [← T_add, sub_eq_add_neg]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem T_pow (m : ℤ) (n : ℕ) : (T m ^ n : R[T;T⁻¹]) = T (n * m) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    m : Int
    n : Nat
    ⊢ Eq (HPow.hPow (LaurentPolynomial.T m) n) (LaurentPolynomial.T (HMul.hMul (↑n …
  -/
  rw [T, T, single_pow n, one_pow, nsmul_eq_mul]
  /-
    🎉 no goals
  -/


/-- The `simp` version of `mul_assoc`, in the presence of `T`'s. -/
@[simp]
theorem mul_T_assoc (f : R[T;T⁻¹]) (m n : ℤ) : f * T m * T n = f * T (m + n) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : LaurentPolynomial R
    m n : Int
    ⊢ Eq (HMul.hMul (HMul.hMul f (LaurentPolynomial.T m)) (LaurentPolynomial.T n)) …
  -/
  simp [← T_add, mul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem single_eq_C_mul_T (r : R) (n : ℤ) :
    (Finsupp.single n r : R[T;T⁻¹]) = (C r * T n : R[T;T⁻¹]) := by
  -- Porting note: was `convert single_mul_single.symm`
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    n : Int
    ⊢ Eq (Finsupp.single n r) (HMul.hMul (LaurentPolynomial.C r) (LaurentPolynomia …
  -/
  simp [C, T, single_mul_single]
  /-
    🎉 no goals
  -/

-- This lemma locks in the right changes and is what Lean proved directly.
-- The actual `simp`-normal form of a Laurent monomial is `C a * T n`, whenever it can be reached.

@[simp]
theorem _root_.Polynomial.toLaurent_C_mul_T (n : ℕ) (r : R) :
    (toLaurent (Polynomial.monomial n r) : R[T;T⁻¹]) = C r * T n :=
  show Finsupp.mapDomain (↑) (monomial n r).toFinsupp = (C r * T n : R[T;T⁻¹]) by
    /-
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      r : R
      ⊢ Eq (Finsupp.mapDomain Nat.cast ((Polynomial.monomial n) r).toFinsupp) (HMul. …
    -/
    rw [toFinsupp_monomial, Finsupp.mapDomain_single, single_eq_C_mul_T]
    /-
      🎉 no goals
    -/


@[simp]
theorem _root_.Polynomial.toLaurent_C (r : R) : toLaurent (Polynomial.C r) = C r := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    ⊢ Eq (Polynomial.toLaurent (Polynomial.C r)) (LaurentPolynomial.C r)
  -/
  convert Polynomial.toLaurent_C_mul_T 0 r
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : Semiring R
    r : R
    ⊢ Eq (LaurentPolynomial.C r) (HMul.hMul (LaurentPolynomial.C r) (LaurentPolyno …
  -/
  simp only [Int.ofNat_zero, T_zero, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Polynomial.toLaurent_comp_C : toLaurent (R := R) ∘ Polynomial.C = C :=
  funext Polynomial.toLaurent_C


@[simp]
theorem _root_.Polynomial.toLaurent_X : (toLaurent Polynomial.X : R[T;T⁻¹]) = T 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.toLaurent Polynomial.X) (LaurentPolynomial.T 1)
  -/
  have : (Polynomial.X : R[X]) = monomial 1 1 := by simp [← C_mul_X_pow_eq_monomial]
  /-
    R : Type u_1
    inst✝ : Semiring R
    this : Eq Polynomial.X ((Polynomial.monomial 1) 1)
    ⊢ Eq (Polynomial.toLaurent Polynomial.X) (LaurentPolynomial.T 1)
  -/
  simp [this, Polynomial.toLaurent_C_mul_T]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Polynomial.toLaurent_one : (Polynomial.toLaurent : R[X] → R[T;T⁻¹]) 1 = 1 :=
  map_one Polynomial.toLaurent


@[simp]
theorem _root_.Polynomial.toLaurent_C_mul_eq (r : R) (f : R[X]) :
    toLaurent (Polynomial.C r * f) = C r * toLaurent f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    f : Polynomial R
    ⊢ Eq (Polynomial.toLaurent (HMul.hMul (Polynomial.C r) f)) (HMul.hMul (Laurent …
  -/
  simp only [_root_.map_mul, Polynomial.toLaurent_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Polynomial.toLaurent_X_pow (n : ℕ) : toLaurent (X ^ n : R[X]) = T n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (Polynomial.toLaurent (HPow.hPow Polynomial.X n)) (LaurentPolynomial.T ↑n)
  -/
  simp only [map_pow, Polynomial.toLaurent_X, T_pow, mul_one]
  /-
    🎉 no goals
  -/


theorem _root_.Polynomial.toLaurent_C_mul_X_pow (n : ℕ) (r : R) :
    toLaurent (Polynomial.C r * X ^ n) = C r * T n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    r : R
    ⊢ Eq (Polynomial.toLaurent (HMul.hMul (Polynomial.C r) (HPow.hPow Polynomial.X …
  -/
  simp only [_root_.map_mul, Polynomial.toLaurent_C, Polynomial.toLaurent_X_pow]
  /-
    🎉 no goals
  -/


instance invertibleT (n : ℤ) : Invertible (T n : R[T;T⁻¹]) where
  invOf := T (-n)
                       /-
                         R : Type u_1
                         S : Type u_2
                         inst✝ : Semiring R
                         n : Int
                         ⊢ Eq (HMul.hMul (LaurentPolynomial.T (Neg.neg n)) (LaurentPolynomial.T n)) 1
                       -/
  invOf_mul_self := by rw [← T_add, neg_add_cancel, T_zero]
                       /-
                         🎉 no goals
                       -/
                       /-
                         R : Type u_1
                         S : Type u_2
                         inst✝ : Semiring R
                         n : Int
                         ⊢ Eq (HMul.hMul (LaurentPolynomial.T n) (LaurentPolynomial.T (Neg.neg n))) 1
                       -/
  mul_invOf_self := by rw [← T_add, add_neg_cancel, T_zero]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem invOf_T (n : ℤ) : ⅟ (T n : R[T;T⁻¹]) = T (-n) :=
  rfl


theorem isUnit_T (n : ℤ) : IsUnit (T n : R[T;T⁻¹]) :=
  isUnit_of_invertible _


@[elab_as_elim]
protected theorem induction_on {M : R[T;T⁻¹] → Prop} (p : R[T;T⁻¹]) (h_C : ∀ a, M (C a))
    (h_add : ∀ {p q}, M p → M q → M (p + q))
    (h_C_mul_T : ∀ (n : ℕ) (a : R), M (C a * T n) → M (C a * T (n + 1)))
    (h_C_mul_T_Z : ∀ (n : ℕ) (a : R), M (C a * T (-n)) → M (C a * T (-n - 1))) : M p := by
  have A : ∀ {n : ℤ} {a : R}, M (C a * T n) := by
    intro n a
    refine Int.induction_on n ?_ ?_ ?_
    · simpa only [T_zero, mul_one] using h_C a
    · exact fun m => h_C_mul_T m a
    · exact fun m => h_C_mul_T_Z m a
  have B : ∀ s : Finset ℤ, M (s.sum fun n : ℤ => C (p.toFun n) * T n) := by
    apply Finset.induction
    · convert h_C 0
      simp only [Finset.sum_empty, _root_.map_zero]
    · intro n s ns ih
      rw [Finset.sum_insert ns]
      exact h_add A ih
  /-
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_C : ∀ (a : R), M (LaurentPolynomial.C a)
    h_add : ∀ {p q : LaurentPolynomial R}, M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    h_C_mul_T_Z : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laure …
    A : ∀ {n : Int} {a : R}, M (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomi …
    B : ∀ (s : Finset Int), M (s.sum fun n => HMul.hMul (LaurentPolynomial.C (p.to …
    ⊢ M p
  -/
  convert B p.support
  /-
    case h.e'_1
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_C : ∀ (a : R), M (LaurentPolynomial.C a)
    h_add : ∀ {p q : LaurentPolynomial R}, M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    h_C_mul_T_Z : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laure …
    A : ∀ {n : Int} {a : R}, M (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomi …
    B : ∀ (s : Finset Int), M (s.sum fun n => HMul.hMul (LaurentPolynomial.C (p.to …
    ⊢ Eq p (p.support.sum fun n => HMul.hMul (LaurentPolynomial.C (p.toFun n)) (La …
  -/
  ext a
  /-
    case h.e'_1.h
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_C : ∀ (a : R), M (LaurentPolynomial.C a)
    h_add : ∀ {p q : LaurentPolynomial R}, M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    h_C_mul_T_Z : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laure …
    A : ∀ {n : Int} {a : R}, M (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomi …
    B : ∀ (s : Finset Int), M (s.sum fun n => HMul.hMul (LaurentPolynomial.C (p.to …
    a : Int
    ⊢ Eq (p a) ((p.support.sum fun n => HMul.hMul (LaurentPolynomial.C (p.toFun n) …
  -/
  simp_rw [← single_eq_C_mul_T]
  -- Porting note: did not make progress in `simp_rw`
  /-
    case h.e'_1.h
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_C : ∀ (a : R), M (LaurentPolynomial.C a)
    h_add : ∀ {p q : LaurentPolynomial R}, M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    h_C_mul_T_Z : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laure …
    A : ∀ {n : Int} {a : R}, M (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomi …
    B : ∀ (s : Finset Int), M (s.sum fun n => HMul.hMul (LaurentPolynomial.C (p.to …
    a : Int
    ⊢ Eq (p a) ((p.support.sum fun x => Finsupp.single x (p.toFun x)) a)
  -/
  rw [Finset.sum_apply']
  /-
    case h.e'_1.h
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_C : ∀ (a : R), M (LaurentPolynomial.C a)
    h_add : ∀ {p q : LaurentPolynomial R}, M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    h_C_mul_T_Z : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laure …
    A : ∀ {n : Int} {a : R}, M (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomi …
    B : ∀ (s : Finset Int), M (s.sum fun n => HMul.hMul (LaurentPolynomial.C (p.to …
    a : Int
    ⊢ Eq (p a) (p.support.sum fun k => (Finsupp.single k (p.toFun k)) a)
  -/
  simp_rw [Finsupp.single_apply, Finset.sum_ite_eq']
  /-
    case h.e'_1.h
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_C : ∀ (a : R), M (LaurentPolynomial.C a)
    h_add : ∀ {p q : LaurentPolynomial R}, M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    h_C_mul_T_Z : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laure …
    A : ∀ {n : Int} {a : R}, M (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomi …
    B : ∀ (s : Finset Int), M (s.sum fun n => HMul.hMul (LaurentPolynomial.C (p.to …
    a : Int
    ⊢ Eq (p a) (ite (Membership.mem p.support a) (p.toFun a) 0)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      M : LaurentPolynomial R → Prop
      p : LaurentPolynomial R
      h_C : ∀ (a : R), M (LaurentPolynomial.C a)
      h_add : ∀ {p q : LaurentPolynomial R}, M p → M q → M (HAdd.hAdd p q)
      h_C_mul_T : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
      h_C_mul_T_Z : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laure …
      A : ∀ {n : Int} {a : R}, M (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomi …
      B : ∀ (s : Finset Int), M (s.sum fun n => HMul.hMul (LaurentPolynomial.C (p.to …
      a : Int
      h : Membership.mem p.support a
      ⊢ Eq (p a) (p.toFun a)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      M : LaurentPolynomial R → Prop
      p : LaurentPolynomial R
      h_C : ∀ (a : R), M (LaurentPolynomial.C a)
      h_add : ∀ {p q : LaurentPolynomial R}, M p → M q → M (HAdd.hAdd p q)
      h_C_mul_T : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
      h_C_mul_T_Z : ∀ (n : Nat) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laure …
      A : ∀ {n : Int} {a : R}, M (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomi …
      B : ∀ (s : Finset Int), M (s.sum fun n => HMul.hMul (LaurentPolynomial.C (p.to …
      a : Int
      h : Not (Membership.mem p.support a)
      ⊢ Eq (p a) 0
    -/
  · exact Finsupp.not_mem_support_iff.mp h
    /-
      🎉 no goals
    -/


/-- To prove something about Laurent polynomials, it suffices to show that
* the condition is closed under taking sums, and
* it holds for monomials.
-/
@[elab_as_elim]
protected theorem induction_on' {M : R[T;T⁻¹] → Prop} (p : R[T;T⁻¹])
    (h_add : ∀ p q, M p → M q → M (p + q)) (h_C_mul_T : ∀ (n : ℤ) (a : R), M (C a * T n)) :
    M p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_add : ∀ (p q : LaurentPolynomial R), M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Int) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    ⊢ M p
  -/
  refine p.induction_on (fun a => ?_) (fun {p q} => h_add p q) ?_ ?_ <;>
      /-
        case refine_1
        R : Type u_1
        inst✝ : Semiring R
        M : LaurentPolynomial R → Prop
        p : LaurentPolynomial R
        h_add : ∀ (p q : LaurentPolynomial R), M p → M q → M (HAdd.hAdd p q)
        h_C_mul_T : ∀ (n : Int) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
        a : R
        ⊢ M (LaurentPolynomial.C a)
      -/
      /-
        🎉 no goals
      -/
      try exact fun n f _ => h_C_mul_T _ f
      /-
        🎉 no goals
      -/
  /-
    case refine_1
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_add : ∀ (p q : LaurentPolynomial R), M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Int) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    a : R
    ⊢ M (LaurentPolynomial.C a)
  -/
  convert h_C_mul_T 0 a
  /-
    case h.e'_1
    R : Type u_1
    inst✝ : Semiring R
    M : LaurentPolynomial R → Prop
    p : LaurentPolynomial R
    h_add : ∀ (p q : LaurentPolynomial R), M p → M q → M (HAdd.hAdd p q)
    h_C_mul_T : ∀ (n : Int) (a : R), M (HMul.hMul (LaurentPolynomial.C a) (Laurent …
    a : R
    ⊢ Eq (LaurentPolynomial.C a) (HMul.hMul (LaurentPolynomial.C a) (LaurentPolyno …
  -/
  exact (mul_one _).symm
  /-
    🎉 no goals
  -/


theorem commute_T (n : ℤ) (f : R[T;T⁻¹]) : Commute (T n) f :=
  f.induction_on' (fun _ _ Tp Tq => Commute.add_right Tp Tq) fun m a =>
    show T n * _ = _ by
      /-
        R : Type u_1
        inst✝ : Semiring R
        n : Int
        f : LaurentPolynomial R
        m : Int
        a : R
        ⊢ Eq (HMul.hMul (LaurentPolynomial.T n) (HMul.hMul (LaurentPolynomial.C a) (La …
      -/
      rw [T, T, ← single_eq_C, single_mul_single, single_mul_single, single_mul_single]
      /-
        R : Type u_1
        inst✝ : Semiring R
        n : Int
        f : LaurentPolynomial R
        m : Int
        a : R
        ⊢ Eq (AddMonoidAlgebra.single (HAdd.hAdd n (HAdd.hAdd 0 m)) (HMul.hMul 1 (HMul …
      -/
      simp [add_comm]
      /-
        🎉 no goals
      -/


@[simp]
theorem T_mul (n : ℤ) (f : R[T;T⁻¹]) : T n * f = f * T n :=
  (commute_T n f).eq


theorem smul_eq_C_mul (r : R) (f : R[T;T⁻¹]) : r • f = C r * f := by
  induction f using LaurentPolynomial.induction_on' with
  | h_add _ _ hp hq =>
    rw [smul_add, mul_add, hp, hq]
  | h_C_mul_T n s =>
    rw [← mul_assoc, ← smul_mul_assoc, mul_left_inj_of_invertible, ← map_mul, ← single_eq_C,
      Finsupp.smul_single', single_eq_C]


/-- `trunc : R[T;T⁻¹] →+ R[X]` maps a Laurent polynomial `f` to the polynomial whose terms of
nonnegative degree coincide with the ones of `f`.  The terms of negative degree of `f` "vanish".
`trunc` is a left-inverse to `Polynomial.toLaurent`. -/
def trunc : R[T;T⁻¹] →+ R[X] :=
  (toFinsuppIso R).symm.toAddMonoidHom.comp <| comapDomain.addMonoidHom fun _ _ => Int.ofNat.inj


@[simp]
theorem trunc_C_mul_T (n : ℤ) (r : R) : trunc (C r * T n) = ite (0 ≤ n) (monomial n.toNat r) 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Int
    r : R
    ⊢ Eq (LaurentPolynomial.trunc (HMul.hMul (LaurentPolynomial.C r) (LaurentPolyn …
  -/
  apply (toFinsuppIso R).injective
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    n : Int
    r : R
    ⊢ Eq ((Polynomial.toFinsuppIso R) (LaurentPolynomial.trunc (HMul.hMul (Laurent …
  -/
  rw [← single_eq_C_mul_T, trunc, AddMonoidHom.coe_comp, Function.comp_apply]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): was `rw`
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    n : Int
    r : R
    ⊢ Eq ((Polynomial.toFinsuppIso R) ((Polynomial.toFinsuppIso R).symm.toAddMonoi …
  -/
  erw [comapDomain.addMonoidHom_apply Int.ofNat_injective]
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    n : Int
    r : R
    ⊢ Eq (Finsupp.comapDomain Int.ofNat (Finsupp.single n r) ⋯) ((Polynomial.toFin …
  -/
  rw [toFinsuppIso_apply]
  -- Porting note: rewrote proof below relative to mathlib3.
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    n : Int
    r : R
    ⊢ Eq (Finsupp.comapDomain Int.ofNat (Finsupp.single n r) ⋯) (ite (LE.le 0 n) ( …
  -/
  by_cases n0 : 0 ≤ n
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      n : Int
      r : R
      n0 : LE.le 0 n
      ⊢ Eq (Finsupp.comapDomain Int.ofNat (Finsupp.single n r) ⋯) (ite (LE.le 0 n) ( …
    -/
  · lift n to ℕ using n0
    /-
      case pos.intro
      R : Type u_1
      inst✝ : Semiring R
      r : R
      n : Nat
      ⊢ Eq (Finsupp.comapDomain Int.ofNat (Finsupp.single (↑n) r) ⋯) (ite (LE.le 0 ↑ …
    -/
    erw [comapDomain_single]
    /-
      case pos.intro
      R : Type u_1
      inst✝ : Semiring R
      r : R
      n : Nat
      ⊢ Eq (Finsupp.single n r) (ite (LE.le 0 ↑n) ((Polynomial.monomial (↑n).toNat)  …
    -/
    simp only [Nat.cast_nonneg, Int.toNat_ofNat, ite_true, toFinsupp_monomial]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      n : Int
      r : R
      n0 : Not (LE.le 0 n)
      ⊢ Eq (Finsupp.comapDomain Int.ofNat (Finsupp.single n r) ⋯) (ite (LE.le 0 n) ( …
    -/
  · lift -n to ℕ using (neg_pos.mpr (not_le.mp n0)).le with m
    /-
      case neg.intro
      R : Type u_1
      inst✝ : Semiring R
      n : Int
      r : R
      n0 : Not (LE.le 0 n)
      m : Nat
      ⊢ Eq (Finsupp.comapDomain Int.ofNat (Finsupp.single n r) ⋯) (ite (LE.le 0 n) ( …
    -/
    rw [toFinsupp_inj, if_neg n0]
    /-
      case neg.intro
      R : Type u_1
      inst✝ : Semiring R
      n : Int
      r : R
      n0 : Not (LE.le 0 n)
      m : Nat
      ⊢ Eq { toFinsupp := Finsupp.comapDomain Int.ofNat (Finsupp.single n r) ⋯ } 0
    -/
    ext a
    /-
      case neg.intro.a
      R : Type u_1
      inst✝ : Semiring R
      n : Int
      r : R
      n0 : Not (LE.le 0 n)
      m a : Nat
      ⊢ Eq ({ toFinsupp := Finsupp.comapDomain Int.ofNat (Finsupp.single n r) ⋯ }.co …
    -/
    have := ((not_le.mp n0).trans_le (Int.ofNat_zero_le a)).ne
    simp only [coeff_ofFinsupp, comapDomain_apply, Int.ofNat_eq_coe, coeff_zero,
      single_eq_of_ne this]


@[simp]
theorem leftInverse_trunc_toLaurent :
    Function.LeftInverse (trunc : R[T;T⁻¹] → R[X]) Polynomial.toLaurent := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Function.LeftInverse ⇑LaurentPolynomial.trunc ⇑Polynomial.toLaurent
  -/
  refine fun f => f.induction_on' ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      ⊢ ∀ (p q : Polynomial R), Eq (LaurentPolynomial.trunc (Polynomial.toLaurent p) …
    -/
  · intro f g hf hg
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      f✝ f g : Polynomial R
      hf : Eq (LaurentPolynomial.trunc (Polynomial.toLaurent f)) f
      hg : Eq (LaurentPolynomial.trunc (Polynomial.toLaurent g)) g
      ⊢ Eq (LaurentPolynomial.trunc (Polynomial.toLaurent (HAdd.hAdd f g))) (HAdd.hA …
    -/
    simp only [hf, hg, _root_.map_add]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      ⊢ ∀ (n : Nat) (a : R), Eq (LaurentPolynomial.trunc (Polynomial.toLaurent ((Pol …
    -/
  · intro n r
    simp only [Polynomial.toLaurent_C_mul_T, trunc_C_mul_T, Int.natCast_nonneg, Int.toNat_natCast,
      if_true]


@[simp]
theorem _root_.Polynomial.trunc_toLaurent (f : R[X]) : trunc (toLaurent f) = f :=
  leftInverse_trunc_toLaurent _


theorem _root_.Polynomial.toLaurent_injective :
    Function.Injective (Polynomial.toLaurent : R[X] → R[T;T⁻¹]) :=
  leftInverse_trunc_toLaurent.injective


@[simp]
theorem _root_.Polynomial.toLaurent_inj (f g : R[X]) : toLaurent f = toLaurent g ↔ f = g :=
  ⟨fun h => Polynomial.toLaurent_injective h, congr_arg _⟩


theorem _root_.Polynomial.toLaurent_ne_zero {f : R[X]} : toLaurent f ≠ 0 ↔ f ≠ 0 :=
  map_ne_zero_iff _ Polynomial.toLaurent_injective


@[simp]
theorem _root_.Polynomial.toLaurent_eq_zero {f : R[X]} : toLaurent f = 0 ↔ f = 0 :=
  map_eq_zero_iff _ Polynomial.toLaurent_injective


theorem exists_T_pow (f : R[T;T⁻¹]) : ∃ (n : ℕ) (f' : R[X]), toLaurent f' = f * T n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : LaurentPolynomial R
    ⊢ Exists fun n => Exists fun f' => Eq (Polynomial.toLaurent f') (HMul.hMul f ( …
  -/
  refine f.induction_on' ?_ fun n a => ?_ <;> clear f
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      ⊢ ∀ (p q : LaurentPolynomial R), (Exists fun n => Exists fun f' => Eq (Polynom …
    -/
  · rintro f g ⟨m, fn, hf⟩ ⟨n, gn, hg⟩
    /-
      case refine_1.intro.intro.intro.intro
      R : Type u_1
      inst✝ : Semiring R
      f g : LaurentPolynomial R
      m : Nat
      fn : Polynomial R
      hf : Eq (Polynomial.toLaurent fn) (HMul.hMul f (LaurentPolynomial.T ↑m))
      n : Nat
      gn : Polynomial R
      hg : Eq (Polynomial.toLaurent gn) (HMul.hMul g (LaurentPolynomial.T ↑n))
      ⊢ Exists fun n => Exists fun f' => Eq (Polynomial.toLaurent f') (HMul.hMul (HA …
    -/
    refine ⟨m + n, fn * X ^ n + gn * X ^ m, ?_⟩
    simp only [hf, hg, add_mul, add_comm (n : ℤ), map_add, map_mul, Polynomial.toLaurent_X_pow,
      mul_T_assoc, Int.ofNat_add]
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      n : Int
      a : R
      ⊢ Exists fun n_1 => Exists fun f' => Eq (Polynomial.toLaurent f') (HMul.hMul ( …
    -/
  · cases' n with n n
      /-
        case refine_2.ofNat
        R : Type u_1
        inst✝ : Semiring R
        a : R
        n : Nat
        ⊢ Exists fun n_1 => Exists fun f' => Eq (Polynomial.toLaurent f') (HMul.hMul ( …
      -/
    · exact ⟨0, Polynomial.C a * X ^ n, by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.negSucc
        R : Type u_1
        inst✝ : Semiring R
        a : R
        n : Nat
        ⊢ Exists fun n_1 => Exists fun f' => Eq (Polynomial.toLaurent f') (HMul.hMul ( …
      -/
    · refine ⟨n + 1, Polynomial.C a, ?_⟩
      simp only [Int.negSucc_eq, Polynomial.toLaurent_C, Int.ofNat_succ, mul_T_assoc,
        neg_add_cancel, T_zero, mul_one]


/-- This is a version of `exists_T_pow` stated as an induction principle. -/
@[elab_as_elim]
theorem induction_on_mul_T {Q : R[T;T⁻¹] → Prop} (f : R[T;T⁻¹])
    (Qf : ∀ {f : R[X]} {n : ℕ}, Q (toLaurent f * T (-n))) : Q f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    Q : LaurentPolynomial R → Prop
    f : LaurentPolynomial R
    Qf : ∀ {f : Polynomial R} {n : Nat}, Q (HMul.hMul (Polynomial.toLaurent f) (La …
    ⊢ Q f
  -/
  rcases f.exists_T_pow with ⟨n, f', hf⟩
  rw [← mul_one f, ← T_zero, ← Nat.cast_zero, ← Nat.sub_self n, Nat.cast_sub rfl.le, T_sub,
    ← mul_assoc, ← hf]
  /-
    case intro.intro
    R : Type u_1
    inst✝ : Semiring R
    Q : LaurentPolynomial R → Prop
    f : LaurentPolynomial R
    Qf : ∀ {f : Polynomial R} {n : Nat}, Q (HMul.hMul (Polynomial.toLaurent f) (La …
    n : Nat
    f' : Polynomial R
    hf : Eq (Polynomial.toLaurent f') (HMul.hMul f (LaurentPolynomial.T ↑n))
    ⊢ Q (HMul.hMul (Polynomial.toLaurent f') (LaurentPolynomial.T (Neg.neg ↑n)))
  -/
  exact Qf
  /-
    🎉 no goals
  -/


/-- Suppose that `Q` is a statement about Laurent polynomials such that
* `Q` is true on *ordinary* polynomials;
* `Q (f * T)` implies `Q f`;
it follow that `Q` is true on all Laurent polynomials. -/
theorem reduce_to_polynomial_of_mul_T (f : R[T;T⁻¹]) {Q : R[T;T⁻¹] → Prop}
    (Qf : ∀ f : R[X], Q (toLaurent f)) (QT : ∀ f, Q (f * T 1) → Q f) : Q f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : LaurentPolynomial R
    Q : LaurentPolynomial R → Prop
    Qf : ∀ (f : Polynomial R), Q (Polynomial.toLaurent f)
    QT : ∀ (f : LaurentPolynomial R), Q (HMul.hMul f (LaurentPolynomial.T 1)) → Q f
    ⊢ Q f
  -/
  induction' f using LaurentPolynomial.induction_on_mul_T with f n
  induction n with
  | zero => simpa only [Nat.cast_zero, neg_zero, T_zero, mul_one] using Qf _
  | succ n hn => convert QT _ _; simpa using hn


theorem support_C_mul_T (a : R) (n : ℤ) : Finsupp.support (C a * T n) ⊆ {n} := by
  -- Porting note: was
  -- simpa only [← single_eq_C_mul_T] using support_single_subset
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    n : Int
    ⊢ HasSubset.Subset (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomial.T n)) …
  -/
  rw [← single_eq_C_mul_T]
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    n : Int
    ⊢ HasSubset.Subset (Finsupp.single n a).support (Singleton.singleton n)
  -/
  exact support_single_subset
  /-
    🎉 no goals
  -/


theorem support_C_mul_T_of_ne_zero {a : R} (a0 : a ≠ 0) (n : ℤ) :
    Finsupp.support (C a * T n) = {n} := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    a0 : Ne a 0
    n : Int
    ⊢ Eq (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomial.T n)).support (Sing …
  -/
  rw [← single_eq_C_mul_T]
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    a0 : Ne a 0
    n : Int
    ⊢ Eq (Finsupp.single n a).support (Singleton.singleton n)
  -/
  exact support_single_ne_zero _ a0
  /-
    🎉 no goals
  -/


/-- The support of a polynomial `f` is a finset in `ℕ`.  The lemma `toLaurent_support f`
shows that the support of `f.toLaurent` is the same finset, but viewed in `ℤ` under the natural
inclusion `ℕ ↪ ℤ`. -/
theorem toLaurent_support (f : R[X]) : f.toLaurent.support = f.support.map Nat.castEmbedding := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq (Polynomial.toLaurent f).support (Finset.map Nat.castEmbedding f.support)
  -/
  generalize hd : f.support = s
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    s : Finset Nat
    hd : Eq f.support s
    ⊢ Eq (Polynomial.toLaurent f).support (Finset.map Nat.castEmbedding s)
  -/
  revert f
  /-
    R : Type u_1
    inst✝ : Semiring R
    s : Finset Nat
    ⊢ ∀ (f : Polynomial R), Eq f.support s → Eq (Polynomial.toLaurent f).support ( …
  -/
  refine Finset.induction_on s ?_ ?_ <;> clear s
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      ⊢ ∀ (f : Polynomial R), Eq f.support EmptyCollection.emptyCollection → Eq (Pol …
    -/
  · intro f hf
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Eq f.support EmptyCollection.emptyCollection
      ⊢ Eq (Polynomial.toLaurent f).support (Finset.map Nat.castEmbedding EmptyColle …
    -/
    rw [Finset.map_empty, Finsupp.support_eq_empty, toLaurent_eq_zero]
    /-
      case refine_1
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Eq f.support EmptyCollection.emptyCollection
      ⊢ Eq f 0
    -/
    exact Polynomial.support_eq_empty.mp hf
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : Semiring R
      ⊢ ∀ ⦃a : Nat⦄ {s : Finset Nat}, Not (Membership.mem s a) → (∀ (f : Polynomial  …
    -/
  · intro a s as hf f fs
    have : (erase a f).toLaurent.support = s.map Nat.castEmbedding := by
      refine hf (f.erase a) ?_
      simp only [fs, Finset.erase_eq_of_not_mem as, Polynomial.support_erase,
        Finset.erase_insert_eq_erase]
    rw [← monomial_add_erase f a, Finset.map_insert, ← this, map_add, Polynomial.toLaurent_C_mul_T,
      support_add_eq, Finset.insert_eq]
      /-
        case refine_2
        R : Type u_1
        inst✝ : Semiring R
        a : Nat
        s : Finset Nat
        as : Not (Membership.mem s a)
        hf : ∀ (f : Polynomial R), Eq f.support s → Eq (Polynomial.toLaurent f).suppor …
        f : Polynomial R
        fs : Eq f.support (Insert.insert a s)
        this : Eq (Polynomial.toLaurent (Polynomial.erase a f)).support (Finset.map Na …
        ⊢ Eq (Union.union (HMul.hMul (LaurentPolynomial.C (f.coeff a)) (LaurentPolynom …
      -/
    · congr
      /-
        case refine_2.e_a
        R : Type u_1
        inst✝ : Semiring R
        a : Nat
        s : Finset Nat
        as : Not (Membership.mem s a)
        hf : ∀ (f : Polynomial R), Eq f.support s → Eq (Polynomial.toLaurent f).suppor …
        f : Polynomial R
        fs : Eq f.support (Insert.insert a s)
        this : Eq (Polynomial.toLaurent (Polynomial.erase a f)).support (Finset.map Na …
        ⊢ Eq (HMul.hMul (LaurentPolynomial.C (f.coeff a)) (LaurentPolynomial.T ↑a)).su …
      -/
      exact support_C_mul_T_of_ne_zero (Polynomial.mem_support_iff.mp (by simp [fs])) _
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type u_1
        inst✝ : Semiring R
        a : Nat
        s : Finset Nat
        as : Not (Membership.mem s a)
        hf : ∀ (f : Polynomial R), Eq f.support s → Eq (Polynomial.toLaurent f).suppor …
        f : Polynomial R
        fs : Eq f.support (Insert.insert a s)
        this : Eq (Polynomial.toLaurent (Polynomial.erase a f)).support (Finset.map Na …
        ⊢ Disjoint (HMul.hMul (LaurentPolynomial.C (f.coeff a)) (LaurentPolynomial.T ↑ …
      -/
    · rw [this]
      /-
        case refine_2
        R : Type u_1
        inst✝ : Semiring R
        a : Nat
        s : Finset Nat
        as : Not (Membership.mem s a)
        hf : ∀ (f : Polynomial R), Eq f.support s → Eq (Polynomial.toLaurent f).suppor …
        f : Polynomial R
        fs : Eq f.support (Insert.insert a s)
        this : Eq (Polynomial.toLaurent (Polynomial.erase a f)).support (Finset.map Na …
        ⊢ Disjoint (HMul.hMul (LaurentPolynomial.C (f.coeff a)) (LaurentPolynomial.T ↑ …
      -/
      exact Disjoint.mono_left (support_C_mul_T _ _) (by simpa)
      /-
        🎉 no goals
      -/


/-- The degree of a Laurent polynomial takes values in `WithBot ℤ`.
If `f : R[T;T⁻¹]` is a Laurent polynomial, then `f.degree` is the maximum of its support of `f`,
or `⊥`, if `f = 0`. -/
def degree (f : R[T;T⁻¹]) : WithBot ℤ :=
  f.support.max


@[simp]
theorem degree_zero : degree (0 : R[T;T⁻¹]) = ⊥ :=
  rfl


@[simp]
theorem degree_eq_bot_iff {f : R[T;T⁻¹]} : f.degree = ⊥ ↔ f = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : LaurentPolynomial R
    ⊢ Iff (Eq f.degree Bot.bot) (Eq f 0)
  -/
  refine ⟨fun h => ?_, fun h => by rw [h, degree_zero]⟩
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : LaurentPolynomial R
    h : Eq f.degree Bot.bot
    ⊢ Eq f 0
  -/
  rw [degree, Finset.max_eq_sup_withBot] at h
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : LaurentPolynomial R
    h : Eq (f.support.sup WithBot.some) Bot.bot
    ⊢ Eq f 0
  -/
  ext n
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f : LaurentPolynomial R
    h : Eq (f.support.sup WithBot.some) Bot.bot
    n : Int
    ⊢ Eq (f n) (0 n)
  -/
  simp_rw [Finset.sup_eq_bot_iff, Finsupp.mem_support_iff, Ne, WithBot.coe_ne_bot] at h
  /-
    case h
    R : Type u_1
    inst✝ : Semiring R
    f : LaurentPolynomial R
    n : Int
    h : ∀ (s : Int), Not (Eq (f s) 0) → False
    ⊢ Eq (f n) (0 n)
  -/
  exact not_not.mp (h n)
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_C_mul_T (n : ℤ) (a : R) (a0 : a ≠ 0) : degree (C a * T n) = n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Int
    a : R
    a0 : Ne a 0
    ⊢ Eq (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomial.T n)).degree ↑n
  -/
  rw [degree, support_C_mul_T_of_ne_zero a0 n]
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Int
    a : R
    a0 : Ne a 0
    ⊢ Eq (Singleton.singleton n).max ↑n
  -/
  exact Finset.max_singleton
  /-
    🎉 no goals
  -/


theorem degree_C_mul_T_ite [DecidableEq R] (n : ℤ) (a : R) :
    degree (C a * T n) = if a = 0 then ⊥ else ↑n := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n : Int
    a : R
    ⊢ Eq (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomial.T n)).degree (ite ( …
  -/
  split_ifs with h <;>
    simp only [h, map_zero, zero_mul, degree_zero, degree_C_mul_T, Ne,
      not_false_iff]


@[simp]
theorem degree_T [Nontrivial R] (n : ℤ) : (T n : R[T;T⁻¹]).degree = n := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    n : Int
    ⊢ Eq (LaurentPolynomial.T n).degree ↑n
  -/
  rw [← one_mul (T n), ← map_one C]
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    n : Int
    ⊢ Eq (HMul.hMul (LaurentPolynomial.C 1) (LaurentPolynomial.T n)).degree ↑n
  -/
  exact degree_C_mul_T n 1 (one_ne_zero : (1 : R) ≠ 0)
  /-
    🎉 no goals
  -/


theorem degree_C {a : R} (a0 : a ≠ 0) : (C a).degree = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    a0 : Ne a 0
    ⊢ Eq (LaurentPolynomial.C a).degree 0
  -/
  rw [← mul_one (C a), ← T_zero]
  /-
    R : Type u_1
    inst✝ : Semiring R
    a : R
    a0 : Ne a 0
    ⊢ Eq (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomial.T 0)).degree 0
  -/
  exact degree_C_mul_T 0 a a0
  /-
    🎉 no goals
  -/


theorem degree_C_ite [DecidableEq R] (a : R) : (C a).degree = if a = 0 then ⊥ else 0 := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    a : R
    ⊢ Eq (LaurentPolynomial.C a).degree (ite (Eq a 0) Bot.bot 0)
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp only [h, map_zero, degree_zero, degree_C, Ne, not_false_iff]
                       /-
                         🎉 no goals
                       -/


theorem degree_C_mul_T_le (n : ℤ) (a : R) : degree (C a * T n) ≤ n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Int
    a : R
    ⊢ LE.le (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomial.T n)).degree ↑n
  -/
  by_cases a0 : a = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      n : Int
      a : R
      a0 : Eq a 0
      ⊢ LE.le (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomial.T n)).degree ↑n
    -/
  · simp only [a0, map_zero, zero_mul, degree_zero, bot_le]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      n : Int
      a : R
      a0 : Not (Eq a 0)
      ⊢ LE.le (HMul.hMul (LaurentPolynomial.C a) (LaurentPolynomial.T n)).degree ↑n
    -/
  · exact (degree_C_mul_T n a a0).le
    /-
      🎉 no goals
    -/


theorem degree_T_le (n : ℤ) : (T n : R[T;T⁻¹]).degree ≤ n :=
                /-
                  R : Type u_1
                  inst✝ : Semiring R
                  n : Int
                  ⊢ Eq (LaurentPolynomial.T n).degree (HMul.hMul (LaurentPolynomial.C 1) (Lauren …
                -/
  (le_of_eq (by rw [map_one, one_mul])).trans (degree_C_mul_T_le n (1 : R))
                /-
                  🎉 no goals
                -/


theorem degree_C_le (a : R) : (C a).degree ≤ 0 :=
                /-
                  R : Type u_1
                  inst✝ : Semiring R
                  a : R
                  ⊢ Eq (LaurentPolynomial.C a).degree (HMul.hMul (LaurentPolynomial.C a) (Lauren …
                -/
  (le_of_eq (by rw [T_zero, mul_one])).trans (degree_C_mul_T_le 0 a)
                /-
                  🎉 no goals
                -/


instance : Module R[X] R[T;T⁻¹] :=
  Module.compHom _ Polynomial.toLaurent


instance (R : Type*) [Semiring R] : IsScalarTower R[X] R[X] R[T;T⁻¹] where
                         /-
                           R✝ : Type u_1
                           S : Type u_2
                           inst✝¹ : Semiring R✝
                           R : Type u_3
                           inst✝ : Semiring R
                           x y : Polynomial R
                           z : LaurentPolynomial R
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
                         -/
  smul_assoc x y z := by dsimp; simp_rw [MulAction.mul_smul]
                                /-
                                  🎉 no goals
                                -/


instance algebraPolynomial (R : Type*) [CommSemiring R] : Algebra R[X] R[T;T⁻¹] :=
  { Polynomial.toLaurent with
                               /-
                                 R✝ : Type u_1
                                 S : Type u_2
                                 inst✝¹ : CommSemiring R✝
                                 R : Type u_3
                                 inst✝ : CommSemiring R
                                 f : Polynomial R
                                 l : LaurentPolynomial R
                                 ⊢ Eq (HMul.hMul (__src✝ f) l) (HMul.hMul l (__src✝ f))
                               -/
    commutes' := fun f l => by simp [mul_comm]
                               /-
                                 🎉 no goals
                               -/
    smul_def' := fun _ _ => rfl }


theorem algebraMap_X_pow (n : ℕ) : algebraMap R[X] R[T;T⁻¹] (X ^ n) = T n :=
  Polynomial.toLaurent_X_pow n


@[simp]
theorem algebraMap_eq_toLaurent (f : R[X]) : algebraMap R[X] R[T;T⁻¹] f = toLaurent f :=
  rfl


theorem isLocalization : IsLocalization (Submonoid.powers (X : R[X])) R[T;T⁻¹] :=
  { map_units' := fun ⟨t, ht⟩ => by
      /-
        R : Type u_1
        inst✝ : CommSemiring R
        x✝ : Subtype fun x => Membership.mem (Submonoid.powers Polynomial.X) x
        t : Polynomial R
        ht : Membership.mem (Submonoid.powers Polynomial.X) t
        ⊢ IsUnit ((algebraMap (Polynomial R) (LaurentPolynomial R)) ↑⟨t, ht⟩)
      -/
      obtain ⟨n, rfl⟩ := ht
      /-
        case intro
        R : Type u_1
        inst✝ : CommSemiring R
        x✝ : Subtype fun x => Membership.mem (Submonoid.powers Polynomial.X) x
        n : Nat
        ⊢ IsUnit ((algebraMap (Polynomial R) (LaurentPolynomial R)) ↑⟨(fun x => HPow.h …
      -/
      rw [algebraMap_eq_toLaurent, toLaurent_X_pow]
      /-
        case intro
        R : Type u_1
        inst✝ : CommSemiring R
        x✝ : Subtype fun x => Membership.mem (Submonoid.powers Polynomial.X) x
        n : Nat
        ⊢ IsUnit (LaurentPolynomial.T ↑n)
      -/
      exact isUnit_T ↑n
      /-
        🎉 no goals
      -/
    surj' := fun f => by
      /-
        R : Type u_1
        inst✝ : CommSemiring R
        f : LaurentPolynomial R
        ⊢ Exists fun x => Eq (HMul.hMul f ((algebraMap (Polynomial R) (LaurentPolynomi …
      -/
      induction' f using LaurentPolynomial.induction_on_mul_T with f n
      /-
        case Qf
        R : Type u_1
        inst✝ : CommSemiring R
        f : Polynomial R
        n : Nat
        ⊢ Exists fun x => Eq (HMul.hMul (HMul.hMul (Polynomial.toLaurent f) (LaurentPo …
      -/
      have : X ^ n ∈ Submonoid.powers (X : R[X]) := ⟨n, rfl⟩
      /-
        case Qf
        R : Type u_1
        inst✝ : CommSemiring R
        f : Polynomial R
        n : Nat
        this : Membership.mem (Submonoid.powers Polynomial.X) (HPow.hPow Polynomial.X n)
        ⊢ Exists fun x => Eq (HMul.hMul (HMul.hMul (Polynomial.toLaurent f) (LaurentPo …
      -/
      refine ⟨(f, ⟨_, this⟩), ?_⟩
      simp only [algebraMap_eq_toLaurent, toLaurent_X_pow, mul_T_assoc, neg_add_cancel, T_zero,
        mul_one]
    exists_of_eq := fun {f g} => by
      /-
        R : Type u_1
        inst✝ : CommSemiring R
        f g : Polynomial R
        ⊢ Eq ((algebraMap (Polynomial R) (LaurentPolynomial R)) f) ((algebraMap (Polyn …
      -/
      rw [algebraMap_eq_toLaurent, algebraMap_eq_toLaurent, Polynomial.toLaurent_inj]
      /-
        R : Type u_1
        inst✝ : CommSemiring R
        f g : Polynomial R
        ⊢ Eq f g → Exists fun c => Eq (HMul.hMul (↑c) f) (HMul.hMul (↑c) g)
      -/
      rintro rfl
      /-
        R : Type u_1
        inst✝ : CommSemiring R
        f : Polynomial R
        ⊢ Exists fun c => Eq (HMul.hMul (↑c) f) (HMul.hMul (↑c) f)
      -/
      exact ⟨1, rfl⟩ }
      /-
        🎉 no goals
      -/


/-- The map which substitutes `T ↦ T⁻¹` into a Laurent polynomial. -/
def invert : R[T;T⁻¹] ≃ₐ[R] R[T;T⁻¹] := AddMonoidAlgebra.domCongr R R <| AddEquiv.neg _


@[simp] lemma invert_T (n : ℤ) : invert (T n : R[T;T⁻¹]) = T (-n) :=
  AddMonoidAlgebra.domCongr_single _ _ _ _ _


@[simp] lemma invert_apply (f : R[T;T⁻¹]) (n : ℤ) : invert f n = f (-n) := rfl


                                                          /-
                                                            R : Type u_3
                                                            inst✝ : CommSemiring R
                                                            ⊢ Eq (Function.comp ⇑LaurentPolynomial.invert ⇑LaurentPolynomial.C) ⇑LaurentPo …
                                                          -/
@[simp] lemma invert_comp_C : invert ∘ (@C R _) = C := by ext; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                          /-
                                                            R : Type u_3
                                                            inst✝ : CommSemiring R
                                                            t : R
                                                            ⊢ Eq (LaurentPolynomial.invert (LaurentPolynomial.C t)) (LaurentPolynomial.C t)
                                                          -/
@[simp] lemma invert_C (t : R) : invert (C t) = C t := by ext; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                                     /-
                                                                       R : Type u_3
                                                                       inst✝ : CommSemiring R
                                                                       x✝ : LaurentPolynomial R
                                                                       ⊢ Eq (LaurentPolynomial.invert (LaurentPolynomial.invert x✝)) x✝
                                                                     -/
lemma involutive_invert : Involutive (invert (R := R)) := fun _ ↦ by ext; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp] lemma invert_symm : (invert (R := R)).symm = invert := rfl


lemma toLaurent_reverse (p : R[X]) :
    toLaurent p.reverse = invert (toLaurent p) * (T p.natDegree) := by
  /-
    R : Type u_3
    inst✝ : CommSemiring R
    p : Polynomial R
    ⊢ Eq (Polynomial.toLaurent p.reverse) (HMul.hMul (LaurentPolynomial.invert (Po …
  -/
  nontriviality R
  induction p using Polynomial.recOnHorner with
  | M0 => simp
  | MC _ _ _ _ ih => simp [add_mul, ← ih]
  | MX _ hp => simpa [natDegree_mul_X hp]


/-- Evaluate a Laurent polynomial at a unit, using scalar multiplication. -/
def smeval : S := Finsupp.sum f fun n r => r • (x ^ n).val


theorem smeval_eq_sum : f.smeval x = Finsupp.sum f fun n r => r • (x ^ n).val := rfl


                                                                     /-
                                                                       R : Type u_1
                                                                       S : Type u_2
                                                                       inst✝³ : Semiring R
                                                                       inst✝² : AddCommMonoid S
                                                                       inst✝¹ : SMulWithZero R S
                                                                       inst✝ : Monoid S
                                                                       f g : LaurentPolynomial R
                                                                       x y : Units S
                                                                       ⊢ Eq f g → Eq x y → Eq (f.smeval x) (g.smeval y)
                                                                     -/
theorem smeval_congr : f = g → x = y → f.smeval x = g.smeval y := by rintro rfl rfl; rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem smeval_zero : (0 : R[T;T⁻¹]).smeval x = (0 : S) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : SMulWithZero R S
    inst✝ : Monoid S
    x : Units S
    ⊢ Eq (LaurentPolynomial.smeval 0 x) 0
  -/
  simp only [smeval_eq_sum, Finsupp.sum_zero_index]
  /-
    🎉 no goals
  -/


theorem smeval_single (n : ℤ) (r : R) : smeval (Finsupp.single n r) x = r • (x ^ n).val := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : SMulWithZero R S
    inst✝ : Monoid S
    x : Units S
    n : Int
    r : R
    ⊢ Eq (LaurentPolynomial.smeval (Finsupp.single n r) x) (HSMul.hSMul r ↑(HPow.h …
  -/
  simp only [smeval_eq_sum]
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : SMulWithZero R S
    inst✝ : Monoid S
    x : Units S
    n : Int
    r : R
    ⊢ Eq ((Finsupp.single n r).sum fun n r => HSMul.hSMul r ↑(HPow.hPow x n)) (HSM …
  -/
  rw [Finsupp.sum_single_index (zero_smul R (x ^ n).val)]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_C_mul_T_n (n : ℤ) (r : R) : (C r * T n).smeval x = r • (x ^ n).val := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : SMulWithZero R S
    inst✝ : Monoid S
    x : Units S
    n : Int
    r : R
    ⊢ Eq ((HMul.hMul (LaurentPolynomial.C r) (LaurentPolynomial.T n)).smeval x) (H …
  -/
  rw [← single_eq_C_mul_T, smeval_single]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_C (r : R) : (C r).smeval x = r • 1 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : SMulWithZero R S
    inst✝ : Monoid S
    x : Units S
    r : R
    ⊢ Eq ((LaurentPolynomial.C r).smeval x) (HSMul.hSMul r 1)
  -/
  rw [← single_eq_C, smeval_single x (0 : ℤ) r, zpow_zero, Units.val_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_T_pow (n : ℤ) (x : Sˣ) : (T n : R[T;T⁻¹]).smeval x = (x ^ n).val := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : MulActionWithZero R S
    inst✝ : Monoid S
    n : Int
    x : Units S
    ⊢ Eq ((LaurentPolynomial.T n).smeval x) ↑(HPow.hPow x n)
  -/
  rw [T, smeval_single, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_one : (1 : R[T;T⁻¹]).smeval x = 1 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : MulActionWithZero R S
    inst✝ : Monoid S
    x : Units S
    ⊢ Eq (LaurentPolynomial.smeval 1 x) 1
  -/
  rw [← T_zero, smeval_T_pow 0 x, zpow_zero, Units.val_eq_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_add : (f + g).smeval x = f.smeval x + g.smeval x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : Module R S
    inst✝ : Monoid S
    f g : LaurentPolynomial R
    x : Units S
    ⊢ Eq ((HAdd.hAdd f g).smeval x) (HAdd.hAdd (f.smeval x) (g.smeval x))
  -/
  simp only [smeval_eq_sum]
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid S
    inst✝¹ : Module R S
    inst✝ : Monoid S
    f g : LaurentPolynomial R
    x : Units S
    ⊢ Eq (Finsupp.sum (HAdd.hAdd f g) fun n r => HSMul.hSMul r ↑(HPow.hPow x n)) ( …
  -/
  rw [Finsupp.sum_add_index (fun n _ => zero_smul R (x ^ n).val) (fun n _ r r' => add_smul r r' _)]
  /-
    🎉 no goals
  -/


@[simp]
theorem smeval_C_mul (r : R) : (C r * f).smeval x = r • (f.smeval x) := by
  induction f using LaurentPolynomial.induction_on' with
  | h_add p q hp hq=>
    rw [mul_add, smeval_add, smeval_add, smul_add, hp, hq]
  | h_C_mul_T n s =>
    rw [← mul_assoc, ← map_mul, smeval_C_mul_T_n, smeval_C_mul_T_n, mul_smul]


variable (R) in
/-- Evaluation as an `R`-linear map. -/
@[simps]
def leval : R[T;T⁻¹] →ₗ[R] S where
  toFun f := f.smeval x
  map_add' f g := smeval_add f g x
                      /-
                        R : Type u_1
                        S : Type u_2
                        inst✝³ : Semiring R
                        inst✝² : AddCommMonoid S
                        inst✝¹ : Module R S
                        inst✝ : Monoid S
                        f✝ g : LaurentPolynomial R
                        x y : Units S
                        r : R
                        f : LaurentPolynomial R
                        ⊢ Eq ({ toFun := fun f => f.smeval x, map_add' := ⋯ }.toFun (HSMul.hSMul r f)) …
                      -/
  map_smul' r f := by simp [smul_eq_C_mul]
                      /-
                        🎉 no goals
                      -/


