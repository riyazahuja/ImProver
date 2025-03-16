/-- The Taylor expansion of a polynomial `f` at `r`. -/
def taylor (r : R) : R[X] →ₗ[R] R[X] where
  toFun f := f.comp (X + C r)
  map_add' _ _ := add_comp
                      /-
                        R : Type u_1
                        inst✝ : Semiring R
                        r✝ : R
                        f✝ : Polynomial R
                        r c : R
                        f : Polynomial R
                        ⊢ Eq ({ toFun := fun f => f.comp (HAdd.hAdd Polynomial.X (Polynomial.C r)), ma …
                      -/
  map_smul' c f := by simp only [smul_eq_C_mul, C_mul_comp, RingHom.id_apply]
                      /-
                        🎉 no goals
                      -/


theorem taylor_apply : taylor r f = f.comp (X + C r) :=
  rfl


@[simp]
                                              /-
                                                R : Type u_1
                                                inst✝ : Semiring R
                                                r : R
                                                ⊢ Eq ((Polynomial.taylor r) Polynomial.X) (HAdd.hAdd Polynomial.X (Polynomial. …
                                              -/
theorem taylor_X : taylor r X = X + C r := by simp only [taylor_apply, X_comp]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
                                                      /-
                                                        R : Type u_1
                                                        inst✝ : Semiring R
                                                        r x : R
                                                        ⊢ Eq ((Polynomial.taylor r) (Polynomial.C x)) (Polynomial.C x)
                                                      -/
theorem taylor_C (x : R) : taylor r (C x) = C x := by simp only [taylor_apply, C_comp]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem taylor_zero' : taylor (0 : R) = LinearMap.id := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.taylor 0) LinearMap.id
  -/
  ext
  simp only [taylor_apply, add_zero, comp_X, _root_.map_zero, LinearMap.id_comp,
    Function.comp_apply, LinearMap.coe_comp]


                                                      /-
                                                        R : Type u_1
                                                        inst✝ : Semiring R
                                                        f : Polynomial R
                                                        ⊢ Eq ((Polynomial.taylor 0) f) f
                                                      -/
theorem taylor_zero (f : R[X]) : taylor 0 f = f := by rw [taylor_zero', LinearMap.id_apply]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                                     /-
                                                       R : Type u_1
                                                       inst✝ : Semiring R
                                                       r : R
                                                       ⊢ Eq ((Polynomial.taylor r) 1) (Polynomial.C 1)
                                                     -/
theorem taylor_one : taylor r (1 : R[X]) = C 1 := by rw [← C_1, taylor_C]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem taylor_monomial (i : ℕ) (k : R) : taylor r (monomial i k) = C k * (X + C r) ^ i := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    i : Nat
    k : R
    ⊢ Eq ((Polynomial.taylor r) ((Polynomial.monomial i) k)) (HMul.hMul (Polynomia …
  -/
  simp [taylor_apply]
  /-
    🎉 no goals
  -/


/-- The `k`th coefficient of `Polynomial.taylor r f` is `(Polynomial.hasseDeriv k f).eval r`. -/
theorem taylor_coeff (n : ℕ) : (taylor r f).coeff n = (hasseDeriv n f).eval r :=
  show (lcoeff R n).comp (taylor r) f = (leval r).comp (hasseDeriv n) f by
    /-
      R : Type u_1
      inst✝ : Semiring R
      r : R
      f : Polynomial R
      n : Nat
      ⊢ Eq (((Polynomial.lcoeff R n).comp (Polynomial.taylor r)) f) (((Polynomial.le …
    -/
    congr 1; clear! f; ext i
    simp only [leval_apply, mul_one, one_mul, eval_monomial, LinearMap.comp_apply, coeff_C_mul,
      hasseDeriv_monomial, taylor_apply, monomial_comp, C_1, (commute_X (C r)).add_pow i,
      map_sum]
    simp only [lcoeff_apply, ← C_eq_natCast, mul_assoc, ← C_pow, ← C_mul, coeff_mul_C,
      (Nat.cast_commute _ _).eq, coeff_X_pow, boole_mul, Finset.sum_ite_eq, Finset.mem_range]
    /-
      case e_a.h.h
      R : Type u_1
      inst✝ : Semiring R
      r : R
      n i : Nat
      ⊢ Eq (ite (LT.lt n (HAdd.hAdd i 1)) (HMul.hMul (HPow.hPow r (HSub.hSub i n)) ↑ …
    -/
    split_ifs with h; · rfl
                        /-
                          🎉 no goals
                        -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      r : R
      n i : Nat
      h : Not (LT.lt n (HAdd.hAdd i 1))
      ⊢ Eq 0 (HMul.hMul (HPow.hPow r (HSub.hSub i n)) ↑(i.choose n))
    -/
    push_neg at h; rw [Nat.choose_eq_zero_of_lt h, Nat.cast_zero, mul_zero]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem taylor_coeff_zero : (taylor r f).coeff 0 = f.eval r := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    f : Polynomial R
    ⊢ Eq (((Polynomial.taylor r) f).coeff 0) (Polynomial.eval r f)
  -/
  rw [taylor_coeff, hasseDeriv_zero, LinearMap.id_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem taylor_coeff_one : (taylor r f).coeff 1 = f.derivative.eval r := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    f : Polynomial R
    ⊢ Eq (((Polynomial.taylor r) f).coeff 1) (Polynomial.eval r (Polynomial.deriva …
  -/
  rw [taylor_coeff, hasseDeriv_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_taylor (p : R[X]) (r : R) : natDegree (taylor r p) = natDegree p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    r : R
    ⊢ Eq ((Polynomial.taylor r) p).natDegree p.natDegree
  -/
  refine map_natDegree_eq_natDegree _ ?_
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    r : R
    ⊢ ∀ (n : Nat) (c : R), Ne c 0 → Eq ((Polynomial.taylor r) ((Polynomial.monomia …
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    r : R
    a✝ : Nontrivial R
    ⊢ ∀ (n : Nat) (c : R), Ne c 0 → Eq ((Polynomial.taylor r) ((Polynomial.monomia …
  -/
  intro n c c0
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    r : R
    a✝ : Nontrivial R
    n : Nat
    c : R
    c0 : Ne c 0
    ⊢ Eq ((Polynomial.taylor r) ((Polynomial.monomial n) c)).natDegree n
  -/
  simp [taylor_monomial, natDegree_C_mul_of_mul_ne_zero, natDegree_pow_X_add_C, c0]
  /-
    🎉 no goals
  -/


@[simp]
theorem taylor_mul {R} [CommSemiring R] (r : R) (p q : R[X]) :
                                                     /-
                                                       R : Type u_2
                                                       inst✝ : CommSemiring R
                                                       r : R
                                                       p q : Polynomial R
                                                       ⊢ Eq ((Polynomial.taylor r) (HMul.hMul p q)) (HMul.hMul ((Polynomial.taylor r) …
                                                     -/
    taylor r (p * q) = taylor r p * taylor r q := by simp only [taylor_apply, mul_comp]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- `Polynomial.taylor` as an `AlgHom` for commutative semirings -/
@[simps!]
def taylorAlgHom {R} [CommSemiring R] (r : R) : R[X] →ₐ[R] R[X] :=
  AlgHom.ofLinearMap (taylor r) (taylor_one r) (taylor_mul r)


theorem taylor_taylor {R} [CommSemiring R] (f : R[X]) (r s : R) :
    taylor r (taylor s f) = taylor (r + s) f := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    f : Polynomial R
    r s : R
    ⊢ Eq ((Polynomial.taylor r) ((Polynomial.taylor s) f)) ((Polynomial.taylor (HA …
  -/
  simp only [taylor_apply, comp_assoc, map_add, add_comp, X_comp, C_comp, C_add, add_assoc]
  /-
    🎉 no goals
  -/


theorem taylor_eval {R} [CommSemiring R] (r : R) (f : R[X]) (s : R) :
    (taylor r f).eval s = f.eval (s + r) := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    r : R
    f : Polynomial R
    s : R
    ⊢ Eq (Polynomial.eval s ((Polynomial.taylor r) f)) (Polynomial.eval (HAdd.hAdd …
  -/
  simp only [taylor_apply, eval_comp, eval_C, eval_X, eval_add]
  /-
    🎉 no goals
  -/


theorem taylor_eval_sub {R} [CommRing R] (r : R) (f : R[X]) (s : R) :
                                               /-
                                                 R : Type u_2
                                                 inst✝ : CommRing R
                                                 r : R
                                                 f : Polynomial R
                                                 s : R
                                                 ⊢ Eq (Polynomial.eval (HSub.hSub s r) ((Polynomial.taylor r) f)) (Polynomial.e …
                                               -/
    (taylor r f).eval (s - r) = f.eval s := by rw [taylor_eval, sub_add_cancel]
                                               /-
                                                 🎉 no goals
                                               -/


theorem taylor_injective {R} [CommRing R] (r : R) : Function.Injective (taylor r) := by
  /-
    R : Type u_2
    inst✝ : CommRing R
    r : R
    ⊢ Function.Injective ⇑(Polynomial.taylor r)
  -/
  intro f g h
  /-
    R : Type u_2
    inst✝ : CommRing R
    r : R
    f g : Polynomial R
    h : Eq ((Polynomial.taylor r) f) ((Polynomial.taylor r) g)
    ⊢ Eq f g
  -/
  apply_fun taylor (-r) at h
  simpa only [taylor_apply, comp_assoc, add_comp, X_comp, C_comp, C_neg, neg_add_cancel_right,
    comp_X] using h


theorem eq_zero_of_hasseDeriv_eq_zero {R} [CommRing R] (f : R[X]) (r : R)
    (h : ∀ k, (hasseDeriv k f).eval r = 0) : f = 0 := by
  /-
    R : Type u_2
    inst✝ : CommRing R
    f : Polynomial R
    r : R
    h : ∀ (k : Nat), Eq (Polynomial.eval r ((Polynomial.hasseDeriv k) f)) 0
    ⊢ Eq f 0
  -/
  apply taylor_injective r
  /-
    case a
    R : Type u_2
    inst✝ : CommRing R
    f : Polynomial R
    r : R
    h : ∀ (k : Nat), Eq (Polynomial.eval r ((Polynomial.hasseDeriv k) f)) 0
    ⊢ Eq ((Polynomial.taylor r) f) ((Polynomial.taylor r) 0)
  -/
  rw [LinearMap.map_zero]
  /-
    case a
    R : Type u_2
    inst✝ : CommRing R
    f : Polynomial R
    r : R
    h : ∀ (k : Nat), Eq (Polynomial.eval r ((Polynomial.hasseDeriv k) f)) 0
    ⊢ Eq ((Polynomial.taylor r) f) 0
  -/
  ext k
  /-
    case a.a
    R : Type u_2
    inst✝ : CommRing R
    f : Polynomial R
    r : R
    h : ∀ (k : Nat), Eq (Polynomial.eval r ((Polynomial.hasseDeriv k) f)) 0
    k : Nat
    ⊢ Eq (((Polynomial.taylor r) f).coeff k) (Polynomial.coeff 0 k)
  -/
  simp only [taylor_coeff, h, coeff_zero]
  /-
    🎉 no goals
  -/


/-- Taylor's formula. -/
theorem sum_taylor_eq {R} [CommRing R] (f : R[X]) (r : R) :
    ((taylor r f).sum fun i a => C a * (X - C r) ^ i) = f := by
  rw [← comp_eq_sum_left, sub_eq_add_neg, ← C_neg, ← taylor_apply, taylor_taylor, neg_add_cancel,
    taylor_zero]


theorem eval_add_of_sq_eq_zero {A} [CommSemiring A] (p : Polynomial A) (x y : A) (hy : y ^ 2 = 0) :
    p.eval (x + y) = p.eval x + p.derivative.eval x * y := by
  rw [add_comm, ← Polynomial.taylor_eval,
    Polynomial.eval_eq_sum_range' ((Nat.lt_succ_self _).trans (Nat.lt_succ_self _)),
    Finset.sum_range_succ', Finset.sum_range_succ']
  /-
    A : Type u_2
    inst✝ : CommSemiring A
    p : Polynomial A
    x y : A
    hy : Eq (HPow.hPow y 2) 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((Finset.range ((Polynomial.taylor x) p).natDegree) …
  -/
  simp [pow_succ, mul_assoc, ← pow_two, hy, add_comm (eval x p)]
  /-
    🎉 no goals
  -/


