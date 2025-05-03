/-- `RatFunc.C a` is the constant rational function `a`. -/
def C : K →+* RatFunc K := algebraMap _ _


@[simp]
theorem algebraMap_eq_C : algebraMap K (RatFunc K) = C :=
  rfl


@[simp]
theorem algebraMap_C (a : K) : algebraMap K[X] (RatFunc K) (Polynomial.C a) = C a :=
  rfl


@[simp]
theorem algebraMap_comp_C : (algebraMap K[X] (RatFunc K)).comp Polynomial.C = C :=
  rfl


theorem smul_eq_C_mul (r : K) (x : RatFunc K) : r • x = C r * x := by
  /-
    K : Type u
    inst✝¹ : CommRing K
    inst✝ : IsDomain K
    r : K
    x : RatFunc K
    ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (RatFunc.C r) x)
  -/
  rw [Algebra.smul_def, algebraMap_eq_C]
  /-
    🎉 no goals
  -/


/-- `RatFunc.X` is the polynomial variable (aka indeterminate). -/
def X : RatFunc K :=
  algebraMap K[X] (RatFunc K) Polynomial.X


@[simp]
theorem algebraMap_X : algebraMap K[X] (RatFunc K) Polynomial.X = X :=
  rfl


@[simp]
theorem num_C (c : K) : num (C c) = Polynomial.C c :=
  num_algebraMap _


@[simp]
theorem denom_C (c : K) : denom (C c) = 1 :=
  denom_algebraMap _


@[simp]
theorem num_X : num (X : RatFunc K) = Polynomial.X :=
  num_algebraMap _


@[simp]
theorem denom_X : denom (X : RatFunc K) = 1 :=
  denom_algebraMap _


theorem X_ne_zero : (X : RatFunc K) ≠ 0 :=
  RatFunc.algebraMap_ne_zero Polynomial.X_ne_zero


/-- Evaluate a rational function `p` given a ring hom `f` from the scalar field
to the target and a value `x` for the variable in the target.

Fractions are reduced by clearing common denominators before evaluating:
`eval id 1 ((X^2 - 1) / (X - 1)) = eval id 1 (X + 1) = 2`, not `0 / 0 = 0`.
-/
def eval (f : K →+* L) (a : L) (p : RatFunc K) : L :=
  (num p).eval₂ f a / (denom p).eval₂ f a


theorem eval_eq_zero_of_eval₂_denom_eq_zero {x : RatFunc K}
                                                                    /-
                                                                      K : Type u
                                                                      inst✝¹ : Field K
                                                                      L : Type u
                                                                      inst✝ : Field L
                                                                      f : RingHom K L
                                                                      a : L
                                                                      x : RatFunc K
                                                                      h : Eq (Polynomial.eval₂ f a x.denom) 0
                                                                      ⊢ Eq (RatFunc.eval f a x) 0
                                                                    -/
    (h : Polynomial.eval₂ f a (denom x) = 0) : eval f a x = 0 := by rw [eval, h, div_zero]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem eval₂_denom_ne_zero {x : RatFunc K} (h : eval f a x ≠ 0) :
    Polynomial.eval₂ f a (denom x) ≠ 0 :=
  mt eval_eq_zero_of_eval₂_denom_eq_zero h


@[simp]
                                                    /-
                                                      K : Type u
                                                      inst✝¹ : Field K
                                                      L : Type u
                                                      inst✝ : Field L
                                                      f : RingHom K L
                                                      a : L
                                                      c : K
                                                      ⊢ Eq (RatFunc.eval f a (RatFunc.C c)) (f c)
                                                    -/
theorem eval_C {c : K} : eval f a (C c) = f c := by simp [eval]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
                                      /-
                                        K : Type u
                                        inst✝¹ : Field K
                                        L : Type u
                                        inst✝ : Field L
                                        f : RingHom K L
                                        a : L
                                        ⊢ Eq (RatFunc.eval f a RatFunc.X) a
                                      -/
theorem eval_X : eval f a X = a := by simp [eval]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
                                         /-
                                           K : Type u
                                           inst✝¹ : Field K
                                           L : Type u
                                           inst✝ : Field L
                                           f : RingHom K L
                                           a : L
                                           ⊢ Eq (RatFunc.eval f a 0) 0
                                         -/
theorem eval_zero : eval f a 0 = 0 := by simp [eval]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
                                        /-
                                          K : Type u
                                          inst✝¹ : Field K
                                          L : Type u
                                          inst✝ : Field L
                                          f : RingHom K L
                                          a : L
                                          ⊢ Eq (RatFunc.eval f a 1) 1
                                        -/
theorem eval_one : eval f a 1 = 1 := by simp [eval]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem eval_algebraMap {S : Type*} [CommSemiring S] [Algebra S K[X]] (p : S) :
    eval f a (algebraMap _ _ p) = (algebraMap _ K[X] p).eval₂ f a := by
  /-
    K : Type u
    inst✝³ : Field K
    L : Type u
    inst✝² : Field L
    f : RingHom K L
    a : L
    S : Type u_1
    inst✝¹ : CommSemiring S
    inst✝ : Algebra S (Polynomial K)
    p : S
    ⊢ Eq (RatFunc.eval f a ((algebraMap S (RatFunc K)) p)) (Polynomial.eval₂ f a ( …
  -/
  simp [eval, IsScalarTower.algebraMap_apply S K[X] (RatFunc K)]
  /-
    🎉 no goals
  -/


/-- `eval` is an additive homomorphism except when a denominator evaluates to `0`.

Counterexample: `eval _ 1 (X / (X-1)) + eval _ 1 (-1 / (X-1)) = 0`
`... ≠ 1 = eval _ 1 ((X-1) / (X-1))`.

See also `RatFunc.eval₂_denom_ne_zero` to make the hypotheses simpler but less general.
-/
theorem eval_add {x y : RatFunc K} (hx : Polynomial.eval₂ f a (denom x) ≠ 0)
    (hy : Polynomial.eval₂ f a (denom y) ≠ 0) : eval f a (x + y) = eval f a x + eval f a y := by
  /-
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    ⊢ Eq (RatFunc.eval f a (HAdd.hAdd x y)) (HAdd.hAdd (RatFunc.eval f a x) (RatFu …
  -/
  unfold eval
  /-
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    ⊢ Eq (HDiv.hDiv (Polynomial.eval₂ f a (HAdd.hAdd x y).num) (Polynomial.eval₂ f …
  -/
  by_cases hxy : Polynomial.eval₂ f a (denom (x + y)) = 0
    /-
      case pos
      K : Type u
      inst✝¹ : Field K
      L : Type u
      inst✝ : Field L
      f : RingHom K L
      a : L
      x y : RatFunc K
      hx : Ne (Polynomial.eval₂ f a x.denom) 0
      hy : Ne (Polynomial.eval₂ f a y.denom) 0
      hxy : Eq (Polynomial.eval₂ f a (HAdd.hAdd x y).denom) 0
      ⊢ Eq (HDiv.hDiv (Polynomial.eval₂ f a (HAdd.hAdd x y).num) (Polynomial.eval₂ f …
    -/
  · have := Polynomial.eval₂_eq_zero_of_dvd_of_eval₂_eq_zero f a (denom_add_dvd x y) hxy
    /-
      case pos
      K : Type u
      inst✝¹ : Field K
      L : Type u
      inst✝ : Field L
      f : RingHom K L
      a : L
      x y : RatFunc K
      hx : Ne (Polynomial.eval₂ f a x.denom) 0
      hy : Ne (Polynomial.eval₂ f a y.denom) 0
      hxy : Eq (Polynomial.eval₂ f a (HAdd.hAdd x y).denom) 0
      this : Eq (Polynomial.eval₂ f a (HMul.hMul x.denom y.denom)) 0
      ⊢ Eq (HDiv.hDiv (Polynomial.eval₂ f a (HAdd.hAdd x y).num) (Polynomial.eval₂ f …
    -/
    rw [Polynomial.eval₂_mul] at this
    /-
      case pos
      K : Type u
      inst✝¹ : Field K
      L : Type u
      inst✝ : Field L
      f : RingHom K L
      a : L
      x y : RatFunc K
      hx : Ne (Polynomial.eval₂ f a x.denom) 0
      hy : Ne (Polynomial.eval₂ f a y.denom) 0
      hxy : Eq (Polynomial.eval₂ f a (HAdd.hAdd x y).denom) 0
      this : Eq (HMul.hMul (Polynomial.eval₂ f a x.denom) (Polynomial.eval₂ f a y.de …
      ⊢ Eq (HDiv.hDiv (Polynomial.eval₂ f a (HAdd.hAdd x y).num) (Polynomial.eval₂ f …
    -/
                                  /-
                                    🎉 no goals
                                  -/
    cases mul_eq_zero.mp this <;> contradiction
                                  /-
                                    🎉 no goals
                                  -/
  rw [div_add_div _ _ hx hy, eq_div_iff (mul_ne_zero hx hy), div_eq_mul_inv, mul_right_comm, ←
    div_eq_mul_inv, div_eq_iff hxy]
  /-
    case neg
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    hxy : Not (Eq (Polynomial.eval₂ f a (HAdd.hAdd x y).denom) 0)
    ⊢ Eq (HMul.hMul (Polynomial.eval₂ f a (HAdd.hAdd x y).num) (HMul.hMul (Polynom …
  -/
  simp only [← Polynomial.eval₂_mul, ← Polynomial.eval₂_add]
  /-
    case neg
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    hxy : Not (Eq (Polynomial.eval₂ f a (HAdd.hAdd x y).denom) 0)
    ⊢ Eq (Polynomial.eval₂ f a (HMul.hMul (HAdd.hAdd x y).num (HMul.hMul x.denom y …
  -/
  congr 1
  /-
    case neg.e_p
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    hxy : Not (Eq (Polynomial.eval₂ f a (HAdd.hAdd x y).denom) 0)
    ⊢ Eq (HMul.hMul (HAdd.hAdd x y).num (HMul.hMul x.denom y.denom)) (HMul.hMul (H …
  -/
  apply num_denom_add
  /-
    🎉 no goals
  -/


/-- `eval` is a multiplicative homomorphism except when a denominator evaluates to `0`.

Counterexample: `eval _ 0 X * eval _ 0 (1/X) = 0 ≠ 1 = eval _ 0 1 = eval _ 0 (X * 1/X)`.

See also `RatFunc.eval₂_denom_ne_zero` to make the hypotheses simpler but less general.
-/
theorem eval_mul {x y : RatFunc K} (hx : Polynomial.eval₂ f a (denom x) ≠ 0)
    (hy : Polynomial.eval₂ f a (denom y) ≠ 0) : eval f a (x * y) = eval f a x * eval f a y := by
  /-
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    ⊢ Eq (RatFunc.eval f a (HMul.hMul x y)) (HMul.hMul (RatFunc.eval f a x) (RatFu …
  -/
  unfold eval
  /-
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    ⊢ Eq (HDiv.hDiv (Polynomial.eval₂ f a (HMul.hMul x y).num) (Polynomial.eval₂ f …
  -/
  by_cases hxy : Polynomial.eval₂ f a (denom (x * y)) = 0
    /-
      case pos
      K : Type u
      inst✝¹ : Field K
      L : Type u
      inst✝ : Field L
      f : RingHom K L
      a : L
      x y : RatFunc K
      hx : Ne (Polynomial.eval₂ f a x.denom) 0
      hy : Ne (Polynomial.eval₂ f a y.denom) 0
      hxy : Eq (Polynomial.eval₂ f a (HMul.hMul x y).denom) 0
      ⊢ Eq (HDiv.hDiv (Polynomial.eval₂ f a (HMul.hMul x y).num) (Polynomial.eval₂ f …
    -/
  · have := Polynomial.eval₂_eq_zero_of_dvd_of_eval₂_eq_zero f a (denom_mul_dvd x y) hxy
    /-
      case pos
      K : Type u
      inst✝¹ : Field K
      L : Type u
      inst✝ : Field L
      f : RingHom K L
      a : L
      x y : RatFunc K
      hx : Ne (Polynomial.eval₂ f a x.denom) 0
      hy : Ne (Polynomial.eval₂ f a y.denom) 0
      hxy : Eq (Polynomial.eval₂ f a (HMul.hMul x y).denom) 0
      this : Eq (Polynomial.eval₂ f a (HMul.hMul x.denom y.denom)) 0
      ⊢ Eq (HDiv.hDiv (Polynomial.eval₂ f a (HMul.hMul x y).num) (Polynomial.eval₂ f …
    -/
    rw [Polynomial.eval₂_mul] at this
    /-
      case pos
      K : Type u
      inst✝¹ : Field K
      L : Type u
      inst✝ : Field L
      f : RingHom K L
      a : L
      x y : RatFunc K
      hx : Ne (Polynomial.eval₂ f a x.denom) 0
      hy : Ne (Polynomial.eval₂ f a y.denom) 0
      hxy : Eq (Polynomial.eval₂ f a (HMul.hMul x y).denom) 0
      this : Eq (HMul.hMul (Polynomial.eval₂ f a x.denom) (Polynomial.eval₂ f a y.de …
      ⊢ Eq (HDiv.hDiv (Polynomial.eval₂ f a (HMul.hMul x y).num) (Polynomial.eval₂ f …
    -/
                                  /-
                                    🎉 no goals
                                  -/
    cases mul_eq_zero.mp this <;> contradiction
                                  /-
                                    🎉 no goals
                                  -/
  rw [div_mul_div_comm, eq_div_iff (mul_ne_zero hx hy), div_eq_mul_inv, mul_right_comm, ←
    div_eq_mul_inv, div_eq_iff hxy]
  /-
    case neg
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    hxy : Not (Eq (Polynomial.eval₂ f a (HMul.hMul x y).denom) 0)
    ⊢ Eq (HMul.hMul (Polynomial.eval₂ f a (HMul.hMul x y).num) (HMul.hMul (Polynom …
  -/
  repeat' rw [← Polynomial.eval₂_mul]
  /-
    case neg
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    hxy : Not (Eq (Polynomial.eval₂ f a (HMul.hMul x y).denom) 0)
    ⊢ Eq (Polynomial.eval₂ f a (HMul.hMul (HMul.hMul x y).num (HMul.hMul x.denom y …
  -/
  congr 1
  /-
    case neg.e_p
    K : Type u
    inst✝¹ : Field K
    L : Type u
    inst✝ : Field L
    f : RingHom K L
    a : L
    x y : RatFunc K
    hx : Ne (Polynomial.eval₂ f a x.denom) 0
    hy : Ne (Polynomial.eval₂ f a y.denom) 0
    hxy : Not (Eq (Polynomial.eval₂ f a (HMul.hMul x y).denom) 0)
    ⊢ Eq (HMul.hMul (HMul.hMul x y).num (HMul.hMul x.denom y.denom)) (HMul.hMul (H …
  -/
  apply num_denom_mul
  /-
    🎉 no goals
  -/


/-- This is the principal ideal generated by `X` in the ring of polynomials over a field K,
  regarded as an element of the height-one-spectrum. -/
def idealX : IsDedekindDomain.HeightOneSpectrum K[X] where
  asIdeal := Ideal.span {X}
                /-
                  K✝ : Type u
                  K : Type u_1
                  inst✝ : Field K
                  ⊢ (Ideal.span (Singleton.singleton Polynomial.X)).IsPrime
                -/
  isPrime := by rw [Ideal.span_singleton_prime]; exacts [Polynomial.prime_X, Polynomial.X_ne_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/
                /-
                  K✝ : Type u
                  K : Type u_1
                  inst✝ : Field K
                  ⊢ Ne (Ideal.span (Singleton.singleton Polynomial.X)) Bot.bot
                -/
  ne_bot  := by rw [ne_eq, Ideal.span_singleton_eq_bot]; exact Polynomial.X_ne_zero
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem idealX_span : (idealX K).asIdeal = Ideal.span {X} := rfl


@[simp]
theorem valuation_X_eq_neg_one :
    (idealX K).valuation (RatFunc.X : RatFunc K) = Multiplicative.ofAdd (-1 : ℤ) := by
  /-
    K : Type u_1
    inst✝ : Field K
    ⊢ Eq ((Polynomial.idealX K).valuation RatFunc.X) ↑(Multiplicative.ofAdd (-1))
  -/
  rw [← RatFunc.algebraMap_X, valuation_of_algebraMap, intValuation_singleton]
    /-
      case hr
      K : Type u_1
      inst✝ : Field K
      ⊢ Ne Polynomial.X 0
    -/
  · exact Polynomial.X_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case hv
      K : Type u_1
      inst✝ : Field K
      ⊢ Eq (Polynomial.idealX K).asIdeal (Ideal.span (Singleton.singleton Polynomial …
    -/
  · exact idealX_span K
    /-
      🎉 no goals
    -/


theorem valuation_of_mk (f : Polynomial K) {g : Polynomial K} (hg : g ≠ 0) :
    (Polynomial.idealX K).valuation (RatFunc.mk f g) =
      (Polynomial.idealX K).intValuation f / (Polynomial.idealX K).intValuation g := by
  /-
    K : Type u_1
    inst✝ : Field K
    f g : Polynomial K
    hg : Ne g 0
    ⊢ Eq ((Polynomial.idealX K).valuation (RatFunc.mk f g)) (HDiv.hDiv ((Polynomia …
  -/
  simp only [RatFunc.mk_eq_mk' _ hg, valuation_of_mk']
  /-
    🎉 no goals
  -/


instance : Valued (RatFunc K) ℤₘ₀ := Valued.mk' (idealX K).valuation


@[simp]
theorem WithZero.valued_def {x : RatFunc K} :
    @Valued.v (RatFunc K) _ _ _ _ x = (idealX K).valuation x := rfl


