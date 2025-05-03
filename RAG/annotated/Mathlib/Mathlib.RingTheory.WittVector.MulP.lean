local notation "𝕎" => WittVector p -- type as `\bbW`


/-- `wittMulN p n` is the family of polynomials that computes
the coefficients of `x * n` in terms of the coefficients of the Witt vector `x`. -/
noncomputable def wittMulN : ℕ → ℕ → MvPolynomial ℕ ℤ
  | 0 => 0
  | n + 1 => fun k => bind₁ (Function.uncurry <| ![wittMulN n, X]) (wittAdd p k)


theorem mulN_coeff (n : ℕ) (x : 𝕎 R) (k : ℕ) :
    (x * n).coeff k = aeval x.coeff (wittMulN p n k) := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    n : Nat
    x : WittVector p R
    k : Nat
    ⊢ Eq ((HMul.hMul x ↑n).coeff k) ((MvPolynomial.aeval x.coeff) (WittVector.witt …
  -/
  induction' n with n ih generalizing k
    /-
      case zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k : Nat
      ⊢ Eq ((HMul.hMul x ↑0).coeff k) ((MvPolynomial.aeval x.coeff) (WittVector.witt …
    -/
  · simp only [Nat.cast_zero, mul_zero, zero_coeff, wittMulN, Pi.zero_apply, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      n : Nat
      ih : ∀ (k : Nat), Eq ((HMul.hMul x ↑n).coeff k) ((MvPolynomial.aeval x.coeff)  …
      k : Nat
      ⊢ Eq ((HMul.hMul x ↑(HAdd.hAdd n 1)).coeff k) ((MvPolynomial.aeval x.coeff) (W …
    -/
  · rw [wittMulN, Nat.cast_add, Nat.cast_one, mul_add, mul_one, aeval_bind₁, add_coeff]
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      n : Nat
      ih : ∀ (k : Nat), Eq ((HMul.hMul x ↑n).coeff k) ((MvPolynomial.aeval x.coeff)  …
      k : Nat
      ⊢ Eq (WittVector.peval (WittVector.wittAdd p k) (Matrix.vecCons (HMul.hMul x ↑ …
    -/
    apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
    /-
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      n : Nat
      ih : ∀ (k : Nat), Eq ((HMul.hMul x ↑n).coeff k) ((MvPolynomial.aeval x.coeff)  …
      k : Nat
      ⊢ Eq (Function.uncurry (Matrix.vecCons (HMul.hMul x ↑n).coeff (Matrix.vecCons  …
    -/
    ext1 ⟨b, i⟩
    /-
      case h.mk
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      n : Nat
      ih : ∀ (k : Nat), Eq ((HMul.hMul x ↑n).coeff k) ((MvPolynomial.aeval x.coeff)  …
      k : Nat
      b : Fin 2
      i : Nat
      ⊢ Eq (Function.uncurry (Matrix.vecCons (HMul.hMul x ↑n).coeff (Matrix.vecCons  …
    -/
    fin_cases b
      /-
        case h.mk.«0»
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝ : CommRing R
        x : WittVector p R
        n : Nat
        ih : ∀ (k : Nat), Eq ((HMul.hMul x ↑n).coeff k) ((MvPolynomial.aeval x.coeff)  …
        k i : Nat
        ⊢ Eq (Function.uncurry (Matrix.vecCons (HMul.hMul x ↑n).coeff (Matrix.vecCons  …
      -/
    · simp [Function.uncurry, Matrix.cons_val_zero, ih]
      /-
        🎉 no goals
      -/
      /-
        case h.mk.«1»
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝ : CommRing R
        x : WittVector p R
        n : Nat
        ih : ∀ (k : Nat), Eq ((HMul.hMul x ↑n).coeff k) ((MvPolynomial.aeval x.coeff)  …
        k i : Nat
        ⊢ Eq (Function.uncurry (Matrix.vecCons (HMul.hMul x ↑n).coeff (Matrix.vecCons  …
      -/
    · simp [Function.uncurry, Matrix.cons_val_one, Matrix.head_cons, aeval_X]
      /-
        🎉 no goals
      -/


/-- Multiplication by `n` is a polynomial function. -/
@[is_poly]
theorem mulN_isPoly (n : ℕ) : IsPoly p fun _ _Rcr x => x * n :=
                                     /-
                                       p : Nat
                                       hp : Fact (Nat.Prime p)
                                       n : Nat
                                       R : Type u_2
                                       _Rcr : CommRing R
                                       x : WittVector p R
                                       ⊢ Eq (HMul.hMul x ↑n).coeff fun n_1 => (MvPolynomial.aeval x.coeff) (WittVecto …
                                     -/
  ⟨⟨wittMulN p n, fun R _Rcr x => by funext k; exact mulN_coeff n x k⟩⟩
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem bind₁_wittMulN_wittPolynomial (n k : ℕ) :
    bind₁ (wittMulN p n) (wittPolynomial p ℤ k) = n * wittPolynomial p ℤ k := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n k : Nat
    ⊢ Eq ((MvPolynomial.bind₁ (WittVector.wittMulN p n)) (wittPolynomial p Int k)) …
  -/
  induction' n with n ih
    /-
      case zero
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Nat
      ⊢ Eq ((MvPolynomial.bind₁ (WittVector.wittMulN p 0)) (wittPolynomial p Int k)) …
    -/
  · simp [wittMulN, Nat.cast_zero, zero_mul, bind₁_zero_wittPolynomial]
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : Nat
      hp : Fact (Nat.Prime p)
      k n : Nat
      ih : Eq ((MvPolynomial.bind₁ (WittVector.wittMulN p n)) (wittPolynomial p Int  …
      ⊢ Eq ((MvPolynomial.bind₁ (WittVector.wittMulN p (HAdd.hAdd n 1))) (wittPolyno …
    -/
  · rw [wittMulN, ← bind₁_bind₁, wittAdd, wittStructureInt_prop]
    /-
      case succ
      p : Nat
      hp : Fact (Nat.Prime p)
      k n : Nat
      ih : Eq ((MvPolynomial.bind₁ (WittVector.wittMulN p n)) (wittPolynomial p Int  …
      ⊢ Eq ((MvPolynomial.bind₁ (Function.uncurry (Matrix.vecCons (WittVector.wittMu …
    -/
    simp only [map_add, Nat.cast_succ, bind₁_X_right]
    /-
      case succ
      p : Nat
      hp : Fact (Nat.Prime p)
      k n : Nat
      ih : Eq ((MvPolynomial.bind₁ (WittVector.wittMulN p n)) (wittPolynomial p Int  …
      ⊢ Eq (HAdd.hAdd ((MvPolynomial.bind₁ (Function.uncurry (Matrix.vecCons (WittVe …
    -/
    rw [add_mul, one_mul, bind₁_rename, bind₁_rename]
    simp only [ih, Function.uncurry, Function.comp_def, bind₁_X_left, AlgHom.id_apply,
      Matrix.cons_val_zero, Matrix.head_cons, Matrix.cons_val_one]


