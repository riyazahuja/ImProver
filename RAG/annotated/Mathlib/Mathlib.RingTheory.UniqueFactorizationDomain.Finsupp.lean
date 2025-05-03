local infixl:50 " ~ᵤ " => Associated


/-- This returns the multiset of irreducible factors as a `Finsupp`. -/
noncomputable def factorization (n : α) : α →₀ ℕ :=
  Multiset.toFinsupp (normalizedFactors n)


theorem factorization_eq_count {n p : α} :
                                                                     /-
                                                                       α : Type u_1
                                                                       inst✝³ : CancelCommMonoidWithZero α
                                                                       inst✝² : UniqueFactorizationMonoid α
                                                                       inst✝¹ : NormalizationMonoid α
                                                                       inst✝ : DecidableEq α
                                                                       n p : α
                                                                       ⊢ Eq ((factorization n) p) (Multiset.count p (UniqueFactorizationMonoid.normal …
                                                                     -/
    factorization n p = Multiset.count p (normalizedFactors n) := by simp [factorization]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
                                                             /-
                                                               α : Type u_1
                                                               inst✝³ : CancelCommMonoidWithZero α
                                                               inst✝² : UniqueFactorizationMonoid α
                                                               inst✝¹ : NormalizationMonoid α
                                                               inst✝ : DecidableEq α
                                                               ⊢ Eq (factorization 0) 0
                                                             -/
theorem factorization_zero : factorization (0 : α) = 0 := by simp [factorization]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
                                                            /-
                                                              α : Type u_1
                                                              inst✝³ : CancelCommMonoidWithZero α
                                                              inst✝² : UniqueFactorizationMonoid α
                                                              inst✝¹ : NormalizationMonoid α
                                                              inst✝ : DecidableEq α
                                                              ⊢ Eq (factorization 1) 0
                                                            -/
theorem factorization_one : factorization (1 : α) = 0 := by simp [factorization]
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The support of `factorization n` is exactly the Finset of normalized factors -/
@[simp]
theorem support_factorization {n : α} :
    (factorization n).support = (normalizedFactors n).toFinset := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : NormalizationMonoid α
    inst✝ : DecidableEq α
    n : α
    ⊢ Eq (factorization n).support (UniqueFactorizationMonoid.normalizedFactors n) …
  -/
  simp [factorization, Multiset.toFinsupp_support]
  /-
    🎉 no goals
  -/


/-- For nonzero `a` and `b`, the power of `p` in `a * b` is the sum of the powers in `a` and `b` -/
@[simp]
theorem factorization_mul {a b : α} (ha : a ≠ 0) (hb : b ≠ 0) :
    factorization (a * b) = factorization a + factorization b := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : NormalizationMonoid α
    inst✝ : DecidableEq α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (factorization (HMul.hMul a b)) (HAdd.hAdd (factorization a) (factorizati …
  -/
  simp [factorization, normalizedFactors_mul ha hb]
  /-
    🎉 no goals
  -/


/-- For any `p`, the power of `p` in `x^n` is `n` times the power in `x` -/
theorem factorization_pow {x : α} {n : ℕ} : factorization (x ^ n) = n • factorization x := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : NormalizationMonoid α
    inst✝ : DecidableEq α
    x : α
    n : Nat
    ⊢ Eq (factorization (HPow.hPow x n)) (HSMul.hSMul n (factorization x))
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : NormalizationMonoid α
    inst✝ : DecidableEq α
    x : α
    n : Nat
    a✝ : α
    ⊢ Eq ((factorization (HPow.hPow x n)) a✝) ((HSMul.hSMul n (factorization x)) a✝)
  -/
  simp [factorization]
  /-
    🎉 no goals
  -/


theorem associated_of_factorization_eq (a b : α) (ha : a ≠ 0) (hb : b ≠ 0)
    (h : factorization a = factorization b) : Associated a b := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : NormalizationMonoid α
    inst✝ : DecidableEq α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    h : Eq (factorization a) (factorization b)
    ⊢ Associated a b
  -/
  simp_rw [factorization, AddEquiv.apply_eq_iff_eq] at h
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : NormalizationMonoid α
    inst✝ : DecidableEq α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    h : Eq (UniqueFactorizationMonoid.normalizedFactors a) (UniqueFactorizationMon …
    ⊢ Associated a b
  -/
  rwa [associated_iff_normalizedFactors_eq_normalizedFactors ha hb]
  /-
    🎉 no goals
  -/


