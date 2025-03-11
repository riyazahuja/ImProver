@[simp]
theorem eval_one_cyclotomic_prime {R : Type*} [CommRing R] {p : ℕ} [hn : Fact p.Prime] :
    eval 1 (cyclotomic p R) = p := by
  simp only [cyclotomic_prime, eval_X, one_pow, Finset.sum_const, eval_pow, eval_finset_sum,
    Finset.card_range, smul_one_eq_cast]


theorem eval₂_one_cyclotomic_prime {R S : Type*} [CommRing R] [Semiring S] (f : R →+* S) {p : ℕ}
                                                          /-
                                                            R : Type u_1
                                                            S : Type u_2
                                                            inst✝² : CommRing R
                                                            inst✝¹ : Semiring S
                                                            f : RingHom R S
                                                            p : Nat
                                                            inst✝ : Fact (Nat.Prime p)
                                                            ⊢ Eq (Polynomial.eval₂ f 1 (Polynomial.cyclotomic p R)) ↑p
                                                          -/
    [Fact p.Prime] : eval₂ f 1 (cyclotomic p R) = p := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem eval_one_cyclotomic_prime_pow {R : Type*} [CommRing R] {p : ℕ} (k : ℕ)
    [hn : Fact p.Prime] : eval 1 (cyclotomic (p ^ (k + 1)) R) = p := by
  simp only [cyclotomic_prime_pow_eq_geom_sum hn.out, eval_X, one_pow, Finset.sum_const, eval_pow,
    eval_finset_sum, Finset.card_range, smul_one_eq_cast]


theorem eval₂_one_cyclotomic_prime_pow {R S : Type*} [CommRing R] [Semiring S] (f : R →+* S)
                                                                                      /-
                                                                                        R : Type u_1
                                                                                        S : Type u_2
                                                                                        inst✝² : CommRing R
                                                                                        inst✝¹ : Semiring S
                                                                                        f : RingHom R S
                                                                                        p k : Nat
                                                                                        inst✝ : Fact (Nat.Prime p)
                                                                                        ⊢ Eq (Polynomial.eval₂ f 1 (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd k 1) …
                                                                                      -/
    {p : ℕ} (k : ℕ) [Fact p.Prime] : eval₂ f 1 (cyclotomic (p ^ (k + 1)) R) = p := by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


private theorem cyclotomic_neg_one_pos {n : ℕ} (hn : 2 < n) {R} [LinearOrderedCommRing R] :
    0 < eval (-1 : R) (cyclotomic n R) := by
  /-
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    ⊢ LT.lt 0 (Polynomial.eval (-1) (Polynomial.cyclotomic n R))
  -/
  haveI := NeZero.of_gt hn
  rw [← map_cyclotomic_int, ← Int.cast_one, ← Int.cast_neg, eval_intCast_map, Int.coe_castRingHom,
    Int.cast_pos]
  suffices 0 < eval (↑(-1 : ℤ)) (cyclotomic n ℝ) by
    rw [← map_cyclotomic_int n ℝ, eval_intCast_map, Int.coe_castRingHom] at this
    simpa only [Int.cast_pos] using this
  /-
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this : NeZero n
    ⊢ LT.lt 0 (Polynomial.eval (↑(-1)) (Polynomial.cyclotomic n Real))
  -/
  simp only [Int.cast_one, Int.cast_neg]
  /-
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this : NeZero n
    ⊢ LT.lt 0 (Polynomial.eval (-1) (Polynomial.cyclotomic n Real))
  -/
  have h0 := cyclotomic_coeff_zero ℝ hn.le
  /-
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this : NeZero n
    h0 : Eq ((Polynomial.cyclotomic n Real).coeff 0) 1
    ⊢ LT.lt 0 (Polynomial.eval (-1) (Polynomial.cyclotomic n Real))
  -/
  rw [coeff_zero_eq_eval_zero] at h0
  /-
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this : NeZero n
    h0 : Eq (Polynomial.eval 0 (Polynomial.cyclotomic n Real)) 1
    ⊢ LT.lt 0 (Polynomial.eval (-1) (Polynomial.cyclotomic n Real))
  -/
  by_contra! hx
  /-
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this : NeZero n
    h0 : Eq (Polynomial.eval 0 (Polynomial.cyclotomic n Real)) 1
    hx : LE.le (Polynomial.eval (-1) (Polynomial.cyclotomic n Real)) 0
    ⊢ False
  -/
  have := intermediate_value_univ (-1) 0 (cyclotomic n ℝ).continuous
  /-
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this✝ : NeZero n
    h0 : Eq (Polynomial.eval 0 (Polynomial.cyclotomic n Real)) 1
    hx : LE.le (Polynomial.eval (-1) (Polynomial.cyclotomic n Real)) 0
    this : HasSubset.Subset (Set.Icc (Polynomial.eval (-1) (Polynomial.cyclotomic  …
    ⊢ False
  -/
  obtain ⟨y, hy : IsRoot _ y⟩ := this (show (0 : ℝ) ∈ Set.Icc _ _ by simpa [h0] using hx)
  /-
    case intro
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this✝ : NeZero n
    h0 : Eq (Polynomial.eval 0 (Polynomial.cyclotomic n Real)) 1
    hx : LE.le (Polynomial.eval (-1) (Polynomial.cyclotomic n Real)) 0
    this : HasSubset.Subset (Set.Icc (Polynomial.eval (-1) (Polynomial.cyclotomic  …
    y : Real
    hy : (Polynomial.cyclotomic n Real).IsRoot y
    ⊢ False
  -/
  rw [@isRoot_cyclotomic_iff] at hy
  /-
    case intro
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this✝ : NeZero n
    h0 : Eq (Polynomial.eval 0 (Polynomial.cyclotomic n Real)) 1
    hx : LE.le (Polynomial.eval (-1) (Polynomial.cyclotomic n Real)) 0
    this : HasSubset.Subset (Set.Icc (Polynomial.eval (-1) (Polynomial.cyclotomic  …
    y : Real
    hy : IsPrimitiveRoot y n
    ⊢ False
  -/
  rw [hy.eq_orderOf] at hn
  /-
    case intro
    n : Nat
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    this✝ : NeZero n
    h0 : Eq (Polynomial.eval 0 (Polynomial.cyclotomic n Real)) 1
    hx : LE.le (Polynomial.eval (-1) (Polynomial.cyclotomic n Real)) 0
    this : HasSubset.Subset (Set.Icc (Polynomial.eval (-1) (Polynomial.cyclotomic  …
    y : Real
    hn : LT.lt 2 (orderOf y)
    hy : IsPrimitiveRoot y n
    ⊢ False
  -/
  exact hn.not_le LinearOrderedRing.orderOf_le_two
  /-
    🎉 no goals
  -/


theorem cyclotomic_pos {n : ℕ} (hn : 2 < n) {R} [LinearOrderedCommRing R] (x : R) :
    0 < eval x (cyclotomic n R) := by
  /-
    n : Nat
    hn : LT.lt 2 n
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    x : R
    ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
  -/
  induction' n using Nat.strong_induction_on with n ih
  /-
    case h
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    x : R
    n : Nat
    ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
    hn : LT.lt 2 n
    ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
  -/
  have hn' : 0 < n := pos_of_gt hn
  /-
    case h
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    x : R
    n : Nat
    ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
    hn : LT.lt 2 n
    hn' : LT.lt 0 n
    ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
  -/
  have hn'' : 1 < n := one_lt_two.trans hn
  /-
    case h
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    x : R
    n : Nat
    ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
    hn : LT.lt 2 n
    hn' : LT.lt 0 n
    hn'' : LT.lt 1 n
    ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
  -/
  have := prod_cyclotomic_eq_geom_sum hn' R
  /-
    case h
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    x : R
    n : Nat
    ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
    hn : LT.lt 2 n
    hn' : LT.lt 0 n
    hn'' : LT.lt 1 n
    this : Eq ((n.divisors.erase 1).prod fun i => Polynomial.cyclotomic i R) ((Fin …
    ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
  -/
  apply_fun eval x at this
  rw [← cons_self_properDivisors hn'.ne', Finset.erase_cons_of_ne _ hn''.ne', Finset.prod_cons,
    eval_mul, eval_geom_sum] at this
  /-
    case h
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    x : R
    n : Nat
    ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
    hn : LT.lt 2 n
    hn' : LT.lt 0 n
    hn'' : LT.lt 1 n
    this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
    ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
  -/
  rcases lt_trichotomy 0 (∑ i ∈ Finset.range n, x ^ i) with (h | h | h)
    /-
      case h.inl
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
    -/
  · apply pos_of_mul_pos_left
      /-
        case h.inl.h
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
        ⊢ LT.lt 0 (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) ?h.inl.b)
      -/
    · rwa [this]
      /-
        🎉 no goals
      -/
    /-
      case h.inl.hb
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      ⊢ LE.le 0 (Polynomial.eval x ((n.properDivisors.erase 1).prod fun x => Polynom …
    -/
    rw [eval_prod]
    /-
      case h.inl.hb
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      ⊢ LE.le 0 ((n.properDivisors.erase 1).prod fun j => Polynomial.eval x (Polynom …
    -/
    refine Finset.prod_nonneg fun i hi => ?_
    /-
      case h.inl.hb
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      i : Nat
      hi : Membership.mem (n.properDivisors.erase 1) i
      ⊢ LE.le 0 (Polynomial.eval x (Polynomial.cyclotomic i R))
    -/
    simp only [Finset.mem_erase, mem_properDivisors] at hi
    /-
      case h.inl.hb
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      i : Nat
      hi : And (Ne i 1) (And (Dvd.dvd i n) (LT.lt i n))
      ⊢ LE.le 0 (Polynomial.eval x (Polynomial.cyclotomic i R))
    -/
    rw [geom_sum_pos_iff hn'.ne'] at h
    /-
      case h.inl.hb
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : Or (Odd n) (LT.lt 0 (HAdd.hAdd x 1))
      i : Nat
      hi : And (Ne i 1) (And (Dvd.dvd i n) (LT.lt i n))
      ⊢ LE.le 0 (Polynomial.eval x (Polynomial.cyclotomic i R))
    -/
    cases' h with hk hx
      /-
        case h.inl.hb.inl
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        i : Nat
        hi : And (Ne i 1) (And (Dvd.dvd i n) (LT.lt i n))
        hk : Odd n
        ⊢ LE.le 0 (Polynomial.eval x (Polynomial.cyclotomic i R))
      -/
    · refine (ih _ hi.2.2 (Nat.two_lt_of_ne ?_ hi.1 ?_)).le <;> rintro rfl
        /-
          case h.inl.hb.inl.refine_1
          R : Type u_1
          inst✝ : LinearOrderedCommRing R
          x : R
          n : Nat
          ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
          hn : LT.lt 2 n
          hn' : LT.lt 0 n
          hn'' : LT.lt 1 n
          this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
          hk : Odd n
          hi : And (Ne 0 1) (And (Dvd.dvd 0 n) (LT.lt 0 n))
          ⊢ False
        -/
      · exact hn'.ne' (zero_dvd_iff.mp hi.2.1)
        /-
          🎉 no goals
        -/
        /-
          case h.inl.hb.inl.refine_2
          R : Type u_1
          inst✝ : LinearOrderedCommRing R
          x : R
          n : Nat
          ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
          hn : LT.lt 2 n
          hn' : LT.lt 0 n
          hn'' : LT.lt 1 n
          this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
          hk : Odd n
          hi : And (Ne 2 1) (And (Dvd.dvd 2 n) (LT.lt 2 n))
          ⊢ False
        -/
      · exact not_odd_iff_even.2 (even_iff_two_dvd.mpr hi.2.1) hk
        /-
          🎉 no goals
        -/
      /-
        case h.inl.hb.inr
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        i : Nat
        hi : And (Ne i 1) (And (Dvd.dvd i n) (LT.lt i n))
        hx : LT.lt 0 (HAdd.hAdd x 1)
        ⊢ LE.le 0 (Polynomial.eval x (Polynomial.cyclotomic i R))
      -/
    · rcases eq_or_ne i 2 with (rfl | hk)
        /-
          case h.inl.hb.inr.inl
          R : Type u_1
          inst✝ : LinearOrderedCommRing R
          x : R
          n : Nat
          ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
          hn : LT.lt 2 n
          hn' : LT.lt 0 n
          hn'' : LT.lt 1 n
          this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
          hx : LT.lt 0 (HAdd.hAdd x 1)
          hi : And (Ne 2 1) (And (Dvd.dvd 2 n) (LT.lt 2 n))
          ⊢ LE.le 0 (Polynomial.eval x (Polynomial.cyclotomic 2 R))
        -/
      · simpa only [eval_X, eval_one, cyclotomic_two, eval_add] using hx.le
        /-
          🎉 no goals
        -/
      /-
        case h.inl.hb.inr.inr
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        i : Nat
        hi : And (Ne i 1) (And (Dvd.dvd i n) (LT.lt i n))
        hx : LT.lt 0 (HAdd.hAdd x 1)
        hk : Ne i 2
        ⊢ LE.le 0 (Polynomial.eval x (Polynomial.cyclotomic i R))
      -/
      refine (ih _ hi.2.2 (Nat.two_lt_of_ne ?_ hi.1 hk)).le
      /-
        case h.inl.hb.inr.inr
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        i : Nat
        hi : And (Ne i 1) (And (Dvd.dvd i n) (LT.lt i n))
        hx : LT.lt 0 (HAdd.hAdd x 1)
        hk : Ne i 2
        ⊢ Ne i 0
      -/
      rintro rfl
      /-
        case h.inl.hb.inr.inr
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        hx : LT.lt 0 (HAdd.hAdd x 1)
        hi : And (Ne 0 1) (And (Dvd.dvd 0 n) (LT.lt 0 n))
        hk : Ne 0 2
        ⊢ False
      -/
      exact hn'.ne' <| zero_dvd_iff.mp hi.2.1
      /-
        🎉 no goals
      -/
    /-
      case h.inr.inl
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : Eq 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
    -/
  · rw [eq_comm, geom_sum_eq_zero_iff_neg_one hn'.ne'] at h
    /-
      case h.inr.inl
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : And (Eq x (-1)) (Even n)
      ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
    -/
    exact h.1.symm ▸ cyclotomic_neg_one_pos hn
    /-
      🎉 no goals
    -/
    /-
      case h.inr.inr
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0
      ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))
    -/
  · apply pos_of_mul_neg_left
      /-
        case h.inr.inr.h
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0
        ⊢ LT.lt (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) ?h.inr.inr. …
      -/
    · rwa [this]
      /-
        🎉 no goals
      -/
    /-
      case h.inr.inr.hb
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0
      ⊢ LE.le (Polynomial.eval x ((n.properDivisors.erase 1).prod fun x => Polynomia …
    -/
    rw [geom_sum_neg_iff hn'.ne'] at h
    have h2 : 2 ∈ n.properDivisors.erase 1 := by
      rw [Finset.mem_erase, mem_properDivisors]
      exact ⟨by decide, even_iff_two_dvd.mp h.1, hn⟩
    /-
      case h.inr.inr.hb
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
      h2 : Membership.mem (n.properDivisors.erase 1) 2
      ⊢ LE.le (Polynomial.eval x ((n.properDivisors.erase 1).prod fun x => Polynomia …
    -/
    rw [eval_prod, ← Finset.prod_erase_mul _ _ h2]
    /-
      case h.inr.inr.hb
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
      hn : LT.lt 2 n
      hn' : LT.lt 0 n
      hn'' : LT.lt 1 n
      this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
      h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
      h2 : Membership.mem (n.properDivisors.erase 1) 2
      ⊢ LE.le (HMul.hMul (((n.properDivisors.erase 1).erase 2).prod fun x_1 => Polyn …
    -/
    apply mul_nonpos_of_nonneg_of_nonpos
      /-
        case h.inr.inr.hb.ha
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
        h2 : Membership.mem (n.properDivisors.erase 1) 2
        ⊢ LE.le 0 (((n.properDivisors.erase 1).erase 2).prod fun x_1 => Polynomial.eva …
      -/
    · refine Finset.prod_nonneg fun i hi => le_of_lt ?_
      /-
        case h.inr.inr.hb.ha
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
        h2 : Membership.mem (n.properDivisors.erase 1) 2
        i : Nat
        hi : Membership.mem ((n.properDivisors.erase 1).erase 2) i
        ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic i R))
      -/
      simp only [Finset.mem_erase, mem_properDivisors] at hi
      /-
        case h.inr.inr.hb.ha
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
        h2 : Membership.mem (n.properDivisors.erase 1) 2
        i : Nat
        hi : And (Ne i 2) (And (Ne i 1) (And (Dvd.dvd i n) (LT.lt i n)))
        ⊢ LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic i R))
      -/
      refine ih _ hi.2.2.2 (Nat.two_lt_of_ne ?_ hi.2.1 hi.1)
      /-
        case h.inr.inr.hb.ha
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
        h2 : Membership.mem (n.properDivisors.erase 1) 2
        i : Nat
        hi : And (Ne i 2) (And (Ne i 1) (And (Dvd.dvd i n) (LT.lt i n)))
        ⊢ Ne i 0
      -/
      rintro rfl
      /-
        case h.inr.inr.hb.ha
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
        h2 : Membership.mem (n.properDivisors.erase 1) 2
        hi : And (Ne 0 2) (And (Ne 0 1) (And (Dvd.dvd 0 n) (LT.lt 0 n)))
        ⊢ False
      -/
      rw [zero_dvd_iff] at hi
      /-
        case h.inr.inr.hb.ha
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
        h2 : Membership.mem (n.properDivisors.erase 1) 2
        hi : And (Ne 0 2) (And (Ne 0 1) (And (Eq n 0) (LT.lt 0 n)))
        ⊢ False
      -/
      exact hn'.ne' hi.2.2.1
      /-
        🎉 no goals
      -/
      /-
        case h.inr.inr.hb.hb
        R : Type u_1
        inst✝ : LinearOrderedCommRing R
        x : R
        n : Nat
        ih : ∀ (m : Nat), LT.lt m n → LT.lt 2 m → LT.lt 0 (Polynomial.eval x (Polynomi …
        hn : LT.lt 2 n
        hn' : LT.lt 0 n
        hn'' : LT.lt 1 n
        this : Eq (HMul.hMul (Polynomial.eval x (Polynomial.cyclotomic n R)) (Polynomi …
        h : And (Even n) (LT.lt (HAdd.hAdd x 1) 0)
        h2 : Membership.mem (n.properDivisors.erase 1) 2
        ⊢ LE.le (Polynomial.eval x (Polynomial.cyclotomic 2 R)) 0
      -/
    · simpa only [eval_X, eval_one, cyclotomic_two, eval_add] using h.right.le
      /-
        🎉 no goals
      -/


theorem cyclotomic_pos_and_nonneg (n : ℕ) {R} [LinearOrderedCommRing R] (x : R) :
    (1 < x → 0 < eval x (cyclotomic n R)) ∧ (1 ≤ x → 0 ≤ eval x (cyclotomic n R)) := by
  /-
    n : Nat
    R : Type u_1
    inst✝ : LinearOrderedCommRing R
    x : R
    ⊢ And (LT.lt 1 x → LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic n R))) (L …
  -/
  rcases n with (_ | _ | _ | n)
    /-
      case zero
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      ⊢ And (LT.lt 1 x → LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic 0 R))) (L …
    -/
  · simp only [cyclotomic_zero, eval_one, zero_lt_one, implies_true, zero_le_one, and_self]
    /-
      🎉 no goals
    -/
  · simp only [zero_add, cyclotomic_one, eval_sub, eval_X, eval_one, sub_pos, imp_self, sub_nonneg,
      and_self]
    /-
      case succ.succ.zero
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      ⊢ And (LT.lt 1 x → LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic (HAdd.hAd …
    -/
  · simp only [zero_add, reduceAdd, cyclotomic_two, eval_add, eval_X, eval_one]
    /-
      case succ.succ.zero
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      ⊢ And (LT.lt 1 x → LT.lt 0 (HAdd.hAdd x 1)) (LE.le 1 x → LE.le 0 (HAdd.hAdd x  …
    -/
                              /-
                                🎉 no goals
                              -/
    constructor <;> intro <;> linarith
                              /-
                                🎉 no goals
                              -/
    /-
      case succ.succ.succ
      R : Type u_1
      inst✝ : LinearOrderedCommRing R
      x : R
      n : Nat
      ⊢ And (LT.lt 1 x → LT.lt 0 (Polynomial.eval x (Polynomial.cyclotomic (HAdd.hAd …
    -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  · constructor <;> intro <;> [skip; apply le_of_lt] <;> apply cyclotomic_pos (by omega)
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Cyclotomic polynomials are always positive on inputs larger than one.
Similar to `cyclotomic_pos` but with the condition on the input rather than index of the
cyclotomic polynomial. -/
theorem cyclotomic_pos' (n : ℕ) {R} [LinearOrderedCommRing R] {x : R} (hx : 1 < x) :
    0 < eval x (cyclotomic n R) :=
  (cyclotomic_pos_and_nonneg n x).1 hx


/-- Cyclotomic polynomials are always nonnegative on inputs one or more. -/
theorem cyclotomic_nonneg (n : ℕ) {R} [LinearOrderedCommRing R] {x : R} (hx : 1 ≤ x) :
    0 ≤ eval x (cyclotomic n R) :=
  (cyclotomic_pos_and_nonneg n x).2 hx


theorem eval_one_cyclotomic_not_prime_pow {R : Type*} [Ring R] {n : ℕ}
    (h : ∀ {p : ℕ}, p.Prime → ∀ k : ℕ, p ^ k ≠ n) : eval 1 (cyclotomic n R) = 1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
    ⊢ Eq (Polynomial.eval 1 (Polynomial.cyclotomic n R)) 1
  -/
  rcases n.eq_zero_or_pos with (rfl | hn')
    /-
      case inl
      R : Type u_1
      inst✝ : Ring R
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) 0
      ⊢ Eq (Polynomial.eval 1 (Polynomial.cyclotomic 0 R)) 1
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
    hn' : GT.gt n 0
    ⊢ Eq (Polynomial.eval 1 (Polynomial.cyclotomic n R)) 1
  -/
  have hn : 1 < n := one_lt_iff_ne_zero_and_ne_one.mpr ⟨hn'.ne', (h Nat.prime_two 0).symm⟩
  /-
    case inr
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
    hn' : GT.gt n 0
    hn : LT.lt 1 n
    ⊢ Eq (Polynomial.eval 1 (Polynomial.cyclotomic n R)) 1
  -/
  rsuffices h | h : eval 1 (cyclotomic n ℤ) = 1 ∨ eval 1 (cyclotomic n ℤ) = -1
    /-
      case inr.inl
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h✝ : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      h : Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) 1
      ⊢ Eq (Polynomial.eval 1 (Polynomial.cyclotomic n R)) 1
    -/
  · have := eval_intCast_map (Int.castRingHom R) (cyclotomic n ℤ) 1
    /-
      case inr.inl
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h✝ : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      h : Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) 1
      this : Eq (Polynomial.eval (↑1) (Polynomial.map (Int.castRingHom R) (Polynomia …
      ⊢ Eq (Polynomial.eval 1 (Polynomial.cyclotomic n R)) 1
    -/
    simpa only [map_cyclotomic, Int.cast_one, h, eq_intCast] using this
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h✝ : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      h : Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) (-1)
      ⊢ Eq (Polynomial.eval 1 (Polynomial.cyclotomic n R)) 1
    -/
  · exfalso
    /-
      case inr.inr
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h✝ : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      h : Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) (-1)
      ⊢ False
    -/
    linarith [cyclotomic_nonneg n (le_refl (1 : ℤ))]
    /-
      🎉 no goals
    -/
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
    hn' : GT.gt n 0
    hn : LT.lt 1 n
    ⊢ Or (Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) 1) (Eq (Polynomial. …
  -/
  rw [← Int.natAbs_eq_natAbs_iff, Int.natAbs_one, Nat.eq_one_iff_not_exists_prime_dvd]
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
    hn' : GT.gt n 0
    hn : LT.lt 1 n
    ⊢ ∀ (p : Nat), Nat.Prime p → Not (Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyc …
  -/
  intro p hp hpe
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
    hn' : GT.gt n 0
    hn : LT.lt 1 n
    p : Nat
    hp : Nat.Prime p
    hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
    ⊢ False
  -/
  haveI := Fact.mk hp
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
    hn' : GT.gt n 0
    hn : LT.lt 1 n
    p : Nat
    hp : Nat.Prime p
    hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
    this : Fact (Nat.Prime p)
    ⊢ False
  -/
  have := prod_cyclotomic_eq_geom_sum hn' ℤ
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
    hn' : GT.gt n 0
    hn : LT.lt 1 n
    p : Nat
    hp : Nat.Prime p
    hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
    this✝ : Fact (Nat.Prime p)
    this : Eq ((n.divisors.erase 1).prod fun i => Polynomial.cyclotomic i Int) ((F …
    ⊢ False
  -/
  apply_fun eval 1 at this
  rw [eval_geom_sum, one_geom_sum, eval_prod, eq_comm, ←
    Finset.prod_sdiff <| @range_pow_padicValNat_subset_divisors' p _ _, Finset.prod_image] at this
    /-
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul ((SDiff.sdiff (n.divisors.erase 1) (Finset.image (fu …
      ⊢ False
    -/
  · simp_rw [eval_one_cyclotomic_prime_pow, Finset.prod_const, Finset.card_range, mul_comm] at this
    /-
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) ((SDiff.sdiff (n. …
      ⊢ False
    -/
    rw [← Finset.prod_sdiff <| show {n} ⊆ _ from _] at this
    /-
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) (HMul.hMul ((SDif …
      ⊢ False
    -/
    swap
    · simp only [singleton_subset_iff, mem_sdiff, mem_erase, Ne, mem_divisors, dvd_refl,
        true_and, mem_image, mem_range, exists_prop, not_exists, not_and]
      /-
        R : Type u_1
        inst✝ : Ring R
        n : Nat
        h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
        hn' : GT.gt n 0
        hn : LT.lt 1 n
        p : Nat
        hp : Nat.Prime p
        hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
        this✝ : Fact (Nat.Prime p)
        this : Eq (↑n) (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) ((SDiff.sdiff (n. …
        ⊢ And (And (Not (Eq n 1)) (Not (Eq n 0))) (∀ (x : Nat), LT.lt x (padicValNat p …
      -/
      exact ⟨⟨hn.ne', hn'.ne'⟩, fun t _ => h hp _⟩
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) (HMul.hMul ((SDif …
      ⊢ False
    -/
    rw [← Int.natAbs_ofNat p, Int.natAbs_dvd_natAbs] at hpe
    /-
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      hpe : Dvd.dvd (↑p) (Polynomial.eval 1 (Polynomial.cyclotomic n Int))
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) (HMul.hMul ((SDif …
      ⊢ False
    -/
    obtain ⟨t, ht⟩ := hpe
    /-
      case intro
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) (HMul.hMul ((SDif …
      t : Int
      ht : Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) (HMul.hMul (↑p) t)
      ⊢ False
    -/
    rw [Finset.prod_singleton, ht, mul_left_comm, mul_comm, ← mul_assoc, mul_assoc] at this
    /-
      case intro
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      this✝ : Fact (Nat.Prime p)
      t : Int
      this : Eq (↑n) (HMul.hMul (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) ↑p) (H …
      ht : Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) (HMul.hMul (↑p) t)
      ⊢ False
    -/
    have : (p : ℤ) ^ padicValNat p n * p ∣ n := ⟨_, this⟩
    /-
      case intro
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      this✝¹ : Fact (Nat.Prime p)
      t : Int
      this✝ : Eq (↑n) (HMul.hMul (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) ↑p) ( …
      ht : Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) (HMul.hMul (↑p) t)
      this : Dvd.dvd (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) ↑p) ↑n
      ⊢ False
    -/
    simp only [← _root_.pow_succ, ← Int.natAbs_dvd_natAbs, Int.natAbs_ofNat, Int.natAbs_pow] at this
    /-
      case intro
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      this✝¹ : Fact (Nat.Prime p)
      t : Int
      this✝ : Eq (↑n) (HMul.hMul (HMul.hMul (HPow.hPow (↑p) (padicValNat p n)) ↑p) ( …
      ht : Eq (Polynomial.eval 1 (Polynomial.cyclotomic n Int)) (HMul.hMul (↑p) t)
      this : Dvd.dvd (HPow.hPow p (HAdd.hAdd (padicValNat p n) 1)) n
      ⊢ False
    -/
    exact pow_succ_padicValNat_not_dvd hn'.ne' this
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul ((SDiff.sdiff (n.divisors.erase 1) (Finset.image (fu …
      ⊢ ∀ (x : Nat), Membership.mem (Finset.range (padicValNat p n)) x → ∀ (y : Nat) …
    -/
  · rintro x - y - hxy
    /-
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul ((SDiff.sdiff (n.divisors.erase 1) (Finset.image (fu …
      x y : Nat
      hxy : Eq (HPow.hPow p (HAdd.hAdd x 1)) (HPow.hPow p (HAdd.hAdd y 1))
      ⊢ Eq x y
    -/
    apply Nat.succ_injective
    /-
      case a
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      h : ∀ {p : Nat}, Nat.Prime p → ∀ (k : Nat), Ne (HPow.hPow p k) n
      hn' : GT.gt n 0
      hn : LT.lt 1 n
      p : Nat
      hp : Nat.Prime p
      hpe : Dvd.dvd p (Polynomial.eval 1 (Polynomial.cyclotomic n Int)).natAbs
      this✝ : Fact (Nat.Prime p)
      this : Eq (↑n) (HMul.hMul ((SDiff.sdiff (n.divisors.erase 1) (Finset.image (fu …
      x y : Nat
      hxy : Eq (HPow.hPow p (HAdd.hAdd x 1)) (HPow.hPow p (HAdd.hAdd y 1))
      ⊢ Eq x.succ y.succ
    -/
    exact Nat.pow_right_injective hp.two_le hxy
    /-
      🎉 no goals
    -/


theorem sub_one_pow_totient_lt_cyclotomic_eval {n : ℕ} {q : ℝ} (hn' : 2 ≤ n) (hq' : 1 < q) :
    (q - 1) ^ totient n < (cyclotomic n ℝ).eval q := by
  /-
    n : Nat
    q : Real
    hn' : LE.le 2 n
    hq' : LT.lt 1 q
    ⊢ LT.lt (HPow.hPow (HSub.hSub q 1) n.totient) (Polynomial.eval q (Polynomial.c …
  -/
  have hn : 0 < n := pos_of_gt hn'
  /-
    n : Nat
    q : Real
    hn' : LE.le 2 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    ⊢ LT.lt (HPow.hPow (HSub.hSub q 1) n.totient) (Polynomial.eval q (Polynomial.c …
  -/
  have hq := zero_lt_one.trans hq'
  have hfor : ∀ ζ' ∈ primitiveRoots n ℂ, q - 1 ≤ ‖↑q - ζ'‖ := by
    intro ζ' hζ'
    rw [mem_primitiveRoots hn] at hζ'
    convert norm_sub_norm_le (↑q) ζ'
    · rw [Complex.norm_real, Real.norm_of_nonneg hq.le]
    · rw [hζ'.norm'_eq_one hn.ne']
  /-
    n : Nat
    q : Real
    hn' : LE.le 2 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    hq : LT.lt 0 q
    hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
    ⊢ LT.lt (HPow.hPow (HSub.hSub q 1) n.totient) (Polynomial.eval q (Polynomial.c …
  -/
  let ζ := Complex.exp (2 * ↑Real.pi * Complex.I / ↑n)
  /-
    n : Nat
    q : Real
    hn' : LE.le 2 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    hq : LT.lt 0 q
    hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
    ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
    ⊢ LT.lt (HPow.hPow (HSub.hSub q 1) n.totient) (Polynomial.eval q (Polynomial.c …
  -/
  have hζ : IsPrimitiveRoot ζ n := Complex.isPrimitiveRoot_exp n hn.ne'
  have hex : ∃ ζ' ∈ primitiveRoots n ℂ, q - 1 < ‖↑q - ζ'‖ := by
    refine ⟨ζ, (mem_primitiveRoots hn).mpr hζ, ?_⟩
    suffices ¬SameRay ℝ (q : ℂ) ζ by
      convert lt_norm_sub_of_not_sameRay this <;>
        simp only [hζ.norm'_eq_one hn.ne', Real.norm_of_nonneg hq.le, Complex.norm_real]
    rw [Complex.sameRay_iff]
    push_neg
    refine ⟨mod_cast hq.ne', hζ.ne_zero hn.ne', ?_⟩
    rw [Complex.arg_ofReal_of_nonneg hq.le, Ne, eq_comm, hζ.arg_eq_zero_iff hn.ne']
    clear_value ζ
    rintro rfl
    linarith [hζ.unique IsPrimitiveRoot.one]
  have : ¬eval (↑q) (cyclotomic n ℂ) = 0 := by
    erw [cyclotomic.eval_apply q n (algebraMap ℝ ℂ)]
    simpa only [Complex.coe_algebraMap, Complex.ofReal_eq_zero] using (cyclotomic_pos' n hq').ne'
  suffices Units.mk0 (Real.toNNReal (q - 1)) (by simp [hq']) ^ totient n <
      Units.mk0 ‖(cyclotomic n ℂ).eval ↑q‖₊ (by simp [this]) by
    simp only [← Units.val_lt_val, Units.val_pow_eq_pow_val, Units.val_mk0, ← NNReal.coe_lt_coe,
      hq'.le, Real.toNNReal_lt_toNNReal_iff_of_nonneg, coe_nnnorm, Complex.norm_eq_abs,
      NNReal.coe_pow, Real.coe_toNNReal', max_eq_left, sub_nonneg] at this
    convert this
    erw [cyclotomic.eval_apply q n (algebraMap ℝ ℂ), eq_comm]
    simp only [cyclotomic_nonneg n hq'.le, Complex.coe_algebraMap, Complex.abs_ofReal, abs_eq_self]
  simp only [cyclotomic_eq_prod_X_sub_primitiveRoots hζ, eval_prod, eval_C, eval_X, eval_sub,
    nnnorm_prod, Units.mk0_prod]
  /-
    n : Nat
    q : Real
    hn' : LE.le 2 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    hq : LT.lt 0 q
    hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
    ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
    hζ : IsPrimitiveRoot ζ n
    hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
    this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
    ⊢ LT.lt (HPow.hPow (Units.mk0 (HSub.hSub q 1).toNNReal ⋯) n.totient) ((primiti …
  -/
  convert Finset.prod_lt_prod' (M := NNRealˣ) _ _
  /-
    case h.e'_3
    n : Nat
    q : Real
    hn' : LE.le 2 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    hq : LT.lt 0 q
    hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
    ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
    hζ : IsPrimitiveRoot ζ n
    hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
    this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
    ⊢ Eq (HPow.hPow (Units.mk0 (HSub.hSub q 1).toNNReal ⋯) n.totient) ((primitiveR …
  -/
  swap; · exact fun _ => Units.mk0 (Real.toNNReal (q - 1)) (by simp [hq'])
          /-
            🎉 no goals
          -/
    /-
      case h.e'_3
      n : Nat
      q : Real
      hn' : LE.le 2 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      ⊢ Eq (HPow.hPow (Units.mk0 (HSub.hSub q 1).toNNReal ⋯) n.totient) ((primitiveR …
    -/
  · simp only [Complex.card_primitiveRoots, prod_const, card_attach]
    /-
      🎉 no goals
    -/
  · simp only [Subtype.coe_mk, Finset.mem_attach, forall_true_left, Subtype.forall, ←
      Units.val_le_val, ← NNReal.coe_le_coe, Complex.abs.nonneg, hq'.le, Units.val_mk0,
      Real.coe_toNNReal', coe_nnnorm, Complex.norm_eq_abs, max_le_iff, tsub_le_iff_right]
    /-
      case convert_5
      n : Nat
      q : Real
      hn' : LE.le 2 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      ⊢ ∀ (a : Complex), Membership.mem (primitiveRoots n Complex) a → And (LE.le q  …
    -/
    intro x hx
    /-
      case convert_5
      n : Nat
      q : Real
      hn' : LE.le 2 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      x : Complex
      hx : Membership.mem (primitiveRoots n Complex) x
      ⊢ And (LE.le q (HAdd.hAdd (Complex.abs (HSub.hSub (↑q) x)) 1)) True
    -/
    simpa only [and_true, tsub_le_iff_right] using hfor x hx
    /-
      🎉 no goals
    -/
  · simp only [Subtype.coe_mk, Finset.mem_attach, exists_true_left, Subtype.exists, ←
      NNReal.coe_lt_coe, ← Units.val_lt_val, Units.val_mk0 _, coe_nnnorm]
    /-
      case convert_6
      n : Nat
      q : Real
      hn' : LE.le 2 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      ⊢ Exists fun a => Exists fun h => And True (LT.lt (↑(HSub.hSub q 1).toNNReal)  …
    -/
    simpa [hq'.le, Real.coe_toNNReal', max_eq_left, sub_nonneg] using hex
    /-
      🎉 no goals
    -/


theorem sub_one_pow_totient_le_cyclotomic_eval {q : ℝ} (hq' : 1 < q) :
    ∀ n, (q - 1) ^ totient n ≤ (cyclotomic n ℝ).eval q
            /-
              q : Real
              hq' : LT.lt 1 q
              ⊢ LE.le (HPow.hPow (HSub.hSub q 1) (Nat.totient 0)) (Polynomial.eval q (Polyno …
            -/
  | 0 => by simp only [totient_zero, _root_.pow_zero, cyclotomic_zero, eval_one, le_refl]
            /-
              🎉 no goals
            -/
            /-
              q : Real
              hq' : LT.lt 1 q
              ⊢ LE.le (HPow.hPow (HSub.hSub q 1) (Nat.totient 1)) (Polynomial.eval q (Polyno …
            -/
  | 1 => by simp only [totient_one, pow_one, cyclotomic_one, eval_sub, eval_X, eval_one, le_refl]
            /-
              🎉 no goals
            -/
  | _ + 2 => (sub_one_pow_totient_lt_cyclotomic_eval le_add_self hq').le


theorem cyclotomic_eval_lt_add_one_pow_totient {n : ℕ} {q : ℝ} (hn' : 3 ≤ n) (hq' : 1 < q) :
    (cyclotomic n ℝ).eval q < (q + 1) ^ totient n := by
  /-
    n : Nat
    q : Real
    hn' : LE.le 3 n
    hq' : LT.lt 1 q
    ⊢ LT.lt (Polynomial.eval q (Polynomial.cyclotomic n Real)) (HPow.hPow (HAdd.hA …
  -/
  have hn : 0 < n := pos_of_gt hn'
  /-
    n : Nat
    q : Real
    hn' : LE.le 3 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    ⊢ LT.lt (Polynomial.eval q (Polynomial.cyclotomic n Real)) (HPow.hPow (HAdd.hA …
  -/
  have hq := zero_lt_one.trans hq'
  have hfor : ∀ ζ' ∈ primitiveRoots n ℂ, ‖↑q - ζ'‖ ≤ q + 1 := by
    intro ζ' hζ'
    rw [mem_primitiveRoots hn] at hζ'
    convert norm_sub_le (↑q) ζ'
    · rw [Complex.norm_real, Real.norm_of_nonneg (zero_le_one.trans_lt hq').le]
    · rw [hζ'.norm'_eq_one hn.ne']
  /-
    n : Nat
    q : Real
    hn' : LE.le 3 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    hq : LT.lt 0 q
    hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
    ⊢ LT.lt (Polynomial.eval q (Polynomial.cyclotomic n Real)) (HPow.hPow (HAdd.hA …
  -/
  let ζ := Complex.exp (2 * ↑Real.pi * Complex.I / ↑n)
  /-
    n : Nat
    q : Real
    hn' : LE.le 3 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    hq : LT.lt 0 q
    hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
    ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
    ⊢ LT.lt (Polynomial.eval q (Polynomial.cyclotomic n Real)) (HPow.hPow (HAdd.hA …
  -/
  have hζ : IsPrimitiveRoot ζ n := Complex.isPrimitiveRoot_exp n hn.ne'
  have hex : ∃ ζ' ∈ primitiveRoots n ℂ, ‖↑q - ζ'‖ < q + 1 := by
    refine ⟨ζ, (mem_primitiveRoots hn).mpr hζ, ?_⟩
    suffices ¬SameRay ℝ (q : ℂ) (-ζ) by
      convert norm_add_lt_of_not_sameRay this using 2
      · rw [Complex.norm_eq_abs, Complex.abs_ofReal]
        symm
        exact abs_eq_self.mpr hq.le
      · simp [abs_of_pos hq, hζ.norm'_eq_one hn.ne', -Complex.norm_eq_abs]
    rw [Complex.sameRay_iff]
    push_neg
    refine ⟨mod_cast hq.ne', neg_ne_zero.mpr <| hζ.ne_zero hn.ne', ?_⟩
    rw [Complex.arg_ofReal_of_nonneg hq.le, Ne, eq_comm]
    intro h
    rw [Complex.arg_eq_zero_iff, Complex.neg_re, neg_nonneg, Complex.neg_im, neg_eq_zero] at h
    have hζ₀ : ζ ≠ 0 := by
      clear_value ζ
      rintro rfl
      exact hn.ne' (hζ.unique IsPrimitiveRoot.zero)
    have : ζ.re < 0 ∧ ζ.im = 0 := ⟨h.1.lt_of_ne ?_, h.2⟩
    · rw [← Complex.arg_eq_pi_iff, hζ.arg_eq_pi_iff hn.ne'] at this
      rw [this] at hζ
      linarith [hζ.unique <| IsPrimitiveRoot.neg_one 0 two_ne_zero.symm]
    · contrapose! hζ₀
      apply Complex.ext <;> simp [hζ₀, h.2]
  have : ¬eval (↑q) (cyclotomic n ℂ) = 0 := by
    erw [cyclotomic.eval_apply q n (algebraMap ℝ ℂ)]
    simp only [Complex.coe_algebraMap, Complex.ofReal_eq_zero]
    exact (cyclotomic_pos' n hq').ne.symm
  suffices Units.mk0 ‖(cyclotomic n ℂ).eval ↑q‖₊ (by simp [this]) <
      Units.mk0 (Real.toNNReal (q + 1)) (by simp; linarith) ^ totient n by
    simp only [← Units.val_lt_val, Units.val_pow_eq_pow_val, Units.val_mk0, ← NNReal.coe_lt_coe,
      hq'.le, Real.toNNReal_lt_toNNReal_iff_of_nonneg, coe_nnnorm, Complex.norm_eq_abs,
      NNReal.coe_pow, Real.coe_toNNReal', max_eq_left, sub_nonneg] at this
    convert this using 2
    · erw [cyclotomic.eval_apply q n (algebraMap ℝ ℂ), eq_comm]
      simp [cyclotomic_nonneg n hq'.le]
    rw [eq_comm, max_eq_left_iff]
    linarith
  simp only [cyclotomic_eq_prod_X_sub_primitiveRoots hζ, eval_prod, eval_C, eval_X, eval_sub,
    nnnorm_prod, Units.mk0_prod]
  /-
    n : Nat
    q : Real
    hn' : LE.le 3 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    hq : LT.lt 0 q
    hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
    ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
    hζ : IsPrimitiveRoot ζ n
    hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
    this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
    ⊢ LT.lt ((primitiveRoots n Complex).attach.prod fun i => Units.mk0 (NNNorm.nnn …
  -/
  convert Finset.prod_lt_prod' (M := NNRealˣ) _ _
  /-
    case h.e'_4
    n : Nat
    q : Real
    hn' : LE.le 3 n
    hq' : LT.lt 1 q
    hn : LT.lt 0 n
    hq : LT.lt 0 q
    hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
    ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
    hζ : IsPrimitiveRoot ζ n
    hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
    this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
    ⊢ Eq (HPow.hPow (Units.mk0 (HAdd.hAdd q 1).toNNReal ⋯) n.totient) ((primitiveR …
  -/
  swap; · exact fun _ => Units.mk0 (Real.toNNReal (q + 1)) (by simp; linarith only [hq'])
          /-
            🎉 no goals
          -/
    /-
      case h.e'_4
      n : Nat
      q : Real
      hn' : LE.le 3 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      ⊢ Eq (HPow.hPow (Units.mk0 (HAdd.hAdd q 1).toNNReal ⋯) n.totient) ((primitiveR …
    -/
  · simp [Complex.card_primitiveRoots]
    /-
      🎉 no goals
    -/
  · simp only [Subtype.coe_mk, Finset.mem_attach, forall_true_left, Subtype.forall, ←
      Units.val_le_val, ← NNReal.coe_le_coe, Complex.abs.nonneg, hq'.le, Units.val_mk0,
      Real.coe_toNNReal, coe_nnnorm, Complex.norm_eq_abs, max_le_iff]
    /-
      case convert_5
      n : Nat
      q : Real
      hn' : LE.le 3 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      ⊢ ∀ (a : Complex), Membership.mem (primitiveRoots n Complex) a → LE.le (Comple …
    -/
    intro x hx
    /-
      case convert_5
      n : Nat
      q : Real
      hn' : LE.le 3 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      x : Complex
      hx : Membership.mem (primitiveRoots n Complex) x
      ⊢ LE.le (Complex.abs (HSub.hSub (↑q) x)) ↑(HAdd.hAdd q 1).toNNReal
    -/
    have : Complex.abs _ ≤ _ := hfor x hx
    /-
      case convert_5
      n : Nat
      q : Real
      hn' : LE.le 3 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this✝ : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      x : Complex
      hx : Membership.mem (primitiveRoots n Complex) x
      this : LE.le (Complex.abs (HSub.hSub (↑q) x)) (HAdd.hAdd q 1)
      ⊢ LE.le (Complex.abs (HSub.hSub (↑q) x)) ↑(HAdd.hAdd q 1).toNNReal
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  · simp only [Subtype.coe_mk, Finset.mem_attach, exists_true_left, Subtype.exists, ←
      NNReal.coe_lt_coe, ← Units.val_lt_val, Units.val_mk0 _, coe_nnnorm]
    /-
      case convert_6
      n : Nat
      q : Real
      hn' : LE.le 3 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
      hζ : IsPrimitiveRoot ζ n
      hex : Exists fun ζ' => And (Membership.mem (primitiveRoots n Complex) ζ') (LT. …
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      ⊢ Exists fun a => Exists fun h => And True (LT.lt (Norm.norm (HSub.hSub (↑q) a …
    -/
    obtain ⟨ζ, hζ, hhζ : Complex.abs _ < _⟩ := hex
    /-
      case convert_6.intro.intro
      n : Nat
      q : Real
      hn' : LE.le 3 n
      hq' : LT.lt 1 q
      hn : LT.lt 0 n
      hq : LT.lt 0 q
      hfor : ∀ (ζ' : Complex), Membership.mem (primitiveRoots n Complex) ζ' → LE.le  …
      ζ✝ : Complex := Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Compl …
      hζ✝ : IsPrimitiveRoot ζ✝ n
      this : Not (Eq (Polynomial.eval (↑q) (Polynomial.cyclotomic n Complex)) 0)
      ζ : Complex
      hζ : Membership.mem (primitiveRoots n Complex) ζ
      hhζ : LT.lt (Complex.abs (HSub.hSub (↑q) ζ)) (HAdd.hAdd q 1)
      ⊢ Exists fun a => Exists fun h => And True (LT.lt (Norm.norm (HSub.hSub (↑q) a …
    -/
    exact ⟨ζ, hζ, by simp [hhζ]⟩
    /-
      🎉 no goals
    -/


theorem cyclotomic_eval_le_add_one_pow_totient {q : ℝ} (hq' : 1 < q) :
    ∀ n, (cyclotomic n ℝ).eval q ≤ (q + 1) ^ totient n
            /-
              q : Real
              hq' : LT.lt 1 q
              ⊢ LE.le (Polynomial.eval q (Polynomial.cyclotomic 0 Real)) (HPow.hPow (HAdd.hA …
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
            /-
              q : Real
              hq' : LT.lt 1 q
              ⊢ LE.le (Polynomial.eval q (Polynomial.cyclotomic 1 Real)) (HPow.hPow (HAdd.hA …
            -/
  | 1 => by simp [add_assoc, add_nonneg, zero_le_one]
            /-
              🎉 no goals
            -/
            /-
              q : Real
              hq' : LT.lt 1 q
              ⊢ LE.le (Polynomial.eval q (Polynomial.cyclotomic 2 Real)) (HPow.hPow (HAdd.hA …
            -/
  | 2 => by simp
            /-
              🎉 no goals
            -/
  | _ + 3 => (cyclotomic_eval_lt_add_one_pow_totient le_add_self hq').le


theorem sub_one_pow_totient_lt_natAbs_cyclotomic_eval {n : ℕ} {q : ℕ} (hn' : 1 < n) (hq : q ≠ 1) :
    (q - 1) ^ totient n < ((cyclotomic n ℤ).eval ↑q).natAbs := by
  /-
    n q : Nat
    hn' : LT.lt 1 n
    hq : Ne q 1
    ⊢ LT.lt (HPow.hPow (HSub.hSub q 1) n.totient) (Polynomial.eval (↑q) (Polynomia …
  -/
  rcases hq.lt_or_lt.imp_left Nat.lt_one_iff.mp with (rfl | hq')
  · rw [zero_tsub, zero_pow (Nat.totient_pos.2 (pos_of_gt hn')).ne', pos_iff_ne_zero,
      Int.natAbs_ne_zero, Nat.cast_zero, ← coeff_zero_eq_eval_zero, cyclotomic_coeff_zero _ hn']
    /-
      case inl
      n : Nat
      hn' : LT.lt 1 n
      hq : Ne 0 1
      ⊢ Ne 1 0
    -/
    exact one_ne_zero
    /-
      🎉 no goals
    -/
  /-
    case inr
    n q : Nat
    hn' : LT.lt 1 n
    hq : Ne q 1
    hq' : LT.lt 1 q
    ⊢ LT.lt (HPow.hPow (HSub.hSub q 1) n.totient) (Polynomial.eval (↑q) (Polynomia …
  -/
  rw [← @Nat.cast_lt ℝ, Nat.cast_pow, Nat.cast_sub hq'.le, Nat.cast_one, Int.cast_natAbs]
  /-
    case inr
    n q : Nat
    hn' : LT.lt 1 n
    hq : Ne q 1
    hq' : LT.lt 1 q
    ⊢ LT.lt (HPow.hPow (HSub.hSub (↑q) 1) n.totient) ↑(abs (Polynomial.eval (↑q) ( …
  -/
  refine (sub_one_pow_totient_lt_cyclotomic_eval hn' (Nat.one_lt_cast.2 hq')).trans_le ?_
  /-
    case inr
    n q : Nat
    hn' : LT.lt 1 n
    hq : Ne q 1
    hq' : LT.lt 1 q
    ⊢ LE.le (Polynomial.eval (↑q) (Polynomial.cyclotomic n Real)) ↑(abs (Polynomia …
  -/
  convert (cyclotomic.eval_apply (q : ℤ) n (algebraMap ℤ ℝ)).trans_le (le_abs_self _)
  /-
    case h.e'_4
    n q : Nat
    hn' : LT.lt 1 n
    hq : Ne q 1
    hq' : LT.lt 1 q
    ⊢ Eq (↑(abs (Polynomial.eval (↑q) (Polynomial.cyclotomic n Int)))) (abs ((alge …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sub_one_lt_natAbs_cyclotomic_eval {n : ℕ} {q : ℕ} (hn' : 1 < n) (hq : q ≠ 1) :
    q - 1 < ((cyclotomic n ℤ).eval ↑q).natAbs :=
  calc
    q - 1 ≤ (q - 1) ^ totient n := Nat.le_self_pow (Nat.totient_pos.2 <| pos_of_gt hn').ne' _
    _ < ((cyclotomic n ℤ).eval ↑q).natAbs := sub_one_pow_totient_lt_natAbs_cyclotomic_eval hn' hq


