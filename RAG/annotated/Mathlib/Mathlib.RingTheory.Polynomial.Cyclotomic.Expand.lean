/-- If `p` is a prime such that `¬ p ∣ n`, then
`expand R p (cyclotomic n R) = (cyclotomic (n * p) R) * (cyclotomic n R)`. -/
@[simp]
theorem cyclotomic_expand_eq_cyclotomic_mul {p n : ℕ} (hp : Nat.Prime p) (hdiv : ¬p ∣ n)
    (R : Type*) [CommRing R] :
    expand R p (cyclotomic n R) = cyclotomic (n * p) R * cyclotomic n R := by
  /-
    p n : Nat
    hp : Nat.Prime p
    hdiv : Not (Dvd.dvd p n)
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq ((Polynomial.expand R p) (Polynomial.cyclotomic n R)) (HMul.hMul (Polynom …
  -/
  rcases Nat.eq_zero_or_pos n with (rfl | hnpos)
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      R : Type u_1
      inst✝ : CommRing R
      hdiv : Not (Dvd.dvd p 0)
      ⊢ Eq ((Polynomial.expand R p) (Polynomial.cyclotomic 0 R)) (HMul.hMul (Polynom …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p n : Nat
    hp : Nat.Prime p
    hdiv : Not (Dvd.dvd p n)
    R : Type u_1
    inst✝ : CommRing R
    hnpos : GT.gt n 0
    ⊢ Eq ((Polynomial.expand R p) (Polynomial.cyclotomic n R)) (HMul.hMul (Polynom …
  -/
  haveI := NeZero.of_pos hnpos
  suffices expand ℤ p (cyclotomic n ℤ) = cyclotomic (n * p) ℤ * cyclotomic n ℤ by
    rw [← map_cyclotomic_int, ← map_expand, this, Polynomial.map_mul, map_cyclotomic_int,
      map_cyclotomic]
  refine eq_of_monic_of_dvd_of_natDegree_le ((cyclotomic.monic _ ℤ).mul (cyclotomic.monic _ ℤ))
    ((cyclotomic.monic n ℤ).expand hp.pos) ?_ ?_
  · refine (IsPrimitive.Int.dvd_iff_map_cast_dvd_map_cast _ _
      (IsPrimitive.mul (cyclotomic.isPrimitive (n * p) ℤ) (cyclotomic.isPrimitive n ℤ))
      ((cyclotomic.monic n ℤ).expand hp.pos).isPrimitive).2 ?_
    /-
      case inr.refine_1
      p n : Nat
      hp : Nat.Prime p
      hdiv : Not (Dvd.dvd p n)
      R : Type u_1
      inst✝ : CommRing R
      hnpos : GT.gt n 0
      this : NeZero n
      ⊢ Dvd.dvd (Polynomial.map (Int.castRingHom Rat) (HMul.hMul (Polynomial.cycloto …
    -/
    rw [Polynomial.map_mul, map_cyclotomic_int, map_cyclotomic_int, map_expand, map_cyclotomic_int]
    /-
      case inr.refine_1
      p n : Nat
      hp : Nat.Prime p
      hdiv : Not (Dvd.dvd p n)
      R : Type u_1
      inst✝ : CommRing R
      hnpos : GT.gt n 0
      this : NeZero n
      ⊢ Dvd.dvd (HMul.hMul (Polynomial.cyclotomic (HMul.hMul n p) Rat) (Polynomial.c …
    -/
    refine IsCoprime.mul_dvd (cyclotomic.isCoprime_rat fun h => ?_) ?_ ?_
      /-
        case inr.refine_1.refine_1
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        h : Eq (HMul.hMul n p) n
        ⊢ False
      -/
    · replace h : n * p = n * 1 := by simp [h]
      /-
        case inr.refine_1.refine_1
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        h : Eq (HMul.hMul n p) (HMul.hMul n 1)
        ⊢ False
      -/
      exact Nat.Prime.ne_one hp (mul_left_cancel₀ hnpos.ne' h)
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_1.refine_2
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        ⊢ Dvd.dvd (Polynomial.cyclotomic (HMul.hMul n p) Rat) ((Polynomial.expand Rat  …
      -/
    · have hpos : 0 < n * p := mul_pos hnpos hp.pos
      /-
        case inr.refine_1.refine_2
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        hpos : LT.lt 0 (HMul.hMul n p)
        ⊢ Dvd.dvd (Polynomial.cyclotomic (HMul.hMul n p) Rat) ((Polynomial.expand Rat  …
      -/
      have hprim := Complex.isPrimitiveRoot_exp _ hpos.ne'
      /-
        case inr.refine_1.refine_2
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        hpos : LT.lt 0 (HMul.hMul n p)
        hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
        ⊢ Dvd.dvd (Polynomial.cyclotomic (HMul.hMul n p) Rat) ((Polynomial.expand Rat  …
      -/
      rw [cyclotomic_eq_minpoly_rat hprim hpos]
      /-
        case inr.refine_1.refine_2
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        hpos : LT.lt 0 (HMul.hMul n p)
        hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
        ⊢ Dvd.dvd (minpoly Rat (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.p …
      -/
      refine minpoly.dvd ℚ _ ?_
      rw [aeval_def, ← eval_map, map_expand, map_cyclotomic, expand_eval, ← IsRoot.def,
        @isRoot_cyclotomic_iff]
      /-
        case inr.refine_1.refine_2
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        hpos : LT.lt 0 (HMul.hMul n p)
        hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
        ⊢ IsPrimitiveRoot (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑ …
      -/
      convert IsPrimitiveRoot.pow_of_dvd hprim hp.ne_zero (dvd_mul_left p n)
      /-
        case h.e'_4
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        hpos : LT.lt 0 (HMul.hMul n p)
        hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
        ⊢ Eq n (HDiv.hDiv (HMul.hMul n p) p)
      -/
      rw [Nat.mul_div_cancel _ (Nat.Prime.pos hp)]
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_1.refine_3
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        ⊢ Dvd.dvd (Polynomial.cyclotomic n Rat) ((Polynomial.expand Rat p) (Polynomial …
      -/
    · have hprim := Complex.isPrimitiveRoot_exp _ hnpos.ne.symm
      /-
        case inr.refine_1.refine_3
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
        ⊢ Dvd.dvd (Polynomial.cyclotomic n Rat) ((Polynomial.expand Rat p) (Polynomial …
      -/
      rw [cyclotomic_eq_minpoly_rat hprim hnpos]
      /-
        case inr.refine_1.refine_3
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
        ⊢ Dvd.dvd (minpoly Rat (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.p …
      -/
      refine minpoly.dvd ℚ _ ?_
      rw [aeval_def, ← eval_map, map_expand, expand_eval, ← IsRoot.def, ←
        cyclotomic_eq_minpoly_rat hprim hnpos, map_cyclotomic, @isRoot_cyclotomic_iff]
      /-
        case inr.refine_1.refine_3
        p n : Nat
        hp : Nat.Prime p
        hdiv : Not (Dvd.dvd p n)
        R : Type u_1
        inst✝ : CommRing R
        hnpos : GT.gt n 0
        this : NeZero n
        hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
        ⊢ IsPrimitiveRoot (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑ …
      -/
      exact IsPrimitiveRoot.pow_of_prime hprim hp hdiv
      /-
        🎉 no goals
      -/
  · rw [natDegree_expand, natDegree_cyclotomic,
      natDegree_mul (cyclotomic_ne_zero _ ℤ) (cyclotomic_ne_zero _ ℤ), natDegree_cyclotomic,
      natDegree_cyclotomic, mul_comm n,
      Nat.totient_mul ((Nat.Prime.coprime_iff_not_dvd hp).2 hdiv), Nat.totient_prime hp,
      mul_comm (p - 1), ← Nat.mul_succ, Nat.sub_one, Nat.succ_pred_eq_of_pos hp.pos]


/-- If `p` is a prime such that `p ∣ n`, then
`expand R p (cyclotomic n R) = cyclotomic (p * n) R`. -/
@[simp]
theorem cyclotomic_expand_eq_cyclotomic {p n : ℕ} (hp : Nat.Prime p) (hdiv : p ∣ n) (R : Type*)
    [CommRing R] : expand R p (cyclotomic n R) = cyclotomic (n * p) R := by
  /-
    p n : Nat
    hp : Nat.Prime p
    hdiv : Dvd.dvd p n
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq ((Polynomial.expand R p) (Polynomial.cyclotomic n R)) (Polynomial.cycloto …
  -/
  rcases n.eq_zero_or_pos with (rfl | hzero)
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      R : Type u_1
      inst✝ : CommRing R
      hdiv : Dvd.dvd p 0
      ⊢ Eq ((Polynomial.expand R p) (Polynomial.cyclotomic 0 R)) (Polynomial.cycloto …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p n : Nat
    hp : Nat.Prime p
    hdiv : Dvd.dvd p n
    R : Type u_1
    inst✝ : CommRing R
    hzero : GT.gt n 0
    ⊢ Eq ((Polynomial.expand R p) (Polynomial.cyclotomic n R)) (Polynomial.cycloto …
  -/
  haveI := NeZero.of_pos hzero
  suffices expand ℤ p (cyclotomic n ℤ) = cyclotomic (n * p) ℤ by
    rw [← map_cyclotomic_int, ← map_expand, this, map_cyclotomic_int]
  refine eq_of_monic_of_dvd_of_natDegree_le (cyclotomic.monic _ ℤ)
    ((cyclotomic.monic n ℤ).expand hp.pos) ?_ ?_
    /-
      case inr.refine_1
      p n : Nat
      hp : Nat.Prime p
      hdiv : Dvd.dvd p n
      R : Type u_1
      inst✝ : CommRing R
      hzero : GT.gt n 0
      this : NeZero n
      ⊢ Dvd.dvd (Polynomial.cyclotomic (HMul.hMul n p) Int) ((Polynomial.expand Int  …
    -/
  · have hpos := Nat.mul_pos hzero hp.pos
    /-
      case inr.refine_1
      p n : Nat
      hp : Nat.Prime p
      hdiv : Dvd.dvd p n
      R : Type u_1
      inst✝ : CommRing R
      hzero : GT.gt n 0
      this : NeZero n
      hpos : GT.gt (HMul.hMul n p) 0
      ⊢ Dvd.dvd (Polynomial.cyclotomic (HMul.hMul n p) Int) ((Polynomial.expand Int  …
    -/
    have hprim := Complex.isPrimitiveRoot_exp _ hpos.ne.symm
    /-
      case inr.refine_1
      p n : Nat
      hp : Nat.Prime p
      hdiv : Dvd.dvd p n
      R : Type u_1
      inst✝ : CommRing R
      hzero : GT.gt n 0
      this : NeZero n
      hpos : GT.gt (HMul.hMul n p) 0
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      ⊢ Dvd.dvd (Polynomial.cyclotomic (HMul.hMul n p) Int) ((Polynomial.expand Int  …
    -/
    rw [cyclotomic_eq_minpoly hprim hpos]
    /-
      case inr.refine_1
      p n : Nat
      hp : Nat.Prime p
      hdiv : Dvd.dvd p n
      R : Type u_1
      inst✝ : CommRing R
      hzero : GT.gt n 0
      this : NeZero n
      hpos : GT.gt (HMul.hMul n p) 0
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      ⊢ Dvd.dvd (minpoly Int (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real.p …
    -/
    refine minpoly.isIntegrallyClosed_dvd (hprim.isIntegral hpos) ?_
    rw [aeval_def, ← eval_map, map_expand, map_cyclotomic, expand_eval, ← IsRoot.def,
      @isRoot_cyclotomic_iff]
    /-
      case inr.refine_1
      p n : Nat
      hp : Nat.Prime p
      hdiv : Dvd.dvd p n
      R : Type u_1
      inst✝ : CommRing R
      hzero : GT.gt n 0
      this : NeZero n
      hpos : GT.gt (HMul.hMul n p) 0
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      ⊢ IsPrimitiveRoot (HPow.hPow (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑ …
    -/
    convert IsPrimitiveRoot.pow_of_dvd hprim hp.ne_zero (dvd_mul_left p n)
    /-
      case h.e'_4
      p n : Nat
      hp : Nat.Prime p
      hdiv : Dvd.dvd p n
      R : Type u_1
      inst✝ : CommRing R
      hzero : GT.gt n 0
      this : NeZero n
      hpos : GT.gt (HMul.hMul n p) 0
      hprim : IsPrimitiveRoot (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑Real. …
      ⊢ Eq n (HDiv.hDiv (HMul.hMul n p) p)
    -/
    rw [Nat.mul_div_cancel _ hp.pos]
    /-
      🎉 no goals
    -/
  · rw [natDegree_expand, natDegree_cyclotomic, natDegree_cyclotomic, mul_comm n,
      Nat.totient_mul_of_prime_of_dvd hp hdiv, mul_comm]


/-- If the `p ^ n`th cyclotomic polynomial is irreducible, so is the `p ^ m`th, for `m ≤ n`. -/
theorem cyclotomic_irreducible_pow_of_irreducible_pow {p : ℕ} (hp : Nat.Prime p) {R} [CommRing R]
    [IsDomain R] {n m : ℕ} (hmn : m ≤ n) (h : Irreducible (cyclotomic (p ^ n) R)) :
    Irreducible (cyclotomic (p ^ m) R) := by
  /-
    p : Nat
    hp : Nat.Prime p
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n m : Nat
    hmn : LE.le m n
    h : Irreducible (Polynomial.cyclotomic (HPow.hPow p n) R)
    ⊢ Irreducible (Polynomial.cyclotomic (HPow.hPow p m) R)
  -/
  rcases m.eq_zero_or_pos with (rfl | hm)
    /-
      case inl
      p : Nat
      hp : Nat.Prime p
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      h : Irreducible (Polynomial.cyclotomic (HPow.hPow p n) R)
      hmn : LE.le 0 n
      ⊢ Irreducible (Polynomial.cyclotomic (HPow.hPow p 0) R)
    -/
  · simpa using irreducible_X_sub_C (1 : R)
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Nat
    hp : Nat.Prime p
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n m : Nat
    hmn : LE.le m n
    h : Irreducible (Polynomial.cyclotomic (HPow.hPow p n) R)
    hm : GT.gt m 0
    ⊢ Irreducible (Polynomial.cyclotomic (HPow.hPow p m) R)
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hmn
  /-
    case inr.intro
    p : Nat
    hp : Nat.Prime p
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    m : Nat
    hm : GT.gt m 0
    k : Nat
    hmn : LE.le m (HAdd.hAdd m k)
    h : Irreducible (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m k)) R)
    ⊢ Irreducible (Polynomial.cyclotomic (HPow.hPow p m) R)
  -/
  induction' k with k hk
    /-
      case inr.intro.zero
      p : Nat
      hp : Nat.Prime p
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      m : Nat
      hm : GT.gt m 0
      hmn : LE.le m (HAdd.hAdd m 0)
      h : Irreducible (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m 0)) R)
      ⊢ Irreducible (Polynomial.cyclotomic (HPow.hPow p m) R)
    -/
  · simpa using h
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.succ
    p : Nat
    hp : Nat.Prime p
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    m : Nat
    hm : GT.gt m 0
    k : Nat
    hk : LE.le m (HAdd.hAdd m k) → Irreducible (Polynomial.cyclotomic (HPow.hPow p …
    hmn : LE.le m (HAdd.hAdd m (HAdd.hAdd k 1))
    h : Irreducible (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m (HAdd.hAdd k  …
    ⊢ Irreducible (Polynomial.cyclotomic (HPow.hPow p m) R)
  -/
  have : m + k ≠ 0 := (add_pos_of_pos_of_nonneg hm k.zero_le).ne'
  /-
    case inr.intro.succ
    p : Nat
    hp : Nat.Prime p
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    m : Nat
    hm : GT.gt m 0
    k : Nat
    hk : LE.le m (HAdd.hAdd m k) → Irreducible (Polynomial.cyclotomic (HPow.hPow p …
    hmn : LE.le m (HAdd.hAdd m (HAdd.hAdd k 1))
    h : Irreducible (Polynomial.cyclotomic (HPow.hPow p (HAdd.hAdd m (HAdd.hAdd k  …
    this : Ne (HAdd.hAdd m k) 0
    ⊢ Irreducible (Polynomial.cyclotomic (HPow.hPow p m) R)
  -/
  rw [Nat.add_succ, pow_succ, ← cyclotomic_expand_eq_cyclotomic hp <| dvd_pow_self p this] at h
  /-
    case inr.intro.succ
    p : Nat
    hp : Nat.Prime p
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    m : Nat
    hm : GT.gt m 0
    k : Nat
    hk : LE.le m (HAdd.hAdd m k) → Irreducible (Polynomial.cyclotomic (HPow.hPow p …
    hmn : LE.le m (HAdd.hAdd m (HAdd.hAdd k 1))
    h : Irreducible ((Polynomial.expand R p) (Polynomial.cyclotomic (HPow.hPow p ( …
    this : Ne (HAdd.hAdd m k) 0
    ⊢ Irreducible (Polynomial.cyclotomic (HPow.hPow p m) R)
  -/
  exact hk (by omega) (of_irreducible_expand hp.ne_zero h)
  /-
    🎉 no goals
  -/


/-- If `Irreducible (cyclotomic (p ^ n) R)` then `Irreducible (cyclotomic p R).` -/
theorem cyclotomic_irreducible_of_irreducible_pow {p : ℕ} (hp : Nat.Prime p) {R} [CommRing R]
    [IsDomain R] {n : ℕ} (hn : n ≠ 0) (h : Irreducible (cyclotomic (p ^ n) R)) :
    Irreducible (cyclotomic p R) :=
  pow_one p ▸ cyclotomic_irreducible_pow_of_irreducible_pow hp hn.bot_lt h


/-- If `R` is of characteristic `p` and `¬p ∣ n`, then
`cyclotomic (n * p) R = (cyclotomic n R) ^ (p - 1)`. -/
theorem cyclotomic_mul_prime_eq_pow_of_not_dvd (R : Type*) {p n : ℕ} [hp : Fact (Nat.Prime p)]
    [Ring R] [CharP R p] (hn : ¬p ∣ n) : cyclotomic (n * p) R = cyclotomic n R ^ (p - 1) := by
  /-
    R : Type u_1
    p n : Nat
    hp : Fact (Nat.Prime p)
    inst✝¹ : Ring R
    inst✝ : CharP R p
    hn : Not (Dvd.dvd p n)
    ⊢ Eq (Polynomial.cyclotomic (HMul.hMul n p) R) (HPow.hPow (Polynomial.cyclotom …
  -/
  letI : Algebra (ZMod p) R := ZMod.algebra _ _
  suffices cyclotomic (n * p) (ZMod p) = cyclotomic n (ZMod p) ^ (p - 1) by
    rw [← map_cyclotomic _ (algebraMap (ZMod p) R), ← map_cyclotomic _ (algebraMap (ZMod p) R),
      this, Polynomial.map_pow]
  /-
    R : Type u_1
    p n : Nat
    hp : Fact (Nat.Prime p)
    inst✝¹ : Ring R
    inst✝ : CharP R p
    hn : Not (Dvd.dvd p n)
    this : Algebra (ZMod p) R := ZMod.algebra R p
    ⊢ Eq (Polynomial.cyclotomic (HMul.hMul n p) (ZMod p)) (HPow.hPow (Polynomial.c …
  -/
  apply mul_right_injective₀ (cyclotomic_ne_zero n <| ZMod p); dsimp
  /-
    case a
    R : Type u_1
    p n : Nat
    hp : Fact (Nat.Prime p)
    inst✝¹ : Ring R
    inst✝ : CharP R p
    hn : Not (Dvd.dvd p n)
    this : Algebra (ZMod p) R := ZMod.algebra R p
    ⊢ Eq (HMul.hMul (Polynomial.cyclotomic n (ZMod p)) (Polynomial.cyclotomic (HMu …
  -/
  rw [← pow_succ', tsub_add_cancel_of_le hp.out.one_lt.le, mul_comm, ← ZMod.expand_card]
  /-
    case a
    R : Type u_1
    p n : Nat
    hp : Fact (Nat.Prime p)
    inst✝¹ : Ring R
    inst✝ : CharP R p
    hn : Not (Dvd.dvd p n)
    this : Algebra (ZMod p) R := ZMod.algebra R p
    ⊢ Eq (HMul.hMul (Polynomial.cyclotomic (HMul.hMul n p) (ZMod p)) (Polynomial.c …
  -/
  conv_rhs => rw [← map_cyclotomic_int]
  rw [← map_expand, cyclotomic_expand_eq_cyclotomic_mul hp.out hn, Polynomial.map_mul,
    map_cyclotomic, map_cyclotomic]


/-- If `R` is of characteristic `p` and `p ∣ n`, then
`cyclotomic (n * p) R = (cyclotomic n R) ^ p`. -/
theorem cyclotomic_mul_prime_dvd_eq_pow (R : Type*) {p n : ℕ} [hp : Fact (Nat.Prime p)] [Ring R]
    [CharP R p] (hn : p ∣ n) : cyclotomic (n * p) R = cyclotomic n R ^ p := by
  /-
    R : Type u_1
    p n : Nat
    hp : Fact (Nat.Prime p)
    inst✝¹ : Ring R
    inst✝ : CharP R p
    hn : Dvd.dvd p n
    ⊢ Eq (Polynomial.cyclotomic (HMul.hMul n p) R) (HPow.hPow (Polynomial.cyclotom …
  -/
  letI : Algebra (ZMod p) R := ZMod.algebra _ _
  suffices cyclotomic (n * p) (ZMod p) = cyclotomic n (ZMod p) ^ p by
    rw [← map_cyclotomic _ (algebraMap (ZMod p) R), ← map_cyclotomic _ (algebraMap (ZMod p) R),
      this, Polynomial.map_pow]
  rw [← ZMod.expand_card, ← map_cyclotomic_int n, ← map_expand,
    cyclotomic_expand_eq_cyclotomic hp.out hn, map_cyclotomic]


/-- If `R` is of characteristic `p` and `¬p ∣ m`, then
`cyclotomic (p ^ k * m) R = (cyclotomic m R) ^ (p ^ k - p ^ (k - 1))`. -/
theorem cyclotomic_mul_prime_pow_eq (R : Type*) {p m : ℕ} [Fact (Nat.Prime p)] [Ring R] [CharP R p]
    (hm : ¬p ∣ m) : ∀ {k}, 0 < k → cyclotomic (p ^ k * m) R = cyclotomic m R ^ (p ^ k - p ^ (k - 1))
  | 1, _ => by
    /-
      R : Type u_1
      p m : Nat
      inst✝² : Fact (Nat.Prime p)
      inst✝¹ : Ring R
      inst✝ : CharP R p
      hm : Not (Dvd.dvd p m)
      x✝ : LT.lt 0 1
      ⊢ Eq (Polynomial.cyclotomic (HMul.hMul (HPow.hPow p 1) m) R) (HPow.hPow (Polyn …
    -/
    rw [pow_one, Nat.sub_self, pow_zero, mul_comm, cyclotomic_mul_prime_eq_pow_of_not_dvd R hm]
    /-
      🎉 no goals
    -/
  | a + 2, _ => by
    /-
      R : Type u_1
      p m : Nat
      inst✝² : Fact (Nat.Prime p)
      inst✝¹ : Ring R
      inst✝ : CharP R p
      hm : Not (Dvd.dvd p m)
      a : Nat
      x✝ : LT.lt 0 (HAdd.hAdd a 2)
      ⊢ Eq (Polynomial.cyclotomic (HMul.hMul (HPow.hPow p (HAdd.hAdd a 2)) m) R) (HP …
    -/
    have hdiv : p ∣ p ^ a.succ * m := ⟨p ^ a * m, by rw [← mul_assoc, pow_succ']⟩
    rw [pow_succ', mul_assoc, mul_comm, cyclotomic_mul_prime_dvd_eq_pow R hdiv,
      cyclotomic_mul_prime_pow_eq _ _ a.succ_pos, ← pow_mul]
      /-
        R : Type u_1
        p m : Nat
        inst✝² : Fact (Nat.Prime p)
        inst✝¹ : Ring R
        inst✝ : CharP R p
        hm : Not (Dvd.dvd p m)
        a : Nat
        x✝ : LT.lt 0 (HAdd.hAdd a 2)
        hdiv : Dvd.dvd p (HMul.hMul (HPow.hPow p a.succ) m)
        ⊢ Eq (HPow.hPow (Polynomial.cyclotomic m R) (HMul.hMul (HSub.hSub (HPow.hPow p …
      -/
    · simp only [tsub_zero, Nat.succ_sub_succ_eq_sub]
      /-
        R : Type u_1
        p m : Nat
        inst✝² : Fact (Nat.Prime p)
        inst✝¹ : Ring R
        inst✝ : CharP R p
        hm : Not (Dvd.dvd p m)
        a : Nat
        x✝ : LT.lt 0 (HAdd.hAdd a 2)
        hdiv : Dvd.dvd p (HMul.hMul (HPow.hPow p a.succ) m)
        ⊢ Eq (HPow.hPow (Polynomial.cyclotomic m R) (HMul.hMul (HSub.hSub (HPow.hPow p …
      -/
      rw [Nat.mul_sub_right_distrib, mul_comm, pow_succ]
      /-
        🎉 no goals
      -/
      /-
        R : Type u_1
        p m : Nat
        inst✝² : Fact (Nat.Prime p)
        inst✝¹ : Ring R
        inst✝ : CharP R p
        hm : Not (Dvd.dvd p m)
        a : Nat
        x✝ : LT.lt 0 (HAdd.hAdd a 2)
        hdiv : Dvd.dvd p (HMul.hMul (HPow.hPow p a.succ) m)
        ⊢ Not (Dvd.dvd p m)
      -/
    · assumption
      /-
        🎉 no goals
      -/


/-- If `R` is of characteristic `p` and `¬p ∣ m`, then `ζ` is a root of `cyclotomic (p ^ k * m) R`
 if and only if it is a primitive `m`-th root of unity. -/
theorem isRoot_cyclotomic_prime_pow_mul_iff_of_charP {m k p : ℕ} {R : Type*} [CommRing R]
    [IsDomain R] [hp : Fact (Nat.Prime p)] [hchar : CharP R p] {μ : R} [NeZero (m : R)] :
    (Polynomial.cyclotomic (p ^ k * m) R).IsRoot μ ↔ IsPrimitiveRoot μ m := by
  /-
    m k p : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    hp : Fact (Nat.Prime p)
    hchar : CharP R p
    μ : R
    inst✝ : NeZero ↑m
    ⊢ Iff ((Polynomial.cyclotomic (HMul.hMul (HPow.hPow p k) m) R).IsRoot μ) (IsPr …
  -/
  rcases k.eq_zero_or_pos with (rfl | hk)
    /-
      case inl
      m p : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hp : Fact (Nat.Prime p)
      hchar : CharP R p
      μ : R
      inst✝ : NeZero ↑m
      ⊢ Iff ((Polynomial.cyclotomic (HMul.hMul (HPow.hPow p 0) m) R).IsRoot μ) (IsPr …
    -/
  · rw [pow_zero, one_mul, isRoot_cyclotomic_iff]
    /-
      🎉 no goals
    -/
  /-
    case inr
    m k p : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    hp : Fact (Nat.Prime p)
    hchar : CharP R p
    μ : R
    inst✝ : NeZero ↑m
    hk : GT.gt k 0
    ⊢ Iff ((Polynomial.cyclotomic (HMul.hMul (HPow.hPow p k) m) R).IsRoot μ) (IsPr …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
  · rw [IsRoot.def, cyclotomic_mul_prime_pow_eq R (NeZero.not_char_dvd R p m) hk, eval_pow]
      at h
    /-
      case inr.refine_1
      m k p : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hp : Fact (Nat.Prime p)
      hchar : CharP R p
      μ : R
      inst✝ : NeZero ↑m
      hk : GT.gt k 0
      h : Eq (HPow.hPow (Polynomial.eval μ (Polynomial.cyclotomic m R)) (HSub.hSub ( …
      ⊢ IsPrimitiveRoot μ m
    -/
    replace h := pow_eq_zero h
    /-
      case inr.refine_1
      m k p : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hp : Fact (Nat.Prime p)
      hchar : CharP R p
      μ : R
      inst✝ : NeZero ↑m
      hk : GT.gt k 0
      h : Eq (Polynomial.eval μ (Polynomial.cyclotomic m R)) 0
      ⊢ IsPrimitiveRoot μ m
    -/
    rwa [← IsRoot.def, isRoot_cyclotomic_iff] at h
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      m k p : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hp : Fact (Nat.Prime p)
      hchar : CharP R p
      μ : R
      inst✝ : NeZero ↑m
      hk : GT.gt k 0
      h : IsPrimitiveRoot μ m
      ⊢ (Polynomial.cyclotomic (HMul.hMul (HPow.hPow p k) m) R).IsRoot μ
    -/
  · rw [← isRoot_cyclotomic_iff, IsRoot.def] at h
    rw [cyclotomic_mul_prime_pow_eq R (NeZero.not_char_dvd R p m) hk, IsRoot.def, eval_pow,
      h, zero_pow]
    /-
      case inr.refine_2
      m k p : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      hp : Fact (Nat.Prime p)
      hchar : CharP R p
      μ : R
      inst✝ : NeZero ↑m
      hk : GT.gt k 0
      h : Eq (Polynomial.eval μ (Polynomial.cyclotomic m R)) 0
      ⊢ Ne (HSub.hSub (HPow.hPow p k) (HPow.hPow p (HSub.hSub k 1))) 0
    -/
    exact Nat.sub_ne_zero_of_lt <| pow_right_strictMono₀ hp.out.one_lt <| Nat.pred_lt hk.ne'
    /-
      🎉 no goals
    -/


