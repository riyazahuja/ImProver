/-- A real number is irrational if it is not equal to any rational number. -/
def Irrational (x : ℝ) :=
  x ∉ Set.range ((↑) : ℚ → ℝ)


theorem irrational_iff_ne_rational (x : ℝ) : Irrational x ↔ ∀ a b : ℤ, x ≠ a / b := by
  simp only [Irrational, Rat.forall, cast_mk, not_exists, Set.mem_range, cast_intCast, cast_div,
    eq_comm]


/-- A transcendental real number is irrational. -/
theorem Transcendental.irrational {r : ℝ} (tr : Transcendental ℚ r) : Irrational r := by
  /-
    r : Real
    tr : Transcendental Rat r
    ⊢ Irrational r
  -/
  rintro ⟨a, rfl⟩
  /-
    case intro
    a : Rat
    tr : Transcendental Rat ↑a
    ⊢ False
  -/
  exact tr (isAlgebraic_algebraMap a)
  /-
    🎉 no goals
  -/


/-- If `x^n`, `n > 0`, is integer and is not the `n`-th power of an integer, then
`x` is irrational. -/
theorem irrational_nrt_of_notint_nrt {x : ℝ} (n : ℕ) (m : ℤ) (hxr : x ^ n = m)
    (hv : ¬∃ y : ℤ, x = y) (hnpos : 0 < n) : Irrational x := by
  /-
    x : Real
    n : Nat
    m : Int
    hxr : Eq (HPow.hPow x n) ↑m
    hv : Not (Exists fun y => Eq x ↑y)
    hnpos : LT.lt 0 n
    ⊢ Irrational x
  -/
  rintro ⟨⟨N, D, P, C⟩, rfl⟩
  /-
    case intro.mk'
    n : Nat
    m : Int
    hnpos : LT.lt 0 n
    N : Int
    D : Nat
    P : Ne D 0
    C : N.natAbs.Coprime D
    hxr : Eq (HPow.hPow (↑{ num := N, den := D, den_nz := P, reduced := C }) n) ↑m
    hv : Not (Exists fun y => Eq ↑{ num := N, den := D, den_nz := P, reduced := C  …
    ⊢ False
  -/
  rw [← cast_pow] at hxr
  have c1 : ((D : ℤ) : ℝ) ≠ 0 := by
    rw [Int.cast_ne_zero, Int.natCast_ne_zero]
    exact P
  /-
    case intro.mk'
    n : Nat
    m : Int
    hnpos : LT.lt 0 n
    N : Int
    D : Nat
    P : Ne D 0
    C : N.natAbs.Coprime D
    hxr : Eq ↑(HPow.hPow { num := N, den := D, den_nz := P, reduced := C } n) ↑m
    hv : Not (Exists fun y => Eq ↑{ num := N, den := D, den_nz := P, reduced := C  …
    c1 : Ne (↑↑D) 0
    ⊢ False
  -/
  have c2 : ((D : ℤ) : ℝ) ^ n ≠ 0 := pow_ne_zero _ c1
  rw [mk'_eq_divInt, cast_pow, cast_mk, div_pow, div_eq_iff_mul_eq c2, ← Int.cast_pow,
    ← Int.cast_pow, ← Int.cast_mul, Int.cast_inj] at hxr
  /-
    case intro.mk'
    n : Nat
    m : Int
    hnpos : LT.lt 0 n
    N : Int
    D : Nat
    P : Ne D 0
    C : N.natAbs.Coprime D
    hxr : Eq (HMul.hMul m (HPow.hPow (↑D) n)) (HPow.hPow N n)
    hv : Not (Exists fun y => Eq ↑{ num := N, den := D, den_nz := P, reduced := C  …
    c1 : Ne (↑↑D) 0
    c2 : Ne (HPow.hPow (↑↑D) n) 0
    ⊢ False
  -/
  have hdivn : (D : ℤ) ^ n ∣ N ^ n := Dvd.intro_left m hxr
  rw [← Int.dvd_natAbs, ← Int.natCast_pow, Int.natCast_dvd_natCast, Int.natAbs_pow,
    Nat.pow_dvd_pow_iff hnpos.ne'] at hdivn
  /-
    case intro.mk'
    n : Nat
    m : Int
    hnpos : LT.lt 0 n
    N : Int
    D : Nat
    P : Ne D 0
    C : N.natAbs.Coprime D
    hxr : Eq (HMul.hMul m (HPow.hPow (↑D) n)) (HPow.hPow N n)
    hv : Not (Exists fun y => Eq ↑{ num := N, den := D, den_nz := P, reduced := C  …
    c1 : Ne (↑↑D) 0
    c2 : Ne (HPow.hPow (↑↑D) n) 0
    hdivn : Dvd.dvd D N.natAbs
    ⊢ False
  -/
  obtain rfl : D = 1 := by rw [← Nat.gcd_eq_right hdivn, C.gcd_eq_one]
  /-
    case intro.mk'
    n : Nat
    m : Int
    hnpos : LT.lt 0 n
    N : Int
    P : Ne 1 0
    C : N.natAbs.Coprime 1
    hxr : Eq (HMul.hMul m (HPow.hPow (↑1) n)) (HPow.hPow N n)
    hv : Not (Exists fun y => Eq ↑{ num := N, den := 1, den_nz := P, reduced := C  …
    c1 : Ne (↑↑1) 0
    c2 : Ne (HPow.hPow (↑↑1) n) 0
    hdivn : Dvd.dvd 1 N.natAbs
    ⊢ False
  -/
  refine hv ⟨N, ?_⟩
  /-
    case intro.mk'
    n : Nat
    m : Int
    hnpos : LT.lt 0 n
    N : Int
    P : Ne 1 0
    C : N.natAbs.Coprime 1
    hxr : Eq (HMul.hMul m (HPow.hPow (↑1) n)) (HPow.hPow N n)
    hv : Not (Exists fun y => Eq ↑{ num := N, den := 1, den_nz := P, reduced := C  …
    c1 : Ne (↑↑1) 0
    c2 : Ne (HPow.hPow (↑↑1) n) 0
    hdivn : Dvd.dvd 1 N.natAbs
    ⊢ Eq ↑{ num := N, den := 1, den_nz := P, reduced := C } ↑N
  -/
  rw [mk'_eq_divInt, Int.ofNat_one, divInt_one, cast_intCast]
  /-
    🎉 no goals
  -/


/-- If `x^n = m` is an integer and `n` does not divide the `multiplicity p m`, then `x`
is irrational. -/
theorem irrational_nrt_of_n_not_dvd_multiplicity {x : ℝ} (n : ℕ) {m : ℤ} (hm : m ≠ 0) (p : ℕ)
    [hp : Fact p.Prime] (hxr : x ^ n = m)
    (hv : multiplicity (p : ℤ) m % n ≠ 0) :
    Irrational x := by
  /-
    x : Real
    n : Nat
    m : Int
    hm : Ne m 0
    p : Nat
    hp : Fact (Nat.Prime p)
    hxr : Eq (HPow.hPow x n) ↑m
    hv : Ne (HMod.hMod (multiplicity (↑p) m) n) 0
    ⊢ Irrational x
  -/
  rcases Nat.eq_zero_or_pos n with (rfl | hnpos)
    /-
      case inl
      x : Real
      m : Int
      hm : Ne m 0
      p : Nat
      hp : Fact (Nat.Prime p)
      hxr : Eq (HPow.hPow x 0) ↑m
      hv : Ne (HMod.hMod (multiplicity (↑p) m) 0) 0
      ⊢ Irrational x
    -/
  · rw [eq_comm, pow_zero, ← Int.cast_one, Int.cast_inj] at hxr
    simp [hxr, multiplicity_of_one_right (mt isUnit_iff_dvd_one.1
      (mt Int.natCast_dvd_natCast.1 hp.1.not_dvd_one)), Nat.zero_mod] at hv
  /-
    case inr
    x : Real
    n : Nat
    m : Int
    hm : Ne m 0
    p : Nat
    hp : Fact (Nat.Prime p)
    hxr : Eq (HPow.hPow x n) ↑m
    hv : Ne (HMod.hMod (multiplicity (↑p) m) n) 0
    hnpos : GT.gt n 0
    ⊢ Irrational x
  -/
  refine irrational_nrt_of_notint_nrt _ _ hxr ?_ hnpos
  /-
    case inr
    x : Real
    n : Nat
    m : Int
    hm : Ne m 0
    p : Nat
    hp : Fact (Nat.Prime p)
    hxr : Eq (HPow.hPow x n) ↑m
    hv : Ne (HMod.hMod (multiplicity (↑p) m) n) 0
    hnpos : GT.gt n 0
    ⊢ Not (Exists fun y => Eq x ↑y)
  -/
  rintro ⟨y, rfl⟩
  /-
    case inr.intro
    n : Nat
    m : Int
    hm : Ne m 0
    p : Nat
    hp : Fact (Nat.Prime p)
    hv : Ne (HMod.hMod (multiplicity (↑p) m) n) 0
    hnpos : GT.gt n 0
    y : Int
    hxr : Eq (HPow.hPow (↑y) n) ↑m
    ⊢ False
  -/
  rw [← Int.cast_pow, Int.cast_inj] at hxr
  /-
    case inr.intro
    n : Nat
    m : Int
    hm : Ne m 0
    p : Nat
    hp : Fact (Nat.Prime p)
    hv : Ne (HMod.hMod (multiplicity (↑p) m) n) 0
    hnpos : GT.gt n 0
    y : Int
    hxr : Eq (HPow.hPow y n) m
    ⊢ False
  -/
  subst m
  /-
    case inr.intro
    n p : Nat
    hp : Fact (Nat.Prime p)
    hnpos : GT.gt n 0
    y : Int
    hm : Ne (HPow.hPow y n) 0
    hv : Ne (HMod.hMod (multiplicity (↑p) (HPow.hPow y n)) n) 0
    ⊢ False
  -/
  have : y ≠ 0 := by rintro rfl; rw [zero_pow hnpos.ne'] at hm; exact hm rfl
  rw [(Int.finiteMultiplicity_iff.2 ⟨by simp [hp.1.ne_one], this⟩).multiplicity_pow
    (Nat.prime_iff_prime_int.1 hp.1), Nat.mul_mod_right] at hv
  /-
    case inr.intro
    n p : Nat
    hp : Fact (Nat.Prime p)
    hnpos : GT.gt n 0
    y : Int
    hm : Ne (HPow.hPow y n) 0
    hv : Ne 0 0
    this : Ne y 0
    ⊢ False
  -/
  exact hv rfl
  /-
    🎉 no goals
  -/


theorem irrational_sqrt_of_multiplicity_odd (m : ℤ) (hm : 0 < m) (p : ℕ) [hp : Fact p.Prime]
    (Hpv : multiplicity (p : ℤ) m % 2 = 1) :
    Irrational (√m) :=
  @irrational_nrt_of_n_not_dvd_multiplicity _ 2 _ (Ne.symm (ne_of_lt hm)) p hp
                                                     /-
                                                       m : Int
                                                       hm : LT.lt 0 m
                                                       p : Nat
                                                       hp : Fact (Nat.Prime p)
                                                       Hpv : Eq (HMod.hMod (multiplicity (↑p) m) 2) 1
                                                       ⊢ Ne (HMod.hMod (multiplicity (↑p) m) 2) 0
                                                     -/
    (sq_sqrt (Int.cast_nonneg.2 <| le_of_lt hm)) (by rw [Hpv]; exact one_ne_zero)
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp] theorem not_irrational_zero : ¬Irrational 0 := not_not_intro ⟨0, Rat.cast_zero⟩

@[simp] theorem not_irrational_one : ¬Irrational 1 := not_not_intro ⟨1, Rat.cast_one⟩


theorem irrational_sqrt_ratCast_iff_of_nonneg {q : ℚ} (hq : 0 ≤ q) :
    Irrational (√q) ↔ ¬IsSquare q := by
  /-
    q : Rat
    hq : LE.le 0 q
    ⊢ Iff (Irrational (↑q).sqrt) (Not (IsSquare q))
  -/
  refine Iff.not (?_ : Exists _ ↔ Exists _)
  /-
    q : Rat
    hq : LE.le 0 q
    ⊢ Iff (Exists fun y => Eq (↑y) (↑q).sqrt) (Exists fun r => Eq q (HMul.hMul r r))
  -/
  constructor
    /-
      case mp
      q : Rat
      hq : LE.le 0 q
      ⊢ (Exists fun y => Eq (↑y) (↑q).sqrt) → Exists fun r => Eq q (HMul.hMul r r)
    -/
  · rintro ⟨y, hy⟩
    /-
      case mp.intro
      q : Rat
      hq : LE.le 0 q
      y : Rat
      hy : Eq (↑y) (↑q).sqrt
      ⊢ Exists fun r => Eq q (HMul.hMul r r)
    -/
    refine ⟨y, Rat.cast_injective (α := ℝ) ?_⟩
    /-
      case mp.intro
      q : Rat
      hq : LE.le 0 q
      y : Rat
      hy : Eq (↑y) (↑q).sqrt
      ⊢ Eq ↑q ↑(HMul.hMul y y)
    -/
    rw [Rat.cast_mul, hy, mul_self_sqrt (Rat.cast_nonneg.2 hq)]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      q : Rat
      hq : LE.le 0 q
      ⊢ (Exists fun r => Eq q (HMul.hMul r r)) → Exists fun y => Eq (↑y) (↑q).sqrt
    -/
  · rintro ⟨q', rfl⟩
    /-
      case mpr.intro
      q' : Rat
      hq : LE.le 0 (HMul.hMul q' q')
      ⊢ Exists fun y => Eq (↑y) (↑(HMul.hMul q' q')).sqrt
    -/
    exact ⟨|q'|, mod_cast (sqrt_mul_self_eq_abs q').symm⟩
    /-
      🎉 no goals
    -/


theorem irrational_sqrt_ratCast_iff {q : ℚ} :
    Irrational (√q) ↔ ¬IsSquare q ∧ 0 ≤ q := by
  /-
    q : Rat
    ⊢ Iff (Irrational (↑q).sqrt) (And (Not (IsSquare q)) (LE.le 0 q))
  -/
  obtain hq | hq := le_or_lt 0 q
    /-
      case inl
      q : Rat
      hq : LE.le 0 q
      ⊢ Iff (Irrational (↑q).sqrt) (And (Not (IsSquare q)) (LE.le 0 q))
    -/
  · simp_rw [irrational_sqrt_ratCast_iff_of_nonneg hq, and_iff_left hq]
    /-
      🎉 no goals
    -/
    /-
      case inr
      q : Rat
      hq : LT.lt q 0
      ⊢ Iff (Irrational (↑q).sqrt) (And (Not (IsSquare q)) (LE.le 0 q))
    -/
  · rw [sqrt_eq_zero_of_nonpos (Rat.cast_nonpos.2 hq.le)]
    /-
      case inr
      q : Rat
      hq : LT.lt q 0
      ⊢ Iff (Irrational 0) (And (Not (IsSquare q)) (LE.le 0 q))
    -/
    simp_rw [not_irrational_zero, false_iff, not_and, not_le, hq, implies_true]
    /-
      🎉 no goals
    -/


theorem irrational_sqrt_intCast_iff_of_nonneg {z : ℤ} (hz : 0 ≤ z) :
    Irrational (√z) ↔ ¬IsSquare z := by
  rw [← Rat.isSquare_intCast_iff, ← irrational_sqrt_ratCast_iff_of_nonneg (mod_cast hz),
    Rat.cast_intCast]


theorem irrational_sqrt_intCast_iff {z : ℤ} :
    Irrational (√z) ↔ ¬IsSquare z ∧ 0 ≤ z := by
  /-
    z : Int
    ⊢ Iff (Irrational (↑z).sqrt) (And (Not (IsSquare z)) (LE.le 0 z))
  -/
  rw [← Rat.cast_intCast, irrational_sqrt_ratCast_iff, Rat.isSquare_intCast_iff, Int.cast_nonneg]
  /-
    🎉 no goals
  -/


theorem irrational_sqrt_natCast_iff {n : ℕ} : Irrational (√n) ↔ ¬IsSquare n := by
  rw [← Rat.isSquare_natCast_iff, ← irrational_sqrt_ratCast_iff_of_nonneg n.cast_nonneg,
    Rat.cast_natCast]


theorem irrational_sqrt_ofNat_iff {n : ℕ} [n.AtLeastTwo] :
    Irrational √(ofNat(n)) ↔ ¬IsSquare ofNat(n) :=
  irrational_sqrt_natCast_iff


theorem Nat.Prime.irrational_sqrt {p : ℕ} (hp : Nat.Prime p) : Irrational (√p) :=
  irrational_sqrt_natCast_iff.mpr hp.not_square


/-- **Irrationality of the Square Root of 2** -/
theorem irrational_sqrt_two : Irrational (√2) := by
  /-
    ⊢ Irrational (Real.sqrt 2)
  -/
  simpa using Nat.prime_two.irrational_sqrt
  /-
    🎉 no goals
  -/


@[deprecated irrational_sqrt_ratCast_iff (since := "2024-06-16")]
theorem irrational_sqrt_rat_iff (q : ℚ) :
    Irrational (√q) ↔ Rat.sqrt q * Rat.sqrt q ≠ q ∧ 0 ≤ q := by
  /-
    q : Rat
    ⊢ Iff (Irrational (↑q).sqrt) (And (Ne (HMul.hMul (Rat.sqrt q) (Rat.sqrt q)) q) …
  -/
  rw [irrational_sqrt_ratCast_iff, ne_eq, ← Rat.exists_mul_self]
  /-
    q : Rat
    ⊢ Iff (And (Not (IsSquare q)) (LE.le 0 q)) (And (Not (Exists fun q_1 => Eq (HM …
  -/
  simp only [eq_comm, IsSquare]
  /-
    🎉 no goals
  -/


/--
This can be used as
```lean
unseal Nat.sqrt.iter in
example : Irrational √24 := by decide
```
-/
instance {n : ℕ} [n.AtLeastTwo] : Decidable (Irrational √(ofNat(n))) :=
  decidable_of_iff' _ irrational_sqrt_ofNat_iff


instance (n : ℕ) : Decidable (Irrational (√n)) :=
  decidable_of_iff' _ irrational_sqrt_natCast_iff


instance (z : ℤ) : Decidable (Irrational (√z)) :=
  decidable_of_iff' _ irrational_sqrt_intCast_iff


instance (q : ℚ) : Decidable (Irrational (√q)) :=
  decidable_of_iff' _ irrational_sqrt_ratCast_iff


theorem ne_rat (h : Irrational x) (q : ℚ) : x ≠ q := fun hq => h ⟨q, hq.symm⟩


theorem ne_int (h : Irrational x) (m : ℤ) : x ≠ m := by
  /-
    x : Real
    h : Irrational x
    m : Int
    ⊢ Ne x ↑m
  -/
  rw [← Rat.cast_intCast]
  /-
    x : Real
    h : Irrational x
    m : Int
    ⊢ Ne x ↑↑m
  -/
  exact h.ne_rat _
  /-
    🎉 no goals
  -/


theorem ne_nat (h : Irrational x) (m : ℕ) : x ≠ m :=
  h.ne_int m


theorem ne_zero (h : Irrational x) : x ≠ 0 := mod_cast h.ne_nat 0


                                                /-
                                                  x : Real
                                                  h : Irrational x
                                                  ⊢ Ne x 1
                                                -/
theorem ne_one (h : Irrational x) : x ≠ 1 := by simpa only [Nat.cast_one] using h.ne_nat 1
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] theorem ne_ofNat (h : Irrational x) (n : ℕ) [n.AtLeastTwo] : x ≠ ofNat(n) :=
  h.ne_nat n


@[simp]
theorem Rat.not_irrational (q : ℚ) : ¬Irrational q := fun h => h ⟨q, rfl⟩


@[simp]
theorem Int.not_irrational (m : ℤ) : ¬Irrational m := fun h => h.ne_int m rfl


@[simp]
theorem Nat.not_irrational (m : ℕ) : ¬Irrational m := fun h => h.ne_nat m rfl


@[simp] theorem not_irrational_ofNat (n : ℕ) [n.AtLeastTwo] : ¬Irrational ofNat(n) :=
  n.not_irrational

/-- If `x + y` is irrational, then at least one of `x` and `y` is irrational. -/
theorem add_cases : Irrational (x + y) → Irrational x ∨ Irrational y := by
  /-
    x y : Real
    ⊢ Irrational (HAdd.hAdd x y) → Or (Irrational x) (Irrational y)
  -/
  delta Irrational
  /-
    x y : Real
    ⊢ Not (Membership.mem (Set.range Rat.cast) (HAdd.hAdd x y)) → Or (Not (Members …
  -/
  contrapose!
  /-
    x y : Real
    ⊢ And (Membership.mem (Set.range Rat.cast) x) (Membership.mem (Set.range Rat.c …
  -/
  rintro ⟨⟨rx, rfl⟩, ⟨ry, rfl⟩⟩
  /-
    case intro.intro.intro
    rx ry : Rat
    ⊢ Membership.mem (Set.range Rat.cast) (HAdd.hAdd ↑rx ↑ry)
  -/
  exact ⟨rx + ry, cast_add rx ry⟩
  /-
    🎉 no goals
  -/


theorem of_rat_add (h : Irrational (q + x)) : Irrational x :=
  h.add_cases.resolve_left q.not_irrational


theorem rat_add (h : Irrational x) : Irrational (q + x) :=
                        /-
                          q : Rat
                          x : Real
                          h : Irrational x
                          ⊢ Irrational (HAdd.hAdd (↑(Neg.neg q)) (HAdd.hAdd (↑q) x))
                        -/
  of_rat_add (-q) <| by rwa [cast_neg, neg_add_cancel_left]
                        /-
                          🎉 no goals
                        -/


theorem of_add_rat : Irrational (x + q) → Irrational x :=
  add_comm (↑q) x ▸ of_rat_add q


theorem add_rat (h : Irrational x) : Irrational (x + q) :=
  add_comm (↑q) x ▸ h.rat_add q


theorem of_int_add (m : ℤ) (h : Irrational (m + x)) : Irrational x := by
  /-
    x : Real
    m : Int
    h : Irrational (HAdd.hAdd (↑m) x)
    ⊢ Irrational x
  -/
  rw [← cast_intCast] at h
  /-
    x : Real
    m : Int
    h : Irrational (HAdd.hAdd (↑↑m) x)
    ⊢ Irrational x
  -/
  exact h.of_rat_add m
  /-
    🎉 no goals
  -/


theorem of_add_int (m : ℤ) (h : Irrational (x + m)) : Irrational x :=
  of_int_add m <| add_comm x m ▸ h


theorem int_add (h : Irrational x) (m : ℤ) : Irrational (m + x) := by
  /-
    x : Real
    h : Irrational x
    m : Int
    ⊢ Irrational (HAdd.hAdd (↑m) x)
  -/
  rw [← cast_intCast]
  /-
    x : Real
    h : Irrational x
    m : Int
    ⊢ Irrational (HAdd.hAdd (↑↑m) x)
  -/
  exact h.rat_add m
  /-
    🎉 no goals
  -/


theorem add_int (h : Irrational x) (m : ℤ) : Irrational (x + m) :=
  add_comm (↑m) x ▸ h.int_add m


theorem of_nat_add (m : ℕ) (h : Irrational (m + x)) : Irrational x :=
  h.of_int_add m


theorem of_add_nat (m : ℕ) (h : Irrational (x + m)) : Irrational x :=
  h.of_add_int m


theorem nat_add (h : Irrational x) (m : ℕ) : Irrational (m + x) :=
  h.int_add m


theorem add_nat (h : Irrational x) (m : ℕ) : Irrational (x + m) :=
  h.add_int m


                                                                                /-
                                                                                  x : Real
                                                                                  h : Irrational (Neg.neg x)
                                                                                  x✝ : Membership.mem (Set.range Rat.cast) x
                                                                                  q : Rat
                                                                                  hx : Eq (↑q) x
                                                                                  ⊢ Eq (↑(Neg.neg q)) (Neg.neg x)
                                                                                -/
theorem of_neg (h : Irrational (-x)) : Irrational x := fun ⟨q, hx⟩ => h ⟨-q, by rw [cast_neg, hx]⟩
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


protected theorem neg (h : Irrational x) : Irrational (-x) :=
               /-
                 x : Real
                 h : Irrational x
                 ⊢ Irrational (Neg.neg (Neg.neg x))
               -/
  of_neg <| by rwa [neg_neg]
               /-
                 🎉 no goals
               -/


theorem sub_rat (h : Irrational x) : Irrational (x - q) := by
  /-
    q : Rat
    x : Real
    h : Irrational x
    ⊢ Irrational (HSub.hSub x ↑q)
  -/
  simpa only [sub_eq_add_neg, cast_neg] using h.add_rat (-q)
  /-
    🎉 no goals
  -/


theorem rat_sub (h : Irrational x) : Irrational (q - x) := by
  /-
    q : Rat
    x : Real
    h : Irrational x
    ⊢ Irrational (HSub.hSub (↑q) x)
  -/
  simpa only [sub_eq_add_neg] using h.neg.rat_add q
  /-
    🎉 no goals
  -/


theorem of_sub_rat (h : Irrational (x - q)) : Irrational x :=
                        /-
                          q : Rat
                          x : Real
                          h : Irrational (HSub.hSub x ↑q)
                          ⊢ Irrational (HAdd.hAdd x ↑(Neg.neg q))
                        -/
  of_add_rat (-q) <| by simpa only [cast_neg, sub_eq_add_neg] using h
                        /-
                          🎉 no goals
                        -/


theorem of_rat_sub (h : Irrational (q - x)) : Irrational x :=
                           /-
                             q : Rat
                             x : Real
                             h : Irrational (HSub.hSub (↑q) x)
                             ⊢ Irrational (HAdd.hAdd (↑q) (Neg.neg x))
                           -/
  of_neg (of_rat_add q (by simpa only [sub_eq_add_neg] using h))
                           /-
                             🎉 no goals
                           -/


theorem sub_int (h : Irrational x) (m : ℤ) : Irrational (x - m) := by
  /-
    x : Real
    h : Irrational x
    m : Int
    ⊢ Irrational (HSub.hSub x ↑m)
  -/
  simpa only [Rat.cast_intCast] using h.sub_rat m
  /-
    🎉 no goals
  -/


theorem int_sub (h : Irrational x) (m : ℤ) : Irrational (m - x) := by
  /-
    x : Real
    h : Irrational x
    m : Int
    ⊢ Irrational (HSub.hSub (↑m) x)
  -/
  simpa only [Rat.cast_intCast] using h.rat_sub m
  /-
    🎉 no goals
  -/


theorem of_sub_int (m : ℤ) (h : Irrational (x - m)) : Irrational x :=
                     /-
                       x : Real
                       m : Int
                       h : Irrational (HSub.hSub x ↑m)
                       ⊢ Irrational (HSub.hSub x ↑↑m)
                     -/
  of_sub_rat m <| by rwa [Rat.cast_intCast]
                     /-
                       🎉 no goals
                     -/


theorem of_int_sub (m : ℤ) (h : Irrational (m - x)) : Irrational x :=
                     /-
                       x : Real
                       m : Int
                       h : Irrational (HSub.hSub (↑m) x)
                       ⊢ Irrational (HSub.hSub (↑↑m) x)
                     -/
  of_rat_sub m <| by rwa [Rat.cast_intCast]
                     /-
                       🎉 no goals
                     -/


theorem sub_nat (h : Irrational x) (m : ℕ) : Irrational (x - m) :=
  h.sub_int m


theorem nat_sub (h : Irrational x) (m : ℕ) : Irrational (m - x) :=
  h.int_sub m


theorem of_sub_nat (m : ℕ) (h : Irrational (x - m)) : Irrational x :=
  h.of_sub_int m


theorem of_nat_sub (m : ℕ) (h : Irrational (m - x)) : Irrational x :=
  h.of_int_sub m


theorem mul_cases : Irrational (x * y) → Irrational x ∨ Irrational y := by
  /-
    x y : Real
    ⊢ Irrational (HMul.hMul x y) → Or (Irrational x) (Irrational y)
  -/
  delta Irrational
  /-
    x y : Real
    ⊢ Not (Membership.mem (Set.range Rat.cast) (HMul.hMul x y)) → Or (Not (Members …
  -/
  contrapose!
  /-
    x y : Real
    ⊢ And (Membership.mem (Set.range Rat.cast) x) (Membership.mem (Set.range Rat.c …
  -/
  rintro ⟨⟨rx, rfl⟩, ⟨ry, rfl⟩⟩
  /-
    case intro.intro.intro
    rx ry : Rat
    ⊢ Membership.mem (Set.range Rat.cast) (HMul.hMul ↑rx ↑ry)
  -/
  exact ⟨rx * ry, cast_mul rx ry⟩
  /-
    🎉 no goals
  -/


theorem of_mul_rat (h : Irrational (x * q)) : Irrational x :=
  h.mul_cases.resolve_right q.not_irrational


theorem mul_rat (h : Irrational x) {q : ℚ} (hq : q ≠ 0) : Irrational (x * q) :=
                       /-
                         x : Real
                         h : Irrational x
                         q : Rat
                         hq : Ne q 0
                         ⊢ Irrational (HMul.hMul (HMul.hMul x ↑q) ↑(Inv.inv q))
                       -/
  of_mul_rat q⁻¹ <| by rwa [mul_assoc, ← cast_mul, mul_inv_cancel₀ hq, cast_one, mul_one]
                       /-
                         🎉 no goals
                       -/


theorem of_rat_mul : Irrational (q * x) → Irrational x :=
  mul_comm x q ▸ of_mul_rat q


theorem rat_mul (h : Irrational x) {q : ℚ} (hq : q ≠ 0) : Irrational (q * x) :=
  mul_comm x q ▸ h.mul_rat hq


theorem of_mul_int (m : ℤ) (h : Irrational (x * m)) : Irrational x :=
                     /-
                       x : Real
                       m : Int
                       h : Irrational (HMul.hMul x ↑m)
                       ⊢ Irrational (HMul.hMul x ↑↑m)
                     -/
  of_mul_rat m <| by rwa [cast_intCast]
                     /-
                       🎉 no goals
                     -/


theorem of_int_mul (m : ℤ) (h : Irrational (m * x)) : Irrational x :=
                     /-
                       x : Real
                       m : Int
                       h : Irrational (HMul.hMul (↑m) x)
                       ⊢ Irrational (HMul.hMul (↑↑m) x)
                     -/
  of_rat_mul m <| by rwa [cast_intCast]
                     /-
                       🎉 no goals
                     -/


theorem mul_int (h : Irrational x) {m : ℤ} (hm : m ≠ 0) : Irrational (x * m) := by
  /-
    x : Real
    h : Irrational x
    m : Int
    hm : Ne m 0
    ⊢ Irrational (HMul.hMul x ↑m)
  -/
  rw [← cast_intCast]
  /-
    x : Real
    h : Irrational x
    m : Int
    hm : Ne m 0
    ⊢ Irrational (HMul.hMul x ↑↑m)
  -/
  refine h.mul_rat ?_
  /-
    x : Real
    h : Irrational x
    m : Int
    hm : Ne m 0
    ⊢ Ne (↑m) 0
  -/
  rwa [Int.cast_ne_zero]
  /-
    🎉 no goals
  -/


theorem int_mul (h : Irrational x) {m : ℤ} (hm : m ≠ 0) : Irrational (m * x) :=
  mul_comm x m ▸ h.mul_int hm


theorem of_mul_nat (m : ℕ) (h : Irrational (x * m)) : Irrational x :=
  h.of_mul_int m


theorem of_nat_mul (m : ℕ) (h : Irrational (m * x)) : Irrational x :=
  h.of_int_mul m


theorem mul_nat (h : Irrational x) {m : ℕ} (hm : m ≠ 0) : Irrational (x * m) :=
  h.mul_int <| Int.natCast_ne_zero.2 hm


theorem nat_mul (h : Irrational x) {m : ℕ} (hm : m ≠ 0) : Irrational (m * x) :=
  h.int_mul <| Int.natCast_ne_zero.2 hm


theorem of_inv (h : Irrational x⁻¹) : Irrational x := fun ⟨q, hq⟩ => h <| hq ▸ ⟨q⁻¹, q.cast_inv⟩


protected theorem inv (h : Irrational x) : Irrational x⁻¹ :=
               /-
                 x : Real
                 h : Irrational x
                 ⊢ Irrational (Inv.inv (Inv.inv x))
               -/
  of_inv <| by rwa [inv_inv]
               /-
                 🎉 no goals
               -/


theorem div_cases (h : Irrational (x / y)) : Irrational x ∨ Irrational y :=
  h.mul_cases.imp id of_inv


theorem of_rat_div (h : Irrational (q / x)) : Irrational x :=
  (h.of_rat_mul q).of_inv


theorem of_div_rat (h : Irrational (x / q)) : Irrational x :=
  h.div_cases.resolve_right q.not_irrational


theorem rat_div (h : Irrational x) {q : ℚ} (hq : q ≠ 0) : Irrational (q / x) :=
  h.inv.rat_mul hq


theorem div_rat (h : Irrational x) {q : ℚ} (hq : q ≠ 0) : Irrational (x / q) := by
  /-
    x : Real
    h : Irrational x
    q : Rat
    hq : Ne q 0
    ⊢ Irrational (HDiv.hDiv x ↑q)
  -/
  rw [div_eq_mul_inv, ← cast_inv]
  /-
    x : Real
    h : Irrational x
    q : Rat
    hq : Ne q 0
    ⊢ Irrational (HMul.hMul x ↑(Inv.inv q))
  -/
  exact h.mul_rat (inv_ne_zero hq)
  /-
    🎉 no goals
  -/


theorem of_int_div (m : ℤ) (h : Irrational (m / x)) : Irrational x :=
  h.div_cases.resolve_left m.not_irrational


theorem of_div_int (m : ℤ) (h : Irrational (x / m)) : Irrational x :=
  h.div_cases.resolve_right m.not_irrational


theorem int_div (h : Irrational x) {m : ℤ} (hm : m ≠ 0) : Irrational (m / x) :=
  h.inv.int_mul hm


theorem div_int (h : Irrational x) {m : ℤ} (hm : m ≠ 0) : Irrational (x / m) := by
  /-
    x : Real
    h : Irrational x
    m : Int
    hm : Ne m 0
    ⊢ Irrational (HDiv.hDiv x ↑m)
  -/
  rw [← cast_intCast]
  /-
    x : Real
    h : Irrational x
    m : Int
    hm : Ne m 0
    ⊢ Irrational (HDiv.hDiv x ↑↑m)
  -/
  refine h.div_rat ?_
  /-
    x : Real
    h : Irrational x
    m : Int
    hm : Ne m 0
    ⊢ Ne (↑m) 0
  -/
  rwa [Int.cast_ne_zero]
  /-
    🎉 no goals
  -/


theorem of_nat_div (m : ℕ) (h : Irrational (m / x)) : Irrational x :=
  h.of_int_div m


theorem of_div_nat (m : ℕ) (h : Irrational (x / m)) : Irrational x :=
  h.of_div_int m


theorem nat_div (h : Irrational x) {m : ℕ} (hm : m ≠ 0) : Irrational (m / x) :=
  h.inv.nat_mul hm


theorem div_nat (h : Irrational x) {m : ℕ} (hm : m ≠ 0) : Irrational (x / m) :=
                  /-
                    x : Real
                    h : Irrational x
                    m : Nat
                    hm : Ne m 0
                    ⊢ Ne (↑m) 0
                  -/
  h.div_int <| by rwa [Int.natCast_ne_zero]
                  /-
                    🎉 no goals
                  -/


theorem of_one_div (h : Irrational (1 / x)) : Irrational x :=
                     /-
                       x : Real
                       h : Irrational (HDiv.hDiv 1 x)
                       ⊢ Irrational (HDiv.hDiv (↑1) x)
                     -/
  of_rat_div 1 <| by rwa [cast_one]
                     /-
                       🎉 no goals
                     -/


theorem of_mul_self (h : Irrational (x * x)) : Irrational x :=
  h.mul_cases.elim id id


theorem of_pow : ∀ n : ℕ, Irrational (x ^ n) → Irrational x
  | 0 => fun h => by
    /-
      x : Real
      h : Irrational (HPow.hPow x 0)
      ⊢ Irrational x
    -/
    rw [pow_zero] at h
    /-
      x : Real
      h : Irrational 1
      ⊢ Irrational x
    -/
    exact (h ⟨1, cast_one⟩).elim
    /-
      🎉 no goals
    -/
  | n + 1 => fun h => by
    /-
      x : Real
      n : Nat
      h : Irrational (HPow.hPow x (HAdd.hAdd n 1))
      ⊢ Irrational x
    -/
    rw [pow_succ] at h
    /-
      x : Real
      n : Nat
      h : Irrational (HMul.hMul (HPow.hPow x n) x)
      ⊢ Irrational x
    -/
    exact h.mul_cases.elim (of_pow n) id
    /-
      🎉 no goals
    -/


open Int in
theorem of_zpow : ∀ m : ℤ, Irrational (x ^ m) → Irrational x
  | (n : ℕ) => fun h => by
    /-
      x : Real
      n : Nat
      h : Irrational (HPow.hPow x ↑n)
      ⊢ Irrational x
    -/
    rw [zpow_natCast] at h
    /-
      x : Real
      n : Nat
      h : Irrational (HPow.hPow x n)
      ⊢ Irrational x
    -/
    exact h.of_pow _
    /-
      🎉 no goals
    -/
  | -[n+1] => fun h => by
    /-
      x : Real
      n : Nat
      h : Irrational (HPow.hPow x (Int.negSucc n))
      ⊢ Irrational x
    -/
    rw [zpow_negSucc] at h
    /-
      x : Real
      n : Nat
      h : Irrational (Inv.inv (HPow.hPow x (HAdd.hAdd n 1)))
      ⊢ Irrational x
    -/
    exact h.of_inv.of_pow _
    /-
      🎉 no goals
    -/


theorem one_lt_natDegree_of_irrational_root (hx : Irrational x) (p_nonzero : p ≠ 0)
    (x_is_root : aeval x p = 0) : 1 < p.natDegree := by
  /-
    x : Real
    p : Polynomial Int
    hx : Irrational x
    p_nonzero : Ne p 0
    x_is_root : Eq ((Polynomial.aeval x) p) 0
    ⊢ LT.lt 1 p.natDegree
  -/
  by_contra rid
  /-
    x : Real
    p : Polynomial Int
    hx : Irrational x
    p_nonzero : Ne p 0
    x_is_root : Eq ((Polynomial.aeval x) p) 0
    rid : Not (LT.lt 1 p.natDegree)
    ⊢ False
  -/
  rcases exists_eq_X_add_C_of_natDegree_le_one (not_lt.1 rid) with ⟨a, b, rfl⟩
  /-
    case intro.intro
    x : Real
    hx : Irrational x
    a b : Int
    p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomia …
    x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C a) Po …
    rid : Not (LT.lt 1 (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polyn …
    ⊢ False
  -/
  clear rid
  /-
    case intro.intro
    x : Real
    hx : Irrational x
    a b : Int
    p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomia …
    x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C a) Po …
    ⊢ False
  -/
  have : (a : ℝ) * x = -b := by simpa [eq_neg_iff_add_eq_zero] using x_is_root
  /-
    case intro.intro
    x : Real
    hx : Irrational x
    a b : Int
    p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomia …
    x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C a) Po …
    this : Eq (HMul.hMul (↑a) x) (Neg.neg ↑b)
    ⊢ False
  -/
  rcases em (a = 0) with (rfl | ha)
    /-
      case intro.intro.inl
      x : Real
      hx : Irrational x
      b : Int
      p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C 0) Polynomial.X) (Polynomia …
      x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C 0) Po …
      this : Eq (HMul.hMul (↑0) x) (Neg.neg ↑b)
      ⊢ False
    -/
  · obtain rfl : b = 0 := by simpa
    /-
      case intro.intro.inl
      x : Real
      hx : Irrational x
      p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C 0) Polynomial.X) (Polynomia …
      x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C 0) Po …
      this : Eq (HMul.hMul (↑0) x) (Neg.neg ↑0)
      ⊢ False
    -/
    simp at p_nonzero
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      x : Real
      hx : Irrational x
      a b : Int
      p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomia …
      x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C a) Po …
      this : Eq (HMul.hMul (↑a) x) (Neg.neg ↑b)
      ha : Not (Eq a 0)
      ⊢ False
    -/
  · rw [mul_comm, ← eq_div_iff_mul_eq, eq_comm] at this
      /-
        case intro.intro.inr
        x : Real
        hx : Irrational x
        a b : Int
        p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomia …
        x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C a) Po …
        this : Eq (HDiv.hDiv (Neg.neg ↑b) ↑a) x
        ha : Not (Eq a 0)
        ⊢ False
      -/
    · refine hx ⟨-b / a, ?_⟩
      /-
        case intro.intro.inr
        x : Real
        hx : Irrational x
        a b : Int
        p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomia …
        x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C a) Po …
        this : Eq (HDiv.hDiv (Neg.neg ↑b) ↑a) x
        ha : Not (Eq a 0)
        ⊢ Eq (↑(HDiv.hDiv (Neg.neg ↑b) ↑a)) x
      -/
      assumption_mod_cast
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.inr
        x : Real
        hx : Irrational x
        a b : Int
        p_nonzero : Ne (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomia …
        x_is_root : Eq ((Polynomial.aeval x) (HAdd.hAdd (HMul.hMul (Polynomial.C a) Po …
        this : Eq (HMul.hMul x ↑a) (Neg.neg ↑b)
        ha : Not (Eq a 0)
        ⊢ Ne (↑a) 0
      -/
    · assumption_mod_cast
      /-
        🎉 no goals
      -/


@[simp]
theorem irrational_rat_add_iff : Irrational (q + x) ↔ Irrational x :=
  ⟨of_rat_add q, rat_add q⟩


@[simp]
theorem irrational_int_add_iff : Irrational (m + x) ↔ Irrational x :=
  ⟨of_int_add m, fun h => h.int_add m⟩


@[simp]
theorem irrational_nat_add_iff : Irrational (n + x) ↔ Irrational x :=
  ⟨of_nat_add n, fun h => h.nat_add n⟩


@[simp]
theorem irrational_add_rat_iff : Irrational (x + q) ↔ Irrational x :=
  ⟨of_add_rat q, add_rat q⟩


@[simp]
theorem irrational_add_int_iff : Irrational (x + m) ↔ Irrational x :=
  ⟨of_add_int m, fun h => h.add_int m⟩


@[simp]
theorem irrational_add_nat_iff : Irrational (x + n) ↔ Irrational x :=
  ⟨of_add_nat n, fun h => h.add_nat n⟩


@[simp]
theorem irrational_rat_sub_iff : Irrational (q - x) ↔ Irrational x :=
  ⟨of_rat_sub q, rat_sub q⟩


@[simp]
theorem irrational_int_sub_iff : Irrational (m - x) ↔ Irrational x :=
  ⟨of_int_sub m, fun h => h.int_sub m⟩


@[simp]
theorem irrational_nat_sub_iff : Irrational (n - x) ↔ Irrational x :=
  ⟨of_nat_sub n, fun h => h.nat_sub n⟩


@[simp]
theorem irrational_sub_rat_iff : Irrational (x - q) ↔ Irrational x :=
  ⟨of_sub_rat q, sub_rat q⟩


@[simp]
theorem irrational_sub_int_iff : Irrational (x - m) ↔ Irrational x :=
  ⟨of_sub_int m, fun h => h.sub_int m⟩


@[simp]
theorem irrational_sub_nat_iff : Irrational (x - n) ↔ Irrational x :=
  ⟨of_sub_nat n, fun h => h.sub_nat n⟩


@[simp]
theorem irrational_neg_iff : Irrational (-x) ↔ Irrational x :=
  ⟨of_neg, Irrational.neg⟩


@[simp]
theorem irrational_inv_iff : Irrational x⁻¹ ↔ Irrational x :=
  ⟨of_inv, Irrational.inv⟩


@[simp]
theorem irrational_rat_mul_iff : Irrational (q * x) ↔ q ≠ 0 ∧ Irrational x :=
  ⟨fun h => ⟨Rat.cast_ne_zero.1 <| left_ne_zero_of_mul h.ne_zero, h.of_rat_mul q⟩, fun h =>
    h.2.rat_mul h.1⟩


@[simp]
theorem irrational_mul_rat_iff : Irrational (x * q) ↔ q ≠ 0 ∧ Irrational x := by
  /-
    q : Rat
    x : Real
    ⊢ Iff (Irrational (HMul.hMul x ↑q)) (And (Ne q 0) (Irrational x))
  -/
  rw [mul_comm, irrational_rat_mul_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_int_mul_iff : Irrational (m * x) ↔ m ≠ 0 ∧ Irrational x := by
  /-
    m : Int
    x : Real
    ⊢ Iff (Irrational (HMul.hMul (↑m) x)) (And (Ne m 0) (Irrational x))
  -/
  rw [← cast_intCast, irrational_rat_mul_iff, Int.cast_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_mul_int_iff : Irrational (x * m) ↔ m ≠ 0 ∧ Irrational x := by
  /-
    m : Int
    x : Real
    ⊢ Iff (Irrational (HMul.hMul x ↑m)) (And (Ne m 0) (Irrational x))
  -/
  rw [← cast_intCast, irrational_mul_rat_iff, Int.cast_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_nat_mul_iff : Irrational (n * x) ↔ n ≠ 0 ∧ Irrational x := by
  /-
    n : Nat
    x : Real
    ⊢ Iff (Irrational (HMul.hMul (↑n) x)) (And (Ne n 0) (Irrational x))
  -/
  rw [← cast_natCast, irrational_rat_mul_iff, Nat.cast_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_mul_nat_iff : Irrational (x * n) ↔ n ≠ 0 ∧ Irrational x := by
  /-
    n : Nat
    x : Real
    ⊢ Iff (Irrational (HMul.hMul x ↑n)) (And (Ne n 0) (Irrational x))
  -/
  rw [← cast_natCast, irrational_mul_rat_iff, Nat.cast_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_rat_div_iff : Irrational (q / x) ↔ q ≠ 0 ∧ Irrational x := by
  /-
    q : Rat
    x : Real
    ⊢ Iff (Irrational (HDiv.hDiv (↑q) x)) (And (Ne q 0) (Irrational x))
  -/
  simp [div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_div_rat_iff : Irrational (x / q) ↔ q ≠ 0 ∧ Irrational x := by
  /-
    q : Rat
    x : Real
    ⊢ Iff (Irrational (HDiv.hDiv x ↑q)) (And (Ne q 0) (Irrational x))
  -/
  rw [div_eq_mul_inv, ← cast_inv, irrational_mul_rat_iff, Ne, inv_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_int_div_iff : Irrational (m / x) ↔ m ≠ 0 ∧ Irrational x := by
  /-
    m : Int
    x : Real
    ⊢ Iff (Irrational (HDiv.hDiv (↑m) x)) (And (Ne m 0) (Irrational x))
  -/
  simp [div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_div_int_iff : Irrational (x / m) ↔ m ≠ 0 ∧ Irrational x := by
  /-
    m : Int
    x : Real
    ⊢ Iff (Irrational (HDiv.hDiv x ↑m)) (And (Ne m 0) (Irrational x))
  -/
  rw [← cast_intCast, irrational_div_rat_iff, Int.cast_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_nat_div_iff : Irrational (n / x) ↔ n ≠ 0 ∧ Irrational x := by
  /-
    n : Nat
    x : Real
    ⊢ Iff (Irrational (HDiv.hDiv (↑n) x)) (And (Ne n 0) (Irrational x))
  -/
  simp [div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem irrational_div_nat_iff : Irrational (x / n) ↔ n ≠ 0 ∧ Irrational x := by
  /-
    n : Nat
    x : Real
    ⊢ Iff (Irrational (HDiv.hDiv x ↑n)) (And (Ne n 0) (Irrational x))
  -/
  rw [← cast_natCast, irrational_div_rat_iff, Nat.cast_ne_zero]
  /-
    🎉 no goals
  -/


/-- There is an irrational number `r` between any two reals `x < r < y`. -/
theorem exists_irrational_btwn {x y : ℝ} (h : x < y) : ∃ r, Irrational r ∧ x < r ∧ r < y :=
  let ⟨q, ⟨hq1, hq2⟩⟩ := exists_rat_btwn ((sub_lt_sub_iff_right (√2)).mpr h)
  ⟨q + √2, irrational_sqrt_two.rat_add _, sub_lt_iff_lt_add.mp hq1, lt_sub_iff_add_lt.mp hq2⟩


