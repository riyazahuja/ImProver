/-- Fermat numbers: the `n`-th Fermat number is defined as `2^(2^n) + 1`. -/
def fermatNumber (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1


@[simp] theorem fermatNumber_zero : fermatNumber 0 = 3 := rfl

@[simp] theorem fermatNumber_one : fermatNumber 1 = 5 := rfl

@[simp] theorem fermatNumber_two : fermatNumber 2 = 17 := rfl


theorem fermatNumber_strictMono : StrictMono fermatNumber := by
  /-
    ⊢ StrictMono Nat.fermatNumber
  -/
  intro m n
  simp only [fermatNumber, add_lt_add_iff_right, Nat.pow_lt_pow_iff_right (one_lt_two : 1 < 2),
    imp_self]


@[deprecated (since := "2024-11-25")] alias strictMono_fermatNumber := fermatNumber_strictMono


lemma fermatNumber_mono : Monotone fermatNumber := fermatNumber_strictMono.monotone

lemma fermatNumber_injective : Injective fermatNumber := fermatNumber_strictMono.injective


lemma three_le_fermatNumber (n : ℕ) : 3 ≤ fermatNumber n := fermatNumber_mono n.zero_le

lemma two_lt_fermatNumber (n : ℕ) : 2 < fermatNumber n := three_le_fermatNumber _


                                                             /-
                                                               n : Nat
                                                               ⊢ Ne n.fermatNumber 1
                                                             -/
lemma fermatNumber_ne_one (n : ℕ) : fermatNumber n ≠ 1 := by have := three_le_fermatNumber n; omega
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


theorem odd_fermatNumber (n : ℕ) : Odd (fermatNumber n) :=
  (even_pow.mpr ⟨even_two, (pow_pos two_pos n).ne'⟩).add_one


theorem prod_fermatNumber (n : ℕ) : ∏ k ∈ range n, fermatNumber k = fermatNumber n - 2 := by
  /-
    n : Nat
    ⊢ Eq ((Finset.range n).prod fun k => k.fermatNumber) (HSub.hSub n.fermatNumber …
  -/
  induction' n with n hn
    /-
      case zero
      ⊢ Eq ((Finset.range 0).prod fun k => k.fermatNumber) (HSub.hSub (Nat.fermatNum …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  rw [prod_range_succ, hn, fermatNumber, fermatNumber, mul_comm,
    (show 2 ^ 2 ^ n + 1 - 2 = 2 ^ 2 ^ n - 1 by omega), ← sq_sub_sq]
  /-
    case succ
    n : Nat
    hn : Eq ((Finset.range n).prod fun k => k.fermatNumber) (HSub.hSub n.fermatNum …
    ⊢ Eq (HSub.hSub (HPow.hPow (HPow.hPow 2 (HPow.hPow 2 n)) 2) (HPow.hPow 1 2)) ( …
  -/
  ring_nf
  /-
    case succ
    n : Nat
    hn : Eq ((Finset.range n).prod fun k => k.fermatNumber) (HSub.hSub n.fermatNum …
    ⊢ Eq (HSub.hSub (HPow.hPow 2 (HMul.hMul (HPow.hPow 2 n) 2)) 1) (HSub.hSub (HAd …
  -/
  omega
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-25")] alias fermatNumber_product := prod_fermatNumber


theorem fermatNumber_eq_prod_add_two (n : ℕ) :
    fermatNumber n = ∏ k ∈ range n, fermatNumber k + 2 := by
  /-
    n : Nat
    ⊢ Eq n.fermatNumber (HAdd.hAdd ((Finset.range n).prod fun k => k.fermatNumber) …
  -/
  rw [prod_fermatNumber, Nat.sub_add_cancel]
  /-
    n : Nat
    ⊢ LE.le 2 n.fermatNumber
  -/
  exact le_of_lt <| two_lt_fermatNumber _
  /-
    🎉 no goals
  -/


theorem fermatNumber_succ (n : ℕ) : fermatNumber (n + 1) = (fermatNumber n - 1) ^ 2 + 1 := by
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd n 1).fermatNumber (HAdd.hAdd (HPow.hPow (HSub.hSub n.fermatNum …
  -/
  rw [fermatNumber, pow_succ, mul_comm, pow_mul']
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (HPow.hPow (HPow.hPow 2 (HPow.hPow 2 n)) 2) 1) (HAdd.hAdd (HPo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem two_mul_fermatNumber_sub_one_sq_le_fermatNumber_sq (n : ℕ) :
    2 * (fermatNumber n - 1) ^ 2 ≤ (fermatNumber (n + 1)) ^ 2 := by
  /-
    n : Nat
    ⊢ LE.le (HMul.hMul 2 (HPow.hPow (HSub.hSub n.fermatNumber 1) 2)) (HPow.hPow (H …
  -/
  simp only [fermatNumber, add_tsub_cancel_right]
  /-
    n : Nat
    ⊢ LE.le (HMul.hMul 2 (HPow.hPow (HPow.hPow 2 (HPow.hPow 2 n)) 2)) (HPow.hPow ( …
  -/
  have : 0 ≤ 1 + 2 ^ (2 ^ n * 4) := le_add_left _ _
  /-
    n : Nat
    this : LE.le 0 (HAdd.hAdd 1 (HPow.hPow 2 (HMul.hMul (HPow.hPow 2 n) 4)))
    ⊢ LE.le (HMul.hMul 2 (HPow.hPow (HPow.hPow 2 (HPow.hPow 2 n)) 2)) (HPow.hPow ( …
  -/
  ring_nf
  /-
    n : Nat
    this : LE.le 0 (HAdd.hAdd 1 (HPow.hPow 2 (HMul.hMul (HPow.hPow 2 n) 4)))
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 (HMul.hMul (HPow.hPow 2 n) 2)) 2) (HAdd.hAdd ( …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem fermatNumber_eq_fermatNumber_sq_sub_two_mul_fermatNumber_sub_one_sq (n : ℕ) :
    fermatNumber (n + 2) = (fermatNumber (n + 1)) ^ 2 - 2 * (fermatNumber n - 1) ^ 2 := by
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd n 2).fermatNumber (HSub.hSub (HPow.hPow (HAdd.hAdd n 1).fermat …
  -/
  simp only [fermatNumber, add_sub_self_right]
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (HPow.hPow 2 (HPow.hPow 2 (HAdd.hAdd n 2))) 1) (HSub.hSub (HPo …
  -/
  rw [← add_sub_self_right (2 ^ 2 ^ (n + 2) + 1) <| 2 * 2 ^ 2 ^ (n + 1)]
  /-
    n : Nat
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HPow.hPow 2 (HPow.hPow 2 (HAdd.hAdd n 2 …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem Int.fermatNumber_eq_fermatNumber_sq_sub_two_mul_fermatNumber_sub_one_sq (n : ℕ) :
    (fermatNumber (n + 2) : ℤ)  = (fermatNumber (n + 1)) ^ 2 - 2 * (fermatNumber n - 1) ^ 2 := by
  rw [Nat.fermatNumber_eq_fermatNumber_sq_sub_two_mul_fermatNumber_sub_one_sq,
    Nat.cast_sub <| two_mul_fermatNumber_sub_one_sq_le_fermatNumber_sq n]
  /-
    n : Nat
    ⊢ Eq (HSub.hSub ↑(HPow.hPow (HAdd.hAdd n 1).fermatNumber 2) ↑(HMul.hMul 2 (HPo …
  -/
  simp only [fermatNumber, push_cast, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


/--
**Goldbach's theorem** : no two distinct Fermat numbers share a common factor greater than one.

From a letter to Euler, see page 37 in [juskevic2022].
-/
theorem coprime_fermatNumber_fermatNumber {m n : ℕ} (hmn : m ≠ n) :
    Coprime (fermatNumber m) (fermatNumber n) := by
  /-
    m n : Nat
    hmn : Ne m n
    ⊢ m.fermatNumber.Coprime n.fermatNumber
  -/
  wlog hmn' : m < n
    /-
      case inr
      m n : Nat
      hmn : Ne m n
      this : ∀ {m n : Nat}, Ne m n → LT.lt m n → m.fermatNumber.Coprime n.fermatNumber
      hmn' : Not (LT.lt m n)
      ⊢ m.fermatNumber.Coprime n.fermatNumber
    -/
  · simpa only [coprime_comm] using this hmn.symm (by omega)
    /-
      🎉 no goals
    -/
  /-
    m n : Nat
    hmn : Ne m n
    hmn' : LT.lt m n
    ⊢ m.fermatNumber.Coprime n.fermatNumber
  -/
  let d := (fermatNumber m).gcd (fermatNumber n)
  /-
    m n : Nat
    hmn : Ne m n
    hmn' : LT.lt m n
    d : Nat := m.fermatNumber.gcd n.fermatNumber
    ⊢ m.fermatNumber.Coprime n.fermatNumber
  -/
  have h_n : d ∣ fermatNumber n := gcd_dvd_right ..
  have h_m : d ∣ 2 := (Nat.dvd_add_right <| (gcd_dvd_left _ _).trans <| dvd_prod_of_mem _
    <| mem_range.mpr hmn').mp <| fermatNumber_eq_prod_add_two _ ▸ h_n
  /-
    m n : Nat
    hmn : Ne m n
    hmn' : LT.lt m n
    d : Nat := m.fermatNumber.gcd n.fermatNumber
    h_n : Dvd.dvd d n.fermatNumber
    h_m : Dvd.dvd d 2
    ⊢ m.fermatNumber.Coprime n.fermatNumber
  -/
  refine ((dvd_prime prime_two).mp h_m).resolve_right fun h_two ↦ ?_
  /-
    m n : Nat
    hmn : Ne m n
    hmn' : LT.lt m n
    d : Nat := m.fermatNumber.gcd n.fermatNumber
    h_n : Dvd.dvd d n.fermatNumber
    h_m : Dvd.dvd d 2
    h_two : Eq d 2
    ⊢ False
  -/
  exact (odd_fermatNumber _).not_two_dvd_nat (h_two ▸ h_n)
  /-
    🎉 no goals
  -/


lemma pairwise_coprime_fermatNumber :
    Pairwise fun m n ↦ Coprime (fermatNumber m) (fermatNumber n) :=
  fun _m _n ↦ coprime_fermatNumber_fermatNumber


/-- Prime `a ^ n + 1` implies `n` is a power of two (**Fermat primes**). -/
theorem pow_of_pow_add_prime {a n : ℕ} (ha : 1 < a) (hn : n ≠ 0) (hP : (a ^ n + 1).Prime) :
    ∃ m : ℕ, n = 2 ^ m := by
  /-
    a n : Nat
    ha : LT.lt 1 a
    hn : Ne n 0
    hP : Nat.Prime (HAdd.hAdd (HPow.hPow a n) 1)
    ⊢ Exists fun m => Eq n (HPow.hPow 2 m)
  -/
  obtain ⟨k, m, hm, rfl⟩ := exists_eq_two_pow_mul_odd hn
  /-
    case intro.intro.intro
    a : Nat
    ha : LT.lt 1 a
    k m : Nat
    hm : Odd m
    hn : Ne (HMul.hMul (HPow.hPow 2 k) m) 0
    hP : Nat.Prime (HAdd.hAdd (HPow.hPow a (HMul.hMul (HPow.hPow 2 k) m)) 1)
    ⊢ Exists fun m_1 => Eq (HMul.hMul (HPow.hPow 2 k) m) (HPow.hPow 2 m_1)
  -/
  rw [pow_mul] at hP
  /-
    case intro.intro.intro
    a : Nat
    ha : LT.lt 1 a
    k m : Nat
    hm : Odd m
    hn : Ne (HMul.hMul (HPow.hPow 2 k) m) 0
    hP : Nat.Prime (HAdd.hAdd (HPow.hPow (HPow.hPow a (HPow.hPow 2 k)) m) 1)
    ⊢ Exists fun m_1 => Eq (HMul.hMul (HPow.hPow 2 k) m) (HPow.hPow 2 m_1)
  -/
  use k
  /-
    case h
    a : Nat
    ha : LT.lt 1 a
    k m : Nat
    hm : Odd m
    hn : Ne (HMul.hMul (HPow.hPow 2 k) m) 0
    hP : Nat.Prime (HAdd.hAdd (HPow.hPow (HPow.hPow a (HPow.hPow 2 k)) m) 1)
    ⊢ Eq (HMul.hMul (HPow.hPow 2 k) m) (HPow.hPow 2 k)
  -/
  replace ha : 1 < a ^ 2 ^ k := one_lt_pow (pow_ne_zero k two_ne_zero) ha
  /-
    case h
    a k m : Nat
    hm : Odd m
    hn : Ne (HMul.hMul (HPow.hPow 2 k) m) 0
    hP : Nat.Prime (HAdd.hAdd (HPow.hPow (HPow.hPow a (HPow.hPow 2 k)) m) 1)
    ha : LT.lt 1 (HPow.hPow a (HPow.hPow 2 k))
    ⊢ Eq (HMul.hMul (HPow.hPow 2 k) m) (HPow.hPow 2 k)
  -/
  let h := hm.nat_add_dvd_pow_add_pow (a ^ 2 ^ k) 1
  /-
    case h
    a k m : Nat
    hm : Odd m
    hn : Ne (HMul.hMul (HPow.hPow 2 k) m) 0
    hP : Nat.Prime (HAdd.hAdd (HPow.hPow (HPow.hPow a (HPow.hPow 2 k)) m) 1)
    ha : LT.lt 1 (HPow.hPow a (HPow.hPow 2 k))
    h : Dvd.dvd (HAdd.hAdd (HPow.hPow a (HPow.hPow 2 k)) 1) (HAdd.hAdd (HPow.hPow  …
    ⊢ Eq (HMul.hMul (HPow.hPow 2 k) m) (HPow.hPow 2 k)
  -/
  rw [one_pow, hP.dvd_iff_eq (Nat.lt_add_right 1 ha).ne', add_left_inj, pow_eq_self_iff ha] at h
  /-
    case h
    a k m : Nat
    hm : Odd m
    hn : Ne (HMul.hMul (HPow.hPow 2 k) m) 0
    hP : Nat.Prime (HAdd.hAdd (HPow.hPow (HPow.hPow a (HPow.hPow 2 k)) m) 1)
    ha : LT.lt 1 (HPow.hPow a (HPow.hPow 2 k))
    h : Eq m 1
    ⊢ Eq (HMul.hMul (HPow.hPow 2 k) m) (HPow.hPow 2 k)
  -/
  rw [h, mul_one]
  /-
    🎉 no goals
  -/


/-- `Fₙ = 2^(2^n)+1` is prime if `3^(2^(2^n-1)) = -1 mod Fₙ` (**Pépin's test**). -/
lemma pepin_primality (n : ℕ) (h : 3 ^ (2 ^ (2 ^ n - 1)) = (-1 : ZMod (fermatNumber n))) :
    (fermatNumber n).Prime := by
  /-
    n : Nat
    h : Eq (HPow.hPow 3 (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1))) (-1)
    ⊢ Nat.Prime n.fermatNumber
  -/
  have := Fact.mk (two_lt_fermatNumber n)
  /-
    n : Nat
    h : Eq (HPow.hPow 3 (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1))) (-1)
    this : Fact (LT.lt 2 n.fermatNumber)
    ⊢ Nat.Prime n.fermatNumber
  -/
  have key : 2 ^ n = 2 ^ n - 1 + 1 := (Nat.sub_add_cancel Nat.one_le_two_pow).symm
  /-
    n : Nat
    h : Eq (HPow.hPow 3 (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1))) (-1)
    this : Fact (LT.lt 2 n.fermatNumber)
    key : Eq (HPow.hPow 2 n) (HAdd.hAdd (HSub.hSub (HPow.hPow 2 n) 1) 1)
    ⊢ Nat.Prime n.fermatNumber
  -/
  apply lucas_primality (p := 2 ^ (2 ^ n) + 1) (a := 3)
    /-
      case ha
      n : Nat
      h : Eq (HPow.hPow 3 (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1))) (-1)
      this : Fact (LT.lt 2 n.fermatNumber)
      key : Eq (HPow.hPow 2 n) (HAdd.hAdd (HSub.hSub (HPow.hPow 2 n) 1) 1)
      ⊢ Eq (HPow.hPow 3 (HSub.hSub (HAdd.hAdd (HPow.hPow 2 (HPow.hPow 2 n)) 1) 1)) 1
    -/
  · rw [Nat.add_sub_cancel, key, pow_succ, pow_mul, ← pow_succ, ← key, h, neg_one_sq]
    /-
      🎉 no goals
    -/
    /-
      case hd
      n : Nat
      h : Eq (HPow.hPow 3 (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1))) (-1)
      this : Fact (LT.lt 2 n.fermatNumber)
      key : Eq (HPow.hPow 2 n) (HAdd.hAdd (HSub.hSub (HPow.hPow 2 n) 1) 1)
      ⊢ ∀ (q : Nat), Nat.Prime q → Dvd.dvd q (HSub.hSub (HAdd.hAdd (HPow.hPow 2 (HPo …
    -/
  · intro p hp1 hp2
    rw [Nat.add_sub_cancel, (Nat.prime_dvd_prime_iff_eq hp1 prime_two).mp (hp1.dvd_of_dvd_pow hp2),
        key, pow_succ, Nat.mul_div_cancel _ two_pos, ← pow_succ, ← key, h]
    /-
      case hd
      n : Nat
      h : Eq (HPow.hPow 3 (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1))) (-1)
      this : Fact (LT.lt 2 n.fermatNumber)
      key : Eq (HPow.hPow 2 n) (HAdd.hAdd (HSub.hSub (HPow.hPow 2 n) 1) 1)
      p : Nat
      hp1 : Nat.Prime p
      hp2 : Dvd.dvd p (HSub.hSub (HAdd.hAdd (HPow.hPow 2 (HPow.hPow 2 n)) 1) 1)
      ⊢ Ne (-1) 1
    -/
    exact neg_one_ne_one
    /-
      🎉 no goals
    -/


/-- `Fₙ = 2^(2^n)+1` is prime if `3^((Fₙ - 1)/2) = -1 mod Fₙ` (**Pépin's test**). -/
lemma pepin_primality' (n : ℕ) (h : 3 ^ ((fermatNumber n - 1) / 2) = (-1 : ZMod (fermatNumber n))) :
    (fermatNumber n).Prime := by
  /-
    n : Nat
    h : Eq (HPow.hPow 3 (HDiv.hDiv (HSub.hSub n.fermatNumber 1) 2)) (-1)
    ⊢ Nat.Prime n.fermatNumber
  -/
  apply pepin_primality
  /-
    case h
    n : Nat
    h : Eq (HPow.hPow 3 (HDiv.hDiv (HSub.hSub n.fermatNumber 1) 2)) (-1)
    ⊢ Eq (HPow.hPow 3 (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1))) (-1)
  -/
  rw [← h]
  /-
    case h
    n : Nat
    h : Eq (HPow.hPow 3 (HDiv.hDiv (HSub.hSub n.fermatNumber 1) 2)) (-1)
    ⊢ Eq (HPow.hPow 3 (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1))) (HPow.hPow 3 (H …
  -/
  congr
  /-
    case h.e_a
    n : Nat
    h : Eq (HPow.hPow 3 (HDiv.hDiv (HSub.hSub n.fermatNumber 1) 2)) (-1)
    ⊢ Eq (HPow.hPow 2 (HSub.hSub (HPow.hPow 2 n) 1)) (HDiv.hDiv (HSub.hSub n.ferma …
  -/
  rw [fermatNumber, add_tsub_cancel_right, Nat.pow_div Nat.one_le_two_pow Nat.zero_lt_two]
  /-
    🎉 no goals
  -/



/-- Prime factors of `a ^ (2 ^ n) + 1` are of form `k * 2 ^ (n + 1) + 1`. -/
lemma pow_pow_add_primeFactors_one_lt {a n p : ℕ} (hp : p.Prime) (hp2 : p ≠ 2)
    (hpdvd : p ∣ a ^ (2 ^ n) + 1) :
    ∃ k, p = k * 2 ^ (n + 1) + 1 := by
  /-
    a n p : Nat
    hp : Nat.Prime p
    hp2 : Ne p 2
    hpdvd : Dvd.dvd p (HAdd.hAdd (HPow.hPow a (HPow.hPow 2 n)) 1)
    ⊢ Exists fun k => Eq p (HAdd.hAdd (HMul.hMul k (HPow.hPow 2 (HAdd.hAdd n 1))) 1)
  -/
  have : Fact (2 < p) := Fact.mk (lt_of_le_of_ne hp.two_le hp2.symm)
  /-
    a n p : Nat
    hp : Nat.Prime p
    hp2 : Ne p 2
    hpdvd : Dvd.dvd p (HAdd.hAdd (HPow.hPow a (HPow.hPow 2 n)) 1)
    this : Fact (LT.lt 2 p)
    ⊢ Exists fun k => Eq p (HAdd.hAdd (HMul.hMul k (HPow.hPow 2 (HAdd.hAdd n 1))) 1)
  -/
  have : Fact p.Prime := Fact.mk hp
  have ha1 : (a : ZMod p) ^ (2 ^ n) = -1 := by
    rw [eq_neg_iff_add_eq_zero]
    exact_mod_cast (natCast_zmod_eq_zero_iff_dvd (a ^ (2 ^ n) + 1) p).mpr hpdvd
  have ha0 : (a : ZMod p) ≠ 0 := by
    intro h
    rw [h, zero_pow (pow_ne_zero n two_ne_zero), zero_eq_neg] at ha1
    exact one_ne_zero ha1
  have ha : orderOf (a : ZMod p) = 2 ^ (n + 1) := by
    apply orderOf_eq_prime_pow
    · rw [ha1]
      exact neg_one_ne_one
    · rw [pow_succ, pow_mul, ha1, neg_one_sq]
  /-
    a n p : Nat
    hp : Nat.Prime p
    hp2 : Ne p 2
    hpdvd : Dvd.dvd p (HAdd.hAdd (HPow.hPow a (HPow.hPow 2 n)) 1)
    this✝ : Fact (LT.lt 2 p)
    this : Fact (Nat.Prime p)
    ha1 : Eq (HPow.hPow (↑a) (HPow.hPow 2 n)) (-1)
    ha0 : Ne (↑a) 0
    ha : Eq (orderOf ↑a) (HPow.hPow 2 (HAdd.hAdd n 1))
    ⊢ Exists fun k => Eq p (HAdd.hAdd (HMul.hMul k (HPow.hPow 2 (HAdd.hAdd n 1))) 1)
  -/
  simpa [ha, dvd_def, Nat.sub_eq_iff_eq_add hp.one_le, mul_comm] using orderOf_dvd_card_sub_one ha0
  /-
    🎉 no goals
  -/

-- Prime factors of `Fₙ = 2 ^ (2 ^ n) + 1`, `1 < n`, are of form `k * 2 ^ (n + 2) + 1`. -/

lemma fermat_primeFactors_one_lt (n p : ℕ) (hn : 1 < n) (hp : p.Prime)
    (hpdvd : p ∣ fermatNumber n) :
    ∃ k, p = k * 2 ^ (n + 2) + 1 := by
  /-
    n p : Nat
    hn : LT.lt 1 n
    hp : Nat.Prime p
    hpdvd : Dvd.dvd p n.fermatNumber
    ⊢ Exists fun k => Eq p (HAdd.hAdd (HMul.hMul k (HPow.hPow 2 (HAdd.hAdd n 2))) 1)
  -/
  have : Fact p.Prime := Fact.mk hp
  have hp2 : p ≠ 2 := by
    exact ((even_pow.mpr ⟨even_two, pow_ne_zero n two_ne_zero⟩).add_one).ne_two_of_dvd_nat hpdvd
  have hp8 : p % 8 = 1 := by
    obtain ⟨k, rfl⟩ := pow_pow_add_primeFactors_one_lt hp hp2 hpdvd
    obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le' hn
    rw [add_assoc, pow_add, ← mul_assoc, ← mod_add_mod, mul_mod]
    norm_num
  /-
    n p : Nat
    hn : LT.lt 1 n
    hp : Nat.Prime p
    hpdvd : Dvd.dvd p n.fermatNumber
    this : Fact (Nat.Prime p)
    hp2 : Ne p 2
    hp8 : Eq (HMod.hMod p 8) 1
    ⊢ Exists fun k => Eq p (HAdd.hAdd (HMul.hMul k (HPow.hPow 2 (HAdd.hAdd n 2))) 1)
  -/
  obtain ⟨a, ha⟩ := (exists_sq_eq_two_iff hp2).mpr (Or.inl hp8)
  suffices h : p ∣ a.val ^ (2 ^ (n + 1)) + 1 by
    exact pow_pow_add_primeFactors_one_lt hp hp2 h
  /-
    case intro
    n p : Nat
    hn : LT.lt 1 n
    hp : Nat.Prime p
    hpdvd : Dvd.dvd p n.fermatNumber
    this : Fact (Nat.Prime p)
    hp2 : Ne p 2
    hp8 : Eq (HMod.hMod p 8) 1
    a : ZMod p
    ha : Eq 2 (HMul.hMul a a)
    ⊢ Dvd.dvd p (HAdd.hAdd (HPow.hPow a.val (HPow.hPow 2 (HAdd.hAdd n 1))) 1)
  -/
  rw [fermatNumber] at hpdvd
  /-
    case intro
    n p : Nat
    hn : LT.lt 1 n
    hp : Nat.Prime p
    hpdvd : Dvd.dvd p (HAdd.hAdd (HPow.hPow 2 (HPow.hPow 2 n)) 1)
    this : Fact (Nat.Prime p)
    hp2 : Ne p 2
    hp8 : Eq (HMod.hMod p 8) 1
    a : ZMod p
    ha : Eq 2 (HMul.hMul a a)
    ⊢ Dvd.dvd p (HAdd.hAdd (HPow.hPow a.val (HPow.hPow 2 (HAdd.hAdd n 1))) 1)
  -/
  rw [← natCast_zmod_eq_zero_iff_dvd, Nat.cast_add _ 1, Nat.cast_one, Nat.cast_pow] at hpdvd ⊢
  /-
    case intro
    n p : Nat
    hn : LT.lt 1 n
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    hpdvd : Eq (HAdd.hAdd (HPow.hPow (↑2) (HPow.hPow 2 n)) 1) 0
    hp2 : Ne p 2
    hp8 : Eq (HMod.hMod p 8) 1
    a : ZMod p
    ha : Eq 2 (HMul.hMul a a)
    ⊢ Eq (HAdd.hAdd (HPow.hPow (↑a.val) (HPow.hPow 2 (HAdd.hAdd n 1))) 1) 0
  -/
  rwa [natCast_val, ZMod.cast_id, pow_succ', pow_mul, sq, ← ha]
  /-
    🎉 no goals
  -/


-- TODO: move to NumberTheory.Mersenne, once we have that.

/-- Prime `a ^ n - 1` implies `a = 2` and prime `n`. -/
theorem prime_of_pow_sub_one_prime {a n : ℕ} (hn1 : n ≠ 1) (hP : (a ^ n - 1).Prime) :
    a = 2 ∧ n.Prime := by
  /-
    a n : Nat
    hn1 : Ne n 1
    hP : Nat.Prime (HSub.hSub (HPow.hPow a n) 1)
    ⊢ And (Eq a 2) (Nat.Prime n)
  -/
  have han1 : 1 < a ^ n := tsub_pos_iff_lt.mp hP.pos
  /-
    a n : Nat
    hn1 : Ne n 1
    hP : Nat.Prime (HSub.hSub (HPow.hPow a n) 1)
    han1 : LT.lt 1 (HPow.hPow a n)
    ⊢ And (Eq a 2) (Nat.Prime n)
  -/
  have hn0 : n ≠ 0 := fun h ↦ (h ▸ han1).ne' rfl
  /-
    a n : Nat
    hn1 : Ne n 1
    hP : Nat.Prime (HSub.hSub (HPow.hPow a n) 1)
    han1 : LT.lt 1 (HPow.hPow a n)
    hn0 : Ne n 0
    ⊢ And (Eq a 2) (Nat.Prime n)
  -/
  have ha1 : 1 < a := (Nat.one_lt_pow_iff hn0).mp han1
  /-
    a n : Nat
    hn1 : Ne n 1
    hP : Nat.Prime (HSub.hSub (HPow.hPow a n) 1)
    han1 : LT.lt 1 (HPow.hPow a n)
    hn0 : Ne n 0
    ha1 : LT.lt 1 a
    ⊢ And (Eq a 2) (Nat.Prime n)
  -/
  have ha0 : 0 < a := one_pos.trans ha1
  have ha2 : a = 2 := by
    contrapose! hn1
    let h := nat_sub_dvd_pow_sub_pow a 1 n
    rw [one_pow, hP.dvd_iff_eq (mt (Nat.sub_eq_iff_eq_add ha1.le).mp hn1), eq_comm] at h
    exact (pow_eq_self_iff ha1).mp (Nat.sub_one_cancel ha0 (pow_pos ha0 n) h).symm
  /-
    a n : Nat
    hn1 : Ne n 1
    hP : Nat.Prime (HSub.hSub (HPow.hPow a n) 1)
    han1 : LT.lt 1 (HPow.hPow a n)
    hn0 : Ne n 0
    ha1 : LT.lt 1 a
    ha0 : LT.lt 0 a
    ha2 : Eq a 2
    ⊢ And (Eq a 2) (Nat.Prime n)
  -/
  subst ha2
  /-
    n : Nat
    hn1 : Ne n 1
    hn0 : Ne n 0
    hP : Nat.Prime (HSub.hSub (HPow.hPow 2 n) 1)
    han1 : LT.lt 1 (HPow.hPow 2 n)
    ha1 : LT.lt 1 2
    ha0 : LT.lt 0 2
    ⊢ And (Eq 2 2) (Nat.Prime n)
  -/
  refine ⟨rfl, Nat.prime_def.mpr ⟨(two_le_iff n).mpr ⟨hn0, hn1⟩, fun d hdn ↦ ?_⟩⟩
  have hinj : ∀ x y, 2 ^ x - 1 = 2 ^ y - 1 → x = y :=
    fun x y h ↦ Nat.pow_right_injective le_rfl (sub_one_cancel (pow_pos ha0 x) (pow_pos ha0 y) h)
  /-
    n : Nat
    hn1 : Ne n 1
    hn0 : Ne n 0
    hP : Nat.Prime (HSub.hSub (HPow.hPow 2 n) 1)
    han1 : LT.lt 1 (HPow.hPow 2 n)
    ha1 : LT.lt 1 2
    ha0 : LT.lt 0 2
    d : Nat
    hdn : Dvd.dvd d n
    hinj : ∀ (x y : Nat), Eq (HSub.hSub (HPow.hPow 2 x) 1) (HSub.hSub (HPow.hPow 2 …
    ⊢ Or (Eq d 1) (Eq d n)
  -/
  let h := nat_sub_dvd_pow_sub_pow (2 ^ d) 1 (n / d)
  /-
    n : Nat
    hn1 : Ne n 1
    hn0 : Ne n 0
    hP : Nat.Prime (HSub.hSub (HPow.hPow 2 n) 1)
    han1 : LT.lt 1 (HPow.hPow 2 n)
    ha1 : LT.lt 1 2
    ha0 : LT.lt 0 2
    d : Nat
    hdn : Dvd.dvd d n
    hinj : ∀ (x y : Nat), Eq (HSub.hSub (HPow.hPow 2 x) 1) (HSub.hSub (HPow.hPow 2 …
    h : Dvd.dvd (HSub.hSub (HPow.hPow 2 d) 1) (HSub.hSub (HPow.hPow (HPow.hPow 2 d …
    ⊢ Or (Eq d 1) (Eq d n)
  -/
  rw [one_pow, ← pow_mul, Nat.mul_div_cancel' hdn] at h
  /-
    n : Nat
    hn1 : Ne n 1
    hn0 : Ne n 0
    hP : Nat.Prime (HSub.hSub (HPow.hPow 2 n) 1)
    han1 : LT.lt 1 (HPow.hPow 2 n)
    ha1 : LT.lt 1 2
    ha0 : LT.lt 0 2
    d : Nat
    hdn : Dvd.dvd d n
    hinj : ∀ (x y : Nat), Eq (HSub.hSub (HPow.hPow 2 x) 1) (HSub.hSub (HPow.hPow 2 …
    h : Dvd.dvd (HSub.hSub (HPow.hPow 2 d) 1) (HSub.hSub (HPow.hPow 2 n) 1)
    ⊢ Or (Eq d 1) (Eq d n)
  -/
  exact (hP.eq_one_or_self_of_dvd (2 ^ d - 1) h).imp (hinj d 1) (hinj d n)
  /-
    🎉 no goals
  -/


