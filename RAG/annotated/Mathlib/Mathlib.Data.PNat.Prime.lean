/-- The canonical map from `Nat.Primes` to `ℕ+` -/
@[coe] def toPNat : Nat.Primes → ℕ+ :=
  fun p => ⟨(p : ℕ), p.property.pos⟩


instance coePNat : Coe Nat.Primes ℕ+ :=
  ⟨toPNat⟩


@[norm_cast]
theorem coe_pnat_nat (p : Nat.Primes) : ((p : ℕ+) : ℕ) = p :=
  rfl


theorem coe_pnat_injective : Function.Injective ((↑) : Nat.Primes → ℕ+) := fun p q h =>
                  /-
                    p q : Nat.Primes
                    h : Eq ↑p ↑q
                    ⊢ Eq ↑p ↑q
                  -/
  Subtype.ext (by injection h)
                  /-
                    🎉 no goals
                  -/


@[norm_cast]
theorem coe_pnat_inj (p q : Nat.Primes) : (p : ℕ+) = (q : ℕ+) ↔ p = q :=
  coe_pnat_injective.eq_iff


/-- The greatest common divisor (gcd) of two positive natural numbers,
  viewed as positive natural number. -/
def gcd (n m : ℕ+) : ℕ+ :=
  ⟨Nat.gcd (n : ℕ) (m : ℕ), Nat.gcd_pos_of_pos_left (m : ℕ) n.pos⟩


/-- The least common multiple (lcm) of two positive natural numbers,
  viewed as positive natural number. -/
def lcm (n m : ℕ+) : ℕ+ :=
  ⟨Nat.lcm (n : ℕ) (m : ℕ), by
    /-
      n m : PNat
      ⊢ LT.lt 0 ((↑n).lcm ↑m)
    -/
    let h := mul_pos n.pos m.pos
    /-
      n m : PNat
      h : LT.lt 0 (HMul.hMul ↑n ↑m) := mul_pos (PNat.pos n) (PNat.pos m)
      ⊢ LT.lt 0 ((↑n).lcm ↑m)
    -/
    rw [← gcd_mul_lcm (n : ℕ) (m : ℕ), mul_comm] at h
    /-
      n m : PNat
      h : LT.lt 0 (HMul.hMul ((↑n).lcm ↑m) ((↑n).gcd ↑m))
      ⊢ LT.lt 0 ((↑n).lcm ↑m)
    -/
    exact pos_of_dvd_of_pos (Dvd.intro (Nat.gcd (n : ℕ) (m : ℕ)) rfl) h⟩
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem gcd_coe (n m : ℕ+) : (gcd n m : ℕ) = Nat.gcd n m :=
  rfl


@[simp, norm_cast]
theorem lcm_coe (n m : ℕ+) : (lcm n m : ℕ) = Nat.lcm n m :=
  rfl


theorem gcd_dvd_left (n m : ℕ+) : gcd n m ∣ n :=
  dvd_iff.2 (Nat.gcd_dvd_left (n : ℕ) (m : ℕ))


theorem gcd_dvd_right (n m : ℕ+) : gcd n m ∣ m :=
  dvd_iff.2 (Nat.gcd_dvd_right (n : ℕ) (m : ℕ))


theorem dvd_gcd {m n k : ℕ+} (hm : k ∣ m) (hn : k ∣ n) : k ∣ gcd m n :=
  dvd_iff.2 (Nat.dvd_gcd (dvd_iff.1 hm) (dvd_iff.1 hn))


theorem dvd_lcm_left (n m : ℕ+) : n ∣ lcm n m :=
  dvd_iff.2 (Nat.dvd_lcm_left (n : ℕ) (m : ℕ))


theorem dvd_lcm_right (n m : ℕ+) : m ∣ lcm n m :=
  dvd_iff.2 (Nat.dvd_lcm_right (n : ℕ) (m : ℕ))


theorem lcm_dvd {m n k : ℕ+} (hm : m ∣ k) (hn : n ∣ k) : lcm m n ∣ k :=
  dvd_iff.2 (@Nat.lcm_dvd (m : ℕ) (n : ℕ) (k : ℕ) (dvd_iff.1 hm) (dvd_iff.1 hn))


theorem gcd_mul_lcm (n m : ℕ+) : gcd n m * lcm n m = n * m :=
  Subtype.eq (Nat.gcd_mul_lcm (n : ℕ) (m : ℕ))


theorem eq_one_of_lt_two {n : ℕ+} : n < 2 → n = 1 := by
  /-
    n : PNat
    ⊢ LT.lt n 2 → Eq n 1
  -/
  intro h; apply le_antisymm; swap
    /-
      case a
      n : PNat
      h : LT.lt n 2
      ⊢ LE.le 1 n
    -/
  · apply PNat.one_le
    /-
      🎉 no goals
    -/
    /-
      case a
      n : PNat
      h : LT.lt n 2
      ⊢ LE.le n 1
    -/
  · exact PNat.lt_add_one_iff.1 h
    /-
      🎉 no goals
    -/


/-- Primality predicate for `ℕ+`, defined in terms of `Nat.Prime`. -/
def Prime (p : ℕ+) : Prop :=
  (p : ℕ).Prime


theorem Prime.one_lt {p : ℕ+} : p.Prime → 1 < p :=
  Nat.Prime.one_lt


theorem prime_two : (2 : ℕ+).Prime :=
  Nat.prime_two


instance {p : ℕ+} [h : Fact p.Prime] : Fact (p : ℕ).Prime := h


instance fact_prime_two : Fact (2 : ℕ+).Prime :=
  ⟨prime_two⟩


theorem prime_three : (3 : ℕ+).Prime :=
  Nat.prime_three


instance fact_prime_three : Fact (3 : ℕ+).Prime :=
  ⟨prime_three⟩


theorem prime_five : (5 : ℕ+).Prime :=
  Nat.prime_five


instance fact_prime_five : Fact (5 : ℕ+).Prime :=
  ⟨prime_five⟩


theorem dvd_prime {p m : ℕ+} (pp : p.Prime) : m ∣ p ↔ m = 1 ∨ m = p := by
  /-
    p m : PNat
    pp : p.Prime
    ⊢ Iff (Dvd.dvd m p) (Or (Eq m 1) (Eq m p))
  -/
  rw [PNat.dvd_iff]
  /-
    p m : PNat
    pp : p.Prime
    ⊢ Iff (Dvd.dvd ↑m ↑p) (Or (Eq m 1) (Eq m p))
  -/
  rw [Nat.dvd_prime pp]
  /-
    p m : PNat
    pp : p.Prime
    ⊢ Iff (Or (Eq (↑m) 1) (Eq ↑m ↑p)) (Or (Eq m 1) (Eq m p))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Prime.ne_one {p : ℕ+} : p.Prime → p ≠ 1 := by
  /-
    p : PNat
    ⊢ p.Prime → Ne p 1
  -/
  intro pp
  /-
    p : PNat
    pp : p.Prime
    ⊢ Ne p 1
  -/
  intro contra
  /-
    p : PNat
    pp : p.Prime
    contra : Eq p 1
    ⊢ False
  -/
  apply Nat.Prime.ne_one pp
  /-
    p : PNat
    pp : p.Prime
    contra : Eq p 1
    ⊢ Eq (↑p) 1
  -/
  rw [PNat.coe_eq_one_iff]
  /-
    p : PNat
    pp : p.Prime
    contra : Eq p 1
    ⊢ Eq p 1
  -/
  apply contra
  /-
    🎉 no goals
  -/


@[simp]
theorem not_prime_one : ¬(1 : ℕ+).Prime :=
  Nat.not_prime_one


theorem Prime.not_dvd_one {p : ℕ+} : p.Prime → ¬p ∣ 1 := fun pp : p.Prime => by
  /-
    p : PNat
    pp : p.Prime
    ⊢ Not (Dvd.dvd p 1)
  -/
  rw [dvd_iff]
  /-
    p : PNat
    pp : p.Prime
    ⊢ Not (Dvd.dvd ↑p ↑1)
  -/
  apply Nat.Prime.not_dvd_one pp
  /-
    🎉 no goals
  -/


theorem exists_prime_and_dvd {n : ℕ+} (hn : n ≠ 1) : ∃ p : ℕ+, p.Prime ∧ p ∣ n := by
  /-
    n : PNat
    hn : Ne n 1
    ⊢ Exists fun p => And p.Prime (Dvd.dvd p n)
  -/
  obtain ⟨p, hp⟩ := Nat.exists_prime_and_dvd (mt coe_eq_one_iff.mp hn)
  /-
    case intro
    n : PNat
    hn : Ne n 1
    p : Nat
    hp : And (Nat.Prime p) (Dvd.dvd p ↑n)
    ⊢ Exists fun p => And p.Prime (Dvd.dvd p n)
  -/
  exists (⟨p, Nat.Prime.pos hp.left⟩ : ℕ+); rw [dvd_iff]; apply hp
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Two pnats are coprime if their gcd is 1. -/
def Coprime (m n : ℕ+) : Prop :=
  m.gcd n = 1


@[simp, norm_cast]
theorem coprime_coe {m n : ℕ+} : Nat.Coprime ↑m ↑n ↔ m.Coprime n := by
  /-
    m n : PNat
    ⊢ Iff ((↑m).Coprime ↑n) (m.Coprime n)
  -/
  unfold Nat.Coprime Coprime
  /-
    m n : PNat
    ⊢ Iff (Eq ((↑m).gcd ↑n) 1) (Eq (m.gcd n) 1)
  -/
  rw [← coe_inj]
  /-
    m n : PNat
    ⊢ Iff (Eq ((↑m).gcd ↑n) 1) (Eq ↑(m.gcd n) ↑1)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Coprime.mul {k m n : ℕ+} : m.Coprime k → n.Coprime k → (m * n).Coprime k := by
  /-
    k m n : PNat
    ⊢ m.Coprime k → n.Coprime k → (HMul.hMul m n).Coprime k
  -/
  repeat rw [← coprime_coe]
  /-
    k m n : PNat
    ⊢ (↑m).Coprime ↑k → (↑n).Coprime ↑k → (↑(HMul.hMul m n)).Coprime ↑k
  -/
  rw [mul_coe]
  /-
    k m n : PNat
    ⊢ (↑m).Coprime ↑k → (↑n).Coprime ↑k → (HMul.hMul ↑m ↑n).Coprime ↑k
  -/
  apply Nat.Coprime.mul
  /-
    🎉 no goals
  -/


theorem Coprime.mul_right {k m n : ℕ+} : k.Coprime m → k.Coprime n → k.Coprime (m * n) := by
  /-
    k m n : PNat
    ⊢ k.Coprime m → k.Coprime n → k.Coprime (HMul.hMul m n)
  -/
  repeat rw [← coprime_coe]
  /-
    k m n : PNat
    ⊢ (↑k).Coprime ↑m → (↑k).Coprime ↑n → (↑k).Coprime ↑(HMul.hMul m n)
  -/
  rw [mul_coe]
  /-
    k m n : PNat
    ⊢ (↑k).Coprime ↑m → (↑k).Coprime ↑n → (↑k).Coprime (HMul.hMul ↑m ↑n)
  -/
  apply Nat.Coprime.mul_right
  /-
    🎉 no goals
  -/


theorem gcd_comm {m n : ℕ+} : m.gcd n = n.gcd m := by
  /-
    m n : PNat
    ⊢ Eq (m.gcd n) (n.gcd m)
  -/
  apply eq
  /-
    case a
    m n : PNat
    ⊢ Eq ↑(m.gcd n) ↑(n.gcd m)
  -/
  simp only [gcd_coe]
  /-
    case a
    m n : PNat
    ⊢ Eq ((↑m).gcd ↑n) ((↑n).gcd ↑m)
  -/
  apply Nat.gcd_comm
  /-
    🎉 no goals
  -/


theorem gcd_eq_left_iff_dvd {m n : ℕ+} : m ∣ n ↔ m.gcd n = m := by
  /-
    m n : PNat
    ⊢ Iff (Dvd.dvd m n) (Eq (m.gcd n) m)
  -/
  rw [dvd_iff]
  /-
    m n : PNat
    ⊢ Iff (Dvd.dvd ↑m ↑n) (Eq (m.gcd n) m)
  -/
  rw [Nat.gcd_eq_left_iff_dvd]
  /-
    m n : PNat
    ⊢ Iff (Eq ((↑m).gcd ↑n) ↑m) (Eq (m.gcd n) m)
  -/
  rw [← coe_inj]
  /-
    m n : PNat
    ⊢ Iff (Eq ((↑m).gcd ↑n) ↑m) (Eq ↑(m.gcd n) ↑m)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem gcd_eq_right_iff_dvd {m n : ℕ+} : m ∣ n ↔ n.gcd m = m := by
  /-
    m n : PNat
    ⊢ Iff (Dvd.dvd m n) (Eq (n.gcd m) m)
  -/
  rw [gcd_comm]
  /-
    m n : PNat
    ⊢ Iff (Dvd.dvd m n) (Eq (m.gcd n) m)
  -/
  apply gcd_eq_left_iff_dvd
  /-
    🎉 no goals
  -/


theorem Coprime.gcd_mul_left_cancel (m : ℕ+) {n k : ℕ+} :
    k.Coprime n → (k * m).gcd n = m.gcd n := by
  /-
    m n k : PNat
    ⊢ k.Coprime n → Eq ((HMul.hMul k m).gcd n) (m.gcd n)
  -/
  intro h; apply eq; simp only [gcd_coe, mul_coe]
  /-
    case a
    m n k : PNat
    h : k.Coprime n
    ⊢ Eq ((HMul.hMul ↑k ↑m).gcd ↑n) ((↑m).gcd ↑n)
  -/
  apply Nat.Coprime.gcd_mul_left_cancel; simpa
                                         /-
                                           🎉 no goals
                                         -/


theorem Coprime.gcd_mul_right_cancel (m : ℕ+) {n k : ℕ+} :
                                                /-
                                                  m n k : PNat
                                                  ⊢ k.Coprime n → Eq ((HMul.hMul m k).gcd n) (m.gcd n)
                                                -/
    k.Coprime n → (m * k).gcd n = m.gcd n := by rw [mul_comm]; apply Coprime.gcd_mul_left_cancel
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem Coprime.gcd_mul_left_cancel_right (m : ℕ+) {n k : ℕ+} :
    k.Coprime m → m.gcd (k * n) = m.gcd n := by
  /-
    m n k : PNat
    ⊢ k.Coprime m → Eq (m.gcd (HMul.hMul k n)) (m.gcd n)
  -/
  intro h; iterate 2 rw [gcd_comm]; symm
  /-
    m n k : PNat
    h : k.Coprime m
    ⊢ Eq ((HMul.hMul k n).gcd m) (n.gcd m)
  -/
  apply Coprime.gcd_mul_left_cancel _ h
  /-
    🎉 no goals
  -/


theorem Coprime.gcd_mul_right_cancel_right (m : ℕ+) {n k : ℕ+} :
    k.Coprime m → m.gcd (n * k) = m.gcd n := by
  /-
    m n k : PNat
    ⊢ k.Coprime m → Eq (m.gcd (HMul.hMul n k)) (m.gcd n)
  -/
  rw [mul_comm]
  /-
    m n k : PNat
    ⊢ k.Coprime m → Eq (m.gcd (HMul.hMul k n)) (m.gcd n)
  -/
  apply Coprime.gcd_mul_left_cancel_right
  /-
    🎉 no goals
  -/


@[simp]
theorem one_gcd {n : ℕ+} : gcd 1 n = 1 := by
  /-
    n : PNat
    ⊢ Eq (PNat.gcd 1 n) 1
  -/
  rw [← gcd_eq_left_iff_dvd]
  /-
    n : PNat
    ⊢ Dvd.dvd 1 n
  -/
  apply one_dvd
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_one {n : ℕ+} : gcd n 1 = 1 := by
  /-
    n : PNat
    ⊢ Eq (n.gcd 1) 1
  -/
  rw [gcd_comm]
  /-
    n : PNat
    ⊢ Eq (PNat.gcd 1 n) 1
  -/
  apply one_gcd
  /-
    🎉 no goals
  -/


@[symm]
theorem Coprime.symm {m n : ℕ+} : m.Coprime n → n.Coprime m := by
  /-
    m n : PNat
    ⊢ m.Coprime n → n.Coprime m
  -/
  unfold Coprime
  /-
    m n : PNat
    ⊢ Eq (m.gcd n) 1 → Eq (n.gcd m) 1
  -/
  rw [gcd_comm]
  /-
    m n : PNat
    ⊢ Eq (n.gcd m) 1 → Eq (n.gcd m) 1
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem one_coprime {n : ℕ+} : (1 : ℕ+).Coprime n :=
  one_gcd


@[simp]
theorem coprime_one {n : ℕ+} : n.Coprime 1 :=
  Coprime.symm one_coprime


theorem Coprime.coprime_dvd_left {m k n : ℕ+} : m ∣ k → k.Coprime n → m.Coprime n := by
  /-
    m k n : PNat
    ⊢ Dvd.dvd m k → k.Coprime n → m.Coprime n
  -/
  rw [dvd_iff]
  /-
    m k n : PNat
    ⊢ Dvd.dvd ↑m ↑k → k.Coprime n → m.Coprime n
  -/
  repeat rw [← coprime_coe]
  /-
    m k n : PNat
    ⊢ Dvd.dvd ↑m ↑k → (↑k).Coprime ↑n → (↑m).Coprime ↑n
  -/
  apply Nat.Coprime.coprime_dvd_left
  /-
    🎉 no goals
  -/


theorem Coprime.factor_eq_gcd_left {a b m n : ℕ+} (cop : m.Coprime n) (am : a ∣ m) (bn : b ∣ n) :
    a = (a * b).gcd m := by
  /-
    a b m n : PNat
    cop : m.Coprime n
    am : Dvd.dvd a m
    bn : Dvd.dvd b n
    ⊢ Eq a ((HMul.hMul a b).gcd m)
  -/
  rw [gcd_eq_left_iff_dvd] at am
  /-
    a b m n : PNat
    cop : m.Coprime n
    am : Eq (a.gcd m) a
    bn : Dvd.dvd b n
    ⊢ Eq a ((HMul.hMul a b).gcd m)
  -/
  conv_lhs => rw [← am]
  /-
    a b m n : PNat
    cop : m.Coprime n
    am : Eq (a.gcd m) a
    bn : Dvd.dvd b n
    ⊢ Eq (a.gcd m) ((HMul.hMul a b).gcd m)
  -/
  rw [eq_comm]
  /-
    a b m n : PNat
    cop : m.Coprime n
    am : Eq (a.gcd m) a
    bn : Dvd.dvd b n
    ⊢ Eq ((HMul.hMul a b).gcd m) (a.gcd m)
  -/
  apply Coprime.gcd_mul_right_cancel a
  /-
    a b m n : PNat
    cop : m.Coprime n
    am : Eq (a.gcd m) a
    bn : Dvd.dvd b n
    ⊢ b.Coprime m
  -/
  apply Coprime.coprime_dvd_left bn cop.symm
  /-
    🎉 no goals
  -/


theorem Coprime.factor_eq_gcd_right {a b m n : ℕ+} (cop : m.Coprime n) (am : a ∣ m) (bn : b ∣ n) :
                            /-
                              a b m n : PNat
                              cop : m.Coprime n
                              am : Dvd.dvd a m
                              bn : Dvd.dvd b n
                              ⊢ Eq a ((HMul.hMul b a).gcd m)
                            -/
    a = (b * a).gcd m := by rw [mul_comm]; apply Coprime.factor_eq_gcd_left cop am bn
                                           /-
                                             🎉 no goals
                                           -/


theorem Coprime.factor_eq_gcd_left_right {a b m n : ℕ+} (cop : m.Coprime n) (am : a ∣ m)
                                           /-
                                             a b m n : PNat
                                             cop : m.Coprime n
                                             am : Dvd.dvd a m
                                             bn : Dvd.dvd b n
                                             ⊢ Eq a (m.gcd (HMul.hMul a b))
                                           -/
    (bn : b ∣ n) : a = m.gcd (a * b) := by rw [gcd_comm]; apply Coprime.factor_eq_gcd_left cop am bn
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem Coprime.factor_eq_gcd_right_right {a b m n : ℕ+} (cop : m.Coprime n) (am : a ∣ m)
    (bn : b ∣ n) : a = m.gcd (b * a) := by
  /-
    a b m n : PNat
    cop : m.Coprime n
    am : Dvd.dvd a m
    bn : Dvd.dvd b n
    ⊢ Eq a (m.gcd (HMul.hMul b a))
  -/
  rw [gcd_comm]
  /-
    a b m n : PNat
    cop : m.Coprime n
    am : Dvd.dvd a m
    bn : Dvd.dvd b n
    ⊢ Eq a ((HMul.hMul b a).gcd m)
  -/
  apply Coprime.factor_eq_gcd_right cop am bn
  /-
    🎉 no goals
  -/


theorem Coprime.gcd_mul (k : ℕ+) {m n : ℕ+} (h : m.Coprime n) :
    k.gcd (m * n) = k.gcd m * k.gcd n := by
  /-
    k m n : PNat
    h : m.Coprime n
    ⊢ Eq (k.gcd (HMul.hMul m n)) (HMul.hMul (k.gcd m) (k.gcd n))
  -/
  rw [← coprime_coe] at h; apply eq
  /-
    case a
    k m n : PNat
    h : (↑m).Coprime ↑n
    ⊢ Eq ↑(k.gcd (HMul.hMul m n)) ↑(HMul.hMul (k.gcd m) (k.gcd n))
  -/
  simp only [gcd_coe, mul_coe]; apply Nat.Coprime.gcd_mul k h
                                /-
                                  🎉 no goals
                                -/


theorem gcd_eq_left {m n : ℕ+} : m ∣ n → m.gcd n = m := by
  /-
    m n : PNat
    ⊢ Dvd.dvd m n → Eq (m.gcd n) m
  -/
  rw [dvd_iff]
  /-
    m n : PNat
    ⊢ Dvd.dvd ↑m ↑n → Eq (m.gcd n) m
  -/
  intro h
  /-
    m n : PNat
    h : Dvd.dvd ↑m ↑n
    ⊢ Eq (m.gcd n) m
  -/
  apply eq
  /-
    case a
    m n : PNat
    h : Dvd.dvd ↑m ↑n
    ⊢ Eq ↑(m.gcd n) ↑m
  -/
  simp only [gcd_coe]
  /-
    case a
    m n : PNat
    h : Dvd.dvd ↑m ↑n
    ⊢ Eq ((↑m).gcd ↑n) ↑m
  -/
  apply Nat.gcd_eq_left h
  /-
    🎉 no goals
  -/


theorem Coprime.pow {m n : ℕ+} (k l : ℕ) (h : m.Coprime n) : (m ^ k : ℕ).Coprime (n ^ l) := by
  /-
    m n : PNat
    k l : Nat
    h : m.Coprime n
    ⊢ (HPow.hPow (↑m) k).Coprime (HPow.hPow (↑n) l)
  -/
  rw [← coprime_coe] at *; apply Nat.Coprime.pow; apply h
                                                  /-
                                                    🎉 no goals
                                                  -/


