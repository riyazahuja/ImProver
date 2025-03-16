theorem gcd_eq_one_iff_coprime {a b : ℤ} : Int.gcd a b = 1 ↔ IsCoprime a b := by
  /-
    a b : Int
    ⊢ Iff (Eq (a.gcd b) 1) (IsCoprime a b)
  -/
  constructor
    /-
      case mp
      a b : Int
      ⊢ Eq (a.gcd b) 1 → IsCoprime a b
    -/
  · intro hg
    /-
      case mp
      a b : Int
      hg : Eq (a.gcd b) 1
      ⊢ IsCoprime a b
    -/
    obtain ⟨ua, -, ha⟩ := exists_unit_of_abs a
    /-
      case mp.intro.intro
      a b : Int
      hg : Eq (a.gcd b) 1
      ua : Int
      ha : Eq (↑a.natAbs) (HMul.hMul ua a)
      ⊢ IsCoprime a b
    -/
    obtain ⟨ub, -, hb⟩ := exists_unit_of_abs b
    /-
      case mp.intro.intro.intro.intro
      a b : Int
      hg : Eq (a.gcd b) 1
      ua : Int
      ha : Eq (↑a.natAbs) (HMul.hMul ua a)
      ub : Int
      hb : Eq (↑b.natAbs) (HMul.hMul ub b)
      ⊢ IsCoprime a b
    -/
    use Nat.gcdA (Int.natAbs a) (Int.natAbs b) * ua, Nat.gcdB (Int.natAbs a) (Int.natAbs b) * ub
    rw [mul_assoc, ← ha, mul_assoc, ← hb, mul_comm, mul_comm _ (Int.natAbs b : ℤ), ←
      Nat.gcd_eq_gcd_ab, ← gcd_eq_natAbs, hg, Int.ofNat_one]
    /-
      case mpr
      a b : Int
      ⊢ IsCoprime a b → Eq (a.gcd b) 1
    -/
  · rintro ⟨r, s, h⟩
    /-
      case mpr.intro.intro
      a b r s : Int
      h : Eq (HAdd.hAdd (HMul.hMul r a) (HMul.hMul s b)) 1
      ⊢ Eq (a.gcd b) 1
    -/
    by_contra hg
    /-
      case mpr.intro.intro
      a b r s : Int
      h : Eq (HAdd.hAdd (HMul.hMul r a) (HMul.hMul s b)) 1
      hg : Not (Eq (a.gcd b) 1)
      ⊢ False
    -/
    obtain ⟨p, ⟨hp, ha, hb⟩⟩ := Nat.Prime.not_coprime_iff_dvd.mp hg
    /-
      case mpr.intro.intro.intro.intro.intro
      a b r s : Int
      h : Eq (HAdd.hAdd (HMul.hMul r a) (HMul.hMul s b)) 1
      hg : Not (Eq (a.gcd b) 1)
      p : Nat
      hp : Nat.Prime p
      ha : Dvd.dvd p a.natAbs
      hb : Dvd.dvd p b.natAbs
      ⊢ False
    -/
    apply Nat.Prime.not_dvd_one hp
    /-
      case mpr.intro.intro.intro.intro.intro
      a b r s : Int
      h : Eq (HAdd.hAdd (HMul.hMul r a) (HMul.hMul s b)) 1
      hg : Not (Eq (a.gcd b) 1)
      p : Nat
      hp : Nat.Prime p
      ha : Dvd.dvd p a.natAbs
      hb : Dvd.dvd p b.natAbs
      ⊢ Dvd.dvd p 1
    -/
    rw [← natCast_dvd_natCast, Int.ofNat_one, ← h]
    /-
      case mpr.intro.intro.intro.intro.intro
      a b r s : Int
      h : Eq (HAdd.hAdd (HMul.hMul r a) (HMul.hMul s b)) 1
      hg : Not (Eq (a.gcd b) 1)
      p : Nat
      hp : Nat.Prime p
      ha : Dvd.dvd p a.natAbs
      hb : Dvd.dvd p b.natAbs
      ⊢ Dvd.dvd (↑p) (HAdd.hAdd (HMul.hMul r a) (HMul.hMul s b))
    -/
    exact dvd_add ((natCast_dvd.mpr ha).mul_left _) ((natCast_dvd.mpr hb).mul_left _)
    /-
      🎉 no goals
    -/


theorem coprime_iff_nat_coprime {a b : ℤ} : IsCoprime a b ↔ Nat.Coprime a.natAbs b.natAbs := by
  /-
    a b : Int
    ⊢ Iff (IsCoprime a b) (a.natAbs.Coprime b.natAbs)
  -/
  rw [← gcd_eq_one_iff_coprime, Nat.coprime_iff_gcd_eq_one, gcd_eq_natAbs]
  /-
    🎉 no goals
  -/


/-- If `gcd a (m * n) ≠ 1`, then `gcd a m ≠ 1` or `gcd a n ≠ 1`. -/
theorem gcd_ne_one_iff_gcd_mul_right_ne_one {a : ℤ} {m n : ℕ} :
    a.gcd (m * n) ≠ 1 ↔ a.gcd m ≠ 1 ∨ a.gcd n ≠ 1 := by
  /-
    a : Int
    m n : Nat
    ⊢ Iff (Ne (a.gcd (HMul.hMul ↑m ↑n)) 1) (Or (Ne (a.gcd ↑m) 1) (Ne (a.gcd ↑n) 1))
  -/
  simp only [gcd_eq_one_iff_coprime, ← not_and_or, not_iff_not, IsCoprime.mul_right_iff]
  /-
    🎉 no goals
  -/


theorem sq_of_gcd_eq_one {a b c : ℤ} (h : Int.gcd a b = 1) (heq : a * b = c ^ 2) :
    ∃ a0 : ℤ, a = a0 ^ 2 ∨ a = -a0 ^ 2 := by
  have h' : IsUnit (GCDMonoid.gcd a b) := by
    rw [← coe_gcd, h, Int.ofNat_one]
    exact isUnit_one
  /-
    a b c : Int
    h : Eq (a.gcd b) 1
    heq : Eq (HMul.hMul a b) (HPow.hPow c 2)
    h' : IsUnit (GCDMonoid.gcd a b)
    ⊢ Exists fun a0 => Or (Eq a (HPow.hPow a0 2)) (Eq a (Neg.neg (HPow.hPow a0 2)))
  -/
  obtain ⟨d, ⟨u, hu⟩⟩ := exists_associated_pow_of_mul_eq_pow h' heq
  /-
    case intro.intro
    a b c : Int
    h : Eq (a.gcd b) 1
    heq : Eq (HMul.hMul a b) (HPow.hPow c 2)
    h' : IsUnit (GCDMonoid.gcd a b)
    d : Int
    u : Units Int
    hu : Eq (HMul.hMul (HPow.hPow d 2) ↑u) a
    ⊢ Exists fun a0 => Or (Eq a (HPow.hPow a0 2)) (Eq a (Neg.neg (HPow.hPow a0 2)))
  -/
  use d
  /-
    case h
    a b c : Int
    h : Eq (a.gcd b) 1
    heq : Eq (HMul.hMul a b) (HPow.hPow c 2)
    h' : IsUnit (GCDMonoid.gcd a b)
    d : Int
    u : Units Int
    hu : Eq (HMul.hMul (HPow.hPow d 2) ↑u) a
    ⊢ Or (Eq a (HPow.hPow d 2)) (Eq a (Neg.neg (HPow.hPow d 2)))
  -/
  rw [← hu]
  /-
    case h
    a b c : Int
    h : Eq (a.gcd b) 1
    heq : Eq (HMul.hMul a b) (HPow.hPow c 2)
    h' : IsUnit (GCDMonoid.gcd a b)
    d : Int
    u : Units Int
    hu : Eq (HMul.hMul (HPow.hPow d 2) ↑u) a
    ⊢ Or (Eq (HMul.hMul (HPow.hPow d 2) ↑u) (HPow.hPow d 2)) (Eq (HMul.hMul (HPow. …
  -/
  cases' Int.units_eq_one_or u with hu' hu' <;>
      /-
        case h.inl
        a b c : Int
        h : Eq (a.gcd b) 1
        heq : Eq (HMul.hMul a b) (HPow.hPow c 2)
        h' : IsUnit (GCDMonoid.gcd a b)
        d : Int
        u : Units Int
        hu : Eq (HMul.hMul (HPow.hPow d 2) ↑u) a
        hu' : Eq u 1
        ⊢ Or (Eq (HMul.hMul (HPow.hPow d 2) ↑u) (HPow.hPow d 2)) (Eq (HMul.hMul (HPow. …
      -/
      /-
        case h.inl
        a b c : Int
        h : Eq (a.gcd b) 1
        heq : Eq (HMul.hMul a b) (HPow.hPow c 2)
        h' : IsUnit (GCDMonoid.gcd a b)
        d : Int
        u : Units Int
        hu : Eq (HMul.hMul (HPow.hPow d 2) ↑u) a
        hu' : Eq u 1
        ⊢ Or (Eq (HMul.hMul (HPow.hPow d 2) ↑1) (HPow.hPow d 2)) (Eq (HMul.hMul (HPow. …
      -/
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        a b c : Int
        h : Eq (a.gcd b) 1
        heq : Eq (HMul.hMul a b) (HPow.hPow c 2)
        h' : IsUnit (GCDMonoid.gcd a b)
        d : Int
        u : Units Int
        hu : Eq (HMul.hMul (HPow.hPow d 2) ↑u) a
        hu' : Eq u (-1)
        ⊢ Or (Eq (HMul.hMul (HPow.hPow d 2) ↑(-1)) (HPow.hPow d 2)) (Eq (HMul.hMul (HP …
      -/
      simp
      /-
        🎉 no goals
      -/


theorem sq_of_coprime {a b c : ℤ} (h : IsCoprime a b) (heq : a * b = c ^ 2) :
    ∃ a0 : ℤ, a = a0 ^ 2 ∨ a = -a0 ^ 2 :=
  sq_of_gcd_eq_one (gcd_eq_one_iff_coprime.mpr h) heq


theorem natAbs_euclideanDomain_gcd (a b : ℤ) :
    Int.natAbs (EuclideanDomain.gcd a b) = Int.gcd a b := by
  /-
    a b : Int
    ⊢ Eq (EuclideanDomain.gcd a b).natAbs (a.gcd b)
  -/
  apply Nat.dvd_antisymm <;> rw [← Int.natCast_dvd_natCast]
    /-
      case a
      a b : Int
      ⊢ Dvd.dvd ↑(EuclideanDomain.gcd a b).natAbs ↑(a.gcd b)
    -/
  · rw [Int.natAbs_dvd]
    /-
      case a
      a b : Int
      ⊢ Dvd.dvd (EuclideanDomain.gcd a b) ↑(a.gcd b)
    -/
    exact Int.dvd_gcd (EuclideanDomain.gcd_dvd_left _ _) (EuclideanDomain.gcd_dvd_right _ _)
    /-
      🎉 no goals
    -/
    /-
      case a
      a b : Int
      ⊢ Dvd.dvd ↑(a.gcd b) ↑(EuclideanDomain.gcd a b).natAbs
    -/
  · rw [Int.dvd_natAbs]
    /-
      case a
      a b : Int
      ⊢ Dvd.dvd (↑(a.gcd b)) (EuclideanDomain.gcd a b)
    -/
    exact EuclideanDomain.dvd_gcd Int.gcd_dvd_left Int.gcd_dvd_right
    /-
      🎉 no goals
    -/


theorem Int.Prime.dvd_mul {m n : ℤ} {p : ℕ} (hp : Nat.Prime p) (h : (p : ℤ) ∣ m * n) :
    p ∣ m.natAbs ∨ p ∣ n.natAbs := by
  /-
    m n : Int
    p : Nat
    hp : Nat.Prime p
    h : Dvd.dvd (↑p) (HMul.hMul m n)
    ⊢ Or (Dvd.dvd p m.natAbs) (Dvd.dvd p n.natAbs)
  -/
  rwa [← hp.dvd_mul, ← Int.natAbs_mul, ← Int.natCast_dvd]
  /-
    🎉 no goals
  -/


theorem Int.Prime.dvd_mul' {m n : ℤ} {p : ℕ} (hp : Nat.Prime p) (h : (p : ℤ) ∣ m * n) :
    (p : ℤ) ∣ m ∨ (p : ℤ) ∣ n := by
  /-
    m n : Int
    p : Nat
    hp : Nat.Prime p
    h : Dvd.dvd (↑p) (HMul.hMul m n)
    ⊢ Or (Dvd.dvd (↑p) m) (Dvd.dvd (↑p) n)
  -/
  rw [Int.natCast_dvd, Int.natCast_dvd]
  /-
    m n : Int
    p : Nat
    hp : Nat.Prime p
    h : Dvd.dvd (↑p) (HMul.hMul m n)
    ⊢ Or (Dvd.dvd p m.natAbs) (Dvd.dvd p n.natAbs)
  -/
  exact Int.Prime.dvd_mul hp h
  /-
    🎉 no goals
  -/


theorem Int.Prime.dvd_pow {n : ℤ} {k p : ℕ} (hp : Nat.Prime p) (h : (p : ℤ) ∣ n ^ k) :
    p ∣ n.natAbs := by
  /-
    n : Int
    k p : Nat
    hp : Nat.Prime p
    h : Dvd.dvd (↑p) (HPow.hPow n k)
    ⊢ Dvd.dvd p n.natAbs
  -/
  rw [Int.natCast_dvd, Int.natAbs_pow] at h
  /-
    n : Int
    k p : Nat
    hp : Nat.Prime p
    h : Dvd.dvd p (HPow.hPow n.natAbs k)
    ⊢ Dvd.dvd p n.natAbs
  -/
  exact hp.dvd_of_dvd_pow h
  /-
    🎉 no goals
  -/


theorem Int.Prime.dvd_pow' {n : ℤ} {k p : ℕ} (hp : Nat.Prime p) (h : (p : ℤ) ∣ n ^ k) :
    (p : ℤ) ∣ n := by
  /-
    n : Int
    k p : Nat
    hp : Nat.Prime p
    h : Dvd.dvd (↑p) (HPow.hPow n k)
    ⊢ Dvd.dvd (↑p) n
  -/
  rw [Int.natCast_dvd]
  /-
    n : Int
    k p : Nat
    hp : Nat.Prime p
    h : Dvd.dvd (↑p) (HPow.hPow n k)
    ⊢ Dvd.dvd p n.natAbs
  -/
  exact Int.Prime.dvd_pow hp h
  /-
    🎉 no goals
  -/


theorem prime_two_or_dvd_of_dvd_two_mul_pow_self_two {m : ℤ} {p : ℕ} (hp : Nat.Prime p)
    (h : (p : ℤ) ∣ 2 * m ^ 2) : p = 2 ∨ p ∣ Int.natAbs m := by
  /-
    m : Int
    p : Nat
    hp : Nat.Prime p
    h : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
    ⊢ Or (Eq p 2) (Dvd.dvd p m.natAbs)
  -/
  cases' Int.Prime.dvd_mul hp h with hp2 hpp
    /-
      case inl
      m : Int
      p : Nat
      hp : Nat.Prime p
      h : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      hp2 : Dvd.dvd p (Int.natAbs 2)
      ⊢ Or (Eq p 2) (Dvd.dvd p m.natAbs)
    -/
  · apply Or.intro_left
    /-
      case inl.h
      m : Int
      p : Nat
      hp : Nat.Prime p
      h : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      hp2 : Dvd.dvd p (Int.natAbs 2)
      ⊢ Eq p 2
    -/
    exact le_antisymm (Nat.le_of_dvd zero_lt_two hp2) (Nat.Prime.two_le hp)
    /-
      🎉 no goals
    -/
    /-
      case inr
      m : Int
      p : Nat
      hp : Nat.Prime p
      h : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      hpp : Dvd.dvd p (HPow.hPow m 2).natAbs
      ⊢ Or (Eq p 2) (Dvd.dvd p m.natAbs)
    -/
  · apply Or.intro_right
    /-
      case inr.h
      m : Int
      p : Nat
      hp : Nat.Prime p
      h : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      hpp : Dvd.dvd p (HPow.hPow m 2).natAbs
      ⊢ Dvd.dvd p m.natAbs
    -/
    rw [sq, Int.natAbs_mul] at hpp
    /-
      case inr.h
      m : Int
      p : Nat
      hp : Nat.Prime p
      h : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      hpp : Dvd.dvd p (HMul.hMul m.natAbs m.natAbs)
      ⊢ Dvd.dvd p m.natAbs
    -/
    exact or_self_iff.mp ((Nat.Prime.dvd_mul hp).mp hpp)
    /-
      🎉 no goals
    -/


theorem Int.exists_prime_and_dvd {n : ℤ} (hn : n.natAbs ≠ 1) : ∃ p, Prime p ∧ p ∣ n := by
  /-
    n : Int
    hn : Ne n.natAbs 1
    ⊢ Exists fun p => And (Prime p) (Dvd.dvd p n)
  -/
  obtain ⟨p, pp, pd⟩ := Nat.exists_prime_and_dvd hn
  /-
    case intro.intro
    n : Int
    hn : Ne n.natAbs 1
    p : Nat
    pp : Nat.Prime p
    pd : Dvd.dvd p n.natAbs
    ⊢ Exists fun p => And (Prime p) (Dvd.dvd p n)
  -/
  exact ⟨p, Nat.prime_iff_prime_int.mp pp, Int.natCast_dvd.mpr pd⟩
  /-
    🎉 no goals
  -/



theorem Int.prime_iff_natAbs_prime {k : ℤ} : Prime k ↔ Nat.Prime k.natAbs :=
  (Int.associated_natAbs k).prime_iff.trans Nat.prime_iff_prime_int.symm


theorem zmultiples_natAbs (a : ℤ) :
    AddSubgroup.zmultiples (a.natAbs : ℤ) = AddSubgroup.zmultiples a :=
  le_antisymm (AddSubgroup.zmultiples_le_of_mem (mem_zmultiples_iff.mpr (dvd_natAbs.mpr dvd_rfl)))
    (AddSubgroup.zmultiples_le_of_mem (mem_zmultiples_iff.mpr (natAbs_dvd.mpr dvd_rfl)))


theorem span_natAbs (a : ℤ) : Ideal.span ({(a.natAbs : ℤ)} : Set ℤ) = Ideal.span {a} := by
  /-
    a : Int
    ⊢ Eq (Ideal.span (Singleton.singleton ↑a.natAbs)) (Ideal.span (Singleton.singl …
  -/
  rw [Ideal.span_singleton_eq_span_singleton]
  /-
    a : Int
    ⊢ Associated (↑a.natAbs) a
  -/
  exact (associated_natAbs _).symm
  /-
    🎉 no goals
  -/


theorem eq_pow_of_mul_eq_pow_odd_left {a b c : ℤ} (hab : IsCoprime a b) {k : ℕ} (hk : Odd k)
    (h : a * b = c ^ k) : ∃ d, a = d ^ k := by
  /-
    a b c : Int
    hab : IsCoprime a b
    k : Nat
    hk : Odd k
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    ⊢ Exists fun d => Eq a (HPow.hPow d k)
  -/
  obtain ⟨d, hd⟩ := exists_associated_pow_of_mul_eq_pow' hab h
  /-
    case intro
    a b c : Int
    hab : IsCoprime a b
    k : Nat
    hk : Odd k
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    d : Int
    hd : Associated (HPow.hPow d k) a
    ⊢ Exists fun d => Eq a (HPow.hPow d k)
  -/
  replace hd := hd.symm
  /-
    case intro
    a b c : Int
    hab : IsCoprime a b
    k : Nat
    hk : Odd k
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    d : Int
    hd : Associated a (HPow.hPow d k)
    ⊢ Exists fun d => Eq a (HPow.hPow d k)
  -/
  rw [associated_iff_natAbs, natAbs_eq_natAbs_iff, ← hk.neg_pow] at hd
  /-
    case intro
    a b c : Int
    hab : IsCoprime a b
    k : Nat
    hk : Odd k
    h : Eq (HMul.hMul a b) (HPow.hPow c k)
    d : Int
    hd : Or (Eq a (HPow.hPow d k)) (Eq a (HPow.hPow (Neg.neg d) k))
    ⊢ Exists fun d => Eq a (HPow.hPow d k)
  -/
                             /-
                               🎉 no goals
                             -/
  obtain rfl | rfl := hd <;> exact ⟨_, rfl⟩
                             /-
                               🎉 no goals
                             -/


@[deprecated (since := "2024-07-12")]
alias eq_pow_of_mul_eq_pow_bit1_left := eq_pow_of_mul_eq_pow_odd_left


theorem eq_pow_of_mul_eq_pow_odd_right {a b c : ℤ} (hab : IsCoprime a b) {k : ℕ} (hk : Odd k)
    (h : a * b = c ^ k) : ∃ d, b = d ^ k :=
                                                         /-
                                                           a b c : Int
                                                           hab : IsCoprime a b
                                                           k : Nat
                                                           hk : Odd k
                                                           h : Eq (HMul.hMul a b) (HPow.hPow c k)
                                                           ⊢ Eq (HMul.hMul b a) (HPow.hPow c k)
                                                         -/
  eq_pow_of_mul_eq_pow_odd_left (c := c) hab.symm hk (by rwa [mul_comm] at h)
                                                         /-
                                                           🎉 no goals
                                                         -/


@[deprecated (since := "2024-07-12")]
alias eq_pow_of_mul_eq_pow_bit1_right := eq_pow_of_mul_eq_pow_odd_right


theorem eq_pow_of_mul_eq_pow_odd {a b c : ℤ} (hab : IsCoprime a b) {k : ℕ} (hk : Odd k)
    (h : a * b = c ^ k) : (∃ d, a = d ^ k) ∧ ∃ e, b = e ^ k :=
  ⟨eq_pow_of_mul_eq_pow_odd_left hab hk h, eq_pow_of_mul_eq_pow_odd_right hab hk h⟩


@[deprecated (since := "2024-07-12")] alias eq_pow_of_mul_eq_pow_bit1 := eq_pow_of_mul_eq_pow_odd


