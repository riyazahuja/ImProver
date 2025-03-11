/-- `n` is a prime power if there is a prime `p` and a positive natural `k` such that `n` can be
written as `p^k`. -/
def IsPrimePow : Prop :=
  ∃ (p : R) (k : ℕ), Prime p ∧ 0 < k ∧ p ^ k = n


theorem isPrimePow_def : IsPrimePow n ↔ ∃ (p : R) (k : ℕ), Prime p ∧ 0 < k ∧ p ^ k = n :=
  Iff.rfl


/-- An equivalent definition for prime powers: `n` is a prime power iff there is a prime `p` and a
natural `k` such that `n` can be written as `p^(k+1)`. -/
theorem isPrimePow_iff_pow_succ : IsPrimePow n ↔ ∃ (p : R) (k : ℕ), Prime p ∧ p ^ (k + 1) = n :=
  (isPrimePow_def _).trans
                                                 /-
                                                   R : Type u_1
                                                   inst✝ : CommMonoidWithZero R
                                                   n : R
                                                   x✝ : Exists fun p => Exists fun k => And (Prime p) (And (LT.lt 0 k) (Eq (HPow. …
                                                   p : R
                                                   k : Nat
                                                   hp : Prime p
                                                   hk : LT.lt 0 k
                                                   hn : Eq (HPow.hPow p k) n
                                                   ⊢ Eq (HPow.hPow p (HAdd.hAdd (HSub.hSub k 1) 1)) n
                                                 -/
    ⟨fun ⟨p, k, hp, hk, hn⟩ => ⟨p, k - 1, hp, by rwa [Nat.sub_add_cancel hk]⟩, fun ⟨_, _, hp, hn⟩ =>
                                                 /-
                                                   🎉 no goals
                                                 -/
      ⟨_, _, hp, Nat.succ_pos', hn⟩⟩


theorem not_isPrimePow_zero [NoZeroDivisors R] : ¬IsPrimePow (0 : R) := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : NoZeroDivisors R
    ⊢ Not (IsPrimePow 0)
  -/
  simp only [isPrimePow_def, not_exists, not_and', and_imp]
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : NoZeroDivisors R
    ⊢ ∀ (x : R) (x_1 : Nat), LT.lt 0 x_1 → Eq (HPow.hPow x x_1) 0 → Not (Prime x)
  -/
  intro x n _hn hx
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : NoZeroDivisors R
    x : R
    n : Nat
    _hn : LT.lt 0 n
    hx : Eq (HPow.hPow x n) 0
    ⊢ Not (Prime x)
  -/
  rw [pow_eq_zero hx]
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    inst✝ : NoZeroDivisors R
    x : R
    n : Nat
    _hn : LT.lt 0 n
    hx : Eq (HPow.hPow x n) 0
    ⊢ Not (Prime 0)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsPrimePow.not_unit {n : R} (h : IsPrimePow n) : ¬IsUnit n :=
  let ⟨_p, _k, hp, hk, hn⟩ := h
  hn ▸ (isUnit_pow_iff hk.ne').not.mpr hp.not_unit


theorem IsUnit.not_isPrimePow {n : R} (h : IsUnit n) : ¬IsPrimePow n := fun h' => h'.not_unit h


theorem not_isPrimePow_one : ¬IsPrimePow (1 : R) :=
  isUnit_one.not_isPrimePow


theorem Prime.isPrimePow {p : R} (hp : Prime p) : IsPrimePow p :=
                             /-
                               R : Type u_1
                               inst✝ : CommMonoidWithZero R
                               p : R
                               hp : Prime p
                               ⊢ Eq (HPow.hPow p 1) p
                             -/
  ⟨p, 1, hp, zero_lt_one, by simp⟩
                             /-
                               🎉 no goals
                             -/


theorem IsPrimePow.pow {n : R} (hn : IsPrimePow n) {k : ℕ} (hk : k ≠ 0) : IsPrimePow (n ^ k) :=
  let ⟨p, k', hp, hk', hn⟩ := hn
                                            /-
                                              R : Type u_1
                                              inst✝ : CommMonoidWithZero R
                                              n : R
                                              hn✝ : IsPrimePow n
                                              k : Nat
                                              hk : Ne k 0
                                              p : R
                                              k' : Nat
                                              hp : Prime p
                                              hk' : LT.lt 0 k'
                                              hn : Eq (HPow.hPow p k') n
                                              ⊢ Eq (HPow.hPow p (HMul.hMul k k')) (HPow.hPow n k)
                                            -/
  ⟨p, k * k', hp, mul_pos hk.bot_lt hk', by rw [pow_mul', hn]⟩
                                            /-
                                              🎉 no goals
                                            -/


theorem IsPrimePow.ne_zero [NoZeroDivisors R] {n : R} (h : IsPrimePow n) : n ≠ 0 := fun t =>
  not_isPrimePow_zero (t ▸ h)


theorem IsPrimePow.ne_one {n : R} (h : IsPrimePow n) : n ≠ 1 := fun t =>
  not_isPrimePow_one (t ▸ h)


theorem isPrimePow_nat_iff (n : ℕ) : IsPrimePow n ↔ ∃ p k : ℕ, Nat.Prime p ∧ 0 < k ∧ p ^ k = n := by
  /-
    n : Nat
    ⊢ Iff (IsPrimePow n) (Exists fun p => Exists fun k => And (Nat.Prime p) (And ( …
  -/
  simp only [isPrimePow_def, Nat.prime_iff]
  /-
    🎉 no goals
  -/


theorem Nat.Prime.isPrimePow {p : ℕ} (hp : p.Prime) : IsPrimePow p :=
  _root_.Prime.isPrimePow (prime_iff.mp hp)


theorem isPrimePow_nat_iff_bounded (n : ℕ) :
    IsPrimePow n ↔ ∃ p : ℕ, p ≤ n ∧ ∃ k : ℕ, k ≤ n ∧ p.Prime ∧ 0 < k ∧ p ^ k = n := by
  /-
    n : Nat
    ⊢ Iff (IsPrimePow n) (Exists fun p => And (LE.le p n) (Exists fun k => And (LE …
  -/
  rw [isPrimePow_nat_iff]
  /-
    n : Nat
    ⊢ Iff (Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq  …
  -/
  refine Iff.symm ⟨fun ⟨p, _, k, _, hp, hk, hn⟩ => ⟨p, k, hp, hk, hn⟩, ?_⟩
  /-
    n : Nat
    ⊢ (Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPo …
  -/
  rintro ⟨p, k, hp, hk, rfl⟩
  /-
    case intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 k
    ⊢ Exists fun p_1 => And (LE.le p_1 (HPow.hPow p k)) (Exists fun k_1 => And (LE …
  -/
  refine ⟨p, ?_, k, (Nat.lt_pow_self hp.one_lt).le, hp, hk, rfl⟩
  /-
    case intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 k
    ⊢ LE.le p (HPow.hPow p k)
  -/
  conv => { lhs; rw [← (pow_one p)] }
  /-
    case intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 k
    ⊢ LE.le (HPow.hPow p 1) (HPow.hPow p k)
  -/
  exact Nat.pow_le_pow_right hp.one_lt.le hk
  /-
    🎉 no goals
  -/


instance {n : ℕ} : Decidable (IsPrimePow n) :=
  decidable_of_iff' _ (isPrimePow_nat_iff_bounded n)


theorem IsPrimePow.dvd {n m : ℕ} (hn : IsPrimePow n) (hm : m ∣ n) (hm₁ : m ≠ 1) : IsPrimePow m := by
  /-
    n m : Nat
    hn : IsPrimePow n
    hm : Dvd.dvd m n
    hm₁ : Ne m 1
    ⊢ IsPrimePow m
  -/
  rw [isPrimePow_nat_iff] at hn ⊢
  /-
    n m : Nat
    hn : Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (H …
    hm : Dvd.dvd m n
    hm₁ : Ne m 1
    ⊢ Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow …
  -/
  rcases hn with ⟨p, k, hp, _hk, rfl⟩
  /-
    case intro.intro.intro.intro
    m : Nat
    hm₁ : Ne m 1
    p k : Nat
    hp : Nat.Prime p
    _hk : LT.lt 0 k
    hm : Dvd.dvd m (HPow.hPow p k)
    ⊢ Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow …
  -/
  obtain ⟨i, hik, rfl⟩ := (Nat.dvd_prime_pow hp).1 hm
  /-
    case intro.intro.intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    _hk : LT.lt 0 k
    i : Nat
    hik : LE.le i k
    hm₁ : Ne (HPow.hPow p i) 1
    hm : Dvd.dvd (HPow.hPow p i) (HPow.hPow p k)
    ⊢ Exists fun p_1 => Exists fun k => And (Nat.Prime p_1) (And (LT.lt 0 k) (Eq ( …
  -/
  refine ⟨p, i, hp, ?_, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    _hk : LT.lt 0 k
    i : Nat
    hik : LE.le i k
    hm₁ : Ne (HPow.hPow p i) 1
    hm : Dvd.dvd (HPow.hPow p i) (HPow.hPow p k)
    ⊢ LT.lt 0 i
  -/
  apply Nat.pos_of_ne_zero
  /-
    case intro.intro.intro.intro.intro.intro.a
    p k : Nat
    hp : Nat.Prime p
    _hk : LT.lt 0 k
    i : Nat
    hik : LE.le i k
    hm₁ : Ne (HPow.hPow p i) 1
    hm : Dvd.dvd (HPow.hPow p i) (HPow.hPow p k)
    ⊢ Ne i 0
  -/
  rintro rfl
  /-
    case intro.intro.intro.intro.intro.intro.a
    p k : Nat
    hp : Nat.Prime p
    _hk : LT.lt 0 k
    hik : LE.le 0 k
    hm₁ : Ne (HPow.hPow p 0) 1
    hm : Dvd.dvd (HPow.hPow p 0) (HPow.hPow p k)
    ⊢ False
  -/
  simp only [pow_zero, ne_eq, not_true_eq_false] at hm₁
  /-
    🎉 no goals
  -/


theorem Nat.disjoint_divisors_filter_isPrimePow {a b : ℕ} (hab : a.Coprime b) :
    Disjoint (a.divisors.filter IsPrimePow) (b.divisors.filter IsPrimePow) := by
  /-
    a b : Nat
    hab : a.Coprime b
    ⊢ Disjoint (Finset.filter IsPrimePow a.divisors) (Finset.filter IsPrimePow b.d …
  -/
  simp only [Finset.disjoint_left, Finset.mem_filter, and_imp, Nat.mem_divisors, not_and]
  /-
    a b : Nat
    hab : a.Coprime b
    ⊢ ∀ ⦃a_1 : Nat⦄, Dvd.dvd a_1 a → Ne a 0 → IsPrimePow a_1 → Dvd.dvd a_1 b → Ne  …
  -/
  rintro n han _ha hn hbn _hb -
  /-
    a b : Nat
    hab : a.Coprime b
    n : Nat
    han : Dvd.dvd n a
    _ha : Ne a 0
    hn : IsPrimePow n
    hbn : Dvd.dvd n b
    _hb : Ne b 0
    ⊢ False
  -/
  exact hn.ne_one (Nat.eq_one_of_dvd_coprimes hab han hbn)
  /-
    🎉 no goals
  -/


theorem IsPrimePow.two_le : ∀ {n : ℕ}, IsPrimePow n → 2 ≤ n
  | 0, h => (not_isPrimePow_zero h).elim
  | 1, h => (not_isPrimePow_one h).elim
  | _n + 2, _ => le_add_self


theorem IsPrimePow.pos {n : ℕ} (hn : IsPrimePow n) : 0 < n :=
  pos_of_gt hn.two_le


theorem IsPrimePow.one_lt {n : ℕ} (h : IsPrimePow n) : 1 < n :=
  h.two_le


