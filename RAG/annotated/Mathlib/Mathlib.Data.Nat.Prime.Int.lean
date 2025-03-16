theorem prime_iff_prime_int {p : ℕ} : p.Prime ↔ _root_.Prime (p : ℤ) :=
  ⟨fun hp =>
    ⟨Int.natCast_ne_zero_iff_pos.2 hp.pos, mt Int.isUnit_iff_natAbs_eq.1 hp.ne_one, fun a b h => by
      /-
        p : Nat
        hp : Nat.Prime p
        a b : Int
        h : Dvd.dvd (↑p) (HMul.hMul a b)
        ⊢ Or (Dvd.dvd (↑p) a) (Dvd.dvd (↑p) b)
      -/
      rw [← Int.dvd_natAbs, Int.natCast_dvd_natCast, Int.natAbs_mul, hp.dvd_mul] at h
      /-
        p : Nat
        hp : Nat.Prime p
        a b : Int
        h : Or (Dvd.dvd p a.natAbs) (Dvd.dvd p b.natAbs)
        ⊢ Or (Dvd.dvd (↑p) a) (Dvd.dvd (↑p) b)
      -/
      rwa [← Int.dvd_natAbs, Int.natCast_dvd_natCast, ← Int.dvd_natAbs, Int.natCast_dvd_natCast]⟩,
      /-
        🎉 no goals
      -/
    fun hp =>
    Nat.prime_iff.2
      ⟨Int.natCast_ne_zero.1 hp.1,
                                          /-
                                            p : Nat
                                            hp : _root_.Prime ↑p
                                            h : Eq p 1
                                            ⊢ False
                                          -/
        (mt Nat.isUnit_iff.1) fun h => by simp [h, not_prime_one] at hp, fun a b => by
                                          /-
                                            🎉 no goals
                                          -/
        /-
          p : Nat
          hp : _root_.Prime ↑p
          a b : Nat
          ⊢ Dvd.dvd p (HMul.hMul a b) → Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        simpa only [Int.natCast_dvd_natCast, (Int.ofNat_mul _ _).symm] using hp.2.2 a b⟩⟩
        /-
          🎉 no goals
        -/


/-- Two prime powers with positive exponents are equal only when the primes and the
exponents are equal. -/
lemma Prime.pow_inj {p q m n : ℕ} (hp : p.Prime) (hq : q.Prime)
    (h : p ^ (m + 1) = q ^ (n + 1)) : p = q ∧ m = n := by
  have H := dvd_antisymm (Prime.dvd_of_dvd_pow hp <| h ▸ dvd_pow_self p (succ_ne_zero m))
    (Prime.dvd_of_dvd_pow hq <| h.symm ▸ dvd_pow_self q (succ_ne_zero n))
  /-
    p q m n : Nat
    hp : Nat.Prime p
    hq : Nat.Prime q
    h : Eq (HPow.hPow p (HAdd.hAdd m 1)) (HPow.hPow q (HAdd.hAdd n 1))
    H : Eq p q
    ⊢ And (Eq p q) (Eq m n)
  -/
  exact ⟨H, succ_inj'.mp <| Nat.pow_right_injective hq.two_le (H ▸ h)⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem prime_ofNat_iff {n : ℕ} :
    Prime (no_index (OfNat.ofNat n : ℤ)) ↔ Nat.Prime (OfNat.ofNat n) :=
  Nat.prime_iff_prime_int.symm


theorem prime_two : Prime (2 : ℤ) :=
  prime_ofNat_iff.mpr Nat.prime_two


theorem prime_three : Prime (3 : ℤ) :=
  prime_ofNat_iff.mpr Nat.prime_three


