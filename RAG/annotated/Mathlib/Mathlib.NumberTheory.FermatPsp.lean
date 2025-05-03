/--
`n` is a probable prime to base `b` if `n` passes the Fermat primality test; that is, `n` divides
`b ^ (n - 1) - 1`.
This definition implies that all numbers are probable primes to base 0 or 1, and that 0 and 1 are
probable primes to any base.
-/
def ProbablePrime (n b : ℕ) : Prop :=
  n ∣ b ^ (n - 1) - 1


/--
`n` is a Fermat pseudoprime to base `b` if `n` is a probable prime to base `b` and is composite. By
this definition, all composite natural numbers are pseudoprimes to base 0 and 1. This definition
also permits `n` to be less than `b`, so that 4 is a pseudoprime to base 5, for example.
-/
def FermatPsp (n b : ℕ) : Prop :=
  ProbablePrime n b ∧ ¬n.Prime ∧ 1 < n


instance decidableProbablePrime (n b : ℕ) : Decidable (ProbablePrime n b) :=
  Nat.decidable_dvd _ _


instance decidablePsp (n b : ℕ) : Decidable (FermatPsp n b) :=
  inferInstanceAs (Decidable (_ ∧ _))


/-- If `n` passes the Fermat primality test to base `b`, then `n` is coprime with `b`, assuming that
`n` and `b` are both positive.
-/
theorem coprime_of_probablePrime {n b : ℕ} (h : ProbablePrime n b) (h₁ : 1 ≤ n) (h₂ : 1 ≤ b) :
    Nat.Coprime n b := by
  /-
    n b : Nat
    h : n.ProbablePrime b
    h₁ : LE.le 1 n
    h₂ : LE.le 1 b
    ⊢ n.Coprime b
  -/
  by_cases h₃ : 2 ≤ n
  · -- To prove that `n` is coprime with `b`, we need to show that for all prime factors of `n`,
    -- we can derive a contradiction if `n` divides `b`.
    /-
      case pos
      n b : Nat
      h : n.ProbablePrime b
      h₁ : LE.le 1 n
      h₂ : LE.le 1 b
      h₃ : LE.le 2 n
      ⊢ n.Coprime b
    -/
    apply Nat.coprime_of_dvd
    -- If `k` is a prime number that divides both `n` and `b`, then we know that `n = m * k` and
    -- `b = j * k` for some natural numbers `m` and `j`. We substitute these into the hypothesis.
    /-
      case pos.H
      n b : Nat
      h : n.ProbablePrime b
      h₁ : LE.le 1 n
      h₂ : LE.le 1 b
      h₃ : LE.le 2 n
      ⊢ ∀ (k : Nat), Nat.Prime k → Dvd.dvd k n → Not (Dvd.dvd k b)
    -/
    rintro k hk ⟨m, rfl⟩ ⟨j, rfl⟩
    -- Because prime numbers do not divide 1, it suffices to show that `k ∣ 1` to prove a
    -- contradiction
    /-
      case pos.H.intro.intro
      k : Nat
      hk : Nat.Prime k
      m : Nat
      h₁ : LE.le 1 (HMul.hMul k m)
      h₃ : LE.le 2 (HMul.hMul k m)
      j : Nat
      h₂ : LE.le 1 (HMul.hMul k j)
      h : (HMul.hMul k m).ProbablePrime (HMul.hMul k j)
      ⊢ False
    -/
    apply Nat.Prime.not_dvd_one hk
    -- Since `n` divides `b ^ (n - 1) - 1`, `k` also divides `b ^ (n - 1) - 1`
    /-
      case pos.H.intro.intro
      k : Nat
      hk : Nat.Prime k
      m : Nat
      h₁ : LE.le 1 (HMul.hMul k m)
      h₃ : LE.le 2 (HMul.hMul k m)
      j : Nat
      h₂ : LE.le 1 (HMul.hMul k j)
      h : (HMul.hMul k m).ProbablePrime (HMul.hMul k j)
      ⊢ Dvd.dvd k 1
    -/
    replace h := dvd_of_mul_right_dvd h
    -- Because `k` divides `b ^ (n - 1) - 1`, if we can show that `k` also divides `b ^ (n - 1)`,
    -- then we know `k` divides 1.
    /-
      case pos.H.intro.intro
      k : Nat
      hk : Nat.Prime k
      m : Nat
      h₁ : LE.le 1 (HMul.hMul k m)
      h₃ : LE.le 2 (HMul.hMul k m)
      j : Nat
      h₂ : LE.le 1 (HMul.hMul k j)
      h : Dvd.dvd k (HSub.hSub (HPow.hPow (HMul.hMul k j) (HSub.hSub (HMul.hMul k m) …
      ⊢ Dvd.dvd k 1
    -/
    rw [Nat.dvd_add_iff_right h, Nat.sub_add_cancel (Nat.one_le_pow _ _ h₂)]
    -- Since `k` divides `b`, `k` also divides any power of `b` except `b ^ 0`. Therefore, it
    -- suffices to show that `n - 1` isn't zero. However, we know that `n - 1` isn't zero because we
    -- assumed `2 ≤ n` when doing `by_cases`.
    /-
      case pos.H.intro.intro
      k : Nat
      hk : Nat.Prime k
      m : Nat
      h₁ : LE.le 1 (HMul.hMul k m)
      h₃ : LE.le 2 (HMul.hMul k m)
      j : Nat
      h₂ : LE.le 1 (HMul.hMul k j)
      h : Dvd.dvd k (HSub.hSub (HPow.hPow (HMul.hMul k j) (HSub.hSub (HMul.hMul k m) …
      ⊢ Dvd.dvd k (HPow.hPow (HMul.hMul k j) (HSub.hSub (HMul.hMul k m) 1))
    -/
    refine dvd_of_mul_right_dvd (dvd_pow_self (k * j) ?_)
    /-
      case pos.H.intro.intro
      k : Nat
      hk : Nat.Prime k
      m : Nat
      h₁ : LE.le 1 (HMul.hMul k m)
      h₃ : LE.le 2 (HMul.hMul k m)
      j : Nat
      h₂ : LE.le 1 (HMul.hMul k j)
      h : Dvd.dvd k (HSub.hSub (HPow.hPow (HMul.hMul k j) (HSub.hSub (HMul.hMul k m) …
      ⊢ Ne (HSub.hSub (HMul.hMul k m) 1) 0
    -/
    omega
    /-
      🎉 no goals
    -/
  -- If `n = 1`, then it follows trivially that `n` is coprime with `b`.
    /-
      case neg
      n b : Nat
      h : n.ProbablePrime b
      h₁ : LE.le 1 n
      h₂ : LE.le 1 b
      h₃ : Not (LE.le 2 n)
      ⊢ n.Coprime b
    -/
  · rw [show n = 1 by omega]
    /-
      case neg
      n b : Nat
      h : n.ProbablePrime b
      h₁ : LE.le 1 n
      h₂ : LE.le 1 b
      h₃ : Not (LE.le 2 n)
      ⊢ Nat.Coprime 1 b
    -/
    norm_num
    /-
      🎉 no goals
    -/


theorem probablePrime_iff_modEq (n : ℕ) {b : ℕ} (h : 1 ≤ b) :
    ProbablePrime n b ↔ b ^ (n - 1) ≡ 1 [MOD n] := by
  /-
    n b : Nat
    h : LE.le 1 b
    ⊢ Iff (n.ProbablePrime b) (n.ModEq (HPow.hPow b (HSub.hSub n 1)) 1)
  -/
  have : 1 ≤ b ^ (n - 1) := one_le_pow₀ h
  -- For exact mod_cast
  /-
    n b : Nat
    h : LE.le 1 b
    this : LE.le 1 (HPow.hPow b (HSub.hSub n 1))
    ⊢ Iff (n.ProbablePrime b) (n.ModEq (HPow.hPow b (HSub.hSub n 1)) 1)
  -/
  rw [Nat.ModEq.comm]
  /-
    n b : Nat
    h : LE.le 1 b
    this : LE.le 1 (HPow.hPow b (HSub.hSub n 1))
    ⊢ Iff (n.ProbablePrime b) (n.ModEq 1 (HPow.hPow b (HSub.hSub n 1)))
  -/
  constructor
    /-
      case mp
      n b : Nat
      h : LE.le 1 b
      this : LE.le 1 (HPow.hPow b (HSub.hSub n 1))
      ⊢ n.ProbablePrime b → n.ModEq 1 (HPow.hPow b (HSub.hSub n 1))
    -/
  · intro h₁
    /-
      case mp
      n b : Nat
      h : LE.le 1 b
      this : LE.le 1 (HPow.hPow b (HSub.hSub n 1))
      h₁ : n.ProbablePrime b
      ⊢ n.ModEq 1 (HPow.hPow b (HSub.hSub n 1))
    -/
    apply Nat.modEq_of_dvd
    /-
      case mp.a
      n b : Nat
      h : LE.le 1 b
      this : LE.le 1 (HPow.hPow b (HSub.hSub n 1))
      h₁ : n.ProbablePrime b
      ⊢ Dvd.dvd (↑n) (HSub.hSub ↑(HPow.hPow b (HSub.hSub n 1)) ↑1)
    -/
    exact mod_cast h₁
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n b : Nat
      h : LE.le 1 b
      this : LE.le 1 (HPow.hPow b (HSub.hSub n 1))
      ⊢ n.ModEq 1 (HPow.hPow b (HSub.hSub n 1)) → n.ProbablePrime b
    -/
  · intro h₁
    /-
      case mpr
      n b : Nat
      h : LE.le 1 b
      this : LE.le 1 (HPow.hPow b (HSub.hSub n 1))
      h₁ : n.ModEq 1 (HPow.hPow b (HSub.hSub n 1))
      ⊢ n.ProbablePrime b
    -/
    exact mod_cast Nat.ModEq.dvd h₁
    /-
      🎉 no goals
    -/


/-- If `n` is a Fermat pseudoprime to base `b`, then `n` is coprime with `b`, assuming that `b` is
positive.

This lemma is a small wrapper based on `coprime_of_probablePrime`
-/
theorem coprime_of_fermatPsp {n b : ℕ} (h : FermatPsp n b) (h₁ : 1 ≤ b) : Nat.Coprime n b := by
  /-
    n b : Nat
    h : n.FermatPsp b
    h₁ : LE.le 1 b
    ⊢ n.Coprime b
  -/
  rcases h with ⟨hp, _, hn₂⟩
  /-
    case intro.intro
    n b : Nat
    h₁ : LE.le 1 b
    hp : n.ProbablePrime b
    left✝ : Not (Nat.Prime n)
    hn₂ : LT.lt 1 n
    ⊢ n.Coprime b
  -/
  exact coprime_of_probablePrime hp (by omega) h₁
  /-
    🎉 no goals
  -/


/-- All composite numbers are Fermat pseudoprimes to base 1.
-/
theorem fermatPsp_base_one {n : ℕ} (h₁ : 1 < n) (h₂ : ¬n.Prime) : FermatPsp n 1 := by
  /-
    n : Nat
    h₁ : LT.lt 1 n
    h₂ : Not (Nat.Prime n)
    ⊢ n.FermatPsp 1
  -/
  refine ⟨show n ∣ 1 ^ (n - 1) - 1 from ?_, h₂, h₁⟩
  /-
    n : Nat
    h₁ : LT.lt 1 n
    h₂ : Not (Nat.Prime n)
    ⊢ Dvd.dvd n (HSub.hSub (HPow.hPow 1 (HSub.hSub n 1)) 1)
  -/
  exact show 0 = 1 ^ (n - 1) - 1 by norm_num ▸ dvd_zero n
  /-
    🎉 no goals
  -/

-- Lemmas that are needed to prove statements in this file, but aren't directly related to Fermat
-- pseudoprimes

private theorem a_id_helper {a b : ℕ} (ha : 2 ≤ a) (hb : 2 ≤ b) : 2 ≤ (a ^ b - 1) / (a - 1) := by
  /-
    a b : Nat
    ha : LE.le 2 a
    hb : LE.le 2 b
    ⊢ LE.le 2 (HDiv.hDiv (HSub.hSub (HPow.hPow a b) 1) (HSub.hSub a 1))
  -/
  change 1 < _
  /-
    a b : Nat
    ha : LE.le 2 a
    hb : LE.le 2 b
    ⊢ LT.lt 1 (HDiv.hDiv (HSub.hSub (HPow.hPow a b) 1) (HSub.hSub a 1))
  -/
  have h₁ : a - 1 ∣ a ^ b - 1 := by simpa only [one_pow] using nat_sub_dvd_pow_sub_pow a 1 b
  /-
    a b : Nat
    ha : LE.le 2 a
    hb : LE.le 2 b
    h₁ : Dvd.dvd (HSub.hSub a 1) (HSub.hSub (HPow.hPow a b) 1)
    ⊢ LT.lt 1 (HDiv.hDiv (HSub.hSub (HPow.hPow a b) 1) (HSub.hSub a 1))
  -/
  rw [Nat.lt_div_iff_mul_lt' h₁, mul_one, tsub_lt_tsub_iff_right (Nat.le_of_succ_le ha)]
  /-
    a b : Nat
    ha : LE.le 2 a
    hb : LE.le 2 b
    h₁ : Dvd.dvd (HSub.hSub a 1) (HSub.hSub (HPow.hPow a b) 1)
    ⊢ LT.lt a (HPow.hPow a b)
  -/
  exact lt_self_pow₀ (Nat.lt_of_succ_le ha) hb
  /-
    🎉 no goals
  -/


private theorem b_id_helper {a b : ℕ} (ha : 2 ≤ a) (hb : 2 < b) : 2 ≤ (a ^ b + 1) / (a + 1) := by
  /-
    a b : Nat
    ha : LE.le 2 a
    hb : LT.lt 2 b
    ⊢ LE.le 2 (HDiv.hDiv (HAdd.hAdd (HPow.hPow a b) 1) (HAdd.hAdd a 1))
  -/
  rw [Nat.le_div_iff_mul_le (Nat.zero_lt_succ _)]
  /-
    a b : Nat
    ha : LE.le 2 a
    hb : LT.lt 2 b
    ⊢ LE.le (HMul.hMul 2 a.succ) (HAdd.hAdd (HPow.hPow a b) 1)
  -/
  apply Nat.succ_le_succ
  calc
    2 * a + 1 ≤ a ^ 2 * a := by nlinarith
    _ = a ^ 3 := by rw [Nat.pow_succ a 2]
    _ ≤ a ^ b := pow_right_mono₀ (Nat.le_of_succ_le ha) hb


private theorem AB_id_helper (b p : ℕ) (_ : 2 ≤ b) (hp : Odd p) :
    (b ^ p - 1) / (b - 1) * ((b ^ p + 1) / (b + 1)) = (b ^ (2 * p) - 1) / (b ^ 2 - 1) := by
  /-
    b p : Nat
    x✝ : LE.le 2 b
    hp : Odd p
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)) (HDi …
  -/
  have q₁ : b - 1 ∣ b ^ p - 1 := by simpa only [one_pow] using nat_sub_dvd_pow_sub_pow b 1 p
  /-
    b p : Nat
    x✝ : LE.le 2 b
    hp : Odd p
    q₁ : Dvd.dvd (HSub.hSub b 1) (HSub.hSub (HPow.hPow b p) 1)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)) (HDi …
  -/
  have q₂ : b + 1 ∣ b ^ p + 1 := by simpa only [one_pow] using hp.nat_add_dvd_pow_add_pow b 1
  /-
    b p : Nat
    x✝ : LE.le 2 b
    hp : Odd p
    q₁ : Dvd.dvd (HSub.hSub b 1) (HSub.hSub (HPow.hPow b p) 1)
    q₂ : Dvd.dvd (HAdd.hAdd b 1) (HAdd.hAdd (HPow.hPow b p) 1)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)) (HDi …
  -/
  convert Nat.div_mul_div_comm q₁ q₂ using 2 <;> rw [mul_comm (_ - 1), ← Nat.sq_sub_sq]
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case h.e'_3.h.e'_5
    b p : Nat
    x✝ : LE.le 2 b
    hp : Odd p
    q₁ : Dvd.dvd (HSub.hSub b 1) (HSub.hSub (HPow.hPow b p) 1)
    q₂ : Dvd.dvd (HAdd.hAdd b 1) (HAdd.hAdd (HPow.hPow b p) 1)
    ⊢ Eq (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) 1) (HSub.hSub (HPow.hPow (HPow.h …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


/-- Used in the proof of `psp_from_prime_psp`
-/
private theorem bp_helper {b p : ℕ} (hb : 0 < b) (hp : 1 ≤ p) :
    b ^ (2 * p) - 1 - (b ^ 2 - 1) = b * (b ^ (p - 1) - 1) * (b ^ p + b) :=
  have hi_bsquared : 1 ≤ b ^ 2 := Nat.one_le_pow _ _ hb
  calc
                                                                          /-
                                                                            b p : Nat
                                                                            hb : LT.lt 0 b
                                                                            hp : LE.le 1 p
                                                                            hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                                                            ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) 1) (HSub.hSub (HPow.h …
                                                                          -/
    b ^ (2 * p) - 1 - (b ^ 2 - 1) = b ^ (2 * p) - (1 + (b ^ 2 - 1)) := by rw [Nat.sub_sub]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                            /-
                                              b p : Nat
                                              hb : LT.lt 0 b
                                              hp : LE.le 1 p
                                              hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                              ⊢ Eq (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) (HAdd.hAdd 1 (HSub.hSub (HPow.hP …
                                            -/
    _ = b ^ (2 * p) - (1 + b ^ 2 - 1) := by rw [Nat.add_sub_assoc hi_bsquared]
                                            /-
                                              🎉 no goals
                                            -/
                                  /-
                                    b p : Nat
                                    hb : LT.lt 0 b
                                    hp : LE.le 1 p
                                    hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                    ⊢ Eq (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) (HSub.hSub (HAdd.hAdd 1 (HPow.hP …
                                  -/
    _ = b ^ (2 * p) - b ^ 2 := by rw [Nat.add_sub_cancel_left]
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    b p : Nat
                                    hb : LT.lt 0 b
                                    hp : LE.le 1 p
                                    hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                    ⊢ Eq (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) (HPow.hPow b 2)) (HSub.hSub (HPo …
                                  -/
    _ = b ^ (p * 2) - b ^ 2 := by rw [mul_comm]
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    b p : Nat
                                    hb : LT.lt 0 b
                                    hp : LE.le 1 p
                                    hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                    ⊢ Eq (HSub.hSub (HPow.hPow b (HMul.hMul p 2)) (HPow.hPow b 2)) (HSub.hSub (HPo …
                                  -/
    _ = (b ^ p) ^ 2 - b ^ 2 := by rw [pow_mul]
                                  /-
                                    🎉 no goals
                                  -/
                                        /-
                                          b p : Nat
                                          hb : LT.lt 0 b
                                          hp : LE.le 1 p
                                          hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                          ⊢ Eq (HSub.hSub (HPow.hPow (HPow.hPow b p) 2) (HPow.hPow b 2)) (HMul.hMul (HAd …
                                        -/
    _ = (b ^ p + b) * (b ^ p - b) := by rw [Nat.sq_sub_sq]
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          b p : Nat
                                          hb : LT.lt 0 b
                                          hp : LE.le 1 p
                                          hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                          ⊢ Eq (HMul.hMul (HAdd.hAdd (HPow.hPow b p) b) (HSub.hSub (HPow.hPow b p) b)) ( …
                                        -/
    _ = (b ^ p - b) * (b ^ p + b) := by rw [mul_comm]
                                        /-
                                          🎉 no goals
                                        -/
                                                  /-
                                                    b p : Nat
                                                    hb : LT.lt 0 b
                                                    hp : LE.le 1 p
                                                    hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                                    ⊢ Eq (HMul.hMul (HSub.hSub (HPow.hPow b p) b) (HAdd.hAdd (HPow.hPow b p) b)) ( …
                                                  -/
    _ = (b ^ (p - 1 + 1) - b) * (b ^ p + b) := by rw [Nat.sub_add_cancel hp]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    b p : Nat
                                                    hb : LT.lt 0 b
                                                    hp : LE.le 1 p
                                                    hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                                    ⊢ Eq (HMul.hMul (HSub.hSub (HPow.hPow b (HAdd.hAdd (HSub.hSub p 1) 1)) b) (HAd …
                                                  -/
    _ = (b * b ^ (p - 1) - b) * (b ^ p + b) := by rw [pow_succ']
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                      /-
                                                        b p : Nat
                                                        hb : LT.lt 0 b
                                                        hp : LE.le 1 p
                                                        hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                                        ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul b (HPow.hPow b (HSub.hSub p 1))) b) (HAd …
                                                      -/
    _ = (b * b ^ (p - 1) - b * 1) * (b ^ p + b) := by rw [mul_one]
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                  /-
                                                    b p : Nat
                                                    hb : LT.lt 0 b
                                                    hp : LE.le 1 p
                                                    hi_bsquared : LE.le 1 (HPow.hPow b 2)
                                                    ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul b (HPow.hPow b (HSub.hSub p 1))) (HMul.h …
                                                  -/
    _ = b * (b ^ (p - 1) - 1) * (b ^ p + b) := by rw [Nat.mul_sub_left_distrib]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- Given a prime `p` which does not divide `b * (b ^ 2 - 1)`, we can produce a number `n` which is
larger than `p` and pseudoprime to base `b`. We do this by defining
`n = ((b ^ p - 1) / (b - 1)) * ((b ^ p + 1) / (b + 1))`

The primary purpose of this definition is to help prove `exists_infinite_pseudoprimes`. For a proof
that `n` is actually pseudoprime to base `b`, see `psp_from_prime_psp`, and for a proof that `n` is
greater than `p`, see `psp_from_prime_gt_p`.

This lemma is intended to be used when `2 ≤ b`, `2 < p`, `p` is prime, and `¬p ∣ b * (b ^ 2 - 1)`,
because those are the hypotheses for `psp_from_prime_psp`.
-/
private def psp_from_prime (b : ℕ) (p : ℕ) : ℕ :=
  (b ^ p - 1) / (b - 1) * ((b ^ p + 1) / (b + 1))


/--
This is a proof that the number produced using `psp_from_prime` is actually pseudoprime to base `b`.
The primary purpose of this lemma is to help prove `exists_infinite_pseudoprimes`.

We use <https://primes.utm.edu/notes/proofs/a_pseudoprimes.html> as a rough outline of the proof.
-/
private theorem psp_from_prime_psp {b : ℕ} (b_ge_two : 2 ≤ b) {p : ℕ} (p_prime : p.Prime)
    (p_gt_two : 2 < p) (not_dvd : ¬p ∣ b * (b ^ 2 - 1)) : FermatPsp (psp_from_prime b p) b := by
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    ⊢ (Nat.psp_from_prime b p).FermatPsp b
  -/
  unfold psp_from_prime
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    ⊢ (HMul.hMul (HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)) (HDiv.h …
  -/
  set A := (b ^ p - 1) / (b - 1)
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    ⊢ (HMul.hMul A (HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1))).Ferm …
  -/
  set B := (b ^ p + 1) / (b + 1)
  -- Inequalities
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have hi_A : 1 < A := a_id_helper (Nat.succ_le_iff.mp b_ge_two) (Nat.Prime.one_lt p_prime)
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have hi_B : 1 < B := b_id_helper (Nat.succ_le_iff.mp b_ge_two) p_gt_two
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have hi_AB : 1 < A * B := one_lt_mul'' hi_A hi_B
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have hi_b : 0 < b := by omega
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have hi_p : 1 ≤ p := Nat.one_le_of_lt p_gt_two
  have hi_bsquared : 0 < b ^ 2 - 1 := by
    -- Porting note: was `by nlinarith [Nat.one_le_pow 2 b hi_b]`
    have := Nat.pow_le_pow_left b_ge_two 2
    omega
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    hi_p : LE.le 1 p
    hi_bsquared : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have hi_bpowtwop : 1 ≤ b ^ (2 * p) := Nat.one_le_pow (2 * p) b hi_b
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    hi_p : LE.le 1 p
    hi_bsquared : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
    hi_bpowtwop : LE.le 1 (HPow.hPow b (HMul.hMul 2 p))
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have hi_bpowpsubone : 1 ≤ b ^ (p - 1) := Nat.one_le_pow (p - 1) b hi_b
  -- Other useful facts
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    hi_p : LE.le 1 p
    hi_bsquared : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
    hi_bpowtwop : LE.le 1 (HPow.hPow b (HMul.hMul 2 p))
    hi_bpowpsubone : LE.le 1 (HPow.hPow b (HSub.hSub p 1))
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have p_odd : Odd p := p_prime.odd_of_ne_two p_gt_two.ne.symm
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    hi_p : LE.le 1 p
    hi_bsquared : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
    hi_bpowtwop : LE.le 1 (HPow.hPow b (HMul.hMul 2 p))
    hi_bpowpsubone : LE.le 1 (HPow.hPow b (HSub.hSub p 1))
    p_odd : Odd p
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have AB_not_prime : ¬Nat.Prime (A * B) := Nat.not_prime_mul hi_A.ne' hi_B.ne'
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    hi_p : LE.le 1 p
    hi_bsquared : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
    hi_bpowtwop : LE.le 1 (HPow.hPow b (HMul.hMul 2 p))
    hi_bpowpsubone : LE.le 1 (HPow.hPow b (HSub.hSub p 1))
    p_odd : Odd p
    AB_not_prime : Not (Nat.Prime (HMul.hMul A B))
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  have AB_id : A * B = (b ^ (2 * p) - 1) / (b ^ 2 - 1) := AB_id_helper _ _ b_ge_two p_odd
  have hd : b ^ 2 - 1 ∣ b ^ (2 * p) - 1 := by
    simpa only [one_pow, pow_mul] using nat_sub_dvd_pow_sub_pow _ 1 p
  -- We know that `A * B` is not prime, and that `1 < A * B`. Since two conditions of being
  -- pseudoprime are satisfied, we only need to show that `A * B` is probable prime to base `b`
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    hi_p : LE.le 1 p
    hi_bsquared : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
    hi_bpowtwop : LE.le 1 (HPow.hPow b (HMul.hMul 2 p))
    hi_bpowpsubone : LE.le 1 (HPow.hPow b (HSub.hSub p 1))
    p_odd : Odd p
    AB_not_prime : Not (Nat.Prime (HMul.hMul A B))
    AB_id : Eq (HMul.hMul A B) (HDiv.hDiv (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) …
    hd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.hMul  …
    ⊢ (HMul.hMul A B).FermatPsp b
  -/
  refine ⟨?_, AB_not_prime, hi_AB⟩
  -- Used to prove that `2 * p * (b ^ 2 - 1) ∣ (b ^ 2 - 1) * (A * B - 1)`.
  have ha₁ : (b ^ 2 - 1) * (A * B - 1) = b * (b ^ (p - 1) - 1) * (b ^ p + b) := by
    apply_fun fun x => x * (b ^ 2 - 1) at AB_id
    rw [Nat.div_mul_cancel hd] at AB_id
    apply_fun fun x => x - (b ^ 2 - 1) at AB_id
    nth_rw 2 [← one_mul (b ^ 2 - 1)] at AB_id
    rw [← Nat.mul_sub_right_distrib, mul_comm] at AB_id
    rw [AB_id]
    exact bp_helper hi_b hi_p
  -- If `b` is even, then `b^p` is also even, so `2 ∣ b^p + b`
  -- If `b` is odd, then `b^p` is also odd, so `2 ∣ b^p + b`
  have ha₂ : 2 ∣ b ^ p + b := by
    -- Porting note: golfed
    rw [← even_iff_two_dvd, Nat.even_add, Nat.even_pow' p_prime.ne_zero]
  -- Since `b` isn't divisible by `p`, `b` is coprime with `p`. we can use Fermat's Little Theorem
  -- to prove this.
  have ha₃ : p ∣ b ^ (p - 1) - 1 := by
    have : ¬p ∣ b := mt (fun h : p ∣ b => dvd_mul_of_dvd_left h _) not_dvd
    have : p.Coprime b := Or.resolve_right (Nat.coprime_or_dvd_of_prime p_prime b) this
    have : IsCoprime (b : ℤ) ↑p := this.symm.isCoprime
    have : ↑b ^ (p - 1) ≡ 1 [ZMOD ↑p] := Int.ModEq.pow_card_sub_one_eq_one p_prime this
    have : ↑p ∣ ↑b ^ (p - 1) - ↑1 := mod_cast Int.ModEq.dvd (Int.ModEq.symm this)
    exact mod_cast this
  -- Because `p - 1` is even, there is a `c` such that `2 * c = p - 1`. `nat_sub_dvd_pow_sub_pow`
  -- implies that `b ^ c - 1 ∣ (b ^ c) ^ 2 - 1`, and `(b ^ c) ^ 2 = b ^ (p - 1)`.
  have ha₄ : b ^ 2 - 1 ∣ b ^ (p - 1) - 1 := by
    cases' p_odd with k hk
    have : 2 ∣ p - 1 := ⟨k, by simp [hk]⟩
    cases' this with c hc
    have : b ^ 2 - 1 ∣ (b ^ 2) ^ c - 1 := by
      simpa only [one_pow] using nat_sub_dvd_pow_sub_pow _ 1 c
    have : b ^ 2 - 1 ∣ b ^ (2 * c) - 1 := by rwa [← pow_mul] at this
    rwa [← hc] at this
  -- Used to prove that `2 * p` divides `A * B - 1`
  have ha₅ : 2 * p * (b ^ 2 - 1) ∣ (b ^ 2 - 1) * (A * B - 1) := by
    suffices q : 2 * p * (b ^ 2 - 1) ∣ b * (b ^ (p - 1) - 1) * (b ^ p + b) by rwa [ha₁]
    -- We already proved that `b ^ 2 - 1 ∣ b ^ (p - 1) - 1`.
    -- Since `2 ∣ b ^ p + b` and `p ∣ b ^ p + b`, if we show that 2 and p are coprime, then we
    -- know that `2 * p ∣ b ^ p + b`
    have q₁ : Nat.Coprime p (b ^ 2 - 1) :=
      haveI q₂ : ¬p ∣ b ^ 2 - 1 := by
        rw [mul_comm] at not_dvd
        exact mt (fun h : p ∣ b ^ 2 - 1 => dvd_mul_of_dvd_left h _) not_dvd
      (Nat.Prime.coprime_iff_not_dvd p_prime).mpr q₂
    have q₂ : p * (b ^ 2 - 1) ∣ b ^ (p - 1) - 1 := Nat.Coprime.mul_dvd_of_dvd_of_dvd q₁ ha₃ ha₄
    have q₃ : p * (b ^ 2 - 1) * 2 ∣ (b ^ (p - 1) - 1) * (b ^ p + b) := mul_dvd_mul q₂ ha₂
    have q₄ : p * (b ^ 2 - 1) * 2 ∣ b * ((b ^ (p - 1) - 1) * (b ^ p + b)) :=
      dvd_mul_of_dvd_right q₃ _
    rwa [mul_assoc, mul_comm, mul_assoc b]
  have ha₆ : 2 * p ∣ A * B - 1 := by
    rw [mul_comm] at ha₅
    exact Nat.dvd_of_mul_dvd_mul_left hi_bsquared ha₅
  -- `A * B` divides `b ^ (2 * p) - 1` because `A * B * (b ^ 2 - 1) = b ^ (2 * p) - 1`.
  -- This can be proven by multiplying both sides of `AB_id` by `b ^ 2 - 1`.
  have ha₇ : A * B ∣ b ^ (2 * p) - 1 := by
    use b ^ 2 - 1
    have : A * B * (b ^ 2 - 1) = (b ^ (2 * p) - 1) / (b ^ 2 - 1) * (b ^ 2 - 1) :=
      congr_arg (fun x : ℕ => x * (b ^ 2 - 1)) AB_id
    simpa only [add_comm, Nat.div_mul_cancel hd, Nat.sub_add_cancel hi_bpowtwop] using this.symm
  -- Since `2 * p ∣ A * B - 1`, there is a number `q` such that `2 * p * q = A * B - 1`.
  -- By `nat_sub_dvd_pow_sub_pow`, we know that `b ^ (2 * p) - 1 ∣ b ^ (2 * p * q) - 1`.
  -- This means that `b ^ (2 * p) - 1 ∣ b ^ (A * B - 1) - 1`.
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    hi_p : LE.le 1 p
    hi_bsquared : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
    hi_bpowtwop : LE.le 1 (HPow.hPow b (HMul.hMul 2 p))
    hi_bpowpsubone : LE.le 1 (HPow.hPow b (HSub.hSub p 1))
    p_odd : Odd p
    AB_not_prime : Not (Nat.Prime (HMul.hMul A B))
    AB_id : Eq (HMul.hMul A B) (HDiv.hDiv (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) …
    hd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.hMul  …
    ha₁ : Eq (HMul.hMul (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HMul.hMul A B) 1 …
    ha₂ : Dvd.dvd 2 (HAdd.hAdd (HPow.hPow b p) b)
    ha₃ : Dvd.dvd p (HSub.hSub (HPow.hPow b (HSub.hSub p 1)) 1)
    ha₄ : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HSub.hSub …
    ha₅ : Dvd.dvd (HMul.hMul (HMul.hMul 2 p) (HSub.hSub (HPow.hPow b 2) 1)) (HMul. …
    ha₆ : Dvd.dvd (HMul.hMul 2 p) (HSub.hSub (HMul.hMul A B) 1)
    ha₇ : Dvd.dvd (HMul.hMul A B) (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) 1)
    ⊢ (HMul.hMul A B).ProbablePrime b
  -/
  cases' ha₆ with q hq
  have ha₈ : b ^ (2 * p) - 1 ∣ b ^ (A * B - 1) - 1 := by
    simpa only [one_pow, pow_mul, hq] using nat_sub_dvd_pow_sub_pow _ 1 q
  -- We have proved that `A * B ∣ b ^ (2 * p) - 1` and `b ^ (2 * p) - 1 ∣ b ^ (A * B - 1) - 1`.
  -- Therefore, `A * B ∣ b ^ (A * B - 1) - 1`.
  /-
    case intro
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    not_dvd : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    hi_A : LT.lt 1 A
    hi_B : LT.lt 1 B
    hi_AB : LT.lt 1 (HMul.hMul A B)
    hi_b : LT.lt 0 b
    hi_p : LE.le 1 p
    hi_bsquared : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
    hi_bpowtwop : LE.le 1 (HPow.hPow b (HMul.hMul 2 p))
    hi_bpowpsubone : LE.le 1 (HPow.hPow b (HSub.hSub p 1))
    p_odd : Odd p
    AB_not_prime : Not (Nat.Prime (HMul.hMul A B))
    AB_id : Eq (HMul.hMul A B) (HDiv.hDiv (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) …
    hd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.hMul  …
    ha₁ : Eq (HMul.hMul (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HMul.hMul A B) 1 …
    ha₂ : Dvd.dvd 2 (HAdd.hAdd (HPow.hPow b p) b)
    ha₃ : Dvd.dvd p (HSub.hSub (HPow.hPow b (HSub.hSub p 1)) 1)
    ha₄ : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HSub.hSub …
    ha₅ : Dvd.dvd (HMul.hMul (HMul.hMul 2 p) (HSub.hSub (HPow.hPow b 2) 1)) (HMul. …
    ha₇ : Dvd.dvd (HMul.hMul A B) (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) 1)
    q : Nat
    hq : Eq (HSub.hSub (HMul.hMul A B) 1) (HMul.hMul (HMul.hMul 2 p) q)
    ha₈ : Dvd.dvd (HSub.hSub (HPow.hPow b (HMul.hMul 2 p)) 1) (HSub.hSub (HPow.hPo …
    ⊢ (HMul.hMul A B).ProbablePrime b
  -/
  exact dvd_trans ha₇ ha₈
  /-
    🎉 no goals
  -/


/--
This is a proof that the number produced using `psp_from_prime` is greater than the prime `p` used
to create it. The primary purpose of this lemma is to help prove `exists_infinite_pseudoprimes`.
-/
private theorem psp_from_prime_gt_p {b : ℕ} (b_ge_two : 2 ≤ b) {p : ℕ} (p_prime : p.Prime)
    (p_gt_two : 2 < p) : p < psp_from_prime b p := by
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    ⊢ LT.lt p (Nat.psp_from_prime b p)
  -/
  unfold psp_from_prime
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    ⊢ LT.lt p (HMul.hMul (HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)) …
  -/
  set A := (b ^ p - 1) / (b - 1)
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    ⊢ LT.lt p (HMul.hMul A (HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1 …
  -/
  set B := (b ^ p + 1) / (b + 1)
  rw [show A * B = (b ^ (2 * p) - 1) / (b ^ 2 - 1) from
      AB_id_helper _ _ b_ge_two (p_prime.odd_of_ne_two p_gt_two.ne.symm)]
  have AB_dvd : b ^ 2 - 1 ∣ b ^ (2 * p) - 1 := by
    simpa only [one_pow, pow_mul] using nat_sub_dvd_pow_sub_pow _ 1 p
  suffices h : p * (b ^ 2 - 1) < b ^ (2 * p) - 1 by
    have h₁ : p * (b ^ 2 - 1) / (b ^ 2 - 1) < (b ^ (2 * p) - 1) / (b ^ 2 - 1) :=
      Nat.div_lt_div_of_lt_of_dvd AB_dvd h
    have h₂ : 0 < b ^ 2 - 1 := by
      linarith [show 3 ≤ b ^ 2 - 1 from le_tsub_of_add_le_left (show 4 ≤ b ^ 2 by nlinarith)]
    rwa [Nat.mul_div_cancel _ h₂] at h₁
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    AB_dvd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.h …
    ⊢ LT.lt (HMul.hMul p (HSub.hSub (HPow.hPow b 2) 1)) (HSub.hSub (HPow.hPow b (H …
  -/
  rw [Nat.mul_sub_left_distrib, mul_one, pow_mul]
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    AB_dvd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.h …
    ⊢ LT.lt (HSub.hSub (HMul.hMul p (HPow.hPow b 2)) p) (HSub.hSub (HPow.hPow (HPo …
  -/
  conv_rhs => rw [← Nat.sub_add_cancel (show 1 ≤ p by omega)]
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    AB_dvd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.h …
    ⊢ LT.lt (HSub.hSub (HMul.hMul p (HPow.hPow b 2)) p) (HSub.hSub (HPow.hPow (HPo …
  -/
  rw [Nat.pow_succ (b ^ 2)]
  suffices h : p * b ^ 2 < (b ^ 2) ^ (p - 1) * b ^ 2 by
    apply gt_of_ge_of_gt
    · exact tsub_le_tsub_left (one_le_of_lt p_gt_two) ((b ^ 2) ^ (p - 1) * b ^ 2)
    · have : p ≤ p * b ^ 2 := Nat.le_mul_of_pos_right _ (show 0 < b ^ 2 by positivity)
      exact tsub_lt_tsub_right_of_le this h
  suffices h : p < (b ^ 2) ^ (p - 1) by
    have : 4 ≤ b ^ 2 := by nlinarith
    have : 0 < b ^ 2 := by omega
    exact mul_lt_mul_of_pos_right h this
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    AB_dvd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.h …
    ⊢ LT.lt p (HPow.hPow (HPow.hPow b 2) (HSub.hSub p 1))
  -/
  rw [← pow_mul, Nat.mul_sub_left_distrib, mul_one]
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    AB_dvd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.h …
    ⊢ LT.lt p (HPow.hPow b (HSub.hSub (HMul.hMul 2 p) 2))
  -/
  have : 2 ≤ 2 * p - 2 := le_tsub_of_add_le_left (show 4 ≤ 2 * p by omega)
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    AB_dvd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.h …
    this : LE.le 2 (HSub.hSub (HMul.hMul 2 p) 2)
    ⊢ LT.lt p (HPow.hPow b (HSub.hSub (HMul.hMul 2 p) 2))
  -/
  have : 2 + p ≤ 2 * p := by omega
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    AB_dvd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.h …
    this✝ : LE.le 2 (HSub.hSub (HMul.hMul 2 p) 2)
    this : LE.le (HAdd.hAdd 2 p) (HMul.hMul 2 p)
    ⊢ LT.lt p (HPow.hPow b (HSub.hSub (HMul.hMul 2 p) 2))
  -/
  have : p ≤ 2 * p - 2 := le_tsub_of_add_le_left this
  /-
    b : Nat
    b_ge_two : LE.le 2 b
    p : Nat
    p_prime : Nat.Prime p
    p_gt_two : LT.lt 2 p
    A : Nat := HDiv.hDiv (HSub.hSub (HPow.hPow b p) 1) (HSub.hSub b 1)
    B : Nat := HDiv.hDiv (HAdd.hAdd (HPow.hPow b p) 1) (HAdd.hAdd b 1)
    AB_dvd : Dvd.dvd (HSub.hSub (HPow.hPow b 2) 1) (HSub.hSub (HPow.hPow b (HMul.h …
    this✝¹ : LE.le 2 (HSub.hSub (HMul.hMul 2 p) 2)
    this✝ : LE.le (HAdd.hAdd 2 p) (HMul.hMul 2 p)
    this : LE.le p (HSub.hSub (HMul.hMul 2 p) 2)
    ⊢ LT.lt p (HPow.hPow b (HSub.hSub (HMul.hMul 2 p) 2))
  -/
  exact this.trans_lt (Nat.lt_pow_self b_ge_two)
  /-
    🎉 no goals
  -/


/-- For all positive bases, there exist infinite **Fermat pseudoprimes** to that base.
Given in this form: for all numbers `b ≥ 1` and `m`, there exists a pseudoprime `n` to base `b` such
that `m ≤ n`. This form is similar to `Nat.exists_infinite_primes`.
-/
theorem exists_infinite_pseudoprimes {b : ℕ} (h : 1 ≤ b) (m : ℕ) :
    ∃ n : ℕ, FermatPsp n b ∧ m ≤ n := by
  /-
    b : Nat
    h : LE.le 1 b
    m : Nat
    ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
  -/
  by_cases b_ge_two : 2 ≤ b
  -- If `2 ≤ b`, then because there exist infinite prime numbers, there is a prime number p with
  -- `m ≤ p` and `¬p ∣ b*(b^2 - 1)`. We pick a prime number `b*(b^2 - 1) + 1 + m ≤ p` because we
  -- automatically know that `p` is greater than m and that it does not divide `b*(b^2 - 1)`
  -- (because `p` can't divide a number less than `p`).
  -- From `p`, we can use the lemmas we proved earlier to show that
  -- `((b^p - 1)/(b - 1)) * ((b^p + 1)/(b + 1))` is a pseudoprime to base `b`.
    /-
      case pos
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
  · have h := Nat.exists_infinite_primes (b * (b ^ 2 - 1) + 1 + m)
    /-
      case pos
      b : Nat
      h✝ : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      h : Exists fun p => And (LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub ( …
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    cases' h with p hp
    /-
      case pos.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp : And (LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2)  …
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    cases' hp with hp₁ hp₂
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₁ : 0 < b := pos_of_gt (Nat.succ_le_iff.mp b_ge_two)
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₂ : 4 ≤ b ^ 2 := pow_le_pow_left' b_ge_two 2
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₃ : 0 < b ^ 2 - 1 := tsub_pos_of_lt (gt_of_ge_of_gt h₂ (by norm_num))
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₄ : 0 < b * (b ^ 2 - 1) := mul_pos h₁ h₃
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₅ : b * (b ^ 2 - 1) < p := by omega
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₆ : ¬p ∣ b * (b ^ 2 - 1) := Nat.not_dvd_of_pos_of_lt h₄ h₅
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
      h₆ : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₇ : b ≤ b * (b ^ 2 - 1) := Nat.le_mul_of_pos_right _ h₃
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
      h₆ : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
      h₇ : LE.le b (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₈ : 2 ≤ b * (b ^ 2 - 1) := le_trans b_ge_two h₇
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
      h₆ : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
      h₇ : LE.le b (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₈ : LE.le 2 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₉ : 2 < p := gt_of_gt_of_ge h₅ h₈
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
      h₆ : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
      h₇ : LE.le b (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₈ : LE.le 2 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₉ : LT.lt 2 p
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    have h₁₀ := psp_from_prime_gt_p b_ge_two hp₂ h₉
    /-
      case pos.intro.intro
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
      h₆ : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
      h₇ : LE.le b (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₈ : LE.le 2 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₉ : LT.lt 2 p
      h₁₀ : LT.lt p (Nat.psp_from_prime b p)
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    use psp_from_prime b p
    /-
      case h
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : LE.le 2 b
      p : Nat
      hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
      hp₂ : Nat.Prime p
      h₁ : LT.lt 0 b
      h₂ : LE.le 4 (HPow.hPow b 2)
      h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
      h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
      h₆ : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
      h₇ : LE.le b (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₈ : LE.le 2 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
      h₉ : LT.lt 2 p
      h₁₀ : LT.lt p (Nat.psp_from_prime b p)
      ⊢ And ((Nat.psp_from_prime b p).FermatPsp b) (LE.le m (Nat.psp_from_prime b p))
    -/
    constructor
      /-
        case h.left
        b : Nat
        h : LE.le 1 b
        m : Nat
        b_ge_two : LE.le 2 b
        p : Nat
        hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
        hp₂ : Nat.Prime p
        h₁ : LT.lt 0 b
        h₂ : LE.le 4 (HPow.hPow b 2)
        h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
        h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
        h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
        h₆ : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
        h₇ : LE.le b (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
        h₈ : LE.le 2 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
        h₉ : LT.lt 2 p
        h₁₀ : LT.lt p (Nat.psp_from_prime b p)
        ⊢ (Nat.psp_from_prime b p).FermatPsp b
      -/
    · exact psp_from_prime_psp b_ge_two hp₂ h₉ h₆
      /-
        🎉 no goals
      -/
      /-
        case h.right
        b : Nat
        h : LE.le 1 b
        m : Nat
        b_ge_two : LE.le 2 b
        p : Nat
        hp₁ : LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))  …
        hp₂ : Nat.Prime p
        h₁ : LT.lt 0 b
        h₂ : LE.le 4 (HPow.hPow b 2)
        h₃ : LT.lt 0 (HSub.hSub (HPow.hPow b 2) 1)
        h₄ : LT.lt 0 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
        h₅ : LT.lt (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)) p
        h₆ : Not (Dvd.dvd p (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1)))
        h₇ : LE.le b (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
        h₈ : LE.le 2 (HMul.hMul b (HSub.hSub (HPow.hPow b 2) 1))
        h₉ : LT.lt 2 p
        h₁₀ : LT.lt p (Nat.psp_from_prime b p)
        ⊢ LE.le m (Nat.psp_from_prime b p)
      -/
    · exact le_trans (show m ≤ p by omega) (le_of_lt h₁₀)
      /-
        🎉 no goals
      -/
  -- If `¬2 ≤ b`, then `b = 1`. Since all composite numbers are pseudoprimes to base 1, we can pick
  -- any composite number greater than m. We choose `2 * (m + 2)` because it is greater than `m` and
  -- is composite for all natural numbers `m`.
    /-
      case neg
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : Not (LE.le 2 b)
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
  · have h₁ : b = 1 := by omega
    /-
      case neg
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : Not (LE.le 2 b)
      h₁ : Eq b 1
      ⊢ Exists fun n => And (n.FermatPsp b) (LE.le m n)
    -/
    rw [h₁]
    /-
      case neg
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : Not (LE.le 2 b)
      h₁ : Eq b 1
      ⊢ Exists fun n => And (n.FermatPsp 1) (LE.le m n)
    -/
    use 2 * (m + 2)
    /-
      case h
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : Not (LE.le 2 b)
      h₁ : Eq b 1
      ⊢ And ((HMul.hMul 2 (HAdd.hAdd m 2)).FermatPsp 1) (LE.le m (HMul.hMul 2 (HAdd. …
    -/
    have : ¬Nat.Prime (2 * (m + 2)) := Nat.not_prime_mul (by omega) (by omega)
    /-
      case h
      b : Nat
      h : LE.le 1 b
      m : Nat
      b_ge_two : Not (LE.le 2 b)
      h₁ : Eq b 1
      this : Not (Nat.Prime (HMul.hMul 2 (HAdd.hAdd m 2)))
      ⊢ And ((HMul.hMul 2 (HAdd.hAdd m 2)).FermatPsp 1) (LE.le m (HMul.hMul 2 (HAdd. …
    -/
    exact ⟨fermatPsp_base_one (by omega) this, by omega⟩
    /-
      🎉 no goals
    -/


theorem frequently_atTop_fermatPsp {b : ℕ} (h : 1 ≤ b) : ∃ᶠ n in Filter.atTop, FermatPsp n b := by
  -- Based on the proof of `Nat.frequently_atTop_modEq_one`
  /-
    b : Nat
    h : LE.le 1 b
    ⊢ Filter.Frequently (fun n => n.FermatPsp b) Filter.atTop
  -/
  refine Filter.frequently_atTop.2 fun n => ?_
  /-
    b : Nat
    h : LE.le 1 b
    n : Nat
    ⊢ Exists fun b_1 => And (GE.ge b_1 n) (b_1.FermatPsp b)
  -/
  obtain ⟨p, hp⟩ := exists_infinite_pseudoprimes h n
  /-
    case intro
    b : Nat
    h : LE.le 1 b
    n p : Nat
    hp : And (p.FermatPsp b) (LE.le n p)
    ⊢ Exists fun b_1 => And (GE.ge b_1 n) (b_1.FermatPsp b)
  -/
  exact ⟨p, hp.2, hp.1⟩
  /-
    🎉 no goals
  -/


/-- Infinite set variant of `Nat.exists_infinite_pseudoprimes`
-/
theorem infinite_setOf_pseudoprimes {b : ℕ} (h : 1 ≤ b) :
    Set.Infinite { n : ℕ | FermatPsp n b } :=
  Nat.frequently_atTop_iff_infinite.mp (frequently_atTop_fermatPsp h)


