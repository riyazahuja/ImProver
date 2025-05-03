/-- If `q ≠ 0`, the `p`-adic norm of a rational `q` is `p ^ (-padicValRat p q)`.
If `q = 0`, the `p`-adic norm of `q` is `0`. -/
def padicNorm (p : ℕ) (q : ℚ) : ℚ :=
  if q = 0 then 0 else (p : ℚ) ^ (-padicValRat p q)


/-- Unfolds the definition of the `p`-adic norm of `q` when `q ≠ 0`. -/
@[simp]
protected theorem eq_zpow_of_nonzero {q : ℚ} (hq : q ≠ 0) :
                                                       /-
                                                         p : Nat
                                                         q : Rat
                                                         hq : Ne q 0
                                                         ⊢ Eq (padicNorm p q) (HPow.hPow (↑p) (Neg.neg (padicValRat p q)))
                                                       -/
    padicNorm p q = (p : ℚ) ^ (-padicValRat p q) := by simp [hq, padicNorm]
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- The `p`-adic norm is nonnegative. -/
protected theorem nonneg (q : ℚ) : 0 ≤ padicNorm p q :=
                        /-
                          p : Nat
                          q : Rat
                          hq : Eq q 0
                          ⊢ LE.le 0 (padicNorm p q)
                        -/
  if hq : q = 0 then by simp [hq, padicNorm]
                        /-
                          🎉 no goals
                        -/
  else by
    /-
      p : Nat
      q : Rat
      hq : Not (Eq q 0)
      ⊢ LE.le 0 (padicNorm p q)
    -/
    unfold padicNorm
    /-
      p : Nat
      q : Rat
      hq : Not (Eq q 0)
      ⊢ LE.le 0 (ite (Eq q 0) 0 (HPow.hPow (↑p) (Neg.neg (padicValRat p q))))
    -/
    split_ifs
    /-
      p : Nat
      q : Rat
      hq : Not (Eq q 0)
      ⊢ LE.le 0 (HPow.hPow (↑p) (Neg.neg (padicValRat p q)))
    -/
    apply zpow_nonneg
    /-
      case ha
      p : Nat
      q : Rat
      hq : Not (Eq q 0)
      ⊢ LE.le 0 ↑p
    -/
    exact mod_cast Nat.zero_le _
    /-
      🎉 no goals
    -/


/-- The `p`-adic norm of `0` is `0`. -/
@[simp]
                                                 /-
                                                   p : Nat
                                                   ⊢ Eq (padicNorm p 0) 0
                                                 -/
protected theorem zero : padicNorm p 0 = 0 := by simp [padicNorm]
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The `p`-adic norm of `1` is `1`. -/
                                                /-
                                                  p : Nat
                                                  ⊢ Eq (padicNorm p 1) 1
                                                -/
protected theorem one : padicNorm p 1 = 1 := by simp [padicNorm]
                                                /-
                                                  🎉 no goals
                                                -/


/-- The `p`-adic norm of `p` is `p⁻¹` if `p > 1`.

See also `padicNorm.padicNorm_p_of_prime` for a version assuming `p` is prime. -/
theorem padicNorm_p (hp : 1 < p) : padicNorm p p = (p : ℚ)⁻¹ := by
  /-
    p : Nat
    hp : LT.lt 1 p
    ⊢ Eq (padicNorm p ↑p) (Inv.inv ↑p)
  -/
  simp [padicNorm, (pos_of_gt hp).ne', padicValNat.self hp]
  /-
    🎉 no goals
  -/


/-- The `p`-adic norm of `p` is `p⁻¹` if `p` is prime.

See also `padicNorm.padicNorm_p` for a version assuming `1 < p`. -/
@[simp]
theorem padicNorm_p_of_prime [Fact p.Prime] : padicNorm p p = (p : ℚ)⁻¹ :=
  padicNorm_p <| Nat.Prime.one_lt Fact.out


/-- The `p`-adic norm of `q` is `1` if `q` is prime and not equal to `p`. -/
theorem padicNorm_of_prime_of_ne {q : ℕ} [p_prime : Fact p.Prime] [q_prime : Fact q.Prime]
    (neq : p ≠ q) : padicNorm p q = 1 := by
  /-
    p q : Nat
    p_prime : Fact (Nat.Prime p)
    q_prime : Fact (Nat.Prime q)
    neq : Ne p q
    ⊢ Eq (padicNorm p ↑q) 1
  -/
  have p : padicValRat p q = 0 := mod_cast padicValNat_primes neq
  /-
    p✝ q : Nat
    p_prime : Fact (Nat.Prime p✝)
    q_prime : Fact (Nat.Prime q)
    neq : Ne p✝ q
    p : Eq (padicValRat p✝ ↑q) 0
    ⊢ Eq (padicNorm p✝ ↑q) 1
  -/
  rw [padicNorm, p]
  /-
    p✝ q : Nat
    p_prime : Fact (Nat.Prime p✝)
    q_prime : Fact (Nat.Prime q)
    neq : Ne p✝ q
    p : Eq (padicValRat p✝ ↑q) 0
    ⊢ Eq (ite (Eq (↑q) 0) 0 (HPow.hPow (↑p✝) (-0))) 1
  -/
  simp [q_prime.1.ne_zero]
  /-
    🎉 no goals
  -/


/-- The `p`-adic norm of `p` is less than `1` if `1 < p`.

See also `padicNorm.padicNorm_p_lt_one_of_prime` for a version assuming `p` is prime. -/
theorem padicNorm_p_lt_one (hp : 1 < p) : padicNorm p p < 1 := by
  /-
    p : Nat
    hp : LT.lt 1 p
    ⊢ LT.lt (padicNorm p ↑p) 1
  -/
  rw [padicNorm_p hp, inv_lt_one_iff₀]
  /-
    p : Nat
    hp : LT.lt 1 p
    ⊢ Or (LE.le (↑p) 0) (LT.lt 1 ↑p)
  -/
  exact mod_cast Or.inr hp
  /-
    🎉 no goals
  -/


/-- The `p`-adic norm of `p` is less than `1` if `p` is prime.

See also `padicNorm.padicNorm_p_lt_one` for a version assuming `1 < p`. -/
theorem padicNorm_p_lt_one_of_prime [Fact p.Prime] : padicNorm p p < 1 :=
  padicNorm_p_lt_one <| Nat.Prime.one_lt Fact.out


/-- `padicNorm p q` takes discrete values `p ^ -z` for `z : ℤ`. -/
protected theorem values_discrete {q : ℚ} (hq : q ≠ 0) : ∃ z : ℤ, padicNorm p q = (p : ℚ) ^ (-z) :=
                       /-
                         p : Nat
                         q : Rat
                         hq : Ne q 0
                         ⊢ Eq (padicNorm p q) (HPow.hPow (↑p) (Neg.neg (padicValRat p q)))
                       -/
  ⟨padicValRat p q, by simp [padicNorm, hq]⟩
                       /-
                         🎉 no goals
                       -/


/-- `padicNorm p` is symmetric. -/
@[simp]
protected theorem neg (q : ℚ) : padicNorm p (-q) = padicNorm p q :=
                        /-
                          p : Nat
                          q : Rat
                          hq : Eq q 0
                          ⊢ Eq (padicNorm p (Neg.neg q)) (padicNorm p q)
                        -/
                        /-
                          🎉 no goals
                        -/
  if hq : q = 0 then by simp [hq] else by simp [padicNorm, hq]
                                          /-
                                            🎉 no goals
                                          -/


/-- If `q ≠ 0`, then `padicNorm p q ≠ 0`. -/
protected theorem nonzero {q : ℚ} (hq : q ≠ 0) : padicNorm p q ≠ 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    hq : Ne q 0
    ⊢ Ne (padicNorm p q) 0
  -/
  rw [padicNorm.eq_zpow_of_nonzero hq]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    hq : Ne q 0
    ⊢ Ne (HPow.hPow (↑p) (Neg.neg (padicValRat p q))) 0
  -/
  apply zpow_ne_zero
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    hq : Ne q 0
    ⊢ Ne (↑p) 0
  -/
  exact mod_cast ne_of_gt hp.1.pos
  /-
    🎉 no goals
  -/


/-- If the `p`-adic norm of `q` is 0, then `q` is `0`. -/
theorem zero_of_padicNorm_eq_zero {q : ℚ} (h : padicNorm p q = 0) : q = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    h : Eq (padicNorm p q) 0
    ⊢ Eq q 0
  -/
  apply by_contradiction; intro hq
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    h : Eq (padicNorm p q) 0
    hq : Not (Eq q 0)
    ⊢ False
  -/
  unfold padicNorm at h; rw [if_neg hq] at h
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    h : Eq (HPow.hPow (↑p) (Neg.neg (padicValRat p q))) 0
    hq : Not (Eq q 0)
    ⊢ False
  -/
  apply absurd h
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    h : Eq (HPow.hPow (↑p) (Neg.neg (padicValRat p q))) 0
    hq : Not (Eq q 0)
    ⊢ Not (Eq (HPow.hPow (↑p) (Neg.neg (padicValRat p q))) 0)
  -/
  apply zpow_ne_zero
  /-
    case a.a
    p : Nat
    hp : Fact (Nat.Prime p)
    q : Rat
    h : Eq (HPow.hPow (↑p) (Neg.neg (padicValRat p q))) 0
    hq : Not (Eq q 0)
    ⊢ Ne (↑p) 0
  -/
  exact mod_cast hp.1.ne_zero
  /-
    🎉 no goals
  -/


/-- The `p`-adic norm is multiplicative. -/
@[simp]
protected theorem mul (q r : ℚ) : padicNorm p (q * r) = padicNorm p q * padicNorm p r :=
                        /-
                          p : Nat
                          hp : Fact (Nat.Prime p)
                          q r : Rat
                          hq : Eq q 0
                          ⊢ Eq (padicNorm p (HMul.hMul q r)) (HMul.hMul (padicNorm p q) (padicNorm p r))
                        -/
  if hq : q = 0 then by simp [hq]
                        /-
                          🎉 no goals
                        -/
  else
                          /-
                            p : Nat
                            hp : Fact (Nat.Prime p)
                            q r : Rat
                            hq : Not (Eq q 0)
                            hr : Eq r 0
                            ⊢ Eq (padicNorm p (HMul.hMul q r)) (HMul.hMul (padicNorm p q) (padicNorm p r))
                          -/
    if hr : r = 0 then by simp [hr]
                          /-
                            🎉 no goals
                          -/
    else by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        ⊢ Eq (padicNorm p (HMul.hMul q r)) (HMul.hMul (padicNorm p q) (padicNorm p r))
      -/
      have : (p : ℚ) ≠ 0 := by simp [hp.1.ne_zero]
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        q r : Rat
        hq : Not (Eq q 0)
        hr : Not (Eq r 0)
        this : Ne (↑p) 0
        ⊢ Eq (padicNorm p (HMul.hMul q r)) (HMul.hMul (padicNorm p q) (padicNorm p r))
      -/
      simp [padicNorm, *, padicValRat.mul, zpow_add₀ this, mul_comm]
      /-
        🎉 no goals
      -/


/-- The `p`-adic norm respects division. -/
@[simp]
protected theorem div (q r : ℚ) : padicNorm p (q / r) = padicNorm p q / padicNorm p r :=
                        /-
                          p : Nat
                          hp : Fact (Nat.Prime p)
                          q r : Rat
                          hr : Eq r 0
                          ⊢ Eq (padicNorm p (HDiv.hDiv q r)) (HDiv.hDiv (padicNorm p q) (padicNorm p r))
                        -/
  if hr : r = 0 then by simp [hr]
                        /-
                          🎉 no goals
                        -/
                                                   /-
                                                     p : Nat
                                                     hp : Fact (Nat.Prime p)
                                                     q r : Rat
                                                     hr : Not (Eq r 0)
                                                     ⊢ Eq (HMul.hMul (padicNorm p (HDiv.hDiv q r)) (padicNorm p r)) (padicNorm p q)
                                                   -/
  else eq_div_of_mul_eq (padicNorm.nonzero hr) (by rw [← padicNorm.mul, div_mul_cancel₀ _ hr])
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- The `p`-adic norm of an integer is at most `1`. -/
protected theorem of_int (z : ℤ) : padicNorm p z ≤ 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    z : Int
    ⊢ LE.le (padicNorm p ↑z) 1
  -/
  obtain rfl | hz := eq_or_ne z 0
    /-
      case inl
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ LE.le (padicNorm p ↑0) 1
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : Nat
      hp : Fact (Nat.Prime p)
      z : Int
      hz : Ne z 0
      ⊢ LE.le (padicNorm p ↑z) 1
    -/
  · rw [padicNorm, if_neg (mod_cast hz)]
    /-
      case inr
      p : Nat
      hp : Fact (Nat.Prime p)
      z : Int
      hz : Ne z 0
      ⊢ LE.le (HPow.hPow (↑p) (Neg.neg (padicValRat p ↑z))) 1
    -/
    exact zpow_le_one_of_nonpos₀ (mod_cast hp.1.one_le) (by simp)
    /-
      🎉 no goals
    -/


private theorem nonarchimedean_aux {q r : ℚ} (h : padicValRat p q ≤ padicValRat p r) :
    padicNorm p (q + r) ≤ max (padicNorm p q) (padicNorm p r) :=
  have hnqp : padicNorm p q ≥ 0 := padicNorm.nonneg _
  have hnrp : padicNorm p r ≥ 0 := padicNorm.nonneg _
                        /-
                          p : Nat
                          hp : Fact (Nat.Prime p)
                          q r : Rat
                          h : LE.le (padicValRat p q) (padicValRat p r)
                          hnqp : GE.ge (padicNorm p q) 0
                          hnrp : GE.ge (padicNorm p r) 0
                          hq : Eq q 0
                          ⊢ LE.le (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
                        -/
  if hq : q = 0 then by simp [hq, max_eq_right hnrp, le_max_right]
                        /-
                          🎉 no goals
                        -/
  else
                          /-
                            p : Nat
                            hp : Fact (Nat.Prime p)
                            q r : Rat
                            h : LE.le (padicValRat p q) (padicValRat p r)
                            hnqp : GE.ge (padicNorm p q) 0
                            hnrp : GE.ge (padicNorm p r) 0
                            hq : Not (Eq q 0)
                            hr : Eq r 0
                            ⊢ LE.le (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
                          -/
    if hr : r = 0 then by simp [hr, max_eq_left hnqp, le_max_left]
                          /-
                            🎉 no goals
                          -/
    else
                                           /-
                                             p : Nat
                                             hp : Fact (Nat.Prime p)
                                             q r : Rat
                                             h : LE.le (padicValRat p q) (padicValRat p r)
                                             hnqp : GE.ge (padicNorm p q) 0
                                             hnrp : GE.ge (padicNorm p r) 0
                                             hq : Not (Eq q 0)
                                             hr : Not (Eq r 0)
                                             hqr : Eq (HAdd.hAdd q r) 0
                                             ⊢ LE.le (padicNorm p (HAdd.hAdd q r)) (padicNorm p q)
                                           -/
      if hqr : q + r = 0 then le_trans (by simpa [hqr] using hnqp) (le_max_left _ _)
                                           /-
                                             🎉 no goals
                                           -/
      else by
        /-
          p : Nat
          hp : Fact (Nat.Prime p)
          q r : Rat
          h : LE.le (padicValRat p q) (padicValRat p r)
          hnqp : GE.ge (padicNorm p q) 0
          hnrp : GE.ge (padicNorm p r) 0
          hq : Not (Eq q 0)
          hr : Not (Eq r 0)
          hqr : Not (Eq (HAdd.hAdd q r) 0)
          ⊢ LE.le (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
        -/
        unfold padicNorm; split_ifs
        /-
          p : Nat
          hp : Fact (Nat.Prime p)
          q r : Rat
          h : LE.le (padicValRat p q) (padicValRat p r)
          hnqp : GE.ge (padicNorm p q) 0
          hnrp : GE.ge (padicNorm p r) 0
          hq : Not (Eq q 0)
          hr : Not (Eq r 0)
          hqr : Not (Eq (HAdd.hAdd q r) 0)
          ⊢ LE.le (HPow.hPow (↑p) (Neg.neg (padicValRat p (HAdd.hAdd q r)))) (Max.max (H …
        -/
        apply le_max_iff.2
        /-
          p : Nat
          hp : Fact (Nat.Prime p)
          q r : Rat
          h : LE.le (padicValRat p q) (padicValRat p r)
          hnqp : GE.ge (padicNorm p q) 0
          hnrp : GE.ge (padicNorm p r) 0
          hq : Not (Eq q 0)
          hr : Not (Eq r 0)
          hqr : Not (Eq (HAdd.hAdd q r) 0)
          ⊢ Or (LE.le (HPow.hPow (↑p) (Neg.neg (padicValRat p (HAdd.hAdd q r)))) (HPow.h …
        -/
        left
        /-
          case h
          p : Nat
          hp : Fact (Nat.Prime p)
          q r : Rat
          h : LE.le (padicValRat p q) (padicValRat p r)
          hnqp : GE.ge (padicNorm p q) 0
          hnrp : GE.ge (padicNorm p r) 0
          hq : Not (Eq q 0)
          hr : Not (Eq r 0)
          hqr : Not (Eq (HAdd.hAdd q r) 0)
          ⊢ LE.le (HPow.hPow (↑p) (Neg.neg (padicValRat p (HAdd.hAdd q r)))) (HPow.hPow  …
        -/
        apply zpow_le_zpow_right₀
          /-
            case h.ha
            p : Nat
            hp : Fact (Nat.Prime p)
            q r : Rat
            h : LE.le (padicValRat p q) (padicValRat p r)
            hnqp : GE.ge (padicNorm p q) 0
            hnrp : GE.ge (padicNorm p r) 0
            hq : Not (Eq q 0)
            hr : Not (Eq r 0)
            hqr : Not (Eq (HAdd.hAdd q r) 0)
            ⊢ LE.le 1 ↑p
          -/
        · exact mod_cast le_of_lt hp.1.one_lt
          /-
            🎉 no goals
          -/
          /-
            case h.hmn
            p : Nat
            hp : Fact (Nat.Prime p)
            q r : Rat
            h : LE.le (padicValRat p q) (padicValRat p r)
            hnqp : GE.ge (padicNorm p q) 0
            hnrp : GE.ge (padicNorm p r) 0
            hq : Not (Eq q 0)
            hr : Not (Eq r 0)
            hqr : Not (Eq (HAdd.hAdd q r) 0)
            ⊢ LE.le (Neg.neg (padicValRat p (HAdd.hAdd q r))) (Neg.neg (padicValRat p q))
          -/
        · apply neg_le_neg
          /-
            case h.hmn.a
            p : Nat
            hp : Fact (Nat.Prime p)
            q r : Rat
            h : LE.le (padicValRat p q) (padicValRat p r)
            hnqp : GE.ge (padicNorm p q) 0
            hnrp : GE.ge (padicNorm p r) 0
            hq : Not (Eq q 0)
            hr : Not (Eq r 0)
            hqr : Not (Eq (HAdd.hAdd q r) 0)
            ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
          -/
          have : padicValRat p q = min (padicValRat p q) (padicValRat p r) := (min_eq_left h).symm
          /-
            case h.hmn.a
            p : Nat
            hp : Fact (Nat.Prime p)
            q r : Rat
            h : LE.le (padicValRat p q) (padicValRat p r)
            hnqp : GE.ge (padicNorm p q) 0
            hnrp : GE.ge (padicNorm p r) 0
            hq : Not (Eq q 0)
            hr : Not (Eq r 0)
            hqr : Not (Eq (HAdd.hAdd q r) 0)
            this : Eq (padicValRat p q) (Min.min (padicValRat p q) (padicValRat p r))
            ⊢ LE.le (padicValRat p q) (padicValRat p (HAdd.hAdd q r))
          -/
          rw [this]
          /-
            case h.hmn.a
            p : Nat
            hp : Fact (Nat.Prime p)
            q r : Rat
            h : LE.le (padicValRat p q) (padicValRat p r)
            hnqp : GE.ge (padicNorm p q) 0
            hnrp : GE.ge (padicNorm p r) 0
            hq : Not (Eq q 0)
            hr : Not (Eq r 0)
            hqr : Not (Eq (HAdd.hAdd q r) 0)
            this : Eq (padicValRat p q) (Min.min (padicValRat p q) (padicValRat p r))
            ⊢ LE.le (Min.min (padicValRat p q) (padicValRat p r)) (padicValRat p (HAdd.hAd …
          -/
          exact min_le_padicValRat_add hqr
          /-
            🎉 no goals
          -/


/-- The `p`-adic norm is nonarchimedean: the norm of `p + q` is at most the max of the norm of `p`
and the norm of `q`. -/
protected theorem nonarchimedean {q r : ℚ} :
    padicNorm p (q + r) ≤ max (padicNorm p q) (padicNorm p r) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    ⊢ LE.le (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
  -/
  wlog hle : padicValRat p q ≤ padicValRat p r generalizing q r
    /-
      case inr
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      this : ∀ {q r : Rat}, LE.le (padicValRat p q) (padicValRat p r) → LE.le (padic …
      hle : Not (LE.le (padicValRat p q) (padicValRat p r))
      ⊢ LE.le (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
    -/
  · rw [add_comm, max_comm]
    /-
      case inr
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      this : ∀ {q r : Rat}, LE.le (padicValRat p q) (padicValRat p r) → LE.le (padic …
      hle : Not (LE.le (padicValRat p q) (padicValRat p r))
      ⊢ LE.le (padicNorm p (HAdd.hAdd r q)) (Max.max (padicNorm p r) (padicNorm p q))
    -/
    exact this (le_of_not_le hle)
    /-
      🎉 no goals
    -/
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hle : LE.le (padicValRat p q) (padicValRat p r)
    ⊢ LE.le (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
  -/
  exact nonarchimedean_aux hle
  /-
    🎉 no goals
  -/


/-- The `p`-adic norm respects the triangle inequality: the norm of `p + q` is at most the norm of
`p` plus the norm of `q`. -/
theorem triangle_ineq (q r : ℚ) : padicNorm p (q + r) ≤ padicNorm p q + padicNorm p r :=
  calc
    padicNorm p (q + r) ≤ max (padicNorm p q) (padicNorm p r) := padicNorm.nonarchimedean
    _ ≤ padicNorm p q + padicNorm p r :=
      max_le_add_of_nonneg (padicNorm.nonneg _) (padicNorm.nonneg _)


/-- The `p`-adic norm of a difference is at most the max of each component. Restates the archimedean
property of the `p`-adic norm. -/
protected theorem sub {q r : ℚ} : padicNorm p (q - r) ≤ max (padicNorm p q) (padicNorm p r) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    ⊢ LE.le (padicNorm p (HSub.hSub q r)) (Max.max (padicNorm p q) (padicNorm p r))
  -/
  rw [sub_eq_add_neg, ← padicNorm.neg r]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    ⊢ LE.le (padicNorm p (HAdd.hAdd q (Neg.neg r))) (Max.max (padicNorm p q) (padi …
  -/
  exact padicNorm.nonarchimedean
  /-
    🎉 no goals
  -/


/-- If the `p`-adic norms of `q` and `r` are different, then the norm of `q + r` is equal to the max
of the norms of `q` and `r`. -/
theorem add_eq_max_of_ne {q r : ℚ} (hne : padicNorm p q ≠ padicNorm p r) :
    padicNorm p (q + r) = max (padicNorm p q) (padicNorm p r) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hne : Ne (padicNorm p q) (padicNorm p r)
    ⊢ Eq (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
  -/
  wlog hlt : padicNorm p r < padicNorm p q
    /-
      case inr
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      hne : Ne (padicNorm p q) (padicNorm p r)
      this : ∀ {p : Nat} [hp : Fact (Nat.Prime p)] {q r : Rat}, Ne (padicNorm p q) ( …
      hlt : Not (LT.lt (padicNorm p r) (padicNorm p q))
      ⊢ Eq (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
    -/
  · rw [add_comm, max_comm]
    /-
      case inr
      p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      hne : Ne (padicNorm p q) (padicNorm p r)
      this : ∀ {p : Nat} [hp : Fact (Nat.Prime p)] {q r : Rat}, Ne (padicNorm p q) ( …
      hlt : Not (LT.lt (padicNorm p r) (padicNorm p q))
      ⊢ Eq (padicNorm p (HAdd.hAdd r q)) (Max.max (padicNorm p r) (padicNorm p q))
    -/
    exact this hne.symm (hne.lt_or_lt.resolve_right hlt)
    /-
      🎉 no goals
    -/
  have : padicNorm p q ≤ max (padicNorm p (q + r)) (padicNorm p r) :=
    calc
      padicNorm p q = padicNorm p (q + r + (-r)) := by ring_nf
      _ ≤ max (padicNorm p (q + r)) (padicNorm p (-r)) := padicNorm.nonarchimedean
      _ = max (padicNorm p (q + r)) (padicNorm p r) := by simp
  have hnge : padicNorm p r ≤ padicNorm p (q + r) := by
    apply le_of_not_gt
    intro hgt
    rw [max_eq_right_of_lt hgt] at this
    exact not_lt_of_ge this hlt
  /-
    p✝ p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hne : Ne (padicNorm p q) (padicNorm p r)
    hlt : LT.lt (padicNorm p r) (padicNorm p q)
    this : LE.le (padicNorm p q) (Max.max (padicNorm p (HAdd.hAdd q r)) (padicNorm …
    hnge : LE.le (padicNorm p r) (padicNorm p (HAdd.hAdd q r))
    ⊢ Eq (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
  -/
  have : padicNorm p q ≤ padicNorm p (q + r) := by rwa [max_eq_left hnge] at this
  /-
    p✝ p : Nat
    hp : Fact (Nat.Prime p)
    q r : Rat
    hne : Ne (padicNorm p q) (padicNorm p r)
    hlt : LT.lt (padicNorm p r) (padicNorm p q)
    this✝ : LE.le (padicNorm p q) (Max.max (padicNorm p (HAdd.hAdd q r)) (padicNor …
    hnge : LE.le (padicNorm p r) (padicNorm p (HAdd.hAdd q r))
    this : LE.le (padicNorm p q) (padicNorm p (HAdd.hAdd q r))
    ⊢ Eq (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
  -/
  apply _root_.le_antisymm
    /-
      case a
      p✝ p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      hne : Ne (padicNorm p q) (padicNorm p r)
      hlt : LT.lt (padicNorm p r) (padicNorm p q)
      this✝ : LE.le (padicNorm p q) (Max.max (padicNorm p (HAdd.hAdd q r)) (padicNor …
      hnge : LE.le (padicNorm p r) (padicNorm p (HAdd.hAdd q r))
      this : LE.le (padicNorm p q) (padicNorm p (HAdd.hAdd q r))
      ⊢ LE.le (padicNorm p (HAdd.hAdd q r)) (Max.max (padicNorm p q) (padicNorm p r))
    -/
  · apply padicNorm.nonarchimedean
    /-
      🎉 no goals
    -/
    /-
      case a
      p✝ p : Nat
      hp : Fact (Nat.Prime p)
      q r : Rat
      hne : Ne (padicNorm p q) (padicNorm p r)
      hlt : LT.lt (padicNorm p r) (padicNorm p q)
      this✝ : LE.le (padicNorm p q) (Max.max (padicNorm p (HAdd.hAdd q r)) (padicNor …
      hnge : LE.le (padicNorm p r) (padicNorm p (HAdd.hAdd q r))
      this : LE.le (padicNorm p q) (padicNorm p (HAdd.hAdd q r))
      ⊢ LE.le (Max.max (padicNorm p q) (padicNorm p r)) (padicNorm p (HAdd.hAdd q r))
    -/
  · rwa [max_eq_left_of_lt hlt]
    /-
      🎉 no goals
    -/


/-- The `p`-adic norm is an absolute value: positive-definite and multiplicative, satisfying the
triangle inequality. -/
instance : IsAbsoluteValue (padicNorm p) where
  abv_nonneg' := padicNorm.nonneg
                                                          /-
                                                            p : Nat
                                                            hp : Fact (Nat.Prime p)
                                                            x✝ : Rat
                                                            hx : Eq x✝ 0
                                                            ⊢ Eq (padicNorm p x✝) 0
                                                          -/
  abv_eq_zero' := ⟨zero_of_padicNorm_eq_zero, fun hx ↦ by simp [hx]⟩
                                                          /-
                                                            🎉 no goals
                                                          -/
  abv_add' := padicNorm.triangle_ineq
  abv_mul' := padicNorm.mul


theorem dvd_iff_norm_le {n : ℕ} {z : ℤ} : ↑(p ^ n) ∣ z ↔ padicNorm p z ≤ (p : ℚ) ^ (-n : ℤ) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    z : Int
    ⊢ Iff (Dvd.dvd (↑(HPow.hPow p n)) z) (LE.le (padicNorm p ↑z) (HPow.hPow (↑p) ( …
  -/
  unfold padicNorm; split_ifs with hz
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      z : Int
      hz : Eq (↑z) 0
      ⊢ Iff (Dvd.dvd (↑(HPow.hPow p n)) z) (LE.le 0 (HPow.hPow (↑p) (Neg.neg ↑n)))
    -/
  · norm_cast at hz
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      z : Int
      hz : Eq z 0
      ⊢ Iff (Dvd.dvd (↑(HPow.hPow p n)) z) (LE.le 0 (HPow.hPow (↑p) (Neg.neg ↑n)))
    -/
    simp [hz]
    /-
      🎉 no goals
    -/
  · rw [zpow_le_zpow_iff_right₀, neg_le_neg_iff, padicValRat.of_int,
      padicValInt.of_ne_one_ne_zero hp.1.ne_one _]
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Nat
        z : Int
        hz : Not (Eq (↑z) 0)
        ⊢ Iff (Dvd.dvd (↑(HPow.hPow p n)) z) (LE.le ↑n ↑(multiplicity (↑p) z))
      -/
    · norm_cast
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Nat
        z : Int
        hz : Not (Eq (↑z) 0)
        ⊢ Iff (Dvd.dvd (↑(HPow.hPow p n)) z) (LE.le n (multiplicity (↑p) z))
      -/
      rw [← FiniteMultiplicity.pow_dvd_iff_le_multiplicity]
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          n : Nat
          z : Int
          hz : Not (Eq (↑z) 0)
          ⊢ Iff (Dvd.dvd (↑(HPow.hPow p n)) z) (Dvd.dvd (HPow.hPow (↑p) n) z)
        -/
      · norm_cast
        /-
          🎉 no goals
        -/
        /-
          case neg.hf
          p : Nat
          hp : Fact (Nat.Prime p)
          n : Nat
          z : Int
          hz : Not (Eq (↑z) 0)
          ⊢ FiniteMultiplicity (↑p) z
        -/
      · apply Int.finiteMultiplicity_iff.2 ⟨by simp [hp.out.ne_one], mod_cast hz⟩
        /-
          🎉 no goals
        -/
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Nat
        z : Int
        hz : Not (Eq (↑z) 0)
        ⊢ Ne z 0
      -/
    · exact_mod_cast hz
      /-
        🎉 no goals
      -/
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        n : Nat
        z : Int
        hz : Not (Eq (↑z) 0)
        ⊢ LT.lt 1 ↑p
      -/
    · exact_mod_cast hp.out.one_lt
      /-
        🎉 no goals
      -/


/-- The `p`-adic norm of an integer `m` is one iff `p` doesn't divide `m`. -/
theorem int_eq_one_iff (m : ℤ) : padicNorm p m = 1 ↔ ¬(p : ℤ) ∣ m := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m : Int
    ⊢ Iff (Eq (padicNorm p ↑m) 1) (Not (Dvd.dvd (↑p) m))
  -/
  nth_rw 2 [← pow_one p]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m : Int
    ⊢ Iff (Eq (padicNorm p ↑m) 1) (Not (Dvd.dvd (↑(HPow.hPow p 1)) m))
  -/
  simp only [dvd_iff_norm_le, Int.cast_natCast, Nat.cast_one, zpow_neg, zpow_one, not_le]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m : Int
    ⊢ Iff (Eq (padicNorm p ↑m) 1) (LT.lt (Inv.inv ↑p) (padicNorm p ↑m))
  -/
  constructor
    /-
      case mp
      p : Nat
      hp : Fact (Nat.Prime p)
      m : Int
      ⊢ Eq (padicNorm p ↑m) 1 → LT.lt (Inv.inv ↑p) (padicNorm p ↑m)
    -/
  · intro h
    /-
      case mp
      p : Nat
      hp : Fact (Nat.Prime p)
      m : Int
      h : Eq (padicNorm p ↑m) 1
      ⊢ LT.lt (Inv.inv ↑p) (padicNorm p ↑m)
    -/
    rw [h, inv_lt_one₀] <;> norm_cast
      /-
        case mp
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h : Eq (padicNorm p ↑m) 1
        ⊢ LT.lt 1 p
      -/
    · exact Nat.Prime.one_lt Fact.out
      /-
        🎉 no goals
      -/
      /-
        case mp
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h : Eq (padicNorm p ↑m) 1
        ⊢ LT.lt 0 p
      -/
    · exact Nat.Prime.pos Fact.out
      /-
        🎉 no goals
      -/
    /-
      case mpr
      p : Nat
      hp : Fact (Nat.Prime p)
      m : Int
      ⊢ LT.lt (Inv.inv ↑p) (padicNorm p ↑m) → Eq (padicNorm p ↑m) 1
    -/
  · simp only [padicNorm]
    /-
      case mpr
      p : Nat
      hp : Fact (Nat.Prime p)
      m : Int
      ⊢ LT.lt (Inv.inv ↑p) (ite (Eq (↑m) 0) 0 (HPow.hPow (↑p) (Neg.neg (padicValRat  …
    -/
    split_ifs
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h✝ : Eq (↑m) 0
        ⊢ LT.lt (Inv.inv ↑p) 0 → Eq 0 1
      -/
    · rw [inv_lt_zero, ← Nat.cast_zero, Nat.cast_lt]
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h✝ : Eq (↑m) 0
        ⊢ LT.lt p 0 → Eq (↑0) 1
      -/
      intro h
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h✝ : Eq (↑m) 0
        h : LT.lt p 0
        ⊢ Eq (↑0) 1
      -/
      exact (Nat.not_lt_zero p h).elim
      /-
        🎉 no goals
      -/
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h✝ : Not (Eq (↑m) 0)
        ⊢ LT.lt (Inv.inv ↑p) (HPow.hPow (↑p) (Neg.neg (padicValRat p ↑m))) → Eq (HPow. …
      -/
    · have : 1 < (p : ℚ) := by norm_cast; exact Nat.Prime.one_lt (Fact.out : Nat.Prime p)
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h✝ : Not (Eq (↑m) 0)
        this : LT.lt 1 ↑p
        ⊢ LT.lt (Inv.inv ↑p) (HPow.hPow (↑p) (Neg.neg (padicValRat p ↑m))) → Eq (HPow. …
      -/
      rw [← zpow_neg_one, zpow_lt_zpow_iff_right₀ this]
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h✝ : Not (Eq (↑m) 0)
        this : LT.lt 1 ↑p
        ⊢ LT.lt (-1) (Neg.neg (padicValRat p ↑m)) → Eq (HPow.hPow (↑p) (Neg.neg (padic …
      -/
      have : 0 ≤ padicValRat p m := by simp only [of_int, Nat.cast_nonneg]
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h✝ : Not (Eq (↑m) 0)
        this✝ : LT.lt 1 ↑p
        this : LE.le 0 (padicValRat p ↑m)
        ⊢ LT.lt (-1) (Neg.neg (padicValRat p ↑m)) → Eq (HPow.hPow (↑p) (Neg.neg (padic …
      -/
      intro h
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        m : Int
        h✝ : Not (Eq (↑m) 0)
        this✝ : LT.lt 1 ↑p
        this : LE.le 0 (padicValRat p ↑m)
        h : LT.lt (-1) (Neg.neg (padicValRat p ↑m))
        ⊢ Eq (HPow.hPow (↑p) (Neg.neg (padicValRat p ↑m))) 1
      -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
      rw [← zpow_zero (p : ℚ), zpow_right_inj₀] <;> linarith
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem int_lt_one_iff (m : ℤ) : padicNorm p m < 1 ↔ (p : ℤ) ∣ m := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m : Int
    ⊢ Iff (LT.lt (padicNorm p ↑m) 1) (Dvd.dvd (↑p) m)
  -/
  rw [← not_iff_not, ← int_eq_one_iff, eq_iff_le_not_lt]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m : Int
    ⊢ Iff (Not (LT.lt (padicNorm p ↑m) 1)) (And (LE.le (padicNorm p ↑m) 1) (Not (L …
  -/
  simp only [padicNorm.of_int, true_and]
  /-
    🎉 no goals
  -/


theorem of_nat (m : ℕ) : padicNorm p m ≤ 1 :=
  padicNorm.of_int (m : ℤ)


/-- The `p`-adic norm of a natural `m` is one iff `p` doesn't divide `m`. -/
theorem nat_eq_one_iff (m : ℕ) : padicNorm p m = 1 ↔ ¬p ∣ m := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m : Nat
    ⊢ Iff (Eq (padicNorm p ↑m) 1) (Not (Dvd.dvd p m))
  -/
  rw [← Int.natCast_dvd_natCast, ← int_eq_one_iff, Int.cast_natCast]
  /-
    🎉 no goals
  -/


theorem nat_lt_one_iff (m : ℕ) : padicNorm p m < 1 ↔ p ∣ m := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m : Nat
    ⊢ Iff (LT.lt (padicNorm p ↑m) 1) (Dvd.dvd p m)
  -/
  rw [← Int.natCast_dvd_natCast, ← int_lt_one_iff, Int.cast_natCast]
  /-
    🎉 no goals
  -/


/-- If a rational is not a p-adic integer, it is not an integer. -/
theorem not_int_of_not_padic_int (p : ℕ) {a : ℚ} [hp : Fact (Nat.Prime p)]
    (H : 1 < padicNorm p a) : ¬ a.isInt := by
  /-
    p : Nat
    a : Rat
    hp : Fact (Nat.Prime p)
    H : LT.lt 1 (padicNorm p a)
    ⊢ Not (Eq a.isInt Bool.true)
  -/
  contrapose! H
  /-
    p : Nat
    a : Rat
    hp : Fact (Nat.Prime p)
    H : Eq a.isInt Bool.true
    ⊢ LE.le (padicNorm p a) 1
  -/
  rw [Rat.eq_num_of_isInt H]
  /-
    p : Nat
    a : Rat
    hp : Fact (Nat.Prime p)
    H : Eq a.isInt Bool.true
    ⊢ LE.le (padicNorm p ↑a.num) 1
  -/
  apply padicNorm.of_int
  /-
    🎉 no goals
  -/


theorem sum_lt {α : Type*} {F : α → ℚ} {t : ℚ} {s : Finset α} :
    s.Nonempty → (∀ i ∈ s, padicNorm p (F i) < t) → padicNorm p (∑ i ∈ s, F i) < t := by
  classical
    refine s.induction_on (by rintro ⟨-, ⟨⟩⟩) ?_
    rintro a S haS IH - ht
    by_cases hs : S.Nonempty
    · rw [Finset.sum_insert haS]
      exact
        lt_of_le_of_lt padicNorm.nonarchimedean
          (max_lt (ht a (Finset.mem_insert_self a S))
            (IH hs fun b hb ↦ ht b (Finset.mem_insert_of_mem hb)))
    · simp_all


theorem sum_le {α : Type*} {F : α → ℚ} {t : ℚ} {s : Finset α} :
    s.Nonempty → (∀ i ∈ s, padicNorm p (F i) ≤ t) → padicNorm p (∑ i ∈ s, F i) ≤ t := by
  classical
    refine s.induction_on (by rintro ⟨-, ⟨⟩⟩) ?_
    rintro a S haS IH - ht
    by_cases hs : S.Nonempty
    · rw [Finset.sum_insert haS]
      exact
        padicNorm.nonarchimedean.trans
          (max_le (ht a (Finset.mem_insert_self a S))
            (IH hs fun b hb ↦ ht b (Finset.mem_insert_of_mem hb)))
    · simp_all


theorem sum_lt' {α : Type*} {F : α → ℚ} {t : ℚ} {s : Finset α}
    (hF : ∀ i ∈ s, padicNorm p (F i) < t) (ht : 0 < t) : padicNorm p (∑ i ∈ s, F i) < t := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    α : Type u_1
    F : α → Rat
    t : Rat
    s : Finset α
    hF : ∀ (i : α), Membership.mem s i → LT.lt (padicNorm p (F i)) t
    ht : LT.lt 0 t
    ⊢ LT.lt (padicNorm p (s.sum fun i => F i)) t
  -/
  obtain rfl | hs := Finset.eq_empty_or_nonempty s
    /-
      case inl
      p : Nat
      hp : Fact (Nat.Prime p)
      α : Type u_1
      F : α → Rat
      t : Rat
      ht : LT.lt 0 t
      hF : ∀ (i : α), Membership.mem EmptyCollection.emptyCollection i → LT.lt (padi …
      ⊢ LT.lt (padicNorm p (EmptyCollection.emptyCollection.sum fun i => F i)) t
    -/
  · simp [ht]
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : Nat
      hp : Fact (Nat.Prime p)
      α : Type u_1
      F : α → Rat
      t : Rat
      s : Finset α
      hF : ∀ (i : α), Membership.mem s i → LT.lt (padicNorm p (F i)) t
      ht : LT.lt 0 t
      hs : s.Nonempty
      ⊢ LT.lt (padicNorm p (s.sum fun i => F i)) t
    -/
  · exact sum_lt hs hF
    /-
      🎉 no goals
    -/


theorem sum_le' {α : Type*} {F : α → ℚ} {t : ℚ} {s : Finset α}
    (hF : ∀ i ∈ s, padicNorm p (F i) ≤ t) (ht : 0 ≤ t) : padicNorm p (∑ i ∈ s, F i) ≤ t := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    α : Type u_1
    F : α → Rat
    t : Rat
    s : Finset α
    hF : ∀ (i : α), Membership.mem s i → LE.le (padicNorm p (F i)) t
    ht : LE.le 0 t
    ⊢ LE.le (padicNorm p (s.sum fun i => F i)) t
  -/
  obtain rfl | hs := Finset.eq_empty_or_nonempty s
    /-
      case inl
      p : Nat
      hp : Fact (Nat.Prime p)
      α : Type u_1
      F : α → Rat
      t : Rat
      ht : LE.le 0 t
      hF : ∀ (i : α), Membership.mem EmptyCollection.emptyCollection i → LE.le (padi …
      ⊢ LE.le (padicNorm p (EmptyCollection.emptyCollection.sum fun i => F i)) t
    -/
  · simp [ht]
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : Nat
      hp : Fact (Nat.Prime p)
      α : Type u_1
      F : α → Rat
      t : Rat
      s : Finset α
      hF : ∀ (i : α), Membership.mem s i → LE.le (padicNorm p (F i)) t
      ht : LE.le 0 t
      hs : s.Nonempty
      ⊢ LE.le (padicNorm p (s.sum fun i => F i)) t
    -/
  · exact sum_le hs hF
    /-
      🎉 no goals
    -/


