/-- A logarithmic upper bound on the multiplicity of a prime in a binomial coefficient. -/
theorem factorization_choose_le_log : (choose n k).factorization p ≤ log p n := by
  /-
    p n k : Nat
    ⊢ LE.le ((n.choose k).factorization p) (Nat.log p n)
  -/
  by_cases h : (choose n k).factorization p = 0
    /-
      case pos
      p n k : Nat
      h : Eq ((n.choose k).factorization p) 0
      ⊢ LE.le ((n.choose k).factorization p) (Nat.log p n)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    p n k : Nat
    h : Not (Eq ((n.choose k).factorization p) 0)
    ⊢ LE.le ((n.choose k).factorization p) (Nat.log p n)
  -/
  have hp : p.Prime := Not.imp_symm (choose n k).factorization_eq_zero_of_non_prime h
  have hkn : k ≤ n := by
    refine le_of_not_lt fun hnk => h ?_
    simp [choose_eq_zero_of_lt hnk]
  /-
    case neg
    p n k : Nat
    h : Not (Eq ((n.choose k).factorization p) 0)
    hp : Nat.Prime p
    hkn : LE.le k n
    ⊢ LE.le ((n.choose k).factorization p) (Nat.log p n)
  -/
  rw [factorization_def _ hp, @padicValNat_def _ ⟨hp⟩ _ (choose_pos hkn)]
  /-
    case neg
    p n k : Nat
    h : Not (Eq ((n.choose k).factorization p) 0)
    hp : Nat.Prime p
    hkn : LE.le k n
    ⊢ LE.le (multiplicity p (n.choose k)) (Nat.log p n)
  -/
  rw [← Nat.cast_le (α := ℕ∞), ← FiniteMultiplicity.emultiplicity_eq_multiplicity]
    /-
      case neg
      p n k : Nat
      h : Not (Eq ((n.choose k).factorization p) 0)
      hp : Nat.Prime p
      hkn : LE.le k n
      ⊢ LE.le (emultiplicity p (n.choose k)) ↑(Nat.log p n)
    -/
  · simp only [hp.emultiplicity_choose hkn (lt_add_one _), Nat.cast_le]
    /-
      case neg
      p n k : Nat
      h : Not (Eq ((n.choose k).factorization p) 0)
      hp : Nat.Prime p
      hkn : LE.le k n
      ⊢ LE.le (Finset.filter (fun i => LE.le (HPow.hPow p i) (HAdd.hAdd (HMod.hMod k …
    -/
    exact (Finset.card_filter_le _ _).trans (le_of_eq (Nat.card_Ico _ _))
    /-
      🎉 no goals
    -/
  /-
    case neg
    p n k : Nat
    h : Not (Eq ((n.choose k).factorization p) 0)
    hp : Nat.Prime p
    hkn : LE.le k n
    ⊢ FiniteMultiplicity p (n.choose k)
  -/
  apply Nat.finiteMultiplicity_iff.2 ⟨hp.ne_one, choose_pos hkn⟩
  /-
    🎉 no goals
  -/


/-- A `pow` form of `Nat.factorization_choose_le` -/
theorem pow_factorization_choose_le (hn : 0 < n) : p ^ (choose n k).factorization p ≤ n :=
  pow_le_of_le_log hn.ne' factorization_choose_le_log


/-- Primes greater than about `sqrt n` appear only to multiplicity 0 or 1
in the binomial coefficient. -/
theorem factorization_choose_le_one (p_large : n < p ^ 2) : (choose n k).factorization p ≤ 1 := by
  /-
    p n k : Nat
    p_large : LT.lt n (HPow.hPow p 2)
    ⊢ LE.le ((n.choose k).factorization p) 1
  -/
  apply factorization_choose_le_log.trans
  /-
    p n k : Nat
    p_large : LT.lt n (HPow.hPow p 2)
    ⊢ LE.le (Nat.log p n) 1
  -/
  rcases eq_or_ne n 0 with (rfl | hn0); · simp
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    p n k : Nat
    p_large : LT.lt n (HPow.hPow p 2)
    hn0 : Ne n 0
    ⊢ LE.le (Nat.log p n) 1
  -/
  exact Nat.lt_succ_iff.1 (log_lt_of_lt_pow hn0 p_large)
  /-
    🎉 no goals
  -/


theorem factorization_choose_of_lt_three_mul (hp' : p ≠ 2) (hk : p ≤ k) (hk' : p ≤ n - k)
    (hn : n < 3 * p) : (choose n k).factorization p = 0 := by
  /-
    p n k : Nat
    hp' : Ne p 2
    hk : LE.le p k
    hk' : LE.le p (HSub.hSub n k)
    hn : LT.lt n (HMul.hMul 3 p)
    ⊢ Eq ((n.choose k).factorization p) 0
  -/
  cases' em' p.Prime with hp hp
    /-
      case inl
      p n k : Nat
      hp' : Ne p 2
      hk : LE.le p k
      hk' : LE.le p (HSub.hSub n k)
      hn : LT.lt n (HMul.hMul 3 p)
      hp : Not (Nat.Prime p)
      ⊢ Eq ((n.choose k).factorization p) 0
    -/
  · exact factorization_eq_zero_of_non_prime (choose n k) hp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p n k : Nat
    hp' : Ne p 2
    hk : LE.le p k
    hk' : LE.le p (HSub.hSub n k)
    hn : LT.lt n (HMul.hMul 3 p)
    hp : Nat.Prime p
    ⊢ Eq ((n.choose k).factorization p) 0
  -/
  cases' lt_or_le n k with hnk hkn
    /-
      case inr.inl
      p n k : Nat
      hp' : Ne p 2
      hk : LE.le p k
      hk' : LE.le p (HSub.hSub n k)
      hn : LT.lt n (HMul.hMul 3 p)
      hp : Nat.Prime p
      hnk : LT.lt n k
      ⊢ Eq ((n.choose k).factorization p) 0
    -/
  · simp [choose_eq_zero_of_lt hnk]
    /-
      🎉 no goals
    -/
  rw [factorization_def _ hp, @padicValNat_def _ ⟨hp⟩ _ (choose_pos hkn),
    ← emultiplicity_eq_zero_iff_multiplicity_eq_zero]
  simp only [hp.emultiplicity_choose hkn (lt_add_one _), cast_eq_zero, Finset.card_eq_zero,
    Finset.filter_eq_empty_iff, not_le]
  /-
    case inr.inr
    p n k : Nat
    hp' : Ne p 2
    hk : LE.le p k
    hk' : LE.le p (HSub.hSub n k)
    hn : LT.lt n (HMul.hMul 3 p)
    hp : Nat.Prime p
    hkn : LE.le k n
    ⊢ ∀ ⦃x : Nat⦄, Membership.mem (Finset.Ico 1 (HAdd.hAdd (Nat.log p n) 1)) x → L …
  -/
  intro i hi
  /-
    case inr.inr
    p n k : Nat
    hp' : Ne p 2
    hk : LE.le p k
    hk' : LE.le p (HSub.hSub n k)
    hn : LT.lt n (HMul.hMul 3 p)
    hp : Nat.Prime p
    hkn : LE.le k n
    i : Nat
    hi : Membership.mem (Finset.Ico 1 (HAdd.hAdd (Nat.log p n) 1)) i
    ⊢ LT.lt (HAdd.hAdd (HMod.hMod k (HPow.hPow p i)) (HMod.hMod (HSub.hSub n k) (H …
  -/
  rcases eq_or_lt_of_le (Finset.mem_Ico.mp hi).1 with (rfl | hi)
    /-
      case inr.inr.inl
      p n k : Nat
      hp' : Ne p 2
      hk : LE.le p k
      hk' : LE.le p (HSub.hSub n k)
      hn : LT.lt n (HMul.hMul 3 p)
      hp : Nat.Prime p
      hkn : LE.le k n
      hi : Membership.mem (Finset.Ico 1 (HAdd.hAdd (Nat.log p n) 1)) 1
      ⊢ LT.lt (HAdd.hAdd (HMod.hMod k (HPow.hPow p 1)) (HMod.hMod (HSub.hSub n k) (H …
    -/
  · rw [pow_one, ← add_lt_add_iff_left (2 * p), ← succ_mul, two_mul, add_add_add_comm]
    exact
      lt_of_le_of_lt
        (add_le_add
          (add_le_add_right (le_mul_of_one_le_right' ((one_le_div_iff hp.pos).mpr hk)) (k % p))
          (add_le_add_right (le_mul_of_one_le_right' ((one_le_div_iff hp.pos).mpr hk'))
            ((n - k) % p)))
        (by rwa [div_add_mod, div_add_mod, add_tsub_cancel_of_le hkn])
  · replace hn : n < p ^ i := by
      have : 3 ≤ p := lt_of_le_of_ne hp.two_le hp'.symm
      calc
        n < 3 * p := hn
        _ ≤ p * p := mul_le_mul_right' this p
        _ = p ^ 2 := (sq p).symm
        _ ≤ p ^ i := pow_right_mono₀ hp.one_lt.le hi
    rwa [mod_eq_of_lt (lt_of_le_of_lt hkn hn), mod_eq_of_lt (lt_of_le_of_lt tsub_le_self hn),
      add_tsub_cancel_of_le hkn]


/-- Primes greater than about `2 * n / 3` and less than `n` do not appear in the factorization of
`centralBinom n`. -/
theorem factorization_centralBinom_of_two_mul_self_lt_three_mul (n_big : 2 < n) (p_le_n : p ≤ n)
    (big : 2 * n < 3 * p) : (centralBinom n).factorization p = 0 := by
  /-
    p n : Nat
    n_big : LT.lt 2 n
    p_le_n : LE.le p n
    big : LT.lt (HMul.hMul 2 n) (HMul.hMul 3 p)
    ⊢ Eq (n.centralBinom.factorization p) 0
  -/
  refine factorization_choose_of_lt_three_mul ?_ p_le_n (p_le_n.trans ?_) big
    /-
      case refine_1
      p n : Nat
      n_big : LT.lt 2 n
      p_le_n : LE.le p n
      big : LT.lt (HMul.hMul 2 n) (HMul.hMul 3 p)
      ⊢ Ne p 2
    -/
  · omega
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p n : Nat
      n_big : LT.lt 2 n
      p_le_n : LE.le p n
      big : LT.lt (HMul.hMul 2 n) (HMul.hMul 3 p)
      ⊢ LE.le n (HSub.hSub (HMul.hMul 2 n) n)
    -/
  · rw [two_mul, add_tsub_cancel_left]
    /-
      🎉 no goals
    -/


theorem factorization_factorial_eq_zero_of_lt (h : n < p) : (factorial n).factorization p = 0 := by
  /-
    p n : Nat
    h : LT.lt n p
    ⊢ Eq (n.factorial.factorization p) 0
  -/
  induction' n with n hn; · simp
                            /-
                              🎉 no goals
                            -/
  rw [factorial_succ, factorization_mul n.succ_ne_zero n.factorial_ne_zero, Finsupp.coe_add,
    Pi.add_apply, hn (lt_of_succ_lt h), add_zero, factorization_eq_zero_of_lt h]


theorem factorization_choose_eq_zero_of_lt (h : n < p) : (choose n k).factorization p = 0 := by
  /-
    p n k : Nat
    h : LT.lt n p
    ⊢ Eq ((n.choose k).factorization p) 0
  -/
  by_cases hnk : n < k; · simp [choose_eq_zero_of_lt hnk]
                          /-
                            🎉 no goals
                          -/
  rw [choose_eq_factorial_div_factorial (le_of_not_lt hnk),
    factorization_div (factorial_mul_factorial_dvd_factorial (le_of_not_lt hnk)), Finsupp.coe_tsub,
    Pi.sub_apply, factorization_factorial_eq_zero_of_lt h, zero_tsub]


/-- If a prime `p` has positive multiplicity in the `n`th central binomial coefficient,
`p` is no more than `2 * n` -/
theorem factorization_centralBinom_eq_zero_of_two_mul_lt (h : 2 * n < p) :
    (centralBinom n).factorization p = 0 :=
  factorization_choose_eq_zero_of_lt h


/-- Contrapositive form of `Nat.factorization_centralBinom_eq_zero_of_two_mul_lt` -/
theorem le_two_mul_of_factorization_centralBinom_pos
    (h_pos : 0 < (centralBinom n).factorization p) : p ≤ 2 * n :=
  le_of_not_lt (pos_iff_ne_zero.mp h_pos ∘ factorization_centralBinom_eq_zero_of_two_mul_lt)


/-- A binomial coefficient is the product of its prime factors, which are at most `n`. -/
theorem prod_pow_factorization_choose (n k : ℕ) (hkn : k ≤ n) :
    (∏ p ∈ Finset.range (n + 1), p ^ (Nat.choose n k).factorization p) = choose n k := by
  conv => -- Porting note: was `nth_rw_rhs`
    rhs
    rw [← factorization_prod_pow_eq_self (choose_pos hkn).ne']
  /-
    n k : Nat
    hkn : LE.le k n
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).prod fun p => HPow.hPow p ((n.choose k).f …
  -/
  rw [eq_comm]
  /-
    n k : Nat
    hkn : LE.le k n
    ⊢ Eq ((n.choose k).factorization.prod fun x1 x2 => HPow.hPow x1 x2) ((Finset.r …
  -/
  apply Finset.prod_subset
    /-
      case h
      n k : Nat
      hkn : LE.le k n
      ⊢ HasSubset.Subset (n.choose k).factorization.support (Finset.range (HAdd.hAdd …
    -/
  · intro p hp
    /-
      case h
      n k : Nat
      hkn : LE.le k n
      p : Nat
      hp : Membership.mem (n.choose k).factorization.support p
      ⊢ Membership.mem (Finset.range (HAdd.hAdd n 1)) p
    -/
    rw [Finset.mem_range]
    /-
      case h
      n k : Nat
      hkn : LE.le k n
      p : Nat
      hp : Membership.mem (n.choose k).factorization.support p
      ⊢ LT.lt p (HAdd.hAdd n 1)
    -/
    contrapose! hp
    /-
      case h
      n k : Nat
      hkn : LE.le k n
      p : Nat
      hp : LE.le (HAdd.hAdd n 1) p
      ⊢ Not (Membership.mem (n.choose k).factorization.support p)
    -/
    rw [Finsupp.mem_support_iff, Classical.not_not, factorization_choose_eq_zero_of_lt hp]
    /-
      🎉 no goals
    -/
    /-
      case hf
      n k : Nat
      hkn : LE.le k n
      ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Not (Membersh …
    -/
  · intro p _ h2
    /-
      case hf
      n k : Nat
      hkn : LE.le k n
      p : Nat
      a✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) p
      h2 : Not (Membership.mem (n.choose k).factorization.support p)
      ⊢ Eq ((fun x1 x2 => HPow.hPow x1 x2) p ((n.choose k).factorization p)) 1
    -/
    simp [Classical.not_not.1 (mt Finsupp.mem_support_iff.2 h2)]
    /-
      🎉 no goals
    -/


/-- The `n`th central binomial coefficient is the product of its prime factors, which are
at most `2n`. -/
theorem prod_pow_factorization_centralBinom (n : ℕ) :
    (∏ p ∈ Finset.range (2 * n + 1), p ^ (centralBinom n).factorization p) = centralBinom n := by
  /-
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd (HMul.hMul 2 n) 1)).prod fun p => HPow.hPow p ( …
  -/
  apply prod_pow_factorization_choose
  /-
    case hkn
    n : Nat
    ⊢ LE.le n (HMul.hMul 2 n)
  -/
  omega
  /-
    🎉 no goals
  -/


