/-- The 2-adic valuation of the n-th harmonic number is the negative of the logarithm
    of n. -/
theorem padicValRat_two_harmonic (n : ℕ) : padicValRat 2 (harmonic n) = -Nat.log 2 n := by
  /-
    n : Nat
    ⊢ Eq (padicValRat 2 (harmonic n)) (Neg.neg ↑(Nat.log 2 n))
  -/
  induction' n with n ih
    /-
      case zero
      ⊢ Eq (padicValRat 2 (harmonic 0)) (Neg.neg ↑(Nat.log 2 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ih : Eq (padicValRat 2 (harmonic n)) (Neg.neg ↑(Nat.log 2 n))
      ⊢ Eq (padicValRat 2 (harmonic (HAdd.hAdd n 1))) (Neg.neg ↑(Nat.log 2 (HAdd.hAd …
    -/
  · rcases eq_or_ne n 0 with rfl | hn
      /-
        case succ.inl
        ih : Eq (padicValRat 2 (harmonic 0)) (Neg.neg ↑(Nat.log 2 0))
        ⊢ Eq (padicValRat 2 (harmonic (HAdd.hAdd 0 1))) (Neg.neg ↑(Nat.log 2 (HAdd.hAd …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case succ.inr
      n : Nat
      ih : Eq (padicValRat 2 (harmonic n)) (Neg.neg ↑(Nat.log 2 n))
      hn : Ne n 0
      ⊢ Eq (padicValRat 2 (harmonic (HAdd.hAdd n 1))) (Neg.neg ↑(Nat.log 2 (HAdd.hAd …
    -/
    rw [harmonic_succ]
    have key : padicValRat 2 (harmonic n) ≠ padicValRat 2 (↑(n + 1))⁻¹ := by
      rw [ih, padicValRat.inv, padicValRat.of_nat, Ne, neg_inj, Nat.cast_inj]
      exact Nat.log_ne_padicValNat_succ hn
    rw [padicValRat.add_eq_min (harmonic_succ n ▸ (harmonic_pos n.succ_ne_zero).ne')
        (harmonic_pos hn).ne' (inv_ne_zero (Nat.cast_ne_zero.mpr n.succ_ne_zero)) key, ih,
        padicValRat.inv, padicValRat.of_nat, min_neg_neg, neg_inj, ← Nat.cast_max, Nat.cast_inj]
    /-
      case succ.inr
      n : Nat
      ih : Eq (padicValRat 2 (harmonic n)) (Neg.neg ↑(Nat.log 2 n))
      hn : Ne n 0
      key : Ne (padicValRat 2 (harmonic n)) (padicValRat 2 (Inv.inv ↑(HAdd.hAdd n 1)))
      ⊢ Eq (Max.max (Nat.log 2 n) (padicValNat 2 n.succ)) (Nat.log 2 (HAdd.hAdd n 1))
    -/
    exact Nat.max_log_padicValNat_succ_eq_log_succ n
    /-
      🎉 no goals
    -/


/-- The 2-adic norm of the n-th harmonic number is 2 raised to the logarithm of n in base 2. -/
lemma padicNorm_two_harmonic {n : ℕ} (hn : n ≠ 0) :
    ‖(harmonic n : ℚ_[2])‖ = 2 ^ (Nat.log 2 n) := by
  rw [padicNormE.eq_padicNorm, padicNorm.eq_zpow_of_nonzero (harmonic_pos hn).ne',
    padicValRat_two_harmonic, neg_neg, zpow_natCast, Rat.cast_pow, Rat.cast_natCast, Nat.cast_ofNat]


/-- The n-th harmonic number is not an integer for n ≥ 2. -/
theorem harmonic_not_int {n : ℕ} (h : 2 ≤ n) : ¬ (harmonic n).isInt := by
  /-
    n : Nat
    h : LE.le 2 n
    ⊢ Not (Eq (harmonic n).isInt Bool.true)
  -/
  apply padicNorm.not_int_of_not_padic_int 2
  rw [padicNorm.eq_zpow_of_nonzero (harmonic_pos (ne_zero_of_lt h)).ne',
      padicValRat_two_harmonic, neg_neg, zpow_natCast]
  /-
    n : Nat
    h : LE.le 2 n
    ⊢ LT.lt 1 (HPow.hPow (↑2) (Nat.log 2 n))
  -/
  exact one_lt_pow₀ one_lt_two (Nat.log_pos one_lt_two h).ne'
  /-
    🎉 no goals
  -/

