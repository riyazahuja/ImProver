/-- The binomial `PMF`: the probability of observing exactly `i` “heads” in a sequence of `n`
independent coin tosses, each having probability `p` of coming up “heads”. -/
noncomputable
def binomial (p : ℝ≥0∞) (h : p ≤ 1) (n : ℕ) : PMF (Fin (n + 1)) :=
  .ofFintype (fun i => p^(i : ℕ) * (1-p)^((Fin.last n - i) : ℕ) * (n.choose i : ℕ)) (by
    /-
      p : ENNReal
      h : LE.le p 1
      n : Nat
      ⊢ Eq (Finset.univ.sum fun a => (fun i => HMul.hMul (HMul.hMul (HPow.hPow p ↑i) …
    -/
    convert (add_pow p (1-p) n).symm
      /-
        case h.e'_2
        p : ENNReal
        h : LE.le p 1
        n : Nat
        ⊢ Eq (Finset.univ.sum fun a => (fun i => HMul.hMul (HMul.hMul (HPow.hPow p ↑i) …
      -/
    · rw [Finset.sum_fin_eq_sum_range]
      /-
        case h.e'_2
        p : ENNReal
        h : LE.le p 1
        n : Nat
        ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => dite (LT.lt i (HAdd.hAdd n 1 …
      -/
      apply Finset.sum_congr rfl
      /-
        case h.e'_2
        p : ENNReal
        h : LE.le p 1
        n : Nat
        ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq (dite (LT. …
      -/
      intro i hi
      /-
        case h.e'_2
        p : ENNReal
        h : LE.le p 1
        n i : Nat
        hi : Membership.mem (Finset.range (HAdd.hAdd n 1)) i
        ⊢ Eq (dite (LT.lt i (HAdd.hAdd n 1)) (fun h => HMul.hMul (HMul.hMul (HPow.hPow …
      -/
      rw [Finset.mem_range] at hi
      /-
        case h.e'_2
        p : ENNReal
        h : LE.le p 1
        n i : Nat
        hi : LT.lt i (HAdd.hAdd n 1)
        ⊢ Eq (dite (LT.lt i (HAdd.hAdd n 1)) (fun h => HMul.hMul (HMul.hMul (HPow.hPow …
      -/
      rw [dif_pos hi, Fin.last]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        p : ENNReal
        h : LE.le p 1
        n : Nat
        ⊢ Eq 1 (HPow.hPow (HAdd.hAdd p (HSub.hSub 1 p)) n)
      -/
    · simp [h])
      /-
        🎉 no goals
      -/


theorem binomial_apply (p : ℝ≥0∞) (h : p ≤ 1) (n : ℕ) (i : Fin (n + 1)) :
    binomial p h n i = p^(i : ℕ) * (1-p)^((Fin.last n - i) : ℕ) * (n.choose i : ℕ) := rfl


@[simp]
theorem binomial_apply_zero (p : ℝ≥0∞) (h : p ≤ 1) (n : ℕ) :
    binomial p h n 0 = (1-p)^n := by
  /-
    p : ENNReal
    h : LE.le p 1
    n : Nat
    ⊢ Eq ((PMF.binomial p h n) 0) (HPow.hPow (HSub.hSub 1 p) n)
  -/
  simp [binomial_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem binomial_apply_last (p : ℝ≥0∞) (h : p ≤ 1) (n : ℕ) :
    binomial p h n (.last n) = p^n := by
  /-
    p : ENNReal
    h : LE.le p 1
    n : Nat
    ⊢ Eq ((PMF.binomial p h n) (Fin.last n)) (HPow.hPow p n)
  -/
  simp [binomial_apply]
  /-
    🎉 no goals
  -/


theorem binomial_apply_self (p : ℝ≥0∞) (h : p ≤ 1) (n : ℕ) :
                                 /-
                                   p : ENNReal
                                   h : LE.le p 1
                                   n : Nat
                                   ⊢ Eq ((PMF.binomial p h n) ↑n) (HPow.hPow p n)
                                 -/
    binomial p h n n = p^n := by simp
                                 /-
                                   🎉 no goals
                                 -/


/-- The binomial distribution on one coin is the bernoully distribution. -/
theorem binomial_one_eq_bernoulli (p : ℝ≥0∞) (h : p ≤ 1) :
    binomial p h 1 = (bernoulli p h).map (cond · 1 0) := by
  /-
    p : ENNReal
    h : LE.le p 1
    ⊢ Eq (PMF.binomial p h 1) (PMF.map (fun x => cond x 1 0) (PMF.bernoulli p h))
  -/
                         /-
                           🎉 no goals
                         -/
  ext i; fin_cases i <;> simp [tsum_bool, binomial_apply]
                         /-
                           🎉 no goals
                         -/


