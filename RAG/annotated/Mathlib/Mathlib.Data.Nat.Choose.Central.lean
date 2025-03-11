/-- The central binomial coefficient, `Nat.choose (2 * n) n`.
-/
def centralBinom (n : ℕ) :=
  (2 * n).choose n


theorem centralBinom_eq_two_mul_choose (n : ℕ) : centralBinom n = (2 * n).choose n :=
  rfl


theorem centralBinom_pos (n : ℕ) : 0 < centralBinom n :=
  choose_pos (Nat.le_mul_of_pos_left _ zero_lt_two)


theorem centralBinom_ne_zero (n : ℕ) : centralBinom n ≠ 0 :=
  (centralBinom_pos n).ne'


@[simp]
theorem centralBinom_zero : centralBinom 0 = 1 :=
  choose_zero_right _


/-- The central binomial coefficient is the largest binomial coefficient.
-/
theorem choose_le_centralBinom (r n : ℕ) : choose (2 * n) r ≤ centralBinom n :=
  calc
    (2 * n).choose r ≤ (2 * n).choose (2 * n / 2) := choose_le_middle r (2 * n)
                               /-
                                 r n : Nat
                                 ⊢ Eq ((HMul.hMul 2 n).choose (HDiv.hDiv (HMul.hMul 2 n) 2)) ((HMul.hMul 2 n).c …
                               -/
    _ = (2 * n).choose n := by rw [Nat.mul_div_cancel_left n zero_lt_two]
                               /-
                                 🎉 no goals
                               -/


theorem two_le_centralBinom (n : ℕ) (n_pos : 0 < n) : 2 ≤ centralBinom n :=
  calc
    2 ≤ 2 * n := Nat.le_mul_of_pos_right _ n_pos
    _ = (2 * n).choose 1 := (choose_one_right (2 * n)).symm
    _ ≤ centralBinom n := choose_le_centralBinom 1 n


/-- An inductive property of the central binomial coefficient.
-/
theorem succ_mul_centralBinom_succ (n : ℕ) :
    (n + 1) * centralBinom (n + 1) = 2 * (2 * n + 1) * centralBinom n :=
  calc
    (n + 1) * (2 * (n + 1)).choose (n + 1) = (2 * n + 2).choose (n + 1) * (n + 1) := mul_comm _ _
                                                 /-
                                                   n : Nat
                                                   ⊢ Eq (HMul.hMul ((HAdd.hAdd (HMul.hMul 2 n) 2).choose (HAdd.hAdd n 1)) (HAdd.h …
                                                 -/
    _ = (2 * n + 1).choose n * (2 * n + 2) := by rw [choose_succ_right_eq, choose_mul_succ_eq]
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                   /-
                                                     n : Nat
                                                     ⊢ Eq (HMul.hMul ((HAdd.hAdd (HMul.hMul 2 n) 1).choose n) (HAdd.hAdd (HMul.hMul …
                                                   -/
    _ = 2 * ((2 * n + 1).choose n * (n + 1)) := by ring
                                                   /-
                                                     🎉 no goals
                                                   -/
    _ = 2 * ((2 * n + 1).choose n * (2 * n + 1 - n)) := by rw [two_mul n, add_assoc,
                                                               Nat.add_sub_cancel_left]
                                                   /-
                                                     n : Nat
                                                     ⊢ Eq (HMul.hMul 2 (HMul.hMul ((HAdd.hAdd (HMul.hMul 2 n) 1).choose n) (HSub.hS …
                                                   -/
    _ = 2 * ((2 * n).choose n * (2 * n + 1)) := by rw [choose_mul_succ_eq]
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                 /-
                                                   n : Nat
                                                   ⊢ Eq (HMul.hMul 2 (HMul.hMul ((HMul.hMul 2 n).choose n) (HAdd.hAdd (HMul.hMul  …
                                                 -/
    _ = 2 * (2 * n + 1) * (2 * n).choose n := by rw [mul_assoc, mul_comm (2 * n + 1)]
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- An exponential lower bound on the central binomial coefficient.
This bound is of interest because it appears in
[Tochiori's refinement of Erdős's proof of Bertrand's postulate](tochiori_bertrand).
-/
theorem four_pow_lt_mul_centralBinom (n : ℕ) (n_big : 4 ≤ n) : 4 ^ n < n * centralBinom n := by
  /-
    n : Nat
    n_big : LE.le 4 n
    ⊢ LT.lt (HPow.hPow 4 n) (HMul.hMul n n.centralBinom)
  -/
  induction' n using Nat.strong_induction_on with n IH
  /-
    case h
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → LE.le 4 m → LT.lt (HPow.hPow 4 m) (HMul.hMul m m …
    n_big : LE.le 4 n
    ⊢ LT.lt (HPow.hPow 4 n) (HMul.hMul n n.centralBinom)
  -/
  rcases lt_trichotomy n 4 with (hn | rfl | hn)
    /-
      case h.inl
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → LE.le 4 m → LT.lt (HPow.hPow 4 m) (HMul.hMul m m …
      n_big : LE.le 4 n
      hn : LT.lt n 4
      ⊢ LT.lt (HPow.hPow 4 n) (HMul.hMul n n.centralBinom)
    -/
  · clear IH; exact False.elim ((not_lt.2 n_big) hn)
              /-
                🎉 no goals
              -/
    /-
      case h.inr.inl
      IH : ∀ (m : Nat), LT.lt m 4 → LE.le 4 m → LT.lt (HPow.hPow 4 m) (HMul.hMul m m …
      n_big : LE.le 4 4
      ⊢ LT.lt (HPow.hPow 4 4) (HMul.hMul 4 (Nat.centralBinom 4))
    -/
  · norm_num [centralBinom, choose]
    /-
      🎉 no goals
    -/
  /-
    case h.inr.inr
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → LE.le 4 m → LT.lt (HPow.hPow 4 m) (HMul.hMul m m …
    n_big : LE.le 4 n
    hn : LT.lt 4 n
    ⊢ LT.lt (HPow.hPow 4 n) (HMul.hMul n n.centralBinom)
  -/
  obtain ⟨n, rfl⟩ : ∃ m, n = m + 1 := Nat.exists_eq_succ_of_ne_zero (Nat.not_eq_zero_of_lt hn)
  calc
    4 ^ (n + 1) < 4 * (n * centralBinom n) := lt_of_eq_of_lt pow_succ' <|
      (mul_lt_mul_left <| zero_lt_four' ℕ).mpr (IH n n.lt_succ_self (Nat.le_of_lt_succ hn))
    _ ≤ 2 * (2 * n + 1) * centralBinom n := by rw [← mul_assoc]; linarith
    _ = (n + 1) * centralBinom (n + 1) := (succ_mul_centralBinom_succ n).symm


/-- An exponential lower bound on the central binomial coefficient.
This bound is weaker than `Nat.four_pow_lt_mul_centralBinom`, but it is of historical interest
because it appears in Erdős's proof of Bertrand's postulate.
-/
theorem four_pow_le_two_mul_self_mul_centralBinom :
    ∀ (n : ℕ) (_ : 0 < n), 4 ^ n ≤ 2 * n * centralBinom n
  | 0, pr => (Nat.not_lt_zero _ pr).elim
               /-
                 x✝ : LT.lt 0 1
                 ⊢ LE.le (HPow.hPow 4 1) (HMul.hMul (HMul.hMul 2 1) (Nat.centralBinom 1))
               -/
  | 1, _ => by norm_num [centralBinom, choose]
               /-
                 🎉 no goals
               -/
               /-
                 x✝ : LT.lt 0 2
                 ⊢ LE.le (HPow.hPow 4 2) (HMul.hMul (HMul.hMul 2 2) (Nat.centralBinom 2))
               -/
  | 2, _ => by norm_num [centralBinom, choose]
               /-
                 🎉 no goals
               -/
               /-
                 x✝ : LT.lt 0 3
                 ⊢ LE.le (HPow.hPow 4 3) (HMul.hMul (HMul.hMul 2 3) (Nat.centralBinom 3))
               -/
  | 3, _ => by norm_num [centralBinom, choose]
               /-
                 🎉 no goals
               -/
  | n + 4, _ =>
    calc
      4 ^ (n+4) ≤ (n+4) * centralBinom (n+4) := (four_pow_lt_mul_centralBinom _ le_add_self).le
      _ ≤ 2 * (n+4) * centralBinom (n+4) := by
        /-
          n : Nat
          x✝ : LT.lt 0 (HAdd.hAdd n 4)
          ⊢ LE.le (HMul.hMul (HAdd.hAdd n 4) (HAdd.hAdd n 4).centralBinom) (HMul.hMul (H …
        -/
        rw [mul_assoc]; refine Nat.le_mul_of_pos_left _ zero_lt_two
                        /-
                          🎉 no goals
                        -/


theorem two_dvd_centralBinom_succ (n : ℕ) : 2 ∣ centralBinom (n + 1) := by
  /-
    n : Nat
    ⊢ Dvd.dvd 2 (HAdd.hAdd n 1).centralBinom
  -/
  use (n + 1 + n).choose n
  rw [centralBinom_eq_two_mul_choose, two_mul, ← add_assoc,
      choose_succ_succ' (n + 1 + n) n, choose_symm_add, ← two_mul]


theorem two_dvd_centralBinom_of_one_le {n : ℕ} (h : 0 < n) : 2 ∣ centralBinom n := by
  /-
    n : Nat
    h : LT.lt 0 n
    ⊢ Dvd.dvd 2 n.centralBinom
  -/
  rw [← Nat.succ_pred_eq_of_pos h]
  /-
    n : Nat
    h : LT.lt 0 n
    ⊢ Dvd.dvd 2 n.pred.succ.centralBinom
  -/
  exact two_dvd_centralBinom_succ n.pred
  /-
    🎉 no goals
  -/


/-- A crucial lemma to ensure that Catalan numbers can be defined via their explicit formula
  `catalan n = n.centralBinom / (n + 1)`. -/
theorem succ_dvd_centralBinom (n : ℕ) : n + 1 ∣ n.centralBinom := by
  have h_s : (n + 1).Coprime (2 * n + 1) := by
    rw [two_mul, add_assoc, coprime_add_self_right, coprime_self_add_left]
    exact coprime_one_left n
  /-
    n : Nat
    h_s : (HAdd.hAdd n 1).Coprime (HAdd.hAdd (HMul.hMul 2 n) 1)
    ⊢ Dvd.dvd (HAdd.hAdd n 1) n.centralBinom
  -/
  apply h_s.dvd_of_dvd_mul_left
  /-
    n : Nat
    h_s : (HAdd.hAdd n 1).Coprime (HAdd.hAdd (HMul.hMul 2 n) 1)
    ⊢ Dvd.dvd (HAdd.hAdd n 1) (HMul.hMul (HAdd.hAdd (HMul.hMul 2 n) 1) n.centralBi …
  -/
  apply Nat.dvd_of_mul_dvd_mul_left zero_lt_two
  /-
    n : Nat
    h_s : (HAdd.hAdd n 1).Coprime (HAdd.hAdd (HMul.hMul 2 n) 1)
    ⊢ Dvd.dvd (HMul.hMul 2 (HAdd.hAdd n 1)) (HMul.hMul 2 (HMul.hMul (HAdd.hAdd (HM …
  -/
  rw [← mul_assoc, ← succ_mul_centralBinom_succ, mul_comm]
  /-
    n : Nat
    h_s : (HAdd.hAdd n 1).Coprime (HAdd.hAdd (HMul.hMul 2 n) 1)
    ⊢ Dvd.dvd (HMul.hMul (HAdd.hAdd n 1) 2) (HMul.hMul (HAdd.hAdd n 1) (HAdd.hAdd  …
  -/
  exact mul_dvd_mul_left _ (two_dvd_centralBinom_succ n)
  /-
    🎉 no goals
  -/


