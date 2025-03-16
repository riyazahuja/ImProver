/-- `choose n k` is the number of `k`-element subsets in an `n`-element set. Also known as binomial
coefficients. -/
def choose : ℕ → ℕ → ℕ
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)


@[simp]
                                                         /-
                                                           n : Nat
                                                           ⊢ Eq (n.choose 0) 1
                                                         -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
theorem choose_zero_right (n : ℕ) : choose n 0 = 1 := by cases n <;> rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
theorem choose_zero_succ (k : ℕ) : choose 0 (succ k) = 0 :=
  rfl


theorem choose_succ_succ (n k : ℕ) : choose (succ n) (succ k) = choose n k + choose n (succ k) :=
  rfl


theorem choose_succ_succ' (n k : ℕ) : choose (n + 1) (k + 1) = choose n k + choose n (k + 1) :=
  rfl


theorem choose_succ_left (n k : ℕ) (hk : 0 < k) :
    choose (n + 1) k = choose n (k - 1) + choose n k := by
  /-
    n k : Nat
    hk : LT.lt 0 k
    ⊢ Eq ((HAdd.hAdd n 1).choose k) (HAdd.hAdd (n.choose (HSub.hSub k 1)) (n.choos …
  -/
  obtain ⟨l, rfl⟩ : ∃ l, k = l + 1 := Nat.exists_eq_add_of_le' hk
  /-
    case intro
    n l : Nat
    hk : LT.lt 0 (HAdd.hAdd l 1)
    ⊢ Eq ((HAdd.hAdd n 1).choose (HAdd.hAdd l 1)) (HAdd.hAdd (n.choose (HSub.hSub  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem choose_succ_right (n k : ℕ) (hn : 0 < n) :
    choose n (k + 1) = choose (n - 1) k + choose (n - 1) (k + 1) := by
  /-
    n k : Nat
    hn : LT.lt 0 n
    ⊢ Eq (n.choose (HAdd.hAdd k 1)) (HAdd.hAdd ((HSub.hSub n 1).choose k) ((HSub.h …
  -/
  obtain ⟨l, rfl⟩ : ∃ l, n = l + 1 := Nat.exists_eq_add_of_le' hn
  /-
    case intro
    k l : Nat
    hn : LT.lt 0 (HAdd.hAdd l 1)
    ⊢ Eq ((HAdd.hAdd l 1).choose (HAdd.hAdd k 1)) (HAdd.hAdd ((HSub.hSub (HAdd.hAd …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem choose_eq_choose_pred_add {n k : ℕ} (hn : 0 < n) (hk : 0 < k) :
    choose n k = choose (n - 1) (k - 1) + choose (n - 1) k := by
  /-
    n k : Nat
    hn : LT.lt 0 n
    hk : LT.lt 0 k
    ⊢ Eq (n.choose k) (HAdd.hAdd ((HSub.hSub n 1).choose (HSub.hSub k 1)) ((HSub.h …
  -/
  obtain ⟨l, rfl⟩ : ∃ l, k = l + 1 := Nat.exists_eq_add_of_le' hk
  /-
    case intro
    n : Nat
    hn : LT.lt 0 n
    l : Nat
    hk : LT.lt 0 (HAdd.hAdd l 1)
    ⊢ Eq (n.choose (HAdd.hAdd l 1)) (HAdd.hAdd ((HSub.hSub n 1).choose (HSub.hSub  …
  -/
  rw [choose_succ_right _ _ hn, Nat.add_one_sub_one]
  /-
    🎉 no goals
  -/


theorem choose_eq_zero_of_lt : ∀ {n k}, n < k → choose n k = 0
  | _, 0, hk => absurd hk (Nat.not_lt_zero _)
  | 0, _ + 1, _ => choose_zero_succ _
  | n + 1, k + 1, hk => by
    /-
      n k : Nat
      hk : LT.lt (HAdd.hAdd n 1) (HAdd.hAdd k 1)
      ⊢ Eq ((HAdd.hAdd n 1).choose (HAdd.hAdd k 1)) 0
    -/
    have hnk : n < k := lt_of_succ_lt_succ hk
    /-
      n k : Nat
      hk : LT.lt (HAdd.hAdd n 1) (HAdd.hAdd k 1)
      hnk : LT.lt n k
      ⊢ Eq ((HAdd.hAdd n 1).choose (HAdd.hAdd k 1)) 0
    -/
    have hnk1 : n < k + 1 := lt_of_succ_lt hk
    /-
      n k : Nat
      hk : LT.lt (HAdd.hAdd n 1) (HAdd.hAdd k 1)
      hnk : LT.lt n k
      hnk1 : LT.lt n (HAdd.hAdd k 1)
      ⊢ Eq ((HAdd.hAdd n 1).choose (HAdd.hAdd k 1)) 0
    -/
    rw [choose_succ_succ, choose_eq_zero_of_lt hnk, choose_eq_zero_of_lt hnk1]
    /-
      🎉 no goals
    -/


@[simp]
theorem choose_self (n : ℕ) : choose n n = 1 := by
  /-
    n : Nat
    ⊢ Eq (n.choose n) 1
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, choose, choose_eq_zero_of_lt (lt_succ_self _)]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem choose_succ_self (n : ℕ) : choose n (succ n) = 0 :=
  choose_eq_zero_of_lt (lt_succ_self _)


@[simp]
                                                      /-
                                                        n : Nat
                                                        ⊢ Eq (n.choose 1) n
                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
lemma choose_one_right (n : ℕ) : choose n 1 = n := by induction n <;> simp [*, choose, Nat.add_comm]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/

-- The `n+1`-st triangle number is `n` more than the `n`-th triangle number

theorem triangle_succ (n : ℕ) : (n + 1) * (n + 1 - 1) / 2 = n * (n - 1) / 2 + n := by
  /-
    n : Nat
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HAdd.hAdd n 1) (HSub.hSub (HAdd.hAdd n 1) 1)) 2) ( …
  -/
  rw [← add_mul_div_left, Nat.mul_comm 2 n, ← Nat.mul_add, Nat.add_sub_cancel, Nat.mul_comm]
  /-
    n : Nat
    ⊢ Eq (HDiv.hDiv (HMul.hMul n (HAdd.hAdd n 1)) 2) (HDiv.hDiv (HMul.hMul n (HAdd …
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases n <;> rfl; apply zero_lt_succ
                   /-
                     🎉 no goals
                   -/


/-- `choose n 2` is the `n`-th triangle number. -/
theorem choose_two_right (n : ℕ) : choose n 2 = n * (n - 1) / 2 := by
  /-
    n : Nat
    ⊢ Eq (n.choose 2) (HDiv.hDiv (HMul.hMul n (HSub.hSub n 1)) 2)
  -/
  induction' n with n ih
    /-
      case zero
      ⊢ Eq (Nat.choose 0 2) (HDiv.hDiv (HMul.hMul 0 (HSub.hSub 0 1)) 2)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ih : Eq (n.choose 2) (HDiv.hDiv (HMul.hMul n (HSub.hSub n 1)) 2)
      ⊢ Eq ((HAdd.hAdd n 1).choose 2) (HDiv.hDiv (HMul.hMul (HAdd.hAdd n 1) (HSub.hS …
    -/
  · rw [triangle_succ n, choose, ih]
    /-
      case succ
      n : Nat
      ih : Eq (n.choose 2) (HDiv.hDiv (HMul.hMul n (HSub.hSub n 1)) 2)
      ⊢ Eq (HAdd.hAdd (n.choose 1) (HDiv.hDiv (HMul.hMul n (HSub.hSub n 1)) 2)) (HAd …
    -/
    simp [Nat.add_comm]
    /-
      🎉 no goals
    -/


theorem choose_pos : ∀ {n k}, k ≤ n → 0 < choose n k
                   /-
                     x✝ : Nat
                     hk : LE.le x✝ 0
                     ⊢ LT.lt 0 (Nat.choose 0 x✝)
                   -/
  | 0, _, hk => by rw [Nat.eq_zero_of_le_zero hk]; decide
                                                   /-
                                                     🎉 no goals
                                                   -/
                      /-
                        n : Nat
                        x✝ : LE.le 0 (HAdd.hAdd n 1)
                        ⊢ LT.lt 0 ((HAdd.hAdd n 1).choose 0)
                      -/
  | n + 1, 0, _ => by simp
                      /-
                        🎉 no goals
                      -/
  | _ + 1, _ + 1, hk => Nat.add_pos_left (choose_pos (le_of_succ_le_succ hk)) _


theorem choose_eq_zero_iff {n k : ℕ} : n.choose k = 0 ↔ n < k :=
  ⟨fun h => lt_of_not_ge (mt Nat.choose_pos h.symm.not_lt), Nat.choose_eq_zero_of_lt⟩


theorem succ_mul_choose_eq : ∀ n k, succ n * choose n k = choose (succ n) (succ k) * succ k
               /-
                 ⊢ Eq (HMul.hMul (Nat.succ 0) (Nat.choose 0 0)) (HMul.hMul ((Nat.succ 0).choose …
               -/
  | 0, 0 => by decide
               /-
                 🎉 no goals
               -/
                   /-
                     k : Nat
                     ⊢ Eq (HMul.hMul (Nat.succ 0) (Nat.choose 0 (HAdd.hAdd k 1))) (HMul.hMul ((Nat. …
                   -/
  | 0, k + 1 => by simp [choose]
                   /-
                     🎉 no goals
                   -/
                   /-
                     n : Nat
                     ⊢ Eq (HMul.hMul (HAdd.hAdd n 1).succ ((HAdd.hAdd n 1).choose 0)) (HMul.hMul (( …
                   -/
  | n + 1, 0 => by simp [choose, mul_succ, Nat.add_comm]
                   /-
                     🎉 no goals
                   -/
  | n + 1, k + 1 => by
    rw [choose_succ_succ (succ n) (succ k), Nat.add_mul, ← succ_mul_choose_eq n, mul_succ, ←
      succ_mul_choose_eq n, Nat.add_right_comm, ← Nat.mul_add, ← choose_succ_succ, ← succ_mul]


theorem choose_mul_factorial_mul_factorial : ∀ {n k}, k ≤ n → choose n k * k ! * (n - k)! = n !
                   /-
                     x✝ : Nat
                     hk : LE.le x✝ 0
                     ⊢ Eq (HMul.hMul (HMul.hMul (Nat.choose 0 x✝) x✝.factorial) (HSub.hSub 0 x✝).fa …
                   -/
  | 0, _, hk => by simp [Nat.eq_zero_of_le_zero hk]
                   /-
                     🎉 no goals
                   -/
                      /-
                        n : Nat
                        x✝ : LE.le 0 (HAdd.hAdd n 1)
                        ⊢ Eq (HMul.hMul (HMul.hMul ((HAdd.hAdd n 1).choose 0) (Nat.factorial 0)) (HSub …
                      -/
  | n + 1, 0, _ => by simp
                      /-
                        🎉 no goals
                      -/
  | n + 1, succ k, hk => by
    /-
      n k : Nat
      hk : LE.le k.succ (HAdd.hAdd n 1)
      ⊢ Eq (HMul.hMul (HMul.hMul ((HAdd.hAdd n 1).choose k.succ) k.succ.factorial) ( …
    -/
    rcases lt_or_eq_of_le hk with hk₁ | hk₁
    · have h : choose n k * k.succ ! * (n - k)! = (k + 1) * n ! := by
        rw [← choose_mul_factorial_mul_factorial (le_of_succ_le_succ hk)]
        simp [factorial_succ, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
      have h₁ : (n - k)! = (n - k) * (n - k.succ)! := by
        rw [← succ_sub_succ, succ_sub (le_of_lt_succ hk₁), factorial_succ]
      have h₂ : choose n (succ k) * k.succ ! * ((n - k) * (n - k.succ)!) = (n - k) * n ! := by
        rw [← choose_mul_factorial_mul_factorial (le_of_lt_succ hk₁)]
        simp [factorial_succ, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
      /-
        case inl
        n k : Nat
        hk : LE.le k.succ (HAdd.hAdd n 1)
        hk₁ : LT.lt k.succ (HAdd.hAdd n 1)
        h : Eq (HMul.hMul (HMul.hMul (n.choose k) k.succ.factorial) (HSub.hSub n k).fa …
        h₁ : Eq (HSub.hSub n k).factorial (HMul.hMul (HSub.hSub n k) (HSub.hSub n k.su …
        h₂ : Eq (HMul.hMul (HMul.hMul (n.choose k.succ) k.succ.factorial) (HMul.hMul ( …
        ⊢ Eq (HMul.hMul (HMul.hMul ((HAdd.hAdd n 1).choose k.succ) k.succ.factorial) ( …
      -/
      have h₃ : k * n ! ≤ n * n ! := Nat.mul_le_mul_right _ (le_of_succ_le_succ hk)
      rw [choose_succ_succ, Nat.add_mul, Nat.add_mul, succ_sub_succ, h, h₁, h₂, Nat.add_mul,
        Nat.mul_sub_right_distrib, factorial_succ, ← Nat.add_sub_assoc h₃, Nat.add_assoc,
        ← Nat.add_mul, Nat.add_sub_cancel_left, Nat.add_comm]
      /-
        case inr
        n k : Nat
        hk : LE.le k.succ (HAdd.hAdd n 1)
        hk₁ : Eq k.succ (HAdd.hAdd n 1)
        ⊢ Eq (HMul.hMul (HMul.hMul ((HAdd.hAdd n 1).choose k.succ) k.succ.factorial) ( …
      -/
    · rw [hk₁]; simp [hk₁, Nat.mul_comm, choose, Nat.sub_self]
                /-
                  🎉 no goals
                -/


theorem choose_mul {n k s : ℕ} (hkn : k ≤ n) (hsk : s ≤ k) :
    n.choose k * k.choose s = n.choose s * (n - s).choose (k - s) :=
                                               /-
                                                 n k s : Nat
                                                 hkn : LE.le k n
                                                 hsk : LE.le s k
                                                 ⊢ LT.lt 0 (HMul.hMul (HMul.hMul (HSub.hSub n k).factorial (HSub.hSub k s).fact …
                                               -/
  have h : 0 < (n - k)! * (k - s)! * s ! := by apply_rules [factorial_pos, Nat.mul_pos]
                                               /-
                                                 🎉 no goals
                                               -/
  Nat.mul_right_cancel h <|
  calc
    n.choose k * k.choose s * ((n - k)! * (k - s)! * s !) =
        n.choose k * (k.choose s * s ! * (k - s)!) * (n - k)! := by
      rw [Nat.mul_assoc, Nat.mul_assoc, Nat.mul_assoc, Nat.mul_assoc _ s !, Nat.mul_assoc,
        Nat.mul_comm (n - k)!, Nat.mul_comm s !]
    _ = n ! := by
      /-
        n k s : Nat
        hkn : LE.le k n
        hsk : LE.le s k
        h : LT.lt 0 (HMul.hMul (HMul.hMul (HSub.hSub n k).factorial (HSub.hSub k s).fa …
        ⊢ Eq (HMul.hMul (HMul.hMul (n.choose k) (HMul.hMul (HMul.hMul (k.choose s) s.f …
      -/
      rw [choose_mul_factorial_mul_factorial hsk, choose_mul_factorial_mul_factorial hkn]
      /-
        🎉 no goals
      -/
    _ = n.choose s * s ! * ((n - s).choose (k - s) * (k - s)! * (n - s - (k - s))!) := by
      rw [choose_mul_factorial_mul_factorial (Nat.sub_le_sub_right hkn _),
        choose_mul_factorial_mul_factorial (hsk.trans hkn)]
    _ = n.choose s * (n - s).choose (k - s) * ((n - k)! * (k - s)! * s !) := by
      rw [Nat.sub_sub_sub_cancel_right hsk, Nat.mul_assoc, Nat.mul_left_comm s !, Nat.mul_assoc,
        Nat.mul_comm (k - s)!, Nat.mul_comm s !, Nat.mul_right_comm, ← Nat.mul_assoc]


theorem choose_eq_factorial_div_factorial {n k : ℕ} (hk : k ≤ n) :
    choose n k = n ! / (k ! * (n - k)!) := by
  /-
    n k : Nat
    hk : LE.le k n
    ⊢ Eq (n.choose k) (HDiv.hDiv n.factorial (HMul.hMul k.factorial (HSub.hSub n k …
  -/
  rw [← choose_mul_factorial_mul_factorial hk, Nat.mul_assoc]
  /-
    n k : Nat
    hk : LE.le k n
    ⊢ Eq (n.choose k) (HDiv.hDiv (HMul.hMul (n.choose k) (HMul.hMul k.factorial (H …
  -/
  exact (mul_div_left _ (Nat.mul_pos (factorial_pos _) (factorial_pos _))).symm
  /-
    🎉 no goals
  -/


theorem add_choose (i j : ℕ) : (i + j).choose j = (i + j)! / (i ! * j !) := by
  rw [choose_eq_factorial_div_factorial (Nat.le_add_left j i), Nat.add_sub_cancel_right,
    Nat.mul_comm]


theorem add_choose_mul_factorial_mul_factorial (i j : ℕ) :
    (i + j).choose j * i ! * j ! = (i + j)! := by
  rw [← choose_mul_factorial_mul_factorial (Nat.le_add_left _ _), Nat.add_sub_cancel_right,
    Nat.mul_right_comm]


theorem factorial_mul_factorial_dvd_factorial {n k : ℕ} (hk : k ≤ n) : k ! * (n - k)! ∣ n ! := by
  /-
    n k : Nat
    hk : LE.le k n
    ⊢ Dvd.dvd (HMul.hMul k.factorial (HSub.hSub n k).factorial) n.factorial
  -/
  rw [← choose_mul_factorial_mul_factorial hk, Nat.mul_assoc]; exact Nat.dvd_mul_left _ _
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem factorial_mul_factorial_dvd_factorial_add (i j : ℕ) : i ! * j ! ∣ (i + j)! := by
  suffices i ! * (i + j - i) ! ∣ (i + j)! by
    rwa [Nat.add_sub_cancel_left i j] at this
  /-
    i j : Nat
    ⊢ Dvd.dvd (HMul.hMul i.factorial (HSub.hSub (HAdd.hAdd i j) i).factorial) (HAd …
  -/
  exact factorial_mul_factorial_dvd_factorial (Nat.le_add_right _ _)
  /-
    🎉 no goals
  -/


@[simp]
theorem choose_symm {n k : ℕ} (hk : k ≤ n) : choose n (n - k) = choose n k := by
  rw [choose_eq_factorial_div_factorial hk, choose_eq_factorial_div_factorial (Nat.sub_le _ _),
    Nat.sub_sub_self hk, Nat.mul_comm]


theorem choose_symm_of_eq_add {n a b : ℕ} (h : n = a + b) : Nat.choose n a = Nat.choose n b := by
  suffices choose n (n - b) = choose n b by
    rw [h, Nat.add_sub_cancel_right] at this; rwa [h]
  /-
    n a b : Nat
    h : Eq n (HAdd.hAdd a b)
    ⊢ Eq (n.choose (HSub.hSub n b)) (n.choose b)
  -/
  exact choose_symm (h ▸ le_add_left _ _)
  /-
    🎉 no goals
  -/


theorem choose_symm_add {a b : ℕ} : choose (a + b) a = choose (a + b) b :=
  choose_symm_of_eq_add rfl


theorem choose_symm_half (m : ℕ) : choose (2 * m + 1) (m + 1) = choose (2 * m + 1) m := by
  /-
    m : Nat
    ⊢ Eq ((HAdd.hAdd (HMul.hMul 2 m) 1).choose (HAdd.hAdd m 1)) ((HAdd.hAdd (HMul. …
  -/
  apply choose_symm_of_eq_add
  /-
    case h
    m : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HAdd.hAdd m 1) m)
  -/
  rw [Nat.add_comm m 1, Nat.add_assoc 1 m m, Nat.add_comm (2 * m) 1, Nat.two_mul m]
  /-
    🎉 no goals
  -/


theorem choose_succ_right_eq (n k : ℕ) : choose n (k + 1) * (k + 1) = choose n k * (n - k) := by
  have e : (n + 1) * choose n k = choose n (k + 1) * (k + 1) + choose n k * (k + 1) := by
    rw [← Nat.add_mul, Nat.add_comm (choose _ _), ← choose_succ_succ, succ_mul_choose_eq]
  /-
    n k : Nat
    e : Eq (HMul.hMul (HAdd.hAdd n 1) (n.choose k)) (HAdd.hAdd (HMul.hMul (n.choos …
    ⊢ Eq (HMul.hMul (n.choose (HAdd.hAdd k 1)) (HAdd.hAdd k 1)) (HMul.hMul (n.choo …
  -/
  rw [← Nat.sub_eq_of_eq_add e, Nat.mul_comm, ← Nat.mul_sub_left_distrib, Nat.add_sub_add_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem choose_succ_self_right : ∀ n : ℕ, (n + 1).choose n = n + 1
  | 0 => rfl
                /-
                  n : Nat
                  ⊢ Eq ((HAdd.hAdd (HAdd.hAdd n 1) 1).choose (HAdd.hAdd n 1)) (HAdd.hAdd (HAdd.h …
                -/
  | n + 1 => by rw [choose_succ_succ, choose_succ_self_right n, choose_self]
                /-
                  🎉 no goals
                -/


theorem choose_mul_succ_eq (n k : ℕ) : n.choose k * (n + 1) = (n + 1).choose k * (n + 1 - k) := by
  cases k with
  | zero => simp
  | succ k =>
    obtain hk | hk := le_or_lt (k + 1) (n + 1)
    · rw [choose_succ_succ, Nat.add_mul, succ_sub_succ, ← choose_succ_right_eq, ← succ_sub_succ,
        Nat.mul_sub_left_distrib, Nat.add_sub_cancel' (Nat.mul_le_mul_left _ hk)]
    · rw [choose_eq_zero_of_lt hk, choose_eq_zero_of_lt (n.lt_succ_self.trans hk), Nat.zero_mul,
        Nat.zero_mul]


theorem ascFactorial_eq_factorial_mul_choose (n k : ℕ) :
    (n + 1).ascFactorial k = k ! * (n + k).choose k := by
  /-
    n k : Nat
    ⊢ Eq ((HAdd.hAdd n 1).ascFactorial k) (HMul.hMul k.factorial ((HAdd.hAdd n k). …
  -/
  rw [Nat.mul_comm]
  /-
    n k : Nat
    ⊢ Eq ((HAdd.hAdd n 1).ascFactorial k) (HMul.hMul ((HAdd.hAdd n k).choose k) k. …
  -/
  apply Nat.mul_right_cancel (n + k - k).factorial_pos
  rw [choose_mul_factorial_mul_factorial <| Nat.le_add_left k n, Nat.add_sub_cancel_right,
    ← factorial_mul_ascFactorial, Nat.mul_comm]


theorem ascFactorial_eq_factorial_mul_choose' (n k : ℕ) :
    n.ascFactorial k = k ! * (n + k - 1).choose k := by
  /-
    n k : Nat
    ⊢ Eq (n.ascFactorial k) (HMul.hMul k.factorial ((HSub.hSub (HAdd.hAdd n k) 1). …
  -/
  cases n
    /-
      case zero
      k : Nat
      ⊢ Eq (Nat.ascFactorial 0 k) (HMul.hMul k.factorial ((HSub.hSub (HAdd.hAdd 0 k) …
    -/
  · cases k
      /-
        case zero.zero
        ⊢ Eq (Nat.ascFactorial 0 0) (HMul.hMul (Nat.factorial 0) ((HSub.hSub (HAdd.hAd …
      -/
    · rw [ascFactorial_zero, choose_zero_right, factorial_zero, Nat.mul_one]
      /-
        🎉 no goals
      -/
    · simp only [zero_ascFactorial, zero_eq, Nat.zero_add, succ_sub_succ_eq_sub,
        Nat.le_zero_eq, Nat.sub_zero, choose_succ_self, Nat.mul_zero]
  /-
    case succ
    k n✝ : Nat
    ⊢ Eq ((HAdd.hAdd n✝ 1).ascFactorial k) (HMul.hMul k.factorial ((HSub.hSub (HAd …
  -/
  rw [ascFactorial_eq_factorial_mul_choose]
  /-
    case succ
    k n✝ : Nat
    ⊢ Eq (HMul.hMul k.factorial ((HAdd.hAdd n✝ k).choose k)) (HMul.hMul k.factoria …
  -/
  simp only [succ_add_sub_one]
  /-
    🎉 no goals
  -/


theorem factorial_dvd_ascFactorial (n k : ℕ) : k ! ∣ n.ascFactorial k :=
  ⟨(n + k - 1).choose k, ascFactorial_eq_factorial_mul_choose' _ _⟩


theorem choose_eq_asc_factorial_div_factorial (n k : ℕ) :
    (n + k).choose k = (n + 1).ascFactorial k / k ! := by
  /-
    n k : Nat
    ⊢ Eq ((HAdd.hAdd n k).choose k) (HDiv.hDiv ((HAdd.hAdd n 1).ascFactorial k) k. …
  -/
  apply Nat.mul_left_cancel k.factorial_pos
  /-
    n k : Nat
    ⊢ Eq (HMul.hMul k.factorial ((HAdd.hAdd n k).choose k)) (HMul.hMul k.factorial …
  -/
  rw [← ascFactorial_eq_factorial_mul_choose]
  /-
    n k : Nat
    ⊢ Eq ((HAdd.hAdd n 1).ascFactorial k) (HMul.hMul k.factorial (HDiv.hDiv ((HAdd …
  -/
  exact (Nat.mul_div_cancel' <| factorial_dvd_ascFactorial _ _).symm
  /-
    🎉 no goals
  -/


theorem choose_eq_asc_factorial_div_factorial' (n k : ℕ) :
    (n + k - 1).choose k = n.ascFactorial k / k ! :=
  Nat.eq_div_of_mul_eq_right k.factorial_ne_zero (ascFactorial_eq_factorial_mul_choose' _ _).symm


theorem descFactorial_eq_factorial_mul_choose (n k : ℕ) : n.descFactorial k = k ! * n.choose k := by
  /-
    n k : Nat
    ⊢ Eq (n.descFactorial k) (HMul.hMul k.factorial (n.choose k))
  -/
  obtain h | h := Nat.lt_or_ge n k
    /-
      case inl
      n k : Nat
      h : LT.lt n k
      ⊢ Eq (n.descFactorial k) (HMul.hMul k.factorial (n.choose k))
    -/
  · rw [descFactorial_eq_zero_iff_lt.2 h, choose_eq_zero_of_lt h, Nat.mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    n k : Nat
    h : GE.ge n k
    ⊢ Eq (n.descFactorial k) (HMul.hMul k.factorial (n.choose k))
  -/
  rw [Nat.mul_comm]
  /-
    case inr
    n k : Nat
    h : GE.ge n k
    ⊢ Eq (n.descFactorial k) (HMul.hMul (n.choose k) k.factorial)
  -/
  apply Nat.mul_right_cancel (n - k).factorial_pos
  /-
    case inr
    n k : Nat
    h : GE.ge n k
    ⊢ Eq (HMul.hMul (n.descFactorial k) (HSub.hSub n k).factorial) (HMul.hMul (HMu …
  -/
  rw [choose_mul_factorial_mul_factorial h, ← factorial_mul_descFactorial h, Nat.mul_comm]
  /-
    🎉 no goals
  -/


theorem factorial_dvd_descFactorial (n k : ℕ) : k ! ∣ n.descFactorial k :=
  ⟨n.choose k, descFactorial_eq_factorial_mul_choose _ _⟩


theorem choose_eq_descFactorial_div_factorial (n k : ℕ) : n.choose k = n.descFactorial k / k ! :=
  Nat.eq_div_of_mul_eq_right k.factorial_ne_zero (descFactorial_eq_factorial_mul_choose _ _).symm


/-- A faster implementation of `choose`, to be used during bytecode evaluation
and in compiled code. -/
def fast_choose n k := Nat.descFactorial n k / Nat.factorial k


@[csimp] lemma choose_eq_fast_choose : Nat.choose = fast_choose :=
  funext (fun _ => funext (Nat.choose_eq_descFactorial_div_factorial _))



/-- Show that `Nat.choose` is increasing for small values of the right argument. -/
theorem choose_le_succ_of_lt_half_left {r n : ℕ} (h : r < n / 2) :
    choose n r ≤ choose n (r + 1) := by
  /-
    r n : Nat
    h : LT.lt r (HDiv.hDiv n 2)
    ⊢ LE.le (n.choose r) (n.choose (HAdd.hAdd r 1))
  -/
  refine Nat.le_of_mul_le_mul_right ?_ (Nat.sub_pos_of_lt (h.trans_le (n.div_le_self 2)))
  /-
    r n : Nat
    h : LT.lt r (HDiv.hDiv n 2)
    ⊢ LE.le (HMul.hMul (n.choose r) (HSub.hSub n r)) (HMul.hMul (n.choose (HAdd.hA …
  -/
  rw [← choose_succ_right_eq]
  /-
    r n : Nat
    h : LT.lt r (HDiv.hDiv n 2)
    ⊢ LE.le (HMul.hMul (n.choose (HAdd.hAdd r 1)) (HAdd.hAdd r 1)) (HMul.hMul (n.c …
  -/
  apply Nat.mul_le_mul_left
  /-
    case h
    r n : Nat
    h : LT.lt r (HDiv.hDiv n 2)
    ⊢ LE.le (HAdd.hAdd r 1) (HSub.hSub n r)
  -/
  rw [← Nat.lt_iff_add_one_le, Nat.lt_sub_iff_add_lt, ← Nat.mul_two]
  /-
    case h
    r n : Nat
    h : LT.lt r (HDiv.hDiv n 2)
    ⊢ LT.lt (HMul.hMul r 2) n
  -/
  exact lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right h Nat.zero_lt_two) (n.div_mul_le_self 2)
  /-
    🎉 no goals
  -/


/-- Show that for small values of the right argument, the middle value is largest. -/
private theorem choose_le_middle_of_le_half_left {n r : ℕ} (hr : r ≤ n / 2) :
    choose n r ≤ choose n (n / 2) := by
  induction hr using decreasingInduction with
  | self => rfl
  | of_succ k hk ih => exact (choose_le_succ_of_lt_half_left hk).trans ih


/-- `choose n r` is maximised when `r` is `n/2`. -/
theorem choose_le_middle (r n : ℕ) : choose n r ≤ choose n (n / 2) := by
  /-
    r n : Nat
    ⊢ LE.le (n.choose r) (n.choose (HDiv.hDiv n 2))
  -/
  cases' le_or_gt r n with b b
    /-
      case inl
      r n : Nat
      b : LE.le r n
      ⊢ LE.le (n.choose r) (n.choose (HDiv.hDiv n 2))
    -/
  · rcases le_or_lt r (n / 2) with a | h
      /-
        case inl.inl
        r n : Nat
        b : LE.le r n
        a : LE.le r (HDiv.hDiv n 2)
        ⊢ LE.le (n.choose r) (n.choose (HDiv.hDiv n 2))
      -/
    · apply choose_le_middle_of_le_half_left a
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        r n : Nat
        b : LE.le r n
        h : LT.lt (HDiv.hDiv n 2) r
        ⊢ LE.le (n.choose r) (n.choose (HDiv.hDiv n 2))
      -/
    · rw [← choose_symm b]
      /-
        case inl.inr
        r n : Nat
        b : LE.le r n
        h : LT.lt (HDiv.hDiv n 2) r
        ⊢ LE.le (n.choose (HSub.hSub n r)) (n.choose (HDiv.hDiv n 2))
      -/
      apply choose_le_middle_of_le_half_left
      /-
        case inl.inr.hr
        r n : Nat
        b : LE.le r n
        h : LT.lt (HDiv.hDiv n 2) r
        ⊢ LE.le (HSub.hSub n r) (HDiv.hDiv n 2)
      -/
      rw [div_lt_iff_lt_mul Nat.zero_lt_two] at h
      rw [le_div_iff_mul_le Nat.zero_lt_two, Nat.mul_sub_right_distrib, Nat.sub_le_iff_le_add,
        ← Nat.sub_le_iff_le_add', Nat.mul_two, Nat.add_sub_cancel]
      /-
        case inl.inr.hr
        r n : Nat
        b : LE.le r n
        h : LT.lt n (HMul.hMul r 2)
        ⊢ LE.le n (HMul.hMul r 2)
      -/
      exact le_of_lt h
      /-
        🎉 no goals
      -/
    /-
      case inr
      r n : Nat
      b : GT.gt r n
      ⊢ LE.le (n.choose r) (n.choose (HDiv.hDiv n 2))
    -/
  · rw [choose_eq_zero_of_lt b]
    /-
      case inr
      r n : Nat
      b : GT.gt r n
      ⊢ LE.le 0 (n.choose (HDiv.hDiv n 2))
    -/
    apply zero_le
    /-
      🎉 no goals
    -/


theorem choose_le_succ (a c : ℕ) : choose a c ≤ choose a.succ c := by
  /-
    a c : Nat
    ⊢ LE.le (a.choose c) (a.succ.choose c)
  -/
              /-
                🎉 no goals
              -/
  cases c <;> simp [Nat.choose_succ_succ]
              /-
                🎉 no goals
              -/


theorem choose_le_add (a b c : ℕ) : choose a c ≤ choose (a + b) c := by
  /-
    a b c : Nat
    ⊢ LE.le (a.choose c) ((HAdd.hAdd a b).choose c)
  -/
  induction' b with b_n b_ih
    /-
      case zero
      a c : Nat
      ⊢ LE.le (a.choose c) ((HAdd.hAdd a 0).choose c)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    a c b_n : Nat
    b_ih : LE.le (a.choose c) ((HAdd.hAdd a b_n).choose c)
    ⊢ LE.le (a.choose c) ((HAdd.hAdd a (HAdd.hAdd b_n 1)).choose c)
  -/
  exact le_trans b_ih (choose_le_succ (a + b_n) c)
  /-
    🎉 no goals
  -/


theorem choose_le_choose {a b : ℕ} (c : ℕ) (h : a ≤ b) : choose a c ≤ choose b c :=
  Nat.add_sub_cancel' h ▸ choose_le_add a (b - a) c


theorem choose_mono (b : ℕ) : Monotone fun a => choose a b := fun _ _ => choose_le_choose b


/--
`multichoose n k` is the number of multisets of cardinality `k` from a type of cardinality `n`. -/
def multichoose : ℕ → ℕ → ℕ
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 =>
    multichoose n (k + 1) + multichoose (n + 1) k


@[simp]
                                                                   /-
                                                                     n : Nat
                                                                     ⊢ Eq (n.multichoose 0) 1
                                                                   -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
theorem multichoose_zero_right (n : ℕ) : multichoose n 0 = 1 := by cases n <;> simp [multichoose]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
                                                                        /-
                                                                          k : Nat
                                                                          ⊢ Eq (Nat.multichoose 0 (HAdd.hAdd k 1)) 0
                                                                        -/
theorem multichoose_zero_succ (k : ℕ) : multichoose 0 (k + 1) = 0 := by simp [multichoose]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem multichoose_succ_succ (n k : ℕ) :
    multichoose (n + 1) (k + 1) = multichoose n (k + 1) + multichoose (n + 1) k := by
  /-
    n k : Nat
    ⊢ Eq ((HAdd.hAdd n 1).multichoose (HAdd.hAdd k 1)) (HAdd.hAdd (n.multichoose ( …
  -/
  simp [multichoose]
  /-
    🎉 no goals
  -/


@[simp]
theorem multichoose_one (k : ℕ) : multichoose 1 k = 1 := by
  /-
    k : Nat
    ⊢ Eq (Nat.multichoose 1 k) 1
  -/
  induction' k with k IH; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    k : Nat
    IH : Eq (Nat.multichoose 1 k) 1
    ⊢ Eq (Nat.multichoose 1 (HAdd.hAdd k 1)) 1
  -/
  simp [multichoose_succ_succ 0 k, IH]
  /-
    🎉 no goals
  -/


@[simp]
theorem multichoose_two (k : ℕ) : multichoose 2 k = k + 1 := by
  /-
    k : Nat
    ⊢ Eq (Nat.multichoose 2 k) (HAdd.hAdd k 1)
  -/
  induction' k with k IH; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    k : Nat
    IH : Eq (Nat.multichoose 2 k) (HAdd.hAdd k 1)
    ⊢ Eq (Nat.multichoose 2 (HAdd.hAdd k 1)) (HAdd.hAdd (HAdd.hAdd k 1) 1)
  -/
  rw [multichoose, IH]
  /-
    case succ
    k : Nat
    IH : Eq (Nat.multichoose 2 k) (HAdd.hAdd k 1)
    ⊢ Eq (HAdd.hAdd (Nat.multichoose 1 (HAdd.hAdd k 1)) (HAdd.hAdd k 1)) (HAdd.hAd …
  -/
  simp [Nat.add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem multichoose_one_right (n : ℕ) : multichoose n 1 = n := by
  /-
    n : Nat
    ⊢ Eq (n.multichoose 1) n
  -/
  induction' n with n IH; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    n : Nat
    IH : Eq (n.multichoose 1) n
    ⊢ Eq ((HAdd.hAdd n 1).multichoose 1) (HAdd.hAdd n 1)
  -/
  simp [multichoose_succ_succ n 0, IH]
  /-
    🎉 no goals
  -/


theorem multichoose_eq : ∀ n k : ℕ, multichoose n k = (n + k - 1).choose k
               /-
                 x✝ : Nat
                 ⊢ Eq (x✝.multichoose 0) ((HSub.hSub (HAdd.hAdd x✝ 0) 1).choose 0)
               -/
  | _, 0 => by simp
               /-
                 🎉 no goals
               -/
                   /-
                     k : Nat
                     ⊢ Eq (Nat.multichoose 0 (HAdd.hAdd k 1)) ((HSub.hSub (HAdd.hAdd 0 (HAdd.hAdd k …
                   -/
  | 0, k + 1 => by simp
                   /-
                     🎉 no goals
                   -/
  | n + 1, k + 1 => by
    /-
      n k : Nat
      ⊢ Eq ((HAdd.hAdd n 1).multichoose (HAdd.hAdd k 1)) ((HSub.hSub (HAdd.hAdd (HAd …
    -/
    have : n + (k + 1) < (n + 1) + (k + 1) := Nat.add_lt_add_right (Nat.lt_succ_self _) _
    /-
      n k : Nat
      this : LT.lt (HAdd.hAdd n (HAdd.hAdd k 1)) (HAdd.hAdd (HAdd.hAdd n 1) (HAdd.hA …
      ⊢ Eq ((HAdd.hAdd n 1).multichoose (HAdd.hAdd k 1)) ((HSub.hSub (HAdd.hAdd (HAd …
    -/
    have : (n + 1) + k < (n + 1) + (k + 1) := Nat.add_lt_add_left (Nat.lt_succ_self _) _
    rw [multichoose_succ_succ, Nat.add_comm, Nat.succ_add_sub_one, ← Nat.add_assoc,
      Nat.choose_succ_succ]
    /-
      n k : Nat
      this✝ : LT.lt (HAdd.hAdd n (HAdd.hAdd k 1)) (HAdd.hAdd (HAdd.hAdd n 1) (HAdd.h …
      this : LT.lt (HAdd.hAdd (HAdd.hAdd n 1) k) (HAdd.hAdd (HAdd.hAdd n 1) (HAdd.hA …
      ⊢ Eq (HAdd.hAdd ((HAdd.hAdd n 1).multichoose k) (n.multichoose (HAdd.hAdd k 1) …
    -/
    simp [multichoose_eq n (k+1), multichoose_eq (n+1) k]
    /-
      🎉 no goals
    -/


