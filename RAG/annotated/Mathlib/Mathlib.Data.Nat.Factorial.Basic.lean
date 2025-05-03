/-- `Nat.factorial n` is the factorial of `n`. -/
def factorial : ℕ → ℕ
  | 0 => 1
  | succ n => succ n * factorial n


/-- factorial notation `n!` -/
scoped notation:10000 n "!" => Nat.factorial n


@[simp] theorem factorial_zero : 0! = 1 :=
  rfl


theorem factorial_succ (n : ℕ) : (n + 1)! = (n + 1) * n ! :=
  rfl



@[simp] theorem factorial_one : 1! = 1 :=
  rfl


@[simp] theorem factorial_two : 2! = 2 :=
  rfl


theorem mul_factorial_pred (hn : 0 < n) : n * (n - 1)! = n ! :=
  Nat.sub_add_cancel (Nat.succ_le_of_lt hn) ▸ rfl


theorem factorial_pos : ∀ n, 0 < n !
  | 0 => Nat.zero_lt_one
  | succ n => Nat.mul_pos (succ_pos _) (factorial_pos n)


theorem factorial_ne_zero (n : ℕ) : n ! ≠ 0 :=
  ne_of_gt (factorial_pos _)


theorem factorial_dvd_factorial {m n} (h : m ≤ n) : m ! ∣ n ! := by
  induction h with
  | refl => exact Nat.dvd_refl _
  | step _ ih => exact Nat.dvd_trans ih (Nat.dvd_mul_left _ _)


theorem dvd_factorial : ∀ {m n}, 0 < m → m ≤ n → m ∣ n !
  | succ _, _, _, h => Nat.dvd_trans (Nat.dvd_mul_right _ _) (factorial_dvd_factorial h)


@[mono, gcongr]
theorem factorial_le {m n} (h : m ≤ n) : m ! ≤ n ! :=
  le_of_dvd (factorial_pos _) (factorial_dvd_factorial h)


theorem factorial_mul_pow_le_factorial : ∀ {m n : ℕ}, m ! * (m + 1) ^ n ≤ (m + n)!
               /-
                 m : Nat
                 ⊢ LE.le (HMul.hMul m.factorial (HPow.hPow (HAdd.hAdd m 1) 0)) (HAdd.hAdd m 0). …
               -/
  | m, 0 => by simp
               /-
                 🎉 no goals
               -/
  | m, n + 1 => by
    /-
      m n : Nat
      ⊢ LE.le (HMul.hMul m.factorial (HPow.hPow (HAdd.hAdd m 1) (HAdd.hAdd n 1))) (H …
    -/
    rw [← Nat.add_assoc, factorial_succ, Nat.mul_comm (_ + 1), Nat.pow_succ, ← Nat.mul_assoc]
    /-
      m n : Nat
      ⊢ LE.le (HMul.hMul (HMul.hMul m.factorial (HPow.hPow (HAdd.hAdd m 1) n)) (HAdd …
    -/
    exact Nat.mul_le_mul factorial_mul_pow_le_factorial (succ_le_succ (le_add_right _ _))
    /-
      🎉 no goals
    -/


theorem factorial_lt (hn : 0 < n) : n ! < m ! ↔ n < m := by
  /-
    m n : Nat
    hn : LT.lt 0 n
    ⊢ Iff (LT.lt n.factorial m.factorial) (LT.lt n m)
  -/
  refine ⟨fun h => not_le.mp fun hmn => Nat.not_le_of_lt h (factorial_le hmn), fun h => ?_⟩
  have : ∀ {n}, 0 < n → n ! < (n + 1)! := by
    intro k hk
    rw [factorial_succ, succ_mul, Nat.lt_add_left_iff_pos]
    exact Nat.mul_pos hk k.factorial_pos
  induction h generalizing hn with
  | refl => exact this hn
  | step hnk ih => exact lt_trans (ih hn) <| this <| lt_trans hn <| lt_of_succ_le hnk


@[gcongr]
lemma factorial_lt_of_lt {m n : ℕ} (hn : 0 < n) (h : n < m) : n ! < m ! := (factorial_lt hn).mpr h


@[simp] lemma one_lt_factorial : 1 < n ! ↔ 1 < n := factorial_lt Nat.one_pos


@[simp]
theorem factorial_eq_one : n ! = 1 ↔ n ≤ 1 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.factorial 1) (LE.le n 1)
  -/
  constructor
    /-
      case mp
      n : Nat
      ⊢ Eq n.factorial 1 → LE.le n 1
    -/
  · intro h
    /-
      case mp
      n : Nat
      h : Eq n.factorial 1
      ⊢ LE.le n 1
    -/
    rw [← not_lt, ← one_lt_factorial, h]
    /-
      case mp
      n : Nat
      h : Eq n.factorial 1
      ⊢ Not (LT.lt 1 1)
    -/
    apply lt_irrefl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      ⊢ LE.le n 1 → Eq n.factorial 1
    -/
                       /-
                         🎉 no goals
                       -/
  · rintro (_|_|_) <;> rfl
                       /-
                         🎉 no goals
                       -/


theorem factorial_inj (hn : 1 < n) : n ! = m ! ↔ n = m := by
  /-
    m n : Nat
    hn : LT.lt 1 n
    ⊢ Iff (Eq n.factorial m.factorial) (Eq n m)
  -/
  refine ⟨fun h => ?_, congr_arg _⟩
  /-
    m n : Nat
    hn : LT.lt 1 n
    h : Eq n.factorial m.factorial
    ⊢ Eq n m
  -/
  obtain hnm | rfl | hnm := lt_trichotomy n m
    /-
      case inl
      m n : Nat
      hn : LT.lt 1 n
      h : Eq n.factorial m.factorial
      hnm : LT.lt n m
      ⊢ Eq n m
    -/
  · rw [← factorial_lt <| lt_of_succ_lt hn, h] at hnm
    /-
      case inl
      m n : Nat
      hn : LT.lt 1 n
      h : Eq n.factorial m.factorial
      hnm : LT.lt m.factorial m.factorial
      ⊢ Eq n m
    -/
    cases lt_irrefl _ hnm
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      n : Nat
      hn : LT.lt 1 n
      h : Eq n.factorial n.factorial
      ⊢ Eq n n
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    m n : Nat
    hn : LT.lt 1 n
    h : Eq n.factorial m.factorial
    hnm : LT.lt m n
    ⊢ Eq n m
  -/
  rw [← one_lt_factorial, h, one_lt_factorial] at hn
  /-
    case inr.inr
    m n : Nat
    hn : LT.lt 1 m
    h : Eq n.factorial m.factorial
    hnm : LT.lt m n
    ⊢ Eq n m
  -/
  rw [← factorial_lt <| lt_of_succ_lt hn, h] at hnm
  /-
    case inr.inr
    m n : Nat
    hn : LT.lt 1 m
    h : Eq n.factorial m.factorial
    hnm : LT.lt m.factorial m.factorial
    ⊢ Eq n m
  -/
  cases lt_irrefl _ hnm
  /-
    🎉 no goals
  -/


theorem factorial_inj' (h : 1 < n ∨ 1 < m) : n ! = m ! ↔ n = m := by
  /-
    m n : Nat
    h : Or (LT.lt 1 n) (LT.lt 1 m)
    ⊢ Iff (Eq n.factorial m.factorial) (Eq n m)
  -/
  obtain hn|hm := h
    /-
      case inl
      m n : Nat
      hn : LT.lt 1 n
      ⊢ Iff (Eq n.factorial m.factorial) (Eq n m)
    -/
  · exact factorial_inj hn
    /-
      🎉 no goals
    -/
    /-
      case inr
      m n : Nat
      hm : LT.lt 1 m
      ⊢ Iff (Eq n.factorial m.factorial) (Eq n m)
    -/
  · rw [eq_comm, factorial_inj hm, eq_comm]
    /-
      🎉 no goals
    -/


theorem self_le_factorial : ∀ n : ℕ, n ≤ n !
  | 0 => Nat.zero_le _
  | k + 1 => Nat.le_mul_of_pos_right _ (Nat.one_le_of_lt k.factorial_pos)


theorem lt_factorial_self {n : ℕ} (hi : 3 ≤ n) : n < n ! := by
  /-
    n : Nat
    hi : LE.le 3 n
    ⊢ LT.lt n n.factorial
  -/
  have : 0 < n := by omega
  /-
    n : Nat
    hi : LE.le 3 n
    this : LT.lt 0 n
    ⊢ LT.lt n n.factorial
  -/
  have hn : 1 < pred n := le_pred_of_lt (succ_le_iff.mp hi)
  /-
    n : Nat
    hi : LE.le 3 n
    this : LT.lt 0 n
    hn : LT.lt 1 n.pred
    ⊢ LT.lt n n.factorial
  -/
  rw [← succ_pred_eq_of_pos ‹0 < n›, factorial_succ]
  exact (Nat.lt_mul_iff_one_lt_right (pred n).succ_pos).2
    ((Nat.lt_of_lt_of_le hn (self_le_factorial _)))


theorem add_factorial_succ_lt_factorial_add_succ {i : ℕ} (n : ℕ) (hi : 2 ≤ i) :
    i + (n + 1)! < (i + n + 1)! := by
  /-
    i n : Nat
    hi : LE.le 2 i
    ⊢ LT.lt (HAdd.hAdd i (HAdd.hAdd n 1).factorial) (HAdd.hAdd (HAdd.hAdd i n) 1). …
  -/
  rw [factorial_succ (i + _), Nat.add_mul, Nat.one_mul]
  /-
    i n : Nat
    hi : LE.le 2 i
    ⊢ LT.lt (HAdd.hAdd i (HAdd.hAdd n 1).factorial) (HAdd.hAdd (HMul.hMul (HAdd.hA …
  -/
  have := (i + n).self_le_factorial
  refine Nat.add_lt_add_of_lt_of_le (Nat.lt_of_le_of_lt ?_ ((Nat.lt_mul_iff_one_lt_right ?_).2 ?_))
                          /-
                            case refine_1
                            i n : Nat
                            hi : LE.le 2 i
                            this : LE.le (HAdd.hAdd i n) (HAdd.hAdd i n).factorial
                            ⊢ LE.le i (HAdd.hAdd i n)
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
    (factorial_le ?_) <;> omega
                          /-
                            🎉 no goals
                          -/


theorem add_factorial_lt_factorial_add {i n : ℕ} (hi : 2 ≤ i) (hn : 1 ≤ n) :
    i + n ! < (i + n)! := by
  /-
    i n : Nat
    hi : LE.le 2 i
    hn : LE.le 1 n
    ⊢ LT.lt (HAdd.hAdd i n.factorial) (HAdd.hAdd i n).factorial
  -/
  cases hn
    /-
      case refl
      i : Nat
      hi : LE.le 2 i
      ⊢ LT.lt (HAdd.hAdd i (Nat.factorial 1)) (HAdd.hAdd i 1).factorial
    -/
  · rw [factorial_one]
    /-
      case refl
      i : Nat
      hi : LE.le 2 i
      ⊢ LT.lt (HAdd.hAdd i 1) (HAdd.hAdd i 1).factorial
    -/
    exact lt_factorial_self (succ_le_succ hi)
    /-
      🎉 no goals
    -/
  /-
    case step
    i : Nat
    hi : LE.le 2 i
    m✝ : Nat
    a✝ : Nat.le 1 m✝
    ⊢ LT.lt (HAdd.hAdd i m✝.succ.factorial) (HAdd.hAdd i m✝.succ).factorial
  -/
  exact add_factorial_succ_lt_factorial_add_succ _ hi
  /-
    🎉 no goals
  -/


theorem add_factorial_succ_le_factorial_add_succ (i : ℕ) (n : ℕ) :
    i + (n + 1)! ≤ (i + (n + 1))! := by
  /-
    i n : Nat
    ⊢ LE.le (HAdd.hAdd i (HAdd.hAdd n 1).factorial) (HAdd.hAdd i (HAdd.hAdd n 1)). …
  -/
  cases (le_or_lt (2 : ℕ) i)
    /-
      case inl
      i n : Nat
      h✝ : LE.le 2 i
      ⊢ LE.le (HAdd.hAdd i (HAdd.hAdd n 1).factorial) (HAdd.hAdd i (HAdd.hAdd n 1)). …
    -/
  · rw [← Nat.add_assoc]
    /-
      case inl
      i n : Nat
      h✝ : LE.le 2 i
      ⊢ LE.le (HAdd.hAdd i (HAdd.hAdd n 1).factorial) (HAdd.hAdd (HAdd.hAdd i n) 1). …
    -/
    apply Nat.le_of_lt
    /-
      case inl.a
      i n : Nat
      h✝ : LE.le 2 i
      ⊢ LT.lt (HAdd.hAdd i (HAdd.hAdd n 1).factorial) (HAdd.hAdd (HAdd.hAdd i n) 1). …
    -/
    apply add_factorial_succ_lt_factorial_add_succ
    /-
      case inl.a.hi
      i n : Nat
      h✝ : LE.le 2 i
      ⊢ LE.le 2 i
    -/
    assumption
    /-
      🎉 no goals
    -/
  · match i with
    | 0 => simp
    | 1 =>
      rw [← Nat.add_assoc, factorial_succ (1 + n), Nat.add_mul, Nat.one_mul, Nat.add_comm 1 n,
        Nat.add_le_add_iff_right]
      exact Nat.mul_pos n.succ_pos n.succ.factorial_pos
    | succ (succ n) => contradiction


theorem add_factorial_le_factorial_add (i : ℕ) {n : ℕ} (n1 : 1 ≤ n) : i + n ! ≤ (i + n)! := by
  /-
    i n : Nat
    n1 : LE.le 1 n
    ⊢ LE.le (HAdd.hAdd i n.factorial) (HAdd.hAdd i n).factorial
  -/
  cases' n1 with h
    /-
      case refl
      i : Nat
      ⊢ LE.le (HAdd.hAdd i (Nat.factorial 1)) (HAdd.hAdd i 1).factorial
    -/
  · exact self_le_factorial _
    /-
      🎉 no goals
    -/
  /-
    case step
    i h : Nat
    a✝ : Nat.le 1 h
    ⊢ LE.le (HAdd.hAdd i h.succ.factorial) (HAdd.hAdd i h.succ).factorial
  -/
  exact add_factorial_succ_le_factorial_add_succ i h
  /-
    🎉 no goals
  -/


theorem factorial_mul_pow_sub_le_factorial {n m : ℕ} (hnm : n ≤ m) : n ! * n ^ (m - n) ≤ m ! := by
  calc
    _ ≤ n ! * (n + 1) ^ (m - n) := Nat.mul_le_mul_left _ (Nat.pow_le_pow_left n.le_succ _)
    _ ≤ _ := by simpa [hnm] using @Nat.factorial_mul_pow_le_factorial n (m - n)


lemma factorial_le_pow : ∀ n, n ! ≤ n ^ n
  | 0 => le_refl _
  | n + 1 =>
    calc
      _ ≤ (n + 1) * n ^ n := Nat.mul_le_mul_left _ n.factorial_le_pow
      _ ≤ (n + 1) * (n + 1) ^ n := Nat.mul_le_mul_left _ (Nat.pow_le_pow_left n.le_succ _)
                  /-
                    n : Nat
                    ⊢ Eq (HMul.hMul (HAdd.hAdd n 1) (HPow.hPow (HAdd.hAdd n 1) n)) (HPow.hPow (HAd …
                  -/
      _ = _ := by rw [pow_succ']
                  /-
                    🎉 no goals
                  -/


/-- `n.ascFactorial k = n (n + 1) ⋯ (n + k - 1)`. This is closely related to `ascPochhammer`, but
much less general. -/
def ascFactorial (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | k + 1 => (n + k) * ascFactorial n k


@[simp]
theorem ascFactorial_zero (n : ℕ) : n.ascFactorial 0 = 1 :=
  rfl


theorem ascFactorial_succ {n k : ℕ} : n.ascFactorial k.succ = (n + k) * n.ascFactorial k :=
  rfl


theorem zero_ascFactorial : ∀ (k : ℕ), (0 : ℕ).ascFactorial k.succ = 0
  | 0 => by
    /-
      ⊢ Eq (Nat.ascFactorial 0 (Nat.succ 0)) 0
    -/
    rw [ascFactorial_succ, ascFactorial_zero, Nat.zero_add, Nat.zero_mul]
    /-
      🎉 no goals
    -/
  | (k+1) => by
    /-
      k : Nat
      ⊢ Eq (Nat.ascFactorial 0 (HAdd.hAdd k 1).succ) 0
    -/
    rw [ascFactorial_succ, zero_ascFactorial k, Nat.mul_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem one_ascFactorial : ∀ (k : ℕ), (1 : ℕ).ascFactorial k = k.factorial
  | 0 => ascFactorial_zero 1
  | (k+1) => by
    /-
      k : Nat
      ⊢ Eq (Nat.ascFactorial 1 (HAdd.hAdd k 1)) (HAdd.hAdd k 1).factorial
    -/
    rw [ascFactorial_succ, one_ascFactorial k, Nat.add_comm, factorial_succ]
    /-
      🎉 no goals
    -/


theorem succ_ascFactorial (n : ℕ) :
    ∀ k, n * n.succ.ascFactorial k = (n + k) * n.ascFactorial k
            /-
              n : Nat
              ⊢ Eq (HMul.hMul n (n.succ.ascFactorial 0)) (HMul.hMul (HAdd.hAdd n 0) (n.ascFa …
            -/
  | 0 => by rw [Nat.add_zero, ascFactorial_zero, ascFactorial_zero]
            /-
              🎉 no goals
            -/
  | k + 1 => by rw [ascFactorial, Nat.mul_left_comm, succ_ascFactorial n k, ascFactorial, succ_add,
    ← Nat.add_assoc]


/-- `(n + 1).ascFactorial k = (n + k) ! / n !` but without ℕ-division. See
`Nat.ascFactorial_eq_div` for the version with ℕ-division. -/
theorem factorial_mul_ascFactorial (n : ℕ) : ∀ k, n ! * (n + 1).ascFactorial k = (n + k)!
            /-
              n : Nat
              ⊢ Eq (HMul.hMul n.factorial ((HAdd.hAdd n 1).ascFactorial 0)) (HAdd.hAdd n 0). …
            -/
  | 0 => by rw [ascFactorial_zero, Nat.add_zero, Nat.mul_one]
            /-
              🎉 no goals
            -/
  | k + 1 => by
    rw [ascFactorial_succ, ← Nat.add_assoc, factorial_succ, Nat.mul_comm (n + 1 + k),
      ← Nat.mul_assoc, factorial_mul_ascFactorial n k, Nat.mul_comm, Nat.add_right_comm]


/-- `n.ascFactorial k = (n + k - 1)! / (n - 1)!` for `n > 0` but without ℕ-division. See
`Nat.ascFactorial_eq_div` for the version with ℕ-division. Consider using
`factorial_mul_ascFactorial` to avoid complications of ℕ-subtraction. -/
theorem factorial_mul_ascFactorial' (n k : ℕ) (h : 0 < n) :
    (n - 1) ! * n.ascFactorial k = (n + k - 1)! := by
  /-
    n k : Nat
    h : LT.lt 0 n
    ⊢ Eq (HMul.hMul (HSub.hSub n 1).factorial (n.ascFactorial k)) (HSub.hSub (HAdd …
  -/
  rw [Nat.sub_add_comm h, Nat.sub_one]
  /-
    n k : Nat
    h : LT.lt 0 n
    ⊢ Eq (HMul.hMul n.pred.factorial (n.ascFactorial k)) (HAdd.hAdd n.pred k).fact …
  -/
  nth_rw 2 [Nat.eq_add_of_sub_eq h rfl]
  /-
    n k : Nat
    h : LT.lt 0 n
    ⊢ Eq (HMul.hMul n.pred.factorial ((HAdd.hAdd (HSub.hSub n (Nat.succ 0)) (Nat.s …
  -/
  rw [Nat.sub_one, factorial_mul_ascFactorial]
  /-
    🎉 no goals
  -/


/-- Avoid in favor of `Nat.factorial_mul_ascFactorial` if you can. ℕ-division isn't worth it. -/
theorem ascFactorial_eq_div (n k : ℕ) : (n + 1).ascFactorial k = (n + k)! / n ! :=
  Nat.eq_div_of_mul_eq_right n.factorial_ne_zero (factorial_mul_ascFactorial _ _)


/-- Avoid in favor of `Nat.factorial_mul_ascFactorial'` if you can. ℕ-division isn't worth it. -/
theorem ascFactorial_eq_div' (n k : ℕ) (h : 0 < n) :
    n.ascFactorial k = (n + k - 1)! / (n - 1) ! :=
  Nat.eq_div_of_mul_eq_right (n - 1).factorial_ne_zero (factorial_mul_ascFactorial' _ _ h)


theorem ascFactorial_of_sub {n k : ℕ} :
    (n - k) * (n - k + 1).ascFactorial k = (n - k).ascFactorial (k + 1) := by
  /-
    n k : Nat
    ⊢ Eq (HMul.hMul (HSub.hSub n k) ((HAdd.hAdd (HSub.hSub n k) 1).ascFactorial k) …
  -/
  rw [succ_ascFactorial, ascFactorial_succ]
  /-
    🎉 no goals
  -/


theorem pow_succ_le_ascFactorial (n : ℕ) : ∀ k : ℕ, n ^ k ≤ n.ascFactorial k
            /-
              n : Nat
              ⊢ LE.le (HPow.hPow n 0) (n.ascFactorial 0)
            -/
  | 0 => by rw [ascFactorial_zero, Nat.pow_zero]
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      n k : Nat
      ⊢ LE.le (HPow.hPow n (HAdd.hAdd k 1)) (n.ascFactorial (HAdd.hAdd k 1))
    -/
    rw [Nat.pow_succ, Nat.mul_comm, ascFactorial_succ, ← succ_ascFactorial]
    exact Nat.mul_le_mul (Nat.le_refl n)
      (Nat.le_trans (Nat.pow_le_pow_left (le_succ n) k) (pow_succ_le_ascFactorial n.succ k))


theorem pow_lt_ascFactorial' (n k : ℕ) : (n + 1) ^ (k + 2) < (n + 1).ascFactorial (k + 2) := by
  /-
    n k : Nat
    ⊢ LT.lt (HPow.hPow (HAdd.hAdd n 1) (HAdd.hAdd k 2)) ((HAdd.hAdd n 1).ascFactor …
  -/
  rw [Nat.pow_succ, ascFactorial, Nat.mul_comm]
  exact Nat.mul_lt_mul_of_lt_of_le' (Nat.lt_add_of_pos_right k.succ_pos)
    (pow_succ_le_ascFactorial n.succ _) (Nat.pow_pos n.succ_pos)


theorem pow_lt_ascFactorial (n : ℕ) : ∀ {k : ℕ}, 2 ≤ k → (n + 1) ^ k < (n + 1).ascFactorial k
            /-
              n : Nat
              ⊢ LE.le 2 0 → LT.lt (HPow.hPow (HAdd.hAdd n 1) 0) ((HAdd.hAdd n 1).ascFactoria …
            -/
  | 0 => by rintro ⟨⟩
            /-
              🎉 no goals
            -/
            /-
              n : Nat
              ⊢ LE.le 2 1 → LT.lt (HPow.hPow (HAdd.hAdd n 1) 1) ((HAdd.hAdd n 1).ascFactoria …
            -/
  | 1 => by intro; contradiction
                   /-
                     🎉 no goals
                   -/
  | k + 2 => fun _ => pow_lt_ascFactorial' n k


theorem ascFactorial_le_pow_add (n : ℕ) : ∀ k : ℕ, (n+1).ascFactorial k ≤ (n + k) ^ k
            /-
              n : Nat
              ⊢ LE.le ((HAdd.hAdd n 1).ascFactorial 0) (HPow.hPow (HAdd.hAdd n 0) 0)
            -/
  | 0 => by rw [ascFactorial_zero, Nat.pow_zero]
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      n k : Nat
      ⊢ LE.le ((HAdd.hAdd n 1).ascFactorial (HAdd.hAdd k 1)) (HPow.hPow (HAdd.hAdd n …
    -/
    rw [ascFactorial_succ, Nat.pow_succ, Nat.mul_comm, ← Nat.add_assoc, Nat.add_right_comm n 1 k]
    exact Nat.mul_le_mul_right _
      (Nat.le_trans (ascFactorial_le_pow_add _ k) (Nat.pow_le_pow_left (le_succ _) _))


theorem ascFactorial_lt_pow_add (n : ℕ) : ∀ {k : ℕ}, 2 ≤ k → (n + 1).ascFactorial k < (n + k) ^ k
            /-
              n : Nat
              ⊢ LE.le 2 0 → LT.lt ((HAdd.hAdd n 1).ascFactorial 0) (HPow.hPow (HAdd.hAdd n 0 …
            -/
  | 0 => by rintro ⟨⟩
            /-
              🎉 no goals
            -/
            /-
              n : Nat
              ⊢ LE.le 2 1 → LT.lt ((HAdd.hAdd n 1).ascFactorial 1) (HPow.hPow (HAdd.hAdd n 1 …
            -/
  | 1 => by intro; contradiction
                   /-
                     🎉 no goals
                   -/
  | k + 2 => fun _ => by
    /-
      n k : Nat
      x✝ : LE.le 2 (HAdd.hAdd k 2)
      ⊢ LT.lt ((HAdd.hAdd n 1).ascFactorial (HAdd.hAdd k 2)) (HPow.hPow (HAdd.hAdd n …
    -/
    rw [Nat.pow_succ, Nat.mul_comm, ascFactorial_succ, succ_add_eq_add_succ n (k + 1)]
    exact Nat.mul_lt_mul_of_le_of_lt (le_refl _) (Nat.lt_of_le_of_lt (ascFactorial_le_pow_add n _)
      (Nat.pow_lt_pow_left (Nat.lt_succ_self _) k.succ_ne_zero)) (succ_pos _)


theorem ascFactorial_pos (n k : ℕ) : 0 < (n + 1).ascFactorial k :=
  Nat.lt_of_lt_of_le (Nat.pow_pos n.succ_pos) (pow_succ_le_ascFactorial (n + 1) k)


/-- `n.descFactorial k = n! / (n - k)!` (as seen in `Nat.descFactorial_eq_div`), but
implemented recursively to allow for "quick" computation when using `norm_num`. This is closely
related to `descPochhammer`, but much less general. -/
def descFactorial (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | k + 1 => (n - k) * descFactorial n k


@[simp]
theorem descFactorial_zero (n : ℕ) : n.descFactorial 0 = 1 :=
  rfl


@[simp]
theorem descFactorial_succ (n k : ℕ) : n.descFactorial (k + 1) = (n - k) * n.descFactorial k :=
  rfl


theorem zero_descFactorial_succ (k : ℕ) : (0 : ℕ).descFactorial (k + 1) = 0 := by
  /-
    k : Nat
    ⊢ Eq (Nat.descFactorial 0 (HAdd.hAdd k 1)) 0
  -/
  rw [descFactorial_succ, Nat.zero_sub, Nat.zero_mul]
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  n : Nat
                                                                  ⊢ Eq (n.descFactorial 1) n
                                                                -/
theorem descFactorial_one (n : ℕ) : n.descFactorial 1 = n := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem succ_descFactorial_succ (n : ℕ) :
    ∀ k : ℕ, (n + 1).descFactorial (k + 1) = (n + 1) * n.descFactorial k
            /-
              n : Nat
              ⊢ Eq ((HAdd.hAdd n 1).descFactorial (HAdd.hAdd 0 1)) (HMul.hMul (HAdd.hAdd n 1 …
            -/
  | 0 => by rw [descFactorial_zero, descFactorial_one, Nat.mul_one]
            /-
              🎉 no goals
            -/
  | succ k => by
    rw [descFactorial_succ, succ_descFactorial_succ _ k, descFactorial_succ, succ_sub_succ,
      Nat.mul_left_comm]


theorem succ_descFactorial (n : ℕ) :
    ∀ k, (n + 1 - k) * (n + 1).descFactorial k = (n + 1) * n.descFactorial k
            /-
              n : Nat
              ⊢ Eq (HMul.hMul (HSub.hSub (HAdd.hAdd n 1) 0) ((HAdd.hAdd n 1).descFactorial 0 …
            -/
  | 0 => by rw [Nat.sub_zero, descFactorial_zero, descFactorial_zero]
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      n k : Nat
      ⊢ Eq (HMul.hMul (HSub.hSub (HAdd.hAdd n 1) (HAdd.hAdd k 1)) ((HAdd.hAdd n 1).d …
    -/
    rw [descFactorial, succ_descFactorial _ k, descFactorial_succ, succ_sub_succ, Nat.mul_left_comm]
    /-
      🎉 no goals
    -/


theorem descFactorial_self : ∀ n : ℕ, n.descFactorial n = n !
            /-
              ⊢ Eq (Nat.descFactorial 0 0) (Nat.factorial 0)
            -/
  | 0 => by rw [descFactorial_zero, factorial_zero]
            /-
              🎉 no goals
            -/
                 /-
                   n : Nat
                   ⊢ Eq (n.succ.descFactorial n.succ) n.succ.factorial
                 -/
  | succ n => by rw [succ_descFactorial_succ, descFactorial_self n, factorial_succ]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem descFactorial_eq_zero_iff_lt {n : ℕ} : ∀ {k : ℕ}, n.descFactorial k = 0 ↔ n < k
            /-
              n : Nat
              ⊢ Iff (Eq (n.descFactorial 0) 0) (LT.lt n 0)
            -/
  | 0 => by simp only [descFactorial_zero, Nat.one_ne_zero, Nat.not_lt_zero]
            /-
              🎉 no goals
            -/
  | succ k => by
    rw [descFactorial_succ, mul_eq_zero, descFactorial_eq_zero_iff_lt, Nat.lt_succ_iff,
      Nat.sub_eq_zero_iff_le, Nat.lt_iff_le_and_ne, or_iff_left_iff_imp, and_imp]
    /-
      n k : Nat
      ⊢ LE.le n k → Ne n k → LE.le n k
    -/
    exact fun h _ => h
    /-
      🎉 no goals
    -/


alias ⟨_, descFactorial_of_lt⟩ := descFactorial_eq_zero_iff_lt


theorem add_descFactorial_eq_ascFactorial (n : ℕ) : ∀ k : ℕ,
    (n + k).descFactorial k = (n + 1).ascFactorial k
            /-
              n : Nat
              ⊢ Eq ((HAdd.hAdd n 0).descFactorial 0) ((HAdd.hAdd n 1).ascFactorial 0)
            -/
  | 0 => by rw [ascFactorial_zero, descFactorial_zero]
            /-
              🎉 no goals
            -/
  | succ k => by
    rw [Nat.add_succ, succ_descFactorial_succ, ascFactorial_succ,
      add_descFactorial_eq_ascFactorial _ k, Nat.add_right_comm]


theorem add_descFactorial_eq_ascFactorial' (n : ℕ) :
    ∀ k : ℕ, (n + k - 1).descFactorial k = n.ascFactorial k
            /-
              n : Nat
              ⊢ Eq ((HSub.hSub (HAdd.hAdd n 0) 1).descFactorial 0) (n.ascFactorial 0)
            -/
  | 0 => by rw [ascFactorial_zero, descFactorial_zero]
            /-
              🎉 no goals
            -/
  | succ k => by
    rw [descFactorial_succ, ascFactorial_succ, ← succ_add_eq_add_succ,
      add_descFactorial_eq_ascFactorial' _ k, ← succ_ascFactorial, succ_add_sub_one,
      Nat.add_sub_cancel]


/-- `n.descFactorial k = n! / (n - k)!` but without ℕ-division. See `Nat.descFactorial_eq_div`
for the version using ℕ-division. -/
theorem factorial_mul_descFactorial : ∀ {n k : ℕ}, k ≤ n → (n - k)! * n.descFactorial k = n !
                        /-
                          n : Nat
                          x✝ : LE.le 0 n
                          ⊢ Eq (HMul.hMul (HSub.hSub n 0).factorial (n.descFactorial 0)) n.factorial
                        -/
  | n, 0 => fun _ => by rw [descFactorial_zero, Nat.mul_one, Nat.sub_zero]
                        /-
                          🎉 no goals
                        -/
  | 0, succ k => fun h => by
    /-
      k : Nat
      h : LE.le k.succ 0
      ⊢ Eq (HMul.hMul (HSub.hSub 0 k.succ).factorial (Nat.descFactorial 0 k.succ)) ( …
    -/
    exfalso
    /-
      k : Nat
      h : LE.le k.succ 0
      ⊢ False
    -/
    exact not_succ_le_zero k h
    /-
      🎉 no goals
    -/
  | succ n, succ k => fun h => by
    rw [succ_descFactorial_succ, succ_sub_succ, ← Nat.mul_assoc, Nat.mul_comm (n - k)!,
      Nat.mul_assoc, factorial_mul_descFactorial (Nat.succ_le_succ_iff.1 h), factorial_succ]


theorem descFactorial_mul_descFactorial {k m n : ℕ} (hkm : k ≤ m) :
    (n - k).descFactorial (m - k) * n.descFactorial k = n.descFactorial m := by
  /-
    k m n : Nat
    hkm : LE.le k m
    ⊢ Eq (HMul.hMul ((HSub.hSub n k).descFactorial (HSub.hSub m k)) (n.descFactori …
  -/
  by_cases hmn : m ≤ n
    /-
      case pos
      k m n : Nat
      hkm : LE.le k m
      hmn : LE.le m n
      ⊢ Eq (HMul.hMul ((HSub.hSub n k).descFactorial (HSub.hSub m k)) (n.descFactori …
    -/
  · apply Nat.mul_left_cancel (n - m).factorial_pos
    rw [factorial_mul_descFactorial hmn, show n - m = (n - k) - (m - k) by omega, ← Nat.mul_assoc,
      factorial_mul_descFactorial (show m - k ≤ n - k by omega),
      factorial_mul_descFactorial (le_trans hkm hmn)]
    /-
      case neg
      k m n : Nat
      hkm : LE.le k m
      hmn : Not (LE.le m n)
      ⊢ Eq (HMul.hMul ((HSub.hSub n k).descFactorial (HSub.hSub m k)) (n.descFactori …
    -/
  · rw [descFactorial_eq_zero_iff_lt.mpr (show n < m by omega)]
    /-
      case neg
      k m n : Nat
      hkm : LE.le k m
      hmn : Not (LE.le m n)
      ⊢ Eq (HMul.hMul ((HSub.hSub n k).descFactorial (HSub.hSub m k)) (n.descFactori …
    -/
    by_cases hkn : k ≤ n
      /-
        case pos
        k m n : Nat
        hkm : LE.le k m
        hmn : Not (LE.le m n)
        hkn : LE.le k n
        ⊢ Eq (HMul.hMul ((HSub.hSub n k).descFactorial (HSub.hSub m k)) (n.descFactori …
      -/
    · rw [descFactorial_eq_zero_iff_lt.mpr (show n - k < m - k by omega), Nat.zero_mul]
      /-
        🎉 no goals
      -/
      /-
        case neg
        k m n : Nat
        hkm : LE.le k m
        hmn : Not (LE.le m n)
        hkn : Not (LE.le k n)
        ⊢ Eq (HMul.hMul ((HSub.hSub n k).descFactorial (HSub.hSub m k)) (n.descFactori …
      -/
    · rw [descFactorial_eq_zero_iff_lt.mpr (show n < k by omega), Nat.mul_zero]
      /-
        🎉 no goals
      -/


/-- Avoid in favor of `Nat.factorial_mul_descFactorial` if you can. ℕ-division isn't worth it. -/
theorem descFactorial_eq_div {n k : ℕ} (h : k ≤ n) : n.descFactorial k = n ! / (n - k)! := by
  /-
    n k : Nat
    h : LE.le k n
    ⊢ Eq (n.descFactorial k) (HDiv.hDiv n.factorial (HSub.hSub n k).factorial)
  -/
  apply Nat.mul_left_cancel (n - k).factorial_pos
  /-
    n k : Nat
    h : LE.le k n
    ⊢ Eq (HMul.hMul (HSub.hSub n k).factorial (n.descFactorial k)) (HMul.hMul (HSu …
  -/
  rw [factorial_mul_descFactorial h]
  /-
    n k : Nat
    h : LE.le k n
    ⊢ Eq n.factorial (HMul.hMul (HSub.hSub n k).factorial (HDiv.hDiv n.factorial ( …
  -/
  exact (Nat.mul_div_cancel' <| factorial_dvd_factorial <| Nat.sub_le n k).symm
  /-
    🎉 no goals
  -/


theorem descFactorial_le (n : ℕ) {k m : ℕ} (h : k ≤ m) :
    k.descFactorial n ≤ m.descFactorial n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [descFactorial_succ, descFactorial_succ]
    exact Nat.mul_le_mul (Nat.sub_le_sub_right h n) ih


theorem pow_sub_le_descFactorial (n : ℕ) : ∀ k : ℕ, (n + 1 - k) ^ k ≤ n.descFactorial k
            /-
              n : Nat
              ⊢ LE.le (HPow.hPow (HSub.hSub (HAdd.hAdd n 1) 0) 0) (n.descFactorial 0)
            -/
  | 0 => by rw [descFactorial_zero, Nat.pow_zero]
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      n k : Nat
      ⊢ LE.le (HPow.hPow (HSub.hSub (HAdd.hAdd n 1) (HAdd.hAdd k 1)) (HAdd.hAdd k 1) …
    -/
    rw [descFactorial_succ, Nat.pow_succ, succ_sub_succ, Nat.mul_comm]
    /-
      n k : Nat
      ⊢ LE.le (HMul.hMul (HSub.hSub n k) (HPow.hPow (HSub.hSub n k) k)) (HMul.hMul ( …
    -/
    apply Nat.mul_le_mul_left
    exact (le_trans (Nat.pow_le_pow_left (Nat.sub_le_sub_right n.le_succ _) k)
      (pow_sub_le_descFactorial n k))


theorem pow_sub_lt_descFactorial' {n : ℕ} :
    ∀ {k : ℕ}, k + 2 ≤ n → (n - (k + 1)) ^ (k + 2) < n.descFactorial (k + 2)
  | 0, h => by
    /-
      n : Nat
      h : LE.le (HAdd.hAdd 0 2) n
      ⊢ LT.lt (HPow.hPow (HSub.hSub n (HAdd.hAdd 0 1)) (HAdd.hAdd 0 2)) (n.descFacto …
    -/
    rw [descFactorial_succ, Nat.pow_succ, Nat.pow_one, descFactorial_one]
    /-
      n : Nat
      h : LE.le (HAdd.hAdd 0 2) n
      ⊢ LT.lt (HMul.hMul (HSub.hSub n (HAdd.hAdd 0 1)) (HSub.hSub n (HAdd.hAdd 0 1)) …
    -/
    exact Nat.mul_lt_mul_of_pos_left (by omega) (Nat.sub_pos_of_lt h)
    /-
      🎉 no goals
    -/
  | k + 1, h => by
    /-
      n k : Nat
      h : LE.le (HAdd.hAdd (HAdd.hAdd k 1) 2) n
      ⊢ LT.lt (HPow.hPow (HSub.hSub n (HAdd.hAdd (HAdd.hAdd k 1) 1)) (HAdd.hAdd (HAd …
    -/
    rw [descFactorial_succ, Nat.pow_succ, Nat.mul_comm]
    /-
      n k : Nat
      h : LE.le (HAdd.hAdd (HAdd.hAdd k 1) 2) n
      ⊢ LT.lt (HMul.hMul (HSub.hSub n (HAdd.hAdd (HAdd.hAdd k 1) 1)) (HPow.hPow (HSu …
    -/
    refine Nat.mul_lt_mul_of_pos_left ?_ (Nat.sub_pos_of_lt h)
    /-
      n k : Nat
      h : LE.le (HAdd.hAdd (HAdd.hAdd k 1) 2) n
      ⊢ LT.lt (HPow.hPow (HSub.hSub n (HAdd.hAdd (HAdd.hAdd k 1) 1)) (HAdd.hAdd k 2) …
    -/
    refine Nat.lt_of_le_of_lt (Nat.pow_le_pow_left (Nat.sub_le_sub_right n.le_succ _) _) ?_
    /-
      n k : Nat
      h : LE.le (HAdd.hAdd (HAdd.hAdd k 1) 2) n
      ⊢ LT.lt (HPow.hPow (HSub.hSub n.succ (HAdd.hAdd (HAdd.hAdd k 1) 1)) (HAdd.hAdd …
    -/
    rw [succ_sub_succ]
    /-
      n k : Nat
      h : LE.le (HAdd.hAdd (HAdd.hAdd k 1) 2) n
      ⊢ LT.lt (HPow.hPow (HSub.hSub n (HAdd.hAdd k 1)) (HAdd.hAdd k 2)) (n.descFacto …
    -/
    exact pow_sub_lt_descFactorial' (Nat.le_trans (le_succ _) h)
    /-
      🎉 no goals
    -/


theorem pow_sub_lt_descFactorial {n : ℕ} :
    ∀ {k : ℕ}, 2 ≤ k → k ≤ n → (n + 1 - k) ^ k < n.descFactorial k
            /-
              n : Nat
              ⊢ LE.le 2 0 → LE.le 0 n → LT.lt (HPow.hPow (HSub.hSub (HAdd.hAdd n 1) 0) 0) (n …
            -/
  | 0 => by rintro ⟨⟩
            /-
              🎉 no goals
            -/
            /-
              n : Nat
              ⊢ LE.le 2 1 → LE.le 1 n → LT.lt (HPow.hPow (HSub.hSub (HAdd.hAdd n 1) 1) 1) (n …
            -/
  | 1 => by intro; contradiction
                   /-
                     🎉 no goals
                   -/
  | k + 2 => fun _ h => by
    /-
      n k : Nat
      x✝ : LE.le 2 (HAdd.hAdd k 2)
      h : LE.le (HAdd.hAdd k 2) n
      ⊢ LT.lt (HPow.hPow (HSub.hSub (HAdd.hAdd n 1) (HAdd.hAdd k 2)) (HAdd.hAdd k 2) …
    -/
    rw [succ_sub_succ]
    /-
      n k : Nat
      x✝ : LE.le 2 (HAdd.hAdd k 2)
      h : LE.le (HAdd.hAdd k 2) n
      ⊢ LT.lt (HPow.hPow (HSub.hSub n (HAdd.hAdd k 1)) (HAdd.hAdd k 2)) (n.descFacto …
    -/
    exact pow_sub_lt_descFactorial' h
    /-
      🎉 no goals
    -/


theorem descFactorial_le_pow (n : ℕ) : ∀ k : ℕ, n.descFactorial k ≤ n ^ k
            /-
              n : Nat
              ⊢ LE.le (n.descFactorial 0) (HPow.hPow n 0)
            -/
  | 0 => by rw [descFactorial_zero, Nat.pow_zero]
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      n k : Nat
      ⊢ LE.le (n.descFactorial (HAdd.hAdd k 1)) (HPow.hPow n (HAdd.hAdd k 1))
    -/
    rw [descFactorial_succ, Nat.pow_succ, Nat.mul_comm _ n]
    /-
      n k : Nat
      ⊢ LE.le (HMul.hMul (HSub.hSub n k) (n.descFactorial k)) (HMul.hMul n (HPow.hPo …
    -/
    exact Nat.mul_le_mul (Nat.sub_le _ _) (descFactorial_le_pow _ k)
    /-
      🎉 no goals
    -/


theorem descFactorial_lt_pow {n : ℕ} (hn : 1 ≤ n) : ∀ {k : ℕ}, 2 ≤ k → n.descFactorial k < n ^ k
            /-
              n : Nat
              hn : LE.le 1 n
              ⊢ LE.le 2 0 → LT.lt (n.descFactorial 0) (HPow.hPow n 0)
            -/
  | 0 => by rintro ⟨⟩
            /-
              🎉 no goals
            -/
            /-
              n : Nat
              hn : LE.le 1 n
              ⊢ LE.le 2 1 → LT.lt (n.descFactorial 1) (HPow.hPow n 1)
            -/
  | 1 => by intro; contradiction
                   /-
                     🎉 no goals
                   -/
  | k + 2 => fun _ => by
    /-
      n : Nat
      hn : LE.le 1 n
      k : Nat
      x✝ : LE.le 2 (HAdd.hAdd k 2)
      ⊢ LT.lt (n.descFactorial (HAdd.hAdd k 2)) (HPow.hPow n (HAdd.hAdd k 2))
    -/
    rw [descFactorial_succ, pow_succ', Nat.mul_comm, Nat.mul_comm n]
    exact Nat.mul_lt_mul_of_le_of_lt (descFactorial_le_pow _ _) (Nat.sub_lt hn k.zero_lt_succ)
      (Nat.pow_pos (Nat.lt_of_succ_le hn))


lemma factorial_two_mul_le (n : ℕ) : (2 * n)! ≤ (2 * n) ^ n * n ! := by
  /-
    n : Nat
    ⊢ LE.le (HMul.hMul 2 n).factorial (HMul.hMul (HPow.hPow (HMul.hMul 2 n) n) n.f …
  -/
  rw [Nat.two_mul, ← factorial_mul_ascFactorial, Nat.mul_comm]
  /-
    n : Nat
    ⊢ LE.le (HMul.hMul ((HAdd.hAdd n 1).ascFactorial n) n.factorial) (HMul.hMul (H …
  -/
  exact Nat.mul_le_mul_right _ (ascFactorial_le_pow_add _ _)
  /-
    🎉 no goals
  -/


lemma two_pow_mul_factorial_le_factorial_two_mul (n : ℕ) : 2 ^ n * n ! ≤ (2 * n) ! := by
  /-
    n : Nat
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 n) n.factorial) (HMul.hMul 2 n).factorial
  -/
  obtain _ | n := n
    /-
      case zero
      ⊢ LE.le (HMul.hMul (HPow.hPow 2 0) (Nat.factorial 0)) (HMul.hMul 2 0).factorial
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 (HAdd.hAdd n 1)) (HAdd.hAdd n 1).factorial) (H …
  -/
  rw [Nat.mul_comm, Nat.two_mul]
  calc
    _ ≤ (n + 1)! * (n + 2) ^ (n + 1) :=
      Nat.mul_le_mul_left _ (pow_le_pow_of_le_left (le_add_left _ _) _)
    _ ≤ _ := Nat.factorial_mul_pow_le_factorial


