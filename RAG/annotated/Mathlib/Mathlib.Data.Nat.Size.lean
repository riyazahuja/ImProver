theorem shiftLeft_eq_mul_pow (m) : ∀ n, m <<< n = m * 2 ^ n := shiftLeft_eq _


theorem shiftLeft'_tt_eq_mul_pow (m) : ∀ n, shiftLeft' true m n + 1 = (m + 1) * 2 ^ n
            /-
              m : Nat
              ⊢ Eq (HAdd.hAdd (Nat.shiftLeft' Bool.true m 0) 1) (HMul.hMul (HAdd.hAdd m 1) ( …
            -/
  | 0 => by simp [shiftLeft', pow_zero, Nat.one_mul]
            /-
              🎉 no goals
            -/
  | k + 1 => by
    rw [shiftLeft', bit_val, Bool.toNat_true, add_assoc, ← Nat.mul_add_one,
      shiftLeft'_tt_eq_mul_pow m k, mul_left_comm, mul_comm 2, pow_succ]


theorem shiftLeft'_ne_zero_left (b) {m} (h : m ≠ 0) (n) : shiftLeft' b m n ≠ 0 := by
  /-
    b : Bool
    m : Nat
    h : Ne m 0
    n : Nat
    ⊢ Ne (Nat.shiftLeft' b m n) 0
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [bit_ne_zero, shiftLeft', *]
                  /-
                    🎉 no goals
                  -/


theorem shiftLeft'_tt_ne_zero (m) : ∀ {n}, (n ≠ 0) → shiftLeft' true m n ≠ 0
  | 0, h => absurd rfl h
                    /-
                      m n✝ : Nat
                      x✝ : Ne n✝.succ 0
                      ⊢ Ne (Nat.shiftLeft' Bool.true m n✝.succ) 0
                    -/
  | succ _, _ => by dsimp [shiftLeft', bit]; omega
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                     /-
                                       ⊢ Eq (Nat.size 0) 0
                                     -/
theorem size_zero : size 0 = 0 := by simp [size]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem size_bit {b n} (h : bit b n ≠ 0) : size (bit b n) = succ (size n) := by
  /-
    b : Bool
    n : Nat
    h : Ne (Nat.bit b n) 0
    ⊢ Eq (Nat.bit b n).size n.size.succ
  -/
  unfold size
  conv =>
    lhs
    rw [binaryRec]
    simp [h]


@[simp]
theorem size_one : size 1 = 1 :=
                                /-
                                  ⊢ Eq (Nat.bit Bool.true 0).size 1
                                -/
  show size (bit true 0) = 1 by rw [size_bit, size_zero]; exact Nat.one_ne_zero
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem size_shiftLeft' {b m n} (h : shiftLeft' b m n ≠ 0) :
    size (shiftLeft' b m n) = size m + n := by
  induction n with
  | zero => simp [shiftLeft']
  | succ n IH =>
    simp only [shiftLeft', ne_eq] at h ⊢
    rw [size_bit h, Nat.add_succ]
    by_cases s0 : shiftLeft' b m n = 0
    case neg => rw [IH s0]
    rw [s0] at h ⊢
    cases b; · exact absurd rfl h
    have : shiftLeft' true m n + 1 = 1 := congr_arg (· + 1) s0
    rw [shiftLeft'_tt_eq_mul_pow] at this
    obtain rfl := succ.inj (eq_one_of_dvd_one ⟨_, this.symm⟩)
    simp only [zero_add, one_mul] at this
    obtain rfl : n = 0 := not_ne_iff.1 fun hn ↦ ne_of_gt (Nat.one_lt_pow hn (by decide)) this
    rw [add_zero]

-- TODO: decide whether `Nat.shiftLeft_eq` (which rewrites the LHS into a power) should be a simp
-- lemma; it was not in mathlib3. Until then, tell the simpNF linter to ignore the issue.

@[simp, nolint simpNF]
theorem size_shiftLeft {m} (h : m ≠ 0) (n) : size (m <<< n) = size m + n := by
  /-
    m : Nat
    h : Ne m 0
    n : Nat
    ⊢ Eq (HShiftLeft.hShiftLeft m n).size (HAdd.hAdd m.size n)
  -/
  simp only [size_shiftLeft' (shiftLeft'_ne_zero_left _ h _), ← shiftLeft'_false]
  /-
    🎉 no goals
  -/


theorem lt_size_self (n : ℕ) : n < 2 ^ size n := by
  /-
    n : Nat
    ⊢ LT.lt n (HPow.hPow 2 n.size)
  -/
  rw [← one_shiftLeft]
  /-
    n : Nat
    ⊢ LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
  -/
  have : ∀ {n}, n = 0 → n < 1 <<< (size n) := by simp
  /-
    n : Nat
    this : ∀ {n : Nat}, Eq n 0 → LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
    ⊢ LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
  -/
  refine binaryRec ?_ ?_ n
    /-
      case refine_1
      n : Nat
      this : ∀ {n : Nat}, Eq n 0 → LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
      ⊢ LT.lt 0 (HShiftLeft.hShiftLeft 1 (Nat.size 0))
    -/
  · apply this rfl
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    n : Nat
    this : ∀ {n : Nat}, Eq n 0 → LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
    ⊢ ∀ (b : Bool) (n : Nat), LT.lt n (HShiftLeft.hShiftLeft 1 n.size) → LT.lt (Na …
  -/
  intro b n IH
  /-
    case refine_2
    n✝ : Nat
    this : ∀ {n : Nat}, Eq n 0 → LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
    b : Bool
    n : Nat
    IH : LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
    ⊢ LT.lt (Nat.bit b n) (HShiftLeft.hShiftLeft 1 (Nat.bit b n).size)
  -/
  by_cases h : bit b n = 0
    /-
      case pos
      n✝ : Nat
      this : ∀ {n : Nat}, Eq n 0 → LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
      b : Bool
      n : Nat
      IH : LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
      h : Eq (Nat.bit b n) 0
      ⊢ LT.lt (Nat.bit b n) (HShiftLeft.hShiftLeft 1 (Nat.bit b n).size)
    -/
  · apply this h
    /-
      🎉 no goals
    -/
  /-
    case neg
    n✝ : Nat
    this : ∀ {n : Nat}, Eq n 0 → LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
    b : Bool
    n : Nat
    IH : LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
    h : Not (Eq (Nat.bit b n) 0)
    ⊢ LT.lt (Nat.bit b n) (HShiftLeft.hShiftLeft 1 (Nat.bit b n).size)
  -/
  rw [size_bit h, shiftLeft_succ, shiftLeft_eq, one_mul]
  /-
    case neg
    n✝ : Nat
    this : ∀ {n : Nat}, Eq n 0 → LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
    b : Bool
    n : Nat
    IH : LT.lt n (HShiftLeft.hShiftLeft 1 n.size)
    h : Not (Eq (Nat.bit b n) 0)
    ⊢ LT.lt (Nat.bit b n) (HMul.hMul 2 (HPow.hPow 2 n.size))
  -/
                              /-
                                🎉 no goals
                              -/
  cases b <;> dsimp [bit] <;> omega
                              /-
                                🎉 no goals
                              -/


theorem size_le {m n : ℕ} : size m ≤ n ↔ m < 2 ^ n :=
                                                                        /-
                                                                          m n : Nat
                                                                          h : LE.le m.size n
                                                                          ⊢ GT.gt 2 0
                                                                        -/
  ⟨fun h => lt_of_lt_of_le (lt_size_self _) (pow_le_pow_of_le_right (by decide) h), by
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    /-
      m n : Nat
      ⊢ LT.lt m (HPow.hPow 2 n) → LE.le m.size n
    -/
    rw [← one_shiftLeft]
    induction m using binaryRec generalizing n with
    | z => simp
    | f b m IH =>
      intro h
      by_cases e : bit b m = 0
      · simp [e]
      rw [size_bit e]
      cases n with
      | zero => exact e.elim (Nat.eq_zero_of_le_zero (le_of_lt_succ h))
      | succ n =>
        apply succ_le_succ (IH _)
        apply Nat.lt_of_mul_lt_mul_left (a := 2)
        simp only [shiftLeft_succ] at *
        refine lt_of_le_of_lt ?_ h
        cases b <;> dsimp [bit] <;> omega⟩


theorem lt_size {m n : ℕ} : m < size n ↔ 2 ^ m ≤ n := by
  /-
    m n : Nat
    ⊢ Iff (LT.lt m n.size) (LE.le (HPow.hPow 2 m) n)
  -/
  rw [← not_lt, Decidable.iff_not_comm, not_lt, size_le]
  /-
    🎉 no goals
  -/


                                                    /-
                                                      n : Nat
                                                      ⊢ Iff (LT.lt 0 n.size) (LT.lt 0 n)
                                                    -/
theorem size_pos {n : ℕ} : 0 < size n ↔ 0 < n := by rw [lt_size]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem size_eq_zero {n : ℕ} : size n = 0 ↔ n = 0 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.size 0) (Eq n 0)
  -/
  simpa [Nat.pos_iff_ne_zero, not_iff_not] using size_pos
  /-
    🎉 no goals
  -/


theorem size_pow {n : ℕ} : size (2 ^ n) = n + 1 :=
                                                     /-
                                                       n : Nat
                                                       ⊢ LT.lt 1 2
                                                     -/
  le_antisymm (size_le.2 <| Nat.pow_lt_pow_right (by decide) (lt_succ_self _))
                                                     /-
                                                       🎉 no goals
                                                     -/
    (lt_size.2 <| le_rfl)


theorem size_le_size {m n : ℕ} (h : m ≤ n) : size m ≤ size n :=
  size_le.2 <| lt_of_le_of_lt h (lt_size_self _)


theorem size_eq_bits_len (n : ℕ) : n.bits.length = n.size := by
  induction n using Nat.binaryRec' with
  | z => simp
  | f _ _ h ih =>
    rw [size_bit, bits_append_bit _ _ h]
    · simp [ih]
    · simpa [bit_eq_zero_iff]


