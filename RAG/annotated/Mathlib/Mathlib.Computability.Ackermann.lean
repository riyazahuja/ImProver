/-- The two-argument Ackermann function, defined so that

- `ack 0 n = n + 1`
- `ack (m + 1) 0 = ack m 1`
- `ack (m + 1) (n + 1) = ack m (ack (m + 1) n)`.

This is of interest as both a fast-growing function, and as an example of a recursive function that
isn't primitive recursive. -/
def ack : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => ack m 1
  | m + 1, n + 1 => ack m (ack (m + 1) n)


@[simp]
                                                 /-
                                                   n : Nat
                                                   ⊢ Eq (ack 0 n) (HAdd.hAdd n 1)
                                                 -/
theorem ack_zero (n : ℕ) : ack 0 n = n + 1 := by rw [ack]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
                                                              /-
                                                                m : Nat
                                                                ⊢ Eq (ack (HAdd.hAdd m 1) 0) (ack m 1)
                                                              -/
theorem ack_succ_zero (m : ℕ) : ack (m + 1) 0 = ack m 1 := by rw [ack]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                                                    /-
                                                                                      m n : Nat
                                                                                      ⊢ Eq (ack (HAdd.hAdd m 1) (HAdd.hAdd n 1)) (ack m (ack (HAdd.hAdd m 1) n))
                                                                                    -/
theorem ack_succ_succ (m n : ℕ) : ack (m + 1) (n + 1) = ack m (ack (m + 1) n) := by rw [ack]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[simp]
theorem ack_one (n : ℕ) : ack 1 n = n + 2 := by
  /-
    n : Nat
    ⊢ Eq (ack 1 n) (HAdd.hAdd n 2)
  -/
  induction' n with n IH
    /-
      case zero
      ⊢ Eq (ack 1 0) (HAdd.hAdd 0 2)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      IH : Eq (ack 1 n) (HAdd.hAdd n 2)
      ⊢ Eq (ack 1 (HAdd.hAdd n 1)) (HAdd.hAdd (HAdd.hAdd n 1) 2)
    -/
  · simp [IH]
    /-
      🎉 no goals
    -/


@[simp]
theorem ack_two (n : ℕ) : ack 2 n = 2 * n + 3 := by
  /-
    n : Nat
    ⊢ Eq (ack 2 n) (HAdd.hAdd (HMul.hMul 2 n) 3)
  -/
  induction' n with n IH
    /-
      case zero
      ⊢ Eq (ack 2 0) (HAdd.hAdd (HMul.hMul 2 0) 3)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      IH : Eq (ack 2 n) (HAdd.hAdd (HMul.hMul 2 n) 3)
      ⊢ Eq (ack 2 (HAdd.hAdd n 1)) (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd n 1)) 3)
    -/
  · simpa [mul_succ]
    /-
      🎉 no goals
    -/

-- Porting note: re-written to get rid of ack_three_aux

@[simp]
theorem ack_three (n : ℕ) : ack 3 n = 2 ^ (n + 3) - 3 := by
  /-
    n : Nat
    ⊢ Eq (ack 3 n) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd n 3)) 3)
  -/
  induction' n with n IH
    /-
      case zero
      ⊢ Eq (ack 3 0) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd 0 3)) 3)
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [ack_succ_succ, IH, ack_two, Nat.succ_add, Nat.pow_succ 2 (n + 3), mul_comm _ 2,
        Nat.mul_sub_left_distrib, ← Nat.sub_add_comm, two_mul 3, Nat.add_sub_add_right]
    /-
      case succ
      n : Nat
      IH : Eq (ack 3 n) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd n 3)) 3)
      ⊢ LE.le (HMul.hMul 2 3) (HMul.hMul 2 (HPow.hPow 2 (HAdd.hAdd n 3)))
    -/
    have H : 2 * 3 ≤ 2 * 2 ^ 3 := by norm_num
    /-
      case succ
      n : Nat
      IH : Eq (ack 3 n) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd n 3)) 3)
      H : LE.le (HMul.hMul 2 3) (HMul.hMul 2 (HPow.hPow 2 3))
      ⊢ LE.le (HMul.hMul 2 3) (HMul.hMul 2 (HPow.hPow 2 (HAdd.hAdd n 3)))
    -/
    apply H.trans
    /-
      case succ
      n : Nat
      IH : Eq (ack 3 n) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd n 3)) 3)
      H : LE.le (HMul.hMul 2 3) (HMul.hMul 2 (HPow.hPow 2 3))
      ⊢ LE.le (HMul.hMul 2 (HPow.hPow 2 3)) (HMul.hMul 2 (HPow.hPow 2 (HAdd.hAdd n 3 …
    -/
    rw [_root_.mul_le_mul_left two_pos]
    /-
      case succ
      n : Nat
      IH : Eq (ack 3 n) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd n 3)) 3)
      H : LE.le (HMul.hMul 2 3) (HMul.hMul 2 (HPow.hPow 2 3))
      ⊢ LE.le (HPow.hPow 2 3) (HPow.hPow 2 (HAdd.hAdd n 3))
    -/
    exact pow_right_mono₀ one_le_two (Nat.le_add_left 3 n)
    /-
      🎉 no goals
    -/


theorem ack_pos : ∀ m n, 0 < ack m n
               /-
                 n : Nat
                 ⊢ LT.lt 0 (ack 0 n)
               -/
  | 0, n => by simp
               /-
                 🎉 no goals
               -/
  | m + 1, 0 => by
    /-
      m : Nat
      ⊢ LT.lt 0 (ack (HAdd.hAdd m 1) 0)
    -/
    rw [ack_succ_zero]
    /-
      m : Nat
      ⊢ LT.lt 0 (ack m 1)
    -/
    apply ack_pos
    /-
      🎉 no goals
    -/
  | m + 1, n + 1 => by
    /-
      m n : Nat
      ⊢ LT.lt 0 (ack (HAdd.hAdd m 1) (HAdd.hAdd n 1))
    -/
    rw [ack_succ_succ]
    /-
      m n : Nat
      ⊢ LT.lt 0 (ack m (ack (HAdd.hAdd m 1) n))
    -/
    apply ack_pos
    /-
      🎉 no goals
    -/


theorem one_lt_ack_succ_left : ∀ m n, 1 < ack (m + 1) n
               /-
                 n : Nat
                 ⊢ LT.lt 1 (ack (HAdd.hAdd 0 1) n)
               -/
  | 0, n => by simp
               /-
                 🎉 no goals
               -/
  | m + 1, 0 => by
    /-
      m : Nat
      ⊢ LT.lt 1 (ack (HAdd.hAdd (HAdd.hAdd m 1) 1) 0)
    -/
    rw [ack_succ_zero]
    /-
      m : Nat
      ⊢ LT.lt 1 (ack (HAdd.hAdd m 1) 1)
    -/
    apply one_lt_ack_succ_left
    /-
      🎉 no goals
    -/
  | m + 1, n + 1 => by
    /-
      m n : Nat
      ⊢ LT.lt 1 (ack (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd n 1))
    -/
    rw [ack_succ_succ]
    /-
      m n : Nat
      ⊢ LT.lt 1 (ack (HAdd.hAdd m 1) (ack (HAdd.hAdd (HAdd.hAdd m 1) 1) n))
    -/
    apply one_lt_ack_succ_left
    /-
      🎉 no goals
    -/


theorem one_lt_ack_succ_right : ∀ m n, 1 < ack m (n + 1)
               /-
                 n : Nat
                 ⊢ LT.lt 1 (ack 0 (HAdd.hAdd n 1))
               -/
  | 0, n => by simp
               /-
                 🎉 no goals
               -/
  | m + 1, n => by
    /-
      m n : Nat
      ⊢ LT.lt 1 (ack (HAdd.hAdd m 1) (HAdd.hAdd n 1))
    -/
    rw [ack_succ_succ]
    /-
      m n : Nat
      ⊢ LT.lt 1 (ack m (ack (HAdd.hAdd m 1) n))
    -/
    cases' exists_eq_succ_of_ne_zero (ack_pos (m + 1) n).ne' with h h
    /-
      case intro
      m n h✝ : Nat
      h : Eq (ack (HAdd.hAdd m 1) n) h✝.succ
      ⊢ LT.lt 1 (ack m (ack (HAdd.hAdd m 1) n))
    -/
    rw [h]
    /-
      case intro
      m n h✝ : Nat
      h : Eq (ack (HAdd.hAdd m 1) n) h✝.succ
      ⊢ LT.lt 1 (ack m h✝.succ)
    -/
    apply one_lt_ack_succ_right
    /-
      🎉 no goals
    -/


theorem ack_strictMono_right : ∀ m, StrictMono (ack m)
                       /-
                         n₁ n₂ : Nat
                         h : LT.lt n₁ n₂
                         ⊢ LT.lt (ack 0 n₁) (ack 0 n₂)
                       -/
  | 0, n₁, n₂, h => by simpa using h
                       /-
                         🎉 no goals
                       -/
  | m + 1, 0, n + 1, _h => by
    /-
      m n : Nat
      _h : LT.lt 0 (HAdd.hAdd n 1)
      ⊢ LT.lt (ack (HAdd.hAdd m 1) 0) (ack (HAdd.hAdd m 1) (HAdd.hAdd n 1))
    -/
    rw [ack_succ_zero, ack_succ_succ]
    /-
      m n : Nat
      _h : LT.lt 0 (HAdd.hAdd n 1)
      ⊢ LT.lt (ack m 1) (ack m (ack (HAdd.hAdd m 1) n))
    -/
    exact ack_strictMono_right _ (one_lt_ack_succ_left m n)
    /-
      🎉 no goals
    -/
  | m + 1, n₁ + 1, n₂ + 1, h => by
    /-
      m n₁ n₂ : Nat
      h : LT.lt (HAdd.hAdd n₁ 1) (HAdd.hAdd n₂ 1)
      ⊢ LT.lt (ack (HAdd.hAdd m 1) (HAdd.hAdd n₁ 1)) (ack (HAdd.hAdd m 1) (HAdd.hAdd …
    -/
    rw [ack_succ_succ, ack_succ_succ]
    /-
      m n₁ n₂ : Nat
      h : LT.lt (HAdd.hAdd n₁ 1) (HAdd.hAdd n₂ 1)
      ⊢ LT.lt (ack m (ack (HAdd.hAdd m 1) n₁)) (ack m (ack (HAdd.hAdd m 1) n₂))
    -/
    apply ack_strictMono_right _ (ack_strictMono_right _ _)
    /-
      m n₁ n₂ : Nat
      h : LT.lt (HAdd.hAdd n₁ 1) (HAdd.hAdd n₂ 1)
      ⊢ LT.lt n₁ n₂
    -/
    rwa [add_lt_add_iff_right] at h
    /-
      🎉 no goals
    -/


theorem ack_mono_right (m : ℕ) : Monotone (ack m) :=
  (ack_strictMono_right m).monotone


theorem ack_injective_right (m : ℕ) : Function.Injective (ack m) :=
  (ack_strictMono_right m).injective


@[simp]
theorem ack_lt_iff_right {m n₁ n₂ : ℕ} : ack m n₁ < ack m n₂ ↔ n₁ < n₂ :=
  (ack_strictMono_right m).lt_iff_lt


@[simp]
theorem ack_le_iff_right {m n₁ n₂ : ℕ} : ack m n₁ ≤ ack m n₂ ↔ n₁ ≤ n₂ :=
  (ack_strictMono_right m).le_iff_le


@[simp]
theorem ack_inj_right {m n₁ n₂ : ℕ} : ack m n₁ = ack m n₂ ↔ n₁ = n₂ :=
  (ack_injective_right m).eq_iff


theorem max_ack_right (m n₁ n₂ : ℕ) : ack m (max n₁ n₂) = max (ack m n₁) (ack m n₂) :=
  (ack_mono_right m).map_max


theorem add_lt_ack : ∀ m n, m + n < ack m n
               /-
                 n : Nat
                 ⊢ LT.lt (HAdd.hAdd 0 n) (ack 0 n)
               -/
  | 0, n => by simp
               /-
                 🎉 no goals
               -/
                   /-
                     m : Nat
                     ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd m 1) 0) (ack (HAdd.hAdd m 1) 0)
                   -/
  | m + 1, 0 => by simpa using add_lt_ack m 1
                   /-
                     🎉 no goals
                   -/
  | m + 1, n + 1 =>
    calc
                                            /-
                                              m n : Nat
                                              ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m 1) n) 1) (HAdd.hAdd m (HAdd.hAdd (H …
                                            -/
      m + 1 + n + 1 ≤ m + (m + n + 2) := by omega
                                            /-
                                              🎉 no goals
                                            -/
      _ < ack m (m + n + 2) := add_lt_ack _ _
      _ ≤ ack m (ack (m + 1) n) :=
                                               /-
                                                 m n : Nat
                                                 ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) 2) (HAdd.hAdd (HAdd.hAdd m 1) n).succ
                                               -/
        ack_mono_right m <| le_of_eq_of_le (by rw [succ_eq_add_one]; ring_nf)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
        <| succ_le_of_lt <| add_lt_ack (m + 1) n
      _ = ack (m + 1) (n + 1) := (ack_succ_succ m n).symm


theorem add_add_one_le_ack (m n : ℕ) : m + n + 1 ≤ ack m n :=
  succ_le_of_lt (add_lt_ack m n)


theorem lt_ack_left (m n : ℕ) : m < ack m n :=
  (self_le_add_right m n).trans_lt <| add_lt_ack m n


theorem lt_ack_right (m n : ℕ) : n < ack m n :=
  (self_le_add_left n m).trans_lt <| add_lt_ack m n

-- we reorder the arguments to appease the equation compiler

private theorem ack_strict_mono_left' : ∀ {m₁ m₂} (n), m₁ < m₂ → ack m₁ n < ack m₂ n
  | m, 0, _ => fun h => (not_lt_zero m h).elim
                                /-
                                  m : Nat
                                  _h : LT.lt 0 (HAdd.hAdd m 1)
                                  ⊢ LT.lt (ack 0 0) (ack (HAdd.hAdd m 1) 0)
                                -/
  | 0, m + 1, 0 => fun _h => by simpa using one_lt_ack_succ_right m 0
                                /-
                                  🎉 no goals
                                -/
  | 0, m + 1, n + 1 => fun h => by
    /-
      m n : Nat
      h : LT.lt 0 (HAdd.hAdd m 1)
      ⊢ LT.lt (ack 0 (HAdd.hAdd n 1)) (ack (HAdd.hAdd m 1) (HAdd.hAdd n 1))
    -/
    rw [ack_zero, ack_succ_succ]
    /-
      m n : Nat
      h : LT.lt 0 (HAdd.hAdd m 1)
      ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd n 1) 1) (ack m (ack (HAdd.hAdd m 1) n))
    -/
    apply lt_of_le_of_lt (le_trans _ <| add_le_add_left (add_add_one_le_ack _ _) m) (add_lt_ack _ _)
    /-
      m n : Nat
      h : LT.lt 0 (HAdd.hAdd m 1)
      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd m (HAdd.hAdd (HAdd.hAdd (HAdd …
    -/
    omega
    /-
      🎉 no goals
    -/
  | m₁ + 1, m₂ + 1, 0 => fun h => by
    /-
      m₁ m₂ : Nat
      h : LT.lt (HAdd.hAdd m₁ 1) (HAdd.hAdd m₂ 1)
      ⊢ LT.lt (ack (HAdd.hAdd m₁ 1) 0) (ack (HAdd.hAdd m₂ 1) 0)
    -/
    simpa using ack_strict_mono_left' 1 ((add_lt_add_iff_right 1).1 h)
    /-
      🎉 no goals
    -/
  | m₁ + 1, m₂ + 1, n + 1 => fun h => by
    /-
      m₁ m₂ n : Nat
      h : LT.lt (HAdd.hAdd m₁ 1) (HAdd.hAdd m₂ 1)
      ⊢ LT.lt (ack (HAdd.hAdd m₁ 1) (HAdd.hAdd n 1)) (ack (HAdd.hAdd m₂ 1) (HAdd.hAd …
    -/
    rw [ack_succ_succ, ack_succ_succ]
    exact
      (ack_strict_mono_left' _ <| (add_lt_add_iff_right 1).1 h).trans
        (ack_strictMono_right _ <| ack_strict_mono_left' n h)


theorem ack_strictMono_left (n : ℕ) : StrictMono fun m => ack m n := fun _m₁ _m₂ =>
  ack_strict_mono_left' n


theorem ack_mono_left (n : ℕ) : Monotone fun m => ack m n :=
  (ack_strictMono_left n).monotone


theorem ack_injective_left (n : ℕ) : Function.Injective fun m => ack m n :=
  (ack_strictMono_left n).injective


@[simp]
theorem ack_lt_iff_left {m₁ m₂ n : ℕ} : ack m₁ n < ack m₂ n ↔ m₁ < m₂ :=
  (ack_strictMono_left n).lt_iff_lt


@[simp]
theorem ack_le_iff_left {m₁ m₂ n : ℕ} : ack m₁ n ≤ ack m₂ n ↔ m₁ ≤ m₂ :=
  (ack_strictMono_left n).le_iff_le


@[simp]
theorem ack_inj_left {m₁ m₂ n : ℕ} : ack m₁ n = ack m₂ n ↔ m₁ = m₂ :=
  (ack_injective_left n).eq_iff


theorem max_ack_left (m₁ m₂ n : ℕ) : ack (max m₁ m₂) n = max (ack m₁ n) (ack m₂ n) :=
  (ack_mono_left n).map_max


theorem ack_le_ack {m₁ m₂ n₁ n₂ : ℕ} (hm : m₁ ≤ m₂) (hn : n₁ ≤ n₂) : ack m₁ n₁ ≤ ack m₂ n₂ :=
  (ack_mono_left n₁ hm).trans <| ack_mono_right m₂ hn


theorem ack_succ_right_le_ack_succ_left (m n : ℕ) : ack m (n + 1) ≤ ack (m + 1) n := by
  /-
    m n : Nat
    ⊢ LE.le (ack m (HAdd.hAdd n 1)) (ack (HAdd.hAdd m 1) n)
  -/
  cases' n with n n
    /-
      case zero
      m : Nat
      ⊢ LE.le (ack m (HAdd.hAdd 0 1)) (ack (HAdd.hAdd m 1) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      m n : Nat
      ⊢ LE.le (ack m (HAdd.hAdd (HAdd.hAdd n 1) 1)) (ack (HAdd.hAdd m 1) (HAdd.hAdd  …
    -/
  · rw [ack_succ_succ]
    /-
      case succ
      m n : Nat
      ⊢ LE.le (ack m (HAdd.hAdd (HAdd.hAdd n 1) 1)) (ack m (ack (HAdd.hAdd m 1) n))
    -/
    apply ack_mono_right m (le_trans _ <| add_add_one_le_ack _ n)
    /-
      m n : Nat
      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m 1) n) …
    -/
    omega
    /-
      🎉 no goals
    -/

-- All the inequalities from this point onwards are specific to the main proof.

private theorem sq_le_two_pow_add_one_minus_three (n : ℕ) : n ^ 2 ≤ 2 ^ (n + 1) - 3 := by
  /-
    n : Nat
    ⊢ LE.le (HPow.hPow n 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd n 1)) 3)
  -/
  induction' n with k hk
    /-
      case zero
      ⊢ LE.le (HPow.hPow 0 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd 0 1)) 3)
    -/
  · norm_num
    /-
      🎉 no goals
    -/
    /-
      case succ
      k : Nat
      hk : LE.le (HPow.hPow k 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd k 1)) 3)
      ⊢ LE.le (HPow.hPow (HAdd.hAdd k 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (HAdd …
    -/
  · cases' k with k k
      /-
        case succ.zero
        hk : LE.le (HPow.hPow 0 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd 0 1)) 3)
        ⊢ LE.le (HPow.hPow (HAdd.hAdd 0 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (HAdd …
      -/
    · norm_num
      /-
        🎉 no goals
      -/
    · rw [add_sq, Nat.pow_succ 2, mul_comm _ 2, two_mul (2 ^ _),
          add_tsub_assoc_of_le, add_comm (2 ^ _), add_assoc]
        /-
          case succ.succ
          k : Nat
          hk : LE.le (HPow.hPow (HAdd.hAdd k 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (H …
          ⊢ LE.le (HAdd.hAdd (HPow.hPow (HAdd.hAdd k 1) 2) (HAdd.hAdd (HMul.hMul (HMul.h …
        -/
      · apply Nat.add_le_add hk
        /-
          case succ.succ
          k : Nat
          hk : LE.le (HPow.hPow (HAdd.hAdd k 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (H …
          ⊢ LE.le (HAdd.hAdd (HMul.hMul (HMul.hMul 2 (HAdd.hAdd k 1)) 1) (HPow.hPow 1 2) …
        -/
        norm_num
        /-
          case succ.succ
          k : Nat
          hk : LE.le (HPow.hPow (HAdd.hAdd k 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (H …
          ⊢ LE.le (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd k 1)) 1) (HPow.hPow 2 (HAdd.hAdd k  …
        -/
        apply succ_le_of_lt
        /-
          case succ.succ.h
          k : Nat
          hk : LE.le (HPow.hPow (HAdd.hAdd k 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (H …
          ⊢ LT.lt (HMul.hMul 2 (HAdd.hAdd k 1)) (HPow.hPow 2 (HAdd.hAdd k 2))
        -/
        rw [Nat.pow_succ, mul_comm _ 2, mul_lt_mul_left (zero_lt_two' ℕ)]
        /-
          case succ.succ.h
          k : Nat
          hk : LE.le (HPow.hPow (HAdd.hAdd k 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (H …
          ⊢ LT.lt (HAdd.hAdd k 1) (HPow.hPow 2 (HAdd.hAdd k 1))
        -/
        exact Nat.lt_two_pow_self
        /-
          🎉 no goals
        -/
        /-
          case succ.succ.h
          k : Nat
          hk : LE.le (HPow.hPow (HAdd.hAdd k 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (H …
          ⊢ LE.le 3 (HPow.hPow 2 (HAdd.hAdd k 2))
        -/
      · rw [Nat.pow_succ, Nat.pow_succ]
        /-
          case succ.succ.h
          k : Nat
          hk : LE.le (HPow.hPow (HAdd.hAdd k 1) 2) (HSub.hSub (HPow.hPow 2 (HAdd.hAdd (H …
          ⊢ LE.le 3 (HMul.hMul (HMul.hMul (HPow.hPow 2 k) 2) 2)
        -/
        linarith [one_le_pow k 2 zero_lt_two]
        /-
          🎉 no goals
        -/


theorem ack_add_one_sq_lt_ack_add_three : ∀ m n, (ack m n + 1) ^ 2 ≤ ack (m + 3) n
               /-
                 n : Nat
                 ⊢ LE.le (HPow.hPow (HAdd.hAdd (ack 0 n) 1) 2) (ack (HAdd.hAdd 0 3) n)
               -/
  | 0, n => by simpa using sq_le_two_pow_add_one_minus_three (n + 2)
               /-
                 🎉 no goals
               -/
  | m + 1, 0 => by
    /-
      m : Nat
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (ack (HAdd.hAdd m 1) 0) 1) 2) (ack (HAdd.hAdd (H …
    -/
    rw [ack_succ_zero, ack_succ_zero]
    /-
      m : Nat
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (ack m 1) 1) 2) (ack (HAdd.hAdd m 3) 1)
    -/
    apply ack_add_one_sq_lt_ack_add_three
    /-
      🎉 no goals
    -/
  | m + 1, n + 1 => by
    /-
      m n : Nat
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (ack (HAdd.hAdd m 1) (HAdd.hAdd n 1)) 1) 2) (ack …
    -/
    rw [ack_succ_succ, ack_succ_succ]
    /-
      m n : Nat
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (ack m (ack (HAdd.hAdd m 1) n)) 1) 2) (ack (HAdd …
    -/
    apply (ack_add_one_sq_lt_ack_add_three _ _).trans (ack_mono_right _ <| ack_mono_left _ _)
    /-
      m n : Nat
      ⊢ LE.le (HAdd.hAdd m 1) (HAdd.hAdd (HAdd.hAdd m 3) 1)
    -/
    omega
    /-
      🎉 no goals
    -/


theorem ack_ack_lt_ack_max_add_two (m n k : ℕ) : ack m (ack n k) < ack (max m n + 2) k :=
  calc
    ack m (ack n k) ≤ ack (max m n) (ack n k) := ack_mono_left _ (le_max_left _ _)
    _ < ack (max m n) (ack (max m n + 1) k) :=
      ack_strictMono_right _ <| ack_strictMono_left k <| lt_succ_of_le <| le_max_right m n
    _ = ack (max m n + 1) (k + 1) := (ack_succ_succ _ _).symm
    _ ≤ ack (max m n + 2) k := ack_succ_right_le_ack_succ_left _ _


theorem ack_add_one_sq_lt_ack_add_four (m n : ℕ) : ack m ((n + 1) ^ 2) < ack (m + 4) n :=
  calc
    ack m ((n + 1) ^ 2) < ack m ((ack m n + 1) ^ 2) :=
      ack_strictMono_right m <| Nat.pow_lt_pow_left (succ_lt_succ <| lt_ack_right m n) two_ne_zero
    _ ≤ ack m (ack (m + 3) n) := ack_mono_right m <| ack_add_one_sq_lt_ack_add_three m n
                                                             /-
                                                               m n : Nat
                                                               ⊢ LE.le m (HAdd.hAdd m 2)
                                                             -/
    _ ≤ ack (m + 2) (ack (m + 3) n) := ack_mono_left _ <| by omega
                                                             /-
                                                               🎉 no goals
                                                             -/
    _ = ack (m + 3) (n + 1) := (ack_succ_succ _ n).symm
    _ ≤ ack (m + 4) n := ack_succ_right_le_ack_succ_left _ n


theorem ack_pair_lt (m n k : ℕ) : ack m (pair n k) < ack (m + 4) (max n k) :=
  (ack_strictMono_right m <| pair_lt_max_add_one_sq n k).trans <|
    ack_add_one_sq_lt_ack_add_four _ _


/-- If `f` is primitive recursive, there exists `m` such that `f n < ack m n` for all `n`. -/
theorem exists_lt_ack_of_nat_primrec {f : ℕ → ℕ} (hf : Nat.Primrec f) :
    ∃ m, ∀ n, f n < ack m n := by
  /-
    f : Nat → Nat
    hf : Nat.Primrec f
    ⊢ Exists fun m => ∀ (n : Nat), LT.lt (f n) (ack m n)
  -/
  induction' hf with f g hf hg IHf IHg f g hf hg IHf IHg f g hf hg IHf IHg
  -- Zero function:
    /-
      case zero
      f : Nat → Nat
      ⊢ Exists fun m => ∀ (n : Nat), LT.lt ((fun x => 0) n) (ack m n)
    -/
  · exact ⟨0, ack_pos 0⟩
    /-
      🎉 no goals
    -/
  -- Successor function:
    /-
      case succ
      f : Nat → Nat
      ⊢ Exists fun m => ∀ (n : Nat), LT.lt n.succ (ack m n)
    -/
  · refine ⟨1, fun n => ?_⟩
    /-
      case succ
      f : Nat → Nat
      n : Nat
      ⊢ LT.lt n.succ (ack 1 n)
    -/
    rw [succ_eq_one_add]
    /-
      case succ
      f : Nat → Nat
      n : Nat
      ⊢ LT.lt (HAdd.hAdd 1 n) (ack 1 n)
    -/
    apply add_lt_ack
    /-
      🎉 no goals
    -/
  -- Left projection:
    /-
      case left
      f : Nat → Nat
      ⊢ Exists fun m => ∀ (n : Nat), LT.lt ((fun n => (Nat.unpair n).1) n) (ack m n)
    -/
  · refine ⟨0, fun n => ?_⟩
    /-
      case left
      f : Nat → Nat
      n : Nat
      ⊢ LT.lt ((fun n => (Nat.unpair n).1) n) (ack 0 n)
    -/
    rw [ack_zero, Nat.lt_succ_iff]
    /-
      case left
      f : Nat → Nat
      n : Nat
      ⊢ LE.le ((fun n => (Nat.unpair n).1) n) n
    -/
    exact unpair_left_le n
    /-
      🎉 no goals
    -/
  -- Right projection:
    /-
      case right
      f : Nat → Nat
      ⊢ Exists fun m => ∀ (n : Nat), LT.lt ((fun n => (Nat.unpair n).2) n) (ack m n)
    -/
  · refine ⟨0, fun n => ?_⟩
    /-
      case right
      f : Nat → Nat
      n : Nat
      ⊢ LT.lt ((fun n => (Nat.unpair n).2) n) (ack 0 n)
    -/
    rw [ack_zero, Nat.lt_succ_iff]
    /-
      case right
      f : Nat → Nat
      n : Nat
      ⊢ LE.le ((fun n => (Nat.unpair n).2) n) n
    -/
    exact unpair_right_le n
    /-
      🎉 no goals
    -/
  /-
    case pair
    f✝ f g : Nat → Nat
    hf : Nat.Primrec f
    hg : Nat.Primrec g
    IHf : Exists fun m => ∀ (n : Nat), LT.lt (f n) (ack m n)
    IHg : Exists fun m => ∀ (n : Nat), LT.lt (g n) (ack m n)
    ⊢ Exists fun m => ∀ (n : Nat), LT.lt ((fun n => Nat.pair (f n) (g n)) n) (ack  …
  -/
  all_goals cases' IHf with a ha; cases' IHg with b hb
  -- Pairing:
  · refine
      ⟨max a b + 3, fun n =>
        (pair_lt_max_add_one_sq _ _).trans_le <|
          (Nat.pow_le_pow_left (add_le_add_right ?_ _) 2).trans <|
            ack_add_one_sq_lt_ack_add_three _ _⟩
    /-
      case pair.intro.intro
      f✝ f g : Nat → Nat
      hf : Nat.Primrec f
      hg : Nat.Primrec g
      a : Nat
      ha : ∀ (n : Nat), LT.lt (f n) (ack a n)
      b : Nat
      hb : ∀ (n : Nat), LT.lt (g n) (ack b n)
      n : Nat
      ⊢ LE.le (Max.max (f n) (g n)) (ack (Max.max a b) n)
    -/
    rw [max_ack_left]
    /-
      case pair.intro.intro
      f✝ f g : Nat → Nat
      hf : Nat.Primrec f
      hg : Nat.Primrec g
      a : Nat
      ha : ∀ (n : Nat), LT.lt (f n) (ack a n)
      b : Nat
      hb : ∀ (n : Nat), LT.lt (g n) (ack b n)
      n : Nat
      ⊢ LE.le (Max.max (f n) (g n)) (Max.max (ack a n) (ack b n))
    -/
    exact max_le_max (ha n).le (hb n).le
    /-
      🎉 no goals
    -/
  -- Composition:
  · exact
      ⟨max a b + 2, fun n =>
        (ha _).trans <| (ack_strictMono_right a <| hb n).trans <| ack_ack_lt_ack_max_add_two a b n⟩
  -- Primitive recursion operator:
  · -- We prove this simpler inequality first.
    have :
      ∀ {m n},
        rec (f m) (fun y IH => g <| pair m <| pair y IH) n < ack (max a b + 9) (m + n) := by
      intro m n
      -- We induct on n.
      induction' n with n IH
      -- The base case is easy.
      · apply (ha m).trans (ack_strictMono_left m <| (le_max_left a b).trans_lt _)
        omega
      · -- We get rid of the first `pair`.
        simp only
        apply (hb _).trans ((ack_pair_lt _ _ _).trans_le _)
        -- If m is the maximum, we get a very weak inequality.
        cases' lt_or_le _ m with h₁ h₁
        · rw [max_eq_left h₁.le]
          exact ack_le_ack (Nat.add_le_add (le_max_right a b) <| by norm_num)
                           (self_le_add_right m _)
        rw [max_eq_right h₁]
        -- We get rid of the second `pair`.
        apply (ack_pair_lt _ _ _).le.trans
        -- If n is the maximum, we get a very weak inequality.
        cases' lt_or_le _ n with h₂ h₂
        · rw [max_eq_left h₂.le, add_assoc]
          exact
            ack_le_ack (Nat.add_le_add (le_max_right a b) <| by norm_num)
              ((le_succ n).trans <| self_le_add_left _ _)
        rw [max_eq_right h₂]
        -- We now use the inductive hypothesis, and some simple algebraic manipulation.
        apply (ack_strictMono_right _ IH).le.trans
        rw [add_succ m, add_succ _ 8, succ_eq_add_one, succ_eq_add_one,
            ack_succ_succ (_ + 8), add_assoc]
        exact ack_mono_left _ (Nat.add_le_add (le_max_right a b) le_rfl)
    -- The proof is now simple.
    /-
      case prec.intro.intro
      f✝ f g : Nat → Nat
      hf : Nat.Primrec f
      hg : Nat.Primrec g
      a : Nat
      ha : ∀ (n : Nat), LT.lt (f n) (ack a n)
      b : Nat
      hb : ∀ (n : Nat), LT.lt (g n) (ack b n)
      this : ∀ {m n : Nat}, LT.lt (Nat.rec (f m) (fun y IH => g (Nat.pair m (Nat.pai …
      ⊢ Exists fun m => ∀ (n : Nat), LT.lt (Nat.unpaired (fun z n => Nat.rec (f z) ( …
    -/
    exact ⟨max a b + 9, fun n => this.trans_le <| ack_mono_right _ <| unpair_add_le n⟩
    /-
      🎉 no goals
    -/


theorem not_nat_primrec_ack_self : ¬Nat.Primrec fun n => ack n n := fun h => by
  /-
    h : Nat.Primrec fun n => ack n n
    ⊢ False
  -/
  cases' exists_lt_ack_of_nat_primrec h with m hm
  /-
    case intro
    h : Nat.Primrec fun n => ack n n
    m : Nat
    hm : ∀ (n : Nat), LT.lt (ack n n) (ack m n)
    ⊢ False
  -/
  exact (hm m).false
  /-
    🎉 no goals
  -/


theorem not_primrec_ack_self : ¬Primrec fun n => ack n n := by
  /-
    ⊢ Not (Primrec fun n => ack n n)
  -/
  rw [Primrec.nat_iff]
  /-
    ⊢ Not (Nat.Primrec fun n => ack n n)
  -/
  exact not_nat_primrec_ack_self
  /-
    🎉 no goals
  -/


/-- The Ackermann function is not primitive recursive. -/
theorem not_primrec₂_ack : ¬Primrec₂ ack := fun h =>
  not_primrec_ack_self <| h.comp Primrec.id Primrec.id

