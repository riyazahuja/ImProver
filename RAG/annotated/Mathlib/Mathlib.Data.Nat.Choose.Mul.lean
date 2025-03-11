theorem choose_mul_add {m n : ℕ} (hn : n ≠ 0) :
    (m * n + n).choose n = (m + 1) * (m * n + n - 1).choose (n - 1) := by
  /-
    m n : Nat
    hn : Ne n 0
    ⊢ Eq ((HAdd.hAdd (HMul.hMul m n) n).choose n) (HMul.hMul (HAdd.hAdd m 1) ((HSu …
  -/
  rw [← Nat.mul_left_inj (mul_ne_zero (factorial_ne_zero (m * n)) (factorial_ne_zero n))]
  /-
    m n : Nat
    hn : Ne n 0
    ⊢ Eq (HMul.hMul ((HAdd.hAdd (HMul.hMul m n) n).choose n) (HMul.hMul (HMul.hMul …
  -/
  set p := n - 1
  /-
    m n : Nat
    hn : Ne n 0
    p : Nat := HSub.hSub n 1
    ⊢ Eq (HMul.hMul ((HAdd.hAdd (HMul.hMul m n) n).choose n) (HMul.hMul (HMul.hMul …
  -/
  have hp : n = p + 1 := (succ_pred_eq_of_ne_zero hn).symm
  /-
    m n : Nat
    hn : Ne n 0
    p : Nat := HSub.hSub n 1
    hp : Eq n (HAdd.hAdd p 1)
    ⊢ Eq (HMul.hMul ((HAdd.hAdd (HMul.hMul m n) n).choose n) (HMul.hMul (HMul.hMul …
  -/
  simp only [hp, add_succ_sub_one]
  calc
    (m * (p + 1) + (p + 1)).choose (p + 1) * ((m * (p+1))! * (p+1)!)
      = (m * (p + 1) + (p + 1)).choose (p + 1) * (m * (p+1))! * (p+1)! := by ring
    _ = (m * (p+ 1) + (p + 1))! := by rw [add_choose_mul_factorial_mul_factorial]
    _ = ((m * (p+ 1) + p) + 1)! := by ring_nf
    _ = ((m * (p + 1) + p) + 1) * (m * (p + 1) + p)! := by rw [factorial_succ]
    _ = (m * (p + 1) + p)! * ((p + 1) * (m + 1)) := by ring
    _ = ((m * (p + 1) + p).choose p * (m * (p+1))! * (p)!) * ((p + 1) * (m + 1)) := by
      rw [add_choose_mul_factorial_mul_factorial]
    _ = (m * (p + 1) + p).choose p * (m * (p+1))! * (((p + 1) * (p)!) * (m + 1)) := by ring
    _ = (m * (p + 1) + p).choose p * (m * (p+1))! * ((p + 1)! * (m + 1)) := by rw [factorial_succ]
    _ = (m + 1) * (m * (p + 1) + p).choose p * ((m * (p + 1))! * (p + 1)!) := by ring


theorem choose_mul_right {m n : ℕ} (hn : n ≠ 0) :
    (m * n).choose n = m * (m * n - 1).choose (n - 1) := by
  /-
    m n : Nat
    hn : Ne n 0
    ⊢ Eq ((HMul.hMul m n).choose n) (HMul.hMul m ((HSub.hSub (HMul.hMul m n) 1).ch …
  -/
  by_cases hm : m = 0
    /-
      case pos
      m n : Nat
      hn : Ne n 0
      hm : Eq m 0
      ⊢ Eq ((HMul.hMul m n).choose n) (HMul.hMul m ((HSub.hSub (HMul.hMul m n) 1).ch …
    -/
  · simp only [hm, zero_mul, choose_eq_zero_iff]
    /-
      case pos
      m n : Nat
      hn : Ne n 0
      hm : Eq m 0
      ⊢ LT.lt 0 n
    -/
    exact Nat.pos_of_ne_zero hn
    /-
      🎉 no goals
    -/
    /-
      case neg
      m n : Nat
      hn : Ne n 0
      hm : Not (Eq m 0)
      ⊢ Eq ((HMul.hMul m n).choose n) (HMul.hMul m ((HSub.hSub (HMul.hMul m n) 1).ch …
    -/
  · set p := m - 1; have hp : m = p + 1 := (succ_pred_eq_of_ne_zero hm).symm
    /-
      case neg
      m n : Nat
      hn : Ne n 0
      hm : Not (Eq m 0)
      p : Nat := HSub.hSub m 1
      hp : Eq m (HAdd.hAdd p 1)
      ⊢ Eq ((HMul.hMul m n).choose n) (HMul.hMul m ((HSub.hSub (HMul.hMul m n) 1).ch …
    -/
    simp only [hp]
    /-
      case neg
      m n : Nat
      hn : Ne n 0
      hm : Not (Eq m 0)
      p : Nat := HSub.hSub m 1
      hp : Eq m (HAdd.hAdd p 1)
      ⊢ Eq ((HMul.hMul (HAdd.hAdd p 1) n).choose n) (HMul.hMul (HAdd.hAdd p 1) ((HSu …
    -/
    rw [add_mul, one_mul, choose_mul_add hn]
    /-
      🎉 no goals
    -/


