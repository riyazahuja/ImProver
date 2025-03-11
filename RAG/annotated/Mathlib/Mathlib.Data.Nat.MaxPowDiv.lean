/--
Tail recursive function which returns the largest `k : ℕ` such that `p ^ k ∣ n` for any `p : ℕ`.
`padicValNat_eq_maxPowDiv` allows the code generator to use this definition for `padicValNat`
-/
def maxPowDiv (p n : ℕ) : ℕ :=
  go 0 p n
where go (k p n : ℕ) : ℕ :=
  if 1 < p ∧ 0 < n ∧ n % p = 0 then
    go (k+1) p (n / p)
  else
    k
  termination_by n
  /-
    k p n : Nat
    h✝ : And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0))
    ⊢ LT.lt (HDiv.hDiv n p) n
  -/
  decreasing_by apply Nat.div_lt_self <;> tauto
  /-
    🎉 no goals
  -/


theorem go_succ {k p n : ℕ} : go (k+1) p n = go k p n + 1 := by
  /-
    k p n : Nat
    ⊢ Eq (Nat.maxPowDiv.go (HAdd.hAdd k 1) p n) (HAdd.hAdd (Nat.maxPowDiv.go k p n …
  -/
  induction k, p, n using go.induct
  case case1 h ih =>
    unfold go
    simp only [if_pos h]
    exact ih
  case case2 h =>
    unfold go
    simp only [if_neg h]


@[simp]
theorem zero_base {n : ℕ} : maxPowDiv 0 n = 0 := by
  /-
    n : Nat
    ⊢ Eq (Nat.maxPowDiv 0 n) 0
  -/
  dsimp [maxPowDiv]
  /-
    n : Nat
    ⊢ Eq (Nat.maxPowDiv.go 0 0 n) 0
  -/
  rw [maxPowDiv.go]
  /-
    n : Nat
    ⊢ Eq (ite (And (LT.lt 1 0) (And (LT.lt 0 n) (Eq (HMod.hMod n 0) 0))) (Nat.maxP …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zero {p : ℕ} : maxPowDiv p 0 = 0 := by
  /-
    p : Nat
    ⊢ Eq (p.maxPowDiv 0) 0
  -/
  dsimp [maxPowDiv]
  /-
    p : Nat
    ⊢ Eq (Nat.maxPowDiv.go 0 p 0) 0
  -/
  rw [maxPowDiv.go]
  /-
    p : Nat
    ⊢ Eq (ite (And (LT.lt 1 p) (And (LT.lt 0 0) (Eq (HMod.hMod 0 p) 0))) (Nat.maxP …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem base_mul_eq_succ {p n : ℕ} (hp : 1 < p) (hn : 0 < n) :
    p.maxPowDiv (p*n) = p.maxPowDiv n + 1 := by
  /-
    p n : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    ⊢ Eq (p.maxPowDiv (HMul.hMul p n)) (HAdd.hAdd (p.maxPowDiv n) 1)
  -/
  have : 0 < p := lt_trans (b := 1) (by simp) hp
  /-
    p n : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    this : LT.lt 0 p
    ⊢ Eq (p.maxPowDiv (HMul.hMul p n)) (HAdd.hAdd (p.maxPowDiv n) 1)
  -/
  dsimp [maxPowDiv]
  /-
    p n : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    this : LT.lt 0 p
    ⊢ Eq (Nat.maxPowDiv.go 0 p (HMul.hMul p n)) (HAdd.hAdd (Nat.maxPowDiv.go 0 p n …
  -/
  rw [maxPowDiv.go, if_pos, mul_div_right _ this]
    /-
      p n : Nat
      hp : LT.lt 1 p
      hn : LT.lt 0 n
      this : LT.lt 0 p
      ⊢ Eq (Nat.maxPowDiv.go (HAdd.hAdd 0 1) p n) (HAdd.hAdd (Nat.maxPowDiv.go 0 p n …
    -/
  · apply go_succ
    /-
      🎉 no goals
    -/
    /-
      case hc
      p n : Nat
      hp : LT.lt 1 p
      hn : LT.lt 0 n
      this : LT.lt 0 p
      ⊢ And (LT.lt 1 p) (And (LT.lt 0 (HMul.hMul p n)) (Eq (HMod.hMod (HMul.hMul p n …
    -/
  · refine ⟨hp, ?_, by simp⟩
    /-
      case hc
      p n : Nat
      hp : LT.lt 1 p
      hn : LT.lt 0 n
      this : LT.lt 0 p
      ⊢ LT.lt 0 (HMul.hMul p n)
    -/
    apply Nat.mul_pos this hn
    /-
      🎉 no goals
    -/


theorem base_pow_mul {p n exp : ℕ} (hp : 1 < p) (hn : 0 < n) :
    p.maxPowDiv (p ^ exp * n) = p.maxPowDiv n + exp := by
  match exp with
  | 0 => simp
  | e + 1 =>
    rw [Nat.pow_succ, mul_assoc, mul_comm, mul_assoc, base_mul_eq_succ hp, mul_comm,
      base_pow_mul hp hn]
    · ac_rfl
    · apply Nat.mul_pos hn <| pow_pos (pos_of_gt hp) e


theorem pow_dvd (p n : ℕ) : p ^ (p.maxPowDiv n) ∣ n := by
  /-
    p n : Nat
    ⊢ Dvd.dvd (HPow.hPow p (p.maxPowDiv n)) n
  -/
  dsimp [maxPowDiv]
  /-
    p n : Nat
    ⊢ Dvd.dvd (HPow.hPow p (Nat.maxPowDiv.go 0 p n)) n
  -/
  rw [go]
  /-
    p n : Nat
    ⊢ Dvd.dvd (HPow.hPow p (ite (And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n …
  -/
  by_cases h : (1 < p ∧ 0 < n ∧ n % p = 0)
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0))
      ⊢ Dvd.dvd (HPow.hPow p (ite (And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n …
    -/
  · have : n / p < n := by apply Nat.div_lt_self <;> aesop
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0))
      this : LT.lt (HDiv.hDiv n p) n
      ⊢ Dvd.dvd (HPow.hPow p (ite (And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n …
    -/
    rw [if_pos h]
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0))
      this : LT.lt (HDiv.hDiv n p) n
      ⊢ Dvd.dvd (HPow.hPow p (Nat.maxPowDiv.go (HAdd.hAdd 0 1) p (HDiv.hDiv n p))) n
    -/
    have ⟨c,hc⟩ := pow_dvd p (n / p)
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0))
      this : LT.lt (HDiv.hDiv n p) n
      c : Nat
      hc : Eq (HDiv.hDiv n p) (HMul.hMul (HPow.hPow p (p.maxPowDiv (HDiv.hDiv n p))) …
      ⊢ Dvd.dvd (HPow.hPow p (Nat.maxPowDiv.go (HAdd.hAdd 0 1) p (HDiv.hDiv n p))) n
    -/
    rw [go_succ, pow_succ]
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0))
      this : LT.lt (HDiv.hDiv n p) n
      c : Nat
      hc : Eq (HDiv.hDiv n p) (HMul.hMul (HPow.hPow p (p.maxPowDiv (HDiv.hDiv n p))) …
      ⊢ Dvd.dvd (HMul.hMul (HPow.hPow p (Nat.maxPowDiv.go 0 p (HDiv.hDiv n p))) p) n
    -/
    nth_rw 2 [← mod_add_div' n p]
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0))
      this : LT.lt (HDiv.hDiv n p) n
      c : Nat
      hc : Eq (HDiv.hDiv n p) (HMul.hMul (HPow.hPow p (p.maxPowDiv (HDiv.hDiv n p))) …
      ⊢ Dvd.dvd (HMul.hMul (HPow.hPow p (Nat.maxPowDiv.go 0 p (HDiv.hDiv n p))) p) ( …
    -/
    rw [h.right.right, zero_add]
    /-
      case pos
      p n : Nat
      h : And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0))
      this : LT.lt (HDiv.hDiv n p) n
      c : Nat
      hc : Eq (HDiv.hDiv n p) (HMul.hMul (HPow.hPow p (p.maxPowDiv (HDiv.hDiv n p))) …
      ⊢ Dvd.dvd (HMul.hMul (HPow.hPow p (Nat.maxPowDiv.go 0 p (HDiv.hDiv n p))) p) ( …
    -/
    exact ⟨c,by nth_rw 1 [hc]; ac_rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      p n : Nat
      h : Not (And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0)))
      ⊢ Dvd.dvd (HPow.hPow p (ite (And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n …
    -/
  · rw [if_neg h]
    /-
      case neg
      p n : Nat
      h : Not (And (LT.lt 1 p) (And (LT.lt 0 n) (Eq (HMod.hMod n p) 0)))
      ⊢ Dvd.dvd (HPow.hPow p 0) n
    -/
    simp
    /-
      🎉 no goals
    -/


theorem le_of_dvd {p n pow : ℕ} (hp : 1 < p) (hn : 0 < n) (h : p ^ pow ∣ n) :
    pow ≤ p.maxPowDiv n := by
  /-
    p n pow : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    h : Dvd.dvd (HPow.hPow p pow) n
    ⊢ LE.le pow (p.maxPowDiv n)
  -/
  have ⟨c, hc⟩ := h
  have : 0 < c := by
    apply Nat.pos_of_ne_zero
    intro h'
    rw [h',mul_zero] at hc
    exact not_eq_zero_of_lt hn hc
  /-
    p n pow : Nat
    hp : LT.lt 1 p
    hn : LT.lt 0 n
    h : Dvd.dvd (HPow.hPow p pow) n
    c : Nat
    hc : Eq n (HMul.hMul (HPow.hPow p pow) c)
    this : LT.lt 0 c
    ⊢ LE.le pow (p.maxPowDiv n)
  -/
  simp [hc, base_pow_mul hp this]
  /-
    🎉 no goals
  -/


