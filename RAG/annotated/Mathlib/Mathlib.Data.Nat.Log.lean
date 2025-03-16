/-- `log b n`, is the logarithm of natural number `n` in base `b`. It returns the largest `k : ℕ`
such that `b^k ≤ n`, so if `b^k = n`, it returns exactly `k`. -/
@[pp_nodot]
def log (b : ℕ) : ℕ → ℕ
  | n => if h : b ≤ n ∧ 1 < b then log b (n / b) + 1 else 0
decreasing_by
  -- putting this in the def triggers the `unusedHavesSuffices` linter:
  -- https://github.com/leanprover-community/batteries/issues/428
  have : n / b < n := div_lt_self ((Nat.zero_lt_one.trans h.2).trans_le h.1) h.2
  decreasing_trivial


@[simp]
theorem log_eq_zero_iff {b n : ℕ} : log b n = 0 ↔ n < b ∨ b ≤ 1 := by
  /-
    b n : Nat
    ⊢ Iff (Eq (Nat.log b n) 0) (Or (LT.lt n b) (LE.le b 1))
  -/
  rw [log, dite_eq_right_iff]
  /-
    b n : Nat
    ⊢ Iff (And (LE.le b n) (LT.lt 1 b) → Eq (HAdd.hAdd (Nat.log b (HDiv.hDiv n b)) …
  -/
  simp only [Nat.add_eq_zero_iff, Nat.one_ne_zero, and_false, imp_false, not_and_or, not_le, not_lt]
  /-
    🎉 no goals
  -/


theorem log_of_lt {b n : ℕ} (hb : n < b) : log b n = 0 :=
  log_eq_zero_iff.2 (Or.inl hb)


theorem log_of_left_le_one {b : ℕ} (hb : b ≤ 1) (n) : log b n = 0 :=
  log_eq_zero_iff.2 (Or.inr hb)


@[simp]
theorem log_pos_iff {b n : ℕ} : 0 < log b n ↔ b ≤ n ∧ 1 < b := by
  /-
    b n : Nat
    ⊢ Iff (LT.lt 0 (Nat.log b n)) (And (LE.le b n) (LT.lt 1 b))
  -/
  rw [Nat.pos_iff_ne_zero, Ne, log_eq_zero_iff, not_or, not_lt, not_le]
  /-
    🎉 no goals
  -/


@[bound]
theorem log_pos {b n : ℕ} (hb : 1 < b) (hbn : b ≤ n) : 0 < log b n :=
  log_pos_iff.2 ⟨hbn, hb⟩


theorem log_of_one_lt_of_le {b n : ℕ} (h : 1 < b) (hn : b ≤ n) : log b n = log b (n / b) + 1 := by
  /-
    b n : Nat
    h : LT.lt 1 b
    hn : LE.le b n
    ⊢ Eq (Nat.log b n) (HAdd.hAdd (Nat.log b (HDiv.hDiv n b)) 1)
  -/
  rw [log]
  /-
    b n : Nat
    h : LT.lt 1 b
    hn : LE.le b n
    ⊢ Eq (dite (And (LE.le b n) (LT.lt 1 b)) (fun h => HAdd.hAdd (Nat.log b (HDiv. …
  -/
  exact if_pos ⟨hn, h⟩
  /-
    🎉 no goals
  -/


@[simp] lemma log_zero_left : ∀ n, log 0 n = 0 := log_of_left_le_one <| Nat.zero_le _


@[simp]
theorem log_zero_right (b : ℕ) : log b 0 = 0 :=
  log_eq_zero_iff.2 (le_total 1 b)


@[simp]
theorem log_one_left : ∀ n, log 1 n = 0 :=
  log_of_left_le_one le_rfl


@[simp]
theorem log_one_right (b : ℕ) : log b 1 = 0 :=
  log_eq_zero_iff.2 (lt_or_le _ _)


/-- `pow b` and `log b` (almost) form a Galois connection. See also `Nat.pow_le_of_le_log` and
`Nat.le_log_of_pow_le` for individual implications under weaker assumptions. -/
theorem pow_le_iff_le_log {b : ℕ} (hb : 1 < b) {x y : ℕ} (hy : y ≠ 0) :
    b ^ x ≤ y ↔ x ≤ log b y := by
  /-
    b : Nat
    hb : LT.lt 1 b
    x y : Nat
    hy : Ne y 0
    ⊢ Iff (LE.le (HPow.hPow b x) y) (LE.le x (Nat.log b y))
  -/
  induction y using Nat.strong_induction_on generalizing x with | h y ih => ?_
  cases x with
  | zero => dsimp; omega
  | succ x =>
    rw [log]; split_ifs with h
    · have b_pos : 0 < b := lt_of_succ_lt hb
      rw [Nat.add_le_add_iff_right, ← ih (y / b) (div_lt_self
        (Nat.pos_iff_ne_zero.2 hy) hb) (Nat.div_pos h.1 b_pos).ne', le_div_iff_mul_le b_pos,
        pow_succ', Nat.mul_comm]
    · exact iff_of_false (fun hby => h ⟨(le_self_pow x.succ_ne_zero _).trans hby, hb⟩)
        (not_succ_le_zero _)


theorem lt_pow_iff_log_lt {b : ℕ} (hb : 1 < b) {x y : ℕ} (hy : y ≠ 0) : y < b ^ x ↔ log b y < x :=
  lt_iff_lt_of_le_iff_le (pow_le_iff_le_log hb hy)


theorem pow_le_of_le_log {b x y : ℕ} (hy : y ≠ 0) (h : x ≤ log b y) : b ^ x ≤ y := by
  /-
    b x y : Nat
    hy : Ne y 0
    h : LE.le x (Nat.log b y)
    ⊢ LE.le (HPow.hPow b x) y
  -/
  refine (le_or_lt b 1).elim (fun hb => ?_) fun hb => (pow_le_iff_le_log hb hy).2 h
  /-
    b x y : Nat
    hy : Ne y 0
    h : LE.le x (Nat.log b y)
    hb : LE.le b 1
    ⊢ LE.le (HPow.hPow b x) y
  -/
  rw [log_of_left_le_one hb, Nat.le_zero] at h
  /-
    b x y : Nat
    hy : Ne y 0
    h : Eq x 0
    hb : LE.le b 1
    ⊢ LE.le (HPow.hPow b x) y
  -/
  rwa [h, Nat.pow_zero, one_le_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem le_log_of_pow_le {b x y : ℕ} (hb : 1 < b) (h : b ^ x ≤ y) : x ≤ log b y := by
  /-
    b x y : Nat
    hb : LT.lt 1 b
    h : LE.le (HPow.hPow b x) y
    ⊢ LE.le x (Nat.log b y)
  -/
  rcases ne_or_eq y 0 with (hy | rfl)
  /-
    case inl
    b x y : Nat
    hb : LT.lt 1 b
    h : LE.le (HPow.hPow b x) y
    hy : Ne y 0
    ⊢ LE.le x (Nat.log b y)
  -/
  exacts [(pow_le_iff_le_log hb hy).1 h, (h.not_lt (Nat.pow_pos (Nat.zero_lt_one.trans hb))).elim]
  /-
    🎉 no goals
  -/


theorem pow_log_le_self (b : ℕ) {x : ℕ} (hx : x ≠ 0) : b ^ log b x ≤ x :=
  pow_le_of_le_log hx le_rfl


theorem log_lt_of_lt_pow {b x y : ℕ} (hy : y ≠ 0) : y < b ^ x → log b y < x :=
  lt_imp_lt_of_le_imp_le (pow_le_of_le_log hy)


theorem lt_pow_of_log_lt {b x y : ℕ} (hb : 1 < b) : log b y < x → y < b ^ x :=
  lt_imp_lt_of_le_imp_le (le_log_of_pow_le hb)


lemma log_lt_self (b : ℕ) {x : ℕ} (hx : x ≠ 0) : log b x < x :=
  match le_or_lt b 1 with
  | .inl h => log_of_left_le_one h x ▸ Nat.pos_iff_ne_zero.2 hx
  | .inr h => log_lt_of_lt_pow hx <| Nat.lt_pow_self h


lemma log_le_self (b x : ℕ) : log b x ≤ x :=
                        /-
                          b x : Nat
                          hx : Eq x 0
                          ⊢ LE.le (Nat.log b x) x
                        -/
  if hx : x = 0 then by simp [hx]
                        /-
                          🎉 no goals
                        -/
  else (log_lt_self b hx).le


theorem lt_pow_succ_log_self {b : ℕ} (hb : 1 < b) (x : ℕ) : x < b ^ (log b x).succ :=
  lt_pow_of_log_lt hb (lt_succ_self _)


theorem log_eq_iff {b m n : ℕ} (h : m ≠ 0 ∨ 1 < b ∧ n ≠ 0) :
    log b n = m ↔ b ^ m ≤ n ∧ n < b ^ (m + 1) := by
  /-
    b m n : Nat
    h : Or (Ne m 0) (And (LT.lt 1 b) (Ne n 0))
    ⊢ Iff (Eq (Nat.log b n) m) (And (LE.le (HPow.hPow b m) n) (LT.lt n (HPow.hPow  …
  -/
  rcases em (1 < b ∧ n ≠ 0) with (⟨hb, hn⟩ | hbn)
  · rw [le_antisymm_iff, ← Nat.lt_succ_iff, ← pow_le_iff_le_log, ← lt_pow_iff_log_lt,
                    /-
                      case inl.intro.hb
                      b m n : Nat
                      h : Or (Ne m 0) (And (LT.lt 1 b) (Ne n 0))
                      hb : LT.lt 1 b
                      hn : Ne n 0
                      ⊢ LT.lt 1 b
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
      and_comm] <;> assumption
                    /-
                      🎉 no goals
                    -/
  /-
    case inr
    b m n : Nat
    h : Or (Ne m 0) (And (LT.lt 1 b) (Ne n 0))
    hbn : Not (And (LT.lt 1 b) (Ne n 0))
    ⊢ Iff (Eq (Nat.log b n) m) (And (LE.le (HPow.hPow b m) n) (LT.lt n (HPow.hPow  …
  -/
  have hm : m ≠ 0 := h.resolve_right hbn
  /-
    case inr
    b m n : Nat
    h : Or (Ne m 0) (And (LT.lt 1 b) (Ne n 0))
    hbn : Not (And (LT.lt 1 b) (Ne n 0))
    hm : Ne m 0
    ⊢ Iff (Eq (Nat.log b n) m) (And (LE.le (HPow.hPow b m) n) (LT.lt n (HPow.hPow  …
  -/
  rw [not_and_or, not_lt, Ne, not_not] at hbn
  /-
    case inr
    b m n : Nat
    h : Or (Ne m 0) (And (LT.lt 1 b) (Ne n 0))
    hbn : Or (LE.le b 1) (Eq n 0)
    hm : Ne m 0
    ⊢ Iff (Eq (Nat.log b n) m) (And (LE.le (HPow.hPow b m) n) (LT.lt n (HPow.hPow  …
  -/
  rcases hbn with (hb | rfl)
    /-
      case inr.inl
      b m n : Nat
      h : Or (Ne m 0) (And (LT.lt 1 b) (Ne n 0))
      hm : Ne m 0
      hb : LE.le b 1
      ⊢ Iff (Eq (Nat.log b n) m) (And (LE.le (HPow.hPow b m) n) (LT.lt n (HPow.hPow  …
    -/
  · obtain rfl | rfl := le_one_iff_eq_zero_or_eq_one.1 hb
    any_goals
      simp only [ne_eq, zero_eq, reduceSucc, lt_self_iff_false,  not_lt_zero, false_and, or_false]
        at h
      simp [h, eq_comm (a := 0), Nat.zero_pow (Nat.pos_iff_ne_zero.2 _)] <;> omega
    /-
      case inr.inr
      b m : Nat
      hm : Ne m 0
      h : Or (Ne m 0) (And (LT.lt 1 b) (Ne 0 0))
      ⊢ Iff (Eq (Nat.log b 0) m) (And (LE.le (HPow.hPow b m) 0) (LT.lt 0 (HPow.hPow  …
    -/
  · simp [@eq_comm _ 0, hm]
    /-
      🎉 no goals
    -/


theorem log_eq_of_pow_le_of_lt_pow {b m n : ℕ} (h₁ : b ^ m ≤ n) (h₂ : n < b ^ (m + 1)) :
    log b n = m := by
  /-
    b m n : Nat
    h₁ : LE.le (HPow.hPow b m) n
    h₂ : LT.lt n (HPow.hPow b (HAdd.hAdd m 1))
    ⊢ Eq (Nat.log b n) m
  -/
  rcases eq_or_ne m 0 with (rfl | hm)
    /-
      case inl
      b n : Nat
      h₁ : LE.le (HPow.hPow b 0) n
      h₂ : LT.lt n (HPow.hPow b (HAdd.hAdd 0 1))
      ⊢ Eq (Nat.log b n) 0
    -/
  · rw [Nat.pow_one] at h₂
    /-
      case inl
      b n : Nat
      h₁ : LE.le (HPow.hPow b 0) n
      h₂ : LT.lt n b
      ⊢ Eq (Nat.log b n) 0
    -/
    exact log_of_lt h₂
    /-
      🎉 no goals
    -/
    /-
      case inr
      b m n : Nat
      h₁ : LE.le (HPow.hPow b m) n
      h₂ : LT.lt n (HPow.hPow b (HAdd.hAdd m 1))
      hm : Ne m 0
      ⊢ Eq (Nat.log b n) m
    -/
  · exact (log_eq_iff (Or.inl hm)).2 ⟨h₁, h₂⟩
    /-
      🎉 no goals
    -/


theorem log_pow {b : ℕ} (hb : 1 < b) (x : ℕ) : log b (b ^ x) = x :=
  log_eq_of_pow_le_of_lt_pow le_rfl (Nat.pow_lt_pow_right hb x.lt_succ_self)


theorem log_eq_one_iff' {b n : ℕ} : log b n = 1 ↔ b ≤ n ∧ n < b * b := by
  /-
    b n : Nat
    ⊢ Iff (Eq (Nat.log b n) 1) (And (LE.le b n) (LT.lt n (HMul.hMul b b)))
  -/
  rw [log_eq_iff (Or.inl Nat.one_ne_zero), Nat.pow_add, Nat.pow_one]
  /-
    🎉 no goals
  -/


theorem log_eq_one_iff {b n : ℕ} : log b n = 1 ↔ n < b * b ∧ 1 < b ∧ b ≤ n :=
  log_eq_one_iff'.trans
    ⟨fun h => ⟨h.2, lt_mul_self_iff.1 (h.1.trans_lt h.2), h.1⟩, fun h => ⟨h.2.2, h.1⟩⟩


theorem log_mul_base {b n : ℕ} (hb : 1 < b) (hn : n ≠ 0) : log b (n * b) = log b n + 1 := by
  /-
    b n : Nat
    hb : LT.lt 1 b
    hn : Ne n 0
    ⊢ Eq (Nat.log b (HMul.hMul n b)) (HAdd.hAdd (Nat.log b n) 1)
  -/
  apply log_eq_of_pow_le_of_lt_pow <;> rw [pow_succ', Nat.mul_comm b]
  exacts [Nat.mul_le_mul_right _ (pow_log_le_self _ hn),
    (Nat.mul_lt_mul_right (Nat.zero_lt_one.trans hb)).2 (lt_pow_succ_log_self hb _)]


theorem pow_log_le_add_one (b : ℕ) : ∀ x, b ^ log b x ≤ x + 1
            /-
              b : Nat
              ⊢ LE.le (HPow.hPow b (Nat.log b 0)) (HAdd.hAdd 0 1)
            -/
  | 0 => by rw [log_zero_right, Nat.pow_zero]
            /-
              🎉 no goals
            -/
  | x + 1 => (pow_log_le_self b x.succ_ne_zero).trans (x + 1).le_succ


theorem log_monotone {b : ℕ} : Monotone (log b) := by
  /-
    b : Nat
    ⊢ Monotone (Nat.log b)
  -/
  refine monotone_nat_of_le_succ fun n => ?_
  /-
    b n : Nat
    ⊢ LE.le (Nat.log b n) (Nat.log b (HAdd.hAdd n 1))
  -/
  rcases le_or_lt b 1 with hb | hb
    /-
      case inl
      b n : Nat
      hb : LE.le b 1
      ⊢ LE.le (Nat.log b n) (Nat.log b (HAdd.hAdd n 1))
    -/
  · rw [log_of_left_le_one hb]
    /-
      case inl
      b n : Nat
      hb : LE.le b 1
      ⊢ LE.le 0 (Nat.log b (HAdd.hAdd n 1))
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/
    /-
      case inr
      b n : Nat
      hb : LT.lt 1 b
      ⊢ LE.le (Nat.log b n) (Nat.log b (HAdd.hAdd n 1))
    -/
  · exact le_log_of_pow_le hb (pow_log_le_add_one _ _)
    /-
      🎉 no goals
    -/


@[mono]
theorem log_mono_right {b n m : ℕ} (h : n ≤ m) : log b n ≤ log b m :=
  log_monotone h


@[mono]
theorem log_anti_left {b c n : ℕ} (hc : 1 < c) (hb : c ≤ b) : log b n ≤ log c n := by
  /-
    b c n : Nat
    hc : LT.lt 1 c
    hb : LE.le c b
    ⊢ LE.le (Nat.log b n) (Nat.log c n)
  -/
  rcases eq_or_ne n 0 with (rfl | hn); · rw [log_zero_right, log_zero_right]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    b c n : Nat
    hc : LT.lt 1 c
    hb : LE.le c b
    hn : Ne n 0
    ⊢ LE.le (Nat.log b n) (Nat.log c n)
  -/
  apply le_log_of_pow_le hc
  calc
    c ^ log b n ≤ b ^ log b n := Nat.pow_le_pow_left hb _
    _ ≤ n := pow_log_le_self _ hn


theorem log_antitone_left {n : ℕ} : AntitoneOn (fun b => log b n) (Set.Ioi 1) := fun _ hc _ _ hb =>
  log_anti_left (Set.mem_Iio.1 hc) hb


@[simp]
theorem log_div_base (b n : ℕ) : log b (n / b) = log b n - 1 := by
  /-
    b n : Nat
    ⊢ Eq (Nat.log b (HDiv.hDiv n b)) (HSub.hSub (Nat.log b n) 1)
  -/
  rcases le_or_lt b 1 with hb | hb
    /-
      case inl
      b n : Nat
      hb : LE.le b 1
      ⊢ Eq (Nat.log b (HDiv.hDiv n b)) (HSub.hSub (Nat.log b n) 1)
    -/
  · rw [log_of_left_le_one hb, log_of_left_le_one hb, Nat.zero_sub]
    /-
      🎉 no goals
    -/
  /-
    case inr
    b n : Nat
    hb : LT.lt 1 b
    ⊢ Eq (Nat.log b (HDiv.hDiv n b)) (HSub.hSub (Nat.log b n) 1)
  -/
  rcases lt_or_le n b with h | h
    /-
      case inr.inl
      b n : Nat
      hb : LT.lt 1 b
      h : LT.lt n b
      ⊢ Eq (Nat.log b (HDiv.hDiv n b)) (HSub.hSub (Nat.log b n) 1)
    -/
  · rw [div_eq_of_lt h, log_of_lt h, log_zero_right]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    b n : Nat
    hb : LT.lt 1 b
    h : LE.le b n
    ⊢ Eq (Nat.log b (HDiv.hDiv n b)) (HSub.hSub (Nat.log b n) 1)
  -/
  rw [log_of_one_lt_of_le hb h, Nat.add_sub_cancel_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem log_div_mul_self (b n : ℕ) : log b (n / b * b) = log b n := by
  /-
    b n : Nat
    ⊢ Eq (Nat.log b (HMul.hMul (HDiv.hDiv n b) b)) (Nat.log b n)
  -/
  rcases le_or_lt b 1 with hb | hb
    /-
      case inl
      b n : Nat
      hb : LE.le b 1
      ⊢ Eq (Nat.log b (HMul.hMul (HDiv.hDiv n b) b)) (Nat.log b n)
    -/
  · rw [log_of_left_le_one hb, log_of_left_le_one hb]
    /-
      🎉 no goals
    -/
  /-
    case inr
    b n : Nat
    hb : LT.lt 1 b
    ⊢ Eq (Nat.log b (HMul.hMul (HDiv.hDiv n b) b)) (Nat.log b n)
  -/
  rcases lt_or_le n b with h | h
    /-
      case inr.inl
      b n : Nat
      hb : LT.lt 1 b
      h : LT.lt n b
      ⊢ Eq (Nat.log b (HMul.hMul (HDiv.hDiv n b) b)) (Nat.log b n)
    -/
  · rw [div_eq_of_lt h, Nat.zero_mul, log_zero_right, log_of_lt h]
    /-
      🎉 no goals
    -/
  rw [log_mul_base hb (Nat.div_pos h (by omega)).ne', log_div_base,
    Nat.sub_add_cancel (succ_le_iff.2 <| log_pos hb h)]


theorem add_pred_div_lt {b n : ℕ} (hb : 1 < b) (hn : 2 ≤ n) : (n + b - 1) / b < n := by
  rw [div_lt_iff_lt_mul (by omega), ← succ_le_iff, ← pred_eq_sub_one,
    succ_pred_eq_of_pos (by omega)]
  /-
    b n : Nat
    hb : LT.lt 1 b
    hn : LE.le 2 n
    ⊢ LE.le (HAdd.hAdd n b) (HMul.hMul n b)
  -/
  exact Nat.add_le_mul hn hb
  /-
    🎉 no goals
  -/


lemma log2_eq_log_two {n : ℕ} : Nat.log2 n = Nat.log 2 n := by
  /-
    n : Nat
    ⊢ Eq n.log2 (Nat.log 2 n)
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      ⊢ Eq (Nat.log2 0) (Nat.log 2 0)
    -/
  · rw [log2_zero, log_zero_right]
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    hn : Ne n 0
    ⊢ Eq n.log2 (Nat.log 2 n)
  -/
  apply eq_of_forall_le_iff
  /-
    case inr.H
    n : Nat
    hn : Ne n 0
    ⊢ ∀ (c : Nat), Iff (LE.le c n.log2) (LE.le c (Nat.log 2 n))
  -/
  intro m
  /-
    case inr.H
    n : Nat
    hn : Ne n 0
    m : Nat
    ⊢ Iff (LE.le m n.log2) (LE.le m (Nat.log 2 n))
  -/
  rw [Nat.le_log2 hn, ← Nat.pow_le_iff_le_log Nat.one_lt_two hn]
  /-
    🎉 no goals
  -/


/-- `clog b n`, is the upper logarithm of natural number `n` in base `b`. It returns the smallest
`k : ℕ` such that `n ≤ b^k`, so if `b^k = n`, it returns exactly `k`. -/
@[pp_nodot]
def clog (b : ℕ) : ℕ → ℕ
  | n => if h : 1 < b ∧ 1 < n then clog b ((n + b - 1) / b) + 1 else 0
decreasing_by
  -- putting this in the def triggers the `unusedHavesSuffices` linter:
  -- https://github.com/leanprover-community/batteries/issues/428
  have : (n + b - 1) / b < n := add_pred_div_lt h.1 h.2
  decreasing_trivial


theorem clog_of_left_le_one {b : ℕ} (hb : b ≤ 1) (n : ℕ) : clog b n = 0 := by
  /-
    b : Nat
    hb : LE.le b 1
    n : Nat
    ⊢ Eq (Nat.clog b n) 0
  -/
  rw [clog, dif_neg fun h : 1 < b ∧ 1 < n => h.1.not_le hb]
  /-
    🎉 no goals
  -/


theorem clog_of_right_le_one {n : ℕ} (hn : n ≤ 1) (b : ℕ) : clog b n = 0 := by
  /-
    n : Nat
    hn : LE.le n 1
    b : Nat
    ⊢ Eq (Nat.clog b n) 0
  -/
  rw [clog, dif_neg fun h : 1 < b ∧ 1 < n => h.2.not_le hn]
  /-
    🎉 no goals
  -/


@[simp] lemma clog_zero_left (n : ℕ) : clog 0 n = 0 := clog_of_left_le_one (Nat.zero_le _) _


@[simp] lemma clog_zero_right (b : ℕ) : clog b 0 = 0 := clog_of_right_le_one (Nat.zero_le _) _


@[simp]
theorem clog_one_left (n : ℕ) : clog 1 n = 0 :=
  clog_of_left_le_one le_rfl _


@[simp]
theorem clog_one_right (b : ℕ) : clog b 1 = 0 :=
  clog_of_right_le_one le_rfl _


theorem clog_of_two_le {b n : ℕ} (hb : 1 < b) (hn : 2 ≤ n) :
                                                  /-
                                                    b n : Nat
                                                    hb : LT.lt 1 b
                                                    hn : LE.le 2 n
                                                    ⊢ Eq (Nat.clog b n) (HAdd.hAdd (Nat.clog b (HDiv.hDiv (HSub.hSub (HAdd.hAdd n  …
                                                  -/
    clog b n = clog b ((n + b - 1) / b) + 1 := by rw [clog, dif_pos (⟨hb, hn⟩ : 1 < b ∧ 1 < n)]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem clog_pos {b n : ℕ} (hb : 1 < b) (hn : 2 ≤ n) : 0 < clog b n := by
  /-
    b n : Nat
    hb : LT.lt 1 b
    hn : LE.le 2 n
    ⊢ LT.lt 0 (Nat.clog b n)
  -/
  rw [clog_of_two_le hb hn]
  /-
    b n : Nat
    hb : LT.lt 1 b
    hn : LE.le 2 n
    ⊢ LT.lt 0 (HAdd.hAdd (Nat.clog b (HDiv.hDiv (HSub.hSub (HAdd.hAdd n b) 1) b)) 1)
  -/
  exact zero_lt_succ _
  /-
    🎉 no goals
  -/


theorem clog_eq_one {b n : ℕ} (hn : 2 ≤ n) (h : n ≤ b) : clog b n = 1 := by
  /-
    b n : Nat
    hn : LE.le 2 n
    h : LE.le n b
    ⊢ Eq (Nat.clog b n) 1
  -/
  rw [clog_of_two_le (hn.trans h) hn, clog_of_right_le_one]
  /-
    case hn
    b n : Nat
    hn : LE.le 2 n
    h : LE.le n b
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (HAdd.hAdd n b) 1) b) 1
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  rw [← Nat.lt_succ_iff, Nat.div_lt_iff_lt_mul] <;> omega
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- `clog b` and `pow b` form a Galois connection. -/
theorem le_pow_iff_clog_le {b : ℕ} (hb : 1 < b) {x y : ℕ} : x ≤ b ^ y ↔ clog b x ≤ y := by
  /-
    b : Nat
    hb : LT.lt 1 b
    x y : Nat
    ⊢ Iff (LE.le x (HPow.hPow b y)) (LE.le (Nat.clog b x) y)
  -/
  induction x using Nat.strong_induction_on generalizing y with | h x ih => ?_
  /-
    case h
    b : Nat
    hb : LT.lt 1 b
    x : Nat
    ih : ∀ (m : Nat), LT.lt m x → ∀ {y : Nat}, Iff (LE.le m (HPow.hPow b y)) (LE.l …
    y : Nat
    ⊢ Iff (LE.le x (HPow.hPow b y)) (LE.le (Nat.clog b x) y)
  -/
  cases y
    /-
      case h.zero
      b : Nat
      hb : LT.lt 1 b
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {y : Nat}, Iff (LE.le m (HPow.hPow b y)) (LE.l …
      ⊢ Iff (LE.le x (HPow.hPow b 0)) (LE.le (Nat.clog b x) 0)
    -/
  · rw [Nat.pow_zero]
    /-
      case h.zero
      b : Nat
      hb : LT.lt 1 b
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {y : Nat}, Iff (LE.le m (HPow.hPow b y)) (LE.l …
      ⊢ Iff (LE.le x 1) (LE.le (Nat.clog b x) 0)
    -/
    refine ⟨fun h => (clog_of_right_le_one h b).le, ?_⟩
    /-
      case h.zero
      b : Nat
      hb : LT.lt 1 b
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {y : Nat}, Iff (LE.le m (HPow.hPow b y)) (LE.l …
      ⊢ LE.le (Nat.clog b x) 0 → LE.le x 1
    -/
    simp_rw [← not_lt]
    /-
      case h.zero
      b : Nat
      hb : LT.lt 1 b
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {y : Nat}, Iff (LE.le m (HPow.hPow b y)) (LE.l …
      ⊢ Not (LT.lt 0 (Nat.clog b x)) → Not (LT.lt 1 x)
    -/
    contrapose!
    /-
      case h.zero
      b : Nat
      hb : LT.lt 1 b
      x : Nat
      ih : ∀ (m : Nat), LT.lt m x → ∀ {y : Nat}, Iff (LE.le m (HPow.hPow b y)) (LE.l …
      ⊢ LT.lt 1 x → LT.lt 0 (Nat.clog b x)
    -/
    exact clog_pos hb
    /-
      🎉 no goals
    -/
  /-
    case h.succ
    b : Nat
    hb : LT.lt 1 b
    x : Nat
    ih : ∀ (m : Nat), LT.lt m x → ∀ {y : Nat}, Iff (LE.le m (HPow.hPow b y)) (LE.l …
    n✝ : Nat
    ⊢ Iff (LE.le x (HPow.hPow b (HAdd.hAdd n✝ 1))) (LE.le (Nat.clog b x) (HAdd.hAd …
  -/
  have b_pos : 0 < b := zero_lt_of_lt hb
  /-
    case h.succ
    b : Nat
    hb : LT.lt 1 b
    x : Nat
    ih : ∀ (m : Nat), LT.lt m x → ∀ {y : Nat}, Iff (LE.le m (HPow.hPow b y)) (LE.l …
    n✝ : Nat
    b_pos : LT.lt 0 b
    ⊢ Iff (LE.le x (HPow.hPow b (HAdd.hAdd n✝ 1))) (LE.le (Nat.clog b x) (HAdd.hAd …
  -/
  rw [clog]; split_ifs with h
  · rw [Nat.add_le_add_iff_right, ← ih ((x + b - 1) / b) (add_pred_div_lt hb h.2),
      Nat.div_le_iff_le_mul_add_pred b_pos, Nat.mul_comm b, ← Nat.pow_succ,
      Nat.add_sub_assoc (Nat.succ_le_of_lt b_pos), Nat.add_le_add_iff_right]
  · exact iff_of_true ((not_lt.1 (not_and.1 h hb)).trans <| succ_le_of_lt <| Nat.pow_pos b_pos)
      (zero_le _)


theorem pow_lt_iff_lt_clog {b : ℕ} (hb : 1 < b) {x y : ℕ} : b ^ y < x ↔ y < clog b x :=
  lt_iff_lt_of_le_iff_le (le_pow_iff_clog_le hb)


theorem clog_pow (b x : ℕ) (hb : 1 < b) : clog b (b ^ x) = x :=
                                 /-
                                   b x : Nat
                                   hb : LT.lt 1 b
                                   z : Nat
                                   ⊢ Iff (LE.le (Nat.clog b (HPow.hPow b x)) z) (LE.le x z)
                                 -/
  eq_of_forall_ge_iff fun z ↦ by rw [← le_pow_iff_clog_le hb, Nat.pow_le_pow_iff_right hb]
                                 /-
                                   🎉 no goals
                                 -/


theorem pow_pred_clog_lt_self {b : ℕ} (hb : 1 < b) {x : ℕ} (hx : 1 < x) :
    b ^ (clog b x).pred < x := by
  /-
    b : Nat
    hb : LT.lt 1 b
    x : Nat
    hx : LT.lt 1 x
    ⊢ LT.lt (HPow.hPow b (Nat.clog b x).pred) x
  -/
  rw [← not_le, le_pow_iff_clog_le hb, not_le]
  /-
    b : Nat
    hb : LT.lt 1 b
    x : Nat
    hx : LT.lt 1 x
    ⊢ LT.lt (Nat.clog b x).pred (Nat.clog b x)
  -/
  exact pred_lt (clog_pos hb hx).ne'
  /-
    🎉 no goals
  -/


theorem le_pow_clog {b : ℕ} (hb : 1 < b) (x : ℕ) : x ≤ b ^ clog b x :=
  (le_pow_iff_clog_le hb).2 le_rfl


@[mono]
theorem clog_mono_right (b : ℕ) {n m : ℕ} (h : n ≤ m) : clog b n ≤ clog b m := by
  /-
    b n m : Nat
    h : LE.le n m
    ⊢ LE.le (Nat.clog b n) (Nat.clog b m)
  -/
  rcases le_or_lt b 1 with hb | hb
    /-
      case inl
      b n m : Nat
      h : LE.le n m
      hb : LE.le b 1
      ⊢ LE.le (Nat.clog b n) (Nat.clog b m)
    -/
  · rw [clog_of_left_le_one hb]
    /-
      case inl
      b n m : Nat
      h : LE.le n m
      hb : LE.le b 1
      ⊢ LE.le 0 (Nat.clog b m)
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/
    /-
      case inr
      b n m : Nat
      h : LE.le n m
      hb : LT.lt 1 b
      ⊢ LE.le (Nat.clog b n) (Nat.clog b m)
    -/
  · rw [← le_pow_iff_clog_le hb]
    /-
      case inr
      b n m : Nat
      h : LE.le n m
      hb : LT.lt 1 b
      ⊢ LE.le n (HPow.hPow b (Nat.clog b m))
    -/
    exact h.trans (le_pow_clog hb _)
    /-
      🎉 no goals
    -/


@[mono]
theorem clog_anti_left {b c n : ℕ} (hc : 1 < c) (hb : c ≤ b) : clog b n ≤ clog c n := by
  /-
    b c n : Nat
    hc : LT.lt 1 c
    hb : LE.le c b
    ⊢ LE.le (Nat.clog b n) (Nat.clog c n)
  -/
  rw [← le_pow_iff_clog_le (lt_of_lt_of_le hc hb)]
  calc
    n ≤ c ^ clog c n := le_pow_clog hc _
    _ ≤ b ^ clog c n := Nat.pow_le_pow_left hb _


theorem clog_monotone (b : ℕ) : Monotone (clog b) := fun _ _ => clog_mono_right _


theorem clog_antitone_left {n : ℕ} : AntitoneOn (fun b : ℕ => clog b n) (Set.Ioi 1) :=
  fun _ hc _ _ hb => clog_anti_left (Set.mem_Iio.1 hc) hb


theorem log_le_clog (b n : ℕ) : log b n ≤ clog b n := by
  /-
    b n : Nat
    ⊢ LE.le (Nat.log b n) (Nat.clog b n)
  -/
  obtain hb | hb := le_or_lt b 1
    /-
      case inl
      b n : Nat
      hb : LE.le b 1
      ⊢ LE.le (Nat.log b n) (Nat.clog b n)
    -/
  · rw [log_of_left_le_one hb]
    /-
      case inl
      b n : Nat
      hb : LE.le b 1
      ⊢ LE.le 0 (Nat.clog b n)
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/
  cases n with
  | zero =>
    rw [log_zero_right]
    exact zero_le _
  | succ n =>
    exact (Nat.pow_le_pow_iff_right hb).1
      ((pow_log_le_self b n.succ_ne_zero).trans <| le_pow_clog hb _)


